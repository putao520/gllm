//! Generator Model Weight Loader - Pure L3 API compatible.

use crate::generator_model::{GeneratorModel, LayerWeights};
use crate::parallel_parser::{
    is_shard_index, load_embedding, load_linear, parse_shards, take_cached_shards, LoadConfig,
    ShardedTensorLoader, TensorLoader,
};
use crate::types::{Error, Result};
use crate::weight_loader::mappings::Architecture;
use crate::weight_loader::shards::ShardIndex;
use crate::weight_loader::WeightLoader;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

impl GeneratorModel {
    pub fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        if is_shard_index(safetensors_path) {
            return self.load_sharded_safetensors(safetensors_path);
        }

        let file = File::open(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to open SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            Error::LoadError(format!(
                "Failed to map SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;
        let loader = WeightLoader::from_bytes(&mmap)?;
        self.load_from_loader(&loader)
    }

    pub fn load_awq(&mut self, path: &Path) -> Result<()> {
        self.load_safetensors(path)
    }

    pub fn load_gguf(&mut self, _path: &Path) -> Result<()> {
        Err(Error::LoadError(
            "GGUF loading is not supported for generator models".into(),
        ))
    }

    fn load_sharded_safetensors(&mut self, index_path: &Path) -> Result<()> {
        let model_dir = index_path.parent().ok_or_else(|| {
            Error::LoadError("Shard index path is missing parent directory".into())
        })?;
        let index = ShardIndex::from_index_file(index_path)?;
        let parsed = match take_cached_shards(model_dir) {
            Some(parsed) => parsed,
            None => parse_shards(index.shard_paths(model_dir), &LoadConfig::default())?,
        };
        let loader = ShardedTensorLoader::new(&parsed, &index)?;
        self.load_from_loader(&loader)
    }

    fn load_from_loader<L: TensorLoader>(&mut self, loader: &L) -> Result<()> {
        let architecture = Architecture::from_model_type(self.config.model_type.as_deref().unwrap_or(""));

        // Load embedding weights
        let embed_names = architecture.embedding_weight_candidates();
        let expected_embed_len = self.embedding_mut().len();
        let mut embed_loaded = false;
        for name in embed_names {
            if loader.has_tensor(name) {
                let embedding = load_embedding(loader, name)?;
                if embedding.data.len() != expected_embed_len {
                    return Err(Error::LoadError(format!(
                        "Embedding weight {} has {} elements, expected {}",
                        name,
                        embedding.data.len(),
                        expected_embed_len
                    )));
                }
                self.embedding_mut().copy_from_slice(&embedding.data);
                embed_loaded = true;
                break;
            }
        }
        if !embed_loaded {
            return Err(Error::LoadError(
                "Embedding weights not found for generator model".into(),
            ));
        }

        // Load layer weights
        let num_layers = self.num_layers();
        for layer_idx in 0..num_layers {
            let layer = self.layer_mut(layer_idx).ok_or_else(|| {
                Error::LoadError(format!("Layer {} out of range", layer_idx))
            })?;
            let mut last_err = None;
            let mut layer_loaded = false;
            for prefix in architecture.layer_prefix_candidates(layer_idx) {
                match load_layer_weights(loader, &prefix, layer, architecture) {
                    Ok(()) => {
                        layer_loaded = true;
                        break;
                    }
                    Err(err) => {
                        if is_missing_layer_weights_error(&err) {
                            last_err = Some(err);
                            continue;
                        }
                        return Err(err);
                    }
                }
            }
            if !layer_loaded {
                return Err(last_err.unwrap_or_else(|| {
                    Error::LoadError(format!("Layer weights not found for layer {}", layer_idx))
                }));
            }
        }

        // Load final norm weights
        let final_norm_names = architecture.final_norm_weight_candidates();
        for name in final_norm_names {
            if loader.has_tensor(name) {
                let norm_tensor = loader.load_tensor(name)?;
                let norm_data = norm_tensor.into_weight_vector()?.data;
                let final_norm = self.final_norm_mut();
                if norm_data.len() != final_norm.len() {
                    return Err(Error::LoadError(format!(
                        "Final norm weight {} has {} elements, expected {}",
                        name,
                        norm_data.len(),
                        final_norm.len()
                    )));
                }
                final_norm.copy_from_slice(&norm_data);
                break;
            }
        }

        // Load LM head weights
        let lm_head_names = architecture.lm_head_weight_candidates();
        let expected_lm_head_len = self.lm_head_mut().len();
        let mut lm_head_loaded = false;
        for name in lm_head_names {
            if loader.has_tensor(name) {
                let lm_head = load_embedding(loader, name)?;
                if lm_head.data.len() != expected_lm_head_len {
                    return Err(Error::LoadError(format!(
                        "LM head weight {} has {} elements, expected {}",
                        name,
                        lm_head.data.len(),
                        expected_lm_head_len
                    )));
                }
                self.lm_head_mut().copy_from_slice(&lm_head.data);
                lm_head_loaded = true;
                break;
            }
        }
        if !lm_head_loaded {
            // Tie embeddings to LM head if not found
            let embedding = self.embedding_mut().to_vec();
            self.lm_head_mut().copy_from_slice(&embedding);
        }

        Ok(())
    }
}

fn is_missing_layer_weights_error(err: &Error) -> bool {
    matches!(err, Error::LoadError(msg) if msg.contains("weights not found for prefix"))
}

fn load_layer_weights<L: TensorLoader>(
    loader: &L,
    prefix: &str,
    layer: &mut LayerWeights,
    architecture: Architecture,
) -> Result<()> {
    if matches!(architecture, Architecture::GptOss) {
        load_gpt_oss_attention_weights(loader, prefix, layer)?;
        load_gpt_oss_ffn_weights(loader, prefix, layer)?;
    } else {
        // Load attention projection weights (Q, K, V, O)
        load_attention_weights(loader, prefix, layer)?;

        // Load FFN weights
        load_ffn_weights(loader, prefix, layer)?;
    }

    // Load layer norms
    if loader.has_tensor(&format!("{}.input_layernorm.weight", prefix)) {
        let norm_tensor = loader.load_tensor(&format!("{}.input_layernorm.weight", prefix))?;
        let norm_data = norm_tensor.into_weight_vector()?.data;
        if norm_data.len() != layer.input_norm.len() {
            return Err(Error::LoadError(format!(
                "Input norm weight has {} elements, expected {}",
                norm_data.len(),
                layer.input_norm.len()
            )));
        }
        layer.input_norm.copy_from_slice(&norm_data);
    }

    if loader.has_tensor(&format!("{}.post_attention_layernorm.weight", prefix)) {
        let norm_tensor = loader.load_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
        let norm_data = norm_tensor.into_weight_vector()?.data;
        if norm_data.len() != layer.post_attn_norm.len() {
            return Err(Error::LoadError(format!(
                "Post attention norm weight has {} elements, expected {}",
                norm_data.len(),
                layer.post_attn_norm.len()
            )));
        }
        layer.post_attn_norm.copy_from_slice(&norm_data);
    }

    Ok(())
}

fn load_gpt_oss_attention_weights<L: TensorLoader>(
    loader: &L,
    prefix: &str,
    layer: &mut LayerWeights,
) -> Result<()> {
    let attn_prefix = format!("{}.attn", prefix);
    let weight_name = format!("{}.c_attn.weight", attn_prefix);
    if !loader.has_tensor(&weight_name) {
        return Err(Error::LoadError(format!(
            "Attention weights not found for prefix {}",
            prefix
        )));
    }

    let hidden_size = layer.input_norm.len();
    let q_rows = projection_rows(&layer.q_weight, hidden_size, "q_proj")?;
    let k_rows = projection_rows(&layer.k_weight, hidden_size, "k_proj")?;
    let v_rows = projection_rows(&layer.v_weight, hidden_size, "v_proj")?;

    let c_attn = loader.load_tensor(&weight_name)?;
    if c_attn.shape.len() != 2 || c_attn.shape[1] != hidden_size {
        return Err(Error::LoadError(format!(
            "c_attn weight shape {:?} is incompatible with hidden_size {}",
            c_attn.shape, hidden_size
        )));
    }
    let qkv = c_attn.split_rows(&[q_rows, k_rows, v_rows])?;
    copy_slice_to_weights(qkv[0], &mut layer.q_weight, "q_proj")?;
    copy_slice_to_weights(qkv[1], &mut layer.k_weight, "k_proj")?;
    copy_slice_to_weights(qkv[2], &mut layer.v_weight, "v_proj")?;

    let bias_name = format!("{}.c_attn.bias", attn_prefix);
    if loader.has_tensor(&bias_name) {
        let c_attn_bias = loader.load_tensor(&bias_name)?;
        let _ = c_attn_bias.split_vector(&[q_rows, k_rows, v_rows])?;
    }

    let o_name = format!("{}.c_proj.weight", attn_prefix);
    let o_linear = load_linear(loader, &o_name, Some(&format!("{}.c_proj.bias", attn_prefix)))?;
    copy_linear_to_slice(&o_linear, &mut layer.o_weight)?;

    Ok(())
}

fn load_gpt_oss_ffn_weights<L: TensorLoader>(
    loader: &L,
    prefix: &str,
    layer: &mut LayerWeights,
) -> Result<()> {
    let ffn_prefix = format!("{}.mlp", prefix);
    let up_name = format!("{}.c_fc.weight", ffn_prefix);
    if !loader.has_tensor(&up_name) {
        return Err(Error::LoadError(format!(
            "FFN weights not found for prefix {}",
            prefix
        )));
    }

    let up_linear = load_linear(loader, &up_name, Some(&format!("{}.c_fc.bias", ffn_prefix)))?;
    copy_linear_to_slice(&up_linear, &mut layer.up_weight)?;

    let down_name = format!("{}.c_proj.weight", ffn_prefix);
    let down_linear = load_linear(
        loader,
        &down_name,
        Some(&format!("{}.c_proj.bias", ffn_prefix)),
    )?;
    copy_linear_to_slice(&down_linear, &mut layer.down_weight)?;

    layer.gate_weight = None;
    Ok(())
}

fn load_attention_weights<L: TensorLoader>(
    loader: &L,
    prefix: &str,
    layer: &mut LayerWeights,
) -> Result<()> {
    // Try different attention weight naming conventions
    let attn_prefixes = [
        format!("{}.self_attn", prefix),
        format!("{}.attention", prefix),
    ];

    for attn_prefix in attn_prefixes {
        let q_name = format!("{}.q_proj.weight", attn_prefix);
        if !loader.has_tensor(&q_name) {
            continue;
        }

        // Load Q projection
        let q_linear = load_linear(loader, &q_name, Some(&format!("{}.q_proj.bias", attn_prefix)))?;
        copy_linear_to_slice(&q_linear, &mut layer.q_weight)?;

        // Load K projection
        let k_name = format!("{}.k_proj.weight", attn_prefix);
        let k_linear = load_linear(loader, &k_name, Some(&format!("{}.k_proj.bias", attn_prefix)))?;
        copy_linear_to_slice(&k_linear, &mut layer.k_weight)?;

        // Load V projection
        let v_name = format!("{}.v_proj.weight", attn_prefix);
        let v_linear = load_linear(loader, &v_name, Some(&format!("{}.v_proj.bias", attn_prefix)))?;
        copy_linear_to_slice(&v_linear, &mut layer.v_weight)?;

        // Load O projection
        let o_name = format!("{}.o_proj.weight", attn_prefix);
        let o_linear = load_linear(loader, &o_name, Some(&format!("{}.o_proj.bias", attn_prefix)))?;
        copy_linear_to_slice(&o_linear, &mut layer.o_weight)?;

        return Ok(());
    }

    Err(Error::LoadError(format!(
        "Attention weights not found for prefix {}",
        prefix
    )))
}

fn load_ffn_weights<L: TensorLoader>(
    loader: &L,
    prefix: &str,
    layer: &mut LayerWeights,
) -> Result<()> {
    // Try different FFN weight naming conventions
    let ffn_prefixes = [
        format!("{}.mlp", prefix),
        format!("{}.feed_forward", prefix),
    ];

    for ffn_prefix in ffn_prefixes {
        // Try gate_proj/up_proj/down_proj naming (LLaMA style)
        let gate_name = format!("{}.gate_proj.weight", ffn_prefix);
        if loader.has_tensor(&gate_name) {
            // LLaMA-style with gate projection
            let gate_linear = load_linear(loader, &gate_name, None)?;
            if let Some(ref mut gate_weight) = layer.gate_weight {
                copy_linear_to_slice(&gate_linear, gate_weight)?;
            }

            let up_name = format!("{}.up_proj.weight", ffn_prefix);
            let up_linear = load_linear(loader, &up_name, None)?;
            copy_linear_to_slice(&up_linear, &mut layer.up_weight)?;

            let down_name = format!("{}.down_proj.weight", ffn_prefix);
            let down_linear = load_linear(loader, &down_name, None)?;
            copy_linear_to_slice(&down_linear, &mut layer.down_weight)?;

            return Ok(());
        }

        // Try w1/w2/w3 naming (some models)
        let w1_name = format!("{}.w1.weight", ffn_prefix);
        if loader.has_tensor(&w1_name) {
            let gate_linear = load_linear(loader, &w1_name, None)?;
            if let Some(ref mut gate_weight) = layer.gate_weight {
                copy_linear_to_slice(&gate_linear, gate_weight)?;
            }

            let w3_name = format!("{}.w3.weight", ffn_prefix);
            let up_linear = load_linear(loader, &w3_name, None)?;
            copy_linear_to_slice(&up_linear, &mut layer.up_weight)?;

            let w2_name = format!("{}.w2.weight", ffn_prefix);
            let down_linear = load_linear(loader, &w2_name, None)?;
            copy_linear_to_slice(&down_linear, &mut layer.down_weight)?;

            return Ok(());
        }

        // Try fc1/fc2 naming (GELU models without gate)
        let fc1_name = format!("{}.fc1.weight", ffn_prefix);
        if loader.has_tensor(&fc1_name) {
            let up_linear = load_linear(
                loader,
                &fc1_name,
                Some(&format!("{}.fc1.bias", ffn_prefix)),
            )?;
            copy_linear_to_slice(&up_linear, &mut layer.up_weight)?;

            let fc2_name = format!("{}.fc2.weight", ffn_prefix);
            let down_linear = load_linear(
                loader,
                &fc2_name,
                Some(&format!("{}.fc2.bias", ffn_prefix)),
            )?;
            copy_linear_to_slice(&down_linear, &mut layer.down_weight)?;

            // No gate projection for GELU models
            return Ok(());
        }
    }

    Err(Error::LoadError(format!(
        "FFN weights not found for prefix {}",
        prefix
    )))
}

/// Copy LinearWeights data to a flat slice.
fn copy_linear_to_slice(linear: &crate::weight_loader::LinearWeights, dest: &mut [f32]) -> Result<()> {
    let src = linear.as_dense_slice().ok_or_else(|| {
        Error::LoadError("Expected dense weights, got quantized".into())
    })?;
    if src.len() != dest.len() {
        return Err(Error::LoadError(format!(
            "Linear weight size mismatch: got {}, expected {}",
            src.len(),
            dest.len()
        )));
    }
    dest.copy_from_slice(src);
    Ok(())
}

fn projection_rows(weights: &[f32], hidden_size: usize, label: &str) -> Result<usize> {
    if hidden_size == 0 {
        return Err(Error::LoadError("Hidden size is zero".into()));
    }
    if weights.len() % hidden_size != 0 {
        return Err(Error::LoadError(format!(
            "{label} weight length {} is not divisible by hidden_size {}",
            weights.len(),
            hidden_size
        )));
    }
    Ok(weights.len() / hidden_size)
}

fn copy_slice_to_weights(src: &[f32], dest: &mut [f32], label: &str) -> Result<()> {
    if src.len() != dest.len() {
        return Err(Error::LoadError(format!(
            "{label} weight size mismatch: got {}, expected {}",
            src.len(),
            dest.len()
        )));
    }
    dest.copy_from_slice(src);
    Ok(())
}
