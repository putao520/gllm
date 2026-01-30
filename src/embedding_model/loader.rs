//! Embedding Model Weight Loader - Pure L3 API compatible.

use crate::embedding_model::{EmbeddingModel, LayerWeights};
use crate::parallel_parser::{
    is_shard_index, load_embedding, load_linear, parse_shards, resolve_weight_name,
    take_cached_shards, LoadConfig, ShardedTensorLoader, TensorLoader,
};
use crate::types::{Error, Result};
use crate::weight_loader::mappings::Architecture;
use crate::weight_loader::shards::ShardIndex;
use crate::weight_loader::WeightLoader;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

impl EmbeddingModel {
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

        // Load embedding weights - try multiple naming conventions
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
                "Embedding weights not found for embedding model".into(),
            ));
        }

        // Load position embeddings (BERT-style)
        let position_embed_names = architecture.position_embedding_weight_candidates();
        let position_embed = self.position_embedding_mut();
        for name in position_embed_names {
            if loader.has_tensor(name) {
                let pos_embed = load_embedding(loader, name)?;
                if pos_embed.data.len() == position_embed.len() {
                    position_embed.copy_from_slice(&pos_embed.data);
                    break;
                }
            }
        }

        // Load token type embeddings (BERT-style)
        let token_type_names = architecture.token_type_embedding_weight_candidates();
        let token_type_embed = self.token_type_embedding_mut();
        for name in token_type_names {
            if loader.has_tensor(name) {
                let tt_embed = load_embedding(loader, name)?;
                if tt_embed.data.len() == token_type_embed.len() {
                    token_type_embed.copy_from_slice(&tt_embed.data);
                    break;
                }
            }
        }

        // Load layer weights
        let num_layers = self.num_layers();
        for layer_idx in 0..num_layers {
            let layer = self.layer_mut(layer_idx).ok_or_else(|| {
                Error::LoadError(format!("Layer {} out of range", layer_idx))
            })?;

            let mut loaded = false;
            for prefix in architecture.layer_prefix_candidates(layer_idx) {
                match try_load_layer_weights(loader, &prefix, layer) {
                    Ok(_) => {
                        loaded = true;
                        break;
                    }
                    Err(_) => {
                        // Try next prefix
                    }
                }
            }
            if !loaded {
                return Err(Error::LoadError(format!(
                    "Layer {} weights not found",
                    layer_idx
                )));
            }
        }

        // Load final norm weights (used by decoder-style models)
        let final_norm_names = architecture.final_norm_weight_candidates();
        for name in final_norm_names {
            if loader.has_tensor(name) {
                let norm_tensor = loader.load_tensor(name)?;
                let norm_data = norm_tensor.into_weight_vector()?.data;
                let final_norm = self.final_norm_mut();
                if norm_data.len() == final_norm.len() {
                    final_norm.copy_from_slice(&norm_data);
                    break;
                }
            }
        }

        // For BERT-style models, also load the embedding LayerNorm
        if self.is_bert_style() {
            let bert_norm_names = architecture.embedding_layer_norm_weight_candidates();
            let final_norm = self.final_norm_mut();
            for name in bert_norm_names {
                if loader.has_tensor(name) {
                    let norm_tensor = loader.load_tensor(name)?;
                    let norm_data = norm_tensor.into_weight_vector()?.data;
                    if norm_data.len() == final_norm.len() {
                        final_norm.copy_from_slice(&norm_data);
                        break;
                    }
                }
            }

            // Load embedding LayerNorm bias
            let bert_norm_bias_names = architecture.embedding_layer_norm_bias_candidates();
            let final_norm_bias = self.final_norm_bias_mut();
            for name in bert_norm_bias_names {
                if loader.has_tensor(name) {
                    let bias_tensor = loader.load_tensor(name)?;
                    let bias_data = bias_tensor.into_weight_vector()?.data;
                    if bias_data.len() == final_norm_bias.len() {
                        final_norm_bias.copy_from_slice(&bias_data);
                        break;
                    }
                }
            }
        }

        Ok(())
    }
}

fn try_load_layer_weights<L: TensorLoader>(
    loader: &L,
    prefix: &str,
    layer: &mut LayerWeights,
) -> Result<()> {
    // Load attention weights
    load_attention_weights(loader, prefix, layer)?;

    // Load FFN weights
    load_ffn_weights(loader, prefix, layer)?;

    // Load layer norms - try multiple naming conventions
    let input_norm_names = [
        format!("{}.input_layernorm.weight", prefix),
        format!("{}.attention.output.LayerNorm.weight", prefix),
        format!("{}.attention.LayerNorm.weight", prefix),
        format!("{}.ln_1.weight", prefix),
        format!("{}.norm1.weight", prefix),
    ];
    for name in &input_norm_names {
        if loader.has_tensor(name) {
            let norm_tensor = loader.load_tensor(name)?;
            let norm_data = norm_tensor.into_weight_vector()?.data;
            if norm_data.len() == layer.input_norm.len() {
                layer.input_norm.copy_from_slice(&norm_data);
                break;
            }
        }
    }

    // Load input norm bias (BERT-style)
    let input_norm_bias_names = [
        format!("{}.attention.output.LayerNorm.bias", prefix),
        format!("{}.attention.LayerNorm.bias", prefix),
        format!("{}.ln_1.bias", prefix),
        format!("{}.norm1.bias", prefix),
    ];
    for name in &input_norm_bias_names {
        if loader.has_tensor(name) {
            let bias_tensor = loader.load_tensor(name)?;
            let bias_data = bias_tensor.into_weight_vector()?.data;
            if bias_data.len() == layer.input_norm_bias.len() {
                layer.input_norm_bias.copy_from_slice(&bias_data);
                break;
            }
        }
    }

    let post_attn_norm_names = [
        format!("{}.post_attention_layernorm.weight", prefix),
        format!("{}.output.LayerNorm.weight", prefix),
        format!("{}.ln_2.weight", prefix),
        format!("{}.norm2.weight", prefix),
    ];
    for name in &post_attn_norm_names {
        if loader.has_tensor(name) {
            let norm_tensor = loader.load_tensor(name)?;
            let norm_data = norm_tensor.into_weight_vector()?.data;
            if norm_data.len() == layer.post_attn_norm.len() {
                layer.post_attn_norm.copy_from_slice(&norm_data);
                break;
            }
        }
    }

    // Load post-attention norm bias (BERT-style)
    let post_attn_norm_bias_names = [
        format!("{}.output.LayerNorm.bias", prefix),
        format!("{}.ln_2.bias", prefix),
        format!("{}.norm2.bias", prefix),
    ];
    for name in &post_attn_norm_bias_names {
        if loader.has_tensor(name) {
            let bias_tensor = loader.load_tensor(name)?;
            let bias_data = bias_tensor.into_weight_vector()?.data;
            if bias_data.len() == layer.post_attn_norm_bias.len() {
                layer.post_attn_norm_bias.copy_from_slice(&bias_data);
                break;
            }
        }
    }

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
        format!("{}.attention.self", prefix),
        format!("{}.attention", prefix),
        format!("{}.attn", prefix),
    ];

    for attn_prefix in attn_prefixes {
        // Try q_proj style
        let q_name = format!("{}.q_proj.weight", attn_prefix);
        if loader.has_tensor(&q_name) {
            let q_linear = load_linear(loader, &q_name, Some(&format!("{}.q_proj.bias", attn_prefix)))?;
            copy_linear_to_slice(&q_linear, &mut layer.q_weight)?;
            copy_bias_to_slice(&q_linear, &mut layer.q_bias);

            let k_name = format!("{}.k_proj.weight", attn_prefix);
            let k_linear = load_linear(loader, &k_name, Some(&format!("{}.k_proj.bias", attn_prefix)))?;
            copy_linear_to_slice(&k_linear, &mut layer.k_weight)?;
            copy_bias_to_slice(&k_linear, &mut layer.k_bias);

            let v_name = format!("{}.v_proj.weight", attn_prefix);
            let v_linear = load_linear(loader, &v_name, Some(&format!("{}.v_proj.bias", attn_prefix)))?;
            copy_linear_to_slice(&v_linear, &mut layer.v_weight)?;
            copy_bias_to_slice(&v_linear, &mut layer.v_bias);

            let o_name = format!("{}.o_proj.weight", attn_prefix);
            let o_linear = load_linear(loader, &o_name, Some(&format!("{}.o_proj.bias", attn_prefix)))?;
            copy_linear_to_slice(&o_linear, &mut layer.o_weight)?;
            copy_bias_to_slice(&o_linear, &mut layer.o_bias);

            return Ok(());
        }

        // Try BERT-style query/key/value
        let query_name = format!("{}.query.weight", attn_prefix);
        if loader.has_tensor(&query_name) {
            let q_linear = load_linear(loader, &query_name, Some(&format!("{}.query.bias", attn_prefix)))?;
            copy_linear_to_slice(&q_linear, &mut layer.q_weight)?;
            copy_bias_to_slice(&q_linear, &mut layer.q_bias);

            let k_linear = load_linear(
                loader,
                &format!("{}.key.weight", attn_prefix),
                Some(&format!("{}.key.bias", attn_prefix)),
            )?;
            copy_linear_to_slice(&k_linear, &mut layer.k_weight)?;
            copy_bias_to_slice(&k_linear, &mut layer.k_bias);

            let v_linear = load_linear(
                loader,
                &format!("{}.value.weight", attn_prefix),
                Some(&format!("{}.value.bias", attn_prefix)),
            )?;
            copy_linear_to_slice(&v_linear, &mut layer.v_weight)?;
            copy_bias_to_slice(&v_linear, &mut layer.v_bias);

            // BERT output projection
            let o_linear = load_linear(
                loader,
                &format!("{}.output.dense.weight", attn_prefix.replace(".self", "")),
                Some(&format!("{}.output.dense.bias", attn_prefix.replace(".self", ""))),
            )?;
            copy_linear_to_slice(&o_linear, &mut layer.o_weight)?;
            copy_bias_to_slice(&o_linear, &mut layer.o_bias);

            return Ok(());
        }
    }

    // Try JinaBERT-style fused QKV: mixer.Wqkv + mixer.out_proj
    let wqkv_base = format!("{}.mixer.Wqkv.weight", prefix);
    if let Some(wqkv_name) = resolve_weight_name(loader, &wqkv_base) {
        let hidden_size = layer.input_norm.len();
        let q_rows = projection_rows(&layer.q_weight, hidden_size, "q_proj")?;
        let k_rows = projection_rows(&layer.k_weight, hidden_size, "k_proj")?;
        let v_rows = projection_rows(&layer.v_weight, hidden_size, "v_proj")?;

        let qkv = loader.load_tensor(&wqkv_name)?;
        if qkv.shape.len() != 2 || qkv.shape[1] != hidden_size {
            return Err(Error::LoadError(format!(
                "Wqkv weight shape {:?} is incompatible with hidden_size {}",
                qkv.shape, hidden_size
            )));
        }
        let chunks = qkv.split_rows(&[q_rows, k_rows, v_rows])?;
        copy_slice_to_weights(chunks[0], &mut layer.q_weight, "q_proj")?;
        copy_slice_to_weights(chunks[1], &mut layer.k_weight, "k_proj")?;
        copy_slice_to_weights(chunks[2], &mut layer.v_weight, "v_proj")?;

        let bias_name = format!("{}.mixer.Wqkv.bias", prefix);
        if loader.has_tensor(&bias_name) {
            let bias = loader.load_tensor(&bias_name)?;
            let bias_chunks = bias.split_vector(&[q_rows, k_rows, v_rows])?;
            copy_slice_to_weights(bias_chunks[0], &mut layer.q_bias, "q_bias")?;
            copy_slice_to_weights(bias_chunks[1], &mut layer.k_bias, "k_bias")?;
            copy_slice_to_weights(bias_chunks[2], &mut layer.v_bias, "v_bias")?;
        }

        let out_linear = load_linear(
            loader,
            &format!("{}.mixer.out_proj.weight", prefix),
            Some(&format!("{}.mixer.out_proj.bias", prefix)),
        )?;
        copy_linear_to_slice(&out_linear, &mut layer.o_weight)?;
        copy_bias_to_slice(&out_linear, &mut layer.o_bias);

        return Ok(());
    }

    // Try NomicBERT-style fused QKV: attn.Wqkv + attn.out_proj
    let nomic_wqkv_base = format!("{}.attn.Wqkv.weight", prefix);
    if let Some(wqkv_name) = resolve_weight_name(loader, &nomic_wqkv_base) {
        let hidden_size = layer.input_norm.len();
        let q_rows = projection_rows(&layer.q_weight, hidden_size, "q_proj")?;
        let k_rows = projection_rows(&layer.k_weight, hidden_size, "k_proj")?;
        let v_rows = projection_rows(&layer.v_weight, hidden_size, "v_proj")?;

        let qkv = loader.load_tensor(&wqkv_name)?;
        if qkv.shape.len() != 2 || qkv.shape[1] != hidden_size {
            return Err(Error::LoadError(format!(
                "Wqkv weight shape {:?} is incompatible with hidden_size {}",
                qkv.shape, hidden_size
            )));
        }
        let chunks = qkv.split_rows(&[q_rows, k_rows, v_rows])?;
        copy_slice_to_weights(chunks[0], &mut layer.q_weight, "q_proj")?;
        copy_slice_to_weights(chunks[1], &mut layer.k_weight, "k_proj")?;
        copy_slice_to_weights(chunks[2], &mut layer.v_weight, "v_proj")?;

        let out_linear = load_linear(
            loader,
            &format!("{}.attn.out_proj.weight", prefix),
            Some(&format!("{}.attn.out_proj.bias", prefix)),
        )?;
        copy_linear_to_slice(&out_linear, &mut layer.o_weight)?;
        copy_bias_to_slice(&out_linear, &mut layer.o_bias);

        return Ok(());
    }

    // Try MPNet-style: attention.attn.q/k/v/o
    let mpnet_q_name = format!("{}.attention.attn.q.weight", prefix);
    if loader.has_tensor(&mpnet_q_name) {
        let q_linear = load_linear(
            loader,
            &mpnet_q_name,
            Some(&format!("{}.attention.attn.q.bias", prefix)),
        )?;
        copy_linear_to_slice(&q_linear, &mut layer.q_weight)?;
        copy_bias_to_slice(&q_linear, &mut layer.q_bias);

        let k_linear = load_linear(
            loader,
            &format!("{}.attention.attn.k.weight", prefix),
            Some(&format!("{}.attention.attn.k.bias", prefix)),
        )?;
        copy_linear_to_slice(&k_linear, &mut layer.k_weight)?;
        copy_bias_to_slice(&k_linear, &mut layer.k_bias);

        let v_linear = load_linear(
            loader,
            &format!("{}.attention.attn.v.weight", prefix),
            Some(&format!("{}.attention.attn.v.bias", prefix)),
        )?;
        copy_linear_to_slice(&v_linear, &mut layer.v_weight)?;
        copy_bias_to_slice(&v_linear, &mut layer.v_bias);

        let o_linear = load_linear(
            loader,
            &format!("{}.attention.attn.o.weight", prefix),
            Some(&format!("{}.attention.attn.o.bias", prefix)),
        )?;
        copy_linear_to_slice(&o_linear, &mut layer.o_weight)?;
        copy_bias_to_slice(&o_linear, &mut layer.o_bias);

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
        format!("{}.intermediate", prefix),
        format!("{}.output", prefix),
    ];

    for ffn_prefix in &ffn_prefixes {
        // LLaMA-style: gate_proj/up_proj/down_proj
        let gate_name = format!("{}.gate_proj.weight", ffn_prefix);
        if loader.has_tensor(&gate_name) {
            let gate_linear = load_linear(loader, &gate_name, None)?;
            if let Some(ref mut gate_weight) = layer.gate_weight {
                copy_linear_to_slice(&gate_linear, gate_weight)?;
            }

            let up_linear = load_linear(loader, &format!("{}.up_proj.weight", ffn_prefix), None)?;
            copy_linear_to_slice(&up_linear, &mut layer.up_weight)?;

            let down_linear = load_linear(loader, &format!("{}.down_proj.weight", ffn_prefix), None)?;
            copy_linear_to_slice(&down_linear, &mut layer.down_weight)?;

            return Ok(());
        }

        // w1/w2/w3 naming
        let w1_name = format!("{}.w1.weight", ffn_prefix);
        if loader.has_tensor(&w1_name) {
            let gate_linear = load_linear(loader, &w1_name, None)?;
            if let Some(ref mut gate_weight) = layer.gate_weight {
                copy_linear_to_slice(&gate_linear, gate_weight)?;
            }

            let up_linear = load_linear(loader, &format!("{}.w3.weight", ffn_prefix), None)?;
            copy_linear_to_slice(&up_linear, &mut layer.up_weight)?;

            let down_linear = load_linear(loader, &format!("{}.w2.weight", ffn_prefix), None)?;
            copy_linear_to_slice(&down_linear, &mut layer.down_weight)?;

            return Ok(());
        }

        // NomicBERT-style: fc11(gate) / fc12(up) / fc2(down)
        let fc11_name = format!("{}.fc11.weight", ffn_prefix);
        if loader.has_tensor(&fc11_name) {
            let gate_linear = load_linear(loader, &fc11_name, None)?;
            if let Some(ref mut gate_weight) = layer.gate_weight {
                copy_linear_to_slice(&gate_linear, gate_weight)?;
            }

            let up_linear = load_linear(loader, &format!("{}.fc12.weight", ffn_prefix), None)?;
            copy_linear_to_slice(&up_linear, &mut layer.up_weight)?;

            let down_linear = load_linear(loader, &format!("{}.fc2.weight", ffn_prefix), None)?;
            copy_linear_to_slice(&down_linear, &mut layer.down_weight)?;

            return Ok(());
        }
    }

    // BERT-style: intermediate.dense + output.dense
    let intermediate_name = format!("{}.intermediate.dense.weight", prefix);
    if loader.has_tensor(&intermediate_name) {
        let up_linear = load_linear(
            loader,
            &intermediate_name,
            Some(&format!("{}.intermediate.dense.bias", prefix)),
        )?;
        copy_linear_to_slice(&up_linear, &mut layer.up_weight)?;
        copy_bias_to_slice(&up_linear, &mut layer.up_bias);

        let down_linear = load_linear(
            loader,
            &format!("{}.output.dense.weight", prefix),
            Some(&format!("{}.output.dense.bias", prefix)),
        )?;
        copy_linear_to_slice(&down_linear, &mut layer.down_weight)?;
        copy_bias_to_slice(&down_linear, &mut layer.down_bias);

        return Ok(());
    }

    // fc1/fc2 naming
    for ffn_prefix in &ffn_prefixes {
        let fc1_name = format!("{}.fc1.weight", ffn_prefix);
        if loader.has_tensor(&fc1_name) {
            let up_linear = load_linear(
                loader,
                &fc1_name,
                Some(&format!("{}.fc1.bias", ffn_prefix)),
            )?;
            copy_linear_to_slice(&up_linear, &mut layer.up_weight)?;

            let down_linear = load_linear(
                loader,
                &format!("{}.fc2.weight", ffn_prefix),
                Some(&format!("{}.fc2.bias", ffn_prefix)),
            )?;
            copy_linear_to_slice(&down_linear, &mut layer.down_weight)?;

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

/// Copy LinearWeights bias to a flat slice (for BERT-style models).
fn copy_bias_to_slice(linear: &crate::weight_loader::LinearWeights, dest: &mut [f32]) {
    if let Some(src) = linear.bias_slice() {
        if src.len() == dest.len() {
            dest.copy_from_slice(src);
        }
    }
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
