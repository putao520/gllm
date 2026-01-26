use crate::moe_generator_model::MoEGeneratorModel;
use crate::parallel_parser::{
    is_shard_index, load_embedding, load_linear, parse_shards, take_cached_shards, LoadConfig,
    ShardedTensorLoader, TensorLoader,
};
use crate::types::{Error, Result};
use crate::weight_loader::shards::ShardIndex;
use crate::weight_loader::WeightLoader;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

impl MoEGeneratorModel {
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
            "GGUF loading is not supported for MoE generator models".into(),
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
        let embed_names = [
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
        ];
        let expected_embed_len = self.embedding.len();
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
                self.embedding = embedding.data;
                embed_loaded = true;
                break;
            }
        }
        if !embed_loaded {
            return Err(Error::LoadError(
                "Embedding weights not found for MoE generator model".into(),
            ));
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", layer_idx);
            if loader.has_tensor(&format!("{}.self_attn.q_proj.weight", prefix)) {
                layer.attention.q_proj = load_linear(
                    loader,
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    Some(&format!("{}.self_attn.q_proj.bias", prefix)),
                )?;
                layer.attention.k_proj = load_linear(
                    loader,
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    Some(&format!("{}.self_attn.k_proj.bias", prefix)),
                )?;
                layer.attention.v_proj = load_linear(
                    loader,
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    Some(&format!("{}.self_attn.v_proj.bias", prefix)),
                )?;
                layer.attention.o_proj = load_linear(
                    loader,
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    Some(&format!("{}.self_attn.o_proj.bias", prefix)),
                )?;
            }

            if loader.has_tensor(&format!("{}.input_layernorm.weight", prefix)) {
                let norm_tensor = loader.load_tensor(&format!("{}.input_layernorm.weight", prefix))?;
                layer.input_norm.weight = norm_tensor.into_weight_vector()?.data;
            }
            if loader.has_tensor(&format!("{}.post_attention_layernorm.weight", prefix)) {
                let norm_tensor =
                    loader.load_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
                layer.post_attn_norm.weight = norm_tensor.into_weight_vector()?.data;
            }

            load_moe_weights(loader, &prefix, layer_idx, &mut layer.moe)?;
        }

        let final_norm_names = [
            "model.norm.weight",
            "transformer.ln_f.weight",
            "transformer.encoder.final_layernorm.weight",
        ];
        for name in final_norm_names {
            if loader.has_tensor(name) {
                let norm_tensor = loader.load_tensor(name)?;
                self.final_norm.weight = norm_tensor.into_weight_vector()?.data;
                break;
            }
        }

        let lm_head_names = [
            "lm_head.weight",
            "model.lm_head.weight",
            "transformer.lm_head.weight",
        ];
        let expected_lm_head_len = self.lm_head.len();
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
                self.lm_head = lm_head.data;
                lm_head_loaded = true;
                break;
            }
        }
        if !lm_head_loaded {
            self.lm_head.clone_from(&self.embedding);
        }

        Ok(())
    }
}

fn load_moe_weights<L: TensorLoader>(
    loader: &L,
    layer_prefix: &str,
    layer_idx: usize,
    moe: &mut crate::moe_layer::MoELayer,
) -> Result<()> {
    let gate_candidates = [
        format!("{layer_prefix}.block_sparse_moe.gate.weight"),
        format!("{layer_prefix}.moe.gate.weight"),
        format!("{layer_prefix}.router.weight"),
        format!("{layer_prefix}.gate.weight"),
    ];
    let expected_gate_len = moe.num_experts * moe.hidden_size;
    let mut gate_loaded = false;
    for name in gate_candidates {
        if loader.has_tensor(&name) {
            let gate_weights = load_embedding(loader, &name)?;
            if gate_weights.data.len() != expected_gate_len {
                return Err(Error::LoadError(format!(
                    "MoE gate weight {} has {} elements, expected {}",
                    name,
                    gate_weights.data.len(),
                    expected_gate_len
                )));
            }
            moe.gate_weights = gate_weights.data;
            gate_loaded = true;
            break;
        }
    }
    if !gate_loaded {
        return Err(Error::LoadError(format!(
            "MoE gate weights not found for layer {layer_idx}"
        )));
    }

    for expert_idx in 0..moe.num_experts {
        let expert = moe.expert_mut(expert_idx).ok_or_else(|| {
            Error::LoadError(format!(
                "MoE expert index {expert_idx} out of range for layer {layer_idx}"
            ))
        })?;
        load_expert_weights(loader, layer_prefix, layer_idx, expert_idx, expert)?;
    }

    Ok(())
}

fn load_expert_weights<L: TensorLoader>(
    loader: &L,
    layer_prefix: &str,
    layer_idx: usize,
    expert_idx: usize,
    expert: &mut crate::moe_layer::ExpertWeights,
) -> Result<()> {
    let base_candidates = [
        format!("{layer_prefix}.block_sparse_moe.experts.{expert_idx}"),
        format!("{layer_prefix}.moe.experts.{expert_idx}"),
        format!("{layer_prefix}.experts.{expert_idx}"),
    ];
    for base in base_candidates {
        if try_load_w1_w2_w3(loader, &base, expert)? {
            return Ok(());
        }
        if try_load_proj_triple(loader, &base, expert)? {
            return Ok(());
        }
    }

    Err(Error::LoadError(format!(
        "MoE expert weights not found for layer {layer_idx} expert {expert_idx}"
    )))
}

fn try_load_w1_w2_w3<L: TensorLoader>(
    loader: &L,
    base: &str,
    expert: &mut crate::moe_layer::ExpertWeights,
) -> Result<bool> {
    let gate_name = format!("{base}.w1.weight");
    let up_name = format!("{base}.w3.weight");
    let down_name = format!("{base}.w2.weight");
    if !loader.has_tensor(&gate_name) {
        return Ok(false);
    }
    if !loader.has_tensor(&up_name) || !loader.has_tensor(&down_name) {
        return Ok(false);
    }

    expert.gate_proj = load_linear(loader, &gate_name, None)?;
    expert.up_proj = load_linear(loader, &up_name, None)?;
    expert.down_proj = load_linear(loader, &down_name, None)?;
    Ok(true)
}

fn try_load_proj_triple<L: TensorLoader>(
    loader: &L,
    base: &str,
    expert: &mut crate::moe_layer::ExpertWeights,
) -> Result<bool> {
    let gate_name = format!("{base}.gate_proj.weight");
    let up_name = format!("{base}.up_proj.weight");
    let down_name = format!("{base}.down_proj.weight");
    if !loader.has_tensor(&gate_name) {
        return Ok(false);
    }
    if !loader.has_tensor(&up_name) || !loader.has_tensor(&down_name) {
        return Ok(false);
    }

    expert.gate_proj = load_linear(
        loader,
        &gate_name,
        Some(&format!("{base}.gate_proj.bias")),
    )?;
    expert.up_proj = load_linear(
        loader,
        &up_name,
        Some(&format!("{base}.up_proj.bias")),
    )?;
    expert.down_proj = load_linear(
        loader,
        &down_name,
        Some(&format!("{base}.down_proj.bias")),
    )?;
    Ok(true)
}
