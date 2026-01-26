use crate::generator_model::GeneratorModel;
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
                "Embedding weights not found for generator model".into(),
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

            if loader.has_tensor(&format!("{}.mlp.gate_proj.weight", prefix)) {
                layer.ffn.gate_proj = load_linear(
                    loader,
                    &format!("{}.mlp.gate_proj.weight", prefix),
                    None,
                )?;
                layer.ffn.up_proj = load_linear(
                    loader,
                    &format!("{}.mlp.up_proj.weight", prefix),
                    None,
                )?;
                layer.ffn.down_proj = load_linear(
                    loader,
                    &format!("{}.mlp.down_proj.weight", prefix),
                    None,
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
