//! Decoder Model for encoder-style inference (embeddings, reranking).
//!
//! This model processes entire sequences at once without persistent KV cache.

use crate::causal_attention::CausalAttention;
use gllm_kernels::backend::BackendImpl;
use crate::decoder_layer::{DecoderLayer, FFNWeights};
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::parallel_parser::{
    is_shard_index, load_embedding, load_linear, parse_shards, take_cached_shards, LoadConfig,
    ShardedTensorLoader, TensorLoader,
};
use crate::rms_norm::RmsNorm;
use crate::tensor::{Matrix, Tensor3};
use crate::types::{Error, Result};
use crate::weight_loader::shards::ShardIndex;
use crate::weight_loader::WeightLoader;
use gllm_kernels::WeightMatrix;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

#[derive(Clone)]
pub struct DecoderModel {
    pub(crate) embeddings: WeightMatrix,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) final_norm: RmsNorm,
    pub(crate) pad_token_id: i64,
    pub(crate) max_position_embeddings: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
}

impl DecoderModel {
    pub fn new(config: ModelConfig, backend: BackendImpl) -> Result<Self> {
        if config.num_hidden_layers == 0 {
            return Err(Error::InvalidConfig(
                "num_hidden_layers must be greater than 0 for decoder model".into(),
            ));
        }
        if config.vocab_size == 0 {
            return Err(Error::InvalidConfig(
                "vocab_size must be greater than 0 for decoder model".into(),
            ));
        }

        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);
        let head_dim = config
            .head_dim
            .unwrap_or_else(|| hidden_size / config.num_attention_heads);
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(config.num_attention_heads);
        let eps = config.rms_norm_eps.unwrap_or(1e-6) as f32;

        let embeddings = WeightMatrix::zeros(config.vocab_size, hidden_size);
        let rope = CausalAttention::build_rope(&config, head_dim);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let attention = CausalAttention::new(&config, rope.clone(), false, backend.clone())?;
            let layer = DecoderLayer {
                attention,
                input_norm: RmsNorm::new(hidden_size, eps),
                post_attn_norm: RmsNorm::new(hidden_size, eps),
                ffn: FFNWeights::zeros(hidden_size, intermediate_size),
                hidden_size,
                intermediate_size,
                use_gelu: config.hidden_act.as_deref() == Some("gelu"),
                backend: backend.clone(),
            };
            layers.push(layer);
        }

        Ok(Self {
            embeddings,
            layers,
            final_norm: RmsNorm::new(hidden_size, eps),
            pad_token_id: config.pad_token_id.unwrap_or(0),
            max_position_embeddings: config.max_position_embeddings,
            hidden_size,
            num_key_value_heads,
            head_dim,
        })
    }

    pub fn forward(&self, tokens: &[Vec<i64>]) -> Result<Tensor3> {
        let (batch, seq_len) = self.sequence_shape(tokens)?;
        let mut hidden_states = self.embed_tokens(tokens, batch, seq_len)?;

        // Create a temporary KV cache for this forward pass
        let mut cache = KVCache::new(
            self.layers.len(),
            self.num_key_value_heads,
            self.head_dim,
            seq_len,
        );

        // Process each batch item separately
        for b in 0..batch {
            let start = b * seq_len * self.hidden_size;
            let end = start + seq_len * self.hidden_size;
            let mut batch_hidden = hidden_states.data[start..end].to_vec();

            // Reset cache for each batch item
            cache.reset();

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                batch_hidden = layer.forward(&batch_hidden, 0, &mut cache, layer_idx)?;
            }

            hidden_states.data[start..end].copy_from_slice(&batch_hidden);
        }

        // Final norm
        self.final_norm.forward_inplace(&mut hidden_states.data);
        Ok(hidden_states)
    }

    pub fn pool_hidden_states(&self, hidden_states: &Tensor3, tokens: &[Vec<i64>]) -> Matrix {
        let batch = tokens.len();
        let mut output = Matrix::zeros(batch, self.hidden_size);
        for (b, ids) in tokens.iter().enumerate() {
            let index = last_token_index(ids, self.pad_token_id, hidden_states.dim1);
            let start = (b * hidden_states.dim1 + index) * self.hidden_size;
            let end = start + self.hidden_size;
            output.data[b * self.hidden_size..(b + 1) * self.hidden_size]
                .copy_from_slice(&hidden_states.data[start..end]);
        }
        output
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

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
        let embed_names = [
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
        ];
        for name in embed_names {
            if loader.has_tensor(name) {
                self.embeddings = load_embedding(loader, name)?;
                break;
            }
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

        Ok(())
    }

    fn sequence_shape(&self, tokens: &[Vec<i64>]) -> Result<(usize, usize)> {
        if tokens.is_empty() {
            return Err(Error::InvalidConfig(
                "At least one input is required".into(),
            ));
        }
        let batch = tokens.len();
        let seq_len = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        if seq_len == 0 {
            return Err(Error::InvalidConfig(
                "input sequence length must be greater than 0".into(),
            ));
        }
        if self.max_position_embeddings > 0 && seq_len > self.max_position_embeddings {
            return Err(Error::InvalidConfig(format!(
                "Sequence length {} exceeds configured maximum {}",
                seq_len, self.max_position_embeddings
            )));
        }
        Ok((batch, seq_len))
    }

    fn embed_tokens(
        &self,
        tokens: &[Vec<i64>],
        batch: usize,
        seq_len: usize,
    ) -> Result<Tensor3> {
        let hidden = self.embeddings.cols;
        let mut data = vec![0.0f32; batch * seq_len * hidden];
        for (b, ids) in tokens.iter().enumerate() {
            for s in 0..seq_len {
                let id = ids.get(s).copied().unwrap_or(self.pad_token_id);
                let idx = safe_token_index(id, self.embeddings.rows);
                let row = self.embeddings.row(idx);
                let start = (b * seq_len + s) * hidden;
                data[start..start + hidden].copy_from_slice(row);
            }
        }
        Tensor3::new(data, batch, seq_len, hidden)
    }
}

fn safe_token_index(id: i64, vocab: usize) -> usize {
    let fallback = if vocab == 0 { 0 } else { vocab - 1 };
    if id < 0 {
        return 0;
    }
    let idx = id as usize;
    if idx < vocab {
        idx
    } else {
        fallback
    }
}

fn last_token_index(ids: &[i64], pad_id: i64, max_len: usize) -> usize {
    if ids.is_empty() {
        return 0;
    }
    let mut idx = ids.len().saturating_sub(1);
    while idx > 0 && ids[idx] == pad_id {
        idx = idx.saturating_sub(1);
    }
    idx.min(max_len.saturating_sub(1))
}
