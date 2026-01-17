use crate::causal_attention::CausalAttention;
use crate::decoder_layer::DecoderLayer;
use crate::model_config::ModelConfig;
use crate::rms_norm::RmsNorm;
use crate::tensor::{Matrix, Tensor3};
use crate::types::{Error, Result};
use crate::weight_loader::{load_embedding, load_linear, WeightLoader};
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
}

impl DecoderModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
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

        let embeddings = WeightMatrix::zeros(config.vocab_size, config.hidden_size);
        let head_dim = config
            .head_dim
            .unwrap_or_else(|| config.hidden_size / config.num_attention_heads);
        let rope = CausalAttention::build_rope(&config, head_dim);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(&config, rope.clone())?);
        }

        Ok(Self {
            embeddings,
            layers,
            final_norm: RmsNorm::new(&config),
            pad_token_id: config.pad_token_id.unwrap_or(0),
            max_position_embeddings: config.max_position_embeddings,
            hidden_size: config.hidden_size,
        })
    }

    pub fn forward(&self, tokens: &[Vec<i64>]) -> Result<Tensor3> {
        let (batch, seq_len) = self.sequence_shape(tokens)?;
        let mut hidden_states = self.embed_tokens(tokens, batch, seq_len)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, 0)?;
        }

        let normed = self.final_norm.forward_3d(&hidden_states.data, batch, seq_len);
        Tensor3::new(normed, batch, seq_len, self.hidden_size)
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

        let embed_names = [
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
        ];
        for name in embed_names {
            if loader.has_tensor(name) {
                self.embeddings = load_embedding(&loader, name)?;
                break;
            }
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", layer_idx);
            if loader.has_tensor(&format!("{}.self_attn.q_proj.weight", prefix)) {
                layer.attention.q_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    Some(&format!("{}.self_attn.q_proj.bias", prefix)),
                )?;
                layer.attention.k_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    Some(&format!("{}.self_attn.k_proj.bias", prefix)),
                )?;
                layer.attention.v_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    Some(&format!("{}.self_attn.v_proj.bias", prefix)),
                )?;
                layer.attention.o_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    Some(&format!("{}.self_attn.o_proj.bias", prefix)),
                )?;
            }

            if loader.has_tensor(&format!("{}.mlp.gate_proj.weight", prefix)) {
                layer.gate_proj = load_linear(
                    &loader,
                    &format!("{}.mlp.gate_proj.weight", prefix),
                    None,
                )?;
                layer.up_proj = load_linear(
                    &loader,
                    &format!("{}.mlp.up_proj.weight", prefix),
                    None,
                )?;
                layer.down_proj = load_linear(
                    &loader,
                    &format!("{}.mlp.down_proj.weight", prefix),
                    None,
                )?;
            }

            if loader.has_tensor(&format!("{}.input_layernorm.weight", prefix)) {
                let norm_tensor = loader.load_tensor(&format!("{}.input_layernorm.weight", prefix))?;
                layer.attention_norm.gamma = norm_tensor.to_weight_vector()?;
            }
            if loader.has_tensor(&format!("{}.post_attention_layernorm.weight", prefix)) {
                let norm_tensor =
                    loader.load_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
                layer.ffn_norm.gamma = norm_tensor.to_weight_vector()?;
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
                self.final_norm.gamma = norm_tensor.to_weight_vector()?;
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
