use crate::causal_attention::{CausalAttention, RotaryPositionEmbedding};
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::rms_norm::RmsNorm;
use crate::tensor::Tensor3;
use crate::types::{Error, Result};
use crate::weight_loader::LinearWeights;
use gllm_kernels::{linear_forward, silu_inplace};
use std::sync::Arc;

#[derive(Clone)]
pub struct DecoderLayer {
    pub(crate) attention_norm: RmsNorm,
    pub(crate) attention: CausalAttention,
    pub(crate) ffn_norm: RmsNorm,
    pub(crate) gate_proj: LinearWeights,
    pub(crate) up_proj: LinearWeights,
    pub(crate) down_proj: LinearWeights,
    hidden_size: usize,
}

impl DecoderLayer {
    pub fn new(config: &ModelConfig, rope: Option<Arc<RotaryPositionEmbedding>>) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate = config
            .intermediate_size
            .unwrap_or(hidden_size.saturating_mul(4));

        let attention_norm = RmsNorm::new(config);
        let attention = CausalAttention::new(config, rope, true)?;
        let ffn_norm = RmsNorm::new(config);
        let gate_proj = LinearWeights::zeros(intermediate, hidden_size);
        let up_proj = LinearWeights::zeros(intermediate, hidden_size);
        let down_proj = LinearWeights::zeros(hidden_size, intermediate);

        Ok(Self {
            attention_norm,
            attention,
            ffn_norm,
            gate_proj,
            up_proj,
            down_proj,
            hidden_size,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor3, position_offset: usize) -> Result<Tensor3> {
        let (batch, seq_len, _) = hidden_states.shape();
        let attn_input = self.attention_norm.forward_3d(&hidden_states.data, batch, seq_len);
        let attn_input = Tensor3::new(attn_input, batch, seq_len, self.hidden_size)?;
        let attn_output = self.attention.forward(&attn_input, position_offset)?;
        let residual = add_tensors(hidden_states, &attn_output)?;

        let ffn_input = self.ffn_norm.forward_3d(&residual.data, batch, seq_len);
        let ffn_input = Tensor3::new(ffn_input, batch, seq_len, self.hidden_size)?;
        let ffn_output = self.ffn_forward(&ffn_input)?;

        add_tensors(&residual, &ffn_output)
    }

    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor3,
        position_offset: usize,
        cache: &mut KVCache,
        layer: usize,
    ) -> Result<Tensor3> {
        let (batch, seq_len, _) = hidden_states.shape();
        let attn_input = self.attention_norm.forward_3d(&hidden_states.data, batch, seq_len);
        let attn_input = Tensor3::new(attn_input, batch, seq_len, self.hidden_size)?;
        let attn_output =
            self.attention
                .forward_with_cache(&attn_input, position_offset, cache, layer)?;
        let residual = add_tensors(hidden_states, &attn_output)?;

        let ffn_input = self.ffn_norm.forward_3d(&residual.data, batch, seq_len);
        let ffn_input = Tensor3::new(ffn_input, batch, seq_len, self.hidden_size)?;
        let ffn_output = self.ffn_forward(&ffn_input)?;

        add_tensors(&residual, &ffn_output)
    }

    fn ffn_forward(&self, input: &Tensor3) -> Result<Tensor3> {
        let (batch, seq_len, hidden) = input.shape();
        if hidden != self.hidden_size {
            return Err(Error::InferenceError(
                "FFN input hidden size mismatch".into(),
            ));
        }
        let rows = batch * seq_len;

        let mut gate = vec![0.0f32; rows * self.gate_proj.weight.rows];
        linear_forward(
            &input.data,
            self.gate_proj.weight.as_slice(),
            self.gate_proj.bias.as_ref().map(|b| b.as_slice()),
            &mut gate,
            rows,
            self.gate_proj.weight.cols,
            self.gate_proj.weight.rows,
        );

        let mut up = vec![0.0f32; rows * self.up_proj.weight.rows];
        linear_forward(
            &input.data,
            self.up_proj.weight.as_slice(),
            self.up_proj.bias.as_ref().map(|b| b.as_slice()),
            &mut up,
            rows,
            self.up_proj.weight.cols,
            self.up_proj.weight.rows,
        );

        if gate.len() != up.len() {
            return Err(Error::InferenceError(
                "FFN gate/up projection size mismatch".into(),
            ));
        }

        silu_inplace(&mut gate);
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g *= u;
        }

        let mut down = vec![0.0f32; rows * self.down_proj.weight.rows];
        linear_forward(
            &gate,
            self.down_proj.weight.as_slice(),
            self.down_proj.bias.as_ref().map(|b| b.as_slice()),
            &mut down,
            rows,
            self.down_proj.weight.cols,
            self.down_proj.weight.rows,
        );

        Tensor3::new(down, batch, seq_len, self.down_proj.weight.rows)
    }
}

fn add_tensors(lhs: &Tensor3, rhs: &Tensor3) -> Result<Tensor3> {
    if lhs.data.len() != rhs.data.len() {
        return Err(Error::InferenceError(
            "Tensor add length mismatch".into(),
        ));
    }
    let mut out = Vec::with_capacity(lhs.data.len());
    for (a, b) in lhs.data.iter().zip(rhs.data.iter()) {
        out.push(a + b);
    }
    Tensor3::new(out, lhs.dim0, lhs.dim1, lhs.dim2)
}
