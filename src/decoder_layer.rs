use crate::causal_attention::CausalAttention;
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::rms_norm::RmsNorm;
use crate::types::Result;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Clone)]
pub struct DecoderLayer<B: Backend> {
    attention_norm: RmsNorm<B>,
    attention: CausalAttention<B>,
    ffn_norm: RmsNorm<B>,
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> DecoderLayer<B> {
    pub fn new(device: &B::Device, config: &ModelConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate = config.intermediate_size.unwrap_or(hidden_size.saturating_mul(4));

        let attention_norm = RmsNorm::new(device, config);
        let attention = CausalAttention::new(device, config)?;
        let ffn_norm = RmsNorm::new(device, config);
        let gate_proj = LinearConfig::new(hidden_size, intermediate).init(device);
        let up_proj = LinearConfig::new(hidden_size, intermediate).init(device);
        let down_proj = LinearConfig::new(intermediate, hidden_size).init(device);

        Ok(Self {
            attention_norm,
            attention,
            ffn_norm,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>, position_offset: usize) -> Tensor<B, 3> {
        let attn_input = self.attention_norm.forward(hidden_states.clone());
        let attn_output = self.attention.forward(attn_input, position_offset);
        let hidden_states = hidden_states + attn_output;

        let ffn_input = self.ffn_norm.forward(hidden_states.clone());
        let gate = silu(self.gate_proj.forward(ffn_input.clone()));
        let up = self.up_proj.forward(ffn_input);
        let ffn_output = self.down_proj.forward(gate * up);

        hidden_states + ffn_output
    }

    pub fn forward_with_cache(
        &self,
        hidden_states: Tensor<B, 3>,
        position_offset: usize,
        cache: &mut KVCache<B>,
        layer: usize,
    ) -> Tensor<B, 3> {
        let attn_input = self.attention_norm.forward(hidden_states.clone());
        let attn_output = self
            .attention
            .forward_with_cache(attn_input, position_offset, cache, layer);
        let hidden_states = hidden_states + attn_output;

        let ffn_input = self.ffn_norm.forward(hidden_states.clone());
        let gate = silu(self.gate_proj.forward(ffn_input.clone()));
        let up = self.up_proj.forward(ffn_input);
        let ffn_output = self.down_proj.forward(gate * up);

        hidden_states + ffn_output
    }
}
