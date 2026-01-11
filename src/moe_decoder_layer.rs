use crate::causal_attention::CausalAttention;
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::moe_layer::MoELayer;
use crate::rope::RotaryPositionEmbedding;
use crate::rms_norm::RmsNorm;
use crate::types::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use serde_json::Value;
use std::sync::Arc;

fn resolve_moe_value(config: &ModelConfig, direct: Option<usize>, keys: &[&str]) -> Option<usize> {
    if direct.is_some() {
        return direct;
    }
    keys.iter().find_map(|key| lookup_extra_usize(&config.extra, key))
}

fn lookup_extra_usize(extra: &Value, key: &str) -> Option<usize> {
    extra.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

#[derive(Clone)]
pub struct MoEDecoderLayer<B: Backend> {
    attention_norm: RmsNorm<B>,
    attention: CausalAttention<B>,
    ffn_norm: RmsNorm<B>,
    moe: MoELayer<B>,
}

impl<B: Backend> MoEDecoderLayer<B> {
    pub fn new(
        device: &B::Device,
        config: &ModelConfig,
        rope: Option<Arc<RotaryPositionEmbedding<B>>>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let fallback_intermediate = config
            .intermediate_size
            .unwrap_or(hidden_size.saturating_mul(4));
        let moe_intermediate = config
            .moe_intermediate_size
            .or_else(|| lookup_extra_usize(&config.extra, "moe_intermediate_size"))
            .unwrap_or(fallback_intermediate);

        let num_experts = resolve_moe_value(
            config,
            config.num_experts,
            &["n_routed_experts", "num_experts", "num_local_experts"],
        )
        .ok_or_else(|| {
            Error::InvalidConfig("MoE config missing num_experts (n_routed_experts)".into())
        })?;
        let num_experts_per_tok = resolve_moe_value(
            config,
            config.num_experts_per_tok,
            &["num_experts_per_tok", "num_experts_per_token", "top_k"],
        )
        .ok_or_else(|| Error::InvalidConfig("MoE config missing num_experts_per_tok".into()))?;
        let n_shared_experts = resolve_moe_value(
            config,
            config.n_shared_experts,
            &["n_shared_experts", "num_shared_experts"],
        )
        .unwrap_or(0);

        if num_experts == 0 {
            return Err(Error::InvalidConfig(
                "num_experts must be greater than 0 for MoE".into(),
            ));
        }
        if num_experts_per_tok == 0 || num_experts_per_tok > num_experts {
            return Err(Error::InvalidConfig(
                "num_experts_per_tok must be in 1..=num_experts for MoE".into(),
            ));
        }

        let attention_norm = RmsNorm::new(device, config);
        let attention = CausalAttention::new(device, config, rope)?;
        let ffn_norm = RmsNorm::new(device, config);
        let moe = MoELayer::new(
            device,
            hidden_size,
            moe_intermediate,
            num_experts,
            num_experts_per_tok,
            n_shared_experts,
        );

        Ok(Self {
            attention_norm,
            attention,
            ffn_norm,
            moe,
        })
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>, position_offset: usize) -> Tensor<B, 3> {
        let attn_input = self.attention_norm.forward(hidden_states.clone());
        let attn_output = self.attention.forward(attn_input, position_offset);
        let hidden_states = hidden_states + attn_output;

        let ffn_input = self.ffn_norm.forward(hidden_states.clone());
        let moe_output = self.moe.forward(ffn_input);

        hidden_states + moe_output
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
        let moe_output = self.moe.forward(ffn_input);

        hidden_states + moe_output
    }
}
