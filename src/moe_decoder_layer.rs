use crate::causal_attention::{CausalAttention, RotaryPositionEmbedding};
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::moe_layer::{MoELayer, MoEScratchGpu, PackedExpertWeights};
use crate::rms_norm::RmsNorm;
use crate::tensor::Tensor3;
use crate::types::{Error, Result};
use gllm_kernels::gpu_types::{GpuTensor, TensorDtype};
use gllm_kernels::DispatchedBackend;
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
pub struct MoEDecoderLayer {
    pub(crate) attention_norm: RmsNorm,
    pub(crate) attention: CausalAttention,
    pub(crate) ffn_norm: RmsNorm,
    pub(crate) moe: MoELayer,
    hidden_size: usize,
}

impl MoEDecoderLayer {
    pub fn new(
        config: &ModelConfig,
        rope: Option<Arc<RotaryPositionEmbedding>>,
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
        .ok_or_else(|| Error::InvalidConfig("MoE config missing num_experts".into()))?;
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

        Ok(Self {
            attention_norm: RmsNorm::new(config),
            attention: CausalAttention::new(config, rope, true)?,
            ffn_norm: RmsNorm::new(config),
            moe: MoELayer::new(
                hidden_size,
                moe_intermediate,
                num_experts,
                num_experts_per_tok,
                n_shared_experts,
            ),
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
        let moe_output = self.moe.forward(&ffn_input)?;

        add_tensors(&residual, &moe_output)
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
        let moe_output = self.moe.forward(&ffn_input)?;

        add_tensors(&residual, &moe_output)
    }

    pub fn forward_inplace_gpu_with_cache(
        &self,
        hidden_states: &mut GpuTensor,
        position_offset: usize,
        cache: &mut KVCache,
        layer: usize,
        packed_weights: &PackedExpertWeights,
        scratch: &mut MoEScratchGpu,
        backend: &DispatchedBackend,
    ) -> Result<()> {
        let backend_type = hidden_states.backend;
        let mut normed_attn_input = GpuTensor::new_temp(
            vec![1, self.hidden_size],
            TensorDtype::F32,
            backend_type,
        )
        .map_err(|e| Error::InferenceError(e.to_string()))?;
        let attn_result = self
            .attention_norm
            .forward_gpu(hidden_states, &mut normed_attn_input)
            .and_then(|_| {
                self.attention.forward_gpu_inplace(
                    &normed_attn_input,
                    hidden_states,
                    position_offset,
                    cache,
                    layer,
                )
            });
        normed_attn_input.release();
        attn_result?;

        let mut normed_ffn_input = GpuTensor::new_temp(
            vec![1, self.hidden_size],
            TensorDtype::F32,
            backend_type,
        )
        .map_err(|e| Error::InferenceError(e.to_string()))?;
        let moe_result = self
            .ffn_norm
            .forward_gpu(hidden_states, &mut normed_ffn_input)
            .and_then(|_| {
                self.moe.forward_inplace_gpu_fused(
                    &normed_ffn_input,
                    hidden_states,
                    packed_weights,
                    scratch,
                    backend,
                )
            });
        normed_ffn_input.release();
        moe_result
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
