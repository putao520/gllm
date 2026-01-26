//! Transformer Decoder Layer using gllm-kernels ops.

use crate::causal_attention::CausalAttention;
use crate::kv_cache::KVCache;
use crate::rms_norm::RmsNorm;
use crate::scratch_buffer::ScratchBuffer;
use crate::types::{Error, Result};
use crate::weight_loader::LinearWeights;
use gllm_kernels::backend::BackendImpl;
use gllm_kernels::{gelu_inplace, silu_inplace};

/// Feed-Forward Network weights.
#[derive(Clone)]
pub struct FFNWeights {
    pub gate_proj: LinearWeights,
    pub up_proj: LinearWeights,
    pub down_proj: LinearWeights,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl FFNWeights {
    pub fn zeros(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: LinearWeights::zeros(intermediate_size, hidden_size),
            up_proj: LinearWeights::zeros(intermediate_size, hidden_size),
            down_proj: LinearWeights::zeros(hidden_size, intermediate_size),
            hidden_size,
            intermediate_size,
        }
    }
}

/// Transformer Decoder Layer.
#[derive(Clone)]
pub struct DecoderLayer {
    pub attention: CausalAttention,
    pub input_norm: RmsNorm,
    pub post_attn_norm: RmsNorm,
    pub ffn: FFNWeights,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub use_gelu: bool,
    pub backend: BackendImpl,
}

impl DecoderLayer {
    /// Forward pass with KV cache.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        position_offset: usize,
        cache: &mut KVCache,
        layer_idx: usize,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.hidden_size();
        if hidden_states.len() % hidden_size != 0 {
            return Err(Error::InferenceError(
                "Decoder layer input length must align with hidden size".into(),
            ));
        }
        let seq_tokens = hidden_states.len() / hidden_size;

        // Pre-attention norm
        let normed = self.input_norm.forward(hidden_states);

        // Self-attention
        let attn_output = self
            .attention
            .forward_with_cache(&normed, position_offset, cache, layer_idx)?;

        // Residual connection (in-place to avoid allocation)
        let mut hidden = attn_output;
        for (h, s) in hidden.iter_mut().zip(hidden_states.iter()) {
            *h += s;
        }

        // Post-attention norm
        let normed = self.post_attn_norm.forward(&hidden);

        // FFN
        let ffn_out = self.ffn_forward(&normed, seq_tokens)?;

        // Residual connection
        for (h, f) in hidden.iter_mut().zip(ffn_out.iter()) {
            *h += f;
        }

        Ok(hidden)
    }

    pub(crate) fn forward_with_scratch(
        &self,
        hidden_states: &[f32],
        position_offset: usize,
        cache: &mut KVCache,
        layer_idx: usize,
        scratch: &mut ScratchBuffer,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.hidden_size();
        if hidden_states.len() % hidden_size != 0 {
            return Err(Error::InferenceError(
                "Decoder layer input length must align with hidden size".into(),
            ));
        }
        let seq_tokens = hidden_states.len() / hidden_size;

        let normed = self.input_norm.forward(hidden_states);
        let mut hidden = vec![0.0f32; hidden_states.len()];
        self.attention.forward_with_cache_scratch(
            &normed,
            position_offset,
            cache,
            layer_idx,
            scratch,
            &mut hidden,
        )?;
        for (h, s) in hidden.iter_mut().zip(hidden_states.iter()) {
            *h += s;
        }

        let normed = self.post_attn_norm.forward(&hidden);
        let workspace = scratch.ffn_workspace(seq_tokens, 1);
        self.ffn_forward_with_buffers(
            &normed,
            seq_tokens,
            workspace.gate,
            workspace.up,
            workspace.output,
        )?;

        for (h, f) in hidden.iter_mut().zip(workspace.output.iter()) {
            *h += f;
        }

        Ok(hidden)
    }

    fn ffn_forward(&self, input: &[f32], batch: usize) -> Result<Vec<f32>> {
        let intermediate_size = self.intermediate_size();
        let mut gate = vec![0.0f32; batch * intermediate_size];
        let mut up = vec![0.0f32; batch * intermediate_size];
        let mut output = vec![0.0f32; batch * self.hidden_size()];
        self.ffn_forward_with_buffers(input, batch, &mut gate, &mut up, &mut output)?;
        Ok(output)
    }

    fn ffn_forward_with_buffers(
        &self,
        input: &[f32],
        batch: usize,
        gate: &mut [f32],
        up: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        self.ffn
            .gate_proj
            .forward(input, gate, batch, &self.backend)?;

        self.ffn
            .up_proj
            .forward(input, up, batch, &self.backend)?;

        if self.use_gelu {
            gelu_inplace(gate);
        } else {
            silu_inplace(gate);
        }

        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g *= u;
        }

        self.ffn
            .down_proj
            .forward(gate, output, batch, &self.backend)?;

        Ok(())
    }

    fn hidden_size(&self) -> usize {
        self.ffn.down_proj.out_features()
    }

    fn intermediate_size(&self) -> usize {
        self.ffn.gate_proj.out_features()
    }
}
