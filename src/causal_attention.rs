//! Causal Attention using gllm-kernels Backend trait.

use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::scratch_buffer::ScratchBuffer;
use crate::types::{Error, Result};
use crate::weight_loader::LinearWeights;
use gllm_kernels::backend::{Backend, TensorSlice, TensorSliceMut};
use gllm_kernels::{rope_apply_inplace, rope_precompute, FlashAttentionConfig, RoPEConfig as KernelRoPEConfig};
use std::borrow::Cow;
use std::sync::Arc;

/// RoPE configuration.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub theta: f64,
    pub dim: usize,
    pub max_seq_len: usize,
    pub ntk_factor: Option<f64>,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            theta: 10000.0,
            dim: 64,
            max_seq_len: 8192,
            ntk_factor: None,
        }
    }
}

/// Rotary Position Embedding.
#[derive(Clone)]
pub struct RotaryPositionEmbedding {
    cos_cached: Vec<f32>,
    sin_cached: Vec<f32>,
    max_cached_len: usize,
}

impl RotaryPositionEmbedding {
    pub fn new(config: RopeConfig) -> Self {
        let dim = config.dim;
        let max_seq_len = config.max_seq_len;
        let half_dim = dim / 2;
        let kernel_config = KernelRoPEConfig {
            dim,
            max_seq_len,
            theta: config.theta,
            ntk_factor: config.ntk_factor,
        };

        let expected_size = max_seq_len * half_dim;
        let mut cos_vals = vec![0.0f32; expected_size];
        let mut sin_vals = vec![0.0f32; expected_size];
        rope_precompute(&mut cos_vals, &mut sin_vals, &kernel_config);

        Self {
            cos_cached: cos_vals,
            sin_cached: sin_vals,
            max_cached_len: max_seq_len,
        }
    }

    pub fn apply_inplace(
        &self,
        tensor: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) {
        let half_dim = head_dim / 2;
        let needed = (position_offset + seq_len) * half_dim;
        if needed > self.cos_cached.len() || needed > self.sin_cached.len() {
            return;
        }
        rope_apply_inplace(
            tensor,
            &self.cos_cached,
            &self.sin_cached,
            batch,
            seq_len,
            num_heads,
            head_dim,
            position_offset,
        );
    }

    pub fn max_cached_len(&self) -> usize {
        self.max_cached_len
    }
}

/// Causal Self-Attention layer.
#[derive(Clone)]
pub struct CausalAttention {
    pub(crate) q_proj: LinearWeights,
    pub(crate) k_proj: LinearWeights,
    pub(crate) v_proj: LinearWeights,
    pub(crate) o_proj: LinearWeights,
    pub(crate) rope: Option<Arc<RotaryPositionEmbedding>>,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) causal: bool,
    backend: Arc<dyn Backend>,
}

impl CausalAttention {
    pub fn new(
        config: &ModelConfig,
        rope: Option<Arc<RotaryPositionEmbedding>>,
        causal: bool,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_attention_heads);

        if hidden_size == 0 || num_attention_heads == 0 {
            return Err(Error::InvalidConfig("Invalid attention config".into()));
        }

        let q_rows = num_attention_heads * head_dim;
        let kv_rows = num_key_value_heads * head_dim;
        let q_proj = LinearWeights::zeros(q_rows, hidden_size);
        let k_proj = LinearWeights::zeros(kv_rows, hidden_size);
        let v_proj = LinearWeights::zeros(kv_rows, hidden_size);
        let o_proj = LinearWeights::zeros(hidden_size, q_rows);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            causal,
            backend,
        })
    }

    /// Forward with KV cache for autoregressive generation.
    pub fn forward_with_cache(
        &self,
        input: &[f32],
        position_offset: usize,
        cache: &mut KVCache,
        layer: usize,
    ) -> Result<Vec<f32>> {
        let batch = 1;
        let hidden_size = self.hidden_size();
        if hidden_size == 0 || input.len() % hidden_size != 0 {
            return Err(Error::InferenceError(
                "Attention input length must align with hidden size".into(),
            ));
        }
        let seq_len = input.len() / hidden_size;
        let q_size = self.num_attention_heads * self.head_dim;
        let kv_size = self.num_key_value_heads * self.head_dim;

        let mut q = vec![0.0f32; batch * seq_len * q_size];
        let mut k = vec![0.0f32; batch * seq_len * kv_size];
        let mut v = vec![0.0f32; batch * seq_len * kv_size];
        let mut attn_out = vec![0.0f32; batch * seq_len * q_size];
        let mut output = vec![0.0f32; batch * seq_len * hidden_size];

        self.forward_with_cache_buffers(
            input,
            position_offset,
            cache,
            layer,
            &mut q,
            &mut k,
            &mut v,
            &mut attn_out,
            &mut output,
            hidden_size,
        )?;

        Ok(output)
    }

    pub(crate) fn forward_with_cache_scratch(
        &self,
        input: &[f32],
        position_offset: usize,
        cache: &mut KVCache,
        layer: usize,
        scratch: &mut ScratchBuffer,
        output: &mut [f32],
    ) -> Result<()> {
        let batch = 1;
        let hidden_size = self.hidden_size();
        if hidden_size == 0 || input.len() % hidden_size != 0 {
            return Err(Error::InferenceError(
                "Attention input length must align with hidden size".into(),
            ));
        }
        let seq_len = input.len() / hidden_size;
        let workspace = scratch.attn_workspace(batch, seq_len);
        self.forward_with_cache_buffers(
            input,
            position_offset,
            cache,
            layer,
            workspace.q,
            workspace.k,
            workspace.v,
            workspace.out,
            output,
            hidden_size,
        )
    }

    fn forward_with_cache_buffers(
        &self,
        input: &[f32],
        position_offset: usize,
        cache: &mut KVCache,
        layer: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        attn_out: &mut [f32],
        output: &mut [f32],
        hidden_size: usize,
    ) -> Result<()> {
        let batch = 1;
        let seq_len = input.len() / hidden_size;
        self.project(input, &self.q_proj, q, batch * seq_len)?;
        self.project(input, &self.k_proj, k, batch * seq_len)?;
        self.project(input, &self.v_proj, v, batch * seq_len)?;

        if let Some(rope) = &self.rope {
            rope.apply_inplace(q, batch, seq_len, self.num_attention_heads, self.head_dim, position_offset);
            rope.apply_inplace(k, batch, seq_len, self.num_key_value_heads, self.head_dim, position_offset);
        }

        cache.update(layer, k, v)?;
        let cached_k = cache.layer_k(layer)?;
        let cached_v = cache.layer_v(layer)?;
        let key_len = cache.cached_len();

        let (k_expanded, v_expanded) = self.expand_kv(cached_k, cached_v, batch, key_len);

        let config = FlashAttentionConfig {
            causal: self.causal && seq_len == key_len,
            num_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            seq_len_q: seq_len,
            seq_len_kv: key_len,
            batch_size: batch,
            ..Default::default()
        };

        self.backend
            .flash_attention(
                TensorSlice::F32(q),
                TensorSlice::F32(k_expanded.as_ref()),
                TensorSlice::F32(v_expanded.as_ref()),
                TensorSliceMut::F32(attn_out),
                config,
            )
            .map_err(|e| Error::InferenceError(e))?;

        self.project(attn_out, &self.o_proj, output, batch * seq_len)?;

        Ok(())
    }

    fn project(
        &self,
        input: &[f32],
        weights: &LinearWeights,
        output: &mut [f32],
        rows: usize,
    ) -> Result<()> {
        weights.forward(input, output, rows, self.backend.as_ref())
    }

    fn expand_kv<'a>(
        &self,
        k: &'a [f32],
        v: &'a [f32],
        batch: usize,
        seq_len: usize,
    ) -> (Cow<'a, [f32]>, Cow<'a, [f32]>) {
        let repeat = self.num_attention_heads / self.num_key_value_heads;
        if repeat == 1 {
            return (Cow::Borrowed(k), Cow::Borrowed(v));
        }

        let head_stride = seq_len * self.head_dim;
        let mut k_out = vec![0.0f32; batch * self.num_attention_heads * head_stride];
        let mut v_out = vec![0.0f32; batch * self.num_attention_heads * head_stride];

        for b in 0..batch {
            for kv in 0..self.num_key_value_heads {
                for r in 0..repeat {
                    let h = kv * repeat + r;
                    let src_base = (b * self.num_key_value_heads + kv) * head_stride;
                    let dst_base = (b * self.num_attention_heads + h) * head_stride;
                    k_out[dst_base..dst_base + head_stride]
                        .copy_from_slice(&k[src_base..src_base + head_stride]);
                    v_out[dst_base..dst_base + head_stride]
                        .copy_from_slice(&v[src_base..src_base + head_stride]);
                }
            }
        }
        (Cow::Owned(k_out), Cow::Owned(v_out))
    }

    /// Forward without KV cache (for BERT-style bidirectional attention).
    /// Input shape: [batch * seq_len, hidden_size]
    pub fn forward(
        &self,
        input: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.hidden_size();
        if hidden_size == 0 || input.len() != batch * seq_len * hidden_size {
            return Err(Error::InferenceError(
                "Attention input length must align with hidden size".into(),
            ));
        }
        let q_size = self.num_attention_heads * self.head_dim;
        let kv_size = self.num_key_value_heads * self.head_dim;

        // Projections
        let mut q = vec![0.0f32; batch * seq_len * q_size];
        let mut k = vec![0.0f32; batch * seq_len * kv_size];
        let mut v = vec![0.0f32; batch * seq_len * kv_size];

        self.project(input, &self.q_proj, &mut q, batch * seq_len)?;
        self.project(input, &self.k_proj, &mut k, batch * seq_len)?;
        self.project(input, &self.v_proj, &mut v, batch * seq_len)?;

        // RoPE (if enabled)
        if let Some(rope) = &self.rope {
            rope.apply_inplace(&mut q, batch, seq_len, self.num_attention_heads, self.head_dim, 0);
            rope.apply_inplace(&mut k, batch, seq_len, self.num_key_value_heads, self.head_dim, 0);
        }

        // Expand KV for GQA if needed
        let (k_expanded, v_expanded) = self.expand_kv(&k, &v, batch, seq_len);

        // Flash Attention via Backend trait
        let mut attn_out = vec![0.0f32; batch * seq_len * q_size];
        let config = FlashAttentionConfig {
            causal: self.causal,
            num_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            seq_len_q: seq_len,
            seq_len_kv: seq_len,
            batch_size: batch,
            ..Default::default()
        };

        self.backend
            .flash_attention(
                TensorSlice::F32(&q),
                TensorSlice::F32(k_expanded.as_ref()),
                TensorSlice::F32(v_expanded.as_ref()),
                TensorSliceMut::F32(&mut attn_out),
                config,
            )
            .map_err(|e| Error::InferenceError(e))?;

        // Output projection
        let mut output = vec![0.0f32; batch * seq_len * hidden_size];
        self.project(&attn_out, &self.o_proj, &mut output, batch * seq_len)?;

        Ok(output)
    }

    pub(crate) fn build_rope(config: &ModelConfig, head_dim: usize) -> Option<Arc<RotaryPositionEmbedding>> {
        let uses_rope = config
            .position_embedding_type
            .as_deref()
            .map_or(false, |t| t == "rope" || t == "rotary")
            || config.rope_theta.is_some();

        if !uses_rope {
            return None;
        }

        let ntk_factor = config
            .rope_scaling
            .as_ref()
            .and_then(|scaling| scaling.get("factor").and_then(|v| v.as_f64()));

        let rope_config = RopeConfig {
            theta: config.rope_theta.unwrap_or(10000.0),
            dim: head_dim,
            max_seq_len: config.max_position_embeddings,
            ntk_factor,
        };
        Some(Arc::new(RotaryPositionEmbedding::new(rope_config)))
    }

    fn hidden_size(&self) -> usize {
        self.q_proj.in_features()
    }
}
