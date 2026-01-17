use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::tensor::Tensor3;
use crate::types::{Error, Result};
use crate::weight_loader::LinearWeights;
use gllm_kernels::{
    linear_forward, rope_apply_inplace, rope_precompute, FlashAttentionConfig, KernelDispatcher,
    KernelFloat, RoPEConfig as KernelRoPEConfig,
};
use std::sync::Arc;

/// RoPE configuration for attention.
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

/// Rotary Position Embedding using gllm-kernels.
#[derive(Clone)]
pub struct RotaryPositionEmbedding {
    cos_cached: Vec<f32>,
    sin_cached: Vec<f32>,
    dim: usize,
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
            dim,
            max_cached_len: max_seq_len,
        }
    }

    pub fn apply_inplace<T: KernelFloat>(
        &self,
        tensor: &mut [T],
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

    pub fn dim(&self) -> usize {
        self.dim
    }
}

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
    pub(crate) hidden_size: usize,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) causal: bool,
    dispatcher: Arc<KernelDispatcher>,
}

impl CausalAttention {
    pub fn new(
        config: &ModelConfig,
        rope: Option<Arc<RotaryPositionEmbedding>>,
        causal: bool,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        if hidden_size == 0 {
            return Err(Error::InvalidConfig(
                "hidden_size must be greater than 0 for attention".into(),
            ));
        }

        let num_attention_heads = config.num_attention_heads;
        if num_attention_heads == 0 {
            return Err(Error::InvalidConfig(
                "num_attention_heads must be greater than 0 for attention".into(),
            ));
        }
        if hidden_size % num_attention_heads != 0 {
            return Err(Error::InvalidConfig(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                hidden_size, num_attention_heads
            )));
        }

        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
        if num_key_value_heads == 0 {
            return Err(Error::InvalidConfig(
                "num_key_value_heads must be greater than 0 for attention".into(),
            ));
        }
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(Error::InvalidConfig(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                num_attention_heads, num_key_value_heads
            )));
        }

        let head_dim = config
            .head_dim
            .unwrap_or_else(|| hidden_size / num_attention_heads);
        if head_dim % 2 != 0 && Self::uses_rope(config) {
            return Err(Error::InvalidConfig(
                "head_dim must be even to apply RoPE".into(),
            ));
        }

        let rope = if Self::uses_rope(config) {
            if rope.is_none() {
                return Err(Error::InvalidConfig(
                    "RoPE enabled but no precomputed cache was provided".into(),
                ));
            }
            rope
        } else {
            None
        };

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
            hidden_size,
            sliding_window: config.sliding_window,
            causal,
            dispatcher: Arc::new(KernelDispatcher::new()),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor3, position_offset: usize) -> Result<Tensor3> {
        let (batch, seq_len, _) = hidden_states.shape();
        let q = self.project(hidden_states, &self.q_proj)?;
        let k = self.project(hidden_states, &self.k_proj)?;
        let v = self.project(hidden_states, &self.v_proj)?;

        let mut q = reshape_to_heads(&q, batch, seq_len, self.num_attention_heads, self.head_dim);
        let mut k = reshape_to_heads(&k, batch, seq_len, self.num_key_value_heads, self.head_dim);
        let v = reshape_to_heads(&v, batch, seq_len, self.num_key_value_heads, self.head_dim);

        if let Some(rope) = &self.rope {
            rope.apply_inplace(
                &mut q,
                batch,
                seq_len,
                self.num_attention_heads,
                self.head_dim,
                position_offset,
            );
            rope.apply_inplace(
                &mut k,
                batch,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
                position_offset,
            );
        }

        let (k, v) = self.repeat_kv(&k, &v, batch, seq_len, seq_len);
        let context = self.attend(&q, &k, &v, batch, seq_len, seq_len, self.causal);
        let merged = merge_heads(&context, batch, seq_len, self.num_attention_heads, self.head_dim);
        self.project_output(&merged, batch, seq_len)
    }

    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor3,
        position_offset: usize,
        cache: &mut KVCache,
        layer: usize,
    ) -> Result<Tensor3> {
        let (batch, seq_len, _) = hidden_states.shape();
        let q = self.project(hidden_states, &self.q_proj)?;
        let k = self.project(hidden_states, &self.k_proj)?;
        let v = self.project(hidden_states, &self.v_proj)?;

        let mut q = reshape_to_heads(&q, batch, seq_len, self.num_attention_heads, self.head_dim);
        let mut k = reshape_to_heads(&k, batch, seq_len, self.num_key_value_heads, self.head_dim);
        let v = reshape_to_heads(&v, batch, seq_len, self.num_key_value_heads, self.head_dim);

        if let Some(rope) = &self.rope {
            rope.apply_inplace(
                &mut q,
                batch,
                seq_len,
                self.num_attention_heads,
                self.head_dim,
                position_offset,
            );
            rope.apply_inplace(
                &mut k,
                batch,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
                position_offset,
            );
        }

        cache.update(layer, &k, &v)?;
        let cached_k = cache.layer_k(layer)?;
        let cached_v = cache.layer_v(layer)?;
        let key_len = cache.seq_len();

        let max_len = cached_k.len()
            / batch
                .max(1)
                .saturating_mul(self.num_key_value_heads.max(1))
                .saturating_mul(self.head_dim.max(1));
        let (k, v) = self.repeat_kv(cached_k, cached_v, batch, key_len, max_len);
        let causal = self.causal && seq_len == key_len;
        let context = self.attend(&q, &k, &v, batch, seq_len, key_len, causal);
        let merged = merge_heads(&context, batch, seq_len, self.num_attention_heads, self.head_dim);
        self.project_output(&merged, batch, seq_len)
    }

    pub(crate) fn build_rope(
        config: &ModelConfig,
        head_dim: usize,
    ) -> Option<Arc<RotaryPositionEmbedding>> {
        if !Self::uses_rope(config) {
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

    fn project(&self, hidden_states: &Tensor3, weights: &LinearWeights) -> Result<Vec<f32>> {
        let (batch, seq_len, hidden) = hidden_states.shape();
        if hidden != weights.weight.cols {
            return Err(Error::InferenceError(
                "Attention projection input shape mismatch".into(),
            ));
        }
        let rows = batch * seq_len;
        let mut output = vec![0.0f32; rows * weights.weight.rows];
        linear_forward(
            &hidden_states.data,
            weights.weight.as_slice(),
            weights.bias.as_ref().map(|b| b.as_slice()),
            &mut output,
            rows,
            weights.weight.cols,
            weights.weight.rows,
        );
        Ok(output)
    }

    fn project_output(&self, input: &[f32], batch: usize, seq_len: usize) -> Result<Tensor3> {
        let rows = batch * seq_len;
        let mut output = vec![0.0f32; rows * self.o_proj.weight.rows];
        linear_forward(
            input,
            self.o_proj.weight.as_slice(),
            self.o_proj.bias.as_ref().map(|b| b.as_slice()),
            &mut output,
            rows,
            self.o_proj.weight.cols,
            self.o_proj.weight.rows,
        );
        Tensor3::new(output, batch, seq_len, self.o_proj.weight.rows)
    }

    fn attend(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq_len_q: usize,
        seq_len_kv: usize,
        causal: bool,
    ) -> Vec<f32> {
        let mut output =
            vec![0.0f32; batch * self.num_attention_heads * seq_len_q * self.head_dim];
        self.dispatcher.flash_attention(
            q,
            k,
            v,
            &mut output,
            FlashAttentionConfig {
                causal,
                use_log_space_softmax: true,
                use_kahan_accumulator: true,
                num_heads: self.num_attention_heads,
                head_dim: self.head_dim,
                seq_len_q,
                seq_len_kv,
                batch_size: batch,
                ..Default::default()
            },
        );
        output
    }

    fn repeat_kv(
        &self,
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq_len: usize,
        src_stride: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let repeat = self.num_attention_heads / self.num_key_value_heads;
        let dst_head_stride = seq_len * self.head_dim;
        let src_head_stride = src_stride * self.head_dim;
        let mut k_out =
            vec![0.0f32; batch * self.num_attention_heads * dst_head_stride];
        let mut v_out =
            vec![0.0f32; batch * self.num_attention_heads * dst_head_stride];

        for b in 0..batch {
            for kv in 0..self.num_key_value_heads {
                for r in 0..repeat {
                    let h = kv * repeat + r;
                    let src_base = (b * self.num_key_value_heads + kv) * src_head_stride;
                    let dst_base = (b * self.num_attention_heads + h) * dst_head_stride;
                    let src_slice = &k[src_base..src_base + dst_head_stride];
                    k_out[dst_base..dst_base + dst_head_stride].copy_from_slice(src_slice);
                    let src_slice = &v[src_base..src_base + dst_head_stride];
                    v_out[dst_base..dst_base + dst_head_stride].copy_from_slice(src_slice);
                }
            }
        }

        (k_out, v_out)
    }

    fn uses_rope(config: &ModelConfig) -> bool {
        config
            .position_embedding_type
            .as_deref()
            .map_or(false, |t| t == "rope" || t == "rotary")
            || config.rope_theta.is_some()
    }
}

fn reshape_to_heads(
    input: &[f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * num_heads * seq_len * head_dim];
    let row_stride = num_heads * head_dim;
    for b in 0..batch {
        for s in 0..seq_len {
            let row_base = (b * seq_len + s) * row_stride;
            for h in 0..num_heads {
                let src = row_base + h * head_dim;
                let dst = (b * num_heads + h) * seq_len * head_dim + s * head_dim;
                output[dst..dst + head_dim]
                    .copy_from_slice(&input[src..src + head_dim]);
            }
        }
    }
    output
}

fn merge_heads(
    input: &[f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    let row_stride = num_heads * head_dim;
    for b in 0..batch {
        for s in 0..seq_len {
            let row_base = (b * seq_len + s) * row_stride;
            for h in 0..num_heads {
                let src = (b * num_heads + h) * seq_len * head_dim + s * head_dim;
                let dst = row_base + h * head_dim;
                output[dst..dst + head_dim]
                    .copy_from_slice(&input[src..src + head_dim]);
            }
        }
    }
    output
}
