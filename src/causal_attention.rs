use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::rope::{RopeConfig, RotaryPositionEmbedding};
use crate::types::{Error, Result};
#[cfg(feature = "paged-attention")]
use crate::paged_attention::PagedKVCache;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::sync::{Arc, OnceLock};

// Zero-cost kernel dispatcher (initialized once)
use gllm_kernels::{KernelDispatcher, FlashAttentionConfig as KernelFlashConfig};

static DISPATCHER: OnceLock<KernelDispatcher> = OnceLock::new();

#[inline(always)]
fn get_dispatcher() -> &'static KernelDispatcher {
    DISPATCHER.get_or_init(KernelDispatcher::new)
}

#[derive(Clone)]
pub struct CausalAttention<B: Backend> {
    pub(crate) q_proj: Linear<B>,
    pub(crate) k_proj: Linear<B>,
    pub(crate) v_proj: Linear<B>,
    pub(crate) o_proj: Linear<B>,
    pub(crate) rope: Option<Arc<RotaryPositionEmbedding<B>>>,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    #[allow(dead_code)] // Used for documentation/debugging
    pub(crate) hidden_size: usize,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) device: B::Device,
}

impl<B: Backend> CausalAttention<B> {
    pub fn new(
        device: &B::Device,
        config: &ModelConfig,
        rope: Option<Arc<RotaryPositionEmbedding<B>>>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        if hidden_size == 0 {
            return Err(Error::InvalidConfig(
                "hidden_size must be greater than 0 for causal attention".into(),
            ));
        }

        let num_attention_heads = config.num_attention_heads;
        if num_attention_heads == 0 {
            return Err(Error::InvalidConfig(
                "num_attention_heads must be greater than 0 for causal attention".into(),
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
                "num_key_value_heads must be greater than 0 for causal attention".into(),
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
        if head_dim % 2 != 0 {
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

        let q_proj = LinearConfig::new(hidden_size, num_attention_heads * head_dim).init(device);
        let k_proj = LinearConfig::new(hidden_size, num_key_value_heads * head_dim).init(device);
        let v_proj = LinearConfig::new(hidden_size, num_key_value_heads * head_dim).init(device);
        let o_proj = LinearConfig::new(num_attention_heads * head_dim, hidden_size).init(device);

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
            device: device.clone(),
        })
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>, position_offset: usize) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden] = hidden_states.dims();

        let q = self
            .q_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_attention_heads, self.head_dim]);
        let k = self
            .k_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]);
        let v = self
            .v_proj
            .forward(hidden_states)
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]);

        let (q, k) = match &self.rope {
            Some(rope) => rope.apply(q, k, position_offset),
            None => (q, k),
        };

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        let context = self.attend(q, k, v, position_offset, seq_len);
        // Reshape to [batch, seq, num_heads * head_dim] for o_proj
        // Note: For models like Qwen3-MoE, num_heads * head_dim != hidden_size
        let attn_out_dim = self.num_attention_heads * self.head_dim;
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, attn_out_dim]);

        self.o_proj.forward(context)
    }

    pub fn forward_with_cache(
        &self,
        hidden_states: Tensor<B, 3>,
        position_offset: usize,
        cache: &mut KVCache<B>,
        layer: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden] = hidden_states.dims();

        let q = self
            .q_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_attention_heads, self.head_dim]);
        let k = self
            .k_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]);
        let v = self
            .v_proj
            .forward(hidden_states)
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]);

        let (q, k) = match &self.rope {
            Some(rope) => rope.apply(q, k, position_offset),
            None => (q, k),
        };

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = cache.update(layer, k, v);

        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        let key_len = k.dims()[2];
        let context = self.attend(q, k, v, position_offset, key_len);
        // Reshape to [batch, seq, num_heads * head_dim] for o_proj
        let attn_out_dim = self.num_attention_heads * self.head_dim;
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, attn_out_dim]);

        self.o_proj.forward(context)
    }

    #[cfg(feature = "paged-attention")]
    pub fn forward_with_paged_cache(
        &self,
        hidden_states: Tensor<B, 3>,
        position_offset: usize,
        cache: &mut PagedKVCache<B>,
        layer: usize,
        seq_id: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden] = hidden_states.dims();
        assert_eq!(
            batch_size, 1,
            "paged attention currently supports batch_size == 1"
        );

        let q = self
            .q_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_attention_heads, self.head_dim]);
        let k = self
            .k_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]);
        let v = self
            .v_proj
            .forward(hidden_states)
            .reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]);

        let (q, k) = match &self.rope {
            Some(rope) => rope.apply(q, k, position_offset),
            None => (q, k),
        };

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let k = k
            .clone()
            .slice([
                0..1,
                0..self.num_key_value_heads,
                0..seq_len,
                0..self.head_dim,
            ])
            .reshape([self.num_key_value_heads, seq_len, self.head_dim]);
        let v = v
            .clone()
            .slice([
                0..1,
                0..self.num_key_value_heads,
                0..seq_len,
                0..self.head_dim,
            ])
            .reshape([self.num_key_value_heads, seq_len, self.head_dim]);

        cache
            .append(layer, seq_id, k, v)
            .expect("paged cache append failed");

        let (k, v) = cache
            .get_kv(layer, seq_id)
            .expect("paged cache lookup failed");
        let key_len = k.dims()[1];
        let k = k.reshape([1, self.num_key_value_heads, key_len, self.head_dim]);
        let v = v.reshape([1, self.num_key_value_heads, key_len, self.head_dim]);

        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        let context = self.attend(q, k, v, position_offset, key_len);
        let attn_out_dim = self.num_attention_heads * self.head_dim;
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, attn_out_dim]);

        self.o_proj.forward(context)
    }

    fn repeat_kv(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.num_key_value_heads == self.num_attention_heads {
            return tensor;
        }

        let repeat = self.num_attention_heads / self.num_key_value_heads;
        let [batch_size, kv_heads, seq_len, head_dim] = tensor.dims();

        tensor
            .reshape([batch_size, kv_heads, 1, seq_len, head_dim])
            .repeat(&[1, 1, repeat, 1, 1])
            .reshape([batch_size, kv_heads * repeat, seq_len, head_dim])
    }

    fn attend(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        _position_offset: usize,
        _key_len: usize,
    ) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_q, head_dim] = q.dims();
        let seq_kv = k.dims()[2];
        let device = q.device();

        // Extract tensor data as f32 slices
        let q_data: Vec<f32> = q.into_data().into_vec().expect("q to f32");
        let k_data: Vec<f32> = k.into_data().into_vec().expect("k to f32");
        let v_data: Vec<f32> = v.into_data().into_vec().expect("v to f32");

        // Output buffer
        let mut output_data = vec![0.0f32; batch_size * num_heads * seq_q * head_dim];

        // Call kernel dispatcher (zero-cost generic dispatch)
        let config = KernelFlashConfig {
            batch_size,
            num_heads,
            seq_len_q: seq_q,
            seq_len_kv: seq_kv,
            head_dim,
            causal: true,
            use_log_space_softmax: seq_q > 4096 || seq_kv > 4096,
            use_kahan_accumulator: seq_q > 4096 || seq_kv > 4096,
            ..Default::default()
        };

        get_dispatcher().flash_attention(&q_data, &k_data, &v_data, &mut output_data, config);

        // Convert back to tensor
        Tensor::from_data(
            TensorData::new(output_data, [batch_size, num_heads, seq_q, head_dim]),
            &device,
        )
    }

    fn build_causal_mask(
        &self,
        query_len: usize,
        key_len: usize,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        let mut data = Vec::with_capacity(query_len * key_len);
        let mask_value = -1.0e4_f32;
        let window = self.sliding_window.unwrap_or(key_len);

        for i in 0..query_len {
            let absolute_pos = position_offset + i;
            let start = if window > 0 {
                absolute_pos.saturating_sub(window.saturating_sub(1))
            } else {
                0
            };
            for j in 0..key_len {
                let allowed = j <= absolute_pos && j >= start;
                data.push(if allowed { 0.0 } else { mask_value });
            }
        }

        Tensor::<B, 2>::from_data(TensorData::new(data, [query_len, key_len]), &self.device)
            .reshape([1, 1, query_len, key_len])
    }

    fn uses_rope(config: &ModelConfig) -> bool {
        config
            .position_embedding_type
            .as_deref()
            .map_or(false, |t| t == "rope" || t == "rotary")
            || config.rope_theta.is_some()
    }

    pub(crate) fn build_rope(
        device: &B::Device,
        config: &ModelConfig,
        head_dim: usize,
    ) -> Option<Arc<RotaryPositionEmbedding<B>>> {
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
        Some(Arc::new(RotaryPositionEmbedding::new(device, rope_config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn causal_attention_preserves_shape() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut config = ModelConfig::default();
        config.hidden_size = 16;
        config.num_attention_heads = 4;
        config.num_key_value_heads = Some(2);
        config.max_position_embeddings = 16;
        config.intermediate_size = Some(32);
        config.vocab_size = 128;
        config.position_embedding_type = Some("rope".to_string());

        let rope = CausalAttention::<NdArray<f32>>::build_rope(&device, &config, 4);
        let attention = CausalAttention::<NdArray<f32>>::new(&device, &config, rope).expect("init");
        let input = Tensor::<NdArray<f32>, 3>::zeros([2, 5, 16], &device);
        let output = attention.forward(input, 0);

        assert_eq!(output.dims(), [2, 5, 16]);
    }
}
