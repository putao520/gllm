use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::rope::{RopeConfig, RotaryPositionEmbedding};
use crate::types::{Error, Result};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::sync::Arc;

#[derive(Clone)]
pub struct CausalAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    rope: Option<Arc<RotaryPositionEmbedding<B>>>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    sliding_window: Option<usize>,
    device: B::Device,
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
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.hidden_size]);

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
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.hidden_size]);

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
        position_offset: usize,
        key_len: usize,
    ) -> Tensor<B, 4> {
        let [batch_size, num_heads, query_len, head_dim] = q.dims();
        let scale = (self.head_dim as f32).sqrt();
        let k_t = k.transpose();

        if query_len > 4096 {
            let mut outputs = Vec::new();
            let mut start = 0usize;
            while start < query_len {
                let end = (start + 2048).min(query_len);
                let q_slice = q
                    .clone()
                    .slice([0..batch_size, 0..num_heads, start..end, 0..head_dim]);
                let mut scores = q_slice.matmul(k_t.clone()) / scale;
                let mask = self.build_causal_mask(end - start, key_len, position_offset + start);
                scores = scores + mask;
                let attn = softmax(scores, 3);
                let context = attn.matmul(v.clone());
                outputs.push(context);
                start = end;
            }
            Tensor::cat(outputs, 2)
        } else {
            let mut scores = q.matmul(k_t) / scale;
            let mask = self.build_causal_mask(query_len, key_len, position_offset);
            scores = scores + mask;
            let attn = softmax(scores, 3);
            attn.matmul(v)
        }
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
    use burn::backend::ndarray::NdArray;

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
