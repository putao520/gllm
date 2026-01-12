use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

#[derive(Debug, Clone, Copy)]
pub struct FlashAttentionConfig {
    pub block_q: usize,
    pub block_kv: usize,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_q: 64,
            block_kv: 64,
        }
    }
}

pub trait FlashAttention<B: Backend> {
    fn forward(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
    ) -> Tensor<B, 4>;
}

impl<B: Backend> FlashAttention<B> for FlashAttentionConfig {
    fn forward(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        causal: bool,
    ) -> Tensor<B, 4> {
        let key_len = k.dims()[2];
        flash_attention_forward(q, k, v, causal, 0, key_len, None, *self)
    }
}

pub fn flash_attention_forward<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    causal: bool,
    position_offset: usize,
    key_len: usize,
    sliding_window: Option<usize>,
    config: FlashAttentionConfig,
) -> Tensor<B, 4> {
    let device = q.device();
    let [batch_size, num_heads, query_len, head_dim] = q.dims();
    let key_len = key_len.min(k.dims()[2]).min(v.dims()[2]);

    if query_len == 0 || key_len == 0 {
        return Tensor::zeros([batch_size, num_heads, query_len, head_dim], &device);
    }

    let block_q = config.block_q.max(1);
    let block_kv = config.block_kv.max(1);
    let window = sliding_window.unwrap_or(key_len);
    let scale = (head_dim as f32).sqrt();

    let mut outputs = Vec::new();
    let mut q_start = 0usize;
    while q_start < query_len {
        let q_end = (q_start + block_q).min(query_len);
        let q_block_len = q_end - q_start;
        let q_block = q
            .clone()
            .slice([0..batch_size, 0..num_heads, q_start..q_end, 0..head_dim]);

        let mut m_i = Tensor::<B, 4>::full(
            [batch_size, num_heads, q_block_len, 1],
            f32::NEG_INFINITY,
            &device,
        );
        let mut l_i =
            Tensor::<B, 4>::zeros([batch_size, num_heads, q_block_len, 1], &device);
        let mut o_i =
            Tensor::<B, 4>::zeros([batch_size, num_heads, q_block_len, head_dim], &device);

        let mut kv_start = 0usize;
        while kv_start < key_len {
            let kv_end = (kv_start + block_kv).min(key_len);
            let kv_block_len = kv_end - kv_start;

            let k_block = k.clone().slice([
                0..batch_size,
                0..num_heads,
                kv_start..kv_end,
                0..head_dim,
            ]);
            let v_block = v.clone().slice([
                0..batch_size,
                0..num_heads,
                kv_start..kv_end,
                0..head_dim,
            ]);

            let mut scores = q_block.clone().matmul(k_block.transpose()) / scale;

            if causal {
                let mask = build_block_causal_mask(
                    &device,
                    q_block_len,
                    kv_block_len,
                    q_start,
                    kv_start,
                    position_offset,
                    window,
                );
                scores = scores + mask;
            }

            let m_ij = scores.clone().max_dim(3);
            let m_new = m_i.clone().max_pair(m_ij);
            let p_ij = (scores - m_new.clone()).exp();
            let m_scale = (m_i.clone() - m_new.clone()).exp();

            let l_new = m_scale.clone() * l_i + p_ij.clone().sum_dim(3);
            let o_new = m_scale * o_i + p_ij.matmul(v_block);

            m_i = m_new;
            l_i = l_new;
            o_i = o_new;
            kv_start = kv_end;
        }

        outputs.push(o_i / l_i);
        q_start = q_end;
    }

    Tensor::cat(outputs, 2)
}

fn build_block_causal_mask<B: Backend>(
    device: &B::Device,
    query_len: usize,
    key_len: usize,
    q_start: usize,
    kv_start: usize,
    position_offset: usize,
    window: usize,
) -> Tensor<B, 4> {
    let mut data = Vec::with_capacity(query_len * key_len);
    let mask_value = -1.0e4_f32;

    for i in 0..query_len {
        let absolute_pos = position_offset + q_start + i;
        let start = if window > 0 {
            absolute_pos.saturating_sub(window.saturating_sub(1))
        } else {
            0
        };
        for j in 0..key_len {
            let absolute_key = kv_start + j;
            let allowed = absolute_key <= absolute_pos && absolute_key >= start;
            data.push(if allowed { 0.0 } else { mask_value });
        }
    }

    Tensor::<B, 2>::from_data(TensorData::new(data, [query_len, key_len]), device)
        .reshape([1, 1, query_len, key_len])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::tensor::activation::softmax;

    fn build_full_causal_mask<B: Backend>(
        device: &B::Device,
        query_len: usize,
        key_len: usize,
        position_offset: usize,
        sliding_window: Option<usize>,
    ) -> Tensor<B, 4> {
        let window = sliding_window.unwrap_or(key_len);
        build_block_causal_mask(
            device,
            query_len,
            key_len,
            0,
            0,
            position_offset,
            window,
        )
    }

    #[test]
    fn test_flash_attention_matches_standard() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 5;
        let head_dim = 4;
        let total = batch_size * num_heads * seq_len * head_dim;

        let q_data: Vec<f32> = (0..total)
            .map(|i| (i as f32 % 11.0) * 0.05)
            .collect();
        let k_data: Vec<f32> = (0..total)
            .map(|i| ((i + 3) as f32 % 13.0) * 0.04)
            .collect();
        let v_data: Vec<f32> = (0..total)
            .map(|i| ((i + 7) as f32 % 17.0) * 0.03)
            .collect();

        let q = Tensor::<NdArray<f32>, 4>::from_data(
            TensorData::new(q_data, [batch_size, num_heads, seq_len, head_dim]),
            &device,
        );
        let k = Tensor::<NdArray<f32>, 4>::from_data(
            TensorData::new(k_data, [batch_size, num_heads, seq_len, head_dim]),
            &device,
        );
        let v = Tensor::<NdArray<f32>, 4>::from_data(
            TensorData::new(v_data, [batch_size, num_heads, seq_len, head_dim]),
            &device,
        );

        let config = FlashAttentionConfig {
            block_q: 2,
            block_kv: 3,
        };
        let output = flash_attention_forward(q.clone(), k.clone(), v.clone(), false, 0, seq_len, None, config);

        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(k.transpose()) / scale;
        let attn = softmax(scores, 3);
        let expected = attn.matmul(v);

        let output_data = output
            .into_data()
            .into_vec::<f32>()
            .expect("flash attention output should be float data");
        let expected_data = expected
            .into_data()
            .into_vec::<f32>()
            .expect("reference attention output should be float data");

        for (idx, (left, right)) in output_data.iter().zip(expected_data.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff < 1e-3,
                "mismatch at index {idx}: {left} vs {right} (diff {diff})"
            );
        }
    }

    #[test]
    fn test_flash_attention_causal_mask() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let batch_size = 1;
        let num_heads = 1;
        let seq_len = 4;
        let head_dim = 3;
        let total = batch_size * num_heads * seq_len * head_dim;

        let q_data: Vec<f32> = (0..total)
            .map(|i| (i as f32 % 7.0) * 0.07)
            .collect();
        let k_data: Vec<f32> = (0..total)
            .map(|i| ((i + 5) as f32 % 9.0) * 0.06)
            .collect();
        let v_data: Vec<f32> = (0..total)
            .map(|i| ((i + 11) as f32 % 11.0) * 0.05)
            .collect();

        let q = Tensor::<NdArray<f32>, 4>::from_data(
            TensorData::new(q_data, [batch_size, num_heads, seq_len, head_dim]),
            &device,
        );
        let k = Tensor::<NdArray<f32>, 4>::from_data(
            TensorData::new(k_data, [batch_size, num_heads, seq_len, head_dim]),
            &device,
        );
        let v = Tensor::<NdArray<f32>, 4>::from_data(
            TensorData::new(v_data, [batch_size, num_heads, seq_len, head_dim]),
            &device,
        );

        let config = FlashAttentionConfig {
            block_q: 2,
            block_kv: 2,
        };
        let output = flash_attention_forward(q.clone(), k.clone(), v.clone(), true, 0, seq_len, None, config);

        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(k.transpose()) / scale;
        let mask = build_full_causal_mask(&device, seq_len, seq_len, 0, None);
        let attn = softmax(scores + mask, 3);
        let expected = attn.matmul(v);

        let output_data = output
            .into_data()
            .into_vec::<f32>()
            .expect("flash attention output should be float data");
        let expected_data = expected
            .into_data()
            .into_vec::<f32>()
            .expect("reference attention output should be float data");

        for (idx, (left, right)) in output_data.iter().zip(expected_data.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff < 1e-3,
                "mismatch at index {idx}: {left} vs {right} (diff {diff})"
            );
        }
    }
}
