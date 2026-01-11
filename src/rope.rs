//! Rotary Position Embedding (RoPE) implementation.
//!
//! RoPE encodes position information directly into the attention mechanism
//! by rotating query and key vectors based on their position in the sequence.
//!
//! Reference: https://arxiv.org/abs/2104.09864

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for RoPE.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Base frequency for position encoding (default: 10000.0).
    pub theta: f64,
    /// Hidden dimension (must be divisible by 2).
    pub dim: usize,
    /// Maximum sequence length for precomputed frequencies.
    pub max_seq_len: usize,
    /// Optional NTK scaling factor.
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

/// Rotary Position Embedding module.
#[derive(Clone)]
pub struct RotaryPositionEmbedding<B: Backend> {
    /// Precomputed cosine values: [max_seq_len, dim/2]
    cos_cached: Tensor<B, 2>,
    /// Precomputed sine values: [max_seq_len, dim/2]
    sin_cached: Tensor<B, 2>,
    max_cached_len: usize,
    dim: usize,
}

impl<B: Backend> RotaryPositionEmbedding<B> {
    /// Create a new RoPE module with precomputed frequency tables.
    pub fn new(device: &B::Device, config: RopeConfig) -> Self {
        let dim = config.dim;
        let max_seq_len = config.max_seq_len;
        let half_dim = dim / 2;

        // Apply NTK scaling if specified
        let theta = if let Some(factor) = config.ntk_factor {
            config.theta * factor
        } else {
            config.theta
        };

        // Compute inverse frequencies: theta^(-2i/dim) for i in 0..dim/2
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exponent = -2.0 * (i as f64) / (dim as f64);
                (theta.powf(exponent)) as f32
            })
            .collect();

        // Compute position indices: [0, 1, 2, ..., max_seq_len-1]
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();

        // Compute frequencies: positions * inv_freq (outer product)
        // Result shape: [max_seq_len, dim/2]
        let mut freqs = Vec::with_capacity(max_seq_len * half_dim);
        for pos in &positions {
            for inv_f in &inv_freq {
                freqs.push(pos * inv_f);
            }
        }

        // Compute cos and sin tables
        let cos_vals: Vec<f32> = freqs.iter().map(|f| f.cos()).collect();
        let sin_vals: Vec<f32> = freqs.iter().map(|f| f.sin()).collect();

        let cos_cached = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(cos_vals, [max_seq_len, half_dim]),
            device,
        );
        let sin_cached = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(sin_vals, [max_seq_len, half_dim]),
            device,
        );

        Self {
            cos_cached,
            sin_cached,
            max_cached_len: max_seq_len,
            dim,
        }
    }

    /// Apply rotary position embedding to query and key tensors.
    ///
    /// Input shapes:
    /// - query: [batch, seq_len, num_heads, head_dim]
    /// - key: [batch, seq_len, num_heads, head_dim]
    ///
    /// Returns rotated (query, key) with the same shapes.
    pub fn apply(
        &self,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        position_offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch, seq_len, _num_heads, head_dim] = query.dims();
        let half_dim = head_dim / 2;
        debug_assert!(position_offset + seq_len <= self.max_cached_len);

        // Get cos/sin for current positions
        let cos = self.cos_cached
            .clone()
            .slice([position_offset..(position_offset + seq_len), 0..half_dim])
            .reshape([1, seq_len, 1, half_dim]);
        let sin = self.sin_cached
            .clone()
            .slice([position_offset..(position_offset + seq_len), 0..half_dim])
            .reshape([1, seq_len, 1, half_dim]);

        // Apply rotation
        let q_rotated = self.rotate_half(query, cos.clone(), sin.clone());
        let k_rotated = self.rotate_half(key, cos, sin);

        (q_rotated, k_rotated)
    }

    /// Apply rotary embedding to a single tensor (for simplified use).
    ///
    /// Input shape: [batch, seq_len, hidden_size]
    /// Returns: [batch, seq_len, hidden_size]
    pub fn apply_to_hidden_states(
        &self,
        hidden_states: Tensor<B, 3>,
        position_offset: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden_size] = hidden_states.dims();
        let half_dim = self.dim / 2;
        debug_assert!(position_offset + seq_len <= self.max_cached_len);

        // Get cos/sin for current positions
        let cos = self.cos_cached
            .clone()
            .slice([position_offset..(position_offset + seq_len), 0..half_dim])
            .reshape([1, seq_len, half_dim]);
        let sin = self.sin_cached
            .clone()
            .slice([position_offset..(position_offset + seq_len), 0..half_dim])
            .reshape([1, seq_len, half_dim]);

        // Split hidden states into first and second half
        let x1 = hidden_states.clone().slice([0..batch, 0..seq_len, 0..half_dim]);
        let x2 = hidden_states.clone().slice([0..batch, 0..seq_len, half_dim..self.dim]);

        // Apply rotation: x1' = x1*cos - x2*sin, x2' = x2*cos + x1*sin
        let x1_rotated = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let x2_rotated = x2 * cos + x1 * sin;

        // Concatenate and return
        Tensor::cat(vec![x1_rotated, x2_rotated], 2)
    }

    /// Rotate half of the tensor dimensions.
    fn rotate_half(
        &self,
        x: Tensor<B, 4>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, seq_len, num_heads, head_dim] = x.dims();
        let half_dim = head_dim / 2;

        // Split into first and second half
        let x1 = x.clone().slice([0..batch, 0..seq_len, 0..num_heads, 0..half_dim]);
        let x2 = x.clone().slice([0..batch, 0..seq_len, 0..num_heads, half_dim..head_dim]);

        // Apply rotation: x1' = x1*cos - x2*sin, x2' = x2*cos + x1*sin
        let x1_rotated = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let x2_rotated = x2 * cos + x1 * sin;

        // Concatenate back
        Tensor::cat(vec![x1_rotated, x2_rotated], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    #[test]
    fn rope_creates_frequency_tables() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let config = RopeConfig {
            theta: 10000.0,
            dim: 64,
            max_seq_len: 512,
            ntk_factor: None,
        };

        let rope = RotaryPositionEmbedding::<NdArray<f32>>::new(&device, config);

        assert_eq!(rope.cos_cached.dims(), [512, 32]);
        assert_eq!(rope.sin_cached.dims(), [512, 32]);
    }

    #[test]
    fn rope_apply_preserves_shape() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let config = RopeConfig {
            theta: 10000.0,
            dim: 64,
            max_seq_len: 512,
            ntk_factor: None,
        };

        let rope = RotaryPositionEmbedding::<NdArray<f32>>::new(&device, config);

        // Create test tensors: [batch=2, seq_len=10, num_heads=8, head_dim=64]
        let query = Tensor::<NdArray<f32>, 4>::zeros([2, 10, 8, 64], &device);
        let key = Tensor::<NdArray<f32>, 4>::zeros([2, 10, 8, 64], &device);

        let (q_rot, k_rot) = rope.apply(query, key, 0);

        assert_eq!(q_rot.dims(), [2, 10, 8, 64]);
        assert_eq!(k_rot.dims(), [2, 10, 8, 64]);
    }
}
