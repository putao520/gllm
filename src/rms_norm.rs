//! RMS Normalization using gllm-kernels ops.
//!
//! RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)

use gllm_kernels::{rms_norm_forward, rms_norm_inplace};

/// RMS Normalization layer weights.
#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
    pub hidden_size: usize,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; hidden_size],
            eps,
            hidden_size,
        }
    }

    pub fn with_weight(weight: Vec<f32>, eps: f32) -> Self {
        let hidden_size = weight.len();
        Self {
            weight,
            eps,
            hidden_size,
        }
    }

    /// Forward pass, returns normalized output.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let batch = input.len() / self.hidden_size;
        let mut output = vec![0.0f32; input.len()];
        rms_norm_forward(input, &self.weight, &mut output, batch, self.hidden_size, self.eps);
        output
    }

    /// In-place forward pass.
    pub fn forward_inplace(&self, data: &mut [f32]) {
        let batch = data.len() / self.hidden_size;
        rms_norm_inplace(data, &self.weight, batch, self.hidden_size, self.eps);
    }
}
