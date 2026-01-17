//! RMS Normalization using gllm-kernels.

use crate::model_config::ModelConfig;
use gllm_kernels::{rms_norm_forward, WeightVector};

#[derive(Clone)]
pub struct RmsNorm {
    pub gamma: WeightVector,
    hidden_size: usize,
    eps: f32,
}

impl RmsNorm {
    pub fn new(config: &ModelConfig) -> Self {
        let hidden_size = config.hidden_size;
        let eps = config.rms_norm_eps.unwrap_or(1e-5) as f32;
        let gamma = WeightVector::ones(hidden_size);
        Self {
            gamma,
            hidden_size,
            eps,
        }
    }

    pub fn with_weight(gamma: WeightVector, hidden_size: usize, eps: f32) -> Self {
        Self {
            gamma,
            hidden_size,
            eps,
        }
    }

    /// Forward pass for 2D input [batch, hidden].
    pub fn forward_2d(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * self.hidden_size];
        rms_norm_forward(
            input,
            self.gamma.as_slice(),
            &mut output,
            batch,
            self.hidden_size,
            self.eps,
        );
        output
    }

    /// Forward pass for 3D input [batch, seq_len, hidden].
    pub fn forward_3d(&self, input: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let rows = batch * seq_len;
        let mut output = vec![0.0f32; rows * self.hidden_size];
        rms_norm_forward(
            input,
            self.gamma.as_slice(),
            &mut output,
            rows,
            self.hidden_size,
            self.eps,
        );
        output
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }
}
