use crate::model_config::ModelConfig;
use crate::types::{Error, Result};

/// Simple performance optimizer to guard sequence and batch sizes.
#[derive(Debug, Clone)]
pub struct PerformanceOptimizer {
    pub batch_size: usize,
    pub max_sequence_length: usize,
    pub memory_limit: usize,
    pub gpu_memory_fraction: f32,
}

impl PerformanceOptimizer {
    pub fn from_config(config: &ModelConfig) -> Self {
        Self {
            batch_size: config
                .max_batch_size
                .unwrap_or_else(|| std::cmp::max(1, config.num_attention_heads)),
            max_sequence_length: config.max_position_embeddings,
            memory_limit: config.memory_limit_mb.unwrap_or(512),
            gpu_memory_fraction: config.gpu_memory_fraction.unwrap_or(1.0),
        }
    }

    pub fn optimize_batch_size(&self, input_length: usize) -> usize {
        let scaled_limit = (self.memory_limit as f32 * self.gpu_memory_fraction.clamp(0.1, 1.0))
            .round()
            .max(1.0) as usize;
        let available = (scaled_limit / (input_length + 1)).max(1);
        available.min(self.batch_size).max(1)
    }

    #[allow(dead_code)]
    pub fn clamp_sequence_length(&self, seq_len: usize) -> usize {
        seq_len.min(self.max_sequence_length)
    }

    pub fn validate_sequence(&self, seq_len: usize) -> Result<()> {
        if seq_len > self.max_sequence_length {
            return Err(Error::InvalidConfig(format!(
                "Sequence length {} exceeds configured maximum {}",
                seq_len, self.max_sequence_length
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_sequence_len() {
        let mut config = ModelConfig::default();
        config.max_position_embeddings = 8;
        let optimizer = PerformanceOptimizer::from_config(&config);
        assert_eq!(optimizer.clamp_sequence_length(10), 8);
    }
}
