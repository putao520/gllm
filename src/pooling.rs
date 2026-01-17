use crate::tensor::{Matrix, Tensor3};

/// Pooling strategy used for sequence representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    Cls,
    Mean,
    Max,
    WeightedMean,
    LastToken,
}

/// Pooling configuration.
#[derive(Debug, Clone, Copy)]
pub struct PoolingConfig {
    /// Whether to L2-normalize pooled vectors.
    pub normalize: bool,
}

impl Default for PoolingConfig {
    fn default() -> Self {
        Self { normalize: true }
    }
}

/// Simple dynamic pooler supporting several strategies.
#[derive(Clone)]
pub struct DynamicPooler {
    strategy: PoolingStrategy,
    config: PoolingConfig,
}

impl DynamicPooler {
    pub fn new(strategy: PoolingStrategy, config: PoolingConfig) -> Self {
        Self { strategy, config }
    }

    /// Pool hidden states; attention mask is optional and used for mean/last strategies.
    pub fn pool(&self, hidden_states: &Tensor3, attention_mask: Option<&[i64]>) -> Matrix {
        let batch_size = hidden_states.dim0;
        let seq_len = hidden_states.dim1;
        let hidden_size = hidden_states.dim2;
        let mut output = Matrix::zeros(batch_size, hidden_size);

        for b in 0..batch_size {
            let mut valid_count = 0usize;
            let mut last_idx = seq_len.saturating_sub(1);
            if let Some(mask) = attention_mask {
                for s in 0..seq_len {
                    let idx = b * seq_len + s;
                    if mask.get(idx).copied().unwrap_or(0) != 0 {
                        valid_count += 1;
                        last_idx = s;
                    }
                }
            } else {
                valid_count = seq_len.max(1);
            }

            let row = output.row_mut(b);
            match self.strategy {
                PoolingStrategy::Cls => {
                    row.copy_from_slice(hidden_states.slice(b, 0));
                }
                PoolingStrategy::LastToken => {
                    row.copy_from_slice(hidden_states.slice(b, last_idx));
                }
                PoolingStrategy::Mean | PoolingStrategy::WeightedMean => {
                    let denom = (valid_count as f32).max(1.0);
                    for s in 0..seq_len {
                        if let Some(mask) = attention_mask {
                            let idx = b * seq_len + s;
                            if mask.get(idx).copied().unwrap_or(0) == 0 {
                                continue;
                            }
                        }
                        let slice = hidden_states.slice(b, s);
                        for i in 0..hidden_size {
                            row[i] += slice[i];
                        }
                    }
                    for i in 0..hidden_size {
                        row[i] /= denom;
                    }
                }
                PoolingStrategy::Max => {
                    for i in 0..hidden_size {
                        row[i] = f32::NEG_INFINITY;
                    }
                    for s in 0..seq_len {
                        let slice = hidden_states.slice(b, s);
                        for i in 0..hidden_size {
                            row[i] = row[i].max(slice[i]);
                        }
                    }
                }
            }

            if self.config.normalize {
                let mut norm = 0.0f32;
                for v in row.iter() {
                    norm += v * v;
                }
                let denom = norm.sqrt().max(1e-6);
                for v in row.iter_mut() {
                    *v /= denom;
                }
            }
        }

        output
    }
}
