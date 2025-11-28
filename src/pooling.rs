use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::linalg::vector_normalize;

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
pub struct DynamicPooler<B: Backend> {
    strategy: PoolingStrategy,
    config: PoolingConfig,
    _marker: core::marker::PhantomData<B>,
}

impl<B: Backend> DynamicPooler<B> {
    pub fn new(strategy: PoolingStrategy, config: PoolingConfig) -> Self {
        Self {
            strategy,
            config,
            _marker: core::marker::PhantomData,
        }
    }

    /// Pool hidden states; attention mask is optional and currently used only for mean strategies.
    pub fn pool(
        &self,
        hidden_states: Tensor<B, 3>,
        _attention_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let pooled = match self.strategy {
            PoolingStrategy::Cls => hidden_states
                .slice([0..batch_size, 0..1, 0..hidden_size])
                .reshape([batch_size, hidden_size]),
            PoolingStrategy::Mean | PoolingStrategy::WeightedMean => {
                hidden_states
                    .mean_dim(1)
                    .reshape([batch_size, hidden_size])
            }
            PoolingStrategy::Max => hidden_states
                .max_dim(1)
                .reshape([batch_size, hidden_size]),
            PoolingStrategy::LastToken => hidden_states
                .slice([0..batch_size, (seq_len - 1)..seq_len, 0..hidden_size])
                .reshape([batch_size, hidden_size]),
        };

        match self.config.normalize {
            true => vector_normalize(pooled, 2.0, 1, 1e-6),
            false => pooled,
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
        use super::*;
        use burn::backend::ndarray::NdArray;

    #[test]
    fn mean_pooling_reduces_dim() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let tensor =
            Tensor::<NdArray<f32>, 3>::from_data([[[1.0f32, 2.0, 3.0], [3.0, 4.0, 5.0]]], &device);
        let pooler = DynamicPooler::new(PoolingStrategy::Mean, PoolingConfig { normalize: false });
        let pooled = pooler.pool(tensor, None);
        assert_eq!(pooled.dims(), [1, 3]);
    }
}
