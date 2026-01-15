use crate::model_config::ModelConfig;
use burn::nn::{RmsNorm as BurnRmsNorm, RmsNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Clone)]
pub struct RmsNorm<B: Backend> {
    pub(crate) inner: BurnRmsNorm<B>,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        let eps = config.rms_norm_eps.unwrap_or(1e-6);
        let inner = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(eps)
            .init(device);
        Self { inner }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.inner.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn rms_norm_preserves_shape() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut config = ModelConfig::default();
        config.hidden_size = 4;

        let layer = RmsNorm::<NdArray<f32>>::new(&device, &config);
        let input = Tensor::<NdArray<f32>, 3>::zeros([2, 3, 4], &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [2, 3, 4]);
    }
}
