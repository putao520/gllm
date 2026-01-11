use crate::causal_attention::CausalAttention;
use crate::decoder_layer::DecoderLayer;
use crate::model_config::ModelConfig;
use crate::rms_norm::RmsNorm;
use crate::types::{Error, Result};
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use safetensors::SafeTensors;
use std::path::Path;

#[derive(Clone)]
pub struct DecoderModel<B: Backend> {
    embeddings: Embedding<B>,
    layers: Vec<DecoderLayer<B>>,
    final_norm: RmsNorm<B>,
    pad_token_id: i64,
    max_position_embeddings: usize,
    hidden_size: usize,
    device: B::Device,
}

impl<B: Backend> DecoderModel<B> {
    pub fn new(device: &B::Device, config: ModelConfig) -> Result<Self> {
        if config.num_hidden_layers == 0 {
            return Err(Error::InvalidConfig(
                "num_hidden_layers must be greater than 0 for decoder model".into(),
            ));
        }
        if config.vocab_size == 0 {
            return Err(Error::InvalidConfig(
                "vocab_size must be greater than 0 for decoder model".into(),
            ));
        }

        let embeddings = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let head_dim = config
            .head_dim
            .unwrap_or_else(|| config.hidden_size / config.num_attention_heads);
        let rope = CausalAttention::build_rope(device, &config, head_dim);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(device, &config, rope.clone())?);
        }

        let final_norm = RmsNorm::new(device, &config);
        let pad_token_id = config.pad_token_id.unwrap_or(0);

        Ok(Self {
            embeddings,
            layers,
            final_norm,
            pad_token_id,
            max_position_embeddings: config.max_position_embeddings,
            hidden_size: config.hidden_size,
            device: device.clone(),
        })
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>> {
        let [_batch_size, seq_len] = input_ids.dims();
        if seq_len == 0 {
            return Err(Error::InvalidConfig(
                "input sequence length must be greater than 0".into(),
            ));
        }
        if self.max_position_embeddings > 0 && seq_len > self.max_position_embeddings {
            return Err(Error::InvalidConfig(format!(
                "Sequence length {} exceeds configured maximum {}",
                seq_len, self.max_position_embeddings
            )));
        }

        let mut hidden_states = self.embeddings.forward(input_ids);
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, 0);
        }

        Ok(self.final_norm.forward(hidden_states))
    }

    pub fn pool_hidden_states(
        &self,
        hidden_states: Tensor<B, 3>,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 2>> {
        let indices = self.last_token_indices(&input_ids)?;
        let [batch_size, _seq_len, hidden_size] = hidden_states.dims();

        let mut gather_indices = Vec::with_capacity(batch_size * hidden_size);
        for index in indices {
            for _ in 0..hidden_size {
                gather_indices.push(index);
            }
        }

        let indices_tensor = Tensor::<B, 3, Int>::from_data(
            TensorData::new(gather_indices, [batch_size, 1, hidden_size]),
            &self.device,
        );

        Ok(hidden_states
            .gather(1, indices_tensor)
            .reshape([batch_size, hidden_size]))
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        let bytes = std::fs::read(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to read SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;

        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|err| Error::LoadError(format!("Invalid SafeTensors: {err}")))?;

        if tensors.len() == 0 {
            return Err(Error::LoadError(
                "SafeTensors file contains no tensors".into(),
            ));
        }

        Ok(())
    }

    fn last_token_indices(&self, input_ids: &Tensor<B, 2, Int>) -> Result<Vec<i64>> {
        let [batch_size, seq_len] = input_ids.dims();
        let data = input_ids
            .clone()
            .into_data()
            .into_vec::<i64>()
            .map_err(|err| Error::InferenceError(err.to_string()))?;

        let mut indices = Vec::with_capacity(batch_size);
        for batch in 0..batch_size {
            let start = batch * seq_len;
            let end = start + seq_len;
            let row = &data[start..end];

            let mut idx = seq_len.saturating_sub(1);
            while idx > 0 && row[idx] == self.pad_token_id {
                idx = idx.saturating_sub(1);
            }
            indices.push(idx as i64);
        }

        Ok(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    #[test]
    fn pool_last_token_respects_padding() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let mut config = ModelConfig::default();
        config.hidden_size = 4;
        config.num_hidden_layers = 1;
        config.num_attention_heads = 1;
        config.num_key_value_heads = Some(1);
        config.intermediate_size = Some(8);
        config.vocab_size = 16;
        config.max_position_embeddings = 8;
        config.position_embedding_type = Some("rope".to_string());
        config.rms_norm_eps = Some(1e-6);
        config.pad_token_id = Some(0);

        let model = DecoderModel::<NdArray<f32>>::new(&device, config).expect("model");

        let hidden_states = Tensor::<NdArray<f32>, 3>::from_data(
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 4.0],
                ],
                [
                    [10.0, 10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0, 20.0],
                    [30.0, 30.0, 30.0, 30.0],
                    [40.0, 40.0, 40.0, 40.0],
                ],
            ],
            &device,
        );

        let input_ids = Tensor::<NdArray<f32>, 2, Int>::from_data(
            [[5i64, 6, 0, 0], [7, 8, 9, 0]],
            &device,
        );

        let pooled = model
            .pool_hidden_states(hidden_states, input_ids)
            .expect("pool");
        let data = pooled
            .into_data()
            .into_vec::<f32>()
            .expect("pooled data");

        assert_eq!(data, vec![2.0, 2.0, 2.0, 2.0, 30.0, 30.0, 30.0, 30.0]);
    }
}
