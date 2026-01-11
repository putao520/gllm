use crate::causal_attention::CausalAttention;
use crate::decoder_layer::DecoderLayer;
use crate::generation::{FinishReason, GenerationConfig, GenerationOutput};
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::sampler::{sample_next_token, SamplingConfig};
use crate::types::{Error, Result};
use crate::engine::TokenizerAdapter;
use crate::rms_norm::RmsNorm;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use safetensors::SafeTensors;
use std::path::Path;

#[derive(Clone)]
pub struct GeneratorModel<B: Backend> {
    embeddings: Embedding<B>,
    layers: Vec<DecoderLayer<B>>,
    final_norm: RmsNorm<B>,
    lm_head: Linear<B>,
    pad_token_id: i64,
    max_position_embeddings: usize,
    vocab_size: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    device: B::Device,
}

impl<B: Backend> GeneratorModel<B> {
    pub fn new(device: &B::Device, config: ModelConfig) -> Result<Self> {
        if config.num_hidden_layers == 0 {
            return Err(Error::InvalidConfig(
                "num_hidden_layers must be greater than 0 for generator model".into(),
            ));
        }
        if config.vocab_size == 0 {
            return Err(Error::InvalidConfig(
                "vocab_size must be greater than 0 for generator model".into(),
            ));
        }

        let embeddings = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
        let head_dim = config
            .head_dim
            .unwrap_or_else(|| config.hidden_size / num_attention_heads);
        let rope = CausalAttention::build_rope(device, &config, head_dim);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(device, &config, rope.clone())?);
        }

        let final_norm = RmsNorm::new(device, &config);
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size).init(device);

        Ok(Self {
            embeddings,
            layers,
            final_norm,
            lm_head,
            pad_token_id: config.pad_token_id.unwrap_or(0),
            max_position_embeddings: config.max_position_embeddings,
            vocab_size: config.vocab_size,
            num_key_value_heads,
            head_dim,
            device: device.clone(),
        })
    }

    pub fn forward_step(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: &mut KVCache<B>,
    ) -> Tensor<B, 2> {
        let [_batch_size, seq_len] = input_ids.dims();
        let position_offset = cache.seq_len();

        let mut hidden_states = self.embeddings.forward(input_ids);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states =
                layer.forward_with_cache(hidden_states, position_offset, cache, layer_idx);
        }

        let hidden_states = self.final_norm.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states);

        let [batch_size, _seq, _vocab] = logits.dims();
        let last_index = seq_len.saturating_sub(1);
        logits
            .slice([0..batch_size, last_index..(last_index + 1), 0..self.vocab_size])
            .reshape([batch_size, self.vocab_size])
    }

    pub fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
    ) -> Result<GenerationOutput> {
        if prompt_ids.is_empty() {
            return Err(Error::InvalidConfig(
                "Prompt tokens are required for generation".into(),
            ));
        }

        if self.max_position_embeddings > 0 && prompt_ids.len() > self.max_position_embeddings {
            return Err(Error::InvalidConfig(format!(
                "Prompt length {} exceeds max position {}",
                prompt_ids.len(),
                self.max_position_embeddings
            )));
        }

        let max_len = if self.max_position_embeddings > 0 {
            self.max_position_embeddings
        } else {
            prompt_ids.len().saturating_add(config.max_new_tokens)
        };
        let mut cache = KVCache::preallocate(
            self.layers.len(),
            max_len,
            1,
            self.num_key_value_heads,
            self.head_dim,
            &self.device,
        );
        let mut tokens = prompt_ids.clone();
        let sampling = SamplingConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
        };

        let mut finish_reason = FinishReason::MaxTokens;
        let mut input_ids = prompt_ids;
        let mut logits = self.forward_step(self.tokens_to_tensor(&input_ids), &mut cache);

        for _ in 0..config.max_new_tokens {
            if self.max_position_embeddings > 0
                && cache.seq_len() >= self.max_position_embeddings
            {
                finish_reason = FinishReason::MaxTokens;
                break;
            }

            let next_tokens = sample_next_token(logits, &sampling, &self.device);
            let next_token = next_tokens.first().copied().unwrap_or(self.pad_token_id);
            tokens.push(next_token);

            if config.stop_tokens.contains(&next_token) {
                finish_reason = FinishReason::StopToken;
                break;
            }

            input_ids = vec![next_token];
            logits = self.forward_step(self.tokens_to_tensor(&input_ids), &mut cache);
        }

        let text = tokenizer.decode(&tokens);
        Ok(GenerationOutput {
            text,
            tokens,
            finish_reason,
        })
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

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn tokens_to_tensor(&self, tokens: &[i64]) -> Tensor<B, 2, Int> {
        let mut data = tokens.to_vec();
        if data.is_empty() {
            data.push(self.pad_token_id);
        }
        let seq_len = data.len();
        let data = TensorData::new(data, [1, seq_len]);
        Tensor::<B, 2, Int>::from_data(data, &self.device)
    }
}
