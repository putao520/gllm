use crate::causal_attention::CausalAttention;
use crate::engine::TokenizerAdapter;
use crate::generation::{FinishReason, GenerationConfig, GenerationOutput};
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::moe_decoder_layer::MoEDecoderLayer;
use crate::rms_norm::RmsNorm;
use crate::sampler::{sample_next_token, SamplingConfig};
use crate::types::{Error, Result};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use std::path::Path;

#[derive(Clone)]
pub struct MoEGeneratorModel<B: Backend> {
    pub(crate) embeddings: Embedding<B>,
    pub(crate) layers: Vec<MoEDecoderLayer<B>>,
    pub(crate) final_norm: RmsNorm<B>,
    pub(crate) lm_head: Linear<B>,
    pub(crate) pad_token_id: i64,
    pub(crate) max_position_embeddings: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) device: B::Device,
}

impl<B: Backend> MoEGeneratorModel<B> {
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
            layers.push(MoEDecoderLayer::new(device, &config, rope.clone())?);
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
        use crate::weight_loader::{load_linear, load_embedding, WeightLoader};
        use burn::module::Param;

        let bytes = std::fs::read(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to read SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;

        let loader = WeightLoader::from_bytes(&bytes)?;

        // Load embeddings
        let embed_names = [
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "transformer.embedding.word_embeddings.weight",
        ];
        for name in embed_names {
            if loader.has_tensor(name) {
                self.embeddings = load_embedding(&loader, name, &self.device)?;
                break;
            }
        }

        // Load MoE decoder layers
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", layer_idx);

            // Load attention weights
            if loader.has_tensor(&format!("{}.self_attn.q_proj.weight", prefix)) {
                layer.attention.q_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    Some(&format!("{}.self_attn.q_proj.bias", prefix)),
                    &self.device,
                )?;
                layer.attention.k_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    Some(&format!("{}.self_attn.k_proj.bias", prefix)),
                    &self.device,
                )?;
                layer.attention.v_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    Some(&format!("{}.self_attn.v_proj.bias", prefix)),
                    &self.device,
                )?;
                layer.attention.o_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    Some(&format!("{}.self_attn.o_proj.bias", prefix)),
                    &self.device,
                )?;
            }

            // Load MoE router (gate)
            if loader.has_tensor(&format!("{}.mlp.gate.weight", prefix)) {
                layer.moe.router.gate = load_linear(
                    &loader,
                    &format!("{}.mlp.gate.weight", prefix),
                    None,
                    &self.device,
                )?;
            }

            // Load MoE experts
            for (expert_idx, expert) in layer.moe.experts.iter_mut().enumerate() {
                let expert_prefix = format!("{}.mlp.experts.{}", prefix, expert_idx);
                if loader.has_tensor(&format!("{}.gate_proj.weight", expert_prefix)) {
                    expert.gate_proj = load_linear(
                        &loader,
                        &format!("{}.gate_proj.weight", expert_prefix),
                        None,
                        &self.device,
                    )?;
                    expert.up_proj = load_linear(
                        &loader,
                        &format!("{}.up_proj.weight", expert_prefix),
                        None,
                        &self.device,
                    )?;
                    expert.down_proj = load_linear(
                        &loader,
                        &format!("{}.down_proj.weight", expert_prefix),
                        None,
                        &self.device,
                    )?;
                }
            }

            // Load shared expert if present
            if let Some(shared) = &mut layer.moe.shared_expert {
                let shared_prefix = format!("{}.mlp.shared_expert", prefix);
                if loader.has_tensor(&format!("{}.gate_proj.weight", shared_prefix)) {
                    shared.gate_proj = load_linear(
                        &loader,
                        &format!("{}.gate_proj.weight", shared_prefix),
                        None,
                        &self.device,
                    )?;
                    shared.up_proj = load_linear(
                        &loader,
                        &format!("{}.up_proj.weight", shared_prefix),
                        None,
                        &self.device,
                    )?;
                    shared.down_proj = load_linear(
                        &loader,
                        &format!("{}.down_proj.weight", shared_prefix),
                        None,
                        &self.device,
                    )?;
                }
            }

            // Load RMSNorm weights
            if loader.has_tensor(&format!("{}.input_layernorm.weight", prefix)) {
                let norm_tensor = loader.load_tensor(&format!("{}.input_layernorm.weight", prefix))?;
                let norm_weight = norm_tensor.to_tensor::<B, 1>(&self.device, [norm_tensor.shape[0]])?;
                layer.attention_norm.inner.gamma = Param::from_tensor(norm_weight);
            }
            if loader.has_tensor(&format!("{}.post_attention_layernorm.weight", prefix)) {
                let norm_tensor = loader.load_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
                let norm_weight = norm_tensor.to_tensor::<B, 1>(&self.device, [norm_tensor.shape[0]])?;
                layer.ffn_norm.inner.gamma = Param::from_tensor(norm_weight);
            }
        }

        // Load final layer norm
        let final_norm_names = ["model.norm.weight", "transformer.ln_f.weight"];
        for name in final_norm_names {
            if loader.has_tensor(name) {
                let norm_tensor = loader.load_tensor(name)?;
                let norm_weight = norm_tensor.to_tensor::<B, 1>(&self.device, [norm_tensor.shape[0]])?;
                self.final_norm.inner.gamma = Param::from_tensor(norm_weight);
                break;
            }
        }

        // Load LM head
        let lm_head_names = ["lm_head.weight", "output.weight"];
        for name in lm_head_names {
            if loader.has_tensor(name) {
                self.lm_head = load_linear(&loader, name, None, &self.device)?;
                break;
            }
        }

        log::info!("Successfully loaded MoE weights from {}", safetensors_path.display());
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
