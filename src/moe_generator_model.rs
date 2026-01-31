//! MoE Generator Model - Pure L3 API wrapper for gllm-kernels.
//!
//! This model is a thin wrapper that:
//! 1. Stores weights in gllm-kernels compatible format
//! 2. Manages KVCache as contiguous memory
//! 3. Delegates all computation to `backend.moe_generator_forward()`

use crate::engine::TokenizerAdapter;
use crate::generation::{FinishReason, GenerationConfig, GenerationOptions, GenerationOutput};
use crate::model_config::ModelConfig;
use crate::registry::Architecture;
use crate::types::{Error, Result};
use gllm_kernels::backend::{Backend, BackendImpl};
use gllm_kernels::SamplingConfig;
use gllm_kernels::{
    Activation, ExpertWeights as KernelExpertWeights, KVCacheState,
    MoEGeneratorForwardConfig, MoETransformerLayerWeights,
};
use gllm_kernels::kernel_types::LogitsTensor;

mod loader;

/// MoE layer weights storage (no forward logic - computation delegated to L3 API).
pub struct MoELayerWeights {
    /// Input RMS norm weights [hidden_size]
    pub input_norm: Vec<f32>,
    /// Q projection weights [num_q_heads * head_dim, hidden_size]
    pub q_weight: Vec<f32>,
    /// K projection weights [num_kv_heads * head_dim, hidden_size]
    pub k_weight: Vec<f32>,
    /// V projection weights [num_kv_heads * head_dim, hidden_size]
    pub v_weight: Vec<f32>,
    /// O projection weights [hidden_size, num_q_heads * head_dim]
    pub o_weight: Vec<f32>,
    /// Post-attention RMS norm weights [hidden_size]
    pub post_attn_norm: Vec<f32>,
    /// Router gate weights [num_experts, hidden_size]
    pub router_weight: Vec<f32>,
    /// Expert FFN weights: Vec of (gate, up, down) for each expert
    pub experts: Vec<MoEExpertWeights>,
    /// Number of experts to activate per token
    pub num_experts_per_tok: usize,
}

/// Expert FFN weights storage.
pub struct MoEExpertWeights {
    /// Gate projection [intermediate_size, hidden_size]
    pub gate: Vec<f32>,
    /// Up projection [intermediate_size, hidden_size]
    pub up: Vec<f32>,
    /// Down projection [hidden_size, intermediate_size]
    pub down: Vec<f32>,
}

impl MoELayerWeights {
    pub fn new(config: &ModelConfig) -> Self {
        let hidden_size = config.hidden_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads.unwrap_or(num_q_heads);
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_q_heads);
        let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);
        let num_experts = config.num_experts.unwrap_or(8);
        let num_experts_per_tok = config.num_experts_per_tok.unwrap_or(2);

        // For MoE models, experts use moe_intermediate_size which is typically smaller
        let expert_intermediate_size = config.moe_intermediate_size.unwrap_or(intermediate_size);

        let experts = (0..num_experts)
            .map(|_| MoEExpertWeights {
                gate: vec![0.0; expert_intermediate_size * hidden_size],
                up: vec![0.0; expert_intermediate_size * hidden_size],
                down: vec![0.0; hidden_size * expert_intermediate_size],
            })
            .collect();

        Self {
            input_norm: vec![0.0; hidden_size],
            q_weight: vec![0.0; num_q_heads * head_dim * hidden_size],
            k_weight: vec![0.0; num_kv_heads * head_dim * hidden_size],
            v_weight: vec![0.0; num_kv_heads * head_dim * hidden_size],
            o_weight: vec![0.0; hidden_size * num_q_heads * head_dim],
            post_attn_norm: vec![0.0; hidden_size],
            router_weight: vec![0.0; num_experts * hidden_size],
            experts,
            num_experts_per_tok,
        }
    }

    /// Convert to gllm-kernels L3 API weight references.
    pub fn as_kernel_weights(&self) -> MoETransformerLayerWeights<'_> {
        let experts: Vec<KernelExpertWeights<'_>> = self
            .experts
            .iter()
            .map(|e| KernelExpertWeights {
                gate: &e.gate,
                up: &e.up,
                down: &e.down,
            })
            .collect();

        MoETransformerLayerWeights {
            input_norm: &self.input_norm,
            q_weight: &self.q_weight,
            k_weight: &self.k_weight,
            v_weight: &self.v_weight,
            o_weight: &self.o_weight,
            post_attn_norm: &self.post_attn_norm,
            router_weight: &self.router_weight,
            experts,
            num_experts_per_tok: self.num_experts_per_tok,
            q_norm: None,
            k_norm: None,
        }
    }
}

/// Contiguous KV cache for L3 API compatibility.
pub struct MoEKVCache {
    /// Key cache [num_layers * num_kv_heads * max_len * head_dim] - contiguous
    k_cache: Vec<f32>,
    /// Value cache [num_layers * num_kv_heads * max_len * head_dim] - contiguous
    v_cache: Vec<f32>,
    /// Current sequence length
    seq_len: usize,
    /// Maximum cache length
    max_len: usize,
    /// Number of layers (kept for documentation)
    #[allow(dead_code)]
    num_layers: usize,
    /// Number of KV heads (kept for documentation)
    #[allow(dead_code)]
    num_kv_heads: usize,
    /// Head dimension (kept for documentation)
    #[allow(dead_code)]
    head_dim: usize,
}

impl MoEKVCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_len: usize,
    ) -> Self {
        let cache_size = num_layers * num_kv_heads * max_len * head_dim;
        Self {
            k_cache: vec![0.0; cache_size],
            v_cache: vec![0.0; cache_size],
            seq_len: 0,
            max_len,
            num_layers,
            num_kv_heads,
            head_dim,
        }
    }

    /// Get mutable KVCacheState for L3 API.
    pub fn as_cache_state(&mut self) -> KVCacheState<'_> {
        KVCacheState {
            k_cache: &mut self.k_cache,
            v_cache: &mut self.v_cache,
            seq_len: self.seq_len,
            max_len: self.max_len,
        }
    }

    /// Update sequence length after forward pass.
    pub fn advance(&mut self, new_tokens: usize) {
        self.seq_len += new_tokens;
    }

    /// Get current sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Reset cache.
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Get max length.
    pub fn max_len(&self) -> usize {
        self.max_len
    }
}

/// MoE Generator Model - L3 API wrapper.
///
/// All computation delegated to `backend.moe_generator_forward()`.
pub struct MoEGeneratorModel {
    /// Embedding weights [vocab_size, hidden_size]
    embedding: Vec<f32>,
    /// Layer weights
    layers: Vec<MoELayerWeights>,
    /// Final RMS norm weights [hidden_size]
    final_norm: Vec<f32>,
    /// LM head weights [vocab_size, hidden_size]
    lm_head: Vec<f32>,
    /// RoPE cos cache
    cos_cache: Vec<f32>,
    /// RoPE sin cache
    sin_cache: Vec<f32>,
    /// Model configuration
    config: ModelConfig,
    /// Backend for computation
    backend: BackendImpl,
    /// Architecture type (SSOT)
    pub architecture: Architecture,
}

impl MoEGeneratorModel {
    pub fn new(config: ModelConfig, backend: BackendImpl, architecture: Architecture) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let vocab_size = config.vocab_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads.unwrap_or(num_q_heads);
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_q_heads);
        let max_position = config.max_position_embeddings;

        // Build layers
        let layers: Vec<MoELayerWeights> = (0..num_layers)
            .map(|_| MoELayerWeights::new(&config))
            .collect();

        // Build RoPE cache
        let (cos_cache, sin_cache) = Self::build_rope_cache(&config, head_dim, max_position);

        Ok(Self {
            embedding: vec![0.0; vocab_size * hidden_size],
            layers,
            final_norm: vec![0.0; hidden_size],
            lm_head: vec![0.0; vocab_size * hidden_size],
            cos_cache,
            sin_cache,
            config,
            backend,
            architecture,
        })
    }

    fn build_rope_cache(config: &ModelConfig, head_dim: usize, max_position: usize) -> (Vec<f32>, Vec<f32>) {
        let theta = config.rope_theta.unwrap_or(10000.0) as f32;
        let half_dim = head_dim / 2;
        // Cache size is max_position * half_dim (not head_dim!)
        // This matches the indexing in rope_apply which uses pos * half_dim + i
        let mut cos_cache = vec![0.0f32; max_position * half_dim];
        let mut sin_cache = vec![0.0f32; max_position * half_dim];

        for pos in 0..max_position {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                cos_cache[pos * half_dim + i] = cos_val;
                sin_cache[pos * half_dim + i] = sin_val;
            }
        }

        (cos_cache, sin_cache)
    }

    /// Create a new KV cache for this model.
    pub fn create_kv_cache(&self) -> MoEKVCache {
        let num_kv_heads = self.config.num_key_value_heads.unwrap_or(self.config.num_attention_heads);
        let head_dim = self.config.head_dim.unwrap_or(
            self.config.hidden_size / self.config.num_attention_heads
        );
        MoEKVCache::new(
            self.config.num_hidden_layers,
            num_kv_heads,
            head_dim,
            self.config.max_position_embeddings,
        )
    }

    /// Forward pass using L3 API.
    pub fn forward(&self, input_ids: &[u32], cache: &mut MoEKVCache) -> Result<LogitsTensor> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(Error::InvalidConfig("Input IDs cannot be empty".into()));
        }

        // Build layer weights for L3 API
        let layer_weights: Vec<MoETransformerLayerWeights<'_>> = self
            .layers
            .iter()
            .map(|l| l.as_kernel_weights())
            .collect();

        // Build config for L3 API
        let forward_config = MoEGeneratorForwardConfig {
            batch_size: 1,
            seq_len,
            hidden_size: self.config.hidden_size,
            num_layers: self.config.num_hidden_layers,
            num_q_heads: self.config.num_attention_heads,
            num_kv_heads: self.config.num_key_value_heads.unwrap_or(self.config.num_attention_heads),
            head_dim: self.config.head_dim.unwrap_or(
                self.config.hidden_size / self.config.num_attention_heads
            ),
            intermediate_size: self.config.intermediate_size.unwrap_or(self.config.hidden_size * 4),
            moe_intermediate_size: self.config.moe_intermediate_size,
            vocab_size: self.config.vocab_size,
            num_experts: self.config.num_experts.unwrap_or(8),
            num_experts_per_tok: self.config.num_experts_per_tok.unwrap_or(2),
            max_seq_len: self.config.max_position_embeddings,
            rms_norm_eps: self.config.rms_norm_eps.unwrap_or(1e-6) as f32,
            rope_theta: self.config.rope_theta.unwrap_or(10000.0) as f32,
            use_rope: true,
            activation: Activation::SiLU,
            position_offset: cache.seq_len(),
        };

        // Get mutable cache state
        let mut cache_state = cache.as_cache_state();

        // Call L3 API - ALL computation happens in gllm-kernels
        let logits = self.backend.moe_generator_forward(
            input_ids,
            &self.embedding,
            &layer_weights,
            &self.final_norm,
            &self.lm_head,
            &self.cos_cache,
            &self.sin_cache,
            &mut cache_state,
            &forward_config,
        ).map_err(|e| Error::InferenceError(e))?;

        // Update cache sequence length
        cache.advance(seq_len);

        Ok(logits)
    }

    /// Sample next token from logits.
    pub fn sample(&self, logits: &LogitsTensor, temperature: f32) -> Result<u32> {
        let vocab_size = self.config.vocab_size;
        let config = SamplingConfig {
            temperature,
            top_p: 0.0,
            top_k: 0,
            seed: None,
        };

        let result = self.backend.sample_from_tensor(logits, vocab_size, &config)
            .map_err(|e| Error::InferenceError(e))?;
        Ok(result[0])
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Access embedding weights for loading.
    pub fn embedding_mut(&mut self) -> &mut [f32] {
        &mut self.embedding
    }

    /// Access lm_head weights for loading.
    pub fn lm_head_mut(&mut self) -> &mut [f32] {
        &mut self.lm_head
    }

    /// Access layer for weight loading.
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut MoELayerWeights> {
        self.layers.get_mut(idx)
    }

    /// Access final norm for weight loading.
    pub fn final_norm_mut(&mut self) -> &mut [f32] {
        &mut self.final_norm
    }

    /// Access cos cache for weight loading.
    pub fn cos_cache_mut(&mut self) -> &mut [f32] {
        &mut self.cos_cache
    }

    /// Access sin cache for weight loading.
    pub fn sin_cache_mut(&mut self) -> &mut [f32] {
        &mut self.sin_cache
    }

    /// Generate text from prompt tokens.
    pub(crate) fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
        _options: &GenerationOptions,
    ) -> Result<GenerationOutput> {
        // Convert i64 tokens to u32
        let prompt_u32: Vec<u32> = prompt_ids
            .iter()
            .map(|&id| {
                if id < 0 || id > u32::MAX as i64 {
                    Err(Error::InvalidConfig(format!("Token ID {} out of u32 range", id)))
                } else {
                    Ok(id as u32)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Create KV cache
        let mut cache = self.create_kv_cache();

        // Process prompt - get logits for last token
        let mut logits = self.forward(&prompt_u32, &mut cache)?;

        // Generation parameters
        let max_new_tokens = config.max_new_tokens;
        let temperature = config.temperature;

        // Convert stop tokens to u32 set for fast lookup
        let stop_tokens: std::collections::HashSet<u32> = config
            .stop_tokens
            .iter()
            .filter_map(|&t| if t >= 0 && t <= u32::MAX as i64 { Some(t as u32) } else { None })
            .collect();

        // Generated tokens (including prompt)
        let mut all_tokens: Vec<i64> = prompt_ids.clone();
        let mut finish_reason = FinishReason::MaxTokens;

        // Generate tokens
        for _ in 0..max_new_tokens {
            // Check if we're at max context length
            if cache.seq_len() >= self.config.max_position_embeddings {
                finish_reason = FinishReason::MaxTokens;
                break;
            }

            // Sample next token from last position's logits
            let next_token = self.sample(&logits, temperature)?;

            // Check for stop token
            if stop_tokens.contains(&next_token) {
                finish_reason = FinishReason::StopToken;
                break;
            }

            // Add to generated tokens
            all_tokens.push(next_token as i64);

            // Forward single token to get next logits
            logits = self.forward(&[next_token], &mut cache)?;
        }

        // Get generated tokens (excluding prompt)
        let generated_tokens: Vec<i64> = all_tokens[prompt_ids.len()..].to_vec();

        // Decode generated tokens to text
        let text = tokenizer.decode(&generated_tokens);

        Ok(GenerationOutput {
            text,
            tokens: generated_tokens,
            finish_reason,
        })
    }
}
