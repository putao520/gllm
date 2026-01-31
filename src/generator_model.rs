//! Generator Model - Pure L3 API wrapper for gllm-kernels.
//!
//! This model is a thin wrapper that:
//! 1. Stores weights in gllm-kernels compatible format
//! 2. Manages KVCache as contiguous memory
//! 3. Delegates all computation to `backend.generator_forward()`

use crate::engine::TokenizerAdapter;
use crate::generation::{FinishReason, GenerationConfig, GenerationOptions, GenerationOutput};
use crate::model_config::ModelConfig;
use crate::types::{Error, Result};
use gllm_kernels::backend::{Backend, BackendImpl};
use gllm_kernels::SamplingConfig;
use gllm_kernels::{Activation, GeneratorForwardConfig, KVCacheState, TransformerLayerWeights};
// GPU-native types for zero-copy generation (ARCH-PERF-001)
use gllm_kernels::kernel_types::{GeneratorModelWeightsGpu, KVCacheGpu, LogitsTensor};

mod loader;

/// Layer weights storage (no forward logic - computation delegated to L3 API).
pub struct LayerWeights {
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
    /// Gate projection weights (LLaMA-style) [intermediate_size, hidden_size]
    pub gate_weight: Option<Vec<f32>>,
    /// Up projection weights [intermediate_size, hidden_size]
    pub up_weight: Vec<f32>,
    /// Down projection weights [hidden_size, intermediate_size]
    pub down_weight: Vec<f32>,
    /// QK norm weights for Q projection (Qwen3-style) [num_q_heads * head_dim]
    pub q_norm: Option<Vec<f32>>,
    /// QK norm weights for K projection (Qwen3-style) [num_kv_heads * head_dim]
    pub k_norm: Option<Vec<f32>>,
}

impl LayerWeights {
    pub fn new(config: &ModelConfig) -> Self {
        let hidden_size = config.hidden_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads.unwrap_or(num_q_heads);
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_q_heads);
        let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);

        // Most LLaMA-style models use gate projection
        let use_gate = config.hidden_act.as_deref() != Some("gelu");

        Self {
            input_norm: vec![0.0; hidden_size],
            q_weight: vec![0.0; num_q_heads * head_dim * hidden_size],
            k_weight: vec![0.0; num_kv_heads * head_dim * hidden_size],
            v_weight: vec![0.0; num_kv_heads * head_dim * hidden_size],
            o_weight: vec![0.0; hidden_size * num_q_heads * head_dim],
            post_attn_norm: vec![0.0; hidden_size],
            gate_weight: if use_gate {
                Some(vec![0.0; intermediate_size * hidden_size])
            } else {
                None
            },
            up_weight: vec![0.0; intermediate_size * hidden_size],
            down_weight: vec![0.0; hidden_size * intermediate_size],
            q_norm: None,
            k_norm: None,
        }
    }

    /// Convert to gllm-kernels L3 API weight references.
    pub fn as_kernel_weights(&self) -> TransformerLayerWeights<'_> {
        TransformerLayerWeights {
            input_norm: &self.input_norm,
            q_weight: &self.q_weight,
            k_weight: &self.k_weight,
            v_weight: &self.v_weight,
            o_weight: &self.o_weight,
            post_attn_norm: &self.post_attn_norm,
            gate_weight: self.gate_weight.as_deref(),
            up_weight: &self.up_weight,
            down_weight: &self.down_weight,
            q_norm: self.q_norm.as_deref(),
            k_norm: self.k_norm.as_deref(),
        }
    }
}

/// Contiguous KV cache for L3 API compatibility.
pub struct GeneratorKVCache {
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

impl GeneratorKVCache {
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

/// Dense Generator Model - L3 API wrapper.
///
/// All computation delegated to `backend.generator_forward()`.
pub struct GeneratorModel {
    /// Embedding weights [vocab_size, hidden_size]
    embedding: Vec<f32>,
    /// Layer weights
    layers: Vec<LayerWeights>,
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
    /// GPU-resident weights for zero-copy generation (ARCH-PERF-001)
    /// Lazily initialized on first GPU generation call
    gpu_weights: Option<GeneratorModelWeightsGpu>,
}

impl GeneratorModel {
    pub fn new(config: ModelConfig, backend: BackendImpl) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let vocab_size = config.vocab_size;
        let num_q_heads = config.num_attention_heads;
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_q_heads);
        let max_position = config.max_position_embeddings;

        // Build layers
        let layers: Vec<LayerWeights> = (0..num_layers)
            .map(|_| LayerWeights::new(&config))
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
            gpu_weights: None,
        })
    }

    fn build_rope_cache(config: &ModelConfig, head_dim: usize, max_position: usize) -> (Vec<f32>, Vec<f32>) {
        let theta = config.rope_theta.unwrap_or(10000.0) as f32;
        #[cfg(debug_assertions)]
        {
            if theta != 10000.0 {
                println!("=== RoPE CONFIG ===");
                println!("  theta: {}", theta);
                println!("  head_dim: {}", head_dim);
                println!("  max_position: {}", max_position);
            }
        }
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
    pub fn create_kv_cache(&self) -> GeneratorKVCache {
        let num_kv_heads = self.config.num_key_value_heads.unwrap_or(self.config.num_attention_heads);
        let head_dim = self.config.head_dim.unwrap_or(
            self.config.hidden_size / self.config.num_attention_heads
        );
        GeneratorKVCache::new(
            self.config.num_hidden_layers,
            num_kv_heads,
            head_dim,
            self.config.max_position_embeddings,
        )
    }

    /// Forward pass using L3 API.
    pub fn forward(&self, input_ids: &[u32], cache: &mut GeneratorKVCache) -> Result<LogitsTensor> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(Error::InvalidConfig("Input IDs cannot be empty".into()));
        }

        // Build layer weights for L3 API
        let layer_weights: Vec<TransformerLayerWeights<'_>> = self
            .layers
            .iter()
            .map(|l| l.as_kernel_weights())
            .collect();

        // Determine activation
        let activation = if self.config.hidden_act.as_deref() == Some("gelu") {
            Activation::GELU
        } else {
            Activation::SiLU
        };

        // Build config for L3 API
        let forward_config = GeneratorForwardConfig {
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
            vocab_size: self.config.vocab_size,
            max_seq_len: self.config.max_position_embeddings,
            rms_norm_eps: self.config.rms_norm_eps.unwrap_or(1e-6) as f32,
            rope_theta: self.config.rope_theta.unwrap_or(10000.0) as f32,
            use_rope: true,
            activation,
            position_offset: cache.seq_len(),
            final_logit_softcapping: self.config.final_logit_softcapping,
        };

        // Get mutable cache state
        let mut cache_state = cache.as_cache_state();

        // Call L3 API - ALL computation happens in gllm-kernels
        let logits = self.backend.generator_forward(
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
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut LayerWeights> {
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

    /// Get reference to config.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Build forward config for L3 API.
    fn build_forward_config(&self, seq_len: usize, position_offset: usize) -> GeneratorForwardConfig {
        let activation = if self.config.hidden_act.as_deref() == Some("gelu") {
            Activation::GELU
        } else {
            Activation::SiLU
        };

        GeneratorForwardConfig {
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
            vocab_size: self.config.vocab_size,
            max_seq_len: self.config.max_position_embeddings,
            rms_norm_eps: self.config.rms_norm_eps.unwrap_or(1e-6) as f32,
            rope_theta: self.config.rope_theta.unwrap_or(10000.0) as f32,
            use_rope: true,
            activation,
            position_offset,
            final_logit_softcapping: self.config.final_logit_softcapping,
        }
    }

    /// Upload weights to GPU for zero-copy generation (ARCH-PERF-001).
    ///
    /// Call this once after loading weights to enable GPU-native generation.
    pub fn upload_weights_to_gpu(&mut self) -> Result<()> {
        let layer_weights: Vec<TransformerLayerWeights<'_>> = self
            .layers
            .iter()
            .map(|l| l.as_kernel_weights())
            .collect();

        let forward_config = self.build_forward_config(1, 0);

        let gpu_weights = self.backend.upload_generator_weights(
            &self.embedding,
            &layer_weights,
            &self.final_norm,
            &self.lm_head,
            &self.cos_cache,
            &self.sin_cache,
            &forward_config,
        ).map_err(|e| Error::InferenceError(e))?;

        self.gpu_weights = Some(gpu_weights);
        Ok(())
    }

    /// Check if GPU weights are available.
    pub fn has_gpu_weights(&self) -> bool {
        self.gpu_weights.is_some()
    }

    /// Allocate GPU-resident KV cache for zero-copy generation (ARCH-PERF-001).
    ///
    /// Uses a smaller default cache size to fit in limited GPU memory.
    /// For GTX 1060 (6GB), we limit to 2048 tokens to leave room for weights.
    pub fn create_gpu_kv_cache(&self) -> Result<KVCacheGpu> {
        self.create_gpu_kv_cache_with_size(None)
    }

    /// Allocate GPU-resident KV cache with custom max length.
    ///
    /// If `max_len` is None, uses a conservative default based on model size
    /// to avoid GPU OOM on limited VRAM devices.
    pub fn create_gpu_kv_cache_with_size(&self, max_len: Option<usize>) -> Result<KVCacheGpu> {
        let num_kv_heads = self.config.num_key_value_heads.unwrap_or(self.config.num_attention_heads);
        let head_dim = self.config.head_dim.unwrap_or(
            self.config.hidden_size / self.config.num_attention_heads
        );

        // Calculate a reasonable default max_len based on available VRAM
        // For 6GB GPU with ~2.5GB weights, leave ~2GB for KV cache
        // KV cache size = 2 * num_layers * num_kv_heads * max_len * head_dim * 4 bytes
        let default_max_len = {
            let num_layers = self.config.num_hidden_layers;
            let kv_cache_per_token = 2 * num_layers * num_kv_heads * head_dim * 4; // bytes
            let available_vram_for_kv = 2 * 1024 * 1024 * 1024; // 2GB conservative estimate
            let max_tokens = available_vram_for_kv / kv_cache_per_token;
            // Cap at 4096 for reasonable memory usage, min 512 for functionality
            max_tokens.clamp(512, 4096)
        };

        let actual_max_len = max_len.unwrap_or(default_max_len)
            .min(self.config.max_position_embeddings);

        log::debug!(
            "Allocating GPU KV cache: layers={}, kv_heads={}, max_len={} (model max={}), head_dim={}",
            self.config.num_hidden_layers, num_kv_heads, actual_max_len,
            self.config.max_position_embeddings, head_dim
        );

        self.backend.alloc_kv_cache_gpu(
            self.config.num_hidden_layers,
            1, // batch_size
            num_kv_heads,
            actual_max_len,
            head_dim,
        ).map_err(|e| Error::InferenceError(e))
    }

    /// GPU-native forward pass that keeps logits on GPU (ARCH-PERF-001).
    ///
    /// Returns LogitsTensor which may remain on GPU, avoiding expensive
    /// GPU→CPU transfer of full vocabulary logits.
    pub fn forward_gpu_native(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KVCacheGpu,
    ) -> Result<LogitsTensor> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(Error::InvalidConfig("Input IDs cannot be empty".into()));
        }

        let gpu_weights = self.gpu_weights.as_ref()
            .ok_or_else(|| Error::InvalidConfig(
                "GPU weights not uploaded. Call upload_weights_to_gpu() first.".into()
            ))?;

        let forward_config = self.build_forward_config(seq_len, kv_cache.seq_len);

        let logits = self.backend.generator_forward_gpu_pure_logits(
            input_ids,
            gpu_weights,
            kv_cache,
            &forward_config,
        ).map_err(|e| Error::InferenceError(e))?;

        Ok(logits)
    }

    /// Sample from LogitsTensor (CPU or GPU) with minimal data transfer (ARCH-PERF-001).
    ///
    /// For GPU tensors:
    /// - Greedy (temp=0): GPU argmax → 1 u32 transfer
    /// - Sampling: GPU top-k → 64 candidates transfer → CPU sampling
    pub fn sample_from_logits(&self, logits: &LogitsTensor, temperature: f32) -> Result<u32> {
        let vocab_size = logits.vocab_size();
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

    /// Generate text using GPU-native zero-copy loop (ARCH-PERF-001).
    ///
    /// This is the high-performance generation path that:
    /// 1. Keeps all weights and KV cache on GPU
    /// 2. Keeps logits on GPU between forward and sampling
    /// 3. Only transfers sampled token IDs (4 bytes per token vs 128KB+)
    pub fn generate_gpu_native(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
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

        // Create GPU KV cache
        let mut kv_cache = self.create_gpu_kv_cache()?;

        // Process prompt - get logits (stays on GPU!)
        let mut logits = self.forward_gpu_native(&prompt_u32, &mut kv_cache)?;

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

        // Generate tokens - GPU-native loop!
        for _ in 0..max_new_tokens {
            // Check if we're at max context length
            if kv_cache.seq_len >= self.config.max_position_embeddings {
                finish_reason = FinishReason::MaxTokens;
                break;
            }

            // Sample next token from GPU logits - minimal transfer!
            let next_token = self.sample_from_logits(&logits, temperature)?;

            // Check for stop token
            if stop_tokens.contains(&next_token) {
                finish_reason = FinishReason::StopToken;
                break;
            }

            // Add to generated tokens
            all_tokens.push(next_token as i64);

            // Forward single token to get next logits (stays on GPU!)
            logits = self.forward_gpu_native(&[next_token], &mut kv_cache)?;
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

    /// Generate text from prompt tokens.
    ///
    /// Automatically uses GPU-native path if weights are uploaded to GPU.
    pub(crate) fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
        _options: &GenerationOptions,
    ) -> Result<GenerationOutput> {
        // Use GPU-native path if weights are uploaded
        if self.has_gpu_weights() {
            return self.generate_gpu_native(prompt_ids, config, tokenizer);
        }

        // Fallback to CPU path
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

/// Trait for generator models (dense and MoE) - using GeneratorKVCache.
pub trait GeneratorInferTrait: Send + Sync {
    fn forward(&self, input_ids: &[u32], cache: &mut GeneratorKVCache) -> Result<LogitsTensor>;
    fn sample(&self, logits: &LogitsTensor, temperature: f32) -> Result<u32>;
    fn max_position_embeddings(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn create_kv_cache(&self) -> GeneratorKVCache;
}

impl GeneratorInferTrait for GeneratorModel {
    fn forward(&self, input_ids: &[u32], cache: &mut GeneratorKVCache) -> Result<LogitsTensor> {
        self.forward(input_ids, cache)
    }

    fn sample(&self, logits: &LogitsTensor, temperature: f32) -> Result<u32> {
        self.sample(logits, temperature)
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings()
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size()
    }

    fn num_layers(&self) -> usize {
        self.num_layers()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn create_kv_cache(&self) -> GeneratorKVCache {
        self.create_kv_cache()
    }
}
