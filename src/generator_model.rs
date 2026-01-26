//! Generator Model for text generation using gllm-kernels Backend.

use crate::causal_attention::CausalAttention;
use crate::decoder_layer::{DecoderLayer, FFNWeights};
use crate::generation::GenerationOptions;
use crate::generation_loop::{generate_with_ops, ForwardOutput, GenerationOps};
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::prompt_cache::PromptCache;
use crate::rms_norm::RmsNorm;
use crate::types::{Error, Result};
use crate::engine::TokenizerAdapter;
use crate::generation::{GenerationConfig, GenerationOutput};
use crate::scratch_buffer::{ScratchBuffer, ScratchConfig};
use gllm_kernels::backend::{Backend, TensorSlice};
use gllm_kernels::linear_forward;
use std::sync::Arc;
use std::sync::Mutex;

/// Dense (non-MoE) Generator Model.
pub struct GeneratorModel {
    embedding: Vec<f32>,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Vec<f32>,
    config: ModelConfig,
    backend: Arc<dyn Backend>,
    prompt_cache: Mutex<PromptCache>,
}

impl GeneratorModel {
    pub fn new(config: ModelConfig, backend: Arc<dyn Backend>) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let vocab_size = config.vocab_size;
        let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);
        let head_dim = config.head_dim.unwrap_or(hidden_size / config.num_attention_heads);

        // Build RoPE if needed
        let rope = CausalAttention::build_rope(&config, head_dim);

        // Initialize layers
        let eps = config.rms_norm_eps.unwrap_or(1e-6) as f32;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let attention = CausalAttention::new(&config, rope.clone(), true, backend.clone())?;
            let layer = DecoderLayer {
                attention,
                input_norm: RmsNorm::new(hidden_size, eps),
                post_attn_norm: RmsNorm::new(hidden_size, eps),
                ffn: FFNWeights::zeros(hidden_size, intermediate_size),
                hidden_size,
                intermediate_size,
                use_gelu: config.hidden_act.as_deref() == Some("gelu"),
                backend: backend.clone(),
            };
            layers.push(layer);
        }

        Ok(Self {
            embedding: vec![0.0; vocab_size * hidden_size],
            layers,
            final_norm: RmsNorm::new(hidden_size, eps),
            lm_head: vec![0.0; vocab_size * hidden_size],
            config,
            backend,
            prompt_cache: Mutex::new(PromptCache::new(0)),
        })
    }

    /// Forward pass for generation.
    pub fn forward(&self, input_ids: &[u32], cache: &mut KVCache) -> Result<Vec<f32>> {
        Ok(self
            .forward_with_hidden_internal(input_ids, cache, None)?
            .logits)
    }

    pub(crate) fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
        options: &GenerationOptions,
    ) -> Result<GenerationOutput> {
        generate_with_ops(self, prompt_ids, config, tokenizer, options, Some(&self.prompt_cache))
    }

    /// Sample next token from logits.
    pub fn sample(&self, logits: &[f32], temperature: f32) -> Result<u32> {
        let vocab_size = logits.len();
        let mut scaled = logits.to_vec();

        // Apply temperature
        if temperature > 0.0 && temperature != 1.0 {
            for l in scaled.iter_mut() {
                *l /= temperature;
            }
        }

        // Argmax for greedy decoding
        self.backend
            .argmax(TensorSlice::F32(&scaled), 1, vocab_size)
            .map(|v| v[0])
            .map_err(|e| Error::InferenceError(e))
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
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut DecoderLayer> {
        self.layers.get_mut(idx)
    }

    /// Access final norm for weight loading.
    pub fn final_norm_mut(&mut self) -> &mut RmsNorm {
        &mut self.final_norm
    }

    fn forward_with_hidden_internal(
        &self,
        input_ids: &[u32],
        cache: &mut KVCache,
        mut scratch: Option<&mut ScratchBuffer>,
    ) -> Result<ForwardOutput> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(Error::InvalidConfig(
                "Input IDs cannot be empty".into(),
            ));
        }
        let position_offset = cache.seq_len();
        let hidden_size = self.config.hidden_size;

        let mut hidden = vec![0.0f32; seq_len * hidden_size];
        for (i, &token_id) in input_ids.iter().enumerate() {
            let token_id = token_id as usize;
            if token_id >= self.config.vocab_size {
                return Err(Error::InvalidConfig(format!(
                    "Token ID {} exceeds vocab size {}",
                    token_id, self.config.vocab_size
                )));
            }
            let src_start = token_id * hidden_size;
            let dst_start = i * hidden_size;
            hidden[dst_start..dst_start + hidden_size]
                .copy_from_slice(&self.embedding[src_start..src_start + hidden_size]);
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Some(scratch) = scratch.as_deref_mut() {
                hidden = layer.forward_with_scratch(
                    &hidden,
                    position_offset,
                    cache,
                    layer_idx,
                    scratch,
                )?;
            } else {
                hidden = layer.forward(&hidden, position_offset, cache, layer_idx)?;
            }
        }

        self.final_norm.forward_inplace(&mut hidden);

        let last_hidden = hidden[(seq_len - 1) * hidden_size..].to_vec();
        let logits = self.logits_from_hidden(&last_hidden)?;

        Ok(ForwardOutput { logits, last_hidden })
    }

    fn logits_from_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.config.hidden_size {
            return Err(Error::InferenceError(
                "Hidden state length does not match model hidden size".into(),
            ));
        }
        let mut logits = vec![0.0f32; self.config.vocab_size];
        linear_forward(
            hidden,
            &self.lm_head,
            None,
            &mut logits,
            1,
            self.config.hidden_size,
            self.config.vocab_size,
        );
        Ok(logits)
    }
}

/// Trait for generator models (dense and MoE).
pub trait GeneratorModelTrait: Send + Sync {
    fn forward(&self, input_ids: &[u32], cache: &mut KVCache) -> Result<Vec<f32>>;
    fn sample(&self, logits: &[f32], temperature: f32) -> Result<u32>;
    fn max_position_embeddings(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn vocab_size(&self) -> usize;
}

impl GeneratorModelTrait for GeneratorModel {
    fn forward(&self, input_ids: &[u32], cache: &mut KVCache) -> Result<Vec<f32>> {
        self.forward(input_ids, cache)
    }

    fn sample(&self, logits: &[f32], temperature: f32) -> Result<u32> {
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
}

impl GenerationOps for GeneratorModel {
    fn forward_with_hidden(
        &self,
        input_ids: &[u32],
        cache: &mut KVCache,
        scratch: Option<&mut ScratchBuffer>,
    ) -> Result<ForwardOutput> {
        self.forward_with_hidden_internal(input_ids, cache, scratch)
    }

    fn logits_from_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        self.logits_from_hidden(hidden)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn num_layers(&self) -> usize {
        self.num_layers()
    }

    fn num_kv_heads(&self) -> usize {
        self.config
            .num_key_value_heads
            .unwrap_or(self.config.num_attention_heads)
    }

    fn head_dim(&self) -> usize {
        self.config
            .head_dim
            .unwrap_or(self.config.hidden_size / self.config.num_attention_heads)
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings()
    }

    fn scratch_config(&self) -> ScratchConfig {
        ScratchConfig::from_model_config(&self.config)
    }
}
