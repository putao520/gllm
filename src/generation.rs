use crate::engine::{EngineBackend, TokenizerAdapter};
use crate::types::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    StopToken,
    MaxTokens,
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub stop_tokens: Vec<i64>,
    /// Optional speculative decoding configuration
    pub speculative_decoding: Option<SpeculativeDecodingConfig>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            stop_tokens: Vec::new(),
            speculative_decoding: None,
        }
    }
}

/// Speculative Decoding algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeDecodingAlgorithm {
    /// EAGLE-3: Adaptive draft length with multi-layer feature fusion
    Eagle3,
    /// SpecEE / LayerSkip: Early-exit speculation
    SpecEE,
    /// Medusa: Multi-head parallel draft generation
    Medusa,
    /// DeFT/Talon: Tree-structured speculation
    TreeAttention,
}

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingConfig {
    /// Algorithm to use
    pub algorithm: SpeculativeDecodingAlgorithm,
    /// Maximum draft length (tokens to speculate ahead)
    pub max_draft_length: usize,
    /// Confidence threshold for draft acceptance (EAGLE-3 / SpecEE)
    pub confidence_threshold: f32,
    /// Number of Medusa heads (only for Medusa algorithm)
    pub num_medusa_heads: usize,
    /// Top-k for candidate selection (Medusa)
    pub medusa_top_k: usize,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            algorithm: SpeculativeDecodingAlgorithm::Eagle3,
            max_draft_length: 5,
            confidence_threshold: 0.8,
            num_medusa_heads: 3,
            medusa_top_k: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub tokens: Vec<i64>,
    pub finish_reason: FinishReason,
}

pub struct GenerationBuilder<'a> {
    pub(crate) engine: &'a EngineBackend,
    pub(crate) tokenizer: &'a TokenizerAdapter,
    pub(crate) prompt: String,
    pub(crate) config: GenerationConfig,
}

impl<'a> GenerationBuilder<'a> {
    pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.config.max_new_tokens = max_new_tokens;
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.config.top_p = top_p;
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.config.top_k = top_k;
        self
    }

    pub fn stop_tokens(mut self, tokens: Vec<i64>) -> Self {
        self.config.stop_tokens = tokens;
        self
    }

    pub fn add_stop_token(mut self, token: i64) -> Self {
        self.config.stop_tokens.push(token);
        self
    }

    /// Enable speculative decoding with custom configuration
    pub fn speculative_decoding(mut self, config: SpeculativeDecodingConfig) -> Self {
        self.config.speculative_decoding = Some(config);
        self
    }

    /// Enable EAGLE-3 speculative decoding with default parameters
    pub fn eagle3(mut self) -> Self {
        self.config.speculative_decoding = Some(SpeculativeDecodingConfig {
            algorithm: SpeculativeDecodingAlgorithm::Eagle3,
            ..Default::default()
        });
        self
    }

    /// Enable Medusa speculative decoding
    pub fn medusa(mut self, num_heads: usize) -> Self {
        self.config.speculative_decoding = Some(SpeculativeDecodingConfig {
            algorithm: SpeculativeDecodingAlgorithm::Medusa,
            num_medusa_heads: num_heads,
            ..Default::default()
        });
        self
    }

    /// Enable SpecEE/LayerSkip early-exit speculation
    pub fn spec_ee(mut self, confidence_threshold: f32) -> Self {
        self.config.speculative_decoding = Some(SpeculativeDecodingConfig {
            algorithm: SpeculativeDecodingAlgorithm::SpecEE,
            confidence_threshold,
            ..Default::default()
        });
        self
    }

    pub fn run(self) -> Result<GenerationOutput> {
        if self.prompt.trim().is_empty() {
            return Err(Error::InvalidConfig(
                "Prompt is required for generation".into(),
            ));
        }

        let max_len = self.engine.max_position_embeddings().unwrap_or(0);
        let prompt_tokens = self
            .tokenizer
            .encode_unpadded(&self.prompt, usize::MAX);

        if prompt_tokens.is_empty() {
            return Err(Error::InvalidConfig(
                "Prompt tokenization produced empty sequence".into(),
            ));
        }

        if max_len > 0 && prompt_tokens.len() > max_len {
            return Err(Error::InvalidConfig(format!(
                "Prompt length {} exceeds max position {}",
                prompt_tokens.len(),
                max_len
            )));
        }

        self.engine
            .run_generate(prompt_tokens, &self.config, self.tokenizer)
    }
}

// Sync API
#[cfg(not(feature = "tokio"))]
impl<'a> GenerationBuilder<'a> {
    pub fn generate(self) -> Result<GenerationOutput> {
        self.run()
    }
}

// Async API
#[cfg(feature = "tokio")]
impl<'a> GenerationBuilder<'a> {
    pub async fn generate(self) -> Result<GenerationOutput> {
        // block_in_place only works on multi-threaded runtime.
        // For current_thread runtime, run synchronously (it's already blocking).
        let handle = tokio::runtime::Handle::current();
        match handle.runtime_flavor() {
            tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| self.run())
            }
            _ => {
                // CurrentThread or other: run directly (already on blocking context)
                self.run()
            }
        }
    }
}
