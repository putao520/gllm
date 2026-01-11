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
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            stop_tokens: Vec::new(),
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

    fn run(self) -> Result<GenerationOutput> {
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
        tokio::task::block_in_place(|| self.run())
    }
}
