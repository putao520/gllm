//! Generation loop skeleton.

use crate::client::{Client, ClientError};

pub struct GenerationBuilder<'a> {
    client: &'a Client,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    session_id: Option<u64>,
}

impl<'a> GenerationBuilder<'a> {
    pub(crate) fn from_prompt(client: &'a Client, prompt: impl Into<String>) -> Self {
        Self {
            client,
            prompt: prompt.into(),
            max_tokens: 256,
            temperature: 0.7,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
        }
    }

    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set session ID for multi-turn conversation KV cache reuse.
    pub fn session_id(mut self, session_id: u64) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn generate(self) -> Result<GenerationResponse, ClientError> {
        self.client.execute_generation(
            self.prompt,
            self.max_tokens,
            self.temperature,
            self.top_k,
            self.top_p,
            self.session_id,
        )
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResponse {
    pub text: String,
    pub request_id: Option<u64>,
}
