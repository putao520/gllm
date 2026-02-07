//! Generation loop skeleton.

use crate::client::{Client, ClientError};

pub struct GenerationBuilder<'a> {
    client: &'a Client,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
}

impl<'a> GenerationBuilder<'a> {
    pub(crate) fn from_prompt(client: &'a Client, prompt: impl Into<String>) -> Self {
        Self {
            client,
            prompt: prompt.into(),
            max_tokens: 256,
            temperature: 0.7,
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

    pub fn generate(self) -> Result<GenerationResponse, ClientError> {
        self.client
            .execute_generation(self.prompt, self.max_tokens, self.temperature)
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResponse {
    pub text: String,
    pub request_id: Option<u64>,
}
