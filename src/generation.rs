//! Generation loop skeleton.

use crate::adapter::Message;
use crate::client::{Client, ClientError};

#[derive(Debug, Clone)]
pub enum Prompt {
    Raw(String),
    Messages(Vec<Message>),
}

pub struct GenerationBuilder<'a> {
    client: &'a Client,
    prompt: Prompt,
    max_tokens: usize,
    temperature: f32,
}

impl<'a> GenerationBuilder<'a> {
    pub(crate) fn from_prompt(client: &'a Client, prompt: impl Into<String>) -> Self {
        Self {
            client,
            prompt: Prompt::Raw(prompt.into()),
            max_tokens: 256,
            temperature: 0.7,
        }
    }

    pub(crate) fn from_messages(client: &'a Client, messages: Vec<Message>) -> Self {
        Self {
            client,
            prompt: Prompt::Messages(messages),
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
        let prompt = match self.prompt {
            Prompt::Raw(text) => text,
            Prompt::Messages(messages) => self.client.render_chat_prompt(&messages)?,
        };
        self.client
            .execute_generation(prompt, self.max_tokens, self.temperature)
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResponse {
    pub text: String,
    pub request_id: Option<u64>,
}
