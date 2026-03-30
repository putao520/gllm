//! Generation loop skeleton.

use crate::client::{Client, GllmError};

/// 流式生成输出块 (per SPEC 04-API-DESIGN §3.1)
///
/// 每个块包含生成的一个或多个 token 及其累积文本。
#[derive(Debug, Clone)]
pub struct GenerationChunk {
    /// 当前块生成的 token ID 列表
    pub tokens: Vec<u32>,
    /// 当前块的累积文本（解码后）
    pub text: String,
    /// 是否为最后一个块
    pub finished: bool,
}

impl GenerationChunk {
    pub(crate) fn new() -> Self {
        Self {
            tokens: Vec::new(),
            text: String::new(),
            finished: false,
        }
    }

    pub(crate) fn with_token(mut self, token: u32, text: String) -> Self {
        self.tokens.push(token);
        self.text = text;
        self
    }

    pub(crate) fn finish(mut self) -> Self {
        self.finished = true;
        self
    }
}

/// 流式生成迭代器 (per SPEC 04-API-DESIGN §3.1)
///
/// 返回 `impl Iterator<Item=GenerationChunk>`，允许逐 token 迭代处理。
pub struct GenerationStream<'a> {
    _client: &'a Client,
    _prompt: String,
    _max_tokens: usize,
    _temperature: f32,
    _top_k: usize,
    _top_p: f32,
    _session_id: Option<u64>,
}

impl<'a> GenerationStream<'a> {
    pub(crate) fn new(
        client: &'a Client,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: Option<u64>,
    ) -> Self {
        Self {
            _client: client,
            _prompt: prompt,
            _max_tokens: max_tokens,
            _temperature: temperature,
            _top_k: top_k,
            _top_p: top_p,
            _session_id: session_id,
        }
    }
}

impl<'a> Iterator for GenerationStream<'a> {
    type Item = GenerationChunk;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: 实现逐 token 生成
        // 当前返回 None 表示"暂未实现"
        None
    }
}

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

    /// 启用流式生成 (per SPEC 04-API-DESIGN §3.1)
    ///
    /// 返回流式生成迭代器，逐 token 返回 `GenerationChunk`。
    pub fn stream(self) -> GenerationStream<'a> {
        GenerationStream::new(
            self.client,
            self.prompt,
            self.max_tokens,
            self.temperature,
            self.top_k,
            self.top_p,
            self.session_id,
        )
    }

    pub fn generate(self) -> Result<GenerationResponse, GllmError> {
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
    pub thinking_content: Option<String>,
    pub request_id: Option<u64>,
}

/// Split thinking content from generated text.
/// Looks for `<think>...</think>` markers and separates them.
pub fn split_thinking_content(text: &str) -> (String, Option<String>) {
    if let Some(start) = text.find("<think>") {
        if let Some(end) = text.find("</think>") {
            if end > start {
                let think_start = start + "<think>".len();
                let thinking = text[think_start..end].trim().to_string();
                let mut answer = String::new();
                answer.push_str(&text[..start]);
                answer.push_str(&text[end + "</think>".len()..]);
                let answer = answer.trim().to_string();
                let thinking = if thinking.is_empty() { None } else { Some(thinking) };
                return (answer, thinking);
            }
        }
    }
    (text.to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_no_thinking() {
        let (text, think) = split_thinking_content("Hello world");
        assert_eq!(text, "Hello world");
        assert!(think.is_none());
    }

    #[test]
    fn split_with_thinking() {
        let input = "<think>reasoning here</think>The answer is 42.";
        let (text, think) = split_thinking_content(input);
        assert_eq!(text, "The answer is 42.");
        assert_eq!(think.unwrap(), "reasoning here");
    }

    #[test]
    fn split_empty_thinking() {
        let input = "<think></think>Just the answer.";
        let (text, think) = split_thinking_content(input);
        assert_eq!(text, "Just the answer.");
        assert!(think.is_none());
    }
}
