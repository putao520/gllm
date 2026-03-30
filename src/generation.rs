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

/// Internal state machine for the streaming iterator.
enum StreamPhase {
    /// Not yet started — need to run full generation.
    Pending {
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: Option<u64>,
    },
    /// Generation completed; yielding tokens one by one.
    Yielding {
        tokens: Vec<u32>,
        index: usize,
    },
    /// All tokens yielded.
    Done,
}

/// 流式生成迭代器 (per SPEC 04-API-DESIGN §3.1)
///
/// 返回 `impl Iterator<Item = Result<GenerationChunk, GllmError>>`，允许逐 token 迭代处理。
///
/// # 实现策略
///
/// 首次调用 `next()` 时执行完整的 `generate()` 生成全部 tokens，
/// 随后逐 token 生成 `GenerationChunk` 并 yield。
/// 这确保 streaming 接口在语义上正确，同时不引入异步执行器依赖。
pub struct GenerationStream {
    state: std::sync::Arc<std::sync::RwLock<Option<crate::client::ClientState>>>,
    phase: StreamPhase,
}

impl GenerationStream {
    pub(crate) fn new(
        client: &Client,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: Option<u64>,
    ) -> Self {
        Self {
            state: client.state_handle(),
            phase: StreamPhase::Pending {
                prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                session_id,
            },
        }
    }
}

impl Iterator for GenerationStream {
    type Item = Result<GenerationChunk, GllmError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.phase {
                StreamPhase::Pending {
                    prompt,
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    session_id,
                } => {
                    // Run full generation under write lock.
                    let result = (|| -> Result<Vec<u32>, GllmError> {
                        let guard = self
                            .state
                            .write()
                            .map_err(|_| GllmError::ExecutorPoisoned)?;
                        let loaded = guard.as_ref().ok_or(GllmError::NoModelLoaded)?;
                        let mut executor = loaded.backend.executor_mut();

                        let text = if let Some(sid) = session_id {
                            executor.generate_with_session(
                                prompt,
                                *max_tokens,
                                *temperature,
                                *top_k,
                                *top_p,
                                *sid,
                            )?
                        } else {
                            executor.generate(
                                prompt,
                                *max_tokens,
                                *temperature,
                                *top_k,
                                *top_p,
                            )?
                        };

                        // Re-encode text to tokens for per-token yielding.
                        drop(executor);
                        let tokens = {
                            let executor = loaded.backend.executor();
                            executor.encode_prompt(&text)?
                        };
                        Ok(tokens)
                    })();

                    match result {
                        Ok(tokens) => {
                            if tokens.is_empty() {
                                self.phase = StreamPhase::Done;
                                return Some(Ok(GenerationChunk {
                                    tokens: Vec::new(),
                                    text: String::new(),
                                    finished: true,
                                }));
                            }
                            self.phase = StreamPhase::Yielding { tokens, index: 0 };
                        }
                        Err(e) => {
                            self.phase = StreamPhase::Done;
                            return Some(Err(e));
                        }
                    }
                }
                StreamPhase::Yielding { tokens, index } => {
                    if *index >= tokens.len() {
                        self.phase = StreamPhase::Done;
                        return None;
                    }

                    let token = tokens[*index];
                    let end = *index + 1;
                    let is_last = end >= tokens.len();

                    // Decode the text up to this token.
                    let text_up_to = {
                        let guard = match self.state.read() {
                            Ok(g) => g,
                            Err(_) => {
                                self.phase = StreamPhase::Done;
                                return Some(Err(GllmError::ExecutorPoisoned));
                            }
                        };
                        let loaded = match guard.as_ref() {
                            Some(s) => s,
                            None => {
                                self.phase = StreamPhase::Done;
                                return Some(Err(GllmError::NoModelLoaded));
                            }
                        };
                        let executor = loaded.backend.executor();
                        executor
                            .decode_tokens(&tokens[..end])
                            .unwrap_or_default()
                    };

                    *index += 1;

                    return Some(Ok(GenerationChunk {
                        tokens: vec![token],
                        text: text_up_to,
                        finished: is_last,
                    }));
                }
                StreamPhase::Done => {
                    return None;
                }
            }
        }
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
    pub fn stream(self) -> GenerationStream {
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
