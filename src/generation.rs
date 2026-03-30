//! Generation loop — async-first design (per SPEC 04-API-DESIGN §3.1).

use crate::client::{Client, GllmError};
use futures::stream::Stream;
use futures::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

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

/// Internal state machine for the streaming future.
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

/// 流式生成 Stream (per SPEC 04-API-DESIGN §3.1 REQ-GEN-004/005)
///
/// 实现 `futures::Stream<Item = Result<GenerationChunk, GllmError>>`，允许异步逐 token 迭代处理。
///
/// # 实现策略
///
/// 首次调用 `poll_next()` 时执行完整的 `generate()` 生成全部 tokens，
/// 随后逐 token 生成 `GenerationChunk` 并 yield。
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

impl Stream for GenerationStream {
    type Item = Result<GenerationChunk, GllmError>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // SAFETY: GenerationStream contains only Arc<RwLock<_>> + enum,
        // both of which are safe to access through Pin.
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match &mut this.phase {
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
                        let guard = this
                            .state
                            .write()
                            .map_err(|_| GllmError::RuntimeError("state lock poisoned".to_string()))?;
                        let loaded = guard.as_ref().ok_or(GllmError::RuntimeError("no model loaded".to_string()))?;
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
                                this.phase = StreamPhase::Done;
                                return Poll::Ready(Some(Ok(GenerationChunk {
                                    tokens: Vec::new(),
                                    text: String::new(),
                                    finished: true,
                                })));
                            }
                            this.phase = StreamPhase::Yielding { tokens, index: 0 };
                        }
                        Err(e) => {
                            this.phase = StreamPhase::Done;
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }
                StreamPhase::Yielding { tokens, index } => {
                    if *index >= tokens.len() {
                        this.phase = StreamPhase::Done;
                        return Poll::Ready(None);
                    }

                    let token = tokens[*index];
                    let end = *index + 1;
                    let is_last = end >= tokens.len();

                    // Decode the text up to this token.
                    let text_up_to = {
                        let guard = match this.state.read() {
                            Ok(g) => g,
                            Err(_) => {
                                this.phase = StreamPhase::Done;
                                return Poll::Ready(Some(Err(GllmError::RuntimeError("state lock poisoned".to_string()))));
                            }
                        };
                        let loaded = match guard.as_ref() {
                            Some(s) => s,
                            None => {
                                this.phase = StreamPhase::Done;
                                return Poll::Ready(Some(Err(GllmError::RuntimeError("no model loaded".to_string()))));
                            }
                        };
                        let executor = loaded.backend.executor();
                        executor
                            .decode_tokens(&tokens[..end])
                            .unwrap_or_default()
                    };

                    *index += 1;

                    return Poll::Ready(Some(Ok(GenerationChunk {
                        tokens: vec![token],
                        text: text_up_to,
                        finished: is_last,
                    })));
                }
                StreamPhase::Done => {
                    return Poll::Ready(None);
                }
            }
        }
    }
}

/// Response from text generation (per SPEC 04-API-DESIGN §3.1).
#[derive(Debug, Clone)]
pub struct GenerationResponse {
    /// Generated text.
    pub text: String,
    /// Thinking content (for models with thinking heads).
    pub thinking_content: Option<String>,
    /// Request ID (for tracking).
    pub request_id: Option<u64>,
}

/// Builder for text generation (per SPEC 04-API-DESIGN §3.1).
///
/// # Example
///
/// ```no_run
/// use gllm::Client;
/// use futures::stream::StreamExt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_empty();
/// // Non-streaming
/// let response = client.generate("Hello, who are you?")
///     .max_tokens(100)
///     .temperature(0.7)
///     .stream(false)
///     .generate()
///     .await?;
/// println!("Generated: {}", response.text);
///
/// // Streaming
/// let mut stream = client.generate("Hello, who are you?")
///     .max_tokens(100)
///     .stream(true)
///     .generate();
///
/// while let Some(chunk) = stream.next().await {
///     let chunk = chunk?;
///     println!("Token: {}", chunk.text);
/// }
/// # Ok(())
/// # }
/// ```
pub struct GenerationBuilder<'a> {
    client: &'a Client,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    session_id: Option<u64>,
    stream: bool,
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
            stream: false,
        }
    }

    /// Set maximum tokens to generate.
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k sampling parameter.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set top-p (nucleus) sampling parameter.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set session ID for multi-turn conversation KV cache reuse.
    pub fn session_id(mut self, session_id: u64) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// 启用/禁用流式生成 (per SPEC 04-API-DESIGN §3.1 REQ-GEN-004)
    ///
    /// - `true`: 返回 `GenerationStream`
    /// - `false`: 返回 `impl Future<Output = Result<GenerationResponse, GllmError>>`
    pub fn stream(mut self, enable: bool) -> Self {
        self.stream = enable;
        self
    }

    /// Execute the generation (per SPEC 04-API-DESIGN §3.1 REQ-GEN-005)
    ///
    /// 根据 `stream` 字段返回不同类型：
    /// - `stream == false`: 返回 `impl Future<Output = Result<GenerationResponse, GllmError>>`
    /// - `stream == true`: 返回 `GenerationStream` (implements `futures::Stream`)
    ///
    /// # Example (Non-streaming)
    ///
    /// ```no_run
    /// # use gllm::Client;
    /// # async fn example(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    /// let response = client.generate("Hello")
    ///     .stream(false)
    ///     .generate()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Example (Streaming)
    ///
    /// ```no_run
    /// # use gllm::Client;
    /// # use futures::stream::StreamExt;
    /// # async fn example(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut stream = client.generate("Hello")
    ///     .stream(true)
    ///     .generate();
    /// while let Some(chunk) = stream.next().await {
    ///     let chunk = chunk?;
    ///     println!("Token: {}", chunk.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate(self) -> GenerationOutput<'a> {
        if self.stream {
            GenerationOutput::Stream(GenerationStream::new(
                self.client,
                self.prompt,
                self.max_tokens,
                self.temperature,
                self.top_k,
                self.top_p,
                self.session_id,
            ))
        } else {
            // Create the async future for non-streaming generation
            let future = async move {
                self.client
                    .execute_generation(
                        self.prompt,
                        self.max_tokens,
                        self.temperature,
                        self.top_k,
                        self.top_p,
                        self.session_id,
                    )
                    .await
            };
            GenerationOutput::Response(Box::pin(future))
        }
    }
}

/// Generation output type (per SPEC 04-API-DESIGN §3.1 REQ-GEN-004/005)
///
/// 封装了流式和非流式两种生成输出，统一 API。
///
/// # Variants
///
/// - `Response(...)`: 非流式，返回 `Future<Output = Result<GenerationResponse, GllmError>>`
/// - `Stream(...)`: 流式，返回 `GenerationStream` (implements `futures::Stream`)
pub enum GenerationOutput<'a> {
    /// 完整生成响应（非流式，Future）
    Response(
        Pin<
            Box<
                dyn Future<Output = Result<GenerationResponse, GllmError>>
                    + Send
                    + 'a,
            >,
        >,
    ),
    /// 流式生成 Stream
    Stream(GenerationStream),
}

impl<'a> GenerationOutput<'a> {
    /// 检查是否为流式输出
    pub fn is_stream(&self) -> bool {
        matches!(self, GenerationOutput::Stream(_))
    }
}

/// Split thinking content from generated text (internal helper).
pub fn split_thinking_content(text: &str) -> (String, Option<String>) {
    // Look for thinking markers like "<thinking>...</thinking>"
    let start_marker = "<thinking>";
    let end_marker = "</thinking>";

    if let Some(start) = text.find(start_marker) {
        if let Some(end) = text.find(end_marker) {
            let thinking = &text[start + start_marker.len()..end];
            let main_text = format!("{}{}", &text[..start], &text[end + end_marker.len()..]);
            return (main_text, Some(thinking.to_string()));
        }
    }

    (text.to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_thinking_content() {
        // With thinking markers
        let (text, thinking) = split_thinking_content("Hello<thinking>I am thinking</thinking> world");
        assert_eq!(text, "Hello world");
        assert_eq!(thinking, Some("I am thinking".to_string()));

        // Without thinking markers
        let (text, thinking) = split_thinking_content("Hello world");
        assert_eq!(text, "Hello world");
        assert!(thinking.is_none());
    }
}
