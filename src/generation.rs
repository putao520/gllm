//! Generation loop — sync-first design (per SPEC 04-API-DESIGN §3.1).
//!
//! No async runtime, no `futures::Stream`, no `Pin<Box<dyn Future>>`.
//! Inference is CPU-bound compute, exposed as synchronous operations.

use std::sync::Arc;

use arc_swap::ArcSwapOption;

use crate::client::{Client, ClientState, GllmError};

/// Hook decision after each decode step.
///
/// per SPEC 04-API-DESIGN §7.4 — guardrail attachment control flow.
#[derive(Debug, Clone, PartialEq)]
pub enum HookDecision {
    /// Continue generation with the current token.
    Continue,
    /// Veto current token, provide reason.
    Veto(String),
    /// Terminate generation entirely.
    Terminate,
}

/// Trait for generation-time hooks (guardrails, probes).
///
/// per SPEC 04-API-DESIGN §7.4 — safety probes can inspect logits
/// and generated tokens to implement content moderation, output filtering,
/// and custom stopping conditions.
///
/// # Example
///
/// ```no_run
/// use gllm::generation::{GenerationHook, HookDecision};
///
/// struct ProfanityFilter {
///     bad_words: Vec<u32>,
/// }
///
/// impl GenerationHook for ProfanityFilter {
///     fn post_step(&self, _logits: &[f32], generated_tokens: &[u32]) -> HookDecision {
///         if let Some(&last_token) = generated_tokens.last() {
///             if self.bad_words.contains(&last_token) {
///                 return HookDecision::Veto("profanity detected".to_string());
///             }
///         }
///         HookDecision::Continue
///     }
/// }
/// ```
pub trait GenerationHook: Send + Sync {
    /// Called after each decode step. Returns whether to continue.
    ///
    /// # Parameters
    ///
    /// - `logits`: Raw model output logits (vocab_size dimensions)
    /// - `generated_tokens`: All generated tokens so far (including current step)
    ///
    /// # Return
    ///
    /// - `HookDecision::Continue`: Accept token and continue generation
    /// - `HookDecision::Veto(reason)`: Reject token, retry sampling
    /// - `HookDecision::Terminate`: Stop generation immediately
    fn post_step(&self, logits: &[f32], generated_tokens: &[u32]) -> HookDecision;
}

/// Simple safety hook that vetoes tokens above a threshold.
///
/// per SPEC 04-API-DESIGN §9.2 — SafetyPolicy::HaltAndVeto
#[derive(Debug)]
pub struct ThresholdHook {
    /// Token IDs to veto
    pub veto_tokens: Vec<u32>,
    /// Maximum allowed vetoes before termination
    pub max_vetoes: usize,
    /// Current veto count
    pub veto_count: std::sync::atomic::AtomicUsize,
}

impl ThresholdHook {
    pub fn new(veto_tokens: Vec<u32>, max_vetoes: usize) -> Self {
        Self {
            veto_tokens,
            max_vetoes,
            veto_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl GenerationHook for ThresholdHook {
    fn post_step(&self, _logits: &[f32], generated_tokens: &[u32]) -> HookDecision {
        if let Some(&last_token) = generated_tokens.last() {
            if self.veto_tokens.contains(&last_token) {
                let count = self.veto_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count + 1 >= self.max_vetoes {
                    return HookDecision::Terminate;
                } else {
                    return HookDecision::Veto(format!(
                        "token {} blocked by threshold policy",
                        last_token
                    ));
                }
            }
        }
        HookDecision::Continue
    }
}

/// Streaming generation output chunk (per SPEC 04-API-DESIGN §3.1)
///
/// Each chunk contains one or more generated tokens and their accumulated text.
#[derive(Debug, Clone)]
pub struct GenerationChunk {
    /// Token IDs in this chunk
    pub tokens: Vec<u32>,
    /// Accumulated decoded text for this chunk
    pub text: String,
    /// Whether this is the final chunk
    pub finished: bool,
}

impl GenerationChunk {
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self {
            tokens: Vec::new(),
            text: String::new(),
            finished: false,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn with_token(mut self, token: u32, text: String) -> Self {
        self.tokens.push(token);
        self.text = text;
        self
    }

    #[allow(dead_code)]
    pub(crate) fn finish(mut self) -> Self {
        self.finished = true;
        self
    }
}

/// Streaming generation iterator (per SPEC 04-API-DESIGN §3.1 REQ-GEN-004/005)
///
/// Implements `Iterator` for synchronous token-by-token iteration.
/// No async runtime, no `futures::Stream` — just a standard Rust iterator.
///
/// On first call to `next()`, executes the full generation synchronously,
/// then yields tokens one-by-one as `GenerationChunk`s.
pub struct GenerationStream {
    state: Arc<ArcSwapOption<ClientState>>,
    phase: StreamPhase,
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
        thinking_budget: Option<usize>,
    },
    /// Generation completed; yielding tokens one by one.
    Yielding {
        tokens: Vec<u32>,
        index: usize,
    },
    /// All tokens yielded.
    Done,
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
        thinking_budget: Option<usize>,
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
                thinking_budget,
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
                    thinking_budget,
                } => {
                    // Run full generation synchronously (lock-free state read)
                    let result = (|| -> Result<Vec<u32>, GllmError> {
                        let loaded = self
                            .state
                            .load_full()
                            .ok_or(GllmError::NoModelLoaded)?;

                        let mut executor = loaded.backend.executor_mut();
                        let text = if let Some(sid) = session_id {
                            executor.generate_with_session(
                                prompt,
                                *max_tokens,
                                *temperature,
                                *top_k,
                                *top_p,
                                *sid,
                                *thinking_budget,
                            )?
                        } else {
                            executor.generate(
                                prompt,
                                *max_tokens,
                                *temperature,
                                *top_k,
                                *top_p,
                                *thinking_budget,
                            )?
                        };

                        // Re-encode text to tokens for per-token yielding
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

                    // Decode the text up to this token (lock-free state read)
                    let text_up_to = {
                        let state_arc = match self.state.load_full() {
                            Some(s) => s,
                            None => {
                                self.phase = StreamPhase::Done;
                                return Some(Err(GllmError::NoModelLoaded));
                            }
                        };
                        let executor = state_arc.backend.executor();
                        match executor.decode_tokens(&tokens[..end]) {
                            Ok(text) => text,
                            Err(e) => {
                                self.phase = StreamPhase::Done;
                                return Some(Err(GllmError::RuntimeError(format!(
                                    "decode_tokens failed: {}",
                                    e
                                ))));
                            }
                        }
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
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_empty();
/// // Non-streaming
/// let response = client.generate("Hello, who are you?")
///     .max_tokens(100)
///     .temperature(0.7)
///     .stream(false)
///     .generate()?;
/// println!("Generated: {}", response.text);
///
/// // Streaming
/// let mut stream = client.generate("Hello, who are you?")
///     .max_tokens(100)
///     .stream(true)
///     .generate();
///
/// while let Some(chunk) = stream.next() {
///     let chunk = chunk?;
///     println!("Token: {}", chunk.text);
/// }
/// # Ok(())
/// # }
/// ```
///
/// 多模态媒体输入源。
///
/// 支持三种输入方式：
/// - 文件路径: 从磁盘加载
/// - Base64: API/Web 场景的内联数据
/// - 原始字节: 内存中已解码的数据
/// - URL: 远程资源 (encoder 负责拉取)
///
/// SPEC 依据: SPEC/04-API-DESIGN.md §3.7.1 四种模式。
#[derive(Debug, Clone)]
pub enum MediaInput {
    /// 文件路径 (本地磁盘)
    File(String),
    /// Base64 编码数据 (含可选 MIME type)
    Base64 { data: String, mime_type: Option<String> },
    /// 原始字节 (已解码的像素/PCM 数据)
    Raw(Vec<u8>),
    /// 远程资源 URL (http/https/s3/file://), 由 encoder 负责拉取。
    /// encoder 实现若不支持网络/远端协议应 Err(RuntimeError::NetworkUnreachable)。
    Url(String),
}

pub struct GenerationBuilder<'a> {
    client: &'a Client,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    session_id: Option<u64>,
    stream: bool,
    /// 多模态: 图像输入 (P3.1 Vision)
    image_input: Option<MediaInput>,
    /// 多模态: 音频输入 (P3.2 Audio)
    audio_input: Option<MediaInput>,
    /// Thinking token budget: None = unlimited, Some(0) = disabled, Some(n) = max n tokens.
    thinking_budget: Option<usize>,
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
            image_input: None,
            audio_input: None,
            thinking_budget: None,
        }
    }

    /// 多模态: 附加图像输入 (Gemma 4 Vision)。
    ///
    /// 支持三种输入方式:
    /// - `MediaInput::File("/path/to/image.jpg")` — 本地文件
    /// - `MediaInput::Base64 { data, mime_type }` — Base64 编码
    /// - `MediaInput::Raw(bytes)` — 已解码的原始像素
    ///
    /// 图像由 SigLIP encoder 编码为视觉 token 序列，
    /// 插入到 prompt 中 `image_token_id` 的位置。
    pub fn image(mut self, input: MediaInput) -> Self {
        self.image_input = Some(input);
        self
    }

    /// 多模态: 附加音频输入 (Gemma 4 Audio)。
    ///
    /// 支持三种输入方式:
    /// - `MediaInput::File("/path/to/audio.wav")` — 本地文件
    /// - `MediaInput::Base64 { data, mime_type }` — Base64 编码
    /// - `MediaInput::Raw(bytes)` — 已解码的原始 PCM
    ///
    /// 音频由 USM Conformer 编码为音频 token 序列，
    /// 插入到 prompt 中 `audio_token_id` 的位置。
    pub fn audio(mut self, input: MediaInput) -> Self {
        self.audio_input = Some(input);
        self
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

    /// Enable/disable streaming generation (per SPEC 04-API-DESIGN §3.1 REQ-GEN-004)
    pub fn stream(mut self, enable: bool) -> Self {
        self.stream = enable;
        self
    }

    /// Set the thinking token budget.
    ///
    /// Controls the maximum number of thinking tokens the model may emit:
    /// - `None` (default): No limit on thinking tokens.
    /// - `Some(0)`: Thinking is disabled entirely.
    /// - `Some(n)`: At most `n` thinking tokens are allowed.
    pub fn thinking_budget(mut self, max_tokens: usize) -> Self {
        self.thinking_budget = Some(max_tokens);
        self
    }

    /// 转换为 CoT Reasoning builder (SPEC 04-API-DESIGN §3.11).
    ///
    /// 对任意 generator LLM (SmolLM2 / Llama / Qwen 等,**不依赖模型自带
    /// thinking_head 权重**) 启用原生多步推理。完全复用现有 `Client::generate`
    /// 公共管线,通过 prompt engineering + 迭代 generate 在 Client 层实现。
    ///
    /// 调用后应链式 `.template(...)` (可选) + `.execute()` 得到
    /// `Result<ReasoningResponse, ClientError>`。
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use gllm::{Client, ReasoningMode};
    /// # fn example(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    /// let answer = client.generate("What is 127 * 83?")
    ///     .reasoning(ReasoningMode::Manual {
    ///         max_reasoning_tokens: 512,
    ///         step_count: 3,
    ///     })
    ///     .execute()?;
    /// println!("Final answer: {}", answer.text);
    /// for (i, step) in answer.reasoning_trace.iter().enumerate() {
    ///     println!("Step {}: {}", i + 1, step);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # 与 `thinking_budget` 的区别
    ///
    /// - `thinking_budget(n)` — 依赖模型自带 thinking_head(qwen3-thinking 等),
    ///   单次 generate 内限制 `<thinking>` token 数。
    /// - `reasoning(mode)` — **任意** LLM,跨多次 generate 的 orchestration。
    ///
    /// 两者可独立使用,不冲突。
    pub fn reasoning(
        self,
        mode: crate::cot_reasoner::ReasoningMode,
    ) -> crate::cot_reasoner::ReasoningBuilder<'a> {
        crate::cot_reasoner::ReasoningBuilder::new(self.client, self.prompt, mode)
    }

    /// Execute the generation (per SPEC 04-API-DESIGN §3.1 REQ-GEN-005)
    ///
    /// Returns either:
    /// - `GenerationOutput::Response(...)`: Synchronous `Result<GenerationResponse, GllmError>`
    /// - `GenerationOutput::Stream(...)`: `GenerationStream` implementing `Iterator`
    ///
    /// # Example (Non-streaming)
    ///
    /// ```no_run
    /// # use gllm::Client;
    /// # fn example(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    /// let response = client.generate("Hello")
    ///     .stream(false)
    ///     .generate()
    ///     .response()?;
    /// println!("{}", response.text);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Example (Streaming)
    ///
    /// ```no_run
    /// # use gllm::Client;
    /// # fn example(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut stream = client.generate("Hello")
    ///     .stream(true)
    ///     .generate()
    ///     stream();
    /// while let Some(chunk) = stream.next() {
    ///     let chunk = chunk?;
    ///     println!("Token: {}", chunk.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate(self) -> GenerationOutput {
        // 多模态输入需要显式的 encoder 调用 + token routing，当前仅在
        // 非 streaming 路径实现；streaming 路径沿用原走法。
        let has_multimodal = self.image_input.is_some() || self.audio_input.is_some();

        if self.stream {
            if has_multimodal {
                // T58 铁律 NO_SILENT_FALLBACK：streaming 多模态尚未接入，
                // 立即 Err 而非静默丢弃媒体输入。
                return GenerationOutput::Response(Err(GllmError::RuntimeError(
                    "multimodal streaming not yet supported (T58 scaffold only)".into(),
                )));
            }
            GenerationOutput::Stream(GenerationStream::new(
                self.client,
                self.prompt,
                self.max_tokens,
                self.temperature,
                self.top_k,
                self.top_p,
                self.session_id,
                self.thinking_budget,
            ))
        } else {
            let result = if has_multimodal {
                self.client.execute_generation_multimodal(
                    self.prompt,
                    self.max_tokens,
                    self.temperature,
                    self.top_k,
                    self.top_p,
                    self.session_id,
                    self.thinking_budget,
                    self.image_input,
                    self.audio_input,
                )
            } else {
                self.client.execute_generation(
                    self.prompt,
                    self.max_tokens,
                    self.temperature,
                    self.top_k,
                    self.top_p,
                    self.session_id,
                    self.thinking_budget,
                )
            };
            GenerationOutput::Response(result)
        }
    }
}

/// Generation output type (per SPEC 04-API-DESIGN §3.1 REQ-GEN-004/005)
///
/// Wraps streaming and non-streaming generation output into a unified type.
///
/// # Variants
///
/// - `Response(...)`: Non-streaming, contains `Result<GenerationResponse, GllmError>`
/// - `Stream(...)`: Streaming, contains `GenerationStream` (implements `Iterator`)
pub enum GenerationOutput {
    /// Complete generation response (non-streaming)
    Response(Result<GenerationResponse, GllmError>),
    /// Streaming generation iterator
    Stream(GenerationStream),
}

impl GenerationOutput {
    /// Check if this is a streaming output
    pub fn is_stream(&self) -> bool {
        matches!(self, GenerationOutput::Stream(_))
    }

    /// Extract the response from a non-streaming output.
    ///
    /// # Panics
    /// Panics if this is a streaming output.
    pub fn response(self) -> Result<GenerationResponse, GllmError> {
        match self {
            GenerationOutput::Response(result) => result,
            GenerationOutput::Stream(_) => panic!("called response() on a streaming output"),
        }
    }

    /// Extract the stream from a streaming output.
    ///
    /// # Panics
    /// Panics if this is a non-streaming output.
    pub fn stream(self) -> GenerationStream {
        match self {
            GenerationOutput::Stream(s) => s,
            GenerationOutput::Response(_) => panic!("called stream() on a non-streaming output"),
        }
    }
}

// ============================================================================
// ThinkingTracker — 实时思考 token 标记
// ============================================================================

/// 思考状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingState {
    /// 正常生成 (非思考阶段)
    Normal,
    /// 正在生成思考内容
    Thinking,
    /// 思考已结束
    Done,
    /// 思考被预算截断
    BudgetExhausted,
}

/// 思考 token 实时追踪器。
///
/// 在生成循环中逐 token 追踪 `<thinking>` / `</thinking>` 标签，
/// 标记每个 token 是否属于思考阶段。供 KV cache 跳过 (T4) 和
/// thinking budget 截断使用。
///
/// 工作方式: 累积解码文本，检测标签出现。
/// 不依赖特殊 token ID（模型可能用不同的 tokenizer 切分标签）。
#[derive(Debug, Clone)]
pub struct ThinkingTracker {
    state: ThinkingState,
    /// 已生成的思考 token 数量
    thinking_token_count: usize,
    /// 思考 token 上限 (None = 无限)
    budget: Option<usize>,
    /// 累积解码文本 (用于检测标签)
    accumulated_text: String,
    /// 已消费的文本位置
    consumed_pos: usize,
}

impl ThinkingTracker {
    pub fn new(budget: Option<usize>) -> Self {
        // budget = Some(0) → 直接标记为 Done (禁止思考)
        let state = if budget == Some(0) {
            ThinkingState::Done
        } else {
            ThinkingState::Normal
        };
        Self {
            state,
            thinking_token_count: 0,
            budget,
            accumulated_text: String::new(),
            consumed_pos: 0,
        }
    }

    /// 喂入新的解码文本片段，更新状态。
    /// 返回当前 token 是否属于思考阶段。
    pub fn feed(&mut self, decoded_text: &str) -> bool {
        if self.state == ThinkingState::Done || self.state == ThinkingState::BudgetExhausted {
            return false;
        }

        self.accumulated_text.push_str(decoded_text);
        let text = &self.accumulated_text[self.consumed_pos..];

        match self.state {
            ThinkingState::Normal => {
                if let Some(pos) = text.find("<thinking>") {
                    self.consumed_pos += pos + "<thinking>".len();
                    self.state = ThinkingState::Thinking;
                    self.thinking_token_count += 1;
                    return true;
                }
                false
            }
            ThinkingState::Thinking => {
                self.thinking_token_count += 1;

                // 检查预算
                if let Some(max) = self.budget {
                    if self.thinking_token_count >= max {
                        self.state = ThinkingState::BudgetExhausted;
                        return true;
                    }
                }

                // 检查结束标签
                if let Some(pos) = text.find("</thinking>") {
                    self.consumed_pos += pos + "</thinking>".len();
                    self.state = ThinkingState::Done;
                }
                true
            }
            _ => false,
        }
    }

    pub fn state(&self) -> ThinkingState {
        self.state
    }

    pub fn is_thinking(&self) -> bool {
        self.state == ThinkingState::Thinking
    }

    pub fn thinking_token_count(&self) -> usize {
        self.thinking_token_count
    }

    pub fn is_budget_exhausted(&self) -> bool {
        self.state == ThinkingState::BudgetExhausted
    }
}

/// Split thinking content from generated text (internal helper).
pub fn split_thinking_content(text: &str) -> (String, Option<String>) {
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
        let (text, thinking) =
            split_thinking_content("Hello<thinking>I am thinking</thinking> world");
        assert_eq!(text, "Hello world");
        assert_eq!(thinking, Some("I am thinking".to_string()));

        // Without thinking markers
        let (text, thinking) = split_thinking_content("Hello world");
        assert_eq!(text, "Hello world");
        assert!(thinking.is_none());
    }

    #[test]
    fn test_thinking_tracker_normal_flow() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(!tracker.feed("Hello "));
        assert_eq!(tracker.state(), ThinkingState::Normal);

        // 进入思考
        assert!(tracker.feed("<thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Thinking);

        // 思考中
        assert!(tracker.feed("let me think..."));
        assert!(tracker.is_thinking());

        // 结束思考
        assert!(tracker.feed("</thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Done);

        // 后续 token 不是思考
        assert!(!tracker.feed(" The answer is 42."));
    }

    #[test]
    fn test_thinking_tracker_budget() {
        let mut tracker = ThinkingTracker::new(Some(3));
        assert!(tracker.feed("<thinking>"));
        assert!(tracker.feed("tok1 ")); // count=2
        assert!(tracker.feed("tok2 ")); // count=3, budget exhausted
        assert!(tracker.is_budget_exhausted());
        assert!(!tracker.feed("tok3 ")); // 超出预算，不再计为思考
    }

    #[test]
    fn test_thinking_tracker_disabled() {
        let mut tracker = ThinkingTracker::new(Some(0));
        assert_eq!(tracker.state(), ThinkingState::Done);
        assert!(!tracker.feed("<thinking>anything</thinking>"));
    }

    // ========================================================================
    // GenerationBuilder multimodal API tests (T58)
    // ========================================================================

    #[test]
    fn generation_builder_image_stores_media_input() {
        let client = Client::new_empty();
        let builder = client
            .generate("hello")
            .image(MediaInput::File("/tmp/test.jpg".into()));
        // image_input is set
        assert!(builder.image_input.is_some());
        assert!(builder.audio_input.is_none());
        match builder.image_input.as_ref().unwrap() {
            MediaInput::File(p) => assert_eq!(p, "/tmp/test.jpg"),
            _ => panic!("expected File variant"),
        }
    }

    #[test]
    fn generation_builder_audio_stores_media_input() {
        let client = Client::new_empty();
        let builder = client
            .generate("hello")
            .audio(MediaInput::Raw(vec![1, 2, 3, 4]));
        assert!(builder.image_input.is_none());
        assert!(builder.audio_input.is_some());
        match builder.audio_input.as_ref().unwrap() {
            MediaInput::Raw(bytes) => assert_eq!(bytes, &[1, 2, 3, 4]),
            _ => panic!("expected Raw variant"),
        }
    }

    #[test]
    fn generation_builder_image_with_base64() {
        let client = Client::new_empty();
        let builder = client.generate("hello").image(MediaInput::Base64 {
            data: "abc".into(),
            mime_type: Some("image/png".into()),
        });
        match builder.image_input.as_ref().unwrap() {
            MediaInput::Base64 { data, mime_type } => {
                assert_eq!(data, "abc");
                assert_eq!(mime_type.as_deref(), Some("image/png"));
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn generation_builder_multimodal_without_model_returns_error() {
        // 无 model loaded + image input → InvalidModelType (no encoder)
        let client = Client::new_empty();
        let out = client
            .generate("hello")
            .image(MediaInput::Raw(vec![0]))
            .generate();
        match out {
            GenerationOutput::Response(Err(e)) => {
                // 无 encoder 注册 → InvalidModelType
                assert!(matches!(e, GllmError::InvalidModelType));
            }
            GenerationOutput::Response(Ok(_)) => panic!("expected error"),
            GenerationOutput::Stream(_) => panic!("stream=false path should not stream"),
        }
    }

    #[test]
    fn generation_builder_stream_plus_multimodal_is_rejected() {
        let client = Client::new_empty();
        let out = client
            .generate("hello")
            .image(MediaInput::Raw(vec![0]))
            .stream(true)
            .generate();
        match out {
            GenerationOutput::Response(Err(e)) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("multimodal streaming"),
                    "expected streaming-not-supported error, got: {msg}"
                );
            }
            _ => panic!("expected streaming+multimodal to error"),
        }
    }

    #[test]
    fn generation_builder_pure_text_unaffected_by_multimodal_fields() {
        // 纯文本 generate（无 .image()/.audio()）应走原 execute_generation
        // 路径，不触碰 multimodal encoder，也不检查多模态 token id。
        let client = Client::new_empty();
        let out = client.generate("hello").generate();
        match out {
            GenerationOutput::Response(Err(e)) => {
                // 未加载模型 → NoModelLoaded (而不是 InvalidModelType)
                assert!(matches!(e, GllmError::NoModelLoaded));
            }
            _ => panic!("expected NoModelLoaded error for text-only generation"),
        }
    }
}
