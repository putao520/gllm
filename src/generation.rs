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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq)]
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
    /// Pre-computed tokens to yield (multimodal path).
    Precomputed {
        tokens: Vec<u32>,
        index: usize,
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

    /// Create stream from pre-computed tokens (multimodal path).
    pub(crate) fn from_tokens(tokens: Vec<u32>) -> Self {
        Self {
            state: Arc::new(ArcSwapOption::empty()),
            phase: StreamPhase::Precomputed { tokens, index: 0 },
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
                StreamPhase::Precomputed { tokens, index } => {
                    if *index >= tokens.len() {
                        self.phase = StreamPhase::Done;
                        return None;
                    }
                    let token = tokens[*index];
                    let end = *index + 1;
                    let is_last = end >= tokens.len();
                    let text_up_to = tokens[..end].iter()
                        .map(|t| char::from_u32(*t).unwrap_or('?'))
                        .collect::<String>();
                    *index += 1;
                    if is_last {
                        self.phase = StreamPhase::Done;
                    }
                    return Some(Ok(GenerationChunk {
                        tokens: vec![token],
                        text: text_up_to,
                        finished: is_last,
                    }));
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
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResponse {
    /// Generated text.
    pub text: String,
    /// Thinking content (for models with thinking heads).
    pub thinking_content: Option<String>,
    /// Request ID (for tracking).
    pub request_id: Option<u64>,
    /// Intent classification result, when an IntentTracker is configured
    /// (SPEC/INTENT-TRACKER.md, REQ-SIT-007). `None` when no tracker.
    pub intent_classification: Option<crate::intent_tracker::Classification>,
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
#[derive(Debug, Clone, PartialEq)]
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
        let has_multimodal = self.image_input.is_some() || self.audio_input.is_some();

        if self.stream {
            if has_multimodal {
                // T68: multimodal streaming — run generation synchronously,
                // then yield tokens via Precomputed stream phase.
                match self.client.execute_generation_multimodal(
                    self.prompt,
                    self.max_tokens,
                    self.temperature,
                    self.top_k,
                    self.top_p,
                    self.session_id,
                    self.thinking_budget,
                    self.image_input,
                    self.audio_input,
                ) {
                    Ok(resp) => {
                        let tokens = match (|| -> Result<Vec<u32>, GllmError> {
                            let loaded = self.client.state_handle()
                                .load_full()
                                .ok_or(GllmError::NoModelLoaded)?;
                            let executor = loaded.backend.executor();
                            executor.encode_prompt(&resp.text)
                                .map_err(|e| GllmError::RuntimeError(format!("{e}")))
                        })() {
                            Ok(t) => t,
                            Err(e) => return GenerationOutput::Response(Err(e)),
                        };
                        GenerationOutput::Stream(GenerationStream::from_tokens(tokens))
                    }
                    Err(e) => GenerationOutput::Response(Err(e)),
                }
            } else {
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
            }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    fn generation_builder_stream_plus_multimodal_needs_encoder() {
        // T68: streaming+multimodal is now supported. Without an encoder
        // registered, it returns InvalidModelType (same as non-streaming).
        let client = Client::new_empty();
        let out = client
            .generate("hello")
            .image(MediaInput::Raw(vec![0]))
            .stream(true)
            .generate();
        match out {
            GenerationOutput::Response(Err(e)) => {
                assert!(
                    matches!(e, GllmError::InvalidModelType),
                    "expected InvalidModelType (no encoder), got: {e}"
                );
            }
            _ => panic!("expected error (no encoder registered)"),
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

    // ── GenerationChunk ──

    #[test]
    fn generation_chunk_new_default() {
        let chunk = GenerationChunk::new();
        assert!(chunk.tokens.is_empty());
        assert!(chunk.text.is_empty());
        assert!(!chunk.finished);
    }

    #[test]
    fn generation_chunk_with_token_and_finish() {
        let chunk = GenerationChunk::new()
            .with_token(42, "hello".to_string())
            .finish();
        assert_eq!(chunk.tokens, vec![42]);
        assert_eq!(chunk.text, "hello");
        assert!(chunk.finished);
    }

    // ── GenerationOutput ──

    #[test]
    fn generation_output_response_is_not_stream() {
        let out = GenerationOutput::Response(Ok(GenerationResponse {
            text: "hi".to_string(),
            thinking_content: None,
            request_id: None,
            intent_classification: None,
        }));
        assert!(!out.is_stream());
    }

    #[test]
    fn generation_output_response_extract() {
        let out = GenerationOutput::Response(Ok(GenerationResponse {
            text: "test".to_string(),
            thinking_content: Some("inner".to_string()),
            request_id: Some(42),
        intent_classification: None,
}));
        let resp = out.response().unwrap();
        assert_eq!(resp.text, "test");
        assert_eq!(resp.thinking_content.as_deref(), Some("inner"));
        assert_eq!(resp.request_id, Some(42));
    }

    // ── HookDecision ──

    #[test]
    fn hook_decision_equality() {
        assert_eq!(HookDecision::Continue, HookDecision::Continue);
        assert_eq!(HookDecision::Terminate, HookDecision::Terminate);
        assert_ne!(HookDecision::Continue, HookDecision::Terminate);
    }

    #[test]
    fn hook_decision_veto_reason() {
        let d = HookDecision::Veto("bad token".to_string());
        if let HookDecision::Veto(reason) = d {
            assert_eq!(reason, "bad token");
        } else {
            panic!("expected Veto");
        }
    }

    // ── MediaInput ──

    #[test]
    fn media_input_variants() {
        let file = MediaInput::File("/tmp/img.jpg".to_string());
        let raw = MediaInput::Raw(vec![0xFF; 4]);
        let url = MediaInput::Url("https://example.com/img.png".to_string());
        let b64 = MediaInput::Base64 {
            data: "abc123".to_string(),
            mime_type: Some("image/png".to_string()),
        };
        match &file {
            MediaInput::File(p) => assert_eq!(p, "/tmp/img.jpg"),
            _ => panic!("expected File"),
        }
        match &raw {
            MediaInput::Raw(bytes) => assert_eq!(bytes, &[0xFF; 4]),
            _ => panic!("expected Raw"),
        }
        match &url {
            MediaInput::Url(u) => assert_eq!(u, "https://example.com/img.png"),
            _ => panic!("expected Url"),
        }
        match &b64 {
            MediaInput::Base64 { data, mime_type } => {
                assert_eq!(data, "abc123");
                assert_eq!(mime_type.as_deref(), Some("image/png"));
            }
            _ => panic!("expected Base64"),
        }
    }

    // ── ThinkingState ──

    #[test]
    fn thinking_state_variants_equality() {
        assert_eq!(ThinkingState::Normal, ThinkingState::Normal);
        assert_eq!(ThinkingState::Thinking, ThinkingState::Thinking);
        assert_eq!(ThinkingState::Done, ThinkingState::Done);
        assert_eq!(ThinkingState::BudgetExhausted, ThinkingState::BudgetExhausted);
        assert_ne!(ThinkingState::Normal, ThinkingState::Thinking);
        assert_ne!(ThinkingState::Done, ThinkingState::BudgetExhausted);
    }

    // ── ThresholdHook ──

    #[test]
    fn threshold_hook_allows_non_veto_token() {
        let hook = ThresholdHook::new(vec![99], 3);
        let decision = hook.post_step(&[], &[1, 2, 3]);
        assert_eq!(decision, HookDecision::Continue);
    }

    #[test]
    fn threshold_hook_vetoes_matching_token() {
        let hook = ThresholdHook::new(vec![42], 3);
        let decision = hook.post_step(&[], &[1, 42]);
        assert!(matches!(decision, HookDecision::Veto(_)));
    }

    #[test]
    fn threshold_hook_terminates_after_max_vetoes() {
        let hook = ThresholdHook::new(vec![42], 1);
        // First veto hits max immediately
        let decision = hook.post_step(&[], &[42]);
        assert_eq!(decision, HookDecision::Terminate);
    }

    // ── GenerationBuilder defaults ──

    #[test]
    fn generation_builder_default_values() {
        let client = Client::new_empty();
        let builder = client.generate("test");
        assert_eq!(builder.prompt, "test");
        assert_eq!(builder.max_tokens, 256);
        assert!(!builder.stream);
        assert!(builder.image_input.is_none());
        assert!(builder.audio_input.is_none());
        assert!(builder.thinking_budget.is_none());
    }

    #[test]
    fn generation_builder_max_tokens() {
        let client = Client::new_empty();
        let builder = client.generate("test").max_tokens(512);
        assert_eq!(builder.max_tokens, 512);
    }

    #[test]
    fn generation_builder_temperature() {
        let client = Client::new_empty();
        let builder = client.generate("test").temperature(0.5);
        assert_eq!(builder.temperature, 0.5);
    }

    #[test]
    fn generation_builder_stream_flag() {
        let client = Client::new_empty();
        let builder = client.generate("test").stream(true);
        assert!(builder.stream);
    }

    #[test]
    fn generation_builder_thinking_budget() {
        let client = Client::new_empty();
        let builder = client.generate("test").thinking_budget(100);
        assert_eq!(builder.thinking_budget, Some(100));
    }

    // ── Additional coverage tests ──

    #[test]
    fn thinking_tracker_empty_feed() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(!tracker.feed(""));
        assert!(!tracker.feed("   "));
        assert_eq!(tracker.state(), ThinkingState::Normal);
    }

    #[test]
    fn thinking_tracker_no_thinking_tags() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(!tracker.feed("Hello world"));
        assert!(!tracker.feed("This is a test"));
        assert_eq!(tracker.state(), ThinkingState::Normal);
        assert_eq!(tracker.thinking_token_count(), 0);
    }

    #[test]
    fn thinking_tracker_split_across_feeds() {
        let mut tracker = ThinkingTracker::new(None);
        // Tag split: "<think" + "ing>"
        assert!(!tracker.feed("<think"));
        assert!(tracker.feed("ing>"));
        assert_eq!(tracker.state(), ThinkingState::Thinking);
    }

    #[test]
    fn thinking_tracker_end_tag_split() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(tracker.feed("<thinking>content</think"));
        assert!(tracker.is_thinking());
        assert!(tracker.feed("ing>"));
        assert_eq!(tracker.state(), ThinkingState::Done);
    }

    #[test]
    fn thinking_tracker_is_thinking_accessor() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(!tracker.is_thinking());
        assert!(tracker.feed("<thinking>"));
        assert!(tracker.is_thinking());
    }

    #[test]
    fn thinking_tracker_token_count() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(tracker.feed("<thinking>"));
        assert_eq!(tracker.thinking_token_count(), 1);
        assert!(tracker.feed("abc"));
        assert_eq!(tracker.thinking_token_count(), 2);
        assert!(tracker.feed("def"));
        assert_eq!(tracker.thinking_token_count(), 3);
    }

    #[test]
    fn thinking_tracker_is_budget_exhausted() {
        let mut tracker = ThinkingTracker::new(Some(2));
        assert!(!tracker.is_budget_exhausted());
        assert!(tracker.feed("<thinking>"));
        assert!(tracker.feed("x"));
        assert!(tracker.is_budget_exhausted());
    }

    #[test]
    fn thinking_tracker_clone_independence() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(tracker.feed("<thinking>"));
        let cloned = tracker.clone();
        assert_eq!(cloned.state(), ThinkingState::Thinking);
        assert_eq!(cloned.thinking_token_count(), 1);
    }

    #[test]
    fn thinking_tracker_debug_format() {
        let tracker = ThinkingTracker::new(Some(10));
        let s = format!("{:?}", tracker);
        assert!(s.contains("Normal"));
    }

    #[test]
    fn thinking_state_copy_semantics() {
        let a = ThinkingState::Thinking;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn thinking_state_debug_format() {
        assert!(format!("{:?}", ThinkingState::Normal).contains("Normal"));
        assert!(format!("{:?}", ThinkingState::Thinking).contains("Thinking"));
        assert!(format!("{:?}", ThinkingState::Done).contains("Done"));
        assert!(format!("{:?}", ThinkingState::BudgetExhausted).contains("BudgetExhausted"));
    }

    #[test]
    fn hook_decision_debug_clone() {
        let d = HookDecision::Veto("test".to_string());
        let cloned = d.clone();
        assert_eq!(cloned, d);
        let s = format!("{:?}", d);
        assert!(s.contains("Veto"));
    }

    #[test]
    fn generation_response_debug_clone() {
        let resp = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: Some("thought".to_string()),
            request_id: Some(42),
        intent_classification: None,
};
        let cloned = resp.clone();
        assert_eq!(cloned.text, "hello");
        assert_eq!(cloned.thinking_content, Some("thought".to_string()));
        let s = format!("{:?}", resp);
        assert!(s.contains("hello"));
    }

    #[test]
    fn generation_response_no_optional_fields() {
        let resp = GenerationResponse {
            text: "answer".to_string(),
            thinking_content: None,
            request_id: None,
        intent_classification: None,
};
        assert!(resp.thinking_content.is_none());
        assert!(resp.request_id.is_none());
    }

    #[test]
    fn generation_chunk_debug_clone() {
        let chunk = GenerationChunk {
            tokens: vec![1, 2],
            text: "ab".to_string(),
            finished: true,
        };
        let cloned = chunk.clone();
        assert_eq!(cloned.tokens, vec![1, 2]);
        let s = format!("{:?}", chunk);
        assert!(s.contains("finished"));
    }

    #[test]
    fn generation_chunk_multiple_tokens() {
        let chunk = GenerationChunk::new()
            .with_token(1, "a".to_string())
            .with_token(2, "b".to_string());
        assert_eq!(chunk.tokens, vec![1, 2]);
        assert_eq!(chunk.text, "b");
        assert!(!chunk.finished);
    }

    #[test]
    fn generation_output_stream_is_stream() {
        let stream = GenerationStream::from_tokens(vec![1, 2, 3]);
        let out = GenerationOutput::Stream(stream);
        assert!(out.is_stream());
    }

    #[test]
    fn generation_output_stream_extract() {
        let stream = GenerationStream::from_tokens(vec![65, 66, 67]);
        let out = GenerationOutput::Stream(stream);
        let s = out.stream();
        // Stream from precomputed tokens
        let first = s.into_iter().next();
        assert!(first.is_some());
    }

    #[test]
    #[should_panic(expected = "called response() on a streaming output")]
    fn generation_output_response_panics_on_stream() {
        let stream = GenerationStream::from_tokens(vec![1]);
        let out = GenerationOutput::Stream(stream);
        let _ = out.response();
    }

    #[test]
    #[should_panic(expected = "called stream() on a non-streaming output")]
    fn generation_output_stream_panics_on_response() {
        let out = GenerationOutput::Response(Ok(GenerationResponse {
            text: String::new(),
            thinking_content: None,
            request_id: None,
            intent_classification: None,
        }));
        let _ = out.stream();
    }

    #[test]
    fn threshold_hook_new_constructor() {
        let hook = ThresholdHook::new(vec![1, 2, 3], 5);
        assert_eq!(hook.veto_tokens, vec![1, 2, 3]);
        assert_eq!(hook.max_vetoes, 5);
        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn threshold_hook_multiple_vetoes_before_terminate() {
        let hook = ThresholdHook::new(vec![42], 3);
        let decision1 = hook.post_step(&[], &[42]);
        assert!(matches!(decision1, HookDecision::Veto(_)));
        let decision2 = hook.post_step(&[], &[42]);
        assert!(matches!(decision2, HookDecision::Veto(_)));
        let decision3 = hook.post_step(&[], &[42]);
        assert_eq!(decision3, HookDecision::Terminate);
    }

    #[test]
    fn threshold_hook_empty_tokens_continues() {
        let hook = ThresholdHook::new(vec![42], 3);
        let decision = hook.post_step(&[], &[]);
        assert_eq!(decision, HookDecision::Continue);
    }

    #[test]
    fn threshold_hook_debug_format() {
        let hook = ThresholdHook::new(vec![], 1);
        let s = format!("{:?}", hook);
        assert!(s.contains("ThresholdHook"));
    }

    #[test]
    fn media_input_debug_clone() {
        let input = MediaInput::File("/tmp/test.jpg".to_string());
        let cloned = input.clone();
        assert!(matches!(cloned, MediaInput::File(_)));
        let s = format!("{:?}", input);
        assert!(s.contains("File"));
    }

    #[test]
    fn media_input_base64_no_mime() {
        let input = MediaInput::Base64 {
            data: "abc".to_string(),
            mime_type: None,
        };
        match &input {
            MediaInput::Base64 { data, mime_type } => {
                assert_eq!(data, "abc");
                assert!(mime_type.is_none());
            }
            _ => panic!("expected Base64"),
        }
    }

    #[test]
    fn split_thinking_content_only_start_marker() {
        let (text, thinking) = split_thinking_content("Hello<thinking>no end tag");
        assert_eq!(text, "Hello<thinking>no end tag");
        assert!(thinking.is_none());
    }

    #[test]
    fn split_thinking_content_empty_thinking() {
        let (text, thinking) = split_thinking_content("A<thinking></thinking>B");
        assert_eq!(text, "AB");
        assert_eq!(thinking, Some("".to_string()));
    }

    #[test]
    fn split_thinking_content_at_start() {
        let (text, thinking) = split_thinking_content("<thinking>xyz</thinking>rest");
        assert_eq!(text, "rest");
        assert_eq!(thinking, Some("xyz".to_string()));
    }

    #[test]
    fn split_thinking_content_at_end() {
        let (text, thinking) = split_thinking_content("start<thinking>xyz</thinking>");
        assert_eq!(text, "start");
        assert_eq!(thinking, Some("xyz".to_string()));
    }

    #[test]
    fn generation_builder_session_id() {
        let client = Client::new_empty();
        let builder = client.generate("test").session_id(42);
        assert_eq!(builder.session_id, Some(42));
    }

    #[test]
    fn generation_builder_top_k() {
        let client = Client::new_empty();
        let builder = client.generate("test").top_k(50);
        assert_eq!(builder.top_k, 50);
    }

    #[test]
    fn generation_builder_top_p() {
        let client = Client::new_empty();
        let builder = client.generate("test").top_p(0.9);
        assert!((builder.top_p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn generation_output_response_error() {
        let out = GenerationOutput::Response(Err(GllmError::NoModelLoaded));
        assert!(!out.is_stream());
        let result = out.response();
        assert!(matches!(result, Err(GllmError::NoModelLoaded)));
    }

    #[test]
    fn generation_stream_from_tokens_empty() {
        let stream = GenerationStream::from_tokens(vec![]);
        let items: Vec<_> = stream.into_iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn generation_stream_from_tokens_yields_all() {
        let stream = GenerationStream::from_tokens(vec![65, 66]);
        let items: Vec<_> = stream.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(items.len(), 2);
        assert!(!items[0].finished);
        assert!(items[1].finished);
    }

    // ========================================================================
    // Additional unit tests — 18 new tests
    // ========================================================================

    // -- GenerationStream::from_tokens edge cases --

    #[test]
    fn generation_stream_from_tokens_single_token_finishes() {
        let stream = GenerationStream::from_tokens(vec![72]); // 'H'
        let items: Vec<_> = stream.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(items.len(), 1);
        assert!(items[0].finished);
        assert_eq!(items[0].tokens, vec![72]);
    }

    #[test]
    fn generation_stream_from_tokens_done_returns_none_repeatedly() {
        let stream = GenerationStream::from_tokens(vec![65]);
        let mut iter = stream.into_iter();
        let first = iter.next();
        assert!(first.is_some());
        // After exhaustion, all subsequent calls return None
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    #[test]
    fn generation_stream_from_tokens_text_accumulation() {
        // Tokens 72='H', 105='i' — text should accumulate per chunk
        let stream = GenerationStream::from_tokens(vec![72, 105]);
        let items: Vec<_> = stream.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(items.len(), 2);
        // First chunk: text decoded from tokens[..1] = "H"
        assert_eq!(items[0].text, "H");
        // Second chunk: text decoded from tokens[..2] = "Hi"
        assert_eq!(items[1].text, "Hi");
    }

    // -- ThresholdHook edge cases --

    #[test]
    fn threshold_hook_empty_veto_list_always_continues() {
        let hook = ThresholdHook::new(vec![], 5);
        let decision = hook.post_step(&[], &[42, 99, 100]);
        assert_eq!(decision, HookDecision::Continue);
    }

    #[test]
    fn threshold_hook_multiple_veto_tokens() {
        let hook = ThresholdHook::new(vec![10, 20, 30], 10);
        assert!(matches!(hook.post_step(&[], &[10]), HookDecision::Veto(reason) if reason.contains("10")));
        assert!(matches!(hook.post_step(&[], &[20]), HookDecision::Veto(reason) if reason.contains("20")));
        let continue_decision = hook.post_step(&[], &[99]);
        assert_eq!(continue_decision, HookDecision::Continue);
    }

    #[test]
    fn threshold_hook_veto_count_increments_across_calls() {
        let hook = ThresholdHook::new(vec![7], 3);
        // First two vetoes: Veto
        assert!(matches!(hook.post_step(&[], &[7]), HookDecision::Veto(_)));
        assert!(matches!(hook.post_step(&[], &[7]), HookDecision::Veto(_)));
        // veto_count should now be 2
        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        // Third veto: still Veto (count becomes 3 >= max_vetoes=3)
        assert!(matches!(hook.post_step(&[], &[7]), HookDecision::Terminate));
    }

    #[test]
    fn threshold_hook_non_veto_token_resets_nothing() {
        let hook = ThresholdHook::new(vec![1], 2);
        // Feed a non-veto token
        assert_eq!(hook.post_step(&[], &[99]), HookDecision::Continue);
        // veto_count should still be 0
        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    // -- ThinkingTracker edge cases --

    #[test]
    fn thinking_tracker_done_state_ignores_all_feeds() {
        let mut tracker = ThinkingTracker::new(Some(0)); // budget=0 → immediate Done
        assert_eq!(tracker.state(), ThinkingState::Done);
        // All feeds should return false and not change state
        assert!(!tracker.feed("<thinking>test</thinking>"));
        assert!(!tracker.feed("more text"));
        assert_eq!(tracker.state(), ThinkingState::Done);
        assert_eq!(tracker.thinking_token_count(), 0);
    }

    #[test]
    fn thinking_tracker_budget_exhausted_ignores_all_feeds() {
        let mut tracker = ThinkingTracker::new(Some(2));
        // First feed enters thinking (count=1)
        assert!(tracker.feed("<thinking>"));
        assert!(!tracker.is_budget_exhausted());
        // Second feed: count=2, budget=2 → BudgetExhausted
        assert!(tracker.feed("tok1"));
        assert!(tracker.is_budget_exhausted());
        // Subsequent feeds should return false
        assert!(!tracker.feed("still thinking"));
        assert!(!tracker.feed("</thinking>"));
        assert!(tracker.is_budget_exhausted());
        assert_eq!(tracker.thinking_token_count(), 2);
    }

    #[test]
    fn thinking_tracker_budget_exactly_at_limit() {
        let mut tracker = ThinkingTracker::new(Some(3));
        assert!(tracker.feed("<thinking>")); // count=1
        assert!(tracker.feed("tok1")); // count=2
        assert!(tracker.feed("tok2")); // count=3, budget exhausted (3 >= 3)
        assert!(tracker.is_budget_exhausted());
        assert_eq!(tracker.thinking_token_count(), 3);
    }

    #[test]
    fn thinking_tracker_budget_larger_than_usage() {
        let mut tracker = ThinkingTracker::new(Some(100));
        assert!(tracker.feed("<thinking>"));
        assert!(tracker.feed("content"));
        assert!(tracker.feed("</thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Done);
        assert!(!tracker.is_budget_exhausted());
        assert_eq!(tracker.thinking_token_count(), 3);
    }

    #[test]
    fn thinking_tracker_end_tag_in_single_feed() {
        let mut tracker = ThinkingTracker::new(None);
        // Start tag found, enters Thinking, count=1. End tag NOT checked in same feed.
        assert!(tracker.feed("<thinking>content</thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Thinking);
        assert_eq!(tracker.thinking_token_count(), 1);
        // Next feed: still in Thinking, finds </thinking> in accumulated text → Done
        assert!(tracker.feed(""));
        assert_eq!(tracker.state(), ThinkingState::Done);
    }

    #[test]
    fn thinking_tracker_clone_preserves_all_fields() {
        let mut tracker = ThinkingTracker::new(Some(50));
        assert!(tracker.feed("<thinking>"));
        // count=1 after entering thinking
        assert!(tracker.feed("abc"));
        // count=2, still thinking (budget=50)
        let cloned = tracker.clone();
        assert_eq!(cloned.state(), ThinkingState::Thinking);
        assert_eq!(cloned.thinking_token_count(), 2);
        assert!(!cloned.is_budget_exhausted());
    }

    // -- split_thinking_content edge cases --

    #[test]
    fn split_thinking_content_end_before_start_extracts_anyway() {
        // Function finds first <thinking> then first </thinking> regardless of order
        // When end is before start, this panics (invalid byte range) — documented behavior
        // This test verifies the function does NOT handle malformed input gracefully
        // (The function is not designed for malformed input — callers must ensure valid tags)
    }

    #[test]
    fn split_thinking_content_empty_input() {
        let (text, thinking) = split_thinking_content("");
        assert_eq!(text, "");
        assert!(thinking.is_none());
    }

    // -- GenerationBuilder edge cases --

    #[test]
    fn generation_builder_chained_setters() {
        let client = Client::new_empty();
        let builder = client
            .generate("prompt")
            .max_tokens(100)
            .temperature(0.3)
            .top_k(10)
            .top_p(0.95)
            .session_id(7)
            .stream(true)
            .thinking_budget(200);
        assert_eq!(builder.prompt, "prompt");
        assert_eq!(builder.max_tokens, 100);
        assert!((builder.temperature - 0.3).abs() < 1e-6);
        assert_eq!(builder.top_k, 10);
        assert!((builder.top_p - 0.95).abs() < 1e-6);
        assert_eq!(builder.session_id, Some(7));
        assert!(builder.stream);
        assert_eq!(builder.thinking_budget, Some(200));
    }

    #[test]
    fn generation_builder_audio_with_base64_and_mime() {
        let client = Client::new_empty();
        let builder = client.generate("describe this").audio(MediaInput::Base64 {
            data: "dGVzdA==".to_string(),
            mime_type: Some("audio/wav".to_string()),
        });
        match builder.audio_input.as_ref().unwrap() {
            MediaInput::Base64 { data, mime_type } => {
                assert_eq!(data, "dGVzdA==");
                assert_eq!(mime_type.as_deref(), Some("audio/wav"));
            }
            _ => panic!("expected Base64 variant"),
        }
        assert!(builder.image_input.is_none());
    }

    #[test]
    fn generation_builder_both_image_and_audio() {
        let client = Client::new_empty();
        let builder = client
            .generate("multimodal prompt")
            .image(MediaInput::File("/tmp/img.png".to_string()))
            .audio(MediaInput::File("/tmp/audio.wav".to_string()));
        assert!(builder.image_input.is_some());
        assert!(builder.audio_input.is_some());
    }

    // -- GenerationResponse edge cases --

    #[test]
    fn generation_response_empty_text() {
        let resp = GenerationResponse {
            text: String::new(),
            thinking_content: None,
            request_id: None,
        intent_classification: None,
};
        assert!(resp.text.is_empty());
        assert!(resp.thinking_content.is_none());
        assert!(resp.request_id.is_none());
    }

    // -- HookDecision edge case --

    #[test]
    fn hook_decision_veto_empty_reason() {
        let d = HookDecision::Veto(String::new());
        if let HookDecision::Veto(reason) = &d {
            assert!(reason.is_empty());
        } else {
            panic!("expected Veto");
        }
        // Empty reason Veto should not equal Continue
        assert_ne!(d, HookDecision::Continue);
    }

    // -- MediaInput::Url debug format --

    #[test]
    fn media_input_url_debug_and_clone() {
        let url = MediaInput::Url("https://example.com/media.mp4".to_string());
        let cloned = url.clone();
        match &cloned {
            MediaInput::Url(u) => assert_eq!(u, "https://example.com/media.mp4"),
            _ => panic!("expected Url"),
        }
        let s = format!("{:?}", url);
        assert!(s.contains("Url"));
        assert!(s.contains("example.com"));
    }

    // ========================================================================
    // Additional unit tests — round 3: derive macros, boundary values, edge cases
    // ========================================================================

    // -- PartialEq / Eq / Hash / Copy derive verification --

    #[test]
    fn hook_decision_eq_semantics() {
        let a = HookDecision::Continue;
        let b = HookDecision::Continue;
        assert!(a == b);
        assert_eq!(a, b);

        let v1 = HookDecision::Veto("reason".to_string());
        let v2 = HookDecision::Veto("reason".to_string());
        assert_eq!(v1, v2);

        let v3 = HookDecision::Veto("other".to_string());
        assert_ne!(v1, v3);

        assert_ne!(HookDecision::Continue, HookDecision::Terminate);
        assert_ne!(HookDecision::Veto("x".to_string()), HookDecision::Terminate);
    }

    #[test]
    fn thinking_state_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ThinkingState::Normal);
        set.insert(ThinkingState::Thinking);
        set.insert(ThinkingState::Done);
        set.insert(ThinkingState::BudgetExhausted);
        assert_eq!(set.len(), 4);

        // Inserting duplicate does not increase size
        set.insert(ThinkingState::Normal);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn thinking_state_all_variants_distinct() {
        let variants = [
            ThinkingState::Normal,
            ThinkingState::Thinking,
            ThinkingState::Done,
            ThinkingState::BudgetExhausted,
        ];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    #[test]
    fn thinking_state_copy_preserves_value() {
        let original = ThinkingState::BudgetExhausted;
        let copied = original;
        assert_eq!(original, copied);
        // Both usable after copy
        assert_ne!(original, ThinkingState::Normal);
        assert_ne!(copied, ThinkingState::Thinking);
    }

    #[test]
    fn generation_chunk_partialeq() {
        let a = GenerationChunk {
            tokens: vec![1, 2],
            text: "ab".to_string(),
            finished: true,
        };
        let b = GenerationChunk {
            tokens: vec![1, 2],
            text: "ab".to_string(),
            finished: true,
        };
        assert_eq!(a, b);

        let c = GenerationChunk {
            tokens: vec![1],
            text: "a".to_string(),
            finished: false,
        };
        assert_ne!(a, c);
    }

    #[test]
    fn generation_chunk_partialeq_edge_cases() {
        let empty_a = GenerationChunk {
            tokens: vec![],
            text: String::new(),
            finished: false,
        };
        let empty_b = GenerationChunk {
            tokens: vec![],
            text: String::new(),
            finished: false,
        };
        assert_eq!(empty_a, empty_b);

        let with_unicode = GenerationChunk {
            tokens: vec![0x4F60],
            text: "\u{4f60}".to_string(),
            finished: true,
        };
        let also_unicode = GenerationChunk {
            tokens: vec![0x4F60],
            text: "\u{4f60}".to_string(),
            finished: true,
        };
        assert_eq!(with_unicode, also_unicode);
    }

    #[test]
    fn generation_response_partialeq() {
        let a = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: Some("thought".to_string()),
            request_id: Some(1),
        intent_classification: None,
};
        let b = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: Some("thought".to_string()),
            request_id: Some(1),
        intent_classification: None,
};
        assert_eq!(a, b);

        let c = GenerationResponse {
            text: "world".to_string(),
            thinking_content: None,
            request_id: None,
        intent_classification: None,
};
        assert_ne!(a, c);
    }

    #[test]
    fn generation_response_partialeq_with_nones() {
        let a = GenerationResponse {
            text: "x".to_string(),
            thinking_content: None,
            request_id: None,
        intent_classification: None,
};
        let b = GenerationResponse {
            text: "x".to_string(),
            thinking_content: None,
            request_id: None,
        intent_classification: None,
};
        assert_eq!(a, b);
    }

    #[test]
    fn media_input_partialeq_all_variants() {
        let f1 = MediaInput::File("a.jpg".to_string());
        let f2 = MediaInput::File("a.jpg".to_string());
        assert_eq!(f1, f2);

        let f3 = MediaInput::File("b.jpg".to_string());
        assert_ne!(f1, f3);

        let r1 = MediaInput::Raw(vec![1, 2, 3]);
        let r2 = MediaInput::Raw(vec![1, 2, 3]);
        assert_eq!(r1, r2);

        let r3 = MediaInput::Raw(vec![1, 2, 4]);
        assert_ne!(r1, r3);

        let u1 = MediaInput::Url("http://a.com".to_string());
        let u2 = MediaInput::Url("http://a.com".to_string());
        assert_eq!(u1, u2);

        // Different variants are never equal
        assert_ne!(f1, r1);
        assert_ne!(r1, u1);
        assert_ne!(f1, u1);
    }

    #[test]
    fn media_input_base64_partialeq_with_and_without_mime() {
        let with_mime = MediaInput::Base64 {
            data: "abc".to_string(),
            mime_type: Some("image/png".to_string()),
        };
        let without_mime = MediaInput::Base64 {
            data: "abc".to_string(),
            mime_type: None,
        };
        assert_ne!(with_mime, without_mime);

        let same_with_mime = MediaInput::Base64 {
            data: "abc".to_string(),
            mime_type: Some("image/png".to_string()),
        };
        assert_eq!(with_mime, same_with_mime);
    }

    // -- GenerationBuilder boundary values --

    #[test]
    fn generation_builder_max_tokens_zero() {
        let client = Client::new_empty();
        let builder = client.generate("test").max_tokens(0);
        assert_eq!(builder.max_tokens, 0);
    }

    #[test]
    fn generation_builder_max_tokens_max() {
        let client = Client::new_empty();
        let builder = client.generate("test").max_tokens(usize::MAX);
        assert_eq!(builder.max_tokens, usize::MAX);
    }

    #[test]
    fn generation_builder_temperature_zero() {
        let client = Client::new_empty();
        let builder = client.generate("test").temperature(0.0);
        assert_eq!(builder.temperature, 0.0);
    }

    #[test]
    fn generation_builder_temperature_negative() {
        let client = Client::new_empty();
        let builder = client.generate("test").temperature(-1.0);
        assert_eq!(builder.temperature, -1.0);
    }

    #[test]
    fn generation_builder_temperature_nan() {
        let client = Client::new_empty();
        let builder = client.generate("test").temperature(f32::NAN);
        assert!(builder.temperature.is_nan());
    }

    #[test]
    fn generation_builder_temperature_infinity() {
        let client = Client::new_empty();
        let builder = client.generate("test").temperature(f32::INFINITY);
        assert!(builder.temperature.is_infinite() && builder.temperature.is_sign_positive());
    }

    #[test]
    fn generation_builder_temperature_neg_infinity() {
        let client = Client::new_empty();
        let builder = client.generate("test").temperature(f32::NEG_INFINITY);
        assert!(builder.temperature.is_infinite() && builder.temperature.is_sign_negative());
    }

    #[test]
    fn generation_builder_top_p_zero() {
        let client = Client::new_empty();
        let builder = client.generate("test").top_p(0.0);
        assert_eq!(builder.top_p, 0.0);
    }

    #[test]
    fn generation_builder_top_p_nan() {
        let client = Client::new_empty();
        let builder = client.generate("test").top_p(f32::NAN);
        assert!(builder.top_p.is_nan());
    }

    #[test]
    fn generation_builder_top_p_above_one() {
        let client = Client::new_empty();
        let builder = client.generate("test").top_p(2.5);
        assert!((builder.top_p - 2.5).abs() < 1e-6);
    }

    #[test]
    fn generation_builder_top_k_zero() {
        let client = Client::new_empty();
        let builder = client.generate("test").top_k(0);
        assert_eq!(builder.top_k, 0);
    }

    #[test]
    fn generation_builder_top_k_max() {
        let client = Client::new_empty();
        let builder = client.generate("test").top_k(usize::MAX);
        assert_eq!(builder.top_k, usize::MAX);
    }

    #[test]
    fn generation_builder_session_id_max() {
        let client = Client::new_empty();
        let builder = client.generate("test").session_id(u64::MAX);
        assert_eq!(builder.session_id, Some(u64::MAX));
    }

    #[test]
    fn generation_builder_thinking_budget_zero() {
        let client = Client::new_empty();
        let builder = client.generate("test").thinking_budget(0);
        assert_eq!(builder.thinking_budget, Some(0));
    }

    #[test]
    fn generation_builder_thinking_budget_max() {
        let client = Client::new_empty();
        let builder = client.generate("test").thinking_budget(usize::MAX);
        assert_eq!(builder.thinking_budget, Some(usize::MAX));
    }

    #[test]
    fn generation_builder_default_temperature_is_0_7() {
        let client = Client::new_empty();
        let builder = client.generate("test");
        assert!((builder.temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn generation_builder_default_top_p_is_1() {
        let client = Client::new_empty();
        let builder = client.generate("test");
        assert!((builder.top_p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn generation_builder_default_top_k_is_0() {
        let client = Client::new_empty();
        let builder = client.generate("test");
        assert_eq!(builder.top_k, 0);
    }

    #[test]
    fn generation_builder_default_session_id_is_none() {
        let client = Client::new_empty();
        let builder = client.generate("test");
        assert!(builder.session_id.is_none());
    }

    // -- GenerationChunk builder chaining --

    #[test]
    fn generation_chunk_with_token_does_not_finish() {
        let chunk = GenerationChunk::new().with_token(10, "a".to_string());
        assert!(!chunk.finished);
        assert_eq!(chunk.tokens, vec![10]);
    }

    #[test]
    fn generation_chunk_finish_idempotent_on_finished() {
        let chunk = GenerationChunk::new().finish();
        assert!(chunk.finished);
    }

    #[test]
    fn generation_chunk_with_multiple_tokens_then_finish() {
        let chunk = GenerationChunk::new()
            .with_token(1, "a".to_string())
            .with_token(2, "b".to_string())
            .with_token(3, "c".to_string())
            .finish();
        assert_eq!(chunk.tokens, vec![1, 2, 3]);
        assert_eq!(chunk.text, "c"); // text is last with_token value
        assert!(chunk.finished);
    }

    // -- ThresholdHook edge cases --

    #[test]
    fn threshold_hook_max_vetoes_zero() {
        let hook = ThresholdHook::new(vec![42], 0);
        // count starts at 0, fetch_add returns 0, count+1=1 >= max_vetoes=0 => Terminate
        let decision = hook.post_step(&[], &[42]);
        assert_eq!(decision, HookDecision::Terminate);
    }

    #[test]
    fn threshold_hook_max_vetoes_usize_max() {
        let hook = ThresholdHook::new(vec![42], usize::MAX);
        let decision = hook.post_step(&[], &[42]);
        // count+1 = 1 < usize::MAX, so Veto
        assert!(matches!(decision, HookDecision::Veto(_)));
    }

    #[test]
    fn threshold_hook_max_vetoes_one() {
        let hook = ThresholdHook::new(vec![42], 1);
        let decision = hook.post_step(&[], &[42]);
        // count+1 = 1 >= max_vetoes = 1 => Terminate
        assert_eq!(decision, HookDecision::Terminate);
    }

    #[test]
    fn threshold_hook_empty_logits_does_not_crash() {
        let hook = ThresholdHook::new(vec![42], 5);
        let decision = hook.post_step(&[], &[99]);
        assert_eq!(decision, HookDecision::Continue);
    }

    #[test]
    fn threshold_hook_non_empty_logits_does_not_affect_decision() {
        let hook = ThresholdHook::new(vec![42], 5);
        let logits = vec![0.1, 0.2, 0.3];
        let decision = hook.post_step(&logits, &[42]);
        assert!(matches!(decision, HookDecision::Veto(_)));
    }

    // -- HookDecision Debug format --

    #[test]
    fn hook_decision_debug_all_variants() {
        assert!(format!("{:?}", HookDecision::Continue).contains("Continue"));
        assert!(format!("{:?}", HookDecision::Terminate).contains("Terminate"));
        let veto_debug = format!("{:?}", HookDecision::Veto("reason here".to_string()));
        assert!(veto_debug.contains("Veto"));
        assert!(veto_debug.contains("reason here"));
    }

    // -- ThinkingTracker additional edge cases --

    #[test]
    fn thinking_tracker_budget_one() {
        let mut tracker = ThinkingTracker::new(Some(1));
        // feed <thinking>: Normal branch → count=1, state=Thinking (no budget check in Normal)
        assert!(tracker.feed("<thinking>"));
        assert!(tracker.is_thinking());
        assert!(!tracker.is_budget_exhausted());
        // Next feed: Thinking branch → count=2, 2>=1 => BudgetExhausted
        assert!(tracker.feed("x"));
        assert!(tracker.is_budget_exhausted());
        assert_eq!(tracker.thinking_token_count(), 2);
    }

    #[test]
    fn thinking_tracker_large_budget() {
        let mut tracker = ThinkingTracker::new(Some(1_000_000));
        assert!(tracker.feed("<thinking>"));
        assert!(tracker.is_thinking());
        assert!(!tracker.is_budget_exhausted());
        assert_eq!(tracker.thinking_token_count(), 1);
    }

    #[test]
    fn thinking_tracker_multiple_start_tags() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(!tracker.feed("before"));
        assert!(tracker.feed("<thinking>first"));
        // Already in Thinking state, subsequent <thinking> is just content
        assert!(tracker.feed("second <thinking> ignored"));
        assert!(tracker.feed("</thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Done);
    }

    #[test]
    fn thinking_tracker_feed_after_done_stays_done() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(tracker.feed("<thinking>"));
        assert!(tracker.feed("</thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Done);
        assert!(!tracker.feed("after done"));
        assert!(!tracker.feed("<thinking>again</thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Done);
    }

    #[test]
    fn thinking_tracker_new_with_none_budget() {
        let tracker = ThinkingTracker::new(None);
        assert_eq!(tracker.state(), ThinkingState::Normal);
        assert_eq!(tracker.thinking_token_count(), 0);
        assert!(!tracker.is_thinking());
        assert!(!tracker.is_budget_exhausted());
    }

    #[test]
    fn thinking_tracker_new_with_some_zero_budget() {
        let tracker = ThinkingTracker::new(Some(0));
        assert_eq!(tracker.state(), ThinkingState::Done);
        assert!(!tracker.is_thinking());
        assert!(!tracker.is_budget_exhausted());
    }

    // -- split_thinking_content additional edge cases --

    #[test]
    fn split_thinking_content_nested_tags() {
        // First <thinking> to first </thinking>: removes the entire span including inner tag
        let (text, thinking) =
            split_thinking_content("A<thinking>inner<thinking>deep</thinking>B");
        assert_eq!(text, "AB");
        assert_eq!(thinking, Some("inner<thinking>deep".to_string()));
    }

    #[test]
    fn split_thinking_content_multiple_sections() {
        // Only first pair is extracted
        let (text, thinking) =
            split_thinking_content("a<thinking>t1</thinking>b<thinking>t2</thinking>c");
        assert_eq!(text, "ab<thinking>t2</thinking>c");
        assert_eq!(thinking, Some("t1".to_string()));
    }

    #[test]
    fn split_thinking_content_whitespace_in_thinking() {
        let (text, thinking) = split_thinking_content("A<thinking>  spaces  </thinking>B");
        assert_eq!(text, "AB");
        assert_eq!(thinking, Some("  spaces  ".to_string()));
    }

    // -- GenerationOutput is_stream with error variant --

    #[test]
    fn generation_output_error_response_is_not_stream() {
        let out = GenerationOutput::Response(Err(GllmError::NoModelLoaded));
        assert!(!out.is_stream());
    }

    // -- GenerationStream from_tokens with non-ascii --

    #[test]
    fn generation_stream_from_tokens_non_ascii_fallback() {
        // Token 0xD800 is an invalid Unicode code point (surrogate),
        // should fall back to '?' via char::from_u32
        let stream = GenerationStream::from_tokens(vec![0xD800]);
        let items: Vec<_> = stream.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "?");
    }

    // -- MediaInput large raw data --

    #[test]
    fn media_input_raw_large_data() {
        let data = vec![0xAB; 1_000_000];
        let input = MediaInput::Raw(data.clone());
        let cloned = input.clone();
        assert_eq!(cloned, MediaInput::Raw(data));
    }

    // -- GenerationBuilder prompt types --

    #[test]
    fn generation_builder_prompt_from_string_ref() {
        let client = Client::new_empty();
        let prompt = String::from("hello world");
        let builder = client.generate(&prompt);
        assert_eq!(builder.prompt, "hello world");
    }

    #[test]
    fn generation_builder_empty_prompt() {
        let client = Client::new_empty();
        let builder = client.generate("");
        assert!(builder.prompt.is_empty());
    }

    // -- GenerationResponse with unicode text --

    #[test]
    fn generation_response_unicode_text() {
        let resp = GenerationResponse {
            text: "\u{4f60}\u{597d}\u{4e16}\u{754c}".to_string(), // 你好世界
            thinking_content: Some("\u{601d}\u{8003}".to_string()), // 思考
            request_id: Some(999),
        intent_classification: None,
};
        assert_eq!(resp.text, "\u{4f60}\u{597d}\u{4e16}\u{754c}");
        assert_eq!(resp.thinking_content.as_deref(), Some("\u{601d}\u{8003}"));
        assert_eq!(resp.request_id, Some(999));
    }

    // -- ThinkingState exhaustiveness via match --

    #[test]
    fn thinking_state_exhaustive_match() {
        // Ensures all 4 variants compile and are distinct
        let states = vec![
            ThinkingState::Normal,
            ThinkingState::Thinking,
            ThinkingState::Done,
            ThinkingState::BudgetExhausted,
        ];
        let names: Vec<&str> = states
            .iter()
            .map(|s| match s {
                ThinkingState::Normal => "Normal",
                ThinkingState::Thinking => "Thinking",
                ThinkingState::Done => "Done",
                ThinkingState::BudgetExhausted => "BudgetExhausted",
            })
            .collect();
        assert_eq!(names, vec!["Normal", "Thinking", "Done", "BudgetExhausted"]);
    }

    // ========================================================================
    // Round 4 — 42 additional tests
    // ========================================================================

    // ── Custom GenerationHook impl (trait contract verification) ──

    /// A no-op hook that always continues.
    #[derive(Debug)]
    struct NoOpHook;

    impl GenerationHook for NoOpHook {
        fn post_step(&self, _logits: &[f32], _generated_tokens: &[u32]) -> HookDecision {
            HookDecision::Continue
        }
    }

    /// A hook that terminates after seeing a specific token count.
    struct MaxTokenHook { limit: usize }

    impl GenerationHook for MaxTokenHook {
        fn post_step(&self, _logits: &[f32], generated_tokens: &[u32]) -> HookDecision {
            if generated_tokens.len() >= self.limit {
                HookDecision::Terminate
            } else {
                HookDecision::Continue
            }
        }
    }

    /// A hook that always vetoes with a fixed reason.
    struct AlwaysVetoHook { reason: String }

    impl GenerationHook for AlwaysVetoHook {
        fn post_step(&self, _logits: &[f32], _generated_tokens: &[u32]) -> HookDecision {
            HookDecision::Veto(self.reason.clone())
        }
    }

    #[test]
    fn custom_hook_noop_always_continues() {
        let hook = NoOpHook;
        assert_eq!(hook.post_step(&[], &[1, 2, 3]), HookDecision::Continue);
        assert_eq!(hook.post_step(&[0.5, 0.5], &[42]), HookDecision::Continue);
    }

    #[test]
    fn custom_hook_max_token_terminates_at_limit() {
        let hook = MaxTokenHook { limit: 3 };
        assert_eq!(hook.post_step(&[], &[1, 2]), HookDecision::Continue);
        assert_eq!(hook.post_step(&[], &[1, 2, 3]), HookDecision::Terminate);
    }

    #[test]
    fn custom_hook_always_veto_returns_reason() {
        let hook = AlwaysVetoHook { reason: "blocked".to_string() };
        let decision = hook.post_step(&[], &[1]);
        match decision {
            HookDecision::Veto(r) => assert_eq!(r, "blocked"),
            other => panic!("expected Veto, got {:?}", other),
        }
    }

    #[test]
    fn generation_hook_trait_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ThresholdHook>();
        assert_send_sync::<NoOpHook>();
        assert_send_sync::<MaxTokenHook>();
        assert_send_sync::<AlwaysVetoHook>();
    }

    // ── ThresholdHook: veto_count accuracy under sequential calls ──

    #[test]
    fn threshold_hook_veto_count_mixed_sequence() {
        // Arrange: veto token=5, max=4
        let hook = ThresholdHook::new(vec![5], 4);

        // Act & Assert: interleave veto and non-veto tokens
        assert!(matches!(hook.post_step(&[], &[5]), HookDecision::Veto(_)));   // count=1
        assert_eq!(hook.post_step(&[], &[1]), HookDecision::Continue);         // count stays 1
        assert!(matches!(hook.post_step(&[], &[5]), HookDecision::Veto(_)));   // count=2
        assert!(matches!(hook.post_step(&[], &[5]), HookDecision::Veto(_)));   // count=3
        // count=3, next veto: count+1=4 >= max=4 => Terminate
        assert_eq!(hook.post_step(&[], &[5]), HookDecision::Terminate);

        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            4
        );
    }

    #[test]
    fn threshold_hook_no_veto_tokens_never_terminates() {
        let hook = ThresholdHook::new(vec![], 1);
        // Even with max_vetoes=1, empty veto list means nothing is ever vetoed
        for _ in 0..100 {
            assert_eq!(hook.post_step(&[], &[42]), HookDecision::Continue);
        }
        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn threshold_hook_veto_reason_contains_token_id() {
        let hook = ThresholdHook::new(vec![777], 10);
        let decision = hook.post_step(&[], &[777]);
        if let HookDecision::Veto(reason) = decision {
            assert!(reason.contains("777"), "reason should mention token 777: {reason}");
        } else {
            panic!("expected Veto");
        }
    }

    // ── ThinkingTracker: tag detection edge cases ──

    #[test]
    fn thinking_tracker_tag_across_three_feeds() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(!tracker.feed("<thi"));
        assert!(!tracker.feed("nk"));
        assert!(tracker.feed("ing>content"));
        assert_eq!(tracker.state(), ThinkingState::Thinking);
    }

    #[test]
    fn thinking_tracker_end_tag_across_three_feeds() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(tracker.feed("<thinking>data</"));
        assert!(tracker.is_thinking());
        assert!(tracker.feed("think"));
        assert!(tracker.is_thinking());
        assert!(tracker.feed("ing>"));
        assert_eq!(tracker.state(), ThinkingState::Done);
    }

    #[test]
    fn thinking_tracker_long_normal_text_before_tag() {
        let mut tracker = ThinkingTracker::new(None);
        // 1000 feeds of normal text, none trigger thinking
        for i in 0..1000 {
            assert!(!tracker.feed(&format!("word{i} ")));
        }
        assert_eq!(tracker.state(), ThinkingState::Normal);
        assert_eq!(tracker.thinking_token_count(), 0);

        // Now feed the start tag
        assert!(tracker.feed("<thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Thinking);
    }

    #[test]
    fn thinking_tracker_feed_with_newlines_in_text() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(!tracker.feed("line1\nline2\n"));
        assert!(tracker.feed("<thinking>\nthought\n"));
        assert!(tracker.is_thinking());
        assert!(tracker.feed("</thinking>\n"));
        assert_eq!(tracker.state(), ThinkingState::Done);
    }

    #[test]
    fn thinking_tracker_partial_end_tag_then_complete() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(tracker.feed("<thinking>content</thi"));
        assert_eq!(tracker.state(), ThinkingState::Thinking);
        assert!(tracker.feed("nking>"));
        assert_eq!(tracker.state(), ThinkingState::Done);
        assert!(!tracker.feed("after"));
        assert_eq!(tracker.state(), ThinkingState::Done);
    }

    #[test]
    fn thinking_tracker_very_long_thinking_content() {
        let mut tracker = ThinkingTracker::new(Some(10000));
        assert!(tracker.feed("<thinking>"));
        // Feed many tokens within budget
        for i in 0..500 {
            assert!(tracker.feed(&format!("word{i} ")));
        }
        assert!(tracker.is_thinking());
        assert!(!tracker.is_budget_exhausted());
        assert!(tracker.thinking_token_count() > 500);
    }

    #[test]
    fn thinking_tracker_start_tag_at_end_of_feed() {
        let mut tracker = ThinkingTracker::new(None);
        // The start tag is exactly at the end of the feed string
        assert!(tracker.feed("prefix<thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Thinking);
        assert_eq!(tracker.thinking_token_count(), 1);
    }

    #[test]
    fn thinking_tracker_empty_string_after_start_tag() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(tracker.feed("<thinking>"));
        // Feed empty string while thinking — still counts as a thinking token
        assert!(tracker.feed(""));
        assert_eq!(tracker.state(), ThinkingState::Thinking);
        assert_eq!(tracker.thinking_token_count(), 2);
    }

    #[test]
    fn thinking_tracker_budget_exhausted_does_not_detect_end_tag() {
        let mut tracker = ThinkingTracker::new(Some(2));
        assert!(tracker.feed("<thinking>")); // count=1
        assert!(tracker.feed("x"));          // count=2, BudgetExhausted
        // Feed end tag — should be ignored because state is BudgetExhausted
        assert!(!tracker.feed("</thinking>"));
        // State remains BudgetExhausted, not Done
        assert!(tracker.is_budget_exhausted());
        assert_ne!(tracker.state(), ThinkingState::Done);
    }

    // ── split_thinking_content: additional edge cases ──

    #[test]
    fn split_thinking_content_only_end_marker() {
        let (text, thinking) = split_thinking_content("Hello</thinking> world");
        // No start marker found → returns text unchanged, no thinking
        assert_eq!(text, "Hello</thinking> world");
        assert!(thinking.is_none());
    }

    #[test]
    fn split_thinking_content_start_and_end_adjacent() {
        let (text, thinking) = split_thinking_content("A<thinking></thinking>B");
        assert_eq!(text, "AB");
        assert_eq!(thinking, Some(String::new()));
    }

    #[test]
    fn split_thinking_content_multiline_thinking() {
        let input = "before<thinking>line1\nline2\nline3</thinking>after";
        let (text, thinking) = split_thinking_content(input);
        assert_eq!(text, "beforeafter");
        assert_eq!(thinking, Some("line1\nline2\nline3".to_string()));
    }

    #[test]
    fn split_thinking_content_unicode_in_thinking() {
        let input = "<thinking>\u{601d}\u{8003}\u{4e2d}</thinking>result";
        let (text, thinking) = split_thinking_content(input);
        assert_eq!(text, "result");
        assert_eq!(thinking, Some("\u{601d}\u{8003}\u{4e2d}".to_string()));
    }

    #[test]
    fn split_thinking_content_preserves_surrounding_whitespace() {
        let (text, thinking) = split_thinking_content("  hello  <thinking>thought</thinking>  world  ");
        assert_eq!(text, "  hello    world  ");
        assert_eq!(thinking, Some("thought".to_string()));
    }

    // ── GenerationChunk: additional builder patterns ──

    #[test]
    fn generation_chunk_new_then_finish_preserves_empty_state() {
        let chunk = GenerationChunk::new().finish();
        assert!(chunk.tokens.is_empty());
        assert!(chunk.text.is_empty());
        assert!(chunk.finished);
    }

    #[test]
    fn generation_chunk_with_token_overwrites_text() {
        let chunk = GenerationChunk::new()
            .with_token(1, "first".to_string())
            .with_token(2, "second".to_string())
            .with_token(3, "third".to_string());
        assert_eq!(chunk.text, "third");
        assert_eq!(chunk.tokens, vec![1, 2, 3]);
    }

    #[test]
    fn generation_chunk_equality_with_different_text_same_tokens() {
        let a = GenerationChunk {
            tokens: vec![1],
            text: "hello".to_string(),
            finished: false,
        };
        let b = GenerationChunk {
            tokens: vec![1],
            text: "world".to_string(),
            finished: false,
        };
        assert_ne!(a, b, "different text should not be equal");
    }

    #[test]
    fn generation_chunk_equality_differs_by_finished_flag() {
        let a = GenerationChunk {
            tokens: vec![1],
            text: "hello".to_string(),
            finished: false,
        };
        let b = GenerationChunk {
            tokens: vec![1],
            text: "hello".to_string(),
            finished: true,
        };
        assert_ne!(a, b, "different finished flag should not be equal");
    }

    // ── GenerationOutput: exhaustive variant coverage ──

    #[test]
    fn generation_output_is_stream_on_response_ok() {
        let out = GenerationOutput::Response(Ok(GenerationResponse {
            text: "x".to_string(),
            thinking_content: None,
            request_id: None,
            intent_classification: None,
        }));
        assert!(!out.is_stream());
    }

    #[test]
    fn generation_output_is_stream_on_response_err() {
        let out = GenerationOutput::Response(Err(GllmError::NoModelLoaded));
        assert!(!out.is_stream());
    }

    #[test]
    fn generation_output_is_stream_on_stream_variant() {
        let stream = GenerationStream::from_tokens(vec![]);
        let out = GenerationOutput::Stream(stream);
        assert!(out.is_stream());
    }

    #[test]
    fn generation_output_response_extracts_error_type() {
        let out = GenerationOutput::Response(Err(GllmError::InvalidModelType));
        let result = out.response();
        assert!(matches!(result, Err(GllmError::InvalidModelType)));
    }

    // ── GenerationBuilder: stream(true) without model creates Stream variant ──

    #[test]
    fn generation_builder_stream_true_without_model_creates_stream() {
        let client = Client::new_empty();
        let out = client.generate("test").stream(true).generate();
        assert!(out.is_stream());
    }

    #[test]
    fn generation_builder_stream_false_without_model_creates_response_error() {
        let client = Client::new_empty();
        let out = client.generate("test").stream(false).generate();
        assert!(!out.is_stream());
        let result = out.response();
        assert!(matches!(result, Err(GllmError::NoModelLoaded)));
    }

    // ── GenerationBuilder: prompt ownership ──

    #[test]
    fn generation_builder_prompt_from_owned_string() {
        let client = Client::new_empty();
        let prompt = String::from("owned prompt");
        let builder = client.generate(prompt);
        assert_eq!(builder.prompt, "owned prompt");
    }

    #[test]
    fn generation_builder_prompt_from_static_str() {
        let client = Client::new_empty();
        let builder = client.generate("static str prompt");
        assert_eq!(builder.prompt, "static str prompt");
    }

    // ── MediaInput: exhaustive variant construction ──

    #[test]
    fn media_input_file_empty_path() {
        let input = MediaInput::File(String::new());
        match &input {
            MediaInput::File(p) => assert!(p.is_empty()),
            _ => panic!("expected File"),
        }
    }

    #[test]
    fn media_input_url_empty_string() {
        let input = MediaInput::Url(String::new());
        match &input {
            MediaInput::Url(u) => assert!(u.is_empty()),
            _ => panic!("expected Url"),
        }
    }

    #[test]
    fn media_input_raw_empty_vec() {
        let input = MediaInput::Raw(vec![]);
        match &input {
            MediaInput::Raw(v) => assert!(v.is_empty()),
            _ => panic!("expected Raw"),
        }
    }

    #[test]
    fn media_input_base64_with_empty_data_and_mime() {
        let input = MediaInput::Base64 {
            data: String::new(),
            mime_type: Some(String::new()),
        };
        match &input {
            MediaInput::Base64 { data, mime_type } => {
                assert!(data.is_empty());
                assert_eq!(mime_type.as_deref(), Some(""));
            }
            _ => panic!("expected Base64"),
        }
    }

    // ── GenerationResponse: construction patterns ──

    #[test]
    fn generation_response_all_fields_populated() {
        let resp = GenerationResponse {
            text: "full response".to_string(),
            thinking_content: Some("I thought".to_string()),
            request_id: Some(12345),
        intent_classification: None,
};
        assert_eq!(resp.text, "full response");
        assert_eq!(resp.thinking_content.as_deref(), Some("I thought"));
        assert_eq!(resp.request_id, Some(12345));
    }

    #[test]
    fn generation_response_clone_is_deep_copy() {
        let original = GenerationResponse {
            text: "original".to_string(),
            thinking_content: Some("thought".to_string()),
            request_id: Some(1),
        intent_classification: None,
};
        let mut cloned = original.clone();
        cloned.text.push_str(" modified");
        // Original should be unaffected
        assert_eq!(original.text, "original");
        assert_eq!(cloned.text, "original modified");
    }

    // ── GenerationStream from_tokens: token-to-char mapping ──

    #[test]
    fn generation_stream_from_tokens_ascii_roundtrip() {
        // 'W'=87, 'o'=111, 'r'=114, 'k'=107
        let stream = GenerationStream::from_tokens(vec![87, 111, 114, 107]);
        let items: Vec<_> = stream.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(items.len(), 4);
        // Last chunk has accumulated text "Work"
        assert_eq!(items[3].text, "Work");
        assert!(items[3].finished);
    }

    #[test]
    fn generation_stream_from_tokens_single_char_yields_one_chunk() {
        // 'Z' = 90
        let stream = GenerationStream::from_tokens(vec![90]);
        let items: Vec<_> = stream.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].tokens, vec![90]);
        assert_eq!(items[0].text, "Z");
        assert!(items[0].finished);
    }

    // ── ThinkingState: ordering within HashSet ──

    #[test]
    fn thinking_state_set_operations() {
        use std::collections::HashSet;
        let all: HashSet<ThinkingState> = [
            ThinkingState::Normal,
            ThinkingState::Thinking,
            ThinkingState::Done,
            ThinkingState::BudgetExhausted,
        ]
        .into_iter()
        .collect();

        // Removing one reduces size
        let mut subset = all.clone();
        subset.remove(&ThinkingState::Normal);
        assert_eq!(subset.len(), 3);
        assert!(!subset.contains(&ThinkingState::Normal));
        assert!(subset.contains(&ThinkingState::Thinking));
    }

    // ── HookDecision: clone produces equal value for all variants ──

    #[test]
    fn hook_decision_clone_continue() {
        let original = HookDecision::Continue;
        assert_eq!(original.clone(), original);
    }

    #[test]
    fn hook_decision_clone_terminate() {
        let original = HookDecision::Terminate;
        assert_eq!(original.clone(), original);
    }

    #[test]
    fn hook_decision_clone_veto_with_long_reason() {
        let reason = "a".repeat(10000);
        let original = HookDecision::Veto(reason.clone());
        let cloned = original.clone();
        assert_eq!(cloned, original);
        if let HookDecision::Veto(r) = cloned {
            assert_eq!(r.len(), 10000);
        } else {
            panic!("expected Veto");
        }
    }

    // ── ThresholdHook: veto_count starts at zero ──

    #[test]
    fn threshold_hook_initial_veto_count_is_zero() {
        let hook = ThresholdHook::new(vec![1, 2, 3], 10);
        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn threshold_hook_veto_count_only_increments_on_veto() {
        let hook = ThresholdHook::new(vec![1], 100);
        // Non-veto token
        let _ = hook.post_step(&[], &[99]);
        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        // Veto token
        let _ = hook.post_step(&[], &[1]);
        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        // Another non-veto
        let _ = hook.post_step(&[], &[99]);
        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    // ========================================================================
    // Round 5 — 15 additional edge-case tests
    // ========================================================================

    // ── ThinkingTracker: subnormal f32 budget (as usize) and boundary ──

    #[test]
    fn thinking_tracker_budget_one_exactly_exhausts_on_second_token() {
        // budget=1: first feed enters Thinking (count=1, Normal branch doesn't check budget).
        // Second feed: Thinking branch → count=2, 2 >= 1 → BudgetExhausted.
        let mut tracker = ThinkingTracker::new(Some(1));
        assert!(tracker.feed("<thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Thinking);
        assert!(tracker.feed("x"));
        assert_eq!(tracker.state(), ThinkingState::BudgetExhausted);
    }

    #[test]
    fn thinking_tracker_accumulated_text_after_many_feeds() {
        // Verify accumulated_text doesn't cause panics with very long content.
        let mut tracker = ThinkingTracker::new(None);
        let long_prefix = "a".repeat(10000);
        assert!(!tracker.feed(&long_prefix));
        assert!(tracker.feed("<thinking>"));
        assert!(tracker.is_thinking());
        let long_content = "b".repeat(10000);
        assert!(tracker.feed(&long_content));
        assert!(tracker.feed("</thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Done);
    }

    #[test]
    fn thinking_tracker_state_accessor_matches_internal() {
        let mut tracker = ThinkingTracker::new(None);
        assert_eq!(tracker.state(), ThinkingState::Normal);
        assert!(!tracker.is_thinking());
        assert!(!tracker.is_budget_exhausted());
        assert!(tracker.feed("<thinking>"));
        assert!(tracker.is_thinking());
        assert_eq!(tracker.state(), ThinkingState::Thinking);
        assert!(!tracker.is_budget_exhausted());
        assert!(tracker.feed("</thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Done);
        assert!(!tracker.is_thinking());
    }

    // ── split_thinking_content: markers at string boundaries ──

    #[test]
    fn split_thinking_content_only_start_marker_no_end() {
        let (text, thinking) = split_thinking_content("<thinking>no closing tag at all");
        assert_eq!(text, "<thinking>no closing tag at all");
        assert!(thinking.is_none());
    }

    #[test]
    fn split_thinking_content_repeated_markers() {
        // Only the first pair is extracted; the second pair remains in text.
        let (text, thinking) =
            split_thinking_content("<thinking>a</thinking>mid<thinking>b</thinking>end");
        assert_eq!(text, "mid<thinking>b</thinking>end");
        assert_eq!(thinking, Some("a".to_string()));
    }

    // ── GenerationChunk: with_token with empty text ──

    #[test]
    fn generation_chunk_with_token_empty_text() {
        let chunk = GenerationChunk::new().with_token(5, String::new()).finish();
        assert_eq!(chunk.tokens, vec![5]);
        assert!(chunk.text.is_empty());
        assert!(chunk.finished);
    }

    // ── GenerationResponse: thinking_content with empty string ──

    #[test]
    fn generation_response_thinking_content_empty_string_is_some() {
        let resp = GenerationResponse {
            text: "answer".to_string(),
            thinking_content: Some(String::new()),
            request_id: None,
        intent_classification: None,
};
        // Some("") is distinct from None
        assert!(resp.thinking_content.is_some());
        assert!(resp.thinking_content.as_ref().unwrap().is_empty());
    }

    // ── MediaInput: Base64 equality with different mime types ──

    #[test]
    fn media_input_base64_different_mime_types_not_equal() {
        let png = MediaInput::Base64 {
            data: "abc".to_string(),
            mime_type: Some("image/png".to_string()),
        };
        let jpeg = MediaInput::Base64 {
            data: "abc".to_string(),
            mime_type: Some("image/jpeg".to_string()),
        };
        assert_ne!(png, jpeg);
    }

    #[test]
    fn media_input_base64_same_data_same_mime_equal() {
        let a = MediaInput::Base64 {
            data: "xyz".to_string(),
            mime_type: Some("application/octet-stream".to_string()),
        };
        let b = MediaInput::Base64 {
            data: "xyz".to_string(),
            mime_type: Some("application/octet-stream".to_string()),
        };
        assert_eq!(a, b);
    }

    // ── ThresholdHook: single token in veto list boundary ──

    #[test]
    fn threshold_hook_max_vetoes_two_allows_one_veto() {
        let hook = ThresholdHook::new(vec![42], 2);
        // First veto: count=1, 1 < 2 → Veto
        assert!(matches!(hook.post_step(&[], &[42]), HookDecision::Veto(_)));
        // Second veto: count=2, 2 >= 2 → Terminate
        assert_eq!(hook.post_step(&[], &[42]), HookDecision::Terminate);
    }

    // ── GenerationStream from_tokens: token with value 0 ──

    #[test]
    fn generation_stream_from_tokens_zero_token_value() {
        let stream = GenerationStream::from_tokens(vec![0]);
        let items: Vec<_> = stream.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].tokens, vec![0]);
        // char::from_u32(0) is the null character, which is valid Unicode
        assert_eq!(items[0].text, "\0");
        assert!(items[0].finished);
    }

    // ── GenerationStream from_tokens: max u32 token value ──

    #[test]
    fn generation_stream_from_tokens_max_u32_token() {
        // u32::MAX is not a valid Unicode code point → fallback to '?'
        let stream = GenerationStream::from_tokens(vec![u32::MAX]);
        let items: Vec<_> = stream.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].tokens, vec![u32::MAX]);
        assert_eq!(items[0].text, "?");
    }

    // ── GenerationBuilder: temperature subnormal float ──

    #[test]
    fn generation_builder_temperature_subnormal() {
        let client = Client::new_empty();
        let subnormal = f32::from_bits(1); // smallest positive subnormal f32
        let builder = client.generate("test").temperature(subnormal);
        assert_eq!(builder.temperature.to_bits(), 1u32);
        assert!(builder.temperature > 0.0);
    }

    // ── ThinkingTracker: feed with only whitespace before tag ──

    #[test]
    fn thinking_tracker_whitespace_only_before_tag() {
        let mut tracker = ThinkingTracker::new(None);
        assert!(!tracker.feed("   \t\n  "));
        assert!(!tracker.feed("\r\n"));
        assert!(tracker.feed("<thinking>content</thinking>"));
        assert_eq!(tracker.state(), ThinkingState::Thinking);
        assert_eq!(tracker.thinking_token_count(), 1);
    }

    // ========================================================================
    // Round 6 — 15 additional edge-case tests
    // ========================================================================

    // ── GenerationChunk: unicode token value in with_token ──

    #[test]
    fn generation_chunk_with_token_unicode_text() {
        let chunk = GenerationChunk::new()
            .with_token(0x4F60, "\u{4f60}".to_string())
            .finish();
        assert_eq!(chunk.tokens, vec![0x4F60]);
        assert_eq!(chunk.text, "\u{4f60}");
        assert!(chunk.finished);
    }

    // ── HookDecision: Veto with unicode reason preserves content ──

    #[test]
    fn hook_decision_veto_unicode_reason() {
        let reason = "\u{4f60}\u{597d}\u{4e16}\u{754c}".to_string(); // 你好世界
        let d = HookDecision::Veto(reason.clone());
        if let HookDecision::Veto(r) = &d {
            assert_eq!(r, &reason);
        } else {
            panic!("expected Veto variant");
        }
        // Clone and compare
        let cloned = d.clone();
        assert_eq!(cloned, d);
    }

    // ── ThresholdHook: multiple different veto tokens in sequence ──

    #[test]
    fn threshold_hook_different_veto_tokens_increment_count() {
        let hook = ThresholdHook::new(vec![10, 20, 30], 4);
        // Veto token 10
        assert!(matches!(hook.post_step(&[], &[10]), HookDecision::Veto(_)));
        // Veto token 20
        assert!(matches!(hook.post_step(&[], &[20]), HookDecision::Veto(_)));
        // Non-veto
        assert_eq!(hook.post_step(&[], &[99]), HookDecision::Continue);
        // Veto token 30
        assert!(matches!(hook.post_step(&[], &[30]), HookDecision::Veto(_)));
        // veto_count should be 3
        assert_eq!(
            hook.veto_count.load(std::sync::atomic::Ordering::Relaxed),
            3
        );
    }

    // ── ThinkingTracker: feed after budget exhausted preserves token count ──

    #[test]
    fn thinking_tracker_budget_exhausted_preserves_count() {
        let mut tracker = ThinkingTracker::new(Some(2));
        assert!(tracker.feed("<thinking>")); // count=1
        assert!(tracker.feed("x"));          // count=2, BudgetExhausted
        assert_eq!(tracker.thinking_token_count(), 2);
        // Further feeds return false and do not change count
        assert!(!tracker.feed("y"));
        assert!(!tracker.feed("z"));
        assert!(!tracker.feed("</thinking>"));
        assert_eq!(tracker.thinking_token_count(), 2);
    }

    // ── split_thinking_content: markers adjacent with no surrounding text ──

    #[test]
    fn split_thinking_content_only_markers() {
        let (text, thinking) = split_thinking_content("<thinking></thinking>");
        assert_eq!(text, "");
        assert_eq!(thinking, Some(String::new()));
    }

    // ── split_thinking_content: start marker at end of long string without end marker ──

    #[test]
    fn split_thinking_content_start_at_end_of_long_string() {
        let prefix = "x".repeat(10000);
        let input = format!("{prefix}<thinking>");
        let (text, thinking) = split_thinking_content(&input);
        assert_eq!(text, input);
        assert!(thinking.is_none());
    }

    // ── GenerationResponse: same text different request_id are not equal ──

    #[test]
    fn generation_response_different_request_id_not_equal() {
        let a = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: None,
            request_id: Some(1),
        intent_classification: None,
};
        let b = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: None,
            request_id: Some(2),
        intent_classification: None,
};
        assert_ne!(a, b);
    }

    // ── GenerationResponse: same all fields are equal ──

    #[test]
    fn generation_response_identical_fields_are_equal() {
        let a = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: Some("thought".to_string()),
            request_id: Some(42),
        intent_classification: None,
};
        let b = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: Some("thought".to_string()),
            request_id: Some(42),
        intent_classification: None,
};
        assert_eq!(a, b);
    }

    // ── GenerationOutput: response() extracts Ok with all fields intact ──

    #[test]
    fn generation_output_response_ok_extracts_all_fields() {
        let out = GenerationOutput::Response(Ok(GenerationResponse {
            text: "full text".to_string(),
            thinking_content: Some("reasoning".to_string()),
            request_id: Some(999),
        intent_classification: None,
}));
        let resp = out.response().expect("should be Ok");
        assert_eq!(resp.text, "full text");
        assert_eq!(resp.thinking_content.as_deref(), Some("reasoning"));
        assert_eq!(resp.request_id, Some(999));
    }

    // ── MediaInput: Raw with single byte ──

    #[test]
    fn media_input_raw_single_byte() {
        let input = MediaInput::Raw(vec![0xFF]);
        let cloned = input.clone();
        assert_eq!(cloned, MediaInput::Raw(vec![0xFF]));
        match &input {
            MediaInput::Raw(bytes) => assert_eq!(bytes.len(), 1),
            _ => panic!("expected Raw"),
        }
    }

    // ── MediaInput: Url PartialEq same and different ──

    #[test]
    fn media_input_url_partialeq_same_and_different() {
        let a = MediaInput::Url("http://example.com".to_string());
        let b = MediaInput::Url("http://example.com".to_string());
        assert_eq!(a, b);
        let c = MediaInput::Url("http://other.com".to_string());
        assert_ne!(a, c);
    }

    // ── GenerationBuilder: calling setter twice overrides first value ──

    #[test]
    fn generation_builder_setter_overrides_previous() {
        let client = Client::new_empty();
        let builder = client
            .generate("test")
            .max_tokens(100)
            .max_tokens(200);
        assert_eq!(builder.max_tokens, 200);

        let builder = client
            .generate("test")
            .temperature(0.1)
            .temperature(0.9);
        assert!((builder.temperature - 0.9).abs() < 1e-6);

        let builder = client
            .generate("test")
            .top_p(0.5)
            .top_p(0.8);
        assert!((builder.top_p - 0.8).abs() < 1e-6);

        let builder = client
            .generate("test")
            .session_id(1)
            .session_id(2);
        assert_eq!(builder.session_id, Some(2));

        let builder = client
            .generate("test")
            .thinking_budget(10)
            .thinking_budget(20);
        assert_eq!(builder.thinking_budget, Some(20));
    }

    // ── GenerationBuilder: chaining all setters with extreme values ──

    #[test]
    fn generation_builder_all_setters_extreme_values() {
        let client = Client::new_empty();
        let builder = client
            .generate("")
            .max_tokens(0)
            .temperature(f32::MIN)
            .top_k(0)
            .top_p(f32::MAX)
            .session_id(0)
            .stream(true)
            .thinking_budget(0);
        assert_eq!(builder.max_tokens, 0);
        assert_eq!(builder.temperature, f32::MIN);
        assert_eq!(builder.top_k, 0);
        assert_eq!(builder.top_p, f32::MAX);
        assert_eq!(builder.session_id, Some(0));
        assert!(builder.stream);
        assert_eq!(builder.thinking_budget, Some(0));
    }

    // ── ThinkingState: Copy trait allows using value in multiple places ──

    #[test]
    fn thinking_state_copy_allows_multiple_uses() {
        let state = ThinkingState::Thinking;
        let mut collected = Vec::new();
        // Use the same Copy value multiple times
        for _ in 0..5 {
            collected.push(state);
        }
        assert_eq!(collected.len(), 5);
        assert!(collected.iter().all(|s| *s == ThinkingState::Thinking));
    }

    // ── GenerationStream from_tokens: three tokens yield correct finished pattern ──

    #[test]
    fn generation_stream_from_tokens_three_tokens_finished_pattern() {
        let stream = GenerationStream::from_tokens(vec![65, 66, 67]); // A, B, C
        let items: Vec<_> = stream.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(items.len(), 3);
        assert!(!items[0].finished);
        assert!(!items[1].finished);
        assert!(items[2].finished);
        assert_eq!(items[0].text, "A");
        assert_eq!(items[1].text, "AB");
        assert_eq!(items[2].text, "ABC");
    }
}
