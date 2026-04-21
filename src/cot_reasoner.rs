//! CoT Reasoner SDK — 任意 LLM 原生 Chain-of-Thought 推理 (REQ-COT-001..009)
//!
//! **SSOT**: `SPEC/COT-REASONER.md`, `SPEC/01-REQUIREMENTS.md §16`,
//! `SPEC/04-API-DESIGN.md §3.11`.
//!
//! # 概念
//!
//! CoT Reasoner 让任意 generator LLM（SmolLM2 / Llama / Qwen 等，**不依赖
//! 模型自带 thinking_head 权重**）通过 prompt engineering + 多轮
//! `Client::generate` iteration 在 Client 层获得原生多步推理能力。
//!
//! # 与 `thinking_budget` 的区别
//!
//! | 关注点 | `thinking_budget(n)` | `reasoning(ReasoningMode)` |
//! |--------|----------------------|----------------------------|
//! | 作用粒度 | 单次 generate 内 `<thinking>` token 数 | 跨多次 generate 调用的 orchestration |
//! | 依赖权重 | 是 (qwen3-thinking 等) | **否** (任意 LLM) |
//! | Step 概念 | 无 | 有 (Manual 指定,Auto 自适应) |
//! | 停止信号 | budget 耗尽 | budget / step count / pattern / entropy |
//!
//! # 铁律
//!
//! - **NO_SILENT_FALLBACK**: generate 失败 → `ReasoningError::GenerationFailed`;
//!   模式配置非法 → `InvalidConfig`; Client state 缺失 → `MissingModel`。
//!   禁止静默返回空 response。
//! - **NO_ISLAND_MODULE**: `Client::reason` 被 `GenerationBuilder::reasoning` →
//!   `ReasoningBuilder::execute` 真实转发调用 (REQ-COT-006)。
//! - **零 Backend 扩展**: 完全复用 `Client::execute_generation` 公共管线。

use thiserror::Error;

use crate::client::{Client, ClientError};

// ============================================================================
// ReasoningMode (SPEC COT-REASONER.md §3.1)
// ============================================================================

/// CoT 推理模式。
///
/// - `Manual`: 用户指定 `max_reasoning_tokens` + `step_count`,引擎严格按预算
///   分配每步 budget,跑满 `step_count` 步或 budget 耗尽即停。
/// - `Auto`: 引擎根据 stop patterns / entropy / budget 自动决定何时停止推理
///   进入 final answer 阶段。
#[derive(Debug, Clone)]
pub enum ReasoningMode {
    /// 固定步长: 用户指定推理预算与步数。
    Manual {
        /// 所有 reasoning step 合计最大 token 数 (不含 final answer)。
        max_reasoning_tokens: usize,
        /// 步数,必须 ≥ 1。
        step_count: usize,
    },
    /// 引擎自动决定步数/停止。
    Auto {
        /// 所有 reasoning step 合计上限 (不含 final answer)。
        max_total_tokens: usize,
        /// `Some(t)` 启用文本启发式熵收敛检测;`None` 禁用。
        /// 真实 logit-level entropy 需 `GenerationResponse` 扩展 (SPEC §5.2)。
        entropy_threshold: Option<f32>,
        /// 任一子串在 chunk 中命中即停。空列表 = 禁用 pattern match。
        stop_patterns: Vec<String>,
    },
}

/// Auto 模式下每 step 的 generate 预算(单步最多生成多少 token)。
///
/// 小于 `max_total_tokens` 时按此值分块;总预算不足时取剩余。
pub const AUTO_STEP_CHUNK: usize = 128;

/// Manual 模式下每 step 最小 budget 阈值。低于此值直接标记 BudgetExhausted。
pub const MIN_STEP_TOKENS: usize = 8;

/// 默认 Auto 模式 stop patterns(SPEC §3.3)。
pub const DEFAULT_STOP_PATTERNS: &[&str] = &[
    "Final Answer:",
    "Final answer:",
    "In conclusion,",
    "Therefore, the answer is",
    "The answer is",
];

// ============================================================================
// ReasoningTemplate (SPEC COT-REASONER.md §3.2)
// ============================================================================

/// CoT prompt 模板。用户可通过 `ReasoningTemplate { .. }` struct literal 部分覆写。
#[derive(Debug, Clone)]
pub struct ReasoningTemplate {
    /// 系统级 reasoning 引导。
    pub system_prompt: String,
    /// 每 step 前置标记。`{n}` 会被替换为 1-based step index。
    pub step_prefix: String,
    /// Final answer 前置标记。
    pub final_prefix: String,
    /// Step 间分隔符。
    pub step_separator: String,
    /// 每 step 采样温度 (> 0)。
    pub temperature: f32,
    /// 每 step top_k 采样参数。
    pub top_k: usize,
    /// 每 step top_p nucleus 采样参数。
    pub top_p: f32,
    /// Final answer 阶段独立 token 预算。
    pub final_answer_budget: usize,
}

impl Default for ReasoningTemplate {
    fn default() -> Self {
        Self {
            system_prompt: "You are a careful reasoner. Break problems into \
                            explicit steps and reach the final answer after \
                            all steps."
                .to_string(),
            step_prefix: "Step {n}:".to_string(),
            final_prefix: "Final Answer:".to_string(),
            step_separator: "\n\n".to_string(),
            temperature: 0.7,
            top_k: 0,
            top_p: 1.0,
            final_answer_budget: 256,
        }
    }
}

impl ReasoningTemplate {
    /// 渲染 step N 的前缀(`{n}` 替换为 1-based index)。
    pub fn render_step_prefix(&self, step_index_1based: usize) -> String {
        self.step_prefix
            .replace("{n}", &step_index_1based.to_string())
    }

    /// 校验参数合法性。
    pub fn validate(&self) -> Result<(), ReasoningError> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(ReasoningError::InvalidConfig(format!(
                "temperature must be finite and > 0, got {}",
                self.temperature
            )));
        }
        if self.final_answer_budget == 0 {
            return Err(ReasoningError::InvalidConfig(
                "final_answer_budget must be > 0".to_string(),
            ));
        }
        if self.step_prefix.is_empty() {
            return Err(ReasoningError::InvalidConfig(
                "step_prefix must not be empty".to_string(),
            ));
        }
        if self.final_prefix.is_empty() {
            return Err(ReasoningError::InvalidConfig(
                "final_prefix must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// ReasoningResponse / ReasoningStopReason (SPEC §8.3/§8.4)
// ============================================================================

/// CoT 推理结果。
#[derive(Debug, Clone)]
pub struct ReasoningResponse {
    /// Final answer 文本(不含 reasoning trace)。
    pub text: String,
    /// 每步推理内容(按执行顺序)。
    pub reasoning_trace: Vec<String>,
    /// 所有 reasoning step 合计产出 token 数(估算)。
    pub total_reasoning_tokens: usize,
    /// 实际执行步数(`== reasoning_trace.len()`)。
    pub actual_steps: usize,
    /// 停止原因。
    pub stopped_reason: ReasoningStopReason,
}

/// CoT 推理停止原因。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningStopReason {
    /// 达到 `max_reasoning_tokens` / `max_total_tokens` 上限。
    BudgetExhausted,
    /// Manual 模式正常完成所有 step。
    StepCountReached,
    /// Auto 模式命中 stop_patterns 中的某个子串。
    PatternMatched(String),
    /// Auto 模式 entropy 启发式估算低于阈值判定收敛。
    EntropyConverged,
    /// Step hook 通过 `StepAction::Halt(reason)` 主动终止。
    HaltByHook(String),
}

// ============================================================================
// Step Hook API (SPEC §7 — REQ-COT-007..009)
// ============================================================================

/// Step 执行上下文 — 在 `on_step_start` 时提供给 hook (REQ-COT-008)。
///
/// 包含当前 step 的元信息和历史累积文本,允许 hook 基于已产出的
/// reasoning 内容做动态决策 (如注入补充 prompt、检测循环并终止)。
#[derive(Debug, Clone)]
pub struct StepContext {
    /// 当前 step 的 0-based index。
    pub step_index: usize,
    /// 剩余总 budget (tokens)。Manual 模式 = 剩余 reasoning_tokens;
    /// Auto 模式 = 剩余 max_total_tokens。
    pub remaining_budget: usize,
    /// 之前所有 step 产出文本的累积拼接 (step_separator 分隔)。
    /// 第 0 步时为空串。
    pub accumulated_text: String,
    /// 当前使用的模型标识。
    pub model_name: String,
}

/// `on_step_start` 返回的流程控制动作 (REQ-COT-009)。
#[derive(Debug, Clone)]
pub enum StepAction {
    /// 正常执行当前 step。
    Continue,
    /// 跳过当前 step(不调用 generate,不消耗 budget)。
    /// trace 中不记录此步。
    Skip,
    /// 在当前 step 的 prompt 末尾追加额外文本后继续执行。
    InjectPrompt(String),
    /// 立即终止 reasoning 循环,附带终止原因。
    Halt(String),
}

/// 单步执行结果 — 传递给 `on_step_end`。
#[derive(Debug, Clone)]
pub struct StepResult {
    /// 当前 step 的 0-based index。
    pub step_index: usize,
    /// 本步 generate 产出的文本 (trimmed)。
    pub chunk_text: String,
    /// 本步消耗的 token 数 (估算)。
    pub tokens_used: usize,
    /// 截至本步结束时所有 step 的累积 reasoning text。
    pub accumulated_text: String,
}

/// `on_step_end` 返回的知识注入。
#[derive(Debug, Clone, Default)]
pub struct StepKnowledge {
    /// 若为 `Some(text)`,text 会被追加到下一步 prompt 的末尾
    /// (在 step_prefix 之后、模型生成之前)。用于动态注入知识。
    pub inject_text: Option<String>,
    /// 若为 `Some(t)`,覆盖下一步的采样 temperature。
    /// 用于基于推理进展动态调整采样策略 (如逐步降温)。
    pub modified_temperature: Option<f32>,
}

/// Step-level 回调 trait。用户实现此 trait 并通过
/// `ReasoningBuilder::with_step_hook()` 注册,引擎在每步边界调用 (REQ-COT-007)。
///
/// # 生命周期
///
/// 1. `on_step_start(&mut self, ctx: &StepContext) -> StepAction`
///    - 在 generate 调用**之前**触发
///    - 返回 `Continue` / `Skip` / `InjectPrompt` / `Halt` 控制流程
///
/// 2. `on_step_end(&mut self, result: &StepResult) -> StepKnowledge`
///    - 在 generate 调用**之后**触发
///    - 接收本步产出的 chunk text 和元信息
///    - 返回 `StepKnowledge` 注入到下一步
pub trait ReasoningStepHook: Send + Sync {
    /// Step 开始前回调。返回 `StepAction` 控制当前步行为。
    fn on_step_start(&mut self, ctx: &StepContext) -> StepAction;

    /// Step 结束后回调。返回 `StepKnowledge` 注入到下一步。
    fn on_step_end(&mut self, result: &StepResult) -> StepKnowledge;
}
// ============================================================================

/// CoT 推理专用错误。
#[derive(Debug, Clone, Error)]
pub enum ReasoningError {
    /// Client 未加载模型。
    #[error("no model loaded for CoT reasoning")]
    MissingModel,

    /// 下游 `Client::generate` 失败。
    #[error("generation failed during reasoning step: {0}")]
    GenerationFailed(String),

    /// ReasoningMode / Template 参数非法。
    #[error("invalid reasoning config: {0}")]
    InvalidConfig(String),
}

impl From<ReasoningError> for ClientError {
    fn from(err: ReasoningError) -> Self {
        match err {
            ReasoningError::MissingModel => ClientError::NoModelLoaded,
            ReasoningError::GenerationFailed(s) => {
                ClientError::RuntimeError(format!("cot_reasoner: {s}"))
            }
            ReasoningError::InvalidConfig(s) => {
                ClientError::RuntimeError(format!("cot_reasoner invalid config: {s}"))
            }
        }
    }
}

// ============================================================================
// Token count & entropy estimation (SPEC §5.2 启发式)
// ============================================================================

/// Token 数估算: 英文文本 ≈ 4 char/token 的启发式近似。
///
/// 用于在不调用 tokenizer 的快速路径估算预算消耗。精确 tokenize 在
/// 某些场景不可用(如 Client::state 已释放),故启发式是设计底线。
///
/// 导出为 `pub` 是为了允许 E2E 测试和用户代码直接验证 budget 估算逻辑
/// (REQ-COT-001 验收需要)。
pub fn estimate_token_count(text: &str) -> usize {
    // Tokenizer 通常把空白 / 标点也计 token,用 char count / 4 而非 byte count
    // 更接近英文 BPE 实际 token 数。
    let chars = text.chars().count();
    (chars + 3) / 4
}

/// 文本熵启发式估算 (SPEC §5.2)。
///
/// 三因素合成:
/// - `diversity = unique_chars / total_chars` ∈ [0, 1]
/// - `repetition_ratio = duplicate_trigram / total_trigram` ∈ [0, 1]
/// - `word_scale = log2(word_count + 1)` 单调递增
///
/// 返回 `diversity * (1 - repetition_ratio) * word_scale`。
/// 高熵(多样、无重复、内容丰富)→ 数值大;低熵(单调、重复、空泛)→ 接近 0。
///
/// 真实 logit-level entropy 计算公式 `H(p) = -Σ p_i log p_i`,需要
/// `GenerationResponse` 暴露 per-token logit 分布,见 SPEC §5.2 未来扩展。
///
/// 导出为 `pub` 是为了允许 E2E 测试直接验证启发式行为(REQ-COT-003)。
pub fn estimate_text_entropy(text: &str) -> f32 {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return 0.0;
    }

    // 1. Character diversity
    let total_chars = trimmed.chars().count() as f32;
    if total_chars < 1.0 {
        return 0.0;
    }
    let unique_chars: std::collections::HashSet<char> = trimmed.chars().collect();
    let diversity = (unique_chars.len() as f32) / total_chars;

    // 2. Trigram repetition ratio
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let word_count = words.len();
    let repetition_ratio = if word_count >= 3 {
        let mut seen = std::collections::HashMap::new();
        let total_trigrams = word_count - 2;
        for i in 0..total_trigrams {
            let key = format!("{} {} {}", words[i], words[i + 1], words[i + 2]);
            *seen.entry(key).or_insert(0u32) += 1;
        }
        let dup: u32 = seen.values().map(|&c| c.saturating_sub(1)).sum();
        (dup as f32) / (total_trigrams as f32)
    } else {
        0.0
    };

    // 3. Word scale (log2 单调,避免短文本 entropy 被膨胀)
    let word_scale = ((word_count + 1) as f32).log2();

    diversity * (1.0 - repetition_ratio) * word_scale
}

/// 测试命中任何一个 stop pattern;命中即返回该 pattern。
///
/// 导出为 `pub` 是为了允许 E2E 测试直接验证 pattern 匹配行为(REQ-COT-002)。
pub fn find_stop_pattern<'a>(text: &str, patterns: &'a [String]) -> Option<&'a String> {
    patterns.iter().find(|p| text.contains(p.as_str()))
}

// ============================================================================
// Client::reason (SPEC §8.1)
// ============================================================================

impl Client {
    /// Chain-of-Thought 推理入口 (REQ-COT-001..006)。
    ///
    /// 完全复用 `Client::generate` 公共管线,不新增 Backend trait 方法。
    /// 无 Step Hook 版本 — 等价于 `reason_with_hook(prompt, mode, template, None)`。
    ///
    /// # 参数
    /// - `prompt`: 用户原始 query(不含 `"Step 1:"` 等前缀,由 template 注入)
    /// - `mode`: `Manual { max_reasoning_tokens, step_count }` 或
    ///   `Auto { max_total_tokens, entropy_threshold, stop_patterns }`
    /// - `template`: `None` 使用 `ReasoningTemplate::default()`;`Some(tpl)` 覆写
    ///
    /// # 错误
    /// - `ClientError::NoModelLoaded` - 调用前未加载模型
    /// - `ClientError::RuntimeError("cot_reasoner: ...")` - 内部某轮 generate 失败
    /// - `ClientError::RuntimeError("cot_reasoner invalid config: ...")` - 参数非法
    pub fn reason(
        &self,
        prompt: &str,
        mode: ReasoningMode,
        template: Option<ReasoningTemplate>,
    ) -> Result<ReasoningResponse, ClientError> {
        self.reason_with_hook(prompt, mode, template, None)
    }

    /// Chain-of-Thought 推理入口 (带 Step Hook)。
    ///
    /// 内部实现: `Client::reason` delegate 到此方法。`ReasoningBuilder::execute`
    /// 直接调用此方法传入 hook。
    pub(crate) fn reason_with_hook(
        &self,
        prompt: &str,
        mode: ReasoningMode,
        template: Option<ReasoningTemplate>,
        mut hook: Option<Box<dyn ReasoningStepHook>>,
    ) -> Result<ReasoningResponse, ClientError> {
        // 预检: 模型加载状态 (早失败避免进入循环)
        let state = self.require_state()?;
        let model_name = state.model_id.clone();
        let mut tpl = template.unwrap_or_default();
        tpl.validate().map_err(ClientError::from)?;

        // 校验 mode
        match &mode {
            ReasoningMode::Manual {
                max_reasoning_tokens,
                step_count,
            } => {
                if *step_count == 0 {
                    return Err(ReasoningError::InvalidConfig(
                        "step_count must be >= 1".to_string(),
                    )
                    .into());
                }
                if *max_reasoning_tokens == 0 {
                    return Err(ReasoningError::InvalidConfig(
                        "max_reasoning_tokens must be > 0".to_string(),
                    )
                    .into());
                }
            }
            ReasoningMode::Auto {
                max_total_tokens,
                entropy_threshold,
                stop_patterns: _,
            } => {
                if *max_total_tokens == 0 {
                    return Err(ReasoningError::InvalidConfig(
                        "max_total_tokens must be > 0".to_string(),
                    )
                    .into());
                }
                if let Some(t) = entropy_threshold {
                    if !t.is_finite() || *t < 0.0 {
                        return Err(ReasoningError::InvalidConfig(format!(
                            "entropy_threshold must be finite and >= 0, got {}",
                            t
                        ))
                        .into());
                    }
                }
            }
        }

        let mut trace: Vec<String> = Vec::new();
        let mut total_tokens: usize = 0;

        // 1. 构造初始上下文
        let mut context =
            format!("{}\n\n{}", tpl.system_prompt.trim(), prompt.trim());

        // 知识注入缓冲: on_step_end 返回的 inject_text 累积到这里,
        // 在下一步 prompt 构造前追加到 context。
        let mut pending_inject: Option<String> = None;

        // 2. Reasoning 循环
        let stopped_reason = match &mode {
            ReasoningMode::Manual {
                max_reasoning_tokens,
                step_count,
            } => self.run_manual_loop(
                &mut context,
                &mut trace,
                &mut total_tokens,
                &mut tpl,
                *max_reasoning_tokens,
                *step_count,
                &model_name,
                &mut hook,
                &mut pending_inject,
            )?,
            ReasoningMode::Auto {
                max_total_tokens,
                entropy_threshold,
                stop_patterns,
            } => self.run_auto_loop(
                &mut context,
                &mut trace,
                &mut total_tokens,
                &mut tpl,
                *max_total_tokens,
                *entropy_threshold,
                stop_patterns,
                &model_name,
                &mut hook,
                &mut pending_inject,
            )?,
        };

        // 3. Final answer 阶段
        let final_prompt = format!(
            "{ctx}{sep}{prefix}",
            ctx = context,
            sep = tpl.step_separator,
            prefix = tpl.final_prefix
        );
        let final_text = self
            .execute_generation(
                final_prompt,
                tpl.final_answer_budget,
                tpl.temperature,
                tpl.top_k,
                tpl.top_p,
                None,
                None,
            )
            .map(|r| r.text)
            .map_err(|e| {
                let msg = format!("final answer generation failed: {e}");
                ReasoningError::GenerationFailed(msg)
            })?;

        Ok(ReasoningResponse {
            text: final_text.trim().to_string(),
            actual_steps: trace.len(),
            reasoning_trace: trace,
            total_reasoning_tokens: total_tokens,
            stopped_reason,
        })
    }

    fn run_manual_loop(
        &self,
        context: &mut String,
        trace: &mut Vec<String>,
        total_tokens: &mut usize,
        tpl: &mut ReasoningTemplate,
        max_reasoning_tokens: usize,
        step_count: usize,
        model_name: &str,
        hook: &mut Option<Box<dyn ReasoningStepHook>>,
        pending_inject: &mut Option<String>,
    ) -> Result<ReasoningStopReason, ClientError> {
        for step in 1..=step_count {
            let step_index_0 = step - 1;

            // 注入上一步 on_step_end 返回的知识
            if let Some(inject) = pending_inject.take() {
                context.push_str(&tpl.step_separator);
                context.push_str(&inject);
            }

            // ── on_step_start ──
            let mut inject_prompt: Option<String> = None;
            if let Some(h) = hook.as_deref_mut() {
                let remaining_budget =
                    max_reasoning_tokens.saturating_sub(*total_tokens);
                let ctx = StepContext {
                    step_index: step_index_0,
                    remaining_budget,
                    accumulated_text: trace.join(&tpl.step_separator),
                    model_name: model_name.to_string(),
                };
                match h.on_step_start(&ctx) {
                    StepAction::Continue => {}
                    StepAction::Skip => continue,
                    StepAction::InjectPrompt(extra) => {
                        inject_prompt = Some(extra);
                    }
                    StepAction::Halt(reason) => {
                        return Ok(ReasoningStopReason::HaltByHook(reason));
                    }
                }
            }

            // 动态分配剩余 budget 到剩余 step
            let remaining_budget = max_reasoning_tokens.saturating_sub(*total_tokens);
            let remaining_steps = step_count - step + 1;
            let step_budget = remaining_budget / remaining_steps.max(1);

            if step_budget < MIN_STEP_TOKENS {
                return Ok(ReasoningStopReason::BudgetExhausted);
            }

            let step_prefix = tpl.render_step_prefix(step);
            let mut step_prompt = format!(
                "{ctx}{sep}{prefix}",
                ctx = context,
                sep = tpl.step_separator,
                prefix = step_prefix
            );
            if let Some(extra) = &inject_prompt {
                step_prompt.push_str(extra);
            }

            let chunk_text = self
                .execute_generation(
                    step_prompt,
                    step_budget,
                    tpl.temperature,
                    tpl.top_k,
                    tpl.top_p,
                    None,
                    None,
                )
                .map(|r| r.text)
                .map_err(|e| {
                    ReasoningError::GenerationFailed(format!(
                        "manual step {step} generation failed: {e}"
                    ))
                })?;

            let chunk_trimmed = chunk_text.trim().to_string();
            if chunk_trimmed.is_empty() {
                // 空 chunk (模型拒绝延续) — 记录 BudgetExhausted 避免死循环
                return Ok(ReasoningStopReason::BudgetExhausted);
            }

            let tokens_used = estimate_token_count(&chunk_trimmed);
            *total_tokens = total_tokens.saturating_add(tokens_used);
            trace.push(chunk_trimmed.clone());

            // 更新 context: 追加 step_prefix + chunk 供下一步继续
            context.push_str(&tpl.step_separator);
            context.push_str(&tpl.render_step_prefix(step));
            context.push(' ');
            context.push_str(&chunk_trimmed);

            // ── on_step_end ──
            if let Some(h) = hook.as_deref_mut() {
                let accumulated = trace.join(&tpl.step_separator);
                let result = StepResult {
                    step_index: step_index_0,
                    chunk_text: chunk_trimmed.clone(),
                    tokens_used,
                    accumulated_text: accumulated,
                };
                let knowledge = h.on_step_end(&result);
                if let Some(text) = knowledge.inject_text {
                    *pending_inject = Some(text);
                }
                if let Some(temp) = knowledge.modified_temperature {
                    tpl.temperature = temp;
                }
            }

            if *total_tokens >= max_reasoning_tokens {
                return Ok(ReasoningStopReason::BudgetExhausted);
            }
        }
        Ok(ReasoningStopReason::StepCountReached)
    }

    fn run_auto_loop(
        &self,
        context: &mut String,
        trace: &mut Vec<String>,
        total_tokens: &mut usize,
        tpl: &mut ReasoningTemplate,
        max_total_tokens: usize,
        entropy_threshold: Option<f32>,
        stop_patterns: &[String],
        model_name: &str,
        hook: &mut Option<Box<dyn ReasoningStepHook>>,
        pending_inject: &mut Option<String>,
    ) -> Result<ReasoningStopReason, ClientError> {
        let mut step: usize = 1;
        // Auto 模式硬上限: 超过此步数必定终止,防止病态输入导致无限循环。
        const AUTO_MAX_STEPS_GUARD: usize = 32;

        loop {
            if step > AUTO_MAX_STEPS_GUARD {
                return Ok(ReasoningStopReason::BudgetExhausted);
            }
            let step_index_0 = step - 1;

            // 注入上一步 on_step_end 返回的知识
            if let Some(inject) = pending_inject.take() {
                context.push_str(&tpl.step_separator);
                context.push_str(&inject);
            }

            // ── on_step_start ──
            let mut inject_prompt: Option<String> = None;
            if let Some(h) = hook.as_deref_mut() {
                let remaining = max_total_tokens.saturating_sub(*total_tokens);
                let ctx = StepContext {
                    step_index: step_index_0,
                    remaining_budget: remaining,
                    accumulated_text: trace.join(&tpl.step_separator),
                    model_name: model_name.to_string(),
                };
                match h.on_step_start(&ctx) {
                    StepAction::Continue => {}
                    StepAction::Skip => {
                        step += 1;
                        continue;
                    }
                    StepAction::InjectPrompt(extra) => {
                        inject_prompt = Some(extra);
                    }
                    StepAction::Halt(reason) => {
                        return Ok(ReasoningStopReason::HaltByHook(reason));
                    }
                }
            }

            let remaining = max_total_tokens.saturating_sub(*total_tokens);
            if remaining < MIN_STEP_TOKENS {
                return Ok(ReasoningStopReason::BudgetExhausted);
            }
            let step_budget = AUTO_STEP_CHUNK.min(remaining);

            let step_prefix = tpl.render_step_prefix(step);
            let mut step_prompt = format!(
                "{ctx}{sep}{prefix}",
                ctx = context,
                sep = tpl.step_separator,
                prefix = step_prefix
            );
            if let Some(extra) = &inject_prompt {
                step_prompt.push_str(extra);
            }

            let chunk_text = self
                .execute_generation(
                    step_prompt,
                    step_budget,
                    tpl.temperature,
                    tpl.top_k,
                    tpl.top_p,
                    None,
                    None,
                )
                .map(|r| r.text)
                .map_err(|e| {
                    ReasoningError::GenerationFailed(format!(
                        "auto step {step} generation failed: {e}"
                    ))
                })?;

            let chunk_trimmed = chunk_text.trim().to_string();
            if chunk_trimmed.is_empty() {
                return Ok(ReasoningStopReason::BudgetExhausted);
            }

            let tokens_used = estimate_token_count(&chunk_trimmed);
            *total_tokens = total_tokens.saturating_add(tokens_used);
            trace.push(chunk_trimmed.clone());

            context.push_str(&tpl.step_separator);
            context.push_str(&tpl.render_step_prefix(step));
            context.push(' ');
            context.push_str(&chunk_trimmed);

            // ── on_step_end ──
            if let Some(h) = hook.as_deref_mut() {
                let accumulated = trace.join(&tpl.step_separator);
                let result = StepResult {
                    step_index: step_index_0,
                    chunk_text: chunk_trimmed.clone(),
                    tokens_used,
                    accumulated_text: accumulated,
                };
                let knowledge = h.on_step_end(&result);
                if let Some(text) = knowledge.inject_text {
                    *pending_inject = Some(text);
                }
                if let Some(temp) = knowledge.modified_temperature {
                    tpl.temperature = temp;
                }
            }

            // Stop signals (pattern 优先, entropy 次之, budget 最后)
            if let Some(matched) = find_stop_pattern(&chunk_trimmed, stop_patterns) {
                return Ok(ReasoningStopReason::PatternMatched(matched.clone()));
            }
            if let Some(t) = entropy_threshold {
                let est = estimate_text_entropy(&chunk_trimmed);
                if est < t {
                    return Ok(ReasoningStopReason::EntropyConverged);
                }
            }
            if *total_tokens >= max_total_tokens {
                return Ok(ReasoningStopReason::BudgetExhausted);
            }

            step += 1;
        }
    }
}

// ============================================================================
// ReasoningBuilder (SPEC §8.2) — 由 GenerationBuilder::reasoning 返回
// ============================================================================

/// CoT reasoning 链式 builder,由 `GenerationBuilder::reasoning(mode)` 构造。
pub struct ReasoningBuilder<'a> {
    client: &'a Client,
    prompt: String,
    mode: ReasoningMode,
    template: Option<ReasoningTemplate>,
    /// Step Hook (可选)。`None` = 无 hook,原有行为不变。
    step_hook: Option<Box<dyn ReasoningStepHook>>,
}

impl<'a> ReasoningBuilder<'a> {
    /// 内部构造器(由 `GenerationBuilder::reasoning` 调用)。
    pub(crate) fn new(client: &'a Client, prompt: String, mode: ReasoningMode) -> Self {
        Self {
            client,
            prompt,
            mode,
            template: None,
            step_hook: None,
        }
    }

    /// 覆写 `ReasoningTemplate`。默认使用 `ReasoningTemplate::default()`。
    pub fn template(mut self, template: ReasoningTemplate) -> Self {
        self.template = Some(template);
        self
    }

    /// 注册 Step Hook (REQ-COT-007..009)。
    ///
    /// Hook 在 `execute()` 内的 step 循环中被调用,生命周期覆盖
    /// 整个 reasoning 过程。多次调用替换前一个 hook (单一 hook 语义)。
    pub fn with_step_hook(mut self, hook: Box<dyn ReasoningStepHook>) -> Self {
        self.step_hook = Some(hook);
        self
    }

    /// 执行 reasoning 并返回完整 `ReasoningResponse`。
    pub fn execute(self) -> Result<ReasoningResponse, ClientError> {
        self.client
            .reason_with_hook(&self.prompt, self.mode, self.template, self.step_hook)
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ReasoningTemplate ──────────────────────────────────────────────

    #[test]
    fn template_default_non_empty() {
        let tpl = ReasoningTemplate::default();
        assert!(!tpl.system_prompt.is_empty());
        assert!(tpl.step_prefix.contains("{n}"));
        assert!(!tpl.final_prefix.is_empty());
        assert_eq!(tpl.step_separator, "\n\n");
        assert_eq!(tpl.temperature, 0.7);
        assert!(tpl.final_answer_budget > 0);
        assert!(tpl.validate().is_ok());
    }

    #[test]
    fn template_render_step_prefix_replaces_n() {
        let tpl = ReasoningTemplate::default();
        assert_eq!(tpl.render_step_prefix(1), "Step 1:");
        assert_eq!(tpl.render_step_prefix(5), "Step 5:");
        assert_eq!(tpl.render_step_prefix(42), "Step 42:");
    }

    #[test]
    fn template_custom_step_prefix_override() {
        let tpl = ReasoningTemplate {
            step_prefix: "Reasoning {n}:".to_string(),
            ..Default::default()
        };
        assert_eq!(tpl.render_step_prefix(3), "Reasoning 3:");
    }

    #[test]
    fn template_validate_rejects_bad_temperature() {
        let mut tpl = ReasoningTemplate::default();
        tpl.temperature = 0.0;
        assert!(matches!(
            tpl.validate(),
            Err(ReasoningError::InvalidConfig(_))
        ));
        tpl.temperature = -1.0;
        assert!(matches!(
            tpl.validate(),
            Err(ReasoningError::InvalidConfig(_))
        ));
        tpl.temperature = f32::NAN;
        assert!(matches!(
            tpl.validate(),
            Err(ReasoningError::InvalidConfig(_))
        ));
        tpl.temperature = f32::INFINITY;
        assert!(matches!(
            tpl.validate(),
            Err(ReasoningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn template_validate_rejects_zero_final_budget() {
        let tpl = ReasoningTemplate {
            final_answer_budget: 0,
            ..Default::default()
        };
        assert!(matches!(
            tpl.validate(),
            Err(ReasoningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn template_validate_rejects_empty_step_prefix() {
        let tpl = ReasoningTemplate {
            step_prefix: String::new(),
            ..Default::default()
        };
        assert!(matches!(
            tpl.validate(),
            Err(ReasoningError::InvalidConfig(_))
        ));
    }

    // ─── estimate_token_count ───────────────────────────────────────────

    #[test]
    fn estimate_token_count_basic() {
        assert_eq!(estimate_token_count(""), 0);
        // "hello" = 5 chars → (5 + 3) / 4 = 2 tokens estimate
        assert_eq!(estimate_token_count("hello"), 2);
        // 16 chars / 4 = 4 tokens
        assert_eq!(estimate_token_count("0123456789abcdef"), 4);
    }

    #[test]
    fn estimate_token_count_monotonic_with_length() {
        let short = estimate_token_count("hi");
        let medium = estimate_token_count("hello world from rust");
        let long = estimate_token_count(&"x".repeat(400));
        assert!(short < medium);
        assert!(medium < long);
        assert_eq!(long, 100); // 400 / 4
    }

    // ─── estimate_text_entropy ──────────────────────────────────────────

    #[test]
    fn entropy_empty_is_zero() {
        assert_eq!(estimate_text_entropy(""), 0.0);
        assert_eq!(estimate_text_entropy("   \n\t"), 0.0);
    }

    #[test]
    fn entropy_repetitive_is_low() {
        // "abc abc abc abc abc abc abc abc" — 多次三 gram 重复
        let repetitive = "abc abc abc abc abc abc abc abc";
        let diverse = "The quick brown fox jumps over the lazy dog today";
        let rep_h = estimate_text_entropy(repetitive);
        let div_h = estimate_text_entropy(diverse);
        assert!(
            div_h > rep_h,
            "diverse text entropy ({}) must exceed repetitive ({})",
            div_h,
            rep_h
        );
    }

    #[test]
    fn entropy_low_diversity_is_low() {
        // 全部同字符 → diversity ≈ 1/len → entropy 很低
        let mono = "aaaaaaaaaaaaaaaaaaaa";
        let mixed = "Let me think about this step by step";
        assert!(estimate_text_entropy(mono) < estimate_text_entropy(mixed));
    }

    // ─── find_stop_pattern ──────────────────────────────────────────────

    #[test]
    fn stop_pattern_finds_substring() {
        let patterns = vec![
            "Final Answer:".to_string(),
            "In conclusion,".to_string(),
        ];
        let matched = find_stop_pattern("... so in conclusion, we get 42.", &patterns);
        // case-sensitive: lowercase "in conclusion" 不匹配 "In conclusion,"
        assert!(matched.is_none());

        let matched2 = find_stop_pattern(
            "After computing, Final Answer: 42",
            &patterns,
        );
        assert_eq!(matched2.map(|s| s.as_str()), Some("Final Answer:"));
    }

    #[test]
    fn stop_pattern_empty_list_returns_none() {
        let patterns: Vec<String> = vec![];
        assert!(find_stop_pattern("Final Answer: 42", &patterns).is_none());
    }

    #[test]
    fn stop_pattern_no_match_returns_none() {
        let patterns = vec!["Final Answer:".to_string()];
        assert!(find_stop_pattern("Just some text.", &patterns).is_none());
    }

    // ─── ReasoningError → ClientError mapping ───────────────────────────

    #[test]
    fn error_maps_missing_model_to_no_model_loaded() {
        let e: ClientError = ReasoningError::MissingModel.into();
        assert!(matches!(e, ClientError::NoModelLoaded));
    }

    #[test]
    fn error_maps_generation_failed_to_runtime_error() {
        let e: ClientError =
            ReasoningError::GenerationFailed("boom".to_string()).into();
        match e {
            ClientError::RuntimeError(msg) => assert!(msg.contains("cot_reasoner")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn error_maps_invalid_config_to_runtime_error() {
        let e: ClientError = ReasoningError::InvalidConfig("bad".to_string()).into();
        match e {
            ClientError::RuntimeError(msg) => {
                assert!(msg.contains("cot_reasoner invalid config"))
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ─── ReasoningMode validation via Client::reason (without model) ──

    #[test]
    fn reason_without_model_returns_no_model_loaded() {
        let client = Client::new_empty();
        let r = client.reason(
            "What is 2+2?",
            ReasoningMode::Manual {
                max_reasoning_tokens: 100,
                step_count: 2,
            },
            None,
        );
        assert!(matches!(r, Err(ClientError::NoModelLoaded)));
    }

    // ─── ReasoningStopReason 语义 ──────────────────────────────────────

    #[test]
    fn stop_reason_eq_variants() {
        assert_eq!(
            ReasoningStopReason::BudgetExhausted,
            ReasoningStopReason::BudgetExhausted
        );
        assert_eq!(
            ReasoningStopReason::StepCountReached,
            ReasoningStopReason::StepCountReached
        );
        assert_eq!(
            ReasoningStopReason::PatternMatched("x".into()),
            ReasoningStopReason::PatternMatched("x".into())
        );
        assert_ne!(
            ReasoningStopReason::PatternMatched("x".into()),
            ReasoningStopReason::PatternMatched("y".into())
        );
        assert_eq!(
            ReasoningStopReason::HaltByHook("reason".into()),
            ReasoningStopReason::HaltByHook("reason".into())
        );
        assert_ne!(
            ReasoningStopReason::HaltByHook("a".into()),
            ReasoningStopReason::HaltByHook("b".into())
        );
    }

    // ─── Auto 模式 entropy 停止单元测试 (REQ-COT-003 启发式) ──────────

    #[test]
    fn entropy_threshold_triggers_on_repetitive_text() {
        // 直接验证启发式函数层面的行为:
        // 即 "对高重复文本, entropy 落到阈值之下"。
        // 这保证 Auto 循环在真实模型产出退化文本时能判定 EntropyConverged。
        let rep = "abc abc abc abc abc abc abc abc abc abc abc abc";
        let entropy = estimate_text_entropy(rep);
        // 阈值 0.5 对于这种高重复 3-gram 应当触发停止
        assert!(
            entropy < 0.5,
            "repetitive text entropy ({entropy}) must be below threshold 0.5"
        );
    }

    #[test]
    fn entropy_threshold_does_not_trigger_on_diverse_text() {
        let div = "The quick brown fox jumps over the lazy dog. \
                   Each step reveals a new piece of logic.";
        let entropy = estimate_text_entropy(div);
        assert!(
            entropy >= 0.5,
            "diverse text entropy ({entropy}) must exceed threshold 0.5"
        );
    }

    // ─── DEFAULT_STOP_PATTERNS ──────────────────────────────────────────

    #[test]
    fn default_stop_patterns_contains_key_phrases() {
        assert!(DEFAULT_STOP_PATTERNS.contains(&"Final Answer:"));
        assert!(DEFAULT_STOP_PATTERNS.contains(&"In conclusion,"));
        assert!(DEFAULT_STOP_PATTERNS.len() >= 3);
    }

    // ─── ReasoningBuilder (纯构造测试,无模型) ───────────────────────────

    #[test]
    fn reasoning_builder_new_stores_fields() {
        let client = Client::new_empty();
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: 100,
            step_count: 3,
        };
        let builder = ReasoningBuilder::new(&client, "prompt".into(), mode);
        // template 默认 None
        assert!(builder.template.is_none());
        // step_hook 默认 None
        assert!(builder.step_hook.is_none());
    }

    #[test]
    fn reasoning_builder_template_override() {
        let client = Client::new_empty();
        let mode = ReasoningMode::Auto {
            max_total_tokens: 512,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        let tpl = ReasoningTemplate {
            step_prefix: "Reasoning {n}:".into(),
            ..Default::default()
        };
        let builder = ReasoningBuilder::new(&client, "q".into(), mode).template(tpl);
        assert!(builder.template.is_some());
    }

    #[test]
    fn reasoning_builder_execute_without_model_errors() {
        let client = Client::new_empty();
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: 100,
            step_count: 2,
        };
        let result = ReasoningBuilder::new(&client, "test".into(), mode).execute();
        assert!(matches!(result, Err(ClientError::NoModelLoaded)));
    }
}
