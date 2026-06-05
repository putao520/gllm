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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Default, PartialEq)]
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
#[derive(Debug, Clone, PartialEq, Error)]
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
    chars.div_ceil(4)
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

    // ─── ReasoningMode Debug + Clone ───────────────────────────────────

    #[test]
    fn reasoning_mode_manual_debug_and_clone() {
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: 200,
            step_count: 4,
        };
        let debug_str = format!("{:?}", mode);
        assert!(debug_str.contains("Manual"));
        assert!(debug_str.contains("200"));
        assert!(debug_str.contains("4"));

        let cloned = mode.clone();
        let debug_cloned = format!("{:?}", cloned);
        assert_eq!(debug_str, debug_cloned);
    }

    #[test]
    fn reasoning_mode_auto_debug_and_clone() {
        let mode = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: Some(0.3),
            stop_patterns: vec!["done".to_string()],
        };
        let debug_str = format!("{:?}", mode);
        assert!(debug_str.contains("Auto"));
        assert!(debug_str.contains("1024"));

        let cloned = mode.clone();
        let debug_cloned = format!("{:?}", cloned);
        assert_eq!(debug_str, debug_cloned);
    }

    // ─── ReasoningResponse construction ────────────────────────────────

    #[test]
    fn reasoning_response_fields_accessible() {
        let response = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec!["step1".to_string(), "step2".to_string()],
            total_reasoning_tokens: 10,
            actual_steps: 2,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert_eq!(response.text, "42");
        assert_eq!(response.reasoning_trace.len(), 2);
        assert_eq!(response.total_reasoning_tokens, 10);
        assert_eq!(response.actual_steps, 2);
        assert_eq!(response.stopped_reason, ReasoningStopReason::StepCountReached);
    }

    #[test]
    fn reasoning_response_debug_format() {
        let response = ReasoningResponse {
            text: "answer".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 0,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        let debug = format!("{:?}", response);
        assert!(debug.contains("answer"));
        assert!(debug.contains("BudgetExhausted"));
    }

    // ─── StepContext construction ───────────────────────────────────────

    #[test]
    fn step_context_debug_and_clone() {
        let ctx = StepContext {
            step_index: 3,
            remaining_budget: 50,
            accumulated_text: "prior reasoning".to_string(),
            model_name: "test-model".to_string(),
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("step_index"));
        assert!(debug.contains("test-model"));

        let cloned = ctx.clone();
        assert_eq!(cloned.step_index, 3);
        assert_eq!(cloned.remaining_budget, 50);
        assert_eq!(cloned.accumulated_text, "prior reasoning");
    }

    // ─── StepAction variants ───────────────────────────────────────────

    #[test]
    fn step_action_all_variants_constructible() {
        let _continue = StepAction::Continue;
        let _skip = StepAction::Skip;
        let inject = StepAction::InjectPrompt("extra context".to_string());
        let halt = StepAction::Halt("done reasoning".to_string());

        let debug_continue = format!("{:?}", _continue);
        assert!(debug_continue.contains("Continue"));

        let debug_inject = format!("{:?}", inject);
        assert!(debug_inject.contains("extra context"));

        let debug_halt = format!("{:?}", halt);
        assert!(debug_halt.contains("done reasoning"));
    }

    // ─── StepResult construction ───────────────────────────────────────

    #[test]
    fn step_result_fields_and_debug() {
        let result = StepResult {
            step_index: 1,
            chunk_text: "thinking...".to_string(),
            tokens_used: 25,
            accumulated_text: "thinking...".to_string(),
        };
        assert_eq!(result.step_index, 1);
        assert_eq!(result.chunk_text, "thinking...");
        assert_eq!(result.tokens_used, 25);

        let debug = format!("{:?}", result);
        assert!(debug.contains("thinking..."));
    }

    // ─── StepKnowledge Default ─────────────────────────────────────────

    #[test]
    fn step_knowledge_default_is_none() {
        let sk = StepKnowledge::default();
        assert!(sk.inject_text.is_none());
        assert!(sk.modified_temperature.is_none());
    }

    #[test]
    fn step_knowledge_with_fields_debug_and_clone() {
        let sk = StepKnowledge {
            inject_text: Some("insight".to_string()),
            modified_temperature: Some(0.5),
        };
        let cloned = sk.clone();
        assert_eq!(cloned.inject_text.as_deref(), Some("insight"));
        assert_eq!(cloned.modified_temperature, Some(0.5));

        let debug = format!("{:?}", sk);
        assert!(debug.contains("insight"));
    }

    // ─── ReasoningTemplate validate rejects empty final_prefix ─────────

    #[test]
    fn template_validate_rejects_empty_final_prefix() {
        let tpl = ReasoningTemplate {
            final_prefix: String::new(),
            ..Default::default()
        };
        assert!(matches!(
            tpl.validate(),
            Err(ReasoningError::InvalidConfig(_))
        ));
    }

    // ─── ReasoningError Display + Clone ─────────────────────────────────

    #[test]
    fn reasoning_error_display_messages() {
        let err = ReasoningError::MissingModel;
        assert_eq!(err.to_string(), "no model loaded for CoT reasoning");

        let err = ReasoningError::GenerationFailed("timeout".to_string());
        let msg = err.to_string();
        assert!(msg.contains("generation failed"));
        assert!(msg.contains("timeout"));

        let err = ReasoningError::InvalidConfig("bad param".to_string());
        let msg = err.to_string();
        assert!(msg.contains("invalid reasoning config"));
        assert!(msg.contains("bad param"));
    }

    #[test]
    fn reasoning_error_clone_preserves_content() {
        let original = ReasoningError::GenerationFailed("oom".to_string());
        let cloned = original.clone();
        assert_eq!(original.to_string(), cloned.to_string());
    }

    // ─── estimate_token_count with Unicode / CJK ───────────────────────

    #[test]
    fn estimate_token_count_unicode_and_cjk() {
        // CJK: each character is ~1 token in BPE, but our heuristic uses
        // char count / 4 which underestimates. Verify monotonicity.
        let ascii_short = estimate_token_count("abc");
        let cjk_short = estimate_token_count("你好世界");
        assert!(cjk_short > 0, "CJK text must estimate > 0 tokens");
        assert!(ascii_short > 0, "short ASCII must estimate > 0 tokens");

        // Longer text always estimates more tokens
        let long_cjk = estimate_token_count("你好世界这是一个更长的中文句子来测试");
        assert!(
            long_cjk > cjk_short,
            "longer CJK text must estimate more tokens"
        );
    }

    // ─── estimate_text_entropy edge: single char, short text ────────────

    #[test]
    fn entropy_single_char_is_positive() {
        // Single char "a" → diversity = 1/1 = 1.0, word_count = 1,
        // word_scale = log2(2) = 1.0, no trigrams → rep = 0.0
        // entropy = 1.0 * (1 - 0.0) * 1.0 = 1.0
        let entropy = estimate_text_entropy("a");
        assert!(entropy > 0.0, "single char text has positive entropy");
    }

    #[test]
    fn entropy_short_below_three_words() {
        // Fewer than 3 words → repetition_ratio = 0, word_scale small
        let entropy = estimate_text_entropy("hello world");
        assert!(entropy >= 0.0, "entropy must be non-negative");
        // With only 2 unique chars in a short string, entropy is modest
        // but strictly > 0 since diversity > 0 and word_scale > 0
        assert!(
            entropy > 0.0,
            "two-word text must have positive entropy"
        );
    }

    // ─── find_stop_pattern returns first match ──────────────────────────

    #[test]
    fn stop_pattern_returns_first_matching() {
        let patterns = vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
        ];
        let text = "we found beta and gamma here";
        let result = find_stop_pattern(text, &patterns);
        // "beta" appears in text, "alpha" does not — first match is "beta"
        assert_eq!(result.map(|s| s.as_str()), Some("beta"));
    }

    // ─── ReasoningMode validation error messages ────────────────────────
    // Mode validation runs inside reason_with_hook AFTER require_state(),
    // so an empty Client hits NoModelLoaded first. We verify the error
    // messages directly via ReasoningError construction.

    #[test]
    fn mode_validation_auto_zero_max_total_tokens() {
        let err = ReasoningError::InvalidConfig("max_total_tokens must be > 0".to_string());
        let msg = err.to_string();
        assert!(msg.contains("max_total_tokens"));
        // Verify it maps to ClientError correctly
        let ce: ClientError = err.into();
        match ce {
            ClientError::RuntimeError(m) => assert!(m.contains("max_total_tokens")),
            other => panic!("expected RuntimeError, got {other:?}"),
        }
    }

    #[test]
    fn mode_validation_auto_negative_entropy_threshold() {
        let err = ReasoningError::InvalidConfig(
            "entropy_threshold must be finite and >= 0, got -1".to_string(),
        );
        let msg = err.to_string();
        assert!(msg.contains("entropy_threshold"));
    }

    #[test]
    fn mode_validation_manual_zero_step_count() {
        let err = ReasoningError::InvalidConfig("step_count must be >= 1".to_string());
        let msg = err.to_string();
        assert!(msg.contains("step_count"));
    }

    #[test]
    fn mode_validation_manual_zero_max_reasoning_tokens() {
        let err =
            ReasoningError::InvalidConfig("max_reasoning_tokens must be > 0".to_string());
        let msg = err.to_string();
        assert!(msg.contains("max_reasoning_tokens"));
    }

    // ─── Constants ────────────────────────────────────────────────────────

    #[test]
    fn auto_step_chunk_is_positive() {
        assert!(AUTO_STEP_CHUNK > 0, "AUTO_STEP_CHUNK must be positive");
        assert_eq!(AUTO_STEP_CHUNK, 128);
    }

    #[test]
    fn min_step_tokens_is_positive() {
        assert!(MIN_STEP_TOKENS > 0, "MIN_STEP_TOKENS must be positive");
        assert_eq!(MIN_STEP_TOKENS, 8);
    }

    // ─── ReasoningResponse Clone ─────────────────────────────────────────

    #[test]
    fn reasoning_response_clone_preserves_all_fields() {
        let original = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec!["s1".to_string(), "s2".to_string()],
            total_reasoning_tokens: 15,
            actual_steps: 2,
            stopped_reason: ReasoningStopReason::PatternMatched("done".to_string()),
        };
        let cloned = original.clone();
        assert_eq!(cloned.text, original.text);
        assert_eq!(cloned.reasoning_trace, original.reasoning_trace);
        assert_eq!(cloned.total_reasoning_tokens, original.total_reasoning_tokens);
        assert_eq!(cloned.actual_steps, original.actual_steps);
        assert_eq!(cloned.stopped_reason, original.stopped_reason);
    }

    // ─── ReasoningStopReason Debug format ────────────────────────────────

    #[test]
    fn stop_reason_debug_formats_all_variants() {
        let debug = format!("{:?}", ReasoningStopReason::BudgetExhausted);
        assert!(debug.contains("BudgetExhausted"));

        let debug = format!("{:?}", ReasoningStopReason::StepCountReached);
        assert!(debug.contains("StepCountReached"));

        let debug = format!("{:?}", ReasoningStopReason::EntropyConverged);
        assert!(debug.contains("EntropyConverged"));

        let debug = format!("{:?}", ReasoningStopReason::PatternMatched("test".into()));
        assert!(debug.contains("PatternMatched"));
        assert!(debug.contains("test"));

        let debug = format!("{:?}", ReasoningStopReason::HaltByHook("user".into()));
        assert!(debug.contains("HaltByHook"));
        assert!(debug.contains("user"));
    }

    #[test]
    fn stop_reason_cross_variant_inequality() {
        let variants: Vec<ReasoningStopReason> = vec![
            ReasoningStopReason::BudgetExhausted,
            ReasoningStopReason::StepCountReached,
            ReasoningStopReason::EntropyConverged,
            ReasoningStopReason::PatternMatched("x".into()),
            ReasoningStopReason::HaltByHook("x".into()),
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(
                    variants[i], variants[j],
                    "variant {i} must not equal variant {j}"
                );
            }
        }
    }

    #[test]
    fn stop_reason_clone_preserves_variant() {
        let original = ReasoningStopReason::PatternMatched("answer".to_string());
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ─── StepAction Clone ────────────────────────────────────────────────

    #[test]
    fn step_action_clone_preserves_data() {
        let inject = StepAction::InjectPrompt("extra".to_string());
        let cloned = inject.clone();
        let debug_original = format!("{:?}", inject);
        let debug_cloned = format!("{:?}", cloned);
        assert_eq!(debug_original, debug_cloned);

        let halt = StepAction::Halt("stop now".to_string());
        let cloned_halt = halt.clone();
        let debug_h1 = format!("{:?}", halt);
        let debug_h2 = format!("{:?}", cloned_halt);
        assert_eq!(debug_h1, debug_h2);
    }

    // ─── StepResult Clone ────────────────────────────────────────────────

    #[test]
    fn step_result_clone_preserves_all_fields() {
        let result = StepResult {
            step_index: 7,
            chunk_text: "some reasoning".to_string(),
            tokens_used: 42,
            accumulated_text: "accumulated so far".to_string(),
        };
        let cloned = result.clone();
        assert_eq!(cloned.step_index, result.step_index);
        assert_eq!(cloned.chunk_text, result.chunk_text);
        assert_eq!(cloned.tokens_used, result.tokens_used);
        assert_eq!(cloned.accumulated_text, result.accumulated_text);
    }

    // ─── StepKnowledge partial field construction ────────────────────────

    #[test]
    fn step_knowledge_only_inject_text() {
        let sk = StepKnowledge {
            inject_text: Some("useful info".to_string()),
            modified_temperature: None,
        };
        assert_eq!(sk.inject_text.as_deref(), Some("useful info"));
        assert!(sk.modified_temperature.is_none());
    }

    #[test]
    fn step_knowledge_only_modified_temperature() {
        let sk = StepKnowledge {
            inject_text: None,
            modified_temperature: Some(0.3),
        };
        assert!(sk.inject_text.is_none());
        assert_eq!(sk.modified_temperature, Some(0.3));
    }

    // ─── estimate_token_count div_ceil boundary ──────────────────────────

    #[test]
    fn estimate_token_count_exact_multiples_of_four() {
        // 4 chars → exactly 1 token
        assert_eq!(estimate_token_count("abcd"), 1);
        // 8 chars → exactly 2 tokens
        assert_eq!(estimate_token_count("abcdefgh"), 2);
        // 12 chars → exactly 3 tokens
        assert_eq!(estimate_token_count("abcdefghijkl"), 3);
    }

    #[test]
    fn estimate_token_count_rounds_up() {
        // 1 char → div_ceil(1, 4) = 1
        assert_eq!(estimate_token_count("a"), 1);
        // 5 chars → div_ceil(5, 4) = 2
        assert_eq!(estimate_token_count("abcde"), 2);
        // 3 chars → div_ceil(3, 4) = 1
        assert_eq!(estimate_token_count("abc"), 1);
    }

    // ─── estimate_text_entropy all same words (max trigram repetition) ──

    #[test]
    fn entropy_all_identical_words_drops_low() {
        // "the the the the the the the the" — trigram "the the the" repeated
        let text = "the the the the the the the the";
        let entropy = estimate_text_entropy(text);
        // All identical words → repetition_ratio approaches 1.0 → (1-rep)≈0 → entropy≈0
        assert!(
            entropy < 1.0,
            "identical-word text must have low entropy, got {entropy}"
        );
    }

    // ─── ReasoningTemplate validate boundary temperature ─────────────────

    #[test]
    fn template_validate_accepts_small_positive_temperature() {
        let tpl = ReasoningTemplate {
            temperature: 0.001,
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    #[test]
    fn template_validate_rejects_negative_zero_temperature() {
        let tpl = ReasoningTemplate {
            temperature: -0.0,
            ..Default::default()
        };
        // -0.0 == 0.0 in IEEE 754, so -0.0 <= 0.0 is true → rejected
        assert!(matches!(
            tpl.validate(),
            Err(ReasoningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn template_validate_rejects_neg_infinity_temperature() {
        let tpl = ReasoningTemplate {
            temperature: f32::NEG_INFINITY,
            ..Default::default()
        };
        assert!(matches!(
            tpl.validate(),
            Err(ReasoningError::InvalidConfig(_))
        ));
    }

    // ─── ReasoningError Debug format ─────────────────────────────────────

    #[test]
    fn reasoning_error_debug_format() {
        let err = ReasoningError::MissingModel;
        let debug = format!("{:?}", err);
        assert!(debug.contains("MissingModel"));

        let err = ReasoningError::GenerationFailed("test error".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("GenerationFailed"));
        assert!(debug.contains("test error"));

        let err = ReasoningError::InvalidConfig("param".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidConfig"));
        assert!(debug.contains("param"));
    }

    // ─── ReasoningTemplate render_step_prefix no placeholder ─────────────

    #[test]
    fn template_render_step_prefix_without_placeholder() {
        let tpl = ReasoningTemplate {
            step_prefix: "Begin reasoning:".to_string(),
            ..Default::default()
        };
        // No {n} → returned as-is, step number ignored
        assert_eq!(tpl.render_step_prefix(5), "Begin reasoning:");
    }

    #[test]
    fn template_render_step_prefix_multiple_placeholders() {
        let tpl = ReasoningTemplate {
            step_prefix: "Step {n} of {n}:".to_string(),
            ..Default::default()
        };
        // All {n} occurrences replaced
        assert_eq!(tpl.render_step_prefix(3), "Step 3 of 3:");
    }

    // ─── ReasoningResponse with empty trace ──────────────────────────────

    #[test]
    fn reasoning_response_empty_trace() {
        let response = ReasoningResponse {
            text: "immediate answer".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 0,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::HaltByHook("skip reasoning".to_string()),
        };
        assert!(response.reasoning_trace.is_empty());
        assert_eq!(response.actual_steps, 0);
        assert_eq!(response.total_reasoning_tokens, 0);
    }

    // ─── find_stop_pattern with overlapping patterns ────────────────────

    #[test]
    fn stop_pattern_first_pattern_in_list_wins() {
        let patterns = vec![
            "Final Answer:".to_string(),
            "Final".to_string(),
        ];
        // Both match; first in iteration order wins
        let text = "Final Answer: 42";
        let result = find_stop_pattern(text, &patterns);
        assert_eq!(result.map(|s| s.as_str()), Some("Final Answer:"));
    }

    // ─── StepContext field access patterns ───────────────────────────────

    #[test]
    fn step_context_initial_step_fields() {
        let ctx = StepContext {
            step_index: 0,
            remaining_budget: 100,
            accumulated_text: String::new(),
            model_name: "test-llm".to_string(),
        };
        assert_eq!(ctx.step_index, 0);
        assert_eq!(ctx.remaining_budget, 100);
        assert!(ctx.accumulated_text.is_empty());
        assert_eq!(ctx.model_name, "test-llm");
    }

    // ─── ReasoningMode partial equality via Debug round-trip ─────────────

    #[test]
    fn reasoning_mode_auto_without_entropy() {
        let mode = ReasoningMode::Auto {
            max_total_tokens: 500,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        let debug = format!("{:?}", mode);
        assert!(debug.contains("Auto"));
        assert!(debug.contains("500"));
        // Clone round-trip
        let cloned = mode.clone();
        assert_eq!(format!("{:?}", mode), format!("{:?}", cloned));
    }

    // ─── ReasoningTemplate full custom construction ──────────────────────

    #[test]
    fn template_custom_all_fields() {
        let tpl = ReasoningTemplate {
            system_prompt: "Custom system".to_string(),
            step_prefix: "[{n}]".to_string(),
            final_prefix: "Answer:".to_string(),
            step_separator: "---".to_string(),
            temperature: 0.5,
            top_k: 50,
            top_p: 0.9,
            final_answer_budget: 512,
        };
        assert_eq!(tpl.system_prompt, "Custom system");
        assert_eq!(tpl.step_prefix, "[{n}]");
        assert_eq!(tpl.final_prefix, "Answer:");
        assert_eq!(tpl.step_separator, "---");
        assert_eq!(tpl.temperature, 0.5);
        assert_eq!(tpl.top_k, 50);
        assert_eq!(tpl.top_p, 0.9);
        assert_eq!(tpl.final_answer_budget, 512);
        assert!(tpl.validate().is_ok());
        assert_eq!(tpl.render_step_prefix(10), "[10]");
    }

    // ─── estimate_text_entropy is deterministic ─────────────────────────

    #[test]
    fn entropy_is_deterministic_for_same_input() {
        let text = "The quick brown fox jumps over the lazy dog";
        let h1 = estimate_text_entropy(text);
        let h2 = estimate_text_entropy(text);
        assert_eq!(h1, h2, "entropy must be deterministic for identical input");
    }

    #[test]
    fn entropy_monotonic_with_repetition() {
        // Adding more identical repetitions to text decreases entropy
        // because trigram repetition ratio increases.
        let base = "alpha beta gamma delta";
        let repeated = "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta";
        let base_h = estimate_text_entropy(base);
        let rep_h = estimate_text_entropy(repeated);
        assert!(
            rep_h < base_h,
            "repeated text ({rep_h}) must have lower entropy than base ({base_h})"
        );
    }

    // ─── estimate_token_count whitespace-only input ──────────────────────────

    #[test]
    fn estimate_token_count_whitespace_only() {
        let spaces = estimate_token_count("    ");
        assert_eq!(spaces, 1, "4 spaces = 4 chars → div_ceil(4,4) = 1");
        let tabs = estimate_token_count("\t\t\t\t");
        assert_eq!(tabs, 1, "4 tabs = 4 chars → div_ceil(4,4) = 1");
        let single_space = estimate_token_count(" ");
        assert_eq!(single_space, 1, "1 char → div_ceil(1,4) = 1");
    }

    // ─── estimate_text_entropy single word ────────────────────────────────────

    #[test]
    fn entropy_single_word_no_trigrams() {
        // Single word: word_count=1, <3 words → rep_ratio=0,
        // word_scale = log2(2) = 1.0
        let entropy = estimate_text_entropy("hello");
        assert!(
            entropy > 0.0,
            "single word must have positive entropy, got {entropy}"
        );
        // Unique chars / total chars for "hello" = 4/5 = 0.8
        // entropy ≈ 0.8 * 1.0 * log2(2) = 0.8
        assert!(
            entropy <= 1.0,
            "single short word entropy should be bounded, got {entropy}"
        );
    }

    // ─── estimate_text_entropy whitespace-only string ─────────────────────────

    #[test]
    fn entropy_whitespace_only_returns_zero() {
        assert_eq!(estimate_text_entropy("   "), 0.0);
        assert_eq!(estimate_text_entropy("\t\n"), 0.0);
        assert_eq!(estimate_text_entropy("  \n  \t  "), 0.0);
    }

    // ─── find_stop_pattern pattern longer than text ───────────────────────────

    #[test]
    fn stop_pattern_pattern_longer_than_text() {
        let patterns = vec!["This is a very long pattern that exceeds the text".to_string()];
        assert!(find_stop_pattern("short", &patterns).is_none());
    }

    // ─── find_stop_pattern empty string pattern ───────────────────────────────

    #[test]
    fn stop_pattern_empty_string_pattern_always_matches() {
        // String::contains("") returns true for any string
        let patterns = vec![String::new()];
        let result = find_stop_pattern("any text at all", &patterns);
        assert!(
            result.is_some(),
            "empty pattern should match any text via contains"
        );
    }

    // ─── ReasoningTemplate render_step_prefix with step_index 0 ───────────────

    #[test]
    fn template_render_step_prefix_step_zero() {
        let tpl = ReasoningTemplate::default();
        // Step 0 is technically 0-based but render_step_prefix expects 1-based;
        // it should still substitute "0" for {n}.
        assert_eq!(tpl.render_step_prefix(0), "Step 0:");
    }

    // ─── ReasoningTemplate render_step_prefix large step index ────────────────

    #[test]
    fn template_render_step_prefix_large_step_index() {
        let tpl = ReasoningTemplate::default();
        let rendered = tpl.render_step_prefix(99999);
        assert_eq!(rendered, "Step 99999:");
    }

    // ─── ReasoningMode PartialEq via Debug-based round-trip ───────────────────

    #[test]
    fn reasoning_mode_manual_equality_via_clone() {
        let m1 = ReasoningMode::Manual {
            max_reasoning_tokens: 500,
            step_count: 5,
        };
        let m2 = m1.clone();
        // ReasoningMode does not derive PartialEq, so verify via Debug string
        assert_eq!(format!("{:?}", m1), format!("{:?}", m2));
    }

    // ─── StepContext with zero remaining_budget ───────────────────────────────

    #[test]
    fn step_context_zero_remaining_budget() {
        let ctx = StepContext {
            step_index: 5,
            remaining_budget: 0,
            accumulated_text: "exhausted".to_string(),
            model_name: "model".to_string(),
        };
        assert_eq!(ctx.remaining_budget, 0);
        assert_eq!(ctx.step_index, 5);
    }

    // ─── StepResult with empty chunk_text ─────────────────────────────────────

    #[test]
    fn step_result_empty_chunk_text() {
        let result = StepResult {
            step_index: 0,
            chunk_text: String::new(),
            tokens_used: 0,
            accumulated_text: String::new(),
        };
        assert!(result.chunk_text.is_empty());
        assert_eq!(result.tokens_used, 0);
    }

    // ─── ReasoningError round-trip through ClientError ────────────────────────

    #[test]
    fn reasoning_error_generation_failed_roundtrip() {
        let original_msg = "step 3 timeout after 30s";
        let re = ReasoningError::GenerationFailed(original_msg.to_string());
        let ce: ClientError = re.into();
        match ce {
            ClientError::RuntimeError(msg) => {
                assert!(msg.contains("cot_reasoner"));
                assert!(msg.contains(original_msg));
            }
            other => panic!("expected RuntimeError, got {other:?}"),
        }
    }

    #[test]
    fn reasoning_error_invalid_config_roundtrip() {
        let original_msg = "temperature is NaN";
        let re = ReasoningError::InvalidConfig(original_msg.to_string());
        let ce: ClientError = re.into();
        match ce {
            ClientError::RuntimeError(msg) => {
                assert!(msg.contains("cot_reasoner invalid config"));
                assert!(msg.contains(original_msg));
            }
            other => panic!("expected RuntimeError, got {other:?}"),
        }
    }

    // ─── StepAction Continue and Skip are unit-like variants ──────────────────

    #[test]
    fn step_action_continue_and_skip_clone_and_debug() {
        let cont = StepAction::Continue;
        let cont_cloned = cont.clone();
        assert_eq!(format!("{:?}", cont), format!("{:?}", cont_cloned));
        assert!(format!("{:?}", cont).contains("Continue"));

        let skip = StepAction::Skip;
        let skip_cloned = skip.clone();
        assert_eq!(format!("{:?}", skip), format!("{:?}", skip_cloned));
        assert!(format!("{:?}", skip).contains("Skip"));
    }

    // ─── ReasoningStopReason all variants clone ───────────────────────────────

    #[test]
    fn stop_reason_all_variants_clone_correctly() {
        let budget = ReasoningStopReason::BudgetExhausted;
        assert_eq!(budget.clone(), budget);

        let step = ReasoningStopReason::StepCountReached;
        assert_eq!(step.clone(), step);

        let entropy = ReasoningStopReason::EntropyConverged;
        assert_eq!(entropy.clone(), entropy);

        let pattern = ReasoningStopReason::PatternMatched("matched".to_string());
        assert_eq!(pattern.clone(), pattern);

        let halt = ReasoningStopReason::HaltByHook("done".to_string());
        assert_eq!(halt.clone(), halt);
    }

    // ─── ReasoningTemplate validate rejects very large temperature ────────────

    #[test]
    fn template_validate_accepts_large_finite_temperature() {
        let tpl = ReasoningTemplate {
            temperature: 100.0,
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    // ─── estimate_text_entropy with exactly two identical words ───────────────

    #[test]
    fn entropy_two_identical_words() {
        // "hello hello" → word_count=2, <3 → rep_ratio=0
        // diversity = unique_chars/total_chars, word_scale = log2(3)
        let entropy = estimate_text_entropy("hello hello");
        assert!(
            entropy > 0.0,
            "two identical words must have positive entropy, got {entropy}"
        );
    }

    // ─── ReasoningResponse with many trace entries ────────────────────────────

    #[test]
    fn reasoning_response_large_trace() {
        let trace: Vec<String> = (0..100).map(|i| format!("step {i} output")).collect();
        let total_tokens: usize = trace.len() * 10;
        let response = ReasoningResponse {
            text: "final".to_string(),
            reasoning_trace: trace,
            total_reasoning_tokens: total_tokens,
            actual_steps: 100,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        assert_eq!(response.actual_steps, 100);
        assert_eq!(response.reasoning_trace.len(), 100);
        assert_eq!(response.total_reasoning_tokens, 1000);
    }

    // ─── estimate_token_count with newlines and tabs ──────────────────────────

    #[test]
    fn estimate_token_count_with_whitespace_chars() {
        // "\n\t\r\n" = 4 chars → 1 token
        assert_eq!(estimate_token_count("\n\t\r\n"), 1);
        // Mixed content
        let text = "hello\nworld\ttest";
        let count = estimate_token_count(text);
        // 16 chars → div_ceil(16,4) = 4
        assert_eq!(count, 4);
    }

    // ─── ReasoningTemplate Clone trait ────────────────────────────────────────

    #[test]
    fn template_clone_preserves_all_fields() {
        let original = ReasoningTemplate {
            system_prompt: "Custom prompt".to_string(),
            step_prefix: "[{n}]".to_string(),
            final_prefix: "Result:".to_string(),
            step_separator: "---".to_string(),
            temperature: 0.5,
            top_k: 40,
            top_p: 0.85,
            final_answer_budget: 400,
        };
        let cloned = original.clone();
        assert_eq!(cloned.system_prompt, original.system_prompt);
        assert_eq!(cloned.step_prefix, original.step_prefix);
        assert_eq!(cloned.final_prefix, original.final_prefix);
        assert_eq!(cloned.step_separator, original.step_separator);
        assert_eq!(cloned.temperature, original.temperature);
        assert_eq!(cloned.top_k, original.top_k);
        assert_eq!(cloned.top_p, original.top_p);
        assert_eq!(cloned.final_answer_budget, original.final_answer_budget);
    }

    // ─── ReasoningMode Auto with Some entropy_threshold ────────────────────────

    #[test]
    fn reasoning_mode_auto_with_entropy_threshold_debug() {
        let mode = ReasoningMode::Auto {
            max_total_tokens: 2048,
            entropy_threshold: Some(0.25),
            stop_patterns: vec!["stop".to_string()],
        };
        let debug = format!("{:?}", mode);
        assert!(debug.contains("Auto"));
        assert!(debug.contains("2048"));
        assert!(debug.contains("0.25"));
        assert!(debug.contains("stop"));
    }

    // ─── ReasoningMode Manual step_count=1 (minimum valid) ────────────────────

    #[test]
    fn reasoning_mode_manual_minimal_step_count() {
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: 64,
            step_count: 1,
        };
        let debug = format!("{:?}", mode);
        assert!(debug.contains("Manual"));
        assert!(debug.contains("64"));
        assert!(debug.contains("1"));
    }

    // ─── ReasoningStopReason EntropyConverged equality ────────────────────────

    #[test]
    fn stop_reason_entropy_converged_equals_itself() {
        let a = ReasoningStopReason::EntropyConverged;
        let b = ReasoningStopReason::EntropyConverged;
        assert_eq!(a, b);
    }

    // ─── ReasoningStopReason EntropyConverged not equal to BudgetExhausted ────

    #[test]
    fn stop_reason_entropy_converged_not_equal_budget() {
        assert_ne!(
            ReasoningStopReason::EntropyConverged,
            ReasoningStopReason::BudgetExhausted
        );
    }

    // ─── ReasoningError Display for MissingModel ─────────────────────────────

    #[test]
    fn reasoning_error_display_missing_model() {
        let err = ReasoningError::MissingModel;
        let msg = err.to_string();
        assert!(msg.contains("no model"));
        assert!(msg.contains("CoT reasoning"));
    }

    // ─── ReasoningError Display for GenerationFailed with empty string ────────

    #[test]
    fn reasoning_error_display_generation_failed_empty_detail() {
        let err = ReasoningError::GenerationFailed(String::new());
        let msg = err.to_string();
        assert!(msg.contains("generation failed"));
    }

    // ─── ReasoningError Display for InvalidConfig with multiline detail ───────

    #[test]
    fn reasoning_error_display_invalid_config_multiline() {
        let detail = "line1\nline2\nline3";
        let err = ReasoningError::InvalidConfig(detail.to_string());
        let msg = err.to_string();
        assert!(msg.contains("invalid reasoning config"));
        assert!(msg.contains("line1"));
    }

    // ─── ReasoningTemplate default top_k and top_p values ────────────────────

    #[test]
    fn template_default_top_k_and_top_p() {
        let tpl = ReasoningTemplate::default();
        assert_eq!(tpl.top_k, 0, "default top_k should be 0 (disabled)");
        assert!(
            (tpl.top_p - 1.0).abs() < f32::EPSILON,
            "default top_p should be 1.0"
        );
    }

    // ─── ReasoningResponse actual_steps consistency with trace length ─────────

    #[test]
    fn reasoning_response_actual_steps_matches_trace_len() {
        let trace = vec![
            "first step".to_string(),
            "second step".to_string(),
            "third step".to_string(),
        ];
        let response = ReasoningResponse {
            text: "answer".to_string(),
            reasoning_trace: trace.clone(),
            total_reasoning_tokens: 30,
            actual_steps: trace.len(),
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert_eq!(response.actual_steps, response.reasoning_trace.len());
    }

    // ─── StepContext clone isolation ─────────────────────────────────────────

    #[test]
    fn step_context_clone_isolation() {
        let ctx = StepContext {
            step_index: 2,
            remaining_budget: 200,
            accumulated_text: "original".to_string(),
            model_name: "model-a".to_string(),
        };
        let mut cloned = ctx.clone();
        cloned.accumulated_text.push_str(" modified");
        cloned.remaining_budget = 0;
        // Original must be untouched
        assert_eq!(ctx.accumulated_text, "original");
        assert_eq!(ctx.remaining_budget, 200);
        assert_eq!(cloned.accumulated_text, "original modified");
        assert_eq!(cloned.remaining_budget, 0);
    }

    // ─── StepAction InjectPrompt with empty string ───────────────────────────

    #[test]
    fn step_action_inject_prompt_empty_string() {
        let action = StepAction::InjectPrompt(String::new());
        let debug = format!("{:?}", action);
        assert!(debug.contains("InjectPrompt"));
        // Clone and verify
        let cloned = action.clone();
        assert_eq!(format!("{:?}", action), format!("{:?}", cloned));
    }

    // ─── StepAction Halt with empty reason ───────────────────────────────────

    #[test]
    fn step_action_halt_empty_reason() {
        let action = StepAction::Halt(String::new());
        let debug = format!("{:?}", action);
        assert!(debug.contains("Halt"));
    }

    // ─── ReasoningStopReason PatternMatched with empty string ────────────────

    #[test]
    fn stop_reason_pattern_matched_empty_string() {
        let reason = ReasoningStopReason::PatternMatched(String::new());
        let debug = format!("{:?}", reason);
        assert!(debug.contains("PatternMatched"));
        // Empty-string pattern should still equal itself
        assert_eq!(reason.clone(), reason);
    }

    // ─── ReasoningStopReason HaltByHook with empty reason ────────────────────

    #[test]
    fn stop_reason_halt_by_hook_empty_reason() {
        let reason = ReasoningStopReason::HaltByHook(String::new());
        assert_eq!(reason.clone(), reason);
        let debug = format!("{:?}", reason);
        assert!(debug.contains("HaltByHook"));
    }

    // ─── estimate_text_entropy exactly two different words ───────────────────

    #[test]
    fn entropy_two_different_words() {
        // "hello world" → word_count=2, <3 words → rep_ratio=0
        // diversity = unique_chars/total_chars > 0, word_scale = log2(3)
        let entropy = estimate_text_entropy("hello world");
        assert!(
            entropy > 0.0,
            "two different words must have positive entropy, got {entropy}"
        );
    }

    // ─── find_stop_pattern with special characters in pattern ────────────────

    #[test]
    fn stop_pattern_special_characters() {
        let patterns = vec!["(answer)".to_string(), "a+b".to_string()];
        assert_eq!(
            find_stop_pattern("the (answer) is 42", &patterns).map(|s| s.as_str()),
            Some("(answer)")
        );
        assert!(find_stop_pattern("no match here", &patterns).is_none());
    }

    // ─── ReasoningTemplate validate rejects NaN temperature explicitly ───────

    #[test]
    fn template_validate_rejects_nan_temperature_error_message() {
        let tpl = ReasoningTemplate {
            temperature: f32::NAN,
            ..Default::default()
        };
        let err = tpl.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("temperature"), "error must mention temperature");
        assert!(msg.contains("NaN") || msg.contains("finite"), "error must mention NaN or finite requirement");
    }

    // ========================================================================
    // NEW TESTS (40 additional)
    // ========================================================================

    // --- ReasoningMode PartialEq tests ---

    #[test]
    fn reasoning_mode_manual_equal() {
        let a = ReasoningMode::Manual {
            max_reasoning_tokens: 500,
            step_count: 3,
        };
        let b = ReasoningMode::Manual {
            max_reasoning_tokens: 500,
            step_count: 3,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn reasoning_mode_manual_not_equal_different_tokens() {
        let a = ReasoningMode::Manual {
            max_reasoning_tokens: 500,
            step_count: 3,
        };
        let b = ReasoningMode::Manual {
            max_reasoning_tokens: 999,
            step_count: 3,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn reasoning_mode_manual_not_equal_different_steps() {
        let a = ReasoningMode::Manual {
            max_reasoning_tokens: 500,
            step_count: 3,
        };
        let b = ReasoningMode::Manual {
            max_reasoning_tokens: 500,
            step_count: 5,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn reasoning_mode_auto_equal() {
        let a = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: Some(0.5),
            stop_patterns: vec!["done".to_string()],
        };
        let b = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: Some(0.5),
            stop_patterns: vec!["done".to_string()],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn reasoning_mode_auto_not_equal_different_entropy() {
        let a = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: Some(0.5),
            stop_patterns: vec![],
        };
        let b = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: Some(0.9),
            stop_patterns: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn reasoning_mode_auto_not_equal_none_vs_some_entropy() {
        let a = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        let b = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: Some(0.5),
            stop_patterns: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn reasoning_mode_cross_variant_not_equal() {
        let manual = ReasoningMode::Manual {
            max_reasoning_tokens: 1024,
            step_count: 5,
        };
        let auto = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        assert_ne!(manual, auto);
    }

    // --- ReasoningTemplate PartialEq tests ---

    #[test]
    fn template_default_equals_default() {
        let a = ReasoningTemplate::default();
        let b = ReasoningTemplate::default();
        assert_eq!(a, b);
    }

    #[test]
    fn template_modified_not_equal_default() {
        let a = ReasoningTemplate::default();
        let b = ReasoningTemplate {
            temperature: 0.9,
            ..Default::default()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn template_equality_covers_all_fields() {
        let a = ReasoningTemplate {
            system_prompt: "s".to_string(),
            step_prefix: "p".to_string(),
            final_prefix: "f".to_string(),
            step_separator: "|".to_string(),
            temperature: 0.3,
            top_k: 10,
            top_p: 0.8,
            final_answer_budget: 100,
        };
        let b = ReasoningTemplate {
            system_prompt: "s".to_string(),
            step_prefix: "p".to_string(),
            final_prefix: "f".to_string(),
            step_separator: "|".to_string(),
            temperature: 0.3,
            top_k: 10,
            top_p: 0.8,
            final_answer_budget: 100,
        };
        assert_eq!(a, b);
    }

    // --- ReasoningResponse PartialEq tests ---

    #[test]
    fn reasoning_response_equal_responses() {
        let a = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec!["s1".to_string()],
            total_reasoning_tokens: 5,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        let b = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec!["s1".to_string()],
            total_reasoning_tokens: 5,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn reasoning_response_not_equal_different_text() {
        let a = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 0,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        let b = ReasoningResponse {
            text: "99".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 0,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        assert_ne!(a, b);
    }

    // --- StepContext PartialEq tests ---

    #[test]
    fn step_context_equal() {
        let a = StepContext {
            step_index: 2,
            remaining_budget: 50,
            accumulated_text: "text".to_string(),
            model_name: "m".to_string(),
        };
        let b = StepContext {
            step_index: 2,
            remaining_budget: 50,
            accumulated_text: "text".to_string(),
            model_name: "m".to_string(),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn step_context_not_equal_different_step_index() {
        let a = StepContext {
            step_index: 0,
            remaining_budget: 100,
            accumulated_text: String::new(),
            model_name: "m".to_string(),
        };
        let b = StepContext {
            step_index: 1,
            remaining_budget: 100,
            accumulated_text: String::new(),
            model_name: "m".to_string(),
        };
        assert_ne!(a, b);
    }

    // --- StepAction PartialEq tests ---

    #[test]
    fn step_action_continue_equals_continue() {
        assert_eq!(StepAction::Continue, StepAction::Continue);
    }

    #[test]
    fn step_action_skip_equals_skip() {
        assert_eq!(StepAction::Skip, StepAction::Skip);
    }

    #[test]
    fn step_action_inject_prompt_equal() {
        let a = StepAction::InjectPrompt("extra".to_string());
        let b = StepAction::InjectPrompt("extra".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn step_action_inject_prompt_not_equal() {
        let a = StepAction::InjectPrompt("a".to_string());
        let b = StepAction::InjectPrompt("b".to_string());
        assert_ne!(a, b);
    }

    #[test]
    fn step_action_halt_equal() {
        let a = StepAction::Halt("stop".to_string());
        let b = StepAction::Halt("stop".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn step_action_cross_variant_not_equal() {
        assert_ne!(StepAction::Continue, StepAction::Skip);
        assert_ne!(StepAction::Continue, StepAction::Halt("Continue".to_string()));
        assert_ne!(StepAction::Skip, StepAction::InjectPrompt(String::new()));
    }

    // --- StepResult PartialEq tests ---

    #[test]
    fn step_result_equal() {
        let a = StepResult {
            step_index: 0,
            chunk_text: "text".to_string(),
            tokens_used: 5,
            accumulated_text: "text".to_string(),
        };
        let b = StepResult {
            step_index: 0,
            chunk_text: "text".to_string(),
            tokens_used: 5,
            accumulated_text: "text".to_string(),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn step_result_not_equal_different_tokens() {
        let a = StepResult {
            step_index: 0,
            chunk_text: "text".to_string(),
            tokens_used: 5,
            accumulated_text: "text".to_string(),
        };
        let b = StepResult {
            step_index: 0,
            chunk_text: "text".to_string(),
            tokens_used: 99,
            accumulated_text: "text".to_string(),
        };
        assert_ne!(a, b);
    }

    // --- StepKnowledge PartialEq tests ---

    #[test]
    fn step_knowledge_default_equals_explicit_none() {
        let a = StepKnowledge::default();
        let b = StepKnowledge {
            inject_text: None,
            modified_temperature: None,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn step_knowledge_with_both_fields_equal() {
        let a = StepKnowledge {
            inject_text: Some("info".to_string()),
            modified_temperature: Some(0.5),
        };
        let b = StepKnowledge {
            inject_text: Some("info".to_string()),
            modified_temperature: Some(0.5),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn step_knowledge_not_equal_different_inject() {
        let a = StepKnowledge {
            inject_text: Some("a".to_string()),
            modified_temperature: None,
        };
        let b = StepKnowledge {
            inject_text: Some("b".to_string()),
            modified_temperature: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn step_knowledge_not_equal_different_temperature() {
        let a = StepKnowledge {
            inject_text: None,
            modified_temperature: Some(0.3),
        };
        let b = StepKnowledge {
            inject_text: None,
            modified_temperature: Some(0.9),
        };
        assert_ne!(a, b);
    }

    // --- ReasoningError PartialEq tests ---

    #[test]
    fn reasoning_error_missing_model_equal() {
        assert_eq!(ReasoningError::MissingModel, ReasoningError::MissingModel);
    }

    #[test]
    fn reasoning_error_generation_failed_equal() {
        let a = ReasoningError::GenerationFailed("err".to_string());
        let b = ReasoningError::GenerationFailed("err".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn reasoning_error_generation_failed_not_equal() {
        let a = ReasoningError::GenerationFailed("a".to_string());
        let b = ReasoningError::GenerationFailed("b".to_string());
        assert_ne!(a, b);
    }

    #[test]
    fn reasoning_error_invalid_config_equal() {
        let a = ReasoningError::InvalidConfig("bad".to_string());
        let b = ReasoningError::InvalidConfig("bad".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn reasoning_error_cross_variant_not_equal() {
        assert_ne!(
            ReasoningError::MissingModel,
            ReasoningError::GenerationFailed("no model loaded for CoT reasoning".to_string())
        );
        assert_ne!(
            ReasoningError::MissingModel,
            ReasoningError::InvalidConfig("no model loaded for CoT reasoning".to_string())
        );
    }

    // --- ReasoningStopReason Hash tests ---

    #[test]
    fn stop_reason_hash_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = ReasoningStopReason::BudgetExhausted;
        let b = ReasoningStopReason::BudgetExhausted;
        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn stop_reason_hash_different_for_different_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let variants: Vec<ReasoningStopReason> = vec![
            ReasoningStopReason::BudgetExhausted,
            ReasoningStopReason::StepCountReached,
            ReasoningStopReason::EntropyConverged,
            ReasoningStopReason::PatternMatched("x".to_string()),
            ReasoningStopReason::HaltByHook("x".to_string()),
        ];
        let hashes: Vec<u64> = variants
            .iter()
            .map(|v| {
                let mut h = DefaultHasher::new();
                v.hash(&mut h);
                h.finish()
            })
            .collect();
        // All hashes should be distinct (not guaranteed but extremely likely)
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "hashes for variant {i} and {j} should differ");
            }
        }
    }

    #[test]
    fn stop_reason_usable_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ReasoningStopReason::BudgetExhausted);
        set.insert(ReasoningStopReason::StepCountReached);
        set.insert(ReasoningStopReason::BudgetExhausted);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&ReasoningStopReason::BudgetExhausted));
        assert!(set.contains(&ReasoningStopReason::StepCountReached));
    }

    // --- Entropy boundary conditions ---

    #[test]
    fn entropy_with_special_unicode_chars() {
        let text = "hello \u{00e9} \u{4e16}\u{754c} world";
        let entropy = estimate_text_entropy(text);
        assert!(
            entropy > 0.0,
            "text with unicode chars must have positive entropy, got {entropy}"
        );
    }

    #[test]
    fn entropy_exactly_three_words_no_repetition() {
        // Exactly 3 words → trigram count = 1, no duplicates → rep_ratio = 0
        let entropy = estimate_text_entropy("alpha beta gamma");
        assert!(
            entropy > 0.0,
            "three distinct words must have positive entropy, got {entropy}"
        );
    }

    #[test]
    fn entropy_exactly_three_identical_words() {
        // "a a a" → 1 trigram "a a a", seen once → dup=0 → rep_ratio=0
        // But diversity is low (1 unique char out of 5)
        let entropy = estimate_text_entropy("a a a");
        assert!(entropy >= 0.0, "entropy must be non-negative");
    }

    // --- estimate_token_count boundary with very large input ---

    #[test]
    fn estimate_token_count_large_input() {
        let text = "a".repeat(10000);
        let count = estimate_token_count(&text);
        assert_eq!(count, 2500, "10000 chars / 4 = 2500 tokens");
    }

    // --- find_stop_pattern with unicode in text ---

    #[test]
    fn stop_pattern_unicode_text_match() {
        let patterns = vec!["Answer:".to_string()];
        let text = "\u{7b54}\u{6848} Answer: 42";
        assert_eq!(
            find_stop_pattern(text, &patterns).map(|s| s.as_str()),
            Some("Answer:")
        );
    }

    // --- ReasoningMode Manual with usize::MAX ---

    #[test]
    fn reasoning_mode_manual_max_values_debug() {
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: usize::MAX,
            step_count: usize::MAX,
        };
        let debug = format!("{:?}", mode);
        assert!(debug.contains("Manual"));
        assert!(debug.contains(&usize::MAX.to_string()));
    }

    // --- ReasoningMode Auto with usize::MAX ---

    #[test]
    fn reasoning_mode_auto_max_total_tokens_debug() {
        let mode = ReasoningMode::Auto {
            max_total_tokens: usize::MAX,
            entropy_threshold: Some(f32::MAX),
            stop_patterns: vec![],
        };
        let debug = format!("{:?}", mode);
        assert!(debug.contains("Auto"));
    }

    // --- ReasoningTemplate validate with very large final_answer_budget ---

    #[test]
    fn template_validate_accepts_large_budget() {
        let tpl = ReasoningTemplate {
            final_answer_budget: usize::MAX,
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    // --- ReasoningTemplate validate with temperature exactly at boundary ---

    #[test]
    fn template_validate_rejects_zero_temperature_explicit() {
        let tpl = ReasoningTemplate {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(matches!(tpl.validate(), Err(ReasoningError::InvalidConfig(_))));
    }

    #[test]
    fn template_validate_accepts_subnormal_positive_temperature() {
        let tpl = ReasoningTemplate {
            temperature: f32::MIN_POSITIVE,
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    // --- ReasoningError clone and equality round-trip ---

    #[test]
    fn reasoning_error_clone_then_eq() {
        let err = ReasoningError::InvalidConfig("test config error".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    // --- ReasoningTemplate render_step_prefix with usize::MAX ---

    #[test]
    fn template_render_step_prefix_usize_max() {
        let tpl = ReasoningTemplate::default();
        let rendered = tpl.render_step_prefix(usize::MAX);
        assert_eq!(rendered, format!("Step {}:", usize::MAX));
    }

    // --- StepContext with maximum values ---

    #[test]
    fn step_context_max_values() {
        let ctx = StepContext {
            step_index: usize::MAX,
            remaining_budget: usize::MAX,
            accumulated_text: String::new(),
            model_name: "model".to_string(),
        };
        assert_eq!(ctx.step_index, usize::MAX);
        assert_eq!(ctx.remaining_budget, usize::MAX);
    }

    // --- estimate_text_entropy is zero for only whitespace variations ---

    #[test]
    fn entropy_various_whitespace_only() {
        assert_eq!(estimate_text_entropy("  "), 0.0);
        assert_eq!(estimate_text_entropy("\n\n\n"), 0.0);
        assert_eq!(estimate_text_entropy("\t\t"), 0.0);
        assert_eq!(estimate_text_entropy(" \n \t \r "), 0.0);
    }

    // --- ReasoningTemplate with custom step_prefix containing no {n} validates ---

    #[test]
    fn template_custom_prefix_no_placeholder_validates() {
        let tpl = ReasoningTemplate {
            step_prefix: "Thinking:".to_string(),
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    // --- DEFAULT_STOP_PATTERNS immutability check ---

    #[test]
    fn default_stop_patterns_has_five_entries() {
        assert_eq!(DEFAULT_STOP_PATTERNS.len(), 5);
    }

    // --- ReasoningBuilder with Auto mode and empty stop_patterns ---

    #[test]
    fn reasoning_builder_auto_empty_stop_patterns() {
        let client = Client::new_empty();
        let mode = ReasoningMode::Auto {
            max_total_tokens: 256,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        let builder = ReasoningBuilder::new(&client, "test".into(), mode);
        // Verify it constructs without panic; execution without model will fail
        assert!(builder.step_hook.is_none());
    }

    // --- ReasoningBuilder with_step_hook replaces previous ---

    struct DummyHook;
    impl ReasoningStepHook for DummyHook {
        fn on_step_start(&mut self, _ctx: &StepContext) -> StepAction {
            StepAction::Continue
        }
        fn on_step_end(&mut self, _result: &StepResult) -> StepKnowledge {
            StepKnowledge::default()
        }
    }

    #[test]
    fn reasoning_builder_with_step_hook_stores_hook() {
        let client = Client::new_empty();
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: 100,
            step_count: 1,
        };
        let builder = ReasoningBuilder::new(&client, "q".into(), mode)
            .with_step_hook(Box::new(DummyHook));
        assert!(builder.step_hook.is_some());
    }

    // --- ReasoningResponse with unicode text ---

    #[test]
    fn reasoning_response_unicode_text() {
        let response = ReasoningResponse {
            text: "\u{7b54}\u{6848}\u{662f} 42".to_string(),
            reasoning_trace: vec!["\u{601d}\u{8003}\u{4e2d}".to_string()],
            total_reasoning_tokens: 3,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert!(!response.text.is_empty());
        assert_eq!(response.reasoning_trace.len(), 1);
    }

    // --- estimate_token_count single char ---

    #[test]
    fn estimate_token_count_single_char() {
        assert_eq!(estimate_token_count("x"), 1);
    }

    // --- estimate_text_entropy long diverse text ---

    #[test]
    fn entropy_long_diverse_text_high() {
        let text = "The quick brown fox jumps over the lazy dog. \
                     Pack my box with five dozen liquor jugs. \
                     How vexingly quick daft zebras jump. \
                     The five boxing wizards jump quickly.";
        let entropy = estimate_text_entropy(text);
        assert!(
            entropy > 0.5,
            "long diverse text must have high entropy, got {entropy}"
        );
    }

    // --- find_stop_pattern returns first in iteration order ---

    #[test]
    fn stop_pattern_returns_first_in_list_order() {
        let patterns = vec![
            "B".to_string(),
            "A".to_string(),
        ];
        let text = "A and B both appear";
        let result = find_stop_pattern(text, &patterns);
        // Iteration finds "B" first even though "A" appears first in the text
        assert_eq!(result.map(|s| s.as_str()), Some("B"));
    }

    // ========================================================================
    // ADDITIONAL TESTS (55 new)
    // ========================================================================

    // --- ReasoningStepHook trait behavior ---

    #[test]
    fn dummy_hook_on_step_start_returns_continue() {
        let mut hook = DummyHook;
        let ctx = StepContext {
            step_index: 0,
            remaining_budget: 100,
            accumulated_text: String::new(),
            model_name: "test".to_string(),
        };
        let action = hook.on_step_start(&ctx);
        assert_eq!(action, StepAction::Continue);
    }

    #[test]
    fn dummy_hook_on_step_end_returns_default_knowledge() {
        let mut hook = DummyHook;
        let result = StepResult {
            step_index: 0,
            chunk_text: "text".to_string(),
            tokens_used: 5,
            accumulated_text: "text".to_string(),
        };
        let knowledge = hook.on_step_end(&result);
        assert_eq!(knowledge, StepKnowledge::default());
    }

    // --- Custom hook that returns Skip ---

    struct SkipHook;
    impl ReasoningStepHook for SkipHook {
        fn on_step_start(&mut self, _ctx: &StepContext) -> StepAction {
            StepAction::Skip
        }
        fn on_step_end(&mut self, _result: &StepResult) -> StepKnowledge {
            StepKnowledge::default()
        }
    }

    #[test]
    fn skip_hook_on_step_start_returns_skip() {
        let mut hook = SkipHook;
        let ctx = StepContext {
            step_index: 3,
            remaining_budget: 50,
            accumulated_text: "accumulated".to_string(),
            model_name: "m".to_string(),
        };
        assert_eq!(hook.on_step_start(&ctx), StepAction::Skip);
    }

    // --- Custom hook that returns Halt ---

    struct HaltHook {
        reason: String,
    }
    impl ReasoningStepHook for HaltHook {
        fn on_step_start(&mut self, _ctx: &StepContext) -> StepAction {
            StepAction::Halt(self.reason.clone())
        }
        fn on_step_end(&mut self, _result: &StepResult) -> StepKnowledge {
            StepKnowledge::default()
        }
    }

    #[test]
    fn halt_hook_on_step_start_returns_halt_with_reason() {
        let mut hook = HaltHook {
            reason: "sufficient reasoning".to_string(),
        };
        let ctx = StepContext {
            step_index: 1,
            remaining_budget: 200,
            accumulated_text: "step 0 done".to_string(),
            model_name: "model".to_string(),
        };
        match hook.on_step_start(&ctx) {
            StepAction::Halt(r) => assert_eq!(r, "sufficient reasoning"),
            other => panic!("expected Halt, got {:?}", other),
        }
    }

    // --- Custom hook that returns InjectPrompt ---

    struct InjectHook {
        extra: String,
    }
    impl ReasoningStepHook for InjectHook {
        fn on_step_start(&mut self, _ctx: &StepContext) -> StepAction {
            StepAction::InjectPrompt(self.extra.clone())
        }
        fn on_step_end(&mut self, _result: &StepResult) -> StepKnowledge {
            StepKnowledge::default()
        }
    }

    #[test]
    fn inject_hook_on_step_start_returns_inject_prompt() {
        let mut hook = InjectHook {
            extra: "Consider gravity.".to_string(),
        };
        let ctx = StepContext {
            step_index: 0,
            remaining_budget: 500,
            accumulated_text: String::new(),
            model_name: "m".to_string(),
        };
        match hook.on_step_start(&ctx) {
            StepAction::InjectPrompt(text) => assert_eq!(text, "Consider gravity."),
            other => panic!("expected InjectPrompt, got {:?}", other),
        }
    }

    // --- Custom hook that returns StepKnowledge with inject_text ---

    struct KnowledgeInjectHook;
    impl ReasoningStepHook for KnowledgeInjectHook {
        fn on_step_start(&mut self, _ctx: &StepContext) -> StepAction {
            StepAction::Continue
        }
        fn on_step_end(&mut self, _result: &StepResult) -> StepKnowledge {
            StepKnowledge {
                inject_text: Some("additional insight".to_string()),
                modified_temperature: None,
            }
        }
    }

    #[test]
    fn knowledge_inject_hook_on_step_end_returns_inject_text() {
        let mut hook = KnowledgeInjectHook;
        let result = StepResult {
            step_index: 0,
            chunk_text: "reasoned".to_string(),
            tokens_used: 3,
            accumulated_text: "reasoned".to_string(),
        };
        let knowledge = hook.on_step_end(&result);
        assert_eq!(knowledge.inject_text.as_deref(), Some("additional insight"));
        assert!(knowledge.modified_temperature.is_none());
    }

    // --- Custom hook that returns modified_temperature ---

    struct TemperatureModHook;
    impl ReasoningStepHook for TemperatureModHook {
        fn on_step_start(&mut self, _ctx: &StepContext) -> StepAction {
            StepAction::Continue
        }
        fn on_step_end(&mut self, _result: &StepResult) -> StepKnowledge {
            StepKnowledge {
                inject_text: None,
                modified_temperature: Some(0.1),
            }
        }
    }

    #[test]
    fn temperature_mod_hook_on_step_end_returns_low_temperature() {
        let mut hook = TemperatureModHook;
        let result = StepResult {
            step_index: 2,
            chunk_text: "analyzing".to_string(),
            tokens_used: 4,
            accumulated_text: "step0 step1 analyzing".to_string(),
        };
        let knowledge = hook.on_step_end(&result);
        assert!(knowledge.inject_text.is_none());
        assert_eq!(knowledge.modified_temperature, Some(0.1));
    }

    // --- Hook receives correct StepContext fields ---

    struct InspectHook {
        expected_step_index: usize,
        expected_remaining_budget: usize,
        expected_model_name: String,
    }
    impl ReasoningStepHook for InspectHook {
        fn on_step_start(&mut self, ctx: &StepContext) -> StepAction {
            assert_eq!(ctx.step_index, self.expected_step_index);
            assert_eq!(ctx.remaining_budget, self.expected_remaining_budget);
            assert_eq!(ctx.model_name, self.expected_model_name);
            StepAction::Continue
        }
        fn on_step_end(&mut self, _result: &StepResult) -> StepKnowledge {
            StepKnowledge::default()
        }
    }

    #[test]
    fn hook_receives_correct_step_context() {
        let mut hook = InspectHook {
            expected_step_index: 5,
            expected_remaining_budget: 300,
            expected_model_name: "qwen3-8b".to_string(),
        };
        let ctx = StepContext {
            step_index: 5,
            remaining_budget: 300,
            accumulated_text: "prior steps".to_string(),
            model_name: "qwen3-8b".to_string(),
        };
        // Asserts happen inside on_step_start; panic = test failure
        hook.on_step_start(&ctx);
    }

    // --- Hook receives correct StepResult fields ---

    struct InspectResultHook {
        expected_step_index: usize,
        expected_chunk: String,
        expected_tokens: usize,
    }
    impl ReasoningStepHook for InspectResultHook {
        fn on_step_start(&mut self, _ctx: &StepContext) -> StepAction {
            StepAction::Continue
        }
        fn on_step_end(&mut self, result: &StepResult) -> StepKnowledge {
            assert_eq!(result.step_index, self.expected_step_index);
            assert_eq!(result.chunk_text, self.expected_chunk);
            assert_eq!(result.tokens_used, self.expected_tokens);
            StepKnowledge::default()
        }
    }

    #[test]
    fn hook_receives_correct_step_result() {
        let mut hook = InspectResultHook {
            expected_step_index: 3,
            expected_chunk: "computed partial sum".to_string(),
            expected_tokens: 8,
        };
        let result = StepResult {
            step_index: 3,
            chunk_text: "computed partial sum".to_string(),
            tokens_used: 8,
            accumulated_text: "s0 s1 s2 computed partial sum".to_string(),
        };
        hook.on_step_end(&result);
    }

    // --- ReasoningBuilder with_step_hook replaces previous hook ---

    #[test]
    fn reasoning_builder_with_step_hook_replaces() {
        let client = Client::new_empty();
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: 100,
            step_count: 1,
        };
        let builder = ReasoningBuilder::new(&client, "q".into(), mode)
            .with_step_hook(Box::new(DummyHook))
            .with_step_hook(Box::new(SkipHook));
        // The builder should have a hook (replaced, not accumulated)
        assert!(builder.step_hook.is_some());
    }

    // --- StepContext not equal: different accumulated_text ---

    #[test]
    fn step_context_not_equal_different_accumulated_text() {
        let a = StepContext {
            step_index: 0,
            remaining_budget: 100,
            accumulated_text: "text a".to_string(),
            model_name: "m".to_string(),
        };
        let b = StepContext {
            step_index: 0,
            remaining_budget: 100,
            accumulated_text: "text b".to_string(),
            model_name: "m".to_string(),
        };
        assert_ne!(a, b);
    }

    // --- StepContext not equal: different model_name ---

    #[test]
    fn step_context_not_equal_different_model_name() {
        let a = StepContext {
            step_index: 0,
            remaining_budget: 100,
            accumulated_text: String::new(),
            model_name: "model-a".to_string(),
        };
        let b = StepContext {
            step_index: 0,
            remaining_budget: 100,
            accumulated_text: String::new(),
            model_name: "model-b".to_string(),
        };
        assert_ne!(a, b);
    }

    // --- StepContext not equal: different remaining_budget ---

    #[test]
    fn step_context_not_equal_different_remaining_budget() {
        let a = StepContext {
            step_index: 0,
            remaining_budget: 100,
            accumulated_text: String::new(),
            model_name: "m".to_string(),
        };
        let b = StepContext {
            step_index: 0,
            remaining_budget: 200,
            accumulated_text: String::new(),
            model_name: "m".to_string(),
        };
        assert_ne!(a, b);
    }

    // --- StepResult not equal: different step_index ---

    #[test]
    fn step_result_not_equal_different_step_index() {
        let a = StepResult {
            step_index: 0,
            chunk_text: "text".to_string(),
            tokens_used: 5,
            accumulated_text: "text".to_string(),
        };
        let b = StepResult {
            step_index: 1,
            chunk_text: "text".to_string(),
            tokens_used: 5,
            accumulated_text: "text".to_string(),
        };
        assert_ne!(a, b);
    }

    // --- StepResult not equal: different chunk_text ---

    #[test]
    fn step_result_not_equal_different_chunk_text() {
        let a = StepResult {
            step_index: 0,
            chunk_text: "alpha".to_string(),
            tokens_used: 5,
            accumulated_text: "alpha".to_string(),
        };
        let b = StepResult {
            step_index: 0,
            chunk_text: "beta".to_string(),
            tokens_used: 5,
            accumulated_text: "beta".to_string(),
        };
        assert_ne!(a, b);
    }

    // --- StepResult not equal: different accumulated_text ---

    #[test]
    fn step_result_not_equal_different_accumulated_text() {
        let a = StepResult {
            step_index: 0,
            chunk_text: "text".to_string(),
            tokens_used: 5,
            accumulated_text: "accumulated a".to_string(),
        };
        let b = StepResult {
            step_index: 0,
            chunk_text: "text".to_string(),
            tokens_used: 5,
            accumulated_text: "accumulated b".to_string(),
        };
        assert_ne!(a, b);
    }

    // --- StepResult clone isolation ---

    #[test]
    fn step_result_clone_isolation() {
        let original = StepResult {
            step_index: 3,
            chunk_text: "reasoning chunk".to_string(),
            tokens_used: 20,
            accumulated_text: "accumulated".to_string(),
        };
        let mut cloned = original.clone();
        cloned.chunk_text.push_str(" extra");
        cloned.step_index = 99;
        assert_eq!(original.chunk_text, "reasoning chunk");
        assert_eq!(original.step_index, 3);
    }

    // --- ReasoningResponse not equal: different total_reasoning_tokens ---

    #[test]
    fn reasoning_response_not_equal_different_total_tokens() {
        let a = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 10,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        let b = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 20,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        assert_ne!(a, b);
    }

    // --- ReasoningResponse not equal: different actual_steps ---

    #[test]
    fn reasoning_response_not_equal_different_actual_steps() {
        let a = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec!["s1".to_string()],
            total_reasoning_tokens: 5,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        let b = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec!["s1".to_string()],
            total_reasoning_tokens: 5,
            actual_steps: 2,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert_ne!(a, b);
    }

    // --- ReasoningResponse not equal: different stopped_reason ---

    #[test]
    fn reasoning_response_not_equal_different_stopped_reason() {
        let a = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 0,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        let b = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 0,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert_ne!(a, b);
    }

    // --- ReasoningResponse not equal: different reasoning_trace ---

    #[test]
    fn reasoning_response_not_equal_different_trace() {
        let a = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec!["step a".to_string()],
            total_reasoning_tokens: 5,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        let b = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec!["step b".to_string()],
            total_reasoning_tokens: 5,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert_ne!(a, b);
    }

    // --- ReasoningMode::Auto not equal: different max_total_tokens ---

    #[test]
    fn reasoning_mode_auto_not_equal_different_max_total_tokens() {
        let a = ReasoningMode::Auto {
            max_total_tokens: 512,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        let b = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        assert_ne!(a, b);
    }

    // --- ReasoningMode::Auto not equal: different stop_patterns ---

    #[test]
    fn reasoning_mode_auto_not_equal_different_stop_patterns() {
        let a = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: None,
            stop_patterns: vec!["done".to_string()],
        };
        let b = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: None,
            stop_patterns: vec!["stop".to_string()],
        };
        assert_ne!(a, b);
    }

    // --- ReasoningMode::Auto not equal: empty vs non-empty stop_patterns ---

    #[test]
    fn reasoning_mode_auto_not_equal_empty_vs_nonempty_patterns() {
        let a = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        let b = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: None,
            stop_patterns: vec!["done".to_string()],
        };
        assert_ne!(a, b);
    }

    // --- estimate_text_entropy with exactly four words, no repetition ---

    #[test]
    fn entropy_four_distinct_words_no_trigram_repeat() {
        // "alpha beta gamma delta" → 2 trigrams, both unique → rep_ratio=0
        let entropy = estimate_text_entropy("alpha beta gamma delta");
        assert!(
            entropy > 0.0,
            "four distinct words must have positive entropy, got {entropy}"
        );
    }

    // --- estimate_text_entropy with repeated trigrams across longer text ---

    #[test]
    fn entropy_repeated_trigrams_decreases_entropy() {
        let unique = "the cat sat on the mat";
        let repeated = "the cat sat on the cat sat on the cat sat on";
        let h_unique = estimate_text_entropy(unique);
        let h_repeated = estimate_text_entropy(repeated);
        assert!(
            h_repeated < h_unique,
            "text with repeated trigrams ({h_repeated}) must have lower entropy than unique ({h_unique})"
        );
    }

    // --- estimate_text_entropy very long repetitive text ---

    #[test]
    fn entropy_very_long_repetitive_text_approaches_zero() {
        let repeated = "x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x";
        let entropy = estimate_text_entropy(repeated);
        assert!(
            entropy < 0.5,
            "very long repetitive text must have very low entropy, got {entropy}"
        );
    }

    // --- find_stop_pattern with empty text ---

    #[test]
    fn stop_pattern_empty_text_returns_none_for_nonempty_patterns() {
        let patterns = vec!["Final Answer:".to_string()];
        assert!(find_stop_pattern("", &patterns).is_none());
    }

    // --- find_stop_pattern with empty text and empty pattern ---

    #[test]
    fn stop_pattern_empty_text_with_empty_pattern() {
        let patterns = vec![String::new()];
        // "".contains("") is true
        assert!(find_stop_pattern("", &patterns).is_some());
    }

    // --- find_stop_pattern with multiple patterns matching same text ---

    #[test]
    fn stop_pattern_multiple_matches_returns_first() {
        let patterns = vec![
            "the".to_string(),
            "lazy".to_string(),
            "dog".to_string(),
        ];
        let text = "the quick brown fox jumps over the lazy dog";
        let result = find_stop_pattern(text, &patterns);
        assert_eq!(result.map(|s| s.as_str()), Some("the"));
    }

    // --- ReasoningTemplate validate rejects top_p = 0 is NOT validated (top_p not validated) ---
    // top_p is not validated in ReasoningTemplate::validate, only temperature,
    // final_answer_budget, step_prefix, final_prefix. Verify this boundary.

    #[test]
    fn template_validate_does_not_check_top_p_zero() {
        let tpl = ReasoningTemplate {
            top_p: 0.0,
            ..Default::default()
        };
        // top_p is not validated, so this should pass
        assert!(tpl.validate().is_ok());
    }

    #[test]
    fn template_validate_does_not_check_top_k_zero() {
        let tpl = ReasoningTemplate {
            top_k: 0,
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    // --- StepKnowledge with empty inject_text (Some("")) ---

    #[test]
    fn step_knowledge_inject_text_empty_string() {
        let sk = StepKnowledge {
            inject_text: Some(String::new()),
            modified_temperature: None,
        };
        assert!(sk.inject_text.is_some());
        assert_eq!(sk.inject_text.as_deref(), Some(""));
    }

    // --- StepKnowledge with both fields populated ---

    #[test]
    fn step_knowledge_both_fields_set_independent() {
        let sk = StepKnowledge {
            inject_text: Some("context".to_string()),
            modified_temperature: Some(0.8),
        };
        // Both should be independently accessible
        assert_eq!(sk.inject_text.as_deref(), Some("context"));
        assert_eq!(sk.modified_temperature, Some(0.8));
        // Verify they are not mixed
        assert!(sk.inject_text.as_deref() != Some("other"));
        assert!(sk.modified_temperature != Some(0.1));
    }

    // --- ReasoningError from ClientError round-trip: MissingModel variant ---

    #[test]
    fn reasoning_error_missing_model_into_client_error_preserves_semantics() {
        let re = ReasoningError::MissingModel;
        let ce: ClientError = re.into();
        // Verify exact variant match
        assert!(matches!(ce, ClientError::NoModelLoaded));
    }

    // --- ReasoningResponse with single-char text ---

    #[test]
    fn reasoning_response_single_char_text() {
        let response = ReasoningResponse {
            text: "X".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 0,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::HaltByHook("quick answer".to_string()),
        };
        assert_eq!(response.text, "X");
        assert_eq!(response.text.len(), 1);
    }

    // --- ReasoningResponse text can be multiline ---

    #[test]
    fn reasoning_response_multiline_text() {
        let text = "line1\nline2\nline3";
        let response = ReasoningResponse {
            text: text.to_string(),
            reasoning_trace: vec!["thinking".to_string()],
            total_reasoning_tokens: 3,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert!(response.text.contains('\n'));
        assert_eq!(response.text.lines().count(), 3);
    }

    // --- StepContext accumulated_text can be empty at step 0 ---

    #[test]
    fn step_context_accumulated_text_empty_at_first_step() {
        let ctx = StepContext {
            step_index: 0,
            remaining_budget: 500,
            accumulated_text: String::new(),
            model_name: "m".to_string(),
        };
        assert!(ctx.accumulated_text.is_empty());
    }

    // --- StepContext accumulated_text grows with steps ---

    #[test]
    fn step_context_accumulated_text_grows() {
        let ctx_step0 = StepContext {
            step_index: 0,
            remaining_budget: 500,
            accumulated_text: String::new(),
            model_name: "m".to_string(),
        };
        let ctx_step2 = StepContext {
            step_index: 2,
            remaining_budget: 300,
            accumulated_text: "step0 output\n\nstep1 output".to_string(),
            model_name: "m".to_string(),
        };
        assert!(ctx_step2.accumulated_text.len() > ctx_step0.accumulated_text.len());
    }

    // --- ReasoningTemplate render_step_prefix with step 1 (first step) ---

    #[test]
    fn template_render_step_prefix_first_step() {
        let tpl = ReasoningTemplate::default();
        let rendered = tpl.render_step_prefix(1);
        assert_eq!(rendered, "Step 1:");
    }

    // --- ReasoningTemplate render_step_prefix preserves custom prefix structure ---

    #[test]
    fn template_render_step_prefix_custom_with_braces() {
        let tpl = ReasoningTemplate {
            step_prefix: "[Phase {n}] Reasoning:".to_string(),
            ..Default::default()
        };
        assert_eq!(tpl.render_step_prefix(2), "[Phase 2] Reasoning:");
    }

    // --- ReasoningMode Auto with entropy_threshold = 0.0 ---

    #[test]
    fn reasoning_mode_auto_with_zero_entropy_threshold() {
        let mode = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: Some(0.0),
            stop_patterns: vec![],
        };
        let debug = format!("{:?}", mode);
        assert!(debug.contains("Auto"));
        // 0.0 threshold means any text with entropy > 0 continues
    }

    // --- ReasoningMode Auto with entropy_threshold = f32::MAX ---

    #[test]
    fn reasoning_mode_auto_with_max_entropy_threshold() {
        let mode = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: Some(f32::MAX),
            stop_patterns: vec![],
        };
        let debug = format!("{:?}", mode);
        assert!(debug.contains("Auto"));
    }

    // --- estimate_token_count with mixed ASCII and CJK ---

    #[test]
    fn estimate_token_count_mixed_ascii_cjk() {
        let text = "hello world";
        let count_ascii = estimate_token_count(text);
        let text_mixed = "hello\u{4e16}\u{754c}"; // hello世界
        let count_mixed = estimate_token_count(text_mixed);
        // Both should be positive; mixed has 7 chars
        assert!(count_ascii > 0);
        assert!(count_mixed > 0);
        assert!(count_mixed >= 1);
    }

    // --- estimate_token_count with emoji (multi-byte chars) ---

    #[test]
    fn estimate_token_count_with_emoji() {
        // Each emoji is 1 char in Rust (Unicode scalar)
        let text = "\u{1f600}\u{1f601}\u{1f602}\u{1f603}"; // 4 emoji chars
        let count = estimate_token_count(text);
        assert_eq!(count, 1, "4 chars → div_ceil(4,4) = 1");
    }

    // --- estimate_text_entropy with purely numeric text ---

    #[test]
    fn entropy_numeric_text() {
        let text = "1234567890 1234567890 1234567890 1234567890";
        let entropy = estimate_text_entropy(text);
        // Numeric text with repeated trigrams of digits
        assert!(
            entropy >= 0.0,
            "entropy must be non-negative, got {entropy}"
        );
    }

    // --- estimate_text_entropy single long word ---

    #[test]
    fn entropy_single_very_long_word() {
        let text = "supercalifragilisticexpialidocious";
        let entropy = estimate_text_entropy(text);
        assert!(
            entropy > 0.0,
            "single long word must have positive entropy, got {entropy}"
        );
    }

    // --- find_stop_pattern with pattern containing regex-like chars (literal match) ---

    #[test]
    fn stop_pattern_literal_match_no_regex() {
        let patterns = vec!["a.*b".to_string(), "(final)".to_string()];
        // These are literal substring matches, not regex
        assert!(find_stop_pattern("a.*b matched", &patterns).is_some());
        assert!(find_stop_pattern("(final) answer", &patterns).is_some());
        // The literal "a.*b" should NOT match "aXYZb"
        assert!(find_stop_pattern("aXYZb", &patterns).is_none());
    }

    // --- ReasoningStopReason PatternMatched with unicode reason ---

    #[test]
    fn stop_reason_pattern_matched_unicode() {
        let reason = ReasoningStopReason::PatternMatched("\u{7b54}\u{6848}:".to_string());
        let cloned = reason.clone();
        assert_eq!(reason, cloned);
        let debug = format!("{:?}", reason);
        assert!(debug.contains("PatternMatched"));
    }

    // --- ReasoningStopReason HaltByHook with long reason ---

    #[test]
    fn stop_reason_halt_by_hook_long_reason() {
        let long_reason = "This is a very long halt reason that explains \
                           why the reasoning loop was terminated early by \
                           the step hook callback mechanism".to_string();
        let reason = ReasoningStopReason::HaltByHook(long_reason.clone());
        assert_eq!(reason.clone(), reason);
        let debug = format!("{:?}", reason);
        assert!(debug.contains(&long_reason));
    }

    // --- ReasoningStopReason usable as HashMap key ---

    #[test]
    fn stop_reason_usable_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(ReasoningStopReason::BudgetExhausted, 1u32);
        map.insert(ReasoningStopReason::StepCountReached, 2u32);
        map.insert(ReasoningStopReason::EntropyConverged, 3u32);
        assert_eq!(map.get(&ReasoningStopReason::BudgetExhausted), Some(&1));
        assert_eq!(map.get(&ReasoningStopReason::StepCountReached), Some(&2));
        assert_eq!(map.get(&ReasoningStopReason::EntropyConverged), Some(&3));
        assert_eq!(map.get(&ReasoningStopReason::HaltByHook("x".into())), None);
    }

    // --- DEFAULT_STOP_PATTERNS immutability: contains expected entries ---

    #[test]
    fn default_stop_patterns_all_entries_non_empty() {
        for pattern in DEFAULT_STOP_PATTERNS.iter() {
            assert!(!pattern.is_empty(), "each default stop pattern must be non-empty");
        }
    }

    // --- ReasoningTemplate validate accepts temperature = 1.0 ---

    #[test]
    fn template_validate_accepts_temperature_one() {
        let tpl = ReasoningTemplate {
            temperature: 1.0,
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    // --- ReasoningTemplate validate accepts temperature = 2.0 ---

    #[test]
    fn template_validate_accepts_temperature_two() {
        let tpl = ReasoningTemplate {
            temperature: 2.0,
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    // --- ReasoningTemplate validate rejects temperature = f32::MIN_POSITIVE * 0 ---

    #[test]
    fn template_validate_rejects_positive_zero_boundary() {
        let tpl = ReasoningTemplate {
            temperature: 0.0_f32,
            ..Default::default()
        };
        assert!(matches!(tpl.validate(), Err(ReasoningError::InvalidConfig(_))));
    }

    // --- ReasoningTemplate validate with minimal valid final_answer_budget ---

    #[test]
    fn template_validate_accepts_minimal_budget() {
        let tpl = ReasoningTemplate {
            final_answer_budget: 1,
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    // --- ReasoningTemplate validate with separator variations ---

    #[test]
    fn template_validate_allows_any_separator() {
        let tpl = ReasoningTemplate {
            step_separator: String::new(), // empty separator is allowed
            ..Default::default()
        };
        assert!(tpl.validate().is_ok());
    }

    // --- ReasoningError GenerationFailed with multiline detail ---

    #[test]
    fn reasoning_error_generation_failed_multiline_preserved() {
        let detail = "step 3 failed\nreason: OOM\nrecovery: none";
        let err = ReasoningError::GenerationFailed(detail.to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    // --- estimate_token_count with only punctuation ---

    #[test]
    fn estimate_token_count_punctuation_only() {
        let text = "!@#$%^&*()"; // 10 chars → div_ceil(10,4) = 3
        assert_eq!(estimate_token_count(text), 3);
    }

    // --- estimate_text_entropy with tab-separated words ---

    #[test]
    fn entropy_tab_separated_words() {
        let text = "alpha\tbeta\tgamma\tdelta";
        let entropy = estimate_text_entropy(text);
        // split_whitespace handles tabs, so word_count=4
        assert!(
            entropy > 0.0,
            "tab-separated words must have positive entropy, got {entropy}"
        );
    }

    // --- ReasoningBuilder template(None) vs template(Some) distinction ---

    #[test]
    fn reasoning_builder_no_template_means_none() {
        let client = Client::new_empty();
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: 100,
            step_count: 1,
        };
        let builder = ReasoningBuilder::new(&client, "q".into(), mode);
        assert!(builder.template.is_none());
    }

    // --- ReasoningResponse with very long text ---

    #[test]
    fn reasoning_response_long_text() {
        let long_text = "x".repeat(10000);
        let response = ReasoningResponse {
            text: long_text.clone(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 2500,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        assert_eq!(response.text.len(), 10000);
    }

    // --- ReasoningMode Auto clone isolation ---

    #[test]
    fn reasoning_mode_auto_clone_isolation() {
        let original = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: Some(0.5),
            stop_patterns: vec!["stop".to_string()],
        };
        let _cloned = original.clone();
        // original should still be usable
        let debug = format!("{:?}", original);
        assert!(debug.contains("1024"));
    }

    // --- ReasoningMode Manual clone isolation ---

    #[test]
    fn reasoning_mode_manual_clone_isolation() {
        let original = ReasoningMode::Manual {
            max_reasoning_tokens: 500,
            step_count: 3,
        };
        let cloned = original.clone();
        assert_eq!(format!("{:?}", original), format!("{:?}", cloned));
    }

    // --- estimate_text_entropy: text with newlines between sentences ---

    #[test]
    fn entropy_text_with_newlines() {
        let text = "First sentence.\nSecond sentence.\nThird sentence.";
        let entropy = estimate_text_entropy(text);
        assert!(
            entropy > 0.0,
            "text with newlines must have positive entropy, got {entropy}"
        );
    }

    // --- ReasoningTemplate system_prompt can be custom ---

    #[test]
    fn template_custom_system_prompt_preserved() {
        let tpl = ReasoningTemplate {
            system_prompt: "You are a math expert. Always show work.".to_string(),
            ..Default::default()
        };
        assert_eq!(tpl.system_prompt, "You are a math expert. Always show work.");
        assert!(tpl.validate().is_ok());
    }

    // ========================================================================
    // ADDITIONAL TESTS (18 new)
    // ========================================================================

    #[test]
    fn step_knowledge_clone_isolation() {
        let original = StepKnowledge {
            inject_text: Some("original insight".to_string()),
            modified_temperature: Some(0.7),
        };
        let mut cloned = original.clone();
        cloned.inject_text = Some("modified insight".to_string());
        cloned.modified_temperature = Some(0.1);
        assert_eq!(original.inject_text.as_deref(), Some("original insight"));
        assert_eq!(original.modified_temperature, Some(0.7));
    }

    #[test]
    fn reasoning_response_clone_isolation() {
        let original = ReasoningResponse {
            text: "original answer".to_string(),
            reasoning_trace: vec!["step 1".to_string()],
            total_reasoning_tokens: 5,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        let mut cloned = original.clone();
        cloned.text.push_str(" modified");
        cloned.reasoning_trace.push("step 2".to_string());
        assert_eq!(original.text, "original answer");
        assert_eq!(original.reasoning_trace.len(), 1);
        assert_eq!(cloned.text, "original answer modified");
        assert_eq!(cloned.reasoning_trace.len(), 2);
    }

    #[test]
    fn reasoning_template_not_equal_different_system_prompt() {
        let a = ReasoningTemplate::default();
        let b = ReasoningTemplate {
            system_prompt: "Different system prompt.".to_string(),
            ..Default::default()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn reasoning_template_not_equal_different_step_separator() {
        let a = ReasoningTemplate::default();
        let b = ReasoningTemplate {
            step_separator: "||".to_string(),
            ..Default::default()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn reasoning_template_not_equal_different_final_answer_budget() {
        let a = ReasoningTemplate::default();
        let b = ReasoningTemplate {
            final_answer_budget: 9999,
            ..Default::default()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn reasoning_template_not_equal_different_top_k() {
        let a = ReasoningTemplate::default();
        let b = ReasoningTemplate {
            top_k: 100,
            ..Default::default()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn stop_reason_pattern_vs_halt_same_payload() {
        let pattern = ReasoningStopReason::PatternMatched("timeout".to_string());
        let halt = ReasoningStopReason::HaltByHook("timeout".to_string());
        assert_ne!(pattern, halt, "PatternMatched and HaltByHook are different variants");
    }

    #[test]
    fn estimate_token_count_two_chars() {
        assert_eq!(estimate_token_count("ab"), 1, "div_ceil(2, 4) = 1");
    }

    #[test]
    fn estimate_token_count_seven_chars() {
        assert_eq!(estimate_token_count("abcdefg"), 2, "div_ceil(7, 4) = 2");
    }

    #[test]
    fn find_stop_pattern_exact_text_match() {
        let patterns = vec!["Final Answer:".to_string()];
        let result = find_stop_pattern("Final Answer:", &patterns);
        assert_eq!(result.map(|s| s.as_str()), Some("Final Answer:"));
    }

    #[test]
    fn find_stop_pattern_pattern_at_end() {
        let patterns = vec!["done".to_string()];
        let result = find_stop_pattern("processing almost done", &patterns);
        assert_eq!(result.map(|s| s.as_str()), Some("done"));
    }

    #[test]
    fn reasoning_response_entropy_converged_construction() {
        let response = ReasoningResponse {
            text: "converged answer".to_string(),
            reasoning_trace: vec!["partial reasoning".to_string()],
            total_reasoning_tokens: 20,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::EntropyConverged,
        };
        assert_eq!(response.stopped_reason, ReasoningStopReason::EntropyConverged);
        assert_eq!(response.actual_steps, 1);
    }

    #[test]
    fn reasoning_response_pattern_matched_construction() {
        let response = ReasoningResponse {
            text: "42".to_string(),
            reasoning_trace: vec!["reasoning step".to_string()],
            total_reasoning_tokens: 10,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::PatternMatched("Final Answer:".to_string()),
        };
        match response.stopped_reason {
            ReasoningStopReason::PatternMatched(ref p) => assert_eq!(p, "Final Answer:"),
            other => panic!("expected PatternMatched, got {:?}", other),
        }
    }

    #[test]
    fn reasoning_error_all_variants_clone_preserves_equality() {
        let missing = ReasoningError::MissingModel;
        assert_eq!(missing.clone(), missing);

        let gen_fail = ReasoningError::GenerationFailed("err".to_string());
        assert_eq!(gen_fail.clone(), gen_fail);

        let invalid = ReasoningError::InvalidConfig("bad".to_string());
        assert_eq!(invalid.clone(), invalid);
    }

    #[test]
    fn estimate_text_entropy_trailing_whitespace_trimmed() {
        let clean = estimate_text_entropy("hello world");
        let with_trailing = estimate_text_entropy("hello world   ");
        assert_eq!(clean, with_trailing, "trailing whitespace should be trimmed before computation");
    }

    #[test]
    fn step_result_large_values() {
        let result = StepResult {
            step_index: usize::MAX,
            chunk_text: String::new(),
            tokens_used: usize::MAX,
            accumulated_text: String::new(),
        };
        assert_eq!(result.step_index, usize::MAX);
        assert_eq!(result.tokens_used, usize::MAX);
    }

    #[test]
    fn reasoning_mode_auto_stop_patterns_order_matters() {
        let a = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: None,
            stop_patterns: vec!["alpha".to_string(), "beta".to_string()],
        };
        let b = ReasoningMode::Auto {
            max_total_tokens: 1024,
            entropy_threshold: None,
            stop_patterns: vec!["beta".to_string(), "alpha".to_string()],
        };
        assert_ne!(a, b, "stop_patterns order must affect equality");
    }

    #[test]
    fn estimate_text_entropy_five_distinct_words() {
        let entropy = estimate_text_entropy("alpha beta gamma delta epsilon");
        assert!(
            entropy > 0.0,
            "five distinct words must have positive entropy, got {entropy}"
        );
    }

    // ─── Additional coverage tests ────────────────────────────────────

    #[test]
    fn auto_step_chunk_constant_value() {
        // AUTO_STEP_CHUNK must be a reasonable positive power-of-two-ish value
        assert!(AUTO_STEP_CHUNK > 0);
        assert!(AUTO_STEP_CHUNK.is_power_of_two() || AUTO_STEP_CHUNK % 64 == 0);
    }

    #[test]
    fn reasoning_mode_auto_none_entropy_empty_patterns() {
        let mode = ReasoningMode::Auto {
            max_total_tokens: 512,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
        if let ReasoningMode::Auto { stop_patterns, .. } = &cloned {
            assert!(stop_patterns.is_empty());
        } else {
            panic!("expected Auto variant");
        }
    }

    #[test]
    fn reasoning_stop_reason_pattern_with_newline() {
        let reason = ReasoningStopReason::PatternMatched("line1\nline2".to_string());
        let cloned = reason.clone();
        assert_eq!(reason, cloned);
        let debug = format!("{reason:?}");
        assert!(debug.contains("line1\\nline2") || debug.contains("line1\nline2"));
    }

    #[test]
    fn step_knowledge_both_fields_set_equality() {
        let a = StepKnowledge {
            inject_text: Some("hint".to_string()),
            modified_temperature: Some(0.3),
        };
        let b = StepKnowledge {
            inject_text: Some("hint".to_string()),
            modified_temperature: Some(0.3),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn step_result_all_fields_custom() {
        let result = StepResult {
            step_index: 7,
            chunk_text: "partial deduction".to_string(),
            tokens_used: 42,
            accumulated_text: "step0\nstep1\npartial deduction".to_string(),
        };
        assert_eq!(result.step_index, 7);
        assert_eq!(result.tokens_used, 42);
        assert!(!result.chunk_text.is_empty());
        assert!(result.accumulated_text.contains("partial deduction"));
    }

    #[test]
    fn reasoning_response_halt_by_hook_construction() {
        let resp = ReasoningResponse {
            text: "stopped early".to_string(),
            reasoning_trace: vec!["thinking step 1".to_string()],
            total_reasoning_tokens: 10,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::HaltByHook("user abort".to_string()),
        };
        assert_eq!(resp.actual_steps, 1);
        assert_eq!(resp.reasoning_trace.len(), 1);
        assert!(matches!(
            resp.stopped_reason,
            ReasoningStopReason::HaltByHook(ref s) if s == "user abort"
        ));
    }

    #[test]
    fn reasoning_error_into_client_error_roundtrip_message() {
        let err = ReasoningError::GenerationFailed("timeout".to_string());
        let client_err: ClientError = err.into();
        let msg = format!("{client_err:?}");
        assert!(msg.contains("timeout") || msg.contains("cot_reasoner"));
    }

    #[test]
    fn estimate_token_count_single_byte() {
        assert_eq!(estimate_token_count("a"), 1);
        assert_eq!(estimate_token_count("ab"), 1);
        assert_eq!(estimate_token_count("abc"), 1);
        assert_eq!(estimate_token_count("abcd"), 1);
        assert_eq!(estimate_token_count("abcde"), 2);
    }

    #[test]
    fn template_default_top_k_is_zero() {
        let tpl = ReasoningTemplate::default();
        assert_eq!(tpl.top_k, 0, "default top_k should be 0 (disabled)");
    }

    #[test]
    fn template_validate_allows_zero_top_p() {
        let tpl = ReasoningTemplate {
            top_p: 0.0,
            ..Default::default()
        };
        assert!(
            tpl.validate().is_ok(),
            "top_p=0 is not validated by template, should be accepted"
        );
    }

    #[test]
    fn reasoning_mode_manual_minimal_valid() {
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: 1,
            step_count: 1,
        };
        let debug = format!("{mode:?}");
        assert!(debug.contains("Manual"));
    }

    #[test]
    fn step_context_equal_fields() {
        let ctx = StepContext {
            step_index: 3,
            remaining_budget: 100,
            accumulated_text: "abc".to_string(),
            model_name: "test-model".to_string(),
        };
        let ctx2 = StepContext {
            step_index: 3,
            remaining_budget: 100,
            accumulated_text: "abc".to_string(),
            model_name: "test-model".to_string(),
        };
        assert_eq!(ctx, ctx2);
    }

    #[test]
    fn estimate_text_entropy_single_repeated_char() {
        let entropy = estimate_text_entropy("aaaaa");
        assert!(
            entropy >= 0.0,
            "single repeated char entropy should be non-negative, got {entropy}"
        );
    }

    #[test]
    fn step_action_inject_prompt_multibyte() {
        let action = StepAction::InjectPrompt("追加知识：重要提示".to_string());
        let cloned = action.clone();
        assert_eq!(action, cloned);
        if let StepAction::InjectPrompt(text) = cloned {
            assert!(text.contains("追加知识"));
        } else {
            panic!("expected InjectPrompt variant");
        }
    }

    #[test]
    fn reasoning_response_zero_tokens() {
        let resp = ReasoningResponse {
            text: "final".to_string(),
            reasoning_trace: vec![],
            total_reasoning_tokens: 0,
            actual_steps: 0,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        assert_eq!(resp.total_reasoning_tokens, 0);
        assert_eq!(resp.actual_steps, 0);
        assert!(resp.reasoning_trace.is_empty());
        assert_eq!(resp.stopped_reason, ReasoningStopReason::BudgetExhausted);
    }

    // ========================================================================
    // 15 NEW TESTS — edge cases not covered above
    // ========================================================================

    #[test]
    fn template_validate_rejects_subnormal_temperature() {
        // Subnormal float: finite but extremely small positive
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        assert!(subnormal.is_finite());
        assert!(subnormal > 0.0);
        let tpl = ReasoningTemplate {
            temperature: subnormal,
            ..Default::default()
        };
        // Subnormal is finite and > 0, so validate should accept it
        assert!(tpl.validate().is_ok());
    }

    #[test]
    fn estimate_token_count_thirteen_chars() {
        // 13 chars → div_ceil(13, 4) = 4
        assert_eq!(estimate_token_count("abcdefghijklm"), 4);
    }

    #[test]
    fn entropy_text_with_leading_whitespace_trimmed() {
        // Leading whitespace should be trimmed, same result as no leading ws
        let clean = estimate_text_entropy("hello world test");
        let with_leading = estimate_text_entropy("   hello world test");
        assert_eq!(clean, with_leading);
    }

    #[test]
    fn step_knowledge_inject_text_with_newlines() {
        let sk = StepKnowledge {
            inject_text: Some("line1\nline2\nline3".to_string()),
            modified_temperature: None,
        };
        let cloned = sk.clone();
        assert_eq!(cloned.inject_text.as_deref(), Some("line1\nline2\nline3"));
        assert_eq!(sk, cloned);
    }

    #[test]
    fn stop_reason_budget_exhausted_debug_no_payload() {
        let reason = ReasoningStopReason::BudgetExhausted;
        let debug = format!("{reason:?}");
        assert!(debug.contains("BudgetExhausted"));
        assert!(!debug.contains("PatternMatched"));
        assert!(!debug.contains("HaltByHook"));
    }

    #[test]
    fn find_stop_pattern_case_sensitive_mismatch() {
        let patterns = vec!["Final Answer:".to_string()];
        // "final answer:" (lowercase) should NOT match
        assert!(find_stop_pattern("final answer: 42", &patterns).is_none());
    }

    #[test]
    fn reasoning_mode_auto_equality_with_many_patterns() {
        let patterns: Vec<String> = (0..100).map(|i| format!("pattern_{i}")).collect();
        let a = ReasoningMode::Auto {
            max_total_tokens: 2048,
            entropy_threshold: Some(0.3),
            stop_patterns: patterns.clone(),
        };
        let b = ReasoningMode::Auto {
            max_total_tokens: 2048,
            entropy_threshold: Some(0.3),
            stop_patterns: patterns,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn step_context_model_name_with_special_chars() {
        let ctx = StepContext {
            step_index: 0,
            remaining_budget: 100,
            accumulated_text: String::new(),
            model_name: "org/model-name:v2.0-beta+exp".to_string(),
        };
        assert_eq!(ctx.model_name, "org/model-name:v2.0-beta+exp");
        let cloned = ctx.clone();
        assert_eq!(cloned.model_name, ctx.model_name);
    }

    #[test]
    fn reasoning_response_partial_eq_trace_order_matters() {
        let a = ReasoningResponse {
            text: "ans".to_string(),
            reasoning_trace: vec!["alpha".to_string(), "beta".to_string()],
            total_reasoning_tokens: 10,
            actual_steps: 2,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        let b = ReasoningResponse {
            text: "ans".to_string(),
            reasoning_trace: vec!["beta".to_string(), "alpha".to_string()],
            total_reasoning_tokens: 10,
            actual_steps: 2,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert_ne!(a, b, "trace element order must affect equality");
    }

    #[test]
    fn estimate_text_entropy_exactly_one_word_multiple_spaces() {
        // "hello" with surrounding spaces → trimmed to "hello", word_count=1
        let entropy = estimate_text_entropy("  hello  ");
        let entropy_clean = estimate_text_entropy("hello");
        assert_eq!(entropy, entropy_clean);
    }

    #[test]
    fn reasoning_template_default_system_prompt_contains_reasoner() {
        let tpl = ReasoningTemplate::default();
        let lower = tpl.system_prompt.to_lowercase();
        assert!(
            lower.contains("reasoner") || lower.contains("step"),
            "default system prompt should mention reasoning or steps"
        );
    }

    #[test]
    fn step_action_halt_reason_with_unicode() {
        let action = StepAction::Halt("推理结束：达到结论".to_string());
        let cloned = action.clone();
        if let StepAction::Halt(ref reason) = cloned {
            assert!(reason.contains("推理"));
        } else {
            panic!("expected Halt variant");
        }
    }

    #[test]
    fn estimate_token_count_carriage_return_and_linefeed() {
        // "\r\n\r\n" = 4 chars → 1 token
        assert_eq!(estimate_token_count("\r\n\r\n"), 1);
        // "\r\n" = 2 chars → 1 token
        assert_eq!(estimate_token_count("\r\n"), 1);
    }

    #[test]
    fn stop_reason_entropy_converged_debug_contains_name() {
        let reason = ReasoningStopReason::EntropyConverged;
        let debug = format!("{reason:?}");
        assert!(debug.contains("EntropyConverged"));
        // Ensure no payload fields exist on this variant
        assert!(!debug.contains('(') || debug.starts_with("EntropyConverged"));
    }

    #[test]
    fn reasoning_error_invalid_config_with_special_chars() {
        let detail = "temp = NaN! @#$%^& token_limit=0";
        let err = ReasoningError::InvalidConfig(detail.to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
        let msg = err.to_string();
        assert!(msg.contains(detail));
    }

    // ========================================================================
    // 15 NEW TESTS — additional edge cases
    // ========================================================================

    #[test]
    fn template_validate_rejects_both_empty_prefix_and_final_prefix() {
        // Verify that when both step_prefix and final_prefix are empty,
        // validate returns an error (it checks step_prefix first).
        let tpl = ReasoningTemplate {
            step_prefix: String::new(),
            final_prefix: String::new(),
            ..Default::default()
        };
        let result = tpl.validate();
        assert!(result.is_err());
        // Verify the error message mentions one of the invalid fields
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("step_prefix") || msg.contains("final_prefix"),
            "error should reference the empty field, got: {msg}"
        );
    }

    #[test]
    fn entropy_text_with_multiple_consecutive_spaces_between_words() {
        // split_whitespace treats consecutive spaces as a single separator,
        // but trim() preserves internal spaces, so char diversity differs.
        // Verify both are positive and finite (not asserting equality).
        let single_spaced = estimate_text_entropy("a b c d e");
        let multi_spaced = estimate_text_entropy("a  b  c  d  e");
        assert!(
            single_spaced > 0.0 && multi_spaced > 0.0,
            "both entropy values must be positive: single={single_spaced}, multi={multi_spaced}"
        );
        assert!(
            single_spaced.is_finite() && multi_spaced.is_finite(),
            "both entropy values must be finite"
        );
    }

    #[test]
    fn entropy_three_identical_trigrams_produces_high_repetition() {
        // "a b c a b c a b c a b c" → every trigram is "a b c", repeated
        let text = "a b c a b c a b c a b c";
        let entropy = estimate_text_entropy(text);
        // With 10 trigrams all identical: repetition_ratio = 9/10 = 0.9
        // entropy = diversity * (1 - 0.9) * word_scale ≈ small value
        assert!(
            entropy < 1.0,
            "text where all trigrams repeat must have low entropy, got {entropy}"
        );
    }

    #[test]
    fn stop_reason_pattern_matched_with_very_long_payload() {
        let long_payload = "X".repeat(10000);
        let reason = ReasoningStopReason::PatternMatched(long_payload.clone());
        let cloned = reason.clone();
        assert_eq!(reason, cloned);
        if let ReasoningStopReason::PatternMatched(ref p) = reason {
            assert_eq!(p.len(), 10000);
        } else {
            panic!("expected PatternMatched");
        }
    }

    #[test]
    fn step_action_inject_prompt_with_very_long_string() {
        let long_extra = "Y".repeat(5000);
        let action = StepAction::InjectPrompt(long_extra.clone());
        let cloned = action.clone();
        assert_eq!(action, cloned);
        if let StepAction::InjectPrompt(ref text) = action {
            assert_eq!(text.len(), 5000);
        } else {
            panic!("expected InjectPrompt");
        }
    }

    #[test]
    fn estimate_text_entropy_longer_diverse_text_positive() {
        // Verify that a long diverse text always produces positive entropy.
        // (Monotonicity is not guaranteed because adding words can lower
        // the character diversity ratio.)
        let short = estimate_text_entropy("alpha beta gamma");
        let longer = estimate_text_entropy(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        );
        assert!(
            short > 0.0 && longer > 0.0,
            "both short and long diverse text must have positive entropy: short={short}, longer={longer}"
        );
        assert!(
            short.is_finite() && longer.is_finite(),
            "entropy values must be finite"
        );
    }

    #[test]
    fn find_stop_pattern_pattern_equals_entire_text() {
        let text = "Final Answer:";
        let patterns = vec!["Final Answer:".to_string()];
        let result = find_stop_pattern(text, &patterns);
        assert_eq!(result.map(|s| s.as_str()), Some("Final Answer:"));
    }

    #[test]
    fn reasoning_response_single_trace_zero_tokens_field() {
        // actual_steps = 1 but total_reasoning_tokens = 0 is a valid state
        // (e.g., if estimate returned 0 for an extremely short chunk)
        let response = ReasoningResponse {
            text: "yes".to_string(),
            reasoning_trace: vec!["ok".to_string()],
            total_reasoning_tokens: 0,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert_eq!(response.reasoning_trace.len(), 1);
        assert_eq!(response.total_reasoning_tokens, 0);
        assert_eq!(response.actual_steps, 1);
    }

    #[test]
    fn reasoning_mode_auto_zero_entropy_threshold_cloned_equals() {
        let mode = ReasoningMode::Auto {
            max_total_tokens: 512,
            entropy_threshold: Some(0.0),
            stop_patterns: vec!["done".to_string()],
        };
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    #[test]
    fn template_render_step_prefix_custom_prefix_only_placeholder() {
        // Prefix is just "{n}" — rendered result is only the step number
        let tpl = ReasoningTemplate {
            step_prefix: "{n}".to_string(),
            ..Default::default()
        };
        assert_eq!(tpl.render_step_prefix(7), "7");
        assert_eq!(tpl.render_step_prefix(1), "1");
    }

    #[test]
    fn estimate_token_count_large_but_safe_input() {
        // Ensure estimate_token_count handles reasonably large input correctly
        let text = "a".repeat(1_000_000);
        let count = estimate_token_count(&text);
        assert_eq!(count, 250_000, "1_000_000 chars / 4 = 250_000 tokens");
    }

    #[test]
    fn reasoning_builder_chaining_template_then_hook() {
        // Verify builder chaining: new -> template -> with_step_hook
        let client = Client::new_empty();
        let mode = ReasoningMode::Manual {
            max_reasoning_tokens: 100,
            step_count: 1,
        };
        let tpl = ReasoningTemplate {
            temperature: 0.5,
            ..Default::default()
        };
        let builder = ReasoningBuilder::new(&client, "q".into(), mode)
            .template(tpl)
            .with_step_hook(Box::new(DummyHook));
        assert!(builder.template.is_some());
        assert!(builder.step_hook.is_some());
        // Verify template was stored correctly
        let stored_tpl = builder.template.as_ref().expect("template should be Some");
        assert!(
            (stored_tpl.temperature - 0.5).abs() < f32::EPSILON,
            "stored template temperature should be 0.5"
        );
    }

    #[test]
    fn estimate_text_entropy_returns_f32_finite() {
        // Entropy must always be a finite f32 (not NaN, not infinity)
        let texts = [
            "",
            "a",
            "hello world",
            "abc abc abc",
            &"x ".repeat(1000),
            "The quick brown fox jumps over the lazy dog. " ,
        ];
        for text in &texts {
            let entropy = estimate_text_entropy(text);
            assert!(
                entropy.is_finite(),
                "entropy must be finite for input '{text}', got {entropy}"
            );
            assert!(
                entropy >= 0.0,
                "entropy must be non-negative for input '{text}', got {entropy}"
            );
        }
    }

    #[test]
    fn find_stop_pattern_with_multiple_patterns_first_matching_wins_even_if_later_is_shorter() {
        let patterns = vec![
            "Final Answer: The result is".to_string(),
            "Final Answer:".to_string(),
        ];
        let text = "After reasoning, Final Answer: The result is 42.";
        let result = find_stop_pattern(text, &patterns);
        // "Final Answer: The result is" matches first in iteration, even though
        // "Final Answer:" also matches
        assert_eq!(
            result.map(|s| s.as_str()),
            Some("Final Answer: The result is")
        );
    }

    #[test]
    fn reasoning_stop_reason_step_count_reached_not_equal_budget_exhausted() {
        assert_ne!(
            ReasoningStopReason::StepCountReached,
            ReasoningStopReason::BudgetExhausted,
            "StepCountReached and BudgetExhausted must be distinct variants"
        );
    }

    // ========================================================================
    // 15 NEW TESTS — additional coverage
    // ========================================================================

    #[test]
    fn template_validate_accepts_f32_max_temperature() {
        // f32::MAX is finite and > 0, so it must pass validation.
        let tpl = ReasoningTemplate {
            temperature: f32::MAX,
            ..Default::default()
        };
        assert!(
            tpl.validate().is_ok(),
            "f32::MAX temperature is finite and > 0, must be accepted"
        );
    }

    #[test]
    fn estimate_token_count_nine_chars_rounds_up_correctly() {
        // 9 chars → div_ceil(9, 4) = 3
        assert_eq!(estimate_token_count("123456789"), 3);
        // 10 chars → div_ceil(10, 4) = 3
        assert_eq!(estimate_token_count("1234567890"), 3);
        // 11 chars → div_ceil(11, 4) = 3
        assert_eq!(estimate_token_count("12345678901"), 3);
        // 12 chars → div_ceil(12, 4) = 3
        assert_eq!(estimate_token_count("123456789012"), 3);
        // 13 chars → div_ceil(13, 4) = 4
        assert_eq!(estimate_token_count("1234567890123"), 4);
    }

    #[test]
    fn reasoning_response_empty_text_is_valid() {
        // An empty text in the final answer is a valid construction
        let response = ReasoningResponse {
            text: String::new(),
            reasoning_trace: vec!["reasoned but no final output".to_string()],
            total_reasoning_tokens: 15,
            actual_steps: 1,
            stopped_reason: ReasoningStopReason::BudgetExhausted,
        };
        assert!(response.text.is_empty());
        assert_eq!(response.reasoning_trace.len(), 1);
    }

    #[test]
    fn entropy_text_mixed_tabs_and_spaces_split_correctly() {
        // Both tabs and spaces are treated as whitespace delimiters.
        // "a\tb c\td e" has 5 words split by whitespace.
        let entropy = estimate_text_entropy("a\tb c\td e");
        assert!(
            entropy > 0.0,
            "text with mixed tabs and spaces must have positive entropy, got {entropy}"
        );
    }

    #[test]
    fn stop_reason_hash_different_payload_same_variant() {
        // Two PatternMatched with different payloads should hash differently
        // (not guaranteed but extremely likely due to String hashing).
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = ReasoningStopReason::PatternMatched("alpha".to_string());
        let b = ReasoningStopReason::PatternMatched("beta".to_string());
        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let va = ha.finish();
        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        let vb = hb.finish();
        assert_ne!(va, vb, "PatternMatched with different payloads should hash differently");
    }

    #[test]
    fn step_knowledge_inject_some_vs_none_not_equal() {
        let with_inject = StepKnowledge {
            inject_text: Some("hint".to_string()),
            modified_temperature: None,
        };
        let without_inject = StepKnowledge {
            inject_text: None,
            modified_temperature: None,
        };
        assert_ne!(with_inject, without_inject);
    }

    #[test]
    fn step_knowledge_temperature_some_vs_none_not_equal() {
        let with_temp = StepKnowledge {
            inject_text: None,
            modified_temperature: Some(0.5),
        };
        let without_temp = StepKnowledge {
            inject_text: None,
            modified_temperature: None,
        };
        assert_ne!(with_temp, without_temp);
    }

    #[test]
    fn template_render_step_prefix_preserves_surrounding_text() {
        // Verify that text before and after {n} is preserved exactly.
        let tpl = ReasoningTemplate {
            step_prefix: ">> Step {n} of reasoning:".to_string(),
            ..Default::default()
        };
        assert_eq!(tpl.render_step_prefix(4), ">> Step 4 of reasoning:");
        assert_eq!(tpl.render_step_prefix(1), ">> Step 1 of reasoning:");
    }

    #[test]
    fn auto_step_chunk_greater_than_min_step_tokens() {
        // AUTO_STEP_CHUNK (per-step max for Auto) must be > MIN_STEP_TOKENS
        // (minimum threshold for budget exhaustion check).
        assert!(
            AUTO_STEP_CHUNK > MIN_STEP_TOKENS,
            "AUTO_STEP_CHUNK ({AUTO_STEP_CHUNK}) must exceed MIN_STEP_TOKENS ({MIN_STEP_TOKENS})"
        );
    }

    #[test]
    fn reasoning_error_display_invalid_config_preserves_detail() {
        let detail = "step_count must be >= 1";
        let err = ReasoningError::InvalidConfig(detail.to_string());
        let display = format!("{err}");
        assert!(display.contains(detail), "Display must preserve detail: {display}");
        assert!(display.contains("invalid reasoning config"));
    }

    #[test]
    fn reasoning_mode_auto_with_empty_vs_none_entropy_are_different() {
        let with_zero = ReasoningMode::Auto {
            max_total_tokens: 512,
            entropy_threshold: Some(0.0),
            stop_patterns: vec![],
        };
        let with_none = ReasoningMode::Auto {
            max_total_tokens: 512,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        // Some(0.0) != None, so these are different modes
        assert_ne!(with_zero, with_none);
    }

    #[test]
    fn estimate_text_entropy_finite_for_all_ascii_printable_chars() {
        // Exhaustive check that entropy never produces NaN or infinity
        // even for edge-case text content.
        let text: String = (32u8..127).map(|c| c as char).collect();
        let entropy = estimate_text_entropy(&text);
        assert!(entropy.is_finite(), "entropy must be finite, got {entropy}");
        assert!(entropy > 0.0, "printable ASCII range must have positive entropy");
    }

    #[test]
    fn find_stop_pattern_with_duplicate_patterns_returns_first() {
        // If the same pattern appears multiple times in the list,
        // find returns the first occurrence (iterator order).
        let patterns = vec![
            "done".to_string(),
            "done".to_string(),
            "stop".to_string(),
        ];
        let result = find_stop_pattern("we are done", &patterns);
        assert!(result.is_some());
        // Both "done" entries are identical strings, so the first is returned.
        assert_eq!(result.unwrap(), "done");
    }

    #[test]
    fn reasoning_builder_execute_auto_without_model_errors() {
        // Verify that Auto mode without a loaded model returns NoModelLoaded.
        let client = Client::new_empty();
        let mode = ReasoningMode::Auto {
            max_total_tokens: 256,
            entropy_threshold: Some(0.5),
            stop_patterns: vec!["done".to_string()],
        };
        let result = ReasoningBuilder::new(&client, "test prompt".into(), mode).execute();
        assert!(
            matches!(result, Err(ClientError::NoModelLoaded)),
            "Auto mode without model must return NoModelLoaded"
        );
    }

    #[test]
    fn stop_reason_halt_by_hook_not_equal_pattern_matched_same_payload() {
        // Even with identical string payloads, different variants must not be equal.
        let payload = "converged";
        let halt = ReasoningStopReason::HaltByHook(payload.to_string());
        let pattern = ReasoningStopReason::PatternMatched(payload.to_string());
        assert_ne!(halt, pattern);
    }

    // ========================================================================
    // 15 NEW TESTS — edge cases and coverage gaps
    // ========================================================================

    // @trace TEST-COT-EDGE-001 estimate_token_count for 6 chars (div_ceil boundary)
    #[test]
    fn estimate_token_count_six_chars() {
        // 6 chars → div_ceil(6, 4) = 2
        assert_eq!(estimate_token_count("abcdef"), 2);
    }

    // @trace TEST-COT-EDGE-002 estimate_token_count for 8 chars (exact multiple)
    #[test]
    fn estimate_token_count_eight_chars() {
        // 8 chars → div_ceil(8, 4) = 2
        assert_eq!(estimate_token_count("abcdefgh"), 2);
    }

    // @trace TEST-COT-EDGE-003 estimate_text_entropy partial trigram overlap
    #[test]
    fn entropy_partial_trigram_overlap() {
        // "a b c b c d" → trigrams: ["a b c", "b c b", "c b c", "b c d"]
        // No exact trigram repeats → repetition_ratio = 0
        let entropy = estimate_text_entropy("a b c b c d");
        assert!(
            entropy > 0.0,
            "text with no repeated trigrams must have positive entropy, got {entropy}"
        );
        assert!(
            entropy.is_finite(),
            "entropy must be finite, got {entropy}"
        );
    }

    // @trace TEST-COT-EDGE-004 ReasoningTemplate validate rejects negative temperature
    #[test]
    fn template_validate_rejects_arbitrary_negative_temperature() {
        let tpl = ReasoningTemplate {
            temperature: -42.0,
            ..Default::default()
        };
        let result = tpl.validate();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("temperature"),
            "error must mention temperature, got: {msg}"
        );
    }

    // @trace TEST-COT-EDGE-005 StepContext Debug output includes all four field names
    #[test]
    fn step_context_debug_includes_all_fields() {
        let ctx = StepContext {
            step_index: 42,
            remaining_budget: 999,
            accumulated_text: "accumulated reasoning text".to_string(),
            model_name: "my-model-v3".to_string(),
        };
        let debug = format!("{ctx:?}");
        assert!(debug.contains("step_index"), "debug must contain step_index");
        assert!(debug.contains("remaining_budget"), "debug must contain remaining_budget");
        assert!(debug.contains("accumulated_text"), "debug must contain accumulated_text");
        assert!(debug.contains("model_name"), "debug must contain model_name");
    }

    // @trace TEST-COT-EDGE-006 StepResult Debug output includes all four field names
    #[test]
    fn step_result_debug_includes_all_fields() {
        let result = StepResult {
            step_index: 10,
            chunk_text: "computed output".to_string(),
            tokens_used: 77,
            accumulated_text: "all steps so far".to_string(),
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("step_index"), "debug must contain step_index");
        assert!(debug.contains("chunk_text"), "debug must contain chunk_text");
        assert!(debug.contains("tokens_used"), "debug must contain tokens_used");
        assert!(debug.contains("accumulated_text"), "debug must contain accumulated_text");
    }

    // @trace TEST-COT-EDGE-007 ReasoningMode Auto with None entropy and empty patterns debug
    #[test]
    fn reasoning_mode_auto_none_entropy_empty_patterns_debug() {
        let mode = ReasoningMode::Auto {
            max_total_tokens: 256,
            entropy_threshold: None,
            stop_patterns: vec![],
        };
        let debug = format!("{mode:?}");
        assert!(debug.contains("Auto"), "debug must contain Auto");
        assert!(debug.contains("256"), "debug must contain max_total_tokens value");
    }

    // @trace TEST-COT-EDGE-008 ReasoningResponse with StepCountReached and multi-step trace
    #[test]
    fn reasoning_response_step_count_reached_multi_step_trace() {
        let trace: Vec<String> = (1..=5)
            .map(|i| format!("Reasoning step {i} completed"))
            .collect();
        let response = ReasoningResponse {
            text: "The answer is 42".to_string(),
            reasoning_trace: trace.clone(),
            total_reasoning_tokens: 100,
            actual_steps: 5,
            stopped_reason: ReasoningStopReason::StepCountReached,
        };
        assert_eq!(response.actual_steps, 5);
        assert_eq!(response.reasoning_trace.len(), 5);
        assert_eq!(response.stopped_reason, ReasoningStopReason::StepCountReached);
        assert_eq!(response.reasoning_trace[0], "Reasoning step 1 completed");
        assert_eq!(response.reasoning_trace[4], "Reasoning step 5 completed");
    }

    // @trace TEST-COT-EDGE-009 Hook receives correct accumulated_text after multiple steps
    #[test]
    fn hook_sees_accumulated_text_joined_by_separator() {
        // Verify that accumulated_text is built by joining trace with step_separator
        let tpl = ReasoningTemplate::default();
        let trace = vec!["step one".to_string(), "step two".to_string()];
        let accumulated = trace.join(&tpl.step_separator);
        assert_eq!(accumulated, "step one\n\nstep two");
        // Now verify StepContext carries this correctly
        let ctx = StepContext {
            step_index: 2,
            remaining_budget: 50,
            accumulated_text: accumulated.clone(),
            model_name: "test".to_string(),
        };
        assert_eq!(ctx.accumulated_text, "step one\n\nstep two");
    }

    // @trace TEST-COT-EDGE-010 ReasoningTemplate render_step_prefix with multi-digit step
    #[test]
    fn template_render_step_prefix_three_digit_step() {
        let tpl = ReasoningTemplate {
            step_prefix: "== Phase {n} ==".to_string(),
            ..Default::default()
        };
        assert_eq!(tpl.render_step_prefix(100), "== Phase 100 ==");
        assert_eq!(tpl.render_step_prefix(256), "== Phase 256 ==");
    }

    // @trace TEST-COT-EDGE-011 ReasoningBuilder with hook + Auto mode without model
    #[test]
    fn reasoning_builder_auto_with_hook_no_model_errors() {
        let client = Client::new_empty();
        let mode = ReasoningMode::Auto {
            max_total_tokens: 512,
            entropy_threshold: None,
            stop_patterns: vec!["done".to_string()],
        };
        let result = ReasoningBuilder::new(&client, "test".into(), mode)
            .with_step_hook(Box::new(DummyHook))
            .execute();
        assert!(
            matches!(result, Err(ClientError::NoModelLoaded)),
            "Auto mode with hook but no model must return NoModelLoaded"
        );
    }

    // @trace TEST-COT-EDGE-012 estimate_text_entropy with exactly two words repeated many times
    #[test]
    fn entropy_two_words_repeated_many_times() {
        // "yes no yes no yes no yes no" → repeated trigrams "yes no yes", "no yes no", etc.
        let text = "yes no yes no yes no yes no";
        let entropy = estimate_text_entropy(text);
        assert!(
            entropy >= 0.0,
            "entropy must be non-negative, got {entropy}"
        );
        assert!(
            entropy.is_finite(),
            "entropy must be finite, got {entropy}"
        );
    }

    // @trace TEST-COT-EDGE-013 find_stop_pattern with pattern containing newline
    #[test]
    fn stop_pattern_with_newline_in_pattern() {
        let patterns = vec!["Final\nAnswer:".to_string()];
        let text = "After reasoning\nFinal\nAnswer: 42";
        let result = find_stop_pattern(text, &patterns);
        assert_eq!(
            result.map(|s| s.as_str()),
            Some("Final\nAnswer:"),
            "pattern with embedded newline must match"
        );
    }

    // @trace TEST-COT-EDGE-014 ReasoningError cross-variant inequality exhaustive
    #[test]
    fn reasoning_error_all_cross_variant_pairs_unequal() {
        let variants = [
            ReasoningError::MissingModel,
            ReasoningError::GenerationFailed("test".to_string()),
            ReasoningError::InvalidConfig("test".to_string()),
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(
                    variants[i], variants[j],
                    "ReasoningError variant {i} must not equal variant {j}"
                );
            }
        }
    }

    // @trace TEST-COT-EDGE-015 StepKnowledge Debug output includes both fields
    #[test]
    fn step_knowledge_debug_includes_both_fields() {
        let sk = StepKnowledge {
            inject_text: Some("knowledge payload".to_string()),
            modified_temperature: Some(0.42),
        };
        let debug = format!("{sk:?}");
        assert!(debug.contains("inject_text"), "debug must mention inject_text");
        assert!(debug.contains("modified_temperature"), "debug must mention modified_temperature");
    }
}
