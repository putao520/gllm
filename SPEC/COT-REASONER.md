# CoT Reasoner SDK — 任意 LLM 原生 Chain-of-Thought 推理协议

> **📌 SSOT**: 本文档定义 gllm 的 Chain-of-Thought (CoT) Reasoner SDK 技术协议。允许**任意** generator LLM (SmolLM2 / Llama / Qwen 等) 在不依赖专用 `thinking_head` 权重的前提下,通过 Client API 获得原生多步推理能力,同时支持 Manual (用户指定步长) 与 Auto (引擎自动判定) 两种模式。

> **需求 SSOT**: `SPEC/01-REQUIREMENTS.md §16` REQ-COT-001..009
>
> **API 定义 SSOT**: `SPEC/04-API-DESIGN.md §3.11`
>
> **关联模块**: `src/cot_reasoner.rs`(SDK 实现)、`src/client.rs::Client::reason`(入口)、`src/generation.rs::GenerationBuilder::reasoning`(便捷 API)

---

## §1 动机与定位

### 1.1 问题域

现有 `GenerationBuilder::thinking_budget(n)` API 仅能在模型**自带 thinking_head** (如 qwen3-thinking) 时限制思考 token 数量。其语义局限:

1. **强耦合专用权重**:依赖模型在训练期烘焙 `<thinking>...</thinking>` token 能力,普通 generator LLM (SmolLM2-135M / Llama-3 / Mistral 等) 无法使用。
2. **无主动 orchestration**:不分步、不控制推理结构、不感知推理结束信号。
3. **无 step 概念**:用户无法指定"3 步逐步推理",引擎也不会在每步间插入分隔。
4. **无自动停止判定**:budget 耗尽即截断,无法基于输出语义信号停止。

### 1.2 核心论断

> **Chain-of-Thought 应是一个**纯客户端 orchestration 模式**,通过 prompt engineering 驱动任意基础 LLM 产出结构化多步推理,无需模型训练期专门微调**。

三个支撑事实:

- 任何能做 in-context learning 的 LLM (7B 以上成熟;1B-3B 小模型效果弱但流程可跑通) 在接到 `"Step 1:"` / `"Step 2:"` prompt prefix 时会延续该结构。
- 生成循环天然可切片: 将 `max_tokens=512` 拆为 3 个 `max_tokens≈170` 的迭代调用,每次把上一轮产物拼入下一轮 prompt,语义上等价于"分步推理"。
- 停止信号可双路: (a) 用户层规则 (pattern match "Final Answer:" 等模板) + (b) 引擎层模式 (budget 耗尽 / step count reached)。

### 1.3 定位

CoT Reasoner (以下简称 **COT**) 是 gllm 的**纯客户端 SDK 层模块**,完全复用现有 `Client::generate(prompt).max_tokens(n).generate().response()?` 管线,**不新增任何 Backend trait 方法、不触及 FusedGraphExecutor、不修改 JIT 管线**。

- **对任意 LLM 工作**: SmolLM2-135M-Instruct / Llama / Qwen / Gemma 均可。
- **对模型参数零修改**: 仅 prompt engineering + 迭代 generate,不依赖权重/架构/JIT。
- **与 thinking_budget 共存不冲突**: `thinking_budget` 控制单次 generate 内的 `<thinking>` tokens(如果模型自带);CoT Reasoner 控制**多次 generate 调用**的跨轮 orchestration。两者作用粒度正交。

### 1.4 与其他 SDK 的关系

| SDK | 作用层 | 是否依赖模型特殊权重 | 是否新增 Backend API |
|-----|--------|----------------------|----------------------|
| **CoT Reasoner** | **Client orchestration (本文档)** | **否** | **否** |
| Semantic Gatekeeper | forward 中间层 pre_node callback | 否(仅需 lm_head + embed) | 是 (q_tap) |
| Head Routing | forward 完成后读 hidden | 否 | 否 |
| thinking_budget | 模型原生 thinking_head | **是** (qwen3-thinking 等) | 否 |

CoT 与 SG 正交: 注册 SG 后调用 `reason(...)`,每轮 step 的 generate 都自然走 SG 注入路径,step 间语义受 SG 残差影响(REQ-COT-005)。

---

## §2 架构总览

### 2.1 数据流

```
┌──────────────────────────────── 用户代码 ────────────────────────────────┐
│                                                                          │
│  client.generate("What is 127 * 83?")                                    │
│    .reasoning(ReasoningMode::Manual {                                    │
│        max_reasoning_tokens: 512,                                        │
│        step_count: 3,                                                    │
│    })                                                                    │
│    .generate().response()?                                               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ 转发
┌──────────────────────────── cot_reasoner.rs ─────────────────────────────┐
│                                                                          │
│  Client::reason(prompt, mode, template):                                 │
│                                                                          │
│  1. 构造初始上下文:                                                       │
│     context = system_prompt + "\n\n" + user_prompt                       │
│     trace: Vec<String> = []                                              │
│     total_tokens = 0                                                     │
│                                                                          │
│  2. Reasoning 循环(按 mode 分派):                                        │
│                                                                          │
│     ── Manual { budget=B, step_count=K } ────────────────               │
│     for step in 1..=K:                                                   │
│         step_budget = (B - total_tokens) / (K - step + 1)                │
│         step_prompt = context + "\n" + step_prefix.replace("{n}", step)  │
│         chunk_text = client.generate(step_prompt)                        │
│                        .max_tokens(step_budget)                          │
│                        .temperature(tpl.temperature)                     │
│                        .generate().response()?.text                      │
│         trace.push(chunk_text)                                           │
│         context += step_separator + chunk_text                           │
│         total_tokens += estimate_token_count(chunk_text)                 │
│         if total_tokens >= B: stopped=BudgetExhausted; break             │
│     stopped = stopped.unwrap_or(StepCountReached)                        │
│                                                                          │
│     ── Auto { max_total_tokens=B, thresh, stop_patterns } ─────         │
│     step = 1                                                             │
│     loop:                                                                │
│         step_budget = AUTO_STEP_CHUNK.min(B - total_tokens)              │
│         step_prompt = context + "\n" + step_prefix.replace("{n}", step)  │
│         chunk_text = generate(...).response()?.text                      │
│         trace.push(chunk_text)                                           │
│         total_tokens += estimate_token_count(chunk_text)                 │
│         if chunk_text 包含任意 stop_patterns[i]:                         │
│             stopped = PatternMatched(i); break                           │
│         if entropy_converged(chunk_text, thresh):                        │
│             stopped = EntropyConverged; break                            │
│         if total_tokens >= B: stopped = BudgetExhausted; break           │
│         step += 1                                                        │
│                                                                          │
│  3. Final answer 阶段:                                                   │
│     final_prompt = context + "\n" + final_prefix                         │
│     final_text = client.generate(final_prompt)                           │
│                    .max_tokens(FINAL_ANSWER_BUDGET).generate().response()?│
│                                                                          │
│  4. 返回 ReasoningResponse { text=final_text, trace, ...}                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键不变量(铁律)

1. **零 Backend API 扩展**: 所有 reasoning 都走 `Client::execute_generation` 公共路径,禁止新增 `Backend` trait 方法或 FusedGraphExecutor 扩展。
2. **零 JIT 重编译**: 每轮 step 只是 `Client::generate(...).response()?`,走已缓存的 `FusedGraphExecutor`,不触发 `compile_model_graphs()`。
3. **NO_SILENT_FALLBACK**: `generate` 失败 → `ReasoningError::GenerationFailed(msg)`;模式配置非法 → `InvalidConfig(msg)`;Client state 缺失 → `MissingModel`。禁止静默返回空 response 或默认答案。
4. **NO_ISLAND_MODULE**: `Client::reason` 必须在 `src/generation.rs::GenerationBuilder::reasoning` → `generate()` 的真实入口被调用,E2E 测试 `test_cot_006_arbitrary_llm` 覆盖。不允许模块自包含但无人调用。Step Hook 的 `ReasoningStepHook` trait 必须 pub 导出到 `src/lib.rs`,且在 `ReasoningBuilder::with_step_hook` 真实注册 (REQ-COT-007..009)。
5. **No model training / weight modification**: COT 是纯 prompt engineering + iteration loop,对模型权重、JIT 产物、forward pass **完全只读**。

---

## §3 ReasoningMode 与 ReasoningTemplate

### 3.1 ReasoningMode enum

```rust
pub enum ReasoningMode {
    /// 固定步长: 用户指定推理预算与步数。
    ///
    /// - `max_reasoning_tokens`: 所有 reasoning step 合计最大 token 数(不含 final answer)
    /// - `step_count`: 步数 (≥ 1)。引擎会为每 step 分配约 `max_reasoning_tokens / step_count` 预算。
    Manual {
        max_reasoning_tokens: usize,
        step_count: usize,
    },

    /// 引擎自动决定步数/停止。
    ///
    /// - `max_total_tokens`: 所有 reasoning step 合计上限(不含 final answer)。耗尽即停。
    /// - `entropy_threshold`: `Some(t)` 启用连续 tokens 信息熵收敛检测(模型越确信熵越低,
    ///   连续 K 个 chunk 的近似熵低于 `t` 判定收敛);`None` 禁用此路径。
    ///   **当前实现状态**: 由于 gllm `GenerationResponse` 不暴露 per-token logit 流,
    ///   enropy 信号退化为**文本启发式**(字符多样性 / 重复短语比率)。
    ///   真实 logit-level 收敛检测依赖 `generation.rs` 暴露 token-level logit stream,
    ///   见 §5.2 未来扩展。
    /// - `stop_patterns`: 任意一个 substring 在 chunk 中命中即停。默认 ["Final Answer:",
    ///   "In conclusion", "Therefore, the answer is"]。
    Auto {
        max_total_tokens: usize,
        entropy_threshold: Option<f32>,
        stop_patterns: Vec<String>,
    },
}
```

### 3.2 ReasoningTemplate

```rust
pub struct ReasoningTemplate {
    /// 系统级 reasoning 引导。默认:
    /// "You are a careful reasoner. Break problems into explicit steps and
    ///  reach the final answer after all steps."
    pub system_prompt: String,

    /// 每 step 前置标记。`{n}` 会被替换为 1-based step index。
    /// 默认: "Step {n}:"
    pub step_prefix: String,

    /// Final answer 前置标记。默认: "Final Answer:"
    pub final_prefix: String,

    /// Step 间分隔符。默认: "\n\n"
    pub step_separator: String,

    /// 每 step 采样温度。默认 0.7。可降至 0.0 做贪婪推理。
    pub temperature: f32,

    /// 每 step top_k / top_p(用户可覆写;默认继承 Client 默认)。
    pub top_k: usize,
    pub top_p: f32,

    /// Final answer 阶段独立预算。默认 256。
    pub final_answer_budget: usize,
}
```

`ReasoningTemplate::default()` 提供开箱即用的英文 reasoning prompt。用户可部分覆写:

```rust
let tpl = ReasoningTemplate {
    step_prefix: "Reasoning {n}:".to_string(),
    ..Default::default()
};
```

### 3.3 默认 stop patterns

```rust
pub const DEFAULT_STOP_PATTERNS: &[&str] = &[
    "Final Answer:",
    "Final answer:",
    "In conclusion,",
    "Therefore, the answer is",
    "The answer is",
];
```

`ReasoningMode::Auto` 的 `stop_patterns` 字段可传入自定义列表完全替换默认。

---

## §4 Prompt Engineering Pipeline

### 4.1 初始上下文构造

```
<system_prompt>\n\n<user_prompt>
```

示例(用户 prompt = "What is 127 * 83?"):

```
You are a careful reasoner. Break problems into explicit steps and reach the final answer after all steps.

What is 127 * 83?
```

### 4.2 Step N prompt 拼接

```
<accumulated_context><step_separator><step_prefix 替换 {n}>
```

示例(第 2 步,已累积 step 1 输出):

```
You are a careful reasoner. ...

What is 127 * 83?

Step 1: Let me break 127 into 100 + 27 to make this easier.

Step 2:
```

模型接到此 prompt 后自然会输出 Step 2 的推理内容。

### 4.3 Final Answer prompt 拼接

```
<accumulated_context_with_all_steps><step_separator><final_prefix>
```

示例:

```
You are a careful reasoner. ...

What is 127 * 83?

Step 1: ...

Step 2: ...

Step 3: ...

Final Answer:
```

### 4.4 trace 保留

每个 step 产出的 chunk text 去除前缀后(optional trim)作为 trace 元素推入 `ReasoningResponse.reasoning_trace: Vec<String>`。`trace.len()` 反映实际执行步数。

---

## §5 Auto 停止信号

### 5.1 Pattern Match (主路径)

每 step chunk text 用 `contains(&pattern)` 匹配 `stop_patterns` 列表中的任一子串。命中即 `ReasoningStopReason::PatternMatched(matched_pattern)` 停止。

**注意**: pattern 匹配发生在 chunk 生成**之后**(事后检测),并非修改 prompt/logit 截停。下一步不会再触发该 chunk 的推理。

### 5.2 Entropy Convergence (当前实现: 文本启发式)

真实信息熵收敛需要 per-token logit stream。当前 gllm `GenerationResponse` 不暴露这一数据。故 §3.1 所述 "entropy_threshold" 字段当前实现为**文本启发式近似**:

- **字符多样性**: `unique_chars / total_chars` 比率,越低代表模型越重复。
- **短语重复**: 连续三 gram 重复比率,高重复 = 模型陷入循环 = 应停止。
- **合成 entropy 估计**: `estimated_entropy = diversity * (1 - repetition_ratio) * log2(word_count + 1)`

若 `estimated_entropy < entropy_threshold` → `EntropyConverged`。

**未来扩展路径(5.2 future)**: 当 `GenerationResponse` 扩展 `token_logits: Option<Vec<Vec<f32>>>` 字段,entropy 检测升级为真实 per-token `H(p) = -Σ p_i log p_i`,保留当前 API 兼容。

### 5.3 Budget Exhausted

`total_tokens + step_budget_estimate > max_total_tokens` → `BudgetExhausted`。`total_tokens` 用 byte-length / 4 的启发式近似(一个英文 token ≈ 4 字符),避免每轮 tokenize 开销。精确的 tokenizer.encode().len() 在 `Client::state` 可用时走精确路径,否则走启发式。

### 5.4 Step Count Reached (Manual only)

Manual 模式下 `for step in 1..=step_count` 正常完成所有 step → `StepCountReached`。不是"提前停止",是正常结束。

---

## §6 Manual 步长协调

### 6.1 Budget 分配

给定 `max_reasoning_tokens = 512, step_count = 3`:

```
step 1 budget = 512 / 3 = 170
step 2 budget = (512 - actual_tokens_step_1) / 2
step 3 budget = max_reasoning_tokens - actual_tokens_so_far
```

动态分配保证"未用完的 budget 会传递给后续 step",避免"每步死磕 170"导致的尾 step 饿死。

### 6.2 Step 间分隔

用 `step_separator`(默认 `"\n\n"`)拼接,模型在 prompt 中看到 `Step 1:... \n\n Step 2:` 形式后自然延续步号。

### 6.3 超budget 截断

若某 step chunk 长度超预期(小模型经常输出过长),下一步 budget 可能为 0 或负。处理:
- `step_budget < MIN_STEP_TOKENS (8)` → 提前终止,设 `stopped=BudgetExhausted`
- 已产出 step 保留在 trace 中,不因后续截断而回滚

---

## §7 Step Hook API — 推理引擎级思考控制 (REQ-COT-007..009)

### 7.1 动机

CoT Reasoner 的 Manual / Auto 模式内部是黑盒循环: 用户传入 `ReasoningMode`,引擎自主执行所有 step,返回最终结果。用户无法:

1. **在每步开始前介入**:注入额外知识(prompt 补充 / temperature 调整)
2. **在每步结束后读取中间结果**:获取累积 reasoning text 供外部分析
3. **动态控制流程**:跳过某步 / 提前终止 / 注入 prompt 后继续

Step Hook API 解决这些问题: 用户实现 `ReasoningStepHook` trait,注册到 `ReasoningBuilder`,引擎在每步边界调用 hook 方法,实现"推理引擎级思考控制"。

### 7.2 核心类型

```rust
/// Step 执行上下文 — 在 `on_step_start` 时提供给 hook。
///
/// 包含当前 step 的元信息和历史累积文本,允许 hook 基于已产出的
/// reasoning 内容做动态决策 (如注入补充 prompt、检测循环并终止)。
pub struct StepContext {
    /// 当前 step 的 0-based index。
    pub step_index: usize,
    /// 剩余总 budget (tokens)。Manual 模式 = 剩余 reasoning_tokens;
    /// Auto 模式 = 剩余 max_total_tokens。
    pub remaining_budget: usize,
    /// 之前所有 step 产出文本的累积拼接 (step_separator 分隔)。
    /// 第 0 步时为空串。
    pub accumulated_text: String,
    /// 当前使用的模型标识 (从 ClientState manifest 读取)。
    pub model_name: String,
}

/// `on_step_start` 返回的流程控制动作。
pub enum StepAction {
    /// 正常执行当前 step。
    Continue,
    /// 跳过当前 step(不调用 generate,不消耗 budget)。
    /// trace 中不记录此步。
    Skip,
    /// 在当前 step 的 prompt 末尾追加额外文本后继续执行。
    /// 注入的文本成为该 step prompt 的一部分,但不影响 template。
    InjectPrompt(String),
    /// 立即终止 reasoning 循环,附带终止原因。
    /// 已产出的 trace 保留,stopped_reason 设为 `HaltByHook(reason)`。
    Halt(String),
}

/// `on_step_end` 返回的知识注入。
pub struct StepKnowledge {
    /// 若为 `Some(text)`,text 会被追加到下一步 prompt 的末尾
    /// (在 step_prefix 之后、模型生成之前)。用于动态注入知识。
    pub inject_text: Option<String>,
    /// 若为 `Some(t)`,覆盖下一步的采样 temperature。
    /// 用于基于推理进展动态调整采样策略 (如逐步降温)。
    pub modified_temperature: Option<f32>,
}

impl Default for StepKnowledge {
    fn default() -> Self {
        Self {
            inject_text: None,
            modified_temperature: None,
        }
    }
}
```

### 7.3 ReasoningStepHook trait

```rust
/// Step-level 回调 trait。用户实现此 trait 并通过
/// `ReasoningBuilder::with_step_hook()` 注册,引擎在每步边界调用。
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
///
/// # 线程安全
///
/// `ReasoningStepHook: Send + Sync` — CoT 内部是同步循环,但 trait bound
/// 保证 hook 可在异步上下文或跨线程 Client 中安全使用。
pub trait ReasoningStepHook: Send + Sync {
    /// Step 开始前回调。返回 `StepAction` 控制当前步行为。
    fn on_step_start(&mut self, ctx: &StepContext) -> StepAction;

    /// Step 结束后回调。返回 `StepKnowledge` 注入到下一步。
    fn on_step_end(&mut self, result: &StepResult) -> StepKnowledge;
}
```

### 7.4 StepResult — `on_step_end` 输入

```rust
/// 单步执行结果 — 传递给 `on_step_end`。
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
```

### 7.5 ReasoningStopReason 扩展

在现有 4 种停止原因基础上新增 `HaltByHook`:

```rust
pub enum ReasoningStopReason {
    BudgetExhausted,
    StepCountReached,
    PatternMatched(String),
    EntropyConverged,
    /// Step hook 通过 `StepAction::Halt(reason)` 主动终止。
    HaltByHook(String),
}
```

### 7.6 ReasoningBuilder API 扩展

```rust
impl<'a> ReasoningBuilder<'a> {
    /// 注册 Step Hook。多次调用替换前一个 hook (单一 hook 语义)。
    ///
    /// Hook 在 `execute()` 内的 step 循环中被调用,生命周期覆盖
    /// 整个 reasoning 过程。
    pub fn with_step_hook(mut self, hook: Box<dyn ReasoningStepHook>) -> Self {
        self.step_hook = Some(hook);
        self
    }
}
```

`ReasoningBuilder` 新增字段:

```rust
pub struct ReasoningBuilder<'a> {
    client: &'a Client,
    prompt: String,
    mode: ReasoningMode,
    template: Option<ReasoningTemplate>,
    /// Step Hook (可选)。`None` = 无 hook,原有行为不变。
    step_hook: Option<Box<dyn ReasoningStepHook>>,
}
```

### 7.7 Hook 在循环中的集成

```
for step in 0..step_count {
    // ── on_step_start ──
    let ctx = StepContext { step_index, remaining_budget, accumulated_text, model_name };
    if let Some(hook) = &mut hook {
        match hook.on_step_start(&ctx) {
            StepAction::Continue => {},
            StepAction::Skip => continue,     // 不消耗 budget,不记录 trace
            StepAction::InjectPrompt(extra) => inject = extra,
            StepAction::Halt(reason) => { stopped = HaltByHook(reason); break; }
        }
    }

    // ── execute generation ──
    let chunk_text = client.generate(step_prompt + inject).response()?.text;

    // ── on_step_end ──
    let result = StepResult { step_index, chunk_text, tokens_used, accumulated_text };
    if let Some(hook) = &mut hook {
        let knowledge = hook.on_step_end(&result);
        if let Some(text) = knowledge.inject_text {
            context += text;   // 注入到下一步 prompt
        }
        if let Some(temp) = knowledge.modified_temperature {
            tpl.temperature = temp;  // 覆盖下一步温度
        }
    }
}
```

### 7.8 Hook 的 temperature 覆盖生命周期

`StepKnowledge::modified_temperature` 设置后,覆盖 `tpl.temperature` 直到下一次
`on_step_end` 再次覆盖或 reasoning 结束。这允许用户实现"逐步降温"策略:

```
Step 0: temperature=0.7 (default)
Step 1: on_step_end → modified_temperature=Some(0.5) → Step 2 用 0.5
Step 2: on_step_end → modified_temperature=Some(0.3) → Step 3 用 0.3
Step 3: on_step_end → None → Step 4 仍用 0.3 (上次覆盖保留)
```

### 7.9 Hook 语义不变量

1. **零开销缺席**: 无 hook (`step_hook = None`) 时,循环不执行任何额外分支判断,与原实现性能完全一致。
2. **Hook 失败不吞错误**: `on_step_start` / `on_step_end` 内部 panic 由 `ReasoningBuilder::execute` 的调用者捕获;不提供 catch_unwind 包装(与 gllm NO_SILENT_FALLBACK 一致)。
3. **Skip 不留空 trace**: `StepAction::Skip` 导致 `continue`,trace 中不插入空元素,`actual_steps` 只计实际 generate 的步数。
4. **HaltByHook 保留已有 trace**: hook 主动 Halt 时,已产出的 reasoning step 保留在 `reasoning_trace` 中,final answer 阶段仍然执行(基于已累积 context 生成最终回答)。
5. **InjectPrompt 单步生效**: `StepAction::InjectPrompt(extra)` 的 extra 仅追加到当前 step 的 prompt,不持久化到 context。若需跨步持久注入,在 `on_step_end` 返回 `StepKnowledge { inject_text: Some(...) }`。
6. **Backward compatible**: `ReasoningBuilder` 不带 hook 时行为与 REQ-COT-001..006 定义完全一致,所有已有 E2E 测试不受影响。

---

## §8 API 设计

### 8.1 Client 方法

```rust
impl Client {
    /// Chain-of-Thought 推理入口。
    ///
    /// - `prompt`: 用户原始 query(不含 "Step 1:" 等前缀,由 template 注入)
    /// - `mode`: Manual (fixed step_count) / Auto (engine-driven)
    /// - `template`: `None` 使用 `ReasoningTemplate::default()`;`Some(tpl)` 覆写
    pub fn reason(
        &self,
        prompt: &str,
        mode: ReasoningMode,
        template: Option<ReasoningTemplate>,
    ) -> Result<ReasoningResponse, ClientError>;
}
```

### 8.2 GenerationBuilder 便捷方法

```rust
impl<'a> GenerationBuilder<'a> {
    /// 将本次 generate 转为 CoT reasoning 调用。
    /// 调用 `.generate()` 时返回 `GenerationOutput::Response(Result<GenerationResponse, ...>)`,
    /// 其中 `.text` 为 final answer,完整 `ReasoningResponse` 通过 `Client::reason` 直接获取。
    pub fn reasoning(self, mode: ReasoningMode) -> ReasoningBuilder<'a>;
}
```

`ReasoningBuilder` 额外提供 `.template(ReasoningTemplate)` 和 `.with_step_hook(hook)` 链式方法。最终 `.execute()` 返回 `Result<ReasoningResponse, ClientError>`。

### 8.3 ReasoningResponse

```rust
pub struct ReasoningResponse {
    /// Final answer 文本(不含 reasoning trace)。
    pub text: String,
    /// 每步推理内容(按执行顺序)。
    pub reasoning_trace: Vec<String>,
    /// 所有 reasoning step 合计产出 token 数(估算或精确)。
    pub total_reasoning_tokens: usize,
    /// 实际执行步数(== reasoning_trace.len())。
    pub actual_steps: usize,
    /// 停止原因。
    pub stopped_reason: ReasoningStopReason,
}
```

### 8.4 ReasoningStopReason

```rust
pub enum ReasoningStopReason {
    /// 达到 max_reasoning_tokens / max_total_tokens 上限。
    BudgetExhausted,
    /// Manual 模式正常完成所有 step(不算提前停止)。
    StepCountReached,
    /// Auto 模式命中 stop_patterns 中某个子串。
    PatternMatched(String),
    /// Auto 模式 entropy 低于阈值判定收敛。
    EntropyConverged,
    /// Step hook 通过 `StepAction::Halt(reason)` 主动终止。
    HaltByHook(String),
}
```

### 8.5 ReasoningError

```rust
pub enum ReasoningError {
    /// Client 未加载模型。
    MissingModel,
    /// 下游 Client::generate 失败,消息透传。
    GenerationFailed(String),
    /// ReasoningMode / Template 参数非法(step_count=0 / budget=0 / NaN temperature 等)。
    InvalidConfig(String),
}
```

`ReasoningError` → `ClientError` 通过 `From` 转换:
- `MissingModel` → `ClientError::NoModelLoaded`
- `GenerationFailed(s)` → `ClientError::RuntimeError(format!("cot_reasoner: {s}"))`
- `InvalidConfig(s)` → `ClientError::RuntimeError(format!("cot_reasoner invalid config: {s}"))`

---

## §9 E2E 验收 (REQ-COT-001..009)

### REQ-COT-001: Manual 模式 budget 精确控制

```
输入: SmolLM2-135M-Instruct, Manual { max_reasoning_tokens=200, step_count=3 }
验收:
  (a) reasoning_trace.len() ≤ 3
  (b) total_reasoning_tokens ≤ 200 (允许 ≤ 10% 溢出,小模型 tokenizer 估算误差)
  (c) stopped_reason ∈ { StepCountReached, BudgetExhausted }
  (d) text (final answer) 非空
```

### REQ-COT-002: Auto 模式 pattern-match 停止

```
输入: Auto { max_total_tokens=1024, entropy_threshold=None, stop_patterns=["Final Answer:"] }
     + 模板注入 "Final Answer:" 前缀的自然语言场景
验收:
  (a) 若生成文本中出现 "Final Answer:" 子串 →
      stopped_reason == PatternMatched("Final Answer:")
  (b) 若未出现(小模型可能不遵循) →
      stopped_reason ∈ { BudgetExhausted, EntropyConverged } (非 PatternMatched)
  (c) reasoning_trace.len() ≥ 1
```

### REQ-COT-003: Auto 模式 entropy-convergence 停止

```
输入: Auto { entropy_threshold=Some(0.5), stop_patterns=vec![] } + 触发重复输出的 prompt
验收:
  (a) estimated_entropy 启发式达到 < 0.5 的 chunk 后停止
  (b) stopped_reason == EntropyConverged
  (c) 单元测试覆盖: 注入文本启发式已知模式直接判定,不依赖真实模型
  (d) 文档说明: 真实 logit-level entropy 检测依赖 GenerationResponse 扩展(§5.2)
```

### REQ-COT-004: Reasoning trace 完整保留

```
输入: Manual { step_count=3 } 成功跑完
验收:
  (a) reasoning_trace.len() == actual_steps
  (b) 每个 trace 元素 !is_empty() (step chunk 非空白)
  (c) trace 拼接 + final answer 语义与原生 generate(longer_budget) 可比(不严格数值相等,
      仅验证结构存在)
```

### REQ-COT-005: 与 SG 正交

```
输入: Client 先 register_semantic_gatekeeper(cfg) 再 reason(...)
验收:
  (a) reason 调用成功,不 panic
  (b) reason 内部每次 Client::generate 都能命中 SG callback(通过对比 SG probe 计数器)
  (c) 不需要修改 SG API
```

当前 SG 实现状态为 🔴 待实现(REQ-SG),REQ-COT-005 单元测试以**构造 mock SG provider 计数器**的形式覆盖,E2E 层随 SG 落地后自然打通。

### REQ-COT-006: NO_ISLAND_MODULE

```
静态验证:
  grep -rn "Client::reason\|client\.reason(" src/ tests/ --include="*.rs" \
    | grep -v "#\[cfg(test)\]" | grep -v "mod tests"
  → 返回 src/generation.rs::GenerationBuilder::reasoning 的真实转发点 (非空)

运行时验证:
  test_cot_006_arbitrary_llm 用 SmolLM2-135M-Instruct 跑通 Client::reason,
  证明调用链: 测试 → GenerationBuilder::reasoning → ReasoningBuilder::execute →
  Client::reason → 多次 Client::execute_generation → Backend::generate
```

### REQ-COT-007: Step Hook trait 定义与生命周期

```
输入: 实现 ReasoningStepHook trait 的结构体,通过 ReasoningBuilder::with_step_hook 注册
验收:
  (a) ReasoningStepHook trait 有 on_step_start + on_step_end 两个方法
  (b) trait bound 包含 Send + Sync
  (c) E2E 测试: 注册 hook 后执行 Manual 3 步 reasoning,验证 on_step_start 和
      on_step_end 各被调用 3 次 (每步各 1 次)
  (d) 不注册 hook 时 (step_hook = None),循环行为与无 hook 完全一致
      (backward compatible,已有测试不受影响)
```

### REQ-COT-008: StepContext 包含 accumulated reasoning text

```
输入: 注册 hook 执行 Manual 3 步 reasoning
验收:
  (a) on_step_start 的 StepContext.accumulated_text 在 step_index=0 时为空串
  (b) on_step_start 的 StepContext.accumulated_text 在 step_index=1 时包含
      step 0 的 chunk_text (trimmed)
  (c) on_step_start 的 StepContext.accumulated_text 在 step_index=2 时包含
      step 0 + step 1 的累积文本
  (d) StepContext.model_name 非空 (从 ClientState manifest 读取)
  (e) StepContext.remaining_budget > 0 且逐步递减
```

### REQ-COT-009: StepAction 支持 Continue/Skip/InjectPrompt/Halt

```
输入: 分别注册返回不同 StepAction 的 hook
验收:
  (a) StepAction::Continue: 正常执行所有 step,stopped_reason = StepCountReached
  (b) StepAction::Skip: 某步被跳过,reasoning_trace 跳过该步 (无空元素),
      actual_steps = step_count - skip_count
  (c) StepAction::InjectPrompt(extra): extra 文本出现在该步的 generate prompt 中,
      chunk_text 语义受注入影响 (验证 chunk 包含注入关键词)
  (d) StepAction::Halt(reason): 提前终止循环,stopped_reason = HaltByHook(reason),
      reason 字符串保留在 stopped_reason 中,已有 trace 保留
  (e) E2E 测试覆盖 InjectPrompt: 注册 hook 在 step 1 注入 "Remember pi=3.14",
      验证 step 1 的 chunk_text 包含 "3.14" 或 "pi"
```

---

## §10 与 Semantic Gatekeeper / Head Routing 正交性

| 关注点 | COT Reasoner | Semantic Gatekeeper | Head Routing |
|--------|--------------|---------------------|--------------|
| 作用层 | Client orchestration (多次调用) | forward 中间层 callback | forward 完成后读 hidden |
| 粒度 | 多轮 generate | 每 decode step 检测层 | 每次 HR API 调用一次 |
| 修改对象 | prompt + iteration | `hidden_state[-1]` 注入残差 | 不修改,只读 |
| 依赖权重 | 否 | 否 | 否 (依赖 tied lm_head/embed) |
| 新增 Backend API | **否** | 是 (q_tap) | 否 |

**组合使用场景**:
- **COT + SG**: `register_semantic_gatekeeper(cfg)` 后调用 `reason(...)`,每 step 的 generate 都自动走 SG 注入,step 间上下文语义被知识增强。
- **COT + HR**: reason 用于生成推理 trace,最终 `Client::classify_binary` 把 trace 送入分类。两者独立 API 调用,自然串联。

---

## §11 实现映射

| SPEC 条目 | 代码位置 |
|-----------|----------|
| `ReasoningMode` / `ReasoningTemplate` / `ReasoningResponse` / `ReasoningStopReason` / `ReasoningError` / `DEFAULT_STOP_PATTERNS` | `src/cot_reasoner.rs` |
| `StepContext` / `StepAction` / `StepKnowledge` / `StepResult` / `ReasoningStepHook` trait | `src/cot_reasoner.rs` |
| `Client::reason` | `src/cot_reasoner.rs` (impl Client 扩展块) |
| `GenerationBuilder::reasoning` / `ReasoningBuilder` | `src/generation.rs` + `src/cot_reasoner.rs` |
| 公共导出 | `src/lib.rs` |
| 单元测试 (模板/budget/entropy 启发式) | `src/cot_reasoner.rs` `#[cfg(test)] mod tests` |
| E2E 测试 (REQ-COT-001..009) | `tests/test_e2e_cot_reasoner.rs` |

---

## §12 变更历史

SPEC 文件不维护变更历史(CLAUDE.md §6 铁律 4: SPEC 防腐化)。版本历史由 git 管理。本 SPEC 当前描述的是**唯一正确设计**,如需更新设计,整段重写对应章节。
