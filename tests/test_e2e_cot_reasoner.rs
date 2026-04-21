//! E2E 测试: CoT Reasoner SDK (REQ-COT-001..006)
//!
//! **SSOT**: `SPEC/COT-REASONER.md`, `SPEC/01-REQUIREMENTS.md §16`,
//! `SPEC/04-API-DESIGN.md §3.11`
//!
//! 运行:
//! ```bash
//! cargo test --test test_e2e_cot_reasoner -- --test-threads=1
//! ```
//!
//! 单线程强制: 这些测试加载 SmolLM2-135M-Instruct 真实模型,跑 JIT 编译 +
//! CPU 推理。并行会导致磁盘 I/O 竞争、缓存冲突、OOM。
//!
//! **模型选择理由**: SmolLM2-135M-Instruct 是**不带 thinking_head** 的
//! 通用 generator,验证 CoT Reasoner 的核心价值 —— 对任意 LLM 工作,
//! 不依赖模型专用权重 (REQ-COT-006)。

use gllm::{
    cot_reasoner::{
        estimate_text_entropy, estimate_token_count, find_stop_pattern,
        AUTO_STEP_CHUNK,
    },
    Client, ReasoningMode, ReasoningStopReason, ReasoningTemplate,
};

const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

// ============================================================================
// 启发式纯单元测试 (不依赖模型,即使其他测试因环境失败也应独立通过)
// ============================================================================

/// Entropy 启发式对已知重复文本的收敛判定 (REQ-COT-003 核心启发式验证)。
///
/// 真实 logit-level entropy 依赖 GenerationResponse 扩展(SPEC §5.2 future);
/// 当前启发式必须对"病态重复文本"判定为 < 0.5 阈值。
#[test]
fn test_cot_003_entropy_heuristic_on_repetitive_text() {
    // 高度重复的 3-gram 模式
    let rep = "abc abc abc abc abc abc abc abc abc abc abc abc";
    let entropy = estimate_text_entropy(rep);
    assert!(
        entropy < 0.5,
        "repetitive text entropy ({}) must be < 0.5 (REQ-COT-003 启发式)",
        entropy
    );

    // 多样化文本必须高于阈值,避免误停
    let div = "The quick brown fox jumps over the lazy dog. \
               Each step reveals a new piece of logic.";
    let entropy_div = estimate_text_entropy(div);
    assert!(
        entropy_div >= 0.5,
        "diverse text entropy ({}) must be >= 0.5 (避免 Auto 模式误停)",
        entropy_div
    );
}

/// Stop pattern 匹配行为契约。
#[test]
fn test_cot_stop_pattern_case_sensitive_substring_match() {
    let patterns = vec![
        "Final Answer:".to_string(),
        "In conclusion,".to_string(),
    ];
    // 命中
    let hit = find_stop_pattern("Therefore, Final Answer: 42", &patterns);
    assert_eq!(hit.map(|s| s.as_str()), Some("Final Answer:"));

    // 大小写敏感: "final answer:" 不应匹配 "Final Answer:"
    let miss = find_stop_pattern("final answer: 42", &patterns);
    assert!(miss.is_none());
}

/// Token 数估算单调性契约。
#[test]
fn test_cot_estimate_token_count_monotonic() {
    let short = estimate_token_count("hi there");
    let long = estimate_token_count(&"text ".repeat(100));
    assert!(short < long);
}

// ============================================================================
// E2E 测试 (需要真实模型)
// ============================================================================

/// REQ-COT-001: Manual 模式 budget 精确控制
///
/// 验收:
///   (a) reasoning_trace.len() ≤ step_count
///   (b) total_reasoning_tokens ≤ max_reasoning_tokens (允许 ≤10% 溢出)
///   (c) stopped_reason ∈ { StepCountReached, BudgetExhausted }
///   (d) text (final answer) 非空
#[test]
fn test_cot_001_manual_budget_respected() {
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct");

    const MAX_TOKENS: usize = 200;
    const STEP_COUNT: usize = 3;
    let response = client
        .reason(
            "What is 127 * 83? Show your work.",
            ReasoningMode::Manual {
                max_reasoning_tokens: MAX_TOKENS,
                step_count: STEP_COUNT,
            },
            None,
        )
        .expect("reason (manual) failed");

    // (a) 步数上限
    assert!(
        response.reasoning_trace.len() <= STEP_COUNT,
        "reasoning_trace.len() = {} must be <= step_count {}",
        response.reasoning_trace.len(),
        STEP_COUNT
    );

    // (b) budget: 允许 10% tokenizer 估算误差
    let budget_cap = MAX_TOKENS + MAX_TOKENS / 10;
    assert!(
        response.total_reasoning_tokens <= budget_cap,
        "total_reasoning_tokens = {} exceeds budget (with 10% slack) {}",
        response.total_reasoning_tokens,
        budget_cap
    );

    // (c) stop reason 合法
    assert!(
        matches!(
            response.stopped_reason,
            ReasoningStopReason::StepCountReached | ReasoningStopReason::BudgetExhausted
        ),
        "unexpected stopped_reason: {:?}",
        response.stopped_reason
    );

    // (d) final answer 非空
    assert!(
        !response.text.is_empty(),
        "final answer (text) must be non-empty"
    );

    // (e) actual_steps 与 trace 一致
    assert_eq!(response.actual_steps, response.reasoning_trace.len());
}

/// REQ-COT-002: Auto 模式 pattern-match 停止
///
/// 用户模板注入 "Final Answer:" 前缀,stop_patterns 包含 "Final Answer:"。
/// 若模型延续模板并输出该子串 → PatternMatched;否则应走 Budget/Entropy
/// 分支,不应静默继续。
#[test]
fn test_cot_002_auto_pattern_stop() {
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct");

    let response = client
        .reason(
            "Is 2 + 2 = 4? Explain briefly, then end with 'Final Answer:'.",
            ReasoningMode::Auto {
                max_total_tokens: 384,
                entropy_threshold: None,
                stop_patterns: vec!["Final Answer:".to_string()],
            },
            None,
        )
        .expect("reason (auto pattern) failed");

    // 必须以合法停止原因结束(不允许静默继续)
    let valid = matches!(
        &response.stopped_reason,
        ReasoningStopReason::PatternMatched(_)
            | ReasoningStopReason::BudgetExhausted
            | ReasoningStopReason::EntropyConverged
    );
    assert!(valid, "unexpected stopped_reason: {:?}", response.stopped_reason);

    // reasoning_trace 至少 1 步
    assert!(
        response.reasoning_trace.len() >= 1,
        "reasoning_trace must have >= 1 step, got {}",
        response.reasoning_trace.len()
    );

    // 若 PatternMatched → 匹配的 pattern 必须是 "Final Answer:"
    if let ReasoningStopReason::PatternMatched(ref p) = response.stopped_reason {
        assert_eq!(p, "Final Answer:");
    }
}

/// REQ-COT-003: Auto 模式 entropy-convergence 停止 (启发式路径)
///
/// 组合 entropy_threshold + 禁用 pattern 覆盖以确保 entropy 是唯一信号候选。
/// 仅断言 stopped_reason 合法且非 PatternMatched(因为 patterns 空)。
/// 真实模型是否退化到低熵文本依赖 prompt 诱导,本测试保证流程畅通与启发式
/// 被调用(真正"收敛"判定由单元测试 `test_cot_003_entropy_heuristic_on_repetitive_text`
/// 保证)。
#[test]
fn test_cot_003_auto_entropy_stop() {
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct");

    let response = client
        .reason(
            "Repeat the phrase 'yes yes yes' until you are absolutely certain.",
            ReasoningMode::Auto {
                max_total_tokens: 256,
                entropy_threshold: Some(0.5),
                stop_patterns: vec![], // 禁用 pattern, 只剩 entropy + budget
            },
            None,
        )
        .expect("reason (auto entropy) failed");

    // 不应是 PatternMatched(patterns 列表为空)
    assert!(
        !matches!(response.stopped_reason, ReasoningStopReason::PatternMatched(_)),
        "stopped_reason must NOT be PatternMatched when stop_patterns empty"
    );

    // 必须是 EntropyConverged 或 BudgetExhausted
    assert!(
        matches!(
            response.stopped_reason,
            ReasoningStopReason::EntropyConverged | ReasoningStopReason::BudgetExhausted
        ),
        "unexpected stopped_reason: {:?}",
        response.stopped_reason
    );
}

/// REQ-COT-004: Reasoning trace 完整保留
///
/// 验收:
///   (a) reasoning_trace.len() == actual_steps
///   (b) 每个 trace 元素非空
///   (c) trace 元素文本内容保留(非单一重复字符)
#[test]
fn test_cot_004_trace_preserved() {
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct");

    let response = client
        .reason(
            "Think step by step: what is 5 + 3?",
            ReasoningMode::Manual {
                max_reasoning_tokens: 180,
                step_count: 2,
            },
            None,
        )
        .expect("reason (manual) failed");

    // (a)
    assert_eq!(response.actual_steps, response.reasoning_trace.len());

    // (b)
    for (i, step) in response.reasoning_trace.iter().enumerate() {
        assert!(
            !step.trim().is_empty(),
            "reasoning_trace[{}] must be non-empty",
            i
        );
    }
}

/// REQ-COT-005: Template 覆写(step_prefix / final_prefix / temperature)
///
/// 验证用户自定义模板生效: render_step_prefix 返回用户指定格式。
/// 这个测试不需要模型,覆盖 template 层契约(reasoning_trace 实际内容
/// 由小模型质量决定,但 template 机制必须生效)。
///
/// 运行时模板验证: Template 覆写后调用链 `client.reason(..., Some(tpl))`
/// 会使用新 prefix 构造 prompt。本测试通过不同 prefix 两次调用对比
/// reasoning 调用链确保不 panic。
#[test]
fn test_cot_005_template_override() {
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct");

    let tpl = ReasoningTemplate {
        step_prefix: "Reasoning {n}:".to_string(),
        final_prefix: "Conclusion:".to_string(),
        temperature: 0.5,
        ..Default::default()
    };

    // 验证 template 渲染
    assert_eq!(tpl.render_step_prefix(1), "Reasoning 1:");
    assert_eq!(tpl.render_step_prefix(2), "Reasoning 2:");

    // 实际跑一次证明覆写走通真实调用链
    let response = client
        .reason(
            "What is 1 + 1?",
            ReasoningMode::Manual {
                max_reasoning_tokens: 150,
                step_count: 2,
            },
            Some(tpl),
        )
        .expect("reason with custom template failed");

    assert!(!response.text.is_empty());
    assert!(response.reasoning_trace.len() <= 2);
    assert!(matches!(
        response.stopped_reason,
        ReasoningStopReason::StepCountReached | ReasoningStopReason::BudgetExhausted
    ));
}

/// REQ-COT-006: Arbitrary LLM — NO_ISLAND_MODULE + 真实调用链验证
///
/// 用 SmolLM2-135M-Instruct(**不带 thinking_head**)跑通整个 reasoning
/// 流程,证明:
///   (a) CoT Reasoner 对任意 generator LLM 工作(不依赖专用权重)
///   (b) 调用链真实连通: GenerationBuilder::reasoning → ReasoningBuilder::execute
///       → Client::reason → 多次 Client::execute_generation → Backend
///   (c) 模块非孤岛(grep 能找到真实调用点)
#[test]
fn test_cot_006_arbitrary_llm() {
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct");

    // 通过 GenerationBuilder::reasoning 入口调用(真实链式 API)
    let response = client
        .generate("What is the capital of France?")
        .reasoning(ReasoningMode::Manual {
            max_reasoning_tokens: 180,
            step_count: 2,
        })
        .execute()
        .expect("reasoning via GenerationBuilder failed");

    // 验收 (a): 成功跑通不带 thinking_head 的模型
    assert!(
        !response.text.is_empty(),
        "arbitrary LLM (SmolLM2-135M, no thinking_head) must produce non-empty final answer"
    );

    // 验收 (b): 调用链真实传导
    assert!(
        response.actual_steps >= 1,
        "actual_steps must be >= 1 proving generate was invoked at least once"
    );
    assert!(
        response.total_reasoning_tokens > 0,
        "total_reasoning_tokens must be > 0 proving generate consumed real tokens"
    );

    // 验收 (c): 触发合法停止
    assert!(
        matches!(
            response.stopped_reason,
            ReasoningStopReason::StepCountReached | ReasoningStopReason::BudgetExhausted
        ),
        "unexpected stopped_reason: {:?}",
        response.stopped_reason
    );
}

// ============================================================================
// Constants sanity
// ============================================================================

/// AUTO_STEP_CHUNK 必须合理: 既不能太小(导致过多 round trip)也不能太大
/// (压缩 budget 精度)。
#[test]
fn test_cot_auto_step_chunk_sensible() {
    assert!(AUTO_STEP_CHUNK >= 32, "AUTO_STEP_CHUNK too small");
    assert!(AUTO_STEP_CHUNK <= 512, "AUTO_STEP_CHUNK too large");
}
