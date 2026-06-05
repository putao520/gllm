//! E2E 测试: Guardrail SDK (REQ-GR-001..005)
//!
//! **SSOT**: `SPEC/GUARDRAIL.md`, `SPEC/01-REQUIREMENTS.md §14`
//!
//! 运行:
//! ```bash
//! cargo test --test test_e2e_guardrail -- --test-threads=1
//! ```
//!
//! 单线程强制: 加载 SmolLM2-135M-Instruct 真实模型 + CPU 推理, 并发会触发
//! 资源竞争 (磁盘 I/O、JIT 缓存、权重共享) 导致结果不稳定.

use gllm::{
    ClassifyBinaryConfig, Client, GuardProbe, GuardProbeWeights, LayerAnchor, SafetyPolicy,
};

const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

fn build_client() -> Client {
    Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct")
}

/// TEST-GR-001: `attach_guardrail` + 零权重探针不触发 veto (REQ-GR-001)
///
/// 场景: 挂一个全零权重的 GuardProbe (w=0, b=0 → sigmoid(0) = 0.5)
/// 配合 `HaltAndVeto { threshold: 0.99 }` (永远达不到的阈值). 前向应正常
/// 完成, attachment.is_vetoed() = false, last_score ≈ 0.5.
#[test]
fn test_gr_001_zero_probe_no_veto() {
    let client = build_client();
    let (hidden_size, num_layers) = client.model_dims().expect("model_dims");
    assert!(num_layers > 0 && hidden_size > 0);

    let weights = GuardProbeWeights {
        weight: vec![0.0f32; hidden_size],
        bias: 0.0,
    };
    let attachment = client
        .attach_guardrail_inline(
            weights,
            LayerAnchor::Relative(0.5),
            SafetyPolicy::HaltAndVeto { threshold: 0.99 },
        )
        .expect("attach_guardrail_inline failed");

    assert!(attachment.id > 0, "attachment id must be positive");
    assert!(
        attachment.actual_layer < num_layers,
        "actual_layer {} out of [0, {})",
        attachment.actual_layer,
        num_layers
    );

    // 跑 classify_binary — 探针应被触发, 但 score ≈ 0.5 < 0.99 → Continue.
    let score = client
        .classify_binary(
            "Is Paris in France?",
            ClassifyBinaryConfig::new("yes", "no"),
        )
        .expect("classify_binary with zero-probe guardrail failed");

    assert!(score.is_finite() && (0.0..=1.0).contains(&score));
    assert!(
        !attachment.is_vetoed(),
        "Zero-probe with threshold=0.99 should NOT veto; got is_vetoed={} last_reason={:?}",
        attachment.is_vetoed(),
        attachment.last_veto_reason()
    );
    // last_score 应在 ~0.5 附近 (真实 hidden_state 经零权重点积 + 零 bias → sigmoid(0) = 0.5)
    let last = attachment
        .last_score()
        .expect("last_score must be populated after forward");
    assert!(
        (last - 0.5).abs() < 1e-3,
        "zero probe score should be 0.5, got {last}"
    );

    client.detach_guardrail(attachment.id).expect("detach failed");
}

/// TEST-GR-002: `HaltAndVeto` 触发 — 强正 bias 使 score → 1 > threshold (REQ-GR-002)
///
/// 探针 w=0, b=+20 → sigmoid(20) ≈ 1.0 > threshold=0.5 → HaltAndVeto.
/// 验证:
/// 1. `classify_binary` 返回 Err(guardrail vetoed)
/// 2. `attachment.is_vetoed()` = true
/// 3. `attachment.last_veto_reason()` 包含 "vetoed" 子串
#[test]
fn test_gr_002_halt_and_veto_triggers() {
    let client = build_client();
    let (hidden_size, _num_layers) = client.model_dims().expect("model_dims");

    let weights = GuardProbeWeights {
        weight: vec![0.0f32; hidden_size],
        bias: 20.0, // sigmoid(20) ≈ 1.0
    };
    let attachment = client
        .attach_guardrail_inline(
            weights,
            LayerAnchor::Relative(0.5),
            SafetyPolicy::HaltAndVeto { threshold: 0.5 },
        )
        .expect("attach_guardrail_inline failed");

    let result = client.classify_binary(
        "Is water wet?",
        ClassifyBinaryConfig::new("yes", "no"),
    );

    assert!(
        result.is_err(),
        "classify_binary under vetoing guardrail must return Err, got Ok({:?})",
        result.ok()
    );
    assert!(
        attachment.is_vetoed(),
        "attachment must be marked vetoed after classify_binary"
    );
    let reason = attachment
        .last_veto_reason()
        .expect("veto reason must be populated");
    assert!(
        reason.to_lowercase().contains("vetoed") || reason.to_lowercase().contains("score"),
        "veto reason should mention veto/score: {reason}"
    );
    let score = attachment
        .last_score()
        .expect("last_score must be populated");
    assert!(
        score > 0.99,
        "strong positive bias probe score should be > 0.99, got {score}"
    );

    client.detach_guardrail(attachment.id).expect("detach failed");
}

/// TEST-GR-003: `LogOnly` 不改变生成 (REQ-GR-003)
///
/// 无论分数多高, LogOnly 只记录 score, 不触发 veto. 验证:
/// 1. classify_binary 正常返回分数 (与无 guardrail 时相同)
/// 2. attachment.is_vetoed() = false
/// 3. attachment.last_score() 被填充
#[test]
fn test_gr_003_log_only_does_not_veto() {
    let client = build_client();
    let (hidden_size, _num_layers) = client.model_dims().expect("model_dims");

    // 基准分数 (无 guardrail)
    let baseline = client
        .classify_binary(
            "Is grass green?",
            ClassifyBinaryConfig::new("yes", "no"),
        )
        .expect("baseline classify_binary failed");

    // 挂 LogOnly + 强 bias 探针
    let attachment = client
        .attach_guardrail_inline(
            GuardProbeWeights {
                weight: vec![0.0f32; hidden_size],
                bias: 100.0,
            },
            LayerAnchor::Relative(0.5),
            SafetyPolicy::LogOnly,
        )
        .expect("attach LogOnly failed");

    let score_with_log = client
        .classify_binary(
            "Is grass green?",
            ClassifyBinaryConfig::new("yes", "no"),
        )
        .expect("classify_binary with LogOnly must still succeed");

    assert!(
        !attachment.is_vetoed(),
        "LogOnly must NOT veto regardless of probe score"
    );
    let last = attachment.last_score().expect("score recorded");
    assert!(last > 0.99, "LogOnly still records high score: {last}");

    // LogOnly 完全不改变前向, 两次分数应严格相等 (确定性前向).
    assert!(
        (score_with_log - baseline).abs() < 1e-4,
        "LogOnly must not perturb logits: baseline={baseline}, with_log={score_with_log}"
    );

    client.detach_guardrail(attachment.id).expect("detach");
}

/// TEST-GR-004: `SampleDowngrade` 记录最低温度 (REQ-GR-004)
///
/// SampleDowngrade 不中断前向但写 downgrade_temp. classify_binary 正常返回.
#[test]
fn test_gr_004_sample_downgrade_records_temperature() {
    let client = build_client();
    let (hidden_size, _num_layers) = client.model_dims().expect("model_dims");

    let attachment = client
        .attach_guardrail_inline(
            GuardProbeWeights {
                weight: vec![0.0f32; hidden_size],
                bias: 10.0, // arbitrary, just need to trigger
            },
            LayerAnchor::Relative(0.5),
            SafetyPolicy::SampleDowngrade {
                min_temperature: 0.3,
            },
        )
        .expect("attach SampleDowngrade failed");

    let _score = client
        .classify_binary("Hello", ClassifyBinaryConfig::new("yes", "no"))
        .expect("classify_binary with SampleDowngrade must succeed");

    assert!(!attachment.is_vetoed(), "SampleDowngrade never vetoes");
    let downgrade = attachment
        .downgraded_temperature()
        .expect("downgrade_temperature must be recorded");
    assert!(
        (downgrade - 0.3).abs() < 1e-6,
        "SampleDowngrade should record min_temperature=0.3, got {downgrade}"
    );

    client.detach_guardrail(attachment.id).expect("detach");
}

/// TEST-GR-005: `attach_guardrail` + `detach_guardrail` 幂等 id 管理 (REQ-GR-005)
///
/// 多次 attach 生成单调递增 id; detach 未知 id 返回 Err; detach 后再次
/// detach 返回 Err.
#[test]
fn test_gr_005_attach_detach_id_lifecycle() {
    let client = build_client();
    let (hidden_size, _num_layers) = client.model_dims().expect("model_dims");

    let a1 = client
        .attach_guardrail_inline(
            GuardProbeWeights {
                weight: vec![0.0; hidden_size],
                bias: 0.0,
            },
            LayerAnchor::Relative(0.5),
            SafetyPolicy::LogOnly,
        )
        .unwrap();
    let a2 = client
        .attach_guardrail_inline(
            GuardProbeWeights {
                weight: vec![0.0; hidden_size],
                bias: 0.0,
            },
            LayerAnchor::Relative(0.7),
            SafetyPolicy::LogOnly,
        )
        .unwrap();
    assert!(a2.id > a1.id, "ids must monotonically increase");

    // detach 未知 id
    assert!(client.detach_guardrail(999_999_999).is_err());

    // detach 正常
    client.detach_guardrail(a1.id).expect("detach a1");
    // 重复 detach
    assert!(
        client.detach_guardrail(a1.id).is_err(),
        "second detach of same id must fail"
    );
    // 另一个仍有效
    client.detach_guardrail(a2.id).expect("detach a2");
}

/// TEST-GR-006: `GuardProbe::from_safetensors` 对缺失文件显式 Err (NO_SILENT_FALLBACK)
#[test]
fn test_gr_006_safetensors_missing_file_errors() {
    let client = build_client();
    let err = client.attach_guardrail(
        GuardProbe::from_safetensors("/tmp/__nonexistent_guardrail_probe__.safetensors"),
        LayerAnchor::Relative(0.5),
        SafetyPolicy::LogOnly,
    );
    assert!(err.is_err(), "attach_guardrail with missing file must error");
}
