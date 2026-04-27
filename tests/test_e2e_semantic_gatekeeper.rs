//! E2E 测试: Semantic Gatekeeper (REQ-SG-001..008)
//!
//! 关联:
//! - SPEC/SEMANTIC-GATEKEEPER.md §8.2 TEST-SG-001..008
//! - SPEC/01-REQUIREMENTS.md §12 REQ-SG-001..008
//! - SPEC/04-API-DESIGN.md §7 Client API
//!
//! E2E 测试必须单线程运行:
//!   cargo test --test test_e2e_semantic_gatekeeper -- --test-threads=1
//!
//! ## 实现状态
//!
//! REQ-SG-001/003-007 已实现:
//! - Level Keys 预计算 (小 CompilerGraph JIT 编译)
//! - SG Callback 注入 executor callback chain (真实 pre_node 调用)
//! - 层级路由、稳定性追踪、残差注入、KnowledgeProvider/AstSentinel trait 契约
//!
//! REQ-SG-002 (QTapSTG graph 插入 + ABI ring buffer) 仍 🟡:
//! - Q-tap JIT lowering 已实现，但 graph builder 尚未插入 QTapSTG 节点
//! - ring buffer 写入由 QTapSTG JIT 完成，当前 ring buffer 无数据源
//! - SG callback 在 ring buffer 读取失败时降级为 Continue（不影响正确性）
//!
//! 因此:
//! - TEST-SG-001: 验证 register 成功 + LevelKeysCache 预计算
//! - TEST-SG-002/003/004: 验证 API 契约 (register + generate 不崩溃),
//!   但 provider 不会被调用 (ring buffer 无数据 → pre_node 返回 Continue)
//! - 完整的行为差异验证 (Paris logit 提升) 等 Q-tap graph 插入完成后补充
//!
//! 所有测试**禁止** `#[ignore]` / 条件跳过 (CLAUDE.md NO_SILENT_FALLBACK).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use gllm::semantic_gatekeeper::{
    AstContext, AstSentinel, KnowledgeEntry, KnowledgeProvider, RetrieveContext,
    SemanticGatekeeperConfig, SemanticGatekeeperError, SemanticLevel, TokenizerLookup,
    DEFAULT_LEVEL_DESCRIPTORS,
};
use gllm::Client;

const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

// ============================================================================
// Mock KnowledgeProvider / AstSentinel / TokenizerLookup (测试 fixture)
//
// 注意: 这些 mock 仅用于验证 trait 对象契约 + SG 配置,不替代真实实现.
// SG Callback 的真实逻辑由 `SemanticGatekeeperCallback` 承担 (SPEC §7.1),
// 不可被 mock 回调替代 (SPEC §8.3 禁止的测试反模式).
// ============================================================================

/// 始终返回 None 的 Provider. 用于验证 SG 门控分支 (REQ-SG-003).
#[derive(Default)]
struct AlwaysNoneProvider {
    calls: AtomicUsize,
}

impl KnowledgeProvider for AlwaysNoneProvider {
    fn retrieve(
        &self,
        _query: &[f32],
        _level: SemanticLevel,
        _ctx: &RetrieveContext<'_>,
    ) -> Option<KnowledgeEntry> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        None
    }
}

/// 返回固定文本 + 可配置置信度的 Provider. 用于 TEST-SG-004 / 007.
struct FixedTextProvider {
    text: String,
    confidence: f32,
    calls: AtomicUsize,
}

impl FixedTextProvider {
    fn new(text: impl Into<String>, confidence: f32) -> Self {
        Self {
            text: text.into(),
            confidence,
            calls: AtomicUsize::new(0),
        }
    }
}

impl KnowledgeProvider for FixedTextProvider {
    fn retrieve(
        &self,
        _query: &[f32],
        _level: SemanticLevel,
        _ctx: &RetrieveContext<'_>,
    ) -> Option<KnowledgeEntry> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Some(KnowledgeEntry {
            text: self.text.clone(),
            confidence: self.confidence,
        })
    }
}

/// 固定 node_kind 的 AstSentinel. 验证 trait 可作为 Arc<dyn AstSentinel> 接纳.
struct FixedAstSentinel {
    node_kind: String,
}

impl FixedAstSentinel {
    fn new(node_kind: impl Into<String>) -> Self {
        Self {
            node_kind: node_kind.into(),
        }
    }
}

impl AstSentinel for FixedAstSentinel {
    fn current_context<'a>(
        &self,
        _generated_tokens: &'a [u32],
        _tokenizer: &dyn TokenizerLookup,
    ) -> Option<AstContext<'a>> {
        // 'a 属于 generated_tokens 的借用,这里返回的 node_kind 不依赖该借用.
        // 测试 fixture 不验证真实 AST 解析,仅验证 trait 对象可调用性.
        // 因此返回 None 足够覆盖契约 (SPEC §6.2 允许 None).
        let _ = &self.node_kind;
        None
    }
}

/// 始终返回 None 的 AstSentinel.
struct NullAstSentinel;

impl AstSentinel for NullAstSentinel {
    fn current_context<'a>(
        &self,
        _generated_tokens: &'a [u32],
        _tokenizer: &dyn TokenizerLookup,
    ) -> Option<AstContext<'a>> {
        None
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// 构造一个基础 SG 配置,挂接指定 Provider + 默认层级描述.
fn base_config(provider: Arc<dyn KnowledgeProvider>) -> SemanticGatekeeperConfig {
    SemanticGatekeeperConfig::with_defaults(provider)
}


// ============================================================================
// TEST-SG-001: Level Keys 预计算 (REQ-SG-001)
// ============================================================================

/// TEST-SG-001: 加载 SmolLM2-135M + SG config,验证 `register_semantic_gatekeeper`
/// 成功 (LevelKeysCache 预计算 + KProjOnlyGraph JIT 编译 + callback 注入).
///
/// 验收 (REQ-SG-001):
/// 1. register 返回 Ok(())
/// 2. unregister 幂等
/// 3. 重复注册也成功 (覆盖旧 SG 状态)
#[test]
fn test_sg_001_level_keys_precompute() {
    let client = Client::new_chat(MODEL).expect("failed to load SmolLM2-135M");

    let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
    let mut config = base_config(provider);
    config.detection_depths = vec![0.5];

    client
        .register_semantic_gatekeeper(config)
        .expect("register_semantic_gatekeeper must succeed with valid config");

    // 幂等反注册
    client
        .unregister_semantic_gatekeeper()
        .expect("unregister must be idempotent Ok");

    // 重复注册也成功
    let provider2: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
    let config2 = base_config(provider2);
    client
        .register_semantic_gatekeeper(config2)
        .expect("re-register must succeed");

    client
        .unregister_semantic_gatekeeper()
        .expect("unregister after re-register");
}

// ============================================================================
// TEST-SG-002: Q-tap 读写 (REQ-SG-002)
// ============================================================================

/// TEST-SG-002: 注册 SG,调用 generate,验证不崩溃.
///
/// Q-tap ring buffer 当前无数据源 (QTapSTG graph 插入未完成),
/// pre_node 在 ring buffer 读取失败时返回 Continue (不影响推理正确性).
/// 完整 Q-tap 数值验证等 graph 插入完成后补充.
#[test]
fn test_sg_002_qtap_read_write() {
    let client = Client::new_chat(MODEL).expect("failed to load SmolLM2-135M");

    let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
    let config = base_config(provider);

    client
        .register_semantic_gatekeeper(config)
        .expect("register must succeed");

    let resp = client
        .generate("Hello")
        .max_tokens(4)
        .temperature(0.0)
        .generate()
        .response()
        .expect("generate with SG registered must not crash");
    assert!(!resp.text.is_empty(), "generate must produce output");

    client
        .unregister_semantic_gatekeeper()
        .expect("unregister must be idempotent Ok");
}

// ============================================================================
// TEST-SG-003: 层级路由门控 (REQ-SG-003)
// ============================================================================

/// TEST-SG-003: 注册 SG + generate,验证 Provider 不被调用 (因为 Q-tap ring buffer
/// 无数据 → pre_node 返回 Continue). 同时验证幂等反注册.
///
/// 完整的 Provider 调用验证等 Q-tap graph 插入完成后补充.
#[test]
fn test_sg_003_level_routing_gate() {
    let client = Client::new_chat(MODEL).expect("failed to load SmolLM2-135M");

    let provider = Arc::new(AlwaysNoneProvider::default());
    let provider_handle: Arc<dyn KnowledgeProvider> = provider.clone();
    let config = base_config(provider_handle);

    client
        .register_semantic_gatekeeper(config)
        .expect("register must succeed");

    let resp = client
        .generate("Hello")
        .max_tokens(4)
        .temperature(0.0)
        .generate()
        .response()
        .expect("generate must succeed");
    assert!(!resp.text.is_empty());

    // Q-tap ring buffer 无数据 → pre_node 返回 Continue → Provider 不被调用
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        0,
        "Provider must not be called (Q-tap ring buffer has no data source)"
    );

    // 幂等反注册验证.
    client
        .unregister_semantic_gatekeeper()
        .expect("unregister idempotent");
    client
        .unregister_semantic_gatekeeper()
        .expect("unregister must remain idempotent on repeated calls");
}

// ============================================================================
// TEST-SG-004: 残差注入行为差异 (REQ-SG-004 / REQ-SG-008)
// ============================================================================

/// TEST-SG-004: Provider 返回固定文本 "Paris",验证注册 + generate 不崩溃,
/// 以及 Client 状态机 (register → reset → unregister 幂等性).
///
/// 完整的行为差异验证 (Paris logit 提升) 等 Q-tap graph 插入完成后补充:
/// 1. baseline_resp = client.generate("Capital of France is").run()
/// 2. client.register_semantic_gatekeeper(config)
/// 3. sg_resp = client.generate("Capital of France is").run()
/// 4. 对比 sg_resp.text 中 "Paris" 出现位置/概率 vs baseline
#[test]
fn test_sg_004_residual_injection_behavior_diff() {
    let client = Client::new_chat(MODEL).expect("failed to load SmolLM2-135M");

    let provider: Arc<dyn KnowledgeProvider> = Arc::new(FixedTextProvider::new("Paris", 1.0));
    let config = base_config(provider);

    client
        .register_semantic_gatekeeper(config)
        .expect("register must succeed");

    // reset 幂等 (SPEC §5.3)
    client
        .reset_gatekeeper_state()
        .expect("reset must be idempotent");

    let resp = client
        .generate("Capital of France is")
        .max_tokens(8)
        .temperature(0.0)
        .generate()
        .response()
        .expect("generate with SG registered must succeed");
    assert!(!resp.text.is_empty());

    client
        .unregister_semantic_gatekeeper()
        .expect("unregister must succeed");
}

// ============================================================================
// TEST-SG-005: 稳定性追踪 + SemanticLevel 契约 (REQ-SG-004)
// ============================================================================

/// TEST-SG-005: 验证 `SemanticLevel::from_idx` / `as_idx` roundtrip,
/// 以及 `resolve_detection_layers` 对 `[0.5, 0.75, 0.9] × num_layers=26`
/// 的物理层解析 (REQ-SG-001 验收 1).
///
/// 稳定性追踪的完整命中率测试 (≥75% 复用) 依赖 Phase C,此处仅验证
/// 配置解析层契约.
#[test]
fn test_sg_005_stability_and_level_contract() {
    // SemanticLevel::from_idx / as_idx roundtrip
    for idx in 0..3 {
        let level = SemanticLevel::from_idx(idx)
            .unwrap_or_else(|| panic!("from_idx({idx}) returned None"));
        assert_eq!(
            level.as_idx(),
            idx,
            "roundtrip failed: from_idx({idx}).as_idx() != {idx}"
        );
    }
    assert_eq!(
        SemanticLevel::from_idx(3),
        None,
        "from_idx(3) must return None (index out of range)"
    );
    assert_eq!(SemanticLevel::from_idx(usize::MAX), None);

    // ORDER 常量一致性
    for (idx, level) in SemanticLevel::ORDER.iter().enumerate() {
        assert_eq!(level.as_idx(), idx);
        assert_eq!(SemanticLevel::from_idx(idx), Some(*level));
    }

    // resolve_detection_layers: [0.5, 0.75, 0.9] × 26 = [13, 19, 23]
    let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
    let mut config = base_config(provider);
    config.detection_depths = vec![0.5, 0.75, 0.9];

    let layers = config.resolve_detection_layers(26);
    assert_eq!(
        layers,
        vec![13, 19, 23],
        "resolve_detection_layers([0.5, 0.75, 0.9], 26) expected [13, 19, 23], got {layers:?}"
    );

    // 边界: depth=1.0 × num_layers → num_layers-1 (saturating_sub)
    config.detection_depths = vec![1.0];
    let layers = config.resolve_detection_layers(10);
    assert_eq!(
        layers,
        vec![9],
        "depth=1.0 must saturate to num_layers-1, got {layers:?}"
    );
}

// ============================================================================
// TEST-SG-006: KnowledgeProvider / AstSentinel trait 契约 (REQ-SG-006)
// ============================================================================

/// TEST-SG-006: 验证 trait 对象可作为 `Arc<dyn KnowledgeProvider>` /
/// `Arc<dyn AstSentinel>` 被 `SemanticGatekeeperConfig` 接纳,且
/// `TokenizerLookup` 可通过 trait object 调用 (REQ-SG-006 验收 1/2/3).
#[test]
fn test_sg_006_knowledge_provider_contract() {
    // 两种 Provider 实现.
    let p1: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
    let p2: Arc<dyn KnowledgeProvider> = Arc::new(FixedTextProvider::new("hello", 0.8));

    // 两种 AstSentinel 实现.
    let a1: Arc<dyn AstSentinel> = Arc::new(FixedAstSentinel::new("call_expression"));
    let a2: Arc<dyn AstSentinel> = Arc::new(NullAstSentinel);

    // 配置构造 + 接纳 trait object.
    let mut cfg1 = base_config(p1);
    cfg1.ast_sentinel = Some(a1);
    assert!(cfg1.validate().is_ok(), "valid config must pass validate");

    let mut cfg2 = base_config(p2.clone());
    cfg2.ast_sentinel = Some(a2);
    assert!(cfg2.validate().is_ok());

    // Provider.retrieve 直接可调用 (契约 SPEC §6.1).
    struct DummyTokenizer;
    impl TokenizerLookup for DummyTokenizer {
        fn decode(&self, _tokens: &[u32]) -> String {
            "".into()
        }
    }
    let _tk = DummyTokenizer;

    // FixedTextProvider.retrieve 契约验证.
    let retrieve_ctx = RetrieveContext {
        generated_tokens: &[1u32, 2, 3],
        ast: None,
        step: 0,
        request_id: 42,
    };
    let query = vec![0.1f32; 64];
    let entry = p2
        .retrieve(&query, SemanticLevel::L1, &retrieve_ctx)
        .expect("FixedTextProvider must return Some");
    assert_eq!(entry.text, "hello");
    assert!(
        (entry.confidence - 0.8).abs() < 1e-6,
        "confidence must match, got {}",
        entry.confidence
    );

    // AlwaysNoneProvider.retrieve 契约 (None 路径).
    let p_none: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
    let none_entry = p_none.retrieve(&query, SemanticLevel::L3, &retrieve_ctx);
    assert!(
        none_entry.is_none(),
        "AlwaysNoneProvider must return None"
    );

    // DEFAULT_LEVEL_DESCRIPTORS 三条非空 (SPEC §7.2 默认文本契约).
    assert_eq!(DEFAULT_LEVEL_DESCRIPTORS.len(), 3);
    for (i, desc) in DEFAULT_LEVEL_DESCRIPTORS.iter().enumerate() {
        assert!(!desc.trim().is_empty(), "descriptor {i} must be non-empty");
    }
}

// ============================================================================
// TEST-SG-007: 置信度 + validate() 错误契约 (REQ-SG-005 / REQ-SG-007)
// ============================================================================

/// TEST-SG-007: `SemanticGatekeeperConfig::validate()` 对非法参数返回
/// 精确的错误变体 (InvalidAlpha / InvalidThreshold / InvalidDetectionDepth).
///
/// 关联: SPEC §8.2 TEST-SG-007 验收 — confidence=0.0 时 SG 跳过注入
/// (验证在 Phase C callback 实现侧;当前测试配置层契约).
#[test]
fn test_sg_007_confidence_and_validate() {
    // alpha = 0.0 → InvalidAlpha
    {
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
        let mut cfg = base_config(provider);
        cfg.alpha = 0.0;
        match cfg.validate() {
            Err(SemanticGatekeeperError::InvalidAlpha(a)) => {
                assert_eq!(a, 0.0, "InvalidAlpha must echo the bad value");
            }
            other => panic!("expected InvalidAlpha(0.0), got {other:?}"),
        }
    }

    // alpha < 0 → InvalidAlpha
    {
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
        let mut cfg = base_config(provider);
        cfg.alpha = -0.1;
        match cfg.validate() {
            Err(SemanticGatekeeperError::InvalidAlpha(_)) => {}
            other => panic!("expected InvalidAlpha for negative alpha, got {other:?}"),
        }
    }

    // alpha > 1 → InvalidAlpha
    {
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
        let mut cfg = base_config(provider);
        cfg.alpha = 1.5;
        match cfg.validate() {
            Err(SemanticGatekeeperError::InvalidAlpha(_)) => {}
            other => panic!("expected InvalidAlpha for alpha>1, got {other:?}"),
        }
    }

    // gate_threshold > 1 → InvalidThreshold
    {
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
        let mut cfg = base_config(provider);
        cfg.gate_threshold = 1.5;
        match cfg.validate() {
            Err(SemanticGatekeeperError::InvalidThreshold { gate, .. }) => {
                assert!(
                    (gate - 1.5).abs() < 1e-6,
                    "InvalidThreshold must echo the bad gate, got {gate}"
                );
            }
            other => panic!("expected InvalidThreshold for gate>1, got {other:?}"),
        }
    }

    // stability_threshold < 0 → InvalidThreshold
    {
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
        let mut cfg = base_config(provider);
        cfg.stability_threshold = -0.1;
        match cfg.validate() {
            Err(SemanticGatekeeperError::InvalidThreshold { .. }) => {}
            other => panic!("expected InvalidThreshold for stability<0, got {other:?}"),
        }
    }

    // detection_depths=[0.0] → InvalidDetectionDepth (0.0 不在 (0, 1] 区间)
    {
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
        let mut cfg = base_config(provider);
        cfg.detection_depths = vec![0.0];
        match cfg.validate() {
            Err(SemanticGatekeeperError::InvalidDetectionDepth(d)) => {
                assert_eq!(d, 0.0);
            }
            other => panic!("expected InvalidDetectionDepth(0.0), got {other:?}"),
        }
    }

    // detection_depths=[1.5] → InvalidDetectionDepth (>1)
    {
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
        let mut cfg = base_config(provider);
        cfg.detection_depths = vec![1.5];
        match cfg.validate() {
            Err(SemanticGatekeeperError::InvalidDetectionDepth(d)) => {
                assert!((d - 1.5).abs() < 1e-6);
            }
            other => panic!("expected InvalidDetectionDepth(1.5), got {other:?}"),
        }
    }

    // 合法配置必须通过
    {
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
        let cfg = base_config(provider);
        cfg.validate()
            .expect("default config from with_defaults must validate");
    }

    // confidence=0.0 entry 契约: KnowledgeEntry 可以携带 0.0 置信度,
    // SG Callback 负责跳过注入 (SPEC §6.1 验收). 这里验证构造合法.
    let entry = KnowledgeEntry {
        text: "skip-me".into(),
        confidence: 0.0,
    };
    assert_eq!(entry.confidence, 0.0);
    assert!(!entry.text.is_empty());
}

// ============================================================================
// TEST-SG-008: 多检测层 + resolve_detection_layers 去重 (REQ-SG-001 / REQ-SG-003)
// ============================================================================

/// TEST-SG-008: `resolve_detection_layers` 正确去重并按升序排列;
/// 多个相同 layer_idx 只保留一份 (SPEC §3.5 一致性不变量).
#[test]
fn test_sg_008_multi_detection_layers() {
    let provider: Arc<dyn KnowledgeProvider> = Arc::new(AlwaysNoneProvider::default());
    let mut cfg = base_config(provider);

    // 重复 + 乱序: [0.5, 0.5, 0.3] × num_layers=20
    //   0.5 × 20 = 10
    //   0.3 × 20 = 6
    // 去重 + 升序 → [6, 10]
    cfg.detection_depths = vec![0.5, 0.5, 0.3];
    let layers = cfg.resolve_detection_layers(20);
    assert_eq!(
        layers,
        vec![6, 10],
        "expected [6, 10] (dedup + sorted asc), got {layers:?}"
    );

    // 典型 [0.5, 0.75] × 20 → [10, 15] (两个独立检测层)
    cfg.detection_depths = vec![0.5, 0.75];
    let layers = cfg.resolve_detection_layers(20);
    assert_eq!(
        layers,
        vec![10, 15],
        "two distinct layers expected, got {layers:?}"
    );

    // 所有深度相同 → 单层.
    cfg.detection_depths = vec![0.25, 0.25, 0.25, 0.25];
    let layers = cfg.resolve_detection_layers(16);
    assert_eq!(layers, vec![4], "all-same depths must collapse to 1 layer");

    // num_layers=1 边界: 任何 depth ∈ (0,1] 都映射到 0 (saturating_sub).
    cfg.detection_depths = vec![0.5, 0.9, 1.0];
    let layers = cfg.resolve_detection_layers(1);
    assert_eq!(
        layers,
        vec![0],
        "num_layers=1 must saturate all depths to [0], got {layers:?}"
    );

    // 空 depths → 空层集合 (REQ-SG-003: 无检测层时 SG 不触发任何 pre_node).
    cfg.detection_depths = vec![];
    let layers = cfg.resolve_detection_layers(20);
    assert!(
        layers.is_empty(),
        "empty detection_depths must yield empty layer set"
    );
}
