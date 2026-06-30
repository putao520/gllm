//! Semantic Gatekeeper — 隐藏状态驱动的免训练知识调度与注入 SDK.
//!
//! 实现 `SPEC/SEMANTIC-GATEKEEPER.md` 定义的运行时知识注入架构:
//!
//! - 模型加载期预计算 3 个层级键 `K_L1/K_L2/K_L3` (描述文本 → tokenizer →
//!   冻结 embed → 检测层 k_proj 投影)
//! - 推理期每到检测层,JIT SgDetect/SgInject Op 从
//!   `FusedAttentionLayer` 的 Q-tap ring buffer 读取 Q 向量,与层级键做
//!   cosine 相似度打分路由 L1/L2/L3,触发 `KnowledgeProvider` 检索知识文本,
//!   用主模型冻结 embed 编码成 `v_knowledge`,以残差相加方式注入.
//!
//! ## 关联文档
//!
//! - `SPEC/SEMANTIC-GATEKEEPER.md` — 技术协议 SSOT
//! - `SPEC/04-API-DESIGN.md §7-§8` — 用户 API + 内部架构
//! - `SPEC/05-OPTIMIZATIONS.md §2.9` — Callback 集成
//! - `SPEC/08-EXECUTOR.md §4.2.1` — FusedAttentionLayer Q-Tap 扩展
//! - `SPEC/01-REQUIREMENTS.md §12` — REQ-SG-001..008

use std::sync::Arc;

use crate::scheduler::types::RequestId;

pub mod active_state;
pub mod callback;
pub mod decode;
pub mod level_keys;
pub mod ring_buffer;
pub mod sg_shared_memory;
pub mod small_graph;

pub use active_state::ActiveState;
pub use callback::SemanticGatekeeperCallback;
pub use level_keys::{precompute as precompute_level_keys, LevelKeysCache, LevelKeysError};
pub use ring_buffer::{GatekeeperRingBuffer, QTapReadError};
pub use sg_shared_memory::SgSharedMemory;
pub use small_graph::{EmbedLookupOnlyGraph, KProjOnlyGraph};

// ============================================================================
// SemanticLevel
// ============================================================================

/// 三级语义层级,对应 Level Keys 的三个描述文本.
///
/// 按 SPEC §7.2 定义:
/// - L1: 符号签名、类型成员
/// - L2: 接口约束、业务规则
/// - L3: 架构分层、模块职责
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticLevel {
    L1,
    L2,
    L3,
}

impl SemanticLevel {
    /// 三个层级的稳定顺序,用于数组索引.
    pub const ORDER: [SemanticLevel; 3] = [Self::L1, Self::L2, Self::L3];

    /// 将数组索引转换为 `SemanticLevel`.
    ///
    /// 返回 `None` 若索引超出 `[0, 2]`.
    pub fn from_idx(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::L1),
            1 => Some(Self::L2),
            2 => Some(Self::L3),
            _ => None,
        }
    }

    /// 当前层级在 `ORDER` 中的索引.
    pub fn as_idx(self) -> usize {
        match self {
            Self::L1 => 0,
            Self::L2 => 1,
            Self::L3 => 2,
        }
    }
}

/// 默认层级描述文本 (SPEC §7.2).
///
/// 用户可通过 `SemanticGatekeeperConfig.level_descriptors` 覆写.
pub const DEFAULT_LEVEL_DESCRIPTORS: [&str; 3] = [
    "struct fields and method signatures",
    "validation rules and invariants",
    "module dependencies and design patterns",
];

// ============================================================================
// KnowledgeEntry + RetrieveContext + AstContext
// ============================================================================

/// `KnowledgeProvider::retrieve` 返回的知识条目.
#[derive(Debug, Clone)]
pub struct KnowledgeEntry {
    /// 将被主模型 tokenizer + 冻结 embed 层编码的文本.
    ///
    /// 保证 `v_knowledge` 与 hidden_state 处于同一语义空间 (SPEC §8.6).
    pub text: String,

    /// 置信度 ∈ [0.0, 1.0],动态调节 `α_effective = α × confidence`.
    ///
    /// 0.0 表示"知道但不建议注入",SG 将跳过残差加法 (REQ-SG-005 验收 3).
    pub confidence: f32,
}

/// 传递给 `KnowledgeProvider::retrieve` 的检索上下文.
pub struct RetrieveContext<'a> {
    /// 最近生成的 token 序列.
    pub generated_tokens: &'a [u32],
    /// AST 哨兵返回的语法上下文,可能为 `None`.
    pub ast: Option<AstContext<'a>>,
    /// 当前 decode step 号.
    pub step: u64,
    /// 请求 ID.
    pub request_id: RequestId,
}

/// AST 哨兵返回的语法上下文 (SPEC §7.4).
#[derive(Debug, Clone, Copy)]
pub struct AstContext<'a> {
    /// Tree-sitter 节点 kind (如 `"member_expression"` / `"call_expression"`).
    pub node_kind: &'a str,
    /// 光标所在的文本范围.
    pub cursor_line: u32,
    pub cursor_column: u32,
    /// 光标前已输入的字符串 (用于补全匹配).
    pub prefix: &'a str,
}

// ============================================================================
// KnowledgeProvider + AstSentinel traits
// ============================================================================

/// 知识源的多态抽象. 用户实现此 trait 挂接本地 / 远程知识库.
///
/// SG 内核在检测层 JIT SgDetect Op 中调用 `retrieve`,返回 `None` 时 SG 不注入,
/// hidden_state 保持原样 (NO_SILENT_FALLBACK 合规 — 显式 None 而非错误默认).
pub trait KnowledgeProvider: Send + Sync {
    fn retrieve(
        &self,
        query: &[f32],
        level: SemanticLevel,
        ctx: &RetrieveContext<'_>,
    ) -> Option<KnowledgeEntry>;
}

/// 可选的语法哨兵,用于 AST 节点驱动的强制刷新 (SPEC §5).
///
/// SG 内核在每个检测层 JIT SgDetect Op 前调用 `current_context`,若返回的
/// `node_kind` 与 `ActiveState.ast_node_kind` 不一致则强制 FullCompute.
pub trait AstSentinel: Send + Sync {
    fn current_context<'a>(
        &self,
        generated_tokens: &'a [u32],
        tokenizer: &dyn TokenizerLookup,
    ) -> Option<AstContext<'a>>;
}

/// 最小化 tokenizer 接口,供 `AstSentinel` 将 token id 序列反解为文本.
///
/// 独立于 `TokenizerHandle` 具体实现,允许用户在不引入 gllm tokenizer 依赖
/// 的前提下实现 `AstSentinel`.
pub trait TokenizerLookup: Send + Sync {
    /// 将 token id 序列反解为字符串. 未知 id 应以跳过或占位符处理.
    fn decode(&self, tokens: &[u32]) -> String;
}

// ============================================================================
// TokenizerEncoder — Level Keys 预计算 & 运行时知识文本编码路径
// ============================================================================

/// 文本 → token id 编码接口.
///
/// Level Keys 预计算 (`level_keys::precompute`) 和运行时知识文本编码均通过
/// 此 trait 调用主模型 tokenizer; 以 trait 形式解耦具体 `TokenizerHandle`
/// 实现,便于单元测试注入 mock,亦允许 SG Provider 端使用独立 tokenizer.
pub trait TokenizerEncoder: Send + Sync {
    /// 将文本编码为 token id 序列.
    ///
    /// `add_special_tokens` 语义由实现者决定: 对主模型 embed 输入通常应为 `false`
    /// (避免 BOS/EOS 污染 mean-pool 结果).
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerEncodeError>;
}

/// `TokenizerEncoder::encode` 错误.
#[derive(Debug, thiserror::Error, Clone)]
pub enum TokenizerEncodeError {
    #[error("empty text not allowed")]
    EmptyText,
    #[error("tokenizer backend error: {0}")]
    Backend(String),
    #[error("token id {token} exceeds vocab size {vocab_size}")]
    TokenOutOfRange { token: u32, vocab_size: usize },
}

// ============================================================================
// SemanticGatekeeperConfig
// ============================================================================

/// SG 注册配置,对应 SPEC §7.1.
pub struct SemanticGatekeeperConfig {
    /// 三个层级的描述文本. 用户可覆写默认值 (SPEC §7.2).
    pub level_descriptors: [String; 3],

    /// 检测层相对深度列表 (如 `[0.5, 0.75, 0.9]`).
    ///
    /// 每个深度 d 映射为 `layer_idx = floor(d × num_layers)`.
    /// 相同 layer_idx 会去重.
    pub detection_depths: Vec<f32>,

    /// 门控阈值 τ. 最高 cosine(Q, K_Lx) < τ 时 SG 不注入.
    ///
    /// 推荐范围 [0.2, 0.5]. 默认 0.35.
    pub gate_threshold: f32,

    /// 稳定性阈值. `cosine(hidden[-1], anchor)` > 该值时复用
    /// `active_state.v_knowledge`,跳过 LSP 检索 (SPEC §5).
    ///
    /// 推荐范围 [0.90, 0.98]. 默认 0.95.
    pub stability_threshold: f32,

    /// 残差相加强度 α. 实际 `α_effective = α × entry.confidence`.
    ///
    /// 推荐范围 [0.10, 0.30]. 默认 0.15.
    pub alpha: f32,

    /// 用户实现的知识源.
    pub knowledge_provider: Arc<dyn KnowledgeProvider>,

    /// 可选 AST 哨兵 (语法驱动的强制刷新).
    pub ast_sentinel: Option<Arc<dyn AstSentinel>>,
}

impl SemanticGatekeeperConfig {
    /// 创建配置,使用默认层级描述文本 + 默认阈值.
    pub fn with_defaults(knowledge_provider: Arc<dyn KnowledgeProvider>) -> Self {
        Self {
            level_descriptors: [
                DEFAULT_LEVEL_DESCRIPTORS[0].to_string(),
                DEFAULT_LEVEL_DESCRIPTORS[1].to_string(),
                DEFAULT_LEVEL_DESCRIPTORS[2].to_string(),
            ],
            detection_depths: vec![0.5, 0.75, 0.9],
            gate_threshold: 0.35,
            stability_threshold: 0.95,
            alpha: 0.15,
            knowledge_provider,
            ast_sentinel: None,
        }
    }
}

// ============================================================================
// Error types
// ============================================================================

/// SG 相关错误 (注册 / 预计算 / 运行时).
#[derive(Debug, thiserror::Error)]
pub enum SemanticGatekeeperError {
    #[error("invalid detection depth {0}: must be in (0.0, 1.0]")]
    InvalidDetectionDepth(f32),
    #[error("invalid threshold: gate={gate}, stability={stability} (both must be in [0, 1])")]
    InvalidThreshold { gate: f32, stability: f32 },
    #[error("invalid alpha {0}: must be in (0.0, 1.0]")]
    InvalidAlpha(f32),
    #[error("level descriptors must be non-empty")]
    EmptyLevelDescriptor,
    #[error("level keys precomputation failed: {0}")]
    PrecomputeFailed(String),
    #[error("ring buffer access failed: {0}")]
    RingBuffer(#[from] QTapReadError),
    #[error("knowledge provider error: {0}")]
    Provider(String),
    #[error("small graph compilation failed: {0}")]
    SmallGraph(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("gatekeeper not registered")]
    NotRegistered,
}

// ============================================================================
// Validation
// ============================================================================

impl SemanticGatekeeperConfig {
    /// 校验配置字段合法性. `Client::register_semantic_gatekeeper` 注册前调用.
    pub fn validate(&self) -> Result<(), SemanticGatekeeperError> {
        if !(0.0..=1.0).contains(&self.gate_threshold)
            || !(0.0..=1.0).contains(&self.stability_threshold)
        {
            return Err(SemanticGatekeeperError::InvalidThreshold {
                gate: self.gate_threshold,
                stability: self.stability_threshold,
            });
        }
        if !(0.0..=1.0).contains(&self.alpha) || self.alpha == 0.0 {
            return Err(SemanticGatekeeperError::InvalidAlpha(self.alpha));
        }
        for &d in &self.detection_depths {
            if !(0.0 < d && d <= 1.0) {
                return Err(SemanticGatekeeperError::InvalidDetectionDepth(d));
            }
        }
        for desc in &self.level_descriptors {
            if desc.trim().is_empty() {
                return Err(SemanticGatekeeperError::EmptyLevelDescriptor);
            }
        }
        Ok(())
    }

    /// 将相对深度解析为物理层索引集合 (去重 + 按升序排列).
    pub fn resolve_detection_layers(&self, num_layers: usize) -> Vec<usize> {
        use std::collections::BTreeSet;
        let mut set = BTreeSet::new();
        for &d in &self.detection_depths {
            let layer = ((d * num_layers as f32).floor() as usize).min(num_layers.saturating_sub(1));
            set.insert(layer);
        }
        set.into_iter().collect()
    }
}

// ============================================================================
// ClientTokenizer — bridge from gllm TokenizerHandle to SG traits
// ============================================================================

/// Wraps `crate::tokenizer::TokenizerHandle` to implement both
/// `TokenizerEncoder` (for level key precomputation) and `TokenizerLookup`
/// (for SG callback AST sentinel text decoding).
pub struct ClientTokenizer(pub crate::tokenizer::TokenizerHandle);

impl TokenizerEncoder for ClientTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerEncodeError> {
        if text.trim().is_empty() {
            return Err(TokenizerEncodeError::EmptyText);
        }
        self.0.encode(text, false).map_err(|e| {
            TokenizerEncodeError::Backend(format!("{e}"))
        })
    }
}

impl TokenizerLookup for ClientTokenizer {
    fn decode(&self, tokens: &[u32]) -> String {
        self.0.decode(tokens, false).unwrap_or_default()
    }
}

/// Fallback tokenizer lookup that returns empty string.
/// Used when the main model tokenizer is not available for callback's AST sentinel.
pub struct NoOpTokenizerLookup;

impl TokenizerLookup for NoOpTokenizerLookup {
    fn decode(&self, _tokens: &[u32]) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SemanticLevel ──

    #[test]
    fn semantic_level_from_idx_valid() {
        assert_eq!(SemanticLevel::from_idx(0), Some(SemanticLevel::L1));
        assert_eq!(SemanticLevel::from_idx(1), Some(SemanticLevel::L2));
        assert_eq!(SemanticLevel::from_idx(2), Some(SemanticLevel::L3));
    }

    #[test]
    fn semantic_level_from_idx_out_of_range() {
        assert_eq!(SemanticLevel::from_idx(3), None);
        assert_eq!(SemanticLevel::from_idx(100), None);
    }

    #[test]
    fn semantic_level_as_idx_roundtrip() {
        for level in SemanticLevel::ORDER {
            assert_eq!(SemanticLevel::from_idx(level.as_idx()), Some(level));
        }
    }

    #[test]
    fn semantic_level_order_has_three() {
        assert_eq!(SemanticLevel::ORDER.len(), 3);
    }

    // ── SemanticGatekeeperConfig::validate ──

    struct DummyProvider;
    impl KnowledgeProvider for DummyProvider {
        fn retrieve(&self, _: &[f32], _: SemanticLevel, _: &RetrieveContext<'_>) -> Option<KnowledgeEntry> {
            None
        }
    }

    fn valid_config() -> SemanticGatekeeperConfig {
        SemanticGatekeeperConfig::with_defaults(Arc::new(DummyProvider))
    }

    #[test]
    fn validate_defaults_ok() {
        assert!(valid_config().validate().is_ok());
    }

    #[test]
    fn validate_rejects_gate_above_one() {
        let mut c = valid_config();
        c.gate_threshold = 1.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_gate() {
        let mut c = valid_config();
        c.gate_threshold = -0.1;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_alpha() {
        let mut c = valid_config();
        c.alpha = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_alpha_above_one() {
        let mut c = valid_config();
        c.alpha = 1.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_detection_depth() {
        let mut c = valid_config();
        c.detection_depths = vec![0.0];
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_detection_depth_above_one() {
        let mut c = valid_config();
        c.detection_depths = vec![1.1];
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty_descriptor() {
        let mut c = valid_config();
        c.level_descriptors[1] = "   ".to_string();
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_boundary_values_ok() {
        let mut c = valid_config();
        c.gate_threshold = 0.0;
        c.stability_threshold = 1.0;
        c.alpha = 1.0;
        c.detection_depths = vec![0.001, 1.0];
        assert!(c.validate().is_ok());
    }

    // ── resolve_detection_layers ──

    #[test]
    fn resolve_layers_basic() {
        let c = valid_config(); // depths [0.5, 0.75, 0.9]
        let layers = c.resolve_detection_layers(32);
        assert_eq!(layers.len(), 3);
        assert!(layers.windows(2).all(|w| w[0] < w[1])); // sorted ascending
    }

    #[test]
    fn resolve_layers_dedup() {
        let mut c = valid_config();
        c.detection_depths = vec![0.5, 0.5, 0.5];
        let layers = c.resolve_detection_layers(10);
        assert_eq!(layers.len(), 1);
    }

    #[test]
    fn resolve_layers_depth_one_is_last_layer() {
        let mut c = valid_config();
        c.detection_depths = vec![1.0];
        let layers = c.resolve_detection_layers(32);
        assert_eq!(layers, vec![31]);
    }

    #[test]
    fn resolve_layers_single_layer_model() {
        let mut c = valid_config();
        c.detection_depths = vec![0.5];
        let layers = c.resolve_detection_layers(1);
        assert_eq!(layers, vec![0]);
    }

    // ── NoOpTokenizerLookup ──

    #[test]
    fn no_op_tokenizer_returns_empty() {
        let t = NoOpTokenizerLookup;
        assert_eq!(t.decode(&[1, 2, 3]), "");
        assert_eq!(t.decode(&[]), "");
    }

    // ── DEFAULT_LEVEL_DESCRIPTORS ──

    #[test]
    fn default_descriptors_non_empty() {
        for desc in &DEFAULT_LEVEL_DESCRIPTORS {
            assert!(!desc.trim().is_empty());
        }
    }

    // ── TokenizerEncodeError ──

    #[test]
    fn tokenizer_encode_error_display() {
        let e = TokenizerEncodeError::EmptyText;
        assert!(e.to_string().contains("empty text"));
        let e = TokenizerEncodeError::TokenOutOfRange { token: 50000, vocab_size: 32000 };
        assert!(e.to_string().contains("50000"));
    }

    // ── KnowledgeEntry ──

    #[test]
    fn knowledge_entry_fields() {
        let e = KnowledgeEntry {
            text: "struct Foo".to_string(),
            confidence: 0.8,
        };
        assert_eq!(e.text, "struct Foo");
        assert!((e.confidence - 0.8).abs() < 1e-6);
    }

    // ════════════════════════════════════════════════════════════════════════
    // New tests — pure logic coverage for untested types, traits, and edges
    // ════════════════════════════════════════════════════════════════════════

    // ── SemanticLevel trait tests ──

    #[test]
    fn semantic_level_debug_format() {
        assert_eq!(format!("{:?}", SemanticLevel::L1), "L1");
        assert_eq!(format!("{:?}", SemanticLevel::L2), "L2");
        assert_eq!(format!("{:?}", SemanticLevel::L3), "L3");
    }

    #[test]
    fn semantic_level_clone_preserves_variant() {
        let l = SemanticLevel::L2;
        let cloned = l.clone();
        assert_eq!(l, cloned);
    }

    #[test]
    fn semantic_level_copy_semantics() {
        let l = SemanticLevel::L3;
        let copied = l; // Copy, not move
        assert_eq!(l, copied); // l still usable after "move"
    }

    #[test]
    fn semantic_level_hash_distinguishes_variants() {
        use std::collections::HashSet;
        let set: HashSet<SemanticLevel> = [SemanticLevel::L1, SemanticLevel::L2, SemanticLevel::L3].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn semantic_level_order_is_ascending() {
        assert_eq!(SemanticLevel::ORDER[0].as_idx(), 0);
        assert_eq!(SemanticLevel::ORDER[1].as_idx(), 1);
        assert_eq!(SemanticLevel::ORDER[2].as_idx(), 2);
    }

    // ── SemanticGatekeeperError Display / Debug / std::error ──

    #[test]
    fn sg_error_display_all_variants() {
        let e = SemanticGatekeeperError::InvalidDetectionDepth(1.5);
        assert!(e.to_string().contains("1.5"), "depth value missing");

        let e = SemanticGatekeeperError::InvalidThreshold { gate: 2.0, stability: -0.5 };
        let msg = e.to_string();
        assert!(msg.contains("gate=2"), "gate missing");
        assert!(msg.contains("stability=-0.5"), "stability missing");

        let e = SemanticGatekeeperError::InvalidAlpha(1.2);
        assert!(e.to_string().contains("1.2"), "alpha value missing");

        let e = SemanticGatekeeperError::EmptyLevelDescriptor;
        assert!(e.to_string().contains("non-empty"));

        let e = SemanticGatekeeperError::PrecomputeFailed("detail".into());
        assert!(e.to_string().contains("detail"));

        let e = SemanticGatekeeperError::Provider("prov err".into());
        assert!(e.to_string().contains("prov err"));

        let e = SemanticGatekeeperError::Tokenizer("tok err".into());
        assert!(e.to_string().contains("tok err"));

        let e = SemanticGatekeeperError::NotRegistered;
        assert!(e.to_string().contains("not registered"));
    }

    #[test]
    fn sg_error_debug_format() {
        let e = SemanticGatekeeperError::InvalidAlpha(0.0);
        let debug = format!("{:?}", e);
        assert!(debug.contains("InvalidAlpha"), "Debug should contain variant name: {debug}");

        let e = SemanticGatekeeperError::NotRegistered;
        assert!(format!("{:?}", e).contains("NotRegistered"));
    }

    #[test]
    fn sg_error_implements_std_error() {
        use std::error::Error;
        let e = SemanticGatekeeperError::NotRegistered;
        let _: &dyn Error = &e; // Compiles = impl verified
    }

    #[test]
    fn sg_error_ring_buffer_from_conversion() {
        let qtap_err = QTapReadError::Uninitialized;
        let sg_err: SemanticGatekeeperError = qtap_err.into();
        assert!(matches!(sg_err, SemanticGatekeeperError::RingBuffer(QTapReadError::Uninitialized)));
        assert!(sg_err.to_string().contains("not initialized"));
    }

    // ── KnowledgeEntry trait tests ──

    #[test]
    fn knowledge_entry_debug() {
        let e = KnowledgeEntry {
            text: "hello".to_string(),
            confidence: 0.5,
        };
        let debug = format!("{:?}", e);
        assert!(debug.contains("hello"), "Debug should contain text: {debug}");
        assert!(debug.contains("0.5"), "Debug should contain confidence: {debug}");
    }

    #[test]
    fn knowledge_entry_clone() {
        let e = KnowledgeEntry {
            text: "abc".to_string(),
            confidence: 0.9,
        };
        let cloned = e.clone();
        assert_eq!(e.text, cloned.text);
        assert_eq!(e.confidence, cloned.confidence);
    }

    #[test]
    fn knowledge_entry_zero_confidence() {
        let e = KnowledgeEntry {
            text: String::new(),
            confidence: 0.0,
        };
        assert_eq!(e.text, "");
        assert_eq!(e.confidence, 0.0);
    }

    // ── AstContext construction and traits ──

    #[test]
    fn ast_context_construction_and_fields() {
        let ctx = AstContext {
            node_kind: "call_expression",
            cursor_line: 10,
            cursor_column: 25,
            prefix: "foo.",
        };
        assert_eq!(ctx.node_kind, "call_expression");
        assert_eq!(ctx.cursor_line, 10);
        assert_eq!(ctx.cursor_column, 25);
        assert_eq!(ctx.prefix, "foo.");
    }

    #[test]
    fn ast_context_debug() {
        let ctx = AstContext {
            node_kind: "member_expression",
            cursor_line: 5,
            cursor_column: 12,
            prefix: "bar",
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("member_expression"), "Debug missing node_kind: {debug}");
        assert!(debug.contains("5"), "Debug missing cursor_line: {debug}");
    }

    #[test]
    fn ast_context_clone() {
        let ctx = AstContext {
            node_kind: "string",
            cursor_line: 1,
            cursor_column: 1,
            prefix: "x",
        };
        let cloned = ctx.clone();
        assert_eq!(ctx.node_kind, cloned.node_kind);
        assert_eq!(ctx.cursor_line, cloned.cursor_line);
        assert_eq!(ctx.cursor_column, cloned.cursor_column);
        assert_eq!(ctx.prefix, cloned.prefix);
    }

    #[test]
    fn ast_context_copy_semantics() {
        let ctx = AstContext {
            node_kind: "id",
            cursor_line: 0,
            cursor_column: 0,
            prefix: "",
        };
        let copied = ctx; // Copy, not move
        assert_eq!(ctx.node_kind, copied.node_kind); // ctx still usable
    }

    // ── RetrieveContext construction ──

    #[test]
    fn retrieve_context_construction() {
        let tokens = [1u32, 2, 3];
        let ast = AstContext {
            node_kind: "fn_call",
            cursor_line: 3,
            cursor_column: 7,
            prefix: "std::",
        };
        let ctx = RetrieveContext {
            generated_tokens: &tokens,
            ast: Some(ast),
            step: 42,
            request_id: 99,
        };
        assert_eq!(ctx.generated_tokens.len(), 3);
        assert!(ctx.ast.is_some());
        assert_eq!(ctx.step, 42);
        assert_eq!(ctx.request_id, 99);
    }

    #[test]
    fn retrieve_context_no_ast() {
        let tokens: [u32; 0] = [];
        let ctx = RetrieveContext {
            generated_tokens: &tokens,
            ast: None,
            step: 0,
            request_id: 1,
        };
        assert!(ctx.generated_tokens.is_empty());
        assert!(ctx.ast.is_none());
    }

    // ── SemanticGatekeeperConfig with_defaults field values ──

    #[test]
    fn with_defaults_sets_all_fields() {
        let c = SemanticGatekeeperConfig::with_defaults(Arc::new(DummyProvider));
        assert_eq!(c.level_descriptors[0], DEFAULT_LEVEL_DESCRIPTORS[0]);
        assert_eq!(c.level_descriptors[1], DEFAULT_LEVEL_DESCRIPTORS[1]);
        assert_eq!(c.level_descriptors[2], DEFAULT_LEVEL_DESCRIPTORS[2]);
        assert_eq!(c.detection_depths, vec![0.5, 0.75, 0.9]);
        assert!((c.gate_threshold - 0.35).abs() < 1e-6);
        assert!((c.stability_threshold - 0.95).abs() < 1e-6);
        assert!((c.alpha - 0.15).abs() < 1e-6);
        assert!(c.ast_sentinel.is_none());
    }

    #[test]
    fn with_defaults_custom_descriptors_override() {
        let mut c = SemanticGatekeeperConfig::with_defaults(Arc::new(DummyProvider));
        c.level_descriptors = [
            "custom L1".to_string(),
            "custom L2".to_string(),
            "custom L3".to_string(),
        ];
        assert_eq!(c.level_descriptors[0], "custom L1");
        assert!(c.validate().is_ok());
    }

    // ── Validation edge cases ──

    #[test]
    fn validate_rejects_negative_stability() {
        let mut c = valid_config();
        c.stability_threshold = -0.1;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_stability_above_one() {
        let mut c = valid_config();
        c.stability_threshold = 1.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_alpha() {
        let mut c = valid_config();
        c.alpha = -0.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_nan_gate_threshold() {
        let mut c = valid_config();
        c.gate_threshold = f32::NAN;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_nan_alpha() {
        let mut c = valid_config();
        c.alpha = f32::NAN;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_infinity_detection_depth() {
        let mut c = valid_config();
        c.detection_depths = vec![f32::INFINITY];
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_detection_depth() {
        let mut c = valid_config();
        c.detection_depths = vec![-0.1];
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty_descriptor_string() {
        let mut c = valid_config();
        c.level_descriptors[0] = String::new();
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_accepts_multiple_valid_depths() {
        let mut c = valid_config();
        c.detection_depths = vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
        assert!(c.validate().is_ok());
    }

    // ── resolve_detection_layers edge cases ──

    #[test]
    fn resolve_layers_zero_layers_model_maps_all_to_zero() {
        let c = valid_config();
        // num_layers=0: floor(d*0) = 0 for all depths, min(0, saturating_sub(1)=0) = 0
        let layers = c.resolve_detection_layers(0);
        assert_eq!(layers, vec![0], "zero-layer model: all depths collapse to layer 0");
    }

    #[test]
    fn resolve_layers_small_model_clamps_to_last() {
        let mut c = valid_config();
        c.detection_depths = vec![0.99];
        let layers = c.resolve_detection_layers(2);
        // floor(0.99 * 2) = 1, min(1, 2-1) = 1
        assert_eq!(layers, vec![1]);
    }

    #[test]
    fn resolve_layers_all_same_depth_deduplicates() {
        let mut c = valid_config();
        c.detection_depths = vec![0.25; 10];
        let layers = c.resolve_detection_layers(100);
        assert_eq!(layers.len(), 1);
    }

    // ── DEFAULT_LEVEL_DESCRIPTORS count ──

    #[test]
    fn default_descriptors_exactly_three() {
        assert_eq!(DEFAULT_LEVEL_DESCRIPTORS.len(), 3);
    }

    // ── TokenizerEncodeError full coverage ──

    #[test]
    fn tokenizer_encode_error_debug() {
        let e = TokenizerEncodeError::EmptyText;
        let debug = format!("{:?}", e);
        assert!(debug.contains("EmptyText"), "Debug: {debug}");

        let e = TokenizerEncodeError::Backend("io".into());
        assert!(format!("{:?}", e).contains("Backend"));

        let e = TokenizerEncodeError::TokenOutOfRange { token: 5, vocab_size: 10 };
        assert!(format!("{:?}", e).contains("TokenOutOfRange"));
    }

    #[test]
    fn tokenizer_encode_error_clone_preserves() {
        let e = TokenizerEncodeError::Backend("original".into());
        let cloned = e.clone();
        assert_eq!(e.to_string(), cloned.to_string());
    }

    #[test]
    fn tokenizer_encode_error_implements_std_error() {
        use std::error::Error;
        let e = TokenizerEncodeError::EmptyText;
        let _: &dyn Error = &e;
    }

    // ── NoOpTokenizerLookup trait object ──

    #[test]
    fn no_op_tokenizer_as_trait_object() {
        let lookup: &dyn TokenizerLookup = &NoOpTokenizerLookup;
        assert_eq!(lookup.decode(&[100, 200, 300]), "");
    }

    // ════════════════════════════════════════════════════════════════════════
    // Additional tests — 45+ new tests for comprehensive coverage
    // ════════════════════════════════════════════════════════════════════════

    // ── SemanticLevel exhaustive variant coverage ──

    #[test]
    fn semantic_level_eq_same_variants() {
        assert_eq!(SemanticLevel::L1, SemanticLevel::L1);
        assert_eq!(SemanticLevel::L2, SemanticLevel::L2);
        assert_eq!(SemanticLevel::L3, SemanticLevel::L3);
    }

    #[test]
    fn semantic_level_neq_different_variants() {
        assert_ne!(SemanticLevel::L1, SemanticLevel::L2);
        assert_ne!(SemanticLevel::L2, SemanticLevel::L3);
        assert_ne!(SemanticLevel::L1, SemanticLevel::L3);
    }

    #[test]
    fn semantic_level_from_idx_usize_max_returns_none() {
        assert_eq!(SemanticLevel::from_idx(usize::MAX), None);
    }

    #[test]
    fn semantic_level_as_idx_returns_correct_values() {
        assert_eq!(SemanticLevel::L1.as_idx(), 0);
        assert_eq!(SemanticLevel::L2.as_idx(), 1);
        assert_eq!(SemanticLevel::L3.as_idx(), 2);
    }

    #[test]
    fn semantic_level_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        SemanticLevel::L1.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        SemanticLevel::L1.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish(), "same variant must produce same hash");
    }

    #[test]
    fn semantic_level_order_matches_as_idx() {
        for (i, &level) in SemanticLevel::ORDER.iter().enumerate() {
            assert_eq!(level.as_idx(), i);
        }
    }

    // ── SemanticGatekeeperError: SmallGraph and RingBuffer variants ──

    #[test]
    fn sg_error_small_graph_display() {
        let e = SemanticGatekeeperError::SmallGraph("compilation blew up".into());
        let msg = e.to_string();
        assert!(msg.contains("compilation blew up"), "SmallGraph display missing detail: {msg}");
    }

    #[test]
    fn sg_error_small_graph_debug() {
        let e = SemanticGatekeeperError::SmallGraph("test".into());
        let debug = format!("{:?}", e);
        assert!(debug.contains("SmallGraph"), "Debug should contain SmallGraph: {debug}");
    }

    #[test]
    fn sg_error_from_qtap_stale() {
        let err = QTapReadError::StaleQTap { buf_step: 5, expected_step: 10 };
        let sg: SemanticGatekeeperError = err.into();
        match &sg {
            SemanticGatekeeperError::RingBuffer(QTapReadError::StaleQTap { buf_step, expected_step }) => {
                assert_eq!(*buf_step, 5);
                assert_eq!(*expected_step, 10);
            }
            other => panic!("expected RingBuffer(StaleQTap), got: {:?}", other),
        }
    }

    #[test]
    fn sg_error_from_qtap_insufficient_capacity() {
        let err = QTapReadError::InsufficientCapacity { capacity: 8, required: 16 };
        let sg: SemanticGatekeeperError = err.into();
        match &sg {
            SemanticGatekeeperError::RingBuffer(QTapReadError::InsufficientCapacity { capacity, required }) => {
                assert_eq!(*capacity, 8);
                assert_eq!(*required, 16);
            }
            other => panic!("expected RingBuffer(InsufficientCapacity), got: {:?}", other),
        }
    }

    // ── SemanticGatekeeperConfig validation: negative detection depth ──

    #[test]
    fn validate_rejects_nan_detection_depth() {
        let mut c = valid_config();
        c.detection_depths = vec![f32::NAN];
        assert!(c.validate().is_err(), "NaN detection depth should be rejected");
    }

    #[test]
    fn validate_rejects_neg_infinity_detection_depth() {
        let mut c = valid_config();
        c.detection_depths = vec![f32::NEG_INFINITY];
        assert!(c.validate().is_err(), "neg infinity detection depth should be rejected");
    }

    #[test]
    fn validate_rejects_nan_stability_threshold() {
        let mut c = valid_config();
        c.stability_threshold = f32::NAN;
        assert!(c.validate().is_err(), "NaN stability should be rejected");
    }

    #[test]
    fn validate_accepts_zero_gate_threshold() {
        let mut c = valid_config();
        c.gate_threshold = 0.0;
        assert!(c.validate().is_ok(), "gate_threshold=0.0 should be valid");
    }

    #[test]
    fn validate_accepts_one_alpha() {
        let mut c = valid_config();
        c.alpha = 1.0;
        assert!(c.validate().is_ok(), "alpha=1.0 should be valid");
    }

    #[test]
    fn validate_rejects_multiple_bad_fields_reports_first() {
        let mut c = valid_config();
        c.gate_threshold = 2.0;
        c.alpha = -1.0;
        c.detection_depths = vec![-0.5];
        let result = c.validate();
        assert!(result.is_err());
        // Should return an error (whichever field is checked first).
    }

    #[test]
    fn validate_rejects_empty_descriptors_all_three() {
        let mut c = valid_config();
        c.level_descriptors = [
            "   ".to_string(),
            "\t".to_string(),
            "\n".to_string(),
        ];
        assert!(c.validate().is_err(), "whitespace-only descriptors should be rejected");
    }

    // ── resolve_detection_layers: edge cases ──

    #[test]
    fn resolve_layers_large_model_correct_indices() {
        let c = valid_config(); // [0.5, 0.75, 0.9]
        let layers = c.resolve_detection_layers(100);
        // floor(0.5 * 100) = 50, floor(0.75 * 100) = 75, floor(0.9 * 100) = 90
        assert_eq!(layers, vec![50, 75, 90]);
    }

    #[test]
    fn resolve_layers_depth_near_zero_produces_first_layer() {
        let mut c = valid_config();
        c.detection_depths = vec![0.0001];
        let layers = c.resolve_detection_layers(1000);
        // floor(0.0001 * 1000) = 0
        assert_eq!(layers, vec![0]);
    }

    #[test]
    fn resolve_layers_empty_depths_produces_empty() {
        let mut c = valid_config();
        c.detection_depths = vec![];
        let layers = c.resolve_detection_layers(100);
        assert!(layers.is_empty());
    }

    #[test]
    fn resolve_layers_depth_one_clamps_to_last() {
        let mut c = valid_config();
        c.detection_depths = vec![1.0, 1.0, 1.0];
        let layers = c.resolve_detection_layers(50);
        // All map to floor(1.0*50)=50, min(50, 49)=49
        assert_eq!(layers, vec![49]);
    }

    #[test]
    fn resolve_layers_two_layer_model() {
        let c = valid_config(); // [0.5, 0.75, 0.9]
        let layers = c.resolve_detection_layers(2);
        // floor(0.5*2)=1, min(1,1)=1; floor(0.75*2)=1; floor(0.9*2)=1
        assert_eq!(layers, vec![1]);
    }

    // ── KnowledgeEntry edge cases ──

    #[test]
    fn knowledge_entry_confidence_one() {
        let e = KnowledgeEntry {
            text: "max confidence".to_string(),
            confidence: 1.0,
        };
        assert!((e.confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn knowledge_entry_confidence_nan() {
        let e = KnowledgeEntry {
            text: "bad".to_string(),
            confidence: f32::NAN,
        };
        assert!(e.confidence.is_nan());
    }

    #[test]
    fn knowledge_entry_confidence_infinity() {
        let e = KnowledgeEntry {
            text: "overflow".to_string(),
            confidence: f32::INFINITY,
        };
        assert!(e.confidence.is_infinite() && e.confidence.is_sign_positive());
    }

    #[test]
    fn knowledge_entry_confidence_neg_infinity() {
        let e = KnowledgeEntry {
            text: "underflow".to_string(),
            confidence: f32::NEG_INFINITY,
        };
        assert!(e.confidence.is_infinite() && e.confidence.is_sign_negative());
    }

    #[test]
    fn knowledge_entry_long_text() {
        let long = "a".repeat(100_000);
        let e = KnowledgeEntry {
            text: long.clone(),
            confidence: 0.5,
        };
        assert_eq!(e.text.len(), 100_000);
    }

    // ── AstContext edge cases ──

    #[test]
    fn ast_context_zero_cursor() {
        let ctx = AstContext {
            node_kind: "root",
            cursor_line: 0,
            cursor_column: 0,
            prefix: "",
        };
        assert_eq!(ctx.cursor_line, 0);
        assert_eq!(ctx.cursor_column, 0);
        assert!(ctx.prefix.is_empty());
    }

    #[test]
    fn ast_context_max_cursor_values() {
        let ctx = AstContext {
            node_kind: "very_long_node_kind_name",
            cursor_line: u32::MAX,
            cursor_column: u32::MAX,
            prefix: "x",
        };
        assert_eq!(ctx.cursor_line, u32::MAX);
        assert_eq!(ctx.cursor_column, u32::MAX);
    }

    #[test]
    fn ast_context_debug_contains_all_fields() {
        let ctx = AstContext {
            node_kind: "fn_decl",
            cursor_line: 42,
            cursor_column: 7,
            prefix: "my_",
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("fn_decl"), "Debug should contain node_kind: {debug}");
        assert!(debug.contains("42"), "Debug should contain cursor_line: {debug}");
        assert!(debug.contains("7"), "Debug should contain cursor_column: {debug}");
        assert!(debug.contains("my_"), "Debug should contain prefix: {debug}");
    }

    // ── RetrieveContext edge cases ──

    #[test]
    fn retrieve_context_step_zero() {
        let tokens: [u32; 0] = [];
        let ctx = RetrieveContext {
            generated_tokens: &tokens,
            ast: None,
            step: 0,
            request_id: 0,
        };
        assert_eq!(ctx.step, 0);
        assert_eq!(ctx.request_id, 0);
    }

    #[test]
    fn retrieve_context_step_max() {
        let tokens = [1u32, 2, 3];
        let ctx = RetrieveContext {
            generated_tokens: &tokens,
            ast: None,
            step: u64::MAX,
            request_id: u64::MAX,
        };
        assert_eq!(ctx.step, u64::MAX);
        assert_eq!(ctx.request_id, u64::MAX);
    }

    #[test]
    fn retrieve_context_with_ast_and_without() {
        let tokens = [10u32, 20];
        let ast = AstContext {
            node_kind: "expr",
            cursor_line: 1,
            cursor_column: 1,
            prefix: "",
        };
        let with = RetrieveContext {
            generated_tokens: &tokens,
            ast: Some(ast),
            step: 5,
            request_id: 100,
        };
        let without = RetrieveContext {
            generated_tokens: &tokens,
            ast: None,
            step: 5,
            request_id: 100,
        };
        assert!(with.ast.is_some());
        assert!(without.ast.is_none());
    }

    // ── TokenizerEncodeError coverage ──

    #[test]
    fn tokenizer_encode_error_backend_display() {
        let e = TokenizerEncodeError::Backend("disk full".into());
        let msg = e.to_string();
        assert!(msg.contains("disk full"), "Backend display: {msg}");
        assert!(msg.contains("tokenizer backend error"), "Backend display should contain prefix: {msg}");
    }

    #[test]
    fn tokenizer_encode_error_token_out_of_range_display() {
        let e = TokenizerEncodeError::TokenOutOfRange { token: 99999, vocab_size: 50000 };
        let msg = e.to_string();
        assert!(msg.contains("99999"), "token id missing: {msg}");
        assert!(msg.contains("50000"), "vocab size missing: {msg}");
    }

    #[test]
    fn tokenizer_encode_error_all_variants_clone() {
        let variants = vec![
            TokenizerEncodeError::EmptyText,
            TokenizerEncodeError::Backend("x".into()),
            TokenizerEncodeError::TokenOutOfRange { token: 1, vocab_size: 2 },
        ];
        for v in &variants {
            let cloned = v.clone();
            assert_eq!(v.to_string(), cloned.to_string());
        }
    }

    // ── SemanticGatekeeperConfig with_defaults detection depths ──

    #[test]
    fn with_defaults_detection_depths_values() {
        let c = SemanticGatekeeperConfig::with_defaults(Arc::new(DummyProvider));
        assert_eq!(c.detection_depths.len(), 3);
        assert!((c.detection_depths[0] - 0.5).abs() < 1e-6);
        assert!((c.detection_depths[1] - 0.75).abs() < 1e-6);
        assert!((c.detection_depths[2] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn with_defaults_gate_threshold_range() {
        let c = SemanticGatekeeperConfig::with_defaults(Arc::new(DummyProvider));
        assert!(c.gate_threshold > 0.0 && c.gate_threshold < 1.0);
    }

    #[test]
    fn with_defaults_stability_threshold_range() {
        let c = SemanticGatekeeperConfig::with_defaults(Arc::new(DummyProvider));
        assert!(c.stability_threshold > 0.0 && c.stability_threshold <= 1.0);
    }

    #[test]
    fn with_defaults_alpha_range() {
        let c = SemanticGatekeeperConfig::with_defaults(Arc::new(DummyProvider));
        assert!(c.alpha > 0.0 && c.alpha < 1.0);
    }

    // ── DEFAULT_LEVEL_DESCRIPTORS content validation ──

    #[test]
    fn default_descriptors_distinct_content() {
        assert_ne!(DEFAULT_LEVEL_DESCRIPTORS[0], DEFAULT_LEVEL_DESCRIPTORS[1]);
        assert_ne!(DEFAULT_LEVEL_DESCRIPTORS[1], DEFAULT_LEVEL_DESCRIPTORS[2]);
        assert_ne!(DEFAULT_LEVEL_DESCRIPTORS[0], DEFAULT_LEVEL_DESCRIPTORS[2]);
    }

    // ── Validation boundary: gate_threshold at exactly 1.0 ──

    #[test]
    fn validate_accepts_gate_threshold_exactly_one() {
        let mut c = valid_config();
        c.gate_threshold = 1.0;
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_accepts_stability_exactly_zero() {
        let mut c = valid_config();
        c.stability_threshold = 0.0;
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_rejects_infinity_alpha() {
        let mut c = valid_config();
        c.alpha = f32::INFINITY;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_neg_infinity_alpha() {
        let mut c = valid_config();
        c.alpha = f32::NEG_INFINITY;
        assert!(c.validate().is_err());
    }

    // ── NoOpTokenizerLookup edge cases ──

    #[test]
    fn no_op_tokenizer_empty_input() {
        let t = NoOpTokenizerLookup;
        assert_eq!(t.decode(&[]), "");
    }

    #[test]
    fn no_op_tokenizer_large_input() {
        let t = NoOpTokenizerLookup;
        let tokens: Vec<u32> = (0..10000).collect();
        assert_eq!(t.decode(&tokens), "");
    }

    // ── KnowledgeEntry clone independence ──

    #[test]
    fn knowledge_entry_clone_independence() {
        let mut e = KnowledgeEntry {
            text: "original".to_string(),
            confidence: 0.5,
        };
        let cloned = e.clone();
        e.text.push_str(" modified");
        assert_ne!(e.text, cloned.text, "clone should be independent");
    }

    // ── AstContext copy independence (field values don't share state) ──

    #[test]
    fn ast_context_copy_field_independence() {
        let ctx1 = AstContext {
            node_kind: "a",
            cursor_line: 1,
            cursor_column: 2,
            prefix: "p",
        };
        let ctx2 = ctx1;
        // Both are Copy types pointing to borrowed &'a str; they are equal
        assert_eq!(ctx1.node_kind, ctx2.node_kind);
        assert_eq!(ctx1.cursor_line, ctx2.cursor_line);
    }

    // ── SemanticGatekeeperConfig validate: single detection depth valid ──

    #[test]
    fn validate_single_detection_depth_valid() {
        let mut c = valid_config();
        c.detection_depths = vec![0.001];
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_detection_depth_exactly_one() {
        let mut c = valid_config();
        c.detection_depths = vec![1.0];
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_detection_depth_just_above_zero() {
        let mut c = valid_config();
        c.detection_depths = vec![f32::MIN_POSITIVE];
        assert!(c.validate().is_ok());
    }

    // ── resolve_detection_layers: usize overflow safety ──

    #[test]
    fn resolve_layers_usize_max_clamps_correctly() {
        let mut c = valid_config();
        c.detection_depths = vec![1.0];
        let layers = c.resolve_detection_layers(usize::MAX);
        // floor(1.0 * usize::MAX) would overflow f32, but saturating_sub ensures safety
        // The result should be a single layer at usize::MAX - 1
        assert_eq!(layers.len(), 1);
    }

    // ── TextEncoderError coverage (from callback module) ──

    #[test]
    fn text_encoder_error_tokenize_display() {
        use callback::TextEncoderError;
        let e = TextEncoderError::Tokenize("bad input".into());
        assert!(e.to_string().contains("bad input"));
        assert!(e.to_string().contains("tokenize"));
    }

    #[test]
    fn text_encoder_error_execute_display() {
        use callback::TextEncoderError;
        let e = TextEncoderError::Execute("segfault".into());
        assert!(e.to_string().contains("segfault"));
        assert!(e.to_string().contains("graph execute"));
    }

    #[test]
    fn text_encoder_error_uninitialized_display() {
        use callback::TextEncoderError;
        let e = TextEncoderError::Uninitialized;
        assert!(e.to_string().contains("not initialized"));
    }

    #[test]
    fn text_encoder_error_debug_all_variants() {
        use callback::TextEncoderError;
        assert!(format!("{:?}", TextEncoderError::Tokenize("a".into())).contains("Tokenize"));
        assert!(format!("{:?}", TextEncoderError::Execute("b".into())).contains("Execute"));
        assert!(format!("{:?}", TextEncoderError::Uninitialized).contains("Uninitialized"));
    }

    #[test]
    fn text_encoder_error_clone_preserves() {
        use callback::TextEncoderError;
        let e = TextEncoderError::Tokenize("original".into());
        let cloned = e.clone();
        assert_eq!(e.to_string(), cloned.to_string());
    }

    #[test]
    fn text_encoder_error_implements_std_error() {
        use callback::TextEncoderError;
        use std::error::Error;
        let e = TextEncoderError::Uninitialized;
        let _: &dyn Error = &e;
    }

    // ── SEMANTIC_GATEKEEPER_PRIORITY constant ──

    #[test]
    fn semantic_gatekeeper_priority_value() {
        assert_eq!(callback::SEMANTIC_GATEKEEPER_PRIORITY, 90);
    }

    // ── LevelKeysCache and LevelKeysError (from level_keys module) ──

    #[test]
    fn level_keys_cache_new_is_empty() {
        let cache = LevelKeysCache::new(4);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.kv_dim(), 4);
        assert!(cache.detection_layers().is_empty());
    }

    #[test]
    fn level_keys_cache_default_is_empty() {
        let cache = LevelKeysCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.kv_dim(), 0);
    }

    #[test]
    fn level_keys_cache_insert_and_get() {
        let mut cache = LevelKeysCache::new(3);
        let keys: [Vec<f32>; 3] = [
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        cache.insert(5, keys.clone()).unwrap();
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        let retrieved = cache.get(5).unwrap();
        assert_eq!(retrieved[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(retrieved[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(retrieved[2], vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn level_keys_cache_get_missing_returns_none() {
        let cache = LevelKeysCache::new(4);
        assert!(cache.get(0).is_none());
        assert!(cache.get(999).is_none());
    }

    #[test]
    fn level_keys_cache_insert_dim_mismatch() {
        let mut cache = LevelKeysCache::new(4);
        let bad_keys: [Vec<f32>; 3] = [
            vec![1.0, 2.0], // wrong dim
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ];
        let err = cache.insert(0, bad_keys).unwrap_err();
        assert!(matches!(err, LevelKeysError::DimMismatch { .. }));
        assert!(cache.is_empty(), "failed insert should not add entry");
    }

    #[test]
    fn level_keys_cache_insert_non_finite() {
        let mut cache = LevelKeysCache::new(3);
        let bad_keys: [Vec<f32>; 3] = [
            vec![1.0, 2.0, f32::NAN],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];
        let err = cache.insert(0, bad_keys).unwrap_err();
        assert!(matches!(err, LevelKeysError::NonFinite { .. }));
    }

    #[test]
    fn level_keys_cache_insert_all_zero() {
        let mut cache = LevelKeysCache::new(3);
        let bad_keys: [Vec<f32>; 3] = [
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];
        let err = cache.insert(0, bad_keys).unwrap_err();
        assert!(matches!(err, LevelKeysError::AllZero { .. }));
    }

    #[test]
    fn level_keys_cache_insert_multiple_layers() {
        let mut cache = LevelKeysCache::new(2);
        let keys_a: [Vec<f32>; 3] = [
            vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],
        ];
        let keys_b: [Vec<f32>; 3] = [
            vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0],
        ];
        cache.insert(10, keys_a).unwrap();
        cache.insert(5, keys_b).unwrap();
        assert_eq!(cache.len(), 2);
        // detection_layers sorted
        assert_eq!(cache.detection_layers(), &[5, 10]);
    }

    #[test]
    fn level_keys_cache_insert_duplicate_layer_replaces() {
        let mut cache = LevelKeysCache::new(2);
        let first: [Vec<f32>; 3] = [
            vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],
        ];
        let second: [Vec<f32>; 3] = [
            vec![10.0, 20.0], vec![30.0, 40.0], vec![50.0, 60.0],
        ];
        cache.insert(3, first).unwrap();
        cache.insert(3, second).unwrap();
        assert_eq!(cache.len(), 1);
        let retrieved = cache.get(3).unwrap();
        assert_eq!(retrieved[0], vec![10.0, 20.0]);
    }

    #[test]
    fn level_keys_cache_detection_layers_sorted() {
        let mut cache = LevelKeysCache::new(2);
        let keys: [Vec<f32>; 3] = [
            vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],
        ];
        cache.insert(30, keys.clone()).unwrap();
        cache.insert(10, keys.clone()).unwrap();
        cache.insert(20, keys).unwrap();
        assert_eq!(cache.detection_layers(), &[10, 20, 30]);
    }

    // ── LevelKeysError display coverage ──

    #[test]
    fn level_keys_error_dim_mismatch_display() {
        let e = LevelKeysError::DimMismatch {
            layer_idx: 5,
            level_idx: 2,
            actual: 3,
            expected: 4,
        };
        let msg = e.to_string();
        assert!(msg.contains("5"), "layer_idx missing: {msg}");
        assert!(msg.contains("2"), "level_idx missing: {msg}");
        assert!(msg.contains("3"), "actual missing: {msg}");
        assert!(msg.contains("4"), "expected missing: {msg}");
    }

    #[test]
    fn level_keys_error_non_finite_display() {
        let e = LevelKeysError::NonFinite { layer_idx: 1, level_idx: 0 };
        let msg = e.to_string();
        assert!(msg.contains("non-finite"), "should contain 'non-finite': {msg}");
        assert!(msg.contains("layer=1"), "should contain layer: {msg}");
        assert!(msg.contains("level_idx=0"), "should contain level: {msg}");
    }

    #[test]
    fn level_keys_error_all_zero_display() {
        let e = LevelKeysError::AllZero { layer_idx: 3, level_idx: 1 };
        let msg = e.to_string();
        assert!(msg.contains("all-zero"), "should contain 'all-zero': {msg}");
        assert!(msg.contains("layer=3"), "should contain layer: {msg}");
    }

    #[test]
    fn level_keys_error_partial_eq() {
        let a = LevelKeysError::DimMismatch { layer_idx: 1, level_idx: 0, actual: 2, expected: 4 };
        let b = LevelKeysError::DimMismatch { layer_idx: 1, level_idx: 0, actual: 2, expected: 4 };
        assert_eq!(a, b);
        let c = LevelKeysError::DimMismatch { layer_idx: 1, level_idx: 0, actual: 3, expected: 4 };
        assert_ne!(a, c);
        let d = LevelKeysError::NonFinite { layer_idx: 1, level_idx: 0 };
        assert_ne!(a, d);
    }

    #[test]
    fn level_keys_error_clone_preserves() {
        let e = LevelKeysError::NonFinite { layer_idx: 7, level_idx: 2 };
        let cloned = e.clone();
        assert_eq!(e, cloned);
    }

    // ── SemanticGatekeeperConfig validation: infinity in negative infinity gate ──

    #[test]
    fn validate_rejects_neg_infinity_gate() {
        let mut c = valid_config();
        c.gate_threshold = f32::NEG_INFINITY;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_pos_infinity_gate() {
        let mut c = valid_config();
        c.gate_threshold = f32::INFINITY;
        assert!(c.validate().is_err());
    }

    // ── SemanticGatekeeperError: Provider variant ──

    #[test]
    fn sg_error_provider_debug() {
        let e = SemanticGatekeeperError::Provider("timeout".into());
        let debug = format!("{:?}", e);
        assert!(debug.contains("Provider"), "Debug should contain Provider: {debug}");
    }

    #[test]
    fn sg_error_tokenizer_debug() {
        let e = SemanticGatekeeperError::Tokenizer("invalid utf8".into());
        let debug = format!("{:?}", e);
        assert!(debug.contains("Tokenizer"), "Debug should contain Tokenizer: {debug}");
    }

    // ── validate: depth just barely above zero ──

    #[test]
    fn validate_rejects_depth_negative_epsilon() {
        let mut c = valid_config();
        c.detection_depths = vec![-f32::MIN_POSITIVE];
        assert!(c.validate().is_err(), "negative epsilon depth should be rejected");
    }

    #[test]
    fn validate_accepts_depth_positive_epsilon() {
        let mut c = valid_config();
        c.detection_depths = vec![f32::MIN_POSITIVE];
        assert!(c.validate().is_ok(), "positive epsilon depth should be accepted");
    }

    // ════════════════════════════════════════════════════════════════════════
    // Additional tests — boundary conditions and untested API surfaces
    // ════════════════════════════════════════════════════════════════════════

    // ── KnowledgeProvider trait object dispatch ──

    #[test]
    fn knowledge_provider_trait_object_returns_some() {
        struct TestProvider;
        impl KnowledgeProvider for TestProvider {
            fn retrieve(
                &self,
                query: &[f32],
                level: SemanticLevel,
                _ctx: &RetrieveContext<'_>,
            ) -> Option<KnowledgeEntry> {
                Some(KnowledgeEntry {
                    text: format!("{:?}: {:?}", level, query.len()),
                    confidence: 0.7,
                })
            }
        }

        // Arrange
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(TestProvider);
        let tokens: [u32; 0] = [];
        let ctx = RetrieveContext {
            generated_tokens: &tokens,
            ast: None,
            step: 1,
            request_id: 42,
        };
        let query = [0.1f32, 0.2, 0.3];

        // Act
        let result = provider.retrieve(&query, SemanticLevel::L2, &ctx);

        // Assert
        let entry = result.expect("provider should return Some");
        assert!(entry.text.contains("L2"), "text should contain level: {}", entry.text);
        assert!((entry.confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn knowledge_provider_trait_object_returns_none() {
        struct EmptyProvider;
        impl KnowledgeProvider for EmptyProvider {
            fn retrieve(&self, _: &[f32], _: SemanticLevel, _: &RetrieveContext<'_>) -> Option<KnowledgeEntry> {
                None
            }
        }

        // Arrange
        let provider: Arc<dyn KnowledgeProvider> = Arc::new(EmptyProvider);
        let tokens = [1u32, 2];
        let ctx = RetrieveContext {
            generated_tokens: &tokens,
            ast: None,
            step: 10,
            request_id: 0,
        };

        // Act
        let result = provider.retrieve(&[0.5], SemanticLevel::L1, &ctx);

        // Assert
        assert!(result.is_none(), "EmptyProvider must return None");
    }

    // ── TokenizerEncoder trait object dispatch ──

    #[test]
    fn tokenizer_encoder_trait_object_rejects_whitespace_only() {
        struct WhitespaceRejectingEncoder;
        impl TokenizerEncoder for WhitespaceRejectingEncoder {
            fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerEncodeError> {
                if text.trim().is_empty() {
                    return Err(TokenizerEncodeError::EmptyText);
                }
                Ok(vec![1, 2, 3])
            }
        }

        // Arrange
        let encoder: &dyn TokenizerEncoder = &WhitespaceRejectingEncoder;

        // Act
        let result = encoder.encode("   \t\n  ");

        // Assert
        let err = result.expect_err("whitespace-only text should fail");
        assert!(matches!(err, TokenizerEncodeError::EmptyText));
    }

    // ── resolve_detection_layers: mixed valid and clamped depths ──

    #[test]
    fn resolve_layers_mixed_depths_produces_correct_set() {
        // Arrange
        let mut c = valid_config();
        c.detection_depths = vec![0.0 + f32::EPSILON, 0.33, 0.5, 0.67, 1.0];
        // 100 layers: 0, 33, 50, 67, 99

        // Act
        let layers = c.resolve_detection_layers(100);

        // Assert
        assert_eq!(layers.len(), 5, "5 distinct depths should produce 5 layers");
        assert_eq!(layers[0], 0);
        assert_eq!(layers[1], 33);
        assert_eq!(layers[2], 50);
        assert_eq!(layers[3], 67);
        assert_eq!(layers[4], 99);
    }

    // ── validate: detection_depth with value exactly zero ──

    #[test]
    fn validate_rejects_depth_exactly_zero() {
        // Arrange
        let mut c = valid_config();
        c.detection_depths = vec![0.0];

        // Act
        let result = c.validate();

        // Assert
        assert!(result.is_err(), "depth=0.0 should be rejected (must be > 0)");
    }

    // ── KnowledgeEntry with Unicode text ──

    #[test]
    fn knowledge_entry_unicode_text_preserved() {
        // Arrange
        let unicode_text = "模型架构：深层Transformer 🧠 中文+emoji";
        let entry = KnowledgeEntry {
            text: unicode_text.to_string(),
            confidence: 1.0,
        };

        // Act
        let cloned = entry.clone();

        // Assert
        assert_eq!(entry.text, unicode_text);
        assert_eq!(cloned.text, unicode_text);
        assert_eq!(entry.text.chars().count(), unicode_text.chars().count());
    }

    // ── SemanticGatekeeperError: PrecomputeFailed variant ──

    #[test]
    fn sg_error_precompute_failed_display_and_debug() {
        // Arrange
        let detail = "cuda OOM during k_proj projection";
        let e = SemanticGatekeeperError::PrecomputeFailed(detail.to_string());

        // Act
        let display = e.to_string();
        let debug = format!("{:?}", e);

        // Assert
        assert!(display.contains(detail), "Display should contain detail: {display}");
        assert!(debug.contains("PrecomputeFailed"), "Debug should contain variant: {debug}");
    }

    // ── SemanticGatekeeperError: NotRegistered variant ──

    #[test]
    fn sg_error_not_registered_is_distinct_from_provider() {
        // Arrange
        let a = SemanticGatekeeperError::NotRegistered;
        let b = SemanticGatekeeperError::Provider("timeout".into());

        // Act & Assert
        assert_ne!(
            format!("{:?}", a),
            format!("{:?}", b),
            "NotRegistered and Provider must be distinct Debug representations"
        );
        assert!(a.to_string().contains("not registered"));
        assert!(b.to_string().contains("timeout"));
    }

    // ── AstSentinel trait object dispatch ──

    #[test]
    fn ast_sentinel_trait_object_returns_context() {
        struct FixedAstSentinel;
        impl AstSentinel for FixedAstSentinel {
            fn current_context<'a>(
                &self,
                tokens: &'a [u32],
                _tokenizer: &dyn TokenizerLookup,
            ) -> Option<AstContext<'a>> {
                if tokens.is_empty() {
                    return None;
                }
                Some(AstContext {
                    node_kind: "test_node",
                    cursor_line: 1,
                    cursor_column: 1,
                    prefix: "",
                })
            }
        }

        // Arrange
        let sentinel: &dyn AstSentinel = &FixedAstSentinel;
        let tokens = [100u32, 200, 300];

        // Act
        let result = sentinel.current_context(&tokens, &NoOpTokenizerLookup);

        // Assert
        let ctx = result.expect("non-empty tokens should produce context");
        assert_eq!(ctx.node_kind, "test_node");
    }

    #[test]
    fn ast_sentinel_trait_object_returns_none_for_empty() {
        struct FixedAstSentinel;
        impl AstSentinel for FixedAstSentinel {
            fn current_context<'a>(
                &self,
                tokens: &'a [u32],
                _tokenizer: &dyn TokenizerLookup,
            ) -> Option<AstContext<'a>> {
                if tokens.is_empty() { None } else { panic!("unexpected") }
            }
        }

        // Arrange
        let sentinel: &dyn AstSentinel = &FixedAstSentinel;
        let tokens: [u32; 0] = [];

        // Act
        let result = sentinel.current_context(&tokens, &NoOpTokenizerLookup);

        // Assert
        assert!(result.is_none(), "empty tokens should yield None");
    }

    // ── validate: all three descriptors whitespace-only ──

    #[test]
    fn validate_rejects_all_whitespace_descriptors() {
        // Arrange
        let mut c = valid_config();
        c.level_descriptors = [
            "  ".to_string(),
            "\t\n".to_string(),
            "\u{00A0}".to_string(), // non-breaking space
        ];

        // Act
        let result = c.validate();

        // Assert
        assert!(result.is_err(), "all-whitespace descriptors should be rejected");
    }
}
