//! Semantic Gatekeeper — 隐藏状态驱动的免训练知识调度与注入 SDK.
//!
//! 实现 `SPEC/SEMANTIC-GATEKEEPER.md` 定义的运行时知识注入架构:
//!
//! - 模型加载期预计算 3 个层级键 `K_L1/K_L2/K_L3` (描述文本 → tokenizer →
//!   冻结 embed → 检测层 k_proj 投影)
//! - 推理期每到检测层,`SemanticGatekeeperCallback.pre_node` 从
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
pub mod level_keys;
pub mod ring_buffer;
pub mod small_graph;

pub use active_state::ActiveState;
pub use callback::SemanticGatekeeperCallback;
pub use level_keys::{precompute as precompute_level_keys, LevelKeysCache, LevelKeysError};
pub use ring_buffer::{GatekeeperRingBuffer, QTapReadError};
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
/// SG 内核在检测层 `pre_node` 中调用 `retrieve`,返回 `None` 时 SG 不注入,
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
/// SG 内核在每个检测层 `pre_node` 前调用 `current_context`,若返回的
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
