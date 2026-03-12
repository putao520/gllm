use std::time::Instant;

/// Physical page identifier.
pub type PageId = usize;
/// Physical page identifier (alias used by memory_manager).
pub type PhysicalId = usize;
/// Unique request identifier.
pub type RequestId = u64;
/// Opaque storage key for swap-out / swap-in.
pub type StorageKey = u64;

/// Page lifecycle state visible to the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageState {
    Free,
    Active,
    Standby,
    SwappedOut,
    Warm,
    Protected,
    Swapped,
}

/// A group of pages that belong to the same request/sequence.
/// Gang scheduling evicts whole groups to avoid intra-sequence fragmentation.
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    pub id: RequestId,
    pub pages: Vec<PageId>,
    pub state: GroupState,
    pub access_count: usize,
    pub last_access: Instant,
    /// Pinned groups are immune to eviction (e.g., during prefill).
    pub is_pinned: bool,
    /// Total number of tokens in the sequence context.
    pub context_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupState {
    Running,
    Swapped,
    Paused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    Chat,
    Embedding,
    Rerank,
}

/// 批处理顺序策略 (ARCH-SCHED-BATCH-ORDER)
///
/// 准确度 > 吞吐量原则：默认使用 StrictRequestIdOrder
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchOrderPolicy {
    /// 严格按 RequestId 排序（确定性，精度优先）- 默认
    #[default]
    StrictRequestIdOrder,
    /// 按入队时间排序（确定性，FIFO）
    FifoOrder,
    /// 允许 vLLM 风格重排（性能优先，不推荐）
    #[deprecated = "Breaks determinism, use StrictRequestIdOrder instead"]
    ThroughputFirst,
}

/// KV Cache 管线类型 (ARCH-SCHED-PIPELINE)
///
/// 用于分离可复用会话内容与可丢弃工作内容
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvPipeline {
    /// 主对话上下文（System/User/Assistant）- 跨轮保留
    Conversation,
    /// 临时上下文（Thinking/Reasoning）- 轮次结束可回收
    Working,
}

/// 带管线标识的虚拟页面 ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelinedVirtualPageId {
    pub pipeline: KvPipeline,
    pub sequence_id: RequestId,
    pub logical_index: usize,
}

/// Per-page metadata needed by HGAL (Hybrid Gang-Aware LIRS).
#[derive(Debug, Clone)]
pub struct PageMetadata {
    pub page_id: PageId,
    pub sequence_id: Option<RequestId>,
    /// Inter-Reference Recency (IRR) value for LIRS-style scoring.
    pub recency: usize,
    pub access_count: usize,
    pub last_access: Instant,
    pub swap_in_time: Option<Instant>,
    /// Whether this page currently belongs to the LIR working set.
    pub is_lir: bool,
    /// Current state mirrored from backend for eviction decisions.
    pub state: PageState,
    /// Warm protection expiry; None means not under warm-up protection.
    pub warm_until: Option<Instant>,
}

impl Default for PageMetadata {
    fn default() -> Self {
        Self {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        }
    }
}
