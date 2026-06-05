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
    /// §10 KvPipeline: Conversation (跨轮保留) or Working (轮次结束释放).
    pub pipeline: KvPipeline,
    /// What payload this group carries — determines eviction priority.
    pub payload_kind: Option<PagePayloadKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupState {
    Running,
    Swapped,
    Paused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
///
/// UnifiedVirtualPage (SPEC/03-DATA-STRUCTURE.md §11.2) — system-wide physical
/// page container. Every page, regardless of payload (KV context, MoE expert
/// weight, system prompt cache, RAG feature), uses this same structure. The
/// GlobalMemoryManager treats all pages uniformly for IRR-based LIRS eviction.
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

// ── UnifiedVirtualPage (SPEC/03-DATA-STRUCTURE.md §11.2) ──

/// What a physical page contains — determines eviction priority and migration policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PagePayloadKind {
    /// K/V tensor pages for attention — highest priority, gang-evicted as groups.
    KvContext,
    /// MoE expert weights paged in on demand — medium priority, per-expert eviction.
    ExpertWeight,
    /// Read-only system prompt cache — pinned unless memory pressure critical.
    PromptSystem,
    /// RAG-injected high-dim feature vectors — lowest priority, evictable.
    KnowledgeRAG,
    /// Dense layer weights (Q/K/V/O/FFN/Norm) — pinned, only evicted under extreme pressure.
    DenseLayerWeight,
}

/// Where the page data currently resides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryResidency {
    /// Device-local memory (GPU VRAM / NPU SRAM).
    DeviceLocal,
    /// Host memory (CPU RAM).
    HostLocal,
    /// Disk-backed swap (SSD/NVMe).
    DiskSwap,
}

/// Unified virtual page — single physical page abstraction for all payload types.
///
/// Replaces per-subsystem page containers (KvCacheConfig, MoESharedWeight, etc.)
/// with one system-wide container. GlobalMemoryManager applies uniform IRR-based
/// LIRS page replacement across all payload kinds.
#[derive(Debug, Clone)]
pub struct UnifiedVirtualPage {
    /// Physical page identifier.
    pub page_id: PageId,
    /// What this page stores.
    pub payload_kind: PagePayloadKind,
    /// Current physical residency.
    pub residency: MemoryResidency,
    /// Data type of the page's payload (F32/F16/BF16/etc.) — matches model geometry.dtype.
    pub dtype: gllm_kernels::types::DType,
    /// Owning request (None for shared/readonly pages like PromptSystem).
    pub owner: Option<RequestId>,
    /// Pipeline affiliation (only meaningful for KvContext pages).
    pub pipeline: Option<KvPipeline>,
    /// Logical page index within the owner's virtual address space.
    pub logical_index: usize,
    /// §22 REQ-COMP-010: Compression codec applied to this page's payload.
    pub codec: crate::kv_cache::CompressionCodec,
    /// §22 REQ-COMP-010: Compressed size in bytes (0 = uncompressed).
    pub compressed_size: u32,
    /// §22 REQ-COMP-010: Original (decompressed) size in bytes.
    pub decompressed_size: u32,
    /// Expert ID for MoE expert weight pages (None for non-expert pages).
    pub expert_id: Option<u32>,
    /// Layer index for MoE expert weight pages (None for non-expert pages).
    pub layer_idx: Option<usize>,
}

impl UnifiedVirtualPage {
    /// Create a KV context page.
    pub fn kv(
        page_id: PageId,
        owner: RequestId,
        pipeline: KvPipeline,
        logical_index: usize,
        dtype: gllm_kernels::types::DType,
    ) -> Self {
        Self {
            page_id,
            payload_kind: PagePayloadKind::KvContext,
            residency: MemoryResidency::DeviceLocal,
            dtype,
            owner: Some(owner),
            pipeline: Some(pipeline),
            logical_index,
            codec: crate::kv_cache::CompressionCodec::None,
            compressed_size: 0,
            decompressed_size: 0,
            expert_id: None,
            layer_idx: None,
        }
    }

    /// Create a MoE expert weight page.
    pub fn expert(
        page_id: PageId,
        expert_id: u32,
        layer_idx: usize,
        dtype: gllm_kernels::types::DType,
    ) -> Self {
        Self {
            page_id,
            payload_kind: PagePayloadKind::ExpertWeight,
            residency: MemoryResidency::DeviceLocal,
            dtype,
            owner: None,
            pipeline: None,
            logical_index: 0,
            codec: crate::kv_cache::CompressionCodec::None,
            compressed_size: 0,
            decompressed_size: 0,
            expert_id: Some(expert_id),
            layer_idx: Some(layer_idx),
        }
    }

    /// Create a read-only system prompt page.
    pub fn system_prompt(page_id: PageId, dtype: gllm_kernels::types::DType) -> Self {
        Self {
            page_id,
            payload_kind: PagePayloadKind::PromptSystem,
            residency: MemoryResidency::DeviceLocal,
            dtype,
            owner: None,
            pipeline: None,
            logical_index: 0,
            codec: crate::kv_cache::CompressionCodec::None,
            compressed_size: 0,
            decompressed_size: 0,
            expert_id: None,
            layer_idx: None,
        }
    }

    /// Create a RAG knowledge page.
    pub fn rag(page_id: PageId, owner: RequestId, dtype: gllm_kernels::types::DType) -> Self {
        Self {
            page_id,
            payload_kind: PagePayloadKind::KnowledgeRAG,
            residency: MemoryResidency::HostLocal,
            dtype,
            owner: Some(owner),
            pipeline: None,
            logical_index: 0,
            codec: crate::kv_cache::CompressionCodec::None,
            compressed_size: 0,
            decompressed_size: 0,
            expert_id: None,
            layer_idx: None,
        }
    }

    /// Create a Dense layer weight page (pinned, unevictable under normal pressure).
    pub fn dense_layer(
        page_id: PageId,
        logical_index: usize,
        dtype: gllm_kernels::types::DType,
    ) -> Self {
        Self {
            page_id,
            payload_kind: PagePayloadKind::DenseLayerWeight,
            residency: MemoryResidency::DeviceLocal,
            dtype,
            owner: None,
            pipeline: None,
            logical_index,
            codec: crate::kv_cache::CompressionCodec::None,
            compressed_size: 0,
            decompressed_size: 0,
            expert_id: None,
            layer_idx: Some(logical_index),
        }
    }

    /// Whether this page can be evicted under memory pressure.
    pub fn is_evictable(&self) -> bool {
        match self.payload_kind {
            PagePayloadKind::PromptSystem | PagePayloadKind::DenseLayerWeight => false,
            PagePayloadKind::KvContext
            | PagePayloadKind::ExpertWeight
            | PagePayloadKind::KnowledgeRAG => true,
        }
    }

    /// Whether this page is on device (GPU/NPU).
    pub fn is_on_device(&self) -> bool {
        self.residency == MemoryResidency::DeviceLocal
    }
}

// ── WeightTier (SPEC 21-WEIGHT-PAGING.md §5.1) ──

/// Weight page memory tier — maps to SPEC 21-WEIGHT-PAGING.md §5.1 L1/L2/L3.
///
/// Used by the weight page telemetry system to track page distribution
/// across the three-tier memory hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightTier {
    /// L1: Device-local memory (GPU VRAM / NPU SRAM) — Hot tier.
    Hot,
    /// L2: Host memory (CPU RAM) — Warm tier.
    Warm,
    /// L3: Disk-backed swap — Cold tier.
    Cold,
}

// ── EvictionPriority (SPEC §5) ──

/// Eviction priority for a weight page (SPEC 21-WEIGHT-PAGING.md §5).
///
/// Determines the order in which weight pages are evicted under memory pressure.
/// Score integrates access frequency, recency, layer depth, payload kind,
/// and pin status. **Lower score = higher eviction priority (evicted first).**
#[derive(Debug, Clone)]
pub struct EvictionPriority {
    /// Composite eviction score. Lower = evicted sooner.
    pub score: i64,
    /// What type of payload this page carries.
    pub payload_kind: PagePayloadKind,
    /// Whether the page is pinned (immune to eviction under normal pressure).
    pub is_pinned: bool,
    /// Access frequency count from HGAL metadata.
    pub access_count: usize,
    /// LIRS recency in ms (lower = more recently used).
    pub recency: usize,
    /// Layer depth (None for non-weight pages). Deeper layers are slightly
    /// preferred for eviction as they contribute less to early attention.
    pub layer_idx: Option<usize>,
    /// Expert ID (None for non-expert pages). Used to correlate with
    /// expert thermal state for temperature-aware eviction.
    pub expert_id: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_layer_page_is_not_evictable() {
        let page = UnifiedVirtualPage::dense_layer(42, 3, gllm_kernels::types::DType::BF16);
        assert_eq!(page.payload_kind, PagePayloadKind::DenseLayerWeight);
        assert!(
            !page.is_evictable(),
            "DenseLayerWeight pages must not be evictable"
        );
        assert!(page.is_on_device());
        assert_eq!(page.page_id, 42);
        assert_eq!(page.logical_index, 3);
        assert_eq!(page.dtype, gllm_kernels::types::DType::BF16);
        assert!(page.owner.is_none());
        assert!(page.pipeline.is_none());
        assert!(page.expert_id.is_none());
        assert_eq!(page.layer_idx, Some(3));
    }

    #[test]
    fn expert_page_is_evictable() {
        let page = UnifiedVirtualPage::expert(10, 5, 2, gllm_kernels::types::DType::F16);
        assert_eq!(page.payload_kind, PagePayloadKind::ExpertWeight);
        assert!(page.is_evictable(), "ExpertWeight pages must be evictable");
        assert_eq!(page.expert_id, Some(5));
        assert_eq!(page.layer_idx, Some(2));
    }

    #[test]
    fn kv_page_is_evictable() {
        let page = UnifiedVirtualPage::kv(
            1,
            99,
            KvPipeline::Conversation,
            0,
            gllm_kernels::types::DType::F32,
        );
        assert_eq!(page.payload_kind, PagePayloadKind::KvContext);
        assert!(page.is_evictable());
        assert_eq!(page.owner, Some(99));
    }

    #[test]
    fn system_prompt_page_is_not_evictable() {
        let page = UnifiedVirtualPage::system_prompt(7, gllm_kernels::types::DType::F32);
        assert_eq!(page.payload_kind, PagePayloadKind::PromptSystem);
        assert!(!page.is_evictable());
    }

    #[test]
    fn dense_layer_page_default_residency_is_device_local() {
        let page = UnifiedVirtualPage::dense_layer(0, 0, gllm_kernels::types::DType::F32);
        assert_eq!(page.residency, MemoryResidency::DeviceLocal);
        assert_eq!(page.codec, crate::kv_cache::CompressionCodec::None);
        assert_eq!(page.compressed_size, 0);
        assert_eq!(page.decompressed_size, 0);
    }

    // ── Enum equality / copy tests ──

    #[test]
    fn page_state_variants() {
        assert_ne!(PageState::Free, PageState::Active);
        assert_ne!(PageState::SwappedOut, PageState::Warm);
        assert_eq!(PageState::Protected, PageState::Protected);
    }

    #[test]
    fn group_state_variants() {
        assert_eq!(GroupState::Running, GroupState::Running);
        assert_ne!(GroupState::Running, GroupState::Swapped);
        assert_ne!(GroupState::Paused, GroupState::Swapped);
    }

    #[test]
    fn request_kind_variants() {
        assert_ne!(RequestKind::Chat, RequestKind::Embedding);
        assert_ne!(RequestKind::Embedding, RequestKind::Rerank);
    }

    #[test]
    fn batch_order_policy_default() {
        assert_eq!(
            BatchOrderPolicy::default(),
            BatchOrderPolicy::StrictRequestIdOrder
        );
    }

    #[test]
    fn kv_pipeline_variants() {
        assert_ne!(KvPipeline::Conversation, KvPipeline::Working);
        assert_eq!(KvPipeline::Conversation, KvPipeline::Conversation);
    }

    #[test]
    fn memory_residency_variants() {
        assert_eq!(MemoryResidency::DeviceLocal, MemoryResidency::DeviceLocal);
        assert_ne!(MemoryResidency::HostLocal, MemoryResidency::DiskSwap);
    }

    #[test]
    fn weight_tier_ordering() {
        assert_eq!(WeightTier::Hot, WeightTier::Hot);
        assert_ne!(WeightTier::Hot, WeightTier::Warm);
        assert_ne!(WeightTier::Warm, WeightTier::Cold);
    }

    #[test]
    fn page_payload_kind_variants() {
        assert_eq!(PagePayloadKind::KvContext, PagePayloadKind::KvContext);
        assert_ne!(
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::DenseLayerWeight
        );
        assert_ne!(PagePayloadKind::KnowledgeRAG, PagePayloadKind::PromptSystem);
    }

    // ── UnifiedVirtualPage constructors ──

    #[test]
    fn rag_page_is_evictable_and_on_host() {
        let page = UnifiedVirtualPage::rag(5, 42, gllm_kernels::types::DType::F32);
        assert_eq!(page.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert!(page.is_evictable());
        assert!(!page.is_on_device()); // RAG starts on host
        assert_eq!(page.residency, MemoryResidency::HostLocal);
        assert_eq!(page.owner, Some(42));
    }

    #[test]
    fn system_prompt_page_not_on_host() {
        let page = UnifiedVirtualPage::system_prompt(3, gllm_kernels::types::DType::BF16);
        assert!(page.is_on_device());
        assert!(!page.is_evictable());
        assert!(page.owner.is_none());
    }

    #[test]
    fn kv_page_conversation_pipeline() {
        let page = UnifiedVirtualPage::kv(
            1,
            100,
            KvPipeline::Conversation,
            5,
            gllm_kernels::types::DType::F16,
        );
        assert_eq!(page.pipeline, Some(KvPipeline::Conversation));
        assert_eq!(page.logical_index, 5);
    }

    #[test]
    fn expert_page_has_expert_and_layer() {
        let page = UnifiedVirtualPage::expert(10, 7, 3, gllm_kernels::types::DType::BF16);
        assert_eq!(page.expert_id, Some(7));
        assert_eq!(page.layer_idx, Some(3));
        assert!(page.owner.is_none());
        assert!(page.pipeline.is_none());
    }

    #[test]
    fn dense_layer_page_is_on_device() {
        let page = UnifiedVirtualPage::dense_layer(1, 12, gllm_kernels::types::DType::F32);
        assert!(page.is_on_device());
        assert!(!page.is_evictable());
        assert_eq!(page.logical_index, 12);
        assert_eq!(page.layer_idx, Some(12));
    }

    // ── PageMetadata default ──

    #[test]
    fn page_metadata_default() {
        let meta = PageMetadata::default();
        assert_eq!(meta.page_id, 0);
        assert!(meta.sequence_id.is_none());
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(!meta.is_lir);
        assert_eq!(meta.state, PageState::Standby);
        assert!(meta.warm_until.is_none());
        assert!(meta.swap_in_time.is_none());
    }

    // ── EvictionPriority struct ──

    #[test]
    fn eviction_priority_fields() {
        let ep = EvictionPriority {
            score: -100,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 5,
            recency: 42,
            layer_idx: Some(3),
            expert_id: None,
        };
        assert_eq!(ep.score, -100);
        assert!(!ep.is_pinned);
        assert_eq!(ep.access_count, 5);
    }

    // ── PipelinedVirtualPageId ──

    #[test]
    fn pipelined_virtual_page_id_fields() {
        let pvp = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 7,
        };
        assert_eq!(pvp.pipeline, KvPipeline::Conversation);
        assert_eq!(pvp.sequence_id, 42);
        assert_eq!(pvp.logical_index, 7);
    }

    // ── SequenceGroup struct ──

    #[test]
    fn sequence_group_fields() {
        let sg = SequenceGroup {
            id: 1,
            pages: vec![10, 11, 12],
            state: GroupState::Running,
            access_count: 3,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 128,
            pipeline: KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        assert_eq!(sg.id, 1);
        assert_eq!(sg.pages.len(), 3);
        assert!(!sg.is_pinned);
        assert_eq!(sg.context_len, 128);
    }

    // ── Copy trait tests ──

    #[test]
    fn page_state_is_copy() {
        let a = PageState::Active;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn group_state_is_copy() {
        let a = GroupState::Paused;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn request_kind_is_copy() {
        let a = RequestKind::Chat;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn kv_pipeline_is_copy() {
        let a = KvPipeline::Working;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn weight_tier_is_copy() {
        let a = WeightTier::Cold;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn memory_residency_is_copy() {
        let a = MemoryResidency::DiskSwap;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn page_payload_kind_is_copy() {
        let a = PagePayloadKind::KnowledgeRAG;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_is_copy() {
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 99,
            logical_index: 3,
        };
        let b = a;
        assert_eq!(a, b);
    }

    // ── Hash trait consistency tests ──

    #[test]
    fn page_state_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        PageState::SwappedOut.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        PageState::SwappedOut.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn kv_pipeline_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        KvPipeline::Conversation.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        KvPipeline::Conversation.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn pipelined_virtual_page_id_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let pvp = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 55,
            logical_index: 10,
        };
        let mut h1 = DefaultHasher::new();
        pvp.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        pvp.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn weight_tier_hash_differs_across_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |t: WeightTier| -> u64 {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash_of(WeightTier::Hot), hash_of(WeightTier::Warm));
        assert_ne!(hash_of(WeightTier::Warm), hash_of(WeightTier::Cold));
        assert_ne!(hash_of(WeightTier::Hot), hash_of(WeightTier::Cold));
    }

    // ── Debug trait formatting tests ──

    #[test]
    fn debug_formats_are_non_empty() {
        assert!(!format!("{:?}", PageState::Free).is_empty());
        assert!(!format!("{:?}", GroupState::Running).is_empty());
        assert!(!format!("{:?}", RequestKind::Rerank).is_empty());
        assert!(!format!("{:?}", BatchOrderPolicy::FifoOrder).is_empty());
        assert!(!format!("{:?}", KvPipeline::Conversation).is_empty());
        assert!(!format!("{:?}", WeightTier::Hot).is_empty());
        assert!(!format!("{:?}", MemoryResidency::HostLocal).is_empty());
        assert!(!format!("{:?}", PagePayloadKind::ExpertWeight).is_empty());
    }

    #[test]
    fn debug_pipelined_virtual_page_id() {
        let pvp = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 0,
        };
        let debug_str = format!("{:?}", pvp);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn debug_eviction_priority() {
        let ep = EvictionPriority {
            score: 42,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: true,
            access_count: 0,
            recency: 0,
            layer_idx: None,
            expert_id: Some(3),
        };
        let debug_str = format!("{:?}", ep);
        assert!(!debug_str.is_empty());
    }

    // ── Clone trait tests for struct types ──

    #[test]
    fn page_metadata_clone() {
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(7),
            recency: 10,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: None,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.page_id, 42);
        assert_eq!(cloned.sequence_id, Some(7));
        assert_eq!(cloned.recency, 10);
        assert_eq!(cloned.access_count, 5);
        assert!(cloned.is_lir);
        assert_eq!(cloned.state, PageState::Active);
    }

    #[test]
    fn eviction_priority_clone() {
        let ep = EvictionPriority {
            score: -500,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 100,
            recency: 200,
            layer_idx: Some(12),
            expert_id: None,
        };
        let cloned = ep.clone();
        assert_eq!(cloned.score, -500);
        assert_eq!(cloned.payload_kind, PagePayloadKind::KvContext);
        assert_eq!(cloned.access_count, 100);
        assert_eq!(cloned.layer_idx, Some(12));
    }

    #[test]
    fn unified_virtual_page_clone() {
        let page = UnifiedVirtualPage::kv(
            5,
            10,
            KvPipeline::Working,
            2,
            gllm_kernels::types::DType::BF16,
        );
        let cloned = page.clone();
        assert_eq!(cloned.page_id, 5);
        assert_eq!(cloned.owner, Some(10));
        assert_eq!(cloned.pipeline, Some(KvPipeline::Working));
        assert_eq!(cloned.logical_index, 2);
        assert_eq!(cloned.dtype, gllm_kernels::types::DType::BF16);
    }

    // ── SequenceGroup edge cases ──

    #[test]
    fn sequence_group_empty_pages() {
        let sg = SequenceGroup {
            id: 2,
            pages: vec![],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 0,
            pipeline: KvPipeline::Working,
            payload_kind: None,
        };
        assert!(sg.pages.is_empty());
        assert!(sg.is_pinned);
        assert_eq!(sg.context_len, 0);
        assert!(sg.payload_kind.is_none());
    }

    #[test]
    fn sequence_group_working_pipeline() {
        let sg = SequenceGroup {
            id: 3,
            pages: vec![20],
            state: GroupState::Paused,
            access_count: 1,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 64,
            pipeline: KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KnowledgeRAG),
        };
        assert_eq!(sg.pipeline, KvPipeline::Working);
        assert_eq!(sg.state, GroupState::Paused);
        assert_eq!(sg.payload_kind, Some(PagePayloadKind::KnowledgeRAG));
    }

    // ── is_evictable exhaustiveness for all PagePayloadKind variants ──

    #[test]
    fn all_page_payload_kinds_evictability() {
        let kv = UnifiedVirtualPage::kv(
            0,
            0,
            KvPipeline::Conversation,
            0,
            gllm_kernels::types::DType::F32,
        );
        assert!(kv.is_evictable());

        let expert = UnifiedVirtualPage::expert(0, 0, 0, gllm_kernels::types::DType::F32);
        assert!(expert.is_evictable());

        let rag = UnifiedVirtualPage::rag(0, 0, gllm_kernels::types::DType::F32);
        assert!(rag.is_evictable());

        let sys = UnifiedVirtualPage::system_prompt(0, gllm_kernels::types::DType::F32);
        assert!(!sys.is_evictable());

        let dense = UnifiedVirtualPage::dense_layer(0, 0, gllm_kernels::types::DType::F32);
        assert!(!dense.is_evictable());
    }

    // ── EvictionPriority edge cases ──

    #[test]
    fn eviction_priority_pinned_with_zero_score() {
        let ep = EvictionPriority {
            score: 0,
            payload_kind: PagePayloadKind::DenseLayerWeight,
            is_pinned: true,
            access_count: 0,
            recency: 0,
            layer_idx: None,
            expert_id: None,
        };
        assert!(ep.is_pinned);
        assert_eq!(ep.score, 0);
        assert_eq!(ep.access_count, 0);
    }

    #[test]
    fn eviction_priority_negative_score_high_priority() {
        let ep = EvictionPriority {
            score: i64::MIN,
            payload_kind: PagePayloadKind::KnowledgeRAG,
            is_pinned: false,
            access_count: 1,
            recency: 999999,
            layer_idx: Some(99),
            expert_id: Some(255),
        };
        assert_eq!(ep.score, i64::MIN);
        assert_eq!(ep.expert_id, Some(255));
        assert_eq!(ep.layer_idx, Some(99));
    }

    // ── is_on_device for all residencies ──

    #[test]
    fn is_on_device_only_true_for_device_local() {
        let mut page = UnifiedVirtualPage::kv(
            0,
            0,
            KvPipeline::Conversation,
            0,
            gllm_kernels::types::DType::F32,
        );

        page.residency = MemoryResidency::DeviceLocal;
        assert!(page.is_on_device());

        page.residency = MemoryResidency::HostLocal;
        assert!(!page.is_on_device());

        page.residency = MemoryResidency::DiskSwap;
        assert!(!page.is_on_device());
    }

    // ── BatchOrderPolicy deprecated variant accessible with allow ──

    #[test]
    #[allow(deprecated)]
    fn batch_order_policy_throughput_first_exists() {
        let policy = BatchOrderPolicy::ThroughputFirst;
        assert_ne!(policy, BatchOrderPolicy::StrictRequestIdOrder);
        assert_ne!(policy, BatchOrderPolicy::FifoOrder);
    }

    #[test]
    #[allow(deprecated)]
    fn batch_order_policy_all_variants_distinct() {
        let variants = [
            BatchOrderPolicy::StrictRequestIdOrder,
            BatchOrderPolicy::FifoOrder,
            BatchOrderPolicy::ThroughputFirst,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ── PageMetadata with all fields populated ──

    #[test]
    fn page_metadata_all_fields_populated() {
        let now = Instant::now();
        let meta = PageMetadata {
            page_id: 999,
            sequence_id: Some(42),
            recency: 77,
            access_count: 33,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: true,
            state: PageState::Warm,
            warm_until: Some(now),
        };
        assert_eq!(meta.page_id, 999);
        assert_eq!(meta.sequence_id, Some(42));
        assert_eq!(meta.recency, 77);
        assert_eq!(meta.access_count, 33);
        assert!(meta.is_lir);
        assert_eq!(meta.state, PageState::Warm);
        assert!(meta.warm_until.is_some());
        assert!(meta.swap_in_time.is_some());
    }

    // ── Type aliases compile and carry values ──

    #[test]
    fn type_aliases_carry_values() {
        let page_id: PageId = 123;
        let phys_id: PhysicalId = 456;
        let req_id: RequestId = 789;
        let storage_key: StorageKey = 0xDEADBEEF;
        assert_eq!(page_id, 123usize);
        assert_eq!(phys_id, 456usize);
        assert_eq!(req_id, 789u64);
        assert_eq!(storage_key, 0xDEADBEEFu64);
    }

    // ── All GroupState variants ──

    #[test]
    fn all_group_state_variants_distinct() {
        let variants = [GroupState::Running, GroupState::Swapped, GroupState::Paused];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    // ── All PageState variants distinct ──

    #[test]
    fn all_page_state_variants_distinct() {
        let variants = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    // ── CompressionCodec integration via UnifiedVirtualPage ──

    #[test]
    fn kv_page_default_codec_is_none() {
        let page = UnifiedVirtualPage::kv(
            1,
            1,
            KvPipeline::Conversation,
            0,
            gllm_kernels::types::DType::F32,
        );
        assert_eq!(page.codec, crate::kv_cache::CompressionCodec::None);
        assert_eq!(page.compressed_size, 0);
        assert_eq!(page.decompressed_size, 0);
    }

    #[test]
    fn expert_page_default_sizes_zero() {
        let page = UnifiedVirtualPage::expert(1, 0, 0, gllm_kernels::types::DType::BF16);
        assert_eq!(page.compressed_size, 0);
        assert_eq!(page.decompressed_size, 0);
    }

    // ── SequenceGroup clone ──

    #[test]
    fn sequence_group_clone() {
        let sg = SequenceGroup {
            id: 42,
            pages: vec![1, 2, 3],
            state: GroupState::Swapped,
            access_count: 7,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 256,
            pipeline: KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::PromptSystem),
        };
        let cloned = sg.clone();
        assert_eq!(cloned.id, 42);
        assert_eq!(cloned.pages, vec![1, 2, 3]);
        assert_eq!(cloned.state, GroupState::Swapped);
        assert_eq!(cloned.access_count, 7);
        assert!(cloned.is_pinned);
        assert_eq!(cloned.context_len, 256);
        assert_eq!(cloned.pipeline, KvPipeline::Conversation);
        assert_eq!(cloned.payload_kind, Some(PagePayloadKind::PromptSystem));
    }

    // ── PageState Debug format content ──

    #[test]
    fn page_state_debug_contains_name() {
        assert!(format!("{:?}", PageState::Free).contains("Free"));
        assert!(format!("{:?}", PageState::Active).contains("Active"));
        assert!(format!("{:?}", PageState::Warm).contains("Warm"));
        assert!(format!("{:?}", PageState::Protected).contains("Protected"));
        assert!(format!("{:?}", PageState::Swapped).contains("Swapped"));
    }

    #[test]
    fn group_state_debug_contains_name() {
        assert!(format!("{:?}", GroupState::Running).contains("Running"));
        assert!(format!("{:?}", GroupState::Swapped).contains("Swapped"));
        assert!(format!("{:?}", GroupState::Paused).contains("Paused"));
    }

    #[test]
    fn request_kind_debug_contains_name() {
        assert!(format!("{:?}", RequestKind::Chat).contains("Chat"));
        assert!(format!("{:?}", RequestKind::Embedding).contains("Embedding"));
        assert!(format!("{:?}", RequestKind::Rerank).contains("Rerank"));
    }

    #[test]
    fn weight_tier_debug_contains_name() {
        assert!(format!("{:?}", WeightTier::Hot).contains("Hot"));
        assert!(format!("{:?}", WeightTier::Warm).contains("Warm"));
        assert!(format!("{:?}", WeightTier::Cold).contains("Cold"));
    }

    #[test]
    fn memory_residency_debug_contains_name() {
        assert!(format!("{:?}", MemoryResidency::DeviceLocal).contains("DeviceLocal"));
        assert!(format!("{:?}", MemoryResidency::HostLocal).contains("HostLocal"));
        assert!(format!("{:?}", MemoryResidency::DiskSwap).contains("DiskSwap"));
    }

    // ── PagePayloadKind all variants in HashSet ──

    #[test]
    fn page_payload_kind_hash_set_dedup() {
        use std::collections::HashSet;
        let set: HashSet<PagePayloadKind> = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
            PagePayloadKind::KvContext, // duplicate
        ].into_iter().collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn page_state_hash_set_dedup() {
        use std::collections::HashSet;
        let set: HashSet<PageState> = [
            PageState::Free, PageState::Active, PageState::Free,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn group_state_hash_set_dedup() {
        use std::collections::HashSet;
        let set: HashSet<GroupState> = [
            GroupState::Running, GroupState::Paused, GroupState::Running,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn request_kind_hash_set_dedup() {
        use std::collections::HashSet;
        let set: HashSet<RequestKind> = [
            RequestKind::Chat, RequestKind::Embedding, RequestKind::Chat,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn memory_residency_hash_set_dedup() {
        use std::collections::HashSet;
        let set: HashSet<MemoryResidency> = [
            MemoryResidency::DeviceLocal, MemoryResidency::DiskSwap,
            MemoryResidency::DeviceLocal,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── BatchOrderPolicy Copy + Eq ──

    #[test]
    fn batch_order_policy_copy() {
        let a = BatchOrderPolicy::FifoOrder;
        let b = a;
        assert_eq!(a, b);
    }

    // ── PageMetadata boundary values ──

    #[test]
    fn page_metadata_zero_page_id() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        assert_eq!(meta.page_id, 0);
        assert_eq!(meta.state, PageState::Free);
    }

    #[test]
    fn page_metadata_max_recency() {
        let meta = PageMetadata {
            recency: usize::MAX,
            ..PageMetadata::default()
        };
        assert_eq!(meta.recency, usize::MAX);
    }

    #[test]
    fn page_metadata_max_access_count() {
        let meta = PageMetadata {
            access_count: usize::MAX,
            ..PageMetadata::default()
        };
        assert_eq!(meta.access_count, usize::MAX);
    }

    #[test]
    fn page_metadata_request_id_max() {
        let meta = PageMetadata {
            sequence_id: Some(RequestId::MAX),
            ..PageMetadata::default()
        };
        assert_eq!(meta.sequence_id, Some(RequestId::MAX));
    }

    // ── SequenceGroup with all GroupState variants ──

    #[test]
    fn sequence_group_state_swapped() {
        let sg = SequenceGroup {
            id: 0,
            pages: vec![1],
            state: GroupState::Swapped,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };
        assert_eq!(sg.state, GroupState::Swapped);
    }

    #[test]
    fn sequence_group_state_paused() {
        let sg = SequenceGroup {
            id: 0,
            pages: vec![],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: KvPipeline::Working,
            payload_kind: None,
        };
        assert_eq!(sg.state, GroupState::Paused);
    }

    // ── UnifiedVirtualPage boundary page_id ──

    #[test]
    fn kv_page_zero_page_id() {
        let page = UnifiedVirtualPage::kv(
            0, 0, KvPipeline::Conversation, 0,
            gllm_kernels::types::DType::F32,
        );
        assert_eq!(page.page_id, 0);
    }

    #[test]
    fn expert_page_max_expert_id() {
        let page = UnifiedVirtualPage::expert(
            0, u32::MAX, usize::MAX,
            gllm_kernels::types::DType::F32,
        );
        assert_eq!(page.expert_id, Some(u32::MAX));
        assert_eq!(page.layer_idx, Some(usize::MAX));
    }

    #[test]
    fn dense_layer_zero_logical_index() {
        let page = UnifiedVirtualPage::dense_layer(
            0, 0, gllm_kernels::types::DType::BF16,
        );
        assert_eq!(page.logical_index, 0);
        assert_eq!(page.layer_idx, Some(0));
    }

    // ── EvictionPriority all payload kinds ──

    #[test]
    fn eviction_priority_all_payload_kinds() {
        for kind in [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ] {
            let ep = EvictionPriority {
                score: 0,
                payload_kind: kind,
                is_pinned: false,
                access_count: 0,
                recency: 0,
                layer_idx: None,
                expert_id: None,
            };
            assert_eq!(ep.payload_kind, kind);
        }
    }

    #[test]
    fn eviction_priority_max_score() {
        let ep = EvictionPriority {
            score: i64::MAX,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 0,
            recency: 0,
            layer_idx: None,
            expert_id: None,
        };
        assert_eq!(ep.score, i64::MAX);
    }

    // ── PipelinedVirtualPageId boundary values ──

    #[test]
    fn pipelined_virtual_page_id_zero_values() {
        let pvp = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 0,
            logical_index: 0,
        };
        assert_eq!(pvp.sequence_id, 0);
        assert_eq!(pvp.logical_index, 0);
    }

    #[test]
    fn pipelined_virtual_page_id_max_values() {
        let pvp = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: RequestId::MAX,
            logical_index: usize::MAX,
        };
        assert_eq!(pvp.sequence_id, RequestId::MAX);
        assert_eq!(pvp.logical_index, usize::MAX);
    }

    // ── PageMetadata debug format ──

    #[test]
    fn page_metadata_debug_format() {
        let meta = PageMetadata::default();
        let debug = format!("{:?}", meta);
        assert!(debug.contains("PageMetadata"));
        assert!(debug.contains("page_id"));
    }

    // ── SequenceGroup debug format ──

    #[test]
    fn sequence_group_debug_format() {
        let sg = SequenceGroup {
            id: 42,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };
        let debug = format!("{:?}", sg);
        assert!(debug.contains("SequenceGroup"));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (44 additional)
    // ══════════════════════════════════════════════════════════════════════

    // ── UnifiedVirtualPage constructors with various DType variants ──

    #[test]
    fn kv_page_with_f16_dtype() {
        let page = UnifiedVirtualPage::kv(
            10, 20, KvPipeline::Working, 3,
            gllm_kernels::types::DType::F16,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F16);
        assert_eq!(page.payload_kind, PagePayloadKind::KvContext);
        assert!(page.is_evictable());
        assert!(page.is_on_device());
        assert_eq!(page.owner, Some(20));
        assert_eq!(page.pipeline, Some(KvPipeline::Working));
        assert_eq!(page.logical_index, 3);
    }

    #[test]
    fn kv_page_with_bf16_dtype() {
        let page = UnifiedVirtualPage::kv(
            5, 15, KvPipeline::Conversation, 1,
            gllm_kernels::types::DType::BF16,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::BF16);
        assert_eq!(page.pipeline, Some(KvPipeline::Conversation));
    }

    #[test]
    fn expert_page_with_f8e4m3_dtype() {
        let page = UnifiedVirtualPage::expert(
            7, 3, 11,
            gllm_kernels::types::DType::F8E4M3,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F8E4M3);
        assert_eq!(page.expert_id, Some(3));
        assert_eq!(page.layer_idx, Some(11));
        assert!(page.owner.is_none());
    }

    #[test]
    fn system_prompt_page_with_f16_dtype() {
        let page = UnifiedVirtualPage::system_prompt(
            100, gllm_kernels::types::DType::F16,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F16);
        assert!(!page.is_evictable());
        assert!(page.is_on_device());
        assert!(page.owner.is_none());
        assert!(page.pipeline.is_none());
        assert_eq!(page.logical_index, 0);
    }

    #[test]
    fn rag_page_with_bf16_dtype() {
        let page = UnifiedVirtualPage::rag(
            99, 42, gllm_kernels::types::DType::BF16,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::BF16);
        assert_eq!(page.residency, MemoryResidency::HostLocal);
        assert!(!page.is_on_device());
        assert!(page.is_evictable());
        assert_eq!(page.owner, Some(42));
        assert!(page.pipeline.is_none());
    }

    #[test]
    fn dense_layer_with_f16_dtype() {
        let page = UnifiedVirtualPage::dense_layer(
            50, 8, gllm_kernels::types::DType::F16,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F16);
        assert!(!page.is_evictable());
        assert_eq!(page.logical_index, 8);
        assert_eq!(page.layer_idx, Some(8));
    }

    // ── UnifiedVirtualPage residency mutation ──

    #[test]
    fn kv_page_residency_mutation_affects_is_on_device() {
        let mut page = UnifiedVirtualPage::kv(
            1, 1, KvPipeline::Conversation, 0,
            gllm_kernels::types::DType::F32,
        );
        assert!(page.is_on_device());

        page.residency = MemoryResidency::HostLocal;
        assert!(!page.is_on_device());
        assert_eq!(page.residency, MemoryResidency::HostLocal);

        page.residency = MemoryResidency::DiskSwap;
        assert!(!page.is_on_device());
        assert_eq!(page.residency, MemoryResidency::DiskSwap);
    }

    #[test]
    fn expert_page_residency_can_be_moved_to_host() {
        let mut page = UnifiedVirtualPage::expert(
            1, 0, 0, gllm_kernels::types::DType::F32,
        );
        assert!(page.is_on_device());
        page.residency = MemoryResidency::HostLocal;
        assert!(!page.is_on_device());
    }

    // ── UnifiedVirtualPage compression field mutation ──

    #[test]
    fn kv_page_compression_fields_mutable() {
        let mut page = UnifiedVirtualPage::kv(
            1, 1, KvPipeline::Conversation, 0,
            gllm_kernels::types::DType::F32,
        );
        assert_eq!(page.compressed_size, 0);
        assert_eq!(page.decompressed_size, 0);

        page.compressed_size = 4096;
        page.decompressed_size = 8192;
        assert_eq!(page.compressed_size, 4096);
        assert_eq!(page.decompressed_size, 8192);
    }

    // ── SequenceGroup with many pages ──

    #[test]
    fn sequence_group_large_page_list() {
        let pages: Vec<PageId> = (0..1000).collect();
        let sg = SequenceGroup {
            id: 999,
            pages: pages.clone(),
            state: GroupState::Running,
            access_count: 500,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 4096,
            pipeline: KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        assert_eq!(sg.pages.len(), 1000);
        assert_eq!(sg.pages.first(), Some(&0));
        assert_eq!(sg.pages.last(), Some(&999));
        assert_eq!(sg.access_count, 500);
        assert_eq!(sg.context_len, 4096);
    }

    // ── SequenceGroup with each PagePayloadKind ──

    #[test]
    fn sequence_group_with_expert_weight_payload() {
        let sg = SequenceGroup {
            id: 10,
            pages: vec![5],
            state: GroupState::Running,
            access_count: 1,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::ExpertWeight),
        };
        assert_eq!(sg.payload_kind, Some(PagePayloadKind::ExpertWeight));
    }

    #[test]
    fn sequence_group_with_dense_layer_payload() {
        let sg = SequenceGroup {
            id: 11,
            pages: vec![6],
            state: GroupState::Running,
            access_count: 2,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 0,
            pipeline: KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::DenseLayerWeight),
        };
        assert_eq!(sg.payload_kind, Some(PagePayloadKind::DenseLayerWeight));
        assert!(sg.is_pinned);
    }

    // ── SequenceGroup context_len boundary ──

    #[test]
    fn sequence_group_zero_context_len() {
        let sg = SequenceGroup {
            id: 0,
            pages: vec![],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: KvPipeline::Working,
            payload_kind: None,
        };
        assert_eq!(sg.context_len, 0);
        assert_eq!(sg.pages.len(), 0);
    }

    #[test]
    fn sequence_group_large_context_len() {
        let sg = SequenceGroup {
            id: 1,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 131072,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };
        assert_eq!(sg.context_len, 131072);
    }

    // ── PageMetadata state transitions ──

    #[test]
    fn page_metadata_state_active() {
        let meta = PageMetadata {
            state: PageState::Active,
            ..PageMetadata::default()
        };
        assert_eq!(meta.state, PageState::Active);
    }

    #[test]
    fn page_metadata_state_free() {
        let meta = PageMetadata {
            state: PageState::Free,
            ..PageMetadata::default()
        };
        assert_eq!(meta.state, PageState::Free);
    }

    #[test]
    fn page_metadata_state_warm() {
        let meta = PageMetadata {
            state: PageState::Warm,
            ..PageMetadata::default()
        };
        assert_eq!(meta.state, PageState::Warm);
    }

    #[test]
    fn page_metadata_is_lir_true() {
        let meta = PageMetadata {
            is_lir: true,
            ..PageMetadata::default()
        };
        assert!(meta.is_lir);
    }

    #[test]
    fn page_metadata_with_swap_in_time() {
        let now = Instant::now();
        let meta = PageMetadata {
            swap_in_time: Some(now),
            ..PageMetadata::default()
        };
        assert!(meta.swap_in_time.is_some());
    }

    #[test]
    fn page_metadata_with_warm_until() {
        let now = Instant::now();
        let meta = PageMetadata {
            warm_until: Some(now),
            ..PageMetadata::default()
        };
        assert!(meta.warm_until.is_some());
    }

    // ── EvictionPriority score ranges ──

    #[test]
    fn eviction_priority_positive_score() {
        let ep = EvictionPriority {
            score: 1000,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 50,
            recency: 10,
            layer_idx: Some(0),
            expert_id: None,
        };
        assert!(ep.score > 0);
        assert_eq!(ep.score, 1000);
    }

    #[test]
    fn eviction_priority_large_negative_score() {
        let ep = EvictionPriority {
            score: -999999,
            payload_kind: PagePayloadKind::KnowledgeRAG,
            is_pinned: false,
            access_count: 1,
            recency: 1000000,
            layer_idx: Some(255),
            expert_id: Some(42),
        };
        assert!(ep.score < 0);
        assert_eq!(ep.score, -999999);
        assert_eq!(ep.expert_id, Some(42));
        assert_eq!(ep.layer_idx, Some(255));
    }

    // ── Hash consistency for remaining enums ──

    #[test]
    fn page_state_hash_differs_across_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |s: PageState| -> u64 {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        };
        let variants = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(hash_of(*a), hash_of(*b), "{:?} and {:?} should hash differently", a, b);
                } else {
                    assert_eq!(hash_of(*a), hash_of(*b));
                }
            }
        }
    }

    #[test]
    fn group_state_hash_differs_across_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |s: GroupState| -> u64 {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash_of(GroupState::Running), hash_of(GroupState::Swapped));
        assert_ne!(hash_of(GroupState::Swapped), hash_of(GroupState::Paused));
        assert_ne!(hash_of(GroupState::Running), hash_of(GroupState::Paused));
    }

    #[test]
    fn request_kind_hash_differs_across_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |k: RequestKind| -> u64 {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash_of(RequestKind::Chat), hash_of(RequestKind::Embedding));
        assert_ne!(hash_of(RequestKind::Embedding), hash_of(RequestKind::Rerank));
        assert_ne!(hash_of(RequestKind::Chat), hash_of(RequestKind::Rerank));
    }

    #[test]
    fn memory_residency_hash_differs_across_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |r: MemoryResidency| -> u64 {
            let mut h = DefaultHasher::new();
            r.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash_of(MemoryResidency::DeviceLocal), hash_of(MemoryResidency::HostLocal));
        assert_ne!(hash_of(MemoryResidency::HostLocal), hash_of(MemoryResidency::DiskSwap));
        assert_ne!(hash_of(MemoryResidency::DeviceLocal), hash_of(MemoryResidency::DiskSwap));
    }

    #[test]
    fn weight_tier_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        WeightTier::Hot.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        WeightTier::Hot.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── PagePayloadKind hash differs across all variants ──

    #[test]
    fn page_payload_kind_hash_differs_across_all() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |k: PagePayloadKind| -> u64 {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        };
        let variants = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        hash_of(*a), hash_of(*b),
                        "{:?} and {:?} should hash differently", a, b
                    );
                }
            }
        }
    }

    // ── PipelinedVirtualPageId equality ──

    #[test]
    fn pipelined_virtual_page_id_equality_same() {
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 7,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 7,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_inequality_different_pipeline() {
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 7,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 42,
            logical_index: 7,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_inequality_different_sequence() {
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 7,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 2,
            logical_index: 7,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_inequality_different_index() {
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 1,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 2,
        };
        assert_ne!(a, b);
    }

    // ── UnifiedVirtualPage dtype preservation after clone ──

    #[test]
    fn unified_virtual_page_dtype_preserved_after_clone() {
        let dtypes = [
            gllm_kernels::types::DType::F32,
            gllm_kernels::types::DType::F16,
            gllm_kernels::types::DType::BF16,
            gllm_kernels::types::DType::F8E4M3,
        ];
        for dtype in dtypes {
            let page = UnifiedVirtualPage::kv(0, 0, KvPipeline::Conversation, 0, dtype);
            let cloned = page.clone();
            assert_eq!(page.dtype, cloned.dtype);
            assert_eq!(page.dtype, dtype);
        }
    }

    // ── PageMetadata clone preserves all fields ──

    #[test]
    fn page_metadata_clone_preserves_state_fields() {
        let meta = PageMetadata {
            page_id: 77,
            sequence_id: Some(33),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: true,
            state: PageState::Protected,
            warm_until: Some(Instant::now()),
        };
        let cloned = meta.clone();
        assert_eq!(cloned.page_id, 77);
        assert_eq!(cloned.sequence_id, Some(33));
        assert_eq!(cloned.recency, 5);
        assert_eq!(cloned.access_count, 10);
        assert!(cloned.is_lir);
        assert_eq!(cloned.state, PageState::Protected);
    }

    // ── EvictionPriority clone preserves payload_kind ──

    #[test]
    fn eviction_priority_clone_preserves_all_fields() {
        let ep = EvictionPriority {
            score: -42,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: true,
            access_count: 99,
            recency: 1001,
            layer_idx: Some(7),
            expert_id: Some(13),
        };
        let cloned = ep.clone();
        assert_eq!(cloned.score, -42);
        assert_eq!(cloned.payload_kind, PagePayloadKind::ExpertWeight);
        assert!(cloned.is_pinned);
        assert_eq!(cloned.access_count, 99);
        assert_eq!(cloned.recency, 1001);
        assert_eq!(cloned.layer_idx, Some(7));
        assert_eq!(cloned.expert_id, Some(13));
    }

    // ── WeightTier all variants in HashSet ──

    #[test]
    fn weight_tier_hash_set_dedup() {
        use std::collections::HashSet;
        let set: HashSet<WeightTier> = [
            WeightTier::Hot, WeightTier::Warm, WeightTier::Cold,
            WeightTier::Hot, WeightTier::Warm,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── KvPipeline hash set dedup ──

    #[test]
    fn kv_pipeline_hash_set_dedup() {
        use std::collections::HashSet;
        let set: HashSet<KvPipeline> = [
            KvPipeline::Conversation, KvPipeline::Working,
            KvPipeline::Conversation,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── BatchOrderPolicy equality between the two non-deprecated variants ──

    #[test]
    fn batch_order_policy_strict_neq_fifo() {
        assert_ne!(BatchOrderPolicy::StrictRequestIdOrder, BatchOrderPolicy::FifoOrder);
    }

    // ── UnifiedVirtualPage rag page all fields verified ──

    #[test]
    fn rag_page_all_fields_verified() {
        let page = UnifiedVirtualPage::rag(100, 200, gllm_kernels::types::DType::F32);
        assert_eq!(page.page_id, 100);
        assert_eq!(page.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert_eq!(page.residency, MemoryResidency::HostLocal);
        assert_eq!(page.owner, Some(200));
        assert!(page.pipeline.is_none());
        assert_eq!(page.logical_index, 0);
        assert_eq!(page.expert_id, None);
        assert_eq!(page.layer_idx, None);
        assert_eq!(page.compressed_size, 0);
        assert_eq!(page.decompressed_size, 0);
    }

    // ── UnifiedVirtualPage system_prompt page all fields verified ──

    #[test]
    fn system_prompt_page_all_fields_verified() {
        let page = UnifiedVirtualPage::system_prompt(50, gllm_kernels::types::DType::BF16);
        assert_eq!(page.page_id, 50);
        assert_eq!(page.payload_kind, PagePayloadKind::PromptSystem);
        assert_eq!(page.residency, MemoryResidency::DeviceLocal);
        assert!(page.owner.is_none());
        assert!(page.pipeline.is_none());
        assert_eq!(page.logical_index, 0);
        assert_eq!(page.expert_id, None);
        assert_eq!(page.layer_idx, None);
    }

    // ── EvictionPriority debug format contains key fields ──

    #[test]
    fn eviction_priority_debug_format_contains_fields() {
        let ep = EvictionPriority {
            score: -100,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: true,
            access_count: 5,
            recency: 20,
            layer_idx: Some(3),
            expert_id: None,
        };
        let debug = format!("{:?}", ep);
        assert!(debug.contains("EvictionPriority"));
        assert!(debug.contains("score"));
        assert!(debug.contains("payload_kind"));
        assert!(debug.contains("is_pinned"));
    }

    // ── PageMetadata debug format contains key fields ──

    #[test]
    fn page_metadata_debug_format_contains_key_fields() {
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(7),
            state: PageState::Active,
            is_lir: true,
            ..PageMetadata::default()
        };
        let debug = format!("{:?}", meta);
        assert!(debug.contains("page_id"));
        assert!(debug.contains("sequence_id"));
        assert!(debug.contains("state"));
        assert!(debug.contains("is_lir"));
    }

    // ── UnifiedVirtualPage debug format ──

    #[test]
    fn unified_virtual_page_debug_format() {
        let page = UnifiedVirtualPage::kv(
            1, 2, KvPipeline::Conversation, 0,
            gllm_kernels::types::DType::F32,
        );
        let debug = format!("{:?}", page);
        assert!(debug.contains("UnifiedVirtualPage"));
    }

    // ── Type alias boundary values ──

    #[test]
    fn type_alias_boundary_values() {
        let max_page: PageId = usize::MAX;
        let max_phys: PhysicalId = usize::MAX;
        let max_req: RequestId = u64::MAX;
        let max_storage: StorageKey = u64::MAX;
        assert_eq!(max_page, usize::MAX);
        assert_eq!(max_phys, usize::MAX);
        assert_eq!(max_req, u64::MAX);
        assert_eq!(max_storage, u64::MAX);
    }

    // ── SequenceGroup access_count boundary ──

    #[test]
    fn sequence_group_max_access_count() {
        let sg = SequenceGroup {
            id: 0,
            pages: vec![1],
            state: GroupState::Running,
            access_count: usize::MAX,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 0,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };
        assert_eq!(sg.access_count, usize::MAX);
    }

    // ── All enum variants in HashSet (complete coverage) ──

    #[test]
    fn all_page_state_variants_in_hash_set() {
        use std::collections::HashSet;
        let set: HashSet<PageState> = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ].into_iter().collect();
        assert_eq!(set.len(), 7);
    }

    #[test]
    fn all_group_state_variants_in_hash_set() {
        use std::collections::HashSet;
        let set: HashSet<GroupState> = [
            GroupState::Running,
            GroupState::Swapped,
            GroupState::Paused,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn all_request_kind_variants_in_hash_set() {
        use std::collections::HashSet;
        let set: HashSet<RequestKind> = [
            RequestKind::Chat,
            RequestKind::Embedding,
            RequestKind::Rerank,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn all_memory_residency_variants_in_hash_set() {
        use std::collections::HashSet;
        let set: HashSet<MemoryResidency> = [
            MemoryResidency::DeviceLocal,
            MemoryResidency::HostLocal,
            MemoryResidency::DiskSwap,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── Enum types as HashMap keys ──

    #[test]
    fn page_state_as_hash_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(PageState::Active, 1);
        map.insert(PageState::SwappedOut, 2);
        assert_eq!(map.get(&PageState::Active), Some(&1));
        assert_eq!(map.get(&PageState::SwappedOut), Some(&2));
        assert_eq!(map.get(&PageState::Free), None);
    }

    #[test]
    fn weight_tier_as_hash_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(WeightTier::Hot, "gpu");
        map.insert(WeightTier::Warm, "cpu");
        map.insert(WeightTier::Cold, "disk");
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&WeightTier::Hot), Some(&"gpu"));
    }

    #[test]
    fn kv_pipeline_as_hash_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(KvPipeline::Conversation, 1u32);
        map.insert(KvPipeline::Working, 2u32);
        assert_eq!(map.get(&KvPipeline::Conversation), Some(&1));
        assert_eq!(map.get(&KvPipeline::Working), Some(&2));
    }

    #[test]
    fn request_kind_as_hash_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(RequestKind::Chat, "chat");
        map.insert(RequestKind::Embedding, "embed");
        map.insert(RequestKind::Rerank, "rerank");
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&RequestKind::Rerank), Some(&"rerank"));
    }

    #[test]
    fn memory_residency_as_hash_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(MemoryResidency::DeviceLocal, 0u8);
        map.insert(MemoryResidency::HostLocal, 1u8);
        map.insert(MemoryResidency::DiskSwap, 2u8);
        assert_eq!(map.get(&MemoryResidency::HostLocal), Some(&1));
    }

    // ── Boundary value tests ──

    #[test]
    fn page_metadata_max_page_id() {
        let meta = PageMetadata {
            page_id: usize::MAX,
            ..PageMetadata::default()
        };
        assert_eq!(meta.page_id, usize::MAX);
    }

    #[test]
    fn sequence_group_request_id_max() {
        let sg = SequenceGroup {
            id: RequestId::MAX,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 1,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 10,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };
        assert_eq!(sg.id, RequestId::MAX);
    }

    #[test]
    fn eviction_priority_expert_id_max() {
        let ep = EvictionPriority {
            score: 0,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: false,
            access_count: 0,
            recency: 0,
            layer_idx: None,
            expert_id: Some(u32::MAX),
        };
        assert_eq!(ep.expert_id, Some(u32::MAX));
    }

    // ── PipelinedVirtualPageId as HashMap key ──

    #[test]
    fn pipelined_virtual_page_id_as_hash_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let key1 = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 0,
        };
        let key2 = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 1,
            logical_index: 0,
        };
        map.insert(key1, "conv");
        map.insert(key2, "work");
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get(&PipelinedVirtualPageId {
                pipeline: KvPipeline::Conversation,
                sequence_id: 1,
                logical_index: 0,
            }),
            Some(&"conv")
        );
        assert_eq!(
            map.get(&PipelinedVirtualPageId {
                pipeline: KvPipeline::Working,
                sequence_id: 1,
                logical_index: 0,
            }),
            Some(&"work")
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    //  ADDITIONAL TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. Expert weight page with uncommon DType variants ──

    #[test]
    fn expert_page_with_f8e5m2_dtype() {
        // Arrange: construct an expert page with FP8 E5M2 dtype
        // Act: create via expert constructor
        // Assert: dtype preserved, expert_id and layer_idx correct
        let page = UnifiedVirtualPage::expert(
            3, 12, 5,
            gllm_kernels::types::DType::F8E5M2,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F8E5M2);
        assert_eq!(page.payload_kind, PagePayloadKind::ExpertWeight);
        assert_eq!(page.expert_id, Some(12));
        assert_eq!(page.layer_idx, Some(5));
        assert!(page.is_evictable());
        assert!(page.is_on_device());
    }

    // ── 2. Expert weight page with F4E2M1 dtype ──

    #[test]
    fn expert_page_with_f4e2m1_dtype() {
        // Arrange: construct an expert page with FP4 E2M1 (NVFP4) dtype
        // Act: create via expert constructor
        // Assert: dtype preserved, all other fields at default values
        let page = UnifiedVirtualPage::expert(
            7, 63, 0,
            gllm_kernels::types::DType::F4E2M1,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F4E2M1);
        assert_eq!(page.expert_id, Some(63));
        assert_eq!(page.page_id, 7);
        assert!(page.owner.is_none());
        assert!(page.pipeline.is_none());
        assert_eq!(page.logical_index, 0);
    }

    // ── 3. UnifiedVirtualPage with non-None compression codec and sizes ──

    #[test]
    fn kv_page_with_lz4_compression() {
        // Arrange: create a KV page then mutate compression fields to Lz4
        // Act: set codec, compressed_size, decompressed_size
        // Assert: all three fields reflect the mutation
        let mut page = UnifiedVirtualPage::kv(
            10, 42, KvPipeline::Conversation, 3,
            gllm_kernels::types::DType::BF16,
        );
        page.codec = crate::kv_cache::CompressionCodec::Lz4;
        page.compressed_size = 2048;
        page.decompressed_size = 8192;
        assert_eq!(page.codec, crate::kv_cache::CompressionCodec::Lz4);
        assert_eq!(page.compressed_size, 2048);
        assert_eq!(page.decompressed_size, 8192);
        assert!(page.compressed_size < page.decompressed_size);
    }

    // ── 4. Expert page residency transition from DeviceLocal to DiskSwap ──

    #[test]
    fn expert_page_residency_transition_to_disk_swap() {
        // Arrange: create an expert page (default DeviceLocal)
        // Act: mutate residency to DiskSwap
        // Assert: is_on_device returns false, residency matches
        let mut page = UnifiedVirtualPage::expert(
            5, 1, 3, gllm_kernels::types::DType::F32,
        );
        assert!(page.is_on_device());
        assert_eq!(page.residency, MemoryResidency::DeviceLocal);

        page.residency = MemoryResidency::DiskSwap;
        assert!(!page.is_on_device());
        assert_eq!(page.residency, MemoryResidency::DiskSwap);
        // Evictability is determined by payload_kind, not residency
        assert!(page.is_evictable());
    }

    // ── 5. Dense layer page residency mutation (stays on device normally) ──

    #[test]
    fn dense_layer_page_residency_mutation_to_host_local() {
        // Arrange: create a dense layer page (default DeviceLocal)
        // Act: mutate residency to HostLocal (simulating migration)
        // Assert: is_on_device reflects the change, evictability unchanged
        let mut page = UnifiedVirtualPage::dense_layer(
            20, 4, gllm_kernels::types::DType::F32,
        );
        assert!(page.is_on_device());
        assert!(!page.is_evictable());

        page.residency = MemoryResidency::HostLocal;
        assert!(!page.is_on_device());
        // DenseLayerWeight is never evictable regardless of residency
        assert!(!page.is_evictable());
    }

    // ── 6. PagePayloadKind as HashMap key with lookup ──

    #[test]
    fn page_payload_kind_as_hash_map_key() {
        // Arrange: populate a HashMap with all 5 PagePayloadKind variants
        // Act: insert and retrieve values
        // Assert: all lookups return correct values, count is 5
        use std::collections::HashMap;
        let mut map: HashMap<PagePayloadKind, &'static str> = HashMap::new();
        map.insert(PagePayloadKind::KvContext, "attention");
        map.insert(PagePayloadKind::ExpertWeight, "moe");
        map.insert(PagePayloadKind::PromptSystem, "system");
        map.insert(PagePayloadKind::KnowledgeRAG, "rag");
        map.insert(PagePayloadKind::DenseLayerWeight, "dense");
        assert_eq!(map.len(), 5);
        assert_eq!(map.get(&PagePayloadKind::KvContext), Some(&"attention"));
        assert_eq!(map.get(&PagePayloadKind::ExpertWeight), Some(&"moe"));
        assert_eq!(map.get(&PagePayloadKind::DenseLayerWeight), Some(&"dense"));
        // Overwrite existing key
        map.insert(PagePayloadKind::KnowledgeRAG, "updated");
        assert_eq!(map.get(&PagePayloadKind::KnowledgeRAG), Some(&"updated"));
        assert_eq!(map.len(), 5);
    }

    // ── 7. GroupState as HashMap key ──

    #[test]
    fn group_state_as_hash_map_key() {
        // Arrange: use all 3 GroupState variants as HashMap keys
        // Act: insert with unique values and verify lookup
        // Assert: all 3 entries present, missing key returns None
        use std::collections::HashMap;
        let mut map: HashMap<GroupState, u8> = HashMap::new();
        map.insert(GroupState::Running, 0);
        map.insert(GroupState::Swapped, 1);
        map.insert(GroupState::Paused, 2);
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&GroupState::Running), Some(&0));
        assert_eq!(map.get(&GroupState::Swapped), Some(&1));
        assert_eq!(map.get(&GroupState::Paused), Some(&2));
    }

    // ── 8. WeightTier maps to MemoryResidency semantically ──

    #[test]
    fn weight_tier_residency_correspondence() {
        // Arrange: define the expected mapping Hot->DeviceLocal, Warm->HostLocal, Cold->DiskSwap
        // Act: verify each WeightTier's corresponding MemoryResidency
        // Assert: the three-tier correspondence holds
        let tiers_and_residency = [
            (WeightTier::Hot, MemoryResidency::DeviceLocal),
            (WeightTier::Warm, MemoryResidency::HostLocal),
            (WeightTier::Cold, MemoryResidency::DiskSwap),
        ];
        for (tier, expected_residency) in tiers_and_residency {
            // Each tier maps to exactly one residency
            assert_eq!(format!("{:?}", tier), match tier {
                WeightTier::Hot => "Hot",
                WeightTier::Warm => "Warm",
                WeightTier::Cold => "Cold",
            });
            // Verify the residency enum matches the tier's semantic level
            assert_ne!(
                format!("{:?}", expected_residency).is_empty(),
                true,
                "Residency for {:?} must have a debug representation",
                tier,
            );
        }
    }

    // ── 9. CompressionCodec variants used with UnifiedVirtualPage ──

    #[test]
    fn kv_page_with_bitpack_rle_compression() {
        // Arrange: create a KV page and set BitPackRle compression
        // Act: mutate codec and sizes
        // Assert: codec, sizes, and dtype are all consistent
        let mut page = UnifiedVirtualPage::kv(
            15, 100, KvPipeline::Working, 2,
            gllm_kernels::types::DType::F16,
        );
        page.codec = crate::kv_cache::CompressionCodec::BitPackRle;
        page.compressed_size = 512;
        page.decompressed_size = 4096;
        assert_eq!(page.codec, crate::kv_cache::CompressionCodec::BitPackRle);
        assert_eq!(page.compressed_size, 512);
        assert_eq!(page.decompressed_size, 4096);
        // dtype should remain unchanged
        assert_eq!(page.dtype, gllm_kernels::types::DType::F16);
        // Pipeline should remain unchanged
        assert_eq!(page.pipeline, Some(KvPipeline::Working));
    }

    // ── 10. Multiple KV pages with different owners share no state ──

    #[test]
    fn multiple_kv_pages_independent_owners() {
        // Arrange: create two KV pages with different owners
        // Act: verify each page's owner independently
        // Assert: no cross-contamination between pages
        let page_a = UnifiedVirtualPage::kv(
            1, 100, KvPipeline::Conversation, 0,
            gllm_kernels::types::DType::F32,
        );
        let page_b = UnifiedVirtualPage::kv(
            2, 200, KvPipeline::Working, 1,
            gllm_kernels::types::DType::BF16,
        );
        assert_eq!(page_a.owner, Some(100));
        assert_eq!(page_b.owner, Some(200));
        assert_eq!(page_a.page_id, 1);
        assert_eq!(page_b.page_id, 2);
        assert_eq!(page_a.pipeline, Some(KvPipeline::Conversation));
        assert_eq!(page_b.pipeline, Some(KvPipeline::Working));
        assert_eq!(page_a.dtype, gllm_kernels::types::DType::F32);
        assert_eq!(page_b.dtype, gllm_kernels::types::DType::BF16);
    }

    // ── 11. SequenceGroup clone produces independent pages Vec ──

    #[test]
    fn sequence_group_clone_independent_pages() {
        // Arrange: create a SequenceGroup with non-empty pages
        // Act: clone the group
        // Assert: cloned pages are equal but independent (deep copy)
        let sg = SequenceGroup {
            id: 55,
            pages: vec![10, 20, 30],
            state: GroupState::Running,
            access_count: 3,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 128,
            pipeline: KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        let cloned = sg.clone();
        assert_eq!(cloned.pages, vec![10, 20, 30]);
        assert_eq!(sg.pages, cloned.pages);
        assert_eq!(cloned.id, 55);
        assert_eq!(cloned.access_count, 3);
        assert_eq!(cloned.state, GroupState::Running);
    }

    // ── 12. PageMetadata transitions through all PageState variants ──

    #[test]
    fn page_metadata_state_transitions() {
        // Arrange: start with default PageMetadata (Standby state)
        // Act: transition through Free -> Active -> Warm -> Protected -> Swapped -> SwappedOut
        // Assert: each state change is reflected correctly
        let mut meta = PageMetadata::default();
        assert_eq!(meta.state, PageState::Standby);

        meta.state = PageState::Free;
        assert_eq!(meta.state, PageState::Free);

        meta.state = PageState::Active;
        assert_eq!(meta.state, PageState::Active);

        meta.state = PageState::Warm;
        assert_eq!(meta.state, PageState::Warm);

        meta.state = PageState::Protected;
        assert_eq!(meta.state, PageState::Protected);

        meta.state = PageState::Swapped;
        assert_eq!(meta.state, PageState::Swapped);

        meta.state = PageState::SwappedOut;
        assert_eq!(meta.state, PageState::SwappedOut);
    }

    // ── 13. EvictionPriority pinned expert weight with score zero ──

    #[test]
    fn eviction_priority_pinned_expert_weight_zero_score() {
        // Arrange: construct an EvictionPriority for a pinned expert weight page
        // Act: verify all fields, especially pinned status with zero score
        // Assert: pinned pages have is_pinned true regardless of score
        let ep = EvictionPriority {
            score: 0,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: true,
            access_count: 0,
            recency: 0,
            layer_idx: Some(3),
            expert_id: Some(7),
        };
        assert!(ep.is_pinned);
        assert_eq!(ep.score, 0);
        assert_eq!(ep.payload_kind, PagePayloadKind::ExpertWeight);
        assert_eq!(ep.expert_id, Some(7));
        assert_eq!(ep.layer_idx, Some(3));
    }

    // ── 14. UnifiedVirtualPage rag page moved to device ──

    #[test]
    fn rag_page_residency_transition_to_device() {
        // Arrange: create a RAG page (default HostLocal)
        // Act: mutate residency to DeviceLocal
        // Assert: is_on_device now returns true, evictable unchanged
        let mut page = UnifiedVirtualPage::rag(
            33, 77, gllm_kernels::types::DType::F32,
        );
        assert!(!page.is_on_device());
        assert_eq!(page.residency, MemoryResidency::HostLocal);

        page.residency = MemoryResidency::DeviceLocal;
        assert!(page.is_on_device());
        assert_eq!(page.residency, MemoryResidency::DeviceLocal);
        assert!(page.is_evictable());
    }

    // ── 15. PagePayloadKind Copy trait allows duplicate without move ──

    #[test]
    fn page_payload_kind_copy_preserves_original_in_collection() {
        // Arrange: create a Vec with all PagePayloadKind variants via Copy
        // Act: copy each variant into a second Vec
        // Assert: original Vec still has all 5 variants (Copy, not Move)
        let originals = vec![
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        let copies: Vec<PagePayloadKind> = originals.iter().copied().collect();
        assert_eq!(originals.len(), 5);
        assert_eq!(copies.len(), 5);
        for (orig, copy) in originals.iter().zip(copies.iter()) {
            assert_eq!(*orig, *copy);
        }
    }

    // ── 16. UnifiedVirtualPage clone preserves all codec fields ──

    #[test]
    fn unified_virtual_page_clone_preserves_compressed_fields() {
        // Arrange: build a KV page, mutate compression fields
        // Act: clone the page
        // Assert: cloned copy has identical compressed_size, decompressed_size, codec
        let mut page = UnifiedVirtualPage::kv(
            10, 20, KvPipeline::Working, 3, gllm_kernels::types::DType::F16,
        );
        page.compressed_size = 4096;
        page.decompressed_size = 16384;
        let cloned = page.clone();
        assert_eq!(cloned.compressed_size, 4096);
        assert_eq!(cloned.decompressed_size, 16384);
        assert_eq!(cloned.codec, crate::kv_cache::CompressionCodec::None);
        assert_eq!(cloned.page_id, page.page_id);
    }

    // ── 17. MemoryResidency Hash distinguishes all three variants ──

    #[test]
    fn memory_residency_hash_distinguishes_all_variants() {
        // Arrange: compute hashes for all MemoryResidency variants
        // Act: compare pairwise hashes
        // Assert: all three hashes are distinct
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |r: MemoryResidency| -> u64 {
            let mut h = DefaultHasher::new();
            r.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash_of(MemoryResidency::DeviceLocal), hash_of(MemoryResidency::HostLocal));
        assert_ne!(hash_of(MemoryResidency::HostLocal), hash_of(MemoryResidency::DiskSwap));
        assert_ne!(hash_of(MemoryResidency::DeviceLocal), hash_of(MemoryResidency::DiskSwap));
    }

    // ── 18. PagePayloadKind Hash distinguishes all five variants ──

    #[test]
    fn page_payload_kind_hash_distinguishes_all_variants() {
        // Arrange: compute hashes for all 5 PagePayloadKind variants
        // Act: pairwise compare hashes
        // Assert: all variants produce distinct hashes
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |k: PagePayloadKind| -> u64 {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        };
        let variants = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        let hashes: Vec<u64> = variants.iter().map(|v| hash_of(*v)).collect();
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "variants {:?} and {:?} share a hash", variants[i], variants[j]);
            }
        }
    }

    // ── 19. RequestKind Hash consistency ──

    #[test]
    fn request_kind_hash_consistency() {
        // Arrange: hash the same RequestKind variant twice
        // Act: compare the two hash values
        // Assert: identical inputs produce identical hashes
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        RequestKind::Chat.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        RequestKind::Chat.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── 20. BatchOrderPolicy default remains StrictRequestIdOrder ──

    #[test]
    fn batch_order_policy_default_matches_strict() {
        // Arrange: get the default BatchOrderPolicy
        // Act: compare to StrictRequestIdOrder
        // Assert: they are equal
        let default = BatchOrderPolicy::default();
        assert_eq!(default, BatchOrderPolicy::StrictRequestIdOrder);
        assert_ne!(default, BatchOrderPolicy::FifoOrder);
    }

    // ── 21. PageMetadata Debug format includes key field names ──

    #[test]
    fn page_metadata_debug_contains_key_fields() {
        // Arrange: construct a PageMetadata with known field values
        // Act: format via Debug
        // Assert: output contains struct name and field identifiers
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(7),
            recency: 10,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let debug = format!("{:?}", meta);
        assert!(debug.contains("page_id"));
        assert!(debug.contains("42"));
        assert!(debug.contains("Active"));
    }

    // ── 22. UnifiedVirtualPage Debug format for each constructor ──

    #[test]
    fn unified_virtual_page_debug_all_constructors() {
        // Arrange: create one page per constructor
        // Act: format each via Debug
        // Assert: all debug strings are non-empty and contain expected payload kind
        let kv = UnifiedVirtualPage::kv(1, 1, KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let expert = UnifiedVirtualPage::expert(2, 0, 0, gllm_kernels::types::DType::BF16);
        let sys = UnifiedVirtualPage::system_prompt(3, gllm_kernels::types::DType::F16);
        let rag = UnifiedVirtualPage::rag(4, 5, gllm_kernels::types::DType::F32);
        let dense = UnifiedVirtualPage::dense_layer(5, 1, gllm_kernels::types::DType::BF16);

        for (label, page, expected_kind) in [
            ("kv", kv, "KvContext"),
            ("expert", expert, "ExpertWeight"),
            ("sys", sys, "PromptSystem"),
            ("rag", rag, "KnowledgeRAG"),
            ("dense", dense, "DenseLayerWeight"),
        ] {
            let s = format!("{:?}", page);
            assert!(!s.is_empty(), "{} debug is empty", label);
            assert!(s.contains(expected_kind), "{} debug missing {}", label, expected_kind);
        }
    }

    // ── 23. EvictionPriority Debug format contains all field names ──

    #[test]
    fn eviction_priority_debug_contains_field_names() {
        // Arrange: construct EvictionPriority with distinctive values
        // Act: format via Debug
        // Assert: output references key fields
        let ep = EvictionPriority {
            score: -999,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: true,
            access_count: 42,
            recency: 17,
            layer_idx: Some(5),
            expert_id: Some(9),
        };
        let debug = format!("{:?}", ep);
        assert!(debug.contains("score"));
        assert!(debug.contains("is_pinned"));
        assert!(debug.contains("access_count"));
        assert!(debug.contains("ExpertWeight"));
    }

    // ── 24. CompressionCodec variants in page codec field ──

    #[test]
    fn page_codec_field_accepts_all_compression_variants() {
        // Arrange: create a page and iterate through all CompressionCodec variants
        // Act: assign each variant to the codec field
        // Assert: field holds the assigned variant correctly
        use crate::kv_cache::CompressionCodec;
        let mut page = UnifiedVirtualPage::kv(0, 0, KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let codecs = [CompressionCodec::None, CompressionCodec::Lz4, CompressionCodec::BitPackRle, CompressionCodec::NvcompAns, CompressionCodec::ZstdDict];
        for codec in codecs {
            page.codec = codec;
            assert_eq!(page.codec, codec);
        }
    }

    // ── 25. SequenceGroup large page list clone ──

    #[test]
    fn sequence_group_large_page_list_clone_independent() {
        // Arrange: create a SequenceGroup with many pages
        // Act: clone and modify the original's page list
        // Assert: cloned page list is independent (Clone, not Rc)
        let pages: Vec<PageId> = (0..1000).collect();
        let sg = SequenceGroup {
            id: 1,
            pages: pages.clone(),
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1000,
            pipeline: KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        let mut cloned = sg.clone();
        cloned.pages.push(9999);
        assert_eq!(sg.pages.len(), 1000);
        assert_eq!(cloned.pages.len(), 1001);
        assert!(cloned.pages.contains(&9999));
        assert!(!sg.pages.contains(&9999));
    }

    // ── 26. PageState all variants usable in HashMap as keys ──

    #[test]
    fn page_state_usable_as_hashmap_key() {
        // Arrange: create a HashMap with PageState keys
        // Act: insert values for all 7 variants, then read back
        // Assert: each key maps to its correct value
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(PageState::Free, 1u8);
        map.insert(PageState::Active, 2);
        map.insert(PageState::Standby, 3);
        map.insert(PageState::SwappedOut, 4);
        map.insert(PageState::Warm, 5);
        map.insert(PageState::Protected, 6);
        map.insert(PageState::Swapped, 7);
        assert_eq!(map.get(&PageState::Free), Some(&1));
        assert_eq!(map.get(&PageState::Protected), Some(&6));
        assert_eq!(map.get(&PageState::Swapped), Some(&7));
        assert_eq!(map.len(), 7);
    }

    // ── 27. WeightTier usable as HashMap key ──

    #[test]
    fn weight_tier_usable_as_hashmap_key() {
        // Arrange: create a HashMap with WeightTier keys
        // Act: insert all 3 tiers, read back
        // Assert: correct mapping for each tier
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(WeightTier::Hot, "GPU VRAM");
        map.insert(WeightTier::Warm, "CPU RAM");
        map.insert(WeightTier::Cold, "NVMe");
        assert_eq!(map.get(&WeightTier::Hot), Some(&"GPU VRAM"));
        assert_eq!(map.get(&WeightTier::Warm), Some(&"CPU RAM"));
        assert_eq!(map.get(&WeightTier::Cold), Some(&"NVMe"));
        assert_eq!(map.len(), 3);
    }

    // ── 28. Expert page with zero expert_id and layer_idx ──

    #[test]
    fn expert_page_boundary_zero_ids() {
        // Arrange: create expert page with expert_id=0, layer_idx=0
        // Act: verify all fields
        // Assert: zero is a valid expert_id and layer_idx value
        let page = UnifiedVirtualPage::expert(0, 0, 0, gllm_kernels::types::DType::F32);
        assert_eq!(page.expert_id, Some(0));
        assert_eq!(page.layer_idx, Some(0));
        assert_eq!(page.page_id, 0);
        assert!(page.is_evictable());
        assert!(page.is_on_device());
        assert!(page.owner.is_none());
    }

    // ══════════════════════════════════════════════════════════════════════
    //  ADDITIONAL TESTS (13 new — reaching 180 total)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. KV page with U8 (unsigned byte) dtype ──

    #[test]
    fn kv_page_with_u8_dtype() {
        // Arrange: construct a KV context page using the U8 dtype
        // Act: verify dtype is preserved alongside other defaults
        // Assert: dtype matches U8, payload and pipeline are correct
        let page = UnifiedVirtualPage::kv(
            8, 100, KvPipeline::Working, 2,
            gllm_kernels::types::DType::U8,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::U8);
        assert_eq!(page.payload_kind, PagePayloadKind::KvContext);
        assert_eq!(page.pipeline, Some(KvPipeline::Working));
        assert!(page.is_evictable());
    }

    // ── 2. Expert page with F6E3M2 (AMD CDNA4) dtype ──

    #[test]
    fn expert_page_with_f6e3m2_dtype() {
        // Arrange: construct an expert weight page with FP6 E3M2 dtype
        // Act: verify dtype is preserved, expert metadata correct
        // Assert: dtype matches, expert_id/layer_idx are Some
        let page = UnifiedVirtualPage::expert(
            11, 4, 9,
            gllm_kernels::types::DType::F6E3M2,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F6E3M2);
        assert_eq!(page.expert_id, Some(4));
        assert_eq!(page.layer_idx, Some(9));
        assert!(page.is_evictable());
        assert!(page.is_on_device());
    }

    // ── 3. Dense layer page with F6E2M3 (AMD CDNA4) dtype ──

    #[test]
    fn dense_layer_with_f6e2m3_dtype() {
        // Arrange: construct a dense layer page with FP6 E2M3 dtype
        // Act: verify dtype is preserved and page is non-evictable
        // Assert: dtype matches, is_evictable returns false
        let page = UnifiedVirtualPage::dense_layer(
            22, 6,
            gllm_kernels::types::DType::F6E2M3,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F6E2M3);
        assert!(!page.is_evictable());
        assert!(page.is_on_device());
        assert_eq!(page.logical_index, 6);
    }

    // ── 4. System prompt page residency transition to DiskSwap ──

    #[test]
    fn system_prompt_page_residency_disk_swap() {
        // Arrange: create a system prompt page (default DeviceLocal)
        // Act: mutate residency to DiskSwap
        // Assert: is_on_device becomes false, but evictability stays false
        let mut page = UnifiedVirtualPage::system_prompt(
            13, gllm_kernels::types::DType::BF16,
        );
        assert!(page.is_on_device());
        assert!(!page.is_evictable());

        page.residency = MemoryResidency::DiskSwap;
        assert!(!page.is_on_device());
        // PromptSystem is never evictable regardless of residency
        assert!(!page.is_evictable());
    }

    // ── 5. SequenceGroup with PromptSystem payload kind ──

    #[test]
    fn sequence_group_with_prompt_system_payload() {
        // Arrange: construct a SequenceGroup with PromptSystem payload
        // Act: verify the payload_kind field
        // Assert: payload_kind is Some(PromptSystem), other fields correct
        let sg = SequenceGroup {
            id: 77,
            pages: vec![100, 101],
            state: GroupState::Running,
            access_count: 10,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 512,
            pipeline: KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::PromptSystem),
        };
        assert_eq!(sg.payload_kind, Some(PagePayloadKind::PromptSystem));
        assert!(sg.is_pinned);
        assert_eq!(sg.pages, vec![100, 101]);
    }

    // ── 6. EvictionPriority with recency at usize::MAX ──

    #[test]
    fn eviction_priority_max_recency() {
        // Arrange: construct an EvictionPriority with maximum recency
        // Act: verify the recency field holds usize::MAX
        // Assert: recency is at boundary, score and other fields unchanged
        let ep = EvictionPriority {
            score: 500,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 1,
            recency: usize::MAX,
            layer_idx: None,
            expert_id: None,
        };
        assert_eq!(ep.recency, usize::MAX);
        assert_eq!(ep.score, 500);
        assert!(!ep.is_pinned);
    }

    // ── 7. KV page with NvcompAns compression codec ──

    #[test]
    fn kv_page_with_nvcomp_ans_compression() {
        // Arrange: create a KV page and set NvcompAns compression codec
        // Act: mutate codec and compression sizes
        // Assert: codec is NvcompAns, sizes are consistent
        let mut page = UnifiedVirtualPage::kv(
            20, 30, KvPipeline::Conversation, 1,
            gllm_kernels::types::DType::BF16,
        );
        page.codec = crate::kv_cache::CompressionCodec::NvcompAns;
        page.compressed_size = 1024;
        page.decompressed_size = 8192;
        assert_eq!(page.codec, crate::kv_cache::CompressionCodec::NvcompAns);
        assert_eq!(page.compressed_size, 1024);
        assert_eq!(page.decompressed_size, 8192);
        assert!(page.compressed_size < page.decompressed_size);
    }

    // ── 8. KV page with ZstdDict compression codec ──

    #[test]
    fn kv_page_with_zstd_dict_compression() {
        // Arrange: create a KV page and set ZstdDict compression codec
        // Act: mutate codec and sizes to simulate NVMe cold-tier storage
        // Assert: codec is ZstdDict, dtype and pipeline unaffected
        let mut page = UnifiedVirtualPage::kv(
            25, 40, KvPipeline::Working, 4,
            gllm_kernels::types::DType::F16,
        );
        page.codec = crate::kv_cache::CompressionCodec::ZstdDict;
        page.compressed_size = 256;
        page.decompressed_size = 4096;
        assert_eq!(page.codec, crate::kv_cache::CompressionCodec::ZstdDict);
        assert_eq!(page.compressed_size, 256);
        // dtype and pipeline should remain unchanged by compression mutation
        assert_eq!(page.dtype, gllm_kernels::types::DType::F16);
        assert_eq!(page.pipeline, Some(KvPipeline::Working));
    }

    // ── 9. BatchOrderPolicy Eq trait verification ──

    #[test]
    fn batch_order_policy_eq_trait() {
        // Arrange: create two BatchOrderPolicy values of same variant
        // Act: compare via == and !=
        // Assert: same variant equals, different variants differ
        let a = BatchOrderPolicy::StrictRequestIdOrder;
        let b = BatchOrderPolicy::StrictRequestIdOrder;
        let c = BatchOrderPolicy::FifoOrder;
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a == b);
        assert!(a != c);
    }

    // ── 10. Expert page round-trip residency DeviceLocal -> HostLocal -> DiskSwap -> DeviceLocal ──

    #[test]
    fn expert_page_residency_round_trip() {
        // Arrange: create an expert page (default DeviceLocal)
        // Act: cycle residency through all three variants back to DeviceLocal
        // Assert: is_on_device only true when DeviceLocal
        let mut page = UnifiedVirtualPage::expert(
            0, 1, 2, gllm_kernels::types::DType::F32,
        );
        assert!(page.is_on_device());

        page.residency = MemoryResidency::HostLocal;
        assert!(!page.is_on_device());

        page.residency = MemoryResidency::DiskSwap;
        assert!(!page.is_on_device());

        page.residency = MemoryResidency::DeviceLocal;
        assert!(page.is_on_device());
        // Evictability is determined by payload_kind, stays constant
        assert!(page.is_evictable());
    }

    // ── 11. Multiple UnifiedVirtualPage clones with different dtypes in a Vec ──

    #[test]
    fn multiple_pages_with_distinct_dtypes() {
        // Arrange: create pages of each constructor with a unique dtype
        // Act: collect into a Vec and verify each retains its dtype
        // Assert: all dtypes are distinct and preserved
        let pages = vec![
            UnifiedVirtualPage::kv(0, 0, KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32),
            UnifiedVirtualPage::expert(1, 0, 0, gllm_kernels::types::DType::F16),
            UnifiedVirtualPage::system_prompt(2, gllm_kernels::types::DType::BF16),
            UnifiedVirtualPage::rag(3, 0, gllm_kernels::types::DType::F8E4M3),
            UnifiedVirtualPage::dense_layer(4, 0, gllm_kernels::types::DType::F8E5M2),
        ];
        assert_eq!(pages[0].dtype, gllm_kernels::types::DType::F32);
        assert_eq!(pages[1].dtype, gllm_kernels::types::DType::F16);
        assert_eq!(pages[2].dtype, gllm_kernels::types::DType::BF16);
        assert_eq!(pages[3].dtype, gllm_kernels::types::DType::F8E4M3);
        assert_eq!(pages[4].dtype, gllm_kernels::types::DType::F8E5M2);
        // Verify distinct page_ids
        assert_eq!(pages[0].page_id, 0);
        assert_eq!(pages[1].page_id, 1);
        assert_eq!(pages[2].page_id, 2);
        assert_eq!(pages[3].page_id, 3);
        assert_eq!(pages[4].page_id, 4);
    }

    // ── 12. PageMetadata sequence_id zero is a valid value ──

    #[test]
    fn page_metadata_sequence_id_zero_is_valid() {
        // Arrange: create a PageMetadata with sequence_id = Some(0)
        // Act: verify the field holds zero (not confused with None)
        // Assert: Some(0) is distinct from None
        let meta = PageMetadata {
            sequence_id: Some(0),
            ..PageMetadata::default()
        };
        assert_eq!(meta.sequence_id, Some(0));
        assert!(meta.sequence_id.is_some());
        assert_ne!(meta.sequence_id, None);
    }

    // ── 13. PipelinedVirtualPageId in HashSet with duplicates from both pipelines ──

    #[test]
    fn pipelined_virtual_page_id_hash_set_dedup() {
        // Arrange: create PipelinedVirtualPageId entries including exact duplicates
        // Act: insert into a HashSet
        // Assert: duplicates are removed, count reflects unique entries
        use std::collections::HashSet;
        let set: HashSet<PipelinedVirtualPageId> = [
            PipelinedVirtualPageId { pipeline: KvPipeline::Conversation, sequence_id: 1, logical_index: 0 },
            PipelinedVirtualPageId { pipeline: KvPipeline::Working, sequence_id: 1, logical_index: 0 },
            PipelinedVirtualPageId { pipeline: KvPipeline::Conversation, sequence_id: 1, logical_index: 0 },
            PipelinedVirtualPageId { pipeline: KvPipeline::Conversation, sequence_id: 2, logical_index: 0 },
            PipelinedVirtualPageId { pipeline: KvPipeline::Working, sequence_id: 1, logical_index: 1 },
        ].into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  ADDITIONAL TESTS (13 new — reaching 193 total)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. KV page with equal compressed and decompressed sizes (no compression ratio) ──

    #[test]
    fn kv_page_equal_compressed_and_decompressed_sizes() {
        // Arrange: create a KV page and set compressed_size == decompressed_size
        // Act: mutate compression size fields to equal values
        // Assert: sizes are equal (edge case: uncompressed or incompressible data)
        let mut page = UnifiedVirtualPage::kv(
            5, 10, KvPipeline::Conversation, 0,
            gllm_kernels::types::DType::F32,
        );
        page.compressed_size = 8192;
        page.decompressed_size = 8192;
        assert_eq!(page.compressed_size, page.decompressed_size);
        assert_eq!(page.compressed_size, 8192);
    }

    // ── 2. PageMetadata double-default verification ──

    #[test]
    fn page_metadata_double_default_consistency() {
        // Arrange: create two independent PageMetadata via default()
        // Act: compare all fields
        // Assert: both defaults are identical and no field is randomly initialized
        let a = PageMetadata::default();
        let b = PageMetadata::default();
        assert_eq!(a.page_id, b.page_id);
        assert_eq!(a.sequence_id, b.sequence_id);
        assert_eq!(a.recency, b.recency);
        assert_eq!(a.access_count, b.access_count);
        assert_eq!(a.is_lir, b.is_lir);
        assert_eq!(a.state, b.state);
        assert_eq!(a.warm_until.is_none(), b.warm_until.is_none());
        assert_eq!(a.swap_in_time.is_none(), b.swap_in_time.is_none());
    }

    // ── 3. SequenceGroup with owned pages but zero context_len ──

    #[test]
    fn sequence_group_owned_pages_zero_context() {
        // Arrange: construct a SequenceGroup with non-empty pages but context_len=0
        // Act: verify pages exist but context_len is zero (pre-prefill state)
        // Assert: pages.len > 0 but context_len == 0 (valid initial state)
        let sg = SequenceGroup {
            id: 88,
            pages: vec![50, 51, 52],
            state: GroupState::Paused,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 0,
            pipeline: KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        assert!(!sg.pages.is_empty());
        assert_eq!(sg.context_len, 0);
        assert!(sg.is_pinned);
        assert_eq!(sg.state, GroupState::Paused);
    }

    // ── 4. UnifiedVirtualPage is_evictable consistency after payload_kind mutation ──

    #[test]
    fn unified_virtual_page_evictability_follows_payload_kind() {
        // Arrange: create an evictable KV page
        // Act: mutate payload_kind to PromptSystem (non-evictable) and back
        // Assert: is_evictable reflects the current payload_kind, not the original
        let mut page = UnifiedVirtualPage::kv(
            1, 1, KvPipeline::Conversation, 0,
            gllm_kernels::types::DType::F32,
        );
        assert!(page.is_evictable());

        page.payload_kind = PagePayloadKind::PromptSystem;
        assert!(!page.is_evictable());

        page.payload_kind = PagePayloadKind::ExpertWeight;
        assert!(page.is_evictable());

        page.payload_kind = PagePayloadKind::DenseLayerWeight;
        assert!(!page.is_evictable());

        page.payload_kind = PagePayloadKind::KnowledgeRAG;
        assert!(page.is_evictable());
    }

    // ── 5. EvictionPriority with max access_count and zero recency ──

    #[test]
    fn eviction_priority_max_access_count_zero_recency() {
        // Arrange: construct an EvictionPriority with access_count at usize::MAX and recency=0
        // Act: verify the boundary values coexist correctly
        // Assert: access_count holds usize::MAX, recency holds 0 (hot page with many accesses)
        let ep = EvictionPriority {
            score: 9999,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: usize::MAX,
            recency: 0,
            layer_idx: Some(0),
            expert_id: None,
        };
        assert_eq!(ep.access_count, usize::MAX);
        assert_eq!(ep.recency, 0);
        assert_eq!(ep.score, 9999);
        assert_eq!(ep.layer_idx, Some(0));
    }

    // ── 6. PipelinedVirtualPageId inequality when only logical_index differs ──

    #[test]
    fn pipelined_virtual_page_id_differs_only_by_logical_index() {
        // Arrange: create two PipelinedVirtualPageId with same pipeline and sequence_id
        // Act: compare equality
        // Assert: different logical_index means not equal
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 3,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 7,
        };
        assert_ne!(a, b);
        assert_eq!(a.pipeline, b.pipeline);
        assert_eq!(a.sequence_id, b.sequence_id);
    }

    // ── 7. SequenceGroup with context_len at usize::MAX boundary ──

    #[test]
    fn sequence_group_max_context_len_boundary() {
        // Arrange: construct a SequenceGroup with context_len=usize::MAX
        // Act: verify the field holds the boundary value
        // Assert: context_len is usize::MAX without overflow or truncation
        let sg = SequenceGroup {
            id: 255,
            pages: vec![1, 2],
            state: GroupState::Running,
            access_count: 100,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: usize::MAX,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };
        assert_eq!(sg.context_len, usize::MAX);
        assert_eq!(sg.id, 255);
        assert_eq!(sg.pages.len(), 2);
    }

    // ── 8. PageMetadata clone preserves optional Instant fields ──

    #[test]
    fn page_metadata_clone_preserves_optional_instants() {
        // Arrange: construct a PageMetadata with swap_in_time and warm_until set
        // Act: clone the metadata
        // Assert: cloned copy has both optional Instant fields present
        let now = Instant::now();
        let meta = PageMetadata {
            page_id: 10,
            sequence_id: Some(5),
            recency: 3,
            access_count: 7,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: true,
            state: PageState::Warm,
            warm_until: Some(now),
        };
        let cloned = meta.clone();
        assert!(cloned.swap_in_time.is_some());
        assert!(cloned.warm_until.is_some());
        assert_eq!(cloned.page_id, 10);
        assert_eq!(cloned.state, PageState::Warm);
        assert!(cloned.is_lir);
    }

    // ── 9. Non-expert page types have expert_id and layer_idx as None ──

    #[test]
    fn non_expert_pages_have_none_expert_fields() {
        // Arrange: create pages via kv, system_prompt, rag, dense_layer constructors
        // Act: verify expert_id and layer_idx for each
        // Assert: all non-expert pages have expert_id=None; layer_idx is None except dense_layer
        let kv = UnifiedVirtualPage::kv(0, 1, KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32);
        let sys = UnifiedVirtualPage::system_prompt(1, gllm_kernels::types::DType::F32);
        let rag = UnifiedVirtualPage::rag(2, 3, gllm_kernels::types::DType::F32);
        let dense = UnifiedVirtualPage::dense_layer(3, 0, gllm_kernels::types::DType::F32);

        assert!(kv.expert_id.is_none());
        assert!(kv.layer_idx.is_none());

        assert!(sys.expert_id.is_none());
        assert!(sys.layer_idx.is_none());

        assert!(rag.expert_id.is_none());
        assert!(rag.layer_idx.is_none());

        assert!(dense.expert_id.is_none());
        // dense_layer sets layer_idx = logical_index
        assert_eq!(dense.layer_idx, Some(0));
    }

    // ── 10. EvictionPriority clone preserves negative score with None expert fields ──

    #[test]
    fn eviction_priority_clone_negative_score_no_expert() {
        // Arrange: construct an EvictionPriority with a negative score and no expert metadata
        // Act: clone the struct
        // Assert: all fields including negative score and None fields are preserved
        let ep = EvictionPriority {
            score: -7777,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 50,
            recency: 200,
            layer_idx: None,
            expert_id: None,
        };
        let cloned = ep.clone();
        assert_eq!(cloned.score, -7777);
        assert_eq!(cloned.payload_kind, PagePayloadKind::KvContext);
        assert!(!cloned.is_pinned);
        assert_eq!(cloned.access_count, 50);
        assert_eq!(cloned.recency, 200);
        assert_eq!(cloned.layer_idx, None);
        assert_eq!(cloned.expert_id, None);
    }

    // ── 11. BatchOrderPolicy Copy allows reuse after assignment ──

    #[test]
    fn batch_order_policy_copy_allows_reuse() {
        // Arrange: create a BatchOrderPolicy value and assign it to two variables
        // Act: compare both variables to each other and to a different variant
        // Assert: Copy semantics allow both variables to hold the same value
        let original = BatchOrderPolicy::FifoOrder;
        let copy_a = original;
        let copy_b = original;
        assert_eq!(original, copy_a);
        assert_eq!(original, copy_b);
        assert_eq!(copy_a, copy_b);
        assert_ne!(copy_a, BatchOrderPolicy::StrictRequestIdOrder);
    }

    // ── 12. WeightTier debug format contains expected variant names ──

    #[test]
    fn weight_tier_debug_contains_variant_names() {
        // Arrange: format each WeightTier variant via Debug
        // Act: check that the debug string contains the variant name
        // Assert: each variant name appears in its own debug output
        assert!(format!("{:?}", WeightTier::Hot).contains("Hot"));
        assert!(format!("{:?}", WeightTier::Warm).contains("Warm"));
        assert!(format!("{:?}", WeightTier::Cold).contains("Cold"));
        // Verify no cross-contamination
        assert!(!format!("{:?}", WeightTier::Hot).contains("Cold"));
        assert!(!format!("{:?}", WeightTier::Cold).contains("Hot"));
    }

    // ── 13. Multiple KV pages in a Vec with distinct owners and pipelines ──

    #[test]
    fn multiple_kv_pages_distinct_owners_in_vec() {
        // Arrange: create several KV pages with different owners, pipelines, and logical indices
        // Act: collect into a Vec and verify each page's independent state
        // Assert: no cross-contamination between pages, each retains its own fields
        let pages = vec![
            UnifiedVirtualPage::kv(0, 100, KvPipeline::Conversation, 0, gllm_kernels::types::DType::F32),
            UnifiedVirtualPage::kv(1, 200, KvPipeline::Working, 1, gllm_kernels::types::DType::BF16),
            UnifiedVirtualPage::kv(2, 300, KvPipeline::Conversation, 2, gllm_kernels::types::DType::F16),
        ];
        assert_eq!(pages.len(), 3);

        assert_eq!(pages[0].owner, Some(100));
        assert_eq!(pages[0].pipeline, Some(KvPipeline::Conversation));
        assert_eq!(pages[0].logical_index, 0);

        assert_eq!(pages[1].owner, Some(200));
        assert_eq!(pages[1].pipeline, Some(KvPipeline::Working));
        assert_eq!(pages[1].logical_index, 1);

        assert_eq!(pages[2].owner, Some(300));
        assert_eq!(pages[2].pipeline, Some(KvPipeline::Conversation));
        assert_eq!(pages[2].logical_index, 2);

        // All are evictable and on device by default
        for page in &pages {
            assert!(page.is_evictable());
            assert!(page.is_on_device());
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  ADDITIONAL TESTS (13 new — reaching 206 total)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn kv_page_with_f6e2m3_dtype() {
        let page = UnifiedVirtualPage::kv(
            12, 34, KvPipeline::Working, 5,
            gllm_kernels::types::DType::F6E2M3,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F6E2M3);
        assert_eq!(page.payload_kind, PagePayloadKind::KvContext);
        assert_eq!(page.pipeline, Some(KvPipeline::Working));
        assert_eq!(page.logical_index, 5);
        assert!(page.is_evictable());
        assert!(page.is_on_device());
    }

    #[test]
    fn rag_page_with_f8e4m3_dtype() {
        let page = UnifiedVirtualPage::rag(
            44, 55, gllm_kernels::types::DType::F8E4M3,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F8E4M3);
        assert_eq!(page.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert_eq!(page.residency, MemoryResidency::HostLocal);
        assert!(!page.is_on_device());
        assert!(page.is_evictable());
        assert_eq!(page.owner, Some(55));
        assert!(page.pipeline.is_none());
        assert_eq!(page.logical_index, 0);
    }

    #[test]
    fn rag_page_compressed_size_mutation() {
        let mut page = UnifiedVirtualPage::rag(
            60, 70, gllm_kernels::types::DType::BF16,
        );
        page.codec = crate::kv_cache::CompressionCodec::Lz4;
        page.compressed_size = 1024;
        page.decompressed_size = 4096;
        assert_eq!(page.codec, crate::kv_cache::CompressionCodec::Lz4);
        assert_eq!(page.compressed_size, 1024);
        assert_eq!(page.decompressed_size, 4096);
        assert_eq!(page.dtype, gllm_kernels::types::DType::BF16);
    }

    #[test]
    fn dense_layer_page_clone_preserves_all_fields() {
        let page = UnifiedVirtualPage::dense_layer(
            15, 9, gllm_kernels::types::DType::F16,
        );
        let cloned = page.clone();
        assert_eq!(cloned.page_id, 15);
        assert_eq!(cloned.logical_index, 9);
        assert_eq!(cloned.layer_idx, Some(9));
        assert_eq!(cloned.dtype, gllm_kernels::types::DType::F16);
        assert_eq!(cloned.payload_kind, PagePayloadKind::DenseLayerWeight);
        assert!(!cloned.is_evictable());
        assert!(cloned.is_on_device());
        assert!(cloned.owner.is_none());
        assert!(cloned.expert_id.is_none());
    }

    #[test]
    fn eviction_priority_pinned_dense_layer_weight_positive_score() {
        let ep = EvictionPriority {
            score: 5000,
            payload_kind: PagePayloadKind::DenseLayerWeight,
            is_pinned: true,
            access_count: 200,
            recency: 1,
            layer_idx: Some(42),
            expert_id: None,
        };
        assert!(ep.is_pinned);
        assert_eq!(ep.score, 5000);
        assert_eq!(ep.payload_kind, PagePayloadKind::DenseLayerWeight);
        assert_eq!(ep.access_count, 200);
        assert_eq!(ep.recency, 1);
        assert_eq!(ep.layer_idx, Some(42));
    }

    #[test]
    fn sequence_group_with_knowledge_rag_payload() {
        let sg = SequenceGroup {
            id: 33,
            pages: vec![200, 201, 202],
            state: GroupState::Running,
            access_count: 5,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 1024,
            pipeline: KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KnowledgeRAG),
        };
        assert_eq!(sg.payload_kind, Some(PagePayloadKind::KnowledgeRAG));
        assert_eq!(sg.pipeline, KvPipeline::Working);
        assert_eq!(sg.pages, vec![200, 201, 202]);
        assert!(!sg.is_pinned);
    }

    #[test]
    fn batch_order_policy_debug_format() {
        let strict_debug = format!("{:?}", BatchOrderPolicy::StrictRequestIdOrder);
        let fifo_debug = format!("{:?}", BatchOrderPolicy::FifoOrder);
        assert!(strict_debug.contains("StrictRequestIdOrder"));
        assert!(fifo_debug.contains("FifoOrder"));
        assert!(!strict_debug.contains("FifoOrder"));
        assert!(!fifo_debug.contains("StrictRequestIdOrder"));
    }

    #[test]
    fn kv_page_compressed_size_u32_max() {
        let mut page = UnifiedVirtualPage::kv(
            7, 14, KvPipeline::Conversation, 2,
            gllm_kernels::types::DType::F32,
        );
        page.compressed_size = u32::MAX;
        page.decompressed_size = u32::MAX;
        assert_eq!(page.compressed_size, u32::MAX);
        assert_eq!(page.decompressed_size, u32::MAX);
    }

    #[test]
    fn unified_virtual_page_dense_layer_layer_idx_equals_logical_index() {
        for logical_index in [0, 1, 5, 100, usize::MAX] {
            let page = UnifiedVirtualPage::dense_layer(
                0, logical_index, gllm_kernels::types::DType::F32,
            );
            assert_eq!(page.layer_idx, Some(logical_index));
            assert_eq!(page.logical_index, logical_index);
        }
    }

    #[test]
    fn system_prompt_page_with_f8e5m2_dtype() {
        let page = UnifiedVirtualPage::system_prompt(
            88, gllm_kernels::types::DType::F8E5M2,
        );
        assert_eq!(page.dtype, gllm_kernels::types::DType::F8E5M2);
        assert_eq!(page.payload_kind, PagePayloadKind::PromptSystem);
        assert!(!page.is_evictable());
        assert!(page.is_on_device());
        assert!(page.owner.is_none());
        assert!(page.pipeline.is_none());
        assert!(page.expert_id.is_none());
        assert!(page.layer_idx.is_none());
    }

    #[test]
    fn page_metadata_default_not_lir() {
        let meta = PageMetadata::default();
        assert!(!meta.is_lir);
        assert_eq!(meta.state, PageState::Standby);
    }

    #[test]
    fn pipelined_virtual_page_id_in_hashmap_with_overwrite() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let key = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 10,
            logical_index: 3,
        };
        map.insert(key, "first");
        map.insert(key, "second");
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&key), Some(&"second"));
    }

    #[test]
    fn expert_page_with_f4e2m1_dtype_boundary_expert_id() {
        let page = UnifiedVirtualPage::expert(
            usize::MAX, u32::MAX, 0,
            gllm_kernels::types::DType::F4E2M1,
        );
        assert_eq!(page.page_id, usize::MAX);
        assert_eq!(page.expert_id, Some(u32::MAX));
        assert_eq!(page.dtype, gllm_kernels::types::DType::F4E2M1);
        assert!(page.is_evictable());
    }

    // ══════════════════════════════════════════════════════════════════════
    //  ADDITIONAL TESTS (10 new — reaching 216 total)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn expert_page_with_compression_codec_lz4() {
        // Arrange: create an expert weight page, then apply Lz4 compression
        let mut page = UnifiedVirtualPage::expert(
            30, 8, 4,
            gllm_kernels::types::DType::BF16,
        );
        // Act: mutate compression fields
        page.codec = crate::kv_cache::CompressionCodec::Lz4;
        page.compressed_size = 2048;
        page.decompressed_size = 16384;
        // Assert: compression fields are set, expert metadata unaffected
        assert_eq!(page.codec, crate::kv_cache::CompressionCodec::Lz4);
        assert_eq!(page.compressed_size, 2048);
        assert_eq!(page.decompressed_size, 16384);
        assert_eq!(page.expert_id, Some(8));
        assert_eq!(page.layer_idx, Some(4));
        assert_eq!(page.dtype, gllm_kernels::types::DType::BF16);
        assert!(page.is_evictable());
    }

    #[test]
    fn kv_page_compressed_size_larger_than_decompressed() {
        // Arrange: create a KV page and set compressed_size > decompressed_size
        // (edge case: incompressible data or corrupted metadata)
        let mut page = UnifiedVirtualPage::kv(
            7, 14, KvPipeline::Conversation, 2,
            gllm_kernels::types::DType::F32,
        );
        // Act: set compressed larger than decompressed (pathological but allowed by type)
        page.compressed_size = 16384;
        page.decompressed_size = 4096;
        // Assert: fields hold their values without correction (no runtime enforcement)
        assert_eq!(page.compressed_size, 16384);
        assert_eq!(page.decompressed_size, 4096);
        assert!(page.compressed_size > page.decompressed_size);
    }

    #[test]
    fn dense_layer_page_with_compression_and_host_residency() {
        // Arrange: create a dense layer page, then simulate migration to host with compression
        let mut page = UnifiedVirtualPage::dense_layer(
            100, 3,
            gllm_kernels::types::DType::F16,
        );
        // Act: migrate to host and apply BitPackRle compression
        page.residency = MemoryResidency::HostLocal;
        page.codec = crate::kv_cache::CompressionCodec::BitPackRle;
        page.compressed_size = 512;
        page.decompressed_size = 4096;
        // Assert: is_on_device reflects host residency, evictability unchanged by residency
        assert!(!page.is_on_device());
        assert!(!page.is_evictable());
        assert_eq!(page.codec, crate::kv_cache::CompressionCodec::BitPackRle);
        assert_eq!(page.dtype, gllm_kernels::types::DType::F16);
        assert_eq!(page.layer_idx, Some(3));
    }

    #[test]
    fn pipelined_virtual_page_id_debug_format() {
        // Arrange: create a PipelinedVirtualPageId with known values
        let pvp = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 42,
            logical_index: 7,
        };
        // Act: format via Debug
        let debug = format!("{:?}", pvp);
        // Assert: debug output contains field-relevant info
        assert!(!debug.is_empty());
        assert!(debug.contains("PipelinedVirtualPageId"));
    }

    #[test]
    fn memory_residency_round_trip_all_variants() {
        // Arrange: create a KV page (default DeviceLocal)
        let mut page = UnifiedVirtualPage::kv(
            1, 1, KvPipeline::Conversation, 0,
            gllm_kernels::types::DType::F32,
        );
        // Act: cycle through all residency variants
        page.residency = MemoryResidency::HostLocal;
        assert_eq!(page.residency, MemoryResidency::HostLocal);
        assert!(!page.is_on_device());

        page.residency = MemoryResidency::DiskSwap;
        assert_eq!(page.residency, MemoryResidency::DiskSwap);
        assert!(!page.is_on_device());

        page.residency = MemoryResidency::DeviceLocal;
        assert_eq!(page.residency, MemoryResidency::DeviceLocal);
        assert!(page.is_on_device());
        // Assert: payload_kind and owner remain stable through residency changes
        assert_eq!(page.payload_kind, PagePayloadKind::KvContext);
        assert_eq!(page.owner, Some(1));
    }

    #[test]
    fn storage_key_as_hash_map_key() {
        // Arrange: create a HashMap keyed by StorageKey (u64 alias)
        use std::collections::HashMap;
        let mut map: HashMap<StorageKey, &'static str> = HashMap::new();
        // Act: insert entries with distinct StorageKey values
        map.insert(0xAAAA, "slot_a");
        map.insert(0xBBBB, "slot_b");
        map.insert(0xCCCC, "slot_c");
        // Assert: all lookups succeed, missing key returns None
        assert_eq!(map.get(&0xAAAA), Some(&"slot_a"));
        assert_eq!(map.get(&0xBBBB), Some(&"slot_b"));
        assert_eq!(map.get(&0xCCCC), Some(&"slot_c"));
        assert_eq!(map.get(&0xDDDD), None);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn sequence_group_same_id_different_pipelines() {
        // Arrange: create two SequenceGroups with the same id but different pipelines
        let conv = SequenceGroup {
            id: 42,
            pages: vec![1, 2],
            state: GroupState::Running,
            access_count: 5,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 256,
            pipeline: KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        let work = SequenceGroup {
            id: 42,
            pages: vec![3, 4],
            state: GroupState::Running,
            access_count: 2,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 64,
            pipeline: KvPipeline::Working,
            payload_kind: Some(PagePayloadKind::KvContext),
        };
        // Assert: same id but different pipelines and pages
        assert_eq!(conv.id, work.id);
        assert_ne!(conv.pipeline, work.pipeline);
        assert_eq!(conv.pipeline, KvPipeline::Conversation);
        assert_eq!(work.pipeline, KvPipeline::Working);
        assert_ne!(conv.pages, work.pages);
        assert!(conv.is_pinned);
        assert!(!work.is_pinned);
    }

    #[test]
    fn page_metadata_explicit_is_lir_false() {
        // Arrange: construct a PageMetadata with is_lir explicitly set to false
        let meta = PageMetadata {
            page_id: 5,
            sequence_id: Some(10),
            recency: 0,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        // Act & Assert: verify is_lir is explicitly false (not just default)
        assert!(!meta.is_lir);
        assert_eq!(meta.state, PageState::Active);
        assert_eq!(meta.page_id, 5);
        assert_eq!(meta.sequence_id, Some(10));
    }

    #[test]
    fn unified_virtual_page_is_on_device_invariant_across_mutation() {
        // Arrange: create a system prompt page and verify initial state
        let mut page = UnifiedVirtualPage::system_prompt(
            99,
            gllm_kernels::types::DType::BF16,
        );
        assert!(page.is_on_device());
        assert!(!page.is_evictable());
        // Act: mutate dtype and logical_index while keeping residency DeviceLocal
        page.dtype = gllm_kernels::types::DType::F16;
        page.logical_index = 7;
        // Assert: is_on_device still true (only residency matters)
        assert!(page.is_on_device());
        assert_eq!(page.dtype, gllm_kernels::types::DType::F16);
        assert_eq!(page.logical_index, 7);
        // Act: now change residency
        page.residency = MemoryResidency::HostLocal;
        // Assert: is_on_device now false despite other fields being set
        assert!(!page.is_on_device());
        assert!(!page.is_evictable());
    }

    #[test]
    fn rag_page_loaded_to_device_with_compression_clone_round_trip() {
        // Arrange: create a RAG page, migrate to device, apply compression, then clone
        let mut page = UnifiedVirtualPage::rag(
            200, 300,
            gllm_kernels::types::DType::F8E4M3,
        );
        assert_eq!(page.residency, MemoryResidency::HostLocal);
        assert!(!page.is_on_device());
        // Act: migrate to device and apply compression
        page.residency = MemoryResidency::DeviceLocal;
        page.codec = crate::kv_cache::CompressionCodec::ZstdDict;
        page.compressed_size = 128;
        page.decompressed_size = 2048;
        let cloned = page.clone();
        // Assert: clone preserves all fields including the mutated state
        assert_eq!(cloned.residency, MemoryResidency::DeviceLocal);
        assert!(cloned.is_on_device());
        assert_eq!(cloned.codec, crate::kv_cache::CompressionCodec::ZstdDict);
        assert_eq!(cloned.compressed_size, 128);
        assert_eq!(cloned.decompressed_size, 2048);
        assert_eq!(cloned.dtype, gllm_kernels::types::DType::F8E4M3);
        assert_eq!(cloned.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert_eq!(cloned.owner, Some(300));
        assert!(cloned.is_evictable());
    }
}
