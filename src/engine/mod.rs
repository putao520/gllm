//! Layer 3: Engine (skeleton).

pub mod arbiter;
pub mod batch_context;
pub mod batch_executor;
pub mod coordinator;
pub mod executor;
pub mod executor_types;
#[cfg(test)]
pub mod executor_tests;
#[cfg(test)]
pub mod executor_types_tests;
pub mod executor_api;
pub mod executor_builder;
pub mod executor_compile;
pub mod executor_step;
pub mod mega_kernel;
pub mod mega_kernel_callback;
pub mod mega_kernel_v2;
pub mod mtp_executor;
pub mod callbacks;

/// 引擎上下文。提供给各类 runtime hook 的引擎访问接口。
///
/// 所有维度字段从模型配置读取 (Ω1 真实性原则).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EngineContext {
    /// 模型总层数（用于语义锚点映射）
    pub num_layers: usize,
    /// 隐藏层维度
    pub hidden_size: usize,
    /// KV cache 页大小
    pub kv_page_size: usize,
    /// KV cache 头数 (GQA 模型, per SPEC §3.3)
    pub num_kv_heads: usize,
    /// 最大序列长度
    pub max_seq_len: usize,
}

impl EngineContext {
    /// 创建新的引擎上下文
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        kv_page_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            num_layers,
            hidden_size,
            kv_page_size,
            num_kv_heads,
            max_seq_len,
        }
    }

    /// 从执行器配置创建引擎上下文 (Ω1: 从实际模型配置读取)
    pub fn from_executor_config(config: &GeneratorForwardConfig) -> Self {
        Self {
            num_layers: config.num_layers(),
            hidden_size: config.hidden_size(),
            kv_page_size: config.paged_kv.page_size,
            num_kv_heads: config.num_kv_heads(),
            max_seq_len: config.max_seq_len(),
        }
    }
}

// Re-export scheduler types
pub use crate::scheduler::batcher::ContinuousBatcher;
pub use crate::scheduler::types::{GroupState, RequestKind, SequenceGroup};
pub use crate::scheduler::{PagedScheduler, ScheduledBatch};
pub use crate::scheduler::types::{PageId, PageState, RequestId};

// Re-export engine types
pub use batch_context::BatchContext;
pub use batch_executor::{BatchExecutor, BatchInferenceState, GenerateRequest, GenerateResult, execute_prefill, prefill_batch, execute_batch, batch_call, batch_lifecycle};
pub use executor::{
    AttentionHeadConfig, AttentionMaskType, AttentionTopology, BackendError, BatchInput,
    GeneratorForwardConfig, KvCacheConfig, KvCacheHandle, LogitsHandle, PagedKvConfig,
    PositionEncoding, RoPEConfig, SamplingConfig, SequenceInput, SwapConfig,
};

pub use coordinator::dispatch::DispatchCoordinator;
pub use coordinator::kv::KvCoordinator;
pub use coordinator::compute::ComputeCoordinator;
pub use coordinator::inference::InferenceCoordinator;
pub use coordinator::observability::ObservabilityCoordinator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_context_new_fields() {
        let ctx = EngineContext::new(32, 4096, 16, 8, 4096);
        assert_eq!(ctx.num_layers, 32);
        assert_eq!(ctx.hidden_size, 4096);
        assert_eq!(ctx.kv_page_size, 16);
        assert_eq!(ctx.num_kv_heads, 8);
        assert_eq!(ctx.max_seq_len, 4096);
    }

    #[test]
    fn engine_context_clone() {
        let ctx = EngineContext::new(24, 2048, 8, 4, 2048);
        let cloned = ctx.clone();
        assert_eq!(cloned.num_layers, 24);
        assert_eq!(cloned.hidden_size, 2048);
    }

    #[test]
    fn engine_context_zero_values() {
        let ctx = EngineContext::new(0, 0, 0, 0, 0);
        assert_eq!(ctx.num_layers, 0);
        assert_eq!(ctx.hidden_size, 0);
    }

    // ---- EngineContext: additional coverage ----

    #[test]
    fn engine_context_clone_independence() {
        // Arrange: create a context and clone it
        let mut ctx = EngineContext::new(12, 1024, 8, 4, 2048);
        let cloned = ctx.clone();

        // Act: mutate the original
        ctx.num_layers = 99;

        // Assert: clone is unaffected
        assert_eq!(cloned.num_layers, 12);
        assert_eq!(ctx.num_layers, 99);
    }

    #[test]
    fn engine_context_unit_values() {
        let ctx = EngineContext::new(1, 1, 1, 1, 1);
        assert_eq!(ctx.num_layers, 1);
        assert_eq!(ctx.hidden_size, 1);
        assert_eq!(ctx.kv_page_size, 1);
        assert_eq!(ctx.num_kv_heads, 1);
        assert_eq!(ctx.max_seq_len, 1);
    }

    #[test]
    fn engine_context_large_values() {
        let ctx = EngineContext::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX);
        assert_eq!(ctx.num_layers, usize::MAX);
        assert_eq!(ctx.hidden_size, usize::MAX);
        assert_eq!(ctx.kv_page_size, usize::MAX);
        assert_eq!(ctx.num_kv_heads, usize::MAX);
        assert_eq!(ctx.max_seq_len, usize::MAX);
    }

    #[test]
    fn engine_context_struct_construction() {
        // Arrange/Act: construct via struct literal (not new())
        let ctx = EngineContext {
            num_layers: 6,
            hidden_size: 512,
            kv_page_size: 4,
            num_kv_heads: 2,
            max_seq_len: 1024,
        };

        // Assert: all fields correctly assigned
        assert_eq!(ctx.num_layers, 6);
        assert_eq!(ctx.hidden_size, 512);
        assert_eq!(ctx.kv_page_size, 4);
        assert_eq!(ctx.num_kv_heads, 2);
        assert_eq!(ctx.max_seq_len, 1024);
    }

    #[test]
    fn engine_context_clone_copies_all_fields() {
        // Arrange
        let ctx = EngineContext::new(48, 8192, 32, 16, 8192);

        // Act
        let cloned = ctx.clone();

        // Assert: every field is independently verified
        assert_eq!(cloned.num_layers, 48);
        assert_eq!(cloned.hidden_size, 8192);
        assert_eq!(cloned.kv_page_size, 32);
        assert_eq!(cloned.num_kv_heads, 16);
        assert_eq!(cloned.max_seq_len, 8192);
    }

    #[test]
    fn engine_context_distinct_instances() {
        // Arrange: two different contexts
        let ctx_a = EngineContext::new(12, 1024, 8, 4, 512);
        let ctx_b = EngineContext::new(24, 2048, 16, 8, 1024);

        // Assert: no cross-contamination
        assert_ne!(ctx_a.num_layers, ctx_b.num_layers);
        assert_ne!(ctx_a.hidden_size, ctx_b.hidden_size);
        assert_ne!(ctx_a.kv_page_size, ctx_b.kv_page_size);
        assert_ne!(ctx_a.num_kv_heads, ctx_b.num_kv_heads);
        assert_ne!(ctx_a.max_seq_len, ctx_b.max_seq_len);
    }

    // ---- Re-exported types accessible through engine module ----

    #[test]
    fn page_state_variants_distinct() {
        // Arrange: all PageState variants
        let free = PageState::Free;
        let active = PageState::Active;
        let standby = PageState::Standby;
        let swapped_out = PageState::SwappedOut;
        let warm = PageState::Warm;
        let protected = PageState::Protected;

        // Assert: each variant is distinct
        assert_ne!(free, active);
        assert_ne!(active, standby);
        assert_ne!(standby, swapped_out);
        assert_ne!(swapped_out, warm);
        assert_ne!(warm, protected);
        assert_eq!(free, PageState::Free);
    }

    #[test]
    fn page_state_copy_trait() {
        // Arrange
        let original = PageState::Active;
        // Act: Copy semantics
        let copied = original;
        // Assert: both usable
        assert_eq!(original, copied);
    }

    #[test]
    fn page_state_hash_in_set() {
        // Arrange
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PageState::Free);
        set.insert(PageState::Active);

        // Assert
        assert!(set.contains(&PageState::Free));
        assert!(set.contains(&PageState::Active));
        assert!(!set.contains(&PageState::SwappedOut));
    }

    #[test]
    fn request_kind_variants_distinct() {
        // Arrange
        let chat = RequestKind::Chat;
        let embedding = RequestKind::Embedding;
        let rerank = RequestKind::Rerank;

        // Assert
        assert_ne!(chat, embedding);
        assert_ne!(embedding, rerank);
        assert_eq!(chat, RequestKind::Chat);
    }

    #[test]
    fn request_kind_copy_trait() {
        // Arrange
        let original = RequestKind::Embedding;
        // Act
        let copied = original;
        // Assert: both usable
        assert_eq!(original, copied);
    }

    #[test]
    fn group_state_variants() {
        // Arrange
        let running = GroupState::Running;
        let swapped = GroupState::Swapped;
        let paused = GroupState::Paused;

        // Assert
        assert_ne!(running, swapped);
        assert_ne!(swapped, paused);
        assert_eq!(running, GroupState::Running);
    }

    // ---- EngineContext: Debug, PartialEq, Eq, Hash ----

    #[test]
    fn engine_context_debug_trait() {
        // Arrange
        let ctx = EngineContext::new(12, 1024, 8, 4, 2048);

        // Act
        let debug_str = format!("{ctx:?}");

        // Assert: Debug output contains the type name and field values
        assert!(debug_str.contains("EngineContext"));
        assert!(debug_str.contains("12"));
        assert!(debug_str.contains("1024"));
    }

    #[test]
    fn engine_context_partial_eq_equal() {
        // Arrange
        let a = EngineContext::new(32, 4096, 16, 8, 4096);
        let b = EngineContext::new(32, 4096, 16, 8, 4096);

        // Assert
        assert_eq!(a, b);
    }

    #[test]
    fn engine_context_partial_eq_different_layers() {
        // Arrange
        let a = EngineContext::new(24, 4096, 16, 8, 4096);
        let b = EngineContext::new(32, 4096, 16, 8, 4096);

        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn engine_context_partial_eq_different_hidden_size() {
        // Arrange
        let a = EngineContext::new(32, 2048, 16, 8, 4096);
        let b = EngineContext::new(32, 4096, 16, 8, 4096);

        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn engine_context_partial_eq_different_page_size() {
        // Arrange
        let a = EngineContext::new(32, 4096, 16, 8, 4096);
        let b = EngineContext::new(32, 4096, 32, 8, 4096);

        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn engine_context_partial_eq_different_kv_heads() {
        // Arrange
        let a = EngineContext::new(32, 4096, 16, 4, 4096);
        let b = EngineContext::new(32, 4096, 16, 8, 4096);

        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn engine_context_partial_eq_different_max_seq() {
        // Arrange
        let a = EngineContext::new(32, 4096, 16, 8, 2048);
        let b = EngineContext::new(32, 4096, 16, 8, 4096);

        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn engine_context_hash_in_set() {
        // Arrange
        use std::collections::HashSet;
        let ctx_a = EngineContext::new(12, 1024, 8, 4, 2048);
        let ctx_b = EngineContext::new(24, 2048, 16, 8, 4096);

        // Act
        let mut set = HashSet::new();
        set.insert(ctx_a.clone());
        set.insert(ctx_b.clone());

        // Assert: both contexts are in the set
        assert!(set.contains(&EngineContext::new(12, 1024, 8, 4, 2048)));
        assert!(set.contains(&EngineContext::new(24, 2048, 16, 8, 4096)));
        assert!(!set.contains(&EngineContext::new(6, 512, 4, 2, 1024)));
    }

    #[test]
    fn engine_context_hash_deduplication() {
        // Arrange
        use std::collections::HashSet;
        let ctx = EngineContext::new(32, 4096, 16, 8, 4096);

        // Act: insert the same context twice
        let mut set = HashSet::new();
        set.insert(ctx.clone());
        set.insert(ctx.clone());

        // Assert: deduplication via Hash + Eq
        assert_eq!(set.len(), 1);
    }

    // ---- PageState: all variants including Swapped, exhaustive equality ----

    #[test]
    fn page_state_swapped_variant() {
        // Arrange
        let swapped = PageState::Swapped;

        // Assert: distinct from all other variants
        assert_ne!(swapped, PageState::Free);
        assert_ne!(swapped, PageState::Active);
        assert_ne!(swapped, PageState::Standby);
        assert_ne!(swapped, PageState::SwappedOut);
        assert_ne!(swapped, PageState::Warm);
        assert_ne!(swapped, PageState::Protected);
        assert_eq!(swapped, PageState::Swapped);
    }

    #[test]
    fn page_state_all_variants_hash_distinct() {
        // Arrange
        use std::collections::HashSet;
        let all_variants = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];

        // Act
        let set: HashSet<PageState> = all_variants.into_iter().collect();

        // Assert: all 7 variants are distinct
        assert_eq!(set.len(), 7);
    }

    #[test]
    fn page_state_debug_trait() {
        // Arrange
        let state = PageState::SwappedOut;

        // Act
        let debug_str = format!("{state:?}");

        // Assert
        assert!(debug_str.contains("SwappedOut"));
    }

    #[test]
    fn page_state_clone_trait() {
        // Arrange
        let original = PageState::Protected;

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(original, cloned);
    }

    // ---- GroupState: Copy, Clone, Debug, exhaustive ----

    #[test]
    fn group_state_copy_trait() {
        // Arrange
        let original = GroupState::Running;

        // Act: Copy semantics
        let copied = original;

        // Assert: both usable
        assert_eq!(original, copied);
    }

    #[test]
    fn group_state_clone_trait() {
        // Arrange
        let original = GroupState::Paused;

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(original, cloned);
    }

    #[test]
    fn group_state_debug_trait() {
        // Arrange
        let state = GroupState::Swapped;

        // Act
        let debug_str = format!("{state:?}");

        // Assert
        assert!(debug_str.contains("Swapped"));
    }

    #[test]
    fn group_state_all_variants_hash_distinct() {
        // Arrange
        use std::collections::HashSet;
        let all = [GroupState::Running, GroupState::Swapped, GroupState::Paused];

        // Act
        let set: HashSet<GroupState> = all.into_iter().collect();

        // Assert
        assert_eq!(set.len(), 3);
    }

    // ---- RequestKind: Clone, Debug, all variants ----

    #[test]
    fn request_kind_clone_trait() {
        // Arrange
        let original = RequestKind::Chat;

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(original, cloned);
    }

    #[test]
    fn request_kind_debug_trait() {
        // Arrange
        let kind = RequestKind::Rerank;

        // Act
        let debug_str = format!("{kind:?}");

        // Assert
        assert!(debug_str.contains("Rerank"));
    }

    #[test]
    fn request_kind_all_variants_hash_distinct() {
        // Arrange
        use std::collections::HashSet;
        let all = [RequestKind::Chat, RequestKind::Embedding, RequestKind::Rerank];

        // Act
        let set: HashSet<RequestKind> = all.into_iter().collect();

        // Assert
        assert_eq!(set.len(), 3);
    }

    // ---- SequenceGroup: construction and field access ----

    #[test]
    fn sequence_group_construction() {
        // Arrange
        use std::time::Instant;
        use crate::scheduler::types::KvPipeline;

        let now = Instant::now();

        // Act
        let sg = SequenceGroup {
            id: 42,
            pages: vec![0, 1, 2, 3],
            state: GroupState::Running,
            access_count: 5,
            last_access: now,
            is_pinned: false,
            context_len: 128,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };

        // Assert
        assert_eq!(sg.id, 42);
        assert_eq!(sg.pages.len(), 4);
        assert_eq!(sg.state, GroupState::Running);
        assert_eq!(sg.access_count, 5);
        assert!(!sg.is_pinned);
        assert_eq!(sg.context_len, 128);
    }

    #[test]
    fn sequence_group_empty_pages() {
        // Arrange
        use std::time::Instant;
        use crate::scheduler::types::KvPipeline;

        // Act
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

        // Assert
        assert!(sg.pages.is_empty());
        assert_eq!(sg.state, GroupState::Paused);
    }

    #[test]
    fn sequence_group_clone() {
        // Arrange
        use std::time::Instant;
        use crate::scheduler::types::KvPipeline;

        let sg = SequenceGroup {
            id: 7,
            pages: vec![10, 20],
            state: GroupState::Swapped,
            access_count: 3,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 64,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };

        // Act
        let cloned = sg.clone();

        // Assert
        assert_eq!(cloned.id, sg.id);
        assert_eq!(cloned.pages, sg.pages);
        assert_eq!(cloned.state, sg.state);
        assert_eq!(cloned.access_count, sg.access_count);
        assert_eq!(cloned.is_pinned, sg.is_pinned);
        assert_eq!(cloned.context_len, sg.context_len);
    }

    #[test]
    fn sequence_group_debug_trait() {
        // Arrange
        use std::time::Instant;
        use crate::scheduler::types::KvPipeline;

        let sg = SequenceGroup {
            id: 99,
            pages: vec![1],
            state: GroupState::Running,
            access_count: 1,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: 10,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };

        // Act
        let debug_str = format!("{sg:?}");

        // Assert
        assert!(debug_str.contains("SequenceGroup"));
    }

    #[test]
    fn sequence_group_with_payload_kind() {
        // Arrange
        use std::time::Instant;
        use crate::scheduler::types::{KvPipeline, PagePayloadKind};

        // Act
        let sg = SequenceGroup {
            id: 100,
            pages: vec![5, 6, 7],
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: true,
            context_len: 256,
            pipeline: KvPipeline::Conversation,
            payload_kind: Some(PagePayloadKind::KvContext),
        };

        // Assert
        assert!(sg.payload_kind.is_some());
        assert_eq!(sg.payload_kind.unwrap(), PagePayloadKind::KvContext);
    }

    // ---- Type aliases: RequestId, PageId ----

    #[test]
    fn request_id_type_alias_is_u64() {
        // Arrange/Act
        let rid: RequestId = 42u64;

        // Assert
        assert_eq!(rid, 42u64);
        assert_eq!(rid, 42);
    }

    #[test]
    fn request_id_zero() {
        let rid: RequestId = 0;
        assert_eq!(rid, 0);
    }

    #[test]
    fn request_id_max() {
        let rid: RequestId = u64::MAX;
        assert_eq!(rid, u64::MAX);
    }

    #[test]
    fn page_id_type_alias_is_usize() {
        // Arrange/Act
        let pid: PageId = 1024;

        // Assert
        assert_eq!(pid, 1024usize);
    }

    #[test]
    fn page_id_zero() {
        let pid: PageId = 0;
        assert_eq!(pid, 0);
    }

    #[test]
    fn page_id_max() {
        let pid: PageId = usize::MAX;
        assert_eq!(pid, usize::MAX);
    }

    // ---- KvPipeline: variants, Copy, Clone, Debug, Hash ----

    #[test]
    fn kv_pipeline_variants_distinct() {
        // Arrange
        let conv = crate::scheduler::types::KvPipeline::Conversation;
        let work = crate::scheduler::types::KvPipeline::Working;

        // Assert
        assert_ne!(conv, work);
        assert_eq!(conv, crate::scheduler::types::KvPipeline::Conversation);
        assert_eq!(work, crate::scheduler::types::KvPipeline::Working);
    }

    #[test]
    fn kv_pipeline_hash_deduplication() {
        // Arrange
        use std::collections::HashSet;
        use crate::scheduler::types::KvPipeline;

        // Act
        let mut set = HashSet::new();
        set.insert(KvPipeline::Conversation);
        set.insert(KvPipeline::Conversation);

        // Assert: same variant deduplicates
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn kv_pipeline_debug_output() {
        // Arrange
        use crate::scheduler::types::KvPipeline;

        // Act
        let debug = format!("{:?}", KvPipeline::Conversation);

        // Assert
        assert!(debug.contains("Conversation"));
    }

    // ---- BatchOrderPolicy: default, variants, Debug ----

    #[test]
    fn batch_order_policy_default_is_strict() {
        // Arrange/Act
        let policy = crate::scheduler::types::BatchOrderPolicy::default();

        // Assert
        assert_eq!(policy, crate::scheduler::types::BatchOrderPolicy::StrictRequestIdOrder);
    }

    #[test]
    fn batch_order_policy_variants_distinct() {
        // Arrange
        use crate::scheduler::types::BatchOrderPolicy;

        let strict = BatchOrderPolicy::StrictRequestIdOrder;
        let fifo = BatchOrderPolicy::FifoOrder;

        // Assert
        assert_ne!(strict, fifo);
        assert_eq!(strict, BatchOrderPolicy::StrictRequestIdOrder);
    }

    // ---- StorageKey type alias ----

    #[test]
    fn storage_key_type_alias_boundary() {
        // Arrange/Act
        let zero: crate::scheduler::types::StorageKey = 0;
        let max: crate::scheduler::types::StorageKey = u64::MAX;

        // Assert
        assert_eq!(zero, 0u64);
        assert_eq!(max, u64::MAX);
    }

    // ---- PhysicalId type alias ----

    #[test]
    fn physical_id_type_alias_boundary() {
        // Arrange/Act
        let zero: crate::scheduler::types::PhysicalId = 0;
        let max: crate::scheduler::types::PhysicalId = usize::MAX;

        // Assert
        assert_eq!(zero, 0usize);
        assert_eq!(max, usize::MAX);
    }

    // ---- PagePayloadKind: all variants distinct ----

    #[test]
    fn page_payload_kind_all_variants_distinct() {
        // Arrange
        use std::collections::HashSet;
        use crate::scheduler::types::PagePayloadKind;

        let all = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];

        // Act
        let set: HashSet<PagePayloadKind> = all.into_iter().collect();

        // Assert: all 5 variants are distinct
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn page_payload_kind_copy_and_debug() {
        // Arrange
        use crate::scheduler::types::PagePayloadKind;

        let original = PagePayloadKind::ExpertWeight;
        let copied = original; // Copy

        // Assert
        assert_eq!(original, copied);
        let debug = format!("{:?}", PagePayloadKind::KvContext);
        assert!(debug.contains("KvContext"));
    }

    // ---- MemoryResidency: variants, Copy, Debug, Hash ----

    #[test]
    fn memory_residency_variants_in_hashset() {
        // Arrange
        use std::collections::HashSet;
        use crate::scheduler::types::MemoryResidency;

        let all = [MemoryResidency::DeviceLocal, MemoryResidency::HostLocal, MemoryResidency::DiskSwap];

        // Act
        let set: HashSet<MemoryResidency> = all.into_iter().collect();

        // Assert: all 3 variants are distinct
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn memory_residency_copy_trait() {
        // Arrange
        use crate::scheduler::types::MemoryResidency;

        let original = MemoryResidency::DiskSwap;
        let copied = original; // Copy

        // Assert: both usable
        assert_eq!(original, copied);
    }

    // ---- UnifiedVirtualPage: factory methods and predicates ----

    #[test]
    fn unified_virtual_page_kv_construction() {
        // Arrange
        use crate::scheduler::types::UnifiedVirtualPage;
        use gllm_kernels::types::DType;

        // Act
        let page = UnifiedVirtualPage::kv(42, 100, crate::scheduler::types::KvPipeline::Conversation, 3, DType::F32);

        // Assert
        assert_eq!(page.page_id, 42);
        assert!(page.is_on_device());
        assert!(page.is_evictable());
        assert_eq!(page.owner, Some(100));
        assert_eq!(page.logical_index, 3);
    }

    #[test]
    fn unified_virtual_page_system_prompt_not_evictable() {
        // Arrange
        use crate::scheduler::types::UnifiedVirtualPage;
        use gllm_kernels::types::DType;

        // Act
        let page = UnifiedVirtualPage::system_prompt(10, DType::F32);

        // Assert
        assert!(!page.is_evictable());
        assert!(page.is_on_device());
        assert!(page.owner.is_none());
    }

    #[test]
    fn unified_virtual_page_rag_evictable_host_local() {
        // Arrange
        use crate::scheduler::types::UnifiedVirtualPage;
        use gllm_kernels::types::DType;

        // Act
        let page = UnifiedVirtualPage::rag(7, 55, DType::BF16);

        // Assert
        assert!(page.is_evictable());
        assert!(!page.is_on_device()); // RAG pages start on host
        assert_eq!(page.owner, Some(55));
    }

    #[test]
    fn unified_virtual_page_dense_layer_not_evictable() {
        // Arrange
        use crate::scheduler::types::UnifiedVirtualPage;
        use gllm_kernels::types::DType;

        // Act
        let page = UnifiedVirtualPage::dense_layer(99, 5, DType::F32);

        // Assert
        assert!(!page.is_evictable());
        assert!(page.is_on_device());
        assert!(page.owner.is_none());
        assert_eq!(page.logical_index, 5);
    }

    // ---- BackendError: Clone and std::error::Error trait ----

    #[test]
    fn backend_error_clone_preserves_message() {
        // Arrange
        let err = BackendError::Cuda("device lost".into());

        // Act
        let cloned = err.clone();

        // Assert: cloned produces the same Display output
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_unimplemented_is_static_str() {
        // Arrange
        let err = BackendError::Unimplemented("flash_attn_sm70");

        // Act
        let display = format!("{err}");

        // Assert: static str is preserved exactly
        assert_eq!(display, "unimplemented: flash_attn_sm70");
    }

    #[test]
    fn backend_error_other_empty_string() {
        // Arrange
        let err = BackendError::Other(String::new());

        // Act
        let display = format!("{err}");

        // Assert: empty string does not panic or produce garbage
        assert_eq!(display, "backend error: ");
    }

    // ---- EngineContext: from_executor_config round-trip ----

    #[test]
    fn engine_context_from_executor_config_matches_new() {
        // Arrange
        let fwd = GeneratorForwardConfig::default_for_test();

        // Act
        let ctx = EngineContext::from_executor_config(&fwd);

        // Assert: all fields match the default_for_test values
        assert_eq!(ctx.num_layers, fwd.num_layers());
        assert_eq!(ctx.hidden_size, fwd.hidden_size());
        assert_eq!(ctx.kv_page_size, fwd.paged_kv.page_size);
        assert_eq!(ctx.num_kv_heads, fwd.num_kv_heads());
        assert_eq!(ctx.max_seq_len, fwd.max_seq_len());
    }

    #[test]
    fn engine_context_eq_symmetry() {
        // Arrange
        let a = EngineContext::new(8, 512, 4, 2, 1024);
        let b = EngineContext::new(8, 512, 4, 2, 1024);

        // Assert: Eq symmetry holds both directions
        assert!(a == b);
        assert!(b == a);
        assert!(!(a != b));
    }

    // ---- KvCacheHandle: boundary values and zero ----

    #[test]
    fn kv_cache_handle_zero_value() {
        // Arrange
        let handle = KvCacheHandle(0);

        // Assert: zero is a valid handle value
        assert_eq!(handle.0, 0);
        assert_eq!(handle, KvCacheHandle(0));
    }

    #[test]
    fn kv_cache_handle_max_u64() {
        // Arrange
        let handle = KvCacheHandle(u64::MAX);

        // Assert: u64::MAX is a valid handle value
        assert_eq!(handle.0, u64::MAX);
        assert_eq!(handle, KvCacheHandle(u64::MAX));
    }

    // ---- AttentionMaskType: Hash in HashSet ----

    #[test]
    fn attention_mask_type_hash_set_dedup() {
        // Arrange
        use std::collections::HashSet;
        let mut set = HashSet::new();

        // Act: insert both variants
        set.insert(AttentionMaskType::Bidirectional);
        set.insert(AttentionMaskType::Causal);

        // Assert: both coexist
        assert_eq!(set.len(), 2);
        assert!(set.contains(&AttentionMaskType::Bidirectional));
        assert!(set.contains(&AttentionMaskType::Causal));
    }

    // ---- PositionEncoding: Copy and exhaustive equality ----

    #[test]
    fn position_encoding_copy_and_reflexive_equality() {
        // Arrange
        let a = PositionEncoding::Rope;
        let b = a; // Copy

        // Assert: copy preserves value
        assert_eq!(a, b);
        assert_eq!(a, PositionEncoding::Rope);
        assert_ne!(a, PositionEncoding::None);
    }

    // ---- SwapConfig: boundary values ----

    #[test]
    fn swap_config_disabled_threshold_zero() {
        // Arrange
        let cfg = SwapConfig {
            enable_swap: false,
            swap_threshold: 0.0,
            lru_granularity: 0,
        };

        // Assert: disabled config with zero values
        assert!(!cfg.enable_swap);
        assert_eq!(cfg.swap_threshold, 0.0);
        assert_eq!(cfg.lru_granularity, 0);
    }

    #[test]
    fn swap_config_max_threshold() {
        // Arrange
        let cfg = SwapConfig {
            enable_swap: true,
            swap_threshold: 1.0,
            lru_granularity: usize::MAX,
        };

        // Assert: extreme values are stored correctly
        assert!(cfg.enable_swap);
        assert_eq!(cfg.swap_threshold, 1.0);
        assert_eq!(cfg.lru_granularity, usize::MAX);
    }

    // ---- BatchInput: empty batch edge case ----

    #[test]
    fn batch_input_empty_sequences() {
        // Arrange/Act
        let batch = BatchInput {
            sequences: vec![],
        };

        // Assert: empty batch is valid
        assert!(batch.sequences.is_empty());
    }

    // ---- SequenceInput: fused_hidden present and empty tokens ----

    #[test]
    fn sequence_input_with_fused_hidden() {
        // Arrange/Act
        let seq = SequenceInput {
            tokens: vec![100, 200],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(vec![0.1, 0.2, 0.3, 0.4]),
        };

        // Assert: fused_hidden is preserved
        assert!(seq.fused_hidden.is_some());
        let hidden = seq.fused_hidden.as_ref().unwrap();
        assert_eq!(hidden.len(), 4);
        assert!((hidden[0] - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn sequence_input_validate_page_table_with_zero_total_pages() {
        // Arrange: empty page table with zero total_pages
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![]),
            fused_hidden: None,
        };

        // Act/Assert: empty page table is valid even with zero total_pages
        assert!(seq.validate_page_table(0).is_ok());
    }

    // ---- Batch 6: 15 additional tests for coverage improvement ----

    #[test]
    fn batch_context_new_zero_seqs() {
        // Arrange/Act: create a BatchContext with zero sequences
        let ctx = BatchContext::new(0);

        // Assert: buffer is exactly header size (no per-seq data)
        assert_eq!(ctx.num_seqs, 0);
        assert_eq!(ctx.data.len(), 96); // BATCH_CTX_HEADER_SIZE
        assert_eq!(ctx.max_batch_size, 0);
        assert!(!ctx.has_v2_extension);
        assert!(ctx.seq_mapping.is_empty());
    }

    #[test]
    fn batch_context_new_single_seq_buffer_size() {
        // Arrange/Act: create a BatchContext with 1 sequence
        let ctx = BatchContext::new(1);

        // Assert: buffer is header (96) + 1 * seq_meta_stride (64) = 160
        assert_eq!(ctx.num_seqs, 1);
        assert_eq!(ctx.data.len(), 96 + 64);
    }

    #[test]
    fn batch_context_new_multiple_seqs_buffer_size() {
        // Arrange/Act: create a BatchContext with 8 sequences
        let ctx = BatchContext::new(8);

        // Assert: buffer is header (96) + 8 * 64 = 608
        assert_eq!(ctx.num_seqs, 8);
        assert_eq!(ctx.data.len(), 96 + 8 * 64);
    }

    #[test]
    fn batch_context_clone_preserves_state() {
        // Arrange
        let mut ctx = BatchContext::new(4);
        ctx.set_num_seqs(2);

        // Act
        let cloned = ctx.clone();

        // Assert: cloned copy matches original
        assert_eq!(cloned.num_seqs, ctx.num_seqs);
        assert_eq!(cloned.max_batch_size, ctx.max_batch_size);
        assert_eq!(cloned.data.len(), ctx.data.len());
    }

    #[test]
    fn batch_context_with_v2_extension_larger_buffer() {
        // Arrange/Act: create BatchContext with v2 extension, max 4 slots, 2 initial
        let ctx = BatchContext::with_v2_extension(4, 2);

        // Assert: buffer is larger than basic header + per-seq data
        // header (96) + max_batch_size * 64 + extension size
        assert!(ctx.has_v2_extension);
        assert_eq!(ctx.num_seqs, 2);
        assert_eq!(ctx.max_batch_size, 4);
        assert!(ctx.data.len() > 96 + 4 * 64);
    }

    #[test]
    fn effective_kv_max_seq_len_identity_passthrough() {
        // Arrange: various sequence lengths from real model configs
        let sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

        // Act/Assert: function is a pure passthrough
        for &size in &sizes {
            assert_eq!(
                crate::engine::executor_types::effective_kv_max_seq_len(size),
                size
            );
        }
    }

    #[test]
    fn backend_error_hip_display_format() {
        // Arrange
        let err = BackendError::Hip("device not found".into());

        // Act
        let display = format!("{err}");

        // Assert
        assert_eq!(display, "HIP error: device not found");
    }

    #[test]
    fn backend_error_metal_display_format() {
        // Arrange
        let err = BackendError::Metal("buffer allocation failed".into());

        // Act
        let display = format!("{err}");

        // Assert
        assert_eq!(display, "Metal error: buffer allocation failed");
    }

    #[test]
    fn backend_error_cpu_display_format() {
        // Arrange
        let err = BackendError::Cpu("stack overflow detected".into());

        // Act
        let display = format!("{err}");

        // Assert
        assert_eq!(display, "CPU error: stack overflow detected");
    }

    #[test]
    fn swap_config_clone_independence() {
        // Arrange
        let original = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.75,
            lru_granularity: 16,
        };

        // Act: clone and mutate
        let mut cloned = original.clone();
        cloned.enable_swap = false;
        cloned.swap_threshold = 0.0;

        // Assert: original unaffected
        assert!(original.enable_swap);
        assert!((original.swap_threshold - 0.75).abs() < 1e-6);
        assert_eq!(original.lru_granularity, 16);
        assert!(!cloned.enable_swap);
    }

    #[test]
    fn kv_cache_handle_sorting_order() {
        // Arrange: unsorted handles
        let mut handles = vec![
            KvCacheHandle(300),
            KvCacheHandle(1),
            KvCacheHandle(42),
            KvCacheHandle(0),
            KvCacheHandle(99),
        ];

        // Act: sort by inner value
        handles.sort_by_key(|h| h.0);

        // Assert: ascending order
        assert_eq!(handles[0], KvCacheHandle(0));
        assert_eq!(handles[1], KvCacheHandle(1));
        assert_eq!(handles[2], KvCacheHandle(42));
        assert_eq!(handles[3], KvCacheHandle(99));
        assert_eq!(handles[4], KvCacheHandle(300));
    }

    #[test]
    fn engine_context_eq_transitivity() {
        // Arrange: three equal contexts
        let a = EngineContext::new(6, 768, 16, 12, 2048);
        let b = EngineContext::new(6, 768, 16, 12, 2048);
        let c = EngineContext::new(6, 768, 16, 12, 2048);

        // Assert: transitivity — a == b && b == c implies a == c
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    #[test]
    fn engine_context_hash_consistent_with_eq() {
        // Arrange: two equal contexts
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = EngineContext::new(12, 1024, 8, 4, 2048);
        let b = EngineContext::new(12, 1024, 8, 4, 2048);

        // Act: compute hashes
        let mut hasher_a = DefaultHasher::new();
        a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();

        let mut hasher_b = DefaultHasher::new();
        b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();

        // Assert: equal objects produce equal hashes (Hash/Eq contract)
        assert_eq!(a, b);
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn page_state_all_seven_variants_exhaustive_match() {
        // Arrange: collect all 7 known PageState variants
        let all = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];

        // Act/Assert: exhaustive match proves no variant is missed
        let mut count = 0;
        for state in &all {
            match state {
                PageState::Free => count += 1,
                PageState::Active => count += 1,
                PageState::Standby => count += 1,
                PageState::SwappedOut => count += 1,
                PageState::Warm => count += 1,
                PageState::Protected => count += 1,
                PageState::Swapped => count += 1,
            }
        }
        assert_eq!(count, 7);
    }

    #[test]
    fn request_kind_all_three_variants_exhaustive_match() {
        // Arrange: all 3 RequestKind variants
        let all = [RequestKind::Chat, RequestKind::Embedding, RequestKind::Rerank];

        // Act/Assert: exhaustive match
        let mut chat_count = 0;
        let mut embed_count = 0;
        let mut rerank_count = 0;
        for kind in &all {
            match kind {
                RequestKind::Chat => chat_count += 1,
                RequestKind::Embedding => embed_count += 1,
                RequestKind::Rerank => rerank_count += 1,
            }
        }
        assert_eq!(chat_count, 1);
        assert_eq!(embed_count, 1);
        assert_eq!(rerank_count, 1);
    }

    // ---- Batch 7: 13 additional edge-case tests ----

    #[test]
    fn sampling_config_zero_temperature() {
        // Arrange: temperature=0.0 (greedy decoding)
        let cfg = SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
        };

        // Assert: zero temperature is valid
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_k, 1);
        assert_eq!(cfg.top_p, 1.0);
    }

    #[test]
    fn sampling_config_zero_top_p() {
        // Arrange: top_p=0.0 (edge case, all tokens excluded)
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
        };

        // Assert: zero top_p is stored as-is
        assert_eq!(cfg.top_p, 0.0);
    }

    #[test]
    fn pipelined_virtual_page_id_construction() {
        // Arrange
        use crate::scheduler::types::PipelinedVirtualPageId;

        // Act
        let pvp = PipelinedVirtualPageId {
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 7,
        };

        // Assert
        assert_eq!(pvp.pipeline, crate::scheduler::types::KvPipeline::Conversation);
        assert_eq!(pvp.sequence_id, 42);
        assert_eq!(pvp.logical_index, 7);
    }

    #[test]
    fn pipelined_virtual_page_id_equality_and_hash() {
        // Arrange
        use std::collections::HashSet;
        use crate::scheduler::types::PipelinedVirtualPageId;
        use crate::scheduler::types::KvPipeline;

        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 0,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 0,
        };
        let c = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 1,
            logical_index: 0,
        };

        // Assert: equality
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Assert: hash deduplication
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
        set.insert(c);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn page_metadata_default_values() {
        // Arrange/Act
        let meta = crate::scheduler::types::PageMetadata::default();

        // Assert: all default fields are zero/empty/standby
        assert_eq!(meta.page_id, 0);
        assert!(meta.sequence_id.is_none());
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(!meta.is_lir);
        assert_eq!(meta.state, PageState::Standby);
        assert!(meta.warm_until.is_none());
        assert!(meta.swap_in_time.is_none());
    }

    #[test]
    fn weight_tier_variants_in_hashset() {
        // Arrange
        use std::collections::HashSet;
        use crate::scheduler::types::WeightTier;

        // Act
        let set: HashSet<WeightTier> = [WeightTier::Hot, WeightTier::Warm, WeightTier::Cold]
            .into_iter()
            .collect();

        // Assert: all 3 tiers are distinct
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn unified_virtual_page_expert_construction() {
        // Arrange
        use crate::scheduler::types::UnifiedVirtualPage;
        use gllm_kernels::types::DType;

        // Act
        let page = UnifiedVirtualPage::expert(55, 3, 12, DType::BF16);

        // Assert
        assert_eq!(page.page_id, 55);
        assert!(page.is_on_device());
        assert!(page.is_evictable());
        assert_eq!(page.expert_id, Some(3));
        assert_eq!(page.layer_idx, Some(12));
        assert!(page.owner.is_none());
    }

    #[test]
    fn sequence_input_validate_page_table_single_valid_page() {
        // Arrange: single page table entry with total_pages=1
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };

        // Act/Assert: page_id=0 is valid when total_pages=1
        assert!(seq.validate_page_table(1).is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_rejects_at_boundary() {
        // Arrange: page_id=1 is out of bounds when total_pages=1
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![1]),
            fused_hidden: None,
        };

        // Act/Assert
        assert!(seq.validate_page_table(1).is_err());
    }

    #[test]
    fn logits_handle_empty_vec() {
        // Arrange/Act: empty logits is a valid handle
        let handle = LogitsHandle { data: vec![] };

        // Assert
        assert!(handle.data.is_empty());
    }

    #[test]
    fn logits_handle_clone_preserves_data() {
        // Arrange
        let handle = LogitsHandle {
            data: vec![0.5, 1.5, 2.5],
        };

        // Act
        let cloned = handle.clone();

        // Assert: clone preserves data
        assert_eq!(cloned.data.len(), 3);
        assert_eq!(cloned.data[0], 0.5);
        assert_eq!(cloned.data[2], 2.5);
    }

    #[test]
    fn attention_topology_clone_preserves_mask_type() {
        // Arrange
        use std::sync::Arc;
        let geo = Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 64,
            num_layers: 4,
            vocab_size: 100,
            intermediate_size: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        });
        let topo = AttentionTopology::causal(geo);

        // Act
        let cloned = topo.clone();

        // Assert: mask type and geometry preserved
        assert_eq!(cloned.mask_type, AttentionMaskType::Causal);
        assert_eq!(cloned.num_heads(), 4);
        assert_eq!(cloned.head_dim(), 16);
    }

    #[test]
    fn rope_config_interleaved_and_precompute_differ() {
        // Arrange: same theta/scale but different interleaved
        let base = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let interleaved = RoPEConfig {
            interleaved: true,
            ..base
        };
        let precomputed = RoPEConfig {
            precompute: true,
            ..base
        };

        // Assert: both fields independently affect equality
        assert_ne!(base, interleaved);
        assert_ne!(base, precomputed);
        assert_ne!(interleaved, precomputed);
    }
}

