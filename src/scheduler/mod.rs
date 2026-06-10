//! Scheduler building blocks (HGAL).
//!
//! This module hosts the gang-aware, LIRS-inspired scheduling logic that
//! complements the engine layer. It is intentionally independent from the
//! backend so it can be unit-tested without GPU involvement.

pub mod allocator;
pub mod batcher;
pub mod hgal;
pub mod jit_types;
pub mod memory_manager;
pub mod observer;
pub mod paged_scheduler;
pub mod policy;
pub mod prefix_index;
pub mod sequence;
pub mod types;
pub mod vllm2024;
pub mod telemetry;
pub mod chunked_prefill;
pub mod compact;
pub mod request_state;
pub mod kv_optimizer;
pub mod dma_helpers;
pub mod migration_actor;
pub mod fault_recovery;
pub mod nvme_swap;
pub mod eviction_worker;
pub mod swap_in_worker;
pub mod three_tier_swap;
pub mod weight_paging;

pub use allocator::BlockAllocator;
pub use batcher::{BatchAction, BatchResult, ContinuousBatcher, ScheduledBatch};
pub use hgal::{HGALConfig, HGALScheduler};
pub use jit_types::{SchedulerDecision, SystemState};
pub use memory_manager::{
    EvictionPolicy, GlobalMemoryManager, MemoryManagerError, PageLocation, PageTable, PrefillPlan,
    SessionId, SessionKvCache, Tier, TierManager, TierUsage, VirtualPageId,
};
pub use observer::{
    BasicObserver, EvictionReason, ObserverError, RuntimeObserver, WeightPageTelemetryEvent,
};
pub use crate::sensors::{CodecStats, CompressionTelemetry};
pub use paged_scheduler::{
    find_donor, BlockTable, LayerAllocHint, LayerPageKey, LayerPageTable, PagedScheduler,
    SchedulerError, SchedulerOutput,
};
pub use policy::{AbsolutePolicy, PolicyConfig, PolicyVariant, SchedulingPolicy};
pub use prefix_index::{KvPrefixIndex, PrefixMatch, TokenId};
pub use sequence::{Sequence, SequenceState};
pub use types::{
    BatchOrderPolicy, EvictionPriority, GroupState, KvPipeline, PageId, PageMetadata, PageState,
    PhysicalId, PipelinedVirtualPageId, RequestId, RequestKind, SequenceGroup, StorageKey,
    WeightTier,
};
pub use fault_recovery::{
    execute_step_fault_plan, generate_step_fault_plan, FaultAction, FaultRecoveryError,
    FaultRecoveryHandler, FaultRecoveryStats, PageFault, StepFaultPlan, WeightPageTable,
};
pub use telemetry::{SequenceTelemetry};
pub use three_tier_swap::{
    ThreeTierSwapConfig, ThreeTierSwapCoordinator, ThreeTierSwapStats, TierMigration,
    TierMigrationPlan, TierMigrationReason,
};
pub use weight_paging::{
    DefragPlan, MultiGpuMigrationRequest, MultiGpuPageMigrator, PcieDmaStatus, PcieDmaTransfer,
    PrefetchTrigger, QuantPageParams, QuantWeightPage, QuantWeightPrefetchQueue,
    WeightPageDefragmenter, WeightPageDistribution, WeightPageDtypeHandler,
};
