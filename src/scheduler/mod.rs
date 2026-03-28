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

pub use allocator::BlockAllocator;
pub use batcher::{BatchAction, BatchResult, ContinuousBatcher, ScheduledBatch};
pub use hgal::{HGALConfig, HGALScheduler};
pub use jit_types::{SchedulerDecision, SystemState};
pub use memory_manager::{
    EvictionPolicy, GlobalMemoryManager, MemoryManagerError, PageLocation, PageTable, PrefillPlan,
    SessionId, SessionKvCache, Tier, TierManager, TierUsage, VirtualPageId,
};
pub use observer::{BasicObserver, ObserverError, RuntimeObserver};
pub use paged_scheduler::{BlockTable, PagedScheduler, SchedulerError, SchedulerOutput};
pub use policy::{PolicyVariant, SchedulingPolicy};
pub use prefix_index::{KvPrefixIndex, PrefixMatch, TokenId};
pub use sequence::{Sequence, SequenceState};
pub use types::{
    BatchOrderPolicy, GroupState, KvPipeline, PageId, PageMetadata, PageState, PhysicalId,
    PipelinedVirtualPageId, RequestId, RequestKind, SequenceGroup, StorageKey,
};
pub use telemetry::{SequenceTelemetry};
