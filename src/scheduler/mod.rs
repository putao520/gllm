//! Scheduler building blocks (HGAL).
//!
//! This module hosts the gang-aware, LIRS-inspired scheduling logic that
//! complements the engine layer. It is intentionally independent from the
//! backend so it can be unit-tested without GPU involvement.

pub mod allocator;
pub mod batcher;
pub mod hgal;
pub mod paged_scheduler;
pub mod sequence;
pub mod types;
pub mod vllm2024;

pub use allocator::BlockAllocator;
pub use batcher::{BatchAction, BatchResult, ContinuousBatcher, ScheduledBatch};
pub use hgal::{HGALConfig, HGALScheduler};
pub use paged_scheduler::{BlockTable, PagedScheduler, SchedulerOutput};
pub use sequence::{Sequence, SequenceState};
pub use types::{GroupState, PageMetadata, SequenceGroup};
