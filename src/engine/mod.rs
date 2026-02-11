//! Layer 3: Engine (skeleton).

pub mod executor;
pub mod pipeline;

pub use pipeline::{PipelineError, UnifiedPipeline};

// Re-export scheduler types from the main scheduler module
pub use crate::scheduler::batcher::ContinuousBatcher;
pub use crate::scheduler::types::{GroupState, RequestKind, SequenceGroup};
pub use crate::scheduler::{PagedScheduler, ScheduledBatch};
pub use gllm_kernels::kernel_types::{PageId, PageState, RequestId};
