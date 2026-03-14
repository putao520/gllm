//! Layer 3: Engine (skeleton).

pub mod executor;
pub mod pipeline;

pub use pipeline::{PipelineError, UnifiedPipeline};

// Re-export scheduler types
pub use crate::scheduler::batcher::ContinuousBatcher;
pub use crate::scheduler::types::{GroupState, RequestKind, SequenceGroup};
pub use crate::scheduler::{PagedScheduler, ScheduledBatch};
pub use crate::scheduler::types::{PageId, PageState, RequestId};

// Re-export engine types
pub use executor::{
    AttentionMaskType, AttentionTopology, BackendError, BatchInput, GeneratorForwardConfig,
    KvCacheConfig, KvCacheHandle, LogitsHandle, PositionEncoding, SamplingConfig, SequenceInput,
    SwapConfig,
};
