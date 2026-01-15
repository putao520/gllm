//! Attention mechanisms for gllm
//!
//! Re-exports from gllm-kernels

mod deterministic;

// Re-export from ops submodules
pub use gllm_kernels::ops::stable_accumulator::{
    AccumulatorConfig,
    HierarchicalAccumulator,
    KahanAccumulator,
    KahanSum,
    StableAccumulator,
    StableRowState,
};
pub use gllm_kernels::ops::paged_attention::{
    BlockManager,
    BlockTable,
    KVBlock,
    PagedAttention,
    PagedKVCache,
};
pub use gllm_kernels::ops::flash_attention::{
    DeterministicConfig,
    FlashAttentionConfig,
    FusedPagedAttention,
    HierarchicalFlashAttention,
    HierarchicalFlashConfig,
};
pub use gllm_kernels::ops::softmax::{
    LogSpaceSoftmax,
    log_add_exp,
    log_sum_exp,
};
pub use gllm_kernels::types::{
    AttentionConfig,
    PagedAttentionConfig,
};

pub use deterministic::{DeterministicConfigExt, DeterministicGuard};
pub use gllm_kernels::ops::paged_attention::PagedKVCache as PagedKvCache;
pub use gllm_kernels::ops::stable_accumulator::StableRowState as LogSpaceRowState;
