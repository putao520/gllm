//! Attention mechanisms for gllm
//!
//! Re-exports from gllm-kernels

mod deterministic;

pub use gllm_kernels::{
    AccumulatorConfig,
    AttentionConfig,
    BlockManager,
    BlockTable,
    FlashAttentionConfig,
    FusedPagedAttention,
    HierarchicalAccumulator,
    HierarchicalFlashAttention,
    KahanAccumulator,
    KahanSum,
    KVBlock,
    LogSpaceSoftmax,
    PagedAttention,
    PagedAttentionConfig,
    PagedKVCache,
    StableAccumulator,
    StableRowState,
    log_add_exp,
    log_sum_exp,
};

pub use gllm_kernels::ops::flash_attention::{
    DeterministicConfig,
    HierarchicalFlashConfig,
};

pub use deterministic::{DeterministicConfigExt, DeterministicGuard};
pub use gllm_kernels::PagedKVCache as PagedKvCache;
pub use gllm_kernels::StableRowState as LogSpaceRowState;
