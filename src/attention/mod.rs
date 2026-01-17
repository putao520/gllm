//! Attention mechanisms for gllm
//!
//! Re-exports from gllm-kernels.
//!
//! Note: Per ADR-001, attention is dispatched through gllm-kernels.
//! Use `KernelDispatcher` for GPU-accelerated attention operations.

mod deterministic;

// Re-export stable accumulator utilities (numerical stability)
pub use gllm_kernels::ops::stable_accumulator::{
    AccumulatorConfig,
    HierarchicalAccumulator,
    KahanAccumulator,
    KahanSum,
    StableAccumulator,
    StableRowState,
};

// Re-export softmax utilities
pub use gllm_kernels::ops::softmax::{
    LogSpaceSoftmax,
    log_add_exp,
    log_sum_exp,
};

// Re-export attention config types
pub use gllm_kernels::types::{
    AttentionConfig,
    PagedAttentionConfig,
};

// Re-export KernelDispatcher configs for GPU-accelerated attention
pub use gllm_kernels::{
    FlashAttentionConfig,
    PagedAttentionConfig as KernelPagedAttentionConfig,
    KernelDispatcher,
};

pub use deterministic::{DeterministicConfigExt, DeterministicGuard};
pub use gllm_kernels::ops::stable_accumulator::StableRowState as LogSpaceRowState;
