pub mod callback_slot;
#[cfg(feature = "nccl")]
pub mod comm_schedule;
#[cfg(feature = "nccl")]
pub mod context_parallel;
pub mod compute;
pub mod dispatch;
pub mod inference;
pub mod kv;
pub mod model_context;
pub mod observability;
pub mod sg_callback_handle;

// L3: PD 分离 + KV 分布 (nccl feature-gated)
#[cfg(feature = "nccl")]
pub use dispatch::pd_disagg;
#[cfg(feature = "nccl")]
pub use kv::kv_transfer;
#[cfg(feature = "nccl")]
pub use kv::kv_distribution;
