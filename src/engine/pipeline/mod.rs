//! Pipeline Parallel (PP) 基础设施 (REQ-DIST-018, REQ-DIST-019, REQ-DIST-020, REQ-DIST-023, REQ-DIST-029, REQ-DIST-030)
//!
//! nccl feature-gated: 非 nccl 构建零影响。
//!
//! - `config`: PipelineConfig — PP 配置与 stage 划分 (REQ-DIST-018)
//! - `topology`: Topology2D + Topology3D — 2D/3D 并行拓扑与通信编排 (REQ-DIST-029, REQ-DIST-030)
//! - `micro_batch`: MicroBatchScheduler — 微批次切分与交错 PP 调度 (REQ-DIST-020, REQ-DIST-023)
//! - `activation_xfer`: ActivationTransport — stage 间激活传输 (REQ-DIST-023)

#[cfg(feature = "nccl")]
pub mod config;
#[cfg(feature = "nccl")]
pub mod topology;
#[cfg(feature = "nccl")]
pub mod micro_batch;
#[cfg(feature = "nccl")]
pub mod activation_xfer;
