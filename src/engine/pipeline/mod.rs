//! Pipeline Parallel (PP) 基础设施 (REQ-DIST-018~027, REQ-DIST-024~025, REQ-DIST-028, REQ-DIST-031)
//!
//! nccl feature-gated: 非 nccl 构建零影响。
//!
//! - `config`: PipelineConfig — PP 配置与 stage 划分 (REQ-DIST-018)
//! - `topology`: Topology2D + Topology3D — 2D/3D 并行拓扑与通信编排 (REQ-DIST-029, REQ-DIST-030)
//! - `micro_batch`: MicroBatchScheduler — 微批次切分与交错 PP 调度 (REQ-DIST-020, REQ-DIST-023)
//! - `activation_xfer`: ActivationTransport + PipelineTransferFuture — stage 间激活传输 (REQ-DIST-021, REQ-DIST-023)
//! - `scheduler`: PipelineScheduler + CommComputeOverlap + SmPartitionConfig + QuantizedActivationConfig + PipelineKvCacheManager — GPipe/1F1B 调度 + 通信计算重叠 + SM 分区 + 量化激活传输 + PP KV Cache (REQ-DIST-021, REQ-DIST-022, REQ-DIST-026, REQ-DIST-027)
//! - `interleaved`: Interleaved1F1B — 交错流水线调度 (REQ-DIST-022)
//! - `bubble`: BubbleAnalyzer + BubbleMetrics — Pipeline bubble 分析与可视化指标 (REQ-DIST-024)
//! - `adaptive`: AdaptiveMicroBatchSizer — 动态微批次大小调整 (REQ-DIST-025)
//! - `gradient_sync`: GradientSync — PP 梯度回传与权重同步 (REQ-DIST-028)
//! - `pd_bridge`: PdPipelineBridge — PP 与 PD 分离协同 (REQ-DIST-031)

#[cfg(feature = "nccl")]
pub mod config;
#[cfg(feature = "nccl")]
pub mod topology;
#[cfg(feature = "nccl")]
pub mod micro_batch;
#[cfg(feature = "nccl")]
pub mod activation_xfer;
#[cfg(feature = "nccl")]
pub mod scheduler;
#[cfg(feature = "nccl")]
pub mod interleaved;
#[cfg(feature = "nccl")]
pub mod bubble;
#[cfg(feature = "nccl")]
pub mod adaptive;
#[cfg(feature = "nccl")]
pub mod gradient_sync;
#[cfg(feature = "nccl")]
pub mod pd_bridge;
