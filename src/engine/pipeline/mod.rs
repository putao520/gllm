//! Pipeline Parallel (PP) 基础设施 (REQ-DIST-018, REQ-DIST-019, REQ-DIST-029)
//!
//! nccl feature-gated: 非 nccl 构建零影响。
//!
//! - `config`: PipelineConfig — PP 配置与 stage 划分 (REQ-DIST-018)
//! - `topology`: Topology2D — 2D 并行 (TP+PP) 拓扑与通信编排 (REQ-DIST-029)

#[cfg(feature = "nccl")]
pub mod config;
#[cfg(feature = "nccl")]
pub mod topology;
