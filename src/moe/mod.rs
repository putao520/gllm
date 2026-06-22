//! MoE (Mixture-of-Experts) 生产级实现 (SPEC §14-§15)
//!
//! ## 模块架构
//! - `routing`: §15.1 核内分发零启动开销 — Top-K 路由 + 容量因子 + 负载均衡
//! - `thermal`: §15.4 冷板凳专家封杀与复活 — 热度追踪 + Deopt + OSR Bailout
//! - `prefetch`: §15.2 TurboQuant + 预瞄 — 专家权重异步预取 + Pipeline 隐藏
//! - `dispatch`: §15.3 CPU/GPU 真正并行 — Core Disaggregation 硬件分发
//! - `hot_patch`: §14.4 Hot JMP Patching — 全局物理共识级 JIT 代码热修补
//! - `prefetch_pipeline`: §14.5 RDMA/PCIe 流水线预取编排
//! - `distributed_dispatch`: REQ-DIST-014 分布式 MoE 调度决策 (nccl feature-gated)
//! - `eplb`: REQ-DIST-015 EPLB 专家负载均衡 (nccl feature-gated)

pub mod dispatch;
pub mod fault_handler;
pub mod hot_patch;
pub mod prefetch;
pub mod prefetch_pipeline;
pub mod routing;
pub mod thermal;

#[cfg(feature = "nccl")]
pub mod distributed_dispatch;
#[cfg(feature = "nccl")]
pub mod eplb;

pub use dispatch::{ExpertHardwareAssignment, MoeDispatchPlan, MoeHardwareDispatcher};
pub use fault_handler::{ExpertFault, ExpertFaultHandler, FaultResolution, FaultStats};
pub use hot_patch::{
    HotPatchManager, HotPatchSummary, PatchInstruction, PatchOperation, PatchResult,
    PatchSafetyCheck, PatchTarget,
};
pub use prefetch::{ExpertPrefetchRequest, ExpertWeightLocation, ExpertWeightPrefetcher};
pub use prefetch_pipeline::{PipelinePrefetchEntry, PrefetchPipeline, PrefetchStage, PrefetchTransferKind};
pub use routing::{
    moe_dispatch, softmax, topk_indices, topk_with_weights, ExpertLoadBalancer, ExpertRouteConfig,
    ExpertRouteTable, ExpertUtilizationStats, TokenRoute,
};
pub use thermal::{
    DeoptHandlingResult, DeoptRequest, EvictionDecision, ExpertHeatLevel, ExpertHeatState,
    ExpertResidency, ExpertThermalManager, ThermalSummary, WorkingSetTracker,
};
