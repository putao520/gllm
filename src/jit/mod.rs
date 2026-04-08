//! JIT 编译时硬件探测与优化
//!
//! 本模块实现 SPEC §12 "空间异构流派与动态块式计算图" 规定的完整硬件感知优化体系：
//!
//! - **§12.1 空间异构 Sub-Batching**: 多流分发到不同硬件分片
//! - **§12.4 黄金装筒规则**: 硬件感知型 Shape Bucketing（禁止预设硬编码数组）
//! - **§12.6 IR 约束变量**: 硬件探测 → JIT 编译器强数学约束变量
//!
//! ## 核心原则
//!
//! - **严禁预设硬编码数组**：禁止使用 `[128, 512, 1024, 2048]` 等静态 Bucket
//! - **真实物理探测**：通过 micro-benchmark 测定寄存器溢出、SMEM 满载、L2 Thrashing 阈值
//! - **黄金装筒塌缩**：将任意 SEQ 长度映射到探测出的"黄金尺寸"（Golden Sizes）
//! - **零退化原则**：禁止 Padding 补零，使用 Ragged Compaction
//! - **运行时演化**：JIT Director Daemon 持续观测 SEQ 分布，热插拔新 Bucket

pub mod profiler;
pub mod histogram;
pub mod variant_registry;
pub mod ragged;
pub mod epilogue;
pub mod gate_skip;
pub mod residual_bypass;
pub mod prefetch;
pub mod sink_tracker;
pub mod compiler_constraints;
pub mod golden_bucket;
pub mod sub_batch;
pub mod epilogue_subsystem;

// Re-export 核心类型
pub use profiler::{ProbeConfig, ProbeResult, LatencyProfiler, ProbeError};
pub use histogram::{SeqBucket, SeqHistogram, HistogramSnapshot};
pub use variant_registry::{
    CodeSection, MechanismId, SpecPhase, VariantKey, CompiledVariant,
    L1iBudgetExceeded, VariantRegistry,
};
pub use ragged::{
    CompactPlatform, RequestActiveMask, CompactIndex, CompactDecision,
    CompactData, ScatterWriter, RaggedCompaction, COMPACT_THRESHOLD,
};
pub use epilogue::{
    EpilogueSignal, TelemetryAggregator,
    GateFirstSkipConfig, GateSkipDecision, GateFirstSkipDetector,
    ResidualBypassConfig, ResidualBypassDecision, ResidualBypassDetector,
    SinkDetectionConfig, AttentionPattern, SinkDetector,
    ExpertThermalState, ExpertThermalTracker,
    SpecScheduleAdvice, SpecScheduleSignal,
};
pub use gate_skip::{
    GateFirstSkipLayer, LayerSkipRecord,
    BatchSkipSummary, BatchSkipAdvice,
};
pub use residual_bypass::{
    ResidualBypassLayer, LayerBypassRecord,
    BypassStats, BypassAdvice,
};
pub use prefetch::{
    PrefetchConfig, PrefetchAdvice, CentroidPrefetch,
    CentroidRecord, PrefetchStats,
};
pub use sink_tracker::{
    SinkTracker, LayerAttentionRecord,
};
pub use compiler_constraints::{
    CompilerConstraints, TileBits, GpuSmPartition,
};
pub use golden_bucket::{
    GoldenSize, GoldenBucketRegistry, EvolveDecision,
};
pub use sub_batch::{
    GraphShape, HardwareKind, SubBatch, HardwarePartition,
    SubBatchDispatcher, DispatchPlan, DispatchReason,
};
pub use epilogue_subsystem::{
    EpilogueConfig, EpilogueSubsystem, EpilogueBatchSummary,
    RequestEpilogueDecision,
};
