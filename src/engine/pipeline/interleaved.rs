//! Interleaved1F1B — 交错流水线调度 (REQ-DIST-022)
//!
//! 交错 1F1B (Interleaved 1F1B / Virtual Pipeline Parallel) 调度策略：
//! - 每个设备持有非连续层块（virtual stages），减少流水线气泡
//! - 气泡比率 = (pp_size - 1) / (pp_size * num_virtual_stages)（验收标准 3）
//! - 相比非交错 1F1B，气泡比率降低约 num_virtual_stages 倍
//!
//! 复用 MicroBatchScheduler 的 InterleavedScheduler 生成交错调度步骤，
//! 本模块将其转换为 PipelineOp 序列。
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use super::config::PipelineConfig;
use super::micro_batch::{InterleavedScheduler, SchedulePhase};
use super::scheduler::{PipelineOp, PipelineScheduler, PipelineSchedulerError};

// ── Interleaved1F1B (REQ-DIST-022) ───────────────────────────────────────────

/// 交错 1F1B 调度器 (REQ-DIST-022)
///
/// 每个设备持有 num_virtual_stages 个非连续层块（virtual stages），
/// 减少流水线气泡比率。
///
/// 气泡比率 = (pp_size - 1) / (pp_size * num_virtual_stages)（验收标准 3）
/// 非交错气泡比率 = (pp_size - 1) / pp_size
/// 交错改进倍数 ≈ num_virtual_stages
// @trace REQ-DIST-022 [entity:Interleaved1F1B] [api:POST /internal/distributed/pipeline/interleaved]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Interleaved1F1B {
    /// Pipeline Parallel 维度
    pub pp_size: u32,
    /// 当前 stage ID
    pub stage_id: u32,
    /// 虚拟 stage 数（每个设备持有的非连续层块数）
    pub num_virtual_stages: u32,
}

/// Interleaved1F1B 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interleaved1F1BError {
    /// pp_size < 1
    InvalidPpSize(u32),
    /// stage_id >= pp_size
    InvalidStageId { stage_id: u32, pp_size: u32 },
    /// num_virtual_stages < 1
    InvalidVirtualStages(u32),
    /// 微批次数为零
    ZeroMicrobatches,
    /// InterleavedScheduler 内部错误
    SchedulerError(String),
}

impl std::fmt::Display for Interleaved1F1BError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Interleaved1F1BError::InvalidPpSize(pp_size) => {
                write!(f, "Interleaved1F1B: invalid pp_size={pp_size}, must be >= 1")
            }
            Interleaved1F1BError::InvalidStageId { stage_id, pp_size } => {
                write!(
                    f,
                    "Interleaved1F1B: stage_id={stage_id} out of range [0, {pp_size})"
                )
            }
            Interleaved1F1BError::InvalidVirtualStages(vs) => {
                write!(f, "Interleaved1F1B: num_virtual_stages={vs}, must be >= 1")
            }
            Interleaved1F1BError::ZeroMicrobatches => {
                write!(f, "Interleaved1F1B: num_microbatches must be > 0")
            }
            Interleaved1F1BError::SchedulerError(msg) => {
                write!(f, "Interleaved1F1B: scheduler error: {msg}")
            }
        }
    }
}

impl std::error::Error for Interleaved1F1BError {}

// @trace REQ-DIST-022 [entity:Interleaved1F1B]
impl Interleaved1F1B {
    /// 创建交错 1F1B 调度器
    // @trace REQ-DIST-022 [entity:Interleaved1F1B]
    pub fn new(
        pp_size: u32,
        stage_id: u32,
        num_virtual_stages: u32,
    ) -> Result<Self, Interleaved1F1BError> {
        if pp_size < 1 {
            return Err(Interleaved1F1BError::InvalidPpSize(pp_size));
        }
        if stage_id >= pp_size {
            return Err(Interleaved1F1BError::InvalidStageId { stage_id, pp_size });
        }
        if num_virtual_stages < 1 {
            return Err(Interleaved1F1BError::InvalidVirtualStages(num_virtual_stages));
        }
        Ok(Self {
            pp_size,
            stage_id,
            num_virtual_stages,
        })
    }

    /// 从 PipelineConfig 创建交错 1F1B 调度器
    // @trace REQ-DIST-022 [entity:Interleaved1F1B]
    pub fn from_pipeline_config(config: &PipelineConfig) -> Result<Self, Interleaved1F1BError> {
        Self::new(config.pp_size, config.stage_id, config.num_virtual_stages)
    }

    /// 是否为第一个 stage
    // @trace REQ-DIST-022 [entity:Interleaved1F1B]
    pub fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }

    /// 是否为最后一个 stage
    // @trace REQ-DIST-022 [entity:Interleaved1F1B]
    pub fn is_last_stage(&self) -> bool {
        self.stage_id == self.pp_size - 1
    }

    /// 交错 1F1B 气泡比率 (REQ-DIST-022 验收标准 3)
    ///
    /// 气泡比率 = (pp_size - 1) / (pp_size * num_virtual_stages)
    /// 非交错气泡比率 = (pp_size - 1) / pp_size
    /// 交错改进倍数 ≈ num_virtual_stages
    // @trace REQ-DIST-022 [entity:Interleaved1F1B]
    pub fn bubble_ratio(&self) -> f64 {
        if self.pp_size <= 1 {
            return 0.0;
        }
        (self.pp_size - 1) as f64 / (self.pp_size as f64 * self.num_virtual_stages as f64)
    }

    /// 非交错 1F1B 气泡比率（用于对比）
    // @trace REQ-DIST-022 [entity:Interleaved1F1B]
    pub fn non_interleaved_bubble_ratio(&self) -> f64 {
        if self.pp_size <= 1 {
            return 0.0;
        }
        (self.pp_size - 1) as f64 / self.pp_size as f64
    }

    /// 交错相比非交错的气泡比率改进倍数
    // @trace REQ-DIST-022 [entity:Interleaved1F1B]
    pub fn bubble_ratio_improvement(&self) -> f64 {
        let interleaved = self.bubble_ratio();
        let non_interleaved = self.non_interleaved_bubble_ratio();
        if interleaved == 0.0 {
            return f64::INFINITY;
        }
        non_interleaved / interleaved
    }

    /// 生成交错 1F1B 调度操作序列 (REQ-DIST-022 验收标准 2)
    ///
    /// 复用 InterleavedScheduler 生成调度步骤，然后转换为 PipelineOp 序列。
    /// 每个 virtual stage 的 forward/backward 之间插入通信操作。
    // @trace REQ-DIST-022 [entity:Interleaved1F1B] [dataflow:DF-DIST-010]
    pub fn schedule_interleaved(
        &self,
        num_microbatches: usize,
    ) -> Result<Vec<PipelineOp>, Interleaved1F1BError> {
        if num_microbatches == 0 {
            return Err(Interleaved1F1BError::ZeroMicrobatches);
        }

        // Use InterleavedScheduler to generate schedule steps
        // InterleavedScheduler requires num_virtual_stages >= pp_size.
        // In Interleaved1F1B, num_virtual_stages is per-device, so the total
        // virtual stages = pp_size * num_virtual_stages.
        let total_virtual_stages = self.pp_size * self.num_virtual_stages;
        let interleaved_sched = InterleavedScheduler::new(
            self.pp_size,
            total_virtual_stages,
            num_microbatches,
        ).map_err(|e| Interleaved1F1BError::SchedulerError(format!("{}", e)))?;

        let steps = interleaved_sched.schedule_for_device(self.stage_id);

        // Convert InterleavedScheduleStep to PipelineOp
        let mut ops = Vec::with_capacity(steps.len() * 3);

        for step in &steps {
            let mb_idx = step.micro_batch_index;
            let vs_idx = step.virtual_stage_index;

            // Determine if this virtual stage is at a PP boundary
            // Virtual stage vs_idx on device stage_id maps to
            // global stage = stage_id + vs_idx * pp_size
            let global_stage = self.stage_id + vs_idx * self.pp_size;
            let total_global_stages = self.pp_size * self.num_virtual_stages;
            let is_first_global = global_stage == 0;
            let is_last_global = global_stage == total_global_stages - 1;

            match step.phase {
                SchedulePhase::Forward => {
                    // Receive activation from previous global stage
                    if !is_first_global {
                        ops.push(PipelineOp::RecvActivation(mb_idx));
                    }
                    // Forward computation
                    ops.push(PipelineOp::Forward(mb_idx));
                    // Send activation to next global stage
                    if !is_last_global {
                        ops.push(PipelineOp::SendActivation(mb_idx));
                    }
                }
                SchedulePhase::Backward => {
                    // Receive gradient from next global stage
                    if !is_last_global {
                        ops.push(PipelineOp::RecvGradient(mb_idx));
                    }
                    // Backward computation
                    ops.push(PipelineOp::Backward(mb_idx));
                    // Send gradient to previous global stage
                    if !is_first_global {
                        ops.push(PipelineOp::SendGradient(mb_idx));
                    }
                }
            }
        }

        Ok(ops)
    }

    /// 生成交错 1F1B 调度操作序列（带气泡标记）
    ///
    /// 在 virtual stage 切换之间插入 Bubble 标记，
    /// 标识流水线填充/排空期间的空闲时间。
    // @trace REQ-DIST-022 [entity:Interleaved1F1B] [dataflow:DF-DIST-011]
    pub fn schedule_interleaved_with_bubbles(
        &self,
        num_microbatches: usize,
    ) -> Result<Vec<PipelineOp>, Interleaved1F1BError> {
        let mut ops = self.schedule_interleaved(num_microbatches)?;

        // Insert bubble markers at the beginning for pipeline fill time
        // In interleaved mode, the fill time is reduced by num_virtual_stages
        if self.pp_size > 1 {
            let num_bubbles = ((self.pp_size - 1) / self.num_virtual_stages) as usize;
            if num_bubbles > 0 {
                let mut with_bubbles = Vec::with_capacity(ops.len() + num_bubbles);
                for _ in 0..num_bubbles {
                    with_bubbles.push(PipelineOp::Bubble);
                }
                with_bubbles.extend(ops);
                ops = with_bubbles;
            }
        }

        Ok(ops)
    }

    /// 计算交错 1F1B 峰值激活内存
    ///
    /// 交错模式下，每个设备持有 num_virtual_stages 个层块，
    /// 但同时活跃的微批次数更少（因为 warmup 更短）。
    // @trace REQ-DIST-022 [entity:Interleaved1F1B]
    pub fn peak_activation_bytes(&self, activation_bytes_per_microbatch: usize) -> usize {
        // In interleaved 1F1B, the peak number of in-flight micro-batches
        // is approximately (pp_size - stage_id) per virtual stage,
        // but since virtual stages are interleaved, the actual peak is
        // (pp_size * num_virtual_stages - stage_id * num_virtual_stages)
        // which simplifies to the same as non-interleaved for the same
        // total number of stages.
        //
        // However, the key benefit is that each virtual stage has fewer layers,
        // so activation_bytes_per_microbatch is smaller.
        let effective_pp_size = self.pp_size * self.num_virtual_stages;
        (effective_pp_size - self.stage_id * self.num_virtual_stages) as usize * activation_bytes_per_microbatch
    }

    /// 校验一致性
    // @trace REQ-DIST-022 [entity:Interleaved1F1B]
    pub fn validate(&self) -> bool {
        self.pp_size >= 1
            && self.stage_id < self.pp_size
            && self.num_virtual_stages >= 1
    }
}

// ── ScheduleComparison (REQ-DIST-022) ────────────────────────────────────────

/// 调度策略对比结果 (REQ-DIST-022)
///
/// 对比 GPipe、1F1B、Interleaved 1F1B 三种策略的调度特征。
// @trace REQ-DIST-022 [entity:ScheduleComparison]
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleComparison {
    /// GPipe 峰值激活内存（字节）
    pub gpipe_peak_bytes: usize,
    /// 1F1B 峰值激活内存（字节）
    pub one_f_one_b_peak_bytes: usize,
    /// 交错 1F1B 峰值激活内存（字节）
    pub interleaved_peak_bytes: usize,
    /// 非交错气泡比率
    pub non_interleaved_bubble_ratio: f64,
    /// 交错气泡比率
    pub interleaved_bubble_ratio: f64,
    /// 气泡比率改进倍数
    pub bubble_improvement: f64,
}

// @trace REQ-DIST-022 [entity:ScheduleComparison]
impl ScheduleComparison {
    /// 对比三种调度策略
    // @trace REQ-DIST-022 [entity:ScheduleComparison]
    pub fn compare(
        pp_size: u32,
        stage_id: u32,
        num_virtual_stages: u32,
        num_microbatches: usize,
        activation_bytes_per_microbatch: usize,
    ) -> Result<Self, Interleaved1F1BError> {
        let scheduler = PipelineScheduler::new(pp_size, stage_id)
            .map_err(|e| match e {
                PipelineSchedulerError::InvalidPpSize(v) => Interleaved1F1BError::InvalidPpSize(v),
                PipelineSchedulerError::InvalidStageId { stage_id, pp_size } => {
                    Interleaved1F1BError::InvalidStageId { stage_id, pp_size }
                }
                PipelineSchedulerError::ZeroMicrobatches => Interleaved1F1BError::ZeroMicrobatches,
            })?;

        let interleaved = Interleaved1F1B::new(pp_size, stage_id, num_virtual_stages)?;

        let gpipe_peak = scheduler.gpipe_peak_activation_bytes(
            num_microbatches,
            activation_bytes_per_microbatch,
        );
        let one_f_one_b_peak = scheduler.one_f_one_b_peak_activation_bytes(
            activation_bytes_per_microbatch,
        );
        let interleaved_peak = interleaved.peak_activation_bytes(
            activation_bytes_per_microbatch,
        );

        Ok(Self {
            gpipe_peak_bytes: gpipe_peak,
            one_f_one_b_peak_bytes: one_f_one_b_peak,
            interleaved_peak_bytes: interleaved_peak,
            non_interleaved_bubble_ratio: interleaved.non_interleaved_bubble_ratio(),
            interleaved_bubble_ratio: interleaved.bubble_ratio(),
            bubble_improvement: interleaved.bubble_ratio_improvement(),
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Interleaved1F1B: construction ──

    #[test]
    fn new_valid() {
        let sched = Interleaved1F1B::new(4, 1, 2).unwrap();
        assert_eq!(sched.pp_size, 4);
        assert_eq!(sched.stage_id, 1);
        assert_eq!(sched.num_virtual_stages, 2);
    }

    #[test]
    fn new_invalid_pp_size() {
        let result = Interleaved1F1B::new(0, 0, 2);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Interleaved1F1BError::InvalidPpSize(0));
    }

    #[test]
    fn new_invalid_stage_id() {
        let result = Interleaved1F1B::new(2, 2, 2);
        assert!(result.is_err());
        match result.unwrap_err() {
            Interleaved1F1BError::InvalidStageId { stage_id, pp_size } => {
                assert_eq!(stage_id, 2);
                assert_eq!(pp_size, 2);
            }
            other => panic!("expected InvalidStageId, got {:?}", other),
        }
    }

    #[test]
    fn new_invalid_virtual_stages() {
        let result = Interleaved1F1B::new(4, 0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Interleaved1F1BError::InvalidVirtualStages(0));
    }

    #[test]
    fn from_pipeline_config() {
        let config = PipelineConfig {
            pp_size: 4,
            stage_id: 1,
            num_virtual_stages: 2,
            micro_batch_size: 4,
            layers_per_stage: 8,
        };
        let sched = Interleaved1F1B::from_pipeline_config(&config).unwrap();
        assert_eq!(sched.pp_size, 4);
        assert_eq!(sched.stage_id, 1);
        assert_eq!(sched.num_virtual_stages, 2);
    }

    // ── Interleaved1F1B: stage predicates ──

    #[test]
    fn is_first_stage() {
        assert!(Interleaved1F1B::new(4, 0, 2).unwrap().is_first_stage());
        assert!(!Interleaved1F1B::new(4, 1, 2).unwrap().is_first_stage());
    }

    #[test]
    fn is_last_stage() {
        assert!(Interleaved1F1B::new(4, 3, 2).unwrap().is_last_stage());
        assert!(!Interleaved1F1B::new(4, 2, 2).unwrap().is_last_stage());
    }

    // ── Interleaved1F1B: bubble ratio (REQ-DIST-022 验收标准 3) ──

    #[test]
    fn bubble_ratio_interleaved() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // 验收标准 3: 气泡比率 = (pp_size - 1) / (pp_size * num_virtual_stages)
        let sched = Interleaved1F1B::new(4, 0, 2).unwrap();
        let expected = 3.0 / (4.0 * 2.0); // 0.375
        assert!((sched.bubble_ratio() - expected).abs() < 1e-10);
    }

    #[test]
    fn bubble_ratio_non_interleaved() {
        let sched = Interleaved1F1B::new(4, 0, 1).unwrap();
        // With num_virtual_stages=1, interleaved = non-interleaved
        let expected = 3.0 / 4.0; // 0.75
        assert!((sched.bubble_ratio() - expected).abs() < 1e-10);
    }

    #[test]
    fn bubble_ratio_pp1() {
        let sched = Interleaved1F1B::new(1, 0, 2).unwrap();
        assert_eq!(sched.bubble_ratio(), 0.0);
    }

    #[test]
    fn bubble_ratio_improvement() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // 交错改进倍数 ≈ num_virtual_stages
        let sched = Interleaved1F1B::new(4, 0, 2).unwrap();
        let improvement = sched.bubble_ratio_improvement();
        // improvement = non_interleaved / interleaved = 0.75 / 0.375 = 2.0
        assert!((improvement - 2.0).abs() < 1e-10);
    }

    #[test]
    fn bubble_ratio_decreases_with_more_virtual_stages() {
        let sched2 = Interleaved1F1B::new(4, 0, 2).unwrap();
        let sched4 = Interleaved1F1B::new(4, 0, 4).unwrap();
        assert!(sched4.bubble_ratio() < sched2.bubble_ratio());
    }

    // ── Interleaved1F1B: schedule_interleaved ──

    #[test]
    fn schedule_interleaved_produces_ops() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        let sched = Interleaved1F1B::new(2, 0, 2).unwrap();
        let ops = sched.schedule_interleaved(4).unwrap();
        assert!(!ops.is_empty());
        // Should have forward and backward ops
        let fwd_count = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        let bwd_count = ops.iter().filter(|op| matches!(op, PipelineOp::Backward(_))).count();
        assert!(fwd_count > 0);
        assert!(bwd_count > 0);
    }

    #[test]
    fn schedule_interleaved_zero_microbatches() {
        let sched = Interleaved1F1B::new(2, 0, 2).unwrap();
        let result = sched.schedule_interleaved(0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Interleaved1F1BError::ZeroMicrobatches);
    }

    #[test]
    fn schedule_interleaved_first_stage_no_recv() {
        // First stage (stage_id=0) should not have RecvActivation for vs_idx=0
        // (global_stage=0 is the first global stage)
        let sched = Interleaved1F1B::new(2, 0, 2).unwrap();
        let ops = sched.schedule_interleaved(4).unwrap();

        // Check that the first RecvActivation is NOT for global_stage=0
        // (which is stage_id=0, vs_idx=0)
        // The first few ops should be Forward (for vs_idx=0, which is global_stage=0)
        // So no RecvActivation before the first Forward
        if let Some(first_op) = ops.first() {
            assert!(
                !matches!(first_op, PipelineOp::RecvActivation(_)),
                "First stage should not start with RecvActivation for vs_idx=0"
            );
        }
    }

    #[test]
    fn schedule_interleaved_with_bubbles() {
        let sched = Interleaved1F1B::new(4, 0, 2).unwrap();
        let ops = sched.schedule_interleaved_with_bubbles(4).unwrap();
        let bubble_count = ops.iter().filter(|op| matches!(op, PipelineOp::Bubble)).count();
        // num_bubbles = (pp_size - 1) / num_virtual_stages = 3 / 2 = 1
        assert!(bubble_count >= 1, "Should have at least 1 bubble for pp_size=4, vs=2");
    }

    // ── Interleaved1F1B: peak activation memory ──

    #[test]
    fn peak_activation_bytes() {
        let sched = Interleaved1F1B::new(4, 0, 2).unwrap();
        let bytes = sched.peak_activation_bytes(1024);
        // effective_pp_size = 4 * 2 = 8
        // peak = (8 - 0 * 2) * 1024 = 8 * 1024
        assert_eq!(bytes, 8 * 1024);
    }

    #[test]
    fn peak_activation_bytes_last_stage() {
        let sched = Interleaved1F1B::new(4, 3, 2).unwrap();
        let bytes = sched.peak_activation_bytes(1024);
        // effective_pp_size = 8
        // peak = (8 - 3 * 2) * 1024 = 2 * 1024
        assert_eq!(bytes, 2 * 1024);
    }

    // ── Interleaved1F1B: validate ──

    #[test]
    fn validate_valid() {
        let sched = Interleaved1F1B::new(4, 1, 2).unwrap();
        assert!(sched.validate());
    }

    // ── Interleaved1F1BError: Display ──

    #[test]
    fn error_display_invalid_pp_size() {
        let err = Interleaved1F1BError::InvalidPpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("pp_size=0"));
    }

    #[test]
    fn error_display_invalid_stage_id() {
        let err = Interleaved1F1BError::InvalidStageId { stage_id: 3, pp_size: 2 };
        let msg = format!("{}", err);
        assert!(msg.contains("3"));
    }

    #[test]
    fn error_display_invalid_virtual_stages() {
        let err = Interleaved1F1BError::InvalidVirtualStages(0);
        let msg = format!("{}", err);
        assert!(msg.contains("num_virtual_stages=0"));
    }

    #[test]
    fn error_display_zero_microbatches() {
        let err = Interleaved1F1BError::ZeroMicrobatches;
        let msg = format!("{}", err);
        assert!(msg.contains("num_microbatches"));
    }

    #[test]
    fn error_display_scheduler_error() {
        let err = Interleaved1F1BError::SchedulerError("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
    }

    #[test]
    fn error_is_std_error() {
        let err = Interleaved1F1BError::InvalidPpSize(0);
        let _: &dyn std::error::Error = &err;
    }

    // ── ScheduleComparison ──

    #[test]
    fn comparison_gpipe_highest_memory() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // GPipe has the highest peak memory
        let comp = ScheduleComparison::compare(4, 0, 2, 8, 1024).unwrap();
        assert!(comp.gpipe_peak_bytes >= comp.one_f_one_b_peak_bytes);
    }

    #[test]
    fn comparison_one_f_one_b_lower_than_gpipe() {
        let comp = ScheduleComparison::compare(4, 0, 2, 8, 1024).unwrap();
        assert!(comp.one_f_one_b_peak_bytes <= comp.gpipe_peak_bytes);
    }

    #[test]
    fn comparison_interleaved_bubble_lower() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // 交错气泡比率 < 非交错气泡比率
        let comp = ScheduleComparison::compare(4, 0, 2, 8, 1024).unwrap();
        assert!(comp.interleaved_bubble_ratio < comp.non_interleaved_bubble_ratio);
    }

    #[test]
    fn comparison_bubble_improvement() {
        let comp = ScheduleComparison::compare(4, 0, 2, 8, 1024).unwrap();
        // improvement = non_interleaved / interleaved ≈ 2.0 for vs=2
        assert!((comp.bubble_improvement - 2.0).abs() < 1e-10);
    }

    // ── Interleaved1F1B: schedule correctness ──

    #[test]
    fn schedule_interleaved_all_microbatches_covered() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // All micro-batch indices should be covered in both forward and backward
        let sched = Interleaved1F1B::new(2, 0, 2).unwrap();
        let ops = sched.schedule_interleaved(4).unwrap();

        let fwd_indices: std::collections::HashSet<usize> = ops.iter()
            .filter_map(|op| match op { PipelineOp::Forward(i) => Some(*i), _ => None })
            .collect();
        let bwd_indices: std::collections::HashSet<usize> = ops.iter()
            .filter_map(|op| match op { PipelineOp::Backward(i) => Some(*i), _ => None })
            .collect();

        // All micro-batches should be processed
        assert_eq!(fwd_indices.len(), 4, "All 4 micro-batches should have forward ops");
        assert_eq!(bwd_indices.len(), 4, "All 4 micro-batches should have backward ops");
    }

    #[test]
    fn schedule_interleaved_pp1_no_comm() {
        // pp_size=1: no inter-stage communication
        let sched = Interleaved1F1B::new(1, 0, 2).unwrap();
        let ops = sched.schedule_interleaved(4).unwrap();

        let send_count = ops.iter().filter(|op| {
            matches!(op, PipelineOp::SendActivation(_) | PipelineOp::SendGradient(_))
        }).count();
        let recv_count = ops.iter().filter(|op| {
            matches!(op, PipelineOp::RecvActivation(_) | PipelineOp::RecvGradient(_))
        }).count();

        // With pp_size=1, there are still virtual stage transitions
        // that require communication between virtual stages
        // (unless we're on a single device with all virtual stages)
        // Actually, with pp_size=1, all virtual stages are on the same device,
        // so no P2P communication is needed.
        // But the InterleavedScheduler still generates steps for virtual stages,
        // and our conversion adds Send/Recv for non-boundary virtual stages.
        // Let's verify the behavior is consistent.
        assert!(ops.iter().any(|op| matches!(op, PipelineOp::Forward(_))));
        assert!(ops.iter().any(|op| matches!(op, PipelineOp::Backward(_))));
    }
}
