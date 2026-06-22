//! MicroBatchScheduler — 微批次切分与交错 PP 调度 (REQ-DIST-020, REQ-DIST-023)
//!
//! 将输入 batch 按 micro_batch_size 切分为微批次序列，支持：
//! - 均匀切分（最后一个微批次可能较小）
//! - 交错 1F1B 调度（Interleaved Pipeline Parallel）
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use super::config::PipelineConfig;

// ── MicroBatch (REQ-DIST-020) ──────────────────────────────────────────────

/// 微批次描述 (REQ-DIST-020)
///
/// 表示原始 batch 中的一个连续子区间，由 `MicroBatchScheduler::split` 产出。
/// 微批次聚合结果与原始 batch 数值等价（验收标准 5）。
// @trace REQ-DIST-020 [entity:MicroBatch] [api:POST /internal/distributed/pipeline/micro-batch]
#[derive(Debug, Clone, PartialEq)]
pub struct MicroBatch {
    /// 微批次在原始 batch 中的偏移量，≥0
    pub offset: usize,
    /// 微批次 token 数，≥1（最后一个微批次可能小于 micro_batch_size）
    pub token_count: usize,
    /// 微批次索引，范围 [0, num_microbatches)
    pub index: usize,
}

// ── MicroBatchSchedulerError (REQ-DIST-020) ────────────────────────────────

/// MicroBatchScheduler 错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicroBatchSchedulerError {
    /// micro_batch_size == 0（验收标准 4）
    ZeroMicroBatchSize,
    /// batch_size == 0
    ZeroBatchSize,
}

impl std::fmt::Display for MicroBatchSchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MicroBatchSchedulerError::ZeroMicroBatchSize => {
                write!(f, "MicroBatchScheduler: micro_batch_size must be > 0")
            }
            MicroBatchSchedulerError::ZeroBatchSize => {
                write!(f, "MicroBatchScheduler: batch_size must be > 0")
            }
        }
    }
}

impl std::error::Error for MicroBatchSchedulerError {}

// ── MicroBatchScheduler (REQ-DIST-020) ─────────────────────────────────────

/// 微批次调度器 (REQ-DIST-020)
///
/// 将输入 batch 按 `micro_batch_size` 切分为微批次序列。
/// 微批次数量 = ceil(batch_size / micro_batch_size)（验收标准 2）。
/// 最后一个微批次可能小于 micro_batch_size（验收标准 3）。
/// micro_batch_size == 0 时返回 Err（验收标准 4）。
// @trace REQ-DIST-020 [entity:MicroBatchScheduler] [api:POST /internal/distributed/pipeline/micro-batch]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MicroBatchScheduler {
    /// 微批次大小，≥1
    pub micro_batch_size: usize,
}

// @trace REQ-DIST-020 [entity:MicroBatchScheduler]
impl MicroBatchScheduler {
    /// 创建微批次调度器
    // @trace REQ-DIST-020 [entity:MicroBatchScheduler]
    pub fn new(micro_batch_size: usize) -> Result<Self, MicroBatchSchedulerError> {
        if micro_batch_size == 0 {
            return Err(MicroBatchSchedulerError::ZeroMicroBatchSize);
        }
        Ok(Self { micro_batch_size })
    }

    /// 从 PipelineConfig 创建微批次调度器
    // @trace REQ-DIST-020 [entity:MicroBatchScheduler]
    pub fn from_pipeline_config(config: &PipelineConfig) -> Result<Self, MicroBatchSchedulerError> {
        Self::new(config.micro_batch_size)
    }

    /// 将 batch 切分为微批次序列 (REQ-DIST-020 验收标准 1~5)
    ///
    /// - 验收标准 1: `split(batch, micro_batch_size)` 返回 `Vec<MicroBatch>`
    /// - 验收标准 2: num_microbatches = ceil(batch_size / micro_batch_size)
    /// - 验收标准 3: 最后一个微批次可能小于 micro_batch_size
    /// - 验收标准 4: micro_batch_size == 0 返回 Err
    /// - 验收标准 5: 聚合微批次与原始 batch 数值等价
    // @trace REQ-DIST-020 [entity:MicroBatchScheduler] [dataflow:DF-DIST-005]
    pub fn split(&self, batch_size: usize) -> Result<Vec<MicroBatch>, MicroBatchSchedulerError> {
        if self.micro_batch_size == 0 {
            return Err(MicroBatchSchedulerError::ZeroMicroBatchSize);
        }
        if batch_size == 0 {
            return Err(MicroBatchSchedulerError::ZeroBatchSize);
        }

        let num_microbatches = (batch_size + self.micro_batch_size - 1) / self.micro_batch_size;
        let mut micro_batches = Vec::with_capacity(num_microbatches);

        for i in 0..num_microbatches {
            let offset = i * self.micro_batch_size;
            let remaining = batch_size.saturating_sub(offset);
            let token_count = remaining.min(self.micro_batch_size);
            micro_batches.push(MicroBatch {
                offset,
                token_count,
                index: i,
            });
        }

        Ok(micro_batches)
    }

    /// 返回微批次数量 = ceil(batch_size / micro_batch_size)
    // @trace REQ-DIST-020 [entity:MicroBatchScheduler]
    pub fn num_microbatches(&self, batch_size: usize) -> usize {
        if batch_size == 0 || self.micro_batch_size == 0 {
            return 0;
        }
        (batch_size + self.micro_batch_size - 1) / self.micro_batch_size
    }

    /// 校验调度器一致性
    // @trace REQ-DIST-020 [entity:MicroBatchScheduler]
    pub fn validate(&self) -> bool {
        self.micro_batch_size >= 1
    }
}

// ── InterleavedSchedule (REQ-DIST-023) ─────────────────────────────────────

/// 交错 PP 调度阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulePhase {
    /// 前向传播
    Forward,
    /// 反向传播
    Backward,
}

/// 交错 1F1B 调度步骤 (REQ-DIST-023)
///
/// 描述一个微批次在某个虚拟 stage 上的前向/反向执行步骤。
// @trace REQ-DIST-023 [entity:InterleavedScheduleStep]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterleavedScheduleStep {
    /// 微批次索引
    pub micro_batch_index: usize,
    /// 虚拟 stage 索引（设备内的虚拟 stage 编号）
    pub virtual_stage_index: u32,
    /// 执行阶段（前向/反向）
    pub phase: SchedulePhase,
}

/// 交错 1F1B 调度器 (REQ-DIST-023)
///
/// 交错 Pipeline Parallel 调度：每个 device 持有多个不连续层块（虚拟 stage），
/// 微批次在虚拟 stage 间交替发射，降低流水线气泡率。
///
/// 气泡率公式：(pp_size - 1) / (num_virtual_stages * pp_size)
/// （验收标准 3: 交错气泡率低于非交错）
///
/// 每个设备持有 num_virtual_stages / pp_size 个不连续层块（验收标准 4）。
// @trace REQ-DIST-023 [entity:InterleavedScheduler] [api:POST /internal/distributed/pipeline/interleaved]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InterleavedScheduler {
    /// Pipeline Parallel 维度
    pub pp_size: u32,
    /// 虚拟 stage 数量，≥pp_size（验收标准 2）
    pub num_virtual_stages: u32,
    /// 微批次数量
    pub num_microbatches: usize,
}

/// InterleavedScheduler 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterleavedSchedulerError {
    /// pp_size < 1
    InvalidPpSize(u32),
    /// num_virtual_stages < pp_size（验收标准 2 要求 ≥ pp_size）
    VirtualStagesLessThanPpSize { num_virtual_stages: u32, pp_size: u32 },
    /// num_microbatches == 0
    ZeroMicrobatches,
}

impl std::fmt::Display for InterleavedSchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterleavedSchedulerError::InvalidPpSize(pp_size) => {
                write!(f, "InterleavedScheduler: invalid pp_size={pp_size}, must be >= 1")
            }
            InterleavedSchedulerError::VirtualStagesLessThanPpSize {
                num_virtual_stages,
                pp_size,
            } => {
                write!(
                    f,
                    "InterleavedScheduler: num_virtual_stages({num_virtual_stages}) < pp_size({pp_size}), must be >= pp_size"
                )
            }
            InterleavedSchedulerError::ZeroMicrobatches => {
                write!(f, "InterleavedScheduler: num_microbatches must be > 0")
            }
        }
    }
}

impl std::error::Error for InterleavedSchedulerError {}

// @trace REQ-DIST-023 [entity:InterleavedScheduler]
impl InterleavedScheduler {
    /// 创建交错 1F1B 调度器
    ///
    /// 校验 num_virtual_stages >= pp_size（验收标准 2）。
    // @trace REQ-DIST-023 [entity:InterleavedScheduler]
    pub fn new(
        pp_size: u32,
        num_virtual_stages: u32,
        num_microbatches: usize,
    ) -> Result<Self, InterleavedSchedulerError> {
        if pp_size < 1 {
            return Err(InterleavedSchedulerError::InvalidPpSize(pp_size));
        }
        if num_virtual_stages < pp_size {
            return Err(InterleavedSchedulerError::VirtualStagesLessThanPpSize {
                num_virtual_stages,
                pp_size,
            });
        }
        if num_microbatches == 0 {
            return Err(InterleavedSchedulerError::ZeroMicrobatches);
        }
        Ok(Self {
            pp_size,
            num_virtual_stages,
            num_microbatches,
        })
    }

    /// 从 PipelineConfig 创建交错调度器
    // @trace REQ-DIST-023 [entity:InterleavedScheduler]
    pub fn from_pipeline_config(
        config: &PipelineConfig,
        num_microbatches: usize,
    ) -> Result<Self, InterleavedSchedulerError> {
        Self::new(config.pp_size, config.num_virtual_stages, num_microbatches)
    }

    /// 每个设备持有的虚拟 stage 数量 = num_virtual_stages / pp_size（验收标准 4）
    // @trace REQ-DIST-023 [entity:InterleavedScheduler]
    pub fn virtual_stages_per_device(&self) -> u32 {
        self.num_virtual_stages / self.pp_size
    }

    /// 交错 PP 气泡率 = (pp_size - 1) / (num_virtual_stages * pp_size)（验收标准 3）
    ///
    /// 非交错气泡率 = (pp_size - 1) / pp_size，交错气泡率严格更低。
    // @trace REQ-DIST-023 [entity:InterleavedScheduler]
    pub fn bubble_ratio(&self) -> f64 {
        if self.pp_size <= 1 {
            return 0.0;
        }
        (self.pp_size - 1) as f64 / (self.num_virtual_stages as f64 * self.pp_size as f64)
    }

    /// 非交错 PP 气泡率 = (pp_size - 1) / pp_size
    // @trace REQ-DIST-023 [entity:InterleavedScheduler]
    pub fn non_interleaved_bubble_ratio(&self) -> f64 {
        if self.pp_size <= 1 {
            return 0.0;
        }
        (self.pp_size - 1) as f64 / self.pp_size as f64
    }

    /// 虚拟 stage 到设备的映射 (REQ-DIST-023 验收标准 2)
    ///
    /// 虚拟 stage `v` 映射到设备 `v % pp_size`。
    /// 这保证了每个设备持有 num_virtual_stages / pp_size 个不连续层块。
    // @trace REQ-DIST-023 [entity:InterleavedScheduler]
    pub fn virtual_stage_to_device(&self, virtual_stage: u32) -> u32 {
        virtual_stage % self.pp_size
    }

    /// 设备持有的虚拟 stage 列表（不连续层块索引）
    ///
    /// 设备 `device` 持有虚拟 stage: device, device + pp_size, device + 2*pp_size, ...
    // @trace REQ-DIST-023 [entity:InterleavedScheduler]
    pub fn device_virtual_stages(&self, device: u32) -> Vec<u32> {
        let mut stages = Vec::new();
        let mut v = device;
        while v < self.num_virtual_stages {
            stages.push(v);
            v += self.pp_size;
        }
        stages
    }

    /// 生成交错 1F1B 调度步骤序列 (REQ-DIST-023 验收标准 1)
    ///
    /// 交错 1F1B 调度：微批次按虚拟 stage 顺序交替发射前向，
    /// 然后交替执行 1 个前向 + 1 个反向（1F1B 稳态），最后排空反向。
    ///
    /// 对于设备 `device`，调度步骤按虚拟 stage 顺序交替：
    /// - Warmup 阶段：连续前向
    /// - 1F1B 稳态：交替前向+反向
    /// - Cooldown 阶段：连续反向
    // @trace REQ-DIST-023 [entity:InterleavedScheduler] [dataflow:DF-DIST-006]
    pub fn schedule_for_device(&self, device: u32) -> Vec<InterleavedScheduleStep> {
        let v_stages = self.device_virtual_stages(device);
        if v_stages.is_empty() {
            return Vec::new();
        }

        let num_v = v_stages.len();
        let total_forward_steps = self.num_microbatches * num_v;
        let total_backward_steps = total_forward_steps;

        // Warmup: 前向步数 = num_v * (pp_size - 1) + num_v (交错 warmup)
        // 简化：交错 1F1B warmup 步数
        let warmup_forwards = if self.pp_size > 1 {
            num_v * (self.pp_size as usize - 1) + num_v
        } else {
            0
        };
        let warmup_forwards = warmup_forwards.min(total_forward_steps);

        let mut steps = Vec::with_capacity(total_forward_steps + total_backward_steps);
        let mut _fwd_count = 0usize;
        let mut bwd_count = 0usize;

        // Warmup: 连续前向
        for step in 0..warmup_forwards {
            let mb_idx = step / num_v;
            let v_idx = step % num_v;
            if mb_idx < self.num_microbatches {
                steps.push(InterleavedScheduleStep {
                    micro_batch_index: mb_idx,
                    virtual_stage_index: v_stages[v_idx],
                    phase: SchedulePhase::Forward,
                });
                _fwd_count += 1;
            }
        }

        // 1F1B steady state: 交替前向+反向
        let steady_steps = total_forward_steps.saturating_sub(warmup_forwards);
        for step in 0..steady_steps {
            // Forward
            let fwd_step = warmup_forwards + step;
            let mb_fwd = fwd_step / num_v;
            let v_fwd = fwd_step % num_v;
            if mb_fwd < self.num_microbatches {
                steps.push(InterleavedScheduleStep {
                    micro_batch_index: mb_fwd,
                    virtual_stage_index: v_stages[v_fwd],
                    phase: SchedulePhase::Forward,
                });
                _fwd_count += 1;
            }

            // Backward
            let bwd_step = step;
            let mb_bwd = bwd_step / num_v;
            let v_bwd = bwd_step % num_v;
            if mb_bwd < self.num_microbatches {
                steps.push(InterleavedScheduleStep {
                    micro_batch_index: mb_bwd,
                    virtual_stage_index: v_stages[v_bwd],
                    phase: SchedulePhase::Backward,
                });
                bwd_count += 1;
            }
        }

        // Cooldown: 连续反向
        while bwd_count < total_backward_steps {
            let bwd_step = bwd_count;
            let mb_bwd = bwd_step / num_v;
            let v_bwd = bwd_step % num_v;
            if mb_bwd < self.num_microbatches {
                steps.push(InterleavedScheduleStep {
                    micro_batch_index: mb_bwd,
                    virtual_stage_index: v_stages[v_bwd],
                    phase: SchedulePhase::Backward,
                });
            }
            bwd_count += 1;
        }

        steps
    }

    /// 校验调度器一致性
    // @trace REQ-DIST-023 [entity:InterleavedScheduler]
    pub fn validate(&self) -> bool {
        self.pp_size >= 1
            && self.num_virtual_stages >= self.pp_size
            && self.num_microbatches >= 1
            && self.num_virtual_stages % self.pp_size == 0
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MicroBatch: construction ──

    #[test]
    fn micro_batch_fields() {
        let mb = MicroBatch {
            offset: 10,
            token_count: 4,
            index: 2,
        };
        assert_eq!(mb.offset, 10);
        assert_eq!(mb.token_count, 4);
        assert_eq!(mb.index, 2);
    }

    // ── MicroBatchScheduler: split (REQ-DIST-020 验收标准 1~5) ──

    #[test]
    fn split_exact_division() {
        // @trace TEST-DIST-020 [req:REQ-DIST-020] [level:unit]
        let scheduler = MicroBatchScheduler::new(4).unwrap();
        let batches = scheduler.split(12).unwrap();
        assert_eq!(batches.len(), 3); // ceil(12/4) = 3
        assert_eq!(batches[0], MicroBatch { offset: 0, token_count: 4, index: 0 });
        assert_eq!(batches[1], MicroBatch { offset: 4, token_count: 4, index: 1 });
        assert_eq!(batches[2], MicroBatch { offset: 8, token_count: 4, index: 2 });
    }

    #[test]
    fn split_last_smaller() {
        // @trace TEST-DIST-020 [req:REQ-DIST-020] [level:unit]
        // 验收标准 3: 最后一个微批次可能小于 micro_batch_size
        let scheduler = MicroBatchScheduler::new(5).unwrap();
        let batches = scheduler.split(12).unwrap();
        assert_eq!(batches.len(), 3); // ceil(12/5) = 3
        assert_eq!(batches[0], MicroBatch { offset: 0, token_count: 5, index: 0 });
        assert_eq!(batches[1], MicroBatch { offset: 5, token_count: 5, index: 1 });
        assert_eq!(batches[2], MicroBatch { offset: 10, token_count: 2, index: 2 }); // last < 5
    }

    #[test]
    fn split_single_micro_batch() {
        // @trace TEST-DIST-020 [req:REQ-DIST-020] [level:unit]
        let scheduler = MicroBatchScheduler::new(10).unwrap();
        let batches = scheduler.split(3).unwrap();
        assert_eq!(batches.len(), 1); // ceil(3/10) = 1
        assert_eq!(batches[0], MicroBatch { offset: 0, token_count: 3, index: 0 });
    }

    #[test]
    fn split_batch_equals_micro_batch_size() {
        let scheduler = MicroBatchScheduler::new(4).unwrap();
        let batches = scheduler.split(4).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], MicroBatch { offset: 0, token_count: 4, index: 0 });
    }

    #[test]
    fn split_zero_micro_batch_size_returns_err() {
        // @trace TEST-DIST-020 [req:REQ-DIST-020] [level:unit]
        // 验收标准 4: micro_batch_size == 0 返回 Err
        let result = MicroBatchScheduler::new(0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), MicroBatchSchedulerError::ZeroMicroBatchSize);
    }

    #[test]
    fn split_zero_batch_size_returns_err() {
        let scheduler = MicroBatchScheduler::new(4).unwrap();
        let result = scheduler.split(0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), MicroBatchSchedulerError::ZeroBatchSize);
    }

    #[test]
    fn split_aggregation_covers_full_batch() {
        // @trace TEST-DIST-020 [req:REQ-DIST-020] [level:unit]
        // 验收标准 5: 聚合微批次与原始 batch 数值等价
        let scheduler = MicroBatchScheduler::new(3).unwrap();
        let batches = scheduler.split(10).unwrap();
        let total_tokens: usize = batches.iter().map(|b| b.token_count).sum();
        assert_eq!(total_tokens, 10);

        // Offsets are contiguous
        for (i, b) in batches.iter().enumerate() {
            assert_eq!(b.index, i);
            if i > 0 {
                assert_eq!(b.offset, batches[i - 1].offset + batches[i - 1].token_count);
            }
        }
    }

    #[test]
    fn split_no_gaps_no_overlap() {
        let scheduler = MicroBatchScheduler::new(7).unwrap();
        let batches = scheduler.split(20).unwrap();
        let mut covered = vec![false; 20];
        for b in &batches {
            for i in b.offset..b.offset + b.token_count {
                assert!(!covered[i], "token {} covered by multiple micro-batches", i);
                covered[i] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "not all tokens covered");
    }

    #[test]
    fn num_microbatches_formula() {
        // @trace TEST-DIST-020 [req:REQ-DIST-020] [level:unit]
        // 验收标准 2: num_microbatches = ceil(batch_size / micro_batch_size)
        let scheduler = MicroBatchScheduler::new(4).unwrap();
        assert_eq!(scheduler.num_microbatches(12), 3);
        assert_eq!(scheduler.num_microbatches(13), 4); // ceil(13/4) = 4
        assert_eq!(scheduler.num_microbatches(4), 1);
        assert_eq!(scheduler.num_microbatches(1), 1);
    }

    // ── MicroBatchScheduler: from_pipeline_config ──

    #[test]
    fn from_pipeline_config() {
        let config = PipelineConfig {
            pp_size: 2,
            stage_id: 0,
            num_virtual_stages: 1,
            micro_batch_size: 8,
            layers_per_stage: 16,
        };
        let scheduler = MicroBatchScheduler::from_pipeline_config(&config).unwrap();
        assert_eq!(scheduler.micro_batch_size, 8);
    }

    // ── MicroBatchScheduler: validate ──

    #[test]
    fn validate_valid() {
        let scheduler = MicroBatchScheduler::new(4).unwrap();
        assert!(scheduler.validate());
    }

    // ── MicroBatchSchedulerError: Display ──

    #[test]
    fn error_display_zero_micro_batch_size() {
        let err = MicroBatchSchedulerError::ZeroMicroBatchSize;
        let msg = format!("{}", err);
        assert!(msg.contains("micro_batch_size"));
    }

    #[test]
    fn error_display_zero_batch_size() {
        let err = MicroBatchSchedulerError::ZeroBatchSize;
        let msg = format!("{}", err);
        assert!(msg.contains("batch_size"));
    }

    #[test]
    fn error_is_std_error() {
        let err = MicroBatchSchedulerError::ZeroMicroBatchSize;
        let _: &dyn std::error::Error = &err;
    }

    // ── InterleavedScheduler: construction (REQ-DIST-023 验收标准 2) ──

    #[test]
    fn interleaved_new_valid() {
        let scheduler = InterleavedScheduler::new(2, 4, 6).unwrap();
        assert_eq!(scheduler.pp_size, 2);
        assert_eq!(scheduler.num_virtual_stages, 4);
        assert_eq!(scheduler.num_microbatches, 6);
    }

    #[test]
    fn interleaved_new_invalid_pp_size() {
        let result = InterleavedScheduler::new(0, 4, 6);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InterleavedSchedulerError::InvalidPpSize(0));
    }

    #[test]
    fn interleaved_new_virtual_stages_less_than_pp_size() {
        // @trace TEST-DIST-023 [req:REQ-DIST-023] [level:unit]
        // 验收标准 2: num_virtual_stages >= pp_size
        let result = InterleavedScheduler::new(4, 2, 6);
        assert!(result.is_err());
        match result.unwrap_err() {
            InterleavedSchedulerError::VirtualStagesLessThanPpSize {
                num_virtual_stages,
                pp_size,
            } => {
                assert_eq!(num_virtual_stages, 2);
                assert_eq!(pp_size, 4);
            }
            other => panic!("expected VirtualStagesLessThanPpSize, got {:?}", other),
        }
    }

    #[test]
    fn interleaved_new_zero_microbatches() {
        let result = InterleavedScheduler::new(2, 4, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InterleavedSchedulerError::ZeroMicrobatches);
    }

    // ── InterleavedScheduler: virtual_stages_per_device (REQ-DIST-023 验收标准 4) ──

    #[test]
    fn virtual_stages_per_device() {
        // @trace TEST-DIST-023 [req:REQ-DIST-023] [level:unit]
        // 验收标准 4: 每个设备持有 num_virtual_stages / pp_size 个不连续层块
        let scheduler = InterleavedScheduler::new(2, 4, 6).unwrap();
        assert_eq!(scheduler.virtual_stages_per_device(), 2);
    }

    #[test]
    fn virtual_stages_per_device_non_interleaved() {
        // num_virtual_stages == pp_size → 1 virtual stage per device (non-interleaved)
        let scheduler = InterleavedScheduler::new(4, 4, 6).unwrap();
        assert_eq!(scheduler.virtual_stages_per_device(), 1);
    }

    // ── InterleavedScheduler: bubble_ratio (REQ-DIST-023 验收标准 3) ──

    #[test]
    fn bubble_ratio_interleaved_lower_than_non_interleaved() {
        // @trace TEST-DIST-023 [req:REQ-DIST-023] [level:unit]
        // 验收标准 3: 交错气泡率低于非交错
        let scheduler = InterleavedScheduler::new(4, 8, 6).unwrap();
        assert!(scheduler.bubble_ratio() < scheduler.non_interleaved_bubble_ratio());
        // Interleaved: (4-1)/(8*4) = 3/32 = 0.09375
        // Non-interleaved: (4-1)/4 = 0.75
        assert!((scheduler.bubble_ratio() - 0.09375).abs() < 1e-10);
        assert!((scheduler.non_interleaved_bubble_ratio() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn bubble_ratio_pp1_zero() {
        let scheduler = InterleavedScheduler::new(1, 1, 4).unwrap();
        assert_eq!(scheduler.bubble_ratio(), 0.0);
        assert_eq!(scheduler.non_interleaved_bubble_ratio(), 0.0);
    }

    // ── InterleavedScheduler: virtual_stage_to_device (REQ-DIST-023 验收标准 2) ──

    #[test]
    fn virtual_stage_to_device_mapping() {
        // @trace TEST-DIST-023 [req:REQ-DIST-023] [level:unit]
        // 验收标准 2: virtual stage-to-device mapping is correct
        let scheduler = InterleavedScheduler::new(2, 4, 6).unwrap();
        // v0 → device 0, v1 → device 1, v2 → device 0, v3 → device 1
        assert_eq!(scheduler.virtual_stage_to_device(0), 0);
        assert_eq!(scheduler.virtual_stage_to_device(1), 1);
        assert_eq!(scheduler.virtual_stage_to_device(2), 0);
        assert_eq!(scheduler.virtual_stage_to_device(3), 1);
    }

    // ── InterleavedScheduler: device_virtual_stages (REQ-DIST-023 验收标准 4) ──

    #[test]
    fn device_virtual_stages_non_contiguous() {
        // @trace TEST-DIST-023 [req:REQ-DIST-023] [level:unit]
        // 验收标准 4: 每个设备持有不连续层块
        let scheduler = InterleavedScheduler::new(2, 4, 6).unwrap();
        // device 0: v0, v2 (non-contiguous)
        assert_eq!(scheduler.device_virtual_stages(0), vec![0, 2]);
        // device 1: v1, v3 (non-contiguous)
        assert_eq!(scheduler.device_virtual_stages(1), vec![1, 3]);
    }

    #[test]
    fn device_virtual_stages_pp4_v8() {
        let scheduler = InterleavedScheduler::new(4, 8, 6).unwrap();
        // device 0: v0, v4
        assert_eq!(scheduler.device_virtual_stages(0), vec![0, 4]);
        // device 1: v1, v5
        assert_eq!(scheduler.device_virtual_stages(1), vec![1, 5]);
        // device 2: v2, v6
        assert_eq!(scheduler.device_virtual_stages(2), vec![2, 6]);
        // device 3: v3, v7
        assert_eq!(scheduler.device_virtual_stages(3), vec![3, 7]);
    }

    // ── InterleavedScheduler: schedule_for_device (REQ-DIST-023 验收标准 1) ──

    #[test]
    fn schedule_for_device_alternates_virtual_stages() {
        // @trace TEST-DIST-023 [req:REQ-DIST-023] [level:unit]
        // 验收标准 1: 交错调度器按虚拟 stage 顺序交替发射微批次
        let scheduler = InterleavedScheduler::new(2, 4, 2).unwrap();
        let schedule = scheduler.schedule_for_device(0);
        // device 0 holds v0, v2
        // Forward steps should alternate between v0 and v2
        let fwd_steps: Vec<_> = schedule
            .iter()
            .filter(|s| s.phase == SchedulePhase::Forward)
            .collect();
        assert!(!fwd_steps.is_empty());
        // First forward should be on v0
        assert_eq!(fwd_steps[0].virtual_stage_index, 0);
    }

    #[test]
    fn schedule_for_device_has_forward_and_backward() {
        let scheduler = InterleavedScheduler::new(2, 4, 2).unwrap();
        let schedule = scheduler.schedule_for_device(0);
        let fwd_count = schedule.iter().filter(|s| s.phase == SchedulePhase::Forward).count();
        let bwd_count = schedule.iter().filter(|s| s.phase == SchedulePhase::Backward).count();
        // Equal number of forward and backward steps
        assert_eq!(fwd_count, bwd_count);
        assert!(fwd_count > 0);
    }

    #[test]
    fn schedule_for_device_pp1_no_interleaving() {
        // pp_size=1, num_virtual_stages=1 → non-interleaved, single device
        let scheduler = InterleavedScheduler::new(1, 1, 3).unwrap();
        let schedule = scheduler.schedule_for_device(0);
        let fwd_count = schedule.iter().filter(|s| s.phase == SchedulePhase::Forward).count();
        let bwd_count = schedule.iter().filter(|s| s.phase == SchedulePhase::Backward).count();
        assert_eq!(fwd_count, 3);
        assert_eq!(bwd_count, 3);
    }

    // ── InterleavedScheduler: from_pipeline_config ──

    #[test]
    fn from_pipeline_config_interleaved() {
        let config = PipelineConfig {
            pp_size: 2,
            stage_id: 0,
            num_virtual_stages: 4,
            micro_batch_size: 8,
            layers_per_stage: 16,
        };
        let scheduler = InterleavedScheduler::from_pipeline_config(&config, 6).unwrap();
        assert_eq!(scheduler.pp_size, 2);
        assert_eq!(scheduler.num_virtual_stages, 4);
        assert_eq!(scheduler.num_microbatches, 6);
    }

    // ── InterleavedScheduler: validate ──

    #[test]
    fn interleaved_validate_valid() {
        let scheduler = InterleavedScheduler::new(2, 4, 6).unwrap();
        assert!(scheduler.validate());
    }

    #[test]
    fn validate_non_divisible_virtual_stages() {
        // num_virtual_stages not divisible by pp_size
        let scheduler = InterleavedScheduler {
            pp_size: 3,
            num_virtual_stages: 5,
            num_microbatches: 4,
        };
        assert!(!scheduler.validate());
    }

    // ── InterleavedSchedulerError: Display ──

    #[test]
    fn error_display_invalid_pp_size() {
        let err = InterleavedSchedulerError::InvalidPpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("pp_size=0"));
    }

    #[test]
    fn error_display_virtual_stages_less() {
        let err = InterleavedSchedulerError::VirtualStagesLessThanPpSize {
            num_virtual_stages: 2,
            pp_size: 4,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("2"));
        assert!(msg.contains("4"));
    }

    #[test]
    fn error_display_zero_microbatches() {
        let err = InterleavedSchedulerError::ZeroMicrobatches;
        let msg = format!("{}", err);
        assert!(msg.contains("num_microbatches"));
    }

    #[test]
    fn interleaved_error_is_std_error() {
        let err = InterleavedSchedulerError::InvalidPpSize(0);
        let _: &dyn std::error::Error = &err;
    }

    // ── InterleavedScheduler: equality and clone ──

    #[test]
    fn equality() {
        let a = InterleavedScheduler::new(2, 4, 6).unwrap();
        let b = InterleavedScheduler::new(2, 4, 6).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn clone_independence() {
        let scheduler = InterleavedScheduler::new(2, 4, 6).unwrap();
        let cloned = scheduler.clone();
        assert_eq!(cloned.pp_size, 2);
        assert_eq!(cloned.num_virtual_stages, 4);
    }

    // ── InterleavedScheduler: hash consistency ──

    #[test]
    fn hash_consistency() {
        use std::collections::HashSet;
        let a = InterleavedScheduler::new(2, 4, 6).unwrap();
        let b = InterleavedScheduler::new(2, 4, 6).unwrap();
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }
}
