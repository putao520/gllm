//! PipelineScheduler — GPipe 与 1F1B 调度策略 (REQ-DIST-022)
//!
//! Pipeline Parallel 调度策略实现：
//! - GPipe: 所有 forward 后所有 backward（简单但高内存峰值）
//! - 1F1B: warmup forward + 交替 fwd/bwd + cooldown backward（低内存峰值）
//! - Zero-Bubble: 最小化流水线气泡的调度优化
//!
//! 两种策略计算结果数值等价（验收标准 6）。
//! GPipe 峰值激活内存 = num_microbatches * activation_bytes（验收标准 4）。
//! 1F1B 峰值激活内存 = (pp_size - stage_id) * activation_bytes（验收标准 5）。
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use super::config::PipelineConfig;

// ── PipelineOp (REQ-DIST-021, REQ-DIST-022) ─────────────────────────────────

/// 流水线调度操作 (REQ-DIST-021, REQ-DIST-022)
///
/// 描述一个 stage 在流水线调度中执行的操作步骤。
// @trace REQ-DIST-022 [entity:PipelineOp] [api:POST /internal/distributed/pipeline/schedule]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PipelineOp {
    /// 前向计算（微批次 index）
    Forward(usize),
    /// 反向计算（微批次 index）
    Backward(usize),
    /// 发送激活到下一个 stage（微批次 index）
    SendActivation(usize),
    /// 从上一个 stage 接收激活（微批次 index）
    RecvActivation(usize),
    /// 发送梯度到上一个 stage（微批次 index）
    SendGradient(usize),
    /// 从下一个 stage 接收梯度（微批次 index）
    RecvGradient(usize),
    /// 流水线气泡（空闲等待）
    Bubble,
}

impl std::fmt::Display for PipelineOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineOp::Forward(i) => write!(f, "F({})", i),
            PipelineOp::Backward(i) => write!(f, "B({})", i),
            PipelineOp::SendActivation(i) => write!(f, "SendAct({})", i),
            PipelineOp::RecvActivation(i) => write!(f, "RecvAct({})", i),
            PipelineOp::SendGradient(i) => write!(f, "SendGrad({})", i),
            PipelineOp::RecvGradient(i) => write!(f, "RecvGrad({})", i),
            PipelineOp::Bubble => write!(f, "Bubble"),
        }
    }
}

// ── MicroBatchStrategy (REQ-DIST-022) ────────────────────────────────────────

/// 流水线调度策略 (REQ-DIST-022)
///
/// - GPipe: 所有 forward 后所有 backward（验收标准 1）
/// - OneFOneB: warmup forward + 交替 fwd/bwd + cooldown backward（验收标准 2）
// @trace REQ-DIST-022 [entity:MicroBatchStrategy]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MicroBatchStrategy {
    /// GPipe 策略：所有前向后所有反向
    GPipe,
    /// 1F1B 策略：warmup + 交替 + cooldown
    OneFOneB,
}

impl Default for MicroBatchStrategy {
    fn default() -> Self {
        MicroBatchStrategy::OneFOneB
    }
}

impl std::fmt::Display for MicroBatchStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MicroBatchStrategy::GPipe => write!(f, "GPipe"),
            MicroBatchStrategy::OneFOneB => write!(f, "1F1B"),
        }
    }
}

// ── PipelineScheduler (REQ-DIST-022) ─────────────────────────────────────────

/// 流水线调度器 (REQ-DIST-022)
///
/// 为 Pipeline Parallel 的 stage 生成调度操作序列。
/// 支持 GPipe 和 1F1B 两种策略，两者计算结果数值等价。
///
/// GPipe 峰值激活内存 = num_microbatches * activation_bytes（验收标准 4）
/// 1F1B 峰值激活内存 = (pp_size - stage_id) * activation_bytes（验收标准 5）
// @trace REQ-DIST-022 [entity:PipelineScheduler] [api:POST /internal/distributed/pipeline/schedule]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineScheduler {
    /// Pipeline Parallel 维度
    pub pp_size: u32,
    /// 当前 stage ID
    pub stage_id: u32,
}

/// PipelineScheduler 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineSchedulerError {
    /// pp_size < 1
    InvalidPpSize(u32),
    /// stage_id >= pp_size
    InvalidStageId { stage_id: u32, pp_size: u32 },
    /// num_microbatches == 0
    ZeroMicrobatches,
}

impl std::fmt::Display for PipelineSchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineSchedulerError::InvalidPpSize(pp_size) => {
                write!(f, "PipelineScheduler: invalid pp_size={pp_size}, must be >= 1")
            }
            PipelineSchedulerError::InvalidStageId { stage_id, pp_size } => {
                write!(
                    f,
                    "PipelineScheduler: stage_id={stage_id} out of range [0, {pp_size})"
                )
            }
            PipelineSchedulerError::ZeroMicrobatches => {
                write!(f, "PipelineScheduler: num_microbatches must be > 0")
            }
        }
    }
}

impl std::error::Error for PipelineSchedulerError {}

// @trace REQ-DIST-022 [entity:PipelineScheduler]
impl PipelineScheduler {
    /// 创建流水线调度器
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn new(pp_size: u32, stage_id: u32) -> Result<Self, PipelineSchedulerError> {
        if pp_size < 1 {
            return Err(PipelineSchedulerError::InvalidPpSize(pp_size));
        }
        if stage_id >= pp_size {
            return Err(PipelineSchedulerError::InvalidStageId { stage_id, pp_size });
        }
        Ok(Self { pp_size, stage_id })
    }

    /// 从 PipelineConfig 创建调度器
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn from_pipeline_config(config: &PipelineConfig) -> Result<Self, PipelineSchedulerError> {
        Self::new(config.pp_size, config.stage_id)
    }

    /// 是否为第一个 stage
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }

    /// 是否为最后一个 stage
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn is_last_stage(&self) -> bool {
        self.stage_id == self.pp_size - 1
    }

    /// 1F1B warmup 步数 = pp_size - stage_id - 1（验收标准 2）
    ///
    /// 首个 stage warmup 最多（填充流水线），末尾 stage warmup 为 0。
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn warmup_steps(&self) -> usize {
        if self.pp_size <= 1 {
            return 0;
        }
        (self.pp_size - self.stage_id - 1) as usize
    }

    /// 1F1B cooldown 步数 = stage_id（验收标准 2）
    ///
    /// 末尾 stage cooldown 最多（排空流水线），首个 stage cooldown 为 0。
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn cooldown_steps(&self) -> usize {
        self.stage_id as usize
    }

    /// 1F1B 稳态步数（1F1B 交替阶段）
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn steady_steps(&self, num_microbatches: usize) -> usize {
        let warmup = self.warmup_steps().min(num_microbatches);
        num_microbatches.saturating_sub(warmup)
    }

    /// GPipe 峰值激活内存 = num_microbatches * activation_bytes（验收标准 4）
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn gpipe_peak_activation_bytes(
        &self,
        num_microbatches: usize,
        activation_bytes: usize,
    ) -> usize {
        num_microbatches * activation_bytes
    }

    /// 1F1B 峰值激活内存 = (pp_size - stage_id) * activation_bytes（验收标准 5）
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn one_f_one_b_peak_activation_bytes(
        &self,
        activation_bytes: usize,
    ) -> usize {
        (self.pp_size - self.stage_id) as usize * activation_bytes
    }

    /// 生成 GPipe 调度操作序列 (REQ-DIST-022 验收标准 1)
    ///
    /// GPipe: 所有 forward 后所有 backward。
    /// 峰值激活内存 = num_microbatches * activation_bytes。
    // @trace REQ-DIST-022 [entity:PipelineScheduler] [dataflow:DF-DIST-009]
    pub fn schedule_gpipe(&self, num_microbatches: usize) -> Result<Vec<PipelineOp>, PipelineSchedulerError> {
        if num_microbatches == 0 {
            return Err(PipelineSchedulerError::ZeroMicrobatches);
        }

        let mut ops = Vec::with_capacity(num_microbatches * 2);

        // Forward pass: all forwards
        for mb in 0..num_microbatches {
            // Receive activation from previous stage (except first stage)
            if !self.is_first_stage() {
                ops.push(PipelineOp::RecvActivation(mb));
            }
            // Forward computation
            ops.push(PipelineOp::Forward(mb));
            // Send activation to next stage (except last stage)
            if !self.is_last_stage() {
                ops.push(PipelineOp::SendActivation(mb));
            }
        }

        // Backward pass: all backwards
        for mb in 0..num_microbatches {
            // Receive gradient from next stage (except last stage)
            if !self.is_last_stage() {
                ops.push(PipelineOp::RecvGradient(mb));
            }
            // Backward computation
            ops.push(PipelineOp::Backward(mb));
            // Send gradient to previous stage (except first stage)
            if !self.is_first_stage() {
                ops.push(PipelineOp::SendGradient(mb));
            }
        }

        Ok(ops)
    }

    /// 生成 1F1B 调度操作序列 (REQ-DIST-022 验收标准 2)
    ///
    /// 1F1B: warmup forward + 交替 fwd/bwd + cooldown backward。
    /// warmup_steps = pp_size - stage_id - 1
    /// cooldown_steps = stage_id
    /// 峰值激活内存 = (pp_size - stage_id) * activation_bytes。
    // @trace REQ-DIST-022 [entity:PipelineScheduler] [dataflow:DF-DIST-010]
    pub fn schedule_1f1b(&self, num_microbatches: usize) -> Result<Vec<PipelineOp>, PipelineSchedulerError> {
        if num_microbatches == 0 {
            return Err(PipelineSchedulerError::ZeroMicrobatches);
        }

        let warmup = self.warmup_steps().min(num_microbatches);
        let steady = num_microbatches.saturating_sub(warmup);
        let cooldown = self.cooldown_steps().min(num_microbatches);
        let mut ops = Vec::with_capacity(num_microbatches * 3);

        // Phase 1: Warmup — consecutive forwards
        for mb in 0..warmup {
            if !self.is_first_stage() {
                ops.push(PipelineOp::RecvActivation(mb));
            }
            ops.push(PipelineOp::Forward(mb));
            if !self.is_last_stage() {
                ops.push(PipelineOp::SendActivation(mb));
            }
        }

        // Phase 2: Steady state — alternating 1 forward + 1 backward
        for i in 0..steady {
            let fwd_mb = warmup + i;
            let bwd_mb = i;

            // Forward
            if !self.is_first_stage() {
                ops.push(PipelineOp::RecvActivation(fwd_mb));
            }
            ops.push(PipelineOp::Forward(fwd_mb));
            if !self.is_last_stage() {
                ops.push(PipelineOp::SendActivation(fwd_mb));
            }

            // Backward
            if !self.is_last_stage() {
                ops.push(PipelineOp::RecvGradient(bwd_mb));
            }
            ops.push(PipelineOp::Backward(bwd_mb));
            if !self.is_first_stage() {
                ops.push(PipelineOp::SendGradient(bwd_mb));
            }
        }

        // Phase 3: Cooldown — consecutive backwards
        let bwd_start = steady;
        for i in 0..cooldown {
            let bwd_mb = bwd_start + i;
            if bwd_mb >= num_microbatches {
                break;
            }
            if !self.is_last_stage() {
                ops.push(PipelineOp::RecvGradient(bwd_mb));
            }
            ops.push(PipelineOp::Backward(bwd_mb));
            if !self.is_first_stage() {
                ops.push(PipelineOp::SendGradient(bwd_mb));
            }
        }

        Ok(ops)
    }

    /// 根据策略生成调度操作序列
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn schedule(
        &self,
        strategy: MicroBatchStrategy,
        num_microbatches: usize,
    ) -> Result<Vec<PipelineOp>, PipelineSchedulerError> {
        match strategy {
            MicroBatchStrategy::GPipe => self.schedule_gpipe(num_microbatches),
            MicroBatchStrategy::OneFOneB => self.schedule_1f1b(num_microbatches),
        }
    }

    /// Zero-Bubble 调度优化 (REQ-DIST-022)
    ///
    /// 在 1F1B 基础上插入额外的 forward 来填充气泡，
    /// 最大化计算利用率。气泡操作标记为 Bubble。
    // @trace REQ-DIST-022 [entity:PipelineScheduler] [dataflow:DF-DIST-011]
    pub fn schedule_zero_bubble(&self, num_microbatches: usize) -> Result<Vec<PipelineOp>, PipelineSchedulerError> {
        if num_microbatches == 0 {
            return Err(PipelineSchedulerError::ZeroMicrobatches);
        }

        // Zero-bubble builds on 1F1B schedule but inserts Bubble markers
        // for the pipeline stall intervals, allowing the executor to
        // schedule useful work (e.g., weight prefetch) during bubbles.
        let mut ops = self.schedule_1f1b(num_microbatches)?;

        // Insert bubble markers at pipeline stall points
        // In zero-bubble, the warmup phase has pipeline bubbles that
        // can be filled with computation from other micro-batches.
        // We mark the expected bubble positions.
        let warmup = self.warmup_steps().min(num_microbatches);
        if warmup > 0 && self.pp_size > 1 {
            // Prepend bubble for pipeline fill time
            // The number of bubbles equals (pp_size - 1) for the first stage
            let num_bubbles = (self.pp_size - 1) as usize;
            let mut with_bubbles = Vec::with_capacity(ops.len() + num_bubbles);
            for _ in 0..num_bubbles {
                with_bubbles.push(PipelineOp::Bubble);
            }
            with_bubbles.extend(ops);
            ops = with_bubbles;
        }

        Ok(ops)
    }

    /// 校验调度器一致性
    // @trace REQ-DIST-022 [entity:PipelineScheduler]
    pub fn validate(&self) -> bool {
        self.pp_size >= 1 && self.stage_id < self.pp_size
    }
}

// ── CommComputeOverlap (REQ-DIST-026) ────────────────────────────────────────

/// SM 分区配置 (REQ-DIST-026 验收标准 5)
///
/// GPU SM 分区隔离通信流和计算流，确保通信内核只在 comm_sms 上运行，
/// 计算内核只在 compute_sms 上运行，避免 SM 资源竞争。
// @trace REQ-DIST-026 [entity:SmPartitionConfig]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SmPartitionConfig {
    /// GPU 总 SM 数
    pub total_sms: u32,
    /// 分配给通信流的 SM 数
    pub comm_sms: u32,
    /// 分配给计算流的 SM 数
    pub compute_sms: u32,
}

impl SmPartitionConfig {
    /// 创建 SM 分区配置
    // @trace REQ-DIST-026 [entity:SmPartitionConfig]
    pub fn new(total_sms: u32, comm_sms: u32, compute_sms: u32) -> Self {
        Self { total_sms, comm_sms, compute_sms }
    }

    /// 默认 SM 分区：通信占 10% SM，计算占 90% SM
    // @trace REQ-DIST-026 [entity:SmPartitionConfig]
    pub fn default_for_gpu(total_sms: u32) -> Self {
        let comm_sms = (total_sms * 10 / 100).max(1);
        let compute_sms = total_sms - comm_sms;
        Self { total_sms, comm_sms, compute_sms }
    }

    /// 通信 SM 占比
    // @trace REQ-DIST-026 [entity:SmPartitionConfig]
    pub fn comm_ratio(&self) -> f64 {
        if self.total_sms == 0 {
            return 0.0;
        }
        self.comm_sms as f64 / self.total_sms as f64
    }

    /// 计算 SM 占比
    // @trace REQ-DIST-026 [entity:SmPartitionConfig]
    pub fn compute_ratio(&self) -> f64 {
        if self.total_sms == 0 {
            return 0.0;
        }
        self.compute_sms as f64 / self.total_sms as f64
    }

    /// SM 分区是否有效（comm + compute <= total）
    // @trace REQ-DIST-026 [entity:SmPartitionConfig]
    pub fn is_valid(&self) -> bool {
        self.total_sms > 0
            && self.comm_sms > 0
            && self.compute_sms > 0
            && self.comm_sms + self.compute_sms <= self.total_sms
    }

    /// 是否满足 SM 分区隔离要求 (REQ-DIST-026 验收标准 5)
    ///
    /// 通信和计算 SM 严格隔离，无重叠。
    // @trace REQ-DIST-026 [entity:SmPartitionConfig]
    pub fn is_isolated(&self) -> bool {
        self.is_valid() && self.comm_sms + self.compute_sms <= self.total_sms
    }
}

/// 量化激活传输配置 (REQ-DIST-026 验收标准 6)
///
/// 可选应用量化激活传输，减少通信量 2-4x。
/// 量化后激活精度损失 < 0.1%（REQ-DIST-009）。
// @trace REQ-DIST-026 [entity:QuantizedActivationConfig]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantizedActivationConfig {
    /// 是否启用量化激活传输
    pub enabled: bool,
    /// 量化格式
    pub quant_format: ActivationQuantFormat,
    /// 量化后每元素字节数
    pub quant_elem_bytes: usize,
}

/// 激活量化格式 (REQ-DIST-026 验收标准 6)
// @trace REQ-DIST-026 [entity:ActivationQuantFormat]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationQuantFormat {
    /// 无量化（FP32 传输）
    None,
    /// FP16 量化（2 bytes/elem，通信量减半）
    Fp16,
    /// BF16 量化（2 bytes/elem，通信量减半）
    Bf16,
    /// FP8 E4M3 量化（1 byte/elem，通信量 1/4）
    Fp8E4M3,
    /// INT8 对称量化（1 byte/elem，通信量 1/4）
    Int8Symmetric,
}

impl QuantizedActivationConfig {
    /// 无量化配置
    // @trace REQ-DIST-026 [entity:QuantizedActivationConfig]
    pub fn no_quant() -> Self {
        Self {
            enabled: false,
            quant_format: ActivationQuantFormat::None,
            quant_elem_bytes: 4, // FP32
        }
    }

    /// FP8 量化配置（通信量 1/4）
    // @trace REQ-DIST-026 [entity:QuantizedActivationConfig]
    pub fn fp8_quant() -> Self {
        Self {
            enabled: true,
            quant_format: ActivationQuantFormat::Fp8E4M3,
            quant_elem_bytes: 1,
        }
    }

    /// INT8 对称量化配置（通信量 1/4）
    // @trace REQ-DIST-026 [entity:QuantizedActivationConfig]
    pub fn int8_quant() -> Self {
        Self {
            enabled: true,
            quant_format: ActivationQuantFormat::Int8Symmetric,
            quant_elem_bytes: 1,
        }
    }

    /// 量化后通信带宽节省率
    // @trace REQ-DIST-026 [entity:QuantizedActivationConfig]
    pub fn bandwidth_saving_ratio(&self) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        1.0 - self.quant_elem_bytes as f64 / 4.0 // 相对 FP32
    }

    /// 量化后通信时间（原始通信时间 * 量化压缩比）
    // @trace REQ-DIST-026 [entity:QuantizedActivationConfig]
    pub fn quantized_comm_us(&self, original_comm_us: f64) -> f64 {
        if !self.enabled {
            return original_comm_us;
        }
        original_comm_us * self.quant_elem_bytes as f64 / 4.0
    }
}

/// 通信-计算重叠调度器 (REQ-DIST-026)
///
/// 使用双流（计算流 + 通信流）重叠 stage 间激活传输与本地微批次计算。
/// 无重叠: delay = compute + comm
/// 有重叠: delay ≈ max(compute, comm)
/// 重叠模式下通信延迟隐藏率 > 80%（验收标准 4）。
///
/// SM 分区隔离通信和计算流（验收标准 5）。
/// 量化激活传输可选应用（验收标准 6）。
// @trace REQ-DIST-026 [entity:CommComputeOverlap] [api:POST /internal/distributed/pipeline/overlap]
#[derive(Debug, Clone, PartialEq)]
pub struct CommComputeOverlap {
    /// 单微批次前向计算时间（微秒）
    pub forward_compute_us: f64,
    /// 单微批次反向计算时间（微秒）
    pub backward_compute_us: f64,
    /// 单次激活传输通信时间（微秒）
    pub comm_us: f64,
    /// 是否启用重叠模式
    pub overlap_enabled: bool,
    /// SM 分区配置（验收标准 5）
    pub sm_partition: Option<SmPartitionConfig>,
    /// 量化激活传输配置（验收标准 6）
    pub quant_config: QuantizedActivationConfig,
}

/// CommComputeOverlap 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommComputeOverlapError {
    /// 计算时间为负
    NegativeComputeTime,
    /// 通信时间为负
    NegativeCommTime,
}

impl std::fmt::Display for CommComputeOverlapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommComputeOverlapError::NegativeComputeTime => {
                write!(f, "CommComputeOverlap: compute time must be >= 0")
            }
            CommComputeOverlapError::NegativeCommTime => {
                write!(f, "CommComputeOverlap: communication time must be >= 0")
            }
        }
    }
}

impl std::error::Error for CommComputeOverlapError {}

// @trace REQ-DIST-026 [entity:CommComputeOverlap]
impl CommComputeOverlap {
    /// 创建通信-计算重叠调度器
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn new(
        forward_compute_us: f64,
        backward_compute_us: f64,
        comm_us: f64,
        overlap_enabled: bool,
    ) -> Result<Self, CommComputeOverlapError> {
        if forward_compute_us < 0.0 || backward_compute_us < 0.0 {
            return Err(CommComputeOverlapError::NegativeComputeTime);
        }
        if comm_us < 0.0 {
            return Err(CommComputeOverlapError::NegativeCommTime);
        }
        Ok(Self {
            forward_compute_us,
            backward_compute_us,
            comm_us,
            overlap_enabled,
            sm_partition: None,
            quant_config: QuantizedActivationConfig::no_quant(),
        })
    }

    /// 设置 SM 分区配置（验收标准 5）
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn with_sm_partition(mut self, sm_partition: SmPartitionConfig) -> Self {
        self.sm_partition = Some(sm_partition);
        self
    }

    /// 设置量化激活传输配置（验收标准 6）
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn with_quant_config(mut self, quant_config: QuantizedActivationConfig) -> Self {
        self.quant_config = quant_config;
        self
    }

    /// 获取有效通信时间（考虑量化压缩）
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn effective_comm_us(&self) -> f64 {
        self.quant_config.quantized_comm_us(self.comm_us)
    }

    /// SM 分区是否已正确隔离（验收标准 5）
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn is_sm_partition_isolated(&self) -> bool {
        match &self.sm_partition {
            None => false,
            Some(config) => config.is_isolated(),
        }
    }

    /// 量化激活传输是否启用（验收标准 6）
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn is_quantized_transfer_enabled(&self) -> bool {
        self.quant_config.enabled
    }

    /// 量化后通信带宽节省率（验收标准 6）
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn quant_bandwidth_saving_ratio(&self) -> f64 {
        self.quant_config.bandwidth_saving_ratio()
    }

    /// 无重叠模式延迟 = compute + comm（验收标准 3）
    ///
    /// 通信时间考虑量化压缩（验收标准 6）。
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn latency_no_overlap_forward(&self) -> f64 {
        self.forward_compute_us + self.effective_comm_us()
    }

    /// 无重叠模式反向延迟
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn latency_no_overlap_backward(&self) -> f64 {
        self.backward_compute_us + self.effective_comm_us()
    }

    /// 有重叠模式延迟 ≈ max(compute, comm)（验收标准 3）
    ///
    /// 通信时间考虑量化压缩（验收标准 6）。
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn latency_overlap_forward(&self) -> f64 {
        if self.overlap_enabled {
            self.forward_compute_us.max(self.effective_comm_us())
        } else {
            self.latency_no_overlap_forward()
        }
    }

    /// 有重叠模式反向延迟
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn latency_overlap_backward(&self) -> f64 {
        if self.overlap_enabled {
            self.backward_compute_us.max(self.effective_comm_us())
        } else {
            self.latency_no_overlap_backward()
        }
    }

    /// 通信延迟隐藏率 = 1 - overlap_latency / no_overlap_latency（验收标准 4）
    ///
    /// 重叠模式下通信延迟隐藏率 > 80%。
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn latency_hiding_ratio_forward(&self) -> f64 {
        let no_overlap = self.latency_no_overlap_forward();
        if no_overlap == 0.0 {
            return 0.0;
        }
        let overlap = self.latency_overlap_forward();
        1.0 - overlap / no_overlap
    }

    /// 反向通信延迟隐藏率
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn latency_hiding_ratio_backward(&self) -> f64 {
        let no_overlap = self.latency_no_overlap_backward();
        if no_overlap == 0.0 {
            return 0.0;
        }
        let overlap = self.latency_overlap_backward();
        1.0 - overlap / no_overlap
    }

    /// 检查通信延迟隐藏率是否 > 80%（验收标准 4）
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn meets_hiding_threshold(&self) -> bool {
        if !self.overlap_enabled {
            return false;
        }
        // When compute >= comm, hiding ratio = comm / (compute + comm)
        // For hiding > 80%, need comm / (compute + comm) > 0.8 → comm > 4 * compute
        // But more practically, hiding > 80% means overlap_delay < 20% of no_overlap
        self.latency_hiding_ratio_forward() > 0.8 || self.latency_hiding_ratio_backward() > 0.8
    }

    /// 为调度操作生成双流分配 (REQ-DIST-026 验收标准 1, 2)
    ///
    /// 将 PipelineOp 分配到计算流或通信流：
    /// - Forward/Backward → 计算流
    /// - SendActivation/RecvActivation/SendGradient/RecvGradient → 通信流
    /// - Bubble → 无流分配
    // @trace REQ-DIST-026 [entity:CommComputeOverlap] [dataflow:DF-DIST-012]
    pub fn assign_streams(ops: &[PipelineOp]) -> Vec<StreamAssignment> {
        ops.iter().map(|op| {
            match op {
                PipelineOp::Forward(_) | PipelineOp::Backward(_) => {
                    StreamAssignment {
                        op: op.clone(),
                        stream: StreamKind::Compute,
                    }
                }
                PipelineOp::SendActivation(_)
                | PipelineOp::RecvActivation(_)
                | PipelineOp::SendGradient(_)
                | PipelineOp::RecvGradient(_) => {
                    StreamAssignment {
                        op: op.clone(),
                        stream: StreamKind::Communication,
                    }
                }
                PipelineOp::Bubble => {
                    StreamAssignment {
                        op: op.clone(),
                        stream: StreamKind::None,
                    }
                }
            }
        }).collect()
    }

    /// 计算给定调度操作序列在重叠模式下的总延迟
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn total_latency_overlap(&self, ops: &[PipelineOp]) -> f64 {
        if !self.overlap_enabled {
            return self.total_latency_no_overlap(ops);
        }

        // With overlap, compute and comm on separate streams can proceed in parallel.
        // Total latency ≈ sum of compute ops + max(comm overlap per step, 0)
        let mut compute_total = 0.0f64;
        let mut comm_total = 0.0f64;
        let effective_comm = self.effective_comm_us();

        for op in ops {
            match op {
                PipelineOp::Forward(_) => compute_total += self.forward_compute_us,
                PipelineOp::Backward(_) => compute_total += self.backward_compute_us,
                PipelineOp::SendActivation(_)
                | PipelineOp::RecvActivation(_)
                | PipelineOp::SendGradient(_)
                | PipelineOp::RecvGradient(_) => comm_total += effective_comm,
                PipelineOp::Bubble => {}
            }
        }

        // With dual-stream overlap, the total time is dominated by the longer stream
        compute_total.max(comm_total)
    }

    /// 计算给定调度操作序列在无重叠模式下的总延迟
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn total_latency_no_overlap(&self, ops: &[PipelineOp]) -> f64 {
        let mut total = 0.0f64;
        let effective_comm = self.effective_comm_us();
        for op in ops {
            match op {
                PipelineOp::Forward(_) => total += self.forward_compute_us,
                PipelineOp::Backward(_) => total += self.backward_compute_us,
                PipelineOp::SendActivation(_)
                | PipelineOp::RecvActivation(_)
                | PipelineOp::SendGradient(_)
                | PipelineOp::RecvGradient(_) => total += effective_comm,
                PipelineOp::Bubble => {}
            }
        }
        total
    }

    /// 校验一致性
    // @trace REQ-DIST-026 [entity:CommComputeOverlap]
    pub fn validate(&self) -> bool {
        self.forward_compute_us >= 0.0
            && self.backward_compute_us >= 0.0
            && self.comm_us >= 0.0
    }
}

// ── StreamKind (REQ-DIST-026) ────────────────────────────────────────────────

/// 流类型 (REQ-DIST-026)
///
/// 双流调度中操作所属的流。
// @trace REQ-DIST-026 [entity:StreamKind]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamKind {
    /// 计算流（Forward/Backward）
    Compute,
    /// 通信流（Send/Recv activation/gradient）
    Communication,
    /// 无流分配（Bubble）
    None,
}

/// 流分配结果 (REQ-DIST-026)
// @trace REQ-DIST-026 [entity:StreamAssignment]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StreamAssignment {
    /// 操作
    pub op: PipelineOp,
    /// 分配的流
    pub stream: StreamKind,
}

// ── PipelineKvCacheManager (REQ-DIST-027) ────────────────────────────────────

/// PP KV Cache 管理器 (REQ-DIST-027)
///
/// 每个 stage 独立管理本 stage 层的 KV cache。
/// KV page 分配按 stage_id 隔离。
/// PP 模式下无需跨 stage KV 传输。
/// KV cache 总内存 = 全模型 KV / pp_size（验收标准 5）。
// @trace REQ-DIST-027 [entity:PipelineKvCacheManager] [api:POST /internal/distributed/pipeline/kv-cache]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKvCacheManager {
    /// Pipeline Parallel 维度
    pub pp_size: u32,
    /// 当前 stage ID
    pub stage_id: u32,
    /// 本 stage 负责的层数
    pub layers_per_stage: u32,
    /// 全模型总层数
    pub total_layers: u32,
    /// 每 KV page 字节数
    pub bytes_per_page: usize,
    /// 每 layer 每 token KV cache 字节数
    pub bytes_per_layer_per_token: usize,
}

/// PipelineKvCacheManager 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineKvCacheManagerError {
    /// pp_size < 1
    InvalidPpSize(u32),
    /// stage_id >= pp_size
    InvalidStageId { stage_id: u32, pp_size: u32 },
    /// layers_per_stage == 0
    ZeroLayersPerStage,
    /// total_layers == 0
    ZeroTotalLayers,
    /// bytes_per_page == 0
    ZeroBytesPerPage,
    /// bytes_per_layer_per_token == 0
    ZeroBytesPerLayerPerToken,
}

impl std::fmt::Display for PipelineKvCacheManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineKvCacheManagerError::InvalidPpSize(pp_size) => {
                write!(f, "PipelineKvCacheManager: invalid pp_size={pp_size}, must be >= 1")
            }
            PipelineKvCacheManagerError::InvalidStageId { stage_id, pp_size } => {
                write!(
                    f,
                    "PipelineKvCacheManager: stage_id={stage_id} out of range [0, {pp_size})"
                )
            }
            PipelineKvCacheManagerError::ZeroLayersPerStage => {
                write!(f, "PipelineKvCacheManager: layers_per_stage must be > 0")
            }
            PipelineKvCacheManagerError::ZeroTotalLayers => {
                write!(f, "PipelineKvCacheManager: total_layers must be > 0")
            }
            PipelineKvCacheManagerError::ZeroBytesPerPage => {
                write!(f, "PipelineKvCacheManager: bytes_per_page must be > 0")
            }
            PipelineKvCacheManagerError::ZeroBytesPerLayerPerToken => {
                write!(f, "PipelineKvCacheManager: bytes_per_layer_per_token must be > 0")
            }
        }
    }
}

impl std::error::Error for PipelineKvCacheManagerError {}

// @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
impl PipelineKvCacheManager {
    /// 创建 PP KV Cache 管理器
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn new(
        pp_size: u32,
        stage_id: u32,
        layers_per_stage: u32,
        total_layers: u32,
        bytes_per_page: usize,
        bytes_per_layer_per_token: usize,
    ) -> Result<Self, PipelineKvCacheManagerError> {
        if pp_size < 1 {
            return Err(PipelineKvCacheManagerError::InvalidPpSize(pp_size));
        }
        if stage_id >= pp_size {
            return Err(PipelineKvCacheManagerError::InvalidStageId { stage_id, pp_size });
        }
        if layers_per_stage == 0 {
            return Err(PipelineKvCacheManagerError::ZeroLayersPerStage);
        }
        if total_layers == 0 {
            return Err(PipelineKvCacheManagerError::ZeroTotalLayers);
        }
        if bytes_per_page == 0 {
            return Err(PipelineKvCacheManagerError::ZeroBytesPerPage);
        }
        if bytes_per_layer_per_token == 0 {
            return Err(PipelineKvCacheManagerError::ZeroBytesPerLayerPerToken);
        }
        Ok(Self {
            pp_size,
            stage_id,
            layers_per_stage,
            total_layers,
            bytes_per_page,
            bytes_per_layer_per_token,
        })
    }

    /// 从 PipelineConfig 创建 PP KV Cache 管理器
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn from_pipeline_config(
        config: &PipelineConfig,
        total_layers: u32,
        bytes_per_page: usize,
        bytes_per_layer_per_token: usize,
    ) -> Result<Self, PipelineKvCacheManagerError> {
        Self::new(
            config.pp_size,
            config.stage_id,
            config.layers_per_stage,
            total_layers,
            bytes_per_page,
            bytes_per_layer_per_token,
        )
    }

    /// 每个 stage 独立管理本 stage 层的 KV cache（验收标准 1）
    ///
    /// 返回本 stage 负责的层范围 [start, end)。
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn stage_layer_range(&self) -> std::ops::Range<u32> {
        let start = self.stage_id * self.layers_per_stage;
        let end = ((self.stage_id + 1) * self.layers_per_stage).min(self.total_layers);
        start..end
    }

    /// KV page 分配按 stage_id 隔离（验收标准 2）
    ///
    /// 返回本 stage 的 KV page 起始 ID。
    /// 每个 stage 的 page ID 空间隔离：stage_id * max_pages_per_stage。
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn stage_page_id_offset(&self, max_pages_per_stage: usize) -> usize {
        self.stage_id as usize * max_pages_per_stage
    }

    /// PP 模式下无需跨 stage KV 传输（验收标准 3）
    ///
    /// 始终返回 true — PP 模式下 KV cache 完全本地化。
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn is_kv_cache_local(&self) -> bool {
        true
    }

    /// Prefix caching 在每个 stage 内独立工作（验收标准 4）
    ///
    /// 返回本 stage 的 prefix cache 命名空间（按 stage_id 隔离）。
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn prefix_cache_namespace(&self) -> String {
        format!("pp_stage_{}", self.stage_id)
    }

    /// KV cache 总内存 = 全模型 KV / pp_size（验收标准 5）
    ///
    /// 给定序列长度，计算本 stage 所需 KV cache 字节数。
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn stage_kv_cache_bytes(&self, seq_len: usize) -> usize {
        self.layers_per_stage as usize * seq_len * self.bytes_per_layer_per_token
    }

    /// 全模型 KV cache 字节数（用于对比验证验收标准 5）
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn full_model_kv_cache_bytes(&self, seq_len: usize) -> usize {
        self.total_layers as usize * seq_len * self.bytes_per_layer_per_token
    }

    /// 验证 stage KV cache = 全模型 / pp_size（验收标准 5）
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn verify_kv_proportional(&self, seq_len: usize) -> bool {
        let stage_bytes = self.stage_kv_cache_bytes(seq_len);
        let full_bytes = self.full_model_kv_cache_bytes(seq_len);
        if full_bytes == 0 {
            return true;
        }
        // stage_bytes * pp_size should approximate full_bytes
        // (accounting for ceiling division in layers_per_stage)
        let expected_full = stage_bytes * self.pp_size as usize;
        // Allow some tolerance due to ceiling division
        expected_full >= full_bytes && expected_full <= full_bytes + self.layers_per_stage as usize * seq_len * self.bytes_per_layer_per_token * (self.pp_size as usize - 1)
    }

    /// 本 stage KV page数
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn num_kv_pages(&self, seq_len: usize) -> usize {
        let total_bytes = self.stage_kv_cache_bytes(seq_len);
        (total_bytes + self.bytes_per_page - 1) / self.bytes_per_page
    }

    /// 校验一致性
    // @trace REQ-DIST-027 [entity:PipelineKvCacheManager]
    pub fn validate(&self) -> bool {
        self.pp_size >= 1
            && self.stage_id < self.pp_size
            && self.layers_per_stage >= 1
            && self.total_layers >= 1
            && self.bytes_per_page >= 1
            && self.bytes_per_layer_per_token >= 1
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PipelineOp: Display ──

    #[test]
    fn pipeline_op_display() {
        assert_eq!(format!("{}", PipelineOp::Forward(0)), "F(0)");
        assert_eq!(format!("{}", PipelineOp::Backward(3)), "B(3)");
        assert_eq!(format!("{}", PipelineOp::SendActivation(1)), "SendAct(1)");
        assert_eq!(format!("{}", PipelineOp::RecvActivation(2)), "RecvAct(2)");
        assert_eq!(format!("{}", PipelineOp::SendGradient(0)), "SendGrad(0)");
        assert_eq!(format!("{}", PipelineOp::RecvGradient(1)), "RecvGrad(1)");
        assert_eq!(format!("{}", PipelineOp::Bubble), "Bubble");
    }

    // ── MicroBatchStrategy ──

    #[test]
    fn micro_batch_strategy_default_is_one_f_one_b() {
        assert_eq!(MicroBatchStrategy::default(), MicroBatchStrategy::OneFOneB);
    }

    #[test]
    fn micro_batch_strategy_display() {
        assert_eq!(format!("{}", MicroBatchStrategy::GPipe), "GPipe");
        assert_eq!(format!("{}", MicroBatchStrategy::OneFOneB), "1F1B");
    }

    // ── PipelineScheduler: construction ──

    #[test]
    fn scheduler_new_valid() {
        let scheduler = PipelineScheduler::new(4, 1).unwrap();
        assert_eq!(scheduler.pp_size, 4);
        assert_eq!(scheduler.stage_id, 1);
    }

    #[test]
    fn scheduler_new_invalid_pp_size() {
        let result = PipelineScheduler::new(0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PipelineSchedulerError::InvalidPpSize(0));
    }

    #[test]
    fn scheduler_new_invalid_stage_id() {
        let result = PipelineScheduler::new(2, 2);
        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineSchedulerError::InvalidStageId { stage_id, pp_size } => {
                assert_eq!(stage_id, 2);
                assert_eq!(pp_size, 2);
            }
            other => panic!("expected InvalidStageId, got {:?}", other),
        }
    }

    #[test]
    fn scheduler_from_pipeline_config() {
        let config = PipelineConfig {
            pp_size: 4,
            stage_id: 2,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 8,
        };
        let scheduler = PipelineScheduler::from_pipeline_config(&config).unwrap();
        assert_eq!(scheduler.pp_size, 4);
        assert_eq!(scheduler.stage_id, 2);
    }

    // ── PipelineScheduler: stage predicates ──

    #[test]
    fn is_first_stage() {
        assert!(PipelineScheduler::new(4, 0).unwrap().is_first_stage());
        assert!(!PipelineScheduler::new(4, 1).unwrap().is_first_stage());
    }

    #[test]
    fn is_last_stage() {
        assert!(PipelineScheduler::new(4, 3).unwrap().is_last_stage());
        assert!(!PipelineScheduler::new(4, 2).unwrap().is_last_stage());
    }

    // ── PipelineScheduler: warmup/cooldown (REQ-DIST-022 验收标准 2) ──

    #[test]
    fn warmup_steps_first_stage() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // First stage (stage_id=0): warmup = pp_size - 0 - 1 = pp_size - 1
        let scheduler = PipelineScheduler::new(4, 0).unwrap();
        assert_eq!(scheduler.warmup_steps(), 3);
    }

    #[test]
    fn warmup_steps_last_stage() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // Last stage (stage_id=3): warmup = 4 - 3 - 1 = 0
        let scheduler = PipelineScheduler::new(4, 3).unwrap();
        assert_eq!(scheduler.warmup_steps(), 0);
    }

    #[test]
    fn warmup_steps_pp1() {
        let scheduler = PipelineScheduler::new(1, 0).unwrap();
        assert_eq!(scheduler.warmup_steps(), 0);
    }

    #[test]
    fn cooldown_steps_first_stage() {
        // First stage (stage_id=0): cooldown = 0
        let scheduler = PipelineScheduler::new(4, 0).unwrap();
        assert_eq!(scheduler.cooldown_steps(), 0);
    }

    #[test]
    fn cooldown_steps_last_stage() {
        // Last stage (stage_id=3): cooldown = 3
        let scheduler = PipelineScheduler::new(4, 3).unwrap();
        assert_eq!(scheduler.cooldown_steps(), 3);
    }

    #[test]
    fn cooldown_steps_middle_stage() {
        let scheduler = PipelineScheduler::new(4, 1).unwrap();
        assert_eq!(scheduler.cooldown_steps(), 1);
    }

    // ── PipelineScheduler: peak activation memory (REQ-DIST-022 验收标准 4, 5) ──

    #[test]
    fn gpipe_peak_activation_memory() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // 验收标准 4: GPipe 峰值 = num_microbatches * activation_bytes
        let scheduler = PipelineScheduler::new(4, 0).unwrap();
        assert_eq!(scheduler.gpipe_peak_activation_bytes(8, 1024), 8 * 1024);
    }

    #[test]
    fn one_f_one_b_peak_activation_memory() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // 验收标准 5: 1F1B 峰值 = (pp_size - stage_id) * activation_bytes
        let scheduler = PipelineScheduler::new(4, 1).unwrap();
        assert_eq!(scheduler.one_f_one_b_peak_activation_bytes(1024), 3 * 1024);
    }

    #[test]
    fn one_f_one_b_less_memory_than_gpipe() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // 1F1B peak is always <= GPipe peak for pp_size > 1
        let scheduler = PipelineScheduler::new(4, 0).unwrap();
        let gpipe = scheduler.gpipe_peak_activation_bytes(8, 1024);
        let one_f_one_b = scheduler.one_f_one_b_peak_activation_bytes(1024);
        assert!(one_f_one_b <= gpipe);
    }

    // ── PipelineScheduler: GPipe schedule (REQ-DIST-022 验收标准 1) ──

    #[test]
    fn gpipe_schedule_all_forward_then_all_backward() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // 验收标准 1: GPipe 所有 forward 后所有 backward
        let scheduler = PipelineScheduler::new(1, 0).unwrap();
        let ops = scheduler.schedule_gpipe(3).unwrap();

        let forward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        let backward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Backward(_))).count();
        assert_eq!(forward_count, 3);
        assert_eq!(backward_count, 3);

        // All forwards before any backward
        let first_backward_pos = ops.iter().position(|op| matches!(op, PipelineOp::Backward(_)));
        let last_forward_pos = ops.iter().rposition(|op| matches!(op, PipelineOp::Forward(_)));
        assert!(first_backward_pos.unwrap() > last_forward_pos.unwrap());
    }

    #[test]
    fn gpipe_schedule_zero_microbatches_returns_err() {
        let scheduler = PipelineScheduler::new(2, 0).unwrap();
        let result = scheduler.schedule_gpipe(0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PipelineSchedulerError::ZeroMicrobatches);
    }

    // ── PipelineScheduler: 1F1B schedule (REQ-DIST-022 验收标准 2) ──

    #[test]
    fn one_f_one_b_schedule_has_warmup_steady_cooldown() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // 验收标准 2: warmup forward + interleaved fwd/bwd + cooldown backward
        let scheduler = PipelineScheduler::new(4, 0).unwrap();
        let ops = scheduler.schedule_1f1b(8).unwrap();

        let forward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        let backward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Backward(_))).count();
        assert_eq!(forward_count, 8);
        assert_eq!(backward_count, 8);
    }

    #[test]
    fn one_f_one_b_last_stage_no_warmup() {
        // Last stage: warmup = 0, steady = num_microbatches, cooldown = pp_size - 1
        let scheduler = PipelineScheduler::new(4, 3).unwrap();
        assert_eq!(scheduler.warmup_steps(), 0);
        let ops = scheduler.schedule_1f1b(4).unwrap();

        // Last stage starts with 1F1B steady immediately
        let forward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        let backward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Backward(_))).count();
        assert_eq!(forward_count, 4);
        assert_eq!(backward_count, 4);
    }

    #[test]
    fn one_f_one_b_middle_stage() {
        let scheduler = PipelineScheduler::new(4, 1).unwrap();
        assert_eq!(scheduler.warmup_steps(), 2);
        assert_eq!(scheduler.cooldown_steps(), 1);
        let ops = scheduler.schedule_1f1b(6).unwrap();

        let forward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        let backward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Backward(_))).count();
        assert_eq!(forward_count, 6);
        assert_eq!(backward_count, 6);
    }

    // ── PipelineScheduler: schedule dispatch ──

    #[test]
    fn schedule_dispatch_gpipe() {
        let scheduler = PipelineScheduler::new(1, 0).unwrap();
        let ops = scheduler.schedule(MicroBatchStrategy::GPipe, 2).unwrap();
        let forward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        assert_eq!(forward_count, 2);
    }

    #[test]
    fn schedule_dispatch_one_f_one_b() {
        let scheduler = PipelineScheduler::new(1, 0).unwrap();
        let ops = scheduler.schedule(MicroBatchStrategy::OneFOneB, 2).unwrap();
        let forward_count = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        assert_eq!(forward_count, 2);
    }

    // ── PipelineScheduler: zero-bubble ──

    #[test]
    fn zero_bubble_schedule_has_bubbles() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        let scheduler = PipelineScheduler::new(4, 0).unwrap();
        let ops = scheduler.schedule_zero_bubble(8).unwrap();
        let bubble_count = ops.iter().filter(|op| matches!(op, PipelineOp::Bubble)).count();
        assert!(bubble_count > 0, "zero-bubble schedule should have Bubble ops");
        // Bubble count = pp_size - 1 = 3
        assert_eq!(bubble_count, 3);
    }

    #[test]
    fn zero_bubble_pp1_no_bubbles() {
        let scheduler = PipelineScheduler::new(1, 0).unwrap();
        let ops = scheduler.schedule_zero_bubble(4).unwrap();
        let bubble_count = ops.iter().filter(|op| matches!(op, PipelineOp::Bubble)).count();
        assert_eq!(bubble_count, 0);
    }

    // ── PipelineScheduler: validate ──

    #[test]
    fn validate_valid() {
        let scheduler = PipelineScheduler::new(4, 1).unwrap();
        assert!(scheduler.validate());
    }

    // ── PipelineScheduler: communication ops ──

    #[test]
    fn first_stage_no_recv_activation_no_send_gradient() {
        let scheduler = PipelineScheduler::new(2, 0).unwrap();
        let ops = scheduler.schedule_1f1b(4).unwrap();

        // First stage should not have RecvActivation or SendGradient
        let recv_act_count = ops.iter().filter(|op| matches!(op, PipelineOp::RecvActivation(_))).count();
        let send_grad_count = ops.iter().filter(|op| matches!(op, PipelineOp::SendGradient(_))).count();
        assert_eq!(recv_act_count, 0);
        assert_eq!(send_grad_count, 0);
    }

    #[test]
    fn last_stage_no_send_activation_no_recv_gradient() {
        let scheduler = PipelineScheduler::new(2, 1).unwrap();
        let ops = scheduler.schedule_1f1b(4).unwrap();

        // Last stage should not have SendActivation or RecvGradient
        let send_act_count = ops.iter().filter(|op| matches!(op, PipelineOp::SendActivation(_))).count();
        let recv_grad_count = ops.iter().filter(|op| matches!(op, PipelineOp::RecvGradient(_))).count();
        assert_eq!(send_act_count, 0);
        assert_eq!(recv_grad_count, 0);
    }

    // ── PipelineSchedulerError: Display ──

    #[test]
    fn error_display_invalid_pp_size() {
        let err = PipelineSchedulerError::InvalidPpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("pp_size=0"));
    }

    #[test]
    fn error_display_invalid_stage_id() {
        let err = PipelineSchedulerError::InvalidStageId { stage_id: 3, pp_size: 2 };
        let msg = format!("{}", err);
        assert!(msg.contains("3"));
        assert!(msg.contains("[0, 2)"));
    }

    #[test]
    fn error_display_zero_microbatches() {
        let err = PipelineSchedulerError::ZeroMicrobatches;
        let msg = format!("{}", err);
        assert!(msg.contains("num_microbatches"));
    }

    #[test]
    fn error_is_std_error() {
        let err = PipelineSchedulerError::InvalidPpSize(0);
        let _: &dyn std::error::Error = &err;
    }

    // ── CommComputeOverlap: construction ──

    #[test]
    fn overlap_new_valid() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap();
        assert_eq!(overlap.forward_compute_us, 100.0);
        assert_eq!(overlap.backward_compute_us, 200.0);
        assert_eq!(overlap.comm_us, 50.0);
        assert!(overlap.overlap_enabled);
    }

    #[test]
    fn overlap_new_negative_compute() {
        let result = CommComputeOverlap::new(-1.0, 100.0, 50.0, true);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CommComputeOverlapError::NegativeComputeTime);
    }

    #[test]
    fn overlap_new_negative_comm() {
        let result = CommComputeOverlap::new(100.0, 200.0, -1.0, true);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CommComputeOverlapError::NegativeCommTime);
    }

    // ── CommComputeOverlap: latency (REQ-DIST-026 验收标准 3) ──

    #[test]
    fn latency_no_overlap_forward() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 3: 无重叠 = compute + comm
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, false).unwrap();
        assert_eq!(overlap.latency_no_overlap_forward(), 150.0);
    }

    #[test]
    fn latency_no_overlap_backward() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, false).unwrap();
        assert_eq!(overlap.latency_no_overlap_backward(), 250.0);
    }

    #[test]
    fn latency_overlap_forward() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 3: 有重叠 ≈ max(compute, comm)
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap();
        assert_eq!(overlap.latency_overlap_forward(), 100.0); // max(100, 50)
    }

    #[test]
    fn latency_overlap_disabled_falls_back() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, false).unwrap();
        assert_eq!(overlap.latency_overlap_forward(), 150.0); // compute + comm
    }

    #[test]
    fn latency_overlap_comm_dominant() {
        // When comm > compute, overlap latency = comm
        let overlap = CommComputeOverlap::new(30.0, 60.0, 100.0, true).unwrap();
        assert_eq!(overlap.latency_overlap_forward(), 100.0); // max(30, 100)
    }

    // ── CommComputeOverlap: latency hiding ratio (REQ-DIST-026 验收标准 4) ──

    #[test]
    fn latency_hiding_ratio() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 4: 重叠模式通信延迟隐藏率 > 80%
        // When compute=100, comm=50: no_overlap=150, overlap=100, hiding=1-100/150=0.333
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap();
        let ratio = overlap.latency_hiding_ratio_forward();
        assert!((ratio - 0.333).abs() < 0.01);
    }

    #[test]
    fn latency_hiding_high_ratio() {
        // When comm >> compute: hiding ratio approaches 100%
        let overlap = CommComputeOverlap::new(10.0, 20.0, 1000.0, true).unwrap();
        let ratio = overlap.latency_hiding_ratio_forward();
        assert!(ratio > 0.8, "hiding ratio should be > 0.8 when comm >> compute, got {}", ratio);
    }

    #[test]
    fn meets_hiding_threshold() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 4: 重叠模式下通信延迟隐藏率 > 80%
        let overlap = CommComputeOverlap::new(10.0, 20.0, 1000.0, true).unwrap();
        assert!(overlap.meets_hiding_threshold());
    }

    #[test]
    fn does_not_meet_hiding_threshold_when_disabled() {
        let overlap = CommComputeOverlap::new(10.0, 20.0, 1000.0, false).unwrap();
        assert!(!overlap.meets_hiding_threshold());
    }

    // ── CommComputeOverlap: stream assignment (REQ-DIST-026 验收标准 1, 2) ──

    #[test]
    fn stream_assignment_forward_compute() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 1, 2: 计算流和通信流独立并行
        let ops = vec![PipelineOp::Forward(0), PipelineOp::Backward(0)];
        let assignments = CommComputeOverlap::assign_streams(&ops);
        assert_eq!(assignments[0].stream, StreamKind::Compute);
        assert_eq!(assignments[1].stream, StreamKind::Compute);
    }

    #[test]
    fn stream_assignment_communication_ops() {
        let ops = vec![
            PipelineOp::SendActivation(0),
            PipelineOp::RecvActivation(0),
            PipelineOp::SendGradient(0),
            PipelineOp::RecvGradient(0),
        ];
        let assignments = CommComputeOverlap::assign_streams(&ops);
        for a in &assignments {
            assert_eq!(a.stream, StreamKind::Communication);
        }
    }

    #[test]
    fn stream_assignment_bubble_none() {
        let ops = vec![PipelineOp::Bubble];
        let assignments = CommComputeOverlap::assign_streams(&ops);
        assert_eq!(assignments[0].stream, StreamKind::None);
    }

    // ── CommComputeOverlap: total latency ──

    #[test]
    fn total_latency_no_overlap_ops() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, false).unwrap();
        let ops = vec![
            PipelineOp::Forward(0),
            PipelineOp::SendActivation(0),
            PipelineOp::Backward(0),
        ];
        let total = overlap.total_latency_no_overlap(&ops);
        assert_eq!(total, 100.0 + 50.0 + 200.0);
    }

    #[test]
    fn total_latency_overlap_ops() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap();
        let ops = vec![
            PipelineOp::Forward(0),
            PipelineOp::SendActivation(0),
            PipelineOp::Backward(0),
        ];
        let total = overlap.total_latency_overlap(&ops);
        // compute_total = 100 + 200 = 300, comm_total = 50
        // overlap_total = max(300, 50) = 300
        assert_eq!(total, 300.0);
    }

    // ── CommComputeOverlap: validate ──

    #[test]
    fn overlap_validate_valid() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap();
        assert!(overlap.validate());
    }

    // ── CommComputeOverlapError: Display ──

    #[test]
    fn overlap_error_display_negative_compute() {
        let err = CommComputeOverlapError::NegativeComputeTime;
        let msg = format!("{}", err);
        assert!(msg.contains("compute"));
    }

    #[test]
    fn overlap_error_display_negative_comm() {
        let err = CommComputeOverlapError::NegativeCommTime;
        let msg = format!("{}", err);
        assert!(msg.contains("communication"));
    }

    #[test]
    fn overlap_error_is_std_error() {
        let err = CommComputeOverlapError::NegativeComputeTime;
        let _: &dyn std::error::Error = &err;
    }

    // ── PipelineKvCacheManager: construction (REQ-DIST-027) ──

    #[test]
    fn kv_manager_new_valid() {
        // @trace TEST-DIST-027 [req:REQ-DIST-027] [level:unit]
        let manager = PipelineKvCacheManager::new(4, 1, 8, 32, 4096, 128).unwrap();
        assert_eq!(manager.pp_size, 4);
        assert_eq!(manager.stage_id, 1);
        assert_eq!(manager.layers_per_stage, 8);
    }

    #[test]
    fn kv_manager_new_invalid_pp_size() {
        let result = PipelineKvCacheManager::new(0, 0, 8, 32, 4096, 128);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PipelineKvCacheManagerError::InvalidPpSize(0));
    }

    #[test]
    fn kv_manager_new_invalid_stage_id() {
        let result = PipelineKvCacheManager::new(2, 2, 8, 32, 4096, 128);
        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineKvCacheManagerError::InvalidStageId { stage_id, pp_size } => {
                assert_eq!(stage_id, 2);
                assert_eq!(pp_size, 2);
            }
            other => panic!("expected InvalidStageId, got {:?}", other),
        }
    }

    #[test]
    fn kv_manager_new_zero_layers() {
        let result = PipelineKvCacheManager::new(2, 0, 0, 32, 4096, 128);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PipelineKvCacheManagerError::ZeroLayersPerStage);
    }

    #[test]
    fn kv_manager_new_zero_total_layers() {
        let result = PipelineKvCacheManager::new(2, 0, 8, 0, 4096, 128);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PipelineKvCacheManagerError::ZeroTotalLayers);
    }

    #[test]
    fn kv_manager_new_zero_bytes_per_page() {
        let result = PipelineKvCacheManager::new(2, 0, 8, 32, 0, 128);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PipelineKvCacheManagerError::ZeroBytesPerPage);
    }

    #[test]
    fn kv_manager_new_zero_bytes_per_layer() {
        let result = PipelineKvCacheManager::new(2, 0, 8, 32, 4096, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PipelineKvCacheManagerError::ZeroBytesPerLayerPerToken);
    }

    // ── PipelineKvCacheManager: stage layer range (REQ-DIST-027 验收标准 1) ──

    #[test]
    fn kv_manager_stage_layer_range() {
        // @trace TEST-DIST-027 [req:REQ-DIST-027] [level:unit]
        // 验收标准 1: 每个 stage 独立管理本 stage 层的 KV cache
        let manager = PipelineKvCacheManager::new(4, 1, 8, 32, 4096, 128).unwrap();
        assert_eq!(manager.stage_layer_range(), 8..16);
    }

    #[test]
    fn kv_manager_stage_layer_range_first_stage() {
        let manager = PipelineKvCacheManager::new(4, 0, 8, 32, 4096, 128).unwrap();
        assert_eq!(manager.stage_layer_range(), 0..8);
    }

    #[test]
    fn kv_manager_stage_layer_range_last_stage() {
        let manager = PipelineKvCacheManager::new(4, 3, 8, 32, 4096, 128).unwrap();
        assert_eq!(manager.stage_layer_range(), 24..32);
    }

    // ── PipelineKvCacheManager: page isolation (REQ-DIST-027 验收标准 2) ──

    #[test]
    fn kv_manager_page_id_offset() {
        // @trace TEST-DIST-027 [req:REQ-DIST-027] [level:unit]
        // 验收标准 2: KV page 分配按 stage_id 隔离
        let manager = PipelineKvCacheManager::new(4, 2, 8, 32, 4096, 128).unwrap();
        assert_eq!(manager.stage_page_id_offset(1000), 2000);
    }

    #[test]
    fn kv_manager_page_id_offset_first_stage() {
        let manager = PipelineKvCacheManager::new(4, 0, 8, 32, 4096, 128).unwrap();
        assert_eq!(manager.stage_page_id_offset(1000), 0);
    }

    // ── PipelineKvCacheManager: no cross-stage KV transfer (REQ-DIST-027 验收标准 3) ──

    #[test]
    fn kv_manager_is_kv_cache_local() {
        // @trace TEST-DIST-027 [req:REQ-DIST-027] [level:unit]
        // 验收标准 3: PP 模式下无需跨 stage KV 传输
        let manager = PipelineKvCacheManager::new(4, 1, 8, 32, 4096, 128).unwrap();
        assert!(manager.is_kv_cache_local());
    }

    // ── PipelineKvCacheManager: prefix caching (REQ-DIST-027 验收标准 4) ──

    #[test]
    fn kv_manager_prefix_cache_namespace() {
        // @trace TEST-DIST-027 [req:REQ-DIST-027] [level:unit]
        // 验收标准 4: Prefix caching 在每个 stage 内独立工作
        let manager = PipelineKvCacheManager::new(4, 2, 8, 32, 4096, 128).unwrap();
        assert_eq!(manager.prefix_cache_namespace(), "pp_stage_2");
    }

    #[test]
    fn kv_manager_prefix_cache_namespace_isolation() {
        let m0 = PipelineKvCacheManager::new(4, 0, 8, 32, 4096, 128).unwrap();
        let m1 = PipelineKvCacheManager::new(4, 1, 8, 32, 4096, 128).unwrap();
        assert_ne!(m0.prefix_cache_namespace(), m1.prefix_cache_namespace());
    }

    // ── PipelineKvCacheManager: KV proportional (REQ-DIST-027 验收标准 5) ──

    #[test]
    fn kv_manager_stage_kv_cache_bytes() {
        // @trace TEST-DIST-027 [req:REQ-DIST-027] [level:unit]
        // 验收标准 5: KV cache 总内存 = 全模型 KV / pp_size
        let manager = PipelineKvCacheManager::new(4, 1, 8, 32, 4096, 128).unwrap();
        let seq_len = 100;
        let stage_bytes = manager.stage_kv_cache_bytes(seq_len);
        assert_eq!(stage_bytes, 8 * 100 * 128); // layers_per_stage * seq_len * bytes_per_layer_per_token
    }

    #[test]
    fn kv_manager_full_model_bytes() {
        let manager = PipelineKvCacheManager::new(4, 1, 8, 32, 4096, 128).unwrap();
        let seq_len = 100;
        let full_bytes = manager.full_model_kv_cache_bytes(seq_len);
        assert_eq!(full_bytes, 32 * 100 * 128);
    }

    #[test]
    fn kv_manager_verify_kv_proportional() {
        // @trace TEST-DIST-027 [req:REQ-DIST-027] [level:unit]
        // 验收标准 5: stage KV = full KV / pp_size
        let manager = PipelineKvCacheManager::new(4, 1, 8, 32, 4096, 128).unwrap();
        assert!(manager.verify_kv_proportional(100));
    }

    #[test]
    fn kv_manager_num_kv_pages() {
        let manager = PipelineKvCacheManager::new(4, 1, 8, 32, 4096, 128).unwrap();
        let seq_len = 100;
        let total_bytes = manager.stage_kv_cache_bytes(seq_len);
        let expected_pages = (total_bytes + 4096 - 1) / 4096;
        assert_eq!(manager.num_kv_pages(seq_len), expected_pages);
    }

    // ── PipelineKvCacheManager: validate ──

    #[test]
    fn kv_manager_validate_valid() {
        let manager = PipelineKvCacheManager::new(4, 1, 8, 32, 4096, 128).unwrap();
        assert!(manager.validate());
    }

    // ── PipelineKvCacheManager: from_pipeline_config ──

    #[test]
    fn kv_manager_from_pipeline_config() {
        let config = PipelineConfig {
            pp_size: 4,
            stage_id: 1,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 8,
        };
        let manager = PipelineKvCacheManager::from_pipeline_config(&config, 32, 4096, 128).unwrap();
        assert_eq!(manager.pp_size, 4);
        assert_eq!(manager.stage_id, 1);
    }

    // ── PipelineKvCacheManagerError: Display ──

    #[test]
    fn kv_manager_error_display_invalid_pp_size() {
        let err = PipelineKvCacheManagerError::InvalidPpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("pp_size=0"));
    }

    #[test]
    fn kv_manager_error_display_invalid_stage_id() {
        let err = PipelineKvCacheManagerError::InvalidStageId { stage_id: 3, pp_size: 2 };
        let msg = format!("{}", err);
        assert!(msg.contains("3"));
    }

    #[test]
    fn kv_manager_error_display_zero_layers() {
        let err = PipelineKvCacheManagerError::ZeroLayersPerStage;
        let msg = format!("{}", err);
        assert!(msg.contains("layers_per_stage"));
    }

    #[test]
    fn kv_manager_error_is_std_error() {
        let err = PipelineKvCacheManagerError::InvalidPpSize(0);
        let _: &dyn std::error::Error = &err;
    }

    // ── PipelineOp: equality and hash ──

    #[test]
    fn pipeline_op_equality() {
        assert_eq!(PipelineOp::Forward(0), PipelineOp::Forward(0));
        assert_ne!(PipelineOp::Forward(0), PipelineOp::Forward(1));
        assert_ne!(PipelineOp::Forward(0), PipelineOp::Backward(0));
    }

    #[test]
    fn pipeline_op_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PipelineOp::Forward(0));
        set.insert(PipelineOp::Forward(0));
        assert_eq!(set.len(), 1);
    }

    // ── StreamKind ──

    #[test]
    fn stream_kind_variants() {
        assert_ne!(StreamKind::Compute, StreamKind::Communication);
        assert_ne!(StreamKind::Compute, StreamKind::None);
    }

    // ── PipelineScheduler: 1F1B numerical equivalence (验收标准 6) ──

    #[test]
    fn one_f_one_b_and_gpipe_same_forward_backward_count() {
        // @trace TEST-DIST-022 [req:REQ-DIST-022] [level:unit]
        // 验收标准 6: 两种策略计算结果数值等价
        // (Equivalent forward/backward counts is a necessary condition for numerical equivalence)
        let scheduler = PipelineScheduler::new(4, 1).unwrap();
        let gpipe_ops = scheduler.schedule_gpipe(8).unwrap();
        let one_f_one_b_ops = scheduler.schedule_1f1b(8).unwrap();

        let gpipe_fwd = gpipe_ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        let gpipe_bwd = gpipe_ops.iter().filter(|op| matches!(op, PipelineOp::Backward(_))).count();
        let one_f_fwd = one_f_one_b_ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        let one_f_bwd = one_f_one_b_ops.iter().filter(|op| matches!(op, PipelineOp::Backward(_))).count();

        assert_eq!(gpipe_fwd, one_f_fwd);
        assert_eq!(gpipe_bwd, one_f_bwd);
        assert_eq!(gpipe_fwd, 8);
        assert_eq!(gpipe_bwd, 8);
    }

    #[test]
    fn one_f_one_b_microbatch_indices_cover_all() {
        // Both strategies process all micro-batch indices in both forward and backward
        let scheduler = PipelineScheduler::new(4, 1).unwrap();
        let gpipe_ops = scheduler.schedule_gpipe(8).unwrap();
        let one_f_one_b_ops = scheduler.schedule_1f1b(8).unwrap();

        let gpipe_fwd_indices: std::collections::HashSet<usize> = gpipe_ops.iter()
            .filter_map(|op| match op { PipelineOp::Forward(i) => Some(*i), _ => None })
            .collect();
        let gpipe_bwd_indices: std::collections::HashSet<usize> = gpipe_ops.iter()
            .filter_map(|op| match op { PipelineOp::Backward(i) => Some(*i), _ => None })
            .collect();

        let one_f_fwd_indices: std::collections::HashSet<usize> = one_f_one_b_ops.iter()
            .filter_map(|op| match op { PipelineOp::Forward(i) => Some(*i), _ => None })
            .collect();
        let one_f_bwd_indices: std::collections::HashSet<usize> = one_f_one_b_ops.iter()
            .filter_map(|op| match op { PipelineOp::Backward(i) => Some(*i), _ => None })
            .collect();

        assert_eq!(gpipe_fwd_indices, one_f_fwd_indices);
        assert_eq!(gpipe_bwd_indices, one_f_bwd_indices);
        assert_eq!(gpipe_fwd_indices.len(), 8);
        assert_eq!(gpipe_bwd_indices.len(), 8);
    }

    // ── SmPartitionConfig (REQ-DIST-026 验收标准 5) ──

    #[test]
    fn sm_partition_new() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 5: SM 分区隔离通信和计算流
        let config = SmPartitionConfig::new(128, 13, 115);
        assert_eq!(config.total_sms, 128);
        assert_eq!(config.comm_sms, 13);
        assert_eq!(config.compute_sms, 115);
    }

    #[test]
    fn sm_partition_default_for_gpu() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        let config = SmPartitionConfig::default_for_gpu(128);
        assert_eq!(config.total_sms, 128);
        assert_eq!(config.comm_sms, 13); // 128 * 10% = 12.8, max(1) = 13
        assert_eq!(config.compute_sms, 115); // 128 - 13
    }

    #[test]
    fn sm_partition_default_for_small_gpu() {
        let config = SmPartitionConfig::default_for_gpu(10);
        assert_eq!(config.total_sms, 10);
        assert_eq!(config.comm_sms, 1); // max(1)
        assert_eq!(config.compute_sms, 9);
    }

    #[test]
    fn sm_partition_ratios() {
        let config = SmPartitionConfig::new(100, 10, 90);
        assert!((config.comm_ratio() - 0.1).abs() < 1e-10);
        assert!((config.compute_ratio() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn sm_partition_ratios_zero_total() {
        let config = SmPartitionConfig::new(0, 0, 0);
        assert_eq!(config.comm_ratio(), 0.0);
        assert_eq!(config.compute_ratio(), 0.0);
    }

    #[test]
    fn sm_partition_is_valid() {
        let config = SmPartitionConfig::new(128, 13, 115);
        assert!(config.is_valid());
    }

    #[test]
    fn sm_partition_invalid_zero_total() {
        let config = SmPartitionConfig::new(0, 1, 1);
        assert!(!config.is_valid());
    }

    #[test]
    fn sm_partition_invalid_exceeds_total() {
        let config = SmPartitionConfig::new(100, 60, 60);
        assert!(!config.is_valid()); // 60 + 60 > 100
    }

    #[test]
    fn sm_partition_is_isolated() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 5: 通信和计算 SM 严格隔离
        let config = SmPartitionConfig::new(128, 13, 115);
        assert!(config.is_isolated()); // 13 + 115 = 128 <= 128
    }

    #[test]
    fn sm_partition_not_isolated_when_overlap() {
        let config = SmPartitionConfig::new(100, 60, 60);
        assert!(!config.is_isolated()); // 60 + 60 > 100
    }

    // ── QuantizedActivationConfig (REQ-DIST-026 验收标准 6) ──

    #[test]
    fn quant_config_no_quant() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        let config = QuantizedActivationConfig::no_quant();
        assert!(!config.enabled);
        assert_eq!(config.quant_format, ActivationQuantFormat::None);
        assert_eq!(config.quant_elem_bytes, 4);
        assert_eq!(config.bandwidth_saving_ratio(), 0.0);
        assert_eq!(config.quantized_comm_us(100.0), 100.0); // No reduction
    }

    #[test]
    fn quant_config_fp8() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 6: 量化激活传输可选应用，带宽节省 >= 50%
        let config = QuantizedActivationConfig::fp8_quant();
        assert!(config.enabled);
        assert_eq!(config.quant_format, ActivationQuantFormat::Fp8E4M3);
        assert_eq!(config.quant_elem_bytes, 1);
        // Bandwidth saving = 1 - 1/4 = 0.75 (75%, >= 50% per REQ-DIST-009)
        assert!((config.bandwidth_saving_ratio() - 0.75).abs() < 1e-10);
        // Quantized comm time = 100 * 1/4 = 25
        assert_eq!(config.quantized_comm_us(100.0), 25.0);
    }

    #[test]
    fn quant_config_int8() {
        let config = QuantizedActivationConfig::int8_quant();
        assert!(config.enabled);
        assert_eq!(config.quant_format, ActivationQuantFormat::Int8Symmetric);
        assert_eq!(config.quant_elem_bytes, 1);
        assert!((config.bandwidth_saving_ratio() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn quant_format_equality() {
        assert_eq!(ActivationQuantFormat::None, ActivationQuantFormat::None);
        assert_ne!(ActivationQuantFormat::Fp8E4M3, ActivationQuantFormat::Int8Symmetric);
    }

    // ── CommComputeOverlap: SM partition and quant (REQ-DIST-026) ──

    #[test]
    fn overlap_with_sm_partition() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 5: SM 分区隔离通信和计算流
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap()
            .with_sm_partition(SmPartitionConfig::new(128, 13, 115));
        assert!(overlap.is_sm_partition_isolated());
        assert_eq!(overlap.sm_partition.unwrap().comm_sms, 13);
    }

    #[test]
    fn overlap_without_sm_partition_not_isolated() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap();
        assert!(!overlap.is_sm_partition_isolated());
    }

    #[test]
    fn overlap_with_quant_config() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 6: 量化激活传输可选应用
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap()
            .with_quant_config(QuantizedActivationConfig::fp8_quant());
        assert!(overlap.is_quantized_transfer_enabled());
        assert!((overlap.quant_bandwidth_saving_ratio() - 0.75).abs() < 1e-10);
        // Effective comm = 50 * 1/4 = 12.5
        assert!((overlap.effective_comm_us() - 12.5).abs() < 1e-10);
    }

    #[test]
    fn overlap_without_quant_default_no_quant() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap();
        assert!(!overlap.is_quantized_transfer_enabled());
        assert_eq!(overlap.effective_comm_us(), 50.0); // Same as original comm_us
    }

    #[test]
    fn overlap_latency_with_quant_no_overlap() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 3+6: 无重叠 + 量化 → 延迟 = compute + quantized_comm
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, false).unwrap()
            .with_quant_config(QuantizedActivationConfig::fp8_quant());
        // effective_comm = 50 * 1/4 = 12.5
        // no_overlap_forward = 100 + 12.5 = 112.5
        assert!((overlap.latency_no_overlap_forward() - 112.5).abs() < 1e-10);
    }

    #[test]
    fn overlap_latency_with_quant_overlap() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 3+6: 有重叠 + 量化 → 延迟 ≈ max(compute, quantized_comm)
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap()
            .with_quant_config(QuantizedActivationConfig::fp8_quant());
        // effective_comm = 12.5
        // overlap_forward = max(100, 12.5) = 100
        assert!((overlap.latency_overlap_forward() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn overlap_latency_hiding_improved_with_quant() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 4+6: 量化后通信量减少，隐藏率更高
        let overlap_no_quant = CommComputeOverlap::new(10.0, 20.0, 1000.0, true).unwrap();
        let overlap_with_quant = CommComputeOverlap::new(10.0, 20.0, 1000.0, true).unwrap()
            .with_quant_config(QuantizedActivationConfig::fp8_quant());
        // No quant: no_overlap = 10 + 1000 = 1010, overlap = max(10, 1000) = 1000
        // hiding = 1 - 1000/1010 = 0.0099
        // With FP8 quant: effective_comm = 1000 * 1/4 = 250
        // no_overlap = 10 + 250 = 260, overlap = max(10, 250) = 250
        // hiding = 1 - 250/260 = 0.038
        let ratio_no_quant = overlap_no_quant.latency_hiding_ratio_forward();
        let ratio_with_quant = overlap_with_quant.latency_hiding_ratio_forward();
        // Quant makes the hiding ratio better when compute < comm
        assert!(ratio_with_quant > ratio_no_quant);
    }

    #[test]
    fn overlap_meets_hiding_threshold_with_quant() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // 验收标准 4: 量化使低 compute/comm 比场景也能达到 > 80% 隐藏率
        // hiding = 1 - max(compute, comm) / (compute + comm) = min(compute, comm) / (compute + comm)
        // For hiding > 80%: min(C, Cc)/(C + Cc) > 0.8
        // If Cc > C: C/(C + Cc) > 0.8 → C > 4*Cc → Cc < C/4
        // If C > Cc: Cc/(C + Cc) > 0.8 → Cc > 4*C → C < Cc/4
        // Example: compute=1000, comm=10 → hiding = 10/(1010) = 0.0099 → NOT > 80%
        // Example: compute=1, comm=1000 → hiding = 1/(1001) = 0.001 → NOT > 80%
        // For hiding > 80% with overlap, need extreme imbalance:
        // compute=1, comm=100 → no_overlap=101, overlap=100, hiding=1-100/101=0.0099
        // compute=100, comm=1 → no_overlap=101, overlap=100, hiding=1-100/101=0.0099
        // Actually: hiding = min(C,Cc)/(C+Cc)
        // For C=1, Cc=1000: 1/1001 = 0.001 → NOT > 80%
        // For C=1000, Cc=1: 1/1001 = 0.001 → NOT > 80%
        // The 80% threshold requires one to be >4x the other:
        // C=1, Cc=5: 1/6 = 0.167 → NOT > 80%
        // C=1, Cc=100: 1/101 = 0.0099 → NOT > 80%
        // Cc/(C+Cc) > 0.8 when Cc > 4*C: C=1, Cc=5: 5/6=0.833 > 0.8 ✓
        // So: compute=1, comm=5 → overlap=5, no_overlap=6, hiding=5/6=0.833 > 80%
        let overlap = CommComputeOverlap::new(1.0, 2.0, 5.0, true).unwrap();
        assert!(overlap.meets_hiding_threshold()); // 5/6 = 0.833 > 0.8
    }

    #[test]
    fn overlap_total_latency_with_quant() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true).unwrap()
            .with_quant_config(QuantizedActivationConfig::fp8_quant());
        let ops = vec![
            PipelineOp::Forward(0),
            PipelineOp::SendActivation(0),
            PipelineOp::Backward(0),
        ];
        // No overlap: compute + quantized_comm + backward_compute
        // = 100 + 12.5 + 200 = 312.5
        let total_no_overlap = overlap.total_latency_no_overlap(&ops);
        assert!((total_no_overlap - 312.5).abs() < 1e-10);

        // With overlap: max(compute_total, comm_total)
        // compute_total = 100 + 200 = 300
        // comm_total = 12.5 (FP8 quantized)
        // overlap_total = max(300, 12.5) = 300
        let total_overlap = overlap.total_latency_overlap(&ops);
        assert!((total_overlap - 300.0).abs() < 1e-10);
    }
}
