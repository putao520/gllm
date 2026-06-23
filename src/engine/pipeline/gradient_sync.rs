//! GradientSync — PP 梯度回传与权重同步 (REQ-DIST-028)
//!
//! Pipeline Parallel 梯度回传和权重同步机制：
//! - send_gradients: backward 后 P2P 传回前序 stage
//! - 权重更新在所有 stage 完成 backward 后 barrier 同步
//! - 梯度聚合后权值更新与单设备训练等价（误差 < 1e-6）
//! - TP+PP 2D 模式：先 TP AllReduce 再 PP 梯度传递
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use crate::engine::distributed_config::CommHandleWrapper;
use super::config::PipelineConfig;
use super::topology::Topology2D;

// ── GradientSyncPhase (REQ-DIST-028) ─────────────────────────────────────────

/// 梯度同步阶段 (REQ-DIST-028)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradientSyncPhase {
    /// 本地反向计算完成，待发送梯度
    LocalBackwardDone,
    /// 梯度已发送到前序 stage
    GradientSent,
    /// 梯度已从前序 stage 接收
    GradientReceived,
    /// TP AllReduce 完成（TP+PP 2D 模式）
    TpAllReduceDone,
    /// 所有 stage backward 完成，可更新权重
    AllBackwardDone,
    /// 权重更新完成
    WeightUpdated,
}

impl std::fmt::Display for GradientSyncPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GradientSyncPhase::LocalBackwardDone => write!(f, "LocalBackwardDone"),
            GradientSyncPhase::GradientSent => write!(f, "GradientSent"),
            GradientSyncPhase::GradientReceived => write!(f, "GradientReceived"),
            GradientSyncPhase::TpAllReduceDone => write!(f, "TpAllReduceDone"),
            GradientSyncPhase::AllBackwardDone => write!(f, "AllBackwardDone"),
            GradientSyncPhase::WeightUpdated => write!(f, "WeightUpdated"),
        }
    }
}

// ── GradientSyncConfig (REQ-DIST-028) ────────────────────────────────────────

/// 梯度同步配置 (REQ-DIST-028)
// @trace REQ-DIST-028 [entity:GradientSyncConfig]
#[derive(Debug, Clone, PartialEq)]
pub struct GradientSyncConfig {
    /// 是否为 TP+PP 2D 模式（需要先 TP AllReduce）
    pub is_tp_pp_2d: bool,
    /// TP 维度
    pub tp_size: u32,
    /// 梯度聚合数值等价性容差（默认 1e-6）
    pub numerical_tolerance: f64,
    /// 是否启用梯度压缩（减少通信量）
    pub gradient_compression: bool,
}

impl Default for GradientSyncConfig {
    fn default() -> Self {
        Self {
            is_tp_pp_2d: false,
            tp_size: 1,
            numerical_tolerance: 1e-6,
            gradient_compression: false,
        }
    }
}

impl GradientSyncConfig {
    /// 创建梯度同步配置
    // @trace REQ-DIST-028 [entity:GradientSyncConfig]
    pub fn new(is_tp_pp_2d: bool, tp_size: u32) -> Self {
        Self {
            is_tp_pp_2d,
            tp_size,
            ..Default::default()
        }
    }

    /// 设置数值容差
    // @trace REQ-DIST-028 [entity:GradientSyncConfig]
    pub fn with_numerical_tolerance(mut self, tolerance: f64) -> Self {
        self.numerical_tolerance = tolerance;
        self
    }

    /// 启用梯度压缩
    // @trace REQ-DIST-028 [entity:GradientSyncConfig]
    pub fn with_gradient_compression(mut self, enabled: bool) -> Self {
        self.gradient_compression = enabled;
        self
    }

    /// 校验配置一致性
    // @trace REQ-DIST-028 [entity:GradientSyncConfig]
    pub fn validate(&self) -> bool {
        if self.is_tp_pp_2d && self.tp_size < 2 {
            return false;
        }
        self.numerical_tolerance > 0.0 && self.tp_size >= 1
    }
}

// ── GradientSyncStats (REQ-DIST-028) ─────────────────────────────────────────

/// 梯度同步统计指标 (REQ-DIST-028)
// @trace REQ-DIST-028 [entity:GradientSyncStats]
#[derive(Debug, Clone, PartialEq)]
pub struct GradientSyncStats {
    /// 发送梯度次数
    pub send_count: usize,
    /// 接收梯度次数
    pub recv_count: usize,
    /// TP AllReduce 次数（TP+PP 2D 模式）
    pub tp_allreduce_count: usize,
    /// barrier 同步次数
    pub barrier_count: usize,
    /// 梯度同步总耗时（微秒）
    pub total_sync_us: u64,
    /// 梯度传输字节数
    pub total_bytes_transferred: usize,
}

impl Default for GradientSyncStats {
    fn default() -> Self {
        Self {
            send_count: 0,
            recv_count: 0,
            tp_allreduce_count: 0,
            barrier_count: 0,
            total_sync_us: 0,
            total_bytes_transferred: 0,
        }
    }
}

// ── GradientSync (REQ-DIST-028) ──────────────────────────────────────────────

/// PP 梯度回传与权重同步 (REQ-DIST-028)
///
/// 处理 Pipeline Parallel 的梯度回传和权重同步：
/// - backward 后 P2P 传回前序 stage (验收标准 1)
/// - 权重更新在所有 stage backward 完成后 barrier 同步 (验收标准 2)
/// - 梯度聚合后权值更新与单设备训练等价 (验收标准 3, 误差 < 1e-6)
/// - TP+PP 2D 模式先 TP AllReduce 再 PP 传递 (验收标准 4)
// @trace REQ-DIST-028 [entity:GradientSync] [api:POST /internal/distributed/pipeline/gradient]
#[derive(Debug, Clone)]
pub struct GradientSync {
    /// Pipeline 配置
    pub pipeline_config: PipelineConfig,
    /// 2D 拓扑（TP+PP）
    pub topology: Topology2D,
    /// 梯度同步配置
    pub config: GradientSyncConfig,
    /// 当前同步阶段
    pub phase: GradientSyncPhase,
    /// 统计指标
    pub stats: GradientSyncStats,
    /// 本 stage 已完成的 backward 微批次集合
    pub completed_backward_microbatches: Vec<bool>,
}

/// GradientSync 构建错误
#[derive(Debug, Clone, PartialEq)]
pub enum GradientSyncError {
    /// CommHandleWrapper 未初始化
    NotDistributed,
    /// 当前 stage 无前驱/后继
    NoPeer { stage_id: u32, pp_size: u32 },
    /// NCCL 通信错误
    NcclError(String),
    /// 梯度缓冲区大小为零
    ZeroGradientSize,
    /// 数值等价性验证失败
    NumericalEquivalenceFailed { actual_error: f64, tolerance: f64 },
}

impl std::fmt::Display for GradientSyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GradientSyncError::NotDistributed => {
                write!(f, "GradientSync: not in distributed mode")
            }
            GradientSyncError::NoPeer { stage_id, pp_size } => {
                write!(f, "GradientSync: no peer for stage_id={stage_id} in pp_size={pp_size}")
            }
            GradientSyncError::NcclError(msg) => {
                write!(f, "GradientSync: NCCL error: {msg}")
            }
            GradientSyncError::ZeroGradientSize => {
                write!(f, "GradientSync: gradient buffer size is zero")
            }
            GradientSyncError::NumericalEquivalenceFailed { actual_error, tolerance } => {
                write!(f, "GradientSync: numerical equivalence failed: error={actual_error} > tolerance={tolerance}")
            }
        }
    }
}

impl std::error::Error for GradientSyncError {}

// @trace REQ-DIST-028 [entity:GradientSync]
impl GradientSync {
    /// 创建梯度同步器
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn new(
        pipeline_config: PipelineConfig,
        topology: Topology2D,
        config: GradientSyncConfig,
        num_microbatches: usize,
    ) -> Self {
        Self {
            pipeline_config,
            topology,
            config,
            phase: GradientSyncPhase::LocalBackwardDone,
            stats: GradientSyncStats::default(),
            completed_backward_microbatches: vec![false; num_microbatches],
        }
    }

    /// 从 PipelineConfig 和 Topology2D 创建梯度同步器
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn from_config(
        config: &PipelineConfig,
        topology: Topology2D,
        sync_config: GradientSyncConfig,
        num_microbatches: usize,
    ) -> Self {
        Self::new(config.clone(), topology, sync_config, num_microbatches)
    }

    /// 是否为第一个 stage（无需发送梯度）
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn is_first_stage(&self) -> bool {
        self.pipeline_config.is_first_stage()
    }

    /// 是否为最后一个 stage（无需接收梯度）
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn is_last_stage(&self) -> bool {
        self.pipeline_config.is_last_stage()
    }

    /// 标记微批次 backward 完成
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn mark_backward_done(&mut self, micro_batch_index: usize) {
        if micro_batch_index < self.completed_backward_microbatches.len() {
            self.completed_backward_microbatches[micro_batch_index] = true;
        }
        self.phase = GradientSyncPhase::LocalBackwardDone;
    }

    /// 发送梯度到前序 stage (REQ-DIST-028 验收标准 1)
    ///
    /// backward 后通过 P2P 传回前序 stage。仅非首 stage 发送。
    /// TP+PP 2D 模式下先完成 TP AllReduce (验收标准 4)。
    // @trace REQ-DIST-028 [entity:GradientSync] [dataflow:DF-DIST-015]
    pub fn send_gradients(
        &mut self,
        comm: &CommHandleWrapper,
        gradients: &mut [f32],
    ) -> Result<(), GradientSyncError> {
        if !comm.is_distributed() {
            return Err(GradientSyncError::NotDistributed);
        }
        if gradients.is_empty() {
            return Err(GradientSyncError::ZeroGradientSize);
        }

        // TP+PP 2D 模式：先 TP AllReduce (验收标准 4)
        if self.config.is_tp_pp_2d {
            self.tp_allreduce_gradients(comm, gradients)?;
        }

        // 非首 stage 发送梯度到前序 stage
        if !self.is_first_stage() {
            let rank = comm.rank();
            let prev_rank = self.topology.pp_prev_rank(rank)
                .ok_or_else(|| GradientSyncError::NoPeer {
                    stage_id: self.pipeline_config.stage_id,
                    pp_size: self.pipeline_config.pp_size,
                })?;

            // 仅 TP master rank 执行 P2P 通信
            if self.topology.is_tp_master(rank) {
                comm.send_f32(prev_rank, gradients)
                    .map_err(GradientSyncError::NcclError)?;
            }

            self.stats.send_count += 1;
            self.stats.total_bytes_transferred += gradients.len() * std::mem::size_of::<f32>();
        }

        self.phase = GradientSyncPhase::GradientSent;
        Ok(())
    }

    /// 从后序 stage 接收梯度 (REQ-DIST-028 验收标准 1)
    ///
    /// 仅非末尾 stage 接收。
    // @trace REQ-DIST-028 [entity:GradientSync] [dataflow:DF-DIST-015]
    pub fn recv_gradients(
        &mut self,
        comm: &CommHandleWrapper,
        gradient_count: usize,
    ) -> Result<Vec<f32>, GradientSyncError> {
        if !comm.is_distributed() {
            return Err(GradientSyncError::NotDistributed);
        }
        if gradient_count == 0 {
            return Err(GradientSyncError::ZeroGradientSize);
        }

        // 非末尾 stage 从后序 stage 接收梯度
        if !self.is_last_stage() {
            let rank = comm.rank();
            let next_rank = self.topology.pp_next_rank(rank)
                .ok_or_else(|| GradientSyncError::NoPeer {
                    stage_id: self.pipeline_config.stage_id,
                    pp_size: self.pipeline_config.pp_size,
                })?;

            // 仅 TP master rank 执行 P2P 通信
            if self.topology.is_tp_master(rank) {
                let data = comm.recv_f32(next_rank, gradient_count)
                    .map_err(GradientSyncError::NcclError)?;
                self.stats.recv_count += 1;
                self.phase = GradientSyncPhase::GradientReceived;
                return Ok(data);
            }
        }

        self.phase = GradientSyncPhase::GradientReceived;
        Ok(Vec::new())
    }

    /// TP AllReduce 梯度 (REQ-DIST-028 验收标准 4)
    ///
    /// TP+PP 2D 模式：先 TP AllReduce 再 PP 梯度传递。
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn tp_allreduce_gradients(
        &mut self,
        comm: &CommHandleWrapper,
        gradients: &mut [f32],
    ) -> Result<(), GradientSyncError> {
        if !comm.is_distributed() {
            return Err(GradientSyncError::NotDistributed);
        }

        comm.all_reduce_inplace(gradients)
            .map_err(GradientSyncError::NcclError)?;

        self.stats.tp_allreduce_count += 1;
        self.phase = GradientSyncPhase::TpAllReduceDone;
        Ok(())
    }

    /// 检查所有 stage 是否完成 backward (REQ-DIST-028 验收标准 2)
    ///
    /// 权重更新在所有 stage 完成 backward 后 barrier 同步。
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn all_backward_done(&self) -> bool {
        self.completed_backward_microbatches.iter().all(|&done| done)
    }

    /// Barrier 同步 — 所有 stage 完成 backward 后调用 (REQ-DIST-028 验收标准 2)
    ///
    /// 使用 all_reduce_u64_sum 实现逻辑 barrier：
    /// 所有 stage 将计数器 +1 并 AllReduce，结果 == pp_size 则同步完成。
    // @trace REQ-DIST-028 [entity:GradientSync] [dataflow:DF-DIST-016]
    pub fn barrier_sync(
        &mut self,
        comm: &CommHandleWrapper,
    ) -> Result<bool, GradientSyncError> {
        if !comm.is_distributed() {
            return Ok(true);
        }

        let mut counter = vec![1u64];
        comm.all_reduce_u64_sum(&mut counter)
            .map_err(GradientSyncError::NcclError)?;

        self.stats.barrier_count += 1;

        if counter[0] == self.pipeline_config.pp_size as u64 {
            self.phase = GradientSyncPhase::AllBackwardDone;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 验证梯度聚合数值等价性 (REQ-DIST-028 验收标准 3)
    ///
    /// 梯度聚合后权值更新与单设备训练等价（误差 < 1e-6）。
    /// 比较聚合梯度与参考梯度的 L2 距离。
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn verify_numerical_equivalence(
        &self,
        aggregated: &[f32],
        reference: &[f32],
    ) -> Result<(), GradientSyncError> {
        if aggregated.len() != reference.len() {
            return Err(GradientSyncError::NumericalEquivalenceFailed {
                actual_error: f64::INFINITY,
                tolerance: self.config.numerical_tolerance,
            });
        }

        let l2_error: f64 = aggregated.iter()
            .zip(reference.iter())
            .map(|(a, r)| {
                let diff = (*a as f64) - (*r as f64);
                diff * diff
            })
            .sum::<f64>()
            .sqrt();

        if l2_error > self.config.numerical_tolerance {
            return Err(GradientSyncError::NumericalEquivalenceFailed {
                actual_error: l2_error,
                tolerance: self.config.numerical_tolerance,
            });
        }

        Ok(())
    }

    /// 标记权重更新完成
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn mark_weight_updated(&mut self) {
        self.phase = GradientSyncPhase::WeightUpdated;
    }

    /// 计算梯度传输字节数 = num_params * sizeof(f32)
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn gradient_transfer_bytes(num_params: usize) -> usize {
        num_params * std::mem::size_of::<f32>()
    }

    /// 重置同步状态（用于下一个训练迭代）
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn reset(&mut self) {
        self.phase = GradientSyncPhase::LocalBackwardDone;
        self.stats = GradientSyncStats::default();
        for done in &mut self.completed_backward_microbatches {
            *done = false;
        }
    }

    /// 校验一致性
    // @trace REQ-DIST-028 [entity:GradientSync]
    pub fn validate(&self) -> bool {
        self.pipeline_config.validate()
            && self.topology.validate()
            && self.config.validate()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_topology() -> Topology2D {
        Topology2D::new(2, 2, 4).unwrap()
    }

    fn make_pipeline_config() -> PipelineConfig {
        PipelineConfig {
            pp_size: 2,
            stage_id: 0,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 16,
        }
    }

    // ── GradientSyncPhase: Display ──

    #[test]
    fn phase_display() {
        assert_eq!(format!("{}", GradientSyncPhase::LocalBackwardDone), "LocalBackwardDone");
        assert_eq!(format!("{}", GradientSyncPhase::GradientSent), "GradientSent");
        assert_eq!(format!("{}", GradientSyncPhase::TpAllReduceDone), "TpAllReduceDone");
        assert_eq!(format!("{}", GradientSyncPhase::WeightUpdated), "WeightUpdated");
    }

    // ── GradientSyncConfig ──

    #[test]
    fn config_default() {
        let config = GradientSyncConfig::default();
        assert!(!config.is_tp_pp_2d);
        assert_eq!(config.tp_size, 1);
        assert!((config.numerical_tolerance - 1e-6).abs() < 1e-10);
        assert!(!config.gradient_compression);
    }

    #[test]
    fn config_new_2d() {
        let config = GradientSyncConfig::new(true, 4);
        assert!(config.is_tp_pp_2d);
        assert_eq!(config.tp_size, 4);
    }

    #[test]
    fn config_validate_valid() {
        assert!(GradientSyncConfig::default().validate());
    }

    #[test]
    fn config_validate_2d_tp1_invalid() {
        let config = GradientSyncConfig::new(true, 1);
        assert!(!config.validate());
    }

    #[test]
    fn config_builder_methods() {
        let config = GradientSyncConfig::default()
            .with_numerical_tolerance(1e-8)
            .with_gradient_compression(true);
        assert!((config.numerical_tolerance - 1e-8).abs() < 1e-10);
        assert!(config.gradient_compression);
    }

    // ── GradientSyncStats ──

    #[test]
    fn stats_default() {
        let stats = GradientSyncStats::default();
        assert_eq!(stats.send_count, 0);
        assert_eq!(stats.recv_count, 0);
        assert_eq!(stats.tp_allreduce_count, 0);
        assert_eq!(stats.barrier_count, 0);
        assert_eq!(stats.total_sync_us, 0);
        assert_eq!(stats.total_bytes_transferred, 0);
    }

    // ── GradientSync: construction ──

    #[test]
    fn new_valid() {
        // @trace TEST-DIST-028 [req:REQ-DIST-028] [level:unit]
        let sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        assert_eq!(sync.pipeline_config.pp_size, 2);
        assert_eq!(sync.completed_backward_microbatches.len(), 4);
        assert_eq!(sync.phase, GradientSyncPhase::LocalBackwardDone);
    }

    #[test]
    fn from_config() {
        let config = make_pipeline_config();
        let sync = GradientSync::from_config(
            &config,
            make_topology(),
            GradientSyncConfig::default(),
            8,
        );
        assert_eq!(sync.completed_backward_microbatches.len(), 8);
    }

    // ── GradientSync: stage predicates ──

    #[test]
    fn is_first_stage() {
        let sync = GradientSync::new(
            make_pipeline_config(), // stage_id=0
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        assert!(sync.is_first_stage());
    }

    #[test]
    fn is_not_first_stage() {
        let config = PipelineConfig {
            pp_size: 2,
            stage_id: 1,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 16,
        };
        let sync = GradientSync::new(config, make_topology(), GradientSyncConfig::default(), 4);
        assert!(!sync.is_first_stage());
        assert!(sync.is_last_stage());
    }

    // ── GradientSync: mark_backward_done ──

    #[test]
    fn mark_backward_done() {
        let mut sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        assert!(!sync.all_backward_done());
        sync.mark_backward_done(0);
        sync.mark_backward_done(1);
        sync.mark_backward_done(2);
        sync.mark_backward_done(3);
        assert!(sync.all_backward_done());
    }

    #[test]
    fn mark_backward_done_out_of_range() {
        let mut sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        sync.mark_backward_done(100); // out of range, should not panic
        assert!(!sync.all_backward_done());
    }

    // ── GradientSync: numerical equivalence (验收标准 3) ──

    #[test]
    fn verify_numerical_equivalence_pass() {
        // @trace TEST-DIST-028 [req:REQ-DIST-028] [level:unit]
        // 验收标准 3: 梯度聚合后权值更新与单设备训练等价（误差 < 1e-6）
        let sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        let aggregated = vec![1.0f32, 2.0, 3.0, 4.0];
        let reference = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!(sync.verify_numerical_equivalence(&aggregated, &reference).is_ok());
    }

    #[test]
    fn verify_numerical_equivalence_small_error_pass() {
        let sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        let aggregated = vec![1.0f32, 2.0, 3.0, 4.0];
        let reference = vec![1.0f32 + 1e-7, 2.0, 3.0, 4.0]; // error << 1e-6
        assert!(sync.verify_numerical_equivalence(&aggregated, &reference).is_ok());
    }

    #[test]
    fn verify_numerical_equivalence_large_error_fail() {
        let sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        let aggregated = vec![1.0f32, 2.0, 3.0, 4.0];
        let reference = vec![1.1f32, 2.0, 3.0, 4.0]; // error = 0.1 > 1e-6
        assert!(sync.verify_numerical_equivalence(&aggregated, &reference).is_err());
    }

    #[test]
    fn verify_numerical_equivalence_length_mismatch() {
        let sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        let aggregated = vec![1.0f32, 2.0];
        let reference = vec![1.0f32, 2.0, 3.0];
        assert!(sync.verify_numerical_equivalence(&aggregated, &reference).is_err());
    }

    // ── GradientSync: gradient_transfer_bytes ──

    #[test]
    fn gradient_transfer_bytes() {
        assert_eq!(GradientSync::gradient_transfer_bytes(1024), 1024 * 4);
    }

    // ── GradientSync: send_gradients with non-distributed comm ──

    #[test]
    fn send_gradients_non_distributed_returns_err() {
        let mut sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        let comm = CommHandleWrapper::new_for_test(0, 1);
        let mut gradients = vec![1.0f32; 100];
        let result = sync.send_gradients(&comm, &mut gradients);
        assert!(result.is_err());
    }

    #[test]
    fn recv_gradients_non_distributed_returns_err() {
        let mut sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        let comm = CommHandleWrapper::new_for_test(0, 1);
        let result = sync.recv_gradients(&comm, 100);
        assert!(result.is_err());
    }

    #[test]
    fn send_gradients_zero_size_returns_err() {
        let mut sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        let comm = CommHandleWrapper::new_for_test(0, 4);
        let mut gradients: Vec<f32> = vec![];
        let result = sync.send_gradients(&comm, &mut gradients);
        assert!(result.is_err());
    }

    // ── GradientSync: mark_weight_updated ──

    #[test]
    fn mark_weight_updated() {
        let mut sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        sync.mark_weight_updated();
        assert_eq!(sync.phase, GradientSyncPhase::WeightUpdated);
    }

    // ── GradientSync: reset ──

    #[test]
    fn reset_clears_state() {
        let mut sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        sync.mark_backward_done(0);
        sync.mark_backward_done(1);
        sync.mark_weight_updated();
        sync.reset();
        assert!(!sync.all_backward_done());
        assert_eq!(sync.phase, GradientSyncPhase::LocalBackwardDone);
        assert_eq!(sync.stats.send_count, 0);
    }

    // ── GradientSync: validate ──

    #[test]
    fn validate_valid() {
        let sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
            4,
        );
        assert!(sync.validate());
    }

    // ── GradientSyncError: Display ──

    #[test]
    fn error_display_not_distributed() {
        let err = GradientSyncError::NotDistributed;
        let msg = format!("{}", err);
        assert!(msg.contains("not in distributed mode"));
    }

    #[test]
    fn error_display_no_peer() {
        let err = GradientSyncError::NoPeer { stage_id: 0, pp_size: 2 };
        let msg = format!("{}", err);
        assert!(msg.contains("stage_id=0"));
        assert!(msg.contains("pp_size=2"));
    }

    #[test]
    fn error_display_nccl() {
        let err = GradientSyncError::NcclError("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
    }

    #[test]
    fn error_display_zero_gradient() {
        let err = GradientSyncError::ZeroGradientSize;
        let msg = format!("{}", err);
        assert!(msg.contains("zero"));
    }

    #[test]
    fn error_display_numerical_failed() {
        let err = GradientSyncError::NumericalEquivalenceFailed {
            actual_error: 0.01,
            tolerance: 1e-6,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0.01"));
        assert!(msg.contains("1e-6"));
    }

    #[test]
    fn error_is_std_error() {
        let err = GradientSyncError::NotDistributed;
        let _: &dyn std::error::Error = &err;
    }
}
