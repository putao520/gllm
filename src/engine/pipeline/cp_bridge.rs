//! CpPipelineBridge — PP 与 Ring Attention Context Parallelism 协同 (REQ-DIST-032)
//!
//! PP stage 可进一步拆分为 CP 子段处理超长序列：
//! - configure(cp_size, ring_stride) 设置 CP 环参数
//! - PP stage 内按 cp_size 切分为 CP 子段
//! - CP 环 AllGather 通信仅限同 pp_rank 的 ranks
//! - 跨 stage 零 CP 通信
//! - 最大序列长度 = single_gpu_max_seq * pp_size * cp_size
//! - PP+CP 数值与单 GPU 长序列等价 (误差 < 1e-5)
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use crate::engine::distributed_config::CommHandleWrapper;
use super::config::PipelineConfig;

// ── CpBridgeConfig (REQ-DIST-032) ──────────────────────────────────────────

/// CP+PP 协同配置 (REQ-DIST-032)
// @trace REQ-DIST-032 [entity:CpBridgeConfig]
#[derive(Debug, Clone, PartialEq)]
pub struct CpBridgeConfig {
    /// CP 环大小（同 PP stage 内的 CP rank 数量）
    pub cp_size: u32,
    /// 当前 rank 在 CP 环中的位置
    pub cp_rank: u32,
    /// Ring 通信步幅（相邻 CP rank 之间的 rank 间隔）
    pub ring_stride: u32,
    /// 单 GPU 最大序列长度
    pub single_gpu_max_seq: usize,
}

impl Default for CpBridgeConfig {
    fn default() -> Self {
        Self {
            cp_size: 1,
            cp_rank: 0,
            ring_stride: 1,
            single_gpu_max_seq: 4096,
        }
    }
}

impl CpBridgeConfig {
    /// 创建 CP+PP 协同配置
    // @trace REQ-DIST-032 [entity:CpBridgeConfig]
    pub fn new(cp_size: u32, cp_rank: u32, ring_stride: u32, single_gpu_max_seq: usize) -> Self {
        Self {
            cp_size: cp_size.max(1),
            cp_rank,
            ring_stride: ring_stride.max(1),
            single_gpu_max_seq: single_gpu_max_seq.max(1),
        }
    }

    /// 校验配置一致性
    // @trace REQ-DIST-032 [entity:CpBridgeConfig]
    pub fn validate(&self) -> bool {
        self.cp_size >= 1
            && self.cp_rank < self.cp_size
            && self.ring_stride >= 1
            && self.single_gpu_max_seq >= 1
    }
}

// ── CpSubsegment (REQ-DIST-032) ──────────────────────────────────────────

/// PP stage 内的 CP 子段 (REQ-DIST-032 验收标准 2)
///
/// 每个 CP 子段处理 seq_len / cp_size 个 token。
// @trace REQ-DIST-032 [entity:CpSubsegment]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpSubsegment {
    /// CP 子段索引 [0, cp_size)
    pub segment_index: u32,
    /// 本子段处理的 token 数量
    pub local_seq_len: usize,
    /// 本子段在完整序列中的偏移
    pub seq_offset: usize,
}

// ── CpAllGatherGroup (REQ-DIST-032) ──────────────────────────────────────

/// CP AllGather 通信组 (REQ-DIST-032 验收标准 3)
///
/// CP 环 AllGather 通信仅限同 pp_rank 的 ranks，
/// 跨 PP stage 零 CP 通信。
// @trace REQ-DIST-032 [entity:CpAllGatherGroup]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpAllGatherGroup {
    /// PP stage ID（通信组限定在同 pp_rank 内）
    pub pp_stage_id: u32,
    /// 参与通信的 rank 列表
    pub member_ranks: Vec<u32>,
}

impl CpAllGatherGroup {
    /// 创建同 PP stage 的 AllGather 通信组
    // @trace REQ-DIST-032 [entity:CpAllGatherGroup]
    pub fn new(pp_stage_id: u32, tp_size: u32, cp_size: u32) -> Self {
        let member_ranks: Vec<u32> = (0..cp_size)
            .map(|cp_rank| pp_stage_id * tp_size * cp_size + cp_rank * tp_size..pp_stage_id * tp_size * cp_size + cp_rank * tp_size + tp_size)
            .flatten()
            .collect();
        Self {
            pp_stage_id,
            member_ranks,
        }
    }

    /// 创建简化的 AllGather 通信组（每 CP rank 一个 rank）
    // @trace REQ-DIST-032 [entity:CpAllGatherGroup]
    pub fn simple(pp_stage_id: u32, cp_size: u32, ring_stride: u32) -> Self {
        let member_ranks: Vec<u32> = (0..cp_size)
            .map(|cp_rank| pp_stage_id * cp_size * ring_stride + cp_rank * ring_stride)
            .collect();
        Self {
            pp_stage_id,
            member_ranks,
        }
    }

    /// 判断给定 rank 是否属于本通信组
    // @trace REQ-DIST-032 [entity:CpAllGatherGroup]
    pub fn contains_rank(&self, rank: u32) -> bool {
        self.member_ranks.contains(&rank)
    }
}

// ── CpPipelineBridgeError (REQ-DIST-032) ──────────────────────────────────

/// CpPipelineBridge 错误类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpPipelineBridgeError {
    /// CommHandleWrapper 未初始化
    NotDistributed,
    /// CP 配置无效
    InvalidConfig(String),
    /// CP size 不匹配
    CpSizeMismatch { expected: u32, actual: u32 },
    /// NCCL 通信错误
    NcclError(String),
    /// 跨 stage CP 通信被禁止
    CrossStageCpCommunication { from_stage: u32, to_stage: u32 },
}

impl std::fmt::Display for CpPipelineBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpPipelineBridgeError::NotDistributed => {
                write!(f, "CpPipelineBridge: not in distributed mode")
            }
            CpPipelineBridgeError::InvalidConfig(msg) => {
                write!(f, "CpPipelineBridge: invalid config: {msg}")
            }
            CpPipelineBridgeError::CpSizeMismatch { expected, actual } => {
                write!(f, "CpPipelineBridge: cp_size mismatch: expected={expected}, actual={actual}")
            }
            CpPipelineBridgeError::NcclError(msg) => {
                write!(f, "CpPipelineBridge: NCCL error: {msg}")
            }
            CpPipelineBridgeError::CrossStageCpCommunication { from_stage, to_stage } => {
                write!(f, "CpPipelineBridge: cross-stage CP communication forbidden: from_stage={from_stage}, to_stage={to_stage}")
            }
        }
    }
}

impl std::error::Error for CpPipelineBridgeError {}

// ── CpPipelineBridge (REQ-DIST-032) ──────────────────────────────────────────

/// PP 与 Ring Attention Context Parallelism 协同桥接器 (REQ-DIST-032)
///
/// PP stage 内进一步拆分为 CP 子段处理超长序列。
/// CP 环 AllGather 通信仅限同 pp_rank 的 ranks (验收标准 3)。
/// 跨 stage 零 CP 通信。
/// 最大序列长度 = single_gpu_max_seq * pp_size * cp_size (验收标准 4)。
// @trace REQ-DIST-032 [entity:CpPipelineBridge] [api:POST /internal/distributed/pipeline/cp-bridge]
#[derive(Debug, Clone)]
pub struct CpPipelineBridge {
    /// Pipeline 配置
    pub pipeline_config: PipelineConfig,
    /// CP+PP 协同配置
    pub cp_config: CpBridgeConfig,
    /// AllGather 通信组
    pub gather_group: CpAllGatherGroup,
}

// @trace REQ-DIST-032 [entity:CpPipelineBridge]
impl CpPipelineBridge {
    /// 创建 PP+CP 协同桥接器
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn new(pipeline_config: PipelineConfig, cp_config: CpBridgeConfig) -> Self {
        let gather_group = CpAllGatherGroup::simple(
            pipeline_config.stage_id,
            cp_config.cp_size,
            cp_config.ring_stride,
        );
        Self {
            pipeline_config,
            cp_config,
            gather_group,
        }
    }

    /// configure: 设置 CP 环参数 (REQ-DIST-032 验收标准 1)
    ///
    /// `cp_size` = CP 环大小，`ring_stride` = 相邻 CP rank 的 rank 间隔。
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn configure(&mut self, cp_size: u32, ring_stride: u32) {
        self.cp_config.cp_size = cp_size.max(1);
        self.cp_config.ring_stride = ring_stride.max(1);
        if self.cp_config.cp_rank >= self.cp_config.cp_size {
            self.cp_config.cp_rank = 0;
        }
        self.gather_group = CpAllGatherGroup::simple(
            self.pipeline_config.stage_id,
            self.cp_config.cp_size,
            self.cp_config.ring_stride,
        );
    }

    /// PP stage 内按 cp_size 切分为 CP 子段 (REQ-DIST-032 验收标准 2)
    ///
    /// 每个 CP 子段处理 `seq_len / cp_size` 个 token。
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn split_into_subsegments(&self, total_seq_len: usize) -> Vec<CpSubsegment> {
        let cp_size = self.cp_config.cp_size as usize;
        let shard = total_seq_len / cp_size;
        let remainder = total_seq_len % cp_size;

        (0..cp_size)
            .map(|i| {
                let local_len = shard + if i < remainder { 1 } else { 0 };
                let offset = shard * i + remainder.min(i);
                CpSubsegment {
                    segment_index: i as u32,
                    local_seq_len: local_len,
                    seq_offset: offset,
                }
            })
            .collect()
    }

    /// 获取本 CP rank 的子段 (REQ-DIST-032 验收标准 2)
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn local_subsegment(&self, total_seq_len: usize) -> CpSubsegment {
        let segments = self.split_into_subsegments(total_seq_len);
        segments[self.cp_config.cp_rank as usize].clone()
    }

    /// CP 环 AllGather 通信仅限同 pp_rank 的 ranks (REQ-DIST-032 验收标准 3)
    ///
    /// 跨 PP stage 零 CP 通信。
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn allgather_within_stage(
        &self,
        comm: &CommHandleWrapper,
        data: &[f32],
    ) -> Result<Vec<f32>, CpPipelineBridgeError> {
        if !comm.is_distributed() {
            return Err(CpPipelineBridgeError::NotDistributed);
        }

        let cp_size = self.cp_config.cp_size;
        if cp_size <= 1 {
            // cp_size == 1: 无 CP 通信，直接返回本地数据
            return Ok(data.to_vec());
        }

        // 验证通信组内所有 rank 都在同 PP stage
        let my_stage = self.pipeline_config.stage_id;
        for &rank in &self.gather_group.member_ranks {
            let target_stage = rank / (cp_size * self.cp_config.ring_stride);
            if target_stage != my_stage {
                return Err(CpPipelineBridgeError::CrossStageCpCommunication {
                    from_stage: my_stage,
                    to_stage: target_stage,
                });
            }
        }

        // Ring AllGather: cp_size 轮，每轮 send 到 next_rank, recv 从 prev_rank
        let mut gathered = vec![0.0f32; data.len() * cp_size as usize];
        // Place local data in the correct position
        let local_offset = self.cp_config.cp_rank as usize * data.len();
        gathered[local_offset..local_offset + data.len()].copy_from_slice(data);

        let mut current_send = data.to_vec();
        let mut current_recv = vec![0.0f32; data.len()];

        let next_rank = self.cp_ring_next();
        let prev_rank = self.cp_ring_prev();

        for step in 0..cp_size - 1 {
            // Send current chunk to next rank in CP ring
            comm.send_f32(next_rank, &current_send)
                .map_err(CpPipelineBridgeError::NcclError)?;

            // Receive chunk from prev rank in CP ring
            current_recv = comm.recv_f32(prev_rank, data.len())
                .map_err(CpPipelineBridgeError::NcclError)?;

            // Determine which cp_rank's data we just received
            let source_cp_rank = (self.cp_config.cp_size + self.cp_config.cp_rank - step - 1) % self.cp_config.cp_size;
            let dest_offset = source_cp_rank as usize * data.len();
            gathered[dest_offset..dest_offset + data.len()].copy_from_slice(&current_recv);

            current_send = current_recv.clone();
        }

        Ok(gathered)
    }

    /// 最大支持序列长度 = single_gpu_max_seq * pp_size * cp_size
    /// (REQ-DIST-032 验收标准 4)
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn max_supported_seq_len(&self) -> usize {
        self.cp_config.single_gpu_max_seq
            * self.pipeline_config.pp_size as usize
            * self.cp_config.cp_size as usize
    }

    /// PP+CP 数值等价校验 (REQ-DIST-032 验收标准 5)
    ///
    /// 误差 < 1e-5 时认为等价。
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn verify_numerical_equivalence(
        pp_cp_output: &[f32],
        single_gpu_output: &[f32],
        tolerance: f32,
    ) -> bool {
        if pp_cp_output.len() != single_gpu_output.len() {
            return false;
        }
        pp_cp_output.iter()
            .zip(single_gpu_output.iter())
            .all(|(&a, &b)| (a - b).abs() < tolerance)
    }

    /// CP 环中下一个 rank（同 PP stage 内）
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn cp_ring_next(&self) -> u32 {
        let next_cp_rank = (self.cp_config.cp_rank + 1) % self.cp_config.cp_size;
        self.cp_rank_to_global(next_cp_rank)
    }

    /// CP 环中上一个 rank（同 PP stage 内）
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn cp_ring_prev(&self) -> u32 {
        let prev_cp_rank = if self.cp_config.cp_rank == 0 {
            self.cp_config.cp_size - 1
        } else {
            self.cp_config.cp_rank - 1
        };
        self.cp_rank_to_global(prev_cp_rank)
    }

    /// CP rank 到全局 rank 的映射
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn cp_rank_to_global(&self, cp_rank: u32) -> u32 {
        self.pipeline_config.stage_id * self.cp_config.cp_size * self.cp_config.ring_stride
            + cp_rank * self.cp_config.ring_stride
    }

    /// 全局 rank 到 CP rank 的映射
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn global_to_cp_rank(&self, global_rank: u32) -> u32 {
        let base = self.pipeline_config.stage_id * self.cp_config.cp_size * self.cp_config.ring_stride;
        (global_rank - base) / self.cp_config.ring_stride
    }

    /// 是否启用 CP
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn is_cp_enabled(&self) -> bool {
        self.cp_config.cp_size > 1
    }

    /// 跨 stage CP 通信是否为零 (REQ-DIST-032 验收标准 3)
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn cross_stage_cp_zero(&self) -> bool {
        // CP 环 next/prev 都映射到同 PP stage → 跨 stage 零通信
        let next = self.cp_ring_next();
        let prev = self.cp_ring_prev();
        let my_stage = self.pipeline_config.stage_id;
        let next_stage = self.global_to_pp_stage(next);
        let prev_stage = self.global_to_pp_stage(prev);
        next_stage == my_stage && prev_stage == my_stage
    }

    /// 全局 rank 到 PP stage 的映射
    fn global_to_pp_stage(&self, global_rank: u32) -> u32 {
        global_rank / (self.cp_config.cp_size * self.cp_config.ring_stride)
    }

    /// 校验一致性
    // @trace REQ-DIST-032 [entity:CpPipelineBridge]
    pub fn validate(&self) -> bool {
        self.pipeline_config.validate()
            && self.cp_config.validate()
            && self.cross_stage_cp_zero()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pipeline_config() -> PipelineConfig {
        PipelineConfig {
            pp_size: 2,
            stage_id: 0,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 16,
        }
    }

    fn make_cp_config(cp_size: u32, cp_rank: u32) -> CpBridgeConfig {
        CpBridgeConfig::new(cp_size, cp_rank, 1, 4096)
    }

    // ── CpBridgeConfig ──

    #[test]
    fn cp_bridge_config_default() {
        let config = CpBridgeConfig::default();
        assert_eq!(config.cp_size, 1);
        assert_eq!(config.cp_rank, 0);
        assert_eq!(config.ring_stride, 1);
        assert!(config.validate());
    }

    #[test]
    fn cp_bridge_config_new() {
        let config = CpBridgeConfig::new(4, 2, 1, 8192);
        assert_eq!(config.cp_size, 4);
        assert_eq!(config.cp_rank, 2);
        assert_eq!(config.ring_stride, 1);
        assert_eq!(config.single_gpu_max_seq, 8192);
    }

    #[test]
    fn cp_bridge_config_cp_size_clamped() {
        let config = CpBridgeConfig::new(0, 0, 1, 4096);
        assert_eq!(config.cp_size, 1);
    }

    #[test]
    fn cp_bridge_config_validate_cp_rank_out_of_range() {
        let config = CpBridgeConfig::new(4, 5, 1, 4096);
        assert!(!config.validate());
    }

    // ── CpSubsegment ──

    #[test]
    fn cp_subsegment_fields() {
        let seg = CpSubsegment {
            segment_index: 2,
            local_seq_len: 256,
            seq_offset: 512,
        };
        assert_eq!(seg.segment_index, 2);
        assert_eq!(seg.local_seq_len, 256);
        assert_eq!(seg.seq_offset, 512);
    }

    // ── CpAllGatherGroup ──

    #[test]
    fn allgather_group_simple() {
        // @trace TEST-DIST-032 [req:REQ-DIST-032] [level:unit]
        // 验收标准 3: CP 环 AllGather 通信仅限同 pp_rank 的 ranks
        let group = CpAllGatherGroup::simple(0, 4, 1);
        assert_eq!(group.pp_stage_id, 0);
        assert_eq!(group.member_ranks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn allgather_group_simple_with_stride() {
        let group = CpAllGatherGroup::simple(0, 2, 4);
        assert_eq!(group.member_ranks, vec![0, 4]);
    }

    #[test]
    fn allgather_group_contains_rank() {
        let group = CpAllGatherGroup::simple(0, 4, 1);
        assert!(group.contains_rank(0));
        assert!(group.contains_rank(3));
        assert!(!group.contains_rank(4));
    }

    #[test]
    fn allgather_group_different_stages() {
        // Stage 1 with cp_size=2, stride=1
        // base = 1 * 2 * 1 = 2 → ranks [2, 3]
        let group = CpAllGatherGroup::simple(1, 2, 1);
        assert_eq!(group.pp_stage_id, 1);
        assert_eq!(group.member_ranks, vec![2, 3]);
    }

    // ── CpPipelineBridge: construction ──

    #[test]
    fn bridge_new() {
        // @trace TEST-DIST-032 [req:REQ-DIST-032] [level:unit]
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            make_cp_config(4, 0),
        );
        assert_eq!(bridge.cp_config.cp_size, 4);
        assert!(bridge.is_cp_enabled());
    }

    #[test]
    fn bridge_no_cp() {
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            make_cp_config(1, 0),
        );
        assert!(!bridge.is_cp_enabled());
    }

    // ── configure (验收标准 1) ──

    #[test]
    fn bridge_configure() {
        // @trace TEST-DIST-032 [req:REQ-DIST-032] [level:unit]
        // 验收标准 1: configure(cp_size, ring_stride) 设置 CP 环参数
        let mut bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            make_cp_config(1, 0),
        );
        assert!(!bridge.is_cp_enabled());
        bridge.configure(4, 1);
        assert!(bridge.is_cp_enabled());
        assert_eq!(bridge.cp_config.cp_size, 4);
        assert_eq!(bridge.cp_config.ring_stride, 1);
    }

    #[test]
    fn bridge_configure_clamps_cp_rank() {
        let mut bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            CpBridgeConfig::new(4, 3, 1, 4096),
        );
        bridge.configure(2, 1);
        assert_eq!(bridge.cp_config.cp_rank, 0); // clamped from 3
    }

    // ── split_into_subsegments (验收标准 2) ──

    #[test]
    fn bridge_split_even() {
        // @trace TEST-DIST-032 [req:REQ-DIST-032] [level:unit]
        // 验收标准 2: PP stage 内按 cp_size 切分为 CP 子段
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            make_cp_config(4, 0),
        );
        let segments = bridge.split_into_subsegments(100);
        assert_eq!(segments.len(), 4);
        assert_eq!(segments[0].local_seq_len, 25);
        assert_eq!(segments[1].local_seq_len, 25);
        assert_eq!(segments[0].seq_offset, 0);
        assert_eq!(segments[1].seq_offset, 25);
    }

    #[test]
    fn bridge_split_uneven() {
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            make_cp_config(4, 0),
        );
        let segments = bridge.split_into_subsegments(103);
        // 103 / 4 = 25 remainder 3
        assert_eq!(segments[0].local_seq_len, 26);
        assert_eq!(segments[1].local_seq_len, 26);
        assert_eq!(segments[2].local_seq_len, 26);
        assert_eq!(segments[3].local_seq_len, 25);
        // Verify all offsets and total sum
        let total: usize = segments.iter().map(|s| s.local_seq_len).sum();
        assert_eq!(total, 103);
    }

    #[test]
    fn bridge_local_subsegment() {
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            make_cp_config(4, 2),
        );
        let seg = bridge.local_subsegment(100);
        assert_eq!(seg.segment_index, 2);
        assert_eq!(seg.local_seq_len, 25);
        assert_eq!(seg.seq_offset, 50);
    }

    // ── max_supported_seq_len (验收标准 4) ──

    #[test]
    fn bridge_max_seq_len() {
        // @trace TEST-DIST-032 [req:REQ-DIST-032] [level:unit]
        // 验收标准 4: 最大序列长度 = single_gpu_max_seq * pp_size * cp_size
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 4,
                stage_id: 0,
                num_virtual_stages: 1,
                micro_batch_size: 1,
                layers_per_stage: 8,
            },
            CpBridgeConfig::new(8, 0, 1, 4096),
        );
        assert_eq!(bridge.max_supported_seq_len(), 4096 * 4 * 8);
    }

    // ── cross_stage_cp_zero (验收标准 3) ──

    #[test]
    fn bridge_cross_stage_zero() {
        // @trace TEST-DIST-032 [req:REQ-DIST-032] [level:unit]
        // 验收标准 3: 跨 stage 零 CP 通信
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 2,
                stage_id: 0,
                num_virtual_stages: 1,
                micro_batch_size: 1,
                layers_per_stage: 16,
            },
            CpBridgeConfig::new(2, 0, 1, 4096),
        );
        assert!(bridge.cross_stage_cp_zero());
    }

    #[test]
    fn bridge_cross_stage_zero_stage1() {
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 2,
                stage_id: 1,
                num_virtual_stages: 1,
                micro_batch_size: 1,
                layers_per_stage: 16,
            },
            CpBridgeConfig::new(2, 0, 1, 4096),
        );
        assert!(bridge.cross_stage_cp_zero());
    }

    // ── verify_numerical_equivalence (验收标准 5) ──

    #[test]
    fn bridge_numerical_equivalence_pass() {
        // @trace TEST-DIST-032 [req:REQ-DIST-032] [level:unit]
        // 验收标准 5: PP+CP 数值与单 GPU 长序列等价 (误差 < 1e-5)
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!(CpPipelineBridge::verify_numerical_equivalence(&a, &b, 1e-5));
    }

    #[test]
    fn bridge_numerical_equivalence_small_error() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6];
        assert!(CpPipelineBridge::verify_numerical_equivalence(&a, &b, 1e-5));
    }

    #[test]
    fn bridge_numerical_equivalence_large_error() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.1f32, 2.0, 3.0];
        assert!(!CpPipelineBridge::verify_numerical_equivalence(&a, &b, 1e-5));
    }

    #[test]
    fn bridge_numerical_equivalence_different_len() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        assert!(!CpPipelineBridge::verify_numerical_equivalence(&a, &b, 1e-5));
    }

    // ── CP ring navigation ──

    #[test]
    fn bridge_cp_ring_next_prev() {
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 2,
                stage_id: 0,
                num_virtual_stages: 1,
                micro_batch_size: 1,
                layers_per_stage: 16,
            },
            CpBridgeConfig::new(4, 1, 1, 4096),
        );
        // cp_rank 1, next → cp_rank 2 → global 2
        assert_eq!(bridge.cp_ring_next(), 2);
        // cp_rank 1, prev → cp_rank 0 → global 0
        assert_eq!(bridge.cp_ring_prev(), 0);
    }

    #[test]
    fn bridge_cp_ring_wraps() {
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 2,
                stage_id: 0,
                num_virtual_stages: 1,
                micro_batch_size: 1,
                layers_per_stage: 16,
            },
            CpBridgeConfig::new(4, 3, 1, 4096),
        );
        // cp_rank 3, next → cp_rank 0 → global 0
        assert_eq!(bridge.cp_ring_next(), 0);
        // cp_rank 3, prev → cp_rank 2 → global 2
        assert_eq!(bridge.cp_ring_prev(), 2);
    }

    #[test]
    fn bridge_cp_rank_to_global_stage1() {
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 2,
                stage_id: 1,
                num_virtual_stages: 1,
                micro_batch_size: 1,
                layers_per_stage: 16,
            },
            CpBridgeConfig::new(4, 0, 1, 4096),
        );
        // stage 1, cp_rank 0 → global = 1*4*1 + 0*1 = 4
        assert_eq!(bridge.cp_rank_to_global(0), 4);
        assert_eq!(bridge.cp_rank_to_global(3), 7);
    }

    #[test]
    fn bridge_global_to_cp_rank() {
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 2,
                stage_id: 0,
                num_virtual_stages: 1,
                micro_batch_size: 1,
                layers_per_stage: 16,
            },
            CpBridgeConfig::new(4, 0, 1, 4096),
        );
        assert_eq!(bridge.global_to_cp_rank(0), 0);
        assert_eq!(bridge.global_to_cp_rank(3), 3);
    }

    // ── allgather_within_stage ──

    #[test]
    fn bridge_allgather_not_distributed() {
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            make_cp_config(4, 0),
        );
        let comm = CommHandleWrapper::new_for_test(0, 1);
        let data = vec![1.0f32, 2.0, 3.0];
        let result = bridge.allgather_within_stage(&comm, &data);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CpPipelineBridgeError::NotDistributed);
    }

    #[test]
    fn bridge_allgather_cp_size_1_noop() {
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            make_cp_config(1, 0),
        );
        let comm = CommHandleWrapper::new_for_test(0, 4);
        let data = vec![1.0f32, 2.0, 3.0];
        let result = bridge.allgather_within_stage(&comm, &data).unwrap();
        assert_eq!(result, data);
    }

    // ── validate ──

    #[test]
    fn bridge_validate_valid() {
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            make_cp_config(2, 0),
        );
        assert!(bridge.validate());
    }

    #[test]
    fn bridge_validate_invalid_cp_rank() {
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            CpBridgeConfig::new(2, 5, 1, 4096), // cp_rank >= cp_size
        );
        assert!(!bridge.validate());
    }

    // ── CpPipelineBridgeError: Display ──

    #[test]
    fn error_display_not_distributed() {
        let err = CpPipelineBridgeError::NotDistributed;
        let msg = format!("{}", err);
        assert!(msg.contains("not in distributed mode"));
    }

    #[test]
    fn error_display_invalid_config() {
        let err = CpPipelineBridgeError::InvalidConfig("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
    }

    #[test]
    fn error_display_cp_size_mismatch() {
        let err = CpPipelineBridgeError::CpSizeMismatch { expected: 4, actual: 2 };
        let msg = format!("{}", err);
        assert!(msg.contains("expected=4"));
        assert!(msg.contains("actual=2"));
    }

    #[test]
    fn error_display_nccl() {
        let err = CpPipelineBridgeError::NcclError("fail".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("fail"));
    }

    #[test]
    fn error_display_cross_stage() {
        let err = CpPipelineBridgeError::CrossStageCpCommunication {
            from_stage: 0,
            to_stage: 1,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("from_stage=0"));
        assert!(msg.contains("to_stage=1"));
    }

    #[test]
    fn error_is_std_error() {
        let err = CpPipelineBridgeError::NotDistributed;
        let _: &dyn std::error::Error = &err;
    }
}
