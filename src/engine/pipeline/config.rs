//! PipelineConfig — PP 配置与 stage 划分 (REQ-DIST-018)
//!
//! Pipeline Parallel 基础配置。每个 stage 持有连续层范围，
//! 层范围按 stage_id 均分：[stage_id * L/pp, (stage_id+1) * L/pp)。
//! pp_size == 1 时退化为单设备执行（零 PP 开销）。
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use crate::engine::distributed_config::{DistributedConfig, ParallelConfig};

// ── PipelineConfig (REQ-DIST-018) ─────────────────────────────────────────

/// Pipeline Parallel 配置 (REQ-DIST-018)
///
/// 封装 PP 维度配置，包括 stage 划分、虚拟 stage（交错 PP）和微批次大小。
/// 各 stage 持有连续层范围 `[stage_id * L/pp, (stage_id+1) * L/pp)`，
/// 无重叠且全覆盖。
///
/// `pp_size == 1` 时退化为单设备执行，`PipelineConfig` 的所有方法均
/// 返回"全范围"或"无 PP"语义，零 PP 开销。
// @trace REQ-DIST-018 [entity:PipelineConfig] [api:POST /internal/distributed/pipeline]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineConfig {
    /// Pipeline Parallel 维度，≥1。pp_size == 1 表示无 PP。
    pub pp_size: u32,
    /// 当前 stage ID，范围 [0, pp_size)。pp_size == 1 时为 0。
    pub stage_id: u32,
    /// 虚拟 stage 数量（交错 PP），≥1。默认 1 表示非交错 PP。
    pub num_virtual_stages: u32,
    /// 微批次大小，≥1。用于流水线填充时切分 batch。
    pub micro_batch_size: usize,
    /// 每 stage 层数（推导字段）。由 `from_distributed_config` 根据
    /// `num_layers / pp_size` 计算。非整除时向上取整保证全覆盖。
    pub layers_per_stage: u32,
}

// @trace REQ-DIST-018 [entity:PipelineConfig]
impl PipelineConfig {
    /// 从 DistributedConfig 推导 PipelineConfig (REQ-DIST-018 验收标准 5)
    ///
    /// 从 `ParallelConfig` 中读取 `pp_size` 和 `rank` 推导 `stage_id`：
    /// `stage_id = rank / (tp_size * cp_size)`。
    /// `layers_per_stage` 由调用方通过 `with_layers_per_stage` 设置。
    ///
    /// # Errors
    /// - `pp_size < 1`
    /// - `stage_id >= pp_size`
    // @trace REQ-DIST-018 [entity:PipelineConfig] [dataflow:DF-DIST-003]
    pub fn from_distributed_config(
        config: &DistributedConfig,
        num_layers: u32,
    ) -> Result<Self, PipelineConfigError> {
        let pp_size = config.parallel.pp_size;
        if pp_size < 1 {
            return Err(PipelineConfigError::InvalidPpSize(pp_size));
        }

        // stage_id 推导：rank / (tp_size * cp_size)
        // 同一 PP stage 内的所有 TP rank 和 CP rank 共享同一 stage_id
        let stage_id = if pp_size == 1 {
            0
        } else {
            config.parallel.rank / (config.parallel.tp_size * config.parallel.cp_size)
        };

        if stage_id >= pp_size {
            return Err(PipelineConfigError::InvalidStageId {
                stage_id,
                pp_size,
            });
        }

        let layers_per_stage = (num_layers + pp_size - 1) / pp_size; // ceil division

        Ok(Self {
            pp_size,
            stage_id,
            num_virtual_stages: 1,
            micro_batch_size: 1,
            layers_per_stage,
        })
    }

    /// 从 ParallelConfig 推导 PipelineConfig（简化入口，不依赖 DistributedConfig）
    ///
    /// 适用于仅持有 ParallelConfig 的场景。
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn from_parallel_config(
        config: &ParallelConfig,
        num_layers: u32,
    ) -> Result<Self, PipelineConfigError> {
        let pp_size = config.pp_size;
        if pp_size < 1 {
            return Err(PipelineConfigError::InvalidPpSize(pp_size));
        }

        let stage_id = if pp_size == 1 {
            0
        } else {
            config.rank / (config.tp_size * config.cp_size)
        };

        if stage_id >= pp_size {
            return Err(PipelineConfigError::InvalidStageId {
                stage_id,
                pp_size,
            });
        }

        let layers_per_stage = (num_layers + pp_size - 1) / pp_size;

        Ok(Self {
            pp_size,
            stage_id,
            num_virtual_stages: 1,
            micro_batch_size: 1,
            layers_per_stage,
        })
    }

    /// 返回本 stage 负责的层范围 `[start, end)` (REQ-DIST-018 验收标准 4)
    ///
    /// 层范围无重叠且全覆盖：所有 stage 的范围并集 = [0, num_layers)。
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn layer_range(&self, num_layers: u32) -> std::ops::Range<u32> {
        let start = self.stage_id * self.layers_per_stage;
        let end = ((self.stage_id + 1) * self.layers_per_stage).min(num_layers);
        start..end
    }

    /// 返回本 stage 负责的层起始索引
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn layer_start(&self) -> u32 {
        self.stage_id * self.layers_per_stage
    }

    /// 返回本 stage 负责的层结束索引（不含），不超过 num_layers
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn layer_end(&self, num_layers: u32) -> u32 {
        ((self.stage_id + 1) * self.layers_per_stage).min(num_layers)
    }

    /// 是否为 PP 模式（pp_size > 1）
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn is_pipeline_parallel(&self) -> bool {
        self.pp_size > 1
    }

    /// 是否为第一个 stage（stage_id == 0）
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }

    /// 是否为最后一个 stage（stage_id == pp_size - 1）
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn is_last_stage(&self) -> bool {
        self.stage_id == self.pp_size - 1
    }

    /// 设置微批次大小（builder pattern）
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn with_micro_batch_size(mut self, size: usize) -> Self {
        self.micro_batch_size = size.max(1);
        self
    }

    /// 设置虚拟 stage 数量（builder pattern，用于交错 PP）
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn with_virtual_stages(mut self, n: u32) -> Self {
        self.num_virtual_stages = n.max(1);
        self
    }

    /// 校验 PipelineConfig 一致性
    ///
    /// 返回 `true` 当且仅当：
    /// - pp_size >= 1
    /// - stage_id < pp_size
    /// - num_virtual_stages >= 1
    /// - micro_batch_size >= 1
    /// - layers_per_stage >= 1
    // @trace REQ-DIST-018 [entity:PipelineConfig]
    pub fn validate(&self) -> bool {
        self.pp_size >= 1
            && self.stage_id < self.pp_size
            && self.num_virtual_stages >= 1
            && self.micro_batch_size >= 1
            && self.layers_per_stage >= 1
    }
}

// ── PipelineConfigError (REQ-DIST-018) ────────────────────────────────────

/// PipelineConfig 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineConfigError {
    /// pp_size 无效（< 1）
    InvalidPpSize(u32),
    /// stage_id 超出范围 [0, pp_size)
    InvalidStageId { stage_id: u32, pp_size: u32 },
}

impl std::fmt::Display for PipelineConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineConfigError::InvalidPpSize(pp_size) => {
                write!(f, "PipelineConfig: invalid pp_size={pp_size}, must be >= 1")
            }
            PipelineConfigError::InvalidStageId { stage_id, pp_size } => {
                write!(
                    f,
                    "PipelineConfig: stage_id={stage_id} out of range [0, {pp_size})"
                )
            }
        }
    }
}

impl std::error::Error for PipelineConfigError {}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parallel_config(tp_size: u32, pp_size: u32, rank: u32, world_size: u32) -> ParallelConfig {
        let cp_size: u32 = 1;
        ParallelConfig {
            tp_size,
            pp_size,
            ep_size: 1,
            cp_size,
            rank,
            world_size,
            unique_id: String::new(),
            stage_id: rank / (tp_size * cp_size),
        }
    }

    fn make_distributed_config(parallel: ParallelConfig) -> DistributedConfig {
        DistributedConfig {
            parallel,
            ..Default::default()
        }
    }

    // ── PipelineConfig: construction ──

    #[test]
    fn from_distributed_config_pp1_single_device() {
        let parallel = make_parallel_config(1, 1, 0, 1);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.pp_size, 1);
        assert_eq!(cfg.stage_id, 0);
        assert_eq!(cfg.layers_per_stage, 32);
        assert!(!cfg.is_pipeline_parallel());
        assert!(cfg.is_first_stage());
        assert!(cfg.is_last_stage());
    }

    #[test]
    fn from_distributed_config_pp2_stage0() {
        // tp=1, pp=2, rank=0 → stage_id = 0 / (1*1) = 0
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.pp_size, 2);
        assert_eq!(cfg.stage_id, 0);
        assert_eq!(cfg.layers_per_stage, 16);
        assert!(cfg.is_pipeline_parallel());
        assert!(cfg.is_first_stage());
        assert!(!cfg.is_last_stage());
    }

    #[test]
    fn from_distributed_config_pp2_stage1() {
        // tp=1, pp=2, rank=1 → stage_id = 1 / (1*1) = 1
        let parallel = make_parallel_config(1, 2, 1, 2);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.pp_size, 2);
        assert_eq!(cfg.stage_id, 1);
        assert_eq!(cfg.layers_per_stage, 16);
        assert!(cfg.is_pipeline_parallel());
        assert!(!cfg.is_first_stage());
        assert!(cfg.is_last_stage());
    }

    #[test]
    fn from_distributed_config_tp2_pp2_stage0_rank0() {
        // tp=2, pp=2, rank=0 → stage_id = 0 / (2*1) = 0
        let parallel = make_parallel_config(2, 2, 0, 4);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.stage_id, 0);
    }

    #[test]
    fn from_distributed_config_tp2_pp2_stage1_rank2() {
        // tp=2, pp=2, rank=2 → stage_id = 2 / (2*1) = 1
        let parallel = make_parallel_config(2, 2, 2, 4);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.stage_id, 1);
    }

    #[test]
    fn from_distributed_config_tp2_pp2_stage1_rank3() {
        // tp=2, pp=2, rank=3 → stage_id = 3 / (2*1) = 1
        let parallel = make_parallel_config(2, 2, 3, 4);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.stage_id, 1);
    }

    // ── PipelineConfig: layer_range ──

    #[test]
    fn layer_range_pp2_32layers() {
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let cfg0 = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg0.layer_range(32), 0..16);

        let parallel = make_parallel_config(1, 2, 1, 2);
        let dist = make_distributed_config(parallel);
        let cfg1 = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg1.layer_range(32), 16..32);
    }

    #[test]
    fn layer_range_pp4_32layers() {
        // 32 / 4 = 8 layers per stage
        for stage in 0..4u32 {
            let parallel = make_parallel_config(1, 4, stage, 4);
            let dist = make_distributed_config(parallel);
            let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
            let range = cfg.layer_range(32);
            assert_eq!(range, stage * 8..(stage + 1) * 8);
        }
    }

    #[test]
    fn layer_range_non_divisible_33layers_pp4() {
        // 33 / 4 = ceil = 9 layers_per_stage
        // stage 0: [0, 9), stage 1: [9, 18), stage 2: [18, 27), stage 3: [27, 33)
        for stage in 0..4u32 {
            let parallel = make_parallel_config(1, 4, stage, 4);
            let dist = make_distributed_config(parallel);
            let cfg = PipelineConfig::from_distributed_config(&dist, 33).unwrap();
            let range = cfg.layer_range(33);
            let expected_end = ((stage + 1) * 9).min(33);
            assert_eq!(range, stage * 9..expected_end, "stage {stage}");
        }
    }

    #[test]
    fn layer_range_covers_all_layers_no_overlap() {
        // All stages' ranges union = [0, num_layers), no overlap
        let num_layers = 48u32;
        let pp_size = 4u32;
        let mut covered = vec![false; num_layers as usize];

        for stage in 0..pp_size {
            let parallel = make_parallel_config(1, pp_size, stage, pp_size);
            let dist = make_distributed_config(parallel);
            let cfg = PipelineConfig::from_distributed_config(&dist, num_layers).unwrap();
            for layer in cfg.layer_range(num_layers) {
                assert!(!covered[layer as usize], "layer {} covered by multiple stages", layer);
                covered[layer as usize] = true;
            }
        }

        assert!(covered.iter().all(|&c| c), "not all layers covered");
    }

    #[test]
    fn layer_range_pp1_covers_all() {
        let parallel = make_parallel_config(1, 1, 0, 1);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.layer_range(32), 0..32);
    }

    // ── PipelineConfig: validate ──

    #[test]
    fn validate_valid_config() {
        let cfg = PipelineConfig {
            pp_size: 2,
            stage_id: 1,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 16,
        };
        assert!(cfg.validate());
    }

    #[test]
    fn validate_pp1_stage0() {
        let cfg = PipelineConfig {
            pp_size: 1,
            stage_id: 0,
            num_virtual_stages: 1,
            micro_batch_size: 1,
            layers_per_stage: 32,
        };
        assert!(cfg.validate());
    }

    #[test]
    fn validate_stage_id_out_of_range() {
        let cfg = PipelineConfig {
            pp_size: 2,
            stage_id: 2, // >= pp_size
            num_virtual_stages: 1,
            micro_batch_size: 1,
            layers_per_stage: 16,
        };
        assert!(!cfg.validate());
    }

    #[test]
    fn validate_zero_micro_batch() {
        let cfg = PipelineConfig {
            pp_size: 1,
            stage_id: 0,
            num_virtual_stages: 1,
            micro_batch_size: 0,
            layers_per_stage: 32,
        };
        assert!(!cfg.validate());
    }

    // ── PipelineConfig: from_distributed_config errors ──

    #[test]
    fn from_distributed_config_stage_id_out_of_range() {
        // rank=5, tp=1, pp=2 → stage_id = 5/1 = 5 >= 2
        let parallel = make_parallel_config(1, 2, 5, 2);
        let dist = make_distributed_config(parallel);
        let result = PipelineConfig::from_distributed_config(&dist, 32);
        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineConfigError::InvalidStageId { stage_id, pp_size } => {
                assert_eq!(stage_id, 5);
                assert_eq!(pp_size, 2);
            }
            other => panic!("expected InvalidStageId, got {:?}", other),
        }
    }

    // ── PipelineConfig: from_parallel_config ──

    #[test]
    fn from_parallel_config_pp1() {
        let parallel = make_parallel_config(1, 1, 0, 1);
        let cfg = PipelineConfig::from_parallel_config(&parallel, 32).unwrap();
        assert_eq!(cfg.pp_size, 1);
        assert_eq!(cfg.stage_id, 0);
        assert_eq!(cfg.layers_per_stage, 32);
    }

    #[test]
    fn from_parallel_config_pp4() {
        let parallel = make_parallel_config(1, 4, 2, 4);
        let cfg = PipelineConfig::from_parallel_config(&parallel, 32).unwrap();
        assert_eq!(cfg.pp_size, 4);
        assert_eq!(cfg.stage_id, 2);
        assert_eq!(cfg.layers_per_stage, 8);
    }

    // ── PipelineConfig: builder methods ──

    #[test]
    fn with_micro_batch_size() {
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32)
            .unwrap()
            .with_micro_batch_size(8);
        assert_eq!(cfg.micro_batch_size, 8);
    }

    #[test]
    fn with_micro_batch_size_clamps_to_1() {
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32)
            .unwrap()
            .with_micro_batch_size(0);
        assert_eq!(cfg.micro_batch_size, 1);
    }

    #[test]
    fn with_virtual_stages() {
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32)
            .unwrap()
            .with_virtual_stages(4);
        assert_eq!(cfg.num_virtual_stages, 4);
    }

    #[test]
    fn with_virtual_stages_clamps_to_1() {
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32)
            .unwrap()
            .with_virtual_stages(0);
        assert_eq!(cfg.num_virtual_stages, 1);
    }

    // ── PipelineConfig: equality and clone ──

    #[test]
    fn equality_same_config() {
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let a = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        let b = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn inequality_different_stage() {
        let parallel0 = make_parallel_config(1, 2, 0, 2);
        let dist0 = make_distributed_config(parallel0);
        let a = PipelineConfig::from_distributed_config(&dist0, 32).unwrap();

        let parallel1 = make_parallel_config(1, 2, 1, 2);
        let dist1 = make_distributed_config(parallel1);
        let b = PipelineConfig::from_distributed_config(&dist1, 32).unwrap();

        assert_ne!(a, b);
    }

    #[test]
    fn clone_independence() {
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let mut cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        let cloned = cfg.clone();
        cfg.micro_batch_size = 99;
        assert_eq!(cloned.micro_batch_size, 1);
    }

    // ── PipelineConfigError: Display ──

    #[test]
    fn error_display_invalid_pp_size() {
        let err = PipelineConfigError::InvalidPpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("pp_size=0"));
        assert!(msg.contains("must be >= 1"));
    }

    #[test]
    fn error_display_invalid_stage_id() {
        let err = PipelineConfigError::InvalidStageId { stage_id: 3, pp_size: 2 };
        let msg = format!("{}", err);
        assert!(msg.contains("stage_id=3"));
        assert!(msg.contains("[0, 2)"));
    }

    // ── PipelineConfigError: std::error::Error ──

    #[test]
    fn error_is_std_error() {
        let err = PipelineConfigError::InvalidPpSize(0);
        let _: &dyn std::error::Error = &err;
    }

    // ── layer_start / layer_end ──

    #[test]
    fn layer_start_and_end() {
        let parallel = make_parallel_config(1, 4, 2, 4);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.layer_start(), 16);
        assert_eq!(cfg.layer_end(32), 24);
    }

    #[test]
    fn layer_end_capped_by_num_layers() {
        let parallel = make_parallel_config(1, 4, 3, 4);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 33).unwrap();
        // layers_per_stage = ceil(33/4) = 9
        // stage 3: start = 27, end = min(36, 33) = 33
        assert_eq!(cfg.layer_start(), 27);
        assert_eq!(cfg.layer_end(33), 33);
    }

    // ── CP-aware stage_id derivation ──

    #[test]
    fn stage_id_with_cp() {
        // tp=1, pp=2, cp=2, rank=2 → stage_id = 2 / (1*2) = 1
        let parallel = ParallelConfig {
            tp_size: 1,
            pp_size: 2,
            ep_size: 1,
            cp_size: 2,
            rank: 2,
            world_size: 4,
            unique_id: String::new(),
            stage_id: 2 / (1 * 2),
        };
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.stage_id, 1);
    }

    #[test]
    fn stage_id_with_cp_rank1_still_stage0() {
        // tp=1, pp=2, cp=2, rank=1 → stage_id = 1 / (1*2) = 0
        let parallel = ParallelConfig {
            tp_size: 1,
            pp_size: 2,
            ep_size: 1,
            cp_size: 2,
            rank: 1,
            world_size: 4,
            unique_id: String::new(),
            stage_id: 1 / (1 * 2),
        };
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();
        assert_eq!(cfg.stage_id, 0);
    }

    // ── Edge cases ──

    #[test]
    fn single_layer_pp2() {
        // 1 layer, pp=2 → layers_per_stage = ceil(1/2) = 1
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 1).unwrap();
        assert_eq!(cfg.layers_per_stage, 1);
        assert_eq!(cfg.layer_range(1), 0..1);

        let parallel = make_parallel_config(1, 2, 1, 2);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 1).unwrap();
        assert_eq!(cfg.layers_per_stage, 1);
        // stage 1: start=1, end=min(2,1)=1 → empty range (degenerate case)
        assert_eq!(cfg.layer_range(1), 1..1);
    }

    #[test]
    fn hash_consistency() {
        use std::collections::HashSet;
        let parallel = make_parallel_config(1, 2, 0, 2);
        let dist = make_distributed_config(parallel);
        let cfg = PipelineConfig::from_distributed_config(&dist, 32).unwrap();

        let mut set = HashSet::new();
        set.insert(cfg.clone());
        set.insert(cfg.clone());
        assert_eq!(set.len(), 1);
    }
}
