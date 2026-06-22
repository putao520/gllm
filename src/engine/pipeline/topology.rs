//! Topology2D — 2D 并行 (TP+PP) 拓扑与通信编排 (REQ-DIST-029)
//!
//! world_size = tp_size * pp_size，rank 按 (tp_rank, pp_rank) 二维映射。
//! 同一 PP stage 内的 TP 组做 AllReduce，跨 PP stage 的 P2P 通信在
//! TP 主 rank（tp_rank=0）间进行。
//!
//! nccl feature-gated: 非 nccl 构建零影响。

// ── Topology2D (REQ-DIST-029) ─────────────────────────────────────────────

/// 2D 并行 (TP+PP) 拓扑 (REQ-DIST-029)
///
/// 将全局 rank 映射到 (tp_rank, pp_rank) 二维坐标，支持：
/// - TP 组内 AllReduce（同 pp_rank 的所有 tp_rank）
/// - PP 组 P2P 通信（同 tp_rank 的所有 pp_rank，主 rank tp_rank=0 间直接通信）
///
/// rank 映射公式：`rank(tp_rank, pp_rank) = tp_rank * pp_size + pp_rank`
/// 逆映射：`tp_rank = rank / pp_size`，`pp_rank = rank % pp_size`
// @trace REQ-DIST-029 [entity:Topology2D] [api:POST /internal/distributed/topology]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Topology2D {
    /// Tensor Parallel 维度
    pub tp_size: u32,
    /// Pipeline Parallel 维度
    pub pp_size: u32,
    /// 全局 world_size = tp_size * pp_size
    pub world_size: u32,
}

/// Topology2D 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Topology2DError {
    /// tp_size * pp_size != world_size
    DimensionMismatch { tp_size: u32, pp_size: u32, world_size: u32 },
    /// tp_size < 1
    InvalidTpSize(u32),
    /// pp_size < 1
    InvalidPpSize(u32),
}

impl std::fmt::Display for Topology2DError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Topology2DError::DimensionMismatch { tp_size, pp_size, world_size } => {
                write!(
                    f,
                    "Topology2D: tp_size({tp_size}) * pp_size({pp_size}) = {} != world_size({world_size})",
                    tp_size * pp_size
                )
            }
            Topology2DError::InvalidTpSize(tp_size) => {
                write!(f, "Topology2D: invalid tp_size={tp_size}, must be >= 1")
            }
            Topology2DError::InvalidPpSize(pp_size) => {
                write!(f, "Topology2D: invalid pp_size={pp_size}, must be >= 1")
            }
        }
    }
}

impl std::error::Error for Topology2DError {}

// @trace REQ-DIST-029 [entity:Topology2D]
impl Topology2D {
    /// 创建 2D 并行拓扑 (REQ-DIST-029 验收标准 1)
    ///
    /// 校验 `tp_size * pp_size == world_size`。
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn new(tp_size: u32, pp_size: u32, world_size: u32) -> Result<Self, Topology2DError> {
        if tp_size < 1 {
            return Err(Topology2DError::InvalidTpSize(tp_size));
        }
        if pp_size < 1 {
            return Err(Topology2DError::InvalidPpSize(pp_size));
        }
        if tp_size * pp_size != world_size {
            return Err(Topology2DError::DimensionMismatch {
                tp_size,
                pp_size,
                world_size,
            });
        }
        Ok(Self {
            tp_size,
            pp_size,
            world_size,
        })
    }

    /// 从 ParallelConfig 创建拓扑
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn from_parallel_config(
        config: &crate::engine::distributed_config::ParallelConfig,
    ) -> Result<Self, Topology2DError> {
        // 2D 拓扑仅考虑 TP+PP，EP 维度不在 Topology2D 范围内
        Self::new(config.tp_size, config.pp_size, config.tp_size * config.pp_size)
    }

    /// rank → (tp_rank, pp_rank) 逆映射 (REQ-DIST-029 验收标准 2)
    ///
    /// `tp_rank = rank / pp_size`，`pp_rank = rank % pp_size`
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn decompose_rank(&self, rank: u32) -> (u32, u32) {
        let tp_rank = rank / self.pp_size;
        let pp_rank = rank % self.pp_size;
        (tp_rank, pp_rank)
    }

    /// (tp_rank, pp_rank) → rank 正映射 (REQ-DIST-029 验收标准 2)
    ///
    /// `rank = tp_rank * pp_size + pp_rank`
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn compose_rank(&self, tp_rank: u32, pp_rank: u32) -> u32 {
        tp_rank * self.pp_size + pp_rank
    }

    /// 返回指定 rank 所属 TP 组的所有 rank (REQ-DIST-029 验收标准 3, 4)
    ///
    /// TP 组 = 同 pp_rank 的所有 tp_rank。
    /// TP 组内 AllReduce 通信量 = hidden_dim * dtype_bytes。
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn tp_group(&self, rank: u32) -> Vec<u32> {
        let (_, pp_rank) = self.decompose_rank(rank);
        (0..self.tp_size)
            .map(|tp_rank| self.compose_rank(tp_rank, pp_rank))
            .collect()
    }

    /// 返回指定 rank 所属 PP 组的所有 rank (REQ-DIST-029 验收标准 3, 5)
    ///
    /// PP 组 = 同 tp_rank 的所有 pp_rank。
    /// P2P 通信在 tp_rank=0 的 ranks 间进行。
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn pp_group(&self, rank: u32) -> Vec<u32> {
        let (tp_rank, _) = self.decompose_rank(rank);
        (0..self.pp_size)
            .map(|pp_rank| self.compose_rank(tp_rank, pp_rank))
            .collect()
    }

    /// 返回 PP 组中 tp_rank=0 的 ranks（P2P 通信端点）(REQ-DIST-029 验收标准 5)
    ///
    /// PP P2P 通信仅在这些 rank 间进行，其他 tp_rank 由 TP broadcast 同步。
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn pp_p2p_endpoints(&self) -> Vec<u32> {
        (0..self.pp_size)
            .map(|pp_rank| self.compose_rank(0, pp_rank))
            .collect()
    }

    /// 返回指定 rank 的 PP 前驱 rank（前一个 stage 的同 tp_rank rank）
    ///
    /// 第一个 stage (pp_rank=0) 返回 None。
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn pp_prev_rank(&self, rank: u32) -> Option<u32> {
        let (tp_rank, pp_rank) = self.decompose_rank(rank);
        if pp_rank == 0 {
            None
        } else {
            Some(self.compose_rank(tp_rank, pp_rank - 1))
        }
    }

    /// 返回指定 rank 的 PP 后继 rank（后一个 stage 的同 tp_rank rank）
    ///
    /// 最后一个 stage (pp_rank=pp_size-1) 返回 None。
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn pp_next_rank(&self, rank: u32) -> Option<u32> {
        let (tp_rank, pp_rank) = self.decompose_rank(rank);
        if pp_rank + 1 >= self.pp_size {
            None
        } else {
            Some(self.compose_rank(tp_rank, pp_rank + 1))
        }
    }

    /// 是否为 TP 主 rank（tp_rank=0）
    ///
    /// PP P2P 通信仅在 tp_rank=0 的 rank 间直接进行。
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn is_tp_master(&self, rank: u32) -> bool {
        let (tp_rank, _) = self.decompose_rank(rank);
        tp_rank == 0
    }

    /// 是否为纯 TP 模式（pp_size == 1）
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn is_tp_only(&self) -> bool {
        self.pp_size == 1
    }

    /// 是否为纯 PP 模式（tp_size == 1）
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn is_pp_only(&self) -> bool {
        self.tp_size == 1
    }

    /// 校验拓扑一致性
    // @trace REQ-DIST-029 [entity:Topology2D]
    pub fn validate(&self) -> bool {
        self.tp_size >= 1
            && self.pp_size >= 1
            && self.tp_size * self.pp_size == self.world_size
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Topology2D: construction ──

    #[test]
    fn new_tp2_pp2_world4() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        assert_eq!(topo.tp_size, 2);
        assert_eq!(topo.pp_size, 2);
        assert_eq!(topo.world_size, 4);
    }

    #[test]
    fn new_tp1_pp1_world1() {
        let topo = Topology2D::new(1, 1, 1).unwrap();
        assert!(topo.is_tp_only());
        assert!(topo.is_pp_only());
    }

    #[test]
    fn new_dimension_mismatch() {
        let result = Topology2D::new(2, 2, 5);
        assert!(result.is_err());
        match result.unwrap_err() {
            Topology2DError::DimensionMismatch { tp_size, pp_size, world_size } => {
                assert_eq!(tp_size, 2);
                assert_eq!(pp_size, 2);
                assert_eq!(world_size, 5);
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn new_invalid_tp_size() {
        let result = Topology2D::new(0, 2, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Topology2DError::InvalidTpSize(0));
    }

    #[test]
    fn new_invalid_pp_size() {
        let result = Topology2D::new(2, 0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Topology2DError::InvalidPpSize(0));
    }

    // ── Topology2D: rank mapping (REQ-DIST-029 验收标准 2) ──

    #[test]
    fn compose_and_decompose_roundtrip() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        for rank in 0..4u32 {
            let (tp_rank, pp_rank) = topo.decompose_rank(rank);
            let recomposed = topo.compose_rank(tp_rank, pp_rank);
            assert_eq!(recomposed, rank, "roundtrip failed for rank={rank}");
        }
    }

    #[test]
    fn compose_rank_formula() {
        // rank(tp_rank, pp_rank) = tp_rank * pp_size + pp_rank
        let topo = Topology2D::new(2, 2, 4).unwrap();
        assert_eq!(topo.compose_rank(0, 0), 0);
        assert_eq!(topo.compose_rank(0, 1), 1);
        assert_eq!(topo.compose_rank(1, 0), 2);
        assert_eq!(topo.compose_rank(1, 1), 3);
    }

    #[test]
    fn decompose_rank_formula() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        assert_eq!(topo.decompose_rank(0), (0, 0));
        assert_eq!(topo.decompose_rank(1), (0, 1));
        assert_eq!(topo.decompose_rank(2), (1, 0));
        assert_eq!(topo.decompose_rank(3), (1, 1));
    }

    #[test]
    fn compose_decompose_tp4_pp3() {
        let topo = Topology2D::new(4, 3, 12).unwrap();
        for rank in 0..12u32 {
            let (tp_rank, pp_rank) = topo.decompose_rank(rank);
            assert!(tp_rank < 4, "tp_rank={tp_rank} >= tp_size=4");
            assert!(pp_rank < 3, "pp_rank={pp_rank} >= pp_size=3");
            assert_eq!(topo.compose_rank(tp_rank, pp_rank), rank);
        }
    }

    // ── Topology2D: TP group (REQ-DIST-029 验收标准 3, 4) ──

    #[test]
    fn tp_group_same_pp_rank() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        // rank 0 (tp=0, pp=0) → TP group = [0, 2] (same pp_rank=0)
        assert_eq!(topo.tp_group(0), vec![0, 2]);
        // rank 1 (tp=0, pp=1) → TP group = [1, 3] (same pp_rank=1)
        assert_eq!(topo.tp_group(1), vec![1, 3]);
        // rank 2 (tp=1, pp=0) → TP group = [0, 2] (same pp_rank=0)
        assert_eq!(topo.tp_group(2), vec![0, 2]);
        // rank 3 (tp=1, pp=1) → TP group = [1, 3] (same pp_rank=1)
        assert_eq!(topo.tp_group(3), vec![1, 3]);
    }

    #[test]
    fn tp_group_tp4_pp2() {
        let topo = Topology2D::new(4, 2, 8).unwrap();
        // rank 0 (tp=0, pp=0) → TP group = [0, 2, 4, 6]
        assert_eq!(topo.tp_group(0), vec![0, 2, 4, 6]);
        // rank 1 (tp=0, pp=1) → TP group = [1, 3, 5, 7]
        assert_eq!(topo.tp_group(1), vec![1, 3, 5, 7]);
    }

    // ── Topology2D: PP group (REQ-DIST-029 验收标准 3, 5) ──

    #[test]
    fn pp_group_same_tp_rank() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        // rank 0 (tp=0, pp=0) → PP group = [0, 1] (same tp_rank=0)
        assert_eq!(topo.pp_group(0), vec![0, 1]);
        // rank 2 (tp=1, pp=0) → PP group = [2, 3] (same tp_rank=1)
        assert_eq!(topo.pp_group(2), vec![2, 3]);
    }

    #[test]
    fn pp_group_tp2_pp4() {
        let topo = Topology2D::new(2, 4, 8).unwrap();
        // rank 0 (tp=0, pp=0) → PP group = [0, 1, 2, 3]
        assert_eq!(topo.pp_group(0), vec![0, 1, 2, 3]);
        // rank 4 (tp=1, pp=0) → PP group = [4, 5, 6, 7]
        assert_eq!(topo.pp_group(4), vec![4, 5, 6, 7]);
    }

    // ── Topology2D: PP P2P endpoints (REQ-DIST-029 验收标准 5) ──

    #[test]
    fn pp_p2p_endpoints_tp2_pp2() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        // tp_rank=0 ranks: [0, 1]
        assert_eq!(topo.pp_p2p_endpoints(), vec![0, 1]);
    }

    #[test]
    fn pp_p2p_endpoints_tp4_pp3() {
        let topo = Topology2D::new(4, 3, 12).unwrap();
        // tp_rank=0 ranks: [0, 1, 2]
        assert_eq!(topo.pp_p2p_endpoints(), vec![0, 1, 2]);
    }

    // ── Topology2D: PP prev/next ──

    #[test]
    fn pp_prev_rank_first_stage_none() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        // rank 0 (tp=0, pp=0) → no prev
        assert_eq!(topo.pp_prev_rank(0), None);
        // rank 2 (tp=1, pp=0) → no prev
        assert_eq!(topo.pp_prev_rank(2), None);
    }

    #[test]
    fn pp_prev_rank_middle_stage() {
        let topo = Topology2D::new(2, 3, 6).unwrap();
        // rank 2 (tp=0, pp=2) → prev = compose(0, 1) = 1
        assert_eq!(topo.pp_prev_rank(2), Some(1));
    }

    #[test]
    fn pp_next_rank_last_stage_none() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        // rank 1 (tp=0, pp=1) → no next
        assert_eq!(topo.pp_next_rank(1), None);
        // rank 3 (tp=1, pp=1) → no next
        assert_eq!(topo.pp_next_rank(3), None);
    }

    #[test]
    fn pp_next_rank_first_stage() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        // rank 0 (tp=0, pp=0) → next = compose(0, 1) = 1
        assert_eq!(topo.pp_next_rank(0), Some(1));
        // rank 2 (tp=1, pp=0) → next = compose(1, 1) = 3
        assert_eq!(topo.pp_next_rank(2), Some(3));
    }

    // ── Topology2D: is_tp_master ──

    #[test]
    fn is_tp_master_rank0() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        assert!(topo.is_tp_master(0));
        assert!(topo.is_tp_master(1));
        assert!(!topo.is_tp_master(2));
        assert!(!topo.is_tp_master(3));
    }

    // ── Topology2D: each rank in exactly one TP group and one PP group ──

    #[test]
    fn each_rank_in_exactly_one_tp_and_pp_group() {
        let topo = Topology2D::new(3, 2, 6).unwrap();
        for rank in 0..6u32 {
            let tp_grp = topo.tp_group(rank);
            let pp_grp = topo.pp_group(rank);
            // Each group contains the rank itself
            assert!(tp_grp.contains(&rank), "rank {rank} not in its TP group");
            assert!(pp_grp.contains(&rank), "rank {rank} not in its PP group");
            // TP group size = tp_size
            assert_eq!(tp_grp.len(), 3);
            // PP group size = pp_size
            assert_eq!(pp_grp.len(), 2);
        }

        // All TP groups are distinct
        let tp_groups: std::collections::HashSet<Vec<u32>> = (0..6).map(|r| topo.tp_group(r)).collect();
        assert_eq!(tp_groups.len(), 2); // 2 distinct TP groups (pp_size=2)

        // All PP groups are distinct
        let pp_groups: std::collections::HashSet<Vec<u32>> = (0..6).map(|r| topo.pp_group(r)).collect();
        assert_eq!(pp_groups.len(), 3); // 3 distinct PP groups (tp_size=3)
    }

    // ── Topology2D: validate ──

    #[test]
    fn validate_valid() {
        let topo = Topology2D::new(2, 2, 4).unwrap();
        assert!(topo.validate());
    }

    // ── Topology2D: from_parallel_config ──

    #[test]
    fn from_parallel_config() {
        let config = crate::engine::distributed_config::ParallelConfig {
            tp_size: 2,
            pp_size: 2,
            ep_size: 1,
            cp_size: 1,
            rank: 0,
            world_size: 4,
            unique_id: String::new(),
            stage_id: 0 / (2 * 1),
        };
        let topo = Topology2D::from_parallel_config(&config).unwrap();
        assert_eq!(topo.tp_size, 2);
        assert_eq!(topo.pp_size, 2);
        assert_eq!(topo.world_size, 4);
    }

    // ── Topology2D: equality and clone ──

    #[test]
    fn equality() {
        let a = Topology2D::new(2, 2, 4).unwrap();
        let b = Topology2D::new(2, 2, 4).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn inequality() {
        let a = Topology2D::new(2, 2, 4).unwrap();
        let b = Topology2D::new(4, 1, 4).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn clone_independence() {
        let mut topo = Topology2D::new(2, 2, 4).unwrap();
        let cloned = topo.clone();
        // Topology2D has no mutable fields, but clone should be independent
        assert_eq!(cloned.tp_size, 2);
        assert_eq!(cloned.pp_size, 2);
        let _ = &mut topo; // just to use mut
    }

    // ── Topology2DError: Display ──

    #[test]
    fn error_display_dimension_mismatch() {
        let err = Topology2DError::DimensionMismatch { tp_size: 2, pp_size: 3, world_size: 5 };
        let msg = format!("{}", err);
        assert!(msg.contains("6")); // 2*3=6
        assert!(msg.contains("5"));
    }

    #[test]
    fn error_display_invalid_tp_size() {
        let err = Topology2DError::InvalidTpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("tp_size=0"));
    }

    #[test]
    fn error_display_invalid_pp_size() {
        let err = Topology2DError::InvalidPpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("pp_size=0"));
    }

    #[test]
    fn error_is_std_error() {
        let err = Topology2DError::DimensionMismatch { tp_size: 2, pp_size: 3, world_size: 5 };
        let _: &dyn std::error::Error = &err;
    }

    // ── Topology2D: hash consistency ──

    #[test]
    fn hash_consistency() {
        use std::collections::HashSet;
        let a = Topology2D::new(2, 2, 4).unwrap();
        let b = Topology2D::new(2, 2, 4).unwrap();
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }

    // ── Topology2D: large topology ──

    #[test]
    fn large_topology_tp8_pp4() {
        let topo = Topology2D::new(8, 4, 32).unwrap();
        assert_eq!(topo.world_size, 32);
        for rank in 0..32u32 {
            let (tp_rank, pp_rank) = topo.decompose_rank(rank);
            assert!(tp_rank < 8);
            assert!(pp_rank < 4);
            assert_eq!(topo.compose_rank(tp_rank, pp_rank), rank);
        }
    }
}

// ── Topology3D (REQ-DIST-030) ──────────────────────────────────────────────

/// 3D 并行 (TP+PP+EP) 拓扑 (REQ-DIST-030)
///
/// 在 2D (TP+PP) 基础上增加 Expert Parallel (EP) 维度。
/// EP 组内执行 MoE AllToAll dispatch/combine（验收标准 4），
/// TP/PP 通信在 EP 维度上独立（验收标准 5）。
///
/// rank 映射公式：
/// `rank(tp_rank, pp_rank, ep_rank) = ep_rank * (tp_size * pp_size) + tp_rank * pp_size + pp_rank`
/// 逆映射：
/// `ep_rank = rank / (tp_size * pp_size)`
/// `tp_rank = (rank % (tp_size * pp_size)) / pp_size`
/// `pp_rank = rank % pp_size`
// @trace REQ-DIST-030 [entity:Topology3D] [api:POST /internal/distributed/topology]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Topology3D {
    /// Tensor Parallel 维度
    pub tp_size: u32,
    /// Pipeline Parallel 维度
    pub pp_size: u32,
    /// Expert Parallel 维度
    pub ep_size: u32,
    /// 全局 world_size = tp_size * pp_size * ep_size
    pub world_size: u32,
}

/// Topology3D 构建错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Topology3DError {
    /// tp_size * pp_size * ep_size != world_size
    DimensionMismatch { tp_size: u32, pp_size: u32, ep_size: u32, world_size: u32 },
    /// tp_size < 1
    InvalidTpSize(u32),
    /// pp_size < 1
    InvalidPpSize(u32),
    /// ep_size < 1
    InvalidEpSize(u32),
}

impl std::fmt::Display for Topology3DError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Topology3DError::DimensionMismatch { tp_size, pp_size, ep_size, world_size } => {
                write!(
                    f,
                    "Topology3D: tp_size({tp_size}) * pp_size({pp_size}) * ep_size({ep_size}) = {} != world_size({world_size})",
                    tp_size * pp_size * ep_size
                )
            }
            Topology3DError::InvalidTpSize(tp_size) => {
                write!(f, "Topology3D: invalid tp_size={tp_size}, must be >= 1")
            }
            Topology3DError::InvalidPpSize(pp_size) => {
                write!(f, "Topology3D: invalid pp_size={pp_size}, must be >= 1")
            }
            Topology3DError::InvalidEpSize(ep_size) => {
                write!(f, "Topology3D: invalid ep_size={ep_size}, must be >= 1")
            }
        }
    }
}

impl std::error::Error for Topology3DError {}

// @trace REQ-DIST-030 [entity:Topology3D]
impl Topology3D {
    /// 创建 3D 并行拓扑 (REQ-DIST-030 验收标准 1, 2)
    ///
    /// 校验 `tp_size * pp_size * ep_size == world_size`。
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn new(
        tp_size: u32,
        pp_size: u32,
        ep_size: u32,
        world_size: u32,
    ) -> Result<Self, Topology3DError> {
        if tp_size < 1 {
            return Err(Topology3DError::InvalidTpSize(tp_size));
        }
        if pp_size < 1 {
            return Err(Topology3DError::InvalidPpSize(pp_size));
        }
        if ep_size < 1 {
            return Err(Topology3DError::InvalidEpSize(ep_size));
        }
        if tp_size * pp_size * ep_size != world_size {
            return Err(Topology3DError::DimensionMismatch {
                tp_size,
                pp_size,
                ep_size,
                world_size,
            });
        }
        Ok(Self {
            tp_size,
            pp_size,
            ep_size,
            world_size,
        })
    }

    /// 从 ParallelConfig 创建 3D 拓扑
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn from_parallel_config(
        config: &crate::engine::distributed_config::ParallelConfig,
    ) -> Result<Self, Topology3DError> {
        Self::new(
            config.tp_size,
            config.pp_size,
            config.ep_size,
            config.world_size,
        )
    }

    /// 降级为 2D 拓扑（丢弃 EP 维度，ep_size=1 子空间）
    ///
    /// 返回当前 EP rank=0 对应的 2D 拓扑视图。
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn to_2d(&self) -> Topology2D {
        Topology2D {
            tp_size: self.tp_size,
            pp_size: self.pp_size,
            world_size: self.tp_size * self.pp_size,
        }
    }

    /// rank → (tp_rank, pp_rank, ep_rank) 逆映射 (REQ-DIST-030 验收标准 3)
    ///
    /// `ep_rank = rank / (tp_size * pp_size)`
    /// `tp_rank = (rank % (tp_size * pp_size)) / pp_size`
    /// `pp_rank = rank % pp_size`
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn decompose_rank(&self, rank: u32) -> (u32, u32, u32) {
        let tp_pp_plane = self.tp_size * self.pp_size;
        let ep_rank = rank / tp_pp_plane;
        let remainder = rank % tp_pp_plane;
        let tp_rank = remainder / self.pp_size;
        let pp_rank = remainder % self.pp_size;
        (tp_rank, pp_rank, ep_rank)
    }

    /// (tp_rank, pp_rank, ep_rank) → rank 正映射 (REQ-DIST-030 验收标准 3)
    ///
    /// `rank = ep_rank * (tp_size * pp_size) + tp_rank * pp_size + pp_rank`
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn compose_rank(&self, tp_rank: u32, pp_rank: u32, ep_rank: u32) -> u32 {
        ep_rank * (self.tp_size * self.pp_size) + tp_rank * self.pp_size + pp_rank
    }

    /// 返回指定 rank 所属 TP 组的所有 rank (REQ-DIST-030 验收标准 5)
    ///
    /// TP 组 = 同 pp_rank + 同 ep_rank 的所有 tp_rank。
    /// TP/PP 通信在 EP 维度上独立。
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn tp_group(&self, rank: u32) -> Vec<u32> {
        let (_, pp_rank, ep_rank) = self.decompose_rank(rank);
        (0..self.tp_size)
            .map(|tp_rank| self.compose_rank(tp_rank, pp_rank, ep_rank))
            .collect()
    }

    /// 返回指定 rank 所属 PP 组的所有 rank (REQ-DIST-030 验收标准 5)
    ///
    /// PP 组 = 同 tp_rank + 同 ep_rank 的所有 pp_rank。
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn pp_group(&self, rank: u32) -> Vec<u32> {
        let (tp_rank, _, ep_rank) = self.decompose_rank(rank);
        (0..self.pp_size)
            .map(|pp_rank| self.compose_rank(tp_rank, pp_rank, ep_rank))
            .collect()
    }

    /// 返回指定 rank 所属 EP 组的所有 rank (REQ-DIST-030 验收标准 4)
    ///
    /// EP 组 = 同 tp_rank + 同 pp_rank 的所有 ep_rank。
    /// EP 组内执行 MoE AllToAll dispatch/combine。
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn ep_group(&self, rank: u32) -> Vec<u32> {
        let (tp_rank, pp_rank, _) = self.decompose_rank(rank);
        (0..self.ep_size)
            .map(|ep_rank| self.compose_rank(tp_rank, pp_rank, ep_rank))
            .collect()
    }

    /// PP P2P 端点：每个 EP 维度内 tp_rank=0 的 ranks
    ///
    /// PP P2P 通信在这些 rank 间进行，跨 EP 维度独立。
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn pp_p2p_endpoints(&self) -> Vec<u32> {
        let mut endpoints = Vec::new();
        for ep_rank in 0..self.ep_size {
            for pp_rank in 0..self.pp_size {
                endpoints.push(self.compose_rank(0, pp_rank, ep_rank));
            }
        }
        endpoints
    }

    /// 返回指定 rank 的 PP 前驱 rank（同 tp_rank + 同 ep_rank）
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn pp_prev_rank(&self, rank: u32) -> Option<u32> {
        let (tp_rank, pp_rank, ep_rank) = self.decompose_rank(rank);
        if pp_rank == 0 {
            None
        } else {
            Some(self.compose_rank(tp_rank, pp_rank - 1, ep_rank))
        }
    }

    /// 返回指定 rank 的 PP 后继 rank（同 tp_rank + 同 ep_rank）
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn pp_next_rank(&self, rank: u32) -> Option<u32> {
        let (tp_rank, pp_rank, ep_rank) = self.decompose_rank(rank);
        if pp_rank + 1 >= self.pp_size {
            None
        } else {
            Some(self.compose_rank(tp_rank, pp_rank + 1, ep_rank))
        }
    }

    /// 是否为 TP 主 rank（tp_rank=0）
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn is_tp_master(&self, rank: u32) -> bool {
        let (tp_rank, _, _) = self.decompose_rank(rank);
        tp_rank == 0
    }

    /// 是否为纯 TP 模式（pp_size == 1 && ep_size == 1）
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn is_tp_only(&self) -> bool {
        self.pp_size == 1 && self.ep_size == 1
    }

    /// 是否为纯 PP 模式（tp_size == 1 && ep_size == 1）
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn is_pp_only(&self) -> bool {
        self.tp_size == 1 && self.ep_size == 1
    }

    /// 是否为纯 EP 模式（tp_size == 1 && pp_size == 1）
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn is_ep_only(&self) -> bool {
        self.tp_size == 1 && self.pp_size == 1
    }

    /// EP 组内 rank 的偏移（在 EP 组内的索引）
    ///
    /// 用于 MoE AllToAll dispatch/combine 中的数据分发。
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn ep_rank_in_group(&self, rank: u32) -> u32 {
        let (_, _, ep_rank) = self.decompose_rank(rank);
        ep_rank
    }

    /// 校验拓扑一致性
    // @trace REQ-DIST-030 [entity:Topology3D]
    pub fn validate(&self) -> bool {
        self.tp_size >= 1
            && self.pp_size >= 1
            && self.ep_size >= 1
            && self.tp_size * self.pp_size * self.ep_size == self.world_size
    }
}

// ── Topology3D Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests_3d {
    use super::*;

    // ── Topology3D: construction ──

    #[test]
    fn new_tp2_pp2_ep2_world8() {
        // @trace TEST-DIST-030 [req:REQ-DIST-030] [level:unit]
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        assert_eq!(topo.tp_size, 2);
        assert_eq!(topo.pp_size, 2);
        assert_eq!(topo.ep_size, 2);
        assert_eq!(topo.world_size, 8);
    }

    #[test]
    fn new_tp1_pp1_ep1_world1() {
        let topo = Topology3D::new(1, 1, 1, 1).unwrap();
        assert!(topo.is_tp_only());
        assert!(topo.is_pp_only());
        assert!(topo.is_ep_only());
    }

    #[test]
    fn new_dimension_mismatch() {
        let result = Topology3D::new(2, 2, 2, 7);
        assert!(result.is_err());
        match result.unwrap_err() {
            Topology3DError::DimensionMismatch { tp_size, pp_size, ep_size, world_size } => {
                assert_eq!(tp_size, 2);
                assert_eq!(pp_size, 2);
                assert_eq!(ep_size, 2);
                assert_eq!(world_size, 7);
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn new_invalid_tp_size() {
        let result = Topology3D::new(0, 2, 2, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Topology3DError::InvalidTpSize(0));
    }

    #[test]
    fn new_invalid_pp_size() {
        let result = Topology3D::new(2, 0, 2, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Topology3DError::InvalidPpSize(0));
    }

    #[test]
    fn new_invalid_ep_size() {
        let result = Topology3D::new(2, 2, 0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), Topology3DError::InvalidEpSize(0));
    }

    // ── Topology3D: rank mapping (REQ-DIST-030 验收标准 3) ──

    #[test]
    fn compose_and_decompose_roundtrip() {
        // @trace TEST-DIST-030 [req:REQ-DIST-030] [level:unit]
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        for rank in 0..8u32 {
            let (tp_rank, pp_rank, ep_rank) = topo.decompose_rank(rank);
            let recomposed = topo.compose_rank(tp_rank, pp_rank, ep_rank);
            assert_eq!(recomposed, rank, "roundtrip failed for rank={rank}");
        }
    }

    #[test]
    fn decompose_rank_formula() {
        // rank = ep_rank * (tp*pp) + tp_rank * pp + pp_rank
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        // rank 0: ep=0, tp=0, pp=0
        assert_eq!(topo.decompose_rank(0), (0, 0, 0));
        // rank 1: ep=0, tp=0, pp=1
        assert_eq!(topo.decompose_rank(1), (0, 1, 0));
        // rank 2: ep=0, tp=1, pp=0
        assert_eq!(topo.decompose_rank(2), (1, 0, 0));
        // rank 3: ep=0, tp=1, pp=1
        assert_eq!(topo.decompose_rank(3), (1, 1, 0));
        // rank 4: ep=1, tp=0, pp=0
        assert_eq!(topo.decompose_rank(4), (0, 0, 1));
        // rank 5: ep=1, tp=0, pp=1
        assert_eq!(topo.decompose_rank(5), (0, 1, 1));
        // rank 6: ep=1, tp=1, pp=0
        assert_eq!(topo.decompose_rank(6), (1, 0, 1));
        // rank 7: ep=1, tp=1, pp=1
        assert_eq!(topo.decompose_rank(7), (1, 1, 1));
    }

    #[test]
    fn compose_decompose_tp2_pp3_ep4() {
        let topo = Topology3D::new(2, 3, 4, 24).unwrap();
        for rank in 0..24u32 {
            let (tp_rank, pp_rank, ep_rank) = topo.decompose_rank(rank);
            assert!(tp_rank < 2, "tp_rank={tp_rank} >= tp_size=2");
            assert!(pp_rank < 3, "pp_rank={pp_rank} >= pp_size=3");
            assert!(ep_rank < 4, "ep_rank={ep_rank} >= ep_size=4");
            assert_eq!(topo.compose_rank(tp_rank, pp_rank, ep_rank), rank);
        }
    }

    // ── Topology3D: TP group (REQ-DIST-030 验收标准 5) ──

    #[test]
    fn tp_group_same_pp_ep_rank() {
        // @trace TEST-DIST-030 [req:REQ-DIST-030] [level:unit]
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        // rank 0 (tp=0, pp=0, ep=0) → TP group = [0, 2] (same pp=0, ep=0)
        assert_eq!(topo.tp_group(0), vec![0, 2]);
        // rank 1 (tp=0, pp=1, ep=0) → TP group = [1, 3] (same pp=1, ep=0)
        assert_eq!(topo.tp_group(1), vec![1, 3]);
        // rank 4 (tp=0, pp=0, ep=1) → TP group = [4, 6] (same pp=0, ep=1)
        assert_eq!(topo.tp_group(4), vec![4, 6]);
        // TP groups across EP dimensions are independent
        assert_ne!(topo.tp_group(0), topo.tp_group(4));
    }

    // ── Topology3D: PP group (REQ-DIST-030 验收标准 5) ──

    #[test]
    fn pp_group_same_tp_ep_rank() {
        // @trace TEST-DIST-030 [req:REQ-DIST-030] [level:unit]
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        // rank 0 (tp=0, pp=0, ep=0) → PP group = [0, 1] (same tp=0, ep=0)
        assert_eq!(topo.pp_group(0), vec![0, 1]);
        // rank 4 (tp=0, pp=0, ep=1) → PP group = [4, 5] (same tp=0, ep=1)
        assert_eq!(topo.pp_group(4), vec![4, 5]);
        // PP groups across EP dimensions are independent
        assert_ne!(topo.pp_group(0), topo.pp_group(4));
    }

    // ── Topology3D: EP group (REQ-DIST-030 验收标准 4) ──

    #[test]
    fn ep_group_same_tp_pp_rank() {
        // @trace TEST-DIST-030 [req:REQ-DIST-030] [level:unit]
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        // rank 0 (tp=0, pp=0, ep=0) → EP group = [0, 4] (same tp=0, pp=0)
        assert_eq!(topo.ep_group(0), vec![0, 4]);
        // rank 1 (tp=0, pp=1, ep=0) → EP group = [1, 5] (same tp=0, pp=1)
        assert_eq!(topo.ep_group(1), vec![1, 5]);
        // rank 2 (tp=1, pp=0, ep=0) → EP group = [2, 6] (same tp=1, pp=0)
        assert_eq!(topo.ep_group(2), vec![2, 6]);
        // rank 3 (tp=1, pp=1, ep=0) → EP group = [3, 7] (same tp=1, pp=1)
        assert_eq!(topo.ep_group(3), vec![3, 7]);
    }

    #[test]
    fn ep_group_moe_allto_all() {
        // @trace TEST-DIST-030 [req:REQ-DIST-030] [level:unit]
        // 验收标准 4: EP 组内执行 MoE AllToAll dispatch/combine
        let topo = Topology3D::new(1, 1, 4, 4).unwrap();
        // rank 0 → EP group = [0, 1, 2, 3] (all ranks in ep group)
        assert_eq!(topo.ep_group(0), vec![0, 1, 2, 3]);
    }

    // ── Topology3D: PP P2P endpoints ──

    #[test]
    fn pp_p2p_endpoints_3d() {
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        // P2P endpoints: tp_rank=0, all pp_rank, all ep_rank
        let endpoints = topo.pp_p2p_endpoints();
        // ep=0: rank 0 (tp=0,pp=0,ep=0), rank 1 (tp=0,pp=1,ep=0)
        // ep=1: rank 4 (tp=0,pp=0,ep=1), rank 5 (tp=0,pp=1,ep=1)
        assert_eq!(endpoints, vec![0, 1, 4, 5]);
    }

    // ── Topology3D: PP prev/next ──

    #[test]
    fn pp_prev_rank_first_stage_none() {
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        // rank 0 (tp=0, pp=0, ep=0) → no prev
        assert_eq!(topo.pp_prev_rank(0), None);
        // rank 4 (tp=0, pp=0, ep=1) → no prev (EP-independent PP)
        assert_eq!(topo.pp_prev_rank(4), None);
    }

    #[test]
    fn pp_next_rank_last_stage_none() {
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        // rank 1 (tp=0, pp=1, ep=0) → no next
        assert_eq!(topo.pp_next_rank(1), None);
        // rank 5 (tp=0, pp=1, ep=1) → no next
        assert_eq!(topo.pp_next_rank(5), None);
    }

    #[test]
    fn pp_prev_next_within_same_ep() {
        // PP communication stays within the same EP dimension
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        // rank 0 (tp=0, pp=0, ep=0) → next = rank 1 (tp=0, pp=1, ep=0)
        assert_eq!(topo.pp_next_rank(0), Some(1));
        // rank 4 (tp=0, pp=0, ep=1) → next = rank 5 (tp=0, pp=1, ep=1)
        assert_eq!(topo.pp_next_rank(4), Some(5));
        // NOT rank 5 (which is in ep=1)
        assert_ne!(topo.pp_next_rank(0), Some(5));
    }

    // ── Topology3D: is_tp_master ──

    #[test]
    fn is_tp_master_3d() {
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        assert!(topo.is_tp_master(0));  // tp=0
        assert!(topo.is_tp_master(1));  // tp=0
        assert!(topo.is_tp_master(4));  // tp=0
        assert!(!topo.is_tp_master(2)); // tp=1
        assert!(!topo.is_tp_master(6)); // tp=1
    }

    // ── Topology3D: mode checks ──

    #[test]
    fn is_tp_only() {
        let topo = Topology3D::new(4, 1, 1, 4).unwrap();
        assert!(topo.is_tp_only());
        assert!(!topo.is_pp_only());
        assert!(!topo.is_ep_only());
    }

    #[test]
    fn is_pp_only() {
        let topo = Topology3D::new(1, 4, 1, 4).unwrap();
        assert!(!topo.is_tp_only());
        assert!(topo.is_pp_only());
        assert!(!topo.is_ep_only());
    }

    #[test]
    fn is_ep_only() {
        let topo = Topology3D::new(1, 1, 4, 4).unwrap();
        assert!(!topo.is_tp_only());
        assert!(!topo.is_pp_only());
        assert!(topo.is_ep_only());
    }

    // ── Topology3D: each rank in exactly one group of each type ──

    #[test]
    fn each_rank_in_exactly_one_group() {
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        for rank in 0..8u32 {
            let tp_grp = topo.tp_group(rank);
            let pp_grp = topo.pp_group(rank);
            let ep_grp = topo.ep_group(rank);
            assert!(tp_grp.contains(&rank), "rank {rank} not in its TP group");
            assert!(pp_grp.contains(&rank), "rank {rank} not in its PP group");
            assert!(ep_grp.contains(&rank), "rank {rank} not in its EP group");
            assert_eq!(tp_grp.len(), 2);
            assert_eq!(pp_grp.len(), 2);
            assert_eq!(ep_grp.len(), 2);
        }

        // Distinct groups
        let tp_groups: std::collections::HashSet<Vec<u32>> =
            (0..8).map(|r| topo.tp_group(r)).collect();
        assert_eq!(tp_groups.len(), 4); // pp_size * ep_size = 2*2

        let pp_groups: std::collections::HashSet<Vec<u32>> =
            (0..8).map(|r| topo.pp_group(r)).collect();
        assert_eq!(pp_groups.len(), 4); // tp_size * ep_size = 2*2

        let ep_groups: std::collections::HashSet<Vec<u32>> =
            (0..8).map(|r| topo.ep_group(r)).collect();
        assert_eq!(ep_groups.len(), 4); // tp_size * pp_size = 2*2
    }

    // ── Topology3D: to_2d ──

    #[test]
    fn to_2d() {
        let topo_3d = Topology3D::new(2, 2, 2, 8).unwrap();
        let topo_2d = topo_3d.to_2d();
        assert_eq!(topo_2d.tp_size, 2);
        assert_eq!(topo_2d.pp_size, 2);
        assert_eq!(topo_2d.world_size, 4);
    }

    // ── Topology3D: from_parallel_config ──

    #[test]
    fn from_parallel_config() {
        let config = crate::engine::distributed_config::ParallelConfig {
            tp_size: 2,
            pp_size: 2,
            ep_size: 2,
            cp_size: 1,
            rank: 0,
            world_size: 8,
            unique_id: String::new(),
            stage_id: 0,
        };
        let topo = Topology3D::from_parallel_config(&config).unwrap();
        assert_eq!(topo.tp_size, 2);
        assert_eq!(topo.pp_size, 2);
        assert_eq!(topo.ep_size, 2);
        assert_eq!(topo.world_size, 8);
    }

    // ── Topology3D: validate ──

    #[test]
    fn validate_valid() {
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        assert!(topo.validate());
    }

    // ── Topology3D: ep_rank_in_group ──

    #[test]
    fn ep_rank_in_group() {
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        // rank 0 (ep=0) → ep_rank_in_group = 0
        assert_eq!(topo.ep_rank_in_group(0), 0);
        // rank 4 (ep=1) → ep_rank_in_group = 1
        assert_eq!(topo.ep_rank_in_group(4), 1);
    }

    // ── Topology3D: large topology ──

    #[test]
    fn large_topology_tp4_pp2_ep2() {
        let topo = Topology3D::new(4, 2, 2, 16).unwrap();
        assert_eq!(topo.world_size, 16);
        for rank in 0..16u32 {
            let (tp_rank, pp_rank, ep_rank) = topo.decompose_rank(rank);
            assert!(tp_rank < 4);
            assert!(pp_rank < 2);
            assert!(ep_rank < 2);
            assert_eq!(topo.compose_rank(tp_rank, pp_rank, ep_rank), rank);
        }
    }

    // ── Topology3D: equality and clone ──

    #[test]
    fn equality() {
        let a = Topology3D::new(2, 2, 2, 8).unwrap();
        let b = Topology3D::new(2, 2, 2, 8).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn inequality() {
        let a = Topology3D::new(2, 2, 2, 8).unwrap();
        let b = Topology3D::new(2, 2, 4, 16).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn clone_independence() {
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();
        let cloned = topo.clone();
        assert_eq!(cloned.tp_size, 2);
        assert_eq!(cloned.pp_size, 2);
        assert_eq!(cloned.ep_size, 2);
    }

    // ── Topology3DError: Display ──

    #[test]
    fn error_display_dimension_mismatch() {
        let err = Topology3DError::DimensionMismatch {
            tp_size: 2,
            pp_size: 2,
            ep_size: 2,
            world_size: 7,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("8")); // 2*2*2=8
        assert!(msg.contains("7"));
    }

    #[test]
    fn error_display_invalid_tp_size() {
        let err = Topology3DError::InvalidTpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("tp_size=0"));
    }

    #[test]
    fn error_display_invalid_pp_size() {
        let err = Topology3DError::InvalidPpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("pp_size=0"));
    }

    #[test]
    fn error_display_invalid_ep_size() {
        let err = Topology3DError::InvalidEpSize(0);
        let msg = format!("{}", err);
        assert!(msg.contains("ep_size=0"));
    }

    #[test]
    fn error_is_std_error() {
        let err = Topology3DError::DimensionMismatch {
            tp_size: 2,
            pp_size: 2,
            ep_size: 2,
            world_size: 7,
        };
        let _: &dyn std::error::Error = &err;
    }

    // ── Topology3D: hash consistency ──

    #[test]
    fn hash_consistency() {
        use std::collections::HashSet;
        let a = Topology3D::new(2, 2, 2, 8).unwrap();
        let b = Topology3D::new(2, 2, 2, 8).unwrap();
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }

    // ── Topology3D: TP/PP communication independent across EP (验收标准 5) ──

    #[test]
    fn tp_pp_communication_independent_across_ep() {
        // @trace TEST-DIST-030 [req:REQ-DIST-030] [level:unit]
        // 验收标准 5: TP/PP 通信在 EP 维度上独立
        let topo = Topology3D::new(2, 2, 2, 8).unwrap();

        // TP groups for different EP ranks should be disjoint
        let tp_group_ep0 = topo.tp_group(0); // ep=0
        let tp_group_ep1 = topo.tp_group(4); // ep=1
        let tp_intersection: Vec<u32> = tp_group_ep0
            .iter()
            .filter(|r| tp_group_ep1.contains(r))
            .copied()
            .collect();
        assert!(tp_intersection.is_empty(), "TP groups across EP should be disjoint");

        // PP groups for different EP ranks should be disjoint
        let pp_group_ep0 = topo.pp_group(0); // ep=0
        let pp_group_ep1 = topo.pp_group(4); // ep=1
        let pp_intersection: Vec<u32> = pp_group_ep0
            .iter()
            .filter(|r| pp_group_ep1.contains(r))
            .copied()
            .collect();
        assert!(pp_intersection.is_empty(), "PP groups across EP should be disjoint");
    }
}
