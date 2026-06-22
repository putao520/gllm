//! DistributedConfig — 分布式推理配置聚合体 (REQ-IB-006~012)
//!
//! nccl feature-gated 配置体系，覆盖并行策略、PD 分离、KV 分布、
//! 通信压缩、MoE 分布式。所有字段均有 Default 值（单机模式），
//! 用户可选择性覆盖。

// ── PdDisaggMode (REQ-IB-008) ──────────────────────────────────────────────

/// PdDisaggMode — Prefill/Decode 分离模式 (REQ-IB-008)
///
/// `Collocated` = Prefill 和 Decode 在同一节点（默认单机模式）。
/// `Disaggregated` = Prefill 和 Decode 分离到不同节点。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PdDisaggMode {
    #[default]
    Collocated,
    Disaggregated,
}

// ── NodeRole (REQ-IB-008) ──────────────────────────────────────────────────

/// NodeRole — 节点角色 (REQ-IB-008)
///
/// `Auto` = 由系统根据 PdDisaggMode 自动推导。
/// `PrefillOnly` / `DecodeOnly` / `Mixed` = 显式指定节点角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NodeRole {
    #[default]
    Auto,
    PrefillOnly,
    DecodeOnly,
    Mixed,
}

// ── KvDistMode (REQ-IB-009) ────────────────────────────────────────────────

/// KvDistMode — KV Cache 分布模式 (REQ-IB-009)
///
/// - `Local`: 本地 KV Cache，无跨节点共享（默认）
/// - `OnDemand`: 按需从远端拉取 KV Cache
/// - `Mirror`: 全量镜像 KV Cache 到所有 TP 节点
/// - `PartialHeadMirror`: 部分头镜像（mirror_heads 控制镜像头数）
/// - `TieredCache`: 分层缓存（本地 + 远端 + 磁盘）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvDistMode {
    #[default]
    Local,
    OnDemand,
    Mirror,
    PartialHeadMirror,
    TieredCache,
}

// ── CommCompressHint (REQ-IB-010) ──────────────────────────────────────────

/// CommCompressHint — 通信压缩偏好 (REQ-IB-010)
///
/// - `Auto`: 由系统根据带宽和消息大小自动决定
/// - `AlwaysCompress`: 始终压缩通信数据
/// - `NeverCompress`: 始终不压缩
/// - `ForceQuant`: 强制量化压缩（最高压缩比）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CommCompressHint {
    #[default]
    Auto,
    AlwaysCompress,
    NeverCompress,
    ForceQuant,
}

// ── ExpertPlacement (REQ-IB-011) ───────────────────────────────────────────

/// ExpertPlacement — MoE 专家放置策略 (REQ-IB-011)
///
/// - `Auto`: 由系统根据专家数量和 GPU 拓扑自动决定
/// - `RoundRobin`: 轮询分配专家到各 GPU
/// - `HotCold`: 热专家常驻 + 冷专家按需加载
/// - `Custom`: 用户自定义放置（通过 external planner）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExpertPlacement {
    #[default]
    Auto,
    RoundRobin,
    HotCold,
    Custom,
}

// ── AllToAllStrategy (REQ-IB-011) ──────────────────────────────────────────

/// AllToAllStrategy — MoE AllToAll 通信策略 (REQ-IB-011)
///
/// - `Auto`: 由系统根据互连拓扑自动选择
/// - `NvlinkAllToAll`: NVLink 直连 AllToAll
/// - `RdmaAllToAll`: RDMA 跨节点 AllToAll
/// - `HierarchicalAllToAll`: 层次化 AllToAll（节点内 NVLink + 节点间 RDMA）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AllToAllStrategy {
    #[default]
    Auto,
    NvlinkAllToAll,
    RdmaAllToAll,
    HierarchicalAllToAll,
}

// ── ParallelConfig (REQ-IB-007) ────────────────────────────────────────────

/// ParallelConfig — 并行策略配置 (REQ-IB-007)
///
/// 定义 Tensor Parallel / Pipeline Parallel / Expert Parallel 维度，
/// 以及当前节点在并行组中的 rank 和全局 world_size。
/// 约束：`tp_size * pp_size * ep_size == world_size` 且 `rank < world_size`。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelConfig {
    /// Tensor Parallel 维度，默认 1，≥1
    pub tp_size: u32,
    /// Pipeline Parallel 维度，默认 1，≥1
    pub pp_size: u32,
    /// Expert Parallel 维度，默认 1，≥1
    pub ep_size: u32,
    /// 当前节点 rank，默认 0，范围 [0, world_size)
    pub rank: u32,
    /// 全局 world_size，默认 1，必须等于 tp*pp*ep
    pub world_size: u32,
    /// NCCL unique_id 字符串，默认 ""
    pub unique_id: String,
}

// @trace REQ-IB-007 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            ep_size: 1,
            rank: 0,
            world_size: 1,
            unique_id: String::new(),
        }
    }
}

// @trace REQ-IB-007 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl ParallelConfig {
    /// 校验并行配置一致性
    ///
    /// 返回 `true` 当且仅当：
    /// - tp_size ≥ 1, pp_size ≥ 1, ep_size ≥ 1
    /// - rank < world_size
    /// - tp_size * pp_size * ep_size == world_size
    pub fn validate(&self) -> bool {
        self.tp_size >= 1
            && self.pp_size >= 1
            && self.ep_size >= 1
            && self.rank < self.world_size
            && self.tp_size * self.pp_size * self.ep_size == self.world_size
    }
}

// ── PdDisaggConfig (REQ-IB-008) ────────────────────────────────────────────

/// PdDisaggConfig — Prefill/Decode 分离配置 (REQ-IB-008)
///
/// 控制 Prefill 和 Decode 是否分离到不同节点，
/// 以及当前节点的角色。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct PdDisaggConfig {
    /// PD 分离模式
    pub mode: PdDisaggMode,
    /// 当前节点角色
    pub role: NodeRole,
}

// @trace REQ-IB-008 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for PdDisaggConfig {
    fn default() -> Self {
        Self {
            mode: PdDisaggMode::Collocated,
            role: NodeRole::Auto,
        }
    }
}

// ── KvDistributionConfig (REQ-IB-009) ──────────────────────────────────────

/// KvDistributionConfig — KV Cache 分布配置 (REQ-IB-009)
///
/// 控制跨节点 KV Cache 的分布策略。
/// `mirror_heads` 仅在 `PartialHeadMirror` 模式下有效，
/// 指定镜像的 KV head 数量。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct KvDistributionConfig {
    /// KV 分布模式
    pub mode: KvDistMode,
    /// 镜像头数，默认 0，仅 PartialHeadMirror 模式有效
    pub mirror_heads: u32,
}

// @trace REQ-IB-009 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for KvDistributionConfig {
    fn default() -> Self {
        Self {
            mode: KvDistMode::Local,
            mirror_heads: 0,
        }
    }
}

// ── CommConfig (REQ-IB-010) ────────────────────────────────────────────────

/// CommConfig — 通信配置 (REQ-IB-010)
///
/// 控制通信-计算重叠策略、压缩偏好、以及算法覆盖。
/// `overlap` 复用 `intent_bias::OverlapHint` (REQ-IB-003)。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct CommConfig {
    /// 通信-计算重叠偏好，复用 REQ-IB-003 OverlapHint
    pub overlap: crate::engine::intent_bias::OverlapHint,
    /// 通信压缩偏好
    pub compress: CommCompressHint,
    /// 算法覆盖名称，默认 ""（不覆盖）
    pub algorithm_override: String,
}

// @trace REQ-IB-010 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for CommConfig {
    fn default() -> Self {
        Self {
            overlap: crate::engine::intent_bias::OverlapHint::Auto,
            compress: CommCompressHint::Auto,
            algorithm_override: String::new(),
        }
    }
}

// ── MoeDistributedConfig (REQ-IB-011) ──────────────────────────────────────

/// MoeDistributedConfig — MoE 分布式配置 (REQ-IB-011)
///
/// 控制 MoE 模型的专家放置策略和 AllToAll 通信策略。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct MoeDistributedConfig {
    /// 专家放置策略
    pub expert_placement: ExpertPlacement,
    /// AllToAll 通信策略
    pub all_to_all: AllToAllStrategy,
}

// @trace REQ-IB-011 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for MoeDistributedConfig {
    fn default() -> Self {
        Self {
            expert_placement: ExpertPlacement::Auto,
            all_to_all: AllToAllStrategy::Auto,
        }
    }
}

// ── DistributedConfig (REQ-IB-006) ─────────────────────────────────────────

/// DistributedConfig — 分布式推理配置聚合体 (REQ-IB-006)
///
/// nccl feature-gated，Default = 单机模式（所有子配置均为默认值）。
/// 聚合 5 个子配置：Parallel / PdDisagg / KvDistribution / Comm / MoeDistributed。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct DistributedConfig {
    /// 并行策略配置
    pub parallel: ParallelConfig,
    /// Prefill/Decode 分离配置
    pub pd_disagg: PdDisaggConfig,
    /// KV Cache 分布配置
    pub kv_distribution: KvDistributionConfig,
    /// 通信配置
    pub comm: CommConfig,
    /// MoE 分布式配置
    pub moe: MoeDistributedConfig,
}

// @trace REQ-IB-006 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            parallel: ParallelConfig::default(),
            pd_disagg: PdDisaggConfig::default(),
            kv_distribution: KvDistributionConfig::default(),
            comm: CommConfig::default(),
            moe: MoeDistributedConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PdDisaggMode (REQ-IB-008) ──────────────────────────────────────────

    #[test]
    fn pd_disagg_mode_default_is_collocated() {
        assert_eq!(PdDisaggMode::default(), PdDisaggMode::Collocated);
    }

    #[test]
    fn pd_disagg_mode_variants_distinct() {
        use std::collections::HashSet;
        let all = [PdDisaggMode::Collocated, PdDisaggMode::Disaggregated];
        let set: HashSet<PdDisaggMode> = all.into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn pd_disagg_mode_copy_trait() {
        let original = PdDisaggMode::Disaggregated;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn pd_disagg_mode_debug_trait() {
        let debug = format!("{:?}", PdDisaggMode::Collocated);
        assert!(debug.contains("Collocated"));
    }

    // ── NodeRole (REQ-IB-008) ──────────────────────────────────────────────

    #[test]
    fn node_role_default_is_auto() {
        assert_eq!(NodeRole::default(), NodeRole::Auto);
    }

    #[test]
    fn node_role_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            NodeRole::Auto,
            NodeRole::PrefillOnly,
            NodeRole::DecodeOnly,
            NodeRole::Mixed,
        ];
        let set: HashSet<NodeRole> = all.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn node_role_copy_trait() {
        let original = NodeRole::PrefillOnly;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn node_role_debug_trait() {
        let debug = format!("{:?}", NodeRole::DecodeOnly);
        assert!(debug.contains("DecodeOnly"));
    }

    // ── KvDistMode (REQ-IB-009) ────────────────────────────────────────────

    #[test]
    fn kv_dist_mode_default_is_local() {
        assert_eq!(KvDistMode::default(), KvDistMode::Local);
    }

    #[test]
    fn kv_dist_mode_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            KvDistMode::Local,
            KvDistMode::OnDemand,
            KvDistMode::Mirror,
            KvDistMode::PartialHeadMirror,
            KvDistMode::TieredCache,
        ];
        let set: HashSet<KvDistMode> = all.into_iter().collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn kv_dist_mode_copy_trait() {
        let original = KvDistMode::Mirror;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn kv_dist_mode_debug_trait() {
        let debug = format!("{:?}", KvDistMode::PartialHeadMirror);
        assert!(debug.contains("PartialHeadMirror"));
    }

    // ── CommCompressHint (REQ-IB-010) ──────────────────────────────────────

    #[test]
    fn comm_compress_hint_default_is_auto() {
        assert_eq!(CommCompressHint::default(), CommCompressHint::Auto);
    }

    #[test]
    fn comm_compress_hint_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            CommCompressHint::Auto,
            CommCompressHint::AlwaysCompress,
            CommCompressHint::NeverCompress,
            CommCompressHint::ForceQuant,
        ];
        let set: HashSet<CommCompressHint> = all.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn comm_compress_hint_copy_trait() {
        let original = CommCompressHint::ForceQuant;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn comm_compress_hint_debug_trait() {
        let debug = format!("{:?}", CommCompressHint::AlwaysCompress);
        assert!(debug.contains("AlwaysCompress"));
    }

    // ── ExpertPlacement (REQ-IB-011) ───────────────────────────────────────

    #[test]
    fn expert_placement_default_is_auto() {
        assert_eq!(ExpertPlacement::default(), ExpertPlacement::Auto);
    }

    #[test]
    fn expert_placement_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            ExpertPlacement::Auto,
            ExpertPlacement::RoundRobin,
            ExpertPlacement::HotCold,
            ExpertPlacement::Custom,
        ];
        let set: HashSet<ExpertPlacement> = all.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn expert_placement_copy_trait() {
        let original = ExpertPlacement::HotCold;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn expert_placement_debug_trait() {
        let debug = format!("{:?}", ExpertPlacement::RoundRobin);
        assert!(debug.contains("RoundRobin"));
    }

    // ── AllToAllStrategy (REQ-IB-011) ──────────────────────────────────────

    #[test]
    fn all_to_all_strategy_default_is_auto() {
        assert_eq!(AllToAllStrategy::default(), AllToAllStrategy::Auto);
    }

    #[test]
    fn all_to_all_strategy_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            AllToAllStrategy::Auto,
            AllToAllStrategy::NvlinkAllToAll,
            AllToAllStrategy::RdmaAllToAll,
            AllToAllStrategy::HierarchicalAllToAll,
        ];
        let set: HashSet<AllToAllStrategy> = all.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn all_to_all_strategy_copy_trait() {
        let original = AllToAllStrategy::NvlinkAllToAll;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn all_to_all_strategy_debug_trait() {
        let debug = format!("{:?}", AllToAllStrategy::RdmaAllToAll);
        assert!(debug.contains("RdmaAllToAll"));
    }

    // ── nccl-gated struct tests ────────────────────────────────────────────

    #[cfg(feature = "nccl")]
    mod nccl_tests {
        use super::*;

        // ── ParallelConfig (REQ-IB-007) ────────────────────────────────────

        #[test]
        fn parallel_config_default_values() {
            let cfg = ParallelConfig::default();
            assert_eq!(cfg.tp_size, 1);
            assert_eq!(cfg.pp_size, 1);
            assert_eq!(cfg.ep_size, 1);
            assert_eq!(cfg.rank, 0);
            assert_eq!(cfg.world_size, 1);
            assert!(cfg.unique_id.is_empty());
        }

        #[test]
        fn parallel_config_validate_default_passes() {
            let cfg = ParallelConfig::default();
            assert!(cfg.validate());
        }

        #[test]
        fn parallel_config_validate_tp2_pp2_ep2_world8() {
            let cfg = ParallelConfig {
                tp_size: 2,
                pp_size: 2,
                ep_size: 2,
                rank: 0,
                world_size: 8,
                unique_id: String::new(),
            };
            assert!(cfg.validate());
        }

        #[test]
        fn parallel_config_validate_mismatch_world_size_fails() {
            let cfg = ParallelConfig {
                tp_size: 2,
                pp_size: 2,
                ep_size: 1,
                rank: 0,
                world_size: 5, // 2*2*1=4 != 5
                unique_id: String::new(),
            };
            assert!(!cfg.validate());
        }

        #[test]
        fn parallel_config_validate_rank_out_of_range_fails() {
            let cfg = ParallelConfig {
                tp_size: 1,
                pp_size: 1,
                ep_size: 1,
                rank: 1, // rank >= world_size
                world_size: 1,
                unique_id: String::new(),
            };
            assert!(!cfg.validate());
        }

        #[test]
        fn parallel_config_validate_zero_tp_fails() {
            let cfg = ParallelConfig {
                tp_size: 0,
                pp_size: 1,
                ep_size: 1,
                rank: 0,
                world_size: 0,
                unique_id: String::new(),
            };
            assert!(!cfg.validate());
        }

        #[test]
        fn parallel_config_equality() {
            let a = ParallelConfig::default();
            let b = ParallelConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn parallel_config_inequality_unique_id() {
            let a = ParallelConfig::default();
            let b = ParallelConfig {
                unique_id: "test-id".to_string(),
                ..Default::default()
            };
            assert_ne!(a, b);
        }

        #[test]
        fn parallel_config_clone_independence() {
            let mut cfg = ParallelConfig {
                tp_size: 4,
                pp_size: 2,
                ep_size: 1,
                rank: 3,
                world_size: 8,
                unique_id: "abc".to_string(),
            };
            let cloned = cfg.clone();
            cfg.tp_size = 1;
            assert_eq!(cloned.tp_size, 4);
        }

        // ── PdDisaggConfig (REQ-IB-008) ────────────────────────────────────

        #[test]
        fn pd_disagg_config_default_values() {
            let cfg = PdDisaggConfig::default();
            assert_eq!(cfg.mode, PdDisaggMode::Collocated);
            assert_eq!(cfg.role, NodeRole::Auto);
        }

        #[test]
        fn pd_disagg_config_custom_values() {
            let cfg = PdDisaggConfig {
                mode: PdDisaggMode::Disaggregated,
                role: NodeRole::PrefillOnly,
            };
            assert_eq!(cfg.mode, PdDisaggMode::Disaggregated);
            assert_eq!(cfg.role, NodeRole::PrefillOnly);
        }

        #[test]
        fn pd_disagg_config_equality() {
            let a = PdDisaggConfig::default();
            let b = PdDisaggConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn pd_disagg_config_clone_independence() {
            let mut cfg = PdDisaggConfig {
                mode: PdDisaggMode::Disaggregated,
                role: NodeRole::DecodeOnly,
            };
            let cloned = cfg.clone();
            cfg.role = NodeRole::Mixed;
            assert_eq!(cloned.role, NodeRole::DecodeOnly);
        }

        // ── KvDistributionConfig (REQ-IB-009) ──────────────────────────────

        #[test]
        fn kv_distribution_config_default_values() {
            let cfg = KvDistributionConfig::default();
            assert_eq!(cfg.mode, KvDistMode::Local);
            assert_eq!(cfg.mirror_heads, 0);
        }

        #[test]
        fn kv_distribution_config_partial_head_mirror() {
            let cfg = KvDistributionConfig {
                mode: KvDistMode::PartialHeadMirror,
                mirror_heads: 4,
            };
            assert_eq!(cfg.mode, KvDistMode::PartialHeadMirror);
            assert_eq!(cfg.mirror_heads, 4);
        }

        #[test]
        fn kv_distribution_config_equality() {
            let a = KvDistributionConfig::default();
            let b = KvDistributionConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn kv_distribution_config_clone_independence() {
            let mut cfg = KvDistributionConfig {
                mode: KvDistMode::Mirror,
                mirror_heads: 8,
            };
            let cloned = cfg.clone();
            cfg.mirror_heads = 2;
            assert_eq!(cloned.mirror_heads, 8);
        }

        // ── CommConfig (REQ-IB-010) ────────────────────────────────────────

        #[test]
        fn comm_config_default_values() {
            let cfg = CommConfig::default();
            assert_eq!(cfg.overlap, crate::engine::intent_bias::OverlapHint::Auto);
            assert_eq!(cfg.compress, CommCompressHint::Auto);
            assert!(cfg.algorithm_override.is_empty());
        }

        #[test]
        fn comm_config_custom_values() {
            let cfg = CommConfig {
                overlap: crate::engine::intent_bias::OverlapHint::PreferOverlap,
                compress: CommCompressHint::AlwaysCompress,
                algorithm_override: "custom_algo".to_string(),
            };
            assert_eq!(cfg.overlap, crate::engine::intent_bias::OverlapHint::PreferOverlap);
            assert_eq!(cfg.compress, CommCompressHint::AlwaysCompress);
            assert_eq!(cfg.algorithm_override, "custom_algo");
        }

        #[test]
        fn comm_config_equality() {
            let a = CommConfig::default();
            let b = CommConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn comm_config_clone_independence() {
            let mut cfg = CommConfig {
                overlap: crate::engine::intent_bias::OverlapHint::ForceDoubleBuffer,
                compress: CommCompressHint::ForceQuant,
                algorithm_override: "test".to_string(),
            };
            let cloned = cfg.clone();
            cfg.compress = CommCompressHint::NeverCompress;
            assert_eq!(cloned.compress, CommCompressHint::ForceQuant);
        }

        // ── MoeDistributedConfig (REQ-IB-011) ──────────────────────────────

        #[test]
        fn moe_distributed_config_default_values() {
            let cfg = MoeDistributedConfig::default();
            assert_eq!(cfg.expert_placement, ExpertPlacement::Auto);
            assert_eq!(cfg.all_to_all, AllToAllStrategy::Auto);
        }

        #[test]
        fn moe_distributed_config_custom_values() {
            let cfg = MoeDistributedConfig {
                expert_placement: ExpertPlacement::HotCold,
                all_to_all: AllToAllStrategy::NvlinkAllToAll,
            };
            assert_eq!(cfg.expert_placement, ExpertPlacement::HotCold);
            assert_eq!(cfg.all_to_all, AllToAllStrategy::NvlinkAllToAll);
        }

        #[test]
        fn moe_distributed_config_equality() {
            let a = MoeDistributedConfig::default();
            let b = MoeDistributedConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn moe_distributed_config_clone_independence() {
            let mut cfg = MoeDistributedConfig {
                expert_placement: ExpertPlacement::RoundRobin,
                all_to_all: AllToAllStrategy::HierarchicalAllToAll,
            };
            let cloned = cfg.clone();
            cfg.expert_placement = ExpertPlacement::Custom;
            assert_eq!(cloned.expert_placement, ExpertPlacement::RoundRobin);
        }

        // ── DistributedConfig (REQ-IB-006) ─────────────────────────────────

        #[test]
        fn distributed_config_default_is_single_node() {
            let cfg = DistributedConfig::default();
            assert_eq!(cfg.parallel, ParallelConfig::default());
            assert_eq!(cfg.pd_disagg, PdDisaggConfig::default());
            assert_eq!(cfg.kv_distribution, KvDistributionConfig::default());
            assert_eq!(cfg.comm, CommConfig::default());
            assert_eq!(cfg.moe, MoeDistributedConfig::default());
        }

        #[test]
        fn distributed_config_equality() {
            let a = DistributedConfig::default();
            let b = DistributedConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn distributed_config_inequality_parallel() {
            let a = DistributedConfig::default();
            let b = DistributedConfig {
                parallel: ParallelConfig {
                    tp_size: 2,
                    ..Default::default()
                },
                ..Default::default()
            };
            assert_ne!(a, b);
        }

        #[test]
        fn distributed_config_clone_independence() {
            let mut cfg = DistributedConfig::default();
            let cloned = cfg.clone();
            cfg.parallel.tp_size = 4;
            assert_eq!(cloned.parallel.tp_size, 1);
        }
    }
}
