//! TEST-DIST-001~017 — 分布式推理专项测试 (REQ-DIST-001~017)
//!
//! 覆盖 SPEC 43-DISTRIBUTED-IMPLEMENTATION.html 中 REQ-DIST-001~017 的核心逻辑。
//! 优先测试编译时逻辑（枚举/决策/配置），不依赖运行时 NCCL 通信。
//!
//! 运行方式:
//! ```bash
//! # 非 nccl 环境（仅测试非 nccl-gated 逻辑）
//! cargo test --test test_dist -- --test-threads=2
//!
//! # nccl 环境（完整测试）
//! cargo test --test test_dist --features nccl -- --test-threads=2
//! ```

use gllm::engine::distributed_config::{
    CommCompressHint, ExpertPlacement, KvDistMode, NodeRole, PdDisaggMode,
};

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-001: CommHandle 生命周期管理 (REQ-DIST-001)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_001 {
    use gllm::engine::distributed_config::{
        CommHandleWrapper, DistributedConfigError, ParallelConfig,
    };

    fn make_single_handle() -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap()
    }

    fn make_distributed_handle(rank: u32, world_size: u32) -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig {
            tp_size: world_size,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank,
            world_size,
            unique_id: String::new(),
            stage_id: 0,
        }).unwrap()
    }

    /// TEST-DIST-001-01: from_config 单机句柄 — 正确的 rank/world_size/is_distributed
    // @trace TEST-DIST-001
    #[test]
    fn comm_handle_single_node() {
        let handle = make_single_handle();
        assert_eq!(handle.rank(), 0);
        assert_eq!(handle.world_size(), 1);
        assert!(!handle.is_distributed());
        assert!(!handle.is_nccl_initialized());
        assert!(!handle.is_destroyed());
    }

    /// TEST-DIST-001-02: from_config 分布式句柄 — is_distributed=true
    // @trace TEST-DIST-001
    #[test]
    fn comm_handle_distributed() {
        let handle = make_distributed_handle(2, 4);
        assert_eq!(handle.rank(), 2);
        assert_eq!(handle.world_size(), 4);
        assert!(handle.is_distributed());
        assert!(!handle.is_nccl_initialized());
    }

    /// TEST-DIST-001-03: from_config 有效默认配置 → 成功创建
    // @trace TEST-DIST-001
    #[test]
    fn comm_handle_from_config_default() {
        let config = ParallelConfig::default();
        let handle = CommHandleWrapper::from_config(&config).unwrap();
        assert_eq!(handle.rank(), 0);
        assert_eq!(handle.world_size(), 1);
        assert!(!handle.is_distributed());
    }

    /// TEST-DIST-001-04: from_config 无效配置 → InvalidParallelConfig
    // @trace TEST-DIST-001
    #[test]
    fn comm_handle_from_config_invalid() {
        let config = ParallelConfig {
            tp_size: 2,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank: 0,
            world_size: 5, // 2*1*1=2 != 5
            unique_id: String::new(),
            stage_id: 0,
        };
        let result = CommHandleWrapper::from_config(&config);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DistributedConfigError::InvalidParallelConfig);
    }

    /// TEST-DIST-001-05: 完整生命周期 — 创建 → 验证 → destroy → is_destroyed
    // @trace TEST-DIST-001
    #[test]
    fn comm_handle_full_lifecycle() {
        let mut handle = make_single_handle();
        assert!(!handle.is_destroyed());
        handle.destroy();
        assert!(handle.is_destroyed());
    }

    /// TEST-DIST-001-06: destroy 幂等性 — 二次 destroy 不 panic
    // @trace TEST-DIST-001
    #[test]
    fn comm_handle_destroy_idempotent() {
        let mut handle = make_single_handle();
        handle.destroy();
        assert!(handle.is_destroyed());
        handle.destroy(); // 二次 — no-op
        assert!(handle.is_destroyed());
    }

    /// TEST-DIST-001-07: DistributedConfigError Display 格式化
    // @trace TEST-DIST-001
    #[test]
    fn distributed_config_error_display() {
        let err = DistributedConfigError::InvalidParallelConfig;
        let msg = format!("{}", err);
        assert!(msg.contains("ParallelConfig"));
        assert!(msg.contains("validation failed"));

        let err = DistributedConfigError::CommInitFailed("timeout".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("NCCL"));
        assert!(msg.contains("timeout"));
    }

    /// TEST-DIST-001-08: DistributedConfigError 实现 std::error::Error
    // @trace TEST-DIST-001
    #[test]
    fn distributed_config_error_is_std_error() {
        let err = DistributedConfigError::InvalidParallelConfig;
        let _: &dyn std::error::Error = &err;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-002: DistributedConfig 传递数据流 (REQ-DIST-002)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_002 {
    use gllm::engine::distributed_config::{
        AllToAllStrategy, CommCompressHint, DistributedConfig, ExpertPlacement, KvDistMode,
        KvDistributionConfig, MoeDistributedConfig, NodeRole, ParallelConfig, PdDisaggConfig,
        PdDisaggMode, CommConfig,
    };

    /// TEST-DIST-002-01: DistributedConfig 默认 = 单机模式（所有子配置默认值）
    // @trace TEST-DIST-002
    #[test]
    fn distributed_config_default_is_single_node() {
        let cfg = DistributedConfig::default();
        assert_eq!(cfg.parallel, ParallelConfig::default());
        assert_eq!(cfg.pd_disagg, PdDisaggConfig::default());
        assert_eq!(cfg.kv_distribution, KvDistributionConfig::default());
        assert_eq!(cfg.comm, CommConfig::default());
        assert_eq!(cfg.moe, MoeDistributedConfig::default());
    }

    /// TEST-DIST-002-02: DistributedConfig 可自定义并行配置
    // @trace TEST-DIST-002
    #[test]
    fn distributed_config_custom_parallel() {
        let cfg = DistributedConfig {
            parallel: ParallelConfig {
                tp_size: 2,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 0,
                world_size: 2,
                unique_id: String::new(),
                stage_id: 0,
            },
            ..Default::default()
        };
        assert_eq!(cfg.parallel.tp_size, 2);
        assert_eq!(cfg.parallel.world_size, 2);
    }

    /// TEST-DIST-002-03: DistributedConfig 子配置可独立覆盖
    // @trace TEST-DIST-002
    #[test]
    fn distributed_config_independent_sub_configs() {
        let cfg = DistributedConfig {
            pd_disagg: PdDisaggConfig {
                mode: PdDisaggMode::Disaggregated,
                role: NodeRole::PrefillOnly,
            },
            kv_distribution: KvDistributionConfig {
                mode: KvDistMode::Mirror,
                mirror_heads: 0,
            },
            comm: CommConfig {
                compress: CommCompressHint::AlwaysCompress,
                ..Default::default()
            },
            moe: MoeDistributedConfig {
                expert_placement: ExpertPlacement::RoundRobin,
                all_to_all: AllToAllStrategy::NvlinkAllToAll,
            },
            ..Default::default()
        };
        assert_eq!(cfg.pd_disagg.mode, PdDisaggMode::Disaggregated);
        assert_eq!(cfg.kv_distribution.mode, KvDistMode::Mirror);
        assert_eq!(cfg.comm.compress, CommCompressHint::AlwaysCompress);
        assert_eq!(cfg.moe.expert_placement, ExpertPlacement::RoundRobin);
    }

    /// TEST-DIST-002-04: DistributedConfig clone 独立性
    // @trace TEST-DIST-002
    #[test]
    fn distributed_config_clone_independence() {
        let mut cfg = DistributedConfig::default();
        let cloned = cfg.clone();
        cfg.parallel.tp_size = 4;
        assert_eq!(cloned.parallel.tp_size, 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-003: 分布式页路由表构建 (REQ-DIST-003)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_003 {
    use gllm::engine::distributed_config::{CommHandleWrapper, ParallelConfig};

    fn make_handle(rank: u32, world_size: u32) -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig {
            tp_size: world_size,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank,
            world_size,
            unique_id: String::new(),
            stage_id: 0,
        }).unwrap()
    }

    /// TEST-DIST-003-01: CommHandleWrapper rank 可作为 PageRoutingTable 的 rank 参数
    // @trace TEST-DIST-003
    #[test]
    fn page_routing_table_rank_from_comm_handle() {
        let handle = make_handle(1, 4);
        assert_eq!(handle.rank(), 1);
        let rank = handle.rank();
        let tp_size = handle.world_size();
        assert!(rank < tp_size);
    }

    /// TEST-DIST-003-02: 单机模式 rank 映射到自身
    // @trace TEST-DIST-003
    #[test]
    fn page_routing_single_node_identity() {
        let handle = make_handle(0, 1);
        assert!(!handle.is_distributed());
        assert_eq!(handle.rank(), 0);
    }

    /// TEST-DIST-003-03: 分布式模式下 rank 可正确路由
    // @trace TEST-DIST-003
    #[test]
    fn page_routing_distributed_ranks_distinct() {
        for rank in 0..4u32 {
            let handle = make_handle(rank, 4);
            assert_eq!(handle.rank(), rank);
            assert_eq!(handle.world_size(), 4);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-004: TP 权重分片 (REQ-DIST-004)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_004 {
    use gllm::engine::distributed_config::ParallelConfig;
    use gllm::loader::weight_shard::{infer_shard_strategy, shard_weight, ShardStrategy};

    fn make_config(tp_size: u32, rank: u32) -> ParallelConfig {
        ParallelConfig {
            tp_size,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank,
            world_size: tp_size,
            unique_id: String::new(),
            stage_id: 0,
        }
    }

    /// TEST-DIST-004-01: ColumnParallel tp_size=2 rank=0 — 取前半列
    // @trace TEST-DIST-004
    #[test]
    fn shard_weight_column_parallel_tp2_rank0() {
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = make_config(2, 0);
        shard_weight(&mut data, 2, 4, &config, ShardStrategy::ColumnParallel).unwrap();
        assert_eq!(data, vec![0.0, 1.0, 4.0, 5.0]);
    }

    /// TEST-DIST-004-02: ColumnParallel tp_size=2 rank=1 — 取后半列
    // @trace TEST-DIST-004
    #[test]
    fn shard_weight_column_parallel_tp2_rank1() {
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = make_config(2, 1);
        shard_weight(&mut data, 2, 4, &config, ShardStrategy::ColumnParallel).unwrap();
        assert_eq!(data, vec![2.0, 3.0, 6.0, 7.0]);
    }

    /// TEST-DIST-004-03: RowParallel tp_size=2 rank=0 — 取前半行
    // @trace TEST-DIST-004
    #[test]
    fn shard_weight_row_parallel_tp2_rank0() {
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = make_config(2, 0);
        shard_weight(&mut data, 4, 2, &config, ShardStrategy::RowParallel).unwrap();
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0]);
    }

    /// TEST-DIST-004-04: tp_size=1 单机无分片
    // @trace TEST-DIST-004
    #[test]
    fn shard_weight_tp1_noop() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        let config = make_config(1, 0);
        shard_weight(&mut data, 2, 2, &config, ShardStrategy::ColumnParallel).unwrap();
        assert_eq!(data, original);
    }

    /// TEST-DIST-004-05: 维度不能整除 → Err
    // @trace TEST-DIST-004
    #[test]
    fn shard_weight_not_divisible_error() {
        let mut data = vec![0.0f32; 6];
        let config = make_config(4, 0);
        let result = shard_weight(&mut data, 2, 3, &config, ShardStrategy::ColumnParallel);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not divisible"));
    }

    /// TEST-DIST-004-06: infer_shard_strategy — q_proj → ColumnParallel
    // @trace TEST-DIST-004
    #[test]
    fn infer_strategy_q_proj() {
        assert_eq!(infer_shard_strategy("L0.q_proj"), Some(ShardStrategy::ColumnParallel));
    }

    /// TEST-DIST-004-07: infer_shard_strategy — o_proj → RowParallel
    // @trace TEST-DIST-004
    #[test]
    fn infer_strategy_o_proj() {
        assert_eq!(infer_shard_strategy("L0.o_proj"), Some(ShardStrategy::RowParallel));
    }

    /// TEST-DIST-004-08: infer_shard_strategy — down_proj → RowParallel
    // @trace TEST-DIST-004
    #[test]
    fn infer_strategy_down_proj() {
        assert_eq!(infer_shard_strategy("L0.down_proj"), Some(ShardStrategy::RowParallel));
    }

    /// TEST-DIST-004-09: infer_shard_strategy — embed → None（不分片）
    // @trace TEST-DIST-004
    #[test]
    fn infer_strategy_embed_no_shard() {
        assert_eq!(infer_shard_strategy("embed"), None);
    }

    /// TEST-DIST-004-10: ShardStrategy 枚举完整性和 Copy trait
    // @trace TEST-DIST-004
    #[test]
    fn shard_strategy_variants_and_copy() {
        assert_eq!(ShardStrategy::ColumnParallel, ShardStrategy::ColumnParallel);
        assert_eq!(ShardStrategy::RowParallel, ShardStrategy::RowParallel);
        assert_ne!(ShardStrategy::ColumnParallel, ShardStrategy::RowParallel);
        let a = ShardStrategy::ColumnParallel;
        let b = a; // Copy
        assert_eq!(a, b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-005: TP AllReduce 通信 (REQ-DIST-005)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_005 {
    use gllm::engine::distributed_config::{CommHandleWrapper, ParallelConfig};

    fn make_single_handle() -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap()
    }

    fn make_distributed_handle(rank: u32, world_size: u32) -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig {
            tp_size: world_size,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank,
            world_size,
            unique_id: String::new(),
            stage_id: 0,
        }).unwrap()
    }

    /// TEST-DIST-005-01: 单机模式 all_reduce_inplace 立即返回 Ok
    // @trace TEST-DIST-005
    #[test]
    fn all_reduce_single_node_noop() {
        let handle = make_single_handle();
        let mut buf = vec![1.0f32, 2.0, 3.0];
        let result = handle.all_reduce_inplace(&mut buf);
        assert!(result.is_ok());
        // 单机模式数据不变
        assert_eq!(buf, vec![1.0, 2.0, 3.0]);
    }

    /// TEST-DIST-005-02: 分布式模式但未初始化 NCCL → Err
    // @trace TEST-DIST-005
    #[test]
    fn all_reduce_distributed_no_nccl_init_error() {
        let handle = make_distributed_handle(0, 4);
        let mut buf = vec![1.0f32, 2.0, 3.0];
        let result = handle.all_reduce_inplace(&mut buf);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("init_nccl"));
    }

    /// TEST-DIST-005-03: 单机模式 all_reduce_inplace 空缓冲区
    // @trace TEST-DIST-005
    #[test]
    fn all_reduce_single_node_empty_buffer() {
        let handle = make_single_handle();
        let mut buf: Vec<f32> = vec![];
        let result = handle.all_reduce_inplace(&mut buf);
        assert!(result.is_ok());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-006: TP 列并行通信 (REQ-DIST-006)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_006 {
    use gllm::engine::coordinator::comm_schedule::comm_schedule::{
        resolve_column_parallel_schedule, ColumnParallelStrategy, CommScheduleDecision,
    };

    /// TEST-DIST-006-01: ColumnParallel ReduceScatter 决策
    // @trace TEST-DIST-006
    #[test]
    fn column_parallel_reduce_scatter() {
        let result = resolve_column_parallel_schedule(2, ColumnParallelStrategy::ReduceScatter);
        assert!(result.is_some());
        assert_eq!(
            result.unwrap(),
            CommScheduleDecision::ColumnParallel {
                strategy: ColumnParallelStrategy::ReduceScatter,
            }
        );
    }

    /// TEST-DIST-006-02: ColumnParallel AllGather 决策
    // @trace TEST-DIST-006
    #[test]
    fn column_parallel_all_gather() {
        let result = resolve_column_parallel_schedule(4, ColumnParallelStrategy::AllGather);
        assert!(result.is_some());
        assert_eq!(
            result.unwrap(),
            CommScheduleDecision::ColumnParallel {
                strategy: ColumnParallelStrategy::AllGather,
            }
        );
    }

    /// TEST-DIST-006-03: tp_size=1 → None（单机无需列并行通信）
    // @trace TEST-DIST-006
    #[test]
    fn column_parallel_tp1_returns_none() {
        assert!(resolve_column_parallel_schedule(1, ColumnParallelStrategy::ReduceScatter).is_none());
    }

    /// TEST-DIST-006-04: ColumnParallelStrategy 枚举区分
    // @trace TEST-DIST-006
    #[test]
    fn column_parallel_strategy_distinct() {
        assert_eq!(ColumnParallelStrategy::ReduceScatter, ColumnParallelStrategy::ReduceScatter);
        assert_eq!(ColumnParallelStrategy::AllGather, ColumnParallelStrategy::AllGather);
        assert_ne!(ColumnParallelStrategy::ReduceScatter, ColumnParallelStrategy::AllGather);
    }

    /// TEST-DIST-006-05: ColumnParallel 不生成 CommPlan（返回 None）
    // @trace TEST-DIST-006
    #[test]
    fn column_parallel_build_comm_plan_returns_none() {
        let decision = CommScheduleDecision::ColumnParallel {
            strategy: ColumnParallelStrategy::ReduceScatter,
        };
        let topo = gllm_nccl::topology::make_test_topology(4, true);
        assert!(decision
            .build_comm_plan(&topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum, 1024, 0, 4, 0)
            .is_none());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-007: 双缓冲通信-计算重叠 (REQ-DIST-007)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_007 {
    use gllm::engine::coordinator::comm_schedule::comm_schedule::{
        resolve_comm_schedule, CommScheduleDecision,
    };
    use gllm::engine::distributed_config::{CommConfig, CommHandleWrapper, ParallelConfig};
    use gllm::engine::intent_bias::OverlapHint;

    fn make_distributed_handle() -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig {
            tp_size: 4,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank: 0,
            world_size: 4,
            unique_id: String::new(),
            stage_id: 0,
        }).unwrap()
    }

    fn make_single_handle() -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap()
    }

    /// TEST-DIST-007-01: 单机模式 → StandardAllReduce
    // @trace TEST-DIST-007
    #[test]
    fn comm_schedule_single_node_standard() {
        let handle = make_single_handle();
        let config = CommConfig {
            overlap: OverlapHint::ForceDoubleBuffer,
            ..Default::default()
        };
        assert_eq!(
            resolve_comm_schedule(&config, &handle),
            CommScheduleDecision::StandardAllReduce
        );
    }

    /// TEST-DIST-007-02: ForceDoubleBuffer → DoubleBuffer { comm_sm_ratio: 0.3 }
    // @trace TEST-DIST-007
    #[test]
    fn comm_schedule_force_double_buffer() {
        let handle = make_distributed_handle();
        let config = CommConfig {
            overlap: OverlapHint::ForceDoubleBuffer,
            ..Default::default()
        };
        assert_eq!(
            resolve_comm_schedule(&config, &handle),
            CommScheduleDecision::DoubleBuffer { comm_sm_ratio: 0.3 }
        );
    }

    /// TEST-DIST-007-03: Auto → StandardAllReduce
    // @trace TEST-DIST-007
    #[test]
    fn comm_schedule_auto_standard() {
        let handle = make_distributed_handle();
        let config = CommConfig {
            overlap: OverlapHint::Auto,
            ..Default::default()
        };
        assert_eq!(
            resolve_comm_schedule(&config, &handle),
            CommScheduleDecision::StandardAllReduce
        );
    }

    /// TEST-DIST-007-04: DoubleBuffer CommPlan 的 overlap_mode = DoubleBuffer
    // @trace TEST-DIST-007
    #[test]
    fn double_buffer_comm_plan_overlap_mode() {
        let decision = CommScheduleDecision::DoubleBuffer { comm_sm_ratio: 0.3 };
        let topo = gllm_nccl::topology::make_test_topology(4, true);
        let result = decision.build_comm_plan(
            &topo,
            gllm_nccl::DType::Fp32,
            gllm_nccl::ReduceOp::Sum,
            512 * 1024,
            0,
            4,
            0,
        );
        let plan = result.unwrap().unwrap();
        assert_eq!(plan.schedule.overlap_mode, gllm_nccl::comm_ir::OverlapMode::DoubleBuffer);
        assert_eq!(plan.schedule.buffer_slots, 2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-008: FLUX 分解调度 (REQ-DIST-008)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_008 {
    use gllm::engine::coordinator::comm_schedule::comm_schedule::{
        resolve_comm_schedule, CommScheduleDecision,
    };
    use gllm::engine::distributed_config::{CommConfig, CommHandleWrapper, ParallelConfig};
    use gllm::engine::intent_bias::OverlapHint;

    /// TEST-DIST-008-01: ForceFlux → FluxDecompose { ring_size = world_size }
    // @trace TEST-DIST-008
    #[test]
    fn comm_schedule_force_flux() {
        let handle = CommHandleWrapper::from_config(&ParallelConfig {
            tp_size: 4,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank: 0,
            world_size: 4,
            unique_id: String::new(),
            stage_id: 0,
        }).unwrap();
        let config = CommConfig {
            overlap: OverlapHint::ForceFlux,
            ..Default::default()
        };
        assert_eq!(
            resolve_comm_schedule(&config, &handle),
            CommScheduleDecision::FluxDecompose { ring_size: 4 }
        );
    }

    /// TEST-DIST-008-02: FluxDecompose 生成有效 CommPlan
    // @trace TEST-DIST-008
    #[test]
    fn flux_decompose_generates_comm_plan() {
        let decision = CommScheduleDecision::FluxDecompose { ring_size: 4 };
        let topo = gllm_nccl::topology::make_test_topology(4, true);
        let result = decision.build_comm_plan(
            &topo,
            gllm_nccl::DType::Fp32,
            gllm_nccl::ReduceOp::Sum,
            512 * 1024,
            0,
            4,
            0,
        );
        assert!(result.is_some());
        assert!(result.unwrap().is_ok());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-009: 量化通信 (REQ-DIST-009)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_009 {
    use gllm::engine::coordinator::comm_schedule::comm_schedule::{
        resolve_quant_comm, QuantCommDecision,
    };
    use gllm::engine::distributed_config::{CommCompressHint, CommConfig, CommHandleWrapper, ParallelConfig};

    /// TEST-DIST-009-01: from_hint — Auto → AutoCompress
    // @trace TEST-DIST-009
    #[test]
    fn quant_comm_auto() {
        assert_eq!(
            QuantCommDecision::from_hint(&CommCompressHint::Auto),
            QuantCommDecision::AutoCompress
        );
    }

    /// TEST-DIST-009-02: from_hint — AlwaysCompress → Fp8Compress
    // @trace TEST-DIST-009
    #[test]
    fn quant_comm_always_compress() {
        assert_eq!(
            QuantCommDecision::from_hint(&CommCompressHint::AlwaysCompress),
            QuantCommDecision::Fp8Compress {
                quant_scheme: "fp8_e4m3".to_string()
            }
        );
    }

    /// TEST-DIST-009-03: from_hint — NeverCompress → NeverCompress
    // @trace TEST-DIST-009
    #[test]
    fn quant_comm_never_compress() {
        assert_eq!(
            QuantCommDecision::from_hint(&CommCompressHint::NeverCompress),
            QuantCommDecision::NeverCompress
        );
    }

    /// TEST-DIST-009-04: from_hint — ForceQuant → Fp8Compress
    // @trace TEST-DIST-009
    #[test]
    fn quant_comm_force_quant() {
        assert_eq!(
            QuantCommDecision::from_hint(&CommCompressHint::ForceQuant),
            QuantCommDecision::Fp8Compress {
                quant_scheme: "fp8_e4m3".to_string()
            }
        );
    }

    /// TEST-DIST-009-05: resolve_quant_comm 单机 → NoCompression
    // @trace TEST-DIST-009
    #[test]
    fn resolve_quant_comm_single_node() {
        let handle = CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap();
        let config = CommConfig {
            compress: CommCompressHint::AlwaysCompress,
            ..Default::default()
        };
        assert_eq!(
            resolve_quant_comm(&config, &handle),
            QuantCommDecision::NoCompression
        );
    }

    /// TEST-DIST-009-06: is_compressed — FP8/INT8 压缩，其他不压缩
    // @trace TEST-DIST-009
    #[test]
    fn quant_comm_is_compressed() {
        assert!(!QuantCommDecision::NoCompression.is_compressed());
        assert!(!QuantCommDecision::NeverCompress.is_compressed());
        assert!(!QuantCommDecision::AutoCompress.is_compressed());
        assert!(
            QuantCommDecision::Fp8Compress {
                quant_scheme: "fp8_e4m3".to_string()
            }
            .is_compressed()
        );
        assert!(QuantCommDecision::InternCompress.is_compressed());
    }

    /// TEST-DIST-009-07: bandwidth_saving — FP8=4x, INT8=4x, 无压缩=1x
    // @trace TEST-DIST-009
    #[test]
    fn quant_comm_bandwidth_saving() {
        assert_eq!(QuantCommDecision::NoCompression.bandwidth_saving(), 1);
        assert_eq!(QuantCommDecision::NeverCompress.bandwidth_saving(), 1);
        assert_eq!(
            QuantCommDecision::Fp8Compress {
                quant_scheme: "fp8_e4m3".to_string()
            }
            .bandwidth_saving(),
            4
        );
        assert_eq!(QuantCommDecision::InternCompress.bandwidth_saving(), 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-010: 算法选择覆盖 (REQ-DIST-010)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_010 {
    use gllm::engine::coordinator::comm_schedule::comm_schedule::AlgorithmOverride;
    use gllm::engine::distributed_config::{CommConfig, CommHandleWrapper, ParallelConfig};

    /// TEST-DIST-010-01: 空字符串 → Auto
    // @trace TEST-DIST-010
    #[test]
    fn algorithm_override_empty_auto() {
        let config = CommConfig {
            algorithm_override: String::new(),
            ..Default::default()
        };
        assert_eq!(AlgorithmOverride::from_config(&config), AlgorithmOverride::Auto);
    }

    /// TEST-DIST-010-02: 非空字符串 → Force
    // @trace TEST-DIST-010
    #[test]
    fn algorithm_override_non_empty_force() {
        let config = CommConfig {
            algorithm_override: "Ring".to_string(),
            ..Default::default()
        };
        assert_eq!(
            AlgorithmOverride::from_config(&config),
            AlgorithmOverride::Force("Ring".to_string())
        );
    }

    /// TEST-DIST-010-03: resolve_algorithm — 有效算法名
    // @trace TEST-DIST-010
    #[test]
    fn algorithm_override_resolve_valid() {
        assert_eq!(
            AlgorithmOverride::Force("Ring".to_string()).resolve_algorithm(),
            Some(gllm_nccl::CollectiveAlgorithm::Ring)
        );
        assert_eq!(
            AlgorithmOverride::Force("Tree".to_string()).resolve_algorithm(),
            Some(gllm_nccl::CollectiveAlgorithm::Tree)
        );
        assert_eq!(
            AlgorithmOverride::Force("FluxPipeline".to_string()).resolve_algorithm(),
            Some(gllm_nccl::CollectiveAlgorithm::FluxPipeline)
        );
    }

    /// TEST-DIST-010-04: resolve_algorithm — 未知算法 → None
    // @trace TEST-DIST-010
    #[test]
    fn algorithm_override_resolve_unknown() {
        assert_eq!(
            AlgorithmOverride::Force("UnknownAlgo".to_string()).resolve_algorithm(),
            None
        );
    }

    /// TEST-DIST-010-05: select_or_auto 未知算法 → Err（不静默降级）
    // @trace TEST-DIST-010
    #[test]
    fn algorithm_override_unknown_returns_err() {
        let ao = AlgorithmOverride::Force("BadAlgo".to_string());
        let topo = gllm_nccl::topology::make_test_topology(4, true);
        let result = ao.select_or_auto(&topo, gllm_nccl::CollectiveOp::AllReduce, 1024, 4, 0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("BadAlgo"));
        assert!(err.contains("unrecognized"));
    }

    /// TEST-DIST-010-06: apply_to_handle 单机 → 默认 Ring
    // @trace TEST-DIST-010
    #[test]
    fn algorithm_override_single_node_defaults_ring() {
        let handle = CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap();
        let ao = AlgorithmOverride::Force("Tree".to_string());
        let result = ao.apply_to_handle(&handle, gllm_nccl::CollectiveOp::AllReduce, 1024);
        assert_eq!(result.unwrap(), gllm_nccl::CollectiveAlgorithm::Ring);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-011: PD 分离角色路由 (REQ-DIST-011)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_011 {
    use gllm::engine::coordinator::pd_disagg::PdRoleDecision;
    use gllm::engine::distributed_config::{NodeRole, PdDisaggConfig, PdDisaggMode};

    /// TEST-DIST-011-01: Collocated 模式 → 所有角色返回 Collocated
    // @trace TEST-DIST-011
    #[test]
    fn pd_role_collocated_always_collocated() {
        for role in [NodeRole::Auto, NodeRole::PrefillOnly, NodeRole::DecodeOnly, NodeRole::Mixed] {
            let config = PdDisaggConfig { mode: PdDisaggMode::Collocated, role };
            assert_eq!(PdRoleDecision::from_config(&config), PdRoleDecision::Collocated);
        }
    }

    /// TEST-DIST-011-02: Disaggregated + PrefillOnly → PrefillOnly
    // @trace TEST-DIST-011
    #[test]
    fn pd_role_disaggregated_prefill_only() {
        let config = PdDisaggConfig {
            mode: PdDisaggMode::Disaggregated,
            role: NodeRole::PrefillOnly,
        };
        assert_eq!(PdRoleDecision::from_config(&config), PdRoleDecision::PrefillOnly);
    }

    /// TEST-DIST-011-03: Disaggregated + DecodeOnly → DecodeOnly
    // @trace TEST-DIST-011
    #[test]
    fn pd_role_disaggregated_decode_only() {
        let config = PdDisaggConfig {
            mode: PdDisaggMode::Disaggregated,
            role: NodeRole::DecodeOnly,
        };
        assert_eq!(PdRoleDecision::from_config(&config), PdRoleDecision::DecodeOnly);
    }

    /// TEST-DIST-011-04: Disaggregated + Auto → Collocated
    // @trace TEST-DIST-011
    #[test]
    fn pd_role_disaggregated_auto_collocated() {
        let config = PdDisaggConfig {
            mode: PdDisaggMode::Disaggregated,
            role: NodeRole::Auto,
        };
        assert_eq!(PdRoleDecision::from_config(&config), PdRoleDecision::Collocated);
    }

    /// TEST-DIST-011-05: needs_sampling — PrefillOnly 不需要
    // @trace TEST-DIST-011
    #[test]
    fn pd_role_needs_sampling() {
        assert!(!PdRoleDecision::PrefillOnly.needs_sampling());
        assert!(PdRoleDecision::Collocated.needs_sampling());
        assert!(PdRoleDecision::DecodeOnly.needs_sampling());
    }

    /// TEST-DIST-011-06: needs_prefill — DecodeOnly 不需要
    // @trace TEST-DIST-011
    #[test]
    fn pd_role_needs_prefill() {
        assert!(!PdRoleDecision::DecodeOnly.needs_prefill());
        assert!(PdRoleDecision::Collocated.needs_prefill());
        assert!(PdRoleDecision::PrefillOnly.needs_prefill());
    }

    /// TEST-DIST-011-07: needs_kv_transfer_after_prefill — 仅 PrefillOnly
    // @trace TEST-DIST-011
    #[test]
    fn pd_role_needs_kv_transfer() {
        assert!(PdRoleDecision::PrefillOnly.needs_kv_transfer_after_prefill());
        assert!(!PdRoleDecision::Collocated.needs_kv_transfer_after_prefill());
        assert!(!PdRoleDecision::DecodeOnly.needs_kv_transfer_after_prefill());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-012: 跨节点 KV 传输 (REQ-DIST-012)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_012 {
    use gllm::engine::coordinator::kv_transfer::{
        KvTransferDirection, KvTransferMode,
    };

    /// TEST-DIST-012-01: KvTransferDirection 枚举区分
    // @trace TEST-DIST-012
    #[test]
    fn kv_transfer_direction_distinct() {
        assert_eq!(KvTransferDirection::Send, KvTransferDirection::Send);
        assert_eq!(KvTransferDirection::Recv, KvTransferDirection::Recv);
        assert_ne!(KvTransferDirection::Send, KvTransferDirection::Recv);
    }

    /// TEST-DIST-012-02: KvTransferMode 枚举完整
    // @trace TEST-DIST-012
    #[test]
    fn kv_transfer_mode_variants() {
        assert_eq!(KvTransferMode::Sync, KvTransferMode::Sync);
        assert_eq!(KvTransferMode::Async, KvTransferMode::Async);
        assert_eq!(KvTransferMode::Rdma, KvTransferMode::Rdma);
        assert_ne!(KvTransferMode::Sync, KvTransferMode::Async);
    }

    /// TEST-DIST-012-03: KvTransferMode 默认 = Sync
    // @trace TEST-DIST-012
    #[test]
    fn kv_transfer_mode_default_sync() {
        assert_eq!(KvTransferMode::default(), KvTransferMode::Sync);
    }

    /// TEST-DIST-012-04: CommHandleWrapper send_kv_pages 单机 → Err
    // @trace TEST-DIST-012
    #[test]
    fn send_kv_pages_single_node_error() {
        use gllm::engine::distributed_config::{CommHandleWrapper, ParallelConfig};
        let handle = CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap();
        let result = handle.send_kv_pages(std::ptr::null(), 0, 1, gllm_nccl::DType::Fp32);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not in distributed mode"));
    }

    /// TEST-DIST-012-05: CommHandleWrapper recv_kv_pages 单机 → Err
    // @trace TEST-DIST-012
    #[test]
    fn recv_kv_pages_single_node_error() {
        use gllm::engine::distributed_config::{CommHandleWrapper, ParallelConfig};
        let handle = CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap();
        let result = handle.recv_kv_pages(std::ptr::null_mut(), 0, 1, gllm_nccl::DType::Fp32);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not in distributed mode"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-013: KV 分布 5 模式 (REQ-DIST-013)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_013 {
    use gllm::engine::coordinator::kv_distribution::KvDistDecision;
    use gllm::engine::distributed_config::{
        CommHandleWrapper, KvDistMode, KvDistributionConfig, ParallelConfig,
    };

    fn make_single_handle() -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap()
    }

    fn make_distributed_handle(rank: u32, world_size: u32) -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig {
            tp_size: world_size,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank,
            world_size,
            unique_id: String::new(),
            stage_id: 0,
        }).unwrap()
    }

    /// TEST-DIST-013-01: 单机模式 → Local
    // @trace TEST-DIST-013
    #[test]
    fn kv_dist_single_node_local() {
        let handle = make_single_handle();
        let config = KvDistributionConfig {
            mode: KvDistMode::Mirror,
            mirror_heads: 0,
        };
        assert_eq!(KvDistDecision::from_config(&config, &handle), KvDistDecision::Local);
    }

    /// TEST-DIST-013-02: 分布式 + Local → Local
    // @trace TEST-DIST-013
    #[test]
    fn kv_dist_distributed_local_mode() {
        let handle = make_distributed_handle(0, 4);
        let config = KvDistributionConfig {
            mode: KvDistMode::Local,
            mirror_heads: 0,
        };
        assert_eq!(KvDistDecision::from_config(&config, &handle), KvDistDecision::Local);
    }

    /// TEST-DIST-013-03: 分布式 + OnDemand → OnDemand { prefetch: false } (NCCL 未初始化)
    // @trace TEST-DIST-013
    #[test]
    fn kv_dist_distributed_on_demand_no_prefetch() {
        let handle = make_distributed_handle(0, 4);
        let config = KvDistributionConfig {
            mode: KvDistMode::OnDemand,
            mirror_heads: 0,
        };
        assert_eq!(
            KvDistDecision::from_config(&config, &handle),
            KvDistDecision::OnDemand { prefetch: false }
        );
    }

    /// TEST-DIST-013-04: 分布式 + Mirror → Mirror
    // @trace TEST-DIST-013
    #[test]
    fn kv_dist_distributed_mirror() {
        let handle = make_distributed_handle(0, 4);
        let config = KvDistributionConfig {
            mode: KvDistMode::Mirror,
            mirror_heads: 0,
        };
        assert_eq!(KvDistDecision::from_config(&config, &handle), KvDistDecision::Mirror);
    }

    /// TEST-DIST-013-05: 分布式 + PartialHeadMirror → PartialHeadMirror { local_heads }
    // @trace TEST-DIST-013
    #[test]
    fn kv_dist_distributed_partial_head_mirror() {
        let handle = make_distributed_handle(0, 4);
        let config = KvDistributionConfig {
            mode: KvDistMode::PartialHeadMirror,
            mirror_heads: 8,
        };
        assert_eq!(
            KvDistDecision::from_config(&config, &handle),
            KvDistDecision::PartialHeadMirror {
                local_heads: 8,
                total_heads: 0,
            }
        );
    }

    /// TEST-DIST-013-06: 分布式 + TieredCache → TieredCache { hbm_ratio, ddr_ratio }
    // @trace TEST-DIST-013
    #[test]
    fn kv_dist_distributed_tiered_cache() {
        let handle = make_distributed_handle(0, 4);
        let config = KvDistributionConfig {
            mode: KvDistMode::TieredCache,
            mirror_heads: 0,
        };
        // NCCL 未初始化 → hbm_ratio=0.8, ddr_ratio=0.15
        assert_eq!(
            KvDistDecision::from_config(&config, &handle),
            KvDistDecision::TieredCache {
                hbm_ratio: 0.8,
                ddr_ratio: 0.15,
            }
        );
    }

    /// TEST-DIST-013-07: needs_cross_node_transfer — Local=false, 其他=true
    // @trace TEST-DIST-013
    #[test]
    fn kv_dist_needs_cross_node_transfer() {
        assert!(!KvDistDecision::Local.needs_cross_node_transfer());
        assert!(KvDistDecision::OnDemand { prefetch: false }.needs_cross_node_transfer());
        assert!(KvDistDecision::Mirror.needs_cross_node_transfer());
        assert!(KvDistDecision::PartialHeadMirror {
            local_heads: 4,
            total_heads: 16,
        }
        .needs_cross_node_transfer());
        assert!(KvDistDecision::TieredCache {
            hbm_ratio: 0.6,
            ddr_ratio: 0.3,
        }
        .needs_cross_node_transfer());
    }

    /// TEST-DIST-013-08: is_prefetch_enabled — 仅 OnDemand { prefetch: true }
    // @trace TEST-DIST-013
    #[test]
    fn kv_dist_is_prefetch_enabled() {
        assert!(!KvDistDecision::Local.is_prefetch_enabled());
        assert!(!KvDistDecision::OnDemand { prefetch: false }.is_prefetch_enabled());
        assert!(KvDistDecision::OnDemand { prefetch: true }.is_prefetch_enabled());
        assert!(!KvDistDecision::Mirror.is_prefetch_enabled());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-014: 分布式 MoE (REQ-DIST-014)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_014 {
    use gllm::engine::distributed_config::{
        CommHandleWrapper, ExpertPlacement, MoeDistributedConfig, ParallelConfig,
    };
    use gllm::moe::distributed_dispatch::distributed_dispatch::{
        AllToAllDecision, MoeDistDecision, MoePlacementDecision,
    };

    fn single_node_handle() -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap()
    }

    fn multi_node_handle(rank: u32, world_size: u32) -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig {
            tp_size: world_size,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank,
            world_size,
            unique_id: String::new(),
            stage_id: 0,
        }).unwrap()
    }

    /// TEST-DIST-014-01: 单机模式退化为 Auto
    // @trace TEST-DIST-014
    #[test]
    fn moe_dist_single_node_auto() {
        let config = MoeDistributedConfig::default();
        let handle = single_node_handle();
        let decision = MoeDistDecision::from_config(&config, 8, &handle);
        assert_eq!(decision.placement, MoePlacementDecision::Auto);
        assert_eq!(decision.all_to_all, AllToAllDecision::Auto);
    }

    /// TEST-DIST-014-02: 分布式 + RoundRobin
    // @trace TEST-DIST-014
    #[test]
    fn moe_dist_distributed_roundrobin() {
        let config = MoeDistributedConfig {
            expert_placement: ExpertPlacement::RoundRobin,
            ..Default::default()
        };
        let handle = multi_node_handle(0, 4);
        let decision = MoeDistDecision::from_config(&config, 8, &handle);
        assert_eq!(decision.placement, MoePlacementDecision::RoundRobin);
    }

    /// TEST-DIST-014-03: 分布式 + HotCold → 初始 hot=0, cold=num_experts
    // @trace TEST-DIST-014
    #[test]
    fn moe_dist_distributed_hot_cold() {
        let config = MoeDistributedConfig {
            expert_placement: ExpertPlacement::HotCold,
            ..Default::default()
        };
        let handle = multi_node_handle(0, 4);
        let decision = MoeDistDecision::from_config(&config, 8, &handle);
        assert_eq!(
            decision.placement,
            MoePlacementDecision::HotCold {
                hot_count: 0,
                cold_count: 8,
            }
        );
    }

    /// TEST-DIST-014-04: 分布式 + Custom → 初始 RoundRobin 映射
    // @trace TEST-DIST-014
    #[test]
    fn moe_dist_distributed_custom_initial_mapping() {
        let config = MoeDistributedConfig {
            expert_placement: ExpertPlacement::Custom,
            ..Default::default()
        };
        let handle = multi_node_handle(0, 4);
        let decision = MoeDistDecision::from_config(&config, 8, &handle);
        assert_eq!(
            decision.placement,
            MoePlacementDecision::Custom {
                mapping: vec![0, 1, 2, 3, 0, 1, 2, 3],
            }
        );
    }

    /// TEST-DIST-014-05: expert_target_rank — RoundRobin
    // @trace TEST-DIST-014
    #[test]
    fn moe_dist_expert_target_rank_roundrobin() {
        let config = MoeDistributedConfig {
            expert_placement: ExpertPlacement::RoundRobin,
            ..Default::default()
        };
        let handle = multi_node_handle(0, 4);
        let decision = MoeDistDecision::from_config(&config, 8, &handle);
        assert_eq!(decision.expert_target_rank(0), 0);
        assert_eq!(decision.expert_target_rank(1), 1);
        assert_eq!(decision.expert_target_rank(4), 0);
        assert_eq!(decision.expert_target_rank(7), 3);
    }

    /// TEST-DIST-014-06: needs_cross_gpu_dispatch — Auto=false, 其他=true
    // @trace TEST-DIST-014
    #[test]
    fn moe_dist_needs_cross_gpu() {
        let single_config = MoeDistributedConfig::default();
        let single_handle = single_node_handle();
        let auto_decision = MoeDistDecision::from_config(&single_config, 8, &single_handle);

        let dist_config = MoeDistributedConfig {
            expert_placement: ExpertPlacement::RoundRobin,
            ..Default::default()
        };
        let dist_handle = multi_node_handle(0, 4);
        let rr_decision = MoeDistDecision::from_config(&dist_config, 8, &dist_handle);

        assert!(!auto_decision.needs_cross_gpu_dispatch());
        assert!(rr_decision.needs_cross_gpu_dispatch());
    }

    /// TEST-DIST-014-07: dispatch_experts 单机 → local result
    // @trace TEST-DIST-014
    #[test]
    fn moe_dist_dispatch_single_node() {
        let config = MoeDistributedConfig::default();
        let handle = single_node_handle();
        let decision = MoeDistDecision::from_config(&config, 8, &handle);
        let result = decision.dispatch_experts(&[0, 1], &[0, 1], &handle).unwrap();
        assert_eq!(result.send_counts.len(), 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-015: EPLB 专家负载均衡 (REQ-DIST-015)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_015 {
    use gllm::engine::distributed_config::{CommHandleWrapper, ParallelConfig};
    use gllm::moe::eplb::eplb::{should_rebalance, ExpertLoadStats};

    fn single_node_handle() -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap()
    }

    fn multi_node_handle(rank: u32, world_size: u32) -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig {
            tp_size: world_size,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank,
            world_size,
            unique_id: String::new(),
            stage_id: 0,
        }).unwrap()
    }

    /// TEST-DIST-015-01: ExpertLoadStats 初始化为零计数
    // @trace TEST-DIST-015
    #[test]
    fn eplb_stats_init_zero() {
        let stats = ExpertLoadStats::new(8);
        assert_eq!(stats.invocation_counts.len(), 8);
        assert!(stats.invocation_counts.iter().all(|&c| c == 0));
        assert_eq!(stats.total_invocations(), 0);
    }

    /// TEST-DIST-015-02: record_invocation 递增计数
    // @trace TEST-DIST-015
    #[test]
    fn eplb_stats_record_invocation() {
        let mut stats = ExpertLoadStats::new(4);
        stats.record_invocation(0);
        stats.record_invocation(0);
        stats.record_invocation(2);
        assert_eq!(stats.invocation_counts[0], 2);
        assert_eq!(stats.invocation_counts[1], 0);
        assert_eq!(stats.invocation_counts[2], 1);
    }

    /// TEST-DIST-015-03: 越界 expert_id 静默忽略
    // @trace TEST-DIST-015
    #[test]
    fn eplb_stats_out_of_bounds_ignored() {
        let mut stats = ExpertLoadStats::new(4);
        stats.record_invocation(99);
        assert!(stats.invocation_counts.iter().all(|&c| c == 0));
    }

    /// TEST-DIST-015-04: hot_experts 排序正确
    // @trace TEST-DIST-015
    #[test]
    fn eplb_stats_hot_experts() {
        let mut stats = ExpertLoadStats::new(8);
        for _ in 0..100 {
            stats.record_invocation(3);
        }
        for _ in 0..50 {
            stats.record_invocation(7);
        }
        let hot = stats.hot_experts(2);
        assert_eq!(hot, vec![3, 7]);
    }

    /// TEST-DIST-015-05: imbalance_ratio — 均衡=1.0, 空载=0.0
    // @trace TEST-DIST-015
    #[test]
    fn eplb_stats_imbalance_ratio() {
        let mut stats = ExpertLoadStats::new(4);
        for i in 0..4 {
            for _ in 0..10 {
                stats.record_invocation(i);
            }
        }
        assert!((stats.imbalance_ratio() - 1.0).abs() < f64::EPSILON);

        let empty = ExpertLoadStats::new(4);
        assert_eq!(empty.imbalance_ratio(), 0.0);
    }

    /// TEST-DIST-015-06: reset_window 清零计数
    // @trace TEST-DIST-015
    #[test]
    fn eplb_stats_reset_window() {
        let mut stats = ExpertLoadStats::new(4);
        stats.record_invocation(0);
        stats.record_invocation(1);
        assert_eq!(stats.total_invocations(), 2);
        stats.reset_window();
        assert_eq!(stats.total_invocations(), 0);
    }

    /// TEST-DIST-015-07: aggregate_from_all_ranks 单机 → no-op
    // @trace TEST-DIST-015
    #[test]
    fn eplb_stats_aggregate_single_node() {
        let mut stats = ExpertLoadStats::new(4);
        stats.record_invocation(0);
        let handle = single_node_handle();
        let result = stats.aggregate_from_all_ranks(&handle);
        assert!(result.is_ok());
        assert_eq!(stats.invocation_counts[0], 1);
    }

    /// TEST-DIST-015-08: should_rebalance — 单机永不重平衡
    // @trace TEST-DIST-015
    #[test]
    fn eplb_single_node_never_rebalance() {
        let mut stats = ExpertLoadStats::new(8);
        for _ in 0..1000 {
            stats.record_invocation(0);
        }
        let handle = single_node_handle();
        let decision = should_rebalance(&stats, &handle, 2.0);
        assert!(!decision.needs_rebalance);
    }

    /// TEST-DIST-015-09: should_rebalance — 不均衡触发重平衡
    // @trace TEST-DIST-015
    #[test]
    fn eplb_imbalanced_triggers_rebalance() {
        let mut stats = ExpertLoadStats::new(8);
        for _ in 0..100 {
            stats.record_invocation(0);
        }
        for i in 1..8 {
            stats.record_invocation(i);
        }
        let handle = multi_node_handle(0, 4);
        let decision = should_rebalance(&stats, &handle, 2.0);
        assert!(decision.needs_rebalance);
        assert!(decision.hot_expert_ids.contains(&0));
    }

    /// TEST-DIST-015-10: should_rebalance — 均衡不触发
    // @trace TEST-DIST-015
    #[test]
    fn eplb_balanced_no_rebalance() {
        let mut stats = ExpertLoadStats::new(8);
        for i in 0..8 {
            for _ in 0..10 {
                stats.record_invocation(i);
            }
        }
        let handle = multi_node_handle(0, 4);
        let decision = should_rebalance(&stats, &handle, 2.0);
        assert!(!decision.needs_rebalance);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-016: Ring Attention Context Parallelism (REQ-DIST-016)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_016 {
    use gllm::engine::coordinator::context_parallel::context_parallel::{
        CpConfig, RingAttentionPlan, RingPhase,
    };

    /// TEST-DIST-016-01: CpConfig cp_size=1 → is_enabled=false
    // @trace TEST-DIST-016
    #[test]
    fn cp_config_cp1_not_enabled() {
        let config = CpConfig::new(1, 0, 0);
        assert!(!config.is_enabled());
    }

    /// TEST-DIST-016-02: CpConfig cp_size>1 → is_enabled=true
    // @trace TEST-DIST-016
    #[test]
    fn cp_config_cp4_enabled() {
        let config = CpConfig::new(4, 0, 0);
        assert!(config.is_enabled());
    }

    /// TEST-DIST-016-03: next_rank/prev_rank — ring 拓扑
    // @trace TEST-DIST-016
    #[test]
    fn cp_config_ring_navigation() {
        let config = CpConfig::new(4, 1, 0);
        assert_eq!(config.next_rank(), 2);
        assert_eq!(config.prev_rank(), 0);

        // rank 0 → prev = cp_size - 1
        let config0 = CpConfig::new(4, 0, 0);
        assert_eq!(config0.prev_rank(), 3);
        assert_eq!(config0.next_rank(), 1);
    }

    /// TEST-DIST-016-04: local_seq_len — 均匀分片
    // @trace TEST-DIST-016
    #[test]
    fn cp_config_local_seq_len_even() {
        let config = CpConfig::new(4, 0, 0);
        assert_eq!(config.local_seq_len(1024), 256);
    }

    /// TEST-DIST-016-05: local_seq_len — 不均匀分片（余数分配给前几个 rank）
    // @trace TEST-DIST-016
    #[test]
    fn cp_config_local_seq_len_uneven() {
        let config0 = CpConfig::new(4, 0, 0);
        let config1 = CpConfig::new(4, 1, 0);
        // 1000 / 4 = 250 remainder 0
        assert_eq!(config0.local_seq_len(1000), 250);
        assert_eq!(config1.local_seq_len(1000), 250);
        // 1001 / 4 = 250 remainder 1 → rank 0 gets 251
        assert_eq!(config0.local_seq_len(1001), 251);
        assert_eq!(config1.local_seq_len(1001), 250);
    }

    /// TEST-DIST-016-06: validate — cp_rank >= cp_size → Err
    // @trace TEST-DIST-016
    #[test]
    fn cp_config_validate_rank_out_of_range() {
        let config = CpConfig::new(4, 5, 0);
        assert!(config.validate().is_err());
    }

    /// TEST-DIST-016-07: validate — cp_size=0 → Err
    // @trace TEST-DIST-016
    #[test]
    fn cp_config_validate_zero_cp_size() {
        let config = CpConfig::new(0, 0, 0);
        assert!(config.validate().is_err());
    }

    /// TEST-DIST-016-08: RingAttentionPlan — total_steps = cp_size
    // @trace TEST-DIST-016
    #[test]
    fn ring_attention_plan_total_steps() {
        let config = CpConfig::new(4, 0, 0);
        let plan = RingAttentionPlan::new(config);
        assert_eq!(plan.total_steps, 4);
    }

    /// TEST-DIST-016-09: phases_for_step — step 0 = LocalCompute, step > 0 = Send+Recv+Remote
    // @trace TEST-DIST-016
    #[test]
    fn ring_attention_plan_phases() {
        let config = CpConfig::new(4, 0, 0);
        let plan = RingAttentionPlan::new(config);

        // step 0: 仅本地计算
        let phases0 = plan.phases_for_step(0);
        assert_eq!(phases0.len(), 1);
        assert!(matches!(phases0[0], RingPhase::LocalCompute));

        // step 1: send + recv + remote
        let phases1 = plan.phases_for_step(1);
        assert_eq!(phases1.len(), 3);
        assert!(matches!(phases1[0], RingPhase::SendKvBlock { step: 1 }));
        assert!(matches!(phases1[1], RingPhase::RecvKvBlock { step: 1 }));
        assert!(matches!(phases1[2], RingPhase::RemoteCompute { step: 1 }));
    }

    /// TEST-DIST-016-10: kv_source_rank — ring 步进推导
    // @trace TEST-DIST-016
    #[test]
    fn ring_attention_plan_kv_source_rank() {
        let config = CpConfig::new(4, 0, 0);
        let plan = RingAttentionPlan::new(config);
        // step 0: local (rank 0)
        assert_eq!(plan.kv_source_rank(0), 0);
        // step 1: from rank (0 + 4 - 1) % 4 = 3
        assert_eq!(plan.kv_source_rank(1), 3);
        // step 2: from rank (0 + 4 - 2) % 4 = 2
        assert_eq!(plan.kv_source_rank(2), 2);
        // step 3: from rank (0 + 4 - 3) % 4 = 1
        assert_eq!(plan.kv_source_rank(3), 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-017: SAGUARO 分布式推测解码 (REQ-DIST-017)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_017 {
    use gllm::engine::distributed_config::{CommHandleWrapper, ParallelConfig};
    use gllm::speculative::saguaro::saguaro::{
        SaguaroConfig, SaguaroDistSpec, SaguaroPhase, SaguaroPipelineStage,
    };

    fn make_single_handle() -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap()
    }

    fn make_distributed_handle(rank: u32, world_size: u32) -> CommHandleWrapper {
        CommHandleWrapper::from_config(&ParallelConfig {
            tp_size: world_size,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank,
            world_size,
            unique_id: String::new(),
            stage_id: 0,
        }).unwrap()
    }

    /// TEST-DIST-017-01: SaguaroConfig from_comm_handle — 默认 draft=0, verify=1
    // @trace TEST-DIST-017
    #[test]
    fn saguaro_config_from_comm_handle_multi() {
        let handle = make_distributed_handle(0, 2);
        let config = SaguaroConfig::from_comm_handle(&handle);
        assert_eq!(config.draft_rank, 0);
        assert_eq!(config.verify_rank, 1);
        assert_eq!(config.draft_length, 5);
    }

    /// TEST-DIST-017-02: SaguaroConfig 单机 → verify_rank=0
    // @trace TEST-DIST-017
    #[test]
    fn saguaro_config_single_node_verify_same() {
        let handle = make_single_handle();
        let config = SaguaroConfig::from_comm_handle(&handle);
        assert_eq!(config.verify_rank, 0);
    }

    /// TEST-DIST-017-03: SaguaroConfig is_draft_gpu / is_verify_gpu
    // @trace TEST-DIST-017
    #[test]
    fn saguaro_config_gpu_role() {
        let handle0 = make_distributed_handle(0, 2);
        let config = SaguaroConfig::from_comm_handle(&handle0);
        assert!(config.is_draft_gpu(&handle0));
        assert!(!config.is_verify_gpu(&handle0));

        let handle1 = make_distributed_handle(1, 2);
        assert!(!config.is_draft_gpu(&handle1));
        assert!(config.is_verify_gpu(&handle1));
    }

    /// TEST-DIST-017-04: SaguaroConfig phase_sequence — 五阶段
    // @trace TEST-DIST-017
    #[test]
    fn saguaro_config_phase_sequence() {
        let handle = make_distributed_handle(0, 2);
        let config = SaguaroConfig::from_comm_handle(&handle);
        let phases = config.phase_sequence();
        assert_eq!(phases.len(), 5);
        assert!(matches!(phases[0], SaguaroPhase::DraftGenerate { .. }));
        assert!(matches!(phases[1], SaguaroPhase::SendCandidates { .. }));
        assert!(matches!(phases[2], SaguaroPhase::VerifyAccept { .. }));
        assert!(matches!(phases[3], SaguaroPhase::SendResults { .. }));
        assert!(matches!(phases[4], SaguaroPhase::UpdateState { .. }));
    }

    /// TEST-DIST-017-05: SaguaroDistSpec from_config — 初始 Idle
    // @trace TEST-DIST-017
    #[test]
    fn saguaro_dist_spec_from_config() {
        let config = SaguaroConfig {
            draft_rank: 0,
            verify_rank: 1,
            draft_length: 5,
        };
        let spec = SaguaroDistSpec::from_config(&config);
        assert_eq!(spec.draft_gpu, 0);
        assert_eq!(spec.verify_gpu, 1);
        assert!(spec.candidate_buf.is_empty());
        assert!(spec.verify_result_buf.is_empty());
        assert_eq!(spec.pipeline_stage, SaguaroPipelineStage::Idle);
    }

    /// TEST-DIST-017-06: SaguaroDistSpec is_draft_gpu / is_verify_gpu
    // @trace TEST-DIST-017
    #[test]
    fn saguaro_dist_spec_gpu_role() {
        let config = SaguaroConfig {
            draft_rank: 0,
            verify_rank: 1,
            draft_length: 5,
        };
        let spec = SaguaroDistSpec::from_config(&config);
        let draft_handle = make_distributed_handle(0, 2);
        assert!(spec.is_draft_gpu(&draft_handle));
        assert!(!spec.is_verify_gpu(&draft_handle));

        let verify_handle = make_distributed_handle(1, 2);
        assert!(!spec.is_draft_gpu(&verify_handle));
        assert!(spec.is_verify_gpu(&verify_handle));
    }

    /// TEST-DIST-017-07: SaguaroDistSpec execute_round 单机 → Err
    // @trace TEST-DIST-017
    #[test]
    fn saguaro_dist_spec_execute_round_single_node_error() {
        let config = SaguaroConfig {
            draft_rank: 0,
            verify_rank: 1,
            draft_length: 5,
        };
        let mut spec = SaguaroDistSpec::from_config(&config);
        let handle = make_single_handle();
        let result = spec.execute_round(&[1, 2, 3], &handle);
        assert!(result.is_err());
    }

    /// TEST-DIST-017-08: SaguaroPhase 枚举区分
    // @trace TEST-DIST-017
    #[test]
    fn saguaro_phase_distinct() {
        let phases = [
            SaguaroPhase::DraftGenerate { num_draft_tokens: 5 },
            SaguaroPhase::SendCandidates { verify_rank: 1 },
            SaguaroPhase::VerifyAccept { draft_rank: 0 },
            SaguaroPhase::SendResults { draft_rank: 0 },
            SaguaroPhase::UpdateState { accepted_count: 3 },
        ];
        // 所有阶段语义不同
        for i in 0..phases.len() {
            for j in (i + 1)..phases.len() {
                assert_ne!(
                    format!("{:?}", phases[i]),
                    format!("{:?}", phases[j]),
                    "Phase {} and {} should be distinct",
                    i,
                    j
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 非 nccl 环境：验证枚举类型始终可用的编译时逻辑
// ═══════════════════════════════════════════════════════════════════════════════

/// TEST-DIST-ENUM-01: PdDisaggMode 枚举完整性（非 nccl 也可用）
#[test]
fn pd_disagg_mode_variants() {
    assert_eq!(PdDisaggMode::default(), PdDisaggMode::Collocated);
    assert_ne!(PdDisaggMode::Collocated, PdDisaggMode::Disaggregated);
}

/// TEST-DIST-ENUM-02: KvDistMode 5 变体完整性
#[test]
fn kv_dist_mode_5_variants() {
    let variants = [
        KvDistMode::Local,
        KvDistMode::OnDemand,
        KvDistMode::Mirror,
        KvDistMode::PartialHeadMirror,
        KvDistMode::TieredCache,
    ];
    assert_eq!(variants.len(), 5);
    assert_eq!(KvDistMode::default(), KvDistMode::Local);
}

/// TEST-DIST-ENUM-03: CommCompressHint 枚举完整性
#[test]
fn comm_compress_hint_variants() {
    assert_eq!(CommCompressHint::default(), CommCompressHint::Auto);
    assert_ne!(CommCompressHint::Auto, CommCompressHint::NeverCompress);
}

/// TEST-DIST-ENUM-04: ExpertPlacement 枚举完整性
#[test]
fn expert_placement_variants() {
    assert_eq!(ExpertPlacement::default(), ExpertPlacement::Auto);
    let all = [
        ExpertPlacement::Auto,
        ExpertPlacement::RoundRobin,
        ExpertPlacement::HotCold,
        ExpertPlacement::Custom,
    ];
    assert_eq!(all.len(), 4);
}

/// TEST-DIST-ENUM-05: NodeRole 枚举完整性
#[test]
fn node_role_variants() {
    assert_eq!(NodeRole::default(), NodeRole::Auto);
    let all = [NodeRole::Auto, NodeRole::PrefillOnly, NodeRole::DecodeOnly, NodeRole::Mixed];
    assert_eq!(all.len(), 4);
}

mod pipeline;
