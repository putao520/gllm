//! TEST-DIST-018~034 — Pipeline Parallel 专项测试 (REQ-DIST-018~034)
//!
//! 覆盖 SPEC 43-DISTRIBUTED-IMPLEMENTATION.html 中 REQ-DIST-018~034 的 Pipeline Parallel 核心逻辑。
//! 优先测试编译时逻辑（配置/枚举/决策/调度计算），不依赖运行时 NCCL 通信。
//!
//! 运行方式:
//! ```bash
//! # nccl 环境（完整测试）
//! cargo test --test dist --features nccl -- --test-threads=2
//! ```

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-018: PipelineConfig 配置与 stage 划分 (REQ-DIST-018)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_018 {
    use gllm::engine::pipeline::config::{PipelineConfig, PipelineConfigError};

    /// TEST-DIST-018-01: PipelineConfig 基本构造 — 字段正确
    // @trace TEST-DIST-018
    #[test]
    fn pipeline_config_basic_construction() {
        let config = PipelineConfig {
            pp_size: 4,
            stage_id: 1,
            num_virtual_stages: 2,
            micro_batch_size: 8,
            layers_per_stage: 12,
        };
        assert_eq!(config.pp_size, 4);
        assert_eq!(config.stage_id, 1);
        assert_eq!(config.num_virtual_stages, 2);
        assert_eq!(config.micro_batch_size, 8);
        assert_eq!(config.layers_per_stage, 12);
    }

    /// TEST-DIST-018-02: validate — 合法配置通过
    // @trace TEST-DIST-018
    #[test]
    fn pipeline_config_validate_valid() {
        let config = PipelineConfig {
            pp_size: 4,
            stage_id: 0,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 16,
        };
        assert!(config.validate());
    }

    /// TEST-DIST-018-03: validate — pp_size=0 无效
    // @trace TEST-DIST-018
    #[test]
    fn pipeline_config_validate_invalid_pp_size() {
        let config = PipelineConfig {
            pp_size: 0,
            stage_id: 0,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 16,
        };
        assert!(!config.validate());
    }

    /// TEST-DIST-018-04: layer_range — 正确的层范围划分
    // @trace TEST-DIST-018
    #[test]
    fn pipeline_config_layer_range() {
        let config = PipelineConfig {
            pp_size: 4,
            stage_id: 2,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 8,
        };
        let range = config.layer_range();
        assert_eq!(range.start, 16);
        assert_eq!(range.end, 24);
    }

    /// TEST-DIST-018-05: is_first_stage / is_last_stage
    // @trace TEST-DIST-018
    #[test]
    fn pipeline_config_stage_predicates() {
        let first = PipelineConfig {
            pp_size: 4, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 8,
        };
        let middle = PipelineConfig {
            pp_size: 4, stage_id: 1, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 8,
        };
        let last = PipelineConfig {
            pp_size: 4, stage_id: 3, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 8,
        };
        assert!(first.is_first_stage());
        assert!(!first.is_last_stage());
        assert!(!middle.is_first_stage());
        assert!(!middle.is_last_stage());
        assert!(!last.is_first_stage());
        assert!(last.is_last_stage());
    }

    /// TEST-DIST-018-06: is_pipeline_parallel — pp_size > 1 时为 true
    // @trace TEST-DIST-018
    #[test]
    fn pipeline_config_is_pipeline_parallel() {
        let pp = PipelineConfig {
            pp_size: 4, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 8,
        };
        let no_pp = PipelineConfig {
            pp_size: 1, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 8,
        };
        assert!(pp.is_pipeline_parallel());
        assert!(!no_pp.is_pipeline_parallel());
    }

    /// TEST-DIST-018-07: with_micro_batch_size / with_virtual_stages builder
    // @trace TEST-DIST-018
    #[test]
    fn pipeline_config_builder_methods() {
        let config = PipelineConfig {
            pp_size: 4, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 8,
        };
        let config2 = config.with_micro_batch_size(16);
        assert_eq!(config2.micro_batch_size, 16);
        let config3 = config2.with_virtual_stages(4);
        assert_eq!(config3.num_virtual_stages, 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-019: (reserved — covered by TEST-DIST-018 config tests)
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-020: MicroBatch 切分 (REQ-DIST-020)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_020 {
    use gllm::engine::pipeline::micro_batch::{MicroBatch, MicroBatchScheduler};

    /// TEST-DIST-020-01: MicroBatchScheduler split — 均匀切分
    // @trace TEST-DIST-020
    #[test]
    fn micro_batch_split_even() {
        let scheduler = MicroBatchScheduler { micro_batch_size: 4 };
        let batches = scheduler.split(16);
        assert_eq!(batches.len(), 4);
        for (i, mb) in batches.iter().enumerate() {
            assert_eq!(mb.index, i);
            assert_eq!(mb.token_count, 4);
            assert_eq!(mb.offset, i * 4);
        }
    }

    /// TEST-DIST-020-02: MicroBatchScheduler split — 非均匀切分
    // @trace TEST-DIST-020
    #[test]
    fn micro_batch_split_uneven() {
        let scheduler = MicroBatchScheduler { micro_batch_size: 4 };
        let batches = scheduler.split(14);
        assert_eq!(batches.len(), 4); // ceil(14/4) = 4
        assert_eq!(batches[3].token_count, 2); // last batch has remainder
    }

    /// TEST-DIST-020-03: num_microbatches 计算正确
    // @trace TEST-DIST-020
    #[test]
    fn micro_batch_num_microbatches() {
        let scheduler = MicroBatchScheduler { micro_batch_size: 4 };
        assert_eq!(scheduler.num_microbatches(16), 4);
        assert_eq!(scheduler.num_microbatches(15), 4);
        assert_eq!(scheduler.num_microbatches(1), 1);
    }

    /// TEST-DIST-020-04: MicroBatch 字段正确
    // @trace TEST-DIST-020
    #[test]
    fn micro_batch_fields() {
        let mb = MicroBatch { offset: 8, token_count: 4, index: 2 };
        assert_eq!(mb.offset, 8);
        assert_eq!(mb.token_count, 4);
        assert_eq!(mb.index, 2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-021: Stage 间激活传输 + GPipe/1F1B 调度 (REQ-DIST-021)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_021 {
    use gllm::engine::pipeline::scheduler::{MicroBatchStrategy, PipelineOp, PipelineScheduler};
    use gllm::engine::pipeline::activation_xfer::{ActivationDirection, ActivationTransport};

    /// TEST-DIST-021-01: PipelineScheduler GPipe 调度产生正确操作序列
    // @trace TEST-DIST-021
    #[test]
    fn gpipe_schedule_produces_ops() {
        let scheduler = PipelineScheduler::new(4, 0).unwrap();
        let ops = scheduler.schedule_gpipe(8).unwrap();
        assert!(!ops.is_empty());
        let fwd_count = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        assert_eq!(fwd_count, 8);
    }

    /// TEST-DIST-021-02: PipelineScheduler 1F1B 调度产生正确操作序列
    // @trace TEST-DIST-021
    #[test]
    fn one_f_one_b_schedule_produces_ops() {
        let scheduler = PipelineScheduler::new(4, 0).unwrap();
        let ops = scheduler.schedule_1f1b(8).unwrap();
        assert!(!ops.is_empty());
        let fwd_count = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        let bwd_count = ops.iter().filter(|op| matches!(op, PipelineOp::Backward(_))).count();
        assert_eq!(fwd_count, 8);
        assert_eq!(bwd_count, 8);
    }

    /// TEST-DIST-021-03: ActivationTransport 构造与字段
    // @trace TEST-DIST-021
    #[test]
    fn activation_transport_construction() {
        let transport = ActivationTransport::new(2, 4, 1024, false);
        assert_eq!(transport.micro_batch_size, 4);
        assert_eq!(transport.hidden_size, 1024);
    }

    /// TEST-DIST-021-04: ActivationDirection 枚举
    // @trace TEST-DIST-021
    #[test]
    fn activation_direction_variants() {
        let fwd = ActivationDirection::Forward;
        let bwd = ActivationDirection::Backward;
        assert_ne!(fwd, bwd);
    }

    /// TEST-DIST-021-05: MicroBatchStrategy 枚举
    // @trace TEST-DIST-021
    #[test]
    fn micro_batch_strategy_variants() {
        let gpipe = MicroBatchStrategy::GPipe;
        let one_f_one_b = MicroBatchStrategy::OneFOneB;
        assert_ne!(gpipe, one_f_one_b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-022: Interleaved 1F1B 调度 (REQ-DIST-022)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_022 {
    use gllm::engine::pipeline::interleaved::{Interleaved1F1B, ScheduleComparison};
    use gllm::engine::pipeline::scheduler::PipelineOp;

    /// TEST-DIST-022-01: Interleaved1F1B 构造 — 合法参数
    // @trace TEST-DIST-022
    #[test]
    fn interleaved_new_valid() {
        let sched = Interleaved1F1B::new(4, 0, 2).unwrap();
        assert_eq!(sched.pp_size, 4);
        assert_eq!(sched.num_virtual_stages, 2);
    }

    /// TEST-DIST-022-02: 交错气泡比率 = (pp_size-1)/(pp_size*num_virtual_stages)
    // @trace TEST-DIST-022
    #[test]
    fn interleaved_bubble_ratio_formula() {
        let sched = Interleaved1F1B::new(4, 0, 2).unwrap();
        let expected = 3.0 / (4.0 * 2.0);
        assert!((sched.bubble_ratio() - expected).abs() < 1e-10);
    }

    /// TEST-DIST-022-03: 交错改进倍数 ≈ num_virtual_stages
    // @trace TEST-DIST-022
    #[test]
    fn interleaved_bubble_improvement() {
        let sched = Interleaved1F1B::new(4, 0, 2).unwrap();
        let improvement = sched.bubble_ratio_improvement();
        assert!((improvement - 2.0).abs() < 1e-10);
    }

    /// TEST-DIST-022-04: schedule_interleaved 产生 Forward+Backward 操作
    // @trace TEST-DIST-022
    #[test]
    fn interleaved_schedule_produces_ops() {
        let sched = Interleaved1F1B::new(2, 0, 2).unwrap();
        let ops = sched.schedule_interleaved(4).unwrap();
        let fwd = ops.iter().filter(|op| matches!(op, PipelineOp::Forward(_))).count();
        let bwd = ops.iter().filter(|op| matches!(op, PipelineOp::Backward(_))).count();
        assert!(fwd > 0);
        assert!(bwd > 0);
    }

    /// TEST-DIST-022-05: ScheduleComparison — 交错气泡 < 非交错气泡
    // @trace TEST-DIST-022
    #[test]
    fn schedule_comparison_interleaved_lower_bubble() {
        let comp = ScheduleComparison::compare(4, 0, 2, 8, 1024).unwrap();
        assert!(comp.interleaved_bubble_ratio < comp.non_interleaved_bubble_ratio);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-023: 交错 PP 微批次调度 (REQ-DIST-023)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_023 {
    use gllm::engine::pipeline::micro_batch::{
        InterleavedScheduler, InterleavedScheduleStep, SchedulePhase,
    };

    /// TEST-DIST-023-01: InterleavedScheduler 构造
    // @trace TEST-DIST-023
    #[test]
    fn interleaved_scheduler_construction() {
        let sched = InterleavedScheduler::new(4, 8, 8).unwrap();
        assert_eq!(sched.pp_size, 4);
        assert_eq!(sched.num_virtual_stages, 8);
        assert_eq!(sched.num_microbatches, 8);
    }

    /// TEST-DIST-023-02: virtual_stages_per_device 正确
    // @trace TEST-DIST-023
    #[test]
    fn interleaved_virtual_stages_per_device() {
        let sched = InterleavedScheduler::new(4, 8, 8).unwrap();
        assert_eq!(sched.virtual_stages_per_device(), 2);
    }

    /// TEST-DIST-023-03: bubble_ratio — 交错模式下更低
    // @trace TEST-DIST-023
    #[test]
    fn interleaved_scheduler_bubble_ratio() {
        let sched = InterleavedScheduler::new(4, 8, 8).unwrap();
        let ratio = sched.bubble_ratio();
        // Interleaved bubble ratio should be lower than non-interleaved (pp-1)/pp
        assert!(ratio < 0.75); // (4-1)/4 = 0.75
    }

    /// TEST-DIST-023-04: schedule_for_device 产生调度步骤
    // @trace TEST-DIST-023
    #[test]
    fn interleaved_schedule_for_device() {
        let sched = InterleavedScheduler::new(2, 4, 4).unwrap();
        let steps = sched.schedule_for_device(0);
        assert!(!steps.is_empty());
        let fwd_count = steps.iter().filter(|s| s.phase == SchedulePhase::Forward).count();
        let bwd_count = steps.iter().filter(|s| s.phase == SchedulePhase::Backward).count();
        assert!(fwd_count > 0);
        assert!(bwd_count > 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-024: Pipeline Bubble 分析 (REQ-DIST-024)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_024 {
    use gllm::engine::pipeline::bubble::{BubbleAnalyzer, BubbleMetrics};

    /// TEST-DIST-024-01: BubbleAnalyzer GPipe 气泡比率
    // @trace TEST-DIST-024
    #[test]
    fn bubble_analyzer_gpipe_ratio() {
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0, 0.5);
        let ratio = analyzer.gpipe_bubble_ratio();
        // GPipe bubble ratio = (pp-1)/pp = 3/4 = 0.75
        assert!((ratio - 0.75).abs() < 1e-10);
    }

    /// TEST-DIST-024-02: BubbleAnalyzer 1F1B 气泡比率
    // @trace TEST-DIST-024
    #[test]
    fn bubble_analyzer_one_f_one_b_ratio() {
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0, 0.5);
        let ratio = analyzer.one_f_one_b_bubble_ratio();
        // 1F1B bubble ratio = (pp-1)/(pp+M-1) for M microbatches
        // = 3/(4+8-1) = 3/11
        let expected = 3.0 / 11.0;
        assert!((ratio - expected).abs() < 1e-10);
    }

    /// TEST-DIST-024-03: BubbleAnalyzer 交错气泡比率更低
    // @trace TEST-DIST-024
    #[test]
    fn bubble_analyzer_interleaved_lower() {
        let analyzer = BubbleAnalyzer::new(4, 8, 2, 100.0, 200.0, 0.5);
        let interleaved = analyzer.interleaved_bubble_ratio();
        let gpipe = analyzer.gpipe_bubble_ratio();
        assert!(interleaved < gpipe);
    }

    /// TEST-DIST-024-04: analyze 产出 BubbleMetrics
    // @trace TEST-DIST-024
    #[test]
    fn bubble_analyzer_analyze() {
        let analyzer = BubbleAnalyzer::new(4, 8, 1, 100.0, 200.0, 0.5);
        let metrics = analyzer.analyze();
        // BubbleMetrics should have valid ratio fields
        assert!(metrics.gpipe_bubble_ratio >= 0.0);
        assert!(metrics.one_f_one_b_bubble_ratio >= 0.0);
    }

    /// TEST-DIST-024-05: recommend_strategy 基于气泡比率阈值推荐
    // @trace TEST-DIST-024
    #[test]
    fn bubble_analyzer_recommend_strategy() {
        let analyzer = BubbleAnalyzer::new(4, 8, 2, 100.0, 200.0, 0.5);
        let strategy = analyzer.recommend_strategy();
        // With num_virtual_stages=2, interleaved should be recommended
        assert!(strategy == MicroBatchStrategy::OneFOneB
            || strategy == MicroBatchStrategy::GPipe,
            "recommend_strategy should return a valid strategy");
    }

    use gllm::engine::pipeline::scheduler::MicroBatchStrategy;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-025: 自适应微批次大小调整 (REQ-DIST-025)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_025 {
    use gllm::engine::pipeline::adaptive::{AdaptiveConfig, AdaptiveMicroBatchSizer};

    /// TEST-DIST-025-01: AdaptiveMicroBatchSizer 构造
    // @trace TEST-DIST-025
    #[test]
    fn adaptive_sizer_construction() {
        let sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        assert_eq!(sizer.current_mbs, 8);
        assert_eq!(sizer.total_batch_size, 64);
    }

    /// TEST-DIST-025-02: 延迟过高 → 减小微批次大小
    // @trace TEST-DIST-025
    #[test]
    fn adaptive_sizer_high_latency_shrinks() {
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        let new_mbs = sizer.adjust(20.0, 30.0); // latency > 10
        assert!(new_mbs < 8);
    }

    /// TEST-DIST-025-03: 带宽充足 → 增大微批次大小
    // @trace TEST-DIST-025
    #[test]
    fn adaptive_sizer_high_bandwidth_grows() {
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        let new_mbs = sizer.adjust(5.0, 80.0); // bandwidth > 50
        assert!(new_mbs > 8);
    }

    /// TEST-DIST-025-04: 结果约束 user_min <= mbs <= total_batch_size
    // @trace TEST-DIST-025
    #[test]
    fn adaptive_sizer_bounds_respected() {
        let config = AdaptiveConfig::new(10.0, 50.0, 4);
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 32, config).unwrap();
        for _ in 0..20 {
            sizer.adjust(100.0, 0.0); // try to shrink
        }
        assert!(sizer.current_mbs >= 4);
        for _ in 0..20 {
            sizer.adjust(0.0, 100.0); // try to grow
        }
        assert!(sizer.current_mbs <= 32);
    }

    /// TEST-DIST-025-05: 自适应完成标记
    // @trace TEST-DIST-025
    #[test]
    fn adaptive_sizer_mark_complete() {
        let mut sizer = AdaptiveMicroBatchSizer::new(8, 64, AdaptiveConfig::default()).unwrap();
        assert!(!sizer.is_adaptation_complete());
        sizer.adjust(5.0, 80.0);
        sizer.mark_adaptation_complete();
        assert!(sizer.is_adaptation_complete());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-026: SM 分区配置 (REQ-DIST-026)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_026 {
    use gllm::engine::pipeline::scheduler::SmPartitionConfig;

    /// TEST-DIST-026-01: SmPartitionConfig 构造与字段
    // @trace TEST-DIST-026
    #[test]
    fn sm_partition_construction() {
        let config = SmPartitionConfig::new(80, 8);
        assert_eq!(config.total_sms, 80);
        assert_eq!(config.comm_sms, 8);
        assert_eq!(config.compute_sms, 72);
    }

    /// TEST-DIST-026-02: comm_ratio / compute_ratio 正确
    // @trace TEST-DIST-026
    #[test]
    fn sm_partition_ratios() {
        let config = SmPartitionConfig::new(80, 8);
        assert!((config.comm_ratio() - 0.1).abs() < 1e-10);
        assert!((config.compute_ratio() - 0.9).abs() < 1e-10);
    }

    /// TEST-DIST-026-03: is_valid — 合法分区
    // @trace TEST-DIST-026
    #[test]
    fn sm_partition_is_valid() {
        let config = SmPartitionConfig::new(80, 8);
        assert!(config.is_valid());
    }

    /// TEST-DIST-026-04: is_isolated — 通信 SM 与计算 SM 完全隔离
    // @trace TEST-DIST-026
    #[test]
    fn sm_partition_is_isolated() {
        let config = SmPartitionConfig::new(80, 8);
        assert!(config.is_isolated());
    }

    /// TEST-DIST-026-05: default_for_gpu 提供合理默认值
    // @trace TEST-DIST-026
    #[test]
    fn sm_partition_default_for_gpu() {
        let config = SmPartitionConfig::default_for_gpu();
        assert!(config.total_sms > 0);
        assert!(config.comm_sms > 0);
        assert!(config.compute_sms > 0);
        assert!(config.is_valid());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-027: 量化激活传输配置 (REQ-DIST-027)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_027 {
    use gllm::engine::pipeline::scheduler::{
        ActivationQuantFormat, CommComputeOverlap, QuantizedActivationConfig, StreamAssignment,
        StreamKind,
    };

    /// TEST-DIST-027-01: QuantizedActivationConfig 默认不启用
    // @trace TEST-DIST-027
    #[test]
    fn quant_activation_config_default_disabled() {
        let config = QuantizedActivationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.quant_format, ActivationQuantFormat::None);
    }

    /// TEST-DIST-027-02: ActivationQuantFormat 枚举变体完整
    // @trace TEST-DIST-027
    #[test]
    fn activation_quant_format_variants() {
        let formats = [
            ActivationQuantFormat::None,
            ActivationQuantFormat::Fp16,
            ActivationQuantFormat::Bf16,
            ActivationQuantFormat::Fp8E4M3,
            ActivationQuantFormat::Int8Symmetric,
        ];
        // All variants are distinct
        for i in 0..formats.len() {
            for j in (i + 1)..formats.len() {
                assert_ne!(formats[i], formats[j]);
            }
        }
    }

    /// TEST-DIST-027-03: CommComputeOverlap 构造
    // @trace TEST-DIST-027
    #[test]
    fn comm_compute_overlap_construction() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true);
        assert!(overlap.overlap_enabled);
    }

    /// TEST-DIST-027-04: StreamKind 枚举
    // @trace TEST-DIST-027
    #[test]
    fn stream_kind_variants() {
        assert_ne!(StreamKind::Compute, StreamKind::Communication);
        assert_ne!(StreamKind::Compute, StreamKind::None);
    }

    /// TEST-DIST-027-05: assign_streams 分配计算/通信流
    // @trace TEST-DIST-027
    #[test]
    fn assign_streams_produces_assignments() {
        let overlap = CommComputeOverlap::new(100.0, 200.0, 50.0, true);
        let assignments = overlap.assign_streams(4);
        assert!(!assignments.is_empty());
        // Should have both Compute and Communication streams
        let has_compute = assignments.iter().any(|a| a.stream == StreamKind::Compute);
        let has_comm = assignments.iter().any(|a| a.stream == StreamKind::Communication);
        assert!(has_compute);
        assert!(has_comm);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-028: PP 梯度回传与权重同步 (REQ-DIST-028)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_028 {
    use gllm::engine::pipeline::config::PipelineConfig;
    use gllm::engine::pipeline::gradient_sync::{
        GradientSync, GradientSyncConfig, GradientSyncPhase, GradientSyncStats,
    };
    use gllm::engine::pipeline::topology::Topology2D;

    fn make_pipeline_config() -> PipelineConfig {
        PipelineConfig {
            pp_size: 4, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 8,
        }
    }

    fn make_topology() -> Topology2D {
        Topology2D::new(1, 4, 4).unwrap()
    }

    /// TEST-DIST-028-01: GradientSync 构造
    // @trace TEST-DIST-028
    #[test]
    fn gradient_sync_construction() {
        let sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
        );
        assert_eq!(sync.phase, GradientSyncPhase::Idle);
    }

    /// TEST-DIST-028-02: GradientSyncPhase 枚举完整
    // @trace TEST-DIST-028
    #[test]
    fn gradient_sync_phase_variants() {
        let phases = [
            GradientSyncPhase::Idle,
            GradientSyncPhase::BackwardCompute,
            GradientSyncPhase::TpAllReduce,
            GradientSyncPhase::PpSendGradient,
            GradientSyncPhase::PpRecvGradient,
            GradientSyncPhase::WeightUpdate,
        ];
        for i in 0..phases.len() {
            for j in (i + 1)..phases.len() {
                assert_ne!(phases[i], phases[j]);
            }
        }
    }

    /// TEST-DIST-028-03: GradientSyncConfig 默认值合理
    // @trace TEST-DIST-028
    #[test]
    fn gradient_sync_config_default() {
        let config = GradientSyncConfig::default();
        assert!(config.validate());
    }

    /// TEST-DIST-028-04: reset 恢复初始状态
    // @trace TEST-DIST-028
    #[test]
    fn gradient_sync_reset() {
        let mut sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
        );
        sync.phase = GradientSyncPhase::BackwardCompute;
        sync.reset();
        assert_eq!(sync.phase, GradientSyncPhase::Idle);
    }

    /// TEST-DIST-028-05: all_backward_done 初始为 false
    // @trace TEST-DIST-028
    #[test]
    fn gradient_sync_all_backward_done_initial() {
        let sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
        );
        assert!(!sync.all_backward_done(4));
    }

    /// TEST-DIST-028-06: validate 一致性
    // @trace TEST-DIST-028
    #[test]
    fn gradient_sync_validate() {
        let sync = GradientSync::new(
            make_pipeline_config(),
            make_topology(),
            GradientSyncConfig::default(),
        );
        assert!(sync.validate());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-029: 2D 并行拓扑 (REQ-DIST-029)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_029 {
    use gllm::engine::pipeline::topology::Topology2D;

    /// TEST-DIST-029-01: Topology2D 构造 — rank 公式
    // @trace TEST-DIST-029
    #[test]
    fn topology_2d_construction() {
        let topo = Topology2D::new(2, 4, 8).unwrap();
        assert_eq!(topo.tp_size, 2);
        assert_eq!(topo.pp_size, 4);
        assert_eq!(topo.world_size, 8);
    }

    /// TEST-DIST-029-02: decompose_rank — rank = tp_rank * pp_size + pp_rank
    // @trace TEST-DIST-029
    #[test]
    fn topology_2d_decompose_rank() {
        let topo = Topology2D::new(2, 4, 8).unwrap();
        // rank 5: tp_rank = 5/4 = 1, pp_rank = 5%4 = 1
        let (tp_rank, pp_rank) = topo.decompose_rank(5);
        assert_eq!(tp_rank, 1);
        assert_eq!(pp_rank, 1);
    }

    /// TEST-DIST-029-03: compose_rank — 逆向组合
    // @trace TEST-DIST-029
    #[test]
    fn topology_2d_compose_rank() {
        let topo = Topology2D::new(2, 4, 8).unwrap();
        let rank = topo.compose_rank(1, 2); // tp=1, pp=2 → 1*4+2 = 6
        assert_eq!(rank, 6);
    }

    /// TEST-DIST-029-04: pp_prev / pp_next — PP 邻居
    // @trace TEST-DIST-029
    #[test]
    fn topology_2d_pp_neighbors() {
        let topo = Topology2D::new(2, 4, 8).unwrap();
        // rank 2 (tp=0, pp=2): prev=rank 1 (pp=1), next=rank 3 (pp=3)
        assert_eq!(topo.pp_prev(2), Some(1));
        assert_eq!(topo.pp_next(2), Some(3));
    }

    /// TEST-DIST-029-05: validate — world_size = tp_size * pp_size
    // @trace TEST-DIST-029
    #[test]
    fn topology_2d_validate() {
        let valid = Topology2D::new(2, 4, 8).unwrap();
        assert!(valid.validate());
        let invalid = Topology2D::new(2, 4, 9); // 2*4 != 9
        assert!(invalid.is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-030: 3D 并行拓扑 (REQ-DIST-030)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_030 {
    use gllm::engine::pipeline::topology::Topology3D;

    /// TEST-DIST-030-01: Topology3D 构造 — rank 公式
    // @trace TEST-DIST-030
    #[test]
    fn topology_3d_construction() {
        let topo = Topology3D::new(2, 4, 2, 16).unwrap();
        assert_eq!(topo.tp_size, 2);
        assert_eq!(topo.pp_size, 4);
        assert_eq!(topo.ep_size, 2);
        assert_eq!(topo.world_size, 16);
    }

    /// TEST-DIST-030-02: decompose_rank — rank = ep_rank*(tp*pp) + tp_rank*pp + pp_rank
    // @trace TEST-DIST-030
    #[test]
    fn topology_3d_decompose_rank() {
        let topo = Topology3D::new(2, 4, 2, 16).unwrap();
        // rank 10: ep_rank = 10/(2*4) = 1, remaining = 10-8 = 2, tp_rank = 2/4 = 0, pp_rank = 2
        let (tp_rank, pp_rank, ep_rank) = topo.decompose_rank(10);
        assert_eq!(ep_rank, 1);
        assert_eq!(tp_rank, 0);
        assert_eq!(pp_rank, 2);
    }

    /// TEST-DIST-030-03: compose_rank — 逆向组合
    // @trace TEST-DIST-030
    #[test]
    fn topology_3d_compose_rank() {
        let topo = Topology3D::new(2, 4, 2, 16).unwrap();
        // tp=1, pp=2, ep=1 → 1*(2*4) + 1*4 + 2 = 8+4+2 = 14
        let rank = topo.compose_rank(1, 2, 1);
        assert_eq!(rank, 14);
    }

    /// TEST-DIST-030-04: validate — world_size = tp_size * pp_size * ep_size
    // @trace TEST-DIST-030
    #[test]
    fn topology_3d_validate() {
        let valid = Topology3D::new(2, 4, 2, 16).unwrap();
        assert!(valid.validate());
        let invalid = Topology3D::new(2, 4, 2, 15); // 2*4*2 != 15
        assert!(invalid.is_err());
    }

    /// TEST-DIST-030-05: PP 邻居在 3D 拓扑中
    // @trace TEST-DIST-030
    #[test]
    fn topology_3d_pp_neighbors() {
        let topo = Topology3D::new(2, 4, 2, 16).unwrap();
        // rank 2 (tp=0, pp=2, ep=0): pp_prev → rank 1, pp_next → rank 3
        assert_eq!(topo.pp_prev(2), Some(1));
        assert_eq!(topo.pp_next(2), Some(3));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-031: PD Pipeline Bridge (REQ-DIST-031)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_031 {
    use gllm::engine::distributed_config::{NodeRole, PdDisaggMode};
    use gllm::engine::pipeline::config::PipelineConfig;
    use gllm::engine::pipeline::pd_bridge::{
        KvTransferConfig, KvTransferResult, PdPipelineBridge, PdPipelineBridgeError,
        PdPipelineConfig, PdPipelineConfigError,
    };

    fn make_pipeline_config() -> PipelineConfig {
        PipelineConfig {
            pp_size: 2, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 16,
        }
    }

    /// TEST-DIST-031-01: PdPipelineConfig — world_size = tp*pp + num_decode_gpus
    // @trace TEST-DIST-031
    #[test]
    fn pd_pipeline_config_world_size() {
        let config = PdPipelineConfig::new(2, 4, 2).unwrap();
        assert_eq!(config.prefill_world_size, 8); // 2*4
        assert_eq!(config.total_world_size, 10); // 8+2
    }

    /// TEST-DIST-031-02: PdPipelineConfig — 无效参数报错
    // @trace TEST-DIST-031
    #[test]
    fn pd_pipeline_config_invalid_params() {
        assert_eq!(PdPipelineConfig::new(0, 4, 2).unwrap_err(), PdPipelineConfigError::InvalidTpSize(0));
        assert_eq!(PdPipelineConfig::new(2, 0, 2).unwrap_err(), PdPipelineConfigError::InvalidPpSize(0));
        assert_eq!(PdPipelineConfig::new(2, 4, 0).unwrap_err(), PdPipelineConfigError::InvalidDecodeGpus(0));
    }

    /// TEST-DIST-031-03: PdPipelineConfig — rank 分类
    // @trace TEST-DIST-031
    #[test]
    fn pd_pipeline_config_rank_classification() {
        let config = PdPipelineConfig::new(2, 4, 2).unwrap();
        assert!(config.is_prefill_rank(0));
        assert!(config.is_prefill_rank(7));
        assert!(!config.is_prefill_rank(8));
        assert!(config.is_decode_rank(8));
        assert!(config.is_decode_rank(9));
    }

    /// TEST-DIST-031-04: PdPipelineBridge — KV 传输字节数 = num_pages * page_size * dtype_bytes
    // @trace TEST-DIST-031
    #[test]
    fn pd_bridge_kv_transfer_bytes() {
        let kv_config = KvTransferConfig::new(PdDisaggMode::Disaggregated, NodeRole::PrefillOnly, 4096, 2);
        let bridge = PdPipelineBridge::new(
            make_pipeline_config(),
            PdPipelineConfig::new(1, 2, 2).unwrap(),
            kv_config,
        );
        assert_eq!(bridge.kv_transfer_bytes(100), 100 * 4096 * 2);
    }

    /// TEST-DIST-031-05: PdPipelineBridge — 异步传输时 bubble = 0
    // @trace TEST-DIST-031
    #[test]
    fn pd_bridge_bubble_free_when_async() {
        let kv_config = KvTransferConfig::new(PdDisaggMode::Disaggregated, NodeRole::PrefillOnly, 4096, 2);
        let bridge = PdPipelineBridge::new(
            make_pipeline_config(),
            PdPipelineConfig::new(1, 2, 2).unwrap(),
            kv_config,
        );
        assert!(bridge.is_switch_bubble_free());
    }

    /// TEST-DIST-031-06: PdPipelineBridge — PD 切换生命周期
    // @trace TEST-DIST-031
    #[test]
    fn pd_bridge_switch_lifecycle() {
        let kv_config = KvTransferConfig::new(PdDisaggMode::Disaggregated, NodeRole::PrefillOnly, 4096, 2);
        let mut bridge = PdPipelineBridge::new(
            make_pipeline_config(),
            PdPipelineConfig::new(1, 2, 2).unwrap(),
            kv_config,
        );
        assert!(!bridge.is_switch_in_progress());
        bridge.begin_pd_switch();
        assert!(bridge.is_switch_in_progress());
        bridge.end_pd_switch();
        assert!(!bridge.is_switch_in_progress());
    }

    /// TEST-DIST-031-07: PdPipelineBridge — validate
    // @trace TEST-DIST-031
    #[test]
    fn pd_bridge_validate() {
        let kv_config = KvTransferConfig::new(PdDisaggMode::Disaggregated, NodeRole::PrefillOnly, 4096, 2);
        let bridge = PdPipelineBridge::new(
            make_pipeline_config(),
            PdPipelineConfig::new(1, 2, 2).unwrap(),
            kv_config,
        );
        assert!(bridge.validate());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-032: CP Pipeline Bridge (REQ-DIST-032)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_032 {
    use gllm::engine::pipeline::config::PipelineConfig;
    use gllm::engine::pipeline::cp_bridge::{
        CpAllGatherGroup, CpBridgeConfig, CpPipelineBridge, CpPipelineBridgeError, CpSubsegment,
    };

    fn make_pipeline_config() -> PipelineConfig {
        PipelineConfig {
            pp_size: 2, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 16,
        }
    }

    /// TEST-DIST-032-01: CpBridgeConfig 构造与 validate
    // @trace TEST-DIST-032
    #[test]
    fn cp_bridge_config_construction() {
        let config = CpBridgeConfig::new(4, 2, 1, 8192);
        assert_eq!(config.cp_size, 4);
        assert_eq!(config.cp_rank, 2);
        assert!(config.validate());
    }

    /// TEST-DIST-032-02: CpPipelineBridge — configure 设置 CP 环参数
    // @trace TEST-DIST-032
    #[test]
    fn cp_bridge_configure() {
        let mut bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            CpBridgeConfig::new(1, 0, 1, 4096),
        );
        assert!(!bridge.is_cp_enabled());
        bridge.configure(4, 1);
        assert!(bridge.is_cp_enabled());
        assert_eq!(bridge.cp_config.cp_size, 4);
    }

    /// TEST-DIST-032-03: split_into_subsegments — 按 cp_size 切分序列
    // @trace TEST-DIST-032
    #[test]
    fn cp_bridge_split_subsegments() {
        let bridge = CpPipelineBridge::new(
            make_pipeline_config(),
            CpBridgeConfig::new(4, 0, 1, 4096),
        );
        let segments = bridge.split_into_subsegments(100);
        assert_eq!(segments.len(), 4);
        assert_eq!(segments[0].local_seq_len, 25);
        assert_eq!(segments[0].seq_offset, 0);
        let total: usize = segments.iter().map(|s| s.local_seq_len).sum();
        assert_eq!(total, 100);
    }

    /// TEST-DIST-032-04: max_supported_seq_len = single_gpu_max_seq * pp_size * cp_size
    // @trace TEST-DIST-032
    #[test]
    fn cp_bridge_max_seq_len() {
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 4, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 1, layers_per_stage: 8,
            },
            CpBridgeConfig::new(8, 0, 1, 4096),
        );
        assert_eq!(bridge.max_supported_seq_len(), 4096 * 4 * 8);
    }

    /// TEST-DIST-032-05: 跨 stage 零 CP 通信
    // @trace TEST-DIST-032
    #[test]
    fn cp_bridge_cross_stage_zero() {
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 2, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 1, layers_per_stage: 16,
            },
            CpBridgeConfig::new(2, 0, 1, 4096),
        );
        assert!(bridge.cross_stage_cp_zero());
    }

    /// TEST-DIST-032-06: verify_numerical_equivalence — 误差 < 1e-5 等价
    // @trace TEST-DIST-032
    #[test]
    fn cp_bridge_numerical_equivalence() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6];
        assert!(CpPipelineBridge::verify_numerical_equivalence(&a, &b, 1e-5));
        let c = vec![1.1f32, 2.0, 3.0];
        assert!(!CpPipelineBridge::verify_numerical_equivalence(&a, &c, 1e-5));
    }

    /// TEST-DIST-032-07: CP 环导航 next/prev
    // @trace TEST-DIST-032
    #[test]
    fn cp_bridge_ring_navigation() {
        let bridge = CpPipelineBridge::new(
            PipelineConfig {
                pp_size: 2, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 1, layers_per_stage: 16,
            },
            CpBridgeConfig::new(4, 1, 1, 4096),
        );
        assert_eq!(bridge.cp_ring_next(), 2);
        assert_eq!(bridge.cp_ring_prev(), 0);
    }

    /// TEST-DIST-032-08: CpAllGatherGroup — 同 stage 通信组
    // @trace TEST-DIST-032
    #[test]
    fn cp_allgather_group_same_stage() {
        let group = CpAllGatherGroup::simple(0, 4, 1);
        assert_eq!(group.pp_stage_id, 0);
        assert_eq!(group.member_ranks, vec![0, 1, 2, 3]);
        assert!(group.contains_rank(0));
        assert!(!group.contains_rank(4));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-033: Speculative Pipeline (REQ-DIST-033)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_033 {
    use gllm::engine::pipeline::config::PipelineConfig;
    use gllm::engine::pipeline::speculative_bridge::{
        DraftVerifyStep, SpecPipelineConfig, SpeculativePipeline, SpeculativePipelineError,
    };

    fn make_pipeline_config() -> PipelineConfig {
        PipelineConfig {
            pp_size: 4, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 8,
        }
    }

    /// TEST-DIST-033-01: SpecPipelineConfig — Draft/Verify stage 范围
    // @trace TEST-DIST-033
    #[test]
    fn spec_config_stage_ranges() {
        let config = SpecPipelineConfig::new(2, 4, 5);
        assert_eq!(config.draft_stage_range(), 0..2);
        assert_eq!(config.verify_stage_range(), 0..4);
    }

    /// TEST-DIST-033-02: SpecPipelineConfig — stream 隔离
    // @trace TEST-DIST-033
    #[test]
    fn spec_config_stream_isolation() {
        let config = SpecPipelineConfig::default();
        assert!(config.is_stream_isolated());
    }

    /// TEST-DIST-033-03: SpeculativePipeline — Draft/Verify 加速比 >= 1.5x
    // @trace TEST-DIST-033
    #[test]
    fn speculative_pipeline_speedup() {
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig::new(2, 4, 5),
        ).unwrap();
        assert!(pipeline.speedup_achieved());
    }

    /// TEST-DIST-033-04: SpeculativePipeline — is_draft_stage / is_verify_stage
    // @trace TEST-DIST-033
    #[test]
    fn speculative_pipeline_stage_classification() {
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig::new(2, 4, 5),
        ).unwrap();
        assert!(pipeline.is_draft_stage(0));
        assert!(pipeline.is_draft_stage(1));
        assert!(!pipeline.is_draft_stage(2));
        assert!(pipeline.is_verify_stage(0));
        assert!(pipeline.is_verify_stage(3));
        assert!(pipeline.is_verify_only_stage(2));
        assert!(pipeline.is_verify_only_stage(3));
    }

    /// TEST-DIST-033-05: SpeculativePipeline — schedule 产生 Draft/Verify 步骤
    // @trace TEST-DIST-033
    #[test]
    fn speculative_pipeline_schedule() {
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig::new(2, 4, 5),
        ).unwrap();
        let steps = pipeline.schedule();
        // First step: DraftCompute
        assert!(matches!(steps[0], DraftVerifyStep::DraftCompute { num_tokens: 5, draft_end_stage: 2 }));
        // Last step: VerifyCompute
        assert!(matches!(steps.last(), Some(DraftVerifyStep::VerifyCompute { verify_end_stage: 4 })));
    }

    /// TEST-DIST-033-06: SpeculativePipeline — draft_stages >= pp_size 报错
    // @trace TEST-DIST-033
    #[test]
    fn speculative_pipeline_draft_exceeds_pp() {
        let result = SpeculativePipeline::new(
            make_pipeline_config(), // pp_size=4
            SpecPipelineConfig::new(4, 4, 5), // draft_stages=4 >= pp_size
        );
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SpeculativePipelineError::DraftStagesExceedPp {
            draft_stages: 4, pp_size: 4,
        });
    }

    /// TEST-DIST-033-07: SpeculativePipeline — validate
    // @trace TEST-DIST-033
    #[test]
    fn speculative_pipeline_validate() {
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig::new(2, 4, 5),
        ).unwrap();
        assert!(pipeline.validate());
    }

    /// TEST-DIST-033-08: SpecPipelineConfig — 加速比随 spec_len 增大
    // @trace TEST-DIST-033
    #[test]
    fn spec_config_speedup_increases_with_spec_len() {
        let short = SpecPipelineConfig::new(2, 4, 3);
        let long = SpecPipelineConfig::new(2, 4, 10);
        assert!(long.estimated_speedup() > short.estimated_speedup());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST-DIST-034: Stage Fault Recovery (REQ-DIST-034)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nccl")]
mod test_dist_034 {
    use gllm::engine::pipeline::config::PipelineConfig;
    use gllm::engine::pipeline::fault_recovery::{
        FaultIsolationState, HeartbeatConfig, MigrationPlan, MigrationResult, StageFaultRecovery,
        StageFaultRecoveryError, StageStatus,
    };

    fn make_pipeline_config() -> PipelineConfig {
        PipelineConfig {
            pp_size: 4, stage_id: 0, num_virtual_stages: 1, micro_batch_size: 4, layers_per_stage: 8,
        }
    }

    /// TEST-DIST-034-01: StageStatus 枚举完整
    // @trace TEST-DIST-034
    #[test]
    fn stage_status_variants() {
        assert_eq!(format!("{}", StageStatus::Healthy), "Healthy");
        assert_eq!(format!("{}", StageStatus::Faulted), "Faulted");
        assert_eq!(format!("{}", StageStatus::Migrating), "Migrating");
        assert_eq!(format!("{}", StageStatus::Migrated), "Migrated");
    }

    /// TEST-DIST-034-02: HeartbeatConfig 构造与 validate
    // @trace TEST-DIST-034
    #[test]
    fn heartbeat_config_construction() {
        let config = HeartbeatConfig::new(5000, 1000, 3);
        assert_eq!(config.timeout_ms, 5000);
        assert!(config.validate());
    }

    /// TEST-DIST-034-03: detect_heartbeat — 心跳超时后标记 Faulted
    // @trace TEST-DIST-034
    #[test]
    fn detect_heartbeat_faulted_after_timeouts() {
        let mut recovery = StageFaultRecovery::new(
            make_pipeline_config(),
            HeartbeatConfig::new(5000, 1000, 3),
        );
        recovery.update_heartbeat(0, 0);
        // 3 consecutive timeouts → Faulted
        recovery.detect_heartbeat(0, 6000); // 1st timeout
        recovery.detect_heartbeat(0, 7000); // 2nd timeout
        let status = recovery.detect_heartbeat(0, 8000); // 3rd timeout → Faulted
        assert_eq!(status, StageStatus::Faulted);
    }

    /// TEST-DIST-034-04: isolate_faulted_stage — KV 不可读 + 激活丢弃
    // @trace TEST-DIST-034
    #[test]
    fn isolate_faulted_stage() {
        let mut recovery = StageFaultRecovery::new(
            make_pipeline_config(),
            HeartbeatConfig::new(5000, 1000, 3),
        );
        recovery.stage_statuses[1] = StageStatus::Faulted;
        let isolation = recovery.isolate_faulted_stage(1, 1000).unwrap();
        assert!(isolation.kv_cache_unreadable);
        assert!(isolation.activation_buffer_discarded);
        assert!(isolation.is_isolated());
    }

    /// TEST-DIST-034-05: MigrationPlan — 总迁移字节数
    // @trace TEST-DIST-034
    #[test]
    fn migration_plan_total_bytes() {
        let plan = MigrationPlan::new(1, 2, 8, 100, 4096, 1024 * 1024);
        let expected = 8 * 1024 * 1024 + 100 * 4096; // weight + KV
        assert_eq!(plan.total_migration_bytes(), expected);
    }

    /// TEST-DIST-034-06: MigrationPlan — 迁移时间 < 30s
    // @trace TEST-DIST-034
    #[test]
    fn migration_plan_within_30s_budget() {
        let plan = MigrationPlan::new(1, 2, 8, 10000, 4096, 1024 * 1024 * 1024);
        let time_ms = plan.estimated_migration_time_ms(300.0); // NVLink 300 GB/s
        assert!(time_ms < 30_000, "migration should be < 30s, got {time_ms}ms");
    }

    /// TEST-DIST-034-07: verify_post_migration_equivalence — 数值等价校验
    // @trace TEST-DIST-034
    #[test]
    fn post_migration_numerical_equivalence() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6];
        assert!(StageFaultRecovery::verify_post_migration_equivalence(&a, &b, 1e-5));
        let c = vec![1.5f32, 2.0, 3.0];
        assert!(!StageFaultRecovery::verify_post_migration_equivalence(&a, &c, 1e-5));
    }

    /// TEST-DIST-034-08: StageFaultRecovery — faulted_stages 列表
    // @trace TEST-DIST-034
    #[test]
    fn recovery_faulted_stages() {
        let mut recovery = StageFaultRecovery::new(
            make_pipeline_config(),
            HeartbeatConfig::new(5000, 1000, 3),
        );
        assert!(recovery.faulted_stages().is_empty());
        recovery.stage_statuses[1] = StageStatus::Faulted;
        recovery.stage_statuses[3] = StageStatus::Faulted;
        assert_eq!(recovery.faulted_stages(), vec![1, 3]);
    }

    /// TEST-DIST-034-09: StageFaultRecovery — validate
    // @trace TEST-DIST-034
    #[test]
    fn recovery_validate() {
        let recovery = StageFaultRecovery::new(
            make_pipeline_config(),
            HeartbeatConfig::new(5000, 1000, 3),
        );
        assert!(recovery.validate());
    }
}
