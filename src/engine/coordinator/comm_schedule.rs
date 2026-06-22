//! 通信调度与算法选择 (REQ-DIST-007, REQ-DIST-008, REQ-DIST-009, REQ-DIST-010)
//!
//! 当 `nccl` feature 启用时，本模块提供：
//! - CommScheduleDecision: 通信-计算重叠调度决策
//! - QuantCommDecision: 量化通信压缩决策（连接 gllm-nccl QuantizedComm）
//! - AlgorithmOverride: 通信算法覆盖（连接 gllm-nccl select_algorithm）

#[cfg(feature = "nccl")]
pub mod comm_schedule {
    use crate::engine::distributed_config::{CommConfig, CommHandleWrapper};
    use crate::engine::intent_bias::OverlapHint;

    // ─── L2-2: 通信调度决策 (REQ-DIST-007, REQ-DIST-008) ───

    /// 通信调度决策 (REQ-DIST-007, REQ-DIST-008)
    /// 也用于 REQ-DIST-005 TP AllReduce 通信调度
    // @trace REQ-DIST-005 [entity:ENT-DIST-TP-COMM] [controlflow:CF-DIST-001]
    // @trace REQ-DIST-007 [entity:ENT-DIST-TP-COMM] [controlflow:CF-DIST-003]
    // @trace REQ-DIST-008 [entity:ENT-DIST-TP-COMM] [controlflow:CF-DIST-004]
    #[derive(Debug, Clone, PartialEq)]
    pub enum CommScheduleDecision {
        /// 标准 AllReduce，无特殊调度
        StandardAllReduce,
        /// 双缓冲重叠 (REQ-DIST-007)：通信和计算交替进行
        /// comm_sm_ratio 指定通信 SM 占总 SM 的比例（0.0~1.0）
        DoubleBuffer {
            /// 通信 SM 占比
            comm_sm_ratio: f32,
        },
        /// FLUX 分解 (REQ-DIST-008)：AllReduce → ReduceScatter + LocalCompute + AllGather
        /// ring_size 通常 = world_size
        FluxDecompose {
            /// Ring 大小（通常 = world_size）
            ring_size: u32,
        },
        /// 列并行通信 (REQ-DIST-006)：DownProj 后 ReduceScatter；AllGather 前置到下一层 QKV
        ColumnParallel {
            /// 列并行通信策略
            strategy: ColumnParallelStrategy,
        },
    }

    /// 列并行通信策略 (REQ-DIST-006)
    ///
    /// 描述列并行层使用的通信原语：
    /// - `ReduceScatter`：DownProj 后，将各 rank 的部分结果按列分片求和聚合
    /// - `AllGather`：下一层 QKV 前，将各 rank 的分片聚合为完整结果
    // @trace REQ-DIST-006 [entity:ENT-DIST-TP-COMM] [dataflow:DF-DIST-003]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ColumnParallelStrategy {
        /// ReduceScatter：DownProj 后调用，将各 rank 部分结果按列分片求和
        ReduceScatter,
        /// AllGather：下一层 QKV 前调用，将各 rank 分片聚合为完整结果
        AllGather,
    }

    impl CommScheduleDecision {
        /// 生成通信计划 (CommPlan) 用于实际的通信执行 (REQ-DIST-007, REQ-DIST-008)
        ///
        /// - DoubleBuffer (REQ-DIST-007): 调用 gllm-nccl `lower_to_comm_plan()` 生成
        ///   带 `OverlapMode::DoubleBuffer` 调度提示的 CommPlan。通信 stream 与计算
        ///   stream 并行执行，Layer N 通信与 Layer N+1 计算重叠。
        /// - FluxDecompose (REQ-DIST-008): 调用 gllm-nccl `lower_to_comm_plan()` 生成
        ///   ReduceScatter + LocalCompute + AllGather 三阶段流水线 CommPlan。
        /// - ColumnParallel (REQ-DIST-006): ReduceScatter/AllGather 通过
        ///   CommHandleWrapper 直接执行，不走 CommPlan 路径。
        /// - StandardAllReduce: 返回 None（使用标准 AllReduce 路径）。
        // @trace REQ-DIST-007 [entity:ENT-DIST-TP-COMM] [dataflow:DF-DIST-004]
        // @trace REQ-DIST-008 [entity:ENT-DIST-TP-COMM] [dataflow:DF-DIST-005]
        pub fn build_comm_plan(
            &self,
            topology: &gllm_nccl::Topology,
            dtype: gllm_nccl::DType,
            reduce_op: gllm_nccl::ReduceOp,
            msg_bytes: usize,
            rank: u32,
            world_size: u32,
            total_sms: u32,
        ) -> Option<Result<gllm_nccl::comm_ir::CommPlan, String>> {
            match self {
                Self::StandardAllReduce => None,
                Self::DoubleBuffer { comm_sm_ratio } => {
                    // REQ-DIST-007: DoubleBuffer 双缓冲通信-计算重叠。
                    // 调用 gllm-nccl lower_to_comm_plan 生成基础 CommPlan，
                    // 然后覆写 schedule 为 DoubleBuffer 模式：
                    // - overlap_mode = DoubleBuffer（通信/计算交替）
                    // - buffer_slots = 2（双缓冲）
                    // - comm_trigger = ConcurrentWithCompute（通信与计算并发触发）
                    // - min_tile_bytes = msg_bytes / world_size（按 rank 分片粒度）
                    let result = gllm_nccl::lower::lower_to_comm_plan(
                        topology,
                        gllm_nccl::CollectiveOp::AllReduce,
                        dtype,
                        reduce_op,
                        msg_bytes,
                        rank,
                        world_size,
                        total_sms,
                    )
                    .map(|mut plan| {
                        // 覆写调度提示为 DoubleBuffer 模式
                        plan.schedule = gllm_nccl::comm_ir::CommScheduleHint {
                            overlap_mode: gllm_nccl::comm_ir::OverlapMode::DoubleBuffer,
                            buffer_slots: 2,
                            comm_trigger: gllm_nccl::comm_ir::CommTrigger::ConcurrentWithCompute,
                            min_tile_bytes: msg_bytes as u64 / world_size.max(1) as u64,
                        };
                        log::trace!(
                            "[comm_schedule] REQ-DIST-007: DoubleBuffer CommPlan generated — \
                             comm_sm_ratio={:.2}, buffer_slots=2, overlap=DoubleBuffer, \
                             min_tile_bytes={}, rank={}, world_size={}",
                            comm_sm_ratio,
                            msg_bytes as u64 / world_size.max(1) as u64,
                            rank,
                            world_size,
                        );
                        plan
                    })
                    .map_err(|e| format!("DoubleBuffer lower_to_comm_plan failed: {:?}", e));
                    Some(result)
                }
                Self::FluxDecompose { ring_size } => {
                    // REQ-DIST-008: FluxDecompose FLUX 分解调度。
                    // 调用 gllm-nccl lower_to_comm_plan，当拓扑支持 FluxPipeline 时
                    // 自动选择 flux_decompose() 生成三阶段流水线：
                    // ReduceScatter → LocalCompute → AllGather
                    let result = gllm_nccl::lower::lower_to_comm_plan(
                        topology,
                        gllm_nccl::CollectiveOp::AllReduce,
                        dtype,
                        reduce_op,
                        msg_bytes,
                        rank,
                        world_size,
                        total_sms,
                    )
                    .map(|plan| {
                        log::trace!(
                            "[comm_schedule] REQ-DIST-008: FluxDecompose CommPlan generated — \
                             ring_size={}, overlap={:?}, buffer_slots={}, nodes={}, \
                             rank={}, world_size={}",
                            ring_size,
                            plan.schedule.overlap_mode,
                            plan.schedule.buffer_slots,
                            plan.nodes.len(),
                            rank,
                            world_size,
                        );
                        plan
                    })
                    .map_err(|e| format!("FluxDecompose lower_to_comm_plan failed: {:?}", e));
                    Some(result)
                }
                // ColumnParallel (REQ-DIST-006): ReduceScatter/AllGather 通过
                // CommHandleWrapper 直接执行，不生成 CommPlan。
                Self::ColumnParallel { .. } => None,
            }
        }
    }

    /// 根据通信配置和硬件状态决定通信调度策略 (REQ-DIST-007, REQ-DIST-008)
    ///
    /// 决策逻辑：
    /// - 单机模式 → StandardAllReduce
    /// - ForceFlux → FluxDecompose（ring_size = world_size）
    /// - ForceDoubleBuffer → DoubleBuffer（comm_sm_ratio = 0.3）
    /// - Auto/PreferOverlap/PreferIsolated → StandardAllReduce
    ///   （未来可根据拓扑信息自动选择 DoubleBuffer 或 FluxDecompose）
    // @trace REQ-DIST-007 [entity:ENT-DIST-TP-COMM] [controlflow:CF-DIST-003]
    // @trace REQ-DIST-008 [entity:ENT-DIST-TP-COMM] [controlflow:CF-DIST-004]
    pub fn resolve_comm_schedule(
        comm_config: &CommConfig,
        comm_handle: &CommHandleWrapper,
    ) -> CommScheduleDecision {
        if !comm_handle.is_distributed() {
            return CommScheduleDecision::StandardAllReduce;
        }

        match comm_config.overlap {
            OverlapHint::Auto => CommScheduleDecision::StandardAllReduce,
            OverlapHint::PreferOverlap => CommScheduleDecision::StandardAllReduce,
            OverlapHint::PreferIsolated => CommScheduleDecision::StandardAllReduce,
            OverlapHint::ForceDoubleBuffer => CommScheduleDecision::DoubleBuffer {
                comm_sm_ratio: 0.3,
            },
            OverlapHint::ForceFlux => CommScheduleDecision::FluxDecompose {
                ring_size: comm_handle.world_size(),
            },
        }
    }

    /// 列并行层通信调度决策 (REQ-DIST-006)
    ///
    /// 根据层类型决定列并行通信使用 ReduceScatter 还是 AllGather：
    /// - DownProj 后 → ReduceScatter：各 rank 的部分结果按列分片求和聚合
    /// - QKV 前置 → AllGather：将各 rank 分片聚合为完整结果
    ///
    /// 当 `tp_size == 1` 时返回 None（单机无需列并行通信）。
    // @trace REQ-DIST-006 [entity:ENT-DIST-TP-COMM] [dataflow:DF-DIST-003] [controlflow:CF-DIST-002]
    pub fn resolve_column_parallel_schedule(
        tp_size: u32,
        strategy: ColumnParallelStrategy,
    ) -> Option<CommScheduleDecision> {
        if tp_size <= 1 {
            return None;
        }
        Some(CommScheduleDecision::ColumnParallel { strategy })
    }

    // ─── L2-3: 量化通信决策 (REQ-DIST-009) ───

    /// 量化通信决策 (REQ-DIST-009)
    ///
    /// 连接 gllm-nccl `QuantizedComm` 实现实际的量化/反量化。
    /// `quant_scheme` 字段映射到 gllm-nccl `QuantScheme`：
    /// - "fp8_e4m3" → QuantScheme::Fp8E4M3
    /// - "fp8_e5m2" → QuantScheme::Fp8E5M2
    /// - "int8" → QuantScheme::Int8Symmetric
    /// - "bf16" → QuantScheme::Bf16
    /// - "fp16" → QuantScheme::Fp16
    #[derive(Debug, Clone, PartialEq)]
    pub enum QuantCommDecision {
        /// 不压缩（默认回退）
        NoCompression,
        /// 自动决策（根据 buffer 大小和延迟阈值）
        AutoCompress,
        /// 始终量化压缩
        AlwaysCompress {
            /// 量化方案名称
            quant_scheme: String,
        },
        /// 永不压缩
        NeverCompress,
    }

    impl QuantCommDecision {
        /// 从 CommCompressHint 构造决策 (REQ-DIST-009)
        pub fn from_hint(hint: &crate::engine::distributed_config::CommCompressHint) -> Self {
            use crate::engine::distributed_config::CommCompressHint;
            match hint {
                CommCompressHint::Auto => Self::AutoCompress,
                CommCompressHint::AlwaysCompress => Self::AlwaysCompress {
                    quant_scheme: "fp8_e4m3".to_string(),
                },
                CommCompressHint::NeverCompress => Self::NeverCompress,
                CommCompressHint::ForceQuant => Self::AlwaysCompress {
                    quant_scheme: "fp8_e4m3".to_string(),
                },
            }
        }

        /// 将 quant_scheme 名称转换为 gllm-nccl QuantScheme
        ///
        /// 返回 None 表示无需量化（NoCompression/NeverCompress/AutoCompress），
        /// 返回 Some(QuantScheme) 表示应执行量化。
        pub fn to_quant_scheme(&self) -> Option<gllm_nccl::lower_types::QuantScheme> {
            use gllm_nccl::lower_types::QuantScheme;
            match self {
                Self::NoCompression | Self::NeverCompress | Self::AutoCompress => None,
                Self::AlwaysCompress { quant_scheme } => match quant_scheme.as_str() {
                    "fp8_e4m3" => Some(QuantScheme::Fp8E4M3),
                    "fp8_e5m2" => Some(QuantScheme::Fp8E5M2),
                    "int8" => Some(QuantScheme::Int8Symmetric),
                    "bf16" => Some(QuantScheme::Bf16),
                    "fp16" => Some(QuantScheme::Fp16),
                    _ => None,
                },
            }
        }

        /// 构建 gllm-nccl QuantizedComm 描述符 (REQ-DIST-009)
        ///
        /// 如果量化激活，返回 `QuantizedComm` 实例用于生成
        /// pre-send quantize / post-recv dequantize 指令。
        /// 如果无需量化，返回 None。
        pub fn build_quantized_comm(&self) -> Option<gllm_nccl::lower::quantized_comm::QuantizedComm> {
            let quant_scheme = self.to_quant_scheme()?;
            let target_dtype = quant_scheme.to_comm_dtype();
            // 源 dtype 默认 FP32（通信基线精度）
            let src_dtype = gllm_nccl::comm_ir::dtype::CommDType::Fp32;
            Some(gllm_nccl::lower::quantized_comm::QuantizedComm::new(
                target_dtype, src_dtype,
            ))
        }

        /// 计算压缩后的带宽节省倍数
        ///
        /// 返回相对于 FP32 的带宽节省因子（1 = 无节省，4 = 4x 节省）。
        pub fn bandwidth_saving(&self) -> usize {
            self.to_quant_scheme()
                .map(|qs| qs.bandwidth_saving())
                .unwrap_or(1)
        }

        /// 计算压缩后的元素字节数
        pub fn compressed_elem_size(&self) -> usize {
            self.to_quant_scheme()
                .map(|qs| qs.compressed_elem_size())
                .unwrap_or(4) // FP32 default
        }
    }

    // ─── L2-4: 算法选择覆盖 (REQ-DIST-010) ───

    /// 通信算法覆盖 (REQ-DIST-010)
    ///
    /// 当 CommConfig.algorithm_override 非空时，强制使用指定算法，
    /// 覆盖 gllm-nccl select_algorithm() 的自动选择。
    #[derive(Debug, Clone, PartialEq)]
    pub struct AlgorithmOverride {
        pub algorithm_name: String,
    }

    impl AlgorithmOverride {
        /// 从 CommConfig 构造算法覆盖 (REQ-DIST-010)
        ///
        /// `CommConfig.algorithm_override` 类型为 `String`；
        /// 空字符串表示不覆盖（返回 None）。
        pub fn from_config(comm_config: &CommConfig) -> Option<Self> {
            if comm_config.algorithm_override.is_empty() {
                None
            } else {
                Some(Self {
                    algorithm_name: comm_config.algorithm_override.clone(),
                })
            }
        }

        /// 解析算法名称为 gllm-nccl CollectiveAlgorithm 枚举 (REQ-DIST-010)
        ///
        /// 如果 algorithm_name 为空或无法识别，返回 None（使用自动选择）。
        /// 否则返回对应的 CollectiveAlgorithm 变体。
        pub fn resolve_algorithm(&self) -> Option<gllm_nccl::CollectiveAlgorithm> {
            match self.algorithm_name.as_str() {
                "Ring" => Some(gllm_nccl::CollectiveAlgorithm::Ring),
                "Tree" => Some(gllm_nccl::CollectiveAlgorithm::Tree),
                "TopoAwareRing" => Some(gllm_nccl::CollectiveAlgorithm::TopoAwareRing),
                "ChunkedPipeline" => Some(gllm_nccl::CollectiveAlgorithm::ChunkedPipeline),
                "HierarchicalRing" => Some(gllm_nccl::CollectiveAlgorithm::HierarchicalRing),
                "HardwareReduce" => Some(gllm_nccl::CollectiveAlgorithm::HardwareReduce),
                "FluxPipeline" => Some(gllm_nccl::CollectiveAlgorithm::FluxPipeline),
                "CollNetDirect" => Some(gllm_nccl::CollectiveAlgorithm::CollNetDirect),
                "Direct" => Some(gllm_nccl::CollectiveAlgorithm::Direct),
                _ => None,
            }
        }

        /// 根据拓扑和消息大小自动选择算法 (REQ-DIST-010)
        ///
        /// 如果有显式覆盖且可识别，使用覆盖值。
        /// 否则调用 gllm-nccl `select_algorithm()` 自动选择。
        pub fn select_or_auto(
            &self,
            topology: &gllm_nccl::Topology,
            op: gllm_nccl::CollectiveOp,
            msg_bytes: usize,
            world_size: usize,
            total_sms: u32,
        ) -> gllm_nccl::CollectiveAlgorithm {
            // 优先使用显式覆盖
            if let Some(algo) = self.resolve_algorithm() {
                return algo;
            }
            // 回退到 gllm-nccl 自动选择
            gllm_nccl::select_algorithm(topology, op, msg_bytes, world_size, total_sms)
                .unwrap_or(gllm_nccl::CollectiveAlgorithm::Ring)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::engine::distributed_config::CommCompressHint;

        fn make_distributed_handle() -> CommHandleWrapper {
            CommHandleWrapper::from_config(&crate::engine::distributed_config::ParallelConfig {
                tp_size: 4,
                pp_size: 1,
                ep_size: 1,
                rank: 0,
                world_size: 4,
                unique_id: String::new(),
            }).unwrap()
        }

        fn make_single_handle() -> CommHandleWrapper {
            CommHandleWrapper::from_config(
                &crate::engine::distributed_config::ParallelConfig::default(),
            ).unwrap()
        }

        fn make_comm_config(overlap: OverlapHint) -> CommConfig {
            CommConfig {
                overlap,
                ..Default::default()
            }
        }

        // ─── CommScheduleDecision 测试 ───

        #[test]
        fn comm_schedule_single_gpu_returns_standard() {
            let handle = make_single_handle();
            let config = make_comm_config(OverlapHint::ForceDoubleBuffer);
            assert_eq!(
                resolve_comm_schedule(&config, &handle),
                CommScheduleDecision::StandardAllReduce
            );
        }

        #[test]
        fn comm_schedule_auto_returns_standard() {
            let handle = make_distributed_handle();
            let config = make_comm_config(OverlapHint::Auto);
            assert_eq!(
                resolve_comm_schedule(&config, &handle),
                CommScheduleDecision::StandardAllReduce
            );
        }

        #[test]
        fn comm_schedule_prefer_overlap_returns_standard() {
            let handle = make_distributed_handle();
            let config = make_comm_config(OverlapHint::PreferOverlap);
            assert_eq!(
                resolve_comm_schedule(&config, &handle),
                CommScheduleDecision::StandardAllReduce
            );
        }

        #[test]
        fn comm_schedule_prefer_isolated_returns_standard() {
            let handle = make_distributed_handle();
            let config = make_comm_config(OverlapHint::PreferIsolated);
            assert_eq!(
                resolve_comm_schedule(&config, &handle),
                CommScheduleDecision::StandardAllReduce
            );
        }

        #[test]
        fn comm_schedule_force_double_buffer() {
            let handle = make_distributed_handle();
            let config = make_comm_config(OverlapHint::ForceDoubleBuffer);
            assert_eq!(
                resolve_comm_schedule(&config, &handle),
                CommScheduleDecision::DoubleBuffer {
                    comm_sm_ratio: 0.3
                }
            );
        }

        #[test]
        fn comm_schedule_force_flux() {
            let handle = make_distributed_handle();
            let config = make_comm_config(OverlapHint::ForceFlux);
            assert_eq!(
                resolve_comm_schedule(&config, &handle),
                CommScheduleDecision::FluxDecompose { ring_size: 4 }
            );
        }

        // ─── CommScheduleDecision::build_comm_plan 测试 ───

        #[test]
        fn comm_schedule_standard_returns_none_plan() {
            let decision = CommScheduleDecision::StandardAllReduce;
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            assert!(decision.build_comm_plan(
                &topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum,
                1024, 0, 4, 0
            ).is_none());
        }

        #[test]
        fn comm_schedule_flux_returns_some_plan() {
            let decision = CommScheduleDecision::FluxDecompose { ring_size: 4 };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            let result = decision.build_comm_plan(
                &topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum,
                512 * 1024, 0, 4, 0
            );
            assert!(result.is_some());
            // Without flux_config, lower_to_comm_plan selects Ring (not FluxPipeline)
            // but still returns a valid CommPlan
            let plan = result.unwrap();
            assert!(plan.is_ok());
        }

        // ─── REQ-DIST-007: DoubleBuffer build_comm_plan 测试 ───

        #[test]
        fn double_buffer_returns_some_plan() {
            let decision = CommScheduleDecision::DoubleBuffer { comm_sm_ratio: 0.3 };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            let result = decision.build_comm_plan(
                &topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum,
                512 * 1024, 0, 4, 0
            );
            assert!(result.is_some());
            let plan = result.unwrap();
            assert!(plan.is_ok());
        }

        #[test]
        fn double_buffer_plan_has_double_buffer_overlap_mode() {
            let decision = CommScheduleDecision::DoubleBuffer { comm_sm_ratio: 0.3 };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            let result = decision.build_comm_plan(
                &topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum,
                512 * 1024, 0, 4, 0
            );
            let plan = result.unwrap().unwrap();
            assert_eq!(plan.schedule.overlap_mode, gllm_nccl::comm_ir::OverlapMode::DoubleBuffer);
        }

        #[test]
        fn double_buffer_plan_has_two_buffer_slots() {
            let decision = CommScheduleDecision::DoubleBuffer { comm_sm_ratio: 0.3 };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            let result = decision.build_comm_plan(
                &topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum,
                512 * 1024, 0, 4, 0
            );
            let plan = result.unwrap().unwrap();
            assert_eq!(plan.schedule.buffer_slots, 2);
        }

        #[test]
        fn double_buffer_plan_has_concurrent_trigger() {
            let decision = CommScheduleDecision::DoubleBuffer { comm_sm_ratio: 0.3 };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            let result = decision.build_comm_plan(
                &topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum,
                512 * 1024, 0, 4, 0
            );
            let plan = result.unwrap().unwrap();
            assert_eq!(plan.schedule.comm_trigger, gllm_nccl::comm_ir::CommTrigger::ConcurrentWithCompute);
        }

        #[test]
        fn double_buffer_plan_min_tile_bytes_divided_by_world_size() {
            let decision = CommScheduleDecision::DoubleBuffer { comm_sm_ratio: 0.3 };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            let msg_bytes = 512 * 1024;
            let result = decision.build_comm_plan(
                &topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum,
                msg_bytes, 0, 4, 0
            );
            let plan = result.unwrap().unwrap();
            assert_eq!(plan.schedule.min_tile_bytes, msg_bytes as u64 / 4);
        }

        #[test]
        fn double_buffer_plan_preserves_allreduce_op() {
            let decision = CommScheduleDecision::DoubleBuffer { comm_sm_ratio: 0.3 };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            let result = decision.build_comm_plan(
                &topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum,
                512 * 1024, 0, 4, 0
            );
            let plan = result.unwrap().unwrap();
            assert_eq!(plan.op, gllm_nccl::comm_ir::CommOp::AllReduce);
        }

        // ─── ColumnParallel (REQ-DIST-006) 测试 ───

        #[test]
        fn column_parallel_reduce_scatter_decision() {
            let decision = CommScheduleDecision::ColumnParallel {
                strategy: ColumnParallelStrategy::ReduceScatter,
            };
            assert_eq!(
                decision,
                CommScheduleDecision::ColumnParallel {
                    strategy: ColumnParallelStrategy::ReduceScatter
                }
            );
        }

        #[test]
        fn column_parallel_all_gather_decision() {
            let decision = CommScheduleDecision::ColumnParallel {
                strategy: ColumnParallelStrategy::AllGather,
            };
            assert_eq!(
                decision,
                CommScheduleDecision::ColumnParallel {
                    strategy: ColumnParallelStrategy::AllGather
                }
            );
        }

        #[test]
        fn column_parallel_build_comm_plan_returns_none() {
            // ColumnParallel 不走 CommPlan 路径
            let decision = CommScheduleDecision::ColumnParallel {
                strategy: ColumnParallelStrategy::ReduceScatter,
            };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            assert!(decision.build_comm_plan(
                &topo, gllm_nccl::DType::Fp32, gllm_nccl::ReduceOp::Sum,
                1024, 0, 4, 0
            ).is_none());
        }

        #[test]
        fn resolve_column_parallel_schedule_tp1_returns_none() {
            assert!(resolve_column_parallel_schedule(1, ColumnParallelStrategy::ReduceScatter).is_none());
            assert!(resolve_column_parallel_schedule(1, ColumnParallelStrategy::AllGather).is_none());
        }

        #[test]
        fn resolve_column_parallel_schedule_tp0_returns_none() {
            assert!(resolve_column_parallel_schedule(0, ColumnParallelStrategy::ReduceScatter).is_none());
        }

        #[test]
        fn resolve_column_parallel_schedule_tp2_reduce_scatter() {
            let result = resolve_column_parallel_schedule(2, ColumnParallelStrategy::ReduceScatter);
            assert!(result.is_some());
            assert_eq!(
                result.unwrap(),
                CommScheduleDecision::ColumnParallel {
                    strategy: ColumnParallelStrategy::ReduceScatter
                }
            );
        }

        #[test]
        fn resolve_column_parallel_schedule_tp4_all_gather() {
            let result = resolve_column_parallel_schedule(4, ColumnParallelStrategy::AllGather);
            assert!(result.is_some());
            assert_eq!(
                result.unwrap(),
                CommScheduleDecision::ColumnParallel {
                    strategy: ColumnParallelStrategy::AllGather
                }
            );
        }

        #[test]
        fn column_parallel_strategy_equality() {
            assert_eq!(ColumnParallelStrategy::ReduceScatter, ColumnParallelStrategy::ReduceScatter);
            assert_eq!(ColumnParallelStrategy::AllGather, ColumnParallelStrategy::AllGather);
            assert_ne!(ColumnParallelStrategy::ReduceScatter, ColumnParallelStrategy::AllGather);
        }

        // ─── QuantCommDecision 测试 ───

        #[test]
        fn quant_comm_auto() {
            assert_eq!(
                QuantCommDecision::from_hint(&CommCompressHint::Auto),
                QuantCommDecision::AutoCompress
            );
        }

        #[test]
        fn quant_comm_always_compress() {
            assert_eq!(
                QuantCommDecision::from_hint(&CommCompressHint::AlwaysCompress),
                QuantCommDecision::AlwaysCompress {
                    quant_scheme: "fp8_e4m3".to_string()
                }
            );
        }

        #[test]
        fn quant_comm_never_compress() {
            assert_eq!(
                QuantCommDecision::from_hint(&CommCompressHint::NeverCompress),
                QuantCommDecision::NeverCompress
            );
        }

        #[test]
        fn quant_comm_force_quant() {
            assert_eq!(
                QuantCommDecision::from_hint(&CommCompressHint::ForceQuant),
                QuantCommDecision::AlwaysCompress {
                    quant_scheme: "fp8_e4m3".to_string()
                }
            );
        }

        #[test]
        fn quant_comm_to_quant_scheme_fp8_e4m3() {
            let decision = QuantCommDecision::AlwaysCompress {
                quant_scheme: "fp8_e4m3".to_string(),
            };
            assert_eq!(
                decision.to_quant_scheme(),
                Some(gllm_nccl::lower_types::QuantScheme::Fp8E4M3)
            );
        }

        #[test]
        fn quant_comm_to_quant_scheme_fp8_e5m2() {
            let decision = QuantCommDecision::AlwaysCompress {
                quant_scheme: "fp8_e5m2".to_string(),
            };
            assert_eq!(
                decision.to_quant_scheme(),
                Some(gllm_nccl::lower_types::QuantScheme::Fp8E5M2)
            );
        }

        #[test]
        fn quant_comm_to_quant_scheme_int8() {
            let decision = QuantCommDecision::AlwaysCompress {
                quant_scheme: "int8".to_string(),
            };
            assert_eq!(
                decision.to_quant_scheme(),
                Some(gllm_nccl::lower_types::QuantScheme::Int8Symmetric)
            );
        }

        #[test]
        fn quant_comm_to_quant_scheme_unknown_returns_none() {
            let decision = QuantCommDecision::AlwaysCompress {
                quant_scheme: "unknown".to_string(),
            };
            assert_eq!(decision.to_quant_scheme(), None);
        }

        #[test]
        fn quant_comm_to_quant_scheme_no_compression_returns_none() {
            assert_eq!(QuantCommDecision::NoCompression.to_quant_scheme(), None);
            assert_eq!(QuantCommDecision::NeverCompress.to_quant_scheme(), None);
            assert_eq!(QuantCommDecision::AutoCompress.to_quant_scheme(), None);
        }

        #[test]
        fn quant_comm_build_quantized_comm_fp8() {
            let decision = QuantCommDecision::AlwaysCompress {
                quant_scheme: "fp8_e4m3".to_string(),
            };
            let qc = decision.build_quantized_comm();
            assert!(qc.is_some());
            let qc = qc.unwrap();
            assert!(qc.is_active());
            assert_eq!(qc.compressed_elem_size(), 1); // FP8 = 1 byte
            assert_eq!(qc.bandwidth_saving(), 4); // 4 bytes → 1 byte
        }

        #[test]
        fn quant_comm_build_quantized_comm_no_compression_returns_none() {
            assert!(QuantCommDecision::NoCompression.build_quantized_comm().is_none());
            assert!(QuantCommDecision::NeverCompress.build_quantized_comm().is_none());
            assert!(QuantCommDecision::AutoCompress.build_quantized_comm().is_none());
        }

        #[test]
        fn quant_comm_bandwidth_saving_fp8() {
            let decision = QuantCommDecision::AlwaysCompress {
                quant_scheme: "fp8_e4m3".to_string(),
            };
            assert_eq!(decision.bandwidth_saving(), 4);
        }

        #[test]
        fn quant_comm_bandwidth_saving_int8() {
            let decision = QuantCommDecision::AlwaysCompress {
                quant_scheme: "int8".to_string(),
            };
            assert_eq!(decision.bandwidth_saving(), 4);
        }

        #[test]
        fn quant_comm_bandwidth_saving_no_compression() {
            assert_eq!(QuantCommDecision::NoCompression.bandwidth_saving(), 1);
            assert_eq!(QuantCommDecision::NeverCompress.bandwidth_saving(), 1);
        }

        #[test]
        fn quant_comm_compressed_elem_size_fp8() {
            let decision = QuantCommDecision::AlwaysCompress {
                quant_scheme: "fp8_e4m3".to_string(),
            };
            assert_eq!(decision.compressed_elem_size(), 1);
        }

        #[test]
        fn quant_comm_compressed_elem_size_no_compression() {
            assert_eq!(QuantCommDecision::NoCompression.compressed_elem_size(), 4);
        }

        // ─── AlgorithmOverride 测试 ───

        #[test]
        fn algorithm_override_empty_string_returns_none() {
            let config = CommConfig {
                algorithm_override: String::new(),
                ..Default::default()
            };
            assert_eq!(AlgorithmOverride::from_config(&config), None);
        }

        #[test]
        fn algorithm_override_non_empty_returns_some() {
            let config = CommConfig {
                algorithm_override: "Ring".to_string(),
                ..Default::default()
            };
            let override_ = AlgorithmOverride::from_config(&config);
            assert!(override_.is_some());
            assert_eq!(override_.unwrap().algorithm_name, "Ring");
        }

        #[test]
        fn algorithm_override_resolve_ring() {
            let ao = AlgorithmOverride { algorithm_name: "Ring".to_string() };
            assert_eq!(ao.resolve_algorithm(), Some(gllm_nccl::CollectiveAlgorithm::Ring));
        }

        #[test]
        fn algorithm_override_resolve_tree() {
            let ao = AlgorithmOverride { algorithm_name: "Tree".to_string() };
            assert_eq!(ao.resolve_algorithm(), Some(gllm_nccl::CollectiveAlgorithm::Tree));
        }

        #[test]
        fn algorithm_override_resolve_flux_pipeline() {
            let ao = AlgorithmOverride { algorithm_name: "FluxPipeline".to_string() };
            assert_eq!(ao.resolve_algorithm(), Some(gllm_nccl::CollectiveAlgorithm::FluxPipeline));
        }

        #[test]
        fn algorithm_override_resolve_hardware_reduce() {
            let ao = AlgorithmOverride { algorithm_name: "HardwareReduce".to_string() };
            assert_eq!(ao.resolve_algorithm(), Some(gllm_nccl::CollectiveAlgorithm::HardwareReduce));
        }

        #[test]
        fn algorithm_override_resolve_chunked_pipeline() {
            let ao = AlgorithmOverride { algorithm_name: "ChunkedPipeline".to_string() };
            assert_eq!(ao.resolve_algorithm(), Some(gllm_nccl::CollectiveAlgorithm::ChunkedPipeline));
        }

        #[test]
        fn algorithm_override_resolve_hierarchical_ring() {
            let ao = AlgorithmOverride { algorithm_name: "HierarchicalRing".to_string() };
            assert_eq!(ao.resolve_algorithm(), Some(gllm_nccl::CollectiveAlgorithm::HierarchicalRing));
        }

        #[test]
        fn algorithm_override_resolve_topo_aware_ring() {
            let ao = AlgorithmOverride { algorithm_name: "TopoAwareRing".to_string() };
            assert_eq!(ao.resolve_algorithm(), Some(gllm_nccl::CollectiveAlgorithm::TopoAwareRing));
        }

        #[test]
        fn algorithm_override_resolve_unknown_returns_none() {
            let ao = AlgorithmOverride { algorithm_name: "UnknownAlgo".to_string() };
            assert_eq!(ao.resolve_algorithm(), None);
        }

        #[test]
        fn algorithm_override_select_or_auto_with_override() {
            let ao = AlgorithmOverride { algorithm_name: "Tree".to_string() };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            let algo = ao.select_or_auto(
                &topo, gllm_nccl::CollectiveOp::AllReduce, 512 * 1024, 4, 0,
            );
            assert_eq!(algo, gllm_nccl::CollectiveAlgorithm::Tree);
        }

        #[test]
        fn algorithm_override_select_or_auto_without_override_falls_back() {
            let ao = AlgorithmOverride { algorithm_name: "UnknownAlgo".to_string() };
            let topo = gllm_nccl::topology::make_test_topology(4, true);
            let algo = ao.select_or_auto(
                &topo, gllm_nccl::CollectiveOp::AllReduce, 512 * 1024, 4, 0,
            );
            // 回退到 gllm-nccl select_algorithm 自动选择
            // NVLink full + 512KB → Ring
            assert_eq!(algo, gllm_nccl::CollectiveAlgorithm::Ring);
        }
    }
}
