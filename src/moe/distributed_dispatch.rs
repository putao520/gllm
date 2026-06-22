//! 分布式 MoE 调度决策 (REQ-DIST-014)
//!
//! 根据 MoeDistributedConfig 和 CommHandleWrapper 决定分布式 MoE 策略：
//! - Expert 放置策略（Auto / RoundRobin / HotCold / Custom）
//! - AllToAll 通信策略（Auto / Nvlink / Rdma / Hierarchical）
//! - 跨 GPU expert dispatch 执行逻辑

#[cfg(feature = "nccl")]
pub mod distributed_dispatch {
    use crate::engine::distributed_config::{
        AllToAllStrategy, CommHandleWrapper, ExpertPlacement, MoeDistributedConfig,
    };

    /// 分布式 MoE 调度决策 (REQ-DIST-014)
    ///
    /// 由 `MoeDistDecision::from_config()` 根据 `MoeDistributedConfig` 和
    /// `CommHandleWrapper` 推导得出，单机模式下退化为 Auto。
    #[derive(Debug, Clone, PartialEq)]
    pub struct MoeDistDecision {
        /// Expert 放置策略决策
        pub placement: MoePlacementDecision,
        /// AllToAll 通信策略决策
        pub all_to_all: AllToAllDecision,
        /// Expert 总数（由 from_config 根据 num_experts 设定）
        num_experts: usize,
        /// World size（由 from_config 根据 CommHandleWrapper 设定）
        world_size: u32,
    }

    /// Expert 放置策略决策
    ///
    /// 与 `ExpertPlacement` 枚举一一对应，但 `HotCold` 和 `Custom` 携带
    /// 运行时决策信息（热/冷 expert 计数、expert→rank 映射）。
    #[derive(Debug, Clone, PartialEq)]
    pub enum MoePlacementDecision {
        /// 自动：均匀分配 expert 到各 GPU
        Auto,
        /// 轮询：Expert i 放置在 GPU (i % num_ranks)
        RoundRobin,
        /// 冷热分离：热 Expert 镜像到多个 GPU，冷 Expert 独占
        HotCold {
            /// 热 expert 数量（镜像到多 GPU）
            hot_count: usize,
            /// 冷 expert 数量（独占单 GPU）
            cold_count: usize,
        },
        /// 自定义：用户指定的 expert → GPU 映射
        Custom {
            /// expert_id → rank 映射
            mapping: Vec<u32>,
        },
    }

    /// AllToAll 通信策略决策
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum AllToAllDecision {
        /// 自动：根据 GPU 拓扑选择
        Auto,
        /// NVLink: GPU 间直接通信（同一节点）
        Nvlink,
        /// RDMA: 跨节点 GPU Direct RDMA
        Rdma,
        /// 层次化: 节点内 NVLink + 节点间 RDMA
        Hierarchical,
    }

    /// Expert dispatch 执行结果 (REQ-DIST-014)
    ///
    /// 跨 GPU AllToAll dispatch 的输出，包含每个 rank 的 token 分组信息。
    #[derive(Debug, Clone, PartialEq)]
    pub struct ExpertDispatchResult {
        /// 每个 rank 的 token 发送数量
        pub send_counts: Vec<u32>,
        /// 每个 rank 的 token 索引列表
        pub send_indices: Vec<Vec<u32>>,
        /// 每个 rank 的 token 接收数量（AllToAll 交换后填充）
        pub recv_counts: Vec<u32>,
    }

    impl ExpertDispatchResult {
        /// 创建单机模式结果：所有 token 在本地处理
        pub fn local(_num_tokens: usize, world_size: u32) -> Self {
            Self {
                send_counts: vec![0; world_size as usize],
                send_indices: vec![vec![]; world_size as usize],
                recv_counts: vec![0; world_size as usize],
            }
        }
    }

    impl MoeDistDecision {
        /// 根据 MoeDistributedConfig、num_experts 和 CommHandleWrapper 决定分布式 MoE 策略
        ///
        /// 单机模式（`!comm_handle.is_distributed()`）时退化为 Auto。
        /// HotCold 模式：初始 hot_count=0, cold_count=num_experts（EPLB 运行时填充 hot_count）。
        /// Custom 模式：初始 RoundRobin 分布（external planner 可通过 update_custom_mapping 覆盖）。
        pub fn from_config(
            config: &MoeDistributedConfig,
            num_experts: usize,
            comm_handle: &CommHandleWrapper,
        ) -> Self {
            if !comm_handle.is_distributed() {
                return Self {
                    placement: MoePlacementDecision::Auto,
                    all_to_all: AllToAllDecision::Auto,
                    num_experts,
                    world_size: 1,
                };
            }

            let world_size = comm_handle.world_size();
            let placement = match config.expert_placement {
                ExpertPlacement::Auto => MoePlacementDecision::Auto,
                ExpertPlacement::RoundRobin => MoePlacementDecision::RoundRobin,
                ExpertPlacement::HotCold => MoePlacementDecision::HotCold {
                    // 初始全部视为冷 expert（EPLB 运行时通过 update_hot_cold_counts 填充）
                    hot_count: 0,
                    cold_count: num_experts,
                },
                ExpertPlacement::Custom => {
                    // 初始使用 RoundRobin 分布，external planner 通过 update_custom_mapping 覆盖
                    let mapping: Vec<u32> = (0..num_experts)
                        .map(|i| i as u32 % world_size)
                        .collect();
                    MoePlacementDecision::Custom { mapping }
                },
            };

            let all_to_all = match config.all_to_all {
                AllToAllStrategy::Auto => AllToAllDecision::Auto,
                AllToAllStrategy::NvlinkAllToAll => AllToAllDecision::Nvlink,
                AllToAllStrategy::RdmaAllToAll => AllToAllDecision::Rdma,
                AllToAllStrategy::HierarchicalAllToAll => AllToAllDecision::Hierarchical,
            };

            Self {
                placement,
                all_to_all,
                num_experts,
                world_size,
            }
        }

        /// 是否需要跨 GPU AllToAll 通信
        ///
        /// 仅 Auto（单机退化的默认策略）不需要跨 GPU 通信；
        /// 其他策略均意味着 expert 分布在多个 GPU 上。
        pub fn needs_cross_gpu_dispatch(&self) -> bool {
            !matches!(self.placement, MoePlacementDecision::Auto)
        }

        /// 用 EPLB 统计数据更新 HotCold 决策的 hot/cold 计数
        ///
        /// 仅当 placement 为 HotCold 时生效，其他策略静默忽略。
        pub fn update_hot_cold_counts(&mut self, hot_count: usize, cold_count: usize) {
            if let MoePlacementDecision::HotCold {
                hot_count: ref mut hc,
                cold_count: ref mut cc,
            } = self.placement
            {
                *hc = hot_count;
                *cc = cold_count;
            }
        }

        /// 用外部 planner 提供的映射更新 Custom 决策
        ///
        /// 仅当 placement 为 Custom 时生效，其他策略静默忽略。
        pub fn update_custom_mapping(&mut self, mapping: Vec<u32>) {
            if let MoePlacementDecision::Custom {
                mapping: ref mut m,
            } = self.placement
            {
                *m = mapping;
            }
        }

        /// 根据 placement 决策确定 expert 所在的 GPU rank
        ///
        /// 用于 dispatch_experts 中的 token 分组。
        pub fn expert_target_rank(&self, expert_id: u32) -> u32 {
            match &self.placement {
                MoePlacementDecision::Auto => 0, // 单机全部在本地
                MoePlacementDecision::RoundRobin => expert_id % self.world_size,
                MoePlacementDecision::Custom { mapping } => {
                    mapping.get(expert_id as usize).copied().unwrap_or(expert_id % self.world_size)
                }
                MoePlacementDecision::HotCold { hot_count, .. } => {
                    // 前 hot_count 个 expert 镜像到所有 GPU → 本地处理
                    // 后续 expert 按 RoundRobin 分配
                    if (expert_id as usize) < *hot_count {
                        0 // 热 expert 在所有 GPU 都有副本，本 rank 可本地处理
                    } else {
                        expert_id % self.world_size
                    }
                },
            }
        }

        /// 执行跨 GPU expert dispatch (AllToAll)
        ///
        /// 根据 placement 决策分组 token 到各 GPU，产出 ExpertDispatchResult。
        /// 单机模式：返回 local result。
        /// 分布式模式：返回分组结果，后续通过 CommHandleWrapper 执行 AllToAll。
        pub fn dispatch_experts(
            &self,
            token_indices: &[u32],
            expert_ids: &[u32],
            comm_handle: &CommHandleWrapper,
        ) -> Result<ExpertDispatchResult, String> {
            if !comm_handle.is_distributed() {
                return Ok(ExpertDispatchResult::local(token_indices.len(), 1));
            }

            let num_ranks = comm_handle.world_size() as usize;
            let mut send_counts = vec![0u32; num_ranks];
            let mut send_indices: Vec<Vec<u32>> = vec![vec![]; num_ranks];

            for (i, &expert_id) in expert_ids.iter().enumerate() {
                let target_rank = self.expert_target_rank(expert_id) as usize;
                if target_rank < num_ranks {
                    send_counts[target_rank] += 1;
                    send_indices[target_rank].push(token_indices[i]);
                }
            }

            // recv_counts 由 AllToAll 交换后填充，初始置 0
            let recv_counts = vec![0u32; num_ranks];

            Ok(ExpertDispatchResult {
                send_counts,
                send_indices,
                recv_counts,
            })
        }

        /// 执行 AllToAll 交换以获取 recv_counts
        ///
        /// 通过 AllGather 聚合所有 rank 的 send_counts，计算本 rank 的 recv_counts。
        pub fn all_to_all_exchange_counts(
            &self,
            result: &mut ExpertDispatchResult,
            comm_handle: &CommHandleWrapper,
        ) -> Result<(), String> {
            if !comm_handle.is_distributed() {
                return Ok(());
            }

            let send_counts_f32: Vec<f32> = result.send_counts.iter().map(|&c| c as f32).collect();
            let gathered = comm_handle.all_gather_f32(&send_counts_f32)?;

            // gathered layout: [rank0_counts, rank1_counts, ...]
            // 本 rank 的 recv_counts[peer] = gathered 中 peer rank 的 send_counts[self_rank_offset_in_peer_chunk]
            // 但 AllToAll 场景下：每个 rank 的 recv_counts[peer] = peer 的 send_counts[self_rank]
            let num_ranks = comm_handle.world_size() as usize;
            let my_rank = comm_handle.rank() as usize;

            for peer in 0..num_ranks {
                let offset = peer * num_ranks;
                result.recv_counts[peer] = gathered[offset + my_rank] as u32;
            }

            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::engine::distributed_config::ParallelConfig;

        fn single_node_handle() -> CommHandleWrapper {
            CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap()
        }

        fn multi_node_handle(rank: u32, world_size: u32) -> CommHandleWrapper {
            let cp_size: u32 = 1;
            CommHandleWrapper::from_config(&ParallelConfig {
                tp_size: world_size,
                pp_size: 1,
                ep_size: 1,
                cp_size,
                rank,
                world_size,
                unique_id: String::new(),
                stage_id: rank / (world_size * cp_size),
            })
            .unwrap()
        }

        // ── from_config: 单机退化为 Auto ──────────────────────────────────────

        #[test]
        fn single_node_defaults_to_auto() {
            let config = MoeDistributedConfig::default();
            let handle = single_node_handle();
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            assert_eq!(decision.placement, MoePlacementDecision::Auto);
            assert_eq!(decision.all_to_all, AllToAllDecision::Auto);
            assert_eq!(decision.num_experts, 8);
        }

        #[test]
        fn single_node_roundrobin_degrades_to_auto() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::RoundRobin,
                all_to_all: AllToAllStrategy::Auto,
            };
            let handle = single_node_handle();
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            // 单机模式下退化为 Auto
            assert_eq!(decision.placement, MoePlacementDecision::Auto);
        }

        // ── from_config: 多机模式 ──────────────────────────────────────────────

        #[test]
        fn multi_node_auto() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::Auto,
                all_to_all: AllToAllStrategy::Auto,
            };
            let handle = multi_node_handle(0, 4);
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            assert_eq!(decision.placement, MoePlacementDecision::Auto);
            assert_eq!(decision.all_to_all, AllToAllDecision::Auto);
        }

        #[test]
        fn multi_node_roundrobin() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::RoundRobin,
                all_to_all: AllToAllStrategy::Auto,
            };
            let handle = multi_node_handle(0, 4);
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            assert_eq!(decision.placement, MoePlacementDecision::RoundRobin);
        }

        #[test]
        fn multi_node_hot_cold_initial_zero_hot() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::HotCold,
                all_to_all: AllToAllStrategy::Auto,
            };
            let handle = multi_node_handle(0, 4);
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            // 初始: hot=0, cold=num_experts(8)
            assert_eq!(
                decision.placement,
                MoePlacementDecision::HotCold {
                    hot_count: 0,
                    cold_count: 8,
                }
            );
        }

        #[test]
        fn multi_node_custom_initial_roundrobin_mapping() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::Custom,
                all_to_all: AllToAllStrategy::Auto,
            };
            let handle = multi_node_handle(0, 4);
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            // 初始使用 RoundRobin 映射: [0, 1, 2, 3, 0, 1, 2, 3]
            assert_eq!(
                decision.placement,
                MoePlacementDecision::Custom {
                    mapping: vec![0, 1, 2, 3, 0, 1, 2, 3],
                }
            );
        }

        #[test]
        fn multi_node_custom_initial_roundrobin_mapping_uneven() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::Custom,
                all_to_all: AllToAllStrategy::Auto,
            };
            let handle = multi_node_handle(0, 4);
            // 7 experts, 4 GPUs → mapping: [0, 1, 2, 3, 0, 1, 2]
            let decision = MoeDistDecision::from_config(&config, 7, &handle);
            assert_eq!(
                decision.placement,
                MoePlacementDecision::Custom {
                    mapping: vec![0, 1, 2, 3, 0, 1, 2],
                }
            );
        }

        #[test]
        fn all_to_all_nvlink() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::Auto,
                all_to_all: AllToAllStrategy::NvlinkAllToAll,
            };
            let handle = multi_node_handle(0, 2);
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            assert_eq!(decision.all_to_all, AllToAllDecision::Nvlink);
        }

        #[test]
        fn all_to_all_rdma() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::Auto,
                all_to_all: AllToAllStrategy::RdmaAllToAll,
            };
            let handle = multi_node_handle(0, 2);
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            assert_eq!(decision.all_to_all, AllToAllDecision::Rdma);
        }

        #[test]
        fn all_to_all_hierarchical() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::Auto,
                all_to_all: AllToAllStrategy::HierarchicalAllToAll,
            };
            let handle = multi_node_handle(0, 4);
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            assert_eq!(decision.all_to_all, AllToAllDecision::Hierarchical);
        }

        // ── needs_cross_gpu_dispatch ────────────────────────────────────────────

        #[test]
        fn auto_needs_no_cross_gpu() {
            let decision = MoeDistDecision {
                placement: MoePlacementDecision::Auto,
                all_to_all: AllToAllDecision::Auto,
                num_experts: 8,
                world_size: 1,
            };
            assert!(!decision.needs_cross_gpu_dispatch());
        }

        #[test]
        fn roundrobin_needs_cross_gpu() {
            let decision = MoeDistDecision {
                placement: MoePlacementDecision::RoundRobin,
                all_to_all: AllToAllDecision::Auto,
                num_experts: 8,
                world_size: 4,
            };
            assert!(decision.needs_cross_gpu_dispatch());
        }

        #[test]
        fn hot_cold_needs_cross_gpu() {
            let decision = MoeDistDecision {
                placement: MoePlacementDecision::HotCold {
                    hot_count: 3,
                    cold_count: 5,
                },
                all_to_all: AllToAllDecision::Auto,
                num_experts: 8,
                world_size: 4,
            };
            assert!(decision.needs_cross_gpu_dispatch());
        }

        #[test]
        fn custom_needs_cross_gpu() {
            let decision = MoeDistDecision {
                placement: MoePlacementDecision::Custom {
                    mapping: vec![0, 1, 0, 1],
                },
                all_to_all: AllToAllDecision::Nvlink,
                num_experts: 4,
                world_size: 2,
            };
            assert!(decision.needs_cross_gpu_dispatch());
        }

        // ── update_hot_cold_counts ──────────────────────────────────────────────

        #[test]
        fn update_hot_cold_on_hot_cold_placement() {
            let mut decision = MoeDistDecision {
                placement: MoePlacementDecision::HotCold {
                    hot_count: 0,
                    cold_count: 8,
                },
                all_to_all: AllToAllDecision::Auto,
                num_experts: 8,
                world_size: 4,
            };
            decision.update_hot_cold_counts(4, 60);
            assert_eq!(
                decision.placement,
                MoePlacementDecision::HotCold {
                    hot_count: 4,
                    cold_count: 60,
                }
            );
        }

        #[test]
        fn update_hot_cold_on_auto_placement_is_noop() {
            let mut decision = MoeDistDecision {
                placement: MoePlacementDecision::Auto,
                all_to_all: AllToAllDecision::Auto,
                num_experts: 8,
                world_size: 1,
            };
            decision.update_hot_cold_counts(4, 60);
            assert_eq!(decision.placement, MoePlacementDecision::Auto);
        }

        // ── update_custom_mapping ───────────────────────────────────────────────

        #[test]
        fn update_custom_mapping_on_custom_placement() {
            let mut decision = MoeDistDecision {
                placement: MoePlacementDecision::Custom {
                    mapping: vec![0, 1, 2, 3],
                },
                all_to_all: AllToAllDecision::Auto,
                num_experts: 4,
                world_size: 4,
            };
            decision.update_custom_mapping(vec![0, 1, 2, 0, 1, 2]);
            assert_eq!(
                decision.placement,
                MoePlacementDecision::Custom {
                    mapping: vec![0, 1, 2, 0, 1, 2],
                }
            );
        }

        #[test]
        fn update_custom_mapping_on_roundrobin_is_noop() {
            let mut decision = MoeDistDecision {
                placement: MoePlacementDecision::RoundRobin,
                all_to_all: AllToAllDecision::Auto,
                num_experts: 8,
                world_size: 4,
            };
            decision.update_custom_mapping(vec![0, 1]);
            assert_eq!(decision.placement, MoePlacementDecision::RoundRobin);
        }

        // ── expert_target_rank ─────────────────────────────────────────────────

        #[test]
        fn expert_target_rank_roundrobin() {
            let decision = MoeDistDecision {
                placement: MoePlacementDecision::RoundRobin,
                all_to_all: AllToAllDecision::Auto,
                num_experts: 8,
                world_size: 4,
            };
            assert_eq!(decision.expert_target_rank(0), 0);
            assert_eq!(decision.expert_target_rank(1), 1);
            assert_eq!(decision.expert_target_rank(2), 2);
            assert_eq!(decision.expert_target_rank(3), 3);
            assert_eq!(decision.expert_target_rank(4), 0);
            assert_eq!(decision.expert_target_rank(7), 3);
        }

        #[test]
        fn expert_target_rank_custom_mapping() {
            let decision = MoeDistDecision {
                placement: MoePlacementDecision::Custom {
                    mapping: vec![2, 3, 0, 1],
                },
                all_to_all: AllToAllDecision::Auto,
                num_experts: 4,
                world_size: 4,
            };
            assert_eq!(decision.expert_target_rank(0), 2);
            assert_eq!(decision.expert_target_rank(1), 3);
            assert_eq!(decision.expert_target_rank(2), 0);
            assert_eq!(decision.expert_target_rank(3), 1);
        }

        #[test]
        fn expert_target_rank_hot_cold() {
            let decision = MoeDistDecision {
                placement: MoePlacementDecision::HotCold {
                    hot_count: 2,
                    cold_count: 6,
                },
                all_to_all: AllToAllDecision::Auto,
                num_experts: 8,
                world_size: 4,
            };
            // hot expert (id < hot_count) → local (rank 0)
            assert_eq!(decision.expert_target_rank(0), 0);
            assert_eq!(decision.expert_target_rank(1), 0);
            // cold expert → RoundRobin
            assert_eq!(decision.expert_target_rank(2), 2);
            assert_eq!(decision.expert_target_rank(3), 3);
            assert_eq!(decision.expert_target_rank(4), 0);
        }

        // ── dispatch_experts ────────────────────────────────────────────────────

        #[test]
        fn dispatch_experts_single_node() {
            let config = MoeDistributedConfig::default();
            let handle = single_node_handle();
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            let result = decision.dispatch_experts(&[0, 1, 2], &[0, 1, 2], &handle).unwrap();
            assert_eq!(result.send_counts.len(), 1);
            assert_eq!(result.send_counts[0], 0); // 单机模式不跨 GPU
        }

        #[test]
        fn dispatch_experts_multi_node_roundrobin() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::RoundRobin,
                all_to_all: AllToAllStrategy::Auto,
            };
            let handle = multi_node_handle(0, 4);
            let decision = MoeDistDecision::from_config(&config, 8, &handle);
            // token 0 → expert 0 (rank 0), token 1 → expert 3 (rank 3), token 2 → expert 7 (rank 3)
            let result = decision
                .dispatch_experts(&[0, 1, 2], &[0, 3, 7], &handle)
                .unwrap();
            assert_eq!(result.send_counts, vec![1, 0, 0, 2]);
            assert_eq!(result.send_indices[0], vec![0]);
            assert_eq!(result.send_indices[3], vec![1, 2]);
        }

        #[test]
        fn dispatch_experts_hot_cold() {
            let config = MoeDistributedConfig {
                expert_placement: ExpertPlacement::HotCold,
                all_to_all: AllToAllStrategy::Auto,
            };
            let handle = multi_node_handle(0, 4);
            let mut decision = MoeDistDecision::from_config(&config, 8, &handle);
            // 设置 hot_count=2
            decision.update_hot_cold_counts(2, 6);
            // expert 0 (hot) → local, expert 5 (cold) → rank 1
            let result = decision
                .dispatch_experts(&[0, 1], &[0, 5], &handle)
                .unwrap();
            assert_eq!(result.send_counts[0], 1); // hot expert 本地处理
            assert_eq!(result.send_counts[1], 1); // cold expert 发到 rank 1
        }

        // ── ExpertDispatchResult ───────────────────────────────────────────────

        #[test]
        fn expert_dispatch_result_local() {
            let result = ExpertDispatchResult::local(10, 1);
            assert_eq!(result.send_counts.len(), 1);
            assert_eq!(result.send_counts[0], 0);
        }

        // ── derive traits ───────────────────────────────────────────────────────

        #[test]
        fn decision_clone_independence() {
            let mut decision = MoeDistDecision {
                placement: MoePlacementDecision::HotCold {
                    hot_count: 3,
                    cold_count: 5,
                },
                all_to_all: AllToAllDecision::Nvlink,
                num_experts: 8,
                world_size: 4,
            };
            let cloned = decision.clone();
            decision.update_hot_cold_counts(10, 20);
            assert_eq!(
                cloned.placement,
                MoePlacementDecision::HotCold {
                    hot_count: 3,
                    cold_count: 5,
                }
            );
        }

        #[test]
        fn decision_equality() {
            let a = MoeDistDecision {
                placement: MoePlacementDecision::RoundRobin,
                all_to_all: AllToAllDecision::Rdma,
                num_experts: 8,
                world_size: 4,
            };
            let b = MoeDistDecision {
                placement: MoePlacementDecision::RoundRobin,
                all_to_all: AllToAllDecision::Rdma,
                num_experts: 8,
                world_size: 4,
            };
            assert_eq!(a, b);
        }
    }
}
