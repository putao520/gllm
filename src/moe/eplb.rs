//! EPLB (Expert Parallel Load Balancing) 专家负载均衡 (REQ-DIST-015)
//!
//! 基于滑动窗口统计 expert 调用频次，检测负载不均衡，
//! 输出热/冷 expert 分类供分布式 MoE 调度决策使用。
//! 支持跨 GPU 统计聚合（通过 AllReduce），确保全局视角的负载均衡。

#[cfg(feature = "nccl")]
pub mod eplb {
    use crate::engine::distributed_config::CommHandleWrapper;
    use std::time::{Duration, Instant};

    /// Expert 负载统计 (REQ-DIST-015)
    ///
    /// 滑动窗口内统计每个 expert 的调用次数，用于 EPLB 重平衡决策。
    /// 默认窗口 60 秒，窗口过期后需重置以开始新一轮统计。
    /// 通过 `aggregate_from_all_ranks()` 可跨 GPU 聚合统计。
    #[derive(Debug, Clone)]
    pub struct ExpertLoadStats {
        /// 每个 expert 的调用次数
        pub invocation_counts: Vec<u64>,
        /// 统计窗口起始时间
        pub window_start: Instant,
        /// 统计窗口长度
        pub window_duration: Duration,
    }

    impl Default for ExpertLoadStats {
        fn default() -> Self {
            Self::new(0)
        }
    }

    impl ExpertLoadStats {
        /// 创建指定 expert 数量的统计实例
        ///
        /// 初始窗口长度 60 秒，所有计数归零。
        pub fn new(num_experts: usize) -> Self {
            Self {
                invocation_counts: vec![0; num_experts],
                window_start: Instant::now(),
                window_duration: Duration::from_secs(60),
            }
        }

        /// 创建自定义窗口长度的统计实例
        pub fn with_window(num_experts: usize, window_duration: Duration) -> Self {
            Self {
                invocation_counts: vec![0; num_experts],
                window_start: Instant::now(),
                window_duration,
            }
        }

        /// 记录一次 expert 调用
        ///
        /// 越界 expert_id 静默忽略（防御性编程）。
        pub fn record_invocation(&mut self, expert_id: usize) {
            if expert_id < self.invocation_counts.len() {
                self.invocation_counts[expert_id] += 1;
            }
        }

        /// 检查统计窗口是否已过期
        pub fn is_window_expired(&self) -> bool {
            self.window_start.elapsed() >= self.window_duration
        }

        /// 重置统计窗口（开始新一轮统计）
        ///
        /// 所有计数归零，窗口起始时间重置为当前时刻。
        pub fn reset_window(&mut self) {
            for count in &mut self.invocation_counts {
                *count = 0;
            }
            self.window_start = Instant::now();
        }

        /// 获取热度排序（调用次数降序）
        ///
        /// 返回 top_k 个最热 expert 的 ID 列表。
        pub fn hot_experts(&self, top_k: usize) -> Vec<usize> {
            let mut indexed: Vec<(u64, usize)> = self
                .invocation_counts
                .iter()
                .enumerate()
                .map(|(i, &c)| (c, i))
                .collect();
            indexed.sort_by(|a, b| b.0.cmp(&a.0));
            indexed.into_iter().take(top_k).map(|(_, i)| i).collect()
        }

        /// 获取冷度排序（调用次数升序）
        ///
        /// 返回 top_k 个最冷 expert 的 ID 列表。
        pub fn cold_experts(&self, top_k: usize) -> Vec<usize> {
            let mut indexed: Vec<(u64, usize)> = self
                .invocation_counts
                .iter()
                .enumerate()
                .map(|(i, &c)| (c, i))
                .collect();
            indexed.sort_by(|a, b| a.0.cmp(&b.0));
            indexed.into_iter().take(top_k).map(|(_, i)| i).collect()
        }

        /// 获取总调用次数
        pub fn total_invocations(&self) -> u64 {
            self.invocation_counts.iter().sum()
        }

        /// 获取平均调用次数
        ///
        /// 无 expert 时返回 0.0。
        pub fn avg_invocation(&self) -> f64 {
            if self.invocation_counts.is_empty() {
                return 0.0;
            }
            self.total_invocations() as f64 / self.invocation_counts.len() as f64
        }

        /// 获取最大调用次数
        pub fn max_invocation(&self) -> u64 {
            self.invocation_counts.iter().copied().max().unwrap_or(0)
        }

        /// 获取最小调用次数
        pub fn min_invocation(&self) -> u64 {
            self.invocation_counts.iter().copied().min().unwrap_or(0)
        }

        /// 计算负载不均衡比率 (max / avg)
        ///
        /// avg 为 0 时返回 0.0（无负载 = 无不均衡）。
        pub fn imbalance_ratio(&self) -> f64 {
            let avg = self.avg_invocation();
            if avg == 0.0 {
                return 0.0;
            }
            self.max_invocation() as f64 / avg
        }

        /// 从所有 GPU 聚合 expert 调用统计 (REQ-DIST-015)
        ///
        /// 通过 AllReduce Sum 聚合所有 GPU 的 invocation_counts，
        /// 使每个 rank 获得全局视角的负载统计。
        /// 单机模式下 no-op（数据已为全局）。
        pub fn aggregate_from_all_ranks(
            &mut self,
            comm_handle: &CommHandleWrapper,
        ) -> Result<(), String> {
            if !comm_handle.is_distributed() {
                return Ok(()); // 单机无需聚合
            }

            // 通过 AllReduce 聚合所有 GPU 的 invocation_counts
            comm_handle.all_reduce_u64_sum(&mut self.invocation_counts)?;

            Ok(())
        }
    }

    /// EPLB 重平衡决策 (REQ-DIST-015)
    ///
    /// 由 `should_rebalance()` 根据统计数据和阈值产出。
    #[derive(Debug, Clone, PartialEq)]
    pub struct EplbDecision {
        /// 是否需要重平衡
        pub needs_rebalance: bool,
        /// 热 expert ID 列表（应镜像到多 GPU）
        pub hot_expert_ids: Vec<usize>,
        /// 冷 expert ID 列表（可独占单 GPU）
        pub cold_expert_ids: Vec<usize>,
    }

    /// 根据 ExpertLoadStats 决定是否需要重平衡 (REQ-DIST-015)
    ///
    /// 判定逻辑：
    /// 1. 单机模式或无统计数据 → 不需要重平衡
    /// 2. 总调用次数为 0 → 不需要重平衡
    /// 3. 不均衡比率 (max/avg) > threshold → 需要重平衡
    ///    - 热 expert: 调用次数 > avg * 2.0
    ///    - 冷 expert: 调用次数 < avg * 0.5
    ///
    /// `imbalance_threshold` 典型值 2.0（最热 expert 调用次数是平均的 2 倍以上）。
    pub fn should_rebalance(
        stats: &ExpertLoadStats,
        comm_handle: &CommHandleWrapper,
        imbalance_threshold: f64,
    ) -> EplbDecision {
        if !comm_handle.is_distributed() || stats.invocation_counts.is_empty() {
            return EplbDecision {
                needs_rebalance: false,
                hot_expert_ids: vec![],
                cold_expert_ids: vec![],
            };
        }

        let total: u64 = stats.invocation_counts.iter().sum();
        if total == 0 {
            return EplbDecision {
                needs_rebalance: false,
                hot_expert_ids: vec![],
                cold_expert_ids: vec![],
            };
        }

        let avg = total as f64 / stats.invocation_counts.len() as f64;
        let max_count = stats.max_invocation() as f64;
        let imbalance_ratio = max_count / avg;

        if imbalance_ratio <= imbalance_threshold {
            return EplbDecision {
                needs_rebalance: false,
                hot_expert_ids: vec![],
                cold_expert_ids: vec![],
            };
        }

        // 热专家：调用次数 > 平均值 2 倍
        let hot: Vec<usize> = stats
            .invocation_counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c as f64 > avg * 2.0)
            .map(|(i, _)| i)
            .collect();

        // 冷专家：调用次数 < 平均值 0.5 倍
        let cold: Vec<usize> = stats
            .invocation_counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| (c as f64) < avg * 0.5)
            .map(|(i, _)| i)
            .collect();

        EplbDecision {
            needs_rebalance: true,
            hot_expert_ids: hot,
            cold_expert_ids: cold,
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
            CommHandleWrapper::from_config(&ParallelConfig {
                tp_size: world_size,
                pp_size: 1,
                ep_size: 1,
                rank,
                world_size,
                unique_id: String::new(),
            })
            .unwrap()
        }

        // ── ExpertLoadStats 基础 ───────────────────────────────────────────────

        #[test]
        fn new_initializes_zero_counts() {
            let stats = ExpertLoadStats::new(8);
            assert_eq!(stats.invocation_counts.len(), 8);
            assert!(stats.invocation_counts.iter().all(|&c| c == 0));
        }

        #[test]
        fn default_is_empty() {
            let stats = ExpertLoadStats::default();
            assert!(stats.invocation_counts.is_empty());
        }

        #[test]
        fn with_window_custom_duration() {
            let stats = ExpertLoadStats::with_window(4, Duration::from_secs(30));
            assert_eq!(stats.invocation_counts.len(), 4);
            assert_eq!(stats.window_duration, Duration::from_secs(30));
        }

        #[test]
        fn record_invocation_increments() {
            let mut stats = ExpertLoadStats::new(4);
            stats.record_invocation(0);
            stats.record_invocation(0);
            stats.record_invocation(2);
            assert_eq!(stats.invocation_counts[0], 2);
            assert_eq!(stats.invocation_counts[1], 0);
            assert_eq!(stats.invocation_counts[2], 1);
            assert_eq!(stats.invocation_counts[3], 0);
        }

        #[test]
        fn record_invocation_out_of_bounds_ignored() {
            let mut stats = ExpertLoadStats::new(4);
            stats.record_invocation(99);
            stats.record_invocation(4);
            assert!(stats.invocation_counts.iter().all(|&c| c == 0));
        }

        #[test]
        fn total_invocations() {
            let mut stats = ExpertLoadStats::new(4);
            stats.record_invocation(0);
            stats.record_invocation(1);
            stats.record_invocation(2);
            stats.record_invocation(2);
            assert_eq!(stats.total_invocations(), 4);
        }

        #[test]
        fn avg_invocation() {
            let mut stats = ExpertLoadStats::new(4);
            stats.record_invocation(0);
            stats.record_invocation(0);
            stats.record_invocation(1);
            assert_eq!(stats.avg_invocation(), 0.75);
        }

        #[test]
        fn avg_invocation_empty() {
            let stats = ExpertLoadStats::new(0);
            assert_eq!(stats.avg_invocation(), 0.0);
        }

        #[test]
        fn max_min_invocation() {
            let mut stats = ExpertLoadStats::new(4);
            stats.record_invocation(0);
            stats.record_invocation(0);
            stats.record_invocation(0);
            // counts: [3, 0, 0, 0]
            assert_eq!(stats.max_invocation(), 3);
            assert_eq!(stats.min_invocation(), 0);
        }

        #[test]
        fn imbalance_ratio_balanced() {
            let mut stats = ExpertLoadStats::new(4);
            // 每个专家调用 10 次 → ratio = 1.0
            for i in 0..4 {
                for _ in 0..10 {
                    stats.record_invocation(i);
                }
            }
            assert!((stats.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn imbalance_ratio_zero_load() {
            let stats = ExpertLoadStats::new(4);
            assert_eq!(stats.imbalance_ratio(), 0.0);
        }

        #[test]
        fn hot_experts_top_k() {
            let mut stats = ExpertLoadStats::new(8);
            // expert 3 最热，expert 7 次热
            for _ in 0..100 {
                stats.record_invocation(3);
            }
            for _ in 0..50 {
                stats.record_invocation(7);
            }
            for _ in 0..10 {
                stats.record_invocation(0);
            }
            let hot = stats.hot_experts(2);
            assert_eq!(hot, vec![3, 7]);
        }

        #[test]
        fn cold_experts_top_k() {
            let mut stats = ExpertLoadStats::new(8);
            for _ in 0..100 {
                stats.record_invocation(3);
            }
            for _ in 0..10 {
                stats.record_invocation(0);
            }
            // expert 1,2,4,5,6,7 从未被调用 → 最冷
            let cold = stats.cold_experts(3);
            // 所有零计数 expert 并列，排序稳定但具体顺序取决于 sort 稳定性
            assert!(cold.iter().all(|&id| id != 3 && id != 0));
        }

        #[test]
        fn reset_window_clears_counts() {
            let mut stats = ExpertLoadStats::new(4);
            stats.record_invocation(0);
            stats.record_invocation(1);
            assert_eq!(stats.total_invocations(), 2);
            stats.reset_window();
            assert_eq!(stats.total_invocations(), 0);
            assert!(stats.invocation_counts.iter().all(|&c| c == 0));
        }

        // ── aggregate_from_all_ranks ─────────────────────────────────────────

        #[test]
        fn aggregate_single_node_is_noop() {
            let mut stats = ExpertLoadStats::new(4);
            stats.record_invocation(0);
            stats.record_invocation(1);
            let handle = single_node_handle();
            // 单机模式：all_reduce_u64_sum 是 no-op，数据不变
            let result = stats.aggregate_from_all_ranks(&handle);
            assert!(result.is_ok());
            assert_eq!(stats.invocation_counts[0], 1);
            assert_eq!(stats.invocation_counts[1], 1);
        }

        #[test]
        fn aggregate_multi_node_without_nccl_returns_error() {
            let mut stats = ExpertLoadStats::new(4);
            stats.record_invocation(0);
            let handle = multi_node_handle(0, 4);
            // 多机模式但未初始化 NCCL → all_reduce_u64_sum 返回错误
            let result = stats.aggregate_from_all_ranks(&handle);
            assert!(result.is_err());
        }

        // ── should_rebalance ────────────────────────────────────────────────────

        #[test]
        fn single_node_never_rebalance() {
            let mut stats = ExpertLoadStats::new(8);
            for _ in 0..1000 {
                stats.record_invocation(0);
            }
            let handle = single_node_handle();
            let decision = should_rebalance(&stats, &handle, 2.0);
            assert!(!decision.needs_rebalance);
        }

        #[test]
        fn empty_stats_no_rebalance() {
            let stats = ExpertLoadStats::new(0);
            let handle = multi_node_handle(0, 4);
            let decision = should_rebalance(&stats, &handle, 2.0);
            assert!(!decision.needs_rebalance);
        }

        #[test]
        fn zero_invocations_no_rebalance() {
            let stats = ExpertLoadStats::new(8);
            let handle = multi_node_handle(0, 4);
            let decision = should_rebalance(&stats, &handle, 2.0);
            assert!(!decision.needs_rebalance);
        }

        #[test]
        fn balanced_load_no_rebalance() {
            let mut stats = ExpertLoadStats::new(8);
            // 每个专家 10 次 → ratio = 1.0 < 2.0
            for i in 0..8 {
                for _ in 0..10 {
                    stats.record_invocation(i);
                }
            }
            let handle = multi_node_handle(0, 4);
            let decision = should_rebalance(&stats, &handle, 2.0);
            assert!(!decision.needs_rebalance);
        }

        #[test]
        fn imbalanced_load_triggers_rebalance() {
            let mut stats = ExpertLoadStats::new(8);
            // expert 0: 100 次（远超平均），其余 7 个各 1 次
            // total = 107, avg = 13.375, max/avg = 100/13.375 ≈ 7.48 > 2.0
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

        #[test]
        fn hot_cold_classification() {
            let mut stats = ExpertLoadStats::new(8);
            // expert 0: 100 次 (hot, > avg*2)
            // expert 1-6: 10 次 (neutral)
            // expert 7: 1 次 (cold, < avg*0.5)
            // total = 161, avg = 20.125
            // hot threshold: > 40.25 → expert 0
            // cold threshold: < 10.0625 → expert 7
            for _ in 0..100 {
                stats.record_invocation(0);
            }
            for i in 1..7 {
                for _ in 0..10 {
                    stats.record_invocation(i);
                }
            }
            stats.record_invocation(7);
            let handle = multi_node_handle(0, 4);
            let decision = should_rebalance(&stats, &handle, 2.0);
            assert!(decision.needs_rebalance);
            assert!(decision.hot_expert_ids.contains(&0));
            assert!(decision.cold_expert_ids.contains(&7));
            // neutral experts not in either list
            for i in 1..7 {
                assert!(!decision.hot_expert_ids.contains(&i));
                assert!(!decision.cold_expert_ids.contains(&i));
            }
        }

        #[test]
        fn threshold_boundary_no_rebalance() {
            let mut stats = ExpertLoadStats::new(4);
            // expert 0: 8 次, expert 1-3: 各 4 次
            // total = 20, avg = 5.0, max/avg = 8/5 = 1.6 < 2.0
            for _ in 0..8 {
                stats.record_invocation(0);
            }
            for i in 1..4 {
                for _ in 0..4 {
                    stats.record_invocation(i);
                }
            }
            let handle = multi_node_handle(0, 4);
            let decision = should_rebalance(&stats, &handle, 2.0);
            assert!(!decision.needs_rebalance);
        }

        #[test]
        fn threshold_boundary_triggers_rebalance() {
            let mut stats = ExpertLoadStats::new(4);
            // expert 0: 11 次, expert 1-3: 各 3 次
            // total = 20, avg = 5.0, max/avg = 11/5 = 2.2 > 2.0
            for _ in 0..11 {
                stats.record_invocation(0);
            }
            for i in 1..4 {
                for _ in 0..3 {
                    stats.record_invocation(i);
                }
            }
            let handle = multi_node_handle(0, 4);
            let decision = should_rebalance(&stats, &handle, 2.0);
            assert!(decision.needs_rebalance);
        }

        #[test]
        fn custom_threshold_higher() {
            let mut stats = ExpertLoadStats::new(4);
            // expert 0: 11 次, expert 1-3: 各 3 次
            // max/avg = 2.2, threshold = 3.0 → no rebalance
            for _ in 0..11 {
                stats.record_invocation(0);
            }
            for i in 1..4 {
                for _ in 0..3 {
                    stats.record_invocation(i);
                }
            }
            let handle = multi_node_handle(0, 4);
            let decision = should_rebalance(&stats, &handle, 3.0);
            assert!(!decision.needs_rebalance);
        }

        // ── EplbDecision derive traits ──────────────────────────────────────────

        #[test]
        fn eplb_decision_equality() {
            let a = EplbDecision {
                needs_rebalance: true,
                hot_expert_ids: vec![0, 3],
                cold_expert_ids: vec![5, 7],
            };
            let b = EplbDecision {
                needs_rebalance: true,
                hot_expert_ids: vec![0, 3],
                cold_expert_ids: vec![5, 7],
            };
            assert_eq!(a, b);
        }

        #[test]
        fn eplb_decision_clone_independence() {
            let a = EplbDecision {
                needs_rebalance: true,
                hot_expert_ids: vec![0, 3],
                cold_expert_ids: vec![5, 7],
            };
            let mut b = a.clone();
            b.hot_expert_ids.push(1);
            assert_eq!(a.hot_expert_ids.len(), 2);
            assert_eq!(b.hot_expert_ids.len(), 3);
        }
    }
}
