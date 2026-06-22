//! Context Parallelism / Ring Attention (REQ-DIST-016)
//!
//! Ring Attention 将长序列分片到多个 GPU，通过环形通信传递 KV 块，
//! 使每个 rank 都能计算完整 attention 而无需将全部 KV 存于本地。
//!
//! 执行流程 (cp_size 次迭代):
//! 1. 本 rank 用本地 Q + 本地 KV 计算局部 attention
//! 2. 发送本地 KV 到 next_rank, 从 prev_rank 接收远端 KV
//! 3. 本 rank 用本地 Q + 远端 KV 计算局部 attention
//! 4. 重复 cp_size 次后每个 rank 处理完所有 KV 块
//!
//! 通信执行通过 CommHandleWrapper 的 send_f32 / recv_f32 实现。

#[cfg(feature = "nccl")]
pub mod context_parallel {
    use crate::engine::distributed_config::CommHandleWrapper;

    /// Context Parallelism 配置 (REQ-DIST-016)
    #[derive(Debug, Clone, PartialEq)]
    pub struct CpConfig {
        /// CP 大小（参与 Ring Attention 的 GPU 数量）
        pub cp_size: u32,
        /// 当前 rank 在 CP 环中的位置
        pub cp_rank: u32,
    }

    impl CpConfig {
        /// 从 CommHandleWrapper 推导 CP 配置
        /// 默认 cp_size = world_size, cp_rank = rank
        pub fn from_comm_handle(comm_handle: &CommHandleWrapper) -> Self {
            Self {
                cp_size: comm_handle.world_size(),
                cp_rank: comm_handle.rank(),
            }
        }

        /// 是否启用 Context Parallelism
        pub fn is_enabled(&self) -> bool {
            self.cp_size > 1
        }

        /// Ring 通信中的下一个 rank
        pub fn next_rank(&self) -> u32 {
            (self.cp_rank + 1) % self.cp_size
        }

        /// Ring 通信中的上一个 rank
        pub fn prev_rank(&self) -> u32 {
            if self.cp_rank == 0 {
                self.cp_size - 1
            } else {
                self.cp_rank - 1
            }
        }

        /// 本 rank 负责的 seq_len 分片大小
        pub fn local_seq_len(&self, total_seq_len: usize) -> usize {
            let shard = total_seq_len / self.cp_size as usize;
            let remainder = total_seq_len % self.cp_size as usize;
            shard + if (self.cp_rank as usize) < remainder { 1 } else { 0 }
        }

        /// 本 rank 负责的 seq_len 分片起始位置
        pub fn local_seq_offset(&self, total_seq_len: usize) -> usize {
            let shard = total_seq_len / self.cp_size as usize;
            let remainder = total_seq_len % self.cp_size as usize;
            shard * self.cp_rank as usize + remainder.min(self.cp_rank as usize)
        }
    }

    /// Ring Attention 通信阶段 (REQ-DIST-016)
    ///
    /// Ring Attention 执行 cp_size 次迭代：
    /// 1. 每次: 本 rank 用本地 Q + 远端 KV 块计算局部 attention
    /// 2. KV 块通过 Ring 通信（send K+V 到 next_rank, recv K+V from prev_rank）
    /// 3. 迭代 cp_size 次后每个 rank 处理完所有 KV 块
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RingPhase {
        /// 阶段 1: 本地 Q + 本地 KV
        LocalCompute,
        /// 阶段 2: 发送本地 KV 到 next_rank
        SendKvBlock {
            /// 当前迭代步
            step: u32,
        },
        /// 阶段 3: 接收远端 KV 从 prev_rank
        RecvKvBlock {
            /// 当前迭代步
            step: u32,
        },
        /// 阶段 4: 本地 Q + 远端 KV 计算
        RemoteCompute {
            /// 当前迭代步
            step: u32,
        },
    }

    /// Ring Attention 执行计划 (REQ-DIST-016)
    pub struct RingAttentionPlan {
        pub config: CpConfig,
        pub total_steps: u32,
    }

    impl RingAttentionPlan {
        pub fn new(config: CpConfig) -> Self {
            Self {
                total_steps: config.cp_size,
                config,
            }
        }

        /// 生成当前 step 的通信阶段序列
        pub fn phases_for_step(&self, step: u32) -> Vec<RingPhase> {
            if step == 0 {
                vec![RingPhase::LocalCompute]
            } else {
                vec![
                    RingPhase::SendKvBlock { step },
                    RingPhase::RecvKvBlock { step },
                    RingPhase::RemoteCompute { step },
                ]
            }
        }

        /// 在 step 步中，KV 块的源 rank
        /// Ring 通信: 每步 KV 块沿环前进 1 位
        /// step s 时，本 rank 持有的 KV 块来自 rank (cp_rank - s) % cp_size
        pub fn kv_source_rank(&self, step: u32) -> u32 {
            if step == 0 {
                self.config.cp_rank
            } else {
                (self.config.cp_rank + self.config.cp_size - step) % self.config.cp_size
            }
        }

        /// 执行一步 Ring 通信: 发送本地 KV + 接收远端 KV
        ///
        /// - step 0: 本地计算，无通信，返回 local_kv 的拷贝
        /// - step > 0: 通过 CommHandleWrapper send/recv 完成环形 KV 块传递
        ///
        /// 返回 RingStepResult 包含本地 KV 和可选的远端 KV 数据。
        pub fn execute_ring_step(
            &self,
            step: u32,
            kv_block: &[f32],
            comm_handle: &CommHandleWrapper,
        ) -> Result<RingStepResult, String> {
            if step == 0 {
                // 第一步: 本地计算，无通信
                return Ok(RingStepResult {
                    local_kv: kv_block.to_vec(),
                    remote_kv: None,
                });
            }

            // 发送本地 KV 到 next_rank
            let next = self.config.next_rank();
            comm_handle.send_f32(next, kv_block)?;

            // 接收远端 KV 从 prev_rank
            let prev = self.config.prev_rank();
            let remote_kv = comm_handle.recv_f32(prev, kv_block.len())?;

            Ok(RingStepResult {
                local_kv: kv_block.to_vec(),
                remote_kv: Some(remote_kv),
            })
        }
    }

    /// Ring Attention 一步执行结果 (REQ-DIST-016)
    #[derive(Debug, Clone, PartialEq)]
    pub struct RingStepResult {
        /// 本 rank 的本地 KV 块数据
        pub local_kv: Vec<f32>,
        /// 远端 KV 块数据（step 0 时为 None）
        pub remote_kv: Option<Vec<f32>>,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn make_config(cp_size: u32, cp_rank: u32) -> CpConfig {
            CpConfig { cp_size, cp_rank }
        }

        // ── CpConfig 基础 ──────────────────────────────────────────────────

        #[test]
        fn cp_config_is_enabled_single() {
            let config = make_config(1, 0);
            assert!(!config.is_enabled());
        }

        #[test]
        fn cp_config_is_enabled_multi() {
            let config = make_config(4, 0);
            assert!(config.is_enabled());
        }

        #[test]
        fn cp_config_next_rank_wraps() {
            let config = make_config(4, 3);
            assert_eq!(config.next_rank(), 0);
        }

        #[test]
        fn cp_config_next_rank_normal() {
            let config = make_config(4, 1);
            assert_eq!(config.next_rank(), 2);
        }

        #[test]
        fn cp_config_prev_rank_wraps() {
            let config = make_config(4, 0);
            assert_eq!(config.prev_rank(), 3);
        }

        #[test]
        fn cp_config_prev_rank_normal() {
            let config = make_config(4, 2);
            assert_eq!(config.prev_rank(), 1);
        }

        // ── local_seq_len / local_seq_offset ───────────────────────────────

        #[test]
        fn local_seq_len_even_split() {
            let config = make_config(4, 0);
            assert_eq!(config.local_seq_len(100), 25);
        }

        #[test]
        fn local_seq_len_uneven_split_low_rank() {
            // 103 / 4 = 25 remainder 3; ranks 0,1,2 get 26, rank 3 gets 25
            let config = make_config(4, 0);
            assert_eq!(config.local_seq_len(103), 26);
        }

        #[test]
        fn local_seq_len_uneven_split_high_rank() {
            let config = make_config(4, 3);
            assert_eq!(config.local_seq_len(103), 25);
        }

        #[test]
        fn local_seq_offset_even() {
            let config = make_config(4, 2);
            assert_eq!(config.local_seq_offset(100), 50);
        }

        #[test]
        fn local_seq_offset_uneven() {
            // 103 / 4 = 25 r3; rank 2 offset = 25*2 + min(3,2) = 52
            let config = make_config(4, 2);
            assert_eq!(config.local_seq_offset(103), 52);
        }

        #[test]
        fn local_seq_offset_first_rank() {
            let config = make_config(4, 0);
            assert_eq!(config.local_seq_offset(103), 0);
        }

        #[test]
        fn local_seq_len_and_offset_sum_to_total() {
            let total = 103;
            let cp_size = 4u32;
            let mut sum = 0usize;
            for rank in 0..cp_size {
                let config = make_config(cp_size, rank);
                let offset = config.local_seq_offset(total);
                let len = config.local_seq_len(total);
                assert_eq!(offset, sum);
                sum += len;
            }
            assert_eq!(sum, total);
        }

        // ── RingAttentionPlan ──────────────────────────────────────────────

        #[test]
        fn plan_total_steps_equals_cp_size() {
            let config = make_config(4, 0);
            let plan = RingAttentionPlan::new(config);
            assert_eq!(plan.total_steps, 4);
        }

        #[test]
        fn plan_phases_step0_local_only() {
            let config = make_config(4, 0);
            let plan = RingAttentionPlan::new(config);
            let phases = plan.phases_for_step(0);
            assert_eq!(phases, vec![RingPhase::LocalCompute]);
        }

        #[test]
        fn plan_phases_step1_three_phases() {
            let config = make_config(4, 0);
            let plan = RingAttentionPlan::new(config);
            let phases = plan.phases_for_step(1);
            assert_eq!(phases.len(), 3);
            assert_eq!(phases[0], RingPhase::SendKvBlock { step: 1 });
            assert_eq!(phases[1], RingPhase::RecvKvBlock { step: 1 });
            assert_eq!(phases[2], RingPhase::RemoteCompute { step: 1 });
        }

        #[test]
        fn plan_kv_source_rank_step0_is_self() {
            let config = make_config(4, 2);
            let plan = RingAttentionPlan::new(config);
            assert_eq!(plan.kv_source_rank(0), 2);
        }

        #[test]
        fn plan_kv_source_rank_ring_rotation() {
            // rank 2, step 1: source = (2 + 4 - 1) % 4 = 1
            // rank 2, step 2: source = (2 + 4 - 2) % 4 = 0
            // rank 2, step 3: source = (2 + 4 - 3) % 4 = 3
            let config = make_config(4, 2);
            let plan = RingAttentionPlan::new(config);
            assert_eq!(plan.kv_source_rank(1), 1);
            assert_eq!(plan.kv_source_rank(2), 0);
            assert_eq!(plan.kv_source_rank(3), 3);
        }

        #[test]
        fn plan_kv_source_rank_covers_all_ranks() {
            let config = make_config(4, 2);
            let plan = RingAttentionPlan::new(config);
            let sources: std::collections::HashSet<u32> = (0..4)
                .map(|s| plan.kv_source_rank(s))
                .collect();
            assert_eq!(sources.len(), 4);
        }

        // ── execute_ring_step ──────────────────────────────────────────────

        #[test]
        fn execute_ring_step_step0_local_only() {
            let config = make_config(4, 0);
            let plan = RingAttentionPlan::new(config);
            let handle = CommHandleWrapper::new_for_test(0, 4);
            let kv_block = vec![1.0f32, 2.0, 3.0];
            let result = plan.execute_ring_step(0, &kv_block, &handle).unwrap();
            assert_eq!(result.local_kv, kv_block);
            assert!(result.remote_kv.is_none());
        }

        #[test]
        fn execute_ring_step_step1_without_nccl_returns_error() {
            let config = make_config(4, 0);
            let plan = RingAttentionPlan::new(config);
            let handle = CommHandleWrapper::new_for_test(0, 4);
            let kv_block = vec![1.0f32, 2.0, 3.0];
            // 多机模式但 NCCL 未初始化 → send_f32 返回错误
            let result = plan.execute_ring_step(1, &kv_block, &handle);
            assert!(result.is_err());
        }

        // ── RingStepResult ──────────────────────────────────────────────────

        #[test]
        fn ring_step_result_equality() {
            let a = RingStepResult {
                local_kv: vec![1.0, 2.0],
                remote_kv: None,
            };
            let b = RingStepResult {
                local_kv: vec![1.0, 2.0],
                remote_kv: None,
            };
            assert_eq!(a, b);
        }

        // ── from_comm_handle ───────────────────────────────────────────────

        #[test]
        fn from_comm_handle_derives_config() {
            let comm = CommHandleWrapper::new_for_test(2, 4);
            let config = CpConfig::from_comm_handle(&comm);
            assert_eq!(config.cp_size, 4);
            assert_eq!(config.cp_rank, 2);
        }

        // ── Ring 通信完整性: next/prev 构成环 ─────────────────────────────

        #[test]
        fn ring_next_prev_roundtrip() {
            for rank in 0..4u32 {
                let config = make_config(4, rank);
                // next_rank 的 prev_rank 应回到自己
                let next = config.next_rank();
                let next_config = make_config(4, next);
                assert_eq!(next_config.prev_rank(), rank);
            }
        }
    }
}
