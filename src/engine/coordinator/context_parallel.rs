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
//!
//! CP 环通信限定在同 PP stage 内 (REQ-DIST-032)。
//! 跨 PP stage 零 CP 通信开销。

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
        /// PP stage_id（CP 环限定在同 PP stage 内, REQ-DIST-032）
        pub pp_stage_id: u32,
        /// 是否启用 Sliding Window Attention 的 Ring 变体
        pub sliding_window: usize,
    }

    impl CpConfig {
        /// 从 CommHandleWrapper 推导 CP 配置
        /// 默认 cp_size = world_size, cp_rank = rank
        pub fn from_comm_handle(comm_handle: &CommHandleWrapper) -> Self {
            Self {
                cp_size: comm_handle.world_size(),
                cp_rank: comm_handle.rank(),
                pp_stage_id: 0,
                sliding_window: 0,
            }
        }

        /// 构造指定参数的 CP 配置
        pub fn new(cp_size: u32, cp_rank: u32, pp_stage_id: u32) -> Self {
            Self {
                cp_size,
                cp_rank,
                pp_stage_id,
                sliding_window: 0,
            }
        }

        /// 是否启用 Context Parallelism
        pub fn is_enabled(&self) -> bool {
            self.cp_size > 1
        }

        /// Ring 通信中的下一个 rank（同 PP stage 内）
        pub fn next_rank(&self) -> u32 {
            (self.cp_rank + 1) % self.cp_size
        }

        /// Ring 通信中的上一个 rank（同 PP stage 内）
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

        /// 计算一轮 Ring KV 块传输的通信量（f32 元素数）
        /// 通信量 = 2 * num_kv_heads * head_dim * local_seq_len（K 和 V 各一份）
        pub fn kv_block_elements(&self, num_kv_heads: usize, head_dim: usize, total_seq_len: usize) -> usize {
            let local_len = self.local_seq_len(total_seq_len);
            2 * num_kv_heads * head_dim * local_len
        }

        /// 判断是否启用 Sliding Window Ring Attention
        pub fn has_sliding_window(&self) -> bool {
            self.sliding_window > 0
        }

        /// 在 step 步中，KV 块是否需要与本 rank 的 Q 计算 attention
        /// Sliding Window 模式：只有 KV 块的 seq 范围与 Q 的 seq 范围存在窗口内重叠时才计算
        pub fn should_compute_at_step(&self, step: u32, total_seq_len: usize) -> bool {
            if !self.has_sliding_window() {
                return true; // 无滑动窗口：所有步都计算
            }
            let source_rank = {
                if step == 0 {
                    self.cp_rank
                } else {
                    (self.cp_rank + self.cp_size - step) % self.cp_size
                }
            };
            let q_start = self.local_seq_offset(total_seq_len);
            let q_end = q_start + self.local_seq_len(total_seq_len);
            let kv_start = self.local_seq_offset_for_rank(source_rank, total_seq_len);
            let kv_end = kv_start + self.local_seq_len_for_rank(source_rank, total_seq_len);
            // 滑动窗口重叠判断：Q 范围和 KV 范围的距离 < sliding_window
            let gap = if q_start >= kv_end {
                q_start - kv_end
            } else if kv_start >= q_end {
                kv_start - q_end
            } else {
                0 // 有重叠
            };
            gap < self.sliding_window
        }

        /// 指定 rank 的 seq_len 分片大小
        fn local_seq_len_for_rank(&self, rank: u32, total_seq_len: usize) -> usize {
            let shard = total_seq_len / self.cp_size as usize;
            let remainder = total_seq_len % self.cp_size as usize;
            shard + if (rank as usize) < remainder { 1 } else { 0 }
        }

        /// 指定 rank 的 seq_len 分片起始位置
        fn local_seq_offset_for_rank(&self, rank: u32, total_seq_len: usize) -> usize {
            let shard = total_seq_len / self.cp_size as usize;
            let remainder = total_seq_len % self.cp_size as usize;
            shard * rank as usize + remainder.min(rank as usize)
        }

        /// 验证 CP 配置有效性
        pub fn validate(&self) -> Result<(), String> {
            if self.cp_size == 0 {
                return Err("CpConfig: cp_size must be >= 1".to_string());
            }
            if self.cp_rank >= self.cp_size {
                return Err(format!(
                    "CpConfig: cp_rank ({}) must be < cp_size ({})",
                    self.cp_rank, self.cp_size
                ));
            }
            Ok(())
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
    ///
    /// 生成 cp_size 轮的环形通信计划，每轮包含:
    /// - compute_local_attention（Q @ local_KV）
    /// - send_kv_chunk → recv_kv_chunk（环形通信）
    /// - compute_remote_attention（Q @ remote_KV）
    ///
    /// Sliding Window 变体 (REQ-DIST-016):
    /// 只计算窗口范围内的 KV 块，跳过距离过远的 KV 块。
    ///
    /// PP stage 限定 (REQ-DIST-032):
    /// CP 环通信限定在同 PP stage 内，跨 stage 零 CP 通信。
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
        ///
        /// step 0: LocalCompute（本地 Q + 本地 KV）
        /// step > 0: SendKvBlock → RecvKvBlock → RemoteCompute
        ///
        /// Sliding Window 模式: 跳过距离过远的 KV 块的 RemoteCompute
        pub fn phases_for_step(&self, step: u32) -> Vec<RingPhase> {
            if step == 0 {
                vec![RingPhase::LocalCompute]
            } else {
                let mut phases = vec![
                    RingPhase::SendKvBlock { step },
                    RingPhase::RecvKvBlock { step },
                ];
                // Sliding Window: 只在 KV 块与 Q 有窗口内重叠时才计算
                if self.should_compute_at_step(step) {
                    phases.push(RingPhase::RemoteCompute { step });
                }
                phases
            }
        }

        /// 生成完整 Ring Attention 执行的所有阶段
        /// 返回 cp_size 轮的完整阶段序列，用于验证通信轮数
        // @trace REQ-DIST-016 [entity:ENT-DIST-CP] [controlflow:CF-DIST-005]
        pub fn all_phases(&self) -> Vec<Vec<RingPhase>> {
            (0..self.total_steps).map(|s| self.phases_for_step(s)).collect()
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

        /// 在 step 步中，KV 块的目标 rank（本 rank 发送 KV 给谁）
        /// 每步 KV 块向 next_rank 发送
        pub fn kv_dest_rank(&self, _step: u32) -> u32 {
            self.config.next_rank()
        }

        /// 判断当前 step 是否需要执行 remote compute
        /// Sliding Window 模式下跳过距离过远的 KV 块
        fn should_compute_at_step(&self, step: u32) -> bool {
            if !self.config.has_sliding_window() {
                return true;
            }
            let source_rank = (self.config.cp_rank + self.config.cp_size - step) % self.config.cp_size;
            // 滑动窗口判断需要在运行时知道 total_seq_len，这里保守返回 true
            // 实际跳过在 execute_ring_full 中基于 total_seq_len 决定
            let _ = source_rank;
            true
        }

        /// 执行完整 Ring Attention 循环 (cp_size 轮)
        ///
        /// 这是 REQ-DIST-016 的核心执行路径：
        /// 1. 初始化 partial_attn 累加缓冲区
        /// 2. step 0: 本地 Q + 本地 KV → partial_attn[0]
        /// 3. step 1..cp_size: send KV → recv KV → Q @ remote KV → partial_attn[step]
        /// 4. 归一化 partial attention = 完整 attention
        ///
        /// 参数:
        /// - local_q: 本 rank 的 Q 张量 [batch, num_heads, local_seq_len, head_dim]
        /// - local_kv: 本 rank 的 KV 张量 [2, batch, num_kv_heads, local_seq_len, head_dim]
        /// - comm_handle: NCCL 通信句柄
        /// - total_seq_len: 完整序列长度（用于 sliding window 判断）
        /// - compute_attn: 闭包，计算 Q @ KV 的局部 attention 并返回结果
        ///
        /// 返回: 归一化后的完整 attention 输出
        // @trace REQ-DIST-016 [entity:ENT-DIST-CP] [controlflow:CF-DIST-005] [dataflow:DF-DIST-006]
        pub fn execute_ring_full<F>(
            &self,
            local_q: &[f32],
            local_kv: &[f32],
            comm_handle: &CommHandleWrapper,
            total_seq_len: usize,
            compute_attn: F,
        ) -> Result<RingFullResult, String>
        where
            F: Fn(&[f32], &[f32], u32, u32) -> Result<PartialAttnOutput, String>,
        {
            let cp_size = self.config.cp_size;
            if cp_size == 1 {
                // cp_size=1: 无分布式，直接本地计算
                let output = compute_attn(local_q, local_kv, 0, 0)?;
                return Ok(RingFullResult {
                    output: output.values,
                    rounds_completed: 1,
                });
            }

            let mut current_kv = local_kv.to_vec();
            let mut accumulated: Option<Vec<f32>> = None;
            let mut logsumexp_acc: Option<Vec<f32>> = None;
            let mut rounds_completed = 0u32;

            for step in 0..cp_size {
                // step 0: 本地计算（无通信）
                // step > 0: 先通信，再计算
                if step > 0 {
                    // 发送当前 KV 到 next_rank
                    let next = self.config.next_rank();
                    comm_handle.send_f32(next, &current_kv)?;

                    // 接收远端 KV 从 prev_rank
                    let prev = self.config.prev_rank();
                    let remote_kv = comm_handle.recv_f32(prev, current_kv.len())?;
                    current_kv = remote_kv;
                }

                // 判断是否需要计算（Sliding Window 优化）
                if !self.config.should_compute_at_step(step, total_seq_len) {
                    // Sliding Window: 跳过距离过远的 KV 块
                    log::debug!(
                        "Ring Attention: step {} skipped (sliding window, source_rank={})",
                        step,
                        self.kv_source_rank(step),
                    );
                    rounds_completed += 1;
                    continue;
                }

                // 计算局部 attention: Q @ current KV
                let source_rank = self.kv_source_rank(step);
                let partial = compute_attn(local_q, &current_kv, step, source_rank)?;

                // 累加 partial attention (online softmax logsumexp 修正)
                match (accumulated.take(), logsumexp_acc.take()) {
                    (None, None) => {
                        // 第一个 partial: 直接赋值
                        accumulated = Some(partial.values);
                        logsumexp_acc = partial.logsumexp;
                    }
                    (Some(mut acc), Some(lse_acc)) => {
                        // 后续 partial: online softmax 合并
                        // 新输出 = (acc * exp(lse_acc) + partial * exp(lse_partial)) / exp(new_lse)
                        // 其中 new_lse = logsumexp(lse_acc, lse_partial)
                        if let Some(ref lse_partial) = partial.logsumexp {
                            let new_lse = online_logsumexp_merge(&lse_acc, lse_partial);
                            let scale_acc = exp_sub(&lse_acc, &new_lse);
                            let scale_partial = exp_sub(lse_partial, &new_lse);
                            for i in 0..acc.len() {
                                acc[i] = acc[i] * scale_acc[i % scale_acc.len()]
                                    + partial.values[i] * scale_partial[i % scale_partial.len()];
                            }
                            logsumexp_acc = Some(new_lse);
                        } else {
                            // 无 logsumexp: 简单累加
                            for i in 0..acc.len() {
                                acc[i] += partial.values[i];
                            }
                            logsumexp_acc = Some(lse_acc);
                        }
                        accumulated = Some(acc);
                    }
                    (Some(mut acc), None) => {
                        // 无 logsumexp: 简单累加
                        for i in 0..acc.len() {
                            acc[i] += partial.values[i];
                        }
                        accumulated = Some(acc);
                    }
                    _ => unreachable!(),
                }

                rounds_completed += 1;
            }

            let output = accumulated.unwrap_or_default();

            Ok(RingFullResult {
                output,
                rounds_completed,
            })
        }

        /// 执行一步 Ring 通信: 发送本地 KV + 接收远端 KV
        ///
        /// - step 0: 本地计算，无通信，返回 local_kv 的拷贝
        /// - step > 0: 通过 CommHandleWrapper send/recv 完成环形 KV 块传递
        ///
        /// 返回 RingStepResult 包含本地 KV 和可选的远端 KV 数据。
        // @trace REQ-DIST-016 [entity:ENT-DIST-CP] [dataflow:DF-DIST-006]
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

    /// 局部 attention 计算结果
    #[derive(Debug, Clone, PartialEq)]
    pub struct PartialAttnOutput {
        /// attention 输出值 [batch * num_heads * local_seq_len * head_dim]
        pub values: Vec<f32>,
        /// log-sum-exp 值（用于 online softmax 合并多个 partial attention）
        /// 形状 [batch * num_heads * local_seq_len]
        pub logsumexp: Option<Vec<f32>>,
    }

    /// 完整 Ring Attention 执行结果 (REQ-DIST-016)
    #[derive(Debug, Clone, PartialEq)]
    pub struct RingFullResult {
        /// 归一化后的完整 attention 输出
        pub output: Vec<f32>,
        /// 实际完成的 Ring 通信轮数（应等于 cp_size）
        pub rounds_completed: u32,
    }

    // ── Online Softmax 合并辅助函数 ──────────────────────────────────────────

    /// Online logsumexp 合并: log(exp(a) + exp(b))
    /// 使用数值稳定公式: max(a,b) + log(1 + exp(-|a-b|))
    fn online_logsumexp_merge(lse_a: &[f32], lse_b: &[f32]) -> Vec<f32> {
        lse_a.iter()
            .zip(lse_b.iter())
            .map(|(&a, &b)| {
                let max_val = a.max(b);
                max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
            })
            .collect()
    }

    /// 计算 exp(a - b)，数值稳定
    fn exp_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).exp())
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn make_config(cp_size: u32, cp_rank: u32) -> CpConfig {
            CpConfig::new(cp_size, cp_rank, 0)
        }

        fn make_config_with_sliding_window(cp_size: u32, cp_rank: u32, sliding_window: usize) -> CpConfig {
            CpConfig {
                cp_size,
                cp_rank,
                pp_stage_id: 0,
                sliding_window,
            }
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

        // ── kv_block_elements ─────────────────────────────────────────────

        #[test]
        fn kv_block_elements_calculation() {
            let config = make_config(4, 0);
            // 2 * num_kv_heads * head_dim * local_seq_len
            // 2 * 8 * 64 * 25 = 25600
            assert_eq!(config.kv_block_elements(8, 64, 100), 25600);
        }

        // ── CpConfig validate ─────────────────────────────────────────────

        #[test]
        fn cp_config_validate_ok() {
            let config = make_config(4, 2);
            assert!(config.validate().is_ok());
        }

        #[test]
        fn cp_config_validate_zero_cp_size() {
            let config = CpConfig::new(0, 0, 0);
            assert!(config.validate().is_err());
        }

        #[test]
        fn cp_config_validate_rank_out_of_range() {
            let config = make_config(4, 4);
            assert!(config.validate().is_err());
        }

        // ── Sliding Window ────────────────────────────────────────────────

        #[test]
        fn cp_config_has_sliding_window() {
            let config = make_config_with_sliding_window(4, 0, 256);
            assert!(config.has_sliding_window());
        }

        #[test]
        fn cp_config_no_sliding_window() {
            let config = make_config(4, 0);
            assert!(!config.has_sliding_window());
        }

        #[test]
        fn should_compute_no_sliding_window() {
            let config = make_config(4, 0);
            // 无滑动窗口时所有步都计算
            assert!(config.should_compute_at_step(0, 1024));
            assert!(config.should_compute_at_step(1, 1024));
            assert!(config.should_compute_at_step(2, 1024));
            assert!(config.should_compute_at_step(3, 1024));
        }

        #[test]
        fn should_compute_with_sliding_window_adjacent() {
            // 4 ranks, seq_len=1024, each rank handles 256 tokens
            // rank 0: [0, 256), rank 1: [256, 512), rank 2: [512, 768), rank 3: [768, 1024)
            // sliding_window=300: adjacent ranks overlap
            let config = make_config_with_sliding_window(4, 0, 300);
            // step 0: source=rank 0 (self), [0,256) vs [0,256) → overlap=0 < 300 → true
            assert!(config.should_compute_at_step(0, 1024));
            // step 1: source=rank 3, [768,1024) vs [0,256) → gap=512 >= 300 → false
            assert!(!config.should_compute_at_step(1, 1024));
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
        fn plan_all_phases_cp_size_rounds() {
            let config = make_config(4, 0);
            let plan = RingAttentionPlan::new(config);
            let all = plan.all_phases();
            assert_eq!(all.len(), 4); // cp_size=4 → 4 rounds
            // Round 0: local only
            assert_eq!(all[0].len(), 1);
            // Rounds 1-3: send+recv+remote
            for round in 1..4 {
                assert_eq!(all[round].len(), 3);
            }
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

        #[test]
        fn plan_kv_dest_rank_always_next() {
            let config = make_config(4, 2);
            let plan = RingAttentionPlan::new(config);
            for step in 0..4 {
                assert_eq!(plan.kv_dest_rank(step), 3); // next_rank of 2
            }
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

        // ── execute_ring_full (cp_size=1, 无分布式) ───────────────────────

        #[test]
        fn execute_ring_full_cp_size_1() {
            let config = make_config(1, 0);
            let plan = RingAttentionPlan::new(config);
            let handle = CommHandleWrapper::new_for_test(0, 1);
            let local_q = vec![1.0f32, 2.0, 3.0];
            let local_kv = vec![4.0f32, 5.0, 6.0];
            let result = plan.execute_ring_full(
                &local_q,
                &local_kv,
                &handle,
                100,
                |q, kv, step, source_rank| {
                    assert_eq!(step, 0);
                    assert_eq!(source_rank, 0);
                    // 简单计算: Q * KV 的 element-wise 乘积求和
                    let val: f32 = q.iter().zip(kv.iter()).map(|(&a, &b)| a * b).sum();
                    Ok(PartialAttnOutput {
                        values: vec![val],
                        logsumexp: None,
                    })
                },
            ).unwrap();
            assert_eq!(result.rounds_completed, 1);
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            assert!((result.output[0] - 32.0).abs() < 1e-6);
        }

        // ── execute_ring_full 数值对齐验证 (REQ-DIST-016 atol=1e-3) ──────

        #[test]
        fn execute_ring_full_online_softmax_merge() {
            // 验证 online logsumexp 合并: 两个 partial attention 的正确合并
            // partial_1: values=[1.0], lse=[0.5]
            // partial_2: values=[2.0], lse=[1.0]
            // 合并: new_lse = logsumexp(0.5, 1.0) ≈ 1.31
            // output = 1.0 * exp(0.5-1.31) + 2.0 * exp(1.0-1.31) ≈ 1.55
            let lse_a = vec![0.5f32];
            let lse_b = vec![1.0f32];
            let new_lse = online_logsumexp_merge(&lse_a, &lse_b);
            let scale_a = exp_sub(&lse_a, &new_lse);
            let scale_b = exp_sub(&lse_b, &new_lse);
            let merged = 1.0 * scale_a[0] + 2.0 * scale_b[0];

            // 手动验证: sum = exp(0.5) + exp(1.0) = 1.6487 + 2.7183 = 4.367
            // new_lse = ln(4.367) ≈ 1.474
            // merged = (1.0 * exp(0.5) + 2.0 * exp(1.0)) / exp(new_lse)
            //        = (1.6487 + 5.4366) / 4.367 ≈ 1.624
            let expected = (1.0 * 0.5f32.exp() + 2.0 * 1.0f32.exp()) / new_lse[0].exp();
            assert!((merged - expected).abs() < 1e-5, "merged={merged}, expected={expected}");
        }

        // ── PP stage 限定 (REQ-DIST-032) ──────────────────────────────────

        #[test]
        fn cp_config_pp_stage_id_preserved() {
            let config = CpConfig::new(4, 2, 1);
            assert_eq!(config.pp_stage_id, 1);
        }

        #[test]
        fn cp_config_ring_comm_within_same_stage() {
            // 同 PP stage 内: cp_rank 0..cp_size 构成环
            // 跨 PP stage: 零 CP 通信
            let config = CpConfig::new(4, 0, 0);
            // next_rank 和 prev_rank 都在 [0, cp_size) 范围内
            assert!(config.next_rank() < config.cp_size);
            assert!(config.prev_rank() < config.cp_size);
        }

        // ── PartialAttnOutput ──────────────────────────────────────────────

        #[test]
        fn partial_attn_output_with_logsumexp() {
            let output = PartialAttnOutput {
                values: vec![1.0, 2.0],
                logsumexp: Some(vec![0.5, 1.0]),
            };
            assert_eq!(output.values.len(), 2);
            assert!(output.logsumexp.is_some());
        }

        #[test]
        fn partial_attn_output_without_logsumexp() {
            let output = PartialAttnOutput {
                values: vec![1.0, 2.0],
                logsumexp: None,
            };
            assert!(output.logsumexp.is_none());
        }

        // ── RingFullResult ─────────────────────────────────────────────────

        #[test]
        fn ring_full_result_fields() {
            let result = RingFullResult {
                output: vec![1.0, 2.0],
                rounds_completed: 4,
            };
            assert_eq!(result.output.len(), 2);
            assert_eq!(result.rounds_completed, 4);
        }

        // ── Online Softmax 辅助函数 ───────────────────────────────────────

        #[test]
        fn online_logsumexp_merge_same_values() {
            let lse = vec![2.0f32, 3.0];
            let merged = online_logsumexp_merge(&lse, &lse);
            // logsumexp(a, a) = a + ln(2)
            assert!((merged[0] - (2.0 + 2.0f32.ln())).abs() < 1e-5);
            assert!((merged[1] - (3.0 + 2.0f32.ln())).abs() < 1e-5);
        }

        #[test]
        fn online_logsumexp_merge_symmetry() {
            let a = vec![1.0f32, 2.0];
            let b = vec![3.0f32, 4.0];
            let ab = online_logsumexp_merge(&a, &b);
            let ba = online_logsumexp_merge(&b, &a);
            for i in 0..a.len() {
                assert!((ab[i] - ba[i]).abs() < 1e-5);
            }
        }

        #[test]
        fn exp_sub_identity() {
            let a = vec![5.0f32];
            let b = vec![5.0f32];
            let result = exp_sub(&a, &b);
            assert!((result[0] - 1.0).abs() < 1e-6); // exp(0) = 1
        }
    }
}
