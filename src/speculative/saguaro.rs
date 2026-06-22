//! SAGUARO 分布式推测解码 (REQ-DIST-017)
//!
//! SAGUARO (Speculative Acceleration with GPU-Uncoupled
//! Adaptive Routing Optimization) 分布式推测解码：
//! 1. Draft GPU: 快速生成 draft tokens
//! 2. 传输: draft tokens 从 draft GPU 发送到 verify GPU
//! 3. Verify GPU: 验证 draft tokens，接受/拒绝
//! 4. 回传: 验证结果从 verify GPU 发送回 draft GPU
//! 5. 更新: Draft GPU 根据验证结果更新状态
//!
//! 与现有 SpecDecodingMode::Saguaro 协同工作：
//! - SpecDecodingMode::Saguaro 定义模式选择（单 GPU → EESD, 多 GPU → SAGUARO）
//! - 本模块定义分布式通信阶段、配置与执行结果
//!
//! 通信执行通过 CommHandleWrapper 的 send_bytes / recv_bytes 实现。
//!
//! SPEC 关联:
//! - REQ-DIST-017: SAGUARO 分布式推测解码
//! - ENT-DIST-SAGUARO: SaguaroDistSpec 实体
//! - DF-DIST-006: CP/SAGUARO 数据流
//! - CF-DIST-005: Ring/Speculative 控制流
//! - TEST-DIST-017: 集成测试

#[cfg(feature = "nccl")]
pub mod saguaro {
    use crate::engine::distributed_config::CommHandleWrapper;

    /// SAGUARO 五阶段流水线 (REQ-DIST-017, ENT-DIST-SAGUARO)
    ///
    /// 与 SPEC ENT-DIST-SAGUARO SaguaroStage 对齐:
    /// DraftGenerate | SendCandidates | VerifyAccept | SendResults | UpdateState
    ///
    /// 流水线重叠 (CF-DIST-005): DraftGenerate(N+1) || VerifyAccept(N)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SaguaroPhase {
        /// DraftGenerate: Draft GPU 快速生成 speculative tokens
        DraftGenerate {
            /// 预生成的 draft token 数量
            num_draft_tokens: u32,
        },
        /// SendCandidates: Draft → Verify 通信
        SendCandidates {
            /// 目标 verify GPU rank
            verify_rank: u32,
        },
        /// VerifyAccept: Verify GPU 验证 draft tokens
        VerifyAccept {
            /// Draft 来源 GPU rank
            draft_rank: u32,
        },
        /// SendResults: Verify → Draft 通信
        SendResults {
            /// 目标 draft GPU rank
            draft_rank: u32,
        },
        /// UpdateState: Draft GPU 根据验证结果更新状态
        UpdateState {
            /// 被接受的 token 数量
            accepted_count: u32,
        },
    }

    /// SAGUARO 配置 (REQ-DIST-017, ENT-DIST-SAGUARO)
    // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [dataflow:DF-DIST-006]
    #[derive(Debug, Clone, PartialEq)]
    pub struct SaguaroConfig {
        /// Draft GPU rank
        pub draft_rank: u32,
        /// Verify GPU rank
        pub verify_rank: u32,
        /// 每次 draft 的 token 数量
        pub draft_length: u32,
    }

    impl SaguaroConfig {
        /// 从 CommHandleWrapper 推导 SAGUARO 配置
        /// 默认: rank 0 = draft, rank 1 = verify
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [lifecycle:init]
        pub fn from_comm_handle(comm_handle: &CommHandleWrapper) -> Self {
            Self {
                draft_rank: 0,
                verify_rank: if comm_handle.world_size() > 1 { 1 } else { 0 },
                draft_length: 5,
            }
        }

        /// 当前 GPU 是否为 Draft GPU (REQ-DIST-017)
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [controlflow:CF-DIST-005]
        pub fn is_draft_gpu(&self, comm_handle: &CommHandleWrapper) -> bool {
            comm_handle.rank() == self.draft_rank
        }

        /// 当前 GPU 是否为 Verify GPU (REQ-DIST-017)
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [controlflow:CF-DIST-005]
        pub fn is_verify_gpu(&self, comm_handle: &CommHandleWrapper) -> bool {
            comm_handle.rank() == self.verify_rank
        }

        /// 生成完整的 SAGUARO 五阶段序列 (REQ-DIST-017, CF-DIST-005)
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [controlflow:CF-DIST-005]
        pub fn phase_sequence(&self) -> Vec<SaguaroPhase> {
            vec![
                SaguaroPhase::DraftGenerate {
                    num_draft_tokens: self.draft_length,
                },
                SaguaroPhase::SendCandidates {
                    verify_rank: self.verify_rank,
                },
                SaguaroPhase::VerifyAccept {
                    draft_rank: self.draft_rank,
                },
                SaguaroPhase::SendResults {
                    draft_rank: self.draft_rank,
                },
                SaguaroPhase::UpdateState { accepted_count: 0 },
            ]
        }

        /// 执行 Draft → Verify 传输 (REQ-DIST-017, DF-DIST-006)
        ///
        /// Draft GPU: 将 draft tokens 序列化为 u32 LE bytes 发送到 verify GPU。
        /// 非 Draft GPU: no-op（仅 Draft GPU 发送）。
        /// 单机模式: 返回 Err（SAGUARO 需要多 GPU）。
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [dataflow:DF-DIST-006]
        pub fn transfer_draft_tokens(
            &self,
            draft_tokens: &[u32],
            comm_handle: &CommHandleWrapper,
        ) -> Result<(), String> {
            if !comm_handle.is_distributed() {
                return Err("SAGUARO requires multi-GPU setup".to_string());
            }

            if comm_handle.rank() == self.draft_rank {
                // Draft GPU: 发送 draft tokens 到 verify GPU
                let token_bytes: Vec<u8> = draft_tokens
                    .iter()
                    .flat_map(|&t| t.to_le_bytes())
                    .collect();
                comm_handle.send_bytes(self.verify_rank, &token_bytes)?;
            }
            Ok(())
        }

        /// 接收 Draft tokens (Verify GPU 侧) (REQ-DIST-017, DF-DIST-006)
        ///
        /// Verify GPU: 从 draft GPU 接收 draft tokens 并反序列化。
        /// 非 Verify GPU: 返回空 Vec。
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [dataflow:DF-DIST-006]
        pub fn receive_draft_tokens(
            &self,
            comm_handle: &CommHandleWrapper,
        ) -> Result<Vec<u32>, String> {
            if !comm_handle.is_distributed() {
                return Err("SAGUARO requires multi-GPU setup".to_string());
            }

            if comm_handle.rank() == self.verify_rank {
                // Verify GPU: 从 draft GPU 接收 draft tokens
                let byte_count = self.draft_length as usize * 4; // u32 = 4 bytes
                let raw = comm_handle.recv_bytes(self.draft_rank, byte_count)?;
                let tokens: Vec<u32> = raw
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(tokens)
            } else {
                Ok(vec![])
            }
        }

        /// 执行 Verify → Draft 回传 (REQ-DIST-017, DF-DIST-006)
        ///
        /// Verify GPU: 将 accepted_count 序列化为 u32 LE bytes 发送回 draft GPU。
        /// 非 Verify GPU: no-op。
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [dataflow:DF-DIST-006]
        pub fn transfer_verify_result(
            &self,
            accepted_count: u32,
            comm_handle: &CommHandleWrapper,
        ) -> Result<(), String> {
            if !comm_handle.is_distributed() {
                return Err("SAGUARO requires multi-GPU setup".to_string());
            }

            if comm_handle.rank() == self.verify_rank {
                // Verify GPU: 发送验证结果回 draft GPU
                let result_bytes = accepted_count.to_le_bytes();
                comm_handle.send_bytes(self.draft_rank, &result_bytes)?;
            }
            Ok(())
        }

        /// 接收 Verify 结果 (Draft GPU 侧) (REQ-DIST-017, DF-DIST-006)
        ///
        /// Draft GPU: 从 verify GPU 接收 accepted_count。
        /// 非 Draft GPU: 返回 0。
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [dataflow:DF-DIST-006]
        pub fn receive_verify_result(
            &self,
            comm_handle: &CommHandleWrapper,
        ) -> Result<u32, String> {
            if !comm_handle.is_distributed() {
                return Err("SAGUARO requires multi-GPU setup".to_string());
            }

            if comm_handle.rank() == self.draft_rank {
                // Draft GPU: 从 verify GPU 接收验证结果
                let raw = comm_handle.recv_bytes(self.verify_rank, 4)?;
                Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
            } else {
                Ok(0)
            }
        }
    }

    /// SAGUARO 累积接受率统计 (REQ-DIST-017)
    ///
    /// 跨多轮 SAGUARO 推测解码累积跟踪 draft/accepted tokens，
    /// 提供准确的接受率统计用于自适应调度决策。
    // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO]
    #[derive(Debug, Clone)]
    pub struct SaguaroAcceptanceTracker {
        /// 累积 draft token 总数
        total_draft_tokens: u64,
        /// 累积 accepted token 总数
        total_accepted_tokens: u64,
        /// EMA 平滑接受率 (α=0.3)
        acceptance_rate_ema: f64,
        /// EMA alpha 系数
        ema_alpha: f64,
    }

    impl SaguaroAcceptanceTracker {
        /// 创建新的接受率跟踪器
        pub fn new() -> Self {
            Self {
                total_draft_tokens: 0,
                total_accepted_tokens: 0,
                acceptance_rate_ema: 0.5,
                ema_alpha: 0.3,
            }
        }

        /// 记录一轮 SAGUARO 结果 (REQ-DIST-017)
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO]
        pub fn record(&mut self, draft_tokens: u32, accepted_tokens: u32) {
            self.total_draft_tokens += draft_tokens as u64;
            self.total_accepted_tokens += accepted_tokens as u64;

            // EMA 更新
            let current_rate = if draft_tokens > 0 {
                accepted_tokens as f64 / draft_tokens as f64
            } else {
                0.0
            };
            self.acceptance_rate_ema = self.ema_alpha * current_rate
                + (1.0 - self.ema_alpha) * self.acceptance_rate_ema;
        }

        /// 获取累积接受率 (REQ-DIST-017)
        pub fn acceptance_rate(&self) -> f64 {
            if self.total_draft_tokens == 0 {
                return 0.0;
            }
            self.total_accepted_tokens as f64 / self.total_draft_tokens as f64
        }

        /// 获取 EMA 平滑接受率 (REQ-DIST-017)
        pub fn acceptance_rate_ema(&self) -> f64 {
            self.acceptance_rate_ema
        }

        /// 获取累积 draft token 总数
        pub fn total_draft_tokens(&self) -> u64 {
            self.total_draft_tokens
        }

        /// 获取累积 accepted token 总数
        pub fn total_accepted_tokens(&self) -> u64 {
            self.total_accepted_tokens
        }

        /// 重置统计
        pub fn reset(&mut self) {
            self.total_draft_tokens = 0;
            self.total_accepted_tokens = 0;
            self.acceptance_rate_ema = 0.5;
        }
    }

    impl Default for SaguaroAcceptanceTracker {
        fn default() -> Self {
            Self::new()
        }
    }

    /// SAGUARO 执行结果 (REQ-DIST-017)
    // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO]
    #[derive(Debug, Clone)]
    pub struct SaguaroResult {
        /// Draft 的 token 数量
        pub draft_tokens: u32,
        /// 被接受的 token 数量
        pub accepted_tokens: u32,
        /// 接受率
        pub acceptance_rate: f64,
        /// 总延迟 (ms)
        pub total_latency_ms: f64,
    }

    impl SaguaroResult {
        /// 从 draft/accepted/latency 创建结果 (REQ-DIST-017)
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO]
        pub fn new(draft_tokens: u32, accepted_tokens: u32, total_latency_ms: f64) -> Self {
            let acceptance_rate = if draft_tokens > 0 {
                accepted_tokens as f64 / draft_tokens as f64
            } else {
                0.0
            };
            Self {
                draft_tokens,
                accepted_tokens,
                acceptance_rate,
                total_latency_ms,
            }
        }

        /// 将结果记录到接受率跟踪器 (REQ-DIST-017)
        pub fn record_to(&self, tracker: &mut SaguaroAcceptanceTracker) {
            tracker.record(self.draft_tokens, self.accepted_tokens);
        }
    }

    /// SAGUARO 流水线阶段 (REQ-DIST-017, CF-DIST-005)
    ///
    /// 跟踪当前 SAGUARO 执行所处的流水线阶段，
    /// 支持通信-计算重叠: DraftGenerate(N+1) || VerifyAccept(N)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SaguaroPipelineStage {
        /// 空闲，未在执行 SAGUARO
        Idle,
        /// Draft GPU 正在生成 draft tokens
        Drafting,
        /// 通信中 (Draft → Verify 或 Verify → Draft)
        Communicating,
        /// Verify GPU 正在验证
        Verifying,
        /// 根据验证结果更新状态
        Updating,
    }

    /// SAGUARO 分布式推测解码实体 (REQ-DIST-017, ENT-DIST-SAGUARO)
    ///
    /// 对应 SPEC ENT-DIST-SAGUARO 的 SaguaroDistSpec:
    /// - draft_gpu: Draft 模型所在 GPU
    /// - verify_gpu: Verify 模型所在 GPU
    /// - candidate_buf: 候选 token 缓冲区
    /// - verify_result_buf: 验证结果缓冲区
    /// - pipeline_stage: 当前流水线阶段
    // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [dataflow:DF-DIST-006]
    #[derive(Debug, Clone)]
    pub struct SaguaroDistSpec {
        /// Draft 模型所在 GPU rank (ENT-DIST-SAGUARO.draft_gpu)
        pub draft_gpu: u32,
        /// Verify 模型所在 GPU rank (ENT-DIST-SAGUARO.verify_gpu)
        pub verify_gpu: u32,
        /// 候选 token 缓冲区 (ENT-DIST-SAGUARO.candidate_buf)
        pub candidate_buf: Vec<u32>,
        /// 验证结果缓冲区 (ENT-DIST-SAGUARO.verify_result_buf)
        pub verify_result_buf: Vec<u32>,
        /// 当前流水线阶段 (ENT-DIST-SAGUARO.pipeline_stage)
        pub pipeline_stage: SaguaroPipelineStage,
        /// 接受率跟踪器 (REQ-DIST-017)
        pub acceptance_tracker: SaguaroAcceptanceTracker,
    }

    impl SaguaroDistSpec {
        /// 从 SaguaroConfig 创建 SaguaroDistSpec (REQ-DIST-017)
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [lifecycle:init]
        pub fn from_config(config: &SaguaroConfig) -> Self {
            Self {
                draft_gpu: config.draft_rank,
                verify_gpu: config.verify_rank,
                candidate_buf: Vec::with_capacity(config.draft_length as usize),
                verify_result_buf: Vec::new(),
                pipeline_stage: SaguaroPipelineStage::Idle,
                acceptance_tracker: SaguaroAcceptanceTracker::new(),
            }
        }

        /// 当前 GPU 是否为 Draft GPU (REQ-DIST-017)
        pub fn is_draft_gpu(&self, comm_handle: &CommHandleWrapper) -> bool {
            comm_handle.rank() == self.draft_gpu
        }

        /// 当前 GPU 是否为 Verify GPU (REQ-DIST-017)
        pub fn is_verify_gpu(&self, comm_handle: &CommHandleWrapper) -> bool {
            comm_handle.rank() == self.verify_gpu
        }

        /// 执行完整的 SAGUARO 一轮通信 (REQ-DIST-017, CF-DIST-005)
        ///
        /// Draft GPU: transfer_draft_tokens → receive_verify_result
        /// Verify GPU: receive_draft_tokens → transfer_verify_result
        /// 记录 acceptance_rate 到 tracker
        ///
        /// 返回 (accepted_count, acceptance_rate)
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [controlflow:CF-DIST-005] [dataflow:DF-DIST-006]
        pub fn execute_round(
            &mut self,
            draft_tokens: &[u32],
            comm_handle: &CommHandleWrapper,
        ) -> Result<(u32, f64), String> {
            let config = SaguaroConfig {
                draft_rank: self.draft_gpu,
                verify_rank: self.verify_gpu,
                draft_length: draft_tokens.len() as u32,
            };

            if self.is_draft_gpu(comm_handle) {
                // Draft GPU 流程:
                // 1. 发送 draft tokens
                self.pipeline_stage = SaguaroPipelineStage::Communicating;
                config.transfer_draft_tokens(draft_tokens, comm_handle)?;

                // 2. 接收验证结果
                let accepted = config.receive_verify_result(comm_handle)?;

                // 3. 更新状态
                self.pipeline_stage = SaguaroPipelineStage::Updating;
                self.verify_result_buf = vec![accepted];

                // 4. 记录统计
                self.acceptance_tracker.record(draft_tokens.len() as u32, accepted);
                let rate = self.acceptance_tracker.acceptance_rate();

                self.pipeline_stage = SaguaroPipelineStage::Idle;
                Ok((accepted, rate))
            } else if self.is_verify_gpu(comm_handle) {
                // Verify GPU 流程:
                // 1. 接收 draft tokens
                self.pipeline_stage = SaguaroPipelineStage::Communicating;
                let tokens = config.receive_draft_tokens(comm_handle)?;

                // 2. 验证 (由外部 verify 逻辑完成，这里只传递结果)
                // 注意: 实际的 verify 逻辑在 SpecDecodingState::verify_phase() 中执行
                // 这里只处理通信阶段
                self.candidate_buf = tokens;

                self.pipeline_stage = SaguaroPipelineStage::Idle;
                Ok((0, 0.0)) // Verify GPU 不计算 acceptance_rate
            } else {
                Ok((0, 0.0)) // 非 Draft/Verify GPU: no-op
            }
        }

        /// 执行 Verify GPU 回传 (REQ-DIST-017)
        ///
        /// Verify GPU 在完成验证后调用此方法回传结果。
        // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [dataflow:DF-DIST-006]
        pub fn send_verify_result(
            &mut self,
            accepted_count: u32,
            comm_handle: &CommHandleWrapper,
        ) -> Result<(), String> {
            let config = SaguaroConfig {
                draft_rank: self.draft_gpu,
                verify_rank: self.verify_gpu,
                draft_length: self.candidate_buf.len() as u32,
            };
            self.pipeline_stage = SaguaroPipelineStage::Communicating;
            let result = config.transfer_verify_result(accepted_count, comm_handle);
            self.pipeline_stage = SaguaroPipelineStage::Idle;
            result
        }

        /// 获取累积接受率 (REQ-DIST-017)
        pub fn acceptance_rate(&self) -> f64 {
            self.acceptance_tracker.acceptance_rate()
        }

        /// 获取 EMA 平滑接受率 (REQ-DIST-017)
        pub fn acceptance_rate_ema(&self) -> f64 {
            self.acceptance_tracker.acceptance_rate_ema()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // ── SaguaroConfig ──────────────────────────────────────────────────

        #[test]
        fn config_from_comm_handle_multi_node() {
            let comm = CommHandleWrapper::new_for_test(0, 2);
            let config = SaguaroConfig::from_comm_handle(&comm);
            assert_eq!(config.draft_rank, 0);
            assert_eq!(config.verify_rank, 1);
            assert_eq!(config.draft_length, 5);
        }

        #[test]
        fn config_from_comm_handle_single_node() {
            let comm = CommHandleWrapper::new_for_test(0, 1);
            let config = SaguaroConfig::from_comm_handle(&comm);
            // 单节点时 verify_rank 回退到 0
            assert_eq!(config.verify_rank, 0);
        }

        #[test]
        fn is_draft_gpu_true() {
            let comm = CommHandleWrapper::new_for_test(0, 2);
            let config = SaguaroConfig::from_comm_handle(&comm);
            assert!(config.is_draft_gpu(&comm));
        }

        #[test]
        fn is_draft_gpu_false() {
            let comm = CommHandleWrapper::new_for_test(1, 2);
            let config = SaguaroConfig::from_comm_handle(&comm);
            assert!(!config.is_draft_gpu(&comm));
        }

        #[test]
        fn is_verify_gpu_true() {
            let comm = CommHandleWrapper::new_for_test(1, 2);
            let config = SaguaroConfig::from_comm_handle(&comm);
            assert!(config.is_verify_gpu(&comm));
        }

        #[test]
        fn is_verify_gpu_false() {
            let comm = CommHandleWrapper::new_for_test(0, 2);
            let config = SaguaroConfig::from_comm_handle(&comm);
            assert!(!config.is_verify_gpu(&comm));
        }

        // ── phase_sequence (五阶段, ENT-DIST-SAGUARO SaguaroStage) ─────────

        #[test]
        fn phase_sequence_has_five_phases() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(phases.len(), 5);
        }

        #[test]
        fn phase_sequence_draft_generate_first() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(
                phases[0],
                SaguaroPhase::DraftGenerate {
                    num_draft_tokens: 5
                }
            );
        }

        #[test]
        fn phase_sequence_send_candidates_second() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(
                phases[1],
                SaguaroPhase::SendCandidates { verify_rank: 1 }
            );
        }

        #[test]
        fn phase_sequence_verify_accept_third() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(phases[2], SaguaroPhase::VerifyAccept { draft_rank: 0 });
        }

        #[test]
        fn phase_sequence_send_results_fourth() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(
                phases[3],
                SaguaroPhase::SendResults { draft_rank: 0 }
            );
        }

        #[test]
        fn phase_sequence_update_state_fifth() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(
                phases[4],
                SaguaroPhase::UpdateState { accepted_count: 0 }
            );
        }

        // ── SaguaroResult ─────────────────────────────────────────────────

        #[test]
        fn result_acceptance_rate_normal() {
            let result = SaguaroResult::new(5, 3, 10.0);
            assert_eq!(result.draft_tokens, 5);
            assert_eq!(result.accepted_tokens, 3);
            assert!((result.acceptance_rate - 0.6).abs() < 1e-9);
            assert!((result.total_latency_ms - 10.0).abs() < 1e-9);
        }

        #[test]
        fn result_acceptance_rate_zero_draft() {
            let result = SaguaroResult::new(0, 0, 5.0);
            assert_eq!(result.acceptance_rate, 0.0);
        }

        #[test]
        fn result_acceptance_rate_full_accept() {
            let result = SaguaroResult::new(5, 5, 8.0);
            assert!((result.acceptance_rate - 1.0).abs() < 1e-9);
        }

        #[test]
        fn result_acceptance_rate_partial() {
            let result = SaguaroResult::new(10, 7, 12.5);
            assert!((result.acceptance_rate - 0.7).abs() < 1e-9);
        }

        // ── SaguaroPhase PartialEq ────────────────────────────────────────

        #[test]
        fn saguaro_phase_equality() {
            let a = SaguaroPhase::DraftGenerate {
                num_draft_tokens: 5,
            };
            let b = SaguaroPhase::DraftGenerate {
                num_draft_tokens: 5,
            };
            assert_eq!(a, b);
        }

        #[test]
        fn saguaro_phase_inequality() {
            let a = SaguaroPhase::DraftGenerate {
                num_draft_tokens: 5,
            };
            let b = SaguaroPhase::DraftGenerate {
                num_draft_tokens: 3,
            };
            assert_ne!(a, b);
        }

        // ── 自定义配置 ────────────────────────────────────────────────────

        #[test]
        fn custom_config_phase_sequence() {
            let config = SaguaroConfig {
                draft_rank: 2,
                verify_rank: 3,
                draft_length: 8,
            };
            let phases = config.phase_sequence();
            assert_eq!(
                phases[0],
                SaguaroPhase::DraftGenerate {
                    num_draft_tokens: 8
                }
            );
            assert_eq!(
                phases[1],
                SaguaroPhase::SendCandidates { verify_rank: 3 }
            );
            assert_eq!(phases[2], SaguaroPhase::VerifyAccept { draft_rank: 2 });
            assert_eq!(
                phases[3],
                SaguaroPhase::SendResults { draft_rank: 2 }
            );
            assert_eq!(
                phases[4],
                SaguaroPhase::UpdateState { accepted_count: 0 }
            );
        }

        #[test]
        fn custom_config_role_detection() {
            let config = SaguaroConfig {
                draft_rank: 2,
                verify_rank: 3,
                draft_length: 5,
            };
            let draft_comm = CommHandleWrapper::new_for_test(2, 4);
            let verify_comm = CommHandleWrapper::new_for_test(3, 4);
            let other_comm = CommHandleWrapper::new_for_test(1, 4);

            assert!(config.is_draft_gpu(&draft_comm));
            assert!(!config.is_draft_gpu(&verify_comm));
            assert!(!config.is_draft_gpu(&other_comm));

            assert!(config.is_verify_gpu(&verify_comm));
            assert!(!config.is_verify_gpu(&draft_comm));
            assert!(!config.is_verify_gpu(&other_comm));
        }

        // ── transfer_draft_tokens / receive_draft_tokens ──────────────────

        #[test]
        fn transfer_draft_tokens_single_node_returns_error() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let comm = CommHandleWrapper::new_for_test(0, 1);
            let result = config.transfer_draft_tokens(&[1, 2, 3, 4, 5], &comm);
            assert!(result.is_err());
        }

        #[test]
        fn transfer_draft_tokens_multi_node_without_nccl_returns_error() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let comm = CommHandleWrapper::new_for_test(0, 2);
            // 多机模式但 NCCL 未初始化 → send_bytes 返回错误
            let result = config.transfer_draft_tokens(&[1, 2, 3, 4, 5], &comm);
            assert!(result.is_err());
        }

        #[test]
        fn transfer_draft_tokens_non_draft_gpu_is_noop() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            // 非 Draft GPU (rank 1) → no-op
            let comm = CommHandleWrapper::new_for_test(1, 2);
            let result = config.transfer_draft_tokens(&[1, 2, 3, 4, 5], &comm);
            assert!(result.is_ok());
        }

        // ── transfer_verify_result / receive_verify_result ────────────────

        #[test]
        fn transfer_verify_result_single_node_returns_error() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let comm = CommHandleWrapper::new_for_test(1, 1);
            let result = config.transfer_verify_result(3, &comm);
            assert!(result.is_err());
        }

        #[test]
        fn transfer_verify_result_non_verify_gpu_is_noop() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            // 非 Verify GPU (rank 0) → no-op
            let comm = CommHandleWrapper::new_for_test(0, 2);
            let result = config.transfer_verify_result(3, &comm);
            assert!(result.is_ok());
        }

        #[test]
        fn receive_draft_tokens_single_node_returns_error() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let comm = CommHandleWrapper::new_for_test(1, 1);
            let result = config.receive_draft_tokens(&comm);
            assert!(result.is_err());
        }

        #[test]
        fn receive_verify_result_single_node_returns_error() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let comm = CommHandleWrapper::new_for_test(0, 1);
            let result = config.receive_verify_result(&comm);
            assert!(result.is_err());
        }

        // ── SaguaroAcceptanceTracker (REQ-DIST-017) ───────────────────────

        #[test]
        fn tracker_initial_state() {
            let tracker = SaguaroAcceptanceTracker::new();
            assert_eq!(tracker.total_draft_tokens(), 0);
            assert_eq!(tracker.total_accepted_tokens(), 0);
            assert!((tracker.acceptance_rate() - 0.0).abs() < 1e-9);
            assert!((tracker.acceptance_rate_ema() - 0.5).abs() < 1e-9);
        }

        #[test]
        fn tracker_record_single_round() {
            let mut tracker = SaguaroAcceptanceTracker::new();
            tracker.record(5, 3);
            assert_eq!(tracker.total_draft_tokens(), 5);
            assert_eq!(tracker.total_accepted_tokens(), 3);
            assert!((tracker.acceptance_rate() - 0.6).abs() < 1e-9);
        }

        #[test]
        fn tracker_record_multiple_rounds() {
            let mut tracker = SaguaroAcceptanceTracker::new();
            tracker.record(5, 3); // 60%
            tracker.record(5, 4); // 80%
            // 累积: 7/10 = 0.7
            assert_eq!(tracker.total_draft_tokens(), 10);
            assert_eq!(tracker.total_accepted_tokens(), 7);
            assert!((tracker.acceptance_rate() - 0.7).abs() < 1e-9);
        }

        #[test]
        fn tracker_ema_updates() {
            let mut tracker = SaguaroAcceptanceTracker::new();
            // 初始 EMA = 0.5, alpha = 0.3
            tracker.record(5, 3); // current_rate = 0.6
            // EMA = 0.3 * 0.6 + 0.7 * 0.5 = 0.18 + 0.35 = 0.53
            assert!((tracker.acceptance_rate_ema() - 0.53).abs() < 1e-9);
        }

        #[test]
        fn tracker_zero_draft_rate() {
            let mut tracker = SaguaroAcceptanceTracker::new();
            tracker.record(0, 0);
            assert_eq!(tracker.total_draft_tokens(), 0);
            assert!((tracker.acceptance_rate() - 0.0).abs() < 1e-9);
        }

        #[test]
        fn tracker_reset() {
            let mut tracker = SaguaroAcceptanceTracker::new();
            tracker.record(5, 3);
            tracker.reset();
            assert_eq!(tracker.total_draft_tokens(), 0);
            assert_eq!(tracker.total_accepted_tokens(), 0);
            assert!((tracker.acceptance_rate_ema() - 0.5).abs() < 1e-9);
        }

        #[test]
        fn tracker_default() {
            let tracker = SaguaroAcceptanceTracker::default();
            assert_eq!(tracker.total_draft_tokens(), 0);
        }

        // ── SaguaroResult.record_to ───────────────────────────────────────

        #[test]
        fn result_record_to_tracker() {
            let result = SaguaroResult::new(5, 3, 10.0);
            let mut tracker = SaguaroAcceptanceTracker::new();
            result.record_to(&mut tracker);
            assert_eq!(tracker.total_draft_tokens(), 5);
            assert_eq!(tracker.total_accepted_tokens(), 3);
        }

        // ── SaguaroDistSpec (ENT-DIST-SAGUARO) ────────────────────────────

        #[test]
        fn dist_spec_from_config() {
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

        #[test]
        fn dist_spec_is_draft_gpu() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let spec = SaguaroDistSpec::from_config(&config);
            let draft_comm = CommHandleWrapper::new_for_test(0, 2);
            let verify_comm = CommHandleWrapper::new_for_test(1, 2);
            assert!(spec.is_draft_gpu(&draft_comm));
            assert!(!spec.is_draft_gpu(&verify_comm));
        }

        #[test]
        fn dist_spec_is_verify_gpu() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let spec = SaguaroDistSpec::from_config(&config);
            let draft_comm = CommHandleWrapper::new_for_test(0, 2);
            let verify_comm = CommHandleWrapper::new_for_test(1, 2);
            assert!(spec.is_verify_gpu(&verify_comm));
            assert!(!spec.is_verify_gpu(&draft_comm));
        }

        #[test]
        fn dist_spec_acceptance_rate_initial() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let spec = SaguaroDistSpec::from_config(&config);
            assert!((spec.acceptance_rate() - 0.0).abs() < 1e-9);
        }

        #[test]
        fn dist_spec_acceptance_rate_after_record() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let mut spec = SaguaroDistSpec::from_config(&config);
            spec.acceptance_tracker.record(5, 3);
            assert!((spec.acceptance_rate() - 0.6).abs() < 1e-9);
        }

        // ── SaguaroPipelineStage ──────────────────────────────────────────

        #[test]
        fn pipeline_stage_idle_default() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let spec = SaguaroDistSpec::from_config(&config);
            assert_eq!(spec.pipeline_stage, SaguaroPipelineStage::Idle);
        }

        #[test]
        fn pipeline_stage_variants_distinct() {
            let stages = [
                SaguaroPipelineStage::Idle,
                SaguaroPipelineStage::Drafting,
                SaguaroPipelineStage::Communicating,
                SaguaroPipelineStage::Verifying,
                SaguaroPipelineStage::Updating,
            ];
            // All variants are distinct
            for i in 0..stages.len() {
                for j in (i + 1)..stages.len() {
                    assert_ne!(stages[i], stages[j]);
                }
            }
        }

        // ── SaguaroDistSpec.execute_round (单机错误路径) ──────────────────

        #[test]
        fn dist_spec_execute_round_single_node_returns_error() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let mut spec = SaguaroDistSpec::from_config(&config);
            let comm = CommHandleWrapper::new_for_test(0, 1);
            let result = spec.execute_round(&[1, 2, 3, 4, 5], &comm);
            assert!(result.is_err());
        }

        #[test]
        fn dist_spec_execute_round_non_draft_non_verify_noop() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let mut spec = SaguaroDistSpec::from_config(&config);
            // rank 2 既不是 draft 也不是 verify → no-op
            let comm = CommHandleWrapper::new_for_test(2, 4);
            let result = spec.execute_round(&[1, 2, 3, 4, 5], &comm);
            assert!(result.is_ok());
            let (accepted, rate) = result.unwrap();
            assert_eq!(accepted, 0);
            assert!((rate - 0.0).abs() < 1e-9);
        }

        // ── SaguaroDistSpec.send_verify_result (单机错误路径) ─────────────

        #[test]
        fn dist_spec_send_verify_result_single_node_returns_error() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let mut spec = SaguaroDistSpec::from_config(&config);
            let comm = CommHandleWrapper::new_for_test(1, 1);
            let result = spec.send_verify_result(3, &comm);
            assert!(result.is_err());
        }
    }
}
