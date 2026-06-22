//! SAGUARO 分布式推测解码 (REQ-DIST-017)
//!
//! SAGUARO (Speculative Acceleration with GPU-Uncoupled
//! Adaptive Routing Optimization) 分布式推测解码：
//! 1. Draft GPU: 快速生成 draft tokens
//! 2. 传输: draft tokens 从 draft GPU 发送到 verify GPU
//! 3. Verify GPU: 验证 draft tokens，接受/拒绝
//! 4. 回传: 验证结果从 verify GPU 发送回 draft GPU
//!
//! 与现有 SpecDecodingMode::Saguaro 协同工作：
//! - SpecDecodingMode::Saguaro 定义模式选择（单 GPU → EESD, 多 GPU → SAGUARO）
//! - 本模块定义分布式通信阶段、配置与执行结果
//!
//! 通信执行通过 CommHandleWrapper 的 send_bytes / recv_bytes 实现。

#[cfg(feature = "nccl")]
pub mod saguaro {
    use crate::engine::distributed_config::CommHandleWrapper;

    /// SAGUARO 三阶段通信 (REQ-DIST-017)
    ///
    /// SAGUARO (Speculative Acceleration with GPU-Uncoupled
    /// Adaptive Routing Optimization) 分布式推测解码：
    /// 1. Draft GPU: 快速生成 speculative tokens
    /// 2. 传输: draft tokens 从 draft GPU 发送到 verify GPU
    /// 3. Verify GPU: 验证 draft tokens，接受/拒绝
    /// 4. 回传: 验证结果从 verify GPU 发送回 draft GPU
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SaguaroPhase {
        /// Draft 阶段: 快速生成 speculative tokens
        Draft {
            /// 预生成的 draft token 数量
            num_draft_tokens: u32,
        },
        /// 传输阶段: Draft → Verify 通信
        TransferDraft {
            /// 目标 verify GPU rank
            verify_rank: u32,
        },
        /// Verify 阶段: 验证 draft tokens
        Verify {
            /// Draft 来源 GPU rank
            draft_rank: u32,
        },
        /// 回传阶段: Verify → Draft 通信
        TransferResult {
            /// 目标 draft GPU rank
            draft_rank: u32,
        },
    }

    /// SAGUARO 配置 (REQ-DIST-017)
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
        pub fn from_comm_handle(comm_handle: &CommHandleWrapper) -> Self {
            Self {
                draft_rank: 0,
                verify_rank: if comm_handle.world_size() > 1 { 1 } else { 0 },
                draft_length: 5,
            }
        }

        /// 当前 GPU 是否为 Draft GPU
        pub fn is_draft_gpu(&self, comm_handle: &CommHandleWrapper) -> bool {
            comm_handle.rank() == self.draft_rank
        }

        /// 当前 GPU 是否为 Verify GPU
        pub fn is_verify_gpu(&self, comm_handle: &CommHandleWrapper) -> bool {
            comm_handle.rank() == self.verify_rank
        }

        /// 生成完整的 SAGUARO 阶段序列
        pub fn phase_sequence(&self) -> Vec<SaguaroPhase> {
            vec![
                SaguaroPhase::Draft {
                    num_draft_tokens: self.draft_length,
                },
                SaguaroPhase::TransferDraft {
                    verify_rank: self.verify_rank,
                },
                SaguaroPhase::Verify {
                    draft_rank: self.draft_rank,
                },
                SaguaroPhase::TransferResult {
                    draft_rank: self.draft_rank,
                },
            ]
        }

        /// 执行 Draft → Verify 传输 (REQ-DIST-017)
        ///
        /// Draft GPU: 将 draft tokens 序列化为 u32 LE bytes 发送到 verify GPU。
        /// 非 Draft GPU: no-op（仅 Draft GPU 发送）。
        /// 单机模式: 返回 Err（SAGUARO 需要多 GPU）。
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

        /// 接收 Draft tokens (Verify GPU 侧) (REQ-DIST-017)
        ///
        /// Verify GPU: 从 draft GPU 接收 draft tokens 并反序列化。
        /// 非 Verify GPU: 返回空 Vec。
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

        /// 执行 Verify → Draft 回传 (REQ-DIST-017)
        ///
        /// Verify GPU: 将 accepted_count 序列化为 u32 LE bytes 发送回 draft GPU。
        /// 非 Verify GPU: no-op。
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

        /// 接收 Verify 结果 (Draft GPU 侧) (REQ-DIST-017)
        ///
        /// Draft GPU: 从 verify GPU 接收 accepted_count。
        /// 非 Draft GPU: 返回 0。
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

    /// SAGUARO 执行结果 (REQ-DIST-017)
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

        // ── phase_sequence ────────────────────────────────────────────────

        #[test]
        fn phase_sequence_has_four_phases() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(phases.len(), 4);
        }

        #[test]
        fn phase_sequence_draft_first() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(
                phases[0],
                SaguaroPhase::Draft {
                    num_draft_tokens: 5
                }
            );
        }

        #[test]
        fn phase_sequence_transfer_draft_second() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(
                phases[1],
                SaguaroPhase::TransferDraft { verify_rank: 1 }
            );
        }

        #[test]
        fn phase_sequence_verify_third() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(phases[2], SaguaroPhase::Verify { draft_rank: 0 });
        }

        #[test]
        fn phase_sequence_transfer_result_fourth() {
            let config = SaguaroConfig {
                draft_rank: 0,
                verify_rank: 1,
                draft_length: 5,
            };
            let phases = config.phase_sequence();
            assert_eq!(
                phases[3],
                SaguaroPhase::TransferResult { draft_rank: 0 }
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
            let a = SaguaroPhase::Draft {
                num_draft_tokens: 5,
            };
            let b = SaguaroPhase::Draft {
                num_draft_tokens: 5,
            };
            assert_eq!(a, b);
        }

        #[test]
        fn saguaro_phase_inequality() {
            let a = SaguaroPhase::Draft {
                num_draft_tokens: 5,
            };
            let b = SaguaroPhase::Draft {
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
                SaguaroPhase::Draft {
                    num_draft_tokens: 8
                }
            );
            assert_eq!(
                phases[1],
                SaguaroPhase::TransferDraft { verify_rank: 3 }
            );
            assert_eq!(phases[2], SaguaroPhase::Verify { draft_rank: 2 });
            assert_eq!(
                phases[3],
                SaguaroPhase::TransferResult { draft_rank: 2 }
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
    }
}
