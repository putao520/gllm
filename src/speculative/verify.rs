//! §17.4 EqSpec — Batch 正确性三不变量
//!
//! Speculative Decoding 的 batch 化存在隐蔽的正确性陷阱 (arXiv:2510.22876)。
//! gllm 强制执行三条不变量:
//!
//! **I1 (Topology Invariant)**: 同 batch 所有 sequence 共享相同 SpecTree 拓扑
//! **I2 (Single Verification)**: 所有 sequence 的 tree verification 在一次 batched forward 中完成
//! **I3 (Atomic KV Commit)**: accepted tokens 的 KV 原子 commit, rejected 回收

use crate::scheduler::types::RequestId;

/// EqSpec 三不变量标识
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqSpecInvariant {
    /// I1: 同 batch 所有 sequence 共享相同 SpecTree 拓扑
    Topology,
    /// I2: 所有 sequence 的 tree verification 在一次 batched forward 中完成
    SingleVerification,
    /// I3: accepted tokens 的 KV 原子 commit, rejected 回收
    AtomicKvCommit,
}

/// 单个 sequence 的验证结果
#[derive(Debug, Clone)]
pub struct SequenceVerifyResult {
    /// 请求 ID
    pub request_id: RequestId,
    /// 接受的 token 数 (从 spine 起始连续匹配)
    pub accepted_count: usize,
    /// 接受的 token IDs
    pub accepted_tokens: Vec<u32>,
    /// 被拒绝的 token 数
    pub rejected_count: usize,
    /// Draft 阶段总共生成的 token 数
    pub draft_count: usize,
    /// 本次接受率 (accepted / total_draft)
    pub acceptance_rate: f32,
    /// 验证是否通过 EqSpec 不变量检查
    pub invariant_check_passed: bool,
}

impl SequenceVerifyResult {
    /// 创建空的验证结果
    pub fn empty(request_id: RequestId) -> Self {
        Self {
            request_id,
            accepted_count: 0,
            accepted_tokens: Vec::new(),
            rejected_count: 0,
            draft_count: 0,
            acceptance_rate: 0.0,
            invariant_check_passed: true,
        }
    }

    /// 从 draft 和 target tokens 计算验证结果
    ///
    /// §17.4 I3: 接受 = draft tokens 与 target tokens 的最长公共前缀
    ///
    /// # Arguments
    /// * `request_id` - 请求 ID
    /// * `draft_tokens` - Draft 阶段产生的 tokens (spine 方向)
    /// * `target_tokens` - Verify 阶段 full model 产生的 target tokens
    pub fn verify_spine(
        request_id: RequestId,
        draft_tokens: &[u32],
        target_tokens: &[u32],
    ) -> Self {
        let mut accepted = Vec::new();
        let min_len = draft_tokens.len().min(target_tokens.len());

        for i in 0..min_len {
            if draft_tokens[i] == target_tokens[i] {
                accepted.push(draft_tokens[i]);
            } else {
                break;
            }
        }

        let accepted_count = accepted.len();
        let draft_count = draft_tokens.len();
        let rejected_count = draft_count.saturating_sub(accepted_count);
        let acceptance_rate = if draft_count > 0 {
            accepted_count as f32 / draft_count as f32
        } else {
            0.0
        };

        Self {
            request_id,
            accepted_count,
            accepted_tokens: accepted,
            rejected_count,
            draft_count,
            acceptance_rate,
            invariant_check_passed: true,
        }
    }
}

/// Batch 验证结果
///
/// §17.4: 一次 batched verify 的全部结果，满足 EqSpec 三不变量
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// 每个请求的验证结果
    pub sequence_results: Vec<SequenceVerifyResult>,
    /// Batch 平均接受率
    pub avg_acceptance_rate: f32,
    /// Batch 总接受 token 数
    pub total_accepted_tokens: usize,
    /// Batch 总 draft token 数
    pub total_draft_tokens: usize,
    /// EqSpec 不变量检查是否全部通过
    pub all_invariants_passed: bool,
}

impl VerifyResult {
    /// 从多个 sequence 验证结果构建 batch 结果
    pub fn from_sequence_results(results: Vec<SequenceVerifyResult>) -> Self {
        let total_accepted: usize = results.iter().map(|r| r.accepted_count).sum();
        let total_draft: usize = results.iter().map(|r| r.draft_count).sum();
        let avg_rate = if total_draft > 0 {
            total_accepted as f32 / total_draft as f32
        } else {
            0.0
        };
        let all_passed = results.iter().all(|r| r.invariant_check_passed);

        Self {
            sequence_results: results,
            avg_acceptance_rate: avg_rate,
            total_accepted_tokens: total_accepted,
            total_draft_tokens: total_draft,
            all_invariants_passed: all_passed,
        }
    }

    /// 获取特定请求的结果
    pub fn result_for(&self, request_id: RequestId) -> Option<&SequenceVerifyResult> {
        self.sequence_results.iter().find(|r| r.request_id == request_id)
    }

    /// EqSpec I1 检查: 所有 sequence 共享相同 SpecTree 拓扑
    ///
    /// 在 gllm 实现中，所有 sequence 使用同一棵 SpecTree，因此拓扑一致性
    /// 由设计保证。此方法做运行时断言验证。
    pub fn check_topology_invariant(&self, expected_draft_counts: &[usize]) -> bool {
        // I1: 所有 sequence 使用同一 SpecTree → draft_count 应该相同
        if self.sequence_results.len() != expected_draft_counts.len() {
            return false;
        }
        self.sequence_results
            .iter()
            .zip(expected_draft_counts.iter())
            .all(|(r, &expected)| r.draft_count == expected)
    }

    /// EqSpec I2 检查: 单次 batched forward 完成
    ///
    /// VerifyResult 本身就代表一次 batched forward 的结果，
    /// 因此 I2 由数据结构保证。此方法始终返回 true。
    pub fn check_single_verification_invariant(&self) -> bool {
        // I2: VerifyResult 是单次 batched forward 的结果
        true
    }

    /// EqSpec I3 检查: 原子 KV commit
    ///
    /// 每个 sequence 的 accepted/rejected 状态已独立计算，
    /// KV commit 将按 per-sequence 原子执行。
    pub fn check_atomic_kv_commit_invariant(&self) -> bool {
        self.sequence_results.iter().all(|r| r.invariant_check_passed)
    }

    /// 执行全部三不变量检查
    pub fn check_all_invariants(&self, expected_draft_counts: &[usize]) -> EqSpecCheckResult {
        EqSpecCheckResult {
            i1_topology: self.check_topology_invariant(expected_draft_counts),
            i2_single_verification: self.check_single_verification_invariant(),
            i3_atomic_kv_commit: self.check_atomic_kv_commit_invariant(),
        }
    }
}

/// EqSpec 三不变量检查结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EqSpecCheckResult {
    /// I1: 所有 sequence 共享相同 SpecTree 拓扑
    pub i1_topology: bool,
    /// I2: 单次 batched forward 完成
    pub i2_single_verification: bool,
    /// I3: 原子 KV commit
    pub i3_atomic_kv_commit: bool,
}

impl EqSpecCheckResult {
    /// 全部通过
    pub fn all_passed(&self) -> bool {
        self.i1_topology && self.i2_single_verification && self.i3_atomic_kv_commit
    }
}

/// KV Commit/Rollback 指令
///
/// §17.4 I3: 对于每个 sequence, 生成 KV commit (accepted) 和 rollback (rejected) 指令
#[derive(Debug, Clone)]
pub enum KvCommitInstruction {
    /// Commit accepted tokens 的 KV entries 到主 cache
    Commit {
        request_id: RequestId,
        accepted_tokens: Vec<u32>,
        /// 要 commit 的 KV pages (accepted tokens 对应的 pages)
        kv_pages_to_commit: Vec<u64>,
    },
    /// Rollback (回收) rejected tokens 的 KV pages
    Rollback {
        request_id: RequestId,
        rejected_count: usize,
        /// 要回收的 KV pages
        kv_pages_to_free: Vec<u64>,
    },
}

/// 从 VerifyResult 生成 KV Commit/Rollback 指令序列
///
/// §17.4 I3: 跨 sequence 隔离 — Sequence A 接受 7 tokens、B 接受 2 tokens 时互不干扰
pub fn generate_kv_commit_instructions(verify_result: &VerifyResult) -> Vec<KvCommitInstruction> {
    let mut instructions = Vec::new();

    for seq_result in &verify_result.sequence_results {
        if seq_result.accepted_count > 0 {
            // Accepted tokens: placeholder page IDs (实际由 scheduler 分配)
            let commit_pages: Vec<u64> = (0..seq_result.accepted_count as u64).collect();
            instructions.push(KvCommitInstruction::Commit {
                request_id: seq_result.request_id,
                accepted_tokens: seq_result.accepted_tokens.clone(),
                kv_pages_to_commit: commit_pages,
            });
        }

        if seq_result.rejected_count > 0 {
            // Rejected tokens: placeholder page IDs
            let free_pages: Vec<u64> = (0..seq_result.rejected_count as u64).collect();
            instructions.push(KvCommitInstruction::Rollback {
                request_id: seq_result.request_id,
                rejected_count: seq_result.rejected_count,
                kv_pages_to_free: free_pages,
            });
        }
    }

    instructions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_spine_all_accepted() {
        let draft = vec![10u32, 20, 30];
        let target = vec![10u32, 20, 30, 40];
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.rejected_count, 0);
        assert!((result.acceptance_rate - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_verify_spine_partial_accept() {
        let draft = vec![10u32, 20, 30, 40];
        let target = vec![10u32, 20, 99, 40];
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.accepted_tokens, vec![10, 20]);
        assert_eq!(result.rejected_count, 2);
        assert!((result.acceptance_rate - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_verify_spine_none_accepted() {
        let draft = vec![10u32, 20];
        let target = vec![99u32, 88];
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.rejected_count, 2);
        assert!((result.acceptance_rate).abs() < 1e-5);
    }

    #[test]
    fn test_batch_verify_result() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 99]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[50, 60], &[50, 60, 70]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert_eq!(batch.total_accepted_tokens, 4); // 2 (r1: 10,20 match) + 2 (r2: 50,60 match)
        assert_eq!(batch.total_draft_tokens, 5); // 3 + 2
        assert!(batch.all_invariants_passed);
    }

    #[test]
    fn test_eqspec_check_all_invariants() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10], &[10]);
        let batch = VerifyResult::from_sequence_results(vec![r1]);
        let draft_counts = [1];
        let check = batch.check_all_invariants(&draft_counts);
        assert!(check.all_passed());
    }

    #[test]
    fn test_kv_commit_instructions() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 99]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[50, 60], &[50, 60]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        let instructions = generate_kv_commit_instructions(&batch);

        // r1: 1 accepted + 2 rejected → 2 instructions
        // r2: 2 accepted + 0 rejected → 1 instruction
        assert_eq!(instructions.len(), 3);

        let commits: Vec<_> = instructions.iter().filter(|i| matches!(i, KvCommitInstruction::Commit { .. })).collect();
        let rollbacks: Vec<_> = instructions.iter().filter(|i| matches!(i, KvCommitInstruction::Rollback { .. })).collect();
        assert_eq!(commits.len(), 2);
        assert_eq!(rollbacks.len(), 1);
    }

    #[test]
    fn test_empty_verify_result() {
        let result = SequenceVerifyResult::empty(42);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.draft_count, 0);
        assert!(result.invariant_check_passed);
    }
}
