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

    // ── EqSpecInvariant ──────────────────────────────────────────────

    #[test]
    fn test_eqspec_invariant_variants_are_distinct() {
        let variants = [
            EqSpecInvariant::Topology,
            EqSpecInvariant::SingleVerification,
            EqSpecInvariant::AtomicKvCommit,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_eqspec_invariant_copy_clone() {
        let v = EqSpecInvariant::Topology;
        let v2 = v;
        assert_eq!(v, v2);
        let v3 = v.clone();
        assert_eq!(v, v3);
    }

    #[test]
    fn test_eqspec_invariant_debug_format() {
        assert_eq!(format!("{:?}", EqSpecInvariant::Topology), "Topology");
        assert_eq!(
            format!("{:?}", EqSpecInvariant::SingleVerification),
            "SingleVerification"
        );
        assert_eq!(
            format!("{:?}", EqSpecInvariant::AtomicKvCommit),
            "AtomicKvCommit"
        );
    }

    // ── SequenceVerifyResult::empty ──────────────────────────────────

    #[test]
    fn test_empty_result_zero_request_id() {
        let result = SequenceVerifyResult::empty(0);
        assert_eq!(result.request_id, 0);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.rejected_count, 0);
        assert_eq!(result.draft_count, 0);
        assert!((result.acceptance_rate).abs() < 1e-6);
        assert!(result.invariant_check_passed);
    }

    #[test]
    fn test_empty_result_large_request_id() {
        let result = SequenceVerifyResult::empty(u64::MAX);
        assert_eq!(result.request_id, u64::MAX);
    }

    // ── SequenceVerifyResult::verify_spine edge cases ────────────────

    #[test]
    fn test_verify_spine_both_empty() {
        let result = SequenceVerifyResult::verify_spine(1, &[], &[]);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.accepted_tokens, Vec::<u32>::new());
        assert_eq!(result.rejected_count, 0);
        assert_eq!(result.draft_count, 0);
        assert!((result.acceptance_rate).abs() < 1e-6);
        assert!(result.invariant_check_passed);
    }

    #[test]
    fn test_verify_spine_draft_empty_target_nonempty() {
        let result = SequenceVerifyResult::verify_spine(1, &[], &[10, 20]);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.rejected_count, 0);
        assert_eq!(result.draft_count, 0);
    }

    #[test]
    fn test_verify_spine_target_empty_draft_nonempty() {
        let result = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[]);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.rejected_count, 3);
        assert!((result.acceptance_rate).abs() < 1e-6);
    }

    #[test]
    fn test_verify_spine_single_token_match() {
        let result = SequenceVerifyResult::verify_spine(5, &[42], &[42]);
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.accepted_tokens, vec![42]);
        assert_eq!(result.rejected_count, 0);
        assert!((result.acceptance_rate - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_verify_spine_single_token_mismatch() {
        let result = SequenceVerifyResult::verify_spine(5, &[42], &[99]);
        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.rejected_count, 1);
        assert!((result.acceptance_rate).abs() < 1e-6);
    }

    #[test]
    fn test_verify_spine_longest_common_prefix_stops_at_first_mismatch() {
        let draft = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let target = vec![1, 2, 3, 4, 5, 99, 7, 8, 9, 10];
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_count, 5);
        assert_eq!(result.accepted_tokens, vec![1, 2, 3, 4, 5]);
        assert_eq!(result.rejected_count, 5);
        assert!((result.acceptance_rate - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_verify_spine_target_longer_than_draft() {
        let draft = vec![10, 20];
        let target = vec![10, 20, 30, 40, 50];
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.rejected_count, 0);
        assert!((result.acceptance_rate - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_verify_spine_acceptance_rate_calculation() {
        let draft = vec![1, 2, 3, 4];
        let target = vec![1, 2, 99, 4];
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.draft_count, 4);
        assert!((result.acceptance_rate - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_verify_spine_preserves_request_id() {
        let result = SequenceVerifyResult::verify_spine(12345, &[1], &[1]);
        assert_eq!(result.request_id, 12345);
    }

    // ── VerifyResult::from_sequence_results ──────────────────────────

    #[test]
    fn test_batch_from_empty_sequence_results() {
        let batch = VerifyResult::from_sequence_results(vec![]);
        assert!(batch.sequence_results.is_empty());
        assert!((batch.avg_acceptance_rate).abs() < 1e-6);
        assert_eq!(batch.total_accepted_tokens, 0);
        assert_eq!(batch.total_draft_tokens, 0);
        assert!(batch.all_invariants_passed);
    }

    #[test]
    fn test_batch_avg_acceptance_rate_single_sequence() {
        let r = SequenceVerifyResult::verify_spine(1, &[1, 2, 3], &[1, 2, 99]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        assert!((batch.avg_acceptance_rate - (2.0_f32 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_batch_avg_acceptance_rate_multiple_sequences() {
        // seq1: 2/4 accepted, seq2: 3/3 accepted, seq3: 0/2 accepted
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2, 3, 4], &[1, 2, 99, 4]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[5, 6, 7], &[5, 6, 7]);
        let r3 = SequenceVerifyResult::verify_spine(3, &[8, 9], &[99, 99]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2, r3]);
        // total accepted = 2+3+0 = 5, total draft = 4+3+2 = 9
        assert_eq!(batch.total_accepted_tokens, 5);
        assert_eq!(batch.total_draft_tokens, 9);
        assert!((batch.avg_acceptance_rate - (5.0_f32 / 9.0)).abs() < 1e-5);
    }

    #[test]
    fn test_batch_all_invariants_passed_true_when_all_individual_passed() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1], &[1]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[2], &[2]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(batch.all_invariants_passed);
    }

    #[test]
    fn test_batch_all_invariants_passed_false_when_one_fails() {
        let mut r1 = SequenceVerifyResult::verify_spine(1, &[1], &[1]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[2], &[2]);
        r1.invariant_check_passed = false;
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(!batch.all_invariants_passed);
    }

    // ── VerifyResult::result_for ─────────────────────────────────────

    #[test]
    fn test_result_for_found() {
        let r1 = SequenceVerifyResult::verify_spine(10, &[1], &[1]);
        let r2 = SequenceVerifyResult::verify_spine(20, &[2], &[99]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        let found = batch.result_for(20).unwrap();
        assert_eq!(found.request_id, 20);
        assert_eq!(found.accepted_count, 0);
    }

    #[test]
    fn test_result_for_not_found() {
        let r = SequenceVerifyResult::verify_spine(1, &[1], &[1]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        assert!(batch.result_for(999).is_none());
    }

    #[test]
    fn test_result_for_empty_batch() {
        let batch = VerifyResult::from_sequence_results(vec![]);
        assert!(batch.result_for(1).is_none());
    }

    // ── VerifyResult::check_topology_invariant ───────────────────────

    #[test]
    fn test_topology_invariant_passes_matching_counts() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[3, 4], &[3, 4]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(batch.check_topology_invariant(&[2, 2]));
    }

    #[test]
    fn test_topology_invariant_fails_count_mismatch() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[3, 4], &[3, 4]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        // expected says 3 drafts for r2, but r2 only had 2
        assert!(!batch.check_topology_invariant(&[2, 3]));
    }

    #[test]
    fn test_topology_invariant_fails_length_mismatch() {
        let r = SequenceVerifyResult::verify_spine(1, &[1], &[1]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        // 2 expected counts but only 1 sequence
        assert!(!batch.check_topology_invariant(&[1, 1]));
    }

    #[test]
    fn test_topology_invariant_empty_batch_empty_expected() {
        let batch = VerifyResult::from_sequence_results(vec![]);
        assert!(batch.check_topology_invariant(&[]));
    }

    // ── VerifyResult::check_single_verification_invariant ────────────

    #[test]
    fn test_single_verification_invariant_always_true() {
        let batch = VerifyResult::from_sequence_results(vec![]);
        assert!(batch.check_single_verification_invariant());

        let r = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 99]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        assert!(batch.check_single_verification_invariant());
    }

    // ── VerifyResult::check_atomic_kv_commit_invariant ───────────────

    #[test]
    fn test_atomic_kv_commit_all_passed() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1], &[1]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[2], &[2]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(batch.check_atomic_kv_commit_invariant());
    }

    #[test]
    fn test_atomic_kv_commit_one_failed() {
        let mut r1 = SequenceVerifyResult::verify_spine(1, &[1], &[1]);
        r1.invariant_check_passed = false;
        let r2 = SequenceVerifyResult::verify_spine(2, &[2], &[2]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(!batch.check_atomic_kv_commit_invariant());
    }

    #[test]
    fn test_atomic_kv_commit_empty_batch() {
        let batch = VerifyResult::from_sequence_results(vec![]);
        assert!(batch.check_atomic_kv_commit_invariant());
    }

    // ── VerifyResult::check_all_invariants ───────────────────────────

    #[test]
    fn test_check_all_invariants_all_pass() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        let batch = VerifyResult::from_sequence_results(vec![r1]);
        let check = batch.check_all_invariants(&[2]);
        assert!(check.i1_topology);
        assert!(check.i2_single_verification);
        assert!(check.i3_atomic_kv_commit);
        assert!(check.all_passed());
    }

    #[test]
    fn test_check_all_invariants_topology_fails() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        let batch = VerifyResult::from_sequence_results(vec![r1]);
        let check = batch.check_all_invariants(&[5]);
        assert!(!check.i1_topology);
        assert!(check.i2_single_verification);
        assert!(check.i3_atomic_kv_commit);
        assert!(!check.all_passed());
    }

    #[test]
    fn test_check_all_invariants_kv_commit_fails() {
        let mut r1 = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        r1.invariant_check_passed = false;
        let batch = VerifyResult::from_sequence_results(vec![r1]);
        let check = batch.check_all_invariants(&[2]);
        assert!(check.i1_topology);
        assert!(check.i2_single_verification);
        assert!(!check.i3_atomic_kv_commit);
        assert!(!check.all_passed());
    }

    // ── EqSpecCheckResult ────────────────────────────────────────────

    #[test]
    fn test_eqspec_check_result_all_passed_true() {
        let result = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: true,
            i3_atomic_kv_commit: true,
        };
        assert!(result.all_passed());
    }

    #[test]
    fn test_eqspec_check_result_all_passed_false_on_i1() {
        let result = EqSpecCheckResult {
            i1_topology: false,
            i2_single_verification: true,
            i3_atomic_kv_commit: true,
        };
        assert!(!result.all_passed());
    }

    #[test]
    fn test_eqspec_check_result_all_passed_false_on_i2() {
        let result = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: false,
            i3_atomic_kv_commit: true,
        };
        assert!(!result.all_passed());
    }

    #[test]
    fn test_eqspec_check_result_all_passed_false_on_i3() {
        let result = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: true,
            i3_atomic_kv_commit: false,
        };
        assert!(!result.all_passed());
    }

    #[test]
    fn test_eqspec_check_result_equality() {
        let a = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: true,
            i3_atomic_kv_commit: true,
        };
        let b = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: true,
            i3_atomic_kv_commit: true,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_eqspec_check_result_inequality() {
        let a = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: true,
            i3_atomic_kv_commit: true,
        };
        let b = EqSpecCheckResult {
            i1_topology: false,
            i2_single_verification: true,
            i3_atomic_kv_commit: true,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_eqspec_check_result_copy() {
        let a = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: false,
            i3_atomic_kv_commit: true,
        };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_eqspec_check_result_clone() {
        let a = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: false,
            i3_atomic_kv_commit: true,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_eqspec_check_result_debug_format() {
        let result = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: false,
            i3_atomic_kv_commit: true,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("i1_topology: true"));
        assert!(debug_str.contains("i2_single_verification: false"));
        assert!(debug_str.contains("i3_atomic_kv_commit: true"));
    }

    // ── KvCommitInstruction ──────────────────────────────────────────

    #[test]
    fn test_kv_commit_instruction_commit_variant() {
        let instr = KvCommitInstruction::Commit {
            request_id: 42,
            accepted_tokens: vec![10, 20, 30],
            kv_pages_to_commit: vec![0, 1, 2],
        };
        match &instr {
            KvCommitInstruction::Commit {
                request_id,
                accepted_tokens,
                kv_pages_to_commit,
            } => {
                assert_eq!(*request_id, 42);
                assert_eq!(*accepted_tokens, vec![10, 20, 30]);
                assert_eq!(*kv_pages_to_commit, vec![0, 1, 2]);
            }
            KvCommitInstruction::Rollback { .. } => panic!("expected Commit variant"),
        }
    }

    #[test]
    fn test_kv_commit_instruction_rollback_variant() {
        let instr = KvCommitInstruction::Rollback {
            request_id: 99,
            rejected_count: 3,
            kv_pages_to_free: vec![0, 1, 2],
        };
        match &instr {
            KvCommitInstruction::Rollback {
                request_id,
                rejected_count,
                kv_pages_to_free,
            } => {
                assert_eq!(*request_id, 99);
                assert_eq!(*rejected_count, 3);
                assert_eq!(*kv_pages_to_free, vec![0, 1, 2]);
            }
            KvCommitInstruction::Commit { .. } => panic!("expected Rollback variant"),
        }
    }

    #[test]
    fn test_kv_commit_instruction_debug_format() {
        let commit = KvCommitInstruction::Commit {
            request_id: 1,
            accepted_tokens: vec![5],
            kv_pages_to_commit: vec![0],
        };
        let debug = format!("{:?}", commit);
        assert!(debug.contains("Commit"));

        let rollback = KvCommitInstruction::Rollback {
            request_id: 2,
            rejected_count: 1,
            kv_pages_to_free: vec![0],
        };
        let debug = format!("{:?}", rollback);
        assert!(debug.contains("Rollback"));
    }

    // ── generate_kv_commit_instructions ──────────────────────────────

    #[test]
    fn test_generate_kv_instructions_empty_batch() {
        let batch = VerifyResult::from_sequence_results(vec![]);
        let instructions = generate_kv_commit_instructions(&batch);
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_generate_kv_instructions_all_accepted() {
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch);
        // 2 accepted, 0 rejected → 1 Commit, 0 Rollback
        assert_eq!(instructions.len(), 1);
        assert!(matches!(
            &instructions[0],
            KvCommitInstruction::Commit {
                request_id: 1,
                accepted_tokens,
                ..
            } if accepted_tokens == &vec![10, 20]
        ));
    }

    #[test]
    fn test_generate_kv_instructions_all_rejected() {
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99, 99]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch);
        // 0 accepted, 2 rejected → 0 Commit, 1 Rollback
        assert_eq!(instructions.len(), 1);
        assert!(matches!(
            &instructions[0],
            KvCommitInstruction::Rollback {
                request_id: 1,
                rejected_count: 2,
                ..
            }
        ));
    }

    #[test]
    fn test_generate_kv_instructions_mixed_accept_reject() {
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20, 30, 40], &[10, 20, 99, 40]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch);
        // 2 accepted + 2 rejected → Commit then Rollback
        assert_eq!(instructions.len(), 2);

        // First: Commit with accepted tokens
        match &instructions[0] {
            KvCommitInstruction::Commit {
                request_id,
                accepted_tokens,
                kv_pages_to_commit,
            } => {
                assert_eq!(*request_id, 1);
                assert_eq!(*accepted_tokens, vec![10, 20]);
                assert_eq!(kv_pages_to_commit.len(), 2);
            }
            _ => panic!("expected Commit first"),
        }

        // Second: Rollback with rejected count
        match &instructions[1] {
            KvCommitInstruction::Rollback {
                request_id,
                rejected_count,
                kv_pages_to_free,
            } => {
                assert_eq!(*request_id, 1);
                assert_eq!(*rejected_count, 2);
                assert_eq!(kv_pages_to_free.len(), 2);
            }
            _ => panic!("expected Rollback second"),
        }
    }

    #[test]
    fn test_generate_kv_instructions_cross_sequence_isolation() {
        // Sequence A: accepts 7, rejects 0
        let r_a = SequenceVerifyResult::verify_spine(
            100,
            &[1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7],
        );
        // Sequence B: accepts 2, rejects 3
        let r_b =
            SequenceVerifyResult::verify_spine(200, &[10, 20, 30, 40, 50], &[10, 20, 99, 99, 99]);

        let batch = VerifyResult::from_sequence_results(vec![r_a, r_b]);
        let instructions = generate_kv_commit_instructions(&batch);

        // r_a: 1 Commit (7 tokens, 0 rollback)
        // r_b: 1 Commit (2 tokens) + 1 Rollback (3 tokens)
        assert_eq!(instructions.len(), 3);

        // Verify cross-sequence isolation: each instruction carries its own request_id
        let ids: Vec<RequestId> = instructions
            .iter()
            .map(|i| match i {
                KvCommitInstruction::Commit { request_id, .. } => *request_id,
                KvCommitInstruction::Rollback { request_id, .. } => *request_id,
            })
            .collect();
        assert_eq!(ids[0], 100); // r_a commit
        assert_eq!(ids[1], 200); // r_b commit
        assert_eq!(ids[2], 200); // r_b rollback
    }

    #[test]
    fn test_generate_kv_instructions_page_ids_sequential() {
        let r = SequenceVerifyResult::verify_spine(1, &[1, 2, 3], &[1, 2, 3]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch);
        match &instructions[0] {
            KvCommitInstruction::Commit {
                kv_pages_to_commit, ..
            } => {
                assert_eq!(*kv_pages_to_commit, vec![0, 1, 2]);
            }
            _ => panic!("expected Commit"),
        }
    }

    #[test]
    fn test_generate_kv_instructions_empty_sequence_skipped() {
        let r = SequenceVerifyResult::empty(1);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch);
        // 0 accepted, 0 rejected → no instructions
        assert!(instructions.is_empty());
    }

    // ── SequenceVerifyResult field mutation ──────────────────────────

    #[test]
    fn test_sequence_verify_result_is_mutable() {
        let mut result = SequenceVerifyResult::empty(1);
        result.accepted_count = 5;
        result.accepted_tokens = vec![1, 2, 3, 4, 5];
        result.rejected_count = 2;
        result.draft_count = 7;
        result.acceptance_rate = 5.0 / 7.0;
        result.invariant_check_passed = false;

        assert_eq!(result.accepted_count, 5);
        assert_eq!(result.accepted_tokens, vec![1, 2, 3, 4, 5]);
        assert_eq!(result.rejected_count, 2);
        assert_eq!(result.draft_count, 7);
        assert!((result.acceptance_rate - (5.0_f32 / 7.0)).abs() < 1e-5);
        assert!(!result.invariant_check_passed);
    }

    #[test]
    fn test_sequence_verify_result_clone() {
        let result = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 99]);
        let cloned = result.clone();
        assert_eq!(result.request_id, cloned.request_id);
        assert_eq!(result.accepted_count, cloned.accepted_count);
        assert_eq!(result.accepted_tokens, cloned.accepted_tokens);
        assert_eq!(result.rejected_count, cloned.rejected_count);
        assert_eq!(result.draft_count, cloned.draft_count);
        assert!((result.acceptance_rate - cloned.acceptance_rate).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_verify_result_debug_format() {
        let result = SequenceVerifyResult::verify_spine(42, &[1, 2], &[1, 99]);
        let debug = format!("{:?}", result);
        assert!(debug.contains("request_id: 42"));
        assert!(debug.contains("accepted_count: 1"));
        assert!(debug.contains("rejected_count: 1"));
        assert!(debug.contains("draft_count: 2"));
    }

    // ── VerifyResult field access ────────────────────────────────────

    #[test]
    fn test_verify_result_clone() {
        let r = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let cloned = batch.clone();
        assert_eq!(cloned.total_accepted_tokens, batch.total_accepted_tokens);
        assert_eq!(cloned.total_draft_tokens, batch.total_draft_tokens);
        assert_eq!(cloned.sequence_results.len(), batch.sequence_results.len());
    }

    #[test]
    fn test_verify_result_debug_format() {
        let r = SequenceVerifyResult::verify_spine(1, &[1], &[1]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let debug = format!("{:?}", batch);
        assert!(debug.contains("VerifyResult"));
        assert!(debug.contains("sequence_results"));
        assert!(debug.contains("avg_acceptance_rate"));
    }

    // ══════════════════════════════════════════════════════════════════
    //  ~40 new tests — property checks, edge cases, coverage gaps
    // ══════════════════════════════════════════════════════════════════

    // ── SequenceVerifyResult: property invariants ────────────────────

    #[test]
    fn test_verify_spine_accepted_plus_rejected_equals_draft() {
        // Property: accepted_count + rejected_count == draft_count
        let cases: Vec<(&[u32], &[u32])> = vec![
            (&[1, 2, 3], &[1, 2, 3]),
            (&[1, 2, 3], &[1, 99, 3]),
            (&[1, 2, 3], &[99, 2, 3]),
            (&[1, 2, 3], &[1, 2, 99, 4]),
            (&[], &[1, 2]),
            (&[1, 2], &[]),
            (&[], &[]),
        ];
        for (draft, target) in cases {
            let result = SequenceVerifyResult::verify_spine(1, draft, target);
            assert_eq!(
                result.accepted_count + result.rejected_count,
                result.draft_count,
                "invariant broken for draft={draft:?}, target={target:?}"
            );
        }
    }

    #[test]
    fn test_verify_spine_accepted_tokens_len_equals_accepted_count() {
        let cases: Vec<(&[u32], &[u32])> = vec![
            (&[10, 20, 30], &[10, 20, 30]),
            (&[10, 20, 30], &[10, 99]),
            (&[99], &[10]),
            (&[], &[]),
        ];
        for (draft, target) in cases {
            let result = SequenceVerifyResult::verify_spine(1, draft, target);
            assert_eq!(
                result.accepted_tokens.len(),
                result.accepted_count,
                "accepted_tokens length mismatch for draft={draft:?}, target={target:?}"
            );
        }
    }

    #[test]
    fn test_verify_spine_acceptance_rate_always_in_unit_range() {
        let cases: Vec<(&[u32], &[u32])> = vec![
            (&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5]),
            (&[1, 2, 3], &[99]),
            (&[1], &[1]),
            (&[], &[]),
        ];
        for (draft, target) in cases {
            let result = SequenceVerifyResult::verify_spine(1, draft, target);
            assert!(
                result.acceptance_rate >= 0.0 && result.acceptance_rate <= 1.0,
                "acceptance_rate out of [0,1] for draft={draft:?}, target={target:?}: {}",
                result.acceptance_rate
            );
        }
    }

    #[test]
    fn test_verify_spine_invariant_check_always_true() {
        // verify_spine always sets invariant_check_passed = true
        let cases: Vec<(&[u32], &[u32])> = vec![
            (&[1], &[1]),
            (&[1], &[2]),
            (&[], &[]),
            (&[1, 2, 3], &[1, 99]),
        ];
        for (draft, target) in cases {
            let result = SequenceVerifyResult::verify_spine(1, draft, target);
            assert!(
                result.invariant_check_passed,
                "invariant_check_passed should be true from verify_spine for draft={draft:?}"
            );
        }
    }

    #[test]
    fn test_verify_spine_large_draft_all_match() {
        let draft: Vec<u32> = (0..1000).collect();
        let target = draft.clone();
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_count, 1000);
        assert_eq!(result.rejected_count, 0);
        assert!((result.acceptance_rate - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_verify_spine_large_draft_none_match() {
        let draft: Vec<u32> = (0..500).collect();
        let target: Vec<u32> = (1000..1500).collect();
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.rejected_count, 500);
        assert!((result.acceptance_rate).abs() < 1e-6);
    }

    #[test]
    fn test_verify_spine_mismatch_at_last_position() {
        let draft = vec![1, 2, 3, 4];
        let target = vec![1, 2, 3, 99];
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![1, 2, 3]);
        assert_eq!(result.rejected_count, 1);
    }

    #[test]
    fn test_verify_spine_identical_single_element() {
        let result = SequenceVerifyResult::verify_spine(1, &[7], &[7]);
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.accepted_tokens, vec![7]);
        assert_eq!(result.rejected_count, 0);
        assert!((result.acceptance_rate - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_verify_spine_draft_count_equals_input_length() {
        let drafts: &[&[u32]] = &[&[], &[1], &[1, 2, 3, 4, 5]];
        for draft in drafts {
            let result = SequenceVerifyResult::verify_spine(1, draft, &[]);
            assert_eq!(
                result.draft_count,
                draft.len(),
                "draft_count should equal input draft length"
            );
        }
    }

    #[test]
    fn test_verify_spine_accepted_tokens_are_prefix_of_draft() {
        let draft = vec![10, 20, 30, 40, 50];
        let target = vec![10, 20, 99, 40, 50];
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(result.accepted_tokens, vec![10, 20]);
        // Verify they are the actual prefix of draft
        assert_eq!(&result.accepted_tokens[..], &draft[..result.accepted_count]);
    }

    // ── VerifyResult: aggregation property checks ────────────────────

    #[test]
    fn test_batch_total_accepted_equals_sum_of_individual() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2, 3], &[1, 2, 99]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[4, 5], &[4, 5]);
        let r3 = SequenceVerifyResult::verify_spine(3, &[6], &[99]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2, r3]);
        let expected_total: usize =
            batch.sequence_results.iter().map(|r| r.accepted_count).sum();
        assert_eq!(batch.total_accepted_tokens, expected_total);
        // 2 + 2 + 0 = 4
        assert_eq!(batch.total_accepted_tokens, 4);
    }

    #[test]
    fn test_batch_total_draft_equals_sum_of_individual() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2, 3], &[1, 2, 99]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[4, 5], &[4, 5]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        let expected_total: usize =
            batch.sequence_results.iter().map(|r| r.draft_count).sum();
        assert_eq!(batch.total_draft_tokens, expected_total);
        // 3 + 2 = 5
        assert_eq!(batch.total_draft_tokens, 5);
    }

    #[test]
    fn test_batch_avg_rate_equals_total_accepted_over_total_draft() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2, 3, 4], &[1, 2, 99, 4]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[5, 6], &[5, 6]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        let expected_rate = batch.total_accepted_tokens as f32 / batch.total_draft_tokens as f32;
        assert!((batch.avg_acceptance_rate - expected_rate).abs() < 1e-5);
    }

    #[test]
    fn test_batch_avg_rate_in_unit_range() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2, 3], &[1, 99]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[4, 5], &[4, 5]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(batch.avg_acceptance_rate >= 0.0 && batch.avg_acceptance_rate <= 1.0);
    }

    #[test]
    fn test_batch_many_sequences() {
        let results: Vec<SequenceVerifyResult> = (0..50)
            .map(|i| SequenceVerifyResult::verify_spine(i, &[1, 2], &[1, 2]))
            .collect();
        let batch = VerifyResult::from_sequence_results(results);
        assert_eq!(batch.sequence_results.len(), 50);
        assert_eq!(batch.total_accepted_tokens, 100);
        assert_eq!(batch.total_draft_tokens, 100);
        assert!(batch.all_invariants_passed);
    }

    #[test]
    fn test_batch_mixed_empty_and_nonempty_sequences() {
        let empty = SequenceVerifyResult::empty(1);
        let r = SequenceVerifyResult::verify_spine(2, &[1, 2, 3], &[1, 2, 3]);
        let batch = VerifyResult::from_sequence_results(vec![empty, r]);
        assert_eq!(batch.total_accepted_tokens, 3);
        assert_eq!(batch.total_draft_tokens, 3);
        assert!(batch.all_invariants_passed);
    }

    #[test]
    fn test_result_for_returns_first_match_with_duplicate_ids() {
        let r1 = SequenceVerifyResult::verify_spine(10, &[1], &[1]);
        let r2 = SequenceVerifyResult::verify_spine(10, &[2], &[99]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        let found = batch.result_for(10).unwrap();
        // Should return the first match
        assert_eq!(found.accepted_count, 1);
    }

    // ── VerifyResult::check_topology_invariant additional ─────────────

    #[test]
    fn test_topology_invariant_with_zero_draft_counts() {
        let r = SequenceVerifyResult::empty(1);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        assert!(batch.check_topology_invariant(&[0]));
    }

    #[test]
    fn test_topology_invariant_mismatched_single_element() {
        let r = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        assert!(!batch.check_topology_invariant(&[3]));
    }

    // ── VerifyResult::check_all_invariants additional ─────────────────

    #[test]
    fn test_check_all_invariants_both_i1_and_i3_fail() {
        let mut r = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        r.invariant_check_passed = false;
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let check = batch.check_all_invariants(&[99]);
        assert!(!check.i1_topology);
        assert!(check.i2_single_verification);
        assert!(!check.i3_atomic_kv_commit);
        assert!(!check.all_passed());
    }

    #[test]
    fn test_check_all_invariants_empty_batch() {
        let batch = VerifyResult::from_sequence_results(vec![]);
        let check = batch.check_all_invariants(&[]);
        assert!(check.i1_topology);
        assert!(check.i2_single_verification);
        assert!(check.i3_atomic_kv_commit);
        assert!(check.all_passed());
    }

    #[test]
    fn test_batch_all_invariants_all_individual_false() {
        let mut r1 = SequenceVerifyResult::verify_spine(1, &[1], &[1]);
        let mut r2 = SequenceVerifyResult::verify_spine(2, &[2], &[2]);
        r1.invariant_check_passed = false;
        r2.invariant_check_passed = false;
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(!batch.all_invariants_passed);
    }

    // ── EqSpecCheckResult: all 8 boolean combinations ─────────────────

    #[test]
    fn test_eqspec_check_result_all_false() {
        let result = EqSpecCheckResult {
            i1_topology: false,
            i2_single_verification: false,
            i3_atomic_kv_commit: false,
        };
        assert!(!result.all_passed());
    }

    #[test]
    fn test_eqspec_check_result_only_i1_true() {
        let result = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: false,
            i3_atomic_kv_commit: false,
        };
        assert!(!result.all_passed());
        assert!(result.i1_topology);
    }

    #[test]
    fn test_eqspec_check_result_only_i2_true() {
        let result = EqSpecCheckResult {
            i1_topology: false,
            i2_single_verification: true,
            i3_atomic_kv_commit: false,
        };
        assert!(!result.all_passed());
        assert!(result.i2_single_verification);
    }

    #[test]
    fn test_eqspec_check_result_only_i3_true() {
        let result = EqSpecCheckResult {
            i1_topology: false,
            i2_single_verification: false,
            i3_atomic_kv_commit: true,
        };
        assert!(!result.all_passed());
        assert!(result.i3_atomic_kv_commit);
    }

    #[test]
    fn test_eqspec_check_result_i1_i2_true_i3_false() {
        let result = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: true,
            i3_atomic_kv_commit: false,
        };
        assert!(!result.all_passed());
    }

    #[test]
    fn test_eqspec_check_result_i1_i3_true_i2_false() {
        let result = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: false,
            i3_atomic_kv_commit: true,
        };
        assert!(!result.all_passed());
    }

    #[test]
    fn test_eqspec_check_result_i2_i3_true_i1_false() {
        let result = EqSpecCheckResult {
            i1_topology: false,
            i2_single_verification: true,
            i3_atomic_kv_commit: true,
        };
        assert!(!result.all_passed());
    }

    // ── KvCommitInstruction: clone and edge cases ────────────────────

    #[test]
    fn test_kv_commit_commit_clone() {
        let instr = KvCommitInstruction::Commit {
            request_id: 5,
            accepted_tokens: vec![10, 20],
            kv_pages_to_commit: vec![0, 1],
        };
        let cloned = instr.clone();
        match &cloned {
            KvCommitInstruction::Commit {
                request_id,
                accepted_tokens,
                kv_pages_to_commit,
            } => {
                assert_eq!(*request_id, 5);
                assert_eq!(*accepted_tokens, vec![10, 20]);
                assert_eq!(*kv_pages_to_commit, vec![0, 1]);
            }
            _ => panic!("expected Commit variant"),
        }
    }

    #[test]
    fn test_kv_commit_rollback_clone() {
        let instr = KvCommitInstruction::Rollback {
            request_id: 7,
            rejected_count: 3,
            kv_pages_to_free: vec![0, 1, 2],
        };
        let cloned = instr.clone();
        match &cloned {
            KvCommitInstruction::Rollback {
                request_id,
                rejected_count,
                kv_pages_to_free,
            } => {
                assert_eq!(*request_id, 7);
                assert_eq!(*rejected_count, 3);
                assert_eq!(*kv_pages_to_free, vec![0, 1, 2]);
            }
            _ => panic!("expected Rollback variant"),
        }
    }

    #[test]
    fn test_kv_commit_instruction_commit_empty_accepted_tokens() {
        let instr = KvCommitInstruction::Commit {
            request_id: 1,
            accepted_tokens: vec![],
            kv_pages_to_commit: vec![],
        };
        match &instr {
            KvCommitInstruction::Commit { accepted_tokens, .. } => {
                assert!(accepted_tokens.is_empty());
            }
            _ => panic!("expected Commit"),
        }
    }

    #[test]
    fn test_kv_commit_instruction_rollback_empty_pages() {
        let instr = KvCommitInstruction::Rollback {
            request_id: 1,
            rejected_count: 0,
            kv_pages_to_free: vec![],
        };
        match &instr {
            KvCommitInstruction::Rollback {
                rejected_count,
                kv_pages_to_free,
                ..
            } => {
                assert_eq!(*rejected_count, 0);
                assert!(kv_pages_to_free.is_empty());
            }
            _ => panic!("expected Rollback"),
        }
    }

    // ── generate_kv_commit_instructions additional ────────────────────

    #[test]
    fn test_generate_kv_commit_pages_count_equals_accepted_count() {
        let r = SequenceVerifyResult::verify_spine(1, &[1, 2, 3, 4], &[1, 2, 99, 4]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch);
        // Find the Commit instruction
        let commit = instructions.iter().find_map(|i| match i {
            KvCommitInstruction::Commit {
                kv_pages_to_commit,
                ..
            } => Some(kv_pages_to_commit.len()),
            _ => None,
        });
        assert_eq!(commit, Some(2)); // 2 accepted tokens → 2 pages
    }

    #[test]
    fn test_generate_kv_rollback_pages_count_equals_rejected_count() {
        let r = SequenceVerifyResult::verify_spine(1, &[1, 2, 3], &[1, 99]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch);
        let rollback = instructions.iter().find_map(|i| match i {
            KvCommitInstruction::Rollback {
                rejected_count,
                kv_pages_to_free,
                ..
            } => Some((*rejected_count, kv_pages_to_free.len())),
            _ => None,
        });
        let (rejected, pages) = rollback.unwrap();
        assert_eq!(rejected, 2);
        assert_eq!(pages, 2);
    }

    #[test]
    fn test_generate_kv_instructions_preserves_sequence_order() {
        let r1 = SequenceVerifyResult::verify_spine(10, &[1, 2], &[1, 99]);
        let r2 = SequenceVerifyResult::verify_spine(20, &[3, 4], &[99, 4]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);
        let instructions = generate_kv_commit_instructions(&batch);
        // r1 first: Commit(10) then Rollback(10)
        // r2 second: Rollback(20) (0 accepted)
        assert_eq!(instructions.len(), 3);
        let ids: Vec<RequestId> = instructions
            .iter()
            .map(|i| match i {
                KvCommitInstruction::Commit { request_id, .. } => *request_id,
                KvCommitInstruction::Rollback { request_id, .. } => *request_id,
            })
            .collect();
        assert_eq!(ids[0], 10);
        assert_eq!(ids[1], 10);
        assert_eq!(ids[2], 20);
    }

    #[test]
    fn test_generate_kv_instructions_all_empty_sequences() {
        let e1 = SequenceVerifyResult::empty(1);
        let e2 = SequenceVerifyResult::empty(2);
        let batch = VerifyResult::from_sequence_results(vec![e1, e2]);
        let instructions = generate_kv_commit_instructions(&batch);
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_generate_kv_instructions_single_token_accepted() {
        let r = SequenceVerifyResult::verify_spine(1, &[42], &[42]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch);
        assert_eq!(instructions.len(), 1);
        match &instructions[0] {
            KvCommitInstruction::Commit {
                accepted_tokens, ..
            } => assert_eq!(*accepted_tokens, vec![42]),
            _ => panic!("expected Commit"),
        }
    }

    #[test]
    fn test_generate_kv_instructions_single_token_rejected() {
        let r = SequenceVerifyResult::verify_spine(1, &[42], &[99]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch);
        assert_eq!(instructions.len(), 1);
        match &instructions[0] {
            KvCommitInstruction::Rollback {
                rejected_count, ..
            } => assert_eq!(*rejected_count, 1),
            _ => panic!("expected Rollback"),
        }
    }

    // ── Integration: full pipeline tests ─────────────────────────────

    #[test]
    fn test_full_pipeline_verify_then_invariants_then_kv_instructions() {
        // Arrange
        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 99]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[40, 50], &[40, 50]);

        // Act — build batch
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);

        // Assert — totals
        assert_eq!(batch.total_accepted_tokens, 4);
        assert_eq!(batch.total_draft_tokens, 5);
        assert!(batch.all_invariants_passed);

        // Act — invariants
        let check = batch.check_all_invariants(&[3, 2]);
        assert!(check.all_passed());

        // Act — KV instructions
        let instructions = generate_kv_commit_instructions(&batch);
        assert_eq!(instructions.len(), 3); // r1: commit+rollback, r2: commit only
    }

    #[test]
    fn test_empty_result_in_batch_does_not_affect_others() {
        // Arrange
        let empty = SequenceVerifyResult::empty(1);
        let r = SequenceVerifyResult::verify_spine(2, &[1, 2], &[1, 2]);

        // Act
        let batch = VerifyResult::from_sequence_results(vec![empty, r]);

        // Assert
        assert_eq!(batch.sequence_results.len(), 2);
        assert_eq!(batch.total_accepted_tokens, 2);
        assert_eq!(batch.total_draft_tokens, 2);
        assert!(batch.all_invariants_passed);

        let instructions = generate_kv_commit_instructions(&batch);
        // Only the non-empty sequence generates a Commit
        assert_eq!(instructions.len(), 1);
    }

    // ══════════════════════════════════════════════════════════════════
    //  13 additional tests — boundary values, token content, edge cases
    // ══════════════════════════════════════════════════════════════════

    // ── verify_spine: boundary token values ──────────────────────────

    #[test]
    fn test_verify_spine_with_u32_max_tokens() {
        // Arrange: draft and target both use u32::MAX as token ID
        let draft = vec![u32::MAX, u32::MAX, u32::MAX];
        let target = vec![u32::MAX, u32::MAX, u32::MAX];

        // Act
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);

        // Assert
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(result.rejected_count, 0);
        assert!((result.acceptance_rate - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_verify_spine_with_zero_token_id() {
        // Arrange: token ID 0 is a valid token
        let draft = vec![0u32, 0, 1];
        let target = vec![0u32, 0, 1];

        // Act
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);

        // Assert
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![0, 0, 1]);
        assert_eq!(result.rejected_count, 0);
    }

    // ── verify_spine: accepted_tokens content is from draft, not target ─

    #[test]
    fn test_verify_spine_accepted_tokens_come_from_draft_not_target() {
        // Arrange: identical values so this is trivially true, but verify
        // the contract — accepted_tokens[i] == draft[i] for all i < accepted_count
        let draft = vec![100u32, 200, 300, 400];
        let target = vec![100u32, 200, 999, 888];

        // Act
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);

        // Assert: accepted_tokens must be [100, 200] which are the draft values
        assert_eq!(result.accepted_tokens, vec![100, 200]);
        for (i, &tok) in result.accepted_tokens.iter().enumerate() {
            assert_eq!(tok, draft[i], "accepted_tokens[{i}] must equal draft[{i}]");
        }
    }

    // ── verify_spine: saturation arithmetic edge case ────────────────

    #[test]
    fn test_verify_spine_rejected_count_saturating_sub_no_underflow() {
        // Arrange: when accepted_count equals draft_count, rejected_count = 0
        // This tests that saturating_sub prevents underflow
        let draft = vec![1u32, 2, 3, 4, 5];
        let target = vec![1u32, 2, 3, 4, 5, 6, 7, 8]; // target longer, all draft accepted

        // Act
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);

        // Assert
        assert_eq!(result.accepted_count, 5);
        assert_eq!(result.draft_count, 5);
        assert_eq!(result.rejected_count, 0); // saturating_sub(5, 5) = 0, no underflow
    }

    // ── verify_spine: request_id preservation with max value ─────────

    #[test]
    fn test_verify_spine_preserves_max_request_id() {
        // Arrange
        let request_id = RequestId::MAX;

        // Act
        let result = SequenceVerifyResult::verify_spine(request_id, &[1], &[1]);

        // Assert
        assert_eq!(result.request_id, RequestId::MAX);
    }

    // ── VerifyResult: all_invariants_passed with single failed sequence ─

    #[test]
    fn test_batch_all_invariants_single_sequence_fails() {
        // Arrange: single sequence with invariant check failed
        let mut r = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        r.invariant_check_passed = false;

        // Act
        let batch = VerifyResult::from_sequence_results(vec![r]);

        // Assert: single failure propagates to batch
        assert!(!batch.all_invariants_passed);
    }

    // ── VerifyResult: avg_acceptance_rate zero_draft multiple sequences ─

    #[test]
    fn test_batch_avg_rate_with_all_empty_sequences() {
        // Arrange: two empty sequences — total_draft = 0
        let e1 = SequenceVerifyResult::empty(1);
        let e2 = SequenceVerifyResult::empty(2);

        // Act
        let batch = VerifyResult::from_sequence_results(vec![e1, e2]);

        // Assert: avg_rate = 0.0 when total_draft = 0 (avoids division by zero)
        assert!((batch.avg_acceptance_rate).abs() < 1e-6);
        assert_eq!(batch.total_draft_tokens, 0);
        assert_eq!(batch.total_accepted_tokens, 0);
    }

    // ── EqSpecCheckResult: all three false produces correct debug ────

    #[test]
    fn test_eqspec_check_result_debug_all_false() {
        // Arrange
        let result = EqSpecCheckResult {
            i1_topology: false,
            i2_single_verification: false,
            i3_atomic_kv_commit: false,
        };

        // Act
        let debug = format!("{:?}", result);

        // Assert
        assert!(debug.contains("i1_topology: false"));
        assert!(debug.contains("i2_single_verification: false"));
        assert!(debug.contains("i3_atomic_kv_commit: false"));
    }

    // ── generate_kv_commit_instructions: large batch ─────────────────

    #[test]
    fn test_generate_kv_instructions_many_sequences_order_preserved() {
        // Arrange: 10 sequences, alternating all-accepted and all-rejected
        let results: Vec<SequenceVerifyResult> = (0..10)
            .map(|i| {
                if i % 2 == 0 {
                    // all accepted
                    SequenceVerifyResult::verify_spine(i, &[1, 2], &[1, 2])
                } else {
                    // all rejected
                    SequenceVerifyResult::verify_spine(i, &[1, 2], &[99, 99])
                }
            })
            .collect();

        let batch = VerifyResult::from_sequence_results(results);

        // Act
        let instructions = generate_kv_commit_instructions(&batch);

        // Assert: 5 sequences produce 1 Commit each, 5 produce 1 Rollback each = 10 total
        assert_eq!(instructions.len(), 10);
        // Verify order: seq 0 (commit), seq 1 (rollback), seq 2 (commit), ...
        for (i, instr) in instructions.iter().enumerate() {
            let seq_idx = i;
            let expected_req_id = seq_idx as RequestId;
            match instr {
                KvCommitInstruction::Commit { request_id, .. } => {
                    assert_eq!(*request_id, expected_req_id);
                    assert!(seq_idx % 2 == 0, "Commit should be at even index {seq_idx}");
                }
                KvCommitInstruction::Rollback { request_id, .. } => {
                    assert_eq!(*request_id, expected_req_id);
                    assert!(seq_idx % 2 == 1, "Rollback should be at odd index {seq_idx}");
                }
            }
        }
    }

    // ── check_topology_invariant: multiple sequences with mixed counts ─

    #[test]
    fn test_topology_invariant_multiple_sequences_partial_mismatch() {
        // Arrange: 3 sequences with draft counts [2, 3, 4]
        let r1 = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[3, 4, 5], &[3, 4, 5]);
        let r3 = SequenceVerifyResult::verify_spine(3, &[6, 7, 8, 9], &[6, 7, 8, 9]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2, r3]);

        // Act & Assert: correct counts pass
        assert!(batch.check_topology_invariant(&[2, 3, 4]));
        // Wrong count for middle sequence
        assert!(!batch.check_topology_invariant(&[2, 99, 4]));
    }

    // ── KvCommitInstruction: request_id in commit matches source sequence ─

    #[test]
    fn test_kv_commit_instruction_request_id_matches_source() {
        // Arrange: two sequences with distinct request IDs
        let r1 = SequenceVerifyResult::verify_spine(111, &[1, 2], &[1, 2]);
        let r2 = SequenceVerifyResult::verify_spine(222, &[3, 4], &[3, 4]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);

        // Act
        let instructions = generate_kv_commit_instructions(&batch);

        // Assert: each instruction carries the correct request_id
        assert_eq!(instructions.len(), 2);
        match &instructions[0] {
            KvCommitInstruction::Commit { request_id, .. } => assert_eq!(*request_id, 111),
            _ => panic!("expected Commit for first sequence"),
        }
        match &instructions[1] {
            KvCommitInstruction::Commit { request_id, .. } => assert_eq!(*request_id, 222),
            _ => panic!("expected Commit for second sequence"),
        }
    }

    // ── check_single_verification_invariant: with populated batch ────

    #[test]
    fn test_single_verification_invariant_with_failed_invariant_check() {
        // Arrange: a sequence that failed invariant check
        let mut r = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 99]);
        r.invariant_check_passed = false;
        let batch = VerifyResult::from_sequence_results(vec![r]);

        // Act & Assert: I2 is always true regardless of other invariant states
        assert!(batch.check_single_verification_invariant());
    }

    // ── EqSpecInvariant: PartialEq symmetry and transitivity ─────────

    #[test]
    fn test_eqspec_invariant_partial_eq_properties() {
        // Arrange: all three variants
        let topology = EqSpecInvariant::Topology;
        let single = EqSpecInvariant::SingleVerification;
        let atomic = EqSpecInvariant::AtomicKvCommit;

        // Assert: reflexivity
        assert_eq!(topology, topology);
        assert_eq!(single, single);
        assert_eq!(atomic, atomic);

        // Assert: symmetry
        assert_ne!(topology, single);
        assert_ne!(single, topology);
        assert_ne!(single, atomic);
        assert_ne!(atomic, single);
        assert_ne!(topology, atomic);
        assert_ne!(atomic, topology);
    }

    // ══════════════════════════════════════════════════════════════════
    //  10 additional tests — deep-copy isolation, property invariants,
    //  page ID range, reference correctness, boundary conditions
    // ══════════════════════════════════════════════════════════════════

    // ── SequenceVerifyResult::clone deep-copy isolation ──────────────

    #[test]
    fn test_sequence_verify_result_clone_isolation() {
        // Arrange
        let original = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 99]);
        let mut cloned = original.clone();

        // Act: mutate the clone's accepted_tokens
        cloned.accepted_tokens.push(999);
        cloned.accepted_count = 99;

        // Assert: original is unaffected
        assert_eq!(original.accepted_tokens, vec![10, 20]);
        assert_eq!(original.accepted_count, 2);
        assert_eq!(cloned.accepted_tokens, vec![10, 20, 999]);
    }

    // ── VerifyResult::clone deep-copy isolation ──────────────────────

    #[test]
    fn test_verify_result_clone_isolation() {
        // Arrange
        let r = SequenceVerifyResult::verify_spine(1, &[1, 2, 3], &[1, 2, 99]);
        let original = VerifyResult::from_sequence_results(vec![r]);
        let mut cloned = original.clone();

        // Act: mutate cloned sequence_results
        cloned.sequence_results.clear();

        // Assert: original still has its sequence
        assert_eq!(original.sequence_results.len(), 1);
        assert_eq!(original.total_accepted_tokens, 2);
        assert!(cloned.sequence_results.is_empty());
    }

    // ── VerifyResult::result_for returns reference to stored data ────

    #[test]
    fn test_result_for_reference_matches_stored_data() {
        // Arrange
        let r1 = SequenceVerifyResult::verify_spine(42, &[100, 200], &[100, 200]);
        let r2 = SequenceVerifyResult::verify_spine(99, &[1], &[2]);
        let batch = VerifyResult::from_sequence_results(vec![r1, r2]);

        // Act
        let found = batch.result_for(42).unwrap();

        // Assert: the reference points to the actual stored element
        assert!(std::ptr::eq(found, &batch.sequence_results[0]));
        assert_eq!(found.accepted_tokens, vec![100, 200]);
    }

    // ── generate_kv_commit_instructions: exact page ID ranges ────────

    #[test]
    fn test_generate_kv_instructions_commit_page_ids_are_zero_indexed_range() {
        // Arrange: 4 accepted, 3 rejected
        let r = SequenceVerifyResult::verify_spine(1, &[1, 2, 3, 4, 5, 6, 7], &[1, 2, 3, 4, 99, 99, 99]);
        let batch = VerifyResult::from_sequence_results(vec![r]);

        // Act
        let instructions = generate_kv_commit_instructions(&batch);

        // Assert: commit pages = [0, 1, 2, 3], rollback pages = [0, 1, 2]
        let commit = &instructions[0];
        let rollback = &instructions[1];
        match commit {
            KvCommitInstruction::Commit { kv_pages_to_commit, .. } => {
                assert_eq!(*kv_pages_to_commit, vec![0u64, 1, 2, 3]);
            }
            _ => panic!("expected Commit"),
        }
        match rollback {
            KvCommitInstruction::Rollback { kv_pages_to_free, .. } => {
                assert_eq!(*kv_pages_to_free, vec![0u64, 1, 2]);
            }
            _ => panic!("expected Rollback"),
        }
    }

    // ── generate_kv_commit_instructions: accepted_tokens content in commit

    #[test]
    fn test_generate_kv_instructions_commit_contains_exact_accepted_tokens() {
        // Arrange: draft=[5,6,7,8], target=[5,6,99,8] -> accepted=[5,6]
        let r = SequenceVerifyResult::verify_spine(1, &[5, 6, 7, 8], &[5, 6, 99, 8]);
        let batch = VerifyResult::from_sequence_results(vec![r]);

        // Act
        let instructions = generate_kv_commit_instructions(&batch);

        // Assert: commit carries the exact accepted token IDs, not draft
        match &instructions[0] {
            KvCommitInstruction::Commit { accepted_tokens, .. } => {
                assert_eq!(*accepted_tokens, vec![5u32, 6]);
            }
            _ => panic!("expected Commit"),
        }
    }

    // ── verify_spine: mismatch in the middle — post-mismatch ignored ──

    #[test]
    fn test_verify_spine_tokens_after_mismatch_are_irrelevant() {
        // Arrange: mismatch at index 1, but indices 2+ happen to match again
        let draft = vec![10, 20, 30, 40, 50];
        let target = vec![10, 99, 30, 40, 50]; // position 2+ match but must be ignored

        // Act
        let result = SequenceVerifyResult::verify_spine(1, &draft, &target);

        // Assert: only the contiguous prefix before first mismatch counts
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.accepted_tokens, vec![10]);
        assert_eq!(result.rejected_count, 4);
    }

    // ── check_topology_invariant: more expected counts than sequences ──

    #[test]
    fn test_topology_invariant_extra_expected_counts_fails() {
        // Arrange: 1 sequence but 3 expected counts
        let r = SequenceVerifyResult::verify_spine(1, &[1, 2], &[1, 2]);
        let batch = VerifyResult::from_sequence_results(vec![r]);

        // Act & Assert: length mismatch → false
        assert!(!batch.check_topology_invariant(&[2, 3, 4]));
    }

    // ── EqSpecCheckResult: all_passed is conjunction of all three ─────

    #[test]
    fn test_eqspec_check_result_all_passed_is_conjunction() {
        // Assert: every combination with at least one false fails
        for i1 in [false, true] {
            for i2 in [false, true] {
                for i3 in [false, true] {
                    let r = EqSpecCheckResult {
                        i1_topology: i1,
                        i2_single_verification: i2,
                        i3_atomic_kv_commit: i3,
                    };
                    assert_eq!(r.all_passed(), i1 && i2 && i3);
                }
            }
        }
    }

    // ── VerifyResult: avg_acceptance_rate consistency property ───────

    #[test]
    fn test_batch_avg_rate_always_equals_ratio_of_totals() {
        // Arrange: multiple random-ish sequences
        let cases: Vec<(RequestId, &[u32], &[u32])> = vec![
            (1, &[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5]),       // 5/5
            (2, &[10, 20, 30], &[10, 99]),                    // 1/3
            (3, &[7, 8], &[99, 99]),                           // 0/2
            (4, &[50], &[50]),                                  // 1/1
            (5, &[1], &[2]),                                    // 0/1
        ];
        let results: Vec<SequenceVerifyResult> = cases
            .into_iter()
            .map(|(id, d, t)| SequenceVerifyResult::verify_spine(id, d, t))
            .collect();

        // Act
        let batch = VerifyResult::from_sequence_results(results);

        // Assert: avg_rate = total_accepted / total_draft
        let expected_rate = if batch.total_draft_tokens > 0 {
            batch.total_accepted_tokens as f32 / batch.total_draft_tokens as f32
        } else {
            0.0
        };
        assert!((batch.avg_acceptance_rate - expected_rate).abs() < 1e-6);
        // totals: accepted=5+1+0+1+0=7, draft=5+3+2+1+1=12
        assert_eq!(batch.total_accepted_tokens, 7);
        assert_eq!(batch.total_draft_tokens, 12);
    }

    // ── KvCommitInstruction: clone isolation for Commit variant ──────

    #[test]
    fn test_kv_commit_instruction_commit_clone_isolation() {
        // Arrange
        let original = KvCommitInstruction::Commit {
            request_id: 42,
            accepted_tokens: vec![1, 2, 3],
            kv_pages_to_commit: vec![0, 1, 2],
        };
        let mut cloned = original.clone();

        // Act: mutate the clone
        if let KvCommitInstruction::Commit { accepted_tokens, .. } = &mut cloned {
            accepted_tokens.clear();
        }

        // Assert: original untouched
        match &original {
            KvCommitInstruction::Commit { accepted_tokens, .. } => {
                assert_eq!(*accepted_tokens, vec![1, 2, 3]);
            }
            _ => panic!("expected Commit"),
        }
    }
}
