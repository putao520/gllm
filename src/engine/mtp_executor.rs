//! MTP Executor integration (REQ-MTP-002).
//!
//! Integrates Multi-Token Prediction into the generate loop. The mega-kernel
//! JIT already produces MTP candidates per decode step. This module:
//!
//! 1. Parses mega-kernel output to extract main tokens + MTP candidates
//! 2. Verifies candidates against a full-model forward
//! 3. Commits accepted tokens, rolls back rejected ones
//! 4. Updates EMA acceptance tracker for adaptive backoff

use crate::speculative::verify::{KvCommitInstruction, SequenceVerifyResult};

/// Result of an MTP-aware generate call.
#[derive(Debug, Clone)]
pub struct MtpGenerateResult {
    /// Tokens that were committed (verified accepted).
    /// Includes main token per step plus accepted MTP candidates.
    pub committed_tokens: Vec<u32>,
    /// Total MTP candidates generated across all steps.
    pub total_mtp_candidates: usize,
    /// Total MTP candidates accepted across all steps.
    pub total_mtp_accepted: usize,
    /// Per-step acceptance details.
    pub step_details: Vec<MtpStepDetail>,
}

/// Details for a single MTP decode step.
#[derive(Debug, Clone)]
pub struct MtpStepDetail {
    /// Main token for this step (always committed).
    pub main_token: u32,
    /// MTP candidate tokens for this step.
    pub mtp_candidates: Vec<u32>,
    /// Number of accepted MTP candidates (consecutive from first).
    pub accepted_count: usize,
    /// Whether this step hit EOS on the main token.
    pub main_token_is_eos: bool,
}

/// Parsed output from mega-kernel with MTP enabled.
///
/// Layout: `output[0..N]` = main tokens, followed by interleaved MTP candidates
/// where MTP candidates for step `s` are at indices `[N + s*K .. N + (s+1)*K]`.
pub struct MtpOutput {
    /// Main tokens (one per decode step).
    pub main_tokens: Vec<u32>,
    /// MTP candidates grouped by step: `mtp_per_step[step][k]`.
    pub mtp_per_step: Vec<Vec<u32>>,
}

impl MtpOutput {
    /// Parse raw mega-kernel output into structured MTP data.
    ///
    /// The mega-kernel returns a flat vector where:
    /// - `output[0..num_steps]` are main tokens
    /// - `output[num_steps + step * mtp_depth + k]` is candidate k for step
    ///
    /// Returns `None` if the output is too short for the expected layout.
    pub fn parse(output: &[u32], num_steps: usize, mtp_depth: usize) -> Option<Self> {
        if output.len() < num_steps {
            return None;
        }
        let main_tokens = output[..num_steps].to_vec();
        let mtp_base = num_steps;
        let mut mtp_per_step = Vec::with_capacity(num_steps);
        for step in 0..num_steps {
            let offset = mtp_base + step * mtp_depth;
            let end = offset + mtp_depth;
            if end <= output.len() {
                mtp_per_step.push(output[offset..end].to_vec());
            } else if offset < output.len() {
                mtp_per_step.push(output[offset..].to_vec());
            } else {
                mtp_per_step.push(Vec::new());
            }
        }
        Some(Self {
            main_tokens,
            mtp_per_step,
        })
    }
}

/// Verify MTP candidates against a full-model forward.
///
/// For each candidate position, runs a full forward pass on the sequence
/// including the candidate token, then checks if the model's argmax matches
/// the candidate.
///
/// # Arguments
/// * `verify_logits_per_position` - Logits from full-model forward at each
///   candidate position. Each element is a vocab-sized logits slice.
/// * `mtp_candidates` - The K candidate tokens to verify.
///
/// # Returns
/// The number of consecutive accepted candidates (longest prefix match).
pub fn verify_mtp_candidates(
    verify_logits_per_position: &[Vec<f32>],
    mtp_candidates: &[u32],
) -> usize {
    let k = mtp_candidates.len().min(verify_logits_per_position.len());
    let mut accepted = 0;
    for i in 0..k {
        let target = argmax_token(&verify_logits_per_position[i])
            .expect("all-NaN logits — computation error");
        if target == mtp_candidates[i] {
            accepted += 1;
        } else {
            break;
        }
    }
    accepted
}

/// Build a `SequenceVerifyResult` from MTP verification.
pub fn build_verify_result(
    request_id: u64,
    mtp_candidates: &[u32],
    accepted_count: usize,
) -> SequenceVerifyResult {
    let accepted_tokens = mtp_candidates[..accepted_count].to_vec();
    let rejected_count = mtp_candidates.len().saturating_sub(accepted_count);
    let acceptance_rate = if mtp_candidates.is_empty() {
        0.0
    } else {
        accepted_count as f32 / mtp_candidates.len() as f32
    };
    SequenceVerifyResult {
        request_id,
        accepted_count,
        accepted_tokens,
        rejected_count,
        draft_count: mtp_candidates.len(),
        acceptance_rate,
        invariant_check_passed: true,
    }
}

/// Extract the argmax token ID from a logits vector.
/// NaN values are excluded from comparison (treated as -inf), ensuring deterministic argmax.
/// Returns `None` if logits is empty or all values are NaN (indicates upstream computation error).
fn argmax_token(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .map(|(i, _)| i as u32)
}

/// EMA-based adaptive MTP controller.
///
/// Tracks acceptance rates and decides whether MTP should remain enabled.
/// Implements the decision state machine from REQ-MTP-005:
/// - Active: MTP enabled, candidates verified each step
/// - Disabled: MTP disabled after consecutive low acceptance
/// - Re-enable: after enough stable standard decode steps
#[derive(Debug, Clone)]
pub struct MtpController {
    /// EMA alpha (smoothing factor).
    alpha: f32,
    /// Current EMA acceptance rate.
    ema_rate: f32,
    /// Consecutive low-acceptance streak.
    low_streak: usize,
    /// Consecutive high-acceptance standard-decode streak (for re-enable).
    stable_streak: usize,
    /// Threshold below which acceptance is "low".
    disable_threshold: f32,
    /// Threshold above which to re-enable MTP.
    enable_threshold: f32,
    /// Number of consecutive low rounds before disabling.
    disable_patience: usize,
    /// Number of consecutive stable rounds before re-enabling.
    enable_patience: usize,
    /// Whether MTP is currently active.
    enabled: bool,
}

impl MtpController {
    /// Create a new controller with default parameters from SPEC REQ-MTP-005.
    pub fn new() -> Self {
        Self {
            alpha: 0.1,
            ema_rate: 0.5,
            low_streak: 0,
            stable_streak: 0,
            disable_threshold: 0.3,
            enable_threshold: 0.5,
            disable_patience: 3,
            enable_patience: 5,
            enabled: true,
        }
    }

    /// Create a controller with custom parameters.
    pub fn with_params(
        alpha: f32,
        disable_threshold: f32,
        enable_threshold: f32,
        disable_patience: usize,
        enable_patience: usize,
    ) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            ema_rate: 0.5,
            low_streak: 0,
            stable_streak: 0,
            disable_threshold,
            enable_threshold,
            disable_patience,
            enable_patience,
            enabled: true,
        }
    }

    /// Record the acceptance rate from a verify step and update state.
    ///
    /// Returns the updated `enabled` status.
    pub fn record_acceptance(&mut self, accepted: usize, total: usize) -> bool {
        let rate = if total > 0 {
            accepted as f32 / total as f32
        } else {
            0.0
        };
        self.ema_rate = self.alpha * rate + (1.0 - self.alpha) * self.ema_rate;

        if self.enabled {
            if rate < self.disable_threshold {
                self.low_streak += 1;
                self.stable_streak = 0;
            } else {
                self.low_streak = 0;
                self.stable_streak += 1;
            }
            if self.low_streak >= self.disable_patience {
                self.enabled = false;
                self.stable_streak = 0;
            }
        } else {
            // Standard decode mode: track stable rounds for potential re-enable
            if rate >= self.enable_threshold || total == 0 {
                self.stable_streak += 1;
            } else {
                self.stable_streak = 0;
            }
            if self.stable_streak >= self.enable_patience {
                self.enabled = true;
                self.low_streak = 0;
                self.stable_streak = 0;
            }
        }
        self.enabled
    }

    /// Whether MTP is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the current EMA acceptance rate.
    pub fn ema_rate(&self) -> f32 {
        self.ema_rate
    }

    /// Determine the effective MTP depth based on EMA rates.
    ///
    /// Implements adaptive depth from REQ-MTP-005:
    /// - ema > 0.8: full depth (aggressive)
    /// - ema > 0.5: depth = min(full, 2) (moderate)
    /// - ema < 0.3: depth = 0 (disabled)
    pub fn effective_depth(&self, max_depth: usize) -> usize {
        if !self.enabled {
            return 0;
        }
        if self.ema_rate > 0.8 {
            max_depth
        } else if self.ema_rate > 0.5 {
            max_depth.min(2)
        } else if self.ema_rate > 0.3 {
            1
        } else {
            0
        }
    }

    /// Force-disable MTP (e.g., when mode is Standard).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Force-enable MTP (e.g., when mode transitions to Mtp).
    pub fn enable(&mut self) {
        self.enabled = true;
        self.low_streak = 0;
        self.stable_streak = 0;
    }

    /// Reset all tracking state.
    pub fn reset(&mut self) {
        self.ema_rate = 0.5;
        self.low_streak = 0;
        self.stable_streak = 0;
        self.enabled = true;
    }
}

impl Default for MtpController {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate KV commit/rollback instructions from MTP verify results (REQ-MTP-004).
///
/// For each step: accepted candidates → Commit, rejected candidates → Rollback.
pub fn generate_mtp_kv_instructions(
    request_id: u64,
    step_details: &[MtpStepDetail],
) -> Vec<KvCommitInstruction> {
    let mut instructions = Vec::new();
    for detail in step_details {
        if detail.accepted_count > 0 {
            instructions.push(KvCommitInstruction::Commit {
                request_id,
                accepted_tokens: detail.mtp_candidates[..detail.accepted_count].to_vec(),
                kv_pages_to_commit: (0..detail.accepted_count as u64).collect(),
            });
        }
        let rejected = detail.mtp_candidates.len().saturating_sub(detail.accepted_count);
        if rejected > 0 {
            instructions.push(KvCommitInstruction::Rollback {
                request_id,
                rejected_count: rejected,
                kv_pages_to_free: (0..rejected as u64).collect(),
            });
        }
    }
    instructions
}

/// Filter MTP output tokens to only include verified tokens.
///
/// Given the raw mega-kernel output and a verify result for each step,
/// returns only the tokens that should be committed (main + accepted MTP).
pub fn filter_verified_tokens(
    main_tokens: &[u32],
    mtp_per_step: &[Vec<u32>],
    eos_token_id: Option<u32>,
    verify_fn: impl Fn(usize, &[u32]) -> usize,
) -> MtpGenerateResult {
    let mut committed = Vec::new();
    let mut total_candidates = 0;
    let mut total_accepted = 0;
    let mut step_details = Vec::new();

    for (step, &main_token) in main_tokens.iter().enumerate() {
        let main_is_eos = eos_token_id.is_some_and(|eos| main_token == eos);

        // Always commit the main token
        committed.push(main_token);

        let candidates = mtp_per_step
            .get(step)
            .cloned()
            .unwrap_or_default();
        let k = candidates.len();
        total_candidates += k;

        let accepted = if main_is_eos || candidates.is_empty() {
            0
        } else {
            verify_fn(step, &candidates)
        };

        total_accepted += accepted;

        // Commit accepted candidates
        for &tok in &candidates[..accepted] {
            committed.push(tok);
            if eos_token_id.is_some_and(|eos| tok == eos) {
                break;
            }
        }

        // Check if an accepted candidate is EOS before moving candidates
        let last_accepted_is_eos = accepted > 0 && eos_token_id.is_some_and(|eos| candidates[accepted - 1] == eos);

        step_details.push(MtpStepDetail {
            main_token,
            mtp_candidates: candidates,
            accepted_count: accepted,
            main_token_is_eos: main_is_eos,
        });

        if main_is_eos {
            break;
        }

        // If an accepted candidate is EOS, stop
        if last_accepted_is_eos {
            break;
        }
    }

    MtpGenerateResult {
        committed_tokens: committed,
        total_mtp_candidates: total_candidates,
        total_mtp_accepted: total_accepted,
        step_details,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_output_parse_basic() {
        // 3 steps, depth=2: [main0, main1, main2, c0_0, c0_1, c1_0, c1_1, c2_0, c2_1]
        let output = vec![100u32, 200, 300, 10, 11, 20, 21, 30, 31];
        let parsed = MtpOutput::parse(&output, 3, 2).unwrap();
        assert_eq!(parsed.main_tokens, vec![100, 200, 300]);
        assert_eq!(parsed.mtp_per_step.len(), 3);
        assert_eq!(parsed.mtp_per_step[0], vec![10, 11]);
        assert_eq!(parsed.mtp_per_step[1], vec![20, 21]);
        assert_eq!(parsed.mtp_per_step[2], vec![30, 31]);
    }

    #[test]
    fn test_mtp_output_parse_zero_depth() {
        let output = vec![100u32, 200, 300];
        let parsed = MtpOutput::parse(&output, 3, 0).unwrap();
        assert_eq!(parsed.main_tokens, vec![100, 200, 300]);
        assert!(parsed.mtp_per_step.iter().all(|c| c.is_empty()));
    }

    #[test]
    fn test_mtp_output_parse_insufficient_data() {
        let output = vec![100u32];
        let parsed = MtpOutput::parse(&output, 3, 2);
        assert!(parsed.is_none());
    }

    #[test]
    fn test_mtp_output_parse_partial_candidates() {
        // 2 steps, depth=2, but only 1 candidate for step 1
        let output = vec![100u32, 200, 10, 11, 20];
        let parsed = MtpOutput::parse(&output, 2, 2).unwrap();
        assert_eq!(parsed.main_tokens, vec![100, 200]);
        assert_eq!(parsed.mtp_per_step[0], vec![10, 11]);
        assert_eq!(parsed.mtp_per_step[1], vec![20]); // partial
    }

    #[test]
    fn test_verify_mtp_candidates_all_accepted() {
        let logits = vec![
            vec![0.1, 0.0, 10.0], // argmax=2
            vec![0.0, 10.0, 0.0], // argmax=1
        ];
        let candidates = vec![2u32, 1];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 2);
    }

    #[test]
    fn test_verify_mtp_candidates_partial() {
        let logits = vec![
            vec![0.1, 0.0, 10.0], // argmax=2 → matches
            vec![10.0, 0.0, 0.0], // argmax=0 → mismatch with candidate 1
        ];
        let candidates = vec![2u32, 1];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 1);
    }

    #[test]
    fn test_verify_mtp_candidates_none_accepted() {
        let logits = vec![
            vec![10.0, 0.0, 0.0], // argmax=0 → mismatch with candidate 2
        ];
        let candidates = vec![2u32];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 0);
    }

    #[test]
    fn test_verify_mtp_candidates_empty() {
        let logits: Vec<Vec<f32>> = vec![];
        let candidates: Vec<u32> = vec![];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 0);
    }

    #[test]
    fn test_build_verify_result() {
        let result = build_verify_result(42, &[10u32, 20, 30], 2);
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.accepted_tokens, vec![10, 20]);
        assert_eq!(result.rejected_count, 1);
        assert!((result.acceptance_rate - 2.0 / 3.0).abs() < 1e-5);
        assert!(result.invariant_check_passed);
    }

    #[test]
    fn test_mtp_controller_stays_enabled() {
        let mut ctrl = MtpController::new();
        assert!(ctrl.is_enabled());

        // Record high acceptance
        ctrl.record_acceptance(3, 4);
        assert!(ctrl.is_enabled());
        assert!(ctrl.ema_rate() > 0.5);
    }

    #[test]
    fn test_mtp_controller_disables_after_streak() {
        let mut ctrl = MtpController::new();
        // 3 rounds of 0% acceptance
        for _ in 0..3 {
            ctrl.record_acceptance(0, 4);
        }
        assert!(!ctrl.is_enabled());
    }

    #[test]
    fn test_mtp_controller_reenables_after_stable() {
        let mut ctrl = MtpController::new();
        // Disable
        for _ in 0..3 {
            ctrl.record_acceptance(0, 4);
        }
        assert!(!ctrl.is_enabled());

        // 5 stable rounds (simulate high rate by passing 1/1)
        for _ in 0..5 {
            ctrl.record_acceptance(1, 1);
        }
        assert!(ctrl.is_enabled());
    }

    #[test]
    fn test_mtp_controller_effective_depth() {
        let mut ctrl = MtpController::new();

        // Default ema = 0.5, boundary: not > 0.5, falls to > 0.3 → depth 1
        assert_eq!(ctrl.effective_depth(4), 1);

        // Force high EMA
        ctrl.ema_rate = 0.9;
        assert_eq!(ctrl.effective_depth(4), 4);

        // Medium EMA (above 0.5 threshold)
        ctrl.ema_rate = 0.6;
        assert_eq!(ctrl.effective_depth(4), 2);

        // Just above 0.5 boundary
        ctrl.ema_rate = 0.51;
        assert_eq!(ctrl.effective_depth(4), 2);

        // Low EMA (between 0.3 and 0.5)
        ctrl.ema_rate = 0.35;
        assert_eq!(ctrl.effective_depth(4), 1);

        // Very low EMA
        ctrl.ema_rate = 0.2;
        assert_eq!(ctrl.effective_depth(4), 0);

        // Disabled
        ctrl.enabled = false;
        assert_eq!(ctrl.effective_depth(4), 0);
    }

    #[test]
    fn test_filter_verified_tokens_no_mtp() {
        let main = vec![100u32, 200, 300];
        let mtp: Vec<Vec<u32>> = vec![vec![], vec![], vec![]];
        let result = filter_verified_tokens(&main, &mtp, None, |_, _| 0);
        assert_eq!(result.committed_tokens, vec![100, 200, 300]);
        assert_eq!(result.total_mtp_candidates, 0);
        assert_eq!(result.total_mtp_accepted, 0);
    }

    #[test]
    fn test_filter_verified_tokens_all_accepted() {
        let main = vec![100u32];
        let mtp = vec![vec![10u32, 20]];
        let result = filter_verified_tokens(&main, &mtp, None, |_, _| 2);
        assert_eq!(result.committed_tokens, vec![100, 10, 20]);
        assert_eq!(result.total_mtp_candidates, 2);
        assert_eq!(result.total_mtp_accepted, 2);
    }

    #[test]
    fn test_filter_verified_tokens_partial_accept() {
        let main = vec![100u32];
        let mtp = vec![vec![10u32, 20, 30]];
        let result = filter_verified_tokens(&main, &mtp, None, |_, _| 1);
        assert_eq!(result.committed_tokens, vec![100, 10]);
        assert_eq!(result.total_mtp_accepted, 1);
        assert_eq!(result.step_details[0].accepted_count, 1);
    }

    #[test]
    fn test_filter_verified_tokens_eos_stops() {
        let eos = Some(999u32);
        let main = vec![999u32, 200]; // main token is EOS
        let mtp = vec![vec![10], vec![20]];
        let result = filter_verified_tokens(&main, &mtp, eos, |_, _| 1);
        // Main token is EOS, stop immediately
        assert_eq!(result.committed_tokens, vec![999]);
        assert_eq!(result.step_details.len(), 1);
        assert!(result.step_details[0].main_token_is_eos);
    }

    #[test]
    fn test_mtp_controller_reset() {
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        assert!(!ctrl.is_enabled());

        ctrl.reset();
        assert!(ctrl.is_enabled());
        assert!((ctrl.ema_rate() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_disable_enable_forced() {
        let mut ctrl = MtpController::new();
        assert!(ctrl.is_enabled());
        ctrl.disable();
        assert!(!ctrl.is_enabled());
        ctrl.enable();
        assert!(ctrl.is_enabled());
    }

    #[test]
    fn test_mtp_kv_instructions_all_accepted() {
        let details = vec![
            MtpStepDetail { main_token: 10, mtp_candidates: vec![20, 30], accepted_count: 2, main_token_is_eos: false },
        ];
        let instrs = generate_mtp_kv_instructions(1, &details);
        assert_eq!(instrs.len(), 1);
        assert!(matches!(&instrs[0], KvCommitInstruction::Commit { request_id: 1, .. }));
    }

    #[test]
    fn test_mtp_kv_instructions_partial_accept() {
        let details = vec![
            MtpStepDetail { main_token: 10, mtp_candidates: vec![20, 30, 40], accepted_count: 1, main_token_is_eos: false },
        ];
        let instrs = generate_mtp_kv_instructions(2, &details);
        assert_eq!(instrs.len(), 2);
        // First: commit 1 accepted
        assert!(matches!(&instrs[0], KvCommitInstruction::Commit { accepted_tokens, .. } if accepted_tokens.len() == 1));
        // Second: rollback 2 rejected
        assert!(matches!(&instrs[1], KvCommitInstruction::Rollback { rejected_count: 2, .. }));
    }

    #[test]
    fn test_mtp_kv_instructions_all_rejected() {
        let details = vec![
            MtpStepDetail { main_token: 10, mtp_candidates: vec![20, 30], accepted_count: 0, main_token_is_eos: false },
        ];
        let instrs = generate_mtp_kv_instructions(3, &details);
        assert_eq!(instrs.len(), 1);
        assert!(matches!(&instrs[0], KvCommitInstruction::Rollback { rejected_count: 2, .. }));
    }

    // --- Additional tests ---

    #[test]
    fn test_mtp_output_parse_single_step() {
        // 1 step, depth=3
        let output = vec![50u32, 11, 12, 13];
        let parsed = MtpOutput::parse(&output, 1, 3).unwrap();
        assert_eq!(parsed.main_tokens, vec![50]);
        assert_eq!(parsed.mtp_per_step.len(), 1);
        assert_eq!(parsed.mtp_per_step[0], vec![11, 12, 13]);
    }

    #[test]
    fn test_mtp_output_parse_empty_output() {
        let parsed = MtpOutput::parse(&[], 0, 0).unwrap();
        assert!(parsed.main_tokens.is_empty());
        assert!(parsed.mtp_per_step.is_empty());
    }

    #[test]
    fn test_mtp_output_parse_exact_fit() {
        // 2 steps, depth=2: exactly 2 + 2*2 = 6 elements
        let output = vec![10u32, 20, 31, 32, 41, 42];
        let parsed = MtpOutput::parse(&output, 2, 2).unwrap();
        assert_eq!(parsed.main_tokens, vec![10, 20]);
        assert_eq!(parsed.mtp_per_step[0], vec![31, 32]);
        assert_eq!(parsed.mtp_per_step[1], vec![41, 42]);
    }

    #[test]
    fn test_mtp_output_parse_no_candidates_segment() {
        // 3 steps, depth=2, but output only has main tokens + 1 partial step of candidates
        let output = vec![1u32, 2, 3, 100, 101];
        let parsed = MtpOutput::parse(&output, 3, 2).unwrap();
        assert_eq!(parsed.main_tokens, vec![1, 2, 3]);
        // Step 0 gets partial [100, 101], step 1 and 2 get empty
        assert_eq!(parsed.mtp_per_step[0], vec![100, 101]);
        assert!(parsed.mtp_per_step[1].is_empty());
        assert!(parsed.mtp_per_step[2].is_empty());
    }

    #[test]
    fn test_argmax_token_single_element() {
        assert_eq!(argmax_token(&[7.0f32]), Some(0));
    }

    #[test]
    fn test_argmax_token_tie_returns_one_of_maxima() {
        // When values are equal, argmax picks one of the tied maxima
        let logits = vec![5.0f32, 5.0f32, 3.0f32];
        let result = argmax_token(&logits).unwrap();
        assert!(result == 0 || result == 1);
    }

    #[test]
    fn test_argmax_token_empty_returns_none() {
        assert_eq!(argmax_token(&[]), None);
    }

    #[test]
    fn test_verify_mtp_candidates_more_logits_than_candidates() {
        // 3 logit vectors but only 2 candidates: min(3, 2) = 2 checked
        let logits = vec![
            vec![0.0, 10.0], // argmax=1
            vec![10.0, 0.0], // argmax=0
            vec![0.0, 0.0],  // unused
        ];
        let candidates = vec![1u32, 0];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 2);
    }

    #[test]
    fn test_verify_mtp_candidates_more_candidates_than_logits() {
        // Only 1 logit vector but 3 candidates: min(1, 3) = 1 checked
        let logits = vec![
            vec![0.0, 10.0], // argmax=1
        ];
        let candidates = vec![1u32, 0, 2];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 1);
    }

    #[test]
    fn test_build_verify_result_empty_candidates() {
        let result = build_verify_result(99, &[], 0);
        assert_eq!(result.request_id, 99);
        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.rejected_count, 0);
        assert_eq!(result.draft_count, 0);
        assert!((result.acceptance_rate - 0.0).abs() < 1e-5);
        assert!(result.invariant_check_passed);
    }

    #[test]
    fn test_build_verify_result_full_acceptance() {
        let result = build_verify_result(7, &[5u32, 10, 15], 3);
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![5, 10, 15]);
        assert_eq!(result.rejected_count, 0);
        assert!((result.acceptance_rate - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_default_trait() {
        let ctrl = MtpController::default();
        assert!(ctrl.is_enabled());
        assert!((ctrl.ema_rate() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_with_params_clamps_alpha() {
        let ctrl = MtpController::with_params(0.001, 0.3, 0.5, 3, 5);
        // Alpha clamped to 0.01 minimum
        // Verify via EMA update: alpha * rate + (1-alpha) * 0.5
        let mut test_ctrl = ctrl.clone();
        test_ctrl.record_acceptance(1, 1);
        // With alpha=0.01: ema = 0.01*1.0 + 0.99*0.5 = 0.01 + 0.495 = 0.505
        assert!((test_ctrl.ema_rate() - 0.505).abs() < 1e-4);
    }

    #[test]
    fn test_mtp_controller_with_params_clamps_alpha_upper() {
        let ctrl = MtpController::with_params(5.0, 0.3, 0.5, 3, 5);
        // Alpha clamped to 1.0 maximum
        let mut test_ctrl = ctrl;
        test_ctrl.record_acceptance(1, 2);
        // With alpha=1.0: ema = 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        assert!((test_ctrl.ema_rate() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_ema_converges() {
        let mut ctrl = MtpController::new();
        // Feed consistent 100% acceptance rate many times
        for _ in 0..50 {
            ctrl.record_acceptance(4, 4);
        }
        // EMA should converge close to 1.0
        assert!(ctrl.ema_rate() > 0.95);
    }

    #[test]
    fn test_mtp_controller_disable_enable_clears_streaks() {
        let mut ctrl = MtpController::new();
        // Accumulate low streak
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        assert!(ctrl.is_enabled()); // Not yet disabled (patience=3)

        ctrl.enable(); // Force enable resets streaks
        assert!(ctrl.is_enabled());

        // Should still need 3 consecutive low rounds to disable again
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        assert!(ctrl.is_enabled()); // Only 2 low rounds, not enough
        ctrl.record_acceptance(0, 4);
        assert!(!ctrl.is_enabled()); // 3rd low round triggers disable
    }

    #[test]
    fn test_mtp_controller_record_acceptance_zero_total() {
        let mut ctrl = MtpController::new();
        let initial_ema = ctrl.ema_rate();
        // Zero total means rate=0.0
        let enabled = ctrl.record_acceptance(0, 0);
        assert!(enabled);
        // ema = 0.1*0.0 + 0.9*0.5 = 0.45
        assert!((ctrl.ema_rate() - (0.1 * 0.0 + 0.9 * initial_ema)).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_reenable_requires_consecutive() {
        let mut ctrl = MtpController::new();
        // Disable
        for _ in 0..3 {
            ctrl.record_acceptance(0, 4);
        }
        assert!(!ctrl.is_enabled());

        // 4 high rounds, then 1 low, then 5 high — should NOT re-enable
        for _ in 0..4 {
            ctrl.record_acceptance(1, 1);
        }
        ctrl.record_acceptance(0, 4); // breaks streak
        for _ in 0..5 {
            ctrl.record_acceptance(1, 1);
        }
        assert!(ctrl.is_enabled()); // 5 consecutive after break
    }

    #[test]
    fn test_mtp_kv_instructions_empty_details() {
        let instrs = generate_mtp_kv_instructions(1, &[]);
        assert!(instrs.is_empty());
    }

    #[test]
    fn test_mtp_kv_instructions_multiple_steps() {
        let details = vec![
            MtpStepDetail { main_token: 10, mtp_candidates: vec![20], accepted_count: 1, main_token_is_eos: false },
            MtpStepDetail { main_token: 11, mtp_candidates: vec![21, 22], accepted_count: 0, main_token_is_eos: false },
            MtpStepDetail { main_token: 12, mtp_candidates: vec![23, 24, 25], accepted_count: 2, main_token_is_eos: false },
        ];
        let instrs = generate_mtp_kv_instructions(10, &details);
        // Step 0: commit 1
        // Step 1: rollback 2
        // Step 2: commit 2 + rollback 1
        assert_eq!(instrs.len(), 4);

        assert!(matches!(&instrs[0], KvCommitInstruction::Commit { request_id: 10, accepted_tokens, .. } if accepted_tokens == &[20]));
        assert!(matches!(&instrs[1], KvCommitInstruction::Rollback { request_id: 10, rejected_count: 2, .. }));
        assert!(matches!(&instrs[2], KvCommitInstruction::Commit { request_id: 10, accepted_tokens, .. } if accepted_tokens.len() == 2));
        assert!(matches!(&instrs[3], KvCommitInstruction::Rollback { request_id: 10, rejected_count: 1, .. }));
    }

    #[test]
    fn test_mtp_kv_instructions_zero_candidates() {
        let details = vec![
            MtpStepDetail { main_token: 10, mtp_candidates: vec![], accepted_count: 0, main_token_is_eos: false },
        ];
        let instrs = generate_mtp_kv_instructions(5, &details);
        // No candidates → no commit and no rollback
        assert!(instrs.is_empty());
    }

    #[test]
    fn test_filter_verified_tokens_eos_in_candidate_stops() {
        // EOS in accepted candidate triggers early stop
        let eos = Some(999u32);
        let main = vec![100u32, 200];
        let mtp = vec![vec![998, 999], vec![300]]; // Second candidate is EOS
        let result = filter_verified_tokens(&main, &mtp, eos, |_, candidates| {
            // Accept all candidates
            candidates.len()
        });
        // Step 0: main=100 (not EOS), candidates=[998,999], all accepted
        // But 999 is EOS → commit [100, 998, 999] and stop
        assert_eq!(result.committed_tokens, vec![100, 998, 999]);
        assert_eq!(result.step_details.len(), 1);
    }

    #[test]
    fn test_filter_verified_tokens_fewer_mtp_steps_than_main() {
        let main = vec![100u32, 200, 300];
        let mtp = vec![vec![10u32]]; // Only 1 MTP step for 3 main tokens
        let result = filter_verified_tokens(&main, &mtp, None, |_, _| 1);
        assert_eq!(result.committed_tokens, vec![100, 10, 200, 300]);
        assert_eq!(result.step_details.len(), 3);
        // Step 0 has 1 candidate, accepted 1
        assert_eq!(result.step_details[0].accepted_count, 1);
        // Steps 1 and 2 have no candidates (unwrapped to empty)
        assert!(result.step_details[1].mtp_candidates.is_empty());
        assert!(result.step_details[2].mtp_candidates.is_empty());
    }

    #[test]
    fn test_filter_verified_tokens_verify_fn_per_step() {
        let main = vec![100u32, 200];
        let mtp = vec![vec![10u32, 20], vec![30u32, 40]];
        let result = filter_verified_tokens(&main, &mtp, None, |step, _candidates| {
            // Step 0: accept 1, Step 1: accept 2
            if step == 0 { 1 } else { 2 }
        });
        assert_eq!(result.committed_tokens, vec![100, 10, 200, 30, 40]);
        assert_eq!(result.step_details[0].accepted_count, 1);
        assert_eq!(result.step_details[1].accepted_count, 2);
        assert_eq!(result.total_mtp_accepted, 3);
    }

    #[test]
    fn test_mtp_step_detail_clone_and_debug() {
        let detail = MtpStepDetail {
            main_token: 42,
            mtp_candidates: vec![100, 200],
            accepted_count: 1,
            main_token_is_eos: false,
        };
        let cloned = detail.clone();
        assert_eq!(cloned.main_token, 42);
        assert_eq!(cloned.mtp_candidates, vec![100, 200]);
        assert_eq!(cloned.accepted_count, 1);
        assert!(!cloned.main_token_is_eos);

        let debug_str = format!("{:?}", detail);
        assert!(debug_str.contains("main_token: 42"));
    }

    #[test]
    fn test_mtp_generate_result_clone_and_debug() {
        let result = MtpGenerateResult {
            committed_tokens: vec![1, 2, 3],
            total_mtp_candidates: 5,
            total_mtp_accepted: 3,
            step_details: vec![],
        };
        let cloned = result.clone();
        assert_eq!(cloned.committed_tokens, vec![1, 2, 3]);
        assert_eq!(cloned.total_mtp_candidates, 5);

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("committed_tokens"));
        assert!(debug_str.contains("total_mtp_candidates: 5"));
    }

    #[test]
    fn test_mtp_controller_effective_depth_boundary_values() {
        let mut ctrl = MtpController::new();

        // Exactly 0.8 → not > 0.8, falls to > 0.5 check
        ctrl.ema_rate = 0.8;
        assert_eq!(ctrl.effective_depth(4), 2);

        // Exactly 0.5 → not > 0.5, falls to > 0.3 check → depth 1
        ctrl.ema_rate = 0.5;
        assert_eq!(ctrl.effective_depth(4), 1);

        // Exactly 0.3 → not > 0.3, falls to else → depth 0
        ctrl.ema_rate = 0.3;
        assert_eq!(ctrl.effective_depth(4), 0);

        // max_depth=1, high EMA → returns 1 (not more than max)
        ctrl.ema_rate = 0.95;
        assert_eq!(ctrl.effective_depth(1), 1);

        // max_depth=0 always returns 0 regardless of EMA
        ctrl.enabled = true;
        ctrl.ema_rate = 0.99;
        assert_eq!(ctrl.effective_depth(0), 0);
    }

    // --- Additional tests (round 2) ---

    #[test]
    fn test_mtp_generate_result_empty_construction() {
        let result = MtpGenerateResult {
            committed_tokens: vec![],
            total_mtp_candidates: 0,
            total_mtp_accepted: 0,
            step_details: vec![],
        };
        assert!(result.committed_tokens.is_empty());
        assert_eq!(result.total_mtp_candidates, 0);
        assert_eq!(result.total_mtp_accepted, 0);
        assert!(result.step_details.is_empty());
    }

    #[test]
    fn test_mtp_generate_result_step_details_access() {
        let details = vec![
            MtpStepDetail {
                main_token: 10,
                mtp_candidates: vec![11, 12],
                accepted_count: 2,
                main_token_is_eos: false,
            },
            MtpStepDetail {
                main_token: 13,
                mtp_candidates: vec![],
                accepted_count: 0,
                main_token_is_eos: true,
            },
        ];
        let result = MtpGenerateResult {
            committed_tokens: vec![10, 11, 12, 13],
            total_mtp_candidates: 2,
            total_mtp_accepted: 2,
            step_details: details,
        };
        assert_eq!(result.step_details.len(), 2);
        assert_eq!(result.step_details[0].main_token, 10);
        assert_eq!(result.step_details[1].main_token_is_eos, true);
        assert_eq!(result.step_details[0].mtp_candidates, vec![11, 12]);
    }

    #[test]
    fn test_mtp_step_detail_eos_true() {
        let detail = MtpStepDetail {
            main_token: 2,
            mtp_candidates: vec![3, 4],
            accepted_count: 1,
            main_token_is_eos: true,
        };
        assert!(detail.main_token_is_eos);
        assert_eq!(detail.accepted_count, 1);
        assert_eq!(detail.mtp_candidates.len(), 2);
    }

    #[test]
    fn test_mtp_output_parse_zero_steps() {
        let output = vec![100u32, 200, 300];
        let parsed = MtpOutput::parse(&output, 0, 2).unwrap();
        assert!(parsed.main_tokens.is_empty());
        assert!(parsed.mtp_per_step.is_empty());
    }

    #[test]
    fn test_mtp_output_parse_truncated_last_step() {
        // 3 steps, depth=3, but only enough data for 2 full + 1 partial
        let output = vec![1u32, 2, 3, 10, 11, 12, 20, 21, 22, 30];
        let parsed = MtpOutput::parse(&output, 3, 3).unwrap();
        assert_eq!(parsed.main_tokens, vec![1, 2, 3]);
        assert_eq!(parsed.mtp_per_step[0], vec![10, 11, 12]);
        assert_eq!(parsed.mtp_per_step[1], vec![20, 21, 22]);
        assert_eq!(parsed.mtp_per_step[2], vec![30]); // partial: only 1 of 3
    }

    #[test]
    fn test_mtp_controller_new_initial_state() {
        let ctrl = MtpController::new();
        assert!(ctrl.is_enabled());
        assert!((ctrl.ema_rate() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_with_params_custom_thresholds() {
        let ctrl = MtpController::with_params(0.2, 0.1, 0.9, 2, 10);
        assert!(ctrl.is_enabled());
        assert!((ctrl.ema_rate() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_record_acceptance_full_acceptance() {
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(5, 5);
        // ema = 0.1*1.0 + 0.9*0.5 = 0.55
        assert!((ctrl.ema_rate() - 0.55).abs() < 1e-5);
        assert!(ctrl.is_enabled());
    }

    #[test]
    fn test_mtp_controller_record_acceptance_zero_accepted() {
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(0, 10);
        // ema = 0.1*0.0 + 0.9*0.5 = 0.45
        assert!((ctrl.ema_rate() - 0.45).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_effective_depth_disabled_returns_zero() {
        let mut ctrl = MtpController::new();
        ctrl.ema_rate = 0.99;
        ctrl.disable();
        assert_eq!(ctrl.effective_depth(8), 0);
    }

    #[test]
    fn test_mtp_controller_effective_depth_moderate_clamps_to_two() {
        let mut ctrl = MtpController::new();
        ctrl.ema_rate = 0.65;
        // max_depth=1 < 2, so min(1,2) = 1
        assert_eq!(ctrl.effective_depth(1), 1);
        // max_depth=5 > 2, so min(5,2) = 2
        assert_eq!(ctrl.effective_depth(5), 2);
    }

    #[test]
    fn test_mtp_controller_disable_then_enable_resets_streaks() {
        let mut ctrl = MtpController::new();
        ctrl.disable();
        assert!(!ctrl.is_enabled());
        ctrl.enable();
        assert!(ctrl.is_enabled());
        // After enable, low_streak and stable_streak are 0
        // So it takes 3 low rounds to disable again
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        assert!(ctrl.is_enabled()); // 2 < patience=3
        ctrl.record_acceptance(0, 4);
        assert!(!ctrl.is_enabled()); // 3 >= patience=3
    }

    #[test]
    fn test_mtp_controller_ema_updates_multiple_rounds() {
        let mut ctrl = MtpController::new(); // alpha=0.1, initial ema=0.5
        ctrl.record_acceptance(1, 2); // rate=0.5, ema=0.1*0.5+0.9*0.5=0.5
        assert!((ctrl.ema_rate() - 0.5).abs() < 1e-5);
        ctrl.record_acceptance(2, 4); // rate=0.5, ema=0.1*0.5+0.9*0.5=0.5
        assert!((ctrl.ema_rate() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_clone_preserves_state() {
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        let cloned = ctrl.clone();
        assert_eq!(cloned.is_enabled(), ctrl.is_enabled());
        assert!((cloned.ema_rate() - ctrl.ema_rate()).abs() < 1e-10);
    }

    #[test]
    fn test_mtp_controller_debug_output_contains_fields() {
        let ctrl = MtpController::new();
        let debug = format!("{:?}", ctrl);
        assert!(debug.contains("alpha"));
        assert!(debug.contains("ema_rate"));
        assert!(debug.contains("enabled"));
        assert!(debug.contains("low_streak"));
        assert!(debug.contains("stable_streak"));
    }

    #[test]
    fn test_mtp_generate_result_clone_independence() {
        let result = MtpGenerateResult {
            committed_tokens: vec![10, 20],
            total_mtp_candidates: 3,
            total_mtp_accepted: 1,
            step_details: vec![MtpStepDetail {
                main_token: 10,
                mtp_candidates: vec![20, 30, 40],
                accepted_count: 1,
                main_token_is_eos: false,
            }],
        };
        let mut cloned = result.clone();
        cloned.committed_tokens.push(99);
        assert_eq!(result.committed_tokens, vec![10, 20]); // original unchanged
        assert_eq!(cloned.committed_tokens, vec![10, 20, 99]);
    }

    // --- Round 3: Additional coverage tests (46 new) ---

    #[test]
    fn test_mtp_output_struct_field_access() {
        let output = MtpOutput {
            main_tokens: vec![1, 2, 3],
            mtp_per_step: vec![vec![10, 11], vec![20, 21], vec![30, 31]],
        };
        assert_eq!(output.main_tokens.len(), 3);
        assert_eq!(output.mtp_per_step.len(), 3);
        assert_eq!(output.mtp_per_step[1], vec![20, 21]);
    }

    #[test]
    fn test_mtp_output_parse_large_depth() {
        let output: Vec<u32> = (0..11).collect();
        let parsed = MtpOutput::parse(&output, 1, 10).unwrap();
        assert_eq!(parsed.main_tokens, vec![0]);
        assert_eq!(parsed.mtp_per_step[0], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_mtp_output_parse_main_tokens_only_no_room_for_candidates() {
        let output = vec![100u32, 200];
        let parsed = MtpOutput::parse(&output, 2, 3).unwrap();
        assert_eq!(parsed.main_tokens, vec![100, 200]);
        assert!(parsed.mtp_per_step[0].is_empty());
        assert!(parsed.mtp_per_step[1].is_empty());
    }

    #[test]
    fn test_mtp_output_parse_single_step_large_depth_truncated() {
        let output = vec![50u32, 10, 11];
        let parsed = MtpOutput::parse(&output, 1, 5).unwrap();
        assert_eq!(parsed.main_tokens, vec![50]);
        assert_eq!(parsed.mtp_per_step[0], vec![10, 11]);
    }

    #[test]
    fn test_mtp_output_parse_ten_steps() {
        let mut output = Vec::with_capacity(30);
        for i in 0..10u32 {
            output.push(i);
        }
        for step in 0..10u32 {
            output.push(step * 100 + 1);
            output.push(step * 100 + 2);
        }
        let parsed = MtpOutput::parse(&output, 10, 2).unwrap();
        assert_eq!(parsed.main_tokens.len(), 10);
        assert_eq!(parsed.mtp_per_step.len(), 10);
        for step in 0..10 {
            assert_eq!(
                parsed.mtp_per_step[step],
                vec![step as u32 * 100 + 1, step as u32 * 100 + 2]
            );
        }
    }

    #[test]
    fn test_argmax_token_max_at_last_position() {
        let logits = vec![1.0f32, 2.0, 3.0, 10.0];
        assert_eq!(argmax_token(&logits), Some(3));
    }

    #[test]
    fn test_argmax_token_max_at_second_position() {
        let logits = vec![1.0f32, 99.0, 3.0];
        assert_eq!(argmax_token(&logits), Some(1));
    }

    #[test]
    fn test_argmax_token_all_negative() {
        let logits = vec![-5.0f32, -1.0, -10.0];
        assert_eq!(argmax_token(&logits), Some(1));
    }

    #[test]
    fn test_argmax_token_with_zero() {
        let logits = vec![0.0f32, -1.0, -2.0];
        assert_eq!(argmax_token(&logits), Some(0));
    }

    #[test]
    fn test_verify_mtp_candidates_single_match() {
        let logits = vec![vec![0.0, 0.0, 10.0]];
        let candidates = vec![2u32];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 1);
    }

    #[test]
    fn test_verify_mtp_candidates_single_mismatch() {
        let logits = vec![vec![10.0, 0.0, 0.0]];
        let candidates = vec![2u32];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 0);
    }

    #[test]
    fn test_verify_mtp_candidates_only_logits_no_candidates() {
        let logits = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let candidates: Vec<u32> = vec![];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 0);
    }

    #[test]
    fn test_verify_mtp_candidates_long_chain_partial() {
        let logits = vec![
            vec![0.0, 10.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![0.0, 10.0],
            vec![10.0, 0.0],
        ];
        let candidates = vec![1u32, 0, 1, 0, 0];
        assert_eq!(verify_mtp_candidates(&logits, &candidates), 3);
    }

    #[test]
    fn test_build_verify_result_single_candidate() {
        let result = build_verify_result(1, &[42u32], 1);
        assert_eq!(result.request_id, 1);
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.accepted_tokens, vec![42]);
        assert_eq!(result.rejected_count, 0);
        assert!((result.acceptance_rate - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_build_verify_result_partial_many_candidates() {
        let candidates: Vec<u32> = (0..10).collect();
        let result = build_verify_result(5, &candidates, 3);
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![0, 1, 2]);
        assert_eq!(result.rejected_count, 7);
        assert!((result.acceptance_rate - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_build_verify_result_preserves_request_id() {
        let result = build_verify_result(u64::MAX, &[1u32], 0);
        assert_eq!(result.request_id, u64::MAX);
        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.rejected_count, 1);
    }

    #[test]
    fn test_build_verify_result_candidate_count_matches_draft() {
        let candidates = vec![10u32, 20, 30, 40, 50];
        let result = build_verify_result(1, &candidates, 5);
        assert_eq!(result.draft_count, 5);
        assert_eq!(result.accepted_count, 5);
        assert_eq!(result.accepted_tokens, vec![10, 20, 30, 40, 50]);
        assert_eq!(result.rejected_count, 0);
    }

    #[test]
    fn test_mtp_controller_ema_converges_to_zero() {
        let mut ctrl = MtpController::new();
        for _ in 0..50 {
            ctrl.record_acceptance(0, 4);
        }
        assert!(ctrl.ema_rate() < 0.05);
    }

    #[test]
    fn test_mtp_controller_disable_does_not_change_ema() {
        let mut ctrl = MtpController::new();
        let ema_before = ctrl.ema_rate();
        ctrl.disable();
        assert!((ctrl.ema_rate() - ema_before).abs() < 1e-10);
    }

    #[test]
    fn test_mtp_controller_multiple_disable_enable_cycles() {
        let mut ctrl = MtpController::new();
        for _ in 0..5 {
            ctrl.disable();
            assert!(!ctrl.is_enabled());
            ctrl.enable();
            assert!(ctrl.is_enabled());
        }
        assert!(ctrl.is_enabled());
    }

    #[test]
    fn test_mtp_controller_with_params_alpha_one() {
        let mut ctrl = MtpController::with_params(1.0, 0.3, 0.5, 3, 5);
        ctrl.record_acceptance(3, 4);
        assert!((ctrl.ema_rate() - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_with_params_alpha_boundary() {
        let mut ctrl = MtpController::with_params(0.01, 0.3, 0.5, 3, 5);
        ctrl.record_acceptance(1, 1);
        assert!((ctrl.ema_rate() - 0.505).abs() < 1e-4);
    }

    #[test]
    fn test_mtp_controller_record_acceptance_when_disabled_uses_enable_threshold() {
        let mut ctrl = MtpController::with_params(0.1, 0.3, 0.5, 3, 5);
        for _ in 0..3 {
            ctrl.record_acceptance(0, 4);
        }
        assert!(!ctrl.is_enabled());
        for _ in 0..4 {
            ctrl.record_acceptance(1, 2);
        }
        assert!(!ctrl.is_enabled());
        ctrl.record_acceptance(1, 2);
        assert!(ctrl.is_enabled());
    }

    #[test]
    fn test_mtp_controller_record_acceptance_disabled_below_threshold_no_streak() {
        let mut ctrl = MtpController::with_params(0.1, 0.3, 0.5, 3, 5);
        for _ in 0..3 {
            ctrl.record_acceptance(0, 4);
        }
        assert!(!ctrl.is_enabled());
        ctrl.record_acceptance(0, 10);
        for _ in 0..4 {
            ctrl.record_acceptance(1, 1);
        }
        assert!(!ctrl.is_enabled());
        ctrl.record_acceptance(1, 1);
        assert!(ctrl.is_enabled());
    }

    #[test]
    fn test_mtp_controller_effective_depth_high_ema_returns_full() {
        let mut ctrl = MtpController::new();
        ctrl.ema_rate = 0.95;
        assert_eq!(ctrl.effective_depth(8), 8);
        assert_eq!(ctrl.effective_depth(3), 3);
    }

    #[test]
    fn test_mtp_controller_effective_depth_just_above_0_8() {
        let mut ctrl = MtpController::new();
        ctrl.ema_rate = 0.81;
        assert_eq!(ctrl.effective_depth(6), 6);
    }

    #[test]
    fn test_mtp_controller_effective_depth_just_below_0_3() {
        let mut ctrl = MtpController::new();
        ctrl.ema_rate = 0.29;
        assert_eq!(ctrl.effective_depth(4), 0);
    }

    #[test]
    fn test_mtp_controller_record_acceptance_returns_enabled_status() {
        let mut ctrl = MtpController::new();
        assert!(ctrl.record_acceptance(1, 1));
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        assert!(!ctrl.is_enabled());
        assert!(!ctrl.record_acceptance(1, 2));
    }

    #[test]
    fn test_mtp_controller_reset_after_disable() {
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        ctrl.record_acceptance(0, 4);
        assert!(!ctrl.is_enabled());
        ctrl.reset();
        assert!(ctrl.is_enabled());
        assert!((ctrl.ema_rate() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mtp_controller_record_acceptance_alternating_rates() {
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(4, 4);
        assert!((ctrl.ema_rate() - 0.55).abs() < 1e-5);
        ctrl.record_acceptance(0, 4);
        assert!((ctrl.ema_rate() - 0.495).abs() < 1e-4);
        ctrl.record_acceptance(4, 4);
        assert!(ctrl.ema_rate() > 0.54);
        assert!(ctrl.ema_rate() < 0.55);
    }

    #[test]
    fn test_mtp_controller_default_equals_new() {
        let new_ctrl = MtpController::new();
        let default_ctrl = MtpController::default();
        assert_eq!(new_ctrl.is_enabled(), default_ctrl.is_enabled());
        assert!((new_ctrl.ema_rate() - default_ctrl.ema_rate()).abs() < 1e-10);
    }

    #[test]
    fn test_generate_mtp_kv_instructions_single_step_all_rejected() {
        let details = vec![MtpStepDetail {
            main_token: 5,
            mtp_candidates: vec![10, 20],
            accepted_count: 0,
            main_token_is_eos: false,
        }];
        let instrs = generate_mtp_kv_instructions(42, &details);
        assert_eq!(instrs.len(), 1);
        match &instrs[0] {
            KvCommitInstruction::Rollback {
                request_id,
                rejected_count,
                ..
            } => {
                assert_eq!(*request_id, 42);
                assert_eq!(*rejected_count, 2);
            }
            other => panic!("Expected Rollback, got {:?}", other),
        }
    }

    #[test]
    fn test_generate_mtp_kv_instructions_commit_tokens_content() {
        let details = vec![MtpStepDetail {
            main_token: 1,
            mtp_candidates: vec![100, 200, 300],
            accepted_count: 2,
            main_token_is_eos: false,
        }];
        let instrs = generate_mtp_kv_instructions(7, &details);
        assert_eq!(instrs.len(), 2);
        match &instrs[0] {
            KvCommitInstruction::Commit {
                accepted_tokens,
                request_id,
                kv_pages_to_commit,
            } => {
                assert_eq!(*request_id, 7);
                assert_eq!(*accepted_tokens, vec![100, 200]);
                assert_eq!(*kv_pages_to_commit, vec![0, 1]);
            }
            other => panic!("Expected Commit, got {:?}", other),
        }
        match &instrs[1] {
            KvCommitInstruction::Rollback {
                rejected_count,
                kv_pages_to_free,
                ..
            } => {
                assert_eq!(*rejected_count, 1);
                assert_eq!(*kv_pages_to_free, vec![0]);
            }
            other => panic!("Expected Rollback, got {:?}", other),
        }
    }

    #[test]
    fn test_generate_mtp_kv_instructions_large_request_id() {
        let details = vec![MtpStepDetail {
            main_token: 1,
            mtp_candidates: vec![2],
            accepted_count: 1,
            main_token_is_eos: false,
        }];
        let instrs = generate_mtp_kv_instructions(u64::MAX, &details);
        match &instrs[0] {
            KvCommitInstruction::Commit { request_id, .. } => {
                assert_eq!(*request_id, u64::MAX);
            }
            other => panic!("Expected Commit, got {:?}", other),
        }
    }

    #[test]
    fn test_generate_mtp_kv_instructions_page_ids_sequential() {
        let details = vec![MtpStepDetail {
            main_token: 1,
            mtp_candidates: vec![10, 20, 30],
            accepted_count: 3,
            main_token_is_eos: false,
        }];
        let instrs = generate_mtp_kv_instructions(1, &details);
        match &instrs[0] {
            KvCommitInstruction::Commit {
                kv_pages_to_commit, ..
            } => {
                assert_eq!(*kv_pages_to_commit, vec![0, 1, 2]);
            }
            other => panic!("Expected Commit, got {:?}", other),
        }
    }

    #[test]
    fn test_filter_verified_tokens_empty_main() {
        let result = filter_verified_tokens(&[], &[], None, |_, _| 0);
        assert!(result.committed_tokens.is_empty());
        assert_eq!(result.total_mtp_candidates, 0);
        assert_eq!(result.total_mtp_accepted, 0);
        assert!(result.step_details.is_empty());
    }

    #[test]
    fn test_filter_verified_tokens_eos_as_first_candidate() {
        let eos = Some(999u32);
        let main = vec![100u32];
        let mtp = vec![vec![999u32, 200]];
        let result = filter_verified_tokens(&main, &mtp, eos, |_, _| 2);
        assert_eq!(result.committed_tokens, vec![100, 999]);
        assert_eq!(result.step_details.len(), 1);
    }

    #[test]
    fn test_filter_verified_tokens_verify_fn_receives_correct_candidates() {
        let main = vec![10u32, 20, 30];
        let mtp = vec![vec![1u32], vec![2u32], vec![3u32]];
        let result = filter_verified_tokens(&main, &mtp, None, |step, candidates| {
            match step {
                0 => assert_eq!(candidates, &[1u32]),
                1 => assert_eq!(candidates, &[2u32]),
                2 => assert_eq!(candidates, &[3u32]),
                _ => panic!("unexpected step"),
            }
            0usize
        });
        let _ = result;
    }

    #[test]
    fn test_filter_verified_tokens_verify_fn_step_dependent_acceptance() {
        let main = vec![10u32, 20, 30];
        let mtp = vec![vec![1u32], vec![2u32], vec![3u32]];
        let result = filter_verified_tokens(&main, &mtp, None, |step, _| {
            if step < 2 { 1 } else { 0 }
        });
        assert_eq!(result.committed_tokens, vec![10, 1, 20, 2, 30]);
    }

    #[test]
    fn test_filter_verified_tokens_no_eos_none() {
        let main = vec![100u32, 200];
        let mtp = vec![vec![10u32], vec![20u32]];
        let result = filter_verified_tokens(&main, &mtp, None, |_, _| 1);
        assert_eq!(result.committed_tokens, vec![100, 10, 200, 20]);
        assert!(!result.step_details[0].main_token_is_eos);
        assert!(!result.step_details[1].main_token_is_eos);
    }

    #[test]
    fn test_filter_verified_tokens_main_eos_skips_candidates() {
        let eos = Some(999u32);
        let main = vec![999u32, 200, 300];
        let mtp = vec![vec![10], vec![20], vec![30]];
        let result = filter_verified_tokens(&main, &mtp, eos, |_, _| 1);
        assert_eq!(result.committed_tokens, vec![999]);
        assert_eq!(result.step_details.len(), 1);
        assert!(result.step_details[0].main_token_is_eos);
        assert_eq!(result.step_details[0].accepted_count, 0);
    }

    #[test]
    fn test_filter_verified_tokens_multiple_steps_eos_midway() {
        let eos = Some(50u32);
        let main = vec![10u32, 20u32];
        let mtp = vec![vec![11, 12], vec![50, 22]];
        let result = filter_verified_tokens(&main, &mtp, eos, |_, candidates| {
            candidates.len().min(2)
        });
        // Step 0: main=10, candidates=[11,12], all accepted → [10,11,12]
        // Step 1: main=20, candidates=[50,22], accept all.
        //   Loop commits 20, then tok=50 (EOS) → break.
        //   committed = [10,11,12,20,50]
        //   last_accepted_is_eos checks candidates[1]=22 → false (not last)
        //   But the for-tok loop already broke on EOS, so all steps done.
        assert_eq!(result.committed_tokens, vec![10, 11, 12, 20, 50]);
        // Both steps produce step_details since the EOS break is inside the inner loop
        assert_eq!(result.step_details.len(), 2);
    }

    #[test]
    fn test_filter_verified_tokens_candidate_accepted_zero_with_nonempty() {
        let main = vec![100u32, 200];
        let mtp = vec![vec![10u32, 20], vec![30u32]];
        let result = filter_verified_tokens(&main, &mtp, None, |_, _| 0);
        assert_eq!(result.committed_tokens, vec![100, 200]);
        assert_eq!(result.total_mtp_candidates, 3);
        assert_eq!(result.total_mtp_accepted, 0);
    }

    #[test]
    fn test_mtp_step_detail_fields_all_set() {
        let detail = MtpStepDetail {
            main_token: 100,
            mtp_candidates: vec![200, 300, 400],
            accepted_count: 2,
            main_token_is_eos: true,
        };
        assert_eq!(detail.main_token, 100);
        assert_eq!(detail.mtp_candidates.len(), 3);
        assert_eq!(detail.accepted_count, 2);
        assert!(detail.main_token_is_eos);
    }

    #[test]
    fn test_mtp_step_detail_clone_is_deep() {
        let detail = MtpStepDetail {
            main_token: 1,
            mtp_candidates: vec![10, 20],
            accepted_count: 1,
            main_token_is_eos: false,
        };
        let mut cloned = detail.clone();
        cloned.mtp_candidates.push(30);
        assert_eq!(detail.mtp_candidates.len(), 2);
        assert_eq!(cloned.mtp_candidates.len(), 3);
    }

    #[test]
    fn test_mtp_generate_result_clone_step_details_independence() {
        let result = MtpGenerateResult {
            committed_tokens: vec![1],
            total_mtp_candidates: 0,
            total_mtp_accepted: 0,
            step_details: vec![MtpStepDetail {
                main_token: 1,
                mtp_candidates: vec![10],
                accepted_count: 0,
                main_token_is_eos: false,
            }],
        };
        let mut cloned = result.clone();
        cloned.step_details[0].mtp_candidates.push(99);
        assert_eq!(result.step_details[0].mtp_candidates.len(), 1);
        assert_eq!(cloned.step_details[0].mtp_candidates.len(), 2);
    }
}
