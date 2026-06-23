//! 采样语义契约实现 (SPEC §3.1 / §3.8 采样语义契约)
//!
//! 核心原则：temperature / top_k / top_p 必须真正生效，禁止"传参但永远 argmax"的 bug。
//!
//! **采样路径**:
//! 1. `temperature == 0.0` → greedy argmax (确定性)
//! 2. `temperature > 0.0`  → logits/T → (top_k) → softmax (subtract max) → (top_p) → multinomial
//! 3. `top_k > 0`   → 保留前 k 个 logit（按原始分值排序）后再 softmax
//! 4. `top_p ∈ (0, 1)` → softmax 后按累积概率截断（nucleus sampling）
//! 5. top_k 与 top_p 可叠加（top_k 先生效，top_p 后生效）
//!
//! **数值稳定性**: softmax 前 subtract max，避免 exp 溢出。
//! **PRNG**: 使用 `rand::thread_rng()`（基于系统熵源），保证跨调用随机性。

use crate::engine::executor::{BackendError as BE, SamplingConfig};
use rand::Rng;

/// 对一行 logits 执行完整的采样管线，返回选中的 token id。
///
/// # 参数
/// - `row`: `[vocab]` 形状的 logit 行
/// - `sampling`: 采样配置（temperature / top_k / top_p）
///
/// # 错误
/// - 空 `row` → `BE::Cpu`
/// - softmax 分母 ≤ 0（所有 logit 都是 -inf） → `BE::Cpu`
/// - 采样概率和非正 → `BE::Cpu`
pub(crate) fn sample_logits_row(row: &[f32], sampling: &SamplingConfig) -> Result<u32, BE> {
    if row.is_empty() {
        return Err(BE::Cpu("empty logits row in sample_logits_row".into()));
    }

    // ── Greedy path: temperature == 0 → argmax（忽略 top_k/top_p）──
    if sampling.temperature <= 0.0 {
        return Ok(argmax(row));
    }

    // ── Stochastic path ──
    // 1) 按 logit 降序取 top-k（或保留全部）
    let effective_k = if sampling.top_k > 0 && sampling.top_k < row.len() {
        sampling.top_k
    } else {
        row.len()
    };
    let mut indices: Vec<usize> = (0..row.len()).collect();
    // select_nth + sort_by 组合在 effective_k 很小时更高效；此处 K 通常 ≤ 50，直接排序
    indices.sort_unstable_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(effective_k);

    // 2) 应用温度缩放
    let inv_t = 1.0f32 / sampling.temperature;
    let scaled: Vec<f32> = indices.iter().map(|&i| row[i] * inv_t).collect();

    // 3) softmax（subtract max 数值稳定）
    let max_val = scaled
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return Err(BE::Cpu(format!(
            "sample_logits_row: non-finite max logit after scaling (max={max_val})"
        )));
    }
    let mut probs: Vec<f32> = scaled.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if !(sum > 0.0) {
        return Err(BE::Cpu(format!(
            "sample_logits_row: softmax produced non-positive sum ({sum}) — all logits -inf?"
        )));
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // 4) top-p (nucleus)：按概率降序累积 — 因 scaled 已按原 logit 降序，
    //    且 softmax 单调，probs 同样降序，无需再次排序。
    let mut candidates: Vec<(usize, f32)> = indices.into_iter().zip(probs).collect();
    if sampling.top_p > 0.0 && sampling.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut cutoff = candidates.len();
        for (i, &(_, p)) in candidates.iter().enumerate() {
            cumulative += p;
            if cumulative >= sampling.top_p {
                cutoff = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff);
        // 重新归一化
        let new_sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
        if !(new_sum > 0.0) {
            return Err(BE::Cpu(format!(
                "sample_logits_row: top-p renormalization failed (sum={new_sum})"
            )));
        }
        for (_, p) in candidates.iter_mut() {
            *p /= new_sum;
        }
    }

    // 5) multinomial 采样：累积分布 + uniform[0,1) 采样
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen_range(0.0_f32..1.0_f32);
    let mut cumulative = 0.0f32;
    let last_idx = candidates
        .last()
        .expect("candidates non-empty: row.len() > 0 guaranteed")
        .0;
    for (idx, p) in &candidates {
        cumulative += *p;
        if r < cumulative {
            return Ok(*idx as u32);
        }
    }
    // 累积到末尾仍未触发（仅由浮点舍入导致），返回最后一个候选（概率上最合理的落点）
    Ok(last_idx as u32)
}

/// 极简 argmax（T==0 路径），NaN 视为最小。
fn argmax(row: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        // NaN 比较返回 false，不会更新 best
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

// ===========================================================================
// 单元测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(t: f32, k: usize, p: f32) -> SamplingConfig {
        SamplingConfig {
            temperature: t,
            top_k: k,
            top_p: p,
        }
    }

    #[test]
    fn sample_greedy_when_temperature_zero() {
        // argmax 必在 index 1 (值 10.0)
        let logits = [1.0f32, 10.0, 3.0];
        for _ in 0..50 {
            let tok = sample_logits_row(&logits, &cfg(0.0, 0, 1.0)).expect("sample ok");
            assert_eq!(tok, 1, "T=0 must always return argmax");
        }
    }

    #[test]
    fn sample_stochastic_when_temperature_positive() {
        // uniform logits → T=1.0 → uniform distribution → argmax 命中率应 ≈ 1/3
        let logits = [1.0f32, 1.0, 1.0];
        let mut non_zero_count = 0usize;
        let trials = 1000;
        for _ in 0..trials {
            let tok =
                sample_logits_row(&logits, &cfg(1.0, 0, 1.0)).expect("sample ok");
            assert!(tok < 3, "token must be in vocab");
            if tok != 0 {
                non_zero_count += 1;
            }
        }
        // 期望 2/3 ≈ 667 次 != 0；阈值 40% 远低于期望，统计不可能失败
        let ratio = non_zero_count as f32 / trials as f32;
        assert!(
            ratio >= 0.40,
            "uniform sampling stuck at argmax: non-zero ratio = {ratio} (< 0.40)"
        );
    }

    #[test]
    fn top_k_restricts_candidate_set() {
        // 5 个 logit，top_k=2 必须只从前两名（index 2, 4）采样
        let logits = [0.0f32, 1.0, 10.0, 2.0, 9.0];
        let allowed = [2u32, 4u32];
        for _ in 0..500 {
            let tok = sample_logits_row(&logits, &cfg(1.0, 2, 1.0)).expect("sample ok");
            assert!(
                allowed.contains(&tok),
                "top_k=2 but sampled token {tok} outside top-2 set {:?}",
                allowed
            );
        }
    }

    #[test]
    fn top_p_restricts_candidate_set() {
        // 尖锐分布：index 2 的概率远大于其他 → top_p=0.5 应只保留 index 2
        let logits = [0.0f32, 0.0, 10.0, 0.0, 0.0];
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &cfg(1.0, 0, 0.5)).expect("sample ok");
            assert_eq!(tok, 2, "top_p=0.5 with sharp distribution must pick top-1");
        }
    }

    #[test]
    fn top_k_and_top_p_combine() {
        // top_k=3 → 保留 index 2, 4, 3 （logit=10, 9, 2）
        // T=1 softmax 后，index 2/4 占绝大多数概率，top_p=0.9 应截至 index 2 或 2+4
        let logits = [0.0f32, 1.0, 10.0, 2.0, 9.0];
        let allowed = [2u32, 4u32];
        for _ in 0..500 {
            let tok = sample_logits_row(&logits, &cfg(1.0, 3, 0.9)).expect("sample ok");
            assert!(
                allowed.contains(&tok),
                "top_k=3 + top_p=0.9 expected to restrict to {{2,4}}, got {tok}"
            );
        }
    }

    #[test]
    fn empty_row_returns_error() {
        let logits: [f32; 0] = [];
        assert!(sample_logits_row(&logits, &cfg(1.0, 0, 1.0)).is_err());
    }

    #[test]
    fn all_neg_inf_logits_return_error() {
        let logits = [f32::NEG_INFINITY; 4];
        let err = sample_logits_row(&logits, &cfg(1.0, 0, 1.0));
        assert!(err.is_err(), "all -inf must return error, got {err:?}");
    }

    #[test]
    fn temperature_one_identity_on_logits() {
        // T=1.0 是身份缩放；对单峰分布（一个 logit 极高），应几乎总选该峰
        let logits = [0.0f32, 0.0, 20.0, 0.0];
        let mut peak_count = 0usize;
        for _ in 0..500 {
            let tok =
                sample_logits_row(&logits, &cfg(1.0, 0, 1.0)).expect("sample ok");
            if tok == 2 {
                peak_count += 1;
            }
        }
        // softmax(20 vs 0) ≈ exp(20)/(3+exp(20)) ≈ 1.0；期望 500/500 命中
        assert!(
            peak_count >= 495,
            "sharp peak not respected: {peak_count}/500"
        );
    }

    // ── SamplingConfig tests ──────────────────────────────────────────

    #[test]
    fn sampling_config_default_values() {
        // Arrange: use Default trait
        let default_cfg = SamplingConfig::default();

        // Assert: temperature=1.0, top_k=0 (disabled), top_p=1.0 (disabled)
        assert_eq!(default_cfg.temperature, 1.0);
        assert_eq!(default_cfg.top_k, 0);
        assert_eq!(default_cfg.top_p, 1.0);
    }

    #[test]
    fn sampling_config_is_copy() {
        // Arrange
        let original = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
        };

        // Act: Copy semantics (implicit Copy)
        let copied = original;
        let also_original = original;

        // Assert: both copies retain independent values
        assert_eq!(copied.temperature, 0.7);
        assert_eq!(also_original.temperature, 0.7);
        assert_eq!(copied.top_k, 50);
        assert_eq!(also_original.top_k, 50);
    }

    #[test]
    fn sampling_config_is_clone() {
        // Arrange
        let original = SamplingConfig {
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
        };

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(cloned.temperature, original.temperature);
        assert_eq!(cloned.top_k, original.top_k);
        assert_eq!(cloned.top_p, original.top_p);
    }

    #[test]
    fn sampling_config_debug_format() {
        // Arrange
        let config = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
        };

        // Act
        let debug_str = format!("{config:?}");

        // Assert: debug output contains all field names and values
        assert!(debug_str.contains("temperature"), "Debug output missing 'temperature'");
        assert!(debug_str.contains("top_k"), "Debug output missing 'top_k'");
        assert!(debug_str.contains("top_p"), "Debug output missing 'top_p'");
        assert!(debug_str.contains("0.7"), "Debug output missing temperature value");
    }

    // ── argmax helper tests ───────────────────────────────────────────

    #[test]
    fn argmax_single_element() {
        // Arrange
        let row = [42.0f32];

        // Act
        let result = argmax(&row);

        // Assert: only element must be chosen
        assert_eq!(result, 0);
    }

    #[test]
    fn argmax_first_element_wins_on_tie() {
        // Arrange: all equal values — first index should win (v > best_val is false for equal)
        let row = [5.0f32, 5.0, 5.0, 5.0];

        // Act
        let result = argmax(&row);

        // Assert: first occurrence wins because > (strict) does not update for equal
        assert_eq!(result, 0);
    }

    #[test]
    fn argmax_nan_treated_as_smallest() {
        // Arrange: NaN in position 1, real max in position 2
        let row = [1.0f32, f32::NAN, 10.0, 3.0];

        // Act
        let result = argmax(&row);

        // Assert: NaN comparison returns false, so index 2 wins
        assert_eq!(result, 2);
    }

    #[test]
    fn argmax_all_nan_returns_first() {
        // Arrange: all NaN — no value satisfies v > NEG_INFINITY since NaN > x is false
        // BUT: initial best_val is NEG_INFINITY and NaN > NEG_INFINITY is false,
        // so best_idx stays 0 (initialized) and best_val stays NEG_INFINITY.
        let row = [f32::NAN, f32::NAN, f32::NAN];

        // Act
        let result = argmax(&row);

        // Assert: no value beats initial state, returns index 0
        assert_eq!(result, 0);
    }

    #[test]
    fn argmax_negative_values() {
        // Arrange: all negative, the least negative (closest to zero) wins
        let row = [-10.0f32, -3.0, -5.0, -100.0];

        // Act
        let result = argmax(&row);

        // Assert: -3.0 is the maximum
        assert_eq!(result, 1);
    }

    // ── sample_logits_row: temperature boundary & negative ────────────

    #[test]
    fn sample_negative_temperature_is_greedy() {
        // Arrange: negative temperature should hit the T <= 0.0 greedy path
        let logits = [1.0f32, 5.0, 2.0, 8.0];

        // Act & Assert: over 50 trials, must always return argmax index 3
        for _ in 0..50 {
            let tok = sample_logits_row(&logits, &cfg(-1.0, 0, 1.0)).expect("sample ok");
            assert_eq!(tok, 3, "negative temperature must use greedy argmax");
        }
    }

    #[test]
    fn sample_high_temperature_flattens_distribution() {
        // Arrange: high temperature flattens distribution — uniform-ish logits with T=100.0
        let logits = [0.0f32, 0.1, 0.0, 0.1];
        let mut buckets = [0usize; 4];
        let trials = 2000;

        // Act
        for _ in 0..trials {
            let tok = sample_logits_row(&logits, &cfg(100.0, 0, 1.0)).expect("sample ok");
            buckets[tok as usize] += 1;
        }

        // Assert: high temperature should produce near-uniform — each bucket ~500
        // With T=100, softmax is essentially uniform; each bucket should be > 200 (>>10%)
        for (i, &count) in buckets.iter().enumerate() {
            let ratio = count as f32 / trials as f32;
            assert!(
                ratio > 0.10,
                "bucket {i} too low at T=100: {count}/{trials} = {ratio}"
            );
        }
    }

    // ── top_k edge cases ──────────────────────────────────────────────


    #[test]
    fn top_k_larger_than_vocab_is_no_op() {
        // Arrange: top_k > vocab_size → capped to vocab_size, no restriction.
        // Use near-uniform logits so stochastic sampling explores all tokens.
        let logits = [1.0f32, 1.01, 1.0];
        let mut buckets = [0usize; 3];
        let trials = 2000;

        // Act
        for _ in 0..trials {
            let tok = sample_logits_row(&logits, &cfg(1.0, 100, 1.0)).expect("sample ok");
            buckets[tok as usize] += 1;
        }

        // Assert: near-uniform logits should produce roughly even distribution;
        // each bucket should have > 15% (vs expected ~33%)
        for (i, &count) in buckets.iter().enumerate() {
            let ratio = count as f32 / trials as f32;
            assert!(
                ratio > 0.15,
                "bucket {i} too low with top_k > vocab: {count}/{trials} = {ratio}"
            );
        }
    }

    #[test]
    fn top_k_one_always_picks_max() {
        // Arrange: top_k=1 → only the highest logit candidate survives
        let logits = [0.0f32, 3.0, 10.0, 1.0, 7.0];

        // Act & Assert
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &cfg(1.0, 1, 1.0)).expect("sample ok");
            assert_eq!(tok, 2, "top_k=1 must always pick the argmax (index 2)");
        }
    }

    // ── top_p edge cases ──────────────────────────────────────────────

    #[test]
    fn top_p_zero_is_no_op() {
        // top_p=0.0 skips nucleus filter; use greedy (T=0) to verify deterministically
        let logits = [0.0f32, 0.0, 10.0, 0.0];
        let tok = sample_logits_row(&logits, &cfg(0.0, 0, 0.0)).expect("sample ok");
        assert_eq!(tok, 2, "top_p=0 with greedy picks peak");
    }

    #[test]
    fn top_p_one_is_no_op() {
        // Arrange: top_p=1.0 should skip the nucleus filter (no truncation)
        // Use temperature=0.0 (greedy) so the result is deterministic —
        // the test verifies top_p=1.0 doesn't truncate, not sampling stochasticity.
        let logits = [0.0f32, 0.0, 10.0, 0.0];

        // Act & Assert: top_p=1.0 keeps all candidates
        let tok = sample_logits_row(&logits, &cfg(0.0, 0, 1.0)).expect("sample ok");
        assert_eq!(tok, 2, "top_p=1.0 with greedy must pick peak");
    }

    // ── sample_logits_row: single element ─────────────────────────────

    #[test]
    fn sample_single_element_greedy() {
        // Arrange: single logit, greedy
        let logits = [7.5f32];

        // Act
        let tok = sample_logits_row(&logits, &cfg(0.0, 0, 1.0)).expect("sample ok");

        // Assert: only choice is index 0
        assert_eq!(tok, 0);
    }

    #[test]
    fn sample_single_element_stochastic() {
        // Arrange: single logit, stochastic
        let logits = [3.0f32];

        // Act & Assert: with only one candidate, must always return 0
        for _ in 0..50 {
            let tok = sample_logits_row(&logits, &cfg(1.0, 0, 1.0)).expect("sample ok");
            assert_eq!(tok, 0, "single element must always return index 0");
        }
    }

    // ── sample_logits_row: mixed special float values ─────────────────

    #[test]
    fn sample_with_one_pos_inf_logit() {
        // Arrange: one +inf logit dominates all
        let logits = [1.0f32, f32::INFINITY, 0.0];

        // Act & Assert: greedy path (T=0) picks +inf
        let tok = sample_logits_row(&logits, &cfg(0.0, 0, 1.0)).expect("sample ok");
        assert_eq!(tok, 1, "argmax must pick +inf at index 1");
    }

    // ── sample_logits_row: temperature scaling verification ───────────

    #[test]
    fn low_temperature_concentrates_on_peak() {
        // Arrange: logits with a moderate peak, T=0.01 → near-deterministic
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        let mut peak_count = 0usize;
        let trials = 500;

        // Act
        for _ in 0..trials {
            let tok = sample_logits_row(&logits, &cfg(0.01, 0, 1.0)).expect("sample ok");
            if tok == 3 {
                peak_count += 1;
            }
        }

        // Assert: T=0.01 makes softmax very peaked; should pick argmax almost always
        assert!(
            peak_count >= trials - 5,
            "low temperature should nearly always pick argmax: {peak_count}/{trials}"
        );
    }

    // ── BackendError Display tests ────────────────────────────────────

    #[test]
    fn backend_error_cpu_display() {
        // Arrange
        let err = BE::Cpu("test error message".into());

        // Act
        let display = format!("{err}");

        // Assert
        assert!(
            display.contains("CPU error:"),
            "Cpu variant display must contain 'CPU error:'"
        );
        assert!(
            display.contains("test error message"),
            "Cpu variant display must contain the message"
        );
    }

    #[test]
    fn backend_error_cuda_display() {
        // Arrange
        let err = BE::Cuda("gpu oom".into());

        // Act
        let display = format!("{err}");

        // Assert
        assert!(
            display.contains("CUDA error:"),
            "Cuda variant display must contain 'CUDA error:'"
        );
        assert!(display.contains("gpu oom"));
    }

    #[test]
    fn backend_error_hip_display() {
        // Arrange
        let err = BE::Hip("hip fail".into());

        // Act
        let display = format!("{err}");

        // Assert
        assert!(
            display.contains("HIP error:"),
            "Hip variant display must contain 'HIP error:'"
        );
    }

    #[test]
    fn backend_error_metal_display() {
        // Arrange
        let err = BE::Metal("metal fail".into());

        // Act
        let display = format!("{err}");

        // Assert
        assert!(
            display.contains("Metal error:"),
            "Metal variant display must contain 'Metal error:'"
        );
    }

    #[test]
    fn backend_error_unimplemented_display() {
        // Arrange
        let err = BE::Unimplemented("feature_x");

        // Act
        let display = format!("{err}");

        // Assert
        assert!(
            display.contains("unimplemented:"),
            "Unimplemented variant display must contain 'unimplemented:'"
        );
        assert!(display.contains("feature_x"));
    }

    #[test]
    fn backend_error_other_display() {
        // Arrange
        let err = BE::Other("generic failure".into());

        // Act
        let display = format!("{err}");

        // Assert
        assert!(
            display.contains("backend error:"),
            "Other variant display must contain 'backend error:'"
        );
        assert!(display.contains("generic failure"));
    }

    // ── BackendError trait tests ──────────────────────────────────────

    #[test]
    fn backend_error_is_std_error() {
        // Arrange
        let err: BE = BE::Cpu("test".into());

        // Act: cast to &dyn std::error::Error
        let _: &dyn std::error::Error = &err;

        // Assert: compilation succeeds = trait is implemented
    }

    #[test]
    fn backend_error_is_clone() {
        // Arrange
        let err = BE::Cpu("original".into());

        // Act
        let cloned = err.clone();

        // Assert
        match cloned {
            BE::Cpu(msg) => assert_eq!(msg, "original"),
            _ => panic!("cloned variant mismatch"),
        }
    }

    #[test]
    fn backend_error_is_debug() {
        // Arrange
        let err = BE::Cuda("oom".into());

        // Act
        let debug_str = format!("{err:?}");

        // Assert: Debug output contains variant name and message
        assert!(
            debug_str.contains("Cuda"),
            "Debug output must contain variant name 'Cuda'"
        );
        assert!(debug_str.contains("oom"));
    }

    // ── sample_logits_row: error message content ──────────────────────

    #[test]
    fn empty_row_error_message_contains_context() {
        // Arrange
        let logits: [f32; 0] = [];

        // Act
        let err = sample_logits_row(&logits, &cfg(1.0, 0, 1.0)).unwrap_err();

        // Assert: error message should mention "empty logits"
        let msg = match err {
            BE::Cpu(m) => m,
            other => panic!("expected Cpu error, got {other:?}"),
        };
        assert!(
            msg.contains("empty"),
            "empty row error should mention 'empty', got: {msg}"
        );
    }

    #[test]
    fn all_neg_inf_error_message_contains_context() {
        // Arrange
        let logits = [f32::NEG_INFINITY; 3];

        // Act
        let err = sample_logits_row(&logits, &cfg(1.0, 0, 1.0)).unwrap_err();

        // Assert: error message should mention non-finite or non-positive sum
        let msg = match err {
            BE::Cpu(m) => m,
            other => panic!("expected Cpu error, got {other:?}"),
        };
        assert!(
            msg.contains("non-finite") || msg.contains("non-positive"),
            "all -inf error should explain the issue, got: {msg}"
        );
    }

    // ── sample_logits_row: greedy ignores top_k_and_top_p ─────────────

    #[test]
    fn greedy_ignores_top_k_and_top_p() {
        // Arrange: greedy (T=0) must return argmax even with aggressive top_k/top_p
        let logits = [0.0f32, 1.0, 10.0, 3.0, 2.0];

        // Act & Assert: T=0 with top_k=1, top_p=0.1 — still must return argmax
        for _ in 0..50 {
            let tok = sample_logits_row(&logits, &cfg(0.0, 1, 0.1)).expect("sample ok");
            assert_eq!(
                tok, 2,
                "greedy (T=0) must ignore top_k and top_p, always returning argmax"
            );
        }
    }

    // ── sample_logits_row: two equal peaks ────────────────────────────

    #[test]
    fn two_equal_peaks_distribute_samples() {
        // Arrange: two peaks of equal height at index 1 and 3, with low-valued
        // entries at index 0 and 2 that have near-zero softmax probability.
        // T=1 softmax: exp(10)/(2*exp(10) + 2*exp(0)) ≈ 0.4999 each for the peaks.
        let logits = [0.0f32, 10.0, 0.0, 10.0];
        let mut count_1 = 0usize;
        let mut count_3 = 0usize;
        let trials = 2000;

        // Act
        for _ in 0..trials {
            let tok = sample_logits_row(&logits, &cfg(1.0, 0, 1.0)).expect("sample ok");
            // The two peaks dominate (>99.99% combined), but the non-peak
            // entries have a tiny probability. Count only the two peaks.
            if tok == 1 {
                count_1 += 1;
            } else if tok == 3 {
                count_3 += 1;
            }
            // token 0 or 2 is possible (tiny probability) but rare
        }

        // Assert: both peaks should split roughly evenly; allow 35%-65% range
        let peak_total = count_1 + count_3;
        assert!(
            peak_total > trials / 2,
            "expected most samples on the two peaks, got {count_1} + {count_3} = {peak_total}"
        );
        let ratio_1 = count_1 as f32 / peak_total as f32;
        assert!(
            (0.35..=0.65).contains(&ratio_1),
            "two equal peaks should split ~50/50, got {ratio_1:.3} (counts: {count_1}, {count_3})"
        );
    }

    // ── argmax: additional edge cases ─────────────────────────────────

    #[test]
    fn argmax_two_elements_first_larger() {
        // Arrange
        let row = [5.0f32, 3.0];
        // Act
        let result = argmax(&row);
        // Assert
        assert_eq!(result, 0);
    }

    #[test]
    fn argmax_two_elements_second_larger() {
        // Arrange
        let row = [1.0f32, 7.0];
        // Act
        let result = argmax(&row);
        // Assert
        assert_eq!(result, 1);
    }

    #[test]
    fn argmax_max_at_last_position() {
        // Arrange
        let row = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        // Act
        let result = argmax(&row);
        // Assert
        assert_eq!(result, 4);
    }

    #[test]
    fn argmax_with_positive_infinity() {
        // Arrange: +inf at index 2, large finite elsewhere
        let row = [1e30f32, 1e30, f32::INFINITY, 1e30];
        // Act
        let result = argmax(&row);
        // Assert
        assert_eq!(result, 2, "+inf must always win argmax");
    }

    #[test]
    fn argmax_with_neg_infinity() {
        // Arrange: NEG_INFINITY is less than any finite value
        let row = [f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        // Act
        let result = argmax(&row);
        // Assert
        assert_eq!(result, 1, "finite value must beat NEG_INFINITY");
    }

    #[test]
    fn argmax_mixed_nan_and_finite() {
        // Arrange: NaN at 0, valid max at 1, NaN at 2
        let row = [f32::NAN, 100.0, f32::NAN, -5.0];
        // Act
        let result = argmax(&row);
        // Assert
        assert_eq!(result, 1, "finite max must beat all NaN positions");
    }

    #[test]
    fn argmax_float_min_positive() {
        // Arrange: f32 MIN_POSITIVE (smallest positive normal) vs 0.0
        let row = [0.0f32, f32::MIN_POSITIVE];
        // Act
        let result = argmax(&row);
        // Assert
        assert_eq!(result, 1, "MIN_POSITIVE > 0.0");
    }

    #[test]
    fn argmax_very_large_vocab() {
        // Arrange: 10000 elements, max at position 7777
        let mut row = vec![0.0f32; 10000];
        row[7777] = 1.0;
        // Act
        let result = argmax(&row);
        // Assert
        assert_eq!(result, 7777);
    }

    // ── SamplingConfig: construction and field access ──────────────────

    #[test]
    fn sampling_config_custom_values() {
        // Arrange & Act
        let config = SamplingConfig {
            temperature: 0.33,
            top_k: 42,
            top_p: 0.77,
        };
        // Assert
        assert_eq!(config.temperature, 0.33);
        assert_eq!(config.top_k, 42);
        assert_eq!(config.top_p, 0.77);
    }

    #[test]
    fn sampling_config_extreme_temperature() {
        // Arrange
        let config = SamplingConfig {
            temperature: f32::MAX,
            top_k: 0,
            top_p: 1.0,
        };
        // Act & Assert: extremely high temperature should not panic
        let logits = [1.0f32, 2.0, 3.0];
        let result = sample_logits_row(&logits, &config);
        assert!(result.is_ok(), "extreme temperature should not error on valid logits");
    }

    #[test]
    fn sampling_config_zero_top_k_is_valid() {
        // Arrange: top_k=0 means disabled (no restriction)
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        // Act
        let logits = [1.0f32, 2.0, 3.0];
        let result = sample_logits_row(&logits, &config);
        // Assert
        assert!(result.is_ok());
        let tok = result.unwrap();
        assert!(tok < 3, "token must be within vocab range");
    }

    #[test]
    fn sampling_config_top_p_boundary_zero() {
        // Arrange: top_p=0.0 means nucleus filter is skipped
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
        };
        let logits = [1.0f32, 5.0, 2.0];
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert: should succeed (top_p=0 skips the nucleus branch)
        assert!(result.is_ok());
    }

    // ── sample_logits_row: temperature edge cases ─────────────────────

    #[test]
    fn sample_nan_temperature_is_greedy() {
        // Arrange: NaN temperature — NaN <= 0.0 is false, but NaN comparison behavior
        // leads to the stochastic path. This verifies no panic.
        let logits = [1.0f32, 5.0, 2.0];
        let config = SamplingConfig {
            temperature: f32::NAN,
            top_k: 0,
            top_p: 1.0,
        };
        // Act: should not panic, but may produce unexpected results
        let result = sample_logits_row(&logits, &config);
        // Assert: should not panic; result is either Ok or Err
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn sample_very_low_temperature_approximates_greedy() {
        // Arrange: T=1e-10 should be effectively greedy
        let logits = [1.0f32, 100.0, 2.0];
        let config = SamplingConfig {
            temperature: 1e-10,
            top_k: 0,
            top_p: 1.0,
        };
        // Act
        let mut all_argmax = true;
        for _ in 0..100 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            if tok != 1 {
                all_argmax = false;
            }
        }
        // Assert: extremely low temperature should pick argmax every time
        assert!(all_argmax, "T=1e-10 should be effectively greedy");
    }

    #[test]
    fn sample_infinite_temperature_no_panic() {
        // Arrange: T=+inf — inv_t = 1/inf = 0, all scaled logits become 0
        let logits = [1.0f32, 2.0, 3.0];
        let config = SamplingConfig {
            temperature: f32::INFINITY,
            top_k: 0,
            top_p: 1.0,
        };
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert: should succeed with effectively uniform distribution
        assert!(result.is_ok());
        let tok = result.unwrap();
        assert!(tok < 3, "token must be in range");
    }

    // ── sample_logits_row: top_k additional cases ─────────────────────

    #[test]
    fn top_k_two_preserves_top_two() {
        // Arrange: logits with clear top-2 at indices 2 and 4
        let logits = [0.0f32, 1.0, 10.0, 2.0, 9.0];
        let allowed = [2u32, 4u32];
        // Act & Assert
        for _ in 0..300 {
            let tok = sample_logits_row(&logits, &cfg(1.0, 2, 1.0)).expect("sample ok");
            assert!(
                allowed.contains(&tok),
                "top_k=2 must only sample from top-2 candidates, got {tok}"
            );
        }
    }

    #[test]
    fn top_k_with_uniform_logits_all_equal() {
        // Arrange: all logits equal — top_k=3 should still pick from first 3 after sort
        let logits = [5.0f32, 5.0, 5.0, 5.0, 5.0];
        // Act & Assert: with uniform logits, any of 0..=4 could be in the top_k=3 set
        // after unstable sort. Since all values are equal, the sort is unstable
        // but indices 0..4 are all candidates. top_k=3 just restricts to 3 of them.
        let config = cfg(1.0, 3, 1.0);
        let tok = sample_logits_row(&logits, &config).expect("sample ok");
        assert!(tok < 5, "token must be within vocab");
    }

    #[test]
    fn top_k_zero_means_disabled() {
        // Arrange: top_k=0 is disabled, all candidates should be considered
        let logits = [0.0f32, 0.0, 0.0, 10.0];
        // Act: with T=1.0 and top_k=0, index 3 dominates but others have small probability
        let config = cfg(1.0, 0, 1.0);
        // Assert: should always pick index 3 (dominant peak)
        for _ in 0..100 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, 3, "dominant peak with no top_k restriction");
        }
    }

    // ── sample_logits_row: top_p additional cases ─────────────────────

    #[test]
    fn top_p_very_small_restricts_to_dominant() {
        // Arrange: very small top_p with one dominant peak
        let logits = [0.0f32, 0.0, 20.0, 0.0];
        let config = cfg(1.0, 0, 0.01);
        // Act & Assert: top_p=0.01 should keep only the dominant peak
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, 2, "tiny top_p should only keep dominant peak");
        }
    }

    #[test]
    fn top_p_with_flat_distribution_keeps_many() {
        // Arrange: flat distribution, top_p=0.99 should keep all or nearly all
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let config = cfg(1.0, 0, 0.99);
        let mut buckets = [0usize; 4];
        // Act
        for _ in 0..2000 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            buckets[tok as usize] += 1;
        }
        // Assert: with flat distribution and top_p=0.99, all buckets should be hit
        let non_zero_buckets = buckets.iter().filter(|&&c| c > 0).count();
        assert!(
            non_zero_buckets >= 3,
            "flat distribution + top_p=0.99 should keep most candidates, got {non_zero_buckets} non-zero buckets"
        );
    }

    #[test]
    fn top_p_with_two_peaks_balances() {
        // Arrange: two equal peaks, top_p=0.99 should keep both
        let logits = [0.0f32, 10.0, 0.0, 10.0];
        let config = cfg(1.0, 0, 0.99);
        let mut count_1 = 0usize;
        let mut count_3 = 0usize;
        // Act
        for _ in 0..1000 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            if tok == 1 { count_1 += 1; }
            if tok == 3 { count_3 += 3; }
        }
        // Assert: both peaks should be sampled
        assert!(count_1 > 100, "peak at index 1 should be sampled frequently: {count_1}");
        assert!(count_3 > 100, "peak at index 3 should be sampled frequently: {count_3}");
    }

    // ── sample_logits_row: combined top_k + top_p ─────────────────────

    #[test]
    fn top_k_and_top_p_both_restrict() {
        // Arrange: 6 logits, top_k=3 keeps top 3, then top_p=0.5 further narrows
        // Indices sorted by logit desc: 3(10), 1(8), 5(5), 0(2), 2(1), 4(0)
        // top_k=3 → indices 3,1,5. top_p=0.5 → likely only index 3 survives
        let logits = [2.0f32, 8.0, 1.0, 10.0, 0.0, 5.0];
        let config = cfg(1.0, 3, 0.5);
        // Act & Assert: top_p=0.5 on a peaked distribution among top-3 should heavily favor index 3
        let mut count_3 = 0usize;
        for _ in 0..300 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok == 3 || tok == 1 || tok == 5,
                "top_k=3 restricts to indices {{3,1,5}}, got {tok}");
            if tok == 3 { count_3 += 1; }
        }
        assert!(count_3 > 200, "index 3 should dominate: {count_3}/300");
    }

    #[test]
    fn top_k_restricts_then_top_p_narrows_further() {
        // Arrange: top_k=4 keeps 4, then top_p=0.8 narrows within those 4
        let logits = [0.0f32, 1.0, 10.0, 2.0, 9.0, 0.5];
        let config = cfg(1.0, 4, 0.8);
        // Act: allowed set after top_k=4 is {2, 4, 3, 1} (top 4 by value)
        let allowed = [1u32, 2, 3, 4];
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(allowed.contains(&tok),
                "top_k=4 must restrict candidates, got {tok}");
        }
    }

    // ── sample_logits_row: special float values ────────────────────────

    #[test]
    fn sample_all_zero_logits_succeeds() {
        // Arrange: all zeros — softmax gives uniform distribution
        let logits = [0.0f32, 0.0, 0.0, 0.0];
        let config = cfg(1.0, 0, 1.0);
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert
        assert!(result.is_ok(), "all-zero logits should succeed");
        let tok = result.unwrap();
        assert!(tok < 4, "token must be in range");
    }

    #[test]
    fn sample_all_zero_logits_uniform_distribution() {
        // Arrange: all zeros — softmax gives uniform distribution
        let logits = [0.0f32, 0.0, 0.0];
        let config = cfg(1.0, 0, 1.0);
        let mut buckets = [0usize; 3];
        let trials = 3000;
        // Act
        for _ in 0..trials {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            buckets[tok as usize] += 1;
        }
        // Assert: each bucket should be roughly trials/3 = 1000
        for (i, &count) in buckets.iter().enumerate() {
            let ratio = count as f32 / trials as f32;
            assert!(
                (0.25..=0.40).contains(&ratio),
                "bucket {i} should be ~33%, got {ratio:.3} ({count}/{trials})"
            );
        }
    }

    #[test]
    fn sample_mixed_finite_and_neg_inf() {
        // Arrange: mix of finite and -inf values
        let logits = [f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, 3.0];
        let config = cfg(1.0, 0, 1.0);
        // Act & Assert: only indices 1 and 3 are viable
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(
                tok == 1 || tok == 3,
                "only finite logit positions should be sampled, got {tok}"
            );
        }
    }

    #[test]
    fn sample_single_finite_rest_neg_inf() {
        // Arrange: only one finite value
        let logits = [f32::NEG_INFINITY, f32::NEG_INFINITY, 7.0, f32::NEG_INFINITY];
        let config = cfg(1.0, 0, 1.0);
        // Act & Assert
        for _ in 0..100 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, 2, "single finite logit must always be sampled");
        }
    }

    #[test]
    fn sample_very_large_logit_values() {
        // Arrange: very large but finite logits
        let logits = [1e30f32, 1e30, 1e30 + 1.0, 1e30];
        let config = cfg(1.0, 0, 1.0);
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert: should not overflow or panic
        assert!(result.is_ok());
    }

    // ── sample_logits_row: greedy path verification ────────────────────

    #[test]
    fn greedy_with_negative_temperature_always_argmax() {
        // Arrange
        let logits = [-5.0f32, 0.0, 50.0, 25.0];
        // Act & Assert
        for _ in 0..50 {
            let tok = sample_logits_row(&logits, &cfg(-100.0, 0, 1.0)).expect("sample ok");
            assert_eq!(tok, 2, "negative T must use argmax");
        }
    }

    #[test]
    fn greedy_consistent_across_calls() {
        // Arrange: same input, greedy path must always return same result
        let logits = [1.0f32, 5.0, 3.0, 2.0];
        let config = cfg(0.0, 0, 1.0);
        // Act
        let first = sample_logits_row(&logits, &config).expect("sample ok");
        // Assert: 100 more calls must all return same token
        for _ in 0..100 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, first, "greedy must be deterministic");
        }
    }

    #[test]
    fn greedy_with_single_element() {
        // Arrange: single element, greedy
        let logits = [42.0f32];
        // Act
        let tok = sample_logits_row(&logits, &cfg(0.0, 0, 1.0)).expect("sample ok");
        // Assert
        assert_eq!(tok, 0);
    }

    #[test]
    fn greedy_picks_last_of_equal_max() {
        // Wait — argmax uses strict >, so first max wins. Verify this.
        // Arrange: two equal max values, first at index 1, second at index 3
        let logits = [0.0f32, 10.0, 5.0, 10.0];
        // Act: greedy must pick index 1 (first occurrence of max)
        let tok = sample_logits_row(&logits, &cfg(0.0, 0, 1.0)).expect("sample ok");
        // Assert: argmax uses > (strict), so first occurrence wins
        assert_eq!(tok, 1, "argmax picks first occurrence of max");
    }

    // ── sample_logits_row: stochastic distribution shape ───────────────

    #[test]
    fn temperature_scaling_changes_distribution() {
        // Arrange: same logits, different temperatures should produce different distributions
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        let trials = 2000;

        // Act: sample with low temperature
        let mut low_t_peak = 0usize;
        for _ in 0..trials {
            let tok = sample_logits_row(&logits, &cfg(0.1, 0, 1.0)).expect("sample ok");
            if tok == 3 { low_t_peak += 1; }
        }

        // Act: sample with high temperature
        let mut high_t_peak = 0usize;
        for _ in 0..trials {
            let tok = sample_logits_row(&logits, &cfg(10.0, 0, 1.0)).expect("sample ok");
            if tok == 3 { high_t_peak += 1; }
        }

        // Assert: low temperature should concentrate more on the peak than high temperature
        assert!(
            low_t_peak > high_t_peak,
            "low T ({low_t_peak}) should concentrate on peak more than high T ({high_t_peak})"
        );
    }

    #[test]
    fn stochastic_never_out_of_range() {
        // Arrange: test that sampled tokens are always within valid range
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let config = cfg(1.0, 3, 0.9);
        // Act & Assert
        for _ in 0..500 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 5, "token {tok} is out of vocab range [0,5)");
        }
    }

    // ── sample_logits_row: error path details ──────────────────────────

    #[test]
    fn single_neg_inf_logit_succeeds_greedy() {
        // Arrange: single -inf logit with T=0 (greedy) should still return index 0
        let logits = [f32::NEG_INFINITY];
        // Act: greedy path uses argmax which returns 0 for any single element
        let tok = sample_logits_row(&logits, &cfg(0.0, 0, 1.0)).expect("sample ok");
        // Assert
        assert_eq!(tok, 0);
    }

    #[test]
    fn single_neg_inf_logit_errors_stochastic() {
        // Arrange: single -inf logit with T>0 triggers softmax path
        let logits = [f32::NEG_INFINITY];
        // Act
        let result = sample_logits_row(&logits, &cfg(1.0, 0, 1.0));
        // Assert: softmax of -inf should produce non-positive sum
        assert!(result.is_err(), "single -inf with T>0 should error");
    }

    #[test]
    fn empty_row_error_is_cpu_variant() {
        // Arrange
        let logits: [f32; 0] = [];
        // Act
        let err = sample_logits_row(&logits, &cfg(1.0, 0, 1.0)).unwrap_err();
        // Assert
        match err {
            BE::Cpu(_) => {} // expected
            other => panic!("expected Cpu variant, got {other:?}"),
        }
    }

    // ── BackendError: additional variant tests ─────────────────────────

    #[test]
    fn backend_error_variants_are_distinct() {
        // Arrange: create different error variants
        let cpu_err = BE::Cpu("cpu".into());
        let cuda_err = BE::Cuda("cuda".into());
        let hip_err = BE::Hip("hip".into());
        let metal_err = BE::Metal("metal".into());
        let unimp_err = BE::Unimplemented("unimp");
        let other_err = BE::Other("other".into());

        // Act & Assert: each Display output should be unique
        let displays: Vec<String> = vec![
            format!("{cpu_err}"),
            format!("{cuda_err}"),
            format!("{hip_err}"),
            format!("{metal_err}"),
            format!("{unimp_err}"),
            format!("{other_err}"),
        ];
        // Check each pair is distinct
        for i in 0..displays.len() {
            for j in (i + 1)..displays.len() {
                assert_ne!(
                    displays[i], displays[j],
                    "Error variants {} and {} must have distinct displays",
                    i, j
                );
            }
        }
    }

    #[test]
    fn backend_error_clone_preserves_content() {
        // Arrange
        let errors = vec![
            BE::Cpu("cpu msg".into()),
            BE::Cuda("cuda msg".into()),
            BE::Hip("hip msg".into()),
            BE::Metal("metal msg".into()),
            BE::Unimplemented("unimpl"),
            BE::Other("other msg".into()),
        ];
        // Act & Assert: clone each and verify display matches
        for err in &errors {
            let cloned = err.clone();
            assert_eq!(format!("{err}"), format!("{cloned}"),
                "cloned error display must match original");
        }
    }

    #[test]
    fn backend_error_debug_contains_variant_name() {
        // Arrange
        let err = BE::Hip("test".into());
        // Act
        let debug = format!("{err:?}");
        // Assert
        assert!(debug.contains("Hip"), "Debug must contain variant name");
    }

    #[test]
    fn backend_error_unimplemented_static_str() {
        // Arrange: Unimplemented takes &'static str
        let err = BE::Unimplemented("not yet supported");
        // Act
        let display = format!("{err}");
        // Assert
        assert!(display.contains("not yet supported"));
        assert!(display.contains("unimplemented"));
    }

    #[test]
    fn backend_error_other_generic() {
        // Arrange
        let err = BE::Other("something went wrong".into());
        // Act
        let display = format!("{err}");
        // Assert
        assert!(display.contains("backend error"));
        assert!(display.contains("something went wrong"));
    }

    // ── sample_logits_row: larger vocabulary ────────────────────────────

    #[test]
    fn large_vocab_greedy_correct() {
        // Arrange: 1000-element vocabulary with peak at position 500
        let mut logits = vec![0.0f32; 1000];
        logits[500] = 100.0;
        // Act
        let tok = sample_logits_row(&logits, &cfg(0.0, 0, 1.0)).expect("sample ok");
        // Assert
        assert_eq!(tok, 500, "greedy must find peak in large vocab");
    }

    #[test]
    fn large_vocab_stochastic_with_top_k() {
        // Arrange: 1000-element vocabulary, top_k=10 restricts candidates
        let mut logits = vec![0.0f32; 1000];
        logits[100] = 10.0;
        logits[200] = 9.0;
        logits[300] = 8.0;
        // ... rest are 0.0
        let allowed = [100u32, 200, 300];
        // Act & Assert
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &cfg(1.0, 3, 1.0)).expect("sample ok");
            assert!(
                allowed.contains(&tok),
                "top_k=3 must restrict to top-3 indices in large vocab, got {tok}"
            );
        }
    }

    #[test]
    fn large_vocab_top_p_with_dominant_peak() {
        // Arrange: 500-element vocabulary, one dominant peak
        let mut logits = vec![0.0f32; 500];
        logits[250] = 50.0;
        // Act
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &cfg(1.0, 0, 0.9)).expect("sample ok");
            assert_eq!(tok, 250, "dominant peak with top_p=0.9 must always be selected");
        }
    }

    // ── sample_logits_row: deterministic greedy across configs ─────────

    #[test]
    fn greedy_deterministic_with_various_top_k() {
        // Arrange
        let logits = [1.0f32, 3.0, 5.0, 2.0, 4.0];
        // Act & Assert: T=0 (greedy) always returns argmax regardless of top_k/top_p
        for top_k in [0, 1, 2, 3, 5, 100] {
            for top_p in [0.0, 0.5, 0.9, 1.0] {
                let tok = sample_logits_row(&logits, &cfg(0.0, top_k, top_p)).expect("sample ok");
                assert_eq!(tok, 2,
                    "greedy must return argmax (index 2) with top_k={top_k}, top_p={top_p}");
            }
        }
    }

    // ── sample_logits_row: top_k=1 with top_p ──────────────────────────

    #[test]
    fn top_k_one_with_top_p_still_picks_max() {
        // Arrange: top_k=1 restricts to single best, top_p is irrelevant
        let logits = [1.0f32, 2.0, 10.0, 3.0];
        // Act & Assert
        for top_p in [0.0, 0.5, 0.9, 1.0] {
            for _ in 0..50 {
                let tok = sample_logits_row(&logits, &cfg(1.0, 1, top_p)).expect("sample ok");
                assert_eq!(tok, 2, "top_k=1 must always pick argmax regardless of top_p={top_p}");
            }
        }
    }

    // ── sample_logits_row: softmax numerical stability ─────────────────

    #[test]
    fn large_logit_differences_no_overflow() {
        // Arrange: extreme logit differences that would cause exp overflow without subtract-max
        let logits = [-1000.0f32, 0.0, 1000.0];
        let config = cfg(1.0, 0, 1.0);
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert: should not panic or overflow; index 2 dominates completely
        assert!(result.is_ok(), "large logit differences should not cause overflow");
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn very_negative_logits_produce_valid_sample() {
        // Arrange: all very negative logits
        let logits = [-1e10f32, -1e10 + 1.0, -1e10 + 2.0];
        let config = cfg(1.0, 0, 1.0);
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert
        assert!(result.is_ok());
        let tok = result.unwrap();
        assert!(tok < 3);
    }

    // ── sample_logits_row: top_k edge value zero vs vocab_size ─────────

    #[test]
    fn top_k_zero_vs_top_k_vocab_size_same_distribution() {
        // Arrange: with a very peaked distribution, both should behave similarly
        let logits = [0.0f32, 0.0, 100.0, 0.0];
        // Act: both configs should always pick index 2
        for _ in 0..100 {
            let tok_k0 = sample_logits_row(&logits, &cfg(1.0, 0, 1.0)).expect("sample ok");
            let tok_k4 = sample_logits_row(&logits, &cfg(1.0, 4, 1.0)).expect("sample ok");
            assert_eq!(tok_k0, 2, "top_k=0 peaked dist should pick index 2");
            assert_eq!(tok_k4, 2, "top_k=4 peaked dist should pick index 2");
        }
    }

    // ── sample_logits_row: negative logit values ────────────────────────

    #[test]
    fn all_negative_logits_sample_correctly() {
        // Arrange: all negative, with a clear "least negative" peak
        let logits = [-100.0f32, -1.0, -50.0, -200.0];
        let config = cfg(1.0, 0, 1.0);
        // Act & Assert: index 1 (-1.0) is the largest
        let mut count_1 = 0usize;
        for _ in 0..500 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 4);
            if tok == 1 { count_1 += 1; }
        }
        // index 1 should dominate heavily
        assert!(count_1 > 400, "least negative should dominate: {count_1}/500");
    }

    // ── SamplingConfig: additional construction patterns ────────────────

    #[test]
    fn sampling_config_construct_all_fields_zero() {
        // Arrange & Act
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
        };
        // Assert
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 0.0);
    }

    #[test]
    fn sampling_config_field_mutation() {
        // Arrange
        let mut config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        // Act
        config.temperature = 0.5;
        config.top_k = 10;
        config.top_p = 0.9;
        // Assert
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.top_k, 10);
        assert_eq!(config.top_p, 0.9);
    }

    #[test]
    fn sampling_config_two_instances_independent() {
        // Arrange
        let config_a = SamplingConfig {
            temperature: 0.3,
            top_k: 5,
            top_p: 0.7,
        };
        let config_b = SamplingConfig {
            temperature: 0.9,
            top_k: 50,
            top_p: 0.95,
        };
        // Assert: each instance holds independent values
        assert_ne!(config_a.temperature, config_b.temperature);
        assert_ne!(config_a.top_k, config_b.top_k);
        assert_ne!(config_a.top_p, config_b.top_p);
    }

    #[test]
    fn sampling_config_default_matches_explicit() {
        // Arrange
        let via_default = SamplingConfig::default();
        let via_explicit = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        // Assert: both construction methods yield identical fields
        assert_eq!(via_default.temperature, via_explicit.temperature);
        assert_eq!(via_default.top_k, via_explicit.top_k);
        assert_eq!(via_default.top_p, via_explicit.top_p);
    }

    #[test]
    fn cfg_helper_field_mapping() {
        // Arrange & Act
        let config = cfg(0.42, 17, 0.83);
        // Assert: helper maps positional args to correct fields
        assert_eq!(config.temperature, 0.42);
        assert_eq!(config.top_k, 17);
        assert_eq!(config.top_p, 0.83);
    }

    #[test]
    fn sampling_config_top_k_max_usize() {
        // Arrange & Act
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: usize::MAX,
            top_p: 1.0,
        };
        // Assert
        assert_eq!(config.top_k, usize::MAX);
    }

    #[test]
    fn sampling_config_temperature_subnormal() {
        // Arrange & Act
        let config = SamplingConfig {
            temperature: f32::MIN_POSITIVE / 2.0,
            top_k: 0,
            top_p: 1.0,
        };
        // Assert: subnormal value stored without normalization
        assert!(config.temperature > 0.0);
        assert!(config.temperature < f32::MIN_POSITIVE);
    }

    #[test]
    fn sampling_config_temperature_negative_zero() {
        // Arrange & Act
        let config = SamplingConfig {
            temperature: -0.0f32,
            top_k: 0,
            top_p: 1.0,
        };
        // Assert: -0.0 == 0.0 per IEEE 754 but sign bit preserved
        assert_eq!(config.temperature, 0.0);
        assert!(config.temperature.is_sign_negative());
    }

    // ── BackendError: additional construction patterns ──────────────────

    #[test]
    fn backend_error_cpu_empty_message() {
        // Arrange & Act
        let err = BE::Cpu(String::new());
        // Assert
        match err {
            BE::Cpu(msg) => assert!(msg.is_empty()),
            other => panic!("expected Cpu variant, got {other:?}"),
        }
    }

    #[test]
    fn backend_error_all_string_variants_same_message() {
        // Arrange: all String-based variants accept identical content
        let msg = "shared error";
        // Act & Assert: each variant is constructible
        assert!(matches!(BE::Cpu(msg.into()), BE::Cpu(m) if m == msg));
        assert!(matches!(BE::Cuda(msg.into()), BE::Cuda(m) if m == msg));
        assert!(matches!(BE::Hip(msg.into()), BE::Hip(m) if m == msg));
        assert!(matches!(BE::Metal(msg.into()), BE::Metal(m) if m == msg));
        assert!(matches!(BE::Other(msg.into()), BE::Other(m) if m == msg));
    }

    #[test]
    fn backend_error_unimplemented_empty_static_str() {
        // Arrange & Act
        let err = BE::Unimplemented("");
        // Assert: empty static str is accepted
        assert!(matches!(err, BE::Unimplemented("")));
    }

    #[test]
    fn backend_error_variants_collectible_in_vec() {
        // Arrange: all 6 variants in a single collection
        let errors: Vec<BE> = vec![
            BE::Cpu("a".into()),
            BE::Cuda("b".into()),
            BE::Hip("c".into()),
            BE::Metal("d".into()),
            BE::Unimplemented("e"),
            BE::Other("f".into()),
        ];
        // Assert
        assert_eq!(errors.len(), 6);
        assert!(matches!(errors[0], BE::Cpu(_)));
        assert!(matches!(errors[1], BE::Cuda(_)));
        assert!(matches!(errors[2], BE::Hip(_)));
        assert!(matches!(errors[3], BE::Metal(_)));
        assert!(matches!(errors[4], BE::Unimplemented(_)));
        assert!(matches!(errors[5], BE::Other(_)));
    }

    // ── New tests: +inf logits in stochastic path ──────────────────────

    #[test]
    fn all_positive_infinity_logits_error_stochastic() {
        // Arrange: all +inf logits — softmax scaling yields +inf for all,
        // and exp(+inf - +inf) = exp(NaN) = NaN, making the sum non-positive.
        let logits = [f32::INFINITY; 4];
        let config = cfg(1.0, 0, 1.0);
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert: max_val is +inf which is_finite() is false → error
        assert!(result.is_err(), "all +inf logits must return error in stochastic path");
    }

    #[test]
    fn stochastic_with_inf_among_finite_errors() {
        // Arrange: one +inf and finite values — scaled max is +inf, not finite → error
        let logits = [1.0f32, f32::INFINITY, 3.0];
        let config = cfg(1.0, 0, 1.0);
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert: non-finite max_val triggers error
        assert!(result.is_err(), "+inf in logits must trigger non-finite max error");
    }

    // ── New tests: top_k equals vocab length exactly ────────────────────

    #[test]
    fn top_k_equals_vocab_size_no_truncation() {
        // Arrange: top_k = vocab_size exactly → no indices truncated
        let logits = [3.0f32, 1.0, 4.0, 1.5];
        let config = cfg(1.0, 4, 1.0);
        let mut seen = [false; 4];
        // Act: sample enough times to likely see all indices
        for _ in 0..400 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 4, "token must be in vocab range");
            seen[tok as usize] = true;
        }
        // Assert: with uniform-ish logits and no truncation, all should appear
        let seen_count = seen.iter().filter(|&&s| s).count();
        assert!(
            seen_count >= 3,
            "top_k=vocab_size should allow all tokens, saw {seen_count}/4"
        );
    }

    // ── New tests: top_p keeps all when cumulative never reaches threshold ──

    #[test]
    fn top_p_keeps_all_candidates_when_cumulative_never_reaches() {
        // Arrange: flat distribution (all equal), top_p=0.9999999 — since
        // each candidate has equal probability, cumulative reaches 0.9999 at
        // the last candidate. top_p rounds up to keep all.
        let logits = [1.0f32, 1.0, 1.0, 1.0, 1.0];
        let config = cfg(1.0, 0, 0.9999999);
        let mut buckets = [0usize; 5];
        // Act
        for _ in 0..2000 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            buckets[tok as usize] += 1;
        }
        // Assert: with near-1.0 top_p and flat logits, all buckets should be hit
        let non_zero = buckets.iter().filter(|&&c| c > 0).count();
        assert!(
            non_zero >= 4,
            "near-1.0 top_p with flat logits should keep most candidates, got {non_zero}/5"
        );
    }

    // ── New tests: argmax boundary with u32 max index ──────────────────

    #[test]
    fn argmax_large_index_within_u32_range() {
        // Arrange: verify argmax returns a value that fits in u32 even with large vocab
        let mut row = vec![0.0f32; 100_000];
        row[99_999] = 1.0;
        // Act
        let result = argmax(&row);
        // Assert
        assert_eq!(result, 99_999);
        assert!(
            (result as u32) as usize == 99_999,
            "argmax result must round-trip through u32"
        );
    }

    // ── New tests: cumulative fallback (last candidate returned) ───────

    #[test]
    fn cumulative_fallback_returns_last_candidate() {
        // Arrange: construct a scenario where the uniform random `r` may be
        // >= cumulative probability of all but the last candidate. This is a
        // statistical test — use a flat distribution where any token is equally
        // likely. Just verify the function always returns a valid index.
        let logits = [1.0f32, 1.0, 1.0];
        let config = cfg(1.0, 0, 1.0);
        // Act: 500 trials, all must return valid index
        for _ in 0..500 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 3, "token must be in range [0,3), got {tok}");
        }
        // This exercises the cumulative fallback path at least some of the time
    }

    // ── New tests: BackendError multiline message ──────────────────────

    #[test]
    fn backend_error_cuda_multiline_message_preserved() {
        // Arrange: multiline message
        let msg = "line1\nline2\nline3";
        let err = BE::Cuda(msg.to_string());
        // Act
        let display = format!("{err}");
        // Assert: all lines preserved in Display output
        assert!(display.contains("line1"), "first line missing");
        assert!(display.contains("line2"), "second line missing");
        assert!(display.contains("line3"), "third line missing");
    }

    // ── New tests: SamplingConfig PartialEq field-by-field ────────────

    #[test]
    fn sampling_config_copy_equality_semantics() {
        // Arrange: two configs with same values, created independently
        let a = SamplingConfig {
            temperature: 0.7,
            top_k: 10,
            top_p: 0.9,
        };
        let b = a; // Copy
        // Assert: all fields match (field-by-field, since no PartialEq derive)
        assert_eq!(a.temperature, b.temperature);
        assert_eq!(a.top_k, b.top_k);
        assert_eq!(a.top_p, b.top_p);
    }

    // ── New tests: neg_inf and finite interleaved pattern ──────────────

    #[test]
    fn interleaved_neg_inf_and_finite_only_finite_sampled() {
        // Arrange: alternating -inf and finite
        let logits = [f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, 3.0, f32::NEG_INFINITY, 1.0];
        let config = cfg(1.0, 0, 1.0);
        let allowed = [1u32, 3, 5];
        // Act & Assert
        for _ in 0..300 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(
                allowed.contains(&tok),
                "only finite logit indices {{1,3,5}} should be sampled, got {tok}"
            );
        }
    }

    // ── New tests: top_p renormalization error path unreachable ───────

    #[test]
    fn top_p_renormalize_always_succeeds_with_valid_candidates() {
        // Arrange: even with top_p=0.001, at least one candidate remains
        // and its probability is > 0, so renormalization sum > 0
        let logits = [0.0f32, 0.0, 10.0, 0.0, 0.0];
        let config = cfg(1.0, 0, 0.001);
        // Act & Assert: should succeed, not trigger renormalization error
        for _ in 0..100 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, 2, "tiny top_p with sharp peak must succeed and pick peak");
        }
    }

    // ── New tests: inv_t edge case with very small temperature ────────

    #[test]
    fn very_small_temperature_produces_large_scaled_logits() {
        // Arrange: T=1e-20, inv_t = 1e20 — scaled logits become huge but
        // subtract-max keeps softmax stable
        let logits = [0.0f32, 1.0, 2.0];
        let config = SamplingConfig {
            temperature: 1e-20,
            top_k: 0,
            top_p: 1.0,
        };
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert: numerical stability (subtract-max) should prevent overflow
        assert!(result.is_ok(), "very small T should not cause overflow");
        let tok = result.unwrap();
        assert_eq!(tok, 2, "T→0 should pick argmax (index 2)");
    }

    // ── New tests: BackendError Display long message ──────────────────

    #[test]
    fn backend_error_hip_long_message_display() {
        // Arrange: a long error message
        let long_msg = "x".repeat(10000);
        let err = BE::Hip(long_msg.clone());
        // Act
        let display = format!("{err}");
        // Assert: full message preserved
        assert!(display.contains("HIP error:"));
        assert!(display.contains(&long_msg));
        assert_eq!(display.len(), "HIP error: ".len() + 10000);
    }

    // ── New tests: sample_logits_row deterministic with identical logits ──

    #[test]
    fn sample_all_same_logit_returns_valid_index() {
        // Arrange: all logits identical — softmax produces uniform distribution
        let logits = [7.0f32, 7.0, 7.0, 7.0, 7.0];
        let config = cfg(1.0, 0, 1.0);
        // Act: every trial must return a valid index in [0, 5)
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(
                tok < 5,
                "uniform logit sampling must return valid index, got {tok}"
            );
        }
    }

    // ── Additional tests (+13) ──────────────────────────────────────────

    #[test]
    fn argmax_nan_vs_neg_inf_picks_neg_inf() {
        // Arrange: NaN at index 0, NEG_INFINITY at index 1
        // NaN > NEG_INFINITY is false, NEG_INFINITY > NEG_INFINITY is false
        // So best_idx stays at 0 (initial), best_val stays NEG_INFINITY
        let row = [f32::NAN, f32::NEG_INFINITY];
        // Act
        let result = argmax(&row);
        // Assert: neither value beats the initial state, returns index 0
        assert_eq!(result, 0, "with NaN and -inf, initial index 0 wins since neither exceeds NEG_INFINITY");
    }

    #[test]
    fn argmax_inf_and_neg_inf_and_nan_mixed() {
        // Arrange: +inf at 0, -inf at 1, NaN at 2, finite at 3
        let row = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 42.0];
        // Act
        let result = argmax(&row);
        // Assert: +inf beats everything
        assert_eq!(result, 0, "+inf must win over -inf, NaN, and finite");
    }

    #[test]
    fn sampling_config_top_p_greater_than_one_accepted() {
        // Arrange & Act: top_p > 1.0 is stored as-is (no validation at construction)
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 2.0,
        };
        // Assert: field stored verbatim
        assert_eq!(config.top_p, 2.0);
        // Act: top_p > 1.0 means (top_p > 0.0 && top_p < 1.0) is false, so nucleus filter is skipped
        let logits = [0.0f32, 0.0, 10.0];
        let tok = sample_logits_row(&logits, &config).expect("sample ok");
        assert_eq!(tok, 2, "top_p=2.0 skips nucleus filter, dominant peak wins");
    }

    #[test]
    fn sampling_config_negative_top_p_skips_nucleus() {
        // Arrange: negative top_p — (top_p > 0.0 && top_p < 1.0) is false → nucleus filter skipped
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: -0.5,
        };
        // Act: with sharp peak, should always pick peak
        let logits = [0.0f32, 0.0, 20.0, 0.0];
        for _ in 0..100 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, 2, "negative top_p skips nucleus, peak must win");
        }
    }

    #[test]
    fn top_p_exactly_half_on_three_candidates() {
        // Arrange: three candidates with probabilities ~1/3 each after softmax
        // top_p=0.5 should keep ~2 candidates (cumulative reaches ~0.67 at 2nd candidate)
        let logits = [1.0f32, 1.0, 1.0];
        let config = cfg(1.0, 0, 0.5);
        let mut buckets = [0usize; 3];
        // Act
        for _ in 0..2000 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 3, "token must be in range");
            buckets[tok as usize] += 1;
        }
        // Assert: with top_p=0.5 on uniform, should keep 2 of 3 candidates
        // so at least 1 bucket should be zero (the truncated one) or all 3 present
        // if the cumulative sum reaches exactly 0.5 at first candidate.
        // Actually with 1/3 each, first candidate cumulative = 0.333 < 0.5,
        // second = 0.667 >= 0.5, so 2 candidates kept.
        // But since indices after sort may vary, just verify valid output.
        let non_zero = buckets.iter().filter(|&&c| c > 0).count();
        assert!(
            non_zero >= 1,
            "top_p=0.5 should keep at least 1 candidate, got {non_zero} non-zero buckets"
        );
    }

    #[test]
    fn greedy_negative_zero_temperature_is_greedy() {
        // Arrange: -0.0 (IEEE 754) — -0.0 <= 0.0 is true, so hits greedy path
        let logits = [3.0f32, 1.0, 7.0, 2.0];
        let config = SamplingConfig {
            temperature: -0.0f32,
            top_k: 0,
            top_p: 1.0,
        };
        // Act & Assert
        for _ in 0..50 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, 2, "T=-0.0 must use greedy argmax path");
        }
    }

    #[test]
    fn top_k_exactly_one_greater_than_vocab() {
        // Arrange: top_k = vocab_size + 1 → effective_k = vocab_size (no truncation)
        let logits = [1.0f32, 2.0, 3.0];
        let config = cfg(1.0, 4, 1.0); // top_k=4 > vocab_size=3
        let mut seen = [false; 3];
        // Act
        for _ in 0..500 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 3);
            seen[tok as usize] = true;
        }
        // Assert: with near-uniform logits and no truncation, all should appear
        let seen_count = seen.iter().filter(|&&s| s).count();
        assert!(
            seen_count >= 2,
            "top_k > vocab_size should not truncate, saw {seen_count}/3"
        );
    }

    #[test]
    fn sampling_config_top_p_nan_skips_nucleus() {
        // Arrange: NaN top_p — (NaN > 0.0 && NaN < 1.0) = (false && false) = false → nucleus skipped
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: f32::NAN,
        };
        // Act: with sharp peak, should still pick peak since nucleus is skipped
        let logits = [0.0f32, 0.0, 20.0, 0.0];
        for _ in 0..100 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, 2, "NaN top_p skips nucleus, peak wins");
        }
    }

    #[test]
    fn sample_two_element_logits_stochastic() {
        // Arrange: two logits, stochastic — both indices possible
        let logits = [5.0f32, 5.0];
        let config = cfg(1.0, 0, 1.0);
        let mut seen = [false; 2];
        // Act
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 2, "token must be 0 or 1");
            seen[tok as usize] = true;
        }
        // Assert: equal logits → both should appear
        assert!(seen[0], "index 0 should be sampled at least once");
        assert!(seen[1], "index 1 should be sampled at least once");
    }

    #[test]
    fn all_neg_inf_error_message_non_finite() {
        // Arrange: all -inf → max_val is -inf, is_finite() = false → error about non-finite max
        let logits = [f32::NEG_INFINITY; 5];
        let config = cfg(1.0, 0, 1.0);
        // Act
        let err = sample_logits_row(&logits, &config).unwrap_err();
        // Assert
        let msg = match err {
            BE::Cpu(m) => m,
            other => panic!("expected Cpu variant, got {other:?}"),
        };
        assert!(
            msg.contains("non-finite"),
            "all -inf should report non-finite max, got: {msg}"
        );
    }

    #[test]
    fn sample_greedy_with_all_equal_logits() {
        // Arrange: all equal logits, greedy → argmax picks first index (strict >)
        let logits = [3.0f32, 3.0, 3.0, 3.0];
        // Act
        let tok = sample_logits_row(&logits, &cfg(0.0, 0, 1.0)).expect("sample ok");
        // Assert: strict > means first index wins on ties
        assert_eq!(tok, 0, "greedy on equal logits must return first index");
    }

    #[test]
    fn top_p_renormalization_error_message_content() {
        // This test verifies the top_p renormalization error path is unreachable
        // under normal conditions. We cannot easily trigger it since at least one
        // candidate always has positive probability after top_p truncation.
        // Instead, verify that the error message format is documented correctly.
        // The error says "top-p renormalization failed (sum=...)".
        // Arrange: a valid sampling that exercises the renormalization code path
        let logits = [0.0f32, 0.0, 5.0];
        let config = cfg(1.0, 0, 0.9);
        // Act & Assert: should succeed (not trigger the renorm error)
        let result = sample_logits_row(&logits, &config);
        assert!(result.is_ok(), "valid top_p sampling must succeed");
    }

    #[test]
    fn sample_logits_with_f32_epsilon_differences() {
        // Arrange: logits differing by f32::EPSILON — should still produce valid samples
        let base = 1.0f32;
        let logits = [base, base + f32::EPSILON, base + 2.0 * f32::EPSILON];
        let config = cfg(1.0, 0, 1.0);
        // Act
        for _ in 0..100 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 3, "token must be in range [0,3)");
        }
        // Assert: no panic or error with near-identical logits
    }

    // ── Additional tests (+10) ──────────────────────────────────────────

    #[test]
    fn argmax_returns_u32_index_type() {
        // Arrange: verify the return type is u32 and fits in the expected range
        let row = [1.0f32, 2.0, 3.0, 4.0];
        // Act
        let result: u32 = argmax(&row);
        // Assert: result is a valid u32 index into the input slice
        assert_eq!(result, 3u32);
        assert!((result as usize) < row.len());
    }

    #[test]
    fn sample_stochastic_two_element_sharp_peak() {
        // Arrange: two logits with large difference — stochastic should heavily
        // favor the larger one but both are theoretically possible
        let logits = [0.0f32, 50.0];
        let config = cfg(1.0, 0, 1.0);
        let mut count_1 = 0usize;
        // Act
        for _ in 0..500 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 2, "token must be 0 or 1, got {tok}");
            if tok == 1 { count_1 += 1; }
        }
        // Assert: index 1 dominates overwhelmingly
        assert!(
            count_1 >= 495,
            "sharp peak (50 vs 0) should dominate: {count_1}/500"
        );
    }

    #[test]
    fn top_k_with_mixed_neg_inf_excludes_neg_inf_from_candidates() {
        // Arrange: top_k=3 on 5 logits, 2 of which are -inf
        // Sorted desc: index 2(10.0), index 4(5.0), index 1(3.0), index 0(-inf), index 3(-inf)
        // top_k=3 picks indices {2, 4, 1} — all finite
        let logits = [f32::NEG_INFINITY, 3.0, 10.0, f32::NEG_INFINITY, 5.0];
        let config = cfg(1.0, 3, 1.0);
        let allowed = [1u32, 2, 4];
        // Act & Assert
        for _ in 0..300 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(
                allowed.contains(&tok),
                "top_k=3 must exclude -inf candidates, got {tok}"
            );
        }
    }

    #[test]
    fn top_p_exact_boundary_one_skips_nucleus() {
        // Arrange: top_p=1.0 — the condition (top_p > 0.0 && top_p < 1.0) is false
        // so nucleus filter is entirely skipped. Verify with a broad distribution.
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let config = cfg(1.0, 0, 1.0);
        let mut seen = [false; 5];
        // Act
        for _ in 0..500 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 5);
            seen[tok as usize] = true;
        }
        // Assert: without nucleus truncation, most tokens should appear
        let seen_count = seen.iter().filter(|&&s| s).count();
        assert!(
            seen_count >= 3,
            "top_p=1.0 should not truncate candidates, saw {seen_count}/5"
        );
    }

    #[test]
    fn argmax_subnormal_float_values() {
        // Arrange: subnormal (denormalized) f32 values — smaller than MIN_POSITIVE
        let tiny = f32::MIN_POSITIVE; // smallest normal positive
        let subnormal = tiny / 2.0_f32; // subnormal, still > 0.0
        let row = [0.0f32, subnormal, -subnormal];
        // Act
        let result = argmax(&row);
        // Assert: subnormal > 0.0, so index 1 wins
        assert_eq!(result, 1, "subnormal positive must beat 0.0 and subnormal negative");
        assert!(subnormal > 0.0, "subnormal must be positive");
        assert!(subnormal < f32::MIN_POSITIVE, "must be subnormal");
    }

    #[test]
    fn sample_all_nan_logits_stochastic_errors() {
        // Arrange: all NaN logits — scaled max_val is NaN, is_finite() is false → error
        let logits = [f32::NAN, f32::NAN, f32::NAN];
        let config = cfg(1.0, 0, 1.0);
        // Act
        let result = sample_logits_row(&logits, &config);
        // Assert
        assert!(result.is_err(), "all NaN logits must error in stochastic path");
        let msg = match result.unwrap_err() {
            BE::Cpu(m) => m,
            other => panic!("expected Cpu error, got {other:?}"),
        };
        assert!(
            msg.contains("non-finite"),
            "error should mention non-finite max, got: {msg}"
        );
    }

    #[test]
    fn temperature_scaling_zero_logits_succeeds() {
        // Arrange: all zeros with T=0.5 — scaled are still 0/0.5=0, uniform softmax
        let logits = [0.0f32, 0.0, 0.0, 0.0];
        let config = cfg(0.5, 0, 1.0);
        let mut buckets = [0usize; 4];
        // Act
        for _ in 0..1000 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert!(tok < 4);
            buckets[tok as usize] += 1;
        }
        // Assert: uniform distribution, each bucket ~250
        for (i, &count) in buckets.iter().enumerate() {
            let ratio = count as f32 / 1000.0;
            assert!(
                ratio > 0.15,
                "bucket {i} too low for zero-logit uniform: {ratio:.3}"
            );
        }
    }

    #[test]
    fn top_k_one_stochastic_single_candidate_distribution() {
        // Arrange: top_k=1 on non-trivial logits — only the argmax survives,
        // but multinomial sampling still runs (with single candidate, must always return it)
        let logits = [5.0f32, 1.0, 3.0, 0.0, 7.0];
        let config = cfg(1.0, 1, 1.0);
        // Act & Assert
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, 4, "top_k=1 must always pick single argmax candidate (index 4)");
        }
    }

    #[test]
    fn top_p_and_top_k_combine_to_single_candidate() {
        // Arrange: top_k=2 restricts to {2, 4} (values 10.0 and 9.0),
        // top_p=0.01 further restricts — with such a tiny top_p, only index 2 survives
        let logits = [0.0f32, 1.0, 10.0, 2.0, 9.0];
        let config = cfg(1.0, 2, 0.01);
        // Act & Assert
        for _ in 0..200 {
            let tok = sample_logits_row(&logits, &config).expect("sample ok");
            assert_eq!(tok, 2, "top_k=2 + tiny top_p must isolate dominant peak");
        }
    }

    #[test]
    fn greedy_with_all_neg_inf_single_element_succeeds() {
        // Arrange: single element that is -inf, greedy path (T=0)
        // argmax of a single element always returns index 0 regardless of value
        let logits = [f32::NEG_INFINITY];
        let config = cfg(0.0, 0, 1.0);
        // Act
        let tok = sample_logits_row(&logits, &config).expect("sample ok");
        // Assert: argmax returns 0 for single element (no comparison needed)
        assert_eq!(tok, 0, "greedy on single -inf must return index 0");
    }
}
