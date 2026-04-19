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
    let mut candidates: Vec<(usize, f32)> = indices.into_iter().zip(probs.into_iter()).collect();
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
}
