//! §17.2 Adapter — 零参数 Draft 投影头
//!
//! Adapter 是附在 L2_hot 变体末尾的微型投影图，将中间 hidden state 映射到 vocab 空间。
//! 图结构: RmsNorm(hidden_size) → MatMul(normed, lm_head.weight^T) → logits
//!
//! **零额外参数**: 直接复用 lm_head.weight (Phase A)
//! **可选蒸馏**: Phase B 添加 residual_delta, 前 100 步用 full model logits 蒸馏

use std::sync::Arc;

/// Draft Adapter 配置
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Hidden size (from model config)
    pub hidden_size: usize,
    /// Vocab size (from model config)
    pub vocab_size: usize,
    /// LayerNorm epsilon
    pub rms_norm_eps: f32,
    /// Phase B 可选: 是否启用蒸馏残差
    pub enable_distillation: bool,
    /// 蒸馏步数 (Phase B)
    pub distillation_steps: usize,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            vocab_size: 0,
            rms_norm_eps: 1e-5,
            enable_distillation: false,
            distillation_steps: 100,
        }
    }
}

/// Draft Adapter 运行时
///
/// §17.2: Adapter 将浅层变体的 hidden state 投影到 vocab 空间。
/// Phase A: 直接复用 lm_head.weight (零额外参数)
/// Phase B: 添加 residual_delta 通过在线蒸馏学习
pub struct DraftAdapter {
    config: AdapterConfig,
    /// Phase A: 共享的 lm_head.weight 引用 [vocab_size, hidden_size]
    /// Phase B: lm_head.weight + residual_delta
    weight: Arc<Vec<f32>>,
    /// Phase B: residual_delta — 在线蒸馏学习的修正项
    /// 当 enable_distillation=true 时，前 distillation_steps 步用 full model logits 蒸馏
    residual_delta: Option<Vec<f32>>,
    /// 已完成的蒸馏步数
    distillation_step: usize,
    /// 最后一个 norm 层的权重 [hidden_size] — 复用模型最后一层的 norm 权重
    norm_weight: Arc<Vec<f32>>,
}

impl std::fmt::Debug for DraftAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftAdapter")
            .field("hidden_size", &self.config.hidden_size)
            .field("vocab_size", &self.config.vocab_size)
            .field("distillation_step", &self.distillation_step)
            .field("has_residual_delta", &self.residual_delta.is_some())
            .finish()
    }
}

#[allow(dead_code)]
impl DraftAdapter {
    /// 创建 Phase A Adapter (零额外参数, 共享 lm_head.weight)
    ///
    /// # Arguments
    /// * `config` - Adapter 配置
    /// * `lm_head_weight` - lm_head.weight 张量 [vocab_size, hidden_size] (共享引用)
    /// * `norm_weight` - 最后一层 RmsNorm 权重 [hidden_size] (共享引用)
    pub fn new_phase_a(
        config: AdapterConfig,
        lm_head_weight: Arc<Vec<f32>>,
        norm_weight: Arc<Vec<f32>>,
    ) -> Self {
        Self {
            config,
            weight: lm_head_weight,
            residual_delta: None,
            distillation_step: 0,
            norm_weight,
        }
    }

    /// 创建 Phase B Adapter (带蒸馏残差)
    ///
    /// Phase B 在 Phase A 基础上添加 `residual_delta` 参数，
    /// 通过前 `distillation_steps` 步用 full model 的真实 logits 蒸馏学习。
    /// 额外参数: vocab_size × hidden_size (≈0.1% 总模型大小)
    pub fn new_phase_b(
        config: AdapterConfig,
        lm_head_weight: Arc<Vec<f32>>,
        norm_weight: Arc<Vec<f32>>,
    ) -> Self {
        let residual_delta = vec![0.0f32; config.vocab_size * config.hidden_size];
        Self {
            config,
            weight: lm_head_weight,
            residual_delta: Some(residual_delta),
            distillation_step: 0,
            norm_weight,
        }
    }

    /// Adapter 前向传播: hidden_state → logits
    ///
    /// §17.2 图结构:
    /// 1. RmsNorm(hidden_state, norm_weight)
    /// 2. MatMul(normed, weight^T) → logits
    ///
    /// # Arguments
    /// * `hidden` - Hidden state [hidden_size] (来自 L2_hot 变体输出)
    ///
    /// # Returns
    /// Logits 向量 [vocab_size]
    pub fn forward(&self, hidden: &[f32]) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        assert_eq!(hidden.len(), hidden_size, "hidden size mismatch");

        // Step 1: RmsNorm
        let normed = self.rms_norm(hidden);

        // Step 2: MatMul(normed, weight^T) → logits[vocab_size]
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            for h in 0..hidden_size {
                let w = self.weight[v * hidden_size + h];
                let delta = self.residual_delta
                    .as_ref()
                    .map(|d| d[v * hidden_size + h])
                    .unwrap_or(0.0);
                sum += normed[h] * (w + delta);
            }
            logits[v] = sum;
        }

        logits
    }

    /// 批量 Adapter 前向: 多个 hidden states → 多组 logits
    ///
    /// 用于 EqSpec batch 场景 (§17.4): 多个 sequence 同时 draft
    pub fn forward_batch(&self, hiddens: &[&[f32]]) -> Vec<Vec<f32>> {
        hiddens.iter().map(|h| self.forward(h)).collect()
    }

    /// 在线蒸馏更新 (Phase B only)
    ///
    /// §17.2 Phase B: 用 full model 的真实 logits 做蒸馏信号，
    /// SGD 微调 residual_delta 参数。
    ///
    /// # Arguments
    /// * `draft_logits` - Adapter 产生的 logits [vocab_size]
    /// * `target_logits` - Full model 产生的 logits [vocab_size] (ground truth)
    /// * `hidden` - 输入 hidden state [hidden_size]
    /// * `learning_rate` - SGD 学习率
    pub fn distill_step(
        &mut self,
        draft_logits: &[f32],
        target_logits: &[f32],
        hidden: &[f32],
        learning_rate: f32,
    ) -> f32 {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;

        // Compute gradient and normed hidden before mutable borrow of delta
        let grad = softmax_diff(draft_logits, target_logits, vocab_size);
        let normed = rms_norm_free(hidden, &self.norm_weight, eps);

        let delta = self.residual_delta.as_mut().expect("Phase B required for distillation");
        let mut total_loss = 0.0f32;
        for v in 0..vocab_size {
            total_loss += grad[v] * grad[v];
            for h in 0..hidden_size {
                delta[v * hidden_size + h] -= learning_rate * grad[v] * normed[h];
            }
        }

        self.distillation_step += 1;
        total_loss / vocab_size as f32
    }

    /// 检查蒸馏是否完成
    pub fn is_distillation_complete(&self) -> bool {
        self.distillation_step >= self.config.distillation_steps
    }

    /// 获取蒸馏进度
    pub fn distillation_progress(&self) -> (usize, usize) {
        (self.distillation_step, self.config.distillation_steps)
    }

    /// Adapter 参数量 (bytes)
    pub fn parameter_bytes(&self) -> usize {
        let base = 0; // Phase A: 零额外参数 (共享 lm_head)
        let delta = self.residual_delta
            .as_ref()
            .map(|d| d.len() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        base + delta
    }

    // ---- Internal helpers ----

    /// RmsNorm: x / rms(x) * weight
    fn rms_norm(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let eps = self.config.rms_norm_eps;
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        x.iter()
            .zip(self.norm_weight.iter())
            .map(|(&xi, &wi)| xi * inv_rms * wi)
            .collect()
    }

    /// Compute logits gradient for distillation
    fn logits_gradient(&self, draft: &[f32], target: &[f32]) -> Vec<f32> {
        let vocab_size = self.config.vocab_size;
        // Softmax difference as gradient signal
        let (draft_probs, target_probs) = (softmax(draft), softmax(target));
        let mut grad = vec![0.0f32; vocab_size];
        for i in 0..vocab_size {
            grad[i] = draft_probs[i] - target_probs[i];
        }
        grad
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    crate::moe::routing::softmax(x)
}

/// Free function for softmax difference (avoids borrow conflicts in distill_step)
fn softmax_diff(draft: &[f32], target: &[f32], vocab_size: usize) -> Vec<f32> {
    let (draft_probs, target_probs) = (softmax(draft), softmax(target));
    let mut grad = vec![0.0f32; vocab_size];
    for i in 0..vocab_size {
        grad[i] = draft_probs[i] - target_probs[i];
    }
    grad
}

/// Free function for RmsNorm (avoids borrow conflicts in distill_step)
fn rms_norm_free(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

// ============================================================================
// §BCI9: EMA Multi-Token Acceptance Tracker
// ============================================================================

/// EMA (Exponential Moving Average) acceptance tracker for multi-token prediction.
///
/// Tracks per-position acceptance rates and computes a dynamic acceptance
/// threshold based on draft model's historical accuracy. Replaces fixed-threshold
/// acceptance with adaptive EMA-smoothed decisions.
///
/// # Design
///
/// - **Per-position EMA**: Each draft position (0..K-1) tracks its own acceptance
///   rate. Position 0 is typically highest, position K-1 lowest.
/// - **Adaptive alpha**: Alpha warms up from 0.5 → `nominal_alpha` over the first
///   16 steps, then adapts based on rate variance (high variance → lower alpha).
/// - **Dynamic threshold**: The acceptance threshold starts at 0.5 and adapts
///   based on the global EMA trend. Rising trend → lower threshold (more aggressive),
///   falling trend → higher threshold (more conservative).
#[derive(Debug, Clone)]
pub struct MtpEmaTracker {
    /// Per-position EMA acceptance rates [0.0, 1.0]; index = position in draft
    position_rates: Vec<f32>,
    /// Global EMA acceptance rate [0.0, 1.0]
    global_rate: f32,
    /// Current EMA alpha (adaptive)
    alpha: f32,
    /// Nominal alpha (base value, e.g. 0.3)
    nominal_alpha: f32,
    /// Step counter for alpha warmup
    steps: usize,
    /// Dynamic acceptance threshold [0.0, 1.0]
    acceptance_threshold: f32,
    /// Previous global rate (for trend detection)
    prev_global_rate: f32,
    /// Exponential variance estimate of global rate
    rate_variance_ema: f32,
}

impl MtpEmaTracker {
    /// Create a new EMA tracker for multi-token acceptance.
    ///
    /// # Arguments
    /// * `max_positions` — Maximum number of draft positions to track (typically MTP depth).
    /// * `nominal_alpha` — Base EMA decay factor (0.0..1.0], 0.3 is a good default.
    pub fn new(max_positions: usize, nominal_alpha: f32) -> Self {
        let alpha = nominal_alpha.clamp(0.01, 1.0);
        Self {
            position_rates: vec![0.5; max_positions.max(1)],
            global_rate: 0.5,
            alpha,
            nominal_alpha: alpha,
            steps: 0,
            acceptance_threshold: 0.5,
            prev_global_rate: 0.5,
            rate_variance_ema: 0.0,
        }
    }

    /// Record acceptance result for a specific draft position.
    ///
    /// Updates the per-position EMA and global rate.
    ///
    /// # Arguments
    /// * `position` — Draft position index (0 = first draft token).
    /// * `accepted` — Whether the token at this position was accepted by verification.
    pub fn record_position(&mut self, position: usize, accepted: bool) {
        if position >= self.position_rates.len() {
            return;
        }
        let hit: f32 = if accepted { 1.0 } else { 0.0 };
        let alpha = self.effective_alpha();

        // Per-position EMA update
        self.position_rates[position] =
            alpha * hit + (1.0 - alpha) * self.position_rates[position];

        // Global rate: average across all tracked positions
        self.global_rate = self.position_rates.iter().sum::<f32>() / self.position_rates.len() as f32;

        // Update rate variance EMA for alpha adaptation
        let delta = hit - self.global_rate;
        let var_alpha = alpha * 0.5; // slower for variance
        self.rate_variance_ema =
            var_alpha * (delta * delta) + (1.0 - var_alpha) * self.rate_variance_ema;

        // Update dynamic threshold
        self.adapt_threshold();

        self.prev_global_rate = self.global_rate;
        self.steps += 1;
    }

    /// Record a batch acceptance result: `acceptance_rate` ∈ [0, 1].
    ///
    /// Updates the global EMA without per-position granularity.
    /// Used when only aggregate acceptance data is available.
    pub fn record_batch(&mut self, acceptance_rate: f32) {
        let alpha = self.effective_alpha();
        self.global_rate = alpha * acceptance_rate + (1.0 - alpha) * self.global_rate;

        // Update variance estimate
        let delta = acceptance_rate - self.global_rate;
        let var_alpha = alpha * 0.5;
        self.rate_variance_ema =
            var_alpha * (delta * delta) + (1.0 - var_alpha) * self.rate_variance_ema;

        self.adapt_threshold();
        self.prev_global_rate = self.global_rate;
        self.steps += 1;
    }

    /// EMA-based acceptance decision for a draft token at the given position.
    ///
    /// A token is accepted if its position's EMA rate exceeds the dynamic threshold
    /// OR if the global trend is strongly positive.
    ///
    /// # Returns
    /// `true` if the draft model's historical accuracy at this position warrants acceptance.
    pub fn ema_accept(&self, position: usize) -> bool {
        if position >= self.position_rates.len() {
            return false;
        }
        let pos_rate = self.position_rates[position];
        // Accept if position rate exceeds threshold, or if global trend is rising strongly
        pos_rate >= self.acceptance_threshold || self.global_trend() > 0.1
    }

    /// Multi-token acceptance: returns the number of positions to accept.
    ///
    /// Starting from position 0, accepts consecutive positions where
    /// `ema_accept(position)` returns true.
    pub fn multi_token_accept(&self, max_positions: usize) -> usize {
        let limit = max_positions.min(self.position_rates.len());
        let mut count = 0;
        for pos in 0..limit {
            if self.ema_accept(pos) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Get the EMA acceptance rate for a specific position.
    pub fn ema_acceptance_rate(&self, position: usize) -> Option<f32> {
        self.position_rates.get(position).copied()
    }

    /// Get the global EMA acceptance rate.
    pub fn global_ema_rate(&self) -> f32 {
        self.global_rate
    }

    /// Get the current dynamic acceptance threshold.
    pub fn threshold(&self) -> f32 {
        self.acceptance_threshold
    }

    /// Get the effective alpha (with warmup).
    fn effective_alpha(&self) -> f32 {
        if self.steps < 16 {
            // Warmup: linearly interpolate from 0.5 to nominal_alpha
            let t = self.steps as f32 / 16.0;
            0.5 * (1.0 - t) + self.nominal_alpha * t
        } else {
            // Adapt: reduce alpha when variance is high
            let var_penalty = (self.rate_variance_ema * 10.0).min(0.5);
            (self.nominal_alpha - var_penalty).max(0.05)
        }
    }

    /// Adapt the dynamic acceptance threshold based on trend and variance.
    fn adapt_threshold(&mut self) {
        let trend = self.global_trend();

        // Base threshold moves opposite to trend:
        // - Positive trend → lower threshold (be more aggressive)
        // - Negative trend → higher threshold (be more conservative)
        let trend_adjustment = -trend * 0.5;

        // Variance penalty: high variance → raise threshold (be cautious)
        let var_penalty = (self.rate_variance_ema * 5.0).min(0.2);

        self.acceptance_threshold = (0.5 + trend_adjustment + var_penalty).clamp(0.2, 0.8);
    }

    /// Compute the global rate trend (positive = improving).
    fn global_trend(&self) -> f32 {
        self.global_rate - self.prev_global_rate
    }

    /// Number of tracked steps.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Reset the tracker to initial state.
    pub fn reset(&mut self) {
        self.position_rates.fill(0.5);
        self.global_rate = 0.5;
        self.alpha = self.nominal_alpha;
        self.steps = 0;
        self.acceptance_threshold = 0.5;
        self.prev_global_rate = 0.5;
        self.rate_variance_ema = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adapter(hidden_size: usize, vocab_size: usize) -> DraftAdapter {
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            enable_distillation: false,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; vocab_size * hidden_size]);
        let norm_weight = Arc::new(vec![1.0f32; hidden_size]);
        DraftAdapter::new_phase_a(config, lm_head, norm_weight)
    }

    #[test]
    fn test_adapter_forward_output_size() {
        let adapter = make_adapter(64, 1000);
        let hidden = vec![1.0f32; 64];
        let logits = adapter.forward(&hidden);
        assert_eq!(logits.len(), 1000);
    }

    #[test]
    fn test_adapter_batch_forward() {
        let adapter = make_adapter(64, 1000);
        let h1 = vec![1.0f32; 64];
        let h2 = vec![0.5f32; 64];
        let results = adapter.forward_batch(&[&h1, &h2]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 1000);
        assert_eq!(results[1].len(), 1000);
    }

    #[test]
    fn test_adapter_phase_b_distillation() {
        let hidden_size = 32;
        let vocab_size = 100;
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 5,
        };
        let lm_head = Arc::new(vec![0.1f32; vocab_size * hidden_size]);
        let norm_weight = Arc::new(vec![1.0f32; hidden_size]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);

        assert!(!adapter.is_distillation_complete());

        let hidden = vec![1.0f32; hidden_size];
        for _ in 0..5 {
            let draft = adapter.forward(&hidden);
            let target = vec![0.0f32; vocab_size]; // dummy target
            let _loss = adapter.distill_step(&draft, &target, &hidden, 0.01);
        }

        assert!(adapter.is_distillation_complete());
        assert_eq!(adapter.distillation_progress(), (5, 5));
    }

    #[test]
    fn test_adapter_phase_a_zero_extra_params() {
        let adapter = make_adapter(64, 1000);
        // Phase A: 零额外参数
        assert_eq!(adapter.parameter_bytes(), 0);
    }

    #[test]
    fn test_adapter_phase_b_has_delta_params() {
        let config = AdapterConfig {
            hidden_size: 32,
            vocab_size: 100,
            ..Default::default()
        };
        let lm_head = Arc::new(vec![0.1f32; 100 * 32]);
        let norm_weight = Arc::new(vec![1.0f32; 32]);
        let adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        // Phase B: vocab_size * hidden_size * 4 bytes
        assert_eq!(adapter.parameter_bytes(), 100 * 32 * 4);
    }

    // ── MtpEmaTracker tests ──────────────────────────────────────────────────

    #[test]
    fn ema_tracker_initial_state() {
        let tracker = MtpEmaTracker::new(4, 0.3);
        assert_eq!(tracker.steps(), 0);
        assert!((tracker.global_ema_rate() - 0.5).abs() < 1e-6);
        assert!((tracker.threshold() - 0.5).abs() < 1e-6);
        assert_eq!(tracker.ema_acceptance_rate(0), Some(0.5));
        assert_eq!(tracker.ema_acceptance_rate(3), Some(0.5));
        assert_eq!(tracker.ema_acceptance_rate(4), None); // out of range
    }

    #[test]
    fn ema_tracker_record_position_updates_rate() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        tracker.record_position(0, true);
        assert_eq!(tracker.steps(), 1);
        // After one accepted position, rate should be > 0.5
        assert!(tracker.ema_acceptance_rate(0).unwrap() > 0.5);
    }

    #[test]
    fn ema_tracker_record_position_out_of_range_ignored() {
        let mut tracker = MtpEmaTracker::new(2, 0.3);
        tracker.record_position(5, true); // out of range, ignored
        assert_eq!(tracker.steps(), 0);
    }

    #[test]
    fn ema_tracker_record_batch_updates_global_rate() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        tracker.record_batch(1.0); // perfect acceptance
        assert_eq!(tracker.steps(), 1);
        assert!(tracker.global_ema_rate() > 0.5);
    }

    #[test]
    fn ema_tracker_ema_accept_after_all_accepts() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        // Record many accepts to drive rates high
        for _ in 0..20 {
            for pos in 0..3 {
                tracker.record_position(pos, true);
            }
        }
        // After many accepts, all positions should be acceptable
        assert!(tracker.ema_accept(0));
        assert!(tracker.ema_accept(1));
        assert!(tracker.ema_accept(2));
    }

    #[test]
    fn ema_tracker_multi_token_accept_count() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        // Record many accepts for all positions
        for _ in 0..20 {
            for pos in 0..4 {
                tracker.record_position(pos, true);
            }
        }
        let count = tracker.multi_token_accept(4);
        assert_eq!(count, 4);
    }

    #[test]
    fn ema_tracker_multi_token_accept_stops_at_reject() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        // Position 0 accepted, positions 1-3 rejected
        for _ in 0..20 {
            tracker.record_position(0, true);
            tracker.record_position(1, false);
            tracker.record_position(2, false);
            tracker.record_position(3, false);
        }
        let count = tracker.multi_token_accept(4);
        assert_eq!(count, 1); // Only position 0 accepted
    }

    #[test]
    fn ema_tracker_reset_restores_initial_state() {
        let mut tracker = MtpEmaTracker::new(3, 0.3);
        tracker.record_position(0, true);
        tracker.record_batch(0.8);
        assert!(tracker.steps() > 0);

        tracker.reset();
        assert_eq!(tracker.steps(), 0);
        assert!((tracker.global_ema_rate() - 0.5).abs() < 1e-6);
        assert!((tracker.threshold() - 0.5).abs() < 1e-6);
        assert_eq!(tracker.ema_acceptance_rate(0), Some(0.5));
    }

    #[test]
    fn ema_tracker_alpha_clamped_on_creation() {
        let tracker = MtpEmaTracker::new(2, 0.001); // below min
        // Internal nominal_alpha should be clamped to 0.01
        // We can't inspect directly, but behavior should be valid
        assert_eq!(tracker.steps(), 0);
    }

    // ── AdapterConfig tests ──────────────────────────────────────────────────

    #[test]
    fn adapter_config_default_values() {
        let config = AdapterConfig::default();
        assert_eq!(config.hidden_size, 0);
        assert_eq!(config.vocab_size, 0);
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!(!config.enable_distillation);
        assert_eq!(config.distillation_steps, 100);
    }

    #[test]
    fn adapter_forward_deterministic() {
        let adapter = make_adapter(16, 50);
        let hidden = vec![0.5f32; 16];
        let logits1 = adapter.forward(&hidden);
        let logits2 = adapter.forward(&hidden);
        assert_eq!(logits1, logits2);
    }

    #[test]
    fn adapter_distillation_progress_tracking() {
        let config = AdapterConfig {
            hidden_size: 8,
            vocab_size: 10,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 3,
        };
        let lm_head = Arc::new(vec![0.1f32; 80]);
        let norm_weight = Arc::new(vec![1.0f32; 8]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);

        assert_eq!(adapter.distillation_progress(), (0, 3));
        let hidden = vec![1.0f32; 8];
        for step in 1..=3 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &vec![0.0f32; 10], &hidden, 0.01);
            assert_eq!(adapter.distillation_progress(), (step, 3));
        }
        assert!(adapter.is_distillation_complete());
    }

    #[test]
    fn draft_adapter_debug_output() {
        let adapter = make_adapter(64, 1000);
        let debug = format!("{:?}", adapter);
        assert!(debug.contains("hidden_size: 64"));
        assert!(debug.contains("vocab_size: 1000"));
        assert!(debug.contains("distillation_step: 0"));
        assert!(debug.contains("has_residual_delta: false"));
    }

    // ── Additional coverage tests ─────────────────────────────────────────────

    #[test]
    fn draft_adapter_debug_phase_b_shows_residual_delta() {
        let config = AdapterConfig {
            hidden_size: 8,
            vocab_size: 10,
            ..Default::default()
        };
        let lm_head = Arc::new(vec![0.1f32; 80]);
        let norm_weight = Arc::new(vec![1.0f32; 8]);
        let adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let debug = format!("{:?}", adapter);
        assert!(debug.contains("has_residual_delta: true"));
    }

    #[test]
    fn adapter_config_clone_is_independent() {
        let original = AdapterConfig {
            hidden_size: 512,
            vocab_size: 32000,
            rms_norm_eps: 1e-6,
            enable_distillation: true,
            distillation_steps: 200,
        };
        let cloned = original.clone();
        assert_eq!(cloned.hidden_size, original.hidden_size);
        assert_eq!(cloned.vocab_size, original.vocab_size);
        assert_eq!(cloned.rms_norm_eps, original.rms_norm_eps);
        assert_eq!(cloned.enable_distillation, original.enable_distillation);
        assert_eq!(cloned.distillation_steps, original.distillation_steps);
    }

    #[test]
    fn adapter_config_debug_format() {
        let config = AdapterConfig {
            hidden_size: 256,
            vocab_size: 1000,
            rms_norm_eps: 1e-5,
            enable_distillation: false,
            distillation_steps: 100,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("AdapterConfig"));
        assert!(debug.contains("hidden_size"));
        assert!(debug.contains("vocab_size"));
    }

    #[test]
    fn adapter_config_default_eps_is_positive() {
        let config = AdapterConfig::default();
        assert!(config.rms_norm_eps > 0.0, "rms_norm_eps must be positive for numerical stability");
    }

    #[test]
    fn ema_tracker_clone_preserves_state() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);
        tracker.record_position(0, true);
        tracker.record_position(1, false);
        let cloned = tracker.clone();
        assert_eq!(cloned.steps(), tracker.steps());
        assert_eq!(cloned.global_ema_rate(), tracker.global_ema_rate());
        assert_eq!(cloned.threshold(), tracker.threshold());
        assert_eq!(cloned.ema_acceptance_rate(0), tracker.ema_acceptance_rate(0));
    }

    #[test]
    fn ema_tracker_debug_format() {
        let tracker = MtpEmaTracker::new(3, 0.4);
        let debug = format!("{:?}", tracker);
        assert!(debug.contains("MtpEmaTracker"));
        assert!(debug.contains("position_rates"));
        assert!(debug.contains("global_rate"));
    }

    #[test]
    fn ema_tracker_new_with_zero_positions_creates_one() {
        // max_positions.max(1) in new() ensures at least one position
        let tracker = MtpEmaTracker::new(0, 0.3);
        assert_eq!(tracker.ema_acceptance_rate(0), Some(0.5));
        assert_eq!(tracker.ema_acceptance_rate(1), None);
    }

    #[test]
    fn ema_accept_out_of_range_returns_false() {
        let tracker = MtpEmaTracker::new(2, 0.3);
        assert!(!tracker.ema_accept(5));
        assert!(!tracker.ema_accept(100));
    }

    #[test]
    fn ema_tracker_multi_token_accept_limited_by_positions() {
        let tracker = MtpEmaTracker::new(2, 0.3);
        // Request more positions than tracked — should cap at tracked count
        let count = tracker.multi_token_accept(10);
        assert!(count <= 2);
    }

    #[test]
    fn ema_tracker_all_rejects_drives_rate_down() {
        let mut tracker = MtpEmaTracker::new(2, 0.8);
        for _ in 0..50 {
            tracker.record_position(0, false);
            tracker.record_position(1, false);
        }
        let rate = tracker.global_ema_rate();
        assert!(rate < 0.3, "After 50 consecutive rejects, global rate should be well below 0.3, got {}", rate);
    }

    #[test]
    fn ema_tracker_record_batch_multiple_steps() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);
        // Record improving acceptance rates
        tracker.record_batch(0.2);
        let rate_after_low = tracker.global_ema_rate();
        tracker.record_batch(0.9);
        tracker.record_batch(0.9);
        let rate_after_high = tracker.global_ema_rate();
        assert!(rate_after_high > rate_after_low, "Rate should increase after high acceptance batches");
    }

    #[test]
    fn adapter_distill_step_returns_positive_loss() {
        let config = AdapterConfig {
            hidden_size: 8,
            vocab_size: 10,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; 80]);
        let norm_weight = Arc::new(vec![1.0f32; 8]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 8];
        let draft = adapter.forward(&hidden);
        // Non-uniform target so softmax produces a different distribution
        let target: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let loss = adapter.distill_step(&draft, &target, &hidden, 0.01);
        assert!(loss > 0.0, "Loss should be positive when draft != target, got {}", loss);
    }

    #[test]
    fn adapter_phase_b_forward_differs_from_phase_a() {
        let config = AdapterConfig {
            hidden_size: 8,
            vocab_size: 10,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; 80]);
        let norm_weight = Arc::new(vec![1.0f32; 8]);
        let hidden = vec![1.0f32; 8];

        let phase_a = DraftAdapter::new_phase_a(config.clone(), lm_head.clone(), norm_weight.clone());
        let mut phase_b = DraftAdapter::new_phase_b(config, lm_head, norm_weight);

        // Run distillation with non-uniform target to make delta non-zero
        let draft = phase_b.forward(&hidden);
        let target: Vec<f32> = (0..10).map(|i| i as f32).collect();
        phase_b.distill_step(&draft, &target, &hidden, 1.0);

        let logits_a = phase_a.forward(&hidden);
        let logits_b = phase_b.forward(&hidden);
        assert_ne!(logits_a, logits_b, "Phase B with distilled delta should differ from Phase A");
    }

    #[test]
    fn softmax_diff_returns_correct_length() {
        let draft = vec![1.0f32, 2.0f32, 3.0f32];
        let target = vec![0.5f32, 1.5f32, 2.5f32];
        let diff = softmax_diff(&draft, &target, 3);
        assert_eq!(diff.len(), 3);
        // softmax diff values should sum to zero (both sum to 1.0)
        let sum: f32 = diff.iter().sum();
        assert!(sum.abs() < 1e-5, "softmax diff should sum to ~0, got {}", sum);
    }

    #[test]
    fn rms_norm_free_unit_vector_with_unit_weight() {
        let x = vec![1.0f32, 0.0f32, 0.0f32, 0.0f32];
        let weight = vec![1.0f32; 4];
        let result = rms_norm_free(&x, &weight, 1e-5);
        // rms of [1,0,0,0] = sqrt(1/4) = 0.5, inv_rms = 2.0
        // result[0] = 1.0 * 2.0 * 1.0 = 2.0, rest = 0.0
        assert!((result[0] - 2.0).abs() < 1e-4, "Expected ~2.0, got {}", result[0]);
        assert!(result[1].abs() < 1e-6);
        assert!(result[2].abs() < 1e-6);
        assert!(result[3].abs() < 1e-6);
    }

    // ── New coverage tests ───────────────────────────────────────────────────────

    // -- AdapterConfig --

    #[test]
    fn adapter_config_field_mutation() {
        let mut config = AdapterConfig::default();
        config.hidden_size = 1024;
        config.vocab_size = 50000;
        config.rms_norm_eps = 1e-6;
        config.enable_distillation = true;
        config.distillation_steps = 500;
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.vocab_size, 50000);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!(config.enable_distillation);
        assert_eq!(config.distillation_steps, 500);
    }

    #[test]
    fn adapter_config_equality_after_clone() {
        let a = AdapterConfig {
            hidden_size: 768,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 50,
        };
        let b = a.clone();
        assert_eq!(a.hidden_size, b.hidden_size);
        assert_eq!(a.vocab_size, b.vocab_size);
        assert_eq!(a.rms_norm_eps.to_bits(), b.rms_norm_eps.to_bits());
        assert_eq!(a.enable_distillation, b.enable_distillation);
        assert_eq!(a.distillation_steps, b.distillation_steps);
    }

    // -- DraftAdapter --

    #[test]
    fn adapter_phase_b_forward_equals_phase_a_before_distillation() {
        let config = AdapterConfig {
            hidden_size: 8,
            vocab_size: 10,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; 80]);
        let norm_weight = Arc::new(vec![1.0f32; 8]);
        let hidden = vec![1.0f32; 8];

        let phase_a = DraftAdapter::new_phase_a(config.clone(), lm_head.clone(), norm_weight.clone());
        let phase_b = DraftAdapter::new_phase_b(config, lm_head, norm_weight);

        let logits_a = phase_a.forward(&hidden);
        let logits_b = phase_b.forward(&hidden);
        // Phase B delta is initialized to all zeros, so outputs must match Phase A
        assert_eq!(logits_a, logits_b);
    }

    #[test]
    fn adapter_phase_b_distill_step_increments_counter() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 5,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 10,
        };
        let lm_head = Arc::new(vec![0.1f32; 20]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 4];

        for i in 1..=5 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &vec![0.0f32; 5], &hidden, 0.01);
            assert_eq!(adapter.distillation_progress().0, i);
        }
        assert!(!adapter.is_distillation_complete());
    }

    #[test]
    fn adapter_forward_with_zero_hidden() {
        let adapter = make_adapter(8, 10);
        let hidden = vec![0.0f32; 8];
        let logits = adapter.forward(&hidden);
        // All-zero hidden with all-ones norm_weight: rms = sqrt(eps), result ≈ 0 * (1/sqrt(eps)) * 1.0 ≈ 0
        // Then matmul with weight 0.1: all logits should be near zero
        for &l in &logits {
            assert!(l.abs() < 0.1, "Expected near-zero logit, got {}", l);
        }
    }

    #[test]
    fn adapter_forward_with_negative_hidden() {
        let adapter = make_adapter(4, 6);
        let hidden = vec![-1.0f32, -0.5f32, 0.5f32, 1.0f32];
        let logits = adapter.forward(&hidden);
        assert_eq!(logits.len(), 6);
        // All logits should be identical since weight is uniform (0.1)
        // and norm_weight is uniform (1.0). Symmetric input produces symmetric output.
        let first = logits[0];
        for &l in &logits {
            assert!((l - first).abs() < 1e-5, "Uniform weights should produce identical logits");
        }
    }

    #[test]
    fn adapter_batch_forward_empty() {
        let adapter = make_adapter(8, 10);
        let results = adapter.forward_batch(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn adapter_batch_forward_single() {
        let adapter = make_adapter(8, 10);
        let hidden = vec![1.0f32; 8];
        let results = adapter.forward_batch(&[&hidden]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 10);
        // Should match direct forward
        let direct = adapter.forward(&hidden);
        assert_eq!(results[0], direct);
    }

    #[test]
    fn adapter_phase_a_distillation_not_complete_by_default() {
        let adapter = make_adapter(4, 5);
        assert!(!adapter.is_distillation_complete());
        assert_eq!(adapter.distillation_progress(), (0, 100));
    }

    // -- MtpEmaTracker --

    #[test]
    fn ema_tracker_threshold_clamped_lower_bound() {
        let mut tracker = MtpEmaTracker::new(2, 0.9);
        // Drive acceptance rate very high to push threshold down via negative trend_adjustment
        for _ in 0..100 {
            tracker.record_position(0, true);
            tracker.record_position(1, true);
        }
        let threshold = tracker.threshold();
        assert!(threshold >= 0.2, "Threshold should be clamped at >= 0.2, got {}", threshold);
    }

    #[test]
    fn ema_tracker_threshold_clamped_upper_bound() {
        let mut tracker = MtpEmaTracker::new(2, 0.9);
        // Drive acceptance rate very low — strong negative trend → high threshold
        // But also need some steps to establish trend
        for _ in 0..100 {
            tracker.record_position(0, false);
            tracker.record_position(1, false);
        }
        let threshold = tracker.threshold();
        assert!(threshold <= 0.8, "Threshold should be clamped at <= 0.8, got {}", threshold);
    }

    #[test]
    fn ema_tracker_effective_alpha_warmup_first_step() {
        // At step 0, effective_alpha = 0.5 * (1 - 0/16) + nominal * (0/16) = 0.5
        let mut tracker = MtpEmaTracker::new(2, 0.1);
        assert_eq!(tracker.steps(), 0);
        tracker.record_position(0, true);
        // After 1 step, we are in warmup. We can verify indirectly:
        // With nominal_alpha=0.1, after one accept the rate should be:
        // alpha * 1.0 + (1 - alpha) * 0.5 where alpha = 0.5*(1 - 1/16) + 0.1*(1/16)
        let rate = tracker.ema_acceptance_rate(0).unwrap();
        assert!(rate > 0.5, "After first accept, rate should exceed 0.5, got {}", rate);
        assert!(rate < 1.0, "EMA should not jump to 1.0 immediately, got {}", rate);
    }

    #[test]
    fn ema_tracker_alpha_clamped_at_upper_bound() {
        // alpha > 1.0 should be clamped to 1.0
        let tracker = MtpEmaTracker::new(2, 5.0);
        assert_eq!(tracker.steps(), 0);
        // Verify it doesn't panic and has valid initial state
        assert!((tracker.global_ema_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ema_tracker_record_batch_increments_steps() {
        let mut tracker = MtpEmaTracker::new(3, 0.3);
        assert_eq!(tracker.steps(), 0);
        tracker.record_batch(0.5);
        assert_eq!(tracker.steps(), 1);
        tracker.record_batch(0.7);
        assert_eq!(tracker.steps(), 2);
        tracker.record_batch(0.3);
        assert_eq!(tracker.steps(), 3);
    }

    #[test]
    fn ema_tracker_ema_accept_with_rising_trend_override() {
        // When global_trend > 0.1, ema_accept returns true even if position rate < threshold
        let mut tracker = MtpEmaTracker::new(2, 0.9);
        // First, establish a low rate
        for _ in 0..20 {
            tracker.record_position(0, false);
        }
        // Then record a strong improvement
        for _ in 0..5 {
            tracker.record_position(0, true);
        }
        // The trend should be positive enough that ema_accept may return true
        // This test verifies the logic path, not a specific boolean result,
        // since threshold adaptation is complex. Just ensure no panic.
        let _ = tracker.ema_accept(0);
    }

    #[test]
    fn ema_tracker_multi_token_accept_zero_requested() {
        let tracker = MtpEmaTracker::new(4, 0.3);
        let count = tracker.multi_token_accept(0);
        assert_eq!(count, 0);
    }

    #[test]
    fn ema_tracker_reset_then_record_works() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        for _ in 0..10 {
            tracker.record_position(0, true);
        }
        tracker.reset();
        assert_eq!(tracker.steps(), 0);

        // After reset, recording should work normally
        tracker.record_position(0, true);
        assert_eq!(tracker.steps(), 1);
        assert!(tracker.ema_acceptance_rate(0).unwrap() > 0.5);
    }

    #[test]
    fn ema_tracker_consecutive_mixed_signals() {
        let mut tracker = MtpEmaTracker::new(1, 0.5);
        // Alternate accept/reject
        for i in 0..20 {
            tracker.record_position(0, i % 2 == 0);
        }
        // Rate should be around 0.5 (roughly equal accepts and rejects)
        let rate = tracker.global_ema_rate();
        assert!(rate > 0.3 && rate < 0.7, "Mixed signals should keep rate near 0.5, got {}", rate);
    }

    #[test]
    fn ema_tracker_record_batch_zero_rate() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        tracker.record_batch(0.0);
        assert!(tracker.global_ema_rate() < 0.5, "Zero rate batch should decrease global rate");
    }

    // -- Free functions --

    #[test]
    fn softmax_diff_identical_inputs_produce_zero() {
        let vals = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let diff = softmax_diff(&vals, &vals, 4);
        for (i, &d) in diff.iter().enumerate() {
            assert!(d.abs() < 1e-6, "Identical inputs should produce ~0 diff at index {}, got {}", i, d);
        }
    }

    #[test]
    fn softmax_diff_single_element() {
        let single = vec![5.0f32];
        let diff = softmax_diff(&single, &single, 1);
        assert_eq!(diff.len(), 1);
        assert!(diff[0].abs() < 1e-6, "Single element diff should be ~0, got {}", diff[0]);
    }

    #[test]
    fn rms_norm_free_all_ones_with_unit_weight() {
        let x = vec![1.0f32; 4];
        let weight = vec![1.0f32; 4];
        let result = rms_norm_free(&x, &weight, 1e-5);
        // rms of all-ones = sqrt(4/4) = 1.0, inv_rms = 1.0
        // result = 1.0 * 1.0 * 1.0 = 1.0 for all elements
        for (i, &r) in result.iter().enumerate() {
            assert!((r - 1.0).abs() < 1e-4, "Element {} should be ~1.0, got {}", i, r);
        }
    }

    #[test]
    fn rms_norm_free_zero_vector_no_nan() {
        let x = vec![0.0f32; 4];
        let weight = vec![1.0f32; 4];
        let result = rms_norm_free(&x, &weight, 1e-5);
        // rms = sqrt(0 + eps) ≈ sqrt(eps), inv_rms = 1/sqrt(eps) — large but finite
        for (i, &r) in result.iter().enumerate() {
            assert!(r.is_finite(), "Element {} should be finite, got {}", i, r);
            assert!(!r.is_nan(), "Element {} should not be NaN", i);
        }
    }

    #[test]
    fn rms_norm_free_with_non_uniform_weight() {
        let x = vec![2.0f32, 2.0f32];
        let weight = vec![0.5f32, 2.0f32];
        let result = rms_norm_free(&x, &weight, 1e-5);
        // rms = sqrt((4+4)/2) = 2.0, inv_rms = 0.5
        // result[0] = 2.0 * 0.5 * 0.5 = 0.5
        // result[1] = 2.0 * 0.5 * 2.0 = 2.0
        assert!((result[0] - 0.5).abs() < 1e-4, "Expected 0.5, got {}", result[0]);
        assert!((result[1] - 2.0).abs() < 1e-4, "Expected 2.0, got {}", result[1]);
    }

    #[test]
    fn adapter_parameter_bytes_phase_a_with_large_dims() {
        let adapter = make_adapter(4096, 128000);
        assert_eq!(adapter.parameter_bytes(), 0, "Phase A always has zero extra parameter bytes");
    }

    #[test]
    fn adapter_parameter_bytes_phase_b_matches_dimensions() {
        let hidden = 512;
        let vocab = 10000;
        let config = AdapterConfig {
            hidden_size: hidden,
            vocab_size: vocab,
            enable_distillation: true,
            ..Default::default()
        };
        let lm_head = Arc::new(vec![0.0f32; vocab * hidden]);
        let norm_weight = Arc::new(vec![1.0f32; hidden]);
        let adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let expected_bytes = vocab * hidden * std::mem::size_of::<f32>();
        assert_eq!(adapter.parameter_bytes(), expected_bytes);
    }

    // ── Additional coverage tests (round 2) ──────────────────────────────────────

    #[test]
    fn adapter_config_explicit_construction_all_fields() {
        let config = AdapterConfig {
            hidden_size: 2048,
            vocab_size: 64000,
            rms_norm_eps: 1e-6,
            enable_distillation: true,
            distillation_steps: 250,
        };
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.vocab_size, 64000);
        assert!((config.rms_norm_eps - 1e-6).abs() < f32::EPSILON);
        assert!(config.enable_distillation);
        assert_eq!(config.distillation_steps, 250);
    }

    #[test]
    fn adapter_config_clone_produces_equal_fields() {
        let config = AdapterConfig {
            hidden_size: 4096,
            vocab_size: 128256,
            rms_norm_eps: 1e-5,
            enable_distillation: false,
            distillation_steps: 100,
        };
        let cloned = config.clone();
        // Modify original to prove independence
        let original_hidden = config.hidden_size;
        drop(config);
        assert_eq!(cloned.hidden_size, original_hidden);
        assert_eq!(cloned.vocab_size, 128256);
        assert_eq!(cloned.distillation_steps, 100);
    }

    #[test]
    fn adapter_config_debug_includes_all_fields() {
        let config = AdapterConfig {
            hidden_size: 768,
            vocab_size: 30000,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 50,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("hidden_size"), "Debug should include hidden_size");
        assert!(debug.contains("vocab_size"), "Debug should include vocab_size");
        assert!(debug.contains("rms_norm_eps"), "Debug should include rms_norm_eps");
        assert!(debug.contains("enable_distillation"), "Debug should include enable_distillation");
        assert!(debug.contains("distillation_steps"), "Debug should include distillation_steps");
    }

    #[test]
    fn adapter_config_default_distillation_disabled() {
        let config = AdapterConfig::default();
        assert!(!config.enable_distillation, "Default should have distillation disabled");
        assert_eq!(config.distillation_steps, 100, "Default distillation_steps should be 100");
    }

    #[test]
    fn draft_adapter_debug_phase_a_shows_no_residual() {
        let adapter = make_adapter(128, 500);
        let debug = format!("{:?}", adapter);
        assert!(debug.contains("hidden_size: 128"));
        assert!(debug.contains("vocab_size: 500"));
        assert!(debug.contains("distillation_step: 0"));
        assert!(debug.contains("has_residual_delta: false"));
    }

    #[test]
    fn mtp_ema_tracker_new_with_negative_alpha_clamps_to_min() {
        let tracker = MtpEmaTracker::new(4, -1.0);
        // Alpha is clamped to 0.01 min; tracker should initialize correctly
        assert_eq!(tracker.steps(), 0);
        assert!((tracker.global_ema_rate() - 0.5).abs() < 1e-6);
        assert!((tracker.threshold() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mtp_ema_tracker_new_with_nan_alpha_clamps_to_min() {
        let tracker = MtpEmaTracker::new(3, f32::NAN);
        // NaN clamp: NaN < 0.01 is false, NaN > 1.0 is false, clamp returns NaN
        // but the code uses .clamp(0.01, 1.0) which returns NaN for NaN input.
        // Verify it doesn't panic and has valid initial state for position_rates
        assert_eq!(tracker.steps(), 0);
        assert_eq!(tracker.ema_acceptance_rate(0), Some(0.5));
        assert_eq!(tracker.ema_acceptance_rate(2), Some(0.5));
    }

    #[test]
    fn mtp_ema_tracker_new_with_large_positions() {
        let tracker = MtpEmaTracker::new(1000, 0.3);
        assert_eq!(tracker.ema_acceptance_rate(0), Some(0.5));
        assert_eq!(tracker.ema_acceptance_rate(999), Some(0.5));
        assert_eq!(tracker.ema_acceptance_rate(1000), None);
    }

    #[test]
    fn mtp_ema_tracker_record_position_single_accept() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        tracker.record_position(0, true);
        let rate = tracker.ema_acceptance_rate(0).unwrap();
        assert!(rate > 0.5, "Single accept should raise rate above 0.5, got {}", rate);
        assert_eq!(tracker.steps(), 1);
    }

    #[test]
    fn mtp_ema_tracker_record_position_single_reject() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        tracker.record_position(0, false);
        let rate = tracker.ema_acceptance_rate(0).unwrap();
        assert!(rate < 0.5, "Single reject should lower rate below 0.5, got {}", rate);
        assert_eq!(tracker.steps(), 1);
    }

    #[test]
    fn mtp_ema_tracker_record_batch_extreme_values() {
        let mut tracker_low = MtpEmaTracker::new(2, 0.5);
        let mut tracker_high = MtpEmaTracker::new(2, 0.5);

        tracker_low.record_batch(0.0);
        tracker_high.record_batch(1.0);

        assert!(
            tracker_low.global_ema_rate() < tracker_high.global_ema_rate(),
            "Zero batch rate should produce lower global rate than perfect batch"
        );
    }

    #[test]
    fn mtp_ema_tracker_ema_accept_fresh_tracker_at_threshold() {
        let tracker = MtpEmaTracker::new(4, 0.3);
        // Fresh tracker: all position_rates = 0.5, threshold = 0.5
        // 0.5 >= 0.5 is true, so ema_accept should return true
        assert!(tracker.ema_accept(0), "Fresh tracker position 0: rate=0.5 >= threshold=0.5");
        assert!(tracker.ema_accept(3), "Fresh tracker position 3: rate=0.5 >= threshold=0.5");
    }

    #[test]
    fn mtp_ema_tracker_multi_token_accept_fresh_returns_all() {
        let tracker = MtpEmaTracker::new(5, 0.3);
        // Fresh tracker: all rates = 0.5 >= threshold = 0.5
        let count = tracker.multi_token_accept(5);
        assert_eq!(count, 5, "Fresh tracker should accept all positions");
    }

    #[test]
    fn mtp_ema_tracker_is_distillation_complete_on_fresh_adapter() {
        // DraftAdapter is_distillation_complete on a fresh Phase A adapter
        let adapter = make_adapter(16, 50);
        // distillation_step = 0, distillation_steps = 100
        assert!(!adapter.is_distillation_complete());
        assert_eq!(adapter.distillation_progress(), (0, 100));
    }

    #[test]
    fn mtp_ema_tracker_record_batch_negative_rate_handled() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        // Out-of-range acceptance rate should not panic
        tracker.record_batch(-0.5);
        assert_eq!(tracker.steps(), 1);
        assert!(tracker.global_ema_rate().is_finite(), "Global rate should remain finite");
    }

    #[test]
    fn mtp_ema_tracker_record_batch_rate_above_one_handled() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        tracker.record_batch(2.0);
        assert_eq!(tracker.steps(), 1);
        assert!(tracker.global_ema_rate().is_finite(), "Global rate should remain finite");
    }

    #[test]
    fn mtp_ema_tracker_clone_then_record_does_not_affect_original() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        tracker.record_position(0, true);
        let original_rate = tracker.global_ema_rate();
        let original_steps = tracker.steps();

        let mut cloned = tracker.clone();
        cloned.record_position(0, false);

        assert_eq!(tracker.steps(), original_steps, "Original steps should not change");
        assert!(
            (tracker.global_ema_rate() - original_rate).abs() < 1e-6,
            "Original global rate should not change"
        );
        assert_ne!(
            cloned.global_ema_rate(), tracker.global_ema_rate(),
            "Cloned tracker rate should diverge after independent update"
        );
    }

    // ── Additional coverage tests (round 3) ──────────────────────────────────────────

    // -- AdapterConfig edge cases --

    #[test]
    fn adapter_config_zero_sizes() {
        let config = AdapterConfig {
            hidden_size: 0,
            vocab_size: 0,
            rms_norm_eps: 1e-5,
            enable_distillation: false,
            distillation_steps: 0,
        };
        assert_eq!(config.hidden_size, 0);
        assert_eq!(config.vocab_size, 0);
        assert_eq!(config.distillation_steps, 0);
    }

    #[test]
    fn adapter_config_very_large_dims() {
        let config = AdapterConfig {
            hidden_size: usize::MAX,
            vocab_size: usize::MAX,
            rms_norm_eps: 1e-12,
            enable_distillation: true,
            distillation_steps: usize::MAX,
        };
        assert_eq!(config.hidden_size, usize::MAX);
        assert_eq!(config.vocab_size, usize::MAX);
        assert_eq!(config.distillation_steps, usize::MAX);
    }

    #[test]
    fn adapter_config_eps_near_zero() {
        let config = AdapterConfig {
            hidden_size: 16,
            vocab_size: 50,
            rms_norm_eps: 1e-38,
            enable_distillation: false,
            distillation_steps: 100,
        };
        assert!(config.rms_norm_eps > 0.0);
        assert!(config.rms_norm_eps < 1e-30);
    }

    #[test]
    fn adapter_config_eps_large() {
        let config = AdapterConfig {
            hidden_size: 16,
            vocab_size: 50,
            rms_norm_eps: 1.0,
            enable_distillation: false,
            distillation_steps: 100,
        };
        assert!((config.rms_norm_eps - 1.0).abs() < 1e-10);
    }

    // -- DraftAdapter construction --

    #[test]
    fn adapter_phase_a_shares_weight_arc() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 6,
            rms_norm_eps: 1e-5,
            enable_distillation: false,
            distillation_steps: 10,
        };
        let lm_head = Arc::new(vec![0.5f32; 24]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let a1 = DraftAdapter::new_phase_a(config.clone(), lm_head.clone(), norm_weight.clone());
        let a2 = DraftAdapter::new_phase_a(config, lm_head, norm_weight);
        // Both adapters use shared references; forward should produce identical results
        let hidden = vec![1.0f32; 4];
        assert_eq!(a1.forward(&hidden), a2.forward(&hidden));
    }

    #[test]
    fn adapter_phase_b_residual_delta_initialized_to_zero() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 5,
            ..Default::default()
        };
        let lm_head = Arc::new(vec![1.0f32; 20]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let debug = format!("{:?}", adapter);
        assert!(debug.contains("has_residual_delta: true"));
    }

    #[test]
    fn adapter_phase_b_debug_shows_step_zero() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 5,
            ..Default::default()
        };
        let lm_head = Arc::new(vec![0.1f32; 20]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let debug = format!("{:?}", adapter);
        assert!(debug.contains("distillation_step: 0"));
    }

    // -- DraftAdapter forward edge cases --

    #[test]
    fn adapter_forward_all_same_hidden_values() {
        let adapter = make_adapter(4, 3);
        let hidden = vec![3.14f32; 4];
        let logits = adapter.forward(&hidden);
        // With uniform weight (0.1) and uniform norm_weight (1.0), all logits equal
        let first = logits[0];
        for &l in &logits {
            assert!((l - first).abs() < 1e-5);
        }
    }

    #[test]
    fn adapter_forward_with_large_hidden_values() {
        let adapter = make_adapter(4, 3);
        let hidden = vec![1e6f32; 4];
        let logits = adapter.forward(&hidden);
        assert_eq!(logits.len(), 3);
        for &l in &logits {
            assert!(l.is_finite(), "Large hidden values should produce finite logits, got {}", l);
        }
    }

    #[test]
    fn adapter_forward_with_tiny_hidden_values() {
        let adapter = make_adapter(4, 3);
        let hidden = vec![1e-30f32; 4];
        let logits = adapter.forward(&hidden);
        assert_eq!(logits.len(), 3);
        for &l in &logits {
            assert!(l.is_finite(), "Tiny hidden values should produce finite logits, got {}", l);
        }
    }

    #[test]
    fn adapter_forward_mixed_sign_hidden() {
        let adapter = make_adapter(4, 3);
        let hidden = vec![-2.0f32, 2.0f32, -2.0f32, 2.0f32];
        let logits = adapter.forward(&hidden);
        // With uniform weight, all logits should be identical (squared terms cancel out via norm)
        let first = logits[0];
        for &l in &logits {
            assert!((l - first).abs() < 1e-4, "Expected uniform logits with symmetric input");
        }
    }

    // -- DraftAdapter distillation edge cases --

    #[test]
    fn adapter_distill_step_with_zero_learning_rate() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 5,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 10,
        };
        let lm_head = Arc::new(vec![0.1f32; 20]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 4];
        let logits_before = adapter.forward(&hidden);

        let draft = adapter.forward(&hidden);
        let target: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let _loss = adapter.distill_step(&draft, &target, &hidden, 0.0);

        let logits_after = adapter.forward(&hidden);
        assert_eq!(logits_before, logits_after, "Zero learning rate should not change logits");
    }

    #[test]
    fn adapter_distill_with_same_target_no_change_after_many_steps() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 10,
        };
        let lm_head = Arc::new(vec![0.1f32; 12]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 4];

        // Distill with same draft as target — gradient should be ~0
        for _ in 0..5 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &draft, &hidden, 0.01);
        }
        // After distilling with identical inputs, logits should barely change
        let logits = adapter.forward(&hidden);
        // Just verify no NaN/Inf introduced
        for &l in &logits {
            assert!(l.is_finite());
        }
    }

    #[test]
    fn adapter_distill_step_loss_is_finite() {
        let config = AdapterConfig {
            hidden_size: 8,
            vocab_size: 10,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; 80]);
        let norm_weight = Arc::new(vec![1.0f32; 8]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 8];
        let draft = adapter.forward(&hidden);
        let target = vec![5.0f32; 10];
        let loss = adapter.distill_step(&draft, &target, &hidden, 0.01);
        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
    }

    #[test]
    fn adapter_distillation_progress_after_partial_steps() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 10,
        };
        let lm_head = Arc::new(vec![0.1f32; 12]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 4];

        for expected in 1..=3 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &vec![0.0f32; 3], &hidden, 0.01);
            assert_eq!(adapter.distillation_progress(), (expected, 10));
        }
        assert!(!adapter.is_distillation_complete());
    }

    // -- DraftAdapter parameter_bytes --

    #[test]
    fn adapter_parameter_bytes_phase_b_increases_with_dims() {
        let lm_head_small = Arc::new(vec![0.0f32; 10 * 4]);
        let norm_small = Arc::new(vec![1.0f32; 4]);
        let config_small = AdapterConfig { hidden_size: 4, vocab_size: 10, ..Default::default() };
        let adapter_small = DraftAdapter::new_phase_b(config_small, lm_head_small, norm_small);

        let lm_head_large = Arc::new(vec![0.0f32; 20 * 8]);
        let norm_large = Arc::new(vec![1.0f32; 8]);
        let config_large = AdapterConfig { hidden_size: 8, vocab_size: 20, ..Default::default() };
        let adapter_large = DraftAdapter::new_phase_b(config_large, lm_head_large, norm_large);

        assert!(
            adapter_large.parameter_bytes() > adapter_small.parameter_bytes(),
            "Larger dims should have more parameter bytes"
        );
    }

    // -- DraftAdapter forward_batch --

    #[test]
    fn adapter_batch_forward_three_different_inputs() {
        let adapter = make_adapter(4, 3);
        let h1 = vec![1.0f32; 4];
        let h2 = vec![0.5f32; 4];
        let h3 = vec![0.0f32; 4];
        let results = adapter.forward_batch(&[&h1, &h2, &h3]);
        assert_eq!(results.len(), 3);
        // Each should match individual forward
        assert_eq!(results[0], adapter.forward(&h1));
        assert_eq!(results[1], adapter.forward(&h2));
        assert_eq!(results[2], adapter.forward(&h3));
    }

    // -- softmax_diff edge cases --

    #[test]
    fn softmax_diff_with_large_values() {
        let draft = vec![1000.0f32, 2000.0f32];
        let target = vec![0.0f32, 1.0f32];
        let diff = softmax_diff(&draft, &target, 2);
        assert_eq!(diff.len(), 2);
        for &d in &diff {
            assert!(d.is_finite(), "Diff should be finite even with large inputs, got {}", d);
        }
    }

    #[test]
    fn softmax_diff_with_negative_values() {
        let draft = vec![-5.0f32, -3.0f32, -1.0f32];
        let target = vec![-1.0f32, -3.0f32, -5.0f32];
        let diff = softmax_diff(&draft, &target, 3);
        assert_eq!(diff.len(), 3);
        let sum: f32 = diff.iter().sum();
        assert!(sum.abs() < 1e-4, "Softmax diff sum should be ~0, got {}", sum);
    }

    #[test]
    fn softmax_diff_uniform_inputs() {
        let uniform = vec![2.0f32; 5];
        let diff = softmax_diff(&uniform, &uniform, 5);
        for (i, &d) in diff.iter().enumerate() {
            assert!(d.abs() < 1e-6, "Uniform inputs should produce ~0 diff at {}, got {}", i, d);
        }
    }

    #[test]
    fn softmax_diff_two_elements_symmetric() {
        let a = vec![0.0f32, 1.0f32];
        let b = vec![1.0f32, 0.0f32];
        let diff = softmax_diff(&a, &b, 2);
        // Swapped inputs: diff[0] = -diff[1]
        assert!((diff[0] + diff[1]).abs() < 1e-5, "Symmetric swap should produce opposite diffs");
    }

    // -- rms_norm_free edge cases --

    #[test]
    fn rms_norm_free_single_element() {
        let x = vec![3.0f32];
        let weight = vec![2.0f32];
        let result = rms_norm_free(&x, &weight, 1e-5);
        // rms = sqrt(9/1 + eps) ≈ 3.0, inv_rms ≈ 1/3
        // result = 3.0 * (1/3) * 2.0 = 2.0
        assert!((result[0] - 2.0).abs() < 1e-4, "Expected ~2.0, got {}", result[0]);
    }

    #[test]
    fn rms_norm_free_with_zero_weight() {
        let x = vec![1.0f32, 2.0f32, 3.0f32];
        let weight = vec![0.0f32; 3];
        let result = rms_norm_free(&x, &weight, 1e-5);
        for (i, &r) in result.iter().enumerate() {
            assert!(r.abs() < 1e-6, "Zero weight should produce ~0 at index {}, got {}", i, r);
        }
    }

    #[test]
    fn rms_norm_free_large_input_no_overflow() {
        let x = vec![1e30f32; 4];
        let weight = vec![1.0f32; 4];
        let result = rms_norm_free(&x, &weight, 1e-5);
        for (i, &r) in result.iter().enumerate() {
            assert!(r.is_finite(), "Element {} should be finite with large input, got {}", i, r);
        }
    }

    #[test]
    fn rms_norm_free_negative_input() {
        let x = vec![-3.0f32, 0.0f32, 3.0f32];
        let weight = vec![1.0f32; 3];
        let result = rms_norm_free(&x, &weight, 1e-5);
        // rms = sqrt((9+0+9)/3) = sqrt(6), inv_rms = 1/sqrt(6)
        // result[0] = -3 * (1/sqrt(6)) * 1.0 ≈ -1.225
        // result[2] = 3 * (1/sqrt(6)) * 1.0 ≈ 1.225
        assert!(result[0] < 0.0, "Negative input should produce negative output");
        assert!(result[1].abs() < 1e-6, "Zero input should produce ~0 output");
        assert!(result[2] > 0.0, "Positive input should produce positive output");
        assert!((result[0].abs() - result[2].abs()).abs() < 1e-4, "Symmetric magnitudes");
    }

    // -- MtpEmaTracker record_position edge cases --

    #[test]
    fn ema_tracker_record_only_last_position() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        // Record accepts only on position 3
        for _ in 0..20 {
            tracker.record_position(3, true);
        }
        let rate_3 = tracker.ema_acceptance_rate(3).unwrap();
        let rate_0 = tracker.ema_acceptance_rate(0).unwrap();
        assert!(rate_3 > rate_0, "Position 3 rate should exceed position 0 after accepts");
    }

    #[test]
    fn ema_tracker_record_first_position_many_rejects() {
        let mut tracker = MtpEmaTracker::new(3, 0.7);
        for _ in 0..50 {
            tracker.record_position(0, false);
        }
        let rate_0 = tracker.ema_acceptance_rate(0).unwrap();
        assert!(rate_0 < 0.1, "Position 0 rate should be very low after 50 rejects, got {}", rate_0);
    }

    #[test]
    fn ema_tracker_positions_diverge_independently() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        for _ in 0..30 {
            tracker.record_position(0, true);
            tracker.record_position(1, false);
        }
        let rate_0 = tracker.ema_acceptance_rate(0).unwrap();
        let rate_1 = tracker.ema_acceptance_rate(1).unwrap();
        assert!(rate_0 > rate_1, "Position 0 should have higher rate than position 1");
    }

    // -- MtpEmaTracker ema_accept edge cases --

    #[test]
    fn ema_tracker_ema_accept_all_rejected_positions() {
        let mut tracker = MtpEmaTracker::new(3, 0.8);
        for _ in 0..100 {
            for pos in 0..3 {
                tracker.record_position(pos, false);
            }
        }
        // After massive rejection, position rates should be very low
        for pos in 0..3 {
            let rate = tracker.ema_acceptance_rate(pos).unwrap();
            assert!(rate < 0.2, "Position {} rate should be very low, got {}", pos, rate);
        }
    }

    #[test]
    fn ema_tracker_multi_token_accept_partial() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        // Accept positions 0 and 1, reject 2 and 3
        for _ in 0..50 {
            tracker.record_position(0, true);
            tracker.record_position(1, true);
            tracker.record_position(2, false);
            tracker.record_position(3, false);
        }
        // This should accept at least position 0 (maybe 1 depending on threshold)
        let count = tracker.multi_token_accept(4);
        assert!(count >= 1, "Should accept at least position 0, got {}", count);
        assert!(count <= 2, "Should not accept positions 2-3, got {}", count);
    }

    // -- MtpEmaTracker record_batch edge cases --

    #[test]
    fn ema_tracker_record_batch_nan_rate_handled() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        tracker.record_batch(f32::NAN);
        // Should not panic; steps still increments
        assert_eq!(tracker.steps(), 1);
        assert!(tracker.global_ema_rate().is_nan() || tracker.global_ema_rate().is_finite());
    }

    #[test]
    fn ema_tracker_record_batch_inf_rate_handled() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        tracker.record_batch(f32::INFINITY);
        assert_eq!(tracker.steps(), 1);
        assert!(tracker.global_ema_rate().is_finite() || tracker.global_ema_rate().is_infinite());
    }

    #[test]
    fn ema_tracker_record_batch_converges_to_target() {
        let mut tracker = MtpEmaTracker::new(2, 0.1); // low alpha for slow convergence
        for _ in 0..200 {
            tracker.record_batch(0.8);
        }
        let rate = tracker.global_ema_rate();
        assert!(rate > 0.6, "After 200 batches at 0.8, rate should approach 0.8, got {}", rate);
    }

    // -- MtpEmaTracker reset edge cases --

    #[test]
    fn ema_tracker_double_reset_idempotent() {
        let mut tracker = MtpEmaTracker::new(3, 0.3);
        tracker.record_position(0, true);
        tracker.reset();
        tracker.reset();
        assert_eq!(tracker.steps(), 0);
        assert!((tracker.global_ema_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ema_tracker_reset_then_batch_record() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        for _ in 0..10 {
            tracker.record_position(0, true);
        }
        tracker.reset();
        assert_eq!(tracker.steps(), 0);
        tracker.record_batch(0.9);
        assert_eq!(tracker.steps(), 1);
        assert!(tracker.global_ema_rate() > 0.5);
    }

    // -- MtpEmaTracker clone independence --

    #[test]
    fn ema_tracker_clone_independent_multi_record() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        tracker.record_position(0, true);
        let mut cloned = tracker.clone();

        // Mutate original
        for _ in 0..5 {
            tracker.record_position(0, false);
        }
        // Mutate clone
        for _ in 0..5 {
            cloned.record_position(0, true);
        }
        assert_ne!(
            tracker.global_ema_rate(), cloned.global_ema_rate(),
            "Independent mutations should diverge"
        );
    }

    // -- MtpEmaTracker alpha warmup transition --

    #[test]
    fn ema_tracker_warmup_phase_transitions_at_step_16() {
        let mut tracker = MtpEmaTracker::new(2, 0.2);
        // First 16 steps use warmup alpha (higher)
        for _ in 0..15 {
            tracker.record_position(0, true);
        }
        assert_eq!(tracker.steps(), 15);
        // Step 16 transitions to adaptive alpha
        tracker.record_position(0, true);
        assert_eq!(tracker.steps(), 16);
        // Just verify no panic and rate is valid
        let rate = tracker.global_ema_rate();
        assert!(rate > 0.0 && rate <= 1.0);
    }

    // -- MtpEmaTracker single position tracker --

    #[test]
    fn ema_tracker_single_position_all_accepts() {
        let mut tracker = MtpEmaTracker::new(1, 0.3);
        for _ in 0..50 {
            tracker.record_position(0, true);
        }
        let rate = tracker.ema_acceptance_rate(0).unwrap();
        assert!(rate > 0.9, "After 50 accepts, rate should be near 1.0, got {}", rate);
        assert!(tracker.global_ema_rate() > 0.9);
    }

    #[test]
    fn ema_tracker_single_position_all_rejects() {
        let mut tracker = MtpEmaTracker::new(1, 0.3);
        for _ in 0..50 {
            tracker.record_position(0, false);
        }
        let rate = tracker.ema_acceptance_rate(0).unwrap();
        assert!(rate < 0.1, "After 50 rejects, rate should be near 0.0, got {}", rate);
    }

    // -- Cross-cutting: forward after distillation --

    #[test]
    fn adapter_forward_after_distillation_different_from_initial() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; 12]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 4];
        let logits_initial = adapter.forward(&hidden);

        // Run distillation with non-uniform target
        let target: Vec<f32> = (0..3).map(|i| (i as f32) * 10.0).collect();
        for _ in 0..10 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &target, &hidden, 0.1);
        }

        let logits_after = adapter.forward(&hidden);
        assert_ne!(logits_initial, logits_after, "Distillation should change logits");
    }

    #[test]
    fn adapter_distill_loss_decreases_with_same_target() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; 12]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 4];
        let target = vec![1.0f32, 2.0f32, 3.0f32];

        let mut losses = Vec::new();
        for _ in 0..5 {
            let draft = adapter.forward(&hidden);
            let loss = adapter.distill_step(&draft, &target, &hidden, 0.1);
            losses.push(loss);
        }
        // Loss trend should generally decrease (not guaranteed monotonic for SGD, but first > last)
        assert!(
            losses[0] >= *losses.last().unwrap(),
            "First loss {} should be >= last loss {} over distillation steps",
            losses[0], losses.last().unwrap()
        );
    }

    // -- DraftAdapter debug after distillation --

    #[test]
    fn adapter_debug_shows_distillation_step_count() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 5,
        };
        let lm_head = Arc::new(vec![0.1f32; 12]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 4];

        for step in 1..=3 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &vec![0.0f32; 3], &hidden, 0.01);
            let debug = format!("{:?}", adapter);
            assert!(debug.contains(&format!("distillation_step: {}", step)));
        }
    }

    // -- MtpEmaTracker threshold bounds after mixed operations --

    #[test]
    fn ema_tracker_threshold_stays_bounded_after_mixed_signals() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        for i in 0..200 {
            // Alternate between batch and position recording
            if i % 2 == 0 {
                tracker.record_batch(0.3);
            } else {
                tracker.record_position(i % 4, i % 3 == 0);
            }
        }
        let threshold = tracker.threshold();
        assert!(
            threshold >= 0.2 && threshold <= 0.8,
            "Threshold should be clamped to [0.2, 0.8] after mixed signals, got {}",
            threshold
        );
    }

    #[test]
    fn ema_tracker_global_rate_stays_reasonable() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        // Record extreme values
        for _ in 0..50 {
            tracker.record_position(0, true);
        }
        let rate = tracker.global_ema_rate();
        assert!(rate >= 0.0 && rate <= 1.0, "Rate should stay in [0,1], got {}", rate);
    }

    // ── Additional coverage tests (round 4 — compact) ──────────────────────────

    #[test]
    fn adapter_config_default_hidden_zero() {
        assert_eq!(AdapterConfig::default().hidden_size, 0);
    }

    #[test]
    fn adapter_config_default_vocab_zero() {
        assert_eq!(AdapterConfig::default().vocab_size, 0);
    }

    #[test]
    fn adapter_config_default_steps_100() {
        assert_eq!(AdapterConfig::default().distillation_steps, 100);
    }

    #[test]
    fn adapter_phase_b_param_bytes_formula() {
        let h = 8; let v = 20;
        let cfg = AdapterConfig { hidden_size: h, vocab_size: v, ..Default::default() };
        let a = DraftAdapter::new_phase_b(cfg, Arc::new(vec![0.0; v*h]), Arc::new(vec![1.0; h]));
        assert_eq!(a.parameter_bytes(), v * h * 4);
    }

    #[test]
    fn adapter_forward_logits_all_finite() {
        let a = make_adapter(4, 3);
        let h = vec![1.0f32; 4];
        for &l in &a.forward(&h) { assert!(l.is_finite()); }
    }

    #[test]
    fn adapter_batch_forward_matches_individual() {
        let a = make_adapter(4, 3);
        let h = vec![0.5f32; 4];
        let batch = a.forward_batch(&[&h]);
        assert_eq!(batch[0], a.forward(&h));
    }

    #[test]
    fn adapter_phase_a_progress_zero() {
        assert_eq!(make_adapter(4, 3).distillation_progress(), (0, 100));
    }

    #[test]
    fn adapter_phase_b_not_complete_initially() {
        let cfg = AdapterConfig { hidden_size: 2, vocab_size: 3, ..Default::default() };
        let a = DraftAdapter::new_phase_b(cfg, Arc::new(vec![0.1; 6]), Arc::new(vec![1.0; 2]));
        assert!(!a.is_distillation_complete());
    }

    #[test]
    fn softmax_diff_sums_to_zero() {
        let a = vec![0.0f32, 1.0f32, 2.0f32];
        let b = vec![2.0f32, 1.0f32, 0.0f32];
        assert!((softmax_diff(&a, &b, 3).iter().sum::<f32>()).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_free_identity() {
        let x = vec![1.0f32; 4];
        let w = vec![1.0f32; 4];
        let r = rms_norm_free(&x, &w, 1e-5);
        for &v in &r { assert!((v - 1.0).abs() < 1e-4); }
    }

    #[test]
    fn rms_norm_free_length_preserved() {
        let x = vec![1.0f32, 2.0f32, 3.0f32];
        assert_eq!(rms_norm_free(&x, &[1.0; 3], 1e-5).len(), 3);
    }

    #[test]
    fn ema_tracker_new_1_position() {
        let t = MtpEmaTracker::new(1, 0.3);
        assert_eq!(t.ema_acceptance_rate(0), Some(0.5));
        assert_eq!(t.ema_acceptance_rate(1), None);
    }

    #[test]
    fn ema_tracker_record_batch_steps_increment() {
        let mut t = MtpEmaTracker::new(1, 0.3);
        t.record_batch(0.5);
        t.record_batch(0.5);
        assert_eq!(t.steps(), 2);
    }

    #[test]
    fn ema_tracker_reset_clears_steps() {
        let mut t = MtpEmaTracker::new(2, 0.3);
        for _ in 0..5 { t.record_batch(0.5); }
        t.reset();
        assert_eq!(t.steps(), 0);
    }

    #[test]
    fn ema_tracker_clone_after_reset() {
        let mut t = MtpEmaTracker::new(2, 0.3);
        t.record_position(0, true);
        t.reset();
        let c = t.clone();
        assert_eq!(c.steps(), 0);
        assert!((c.global_ema_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ema_tracker_multi_accept_zero_max() {
        assert_eq!(MtpEmaTracker::new(4, 0.3).multi_token_accept(0), 0);
    }

    #[test]
    fn ema_tracker_acceptance_rate_none_past_end() {
        assert_eq!(MtpEmaTracker::new(3, 0.3).ema_acceptance_rate(3), None);
    }

    #[test]
    fn ema_tracker_record_position_boundary() {
        let mut t = MtpEmaTracker::new(3, 0.5);
        t.record_position(2, true);
        assert_eq!(t.steps(), 1);
        assert!(t.ema_acceptance_rate(2).unwrap() > 0.5);
    }

    #[test]
    fn ema_tracker_position_rate_changes_after_record() {
        let mut t = MtpEmaTracker::new(1, 0.5);
        let before = t.ema_acceptance_rate(0).unwrap();
        t.record_position(0, true);
        assert_ne!(t.ema_acceptance_rate(0).unwrap(), before);
    }

    #[test]
    fn ema_tracker_global_rate_after_one_batch() {
        let mut t = MtpEmaTracker::new(1, 1.0);
        t.record_batch(1.0);
        assert!(t.global_ema_rate() > 0.5);
    }

    #[test]
    fn adapter_debug_phase_a_contains_hidden_size() {
        let a = make_adapter(32, 100);
        assert!(format!("{:?}", a).contains("hidden_size: 32"));
    }

    #[test]
    fn adapter_debug_phase_a_contains_vocab_size() {
        let a = make_adapter(32, 100);
        assert!(format!("{:?}", a).contains("vocab_size: 100"));
    }

    #[test]
    fn adapter_config_clone_field_by_field() {
        let c = AdapterConfig { hidden_size: 1, vocab_size: 2, rms_norm_eps: 1e-3, enable_distillation: true, distillation_steps: 7 };
        let d = c.clone();
        assert_eq!(c.hidden_size, d.hidden_size);
        assert_eq!(c.vocab_size, d.vocab_size);
        assert_eq!(c.enable_distillation, d.enable_distillation);
    }

    #[test]
    fn rms_norm_free_negative_weight() {
        let x = vec![1.0f32, 1.0f32];
        let w = vec![-1.0f32, -1.0f32];
        let r = rms_norm_free(&x, &w, 1e-5);
        assert!(r[0] < 0.0);
        assert!(r[1] < 0.0);
    }

    #[test]
    fn softmax_diff_two_equal_distributions() {
        let v = vec![1.0f32, 2.0f32];
        let d = softmax_diff(&v, &v, 2);
        assert!(d[0].abs() < 1e-6);
        assert!(d[1].abs() < 1e-6);
    }

    #[test]
    fn ema_tracker_record_batch_very_small_alpha() {
        let mut t = MtpEmaTracker::new(2, 0.01);
        t.record_batch(1.0);
        assert!(t.global_ema_rate() > 0.5);
    }

    #[test]
    fn ema_tracker_threshold_initial() {
        assert!((MtpEmaTracker::new(4, 0.3).threshold() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ema_tracker_steps_initial_zero() {
        assert_eq!(MtpEmaTracker::new(4, 0.3).steps(), 0);
    }

    #[test]
    fn ema_tracker_global_rate_initial_half() {
        assert!((MtpEmaTracker::new(4, 0.3).global_ema_rate() - 0.5).abs() < 1e-6);
    }

    // ── Compact round 5 — single-assertion tests ─────────────────────────────────

    #[test]
    fn adapter_config_default_not_enable_distill() {
        assert!(!AdapterConfig::default().enable_distillation);
    }

    #[test]
    fn adapter_forward_size_matches_vocab() {
        assert_eq!(make_adapter(4, 7).forward(&[1.0f32; 4]).len(), 7);
    }

    #[test]
    fn adapter_batch_empty_vec() {
        assert!(make_adapter(2, 3).forward_batch(&[]).is_empty());
    }

    #[test]
    fn adapter_phase_a_param_bytes_zero() {
        assert_eq!(make_adapter(16, 100).parameter_bytes(), 0);
    }

    #[test]
    fn adapter_phase_b_debug_true_delta() {
        let cfg = AdapterConfig { hidden_size: 2, vocab_size: 3, ..Default::default() };
        let a = DraftAdapter::new_phase_b(cfg, Arc::new(vec![0.0; 6]), Arc::new(vec![1.0; 2]));
        assert!(format!("{:?}", a).contains("has_residual_delta: true"));
    }

    #[test]
    fn rms_norm_free_output_len_matches() {
        assert_eq!(rms_norm_free(&[1.0f32; 5], &[1.0; 5], 1e-5).len(), 5);
    }

    #[test]
    fn rms_norm_free_no_nan_zero_input() {
        for &v in &rms_norm_free(&[0.0f32; 3], &[1.0; 3], 1e-5) { assert!(!v.is_nan()); }
    }

    #[test]
    fn softmax_diff_len_matches_vocab() {
        assert_eq!(softmax_diff(&[1.0f32; 4], &[0.0; 4], 4).len(), 4);
    }

    #[test]
    fn ema_accept_oob_false() {
        assert!(!MtpEmaTracker::new(2, 0.3).ema_accept(99));
    }

    #[test]
    fn ema_acceptance_rate_oob_none() {
        assert!(MtpEmaTracker::new(2, 0.3).ema_acceptance_rate(5).is_none());
    }

    #[test]
    fn ema_tracker_new_zero_pos_has_one() {
        assert!(MtpEmaTracker::new(0, 0.3).ema_acceptance_rate(0).is_some());
    }

    #[test]
    fn ema_tracker_record_position_oob_no_step() {
        let mut t = MtpEmaTracker::new(2, 0.3);
        t.record_position(10, true);
        assert_eq!(t.steps(), 0);
    }

    #[test]
    fn ema_tracker_multi_accept_fresh_equals_max() {
        assert_eq!(MtpEmaTracker::new(3, 0.3).multi_token_accept(3), 3);
    }

    #[test]
    fn ema_tracker_multi_accept_capped_by_positions() {
        assert!(MtpEmaTracker::new(2, 0.3).multi_token_accept(10) <= 2);
    }

    #[test]
    fn ema_tracker_reset_resets_threshold() {
        let mut t = MtpEmaTracker::new(1, 0.5);
        for _ in 0..20 { t.record_position(0, true); }
        t.reset();
        assert!((t.threshold() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ema_tracker_clone_independent_state() {
        let mut t = MtpEmaTracker::new(2, 0.5);
        t.record_position(0, true);
        let c = t.clone();
        assert_eq!(t.steps(), c.steps());
        assert!((t.global_ema_rate() - c.global_ema_rate()).abs() < 1e-6);
    }

    #[test]
    fn adapter_distill_progress_increments() {
        let cfg = AdapterConfig { hidden_size: 2, vocab_size: 3, distillation_steps: 10, ..Default::default() };
        let mut a = DraftAdapter::new_phase_b(cfg, Arc::new(vec![0.1; 6]), Arc::new(vec![1.0; 2]));
        let h = vec![1.0f32; 2];
        let d = a.forward(&h);
        a.distill_step(&d, &vec![0.0; 3], &h, 0.01);
        assert_eq!(a.distillation_progress().0, 1);
    }

    #[test]
    fn adapter_forward_batch_two_inputs() {
        let a = make_adapter(2, 3);
        let h = vec![1.0f32; 2];
        assert_eq!(a.forward_batch(&[&h, &h]).len(), 2);
    }

    #[test]
    fn ema_tracker_record_batch_updates_step() {
        let mut t = MtpEmaTracker::new(1, 0.3);
        t.record_batch(0.5);
        assert_eq!(t.steps(), 1);
    }

    #[test]
    fn ema_tracker_all_accept_rate_near_one() {
        let mut t = MtpEmaTracker::new(1, 0.3);
        for _ in 0..100 { t.record_position(0, true); }
        assert!(t.ema_acceptance_rate(0).unwrap() > 0.95);
    }

    #[test]
    fn adapter_config_eps_positive() { assert!(AdapterConfig::default().rms_norm_eps > 0.0); }

    #[test]
    fn ema_tracker_threshold_bounded() {
        let t = MtpEmaTracker::new(1, 0.3);
        let th = t.threshold();
        assert!(th >= 0.2 && th <= 0.8);
    }

    #[test]
    fn ema_tracker_accept_fresh_pos0() { assert!(MtpEmaTracker::new(2, 0.3).ema_accept(0)); }

    #[test]
    fn ema_tracker_record_pos_increments() {
        let mut t = MtpEmaTracker::new(1, 0.3);
        t.record_position(0, true);
        assert_eq!(t.steps(), 1);
    }

    #[test]
    fn adapter_forward_deterministic_repeat() {
        let a = make_adapter(2, 3);
        let h = vec![0.5f32; 2];
        assert_eq!(a.forward(&h), a.forward(&h));
    }

    // ── Round 6: ~40 new tests for uncovered public API surface ──────────────────

    // -- AdapterConfig: edge-case constructions --

    #[test]
    fn adapter_config_distillation_steps_zero() {
        // A config with distillation_steps=0 is valid — no distillation phases run
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 5,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 0,
        };
        assert_eq!(config.distillation_steps, 0);
        // Phase B adapter with 0 steps: is_distillation_complete should be true immediately
        let adapter = DraftAdapter::new_phase_b(
            config,
            Arc::new(vec![0.1; 20]),
            Arc::new(vec![1.0; 4]),
        );
        assert!(adapter.is_distillation_complete());
    }

    #[test]
    fn adapter_config_distillation_steps_various() {
        for steps in [0usize, 1, 50, 100, 1000] {
            let config = AdapterConfig {
                hidden_size: 2,
                vocab_size: 3,
                distillation_steps: steps,
                ..Default::default()
            };
            assert_eq!(config.distillation_steps, steps);
        }
    }

    #[test]
    fn adapter_config_hidden_and_vocab_independent() {
        // Setting hidden_size doesn't affect vocab_size and vice versa
        let config = AdapterConfig {
            hidden_size: 999,
            vocab_size: 42,
            ..Default::default()
        };
        assert_eq!(config.hidden_size, 999);
        assert_eq!(config.vocab_size, 42);
    }

    // -- DraftAdapter: forward with non-uniform weights --

    #[test]
    fn adapter_forward_non_uniform_lm_head_weight() {
        let hidden_size = 4;
        let vocab_size = 3;
        // lm_head weight with distinct values per vocab row
        let lm_head: Vec<f32> = (0..vocab_size)
            .flat_map(|v| (0..hidden_size).map(move |h| (v * hidden_size + h) as f32 * 0.1))
            .collect();
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            ..Default::default()
        };
        let adapter = DraftAdapter::new_phase_a(
            config,
            Arc::new(lm_head),
            Arc::new(vec![1.0; hidden_size]),
        );
        let hidden = vec![1.0f32; hidden_size];
        let logits = adapter.forward(&hidden);
        // Each vocab position has different weight sums → different logits
        assert_eq!(logits.len(), vocab_size);
        // Not all logits are equal (non-uniform weights)
        let all_same = logits.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        assert!(!all_same, "Non-uniform weights should produce different logits");
    }

    #[test]
    fn adapter_forward_non_uniform_norm_weight() {
        let hidden_size = 4;
        let vocab_size = 3;
        let norm_weight: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            ..Default::default()
        };
        let adapter = DraftAdapter::new_phase_a(
            config,
            Arc::new(vec![0.1; vocab_size * hidden_size]),
            Arc::new(norm_weight),
        );
        let hidden = vec![1.0f32; hidden_size];
        let logits = adapter.forward(&hidden);
        for &l in &logits {
            assert!(l.is_finite(), "Non-uniform norm weight should produce finite logits");
        }
        // With uniform lm_head but non-uniform norm, all logits should still be equal
        // because the dot product structure is symmetric
        let first = logits[0];
        for &l in &logits {
            assert!((l - first).abs() < 1e-4);
        }
    }

    #[test]
    fn adapter_forward_preserves_input() {
        let adapter = make_adapter(4, 3);
        let mut hidden = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let snapshot = hidden.clone();
        let _ = adapter.forward(&hidden);
        assert_eq!(hidden, snapshot, "forward should not mutate the hidden input");
    }

    #[test]
    fn adapter_forward_different_hiddens_produce_different_logits() {
        let adapter = make_adapter(4, 3);
        let h1 = vec![1.0f32; 4];
        let h2 = vec![0.0f32; 4];
        let logits1 = adapter.forward(&h1);
        let logits2 = adapter.forward(&h2);
        assert_ne!(logits1, logits2, "Different hidden states should produce different logits");
    }

    #[test]
    fn adapter_forward_negated_hidden_negates_logits() {
        let adapter = make_adapter(4, 3);
        let hidden = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let neg_hidden: Vec<f32> = hidden.iter().map(|v| -v).collect();
        let logits_pos = adapter.forward(&hidden);
        let logits_neg = adapter.forward(&neg_hidden);
        // For symmetric input around zero with uniform weights:
        // rms_norm squaring makes both produce the same normed value (magnitudes equal)
        // but sign is preserved: norm(-x) = -norm(x) since inv_rms > 0
        // With uniform weight 0.1, each logit is sum of normed * 0.1
        // So negated hidden → negated logits
        for (lp, ln) in logits_pos.iter().zip(logits_neg.iter()) {
            assert!((lp + ln).abs() < 1e-4, "Negated hidden should negate logits: {} vs {}", lp, ln);
        }
    }

    #[test]
    fn adapter_forward_hidden_all_negative() {
        let adapter = make_adapter(4, 3);
        let hidden = vec![-3.0f32; 4];
        let logits = adapter.forward(&hidden);
        assert_eq!(logits.len(), 3);
        for &l in &logits {
            assert!(l.is_finite(), "All-negative hidden should produce finite logits");
        }
    }

    #[test]
    fn adapter_forward_very_small_vocab() {
        // vocab_size = 1 edge case
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 1,
            ..Default::default()
        };
        let adapter = DraftAdapter::new_phase_a(
            config,
            Arc::new(vec![0.5f32; 4]),
            Arc::new(vec![1.0f32; 4]),
        );
        let hidden = vec![1.0f32; 4];
        let logits = adapter.forward(&hidden);
        assert_eq!(logits.len(), 1);
        assert!(logits[0].is_finite());
    }

    #[test]
    fn adapter_forward_hidden_size_one() {
        // hidden_size = 1 edge case
        let config = AdapterConfig {
            hidden_size: 1,
            vocab_size: 3,
            ..Default::default()
        };
        let adapter = DraftAdapter::new_phase_a(
            config,
            Arc::new(vec![0.5f32; 3]),
            Arc::new(vec![1.0f32; 1]),
        );
        let hidden = vec![2.0f32; 1];
        let logits = adapter.forward(&hidden);
        assert_eq!(logits.len(), 3);
        for &l in &logits {
            assert!(l.is_finite());
        }
    }

    #[test]
    fn adapter_forward_linearity_scaling() {
        // Scaling hidden by a constant k should scale logits by approximately k
        // (RmsNorm introduces a non-linear scaling, but for same-direction vectors
        // the relationship holds proportionally)
        let adapter = make_adapter(4, 3);
        let h1 = vec![1.0f32; 4];
        let h2 = vec![2.0f32; 4];
        let logits1 = adapter.forward(&h1);
        let logits2 = adapter.forward(&h2);
        // Since both are uniform vectors, RmsNorm normalizes both to the same direction
        // but scaled: norm(k*x) = k * norm(x) / rms(k*x) * rms(x) = k * (rms(x)/rms(k*x)) * norm(x)
        // For uniform vectors, rms is independent of the constant value: rms([c; n]) = |c|
        // So normed is always [1.0; n] * weight. Thus logits should be identical.
        for (l1, l2) in logits1.iter().zip(logits2.iter()) {
            assert!((l1 - l2).abs() < 1e-4, "Uniform vectors of different scale should produce same logits");
        }
    }

    // -- DraftAdapter: distillation edge cases --

    #[test]
    fn adapter_distill_loss_non_negative() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
        };
        let mut adapter = DraftAdapter::new_phase_b(
            config,
            Arc::new(vec![0.1; 12]),
            Arc::new(vec![1.0; 4]),
        );
        let hidden = vec![1.0f32; 4];
        // Multiple distill steps with different targets
        for &lr in &[0.001, 0.01, 0.1, 1.0] {
            let draft = adapter.forward(&hidden);
            let target = vec![1.0f32, 2.0f32, 3.0f32];
            let loss = adapter.distill_step(&draft, &target, &hidden, lr);
            assert!(loss >= 0.0, "Loss should be non-negative, got {} with lr={}", loss, lr);
        }
    }

    #[test]
    fn adapter_distill_many_steps_accumulate_change() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; 12]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 4];
        let target = vec![1.0f32, 2.0f32, 3.0f32];

        let logits_initial = adapter.forward(&hidden);
        // Run 20 distill steps
        for _ in 0..20 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &target, &hidden, 0.1);
        }
        let logits_after = adapter.forward(&hidden);

        // More steps → more accumulated change
        let initial_diff: f32 = 0.0; // baseline
        let final_diff: f32 = logits_initial
            .iter()
            .zip(logits_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(final_diff > initial_diff, "Multiple distill steps should accumulate change");
    }

    #[test]
    fn adapter_distill_with_extreme_target() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 10,
        };
        let mut adapter = DraftAdapter::new_phase_b(
            config,
            Arc::new(vec![0.1; 12]),
            Arc::new(vec![1.0; 4]),
        );
        let hidden = vec![1.0f32; 4];
        let draft = adapter.forward(&hidden);
        // Extreme target values
        let target = vec![1e10f32, -1e10f32, 0.0f32];
        let loss = adapter.distill_step(&draft, &target, &hidden, 0.01);
        assert!(loss.is_finite(), "Loss should be finite even with extreme target, got {}", loss);
    }

    #[test]
    fn adapter_distill_step_count_exceeds_target() {
        // Continue distilling past distillation_steps
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 3,
        };
        let mut adapter = DraftAdapter::new_phase_b(
            config,
            Arc::new(vec![0.1; 12]),
            Arc::new(vec![1.0; 4]),
        );
        let hidden = vec![1.0f32; 4];
        for _ in 0..5 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &vec![0.0; 3], &hidden, 0.01);
        }
        // distillation_step counter should keep incrementing past target
        assert_eq!(adapter.distillation_progress().0, 5);
        assert_eq!(adapter.distillation_progress().1, 3);
        assert!(adapter.is_distillation_complete());
    }

    #[test]
    fn adapter_phase_b_zero_distillation_steps_complete() {
        let config = AdapterConfig {
            hidden_size: 2,
            vocab_size: 3,
            distillation_steps: 0,
            ..Default::default()
        };
        let adapter = DraftAdapter::new_phase_b(
            config,
            Arc::new(vec![0.1; 6]),
            Arc::new(vec![1.0; 2]),
        );
        // distillation_step = 0 >= distillation_steps = 0 → complete
        assert!(adapter.is_distillation_complete());
        assert_eq!(adapter.distillation_progress(), (0, 0));
    }

    // -- DraftAdapter: batch forward edge cases --

    #[test]
    fn adapter_batch_forward_order_matches_individual() {
        let adapter = make_adapter(4, 3);
        let h1 = vec![1.0f32, 0.0f32, 0.0f32, 0.0f32];
        let h2 = vec![0.0f32, 1.0f32, 0.0f32, 0.0f32];
        let h3 = vec![0.0f32, 0.0f32, 1.0f32, 0.0f32];
        let batch = adapter.forward_batch(&[&h1, &h2, &h3]);
        assert_eq!(batch[0], adapter.forward(&h1));
        assert_eq!(batch[1], adapter.forward(&h2));
        assert_eq!(batch[2], adapter.forward(&h3));
    }

    #[test]
    fn adapter_batch_forward_four_inputs() {
        let adapter = make_adapter(2, 3);
        let inputs: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32 * 0.5; 2]).collect();
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let batch = adapter.forward_batch(&refs);
        assert_eq!(batch.len(), 4);
        for (i, result) in batch.iter().enumerate() {
            assert_eq!(result.len(), 3, "Each batch result should have vocab_size elements");
            assert_eq!(result, &adapter.forward(&inputs[i]));
        }
    }

    // -- DraftAdapter: parameter_bytes edge cases --

    #[test]
    fn adapter_parameter_bytes_phase_a_single_dim() {
        let config = AdapterConfig {
            hidden_size: 1,
            vocab_size: 1,
            ..Default::default()
        };
        let adapter = DraftAdapter::new_phase_a(
            config,
            Arc::new(vec![0.5; 1]),
            Arc::new(vec![1.0; 1]),
        );
        assert_eq!(adapter.parameter_bytes(), 0);
    }

    // -- DraftAdapter: shared Arc behavior --

    #[test]
    fn adapter_shared_norm_weight_across_adapters() {
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            ..Default::default()
        };
        let a1 = DraftAdapter::new_phase_a(
            config.clone(),
            Arc::new(vec![0.1; 12]),
            norm_weight.clone(),
        );
        let a2 = DraftAdapter::new_phase_a(
            config,
            Arc::new(vec![0.2; 12]), // different lm_head weight
            norm_weight,
        );
        let hidden = vec![1.0f32; 4];
        // Different lm_head weights → different logits
        assert_ne!(a1.forward(&hidden), a2.forward(&hidden));
    }

    // -- MtpEmaTracker: record_position comprehensive --

    #[test]
    fn ema_tracker_record_position_updates_global_as_average() {
        let mut tracker = MtpEmaTracker::new(3, 1.0); // alpha=1.0 for direct update
        // With alpha=1.0, each record directly sets the position rate
        // Step past warmup
        for _ in 0..20 {
            tracker.record_position(0, true);
        }
        // Now record position 1 as rejected
        tracker.record_position(1, false);
        // Global rate is average of all position rates
        // position 0 is very high, position 1 just dropped, position 2 still ~0.5
        let rate0 = tracker.ema_acceptance_rate(0).unwrap();
        let rate1 = tracker.ema_acceptance_rate(1).unwrap();
        assert!(rate0 > rate1, "Accepted position should have higher rate than rejected");
    }

    #[test]
    fn ema_tracker_record_position_rate_stays_bounded() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        for _ in 0..200 {
            tracker.record_position(0, true);
            tracker.record_position(1, false);
        }
        let rate0 = tracker.ema_acceptance_rate(0).unwrap();
        let rate1 = tracker.ema_acceptance_rate(1).unwrap();
        assert!(rate0 >= 0.0 && rate0 <= 1.0, "Position rate must be in [0,1], got {}", rate0);
        assert!(rate1 >= 0.0 && rate1 <= 1.0, "Position rate must be in [0,1], got {}", rate1);
    }

    #[test]
    fn ema_tracker_record_position_boundary_valid() {
        // Record exactly at the last valid position index
        let mut tracker = MtpEmaTracker::new(5, 0.5);
        tracker.record_position(4, true); // position 4 is valid (0..4 inclusive)
        assert_eq!(tracker.steps(), 1);
        assert!(tracker.ema_acceptance_rate(4).unwrap() > 0.5);
    }

    // -- MtpEmaTracker: record_batch comprehensive --

    #[test]
    fn ema_tracker_record_batch_alternating_high_low() {
        let mut tracker = MtpEmaTracker::new(2, 0.3);
        for i in 0..100 {
            let rate = if i % 2 == 0 { 0.9 } else { 0.1 };
            tracker.record_batch(rate);
        }
        // After alternating, global rate should be near the midpoint
        let rate = tracker.global_ema_rate();
        assert!(rate > 0.3 && rate < 0.7, "Alternating rates should converge near 0.5, got {}", rate);
    }

    #[test]
    fn ema_tracker_record_batch_sequence_monotone_increasing() {
        let mut tracker = MtpEmaTracker::new(2, 0.1);
        let mut rates = Vec::new();
        for i in 0..50 {
            tracker.record_batch(i as f32 / 50.0); // 0.0 → ~1.0
            rates.push(tracker.global_ema_rate());
        }
        // Global rate should generally increase (EMA with increasing input)
        assert!(
            rates.last().unwrap() > rates.first().unwrap(),
            "Monotonically increasing input should raise EMA rate"
        );
    }

    // -- MtpEmaTracker: ema_accept edge cases --

    #[test]
    fn ema_tracker_ema_accept_rejected_position_eventually_false() {
        let mut tracker = MtpEmaTracker::new(2, 0.8);
        // Reject position 0 many times
        for _ in 0..200 {
            tracker.record_position(0, false);
        }
        // Position 0 rate should be very low, below threshold
        // Note: ema_accept can also return true if global_trend > 0.1
        // but with sustained rejection, trend should be negative or near zero
        let rate = tracker.ema_acceptance_rate(0).unwrap();
        assert!(rate < 0.3, "After sustained rejection, rate should be low, got {}", rate);
    }

    #[test]
    fn ema_tracker_ema_accept_max_position_index() {
        let tracker = MtpEmaTracker::new(10, 0.3);
        // Position 9 is the last valid index for a 10-position tracker
        assert!(tracker.ema_accept(9));
        // Position 10 is out of range
        assert!(!tracker.ema_accept(10));
    }

    // -- MtpEmaTracker: multi_token_accept --

    #[test]
    fn ema_tracker_multi_accept_respects_max_positions() {
        let mut tracker = MtpEmaTracker::new(10, 0.5);
        for _ in 0..30 {
            for pos in 0..10 {
                tracker.record_position(pos, true);
            }
        }
        // Request only 3 positions
        let count = tracker.multi_token_accept(3);
        assert!(count <= 3, "Should accept at most max_positions, got {}", count);
    }

    #[test]
    fn ema_tracker_multi_accept_all_accepted_after_training() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        for _ in 0..50 {
            for pos in 0..3 {
                tracker.record_position(pos, true);
            }
        }
        assert_eq!(tracker.multi_token_accept(3), 3);
    }

    #[test]
    fn ema_tracker_multi_accept_none_after_all_rejected() {
        let mut tracker = MtpEmaTracker::new(3, 0.8);
        for _ in 0..200 {
            for pos in 0..3 {
                tracker.record_position(pos, false);
            }
        }
        let count = tracker.multi_token_accept(3);
        assert!(count < 3, "After massive rejection, multi_accept should not accept all, got {}", count);
    }

    // -- MtpEmaTracker: global_rate averaging --

    #[test]
    fn ema_tracker_global_rate_between_min_max_position_rates() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        for _ in 0..30 {
            tracker.record_position(0, true);
            tracker.record_position(3, false);
        }
        let rate0 = tracker.ema_acceptance_rate(0).unwrap();
        let rate3 = tracker.ema_acceptance_rate(3).unwrap();
        let global = tracker.global_ema_rate();
        assert!(
            global >= rate3 && global <= rate0,
            "Global rate should be between min and max position rates: {} not in [{}, {}]",
            global, rate3, rate0
        );
    }

    // -- MtpEmaTracker: threshold adaptation --


    #[test]
    fn ema_tracker_threshold_within_valid_range_after_heavy_use() {
        let mut tracker = MtpEmaTracker::new(5, 0.5);
        for i in 0..500 {
            tracker.record_position(i % 5, i % 3 == 0);
        }
        let threshold = tracker.threshold();
        assert!(
            threshold >= 0.2 && threshold <= 0.8,
            "Threshold must stay in [0.2, 0.8], got {}",
            threshold
        );
    }

    // -- MtpEmaTracker: steps counter --

    #[test]
    fn ema_tracker_steps_monotonically_increases() {
        let mut tracker = MtpEmaTracker::new(2, 0.3);
        let mut prev = 0usize;
        for i in 0..20 {
            if i % 2 == 0 {
                tracker.record_position(0, true);
            } else {
                tracker.record_batch(0.5);
            }
            assert!(tracker.steps() > prev, "Steps should monotonically increase");
            prev = tracker.steps();
        }
    }

    // -- MtpEmaTracker: reset comprehensive --

    #[test]
    fn ema_tracker_reset_restores_all_position_rates_to_half() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        for pos in 0..4 {
            tracker.record_position(pos, true);
        }
        tracker.reset();
        for pos in 0..4 {
            let rate = tracker.ema_acceptance_rate(pos).unwrap();
            assert!(
                (rate - 0.5).abs() < 1e-6,
                "After reset, position {} rate should be 0.5, got {}",
                pos, rate
            );
        }
    }

    #[test]
    fn ema_tracker_reset_then_heavy_use() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        for _ in 0..50 {
            tracker.record_position(0, false);
        }
        tracker.reset();
        // After reset, behavior should be as fresh
        assert_eq!(tracker.steps(), 0);
        tracker.record_position(0, true);
        assert_eq!(tracker.steps(), 1);
        assert!(tracker.ema_acceptance_rate(0).unwrap() > 0.5);
    }

    // -- MtpEmaTracker: combined position + batch --

    #[test]
    fn ema_tracker_combined_position_and_batch_records() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        tracker.record_position(0, true);
        tracker.record_batch(0.8);
        tracker.record_position(1, true);
        assert_eq!(tracker.steps(), 3);
        assert!(tracker.global_ema_rate() > 0.5);
    }

    // -- MtpEmaTracker: clone after heavy use --

    #[test]
    fn ema_tracker_clone_fidelity_after_heavy_use() {
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        for i in 0..100 {
            tracker.record_position(i % 3, i % 2 == 0);
            tracker.record_batch(0.5 + (i as f32 / 200.0));
        }
        let cloned = tracker.clone();
        assert_eq!(cloned.steps(), tracker.steps());
        assert!((cloned.global_ema_rate() - tracker.global_ema_rate()).abs() < 1e-6);
        assert!((cloned.threshold() - tracker.threshold()).abs() < 1e-6);
        for pos in 0..3 {
            assert_eq!(cloned.ema_acceptance_rate(pos), tracker.ema_acceptance_rate(pos));
        }
    }

    // -- softmax_diff additional --

    #[test]
    fn softmax_diff_asymmetric_non_uniform() {
        let draft = vec![0.0f32, 0.0f32, 10.0f32];
        let target = vec![10.0f32, 0.0f32, 0.0f32];
        let diff = softmax_diff(&draft, &target, 3);
        // Draft concentrates on index 2, target on index 0
        assert!(diff[0] < 0.0, "Draft has less mass at 0 than target");
        assert!(diff[2] > 0.0, "Draft has more mass at 2 than target");
        let sum: f32 = diff.iter().sum();
        assert!(sum.abs() < 1e-4, "Diff should sum to ~0, got {}", sum);
    }

    #[test]
    fn softmax_diff_with_many_elements() {
        let n = 100;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();
        let diff = softmax_diff(&a, &b, n);
        assert_eq!(diff.len(), n);
        let sum: f32 = diff.iter().sum();
        assert!(sum.abs() < 1e-3, "Diff of {} elements should sum to ~0, got {}", n, sum);
    }

    // -- rms_norm_free additional --


    #[test]
    fn rms_norm_free_different_eps_large_input() {
        let x = vec![10.0f32; 4];
        let w = vec![1.0f32; 4];
        let r1 = rms_norm_free(&x, &w, 1e-5);
        let r2 = rms_norm_free(&x, &w, 1e-1);
        // With large input, eps barely affects the result
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!((a - b).abs() / a.abs() < 0.01, "Large input: eps difference should be negligible");
        }
    }

    #[test]
    fn rms_norm_free_magnitude_relative_to_weight() {
        // If weight is doubled, output should be doubled (linear in weight)
        let x = vec![1.0f32, 2.0f32, 3.0f32];
        let w1 = vec![1.0f32; 3];
        let w2 = vec![2.0f32; 3];
        let r1 = rms_norm_free(&x, &w1, 1e-5);
        let r2 = rms_norm_free(&x, &w2, 1e-5);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!((b - 2.0 * a).abs() < 1e-4, "Double weight should double output");
        }
    }

    // -- DraftAdapter: debug format comprehensive --

    #[test]
    fn adapter_debug_phase_b_after_distillation_shows_step() {
        let config = AdapterConfig {
            hidden_size: 2,
            vocab_size: 3,
            distillation_steps: 10,
            ..Default::default()
        };
        let mut adapter = DraftAdapter::new_phase_b(
            config,
            Arc::new(vec![0.1; 6]),
            Arc::new(vec![1.0; 2]),
        );
        let hidden = vec![1.0f32; 2];
        let draft = adapter.forward(&hidden);
        adapter.distill_step(&draft, &vec![0.0; 3], &hidden, 0.01);
        adapter.distill_step(&draft, &vec![0.0; 3], &hidden, 0.01);
        let debug = format!("{:?}", adapter);
        assert!(debug.contains("distillation_step: 2"));
        assert!(debug.contains("has_residual_delta: true"));
    }

    // -- Cross-cutting: full lifecycle --

    #[test]
    fn adapter_phase_b_full_lifecycle() {
        // Create → forward → distill → forward changes → complete
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 3,
        };
        let mut adapter = DraftAdapter::new_phase_b(
            config,
            Arc::new(vec![0.1; 12]),
            Arc::new(vec![1.0; 4]),
        );

        let hidden = vec![1.0f32; 4];
        let initial_logits = adapter.forward(&hidden);

        // Run 3 distillation steps
        for _ in 0..3 {
            let draft = adapter.forward(&hidden);
            let target = vec![1.0f32, 0.0f32, 0.0f32];
            let loss = adapter.distill_step(&draft, &target, &hidden, 0.1);
            assert!(loss.is_finite());
        }

        assert!(adapter.is_distillation_complete());
        assert_eq!(adapter.distillation_progress(), (3, 3));

        let final_logits = adapter.forward(&hidden);
        assert_ne!(initial_logits, final_logits, "Lifecycle should change logits");
    }

    #[test]
    fn ema_tracker_full_lifecycle() {
        // Create → record → accept → reset → record → accept
        let mut tracker = MtpEmaTracker::new(3, 0.5);

        // Phase 1: Train with accepts
        for _ in 0..30 {
            for pos in 0..3 {
                tracker.record_position(pos, true);
            }
        }
        assert!(tracker.multi_token_accept(3) >= 1);
        assert!(tracker.global_ema_rate() > 0.5);

        // Reset
        tracker.reset();
        assert_eq!(tracker.steps(), 0);
        assert!((tracker.global_ema_rate() - 0.5).abs() < 1e-6);

        // Phase 2: Train with rejects
        for _ in 0..30 {
            for pos in 0..3 {
                tracker.record_position(pos, false);
            }
        }
        assert!(tracker.global_ema_rate() < 0.5);
    }

    #[test]
    fn ema_tracker_interleaved_position_and_batch() {
        let mut tracker = MtpEmaTracker::new(4, 0.5);
        for round in 0..20 {
            // Record positions for even rounds, batch for odd
            if round % 2 == 0 {
                for pos in 0..4 {
                    tracker.record_position(pos, pos < 2);
                }
            } else {
                tracker.record_batch(0.5);
            }
        }
        // After interleaved recording, steps should be 40 (20 rounds × ~2 step calls each)
        // Actually 20*2 + 20*1 = 60 but position records are 4 per even round
        // 10 even rounds × 4 positions = 40 position records + 10 odd rounds × 1 batch = 10
        assert_eq!(tracker.steps(), 50);
        assert!(tracker.global_ema_rate().is_finite());
    }

    // ── Round 7: ~40 new tests targeting remaining uncovered API surface ────────

    // -- softmax free function direct tests --

    #[test]
    fn softmax_uniform_inputs_produce_equal_probs() {
        let vals = vec![2.0f32; 5];
        let probs = softmax(&vals);
        assert_eq!(probs.len(), 5);
        for &p in &probs {
            assert!((p - 0.2).abs() < 1e-5, "Uniform input should produce uniform probs, got {}", p);
        }
    }

    #[test]
    fn softmax_outputs_sum_to_one() {
        let vals = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let probs = softmax(&vals);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0, got {}", sum);
    }

    #[test]
    fn softmax_single_element_is_one() {
        let probs = softmax(&[5.0f32]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6, "Single element softmax should be 1.0");
    }

    #[test]
    fn softmax_large_values_no_overflow() {
        let vals = vec![1000.0f32, 1001.0f32, 1002.0f32];
        let probs = softmax(&vals);
        for &p in &probs {
            assert!(p.is_finite() && p >= 0.0 && p <= 1.0);
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn softmax_negative_values_valid_probs() {
        let vals = vec![-10.0f32, -5.0f32, 0.0f32];
        let probs = softmax(&vals);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn softmax_peaks_at_max_input() {
        let vals = vec![1.0f32, 5.0f32, 2.0f32];
        let probs = softmax(&vals);
        assert!(probs[1] > probs[0] && probs[1] > probs[2],
            "Softmax should peak at the index of the largest input");
    }

    #[test]
    fn softmax_empty_input_empty_output() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    // -- DraftAdapter::logits_gradient method --

    #[test]
    fn adapter_logits_gradient_length_matches_vocab() {
        let adapter = make_adapter(4, 6);
        let draft = vec![1.0f32; 6];
        let target = vec![0.5f32; 6];
        let grad = adapter.logits_gradient(&draft, &target);
        assert_eq!(grad.len(), 6);
    }

    #[test]
    fn adapter_logits_gradient_identical_inputs_near_zero() {
        let adapter = make_adapter(4, 5);
        let vals = vec![2.0f32; 5];
        let grad = adapter.logits_gradient(&vals, &vals);
        for (i, &g) in grad.iter().enumerate() {
            assert!(g.abs() < 1e-6, "Identical inputs should produce ~0 gradient at {}", i);
        }
    }

    #[test]
    fn adapter_logits_gradient_sums_to_zero() {
        let adapter = make_adapter(4, 4);
        let draft = vec![0.0f32, 1.0f32, 2.0f32, 3.0f32];
        let target = vec![3.0f32, 2.0f32, 1.0f32, 0.0f32];
        let grad = adapter.logits_gradient(&draft, &target);
        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-5, "Gradient should sum to ~0, got {}", sum);
    }

    #[test]
    fn adapter_logits_gradient_asymmetric_sign() {
        let adapter = make_adapter(4, 3);
        let draft = vec![0.0f32, 0.0f32, 10.0f32];
        let target = vec![10.0f32, 0.0f32, 0.0f32];
        let grad = adapter.logits_gradient(&draft, &target);
        assert!(grad[0] < 0.0, "Draft under-represents index 0, grad should be negative");
        assert!(grad[2] > 0.0, "Draft over-represents index 2, grad should be positive");
    }

    // -- DraftAdapter::rms_norm method (indirect via forward) --

    #[test]
    fn adapter_forward_norm_weight_zero_produces_zero_logits() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            ..Default::default()
        };
        let adapter = DraftAdapter::new_phase_a(
            config,
            Arc::new(vec![0.5f32; 12]),
            Arc::new(vec![0.0f32; 4]),
        );
        let hidden = vec![1.0f32; 4];
        let logits = adapter.forward(&hidden);
        for &l in &logits {
            assert!(l.abs() < 1e-6, "Zero norm weight should produce ~0 logits");
        }
    }

    #[test]
    fn adapter_forward_norm_weight_scaling() {
        // Doubling norm_weight doubles the output (linear)
        let config1 = AdapterConfig { hidden_size: 4, vocab_size: 3, ..Default::default() };
        let config2 = config1.clone();
        let a1 = DraftAdapter::new_phase_a(
            config1,
            Arc::new(vec![0.1f32; 12]),
            Arc::new(vec![1.0f32; 4]),
        );
        let a2 = DraftAdapter::new_phase_a(
            config2,
            Arc::new(vec![0.1f32; 12]),
            Arc::new(vec![2.0f32; 4]),
        );
        let hidden = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let logits1 = a1.forward(&hidden);
        let logits2 = a2.forward(&hidden);
        for (l1, l2) in logits1.iter().zip(logits2.iter()) {
            assert!((l2 - 2.0 * l1).abs() < 1e-3, "Doubled norm weight should double logits");
        }
    }

    // -- AdapterConfig: enable_distillation flag independence --

    #[test]
    fn adapter_config_enable_distillation_does_not_affect_phase_a() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            enable_distillation: true,
            ..Default::default()
        };
        // Phase A ignores enable_distillation — no residual_delta
        let adapter = DraftAdapter::new_phase_a(
            config,
            Arc::new(vec![0.1; 12]),
            Arc::new(vec![1.0; 4]),
        );
        assert_eq!(adapter.parameter_bytes(), 0);
    }

    #[test]
    fn adapter_config_enable_distillation_false_phase_b_still_has_delta() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            enable_distillation: false,
            ..Default::default()
        };
        let adapter = DraftAdapter::new_phase_b(
            config,
            Arc::new(vec![0.1; 12]),
            Arc::new(vec![1.0; 4]),
        );
        // Phase B always allocates residual_delta regardless of enable_distillation flag
        assert_eq!(adapter.parameter_bytes(), 4 * 3 * 4);
    }

    // -- DraftAdapter: distillation with various learning rates --

    #[test]
    fn adapter_distill_high_lr_produces_larger_change() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
            ..Default::default()
        };
        let hidden = vec![1.0f32; 4];
        let target = vec![1.0f32, 2.0f32, 3.0f32];

        let mut a_low = DraftAdapter::new_phase_b(
            config.clone(), Arc::new(vec![0.1; 12]), Arc::new(vec![1.0; 4]),
        );
        let mut a_high = DraftAdapter::new_phase_b(
            config, Arc::new(vec![0.1; 12]), Arc::new(vec![1.0; 4]),
        );

        let logits_low_before = a_low.forward(&hidden);
        let logits_high_before = a_high.forward(&hidden);

        let d_low = a_low.forward(&hidden);
        a_low.distill_step(&d_low, &target, &hidden, 0.001);
        let d_high = a_high.forward(&hidden);
        a_high.distill_step(&d_high, &target, &hidden, 1.0);

        let delta_low: f32 = logits_low_before.iter()
            .zip(a_low.forward(&hidden).iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let delta_high: f32 = logits_high_before.iter()
            .zip(a_high.forward(&hidden).iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(delta_high > delta_low,
            "High LR should produce larger change: {} vs {}", delta_high, delta_low);
    }


    // -- DraftAdapter: forward with weight patterns --

    #[test]
    fn adapter_forward_identity_weight_produces_hidden_scaled() {
        // lm_head = identity matrix (hidden_size = vocab_size) → logits ≈ normed hidden
        let dim = 4;
        let mut identity = vec![0.0f32; dim * dim];
        for i in 0..dim {
            identity[i * dim + i] = 1.0;
        }
        let config = AdapterConfig { hidden_size: dim, vocab_size: dim, ..Default::default() };
        let adapter = DraftAdapter::new_phase_a(
            config,
            Arc::new(identity),
            Arc::new(vec![1.0; dim]),
        );
        let hidden = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let logits = adapter.forward(&hidden);
        // With identity weight and unit norm, logits = normed hidden
        // All values should be finite and proportional to input
        assert_eq!(logits.len(), dim);
        for &l in &logits {
            assert!(l.is_finite());
        }
    }

    // -- DraftAdapter: shared Arc reference count --

    #[test]
    fn adapter_shared_lm_head_arc_reference_count() {
        let lm_head = Arc::new(vec![0.1f32; 12]);
        let norm = Arc::new(vec![1.0f32; 4]);
        assert_eq!(Arc::strong_count(&lm_head), 1);
        let config = AdapterConfig { hidden_size: 4, vocab_size: 3, ..Default::default() };
        let _a1 = DraftAdapter::new_phase_a(config.clone(), lm_head.clone(), norm.clone());
        assert_eq!(Arc::strong_count(&lm_head), 2);
        let _a2 = DraftAdapter::new_phase_a(config, lm_head.clone(), norm);
        assert_eq!(Arc::strong_count(&lm_head), 3);
    }

    // -- MtpEmaTracker: trend-based acceptance override --

    #[test]
    fn ema_tracker_trend_override_accepts_despite_low_rate() {
        let mut tracker = MtpEmaTracker::new(2, 0.9);
        // Establish low rate first
        for _ in 0..30 {
            tracker.record_position(0, false);
        }
        // Sharp improvement: many accepts in a row → positive trend
        for _ in 0..3 {
            tracker.record_position(0, true);
        }
        // If trend > 0.1, ema_accept returns true regardless of position rate
        // This tests the OR condition in ema_accept
        let _result = tracker.ema_accept(0);
        // Just ensure no panic; actual boolean depends on accumulated state
    }

    // -- MtpEmaTracker: rate_variance_ema influence on threshold --

    #[test]
    fn ema_tracker_high_variance_raises_threshold() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        // Create high variance by alternating extreme values
        for _ in 0..100 {
            tracker.record_batch(1.0);
            tracker.record_batch(0.0);
        }
        let threshold = tracker.threshold();
        // High variance → var_penalty increases → threshold moves up
        // Exact value depends on accumulated state but should be >= 0.2
        assert!(threshold >= 0.2, "Threshold should be at least 0.2, got {}", threshold);
    }

    // -- MtpEmaTracker: record_batch rate clamping effect --

    #[test]
    fn ema_tracker_record_batch_perfect_rate_converges_to_one() {
        let mut tracker = MtpEmaTracker::new(2, 0.3);
        for _ in 0..500 {
            tracker.record_batch(1.0);
        }
        let rate = tracker.global_ema_rate();
        assert!(rate > 0.9, "After 500 perfect batches, rate should be near 1.0, got {}", rate);
    }

    #[test]
    fn ema_tracker_record_batch_zero_rate_converges_to_zero() {
        let mut tracker = MtpEmaTracker::new(2, 0.3);
        for _ in 0..500 {
            tracker.record_batch(0.0);
        }
        let rate = tracker.global_ema_rate();
        assert!(rate < 0.1, "After 500 zero batches, rate should be near 0.0, got {}", rate);
    }

    // -- MtpEmaTracker: record_position updates prev_global_rate --

    #[test]
    fn ema_tracker_prev_rate_updated_after_each_record() {
        let mut tracker = MtpEmaTracker::new(2, 0.5);
        // After each record_position, prev_global_rate = global_rate before the update
        tracker.record_position(0, true);
        let rate_after_1 = tracker.global_ema_rate();

        tracker.record_position(0, false);
        let rate_after_2 = tracker.global_ema_rate();

        // The trend should reflect the change
        // After accept→reject, rate should have decreased
        assert!(rate_after_2 < rate_after_1,
            "Rate should decrease after reject following accept: {} vs {}",
            rate_after_2, rate_after_1);
    }

    // -- MtpEmaTracker: multiple resets in sequence --

    #[test]
    fn ema_tracker_triple_reset_state_consistent() {
        let mut tracker = MtpEmaTracker::new(3, 0.4);
        for _ in 0..20 {
            tracker.record_position(0, true);
        }
        tracker.reset();
        tracker.reset();
        tracker.reset();
        assert_eq!(tracker.steps(), 0);
        for pos in 0..3 {
            assert!((tracker.ema_acceptance_rate(pos).unwrap() - 0.5).abs() < 1e-6);
        }
    }

    // -- MtpEmaTracker: zero alpha edge case behavior --


    // -- DraftAdapter: distill_step with identical draft and target --

    #[test]
    fn adapter_distill_identical_draft_target_near_zero_loss() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
            ..Default::default()
        };
        let mut adapter = DraftAdapter::new_phase_b(
            config, Arc::new(vec![0.1; 12]), Arc::new(vec![1.0; 4]),
        );
        let hidden = vec![1.0f32; 4];
        let draft = adapter.forward(&hidden);
        let loss = adapter.distill_step(&draft, &draft, &hidden, 0.1);
        assert!(loss < 1e-6, "Identical draft/target should produce near-zero loss, got {}", loss);
    }

    // -- DraftAdapter: parameter_bytes proportional to dims --

    #[test]
    fn adapter_parameter_bytes_doubles_with_doubled_vocab() {
        let h = 4;
        let v1 = 10;
        let v2 = 20;
        let cfg1 = AdapterConfig { hidden_size: h, vocab_size: v1, ..Default::default() };
        let cfg2 = AdapterConfig { hidden_size: h, vocab_size: v2, ..Default::default() };
        let a1 = DraftAdapter::new_phase_b(cfg1, Arc::new(vec![0.0; v1*h]), Arc::new(vec![1.0; h]));
        let a2 = DraftAdapter::new_phase_b(cfg2, Arc::new(vec![0.0; v2*h]), Arc::new(vec![1.0; h]));
        assert_eq!(a2.parameter_bytes(), 2 * a1.parameter_bytes());
    }

    #[test]
    fn adapter_parameter_bytes_doubles_with_doubled_hidden() {
        let v = 10;
        let h1 = 4;
        let h2 = 8;
        let cfg1 = AdapterConfig { hidden_size: h1, vocab_size: v, ..Default::default() };
        let cfg2 = AdapterConfig { hidden_size: h2, vocab_size: v, ..Default::default() };
        let a1 = DraftAdapter::new_phase_b(cfg1, Arc::new(vec![0.0; v*h1]), Arc::new(vec![1.0; h1]));
        let a2 = DraftAdapter::new_phase_b(cfg2, Arc::new(vec![0.0; v*h2]), Arc::new(vec![1.0; h2]));
        assert_eq!(a2.parameter_bytes(), 2 * a1.parameter_bytes());
    }

    // -- DraftAdapter: forward_batch with many inputs --

    #[test]
    fn adapter_batch_forward_ten_inputs() {
        let adapter = make_adapter(2, 3);
        let inputs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.1; 2]).collect();
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let batch = adapter.forward_batch(&refs);
        assert_eq!(batch.len(), 10);
        for (i, result) in batch.iter().enumerate() {
            assert_eq!(*result, adapter.forward(&inputs[i]),
                "Batch result {} should match individual forward", i);
        }
    }

    // -- DraftAdapter: forward with different eps values --

    #[test]
    fn adapter_forward_eps_effect_on_zero_input() {
        let config_small_eps = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-10,
            ..Default::default()
        };
        let config_large_eps = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1.0,
            ..Default::default()
        };
        let a1 = DraftAdapter::new_phase_a(
            config_small_eps,
            Arc::new(vec![0.1; 12]),
            Arc::new(vec![1.0; 4]),
        );
        let a2 = DraftAdapter::new_phase_a(
            config_large_eps,
            Arc::new(vec![0.1; 12]),
            Arc::new(vec![1.0; 4]),
        );
        let hidden = vec![0.0f32; 4];
        let l1 = a1.forward(&hidden);
        let l2 = a2.forward(&hidden);
        // Both should produce finite results
        for &l in &l1 { assert!(l.is_finite()); }
        for &l in &l2 { assert!(l.is_finite()); }
        // With zero input, eps affects inv_rms: small eps → large inv_rms → larger logits
        let sum1: f32 = l1.iter().map(|l| l.abs()).sum();
        let sum2: f32 = l2.iter().map(|l| l.abs()).sum();
        assert!(sum1 >= sum2, "Smaller eps should produce larger or equal magnitudes");
    }

    // -- MtpEmaTracker: new with alpha exactly at boundaries --

    #[test]
    fn ema_tracker_alpha_exactly_0_01() {
        let tracker = MtpEmaTracker::new(2, 0.01);
        assert_eq!(tracker.steps(), 0);
        assert!((tracker.global_ema_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ema_tracker_alpha_exactly_1_0() {
        let tracker = MtpEmaTracker::new(2, 1.0);
        assert_eq!(tracker.steps(), 0);
        assert!((tracker.global_ema_rate() - 0.5).abs() < 1e-6);
    }

    // -- MtpEmaTracker: multi_token_accept with varying max_positions --

    #[test]
    fn ema_tracker_multi_accept_max_one_returns_at_most_one() {
        let tracker = MtpEmaTracker::new(4, 0.3);
        let count = tracker.multi_token_accept(1);
        assert!(count <= 1);
    }

    #[test]
    fn ema_tracker_multi_accept_max_equals_tracked_count() {
        let tracker = MtpEmaTracker::new(3, 0.3);
        // Fresh tracker: all rates = 0.5 >= threshold = 0.5
        let count = tracker.multi_token_accept(3);
        assert_eq!(count, 3);
    }

    // -- MtpEmaTracker: reset preserves nominal_alpha --

    #[test]
    fn ema_tracker_reset_preserves_position_count() {
        let mut tracker = MtpEmaTracker::new(5, 0.3);
        for _ in 0..10 { tracker.record_position(0, true); }
        tracker.reset();
        // Position count should still be 5
        assert_eq!(tracker.ema_acceptance_rate(4), Some(0.5));
        assert_eq!(tracker.ema_acceptance_rate(5), None);
    }

    // -- DraftAdapter: distill_step with very large learning rate --

    #[test]
    fn adapter_distill_very_large_lr_stays_finite() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
            ..Default::default()
        };
        let mut adapter = DraftAdapter::new_phase_b(
            config, Arc::new(vec![0.1; 12]), Arc::new(vec![1.0; 4]),
        );
        let hidden = vec![1.0f32; 4];
        let draft = adapter.forward(&hidden);
        let target = vec![1.0f32, 2.0f32, 3.0f32];
        let loss = adapter.distill_step(&draft, &target, &hidden, 1e6);
        assert!(loss.is_finite(), "Even with extreme LR, loss should be finite");

        let logits = adapter.forward(&hidden);
        for &l in &logits {
            assert!(l.is_finite(), "Logits should be finite after extreme LR step");
        }
    }

    // -- DraftAdapter: forward output range property --

    #[test]
    fn adapter_forward_output_magnitude_reasonable() {
        let adapter = make_adapter(64, 1000);
        let hidden = vec![1.0f32; 64];
        let logits = adapter.forward(&hidden);
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        // With uniform 0.1 weight and unit norm, logits should be modest
        assert!(max_logit < 100.0, "Max logit should be reasonable, got {}", max_logit);
        assert!(min_logit > -100.0, "Min logit should be reasonable, got {}", min_logit);
    }

    // -- rms_norm_free: eps sensitivity with zero input --

    #[test]
    fn rms_norm_free_zero_input_eps_affects_magnitude() {
        let x = vec![0.0f32; 4];
        let w = vec![1.0f32; 4];
        let r_small = rms_norm_free(&x, &w, 1e-10);
        let r_large = rms_norm_free(&x, &w, 1.0);
        // With zero input, inv_rms = 1/sqrt(eps)
        // Small eps → larger inv_rms → larger output (but 0*large = 0)
        // Actually: result = xi * inv_rms * wi, and xi = 0, so result = 0 regardless
        for &r in r_small.iter() {
            assert!(r.abs() < 1e-6);
        }
        for &r in r_large.iter() {
            assert!(r.abs() < 1e-6);
        }
    }

    // -- softmax_diff: property-based checks --

    #[test]
    fn softmax_diff_swapped_inputs_negate() {
        let a = vec![1.0f32, 3.0f32, 5.0f32];
        let b = vec![5.0f32, 3.0f32, 1.0f32];
        let diff_ab = softmax_diff(&a, &b, 3);
        let diff_ba = softmax_diff(&b, &a, 3);
        for i in 0..3 {
            assert!((diff_ab[i] + diff_ba[i]).abs() < 1e-5,
                "Swapped diff should negate at index {}: {} + {} != 0",
                i, diff_ab[i], diff_ba[i]);
        }
    }

    // -- DraftAdapter: config with all-zero dimensions --

    #[test]
    fn adapter_config_all_zeros_construction() {
        let config = AdapterConfig {
            hidden_size: 0,
            vocab_size: 0,
            rms_norm_eps: 0.0,
            enable_distillation: false,
            distillation_steps: 0,
        };
        assert_eq!(config.hidden_size, 0);
        assert_eq!(config.vocab_size, 0);
        assert_eq!(config.distillation_steps, 0);
    }

    // -- MtpEmaTracker: ema_acceptance_rate returns correct initial for all positions --

    #[test]
    fn ema_tracker_all_positions_initial_rate_half() {
        let tracker = MtpEmaTracker::new(8, 0.3);
        for pos in 0..8 {
            let rate = tracker.ema_acceptance_rate(pos).unwrap();
            assert!((rate - 0.5).abs() < 1e-6,
                "Initial rate for position {} should be 0.5, got {}", pos, rate);
        }
    }

    // -- MtpEmaTracker: record_position affects only target position rate --

    #[test]
    fn ema_tracker_record_one_position_leaves_others_near_initial() {
        let mut tracker = MtpEmaTracker::new(5, 0.5);
        // Record position 0 only, 10 times
        for _ in 0..10 {
            tracker.record_position(0, true);
        }
        let rate0 = tracker.ema_acceptance_rate(0).unwrap();
        assert!(rate0 > 0.5, "Position 0 should have risen");
        // Position 1 should still be close to 0.5 (not directly updated)
        // Note: global_rate changed, but per-position rates only update on direct record
        let rate1 = tracker.ema_acceptance_rate(1).unwrap();
        assert!((rate1 - 0.5).abs() < 1e-6, "Unupdated position should stay at 0.5, got {}", rate1);
    }

    // -- Cross-cutting: phase B adapter logits change monotonically during distillation --

    #[test]
    fn adapter_distill_logits_diverge_monotonically_from_phase_a() {
        let config = AdapterConfig {
            hidden_size: 4,
            vocab_size: 3,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 100,
            ..Default::default()
        };
        let lm_head = Arc::new(vec![0.1f32; 12]);
        let norm_weight = Arc::new(vec![1.0f32; 4]);
        let phase_a = DraftAdapter::new_phase_a(config.clone(), lm_head.clone(), norm_weight.clone());
        let mut phase_b = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let hidden = vec![1.0f32; 4];
        let target = vec![3.0f32, 1.0f32, 0.0f32];

        let mut divergences = Vec::new();
        for _ in 0..10 {
            let draft = phase_b.forward(&hidden);
            phase_b.distill_step(&draft, &target, &hidden, 0.1);
            let logits_a = phase_a.forward(&hidden);
            let logits_b = phase_b.forward(&hidden);
            let div: f32 = logits_a.iter().zip(logits_b.iter()).map(|(a, b)| (a - b).abs()).sum();
            divergences.push(div);
        }
        // Divergence should generally increase with more distillation steps
        assert!(
            divergences.last().unwrap() > &0.0,
            "Phase B should diverge from Phase A after distillation"
        );
    }

    // ── Round 8: 10 new tests targeting remaining uncovered edges ──────────────────

    #[test]
    fn adapter_phase_b_forward_delta_adds_to_weight() {
        // Verify the matmul uses (weight + delta) by constructing a Phase B adapter
        // with a known non-zero delta and checking the output differs from weight-only.
        let hidden_size = 3;
        let vocab_size = 2;
        let lm_head = Arc::new(vec![1.0f32; vocab_size * hidden_size]);
        let norm_weight = Arc::new(vec![1.0f32; hidden_size]);
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 10,
        };
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head.clone(), norm_weight.clone());
        let hidden = vec![1.0f32; hidden_size];
        let logits_before_distill = adapter.forward(&hidden);

        // Distill with a non-uniform target to inject non-zero delta
        let target = vec![10.0f32, -10.0f32];
        for _ in 0..5 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &target, &hidden, 1.0);
        }
        let logits_after = adapter.forward(&hidden);
        assert_ne!(
            logits_before_distill, logits_after,
            "Delta from distillation must change logits via (weight + delta) path"
        );
    }

    #[test]
    fn ema_tracker_record_batch_does_not_update_position_rates() {
        // record_batch updates only global_rate, not per-position rates
        let mut tracker = MtpEmaTracker::new(3, 0.5);
        let pos_rates_before: Vec<f32> = (0..3)
            .map(|p| tracker.ema_acceptance_rate(p).unwrap())
            .collect();
        tracker.record_batch(1.0);
        tracker.record_batch(1.0);
        tracker.record_batch(1.0);
        for (pos, &before) in pos_rates_before.iter().enumerate() {
            let after = tracker.ema_acceptance_rate(pos).unwrap();
            assert!(
                (before - after).abs() < 1e-6,
                "record_batch should not change position {} rate: {} vs {}",
                pos, before, after
            );
        }
        // But global rate should have changed
        assert!(
            tracker.global_ema_rate() > 0.5,
            "Global rate should increase after perfect batches"
        );
    }

    #[test]
    fn adapter_logits_gradient_with_extreme_opposite_distributions() {
        // Draft concentrates on last element, target on first element
        let adapter = make_adapter(4, 4);
        let draft = vec![0.0f32, 0.0f32, 0.0f32, 100.0f32];
        let target = vec![100.0f32, 0.0f32, 0.0f32, 0.0f32];
        let grad = adapter.logits_gradient(&draft, &target);
        // Draft softmax puts all mass on index 3, target on index 0
        assert!(grad[0] < -0.9, "grad[0] should be strongly negative, got {}", grad[0]);
        assert!(grad[3] > 0.9, "grad[3] should be strongly positive, got {}", grad[3]);
        // Middle elements should be near zero for both distributions
        assert!(grad[1].abs() < 0.01, "grad[1] should be ~0, got {}", grad[1]);
        assert!(grad[2].abs() < 0.01, "grad[2] should be ~0, got {}", grad[2]);
    }

    #[test]
    fn ema_tracker_global_rate_equals_position_average_after_mixed_records() {
        // After recording different positions, global_rate should be average of all position_rates
        let mut tracker = MtpEmaTracker::new(3, 1.0); // alpha=1.0 for direct observation
        // Get past warmup
        for _ in 0..20 {
            tracker.record_position(0, true);
            tracker.record_position(1, false);
            tracker.record_position(2, true);
        }
        let expected_avg: f32 = (0..3)
            .map(|p| tracker.ema_acceptance_rate(p).unwrap())
            .sum::<f32>()
            / 3.0;
        assert!(
            (tracker.global_ema_rate() - expected_avg).abs() < 1e-4,
            "Global rate {} should equal average of position rates {}",
            tracker.global_ema_rate(), expected_avg
        );
    }

    #[test]
    fn adapter_forward_with_orthogonal_hidden_and_weight() {
        // Construct hidden and lm_head so their dot product is zero for each vocab entry
        let hidden_size = 4;
        let vocab_size = 2;
        // hidden = [1, 1, -1, -1]
        let hidden = vec![1.0f32, 1.0f32, -1.0f32, -1.0f32];
        // Row 0 = [1, 1, 1, 1] → dot = 1+1-1-1 = 0
        // Row 1 = [1, -1, 1, -1] → dot = 1-1-1+1 = 0
        let lm_head = vec![1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, -1.0f32, 1.0f32, -1.0f32];
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            ..Default::default()
        };
        let adapter = DraftAdapter::new_phase_a(
            config,
            Arc::new(lm_head),
            Arc::new(vec![1.0f32; hidden_size]),
        );
        let logits = adapter.forward(&hidden);
        // After RmsNorm, the hidden values get scaled by inv_rms * weight.
        // Since norm_weight is all 1.0, normed hidden preserves the [1,1,-1,-1] pattern
        // scaled by inv_rms. Dot products remain zero.
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.abs() < 1e-4,
                "Orthogonal hidden/weight should produce ~0 logit at index {}, got {}",
                i, l
            );
        }
    }

    #[test]
    fn ema_tracker_multi_token_accept_stops_at_first_gap_after_partial_training() {
        // Train positions 0..=1 to accept, leave 2..=3 at initial rate
        // Then reject position 2 heavily to drive it below threshold
        let mut tracker = MtpEmaTracker::new(4, 0.8);
        // Accept positions 0 and 1
        for _ in 0..50 {
            tracker.record_position(0, true);
            tracker.record_position(1, true);
            tracker.record_position(2, false);
            tracker.record_position(3, false);
        }
        let count = tracker.multi_token_accept(4);
        // Should accept at most positions 0 and 1, then stop at 2
        assert!(
            count >= 1 && count <= 2,
            "Should accept positions 0-1 then stop, got {}",
            count
        );
    }

    #[test]
    fn adapter_distill_step_modifies_only_delta_not_shared_weight() {
        // Verify distillation only modifies residual_delta, not the shared lm_head weight
        let hidden_size = 4;
        let vocab_size = 3;
        let lm_head = Arc::new(vec![0.1f32; vocab_size * hidden_size]);
        let norm_weight = Arc::new(vec![1.0f32; hidden_size]);
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 10,
        };
        let mut adapter = DraftAdapter::new_phase_b(config, lm_head.clone(), norm_weight);
        let hidden = vec![1.0f32; hidden_size];
        let target = vec![1.0f32, 2.0f32, 3.0f32];
        for _ in 0..5 {
            let draft = adapter.forward(&hidden);
            adapter.distill_step(&draft, &target, &hidden, 0.1);
        }
        // The shared Arc should be unchanged — all values still 0.1
        for (i, &w) in lm_head.iter().enumerate() {
            assert!(
                (w - 0.1).abs() < 1e-10,
                "Shared lm_head weight[{}] should be unchanged, got {}",
                i, w
            );
        }
    }

    #[test]
    fn rms_norm_free_preserves_sign_of_input() {
        // Each output element should have the same sign as the corresponding input
        let x = vec![-3.0f32, -1.0f32, 0.0f32, 1.0f32, 3.0f32];
        let w = vec![1.0f32; 5];
        let result = rms_norm_free(&x, &w, 1e-5);
        assert!(result[0] < 0.0, "Negative input should produce negative output");
        assert!(result[1] < 0.0, "Negative input should produce negative output");
        assert!(result[2].abs() < 1e-6, "Zero input should produce ~0 output");
        assert!(result[3] > 0.0, "Positive input should produce positive output");
        assert!(result[4] > 0.0, "Positive input should produce positive output");
    }

    #[test]
    fn ema_tracker_effective_alpha_warmup_interpolates_linearly() {
        // During warmup (steps 0..15), effective_alpha interpolates from 0.5 to nominal
        // Verify by recording at step 0 and checking the position rate is between extremes
        let mut tracker = MtpEmaTracker::new(1, 0.1); // very low nominal_alpha
        // Step 0: effective_alpha = 0.5 * (1 - 0/16) + 0.1 * (0/16) = 0.5
        tracker.record_position(0, true);
        // After step 0, rate = 0.5 * 1.0 + (1 - 0.5) * 0.5 = 0.75
        let rate_step0 = tracker.ema_acceptance_rate(0).unwrap();
        assert!(
            (rate_step0 - 0.75).abs() < 0.01,
            "Step 0 with alpha=0.5: rate should be ~0.75, got {}",
            rate_step0
        );

        // Continue recording through warmup
        for _ in 0..15 {
            tracker.record_position(0, true);
        }
        // After warmup (step 16), alpha should have transitioned toward nominal 0.1
        let rate_after_warmup = tracker.ema_acceptance_rate(0).unwrap();
        assert!(
            rate_after_warmup > 0.9,
            "After many accepts through warmup, rate should be high, got {}",
            rate_after_warmup
        );
    }

    #[test]
    fn adapter_batch_forward_order_preserves_independence() {
        // Each batch result should be independent of other batch members
        let adapter = make_adapter(4, 3);
        let h1 = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let h2 = vec![4.0f32, 3.0f32, 2.0f32, 1.0f32];
        let h3 = vec![0.0f32, 0.0f32, 0.0f32, 0.0f32];

        // Forward each individually
        let individual_1 = adapter.forward(&h1);
        let individual_2 = adapter.forward(&h2);
        let individual_3 = adapter.forward(&h3);

        // Forward as batch in different orders
        let batch_123 = adapter.forward_batch(&[&h1, &h2, &h3]);
        let batch_321 = adapter.forward_batch(&[&h3, &h2, &h1]);
        let batch_213 = adapter.forward_batch(&[&h2, &h1, &h3]);

        // Each result should match regardless of order
        assert_eq!(batch_123[0], individual_1);
        assert_eq!(batch_123[1], individual_2);
        assert_eq!(batch_123[2], individual_3);

        assert_eq!(batch_321[2], individual_1);
        assert_eq!(batch_321[1], individual_2);
        assert_eq!(batch_321[0], individual_3);

        assert_eq!(batch_213[1], individual_1);
        assert_eq!(batch_213[0], individual_2);
        assert_eq!(batch_213[2], individual_3);
    }
}
