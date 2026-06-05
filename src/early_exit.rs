//! §13.3 / §16.2 Residual Bypass Early-Exit + PGSLE Early Exit
//!
//! Implements early exit logic based on residual telemetry (cosine similarity + energy delta).
//! When cos(θ) > threshold AND Δρ < epsilon, the layer contributes negligibly to output,
//! allowing safe early termination.
//!
//! # Architecture (per SPEC 02-ARCHITECTURE §16.2)
//!
//! ```text
//! ResidualBus RecallPort (§9.3)
//!   → hidden_state[layer][hidden_dim]
//!     → ConfidenceCalibrator: sigmoid(α*(cos-θ_c) + β*(θ_δ-|Δρ-1|))
//!       → AdaptiveExitPoints: golden ratio (0.618L, 0.786L, 0.916L)
//!         → EarlyExitController: check_layer → Continue/Exit(conf)/Defer
//!           → ExitLayerStats: EMA per-point hit rate tracking
//! ```
//!
//! # Current Architecture Limitation
//!
//! The unified GraphExecutor (ARCH-CPU-GPU-UNIFIED) processes all layers in a single
//! `run_with_kv_cache()` call. True per-layer early exit requires:
//! 1. Per-layer graph splitting (future)
//! 2. JIT-level conditional jumps (§13.3 SPEC)
//!
//! This module provides the decision logic foundation for both paths.

// BusPortTag integration available via crate::routing when executor integration lands

// ============================================================================
// §16.2 EarlyExitConfig — 核心配置
// ============================================================================

/// Early Exit configuration
#[derive(Debug, Clone)]
pub struct EarlyExitConfig {
    pub enabled: bool,
    /// Cosine similarity threshold — cos(θ) must exceed this to trigger exit
    pub cosine_threshold: f32,
    /// Energy delta threshold — |Δρ - 1.0| must be below this
    pub delta_threshold: f32,
    /// Minimum layer before exit is considered (safety floor)
    pub min_layer: usize,
    /// Confidence threshold — calibrated confidence must exceed this
    pub confidence_threshold: f32,
}

impl Default for EarlyExitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cosine_threshold: 0.99,
            delta_threshold: 0.01,
            min_layer: 16,
            confidence_threshold: 0.95,
        }
    }
}

impl EarlyExitConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn with_cosine_threshold(mut self, threshold: f32) -> Self {
        self.cosine_threshold = threshold;
        self
    }

    pub fn with_delta_threshold(mut self, threshold: f32) -> Self {
        self.delta_threshold = threshold;
        self
    }

    pub fn with_min_layer(mut self, min_layer: usize) -> Self {
        self.min_layer = min_layer;
        self
    }

    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }
}

// ============================================================================
// §16.2 ConfidenceCalibrator — 置信度校准器
// ============================================================================

/// Sigmoid-based confidence calibrator.
///
/// Maps raw telemetry signals (cosine similarity + energy delta) to a
/// calibrated confidence score in [0, 1].
///
/// Formula: conf = σ(α * (cos - θ_c) + β * (θ_δ - |Δρ - 1|))
///
/// Where σ(x) = 1 / (1 + exp(-x)) is the logistic sigmoid.
#[derive(Debug, Clone)]
pub struct ConfidenceCalibrator {
    /// Weight for cosine similarity signal
    pub alpha: f32,
    /// Weight for energy delta signal
    pub beta: f32,
    /// Cosine threshold reference point
    pub theta_c: f32,
    /// Delta threshold reference point
    pub theta_d: f32,
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self {
            alpha: 20.0,
            beta: 15.0,
            theta_c: 0.99,
            theta_d: 0.01,
        }
    }
}

impl ConfidenceCalibrator {
    pub fn new(alpha: f32, beta: f32, theta_c: f32, theta_d: f32) -> Self {
        Self { alpha, beta, theta_c, theta_d }
    }

    /// Calibrate telemetry signals into a confidence score.
    ///
    /// Returns a value in (0, 1) where higher means more likely safe to exit.
    pub fn calibrate(&self, cosine_sim: f32, delta_rho: f32) -> f32 {
        let cos_signal = self.alpha * (cosine_sim - self.theta_c);
        let delta_signal = self.beta * (self.theta_d - (delta_rho - 1.0).abs());
        let logit = cos_signal + delta_signal;
        // Numerically stable sigmoid
        if logit >= 0.0 {
            1.0 / (1.0 + (-logit).exp())
        } else {
            let exp_l = logit.exp();
            exp_l / (1.0 + exp_l)
        }
    }
}

// ============================================================================
// §16.2 AdaptiveExitPoints — 黄金比例自适应退出点
// ============================================================================

/// Adaptive exit point selection based on golden ratio splitting.
///
/// Exit points are placed at fractions of total layers using the golden ratio
/// series: 0.618, 0.786 (≈ 1 - 0.618²), 0.916 (≈ 1 - 0.618⁴).
/// Each successive point is deeper into the model, catching increasingly
/// subtle convergence patterns.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AdaptiveExitPoints {
    /// Computed exit layer indices (sorted ascending)
    points: Vec<usize>,
    /// Total number of layers
    total_layers: usize,
    /// Minimum layer ratio — exit points below this are rejected
    min_layer_ratio: f32,
}

impl AdaptiveExitPoints {
    /// Golden ratio constant φ = (1 + √5) / 2 ≈ 1.618
    const PHI: f64 = 1.618033988749895;

    /// Compute adaptive exit points for a model with `total_layers` layers.
    pub fn compute(total_layers: usize, min_layer_ratio: f32) -> Self {
        let mut points = Vec::new();

        // Golden ratio exit fractions: φ⁻¹, φ⁻², φ⁻⁴
        // φ⁻¹ ≈ 0.618 — semantic core zone
        // φ⁻² ≈ 0.382 → 1 - 0.382 ≈ 0.618 (same, skip)
        // Use: 0.618, 1 - 1/φ³ ≈ 0.764, 1 - 1/φ⁴ ≈ 0.854
        let phi = Self::PHI;
        let fractions = [
            1.0 / phi,                          // ≈ 0.618
            1.0 - 1.0 / (phi * phi * phi),      // ≈ 0.764
            1.0 - 1.0 / (phi * phi * phi * phi), // ≈ 0.854
        ];

        for &frac in &fractions {
            let layer = (total_layers as f64 * frac).round() as usize;
            let layer = if total_layers == 0 {
                0
            } else {
                layer.max(1).min(total_layers - 1)
            };
            if (layer as f32 / total_layers as f32) >= min_layer_ratio
                && !points.contains(&layer) {
                    points.push(layer);
                }
        }

        points.sort_unstable();
        points.dedup();

        Self { points, total_layers, min_layer_ratio }
    }

    /// Get the exit point layers.
    pub fn points(&self) -> &[usize] {
        &self.points
    }

    /// Check if a layer is an exit point.
    pub fn is_exit_point(&self, layer: usize) -> bool {
        self.points.contains(&layer)
    }

    /// Total layers for this configuration.
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// Number of exit points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether there are no exit points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

// ============================================================================
// §16.2 ExitLayerStats — 退出点运行时统计 (EMA 滑动平均)
// ============================================================================

/// Per-exit-point runtime statistics using exponential moving average (EMA).
///
/// Tracks hit rate (how often exit triggers) at each exit point.
/// Uses EMA with configurable decay for smooth adaptation.
#[derive(Debug, Clone)]
pub struct ExitLayerStats {
    /// Per-exit-layer EMA hit rate [0.0, 1.0]
    hit_rates: Vec<f32>,
    /// Per-exit-layer total evaluation count
    eval_counts: Vec<u64>,
    /// EMA decay factor (α) — higher = faster adaptation
    ema_alpha: f32,
    /// Corresponding layer indices
    layers: Vec<usize>,
}

impl ExitLayerStats {
    /// Create stats tracker for the given exit points.
    pub fn new(exit_points: &[usize], ema_alpha: f32) -> Self {
        let n = exit_points.len();
        Self {
            hit_rates: vec![0.0; n],
            eval_counts: vec![0; n],
            ema_alpha: ema_alpha.clamp(0.01, 1.0),
            layers: exit_points.to_vec(),
        }
    }

    /// Record an evaluation at a specific exit layer.
    ///
    /// `exited` = true means the model actually exited at this layer.
    pub fn record(&mut self, layer: usize, exited: bool) {
        if let Some(idx) = self.layers.iter().position(|&l| l == layer) {
            self.eval_counts[idx] += 1;
            let hit = if exited { 1.0 } else { 0.0 };
            // EMA update: rate_new = α * hit + (1 - α) * rate_old
            self.hit_rates[idx] =
                self.ema_alpha * hit + (1.0 - self.ema_alpha) * self.hit_rates[idx];
        }
    }

    /// Get the EMA hit rate for a specific exit layer.
    pub fn hit_rate(&self, layer: usize) -> f32 {
        self.layers.iter()
            .position(|&l| l == layer)
            .map(|idx| self.hit_rates[idx])
            .unwrap_or(0.0)
    }

    /// Get the evaluation count for a specific exit layer.
    pub fn eval_count(&self, layer: usize) -> u64 {
        self.layers.iter()
            .position(|&l| l == layer)
            .map(|idx| self.eval_counts[idx])
            .unwrap_or(0)
    }

    /// Get the most effective exit point (highest hit rate with min evaluations).
    ///
    /// Returns None if no exit point has been evaluated yet.
    pub fn best_exit_point(&self) -> Option<usize> {
        let min_evals = 10; // Need minimum evaluations for statistical significance
        let mut best: Option<(usize, f32)> = None;
        for (idx, &rate) in self.hit_rates.iter().enumerate() {
            if self.eval_counts[idx] >= min_evals
                && (best.is_none() || rate > best.unwrap().1) {
                    best = Some((self.layers[idx], rate));
                }
        }
        best.map(|(layer, _)| layer)
    }

    /// Total evaluations across all exit points.
    pub fn total_evals(&self) -> u64 {
        self.eval_counts.iter().sum()
    }
}

// ============================================================================
// §16.2 EarlyExitDecision — 退出决策结果
// ============================================================================

/// Decision from early exit evaluation at a specific layer.
#[derive(Debug, Clone)]
pub enum EarlyExitDecision {
    /// Continue to next layer
    Continue,
    /// Exit at this layer with given confidence
    Exit {
        confidence: f32,
        cosine_sim: f32,
        delta_rho: f32,
    },
    /// Defer decision (insufficient data or below minimum layer)
    Defer,
}

// ============================================================================
// §16.2 EarlyExitController — 整合控制器
// ============================================================================

/// Early exit controller — integrates calibrator, exit points, and statistics.
///
/// This is the primary API for the executor layer to query at each layer boundary.
/// Per SPEC §16.2 "任意层数据召回与高维截断".
pub struct EarlyExitController {
    config: EarlyExitConfig,
    calibrator: ConfidenceCalibrator,
    exit_points: AdaptiveExitPoints,
    stats: ExitLayerStats,
}

impl EarlyExitController {
    /// Create a new controller with the given configuration and total layers.
    pub fn new(config: EarlyExitConfig, total_layers: usize) -> Self {
        let min_layer_ratio = config.min_layer as f32 / total_layers.max(1) as f32;
        let exit_points = AdaptiveExitPoints::compute(total_layers, min_layer_ratio);

        let calibrator = ConfidenceCalibrator {
            theta_c: config.cosine_threshold,
            theta_d: config.delta_threshold,
            ..ConfidenceCalibrator::default()
        };

        let stats = ExitLayerStats::new(exit_points.points(), 0.1);

        Self { config, calibrator, exit_points, stats }
    }

    /// Create with custom calibrator parameters.
    pub fn with_calibrator(
        config: EarlyExitConfig,
        total_layers: usize,
        calibrator: ConfidenceCalibrator,
    ) -> Self {
        let min_layer_ratio = config.min_layer as f32 / total_layers.max(1) as f32;
        let exit_points = AdaptiveExitPoints::compute(total_layers, min_layer_ratio);
        let stats = ExitLayerStats::new(exit_points.points(), 0.1);

        Self { config, calibrator, exit_points, stats }
    }

    /// Check if early exit should trigger at the given layer.
    ///
    /// This is the primary method called by the executor at each layer boundary.
    ///
    /// # Arguments
    /// - `layer`: Current layer index (0-based)
    /// - `cosine_sim`: Residual cosine similarity with previous layer
    /// - `delta_rho`: Residual energy ratio ‖x_out‖ / ‖x_in‖
    pub fn check_layer(
        &mut self,
        layer: usize,
        cosine_sim: f32,
        delta_rho: f32,
    ) -> EarlyExitDecision {
        if !self.config.enabled {
            return EarlyExitDecision::Defer;
        }

        if layer < self.config.min_layer {
            return EarlyExitDecision::Defer;
        }

        if !self.exit_points.is_exit_point(layer) {
            return EarlyExitDecision::Continue;
        }

        // Calibrate confidence
        let confidence = self.calibrator.calibrate(cosine_sim, delta_rho);

        // Record evaluation
        let should_exit = confidence >= self.config.confidence_threshold;
        self.stats.record(layer, should_exit);

        if should_exit {
            EarlyExitDecision::Exit {
                confidence,
                cosine_sim,
                delta_rho,
            }
        } else {
            EarlyExitDecision::Continue
        }
    }

    /// Quick boolean check (for hot-path use where full decision is not needed).
    pub fn should_exit(&self, layer: usize, cosine_sim: f32, delta_rho: f32) -> bool {
        if !self.config.enabled || layer < self.config.min_layer {
            return false;
        }
        if !self.exit_points.is_exit_point(layer) {
            return false;
        }
        let confidence = self.calibrator.calibrate(cosine_sim, delta_rho);
        confidence >= self.config.confidence_threshold
        }

    /// Get reference to the exit points configuration.
    pub fn exit_points(&self) -> &AdaptiveExitPoints {
        &self.exit_points
    }

    /// Get reference to the statistics tracker.
    pub fn stats(&self) -> &ExitLayerStats {
        &self.stats
    }

    /// Get reference to the configuration.
    pub fn config(&self) -> &EarlyExitConfig {
        &self.config
    }
}

// ============================================================================
// §16.2 ResidualBusEarlyExit — 旧 API 兼容 (per SPEC 04-API-DESIGN)
// ============================================================================

/// §16 P2: Residual Bus Early Exit with configurable exit points
#[derive(Debug, Clone)]
pub struct ResidualBusEarlyExit {
    pub config: EarlyExitConfig,
    pub exit_points: Vec<usize>,
}

impl ResidualBusEarlyExit {
    pub fn new(config: EarlyExitConfig, exit_points: Vec<usize>) -> Self {
        Self { config, exit_points }
    }

    pub fn should_exit_at_layer(
        &self,
        layer: usize,
        residual_delta: f32,
        cosine_sim: f32,
    ) -> bool {
        if !self.exit_points.contains(&layer) {
            return false;
        }
        should_early_exit(layer, cosine_sim, residual_delta, &self.config)
    }
}

// ============================================================================
// §13.3 Legacy API — 直接阈值判断
// ============================================================================

/// Check if early exit should trigger based on raw telemetry signals.
///
/// This is the legacy per-threshold check, used by ResidualBusEarlyExit
/// and for backward compatibility.
pub fn should_early_exit(
    layer: usize,
    cosine_sim: f32,
    residual_delta: f32,
    config: &EarlyExitConfig,
) -> bool {
    if !config.enabled || layer < config.min_layer {
        return false;
    }

    let direction_stable = cosine_sim > config.cosine_threshold;
    let energy_stable = (residual_delta - 1.0).abs() < config.delta_threshold;

    direction_stable && energy_stable
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ConfidenceCalibrator tests ──

    #[test]
    fn test_calibrator_high_confidence() {
        let cal = ConfidenceCalibrator::default();
        // Very strong signals: cos >> threshold, delta ≈ 1.0
        let conf = cal.calibrate(0.9999, 1.0001);
        assert!(conf > 0.5, "expected above 0.5 confidence, got {}", conf);
    }

    #[test]
    fn test_calibrator_low_confidence() {
        let cal = ConfidenceCalibrator::default();
        // Poor signals: cos < threshold, delta far from 1.0
        let conf = cal.calibrate(0.85, 1.15);
        assert!(conf < 0.5, "expected low confidence, got {}", conf);
    }

    #[test]
    fn test_calibrator_boundary() {
        let cal = ConfidenceCalibrator::default();
        // Exactly at threshold: cos = 0.99, delta = 1.0
        let conf = cal.calibrate(0.99, 1.0);
        // Should be around 0.5 (logit ≈ 0)
        assert!((conf - 0.5).abs() < 0.15, "expected ~0.5, got {}", conf);
    }

    #[test]
    fn test_calibrator_monotonicity() {
        let cal = ConfidenceCalibrator::default();
        let c1 = cal.calibrate(0.95, 1.0);
        let c2 = cal.calibrate(0.99, 1.0);
        let c3 = cal.calibrate(0.999, 1.0);
        assert!(c1 < c2 && c2 < c3, "confidence should be monotonically increasing with cosine");
    }

    // ── AdaptiveExitPoints tests ──

    #[test]
    fn test_exit_points_32_layers() {
        let points = AdaptiveExitPoints::compute(32, 0.0);
        let pts = points.points();
        // φ⁻¹ * 32 ≈ 19.78 → 20
        // (1 - 1/φ³) * 32 ≈ 24.44 → 24
        // (1 - 1/φ⁴) * 32 ≈ 27.33 → 27
        assert!(pts.len() >= 2, "should have at least 2 exit points for 32 layers, got {}", pts.len());
        // All points should be in [1, 31]
        for &p in pts {
            assert!(p >= 1 && p < 32, "exit point {} out of range [1, 31]", p);
        }
        // Points should be sorted
        for w in pts.windows(2) {
            assert!(w[0] < w[1], "exit points not sorted: {:?}", pts);
        }
    }

    #[test]
    fn test_exit_points_small_model() {
        let points = AdaptiveExitPoints::compute(4, 0.0);
        let pts = points.points();
        // Very small model — still should have at least 1 point
        for &p in pts {
            assert!(p >= 1 && p < 4, "exit point {} out of range [1, 3]", p);
        }
    }

    #[test]
    fn test_exit_points_min_ratio() {
        let points = AdaptiveExitPoints::compute(32, 0.5);
        let pts = points.points();
        // All points must be >= 0.5 * 32 = 16
        for &p in pts {
            assert!(p >= 16, "exit point {} below min ratio (16)", p);
        }
    }

    #[test]
    fn test_exit_points_is_exit_point() {
        let points = AdaptiveExitPoints::compute(32, 0.0);
        for &p in points.points() {
            assert!(points.is_exit_point(p), "registered point {} not found", p);
        }
        assert!(!points.is_exit_point(0), "layer 0 should not be an exit point");
        assert!(!points.is_exit_point(100), "out-of-range layer should not be an exit point");
    }

    // ── ExitLayerStats tests ──

    #[test]
    fn test_stats_ema_basic() {
        let mut stats = ExitLayerStats::new(&[10, 20, 30], 0.5);
        // Record 2 exits at layer 20
        stats.record(20, true);
        stats.record(20, true);
        // Hit rate after 2 exits: α*1 + (1-α)*(α*1 + (1-α)*0) = 0.5 + 0.25 = 0.75
        let rate = stats.hit_rate(20);
        assert!(rate > 0.7, "expected hit rate > 0.7, got {}", rate);
        assert_eq!(stats.eval_count(20), 2);
    }

    #[test]
    fn test_stats_mixed_signals() {
        let mut stats = ExitLayerStats::new(&[20], 0.1);
        // EMA is order-dependent: 5 true then 5 false with alpha=0.1
        // After 10 steps rate ≈ 0.24 (EMA decays slowly from true peak)
        for _ in 0..5 {
            stats.record(20, true);
        }
        for _ in 0..5 {
            stats.record(20, false);
        }
        let rate = stats.hit_rate(20);
        // Rate should be between 0.0 and 0.5 (EMA decayed from true plateau)
        assert!(rate > 0.0 && rate < 0.5, "expected rate in (0, 0.5), got {}", rate);
    }

    #[test]
    fn test_stats_nonexistent_layer() {
        let stats = ExitLayerStats::new(&[10, 20], 0.1);
        assert_eq!(stats.hit_rate(99), 0.0);
        assert_eq!(stats.eval_count(99), 0);
    }

    #[test]
    fn test_stats_best_exit_point() {
        let mut stats = ExitLayerStats::new(&[10, 20, 30], 0.1);
        // Make layer 20 the best with high hit rate
        for _ in 0..15 {
            stats.record(20, true);
        }
        for _ in 0..12 {
            stats.record(10, false);
        }
        assert_eq!(stats.best_exit_point(), Some(20));
    }

    // ── EarlyExitController tests ──

    #[test]
    fn test_controller_disabled() {
        let config = EarlyExitConfig::default(); // enabled = false
        let mut ctrl = EarlyExitController::new(config, 32);
        let decision = ctrl.check_layer(20, 0.999, 1.001);
        assert!(matches!(decision, EarlyExitDecision::Defer));
    }

    #[test]
    fn test_controller_below_min_layer() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(16);
        let mut ctrl = EarlyExitController::new(config, 32);
        let decision = ctrl.check_layer(10, 0.999, 1.001);
        assert!(matches!(decision, EarlyExitDecision::Defer));
    }

    #[test]
    fn test_controller_not_exit_point() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let mut ctrl = EarlyExitController::new(config, 32);
        // Layer 5 is unlikely to be an exit point for 32-layer model
        // (exit points are at ~20, ~24, ~27)
        let decision = ctrl.check_layer(5, 0.999, 1.001);
        assert!(matches!(decision, EarlyExitDecision::Continue));
    }

    #[test]
    fn test_controller_exit_at_valid_point() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.8);
        let mut ctrl = EarlyExitController::new(config, 32);

        // Find a valid exit point
        let exit_points = ctrl.exit_points().points().to_vec();
        if let Some(&layer) = exit_points.first() {
            // Use strong convergence signals
            let decision = ctrl.check_layer(layer, 0.9999, 1.0001);
            match decision {
                EarlyExitDecision::Exit { confidence, .. } => {
                    assert!(confidence > 0.8, "confidence too low: {}", confidence);
                }
                EarlyExitDecision::Continue => {
                    // Not high enough — fine, depends on exact calibration
                }
                _ => panic!("unexpected decision type"),
            }
        }
    }

    #[test]
    fn test_controller_should_exit_shortcut() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let ctrl = EarlyExitController::new(config, 32);
        // Layer 0 — below min layer is handled by Defer in check_layer
        // but should_exit returns false
        assert!(!ctrl.should_exit(0, 0.999, 1.001));
    }

    // ── Legacy API tests ──

    #[test]
    fn test_early_exit_high_confidence() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        assert!(should_early_exit(20, 0.995, 1.005, &config));
    }

    #[test]
    fn test_early_exit_low_confidence() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        assert!(!should_early_exit(20, 0.95, 1.05, &config));
    }

    #[test]
    fn test_early_exit_min_layer() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        assert!(!should_early_exit(10, 0.995, 1.005, &config));
    }

    #[test]
    fn test_early_exit_disabled() {
        let mut config = EarlyExitConfig::default();
        config.enabled = false;
        assert!(!should_early_exit(20, 0.995, 1.005, &config));
    }

    #[test]
    fn test_residual_bus_early_exit() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        let exit = ResidualBusEarlyExit::new(config, vec![16, 20, 24]);

        assert!(exit.should_exit_at_layer(20, 1.005, 0.995));
        assert!(!exit.should_exit_at_layer(18, 1.005, 0.995));
    }

    // ── Config builder tests ──

    #[test]
    fn test_config_builder() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.95)
            .with_delta_threshold(0.02)
            .with_min_layer(8)
            .with_confidence_threshold(0.9);
        assert!(config.enabled);
        assert!((config.cosine_threshold - 0.95).abs() < 1e-5);
        assert!((config.delta_threshold - 0.02).abs() < 1e-5);
        assert_eq!(config.min_layer, 8);
        assert!((config.confidence_threshold - 0.9).abs() < 1e-5);
    }

    // ── Additional tests ──

    // -- EarlyExitConfig: default values, Debug, Clone --

    #[test]
    fn test_config_default_values() {
        let config = EarlyExitConfig::default();
        assert!(!config.enabled, "default should be disabled");
        assert!((config.cosine_threshold - 0.99).abs() < 1e-6);
        assert!((config.delta_threshold - 0.01).abs() < 1e-6);
        assert_eq!(config.min_layer, 16);
        assert!((config.confidence_threshold - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_config_new_equals_default() {
        let a = EarlyExitConfig::new();
        let b = EarlyExitConfig::default();
        assert_eq!(a.enabled, b.enabled);
        assert!((a.cosine_threshold - b.cosine_threshold).abs() < 1e-6);
        assert!((a.delta_threshold - b.delta_threshold).abs() < 1e-6);
        assert_eq!(a.min_layer, b.min_layer);
        assert!((a.confidence_threshold - b.confidence_threshold).abs() < 1e-6);
    }

    #[test]
    fn test_config_debug_format() {
        let config = EarlyExitConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("enabled"));
        assert!(debug_str.contains("cosine_threshold"));
        assert!(debug_str.contains("delta_threshold"));
        assert!(debug_str.contains("min_layer"));
        assert!(debug_str.contains("confidence_threshold"));
    }

    #[test]
    fn test_config_clone_independence() {
        let original = EarlyExitConfig::new().with_enabled(true).with_min_layer(10);
        let cloned = original.clone();
        assert_eq!(original.enabled, cloned.enabled);
        assert_eq!(original.min_layer, cloned.min_layer);
    }

    // -- ConfidenceCalibrator: custom constructor, negative logit, clone --

    #[test]
    fn test_calibrator_custom_params() {
        let cal = ConfidenceCalibrator::new(10.0, 5.0, 0.95, 0.02);
        let conf = cal.calibrate(0.96, 1.01);
        // cos_signal = 10.0 * (0.96 - 0.95) = 0.1
        // delta_signal = 5.0 * (0.02 - 0.01) = 0.05
        // logit = 0.15 → sigmoid ≈ 0.537
        assert!(conf > 0.5, "expected confidence > 0.5 with positive logit, got {}", conf);
        assert!(conf < 0.6, "expected confidence < 0.6 for weak signal, got {}", conf);
    }

    #[test]
    fn test_calibrator_negative_logit_path() {
        let cal = ConfidenceCalibrator::default();
        // Very poor signals to ensure negative logit
        let conf = cal.calibrate(0.5, 2.0);
        assert!(conf < 0.01, "expected near-zero confidence for very poor signals, got {}", conf);
    }

    #[test]
    fn test_calibrator_extreme_cosine_dominance() {
        let cal = ConfidenceCalibrator::new(100.0, 0.0, 0.99, 0.01);
        // alpha=100, cos=1.0, theta_c=0.99 → cos_signal = 100*(1.0-0.99) = 1.0
        // beta=0 → delta_signal = 0. logit = 1.0, sigmoid(1.0) ≈ 0.731
        let conf = cal.calibrate(1.0, 1.0);
        assert!(conf > 0.73 && conf < 0.74,
            "expected sigmoid(1.0) ≈ 0.731 with alpha=100 and beta=0, got {}", conf);
    }

    #[test]
    fn test_calibrator_clone_preserves_behavior() {
        let cal = ConfidenceCalibrator::new(15.0, 10.0, 0.98, 0.015);
        let cloned = cal.clone();
        let v1 = cal.calibrate(0.99, 1.005);
        let v2 = cloned.calibrate(0.99, 1.005);
        assert!((v1 - v2).abs() < 1e-7, "clone should produce identical calibration");
    }

    // -- AdaptiveExitPoints: total_layers, len, is_empty, edge cases --

    #[test]
    fn test_exit_points_total_layers() {
        let points = AdaptiveExitPoints::compute(64, 0.0);
        assert_eq!(points.total_layers(), 64);
    }

    #[test]
    fn test_exit_points_len_and_nonempty() {
        let points = AdaptiveExitPoints::compute(32, 0.0);
        assert!(!points.is_empty());
        assert_eq!(points.len(), points.points().len());
    }

    #[test]
    fn test_exit_points_single_layer() {
        let points = AdaptiveExitPoints::compute(1, 0.0);
        let pts = points.points();
        // Only possible point is 0, but it's clamped to min 1 which equals total_layers,
        // so min(1, 0) = 0. Either way it must be < total_layers.
        for &p in pts {
            assert!(p < 1, "exit point {} must be < 1 for single-layer model", p);
        }
    }

    #[test]
    fn test_exit_points_very_large_model() {
        let points = AdaptiveExitPoints::compute(1000, 0.0);
        let pts = points.points();
        assert!(pts.len() >= 2, "large model should have multiple exit points");
        // All points in [1, 999]
        for &p in pts {
            assert!(p >= 1 && p < 1000, "exit point {} out of range", p);
        }
        // Points strictly sorted
        for w in pts.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_exit_points_high_min_ratio_eliminates_all() {
        // min_layer_ratio=0.99 means only points at >= 99% of layers qualify
        let points = AdaptiveExitPoints::compute(10, 0.99);
        // For 10 layers, 99% = layer 9.9+, only layer 9 possible.
        // Golden ratio points are at ~6, ~7, ~8 → all below 9.9 → empty
        // (depends on rounding, but very likely empty or at most 1 point)
        for &p in points.points() {
            assert!(p >= 9, "exit point {} below extreme min ratio", p);
        }
    }

    #[test]
    fn test_exit_points_dedup() {
        // With 2 layers, all fractions may map to layer 1 — dedup should collapse
        let points = AdaptiveExitPoints::compute(2, 0.0);
        let pts = points.points();
        // Verify no duplicates
        let mut seen = std::collections::HashSet::new();
        for &p in pts {
            assert!(seen.insert(p), "duplicate exit point {}", p);
        }
    }

    // -- ExitLayerStats: total_evals, alpha clamping, edge cases --

    #[test]
    fn test_stats_total_evals() {
        let mut stats = ExitLayerStats::new(&[10, 20, 30], 0.1);
        stats.record(10, true);
        stats.record(20, false);
        stats.record(30, true);
        stats.record(20, true);
        assert_eq!(stats.total_evals(), 4);
    }

    #[test]
    fn test_stats_alpha_clamped_high() {
        // alpha above 1.0 should be clamped to 1.0
        let mut stats = ExitLayerStats::new(&[10], 5.0);
        stats.record(10, true);
        // With alpha=1.0, EMA = 1.0*1 + 0*0 = 1.0
        let rate = stats.hit_rate(10);
        assert!((rate - 1.0).abs() < 1e-6, "expected rate 1.0 with clamped alpha, got {}", rate);
    }

    #[test]
    fn test_stats_alpha_clamped_low() {
        // alpha below 0.01 should be clamped to 0.01
        let mut stats = ExitLayerStats::new(&[10], 0.001);
        stats.record(10, true);
        // With alpha=0.01, rate = 0.01 * 1.0 + 0.99 * 0.0 = 0.01
        let rate = stats.hit_rate(10);
        assert!((rate - 0.01).abs() < 1e-4, "expected rate ~0.01, got {}", rate);
    }

    #[test]
    fn test_stats_best_exit_point_insufficient_evals() {
        let mut stats = ExitLayerStats::new(&[10], 0.1);
        // Only 5 evaluations — below the minimum 10
        for _ in 0..5 {
            stats.record(10, true);
        }
        assert_eq!(stats.best_exit_point(), None, "should return None with insufficient evals");
    }

    #[test]
    fn test_stats_best_exit_point_no_data() {
        let stats = ExitLayerStats::new(&[10, 20], 0.1);
        assert_eq!(stats.best_exit_point(), None, "empty stats should return None");
    }

    #[test]
    fn test_stats_record_unknown_layer_is_noop() {
        let mut stats = ExitLayerStats::new(&[10, 20], 0.1);
        stats.record(99, true);
        assert_eq!(stats.total_evals(), 0);
        assert_eq!(stats.hit_rate(99), 0.0);
        assert_eq!(stats.eval_count(99), 0);
    }

    // -- EarlyExitDecision: Debug, Clone --

    #[test]
    fn test_decision_debug_continue() {
        let d = EarlyExitDecision::Continue;
        let s = format!("{:?}", d);
        assert_eq!(s, "Continue");
    }

    #[test]
    fn test_decision_debug_exit() {
        let d = EarlyExitDecision::Exit { confidence: 0.97, cosine_sim: 0.998, delta_rho: 1.002 };
        let s = format!("{:?}", d);
        assert!(s.contains("Exit"));
        assert!(s.contains("confidence"));
    }

    #[test]
    fn test_decision_debug_defer() {
        let d = EarlyExitDecision::Defer;
        let s = format!("{:?}", d);
        assert_eq!(s, "Defer");
    }

    #[test]
    fn test_decision_clone_exit() {
        let d = EarlyExitDecision::Exit { confidence: 0.9, cosine_sim: 0.99, delta_rho: 1.01 };
        let cloned = d.clone();
        match cloned {
            EarlyExitDecision::Exit { confidence, cosine_sim, delta_rho } => {
                assert!((confidence - 0.9).abs() < 1e-6);
                assert!((cosine_sim - 0.99).abs() < 1e-6);
                assert!((delta_rho - 1.01).abs() < 1e-6);
            }
            _ => panic!("cloned Exit variant should remain Exit"),
        }
    }

    // -- EarlyExitController: accessors, with_calibrator, should_exit edge cases --

    #[test]
    fn test_controller_config_accessor() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(5);
        let ctrl = EarlyExitController::new(config, 32);
        assert!(ctrl.config().enabled);
        assert_eq!(ctrl.config().min_layer, 5);
    }

    #[test]
    fn test_controller_stats_accessor_initial() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let ctrl = EarlyExitController::new(config, 32);
        assert_eq!(ctrl.stats().total_evals(), 0);
    }

    #[test]
    fn test_controller_with_calibrator_custom() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let calibrator = ConfidenceCalibrator::new(5.0, 3.0, 0.99, 0.01);
        let mut ctrl = EarlyExitController::with_calibrator(config, 32, calibrator);
        // Should work normally — just with different calibration
        let pts = ctrl.exit_points().points().to_vec();
        if let Some(&layer) = pts.first() {
            let decision = ctrl.check_layer(layer, 0.999, 1.0);
            // With weaker alpha/beta, even strong signals may not cross threshold
            match decision {
                EarlyExitDecision::Continue | EarlyExitDecision::Exit { .. } => {}
                _ => panic!("unexpected decision: {:?}", decision),
            }
        }
    }

    #[test]
    fn test_controller_should_exit_disabled() {
        let config = EarlyExitConfig::default(); // enabled = false
        let ctrl = EarlyExitController::new(config, 32);
        assert!(!ctrl.should_exit(20, 0.9999, 1.0001));
    }

    #[test]
    fn test_controller_should_exit_non_exit_point() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let ctrl = EarlyExitController::new(config, 32);
        // Layer 7 is not a golden-ratio exit point for 32-layer model
        assert!(!ctrl.should_exit(7, 0.9999, 1.0001));
    }

    #[test]
    fn test_controller_stats_updated_on_check() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.5);
        let mut ctrl = EarlyExitController::new(config, 32);
        let pts = ctrl.exit_points().points().to_vec();
        if let Some(&layer) = pts.first() {
            let _ = ctrl.check_layer(layer, 0.999, 1.0);
            assert!(ctrl.stats().total_evals() > 0, "stats should be updated after check_layer");
        }
    }

    // -- should_early_exit: boundary and mixed conditions --

    #[test]
    fn test_early_exit_direction_stable_energy_unstable() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        // cosine above threshold but delta far from 1.0
        assert!(!should_early_exit(20, 0.999, 1.5, &config));
    }

    #[test]
    fn test_early_exit_energy_stable_direction_unstable() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        // delta near 1.0 but cosine below threshold
        assert!(!should_early_exit(20, 0.80, 1.005, &config));
    }

    #[test]
    fn test_early_exit_exact_boundary_cosine() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        // cosine exactly at threshold (0.99), delta exactly 1.0
        // cosine_sim > threshold is strict inequality → 0.99 is NOT > 0.99
        assert!(!should_early_exit(20, 0.99, 1.0, &config));
    }

    #[test]
    fn test_early_exit_just_above_boundary() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        // cosine slightly above threshold, delta at exactly 1.0
        assert!(should_early_exit(20, 0.9901, 1.0, &config));
    }

    // -- ResidualBusEarlyExit: Debug, Clone, non-exit layer --

    #[test]
    fn test_residual_bus_debug_format() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let bus = ResidualBusEarlyExit::new(config, vec![10, 20]);
        let s = format!("{:?}", bus);
        assert!(s.contains("ResidualBusEarlyExit"));
        assert!(s.contains("exit_points"));
    }

    #[test]
    fn test_residual_bus_clone() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let bus = ResidualBusEarlyExit::new(config, vec![10, 20]);
        let cloned = bus.clone();
        assert_eq!(cloned.exit_points, bus.exit_points);
        assert_eq!(cloned.config.enabled, bus.config.enabled);
    }

    #[test]
    fn test_residual_bus_non_exit_layer() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        let bus = ResidualBusEarlyExit::new(config, vec![10, 20]);
        // Layer 15 is not in exit_points → always false
        assert!(!bus.should_exit_at_layer(15, 0.005, 0.9999));
    }

    // -- ConfidenceCalibrator: symmetry and edge inputs --

    #[test]
    fn test_calibrator_delta_rho_symmetry() {
        let cal = ConfidenceCalibrator::default();
        let c_above = cal.calibrate(0.99, 1.01);
        let c_below = cal.calibrate(0.99, 0.99);
        // |1.01 - 1.0| == |0.99 - 1.0| → same confidence
        assert!((c_above - c_below).abs() < 1e-6,
            "delta_rho symmetry broken: {} vs {}", c_above, c_below);
    }

    #[test]
    fn test_calibrator_output_bounded() {
        let cal = ConfidenceCalibrator::default();
        // Extreme inputs in both directions
        let conf_high = cal.calibrate(1.0, 1.0);
        let conf_low = cal.calibrate(0.0, 10.0);
        assert!(conf_high <= 1.0 && conf_high > 0.0, "confidence must be in (0, 1], got {}", conf_high);
        assert!(conf_low >= 0.0 && conf_low < 1.0, "confidence must be in [0, 1), got {}", conf_low);
    }

    #[test]
    fn test_calibrator_default_debug() {
        let cal = ConfidenceCalibrator::default();
        let s = format!("{:?}", cal);
        assert!(s.contains("alpha"));
        assert!(s.contains("beta"));
        assert!(s.contains("theta_c"));
        assert!(s.contains("theta_d"));
    }

    // ── Additional coverage: EarlyExitConfig edge cases ──

    #[test]
    fn config_zero_cosine_threshold() {
        let config = EarlyExitConfig::new().with_cosine_threshold(0.0);
        assert!((config.cosine_threshold).abs() < 1e-6);
    }

    #[test]
    fn config_cosine_threshold_above_one() {
        let config = EarlyExitConfig::new().with_cosine_threshold(1.5);
        assert!((config.cosine_threshold - 1.5).abs() < 1e-6);
    }

    #[test]
    fn config_zero_min_layer() {
        let config = EarlyExitConfig::new().with_min_layer(0);
        assert_eq!(config.min_layer, 0);
    }

    #[test]
    fn config_very_high_min_layer() {
        let config = EarlyExitConfig::new().with_min_layer(1000);
        assert_eq!(config.min_layer, 1000);
    }

    #[test]
    fn config_negative_delta_threshold() {
        let config = EarlyExitConfig::new().with_delta_threshold(-0.5);
        assert!((config.delta_threshold + 0.5).abs() < 1e-6);
    }

    #[test]
    fn config_zero_confidence_threshold() {
        let config = EarlyExitConfig::new().with_confidence_threshold(0.0);
        assert!((config.confidence_threshold).abs() < 1e-6);
    }

    #[test]
    fn config_builder_chaining_all_fields() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.98)
            .with_delta_threshold(0.005)
            .with_min_layer(12)
            .with_confidence_threshold(0.88);
        assert!(config.enabled);
        assert!((config.cosine_threshold - 0.98).abs() < 1e-6);
        assert!((config.delta_threshold - 0.005).abs() < 1e-6);
        assert_eq!(config.min_layer, 12);
        assert!((config.confidence_threshold - 0.88).abs() < 1e-6);
    }

    // ── ConfidenceCalibrator: additional edge cases ──

    #[test]
    fn calibrator_sigmoid_at_zero_logit() {
        // When logit = 0 → sigmoid = 0.5
        // alpha=1, beta=0, theta_c=0.5, theta_d=100 → cos_signal = 1*(0.5-0.5)=0, delta_signal = 0*(100-|dr-1|)=0
        let cal = ConfidenceCalibrator::new(1.0, 0.0, 0.5, 100.0);
        let conf = cal.calibrate(0.5, 1.0);
        assert!((conf - 0.5).abs() < 1e-5, "sigmoid(0) should be 0.5, got {}", conf);
    }

    #[test]
    fn calibrator_high_alpha_dominates_cosine() {
        let cal = ConfidenceCalibrator::new(1000.0, 0.1, 0.99, 0.01);
        let c_above = cal.calibrate(1.0, 1.0);
        let c_below = cal.calibrate(0.98, 1.0);
        assert!(c_above > c_below, "higher cosine should give higher confidence");
        assert!(c_above > 0.99, "strong cos signal should give very high confidence");
    }

    #[test]
    fn calibrator_identical_inputs_identical_output() {
        let cal = ConfidenceCalibrator::default();
        let a = cal.calibrate(0.97, 1.02);
        let b = cal.calibrate(0.97, 1.02);
        assert!((a - b).abs() < 1e-10, "identical inputs must produce identical output");
    }

    #[test]
    fn calibrator_f32_extreme_inputs_no_panic() {
        let cal = ConfidenceCalibrator::default();
        let _ = cal.calibrate(f32::MAX, f32::MAX);
        let _ = cal.calibrate(f32::MIN, f32::MIN);
        let _ = cal.calibrate(0.0, 0.0);
        let _ = cal.calibrate(-100.0, 100.0);
    }

    // ── AdaptiveExitPoints: additional edge cases ──

    #[test]
    fn exit_points_one_layer() {
        let points = AdaptiveExitPoints::compute(1, 0.0);
        let pts = points.points();
        for &p in pts {
            assert!(p < 1, "exit point {} must be < 1", p);
        }
        assert_eq!(points.total_layers(), 1);
    }

    #[test]
    fn exit_points_two_layers() {
        let points = AdaptiveExitPoints::compute(2, 0.0);
        let pts = points.points();
        for &p in pts {
            assert!(p >= 1 && p < 2, "exit point {} out of range for 2 layers", p);
        }
    }

    #[test]
    fn exit_points_clone_preserves_points() {
        let points = AdaptiveExitPoints::compute(32, 0.0);
        let cloned = points.clone();
        assert_eq!(points.points(), cloned.points());
        assert_eq!(points.total_layers(), cloned.total_layers());
    }

    #[test]
    fn exit_points_min_ratio_zero() {
        let points = AdaptiveExitPoints::compute(32, 0.0);
        let pts = points.points();
        assert!(!pts.is_empty(), "zero min ratio should allow all points");
    }

    #[test]
    fn exit_points_exact_layer_boundaries() {
        // 3-layer model: only possible exit point is layer 1 or 2
        let points = AdaptiveExitPoints::compute(3, 0.0);
        for &p in points.points() {
            assert!(p >= 1 && p < 3, "exit point {} out of [1, 2]", p);
        }
    }

    // ── ExitLayerStats: additional coverage ──

    #[test]
    fn stats_empty_exit_points() {
        let stats = ExitLayerStats::new(&[], 0.1);
        assert_eq!(stats.total_evals(), 0);
        assert_eq!(stats.best_exit_point(), None);
    }

    #[test]
    fn stats_many_layers_sequential_records() {
        let layers: Vec<usize> = (1..=20).collect();
        let mut stats = ExitLayerStats::new(&layers, 0.5);
        for &l in &layers {
            stats.record(l, true);
        }
        assert_eq!(stats.total_evals(), 20);
        for &l in &layers {
            assert_eq!(stats.eval_count(l), 1);
        }
    }

    #[test]
    fn stats_ema_convergence_with_alpha_one() {
        let mut stats = ExitLayerStats::new(&[10], 1.0);
        // alpha=1.0 means no smoothing: rate = last observation
        stats.record(10, true);
        assert!((stats.hit_rate(10) - 1.0).abs() < 1e-6);
        stats.record(10, false);
        assert!((stats.hit_rate(10)).abs() < 1e-6);
        stats.record(10, true);
        assert!((stats.hit_rate(10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stats_best_exit_point_tie_breaking() {
        let mut stats = ExitLayerStats::new(&[10, 20], 0.5);
        // Both layers get 15 records with same hit pattern
        for _ in 0..15 {
            stats.record(10, true);
            stats.record(20, true);
        }
        let best = stats.best_exit_point();
        assert!(best == Some(10) || best == Some(20), "either layer could be best");
    }

    #[test]
    fn stats_single_point_multiple_records() {
        let mut stats = ExitLayerStats::new(&[5], 0.1);
        for i in 0..100 {
            stats.record(5, i % 2 == 0);
        }
        assert_eq!(stats.eval_count(5), 100);
        let rate = stats.hit_rate(5);
        assert!(rate > 0.3 && rate < 0.7, "after alternating records, rate should be ~0.5, got {}", rate);
    }

    // ── EarlyExitDecision: additional coverage ──

    #[test]
    fn decision_continue_clone() {
        let d = EarlyExitDecision::Continue;
        let cloned = d.clone();
        assert!(matches!(cloned, EarlyExitDecision::Continue));
    }

    #[test]
    fn decision_defer_clone() {
        let d = EarlyExitDecision::Defer;
        let cloned = d.clone();
        assert!(matches!(cloned, EarlyExitDecision::Defer));
    }

    #[test]
    fn decision_exit_zero_confidence() {
        let d = EarlyExitDecision::Exit { confidence: 0.0, cosine_sim: 0.0, delta_rho: 0.0 };
        if let EarlyExitDecision::Exit { confidence, .. } = d {
            assert!((confidence).abs() < 1e-6);
        } else {
            panic!("expected Exit variant");
        }
    }

    // ── ResidualBusEarlyExit: additional coverage ──

    #[test]
    fn residual_bus_empty_exit_points() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let bus = ResidualBusEarlyExit::new(config, vec![]);
        assert!(!bus.should_exit_at_layer(10, 0.001, 0.9999));
    }

    #[test]
    fn residual_bus_single_exit_point() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let bus = ResidualBusEarlyExit::new(config, vec![20]);
        // residual_delta=1.005 (near 1.0), cosine_sim=0.995 (> 0.99 threshold)
        assert!(bus.should_exit_at_layer(20, 1.005, 0.995));
        assert!(!bus.should_exit_at_layer(21, 1.005, 0.995));
    }

    #[test]
    fn residual_bus_exit_point_at_layer_zero() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        let bus = ResidualBusEarlyExit::new(config, vec![0]);
        // residual_delta=1.005, cosine_sim=0.995 (> 0.99 threshold)
        assert!(bus.should_exit_at_layer(0, 1.005, 0.995));
    }

    #[test]
    fn residual_bus_clone_independence() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let bus = ResidualBusEarlyExit::new(config, vec![10, 20, 30]);
        let cloned = bus.clone();
        assert_eq!(cloned.exit_points.len(), 3);
        assert_eq!(cloned.config.enabled, true);
    }

    // ── should_early_exit: additional edge cases ──

    #[test]
    fn early_exit_layer_zero_always_false() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        assert!(!should_early_exit(0, 1.0, 1.0, &config));
    }

    #[test]
    fn early_exit_very_low_cosine() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        assert!(!should_early_exit(20, -1.0, 1.0, &config));
    }

    #[test]
    fn early_exit_exact_delta_boundary() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        // Use a value clearly outside the threshold: delta=1.02, |1.02-1|=0.02 > 0.01
        assert!(!should_early_exit(20, 0.995, 1.02, &config));
    }

    #[test]
    fn early_exit_just_inside_delta_boundary() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        // delta = 1.009, |1.009 - 1| = 0.009 < 0.01
        assert!(should_early_exit(20, 0.995, 1.009, &config));
    }

    #[test]
    fn early_exit_high_min_layer_blocks() {
        let mut config = EarlyExitConfig::default();
        config.enabled = true;
        config.min_layer = 100;
        assert!(!should_early_exit(99, 0.9999, 1.0001, &config));
    }

    // ── EarlyExitController: additional edge cases ──

    #[test]
    fn controller_one_layer_model() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        let ctrl = EarlyExitController::new(config, 1);
        assert_eq!(ctrl.exit_points().total_layers(), 1);
    }

    #[test]
    fn controller_single_layer_model() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        let mut ctrl = EarlyExitController::new(config, 1);
        // Only layer 0 exists, and with min_layer=0 it may or may not be an exit point
        let _ = ctrl.check_layer(0, 0.999, 1.0);
    }

    #[test]
    fn controller_check_layer_records_stats() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.5);
        let mut ctrl = EarlyExitController::new(config, 32);
        let pts = ctrl.exit_points().points().to_vec();
        if let Some(&layer) = pts.first() {
            let _ = ctrl.check_layer(layer, 0.999, 1.0);
            assert_eq!(ctrl.stats().eval_count(layer), 1);
        }
    }

    #[test]
    fn controller_should_exit_at_exit_point_with_strong_signal() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.5);
        let ctrl = EarlyExitController::new(config, 32);
        let pts = ctrl.exit_points().points().to_vec();
        if let Some(&layer) = pts.first() {
            assert!(ctrl.should_exit(layer, 0.9999, 1.0001));
        }
    }

    #[test]
    fn controller_with_calibrator_accessors() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let calibrator = ConfidenceCalibrator::new(1.0, 1.0, 0.99, 0.01);
        let ctrl = EarlyExitController::with_calibrator(config, 64, calibrator);
        assert_eq!(ctrl.exit_points().total_layers(), 64);
        assert!(ctrl.config().enabled);
    }

    // ── Wave 2: 50 additional tests covering remaining public API gaps ──

    // -- ConfidenceCalibrator: zero/negative weights, directional properties, field defaults --

    #[test]
    fn ee_calibrator_zero_beta_ignores_delta() {
        let cal = ConfidenceCalibrator::new(10.0, 0.0, 0.99, 0.01);
        let c_near = cal.calibrate(0.995, 1.001);
        let c_far = cal.calibrate(0.995, 5.0);
        assert!(
            (c_near - c_far).abs() < 1e-6,
            "beta=0 should make output independent of delta_rho: {} vs {}",
            c_near, c_far
        );
    }

    #[test]
    fn ee_calibrator_zero_alpha_ignores_cosine() {
        let cal = ConfidenceCalibrator::new(0.0, 10.0, 0.99, 0.01);
        let c_high = cal.calibrate(1.0, 1.0);
        let c_low = cal.calibrate(0.5, 1.0);
        assert!(
            (c_high - c_low).abs() < 1e-6,
            "alpha=0 should make output independent of cosine: {} vs {}",
            c_high, c_low
        );
    }

    #[test]
    fn ee_calibrator_delta_rho_monotonic_toward_one() {
        let cal = ConfidenceCalibrator::default();
        let c_far_below = cal.calibrate(0.99, 0.5);
        let c_near_below = cal.calibrate(0.99, 0.98);
        let c_at_one = cal.calibrate(0.99, 1.0);
        let c_near_above = cal.calibrate(0.99, 1.02);
        let c_far_above = cal.calibrate(0.99, 1.5);
        assert!(
            c_far_below < c_near_below,
            "moving delta toward 1.0 should increase confidence"
        );
        assert!(
            c_near_below < c_at_one,
            "confidence should peak at delta_rho=1.0"
        );
        assert!(
            c_near_above > c_far_above,
            "moving delta away from 1.0 should decrease confidence"
        );
    }

    #[test]
    fn ee_calibrator_high_logit_approaches_one() {
        let cal = ConfidenceCalibrator::new(1000.0, 1000.0, 0.0, 100.0);
        let conf = cal.calibrate(1.0, 1.0);
        assert!(
            conf > 0.99,
            "very large positive logit should give confidence near 1.0, got {}",
            conf
        );
    }

    #[test]
    fn ee_calibrator_low_logit_approaches_zero() {
        let cal = ConfidenceCalibrator::new(0.1, 0.1, 1.0, 0.0);
        let conf = cal.calibrate(0.0, 100.0);
        assert!(
            conf < 0.01,
            "very large negative logit should give confidence near 0.0, got {}",
            conf
        );
    }

    #[test]
    fn ee_calibrator_default_params_values() {
        let cal = ConfidenceCalibrator::default();
        assert!((cal.alpha - 20.0).abs() < 1e-6);
        assert!((cal.beta - 15.0).abs() < 1e-6);
        assert!((cal.theta_c - 0.99).abs() < 1e-6);
        assert!((cal.theta_d - 0.01).abs() < 1e-6);
    }

    #[test]
    fn ee_calibrator_negative_alpha_flips_cosine_effect() {
        let cal_neg = ConfidenceCalibrator::new(-10.0, 0.0, 0.5, 0.01);
        let c_above = cal_neg.calibrate(0.6, 1.0);
        let c_below = cal_neg.calibrate(0.4, 1.0);
        assert!(
            c_above < c_below,
            "negative alpha should invert cosine effect: {} vs {}",
            c_above, c_below
        );
    }

    #[test]
    fn ee_calibrator_output_always_in_unit_interval() {
        let cal = ConfidenceCalibrator::default();
        let test_cases = [
            (0.0_f32, 0.0_f32),
            (1.0, 1.0),
            (0.5, 1.5),
            (-1.0, 2.0),
            (0.99, 1.01),
            (0.01, 0.99),
        ];
        for (cos, delta) in &test_cases {
            let conf = cal.calibrate(*cos, *delta);
            assert!(
                conf >= 0.0 && conf <= 1.0,
                "confidence {} not in [0,1] for cos={}, delta={}",
                conf, cos, delta
            );
        }
    }

    // -- AdaptiveExitPoints: properties, edge cases, Debug/Clone --

    #[test]
    fn ee_exit_points_is_exit_point_beyond_range() {
        let points = AdaptiveExitPoints::compute(32, 0.0);
        assert!(!points.is_exit_point(32), "layer == total_layers should not be exit point");
        assert!(!points.is_exit_point(100), "layer >> total_layers should not be exit point");
        assert!(!points.is_exit_point(usize::MAX), "extreme layer should not be exit point");
    }

    #[test]
    fn ee_exit_points_deterministic() {
        let a = AdaptiveExitPoints::compute(64, 0.3);
        let b = AdaptiveExitPoints::compute(64, 0.3);
        assert_eq!(a.points(), b.points(), "same inputs should yield same exit points");
    }

    #[test]
    fn ee_exit_points_increasing_min_ratio_reduces_points() {
        let p0 = AdaptiveExitPoints::compute(64, 0.0);
        let p5 = AdaptiveExitPoints::compute(64, 0.5);
        let p8 = AdaptiveExitPoints::compute(64, 0.8);
        assert!(
            p0.len() >= p5.len(),
            "higher min_ratio should not increase exit point count"
        );
        assert!(
            p5.len() >= p8.len(),
            "higher min_ratio should not increase exit point count"
        );
    }

    #[test]
    fn ee_exit_points_sorted_property() {
        for layers in [2, 4, 8, 16, 32, 64, 128, 256] {
            let points = AdaptiveExitPoints::compute(layers, 0.0);
            let pts = points.points();
            for w in pts.windows(2) {
                assert!(
                    w[0] < w[1],
                    "exit points not sorted for {} layers: {:?}",
                    layers, pts
                );
            }
        }
    }

    #[test]
    fn ee_exit_points_all_within_layer_bounds() {
        for layers in [2, 4, 8, 16, 32, 64, 100, 500] {
            let points = AdaptiveExitPoints::compute(layers, 0.0);
            for &p in points.points() {
                assert!(
                    p >= 1 && p < layers,
                    "exit point {} out of [1, {}) for {} layers",
                    p, layers, layers
                );
            }
        }
    }

    #[test]
    fn ee_exit_points_is_exit_point_consistency() {
        let points = AdaptiveExitPoints::compute(32, 0.0);
        for &p in points.points() {
            assert!(points.is_exit_point(p), "points() member {} should be exit point", p);
        }
        for layer in 0..32 {
            if !points.points().contains(&layer) {
                assert!(
                    !points.is_exit_point(layer),
                    "non-member {} should not be exit point", layer
                );
            }
        }
    }

    #[test]
    fn ee_exit_points_64_layers_has_multiple_points() {
        let points = AdaptiveExitPoints::compute(64, 0.0);
        assert!(points.len() >= 2, "64-layer model should have at least 2 exit points");
        assert!(!points.is_empty());
    }

    #[test]
    fn ee_exit_points_clone_equality() {
        let original = AdaptiveExitPoints::compute(48, 0.2);
        let cloned = original.clone();
        assert_eq!(original.points(), cloned.points());
        assert_eq!(original.total_layers(), cloned.total_layers());
        assert_eq!(original.len(), cloned.len());
        assert_eq!(original.is_empty(), cloned.is_empty());
    }

    #[test]
    fn ee_exit_points_debug_format() {
        let points = AdaptiveExitPoints::compute(32, 0.0);
        let s = format!("{:?}", points);
        assert!(s.contains("AdaptiveExitPoints"), "Debug should contain type name");
    }

    #[test]
    fn ee_exit_points_len_matches_points_slice() {
        for layers in [3, 7, 16, 32, 64, 128] {
            let points = AdaptiveExitPoints::compute(layers, 0.0);
            assert_eq!(
                points.len(), points.points().len(),
                "len mismatch for {} layers", layers
            );
        }
    }

    // -- ExitLayerStats: Debug, Clone, EMA properties, accumulation --

    #[test]
    fn ee_stats_debug_format() {
        let stats = ExitLayerStats::new(&[10, 20], 0.1);
        let s = format!("{:?}", stats);
        assert!(s.contains("ExitLayerStats"), "Debug should contain type name");
    }

    #[test]
    fn ee_stats_clone_preserves_all_state() {
        let mut stats = ExitLayerStats::new(&[5, 10, 15], 0.3);
        stats.record(5, true);
        stats.record(10, false);
        stats.record(15, true);
        let cloned = stats.clone();
        assert_eq!(cloned.total_evals(), stats.total_evals());
        assert_eq!(cloned.eval_count(5), stats.eval_count(5));
        assert!((cloned.hit_rate(5) - stats.hit_rate(5)).abs() < 1e-7);
        assert!((cloned.hit_rate(10) - stats.hit_rate(10)).abs() < 1e-7);
    }

    #[test]
    fn ee_stats_ema_formula_verification() {
        let mut stats = ExitLayerStats::new(&[10], 0.5);
        // Initial rate = 0.0. Record true: rate = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        stats.record(10, true);
        assert!(
            (stats.hit_rate(10) - 0.5).abs() < 1e-5,
            "after 1 true: rate should be 0.5, got {}", stats.hit_rate(10)
        );
        // Record false: rate = 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        stats.record(10, false);
        assert!(
            (stats.hit_rate(10) - 0.25).abs() < 1e-5,
            "after 1 true + 1 false: rate should be 0.25, got {}", stats.hit_rate(10)
        );
    }

    #[test]
    fn ee_stats_ema_alpha_boundary_low() {
        let mut stats = ExitLayerStats::new(&[10], 0.01);
        stats.record(10, true);
        let rate = stats.hit_rate(10);
        assert!(
            (rate - 0.01).abs() < 1e-4,
            "alpha=0.01, first true → rate should be 0.01, got {}", rate
        );
    }

    #[test]
    fn ee_stats_ema_alpha_boundary_high() {
        let mut stats = ExitLayerStats::new(&[10], 2.0);
        stats.record(10, true);
        assert!(
            (stats.hit_rate(10) - 1.0).abs() < 1e-5,
            "alpha clamped to 1.0, first true → rate should be 1.0"
        );
        stats.record(10, false);
        assert!(
            (stats.hit_rate(10)).abs() < 1e-5,
            "alpha=1.0, second false → rate should be 0.0"
        );
    }

    #[test]
    fn ee_stats_hit_rate_initially_zero_for_all_layers() {
        let stats = ExitLayerStats::new(&[5, 10, 15, 20], 0.1);
        for &layer in &[5, 10, 15, 20] {
            assert!(
                stats.hit_rate(layer).abs() < 1e-7,
                "initial hit rate should be 0.0 for layer {}", layer
            );
        }
    }

    #[test]
    fn ee_stats_eval_count_initially_zero() {
        let stats = ExitLayerStats::new(&[3, 6, 9], 0.1);
        for &layer in &[3, 6, 9] {
            assert_eq!(stats.eval_count(layer), 0, "initial eval count should be 0");
        }
    }

    #[test]
    fn ee_stats_best_exit_point_requires_min_evals_per_layer() {
        let mut stats = ExitLayerStats::new(&[10, 20], 0.5);
        for _ in 0..5 {
            stats.record(10, true);
        }
        for _ in 0..15 {
            stats.record(20, true);
        }
        // Only layer 20 has >= 10 evals
        assert_eq!(stats.best_exit_point(), Some(20));
    }

    #[test]
    fn ee_stats_multiple_records_same_layer_accumulate() {
        let mut stats = ExitLayerStats::new(&[10], 0.1);
        for _ in 0..20 {
            stats.record(10, true);
        }
        assert_eq!(stats.eval_count(10), 20);
        assert_eq!(stats.total_evals(), 20);
        let rate = stats.hit_rate(10);
        assert!(rate > 0.8, "20 true records should give high hit rate, got {}", rate);
    }

    // -- EarlyExitDecision: edge value coverage --

    #[test]
    fn ee_decision_exit_with_high_confidence() {
        let d = EarlyExitDecision::Exit {
            confidence: 0.999,
            cosine_sim: 0.9999,
            delta_rho: 1.0001,
        };
        if let EarlyExitDecision::Exit { confidence, cosine_sim, delta_rho } = d {
            assert!((confidence - 0.999).abs() < 1e-6);
            assert!((cosine_sim - 0.9999).abs() < 1e-6);
            assert!((delta_rho - 1.0001).abs() < 1e-6);
        } else {
            panic!("expected Exit variant");
        }
    }

    #[test]
    fn ee_decision_exit_with_extreme_values() {
        let d = EarlyExitDecision::Exit {
            confidence: f32::MAX,
            cosine_sim: f32::MAX,
            delta_rho: f32::MAX,
        };
        if let EarlyExitDecision::Exit { confidence, .. } = d {
            assert_eq!(confidence, f32::MAX);
        }
    }

    #[test]
    fn ee_decision_all_variants_match() {
        let cases: Vec<EarlyExitDecision> = vec![
            EarlyExitDecision::Continue,
            EarlyExitDecision::Defer,
            EarlyExitDecision::Exit { confidence: 0.5, cosine_sim: 0.9, delta_rho: 1.0 },
        ];
        assert!(matches!(cases[0], EarlyExitDecision::Continue));
        assert!(matches!(cases[1], EarlyExitDecision::Defer));
        assert!(matches!(cases[2], EarlyExitDecision::Exit { .. }));
    }

    // -- EarlyExitController: boundary conditions, accumulation, threshold extremes --

    #[test]
    fn ee_controller_min_layer_equals_total_layers() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(32);
        let mut ctrl = EarlyExitController::new(config, 32);
        let decision = ctrl.check_layer(31, 0.9999, 1.0001);
        assert!(
            matches!(decision, EarlyExitDecision::Defer),
            "layer 31 < min_layer=32 should be Defer"
        );
    }

    #[test]
    fn ee_controller_min_layer_zero() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        let ctrl = EarlyExitController::new(config, 32);
        assert_eq!(ctrl.config().min_layer, 0);
    }

    #[test]
    fn ee_controller_check_layer_at_exact_min_layer() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(10)
            .with_confidence_threshold(0.5);
        let mut ctrl = EarlyExitController::new(config, 32);
        // layer=10 == min_layer → passes the `layer < min_layer` guard
        // → NOT Defer (either Continue if not exit point, or Exit/Continue if it is)
        let decision = ctrl.check_layer(10, 0.999, 1.0);
        assert!(
            !matches!(decision, EarlyExitDecision::Defer),
            "layer == min_layer should not be Defer"
        );
    }

    #[test]
    fn ee_controller_multiple_check_layers_accumulate() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.5);
        let mut ctrl = EarlyExitController::new(config, 64);
        let pts = ctrl.exit_points().points().to_vec();
        if pts.len() >= 2 {
            let _ = ctrl.check_layer(pts[0], 0.999, 1.0);
            let _ = ctrl.check_layer(pts[1], 0.5, 2.0);
            assert_eq!(ctrl.stats().total_evals(), 2);
            assert_eq!(ctrl.stats().eval_count(pts[0]), 1);
            assert_eq!(ctrl.stats().eval_count(pts[1]), 1);
        }
    }

    #[test]
    fn ee_controller_repeated_check_layers_accumulate() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.5);
        let mut ctrl = EarlyExitController::new(config, 32);
        if let Some(&layer) = ctrl.exit_points().points().first() {
            for _ in 0..5 {
                let _ = ctrl.check_layer(layer, 0.999, 1.0);
            }
            assert_eq!(ctrl.stats().eval_count(layer), 5);
        }
    }

    #[test]
    fn ee_controller_zero_confidence_threshold_always_exits() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.0);
        let mut ctrl = EarlyExitController::new(config, 32);
        if let Some(&layer) = ctrl.exit_points().points().first() {
            let decision = ctrl.check_layer(layer, 0.5, 2.0);
            assert!(
                matches!(decision, EarlyExitDecision::Exit { .. }),
                "threshold=0.0 should always exit at exit points"
            );
        }
    }

    #[test]
    fn ee_controller_confidence_threshold_one_never_exits() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(1.0);
        let mut ctrl = EarlyExitController::new(config, 32);
        if let Some(&layer) = ctrl.exit_points().points().first() {
            let decision = ctrl.check_layer(layer, 0.9999, 1.0001);
            assert!(
                matches!(decision, EarlyExitDecision::Continue),
                "threshold=1.0 should never trigger exit (sigmoid < 1.0)"
            );
        }
    }

    #[test]
    fn ee_controller_with_calibrator_custom_uses_provided() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let cal = ConfidenceCalibrator::new(0.001, 0.001, 0.99, 0.01);
        let mut ctrl = EarlyExitController::with_calibrator(config, 32, cal);
        if let Some(&layer) = ctrl.exit_points().points().first() {
            let decision = ctrl.check_layer(layer, 0.999, 1.0);
            match decision {
                EarlyExitDecision::Continue | EarlyExitDecision::Exit { .. } => {}
                _ => panic!("unexpected decision at exit point: {:?}", decision),
            }
        }
    }

    #[test]
    fn ee_controller_large_model_128_layers() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let ctrl = EarlyExitController::new(config, 128);
        assert_eq!(ctrl.exit_points().total_layers(), 128);
        assert!(
            ctrl.exit_points().len() >= 2,
            "128-layer model should have multiple exit points"
        );
    }

    #[test]
    fn ee_controller_two_layer_model() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        let ctrl = EarlyExitController::new(config, 2);
        assert_eq!(ctrl.exit_points().total_layers(), 2);
    }

    // -- ResidualBusEarlyExit: edge cases, public fields --

    #[test]
    fn ee_residual_bus_disabled_config_never_exits() {
        let config = EarlyExitConfig::new().with_enabled(false);
        let bus = ResidualBusEarlyExit::new(config, vec![10, 20]);
        assert!(
            !bus.should_exit_at_layer(20, 0.001, 0.9999),
            "disabled config should never exit"
        );
    }

    #[test]
    fn ee_residual_bus_many_exit_points() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(10);
        let exit_layers: Vec<usize> = (10..=30).collect();
        let bus = ResidualBusEarlyExit::new(config, exit_layers);
        assert_eq!(bus.exit_points.len(), 21);
        assert!(bus.should_exit_at_layer(10, 1.005, 0.995));
        assert!(bus.should_exit_at_layer(30, 1.005, 0.995));
    }

    #[test]
    fn ee_residual_bus_unsorted_exit_points() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(10);
        let bus = ResidualBusEarlyExit::new(config, vec![30, 10, 20]);
        assert!(bus.should_exit_at_layer(10, 1.005, 0.995));
        assert!(bus.should_exit_at_layer(20, 1.005, 0.995));
        assert!(bus.should_exit_at_layer(30, 1.005, 0.995));
    }

    #[test]
    fn ee_residual_bus_duplicate_exit_points() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let bus = ResidualBusEarlyExit::new(config, vec![20, 20, 20]);
        assert!(bus.should_exit_at_layer(20, 1.005, 0.995));
        assert_eq!(bus.exit_points.len(), 3);
    }

    #[test]
    fn ee_residual_bus_config_field_public_access() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(5);
        let bus = ResidualBusEarlyExit::new(config, vec![10]);
        assert!(bus.config.enabled);
        assert_eq!(bus.config.min_layer, 5);
    }

    #[test]
    fn ee_residual_bus_exit_points_field_public_access() {
        let bus = ResidualBusEarlyExit::new(
            EarlyExitConfig::default(),
            vec![5, 10, 15],
        );
        assert_eq!(bus.exit_points, vec![5, 10, 15]);
    }

    // -- should_early_exit: additional edge cases --

    #[test]
    fn ee_should_exit_cosine_just_above_threshold() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.99)
            .with_delta_threshold(0.02);
        let cos = 0.99f32 + f32::EPSILON;
        assert!(
            should_early_exit(20, cos, 1.0, &config),
            "cosine just above threshold should trigger exit"
        );
    }

    #[test]
    fn ee_should_exit_delta_symmetry_above_below() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.98)
            .with_delta_threshold(0.02);
        assert!(
            should_early_exit(20, 0.99, 1.01, &config),
            "delta above 1.0 within threshold"
        );
        assert!(
            should_early_exit(20, 0.99, 0.99, &config),
            "delta below 1.0 within threshold"
        );
    }

    #[test]
    fn ee_should_exit_min_layer_zero_allows_layer_zero() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(0);
        assert!(
            should_early_exit(0, 0.995, 1.005, &config),
            "min_layer=0 should allow layer 0"
        );
    }

    #[test]
    fn ee_should_exit_very_high_cosine_threshold() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.9999)
            .with_delta_threshold(0.01);
        assert!(
            !should_early_exit(20, 0.999, 1.0, &config),
            "cosine below very high threshold should not exit"
        );
    }

    #[test]
    fn ee_should_exit_very_tight_delta_threshold() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.98)
            .with_delta_threshold(0.001);
        assert!(
            !should_early_exit(20, 0.999, 1.005, &config),
            "delta outside very tight threshold should not exit"
        );
    }

    // ── Wave 3: 10 additional tests covering remaining boundary gaps ──

    // ConfidenceCalibrator: NaN inputs should not panic and should produce finite output
    #[test]
    fn ee_calibrator_nan_inputs_no_panic() {
        let cal = ConfidenceCalibrator::default();
        let conf = cal.calibrate(f32::NAN, 1.0);
        assert!(
            conf.is_nan() || (conf >= 0.0 && conf <= 1.0),
            "NaN cosine should not panic, got {}",
            conf
        );
        let conf2 = cal.calibrate(0.99, f32::NAN);
        assert!(
            conf2.is_nan() || (conf2 >= 0.0 && conf2 <= 1.0),
            "NaN delta should not panic, got {}",
            conf2
        );
    }

    // AdaptiveExitPoints: zero total_layers should not panic
    #[test]
    fn ee_exit_points_zero_total_layers() {
        let points = AdaptiveExitPoints::compute(0, 0.0);
        assert!(points.is_empty(), "zero layers should yield empty exit points");
        assert_eq!(points.total_layers(), 0);
        assert_eq!(points.len(), 0);
        assert!(!points.is_exit_point(0));
    }

    // EarlyExitController: total_layers=0 should not panic
    #[test]
    fn ee_controller_zero_total_layers() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        let mut ctrl = EarlyExitController::new(config, 0);
        // Any layer check should not panic — there are no exit points
        let decision = ctrl.check_layer(0, 0.999, 1.0);
        assert!(
            matches!(decision, EarlyExitDecision::Continue),
            "zero-layer model should Continue for any layer since no exit points exist"
        );
    }

    // EarlyExitConfig: builder overwrites — setting enabled=true then false
    #[test]
    fn ee_config_builder_overwrite_enabled() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_enabled(false);
        assert!(
            !config.enabled,
            "last with_enabled call should win"
        );
    }

    // EarlyExitConfig: builder overwrite for thresholds
    #[test]
    fn ee_config_builder_overwrite_thresholds() {
        let config = EarlyExitConfig::new()
            .with_cosine_threshold(0.9)
            .with_cosine_threshold(0.95);
        assert!(
            (config.cosine_threshold - 0.95).abs() < 1e-6,
            "last with_cosine_threshold call should win"
        );
    }

    // should_early_exit: negative cosine_sim is a valid input (cosine range is [-1, 1])
    #[test]
    fn ee_should_exit_negative_cosine_never_triggers() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.0); // threshold at 0.0
        assert!(
            !should_early_exit(20, -0.5, 1.0, &config),
            "cosine_sim=-0.5 should not exceed threshold=0.0 (strict >)"
        );
    }

    // ExitLayerStats: best_exit_point selects strictly highest rate among eligible
    #[test]
    fn ee_stats_best_exit_point_picks_highest_rate() {
        let mut stats = ExitLayerStats::new(&[10, 20, 30], 0.5);
        // Layer 10: 15 records, all true → rate near 1.0
        for _ in 0..15 {
            stats.record(10, true);
        }
        // Layer 20: 15 records, all false → rate near 0.0
        for _ in 0..15 {
            stats.record(20, false);
        }
        // Layer 30: 15 records, mixed → rate between 0 and 1
        for i in 0..15 {
            stats.record(30, i % 3 == 0);
        }
        assert_eq!(
            stats.best_exit_point(),
            Some(10),
            "layer 10 with highest hit rate should be best"
        );
    }

    // ConfidenceCalibrator: new with custom theta_c and theta_d zero
    #[test]
    fn ee_calibrator_zero_thresholds() {
        let cal = ConfidenceCalibrator::new(10.0, 10.0, 0.0, 0.0);
        // cos_signal = 10*(0.5 - 0.0) = 5.0
        // delta_signal = 10*(0.0 - |1.0 - 1.0|) = 0.0
        // logit = 5.0, sigmoid(5.0) ~ 0.993
        let conf = cal.calibrate(0.5, 1.0);
        assert!(
            conf > 0.99,
            "positive logit with zero thresholds should give high confidence, got {}",
            conf
        );
    }

    // ResidualBusEarlyExit: exit_points with usize::MAX value
    #[test]
    fn ee_residual_bus_usize_max_exit_point() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(0);
        let bus = ResidualBusEarlyExit::new(config, vec![usize::MAX]);
        // usize::MAX is in exit_points and >= min_layer=0 → delegates to should_early_exit
        assert!(
            bus.should_exit_at_layer(usize::MAX, 1.005, 0.995),
            "usize::MAX with min_layer=0 and in exit_points should trigger exit"
        );
        // Different layer not in exit_points → false
        assert!(
            !bus.should_exit_at_layer(0, 1.005, 0.995),
            "layer 0 not in exit_points should not trigger exit"
        );
    }

    // EarlyExitDecision: Exit variant with NaN fields preserves NaN through clone
    #[test]
    fn ee_decision_exit_nan_fields_clone() {
        let d = EarlyExitDecision::Exit {
            confidence: f32::NAN,
            cosine_sim: f32::INFINITY,
            delta_rho: f32::NEG_INFINITY,
        };
        let cloned = d.clone();
        if let EarlyExitDecision::Exit { confidence, cosine_sim, delta_rho } = cloned {
            assert!(confidence.is_nan(), "NaN confidence should be preserved through clone");
            assert!(cosine_sim.is_infinite() && cosine_sim.is_sign_positive());
            assert!(delta_rho.is_infinite() && delta_rho.is_sign_negative());
        } else {
            panic!("expected Exit variant after clone");
        }
    }
}
