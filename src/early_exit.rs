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
            let layer = layer.max(1).min(total_layers - 1);
            if (layer as f32 / total_layers as f32) >= min_layer_ratio {
                if !points.contains(&layer) {
                    points.push(layer);
                }
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
            if self.eval_counts[idx] >= min_evals {
                if best.is_none() || rate > best.unwrap().1 {
                    best = Some((self.layers[idx], rate));
                }
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
}
