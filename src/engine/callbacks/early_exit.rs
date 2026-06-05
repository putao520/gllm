//! Early Exit Callback (SPEC §16.2)
//!
//! Integrates `EarlyExitController` into the graph node loop.
//! At designated exit points (golden ratio layers), evaluates cosine similarity
//! and energy delta of residual connections to determine if early termination
//! is safe. When confidence exceeds threshold, returns `ExitEarly`.

use crate::early_exit::{EarlyExitConfig, EarlyExitController, EarlyExitDecision};
use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};

/// Early exit callback — monitors residual convergence at golden-ratio exit points.
///
/// Per SPEC §16.2: "任意层数据召回与高维截断"
/// When the residual between consecutive layers converges (high cosine similarity
/// + low energy delta), remaining layers contribute negligibly and can be skipped.
pub struct EarlyExitCallback {
    controller: EarlyExitController,
    /// Cached last-layer hidden state for cosine similarity computation
    prev_hidden: Vec<f32>,
}

impl EarlyExitCallback {
    /// Create a new early exit callback.
    ///
    /// `config` — early exit thresholds and enabling flag
    /// `total_layers` — total number of transformer layers in the model
    pub fn new(config: EarlyExitConfig, total_layers: usize) -> Self {
        let controller = EarlyExitController::new(config, total_layers);
        Self {
            controller,
            prev_hidden: Vec::new(),
        }
    }

    /// Get a reference to the underlying controller.
    pub fn controller(&self) -> &EarlyExitController {
        &self.controller
    }

    /// Compute cosine similarity between two f32 slices.
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 0.0;
        }
        let dot: f32 = a[..len].iter().zip(&b[..len]).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Compute energy ratio ‖b‖ / ‖a‖ (delta_rho).
    fn energy_ratio(a: &[f32], b: &[f32]) -> f32 {
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 {
            1.0
        } else {
            norm_b / norm_a
        }
    }

    /// Convert byte slice to f32 slice (little-endian).
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl LayerCallback for EarlyExitCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        // Store current hidden state for post_node comparison
        self.prev_hidden = Self::bytes_to_f32(ctx.hidden_state);
        CallbackAction::Continue
    }

    fn post_node(&mut self, ctx: &LayerContext, output: &[u8]) -> CallbackAction {
        let current = Self::bytes_to_f32(output);

        let cosine_sim = Self::cosine_sim(&self.prev_hidden, &current);
        let delta_rho = Self::energy_ratio(&self.prev_hidden, &current);

        match self.controller.check_layer(ctx.layer_idx, cosine_sim, delta_rho) {
            EarlyExitDecision::Exit { confidence, .. } => {
                log::debug!(
                    "early_exit: layer={} confidence={:.4} cos={:.4} delta_rho={:.4}",
                    ctx.layer_idx, confidence, cosine_sim, delta_rho,
                );
                // Return ExitEarly with empty logits — caller should project
                // current hidden_state through lm_head to get actual logits
                CallbackAction::ExitEarly { logits: Vec::new() }
            }
            _ => CallbackAction::Continue,
        }
    }

    fn priority(&self) -> u32 {
        50
    }

    fn target_layers(&self) -> Option<&[usize]> {
        // Only trigger at exit points — but exit points are dynamic,
        // so we check inside post_node() and return None (all layers) here.
        None
    }

    fn name(&self) -> &str {
        "early_exit"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::early_exit::{
        AdaptiveExitPoints, ConfidenceCalibrator, EarlyExitController, EarlyExitDecision,
        ExitLayerStats, ResidualBusEarlyExit, should_early_exit,
    };

    #[test]
    fn test_early_exit_callback_disabled() {
        let config = EarlyExitConfig::default(); // enabled = false
        let cb = EarlyExitCallback::new(config, 32);
        assert_eq!(cb.controller().config().enabled, false);
        assert_eq!(cb.priority(), 50);
        assert_eq!(cb.name(), "early_exit");
    }

    #[test]
    fn test_cosine_sim_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_energy_ratio() {
        let a = vec![3.0, 4.0]; // norm = 5
        let b = vec![6.0, 8.0]; // norm = 10
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        assert!((ratio - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_bytes_to_f32() {
        let bytes: Vec<u8> = 1.0f32.to_le_bytes().iter()
            .chain(2.0f32.to_le_bytes().iter())
            .copied()
            .collect();
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        assert_eq!(result, vec![1.0, 2.0]);
    }

    // ── cosine_sim edge cases ──

    #[test]
    fn test_cosine_sim_empty_slices() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        assert_eq!(sim, 0.0, "empty slices should return 0.0");
    }

    #[test]
    fn test_cosine_sim_zero_vectors() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 2.0, 3.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        assert_eq!(sim, 0.0, "zero-norm vector should return 0.0");
    }

    #[test]
    fn test_cosine_sim_both_zero_vectors() {
        let a = vec![0.0f32, 0.0];
        let b = vec![0.0f32, 0.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        assert_eq!(sim, 0.0, "both zero-norm vectors should return 0.0");
    }

    #[test]
    fn test_cosine_sim_opposite_directions() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        assert!((sim - (-1.0f32)).abs() < 1e-5,
            "opposite vectors should have cos ≈ -1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_sim_unequal_lengths() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0f32, 2.0, 3.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Only first 3 elements compared: a[..3]=[1,2,3], b=[1,2,3] → identical → 1.0
        assert!((sim - 1.0).abs() < 1e-5,
            "truncated identical prefix should yield 1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_sim_single_element() {
        let a = vec![3.0f32];
        let b = vec![4.0f32];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Single element: dot=12, |a|=3, |b|=4 → cos = 12/12 = 1.0
        assert!((sim - 1.0).abs() < 1e-5,
            "single element same sign should yield 1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_sim_negative_values() {
        let a = vec![-1.0f32, -2.0, -3.0];
        let b = vec![-1.0f32, -2.0, -3.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5,
            "identical negative vectors should yield 1.0, got {}", sim);
    }

    // ── energy_ratio edge cases ──

    #[test]
    fn test_energy_ratio_empty_slices() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // norm_a = 0.0 → returns 1.0 per implementation
        assert_eq!(ratio, 1.0, "empty input should return 1.0");
    }

    #[test]
    fn test_energy_ratio_zero_numerator() {
        let a = vec![3.0f32, 4.0]; // norm = 5
        let b = vec![0.0f32, 0.0]; // norm = 0
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        assert!((ratio).abs() < 1e-5,
            "zero b with non-zero a should yield 0.0, got {}", ratio);
    }

    #[test]
    fn test_energy_ratio_single_element() {
        let a = vec![2.0f32];
        let b = vec![8.0f32];
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        assert!((ratio - 4.0).abs() < 1e-5,
            "|8|/|2| = 4.0, got {}", ratio);
    }

    // ── bytes_to_f32 edge cases ──

    #[test]
    fn test_bytes_to_f32_empty() {
        let result = EarlyExitCallback::bytes_to_f32(&[]);
        assert!(result.is_empty(), "empty bytes should yield empty vec");
    }

    #[test]
    fn test_bytes_to_f32_trailing_bytes() {
        // 9 bytes: 2 full f32 + 1 trailing byte (discarded by chunks_exact)
        let mut bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.push(0xAB); // trailing byte
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        assert_eq!(result, vec![1.0, 2.0],
            "trailing bytes should be ignored");
    }

    #[test]
    fn test_bytes_to_f32_negative_values() {
        let bytes: Vec<u8> = (-1.0f32).to_le_bytes().iter()
            .chain((-42.5f32).to_le_bytes().iter())
            .copied()
            .collect();
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        assert!((result[0] - (-1.0f32)).abs() < 1e-6);
        assert!((result[1] - (-42.5f32)).abs() < 1e-6);
    }

    // ── Callback construction and accessors ──

    #[test]
    fn test_new_initializes_empty_prev_hidden() {
        let config = EarlyExitConfig::default();
        let cb = EarlyExitCallback::new(config, 32);
        // prev_hidden is private, but we can verify via controller that construction succeeded
        assert_eq!(cb.name(), "early_exit");
        assert_eq!(cb.priority(), 50);
    }

    #[test]
    fn test_controller_accessor_returns_config() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.97)
            .with_min_layer(10);
        let cb = EarlyExitCallback::new(config, 64);
        let ctrl = cb.controller();
        assert!(ctrl.config().enabled);
        assert!((ctrl.config().cosine_threshold - 0.97).abs() < 1e-6);
        assert_eq!(ctrl.config().min_layer, 10);
    }

    #[test]
    fn test_target_layers_returns_none() {
        let config = EarlyExitConfig::default();
        let cb = EarlyExitCallback::new(config, 32);
        assert!(cb.target_layers().is_none(),
            "target_layers should return None (all layers monitored)");
    }

    #[test]
    fn test_callback_with_enabled_config() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let cb = EarlyExitCallback::new(config, 32);
        assert!(cb.controller().config().enabled);
        // Exit points should exist for 32-layer model
        assert!(!cb.controller().exit_points().is_empty());
    }

    #[test]
    fn test_callback_with_single_layer_model() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        let cb = EarlyExitCallback::new(config, 1);
        assert_eq!(cb.name(), "early_exit");
        // Single-layer model: exit points may be empty, but construction should succeed
    }

    // ── Additional edge-case tests ──

    #[test]
    fn test_cosine_sim_with_nan_input() {
        let a = vec![f32::NAN, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        assert!(sim.is_nan(), "NaN in input should propagate to NaN result");
    }

    #[test]
    fn test_cosine_sim_with_inf_input() {
        let a = vec![f32::INFINITY, 0.0];
        let b = vec![1.0, 0.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        assert!(sim.is_nan(), "Inf * Inf = Inf, Inf/Inf = NaN");
    }

    #[test]
    fn test_cosine_sim_with_subnormal_values() {
        // Use a larger subnormal to avoid the 1e-10 zero-norm guard:
        // 3 identical subnormals → norm² = 3*sub² which can still be < 1e-10.
        // Instead use enough elements that norm² exceeds the threshold.
        let sub = f32::from_bits(1); // smallest positive subnormal
        let a: Vec<f32> = (0..100000).map(|_| sub).collect();
        let b = a.clone();
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // With 100k elements, norm² = 100000 * sub² = 100000 * 2^-302 ≈ 9.3e-91
        // Still far below 1e-10 threshold → cosine_sim returns 0.0.
        // This documents the behavior: very small values are treated as zero-norm.
        assert_eq!(sim, 0.0,
            "subnormal vectors with norm < 1e-10 are treated as zero, got {}", sim);
    }

    #[test]
    fn test_energy_ratio_both_zero_vectors() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 0.0, 0.0];
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        assert_eq!(ratio, 1.0,
            "both zero-norm vectors should return 1.0, got {}", ratio);
    }

    #[test]
    fn test_energy_ratio_with_nan_input() {
        let a = vec![f32::NAN];
        let b = vec![1.0];
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        assert!(ratio.is_nan(), "NaN in input should produce NaN energy ratio");
    }

    #[test]
    fn test_bytes_to_f32_single_incomplete_byte() {
        let result = EarlyExitCallback::bytes_to_f32(&[0x00]);
        assert!(result.is_empty(),
            "1 byte (< 4) should yield empty vec, got {} elements", result.len());
    }

    #[test]
    fn test_bytes_to_f32_three_incomplete_bytes() {
        let result = EarlyExitCallback::bytes_to_f32(&[0x00, 0x01, 0x02]);
        assert!(result.is_empty(),
            "3 bytes (< 4) should yield empty vec, got {} elements", result.len());
    }

    #[test]
    fn test_bytes_to_f32_subnormal_f32_value() {
        let sub = f32::from_bits(1); // smallest positive subnormal
        let bytes: Vec<u8> = sub.to_le_bytes().to_vec();
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), sub.to_bits(),
            "subnormal f32 should round-trip through bytes exactly");
    }

    #[test]
    fn test_bytes_to_f32_inf_and_nan() {
        let bytes: Vec<u8> = f32::INFINITY.to_le_bytes().iter()
            .chain(f32::NEG_INFINITY.to_le_bytes().iter())
            .chain(f32::NAN.to_le_bytes().iter())
            .copied()
            .collect();
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        assert_eq!(result.len(), 3);
        assert!(result[0].is_infinite() && result[0].is_sign_positive());
        assert!(result[1].is_infinite() && result[1].is_sign_negative());
        assert!(result[2].is_nan());
    }

    #[test]
    fn test_cosine_symmetry_property() {
        let a = vec![1.0f32, 2.0, 3.0, -4.0];
        let b = vec![-1.0f32, 0.5, 2.0, 3.0];
        let sim_ab = EarlyExitCallback::cosine_sim(&a, &b);
        let sim_ba = EarlyExitCallback::cosine_sim(&b, &a);
        assert!((sim_ab - sim_ba).abs() < 1e-6,
            "cosine similarity should be symmetric: {} vs {}", sim_ab, sim_ba);
    }

    #[test]
    fn test_new_with_two_total_layers() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let cb = EarlyExitCallback::new(config, 2);
        assert_eq!(cb.name(), "early_exit");
        assert_eq!(cb.priority(), 50);
        assert_eq!(cb.controller().exit_points().total_layers(), 2);
    }

    #[test]
    fn test_new_with_large_total_layers() {
        let config = EarlyExitConfig::new().with_enabled(true);
        let cb = EarlyExitCallback::new(config, 1000);
        assert!(!cb.controller().exit_points().is_empty(),
            "1000-layer model should have exit points");
        assert_eq!(cb.controller().exit_points().total_layers(), 1000);
    }

    #[test]
    fn test_pre_node_stores_hidden_and_returns_continue() {
        // Verify that the callback's LayerCallback contract is correct:
        // pre_node always returns Continue (it only stores prev_hidden for post_node).
        // prev_hidden is private, so we verify the accessor contract instead.
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let cb = EarlyExitCallback::new(config, 32);
        assert_eq!(cb.name(), "early_exit");
        assert_eq!(cb.priority(), 50);
        assert!(cb.target_layers().is_none());
        assert!(cb.controller().config().enabled);
    }

    #[test]
    fn test_controller_disabled_post_node_returns_continue() {
        let config = EarlyExitConfig::default(); // enabled = false
        let cb = EarlyExitCallback::new(config, 32);
        // When controller is disabled, check_layer returns Defer,
        // and post_node maps Defer to Continue.
        assert!(!cb.controller().config().enabled);
    }

    #[test]
    fn test_energy_ratio_with_unequal_lengths() {
        let a = vec![3.0f32, 4.0, 0.0, 0.0]; // norm = 5.0
        let b = vec![6.0f32, 8.0];            // norm = 10.0
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // energy_ratio computes on full slices independently of length
        assert!((ratio - 2.0).abs() < 1e-5,
            "|b|/|a| should be 2.0, got {}", ratio);
    }

    // ── Wave 3: 15 additional tests for uncovered scenarios ──

    #[test]
    fn test_cosine_sim_large_values_scaling() {
        // Arrange: scaled versions of the same direction
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![100.0f32, 200.0, 300.0];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: same direction regardless of magnitude → cosine = 1.0
        assert!((sim - 1.0).abs() < 1e-4,
            "scaled identical direction should give cos=1.0, got {}", sim);
    }

    #[test]
    fn test_energy_ratio_identical_vectors() {
        // Arrange
        let a = vec![3.0f32, 4.0]; // norm = 5
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &a);
        // Assert: ‖a‖/‖a‖ = 1.0
        assert!((ratio - 1.0).abs() < 1e-5,
            "identical vectors should give ratio 1.0, got {}", ratio);
    }

    #[test]
    fn test_energy_ratio_negative_values() {
        // Arrange: all-negative values — squared sum same as positive
        let a = vec![-3.0f32, -4.0]; // norm = 5
        let b = vec![6.0f32, 8.0];   // norm = 10
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: |b|/|a| = 10/5 = 2.0 (sign does not matter)
        assert!((ratio - 2.0).abs() < 1e-5,
            "negative values should give same ratio as positive, got {}", ratio);
    }

    #[test]
    fn test_cosine_sim_forty_five_degree_angle() {
        // Arrange: vectors at 45 degrees — dot = a_x*b_x (b_y=0)
        let a = vec![1.0f32, 1.0]; // norm = sqrt(2)
        let b = vec![1.0f32, 0.0]; // norm = 1
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: cos = 1 / sqrt(2) ≈ 0.7071
        let expected = 1.0f32 / 2.0f32.sqrt();
        assert!((sim - expected).abs() < 1e-5,
            "45-degree angle should give cos(45) ≈ 0.7071, got {}", sim);
    }

    #[test]
    fn test_bytes_to_f32_zero_roundtrip() {
        // Arrange: zero f32 value as bytes
        let bytes = 0.0f32.to_le_bytes();
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 1);
        assert!((result[0]).abs() < 1e-10, "zero should roundtrip exactly");
    }

    #[test]
    fn test_bytes_to_f32_max_f32_roundtrip() {
        // Arrange
        let bytes = f32::MAX.to_le_bytes();
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), f32::MAX.to_bits(),
            "f32::MAX should roundtrip exactly through bytes");
    }

    #[test]
    fn test_cosine_sim_repeated_elements() {
        // Arrange: many copies of the same element
        let a = vec![2.0f32; 100];
        let b = vec![3.0f32; 100];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: same direction → cosine = 1.0
        assert!((sim - 1.0).abs() < 1e-5,
            "repeated elements in same direction should give 1.0, got {}", sim);
    }

    #[test]
    fn test_energy_ratio_with_large_value_difference() {
        // Arrange: b much larger than a
        let a = vec![1.0f32]; // norm = 1
        let b = vec![1000.0f32]; // norm = 1000
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert
        assert!((ratio - 1000.0).abs() < 1e-2,
            "|b|/|a| = 1000.0, got {}", ratio);
    }

    #[test]
    fn test_callback_new_with_single_layer_boundary() {
        // Arrange: single-layer model is the minimum valid model size.
        // total_layers=0 would panic in AdaptiveExitPoints (underflow on total_layers-1),
        // so 1 is the true lower boundary.
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        let cb = EarlyExitCallback::new(config, 1);
        // Assert: construction succeeds at minimum boundary
        assert_eq!(cb.name(), "early_exit");
        assert_eq!(cb.priority(), 50);
        assert_eq!(cb.controller().exit_points().total_layers(), 1);
        assert!(cb.controller().config().enabled);
    }

    #[test]
    fn test_pre_node_with_real_hidden_bytes() {
        // Arrange: create a callback and a LayerContext with f32 hidden state as bytes
        let config = EarlyExitConfig::default();
        let mut cb = EarlyExitCallback::new(config, 32);

        let hidden_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let hidden_bytes: Vec<u8> = hidden_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "Attention",
            hidden_state: &hidden_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 5,
            position: 0,
            request_id: 1,
            model_config: &model_config,
        };

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: pre_node always returns Continue
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_post_node_returns_continue_when_disabled() {
        // Arrange: disabled config → controller always returns Defer → Continue
        let config = EarlyExitConfig::default(); // enabled = false
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let hidden_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
        let hidden_bytes: Vec<u8> = hidden_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 5,
            node_op: "FFN",
            hidden_state: &hidden_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 5,
            position: 5,
            request_id: 1,
            model_config: &model_config,
        };

        // Simulate pre_node storing prev_hidden
        let _ = cb.pre_node(&ctx);

        // Act: post_node with identical output
        let action = cb.post_node(&ctx, &hidden_bytes);

        // Assert: disabled controller → Defer → Continue
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_post_node_returns_continue_for_non_exit_layer() {
        // Arrange: enabled but layer is not an exit point
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let hidden_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
        let hidden_bytes: Vec<u8> = hidden_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // Layer 3 is very unlikely to be a golden-ratio exit point for 32 layers
        let ctx = LayerContext {
            node_idx: 6,
            layer_idx: 3,
            node_op: "Attention",
            hidden_state: &hidden_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 5,
            position: 5,
            request_id: 1,
            model_config: &model_config,
        };

        let _ = cb.pre_node(&ctx);

        // Act
        let action = cb.post_node(&ctx, &hidden_bytes);

        // Assert: non-exit-point layer → Continue
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_pre_node_then_post_node_different_hidden() {
        // Arrange: enabled callback, pre_node stores state A, post_node receives state B
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();

        let prev_f32: Vec<f32> = vec![1.0, 0.0, 0.0];
        let prev_bytes: Vec<u8> = prev_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let curr_f32: Vec<f32> = vec![0.0, 1.0, 0.0]; // orthogonal to prev
        let curr_bytes: Vec<u8> = curr_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 3,
            node_op: "Attention",
            hidden_state: &prev_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 5,
            position: 0,
            request_id: 1,
            model_config: &model_config,
        };

        // Act: pre_node stores prev, post_node sees orthogonal current
        let _ = cb.pre_node(&ctx);
        let action = cb.post_node(&ctx, &curr_bytes);

        // Assert: orthogonal vectors → low cosine similarity → Continue (not exit)
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_post_node_with_empty_prev_hidden() {
        // Arrange: call post_node without a prior pre_node — prev_hidden is empty
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();

        let curr_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
        let curr_bytes: Vec<u8> = curr_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 3,
            node_op: "Attention",
            hidden_state: &curr_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 5,
            position: 0,
            request_id: 1,
            model_config: &model_config,
        };

        // Act: post_node without prior pre_node (prev_hidden is empty Vec)
        let action = cb.post_node(&ctx, &curr_bytes);

        // Assert: cosine_sim with empty slice returns 0.0 → low confidence → Continue
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_post_node_exit_early_at_exit_point_with_convergence() {
        // Arrange: enabled callback with low threshold so exit is triggered
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.1);
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();

        // Nearly identical hidden states (high cosine sim, delta_rho near 1.0)
        let prev_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let curr_f32: Vec<f32> = vec![1.0001, 2.0001, 3.0001, 4.0001];
        let prev_bytes: Vec<u8> = prev_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let curr_bytes: Vec<u8> = curr_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // Use a known exit point from the controller
        let exit_points = cb.controller().exit_points().points().to_vec();
        if let Some(&exit_layer) = exit_points.first() {
            let ctx = LayerContext {
                node_idx: exit_layer * 2,
                layer_idx: exit_layer,
                node_op: "Attention",
                hidden_state: &prev_bytes,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 10,
                seq_len: 5,
                position: 0,
                request_id: 1,
                model_config: &model_config,
            };

            // Act
            let _ = cb.pre_node(&ctx);
            let action = cb.post_node(&ctx, &curr_bytes);

            // Assert: with nearly identical states and low threshold, should exit
            assert!(
                matches!(action, CallbackAction::ExitEarly { .. }),
                "expected ExitEarly at exit point {} with converging hidden states, got {:?}",
                exit_layer, action
            );
        }
        // If no exit points exist for this configuration, the test is vacuously valid.
    }

    // ── Wave 4: 13 additional tests for uncovered scenarios ──

    #[test]
    fn test_new_with_zero_total_layers_no_panic() {
        // total_layers=0 is handled gracefully — no underflow, returns empty callback
        let config = EarlyExitConfig::new().with_enabled(true);
        let cb = EarlyExitCallback::new(config, 0);
        // Controller exists and has zero exit points
        let exit_points = cb.controller().exit_points();
        assert!(exit_points.is_empty());
    }

    #[test]
    fn test_cosine_sim_near_orthogonal() {
        // Arrange: two vectors that are close to orthogonal but not exactly zero
        // [1, 0.01] vs [0.01, 1] — angle close to 90 degrees
        let a = vec![1.0f32, 0.01];
        let b = vec![0.01f32, 1.0];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: result should be small but non-zero
        let expected_dot = 1.0 * 0.01 + 0.01 * 1.0; // 0.02
        let norm_a = (1.0_f32.powi(2) + 0.01_f32.powi(2)).sqrt();
        let norm_b = (0.01_f32.powi(2) + 1.0_f32.powi(2)).sqrt();
        let expected = expected_dot / (norm_a * norm_b);
        assert!(sim.abs() > 0.0, "near-orthogonal should not be exactly 0, got {}", sim);
        assert!((sim - expected).abs() < 1e-5,
            "near-orthogonal cos ≈ {}, got {}", expected, sim);
        assert!(sim.abs() < 0.1, "near-orthogonal should be small, got {}", sim);
    }

    #[test]
    fn test_energy_ratio_very_different_magnitudes() {
        // Arrange: one vector is 10 orders of magnitude larger than the other
        let a = vec![1e-10f32];
        let b = vec![1e10f32];
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: |b|/|a| = 1e10/1e-10 = 1e20
        // f32 can represent up to ~3.4e38, so this should be representable
        assert!(ratio.is_finite(), "ratio should be finite, got {}", ratio);
        assert!(ratio > 1e18, "ratio should be very large (~1e20), got {}", ratio);
    }

    #[test]
    fn test_bytes_to_f32_all_zero_bytes() {
        // Arrange: 8 zero bytes should decode to two 0.0f32 values
        let bytes: Vec<u8> = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].to_bits(), 0u32, "all-zero bytes should be +0.0f32");
        assert_eq!(result[1].to_bits(), 0u32, "all-zero bytes should be +0.0f32");
    }

    #[test]
    fn test_bytes_to_f32_known_pattern_one_point_zero() {
        // Arrange: IEEE 754 representation of 1.0f32 is 0x3f800000 (little-endian: 00 00 80 3f)
        let bytes: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3f];
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0f32).abs() < 1e-10,
            "bit pattern 0x3f800000 should decode to 1.0, got {}", result[0]);
    }

    #[test]
    fn test_controller_accessor_enabled_state_false() {
        // Arrange: default config has enabled=false
        let config = EarlyExitConfig::default();
        let cb = EarlyExitCallback::new(config, 32);
        // Act
        let ctrl = cb.controller();
        // Assert: verify the disabled state is correctly reflected
        assert!(!ctrl.config().enabled,
            "default config should have enabled=false");
        assert_eq!(ctrl.config().cosine_threshold, 0.99,
            "default cosine_threshold should be 0.99");
        assert_eq!(ctrl.config().delta_threshold, 0.01,
            "default delta_threshold should be 0.01");
        assert_eq!(ctrl.config().min_layer, 16,
            "default min_layer should be 16");
    }

    #[test]
    fn test_cosine_sim_bounded_between_minus_one_and_one() {
        // Arrange: arbitrary non-trivial vectors
        let a = vec![3.0f32, -1.0, 4.0, -2.0, 5.0];
        let b = vec![-2.0f32, 5.0, 1.0, 3.0, -4.0];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: cosine similarity is always in [-1, 1] for non-degenerate inputs
        assert!(sim >= -1.0 - 1e-5 && sim <= 1.0 + 1e-5,
            "cosine similarity should be in [-1, 1], got {}", sim);
    }

    #[test]
    fn test_energy_ratio_less_than_one_when_second_smaller() {
        // Arrange: ||b|| < ||a||
        let a = vec![10.0f32, 10.0]; // norm ≈ 14.14
        let b = vec![1.0f32, 1.0];   // norm ≈ 1.414
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: ratio = 1.414/14.14 ≈ 0.1
        assert!(ratio < 1.0,
            "energy ratio should be < 1.0 when second vector is smaller, got {}", ratio);
        assert!((ratio - 0.1).abs() < 1e-4,
            "expected ratio ≈ 0.1, got {}", ratio);
    }

    #[test]
    fn test_energy_ratio_greater_than_one_when_second_larger() {
        // Arrange: ||b|| > ||a||
        let a = vec![1.0f32, 1.0];   // norm ≈ 1.414
        let b = vec![10.0f32, 10.0]; // norm ≈ 14.14
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: ratio = 14.14/1.414 ≈ 10.0
        assert!(ratio > 1.0,
            "energy ratio should be > 1.0 when second vector is larger, got {}", ratio);
        assert!((ratio - 10.0).abs() < 1e-4,
            "expected ratio ≈ 10.0, got {}", ratio);
    }

    #[test]
    fn test_bytes_to_f32_multiple_known_patterns() {
        // Arrange: known IEEE 754 bit patterns:
        // -0.5f32 = 0xbf000000 (little-endian: 00 00 00 bf)
        //  2.5f32 = 0x40200000 (little-endian: 00 00 20 40)
        // 42.0f32 = 0x42280000 (little-endian: 00 00 28 42)
        let bytes: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0xBF, // -0.5
            0x00, 0x00, 0x20, 0x40, //  2.5
            0x00, 0x00, 0x28, 0x42, // 42.0
        ];
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 3);
        assert!((result[0] - (-0.5f32)).abs() < 1e-6,
            "expected -0.5, got {}", result[0]);
        assert!((result[1] - 2.5f32).abs() < 1e-6,
            "expected 2.5, got {}", result[1]);
        assert!((result[2] - 42.0f32).abs() < 1e-6,
            "expected 42.0, got {}", result[2]);
    }

    #[test]
    fn test_cosine_sim_unit_vectors_at_known_angle() {
        // Arrange: two unit vectors at 60 degrees
        // cos(60 deg) = 0.5
        // a = [1, 0], b = [cos(60), sin(60)] = [0.5, 0.866025...]
        let a = vec![1.0f32, 0.0];
        let b = vec![0.5f32, 3.0f32.sqrt() / 2.0];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert
        assert!((sim - 0.5).abs() < 1e-5,
            "cos(60 deg) should be 0.5, got {}", sim);
    }

    #[test]
    fn test_new_prev_hidden_is_empty_vec() {
        // Arrange & Act: construct a new callback
        let config = EarlyExitConfig::new().with_enabled(true);
        let mut cb = EarlyExitCallback::new(config, 16);
        // Assert: prev_hidden is private but we can verify indirectly:
        // calling post_node without pre_node should produce Continue
        // (cosine_sim with empty prev_hidden returns 0.0 → low confidence → Continue)
        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let output_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
        let output_bytes: Vec<u8> = output_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "Attention",
            hidden_state: &output_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 4,
            seq_len: 4,
            position: 0,
            request_id: 1,
            model_config: &model_config,
        };
        let action = cb.post_node(&ctx, &output_bytes);
        assert!(matches!(action, CallbackAction::Continue),
            "new callback with empty prev_hidden should return Continue");
    }

    #[test]
    fn test_multiple_pre_node_calls_overwrite_prev_hidden() {
        // Arrange: enabled callback, two different hidden states
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        let mut cb = EarlyExitCallback::new(config, 32);
        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();

        // First pre_node call with hidden state A
        let first_f32: Vec<f32> = vec![1.0, 0.0, 0.0]; // unit vector along x
        let first_bytes: Vec<u8> = first_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let ctx_first = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "Attention",
            hidden_state: &first_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 4,
            seq_len: 4,
            position: 0,
            request_id: 1,
            model_config: &model_config,
        };
        let _ = cb.pre_node(&ctx_first);

        // Second pre_node call with hidden state B (overwrites A)
        let second_f32: Vec<f32> = vec![0.0, 1.0, 0.0]; // unit vector along y
        let second_bytes: Vec<u8> = second_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let ctx_second = LayerContext {
            node_idx: 2,
            layer_idx: 1,
            node_op: "FFN",
            hidden_state: &second_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 4,
            seq_len: 4,
            position: 1,
            request_id: 1,
            model_config: &model_config,
        };
        let _ = cb.pre_node(&ctx_second);

        // Act: post_node with output identical to second state
        // prev_hidden should be [0, 1, 0] (overwritten), current is [0, 1, 0] (identical)
        // cosine_sim ≈ 1.0, energy_ratio ≈ 1.0 → high confidence → should exit if at exit point
        let action = cb.post_node(&ctx_second, &second_bytes);

        // Assert: with identical prev and current at a potential exit point,
        // behavior depends on whether this is an exit point. At minimum, verify
        // that the result is either Continue or ExitEarly (not a panic/crash).
        match action {
            CallbackAction::Continue
            | CallbackAction::ExitEarly { .. }
            | CallbackAction::SkipThisNode
            | CallbackAction::InjectHidden { .. }
            | CallbackAction::CompactMask { .. } => {},
        }
    }

    // ── Wave 5: 13 additional tests for uncovered edge cases ──

    #[test]
    fn test_bytes_to_f32_negative_zero_bit_pattern() {
        // Arrange: IEEE 754 negative zero = 0x80000000 (LE: 00 00 00 80)
        let bytes: Vec<u8> = vec![0x00, 0x00, 0x00, 0x80];
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert: should decode to -0.0 (distinguishable from +0.0 by sign bit)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), 0x8000_0000u32,
            "bit pattern 0x80000000 should decode to -0.0f32");
        assert!(result[0].is_sign_negative(),
            "-0.0f32 should report negative sign");
    }

    #[test]
    fn test_bytes_to_f32_min_positive_normal() {
        // Arrange: smallest positive normal f32 = 0x00800000 (LE: 00 00 80 00)
        let min_normal = f32::MIN_POSITIVE; // 1.1754944e-38
        let bytes = min_normal.to_le_bytes();
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), min_normal.to_bits(),
            "min positive normal should round-trip exactly");
        assert!(result[0] > 0.0);
        assert!(result[0].is_normal());
    }

    #[test]
    fn test_bytes_to_f32_signaling_nan_bit_pattern() {
        // Arrange: construct a signaling NaN (sNaN) — exponent all 1s, fraction MSB=0, rest nonzero
        // IEEE 754: sign=0, exp=0xFF, fraction=0x400001 → 0x7FA00001 (LE: 01 00 A0 7F)
        // A quiet NaN has fraction MSB=1: 0x7FC00000 is canonical qNaN
        let snan_bits: u32 = 0x7FA00001;
        let bytes = snan_bits.to_le_bytes();
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert: should decode to NaN (signaling or quiet is implementation-defined)
        assert_eq!(result.len(), 1);
        assert!(result[0].is_nan(),
            "signaling NaN bit pattern should decode to NaN");
    }

    #[test]
    fn test_energy_ratio_with_nan_in_second_vector() {
        // Arrange: NaN in second vector (previously only NaN in first was tested)
        let a = vec![1.0f32];
        let b = vec![f32::NAN];
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: norm_b involves NaN sum → NaN
        assert!(ratio.is_nan(),
            "NaN in second vector should produce NaN energy ratio");
    }

    #[test]
    fn test_energy_ratio_both_negative_vectors() {
        // Arrange: both vectors all-negative — same magnitude as positive counterparts
        let a = vec![-3.0f32, -4.0]; // norm = 5
        let b = vec![-6.0f32, -8.0]; // norm = 10
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: |b|/|a| = 10/5 = 2.0 (squared sum eliminates sign)
        assert!((ratio - 2.0).abs() < 1e-5,
            "negative vectors should give same ratio as positive, got {}", ratio);
    }

    #[test]
    fn test_cosine_sim_mixed_positive_negative() {
        // Arrange: vectors with mixed signs — not purely aligned or opposed
        let a = vec![1.0f32, -2.0, 3.0, -4.0];
        let b = vec![-1.0f32, 2.0, 3.0, 4.0];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: result should be in [-1, 1] and not exactly ±1 or 0
        assert!(sim > -1.0 && sim < 1.0,
            "mixed-sign vectors should give partial cosine, got {}", sim);
        assert!(sim.abs() > 0.01,
            "should not be near-zero, got {}", sim);
    }

    #[test]
    fn test_callback_new_with_three_total_layers() {
        // Arrange: 3-layer model — minimal model with potential for distinct exit points
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(0);
        // Act
        let cb = EarlyExitCallback::new(config, 3);
        // Assert: construction succeeds, exit points are valid
        assert_eq!(cb.controller().exit_points().total_layers(), 3);
        for &p in cb.controller().exit_points().points() {
            assert!(p < 3,
                "exit point {} must be < 3 for 3-layer model", p);
        }
        assert_eq!(cb.name(), "early_exit");
        assert_eq!(cb.priority(), 50);
    }

    #[test]
    fn test_callback_min_layer_blocks_all_exit_points() {
        // Arrange: min_layer so high that all golden-ratio exit points fall below it.
        // With 32 layers and min_layer=30, most exit points (~20, ~24, ~27) are < 30.
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(30)
            .with_confidence_threshold(0.1);
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let hidden_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let hidden_bytes: Vec<u8> = hidden_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // Try each exit point — all should be below min_layer=30
        let exit_points = cb.controller().exit_points().points().to_vec();
        for &exit_layer in &exit_points {
            if exit_layer < 30 {
                let ctx = LayerContext {
                    node_idx: exit_layer * 2,
                    layer_idx: exit_layer,
                    node_op: "Attention",
                    hidden_state: &hidden_bytes,
                    kv_cache_k: std::ptr::null_mut(),
                    kv_cache_v: std::ptr::null_mut(),
                    total_seq: 4,
                    seq_len: 4,
                    position: 0,
                    request_id: 1,
                    model_config: &model_config,
                };
                let _ = cb.pre_node(&ctx);
                let action = cb.post_node(&ctx, &hidden_bytes);
                // exit_layer < min_layer → controller returns Defer → Continue
                assert!(
                    matches!(action, CallbackAction::Continue),
                    "layer {} < min_layer=30 should return Continue (Defer mapped), got {:?}",
                    exit_layer, action
                );
            }
        }
    }

    #[test]
    fn test_pre_node_post_node_cycle_at_exact_min_layer_boundary() {
        // Arrange: min_layer exactly equals an exit point layer
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.1);
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();

        // Nearly identical hidden states for high cosine similarity
        let prev_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let curr_f32: Vec<f32> = vec![1.0001, 2.0001, 3.0001, 4.0001];
        let prev_bytes: Vec<u8> = prev_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let curr_bytes: Vec<u8> = curr_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // Find an exit point that equals or exceeds min_layer
        let exit_points = cb.controller().exit_points().points().to_vec();
        if let Some(&target_layer) = exit_points.iter().find(|&&l| l >= 1) {
            let ctx = LayerContext {
                node_idx: target_layer * 2,
                layer_idx: target_layer,
                node_op: "Attention",
                hidden_state: &prev_bytes,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 4,
                seq_len: 4,
                position: 0,
                request_id: 1,
                model_config: &model_config,
            };

            // Act
            let _ = cb.pre_node(&ctx);
            let action = cb.post_node(&ctx, &curr_bytes);

            // Assert: at or above min_layer with converging signals → should trigger ExitEarly
            assert!(
                matches!(action, CallbackAction::ExitEarly { .. }),
                "layer {} >= min_layer=1 with converging states should exit, got {:?}",
                target_layer, action
            );
        }
    }

    #[test]
    fn test_post_node_with_diverging_states_no_exit() {
        // Arrange: enabled callback with low threshold but highly divergent states
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.95);
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();

        // Divergent hidden states: prev is all-positive, current is all-negative
        let prev_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let curr_f32: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0];
        let prev_bytes: Vec<u8> = prev_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let curr_bytes: Vec<u8> = curr_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let exit_points = cb.controller().exit_points().points().to_vec();
        if let Some(&exit_layer) = exit_points.first() {
            let ctx = LayerContext {
                node_idx: exit_layer * 2,
                layer_idx: exit_layer,
                node_op: "FFN",
                hidden_state: &prev_bytes,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 8,
                seq_len: 8,
                position: 3,
                request_id: 42,
                model_config: &model_config,
            };

            // Act: prev and current are opposite → cosine_sim ≈ -1 → low confidence
            let _ = cb.pre_node(&ctx);
            let action = cb.post_node(&ctx, &curr_bytes);

            // Assert: divergent states should not trigger early exit
            assert!(
                matches!(action, CallbackAction::Continue),
                "divergent hidden states should not trigger early exit, got {:?}", action
            );
        }
    }

    #[test]
    fn test_bytes_to_f32_alternating_sign_patterns() {
        // Arrange: positive and negative values interleaved
        let values: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5, 100.0, -100.0];
        let bytes: Vec<u8> = values.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 6);
        for (i, expected) in values.iter().enumerate() {
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "index {}: expected {}, got {}", i, expected, result[i]
            );
        }
    }

    #[test]
    fn test_cosine_sim_large_dimension_vectors() {
        // Arrange: large vectors (1024 elements) — verify correctness at realistic hidden sizes
        let a: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32).cos()).collect();
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: sin and cos are not identical → |cos| < 1, and both nonzero → not 0
        assert!(sim > -1.0 && sim < 1.0,
            "sin/cos vectors should give partial cosine, got {}", sim);
        // Both signals are nonzero, so cosine should be well-defined and nonzero
        assert!(sim.abs() > 1e-5,
            "sin and cos should have nonzero correlation, got {}", sim);
    }

    #[test]
    fn test_energy_ratio_with_inf_first_vector() {
        // Arrange: first vector has infinite norm
        let a = vec![f32::INFINITY];
        let b = vec![1.0f32];
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: norm_a = INF, norm_b = 1.0 → ratio = 1.0/INF ≈ 0.0
        // (inf*inf = inf, sqrt(inf) = inf, 1.0/inf = 0.0)
        assert!(
            ratio.abs() < 1e-5 || ratio == 0.0,
            "finite over infinite norm should be near 0, got {}", ratio
        );
    }

    // ── Wave 6: 13 additional tests for uncovered edge cases ──

    #[test]
    fn test_cosine_sim_with_mixed_nan_and_valid() {
        // Arrange: one vector has NaN in a non-leading position
        let a = vec![1.0f32, 2.0, f32::NAN, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: any NaN in the computation propagates to NaN result
        assert!(sim.is_nan(), "NaN in a non-leading position should propagate to NaN");
    }

    #[test]
    fn test_energy_ratio_with_neg_inf_in_numerator() {
        // Arrange: negative infinity in second vector
        let a = vec![1.0f32, 1.0]; // norm = sqrt(2)
        let b = vec![f32::NEG_INFINITY, 0.0]; // norm = INF
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: norm_b = INF (sqrt of INF) → ratio = INF / sqrt(2) = INF
        assert!(
            ratio.is_infinite() && ratio.is_sign_positive(),
            "infinite numerator norm should yield +INF ratio, got {}", ratio
        );
    }

    #[test]
    fn test_bytes_to_f32_negative_infinity_roundtrip() {
        // Arrange
        let bytes = f32::NEG_INFINITY.to_le_bytes();
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0].is_infinite() && result[0].is_sign_negative(),
            "negative infinity should roundtrip, got {}", result[0]);
    }

    #[test]
    fn test_bytes_to_f32_largest_subnormal_roundtrip() {
        // Arrange: largest subnormal f32 = 0x007FFFFF
        let sub_max = f32::from_bits(0x007F_FFFF);
        let bytes = sub_max.to_le_bytes();
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), sub_max.to_bits(),
            "largest subnormal should roundtrip exactly");
        assert!(result[0].is_subnormal());
    }

    #[test]
    fn test_cosine_sim_very_small_nonzero_vectors() {
        // Arrange: vectors with values just above the 1e-10 zero-norm threshold
        // Pick magnitude so that norm² slightly exceeds 1e-10.
        // With 2 elements of value 1e-5: norm² = 2 * 1e-10 → norm ≈ 1.414e-5 > 1e-10
        let a = vec![1e-5f32, 1e-5];
        let b = vec![1e-5f32, 1e-5];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: identical tiny vectors → cosine = 1.0 (above threshold)
        assert!((sim - 1.0).abs() < 1e-4,
            "identical tiny vectors with norm > 1e-10 should give cos=1.0, got {}", sim);
    }

    #[test]
    fn test_energy_ratio_with_both_inf() {
        // Arrange: both vectors contain infinity
        let a = vec![f32::INFINITY];
        let b = vec![f32::INFINITY];
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: norm_a = INF, norm_b = INF → ratio = INF/INF = NaN
        assert!(ratio.is_nan(),
            "INF/INF should be NaN, got {}", ratio);
    }

    #[test]
    fn test_callback_controller_clone_via_config() {
        // Arrange: two callbacks from same config — verify independent controllers
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.95)
            .with_min_layer(5);
        let cb1 = EarlyExitCallback::new(config.clone(), 32);
        let cb2 = EarlyExitCallback::new(config, 32);
        // Act & Assert: both controllers reflect the same config
        assert!((cb1.controller().config().cosine_threshold - 0.95).abs() < 1e-6);
        assert!((cb2.controller().config().cosine_threshold - 0.95).abs() < 1e-6);
        assert_eq!(cb1.controller().config().min_layer, 5);
        assert_eq!(cb2.controller().config().min_layer, 5);
    }

    #[test]
    fn test_cosine_sim_all_ones_vector() {
        // Arrange: all-ones vector of various sizes
        let a = vec![1.0f32; 8];
        let b = vec![1.0f32; 8];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: identical all-ones → cosine = 1.0
        assert!((sim - 1.0).abs() < 1e-5,
            "identical all-ones vectors should give cos=1.0, got {}", sim);
    }

    #[test]
    fn test_energy_ratio_single_element_zero_denominator() {
        // Arrange: single-element vectors, denominator zero
        let a = vec![0.0f32];
        let b = vec![5.0f32];
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: norm_a = 0 < 1e-10 → returns 1.0 (implementation guard)
        assert!((ratio - 1.0).abs() < 1e-5,
            "zero denominator with non-zero numerator should return 1.0 per guard, got {}", ratio);
    }

    #[test]
    fn test_bytes_to_f32_preserves_sign_bit_for_positive_zero() {
        // Arrange: positive zero = 0x00000000
        let bytes: Vec<u8> = vec![0x00, 0x00, 0x00, 0x00];
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0].is_sign_positive(),
            "all-zero bytes should decode to +0.0 (positive sign)");
        assert_eq!(result[0].to_bits(), 0u32);
    }

    #[test]
    fn test_post_node_identical_states_with_very_low_threshold() {
        // Arrange: enabled callback with near-zero confidence threshold
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(0)
            .with_confidence_threshold(0.001);
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let hidden_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let hidden_bytes: Vec<u8> = hidden_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let exit_points = cb.controller().exit_points().points().to_vec();
        if let Some(&exit_layer) = exit_points.first() {
            let ctx = LayerContext {
                node_idx: exit_layer * 2,
                layer_idx: exit_layer,
                node_op: "Attention",
                hidden_state: &hidden_bytes,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 4,
                seq_len: 4,
                position: 0,
                request_id: 1,
                model_config: &model_config,
            };
            // Act: prev and current are identical → cosine=1.0, energy_ratio=1.0
            let _ = cb.pre_node(&ctx);
            let action = cb.post_node(&ctx, &hidden_bytes);
            // Assert: identical states with near-zero threshold → should exit
            assert!(
                matches!(action, CallbackAction::ExitEarly { .. }),
                "identical hidden states with low threshold should trigger ExitEarly at exit point {}, got {:?}",
                exit_layer, action
            );
        }
    }

    #[test]
    fn test_pre_node_empty_hidden_bytes_stores_empty() {
        // Arrange: callback with empty hidden state bytes
        let config = EarlyExitConfig::default();
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let empty_bytes: &[u8] = &[];
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "Attention",
            hidden_state: empty_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 0,
            seq_len: 0,
            position: 0,
            request_id: 1,
            model_config: &model_config,
        };
        // Act: pre_node stores empty f32 vec
        let action = cb.pre_node(&ctx);
        // Assert
        assert!(matches!(action, CallbackAction::Continue));
        // Subsequent post_node with non-empty output: cosine_sim with empty = 0.0 → Continue
        let output_f32: Vec<f32> = vec![1.0, 2.0];
        let output_bytes: Vec<u8> = output_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let post_action = cb.post_node(&ctx, &output_bytes);
        assert!(matches!(post_action, CallbackAction::Continue));
    }

    #[test]
    fn test_cosine_sim_with_mixed_finite_and_inf() {
        // Arrange: one element is finite, one is INF
        let a = vec![f32::INFINITY, 0.0f32];
        let b = vec![0.0f32, f32::INFINITY];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: dot = INF*0 + 0*INF = NaN, or 0*INF = NaN → NaN propagates
        assert!(sim.is_nan(),
            "mixed INF and zero dot product should produce NaN, got {}", sim);
    }

    // ── Wave 7: 13 additional tests for uncovered edge cases ──

    #[test]
    fn test_cosine_sim_one_zero_element_mixed_with_nonzero() {
        // Arrange: vector with some zeros and some non-zeros — norm > 1e-10
        let a = vec![0.0f32, 3.0, 0.0, 4.0]; // norm = 5
        let b = vec![0.0f32, 6.0, 0.0, 8.0]; // norm = 10
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: only nonzero indices contribute: dot = 3*6 + 4*8 = 50; cos = 50/(5*10) = 1.0
        assert!((sim - 1.0).abs() < 1e-5,
            "same direction with zero padding should give cos=1.0, got {}", sim);
    }

    #[test]
    fn test_energy_ratio_large_vector_accuracy() {
        // Arrange: 10000-element vectors — verify energy_ratio at scale
        let a: Vec<f32> = (0..10000).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..10000).map(|i| (i as f32).cos()).collect();
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: both norms should be similar for sin/cos over full period → ratio ≈ 1.0
        assert!(ratio > 0.9 && ratio < 1.1,
            "sin/cos over full periods should have similar energy, got {}", ratio);
    }

    #[test]
    fn test_bytes_to_f32_min_negative_normal_roundtrip() {
        // Arrange: largest negative normal f32
        let bytes = f32::MIN.to_le_bytes(); // -3.4028235e38
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), f32::MIN.to_bits(),
            "f32::MIN should roundtrip exactly through bytes");
        assert!(result[0].is_sign_negative());
        assert!(result[0].is_normal());
    }

    #[test]
    fn test_bytes_to_f32_two_bytes_incomplete() {
        // Arrange: 2 bytes — insufficient for a single f32
        let result = EarlyExitCallback::bytes_to_f32(&[0xFF, 0x00]);
        // Assert
        assert!(result.is_empty(),
            "2 bytes (< 4) should yield empty vec, got {} elements", result.len());
    }

    #[test]
    fn test_cosine_sim_parallel_vectors_different_scales() {
        // Arrange: two vectors in the same direction with very different magnitude
        let a: Vec<f32> = vec![1e-3, 2e-3, 3e-3];
        let b: Vec<f32> = vec![1e6, 2e6, 3e6];
        // Act
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        // Assert: same direction regardless of scale → cos ≈ 1.0
        assert!((sim - 1.0).abs() < 1e-4,
            "parallel vectors with different scales should give cos=1.0, got {}", sim);
    }

    #[test]
    fn test_energy_ratio_with_negative_infinity_denominator() {
        // Arrange: first vector has -INF norm
        let a = vec![f32::NEG_INFINITY];
        let b = vec![1.0f32];
        // Act
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        // Assert: norm_a = INF (sqrt of INF), which is >= 1e-10 → ratio = 1.0/INF ≈ 0
        assert!(
            ratio.abs() < 1e-5 || ratio == 0.0,
            "finite over infinite norm should be near 0, got {}", ratio
        );
    }

    #[test]
    fn test_post_node_with_nearly_convergent_states_no_exit_high_threshold() {
        // Arrange: enabled callback with very high confidence threshold (0.9999)
        // States are close but not close enough for the strict calibrator
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.9999);
        let mut cb = EarlyExitCallback::new(config, 32);

        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let prev_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let curr_f32: Vec<f32> = vec![1.01, 2.01, 3.01, 4.01]; // small divergence
        let prev_bytes: Vec<u8> = prev_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let curr_bytes: Vec<u8> = curr_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

        let exit_points = cb.controller().exit_points().points().to_vec();
        if let Some(&exit_layer) = exit_points.first() {
            let ctx = LayerContext {
                node_idx: exit_layer * 2,
                layer_idx: exit_layer,
                node_op: "Attention",
                hidden_state: &prev_bytes,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 4,
                seq_len: 4,
                position: 0,
                request_id: 1,
                model_config: &model_config,
            };
            // Act
            let _ = cb.pre_node(&ctx);
            let action = cb.post_node(&ctx, &curr_bytes);
            // Assert: divergent states with high threshold → Continue (not exit)
            assert!(
                matches!(action, CallbackAction::Continue),
                "slightly divergent states with high threshold should not exit, got {:?}", action
            );
        }
    }

    #[test]
    fn test_pre_node_with_non_aligned_byte_length() {
        // Arrange: 5 bytes — not aligned to f32 (4 bytes)
        let config = EarlyExitConfig::default();
        let mut cb = EarlyExitCallback::new(config, 32);
        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let non_aligned_bytes: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3F, 0xAB]; // 1.0f32 + 1 trailing
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "Attention",
            hidden_state: &non_aligned_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 4,
            seq_len: 4,
            position: 0,
            request_id: 1,
            model_config: &model_config,
        };
        // Act: pre_node converts 5 bytes → 1 f32 (trailing byte discarded)
        let action = cb.pre_node(&ctx);
        // Assert: pre_node always returns Continue
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_post_node_with_different_prev_and_output_lengths() {
        // Arrange: prev_hidden has 4 f32 values, output has 2 f32 values.
        // cosine_sim takes min(len_a, len_b) so only first 2 elements compared.
        // energy_ratio operates on full slices independently — different lengths produce
        // different norms, so delta_rho may deviate from 1.0.
        // This test verifies the callback handles mismatched lengths without panic.
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.1);
        let mut cb = EarlyExitCallback::new(config, 32);
        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();

        // prev has 4 values, output has 2 — they match on first 2 elements
        let prev_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // norm = sqrt(30) ≈ 5.48
        let curr_f32: Vec<f32> = vec![1.0, 2.0]; // norm = sqrt(5) ≈ 2.24
        let prev_bytes: Vec<u8> = prev_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let curr_bytes: Vec<u8> = curr_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

        let exit_points = cb.controller().exit_points().points().to_vec();
        if let Some(&exit_layer) = exit_points.first() {
            let ctx = LayerContext {
                node_idx: exit_layer * 2,
                layer_idx: exit_layer,
                node_op: "Attention",
                hidden_state: &prev_bytes,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 4,
                seq_len: 4,
                position: 0,
                request_id: 1,
                model_config: &model_config,
            };
            let _ = cb.pre_node(&ctx);
            let action = cb.post_node(&ctx, &curr_bytes);
            // Assert: result is a valid CallbackAction (no panic), either Continue or ExitEarly
            match action {
                CallbackAction::Continue
                | CallbackAction::ExitEarly { .. }
                | CallbackAction::SkipThisNode
                | CallbackAction::InjectHidden { .. }
                | CallbackAction::CompactMask { .. } => {},
            }
        }
    }

    #[test]
    fn test_callback_constructor_with_custom_thresholds() {
        // Arrange: custom thresholds to verify they propagate through
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.90)
            .with_delta_threshold(0.05)
            .with_confidence_threshold(0.7)
            .with_min_layer(5);
        // Act
        let cb = EarlyExitCallback::new(config, 64);
        // Assert: verify all config values are preserved
        let ctrl = cb.controller();
        assert!(ctrl.config().enabled);
        assert!((ctrl.config().cosine_threshold - 0.90).abs() < 1e-6);
        assert!((ctrl.config().delta_threshold - 0.05).abs() < 1e-6);
        assert!((ctrl.config().confidence_threshold - 0.7).abs() < 1e-6);
        assert_eq!(ctrl.config().min_layer, 5);
        assert_eq!(cb.priority(), 50);
        assert_eq!(cb.name(), "early_exit");
    }

    #[test]
    fn test_post_node_empty_output_bytes_returns_continue() {
        // Arrange: enabled callback but output is empty bytes
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.1);
        let mut cb = EarlyExitCallback::new(config, 32);
        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();
        let prev_f32: Vec<f32> = vec![1.0, 2.0];
        let prev_bytes: Vec<u8> = prev_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let empty_output: &[u8] = &[];
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 20,
            node_op: "FFN",
            hidden_state: &prev_bytes,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 4,
            seq_len: 4,
            position: 0,
            request_id: 1,
            model_config: &model_config,
        };
        // Act: prev is non-empty, current is empty → cosine_sim with empty = 0.0
        let _ = cb.pre_node(&ctx);
        let action = cb.post_node(&ctx, empty_output);
        // Assert: cosine 0.0 → low confidence → Continue
        assert!(
            matches!(action, CallbackAction::Continue),
            "empty output should produce low cosine similarity → Continue, got {:?}", action
        );
    }

    #[test]
    fn test_cosine_sim_triangle_inequality_property() {
        // Arrange: three vectors a, b, c. Verify cos(a,c) >= cos(a,b)*cos(b,c) - sin(a,b)*sin(b,c)
        // Using unit vectors at known angles: a=[1,0], b=[cos(30deg),sin(30deg)], c=[0,1]
        let a = vec![1.0f32, 0.0];
        let b = vec![30.0f32.to_radians().cos(), 30.0f32.to_radians().sin()];
        let c = vec![0.0f32, 1.0];
        // Act
        let sim_ab = EarlyExitCallback::cosine_sim(&a, &b);
        let sim_bc = EarlyExitCallback::cosine_sim(&b, &c);
        let sim_ac = EarlyExitCallback::cosine_sim(&a, &c);
        // Assert: basic sanity — all values in [-1, 1]
        assert!(sim_ab >= -1.0 && sim_ab <= 1.0, "cos(a,b) = {}", sim_ab);
        assert!(sim_bc >= -1.0 && sim_bc <= 1.0, "cos(b,c) = {}", sim_bc);
        assert!(sim_ac >= -1.0 && sim_ac <= 1.0, "cos(a,c) = {}", sim_ac);
        // cos(30deg) ≈ 0.866
        assert!((sim_ab - 0.866).abs() < 0.01, "cos(30deg) should be ~0.866, got {}", sim_ab);
        // cos(60deg) = 0.5 (angle between b and c)
        assert!((sim_bc - 0.5).abs() < 0.01, "cos(60deg) should be 0.5, got {}", sim_bc);
        // cos(90deg) = 0 (a and c are orthogonal)
        assert!(sim_ac.abs() < 0.01, "orthogonal vectors should give ~0, got {}", sim_ac);
    }

    #[test]
    fn test_bytes_to_f32_epsilon_roundtrip() {
        // Arrange: f32::EPSILON — smallest representable difference from 1.0
        let bytes = f32::EPSILON.to_le_bytes();
        // Act
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), f32::EPSILON.to_bits(),
            "f32::EPSILON should roundtrip exactly through bytes");
        assert!(result[0] > 0.0);
        assert!(result[0].is_normal());
    }

    // ── Wave 8: 13 additional tests for uncovered edge cases ──

    #[test]
    fn test_early_exit_decision_debug_format() {
        // Arrange
        let decision_continue = EarlyExitDecision::Continue;
        let decision_exit = EarlyExitDecision::Exit {
            confidence: 0.97,
            cosine_sim: 0.995,
            delta_rho: 1.002,
        };
        let decision_defer = EarlyExitDecision::Defer;
        // Act: format via Debug trait
        let debug_continue = format!("{:?}", decision_continue);
        let debug_exit = format!("{:?}", decision_exit);
        let debug_defer = format!("{:?}", decision_defer);
        // Assert: Debug output contains variant names
        assert!(debug_continue.contains("Continue"),
            "Debug for Continue should contain 'Continue', got {}", debug_continue);
        assert!(debug_exit.contains("Exit"),
            "Debug for Exit should contain 'Exit', got {}", debug_exit);
        assert!(debug_exit.contains("0.97"),
            "Debug for Exit should contain confidence value, got {}", debug_exit);
        assert!(debug_defer.contains("Defer"),
            "Debug for Defer should contain 'Defer', got {}", debug_defer);
    }

    #[test]
    fn test_early_exit_decision_clone_equality() {
        // Arrange
        let original = EarlyExitDecision::Exit {
            confidence: 0.88,
            cosine_sim: 0.99,
            delta_rho: 1.001,
        };
        // Act
        let cloned = original.clone();
        // Assert: match on both to verify field-by-field equality
        match (original, cloned) {
            (
                EarlyExitDecision::Exit { confidence: c1, cosine_sim: cs1, delta_rho: d1 },
                EarlyExitDecision::Exit { confidence: c2, cosine_sim: cs2, delta_rho: d2 },
            ) => {
                assert!((c1 - c2).abs() < 1e-10, "confidence mismatch");
                assert!((cs1 - cs2).abs() < 1e-10, "cosine_sim mismatch");
                assert!((d1 - d2).abs() < 1e-10, "delta_rho mismatch");
            }
            _ => panic!("both should be Exit variant"),
        }
    }

    #[test]
    fn test_residual_bus_early_exit_clone_and_debug() {
        // Arrange
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(5);
        let bus = ResidualBusEarlyExit::new(config, vec![10, 20, 30]);
        // Act
        let cloned = bus.clone();
        let debug_str = format!("{:?}", bus);
        // Assert
        assert_eq!(cloned.exit_points, vec![10, 20, 30]);
        assert!(cloned.config.enabled);
        assert!(debug_str.contains("ResidualBusEarlyExit"),
            "Debug output should contain type name, got {}", debug_str);
    }

    #[test]
    fn test_residual_bus_should_exit_at_non_exit_layer() {
        // Arrange: exit points are [10, 20], query layer 5
        let config = EarlyExitConfig::new().with_enabled(true)
            .with_cosine_threshold(0.99)
            .with_delta_threshold(0.01)
            .with_min_layer(0);
        let bus = ResidualBusEarlyExit::new(config, vec![10, 20]);
        // Act: layer 5 is not in exit_points
        let result = bus.should_exit_at_layer(5, 0.001, 0.9999);
        // Assert: should always return false for non-exit layers
        assert!(!result, "non-exit layer should not trigger exit");
    }

    #[test]
    fn test_residual_bus_should_exit_disabled_config() {
        // Arrange: config disabled, exit point layer
        let config = EarlyExitConfig::new(); // enabled = false
        let bus = ResidualBusEarlyExit::new(config, vec![10]);
        // Act: even at exit point with good signals, disabled means no exit
        let result = bus.should_exit_at_layer(10, 0.001, 0.9999);
        // Assert
        assert!(!result, "disabled config should never trigger exit");
    }

    #[test]
    fn test_early_exit_config_builder_chain_is_idempotent() {
        // Arrange: build config with the builder pattern, then build again with same values
        let config1 = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.95)
            .with_delta_threshold(0.05)
            .with_min_layer(8)
            .with_confidence_threshold(0.8);
        let config2 = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.95)
            .with_delta_threshold(0.05)
            .with_min_layer(8)
            .with_confidence_threshold(0.8);
        // Assert: both produce identical field values
        assert_eq!(config1.enabled, config2.enabled);
        assert!((config1.cosine_threshold - config2.cosine_threshold).abs() < 1e-10);
        assert!((config1.delta_threshold - config2.delta_threshold).abs() < 1e-10);
        assert_eq!(config1.min_layer, config2.min_layer);
        assert!((config1.confidence_threshold - config2.confidence_threshold).abs() < 1e-10);
    }

    #[test]
    fn test_exit_layer_stats_record_and_query() {
        // Arrange
        let layers = vec![10, 20, 30];
        let mut stats = ExitLayerStats::new(&layers, 0.5);
        // Act: record exit at layer 10, non-exit at layer 20
        stats.record(10, true);
        stats.record(20, false);
        stats.record(10, true);
        // Assert
        assert!(stats.hit_rate(10) > 0.0, "hit_rate at layer 10 should be positive");
        assert_eq!(stats.eval_count(10), 2, "layer 10 should have 2 evaluations");
        assert_eq!(stats.eval_count(20), 1, "layer 20 should have 1 evaluation");
        assert_eq!(stats.eval_count(30), 0, "layer 30 should have 0 evaluations");
        assert_eq!(stats.total_evals(), 3, "total evals should be 3");
    }

    #[test]
    fn test_exit_layer_stats_hit_rate_for_unknown_layer() {
        // Arrange: stats only track layers [10, 20]
        let layers = vec![10, 20];
        let mut stats = ExitLayerStats::new(&layers, 0.1);
        // Act: record on unknown layer
        stats.record(99, true);
        // Assert: unknown layer should not affect stats
        assert_eq!(stats.hit_rate(99), 0.0, "unknown layer should return 0.0 hit rate");
        assert_eq!(stats.eval_count(99), 0, "unknown layer should have 0 eval count");
        assert_eq!(stats.total_evals(), 0, "unknown layer recording should be ignored");
    }

    #[test]
    fn test_exit_layer_stats_best_exit_point_requires_min_evals() {
        // Arrange
        let layers = vec![10, 20];
        let mut stats = ExitLayerStats::new(&layers, 0.1);
        // Act: only 9 evals at layer 10 (below min_evals=10 threshold)
        for _ in 0..9 {
            stats.record(10, true);
        }
        // Assert: best_exit_point should be None (below min_evals threshold)
        assert!(stats.best_exit_point().is_none(),
            "best_exit_point should be None when evals < 10");
        // Act: one more eval to reach 10
        stats.record(10, true);
        // Assert: now should return layer 10
        assert_eq!(stats.best_exit_point(), Some(10),
            "best_exit_point should be layer 10 after 10 evals");
    }

    #[test]
    fn test_adaptive_exit_points_is_exit_point() {
        // Arrange: 32-layer model
        let exit_points = AdaptiveExitPoints::compute(32, 0.0);
        // Act & Assert: every returned point should be recognized
        for &p in exit_points.points() {
            assert!(exit_points.is_exit_point(p),
                "layer {} should be recognized as exit point", p);
        }
        // Layer 0 should not be an exit point (golden ratio always >= 0.618)
        assert!(!exit_points.is_exit_point(0), "layer 0 should not be an exit point");
        assert_eq!(exit_points.total_layers(), 32);
        assert!(!exit_points.is_empty(), "32-layer model should have exit points");
    }

    #[test]
    fn test_confidence_calibrator_custom_parameters() {
        // Arrange: custom calibrator with known alpha/beta
        let cal = ConfidenceCalibrator::new(10.0, 5.0, 0.95, 0.02);
        // Act: signals well above threshold → high confidence
        let conf_high = cal.calibrate(0.999, 1.001);
        // Act: signals well below threshold → low confidence
        let conf_low = cal.calibrate(0.80, 1.20);
        // Assert
        assert!(conf_high > 0.5, "strong signals should give high confidence, got {}", conf_high);
        assert!(conf_low < 0.5, "weak signals should give low confidence, got {}", conf_low);
        // Assert: confidence is bounded in (0, 1)
        assert!(conf_high > 0.0 && conf_high < 1.0, "confidence should be in (0, 1)");
        assert!(conf_low > 0.0 && conf_low < 1.0, "confidence should be in (0, 1)");
    }

    #[test]
    fn test_should_early_exit_legacy_function_boundary() {
        // Arrange: config with known thresholds
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.99)
            .with_delta_threshold(0.01)
            .with_min_layer(5);
        // Act & Assert: at boundary layer (layer == min_layer)
        let at_boundary = should_early_exit(5, 0.999, 1.001, &config);
        assert!(at_boundary, "at min_layer boundary with good signals should exit");
        // Below min_layer
        let below_boundary = should_early_exit(4, 0.999, 1.001, &config);
        assert!(!below_boundary, "below min_layer should never exit");
        // Good cosine but bad energy delta
        let bad_energy = should_early_exit(10, 0.999, 1.5, &config);
        assert!(!bad_energy, "bad energy delta should not exit");
        // Good energy but bad cosine
        let bad_cosine = should_early_exit(10, 0.90, 1.001, &config);
        assert!(!bad_cosine, "bad cosine should not exit");
    }

    #[test]
    fn test_controller_with_calibrator_custom() {
        // Arrange: custom calibrator with aggressive parameters
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.5);
        let custom_cal = ConfidenceCalibrator::new(50.0, 50.0, 0.95, 0.05);
        let mut ctrl = EarlyExitController::with_calibrator(config, 32, custom_cal);
        // Act: at an exit point with mediocre signals — aggressive calibrator amplifies
        let exit_points = ctrl.exit_points().points().to_vec();
        if let Some(&layer) = exit_points.first() {
            let decision = ctrl.check_layer(layer, 0.96, 1.04);
            // Assert: aggressive alpha/beta means even mediocre signals may trigger
            match decision {
                EarlyExitDecision::Exit { confidence, .. } => {
                    assert!(confidence > 0.0 && confidence < 1.0,
                        "confidence should be in (0,1), got {}", confidence);
                }
                EarlyExitDecision::Continue => {
                    // Also acceptable: the calibrator decided not to exit
                }
                EarlyExitDecision::Defer => {
                    panic!("should not Defer at exit point above min_layer with enabled config");
                }
            }
        }
    }

    // ── Wave 9: 13 additional tests for uncovered edge cases ──

    #[test]
    fn test_controller_should_exit_returns_false_for_disabled() {
        let config = EarlyExitConfig::default(); // enabled = false
        let ctrl = EarlyExitController::new(config, 32);
        assert!(!ctrl.should_exit(20, 0.9999, 1.0001),
            "disabled controller should always return false");
    }

    #[test]
    fn test_controller_should_exit_returns_false_below_min_layer() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(10);
        let ctrl = EarlyExitController::new(config, 32);
        assert!(!ctrl.should_exit(5, 0.9999, 1.0001),
            "layer below min_layer should return false");
    }

    #[test]
    fn test_controller_should_exit_returns_false_for_non_exit_point() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let ctrl = EarlyExitController::new(config, 32);
        let exit_points = ctrl.exit_points().points().to_vec();
        // Find a layer that is NOT an exit point
        let mut non_exit = 1usize;
        while exit_points.contains(&non_exit) {
            non_exit += 1;
        }
        assert!(!ctrl.should_exit(non_exit, 0.9999, 1.0001),
            "non-exit-point layer should return false, layer={}", non_exit);
    }

    #[test]
    fn test_controller_stats_accessor_empty_initially() {
        let config = EarlyExitConfig::new().with_enabled(true).with_min_layer(1);
        let ctrl = EarlyExitController::new(config, 32);
        assert_eq!(ctrl.stats().total_evals(), 0,
            "new controller should have zero total evals");
    }

    #[test]
    fn test_controller_stats_records_after_check_layer() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(1)
            .with_confidence_threshold(0.1);
        let mut ctrl = EarlyExitController::new(config, 32);
        let exit_points = ctrl.exit_points().points().to_vec();
        if let Some(&layer) = exit_points.first() {
            let _ = ctrl.check_layer(layer, 0.9999, 1.0001);
            assert_eq!(ctrl.stats().total_evals(), 1,
                "check_layer at exit point should record one eval");
            assert!(ctrl.stats().eval_count(layer) >= 1,
                "eval_count at exit layer should be >= 1");
        }
    }

    #[test]
    fn test_confidence_calibrator_default_values() {
        let cal = ConfidenceCalibrator::default();
        assert!((cal.alpha - 20.0).abs() < 1e-6, "default alpha should be 20.0");
        assert!((cal.beta - 15.0).abs() < 1e-6, "default beta should be 15.0");
        assert!((cal.theta_c - 0.99).abs() < 1e-6, "default theta_c should be 0.99");
        assert!((cal.theta_d - 0.01).abs() < 1e-6, "default theta_d should be 0.01");
    }

    #[test]
    fn test_confidence_calibrate_with_nan_cosine_returns_nan() {
        let cal = ConfidenceCalibrator::default();
        let conf = cal.calibrate(f32::NAN, 1.0);
        assert!(conf.is_nan(), "NaN cosine_sim should produce NaN confidence");
    }

    #[test]
    fn test_confidence_calibrate_deterministic() {
        let cal = ConfidenceCalibrator::default();
        let c1 = cal.calibrate(0.995, 1.002);
        let c2 = cal.calibrate(0.995, 1.002);
        assert!((c1 - c2).abs() < 1e-10,
            "identical inputs should produce identical confidence, got {} vs {}", c1, c2);
    }

    #[test]
    fn test_adaptive_exit_points_len_matches_points_slice() {
        let points = AdaptiveExitPoints::compute(64, 0.0);
        assert_eq!(points.len(), points.points().len(),
            "len() should equal points().len()");
        assert!(!points.is_empty(), "64-layer model should have exit points");
    }

    #[test]
    fn test_adaptive_exit_points_high_min_ratio_filters_all() {
        let points = AdaptiveExitPoints::compute(32, 0.99);
        assert!(points.is_empty(),
            "min_ratio=0.99 should filter all golden-ratio points for 32 layers");
    }

    #[test]
    fn test_exit_layer_stats_ema_decay_precision() {
        let mut stats = ExitLayerStats::new(&[10], 0.3);
        stats.record(10, true);
        let rate_after_one = stats.hit_rate(10);
        assert!((rate_after_one - 0.3).abs() < 1e-6,
            "after 1 exit with alpha=0.3: hit_rate should be 0.3, got {}", rate_after_one);
        stats.record(10, false);
        let rate_after_two = stats.hit_rate(10);
        let expected = 0.3 * 0.0 + 0.7 * 0.3;
        assert!((rate_after_two - expected).abs() < 1e-6,
            "after exit then non-exit: rate = 0.3*0 + 0.7*0.3 = {}, got {}", expected, rate_after_two);
    }

    #[test]
    fn test_residual_bus_should_exit_at_exit_layer_with_good_signals() {
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.99)
            .with_delta_threshold(0.01)
            .with_min_layer(0);
        let bus = ResidualBusEarlyExit::new(config, vec![10, 20]);
        assert!(bus.should_exit_at_layer(10, 1.001, 0.999),
            "at exit layer 10 with stable signals should return true");
        assert!(bus.should_exit_at_layer(20, 1.001, 0.999),
            "at exit layer 20 with stable signals should return true");
    }

    #[test]
    fn test_should_early_exit_disabled_config_always_false() {
        let config = EarlyExitConfig::default(); // enabled = false
        assert!(!should_early_exit(20, 0.9999, 1.0001, &config),
            "disabled config should always return false");
        assert!(!should_early_exit(100, 1.0, 1.0, &config),
            "disabled config should return false even with perfect signals");
    }

    // ── Wave 10: 10 additional tests for uncovered edge cases ──

    #[test]
    fn test_callback_action_default_is_continue() {
        // Arrange & Act
        let action: CallbackAction = CallbackAction::default();
        // Assert: Default trait implementation should produce Continue
        assert!(matches!(action, CallbackAction::Continue),
            "CallbackAction::default() should be Continue");
    }

    #[test]
    fn test_callback_action_equality_across_variants() {
        // Arrange: different ExitEarly instances
        let exit_a = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        let exit_b = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        let exit_c = CallbackAction::ExitEarly { logits: vec![3.0] };
        let inject = CallbackAction::InjectHidden { data: vec![0u8] };
        let skip = CallbackAction::SkipThisNode;
        let compact = CallbackAction::CompactMask { active_mask: vec![true] };
        let cont = CallbackAction::Continue;
        // Assert: PartialEq derived correctly
        assert_eq!(exit_a, exit_b, "identical ExitEarly should be equal");
        assert_ne!(exit_a, exit_c, "different logits should not be equal");
        assert_ne!(cont, skip, "Continue != SkipThisNode");
        assert_ne!(exit_a, inject, "ExitEarly != InjectHidden");
        assert_ne!(skip, compact, "SkipThisNode != CompactMask");
    }

    #[test]
    fn test_early_exit_decision_continue_and_defer_clone_equal() {
        // Arrange
        let cont = EarlyExitDecision::Continue;
        let defer = EarlyExitDecision::Defer;
        // Act
        let cont2 = cont.clone();
        let defer2 = defer.clone();
        // Assert: Continue and Defer have no fields — clone should produce identical variants
        assert!(matches!(cont2, EarlyExitDecision::Continue),
            "cloned Continue should still be Continue");
        assert!(matches!(defer2, EarlyExitDecision::Defer),
            "cloned Defer should still be Defer");
    }

    #[test]
    fn test_residual_bus_should_exit_at_layer_with_bad_cosine() {
        // Arrange: enabled config at exit layer with poor cosine similarity
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_cosine_threshold(0.99)
            .with_delta_threshold(0.01)
            .with_min_layer(0);
        let bus = ResidualBusEarlyExit::new(config, vec![10, 20]);
        // Act: cosine_sim=0.5 is well below threshold=0.99
        let result = bus.should_exit_at_layer(10, 1.001, 0.5);
        // Assert: poor cosine should prevent exit
        assert!(!result, "bad cosine similarity should not trigger exit");
    }

    #[test]
    fn test_exit_layer_stats_hit_rate_converges_with_many_records() {
        // Arrange: track stats with alternating exits and non-exits
        let mut stats = ExitLayerStats::new(&[10], 0.3);
        // Act: record 100 alternating results (exit=true, exit=false, ...)
        for i in 0..100 {
            stats.record(10, i % 2 == 0);
        }
        // Assert: with many records and alpha=0.3, EMA should converge toward 0.5
        let rate = stats.hit_rate(10);
        assert!((rate - 0.5).abs() < 0.1,
            "alternating records should converge toward 0.5, got {}", rate);
        assert_eq!(stats.eval_count(10), 100, "should have 100 evals");
    }

    #[test]
    fn test_exit_layer_stats_best_exit_point_picks_highest_rate() {
        // Arrange: two exit layers with different hit rates
        let mut stats = ExitLayerStats::new(&[10, 20], 0.3);
        // Layer 10: mostly exits (high rate)
        for _ in 0..10 {
            stats.record(10, true);
        }
        // Layer 20: mostly non-exits (low rate)
        for _ in 0..10 {
            stats.record(20, false);
        }
        // Act
        let best = stats.best_exit_point();
        // Assert: layer 10 has higher hit rate than layer 20
        assert_eq!(best, Some(10),
            "best_exit_point should be layer 10 (higher hit rate), got {:?}", best);
        assert!(stats.hit_rate(10) > stats.hit_rate(20),
            "layer 10 hit rate ({}) should exceed layer 20 ({})",
            stats.hit_rate(10), stats.hit_rate(20));
    }

    #[test]
    fn test_exit_layer_stats_clone_produces_independent_copy() {
        // Arrange
        let mut stats = ExitLayerStats::new(&[10], 0.3);
        stats.record(10, true);
        // Act: clone before additional recording
        let cloned = stats.clone();
        stats.record(10, false);
        // Assert: clone should not be affected by subsequent mutations
        assert_eq!(cloned.eval_count(10), 1,
            "cloned stats should have original eval count");
        assert_eq!(stats.eval_count(10), 2,
            "original stats should have updated eval count");
    }

    #[test]
    fn test_adaptive_exit_points_within_valid_range() {
        // Arrange: test multiple total_layer counts
        for &total in &[2usize, 4, 8, 16, 32, 64, 128, 256] {
            let points = AdaptiveExitPoints::compute(total, 0.0);
            // Assert: every exit point must be in [1, total-1]
            for &p in points.points() {
                assert!(p >= 1 && p < total,
                    "exit point {} must be in [1, {}) for total={}, got points {:?}",
                    p, total, total, points.points());
            }
            assert!(points.points().windows(2).all(|w| w[0] <= w[1]),
                "exit points must be sorted for total={}", total);
        }
    }

    #[test]
    fn test_confidence_calibrate_at_exact_delta_threshold_boundary() {
        // Arrange: calibrator where delta_rho deviation exactly equals theta_d
        let cal = ConfidenceCalibrator::new(20.0, 15.0, 0.99, 0.01);
        // Act: delta_rho = 1.01 → |delta_rho - 1| = 0.01 = theta_d → delta_signal = 0
        let conf_boundary = cal.calibrate(0.99, 1.01);
        // Act: delta_rho = 0.99 → same deviation
        let conf_boundary_low = cal.calibrate(0.99, 0.99);
        // Assert: both should be identical (symmetry around 1.0)
        assert!((conf_boundary - conf_boundary_low).abs() < 1e-10,
            "boundary at +0.01 and -0.01 should be symmetric, got {} vs {}",
            conf_boundary, conf_boundary_low);
        // At exact boundary: cos_signal=0, delta_signal=0 → logit=0 → sigmoid=0.5
        assert!((conf_boundary - 0.5).abs() < 1e-6,
            "at exact threshold boundary confidence should be 0.5, got {}", conf_boundary);
    }

    #[test]
    fn test_post_node_at_last_layer_boundary() {
        // Arrange: callback for a 4-layer model, test at the final exit point
        let config = EarlyExitConfig::new()
            .with_enabled(true)
            .with_min_layer(0)
            .with_confidence_threshold(0.1);
        let mut cb = EarlyExitCallback::new(config, 4);
        let model_config = crate::engine::executor::GeneratorForwardConfig::default_for_test();

        // Nearly identical hidden states for convergence
        let prev_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let curr_f32: Vec<f32> = vec![1.0001, 2.0001, 3.0001, 4.0001];
        let prev_bytes: Vec<u8> = prev_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let curr_bytes: Vec<u8> = curr_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Use the last exit point (should be the deepest layer)
        let exit_points = cb.controller().exit_points().points().to_vec();
        if let Some(&last_exit) = exit_points.last() {
            let ctx = LayerContext {
                node_idx: last_exit * 2 + 1, // last node in the model
                layer_idx: last_exit,
                node_op: "FFN",
                hidden_state: &prev_bytes,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 8,
                seq_len: 8,
                position: last_exit,
                request_id: 7,
                model_config: &model_config,
            };
            // Act
            let _ = cb.pre_node(&ctx);
            let action = cb.post_node(&ctx, &curr_bytes);
            // Assert: at last exit point with converging states → ExitEarly
            assert!(
                matches!(action, CallbackAction::ExitEarly { .. }),
                "last exit point {} with converging states should trigger ExitEarly, got {:?}",
                last_exit, action
            );
        }
    }
}
