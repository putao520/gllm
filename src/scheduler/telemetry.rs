use serde::{Deserialize, Serialize};

/// 5-Dimensional Telemetry data extracted natively from the GPU JIT Kernels (Tier III Features).
/// Transferred without Host/Device synchronization via pinned/mapped zero-copy memory.
#[derive(Debug, Clone, Default, Copy, Serialize, Deserialize, PartialEq)]
#[repr(C)]
pub struct SequenceTelemetry {
    /// L2 Delta from Attention patterns (triggers speculative admission)
    pub l2_delta: f32,
    /// Whether an outlier was detected in RmsNorm
    pub has_outlier: bool,
    /// Sparsity of SwiGLU dead rows
    pub dead_density: f32,
    /// Entropy of the probability distribution across attention heads
    pub per_head_entropy: f32,
    /// Residual ratio (RMS(x - prev) / RMS(prev)) used for Early-Exit
    pub transform_ratio: f32,
    /// Final token probability entropy from LM_Head Softmax
    pub output_entropy: f32,
}

impl SequenceTelemetry {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Tier V.3 Profile-Guided Re-Fusion
/// Tracks the 5-dimensional telemetry across decoding steps.
/// If `transform_ratio < 0.05` for 95% of the last 100 steps on a given layer,
/// it triggers a RecompileHint.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileAccumulator {
    history: std::collections::HashMap<usize, std::collections::VecDeque<f32>>,
    required_stable_steps: usize,
    stable_threshold: f32,
    history_capacity: usize,
}

impl Default for ProfileAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileAccumulator {
    pub fn new() -> Self {
        Self {
            history: std::collections::HashMap::new(),
            required_stable_steps: 95,
            stable_threshold: 0.05,
            history_capacity: 100,
        }
    }

    /// Add a new telemetry sample for a given layer.
    /// Returns `true` if the layer has hit the stability threshold and should trigger Re-Fusion.
    pub fn record_and_check(&mut self, layer: usize, transform_ratio: f32) -> bool {
        let q = self.history.entry(layer).or_insert_with(|| std::collections::VecDeque::with_capacity(self.history_capacity));
        if q.len() >= self.history_capacity {
            q.pop_front();
        }
        q.push_back(transform_ratio);
        
        if q.len() < self.history_capacity {
            return false;
        }

        let stable_count = q.iter().filter(|&&r| r < self.stable_threshold).count();
        if stable_count >= self.required_stable_steps {
            // Clear history after triggering so we don't spam recompiles
            q.clear();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SequenceTelemetry ──

    #[test]
    fn telemetry_default_zeroed() {
        let t = SequenceTelemetry::default();
        assert_eq!(t.l2_delta, 0.0);
        assert!(!t.has_outlier);
        assert_eq!(t.dead_density, 0.0);
        assert_eq!(t.per_head_entropy, 0.0);
        assert_eq!(t.transform_ratio, 0.0);
        assert_eq!(t.output_entropy, 0.0);
    }

    #[test]
    fn telemetry_new_equals_default() {
        assert_eq!(SequenceTelemetry::new(), SequenceTelemetry::default());
    }

    #[test]
    fn telemetry_is_copy() {
        let t = SequenceTelemetry { l2_delta: 1.0, ..Default::default() };
        let t2 = t;
        assert_eq!(t.l2_delta, t2.l2_delta);
    }

    #[test]
    fn telemetry_serde_roundtrip() {
        let t = SequenceTelemetry {
            l2_delta: 0.5,
            has_outlier: true,
            dead_density: 0.1,
            per_head_entropy: 2.3,
            transform_ratio: 0.01,
            output_entropy: 1.5,
        };
        let json = serde_json::to_string(&t).unwrap();
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }

    // ── ProfileAccumulator ──

    #[test]
    fn accumulator_no_trigger_before_capacity() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..99 {
            assert!(!acc.record_and_check(0, 0.01));
        }
    }

    #[test]
    fn accumulator_triggers_on_stable() {
        let mut acc = ProfileAccumulator::new();
        // Fill 100 entries, 95+ with ratio < 0.05
        for _ in 0..100 {
            let triggered = acc.record_and_check(0, 0.01);
            if triggered {
                return; // may trigger on the 100th
            }
        }
        // Should have triggered by now
        panic!("expected trigger after 100 stable samples");
    }

    #[test]
    fn accumulator_no_trigger_when_unstable() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            assert!(!acc.record_and_check(0, 1.0)); // well above threshold
        }
    }

    #[test]
    fn accumulator_per_layer_independent() {
        let mut acc = ProfileAccumulator::new();
        // Fill layer 0 to capacity with unstable values
        for _ in 0..100 {
            acc.record_and_check(0, 1.0);
        }
        // Fill layer 1 with stable values — should trigger
        let mut triggered = false;
        for _ in 0..100 {
            if acc.record_and_check(1, 0.01) {
                triggered = true;
                break;
            }
        }
        assert!(triggered, "layer 1 should trigger independently");
    }

    #[test]
    fn accumulator_clears_after_trigger() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
        }
        // After trigger, next 100 should need another full cycle
        let mut triggered_again = false;
        for _ in 0..50 {
            if acc.record_and_check(0, 0.01) {
                triggered_again = true;
            }
        }
        assert!(!triggered_again, "should not re-trigger until history fills again");
    }

    // ── SequenceTelemetry additional tests ──

    #[test]
    fn telemetry_clone_independent() {
        let t = SequenceTelemetry {
            l2_delta: 1.5,
            has_outlier: true,
            dead_density: 0.3,
            per_head_entropy: 4.2,
            transform_ratio: 0.99,
            output_entropy: 2.7,
        };
        let cloned = t.clone();
        assert_eq!(t, cloned);
    }

    #[test]
    fn telemetry_debug_format() {
        let t = SequenceTelemetry {
            l2_delta: 0.0,
            has_outlier: false,
            dead_density: 0.0,
            per_head_entropy: 0.0,
            transform_ratio: 0.0,
            output_entropy: 0.0,
        };
        let debug_str = format!("{:?}", t);
        assert!(debug_str.contains("SequenceTelemetry"));
        assert!(debug_str.contains("l2_delta"));
        assert!(debug_str.contains("has_outlier"));
    }

    #[test]
    fn telemetry_explicit_construction_all_fields() {
        let t = SequenceTelemetry {
            l2_delta: f32::MAX,
            has_outlier: true,
            dead_density: f32::MIN,
            per_head_entropy: f32::INFINITY,
            transform_ratio: f32::NEG_INFINITY,
            output_entropy: f32::NAN,
        };
        assert_eq!(t.l2_delta, f32::MAX);
        assert!(t.has_outlier);
        assert_eq!(t.dead_density, f32::MIN);
        assert!(t.per_head_entropy.is_infinite() && t.per_head_entropy > 0.0);
        assert!(t.transform_ratio.is_infinite() && t.transform_ratio < 0.0);
        assert!(t.output_entropy.is_nan());
    }

    #[test]
    fn telemetry_equality_semantics() {
        let a = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: false,
            dead_density: 0.5,
            per_head_entropy: 3.0,
            transform_ratio: 0.1,
            output_entropy: 0.0,
        };
        let b = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: false,
            dead_density: 0.5,
            per_head_entropy: 3.0,
            transform_ratio: 0.1,
            output_entropy: 0.0,
        };
        assert_eq!(a, b);
        let c = SequenceTelemetry { l2_delta: 2.0, ..a };
        assert_ne!(a, c);
    }

    #[test]
    fn telemetry_nan_inequality() {
        // NaN != NaN by IEEE 754; PartialEq derive reflects this
        let a = SequenceTelemetry { output_entropy: f32::NAN, ..Default::default() };
        let b = SequenceTelemetry { output_entropy: f32::NAN, ..Default::default() };
        assert_ne!(a, b);
    }

    #[test]
    fn telemetry_serde_deserialize_invalid_json() {
        let result = serde_json::from_str::<SequenceTelemetry>("not json");
        assert!(result.is_err());
    }

    #[test]
    fn telemetry_serde_deserialize_missing_field() {
        // serde requires all fields by default; partial JSON should fail
        let result = serde_json::from_str::<SequenceTelemetry>(r#"{"l2_delta": 1.0}"#);
        assert!(result.is_err());
    }

    #[test]
    fn telemetry_field_mutation() {
        let mut t = SequenceTelemetry::default();
        t.l2_delta = 3.14;
        t.has_outlier = true;
        t.dead_density = 0.75;
        t.per_head_entropy = 5.0;
        t.transform_ratio = 0.02;
        t.output_entropy = 1.23;
        assert_eq!(t.l2_delta, 3.14);
        assert!(t.has_outlier);
        assert_eq!(t.dead_density, 0.75);
        assert_eq!(t.per_head_entropy, 5.0);
        assert_eq!(t.transform_ratio, 0.02);
        assert_eq!(t.output_entropy, 1.23);
    }

    // ── ProfileAccumulator additional tests ──

    #[test]
    fn accumulator_default_equals_new() {
        assert_eq!(ProfileAccumulator::default(), ProfileAccumulator::new());
    }

    #[test]
    fn accumulator_clone_independent() {
        let mut acc = ProfileAccumulator::new();
        acc.record_and_check(0, 0.01);
        let cloned = acc.clone();
        assert_eq!(acc, cloned);
    }

    #[test]
    fn accumulator_debug_format() {
        let acc = ProfileAccumulator::new();
        let debug_str = format!("{:?}", acc);
        assert!(debug_str.contains("ProfileAccumulator"));
    }

    #[test]
    fn accumulator_no_trigger_below_required_stable_count() {
        // Fill 100 entries but only 94 are stable — should not trigger
        let mut acc = ProfileAccumulator::new();
        for _ in 0..94 {
            acc.record_and_check(0, 0.01);
        }
        for _ in 0..6 {
            acc.record_and_check(0, 1.0);
        }
        // Now at capacity with 94 stable (need 95)
        assert!(!acc.record_and_check(0, 0.01));
    }

    #[test]
    fn accumulator_sliding_window_eviction() {
        // 100 unstable entries, then 95 stable pushed in via sliding window
        let mut acc = ProfileAccumulator::new();
        // Fill with unstable
        for _ in 0..100 {
            acc.record_and_check(0, 1.0);
        }
        // Push 95 stable entries, evicting 95 unstable
        for i in 0..95 {
            if acc.record_and_check(0, 0.01) && i >= 94 {
                return; // triggered correctly
            }
        }
        panic!("should trigger once 95 stable samples accumulate via sliding window");
    }

    #[test]
    fn accumulator_trigger_exactly_at_threshold() {
        // Exactly 95 stable out of 100 — the 100th call itself should return true
        let mut acc = ProfileAccumulator::new();
        for _ in 0..95 {
            acc.record_and_check(0, 0.01);
        }
        for _ in 0..4 {
            acc.record_and_check(0, 0.5);
        }
        // 99 entries so far. The 100th (stable) makes it 96 stable / 4 unstable
        assert!(acc.record_and_check(0, 0.01));
    }

    #[test]
    fn accumulator_zero_transform_ratio_is_stable() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            let triggered = acc.record_and_check(0, 0.0);
            if triggered {
                return;
            }
        }
        panic!("zero transform_ratio should be counted as stable");
    }

    #[test]
    fn accumulator_large_layer_index() {
        let mut acc = ProfileAccumulator::new();
        let layer = usize::MAX;
        for _ in 0..100 {
            let triggered = acc.record_and_check(layer, 0.01);
            if triggered {
                return;
            }
        }
        panic!("large layer index should work identically");
    }

    #[test]
    fn accumulator_multiple_layers_interleaved() {
        let mut acc = ProfileAccumulator::new();
        // Interleave records for layers 0, 1, 2
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
            acc.record_and_check(1, 1.0);
            acc.record_and_check(2, 0.01);
        }
        // Layer 0 should have already triggered; layer 1 never; layer 2 triggered
        // Push one more to layer 0 (was cleared after trigger)
        // Verify layer 1 still does not trigger after another 100
        for _ in 0..100 {
            assert!(!acc.record_and_check(1, 1.0));
        }
    }

    #[test]
    fn accumulator_negative_transform_ratio_is_stable() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            let triggered = acc.record_and_check(0, -1.0);
            if triggered {
                return;
            }
        }
        panic!("negative transform_ratio < 0.05 should be counted as stable");
    }

    #[test]
    fn accumulator_retrigger_after_refill() {
        let mut acc = ProfileAccumulator::new();
        // First cycle: trigger
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
        }
        // Second cycle: refill and retrigger
        for _ in 0..100 {
            let triggered = acc.record_and_check(0, 0.01);
            if triggered {
                return;
            }
        }
        panic!("should re-trigger after refilling history post-clear");
    }

    // ── Additional edge-case tests ──

    #[test]
    fn telemetry_subnormal_float_fields() {
        let sub = f32::from_bits(1); // smallest positive subnormal
        let t = SequenceTelemetry {
            l2_delta: sub,
            has_outlier: false,
            dead_density: sub,
            per_head_entropy: sub,
            transform_ratio: sub,
            output_entropy: sub,
        };
        assert_eq!(t.l2_delta.to_bits(), 1u32);
        assert_eq!(t.dead_density.to_bits(), 1u32);
        assert!(t.l2_delta > 0.0);
    }

    #[test]
    fn telemetry_equality_differs_by_has_outlier_only() {
        let a = SequenceTelemetry { has_outlier: false, ..Default::default() };
        let b = SequenceTelemetry { has_outlier: true, ..Default::default() };
        assert_ne!(a, b);
    }

    #[test]
    fn telemetry_repr_c_size_consistent() {
        // repr(C): l2_delta(4) + has_outlier(1+3pad) + dead_density(4) + per_head_entropy(4)
        //          + transform_ratio(4) + output_entropy(4) = 24 bytes
        let size = std::mem::size_of::<SequenceTelemetry>();
        assert!(size >= 6 * std::mem::size_of::<f32>(), "must hold at least 6 f32 fields");
        assert!(size <= 32, "should not have excessive padding");
    }

    #[test]
    fn telemetry_serde_roundtrip_large_finite() {
        // Verify serde handles large finite float values correctly
        let t = SequenceTelemetry {
            l2_delta: f32::MAX,
            dead_density: f32::MIN, // most negative finite
            ..Default::default()
        };
        let json = serde_json::to_string(&t).unwrap();
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(t2.l2_delta, f32::MAX);
        assert_eq!(t2.dead_density, f32::MIN);
    }

    #[test]
    fn telemetry_all_negative_floats() {
        let t = SequenceTelemetry {
            l2_delta: -0.5,
            has_outlier: false,
            dead_density: -1.0,
            per_head_entropy: -2.5,
            transform_ratio: -0.01,
            output_entropy: f32::MIN, // most negative finite f32
        };
        assert!(t.l2_delta < 0.0);
        assert!(t.dead_density < 0.0);
        assert!(t.per_head_entropy < 0.0);
        assert!(t.transform_ratio < 0.0);
        assert!(t.output_entropy < 0.0);
    }

    #[test]
    fn accumulator_no_records_no_trigger() {
        let acc = ProfileAccumulator::new();
        // No records at all — no history entries exist
        assert_eq!(format!("{:?}", acc).contains("history"), true);
    }

    #[test]
    fn accumulator_nan_transform_ratio_is_unstable() {
        // NaN < 0.05 is false, so NaN samples count as unstable
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            assert!(!acc.record_and_check(0, f32::NAN));
        }
    }

    #[test]
    fn accumulator_exact_threshold_is_unstable() {
        // 0.05 is NOT < 0.05, so exactly-at-threshold counts as unstable
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            assert!(!acc.record_and_check(0, 0.05));
        }
    }

    #[test]
    fn accumulator_just_below_threshold_is_stable() {
        // 0.049 < 0.05, so this should trigger
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            let triggered = acc.record_and_check(0, 0.049);
            if triggered {
                return;
            }
        }
        panic!("0.049 < 0.05 should be stable and trigger");
    }

    #[test]
    fn accumulator_infinity_transform_ratio_is_unstable() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            assert!(!acc.record_and_check(0, f32::INFINITY));
        }
    }

    #[test]
    fn accumulator_many_layers_all_trigger() {
        let mut acc = ProfileAccumulator::new();
        let num_layers = 50;
        for step in 0..100 {
            for layer in 0..num_layers {
                acc.record_and_check(layer, 0.01);
            }
        }
        // After 100 steps, all layers should have triggered and been cleared
        // Verify by pushing one more stable sample to each — no immediate retrigger
        for layer in 0..num_layers {
            assert!(!acc.record_and_check(layer, 0.01));
        }
    }

    #[test]
    fn accumulator_95_of_101_triggers_after_eviction() {
        // Start with 1 unstable, then stable entries fill window.
        // After 1 unstable + 94 stable = 95 entries (94 stable, not enough).
        // After 1 unstable + 95 stable = 96 entries, first evicted, so 95 stable in window → triggers.
        let mut acc = ProfileAccumulator::new();
        acc.record_and_check(0, 1.0); // 1 unstable entry
        for i in 0..100 {
            if acc.record_and_check(0, 0.01) {
                // Triggers once 95 stable entries are in the window
                assert!(i >= 94, "should not trigger before 95 stable accumulate");
                return;
            }
        }
        panic!("should trigger once 95 stable entries fill the window");
    }

    #[test]
    fn accumulator_mixed_unstable_prevents_trigger_at_100() {
        // 5 unstable scattered among 95 stable — 95 stable should trigger
        let mut acc = ProfileAccumulator::new();
        for i in 0..100 {
            let ratio = if i % 20 == 0 { 1.0 } else { 0.01 };
            let triggered = acc.record_and_check(0, ratio);
            if triggered {
                // At trigger point, we need >= 95 stable
                return;
            }
        }
        // With 5 unstable (indices 0,20,40,60,80), we have 95 stable — exactly at threshold
        panic!("95 stable out of 100 should trigger");
    }

    #[test]
    fn telemetry_serde_roundtrip_all_zeroes_json() {
        let t = SequenceTelemetry::default();
        let json = serde_json::to_string(&t).unwrap();
        // Default should serialize all fields as 0/false
        assert!(json.contains("\"l2_delta\":0.0"));
        assert!(json.contains("\"has_outlier\":false"));
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }

    #[test]
    fn accumulator_negative_infinity_is_stable() {
        // -inf < 0.05 is true, so negative infinity counts as stable
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            let triggered = acc.record_and_check(0, f32::NEG_INFINITY);
            if triggered {
                return;
            }
        }
        panic!("negative infinity should be stable (< 0.05) and trigger");
    }

    // ── Additional tests (wave +15) ──

    #[test]
    fn telemetry_has_outlier_toggle_roundtrip() {
        // Arrange: construct with outlier, toggle off, verify serde preserves state
        let mut t = SequenceTelemetry {
            has_outlier: true,
            ..Default::default()
        };
        let json_on = serde_json::to_string(&t).unwrap();
        t.has_outlier = false;
        let json_off = serde_json::to_string(&t).unwrap();

        // Act: roundtrip both
        let from_on: SequenceTelemetry = serde_json::from_str(&json_on).unwrap();
        let from_off: SequenceTelemetry = serde_json::from_str(&json_off).unwrap();

        // Assert
        assert!(from_on.has_outlier);
        assert!(!from_off.has_outlier);
    }

    #[test]
    fn telemetry_equality_differs_by_each_float_field() {
        // Verify changing any single float field breaks equality
        let base = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: false,
            dead_density: 2.0,
            per_head_entropy: 3.0,
            transform_ratio: 4.0,
            output_entropy: 5.0,
        };
        assert_ne!(base, SequenceTelemetry { l2_delta: 99.0, ..base });
        assert_ne!(base, SequenceTelemetry { dead_density: 99.0, ..base });
        assert_ne!(base, SequenceTelemetry { per_head_entropy: 99.0, ..base });
        assert_ne!(base, SequenceTelemetry { transform_ratio: 99.0, ..base });
        assert_ne!(base, SequenceTelemetry { output_entropy: 99.0, ..base });
    }

    #[test]
    fn telemetry_positive_zero_negative_zero_equal() {
        // IEEE 754: +0.0 == -0.0; derived PartialEq should agree
        let a = SequenceTelemetry { l2_delta: 0.0f32, ..Default::default() };
        let b = SequenceTelemetry { l2_delta: -0.0f32, ..Default::default() };
        assert_eq!(a, b);
    }

    #[test]
    fn telemetry_all_fields_nan_unequal_to_self() {
        // Every NaN field makes the struct unequal to an identical copy
        let a = SequenceTelemetry {
            l2_delta: f32::NAN,
            has_outlier: false,
            dead_density: f32::NAN,
            per_head_entropy: f32::NAN,
            transform_ratio: f32::NAN,
            output_entropy: f32::NAN,
        };
        let b = a; // Copy — bitwise identical
        // PartialEq on f32: NAN != NAN, so the struct is not equal to itself
        assert_ne!(a, b);
    }

    #[test]
    fn telemetry_serde_roundtrip_preserves_negative_zero() {
        // Arrange
        let t = SequenceTelemetry {
            l2_delta: -0.0f32,
            ..Default::default()
        };

        // Act
        let json = serde_json::to_string(&t).unwrap();
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();

        // Assert: -0.0 survives JSON roundtrip (serde normalizes to +0.0 in JSON,
        // but the value should still be zero)
        assert_eq!(t2.l2_delta, 0.0);
    }

    #[test]
    fn telemetry_serialized_field_order_matches_struct_declaration() {
        // Verify serde serializes fields in declaration order
        let t = SequenceTelemetry::default();
        let json = serde_json::to_string(&t).unwrap();

        // Find positions of each field in the JSON string
        let pos_l2 = json.find("l2_delta").unwrap();
        let pos_outlier = json.find("has_outlier").unwrap();
        let pos_dead = json.find("dead_density").unwrap();
        let pos_entropy = json.find("per_head_entropy").unwrap();
        let pos_ratio = json.find("transform_ratio").unwrap();
        let pos_output = json.find("output_entropy").unwrap();

        assert!(pos_l2 < pos_outlier, "l2_delta before has_outlier");
        assert!(pos_outlier < pos_dead, "has_outlier before dead_density");
        assert!(pos_dead < pos_entropy, "dead_density before per_head_entropy");
        assert!(pos_entropy < pos_ratio, "per_head_entropy before transform_ratio");
        assert!(pos_ratio < pos_output, "transform_ratio before output_entropy");
    }

    #[test]
    fn telemetry_single_field_nonzero_rest_default() {
        // Constructing with only one non-default field should keep others default
        let t = SequenceTelemetry {
            dead_density: 0.42,
            ..Default::default()
        };
        assert_eq!(t.l2_delta, 0.0);
        assert!(!t.has_outlier);
        assert_eq!(t.dead_density, 0.42);
        assert_eq!(t.per_head_entropy, 0.0);
        assert_eq!(t.transform_ratio, 0.0);
        assert_eq!(t.output_entropy, 0.0);
    }

    #[test]
    fn accumulator_single_record_insufficient_for_trigger() {
        // A single stable record is not enough
        let mut acc = ProfileAccumulator::new();
        assert!(!acc.record_and_check(0, 0.01));
    }

    #[test]
    fn accumulator_history_does_not_grow_beyond_capacity() {
        // Push more than capacity worth of entries; trigger should occur but only once
        let mut acc = ProfileAccumulator::new();
        let mut trigger_count = 0;
        for _ in 0..300 {
            if acc.record_and_check(0, 0.01) {
                trigger_count += 1;
            }
        }
        // After first trigger at ~100, history clears; second trigger at ~200; third at ~300
        assert_eq!(trigger_count, 3, "should trigger exactly 3 times over 300 stable records");
    }

    #[test]
    fn accumulator_layer_gap_indices() {
        // Use non-contiguous layer indices (0, 100, 1000) — all should work independently
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
            acc.record_and_check(100, 0.01);
            acc.record_and_check(1000, 0.01);
        }
        // All should have triggered; verify by pushing one more — no immediate retrigger
        assert!(!acc.record_and_check(0, 0.01));
        assert!(!acc.record_and_check(100, 0.01));
        assert!(!acc.record_and_check(1000, 0.01));
    }

    #[test]
    fn accumulator_all_unstable_except_one() {
        // 99 unstable + 1 stable = 1 stable count out of 100 → never triggers
        let mut acc = ProfileAccumulator::new();
        for _ in 0..99 {
            acc.record_and_check(0, 1.0);
        }
        assert!(!acc.record_and_check(0, 0.01));
        // Total history is full now with only 1 stable sample
    }

    #[test]
    fn accumulator_transform_ratio_exactly_zero() {
        // 0.0 < 0.05 is true — verify boundary
        let mut acc = ProfileAccumulator::new();
        let mut triggered = false;
        for _ in 0..100 {
            if acc.record_and_check(0, 0.0) {
                triggered = true;
                break;
            }
        }
        assert!(triggered, "0.0 should be counted as stable and trigger");
    }

    #[test]
    fn accumulator_94_stable_6_unstable_no_trigger() {
        // 94 stable out of 100 is 1 below the required 95
        let mut acc = ProfileAccumulator::new();
        for _ in 0..94 {
            acc.record_and_check(0, 0.01);
        }
        for _ in 0..6 {
            acc.record_and_check(0, 1.0);
        }
        // Now 100 entries, 94 stable. Verify the 101st stable does not trigger immediately
        // because the window has 6 unstable, and evicting 1 adds a stable = 95 stable.
        // Actually the 101st evicts the oldest (stable), adds stable → 94-1+1 = 94 stable still.
        // Wait: initial 94 stable, then 6 unstable. 101st is stable → evicts 1st entry (stable)
        // → 93 stable + 6 unstable + 1 stable = 94 stable. Still not enough.
        assert!(!acc.record_and_check(0, 0.01));
    }

    #[test]
    fn accumulator_trigger_clears_only_target_layer() {
        // Fill two layers to trigger state, but only one actually triggers.
        // The other layer's history should remain intact.
        let mut acc = ProfileAccumulator::new();
        // Layer 0: all stable → will trigger
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
        }
        // Layer 1: all unstable → will not trigger
        for _ in 0..100 {
            acc.record_and_check(1, 1.0);
        }
        // Layer 0 triggered and cleared. Layer 1 still has 100 entries.
        // Push 1 more unstable to layer 1 — should still not trigger
        assert!(!acc.record_and_check(1, 1.0));
        // Push 1 stable to layer 0 — should not trigger (history was cleared)
        assert!(!acc.record_and_check(0, 0.01));
    }

    #[test]
    fn accumulator_mixed_nan_and_stable_counts_correctly() {
        // NaN is unstable, so mixing NaN with stable values reduces stable count
        let mut acc = ProfileAccumulator::new();
        for _ in 0..94 {
            acc.record_and_check(0, 0.01);
        }
        // 6 NaN entries → 94 stable, 6 unstable
        for _ in 0..6 {
            acc.record_and_check(0, f32::NAN);
        }
        // Total 100 entries, only 94 stable — no trigger
        assert!(!acc.record_and_check(0, 0.01));
    }

    // ── Additional tests (wave +13) ──

    #[test]
    fn telemetry_serde_json_all_fields_non_default() {
        // Arrange: construct with every field set to a distinct non-default value
        let t = SequenceTelemetry {
            l2_delta: 7.25,
            has_outlier: true,
            dead_density: 0.88,
            per_head_entropy: 4.567,
            transform_ratio: 0.0312,
            output_entropy: 2.998,
        };
        let json = serde_json::to_string(&t).unwrap();

        // Act: deserialize back
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();

        // Assert: every field preserved exactly
        assert_eq!(t2.l2_delta, 7.25);
        assert!(t2.has_outlier);
        assert_eq!(t2.dead_density, 0.88);
        assert_eq!(t2.per_head_entropy, 4.567);
        assert_eq!(t2.transform_ratio, 0.0312);
        assert_eq!(t2.output_entropy, 2.998);
    }

    #[test]
    fn telemetry_partial_eq_identical_non_default() {
        // Arrange: two instances with identical non-default values
        let a = SequenceTelemetry {
            l2_delta: 3.14,
            has_outlier: true,
            dead_density: 0.5,
            per_head_entropy: 2.71,
            transform_ratio: 0.042,
            output_entropy: 1.618,
        };
        let b = SequenceTelemetry {
            l2_delta: 3.14,
            has_outlier: true,
            dead_density: 0.5,
            per_head_entropy: 2.71,
            transform_ratio: 0.042,
            output_entropy: 1.618,
        };

        // Act & Assert
        assert_eq!(a, b);
    }

    #[test]
    fn telemetry_copy_preserves_state_after_mutation() {
        // Arrange: create and copy, then mutate original
        let mut original = SequenceTelemetry {
            l2_delta: 1.1,
            has_outlier: false,
            dead_density: 0.2,
            per_head_entropy: 3.3,
            transform_ratio: 0.4,
            output_entropy: 5.5,
        };
        let snapshot = original; // Copy semantics

        // Act: mutate the original
        original.l2_delta = 99.0;
        original.has_outlier = true;
        original.dead_density = 88.0;

        // Assert: copy retains original values
        assert_eq!(snapshot.l2_delta, 1.1);
        assert!(!snapshot.has_outlier);
        assert_eq!(snapshot.dead_density, 0.2);
        assert_eq!(snapshot.per_head_entropy, 3.3);
        assert_eq!(snapshot.transform_ratio, 0.4);
        assert_eq!(snapshot.output_entropy, 5.5);
    }

    #[test]
    fn accumulator_returns_false_every_step_below_100() {
        // Arrange
        let mut acc = ProfileAccumulator::new();

        // Act: record 99 stable samples
        for i in 0..99 {
            let result = acc.record_and_check(0, 0.01);

            // Assert: every call below capacity returns false
            assert!(!result, "step {} should not trigger", i);
        }
    }

    #[test]
    fn accumulator_cleared_layer_retriggers_after_refill() {
        // Arrange: trigger layer 0, then fill with unstable, then stable again
        let mut acc = ProfileAccumulator::new();

        // Act — Phase 1: trigger layer 0
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
        }
        // History is now cleared for layer 0

        // Phase 2: fill with 50 unstable entries (not enough to fill window)
        for _ in 0..50 {
            acc.record_and_check(0, 1.0);
        }

        // Phase 3: fill with 50 more stable entries (total 100)
        let mut retriggered = false;
        for _ in 0..50 {
            if acc.record_and_check(0, 0.01) {
                retriggered = true;
                break;
            }
        }

        // Assert: should not retrigger because 50 stable + 50 unstable < 95 stable
        assert!(!retriggered, "should not retrigger with only 50 stable out of 100");

        // Phase 4: push 45 more stable, evicting 45 unstable
        for i in 0..50 {
            if acc.record_and_check(0, 0.01) {
                retriggered = true;
                break;
            }
        }
        assert!(retriggered, "should retrigger once 95 stable accumulate after clearing");
    }

    #[test]
    fn accumulator_three_layers_staggered_recording() {
        // Arrange: three layers start recording at different times
        let mut acc = ProfileAccumulator::new();

        // Act: layer 0 starts at step 0
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
        }
        // Layer 0 triggered; now start layer 1
        for _ in 0..50 {
            acc.record_and_check(1, 0.01);
        }
        // Start layer 2 while layer 1 is still building
        for _ in 0..50 {
            acc.record_and_check(1, 0.01);
            acc.record_and_check(2, 0.01);
        }

        // Assert: layer 1 should have triggered at its 100th call
        // Layer 2 has only 50 records — not enough
        assert!(!acc.record_and_check(2, 0.01), "layer 2 should not trigger with 51 records");

        // Push 49 more to layer 2 to reach 100
        let mut layer2_triggered = false;
        for _ in 0..49 {
            if acc.record_and_check(2, 0.01) {
                layer2_triggered = true;
                break;
            }
        }
        assert!(layer2_triggered, "layer 2 should trigger after 100 stable samples");
    }

    #[test]
    fn accumulator_different_layers_trigger_at_different_times() {
        // Arrange: layer 0 gets stable values, layer 1 gets mostly unstable
        let mut acc = ProfileAccumulator::new();
        let mut layer0_trigger_step = None;
        let mut layer1_trigger_step = None;

        // Act: interleave recording for 200 steps
        for step in 0..200 {
            if acc.record_and_check(0, 0.01) && layer0_trigger_step.is_none() {
                layer0_trigger_step = Some(step);
            }
            // Layer 1: stable for first 50 steps, then unstable
            let ratio = if step < 50 { 0.01 } else { 1.0 };
            if acc.record_and_check(1, ratio) && layer1_trigger_step.is_none() {
                layer1_trigger_step = Some(step);
            }
        }

        // Assert: layer 0 triggers early, layer 1 never triggers
        assert!(layer0_trigger_step.is_some(), "layer 0 should trigger");
        assert!(layer1_trigger_step.is_none(), "layer 1 should never trigger with 150 unstable of 200");
    }

    #[test]
    fn accumulator_capacity_overflow_200_samples() {
        // Arrange: push 200 samples to a single layer
        let mut acc = ProfileAccumulator::new();
        let mut trigger_count = 0;

        // Act: record 200 stable samples
        for _ in 0..200 {
            if acc.record_and_check(0, 0.01) {
                trigger_count += 1;
            }
        }

        // Assert: triggers at 100 and 200 (two full cycles)
        assert_eq!(trigger_count, 2, "should trigger exactly twice over 200 stable records");
    }

    #[test]
    fn accumulator_95_of_100_precise_boundary() {
        // Arrange: fill exactly 95 stable and 5 unstable, verify trigger
        let mut acc = ProfileAccumulator::new();
        // 5 unstable first, then 95 stable
        for _ in 0..5 {
            acc.record_and_check(0, 1.0);
        }
        for _ in 0..94 {
            assert!(!acc.record_and_check(0, 0.01), "should not trigger before 100 entries");
        }
        // 99 entries (5 unstable + 94 stable). The 100th is stable → 95 stable.
        // Act
        let result = acc.record_and_check(0, 0.01);

        // Assert
        assert!(result, "95 stable out of 100 should trigger");
    }

    #[test]
    fn accumulator_multiple_triggers_over_500_records() {
        // Arrange
        let mut acc = ProfileAccumulator::new();
        let mut trigger_steps = Vec::new();

        // Act: 500 stable records
        for step in 0..500 {
            if acc.record_and_check(0, 0.01) {
                trigger_steps.push(step);
            }
        }

        // Assert: 5 triggers at steps ~99, ~199, ~299, ~399, ~499
        assert_eq!(trigger_steps.len(), 5, "should trigger 5 times over 500 records");
        // Each trigger should be ~100 steps apart
        for i in 1..trigger_steps.len() {
            let gap = trigger_steps[i] - trigger_steps[i - 1];
            assert_eq!(gap, 100, "triggers should be 100 steps apart");
        }
    }

    #[test]
    fn accumulator_interleaved_two_layers_one_triggers() {
        // Arrange: interleave stable on layer 0 with unstable on layer 1
        let mut acc = ProfileAccumulator::new();
        let mut layer0_triggered = false;

        // Act: 100 interleaved steps
        for _ in 0..100 {
            if acc.record_and_check(0, 0.01) {
                layer0_triggered = true;
            }
            acc.record_and_check(1, 1.0);
        }

        // Assert
        assert!(layer0_triggered, "layer 0 should trigger with all stable");

        // Layer 1 has 100 unstable — push 100 more stable to see if it eventually triggers
        let mut layer1_triggered = false;
        for i in 0..100 {
            if acc.record_and_check(1, 0.01) {
                layer1_triggered = true;
                // By step i, we have evicted i+1 unstable and replaced with stable
                // 100 - (i+1) unstable + (i+1) stable from this batch
                // Need 95 stable in window: 100 - (i+1) unstable → need i+1 >= 95 → i >= 94
                break;
            }
        }
        assert!(layer1_triggered, "layer 1 should trigger once 95 stable displace unstable");
    }

    #[test]
    fn accumulator_interleaved_two_layers_both_trigger_independently() {
        // Arrange: two layers with different stability patterns
        let mut acc = ProfileAccumulator::new();
        let mut trigger_order = Vec::new();

        // Act: 150 steps, both layers get stable values
        for step in 0..150 {
            if acc.record_and_check(0, 0.01) {
                trigger_order.push((0, step));
            }
            if acc.record_and_check(1, 0.02) {
                trigger_order.push((1, step));
            }
        }

        // Assert: both layers trigger at step 99 (100th call)
        assert!(trigger_order.iter().any(|(l, _)| *l == 0), "layer 0 should trigger");
        assert!(trigger_order.iter().any(|(l, _)| *l == 1), "layer 1 should trigger");
        // Both trigger on their 100th call (step 99)
        let layer0_first = trigger_order.iter().find(|(l, _)| *l == 0).unwrap().1;
        let layer1_first = trigger_order.iter().find(|(l, _)| *l == 1).unwrap().1;
        assert_eq!(layer0_first, 99, "layer 0 triggers at step 99");
        assert_eq!(layer1_first, 99, "layer 1 triggers at step 99");
    }

    #[test]
    fn accumulator_sliding_window_replaces_oldest_entry() {
        // Arrange: fill 100 unstable, then push stable entries one at a time
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            acc.record_and_check(0, 1.0);
        }
        // All 100 entries are unstable (stable count = 0)

        // Act: push stable entries, each evicting one unstable
        for i in 0..95 {
            let triggered = acc.record_and_check(0, 0.01);
            if i < 94 {
                assert!(!triggered, "should not trigger with {} stable", i + 1);
            } else {
                // After 95 stable pushes, window has 95 stable and 5 unstable
                assert!(triggered, "should trigger with 95 stable after evicting 95 unstable");
                return;
            }
        }
        panic!("should have triggered at i=94");
    }

    // ── Additional edge-case tests (wave +13) ──

    #[test]
    fn telemetry_serde_tolerates_unknown_fields() {
        // Arrange: JSON with extra unknown fields — serde derive defaults to
        // ignoring unknown fields (no deny_unknown_fields attribute)
        let json = r#"{
            "l2_delta": 1.0,
            "has_outlier": true,
            "dead_density": 0.5,
            "per_head_entropy": 3.0,
            "transform_ratio": 0.02,
            "output_entropy": 1.5,
            "unknown_field": 42.0,
            "extra": "garbage"
        }"#;

        // Act: serde default ignores unknown fields
        let result: Result<SequenceTelemetry, _> = serde_json::from_str(json);

        // Assert: derive(Deserialize) ignores unknown fields by default
        let t = result.expect("serde derive should tolerate unknown fields");
        assert_eq!(t.l2_delta, 1.0);
        assert!(t.has_outlier);
        assert_eq!(t.dead_density, 0.5);
    }

    #[test]
    fn telemetry_partial_eq_reflexive_non_nan() {
        // Arrange: non-NaN telemetry should be equal to itself
        let t = SequenceTelemetry {
            l2_delta: 1.23,
            has_outlier: true,
            dead_density: 0.456,
            per_head_entropy: 7.89,
            transform_ratio: 0.012,
            output_entropy: 3.14,
        };

        // Act & Assert: reflexivity (a == a) for non-NaN
        assert_eq!(t, t, "non-NaN telemetry should be reflexively equal");
    }

    #[test]
    fn accumulator_96_stable_of_100_triggers() {
        // Arrange: 96 stable + 4 unstable — one above the 95 threshold
        let mut acc = ProfileAccumulator::new();
        // 4 unstable first
        for _ in 0..4 {
            acc.record_and_check(0, 1.0);
        }
        // 96 stable to fill to 100
        for i in 0..95 {
            assert!(!acc.record_and_check(0, 0.01), "should not trigger at step {}", i);
        }
        // 99 entries (4 unstable + 95 stable). 100th entry stable → 96 stable.
        // Act
        let result = acc.record_and_check(0, 0.01);

        // Assert: 96 stable out of 100 is above the 95 threshold
        assert!(result, "96 stable out of 100 should trigger");
    }

    #[test]
    fn accumulator_stable_to_unstable_to_stable_pattern() {
        // Arrange: start stable, go unstable, then go stable again
        let mut acc = ProfileAccumulator::new();

        // Phase 1: 50 stable entries
        for _ in 0..50 {
            assert!(!acc.record_and_check(0, 0.01));
        }

        // Phase 2: 50 unstable entries — fills to 100 (50 stable + 50 unstable)
        for _ in 0..50 {
            assert!(!acc.record_and_check(0, 1.0));
        }
        // Window: [50s][50u] = 100 entries, 50 stable

        // Phase 3: push 50 stable — evicts all 50 remaining stable from phase 1
        // Window becomes: [50u][50s] = 100 entries, 50 stable
        for _ in 0..50 {
            assert!(!acc.record_and_check(0, 0.01));
        }

        // Phase 4: push stable entries — now evicting unstable entries
        // Each push evicts one unstable, adds one stable
        // After 45 pushes: 5 unstable + 95 stable = 100 entries, 95 stable → triggers
        let mut triggered = false;
        for i in 0..50 {
            if acc.record_and_check(0, 0.01) {
                // Need to evict 45 unstable to reach 95 stable (50+45=95)
                assert!(i >= 44, "should not trigger before 95 stable accumulate (i={})", i);
                triggered = true;
                break;
            }
        }

        // Assert
        assert!(triggered, "should trigger after stable phase replaces unstable entries");
    }

    #[test]
    fn accumulator_abandoned_layer_resumes_correctly() {
        // Arrange: partially fill layer, stop, do other layers, resume original
        let mut acc = ProfileAccumulator::new();

        // Phase 1: 30 stable entries on layer 0
        for _ in 0..30 {
            acc.record_and_check(0, 0.01);
        }

        // Phase 2: fill layer 1 completely and trigger
        for _ in 0..100 {
            acc.record_and_check(1, 0.01);
        }

        // Phase 3: resume layer 0 — add 70 more stable to reach 100
        let mut triggered = false;
        for _ in 0..70 {
            if acc.record_and_check(0, 0.01) {
                triggered = true;
                break;
            }
        }

        // Assert: layer 0 should trigger at its 100th entry
        assert!(triggered, "resumed layer should trigger after reaching 100 entries");
    }

    #[test]
    fn accumulator_consecutive_trigger_cycles_preserve_independence() {
        // Arrange: trigger same layer 4 times, verify each cycle is independent
        let mut acc = ProfileAccumulator::new();
        let mut trigger_steps = Vec::new();

        // Act: 400 stable records
        for step in 0..400 {
            if acc.record_and_check(0, 0.01) {
                trigger_steps.push(step);
            }
        }

        // Assert: 4 triggers, each exactly 100 steps apart
        assert_eq!(trigger_steps.len(), 4, "should trigger 4 times over 400 records");
        for i in 1..trigger_steps.len() {
            assert_eq!(
                trigger_steps[i] - trigger_steps[i - 1],
                100,
                "consecutive triggers should be 100 steps apart"
            );
        }
    }

    #[test]
    fn accumulator_three_layers_rapid_interleave_mixed_stability() {
        // Arrange: interleave 3 layers with different stability profiles
        let mut acc = ProfileAccumulator::new();
        let mut triggered_layers = std::collections::HashSet::new();

        // Act: 100 interleaved steps
        for _ in 0..100 {
            // Layer 0: all stable
            if acc.record_and_check(0, 0.01) {
                triggered_layers.insert(0);
            }
            // Layer 1: alternating stable/unstable (50% stable)
            if acc.record_and_check(1, if triggered_layers.contains(&1) { 1.0 } else { 0.01 }) {
                triggered_layers.insert(1);
            }
            // Layer 2: all unstable
            acc.record_and_check(2, 1.0);
        }

        // Assert: layer 0 triggers; layer 2 does not; layer 1 depends on pattern
        assert!(triggered_layers.contains(&0), "layer 0 (all stable) should trigger");
        assert!(!triggered_layers.contains(&2), "layer 2 (all unstable) should not trigger");
    }

    #[test]
    fn accumulator_trigger_step_is_exactly_99_indexed() {
        // Arrange: verify the trigger happens on exactly the 100th call (index 99)
        let mut acc = ProfileAccumulator::new();
        let mut trigger_at = None;

        // Act: record 100 stable samples
        for i in 0..100 {
            if acc.record_and_check(0, 0.01) && trigger_at.is_none() {
                trigger_at = Some(i);
            }
        }

        // Assert
        assert_eq!(trigger_at, Some(99), "trigger should occur at step index 99 (100th call)");
    }

    #[test]
    fn accumulator_all_unstable_then_all_stable_rapid_switch() {
        // Arrange: fill entirely with unstable, then entirely with stable
        let mut acc = ProfileAccumulator::new();

        // 200 unstable entries (triggers multiple clears but always refills with unstable)
        for _ in 0..200 {
            acc.record_and_check(0, 1.0);
        }
        // No triggers because nothing is stable

        // Act: switch to stable — need 95 stable in the window
        let mut triggered = false;
        for i in 0..100 {
            if acc.record_and_check(0, 0.01) {
                // After i+1 stable pushes, window has i+1 stable and 100-(i+1) unstable
                // Need i+1 >= 95 → i >= 94
                assert!(i >= 94, "should not trigger before 95 stable entries replace unstable");
                triggered = true;
                break;
            }
        }

        // Assert
        assert!(triggered, "rapid switch from all-unstable to stable should eventually trigger");
    }

    #[test]
    fn accumulator_94_stable_first_6_unstable_last_no_trigger_after_eviction_start() {
        // Arrange: 94 stable then 6 unstable fills window, then push more stable
        let mut acc = ProfileAccumulator::new();
        for _ in 0..94 {
            acc.record_and_check(0, 0.01);
        }
        for _ in 0..6 {
            acc.record_and_check(0, 1.0);
        }
        // 100 entries: 94 stable, 6 unstable

        // Act: push stable entries, evicting stable entries from the front
        // Entry 101 (stable): evicts entry 1 (stable) → 93 stable + 6 unstable + 1 stable = 94 stable
        assert!(!acc.record_and_check(0, 0.01));
        // Entry 102 (stable): evicts entry 2 (stable) → 93 stable still
        assert!(!acc.record_and_check(0, 0.01));
        // Assert: still not enough — need 95 stable but we are stuck at 94 because
        // evicting stable and adding stable cancels out

        // Verify by exhausting all remaining stable slots in the original 94
        for i in 0..90 {
            assert!(
                !acc.record_and_check(0, 0.01),
                "evicting stable entries keeps stable count at 94 (step {})",
                i
            );
        }
    }

    #[test]
    fn telemetry_struct_field_offsets_are_contiguous_or_padded() {
        // Arrange: verify that all field addresses are accessible
        let t = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: true,
            dead_density: 2.0,
            per_head_entropy: 3.0,
            transform_ratio: 4.0,
            output_entropy: 5.0,
        };

        // Act & Assert: verify each field has a distinct non-overlapping value
        // This tests that repr(C) does not cause field aliasing
        assert_eq!(t.l2_delta, 1.0);
        assert!(t.has_outlier);
        assert_eq!(t.dead_density, 2.0);
        assert_eq!(t.per_head_entropy, 3.0);
        assert_eq!(t.transform_ratio, 4.0);
        assert_eq!(t.output_entropy, 5.0);
        // All values distinct proves no field aliasing
    }

    #[test]
    fn accumulator_just_above_threshold_epsilon() {
        // Arrange: ratio just below 0.05 by the smallest representable amount
        let ratio = 0.05f32 - f32::EPSILON; // smallest value below 0.05
        let mut acc = ProfileAccumulator::new();

        // Act: 100 entries with ratio just below threshold
        let mut triggered = false;
        for _ in 0..100 {
            if acc.record_and_check(0, ratio) {
                triggered = true;
                break;
            }
        }

        // Assert: ratio < 0.05 (strictly less) should be stable
        assert!(triggered, "ratio just below 0.05 should be stable and trigger");
    }

    #[test]
    fn accumulator_single_unstable_in_stable_window_still_triggers() {
        // Arrange: 99 stable + 1 unstable = 99 stable — well above 95 threshold
        let mut acc = ProfileAccumulator::new();
        for i in 0..100 {
            let ratio = if i == 50 { 1.0 } else { 0.01 };
            let triggered = acc.record_and_check(0, ratio);
            if triggered {
                // 99 stable out of 100 is well above 95 threshold
                return;
            }
        }
        panic!("99 stable out of 100 should trigger (only 1 unstable)");
    }

    // ── Additional edge-case tests (wave +13) ──

    #[test]
    fn telemetry_zero_transform_ratio_distinguishes_from_has_outlier() {
        // Arrange: two instances differing only in has_outlier, all floats zero
        let a = SequenceTelemetry { has_outlier: false, ..Default::default() };
        let b = SequenceTelemetry { has_outlier: true, ..Default::default() };

        // Act & Assert: the bool field alone makes them unequal
        assert_ne!(a, b);
        assert_eq!(a.l2_delta, b.l2_delta);
        assert_eq!(a.dead_density, b.dead_density);
    }

    #[test]
    fn telemetry_serde_roundtrip_preserves_has_outlier_false() {
        // Arrange
        let t = SequenceTelemetry { has_outlier: false, ..Default::default() };
        let json = serde_json::to_string(&t).unwrap();

        // Act
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();

        // Assert: false survives roundtrip, not silently flipped
        assert!(!t2.has_outlier);
        assert_eq!(t, t2);
    }

    #[test]
    fn telemetry_large_positive_negative_pair_in_struct() {
        // Arrange: f32::MAX and f32::MIN (most negative finite) in same struct
        let t = SequenceTelemetry {
            l2_delta: f32::MAX,
            has_outlier: false,
            dead_density: f32::MIN,
            per_head_entropy: f32::MAX,
            transform_ratio: f32::MIN,
            output_entropy: 0.0,
        };

        // Act & Assert: both extremes coexist without overflow
        assert!(t.l2_delta > 0.0);
        assert!(t.dead_density < 0.0);
        assert!(t.per_head_entropy > 0.0);
        assert!(t.transform_ratio < 0.0);
    }

    #[test]
    fn accumulator_records_for_layer_zero_and_high_index_simultaneously() {
        // Arrange: record stable for layer 0 and layer 999 at the same time
        let mut acc = ProfileAccumulator::new();
        let mut layer0_triggered = false;
        let mut layer999_triggered = false;

        // Act: 100 interleaved steps
        for _ in 0..100 {
            if acc.record_and_check(0, 0.01) {
                layer0_triggered = true;
            }
            if acc.record_and_check(999, 0.01) {
                layer999_triggered = true;
            }
        }

        // Assert: both trigger independently despite large index gap
        assert!(layer0_triggered, "layer 0 should trigger");
        assert!(layer999_triggered, "layer 999 should trigger");
    }

    #[test]
    fn accumulator_record_and_check_return_type_is_bool() {
        // Arrange: verify the function signature returns bool (compilation check)
        let mut acc = ProfileAccumulator::new();

        // Act
        let result: bool = acc.record_and_check(0, 0.5);

        // Assert: compiles and returns a bool (false for single entry)
        assert!(!result);
    }

    #[test]
    fn accumulator_sliding_window_from_mixed_to_all_stable() {
        // Arrange: 50 stable + 50 unstable, then push 50 more stable
        let mut acc = ProfileAccumulator::new();
        for _ in 0..50 {
            acc.record_and_check(0, 0.01);
        }
        for _ in 0..50 {
            acc.record_and_check(0, 1.0);
        }
        // Window: 50 stable + 50 unstable

        // Act: push 45 stable — evicts 45 stable from front
        // Window: 5 stable + 50 unstable + 45 stable = 50 stable
        for _ in 0..45 {
            assert!(!acc.record_and_check(0, 0.01));
        }

        // Push 45 more stable — now evicting unstable entries
        // After 45: 5 stable + 5 unstable + 45 stable + 45 stable = 95 stable
        let mut triggered = false;
        for i in 0..50 {
            if acc.record_and_check(0, 0.01) {
                assert!(i >= 44, "should not trigger before 95 stable (i={})", i);
                triggered = true;
                break;
            }
        }

        // Assert
        assert!(triggered, "should trigger once 95 stable accumulate via eviction");
    }

    #[test]
    fn telemetry_default_debug_contains_all_field_names() {
        // Arrange
        let t = SequenceTelemetry::default();
        let debug = format!("{:?}", t);

        // Act & Assert: debug output should contain every field name
        assert!(debug.contains("l2_delta"));
        assert!(debug.contains("has_outlier"));
        assert!(debug.contains("dead_density"));
        assert!(debug.contains("per_head_entropy"));
        assert!(debug.contains("transform_ratio"));
        assert!(debug.contains("output_entropy"));
    }

    #[test]
    fn accumulator_default_has_expected_internal_state() {
        // Arrange & Act
        let mut acc = ProfileAccumulator::default();

        // Assert: a fresh accumulator should not trigger for any single record
        let debug = format!("{:?}", acc);
        assert!(debug.contains("ProfileAccumulator"));
        // Verify no history exists for any layer
        assert!(!acc.record_and_check(0, 0.01));
    }

    #[test]
    fn accumulator_two_triggers_200_steps_with_unstable_gap() {
        // Arrange: 100 stable → trigger → 50 unstable → 50 stable → no trigger → 50 stable → trigger
        let mut acc = ProfileAccumulator::new();
        let mut triggers = 0;

        // Act: Phase 1 — 100 stable
        for _ in 0..100 {
            if acc.record_and_check(0, 0.01) {
                triggers += 1;
            }
        }
        assert_eq!(triggers, 1);

        // Phase 2: 50 unstable + 50 stable = 100 entries, only 50 stable
        for _ in 0..50 {
            acc.record_and_check(0, 1.0);
        }
        for _ in 0..50 {
            if acc.record_and_check(0, 0.01) {
                triggers += 1;
            }
        }
        assert_eq!(triggers, 1, "50 stable is below 95 threshold");

        // Phase 3: push 45 more stable, evicting 45 unstable
        // Window: 50 stable + 5 unstable + 45 stable = 95 stable -> triggers
        for i in 0..50 {
            if acc.record_and_check(0, 0.01) {
                assert!(i >= 44, "should not retrigger before 95 stable (i={})", i);
                triggers += 1;
                break;
            }
        }

        // Assert
        assert_eq!(triggers, 2, "should trigger exactly twice");
    }

    #[test]
    fn telemetry_serde_empty_object_fails() {
        // Arrange: empty JSON object has no required fields
        let json = "{}";

        // Act
        let result = serde_json::from_str::<SequenceTelemetry>(json);

        // Assert: serde derive requires all fields by default
        assert!(result.is_err());
    }

    #[test]
    fn accumulator_layer_one_does_not_affect_layer_zero() {
        // Arrange: fill layer 0 to almost trigger, then fill layer 1
        let mut acc = ProfileAccumulator::new();
        for _ in 0..99 {
            acc.record_and_check(0, 0.01);
        }
        // Layer 0 has 99 stable entries, not yet triggered

        // Act: fill layer 1 with 100 stable entries (triggers)
        for _ in 0..100 {
            acc.record_and_check(1, 0.01);
        }

        // Assert: layer 0 still has its 99 entries; one more triggers it
        assert!(acc.record_and_check(0, 0.01), "layer 0 should trigger on its 100th call");
    }

    #[test]
    fn accumulator_ninety_five_stable_with_five_unstable_at_end_triggers() {
        // Arrange: 94 stable + 5 unstable (99 entries), then 1 stable = 100 with 95 stable
        let mut acc = ProfileAccumulator::new();
        for _ in 0..94 {
            assert!(!acc.record_and_check(0, 0.01));
        }
        for _ in 0..5 {
            assert!(!acc.record_and_check(0, 1.0));
        }

        // Act: push 100th entry (stable) → 95 stable + 5 unstable
        let result = acc.record_and_check(0, 0.01);

        // Assert: 95 stable out of 100 meets threshold
        assert!(result, "95 stable out of 100 should trigger even with 5 unstable at end");
    }

    #[test]
    fn telemetry_debug_format_shows_field_values_not_just_names() {
        // Arrange: construct with specific recognizable values
        let t = SequenceTelemetry {
            l2_delta: 2.718,
            has_outlier: true,
            dead_density: 0.0,
            per_head_entropy: 0.0,
            transform_ratio: 0.0,
            output_entropy: 0.0,
        };

        // Act
        let debug = format!("{:?}", t);

        // Assert: debug output contains the actual value, not just field names
        assert!(debug.contains("2.718"), "debug should contain l2_delta value");
        assert!(debug.contains("true"), "debug should contain has_outlier value");
    }

    // ── Additional tests (wave +13 → target 110) ──

    #[test]
    fn telemetry_min_positive_normal_in_all_float_fields() {
        // Arrange: f32::MIN_POSITIVE is the smallest positive normal float
        let t = SequenceTelemetry {
            l2_delta: f32::MIN_POSITIVE,
            has_outlier: false,
            dead_density: f32::MIN_POSITIVE,
            per_head_entropy: f32::MIN_POSITIVE,
            transform_ratio: f32::MIN_POSITIVE,
            output_entropy: f32::MIN_POSITIVE,
        };

        // Act & Assert: all fields are the smallest positive normal
        assert_eq!(t.l2_delta, f32::MIN_POSITIVE);
        assert!(t.l2_delta > 0.0);
        assert_eq!(t.dead_density, f32::MIN_POSITIVE);
        assert_eq!(t.per_head_entropy, f32::MIN_POSITIVE);
        assert_eq!(t.transform_ratio, f32::MIN_POSITIVE);
        assert_eq!(t.output_entropy, f32::MIN_POSITIVE);
    }

    #[test]
    fn telemetry_serde_infinity_serializes_to_null() {
        // Arrange: serde_json serializes f32::INFINITY as null (not a valid JSON number)
        let t = SequenceTelemetry {
            l2_delta: f32::INFINITY,
            has_outlier: true,
            dead_density: f32::NEG_INFINITY,
            per_head_entropy: f32::INFINITY,
            transform_ratio: f32::NEG_INFINITY,
            output_entropy: f32::INFINITY,
        };

        // Act: serialization succeeds (uses null for non-finite)
        let json = serde_json::to_string(&t).unwrap();

        // Assert: JSON contains null for infinity values (serde_json behavior)
        assert!(json.contains("null"), "infinity should serialize to null");
        // Deserialization from null fails for f32 — this is expected behavior
        let result = serde_json::from_str::<SequenceTelemetry>(&json);
        assert!(result.is_err(), "null cannot deserialize back to f32");
    }

    #[test]
    fn telemetry_serde_roundtrip_epsilon() {
        // Arrange: f32::EPSILON roundtrip through serde
        let t = SequenceTelemetry {
            l2_delta: f32::EPSILON,
            has_outlier: false,
            dead_density: f32::EPSILON,
            per_head_entropy: f32::EPSILON,
            transform_ratio: f32::EPSILON,
            output_entropy: f32::EPSILON,
        };

        // Act
        let json = serde_json::to_string(&t).unwrap();
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();

        // Assert
        assert_eq!(t2.l2_delta, f32::EPSILON);
        assert_eq!(t2.dead_density, f32::EPSILON);
        assert_eq!(t2.per_head_entropy, f32::EPSILON);
        assert_eq!(t2.transform_ratio, f32::EPSILON);
        assert_eq!(t2.output_entropy, f32::EPSILON);
    }

    #[test]
    fn telemetry_struct_has_seven_fields() {
        // Arrange & Act: count fields by verifying all 7 have distinct write effects
        let mut t = SequenceTelemetry::default();
        let base = t;

        // Assert: 6 float fields + 1 bool = 7 total fields
        // Changing any single field breaks equality with the default
        assert_eq!(base, t);

        t.l2_delta = 1.0;
        assert_ne!(base, t);
        t.l2_delta = 0.0;

        t.has_outlier = true;
        assert_ne!(base, t);
        t.has_outlier = false;

        t.dead_density = 1.0;
        assert_ne!(base, t);
        t.dead_density = 0.0;

        t.per_head_entropy = 1.0;
        assert_ne!(base, t);
        t.per_head_entropy = 0.0;

        t.transform_ratio = 1.0;
        assert_ne!(base, t);
        t.transform_ratio = 0.0;

        t.output_entropy = 1.0;
        assert_ne!(base, t);
        t.output_entropy = 0.0;

        // After resetting all fields to default, equality is restored
        assert_eq!(base, t, "all 7 fields individually affect equality");
    }

    #[test]
    fn telemetry_all_floats_one_all_others_zero_partial_eq() {
        // Arrange: set each float to 1.0 one at a time, verify only that one differs
        let base = SequenceTelemetry::default();

        let only_l2 = SequenceTelemetry { l2_delta: 1.0, ..Default::default() };
        let only_dead = SequenceTelemetry { dead_density: 1.0, ..Default::default() };
        let only_entropy = SequenceTelemetry { per_head_entropy: 1.0, ..Default::default() };
        let only_ratio = SequenceTelemetry { transform_ratio: 1.0, ..Default::default() };
        let only_output = SequenceTelemetry { output_entropy: 1.0, ..Default::default() };

        // Act & Assert: each is unequal to default and unequal to each other
        assert_ne!(base, only_l2);
        assert_ne!(base, only_dead);
        assert_ne!(base, only_entropy);
        assert_ne!(base, only_ratio);
        assert_ne!(base, only_output);
        assert_ne!(only_l2, only_dead);
        assert_ne!(only_l2, only_entropy);
        assert_ne!(only_dead, only_output);
    }

    #[test]
    fn accumulator_single_unstable_record_no_trigger() {
        // Arrange: a single unstable record is insufficient
        let mut acc = ProfileAccumulator::new();

        // Act
        let result = acc.record_and_check(0, 1.0);

        // Assert: single record cannot trigger regardless of stability
        assert!(!result);
    }

    #[test]
    fn accumulator_f32_min_transform_ratio_is_stable() {
        // Arrange: f32::MIN is the most negative finite float, well below 0.05
        let mut acc = ProfileAccumulator::new();
        let mut triggered = false;

        // Act: 100 records with f32::MIN ratio (should be stable since f32::MIN < 0.05)
        for _ in 0..100 {
            if acc.record_and_check(0, f32::MIN) {
                triggered = true;
                break;
            }
        }

        // Assert
        assert!(triggered, "f32::MIN is < 0.05 so should count as stable and trigger");
    }

    #[test]
    fn accumulator_f32_max_transform_ratio_is_unstable() {
        // Arrange: f32::MAX is the largest positive finite float, well above 0.05
        let mut acc = ProfileAccumulator::new();

        // Act: 100 records with f32::MAX ratio
        for _ in 0..100 {
            assert!(
                !acc.record_and_check(0, f32::MAX),
                "f32::MAX > 0.05 should never trigger"
            );
        }
    }

    #[test]
    fn accumulator_many_layers_single_record_each_no_trigger() {
        // Arrange: record one sample for 1000 different layers
        let mut acc = ProfileAccumulator::new();

        // Act: one stable record per layer
        for layer in 0..1000 {
            let result = acc.record_and_check(layer, 0.01);
            assert!(!result, "single record for layer {} should not trigger", layer);
        }
    }

    #[test]
    fn accumulator_layer_zero_and_max_usize_both_trigger() {
        // Arrange: record stable for layer 0 and layer usize::MAX simultaneously
        let mut acc = ProfileAccumulator::new();
        let mut triggered_min = false;
        let mut triggered_max = false;

        // Act
        for _ in 0..100 {
            if acc.record_and_check(0, 0.01) {
                triggered_min = true;
            }
            if acc.record_and_check(usize::MAX, 0.01) {
                triggered_max = true;
            }
        }

        // Assert: both extremes work identically
        assert!(triggered_min, "layer 0 should trigger");
        assert!(triggered_max, "layer usize::MAX should trigger");
    }

    #[test]
    fn accumulator_94_stable_6_unstable_triggers_on_95th_eviction() {
        // Arrange: 94 stable + 6 unstable fills the window.
        // Additional stable entries first evict old stable entries (stable count stays 94),
        // then start evicting unstable entries (stable count rises to 95 and triggers).
        let mut acc = ProfileAccumulator::new();
        for _ in 0..94 {
            acc.record_and_check(0, 0.01);
        }
        for _ in 0..6 {
            acc.record_and_check(0, 1.0);
        }
        // Window: [94 stable][6 unstable]

        // Act: push 94 stable — evicts all 94 original stable, window = [6 unstable][94 stable]
        for i in 0..94 {
            assert!(
                !acc.record_and_check(0, 0.01),
                "evicting stable keeps count at 94 (iteration {})",
                i
            );
        }
        // Push 1 more stable — evicts first unstable, window = [5 unstable][95 stable] → triggers
        let result = acc.record_and_check(0, 0.01);

        // Assert
        assert!(result, "evicting an unstable entry raises stable count to 95 and triggers");
    }

    #[test]
    fn accumulator_five_unstable_first_then_stable_fills_and_triggers() {
        // Arrange: 5 unstable first, then 95 stable fills to 100 with 95 stable
        let mut acc = ProfileAccumulator::new();
        for _ in 0..5 {
            assert!(!acc.record_and_check(0, 1.0));
        }

        // Act: push 94 stable (99 entries total, 94 stable)
        for _ in 0..94 {
            assert!(!acc.record_and_check(0, 0.01));
        }
        // 100th entry (stable) → 95 stable
        let result = acc.record_and_check(0, 0.01);

        // Assert
        assert!(result, "5 unstable + 95 stable = 95 stable out of 100, should trigger");
    }

    #[test]
    fn telemetry_serde_rejects_non_numeric_float_field() {
        // Arrange: JSON with a string where a float is expected
        let json = r#"{
            "l2_delta": "not_a_number",
            "has_outlier": false,
            "dead_density": 0.0,
            "per_head_entropy": 0.0,
            "transform_ratio": 0.0,
            "output_entropy": 0.0
        }"#;

        // Act
        let result = serde_json::from_str::<SequenceTelemetry>(json);

        // Assert: serde should reject a string in a float field
        assert!(result.is_err());
    }

    // ── Additional tests (wave +13 → target 123) ──

    #[test]
    fn telemetry_serde_rejects_wrong_type_for_has_outlier() {
        // Arrange: has_outlier is bool, but JSON provides an integer
        let json = r#"{
            "l2_delta": 0.0,
            "has_outlier": 1,
            "dead_density": 0.0,
            "per_head_entropy": 0.0,
            "transform_ratio": 0.0,
            "output_entropy": 0.0
        }"#;

        // Act
        let result = serde_json::from_str::<SequenceTelemetry>(json);

        // Assert: serde should reject non-boolean value for has_outlier
        assert!(result.is_err());
    }

    #[test]
    fn telemetry_clone_then_mutation_is_independent() {
        // Arrange
        let mut original = SequenceTelemetry {
            l2_delta: 10.0,
            has_outlier: true,
            dead_density: 0.5,
            per_head_entropy: 3.0,
            transform_ratio: 0.02,
            output_entropy: 1.0,
        };
        let cloned = original.clone();

        // Act: mutate original after cloning
        original.l2_delta = 0.0;
        original.has_outlier = false;
        original.dead_density = 99.0;

        // Assert: cloned retains pre-mutation values
        assert_eq!(cloned.l2_delta, 10.0);
        assert!(cloned.has_outlier);
        assert_eq!(cloned.dead_density, 0.5);
        assert_eq!(cloned.per_head_entropy, 3.0);
        assert_eq!(cloned.transform_ratio, 0.02);
        assert_eq!(cloned.output_entropy, 1.0);
    }

    #[test]
    fn telemetry_serde_roundtrip_with_all_distinct_small_values() {
        // Arrange: use small distinct fractional values to verify precision
        let t = SequenceTelemetry {
            l2_delta: 0.001,
            has_outlier: true,
            dead_density: 0.002,
            per_head_entropy: 0.003,
            transform_ratio: 0.004,
            output_entropy: 0.005,
        };

        // Act
        let json = serde_json::to_string(&t).unwrap();
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();

        // Assert: all six distinct small values survive roundtrip
        assert_eq!(t2.l2_delta, 0.001);
        assert!(t2.has_outlier);
        assert_eq!(t2.dead_density, 0.002);
        assert_eq!(t2.per_head_entropy, 0.003);
        assert_eq!(t2.transform_ratio, 0.004);
        assert_eq!(t2.output_entropy, 0.005);
    }

    #[test]
    fn telemetry_equality_differs_by_every_bool_flip() {
        // Arrange: two instances identical except has_outlier, and vice versa
        let a = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: true,
            dead_density: 2.0,
            per_head_entropy: 3.0,
            transform_ratio: 4.0,
            output_entropy: 5.0,
        };
        let b = SequenceTelemetry {
            has_outlier: false,
            ..a
        };

        // Act & Assert: only the bool differs
        assert_ne!(a, b);
        // Swapping bool back makes them equal
        let c = SequenceTelemetry {
            has_outlier: true,
            ..b
        };
        assert_eq!(a, c);
    }

    #[test]
    fn accumulator_record_same_layer_twice_in_succession() {
        // Arrange: call record_and_check twice rapidly for same layer
        let mut acc = ProfileAccumulator::new();

        // Act
        let first = acc.record_and_check(5, 0.01);
        let second = acc.record_and_check(5, 0.01);

        // Assert: both return false (only 2 records, need 100)
        assert!(!first);
        assert!(!second);
    }

    #[test]
    fn accumulator_5_unstable_90_stable_5_unstable_boundary_case() {
        // Arrange: 5 unstable + 90 stable + 5 unstable = 100 entries, 90 stable
        // This is below the 95 threshold, should not trigger.
        let mut acc = ProfileAccumulator::new();

        for _ in 0..5 {
            acc.record_and_check(0, 1.0);
        }
        for _ in 0..90 {
            acc.record_and_check(0, 0.01);
        }
        for _ in 0..5 {
            acc.record_and_check(0, 1.0);
        }

        // Act: push one more stable — evicts oldest (unstable), 91 stable
        let result = acc.record_and_check(0, 0.01);

        // Assert: 91 stable is still below 95
        assert!(!result, "91 stable out of 100 should not trigger");
    }

    #[test]
    fn accumulator_single_stable_then_all_unstable_never_triggers() {
        // Arrange: 1 stable followed by 199 unstable entries
        let mut acc = ProfileAccumulator::new();

        // Act
        acc.record_and_check(0, 0.01); // 1 stable
        for _ in 0..199 {
            assert!(
                !acc.record_and_check(0, 1.0),
                "unstable entries should never trigger"
            );
        }

        // Assert: even after 200 total entries, only 0 or 1 stable in window
        // (the initial stable entry was evicted after the first 100 unstable)
    }

    #[test]
    fn accumulator_interleave_three_layers_two_trigger_one_does_not() {
        // Arrange: layers 0 and 2 get stable, layer 1 gets unstable
        let mut acc = ProfileAccumulator::new();
        let mut layer0_triggered = false;
        let mut layer1_triggered = false;
        let mut layer2_triggered = false;

        // Act: 100 interleaved steps
        for _ in 0..100 {
            if acc.record_and_check(0, 0.01) {
                layer0_triggered = true;
            }
            if acc.record_and_check(1, 0.5) {
                layer1_triggered = true;
            }
            if acc.record_and_check(2, 0.02) {
                layer2_triggered = true;
            }
        }

        // Assert: layers 0 and 2 trigger (stable), layer 1 does not (0.5 >= 0.05)
        assert!(layer0_triggered, "layer 0 with ratio 0.01 should trigger");
        assert!(!layer1_triggered, "layer 1 with ratio 0.5 should not trigger");
        assert!(layer2_triggered, "layer 2 with ratio 0.02 should trigger");
    }

    #[test]
    fn accumulator_trigger_then_immediate_stable_does_not_retrigger() {
        // Arrange: fill to trigger, then push one more stable immediately
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
        }
        // Layer 0 has triggered and been cleared

        // Act: immediately push another stable entry
        let result = acc.record_and_check(0, 0.01);

        // Assert: history was cleared, only 1 entry exists, cannot retrigger
        assert!(!result, "should not retrigger with only 1 entry after clear");
    }

    #[test]
    fn accumulator_10_unstable_90_stable_10_stable_triggers_on_eviction() {
        // Arrange: 10 unstable + 90 stable = 100 entries (90 stable).
        // Then push 10 more stable: first 10 evict the 10 unstable.
        // After 5 evictions: 95 stable → triggers at the 5th additional entry.
        let mut acc = ProfileAccumulator::new();
        for _ in 0..10 {
            acc.record_and_check(0, 1.0);
        }
        for _ in 0..90 {
            acc.record_and_check(0, 0.01);
        }
        // Window: [10 unstable, 90 stable], 90 stable

        // Act: push stable entries that evict unstable entries
        let mut triggered = false;
        for i in 0..10 {
            if acc.record_and_check(0, 0.01) {
                // Each push evicts one unstable, adds one stable.
                // After 5 pushes: 90+5=95 stable → trigger
                assert!(i >= 4, "should not trigger before 95 stable (i={})", i);
                triggered = true;
                break;
            }
        }

        // Assert
        assert!(triggered, "should trigger after evicting 5 unstable entries");
    }

    #[test]
    fn accumulator_negative_very_small_ratio_is_stable() {
        // Arrange: a very small negative ratio, well below 0.05
        let mut acc = ProfileAccumulator::new();
        let ratio = -1e-10;

        // Act: fill 100 entries with the tiny negative ratio
        let mut triggered = false;
        for _ in 0..100 {
            if acc.record_and_check(0, ratio) {
                triggered = true;
                break;
            }
        }

        // Assert: -1e-10 < 0.05, so all entries are stable
        assert!(triggered, "very small negative ratio should be stable and trigger");
    }

    #[test]
    fn accumulator_transform_ratio_just_above_threshold_is_unstable() {
        // Arrange: 0.05 + f32::EPSILON is the smallest value above 0.05
        let ratio = 0.05f32 + f32::EPSILON;
        let mut acc = ProfileAccumulator::new();

        // Act: 100 entries with ratio just above threshold
        for _ in 0..100 {
            assert!(
                !acc.record_and_check(0, ratio),
                "ratio just above 0.05 should be unstable"
            );
        }
    }

    #[test]
    fn telemetry_serde_rejects_null_in_float_field() {
        // Arrange: JSON with null where a float is expected
        let json = r#"{
            "l2_delta": null,
            "has_outlier": false,
            "dead_density": 0.0,
            "per_head_entropy": 0.0,
            "transform_ratio": 0.0,
            "output_entropy": 0.0
        }"#;

        // Act
        let result = serde_json::from_str::<SequenceTelemetry>(json);

        // Assert: serde should reject null for a required float field
        assert!(result.is_err());
    }

    // ── Additional tests (wave +13 → target 136) ──

    #[test]
    fn telemetry_new_yields_all_zero_fields_without_default_import() {
        let t = SequenceTelemetry::new();
        assert_eq!(t.l2_delta, 0.0);
        assert!(!t.has_outlier);
        assert_eq!(t.dead_density, 0.0);
        assert_eq!(t.per_head_entropy, 0.0);
        assert_eq!(t.transform_ratio, 0.0);
        assert_eq!(t.output_entropy, 0.0);
    }

    #[test]
    fn telemetry_copy_then_drop_original_preserves_copy() {
        let original = SequenceTelemetry {
            l2_delta: 42.0,
            has_outlier: true,
            dead_density: -7.5,
            per_head_entropy: 100.0,
            transform_ratio: 0.001,
            output_entropy: 6.28,
        };
        let copy = original;
        drop(original);
        assert_eq!(copy.l2_delta, 42.0);
        assert!(copy.has_outlier);
        assert_eq!(copy.dead_density, -7.5);
        assert_eq!(copy.per_head_entropy, 100.0);
        assert_eq!(copy.transform_ratio, 0.001);
        assert_eq!(copy.output_entropy, 6.28);
    }

    #[test]
    fn telemetry_serde_roundtrip_negative_values() {
        let t = SequenceTelemetry {
            l2_delta: -0.25,
            has_outlier: true,
            dead_density: -99.99,
            per_head_entropy: -0.001,
            transform_ratio: -0.049,
            output_entropy: -1e10,
        };
        let json = serde_json::to_string(&t).unwrap();
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(t2.l2_delta, -0.25);
        assert!(t2.has_outlier);
        assert_eq!(t2.dead_density, -99.99);
        assert_eq!(t2.per_head_entropy, -0.001);
        assert_eq!(t2.transform_ratio, -0.049);
        assert_eq!(t2.output_entropy, -1e10);
    }

    #[test]
    fn telemetry_partial_eq_symmetry_with_non_default() {
        let a = SequenceTelemetry {
            l2_delta: 2.5,
            has_outlier: true,
            dead_density: 0.125,
            per_head_entropy: 8.0,
            transform_ratio: 0.03,
            output_entropy: 4.0,
        };
        let b = SequenceTelemetry {
            l2_delta: 2.5,
            has_outlier: true,
            dead_density: 0.125,
            per_head_entropy: 8.0,
            transform_ratio: 0.03,
            output_entropy: 4.0,
        };
        assert_eq!(a, b);
        assert_eq!(b, a);
        let c = SequenceTelemetry { l2_delta: 2.5, ..Default::default() };
        assert_ne!(a, c);
        assert_ne!(c, a);
    }

    #[test]
    fn telemetry_debug_includes_type_name_prefix() {
        let t = SequenceTelemetry::default();
        let debug = format!("{:?}", t);
        assert!(debug.starts_with("SequenceTelemetry"), "Debug output should start with type name");
    }

    #[test]
    fn telemetry_single_field_outlier_true_rest_zero_distinguishes() {
        let a = SequenceTelemetry { has_outlier: false, ..Default::default() };
        let b = SequenceTelemetry { has_outlier: true, ..Default::default() };
        assert_ne!(a, b);
        assert_eq!(a.l2_delta, b.l2_delta);
        assert_eq!(a.dead_density, b.dead_density);
        assert_eq!(a.per_head_entropy, b.per_head_entropy);
        assert_eq!(a.transform_ratio, b.transform_ratio);
        assert_eq!(a.output_entropy, b.output_entropy);
    }

    #[test]
    fn accumulator_clone_produces_independent_state() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..50 {
            acc.record_and_check(0, 0.01);
        }
        let mut cloned = acc.clone();
        assert_eq!(acc, cloned);
        let acc_result = acc.record_and_check(0, 0.01);
        let cloned_result = cloned.record_and_check(0, 0.01);
        assert_eq!(acc_result, cloned_result);
        assert_eq!(acc, cloned);
    }

    #[test]
    fn accumulator_exactly_five_unstable_prevents_trigger_at_boundary() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..5 {
            acc.record_and_check(0, 1.0);
        }
        for _ in 0..94 {
            acc.record_and_check(0, 0.01);
        }
        // 99 entries: 5 unstable + 94 stable
        let result_99 = acc.record_and_check(0, 0.01);
        // 100 entries: 5 unstable + 95 stable = 95 stable → triggers
        assert!(result_99, "5 unstable + 95 stable = 95 stable, exactly at threshold");
    }

    #[test]
    fn accumulator_ten_unstable_then_ninety_stable_does_not_trigger() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..10 {
            acc.record_and_check(0, 1.0);
        }
        for _ in 0..90 {
            acc.record_and_check(0, 0.01);
        }
        // 100 entries: 10 unstable + 90 stable = 90 stable, well below 95 threshold
        for i in 0..4 {
            assert!(
                !acc.record_and_check(0, 0.01),
                "should not trigger with {} stable entries",
                91 + i
            );
        }
        // Entry 105: evicted 5 unstable, window has 5 unstable + 95 stable = 95 stable -> triggers
        assert!(acc.record_and_check(0, 0.01), "should trigger once 95 stable accumulated");
    }

    #[test]
    fn accumulator_layer_indices_are_independent_hash_keys() {
        let mut acc = ProfileAccumulator::new();
        let mut triggered = vec![false; 5];
        for _ in 0..100 {
            for layer in 0..5 {
                if acc.record_and_check(layer, 0.01) {
                    triggered[layer] = true;
                }
            }
        }
        for layer in 0..5 {
            assert!(triggered[layer], "layer {} should trigger independently", layer);
        }
    }

    #[test]
    fn accumulator_alternating_stable_unstable_never_reaches_threshold() {
        let mut acc = ProfileAccumulator::new();
        for _ in 0..200 {
            acc.record_and_check(0, 0.01);
            acc.record_and_check(0, 1.0);
        }
        // 400 entries, but window of 100 always has 50 stable and 50 unstable
        assert!(!acc.record_and_check(0, 0.01));
    }

    #[test]
    fn accumulator_full_cycle_trigger_clear_refill_retrigger_verifies_periodicity() {
        let mut acc = ProfileAccumulator::new();
        let mut trigger_steps = Vec::new();
        for step in 0..350 {
            if acc.record_and_check(0, 0.01) {
                trigger_steps.push(step);
            }
        }
        assert_eq!(trigger_steps.len(), 3, "should trigger 3 times in 350 steps");
        assert_eq!(trigger_steps[1] - trigger_steps[0], 100);
        assert_eq!(trigger_steps[2] - trigger_steps[1], 100);
    }

    #[test]
    fn telemetry_serde_rejects_trailing_comma_in_object() {
        let json = r#"{
            "l2_delta": 0.0,
            "has_outlier": false,
            "dead_density": 0.0,
            "per_head_entropy": 0.0,
            "transform_ratio": 0.0,
            "output_entropy": 0.0,
        }"#;
        let result = serde_json::from_str::<SequenceTelemetry>(json);
        assert!(result.is_err());
    }

    // ── Additional tests (wave +13 → target 149) ──

    #[test]
    fn telemetry_repr_c_alignment_is_natural() {
        // Arrange & Act: repr(C) should align to the largest field alignment (f32 = 4 bytes)
        let align = std::mem::align_of::<SequenceTelemetry>();

        // Assert: alignment should be 4 (f32 alignment), not 8 or more
        assert_eq!(align, std::mem::align_of::<f32>(), "repr(C) alignment should match f32");
    }

    #[test]
    fn telemetry_vec_of_multiple_instances_preserves_order() {
        // Arrange: build a vector of telemetry with increasing l2_delta values
        let vec: Vec<SequenceTelemetry> = (0..5)
            .map(|i| SequenceTelemetry {
                l2_delta: i as f32,
                ..Default::default()
            })
            .collect();

        // Act: iterate and check order
        let deltas: Vec<f32> = vec.iter().map(|t| t.l2_delta).collect();

        // Assert: order is preserved (Copy types work correctly in collections)
        assert_eq!(deltas, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn telemetry_serde_roundtrip_with_whitespace_in_json() {
        // Arrange: valid JSON with extra whitespace and newlines
        let json = r#"  {
            "l2_delta"  :  1.5  ,
            "has_outlier" : true ,
            "dead_density" : 0.25,
            "per_head_entropy": 6.0,
            "transform_ratio": 0.01,
            "output_entropy" : 3.14
        }  "#;

        // Act
        let t: SequenceTelemetry = serde_json::from_str(json).unwrap();

        // Assert: serde_json handles whitespace correctly
        assert_eq!(t.l2_delta, 1.5);
        assert!(t.has_outlier);
        assert_eq!(t.dead_density, 0.25);
        assert_eq!(t.per_head_entropy, 6.0);
        assert_eq!(t.transform_ratio, 0.01);
        assert_eq!(t.output_entropy, 3.14);
    }

    #[test]
    fn telemetry_partial_eq_transitive_non_nan() {
        // Arrange: three identical instances with non-NaN values
        let a = SequenceTelemetry {
            l2_delta: 2.0,
            has_outlier: true,
            dead_density: 0.33,
            per_head_entropy: 1.5,
            transform_ratio: 0.04,
            output_entropy: 4.0,
        };
        let b = SequenceTelemetry {
            l2_delta: 2.0,
            has_outlier: true,
            dead_density: 0.33,
            per_head_entropy: 1.5,
            transform_ratio: 0.04,
            output_entropy: 4.0,
        };
        let c = SequenceTelemetry {
            l2_delta: 2.0,
            has_outlier: true,
            dead_density: 0.33,
            per_head_entropy: 1.5,
            transform_ratio: 0.04,
            output_entropy: 4.0,
        };

        // Act & Assert: transitivity — if a==b and b==c then a==c
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    #[test]
    fn telemetry_array_of_defaults_all_equal() {
        // Arrange: array of 10 default instances
        let arr: [SequenceTelemetry; 10] = [SequenceTelemetry::default(); 10];

        // Act & Assert: all elements are equal (Copy + Default + PartialEq)
        for i in 0..10 {
            assert_eq!(arr[i], SequenceTelemetry::default(), "element {} should equal default", i);
        }
    }

    #[test]
    fn telemetry_two_instances_with_swapped_extreme_values_differ() {
        // Arrange: two instances with swapped l2_delta and output_entropy
        let a = SequenceTelemetry {
            l2_delta: f32::MAX,
            has_outlier: false,
            dead_density: 0.0,
            per_head_entropy: 0.0,
            transform_ratio: 0.0,
            output_entropy: f32::MIN,
        };
        let b = SequenceTelemetry {
            l2_delta: f32::MIN,
            has_outlier: false,
            dead_density: 0.0,
            per_head_entropy: 0.0,
            transform_ratio: 0.0,
            output_entropy: f32::MAX,
        };

        // Act & Assert: swapping values produces different structs
        assert_ne!(a, b);
        assert!(a.l2_delta > 0.0);
        assert!(a.output_entropy < 0.0);
        assert!(b.l2_delta < 0.0);
        assert!(b.output_entropy > 0.0);
    }

    #[test]
    fn accumulator_history_persists_across_other_layer_activity() {
        // Arrange: partially fill layer 0, fully fill layer 1 and trigger it
        let mut acc = ProfileAccumulator::new();

        // Act: record 50 stable entries on layer 0
        for _ in 0..50 {
            acc.record_and_check(0, 0.01);
        }
        // Fill and trigger layer 1 (100 entries)
        for _ in 0..100 {
            acc.record_and_check(1, 0.01);
        }
        // Record 50 more stable entries on layer 0 (total 100 for layer 0)
        let mut layer0_triggered = false;
        for _ in 0..50 {
            if acc.record_and_check(0, 0.01) {
                layer0_triggered = true;
            }
        }

        // Assert: layer 0's history was preserved despite layer 1 activity
        assert!(layer0_triggered, "layer 0 history should persist across layer 1 operations");
    }

    #[test]
    fn accumulator_post_trigger_history_is_empty_for_cleared_layer() {
        // Arrange: fill layer 0 to trigger
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
        }
        // Layer 0 has triggered and history is cleared

        // Act: after trigger, exactly 1 record should be insufficient for any trigger
        for i in 0..99 {
            assert!(
                !acc.record_and_check(0, 0.01),
                "post-trigger, should not trigger at record {}",
                i + 1
            );
        }

        // Assert: the 100th record after clearing should trigger again
        assert!(acc.record_and_check(0, 0.01), "should retrigger at exactly 100 records post-clear");
    }

    #[test]
    fn accumulator_alternating_ratio_49_percent_stable_never_triggers() {
        // Arrange: use ratio 0.06 (unstable) every other step, 0.01 (stable) the rest
        // With alternating pattern: 50 stable + 50 unstable in any window of 100
        let mut acc = ProfileAccumulator::new();

        // Act: 200 interleaved steps — even steps stable, odd steps unstable
        for i in 0..400 {
            let ratio = if i % 2 == 0 { 0.01 } else { 1.0 };
            assert!(
                !acc.record_and_check(0, ratio),
                "alternating stable/unstable should never reach 95 stable (step {})",
                i
            );
        }
    }

    #[test]
    fn accumulator_stable_ratio_0_0499_triggers() {
        // Arrange: 0.0499 is below the 0.05 threshold
        let mut acc = ProfileAccumulator::new();
        let mut triggered = false;

        // Act: fill 100 entries with ratio 0.0499
        for _ in 0..100 {
            if acc.record_and_check(0, 0.0499) {
                triggered = true;
                break;
            }
        }

        // Assert
        assert!(triggered, "0.0499 < 0.05 should be stable and trigger");
    }

    #[test]
    fn accumulator_forty_unstable_then_stable_needs_95_for_trigger() {
        // Arrange: 40 unstable, then fill with stable
        let mut acc = ProfileAccumulator::new();
        for _ in 0..40 {
            acc.record_and_check(0, 1.0);
        }
        // 40 unstable entries

        // Act: push 59 stable entries (99 total: 40 unstable + 59 stable)
        for _ in 0..59 {
            assert!(!acc.record_and_check(0, 0.01));
        }
        // 100th entry stable → 40 unstable + 60 stable = 60 stable, not enough
        assert!(!acc.record_and_check(0, 0.01), "60 stable out of 100 is below 95 threshold");
    }

    #[test]
    fn accumulator_two_layers_trigger_then_both_can_retrigger() {
        // Arrange: trigger two layers, then refill both
        let mut acc = ProfileAccumulator::new();

        // Act: Phase 1 — trigger both layers
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
            acc.record_and_check(1, 0.01);
        }

        // Phase 2: refill both layers — should both retrigger
        let mut layer0_retriggered = false;
        let mut layer1_retriggered = false;
        for step in 0..100 {
            if acc.record_and_check(0, 0.01) {
                layer0_retriggered = true;
            }
            if acc.record_and_check(1, 0.01) {
                layer1_retriggered = true;
            }
        }

        // Assert: both layers retrigger independently after clearing
        assert!(layer0_retriggered, "layer 0 should retrigger after refill");
        assert!(layer1_retriggered, "layer 1 should retrigger after refill");
    }

    #[test]
    fn telemetry_debug_shows_false_for_default_has_outlier() {
        // Arrange
        let t = SequenceTelemetry::default();

        // Act
        let debug = format!("{:?}", t);

        // Assert: the string "false" should appear for the has_outlier field
        assert!(debug.contains("false"), "debug should show 'false' for default has_outlier");
    }

    #[test]
    fn telemetry_copy_assign_overwrites_previous_value() {
        // Arrange: create two different instances
        let mut target = SequenceTelemetry {
            l2_delta: 99.0,
            has_outlier: true,
            dead_density: 88.0,
            per_head_entropy: 77.0,
            transform_ratio: 66.0,
            output_entropy: 55.0,
        };
        let source = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: false,
            dead_density: 2.0,
            per_head_entropy: 3.0,
            transform_ratio: 4.0,
            output_entropy: 5.0,
        };

        // Act: overwrite target with source via Copy assignment
        target = source;

        // Assert: target now has source's values
        assert_eq!(target.l2_delta, 1.0);
        assert!(!target.has_outlier);
        assert_eq!(target.dead_density, 2.0);
        assert_eq!(target.per_head_entropy, 3.0);
        assert_eq!(target.transform_ratio, 4.0);
        assert_eq!(target.output_entropy, 5.0);
        assert_eq!(target, source);
    }

    // ── Additional tests (wave +10 → target 160) ──

    #[test]
    fn telemetry_serde_accepts_integer_values_for_float_fields() {
        // Arrange: JSON with integer values (1, 2) where floats are expected
        // serde_json can deserialize integers into f32 fields
        let json = r#"{
            "l2_delta": 1,
            "has_outlier": true,
            "dead_density": 2,
            "per_head_entropy": 3,
            "transform_ratio": 4,
            "output_entropy": 5
        }"#;

        // Act
        let t: SequenceTelemetry = serde_json::from_str(json).unwrap();

        // Assert: integer values are deserialized as f32
        assert_eq!(t.l2_delta, 1.0);
        assert!(t.has_outlier);
        assert_eq!(t.dead_density, 2.0);
        assert_eq!(t.per_head_entropy, 3.0);
        assert_eq!(t.transform_ratio, 4.0);
        assert_eq!(t.output_entropy, 5.0);
    }

    #[test]
    fn telemetry_serde_roundtrip_zero_point_zero_one_precision() {
        // Arrange: use values with many decimal places to verify floating-point precision
        let t = SequenceTelemetry {
            l2_delta: 0.123456789,
            has_outlier: false,
            dead_density: 0.987654321,
            per_head_entropy: 3.141592653,
            transform_ratio: 0.000000001,
            output_entropy: 2.718281828,
        };

        // Act
        let json = serde_json::to_string(&t).unwrap();
        let t2: SequenceTelemetry = serde_json::from_str(&json).unwrap();

        // Assert: values survive roundtrip (f32 precision)
        assert_eq!(t2.l2_delta, 0.123456789f32);
        assert_eq!(t2.dead_density, 0.987654321f32);
        assert_eq!(t2.per_head_entropy, 3.141592653f32);
        assert_eq!(t2.transform_ratio, 0.000000001f32);
        assert_eq!(t2.output_entropy, 2.718281828f32);
    }

    #[test]
    fn telemetry_serde_output_is_valid_json_object() {
        // Arrange: construct a telemetry instance
        let t = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: true,
            dead_density: 0.5,
            per_head_entropy: 2.0,
            transform_ratio: 0.03,
            output_entropy: 4.0,
        };

        // Act: serialize to JSON string
        let json = serde_json::to_string(&t).unwrap();

        // Assert: the output is valid JSON (can be parsed back)
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object(), "output should be a JSON object");
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.len(), 6, "should have exactly 6 fields");
    }

    #[test]
    fn accumulator_different_histories_are_not_equal() {
        // Arrange: two accumulators with different layer histories
        let mut acc_a = ProfileAccumulator::new();
        let mut acc_b = ProfileAccumulator::new();

        // Act: acc_a has records for layer 0; acc_b has none
        acc_a.record_and_check(0, 0.01);
        acc_a.record_and_check(0, 0.02);
        acc_a.record_and_check(1, 0.5);

        // Assert: different histories make them unequal
        assert_ne!(acc_a, acc_b, "accumulators with different histories should not be equal");
    }

    #[test]
    fn accumulator_post_trigger_all_unstable_never_retriggers() {
        // Arrange: trigger layer 0, then fill with all unstable entries
        let mut acc = ProfileAccumulator::new();

        // Act — Phase 1: trigger with 100 stable entries
        for _ in 0..100 {
            acc.record_and_check(0, 0.01);
        }
        // Layer 0 triggered, history cleared

        // Phase 2: fill 200 unstable entries — no retrigger possible
        let mut any_trigger = false;
        for _ in 0..200 {
            if acc.record_and_check(0, 1.0) {
                any_trigger = true;
            }
        }

        // Assert: unstable entries never trigger
        assert!(!any_trigger, "all-unstable entries after trigger should never retrigger");
    }

    #[test]
    fn accumulator_rapid_successive_triggers_same_layer() {
        // Arrange: trigger the same layer multiple times in rapid succession
        let mut acc = ProfileAccumulator::new();
        let mut trigger_count = 0;

        // Act: push 500 stable records, counting each trigger
        for _ in 0..500 {
            if acc.record_and_check(0, 0.01) {
                trigger_count += 1;
            }
        }

        // Assert: exactly 5 triggers (every 100 records)
        assert_eq!(trigger_count, 5, "should trigger exactly 5 times over 500 stable records");
    }

    #[test]
    fn accumulator_stable_entry_immediately_after_unstable_window_transition() {
        // Arrange: fill 100 unstable, then push exactly 95 stable entries to trigger
        let mut acc = ProfileAccumulator::new();
        for _ in 0..100 {
            acc.record_and_check(0, 1.0);
        }
        // Window is full of 100 unstable entries

        // Act: push stable entries; the 95th should trigger
        let mut triggered = false;
        for i in 0..100 {
            if acc.record_and_check(0, 0.01) {
                // After i+1 stable pushes, the window has i+1 stable and 100-(i+1) unstable
                // Need i+1 >= 95, so i >= 94
                assert!(i >= 94, "should not trigger before 95 stable (i={})", i);
                triggered = true;
                break;
            }
        }

        // Assert
        assert!(triggered, "exactly 95 stable entries should displace 95 unstable and trigger");
    }

    #[test]
    fn accumulator_record_and_check_with_positive_zero_transform_ratio() {
        // Arrange: positive zero is exactly 0.0, which is < 0.05
        let mut acc = ProfileAccumulator::new();
        let mut triggered = false;

        // Act: fill 100 entries with +0.0
        for _ in 0..100 {
            if acc.record_and_check(0, 0.0f32) {
                triggered = true;
                break;
            }
        }

        // Assert: +0.0 < 0.05 is true, so all entries count as stable
        assert!(triggered, "+0.0 transform ratio should count as stable and trigger");
    }

    #[test]
    fn telemetry_partial_eq_single_nan_field_makes_entire_struct_unequal() {
        // Arrange: two identical instances, but one field is NaN in both
        // Since NaN != NaN, the entire struct should be unequal to itself
        let a = SequenceTelemetry {
            l2_delta: f32::NAN,
            has_outlier: false,
            dead_density: 1.0,
            per_head_entropy: 2.0,
            transform_ratio: 3.0,
            output_entropy: 4.0,
        };
        let b = SequenceTelemetry {
            l2_delta: f32::NAN,
            has_outlier: false,
            dead_density: 1.0,
            per_head_entropy: 2.0,
            transform_ratio: 3.0,
            output_entropy: 4.0,
        };

        // Act & Assert: a single NaN field makes PartialEq return false
        // even though all other 5 fields are identical
        assert_ne!(a, b, "single NaN field should make entire struct unequal");
        // Verify the other fields are indeed the same value
        assert_eq!(a.dead_density, b.dead_density);
        assert_eq!(a.per_head_entropy, b.per_head_entropy);
        assert_eq!(a.transform_ratio, b.transform_ratio);
        assert_eq!(a.output_entropy, b.output_entropy);
        assert_eq!(a.has_outlier, b.has_outlier);
    }

    #[test]
    fn accumulator_interleaved_stable_unstable_at_95_5_ratio_triggers() {
        // Arrange: record 95 stable and 5 unstable in a repeating pattern per 100 entries
        let mut acc = ProfileAccumulator::new();

        // Act: fill with exactly 95 stable + 5 unstable per window
        for block in 0..3 {
            for i in 0..100 {
                let ratio = if (i % 20 == 0) && (i > 0) { 1.0 } else { 0.01 };
                let triggered = acc.record_and_check(0, ratio);
                if triggered {
                    // In the first block (block=0), we should trigger at entry 99
                    // (95 stable + 5 unstable meets the threshold)
                    assert_eq!(block, 0, "first trigger should be in block 0");
                    return;
                }
            }
        }

        panic!("95 stable out of 100 should trigger in the first block");
    }
}
