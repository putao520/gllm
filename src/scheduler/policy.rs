use super::jit_types::{SchedulerDecision, SystemState};

/// Configurable thresholds for scheduling policies.
#[derive(Debug, Clone)]
pub struct PolicyConfig {
    /// Memory pressure emergency threshold (triggers batch reduction).
    pub pressure_emergency: f32,
    /// Memory pressure ceiling for aggressive expansion.
    pub pressure_aggressive_ceiling: f32,
    /// KV fragmentation defrag threshold.
    pub frag_defrag_threshold: f32,
    /// Waiting queue length to trigger aggressive mode.
    pub queue_aggressive_trigger: usize,
    /// Safe mode max batch size.
    pub batch_safe: usize,
    /// Normal mode max batch size.
    pub batch_normal: usize,
    /// Aggressive mode max batch size.
    pub batch_aggressive: usize,
}

impl PolicyConfig {
    pub fn absolute() -> Self {
        Self {
            pressure_emergency: 0.9,
            pressure_aggressive_ceiling: 1.0,
            frag_defrag_threshold: 0.5,
            queue_aggressive_trigger: usize::MAX,
            batch_safe: 32,
            batch_normal: 32,
            batch_aggressive: 32,
        }
    }
}

/// JIT Scheduling Policy Interface.
/// Pure logic component.
pub trait SchedulingPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision;
}

/// Absolute Policy (Single-Track Scheduling)
///
/// SPEC 07-OBSERVABILITY §2.3 AbsolutePolicy:
/// - memory_pressure > emergency: emergency mode
/// - kv_fragmentation > frag_threshold: defrag mode
/// - otherwise: safe mode
pub struct AbsolutePolicy {
    config: PolicyConfig,
}

impl Default for AbsolutePolicy {
    fn default() -> Self {
        Self { config: PolicyConfig::absolute() }
    }
}

impl AbsolutePolicy {
    pub fn with_config(config: PolicyConfig) -> Self {
        Self { config }
    }
}

impl SchedulingPolicy for AbsolutePolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        // Phase 2: entropy-driven batch reduction — high entropy → smaller batch for quality
        let entropy_cap = if state.logits_entropy > 8.0 {
            // Very high entropy: model is highly uncertain, limit batch for quality
            (state.current_running_len.max(1)).min(self.config.batch_safe)
        } else {
            self.config.batch_safe
        };

        // Phase 2: attention sparsity → reduce batch to exploit sparsity (fewer active tokens)
        let sparsity_cap = if state.attention_sparsity > 0.7 {
            // Clamp reduction factor to [0.0, 1.0] so extreme sparsity (>2.0)
            // still produces a meaningful (but small) batch size instead of
            // clamping the final product to 0 and falling back to current_running_len.
            let reduction_factor = (1.0 - state.attention_sparsity * 0.5).clamp(0.0, 1.0);
            let reduced = (self.config.batch_safe as f32 * reduction_factor) as usize;
            reduced.max(state.current_running_len.max(1))
        } else {
            self.config.batch_safe
        };

        let nominal_batch = entropy_cap.min(sparsity_cap);

        if state.memory_pressure > self.config.pressure_emergency {
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1),
                admit_new_prefill: false,
                // Guard: negative memory_pressure → usize wrap
                force_swap_out_count: (state.memory_pressure * 3.0).ceil().clamp(0.0, 1000.0) as usize,
            }
        } else if state.kv_fragmentation > self.config.frag_defrag_threshold {
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1),
                admit_new_prefill: false,
                force_swap_out_count: 1,
            }
        } else {
            SchedulerDecision {
                max_batch_size: nominal_batch,
                admit_new_prefill: true,
                force_swap_out_count: 0,
            }
        }
    }
}

/// Enum Dispatch for Zero-Cost Abstraction
#[derive(Clone, Default)]
pub enum PolicyVariant {
    #[default]
    Absolute,
}

impl PolicyVariant {
    #[inline(always)]
    pub fn decide(&self, state: &SystemState) -> SchedulerDecision {
        match self {
            Self::Absolute => AbsolutePolicy::default().decide(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with(pressure: f32, frag: f32, waiting: usize, running: usize) -> SystemState {
        SystemState {
            memory_pressure: pressure,
            kv_fragmentation: frag,
            waiting_queue_len: waiting,
            current_running_len: running,
            current_batch_size: running,
            ..Default::default()
        }
    }

    #[test]
    fn absolute_emergency_mode() {
        let d = AbsolutePolicy::default().decide(&state_with(0.95, 0.0, 0, 4));
        assert_eq!(d.max_batch_size, 4);
        assert!(!d.admit_new_prefill);
        assert!(d.force_swap_out_count > 0);
    }

    #[test]
    fn absolute_defrag_mode() {
        let d = AbsolutePolicy::default().decide(&state_with(0.5, 0.6, 0, 3));
        assert_eq!(d.max_batch_size, 3);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
    }

    #[test]
    fn absolute_safe_mode() {
        let d = AbsolutePolicy::default().decide(&state_with(0.3, 0.2, 0, 2));
        assert_eq!(d.max_batch_size, 32);
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    #[test]
    fn policy_variant_dispatch() {
        let state = state_with(0.3, 0.2, 5, 2);
        let a = PolicyVariant::Absolute.decide(&state);
        // Should produce valid decision
        assert!(a.admit_new_prefill);
    }

    #[test]
    fn custom_config_overrides() {
        let mut config = PolicyConfig::absolute();
        config.batch_safe = 128;
        config.pressure_emergency = 0.95;
        let policy = AbsolutePolicy::with_config(config);
        // Safe mode uses custom batch size
        let d = policy.decide(&state_with(0.5, 0.0, 10, 4));
        assert_eq!(d.max_batch_size, 128);
        // Emergency threshold raised — 0.9 no longer triggers emergency
        let d = policy.decide(&state_with(0.9, 0.0, 0, 5));
        assert_eq!(d.max_batch_size, 128);
    }

    // ── PolicyConfig ──

    #[test]
    fn policy_config_absolute_defaults() {
        let config = PolicyConfig::absolute();
        assert!((config.pressure_emergency - 0.9).abs() < 1e-6);
        assert!((config.pressure_aggressive_ceiling - 1.0).abs() < 1e-6);
        assert!((config.frag_defrag_threshold - 0.5).abs() < 1e-6);
        assert_eq!(config.queue_aggressive_trigger, usize::MAX);
        assert_eq!(config.batch_safe, 32);
        assert_eq!(config.batch_normal, 32);
        assert_eq!(config.batch_aggressive, 32);
    }

    #[test]
    fn policy_config_debug() {
        let config = PolicyConfig::absolute();
        let debug = format!("{config:?}");
        assert!(debug.contains("pressure_emergency"));
        assert!(debug.contains("batch_safe"));
    }

    #[test]
    fn policy_config_clone() {
        let a = PolicyConfig::absolute();
        let b = a.clone();
        assert_eq!(a.batch_safe, b.batch_safe);
        assert_eq!(a.pressure_emergency.to_bits(), b.pressure_emergency.to_bits());
    }

    // ── AbsolutePolicy ──

    #[test]
    fn absolute_policy_default_uses_absolute_config() {
        let policy = AbsolutePolicy::default();
        let d = policy.decide(&state_with(0.3, 0.1, 0, 4));
        assert_eq!(d.max_batch_size, 32);
    }

    #[test]
    fn emergency_force_swap_scales_with_pressure() {
        let d_low = AbsolutePolicy::default().decide(&state_with(0.91, 0.0, 0, 4));
        let d_high = AbsolutePolicy::default().decide(&state_with(0.99, 0.0, 0, 4));
        assert!(d_high.force_swap_out_count >= d_low.force_swap_out_count);
    }

    #[test]
    fn emergency_zero_running_gives_min_batch_one() {
        let d = AbsolutePolicy::default().decide(&state_with(0.95, 0.0, 0, 0));
        assert_eq!(d.max_batch_size, 1);
    }

    // ── PolicyVariant ──

    #[test]
    fn policy_variant_default_is_absolute() {
        let v = PolicyVariant::default();
        let d = v.decide(&state_with(0.3, 0.1, 0, 2));
        assert!(d.admit_new_prefill);
    }

    #[test]
    fn policy_variant_clone() {
        let a = PolicyVariant::default();
        let b = a.clone();
        let state = state_with(0.3, 0.1, 0, 2);
        let da = a.decide(&state);
        let db = b.decide(&state);
        assert_eq!(da.max_batch_size, db.max_batch_size);
    }

    // ── Entropy/sparsity integration ──

    #[test]
    fn high_entropy_reduces_batch() {
        let mut state = state_with(0.3, 0.1, 0, 4);
        state.logits_entropy = 10.0;
        let d = AbsolutePolicy::default().decide(&state);
        assert!(d.max_batch_size <= 4, "high entropy should cap batch to running len");
    }

    #[test]
    fn high_sparsity_reduces_batch() {
        let mut state = state_with(0.3, 0.1, 0, 10);
        state.attention_sparsity = 0.9;
        let d = AbsolutePolicy::default().decide(&state);
        assert!(d.max_batch_size < 32, "high sparsity should reduce batch");
    }

    #[test]
    fn boundary_pressure_exactly_emergency() {
        // pressure_emergency=0.9, pressure=0.9 → NOT > 0.9 → safe mode
        let d = AbsolutePolicy::default().decide(&state_with(0.9, 0.1, 0, 4));
        assert!(d.admit_new_prefill, "exactly at threshold should be safe");
    }

    // ── PolicyConfig field construction & edge values ──

    #[test]
    fn policy_config_manual_construction() {
        let config = PolicyConfig {
            pressure_emergency: 0.8,
            pressure_aggressive_ceiling: 0.95,
            frag_defrag_threshold: 0.4,
            queue_aggressive_trigger: 64,
            batch_safe: 16,
            batch_normal: 48,
            batch_aggressive: 96,
        };
        assert!((config.pressure_emergency - 0.8).abs() < 1e-6);
        assert!((config.pressure_aggressive_ceiling - 0.95).abs() < 1e-6);
        assert!((config.frag_defrag_threshold - 0.4).abs() < 1e-6);
        assert_eq!(config.queue_aggressive_trigger, 64);
        assert_eq!(config.batch_safe, 16);
        assert_eq!(config.batch_normal, 48);
        assert_eq!(config.batch_aggressive, 96);
    }

    #[test]
    fn policy_config_zero_thresholds() {
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            pressure_aggressive_ceiling: 0.0,
            frag_defrag_threshold: 0.0,
            queue_aggressive_trigger: 0,
            batch_safe: 0,
            batch_normal: 0,
            batch_aggressive: 0,
        };
        assert_eq!(config.pressure_emergency, 0.0);
        assert_eq!(config.frag_defrag_threshold, 0.0);
        assert_eq!(config.batch_safe, 0);
    }

    #[test]
    fn policy_config_max_thresholds() {
        let config = PolicyConfig {
            pressure_emergency: f32::MAX,
            pressure_aggressive_ceiling: f32::MAX,
            frag_defrag_threshold: f32::MAX,
            queue_aggressive_trigger: usize::MAX,
            batch_safe: usize::MAX,
            batch_normal: usize::MAX,
            batch_aggressive: usize::MAX,
        };
        assert_eq!(config.pressure_emergency, f32::MAX);
        assert_eq!(config.batch_safe, usize::MAX);
    }

    #[test]
    fn policy_config_clone_independence() {
        let a = PolicyConfig::absolute();
        let b = a.clone();
        // Mutate a copy to prove independence
        let mut mutated = a.clone();
        mutated.batch_safe = 999;
        assert_eq!(b.batch_safe, 32, "clone must be independent");
        assert_eq!(mutated.batch_safe, 999);
    }

    // ── AbsolutePolicy::with_config ──

    #[test]
    fn absolute_policy_with_config_custom_thresholds() {
        let config = PolicyConfig {
            pressure_emergency: 0.5,
            frag_defrag_threshold: 0.9,
            batch_safe: 8,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // pressure 0.6 > 0.5 emergency threshold
        let d = policy.decide(&state_with(0.6, 0.0, 0, 4));
        assert!(!d.admit_new_prefill, "0.6 > 0.5 should trigger emergency");
        assert_eq!(d.max_batch_size, 4);
    }

    #[test]
    fn absolute_policy_with_config_high_frag_threshold() {
        let config = PolicyConfig {
            frag_defrag_threshold: 0.99,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // frag=0.8 < 0.99 → safe mode
        let d = policy.decide(&state_with(0.3, 0.8, 0, 4));
        assert!(d.admit_new_prefill, "0.8 < 0.99 should be safe");
    }

    #[test]
    fn absolute_policy_zero_emergency_threshold() {
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Any positive pressure > 0.0 → emergency
        let d = policy.decide(&state_with(0.01, 0.0, 0, 3));
        assert!(!d.admit_new_prefill, "0.01 > 0.0 should trigger emergency");
    }

    #[test]
    fn absolute_policy_max_emergency_threshold() {
        let config = PolicyConfig {
            pressure_emergency: f32::MAX,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Even extreme pressure < f32::MAX → safe mode
        let d = policy.decide(&state_with(1000000.0, 0.0, 0, 4));
        assert!(d.admit_new_prefill, "pressure below f32::MAX threshold → safe");
    }

    // ── Defrag boundary conditions ──

    #[test]
    fn defrag_exactly_at_threshold() {
        // frag_defrag_threshold=0.5, frag=0.5 → NOT > 0.5 → safe mode
        let d = AbsolutePolicy::default().decide(&state_with(0.3, 0.5, 0, 4));
        assert!(d.admit_new_prefill, "exactly at frag threshold should be safe");
    }

    #[test]
    fn defrag_just_above_threshold() {
        // frag=0.5001 > 0.5 → defrag mode
        let d = AbsolutePolicy::default().decide(&state_with(0.3, 0.5001, 0, 4));
        assert!(!d.admit_new_prefill, "just above frag threshold should trigger defrag");
        assert_eq!(d.force_swap_out_count, 1);
    }

    #[test]
    fn defrag_zero_running_gives_min_batch_one() {
        let d = AbsolutePolicy::default().decide(&state_with(0.3, 0.6, 0, 0));
        assert_eq!(d.max_batch_size, 1, "zero running should give batch=1");
        assert!(!d.admit_new_prefill);
    }

    #[test]
    fn defrag_zero_threshold_triggers_on_any_frag() {
        let config = PolicyConfig {
            frag_defrag_threshold: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // frag=0.001 > 0.0 → defrag
        let d = policy.decide(&state_with(0.3, 0.001, 0, 5));
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
    }

    // ── Emergency force_swap_out calculation ──

    #[test]
    fn emergency_force_swap_out_exact_calculation() {
        // pressure=0.4 → (0.4 * 3.0).ceil() = 2
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(0.4, 0.0, 0, 4));
        assert_eq!(d.force_swap_out_count, 2);
    }

    #[test]
    fn emergency_force_swap_out_fractional_pressure() {
        // pressure=0.34 → (0.34 * 3.0).ceil() = (1.02).ceil() = 2
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(0.34, 0.0, 0, 4));
        assert_eq!(d.force_swap_out_count, 2);
    }

    #[test]
    fn emergency_force_swap_out_very_high_pressure() {
        // pressure=10.0 → (10.0 * 3.0).ceil() = 30
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(10.0, 0.0, 0, 4));
        assert_eq!(d.force_swap_out_count, 30);
    }

    // ── Entropy edge cases ──

    #[test]
    fn entropy_exactly_at_threshold() {
        // logits_entropy=8.0 → NOT > 8.0 → no entropy cap, uses batch_safe
        let mut state = state_with(0.3, 0.1, 0, 100);
        state.logits_entropy = 8.0;
        let d = AbsolutePolicy::default().decide(&state);
        assert_eq!(d.max_batch_size, 32, "entropy exactly 8.0 should not cap");
    }

    #[test]
    fn entropy_just_above_threshold() {
        // logits_entropy=8.001 → > 8.0 → entropy cap = min(running, batch_safe)
        let mut state = state_with(0.3, 0.1, 0, 4);
        state.logits_entropy = 8.001;
        let d = AbsolutePolicy::default().decide(&state);
        assert_eq!(d.max_batch_size, 4, "high entropy should cap to running_len");
    }

    #[test]
    fn entropy_high_but_running_exceeds_batch_safe() {
        let mut state = state_with(0.3, 0.1, 0, 100);
        state.logits_entropy = 10.0;
        let d = AbsolutePolicy::default().decide(&state);
        assert_eq!(d.max_batch_size, 32, "entropy cap = min(100, 32) = 32");
    }

    #[test]
    fn entropy_zero_value() {
        let mut state = state_with(0.3, 0.1, 0, 4);
        state.logits_entropy = 0.0;
        let d = AbsolutePolicy::default().decide(&state);
        assert_eq!(d.max_batch_size, 32, "zero entropy → no cap");
    }

    // ── Sparsity edge cases ──

    #[test]
    fn sparsity_exactly_at_threshold() {
        // attention_sparsity=0.7 → NOT > 0.7 → no sparsity cap
        let mut state = state_with(0.3, 0.1, 0, 4);
        state.attention_sparsity = 0.7;
        let d = AbsolutePolicy::default().decide(&state);
        assert_eq!(d.max_batch_size, 32, "sparsity exactly 0.7 should not cap");
    }

    #[test]
    fn sparsity_just_above_threshold() {
        // attention_sparsity=0.71 → > 0.7 → sparsity cap active
        let mut state = state_with(0.3, 0.1, 0, 10);
        state.attention_sparsity = 0.71;
        let d = AbsolutePolicy::default().decide(&state);
        // sparsity_cap = (32 * (1.0 - 0.71*0.5)).max(10) = (32*0.645).max(10) = 20.max(10) = 20
        assert!(d.max_batch_size < 32, "sparsity above threshold should reduce batch");
    }

    #[test]
    fn sparsity_near_one_reduces_heavily() {
        // attention_sparsity=0.99 → cap = (32 * (1.0 - 0.99*0.5)).max(running) = (32*0.505).max(running)
        let mut state = state_with(0.3, 0.1, 0, 2);
        state.attention_sparsity = 0.99;
        let d = AbsolutePolicy::default().decide(&state);
        assert!(d.max_batch_size <= 32);
        assert!(d.max_batch_size >= 2, "should be at least running_len");
    }

    #[test]
    fn sparsity_one_with_zero_running() {
        let mut state = state_with(0.3, 0.1, 0, 0);
        state.attention_sparsity = 1.0;
        let d = AbsolutePolicy::default().decide(&state);
        // reduced = (32 * (1.0 - 1.0*0.5)) = 16, then .max(0.max(1)) = max(16, 1) = 16
        assert_eq!(d.max_batch_size, 16);
    }

    // ── Combined entropy + sparsity ──

    #[test]
    fn entropy_and_sparsity_both_high_take_minimum() {
        let mut state = state_with(0.3, 0.1, 0, 2);
        state.logits_entropy = 10.0;
        state.attention_sparsity = 0.9;
        let d = AbsolutePolicy::default().decide(&state);
        // entropy_cap = min(2, 32) = 2
        // sparsity_cap = (32 * (1.0 - 0.9*0.5)) = (32*0.55) = 17, .max(2) = 17
        // nominal_batch = min(2, 17) = 2
        assert_eq!(d.max_batch_size, 2);
    }

    #[test]
    fn entropy_high_sparsity_low() {
        let mut state = state_with(0.3, 0.1, 0, 4);
        state.logits_entropy = 12.0;
        state.attention_sparsity = 0.1;
        let d = AbsolutePolicy::default().decide(&state);
        // entropy_cap = min(4, 32) = 4
        // sparsity_cap = 32 (below threshold)
        // nominal_batch = min(4, 32) = 4
        assert_eq!(d.max_batch_size, 4);
    }

    // ── Priority: emergency > defrag > safe ──

    #[test]
    fn emergency_takes_priority_over_defrag() {
        // Both emergency and defrag thresholds exceeded → emergency wins
        let d = AbsolutePolicy::default().decide(&state_with(0.95, 0.8, 0, 5));
        assert!(!d.admit_new_prefill);
        assert!(d.force_swap_out_count > 1, "emergency should scale swap_out");
    }

    #[test]
    fn defrag_takes_priority_over_entropy() {
        // Defrag triggered, entropy high → defrag decision, batch = running
        let mut state = state_with(0.3, 0.8, 0, 4);
        state.logits_entropy = 10.0;
        let d = AbsolutePolicy::default().decide(&state);
        // defrag: batch = max(4, 1) = 4, admit=false, swap=1
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
        assert_eq!(d.max_batch_size, 4);
    }

    // ── PolicyVariant exhaustive coverage ──

    #[test]
    fn policy_variant_clone_matches_original() {
        let original = PolicyVariant::Absolute;
        let cloned = original.clone();
        let state = state_with(0.3, 0.1, 0, 3);
        let d_orig = original.decide(&state);
        let d_clone = cloned.decide(&state);
        assert_eq!(d_orig, d_clone);
    }

    #[test]
    fn policy_variant_default_matches_absolute() {
        let variant = PolicyVariant::default();
        let absolute = AbsolutePolicy::default();
        let state = state_with(0.5, 0.3, 10, 4);
        assert_eq!(variant.decide(&state), absolute.decide(&state));
    }

    // ── SchedulingPolicy trait object ──

    #[test]
    fn scheduling_policy_trait_object_dispatch() {
        let policy: Box<dyn SchedulingPolicy> = Box::new(AbsolutePolicy::default());
        let state = state_with(0.95, 0.0, 0, 3);
        let d = policy.decide(&state);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 3);
    }

    #[test]
    fn scheduling_policy_trait_object_safe_mode() {
        let policy: Box<dyn SchedulingPolicy> = Box::new(AbsolutePolicy::default());
        let state = state_with(0.3, 0.1, 5, 4);
        let d = policy.decide(&state);
        assert!(d.admit_new_prefill);
    }

    // ── All-zero state ──

    #[test]
    fn safe_mode_with_all_zero_state() {
        let d = AbsolutePolicy::default().decide(&SystemState::default());
        // pressure=0, frag=0, entropy=0, sparsity=0, running=0
        // entropy_cap: 0 not > 8.0 → batch_safe=32
        // sparsity_cap: 0 not > 0.7 → batch_safe=32
        // nominal_batch = 32
        // safe mode: batch=32, admit=true, swap=0
        assert_eq!(d.max_batch_size, 32);
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    #[test]
    fn emergency_with_all_zero_state_and_zero_threshold() {
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // pressure=0.0 NOT > 0.0 → safe mode (not emergency)
        let d = policy.decide(&SystemState::default());
        assert!(d.admit_new_prefill, "0.0 > 0.0 is false → safe mode");
    }

    // ══════════════════════════════════════════════════════════════════════
    //  ~40 NEW TESTS: additional coverage for public types & methods
    // ══════════════════════════════════════════════════════════════════════

    // ── PolicyConfig: all fields readable after absolute() ──

    #[test]
    fn policy_config_absolute_pressure_aggressive_ceiling_is_one() {
        let config = PolicyConfig::absolute();
        assert!((config.pressure_aggressive_ceiling - 1.0).abs() < 1e-6);
    }

    #[test]
    fn policy_config_absolute_all_batch_sizes_equal() {
        let config = PolicyConfig::absolute();
        assert_eq!(
            config.batch_safe, config.batch_normal,
            "absolute mode: batch_safe == batch_normal"
        );
        assert_eq!(
            config.batch_normal, config.batch_aggressive,
            "absolute mode: batch_normal == batch_aggressive"
        );
    }

    #[test]
    fn policy_config_absolute_queue_trigger_is_max() {
        let config = PolicyConfig::absolute();
        assert_eq!(config.queue_aggressive_trigger, usize::MAX);
    }

    // ── PolicyConfig: field mutation independence ──

    #[test]
    fn policy_config_field_mutation_does_not_affect_clone() {
        let mut a = PolicyConfig::absolute();
        let b = a.clone();
        a.pressure_emergency = 0.0;
        a.batch_safe = 0;
        assert!(
            (b.pressure_emergency - 0.9).abs() < 1e-6,
            "clone must preserve original value"
        );
        assert_eq!(b.batch_safe, 32);
    }

    #[test]
    fn policy_config_negative_thresholds_allowed() {
        // PolicyConfig has no validation — negative values are structurally legal
        let config = PolicyConfig {
            pressure_emergency: -1.0,
            frag_defrag_threshold: -0.5,
            ..PolicyConfig::absolute()
        };
        assert!(config.pressure_emergency < 0.0);
        assert!(config.frag_defrag_threshold < 0.0);
    }

    #[test]
    fn policy_config_nan_threshold() {
        let config = PolicyConfig {
            pressure_emergency: f32::NAN,
            ..PolicyConfig::absolute()
        };
        assert!(config.pressure_emergency.is_nan());
    }

    #[test]
    fn policy_config_infinity_threshold() {
        let config = PolicyConfig {
            pressure_emergency: f32::INFINITY,
            frag_defrag_threshold: f32::NEG_INFINITY,
            ..PolicyConfig::absolute()
        };
        assert!(config.pressure_emergency.is_infinite());
        assert!(config.frag_defrag_threshold.is_infinite());
    }

    // ── PolicyConfig: Debug includes all fields ──

    #[test]
    fn policy_config_debug_includes_all_fields() {
        let config = PolicyConfig::absolute();
        let debug = format!("{config:?}");
        assert!(debug.contains("pressure_emergency"));
        assert!(debug.contains("pressure_aggressive_ceiling"));
        assert!(debug.contains("frag_defrag_threshold"));
        assert!(debug.contains("queue_aggressive_trigger"));
        assert!(debug.contains("batch_safe"));
        assert!(debug.contains("batch_normal"));
        assert!(debug.contains("batch_aggressive"));
    }

    // ── AbsolutePolicy: with_config stores config ──

    #[test]
    fn absolute_policy_with_config_preserves_custom_batch_safe() {
        let config = PolicyConfig {
            batch_safe: 64,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(0.1, 0.1, 0, 100));
        assert_eq!(d.max_batch_size, 64);
    }

    #[test]
    fn absolute_policy_with_config_preserves_custom_frag_threshold() {
        let config = PolicyConfig {
            frag_defrag_threshold: 0.1,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // frag=0.2 > 0.1 → defrag mode
        let d = policy.decide(&state_with(0.1, 0.2, 0, 4));
        assert!(!d.admit_new_prefill);
    }

    #[test]
    fn absolute_policy_with_config_nan_emergency_never_triggers() {
        // NaN comparisons always return false, so pressure > NaN → false
        let config = PolicyConfig {
            pressure_emergency: f32::NAN,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(f32::MAX, 0.0, 0, 4));
        assert!(
            d.admit_new_prefill,
            "NaN threshold → pressure comparison always false → safe mode"
        );
    }

    #[test]
    fn absolute_policy_with_config_nan_frag_never_triggers() {
        let config = PolicyConfig {
            frag_defrag_threshold: f32::NAN,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(0.1, f32::MAX, 0, 4));
        assert!(
            d.admit_new_prefill,
            "NaN frag threshold → comparison always false → safe mode"
        );
    }

    // ── AbsolutePolicy: multiple decide calls are idempotent ──

    #[test]
    fn absolute_policy_decide_is_idempotent() {
        let policy = AbsolutePolicy::default();
        let state = state_with(0.5, 0.3, 10, 8);
        let d1 = policy.decide(&state);
        let d2 = policy.decide(&state);
        let d3 = policy.decide(&state);
        assert_eq!(d1, d2);
        assert_eq!(d2, d3);
    }

    // ── AbsolutePolicy: varying running_len ──

    #[test]
    fn emergency_batch_size_tracks_running_len() {
        for running in [1, 5, 10, 50, 100] {
            let d = AbsolutePolicy::default().decide(&state_with(0.95, 0.0, 0, running));
            assert_eq!(
                d.max_batch_size, running,
                "emergency mode: batch should equal running_len"
            );
        }
    }

    #[test]
    fn defrag_batch_size_tracks_running_len() {
        for running in [1, 5, 10, 50, 100] {
            let d = AbsolutePolicy::default().decide(&state_with(0.5, 0.6, 0, running));
            assert_eq!(
                d.max_batch_size, running,
                "defrag mode: batch should equal running_len"
            );
        }
    }

    // ── AbsolutePolicy: state with different queue lengths ──

    #[test]
    fn safe_mode_independent_of_queue_len() {
        // AbsolutePolicy safe mode doesn't use queue length
        for queue in [0, 10, 100, usize::MAX] {
            let d = AbsolutePolicy::default().decide(&state_with(0.1, 0.1, queue, 4));
            assert!(
                d.admit_new_prefill,
                "safe mode should admit regardless of queue len"
            );
            assert_eq!(d.max_batch_size, 32);
        }
    }

    // ── AbsolutePolicy: pressure at 0.0 and 1.0 boundaries ──

    #[test]
    fn pressure_zero_safe_mode() {
        let d = AbsolutePolicy::default().decide(&state_with(0.0, 0.0, 0, 4));
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    #[test]
    fn pressure_one_emergency_mode() {
        // 1.0 > 0.9 → emergency
        let d = AbsolutePolicy::default().decide(&state_with(1.0, 0.0, 0, 4));
        assert!(!d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 4);
    }

    // ── Entropy: continuous property check ──

    #[test]
    fn entropy_batch_monotonically_nonincreasing() {
        // As entropy increases, batch should not increase
        let mut prev_batch = usize::MAX;
        for entropy in [0.0, 4.0, 8.0, 9.0, 12.0, 20.0] {
            let mut state = state_with(0.1, 0.1, 0, 4);
            state.logits_entropy = entropy;
            let d = AbsolutePolicy::default().decide(&state);
            assert!(
                d.max_batch_size <= prev_batch,
                "entropy={entropy}: batch={} > prev={prev_batch}",
                d.max_batch_size
            );
            prev_batch = d.max_batch_size;
        }
    }

    #[test]
    fn entropy_cap_never_exceeds_batch_safe() {
        for entropy in [0.0, 5.0, 8.0, 10.0, 100.0] {
            let mut state = state_with(0.1, 0.1, 0, 200);
            state.logits_entropy = entropy;
            let d = AbsolutePolicy::default().decide(&state);
            assert!(
                d.max_batch_size <= 32,
                "entropy={entropy}: batch {} > batch_safe=32",
                d.max_batch_size
            );
        }
    }

    // ── Sparsity: continuous property check ──

    #[test]
    fn sparsity_cap_monotonically_nonincreasing() {
        let mut prev_batch = usize::MAX;
        for sparsity in [0.0, 0.5, 0.7, 0.8, 0.95, 1.0] {
            let mut state = state_with(0.1, 0.1, 0, 2);
            state.attention_sparsity = sparsity;
            let d = AbsolutePolicy::default().decide(&state);
            assert!(
                d.max_batch_size <= prev_batch,
                "sparsity={sparsity}: batch={} > prev={prev_batch}",
                d.max_batch_size
            );
            prev_batch = d.max_batch_size;
        }
    }

    #[test]
    fn sparsity_cap_never_below_running_len() {
        // sparsity reduction has .max(running_len.max(1)) guard
        for sparsity in [0.0, 0.5, 0.7, 0.8, 0.9, 1.0] {
            let mut state = state_with(0.1, 0.1, 0, 10);
            state.attention_sparsity = sparsity;
            let d = AbsolutePolicy::default().decide(&state);
            assert!(
                d.max_batch_size >= 10,
                "sparsity={sparsity}: batch {} < running_len=10",
                d.max_batch_size
            );
        }
    }

    // ── Emergency: swap_out formula ──

    #[test]
    fn emergency_swap_out_formula_property() {
        // swap_out = ceil(pressure * 3.0)
        // For integer pressures, result = pressure * 3
        for pressure_int in 1..=10u32 {
            let pressure = pressure_int as f32;
            let d = AbsolutePolicy::default().decide(&state_with(
                // Ensure emergency triggers: must be > 0.9
                0.9 + pressure * 0.01,
                0.0,
                0,
                4,
            ));
            assert!(
                d.force_swap_out_count > 0,
                "pressure={pressure}: swap_out should be > 0"
            );
        }
    }

    #[test]
    fn emergency_swap_out_zero_pressure_at_zero_threshold() {
        // pressure=0.0, threshold=0.0 → NOT > → no emergency → swap=0
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(0.0, 0.0, 0, 4));
        assert_eq!(d.force_swap_out_count, 0, "no emergency → no swap");
    }

    // ── Defrag: always swap_out_count = 1 ──

    #[test]
    fn defrag_swap_out_always_one() {
        for frag in [0.51, 0.6, 0.8, 0.99, 1.0] {
            let d = AbsolutePolicy::default().decide(&state_with(0.1, frag, 0, 4));
            if !d.admit_new_prefill {
                assert_eq!(
                    d.force_swap_out_count, 1,
                    "defrag mode: swap_out should always be 1, frag={frag}"
                );
            }
        }
    }

    // ── SystemState fields integration with policy ──

    #[test]
    fn policy_ignores_swap_io_rate() {
        // swap_io_rate is not used by AbsolutePolicy.decide
        let mut state_a = state_with(0.1, 0.1, 0, 4);
        let mut state_b = state_a;
        state_a.swap_io_rate = 0.0;
        state_b.swap_io_rate = 1000000.0;
        let d_a = AbsolutePolicy::default().decide(&state_a);
        let d_b = AbsolutePolicy::default().decide(&state_b);
        assert_eq!(d_a, d_b, "swap_io_rate should not affect AbsolutePolicy");
    }

    #[test]
    fn policy_ignores_mean_context_len() {
        let mut state_a = state_with(0.1, 0.1, 0, 4);
        let mut state_b = state_a;
        state_a.mean_context_len = 0;
        state_b.mean_context_len = 100000;
        let d_a = AbsolutePolicy::default().decide(&state_a);
        let d_b = AbsolutePolicy::default().decide(&state_b);
        assert_eq!(d_a, d_b, "mean_context_len should not affect AbsolutePolicy");
    }

    #[test]
    fn policy_ignores_moe_metrics() {
        let mut state_a = state_with(0.1, 0.1, 0, 4);
        let mut state_b = state_a;
        state_b.moe_fault_rate = 0.5;
        state_b.moe_avg_recovery_us = 1000.0;
        state_b.moe_working_set_size = 64;
        let d_a = AbsolutePolicy::default().decide(&state_a);
        let d_b = AbsolutePolicy::default().decide(&state_b);
        assert_eq!(d_a, d_b, "MoE metrics should not affect AbsolutePolicy");
    }

    #[test]
    fn policy_ignores_weight_page_metrics() {
        let mut state_a = state_with(0.1, 0.1, 0, 4);
        let mut state_b = state_a;
        state_b.weight_page_total = 10000;
        state_b.weight_pages_l1 = 5000;
        state_b.weight_pages_l2 = 3000;
        state_b.weight_pages_l3 = 2000;
        state_b.weight_eviction_count = 999;
        state_b.weight_recovery_count = 888;
        let d_a = AbsolutePolicy::default().decide(&state_a);
        let d_b = AbsolutePolicy::default().decide(&state_b);
        assert_eq!(d_a, d_b, "weight page metrics should not affect AbsolutePolicy");
    }

    #[test]
    fn policy_ignores_current_batch_size() {
        let mut state_a = state_with(0.1, 0.1, 0, 4);
        let mut state_b = state_a;
        state_a.current_batch_size = 1;
        state_b.current_batch_size = 200;
        let d_a = AbsolutePolicy::default().decide(&state_a);
        let d_b = AbsolutePolicy::default().decide(&state_b);
        assert_eq!(d_a, d_b, "current_batch_size should not affect AbsolutePolicy");
    }

    // ── SchedulingPolicy trait: multiple trait objects ──

    #[test]
    fn scheduling_policy_two_trait_objects_same_decision() {
        let p1: Box<dyn SchedulingPolicy> = Box::new(AbsolutePolicy::default());
        let p2: Box<dyn SchedulingPolicy> = Box::new(AbsolutePolicy::default());
        let state = state_with(0.3, 0.1, 5, 4);
        assert_eq!(p1.decide(&state), p2.decide(&state));
    }

    #[test]
    fn scheduling_policy_trait_object_emergency() {
        let policy: Box<dyn SchedulingPolicy> = Box::new(AbsolutePolicy::with_config(
            PolicyConfig {
                pressure_emergency: 0.5,
                ..PolicyConfig::absolute()
            },
        ));
        let state = state_with(0.6, 0.0, 0, 8);
        let d = policy.decide(&state);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 8);
    }

    #[test]
    fn scheduling_policy_trait_object_defrag() {
        let policy: Box<dyn SchedulingPolicy> = Box::new(AbsolutePolicy::with_config(
            PolicyConfig {
                frag_defrag_threshold: 0.3,
                ..PolicyConfig::absolute()
            },
        ));
        let state = state_with(0.1, 0.5, 0, 6);
        let d = policy.decide(&state);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 6);
    }

    // ── PolicyVariant: exhaustive match over variants ──

    #[test]
    fn policy_variant_all_variants_produce_valid_decision() {
        let variants: Vec<PolicyVariant> = vec![PolicyVariant::Absolute];
        let state = state_with(0.3, 0.1, 5, 4);
        for v in &variants {
            let d = v.decide(&state);
            assert!(
                d.max_batch_size > 0,
                "all variants must produce positive batch size"
            );
        }
    }

    #[test]
    fn policy_variant_absolute_matches_trait_object() {
        let variant = PolicyVariant::Absolute;
        let trait_obj: Box<dyn SchedulingPolicy> = Box::new(AbsolutePolicy::default());
        let state = state_with(0.5, 0.3, 20, 10);
        assert_eq!(variant.decide(&state), trait_obj.decide(&state));
    }

    // ── PolicyVariant: default + clone roundtrip ──

    #[test]
    fn policy_variant_default_clone_roundtrip() {
        let a = PolicyVariant::default();
        let b = a.clone();
        let c = b.clone();
        let state = state_with(0.95, 0.0, 0, 3);
        assert_eq!(a.decide(&state), b.decide(&state));
        assert_eq!(b.decide(&state), c.decide(&state));
    }

    // ── Safe mode: entropy and sparsity both zero ──

    #[test]
    fn safe_mode_both_entropy_and_sparsity_zero() {
        let mut state = state_with(0.1, 0.1, 0, 50);
        state.logits_entropy = 0.0;
        state.attention_sparsity = 0.0;
        let d = AbsolutePolicy::default().decide(&state);
        assert_eq!(d.max_batch_size, 32);
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    // ── Safe mode: very large running_len with custom batch_safe ──

    #[test]
    fn safe_mode_large_running_respects_batch_safe_cap() {
        let config = PolicyConfig {
            batch_safe: 16,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(0.1, 0.1, 0, 10000));
        assert_eq!(
            d.max_batch_size, 16,
            "safe mode should cap at batch_safe regardless of running_len"
        );
    }

    // ── Custom config: very small batch_safe ──

    #[test]
    fn safe_mode_batch_safe_one() {
        let config = PolicyConfig {
            batch_safe: 1,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(0.1, 0.1, 0, 100));
        assert_eq!(d.max_batch_size, 1);
    }

    // ── Emergency + high entropy interaction ──

    #[test]
    fn emergency_ignores_entropy_cap() {
        // Emergency mode uses running_len directly, not nominal_batch
        let mut state = state_with(0.95, 0.0, 0, 8);
        state.logits_entropy = 20.0; // very high
        let d = AbsolutePolicy::default().decide(&state);
        assert_eq!(d.max_batch_size, 8, "emergency should use running_len, not entropy cap");
    }

    #[test]
    fn defrag_ignores_entropy_cap() {
        // Defrag mode uses running_len directly
        let mut state = state_with(0.1, 0.8, 0, 12);
        state.logits_entropy = 20.0;
        let d = AbsolutePolicy::default().decide(&state);
        assert_eq!(
            d.max_batch_size, 12,
            "defrag should use running_len, not entropy cap"
        );
    }

    #[test]
    fn defrag_ignores_sparsity_cap() {
        let mut state = state_with(0.1, 0.8, 0, 7);
        state.attention_sparsity = 0.99;
        let d = AbsolutePolicy::default().decide(&state);
        assert_eq!(
            d.max_batch_size, 7,
            "defrag should use running_len, not sparsity cap"
        );
    }

    // ── Property: force_swap_out_count is always 0 in safe mode ──

    #[test]
    fn safe_mode_always_zero_swap_out() {
        for pressure in [0.0, 0.3, 0.5, 0.89] {
            for frag in [0.0, 0.3, 0.49] {
                let d = AbsolutePolicy::default().decide(&state_with(pressure, frag, 0, 4));
                if d.admit_new_prefill {
                    assert_eq!(
                        d.force_swap_out_count, 0,
                        "safe mode: swap_out should always be 0, p={pressure}, f={frag}"
                    );
                }
            }
        }
    }

    // ── Property: non-safe modes never admit prefill ──

    #[test]
    fn emergency_and_defrag_never_admit_prefill() {
        // Emergency: pressure > 0.9
        let d_emergency = AbsolutePolicy::default().decide(&state_with(0.95, 0.0, 0, 4));
        assert!(!d_emergency.admit_new_prefill);

        // Defrag: frag > 0.5, pressure not emergency
        let d_defrag = AbsolutePolicy::default().decide(&state_with(0.5, 0.6, 0, 4));
        assert!(!d_defrag.admit_new_prefill);
    }

    // ── NaN pressure state in emergency check ──

    #[test]
    fn nan_pressure_does_not_trigger_emergency() {
        // Arrange: NaN pressure — NaN > 0.9 is false per IEEE 754
        let mut state = state_with(0.0, 0.0, 0, 4);
        state.memory_pressure = f32::NAN;
        let policy = AbsolutePolicy::default();
        // Act
        let d = policy.decide(&state);
        // Assert: NaN comparison fails → falls through to safe mode
        assert!(
            d.admit_new_prefill,
            "NaN pressure should not trigger emergency, should fall to safe mode"
        );
        assert_eq!(d.force_swap_out_count, 0);
    }

    // ── Config with pressure_emergency > 1.0 (super-normalized threshold) ──

    #[test]
    fn super_normalized_emergency_threshold_never_triggers() {
        // Arrange: threshold at 1.5 means pressure in [0,1] range can never trigger
        let config = PolicyConfig {
            pressure_emergency: 1.5,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Act: pressure at maximum valid value 1.0
        let d = policy.decide(&state_with(1.0, 0.0, 0, 4));
        // Assert: 1.0 NOT > 1.5 → safe mode
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    // ── Emergency with very small positive pressure just above zero threshold ──

    #[test]
    fn emergency_tiny_pressure_with_zero_threshold() {
        // Arrange: threshold=0.0, pressure=f32 smallest positive subnormal
        let tiny = f32::from_bits(1u32);
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Act
        let d = policy.decide(&state_with(tiny, 0.0, 0, 3));
        // Assert: tiny > 0.0 → emergency mode
        assert!(!d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 3);
        // swap_out = ceil(tiny * 3.0) — tiny * 3.0 is still subnormal, ceil gives 1
        assert!(d.force_swap_out_count >= 1);
    }

    // ── Safe mode with both entropy and sparsity just below their thresholds ──

    #[test]
    fn safe_mode_entropy_and_sparsity_just_below_thresholds() {
        // Arrange: entropy just below 8.0, sparsity just below 0.7
        let mut state = state_with(0.1, 0.1, 0, 4);
        state.logits_entropy = 7.999;
        state.attention_sparsity = 0.699;
        let policy = AbsolutePolicy::default();
        // Act
        let d = policy.decide(&state);
        // Assert: neither threshold crossed → full batch_safe
        assert_eq!(d.max_batch_size, 32);
        assert!(d.admit_new_prefill);
    }

    // ── Config struct update syntax from absolute() ──

    #[test]
    fn policy_config_struct_update_from_absolute() {
        // Arrange: override only 2 fields, inherit the rest
        let config = PolicyConfig {
            batch_safe: 64,
            pressure_emergency: 0.8,
            ..PolicyConfig::absolute()
        };
        // Assert: overridden fields
        assert!((config.pressure_emergency - 0.8).abs() < 1e-6);
        assert_eq!(config.batch_safe, 64);
        // Assert: inherited fields from absolute()
        assert!((config.pressure_aggressive_ceiling - 1.0).abs() < 1e-6);
        assert!((config.frag_defrag_threshold - 0.5).abs() < 1e-6);
        assert_eq!(config.queue_aggressive_trigger, usize::MAX);
        assert_eq!(config.batch_normal, 32);
        assert_eq!(config.batch_aggressive, 32);
    }

    // ── Emergency mode ignores sparsity entirely ──

    #[test]
    fn emergency_ignores_sparsity() {
        // Arrange: emergency pressure, max sparsity
        let mut state = state_with(0.95, 0.0, 0, 6);
        state.attention_sparsity = 1.0;
        let d = AbsolutePolicy::default().decide(&state);
        // Assert: emergency uses running_len directly, sparsity has no effect
        assert_eq!(d.max_batch_size, 6);
        assert!(!d.admit_new_prefill);
    }

    // ── Custom config with very large batch_safe and high entropy ──

    #[test]
    fn large_batch_safe_with_high_entropy_caps_at_running_len() {
        // Arrange: batch_safe=1000, entropy triggers cap
        let config = PolicyConfig {
            batch_safe: 1000,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let mut state = state_with(0.1, 0.1, 0, 50);
        state.logits_entropy = 10.0;
        // Act
        let d = policy.decide(&state);
        // Assert: entropy cap = min(50, 1000) = 50
        assert_eq!(d.max_batch_size, 50);
    }

    // ── Defrag with max frag threshold config ──

    #[test]
    fn defrag_with_max_threshold_never_triggers() {
        // Arrange: frag threshold at f32::MAX — no realistic frag can exceed it
        let config = PolicyConfig {
            frag_defrag_threshold: f32::MAX,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Act: frag at 1.0 (maximum valid)
        let d = policy.decide(&state_with(0.1, 1.0, 0, 4));
        // Assert: 1.0 NOT > f32::MAX → safe mode
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    // ── Safe mode: entropy caps batch below running_len when batch_safe is small ──

    #[test]
    fn entropy_caps_when_running_exceeds_small_batch_safe() {
        // Arrange: batch_safe=8, high entropy, running=20
        let config = PolicyConfig {
            batch_safe: 8,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let mut state = state_with(0.1, 0.1, 0, 20);
        state.logits_entropy = 10.0;
        // Act
        let d = policy.decide(&state);
        // Assert: entropy_cap = min(20, 8) = 8
        assert_eq!(d.max_batch_size, 8);
    }

    // ── Safe mode: extremely high sparsity with zero running ──

    #[test]
    fn sparsity_max_with_zero_running_and_small_batch_safe() {
        // Arrange: sparsity=1.0, running=0, batch_safe=8
        let config = PolicyConfig {
            batch_safe: 8,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let mut state = state_with(0.1, 0.1, 0, 0);
        state.attention_sparsity = 1.0;
        // Act
        let d = policy.decide(&state);
        // reduced = (8 * (1.0 - 1.0*0.5)) = 4, .max(0.max(1)) = max(4, 1) = 4
        assert_eq!(d.max_batch_size, 4);
    }

    // ── Pressure at negative value with default threshold ──

    #[test]
    fn negative_pressure_safe_mode() {
        // Arrange: negative pressure is below 0.9 threshold
        let d = AbsolutePolicy::default().decide(&state_with(-0.5, 0.0, 0, 4));
        // Assert: -0.5 NOT > 0.9 → safe mode
        assert!(d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 32);
        assert_eq!(d.force_swap_out_count, 0);
    }

    // ── Negative frag with zero threshold triggers defrag ──

    #[test]
    fn negative_frag_with_negative_threshold_no_trigger() {
        // Arrange: frag=-0.1, threshold=-0.5 → -0.1 > -0.5 is true → defrag
        let config = PolicyConfig {
            frag_defrag_threshold: -0.5,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        let d = policy.decide(&state_with(0.1, -0.1, 0, 4));
        // Assert: -0.1 > -0.5 → defrag mode
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
        assert_eq!(d.max_batch_size, 4);
    }

    // ── Emergency force_swap_out with very small fractional pressure ──

    #[test]
    fn emergency_swap_out_minimum_is_one() {
        // Arrange: threshold=0.0, tiny pressure
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Act: pressure=0.001 → swap = ceil(0.001*3) = ceil(0.003) = 1
        let d = policy.decide(&state_with(0.001, 0.0, 0, 4));
        // Assert: at minimum 1 swap even for tiny pressure
        assert!(d.force_swap_out_count >= 1);
    }

    // ── PolicyConfig: all batch fields set to same custom value ──

    #[test]
    fn policy_config_all_batches_same_custom_value() {
        // Arrange: all batch sizes set to 1 (minimum viable)
        let config = PolicyConfig {
            batch_safe: 1,
            batch_normal: 1,
            batch_aggressive: 1,
            ..PolicyConfig::absolute()
        };
        // Assert: all three are equal
        assert_eq!(config.batch_safe, config.batch_normal);
        assert_eq!(config.batch_normal, config.batch_aggressive);
        assert_eq!(config.batch_safe, 1);
    }

    // ── Emergency with subnormal pressure and zero threshold ──

    #[test]
    fn emergency_subnormal_pressure_swap_out_calculation() {
        // Arrange: subnormal pressure value
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Act
        let d = policy.decide(&state_with(subnormal, 0.0, 0, 5));
        // Assert: subnormal * 3.0 is still tiny, ceil gives 1
        assert_eq!(d.max_batch_size, 5);
        assert!(!d.admit_new_prefill);
        assert!(
            d.force_swap_out_count >= 1,
            "subnormal pressure should still produce at least 1 swap_out"
        );
    }

    // ── Property: batch_size >= 1 always ──

    #[test]
    fn decision_batch_size_nonzero_with_running_or_safe_batch() {
        // When running > 0, emergency/defrag guarantee batch >= running (>= 1).
        // When running == 0, safe mode uses nominal_batch which is >= batch_safe.
        // AbsolutePolicy only guarantees batch >= 1 when emergency/defrag path kicks in
        // (via running_len.max(1)) or batch_safe > 0.
        for running in [1, 5, 20] {
            for pressure in [0.0, 0.5, 1.0] {
                for frag in [0.0, 0.5, 1.0] {
                    let d = AbsolutePolicy::default().decide(&state_with(pressure, frag, 0, running));
                    assert!(
                        d.max_batch_size >= 1,
                        "batch must be >= 1: p={pressure}, f={frag}, r={running}"
                    );
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Wave 20: 10 additional tests — uncovered edge cases
    // ══════════════════════════════════════════════════════════════════════

    // ── PolicyVariant dispatches emergency and defrag correctly ──

    #[test]
    fn policy_variant_emergency_dispatch() {
        // Arrange: state that triggers emergency via PolicyVariant
        let variant = PolicyVariant::default();
        let state = state_with(0.95, 0.0, 0, 7);
        // Act
        let d = variant.decide(&state);
        // Assert: emergency — no prefill, batch equals running
        assert!(!d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 7);
        assert!(d.force_swap_out_count > 0);
    }

    #[test]
    fn policy_variant_defrag_dispatch() {
        // Arrange: state that triggers defrag via PolicyVariant
        let variant = PolicyVariant::default();
        let state = state_with(0.3, 0.7, 0, 5);
        // Act
        let d = variant.decide(&state);
        // Assert: defrag — no prefill, swap_out = 1
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
        assert_eq!(d.max_batch_size, 5);
    }

    // ── with_config(empty) vs default: same behavior ──

    #[test]
    fn with_config_absolute_equals_default() {
        // Arrange: with_config(absolute()) should behave identically to default()
        let via_default = AbsolutePolicy::default();
        let via_config = AbsolutePolicy::with_config(PolicyConfig::absolute());
        let state = state_with(0.5, 0.3, 10, 8);
        // Assert: identical decisions across all state combinations
        for pressure in [0.1, 0.95] {
            for frag in [0.1, 0.6] {
                let s = state_with(pressure, frag, 0, 4);
                assert_eq!(via_default.decide(&s), via_config.decide(&s));
            }
        }
        assert_eq!(via_default.decide(&state), via_config.decide(&state));
    }

    // ── Emergency swap_out with negative pressure gives 0 via ceil ──

    #[test]
    fn emergency_negative_pressure_swap_out_calculation() {
        // Arrange: threshold=0.0 so any positive pressure triggers emergency
        let config = PolicyConfig {
            pressure_emergency: 0.0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Negative pressure = memory is abundant, not emergency.
        // -0.5 > 0.0 is false, so policy falls to safe/defrag check.
        // frag=0.0 < 0.5 threshold → safe → admit=true.
        let d = policy.decide(&state_with(-0.5, 0.0, 0, 3));
        assert!(d.admit_new_prefill, "negative pressure means memory is abundant");
        assert_eq!(d.max_batch_size, 32, "safe mode uses batch_safe");
        assert_eq!(d.force_swap_out_count, 0);
    }

    // ── Infinity pressure triggers emergency with huge swap_out ──

    #[test]
    fn infinity_pressure_emergency_swap_out() {
        // Arrange: default threshold 0.9, pressure = +infinity
        let d = AbsolutePolicy::default().decide(&state_with(f32::INFINITY, 0.0, 0, 4));
        // Act & Assert: inf > 0.9 → emergency
        assert!(!d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 4);
        // swap_out = ceil(inf * 3.0) = inf, as usize → 0 (wrapping for inf)
        // The key invariant: emergency mode is entered.
    }

    // ── Infinity frag triggers defrag ──

    #[test]
    fn infinity_frag_triggers_defrag() {
        // Arrange: safe pressure, infinite fragmentation
        let d = AbsolutePolicy::default().decide(&state_with(0.1, f32::INFINITY, 0, 6));
        // Assert: inf > 0.5 → defrag mode
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
        assert_eq!(d.max_batch_size, 6);
    }

    // ── batch_normal and batch_aggressive are unused by AbsolutePolicy ──

    #[test]
    fn absolute_policy_ignores_batch_normal_and_aggressive() {
        // Arrange: set batch_normal and batch_aggressive to extreme values
        let config = PolicyConfig {
            batch_safe: 32,
            batch_normal: 1,
            batch_aggressive: 1000,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Act: safe mode should use batch_safe only
        let d = policy.decide(&state_with(0.1, 0.1, 0, 4));
        // Assert: batch_normal and batch_aggressive have no effect
        assert_eq!(d.max_batch_size, 32);
    }

    // ── Safe mode with batch_safe = 0 and zero running ──

    #[test]
    fn safe_mode_batch_safe_zero_running_zero_gives_zero_batch() {
        // Arrange: batch_safe=0, zero running, no entropy/sparsity
        let config = PolicyConfig {
            batch_safe: 0,
            ..PolicyConfig::absolute()
        };
        let policy = AbsolutePolicy::with_config(config);
        // Act: safe mode with nominal_batch = min(32, 32) = 32
        // entropy_cap: 0 not > 8 → batch_safe=0
        // sparsity_cap: 0 not > 0.7 → batch_safe=0
        // nominal_batch = min(0, 0) = 0
        let d = policy.decide(&state_with(0.1, 0.1, 0, 0));
        // Assert: batch_safe=0 flows through to max_batch_size
        assert_eq!(d.max_batch_size, 0);
    }

    // ── SchedulingPolicy trait: repeated calls return same result ──

    #[test]
    fn scheduling_policy_trait_repeated_calls_deterministic() {
        // Arrange: boxed trait object with a state triggering defrag
        let policy: Box<dyn SchedulingPolicy> = Box::new(AbsolutePolicy::default());
        let state = state_with(0.3, 0.6, 5, 10);
        // Act: call decide 5 times
        let decisions: Vec<_> = (0..5).map(|_| policy.decide(&state)).collect();
        // Assert: all decisions are pairwise equal
        for i in 1..decisions.len() {
            assert_eq!(decisions[i - 1], decisions[i]);
        }
        assert!(!decisions[0].admit_new_prefill);
        assert_eq!(decisions[0].max_batch_size, 10);
    }

    // ── Very large running_len (near usize::MAX) ──

    #[test]
    fn emergency_with_running_len_near_usize_max() {
        // Arrange: emergency mode with very large running_len
        let large_running = usize::MAX - 1;
        let d = AbsolutePolicy::default().decide(&state_with(0.95, 0.0, 0, large_running));
        // Assert: batch = running_len.max(1) = running_len (no overflow)
        assert_eq!(d.max_batch_size, large_running);
        assert!(!d.admit_new_prefill);
    }

    #[test]
    fn defrag_with_running_len_near_usize_max() {
        // Arrange: defrag mode with very large running_len
        let large_running = usize::MAX - 1;
        let d = AbsolutePolicy::default().decide(&state_with(0.3, 0.6, 0, large_running));
        // Assert: batch = running_len.max(1) = running_len (no overflow)
        assert_eq!(d.max_batch_size, large_running);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
    }
}
