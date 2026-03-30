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
    pub fn accuracy_first() -> Self {
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

/// Accuracy First Policy (Single-Track Scheduling)
///
/// SPEC 07-OBSERVABILITY §2.2 decision matrix:
/// - memory_pressure > emergency: emergency mode
/// - kv_fragmentation > frag_threshold: defrag mode
/// - otherwise: safe mode
pub struct AccuracyFirstPolicy {
    config: PolicyConfig,
}

impl Default for AccuracyFirstPolicy {
    fn default() -> Self {
        Self { config: PolicyConfig::accuracy_first() }
    }
}

impl AccuracyFirstPolicy {
    pub fn with_config(config: PolicyConfig) -> Self {
        Self { config }
    }
}

impl SchedulingPolicy for AccuracyFirstPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        if state.memory_pressure > self.config.pressure_emergency {
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1),
                admit_new_prefill: false,
                force_swap_out_count: (state.memory_pressure * 3.0).ceil() as usize,
            }
        } else if state.kv_fragmentation > self.config.frag_defrag_threshold {
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1),
                admit_new_prefill: false,
                force_swap_out_count: 1,
            }
        } else {
            SchedulerDecision {
                max_batch_size: self.config.batch_safe,
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
    Accuracy,
}

impl PolicyVariant {
    #[inline(always)]
    pub fn decide(&self, state: &SystemState) -> SchedulerDecision {
        match self {
            Self::Accuracy => AccuracyFirstPolicy::default().decide(state),
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
    fn accuracy_first_emergency_mode() {
        let d = AccuracyFirstPolicy::default().decide(&state_with(0.95, 0.0, 0, 4));
        assert_eq!(d.max_batch_size, 4);
        assert!(!d.admit_new_prefill);
        assert!(d.force_swap_out_count > 0);
    }

    #[test]
    fn accuracy_first_defrag_mode() {
        let d = AccuracyFirstPolicy::default().decide(&state_with(0.5, 0.6, 0, 3));
        assert_eq!(d.max_batch_size, 3);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
    }

    #[test]
    fn accuracy_first_safe_mode() {
        let d = AccuracyFirstPolicy::default().decide(&state_with(0.3, 0.2, 0, 2));
        assert_eq!(d.max_batch_size, 32);
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    #[test]
    fn policy_variant_dispatch() {
        let state = state_with(0.3, 0.2, 5, 2);
        let a = PolicyVariant::Accuracy.decide(&state);
        // Should produce valid decision
        assert!(a.admit_new_prefill);
    }

    #[test]
    fn custom_config_overrides() {
        let mut config = PolicyConfig::accuracy_first();
        config.batch_safe = 128;
        config.pressure_emergency = 0.95;
        let policy = AccuracyFirstPolicy::with_config(config);
        // Safe mode uses custom batch size
        let d = policy.decide(&state_with(0.5, 0.0, 10, 4));
        assert_eq!(d.max_batch_size, 128);
        // Emergency threshold raised — 0.9 no longer triggers emergency
        let d = policy.decide(&state_with(0.9, 0.0, 0, 5));
        assert_eq!(d.max_batch_size, 128);
    }
}
