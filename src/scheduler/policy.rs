use super::jit_types::{KernelStrategy, SchedulerDecision, SystemState};

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

    pub fn throughput_first() -> Self {
        Self {
            pressure_emergency: 0.9,
            pressure_aggressive_ceiling: 0.8,
            frag_defrag_threshold: 0.5,
            queue_aggressive_trigger: 50,
            batch_safe: 32,
            batch_normal: 32,
            batch_aggressive: 256,
        }
    }

    pub fn balanced() -> Self {
        Self {
            pressure_emergency: 0.85,
            pressure_aggressive_ceiling: 0.7,
            frag_defrag_threshold: 0.5,
            queue_aggressive_trigger: 20,
            batch_safe: 32,
            batch_normal: 48,
            batch_aggressive: 64,
        }
    }
}

/// JIT Scheduling Policy Interface.
/// Pure logic component.
pub trait SchedulingPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision;
}

/// 1. Accuracy First Policy (Default/Fallback)
///
/// SPEC §3.1 decision matrix:
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
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        } else if state.kv_fragmentation > self.config.frag_defrag_threshold {
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1),
                admit_new_prefill: false,
                force_swap_out_count: 1,
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        } else {
            SchedulerDecision {
                max_batch_size: self.config.batch_safe,
                admit_new_prefill: true,
                force_swap_out_count: 0,
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        }
    }
}

/// 2. Throughput First Policy
///
/// SPEC §3.2:
/// - waiting_queue_len > trigger AND memory_pressure < ceiling: aggressive mode
/// - otherwise: fall back to AccuracyFirst decision
pub struct ThroughputFirstPolicy {
    config: PolicyConfig,
    fallback_config: PolicyConfig,
}

impl Default for ThroughputFirstPolicy {
    fn default() -> Self {
        Self {
            config: PolicyConfig::throughput_first(),
            fallback_config: PolicyConfig::accuracy_first(),
        }
    }
}

impl ThroughputFirstPolicy {
    pub fn with_config(config: PolicyConfig) -> Self {
        Self {
            fallback_config: PolicyConfig::accuracy_first(),
            config,
        }
    }
}

impl SchedulingPolicy for ThroughputFirstPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        if state.waiting_queue_len > self.config.queue_aggressive_trigger
            && state.memory_pressure < self.config.pressure_aggressive_ceiling
        {
            SchedulerDecision {
                max_batch_size: self.config.batch_aggressive,
                admit_new_prefill: true,
                force_swap_out_count: 0,
                kernel_strategy: KernelStrategy::ThroughputFirst,
            }
        } else {
            AccuracyFirstPolicy::with_config(self.fallback_config.clone()).decide(state)
        }
    }
}

/// 3. Balanced Policy
///
/// SPEC §3.3:
/// - memory_pressure > emergency: conservative mode
/// - waiting_queue_len > trigger AND memory_pressure < ceiling: moderate aggressive
/// - otherwise: normal mode
pub struct BalancedPolicy {
    config: PolicyConfig,
}

impl Default for BalancedPolicy {
    fn default() -> Self {
        Self { config: PolicyConfig::balanced() }
    }
}

impl BalancedPolicy {
    pub fn with_config(config: PolicyConfig) -> Self {
        Self { config }
    }
}

impl SchedulingPolicy for BalancedPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        if state.memory_pressure > self.config.pressure_emergency {
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1),
                admit_new_prefill: false,
                force_swap_out_count: 1,
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        } else if state.waiting_queue_len > self.config.queue_aggressive_trigger
            && state.memory_pressure < self.config.pressure_aggressive_ceiling
        {
            SchedulerDecision {
                max_batch_size: self.config.batch_aggressive,
                admit_new_prefill: true,
                force_swap_out_count: 0,
                kernel_strategy: KernelStrategy::ThroughputFirst,
            }
        } else {
            SchedulerDecision {
                max_batch_size: self.config.batch_normal,
                admit_new_prefill: true,
                force_swap_out_count: 0,
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        }
    }
}

/// Enum Dispatch for Zero-Cost Abstraction
#[derive(Clone, Default)]
pub enum PolicyVariant {
    #[default]
    Accuracy,
    Throughput,
    Balanced,
    Custom {
        accuracy: PolicyConfig,
        throughput: PolicyConfig,
        balanced: PolicyConfig,
    },
}

impl PolicyVariant {
    #[inline(always)]
    pub fn decide(&self, state: &SystemState) -> SchedulerDecision {
        match self {
            Self::Accuracy => AccuracyFirstPolicy::default().decide(state),
            Self::Throughput => ThroughputFirstPolicy::default().decide(state),
            Self::Balanced => BalancedPolicy::default().decide(state),
            Self::Custom { accuracy, throughput, balanced } => {
                // Use balanced as the primary strategy with custom configs
                let _ = (accuracy, throughput); // reserved for future per-variant dispatch
                BalancedPolicy::with_config(balanced.clone()).decide(state)
            }
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
        assert_eq!(d.kernel_strategy, KernelStrategy::AccuracyFirst);
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
    fn throughput_first_aggressive() {
        let d = ThroughputFirstPolicy::default().decide(&state_with(0.5, 0.0, 60, 4));
        assert_eq!(d.max_batch_size, 256);
        assert!(d.admit_new_prefill);
        assert_eq!(d.kernel_strategy, KernelStrategy::ThroughputFirst);
    }

    #[test]
    fn throughput_first_fallback() {
        let d = ThroughputFirstPolicy::default().decide(&state_with(0.5, 0.0, 10, 4));
        // Falls back to AccuracyFirst safe mode
        assert_eq!(d.max_batch_size, 32);
        assert_eq!(d.kernel_strategy, KernelStrategy::AccuracyFirst);
    }

    #[test]
    fn balanced_conservative() {
        let d = BalancedPolicy::default().decide(&state_with(0.9, 0.0, 0, 5));
        assert_eq!(d.max_batch_size, 5);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
        assert_eq!(d.kernel_strategy, KernelStrategy::AccuracyFirst);
    }

    #[test]
    fn balanced_moderate_aggressive() {
        let d = BalancedPolicy::default().decide(&state_with(0.5, 0.0, 30, 4));
        assert_eq!(d.max_batch_size, 64);
        assert!(d.admit_new_prefill);
        assert_eq!(d.kernel_strategy, KernelStrategy::ThroughputFirst);
    }

    #[test]
    fn balanced_normal() {
        let d = BalancedPolicy::default().decide(&state_with(0.5, 0.0, 10, 4));
        assert_eq!(d.max_batch_size, 48);
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    #[test]
    fn policy_variant_dispatch() {
        let state = state_with(0.3, 0.2, 5, 2);
        let a = PolicyVariant::Accuracy.decide(&state);
        let t = PolicyVariant::Throughput.decide(&state);
        let b = PolicyVariant::Balanced.decide(&state);
        // All should produce valid decisions
        assert_eq!(a.kernel_strategy, KernelStrategy::AccuracyFirst);
        // Throughput falls back to accuracy in low-queue scenario
        assert_eq!(t.kernel_strategy, KernelStrategy::AccuracyFirst);
        assert_eq!(b.max_batch_size, 48);
    }

    #[test]
    fn kernel_strategy_propagation() {
        // AccuracyFirst always returns AccuracyFirst strategy
        let d = AccuracyFirstPolicy::default().decide(&state_with(0.95, 0.0, 0, 1));
        assert_eq!(d.kernel_strategy, KernelStrategy::AccuracyFirst);
        // ThroughputFirst aggressive returns ThroughputFirst
        let d = ThroughputFirstPolicy::default().decide(&state_with(0.5, 0.0, 60, 4));
        assert_eq!(d.kernel_strategy, KernelStrategy::ThroughputFirst);
        // Balanced moderate aggressive returns ThroughputFirst
        let d = BalancedPolicy::default().decide(&state_with(0.5, 0.0, 30, 4));
        assert_eq!(d.kernel_strategy, KernelStrategy::ThroughputFirst);
    }

    #[test]
    fn custom_config_overrides() {
        let mut config = PolicyConfig::balanced();
        config.batch_normal = 128;
        config.pressure_emergency = 0.95;
        let policy = BalancedPolicy::with_config(config);
        // Normal mode uses custom batch size
        let d = policy.decide(&state_with(0.5, 0.0, 10, 4));
        assert_eq!(d.max_batch_size, 128);
        // Emergency threshold raised — 0.9 no longer triggers emergency
        let d = policy.decide(&state_with(0.9, 0.0, 0, 5));
        assert_eq!(d.max_batch_size, 128);
    }
}
