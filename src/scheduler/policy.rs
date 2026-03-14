use super::jit_types::{KernelStrategy, SchedulerDecision, SystemState};

/// JIT Scheduling Policy Interface.
/// Pure logic component.
pub trait SchedulingPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision;
}

/// 1. Accuracy First Policy (Default/Fallback)
///
/// SPEC §3.1 decision matrix:
/// - memory_pressure > 0.9: emergency mode
/// - kv_fragmentation > 0.5: defrag mode
/// - otherwise: safe mode (max_batch=32)
#[derive(Default)]
pub struct AccuracyFirstPolicy;

impl SchedulingPolicy for AccuracyFirstPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        if state.memory_pressure > 0.9 {
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1),
                admit_new_prefill: false,
                force_swap_out_count: (state.memory_pressure * 3.0).ceil() as usize,
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        } else if state.kv_fragmentation > 0.5 {
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1),
                admit_new_prefill: false,
                force_swap_out_count: 1,
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        } else {
            SchedulerDecision {
                max_batch_size: 32,
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
/// - waiting_queue_len > 50 AND memory_pressure < 0.8: aggressive mode
/// - otherwise: fall back to AccuracyFirst decision
#[derive(Default)]
pub struct ThroughputFirstPolicy;

impl SchedulingPolicy for ThroughputFirstPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        if state.waiting_queue_len > 50 && state.memory_pressure < 0.8 {
            SchedulerDecision {
                max_batch_size: 256,
                admit_new_prefill: true,
                force_swap_out_count: 0,
                kernel_strategy: KernelStrategy::ThroughputFirst,
            }
        } else {
            AccuracyFirstPolicy.decide(state)
        }
    }
}

/// 3. Balanced Policy
///
/// SPEC §3.3:
/// - memory_pressure > 0.85: conservative mode
/// - waiting_queue_len > 20 AND memory_pressure < 0.7: moderate aggressive
/// - otherwise: normal (max_batch=48)
#[derive(Default)]
pub struct BalancedPolicy;

impl SchedulingPolicy for BalancedPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        if state.memory_pressure > 0.85 {
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1),
                admit_new_prefill: false,
                force_swap_out_count: 1,
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        } else if state.waiting_queue_len > 20 && state.memory_pressure < 0.7 {
            SchedulerDecision {
                max_batch_size: 64,
                admit_new_prefill: true,
                force_swap_out_count: 0,
                kernel_strategy: KernelStrategy::ThroughputFirst,
            }
        } else {
            SchedulerDecision {
                max_batch_size: 48,
                admit_new_prefill: true,
                force_swap_out_count: 0,
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        }
    }
}

/// Enum Dispatch for Zero-Cost Abstraction
#[derive(Clone, Copy, Default)]
pub enum PolicyVariant {
    #[default]
    Accuracy,
    Throughput,
    Balanced,
}

impl PolicyVariant {
    #[inline(always)]
    pub fn decide(&self, state: &SystemState) -> SchedulerDecision {
        match self {
            Self::Accuracy => AccuracyFirstPolicy.decide(state),
            Self::Throughput => ThroughputFirstPolicy.decide(state),
            Self::Balanced => BalancedPolicy.decide(state),
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
        let d = AccuracyFirstPolicy.decide(&state_with(0.95, 0.0, 0, 4));
        assert_eq!(d.max_batch_size, 4);
        assert!(!d.admit_new_prefill);
        assert!(d.force_swap_out_count > 0);
        assert_eq!(d.kernel_strategy, KernelStrategy::AccuracyFirst);
    }

    #[test]
    fn accuracy_first_defrag_mode() {
        let d = AccuracyFirstPolicy.decide(&state_with(0.5, 0.6, 0, 3));
        assert_eq!(d.max_batch_size, 3);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
    }

    #[test]
    fn accuracy_first_safe_mode() {
        let d = AccuracyFirstPolicy.decide(&state_with(0.3, 0.2, 0, 2));
        assert_eq!(d.max_batch_size, 32);
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    #[test]
    fn throughput_first_aggressive() {
        let d = ThroughputFirstPolicy.decide(&state_with(0.5, 0.0, 60, 4));
        assert_eq!(d.max_batch_size, 256);
        assert!(d.admit_new_prefill);
        assert_eq!(d.kernel_strategy, KernelStrategy::ThroughputFirst);
    }

    #[test]
    fn throughput_first_fallback() {
        let d = ThroughputFirstPolicy.decide(&state_with(0.5, 0.0, 10, 4));
        // Falls back to AccuracyFirst safe mode
        assert_eq!(d.max_batch_size, 32);
        assert_eq!(d.kernel_strategy, KernelStrategy::AccuracyFirst);
    }

    #[test]
    fn balanced_conservative() {
        let d = BalancedPolicy.decide(&state_with(0.9, 0.0, 0, 5));
        assert_eq!(d.max_batch_size, 5);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 1);
        assert_eq!(d.kernel_strategy, KernelStrategy::AccuracyFirst);
    }

    #[test]
    fn balanced_moderate_aggressive() {
        let d = BalancedPolicy.decide(&state_with(0.5, 0.0, 30, 4));
        assert_eq!(d.max_batch_size, 64);
        assert!(d.admit_new_prefill);
        assert_eq!(d.kernel_strategy, KernelStrategy::ThroughputFirst);
    }

    #[test]
    fn balanced_normal() {
        let d = BalancedPolicy.decide(&state_with(0.5, 0.0, 10, 4));
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
        let d = AccuracyFirstPolicy.decide(&state_with(0.95, 0.0, 0, 1));
        assert_eq!(d.kernel_strategy, KernelStrategy::AccuracyFirst);
        // ThroughputFirst aggressive returns ThroughputFirst
        let d = ThroughputFirstPolicy.decide(&state_with(0.5, 0.0, 60, 4));
        assert_eq!(d.kernel_strategy, KernelStrategy::ThroughputFirst);
        // Balanced moderate aggressive returns ThroughputFirst
        let d = BalancedPolicy.decide(&state_with(0.5, 0.0, 30, 4));
        assert_eq!(d.kernel_strategy, KernelStrategy::ThroughputFirst);
    }
}
