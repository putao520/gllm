use super::jit_types::{KernelStrategy, SchedulerDecision, SystemState};

/// JIT Scheduling Policy Interface.
/// Pure logic component.
pub trait SchedulingPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision;
}

/// 1. Accuracy First Policy (Default/Fallback)
///
/// Trigger: High memory pressure OR instability detected.
#[derive(Default)]
pub struct AccuracyFirstPolicy;

impl SchedulingPolicy for AccuracyFirstPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        // Conservative thresholds
        let high_pressure = state.memory_pressure > 0.9;

        if high_pressure {
            // Emergency Mode: Stop entry, maybe evict
            SchedulerDecision {
                max_batch_size: state.current_running_len.max(1), // Don't grow
                admit_new_prefill: false,
                force_swap_out_count: 1, // Try to free up space
                kernel_strategy: KernelStrategy::AccuracyFirst,
            }
        } else {
            // Safe Mode: Moderate throughput
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
/// Trigger: Low pressure AND high backlog.
#[derive(Default)]
pub struct ThroughputFirstPolicy;

impl SchedulingPolicy for ThroughputFirstPolicy {
    fn decide(&self, _state: &SystemState) -> SchedulerDecision {
        // Aggressive utilization
        SchedulerDecision {
            max_batch_size: 256, // Hardware limit-ish
            admit_new_prefill: true,
            force_swap_out_count: 0,
            kernel_strategy: KernelStrategy::ThroughputFirst,
        }
    }
}

/// Enum Dispatch for Zero-Cost Abstraction
#[derive(Clone, Copy, Default)]
pub enum PolicyVariant {
    #[default]
    Accuracy,
    Throughput,
}

impl PolicyVariant {
    #[inline(always)]
    pub fn decide(&self, state: &SystemState) -> SchedulerDecision {
        match self {
            Self::Accuracy => AccuracyFirstPolicy.decide(state),
            Self::Throughput => ThroughputFirstPolicy.decide(state),
        }
    }
}
