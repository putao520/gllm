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
