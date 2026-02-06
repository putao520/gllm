/// System state snapshot for JIT decision making.
/// Must be zero-cost to copy (small struct).
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemState {
    // Resource Metrics
    pub memory_pressure: f32,       // [0.0, 1.0]
    pub kv_fragmentation: f32,      // [0.0, 1.0]

    // Load Metrics
    pub waiting_queue_len: usize,
    pub current_running_len: usize,
    pub mean_context_len: usize,

    // Data Observability (Placeholder for Phase 2)
    pub logits_entropy: f32,
}

/// Kernel execution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelStrategy {
    AccuracyFirst,      // FP32 Accumulation, Deterministic
    ThroughputFirst,    // BF16/FP16, Aggressive Fused
}

/// JIT Decision output.
#[derive(Debug, Clone)]
pub struct SchedulerDecision {
    pub max_batch_size: usize,
    pub admit_new_prefill: bool,
    pub force_swap_out_count: usize,
    pub kernel_strategy: KernelStrategy,
}

impl Default for SchedulerDecision {
    fn default() -> Self {
        Self {
            max_batch_size: 1,
            admit_new_prefill: false,
            force_swap_out_count: 0,
            kernel_strategy: KernelStrategy::AccuracyFirst,
        }
    }
}
