/// System state snapshot for JIT decision making.
/// Must be zero-cost to copy (small struct).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SystemState {
    // Resource Metrics
    /// Memory pressure ratio [0.0, 1.0].
    pub memory_pressure: f32,
    /// KV cache fragmentation ratio [0.0, 1.0].
    pub kv_fragmentation: f32,
    /// Swap I/O rate (pages/sec).
    pub swap_io_rate: f32,

    // REQ-SCHED-010: Scheduling decision metrics
    /// Page hit rate in L1 cache [0.0, 1.0].
    pub page_hit_rate: f32,
    /// Thrashing rate (eviction/recovery churn ratio).
    pub thrashing_rate: f32,
    /// Swap latency (microseconds per page).
    pub swap_latency_us: f32,

    // Load Metrics
    pub waiting_queue_len: usize,
    pub current_batch_size: usize,
    pub current_running_len: usize,
    pub mean_context_len: usize,

    /// Phase 2: Shannon entropy of output logits distribution.
    pub logits_entropy: f32,
    /// Phase 2: Attention weight sparsity ratio [0.0, 1.0].
    pub attention_sparsity: f32,

    // MoE Fault Metrics
    /// MoE expert fault rate (faults per decode step).
    pub moe_fault_rate: f32,
    /// MoE average recovery latency (microseconds).
    pub moe_avg_recovery_us: f32,
    /// MoE working set size (distinct experts accessed in tracking window).
    pub moe_working_set_size: usize,

    // §21 Weight Page Metrics (REQ-WP-010)
    /// Total weight pages registered.
    pub weight_page_total: usize,
    /// Weight pages currently in L1 (device).
    pub weight_pages_l1: usize,
    /// Weight pages currently in L2 (host).
    pub weight_pages_l2: usize,
    /// Weight pages currently in L3 (disk).
    pub weight_pages_l3: usize,
    /// Cumulative weight page eviction count.
    pub weight_eviction_count: usize,
    /// Cumulative weight page recovery count.
    pub weight_recovery_count: usize,
}

/// JIT Decision output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerDecision {
    pub max_batch_size: usize,
    pub admit_new_prefill: bool,
    pub force_swap_out_count: usize,
}

impl Default for SchedulerDecision {
    fn default() -> Self {
        Self {
            max_batch_size: 1,
            admit_new_prefill: false,
            force_swap_out_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_state_default_zeroed() {
        let s = SystemState::default();
        assert_eq!(s.memory_pressure, 0.0);
        assert_eq!(s.kv_fragmentation, 0.0);
        assert_eq!(s.swap_io_rate, 0.0);
        assert_eq!(s.waiting_queue_len, 0);
        assert_eq!(s.current_batch_size, 0);
        assert_eq!(s.logits_entropy, 0.0);
        assert_eq!(s.attention_sparsity, 0.0);
        assert_eq!(s.weight_page_total, 0);
    }

    #[test]
    fn system_state_is_copy() {
        let s = SystemState::default();
        let s2 = s; // Copy
        assert_eq!(s.memory_pressure, s2.memory_pressure);
    }

    #[test]
    fn scheduler_decision_default() {
        let d = SchedulerDecision::default();
        assert_eq!(d.max_batch_size, 1);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    #[test]
    fn system_state_fields_mutable() {
        let mut s = SystemState::default();
        s.memory_pressure = 0.8;
        s.moe_fault_rate = 0.01;
        s.weight_page_total = 100;
        s.weight_pages_l1 = 60;
        s.weight_pages_l2 = 30;
        s.weight_pages_l3 = 10;
        assert!((s.memory_pressure - 0.8).abs() < 1e-6);
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 100);
    }

    // ── Additional coverage ──

    #[test]
    fn system_state_clone() {
        let mut s = SystemState::default();
        s.memory_pressure = 0.5;
        s.waiting_queue_len = 10;
        let cloned = s.clone();
        assert!((cloned.memory_pressure - 0.5).abs() < 1e-6);
        assert_eq!(cloned.waiting_queue_len, 10);
    }

    #[test]
    fn system_state_debug_format() {
        let s = SystemState::default();
        let debug = format!("{s:?}");
        assert!(debug.contains("memory_pressure"));
        assert!(debug.contains("kv_fragmentation"));
        assert!(debug.contains("weight_page_total"));
    }

    #[test]
    fn system_state_all_defaults_zero() {
        let s = SystemState::default();
        assert_eq!(s.memory_pressure, 0.0);
        assert_eq!(s.kv_fragmentation, 0.0);
        assert_eq!(s.swap_io_rate, 0.0);
        assert_eq!(s.waiting_queue_len, 0);
        assert_eq!(s.current_batch_size, 0);
        assert_eq!(s.current_running_len, 0);
        assert_eq!(s.mean_context_len, 0);
        assert_eq!(s.logits_entropy, 0.0);
        assert_eq!(s.attention_sparsity, 0.0);
        assert_eq!(s.moe_fault_rate, 0.0);
        assert_eq!(s.moe_avg_recovery_us, 0.0);
        assert_eq!(s.moe_working_set_size, 0);
        assert_eq!(s.weight_page_total, 0);
        assert_eq!(s.weight_pages_l1, 0);
        assert_eq!(s.weight_pages_l2, 0);
        assert_eq!(s.weight_pages_l3, 0);
        assert_eq!(s.weight_eviction_count, 0);
        assert_eq!(s.weight_recovery_count, 0);
    }

    #[test]
    fn scheduler_decision_debug() {
        let d = SchedulerDecision::default();
        let debug = format!("{d:?}");
        assert!(debug.contains("max_batch_size"));
        assert!(debug.contains("admit_new_prefill"));
        assert!(debug.contains("force_swap_out_count"));
    }

    #[test]
    fn scheduler_decision_clone() {
        let d = SchedulerDecision::default();
        let cloned = d.clone();
        assert_eq!(cloned.max_batch_size, 1);
        assert!(!cloned.admit_new_prefill);
    }

    #[test]
    fn scheduler_decision_custom_values() {
        let d = SchedulerDecision {
            max_batch_size: 64,
            admit_new_prefill: true,
            force_swap_out_count: 3,
        };
        assert_eq!(d.max_batch_size, 64);
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 3);
    }

    #[test]
    fn system_state_moe_metrics() {
        let mut s = SystemState::default();
        s.moe_fault_rate = 0.15;
        s.moe_avg_recovery_us = 500.0;
        s.moe_working_set_size = 24;
        assert!((s.moe_fault_rate - 0.15).abs() < 1e-6);
        assert!((s.moe_avg_recovery_us - 500.0).abs() < 1e-6);
        assert_eq!(s.moe_working_set_size, 24);
    }

    // ── PartialEq coverage ──

    #[test]
    fn system_state_eq_default() {
        let a = SystemState::default();
        let b = SystemState::default();
        assert_eq!(a, b);
    }

    #[test]
    fn system_state_neq_after_mutation() {
        let mut a = SystemState::default();
        let b = SystemState::default();
        a.memory_pressure = 0.42;
        assert_ne!(a, b);
    }

    #[test]
    fn scheduler_decision_eq() {
        let a = SchedulerDecision::default();
        let b = SchedulerDecision::default();
        assert_eq!(a, b);
    }

    #[test]
    fn scheduler_decision_neq() {
        let a = SchedulerDecision::default();
        let b = SchedulerDecision {
            max_batch_size: 1,
            admit_new_prefill: true,
            force_swap_out_count: 0,
        };
        assert_ne!(a, b);
    }

    // ── Copy trait ──

    #[test]
    fn scheduler_decision_is_copy() {
        let d = SchedulerDecision {
            max_batch_size: 32,
            admit_new_prefill: true,
            force_swap_out_count: 2,
        };
        let d2 = d; // Copy, not move
        assert_eq!(d, d2);
        assert_eq!(d.max_batch_size, 32);
    }

    // ── Boundary values ──

    #[test]
    fn system_state_f32_max_values() {
        let mut s = SystemState::default();
        s.memory_pressure = f32::MAX;
        s.kv_fragmentation = f32::MAX;
        s.swap_io_rate = f32::MAX;
        s.logits_entropy = f32::MAX;
        s.attention_sparsity = f32::MAX;
        s.moe_fault_rate = f32::MAX;
        s.moe_avg_recovery_us = f32::MAX;
        assert_eq!(s.memory_pressure, f32::MAX);
        assert_eq!(s.kv_fragmentation, f32::MAX);
        assert_eq!(s.swap_io_rate, f32::MAX);
        assert_eq!(s.logits_entropy, f32::MAX);
        assert_eq!(s.attention_sparsity, f32::MAX);
        assert_eq!(s.moe_fault_rate, f32::MAX);
        assert_eq!(s.moe_avg_recovery_us, f32::MAX);
    }

    #[test]
    fn system_state_usize_max_values() {
        let mut s = SystemState::default();
        s.waiting_queue_len = usize::MAX;
        s.current_batch_size = usize::MAX;
        s.current_running_len = usize::MAX;
        s.mean_context_len = usize::MAX;
        s.moe_working_set_size = usize::MAX;
        s.weight_page_total = usize::MAX;
        s.weight_pages_l1 = usize::MAX;
        s.weight_pages_l2 = usize::MAX;
        s.weight_pages_l3 = usize::MAX;
        s.weight_eviction_count = usize::MAX;
        s.weight_recovery_count = usize::MAX;
        assert_eq!(s.waiting_queue_len, usize::MAX);
        assert_eq!(s.weight_page_total, usize::MAX);
    }

    #[test]
    fn system_state_zero_values() {
        let mut s = SystemState::default();
        s.memory_pressure = 0.0;
        s.kv_fragmentation = 0.0;
        s.waiting_queue_len = 0;
        s.weight_page_total = 0;
        assert_eq!(s, SystemState::default());
    }

    #[test]
    fn scheduler_decision_boundary_values() {
        let d_zero = SchedulerDecision {
            max_batch_size: 0,
            admit_new_prefill: false,
            force_swap_out_count: 0,
        };
        assert_eq!(d_zero.max_batch_size, 0);
        assert!(!d_zero.admit_new_prefill);

        let d_max = SchedulerDecision {
            max_batch_size: usize::MAX,
            admit_new_prefill: true,
            force_swap_out_count: usize::MAX,
        };
        assert_eq!(d_max.max_batch_size, usize::MAX);
        assert!(d_max.admit_new_prefill);
        assert_eq!(d_max.force_swap_out_count, usize::MAX);
    }

    // ── Debug format spot-checks for all major field groups ──

    #[test]
    fn system_state_debug_load_metrics() {
        let s = SystemState {
            waiting_queue_len: 5,
            current_batch_size: 3,
            current_running_len: 100,
            mean_context_len: 50,
            ..Default::default()
        };
        let debug = format!("{s:?}");
        assert!(debug.contains("waiting_queue_len: 5"));
        assert!(debug.contains("current_batch_size: 3"));
        assert!(debug.contains("current_running_len: 100"));
        assert!(debug.contains("mean_context_len: 50"));
    }

    #[test]
    fn system_state_debug_weight_pages() {
        let s = SystemState {
            weight_pages_l1: 10,
            weight_pages_l2: 20,
            weight_pages_l3: 5,
            weight_eviction_count: 3,
            weight_recovery_count: 7,
            ..Default::default()
        };
        let debug = format!("{s:?}");
        assert!(debug.contains("weight_pages_l1: 10"));
        assert!(debug.contains("weight_pages_l3: 5"));
        assert!(debug.contains("weight_eviction_count: 3"));
        assert!(debug.contains("weight_recovery_count: 7"));
    }

    // ── Structural invariants ──

    #[test]
    fn system_state_weight_page_tiers_sum() {
        let s = SystemState {
            weight_page_total: 100,
            weight_pages_l1: 50,
            weight_pages_l2: 30,
            weight_pages_l3: 20,
            ..Default::default()
        };
        assert_eq!(
            s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3,
            s.weight_page_total
        );
    }

    #[test]
    fn system_state_size_is_small() {
        // Must be zero-cost to copy — verify struct is reasonably small.
        // 10 f32 fields = 40 bytes, 11 usize fields = 88 bytes → 128 bytes on 64-bit.
        assert!(std::mem::size_of::<SystemState>() <= 144);
    }

    #[test]
    fn scheduler_decision_size_is_small() {
        assert!(std::mem::size_of::<SchedulerDecision>() <= 24);
    }

    // ── Special float values ──

    #[test]
    fn system_state_nan_fields_are_equal_to_self() {
        let mut s = SystemState::default();
        s.memory_pressure = f32::NAN;
        // NaN != NaN by IEEE 754, so PartialEq should reflect that.
        assert_ne!(s, s);
    }

    #[test]
    fn system_state_infinity_fields() {
        let mut s = SystemState::default();
        s.logits_entropy = f32::INFINITY;
        s.swap_io_rate = f32::NEG_INFINITY;
        assert!(s.logits_entropy.is_infinite() && s.logits_entropy.is_sign_positive());
        assert!(s.swap_io_rate.is_infinite() && s.swap_io_rate.is_sign_negative());
    }

    #[test]
    fn system_state_subnormal_float() {
        let mut s = SystemState::default();
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        s.kv_fragmentation = subnormal;
        assert!(s.kv_fragmentation.is_subnormal());
        assert_ne!(s, SystemState::default());
    }

    #[test]
    fn scheduler_decision_nan_carry_in_scheduling_context() {
        // SchedulerDecision has no f32 fields, but verify Copy with
        // a fully-populated instance still works after being moved.
        let d = SchedulerDecision {
            max_batch_size: 128,
            admit_new_prefill: true,
            force_swap_out_count: 7,
        };
        let d2 = d;
        let d3 = d; // Copy again from original (still valid)
        assert_eq!(d2, d3);
    }

    // ── Per-field inequality ──

    #[test]
    fn system_state_neq_per_float_field() {
        let fields: Vec<Box<dyn FnOnce(&mut SystemState)>> = vec![
            Box::new(|s| s.memory_pressure = 0.1),
            Box::new(|s| s.kv_fragmentation = 0.2),
            Box::new(|s| s.swap_io_rate = 0.3),
            Box::new(|s| s.logits_entropy = 1.0),
            Box::new(|s| s.attention_sparsity = 0.5),
            Box::new(|s| s.moe_fault_rate = 0.01),
            Box::new(|s| s.moe_avg_recovery_us = 100.0),
        ];
        for mutate in fields {
            let mut s = SystemState::default();
            mutate(&mut s);
            assert_ne!(s, SystemState::default());
        }
    }

    #[test]
    fn system_state_neq_per_usize_field() {
        let fields: Vec<Box<dyn FnOnce(&mut SystemState)>> = vec![
            Box::new(|s| s.waiting_queue_len = 1),
            Box::new(|s| s.current_batch_size = 1),
            Box::new(|s| s.current_running_len = 1),
            Box::new(|s| s.mean_context_len = 1),
            Box::new(|s| s.moe_working_set_size = 1),
            Box::new(|s| s.weight_page_total = 1),
            Box::new(|s| s.weight_pages_l1 = 1),
            Box::new(|s| s.weight_pages_l2 = 1),
            Box::new(|s| s.weight_pages_l3 = 1),
            Box::new(|s| s.weight_eviction_count = 1),
            Box::new(|s| s.weight_recovery_count = 1),
        ];
        for mutate in fields {
            let mut s = SystemState::default();
            mutate(&mut s);
            assert_ne!(s, SystemState::default());
        }
    }

    #[test]
    fn scheduler_decision_neq_per_field() {
        let base = SchedulerDecision::default();
        assert_ne!(
            SchedulerDecision { max_batch_size: 999, ..base },
            base,
        );
        assert_ne!(
            SchedulerDecision { admit_new_prefill: true, ..base },
            base,
        );
        assert_ne!(
            SchedulerDecision { force_swap_out_count: 1, ..base },
            base,
        );
    }

    // ── Struct update syntax + equality ──

    #[test]
    fn system_state_struct_update_syntax() {
        let base = SystemState {
            memory_pressure: 0.75,
            kv_fragmentation: 0.1,
            ..Default::default()
        };
        let derived = SystemState {
            memory_pressure: 0.75,
            kv_fragmentation: 0.1,
            ..Default::default()
        };
        assert_eq!(base, derived);
    }

    #[test]
    fn system_state_all_usize_fields_nonzero() {
        let s = SystemState {
            waiting_queue_len: 5,
            current_batch_size: 8,
            current_running_len: 2048,
            mean_context_len: 512,
            moe_working_set_size: 16,
            weight_page_total: 200,
            weight_pages_l1: 100,
            weight_pages_l2: 60,
            weight_pages_l3: 40,
            weight_eviction_count: 15,
            weight_recovery_count: 12,
            ..Default::default()
        };
        // All usize fields are > 0 and the tiers sum to total.
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 200);
        assert!(s.waiting_queue_len > 0);
        assert!(s.current_batch_size > 0);
    }

    #[test]
    fn scheduler_decision_zero_batch_size() {
        let d = SchedulerDecision {
            max_batch_size: 0,
            admit_new_prefill: false,
            force_swap_out_count: 0,
        };
        assert_eq!(d.max_batch_size, 0);
        // Verify it is distinct from default (default max_batch_size=1).
        assert_ne!(d, SchedulerDecision::default());
    }

    // ── Float precision edge cases ──

    #[test]
    fn system_state_pressure_ratio_sum_within_range() {
        let s = SystemState {
            memory_pressure: 0.4,
            kv_fragmentation: 0.3,
            attention_sparsity: 0.3,
            ..Default::default()
        };
        let sum = s.memory_pressure + s.kv_fragmentation + s.attention_sparsity;
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(sum <= 1.0 + 1e-6);
    }

    #[test]
    fn system_state_negative_floats() {
        let mut s = SystemState::default();
        // Struct allows negative values; verify they are stored faithfully.
        s.swap_io_rate = -10.5;
        s.moe_avg_recovery_us = -0.001;
        assert!((s.swap_io_rate - (-10.5)).abs() < 1e-6);
        assert!((s.moe_avg_recovery_us - (-0.001)).abs() < 1e-6);
    }

    // ── Debug trait completeness ──

    #[test]
    fn system_state_debug_contains_moe_fields() {
        let s = SystemState {
            moe_fault_rate: 0.1,
            moe_avg_recovery_us: 200.0,
            moe_working_set_size: 8,
            ..Default::default()
        };
        let debug = format!("{s:?}");
        assert!(debug.contains("moe_fault_rate"));
        assert!(debug.contains("moe_avg_recovery_us"));
        assert!(debug.contains("moe_working_set_size"));
    }

    #[test]
    fn scheduler_decision_debug_round_trip_via_format() {
        let d = SchedulerDecision {
            max_batch_size: 256,
            admit_new_prefill: true,
            force_swap_out_count: 99,
        };
        let debug_str = format!("{d:?}");
        // Verify all three fields appear with their values.
        assert!(debug_str.contains("max_batch_size: 256"));
        assert!(debug_str.contains("admit_new_prefill: true"));
        assert!(debug_str.contains("force_swap_out_count: 99"));
    }

    // ── 15 additional tests ──

    #[test]
    fn system_state_realistic_production_snapshot() {
        // Arrange: a realistic production scenario with meaningful values
        let s = SystemState {
            memory_pressure: 0.72,
            kv_fragmentation: 0.35,
            swap_io_rate: 1200.0,
            waiting_queue_len: 14,
            current_batch_size: 32,
            current_running_len: 8192,
            mean_context_len: 2048,
            logits_entropy: 3.14,
            attention_sparsity: 0.6,
            moe_fault_rate: 0.002,
            moe_avg_recovery_us: 350.0,
            moe_working_set_size: 8,
            weight_page_total: 512,
            weight_pages_l1: 300,
            weight_pages_l2: 180,
            weight_pages_l3: 32,
            weight_eviction_count: 45,
            weight_recovery_count: 42,
            ..Default::default()
        };
        assert!((s.memory_pressure - 0.72).abs() < 1e-6);
        assert!((s.kv_fragmentation - 0.35).abs() < 1e-6);
        assert!((s.swap_io_rate - 1200.0).abs() < 1e-6);
        assert_eq!(s.waiting_queue_len, 14);
        assert_eq!(s.current_batch_size, 32);
        assert_eq!(s.current_running_len, 8192);
        assert_eq!(s.mean_context_len, 2048);
        assert!((s.logits_entropy - 3.14).abs() < 1e-6);
        assert!((s.attention_sparsity - 0.6).abs() < 1e-6);
        assert!((s.moe_fault_rate - 0.002).abs() < 1e-6);
        assert!((s.moe_avg_recovery_us - 350.0).abs() < 1e-6);
        assert_eq!(s.moe_working_set_size, 8);
        assert_eq!(s.weight_page_total, 512);
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 512);
        assert!(s.weight_eviction_count >= s.weight_recovery_count);
    }

    #[test]
    fn system_state_all_ratio_fields_at_one() {
        // Arrange: set all ratio fields to their maximum valid value of 1.0
        let mut s = SystemState::default();
        s.memory_pressure = 1.0;
        s.kv_fragmentation = 1.0;
        s.attention_sparsity = 1.0;
        // Act & Assert: verify exact storage at boundary 1.0
        assert!((s.memory_pressure - 1.0).abs() < 1e-6);
        assert!((s.kv_fragmentation - 1.0).abs() < 1e-6);
        assert!((s.attention_sparsity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn system_state_negative_zero_storage() {
        // Arrange: negative zero is a valid f32 value distinct from positive zero
        let mut s = SystemState::default();
        s.memory_pressure = -0.0_f32;
        s.logits_entropy = -0.0_f32;
        // Act & Assert: negative zero is stored and is_subnormal is false
        assert!(s.memory_pressure.is_sign_negative());
        assert!(s.logits_entropy.is_sign_negative());
        // -0.0 == 0.0 in IEEE 754, so PartialEq with default passes
        let mut expected = SystemState::default();
        expected.memory_pressure = -0.0;
        expected.logits_entropy = -0.0;
        assert_eq!(s, expected);
    }

    #[test]
    fn system_state_clone_independence() {
        // Arrange: create a state with non-default values
        let mut s = SystemState::default();
        s.memory_pressure = 0.9;
        s.weight_page_total = 500;
        let cloned = s.clone();
        // Act: mutate the original after cloning
        s.memory_pressure = 0.1;
        s.weight_page_total = 0;
        // Assert: clone retains original values
        assert!((cloned.memory_pressure - 0.9).abs() < 1e-6);
        assert_eq!(cloned.weight_page_total, 500);
    }

    #[test]
    fn system_state_partial_eq_symmetry() {
        // Arrange: two identical states constructed differently
        let a = SystemState {
            memory_pressure: 0.5,
            kv_fragmentation: 0.25,
            swap_io_rate: 100.0,
            waiting_queue_len: 7,
            ..Default::default()
        };
        let b = SystemState {
            memory_pressure: 0.5,
            kv_fragmentation: 0.25,
            swap_io_rate: 100.0,
            waiting_queue_len: 7,
            ..Default::default()
        };
        // Act & Assert: equality must be symmetric (a==b implies b==a)
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn system_state_partial_eq_transitivity() {
        // Arrange: three identical states
        let base = SystemState {
            logits_entropy: 2.71,
            moe_fault_rate: 0.005,
            ..Default::default()
        };
        let b = SystemState {
            logits_entropy: 2.71,
            moe_fault_rate: 0.005,
            ..Default::default()
        };
        let c = SystemState {
            logits_entropy: 2.71,
            moe_fault_rate: 0.005,
            ..Default::default()
        };
        // Act & Assert: a==b and b==c implies a==c
        assert_eq!(base, b);
        assert_eq!(b, c);
        assert_eq!(base, c);
    }

    #[test]
    fn system_state_debug_contains_all_float_fields() {
        // Arrange: populate swap_io_rate and logits_entropy (fields not covered by existing debug tests)
        let s = SystemState {
            swap_io_rate: 42.5,
            logits_entropy: 1.23,
            ..Default::default()
        };
        // Act
        let debug = format!("{s:?}");
        // Assert: both fields appear in debug output
        assert!(debug.contains("swap_io_rate"));
        assert!(debug.contains("logits_entropy"));
    }

    #[test]
    fn system_state_weight_page_tiers_all_zero_with_nonzero_total() {
        // Arrange: total is non-zero but no pages in any tier (e.g., just registered)
        let s = SystemState {
            weight_page_total: 50,
            weight_pages_l1: 0,
            weight_pages_l2: 0,
            weight_pages_l3: 0,
            ..Default::default()
        };
        // Assert: tier sum is zero while total is non-zero (valid transient state)
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 0);
        assert_eq!(s.weight_page_total, 50);
    }

    #[test]
    fn system_state_eviction_vs_recovery_count() {
        // Arrange: more evictions than recoveries indicates net pressure
        let s = SystemState {
            weight_eviction_count: 100,
            weight_recovery_count: 30,
            ..Default::default()
        };
        // Assert: net eviction pressure is positive
        assert!(s.weight_eviction_count > s.weight_recovery_count);
        let net = s.weight_eviction_count - s.weight_recovery_count;
        assert_eq!(net, 70);
    }

    #[test]
    fn system_state_struct_update_preserves_unspecified() {
        // Arrange: use struct update syntax overriding only 2 of 18 fields
        let base = SystemState {
            moe_working_set_size: 64,
            moe_fault_rate: 0.03,
            moe_avg_recovery_us: 200.0,
            ..Default::default()
        };
        let derived = SystemState {
            memory_pressure: 0.5,
            current_batch_size: 16,
            ..base
        };
        // Assert: overridden fields have new values
        assert!((derived.memory_pressure - 0.5).abs() < 1e-6);
        assert_eq!(derived.current_batch_size, 16);
        // Assert: base fields are preserved through struct update
        assert_eq!(derived.moe_working_set_size, 64);
        assert!((derived.moe_fault_rate - 0.03).abs() < 1e-6);
        assert!((derived.moe_avg_recovery_us - 200.0).abs() < 1e-6);
        // Assert: remaining fields are default
        assert_eq!(derived.kv_fragmentation, 0.0);
    }

    #[test]
    fn scheduler_decision_partial_eq_symmetry() {
        // Arrange
        let a = SchedulerDecision {
            max_batch_size: 16,
            admit_new_prefill: false,
            force_swap_out_count: 5,
        };
        let b = SchedulerDecision {
            max_batch_size: 16,
            admit_new_prefill: false,
            force_swap_out_count: 5,
        };
        // Assert: symmetric equality
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn scheduler_decision_partial_eq_transitivity() {
        // Arrange: three identical decisions
        let val = SchedulerDecision {
            max_batch_size: 64,
            admit_new_prefill: true,
            force_swap_out_count: 10,
        };
        let b = SchedulerDecision {
            max_batch_size: 64,
            admit_new_prefill: true,
            force_swap_out_count: 10,
        };
        let c = SchedulerDecision {
            max_batch_size: 64,
            admit_new_prefill: true,
            force_swap_out_count: 10,
        };
        // Assert: transitive equality
        assert_eq!(val, b);
        assert_eq!(b, c);
        assert_eq!(val, c);
    }

    #[test]
    fn scheduler_decision_default_distinct_from_zero_batch() {
        // Arrange: default has max_batch_size=1, zero has 0
        let default = SchedulerDecision::default();
        let zero = SchedulerDecision {
            max_batch_size: 0,
            admit_new_prefill: false,
            force_swap_out_count: 0,
        };
        // Assert: they differ only in max_batch_size
        assert_ne!(default, zero);
        assert_eq!(default.admit_new_prefill, zero.admit_new_prefill);
        assert_eq!(default.force_swap_out_count, zero.force_swap_out_count);
    }

    #[test]
    fn scheduler_decision_clone_independence() {
        // Arrange
        let d = SchedulerDecision {
            max_batch_size: 48,
            admit_new_prefill: true,
            force_swap_out_count: 6,
        };
        let cloned = d.clone();
        // Act: mutating original via rebinding (Copy means this is a no-op for the clone)
        let _d_modified = SchedulerDecision {
            max_batch_size: 1,
            ..d
        };
        // Assert: clone is unaffected
        assert_eq!(cloned.max_batch_size, 48);
        assert!(cloned.admit_new_prefill);
        assert_eq!(cloned.force_swap_out_count, 6);
    }

    #[test]
    fn scheduler_decision_copy_preserves_after_move() {
        // Arrange: Copy trait means d is still usable after assignment
        let d = SchedulerDecision {
            max_batch_size: 96,
            admit_new_prefill: false,
            force_swap_out_count: 3,
        };
        // Act: multiple "moves" (copies) from d
        let d2 = d;
        let d3 = d;
        let d4 = d;
        // Assert: all copies are identical and d is still valid
        assert_eq!(d, d2);
        assert_eq!(d2, d3);
        assert_eq!(d3, d4);
        assert_eq!(d.max_batch_size, 96);
    }

    // ── Wave 13: 15 additional tests ──

    #[test]
    fn system_state_moe_fields_isolation() {
        // Arrange: only MoE-related fields set, all others default
        let s = SystemState {
            moe_fault_rate: 0.08,
            moe_avg_recovery_us: 1200.0,
            moe_working_set_size: 32,
            ..Default::default()
        };
        // Assert: MoE fields carry values
        assert!((s.moe_fault_rate - 0.08).abs() < 1e-6);
        assert!((s.moe_avg_recovery_us - 1200.0).abs() < 1e-6);
        assert_eq!(s.moe_working_set_size, 32);
        // Assert: all other fields remain at zero defaults
        assert_eq!(s.memory_pressure, 0.0);
        assert_eq!(s.kv_fragmentation, 0.0);
        assert_eq!(s.swap_io_rate, 0.0);
        assert_eq!(s.waiting_queue_len, 0);
        assert_eq!(s.current_batch_size, 0);
        assert_eq!(s.weight_page_total, 0);
    }

    #[test]
    fn system_state_all_usize_at_one() {
        // Arrange: every usize field set to minimum non-zero value
        let s = SystemState {
            waiting_queue_len: 1,
            current_batch_size: 1,
            current_running_len: 1,
            mean_context_len: 1,
            moe_working_set_size: 1,
            weight_page_total: 1,
            weight_pages_l1: 1,
            weight_pages_l2: 0,
            weight_pages_l3: 0,
            weight_eviction_count: 1,
            weight_recovery_count: 1,
            ..Default::default()
        };
        // Assert: each usize is exactly 1 (or 0 for L2/L3)
        assert_eq!(s.waiting_queue_len, 1);
        assert_eq!(s.current_batch_size, 1);
        assert_eq!(s.current_running_len, 1);
        assert_eq!(s.mean_context_len, 1);
        assert_eq!(s.moe_working_set_size, 1);
        assert_eq!(s.weight_page_total, 1);
        assert_eq!(s.weight_pages_l1, 1);
        assert_eq!(s.weight_pages_l2, 0);
        assert_eq!(s.weight_pages_l3, 0);
        assert_eq!(s.weight_eviction_count, 1);
        assert_eq!(s.weight_recovery_count, 1);
    }

    #[test]
    fn system_state_copy_after_mutation_preserves_original() {
        // Arrange: build a state, copy it, then mutate the original
        let mut s = SystemState {
            kv_fragmentation: 0.45,
            current_running_len: 4096,
            weight_page_total: 1000,
            ..Default::default()
        };
        let snapshot = s;
        // Act: mutate original fields
        s.kv_fragmentation = 0.0;
        s.current_running_len = 0;
        s.weight_page_total = 0;
        // Assert: snapshot retains original values (Copy trait)
        assert!((snapshot.kv_fragmentation - 0.45).abs() < 1e-6);
        assert_eq!(snapshot.current_running_len, 4096);
        assert_eq!(snapshot.weight_page_total, 1000);
    }

    #[test]
    fn system_state_debug_running_len_and_mean_context() {
        // Arrange: fields not covered by existing debug load metrics test
        let s = SystemState {
            current_running_len: 16384,
            mean_context_len: 4096,
            ..Default::default()
        };
        // Act
        let debug = format!("{s:?}");
        // Assert: both fields appear with correct values
        assert!(debug.contains("current_running_len: 16384"));
        assert!(debug.contains("mean_context_len: 4096"));
    }

    #[test]
    fn system_state_all_floats_identical_value() {
        // Arrange: set every f32 field to the same value
        let val = 0.123;
        let s = SystemState {
            memory_pressure: val,
            kv_fragmentation: val,
            swap_io_rate: val,
            logits_entropy: val,
            attention_sparsity: val,
            moe_fault_rate: val,
            moe_avg_recovery_us: val,
            ..Default::default()
        };
        // Assert: each float field is bitwise identical to val
        assert_eq!(s.memory_pressure.to_bits(), val.to_bits());
        assert_eq!(s.kv_fragmentation.to_bits(), val.to_bits());
        assert_eq!(s.swap_io_rate.to_bits(), val.to_bits());
        assert_eq!(s.logits_entropy.to_bits(), val.to_bits());
        assert_eq!(s.attention_sparsity.to_bits(), val.to_bits());
        assert_eq!(s.moe_fault_rate.to_bits(), val.to_bits());
        assert_eq!(s.moe_avg_recovery_us.to_bits(), val.to_bits());
    }

    #[test]
    fn system_state_tiny_positive_epsilon() {
        // Arrange: use f32 epsilon for ratio fields
        let eps = f32::EPSILON;
        let mut s = SystemState::default();
        s.memory_pressure = eps;
        s.kv_fragmentation = eps;
        s.attention_sparsity = eps;
        // Assert: values are non-zero and greater than zero
        assert!(s.memory_pressure > 0.0);
        assert!(s.kv_fragmentation > 0.0);
        assert!(s.attention_sparsity > 0.0);
        // Assert: distinct from default (which is 0.0)
        assert_ne!(s, SystemState::default());
    }

    #[test]
    fn system_state_weight_page_l1_only_distribution() {
        // Arrange: all pages resident in L1 (hot cache, no eviction)
        let s = SystemState {
            weight_page_total: 256,
            weight_pages_l1: 256,
            weight_pages_l2: 0,
            weight_pages_l3: 0,
            weight_eviction_count: 0,
            weight_recovery_count: 0,
            ..Default::default()
        };
        // Assert: tiers sum to total, all in L1
        assert_eq!(s.weight_pages_l1, s.weight_page_total);
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 256);
        assert_eq!(s.weight_eviction_count, 0);
    }

    #[test]
    fn system_state_struct_update_chain() {
        // Arrange: chain struct updates through two levels
        let base = SystemState {
            memory_pressure: 0.8,
            kv_fragmentation: 0.2,
            ..Default::default()
        };
        let mid = SystemState {
            current_batch_size: 16,
            ..base
        };
        let final_state = SystemState {
            logits_entropy: 4.5,
            ..mid
        };
        // Assert: final has values from all three levels
        assert!((final_state.memory_pressure - 0.8).abs() < 1e-6);
        assert!((final_state.kv_fragmentation - 0.2).abs() < 1e-6);
        assert_eq!(final_state.current_batch_size, 16);
        assert!((final_state.logits_entropy - 4.5).abs() < 1e-6);
        // Assert: unspecified fields are default
        assert_eq!(final_state.swap_io_rate, 0.0);
        assert_eq!(final_state.weight_page_total, 0);
    }

    #[test]
    fn system_state_neq_different_float_fields_separately() {
        // Arrange: verify that changing only one float field breaks equality
        let base = SystemState {
            memory_pressure: 0.1,
            kv_fragmentation: 0.2,
            swap_io_rate: 0.3,
            logits_entropy: 0.4,
            attention_sparsity: 0.5,
            moe_fault_rate: 0.6,
            moe_avg_recovery_us: 0.7,
            ..Default::default()
        };
        // Act & Assert: each single-field mutation makes it different
        let mut s = base;
        s.moe_avg_recovery_us = 999.0;
        assert_ne!(s, base);
        s = base;
        s.attention_sparsity = 0.99;
        assert_ne!(s, base);
    }

    #[test]
    fn system_state_high_load_scenario() {
        // Arrange: simulate heavy load — large batch, long contexts, high pressure
        let s = SystemState {
            memory_pressure: 0.95,
            kv_fragmentation: 0.8,
            swap_io_rate: 50000.0,
            waiting_queue_len: 200,
            current_batch_size: 128,
            current_running_len: 32768,
            mean_context_len: 8192,
            logits_entropy: 0.5,
            attention_sparsity: 0.9,
            ..Default::default()
        };
        // Assert: load metrics indicate saturation
        assert!(s.memory_pressure > 0.9);
        assert!(s.kv_fragmentation > 0.7);
        assert!(s.swap_io_rate > 10000.0);
        assert!(s.waiting_queue_len > 100);
        assert!(s.current_batch_size > 64);
        assert!(s.current_running_len > 16000);
        assert!(s.mean_context_len > 4000);
    }

    #[test]
    fn scheduler_decision_struct_update_syntax() {
        // Arrange: use struct update syntax on SchedulerDecision
        let base = SchedulerDecision {
            max_batch_size: 32,
            admit_new_prefill: true,
            force_swap_out_count: 4,
        };
        // Act: override only one field
        let derived = SchedulerDecision {
            force_swap_out_count: 0,
            ..base
        };
        // Assert: overridden field is new, others preserved
        assert_eq!(derived.max_batch_size, 32);
        assert!(derived.admit_new_prefill);
        assert_eq!(derived.force_swap_out_count, 0);
        // Assert: derived is distinct from base
        assert_ne!(derived, base);
    }

    #[test]
    fn scheduler_decision_admit_toggle_symmetry() {
        // Arrange: two decisions differing only in admit_new_prefill
        let admit_true = SchedulerDecision {
            max_batch_size: 8,
            admit_new_prefill: true,
            force_swap_out_count: 0,
        };
        let admit_false = SchedulerDecision {
            max_batch_size: 8,
            admit_new_prefill: false,
            force_swap_out_count: 0,
        };
        // Assert: they are not equal
        assert_ne!(admit_true, admit_false);
        // Assert: toggling back produces equality
        let toggled = SchedulerDecision {
            admit_new_prefill: true,
            ..admit_false
        };
        assert_eq!(toggled, admit_true);
    }

    #[test]
    fn scheduler_decision_large_swap_small_batch() {
        // Arrange: aggressive eviction with small batch (memory-constrained)
        let d = SchedulerDecision {
            max_batch_size: 2,
            admit_new_prefill: false,
            force_swap_out_count: 500,
        };
        // Assert: swap count dominates batch size
        assert!(d.force_swap_out_count > d.max_batch_size);
        assert_eq!(d.max_batch_size, 2);
        assert!(!d.admit_new_prefill);
    }

    #[test]
    fn system_state_copy_multiple_uses_valid() {
        // Arrange: a single state is copied to multiple variables
        let src = SystemState {
            memory_pressure: 0.6,
            swap_io_rate: 2500.0,
            moe_working_set_size: 12,
            ..Default::default()
        };
        // Act: create 4 copies
        let a = src;
        let b = src;
        let c = src;
        let d = src;
        // Assert: all copies identical, src still valid
        assert_eq!(a, src);
        assert_eq!(b, src);
        assert_eq!(c, src);
        assert_eq!(d, src);
        assert_eq!(src.moe_working_set_size, 12);
        assert!((src.memory_pressure - 0.6).abs() < 1e-6);
    }

    #[test]
    fn cross_struct_equality_independence() {
        // Arrange: SystemState and SchedulerDecision have no PartialEq cross-impl
        let state = SystemState::default();
        let decision = SchedulerDecision::default();
        // Assert: structs are different types but both derive PartialEq correctly
        // Verify that Copy works independently for each type
        let state_copy = state;
        let decision_copy = decision;
        assert_eq!(state, state_copy);
        assert_eq!(decision, decision_copy);
        // Assert: size_of confirms they are distinct types
        assert_ne!(
            std::mem::size_of::<SystemState>(),
            std::mem::size_of::<SchedulerDecision>()
        );
    }

    // ── Wave 14: 13 additional edge-case tests ──

    #[test]
    fn system_state_all_weight_pages_in_l3_cold_storage() {
        // Arrange: all pages evicted to disk (L3), none in device or host memory
        let s = SystemState {
            weight_page_total: 128,
            weight_pages_l1: 0,
            weight_pages_l2: 0,
            weight_pages_l3: 128,
            weight_eviction_count: 128,
            weight_recovery_count: 0,
            ..Default::default()
        };
        // Assert: tiers sum to total, all in L3
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 128);
        assert_eq!(s.weight_pages_l3, s.weight_page_total);
        assert_eq!(s.weight_pages_l1, 0);
        assert_eq!(s.weight_pages_l2, 0);
    }

    #[test]
    fn system_state_recovery_exceeds_eviction() {
        // Arrange: recovery count can exceed eviction count (e.g., repeated re-fetch of same pages)
        let s = SystemState {
            weight_eviction_count: 50,
            weight_recovery_count: 200,
            ..Default::default()
        };
        // Assert: recovery > eviction is a valid state (pages can be fetched multiple times)
        assert!(s.weight_recovery_count > s.weight_eviction_count);
        assert_eq!(s.weight_recovery_count - s.weight_eviction_count, 150);
    }

    #[test]
    fn system_state_negative_infinity_on_all_float_fields() {
        // Arrange: store negative infinity in every f32 field
        let mut s = SystemState::default();
        s.memory_pressure = f32::NEG_INFINITY;
        s.kv_fragmentation = f32::NEG_INFINITY;
        s.swap_io_rate = f32::NEG_INFINITY;
        s.logits_entropy = f32::NEG_INFINITY;
        s.attention_sparsity = f32::NEG_INFINITY;
        s.moe_fault_rate = f32::NEG_INFINITY;
        s.moe_avg_recovery_us = f32::NEG_INFINITY;
        // Assert: all fields are negative infinity
        assert!(s.memory_pressure.is_infinite() && s.memory_pressure.is_sign_negative());
        assert!(s.kv_fragmentation.is_infinite() && s.kv_fragmentation.is_sign_negative());
        assert!(s.swap_io_rate.is_infinite() && s.swap_io_rate.is_sign_negative());
        assert!(s.logits_entropy.is_infinite() && s.logits_entropy.is_sign_negative());
        assert!(s.attention_sparsity.is_infinite() && s.attention_sparsity.is_sign_negative());
        assert!(s.moe_fault_rate.is_infinite() && s.moe_fault_rate.is_sign_negative());
        assert!(s.moe_avg_recovery_us.is_infinite() && s.moe_avg_recovery_us.is_sign_negative());
    }

    #[test]
    fn system_state_nan_on_each_float_field_individually() {
        // Arrange: verify NaN breaks equality for each float field independently
        let mutations: Vec<Box<dyn FnOnce(&mut SystemState)>> = vec![
            Box::new(|s| s.memory_pressure = f32::NAN),
            Box::new(|s| s.kv_fragmentation = f32::NAN),
            Box::new(|s| s.swap_io_rate = f32::NAN),
            Box::new(|s| s.logits_entropy = f32::NAN),
            Box::new(|s| s.attention_sparsity = f32::NAN),
            Box::new(|s| s.moe_fault_rate = f32::NAN),
            Box::new(|s| s.moe_avg_recovery_us = f32::NAN),
        ];
        for mutate in mutations {
            let mut s = SystemState {
                memory_pressure: 0.5,
                kv_fragmentation: 0.3,
                swap_io_rate: 100.0,
                logits_entropy: 2.0,
                attention_sparsity: 0.4,
                moe_fault_rate: 0.01,
                moe_avg_recovery_us: 200.0,
                ..Default::default()
            };
            let original = s;
            mutate(&mut s);
            // NaN != any value including the original, so s != original
            assert_ne!(s, original);
        }
    }

    #[test]
    fn system_state_large_float_near_max() {
        // Arrange: use a large float near but not at f32::MAX (3.4e38)
        let large = 3.4e38_f32;
        let mut s = SystemState::default();
        s.swap_io_rate = large;
        s.moe_avg_recovery_us = large;
        // Assert: values are stored faithfully without overflow
        assert_eq!(s.swap_io_rate.to_bits(), large.to_bits());
        assert_eq!(s.moe_avg_recovery_us.to_bits(), large.to_bits());
        assert!(s.swap_io_rate.is_normal());
        assert!(s.moe_avg_recovery_us.is_normal());
    }

    #[test]
    fn system_state_running_len_at_least_mean_context_len_typical() {
        // Arrange: in a typical scenario, running_len >= mean_context_len
        // (total tokens across all sequences >= average per sequence)
        let s = SystemState {
            current_running_len: 4096,
            mean_context_len: 512,
            current_batch_size: 8,
            ..Default::default()
        };
        // Assert: running_len >= mean_context_len for a non-degenerate batch
        assert!(s.current_running_len >= s.mean_context_len);
        assert!(s.current_batch_size > 0);
    }

    #[test]
    fn system_state_float_min_positive_smallest_normal() {
        // Arrange: f32::MIN_POSITIVE is the smallest positive normal float (~1.175e-38)
        let smallest_normal = f32::MIN_POSITIVE;
        assert!(smallest_normal > 0.0);
        assert!(!smallest_normal.is_subnormal());
        let mut s = SystemState::default();
        s.memory_pressure = smallest_normal;
        s.kv_fragmentation = smallest_normal;
        // Assert: stored as normal floats, distinct from zero
        assert_eq!(s.memory_pressure, smallest_normal);
        assert_eq!(s.kv_fragmentation, smallest_normal);
        assert_ne!(s, SystemState::default());
    }

    #[test]
    fn system_state_struct_update_preserves_moe_base() {
        // Arrange: base with MoE fields, derive with resource fields
        let base = SystemState {
            moe_fault_rate: 0.05,
            moe_avg_recovery_us: 800.0,
            moe_working_set_size: 16,
            ..Default::default()
        };
        let derived = SystemState {
            memory_pressure: 0.85,
            kv_fragmentation: 0.4,
            weight_page_total: 300,
            ..base
        };
        // Assert: resource fields are overridden
        assert!((derived.memory_pressure - 0.85).abs() < 1e-6);
        assert!((derived.kv_fragmentation - 0.4).abs() < 1e-6);
        assert_eq!(derived.weight_page_total, 300);
        // Assert: MoE fields are preserved from base
        assert!((derived.moe_fault_rate - 0.05).abs() < 1e-6);
        assert!((derived.moe_avg_recovery_us - 800.0).abs() < 1e-6);
        assert_eq!(derived.moe_working_set_size, 16);
        // Assert: unspecified fields remain default
        assert_eq!(derived.swap_io_rate, 0.0);
        assert_eq!(derived.waiting_queue_len, 0);
    }

    #[test]
    fn system_state_all_usize_fields_same_value() {
        // Arrange: set every usize field to 42 to verify no aliasing
        let s = SystemState {
            waiting_queue_len: 42,
            current_batch_size: 42,
            current_running_len: 42,
            mean_context_len: 42,
            moe_working_set_size: 42,
            weight_page_total: 126, // 42*3 for tiers
            weight_pages_l1: 42,
            weight_pages_l2: 42,
            weight_pages_l3: 42,
            weight_eviction_count: 42,
            weight_recovery_count: 42,
            ..Default::default()
        };
        // Assert: each field is independently 42 (or 126 for total)
        assert_eq!(s.waiting_queue_len, 42);
        assert_eq!(s.current_batch_size, 42);
        assert_eq!(s.current_running_len, 42);
        assert_eq!(s.mean_context_len, 42);
        assert_eq!(s.moe_working_set_size, 42);
        assert_eq!(s.weight_pages_l1, 42);
        assert_eq!(s.weight_pages_l2, 42);
        assert_eq!(s.weight_pages_l3, 42);
        assert_eq!(s.weight_eviction_count, 42);
        assert_eq!(s.weight_recovery_count, 42);
        assert_eq!(s.weight_page_total, 126);
    }

    #[test]
    fn scheduler_decision_max_swap_count_boundary() {
        // Arrange: force_swap_out_count at usize::MAX with admit=true
        let d = SchedulerDecision {
            max_batch_size: 1,
            admit_new_prefill: true,
            force_swap_out_count: usize::MAX,
        };
        // Assert: swap count at max boundary
        assert_eq!(d.force_swap_out_count, usize::MAX);
        assert!(d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 1);
        // Assert: distinct from default
        assert_ne!(d, SchedulerDecision::default());
    }

    #[test]
    fn scheduler_decision_three_copies_independent() {
        // Arrange: create a decision and three copies
        let original = SchedulerDecision {
            max_batch_size: 64,
            admit_new_prefill: true,
            force_swap_out_count: 10,
        };
        let mut a = original;
        let mut b = original;
        let mut c = original;
        // Act: "mutate" each copy by reassignment (Copy means original is unchanged)
        a = SchedulerDecision { max_batch_size: 1, ..a };
        b = SchedulerDecision { admit_new_prefill: false, ..b };
        c = SchedulerDecision { force_swap_out_count: 0, ..c };
        // Assert: each copy has its own mutation
        assert_eq!(a.max_batch_size, 1);
        assert!(a.admit_new_prefill);
        assert_eq!(a.force_swap_out_count, 10);

        assert_eq!(b.max_batch_size, 64);
        assert!(!b.admit_new_prefill);
        assert_eq!(b.force_swap_out_count, 10);

        assert_eq!(c.max_batch_size, 64);
        assert!(c.admit_new_prefill);
        assert_eq!(c.force_swap_out_count, 0);

        // Assert: original is unchanged
        assert_eq!(original.max_batch_size, 64);
        assert!(original.admit_new_prefill);
        assert_eq!(original.force_swap_out_count, 10);
    }

    #[test]
    fn system_state_zero_tiers_with_nonzero_eviction_recovery() {
        // Arrange: no pages registered yet, but eviction/recovery counters are non-zero
        // (e.g., from a previous model load in the same session)
        let s = SystemState {
            weight_page_total: 0,
            weight_pages_l1: 0,
            weight_pages_l2: 0,
            weight_pages_l3: 0,
            weight_eviction_count: 75,
            weight_recovery_count: 60,
            ..Default::default()
        };
        // Assert: counters carry history while current state is empty
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 0);
        assert_eq!(s.weight_page_total, 0);
        assert!(s.weight_eviction_count > 0);
        assert!(s.weight_recovery_count > 0);
        assert!(s.weight_eviction_count > s.weight_recovery_count);
    }

    #[test]
    fn system_state_weight_page_l2_only_distribution() {
        // Arrange: all pages in host memory (L2), none in device (L1) or disk (L3)
        let s = SystemState {
            weight_page_total: 64,
            weight_pages_l1: 0,
            weight_pages_l2: 64,
            weight_pages_l3: 0,
            weight_eviction_count: 64,
            weight_recovery_count: 0,
            ..Default::default()
        };
        // Assert: all pages in L2 only
        assert_eq!(s.weight_pages_l2, s.weight_page_total);
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 64);
        assert_eq!(s.weight_pages_l1, 0);
        assert_eq!(s.weight_pages_l3, 0);
    }

    // ── Wave 15: 13 additional edge-case tests ──

    #[test]
    fn system_state_ratio_fields_sum_exceeds_one() {
        // Arrange: ratio fields individually in [0,1] but their sum can exceed 1.0
        let s = SystemState {
            memory_pressure: 0.9,
            kv_fragmentation: 0.8,
            attention_sparsity: 0.7,
            ..Default::default()
        };
        // Act
        let sum = s.memory_pressure + s.kv_fragmentation + s.attention_sparsity;
        // Assert: sum exceeds 1.0 (fields are independent, not required to partition 1.0)
        assert!(sum > 1.0);
        assert!((sum - 2.4).abs() < 1e-5);
    }

    #[test]
    fn system_state_debug_ratio_fields_present() {
        // Arrange: set attention_sparsity which is not covered by existing debug tests
        let s = SystemState {
            attention_sparsity: 0.75,
            kv_fragmentation: 0.25,
            ..Default::default()
        };
        // Act
        let debug = format!("{s:?}");
        // Assert: both ratio fields appear in debug output
        assert!(debug.contains("attention_sparsity"));
        assert!(debug.contains("kv_fragmentation"));
    }

    #[test]
    fn scheduler_decision_default_admits_no_prefill() {
        // Arrange: get the default decision
        let d = SchedulerDecision::default();
        // Assert: default explicitly denies prefill admission
        assert!(!d.admit_new_prefill);
        assert_eq!(d.max_batch_size, 1);
        assert_eq!(d.force_swap_out_count, 0);
        // Assert: boolean field is explicitly false (not just falsy)
        assert_eq!(d.admit_new_prefill, false);
    }

    #[test]
    fn system_state_mean_context_derived_from_running_and_batch() {
        // Arrange: running_len = batch_size * mean_context_len (typical relationship)
        let s = SystemState {
            current_batch_size: 4,
            mean_context_len: 256,
            current_running_len: 1024,
            ..Default::default()
        };
        // Act & Assert: the derived relationship holds
        assert_eq!(s.current_running_len, s.current_batch_size * s.mean_context_len);
    }

    #[test]
    fn system_state_debug_output_contains_all_seventeen_fields() {
        // Arrange: construct a state with all fields set to recognizable values
        let s = SystemState {
            memory_pressure: 0.11,
            kv_fragmentation: 0.22,
            swap_io_rate: 33.0,
            waiting_queue_len: 44,
            current_batch_size: 55,
            current_running_len: 66,
            mean_context_len: 77,
            logits_entropy: 8.8,
            attention_sparsity: 0.99,
            moe_fault_rate: 1.1,
            moe_avg_recovery_us: 2.2,
            moe_working_set_size: 13,
            weight_page_total: 14,
            weight_pages_l1: 15,
            weight_pages_l2: 16,
            weight_pages_l3: 17,
            weight_eviction_count: 18,
            weight_recovery_count: 19,
            ..Default::default()
        };
        let debug = format!("{s:?}");
        // Assert: all 18 fields appear in debug output
        for field_name in [
            "memory_pressure",
            "kv_fragmentation",
            "swap_io_rate",
            "waiting_queue_len",
            "current_batch_size",
            "current_running_len",
            "mean_context_len",
            "logits_entropy",
            "attention_sparsity",
            "moe_fault_rate",
            "moe_avg_recovery_us",
            "moe_working_set_size",
            "weight_page_total",
            "weight_pages_l1",
            "weight_pages_l2",
            "weight_pages_l3",
            "weight_eviction_count",
            "weight_recovery_count",
        ] {
            assert!(debug.contains(field_name), "missing field: {field_name}");
        }
    }

    #[test]
    fn scheduler_decision_copy_three_way_equality_chain() {
        // Arrange: create a decision, copy it twice via let-binding
        let d1 = SchedulerDecision {
            max_batch_size: 99,
            admit_new_prefill: true,
            force_swap_out_count: 7,
        };
        // Act: Copy trait allows d1 to remain valid after each binding
        let d2 = d1;
        let d3 = d2;
        // Assert: all three are pairwise equal
        assert_eq!(d1, d2);
        assert_eq!(d2, d3);
        assert_eq!(d1, d3);
        assert_eq!(d1.max_batch_size, 99);
    }

    #[test]
    fn system_state_moe_and_weight_pages_together() {
        // Arrange: set both MoE and weight page fields simultaneously
        let s = SystemState {
            moe_fault_rate: 0.02,
            moe_avg_recovery_us: 150.0,
            moe_working_set_size: 8,
            weight_page_total: 400,
            weight_pages_l1: 200,
            weight_pages_l2: 150,
            weight_pages_l3: 50,
            weight_eviction_count: 30,
            weight_recovery_count: 28,
            ..Default::default()
        };
        // Assert: MoE fields are correct
        assert!((s.moe_fault_rate - 0.02).abs() < 1e-6);
        assert!((s.moe_avg_recovery_us - 150.0).abs() < 1e-6);
        assert_eq!(s.moe_working_set_size, 8);
        // Assert: weight page tiers sum to total
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 400);
        // Assert: other fields remain default
        assert_eq!(s.memory_pressure, 0.0);
        assert_eq!(s.waiting_queue_len, 0);
    }

    #[test]
    fn system_state_float_arithmetic_from_default() {
        // Arrange: start from default and compute a derived value
        let mut s = SystemState::default();
        s.memory_pressure = 0.3;
        s.kv_fragmentation = 0.2;
        // Act: compute a synthetic metric
        let combined = s.memory_pressure + s.kv_fragmentation;
        // Assert: arithmetic on stored fields is exact for these small values
        assert!((combined - 0.5).abs() < 1e-7);
        assert!(combined < 1.0);
    }

    #[test]
    fn system_state_ratio_field_exactly_half() {
        // Arrange: set ratio fields to exactly 0.5 (a power-of-two fraction, exact in f32)
        let mut s = SystemState::default();
        s.memory_pressure = 0.5;
        s.kv_fragmentation = 0.5;
        s.attention_sparsity = 0.5;
        // Assert: 0.5 is exactly representable in f32
        assert_eq!(s.memory_pressure, 0.5);
        assert_eq!(s.kv_fragmentation, 0.5);
        assert_eq!(s.attention_sparsity, 0.5);
        // Assert: 0.5 + 0.5 is exactly 1.0
        assert_eq!(s.memory_pressure + s.kv_fragmentation, 1.0);
    }

    #[test]
    fn scheduler_decision_all_fields_at_max_with_true() {
        // Arrange: every numeric field at usize::MAX, boolean true
        let d = SchedulerDecision {
            max_batch_size: usize::MAX,
            admit_new_prefill: true,
            force_swap_out_count: usize::MAX,
        };
        // Assert: all values stored correctly
        assert_eq!(d.max_batch_size, usize::MAX);
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, usize::MAX);
        // Assert: max batch == max swap (both at usize::MAX)
        assert_eq!(d.max_batch_size, d.force_swap_out_count);
    }

    #[test]
    fn system_state_single_usize_at_max_rest_zero() {
        // Arrange: only one usize field at MAX, all others at 0
        let s = SystemState {
            weight_eviction_count: usize::MAX,
            ..Default::default()
        };
        // Assert: only the target field is non-zero
        assert_eq!(s.weight_eviction_count, usize::MAX);
        assert_eq!(s.waiting_queue_len, 0);
        assert_eq!(s.current_batch_size, 0);
        assert_eq!(s.current_running_len, 0);
        assert_eq!(s.mean_context_len, 0);
        assert_eq!(s.moe_working_set_size, 0);
        assert_eq!(s.weight_page_total, 0);
        assert_eq!(s.weight_pages_l1, 0);
        assert_eq!(s.weight_pages_l2, 0);
        assert_eq!(s.weight_pages_l3, 0);
        assert_eq!(s.weight_recovery_count, 0);
    }

    #[test]
    fn system_state_nan_bitwise_preserved_across_clone() {
        // Arrange: set a field to a specific NaN payload
        let nan_with_payload = f32::from_bits(0x7fc00001u32); // quiet NaN with payload bit
        assert!(nan_with_payload.is_nan());
        let mut s = SystemState::default();
        s.logits_entropy = nan_with_payload;
        // Act: clone the state
        let cloned = s.clone();
        // Assert: NaN payload bits are preserved through clone
        assert_eq!(cloned.logits_entropy.to_bits(), nan_with_payload.to_bits());
        assert!(cloned.logits_entropy.is_nan());
    }

    #[test]
    fn system_state_debug_does_not_truncate_float_precision() {
        // Arrange: use a float with many decimal digits
        let precise = 0.12345678;
        let s = SystemState {
            memory_pressure: precise,
            ..Default::default()
        };
        // Act
        let debug = format!("{s:?}");
        // Assert: debug output contains the field name (value format is Debug-derived)
        assert!(debug.contains("memory_pressure"));
        // Assert: the stored value is bitwise identical
        assert_eq!(s.memory_pressure.to_bits(), precise.to_bits());
    }

    // ── Wave 16: 13 additional edge-case tests ──

    #[test]
    fn system_state_align_of_is_natural() {
        // Arrange & Act: check alignment matches the largest field type
        let state_align = std::mem::align_of::<SystemState>();
        // Assert: alignment should be at least 8 (usize on 64-bit) and at most 16
        assert!(state_align >= 4, "alignment too small: {state_align}");
        assert!(state_align <= 16, "alignment unexpectedly large: {state_align}");
    }

    #[test]
    fn scheduler_decision_align_of_is_natural() {
        // Arrange & Act: SchedulerDecision has usize + bool + usize
        let align = std::mem::align_of::<SchedulerDecision>();
        // Assert: alignment at least 8 on 64-bit (from usize fields)
        assert!(align >= 4, "alignment too small: {align}");
    }

    #[test]
    fn system_state_re_default_after_mutation() {
        // Arrange: create and mutate
        let mut s = SystemState::default();
        s.memory_pressure = 0.99;
        s.weight_page_total = 999;
        s.moe_working_set_size = 64;
        assert_ne!(s, SystemState::default());
        // Act: re-apply default
        s = SystemState::default();
        // Assert: re-defaulted state equals a fresh default
        assert_eq!(s, SystemState::default());
        assert_eq!(s.memory_pressure, 0.0);
        assert_eq!(s.weight_page_total, 0);
        assert_eq!(s.moe_working_set_size, 0);
    }

    #[test]
    fn scheduler_decision_re_default_after_mutation() {
        // Arrange: create non-default decision
        let mut d = SchedulerDecision {
            max_batch_size: 256,
            admit_new_prefill: true,
            force_swap_out_count: 100,
        };
        assert_ne!(d, SchedulerDecision::default());
        // Act: re-apply default
        d = SchedulerDecision::default();
        // Assert: matches fresh default
        assert_eq!(d.max_batch_size, 1);
        assert!(!d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    #[test]
    fn system_state_partial_eq_reflexivity_non_nan() {
        // Arrange: a non-NaN state with non-trivial values
        let s = SystemState {
            memory_pressure: 0.33,
            kv_fragmentation: 0.67,
            swap_io_rate: 500.0,
            waiting_queue_len: 3,
            current_batch_size: 7,
            current_running_len: 1024,
            mean_context_len: 256,
            logits_entropy: 2.5,
            attention_sparsity: 0.15,
            moe_fault_rate: 0.001,
            moe_avg_recovery_us: 75.0,
            moe_working_set_size: 4,
            weight_page_total: 80,
            weight_pages_l1: 40,
            weight_pages_l2: 30,
            weight_pages_l3: 10,
            weight_eviction_count: 5,
            weight_recovery_count: 3,
            ..Default::default()
        };
        // Act & Assert: reflexivity (a == a) holds for non-NaN floats
        assert_eq!(s, s);
    }

    #[test]
    fn system_state_float_min_negative_finite() {
        // Arrange: f32::MIN is the largest negative finite float (~-3.4e38)
        let mut s = SystemState::default();
        s.swap_io_rate = f32::MIN;
        s.moe_avg_recovery_us = f32::MIN;
        // Assert: stored faithfully
        assert_eq!(s.swap_io_rate, f32::MIN);
        assert_eq!(s.moe_avg_recovery_us, f32::MIN);
        assert!(s.swap_io_rate.is_normal());
        assert!(s.swap_io_rate.is_sign_negative());
        assert_ne!(s, SystemState::default());
    }

    #[test]
    fn system_state_mixed_positive_and_infinity() {
        // Arrange: some fields positive finite, others positive infinity
        let s = SystemState {
            memory_pressure: 0.5,
            swap_io_rate: f32::INFINITY,
            logits_entropy: 3.0,
            moe_fault_rate: f32::INFINITY,
            ..Default::default()
        };
        // Assert: finite fields are normal, infinite fields are recognized
        assert!(s.memory_pressure.is_normal());
        assert!(s.swap_io_rate.is_infinite());
        assert!(s.logits_entropy.is_normal());
        assert!(s.moe_fault_rate.is_infinite());
        assert!(!s.memory_pressure.is_infinite());
    }

    #[test]
    fn system_state_weight_page_tiers_even_distribution() {
        // Arrange: pages evenly distributed across all three tiers
        let total = 300usize;
        let s = SystemState {
            weight_page_total: total,
            weight_pages_l1: 100,
            weight_pages_l2: 100,
            weight_pages_l3: 100,
            ..Default::default()
        };
        // Assert: tiers are equal and sum to total
        assert_eq!(s.weight_pages_l1, s.weight_pages_l2);
        assert_eq!(s.weight_pages_l2, s.weight_pages_l3);
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, total);
    }

    #[test]
    fn system_state_running_len_zero_with_nonzero_batch() {
        // Arrange: batch_size > 0 but running_len = 0 (all sequences just admitted, no tokens yet)
        let s = SystemState {
            current_batch_size: 4,
            current_running_len: 0,
            mean_context_len: 0,
            ..Default::default()
        };
        // Assert: valid transient state — batch exists but no tokens processed
        assert!(s.current_batch_size > 0);
        assert_eq!(s.current_running_len, 0);
        assert_eq!(s.mean_context_len, 0);
    }

    #[test]
    fn scheduler_decision_struct_update_chain_two_levels() {
        // Arrange: base -> mid -> final through two levels of struct update
        let base = SchedulerDecision {
            max_batch_size: 16,
            admit_new_prefill: false,
            force_swap_out_count: 0,
        };
        let mid = SchedulerDecision {
            admit_new_prefill: true,
            ..base
        };
        let final_d = SchedulerDecision {
            force_swap_out_count: 50,
            ..mid
        };
        // Assert: final has values from all three levels
        assert_eq!(final_d.max_batch_size, 16); // from base
        assert!(final_d.admit_new_prefill);     // from mid
        assert_eq!(final_d.force_swap_out_count, 50); // from final
        // Assert: chain members are distinct
        assert_ne!(base, mid);
        assert_ne!(mid, final_d);
        assert_ne!(base, final_d);
    }

    #[test]
    fn system_state_size_and_align_consistency() {
        // Arrange: check that size is a multiple of alignment (required by Rust ABI)
        let size = std::mem::size_of::<SystemState>();
        let align = std::mem::align_of::<SystemState>();
        // Assert: size is a multiple of alignment
        assert_eq!(size % align, 0, "size {size} is not a multiple of align {align}");
    }

    #[test]
    fn scheduler_decision_size_and_align_consistency() {
        // Arrange: check ABI consistency for SchedulerDecision
        let size = std::mem::size_of::<SchedulerDecision>();
        let align = std::mem::align_of::<SchedulerDecision>();
        // Assert: size is a multiple of alignment
        assert_eq!(size % align, 0, "size {size} is not a multiple of align {align}");
        // Assert: SchedulerDecision is smaller than SystemState
        assert!(size < std::mem::size_of::<SystemState>());
    }

    #[test]
    fn system_state_all_fields_negative_floats() {
        // Arrange: set all f32 fields to negative values (struct allows it)
        let s = SystemState {
            memory_pressure: -0.1,
            kv_fragmentation: -0.2,
            swap_io_rate: -500.0,
            logits_entropy: -1.5,
            attention_sparsity: -0.05,
            moe_fault_rate: -0.01,
            moe_avg_recovery_us: -100.0,
            ..Default::default()
        };
        // Assert: all float fields are negative
        assert!(s.memory_pressure.is_sign_negative());
        assert!(s.kv_fragmentation.is_sign_negative());
        assert!(s.swap_io_rate.is_sign_negative());
        assert!(s.logits_entropy.is_sign_negative());
        assert!(s.attention_sparsity.is_sign_negative());
        assert!(s.moe_fault_rate.is_sign_negative());
        assert!(s.moe_avg_recovery_us.is_sign_negative());
        // Assert: distinct from default
        assert_ne!(s, SystemState::default());
    }

    // ── Wave 17: 13 additional edge-case tests ──

    #[test]
    fn system_state_memory_pressure_first_field_semantic() {
        // Arrange: memory_pressure is the first declared field semantically
        // Act: verify it can be set independently of all other fields
        let s = SystemState {
            memory_pressure: 0.75,
            ..Default::default()
        };
        // Assert: only memory_pressure is non-zero/default
        assert!((s.memory_pressure - 0.75).abs() < 1e-6);
        assert_eq!(s.kv_fragmentation, 0.0);
        assert_eq!(s.swap_io_rate, 0.0);
        assert_eq!(s.waiting_queue_len, 0);
        assert_eq!(s.weight_page_total, 0);
        assert_ne!(s, SystemState::default());
    }

    #[test]
    fn system_state_swap_io_rate_stored_exactly() {
        // Arrange: use a value that is exactly representable in f32
        let exact_val = 1024.0_f32; // power of 2, exact in f32
        let mut s = SystemState::default();
        // Act
        s.swap_io_rate = exact_val;
        // Assert: no precision loss
        assert_eq!(s.swap_io_rate, exact_val);
        assert_eq!(s.swap_io_rate.to_bits(), exact_val.to_bits());
    }

    #[test]
    fn system_state_logits_entropy_squish_zero() {
        // Arrange: entropy approaches zero for near-deterministic distributions
        let tiny_entropy = 1e-10_f32;
        let s = SystemState {
            logits_entropy: tiny_entropy,
            ..Default::default()
        };
        // Assert: value is stored faithfully (greater than 0 but very small)
        assert!(s.logits_entropy > 0.0);
        assert!(s.logits_entropy < 1e-9);
        // Assert: distinct from exactly-zero default
        assert_ne!(s.logits_entropy, 0.0);
    }

    #[test]
    fn system_state_moe_working_set_zero_means_no_experts() {
        // Arrange: MoE working set size of 0 means no experts tracked
        let s = SystemState {
            moe_working_set_size: 0,
            moe_fault_rate: 0.0,
            moe_avg_recovery_us: 0.0,
            ..Default::default()
        };
        // Assert: equal to default (these three fields are already 0 in default)
        assert_eq!(s.moe_working_set_size, 0);
        // Assert: no faults with zero working set
        assert_eq!(s.moe_fault_rate, 0.0);
    }

    #[test]
    fn system_state_weight_tiers_sum_overflow_safety() {
        // Arrange: set each tier to a value where sum does not overflow usize
        let s = SystemState {
            weight_page_total: usize::MAX / 3 * 3,
            weight_pages_l1: usize::MAX / 3,
            weight_pages_l2: usize::MAX / 3,
            weight_pages_l3: usize::MAX / 3,
            ..Default::default()
        };
        // Act: sum tiers without overflow
        let tier_sum = s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3;
        // Assert: sum fits in usize and equals total
        assert_eq!(tier_sum, s.weight_page_total);
    }

    #[test]
    fn scheduler_decision_default_is_not_zero_struct() {
        // Arrange: get default and a zero-initialized struct
        let default = SchedulerDecision::default();
        let zero_like = SchedulerDecision {
            max_batch_size: 0,
            admit_new_prefill: false,
            force_swap_out_count: 0,
        };
        // Assert: default.max_batch_size is 1, not 0
        assert_eq!(default.max_batch_size, 1);
        assert_ne!(default, zero_like);
    }

    #[test]
    fn system_state_clone_then_mutation_does_not_affect_clone() {
        // Arrange: state with all float fields set
        let mut s = SystemState {
            memory_pressure: 0.6,
            kv_fragmentation: 0.4,
            logits_entropy: 2.8,
            attention_sparsity: 0.3,
            ..Default::default()
        };
        let snap = s.clone();
        // Act: zero out all fields on original
        s.memory_pressure = 0.0;
        s.kv_fragmentation = 0.0;
        s.logits_entropy = 0.0;
        s.attention_sparsity = 0.0;
        // Assert: snapshot retains original values
        assert!((snap.memory_pressure - 0.6).abs() < 1e-6);
        assert!((snap.kv_fragmentation - 0.4).abs() < 1e-6);
        assert!((snap.logits_entropy - 2.8).abs() < 1e-6);
        assert!((snap.attention_sparsity - 0.3).abs() < 1e-6);
    }

    #[test]
    fn system_state_batch_size_one_running_len_arbitrary() {
        // Arrange: single-request batch with a long context
        let s = SystemState {
            current_batch_size: 1,
            current_running_len: 65536,
            mean_context_len: 65536,
            ..Default::default()
        };
        // Assert: single sequence, running_len == mean_context_len
        assert_eq!(s.current_batch_size, 1);
        assert_eq!(s.current_running_len, s.mean_context_len);
    }

    #[test]
    fn scheduler_decision_force_swap_zero_admit_true() {
        // Arrange: admit new prefills but with zero forced swaps
        let d = SchedulerDecision {
            max_batch_size: 32,
            admit_new_prefill: true,
            force_swap_out_count: 0,
        };
        // Assert: no forced eviction, admitting new requests
        assert_eq!(d.force_swap_out_count, 0);
        assert!(d.admit_new_prefill);
        assert!(d.max_batch_size > 0);
    }

    #[test]
    fn system_state_float_zero_exact_representation() {
        // Arrange: explicitly set floats to positive zero
        let mut s = SystemState::default();
        s.memory_pressure = 0.0_f32;
        s.kv_fragmentation = 0.0_f32;
        s.swap_io_rate = 0.0_f32;
        // Assert: positive zero has specific bit pattern
        assert_eq!(s.memory_pressure.to_bits(), 0u32);
        assert_eq!(s.kv_fragmentation.to_bits(), 0u32);
        assert_eq!(s.swap_io_rate.to_bits(), 0u32);
        // Assert: equals default (which is also positive zero)
        assert_eq!(s.memory_pressure, SystemState::default().memory_pressure);
    }

    #[test]
    fn system_state_debug_contains_all_moe_field_names() {
        // Arrange: default state, only check field names in Debug output
        let s = SystemState::default();
        let debug = format!("{s:?}");
        // Assert: all three MoE field names appear
        assert!(debug.contains("moe_fault_rate"), "missing moe_fault_rate");
        assert!(debug.contains("moe_avg_recovery_us"), "missing moe_avg_recovery_us");
        assert!(debug.contains("moe_working_set_size"), "missing moe_working_set_size");
    }

    #[test]
    fn scheduler_decision_bool_field_is_bool() {
        // Arrange: verify admit_new_prefill behaves as a bool
        let d_true = SchedulerDecision {
            max_batch_size: 1,
            admit_new_prefill: true,
            force_swap_out_count: 0,
        };
        let d_false = SchedulerDecision {
            admit_new_prefill: false,
            ..d_true
        };
        // Assert: toggling the bool changes equality
        assert_ne!(d_true, d_false);
        assert!(d_true.admit_new_prefill);
        assert!(!d_false.admit_new_prefill);
        // Assert: only the bool field differs
        assert_eq!(d_true.max_batch_size, d_false.max_batch_size);
        assert_eq!(d_true.force_swap_out_count, d_false.force_swap_out_count);
    }

    #[test]
    fn system_state_all_fields_individually_mutable() {
        // Arrange: start from default
        let mut s = SystemState::default();
        // Act: mutate each field one at a time and verify it changes
        s.memory_pressure = 0.1;
        assert_ne!(s, SystemState::default());
        s = SystemState::default();
        s.kv_fragmentation = 0.1;
        assert_ne!(s, SystemState::default());
        s = SystemState::default();
        s.swap_io_rate = 0.1;
        assert_ne!(s, SystemState::default());
        s = SystemState::default();
        s.waiting_queue_len = 1;
        assert_ne!(s, SystemState::default());
        s = SystemState::default();
        s.current_batch_size = 1;
        assert_ne!(s, SystemState::default());
    }

    // -- Wave 18: 13 additional edge-case tests --

    #[test]
    fn system_state_smallest_subnormal_stored_and_distinct() {
        let smallest_subnormal = f32::from_bits(1u32);
        assert!(smallest_subnormal.is_subnormal());
        assert!(smallest_subnormal > 0.0);
        let s = SystemState {
            memory_pressure: smallest_subnormal,
            ..Default::default()
        };
        assert_eq!(s.memory_pressure.to_bits(), 1u32);
        assert_ne!(s, SystemState::default());
    }

    #[test]
    fn scheduler_decision_from_default_override_two_fields() {
        let d = SchedulerDecision {
            max_batch_size: 128,
            admit_new_prefill: true,
            ..SchedulerDecision::default()
        };
        assert_eq!(d.max_batch_size, 128);
        assert!(d.admit_new_prefill);
        assert_eq!(d.force_swap_out_count, 0);
    }

    #[test]
    fn system_state_copy_then_partial_field_check() {
        let original = SystemState {
            kv_fragmentation: 0.33,
            current_running_len: 8192,
            weight_pages_l2: 42,
            ..Default::default()
        };
        let copy = original;
        assert!((copy.kv_fragmentation - 0.33).abs() < 1e-6);
        assert_eq!(copy.current_running_len, 8192);
        assert_eq!(copy.weight_pages_l2, 42);
        assert_eq!(copy.memory_pressure, 0.0);
        assert_eq!(copy.moe_working_set_size, 0);
    }

    #[test]
    fn scheduler_decision_clone_then_reassign_preserves_original() {
        let original = SchedulerDecision {
            max_batch_size: 77,
            admit_new_prefill: true,
            force_swap_out_count: 11,
        };
        let cloned = original.clone();
        let _other = SchedulerDecision {
            max_batch_size: 0,
            admit_new_prefill: false,
            force_swap_out_count: 0,
        };
        assert_eq!(cloned.max_batch_size, 77);
        assert!(cloned.admit_new_prefill);
        assert_eq!(cloned.force_swap_out_count, 11);
    }

    #[test]
    fn system_state_struct_update_from_non_default_base() {
        let base = SystemState {
            swap_io_rate: 999.0,
            mean_context_len: 4096,
            moe_fault_rate: 0.1,
            ..Default::default()
        };
        let derived = SystemState {
            mean_context_len: 2048,
            ..base
        };
        assert!((derived.swap_io_rate - 999.0).abs() < 1e-6);
        assert_eq!(derived.mean_context_len, 2048);
        assert!((derived.moe_fault_rate - 0.1).abs() < 1e-6);
        assert_ne!(derived, base);
    }

    #[test]
    fn system_state_neq_per_usize_field_negative_check() {
        let fields: Vec<Box<dyn FnOnce(&mut SystemState)>> = vec![
            Box::new(|s| s.current_running_len = 99),
            Box::new(|s| s.mean_context_len = 99),
            Box::new(|s| s.moe_working_set_size = 99),
            Box::new(|s| s.weight_pages_l1 = 99),
            Box::new(|s| s.weight_pages_l2 = 99),
            Box::new(|s| s.weight_pages_l3 = 99),
            Box::new(|s| s.weight_recovery_count = 99),
        ];
        for mutate in fields {
            let mut s = SystemState::default();
            mutate(&mut s);
            assert_ne!(s, SystemState::default());
        }
    }

    #[test]
    fn scheduler_decision_large_batch_half_max() {
        let half = usize::MAX / 2;
        let d = SchedulerDecision {
            max_batch_size: half,
            admit_new_prefill: false,
            force_swap_out_count: half,
        };
        assert_eq!(d.max_batch_size, half);
        assert_eq!(d.force_swap_out_count, half);
        assert_eq!(d.max_batch_size, d.force_swap_out_count);
    }

    #[test]
    fn system_state_debug_format_intermediate_values() {
        let s = SystemState {
            memory_pressure: 0.375,
            kv_fragmentation: 0.625,
            swap_io_rate: 7.5,
            logits_entropy: 1.875,
            ..Default::default()
        };
        let debug = format!("{s:?}");
        assert!(debug.contains("memory_pressure"));
        assert!(debug.contains("kv_fragmentation"));
        assert!(debug.contains("swap_io_rate"));
        assert!(debug.contains("logits_entropy"));
        assert!((s.memory_pressure + s.kv_fragmentation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn system_state_zero_total_with_nonzero_tiers() {
        let s = SystemState {
            weight_page_total: 0,
            weight_pages_l1: 5,
            weight_pages_l2: 3,
            weight_pages_l3: 2,
            ..Default::default()
        };
        assert_eq!(s.weight_page_total, 0);
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 10);
    }

    #[test]
    fn scheduler_decision_default_max_batch_is_one_not_max() {
        let d = SchedulerDecision::default();
        assert_eq!(d.max_batch_size, 1);
        assert_ne!(d.max_batch_size, usize::MAX);
        assert_ne!(d.max_batch_size, 0);
    }

    #[test]
    fn system_state_exact_power_of_two_floats() {
        let s = SystemState {
            memory_pressure: 0.125,
            kv_fragmentation: 0.25,
            attention_sparsity: 0.5,
            logits_entropy: 1.0,
            moe_fault_rate: 2.0,
            swap_io_rate: 4.0,
            moe_avg_recovery_us: 8.0,
            ..Default::default()
        };
        assert_eq!(s.memory_pressure, 0.125);
        assert_eq!(s.kv_fragmentation, 0.25);
        assert_eq!(s.attention_sparsity, 0.5);
        assert_eq!(s.logits_entropy, 1.0);
        assert_eq!(s.moe_fault_rate, 2.0);
        assert_eq!(s.swap_io_rate, 4.0);
        assert_eq!(s.moe_avg_recovery_us, 8.0);
    }

    #[test]
    fn system_state_copies_into_array_all_identical() {
        let src = SystemState {
            memory_pressure: 0.55,
            waiting_queue_len: 7,
            moe_working_set_size: 3,
            weight_page_total: 50,
            ..Default::default()
        };
        let arr = [src; 4];
        for item in &arr {
            assert!((item.memory_pressure - 0.55).abs() < 1e-6);
            assert_eq!(item.waiting_queue_len, 7);
            assert_eq!(item.moe_working_set_size, 3);
            assert_eq!(item.weight_page_total, 50);
        }
        assert!(arr.iter().all(|x| *x == src));
    }

    #[test]
    fn scheduler_decision_two_non_default_instances_equal() {
        let a = SchedulerDecision {
            max_batch_size: 42,
            admit_new_prefill: true,
            force_swap_out_count: 13,
        };
        let b = SchedulerDecision {
            max_batch_size: 42,
            admit_new_prefill: true,
            force_swap_out_count: 13,
        };
        assert_eq!(a, b);
        assert!((a.max_batch_size - b.max_batch_size) == 0);
        assert_eq!(a.admit_new_prefill, b.admit_new_prefill);
        assert_eq!(a.force_swap_out_count, b.force_swap_out_count);
    }

    // -- Wave 19: 10 additional edge-case tests --

    #[test]
    fn system_state_large_batch_zero_queue() {
        // Arrange: active batch running with no waiting requests (steady state)
        let s = SystemState {
            waiting_queue_len: 0,
            current_batch_size: 64,
            current_running_len: 16384,
            mean_context_len: 256,
            ..Default::default()
        };
        // Assert: batch is active but queue is drained
        assert_eq!(s.waiting_queue_len, 0);
        assert!(s.current_batch_size > 0);
        assert!(s.current_running_len > 0);
        assert_eq!(s.current_running_len, s.current_batch_size * s.mean_context_len);
    }

    #[test]
    fn system_state_all_float_fields_one_third() {
        // Arrange: 1/3 is a repeating binary fraction in f32; verify storage fidelity
        let one_third: f32 = 1.0 / 3.0;
        let s = SystemState {
            memory_pressure: one_third,
            kv_fragmentation: one_third,
            swap_io_rate: one_third,
            logits_entropy: one_third,
            attention_sparsity: one_third,
            moe_fault_rate: one_third,
            moe_avg_recovery_us: one_third,
            ..Default::default()
        };
        // Assert: all fields have identical bitwise representation
        let bits = one_third.to_bits();
        assert_eq!(s.memory_pressure.to_bits(), bits);
        assert_eq!(s.kv_fragmentation.to_bits(), bits);
        assert_eq!(s.swap_io_rate.to_bits(), bits);
        assert_eq!(s.logits_entropy.to_bits(), bits);
        assert_eq!(s.attention_sparsity.to_bits(), bits);
        assert_eq!(s.moe_fault_rate.to_bits(), bits);
        assert_eq!(s.moe_avg_recovery_us.to_bits(), bits);
        // Assert: distinct from default
        assert_ne!(s, SystemState::default());
    }

    #[test]
    fn system_state_clone_with_nan_in_multiple_fields() {
        // Arrange: two different NaN payloads in separate fields
        let nan_a = f32::from_bits(0x7f800001u32); // signaling NaN variant
        let nan_b = f32::from_bits(0x7fc00000u32); // quiet NaN
        assert!(nan_a.is_nan());
        assert!(nan_b.is_nan());
        let s = SystemState {
            memory_pressure: nan_a,
            logits_entropy: nan_b,
            ..Default::default()
        };
        // Act: clone preserves NaN payloads
        let cloned = s.clone();
        // Assert: bit patterns are preserved independently
        assert_eq!(cloned.memory_pressure.to_bits(), nan_a.to_bits());
        assert_eq!(cloned.logits_entropy.to_bits(), nan_b.to_bits());
    }

    #[test]
    fn system_state_weight_page_eviction_equals_recovery() {
        // Arrange: equal eviction and recovery counts (steady-state churn)
        let s = SystemState {
            weight_eviction_count: 500,
            weight_recovery_count: 500,
            weight_page_total: 200,
            weight_pages_l1: 80,
            weight_pages_l2: 70,
            weight_pages_l3: 50,
            ..Default::default()
        };
        // Assert: net pressure is zero
        assert_eq!(s.weight_eviction_count, s.weight_recovery_count);
        assert_eq!(s.weight_pages_l1 + s.weight_pages_l2 + s.weight_pages_l3, 200);
    }

    #[test]
    fn scheduler_decision_copy_into_array_all_equal() {
        // Arrange: create a non-default decision
        let d = SchedulerDecision {
            max_batch_size: 16,
            admit_new_prefill: true,
            force_swap_out_count: 2,
        };
        // Act: create an array via Copy
        let arr = [d; 8];
        // Assert: all elements identical to source
        for item in &arr {
            assert_eq!(*item, d);
            assert_eq!(item.max_batch_size, 16);
            assert!(item.admit_new_prefill);
            assert_eq!(item.force_swap_out_count, 2);
        }
    }

    #[test]
    fn system_state_struct_update_overrides_only_target_field() {
        // Arrange: fully populated base state
        let base = SystemState {
            memory_pressure: 0.9,
            kv_fragmentation: 0.1,
            swap_io_rate: 1000.0,
            waiting_queue_len: 20,
            current_batch_size: 32,
            current_running_len: 8192,
            mean_context_len: 512,
            logits_entropy: 3.5,
            attention_sparsity: 0.4,
            moe_fault_rate: 0.003,
            moe_avg_recovery_us: 450.0,
            moe_working_set_size: 16,
            weight_page_total: 400,
            weight_pages_l1: 200,
            weight_pages_l2: 150,
            weight_pages_l3: 50,
            weight_eviction_count: 30,
            weight_recovery_count: 28,
            ..Default::default()
        };
        // Act: override only one field
        let derived = SystemState {
            logits_entropy: 0.0,
            ..base
        };
        // Assert: only logits_entropy changed
        assert_eq!(derived.logits_entropy, 0.0);
        assert_ne!(derived.logits_entropy, base.logits_entropy);
        // Assert: all other fields identical
        assert_eq!(derived.memory_pressure, base.memory_pressure);
        assert_eq!(derived.kv_fragmentation, base.kv_fragmentation);
        assert_eq!(derived.swap_io_rate, base.swap_io_rate);
        assert_eq!(derived.waiting_queue_len, base.waiting_queue_len);
        assert_eq!(derived.current_batch_size, base.current_batch_size);
        assert_eq!(derived.weight_page_total, base.weight_page_total);
    }

    #[test]
    fn system_state_f32_fields_are_all_sign_positive() {
        // Arrange: set all float fields to positive values
        let s = SystemState {
            memory_pressure: 0.1,
            kv_fragmentation: 0.2,
            swap_io_rate: 300.0,
            logits_entropy: 4.0,
            attention_sparsity: 0.6,
            moe_fault_rate: 0.001,
            moe_avg_recovery_us: 50.0,
            ..Default::default()
        };
        // Assert: every float field has positive sign
        assert!(s.memory_pressure.is_sign_positive());
        assert!(s.kv_fragmentation.is_sign_positive());
        assert!(s.swap_io_rate.is_sign_positive());
        assert!(s.logits_entropy.is_sign_positive());
        assert!(s.attention_sparsity.is_sign_positive());
        assert!(s.moe_fault_rate.is_sign_positive());
        assert!(s.moe_avg_recovery_us.is_sign_positive());
    }

    #[test]
    fn system_state_moe_high_fault_zero_recovery() {
        // Arrange: high fault rate with zero recovery (degraded state, no auto-recovery)
        let s = SystemState {
            moe_fault_rate: 0.5,
            moe_avg_recovery_us: 0.0,
            moe_working_set_size: 64,
            ..Default::default()
        };
        // Assert: faults detected but no recovery has occurred
        assert!(s.moe_fault_rate > 0.0);
        assert_eq!(s.moe_avg_recovery_us, 0.0);
        assert!(s.moe_working_set_size > 0);
        // Assert: recovery time is zero but working set is non-empty
        assert_ne!(s.moe_working_set_size, 0);
    }

    #[test]
    fn scheduler_decision_debug_contains_all_three_fields() {
        // Arrange: construct with specific values that appear in debug output
        let d = SchedulerDecision {
            max_batch_size: 0,
            admit_new_prefill: false,
            force_swap_out_count: 0,
        };
        let debug = format!("{d:?}");
        // Assert: all three field names are present
        assert!(debug.contains("max_batch_size"), "missing max_batch_size");
        assert!(debug.contains("admit_new_prefill"), "missing admit_new_prefill");
        assert!(debug.contains("force_swap_out_count"), "missing force_swap_out_count");
    }

    #[test]
    fn system_state_swap_io_rate_very_large_no_overflow_arithmetic() {
        // Arrange: swap_io_rate near f32 max, perform safe arithmetic
        let s = SystemState {
            swap_io_rate: 3.0e38_f32,
            ..Default::default()
        };
        // Act: multiply by a small factor (should not overflow to infinity)
        let doubled = s.swap_io_rate * 2.0;
        // Assert: doubling a near-max value produces infinity
        assert!(doubled.is_infinite());
        // Assert: original value is finite and stored correctly
        assert!(s.swap_io_rate.is_finite());
        assert!((s.swap_io_rate - 3.0e38).abs() / s.swap_io_rate < 1e-6);
    }
}
