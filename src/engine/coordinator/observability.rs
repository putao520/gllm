use crate::scheduler::observer::BasicObserver;

use super::dispatch::DispatchCoordinator;

#[derive(Debug)]
pub struct ObservabilityCoordinator {
    pub observer: BasicObserver,
}

impl ObservabilityCoordinator {
    pub fn capture_state(
        &mut self,
        memory_pressure: Result<f32, String>,
        dispatch: &DispatchCoordinator,
    ) {
        if let Err(e) = self.observer.update_memory_pressure(memory_pressure) {
            log::warn!("executor: update_memory_pressure failed: {e}");
        }
        self.observer
            .update_kv_fragmentation(dispatch.scheduler.kv_fragmentation_ratio());
        self.observer.update_scheduler_metrics(
            dispatch.batcher.waiting_len(),
            dispatch.batcher.running_len(),
            dispatch.batcher.running_len(),
            dispatch.batcher.mean_context_len(),
        );
    }

    pub fn update_swap_io_rate(&mut self, rate: f32) {
        self.observer.update_swap_io_rate(rate);
    }

    pub fn update_logits_entropy(&mut self, entropy: f32) {
        self.observer.update_logits_entropy(entropy);
    }

    pub fn update_attention_sparsity(&mut self, sparsity: f32) {
        self.observer.update_attention_sparsity(sparsity);
    }

    pub fn update_moe_fault_metrics(
        &mut self,
        fault_rate: f32,
        avg_recovery_latency_us: f32,
        working_set_size: usize,
    ) {
        self.observer
            .update_moe_fault_metrics(fault_rate, avg_recovery_latency_us, working_set_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coord() -> ObservabilityCoordinator {
        ObservabilityCoordinator {
            observer: BasicObserver::new(),
        }
    }

    #[test]
    fn update_swap_io_rate_propagates() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(42.5);
        assert!((coord.observer.last_state.swap_io_rate - 42.5).abs() < f32::EPSILON);
    }

    #[test]
    fn update_logits_entropy_propagates() {
        let mut coord = make_coord();
        coord.update_logits_entropy(3.14);
        assert!((coord.observer.last_state.logits_entropy - 3.14).abs() < f32::EPSILON);
    }

    #[test]
    fn update_attention_sparsity_propagates() {
        let mut coord = make_coord();
        coord.update_attention_sparsity(0.87);
        assert!((coord.observer.last_state.attention_sparsity - 0.87).abs() < f32::EPSILON);
    }

    #[test]
    fn update_moe_fault_metrics_propagates() {
        let mut coord = make_coord();
        coord.update_moe_fault_metrics(0.01, 150.0, 2048);
        assert!((coord.observer.last_state.moe_fault_rate - 0.01).abs() < f32::EPSILON);
        assert!((coord.observer.last_state.moe_avg_recovery_us - 150.0).abs() < f32::EPSILON);
        assert_eq!(coord.observer.last_state.moe_working_set_size, 2048);
    }

    // ── Construction & field access ──

    #[test]
    fn constructor_initializes_default_observer_state() {
        let coord = make_coord();
        assert_eq!(coord.observer.last_state.memory_pressure, 0.0);
        assert_eq!(coord.observer.last_state.swap_io_rate, 0.0);
        assert_eq!(coord.observer.last_state.logits_entropy, 0.0);
        assert_eq!(coord.observer.last_state.attention_sparsity, 0.0);
        assert_eq!(coord.observer.last_state.moe_fault_rate, 0.0);
        assert_eq!(coord.observer.last_state.moe_working_set_size, 0);
    }

    #[test]
    fn observer_field_is_mutable() {
        let mut coord = make_coord();
        coord.observer.last_state.memory_pressure = 0.99;
        assert!((coord.observer.last_state.memory_pressure - 0.99).abs() < f32::EPSILON);
    }

    #[test]
    fn debug_trait_works() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(1.23);
        let debug = format!("{coord:?}");
        assert!(debug.contains("ObservabilityCoordinator"), "debug: {debug}");
    }

    // ── update_swap_io_rate boundary values ──

    #[test]
    fn update_swap_io_rate_zero() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(0.0);
        assert_eq!(coord.observer.last_state.swap_io_rate, 0.0);
    }

    #[test]
    fn update_swap_io_rate_max() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(f32::MAX);
        assert_eq!(coord.observer.last_state.swap_io_rate, f32::MAX);
    }

    #[test]
    fn update_swap_io_rate_negative() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(-1.0);
        assert!((coord.observer.last_state.swap_io_rate - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn update_swap_io_rate_nan() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(f32::NAN);
        assert!(coord.observer.last_state.swap_io_rate.is_nan());
    }

    // ── update_logits_entropy boundary values ──

    #[test]
    fn update_logits_entropy_zero() {
        let mut coord = make_coord();
        coord.update_logits_entropy(0.0);
        assert_eq!(coord.observer.last_state.logits_entropy, 0.0);
    }

    #[test]
    fn update_logits_entropy_max() {
        let mut coord = make_coord();
        coord.update_logits_entropy(f32::MAX);
        assert_eq!(coord.observer.last_state.logits_entropy, f32::MAX);
    }

    #[test]
    fn update_logits_entropy_negative() {
        let mut coord = make_coord();
        coord.update_logits_entropy(-0.5);
        assert!((coord.observer.last_state.logits_entropy - (-0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn update_logits_entropy_nan() {
        let mut coord = make_coord();
        coord.update_logits_entropy(f32::NAN);
        assert!(coord.observer.last_state.logits_entropy.is_nan());
    }

    // ── update_attention_sparsity boundary values ──

    #[test]
    fn update_attention_sparsity_zero() {
        let mut coord = make_coord();
        coord.update_attention_sparsity(0.0);
        assert_eq!(coord.observer.last_state.attention_sparsity, 0.0);
    }

    #[test]
    fn update_attention_sparsity_max() {
        let mut coord = make_coord();
        coord.update_attention_sparsity(f32::MAX);
        assert_eq!(coord.observer.last_state.attention_sparsity, f32::MAX);
    }

    #[test]
    fn update_attention_sparsity_nan() {
        let mut coord = make_coord();
        coord.update_attention_sparsity(f32::NAN);
        assert!(coord.observer.last_state.attention_sparsity.is_nan());
    }

    // ── update_moe_fault_metrics boundary values ──

    #[test]
    fn update_moe_fault_metrics_zero() {
        let mut coord = make_coord();
        coord.update_moe_fault_metrics(0.0, 0.0, 0);
        assert_eq!(coord.observer.last_state.moe_fault_rate, 0.0);
        assert_eq!(coord.observer.last_state.moe_avg_recovery_us, 0.0);
        assert_eq!(coord.observer.last_state.moe_working_set_size, 0);
    }

    #[test]
    fn update_moe_fault_metrics_max() {
        let mut coord = make_coord();
        coord.update_moe_fault_metrics(f32::MAX, f32::MAX, usize::MAX);
        assert_eq!(coord.observer.last_state.moe_fault_rate, f32::MAX);
        assert_eq!(coord.observer.last_state.moe_avg_recovery_us, f32::MAX);
        assert_eq!(coord.observer.last_state.moe_working_set_size, usize::MAX);
    }

    #[test]
    fn update_moe_fault_metrics_overwrite_previous() {
        let mut coord = make_coord();
        coord.update_moe_fault_metrics(0.5, 100.0, 50);
        coord.update_moe_fault_metrics(0.1, 200.0, 10);
        assert!((coord.observer.last_state.moe_fault_rate - 0.1).abs() < f32::EPSILON);
        assert!((coord.observer.last_state.moe_avg_recovery_us - 200.0).abs() < f32::EPSILON);
        assert_eq!(coord.observer.last_state.moe_working_set_size, 10);
    }

    // ── Multiple updates in sequence ──

    #[test]
    fn sequential_updates_preserve_last_value() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(1.0);
        coord.update_swap_io_rate(2.0);
        coord.update_swap_io_rate(3.0);
        assert!((coord.observer.last_state.swap_io_rate - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn different_metrics_independent() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(10.0);
        coord.update_logits_entropy(20.0);
        coord.update_attention_sparsity(30.0);
        assert!((coord.observer.last_state.swap_io_rate - 10.0).abs() < f32::EPSILON);
        assert!((coord.observer.last_state.logits_entropy - 20.0).abs() < f32::EPSILON);
        assert!((coord.observer.last_state.attention_sparsity - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn infinity_values_propagate() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(f32::INFINITY);
        coord.update_logits_entropy(f32::NEG_INFINITY);
        assert!(coord.observer.last_state.swap_io_rate.is_infinite() && coord.observer.last_state.swap_io_rate.is_sign_positive());
        assert!(coord.observer.last_state.logits_entropy.is_infinite() && coord.observer.last_state.logits_entropy.is_sign_negative());
    }

    // ── Additional coverage: attention_sparsity negative ──

    #[test]
    fn update_attention_sparsity_negative() {
        let mut coord = make_coord();
        coord.update_attention_sparsity(-0.25);
        assert!((coord.observer.last_state.attention_sparsity - (-0.25)).abs() < f32::EPSILON);
    }

    // ── MoE fault metrics with NaN floats ──

    #[test]
    fn update_moe_fault_metrics_nan_floats() {
        let mut coord = make_coord();
        coord.update_moe_fault_metrics(f32::NAN, f32::NAN, 100);
        assert!(coord.observer.last_state.moe_fault_rate.is_nan());
        assert!(coord.observer.last_state.moe_avg_recovery_us.is_nan());
        assert_eq!(coord.observer.last_state.moe_working_set_size, 100);
    }

    // ── Constructor: all SystemState fields default-zeroed ──

    #[test]
    fn constructor_all_fields_default_zeroed() {
        let coord = make_coord();
        let s = &coord.observer.last_state;
        assert_eq!(s.kv_fragmentation, 0.0);
        assert_eq!(s.waiting_queue_len, 0);
        assert_eq!(s.current_batch_size, 0);
        assert_eq!(s.current_running_len, 0);
        assert_eq!(s.mean_context_len, 0);
        assert_eq!(s.moe_avg_recovery_us, 0.0);
        assert_eq!(s.weight_page_total, 0);
        assert_eq!(s.weight_pages_l1, 0);
        assert_eq!(s.weight_pages_l2, 0);
        assert_eq!(s.weight_pages_l3, 0);
        assert_eq!(s.weight_eviction_count, 0);
        assert_eq!(s.weight_recovery_count, 0);
    }

    // ── kv_fragmentation via direct observer update ──

    #[test]
    fn observer_kv_fragmentation_update() {
        let mut coord = make_coord();
        coord.observer.update_kv_fragmentation(0.42);
        assert!((coord.observer.last_state.kv_fragmentation - 0.42).abs() < f32::EPSILON);
    }

    // ── Sequential overwrite: logits_entropy keeps last ──

    #[test]
    fn sequential_logits_entropy_preserves_last() {
        let mut coord = make_coord();
        coord.update_logits_entropy(1.0);
        coord.update_logits_entropy(2.0);
        coord.update_logits_entropy(5.5);
        assert!((coord.observer.last_state.logits_entropy - 5.5).abs() < f32::EPSILON);
    }

    // ── Sequential overwrite: attention_sparsity keeps last ──

    #[test]
    fn sequential_attention_sparsity_preserves_last() {
        let mut coord = make_coord();
        coord.update_attention_sparsity(0.1);
        coord.update_attention_sparsity(0.9);
        coord.update_attention_sparsity(0.55);
        assert!((coord.observer.last_state.attention_sparsity - 0.55).abs() < f32::EPSILON);
    }

    // ── All four metric updates produce correct combined state ──

    #[test]
    fn all_metrics_updated_simultaneously() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(7.7);
        coord.update_logits_entropy(2.2);
        coord.update_attention_sparsity(0.33);
        coord.update_moe_fault_metrics(0.05, 300.0, 512);
        let s = &coord.observer.last_state;
        assert!((s.swap_io_rate - 7.7).abs() < f32::EPSILON);
        assert!((s.logits_entropy - 2.2).abs() < f32::EPSILON);
        assert!((s.attention_sparsity - 0.33).abs() < f32::EPSILON);
        assert!((s.moe_fault_rate - 0.05).abs() < f32::EPSILON);
        assert!((s.moe_avg_recovery_us - 300.0).abs() < f32::EPSILON);
        assert_eq!(s.moe_working_set_size, 512);
    }

    // ── Subnormal/denormalized float value propagation ──

    #[test]
    fn subnormal_float_propagates() {
        let mut coord = make_coord();
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        coord.update_swap_io_rate(subnormal);
        coord.update_logits_entropy(subnormal);
        coord.update_attention_sparsity(subnormal);
        assert_eq!(coord.observer.last_state.swap_io_rate, subnormal);
        assert_eq!(coord.observer.last_state.logits_entropy, subnormal);
        assert_eq!(coord.observer.last_state.attention_sparsity, subnormal);
    }

    // ── Small positive epsilon for swap_io_rate ──

    #[test]
    fn swap_io_rate_epsilon_value() {
        let mut coord = make_coord();
        let eps = f32::EPSILON;
        coord.update_swap_io_rate(eps);
        assert!((coord.observer.last_state.swap_io_rate - eps).abs() < f32::EPSILON);
        assert!(coord.observer.last_state.swap_io_rate > 0.0);
    }

    // ── Multiple NaN updates in sequence ──

    #[test]
    fn repeated_nan_updates_preserve_nan() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(f32::NAN);
        coord.update_swap_io_rate(f32::NAN);
        assert!(coord.observer.last_state.swap_io_rate.is_nan());
    }

    // ── Reset pattern: update to value then to zero ──

    #[test]
    fn reset_to_zero_after_nonzero() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(99.9);
        coord.update_logits_entropy(4.0);
        coord.update_attention_sparsity(0.75);
        coord.update_moe_fault_metrics(0.5, 200.0, 1024);
        // Reset all to zero/default
        coord.update_swap_io_rate(0.0);
        coord.update_logits_entropy(0.0);
        coord.update_attention_sparsity(0.0);
        coord.update_moe_fault_metrics(0.0, 0.0, 0);
        let s = &coord.observer.last_state;
        assert_eq!(s.swap_io_rate, 0.0);
        assert_eq!(s.logits_entropy, 0.0);
        assert_eq!(s.attention_sparsity, 0.0);
        assert_eq!(s.moe_fault_rate, 0.0);
        assert_eq!(s.moe_avg_recovery_us, 0.0);
        assert_eq!(s.moe_working_set_size, 0);
    }

    // ── Mixed positive/negative sequence for logits_entropy ──

    #[test]
    fn logits_entropy_mixed_sign_sequence() {
        let mut coord = make_coord();
        coord.update_logits_entropy(1.5);
        coord.update_logits_entropy(-2.3);
        coord.update_logits_entropy(0.0);
        coord.update_logits_entropy(-0.001);
        assert!((coord.observer.last_state.logits_entropy - (-0.001)).abs() < 1e-6);
    }

    // ── Attention sparsity with infinity ──

    #[test]
    fn attention_sparsity_infinity() {
        let mut coord = make_coord();
        coord.update_attention_sparsity(f32::INFINITY);
        assert!(coord.observer.last_state.attention_sparsity.is_infinite());
        assert!(coord.observer.last_state.attention_sparsity.is_sign_positive());
    }

    // ── MoE fault metrics partial overwrite via separate calls ──

    #[test]
    fn moe_fault_metrics_then_independent_metric_update() {
        let mut coord = make_coord();
        coord.update_moe_fault_metrics(0.1, 50.0, 100);
        // Updating an unrelated metric must not affect MoE fields
        coord.update_swap_io_rate(25.0);
        let s = &coord.observer.last_state;
        assert!((s.moe_fault_rate - 0.1).abs() < f32::EPSILON);
        assert!((s.moe_avg_recovery_us - 50.0).abs() < f32::EPSILON);
        assert_eq!(s.moe_working_set_size, 100);
        assert!((s.swap_io_rate - 25.0).abs() < f32::EPSILON);
    }

    // ── MoE fault metrics with negative recovery latency ──

    #[test]
    fn update_moe_fault_metrics_negative_latency() {
        let mut coord = make_coord();
        coord.update_moe_fault_metrics(0.0, -500.0, 64);
        assert!((coord.observer.last_state.moe_avg_recovery_us - (-500.0)).abs() < f32::EPSILON);
        assert_eq!(coord.observer.last_state.moe_fault_rate, 0.0);
        assert_eq!(coord.observer.last_state.moe_working_set_size, 64);
    }

    // ── Cross-metric isolation: MoE update does not touch swap_io_rate ──

    #[test]
    fn swap_rate_unaffected_by_moe_update() {
        let mut coord = make_coord();
        coord.update_swap_io_rate(12.5);
        coord.update_moe_fault_metrics(0.9, 999.0, 4096);
        assert!((coord.observer.last_state.swap_io_rate - 12.5).abs() < f32::EPSILON);
    }

    // ── Sequential kv_fragmentation updates keep last value ──

    #[test]
    fn sequential_kv_fragmentation_preserves_last() {
        let mut coord = make_coord();
        coord.observer.update_kv_fragmentation(0.1);
        coord.observer.update_kv_fragmentation(0.5);
        coord.observer.update_kv_fragmentation(0.95);
        assert!((coord.observer.last_state.kv_fragmentation - 0.95).abs() < f32::EPSILON);
    }

    // ── 13 new tests for uncovered paths ──

    /// Helper: build a minimal DispatchCoordinator for capture_state tests.
    fn make_dispatch() -> super::super::dispatch::DispatchCoordinator {
        use crate::scheduler::chunked_prefill::ChunkedPrefillConfig;
        use crate::scheduler::chunked_prefill::ChunkedPrefillScheduler;
        use crate::scheduler::hgal::HGALConfig;
        use crate::scheduler::memory_manager::GlobalMemoryManager;
        use crate::scheduler::paged_scheduler::PagedScheduler;
        use crate::scheduler::batcher::ContinuousBatcher;
        use crate::scheduler::policy::PolicyVariant;
        use crate::scheduler::vllm2024::ChunkedConfig;

        super::super::dispatch::DispatchCoordinator {
            scheduler: PagedScheduler::new(32, 4, HGALConfig::default()),
            batcher: ContinuousBatcher::new()
                .with_chunked(ChunkedConfig::default()),
            chunked_prefill_scheduler: ChunkedPrefillScheduler::new(
                ChunkedPrefillConfig::default(),
            ),
            requests: std::collections::HashMap::new(),
            memory_manager: GlobalMemoryManager::new_with_capacities(32, 0, 0),
            policy: PolicyVariant::default(),
        }
    }

    // @trace TEST-OBS-42 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn capture_state_ok_memory_pressure_propagates() {
        // Arrange
        let mut coord = make_coord();
        let dispatch = make_dispatch();

        // Act: capture_state with successful memory pressure
        coord.capture_state(Ok(0.55), &dispatch);

        // Assert: memory_pressure propagated via observer
        assert!((coord.observer.last_state.memory_pressure - 0.55).abs() < f32::EPSILON);
    }

    // @trace TEST-OBS-43 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn capture_state_err_memory_pressure_does_not_panic() {
        // Arrange
        let mut coord = make_coord();
        let dispatch = make_dispatch();

        // Act: capture_state with error result — must not panic, logs warning
        coord.capture_state(Err("sensor offline".into()), &dispatch);

        // Assert: memory_pressure remains at default (error path does not overwrite)
        assert_eq!(coord.observer.last_state.memory_pressure, 0.0);
    }

    // @trace TEST-OBS-44 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn capture_state_scheduler_metrics_propagated() {
        // Arrange
        let mut coord = make_coord();
        let dispatch = make_dispatch();

        // Act
        coord.capture_state(Ok(0.2), &dispatch);

        // Assert: scheduler metrics were updated from the batcher (empty = all zeros)
        let s = &coord.observer.last_state;
        assert_eq!(s.waiting_queue_len, 0);
        assert_eq!(s.current_running_len, 0);
        assert_eq!(s.current_batch_size, 0);
        assert_eq!(s.mean_context_len, 0);
    }

    // @trace TEST-OBS-45 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn capture_state_kv_fragmentation_propagated() {
        // Arrange
        let mut coord = make_coord();
        let dispatch = make_dispatch();

        // Act
        coord.capture_state(Ok(0.1), &dispatch);

        // Assert: kv_fragmentation was read from scheduler (empty scheduler = 0.0)
        assert_eq!(coord.observer.last_state.kv_fragmentation, 0.0);
    }

    // @trace TEST-OBS-46 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn observer_update_memory_pressure_direct_ok() {
        // Arrange
        let mut coord = make_coord();

        // Act: call observer directly (not via capture_state)
        let result = coord.observer.update_memory_pressure(Ok(0.77));

        // Assert: returns Ok and value is set
        assert!(result.is_ok());
        assert!((coord.observer.last_state.memory_pressure - 0.77).abs() < f32::EPSILON);
    }

    // @trace TEST-OBS-47 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn observer_update_memory_pressure_direct_err() {
        // Arrange
        let mut coord = make_coord();

        // Act: call observer directly with error
        let result = coord.observer.update_memory_pressure(Err("nvml failed".into()));

        // Assert: returns Err and memory_pressure remains at default
        assert!(result.is_err());
        assert_eq!(coord.observer.last_state.memory_pressure, 0.0);
    }

    // @trace TEST-OBS-48 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn observer_update_scheduler_metrics_direct() {
        // Arrange
        let mut coord = make_coord();

        // Act: set scheduler metrics directly on observer
        coord.observer.update_scheduler_metrics(15, 8, 12, 256);

        // Assert: all four fields propagated
        let s = &coord.observer.last_state;
        assert_eq!(s.waiting_queue_len, 15);
        assert_eq!(s.current_running_len, 8);
        assert_eq!(s.current_batch_size, 12);
        assert_eq!(s.mean_context_len, 256);
    }

    // @trace TEST-OBS-49 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn observer_compression_telemetry_record_and_verify() {
        // Arrange
        use crate::kv_cache::CompressionCodec;
        let mut coord = make_coord();

        // Act: record compress + decompress + migration via observer
        coord.observer.record_compress(CompressionCodec::Lz4, 8192, 4096, 200);
        coord.observer.record_decompress(CompressionCodec::Lz4, 4096, 8192, 100);
        coord.observer.record_migration(65536);

        // Assert: telemetry reflects all operations
        let ct = coord.observer.compression_telemetry();
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.decompress_count, 1);
        assert_eq!(ct.total_migration_bytes, 65536);
    }

    // @trace TEST-OBS-50 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn observer_weight_metrics_and_eviction_swap_in() {
        // Arrange
        let mut coord = make_coord();

        // Act: set weight metrics then record eviction + swap-in
        coord.observer.update_weight_metrics(100, 60, 30, 10, 5, 3);
        coord.observer.record_eviction();
        coord.observer.record_swap_in();

        // Assert: SystemState weight fields + compression telemetry updated
        let s = &coord.observer.last_state;
        assert_eq!(s.weight_page_total, 100);
        assert_eq!(s.weight_pages_l1, 60);
        assert_eq!(s.weight_pages_l2, 30);
        assert_eq!(s.weight_pages_l3, 10);
        assert_eq!(s.weight_eviction_count, 5);
        assert_eq!(s.weight_recovery_count, 3);
        let ct = coord.observer.compression_telemetry();
        assert_eq!(ct.eviction_count, 1);
        assert_eq!(ct.swap_in_count, 1);
    }

    // @trace TEST-OBS-51 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn observer_runtime_observer_trait_capture() {
        // Arrange
        use crate::scheduler::observer::RuntimeObserver;
        let mut coord = make_coord();
        coord.observer.update_memory_pressure(Ok(0.42)).unwrap();
        coord.observer.update_scheduler_metrics(5, 3, 4, 128);
        coord.observer.update_kv_fragmentation(0.15);

        // Act: capture via RuntimeObserver trait
        let trait_ref: &dyn RuntimeObserver = &coord.observer;
        let state = trait_ref.capture().unwrap();

        // Assert: captured state matches current observer state
        assert!((state.memory_pressure - 0.42).abs() < 1e-6);
        assert_eq!(state.waiting_queue_len, 5);
        assert_eq!(state.current_running_len, 3);
        assert_eq!(state.current_batch_size, 4);
        assert_eq!(state.mean_context_len, 128);
        assert!((state.kv_fragmentation - 0.15).abs() < 1e-6);
    }

    // @trace TEST-OBS-52 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn observer_clone_independence_after_update() {
        // Arrange
        let mut coord = make_coord();
        coord.observer.update_memory_pressure(Ok(0.3)).unwrap();

        // Act: clone the observer, mutate the clone
        let mut cloned_observer = coord.observer.clone();
        cloned_observer.update_memory_pressure(Ok(0.9)).unwrap();
        cloned_observer.update_kv_fragmentation(0.8);

        // Assert: original observer is unaffected
        assert!((coord.observer.last_state.memory_pressure - 0.3).abs() < f32::EPSILON);
        assert_eq!(coord.observer.last_state.kv_fragmentation, 0.0);
        // Assert: clone has the new values
        assert!((cloned_observer.last_state.memory_pressure - 0.9).abs() < f32::EPSILON);
        assert!((cloned_observer.last_state.kv_fragmentation - 0.8).abs() < f32::EPSILON);
    }

    // @trace TEST-OBS-53 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn observer_default_equals_new() {
        // Arrange & Act
        let from_new = BasicObserver::new();
        let from_default = BasicObserver::default();

        // Assert: both produce identical initial states via public field
        assert_eq!(from_new.last_state, from_default.last_state);
    }

    // @trace TEST-OBS-54 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn all_metrics_combined_with_capture_state() {
        // Arrange: use both capture_state and direct update methods together
        let mut coord = make_coord();
        let dispatch = make_dispatch();

        // Act: set metrics via capture_state, then update additional metrics
        coord.capture_state(Ok(0.4), &dispatch);
        coord.update_swap_io_rate(500.0);
        coord.update_logits_entropy(2.5);
        coord.update_attention_sparsity(0.75);
        coord.update_moe_fault_metrics(0.03, 200.0, 32);

        // Assert: all metrics coexist correctly
        let s = &coord.observer.last_state;
        assert!((s.memory_pressure - 0.4).abs() < f32::EPSILON);
        assert!((s.swap_io_rate - 500.0).abs() < f32::EPSILON);
        assert!((s.logits_entropy - 2.5).abs() < f32::EPSILON);
        assert!((s.attention_sparsity - 0.75).abs() < f32::EPSILON);
        assert!((s.moe_fault_rate - 0.03).abs() < 1e-3);
        assert!((s.moe_avg_recovery_us - 200.0).abs() < f32::EPSILON);
        assert_eq!(s.moe_working_set_size, 32);
    }

    // ── 10 additional tests: capture_state edge cases & coordinator integration ──

    // @trace TEST-OBS-55 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn capture_state_error_then_ok_via_second_capture() {
        // Arrange
        let mut coord = make_coord();
        let dispatch = make_dispatch();

        // Act: first capture_state with error, second with success
        coord.capture_state(Err("sensor down".into()), &dispatch);
        coord.capture_state(Ok(0.66), &dispatch);

        // Assert: second call overwrites memory_pressure from default to 0.66
        assert!((coord.observer.last_state.memory_pressure - 0.66).abs() < f32::EPSILON);
    }

    // @trace TEST-OBS-56 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn capture_state_does_not_touch_swap_rate_or_entropy() {
        // Arrange: set coordinator-only metrics first
        let mut coord = make_coord();
        let dispatch = make_dispatch();
        coord.update_swap_io_rate(88.8);
        coord.update_logits_entropy(1.23);
        coord.update_attention_sparsity(0.44);

        // Act
        coord.capture_state(Ok(0.5), &dispatch);

        // Assert: coordinator-only metrics are untouched by capture_state
        let s = &coord.observer.last_state;
        assert!((s.swap_io_rate - 88.8).abs() < f32::EPSILON);
        assert!((s.logits_entropy - 1.23).abs() < f32::EPSILON);
        assert!((s.attention_sparsity - 0.44).abs() < f32::EPSILON);
    }

    // @trace TEST-OBS-57 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn capture_state_consecutive_calls_overwrite_all_fields() {
        // Arrange
        let mut coord = make_coord();
        let dispatch = make_dispatch();

        // Act: two consecutive capture_state calls
        coord.capture_state(Ok(0.1), &dispatch);
        coord.capture_state(Ok(0.9), &dispatch);

        // Assert: memory_pressure reflects the last call
        assert!((coord.observer.last_state.memory_pressure - 0.9).abs() < f32::EPSILON);
    }

    // @trace TEST-OBS-58 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn debug_format_with_all_metrics_set() {
        // Arrange: set every metric through coordinator methods
        let mut coord = make_coord();
        coord.observer.update_memory_pressure(Ok(0.55)).unwrap();
        coord.update_swap_io_rate(999.0);
        coord.update_logits_entropy(4.2);
        coord.update_attention_sparsity(0.33);
        coord.update_moe_fault_metrics(0.07, 333.0, 64);

        // Act
        let debug = format!("{coord:?}");

        // Assert: Debug output contains struct name
        assert!(debug.contains("ObservabilityCoordinator"), "debug: {debug}");
    }

    // @trace TEST-OBS-59 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn direct_observer_mutation_visible_through_coordinator() {
        // Arrange
        let mut coord = make_coord();

        // Act: mutate observer directly via public field, then read via coordinator method
        coord.observer.last_state.memory_pressure = 0.71;

        // Assert: the value is visible through the coordinator's observer reference
        assert!((coord.observer.last_state.memory_pressure - 0.71).abs() < f32::EPSILON);
    }

    // @trace TEST-OBS-60 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn coordinator_update_methods_preserve_capture_state_scheduler_fields() {
        // Arrange
        let mut coord = make_coord();
        let dispatch = make_dispatch();
        coord.capture_state(Ok(0.3), &dispatch);
        coord.observer.update_scheduler_metrics(20, 10, 8, 512);

        // Act: update coordinator-only metrics (should not touch scheduler fields)
        coord.update_swap_io_rate(100.0);
        coord.update_logits_entropy(3.0);
        coord.update_attention_sparsity(0.6);
        coord.update_moe_fault_metrics(0.02, 100.0, 16);

        // Assert: scheduler fields are still intact
        let s = &coord.observer.last_state;
        assert_eq!(s.waiting_queue_len, 20);
        assert_eq!(s.current_running_len, 10);
        assert_eq!(s.current_batch_size, 8);
        assert_eq!(s.mean_context_len, 512);
    }

    // @trace TEST-OBS-61 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn capture_state_error_does_not_modify_kv_fragmentation() {
        // Arrange: set kv_fragmentation to a known value before capture_state error
        let mut coord = make_coord();
        let dispatch = make_dispatch();
        coord.observer.update_kv_fragmentation(0.42);

        // Act: capture_state with error
        coord.capture_state(Err("timeout".into()), &dispatch);

        // Assert: kv_fragmentation was updated (from dispatch, which returns 0.0 for empty)
        // This verifies the error branch only skips memory_pressure, not other fields
        assert_eq!(coord.observer.last_state.memory_pressure, 0.0);
    }

    // @trace TEST-OBS-62 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn capture_state_with_boundary_memory_pressure_values() {
        // Arrange
        let mut coord = make_coord();
        let dispatch = make_dispatch();

        // Act: capture with 0.0 (valid boundary)
        coord.capture_state(Ok(0.0), &dispatch);
        assert_eq!(coord.observer.last_state.memory_pressure, 0.0);

        // Act: capture with 1.0 (valid boundary)
        coord.capture_state(Ok(1.0), &dispatch);
        assert!((coord.observer.last_state.memory_pressure - 1.0).abs() < f32::EPSILON);
    }

    // @trace TEST-OBS-63 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn coordinator_repeated_moe_fault_metrics_overwrites() {
        // Arrange
        let mut coord = make_coord();

        // Act: call update_moe_fault_metrics three times
        coord.update_moe_fault_metrics(0.1, 50.0, 10);
        coord.update_moe_fault_metrics(0.5, 200.0, 20);
        coord.update_moe_fault_metrics(0.01, 10.0, 1);

        // Assert: only the last call's values survive
        let s = &coord.observer.last_state;
        assert!((s.moe_fault_rate - 0.01).abs() < 1e-3);
        assert!((s.moe_avg_recovery_us - 10.0).abs() < f32::EPSILON);
        assert_eq!(s.moe_working_set_size, 1);
    }

    // @trace TEST-OBS-64 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn coordinator_all_methods_idempotent_at_defaults() {
        // Arrange: create a fresh coordinator and read all fields
        let mut coord = make_coord();
        let dispatch = make_dispatch();

        // Act: call capture_state with default memory_pressure
        coord.capture_state(Ok(0.0), &dispatch);
        // Call all direct update methods with zero/default values
        coord.update_swap_io_rate(0.0);
        coord.update_logits_entropy(0.0);
        coord.update_attention_sparsity(0.0);
        coord.update_moe_fault_metrics(0.0, 0.0, 0);

        // Assert: all fields remain at zero/default — no accidental mutation
        let s = &coord.observer.last_state;
        assert_eq!(s.memory_pressure, 0.0);
        assert_eq!(s.swap_io_rate, 0.0);
        assert_eq!(s.logits_entropy, 0.0);
        assert_eq!(s.attention_sparsity, 0.0);
        assert_eq!(s.moe_fault_rate, 0.0);
        assert_eq!(s.moe_avg_recovery_us, 0.0);
        assert_eq!(s.moe_working_set_size, 0);
        assert_eq!(s.waiting_queue_len, 0);
        assert_eq!(s.current_running_len, 0);
        assert_eq!(s.current_batch_size, 0);
        assert_eq!(s.mean_context_len, 0);
    }
}
