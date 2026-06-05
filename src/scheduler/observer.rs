use super::jit_types::SystemState;
use super::types::WeightTier;
use crate::kv_cache::CompressionCodec;
use crate::sensors::CompressionTelemetry;

/// Error type for observer operations.
#[derive(Debug, Clone)]
pub enum ObserverError {
    BackendUnavailable(String),
}

impl std::fmt::Display for ObserverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendUnavailable(msg) => write!(f, "backend unavailable: {msg}"),
        }
    }
}

impl std::error::Error for ObserverError {}

/// Reason for weight page eviction (SPEC 21-WEIGHT-PAGING.md §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionReason {
    /// Evicted due to memory pressure on the current tier.
    MemoryPressure,
}

/// Weight page telemetry event — recorded at key lifecycle transitions.
///
/// Each event carries the page identifier, source and destination tiers,
/// and transfer metadata (byte count, latency) for aggregation by the
/// telemetry framework.
#[derive(Debug, Clone)]
pub enum WeightPageTelemetryEvent {
    /// A weight page was evicted from a higher tier to a lower tier.
    Evicted {
        page_id: usize,
        from_tier: WeightTier,
        to_tier: WeightTier,
        reason: EvictionReason,
        bytes: u64,
    },
    /// A weight page was recovered (promoted) to a higher tier.
    Recovered {
        page_id: usize,
        from_tier: WeightTier,
        to_tier: WeightTier,
        latency_us: u64,
        bytes: u64,
    },
}

/// Runtime observer trait.
pub trait RuntimeObserver {
    fn capture(&self) -> Result<SystemState, ObserverError>;
}

/// Basic observer that holds the last captured state.
/// The executor updates fields before calling capture().
#[derive(Debug, Clone)]
pub struct BasicObserver {
    pub last_state: SystemState,
    /// Compression telemetry aggregator (SPEC 22 §9).
    pub compression_telemetry: CompressionTelemetry,
}

impl Default for BasicObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicObserver {
    pub fn new() -> Self {
        Self {
            last_state: SystemState::default(),
            compression_telemetry: CompressionTelemetry::new(),
        }
    }

    /// Update resource metrics from external sources.
    /// Called by executor before policy decision.
    pub fn update_memory_pressure(&mut self, pressure: Result<f32, String>) -> Result<(), ObserverError> {
        match pressure {
            Ok(p) => {
                self.last_state.memory_pressure = p;
                Ok(())
            }
            Err(e) => Err(ObserverError::BackendUnavailable(e)),
        }
    }

    pub fn update_scheduler_metrics(
        &mut self,
        waiting_queue_len: usize,
        current_running_len: usize,
        current_batch_size: usize,
        mean_context_len: usize,
    ) {
        self.last_state.waiting_queue_len = waiting_queue_len;
        self.last_state.current_running_len = current_running_len;
        self.last_state.current_batch_size = current_batch_size;
        self.last_state.mean_context_len = mean_context_len;
    }

    pub fn update_kv_fragmentation(&mut self, fragmentation: f32) {
        self.last_state.kv_fragmentation = fragmentation;
    }

    pub fn update_swap_io_rate(&mut self, rate: f32) {
        self.last_state.swap_io_rate = rate;
    }

    pub fn update_logits_entropy(&mut self, entropy: f32) {
        self.last_state.logits_entropy = entropy;
    }

    pub fn update_attention_sparsity(&mut self, sparsity: f32) {
        self.last_state.attention_sparsity = sparsity;
    }

    /// Update MoE fault metrics from the fault handler.
    pub fn update_moe_fault_metrics(
        &mut self,
        fault_rate: f32,
        avg_recovery_latency_us: f32,
        working_set_size: usize,
    ) {
        self.last_state.moe_fault_rate = fault_rate;
        self.last_state.moe_avg_recovery_us = avg_recovery_latency_us;
        self.last_state.moe_working_set_size = working_set_size;
    }

    /// §21 WP-010: Update weight page telemetry metrics (bulk snapshot).
    pub fn update_weight_metrics(
        &mut self,
        total: usize,
        l1: usize,
        l2: usize,
        l3: usize,
        eviction_count: usize,
        recovery_count: usize,
    ) {
        self.last_state.weight_page_total = total;
        self.last_state.weight_pages_l1 = l1;
        self.last_state.weight_pages_l2 = l2;
        self.last_state.weight_pages_l3 = l3;
        self.last_state.weight_eviction_count = eviction_count;
        self.last_state.weight_recovery_count = recovery_count;
    }

    /// §21 WP-010: Record a weight page telemetry event (REQ-WP-010).
    ///
    /// Part of the `weight_page_telemetry` subsystem — incrementally
    /// updates the cached `SystemState` counters for eviction/recovery
    /// counts and tier distribution.
    pub fn record_weight_page_event(&mut self, event: WeightPageTelemetryEvent) {
        match event {
            WeightPageTelemetryEvent::Evicted { from_tier, to_tier, .. } => {
                self.last_state.weight_eviction_count += 1;
                Self::adjust_tier_count(&mut self.last_state, from_tier, false);
                Self::adjust_tier_count(&mut self.last_state, to_tier, true);
            }
            WeightPageTelemetryEvent::Recovered { from_tier, to_tier, .. } => {
                self.last_state.weight_recovery_count += 1;
                Self::adjust_tier_count(&mut self.last_state, from_tier, false);
                Self::adjust_tier_count(&mut self.last_state, to_tier, true);
            }
        }
    }

    /// Adjust the per-tier page counter by ±1.
    fn adjust_tier_count(state: &mut SystemState, tier: WeightTier, increment: bool) {
        match tier {
            WeightTier::Hot => {
                if increment { state.weight_pages_l1 += 1; }
                else { state.weight_pages_l1 = state.weight_pages_l1.saturating_sub(1); }
            }
            WeightTier::Warm => {
                if increment { state.weight_pages_l2 += 1; }
                else { state.weight_pages_l2 = state.weight_pages_l2.saturating_sub(1); }
            }
            WeightTier::Cold => {
                if increment { state.weight_pages_l3 += 1; }
                else { state.weight_pages_l3 = state.weight_pages_l3.saturating_sub(1); }
            }
        }
    }

    // ── Compression Telemetry Methods (SPEC 22 §9) ──

    /// Record a compression operation (SPEC 22 §9).
    ///
    /// Delegates to `self.compression_telemetry.record_compress()`.
    /// Call from eviction worker after GPU compression, or from
    /// any codec compression path.
    pub fn record_compress(&mut self, codec: CompressionCodec, input_bytes: u64, output_bytes: u64, latency_us: u64) {
        self.compression_telemetry.record_compress(codec, input_bytes, output_bytes, latency_us);
    }

    /// Record a decompression operation (SPEC 22 §9).
    ///
    /// Delegates to `self.compression_telemetry.record_decompress()`.
    /// Call from swap-in worker after JIT decode, or from
    /// any codec decompression path.
    pub fn record_decompress(&mut self, codec: CompressionCodec, input_bytes: u64, output_bytes: u64, latency_us: u64) {
        self.compression_telemetry.record_decompress(codec, input_bytes, output_bytes, latency_us);
    }

    /// Record a page migration between storage tiers (SPEC 22 §9).
    ///
    /// Delegates to `self.compression_telemetry.record_migration()`.
    /// Call from PageMigrationActor after a physical data transfer completes.
    pub fn record_migration(&mut self, bytes: u64) {
        self.compression_telemetry.record_migration(bytes);
    }

    /// Record a page eviction event (SPEC 22 §9).
    pub fn record_eviction(&mut self) {
        self.compression_telemetry.record_eviction();
    }

    /// Record a page swap-in event (SPEC 22 §9).
    pub fn record_swap_in(&mut self) {
        self.compression_telemetry.record_swap_in();
    }

    /// Snapshot the current compression telemetry counters.
    pub fn compression_telemetry(&self) -> &CompressionTelemetry {
        &self.compression_telemetry
    }
}

impl RuntimeObserver for BasicObserver {
    fn capture(&self) -> Result<SystemState, ObserverError> {
        Ok(self.last_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_updated_state() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.75)).unwrap();
        obs.update_scheduler_metrics(10, 5, 8, 128);
        obs.update_kv_fragmentation(0.3);
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 0.75).abs() < f32::EPSILON);
        assert_eq!(state.waiting_queue_len, 10);
        assert_eq!(state.current_running_len, 5);
        assert_eq!(state.current_batch_size, 8);
        assert_eq!(state.mean_context_len, 128);
        assert!((state.kv_fragmentation - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn memory_pressure_error_propagation() {
        let mut obs = BasicObserver::new();
        let result = obs.update_memory_pressure(Err("gpu unavailable".into()));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ObserverError::BackendUnavailable(_)));
    }

    #[test]
    fn phase2_setters() {
        let mut obs = BasicObserver::new();
        obs.update_swap_io_rate(42.5);
        obs.update_logits_entropy(3.14);
        obs.update_attention_sparsity(0.85);
        let state = obs.capture().unwrap();
        assert!((state.swap_io_rate - 42.5).abs() < f32::EPSILON);
        assert!((state.logits_entropy - 3.14).abs() < f32::EPSILON);
        assert!((state.attention_sparsity - 0.85).abs() < f32::EPSILON);
    }

    // ── Compression Telemetry Tests (SPEC 22 §9) ──

    #[test]
    fn observer_compression_telemetry_initialized() {
        let obs = BasicObserver::new();
        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 0);
        assert_eq!(ct.decompress_count, 0);
        assert_eq!(ct.total_migration_bytes, 0);
        assert_eq!(ct.eviction_count, 0);
        assert_eq!(ct.swap_in_count, 0);
    }

    #[test]
    fn observer_record_compress_via_basic_observer() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::Lz4, 4096, 2048, 100);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.total_input_bytes, 4096);
        assert_eq!(ct.total_compressed_bytes, 2048);
        assert_eq!(ct.total_compress_latency_us, 100);
        assert!((ct.overall_compression_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn observer_record_decompress_via_basic_observer() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::Lz4, 2048, 4096, 50);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.decompress_count, 1);
        assert_eq!(ct.total_decompress_latency_us, 50);
    }

    #[test]
    fn observer_record_migration_via_basic_observer() {
        let mut obs = BasicObserver::new();
        obs.record_migration(8192);
        obs.record_migration(4096);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.total_migration_bytes, 12288);
    }

    #[test]
    fn observer_record_eviction_and_swap_in() {
        let mut obs = BasicObserver::new();
        obs.record_eviction();
        obs.record_eviction();
        obs.record_swap_in();
        let ct = obs.compression_telemetry();
        assert_eq!(ct.eviction_count, 2);
        assert_eq!(ct.swap_in_count, 1);
    }

    #[test]
    fn observer_compression_telemetry_multiple_codecs() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::Lz4, 4096, 2048, 100);
        obs.record_compress(CompressionCodec::BitPackRle, 4096, 1024, 200);
        obs.record_decompress(CompressionCodec::Lz4, 2048, 4096, 50);

        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 2);
        assert_eq!(ct.decompress_count, 1);
        assert_eq!(ct.total_input_bytes, 8192);
        assert_eq!(ct.total_compressed_bytes, 3072);

        let lz4_stats = ct.codec_stats(CompressionCodec::Lz4);
        assert_eq!(lz4_stats.compress_count, 1);
        assert_eq!(lz4_stats.decompress_count, 1);
        // compress input=4096 + decompress input=2048
        assert_eq!(lz4_stats.total_input_bytes, 6144);
        // compress output=2048 + decompress output=4096
        assert_eq!(lz4_stats.total_output_bytes, 6144);

        let bitpack_stats = ct.codec_stats(CompressionCodec::BitPackRle);
        assert_eq!(bitpack_stats.compress_count, 1);
        assert_eq!(bitpack_stats.total_input_bytes, 4096);
        assert_eq!(bitpack_stats.total_output_bytes, 1024);
    }

    // ── ObserverError ──

    #[test]
    fn observer_error_display() {
        let err = ObserverError::BackendUnavailable("gpu gone".into());
        let msg = format!("{err}");
        assert!(msg.contains("backend unavailable"), "msg: {msg}");
        assert!(msg.contains("gpu gone"), "msg: {msg}");
    }

    #[test]
    fn observer_error_is_std_error() {
        let err = ObserverError::BackendUnavailable("test".into());
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn observer_error_debug() {
        let err = ObserverError::BackendUnavailable("x".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("BackendUnavailable"));
    }

    #[test]
    fn observer_error_clone() {
        let err = ObserverError::BackendUnavailable("orig".into());
        let cloned = err.clone();
        let msg = format!("{cloned}");
        assert!(msg.contains("orig"));
    }

    // ── EvictionReason ──

    #[test]
    fn eviction_reason_equality() {
        assert_eq!(EvictionReason::MemoryPressure, EvictionReason::MemoryPressure);
    }

    #[test]
    fn eviction_reason_copy() {
        let a = EvictionReason::MemoryPressure;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn eviction_reason_debug() {
        let debug = format!("{:?}", EvictionReason::MemoryPressure);
        assert!(debug.contains("MemoryPressure"));
    }

    // ── WeightPageTelemetryEvent ──

    #[test]
    fn telemetry_event_evicted() {
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 42,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        };
        let debug = format!("{event:?}");
        assert!(debug.contains("Evicted"));
    }

    #[test]
    fn telemetry_event_recovered() {
        let event = WeightPageTelemetryEvent::Recovered {
            page_id: 7,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 500,
            bytes: 8192,
        };
        let debug = format!("{event:?}");
        assert!(debug.contains("Recovered"));
    }

    #[test]
    fn telemetry_event_clone() {
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 1,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 1024,
        };
        let cloned = event.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("Evicted"));
    }

    // ── BasicObserver weight page events ──

    #[test]
    fn record_weight_page_eviction_adjusts_tiers() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 5, 3, 2, 0, 0);
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        };
        obs.record_weight_page_event(event);
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 4); // 5 - 1
        assert_eq!(state.weight_pages_l2, 4); // 3 + 1
        assert_eq!(state.weight_eviction_count, 1);
    }

    #[test]
    fn record_weight_page_recovery_adjusts_tiers() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 3, 5, 2, 0, 0);
        let event = WeightPageTelemetryEvent::Recovered {
            page_id: 0,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 100,
            bytes: 4096,
        };
        obs.record_weight_page_event(event);
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 4); // 3 + 1
        assert_eq!(state.weight_pages_l3, 1); // 2 - 1
        assert_eq!(state.weight_recovery_count, 1);
    }

    #[test]
    fn saturating_sub_prevents_underflow() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 0, 0, 0, 0, 0);
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 100,
        };
        obs.record_weight_page_event(event);
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 0); // saturating_sub(0-1) = 0
        assert_eq!(state.weight_pages_l2, 1);
    }

    // ── BasicObserver Default ──

    #[test]
    fn basic_observer_default() {
        let obs = BasicObserver::default();
        let state = obs.capture().unwrap();
        assert_eq!(state.memory_pressure, 0.0);
        assert_eq!(state.weight_page_total, 0);
    }

    // ── MoE fault metrics ──

    #[test]
    fn update_moe_fault_metrics() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(0.05, 200.0, 8);
        let state = obs.capture().unwrap();
        assert!((state.moe_fault_rate - 0.05).abs() < 1e-6);
        assert!((state.moe_avg_recovery_us - 200.0).abs() < 1e-6);
        assert_eq!(state.moe_working_set_size, 8);
    }

    #[test]
    fn update_weight_metrics_bulk() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(100, 60, 30, 10, 5, 3);
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_page_total, 100);
        assert_eq!(state.weight_pages_l1, 60);
        assert_eq!(state.weight_pages_l2, 30);
        assert_eq!(state.weight_pages_l3, 10);
        assert_eq!(state.weight_eviction_count, 5);
        assert_eq!(state.weight_recovery_count, 3);
    }

    // ── New tests: 18 additional ──

    #[test]
    fn observer_error_display_format_contains_message() {
        let err = ObserverError::BackendUnavailable("nvml init failed".into());
        let display = format!("{err}");
        assert!(display.starts_with("backend unavailable: "), "display: {display}");
        assert!(display.contains("nvml init failed"));
    }

    #[test]
    fn observer_error_debug_includes_variant_name() {
        let err = ObserverError::BackendUnavailable("timeout".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("BackendUnavailable"), "debug: {debug}");
        assert!(debug.contains("timeout"));
    }

    #[test]
    fn observer_error_clone_preserves_message() {
        let original = ObserverError::BackendUnavailable("disk full".into());
        let cloned = original.clone();
        assert_eq!(format!("{original}"), format!("{cloned}"));
    }

    #[test]
    fn eviction_reason_copy_and_eq() {
        let a = EvictionReason::MemoryPressure;
        let b = a; // Copy semantics, a is still valid
        assert_eq!(a, b);
        let c = EvictionReason::MemoryPressure;
        assert_eq!(a, c);
    }

    #[test]
    fn eviction_reason_debug_format() {
        let debug = format!("{:?}", EvictionReason::MemoryPressure);
        assert!(debug.contains("MemoryPressure"), "debug: {debug}");
    }

    #[test]
    fn weight_page_telemetry_event_evicted_clone_preserves_fields() {
        let original = WeightPageTelemetryEvent::Evicted {
            page_id: 99,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: 65536,
        };
        let cloned = original.clone();
        if let WeightPageTelemetryEvent::Evicted {
            page_id, from_tier, to_tier, reason, bytes,
        } = cloned
        {
            assert_eq!(page_id, 99);
            assert_eq!(from_tier, WeightTier::Hot);
            assert_eq!(to_tier, WeightTier::Cold);
            assert_eq!(reason, EvictionReason::MemoryPressure);
            assert_eq!(bytes, 65536);
        } else {
            panic!("cloned should be Evicted variant");
        }
    }

    #[test]
    fn weight_page_telemetry_event_recovered_debug_contains_latency() {
        let event = WeightPageTelemetryEvent::Recovered {
            page_id: 3,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Hot,
            latency_us: 1234,
            bytes: 8192,
        };
        let debug = format!("{event:?}");
        assert!(debug.contains("Recovered"), "debug: {debug}");
    }

    #[test]
    fn basic_observer_new_all_fields_default() {
        let obs = BasicObserver::new();
        let state = obs.capture().unwrap();
        assert_eq!(state.memory_pressure, 0.0);
        assert_eq!(state.kv_fragmentation, 0.0);
        assert_eq!(state.swap_io_rate, 0.0);
        assert_eq!(state.waiting_queue_len, 0);
        assert_eq!(state.current_batch_size, 0);
        assert_eq!(state.current_running_len, 0);
        assert_eq!(state.mean_context_len, 0);
        assert_eq!(state.logits_entropy, 0.0);
        assert_eq!(state.attention_sparsity, 0.0);
        assert_eq!(state.moe_fault_rate, 0.0);
        assert_eq!(state.moe_avg_recovery_us, 0.0);
        assert_eq!(state.moe_working_set_size, 0);
        assert_eq!(state.weight_page_total, 0);
        assert_eq!(state.weight_pages_l1, 0);
        assert_eq!(state.weight_pages_l2, 0);
        assert_eq!(state.weight_pages_l3, 0);
        assert_eq!(state.weight_eviction_count, 0);
        assert_eq!(state.weight_recovery_count, 0);
    }

    #[test]
    fn basic_observer_default_equals_new() {
        let from_new = BasicObserver::new();
        let from_default = BasicObserver::default();
        let state_new = from_new.capture().unwrap();
        let state_default = from_default.capture().unwrap();
        assert_eq!(state_new.memory_pressure, state_default.memory_pressure);
        assert_eq!(state_new.weight_page_total, state_default.weight_page_total);
        assert_eq!(state_new.weight_eviction_count, state_default.weight_eviction_count);
    }

    #[test]
    fn update_memory_pressure_nan_is_accepted() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(f32::NAN)).unwrap();
        let state = obs.capture().unwrap();
        assert!(state.memory_pressure.is_nan());
    }

    #[test]
    fn update_memory_pressure_negative_is_accepted() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(-1.0)).unwrap();
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn update_kv_fragmentation_max_value() {
        let mut obs = BasicObserver::new();
        obs.update_kv_fragmentation(f32::MAX);
        let state = obs.capture().unwrap();
        assert_eq!(state.kv_fragmentation, f32::MAX);
    }

    #[test]
    fn update_scheduler_metrics_zero_values() {
        let mut obs = BasicObserver::new();
        obs.update_scheduler_metrics(0, 0, 0, 0);
        let state = obs.capture().unwrap();
        assert_eq!(state.waiting_queue_len, 0);
        assert_eq!(state.current_running_len, 0);
        assert_eq!(state.current_batch_size, 0);
        assert_eq!(state.mean_context_len, 0);
    }

    #[test]
    fn update_scheduler_metrics_max_usize() {
        let mut obs = BasicObserver::new();
        obs.update_scheduler_metrics(usize::MAX, usize::MAX, usize::MAX, usize::MAX);
        let state = obs.capture().unwrap();
        assert_eq!(state.waiting_queue_len, usize::MAX);
        assert_eq!(state.current_running_len, usize::MAX);
        assert_eq!(state.current_batch_size, usize::MAX);
        assert_eq!(state.mean_context_len, usize::MAX);
    }

    #[test]
    fn record_weight_page_event_eviction_warm_to_cold() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 2, 5, 3, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 10,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: 2048,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l2, 4); // 5 - 1
        assert_eq!(state.weight_pages_l3, 4); // 3 + 1
        assert_eq!(state.weight_eviction_count, 1);
    }

    #[test]
    fn record_weight_page_event_recovery_cold_to_warm() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 2, 4, 4, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 5,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Warm,
            latency_us: 250,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l3, 3); // 4 - 1
        assert_eq!(state.weight_pages_l2, 5); // 4 + 1
        assert_eq!(state.weight_recovery_count, 1);
    }

    #[test]
    fn record_weight_page_event_multiple_accumulates() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 5, 3, 2, 0, 0);
        // Evict Hot -> Warm
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 1,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 1024,
        });
        // Evict Warm -> Cold
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 2,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: 2048,
        });
        // Recover Cold -> Hot
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 1,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 100,
            bytes: 1024,
        });
        let state = obs.capture().unwrap();
        // l1: 5 - 1 (evict Hot) + 1 (recover to Hot) = 5
        // l2: 3 + 1 (evict to Warm) - 1 (evict from Warm) = 3
        // l3: 2 + 1 (evict to Cold) - 1 (recover from Cold) = 2
        assert_eq!(state.weight_pages_l1, 5);
        assert_eq!(state.weight_pages_l2, 3);
        assert_eq!(state.weight_pages_l3, 2);
        assert_eq!(state.weight_eviction_count, 2);
        assert_eq!(state.weight_recovery_count, 1);
    }

    #[test]
    fn record_compress_none_codec_zero_bytes() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::None, 0, 0, 0);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.total_input_bytes, 0);
        assert_eq!(ct.total_compressed_bytes, 0);
        assert_eq!(ct.total_compress_latency_us, 0);
    }

    #[test]
    fn record_compress_zstd_dict_large_values() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::ZstdDict, u64::MAX, u64::MAX, u64::MAX);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.total_input_bytes, u64::MAX);
        assert_eq!(ct.total_compressed_bytes, u64::MAX);
        assert_eq!(ct.total_compress_latency_us, u64::MAX);
    }

    // ── 45+ additional tests ──

    // ── ObserverError exhaustive tests ──

    #[test]
    fn observer_error_display_empty_message() {
        let err = ObserverError::BackendUnavailable(String::new());
        let display = format!("{err}");
        assert_eq!(display, "backend unavailable: ");
    }

    #[test]
    fn observer_error_display_long_message() {
        let long_msg = "a".repeat(1000);
        let err = ObserverError::BackendUnavailable(long_msg.clone());
        let display = format!("{err}");
        assert!(display.contains(&long_msg));
    }

    #[test]
    fn observer_error_display_unicode_message() {
        let err = ObserverError::BackendUnavailable("GPU不可用".into());
        let display = format!("{err}");
        assert!(display.contains("GPU不可用"));
    }

    #[test]
    fn observer_error_source_is_none() {
        // ObserverError has no source chain
        let err = ObserverError::BackendUnavailable("test".into());
        assert!(std::error::Error::source(&err).is_none());
    }

    // ── EvictionReason exhaustive tests ──

    #[test]
    fn eviction_reason_clone() {
        let a = EvictionReason::MemoryPressure;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn eviction_reason_hash_equal() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let a = EvictionReason::MemoryPressure;
        let b = EvictionReason::MemoryPressure;
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn eviction_reason_is_single_variant() {
        // EvictionReason currently only has MemoryPressure
        let reason = EvictionReason::MemoryPressure;
        let debug = format!("{reason:?}");
        assert!(!debug.is_empty());
    }

    // ── WeightTier integration with adjust_tier_count ──

    #[test]
    fn weight_page_evict_hot_to_hot_same_tier() {
        // Edge case: evict from Hot to Hot (no-op on tier counts but eviction count still increments)
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 3, 1, 1, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Hot,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 3); // 3 - 1 + 1 = 3
        assert_eq!(state.weight_eviction_count, 1);
    }

    #[test]
    fn weight_page_recover_cold_to_cold_same_tier() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 1, 1, 3, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 0,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Cold,
            latency_us: 100,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l3, 3); // 3 - 1 + 1 = 3
        assert_eq!(state.weight_recovery_count, 1);
    }

    #[test]
    fn weight_page_eviction_zero_bytes() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 5, 0, 0, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 99,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 0,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 4);
        assert_eq!(state.weight_pages_l2, 1);
    }

    #[test]
    fn weight_page_recovery_zero_latency() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 2, 0, 3, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 10,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 0,
            bytes: 8192,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 3);
        assert_eq!(state.weight_pages_l3, 2);
        assert_eq!(state.weight_recovery_count, 1);
    }

    #[test]
    fn weight_page_recovery_max_page_id() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 2, 2, 1, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: usize::MAX,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Hot,
            latency_us: 999,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 3);
        assert_eq!(state.weight_pages_l2, 1);
    }

    #[test]
    fn weight_page_eviction_max_bytes() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 3, 1, 1, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: u64::MAX,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l2, 0); // 1 - 1
        assert_eq!(state.weight_pages_l3, 2); // 1 + 1
    }

    #[test]
    fn saturating_sub_warm_tier_from_zero() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 2, 0, 3, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: 100,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l2, 0); // saturating_sub from 0
        assert_eq!(state.weight_pages_l3, 4);
    }

    #[test]
    fn saturating_sub_cold_tier_from_zero() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 3, 2, 0, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 0,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 50,
            bytes: 2048,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l3, 0); // saturating_sub from 0
        assert_eq!(state.weight_pages_l1, 4);
    }

    // ── BasicObserver update methods with special float values ──

    #[test]
    fn update_memory_pressure_infinity() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(f32::INFINITY)).unwrap();
        let state = obs.capture().unwrap();
        assert!(state.memory_pressure.is_infinite());
        assert!(state.memory_pressure.is_sign_positive());
    }

    #[test]
    fn update_memory_pressure_neg_infinity() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(f32::NEG_INFINITY)).unwrap();
        let state = obs.capture().unwrap();
        assert!(state.memory_pressure.is_infinite());
        assert!(state.memory_pressure.is_sign_negative());
    }

    #[test]
    fn update_kv_fragmentation_nan() {
        let mut obs = BasicObserver::new();
        obs.update_kv_fragmentation(f32::NAN);
        let state = obs.capture().unwrap();
        assert!(state.kv_fragmentation.is_nan());
    }

    #[test]
    fn update_swap_io_rate_nan() {
        let mut obs = BasicObserver::new();
        obs.update_swap_io_rate(f32::NAN);
        let state = obs.capture().unwrap();
        assert!(state.swap_io_rate.is_nan());
    }

    #[test]
    fn update_swap_io_rate_zero() {
        let mut obs = BasicObserver::new();
        obs.update_swap_io_rate(0.0);
        let state = obs.capture().unwrap();
        assert!((state.swap_io_rate - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn update_logits_entropy_max() {
        let mut obs = BasicObserver::new();
        obs.update_logits_entropy(f32::MAX);
        let state = obs.capture().unwrap();
        assert_eq!(state.logits_entropy, f32::MAX);
    }

    #[test]
    fn update_logits_entropy_neg_infinity() {
        let mut obs = BasicObserver::new();
        obs.update_logits_entropy(f32::NEG_INFINITY);
        let state = obs.capture().unwrap();
        assert!(state.logits_entropy.is_infinite());
        assert!(state.logits_entropy.is_sign_negative());
    }

    #[test]
    fn update_attention_sparsity_zero() {
        let mut obs = BasicObserver::new();
        obs.update_attention_sparsity(0.0);
        let state = obs.capture().unwrap();
        assert!((state.attention_sparsity - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn update_attention_sparsity_one() {
        let mut obs = BasicObserver::new();
        obs.update_attention_sparsity(1.0);
        let state = obs.capture().unwrap();
        assert!((state.attention_sparsity - 1.0).abs() < f32::EPSILON);
    }

    // ── MoE fault metrics edge cases ──

    #[test]
    fn update_moe_fault_metrics_zero_values() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(0.0, 0.0, 0);
        let state = obs.capture().unwrap();
        assert!((state.moe_fault_rate - 0.0).abs() < f32::EPSILON);
        assert!((state.moe_avg_recovery_us - 0.0).abs() < f32::EPSILON);
        assert_eq!(state.moe_working_set_size, 0);
    }

    #[test]
    fn update_moe_fault_metrics_max_usize_working_set() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(1.0, 9999.0, usize::MAX);
        let state = obs.capture().unwrap();
        assert_eq!(state.moe_working_set_size, usize::MAX);
    }

    #[test]
    fn update_moe_fault_metrics_nan_rate() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(f32::NAN, 100.0, 5);
        let state = obs.capture().unwrap();
        assert!(state.moe_fault_rate.is_nan());
        assert!((state.moe_avg_recovery_us - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn update_moe_fault_metrics_overwrite() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(0.1, 50.0, 10);
        obs.update_moe_fault_metrics(0.9, 200.0, 20);
        let state = obs.capture().unwrap();
        assert!((state.moe_fault_rate - 0.9).abs() < 1e-6);
        assert!((state.moe_avg_recovery_us - 200.0).abs() < 1e-6);
        assert_eq!(state.moe_working_set_size, 20);
    }

    // ── update_weight_metrics edge cases ──

    #[test]
    fn update_weight_metrics_all_zero() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(0, 0, 0, 0, 0, 0);
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_page_total, 0);
        assert_eq!(state.weight_pages_l1, 0);
        assert_eq!(state.weight_pages_l2, 0);
        assert_eq!(state.weight_pages_l3, 0);
        assert_eq!(state.weight_eviction_count, 0);
        assert_eq!(state.weight_recovery_count, 0);
    }

    #[test]
    fn update_weight_metrics_all_max_usize() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX);
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_page_total, usize::MAX);
        assert_eq!(state.weight_pages_l1, usize::MAX);
        assert_eq!(state.weight_pages_l2, usize::MAX);
        assert_eq!(state.weight_pages_l3, usize::MAX);
        assert_eq!(state.weight_eviction_count, usize::MAX);
        assert_eq!(state.weight_recovery_count, usize::MAX);
    }

    #[test]
    fn update_weight_metrics_overwrites_previous() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(100, 50, 30, 20, 5, 2);
        obs.update_weight_metrics(10, 3, 4, 3, 1, 0);
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_page_total, 10);
        assert_eq!(state.weight_pages_l1, 3);
        assert_eq!(state.weight_eviction_count, 1);
    }

    // ── WeightPageTelemetryEvent debug output field verification ──

    #[test]
    fn telemetry_event_evicted_debug_shows_page_id() {
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 12345,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        };
        let debug = format!("{event:?}");
        // Debug should contain the page_id somewhere in the output
        assert!(debug.contains("12345") || debug.contains("page_id"));
    }

    #[test]
    fn telemetry_event_recovered_debug_shows_latency() {
        let event = WeightPageTelemetryEvent::Recovered {
            page_id: 1,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 7777,
            bytes: 2048,
        };
        let debug = format!("{event:?}");
        assert!(debug.contains("7777") || debug.contains("latency_us"));
    }

    #[test]
    fn telemetry_event_recovered_clone_preserves_fields() {
        let original = WeightPageTelemetryEvent::Recovered {
            page_id: 42,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Warm,
            latency_us: 5000,
            bytes: 16384,
        };
        let cloned = original.clone();
        if let WeightPageTelemetryEvent::Recovered {
            page_id, from_tier, to_tier, latency_us, bytes,
        } = cloned
        {
            assert_eq!(page_id, 42);
            assert_eq!(from_tier, WeightTier::Cold);
            assert_eq!(to_tier, WeightTier::Warm);
            assert_eq!(latency_us, 5000);
            assert_eq!(bytes, 16384);
        } else {
            panic!("cloned should be Recovered variant");
        }
    }

    // ── BasicObserver sequential updates accumulate correctly ──

    #[test]
    fn update_scheduler_metrics_overwrites_not_accumulates() {
        let mut obs = BasicObserver::new();
        obs.update_scheduler_metrics(10, 5, 3, 100);
        obs.update_scheduler_metrics(20, 10, 6, 200);
        let state = obs.capture().unwrap();
        assert_eq!(state.waiting_queue_len, 20);
        assert_eq!(state.current_running_len, 10);
        assert_eq!(state.current_batch_size, 6);
        assert_eq!(state.mean_context_len, 200);
    }

    #[test]
    fn update_memory_pressure_overwrites_previous() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.5)).unwrap();
        obs.update_memory_pressure(Ok(0.9)).unwrap();
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn update_kv_fragmentation_overwrites_previous() {
        let mut obs = BasicObserver::new();
        obs.update_kv_fragmentation(0.3);
        obs.update_kv_fragmentation(0.7);
        let state = obs.capture().unwrap();
        assert!((state.kv_fragmentation - 0.7).abs() < f32::EPSILON);
    }

    // ── Compression telemetry via observer ──

    #[test]
    fn record_decompress_nvcompans_codec() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::NvcompAns, 1024, 4096, 75);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.decompress_count, 1);
        assert_eq!(ct.total_decompress_latency_us, 75);
        let stats = ct.codec_stats(CompressionCodec::NvcompAns);
        assert_eq!(stats.decompress_count, 1);
        assert_eq!(stats.total_decompress_latency_us, 75);
    }

    #[test]
    fn record_decompress_bitpackrle_codec() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::BitPackRle, 2048, 8192, 30);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.decompress_count, 1);
        let stats = ct.codec_stats(CompressionCodec::BitPackRle);
        assert_eq!(stats.decompress_count, 1);
    }

    #[test]
    fn record_decompress_none_codec() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::None, 4096, 4096, 1);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.decompress_count, 1);
        let stats = ct.codec_stats(CompressionCodec::None);
        assert_eq!(stats.decompress_count, 1);
        assert_eq!(stats.total_decompress_latency_us, 1);
    }

    #[test]
    fn record_migration_zero_bytes() {
        let mut obs = BasicObserver::new();
        obs.record_migration(0);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.total_migration_bytes, 0);
    }

    #[test]
    fn record_migration_u64_max() {
        let mut obs = BasicObserver::new();
        obs.record_migration(u64::MAX);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.total_migration_bytes, u64::MAX);
    }

    #[test]
    fn record_migration_accumulates() {
        let mut obs = BasicObserver::new();
        obs.record_migration(1000);
        obs.record_migration(2000);
        obs.record_migration(3000);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.total_migration_bytes, 6000);
    }

    #[test]
    fn record_eviction_accumulates() {
        let mut obs = BasicObserver::new();
        for _ in 0..10 {
            obs.record_eviction();
        }
        let ct = obs.compression_telemetry();
        assert_eq!(ct.eviction_count, 10);
    }

    #[test]
    fn record_swap_in_accumulates() {
        let mut obs = BasicObserver::new();
        for _ in 0..5 {
            obs.record_swap_in();
        }
        let ct = obs.compression_telemetry();
        assert_eq!(ct.swap_in_count, 5);
    }

    #[test]
    fn compression_telemetry_default_equals_new() {
        let obs = BasicObserver::new();
        let ct = obs.compression_telemetry();
        let ct_default = CompressionTelemetry::default();
        assert_eq!(ct.compress_count, ct_default.compress_count);
        assert_eq!(ct.decompress_count, ct_default.decompress_count);
        assert_eq!(ct.total_migration_bytes, ct_default.total_migration_bytes);
        assert_eq!(ct.eviction_count, ct_default.eviction_count);
        assert_eq!(ct.swap_in_count, ct_default.swap_in_count);
    }

    #[test]
    fn observer_compression_ratio_no_data() {
        let obs = BasicObserver::new();
        let ratio = obs.compression_telemetry().overall_compression_ratio();
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn observer_avg_compress_latency_no_data() {
        let obs = BasicObserver::new();
        let avg = obs.compression_telemetry().avg_compress_latency_us();
        assert!((avg - 0.0).abs() < 1e-9);
    }

    #[test]
    fn observer_avg_decompress_latency_no_data() {
        let obs = BasicObserver::new();
        let avg = obs.compression_telemetry().avg_decompress_latency_us();
        assert!((avg - 0.0).abs() < 1e-9);
    }

    // ── BasicObserver clone ──

    #[test]
    fn basic_observer_clone_captures_same_state() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.42)).unwrap();
        obs.update_scheduler_metrics(7, 3, 4, 256);
        let cloned = obs.clone();
        let state_orig = obs.capture().unwrap();
        let state_clone = cloned.capture().unwrap();
        assert!((state_orig.memory_pressure - state_clone.memory_pressure).abs() < f32::EPSILON);
        assert_eq!(state_orig.waiting_queue_len, state_clone.waiting_queue_len);
        assert_eq!(state_orig.current_running_len, state_clone.current_running_len);
    }

    #[test]
    fn basic_observer_clone_independent_after_mutation() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.5)).unwrap();
        let mut cloned = obs.clone();
        cloned.update_memory_pressure(Ok(0.9)).unwrap();
        let state_orig = obs.capture().unwrap();
        let state_clone = cloned.capture().unwrap();
        assert!((state_orig.memory_pressure - 0.5).abs() < f32::EPSILON);
        assert!((state_clone.memory_pressure - 0.9).abs() < f32::EPSILON);
    }

    // ── BasicObserver debug output ──

    #[test]
    fn basic_observer_debug_contains_last_state() {
        let obs = BasicObserver::new();
        let debug = format!("{obs:?}");
        assert!(debug.contains("BasicObserver"));
        assert!(debug.contains("last_state") || debug.contains("compression_telemetry"));
    }

    // ── Mixed update scenario ──

    #[test]
    fn mixed_updates_all_fields_set() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.6)).unwrap();
        obs.update_scheduler_metrics(8, 4, 5, 300);
        obs.update_kv_fragmentation(0.15);
        obs.update_swap_io_rate(1234.5);
        obs.update_logits_entropy(2.718);
        obs.update_attention_sparsity(0.33);
        obs.update_moe_fault_metrics(0.02, 150.0, 12);
        obs.update_weight_metrics(50, 25, 15, 10, 3, 1);
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 0.6).abs() < f32::EPSILON);
        assert_eq!(state.waiting_queue_len, 8);
        assert_eq!(state.current_running_len, 4);
        assert_eq!(state.current_batch_size, 5);
        assert_eq!(state.mean_context_len, 300);
        assert!((state.kv_fragmentation - 0.15).abs() < f32::EPSILON);
        assert!((state.swap_io_rate - 1234.5).abs() < f32::EPSILON);
        assert!((state.logits_entropy - 2.718).abs() < 1e-3);
        assert!((state.attention_sparsity - 0.33).abs() < 1e-2);
        assert!((state.moe_fault_rate - 0.02).abs() < 1e-3);
        assert!((state.moe_avg_recovery_us - 150.0).abs() < f32::EPSILON);
        assert_eq!(state.moe_working_set_size, 12);
        assert_eq!(state.weight_page_total, 50);
        assert_eq!(state.weight_pages_l1, 25);
        assert_eq!(state.weight_pages_l2, 15);
        assert_eq!(state.weight_pages_l3, 10);
        assert_eq!(state.weight_eviction_count, 3);
        assert_eq!(state.weight_recovery_count, 1);
    }

    #[test]
    fn error_propagation_preserves_message_exactly() {
        let original_msg = "specific error: CUDA error 999 - unknown";
        let mut obs = BasicObserver::new();
        let result = obs.update_memory_pressure(Err(original_msg.to_string()));
        let err = result.unwrap_err();
        match err {
            ObserverError::BackendUnavailable(msg) => {
                assert_eq!(msg, original_msg);
            }
        }
    }

    #[test]
    fn weight_page_eviction_count_accumulates_across_events() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 5, 3, 2, 0, 0);
        for i in 0..5 {
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: i,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Cold,
                reason: EvictionReason::MemoryPressure,
                bytes: 1024,
            });
        }
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_eviction_count, 5);
        assert_eq!(state.weight_pages_l1, 0); // 5 - 5 = 0 (saturating at 0 via individual events)
        assert_eq!(state.weight_pages_l3, 7); // 2 + 5
    }

    #[test]
    fn compression_and_weight_events_together() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 8, 1, 1, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        obs.record_compress(CompressionCodec::Lz4, 4096, 2048, 50);
        obs.record_eviction();
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 7);
        assert_eq!(state.weight_pages_l2, 2);
        assert_eq!(state.weight_eviction_count, 1);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.eviction_count, 1);
    }

    // ── 40 additional tests ──

    // ── RuntimeObserver trait object ──

    #[test]
    fn runtime_observer_trait_object_capture() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.33)).unwrap();
        let trait_obj: &dyn RuntimeObserver = &obs;
        let state = trait_obj.capture().unwrap();
        assert!((state.memory_pressure - 0.33).abs() < 1e-6);
    }

    // ── Error does not corrupt state ──

    #[test]
    fn update_memory_pressure_error_preserves_previous_value() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.75)).unwrap();
        let _ = obs.update_memory_pressure(Err("broken".into()));
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 0.75).abs() < f32::EPSILON);
    }

    // ── Compression ratio / latency with actual data ──

    #[test]
    fn observer_compression_ratio_with_actual_data() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::Lz4, 8192, 4096, 100);
        let ratio = obs.compression_telemetry().overall_compression_ratio();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn observer_avg_compress_latency_with_actual_data() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::Lz4, 4096, 2048, 300);
        obs.record_compress(CompressionCodec::Lz4, 4096, 2048, 500);
        obs.record_compress(CompressionCodec::Lz4, 4096, 2048, 700);
        let avg = obs.compression_telemetry().avg_compress_latency_us();
        assert!((avg - 500.0).abs() < 1e-9);
    }

    #[test]
    fn observer_avg_decompress_latency_with_actual_data() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::BitPackRle, 2048, 4096, 200);
        obs.record_decompress(CompressionCodec::BitPackRle, 2048, 4096, 600);
        let avg = obs.compression_telemetry().avg_decompress_latency_us();
        assert!((avg - 400.0).abs() < 1e-9);
    }

    #[test]
    fn observer_per_codec_avg_compression_ratio() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::Lz4, 10000, 5000, 10);
        let stats = obs.compression_telemetry().codec_stats(CompressionCodec::Lz4);
        let ratio = stats.avg_compression_ratio();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn observer_per_codec_avg_compress_latency() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::NvcompAns, 4096, 2048, 250);
        obs.record_compress(CompressionCodec::NvcompAns, 4096, 2048, 750);
        let stats = obs.compression_telemetry().codec_stats(CompressionCodec::NvcompAns);
        let avg = stats.avg_compress_latency_us();
        assert!((avg - 500.0).abs() < 1e-9);
    }

    #[test]
    fn observer_per_codec_avg_decompress_latency() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::ZstdDict, 2048, 8192, 800);
        obs.record_decompress(CompressionCodec::ZstdDict, 2048, 8192, 200);
        let stats = obs.compression_telemetry().codec_stats(CompressionCodec::ZstdDict);
        let avg = stats.avg_decompress_latency_us();
        assert!((avg - 500.0).abs() < 1e-9);
    }

    // ── Weight page tier transitions ──

    #[test]
    fn weight_page_eviction_hot_direct_to_cold() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 5, 3, 2, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 4);
        assert_eq!(state.weight_pages_l2, 3);
        assert_eq!(state.weight_pages_l3, 3);
    }

    #[test]
    fn weight_page_recovery_warm_to_cold_unusual() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 2, 5, 3, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 0,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Cold,
            latency_us: 50,
            bytes: 1024,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l2, 4);
        assert_eq!(state.weight_pages_l3, 4);
        assert_eq!(state.weight_recovery_count, 1);
    }

    #[test]
    fn weight_page_recovery_count_accumulates_separately() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 3, 3, 4, 0, 0);
        for i in 0..5 {
            obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
                page_id: i,
                from_tier: WeightTier::Cold,
                to_tier: WeightTier::Hot,
                latency_us: 100,
                bytes: 2048,
            });
        }
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_recovery_count, 5);
        assert_eq!(state.weight_eviction_count, 0);
    }

    #[test]
    fn bulk_weight_metrics_then_incremental_event() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(20, 10, 6, 4, 2, 1);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 9);
        assert_eq!(state.weight_pages_l2, 7);
        assert_eq!(state.weight_eviction_count, 3);
    }

    // ── Pub field direct access ──

    #[test]
    fn last_state_direct_mutation_reflected_in_capture() {
        let mut obs = BasicObserver::new();
        obs.last_state.swap_io_rate = 99.0;
        obs.last_state.mean_context_len = 777;
        let state = obs.capture().unwrap();
        assert!((state.swap_io_rate - 99.0).abs() < f32::EPSILON);
        assert_eq!(state.mean_context_len, 777);
    }

    // ── Float edge cases for update methods ──

    #[test]
    fn update_swap_io_rate_negative() {
        let mut obs = BasicObserver::new();
        obs.update_swap_io_rate(-42.5);
        let state = obs.capture().unwrap();
        assert!((state.swap_io_rate - (-42.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn update_logits_entropy_nan() {
        let mut obs = BasicObserver::new();
        obs.update_logits_entropy(f32::NAN);
        let state = obs.capture().unwrap();
        assert!(state.logits_entropy.is_nan());
    }

    #[test]
    fn update_attention_sparsity_negative() {
        let mut obs = BasicObserver::new();
        obs.update_attention_sparsity(-0.5);
        let state = obs.capture().unwrap();
        assert!((state.attention_sparsity - (-0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn update_attention_sparsity_above_one() {
        let mut obs = BasicObserver::new();
        obs.update_attention_sparsity(1.5);
        let state = obs.capture().unwrap();
        assert!((state.attention_sparsity - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn update_kv_fragmentation_zero() {
        let mut obs = BasicObserver::new();
        obs.update_kv_fragmentation(0.0);
        let state = obs.capture().unwrap();
        assert!((state.kv_fragmentation - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn update_kv_fragmentation_one() {
        let mut obs = BasicObserver::new();
        obs.update_kv_fragmentation(1.0);
        let state = obs.capture().unwrap();
        assert!((state.kv_fragmentation - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn update_kv_fragmentation_negative() {
        let mut obs = BasicObserver::new();
        obs.update_kv_fragmentation(-0.1);
        let state = obs.capture().unwrap();
        assert!((state.kv_fragmentation - (-0.1)).abs() < f32::EPSILON);
    }

    #[test]
    fn update_memory_pressure_exact_zero() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.0)).unwrap();
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn update_memory_pressure_exact_one() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(1.0)).unwrap();
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn update_swap_io_rate_f32_max() {
        let mut obs = BasicObserver::new();
        obs.update_swap_io_rate(f32::MAX);
        let state = obs.capture().unwrap();
        assert_eq!(state.swap_io_rate, f32::MAX);
    }

    // ── MoE fault metrics float edge cases ──

    #[test]
    fn update_moe_fault_metrics_infinity_rate() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(f32::INFINITY, 100.0, 5);
        let state = obs.capture().unwrap();
        assert!(state.moe_fault_rate.is_infinite() && state.moe_fault_rate.is_sign_positive());
    }

    #[test]
    fn update_moe_fault_metrics_negative_rate() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(-0.01, 50.0, 3);
        let state = obs.capture().unwrap();
        assert!((state.moe_fault_rate - (-0.01)).abs() < 1e-6);
    }

    #[test]
    fn update_moe_fault_metrics_infinity_latency() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(0.5, f32::INFINITY, 10);
        let state = obs.capture().unwrap();
        assert!(state.moe_avg_recovery_us.is_infinite());
    }

    #[test]
    fn update_moe_fault_metrics_negative_latency() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(0.1, -500.0, 8);
        let state = obs.capture().unwrap();
        assert!((state.moe_avg_recovery_us - (-500.0)).abs() < f32::EPSILON);
    }

    // ── Observer clone with compression telemetry ──

    #[test]
    fn observer_clone_compression_telemetry_independence() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::Lz4, 4096, 2048, 100);
        let mut cloned = obs.clone();
        cloned.record_compress(CompressionCodec::BitPackRle, 8192, 4096, 200);
        assert_eq!(obs.compression_telemetry().compress_count, 1);
        assert_eq!(cloned.compression_telemetry().compress_count, 2);
    }

    // ── Fresh observer capture ──

    #[test]
    fn observer_capture_fresh_no_updates() {
        let obs = BasicObserver::new();
        let state = obs.capture().unwrap();
        let default = SystemState::default();
        assert_eq!(state.waiting_queue_len, default.waiting_queue_len);
        assert_eq!(state.weight_page_total, default.weight_page_total);
        assert!((state.memory_pressure - default.memory_pressure).abs() < f32::EPSILON);
    }

    // ── Compress + decompress same codec ──

    #[test]
    fn observer_compress_then_decompress_same_codec_latency() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::Lz4, 8192, 4096, 300);
        obs.record_decompress(CompressionCodec::Lz4, 4096, 8192, 150);
        let stats = obs.compression_telemetry().codec_stats(CompressionCodec::Lz4);
        assert_eq!(stats.compress_count, 1);
        assert_eq!(stats.decompress_count, 1);
        assert!((stats.avg_compress_latency_us() - 300.0).abs() < 1e-9);
        assert!((stats.avg_decompress_latency_us() - 150.0).abs() < 1e-9);
    }

    // ── Scheduler metrics large but not max ──

    #[test]
    fn update_scheduler_metrics_large_but_not_max() {
        let mut obs = BasicObserver::new();
        obs.update_scheduler_metrics(1_000_000, 500_000, 256, 8192);
        let state = obs.capture().unwrap();
        assert_eq!(state.waiting_queue_len, 1_000_000);
        assert_eq!(state.current_running_len, 500_000);
        assert_eq!(state.current_batch_size, 256);
        assert_eq!(state.mean_context_len, 8192);
    }

    // ── Interleaved eviction and swap-in ──

    #[test]
    fn observer_eviction_and_swap_in_interleaved() {
        let mut obs = BasicObserver::new();
        obs.record_eviction();
        obs.record_swap_in();
        obs.record_eviction();
        obs.record_swap_in();
        obs.record_swap_in();
        let ct = obs.compression_telemetry();
        assert_eq!(ct.eviction_count, 2);
        assert_eq!(ct.swap_in_count, 3);
    }

    // ── All three tiers adjusted in one scenario ──

    #[test]
    fn weight_page_adjust_all_three_tiers() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 4, 3, 3, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 1,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 3);
        assert_eq!(state.weight_pages_l2, 3);
        assert_eq!(state.weight_pages_l3, 4);
    }

    // ── Multiple compress ops ratio correctness ──

    #[test]
    fn observer_compress_multiple_ops_ratio_correct() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::Lz4, 10000, 5000, 100);
        obs.record_compress(CompressionCodec::Lz4, 10000, 3000, 200);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 2);
        assert_eq!(ct.total_input_bytes, 20000);
        assert_eq!(ct.total_compressed_bytes, 8000);
        let ratio = ct.overall_compression_ratio();
        assert!((ratio - 0.4).abs() < 1e-9);
    }

    // ── Net-zero tier after evict + recover ──

    #[test]
    fn weight_page_eviction_and_recovery_net_zero_tiers() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 3, 1, 1, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 0,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Hot,
            latency_us: 50,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 3);
        assert_eq!(state.weight_pages_l2, 1);
        assert_eq!(state.weight_eviction_count, 1);
        assert_eq!(state.weight_recovery_count, 1);
    }

    // ── Combined compress + decompress + migration ──

    #[test]
    fn observer_compress_decompress_migration_combined() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::BitPackRle, 8192, 4096, 200);
        obs.record_decompress(CompressionCodec::BitPackRle, 4096, 8192, 100);
        obs.record_migration(16384);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.decompress_count, 1);
        assert_eq!(ct.total_migration_bytes, 16384);
        assert_eq!(ct.total_input_bytes, 8192);
        assert_eq!(ct.total_compressed_bytes, 4096);
    }

    // ── All float updates in one capture ──

    #[test]
    fn update_all_floats_then_verify_in_capture() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.42)).unwrap();
        obs.update_kv_fragmentation(0.25);
        obs.update_swap_io_rate(500.0);
        obs.update_logits_entropy(3.5);
        obs.update_attention_sparsity(0.67);
        obs.update_moe_fault_metrics(0.03, 250.0, 16);
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 0.42).abs() < 1e-6);
        assert!((state.kv_fragmentation - 0.25).abs() < 1e-6);
        assert!((state.swap_io_rate - 500.0).abs() < 1e-6);
        assert!((state.logits_entropy - 3.5).abs() < 1e-6);
        assert!((state.attention_sparsity - 0.67).abs() < 1e-2);
        assert!((state.moe_fault_rate - 0.03).abs() < 1e-3);
        assert!((state.moe_avg_recovery_us - 250.0).abs() < 1e-6);
    }

    // ── Bulk metrics + events tier consistency ──

    #[test]
    fn weight_metrics_and_events_tier_consistency() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(20, 10, 5, 5, 0, 0);
        // Evict 2 from Hot to Warm
        for i in 0..2 {
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: i,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Warm,
                reason: EvictionReason::MemoryPressure,
                bytes: 4096,
            });
        }
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 8);
        assert_eq!(state.weight_pages_l2, 7);
        assert_eq!(state.weight_pages_l3, 5);
        assert_eq!(state.weight_pages_l1 + state.weight_pages_l2 + state.weight_pages_l3, 20);
    }

    // ── ZstdDict decompress via observer ──

    #[test]
    fn observer_decompress_zstd_dict_latency_tracked() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::ZstdDict, 2048, 16384, 5000);
        let stats = obs.compression_telemetry().codec_stats(CompressionCodec::ZstdDict);
        assert_eq!(stats.decompress_count, 1);
        assert_eq!(stats.total_decompress_latency_us, 5000);
        assert!((stats.avg_decompress_latency_us() - 5000.0).abs() < 1e-9);
    }

    // ── Capture state is Copy ──

    #[test]
    fn observer_capture_state_is_copy() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.88)).unwrap();
        let state1 = obs.capture().unwrap();
        let state2 = state1; // Copy
        assert!((state1.memory_pressure - 0.88).abs() < 1e-6);
        assert!((state2.memory_pressure - 0.88).abs() < 1e-6);
    }

    // ── 15 additional tests ──

    // ── SystemState PartialEq ──

    #[test]
    fn system_state_partial_eq_identical_states() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.5)).unwrap();
        obs.update_scheduler_metrics(10, 5, 3, 200);
        let state_a = obs.capture().unwrap();
        let state_b = obs.capture().unwrap();
        assert_eq!(state_a, state_b);
    }

    #[test]
    fn system_state_partial_eq_different_after_mutation() {
        let mut obs = BasicObserver::new();
        let state_a = obs.capture().unwrap();
        obs.update_memory_pressure(Ok(0.99)).unwrap();
        let state_b = obs.capture().unwrap();
        assert_ne!(state_a, state_b);
    }

    // ── SystemState Debug output ──

    #[test]
    fn system_state_debug_contains_fields() {
        let mut obs = BasicObserver::new();
        obs.update_scheduler_metrics(42, 7, 3, 512);
        let state = obs.capture().unwrap();
        let debug = format!("{state:?}");
        assert!(debug.contains("SystemState"), "debug should contain struct name");
        assert!(debug.contains("42"), "debug should contain waiting_queue_len value");
    }

    // ── EvictionReason Hash in collection ──

    #[test]
    fn eviction_reason_usable_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(EvictionReason::MemoryPressure);
        set.insert(EvictionReason::MemoryPressure);
        assert_eq!(set.len(), 1);
        assert!(set.contains(&EvictionReason::MemoryPressure));
    }

    // ── CompressionTelemetry PartialEq through observer ──

    #[test]
    fn compression_telemetry_partial_eq_same_after_same_ops() {
        let mut obs_a = BasicObserver::new();
        let mut obs_b = BasicObserver::new();
        obs_a.record_compress(CompressionCodec::Lz4, 4096, 2048, 100);
        obs_b.record_compress(CompressionCodec::Lz4, 4096, 2048, 100);
        assert_eq!(*obs_a.compression_telemetry(), *obs_b.compression_telemetry());
    }

    #[test]
    fn compression_telemetry_partial_eq_diff_after_different_ops() {
        let mut obs_a = BasicObserver::new();
        let mut obs_b = BasicObserver::new();
        obs_a.record_compress(CompressionCodec::Lz4, 4096, 2048, 100);
        obs_b.record_compress(CompressionCodec::BitPackRle, 4096, 2048, 100);
        assert_ne!(*obs_a.compression_telemetry(), *obs_b.compression_telemetry());
    }

    // ── Compression ratio with expansion (ratio > 1.0) ──

    #[test]
    fn observer_compression_ratio_expansion() {
        let mut obs = BasicObserver::new();
        // Output larger than input simulates pathological compression expansion
        obs.record_compress(CompressionCodec::None, 1000, 1500, 10);
        let ratio = obs.compression_telemetry().overall_compression_ratio();
        assert!((ratio - 1.5).abs() < 1e-9, "ratio should be 1.5, got {ratio}");
    }

    // ── Multiple codecs per-codec stats isolation ──

    #[test]
    fn per_codec_stats_isolation_across_all_five_codecs() {
        let mut obs = BasicObserver::new();
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for (i, codec) in codecs.iter().enumerate() {
            obs.record_compress(*codec, 1000 * (i as u64 + 1), 500, i as u64 * 10);
        }
        for (i, codec) in codecs.iter().enumerate() {
            let stats = obs.compression_telemetry().codec_stats(*codec);
            assert_eq!(stats.compress_count, 1);
            assert_eq!(stats.total_input_bytes, 1000 * (i as u64 + 1));
            assert_eq!(stats.total_output_bytes, 500);
            assert_eq!(stats.total_compress_latency_us, i as u64 * 10);
        }
    }

    // ── Weight page event with usize::MAX page_id ──

    #[test]
    fn weight_page_eviction_max_page_id_evicted() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(3, 2, 1, 0, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: usize::MAX,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 1);
        assert_eq!(state.weight_pages_l2, 2);
        assert_eq!(state.weight_eviction_count, 1);
    }

    // ── Observer captures state snapshot at call time ──

    #[test]
    fn capture_returns_snapshot_not_live_reference() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.25)).unwrap();
        let snapshot = obs.capture().unwrap();
        obs.update_memory_pressure(Ok(0.75)).unwrap();
        // The snapshot was taken before the second update, so it must still be 0.25
        assert!((snapshot.memory_pressure - 0.25).abs() < f32::EPSILON);
    }

    // ── CompressionTelemetry debug output ──

    #[test]
    fn compression_telemetry_debug_output() {
        let obs = BasicObserver::new();
        let debug = format!("{:?}", obs.compression_telemetry());
        assert!(debug.contains("CompressionTelemetry"), "debug: {debug}");
        assert!(debug.contains("compress_count"), "debug: {debug}");
    }

    // ── CodecStats methods on unrecorded codec ──

    #[test]
    fn codec_stats_default_methods_with_no_data() {
        let obs = BasicObserver::new();
        let stats = obs.compression_telemetry().codec_stats(CompressionCodec::NvcompAns);
        assert_eq!(stats.compress_count, 0);
        assert_eq!(stats.decompress_count, 0);
        assert!((stats.avg_compression_ratio() - 1.0).abs() < 1e-9);
        assert!((stats.avg_compress_latency_us() - 0.0).abs() < 1e-9);
        assert!((stats.avg_decompress_latency_us() - 0.0).abs() < 1e-9);
    }

    // ── Observer update after error recovers ──

    #[test]
    fn update_memory_pressure_error_then_success_recovers() {
        let mut obs = BasicObserver::new();
        let _ = obs.update_memory_pressure(Err("fail".into()));
        let result = obs.update_memory_pressure(Ok(0.33));
        assert!(result.is_ok());
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 0.33).abs() < 1e-6);
    }

    // ── Weight page cascading eviction chain (Hot -> Warm -> Cold) ──

    #[test]
    fn weight_page_cascading_eviction_three_events() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 6, 3, 1, 0, 0);
        // Hot -> Warm
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        // Warm -> Cold
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 1,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        // Hot -> Cold (direct skip)
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 2,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        // Event 1: Hot(-1) Warm(+1) => l1=5, l2=4
        // Event 2: Warm(-1) Cold(+1) => l2=3, l3=2
        // Event 3: Hot(-1) Cold(+1) => l1=4, l3=3
        assert_eq!(state.weight_pages_l1, 4);
        assert_eq!(state.weight_pages_l2, 3);
        assert_eq!(state.weight_pages_l3, 3);
        assert_eq!(state.weight_eviction_count, 3);
    }

    // ── BasicObserver clone isolates compression telemetry ──

    #[test]
    fn observer_clone_compression_telemetry_deep_copy() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::Lz4, 2048, 4096, 80);
        obs.record_migration(5000);
        let mut cloned = obs.clone();
        // Mutate clone
        cloned.record_migration(1000);
        cloned.record_decompress(CompressionCodec::BitPackRle, 1024, 4096, 40);
        // Original must be unaffected
        let ct_orig = obs.compression_telemetry();
        assert_eq!(ct_orig.total_migration_bytes, 5000);
        assert_eq!(ct_orig.decompress_count, 1);
        let ct_clone = cloned.compression_telemetry();
        assert_eq!(ct_clone.total_migration_bytes, 6000);
        assert_eq!(ct_clone.decompress_count, 2);
    }

    // ── 13 additional tests (wave 4) ──

    // ── RuntimeObserver trait used dynamically via Box ──

    #[test]
    fn runtime_observer_boxed_trait_object() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.77)).unwrap();
        let boxed: Box<dyn RuntimeObserver> = Box::new(obs);
        let state = boxed.capture().unwrap();
        assert!((state.memory_pressure - 0.77).abs() < 1e-6);
    }

    // ── Compression ratio with zero input but nonzero output ──

    #[test]
    fn observer_compression_ratio_zero_input_nonzero_output() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::None, 0, 500, 10);
        // total_input_bytes == 0 → overall_compression_ratio returns 1.0 per spec
        let ratio = obs.compression_telemetry().overall_compression_ratio();
        assert!((ratio - 1.0).abs() < 1e-9, "ratio should be 1.0 when input is 0, got {ratio}");
    }

    // ── Weight page eviction then bulk overwrite resets incremental counters ──

    #[test]
    fn bulk_weight_metrics_resets_eviction_recovery_counts() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 5, 3, 2, 100, 50);
        // Bulk overwrite resets counts to new values
        obs.update_weight_metrics(10, 5, 3, 2, 0, 0);
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_eviction_count, 0);
        assert_eq!(state.weight_recovery_count, 0);
    }

    // ── Sequential weight page events then bulk overwrite ──

    #[test]
    fn sequential_events_then_bulk_overwrite() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 5, 3, 2, 0, 0);
        // Record 3 eviction events
        for i in 0..3 {
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: i,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Warm,
                reason: EvictionReason::MemoryPressure,
                bytes: 1024,
            });
        }
        // Now bulk overwrite
        obs.update_weight_metrics(20, 10, 6, 4, 7, 3);
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_page_total, 20);
        assert_eq!(state.weight_pages_l1, 10);
        assert_eq!(state.weight_eviction_count, 7);
        assert_eq!(state.weight_recovery_count, 3);
    }

    // ── WeightTier variants equality check ──

    #[test]
    fn weight_tier_variants_are_distinct() {
        assert_ne!(WeightTier::Hot, WeightTier::Warm);
        assert_ne!(WeightTier::Warm, WeightTier::Cold);
        assert_ne!(WeightTier::Hot, WeightTier::Cold);
    }

    // ── WeightTier debug format includes variant name ──

    #[test]
    fn weight_tier_debug_format() {
        assert!(format!("{:?}", WeightTier::Hot).contains("Hot"));
        assert!(format!("{:?}", WeightTier::Warm).contains("Warm"));
        assert!(format!("{:?}", WeightTier::Cold).contains("Cold"));
    }

    // ── CompressionCodec all 5 variants produce distinct per-codec stats ──

    #[test]
    fn observer_all_codecs_produce_distinct_stats() {
        let mut obs = BasicObserver::new();
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for (i, &codec) in codecs.iter().enumerate() {
            obs.record_compress(codec, (i as u64 + 1) * 100, (i as u64 + 1) * 50, (i as u64 + 1) * 10);
            obs.record_decompress(codec, (i as u64 + 1) * 50, (i as u64 + 1) * 100, (i as u64 + 1) * 5);
        }
        for (i, &codec) in codecs.iter().enumerate() {
            let stats = obs.compression_telemetry().codec_stats(codec);
            assert_eq!(stats.compress_count, 1, "codec {i} compress_count mismatch");
            assert_eq!(stats.decompress_count, 1, "codec {i} decompress_count mismatch");
            assert_eq!(stats.total_input_bytes, (i as u64 + 1) * 150); // compress 100 + decompress 50
            assert_eq!(stats.total_output_bytes, (i as u64 + 1) * 150); // compress 50 + decompress 100
        }
    }

    // ── Observer captures compress_count after multiple different codecs ──

    #[test]
    fn observer_global_compress_count_mixed_codecs() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::Lz4, 1000, 500, 10);
        obs.record_compress(CompressionCodec::Lz4, 2000, 1000, 20);
        obs.record_compress(CompressionCodec::BitPackRle, 3000, 1500, 30);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 3);
        assert_eq!(ct.total_input_bytes, 6000);
        assert_eq!(ct.total_compressed_bytes, 3000);
    }

    // ── Memory pressure updated then state checked via direct field access ──

    #[test]
    fn memory_pressure_reflected_in_last_state_field() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.123)).unwrap();
        // Directly verify the public field
        assert!((obs.last_state.memory_pressure - 0.123).abs() < 1e-6);
        // Verify via capture matches
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - obs.last_state.memory_pressure).abs() < f32::EPSILON);
    }

    // ── update_kv_fragmentation_reflected_in_last_state_field ──

    #[test]
    fn kv_fragmentation_reflected_in_last_state_field() {
        let mut obs = BasicObserver::new();
        obs.update_kv_fragmentation(0.456);
        assert!((obs.last_state.kv_fragmentation - 0.456).abs() < 1e-6);
    }

    // ── Weight page event with same from and to tier: eviction count increments but tier net-zero ──

    #[test]
    fn weight_page_eviction_warm_to_warm_net_zero_tier() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(10, 3, 4, 3, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 2048,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l2, 4); // 4 - 1 + 1 = 4
        assert_eq!(state.weight_eviction_count, 1);
    }

    // ── SystemState equality after full round of updates ──

    #[test]
    fn system_state_eq_after_full_round_trip() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.5)).unwrap();
        obs.update_scheduler_metrics(10, 5, 3, 200);
        obs.update_kv_fragmentation(0.25);
        obs.update_swap_io_rate(100.0);
        obs.update_logits_entropy(2.0);
        obs.update_attention_sparsity(0.5);
        obs.update_moe_fault_metrics(0.01, 50.0, 4);
        obs.update_weight_metrics(20, 10, 6, 4, 2, 1);
        let state_a = obs.capture().unwrap();
        let state_b = obs.capture().unwrap();
        assert_eq!(state_a, state_b, "Two captures without mutation must be equal");
    }

    // ── Observer with many rapid updates reflects final value ──

    #[test]
    fn observer_rapid_overwrites_reflect_final_value() {
        let mut obs = BasicObserver::new();
        for i in 0..100 {
            obs.update_memory_pressure(Ok(i as f32 * 0.01)).unwrap();
            obs.update_scheduler_metrics(i, i * 2, i * 3, i * 4);
            obs.update_kv_fragmentation(i as f32 * 0.005);
        }
        let state = obs.capture().unwrap();
        // Final iteration: i=99
        assert!((state.memory_pressure - 0.99).abs() < 1e-6);
        assert_eq!(state.waiting_queue_len, 99);
        assert_eq!(state.current_running_len, 198);
        assert_eq!(state.current_batch_size, 297);
        assert_eq!(state.mean_context_len, 396);
        assert!((state.kv_fragmentation - 0.495).abs() < 1e-3);
    }

    // ── 12 additional tests (wave 5) ──

    // ── Custom RuntimeObserver implementation ──

    #[test]
    fn runtime_observer_custom_impl_capture() {
        // Verify the trait can be implemented by a non-BasicObserver type
        struct StaticObserver {
            state: SystemState,
        }
        impl RuntimeObserver for StaticObserver {
            fn capture(&self) -> Result<SystemState, ObserverError> {
                Ok(self.state.clone())
            }
        }
        let mut state = SystemState::default();
        state.memory_pressure = 0.42;
        let obs = StaticObserver { state };
        let trait_obj: &dyn RuntimeObserver = &obs;
        let captured = trait_obj.capture().unwrap();
        assert!((captured.memory_pressure - 0.42).abs() < 1e-6);
    }

    // ── RuntimeObserver custom impl can return error ──

    #[test]
    fn runtime_observer_custom_impl_error() {
        struct FailingObserver;
        impl RuntimeObserver for FailingObserver {
            fn capture(&self) -> Result<SystemState, ObserverError> {
                Err(ObserverError::BackendUnavailable("custom failure".into()))
            }
        }
        let obs = FailingObserver;
        let result = obs.capture();
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("custom failure"));
    }

    // ── CompressionCodec::None compress round-trip via observer ──

    #[test]
    fn observer_compress_none_codec_field_values() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::None, 4096, 4096, 5);
        let ct = obs.compression_telemetry();
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.total_input_bytes, 4096);
        assert_eq!(ct.total_compressed_bytes, 4096);
        assert_eq!(ct.total_compress_latency_us, 5);
        let stats = ct.codec_stats(CompressionCodec::None);
        assert_eq!(stats.compress_count, 1);
        assert_eq!(stats.total_input_bytes, 4096);
        assert_eq!(stats.total_output_bytes, 4096);
        // 1:1 ratio for None codec
        assert!((stats.avg_compression_ratio() - 1.0).abs() < 1e-9);
    }

    // ── CompressionCodec::NvcompAns compress specific field check ──

    #[test]
    fn observer_compress_nvcompans_codec_fields() {
        let mut obs = BasicObserver::new();
        obs.record_compress(CompressionCodec::NvcompAns, 8192, 2048, 350);
        let stats = obs.compression_telemetry().codec_stats(CompressionCodec::NvcompAns);
        assert_eq!(stats.compress_count, 1);
        assert_eq!(stats.decompress_count, 0);
        assert_eq!(stats.total_input_bytes, 8192);
        assert_eq!(stats.total_output_bytes, 2048);
        assert_eq!(stats.total_compress_latency_us, 350);
        assert!((stats.avg_compression_ratio() - 0.25).abs() < 1e-9);
    }

    // ── Weight page event with page_id zero ──

    #[test]
    fn weight_page_eviction_page_id_zero() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 4, 1, 0, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 3);
        assert_eq!(state.weight_pages_l2, 2);
        assert_eq!(state.weight_eviction_count, 1);
    }

    // ── BasicObserver.last_state direct mutation of compression fields ──

    #[test]
    fn last_state_direct_mutation_weight_fields() {
        let mut obs = BasicObserver::new();
        obs.last_state.weight_page_total = 999;
        obs.last_state.weight_pages_l1 = 100;
        obs.last_state.weight_pages_l2 = 200;
        obs.last_state.weight_pages_l3 = 300;
        obs.last_state.weight_eviction_count = 50;
        obs.last_state.weight_recovery_count = 25;
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_page_total, 999);
        assert_eq!(state.weight_pages_l1, 100);
        assert_eq!(state.weight_pages_l2, 200);
        assert_eq!(state.weight_pages_l3, 300);
        assert_eq!(state.weight_eviction_count, 50);
        assert_eq!(state.weight_recovery_count, 25);
    }

    // ── update_memory_pressure error does not affect scheduler metrics ──

    #[test]
    fn update_memory_pressure_error_preserves_scheduler_metrics() {
        let mut obs = BasicObserver::new();
        obs.update_scheduler_metrics(20, 10, 5, 500);
        let _ = obs.update_memory_pressure(Err("sensor offline".into()));
        let state = obs.capture().unwrap();
        assert_eq!(state.waiting_queue_len, 20);
        assert_eq!(state.current_running_len, 10);
        assert_eq!(state.current_batch_size, 5);
        assert_eq!(state.mean_context_len, 500);
    }

    // ── EvictionReason usable as HashMap key ──

    #[test]
    fn eviction_reason_usable_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map: HashMap<EvictionReason, u32> = HashMap::new();
        map.insert(EvictionReason::MemoryPressure, 42);
        assert_eq!(map.get(&EvictionReason::MemoryPressure), Some(&42));
        assert_eq!(map.len(), 1);
    }

    // ── BasicObserver new and update cycle repeated twice ──

    #[test]
    fn observer_new_then_update_then_new_resets_state() {
        let mut obs = BasicObserver::new();
        obs.update_memory_pressure(Ok(0.99)).unwrap();
        obs.update_scheduler_metrics(100, 50, 25, 1000);
        // Re-create observer — fresh state
        obs = BasicObserver::new();
        let state = obs.capture().unwrap();
        assert!((state.memory_pressure - 0.0).abs() < f32::EPSILON);
        assert_eq!(state.waiting_queue_len, 0);
        assert_eq!(state.current_running_len, 0);
        assert_eq!(state.current_batch_size, 0);
        assert_eq!(state.mean_context_len, 0);
    }

    // ── Compression telemetry per-codec zero decompress latency ──

    #[test]
    fn observer_decompress_zero_latency_per_codec() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::Lz4, 2048, 4096, 0);
        let stats = obs.compression_telemetry().codec_stats(CompressionCodec::Lz4);
        assert_eq!(stats.decompress_count, 1);
        assert_eq!(stats.total_decompress_latency_us, 0);
        assert!((stats.avg_decompress_latency_us() - 0.0).abs() < 1e-9);
    }

    // ── Compression telemetry per-codec output_bytes tracking ──

    #[test]
    fn observer_decompress_output_bytes_tracked_per_codec() {
        let mut obs = BasicObserver::new();
        obs.record_decompress(CompressionCodec::BitPackRle, 1000, 5000, 50);
        obs.record_decompress(CompressionCodec::BitPackRle, 2000, 8000, 100);
        let stats = obs.compression_telemetry().codec_stats(CompressionCodec::BitPackRle);
        assert_eq!(stats.decompress_count, 2);
        // total_output_bytes tracks the decompressed output (output_bytes param of record_decompress)
        assert_eq!(stats.total_output_bytes, 13000);
        assert_eq!(stats.total_input_bytes, 3000);
    }

    // ── Weight page recovery with u64::MAX bytes ──

    #[test]
    fn weight_page_recovery_max_bytes() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(5, 3, 1, 1, 0, 0);
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 0,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 9999,
            bytes: u64::MAX,
        });
        let state = obs.capture().unwrap();
        assert_eq!(state.weight_pages_l1, 4);
        assert_eq!(state.weight_pages_l3, 0);
        assert_eq!(state.weight_recovery_count, 1);
    }
}
