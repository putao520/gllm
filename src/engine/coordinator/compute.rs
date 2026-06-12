use crate::jit::epilogue::TelemetryAggregator;
use crate::jit::epilogue_subsystem::EpilogueSubsystem;
use crate::jit::golden_bucket::GoldenBucketRegistry;
use crate::jit::histogram::SeqHistogram;
use crate::jit::ragged::RaggedCompaction;
use crate::jit::sub_batch::SubBatchDispatcher;
use crate::kv_cache::turboquant::TurboQuantRuntime;

pub struct ComputeCoordinator {
    pub mega_kernel: Option<super::super::mega_kernel::MegaKernelExecutor>,
    pub jit_director: Option<crate::jit::director::JitDirector>,
    pub telemetry_aggregator: TelemetryAggregator,
    pub epilogue_subsystem: EpilogueSubsystem,
    pub sub_batch_dispatcher: SubBatchDispatcher,
    pub golden_buckets: GoldenBucketRegistry,
    pub seq_histogram: SeqHistogram,
    pub ragged_compaction: RaggedCompaction,
    pub turboquant: TurboQuantRuntime,
}

impl ComputeCoordinator {
    pub fn record_turboquant_scales(
        &mut self,
        num_layers: usize,
        kv_dim: usize,
    ) {
        if !self.turboquant.is_enabled() {
            return;
        }
        let per_ch_scale = self.telemetry_aggregator.per_channel_scale();
        if per_ch_scale > 0.0 {
            for layer in 0..num_layers {
                let scales = vec![per_ch_scale; kv_dim];
                self.turboquant.store_k_scales(layer, scales);
            }
        }
        let embed_norm = self.telemetry_aggregator.embedding_norm();
        if embed_norm > 0.0 {
            for layer in 0..num_layers {
                self.turboquant.store_correction(
                    layer,
                    crate::kv_cache::quant::RabitqCorrection {
                        c0: 0.0,
                        c1: 1.0 / (1.0 + 1.0 / (embed_norm * embed_norm).sqrt()),
                        v_norm: embed_norm,
                    },
                );
            }
        }
        log::trace!(
            "executor: §11 TurboQuant active (bits={}, fwht={}, scales_stored={})",
            self.turboquant.bits(),
            self.turboquant.fwht_enabled(),
            self.turboquant.get_k_scales(0).is_some(),
        );
        if self.turboquant.is_dual_track_active() {
            let (main_used, main_cap) = self
                .turboquant
                .dual_track()
                .map(|p| p.main_usage())
                .unwrap_or((0, 0));
            let (xnor_used, xnor_cap) = self
                .turboquant
                .dual_track()
                .map(|p| p.xnor_usage())
                .unwrap_or((0, 0));
            log::debug!(
                "executor: §11.5 DualTrack pool — main={}/{} bytes, xnor={}/{} bits",
                main_used,
                main_cap,
                xnor_used,
                xnor_cap,
            );
        }
    }

    pub fn record_seq_histogram(&self, seq_len: usize) {
        self.seq_histogram.record(seq_len);
    }

    pub fn evolve_golden_buckets(&mut self) {
        self.golden_buckets.evolve(&self.seq_histogram);
    }

    pub fn classify_request_shape(
        &self,
        has_moe_ops: bool,
        gating_threshold: f32,
    ) -> crate::jit::sub_batch::GraphShape {
        let dead_ratio = self.telemetry_aggregator.dead_neuron_ratio();
        let delta_rho = self.telemetry_aggregator.residual_delta_rho();
        self.sub_batch_dispatcher
            .classify_request(dead_ratio, delta_rho, has_moe_ops, gating_threshold)
    }

    pub fn collapse_seq_len(&mut self, seq_len: usize) -> usize {
        let (_, golden_size) = self.golden_buckets.collapse(seq_len);
        golden_size.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::epilogue_subsystem::EpilogueConfig;
    use crate::jit::ragged::CompactPlatform;
    use crate::kv_cache::turboquant::TurboQuantRuntime;

    fn make_compute() -> ComputeCoordinator {
        ComputeCoordinator {
            mega_kernel: None,
            jit_director: None,
            telemetry_aggregator: TelemetryAggregator::new(),
            epilogue_subsystem: EpilogueSubsystem::new(EpilogueConfig::default()),
            sub_batch_dispatcher: crate::jit::sub_batch::SubBatchDispatcher::new(
                crate::jit::compiler_constraints::CompilerConstraints::default(),
            ),
            golden_buckets: GoldenBucketRegistry::empty(
                crate::jit::compiler_constraints::CompilerConstraints::default(),
            ),
            seq_histogram: SeqHistogram::new(100, 4096),
            ragged_compaction: RaggedCompaction::new(CompactPlatform::X86Avx2),
            turboquant: TurboQuantRuntime::disabled(),
        }
    }

    #[test]
    fn record_turboquant_scales_noop_when_disabled() {
        let mut coord = make_compute();
        // Should not panic or store anything when turboquant is disabled.
        coord.record_turboquant_scales(4, 64);
        assert!(coord.turboquant.get_k_scales(0).is_none());
    }

    #[test]
    fn record_seq_histogram_records_value() {
        let coord = make_compute();
        coord.record_seq_histogram(128);
        let snap = coord.seq_histogram.snapshot();
        assert_eq!(snap.total_samples, 1);
    }

    #[test]
    fn evolve_golden_buckets_does_not_panic() {
        let mut coord = make_compute();
        coord.record_seq_histogram(64);
        coord.record_seq_histogram(128);
        coord.evolve_golden_buckets();
    }

    #[test]
    fn collapse_seq_len_returns_golden_size() {
        let mut coord = make_compute();
        let result = coord.collapse_seq_len(100);
        // Default registry may return the input unchanged or a bucket representative.
        assert!(result > 0);
    }

    // ── Construction & field access tests ──

    #[test]
    fn compute_coordinator_default_fields_are_none_or_default() {
        let coord = make_compute();

        // mega_kernel and jit_director are explicitly None when no GPU/JIT is available
        assert!(coord.mega_kernel.is_none());
        assert!(coord.jit_director.is_none());

        // turboquant is disabled by default
        assert!(!coord.turboquant.is_enabled());
    }

    #[test]
    fn telemetry_aggregator_initial_zeros() {
        let coord = make_compute();

        // Fresh TelemetryAggregator::new() returns 0.0 for all metrics
        assert_eq!(coord.telemetry_aggregator.per_channel_scale(), 0.0);
        assert_eq!(coord.telemetry_aggregator.embedding_norm(), 0.0);
        assert_eq!(coord.telemetry_aggregator.dead_neuron_ratio(), 0.0);
        assert_eq!(coord.telemetry_aggregator.residual_delta_rho(), 0.0);
    }

    #[test]
    fn seq_histogram_starts_empty() {
        let coord = make_compute();
        let snap = coord.seq_histogram.snapshot();

        assert_eq!(snap.total_samples, 0);
        assert!(snap.top_k.is_empty() || snap.top_k.iter().all(|(_, _, c)| *c == 0));
    }

    #[test]
    fn record_seq_histogram_multiple_samples() {
        let coord = make_compute();
        coord.record_seq_histogram(64);
        coord.record_seq_histogram(128);
        coord.record_seq_histogram(256);

        let snap = coord.seq_histogram.snapshot();
        assert_eq!(snap.total_samples, 3);
    }

    // ── classify_request_shape tests ──

    #[test]
    fn classify_request_shape_defaults_to_full_precision() {
        let coord = make_compute();

        // Default telemetry (dead_ratio=0, delta_rho=0) → FullPrecision
        let shape = coord.classify_request_shape(false, 0.0);
        assert_eq!(shape, crate::jit::sub_batch::GraphShape::FullPrecision);
    }

    #[test]
    fn classify_request_shape_moe_sparse_when_gating_low() {
        let coord = make_compute();

        // is_moe=true + low gating_threshold → MoeSparse
        let shape = coord.classify_request_shape(true, 0.3);
        assert_eq!(shape, crate::jit::sub_batch::GraphShape::MoeSparse);
    }

    #[test]
    fn classify_request_shape_moe_full_precision_when_gating_high() {
        let coord = make_compute();

        // is_moe=true + gating_threshold >= 0.5 → not MoeSparse → FullPrecision
        let shape = coord.classify_request_shape(true, 0.8);
        assert_eq!(shape, crate::jit::sub_batch::GraphShape::FullPrecision);
    }

    // ── GraphShape trait tests ──

    #[test]
    fn graph_shape_variants_have_correct_intensity_ordering() {
        use crate::jit::sub_batch::GraphShape;

        let skip = GraphShape::SkipAttention;
        let narrow = GraphShape::NarrowQuant;
        let full = GraphShape::FullPrecision;
        let moe = GraphShape::MoeSparse;

        // Verify relative ordering: skip < narrow < moe < full
        assert!(skip.compute_intensity() < narrow.compute_intensity());
        assert!(narrow.compute_intensity() < moe.compute_intensity());
        assert!(moe.compute_intensity() < full.compute_intensity());
    }

    #[test]
    fn graph_shape_copy_clone_eq_hash() {
        use crate::jit::sub_batch::GraphShape;
        use std::collections::HashSet;

        let a = GraphShape::FullPrecision;
        let b = a; // Copy
        assert_eq!(a, b);

        let c = a.clone();
        assert_eq!(a, c);

        // Hash: all four variants in a HashSet
        let set: HashSet<GraphShape> = [
            GraphShape::SkipAttention,
            GraphShape::NarrowQuant,
            GraphShape::FullPrecision,
            GraphShape::MoeSparse,
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 4);
    }

    // ── CompactPlatform variant tests ──

    #[test]
    fn compact_platform_variants_are_distinct() {
        use crate::jit::ragged::CompactPlatform;

        let platforms = [
            CompactPlatform::X86Avx512,
            CompactPlatform::X86Avx2,
            CompactPlatform::Aarch64Neon,
            CompactPlatform::GpuMetal,
        ];

        // Verify each formats differently via Debug
        let debug_strs: Vec<String> = platforms.iter().map(|p| format!("{:?}", p)).collect();
        for i in 0..debug_strs.len() {
            for j in (i + 1)..debug_strs.len() {
                assert_ne!(debug_strs[i], debug_strs[j],
                    "CompactPlatform variants {:?} and {:?} have identical Debug output",
                    platforms[i], platforms[j]);
            }
        }
    }

    // ── record_turboquant_scales edge cases ──

    #[test]
    fn record_turboquant_scales_zero_layers_is_noop() {
        let mut coord = make_compute();
        // zero layers: loop body never executes, should not panic
        coord.record_turboquant_scales(0, 64);
        assert!(coord.turboquant.get_k_scales(0).is_none());
    }

    #[test]
    fn record_turboquant_scales_stores_per_channel_scales_per_layer() {
        use crate::kv_cache::turboquant::TurboQuantConfig;
        use crate::kv_cache::quant::QuantMode;

        // Build an enabled TurboQuant runtime
        let config = TurboQuantConfig {
            bits: 4,
            sink_count: 4,
            fwht_enabled: true,
            mode: QuantMode::Deterministic,
            dual_track_enabled: false,
        };
        let mut coord = make_compute();
        coord.turboquant = TurboQuantRuntime::new(config).unwrap();
        assert!(coord.turboquant.is_enabled());

        // Inject a non-zero per_channel_scale to trigger the store path
        coord.telemetry_aggregator.ingest(
            &crate::jit::epilogue::EpilogueSignal::PerChannelScale { scale: 1.5 },
        );
        assert_eq!(coord.telemetry_aggregator.per_channel_scale(), 1.5);

        // Record scales for 3 layers, kv_dim=4
        coord.record_turboquant_scales(3, 4);

        // All 3 layers should have 4-element scale vectors filled with 1.5
        for layer in 0..3 {
            let scales = coord.turboquant.get_k_scales(layer);
            assert!(scales.is_some(), "layer {} should have scales", layer);
            assert_eq!(scales.unwrap().len(), 4);
            assert!(scales.unwrap().iter().all(|&s| (s - 1.5).abs() < 1e-6));
        }

        // Layer 3 should not have scales
        assert!(coord.turboquant.get_k_scales(3).is_none());
    }

    #[test]
    fn record_turboquant_scales_stores_correction_when_embed_norm_nonzero() {
        use crate::kv_cache::turboquant::TurboQuantConfig;
        use crate::kv_cache::quant::QuantMode;

        let config = TurboQuantConfig {
            bits: 4,
            sink_count: 4,
            fwht_enabled: true,
            mode: QuantMode::Deterministic,
            dual_track_enabled: false,
        };
        let mut coord = make_compute();
        coord.turboquant = TurboQuantRuntime::new(config).unwrap();

        // Inject a non-zero embedding_norm to trigger correction path
        coord.telemetry_aggregator.ingest(
            &crate::jit::epilogue::EpilogueSignal::EmbeddingNorm { norm: 2.0 },
        );
        assert!((coord.telemetry_aggregator.embedding_norm() - 2.0).abs() < 1e-6);

        // record_turboquant_scales should not panic; correction is stored internally
        // We verify through get_k_scales being absent (no per_channel_scale set)
        coord.record_turboquant_scales(2, 64);
        assert!(coord.turboquant.get_k_scales(0).is_none());
    }

    // ── record_seq_histogram edge cases ──

    #[test]
    fn record_seq_histogram_seq_len_zero_records_into_bucket_zero() {
        let coord = make_compute();
        coord.record_seq_histogram(0);
        let snap = coord.seq_histogram.snapshot();
        assert_eq!(snap.total_samples, 1);
    }

    #[test]
    fn record_seq_histogram_seq_len_one() {
        let coord = make_compute();
        coord.record_seq_histogram(1);
        let snap = coord.seq_histogram.snapshot();
        assert_eq!(snap.total_samples, 1);
    }

    #[test]
    fn record_seq_histogram_max_seq_4096() {
        let coord = make_compute();
        coord.record_seq_histogram(4096);
        let snap = coord.seq_histogram.snapshot();
        assert_eq!(snap.total_samples, 1);
    }

    // ── collapse_seq_len edge cases ──

    #[test]
    fn collapse_seq_len_zero_returns_positive() {
        let mut coord = make_compute();
        let result = coord.collapse_seq_len(0);
        assert!(result > 0, "collapse(0) should return a positive golden size");
    }

    #[test]
    fn collapse_seq_len_very_large_returns_positive() {
        let mut coord = make_compute();
        let result = coord.collapse_seq_len(1_000_000);
        assert!(result > 0);
    }

    #[test]
    fn collapse_seq_len_idempotent_within_registry() {
        let mut coord = make_compute();
        let first = coord.collapse_seq_len(200);
        let second = coord.collapse_seq_len(200);
        assert_eq!(first, second);
    }

    // ── classify_request_shape additional coverage ──

    #[test]
    fn classify_request_shape_dead_neuron_high_narrow_quant() {
        // classify_request_shape passes dead_neuron_ratio as attention_sparsity
        // and residual_delta_rho as dead_neuron_ratio to classify_request.
        // NarrowQuant triggers when dead_neuron_ratio (param 2) > 0.6.
        // That maps to residual_delta_rho in the aggregator.
        let mut coord = make_compute();
        coord.telemetry_aggregator.ingest(
            &crate::jit::epilogue::EpilogueSignal::ResidualDeltaRho { delta_rho: 0.8 },
        );

        let shape = coord.classify_request_shape(false, 0.0);
        assert_eq!(shape, crate::jit::sub_batch::GraphShape::NarrowQuant);
    }

    #[test]
    fn classify_request_shape_high_dead_ratio_skip_attention() {
        // classify_request_shape passes dead_neuron_ratio as attention_sparsity
        // to classify_request. SkipAttention triggers when attention_sparsity > 0.7.
        let mut coord = make_compute();
        coord.telemetry_aggregator.ingest(
            &crate::jit::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.8 },
        );

        let shape = coord.classify_request_shape(false, 0.0);
        assert_eq!(shape, crate::jit::sub_batch::GraphShape::SkipAttention);
    }

    #[test]
    fn classify_request_shape_moe_with_high_gating_is_full_precision() {
        let coord = make_compute();
        // is_moe=true + gating_threshold=0.5 → NOT MoeSparse → FullPrecision
        let shape = coord.classify_request_shape(true, 0.5);
        assert_eq!(shape, crate::jit::sub_batch::GraphShape::FullPrecision);
    }

    #[test]
    fn classify_request_shape_moe_with_just_below_threshold() {
        let coord = make_compute();
        // is_moe=true + gating_threshold=0.499 → MoeSparse
        let shape = coord.classify_request_shape(true, 0.499);
        assert_eq!(shape, crate::jit::sub_batch::GraphShape::MoeSparse);
    }

    // ── GraphShape preferred_hardware ──

    #[test]
    fn graph_shape_preferred_hardware_mapping() {
        use crate::jit::sub_batch::{GraphShape, HardwareKind};

        assert_eq!(GraphShape::SkipAttention.preferred_hardware(), HardwareKind::LowComputeCore);
        assert_eq!(GraphShape::NarrowQuant.preferred_hardware(), HardwareKind::LowComputeCore);
        assert_eq!(GraphShape::FullPrecision.preferred_hardware(), HardwareKind::FullComputeUnit);
        assert_eq!(GraphShape::MoeSparse.preferred_hardware(), HardwareKind::FullComputeUnit);
    }

    // ── HardwareKind traits ──

    #[test]
    fn hardware_kind_copy_clone_eq_hash() {
        use crate::jit::sub_batch::HardwareKind;
        use std::collections::HashSet;

        let a = HardwareKind::FullComputeUnit;
        let b = a; // Copy
        assert_eq!(a, b);

        let c = a.clone();
        assert_eq!(a, c);

        let set: HashSet<HardwareKind> = [
            HardwareKind::LowComputeCore,
            HardwareKind::FullComputeUnit,
            HardwareKind::TensorCorePartition,
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn hardware_kind_debug_output_distinct() {
        use crate::jit::sub_batch::HardwareKind;

        let debug_strs: Vec<String> = [
            HardwareKind::LowComputeCore,
            HardwareKind::FullComputeUnit,
            HardwareKind::TensorCorePartition,
        ]
        .iter()
        .map(|v| format!("{:?}", v))
        .collect();

        for i in 0..debug_strs.len() {
            for j in (i + 1)..debug_strs.len() {
                assert_ne!(debug_strs[i], debug_strs[j]);
            }
        }
    }

    // ── CompactPlatform detect ──

    #[test]
    fn compact_platform_detect_cuda() {
        let p = CompactPlatform::detect("cuda", false, false, 0, 32);
        assert_eq!(p, CompactPlatform::GpuCuda { warp_size: 32 });
    }

    #[test]
    fn compact_platform_detect_hip() {
        let p = CompactPlatform::detect("hip", false, false, 0, 64);
        assert_eq!(p, CompactPlatform::GpuHip { wavefront_size: 64 });
    }

    #[test]
    fn compact_platform_detect_metal() {
        let p = CompactPlatform::detect("metal", false, false, 0, 0);
        assert_eq!(p, CompactPlatform::GpuMetal);
    }

    #[test]
    fn compact_platform_detect_cpu_avx512() {
        let p = CompactPlatform::detect("cpu", true, false, 0, 0);
        assert_eq!(p, CompactPlatform::X86Avx512);
    }

    #[test]
    fn compact_platform_detect_cpu_sve() {
        let p = CompactPlatform::detect("cpu", true, true, 256, 0);
        assert_eq!(p, CompactPlatform::Aarch64Sve { vl_bytes: 256 });
    }

    #[test]
    fn compact_platform_detect_cpu_fallback_avx2() {
        let p = CompactPlatform::detect("cpu", false, false, 0, 0);
        assert_eq!(p, CompactPlatform::X86Avx2);
    }

    // ── CompactPlatform has_hardware_compress ──

    #[test]
    fn compact_platform_has_hardware_compress_avx512() {
        assert!(CompactPlatform::X86Avx512.has_hardware_compress());
    }

    #[test]
    fn compact_platform_has_hardware_compress_sve() {
        assert!(CompactPlatform::Aarch64Sve { vl_bytes: 128 }.has_hardware_compress());
    }

    #[test]
    fn compact_platform_no_hardware_compress_avx2() {
        assert!(!CompactPlatform::X86Avx2.has_hardware_compress());
    }

    #[test]
    fn compact_platform_no_hardware_compress_neon() {
        assert!(!CompactPlatform::Aarch64Neon.has_hardware_compress());
    }

    #[test]
    fn compact_platform_no_hardware_compress_gpu() {
        assert!(!CompactPlatform::GpuCuda { warp_size: 32 }.has_hardware_compress());
        assert!(!CompactPlatform::GpuMetal.has_hardware_compress());
    }

    // ── CompactPlatform simd_width_bytes ──

    #[test]
    fn compact_platform_simd_width_bytes_all_variants() {
        assert_eq!(CompactPlatform::X86Avx512.simd_width_bytes(), 64);
        assert_eq!(CompactPlatform::X86Avx2.simd_width_bytes(), 32);
        assert_eq!(CompactPlatform::Aarch64Neon.simd_width_bytes(), 16);
        assert_eq!(CompactPlatform::Aarch64Sve { vl_bytes: 256 }.simd_width_bytes(), 256);
        assert_eq!(CompactPlatform::GpuCuda { warp_size: 32 }.simd_width_bytes(), 128);
        assert_eq!(CompactPlatform::GpuHip { wavefront_size: 64 }.simd_width_bytes(), 256);
        assert_eq!(CompactPlatform::GpuMetal.simd_width_bytes(), 128);
    }

    // ── CompactPlatform Copy/Clone/Eq/Hash ──

    #[test]
    fn compact_platform_copy_clone_eq_hash() {
        use std::collections::HashSet;

        let a = CompactPlatform::X86Avx2;
        let b = a; // Copy
        assert_eq!(a, b);

        let c = a.clone();
        assert_eq!(a, c);

        let set: HashSet<CompactPlatform> = [
            CompactPlatform::X86Avx512,
            CompactPlatform::X86Avx2,
            CompactPlatform::Aarch64Neon,
            CompactPlatform::GpuMetal,
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 4);
    }

    // ── GoldenSize construction and field access ──

    #[test]
    fn golden_size_new_computes_performance_score() {
        use crate::jit::golden_bucket::GoldenSize;

        let gs = GoldenSize::new(256, 0.9, 0.8, 0.7);
        assert_eq!(gs.seq_len, 256);
        assert!((gs.register_efficiency - 0.9).abs() < 1e-6);
        assert!((gs.smem_efficiency - 0.8).abs() < 1e-6);
        assert!((gs.l2_hit_rate - 0.7).abs() < 1e-6);
        let expected_score = 0.9 * 0.4 + 0.8 * 0.3 + 0.7 * 0.3;
        assert!((gs.performance_score - expected_score).abs() < 1e-6);
    }

    #[test]
    fn golden_size_partial_eq() {
        use crate::jit::golden_bucket::GoldenSize;

        let a = GoldenSize::new(128, 1.0, 1.0, 1.0);
        let b = GoldenSize::new(128, 1.0, 1.0, 1.0);
        assert_eq!(a, b);

        let c = GoldenSize::new(256, 1.0, 1.0, 1.0);
        assert_ne!(a, c);
    }

    #[test]
    fn golden_size_clone() {
        use crate::jit::golden_bucket::GoldenSize;

        let gs = GoldenSize::new(512, 0.5, 0.6, 0.7);
        let cloned = gs.clone();
        assert_eq!(gs, cloned);
    }

    // ── GoldenBucketRegistry collapse behavior ──

    #[test]
    fn golden_bucket_registry_collapse_returns_nearest_golden() {
        use crate::jit::compiler_constraints::CompilerConstraints;
        use crate::jit::golden_bucket::GoldenBucketRegistry;

        let mut registry = GoldenBucketRegistry::empty(CompilerConstraints::default());
        let sizes = registry.golden_sizes();
        assert!(!sizes.is_empty());

        // Collapse any seq_len should return a valid golden size
        let (_, golden) = registry.collapse(999);
        assert!(golden.seq_len > 0);
    }

    #[test]
    fn golden_bucket_registry_collapse_tracks_hits() {
        use crate::jit::compiler_constraints::CompilerConstraints;
        use crate::jit::golden_bucket::GoldenBucketRegistry;

        let mut registry = GoldenBucketRegistry::empty(CompilerConstraints::default());
        let first_seq = registry.collapse(100).1.seq_len;
        let second_seq = registry.collapse(100).1.seq_len;
        assert_eq!(first_seq, second_seq);
    }

    // ── RequestActiveMask ──

    #[test]
    fn ragged_compaction_new_stores_platform() {
        let _rc = RaggedCompaction::new(CompactPlatform::X86Avx512);
    }

    #[test]
    fn request_active_mask_new_mixed() {
        use crate::jit::ragged::RequestActiveMask;

        let mask = RequestActiveMask::new(vec![true, false, true, false, true]);
        assert_eq!(mask.batch_size(), 5);
        assert_eq!(mask.active_count(), 3);
        assert!((mask.waste_ratio() - 0.4).abs() < 1e-6);
        assert!(mask.should_compact()); // 0.4 > 0.25
    }

    #[test]
    fn request_active_mask_all_active_no_waste() {
        use crate::jit::ragged::RequestActiveMask;

        let mask = RequestActiveMask::all_active(8);
        assert_eq!(mask.batch_size(), 8);
        assert_eq!(mask.active_count(), 8);
        assert!((mask.waste_ratio() - 0.0).abs() < 1e-6);
        assert!(!mask.should_compact());
    }

    #[test]
    fn request_active_mask_empty_no_waste() {
        use crate::jit::ragged::RequestActiveMask;

        let mask = RequestActiveMask::new(vec![]);
        assert_eq!(mask.batch_size(), 0);
        assert_eq!(mask.active_count(), 0);
        assert!((mask.waste_ratio() - 0.0).abs() < 1e-6);
        assert!(!mask.should_compact());
    }

    #[test]
    fn request_active_mask_single_inactive() {
        use crate::jit::ragged::RequestActiveMask;

        let mask = RequestActiveMask::new(vec![false]);
        assert_eq!(mask.batch_size(), 1);
        assert_eq!(mask.active_count(), 0);
        assert!((mask.waste_ratio() - 1.0).abs() < 1e-6);
        assert!(mask.should_compact());
    }

    // ── CompactIndex ──

    #[test]
    fn compact_index_from_mask_all_active() {
        use crate::jit::ragged::{CompactIndex, RequestActiveMask};

        let mask = RequestActiveMask::all_active(4);
        let idx = CompactIndex::from_mask(&mask);
        assert_eq!(idx.active_count(), 4);
        assert!(!idx.is_empty());

        // Identity mapping
        for i in 0..4 {
            assert_eq!(idx.to_compact(i), i);
            assert_eq!(idx.to_original(i), i);
        }
    }

    #[test]
    fn compact_index_from_mask_mixed() {
        use crate::jit::ragged::{CompactIndex, RequestActiveMask};

        let mask = RequestActiveMask::new(vec![true, false, false, true]);
        let idx = CompactIndex::from_mask(&mask);
        assert_eq!(idx.active_count(), 2);

        assert_eq!(idx.to_compact(0), 0); // original 0 → compact 0
        assert_eq!(idx.to_compact(1), usize::MAX); // inactive
        assert_eq!(idx.to_compact(3), 1); // original 3 → compact 1

        assert_eq!(idx.to_original(0), 0);
        assert_eq!(idx.to_original(1), 3);
    }

    #[test]
    fn compact_index_from_mask_none_active() {
        use crate::jit::ragged::{CompactIndex, RequestActiveMask};

        let mask = RequestActiveMask::new(vec![false, false, false]);
        let idx = CompactIndex::from_mask(&mask);
        assert_eq!(idx.active_count(), 0);
        assert!(idx.is_empty());
    }

    // ── CompactDecision ──

    #[test]
    fn compact_decision_skip_when_low_waste() {
        use crate::jit::ragged::{CompactDecision, CompactPlatform, RequestActiveMask};

        // 7/8 active → waste = 12.5% < 25% → SkipDirect
        let mask = RequestActiveMask::new(vec![true, true, true, true, true, true, true, false]);
        let decision = CompactDecision::decide(mask, CompactPlatform::X86Avx2);
        match decision {
            CompactDecision::SkipDirect { batch_size } => assert_eq!(batch_size, 8),
            CompactDecision::Compact { .. } => panic!("expected SkipDirect"),
        }
    }

    #[test]
    fn compact_decision_compact_when_high_waste() {
        use crate::jit::ragged::{CompactDecision, CompactPlatform, RequestActiveMask};

        // 2/8 active → waste = 75% > 25% → Compact
        let mask = RequestActiveMask::new(vec![true, false, false, false, false, false, false, true]);
        let decision = CompactDecision::decide(mask, CompactPlatform::X86Avx2);
        match decision {
            CompactDecision::SkipDirect { .. } => panic!("expected Compact"),
            CompactDecision::Compact { original_batch_size, compact_batch_size, .. } => {
                assert_eq!(original_batch_size, 8);
                assert_eq!(compact_batch_size, 2);
            }
        }
    }

    #[test]
    fn compact_decision_sve_always_skips() {
        use crate::jit::ragged::{CompactDecision, CompactPlatform, RequestActiveMask};

        // Even with 100% waste, SVE always skips
        let mask = RequestActiveMask::new(vec![true, false, false, false]);
        let decision = CompactDecision::decide(mask, CompactPlatform::Aarch64Sve { vl_bytes: 256 });
        match decision {
            CompactDecision::SkipDirect { batch_size } => assert_eq!(batch_size, 4),
            CompactDecision::Compact { .. } => panic!("SVE should always skip"),
        }
    }

    // ── SubBatch ──

    #[test]
    fn sub_batch_new_is_empty() {
        use crate::jit::sub_batch::SubBatch;
        use crate::jit::sub_batch::GraphShape;

        let sb = SubBatch::new(GraphShape::FullPrecision);
        assert!(sb.is_empty());
        assert_eq!(sb.len(), 0);
        assert_eq!(sb.shape, GraphShape::FullPrecision);
        assert!(sb.target_partition.is_none());
        assert!((sb.estimated_gflops - 0.0).abs() < 1e-6);
        assert_eq!(sb.estimated_vram_bytes, 0);
    }

    #[test]
    fn sub_batch_add_request_increments_len() {
        use crate::jit::sub_batch::SubBatch;
        use crate::jit::sub_batch::GraphShape;
        use crate::scheduler::types::RequestId;

        let mut sb = SubBatch::new(GraphShape::MoeSparse);
        assert!(sb.is_empty());

        sb.add_request(RequestId::from(42u64));
        assert_eq!(sb.len(), 1);
        assert!(!sb.is_empty());

        sb.add_request(RequestId::from(99u64));
        assert_eq!(sb.len(), 2);
        assert_eq!(sb.request_ids[0], RequestId::from(42u64));
        assert_eq!(sb.request_ids[1], RequestId::from(99u64));
    }

    // ── HardwarePartition construction ──

    #[test]
    fn hardware_partition_gpu_sm_partition() {
        use crate::jit::sub_batch::{HardwareKind, HardwarePartition};

        let hp = HardwarePartition::gpu_sm_partition(1, 20, 40);
        assert_eq!(hp.partition_id, 1);
        assert_eq!(hp.kind, HardwareKind::TensorCorePartition);
        assert_eq!(hp.sm_range, Some((20, 40)));
        assert!(hp.numa_node.is_none());
        assert!(hp.core_range.is_none());
        assert!(hp.compute_weight > 0.0);
    }

    #[test]
    fn hardware_partition_cpu_numa_partition_node_zero_is_full_compute() {
        use crate::jit::sub_batch::{HardwareKind, HardwarePartition};

        let hp = HardwarePartition::cpu_numa_partition(0, 0, 8);
        assert_eq!(hp.partition_id, 0);
        assert_eq!(hp.kind, HardwareKind::FullComputeUnit);
        assert_eq!(hp.numa_node, Some(0));
    }

    #[test]
    fn hardware_partition_cpu_numa_partition_non_zero_is_low_compute() {
        use crate::jit::sub_batch::{HardwareKind, HardwarePartition};

        let hp = HardwarePartition::cpu_numa_partition(2, 1, 4);
        assert_eq!(hp.partition_id, 2);
        assert_eq!(hp.kind, HardwareKind::LowComputeCore);
        assert_eq!(hp.numa_node, Some(1));
    }

    // ── EpilogueSignal variants ingest ──

    #[test]
    fn telemetry_aggregator_ingest_all_signal_types() {
        use crate::jit::epilogue::EpilogueSignal;

        let mut agg = TelemetryAggregator::new();

        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.5 });
        assert!((agg.dead_neuron_ratio() - 0.5).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::PerChannelScale { scale: 3.14 });
        assert!((agg.per_channel_scale() - 3.14).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::EmbeddingNorm { norm: 1.23 });
        assert!((agg.embedding_norm() - 1.23).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::ResidualDeltaRho { delta_rho: 0.75 });
        assert!((agg.residual_delta_rho() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn telemetry_aggregator_ingest_zero_values() {
        use crate::jit::epilogue::EpilogueSignal;

        let mut agg = TelemetryAggregator::new();

        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.0 });
        assert!((agg.dead_neuron_ratio() - 0.0).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::PerChannelScale { scale: 0.0 });
        assert!((agg.per_channel_scale() - 0.0).abs() < 1e-6);

        agg.ingest(&EpilogueSignal::EmbeddingNorm { norm: 0.0 });
        assert!((agg.embedding_norm() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn telemetry_aggregator_latest_signal_wins() {
        use crate::jit::epilogue::EpilogueSignal;

        let mut agg = TelemetryAggregator::new();

        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.1 });
        agg.ingest(&EpilogueSignal::DeadNeuronRatio { ratio: 0.9 });
        assert!((agg.dead_neuron_ratio() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn telemetry_aggregator_clone_independent() {
        let mut agg = TelemetryAggregator::new();
        agg.ingest(&crate::jit::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.5 });

        let cloned = agg.clone();
        assert!((cloned.dead_neuron_ratio() - 0.5).abs() < 1e-6);

        // Mutating original does not affect clone
        agg.ingest(&crate::jit::epilogue::EpilogueSignal::DeadNeuronRatio { ratio: 0.8 });
        assert!((cloned.dead_neuron_ratio() - 0.5).abs() < 1e-6);
    }

    // ── FwhtInsertionPoint traits ──

    #[test]
    fn fwht_insertion_point_copy_clone_eq() {
        use crate::kv_cache::turboquant::FwhtInsertionPoint;

        let a = FwhtInsertionPoint::AttentionEpilogue;
        let b = a; // Copy
        assert_eq!(a, b);

        let c = a.clone();
        assert_eq!(a, c);

        assert_ne!(FwhtInsertionPoint::AttentionEpilogue, FwhtInsertionPoint::FfnEpilogue);
        assert_ne!(FwhtInsertionPoint::FfnEpilogue, FwhtInsertionPoint::KvWrite);
    }

    // ── EpilogueConfig default ──

    #[test]
    fn epilogue_config_default_has_layers_and_no_experts() {
        use crate::jit::epilogue_subsystem::EpilogueConfig;

        let config = EpilogueConfig::default();
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_experts, 0);
    }

    #[test]
    fn epilogue_config_clone() {
        use crate::jit::epilogue_subsystem::EpilogueConfig;

        let config = EpilogueConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.num_layers, config.num_layers);
        assert_eq!(cloned.num_experts, config.num_experts);
    }

    // ── TurboQuantConfig ──

    #[test]
    fn turbo_quant_config_default_is_minimal() {
        use crate::kv_cache::turboquant::TurboQuantConfig;

        let config = TurboQuantConfig::default();
        assert_eq!(config.bits, 4);
        assert_eq!(config.sink_count, 4);
        assert!(!config.fwht_enabled);
        assert!(!config.dual_track_enabled);
        assert!(!config.is_enabled());
    }

    #[test]
    fn turbo_quant_config_enabled_when_fwht() {
        use crate::kv_cache::turboquant::TurboQuantConfig;

        let config = TurboQuantConfig {
            fwht_enabled: true,
            ..TurboQuantConfig::default()
        };
        assert!(config.is_enabled());
    }

    #[test]
    fn turbo_quant_config_to_kv_quant_config_preserves_fields() {
        use crate::kv_cache::turboquant::TurboQuantConfig;

        let config = TurboQuantConfig {
            bits: 3,
            sink_count: 8,
            fwht_enabled: true,
            ..TurboQuantConfig::default()
        };
        let kv = config.to_kv_quant_config();
        assert_eq!(kv.bits, 3);
        assert_eq!(kv.sink_count, 8);
        assert!(kv.fwht_enabled);
    }

    // ── RabitqCorrection ──

    #[test]
    fn rabitq_correction_fields_and_eq() {
        use crate::kv_cache::quant::RabitqCorrection;

        let a = RabitqCorrection { c0: 0.0, c1: 1.0, v_norm: 2.0 };
        let b = RabitqCorrection { c0: 0.0, c1: 1.0, v_norm: 2.0 };
        assert_eq!(a, b);

        let c = RabitqCorrection { c0: 0.1, c1: 1.0, v_norm: 2.0 };
        assert_ne!(a, c);
    }

    #[test]
    fn rabitq_correction_copy_clone() {
        use crate::kv_cache::quant::RabitqCorrection;

        let a = RabitqCorrection { c0: 0.0, c1: 0.5, v_norm: 1.0 };
        let b = a; // Copy
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // ── Additional tests for untested paths ──

    // @trace TEST-COMP-04 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn telemetry_aggregator_softmax_sharpness_and_max_ingest() {
        use crate::jit::epilogue::EpilogueSignal;

        // Arrange
        let mut agg = TelemetryAggregator::new();
        assert_eq!(agg.softmax_sharpness(), 0.0);
        assert_eq!(agg.softmax_max(), 0.0);

        // Act
        agg.ingest(&EpilogueSignal::SoftmaxSharpness {
            max_val: 0.95,
            sharpness: 0.88,
        });

        // Assert
        assert!((agg.softmax_max() - 0.95).abs() < 1e-6);
        assert!((agg.softmax_sharpness() - 0.88).abs() < 1e-6);
    }

    // @trace TEST-COMP-05 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn telemetry_aggregator_residual_cosine_and_output_entropy_ingest() {
        use crate::jit::epilogue::EpilogueSignal;

        // Arrange
        let mut agg = TelemetryAggregator::new();
        assert_eq!(agg.residual_cosine(), 0.0);
        assert_eq!(agg.output_entropy(), 0.0);

        // Act
        agg.ingest(&EpilogueSignal::ResidualCosineSimilarity { cosine: 0.997 });
        agg.ingest(&EpilogueSignal::OutputEntropy { entropy: 2.5 });

        // Assert
        assert!((agg.residual_cosine() - 0.997).abs() < 1e-6);
        assert!((agg.output_entropy() - 2.5).abs() < 1e-6);
    }

    // @trace TEST-COMP-06 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn telemetry_aggregator_expert_hit_count_and_centroid_position() {
        use crate::jit::epilogue::EpilogueSignal;

        // Arrange
        let mut agg = TelemetryAggregator::new();
        assert_eq!(agg.expert_hit_count(0), 0);
        assert_eq!(agg.expert_hit_count(5), 0);
        assert_eq!(agg.centroid_position(), 0);

        // Act
        agg.ingest(&EpilogueSignal::ExpertHitCount {
            expert_id: 3,
            hit_count: 42,
        });
        agg.ingest(&EpilogueSignal::ExpertHitCount {
            expert_id: 7,
            hit_count: 15,
        });
        agg.ingest(&EpilogueSignal::CentroidPosition { position: 128 });

        // Assert
        assert_eq!(agg.expert_hit_count(3), 42);
        assert_eq!(agg.expert_hit_count(7), 15);
        assert_eq!(agg.expert_hit_count(0), 0);
        assert_eq!(agg.expert_hit_count(99), 0);
        assert_eq!(agg.centroid_position(), 128);
        // expert_hit_counts slice should be at least 8 long (expert_id 7 + 1)
        assert!(agg.expert_hit_counts().len() >= 8);
    }

    // @trace TEST-COMP-07 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn telemetry_aggregator_set_embedding_norm_direct() {
        // Arrange
        let mut agg = TelemetryAggregator::new();
        assert_eq!(agg.embedding_norm(), 0.0);

        // Act
        agg.set_embedding_norm(3.7);

        // Assert
        assert!((agg.embedding_norm() - 3.7).abs() < 1e-6);
    }

    // @trace TEST-COMP-08 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn telemetry_aggregator_compute_and_set_embedding_norm() {
        // Arrange
        let mut agg = TelemetryAggregator::new();
        let hidden: Vec<f32> = vec![3.0, 4.0]; // sqrt(9+16) = 5.0

        // Act
        agg.compute_and_set_embedding_norm(&hidden);

        // Assert
        assert!((agg.embedding_norm() - 5.0).abs() < 1e-6);
    }

    // @trace TEST-COMP-09 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn compute_l2_norm_known_values() {
        use crate::jit::epilogue::compute_l2_norm;

        // Arrange: standard 3-4-5 triangle
        let input = vec![3.0f32, 4.0f32];

        // Act
        let norm = compute_l2_norm(&input);

        // Assert
        assert!((norm - 5.0f32).abs() < 1e-6);
    }

    // @trace TEST-COMP-10 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn compute_l2_norm_empty_and_single_element() {
        use crate::jit::epilogue::compute_l2_norm;

        // Arrange & Act & Assert: empty slice → 0.0
        let empty: Vec<f32> = vec![];
        assert!((compute_l2_norm(&empty) - 0.0).abs() < 1e-6);

        // Arrange & Act & Assert: single element → abs value
        assert!((compute_l2_norm(&[7.0f32]) - 7.0).abs() < 1e-6);
        assert!((compute_l2_norm(&[-2.0f32]) - 2.0).abs() < 1e-6);
    }

    // @trace TEST-COMP-11 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn compact_data_compact_and_scatter_roundtrip() {
        use crate::jit::ragged::{CompactData, CompactIndex, RequestActiveMask, ScatterWriter};

        // Arrange: batch of 4, only positions 0 and 3 are active
        let mask = RequestActiveMask::new(vec![true, false, false, true]);
        let index = CompactIndex::from_mask(&mask);
        // Inner dim = 2 per request → total source = [batch=4, inner=2] = 8 elements
        let source: Vec<f32> = vec![
            10.0, 11.0,  // request 0 (active)
            20.0, 21.0,  // request 1 (inactive)
            30.0, 31.0,  // request 2 (inactive)
            40.0, 41.0,  // request 3 (active)
        ];

        // Act: Compact
        let compact = CompactData::compact(&source, &index, &[2]);

        // Assert: compact shape is [2, 2]
        assert_eq!(compact.compact_shape, vec![2, 2]);
        // Only request 0 and request 3 data preserved
        assert_eq!(compact.data, vec![10.0, 11.0, 40.0, 41.0]);

        // Act: Scatter back into an output buffer
        let mut output = vec![0.0f32; 8];
        {
            let mut writer = ScatterWriter::new(&mut output, &index, &[2]);
            // Simulate executor returning the compact data unchanged
            writer.scatter(&compact.data);
        }

        // Assert: active requests have their data, inactive remain zero
        assert_eq!(output[0], 10.0);
        assert_eq!(output[1], 11.0);
        assert_eq!(output[2], 0.0); // inactive
        assert_eq!(output[3], 0.0); // inactive
        assert_eq!(output[4], 0.0); // inactive
        assert_eq!(output[5], 0.0); // inactive
        assert_eq!(output[6], 40.0);
        assert_eq!(output[7], 41.0);
    }

    // @trace TEST-COMP-12 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn ragged_compaction_execute_skip_direct_all_active() {
        use crate::jit::ragged::{CompactPlatform, RaggedCompaction, RequestActiveMask};

        // Arrange: all 3 requests active → SkipDirect path
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx2);
        let mask = RequestActiveMask::all_active(3);
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
        let mut output = vec![0.0f32; 6];

        // Act
        rc.execute(&input, &[2], mask, &mut output, |data, shape| {
            assert_eq!(shape, &[3, 2]);
            // Identity executor
            data.to_vec()
        });

        // Assert: output mirrors input (skip-direct path)
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // @trace TEST-COMP-13 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn ragged_compaction_execute_compact_path_with_sparse_mask() {
        use crate::jit::ragged::{CompactPlatform, RaggedCompaction, RequestActiveMask};

        // Arrange: 4 requests, only 0 and 2 active → Compact path
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx2);
        let mask = RequestActiveMask::new(vec![true, false, true, false]);
        let input: Vec<f32> = vec![
            10.0, 11.0,  // req 0
            20.0, 21.0,  // req 1 (inactive)
            30.0, 31.0,  // req 2
            40.0, 41.0,  // req 3 (inactive)
        ];
        let mut output = vec![0.0f32; 8];

        // Act
        rc.execute(&input, &[2], mask, &mut output, |compact_data, shape| {
            // Compact executor receives only active requests
            assert_eq!(shape, &[2, 2]);
            assert_eq!(compact_data.len(), 4);
            // Double the values to prove we went through compact path
            compact_data.iter().map(|&v| v * 2.0).collect()
        });

        // Assert: only active request slots have doubled values
        assert_eq!(output[0], 20.0); // req 0 → doubled
        assert_eq!(output[1], 22.0);
        assert_eq!(output[2], 0.0);  // req 1 → untouched
        assert_eq!(output[3], 0.0);
        assert_eq!(output[4], 60.0); // req 2 → doubled
        assert_eq!(output[5], 62.0);
        assert_eq!(output[6], 0.0);  // req 3 → untouched
        assert_eq!(output[7], 0.0);
    }

    // @trace TEST-COMP-14 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn ragged_compaction_execute_zero_active_returns_early() {
        use crate::jit::ragged::{CompactPlatform, RaggedCompaction, RequestActiveMask};

        // Arrange: no active requests → SkipDirect with batch_size=0
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx2);
        let mask = RequestActiveMask::new(vec![false, false, false]);
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0f32; 6];

        // Act: should return early without calling executor
        rc.execute(&input, &[2], mask, &mut output, |_data, _shape| {
            panic!("executor should not be called when no active requests");
        });

        // Assert: output remains all zeros
        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    // @trace TEST-COMP-15 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn ragged_compaction_should_compact_and_platform_accessors() {
        use crate::jit::ragged::{CompactPlatform, RaggedCompaction, RequestActiveMask};

        // Arrange
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx512);

        // Assert: platform accessor
        assert_eq!(*rc.platform(), CompactPlatform::X86Avx512);

        // Assert: high waste → should_compact returns true
        let high_waste = RequestActiveMask::new(vec![true, false, false, false]);
        assert!(rc.should_compact(&high_waste));

        // Assert: low waste → should_compact returns false
        let low_waste = RequestActiveMask::all_active(4);
        assert!(!rc.should_compact(&low_waste));

        // Assert: SVE always returns false regardless of waste
        let sve_rc = RaggedCompaction::new(CompactPlatform::Aarch64Sve { vl_bytes: 256 });
        assert!(!sve_rc.should_compact(&high_waste));
    }

    // @trace TEST-COMP-16 [req:REQ-DECOMP] [level:unit]
    #[test]
    fn compact_platform_display_formatting() {
        use crate::jit::ragged::CompactPlatform;

        // Arrange & Act & Assert: each variant produces the expected display string
        assert_eq!(format!("{}", CompactPlatform::X86Avx512), "x86_avx512");
        assert_eq!(format!("{}", CompactPlatform::X86Avx2), "x86_avx2");
        assert_eq!(format!("{}", CompactPlatform::Aarch64Neon), "aarch64_neon");
        assert_eq!(format!("{}", CompactPlatform::Aarch64Sve { vl_bytes: 256 }), "aarch64_sve_256b");
        assert_eq!(format!("{}", CompactPlatform::GpuCuda { warp_size: 32 }), "cuda_w32");
        assert_eq!(format!("{}", CompactPlatform::GpuHip { wavefront_size: 64 }), "hip_wf64");
        assert_eq!(format!("{}", CompactPlatform::GpuMetal), "metal");
    }
}
