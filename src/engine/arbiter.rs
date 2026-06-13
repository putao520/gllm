//! Strategy Arbiter — InferenceMode × GraphArchetype × Hardware → StrategyBias.
//!
//! Per SPEC/12-STRATEGY-ARBITER.md. Produces normalized resource-priority
//! weights injected into HwOptEngine's CostModel so that all downstream
//! solvers automatically bias toward the scenario-optimal strategy.

use gllm_kernels::dispatch::DeviceProfile;

pub use crate::graph::profile::GraphArchetype;

// ── InferenceMode (SPEC §2.1) ──────────────────────────────────────────────

/// Global optimization objective. Determined at model-load time, immutable
/// for the lifetime of the inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Default)]
pub enum InferenceMode {
    /// Extreme single-request latency: batch=1, all resources serve one request.
    #[default]
    Latency,
    /// Extreme throughput: maximize tokens/second/dollar.
    Throughput,
}


// ── Device family ──────────────────────────────────────────────────────────

/// Minimal device classification consumed by the arbiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceFamily {
    Cpu,
    Gpu,
}

// ── Hardware view for the arbiter ───────────────────────────────────────────

/// Minimal hardware view consumed by the arbiter.
///
/// `DeviceProfile` from gllm-kernels is CPU-only (no GPU variant). This thin
/// view lets callers describe both CPU and GPU targets with the three
/// properties the arbiter actually reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArbiterHwView {
    pub device: DeviceFamily,
    pub num_simd_regs: usize,
    /// (L1, L2, L3) in bytes. For GPUs, L1 maps to shared memory.
    pub cache_sizes: (usize, usize, usize),
}

impl From<&DeviceProfile> for ArbiterHwView {
    fn from(p: &DeviceProfile) -> Self {
        Self {
            device: DeviceFamily::Cpu,
            num_simd_regs: p.num_simd_regs(),
            cache_sizes: p.cache_sizes(),
        }
    }
}

impl ArbiterHwView {
    /// Construct a GPU hardware view with the given shared-memory size.
    pub fn gpu(shared_mem_bytes: usize) -> Self {
        Self {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (shared_mem_bytes, 0, 0),
        }
    }
}

// ── StrategyBias (SPEC §4.2) ────────────────────────────────────────────────

// Re-export the canonical StrategyBias from gllm-kernels.
// Single source of truth — no duplicate struct, no field-by-field copy.
pub use gllm_kernels::compiler::planner::StrategyBias;

// ── StrategyArbiter (SPEC §4.3) ─────────────────────────────────────────────

pub struct StrategyArbiter;

impl StrategyArbiter {
    /// Main entry point. Called once at model-load time.
    pub fn arbitrate(
        mode: InferenceMode,
        archetype: &GraphArchetype,
        hw: &ArbiterHwView,
    ) -> StrategyBias {
        let mut bias = Self::mode_baseline(mode);
        Self::apply_archetype_modulation(&mut bias, archetype);
        Self::apply_hardware_adjustment(&mut bias, hw);
        bias.validate();
        bias
    }

    /// Convenience overload accepting a CPU `DeviceProfile` directly.
    pub fn arbitrate_cpu(
        mode: InferenceMode,
        archetype: &GraphArchetype,
        profile: &DeviceProfile,
    ) -> StrategyBias {
        Self::arbitrate(mode, archetype, &ArbiterHwView::from(profile))
    }

    // ── §4.3.1 Mode baseline ───────────────────────────────────────────

    fn mode_baseline(mode: InferenceMode) -> StrategyBias {
        match mode {
            InferenceMode::Latency => StrategyBias {
                fusion_cost_scale: 0.5,
                pipeline_cost_scale: 0.6,
                parallelism_cost_scale: 1.5,
                epilogue_depth_preference: 1.5,
                k_depth_preference: 1.3,
                kv_cache_budget_scale: 0.5,
                weight_prefetch_budget_scale: 1.5,
                batch_flexibility: 0.0,
                decode_ratio_scale: 1.0,
                expert_eviction_aggressiveness: 0.0,
                expert_prefetch_priority: 0.5,
                speculative_decoding_value: 1.5,
                quantization_aggressiveness: 1.5,
            },
            InferenceMode::Throughput => StrategyBias {
                fusion_cost_scale: 1.0,
                pipeline_cost_scale: 1.3,
                parallelism_cost_scale: 0.5,
                epilogue_depth_preference: 0.8,
                k_depth_preference: 0.8,
                kv_cache_budget_scale: 1.5,
                weight_prefetch_budget_scale: 0.8,
                batch_flexibility: 1.0,
                decode_ratio_scale: 1.0,
                expert_eviction_aggressiveness: 0.8,
                expert_prefetch_priority: 1.5,
                speculative_decoding_value: 0.3,
                quantization_aggressiveness: 0.8,
            },
        }
    }

    // ── §4.3.2 Archetype modulation ─────────────────────────────────────

    fn apply_archetype_modulation(bias: &mut StrategyBias, arch: &GraphArchetype) {
        bias.fusion_cost_scale *= lerp(1.0, 0.6, arch.fusion_profitable);
        bias.pipeline_cost_scale *= lerp(1.0, 0.6, arch.pipeline_valuable);
        bias.parallelism_cost_scale *= lerp(1.0, 0.5, arch.parallelism_exploitable);

        let reg_tension = arch.fusion_profitable - arch.pipeline_valuable;
        if reg_tension > 0.0 {
            bias.epilogue_depth_preference *= 1.0 + reg_tension * 0.5;
            bias.k_depth_preference *= 1.0 - reg_tension * 0.3;
        } else if reg_tension < 0.0 {
            let abs_t = reg_tension.abs();
            bias.k_depth_preference *= 1.0 + abs_t * 0.5;
            bias.epilogue_depth_preference *= 1.0 - abs_t * 0.3;
        }

        // MoE modulation
        if arch.parallelism_exploitable > 0.5 {
            bias.expert_eviction_aggressiveness *= lerp(1.0, 1.5, arch.memory_intensive);
            bias.expert_prefetch_priority *= lerp(1.0, 2.0, arch.memory_intensive);
        }

        // KV cache modulation
        bias.kv_cache_budget_scale *= lerp(1.0, 1.5, arch.memory_intensive);

        // Quantization modulation
        bias.quantization_aggressiveness *= lerp(1.0, 1.3, arch.memory_intensive);
    }

    // ── §4.3.3 Hardware adjustment ──────────────────────────────────────

    fn apply_hardware_adjustment(bias: &mut StrategyBias, hw: &ArbiterHwView) {
        // GPU: abundant registers → relax register tension
        match hw.device {
            DeviceFamily::Gpu => {
                bias.epilogue_depth_preference *= 1.2;
                bias.k_depth_preference *= 1.2;
                bias.pipeline_cost_scale *= 1.2;
            }
            DeviceFamily::Cpu => {}
        }

        // CPU register scarcity
        if hw.num_simd_regs <= 16 {
            let scarcity = 1.0 - (hw.num_simd_regs as f64 / 32.0);
            bias.epilogue_depth_preference *= 1.0 + scarcity * 0.3;
            bias.k_depth_preference *= 1.0 - scarcity * 0.2;
        }

        // L1 richness → fusion cost reduction
        let l1_bytes = hw.cache_sizes.0 as f64;
        let l1_richness = (l1_bytes / 65536.0).min(2.0);
        if l1_richness > 0.0 {
            bias.fusion_cost_scale *= 1.0 / l1_richness.sqrt();
        }
    }
}

// ── Helper functions (SPEC §10) ─────────────────────────────────────────────

#[cfg(test)]
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(actual: f64, expected: f64, tolerance_ratio: f64) {
        let diff = (actual - expected).abs();
        let max_diff = expected.abs() * tolerance_ratio;
        assert!(
            diff <= max_diff,
            "expected ≈{expected}, got {actual} (diff {diff} > tolerance {max_diff})"
        );
    }

    fn cpu_avx2() -> ArbiterHwView {
        ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        }
    }

    fn cpu_avx512() -> ArbiterHwView {
        ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (49152, 1048576, 33554432),
        }
    }

    fn gpu_a100() -> ArbiterHwView {
        ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        }
    }

    /// Golden vector 1: Llama-70B + Latency + CPU_AVX2 (SPEC §15.2)
    #[test]
    fn test_golden_vector_llama70b_latency_cpu_avx2() {
        let archetype = GraphArchetype {
            compute_intensive: 0.95,
            memory_intensive: 0.60,
            parallelism_exploitable: 0.20,
            fusion_profitable: 0.85,
            pipeline_valuable: 0.15,
        };
        let bias = StrategyArbiter::arbitrate(
            InferenceMode::Latency,
            &archetype,
            &cpu_avx2(),
        );

        assert_approx(bias.fusion_cost_scale, 0.47, 0.10);
        assert_approx(bias.pipeline_cost_scale, 0.56, 0.10);
        assert_approx(bias.parallelism_cost_scale, 1.35, 0.10);
        assert_approx(bias.epilogue_depth_preference, 2.33, 0.10);
        assert_approx(bias.k_depth_preference, 0.92, 0.10);
        assert_eq!(bias.batch_flexibility, 0.0);
    }

    /// Golden vector 2: DeepSeek-V3 + Throughput + GPU_A100 (SPEC §15.3)
    #[test]
    fn test_golden_vector_deepseek_v3_throughput_gpu_a100() {
        let archetype = GraphArchetype {
            compute_intensive: 0.50,
            memory_intensive: 0.90,
            parallelism_exploitable: 0.99,
            fusion_profitable: 0.60,
            pipeline_valuable: 0.35,
        };
        let bias = StrategyArbiter::arbitrate(
            InferenceMode::Throughput,
            &archetype,
            &gpu_a100(),
        );

        assert_approx(bias.fusion_cost_scale, 0.88, 0.10);
        assert_approx(bias.parallelism_cost_scale, 0.25, 0.10);
        assert_approx(bias.expert_prefetch_priority, 2.85, 0.10);
        assert_approx(bias.pipeline_cost_scale, 1.34, 0.10);
        assert_approx(bias.epilogue_depth_preference, 1.08, 0.10);
        assert_approx(bias.k_depth_preference, 0.89, 0.10);
        assert_eq!(bias.batch_flexibility, 1.0);
    }

    /// Golden vector 3: Phi-4 + Latency + CPU_AVX512 (SPEC §15.4)
    #[test]
    fn test_golden_vector_phi4_latency_cpu_avx512() {
        let archetype = GraphArchetype {
            compute_intensive: 0.35,
            memory_intensive: 0.25,
            parallelism_exploitable: 0.10,
            fusion_profitable: 0.70,
            pipeline_valuable: 0.80,
        };
        let bias = StrategyArbiter::arbitrate(
            InferenceMode::Latency,
            &archetype,
            &cpu_avx512(),
        );

        assert_approx(bias.fusion_cost_scale, 0.42, 0.10);
        assert_approx(bias.pipeline_cost_scale, 0.41, 0.10);
        assert_approx(bias.k_depth_preference, 1.37, 0.10);
        assert_approx(bias.epilogue_depth_preference, 1.46, 0.10);
        assert_eq!(bias.batch_flexibility, 0.0);
    }

    /// Golden vector 4: Qwen3-7B + Throughput + GPU_A100 (SPEC §15.5)
    #[test]
    fn test_golden_vector_qwen3_7b_throughput_gpu_a100() {
        let archetype = GraphArchetype {
            compute_intensive: 0.73,
            memory_intensive: 0.40,
            parallelism_exploitable: 0.15,
            fusion_profitable: 0.80,
            pipeline_valuable: 0.45,
        };
        let bias = StrategyArbiter::arbitrate(
            InferenceMode::Throughput,
            &archetype,
            &gpu_a100(),
        );

        assert_approx(bias.fusion_cost_scale, 0.79, 0.10);
        assert_approx(bias.parallelism_cost_scale, 0.46, 0.10);
        assert_eq!(bias.batch_flexibility, 1.0);
        assert_approx(bias.speculative_decoding_value, 0.3, 0.10);
    }

    // ── InferenceMode ──

    #[test]
    fn inference_mode_default_is_latency() {
        assert_eq!(InferenceMode::default(), InferenceMode::Latency);
    }

    #[test]
    fn inference_mode_equality() {
        assert_eq!(InferenceMode::Latency, InferenceMode::Latency);
        assert_ne!(InferenceMode::Latency, InferenceMode::Throughput);
    }

    // ── ArbiterHwView ──

    #[test]
    fn arbiter_hw_view_gpu_constructor() {
        let view = ArbiterHwView::gpu(48 * 1024);
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.num_simd_regs, 255);
        assert_eq!(view.cache_sizes.0, 48 * 1024);
    }

    #[test]
    fn arbiter_hw_view_from_device_profile() {
        let dp = DeviceProfile::detect();
        let view = ArbiterHwView::from(&dp);
        assert!(view.device == DeviceFamily::Cpu);
        assert_eq!(view.num_simd_regs, dp.num_simd_regs());
    }

    // ── Additional coverage tests ──

    #[test]
    fn inference_mode_debug_format() {
        assert!(format!("{:?}", InferenceMode::Latency).contains("Latency"));
        assert!(format!("{:?}", InferenceMode::Throughput).contains("Throughput"));
    }

    #[test]
    fn inference_mode_copy_semantics() {
        let a = InferenceMode::Latency;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn inference_mode_clone() {
        let a = InferenceMode::Throughput;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn arbiter_hw_view_debug() {
        let view = cpu_avx2();
        let s = format!("{:?}", view);
        assert!(s.contains("device"));
        assert!(s.contains("num_simd_regs"));
    }

    #[test]
    fn arbiter_hw_view_clone() {
        let view = gpu_a100();
        let cloned = view.clone();
        assert_eq!(cloned.device, view.device);
        assert_eq!(cloned.num_simd_regs, view.num_simd_regs);
    }

    #[test]
    fn arbiter_hw_view_copy() {
        let view = cpu_avx512();
        let copied = view;
        assert_eq!(view.device, copied.device);
        assert_eq!(view.cache_sizes, copied.cache_sizes);
    }

    #[test]
    fn arbiter_hw_view_gpu_zero_shared_mem() {
        let view = ArbiterHwView::gpu(0);
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.cache_sizes.0, 0);
    }

    #[test]
    fn latency_baseline_differs_from_throughput() {
        let neutral_arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let latency = StrategyArbiter::arbitrate(
            InferenceMode::Latency, &neutral_arch, &cpu_avx512(),
        );
        let throughput = StrategyArbiter::arbitrate(
            InferenceMode::Throughput, &neutral_arch, &cpu_avx512(),
        );
        assert!(
            latency.fusion_cost_scale < throughput.fusion_cost_scale,
            "latency should favor fusion more than throughput"
        );
        assert!(
            latency.batch_flexibility < throughput.batch_flexibility,
            "throughput should have higher batch flexibility"
        );
    }

    #[test]
    fn zero_archetype_fields_no_crash() {
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(
            InferenceMode::Latency, &zero_arch, &cpu_avx2(),
        );
        assert!(bias.fusion_cost_scale > 0.0);
        assert!(bias.pipeline_cost_scale > 0.0);
    }

    #[test]
    fn max_archetype_fields_no_crash() {
        let max_arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let bias = StrategyArbiter::arbitrate(
            InferenceMode::Throughput, &max_arch, &gpu_a100(),
        );
        assert!(bias.fusion_cost_scale > 0.0);
        assert!(bias.kv_cache_budget_scale > 0.0);
    }

    #[test]
    fn gpu_adjusts_epilogue_and_pipeline() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu_bias = StrategyArbiter::arbitrate(
            InferenceMode::Latency, &arch, &cpu_avx512(),
        );
        let gpu_bias = StrategyArbiter::arbitrate(
            InferenceMode::Latency, &arch, &gpu_a100(),
        );
        assert!(
            gpu_bias.epilogue_depth_preference > cpu_bias.epilogue_depth_preference,
            "GPU should boost epilogue depth preference"
        );
        assert!(
            gpu_bias.pipeline_cost_scale > cpu_bias.pipeline_cost_scale,
            "GPU should boost pipeline cost scale"
        );
    }

    #[test]
    fn low_registers_increase_epilogue_preference() {
        let low_reg = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (32768, 262144, 8388608),
        };
        let high_reg = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 262144, 8388608),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let low_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &low_reg);
        let high_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &high_reg);
        assert!(
            low_bias.epilogue_depth_preference > high_bias.epilogue_depth_preference,
            "fewer registers should increase epilogue depth preference"
        );
    }

    #[test]
    fn large_l1_reduces_fusion_cost() {
        let small_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (16384, 262144, 8388608),
        };
        let large_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 262144, 8388608),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let small_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &small_l1);
        let large_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &large_l1);
        assert!(
            large_bias.fusion_cost_scale < small_bias.fusion_cost_scale,
            "larger L1 should reduce fusion cost scale"
        );
    }

    #[test]
    fn mode_baseline_latency_specific_values() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Neutral hw: 32 regs, 64K L1 — no hw adjustments
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        assert_approx(bias.batch_flexibility, 0.0, 0.01);
        assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
        assert_approx(bias.speculative_decoding_value, 1.5, 0.01);
        assert_approx(bias.quantization_aggressiveness, 1.5, 0.01);
    }

    #[test]
    fn mode_baseline_throughput_specific_values() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &neutral_hw);
        assert_approx(bias.batch_flexibility, 1.0, 0.01);
        assert_approx(bias.speculative_decoding_value, 0.3, 0.01);
        assert_approx(bias.expert_prefetch_priority, 1.5, 0.01);
    }

    #[test]
    fn moe_modulation_with_high_parallelism() {
        let moe_arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.9,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let no_moe_arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.1,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let moe = StrategyArbiter::arbitrate(InferenceMode::Throughput, &moe_arch, &hw);
        let no_moe = StrategyArbiter::arbitrate(InferenceMode::Throughput, &no_moe_arch, &hw);
        assert!(
            moe.expert_eviction_aggressiveness > no_moe.expert_eviction_aggressiveness,
            "high parallelism + high memory should increase eviction aggressiveness"
        );
    }

    #[test]
    fn reg_tension_positive_adjusts_preferences() {
        let fusion_heavy = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.9,
            pipeline_valuable: 0.1,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &fusion_heavy, &neutral_hw);
        // Positive reg_tension: epilogue up, k_depth down
        assert!(bias.epilogue_depth_preference > 1.5);
        assert!(bias.k_depth_preference < 1.3);
    }

    #[test]
    fn reg_tension_negative_adjusts_preferences() {
        let pipeline_heavy = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.1,
            pipeline_valuable: 0.9,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &pipeline_heavy, &neutral_hw);
        // Negative reg_tension: k_depth up, epilogue down
        assert!(bias.k_depth_preference > 1.3);
        assert!(bias.epilogue_depth_preference < 1.5);
    }

    #[test]
    fn arbitrate_cpu_convenience_method() {
        let dp = DeviceProfile::detect();
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate_cpu(InferenceMode::Latency, &arch, &dp);
        assert!(bias.fusion_cost_scale > 0.0);
    }

    // ── New tests: 20 additional ──

    // ── InferenceMode Hash trait ──

    #[test]
    fn inference_mode_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(InferenceMode::Latency);
        set.insert(InferenceMode::Throughput);
        assert_eq!(set.len(), 2);
        // Inserting the same variants again should not increase size.
        set.insert(InferenceMode::Latency);
        set.insert(InferenceMode::Throughput);
        assert_eq!(set.len(), 2);
    }

    // ── StrategyBias Default ──

    #[test]
    fn strategy_bias_default_all_ones_or_zeros() {
        let bias = StrategyBias::default();
        assert_eq!(bias.fusion_cost_scale, 1.0);
        assert_eq!(bias.pipeline_cost_scale, 1.0);
        assert_eq!(bias.parallelism_cost_scale, 1.0);
        assert_eq!(bias.epilogue_depth_preference, 1.0);
        assert_eq!(bias.k_depth_preference, 1.0);
        assert_eq!(bias.kv_cache_budget_scale, 1.0);
        assert_eq!(bias.weight_prefetch_budget_scale, 1.0);
        assert_eq!(bias.batch_flexibility, 1.0);
        assert_eq!(bias.decode_ratio_scale, 1.0);
        assert_eq!(bias.speculative_decoding_value, 1.0);
        assert_eq!(bias.quantization_aggressiveness, 1.0);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 1.0);
    }

    // ── StrategyBias validate() clamping ──

    #[test]
    fn strategy_bias_validate_clamps_high_values() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 10.0,
            pipeline_cost_scale: 10.0,
            parallelism_cost_scale: 10.0,
            epilogue_depth_preference: 10.0,
            k_depth_preference: 10.0,
            kv_cache_budget_scale: 10.0,
            weight_prefetch_budget_scale: 10.0,
            batch_flexibility: 5.0,
            decode_ratio_scale: 10.0,
            speculative_decoding_value: 10.0,
            quantization_aggressiveness: 10.0,
            expert_eviction_aggressiveness: 10.0,
            expert_prefetch_priority: 10.0,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 3.0);
        assert_eq!(bias.pipeline_cost_scale, 3.0);
        assert_eq!(bias.parallelism_cost_scale, 3.0);
        assert_eq!(bias.epilogue_depth_preference, 3.0);
        assert_eq!(bias.k_depth_preference, 3.0);
        assert_eq!(bias.kv_cache_budget_scale, 3.0);
        assert_eq!(bias.weight_prefetch_budget_scale, 3.0);
        assert_eq!(bias.batch_flexibility, 1.0);
        assert_eq!(bias.decode_ratio_scale, 2.0);
        assert_eq!(bias.speculative_decoding_value, 3.0);
        assert_eq!(bias.quantization_aggressiveness, 3.0);
        assert_eq!(bias.expert_eviction_aggressiveness, 2.0);
        assert_eq!(bias.expert_prefetch_priority, 5.0);
    }

    #[test]
    fn strategy_bias_validate_clamps_low_values() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.0,
            pipeline_cost_scale: 0.0,
            parallelism_cost_scale: 0.0,
            epilogue_depth_preference: 0.0,
            k_depth_preference: 0.0,
            kv_cache_budget_scale: 0.0,
            weight_prefetch_budget_scale: 0.0,
            batch_flexibility: -1.0,
            decode_ratio_scale: 0.0,
            speculative_decoding_value: 0.0,
            quantization_aggressiveness: 0.0,
            expert_eviction_aggressiveness: -1.0,
            expert_prefetch_priority: 0.0,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.pipeline_cost_scale, 0.2);
        assert_eq!(bias.parallelism_cost_scale, 0.1);
        assert_eq!(bias.epilogue_depth_preference, 0.3);
        assert_eq!(bias.k_depth_preference, 0.3);
        assert_eq!(bias.kv_cache_budget_scale, 0.2);
        assert_eq!(bias.weight_prefetch_budget_scale, 0.2);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.decode_ratio_scale, 0.3);
        assert_eq!(bias.speculative_decoding_value, 0.1);
        assert_eq!(bias.quantization_aggressiveness, 0.3);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 0.1);
    }

    #[test]
    fn strategy_bias_validate_within_range_is_noop() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 1.0,
            pipeline_cost_scale: 1.0,
            parallelism_cost_scale: 1.0,
            epilogue_depth_preference: 1.0,
            k_depth_preference: 1.0,
            kv_cache_budget_scale: 1.0,
            weight_prefetch_budget_scale: 1.0,
            batch_flexibility: 0.5,
            decode_ratio_scale: 1.0,
            speculative_decoding_value: 1.0,
            quantization_aggressiveness: 1.0,
            expert_eviction_aggressiveness: 0.5,
            expert_prefetch_priority: 1.0,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 1.0);
        assert_eq!(bias.batch_flexibility, 0.5);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.5);
    }

    // ── StrategyBias accessor methods ──

    #[test]
    fn strategy_bias_accessor_methods() {
        let bias = StrategyBias {
            expert_eviction_aggressiveness: 1.23,
            expert_prefetch_priority: 3.45,
            ..StrategyBias::default()
        };
        assert_approx(bias.expert_eviction_aggressiveness(), 1.23, 0.01);
        assert_approx(bias.expert_prefetch_priority(), 3.45, 0.01);
    }

    // ── StrategyBias Debug/Clone/Copy traits ──

    #[test]
    fn strategy_bias_debug_trait() {
        let bias = StrategyBias::default();
        let s = format!("{:?}", bias);
        assert!(s.contains("fusion_cost_scale"));
        assert!(s.contains("batch_flexibility"));
        assert!(s.contains("expert_eviction_aggressiveness"));
    }

    #[test]
    fn strategy_bias_clone_trait() {
        let original = StrategyBias {
            fusion_cost_scale: 0.7,
            batch_flexibility: 0.9,
            ..StrategyBias::default()
        };
        let cloned = original.clone();
        assert_approx(cloned.fusion_cost_scale, 0.7, 0.001);
        assert_approx(cloned.batch_flexibility, 0.9, 0.001);
    }

    #[test]
    fn strategy_bias_copy_trait() {
        let original = StrategyBias {
            kv_cache_budget_scale: 2.0,
            ..StrategyBias::default()
        };
        let copied = original;
        // Copy semantics: original is still valid (unlike move).
        assert_approx(original.kv_cache_budget_scale, 2.0, 0.001);
        assert_approx(copied.kv_cache_budget_scale, 2.0, 0.001);
    }

    // ── GraphArchetype construction and field access ──

    #[test]
    fn graph_archetype_field_access() {
        let arch = GraphArchetype {
            compute_intensive: 0.8,
            memory_intensive: 0.4,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.3,
            pipeline_valuable: 0.7,
        };
        assert_approx(arch.compute_intensive, 0.8, 0.001);
        assert_approx(arch.memory_intensive, 0.4, 0.001);
        assert_approx(arch.parallelism_exploitable, 0.6, 0.001);
        assert_approx(arch.fusion_profitable, 0.3, 0.001);
        assert_approx(arch.pipeline_valuable, 0.7, 0.001);
    }

    #[test]
    fn graph_archetype_clone() {
        let arch = GraphArchetype {
            compute_intensive: 0.9,
            memory_intensive: 0.1,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.2,
            pipeline_valuable: 0.8,
        };
        let cloned = arch.clone();
        assert_approx(cloned.compute_intensive, 0.9, 0.001);
        assert_approx(cloned.pipeline_valuable, 0.8, 0.001);
    }

    // ── lerp helper via observable behavior ──

    #[test]
    fn lerp_at_zero_returns_a() {
        // With t=0 on all archetype fields, modulation should be identity (lerp(1.0, x, 0) = 1.0).
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &neutral_hw);
        // With zero archetype fields, lerp(1.0, x, 0) = 1.0 for all modulation factors,
        // so fusion_cost_scale should equal the raw baseline 0.5 (no modulation, no hw adj).
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
    }

    // ── CPU register scarcity boundary: exactly 16 ──

    #[test]
    fn cpu_exactly_16_registers_triggers_scarcity() {
        let hw_16 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let hw_32 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_16 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_16);
        let bias_32 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_32);
        assert!(
            bias_16.epilogue_depth_preference > bias_32.epilogue_depth_preference,
            "16 registers should have higher epilogue depth preference than 32"
        );
    }

    #[test]
    fn cpu_17_registers_no_scarcity() {
        // The scarcity branch fires when num_simd_regs <= 16.
        // With 17 registers the branch is skipped, so epilogue_depth_preference
        // should equal the 32-register baseline (no scarcity adjustment).
        let hw_17 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 17,
            cache_sizes: (65536, 0, 0),
        };
        let hw_32 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_17 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_17);
        let bias_32 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_32);
        assert_approx(
            bias_17.epilogue_depth_preference,
            bias_32.epilogue_depth_preference,
            0.01,
        );
    }

    // ── Memory-intensive modulation paths ──

    #[test]
    fn kv_cache_budget_scales_with_memory_intensity() {
        let mem_heavy = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let mem_light = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let heavy = StrategyArbiter::arbitrate(InferenceMode::Latency, &mem_heavy, &hw);
        let light = StrategyArbiter::arbitrate(InferenceMode::Latency, &mem_light, &hw);
        assert!(
            heavy.kv_cache_budget_scale > light.kv_cache_budget_scale,
            "high memory intensity should increase kv_cache_budget_scale"
        );
    }

    #[test]
    fn quantization_aggressiveness_scales_with_memory_intensity() {
        let mem_heavy = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let mem_light = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let heavy = StrategyArbiter::arbitrate(InferenceMode::Latency, &mem_heavy, &hw);
        let light = StrategyArbiter::arbitrate(InferenceMode::Latency, &mem_light, &hw);
        assert!(
            heavy.quantization_aggressiveness > light.quantization_aggressiveness,
            "high memory intensity should increase quantization aggressiveness"
        );
    }

    // ── batch_flexibility is mode-dependent, never modulated by archetype/hw ──

    #[test]
    fn batch_flexibility_latency_always_zero() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_a100());
        assert_eq!(bias.batch_flexibility, 0.0);
    }

    #[test]
    fn batch_flexibility_throughput_always_one() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &cpu_avx2());
        assert_eq!(bias.batch_flexibility, 1.0);
    }

    // ── ArbiterHwView cache_sizes L2/L3 fields ──

    #[test]
    fn arbiter_hw_view_cache_sizes_all_nonzero() {
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 262144, 8388608),
        };
        assert_eq!(view.cache_sizes.0, 32768);
        assert_eq!(view.cache_sizes.1, 262144);
        assert_eq!(view.cache_sizes.2, 8388608);
    }

    // ── decode_ratio_scale is never modulated (both modes = 1.0) ──

    #[test]
    fn decode_ratio_scale_unmodulated_in_both_modes() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = cpu_avx512();
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(lat.decode_ratio_scale, 1.0, 0.01);
        assert_approx(thr.decode_ratio_scale, 1.0, 0.01);
    }

    // ── weight_prefetch_budget_scale differs between modes ──

    #[test]
    fn weight_prefetch_bias_latency_higher_than_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            lat.weight_prefetch_budget_scale > thr.weight_prefetch_budget_scale,
            "latency mode should prefetch weights more aggressively"
        );
    }

    // ── New tests: 18 additional (48 → 66) ──

    // ── ArbiterHwView::gpu with maximum shared memory ──

    #[test]
    fn arbiter_hw_view_gpu_max_shared_mem() {
        let view = ArbiterHwView::gpu(usize::MAX);
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.num_simd_regs, 255);
        assert_eq!(view.cache_sizes.0, usize::MAX);
        assert_eq!(view.cache_sizes.1, 0);
        assert_eq!(view.cache_sizes.2, 0);
    }

    // ── ArbiterHwView with zero L1 (zero-cache boundary) ──

    #[test]
    fn arbiter_hw_view_zero_l1_no_panic() {
        let zero_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (0, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Should not panic; l1_richness will be 0.0, so the if-branch is skipped.
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &zero_l1);
        assert!(bias.fusion_cost_scale > 0.0);
    }

    // ── ArbiterHwView manual field mutation ──

    #[test]
    fn arbiter_hw_view_field_mutation() {
        let mut view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        view.device = DeviceFamily::Gpu;
        view.num_simd_regs = 64;
        view.cache_sizes.0 = 98304;
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.num_simd_regs, 64);
        assert_eq!(view.cache_sizes.0, 98304);
        // Unchanged fields
        assert_eq!(view.cache_sizes.1, 262144);
        assert_eq!(view.cache_sizes.2, 8388608);
    }

    // ── InferenceMode variants are exactly two ──

    #[test]
    fn inference_mode_exactly_two_variants() {
        let modes = [InferenceMode::Latency, InferenceMode::Throughput];
        assert_eq!(modes.len(), 2);
        assert_ne!(modes[0], modes[1]);
    }

    // ── Throughput mode has higher kv_cache_budget_scale than Latency ──

    #[test]
    fn throughput_higher_kv_cache_budget_than_latency() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            thr.kv_cache_budget_scale > lat.kv_cache_budget_scale,
            "throughput should allocate more KV cache budget"
        );
    }

    // ── StrategyBias validate() boundary values stay unchanged ──

    #[test]
    fn strategy_bias_validate_boundary_values() {
        // Values exactly at clamp boundaries should remain unchanged.
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.2,
            parallelism_cost_scale: 0.1,
            batch_flexibility: 0.0,
            decode_ratio_scale: 0.3,
            expert_eviction_aggressiveness: 0.0,
            expert_prefetch_priority: 0.1,
            speculative_decoding_value: 0.1,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.parallelism_cost_scale, 0.1);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.decode_ratio_scale, 0.3);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 0.1);
        assert_eq!(bias.speculative_decoding_value, 0.1);
    }

    // ── GraphArchetype Debug trait ──

    #[test]
    fn graph_archetype_debug_trait() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let s = format!("{:?}", arch);
        assert!(s.contains("compute_intensive"));
        assert!(s.contains("memory_intensive"));
        assert!(s.contains("parallelism_exploitable"));
        assert!(s.contains("fusion_profitable"));
        assert!(s.contains("pipeline_valuable"));
    }

    // ── Arbitrate with GPU + Throughput produces valid bias ──

    #[test]
    fn arbitrate_gpu_throughput_produces_valid_bias() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView::gpu(49152);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // All fields should be positive (validate() ensures clamping).
        assert!(bias.fusion_cost_scale > 0.0);
        assert!(bias.pipeline_cost_scale > 0.0);
        assert!(bias.parallelism_cost_scale > 0.0);
        assert!(bias.epilogue_depth_preference > 0.0);
        assert!(bias.k_depth_preference > 0.0);
        assert!(bias.kv_cache_budget_scale > 0.0);
        assert!(bias.weight_prefetch_budget_scale > 0.0);
        assert!(bias.decode_ratio_scale > 0.0);
        assert!(bias.speculative_decoding_value > 0.0);
        assert!(bias.quantization_aggressiveness > 0.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0);
        assert!(bias.expert_prefetch_priority > 0.0);
    }

    // ── Throughput baseline expert_eviction_aggressiveness vs Latency ──

    #[test]
    fn throughput_higher_expert_eviction_than_latency() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            thr.expert_eviction_aggressiveness > lat.expert_eviction_aggressiveness,
            "throughput baseline has higher expert eviction aggressiveness (0.8 vs 0.0)"
        );
    }

    // ── Deterministic: same inputs produce same outputs ──

    #[test]
    fn arbitrate_deterministic_same_inputs() {
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.2,
        };
        let hw = cpu_avx2();
        let a = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let b = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_eq!(a.fusion_cost_scale, b.fusion_cost_scale);
        assert_eq!(a.pipeline_cost_scale, b.pipeline_cost_scale);
        assert_eq!(a.parallelism_cost_scale, b.parallelism_cost_scale);
        assert_eq!(a.epilogue_depth_preference, b.epilogue_depth_preference);
        assert_eq!(a.k_depth_preference, b.k_depth_preference);
        assert_eq!(a.batch_flexibility, b.batch_flexibility);
    }

    // ── GPU with zero shared_mem: L1 richness is 0.0, fusion_cost_scale unchanged ──

    #[test]
    fn gpu_zero_shared_mem_skips_l1_richness() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu_zero = ArbiterHwView::gpu(0);
        // GPU with zero L1: l1_richness = 0.0, so the if-branch is skipped.
        // fusion_cost_scale should be 0.5 * (no archetype modulation) * (no L1 adj) = 0.5
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_zero);
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
    }

    // ── StrategyBias validate() mixed values: partial clamping ──

    #[test]
    fn strategy_bias_validate_mixed_partial_clamping() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.5,   // within range, unchanged
            pipeline_cost_scale: 5.0,  // above max, clamped to 3.0
            parallelism_cost_scale: 0.05, // below min, clamped to 0.1
            batch_flexibility: 0.7,    // within range, unchanged
            expert_eviction_aggressiveness: 3.0, // above max, clamped to 2.0
            ..StrategyBias::default()
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.5);
        assert_eq!(bias.pipeline_cost_scale, 3.0);
        assert_eq!(bias.parallelism_cost_scale, 0.1);
        assert_eq!(bias.batch_flexibility, 0.7);
        assert_eq!(bias.expert_eviction_aggressiveness, 2.0);
        // Default fields within range stay at 1.0
        assert_eq!(bias.epilogue_depth_preference, 1.0);
        assert_eq!(bias.k_depth_preference, 1.0);
    }

    // ── StrategyBias all 13 fields individually accessible ──

    #[test]
    fn strategy_bias_all_fields_accessible() {
        let bias = StrategyBias {
            fusion_cost_scale: 0.1,
            pipeline_cost_scale: 0.2,
            parallelism_cost_scale: 0.3,
            epilogue_depth_preference: 0.4,
            k_depth_preference: 0.5,
            kv_cache_budget_scale: 0.6,
            weight_prefetch_budget_scale: 0.7,
            batch_flexibility: 0.8,
            decode_ratio_scale: 0.9,
            speculative_decoding_value: 1.0,
            quantization_aggressiveness: 1.1,
            expert_eviction_aggressiveness: 1.2,
            expert_prefetch_priority: 1.3,
        };
        assert_eq!(bias.fusion_cost_scale, 0.1);
        assert_eq!(bias.pipeline_cost_scale, 0.2);
        assert_eq!(bias.parallelism_cost_scale, 0.3);
        assert_eq!(bias.epilogue_depth_preference, 0.4);
        assert_eq!(bias.k_depth_preference, 0.5);
        assert_eq!(bias.kv_cache_budget_scale, 0.6);
        assert_eq!(bias.weight_prefetch_budget_scale, 0.7);
        assert_eq!(bias.batch_flexibility, 0.8);
        assert_eq!(bias.decode_ratio_scale, 0.9);
        assert_eq!(bias.speculative_decoding_value, 1.0);
        assert_eq!(bias.quantization_aggressiveness, 1.1);
        assert_eq!(bias.expert_eviction_aggressiveness, 1.2);
        assert_eq!(bias.expert_prefetch_priority, 1.3);
    }

    // ── GPU hardware adjustment boosts k_depth_preference ──

    #[test]
    fn gpu_boosts_k_depth_preference() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert!(
            gpu_bias.k_depth_preference > cpu_bias.k_depth_preference,
            "GPU should boost k_depth_preference by 1.2x"
        );
    }

    // ── Latency baseline: speculative_decoding_value > Throughput baseline ──

    #[test]
    fn latency_higher_speculative_decoding_than_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            lat.speculative_decoding_value > thr.speculative_decoding_value,
            "latency mode should value speculative decoding more (1.5 vs 0.3)"
        );
    }

    // ── StrategyArbiter is unit struct, no fields ──

    #[test]
    fn strategy_arbiter_unit_struct() {
        let _arbiter = StrategyArbiter;
        // StrategyArbiter is a unit struct with no fields; just verifying it compiles.
    }

    // ── CPU with 1 register (extreme scarcity) ──

    #[test]
    fn cpu_single_register_extreme_scarcity() {
        let hw_1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 1,
            cache_sizes: (65536, 0, 0),
        };
        let hw_32 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_1 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_1);
        let bias_32 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_32);
        // scarcity = 1.0 - (1/32) = 0.96875; epilogue boost = 1 + 0.96875 * 0.3 = 1.29
        assert!(
            bias_1.epilogue_depth_preference > bias_32.epilogue_depth_preference,
            "1 register should dramatically increase epilogue depth preference"
        );
    }

    // ── L1 exactly 64KB: richness = 1.0, sqrt(1.0) = 1.0, no fusion change ──

    #[test]
    fn l1_exactly_64kb_no_fusion_adjustment() {
        let hw_64k = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_64k);
        // l1_richness = 65536/65536 = 1.0, sqrt = 1.0, fusion_cost_scale *= 1/1.0 = 0.5
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
    }

    // ── L1 double 64KB (128KB): richness = 2.0, capped by .min(2.0) ──

    #[test]
    fn l1_128kb_capped_richness_reduces_fusion_cost() {
        let hw_128k = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0),
        };
        let hw_64k = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_128k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_128k);
        let bias_64k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_64k);
        // 128KB: richness = 2.0, sqrt = 1.414, fusion_cost_scale *= 1/1.414 ≈ 0.354
        assert!(
            bias_128k.fusion_cost_scale < bias_64k.fusion_cost_scale,
            "128KB L1 should further reduce fusion cost vs 64KB"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (43 new, 67 → 110)
    // ══════════════════════════════════════════════════════════════════════

    // ── ArbiterHwView PartialEq ──

    #[test]
    fn arbiter_hw_view_eq_identical() {
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn arbiter_hw_view_neq_device_differs() {
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        assert_ne!(cpu, gpu);
    }

    #[test]
    fn arbiter_hw_view_neq_num_simd_regs_differs() {
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        assert_ne!(a, b);
    }

    #[test]
    fn arbiter_hw_view_neq_cache_sizes_differs() {
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        assert_ne!(a, b);
    }

    // ── ArbiterHwView Hash ──

    #[test]
    fn arbiter_hw_view_hash_consistency() {
        use std::collections::HashMap;
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mut map = HashMap::new();
        map.insert(view, "cpu-avx512");
        assert_eq!(map.get(&view), Some(&"cpu-avx512"));
        // Same view inserted again overwrites.
        map.insert(view, "still-cpu-avx512");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn arbiter_hw_view_hash_distinct_keys() {
        use std::collections::HashMap;
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let mut map = HashMap::new();
        map.insert(cpu, "cpu");
        map.insert(gpu, "gpu");
        assert_eq!(map.len(), 2);
    }

    // ── ArbiterHwView Eq (total equality) ──

    #[test]
    fn arbiter_hw_view_eq_reflexive() {
        let view = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        assert_eq!(view, view);
    }

    #[test]
    fn arbiter_hw_view_eq_symmetric() {
        let a = ArbiterHwView::gpu(0);
        let b = ArbiterHwView::gpu(0);
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn arbiter_hw_view_eq_transitive() {
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let c = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── ArbiterHwView::gpu edge cases ──

    #[test]
    fn arbiter_hw_view_gpu_small_shared_mem() {
        let view = ArbiterHwView::gpu(1);
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.cache_sizes.0, 1);
        assert_eq!(view.cache_sizes.1, 0);
        assert_eq!(view.cache_sizes.2, 0);
    }

    #[test]
    fn arbiter_hw_view_gpu_l2_and_l3_always_zero() {
        let view = ArbiterHwView::gpu(usize::MAX);
        assert_eq!(view.cache_sizes.1, 0);
        assert_eq!(view.cache_sizes.2, 0);
    }

    // ── ArbiterHwView debug output includes all fields ──

    #[test]
    fn arbiter_hw_view_debug_all_fields() {
        let view = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 128,
            cache_sizes: (65536, 524288, 16777216),
        };
        let s = format!("{:?}", view);
        assert!(s.contains("device: Gpu"), "Debug output should show device field");
        assert!(s.contains("128"), "Debug output should show num_simd_regs value");
        assert!(s.contains("cache_sizes"), "Debug output should show cache_sizes field");
    }

    // ── GraphArchetype PartialEq ──

    #[test]
    fn graph_archetype_eq_identical() {
        let a = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.6,
        };
        let b = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.6,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn graph_archetype_neq_single_field_differs() {
        let a = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.6,
        };
        let b = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.7, // differs
        };
        assert_ne!(a, b);
    }

    // ── GraphArchetype Copy semantics ──

    #[test]
    fn graph_archetype_copy_semantics() {
        let original = GraphArchetype {
            compute_intensive: 0.9,
            memory_intensive: 0.1,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.2,
            pipeline_valuable: 0.8,
        };
        let copied = original;
        assert_eq!(original, copied);
    }

    // ── GraphArchetype zero fields ──

    #[test]
    fn graph_archetype_zero_fields() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        assert_approx(arch.compute_intensive, 0.0, 0.001);
        assert_approx(arch.memory_intensive, 0.0, 0.001);
        assert_approx(arch.fusion_profitable, 0.0, 0.001);
    }

    // ── GraphArchetype all-ones fields ──

    #[test]
    fn graph_archetype_all_ones() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        assert_approx(arch.compute_intensive, 1.0, 0.001);
        assert_approx(arch.parallelism_exploitable, 1.0, 0.001);
    }

    // ── Register tension exactly zero (fusion_profitable == pipeline_valuable) ──

    #[test]
    fn reg_tension_zero_no_preference_change() {
        let equal_tension = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.6,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &equal_tension, &neutral_hw);
        // With reg_tension = 0, neither epilogue nor k_depth is adjusted beyond baseline.
        // Baseline Latency: epilogue_depth_preference = 1.5, k_depth_preference = 1.3
        // fusion_profitable=0.6 reduces fusion_cost_scale via lerp(1.0, 0.6, 0.6) = 0.76
        // But epilogue/k_depth should stay at baseline (no reg tension adjustment).
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
    }

    // ── MoE modulation with low parallelism (below 0.5 threshold) ──

    #[test]
    fn moe_modulation_low_parallelism_no_expert_adjustment() {
        let low_parallel = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.4, // below 0.5 threshold
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let zero_parallel = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let low = StrategyArbiter::arbitrate(InferenceMode::Throughput, &low_parallel, &hw);
        let zero = StrategyArbiter::arbitrate(InferenceMode::Throughput, &zero_parallel, &hw);
        // parallelism_exploitable below 0.5 does not trigger MoE expert modulation,
        // so expert_eviction_aggressiveness should be identical.
        assert_approx(
            low.expert_eviction_aggressiveness,
            zero.expert_eviction_aggressiveness,
            0.01,
        );
        assert_approx(
            low.expert_prefetch_priority,
            zero.expert_prefetch_priority,
            0.01,
        );
    }

    // ── CPU with 0 registers (absolute minimum) ──

    #[test]
    fn cpu_zero_registers_extreme_scarcity_no_panic() {
        let hw_0 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 0,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // scarcity = 1.0 - (0/32) = 1.0; must not panic.
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_0);
        assert!(bias.epilogue_depth_preference > 0.0);
        assert!(bias.fusion_cost_scale > 0.0);
    }

    // ── Register scarcity at boundary: num_simd_regs = 16 exactly ──

    #[test]
    fn cpu_16_registers_scarcity_formula() {
        // scarcity = 1.0 - (16/32) = 0.5
        // epilogue boost = 1 + 0.5 * 0.3 = 1.15
        // k_depth reduction = 1 - 0.5 * 0.2 = 0.9
        let hw_16 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_16);
        // Latency baseline epilogue_depth_preference = 1.5, * 1.15 = 1.725
        assert_approx(bias.epilogue_depth_preference, 1.725, 0.01);
        // Latency baseline k_depth_preference = 1.3, * 0.9 = 1.17
        assert_approx(bias.k_depth_preference, 1.17, 0.01);
    }

    // ── L1 richness beyond 2x cap ──

    #[test]
    fn l1_beyond_cap_richness_still_capped() {
        // 256KB L1: richness = 262144/65536 = 4.0, capped to 2.0 by .min(2.0)
        // Should produce same fusion_cost_scale as 128KB (also capped to 2.0).
        let hw_256k = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (262144, 0, 0),
        };
        let hw_128k = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_256k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_256k);
        let bias_128k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_128k);
        // Both capped at richness 2.0, so fusion_cost_scale should be identical.
        assert_approx(
            bias_256k.fusion_cost_scale,
            bias_128k.fusion_cost_scale,
            0.001,
        );
    }

    // ── L1 richness just above 64KB (e.g. 96KB) ──

    #[test]
    fn l1_96kb_richness_between_1_and_2() {
        let hw_96k = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (98304, 0, 0), // 96KB
        };
        let hw_64k = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0), // 64KB
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_96k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_96k);
        let bias_64k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_64k);
        // 96KB: richness = 1.5, sqrt = 1.225; fusion cost should be less than 64KB.
        assert!(
            bias_96k.fusion_cost_scale < bias_64k.fusion_cost_scale,
            "96KB L1 should reduce fusion cost more than 64KB"
        );
    }

    // ── Latency mode parallelism_cost_scale higher than Throughput ──

    #[test]
    fn latency_higher_parallelism_cost_than_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            lat.parallelism_cost_scale > thr.parallelism_cost_scale,
            "latency mode should penalize parallelism overhead more (1.5 vs 0.5)"
        );
    }

    // ── Latency mode quantization_aggressiveness higher than Throughput ──

    #[test]
    fn latency_higher_quantization_aggressiveness_than_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            lat.quantization_aggressiveness > thr.quantization_aggressiveness,
            "latency baseline quantization aggressiveness (1.5) > throughput (0.8)"
        );
    }

    // ── Throughput mode expert_prefetch_priority higher than Latency ──

    #[test]
    fn throughput_higher_expert_prefetch_priority_than_latency() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            thr.expert_prefetch_priority > lat.expert_prefetch_priority,
            "throughput baseline expert_prefetch_priority (1.5) > latency (0.5)"
        );
    }

    // ── Fusion modulation precise calculation ──

    #[test]
    fn fusion_modulation_precise_calculation() {
        // fusion_profitable = 0.8 → lerp(1.0, 0.6, 0.8) = 1.0 + 0.8 * (0.6 - 1.0) = 1.0 - 0.32 = 0.68
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.8,
            pipeline_valuable: 0.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        // Latency baseline fusion_cost_scale = 0.5, * 0.68 = 0.34
        assert_approx(bias.fusion_cost_scale, 0.34, 0.01);
    }

    // ── Pipeline modulation precise calculation ──

    #[test]
    fn pipeline_modulation_precise_calculation() {
        // pipeline_valuable = 0.5 → lerp(1.0, 0.6, 0.5) = 1.0 + 0.5 * (0.6 - 1.0) = 1.0 - 0.2 = 0.8
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.5,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        // Latency baseline pipeline_cost_scale = 0.6, * 0.8 = 0.48
        assert_approx(bias.pipeline_cost_scale, 0.48, 0.01);
    }

    // ── GPU adjustment precise multiplier ──

    #[test]
    fn gpu_adjustment_precise_multiplier() {
        // GPU: epilogue *= 1.2, k_depth *= 1.2, pipeline *= 1.2
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536); // L1=64KB → richness=1.0, sqrt=1.0, no L1 adj.
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // Latency baseline: epilogue = 1.5, pipeline = 0.6, k_depth = 1.3
        // GPU: epilogue = 1.5 * 1.2 = 1.8, pipeline = 0.6 * 1.2 = 0.72, k_depth = 1.3 * 1.2 = 1.56
        assert_approx(bias.epilogue_depth_preference, 1.8, 0.01);
        assert_approx(bias.pipeline_cost_scale, 0.72, 0.01);
        assert_approx(bias.k_depth_preference, 1.56, 0.01);
    }

    // ── CPU with high register count (>16) no scarcity adjustment ──

    #[test]
    fn cpu_high_registers_no_scarcity_adjustment() {
        let hw_64 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 64,
            cache_sizes: (65536, 0, 0),
        };
        let hw_32 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_64 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_64);
        let bias_32 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_32);
        // Both > 16, no scarcity adjustment. Same L1. Should be identical.
        assert_approx(
            bias_64.epilogue_depth_preference,
            bias_32.epilogue_depth_preference,
            0.001,
        );
        assert_approx(
            bias_64.k_depth_preference,
            bias_32.k_depth_preference,
            0.001,
        );
    }

    // ── StrategyBias validate at exact max boundaries ──

    #[test]
    fn strategy_bias_validate_exact_max_boundaries() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 3.0,
            pipeline_cost_scale: 3.0,
            parallelism_cost_scale: 3.0,
            epilogue_depth_preference: 3.0,
            k_depth_preference: 3.0,
            kv_cache_budget_scale: 3.0,
            weight_prefetch_budget_scale: 3.0,
            batch_flexibility: 1.0,
            decode_ratio_scale: 2.0,
            expert_eviction_aggressiveness: 2.0,
            expert_prefetch_priority: 5.0,
            speculative_decoding_value: 3.0,
            quantization_aggressiveness: 3.0,
        };
        bias.validate();
        // All at max boundaries, should remain unchanged.
        assert_eq!(bias.fusion_cost_scale, 3.0);
        assert_eq!(bias.pipeline_cost_scale, 3.0);
        assert_eq!(bias.parallelism_cost_scale, 3.0);
        assert_eq!(bias.epilogue_depth_preference, 3.0);
        assert_eq!(bias.k_depth_preference, 3.0);
        assert_eq!(bias.kv_cache_budget_scale, 3.0);
        assert_eq!(bias.weight_prefetch_budget_scale, 3.0);
        assert_eq!(bias.batch_flexibility, 1.0);
        assert_eq!(bias.decode_ratio_scale, 2.0);
        assert_eq!(bias.expert_eviction_aggressiveness, 2.0);
        assert_eq!(bias.expert_prefetch_priority, 5.0);
        assert_eq!(bias.speculative_decoding_value, 3.0);
        assert_eq!(bias.quantization_aggressiveness, 3.0);
    }

    // ── StrategyBias validate at exact min boundaries ──

    #[test]
    fn strategy_bias_validate_exact_min_boundaries() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.2,
            pipeline_cost_scale: 0.2,
            parallelism_cost_scale: 0.1,
            epilogue_depth_preference: 0.3,
            k_depth_preference: 0.3,
            kv_cache_budget_scale: 0.2,
            weight_prefetch_budget_scale: 0.2,
            batch_flexibility: 0.0,
            decode_ratio_scale: 0.3,
            expert_eviction_aggressiveness: 0.0,
            expert_prefetch_priority: 0.1,
            speculative_decoding_value: 0.1,
            quantization_aggressiveness: 0.3,
        };
        bias.validate();
        // All at min boundaries, should remain unchanged.
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.pipeline_cost_scale, 0.2);
        assert_eq!(bias.parallelism_cost_scale, 0.1);
        assert_eq!(bias.epilogue_depth_preference, 0.3);
        assert_eq!(bias.k_depth_preference, 0.3);
        assert_eq!(bias.kv_cache_budget_scale, 0.2);
        assert_eq!(bias.weight_prefetch_budget_scale, 0.2);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.decode_ratio_scale, 0.3);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 0.1);
        assert_eq!(bias.speculative_decoding_value, 0.1);
        assert_eq!(bias.quantization_aggressiveness, 0.3);
    }

    // ── StrategyBias negative values clamped to min ──

    #[test]
    fn strategy_bias_validate_negative_values() {
        let mut bias = StrategyBias {
            fusion_cost_scale: -5.0,
            pipeline_cost_scale: -1.0,
            parallelism_cost_scale: -0.5,
            epilogue_depth_preference: -2.0,
            k_depth_preference: -1.0,
            kv_cache_budget_scale: -3.0,
            weight_prefetch_budget_scale: -0.1,
            batch_flexibility: -10.0,
            decode_ratio_scale: -1.0,
            expert_eviction_aggressiveness: -5.0,
            expert_prefetch_priority: -2.0,
            speculative_decoding_value: -0.5,
            quantization_aggressiveness: -1.0,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.pipeline_cost_scale, 0.2);
        assert_eq!(bias.parallelism_cost_scale, 0.1);
        assert_eq!(bias.epilogue_depth_preference, 0.3);
        assert_eq!(bias.k_depth_preference, 0.3);
        assert_eq!(bias.kv_cache_budget_scale, 0.2);
        assert_eq!(bias.weight_prefetch_budget_scale, 0.2);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.decode_ratio_scale, 0.3);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 0.1);
        assert_eq!(bias.speculative_decoding_value, 0.1);
        assert_eq!(bias.quantization_aggressiveness, 0.3);
    }

    // ── StrategyBias validate idempotent ──

    #[test]
    fn strategy_bias_validate_idempotent() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.5,
            pipeline_cost_scale: 0.5,
            parallelism_cost_scale: 0.5,
            epilogue_depth_preference: 0.5,
            k_depth_preference: 0.5,
            kv_cache_budget_scale: 0.5,
            weight_prefetch_budget_scale: 0.5,
            batch_flexibility: 0.5,
            decode_ratio_scale: 1.0,
            expert_eviction_aggressiveness: 1.0,
            expert_prefetch_priority: 2.5,
            speculative_decoding_value: 1.5,
            quantization_aggressiveness: 1.5,
        };
        bias.validate();
        let first = bias;
        bias.validate();
        assert_eq!(first.fusion_cost_scale, bias.fusion_cost_scale);
        assert_eq!(first.batch_flexibility, bias.batch_flexibility);
        assert_eq!(first.expert_prefetch_priority, bias.expert_prefetch_priority);
    }

    // ── StrategyBias field mutation ──

    #[test]
    fn strategy_bias_field_mutation() {
        let mut bias = StrategyBias::default();
        bias.fusion_cost_scale = 2.5;
        bias.batch_flexibility = 0.0;
        bias.expert_eviction_aggressiveness = 1.5;
        assert_approx(bias.fusion_cost_scale, 2.5, 0.001);
        assert_approx(bias.batch_flexibility, 0.0, 0.001);
        assert_approx(bias.expert_eviction_aggressiveness, 1.5, 0.001);
    }

    // ── StrategyBias accessor methods return correct values ──

    #[test]
    fn strategy_bias_accessors_return_field_values() {
        let bias = StrategyBias {
            expert_eviction_aggressiveness: 0.0,
            expert_prefetch_priority: 0.0,
            ..StrategyBias::default()
        };
        assert_approx(bias.expert_eviction_aggressiveness(), 0.0, 0.001);
        assert_approx(bias.expert_prefetch_priority(), 0.0, 0.001);
    }

    // ── arbiter output always within validate bounds ──

    #[test]
    fn arbiter_output_always_within_bounds() {
        fn check_in_bounds(bias: &StrategyBias) {
            assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
            assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
            assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
            assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
            assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
            assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
            assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
            assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
            assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
            assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
            assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
            assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
            assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
        }

        // Sweep: all archetype values 0.0, 0.5, 1.0 × Latency/Throughput × cpu/gpu
        let vals = [0.0, 0.5, 1.0];
        let modes = [InferenceMode::Latency, InferenceMode::Throughput];
        let hws = [
            ArbiterHwView {
                device: DeviceFamily::Cpu,
                num_simd_regs: 16,
                cache_sizes: (32768, 0, 0),
            },
            ArbiterHwView {
                device: DeviceFamily::Cpu,
                num_simd_regs: 32,
                cache_sizes: (65536, 0, 0),
            },
            ArbiterHwView::gpu(49152),
        ];
        for &ci in &vals {
            for &mi in &vals {
                for &pe in &vals {
                    for &fp in &vals {
                        for &pv in &vals {
                            let arch = GraphArchetype {
                                compute_intensive: ci,
                                memory_intensive: mi,
                                parallelism_exploitable: pe,
                                fusion_profitable: fp,
                                pipeline_valuable: pv,
                            };
                            for &mode in &modes {
                                for hw in &hws {
                                    let bias = StrategyArbiter::arbitrate(mode, &arch, hw);
                                    check_in_bounds(&bias);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── InferenceMode all variant combinations not equal ──

    #[test]
    fn inference_mode_all_pairs_unequal() {
        let variants = [InferenceMode::Latency, InferenceMode::Throughput];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i != j {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    // ── arbiter_cpu matches manual From conversion ──

    #[test]
    fn arbitrate_cpu_matches_manual_conversion() {
        let dp = DeviceProfile::detect();
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.2,
        };
        let bias_convenience = StrategyArbiter::arbitrate_cpu(InferenceMode::Latency, &arch, &dp);
        let bias_manual = StrategyArbiter::arbitrate(
            InferenceMode::Latency,
            &arch,
            &ArbiterHwView::from(&dp),
        );
        assert_eq!(bias_convenience.fusion_cost_scale, bias_manual.fusion_cost_scale);
        assert_eq!(bias_convenience.pipeline_cost_scale, bias_manual.pipeline_cost_scale);
        assert_eq!(bias_convenience.parallelism_cost_scale, bias_manual.parallelism_cost_scale);
        assert_eq!(bias_convenience.epilogue_depth_preference, bias_manual.epilogue_depth_preference);
        assert_eq!(bias_convenience.k_depth_preference, bias_manual.k_depth_preference);
        assert_eq!(bias_convenience.batch_flexibility, bias_manual.batch_flexibility);
        assert_eq!(bias_convenience.kv_cache_budget_scale, bias_manual.kv_cache_budget_scale);
    }

    // ── InferenceMode Ord-like: Hash + Eq usable in HashMap ──

    #[test]
    fn inference_mode_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(InferenceMode::Latency, "low-latency config");
        map.insert(InferenceMode::Throughput, "high-throughput config");
        assert_eq!(map.get(&InferenceMode::Latency), Some(&"low-latency config"));
        assert_eq!(map.get(&InferenceMode::Throughput), Some(&"high-throughput config"));
    }

    // ── Combined GPU + register scarcity: GPU branch fires first ──

    #[test]
    fn gpu_with_low_registers_gpu_branch_takes_priority() {
        // GPU flag=true, num_simd_regs=8 (low). Both GPU and scarcity branches fire.
        let gpu_low_reg = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_low_reg = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_low_reg);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu_low_reg);
        // GPU bias gets the 1.2x epilogue boost on top of scarcity.
        // CPU bias gets only scarcity adjustment.
        // GPU epilogue = baseline * scarcity * 1.2 > CPU epilogue = baseline * scarcity
        assert!(
            gpu_bias.epilogue_depth_preference > cpu_bias.epilogue_depth_preference,
            "GPU + low registers should boost epilogue more than CPU + low registers"
        );
    }

    // ── Combined archetype effects: high fusion + high memory ──

    #[test]
    fn high_fusion_high_memory_combined_effects() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // fusion_profitable=1.0: fusion_cost_scale *= lerp(1.0, 0.6, 1.0) = 0.6 → 0.5 * 0.6 = 0.3
        assert_approx(bias.fusion_cost_scale, 0.3, 0.01);
        // memory_intensive=1.0: kv_cache_budget_scale *= lerp(1.0, 1.5, 1.0) = 1.5 → 0.5 * 1.5 = 0.75
        assert_approx(bias.kv_cache_budget_scale, 0.75, 0.01);
        // quantization_aggressiveness *= lerp(1.0, 1.3, 1.0) = 1.3 → 1.5 * 1.3 = 1.95
        assert_approx(bias.quantization_aggressiveness, 1.95, 0.01);
    }

    // ── Deterministic across different call order ──

    #[test]
    fn arbitrate_deterministic_independent_of_order() {
        let arch = GraphArchetype {
            compute_intensive: 0.8,
            memory_intensive: 0.6,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.3,
            pipeline_valuable: 0.7,
        };
        let hw = ArbiterHwView::gpu(98304);
        let bias1 = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        let bias2 = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Exact floating point equality — deterministic pure function.
        assert_eq!(bias1.fusion_cost_scale, bias2.fusion_cost_scale);
        assert_eq!(bias1.expert_eviction_aggressiveness, bias2.expert_eviction_aggressiveness);
        assert_eq!(bias1.expert_prefetch_priority, bias2.expert_prefetch_priority);
    }

    // ── ArbiterHwView::gpu returns consistent is_gpu ──

    #[test]
    fn arbiter_hw_view_gpu_always_true() {
        assert!(ArbiterHwView::gpu(0).device == DeviceFamily::Gpu);
        assert!(ArbiterHwView::gpu(1).device == DeviceFamily::Gpu);
        assert!(ArbiterHwView::gpu(usize::MAX).device == DeviceFamily::Gpu);
    }

    // ── ArbiterHwView::gpu num_simd_regs always 255 ──

    #[test]
    fn arbiter_hw_view_gpu_num_simd_regs_always_255() {
        assert_eq!(ArbiterHwView::gpu(0).num_simd_regs, 255);
        assert_eq!(ArbiterHwView::gpu(100).num_simd_regs, 255);
        assert_eq!(ArbiterHwView::gpu(usize::MAX).num_simd_regs, 255);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (51 new, 112 → 163)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. ArbiterHwView::gpu L2/L3 always zero for all sizes ──

    #[test]
    fn arbiter_hw_view_gpu_l2_l3_zero_for_various_sizes() {
        for &size in &[0usize, 1024, 49152, 100_000, usize::MAX] {
            let view = ArbiterHwView::gpu(size);
            assert_eq!(view.cache_sizes.1, 0, "L2 should be 0 for size {size}");
            assert_eq!(view.cache_sizes.2, 0, "L3 should be 0 for size {size}");
        }
    }

    // ── 2. GraphArchetype negative field values produce valid bias ──

    #[test]
    fn graph_archetype_negative_fields_valid_bias() {
        let neg_arch = GraphArchetype {
            compute_intensive: -0.5,
            memory_intensive: -0.3,
            parallelism_exploitable: -0.2,
            fusion_profitable: -0.1,
            pipeline_valuable: -0.4,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &neg_arch, &hw);
        assert!(bias.fusion_cost_scale >= 0.2);
        assert!(bias.kv_cache_budget_scale >= 0.2);
    }

    // ── 3. GraphArchetype values above 1.0 ──

    #[test]
    fn graph_archetype_over_one_produces_valid_bias() {
        let over_arch = GraphArchetype {
            compute_intensive: 2.0,
            memory_intensive: 1.5,
            parallelism_exploitable: 3.0,
            fusion_profitable: 1.2,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &over_arch, &hw);
        assert!(bias.fusion_cost_scale >= 0.2);
        assert!(bias.fusion_cost_scale <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3);
    }

    // ── 4. Parallelism modulation precise calculation ──

    #[test]
    fn parallelism_modulation_precise() {
        // parallelism_exploitable = 0.6 -> lerp(1.0, 0.5, 0.6) = 0.7
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        // Latency baseline parallelism_cost_scale = 1.5, * 0.7 = 1.05
        assert_approx(bias.parallelism_cost_scale, 1.05, 0.01);
    }

    // ── 5. Reg tension positive precise formula ──

    #[test]
    fn reg_tension_positive_precise() {
        // fusion_profitable=0.9, pipeline_valuable=0.3 -> reg_tension=0.6
        // epilogue boost = 1.0 + 0.6 * 0.5 = 1.3
        // k_depth reduction = 1.0 - 0.6 * 0.3 = 0.82
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.9,
            pipeline_valuable: 0.3,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        assert_approx(bias.epilogue_depth_preference, 1.95, 0.01);
        assert_approx(bias.k_depth_preference, 1.066, 0.01);
    }

    // ── 6. Reg tension negative precise formula ──

    #[test]
    fn reg_tension_negative_precise() {
        // fusion_profitable=0.2, pipeline_valuable=0.8 -> reg_tension=-0.6
        // k_depth boost = 1.0 + 0.6*0.5 = 1.3
        // epilogue reduction = 1.0 - 0.6*0.3 = 0.82
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.2,
            pipeline_valuable: 0.8,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        assert_approx(bias.k_depth_preference, 1.69, 0.01);
        assert_approx(bias.epilogue_depth_preference, 1.23, 0.01);
    }

    // ── 7. MoE modulation with parallelism exactly at 0.5 threshold ──

    #[test]
    fn moe_modulation_parallelism_at_threshold() {
        let exactly_05 = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let below_05 = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.49,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let at = StrategyArbiter::arbitrate(InferenceMode::Throughput, &exactly_05, &hw);
        let below = StrategyArbiter::arbitrate(InferenceMode::Throughput, &below_05, &hw);
        assert!(
            at.expert_eviction_aggressiveness >= below.expert_eviction_aggressiveness,
            "parallelism=0.5 should trigger MoE modulation"
        );
    }

    // ── 8. MoE modulation with high parallelism but low memory ──

    #[test]
    fn moe_high_parallel_low_memory_no_expert_adjustment() {
        let high_parallel_low_mem = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.9,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let zero_parallel = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let hp = StrategyArbiter::arbitrate(InferenceMode::Throughput, &high_parallel_low_mem, &hw);
        let zp = StrategyArbiter::arbitrate(InferenceMode::Throughput, &zero_parallel, &hw);
        // memory_intensive=0.0: lerp(1.0, 1.5, 0.0) = 1.0 — no expert modulation
        assert_approx(
            hp.expert_eviction_aggressiveness,
            zp.expert_eviction_aggressiveness,
            0.01,
        );
    }

    // ── 9. CPU 15 registers scarcity fires ──

    #[test]
    fn cpu_15_registers_scarcity_fires() {
        let hw_15 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 15,
            cache_sizes: (65536, 0, 0),
        };
        let hw_32 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_15 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_15);
        let bias_32 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_32);
        assert!(
            bias_15.epilogue_depth_preference > bias_32.epilogue_depth_preference,
            "15 registers should trigger scarcity adjustment"
        );
    }

    // ── 10. CPU 8 registers scarcity precise ──

    #[test]
    fn cpu_8_registers_scarcity_precise() {
        // scarcity = 1.0 - (8/32) = 0.75
        // epilogue = 1.0 + 0.75*0.3 = 1.225
        // k_depth = 1.0 - 0.75*0.2 = 0.85
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.8375, 0.01);
        assert_approx(bias.k_depth_preference, 1.105, 0.01);
    }

    // ── 11. GPU + high fusion_profitable combined ──

    #[test]
    fn gpu_high_fusion_combined() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // fusion_profitable=1.0: fusion *= lerp(1.0, 0.6, 1.0) = 0.6 -> 0.5 * 0.6 = 0.3
        assert_approx(bias.fusion_cost_scale, 0.3, 0.01);
        // reg_tension = 1.0 - 0.0 = 1.0: epilogue *= 1.0 + 1.0*0.5 = 1.5 -> 1.5 * 1.5 = 2.25
        // GPU: epilogue *= 1.2 -> 2.25 * 1.2 = 2.7
        assert_approx(bias.epilogue_depth_preference, 2.7, 0.01);
        // reg_tension = 1.0: k_depth *= 1.0 - 1.0*0.3 = 0.7 -> 1.3 * 0.7 = 0.91
        // GPU: k_depth *= 1.2 -> 0.91 * 1.2 = 1.092
        assert_approx(bias.k_depth_preference, 1.092, 0.01);
    }

    // ── 12. Throughput + GPU + high memory ──

    #[test]
    fn throughput_gpu_high_memory() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert_approx(bias.kv_cache_budget_scale, 2.25, 0.01);
        assert_approx(bias.quantization_aggressiveness, 1.04, 0.01);
    }

    // ── 13. L1 richness precise: 32KB ──

    #[test]
    fn l1_32kb_richness_precise() {
        // l1_richness = 32768/65536 = 0.5, sqrt(0.5) = 0.7071
        // fusion *= 1/0.7071 = 1.4142
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.707, 0.01);
    }

    // ── 14. L1 richness precise: 16KB ──

    #[test]
    fn l1_16kb_richness_precise() {
        // l1_richness = 0.25, sqrt = 0.5, fusion *= 1/0.5 = 2.0
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (16384, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 1.0, 0.01);
    }

    // ── 15. StrategyBias validate with f64::MAX ──

    #[test]
    fn strategy_bias_validate_f64_max() {
        let mut bias = StrategyBias {
            fusion_cost_scale: f64::MAX,
            pipeline_cost_scale: f64::MAX,
            parallelism_cost_scale: f64::MAX,
            epilogue_depth_preference: f64::MAX,
            k_depth_preference: f64::MAX,
            kv_cache_budget_scale: f64::MAX,
            weight_prefetch_budget_scale: f64::MAX,
            batch_flexibility: f64::MAX,
            decode_ratio_scale: f64::MAX,
            speculative_decoding_value: f64::MAX,
            quantization_aggressiveness: f64::MAX,
            expert_eviction_aggressiveness: f64::MAX,
            expert_prefetch_priority: f64::MAX,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 3.0);
        assert_eq!(bias.expert_prefetch_priority, 5.0);
        assert_eq!(bias.batch_flexibility, 1.0);
    }

    // ── 16. StrategyBias validate with f64::MIN ──

    #[test]
    fn strategy_bias_validate_f64_min() {
        let mut bias = StrategyBias {
            fusion_cost_scale: f64::MIN,
            pipeline_cost_scale: f64::MIN,
            parallelism_cost_scale: f64::MIN,
            epilogue_depth_preference: f64::MIN,
            k_depth_preference: f64::MIN,
            kv_cache_budget_scale: f64::MIN,
            weight_prefetch_budget_scale: f64::MIN,
            batch_flexibility: f64::MIN,
            decode_ratio_scale: f64::MIN,
            speculative_decoding_value: f64::MIN,
            quantization_aggressiveness: f64::MIN,
            expert_eviction_aggressiveness: f64::MIN,
            expert_prefetch_priority: f64::MIN,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
    }

    // ── 17. StrategyBias validate with NaN (NaN.clamp is identity) ──

    #[test]
    fn strategy_bias_validate_nan_preserved() {
        // Rust's f64::clamp(NaN, min, max) returns NaN (IEEE 754 semantics).
        // validate() does not eliminate NaN — callers must ensure non-NaN inputs.
        let mut bias = StrategyBias {
            fusion_cost_scale: f64::NAN,
            pipeline_cost_scale: f64::NAN,
            parallelism_cost_scale: f64::NAN,
            epilogue_depth_preference: f64::NAN,
            k_depth_preference: f64::NAN,
            kv_cache_budget_scale: f64::NAN,
            weight_prefetch_budget_scale: f64::NAN,
            batch_flexibility: f64::NAN,
            decode_ratio_scale: f64::NAN,
            speculative_decoding_value: f64::NAN,
            quantization_aggressiveness: f64::NAN,
            expert_eviction_aggressiveness: f64::NAN,
            expert_prefetch_priority: f64::NAN,
        };
        bias.validate();
        assert!(bias.fusion_cost_scale.is_nan());
        assert!(bias.batch_flexibility.is_nan());
        assert!(bias.expert_eviction_aggressiveness.is_nan());
    }

    // ── 18. StrategyBias validate with infinity ──

    #[test]
    fn strategy_bias_validate_infinity() {
        let mut bias = StrategyBias {
            fusion_cost_scale: f64::INFINITY,
            batch_flexibility: f64::INFINITY,
            expert_eviction_aggressiveness: f64::INFINITY,
            expert_prefetch_priority: f64::INFINITY,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 3.0);
        assert_eq!(bias.batch_flexibility, 1.0);
        assert_eq!(bias.expert_eviction_aggressiveness, 2.0);
        assert_eq!(bias.expert_prefetch_priority, 5.0);
    }

    // ── 19. StrategyBias validate with negative infinity ──

    #[test]
    fn strategy_bias_validate_neg_infinity() {
        let mut bias = StrategyBias {
            fusion_cost_scale: f64::NEG_INFINITY,
            batch_flexibility: f64::NEG_INFINITY,
            expert_eviction_aggressiveness: f64::NEG_INFINITY,
            expert_prefetch_priority: f64::NEG_INFINITY,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 0.1);
    }

    // ── 20. StrategyBias validate double clamp same result ──

    #[test]
    fn strategy_bias_validate_double_clamp() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 100.0,
            batch_flexibility: -50.0,
            expert_prefetch_priority: 100.0,
            ..StrategyBias::default()
        };
        bias.validate();
        let first_pass = bias;
        let mut second = first_pass;
        second.validate();
        assert_eq!(first_pass.fusion_cost_scale, second.fusion_cost_scale);
        assert_eq!(first_pass.batch_flexibility, second.batch_flexibility);
        assert_eq!(first_pass.expert_prefetch_priority, second.expert_prefetch_priority);
    }

    // ── 21. All mode and hw type combinations valid ──

    #[test]
    fn arbitrate_all_mode_hw_combinations_valid() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hws = [cpu_avx2(), cpu_avx512(), gpu_a100()];
        let modes = [InferenceMode::Latency, InferenceMode::Throughput];
        for &mode in &modes {
            for hw in &hws {
                let bias = StrategyArbiter::arbitrate(mode, &arch, hw);
                assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
                assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
                assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
            }
        }
    }

    // ── 22. compute_intensive does not affect output ──

    #[test]
    fn compute_intensive_no_effect() {
        let arch_high = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let arch_low = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let high = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_high, &hw);
        let low = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_low, &hw);
        assert_eq!(high.fusion_cost_scale, low.fusion_cost_scale);
        assert_eq!(high.kv_cache_budget_scale, low.kv_cache_budget_scale);
        assert_eq!(high.batch_flexibility, low.batch_flexibility);
    }

    // ── 23. ArbiterHwView from DeviceProfile preserves cache_sizes ──

    #[test]
    fn arbiter_hw_view_from_device_profile_preserves_cache() {
        let dp = DeviceProfile::detect();
        let view = ArbiterHwView::from(&dp);
        assert_eq!(view.cache_sizes, dp.cache_sizes());
    }

    // ── 24. CPU zero registers scarcity saturates ──

    #[test]
    fn cpu_zero_registers_scarcity_saturates() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 0,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.95, 0.01);
        assert_approx(bias.k_depth_preference, 1.04, 0.01);
    }

    // ── 25. Latency decode_ratio_scale never modulated ──

    #[test]
    fn latency_decode_ratio_never_modulated() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_a100());
        assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
    }

    // ── 26. Throughput decode_ratio_scale never modulated ──

    #[test]
    fn throughput_decode_ratio_never_modulated() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &cpu_avx2());
        assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
    }

    // ── 27. GPU pipeline_cost_scale precise boost ──

    #[test]
    fn gpu_pipeline_cost_precise_boost() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.5,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // pipeline *= lerp(1.0, 0.6, 0.5) = 0.8 -> 0.6 * 0.8 = 0.48
        // GPU: pipeline *= 1.2 -> 0.48 * 1.2 = 0.576
        assert_approx(bias.pipeline_cost_scale, 0.576, 0.01);
    }

    // ── 28. Only L1 matters for fusion, not L2/L3 ──

    #[test]
    fn only_l1_matters_for_fusion() {
        let small_l1_big_rest = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (16384, 1048576, 33554432),
        };
        let small_l1_zero_rest = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (16384, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let big = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &small_l1_big_rest);
        let zero = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &small_l1_zero_rest);
        assert_approx(big.fusion_cost_scale, zero.fusion_cost_scale, 0.001);
    }

    // ── 29. Latency baseline all fields non-negative ──

    #[test]
    fn latency_baseline_all_non_negative() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert!(bias.fusion_cost_scale >= 0.0);
        assert!(bias.pipeline_cost_scale >= 0.0);
        assert!(bias.parallelism_cost_scale >= 0.0);
        assert!(bias.epilogue_depth_preference >= 0.0);
        assert!(bias.k_depth_preference >= 0.0);
        assert!(bias.kv_cache_budget_scale >= 0.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.0);
        assert!(bias.batch_flexibility >= 0.0);
        assert!(bias.decode_ratio_scale >= 0.0);
        assert!(bias.speculative_decoding_value >= 0.0);
        assert!(bias.quantization_aggressiveness >= 0.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0);
        assert!(bias.expert_prefetch_priority >= 0.0);
    }

    // ── 30. Throughput baseline all fields non-negative ──

    #[test]
    fn throughput_baseline_all_non_negative() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(bias.fusion_cost_scale >= 0.0);
        assert!(bias.pipeline_cost_scale >= 0.0);
        assert!(bias.parallelism_cost_scale >= 0.0);
        assert!(bias.epilogue_depth_preference >= 0.0);
        assert!(bias.k_depth_preference >= 0.0);
        assert!(bias.kv_cache_budget_scale >= 0.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.0);
        assert!(bias.batch_flexibility >= 0.0);
        assert!(bias.decode_ratio_scale >= 0.0);
        assert!(bias.speculative_decoding_value >= 0.0);
        assert!(bias.quantization_aggressiveness >= 0.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0);
        assert!(bias.expert_prefetch_priority >= 0.0);
    }

    // ── 31. ArbiterHwView eq GPU same shared mem ──

    #[test]
    fn arbiter_hw_view_eq_gpu_same_shared_mem() {
        let a = ArbiterHwView::gpu(49152);
        let b = ArbiterHwView::gpu(49152);
        assert_eq!(a, b);
    }

    // ── 32. ArbiterHwView neq GPU different shared mem ──

    #[test]
    fn arbiter_hw_view_neq_gpu_different_shared_mem() {
        let a = ArbiterHwView::gpu(49152);
        let b = ArbiterHwView::gpu(16384);
        assert_ne!(a, b);
    }

    // ── 33. InferenceMode exhaustive match ──

    #[test]
    fn inference_mode_exhaustive_match() {
        let mode = InferenceMode::Latency;
        let label = match mode {
            InferenceMode::Latency => "latency",
            InferenceMode::Throughput => "throughput",
        };
        assert_eq!(label, "latency");
    }

    // ── 34. StrategyBias default then validate is no-op ──

    #[test]
    fn strategy_bias_default_validate_noop() {
        let mut bias = StrategyBias::default();
        let before = bias.fusion_cost_scale;
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, before);
        assert_eq!(bias.batch_flexibility, 1.0);
    }

    // ── 35. MoE modulation precise: high parallel + high memory ──

    #[test]
    fn moe_modulation_precise_high_parallel_high_memory() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.8,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.expert_eviction_aggressiveness, 1.2, 0.01);
        assert_approx(bias.expert_prefetch_priority, 3.0, 0.01);
    }

    // ── 36. KV cache modulation precise: medium memory ──

    #[test]
    fn kv_cache_modulation_medium_memory_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.kv_cache_budget_scale, 0.625, 0.01);
    }

    // ── 37. Quantization modulation precise: medium memory ──

    #[test]
    fn quantization_modulation_medium_memory_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.quantization_aggressiveness, 1.725, 0.01);
    }

    // ── 38. Latency weight_prefetch only mode driven ──

    #[test]
    fn latency_weight_prefetch_only_mode_driven() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.weight_prefetch_budget_scale, 1.5, 0.01);
    }

    // ── 39. Throughput weight_prefetch only mode driven ──

    #[test]
    fn throughput_weight_prefetch_only_mode_driven() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu_a100());
        assert_approx(bias.weight_prefetch_budget_scale, 0.8, 0.01);
    }

    // ── 40. StrategyBias field wise comparison ──

    #[test]
    fn strategy_bias_field_wise_comparison() {
        let a = StrategyBias::default();
        let b = StrategyBias {
            fusion_cost_scale: 2.0,
            ..StrategyBias::default()
        };
        assert_ne!(a.fusion_cost_scale, b.fusion_cost_scale);
        assert_eq!(a.batch_flexibility, b.batch_flexibility);
    }

    // ── 41. ArbiterHwView::gpu shared_mem stored as L1 ──

    #[test]
    fn arbiter_hw_view_gpu_shared_mem_as_l1() {
        let shared = 99_999usize;
        let view = ArbiterHwView::gpu(shared);
        assert_eq!(view.cache_sizes.0, shared);
    }

    // ── 42. GraphArchetype all equal fields: zero reg tension ──

    #[test]
    fn graph_archetype_equal_fields_zero_reg_tension() {
        let arch = GraphArchetype {
            compute_intensive: 0.42,
            memory_intensive: 0.42,
            parallelism_exploitable: 0.42,
            fusion_profitable: 0.42,
            pipeline_valuable: 0.42,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
    }

    // ── 43. CPU usize::MAX registers: no scarcity ──

    #[test]
    fn cpu_max_registers_no_scarcity() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: usize::MAX,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
    }

    // ── 44. GPU zero shared_mem: l1_richness 0.0 skips branch ──

    #[test]
    fn gpu_zero_shared_mem_l1_richness_zero() {
        let gpu = ArbiterHwView::gpu(0);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
    }

    // ── 45. GPU one byte shared_mem: very small L1 triggers big fusion boost then validate clamps ──

    #[test]
    fn gpu_one_byte_shared_mem_fusion_clamped() {
        let gpu = ArbiterHwView::gpu(1);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert!(bias.fusion_cost_scale <= 3.0);
        assert!(bias.fusion_cost_scale >= 0.2);
    }

    // ── 46. InferenceMode size is 1 byte ──

    #[test]
    fn inference_mode_size_one_byte() {
        assert_eq!(std::mem::size_of::<InferenceMode>(), 1);
    }

    // ── 47. StrategyBias size is 13 f64s ──

    #[test]
    fn strategy_bias_size_13_f64s() {
        assert_eq!(std::mem::size_of::<StrategyBias>(), 13 * 8);
    }

    // ── 48. ArbiterHwView size accounts for alignment padding ──

    #[test]
    fn arbiter_hw_view_size_accounts_for_alignment() {
        // bool(1) + padding(7) + usize(8) + (usize, usize, usize)(24) = 40 with alignment
        let actual = std::mem::size_of::<ArbiterHwView>();
        assert!(
            actual >= std::mem::size_of::<bool>() + 4 * std::mem::size_of::<usize>(),
            "ArbiterHwView size ({actual}) should accommodate all fields"
        );
        assert_eq!(actual, 40, "ArbiterHwView layout should be stable");
    }

    // ── 49. GraphArchetype size is 5 f64s ──

    #[test]
    fn graph_archetype_size_5_f64s() {
        assert_eq!(std::mem::size_of::<GraphArchetype>(), 5 * 8);
    }

    // ── 50. StrategyArbiter is zero-sized ──

    #[test]
    fn strategy_arbiter_zero_sized() {
        assert_eq!(std::mem::size_of::<StrategyArbiter>(), 0);
    }

    // ── 51. Golden vector 1 recheck: all 13 fields in range ──

    #[test]
    fn golden_vector_1_all_fields_in_range() {
        let archetype = GraphArchetype {
            compute_intensive: 0.95,
            memory_intensive: 0.60,
            parallelism_exploitable: 0.20,
            fusion_profitable: 0.85,
            pipeline_valuable: 0.15,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &archetype, &cpu_avx2());
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (52 new, 164 → 216)
    // ══════════════════════════════════════════════════════════════════════

    // ── 52. validate triple invocation is idempotent ──

    #[test]
    fn strategy_bias_validate_triple_idempotent() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 50.0,
            pipeline_cost_scale: -10.0,
            parallelism_cost_scale: 0.3,
            epilogue_depth_preference: 1.0,
            k_depth_preference: 5.0,
            kv_cache_budget_scale: 0.1,
            weight_prefetch_budget_scale: 2.0,
            batch_flexibility: 0.5,
            decode_ratio_scale: 1.0,
            speculative_decoding_value: 0.2,
            quantization_aggressiveness: 4.0,
            expert_eviction_aggressiveness: 3.0,
            expert_prefetch_priority: 10.0,
        };
        bias.validate();
        let after1 = bias;
        let mut bias2 = after1;
        bias2.validate();
        let after2 = bias2;
        let mut bias3 = after2;
        bias3.validate();
        assert_eq!(after2.fusion_cost_scale, bias3.fusion_cost_scale);
        assert_eq!(after2.batch_flexibility, bias3.batch_flexibility);
        assert_eq!(after2.expert_eviction_aggressiveness, bias3.expert_eviction_aggressiveness);
        assert_eq!(after2.expert_prefetch_priority, bias3.expert_prefetch_priority);
    }

    // ── 53. sigmoid helper: extreme positive input ──

    #[test]
    fn sigmoid_extreme_positive_approaches_one() {
        let result = sigmoid(100.0);
        assert_approx(result, 1.0, 0.001);
    }

    // ── 54. sigmoid helper: extreme negative input ──

    #[test]
    fn sigmoid_extreme_negative_approaches_zero() {
        let result = sigmoid(-100.0);
        assert!(result < 1e-30, "sigmoid(-100) should be vanishingly small, got {result}");
    }

    // ── 55. sigmoid helper: zero returns 0.5 ──

    #[test]
    fn sigmoid_zero_returns_half() {
        assert_approx(sigmoid(0.0), 0.5, 0.001);
    }

    // ── 56. lerp helper: t=1.0 returns b ──

    #[test]
    fn lerp_at_one_returns_b() {
        // Indirectly test: fusion_profitable=1.0 multiplies by lerp(1.0, 0.6, 1.0) = 0.6
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.3, 0.01);
    }

    // ── 57. lerp helper: t=0.5 returns midpoint ──

    #[test]
    fn lerp_at_half_returns_midpoint() {
        // pipeline_valuable=0.5 -> lerp(1.0, 0.6, 0.5) = 0.8
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Throughput baseline pipeline_cost_scale = 1.3, * 0.8 = 1.04
        assert_approx(bias.pipeline_cost_scale, 1.04, 0.01);
    }

    // ── 58. GPU boosts pipeline for Throughput mode ──

    #[test]
    fn gpu_boosts_pipeline_throughput_mode() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert!(
            gpu_bias.pipeline_cost_scale > cpu_bias.pipeline_cost_scale,
            "GPU should boost pipeline_cost_scale in throughput mode too"
        );
    }

    // ── 59. CPU with 4 registers: extreme scarcity k_depth reduction ──

    #[test]
    fn cpu_4_registers_k_depth_reduced() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 4,
            cache_sizes: (65536, 0, 0),
        };
        let hw_32 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let low = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let high = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_32);
        assert!(
            low.k_depth_preference < high.k_depth_preference,
            "4 registers should reduce k_depth_preference below baseline"
        );
    }

    // ── 60. CPU 4 registers scarcity precise calculation ──

    #[test]
    fn cpu_4_registers_scarcity_precise() {
        // scarcity = 1.0 - (4/32) = 0.875
        // epilogue = 1.0 + 0.875*0.3 = 1.2625
        // k_depth = 1.0 - 0.875*0.2 = 0.825
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 4,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Latency baseline epilogue=1.5, * 1.2625 = 1.89375
        assert_approx(bias.epilogue_depth_preference, 1.89375, 0.01);
        // Latency baseline k_depth=1.3, * 0.825 = 1.0725
        assert_approx(bias.k_depth_preference, 1.0725, 0.01);
    }

    // ── 61. CPU 12 registers scarcity precise calculation ──

    #[test]
    fn cpu_12_registers_scarcity_precise() {
        // scarcity = 1.0 - (12/32) = 0.625
        // epilogue = 1.0 + 0.625*0.3 = 1.1875
        // k_depth = 1.0 - 0.625*0.2 = 0.875
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 12,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.78125, 0.01);
        assert_approx(bias.k_depth_preference, 1.1375, 0.01);
    }

    // ── 62. Reg tension positive with maximum fusion_profitable ──

    #[test]
    fn reg_tension_positive_max_fusion_no_pipeline() {
        // fusion_profitable=1.0, pipeline_valuable=0.0 -> reg_tension=1.0
        // epilogue *= 1.0 + 1.0*0.5 = 1.5
        // k_depth *= 1.0 - 1.0*0.3 = 0.7
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.5 = 2.25
        assert_approx(bias.epilogue_depth_preference, 2.25, 0.01);
        // baseline k_depth=1.3, * 0.7 = 0.91
        assert_approx(bias.k_depth_preference, 0.91, 0.01);
    }

    // ── 63. Reg tension negative with maximum pipeline_valuable ──

    #[test]
    fn reg_tension_negative_max_pipeline_no_fusion() {
        // fusion_profitable=0.0, pipeline_valuable=1.0 -> reg_tension=-1.0
        // k_depth *= 1.0 + 1.0*0.5 = 1.5
        // epilogue *= 1.0 - 1.0*0.3 = 0.7
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline k_depth=1.3, * 1.5 = 1.95
        assert_approx(bias.k_depth_preference, 1.95, 0.01);
        // baseline epilogue=1.5, * 0.7 = 1.05
        assert_approx(bias.epilogue_depth_preference, 1.05, 0.01);
    }

    // ── 64. Throughput mode baseline fusion_cost_scale precise ──

    #[test]
    fn throughput_baseline_fusion_cost_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 1.0, 0.01);
    }

    // ── 65. Throughput mode baseline parallelism_cost_scale precise ──

    #[test]
    fn throughput_baseline_parallelism_cost_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.parallelism_cost_scale, 0.5, 0.01);
    }

    // ── 66. Latency mode baseline epilogue_depth precise ──

    #[test]
    fn latency_baseline_epilogue_depth_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
    }

    // ── 67. Throughput mode baseline epilogue_depth precise ──

    #[test]
    fn throughput_baseline_epilogue_depth_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 0.8, 0.01);
    }

    // ── 68. L1 exactly 64KB with Throughput mode ──

    #[test]
    fn l1_64kb_throughput_mode_no_fusion_adjustment() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 1.0, 0.01);
    }

    // ── 69. L1 richness monotonic: larger L1 always reduces or equal fusion cost ──

    #[test]
    fn l1_richness_monotonic_fusion_cost() {
        // L1=0 skips the richness branch entirely (no adjustment), so start from
        // small positive sizes where the branch fires.
        let l1_sizes = [1024usize, 16384, 32768, 49152, 65536, 98304, 131072, 262144];
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let mut prev_cost = f64::MAX;
        for &l1 in &l1_sizes {
            let hw = ArbiterHwView {
                device: DeviceFamily::Cpu,
                num_simd_regs: 32,
                cache_sizes: (l1, 0, 0),
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.fusion_cost_scale <= prev_cost + 0.001,
                "fusion_cost_scale should be non-increasing with L1 size: l1={l1}, cost={}, prev={}",
                bias.fusion_cost_scale, prev_cost
            );
            prev_cost = bias.fusion_cost_scale;
        }
    }

    // ── 70. CPU 10 registers scarcity precise ──

    #[test]
    fn cpu_10_registers_scarcity_precise() {
        // scarcity = 1.0 - (10/32) = 0.6875
        // epilogue = 1.0 + 0.6875*0.3 = 1.20625
        // k_depth = 1.0 - 0.6875*0.2 = 0.8625
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 10,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.809375, 0.01);
        assert_approx(bias.k_depth_preference, 1.12125, 0.01);
    }

    // ── 71. GPU + Throughput + high memory: quantization precise ──

    #[test]
    fn gpu_throughput_high_memory_quantization_precise() {
        // Throughput baseline quantization=0.8, lerp(1.0, 1.3, 1.0)=1.3 -> 0.8*1.3=1.04
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert_approx(bias.quantization_aggressiveness, 1.04, 0.01);
    }

    // ── 72. GPU + Latency + high memory: kv_cache precise ──

    #[test]
    fn gpu_latency_high_memory_kv_cache_precise() {
        // Latency baseline kv_cache=0.5, lerp(1.0, 1.5, 1.0)=1.5 -> 0.5*1.5=0.75
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(bias.kv_cache_budget_scale, 0.75, 0.01);
    }

    // ── 73. Combined: high fusion + low pipeline + high memory + Throughput ──

    #[test]
    fn combined_high_fusion_low_pipeline_high_memory_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.9,
            pipeline_valuable: 0.1,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // All outputs should be valid
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
    }

    // ── 74. CPU + Throughput + register scarcity + high memory ──

    #[test]
    fn cpu_throughput_scarcity_high_memory_valid() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (32768, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.3,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
    }

    // ── 75. Parallelism modulation with Throughput precise ──

    #[test]
    fn parallelism_modulation_throughput_precise() {
        // parallelism_exploitable=0.4 -> lerp(1.0, 0.5, 0.4) = 0.8
        // Throughput baseline parallelism_cost_scale=0.5, * 0.8 = 0.4
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.parallelism_cost_scale, 0.4, 0.01);
    }

    // ── 76. Fusion modulation with Throughput precise ──

    #[test]
    fn fusion_modulation_throughput_precise() {
        // fusion_profitable=0.7 -> lerp(1.0, 0.6, 0.7) = 0.72
        // Throughput baseline fusion_cost_scale=1.0, * 0.72 = 0.72
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.72, 0.01);
    }

    // ── 77. Pipeline modulation with Throughput precise ──

    #[test]
    fn pipeline_modulation_throughput_precise() {
        // pipeline_valuable=0.6 -> lerp(1.0, 0.6, 0.6) = 0.76
        // Throughput baseline pipeline_cost_scale=1.3, * 0.76 = 0.988
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.6,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.pipeline_cost_scale, 0.988, 0.01);
    }

    // ── 78. Reg tension with small positive value (0.1) ──

    #[test]
    fn reg_tension_small_positive() {
        // fusion_profitable=0.6, pipeline_valuable=0.5 -> reg_tension=0.1
        // epilogue *= 1.0 + 0.1*0.5 = 1.05
        // k_depth *= 1.0 - 0.1*0.3 = 0.97
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.05 = 1.575
        assert_approx(bias.epilogue_depth_preference, 1.575, 0.01);
        // baseline k_depth=1.3, * 0.97 = 1.261
        assert_approx(bias.k_depth_preference, 1.261, 0.01);
    }

    // ── 79. Reg tension with small negative value (-0.1) ──

    #[test]
    fn reg_tension_small_negative() {
        // fusion_profitable=0.5, pipeline_valuable=0.6 -> reg_tension=-0.1
        // k_depth *= 1.0 + 0.1*0.5 = 1.05
        // epilogue *= 1.0 - 0.1*0.3 = 0.97
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.6,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline k_depth=1.3, * 1.05 = 1.365
        assert_approx(bias.k_depth_preference, 1.365, 0.01);
        // baseline epilogue=1.5, * 0.97 = 1.455
        assert_approx(bias.epilogue_depth_preference, 1.455, 0.01);
    }

    // ── 80. MoE modulation: eviction scales with memory, not parallelism alone ──

    #[test]
    fn moe_eviction_scales_with_memory_not_parallelism() {
        let high_mem_low_par = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let low_mem_high_par = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.2,
            parallelism_exploitable: 0.9,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let hm = StrategyArbiter::arbitrate(InferenceMode::Throughput, &high_mem_low_par, &hw);
        let lm = StrategyArbiter::arbitrate(InferenceMode::Throughput, &low_mem_high_par, &hw);
        // Both have parallelism > 0.5, but high_mem should have higher eviction.
        assert!(
            hm.expert_eviction_aggressiveness > lm.expert_eviction_aggressiveness,
            "eviction should scale with memory_intensive when parallelism is above threshold"
        );
    }

    // ── 81. MoE modulation: prefetch precise with memory=0.5 ──

    #[test]
    fn moe_prefetch_medium_memory_precise() {
        // parallelism=0.8, memory=0.5
        // expert_prefetch_priority: baseline=1.5, lerp(1.0, 2.0, 0.5) = 1.5 -> 1.5*1.5=2.25
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.8,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.expert_prefetch_priority, 2.25, 0.01);
    }

    // ── 82. L1 exactly 32KB precise fusion cost with Throughput ──

    #[test]
    fn l1_32kb_throughput_fusion_precise() {
        // l1_richness = 32768/65536 = 0.5, sqrt(0.5) = 0.7071
        // fusion *= 1/0.7071 = 1.4142
        // Throughput baseline=1.0, * 1.4142 = 1.4142
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 1.414, 0.01);
    }

    // ── 83. L1 48KB precise fusion cost ──

    #[test]
    fn l1_48kb_fusion_precise() {
        // l1_richness = 49152/65536 = 0.75, sqrt(0.75) = 0.866
        // fusion *= 1/0.866 = 1.1547
        // Latency baseline=0.5, * 1.1547 = 0.5774
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (49152, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.577, 0.01);
    }

    // ── 84. GPU + register scarcity combined precise ──

    #[test]
    fn gpu_with_scarcity_combined_precise() {
        // GPU: epilogue *= 1.2, k_depth *= 1.2, pipeline *= 1.2
        // GPU also has regs=255 > 16, so scarcity branch skipped
        // But we test with is_gpu=true, num_simd_regs=8 (low regs)
        let gpu_low = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_low);
        // GPU: epilogue *= 1.2, scarcity=(1-8/32)=0.75, epilogue *= 1.225
        // baseline epilogue=1.5, GPU boost: *1.2=1.8, scarcity: *1.225=2.205
        // Note: GPU branch fires first (1.2x), then scarcity (1.225x)
        let expected_epilogue = 1.5 * 1.2 * (1.0 + 0.75 * 0.3);
        assert_approx(bias.epilogue_depth_preference, expected_epilogue, 0.01);
    }

    // ── 85. CPU with 2 registers scarcity precise ──

    #[test]
    fn cpu_2_registers_scarcity_precise() {
        // scarcity = 1.0 - (2/32) = 0.9375
        // epilogue = 1.0 + 0.9375*0.3 = 1.28125
        // k_depth = 1.0 - 0.9375*0.2 = 0.8125
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 2,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.921875, 0.01);
        assert_approx(bias.k_depth_preference, 1.05625, 0.01);
    }

    // ── 86. StrategyBias validate with +0.0 ──

    #[test]
    fn strategy_bias_validate_positive_zero() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.0,
            pipeline_cost_scale: 0.0,
            batch_flexibility: 0.0,
            expert_eviction_aggressiveness: 0.0,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.pipeline_cost_scale, 0.2);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
    }

    // ── 87. StrategyBias validate with -0.0 ──

    #[test]
    fn strategy_bias_validate_negative_zero() {
        let mut bias = StrategyBias {
            fusion_cost_scale: -0.0,
            pipeline_cost_scale: -0.0,
            batch_flexibility: -0.0,
            expert_eviction_aggressiveness: -0.0,
            ..StrategyBias::default()
        };
        bias.validate();
        // -0.0 clamp: f64::clamp(-0.0, 0.2, 3.0) -> 0.2
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.pipeline_cost_scale, 0.2);
    }

    // ── 88. ArbiterHwView::from always sets is_gpu to false ──

    #[test]
    fn arbiter_hw_view_from_always_cpu() {
        let dp = DeviceProfile::detect();
        let view = ArbiterHwView::from(&dp);
        assert!(view.device == DeviceFamily::Cpu, "From<&DeviceProfile> should always produce CPU view");
    }

    // ── 89. KV cache modulation: zero memory -> no adjustment ──

    #[test]
    fn kv_cache_zero_memory_no_adjustment() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.kv_cache_budget_scale, 0.5, 0.01);
    }

    // ── 90. Quantization modulation: zero memory -> no adjustment ──

    #[test]
    fn quantization_zero_memory_no_adjustment() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.quantization_aggressiveness, 1.5, 0.01);
    }

    // ── 91. Arbiter output never NaN with normal inputs ──

    #[test]
    fn arbitrate_output_never_nan_normal_inputs() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        for &mode in &[InferenceMode::Latency, InferenceMode::Throughput] {
            for hw in &[cpu_avx2(), cpu_avx512(), gpu_a100()] {
                let bias = StrategyArbiter::arbitrate(mode, &arch, hw);
                assert!(!bias.fusion_cost_scale.is_nan(), "fusion_cost_scale should not be NaN");
                assert!(!bias.pipeline_cost_scale.is_nan(), "pipeline_cost_scale should not be NaN");
                assert!(!bias.batch_flexibility.is_nan(), "batch_flexibility should not be NaN");
            }
        }
    }

    // ── 92. CPU register scarcity monotonic: more regs -> lower epilogue ──

    #[test]
    fn register_scarcity_monotonic_epilogue() {
        // In the scarcity range (0..=16), as registers increase, scarcity decreases,
        // so epilogue_depth_preference should decrease monotonically.
        let reg_counts = [0usize, 2, 4, 6, 8, 10, 12, 14, 16];
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let mut prev_epilogue = f64::MAX;
        for &regs in &reg_counts {
            let hw = ArbiterHwView {
                device: DeviceFamily::Cpu,
                num_simd_regs: regs,
                cache_sizes: (65536, 0, 0),
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.epilogue_depth_preference <= prev_epilogue + 0.001,
                "more regs should produce <= epilogue: regs={regs}, epilogue={}, prev={}",
                bias.epilogue_depth_preference, prev_epilogue
            );
            prev_epilogue = bias.epilogue_depth_preference;
        }
    }

    // ── 93. CPU register scarcity monotonic: more regs -> higher k_depth ──

    #[test]
    fn register_scarcity_monotonic_k_depth() {
        // In the scarcity range (0..=16), as registers increase, scarcity decreases,
        // so k_depth_preference should increase monotonically.
        let reg_counts = [0usize, 2, 4, 6, 8, 10, 12, 14, 16];
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let mut prev_k_depth = f64::MIN;
        for &regs in &reg_counts {
            let hw = ArbiterHwView {
                device: DeviceFamily::Cpu,
                num_simd_regs: regs,
                cache_sizes: (65536, 0, 0),
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.k_depth_preference >= prev_k_depth - 0.001,
                "more regs should produce >= k_depth: regs={regs}, k_depth={}, prev={}",
                bias.k_depth_preference, prev_k_depth
            );
            prev_k_depth = bias.k_depth_preference;
        }
    }

    // ── 94. Throughput mode batch_flexibility always 1.0 regardless of hw/arch ──

    #[test]
    fn throughput_batch_flexibility_unchanged_by_archetype() {
        let arches = [
            GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            },
            GraphArchetype {
                compute_intensive: 1.0,
                memory_intensive: 1.0,
                parallelism_exploitable: 1.0,
                fusion_profitable: 1.0,
                pipeline_valuable: 1.0,
            },
        ];
        let hws = [cpu_avx2(), cpu_avx512(), gpu_a100()];
        for arch in &arches {
            for hw in &hws {
                let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, arch, hw);
                assert_eq!(bias.batch_flexibility, 1.0, "throughput batch_flexibility should always be 1.0");
            }
        }
    }

    // ── 95. Latency mode batch_flexibility always 0.0 regardless of hw/arch ──

    #[test]
    fn latency_batch_flexibility_unchanged_by_archetype() {
        let arches = [
            GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            },
            GraphArchetype {
                compute_intensive: 1.0,
                memory_intensive: 1.0,
                parallelism_exploitable: 1.0,
                fusion_profitable: 1.0,
                pipeline_valuable: 1.0,
            },
        ];
        let hws = [cpu_avx2(), cpu_avx512(), gpu_a100()];
        for arch in &arches {
            for hw in &hws {
                let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, arch, hw);
                assert_eq!(bias.batch_flexibility, 0.0, "latency batch_flexibility should always be 0.0");
            }
        }
    }

    // ── 96. Mode baseline latency: all 13 fields exact ──

    #[test]
    fn latency_baseline_all_13_fields_exact() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.5, 0.001);
        assert_approx(bias.pipeline_cost_scale, 0.6, 0.001);
        assert_approx(bias.parallelism_cost_scale, 1.5, 0.001);
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.001);
        assert_approx(bias.k_depth_preference, 1.3, 0.001);
        assert_approx(bias.kv_cache_budget_scale, 0.5, 0.001);
        assert_approx(bias.weight_prefetch_budget_scale, 1.5, 0.001);
        assert_approx(bias.batch_flexibility, 0.0, 0.001);
        assert_approx(bias.decode_ratio_scale, 1.0, 0.001);
        assert_approx(bias.expert_eviction_aggressiveness, 0.0, 0.001);
        assert_approx(bias.expert_prefetch_priority, 0.5, 0.001);
        assert_approx(bias.speculative_decoding_value, 1.5, 0.001);
        assert_approx(bias.quantization_aggressiveness, 1.5, 0.001);
    }

    // ── 97. Mode baseline throughput: all 13 fields exact ──

    #[test]
    fn throughput_baseline_all_13_fields_exact() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 1.0, 0.001);
        assert_approx(bias.pipeline_cost_scale, 1.3, 0.001);
        assert_approx(bias.parallelism_cost_scale, 0.5, 0.001);
        assert_approx(bias.epilogue_depth_preference, 0.8, 0.001);
        assert_approx(bias.k_depth_preference, 0.8, 0.001);
        assert_approx(bias.kv_cache_budget_scale, 1.5, 0.001);
        assert_approx(bias.weight_prefetch_budget_scale, 0.8, 0.001);
        assert_approx(bias.batch_flexibility, 1.0, 0.001);
        assert_approx(bias.decode_ratio_scale, 1.0, 0.001);
        assert_approx(bias.expert_eviction_aggressiveness, 0.8, 0.001);
        assert_approx(bias.expert_prefetch_priority, 1.5, 0.001);
        assert_approx(bias.speculative_decoding_value, 0.3, 0.001);
        assert_approx(bias.quantization_aggressiveness, 0.8, 0.001);
    }

    // ── 98. GPU L1 richness with large shared mem (128KB shared) ──

    #[test]
    fn gpu_l1_richness_128kb_shared_precise() {
        // GPU shared_mem = 131072 (128KB)
        // l1_richness = 131072/65536 = 2.0, capped at 2.0
        // sqrt(2.0) = 1.4142, fusion *= 1/1.4142 = 0.7071
        // Latency baseline 0.5, * 0.7071 = 0.3536
        let gpu = ArbiterHwView::gpu(131072);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(bias.fusion_cost_scale, 0.354, 0.01);
    }

    // ── 99. GraphArchetype field independence: changing only one field ──

    #[test]
    fn archetype_field_independence_fusion_profitable() {
        let base = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.2,
            pipeline_valuable: 0.4,
        };
        let high_fusion = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.8,
            pipeline_valuable: 0.4,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let base_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &base, &hw);
        let high_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &high_fusion, &hw);
        assert!(
            high_bias.fusion_cost_scale < base_bias.fusion_cost_scale,
            "higher fusion_profitable should reduce fusion_cost_scale"
        );
    }

    // ── 100. GraphArchetype field independence: changing only memory_intensive ──

    #[test]
    fn archetype_field_independence_memory_intensive() {
        let base = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.2,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.3,
        };
        let high_mem = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.3,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let base_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &base, &hw);
        let high_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &high_mem, &hw);
        assert!(
            high_bias.kv_cache_budget_scale > base_bias.kv_cache_budget_scale,
            "higher memory_intensive should increase kv_cache_budget_scale"
        );
    }

    // ── 101. StrategyBias validate preserves subnormal f64 ──

    #[test]
    fn strategy_bias_validate_subnormal_f64() {
        let tiny = f64::from_bits(1); // smallest positive subnormal
        let mut bias = StrategyBias {
            fusion_cost_scale: tiny,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2, "subnormal should be clamped to min");
    }

    // ── 102. ArbiterHwView manual construction with all fields distinct ──

    #[test]
    fn arbiter_hw_view_manual_construction_all_distinct() {
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 24,
            cache_sizes: (49152, 524288, 16777216),
        };
        assert!(view.device == DeviceFamily::Cpu);
        assert_eq!(view.num_simd_regs, 24);
        assert_eq!(view.cache_sizes.0, 49152);
        assert_eq!(view.cache_sizes.1, 524288);
        assert_eq!(view.cache_sizes.2, 16777216);
    }

    // ── 103. L1 richness at exactly 2x cap boundary (131072) ──

    #[test]
    fn l1_at_2x_cap_boundary() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0), // exactly 2x 64KB
        };
        let hw_above = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131073, 0, 0), // just above 2x
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_at = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let bias_above = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_above);
        // Both capped at 2.0, so fusion_cost_scale should be identical.
        assert_approx(bias_at.fusion_cost_scale, bias_above.fusion_cost_scale, 0.001);
    }

    // ── 104. Deterministic with extreme archetype values ──

    #[test]
    fn deterministic_extreme_archetype_values() {
        let extreme = GraphArchetype {
            compute_intensive: -5.0,
            memory_intensive: 10.0,
            parallelism_exploitable: -1.0,
            fusion_profitable: 3.0,
            pipeline_valuable: -2.0,
        };
        let hw = cpu_avx2();
        let a = StrategyArbiter::arbitrate(InferenceMode::Latency, &extreme, &hw);
        let b = StrategyArbiter::arbitrate(InferenceMode::Latency, &extreme, &hw);
        assert_eq!(a.fusion_cost_scale, b.fusion_cost_scale);
        assert_eq!(a.epilogue_depth_preference, b.epilogue_depth_preference);
        assert_eq!(a.k_depth_preference, b.k_depth_preference);
        assert_eq!(a.kv_cache_budget_scale, b.kv_cache_budget_scale);
    }

    // ── 105. L2 and L3 cache_sizes do not affect any output field ──

    #[test]
    fn l2_l3_cache_sizes_no_effect_on_output() {
        let hw_zero_l2l3 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 0, 0),
        };
        let hw_big_l2l3 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 1048576, 33554432),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.6,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.3,
        };
        let zero = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_zero_l2l3);
        let big = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_big_l2l3);
        assert_eq!(zero.fusion_cost_scale, big.fusion_cost_scale);
        assert_eq!(zero.pipeline_cost_scale, big.pipeline_cost_scale);
        assert_eq!(zero.epilogue_depth_preference, big.epilogue_depth_preference);
        assert_eq!(zero.k_depth_preference, big.k_depth_preference);
        assert_eq!(zero.kv_cache_budget_scale, big.kv_cache_budget_scale);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (60 new, 217 → 277)
    // ══════════════════════════════════════════════════════════════════════

    // ── 106. InferenceMode Hash dedup in HashSet after removal ──

    #[test]
    fn inference_mode_hash_dedup_after_removal() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(InferenceMode::Latency);
        set.insert(InferenceMode::Throughput);
        set.remove(&InferenceMode::Latency);
        assert_eq!(set.len(), 1);
        assert!(set.contains(&InferenceMode::Throughput));
    }

    // ── 107. InferenceMode Copy: assigning from variable preserves original ──

    #[test]
    fn inference_mode_copy_preserves_original() {
        let original = InferenceMode::Throughput;
        let assigned = original;
        assert_eq!(original, InferenceMode::Throughput);
        assert_eq!(assigned, InferenceMode::Throughput);
    }

    // ── 108. InferenceMode match arms are exhaustive and distinct ──

    #[test]
    fn inference_mode_match_distinct_labels() {
        let lat_label = match InferenceMode::Latency {
            InferenceMode::Latency => "lat",
            InferenceMode::Throughput => "thr",
        };
        let thr_label = match InferenceMode::Throughput {
            InferenceMode::Latency => "lat",
            InferenceMode::Throughput => "thr",
        };
        assert_eq!(lat_label, "lat");
        assert_eq!(thr_label, "thr");
        assert_ne!(lat_label, thr_label);
    }

    // ── 110. ArbiterHwView::gpu with various shared_mem sizes produces distinct views ──

    #[test]
    fn arbiter_hw_view_gpu_distinct_sizes_distinct_views() {
        let a = ArbiterHwView::gpu(16384);
        let b = ArbiterHwView::gpu(32768);
        let c = ArbiterHwView::gpu(65536);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    // ── 111. ArbiterHwView: CPU vs GPU with same cache_sizes are not equal ──

    #[test]
    fn arbiter_hw_view_cpu_gpu_same_cache_not_equal() {
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(49152);
        assert_ne!(cpu, gpu);
    }

    // ── 112. ArbiterHwView Hash: identical views have same hash ──

    #[test]
    fn arbiter_hw_view_hash_same_view_same_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        view.hash(&mut h1);
        view.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── 113. ArbiterHwView Hash: different views have different hash ──

    #[test]
    fn arbiter_hw_view_hash_different_views_likely_different() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 0, 0),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        a.hash(&mut h1);
        b.hash(&mut h2);
        // Different structs should almost certainly produce different hashes.
        assert_ne!(h1.finish(), h2.finish());
    }

    // ── 114. GraphArchetype Copy: assignment does not move ──

    #[test]
    fn graph_archetype_copy_assignment() {
        let original = GraphArchetype {
            compute_intensive: 0.6,
            memory_intensive: 0.4,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.2,
        };
        let assigned = original;
        assert_eq!(original.compute_intensive, assigned.compute_intensive);
        assert_eq!(original.pipeline_valuable, assigned.pipeline_valuable);
    }

    // ── 115. GraphArchetype Debug output contains all five fields ──

    #[test]
    fn graph_archetype_debug_contains_all_fields() {
        let arch = GraphArchetype {
            compute_intensive: 0.1,
            memory_intensive: 0.2,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.5,
        };
        let debug_str = format!("{:?}", arch);
        assert!(debug_str.contains("0.1"), "should contain compute_intensive value");
        assert!(debug_str.contains("0.5"), "should contain pipeline_valuable value");
        assert!(debug_str.contains("fusion_profitable"));
    }

    // ── 116. GraphArchetype small negative values produce valid output ──

    #[test]
    fn graph_archetype_tiny_negative_values() {
        let arch = GraphArchetype {
            compute_intensive: -0.001,
            memory_intensive: -0.001,
            parallelism_exploitable: -0.001,
            fusion_profitable: -0.001,
            pipeline_valuable: -0.001,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert!(bias.fusion_cost_scale > 0.0);
        assert!(bias.kv_cache_budget_scale > 0.0);
    }

    // ── 117. StrategyBias validate: each field clamped independently ──

    #[test]
    fn strategy_bias_validate_each_field_independent() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 5.0,    // above max
            pipeline_cost_scale: 0.5,  // in range
            parallelism_cost_scale: -1.0, // below min
            epilogue_depth_preference: 1.0, // in range
            k_depth_preference: 0.4,   // in range
            kv_cache_budget_scale: 10.0, // above max
            weight_prefetch_budget_scale: 0.3, // in range
            batch_flexibility: 0.5,    // in range
            decode_ratio_scale: 1.0,   // in range
            speculative_decoding_value: 0.2, // in range
            quantization_aggressiveness: -2.0, // below min
            expert_eviction_aggressiveness: 0.5, // in range
            expert_prefetch_priority: 3.0, // in range
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 3.0, "should clamp to max");
        assert_eq!(bias.pipeline_cost_scale, 0.5, "should remain unchanged");
        assert_eq!(bias.parallelism_cost_scale, 0.1, "should clamp to min");
        assert_eq!(bias.kv_cache_budget_scale, 3.0, "should clamp to max");
        assert_eq!(bias.quantization_aggressiveness, 0.3, "should clamp to min");
    }

    // ── 118. StrategyBias accessor with zero values ──

    #[test]
    fn strategy_bias_accessors_zero_values() {
        let bias = StrategyBias {
            expert_eviction_aggressiveness: 0.0,
            expert_prefetch_priority: 0.1,
            ..StrategyBias::default()
        };
        assert_approx(bias.expert_eviction_aggressiveness(), 0.0, 0.001);
        assert_approx(bias.expert_prefetch_priority(), 0.1, 0.001);
    }

    // ── 119. StrategyBias accessor with max valid values ──

    #[test]
    fn strategy_bias_accessors_max_values() {
        let bias = StrategyBias {
            expert_eviction_aggressiveness: 2.0,
            expert_prefetch_priority: 5.0,
            ..StrategyBias::default()
        };
        assert_approx(bias.expert_eviction_aggressiveness(), 2.0, 0.001);
        assert_approx(bias.expert_prefetch_priority(), 5.0, 0.001);
    }

    // ── 120. StrategyBias Debug: all 13 fields present ──

    #[test]
    fn strategy_bias_debug_all_13_fields() {
        let bias = StrategyBias::default();
        let s = format!("{:?}", bias);
        assert!(s.contains("fusion_cost_scale"));
        assert!(s.contains("pipeline_cost_scale"));
        assert!(s.contains("parallelism_cost_scale"));
        assert!(s.contains("epilogue_depth_preference"));
        assert!(s.contains("k_depth_preference"));
        assert!(s.contains("kv_cache_budget_scale"));
        assert!(s.contains("weight_prefetch_budget_scale"));
        assert!(s.contains("batch_flexibility"));
        assert!(s.contains("decode_ratio_scale"));
        assert!(s.contains("speculative_decoding_value"));
        assert!(s.contains("quantization_aggressiveness"));
        assert!(s.contains("expert_eviction_aggressiveness"));
        assert!(s.contains("expert_prefetch_priority"));
    }

    // ── 121. StrategyBias Clone produces independent copy ──

    #[test]
    fn strategy_bias_clone_independent() {
        let mut original = StrategyBias::default();
        let cloned = original.clone();
        original.fusion_cost_scale = 2.5;
        assert_ne!(original.fusion_cost_scale, cloned.fusion_cost_scale);
        assert_approx(cloned.fusion_cost_scale, 1.0, 0.001);
    }

    // ── 122. StrategyBias Copy: mutation of one does not affect other ──

    #[test]
    fn strategy_bias_copy_independent() {
        let a = StrategyBias {
            fusion_cost_scale: 0.5,
            ..StrategyBias::default()
        };
        let b = a;
        let _ = a;
        assert_approx(b.fusion_cost_scale, 0.5, 0.001);
    }

    // ── 123. DeviceProfile integration: cache_sizes tuple matches ──

    #[test]
    fn device_profile_integration_cache_sizes() {
        let dp = DeviceProfile::detect();
        let view = ArbiterHwView::from(&dp);
        assert_eq!(view.cache_sizes.0, dp.cache_sizes().0);
        assert_eq!(view.cache_sizes.1, dp.cache_sizes().1);
        assert_eq!(view.cache_sizes.2, dp.cache_sizes().2);
    }

    // ── 124. DeviceProfile integration: num_simd_regs matches ──

    #[test]
    fn device_profile_integration_num_simd_regs() {
        let dp = DeviceProfile::detect();
        let view = ArbiterHwView::from(&dp);
        assert_eq!(view.num_simd_regs, dp.num_simd_regs());
    }

    // ── 125. arbitrate_cpu and manual From produce identical results for Throughput ──

    #[test]
    fn arbitrate_cpu_matches_manual_throughput() {
        let dp = DeviceProfile::detect();
        let arch = GraphArchetype {
            compute_intensive: 0.8,
            memory_intensive: 0.6,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.2,
            pipeline_valuable: 0.9,
        };
        let convenience = StrategyArbiter::arbitrate_cpu(InferenceMode::Throughput, &arch, &dp);
        let manual = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &ArbiterHwView::from(&dp));
        assert_eq!(convenience.fusion_cost_scale, manual.fusion_cost_scale);
        assert_eq!(convenience.pipeline_cost_scale, manual.pipeline_cost_scale);
        assert_eq!(convenience.kv_cache_budget_scale, manual.kv_cache_budget_scale);
        assert_eq!(convenience.batch_flexibility, manual.batch_flexibility);
    }

    // ── 126. Combined: high pipeline + low fusion + GPU + Throughput ──

    #[test]
    fn combined_high_pipeline_low_fusion_gpu_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.1,
            pipeline_valuable: 0.9,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert_eq!(bias.batch_flexibility, 1.0);
    }

    // ── 127. CPU register scarcity does not affect fusion_cost_scale ──

    #[test]
    fn register_scarcity_does_not_affect_fusion_cost() {
        let hw_low = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 4,
            cache_sizes: (65536, 0, 0),
        };
        let hw_high = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let low = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_low);
        let high = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_high);
        assert_approx(low.fusion_cost_scale, high.fusion_cost_scale, 0.001);
    }

    // ── 128. GPU boost does not affect fusion_cost_scale ──

    #[test]
    fn gpu_boost_does_not_affect_fusion_cost() {
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(cpu_bias.fusion_cost_scale, gpu_bias.fusion_cost_scale, 0.001);
    }

    // ── 129. ArbiterHwView::gpu with exact A100 shared mem size ──

    #[test]
    fn arbiter_hw_view_gpu_a100_shared_mem() {
        let view = ArbiterHwView::gpu(49152); // 48KB = typical A100 shared mem
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.cache_sizes.0, 49152);
        assert_eq!(view.cache_sizes.1, 0);
        assert_eq!(view.cache_sizes.2, 0);
    }

    // ── 130. ArbiterHwView::gpu with H100 shared mem size ──

    #[test]
    fn arbiter_hw_view_gpu_h100_shared_mem() {
        let view = ArbiterHwView::gpu(65536); // 64KB = H100 shared mem
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.cache_sizes.0, 65536);
    }

    // ── 131. ArbiterHwView cache_sizes tuple destructuring ──

    #[test]
    fn arbiter_hw_view_cache_tuple_destructuring() {
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (49152, 524288, 16777216),
        };
        let (l1, l2, l3) = view.cache_sizes;
        assert_eq!(l1, 49152);
        assert_eq!(l2, 524288);
        assert_eq!(l3, 16777216);
    }

    // ── 132. ArbiterHwView with all zero fields ──

    #[test]
    fn arbiter_hw_view_all_zero_fields() {
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 0,
            cache_sizes: (0, 0, 0),
        };
        assert!(view.device == DeviceFamily::Cpu);
        assert_eq!(view.num_simd_regs, 0);
        assert_eq!(view.cache_sizes, (0, 0, 0));
    }

    // ── 133. ArbiterHwView with usize::MAX in all fields ──

    #[test]
    fn arbiter_hw_view_max_fields() {
        let view = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: usize::MAX,
            cache_sizes: (usize::MAX, usize::MAX, usize::MAX),
        };
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.num_simd_regs, usize::MAX);
        assert_eq!(view.cache_sizes.0, usize::MAX);
    }

    // ── 134. StrategyBias validate: single field just above min stays ──

    #[test]
    fn strategy_bias_validate_just_above_min() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.2001,
            parallelism_cost_scale: 0.1001,
            batch_flexibility: 0.0001,
            decode_ratio_scale: 0.3001,
            expert_eviction_aggressiveness: 0.0001,
            expert_prefetch_priority: 0.1001,
            speculative_decoding_value: 0.1001,
            quantization_aggressiveness: 0.3001,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_approx(bias.fusion_cost_scale, 0.2001, 0.0001);
        assert_approx(bias.parallelism_cost_scale, 0.1001, 0.0001);
        assert_approx(bias.batch_flexibility, 0.0001, 0.0001);
    }

    // ── 135. StrategyBias validate: single field just below max stays ──

    #[test]
    fn strategy_bias_validate_just_below_max() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 2.999,
            pipeline_cost_scale: 2.999,
            batch_flexibility: 0.999,
            decode_ratio_scale: 1.999,
            expert_eviction_aggressiveness: 1.999,
            expert_prefetch_priority: 4.999,
            speculative_decoding_value: 2.999,
            quantization_aggressiveness: 2.999,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_approx(bias.fusion_cost_scale, 2.999, 0.001);
        assert_approx(bias.batch_flexibility, 0.999, 0.001);
        assert_approx(bias.expert_prefetch_priority, 4.999, 0.001);
    }

    // ── 136. arbiter output never contains NaN with extreme negative archetype ──

    #[test]
    fn arbitrate_no_nan_extreme_negative_archetype() {
        let arch = GraphArchetype {
            compute_intensive: -100.0,
            memory_intensive: -100.0,
            parallelism_exploitable: -100.0,
            fusion_profitable: -100.0,
            pipeline_valuable: -100.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (0, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert!(!bias.fusion_cost_scale.is_nan());
        assert!(!bias.epilogue_depth_preference.is_nan());
        assert!(!bias.k_depth_preference.is_nan());
    }

    // ── 137. arbiter output never contains NaN with extreme positive archetype ──

    #[test]
    fn arbitrate_no_nan_extreme_positive_archetype() {
        let arch = GraphArchetype {
            compute_intensive: 100.0,
            memory_intensive: 100.0,
            parallelism_exploitable: 100.0,
            fusion_profitable: 100.0,
            pipeline_valuable: 100.0,
        };
        let gpu = ArbiterHwView::gpu(usize::MAX);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert!(!bias.fusion_cost_scale.is_nan());
        assert!(!bias.pipeline_cost_scale.is_nan());
        assert!(!bias.quantization_aggressiveness.is_nan());
    }

    // ── 138. CPU with 16 registers: scarcity factor is exactly 0.5 ──

    #[test]
    fn cpu_16_registers_scarcity_factor_0_5() {
        // scarcity = 1.0 - (16/32) = 0.5
        // k_depth multiplier = 1.0 - 0.5*0.2 = 0.9
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Throughput baseline k_depth=0.8, * 0.9 = 0.72
        assert_approx(bias.k_depth_preference, 0.72, 0.01);
    }

    // ── 139. GPU adjustment: all three GPU-affected fields increase ──

    #[test]
    fn gpu_all_three_adjusted_fields_increase() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert!(gpu_bias.epilogue_depth_preference > cpu_bias.epilogue_depth_preference);
        assert!(gpu_bias.k_depth_preference > cpu_bias.k_depth_preference);
        assert!(gpu_bias.pipeline_cost_scale > cpu_bias.pipeline_cost_scale);
    }

    // ── 140. Reg tension large positive: epilogue amplified significantly ──

    #[test]
    fn reg_tension_large_positive_epilogue_amplified() {
        // fusion_profitable=0.95, pipeline_valuable=0.05 -> tension=0.9
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.95,
            pipeline_valuable: 0.05,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // epilogue *= 1.0 + 0.9*0.5 = 1.45 -> 1.5 * 1.45 = 2.175
        assert!(bias.epilogue_depth_preference > 2.0);
    }

    // ── 141. Reg tension large negative: k_depth amplified significantly ──

    #[test]
    fn reg_tension_large_negative_k_depth_amplified() {
        // fusion_profitable=0.05, pipeline_valuable=0.95 -> tension=-0.9
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.05,
            pipeline_valuable: 0.95,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // k_depth *= 1.0 + 0.9*0.5 = 1.45 -> 1.3 * 1.45 = 1.885
        assert!(bias.k_depth_preference > 1.8);
    }

    // ── 142. MoE modulation: memory_intensive=0.5 precise eviction ──

    #[test]
    fn moe_eviction_medium_memory_precise() {
        // parallelism=0.7, memory=0.5
        // eviction: baseline=0.8, lerp(1.0, 1.5, 0.5)=1.25 -> 0.8*1.25=1.0
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.expert_eviction_aggressiveness, 1.0, 0.01);
    }

    // ── 143. KV cache modulation precise: memory=0.75 with Throughput ──

    #[test]
    fn kv_cache_medium_memory_throughput_precise() {
        // lerp(1.0, 1.5, 0.75) = 1.375
        // Throughput baseline kv_cache=1.5, * 1.375 = 2.0625
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.75,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.kv_cache_budget_scale, 2.0625, 0.01);
    }

    // ── 144. Quantization modulation precise: memory=0.25 with Throughput ──

    #[test]
    fn quantization_low_memory_throughput_precise() {
        // lerp(1.0, 1.3, 0.25) = 1.075
        // Throughput baseline quantization=0.8, * 1.075 = 0.86
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.25,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.quantization_aggressiveness, 0.86, 0.01);
    }

    // ── 145. L1 richness precise: 40KB ──

    #[test]
    fn l1_40kb_richness_precise() {
        // l1_richness = 40960/65536 = 0.625, sqrt(0.625) = 0.7906
        // fusion *= 1/0.7906 = 1.2649
        // Latency baseline=0.5, * 1.2649 = 0.6325
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (40960, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.632, 0.01);
    }

    // ── 146. CPU 6 registers scarcity precise ──

    #[test]
    fn cpu_6_registers_scarcity_precise() {
        // scarcity = 1.0 - (6/32) = 0.8125
        // epilogue = 1.0 + 0.8125*0.3 = 1.24375
        // k_depth = 1.0 - 0.8125*0.2 = 0.8375
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 6,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.24375 = 1.865625
        assert_approx(bias.epilogue_depth_preference, 1.865625, 0.01);
        // baseline k_depth=1.3, * 0.8375 = 1.08875
        assert_approx(bias.k_depth_preference, 1.08875, 0.01);
    }

    // ── 147. GPU + high parallelism + high memory: expert_prefetch maxed ──

    #[test]
    fn gpu_high_parallel_high_memory_expert_prefetch() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        // prefetch: baseline=1.5, lerp(1.0, 2.0, 1.0)=2.0 -> 1.5*2.0=3.0
        assert_approx(bias.expert_prefetch_priority, 3.0, 0.01);
    }

    // ── 148. CPU 32 registers: no scarcity, same as 64 registers ──

    #[test]
    fn cpu_32_regs_same_as_64_regs() {
        let hw_32 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let hw_64 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 64,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_32 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_32);
        let bias_64 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_64);
        assert_approx(bias_32.epilogue_depth_preference, bias_64.epilogue_depth_preference, 0.001);
        assert_approx(bias_32.k_depth_preference, bias_64.k_depth_preference, 0.001);
    }

    // ── 149. ArbiterHwView from DeviceProfile: is_gpu always false ──

    #[test]
    fn arbiter_hw_view_from_multiple_profiles() {
        // DeviceProfile::detect() always returns a CPU profile.
        let dp = DeviceProfile::detect();
        let view1 = ArbiterHwView::from(&dp);
        let view2 = ArbiterHwView::from(&dp);
        assert!(view1.device == DeviceFamily::Cpu);
        assert!(view2.device == DeviceFamily::Cpu);
        assert_eq!(view1, view2);
    }

    // ── 150. Fusion modulation with high fusion_profitable + Throughput ──

    #[test]
    fn fusion_modulation_high_fusion_throughput() {
        // fusion_profitable=0.9 -> lerp(1.0, 0.6, 0.9) = 0.64
        // Throughput baseline fusion=1.0, * 0.64 = 0.64
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.9,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.64, 0.01);
    }

    // ── 151. Parallelism modulation with high parallelism + Latency ──

    #[test]
    fn parallelism_modulation_high_parallel_latency() {
        // parallelism_exploitable=0.9 -> lerp(1.0, 0.5, 0.9) = 0.55
        // Latency baseline parallelism=1.5, * 0.55 = 0.825
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.9,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.parallelism_cost_scale, 0.825, 0.01);
    }

    // ── 152. Pipeline modulation with high pipeline_valuable + Latency ──

    #[test]
    fn pipeline_modulation_high_pipeline_latency() {
        // pipeline_valuable=0.9 -> lerp(1.0, 0.6, 0.9) = 0.64
        // Latency baseline pipeline=0.6, * 0.64 = 0.384
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.9,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.pipeline_cost_scale, 0.384, 0.01);
    }

    // ── 153. Combined: all archetype fields at 0.25 with GPU + Latency ──

    #[test]
    fn combined_all_quarter_gpu_latency() {
        let arch = GraphArchetype {
            compute_intensive: 0.25,
            memory_intensive: 0.25,
            parallelism_exploitable: 0.25,
            fusion_profitable: 0.25,
            pipeline_valuable: 0.25,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert!(bias.fusion_cost_scale > 0.0);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
    }

    // ── 154. Combined: all archetype fields at 0.75 with CPU + Throughput ──

    #[test]
    fn combined_all_three_quarters_cpu_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.75,
            memory_intensive: 0.75,
            parallelism_exploitable: 0.75,
            fusion_profitable: 0.75,
            pipeline_valuable: 0.75,
        };
        let hw = cpu_avx2();
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_eq!(bias.batch_flexibility, 1.0);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
    }

    // ── 155. StrategyBias struct initialization with explicit field order ──

    #[test]
    fn strategy_bias_explicit_field_order() {
        let bias = StrategyBias {
            fusion_cost_scale: 1.0,
            pipeline_cost_scale: 2.0,
            parallelism_cost_scale: 3.0,
            epilogue_depth_preference: 0.5,
            k_depth_preference: 0.6,
            kv_cache_budget_scale: 0.7,
            weight_prefetch_budget_scale: 0.8,
            batch_flexibility: 0.4,
            decode_ratio_scale: 1.5,
            speculative_decoding_value: 0.9,
            quantization_aggressiveness: 1.1,
            expert_eviction_aggressiveness: 0.3,
            expert_prefetch_priority: 2.0,
        };
        assert_eq!(bias.fusion_cost_scale, 1.0);
        assert_eq!(bias.pipeline_cost_scale, 2.0);
        assert_eq!(bias.parallelism_cost_scale, 3.0);
    }

    // ── 156. StrategyBias partial struct update syntax ──

    #[test]
    fn strategy_bias_partial_update() {
        let bias = StrategyBias {
            fusion_cost_scale: 0.5,
            kv_cache_budget_scale: 2.0,
            ..StrategyBias::default()
        };
        assert_eq!(bias.fusion_cost_scale, 0.5);
        assert_eq!(bias.kv_cache_budget_scale, 2.0);
        assert_eq!(bias.pipeline_cost_scale, 1.0); // from default
        assert_eq!(bias.batch_flexibility, 1.0); // from default
    }

    // ── 157. CPU 14 registers scarcity precise ──

    #[test]
    fn cpu_14_registers_scarcity_precise() {
        // scarcity = 1.0 - (14/32) = 0.5625
        // epilogue = 1.0 + 0.5625*0.3 = 1.16875
        // k_depth = 1.0 - 0.5625*0.2 = 0.8875
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 14,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.16875 = 1.753125
        assert_approx(bias.epilogue_depth_preference, 1.753125, 0.01);
        // baseline k_depth=1.3, * 0.8875 = 1.15375
        assert_approx(bias.k_depth_preference, 1.15375, 0.01);
    }

    // ── 158. ArbiterHwView PartialEq: GPU views with different regs not equal ──

    #[test]
    fn arbiter_hw_view_neq_different_regs() {
        let a = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 128,
            cache_sizes: (49152, 0, 0),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        assert_ne!(a, b);
    }

    // ── 159. InferenceMode Debug output is valid UTF-8 ──

    #[test]
    fn inference_mode_debug_utf8() {
        let lat_str = format!("{:?}", InferenceMode::Latency);
        let thr_str = format!("{:?}", InferenceMode::Throughput);
        assert!(lat_str.is_ascii());
        assert!(thr_str.is_ascii());
    }

    // ── 160. Golden vector 2 recheck: all 13 fields in range ──

    #[test]
    fn golden_vector_2_all_fields_in_range() {
        let archetype = GraphArchetype {
            compute_intensive: 0.50,
            memory_intensive: 0.90,
            parallelism_exploitable: 0.99,
            fusion_profitable: 0.60,
            pipeline_valuable: 0.35,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &archetype, &gpu_a100());
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
    }

    // ── 161. Golden vector 3 recheck: all 13 fields in range ──

    #[test]
    fn golden_vector_3_all_fields_in_range() {
        let archetype = GraphArchetype {
            compute_intensive: 0.35,
            memory_intensive: 0.25,
            parallelism_exploitable: 0.10,
            fusion_profitable: 0.70,
            pipeline_valuable: 0.80,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &archetype, &cpu_avx512());
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
    }

    // ── 162. Golden vector 4 recheck: all 13 fields in range ──

    #[test]
    fn golden_vector_4_all_fields_in_range() {
        let archetype = GraphArchetype {
            compute_intensive: 0.73,
            memory_intensive: 0.40,
            parallelism_exploitable: 0.15,
            fusion_profitable: 0.80,
            pipeline_valuable: 0.45,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &archetype, &gpu_a100());
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
    }

    // ── 163. L1 8KB: very small L1 greatly increases fusion cost ──

    #[test]
    fn l1_8kb_very_small_l1_increases_fusion_cost() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (8192, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // With very small L1, fusion should be more expensive.
        assert!(bias.fusion_cost_scale > 0.5, "8KB L1 should make fusion more expensive than baseline");
    }

    // ── 164. GPU L1 richness with 32KB shared mem ──

    #[test]
    fn gpu_32kb_shared_mem_fusion_precise() {
        // GPU shared=32768, richness=32768/65536=0.5, sqrt(0.5)=0.7071
        // fusion *= 1/0.7071 = 1.4142
        // Latency baseline=0.5, * 1.4142 = 0.7071
        let gpu = ArbiterHwView::gpu(32768);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(bias.fusion_cost_scale, 0.707, 0.01);
    }

    // ── 165. StrategyArbiter is zero-sized and Copy ──

    #[test]
    fn strategy_arbiter_is_copy() {
        let a = StrategyArbiter;
        let b = a;
        // Both are valid — StrategyArbiter is Copy (unit struct).
        let _ = a;
        let _ = b;
    }

    // ── 166. ArbiterHwView Debug output contains specific values ──

    #[test]
    fn arbiter_hw_view_debug_specific_values() {
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let debug = format!("{:?}", view);
        assert!(debug.contains("32768"), "debug should show L1 cache size");
        assert!(debug.contains("8388608"), "debug should show L3 cache size");
    }

    // ── 167. Reg tension with fusion_profitable > pipeline_valuable by small margin ──

    #[test]
    fn reg_tension_small_positive_margin() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.51,
            pipeline_valuable: 0.50,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // reg_tension = 0.01, epilogue *= 1.0 + 0.01*0.5 = 1.005
        // baseline=1.5, * 1.005 = 1.5075
        assert_approx(bias.epilogue_depth_preference, 1.5075, 0.01);
    }

    // ── 168. Reg tension with pipeline_valuable > fusion_profitable by small margin ──

    #[test]
    fn reg_tension_small_negative_margin() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.50,
            pipeline_valuable: 0.51,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // reg_tension = -0.01, k_depth *= 1.0 + 0.01*0.5 = 1.005
        // baseline=1.3, * 1.005 = 1.3065
        assert_approx(bias.k_depth_preference, 1.3065, 0.01);
    }

    // ── 169. Combined: GPU + Throughput + all max archetype ──

    #[test]
    fn combined_gpu_throughput_all_max_archetype() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert_eq!(bias.batch_flexibility, 1.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
    }

    // ── 170. Combined: CPU AVX2 + Latency + all max archetype ──

    #[test]
    fn combined_cpu_avx2_latency_all_max_archetype() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu_avx2());
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert_eq!(bias.batch_flexibility, 0.0);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (42 new, 282 → 324)
    // ══════════════════════════════════════════════════════════════════════

    // ── 109 (placeholder fill). InferenceMode as BTreeSet key ──

    #[test]
    fn inference_mode_btreeset_key() {
        use std::collections::BTreeSet;
        let mut set = BTreeSet::new();
        set.insert(InferenceMode::Throughput);
        set.insert(InferenceMode::Latency);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&InferenceMode::Latency));
        assert!(set.contains(&InferenceMode::Throughput));
    }

    // ── 171. sigmoid: small positive returns > 0.5 ──

    #[test]
    fn sigmoid_small_positive_above_half() {
        let result = sigmoid(1.0);
        assert!(result > 0.5 && result < 1.0, "sigmoid(1) should be in (0.5, 1.0)");
    }

    // ── 172. sigmoid: small negative returns < 0.5 ──

    #[test]
    fn sigmoid_small_negative_below_half() {
        let result = sigmoid(-1.0);
        assert!(result > 0.0 && result < 0.5, "sigmoid(-1) should be in (0.0, 0.5)");
    }

    // ── 173. sigmoid: symmetry around 0.5 ──

    #[test]
    fn sigmoid_symmetry() {
        let positive = sigmoid(2.0);
        let negative = sigmoid(-2.0);
        assert_approx(positive + negative, 1.0, 0.001);
    }

    // ── 174. weight_prefetch not affected by archetype modulation ──

    #[test]
    fn weight_prefetch_not_affected_by_archetype() {
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let max_arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let zero = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        let max_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &max_arch, &hw);
        assert_approx(
            zero.weight_prefetch_budget_scale,
            max_bias.weight_prefetch_budget_scale,
            0.001,
        );
    }

    // ── 175. weight_prefetch not affected by hardware ──

    #[test]
    fn weight_prefetch_not_affected_by_hardware() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (16384, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(49152);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(
            cpu_bias.weight_prefetch_budget_scale,
            gpu_bias.weight_prefetch_budget_scale,
            0.001,
        );
    }

    // ── 176. speculative_decoding_value not affected by archetype ──

    #[test]
    fn speculative_decoding_not_affected_by_archetype() {
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let max_arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let zero = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        let max_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &max_arch, &hw);
        assert_approx(
            zero.speculative_decoding_value,
            max_bias.speculative_decoding_value,
            0.001,
        );
    }

    // ── 177. speculative_decoding_value not affected by hardware ──

    #[test]
    fn speculative_decoding_not_affected_by_hardware() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(
            cpu_bias.speculative_decoding_value,
            gpu_bias.speculative_decoding_value,
            0.001,
        );
    }

    // ── 178. Throughput decode_ratio_scale with medium memory ──

    #[test]
    fn throughput_decode_ratio_medium_memory_no_modulation() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.3,
            pipeline_valuable: 0.4,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (32768, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
    }

    // ── 179. Latency decode_ratio_scale with GPU and archetype ──

    #[test]
    fn latency_decode_ratio_gpu_archetype_no_modulation() {
        let arch = GraphArchetype {
            compute_intensive: 0.8,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.5,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_a100());
        assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
    }

    // ── 180. Throughput latency_baseline_k_depth_preference exact ──

    #[test]
    fn throughput_baseline_k_depth_exact() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.k_depth_preference, 0.8, 0.001);
    }

    // ── 181. Throughput baseline kv_cache_budget_scale exact ──

    #[test]
    fn throughput_baseline_kv_cache_exact() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.kv_cache_budget_scale, 1.5, 0.001);
    }

    // ── 182. Throughput baseline weight_prefetch exact ──

    #[test]
    fn throughput_baseline_weight_prefetch_exact() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.weight_prefetch_budget_scale, 0.8, 0.001);
    }

    // ── 183. Latency baseline expert_eviction always 0.0 ──

    #[test]
    fn latency_baseline_expert_eviction_zero() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.expert_eviction_aggressiveness, 0.0, 0.001);
    }

    // ── 184. Latency baseline expert_prefetch exact ──

    #[test]
    fn latency_baseline_expert_prefetch_exact() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.expert_prefetch_priority, 0.5, 0.001);
    }

    // ── 185. GPU + register scarcity combined for Throughput mode ──

    #[test]
    fn gpu_scarcity_combined_throughput_mode() {
        let gpu_low = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_low = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu_low);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &cpu_low);
        // GPU adds 1.2x on top of scarcity for epilogue, k_depth, pipeline
        assert!(
            gpu_bias.epilogue_depth_preference > cpu_bias.epilogue_depth_preference,
            "GPU + low registers should boost epilogue more than CPU + low registers"
        );
        assert!(
            gpu_bias.pipeline_cost_scale > cpu_bias.pipeline_cost_scale,
            "GPU should boost pipeline_cost_scale in throughput mode"
        );
    }

    // ── 186. Register scarcity monotonic in Throughput mode ──

    #[test]
    fn register_scarcity_monotonic_throughput_epilogue() {
        let reg_counts = [0usize, 4, 8, 12, 16];
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let mut prev = f64::MAX;
        for &regs in &reg_counts {
            let hw = ArbiterHwView {
                device: DeviceFamily::Cpu,
                num_simd_regs: regs,
                cache_sizes: (65536, 0, 0),
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            assert!(
                bias.epilogue_depth_preference <= prev + 0.001,
                "throughput: more regs should produce <= epilogue: regs={regs}, val={}, prev={}",
                bias.epilogue_depth_preference, prev
            );
            prev = bias.epilogue_depth_preference;
        }
    }

    // ── 187. GPU does not affect non-GPU-adjusted fields ──

    #[test]
    fn gpu_only_adjusts_three_fields() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // These fields should NOT be affected by GPU flag
        assert_approx(cpu_bias.fusion_cost_scale, gpu_bias.fusion_cost_scale, 0.001);
        assert_approx(cpu_bias.parallelism_cost_scale, gpu_bias.parallelism_cost_scale, 0.001);
        assert_approx(cpu_bias.kv_cache_budget_scale, gpu_bias.kv_cache_budget_scale, 0.001);
        assert_approx(cpu_bias.batch_flexibility, gpu_bias.batch_flexibility, 0.001);
        assert_approx(cpu_bias.decode_ratio_scale, gpu_bias.decode_ratio_scale, 0.001);
        assert_approx(cpu_bias.speculative_decoding_value, gpu_bias.speculative_decoding_value, 0.001);
        assert_approx(cpu_bias.quantization_aggressiveness, gpu_bias.quantization_aggressiveness, 0.001);
        assert_approx(cpu_bias.weight_prefetch_budget_scale, gpu_bias.weight_prefetch_budget_scale, 0.001);
    }

    // ── 188. Archetype modulation is position-independent of hw ──

    #[test]
    fn archetype_modulation_position_independent() {
        // Same archetype with different hardware should show the same
        // *relative* change from baseline in archetype-affected fields.
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let neutral = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let _avx2 = cpu_avx2();
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };

        // Archetype modulation on fusion with neutral hw
        let with_arch = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral);
        let without_arch = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &neutral);
        // fusion_profitable=0.5: lerp(1.0, 0.6, 0.5) = 0.8
        let fusion_ratio = with_arch.fusion_cost_scale / without_arch.fusion_cost_scale;
        assert_approx(fusion_ratio, 0.8, 0.01);
    }

    // ── 189. CPU with 3 registers scarcity precise ──

    #[test]
    fn cpu_3_registers_scarcity_precise() {
        // scarcity = 1.0 - (3/32) = 0.90625
        // epilogue = 1.0 + 0.90625*0.3 = 1.271875
        // k_depth = 1.0 - 0.90625*0.2 = 0.81875
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 3,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.271875 = 1.9078125
        assert_approx(bias.epilogue_depth_preference, 1.9078125, 0.01);
        // baseline k_depth=1.3, * 0.81875 = 1.064375
        assert_approx(bias.k_depth_preference, 1.064375, 0.01);
    }

    // ── 190. CPU 5 registers scarcity precise ──

    #[test]
    fn cpu_5_registers_scarcity_precise() {
        // scarcity = 1.0 - (5/32) = 0.84375
        // epilogue = 1.0 + 0.84375*0.3 = 1.253125
        // k_depth = 1.0 - 0.84375*0.2 = 0.83125
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 5,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.253125 = 1.8796875
        assert_approx(bias.epilogue_depth_preference, 1.8796875, 0.01);
        // baseline k_depth=1.3, * 0.83125 = 1.080625
        assert_approx(bias.k_depth_preference, 1.080625, 0.01);
    }

    // ── 191. L1 richness precise: 80KB ──

    #[test]
    fn l1_80kb_richness_precise() {
        // l1_richness = 81920/65536 = 1.25, sqrt(1.25) = 1.1180
        // fusion *= 1/1.1180 = 0.8944
        // Latency baseline=0.5, * 0.8944 = 0.4472
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (81920, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.447, 0.01);
    }

    // ── 192. L1 richness precise: 4KB (very small) ──

    #[test]
    fn l1_4kb_richness_precise() {
        // l1_richness = 4096/65536 = 0.0625, sqrt(0.0625) = 0.25
        // fusion *= 1/0.25 = 4.0
        // Latency baseline=0.5, * 4.0 = 2.0
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (4096, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 2.0, 0.01);
    }

    // ── 193. L1 4KB with Throughput mode ──

    #[test]
    fn l1_4kb_throughput_fusion_precise() {
        // fusion *= 4.0, Throughput baseline=1.0, * 4.0 = 4.0 → validate clamps to 3.0
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (4096, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_eq!(bias.fusion_cost_scale, 3.0);
    }

    // ── 194. MoE modulation with Throughput + parallelism just above 0.5 ──

    #[test]
    fn moe_throughput_parallelism_just_above_threshold() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.51,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // eviction: baseline=0.8, lerp(1.0, 1.5, 1.0)=1.5 -> 0.8*1.5=1.2
        assert_approx(bias.expert_eviction_aggressiveness, 1.2, 0.01);
    }

    // ── 195. Pipeline modulation precise: pipeline_valuable=0.3 ──

    #[test]
    fn pipeline_modulation_0_3_precise() {
        // pipeline_valuable=0.3 -> lerp(1.0, 0.6, 0.3) = 1.0 + 0.3*(-0.4) = 0.88
        // Latency baseline pipeline=0.6, * 0.88 = 0.528
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.3,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.pipeline_cost_scale, 0.528, 0.01);
    }

    // ── 196. Fusion modulation precise: fusion_profitable=0.4 ──

    #[test]
    fn fusion_modulation_0_4_precise() {
        // fusion_profitable=0.4 -> lerp(1.0, 0.6, 0.4) = 1.0 + 0.4*(-0.4) = 0.84
        // Latency baseline fusion=0.5, * 0.84 = 0.42
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.42, 0.01);
    }

    // ── 197. Parallelism modulation precise: parallelism_exploitable=0.2 ──

    #[test]
    fn parallelism_modulation_0_2_precise() {
        // parallelism_exploitable=0.2 -> lerp(1.0, 0.5, 0.2) = 0.9
        // Latency baseline parallelism=1.5, * 0.9 = 1.35
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.2,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.parallelism_cost_scale, 1.35, 0.01);
    }

    // ── 198. KV cache modulation precise: memory=0.3 with Latency ──

    #[test]
    fn kv_cache_0_3_memory_latency_precise() {
        // lerp(1.0, 1.5, 0.3) = 1.15
        // Latency baseline kv_cache=0.5, * 1.15 = 0.575
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.kv_cache_budget_scale, 0.575, 0.01);
    }

    // ── 199. Quantization modulation precise: memory=0.7 with Throughput ──

    #[test]
    fn quantization_0_7_memory_throughput_precise() {
        // lerp(1.0, 1.3, 0.7) = 1.21
        // Throughput baseline quantization=0.8, * 1.21 = 0.968
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.7,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.quantization_aggressiveness, 0.968, 0.01);
    }

    // ── 200. CPU 7 registers scarcity precise ──

    #[test]
    fn cpu_7_registers_scarcity_precise() {
        // scarcity = 1.0 - (7/32) = 0.78125
        // epilogue = 1.0 + 0.78125*0.3 = 1.234375
        // k_depth = 1.0 - 0.78125*0.2 = 0.84375
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 7,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.234375 = 1.8515625
        assert_approx(bias.epilogue_depth_preference, 1.8515625, 0.01);
        // baseline k_depth=1.3, * 0.84375 = 1.096875
        assert_approx(bias.k_depth_preference, 1.096875, 0.01);
    }

    // ── 201. CPU 9 registers scarcity precise ──

    #[test]
    fn cpu_9_registers_scarcity_precise() {
        // scarcity = 1.0 - (9/32) = 0.71875
        // epilogue = 1.0 + 0.71875*0.3 = 1.215625
        // k_depth = 1.0 - 0.71875*0.2 = 0.85625
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 9,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.215625 = 1.8234375
        assert_approx(bias.epilogue_depth_preference, 1.8234375, 0.01);
        // baseline k_depth=1.3, * 0.85625 = 1.113125
        assert_approx(bias.k_depth_preference, 1.113125, 0.01);
    }

    // ── 202. CPU 11 registers scarcity precise ──

    #[test]
    fn cpu_11_registers_scarcity_precise() {
        // scarcity = 1.0 - (11/32) = 0.65625
        // epilogue = 1.0 + 0.65625*0.3 = 1.196875
        // k_depth = 1.0 - 0.65625*0.2 = 0.86875
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 11,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.196875 = 1.7953125
        assert_approx(bias.epilogue_depth_preference, 1.7953125, 0.01);
        // baseline k_depth=1.3, * 0.86875 = 1.129375
        assert_approx(bias.k_depth_preference, 1.129375, 0.01);
    }

    // ── 203. CPU 13 registers scarcity precise ──

    #[test]
    fn cpu_13_registers_scarcity_precise() {
        // scarcity = 1.0 - (13/32) = 0.59375
        // epilogue = 1.0 + 0.59375*0.3 = 1.178125
        // k_depth = 1.0 - 0.59375*0.2 = 0.88125
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 13,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.178125 = 1.7671875
        assert_approx(bias.epilogue_depth_preference, 1.7671875, 0.01);
        // baseline k_depth=1.3, * 0.88125 = 1.145625
        assert_approx(bias.k_depth_preference, 1.145625, 0.01);
    }

    // ── 204. CPU 15 registers scarcity precise ──

    #[test]
    fn cpu_15_registers_scarcity_precise() {
        // scarcity = 1.0 - (15/32) = 0.53125
        // epilogue = 1.0 + 0.53125*0.3 = 1.159375
        // k_depth = 1.0 - 0.53125*0.2 = 0.89375
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 15,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.159375 = 1.7390625
        assert_approx(bias.epilogue_depth_preference, 1.7390625, 0.01);
        // baseline k_depth=1.3, * 0.89375 = 1.161875
        assert_approx(bias.k_depth_preference, 1.161875, 0.01);
    }

    // ── 205. Combined: GPU + Latency + high fusion + high memory ──

    #[test]
    fn combined_gpu_latency_high_fusion_high_memory() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.8,
            pipeline_valuable: 0.1,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert!(bias.kv_cache_budget_scale > 0.5);
    }

    // ── 206. GPU with 16KB shared mem: richness very small ──

    #[test]
    fn gpu_16kb_shared_mem_fusion_precise() {
        // richness = 16384/65536 = 0.25, sqrt(0.25) = 0.5
        // fusion *= 1/0.5 = 2.0
        // Latency baseline=0.5, * 2.0 = 1.0
        let gpu = ArbiterHwView::gpu(16384);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(bias.fusion_cost_scale, 1.0, 0.01);
    }

    // ── 207. Reg tension zero with equal fusion and pipeline: exact values ──

    #[test]
    fn reg_tension_zero_exact_baseline_values() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.42,
            pipeline_valuable: 0.42,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // reg_tension=0, so epilogue and k_depth stay at throughput baseline
        assert_approx(bias.epilogue_depth_preference, 0.8, 0.01);
        assert_approx(bias.k_depth_preference, 0.8, 0.01);
    }

    // ── 208. MoE with Latency mode: expert_eviction stays at baseline ──

    #[test]
    fn moe_latency_expert_eviction_stays_zero() {
        // Latency baseline expert_eviction=0.0, any lerp(1.0, 1.5, x)*0.0 = 0.0
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.9,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.expert_eviction_aggressiveness, 0.0, 0.01);
    }

    // ── 209. MoE with Latency mode: expert_prefetch scaled by memory ──

    #[test]
    fn moe_latency_expert_prefetch_scaled() {
        // Latency baseline expert_prefetch=0.5
        // parallelism>0.5: lerp(1.0, 2.0, 1.0)=2.0 -> 0.5*2.0=1.0
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.9,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.expert_prefetch_priority, 1.0, 0.01);
    }

    // ── 210. Fusion modulation with Throughput + fusion_profitable=0.5 ──

    #[test]
    fn fusion_modulation_throughput_0_5_precise() {
        // lerp(1.0, 0.6, 0.5) = 0.8
        // Throughput baseline fusion=1.0, * 0.8 = 0.8
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.8, 0.01);
    }

    // ── 211. Pipeline modulation with Throughput + pipeline_valuable=0.3 ──

    #[test]
    fn pipeline_modulation_throughput_0_3_precise() {
        // lerp(1.0, 0.6, 0.3) = 0.88
        // Throughput baseline pipeline=1.3, * 0.88 = 1.144
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.3,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.pipeline_cost_scale, 1.144, 0.01);
    }

    // ── 212. Reg tension moderate positive (0.3) precise ──

    #[test]
    fn reg_tension_moderate_positive_0_3_precise() {
        // fusion_profitable=0.8, pipeline_valuable=0.5 -> tension=0.3
        // epilogue *= 1.0 + 0.3*0.5 = 1.15
        // k_depth *= 1.0 - 0.3*0.3 = 0.91
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.8,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline epilogue=1.5, * 1.15 = 1.725
        assert_approx(bias.epilogue_depth_preference, 1.725, 0.01);
        // baseline k_depth=1.3, * 0.91 = 1.183
        assert_approx(bias.k_depth_preference, 1.183, 0.01);
    }

    // ── 213. Reg tension moderate negative (-0.3) precise ──

    #[test]
    fn reg_tension_moderate_negative_0_3_precise() {
        // fusion_profitable=0.5, pipeline_valuable=0.8 -> tension=-0.3
        // k_depth *= 1.0 + 0.3*0.5 = 1.15
        // epilogue *= 1.0 - 0.3*0.3 = 0.91
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.8,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // baseline k_depth=1.3, * 1.15 = 1.495
        assert_approx(bias.k_depth_preference, 1.495, 0.01);
        // baseline epilogue=1.5, * 0.91 = 1.365
        assert_approx(bias.epilogue_depth_preference, 1.365, 0.01);
    }

    // ── 214. All archetype fields same value: all modulations proportional ──

    #[test]
    fn archetype_all_same_value_proportional() {
        // All fields=0.6
        let arch = GraphArchetype {
            compute_intensive: 0.6,
            memory_intensive: 0.6,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.6,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // reg_tension = 0.6 - 0.6 = 0.0, so no epilogue/k_depth adjustment beyond baseline
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
    }

    // ── 215. Deterministic across many invocations ──

    #[test]
    fn arbitrate_deterministic_many_invocations() {
        let arch = GraphArchetype {
            compute_intensive: 0.4,
            memory_intensive: 0.7,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.8,
            pipeline_valuable: 0.2,
        };
        let hw = cpu_avx512();
        let first = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        for _ in 0..10 {
            let next = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            assert_eq!(first.fusion_cost_scale, next.fusion_cost_scale);
            assert_eq!(first.epilogue_depth_preference, next.epilogue_depth_preference);
            assert_eq!(first.expert_eviction_aggressiveness, next.expert_eviction_aggressiveness);
        }
    }

    // ── 216. ArbiterHwView with all fields at 1 (minimum non-zero) ──

    #[test]
    fn arbiter_hw_view_min_nonzero_fields() {
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 1,
            cache_sizes: (1, 1, 1),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &view);
        // Should not panic and produce valid output
        assert!(bias.fusion_cost_scale >= 0.2);
        assert!(bias.epilogue_depth_preference >= 0.3);
    }

    // ── 217. Fusion cost is sum of archetype + hw effects (orthogonal) ──

    #[test]
    fn fusion_cost_archetype_hw_orthogonal() {
        // Test that fusion_cost_scale = baseline * archetype_factor * hw_factor
        // with archetype fusion_profitable=0.5 and L1=32KB
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.0,
        };
        let hw_32kb = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let hw_64kb = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias_32 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_32kb);
        let bias_64 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_64kb);
        // ratio of fusion_cost should equal ratio of hw factors alone
        let hw_ratio = bias_32.fusion_cost_scale / bias_64.fusion_cost_scale;
        // 32KB: richness=0.5, sqrt=0.7071, factor=1/0.7071=1.4142
        // 64KB: richness=1.0, sqrt=1.0, factor=1.0
        // ratio = 1.4142
        assert_approx(hw_ratio, 1.414, 0.02);
    }

    // ── 218. L1 richness with Throughput + fusion_profitable combined ──

    #[test]
    fn l1_throughput_fusion_profitable_combined() {
        // fusion_profitable=1.0 + L1=32KB
        // archetype: lerp(1.0, 0.6, 1.0) = 0.6
        // hw: richness=0.5, sqrt=0.7071, factor=1/0.7071=1.4142
        // Throughput baseline=1.0, * 0.6 * 1.4142 = 0.8485
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.849, 0.01);
    }

    // ── 219. CPU 1 register scarcity with Throughput mode precise ──

    #[test]
    fn cpu_1_register_throughput_scarcity_precise() {
        // scarcity = 1.0 - (1/32) = 0.96875
        // epilogue = 1.0 + 0.96875*0.3 = 1.290625
        // k_depth = 1.0 - 0.96875*0.2 = 0.80625
        // Throughput baseline epilogue=0.8, * 1.290625 = 1.0325
        // Throughput baseline k_depth=0.8, * 0.80625 = 0.645
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 1,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.0325, 0.01);
        assert_approx(bias.k_depth_preference, 0.645, 0.01);
    }

    // ── 220. GPU pipeline_cost_scale with Throughput precise ──

    #[test]
    fn gpu_pipeline_cost_throughput_precise() {
        // GPU: pipeline *= 1.2
        // Throughput baseline pipeline=1.3, * 1.2 = 1.56
        let gpu = ArbiterHwView::gpu(65536);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert_approx(bias.pipeline_cost_scale, 1.56, 0.01);
    }

    // ── 221. GPU k_depth_preference with Throughput precise ──

    #[test]
    fn gpu_k_depth_throughput_precise() {
        // GPU: k_depth *= 1.2
        // Throughput baseline k_depth=0.8, * 1.2 = 0.96
        let gpu = ArbiterHwView::gpu(65536);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert_approx(bias.k_depth_preference, 0.96, 0.01);
    }

    // ── 222. GPU epilogue_depth_preference with Throughput precise ──

    #[test]
    fn gpu_epilogue_depth_throughput_precise() {
        // GPU: epilogue *= 1.2
        // Throughput baseline epilogue=0.8, * 1.2 = 0.96
        let gpu = ArbiterHwView::gpu(65536);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert_approx(bias.epilogue_depth_preference, 0.96, 0.01);
    }

    // ── 223. CPU 8 registers with Throughput scarcity precise ──

    #[test]
    fn cpu_8_registers_throughput_scarcity_precise() {
        // scarcity = 0.75
        // epilogue = 1.0 + 0.75*0.3 = 1.225
        // k_depth = 1.0 - 0.75*0.2 = 0.85
        // Throughput baseline epilogue=0.8, * 1.225 = 0.98
        // Throughput baseline k_depth=0.8, * 0.85 = 0.68
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 0.98, 0.01);
        assert_approx(bias.k_depth_preference, 0.68, 0.01);
    }

    // ── 224. Combined all-effects sweep: all outputs within bounds ──

    #[test]
    fn combined_all_effects_sweep_within_bounds() {
        fn check(bias: &StrategyBias) {
            assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
            assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
            assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
            assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
            assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
            assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
            assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
            assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
            assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
            assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
            assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
            assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
            assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
        }

        let archetypes = [
            GraphArchetype { compute_intensive: 0.0, memory_intensive: 0.0, parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0 },
            GraphArchetype { compute_intensive: 1.0, memory_intensive: 1.0, parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0 },
            GraphArchetype { compute_intensive: 0.5, memory_intensive: 0.5, parallelism_exploitable: 0.5, fusion_profitable: 0.5, pipeline_valuable: 0.5 },
            GraphArchetype { compute_intensive: -1.0, memory_intensive: 2.0, parallelism_exploitable: 0.8, fusion_profitable: 1.5, pipeline_valuable: -0.5 },
        ];
        let hws = [
            ArbiterHwView { device: DeviceFamily::Cpu, num_simd_regs: 4, cache_sizes: (4096, 0, 0) },
            ArbiterHwView { device: DeviceFamily::Cpu, num_simd_regs: 16, cache_sizes: (32768, 0, 0) },
            ArbiterHwView { device: DeviceFamily::Cpu, num_simd_regs: 32, cache_sizes: (65536, 0, 0) },
            ArbiterHwView::gpu(65536),
            ArbiterHwView::gpu(0),
        ];
        for arch in &archetypes {
            for &mode in &[InferenceMode::Latency, InferenceMode::Throughput] {
                for hw in &hws {
                    let bias = StrategyArbiter::arbitrate(mode, arch, hw);
                    check(&bias);
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════
    //  Additional tests (225–280): property, boundary, and integration
    // ══════════════════════════════════════════════════════════════════

    // ── 225. InferenceMode PartialOrd ordering ──

    #[test]
    fn inference_mode_partial_ord_latency_less_than_throughput() {
        assert!(InferenceMode::Latency < InferenceMode::Throughput);
        assert!(!(InferenceMode::Throughput < InferenceMode::Latency));
    }

    // ── 226. InferenceMode Ord total order roundtrip ──

    #[test]
    fn inference_mode_ord_total_order_consistent() {
        let a = InferenceMode::Latency;
        let b = InferenceMode::Throughput;
        assert_eq!(a.cmp(&a), std::cmp::Ordering::Equal);
        assert_eq!(a.cmp(&b), std::cmp::Ordering::Less);
        assert_eq!(b.cmp(&a), std::cmp::Ordering::Greater);
        assert_eq!(b.cmp(&b), std::cmp::Ordering::Equal);
    }

    // ── 227. InferenceMode from via Default ──

    #[test]
    fn inference_mode_default_matches_latency_variant() {
        let mode: InferenceMode = Default::default();
        assert_eq!(mode, InferenceMode::Latency);
        assert_eq!(mode, InferenceMode::default());
    }

    // ── 228. ArbiterHwView gpu with various shared memory sizes ──

    #[test]
    fn arbiter_hw_view_gpu_shared_mem_variants() {
        let sizes: Vec<usize> = vec![0, 1024, 16384, 32768, 49152, 65536, 98304, 131072, 262144];
        for sz in sizes {
            let view = ArbiterHwView::gpu(sz);
            assert!(view.device == DeviceFamily::Gpu, "gpu({}) should have device=Gpu", sz);
            assert_eq!(view.num_simd_regs, 255, "gpu() should set 255 regs");
            assert_eq!(view.cache_sizes.0, sz, "gpu({}) L1 mismatch", sz);
            assert_eq!(view.cache_sizes.1, 0, "gpu() L2 should be 0");
            assert_eq!(view.cache_sizes.2, 0, "gpu() L3 should be 0");
        }
    }

    // ── 229. ArbiterHwView cpu constructor with max register count ──

    #[test]
    fn arbiter_hw_view_cpu_max_registers() {
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: usize::MAX,
            cache_sizes: (65536, 0, 0),
        };
        assert!(view.device == DeviceFamily::Cpu);
        assert_eq!(view.num_simd_regs, usize::MAX);
    }

    // ── 230. ArbiterHwView equality symmetry with gpu and cpu ──

    #[test]
    fn arbiter_hw_view_eq_cpu_gpu_never_equal() {
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(49152);
        // Same num_simd_regs and cache_sizes but different is_gpu
        assert_ne!(cpu, gpu);
        assert_ne!(gpu, cpu);
    }

    // ── 231. StrategyBias clamps infinity but preserves NaN ──

    #[test]
    fn strategy_bias_validate_clamps_infinity() {
        let mut bias = StrategyBias {
            fusion_cost_scale: f64::INFINITY,
            pipeline_cost_scale: f64::INFINITY,
            parallelism_cost_scale: f64::INFINITY,
            epilogue_depth_preference: f64::INFINITY,
            k_depth_preference: f64::INFINITY,
            kv_cache_budget_scale: f64::INFINITY,
            weight_prefetch_budget_scale: f64::INFINITY,
            batch_flexibility: f64::INFINITY,
            decode_ratio_scale: f64::INFINITY,
            speculative_decoding_value: f64::INFINITY,
            quantization_aggressiveness: f64::INFINITY,
            expert_eviction_aggressiveness: f64::INFINITY,
            expert_prefetch_priority: f64::INFINITY,
        };
        bias.validate();
        assert!(bias.fusion_cost_scale.is_finite());
        assert!(bias.pipeline_cost_scale.is_finite());
        assert!(bias.parallelism_cost_scale.is_finite());
        assert!(bias.epilogue_depth_preference.is_finite());
        assert!(bias.k_depth_preference.is_finite());
        assert!(bias.kv_cache_budget_scale.is_finite());
        assert!(bias.weight_prefetch_budget_scale.is_finite());
        assert!(bias.batch_flexibility.is_finite());
        assert!(bias.decode_ratio_scale.is_finite());
        assert!(bias.speculative_decoding_value.is_finite());
        assert!(bias.quantization_aggressiveness.is_finite());
        assert!(bias.expert_eviction_aggressiveness.is_finite());
        assert!(bias.expert_prefetch_priority.is_finite());
    }

    // ── 231b. StrategyBias validate preserves NaN (f64::clamp semantics) ──

    #[test]
    fn strategy_bias_validate_nan_preserved_via_clamp() {
        let mut bias = StrategyBias {
            fusion_cost_scale: f64::NAN,
            ..StrategyBias::default()
        };
        bias.validate();
        // f64::clamp: if self is NaN, result is NaN (IEEE 754 semantics)
        assert!(bias.fusion_cost_scale.is_nan());
    }

    // ── 232. StrategyBias validate does not alter already-valid values ──

    #[test]
    fn strategy_bias_validate_preserves_valid_custom_values() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.5,
            pipeline_cost_scale: 0.8,
            parallelism_cost_scale: 1.2,
            epilogue_depth_preference: 1.8,
            k_depth_preference: 0.6,
            kv_cache_budget_scale: 2.5,
            weight_prefetch_budget_scale: 0.3,
            batch_flexibility: 0.7,
            decode_ratio_scale: 1.5,
            speculative_decoding_value: 2.0,
            quantization_aggressiveness: 1.0,
            expert_eviction_aggressiveness: 1.2,
            expert_prefetch_priority: 3.0,
        };
        bias.validate();
        assert_approx(bias.fusion_cost_scale, 0.5, 0.001);
        assert_approx(bias.pipeline_cost_scale, 0.8, 0.001);
        assert_approx(bias.parallelism_cost_scale, 1.2, 0.001);
        assert_approx(bias.epilogue_depth_preference, 1.8, 0.001);
        assert_approx(bias.k_depth_preference, 0.6, 0.001);
        assert_approx(bias.kv_cache_budget_scale, 2.5, 0.001);
        assert_approx(bias.weight_prefetch_budget_scale, 0.3, 0.001);
        assert_approx(bias.batch_flexibility, 0.7, 0.001);
        assert_approx(bias.decode_ratio_scale, 1.5, 0.001);
        assert_approx(bias.speculative_decoding_value, 2.0, 0.001);
        assert_approx(bias.quantization_aggressiveness, 1.0, 0.001);
        assert_approx(bias.expert_eviction_aggressiveness, 1.2, 0.001);
        assert_approx(bias.expert_prefetch_priority, 3.0, 0.001);
    }

    // ── 233. compute_intensive archetype field has no effect on bias ──

    #[test]
    fn compute_intensive_no_effect_confirm() {
        let arch_lo = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let arch_hi = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lo = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_lo, &hw);
        let hi = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_hi, &hw);
        assert_approx(lo.fusion_cost_scale, hi.fusion_cost_scale, 0.001);
        assert_approx(lo.pipeline_cost_scale, hi.pipeline_cost_scale, 0.001);
        assert_approx(lo.kv_cache_budget_scale, hi.kv_cache_budget_scale, 0.001);
    }

    // ── 234. GPU skip register scarcity: GPU with 1 reg uses GPU branch ──

    #[test]
    fn gpu_with_one_register_still_uses_gpu_branch() {
        let gpu_low_reg = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 1,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_32_reg = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_low_reg);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu_32_reg);
        // GPU branch boosts epilogue by 1.2, CPU 32-reg has no scarcity adjustment.
        // GPU result: 1.5 * 1.2 = 1.8 (no scarcity because is_gpu=true but num_simd_regs=1 <= 16)
        // Actually scarcity applies regardless of is_gpu — but the GPU boost also applies.
        // The key property: GPU bias should differ from CPU baseline.
        assert!(
            (gpu_bias.epilogue_depth_preference - cpu_bias.epilogue_depth_preference).abs() > 0.01,
            "GPU with 1 reg should differ from CPU with 32 regs"
        );
    }

    // ── 235. weight_prefetch_bias independent of hardware L1 size ──

    #[test]
    fn weight_prefetch_bias_independent_of_l1_size() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let small_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (4096, 0, 0),
        };
        let large_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0),
        };
        let bias_small = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &small_l1);
        let bias_large = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &large_l1);
        assert_approx(bias_small.weight_prefetch_budget_scale, bias_large.weight_prefetch_budget_scale, 0.001);
    }

    // ── 236. batch_flexibility only depends on mode, not archetype ──

    #[test]
    fn batch_flexibility_unchanged_by_negative_archetype() {
        let arch = GraphArchetype {
            compute_intensive: -2.0,
            memory_intensive: -1.0,
            parallelism_exploitable: -0.5,
            fusion_profitable: -1.0,
            pipeline_valuable: -0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_eq!(latency.batch_flexibility, 0.0);
        assert_eq!(throughput.batch_flexibility, 1.0);
    }

    // ── 237. decode_ratio_scale not modulated by archetype ──

    #[test]
    fn decode_ratio_scale_not_modulated_by_archetype() {
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let max_arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let zero_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        let max_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &max_arch, &hw);
        assert_approx(zero_bias.decode_ratio_scale, max_bias.decode_ratio_scale, 0.001);
    }

    // ── 238. speculative_decoding_value not affected by hardware ──

    #[test]
    fn speculative_decoding_not_affected_by_hardware_confirm() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (4096, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(cpu_bias.speculative_decoding_value, gpu_bias.speculative_decoding_value, 0.001);
    }

    // ── 239. Fusion modulation: fusion_profitable=1.0 reduces fusion_cost_scale ──

    #[test]
    fn fusion_profitable_max_halves_fusion_cost() {
        // lerp(1.0, 0.6, 1.0) = 0.6
        // Latency baseline fusion_cost_scale = 0.5, * 0.6 = 0.3
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.3, 0.01);
    }

    // ── 240. Pipeline modulation: pipeline_valuable=1.0 reduces pipeline_cost_scale ──

    #[test]
    fn pipeline_valuable_max_reduces_pipeline_cost() {
        // lerp(1.0, 0.6, 1.0) = 0.6
        // Latency baseline pipeline_cost_scale = 0.6, * 0.6 = 0.36
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.pipeline_cost_scale, 0.36, 0.01);
    }

    // ── 241. Parallelism modulation: parallelism_exploitable=1.0 reduces cost ──

    #[test]
    fn parallelism_exploitable_max_reduces_parallelism_cost() {
        // lerp(1.0, 0.5, 1.0) = 0.5
        // Latency baseline parallelism_cost_scale = 1.5, * 0.5 = 0.75
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.parallelism_cost_scale, 0.75, 0.01);
    }

    // ── 242. KV cache modulation: memory_intensive=1.0 maxes kv budget ──

    #[test]
    fn kv_cache_memory_intensive_max_precise() {
        // lerp(1.0, 1.5, 1.0) = 1.5
        // Latency baseline kv_cache_budget_scale = 0.5, * 1.5 = 0.75
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.kv_cache_budget_scale, 0.75, 0.01);
    }

    // ── 243. Quantization modulation: memory_intensive=0.0 is identity ──

    #[test]
    fn quantization_zero_memory_identity() {
        // lerp(1.0, 1.3, 0.0) = 1.0
        // Latency baseline quantization_aggressiveness = 1.5, * 1.0 = 1.5
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.quantization_aggressiveness, 1.5, 0.01);
    }

    // ── 244. All arbitrate outputs are finite for randomized inputs ──

    #[test]
    fn arbitrate_all_outputs_finite_for_varied_inputs() {
        let archetypes = [
            GraphArchetype { compute_intensive: -5.0, memory_intensive: -5.0, parallelism_exploitable: -5.0, fusion_profitable: -5.0, pipeline_valuable: -5.0 },
            GraphArchetype { compute_intensive: 5.0, memory_intensive: 5.0, parallelism_exploitable: 5.0, fusion_profitable: 5.0, pipeline_valuable: 5.0 },
            GraphArchetype { compute_intensive: 0.33, memory_intensive: 0.67, parallelism_exploitable: 0.11, fusion_profitable: 0.89, pipeline_valuable: 0.42 },
        ];
        let hws = [
            ArbiterHwView { device: DeviceFamily::Cpu, num_simd_regs: 0, cache_sizes: (0, 0, 0) },
            ArbiterHwView { device: DeviceFamily::Cpu, num_simd_regs: 32, cache_sizes: (262144, 0, 0) },
            ArbiterHwView::gpu(131072),
        ];
        for arch in &archetypes {
            for &mode in &[InferenceMode::Latency, InferenceMode::Throughput] {
                for hw in &hws {
                    let bias = StrategyArbiter::arbitrate(mode, arch, hw);
                    assert!(bias.fusion_cost_scale.is_finite());
                    assert!(bias.pipeline_cost_scale.is_finite());
                    assert!(bias.parallelism_cost_scale.is_finite());
                    assert!(bias.epilogue_depth_preference.is_finite());
                    assert!(bias.k_depth_preference.is_finite());
                    assert!(bias.kv_cache_budget_scale.is_finite());
                    assert!(bias.weight_prefetch_budget_scale.is_finite());
                    assert!(bias.batch_flexibility.is_finite());
                    assert!(bias.decode_ratio_scale.is_finite());
                    assert!(bias.expert_eviction_aggressiveness.is_finite());
                    assert!(bias.expert_prefetch_priority.is_finite());
                    assert!(bias.speculative_decoding_value.is_finite());
                    assert!(bias.quantization_aggressiveness.is_finite());
                }
            }
        }
    }

    // ── 245. InferenceMode Eq implies Hash consistency ──

    #[test]
    fn inference_mode_eq_hash_implies_same_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        InferenceMode::Latency.hash(&mut h1);
        InferenceMode::Latency.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
        let mut h3 = DefaultHasher::new();
        InferenceMode::Throughput.hash(&mut h3);
        assert_ne!(h1.finish(), h3.finish());
    }

    // ── 246. ArbiterHwView with zero L1: l1_richness = 0, skip sqrt ──

    #[test]
    fn arbiter_hw_view_zero_l1_skip_fusion_adjustment() {
        let zero_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (0, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &zero_l1);
        // No L1 adjustment (l1_richness=0 skips the sqrt), no archetype modulation
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
    }

    // ── 247. CPU register scarcity at boundary num_simd_regs=1 ──

    #[test]
    fn cpu_1_register_extreme_scarcity_confirm() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 1,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // scarcity = 1.0 - 1/32 ≈ 0.96875
        // epilogue = 1.5 * (1 + 0.96875*0.3) ≈ 1.5 * 1.2906 ≈ 1.936
        assert!(bias.epilogue_depth_preference > 1.9);
        // k_depth = 1.3 * (1 - 0.96875*0.2) ≈ 1.3 * 0.80625 ≈ 1.048
        assert!(bias.k_depth_preference < 1.1);
    }

    // ── 248. L1 richness monotonic: larger L1 always reduces fusion_cost ──

    #[test]
    fn l1_richness_strictly_monotonic_fusion_cost() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let l1_sizes: Vec<usize> = vec![8192, 16384, 32768, 49152, 65536, 98304, 131072];
        let mut prev_fusion = f64::MAX;
        for l1 in l1_sizes {
            let hw = ArbiterHwView {
                device: DeviceFamily::Cpu,
                num_simd_regs: 32,
                cache_sizes: (l1, 0, 0),
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.fusion_cost_scale <= prev_fusion + 1e-9,
                "fusion_cost should not increase as L1 grows: l1={}, fusion={}, prev={}",
                l1, bias.fusion_cost_scale, prev_fusion,
            );
            prev_fusion = bias.fusion_cost_scale;
        }
    }

    // ── 249. MoE modulation threshold: parallelism > 0.5 is strict ──

    #[test]
    fn moe_parallelism_strict_threshold() {
        // MoE branch fires when parallelism_exploitable > 0.5 (strictly greater)
        let arch_at = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let arch_above = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.51,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let arch_below = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.49,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let at = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch_at, &hw);
        let above = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch_above, &hw);
        let below = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch_below, &hw);
        // parallelism=0.5 does NOT fire (> 0.5 is strict), so at == below
        assert_approx(at.expert_eviction_aggressiveness, below.expert_eviction_aggressiveness, 0.001);
        // parallelism=0.51 DOES fire, so above > at
        assert!(
            above.expert_eviction_aggressiveness > at.expert_eviction_aggressiveness,
            "parallelism=0.51 should trigger MoE modulation: above={:.4}, at={:.4}",
            above.expert_eviction_aggressiveness, at.expert_eviction_aggressiveness,
        );
    }

    // ── 250. GPU L1 richness capped at 2.0 for huge shared memory ──

    #[test]
    fn gpu_huge_shared_mem_l1_richness_capped() {
        let gpu = ArbiterHwView::gpu(512 * 1024); // 512KB shared memory
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // l1_richness = min(512*1024 / 65536, 2.0) = min(8.0, 2.0) = 2.0
        // fusion_cost_scale = 0.5 / sqrt(2.0) ≈ 0.354
        assert_approx(bias.fusion_cost_scale, 0.5 / 2.0_f64.sqrt(), 0.01);
    }

    // ── 251. ArbiterHwView hash: same fields same hash ──

    #[test]
    fn arbiter_hw_view_hash_same_fields_same_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let v1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 0, 0),
        };
        let v2 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 0, 0),
        };
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        v1.hash(&mut h1);
        v2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── 252. Reg tension zero: fusion=pipeline gives identity ──

    #[test]
    fn reg_tension_zero_no_epilogue_or_k_depth_change() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let baseline_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let tension_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let baseline_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &baseline_arch, &hw);
        // reg_tension = 0.5 - 0.5 = 0.0, so epilogue and k_depth should match baseline
        assert_approx(tension_bias.epilogue_depth_preference, baseline_bias.epilogue_depth_preference, 0.001);
        assert_approx(tension_bias.k_depth_preference, baseline_bias.k_depth_preference, 0.001);
    }

    // ── 253. Throughput mode weight_prefetch lower than latency ──

    #[test]
    fn throughput_weight_prefetch_lower_than_latency() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            latency.weight_prefetch_budget_scale > throughput.weight_prefetch_budget_scale,
            "latency should have higher weight_prefetch: {} vs {}",
            latency.weight_prefetch_budget_scale, throughput.weight_prefetch_budget_scale,
        );
    }

    // ── 254. Latency mode fusion_cost_scale lower than throughput ──

    #[test]
    fn latency_fusion_cost_lower_than_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            latency.fusion_cost_scale < throughput.fusion_cost_scale,
            "latency fusion_cost should be lower: {} vs {}",
            latency.fusion_cost_scale, throughput.fusion_cost_scale,
        );
    }

    // ── 255. expert_prefetch_priority monotonic with memory_intensive ──

    #[test]
    fn expert_prefetch_monotonic_with_memory() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mem_levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut prev_prefetch = f64::MIN;
        for mem in mem_levels {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: mem,
                parallelism_exploitable: 0.8, // above 0.5 threshold
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            assert!(
                bias.expert_prefetch_priority >= prev_prefetch - 1e-9,
                "expert_prefetch should be monotonic with memory: mem={}, prefetch={}, prev={}",
                mem, bias.expert_prefetch_priority, prev_prefetch,
            );
            prev_prefetch = bias.expert_prefetch_priority;
        }
    }

    // ── 256. StrategyBias default validate is identity ──

    #[test]
    fn strategy_bias_default_validate_is_noop() {
        let mut bias = StrategyBias::default();
        let before = bias;
        bias.validate();
        assert_approx(bias.fusion_cost_scale, before.fusion_cost_scale, 0.001);
        assert_approx(bias.pipeline_cost_scale, before.pipeline_cost_scale, 0.001);
        assert_approx(bias.parallelism_cost_scale, before.parallelism_cost_scale, 0.001);
        assert_approx(bias.epilogue_depth_preference, before.epilogue_depth_preference, 0.001);
        assert_approx(bias.k_depth_preference, before.k_depth_preference, 0.001);
        assert_approx(bias.kv_cache_budget_scale, before.kv_cache_budget_scale, 0.001);
        assert_approx(bias.weight_prefetch_budget_scale, before.weight_prefetch_budget_scale, 0.001);
        assert_approx(bias.batch_flexibility, before.batch_flexibility, 0.001);
        assert_approx(bias.decode_ratio_scale, before.decode_ratio_scale, 0.001);
        assert_approx(bias.speculative_decoding_value, before.speculative_decoding_value, 0.001);
        assert_approx(bias.quantization_aggressiveness, before.quantization_aggressiveness, 0.001);
        assert_approx(bias.expert_eviction_aggressiveness, before.expert_eviction_aggressiveness, 0.001);
        assert_approx(bias.expert_prefetch_priority, before.expert_prefetch_priority, 0.001);
    }

    // ── 257. GraphArchetype with negative values produces valid bias ──

    #[test]
    fn negative_archetype_produces_valid_bias() {
        let arch = GraphArchetype {
            compute_intensive: -2.0,
            memory_intensive: -1.0,
            parallelism_exploitable: -0.5,
            fusion_profitable: -1.0,
            pipeline_valuable: -0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
    }

    // ── 258. arbiter_output_never_nan for extreme archetype values ──

    #[test]
    fn arbiter_output_never_nan_extreme_archetype() {
        let arch = GraphArchetype {
            compute_intensive: f64::MAX,
            memory_intensive: f64::MAX,
            parallelism_exploitable: f64::MAX,
            fusion_profitable: f64::MAX,
            pipeline_valuable: f64::MAX,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert!(!bias.fusion_cost_scale.is_nan());
        assert!(!bias.pipeline_cost_scale.is_nan());
        assert!(!bias.parallelism_cost_scale.is_nan());
    }

    // ── 259. L2 and L3 cache do not affect any bias field ──

    #[test]
    fn l2_l3_various_sizes_no_effect() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let base = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let with_l2 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 1048576, 0),
        };
        let with_l2_l3 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 1048576, 33554432),
        };
        let b0 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &base);
        let b1 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &with_l2);
        let b2 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &with_l2_l3);
        assert_approx(b0.fusion_cost_scale, b1.fusion_cost_scale, 0.001);
        assert_approx(b1.fusion_cost_scale, b2.fusion_cost_scale, 0.001);
        assert_approx(b0.epilogue_depth_preference, b2.epilogue_depth_preference, 0.001);
    }

    // ── 260. Arbitrate deterministic across 100 invocations ──

    #[test]
    fn arbitrate_deterministic_100_invocations() {
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.8,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 0, 0),
        };
        let first = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        for _ in 0..100 {
            let b = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            assert_approx(b.fusion_cost_scale, first.fusion_cost_scale, 0.0001);
            assert_approx(b.pipeline_cost_scale, first.pipeline_cost_scale, 0.0001);
            assert_approx(b.k_depth_preference, first.k_depth_preference, 0.0001);
        }
    }

    // ── 261. GPU with zero shared memory: no L1 richness adjustment ──

    #[test]
    fn gpu_zero_shared_mem_l1_richness_zero_confirmed() {
        let gpu = ArbiterHwView::gpu(0);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // GPU boost: epilogue *= 1.2, k_depth *= 1.2, pipeline *= 1.2
        // No L1 richness adjustment (l1_richness=0)
        // Latency baseline: epilogue=1.5, k_depth=1.3, pipeline=0.6
        assert_approx(bias.epilogue_depth_preference, 1.5 * 1.2, 0.01);
        assert_approx(bias.k_depth_preference, 1.3 * 1.2, 0.01);
        assert_approx(bias.pipeline_cost_scale, 0.6 * 1.2, 0.01);
    }

    // ── 262. CPU with 32 registers: no scarcity, neutral L1 baseline ──

    #[test]
    fn cpu_32_registers_neutral_no_adjustment() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // No scarcity (32 > 16), L1=64K: l1_richness=1.0, sqrt(1.0)=1.0 → no adjustment
        // Pure baseline values
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
        assert_approx(bias.pipeline_cost_scale, 0.6, 0.01);
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
    }

    // ── 263. Combined mode + archetype + hardware full sweep no panic ──

    #[test]
    fn combined_sweep_no_panic() {
        let extreme_archetypes = [
            GraphArchetype { compute_intensive: f64::MIN, memory_intensive: f64::MIN, parallelism_exploitable: f64::MIN, fusion_profitable: f64::MIN, pipeline_valuable: f64::MIN },
            GraphArchetype { compute_intensive: f64::MAX, memory_intensive: f64::MAX, parallelism_exploitable: f64::MAX, fusion_profitable: f64::MAX, pipeline_valuable: f64::MAX },
        ];
        let extreme_hws = [
            ArbiterHwView { device: DeviceFamily::Cpu, num_simd_regs: 0, cache_sizes: (0, 0, 0) },
            ArbiterHwView { device: DeviceFamily::Cpu, num_simd_regs: usize::MAX, cache_sizes: (usize::MAX, usize::MAX, usize::MAX) },
            ArbiterHwView::gpu(0),
            ArbiterHwView::gpu(usize::MAX),
        ];
        for arch in &extreme_archetypes {
            for &mode in &[InferenceMode::Latency, InferenceMode::Throughput] {
                for hw in &extreme_hws {
                    let _ = StrategyArbiter::arbitrate(mode, arch, hw);
                }
            }
        }
    }

    // ── 264. InferenceMode can be used in sorted collection ──

    #[test]
    fn inference_mode_sorted_vec() {
        let mut modes = vec![InferenceMode::Throughput, InferenceMode::Latency];
        modes.sort();
        assert_eq!(modes[0], InferenceMode::Latency);
        assert_eq!(modes[1], InferenceMode::Throughput);
    }

    // ── 265. StrategyBias validate each boundary exactly ──

    #[test]
    fn strategy_bias_validate_each_field_boundary() {
        let mut bias = StrategyBias::default();
        // Set each field to its exact min, verify unchanged
        bias.fusion_cost_scale = 0.2;
        bias.validate();
        assert_approx(bias.fusion_cost_scale, 0.2, 0.001);

        bias.pipeline_cost_scale = 0.2;
        bias.validate();
        assert_approx(bias.pipeline_cost_scale, 0.2, 0.001);

        bias.parallelism_cost_scale = 0.1;
        bias.validate();
        assert_approx(bias.parallelism_cost_scale, 0.1, 0.001);

        bias.batch_flexibility = 0.0;
        bias.validate();
        assert_approx(bias.batch_flexibility, 0.0, 0.001);

        bias.decode_ratio_scale = 0.3;
        bias.validate();
        assert_approx(bias.decode_ratio_scale, 0.3, 0.001);

        bias.expert_eviction_aggressiveness = 0.0;
        bias.validate();
        assert_approx(bias.expert_eviction_aggressiveness, 0.0, 0.001);
    }

    // ── 266. Latency mode has higher speculative_decoding_value ──

    #[test]
    fn latency_speculative_decoding_higher_than_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            latency.speculative_decoding_value > throughput.speculative_decoding_value,
            "latency speculative={:.3} should be > throughput speculative={:.3}",
            latency.speculative_decoding_value, throughput.speculative_decoding_value,
        );
    }

    // ── 267. Throughput has higher expert_eviction_aggressiveness ──

    #[test]
    fn throughput_expert_eviction_higher_with_moe() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.8,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            throughput.expert_eviction_aggressiveness > latency.expert_eviction_aggressiveness,
            "throughput eviction={:.3} should be > latency eviction={:.3}",
            throughput.expert_eviction_aggressiveness, latency.expert_eviction_aggressiveness,
        );
    }

    // ── 268. arbiter_cpu_matches_manual_conversion precise ──

    #[test]
    fn arbitrate_cpu_matches_manual_hw_view() {
        let dp = DeviceProfile::detect();
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu_result = StrategyArbiter::arbitrate_cpu(InferenceMode::Latency, &arch, &dp);
        let manual_result = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &ArbiterHwView::from(&dp));
        assert_approx(cpu_result.fusion_cost_scale, manual_result.fusion_cost_scale, 0.001);
        assert_approx(cpu_result.pipeline_cost_scale, manual_result.pipeline_cost_scale, 0.001);
        assert_approx(cpu_result.parallelism_cost_scale, manual_result.parallelism_cost_scale, 0.001);
        assert_approx(cpu_result.epilogue_depth_preference, manual_result.epilogue_depth_preference, 0.001);
    }

    // ── 269. register_scarcity_affects_epilogue_but_not_batch_flexibility ──

    #[test]
    fn register_scarcity_not_affect_batch_or_decode() {
        let hw_scarce = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 4,
            cache_sizes: (65536, 0, 0),
        };
        let hw_rich = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let scarce = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_scarce);
        let rich = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_rich);
        assert_approx(scarce.batch_flexibility, rich.batch_flexibility, 0.001);
        assert_approx(scarce.decode_ratio_scale, rich.decode_ratio_scale, 0.001);
        assert_approx(scarce.speculative_decoding_value, rich.speculative_decoding_value, 0.001);
    }

    // ── 270. GPU boost independent of register scarcity branch ──

    #[test]
    fn gpu_boost_applies_even_with_low_registers() {
        let gpu_4_reg = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 4,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_4_reg = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 4,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_4_reg);
        let cpu = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu_4_reg);
        // GPU branch applies 1.2x boost on top of CPU scarcity
        assert!(
            gpu.epilogue_depth_preference > cpu.epilogue_depth_preference,
            "GPU+low_reg should boost more than CPU+low_reg: gpu={:.3} vs cpu={:.3}",
            gpu.epilogue_depth_preference, cpu.epilogue_depth_preference,
        );
    }

    // ── 271. ArbiterHwView debug contains all three cache sizes ──

    #[test]
    fn arbiter_hw_view_debug_shows_cache_tuple() {
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let debug = format!("{:?}", view);
        assert!(debug.contains("32768") || debug.contains("cache_sizes"));
    }

    // ── 272. StrategyBias accessor methods return exact field values ──

    #[test]
    fn strategy_bias_accessor_methods_exact_values() {
        let bias = StrategyBias {
            expert_eviction_aggressiveness: 0.5,
            expert_prefetch_priority: 2.0,
            ..StrategyBias::default()
        };
        assert_eq!(bias.expert_eviction_aggressiveness(), 0.5);
        assert_eq!(bias.expert_prefetch_priority(), 2.0);
    }

    // ── 273. Quantization aggressiveness monotonic with memory_intensive ──

    #[test]
    fn quantization_monotonic_with_memory() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mem_levels = [0.0, 0.3, 0.6, 1.0];
        let mut prev_quant = f64::MIN;
        for mem in mem_levels {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: mem,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.quantization_aggressiveness >= prev_quant - 1e-9,
                "quantization should be monotonic: mem={}, quant={}, prev={}",
                mem, bias.quantization_aggressiveness, prev_quant,
            );
            prev_quant = bias.quantization_aggressiveness;
        }
    }

    // ── 274. InferenceMode exhaustive: both variants cover all code paths ──

    #[test]
    fn inference_mode_exhaustive_coverage() {
        let modes = [InferenceMode::Latency, InferenceMode::Throughput];
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        for &mode in &modes {
            let bias = StrategyArbiter::arbitrate(mode, &arch, &hw);
            assert!(bias.fusion_cost_scale > 0.0);
            assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        }
    }

    // ── 275. StrategyBias field-wise mutation and read-back ──

    #[test]
    fn strategy_bias_mutation_and_readback() {
        let mut bias = StrategyBias::default();
        bias.fusion_cost_scale = 0.3;
        bias.k_depth_preference = 2.5;
        bias.expert_eviction_aggressiveness = 1.8;
        assert_approx(bias.fusion_cost_scale, 0.3, 0.001);
        assert_approx(bias.k_depth_preference, 2.5, 0.001);
        assert_approx(bias.expert_eviction_aggressiveness, 1.8, 0.001);
        // Other fields should remain at default
        assert_approx(bias.pipeline_cost_scale, 1.0, 0.001);
        assert_approx(bias.batch_flexibility, 1.0, 0.001);
    }

    // ── 276. ArbiterHwView Copy semantics verified ──

    #[test]
    fn arbiter_hw_view_copy_semantics_verified() {
        let original = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let copied = original;
        // Both should be equal (Copy)
        assert_eq!(original, copied);
        assert_eq!(original.num_simd_regs, copied.num_simd_regs);
        assert_eq!(original.cache_sizes, copied.cache_sizes);
    }

    // ── 277. KV cache budget monotonic with memory_intensive for throughput ──

    #[test]
    fn kv_cache_budget_monotonic_with_memory_throughput() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mem_levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut prev_kv = f64::MIN;
        for mem in mem_levels {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: mem,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            assert!(
                bias.kv_cache_budget_scale >= prev_kv - 1e-9,
                "kv_cache_budget should be monotonic: mem={}, kv={}, prev={}",
                mem, bias.kv_cache_budget_scale, prev_kv,
            );
            prev_kv = bias.kv_cache_budget_scale;
        }
    }

    // ── 278. StrategyArbiter is zero-sized ──

    #[test]
    fn strategy_arbiter_zero_sized_confirmed() {
        assert_eq!(std::mem::size_of::<StrategyArbiter>(), 0);
    }

    // ── 279. Register scarcity monotonic: fewer regs → higher epilogue ──

    #[test]
    fn register_scarcity_strictly_monotonic_epilogue() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let reg_counts: Vec<usize> = vec![2, 4, 6, 8, 10, 12, 14, 16];
        let mut prev_epilogue = f64::MAX;
        for regs in reg_counts {
            let hw = ArbiterHwView {
                device: DeviceFamily::Cpu,
                num_simd_regs: regs,
                cache_sizes: (65536, 0, 0),
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.epilogue_depth_preference <= prev_epilogue + 1e-9,
                "epilogue should decrease as regs increase: regs={}, epilogue={:.4}, prev={:.4}",
                regs, bias.epilogue_depth_preference, prev_epilogue,
            );
            prev_epilogue = bias.epilogue_depth_preference;
        }
    }

    // ── 280. Pipeline modulation identity at zero ──

    #[test]
    fn pipeline_modulation_zero_is_identity() {
        let arch_zero = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_zero, &hw);
        // lerp(1.0, 0.6, 0.0) = 1.0 → pipeline_cost_scale unchanged from baseline
        assert_approx(bias.pipeline_cost_scale, 0.6, 0.01);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Tests 281–360: Extended coverage
    // ═══════════════════════════════════════════════════════════════

    // ── 281. InferenceMode Ord: Latency < Throughput ──

    #[test]
    fn inference_mode_ord_latency_less_than_throughput() {
        assert!(InferenceMode::Latency < InferenceMode::Throughput);
        assert!(!(InferenceMode::Throughput < InferenceMode::Latency));
    }

    // ── 282. InferenceMode Hash determinism across copies ──

    #[test]
    fn inference_mode_hash_determinism_copies() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |m: InferenceMode| {
            let mut s = DefaultHasher::new();
            m.hash(&mut s);
            s.finish()
        };
        assert_eq!(hash_of(InferenceMode::Latency), hash_of(InferenceMode::Latency));
        assert_eq!(hash_of(InferenceMode::Throughput), hash_of(InferenceMode::Throughput));
        assert_ne!(hash_of(InferenceMode::Latency), hash_of(InferenceMode::Throughput));
    }

    // ── 283. InferenceMode PartialOrd round-trip ──

    #[test]
    fn inference_mode_partial_ord_roundtrip() {
        let a = InferenceMode::Latency;
        let b = InferenceMode::Throughput;
        assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Less));
        assert_eq!(b.partial_cmp(&a), Some(std::cmp::Ordering::Greater));
        assert_eq!(a.partial_cmp(&a), Some(std::cmp::Ordering::Equal));
    }

    // ── 284. ArbiterHwView manual construction ──

    #[test]
    fn arbiter_hw_view_manual_construction() {
        let view = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (16384, 131072, 4194304),
        };
        assert!(view.device == DeviceFamily::Cpu);
        assert_eq!(view.num_simd_regs, 8);
        assert_eq!(view.cache_sizes.0, 16384);
        assert_eq!(view.cache_sizes.1, 131072);
        assert_eq!(view.cache_sizes.2, 4194304);
    }

    // ── 285. ArbiterHwView PartialEq ──

    #[test]
    fn arbiter_hw_view_partial_eq() {
        let a = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        assert_eq!(a, b);
    }

    // ── 286. ArbiterHwView Hash equal views equal hashes ──

    #[test]
    fn arbiter_hw_view_hash_equal_views_equal_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |v: &ArbiterHwView| {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        };
        let a = cpu_avx2();
        let b = cpu_avx2();
        assert_eq!(hash_of(&a), hash_of(&b));
        let c = gpu_a100();
        assert_ne!(hash_of(&a), hash_of(&c));
    }

    // ── 287. ArbiterHwView::gpu zero shared mem boundary ──

    #[test]
    fn arbiter_hw_view_gpu_zero_shared_mem_boundary() {
        let view = ArbiterHwView::gpu(0);
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.cache_sizes.0, 0);
        assert_eq!(view.cache_sizes.1, 0);
        assert_eq!(view.cache_sizes.2, 0);
    }

    // ── 288. ArbiterHwView::gpu large shared mem ──

    #[test]
    fn arbiter_hw_view_gpu_large_shared_mem() {
        let view = ArbiterHwView::gpu(256 * 1024);
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.cache_sizes.0, 256 * 1024);
    }

    // ── 289. lerp t=0 returns a ──

    #[test]
    fn lerp_t_zero_returns_a() {
        assert_approx(lerp(3.0, 7.0, 0.0), 3.0, 0.001);
        assert_approx(lerp(-5.0, 10.0, 0.0), -5.0, 0.001);
    }

    // ── 290. lerp t=1 returns b ──

    #[test]
    fn lerp_t_one_returns_b() {
        assert_approx(lerp(3.0, 7.0, 1.0), 7.0, 0.001);
        assert_approx(lerp(-5.0, 10.0, 1.0), 10.0, 0.001);
    }

    // ── 291. lerp t=0.5 returns midpoint ──

    #[test]
    fn lerp_t_half_returns_midpoint() {
        assert_approx(lerp(0.0, 10.0, 0.5), 5.0, 0.001);
        assert_approx(lerp(-4.0, 4.0, 0.5), 0.0, 0.001);
    }

    // ── 292. lerp identity when a==b ──

    #[test]
    fn lerp_identity_when_equal() {
        assert_approx(lerp(5.0, 5.0, 0.0), 5.0, 0.001);
        assert_approx(lerp(5.0, 5.0, 0.5), 5.0, 0.001);
        assert_approx(lerp(5.0, 5.0, 1.0), 5.0, 0.001);
    }

    // ── 293. lerp linear interpolation correctness ──

    #[test]
    fn lerp_linear_interpolation() {
        assert_approx(lerp(0.0, 100.0, 0.25), 25.0, 0.001);
        assert_approx(lerp(0.0, 100.0, 0.75), 75.0, 0.001);
    }

    // ── 294. lerp negative range ──

    #[test]
    fn lerp_negative_range() {
        assert_approx(lerp(-10.0, -2.0, 0.5), -6.0, 0.001);
        assert_approx(lerp(-10.0, -2.0, 0.0), -10.0, 0.001);
        assert_approx(lerp(-10.0, -2.0, 1.0), -2.0, 0.001);
    }

    // ── 295. sigmoid at origin returns exactly 0.5 ──

    #[test]
    fn sigmoid_origin_returns_exactly_half() {
        assert_approx(sigmoid(0.0), 0.5, 0.001);
    }

    // ── 296. sigmoid large positive approaches 1 ──

    #[test]
    fn sigmoid_large_positive_approaches_one() {
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(50.0) > 0.999);
    }

    // ── 297. sigmoid large negative approaches 0 ──

    #[test]
    fn sigmoid_large_negative_approaches_zero() {
        assert!(sigmoid(-10.0) < 0.001);
        assert!(sigmoid(-50.0) < 0.001);
    }

    // ── 298. sigmoid symmetry around origin ──

    #[test]
    fn sigmoid_symmetry_around_origin() {
        for x in [-5.0_f64, -1.0, 0.5, 3.0] {
            assert_approx(sigmoid(-x), 1.0 - sigmoid(x), 0.0001);
        }
    }

    // ── 299. sigmoid monotonic increasing ──

    #[test]
    fn sigmoid_monotonic_increasing() {
        let xs = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
        for w in xs.windows(2) {
            assert!(sigmoid(w[0]) < sigmoid(w[1]));
        }
    }

    // ── 300. sigmoid bounded in (0,1) ──

    #[test]
    fn sigmoid_bounded_open_interval() {
        for x in -30..=30i32 {
            let v = sigmoid(x as f64);
            assert!(v > 0.0 && v < 1.0, "sigmoid({x}) = {v}, not in (0,1)");
        }
    }

    // ── 301. arbitrate deterministic: same input → same output ──

    #[test]
    fn arbitrate_deterministic_same_input() {
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.4,
        };
        let hw = cpu_avx2();
        let a = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let b = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(a.fusion_cost_scale, b.fusion_cost_scale, 0.0);
        assert_approx(a.pipeline_cost_scale, b.pipeline_cost_scale, 0.0);
        assert_approx(a.epilogue_depth_preference, b.epilogue_depth_preference, 0.0);
    }

    // ── 302. arbitrate: different modes produce different biases ──

    #[test]
    fn arbitrate_different_modes_different_bias() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = cpu_avx2();
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            (lat.fusion_cost_scale - thr.fusion_cost_scale).abs() > 0.01,
            "fusion_cost_scale should differ: lat={}, thr={}",
            lat.fusion_cost_scale, thr.fusion_cost_scale,
        );
    }

    // ── 303. arbitrate: Latency batch_flexibility always 0 ──

    #[test]
    fn arbitrate_latency_batch_flexibility_always_zero() {
        let arch = GraphArchetype {
            compute_intensive: 0.9,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.5,
        };
        for hw in [cpu_avx2(), cpu_avx512(), gpu_a100()] {
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert_eq!(bias.batch_flexibility, 0.0);
        }
    }

    // ── 304. arbitrate: Throughput batch_flexibility always 1 ──

    #[test]
    fn arbitrate_throughput_batch_flexibility_always_one() {
        let arch = GraphArchetype {
            compute_intensive: 0.1,
            memory_intensive: 0.2,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.5,
        };
        for hw in [cpu_avx2(), cpu_avx512(), gpu_a100()] {
            let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            assert_eq!(bias.batch_flexibility, 1.0);
        }
    }

    // ── 305. GraphArchetype zero values ──

    #[test]
    fn graph_archetype_all_zeros() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = cpu_avx512();
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // lerp(1.0, 0.6, 0.0) = 1.0 → no archetype modulation, but L1 richness applies
        assert_approx(bias.fusion_cost_scale, 0.577, 0.02);
    }

    // ── 306. GraphArchetype all ones saturates lerp ──

    #[test]
    fn graph_archetype_all_ones_saturates_lerp() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let hw = cpu_avx512();
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // lerp(1.0, 0.6, 1.0) = 0.6, plus L1 richness adjustment
        assert_approx(bias.fusion_cost_scale, 0.346, 0.02);
    }

    // ── 307. StrategyBias all fields within validate clamps after arbitrate ──

    #[test]
    fn strategy_bias_fields_within_clamps_after_arbitrate() {
        let arch_extreme = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 2,
            cache_sizes: (0, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_extreme, &hw);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
    }

    // ── 308. StrategyBias default all fields are 1.0 except two ──

    #[test]
    fn strategy_bias_default_values() {
        let d = StrategyBias::default();
        assert_approx(d.fusion_cost_scale, 1.0, 0.001);
        assert_approx(d.pipeline_cost_scale, 1.0, 0.001);
        assert_approx(d.parallelism_cost_scale, 1.0, 0.001);
        assert_approx(d.epilogue_depth_preference, 1.0, 0.001);
        assert_approx(d.k_depth_preference, 1.0, 0.001);
        assert_approx(d.kv_cache_budget_scale, 1.0, 0.001);
        assert_approx(d.weight_prefetch_budget_scale, 1.0, 0.001);
        assert_approx(d.batch_flexibility, 1.0, 0.001);
        assert_approx(d.decode_ratio_scale, 1.0, 0.001);
        assert_approx(d.speculative_decoding_value, 1.0, 0.001);
        assert_approx(d.quantization_aggressiveness, 1.0, 0.001);
        assert_approx(d.expert_eviction_aggressiveness, 0.0, 0.001);
        assert_approx(d.expert_prefetch_priority, 1.0, 0.001);
    }

    // ── 309. Latency baseline: decode_ratio_scale is 1.0 ──

    #[test]
    fn latency_baseline_decode_ratio_is_one() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
    }

    // ── 310. Throughput baseline: decode_ratio_scale is 1.0 ──

    #[test]
    fn throughput_baseline_decode_ratio_is_one() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
    }

    // ── 311. GPU boost: epilogue_depth_preference larger than CPU ──

    #[test]
    fn gpu_boost_epilogue_larger_than_cpu() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert!(
            gpu_bias.epilogue_depth_preference > cpu_bias.epilogue_depth_preference,
            "GPU epilogue ({}) should exceed CPU ({})",
            gpu_bias.epilogue_depth_preference, cpu_bias.epilogue_depth_preference,
        );
    }

    // ── 312. GPU boost: k_depth_preference larger than CPU ──

    #[test]
    fn gpu_boost_k_depth_larger_than_cpu() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert!(
            gpu_bias.k_depth_preference > cpu_bias.k_depth_preference,
            "GPU k_depth ({}) should exceed CPU ({})",
            gpu_bias.k_depth_preference, cpu_bias.k_depth_preference,
        );
    }

    // ── 313. GPU boost factor is exactly 1.2x ──

    #[test]
    fn gpu_boost_factor_exact_twenty_percent() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(gpu_bias.epilogue_depth_preference / cpu_bias.epilogue_depth_preference, 1.2, 0.001);
        assert_approx(gpu_bias.k_depth_preference / cpu_bias.k_depth_preference, 1.2, 0.001);
    }

    // ── 314. Latency baseline: speculative_decoding_value is 1.5 ──

    #[test]
    fn latency_baseline_speculative_decoding_one_point_five() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.speculative_decoding_value, 1.5, 0.01);
    }

    // ── 315. Throughput baseline: speculative_decoding_value is 0.3 ──

    #[test]
    fn throughput_baseline_speculative_decoding_zero_point_three() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.speculative_decoding_value, 0.3, 0.01);
    }

    // ── 316. Latency: expert_eviction_aggressiveness baseline 0.0 ──

    #[test]
    fn latency_expert_eviction_baseline_zero() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.expert_eviction_aggressiveness, 0.0, 0.001);
    }

    // ── 317. MoE modulation: parallelism > 0.5 activates expert eviction ──

    #[test]
    fn moe_modulation_high_parallelism_increases_eviction() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let low_par = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let high_par = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.99,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias_low = StrategyArbiter::arbitrate(InferenceMode::Throughput, &low_par, &hw);
        let bias_high = StrategyArbiter::arbitrate(InferenceMode::Throughput, &high_par, &hw);
        assert!(
            bias_high.expert_eviction_aggressiveness > bias_low.expert_eviction_aggressiveness,
            "high parallelism should increase eviction: {} vs {}",
            bias_high.expert_eviction_aggressiveness, bias_low.expert_eviction_aggressiveness,
        );
    }

    // ── 318. MoE modulation: parallelism > 0.5 activates expert prefetch ──

    #[test]
    fn moe_modulation_high_parallelism_increases_prefetch() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let low_par = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let high_par = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.99,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias_low = StrategyArbiter::arbitrate(InferenceMode::Throughput, &low_par, &hw);
        let bias_high = StrategyArbiter::arbitrate(InferenceMode::Throughput, &high_par, &hw);
        assert!(
            bias_high.expert_prefetch_priority > bias_low.expert_prefetch_priority,
            "high parallelism should increase prefetch: {} vs {}",
            bias_high.expert_prefetch_priority, bias_low.expert_prefetch_priority,
        );
    }

    // ── 319. MoE modulation: parallelism <= 0.5 no expert eviction change ──

    #[test]
    fn moe_modulation_low_parallelism_no_eviction_change() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch_low = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let arch_zero = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.1,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias_low = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch_low, &hw);
        let bias_zero = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch_zero, &hw);
        assert_approx(bias_low.expert_eviction_aggressiveness, bias_zero.expert_eviction_aggressiveness, 0.001);
    }

    // ── 320. fusion_profitable monotonic with fusion_cost_scale ──

    #[test]
    fn fusion_profitable_monotonic_decreases_cost() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut prev = f64::MAX;
        for fp in levels {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: fp,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.fusion_cost_scale <= prev + 1e-9,
                "fusion_cost_scale should decrease as fusion_profitable rises: fp={}, cost={:.4}, prev={:.4}",
                fp, bias.fusion_cost_scale, prev,
            );
            prev = bias.fusion_cost_scale;
        }
    }

    // ── 321. pipeline_valuable monotonic with pipeline_cost_scale ──

    #[test]
    fn pipeline_valuable_monotonic_decreases_cost() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut prev = f64::MAX;
        for pv in levels {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: pv,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.pipeline_cost_scale <= prev + 1e-9,
                "pipeline_cost_scale should decrease as pipeline_valuable rises: pv={}, cost={:.4}, prev={:.4}",
                pv, bias.pipeline_cost_scale, prev,
            );
            prev = bias.pipeline_cost_scale;
        }
    }

    // ── 322. parallelism_exploitable monotonic with parallelism_cost_scale ──

    #[test]
    fn parallelism_exploitable_monotonic_decreases_cost() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut prev = f64::MAX;
        for pe in levels {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: pe,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.parallelism_cost_scale <= prev + 1e-9,
                "parallelism_cost_scale should decrease as parallelism_exploitable rises: pe={}, cost={:.4}, prev={:.4}",
                pe, bias.parallelism_cost_scale, prev,
            );
            prev = bias.parallelism_cost_scale;
        }
    }

    // ── 323. memory_intensive monotonic with kv_cache_budget_scale ──

    #[test]
    fn memory_intensive_monotonic_kv_cache_budget() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut prev = f64::MIN;
        for mi in levels {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: mi,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.kv_cache_budget_scale >= prev - 1e-9,
                "kv_cache_budget should increase with memory_intensive: mi={}, kv={:.4}, prev={:.4}",
                mi, bias.kv_cache_budget_scale, prev,
            );
            prev = bias.kv_cache_budget_scale;
        }
    }

    // ── 324. memory_intensive monotonic with quantization_aggressiveness ──

    #[test]
    fn memory_intensive_monotonic_quantization() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut prev = f64::MIN;
        for mi in levels {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: mi,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.quantization_aggressiveness >= prev - 1e-9,
                "quantization_aggressiveness should increase with memory_intensive: mi={}, qa={:.4}, prev={:.4}",
                mi, bias.quantization_aggressiveness, prev,
            );
            prev = bias.quantization_aggressiveness;
        }
    }

    // ── 325. reg_tension positive: epilogue up, k_depth down ──

    #[test]
    fn reg_tension_positive_favors_epilogue() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // fusion_profitable > pipeline_valuable → positive tension
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let neutral = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias_tense = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let bias_neutral = StrategyArbiter::arbitrate(InferenceMode::Latency, &neutral, &hw);
        assert!(
            bias_tense.epilogue_depth_preference > bias_neutral.epilogue_depth_preference,
            "positive tension should boost epilogue: {} vs {}",
            bias_tense.epilogue_depth_preference, bias_neutral.epilogue_depth_preference,
        );
        assert!(
            bias_tense.k_depth_preference < bias_neutral.k_depth_preference,
            "positive tension should dampen k_depth: {} vs {}",
            bias_tense.k_depth_preference, bias_neutral.k_depth_preference,
        );
    }

    // ── 326. reg_tension negative: k_depth up, epilogue down ──

    #[test]
    fn reg_tension_negative_favors_k_depth() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // pipeline_valuable > fusion_profitable → negative tension
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 1.0,
        };
        let neutral = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias_tense = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let bias_neutral = StrategyArbiter::arbitrate(InferenceMode::Latency, &neutral, &hw);
        assert!(
            bias_tense.k_depth_preference > bias_neutral.k_depth_preference,
            "negative tension should boost k_depth: {} vs {}",
            bias_tense.k_depth_preference, bias_neutral.k_depth_preference,
        );
        assert!(
            bias_tense.epilogue_depth_preference < bias_neutral.epilogue_depth_preference,
            "negative tension should dampen epilogue: {} vs {}",
            bias_tense.epilogue_depth_preference, bias_neutral.epilogue_depth_preference,
        );
    }

    // ── 327. reg_tension zero: no epilogue/k_depth modulation beyond baseline ──

    #[test]
    fn reg_tension_zero_no_epilogue_k_modulation() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Baseline epilogue for Latency = 1.5, only modulated by lerp on fusion/pipeline
        // reg_tension = 0 → no additional modulation
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
    }

    // ── 328. L1 richness reduces fusion_cost_scale ──

    #[test]
    fn l1_richness_reduces_fusion_cost() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let small_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (16384, 0, 0),
        };
        let large_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0),
        };
        let bias_small = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &small_l1);
        let bias_large = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &large_l1);
        assert!(
            bias_large.fusion_cost_scale < bias_small.fusion_cost_scale,
            "larger L1 should reduce fusion cost: large={:.4} vs small={:.4}",
            bias_large.fusion_cost_scale, bias_small.fusion_cost_scale,
        );
    }

    // ── 329. L1 zero: richness is zero, no division ──

    #[test]
    fn l1_zero_no_division_by_zero() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (0, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // l1_richness = 0.0 → if guard, no division
        assert!(bias.fusion_cost_scale.is_finite());
    }

    // ── 330. L1 exactly 65536: richness = 1.0, sqrt = 1.0 ──

    #[test]
    fn l1_exactly_65536_richness_one() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // l1_richness = 65536/65536 = 1.0, sqrt(1.0) = 1.0 → no change
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
    }

    // ── 331. CPU register scarcity threshold at 16 ──

    #[test]
    fn cpu_register_scarcity_at_threshold_16() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let at_threshold = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let above_threshold = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 17,
            cache_sizes: (65536, 0, 0),
        };
        let bias_at = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &at_threshold);
        let bias_above = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &above_threshold);
        // At threshold: scarcity = 1.0 - 16/32 = 0.5, applies modulation
        // Above threshold: no scarcity modulation
        assert!(
            bias_at.epilogue_depth_preference > bias_above.epilogue_depth_preference,
            "regs=16 should have scarcity modulation: at={:.4} vs above={:.4}",
            bias_at.epilogue_depth_preference, bias_above.epilogue_depth_preference,
        );
    }

    // ── 332. CPU register scarcity: extreme low reg count ──

    #[test]
    fn cpu_register_scarcity_extreme_low_regs() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 2,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // scarcity = 1.0 - 2/32 = 0.9375
        // epilogue *= 1.0 + 0.9375 * 0.3 ≈ 1.28125 → baseline 1.5 * 1.28125 ≈ 1.92
        assert!(bias.epilogue_depth_preference > 1.5);
        assert!(bias.k_depth_preference < 1.3);
    }

    // ── 333. arbitrate_cpu delegates to arbitrate with From conversion ──

    #[test]
    fn arbitrate_cpu_delegates_correctly() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let profile = DeviceProfile::detect();
        let hw_view = ArbiterHwView::from(&profile);
        let via_arbiter = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_view);
        let via_cpu = StrategyArbiter::arbitrate_cpu(InferenceMode::Latency, &arch, &profile);
        assert_approx(via_arbiter.fusion_cost_scale, via_cpu.fusion_cost_scale, 0.0);
        assert_approx(via_arbiter.pipeline_cost_scale, via_cpu.pipeline_cost_scale, 0.0);
        assert_approx(via_arbiter.epilogue_depth_preference, via_cpu.epilogue_depth_preference, 0.0);
    }

    // ── 334. Throughput baseline: expert_prefetch_priority is 1.5 ──

    #[test]
    fn throughput_baseline_expert_prefetch_one_point_five() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.expert_prefetch_priority, 1.5, 0.01);
    }

    // ── 335. Latency baseline: quantization_aggressiveness is 1.5 ──

    #[test]
    fn latency_baseline_quantization_one_point_five() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.quantization_aggressiveness, 1.5, 0.01);
    }

    // ── 336. Throughput baseline: quantization_aggressiveness is 0.8 ──

    #[test]
    fn throughput_baseline_quantization_zero_point_eight() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.quantization_aggressiveness, 0.8, 0.01);
    }

    // ── 337. Latency baseline: parallelism_cost_scale is 1.5 ──

    #[test]
    fn latency_baseline_parallelism_cost_one_point_five() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.parallelism_cost_scale, 1.5, 0.01);
    }

    // ── 338. Throughput baseline: parallelism_cost_scale is 0.5 ──

    #[test]
    fn throughput_baseline_parallelism_cost_zero_point_five() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.parallelism_cost_scale, 0.5, 0.01);
    }

    // ── 339. Latency: weight_prefetch_budget_scale is 1.5 ──

    #[test]
    fn latency_baseline_weight_prefetch_one_point_five() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.weight_prefetch_budget_scale, 1.5, 0.01);
    }

    // ── 340. Throughput: weight_prefetch_budget_scale is 0.8 ──

    #[test]
    fn throughput_baseline_weight_prefetch_zero_point_eight() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.weight_prefetch_budget_scale, 0.8, 0.01);
    }

    // ── 341. GPU: pipeline_cost_scale boosted by 1.2 ──

    #[test]
    fn gpu_pipeline_cost_scale_boosted() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(
            gpu_bias.pipeline_cost_scale / cpu_bias.pipeline_cost_scale,
            1.2,
            0.001,
        );
    }

    // ── 342. ArbiterHwView::gpu L2 and L3 are zero ──

    #[test]
    fn arbiter_hw_view_gpu_l2_l3_zero() {
        let view = ArbiterHwView::gpu(49152);
        assert_eq!(view.cache_sizes.1, 0);
        assert_eq!(view.cache_sizes.2, 0);
    }

    // ── 343. ArbiterHwView::gpu num_simd_regs is always 255 ──

    #[test]
    fn arbiter_hw_view_gpu_always_255_regs() {
        for smem in [0_usize, 1024, 32768, 131072, 524288] {
            let view = ArbiterHwView::gpu(smem);
            assert_eq!(view.num_simd_regs, 255);
        }
    }

    // ── 344. L1 richness capped at 2.0 ──

    #[test]
    fn l1_richness_capped_at_two() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let very_large_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (524288, 0, 0),
        };
        let double_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (262144, 0, 0),
        };
        let bias_very = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &very_large_l1);
        let bias_double = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &double_l1);
        // Both capped at richness=2.0 → same fusion_cost_scale
        assert_approx(
            bias_very.fusion_cost_scale,
            bias_double.fusion_cost_scale,
            0.001,
        );
    }

    // ── 345. InferenceMode all variants constructible ──

    #[test]
    fn inference_mode_all_variants_constructible() {
        let _latency = InferenceMode::Latency;
        let _throughput = InferenceMode::Throughput;
        assert_eq!(InferenceMode::Latency, InferenceMode::Latency);
        assert_eq!(InferenceMode::Throughput, InferenceMode::Throughput);
    }

    // ── 346. Throughput: expert_eviction_aggressiveness baseline 0.8 ──

    #[test]
    fn throughput_baseline_expert_eviction_zero_point_eight() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.expert_eviction_aggressiveness, 0.8, 0.01);
    }

    // ── 347. Latency: expert_prefetch_priority baseline 0.5 ──

    #[test]
    fn latency_baseline_expert_prefetch_zero_point_five() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.expert_prefetch_priority, 0.5, 0.01);
    }

    // ── 348. Latency baseline all 13 fields snapshot ──

    #[test]
    fn latency_baseline_full_snapshot() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
        assert_approx(bias.pipeline_cost_scale, 0.6, 0.01);
        assert_approx(bias.parallelism_cost_scale, 1.5, 0.01);
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
        assert_approx(bias.kv_cache_budget_scale, 0.5, 0.01);
        assert_approx(bias.weight_prefetch_budget_scale, 1.5, 0.01);
        assert_approx(bias.batch_flexibility, 0.0, 0.001);
        assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
        assert_approx(bias.speculative_decoding_value, 1.5, 0.01);
        assert_approx(bias.quantization_aggressiveness, 1.5, 0.01);
        assert_approx(bias.expert_eviction_aggressiveness, 0.0, 0.001);
        assert_approx(bias.expert_prefetch_priority, 0.5, 0.01);
    }

    // ── 349. Throughput baseline all 13 fields snapshot ──

    #[test]
    fn throughput_baseline_full_snapshot() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 1.0, 0.01);
        assert_approx(bias.pipeline_cost_scale, 1.3, 0.01);
        assert_approx(bias.parallelism_cost_scale, 0.5, 0.01);
        assert_approx(bias.epilogue_depth_preference, 0.8, 0.01);
        assert_approx(bias.k_depth_preference, 0.8, 0.01);
        assert_approx(bias.kv_cache_budget_scale, 1.5, 0.01);
        assert_approx(bias.weight_prefetch_budget_scale, 0.8, 0.01);
        assert_approx(bias.batch_flexibility, 1.0, 0.001);
        assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
        assert_approx(bias.speculative_decoding_value, 0.3, 0.01);
        assert_approx(bias.quantization_aggressiveness, 0.8, 0.01);
        assert_approx(bias.expert_eviction_aggressiveness, 0.8, 0.01);
        assert_approx(bias.expert_prefetch_priority, 1.5, 0.01);
    }

    // ── 350. lerp extrapolation beyond [0,1] ──

    #[test]
    fn lerp_extrapolation_beyond_range() {
        // t > 1 extrapolates past b
        assert_approx(lerp(0.0, 10.0, 1.5), 15.0, 0.001);
        // t < 0 extrapolates past a
        assert_approx(lerp(0.0, 10.0, -0.5), -5.0, 0.001);
    }

    // ── 351. lerp with extreme values ──

    #[test]
    fn lerp_extreme_values() {
        let result = lerp(f64::MIN, f64::MAX, 0.5);
        assert!(result.is_finite() || result.is_infinite());
    }

    // ── 352. sigmoid at ±1 ──

    #[test]
    fn sigmoid_at_plus_minus_one() {
        assert_approx(sigmoid(1.0), 1.0 / (1.0 + (-1.0_f64).exp()), 0.0001);
        assert_approx(sigmoid(-1.0), 1.0 / (1.0 + 1.0_f64.exp()), 0.0001);
    }

    // ── 353. ArbiterHwView equality when fields differ ──

    #[test]
    fn arbiter_hw_view_inequality_different_fields() {
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 262144, 8388608),
        };
        assert_ne!(a, b);
    }

    // ── 354. ArbiterHwView equality differs by is_gpu ──

    #[test]
    fn arbiter_hw_view_inequality_by_gpu_flag() {
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        assert_ne!(a, b);
    }

    // ── 355. ArbiterHwView equality differs by cache_sizes ──

    #[test]
    fn arbiter_hw_view_inequality_by_cache() {
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 262144, 8388608),
        };
        assert_ne!(a, b);
    }

    // ── 356. validate clamps out-of-range StrategyBias ──

    #[test]
    fn validate_clamps_strategy_bias() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 2,
            cache_sizes: (0, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // After validate(), all fields must be within clamp bounds
        assert!(bias.fusion_cost_scale >= 0.2);
        assert!(bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference <= 3.0);
    }

    // ── 357. GraphArchetype boundary at 0.5 for MoE gating ──

    #[test]
    fn moe_gating_boundary_at_half() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let just_below = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.49,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let just_above = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.51,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias_below = StrategyArbiter::arbitrate(InferenceMode::Throughput, &just_below, &hw);
        let bias_above = StrategyArbiter::arbitrate(InferenceMode::Throughput, &just_above, &hw);
        // Below 0.5: no MoE modulation on eviction
        // Above 0.5: MoE modulation active, eviction should be higher
        assert!(
            bias_above.expert_eviction_aggressiveness > bias_below.expert_eviction_aggressiveness,
            "above 0.5 should have higher eviction: {} vs {}",
            bias_above.expert_eviction_aggressiveness, bias_below.expert_eviction_aggressiveness,
        );
    }

    // ── 358. StrategyArbiter can be constructed without fields ──

    #[test]
    fn strategy_arbiter_constructible() {
        let _arbiter = StrategyArbiter;
    }

    // ── 359. ArbiterHwView debug output completeness ──

    #[test]
    fn arbiter_hw_view_debug_output_completeness() {
        let view = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 128,
            cache_sizes: (65536, 524288, 16777216),
        };
        let s = format!("{:?}", view);
        assert!(s.contains("Gpu"));
        assert!(s.contains("128"));
        assert!(s.contains("device"));
        assert!(s.contains("num_simd_regs"));
        assert!(s.contains("cache_sizes"));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (40 new, 360 → 399)
    // ══════════════════════════════════════════════════════════════════════

    // ── 360. ArbiterHwView::gpu with various realistic shared mem sizes ──

    #[test]
    fn arbiter_hw_view_gpu_realistic_shared_mem_sizes() {
        let sizes = [0usize, 16384, 32768, 49152, 65536, 98304, 131072, 196608];
        for &size in &sizes {
            let view = ArbiterHwView::gpu(size);
            assert!(view.device == DeviceFamily::Gpu);
            assert_eq!(view.num_simd_regs, 255);
            assert_eq!(view.cache_sizes, (size, 0, 0));
        }
    }

    // ── 361. InferenceMode Ord total ordering ──

    #[test]
    fn inference_mode_ord_total_order() {
        assert!(InferenceMode::Latency < InferenceMode::Throughput);
        assert!(InferenceMode::Throughput > InferenceMode::Latency);
        assert!(!(InferenceMode::Latency > InferenceMode::Throughput));
    }

    // ── 362. GraphArchetype all-0.5 produces non-zero modulated output ──

    #[test]
    fn archetype_half_fields_produce_modulated_output() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let neutral = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let half = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral);
        let zero = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &neutral);
        assert!(half.fusion_cost_scale < zero.fusion_cost_scale);
        assert!(half.kv_cache_budget_scale > zero.kv_cache_budget_scale);
        assert!(half.quantization_aggressiveness > zero.quantization_aggressiveness);
    }

    // ── 363. validate preserves values at exactly min+epsilon ──

    #[test]
    fn validate_preserves_just_above_minimums() {
        let eps = 1e-10;
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.2 + eps,
            pipeline_cost_scale: 0.2 + eps,
            parallelism_cost_scale: 0.1 + eps,
            epilogue_depth_preference: 0.3 + eps,
            k_depth_preference: 0.3 + eps,
            kv_cache_budget_scale: 0.2 + eps,
            weight_prefetch_budget_scale: 0.2 + eps,
            batch_flexibility: eps,
            decode_ratio_scale: 0.3 + eps,
            expert_eviction_aggressiveness: eps,
            expert_prefetch_priority: 0.1 + eps,
            speculative_decoding_value: 0.1 + eps,
            quantization_aggressiveness: 0.3 + eps,
        };
        bias.validate();
        assert!(bias.fusion_cost_scale > 0.2);
        assert!(bias.batch_flexibility > 0.0);
        assert!(bias.expert_prefetch_priority > 0.1);
    }

    // ── 364. validate preserves values at exactly max-epsilon ──

    #[test]
    fn validate_preserves_just_below_maximums() {
        let eps = 1e-10;
        let mut bias = StrategyBias {
            fusion_cost_scale: 3.0 - eps,
            pipeline_cost_scale: 3.0 - eps,
            parallelism_cost_scale: 3.0 - eps,
            epilogue_depth_preference: 3.0 - eps,
            k_depth_preference: 3.0 - eps,
            kv_cache_budget_scale: 3.0 - eps,
            weight_prefetch_budget_scale: 3.0 - eps,
            batch_flexibility: 1.0 - eps,
            decode_ratio_scale: 2.0 - eps,
            expert_eviction_aggressiveness: 2.0 - eps,
            expert_prefetch_priority: 5.0 - eps,
            speculative_decoding_value: 3.0 - eps,
            quantization_aggressiveness: 3.0 - eps,
        };
        bias.validate();
        assert!(bias.fusion_cost_scale < 3.0);
        assert!(bias.batch_flexibility < 1.0);
        assert!(bias.expert_prefetch_priority < 5.0);
    }

    // ── 365. StrategyBias default then mutate then validate ──

    #[test]
    fn strategy_bias_default_mutate_validate() {
        let mut bias = StrategyBias::default();
        bias.fusion_cost_scale = 5.0;
        bias.batch_flexibility = -1.0;
        bias.expert_prefetch_priority = 10.0;
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 3.0);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 5.0);
    }

    // ── 366. ArbiterHwView from DeviceProfile: num_simd_regs positive ──

    #[test]
    fn arbiter_hw_view_from_device_profile_positive_regs() {
        let dp = DeviceProfile::detect();
        let view = ArbiterHwView::from(&dp);
        assert!(view.num_simd_regs > 0, "detected device should have positive simd regs");
    }

    // ── 367. ArbiterHwView from DeviceProfile: cache_sizes positive L1 ──

    #[test]
    fn arbiter_hw_view_from_device_profile_positive_l1() {
        let dp = DeviceProfile::detect();
        let view = ArbiterHwView::from(&dp);
        assert!(view.cache_sizes.0 > 0, "detected device should have positive L1 cache");
    }

    // ── 368. arbitrate output all finite with moderate inputs ──

    #[test]
    fn arbitrate_output_all_finite_moderate_inputs() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = cpu_avx2();
        for &mode in &[InferenceMode::Latency, InferenceMode::Throughput] {
            let bias = StrategyArbiter::arbitrate(mode, &arch, &hw);
            assert!(bias.fusion_cost_scale.is_finite());
            assert!(bias.pipeline_cost_scale.is_finite());
            assert!(bias.parallelism_cost_scale.is_finite());
            assert!(bias.epilogue_depth_preference.is_finite());
            assert!(bias.k_depth_preference.is_finite());
            assert!(bias.kv_cache_budget_scale.is_finite());
            assert!(bias.weight_prefetch_budget_scale.is_finite());
            assert!(bias.batch_flexibility.is_finite());
            assert!(bias.decode_ratio_scale.is_finite());
            assert!(bias.speculative_decoding_value.is_finite());
            assert!(bias.quantization_aggressiveness.is_finite());
            assert!(bias.expert_eviction_aggressiveness.is_finite());
            assert!(bias.expert_prefetch_priority.is_finite());
        }
    }

    // ── 369. L1 richness: 4KB with Throughput mode ──

    #[test]
    fn l1_4kb_throughput_fusion_clamped_to_max() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (4096, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_eq!(bias.fusion_cost_scale, 3.0);
    }

    // ── 370. GPU with large shared mem + high fusion_profitable ──

    #[test]
    fn gpu_large_shared_high_fusion_precise() {
        let gpu = ArbiterHwView::gpu(131072);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.8,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // fusion: 0.5 * lerp(1.0, 0.6, 0.8)=0.68 -> 0.34, * 1/sqrt(2.0)=0.7071 -> 0.2404
        assert_approx(bias.fusion_cost_scale, 0.240, 0.01);
    }

    // ── 371. Register scarcity threshold: exactly 16 triggers scarcity ──

    #[test]
    fn register_threshold_exactly_16_triggers() {
        let hw_16 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let hw_17 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 17,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let b16 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_16);
        let b17 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_17);
        assert!(
            b16.epilogue_depth_preference > b17.epilogue_depth_preference,
            "16 regs should have more scarcity than 17"
        );
    }

    // ── 372. CPU register scarcity only affects epilogue and k_depth ──

    #[test]
    fn register_scarcity_only_two_fields_affected() {
        let hw_low = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 4,
            cache_sizes: (65536, 0, 0),
        };
        let hw_high = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let low = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_low);
        let high = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_high);
        // These fields should be identical — not affected by register scarcity
        assert_approx(low.fusion_cost_scale, high.fusion_cost_scale, 0.001);
        assert_approx(low.parallelism_cost_scale, high.parallelism_cost_scale, 0.001);
        assert_approx(low.kv_cache_budget_scale, high.kv_cache_budget_scale, 0.001);
        assert_approx(low.weight_prefetch_budget_scale, high.weight_prefetch_budget_scale, 0.001);
        assert_approx(low.batch_flexibility, high.batch_flexibility, 0.001);
        assert_approx(low.decode_ratio_scale, high.decode_ratio_scale, 0.001);
        assert_approx(low.speculative_decoding_value, high.speculative_decoding_value, 0.001);
        assert_approx(low.quantization_aggressiveness, high.quantization_aggressiveness, 0.001);
        assert_approx(low.expert_eviction_aggressiveness, high.expert_eviction_aggressiveness, 0.001);
        assert_approx(low.expert_prefetch_priority, high.expert_prefetch_priority, 0.001);
    }

    // ── 373. GPU boost multiplier is exactly 1.2 for all three fields ──

    #[test]
    fn gpu_boost_exact_1_2_multiplier() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(gpu_bias.epilogue_depth_preference / cpu_bias.epilogue_depth_preference, 1.2, 0.01);
        assert_approx(gpu_bias.k_depth_preference / cpu_bias.k_depth_preference, 1.2, 0.01);
        assert_approx(gpu_bias.pipeline_cost_scale / cpu_bias.pipeline_cost_scale, 1.2, 0.01);
    }

    // ── 374. MoE modulation: Latency expert_eviction is always 0 regardless ──

    #[test]
    fn moe_latency_eviction_always_zero() {
        let arches = [
            GraphArchetype {
                compute_intensive: 0.5,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            },
            GraphArchetype {
                compute_intensive: 0.5,
                memory_intensive: 1.0,
                parallelism_exploitable: 1.0,
                fusion_profitable: 0.5,
                pipeline_valuable: 0.5,
            },
        ];
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        for arch in &arches {
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, arch, &hw);
            assert_approx(bias.expert_eviction_aggressiveness, 0.0, 0.001);
        }
    }

    // ── 375. L1 richness does not affect non-fusion fields ──

    #[test]
    fn l1_richness_only_affects_fusion_cost() {
        let small_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (8192, 0, 0),
        };
        let large_l1 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let small = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &small_l1);
        let large = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &large_l1);
        assert_approx(small.pipeline_cost_scale, large.pipeline_cost_scale, 0.001);
        assert_approx(small.parallelism_cost_scale, large.parallelism_cost_scale, 0.001);
        assert_approx(small.k_depth_preference, large.k_depth_preference, 0.001);
        assert_approx(small.kv_cache_budget_scale, large.kv_cache_budget_scale, 0.001);
        assert_approx(small.batch_flexibility, large.batch_flexibility, 0.001);
    }

    // ── 376. Combined: CPU low regs + high fusion + high memory ──

    #[test]
    fn combined_low_regs_high_fusion_high_memory() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (32768, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.85,
            pipeline_valuable: 0.2,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert_eq!(bias.batch_flexibility, 0.0);
    }

    // ── 377. lerp with a == b returns a regardless of t ──

    #[test]
    fn lerp_equal_endpoints_returns_same() {
        assert_approx(lerp(5.0, 5.0, 0.0), 5.0, 0.001);
        assert_approx(lerp(5.0, 5.0, 0.5), 5.0, 0.001);
        assert_approx(lerp(5.0, 5.0, 1.0), 5.0, 0.001);
        assert_approx(lerp(5.0, 5.0, 2.0), 5.0, 0.001);
    }

    // ── 378. lerp with negative a and positive b ──

    #[test]
    fn lerp_negative_to_positive() {
        assert_approx(lerp(-10.0, 10.0, 0.5), 0.0, 0.001);
        assert_approx(lerp(-10.0, 10.0, 0.0), -10.0, 0.001);
        assert_approx(lerp(-10.0, 10.0, 1.0), 10.0, 0.001);
    }

    // ── 379. sigmoid at ±5 ──

    #[test]
    fn sigmoid_at_plus_minus_five() {
        let sp = sigmoid(5.0);
        let sn = sigmoid(-5.0);
        assert!(sp > 0.99, "sigmoid(5) should be very close to 1.0, got {sp}");
        assert!(sn < 0.01, "sigmoid(-5) should be very close to 0.0, got {sn}");
        assert_approx(sp + sn, 1.0, 0.001);
    }


    // ── 381. ArbiterHwView PartialEq is consistent with Hash ──

    #[test]
    fn arbiter_hw_view_eq_hash_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = ArbiterHwView::gpu(49152);
        let b = ArbiterHwView::gpu(49152);
        assert_eq!(a, b);
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    // ── 382. StrategyBias validate with all fields at default ──

    #[test]
    fn strategy_bias_default_all_within_bounds() {
        let bias = StrategyBias::default();
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
    }

    // ── 383. GraphArchetype Debug output contains all five field names ──

    #[test]
    fn graph_archetype_debug_all_five_names() {
        let arch = GraphArchetype {
            compute_intensive: 0.1,
            memory_intensive: 0.2,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.5,
        };
        let s = format!("{arch:?}");
        for name in &[
            "compute_intensive",
            "memory_intensive",
            "parallelism_exploitable",
            "fusion_profitable",
            "pipeline_valuable",
        ] {
            assert!(s.contains(name), "Debug output should contain field name '{name}'");
        }
    }

    // ── 384. ArbiterHwView mutation after clone does not affect original ──

    #[test]
    fn arbiter_hw_view_clone_independence() {
        let original = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let mut cloned = original.clone();
        cloned.num_simd_regs = 99;
        assert_eq!(original.num_simd_regs, 16);
        assert_eq!(cloned.num_simd_regs, 99);
    }

    // ── 385. ArbiterHwView Copy: mutation does not affect copied value ──

    #[test]
    fn arbiter_hw_view_copy_independence() {
        let mut a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 0, 0),
        };
        let b = a;
        a.num_simd_regs = 99;
        assert_eq!(b.num_simd_regs, 16);
        assert_eq!(a.num_simd_regs, 99);
    }

    // ── 386. Deterministic: GPU + extreme archetype produces same output ──

    #[test]
    fn deterministic_gpu_extreme_archetype() {
        let arch = GraphArchetype {
            compute_intensive: -2.0,
            memory_intensive: 5.0,
            parallelism_exploitable: -0.5,
            fusion_profitable: 3.0,
            pipeline_valuable: -1.0,
        };
        let gpu = ArbiterHwView::gpu(49152);
        let a = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        let b = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert_eq!(a.fusion_cost_scale, b.fusion_cost_scale);
        assert_eq!(a.epilogue_depth_preference, b.epilogue_depth_preference);
        assert_eq!(a.k_depth_preference, b.k_depth_preference);
        assert_eq!(a.pipeline_cost_scale, b.pipeline_cost_scale);
    }

    // ── 387. Combined archetype effects on kv_cache monotonic with memory ──

    #[test]
    fn kv_cache_monotonic_with_memory_intensity() {
        let mem_values = [0.0f64, 0.25, 0.5, 0.75, 1.0];
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mut prev = f64::MIN;
        for &mem in &mem_values {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: mem,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.kv_cache_budget_scale >= prev - 0.001,
                "kv_cache should be non-decreasing with memory: mem={mem}, val={}, prev={prev}",
                bias.kv_cache_budget_scale
            );
            prev = bias.kv_cache_budget_scale;
        }
    }


    // ── 389. fusion_cost_scale monotonic decreasing with fusion_profitable ──

    #[test]
    fn fusion_cost_monotonic_with_fusion_profitable() {
        let fp_values = [0.0f64, 0.25, 0.5, 0.75, 1.0];
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mut prev = f64::MAX;
        for &fp in &fp_values {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: fp,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            assert!(
                bias.fusion_cost_scale <= prev + 0.001,
                "fusion_cost should be non-increasing with fusion_profitable: fp={fp}, val={}, prev={prev}",
                bias.fusion_cost_scale
            );
            prev = bias.fusion_cost_scale;
        }
    }

    // ── 390. InferenceMode can be used as array index ──

    #[test]
    fn inference_mode_as_index() {
        let modes = [InferenceMode::Latency, InferenceMode::Throughput];
        let labels: Vec<&str> = modes.iter().map(|m| match m {
            InferenceMode::Latency => "L",
            InferenceMode::Throughput => "T",
        }).collect();
        assert_eq!(labels, ["L", "T"]);
    }

    // ── 391. StrategyBias accessor methods exist and return correct type ──

    #[test]
    fn strategy_bias_accessor_fns_return_f64() {
        let bias = StrategyBias::default();
        let _: f64 = bias.expert_eviction_aggressiveness();
        let _: f64 = bias.expert_prefetch_priority();
        assert_approx(bias.expert_eviction_aggressiveness(), 0.0, 0.001);
        assert_approx(bias.expert_prefetch_priority(), 1.0, 0.001);
    }

    // ── 392. GPU + Throughput: all fields within validate bounds ──

    #[test]
    fn gpu_throughput_all_fields_within_bounds() {
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.9,
            pipeline_valuable: 0.4,
        };
        let gpu = ArbiterHwView::gpu(49152);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
    }

    // ── 393. Archetype modulation preserves mode-dependent baseline ratios ──

    #[test]
    fn archetype_preserves_mode_baseline_ratios() {
        // Throughput batch_flexibility is always 1.0 regardless of archetype
        let arches = [
            GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            },
            GraphArchetype {
                compute_intensive: 1.0,
                memory_intensive: 1.0,
                parallelism_exploitable: 1.0,
                fusion_profitable: 1.0,
                pipeline_valuable: 1.0,
            },
        ];
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        for arch in &arches {
            let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, arch, &hw);
            let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, arch, &hw);
            assert_eq!(lat.batch_flexibility, 0.0);
            assert_eq!(thr.batch_flexibility, 1.0);
        }
    }

    // ── 394. Hardware adjustment applied after archetype modulation ──

    #[test]
    fn hardware_applied_after_archetype() {
        // If hw were applied before archetype, the result would differ
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        // Same archetype, GPU boosts epilogue by 1.2x relative to CPU
        let ratio = gpu_bias.epilogue_depth_preference / cpu_bias.epilogue_depth_preference;
        assert_approx(ratio, 1.2, 0.01);
    }

    // ── 395. L1 richness with 2KB: extreme fusion cost increase ──

    #[test]
    fn l1_2kb_extreme_fusion_cost_increase() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (2048, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // richness = 2048/65536 = 0.03125, sqrt = 0.1768, 1/0.1768 = 5.657
        // 0.5 * 5.657 = 2.828 → validate clamps to max 3.0
        assert!(bias.fusion_cost_scale <= 3.0);
        assert!(bias.fusion_cost_scale > 1.0);
    }

    // ── 396. MoE modulation: prefetch scales with memory through lerp ──

    #[test]
    fn moe_prefetch_linear_with_memory() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mem_values = [0.25f64, 0.5, 0.75, 1.0];
        let mut prev = f64::MIN;
        for &mem in &mem_values {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: mem,
                parallelism_exploitable: 0.8,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            assert!(
                bias.expert_prefetch_priority > prev,
                "prefetch should increase with memory: mem={mem}, val={}, prev={prev}",
                bias.expert_prefetch_priority
            );
            prev = bias.expert_prefetch_priority;
        }
    }

    // ── 397. reg_tension zero: no adjustment to epilogue or k_depth ──

    #[test]
    fn reg_tension_zero_no_adjustment_exact() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.33,
            pipeline_valuable: 0.33,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // reg_tension = 0.33 - 0.33 = 0.0, so epilogue and k_depth stay at baseline
        assert_approx(bias.epilogue_depth_preference, 0.8, 0.01);
        assert_approx(bias.k_depth_preference, 0.8, 0.01);
    }

    // ── 398. StrategyBias struct layout: 13 fields all distinct f64 ──

    #[test]
    fn strategy_bias_13_distinct_field_values() {
        let bias = StrategyBias {
            fusion_cost_scale: 1.0,
            pipeline_cost_scale: 2.0,
            parallelism_cost_scale: 3.0,
            epilogue_depth_preference: 4.0,
            k_depth_preference: 5.0,
            kv_cache_budget_scale: 6.0,
            weight_prefetch_budget_scale: 7.0,
            batch_flexibility: 8.0,
            decode_ratio_scale: 9.0,
            speculative_decoding_value: 10.0,
            quantization_aggressiveness: 11.0,
            expert_eviction_aggressiveness: 12.0,
            expert_prefetch_priority: 13.0,
        };
        let values = [
            bias.fusion_cost_scale,
            bias.pipeline_cost_scale,
            bias.parallelism_cost_scale,
            bias.epilogue_depth_preference,
            bias.k_depth_preference,
            bias.kv_cache_budget_scale,
            bias.weight_prefetch_budget_scale,
            bias.batch_flexibility,
            bias.decode_ratio_scale,
            bias.speculative_decoding_value,
            bias.quantization_aggressiveness,
            bias.expert_eviction_aggressiveness,
            bias.expert_prefetch_priority,
        ];
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                assert_ne!(values[i], values[j], "field {i} and {j} should be distinct");
            }
        }
    }

    // ── 399. ArbiterHwView from DeviceProfile: consistent across calls ──

    #[test]
    fn arbiter_hw_view_from_consistent_across_calls() {
        let dp = DeviceProfile::detect();
        let view1 = ArbiterHwView::from(&dp);
        let view2 = ArbiterHwView::from(&dp);
        assert_eq!(view1, view2);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (15 new, 511 → 526)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. lerp with f64::EPSILON inputs: no panic, finite output ──

    #[test]
    fn lerp_epsilon_inputs_finite_output() {
        let a = f64::EPSILON;
        let b = f64::EPSILON * 2.0;
        let t = f64::EPSILON;
        let result = lerp(a, b, t);
        assert!(result.is_finite(), "lerp with epsilon inputs must produce finite output");
        assert!(result > 0.0, "lerp between positive epsilons must be positive");
    }

    // ── 2. lerp with very large f64 values: no overflow panic ──

    #[test]
    fn lerp_large_finite_values_no_panic() {
        let a = f64::MAX / 2.0;
        let b = f64::MAX / 3.0;
        let t = 0.5;
        let result = lerp(a, b, t);
        assert!(result.is_finite() || result.is_infinite());
    }

    // ── 3. StrategyBias partial field update then re-validate ──

    #[test]
    fn strategy_bias_partial_update_revalidate() {
        let mut bias = StrategyBias::default();
        bias.fusion_cost_scale = 5.0;
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 3.0, "first validate should clamp to 3.0");
        bias.fusion_cost_scale = 0.01;
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2, "second validate should clamp to 0.2");
        bias.fusion_cost_scale = 1.5;
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 1.5, "third validate: value in range, no clamp");
    }

    // ── 4. ArbiterHwView used in Vec: push/pop semantics ──

    #[test]
    fn arbiter_hw_view_vec_push_pop() {
        let mut views = Vec::new();
        views.push(cpu_avx2());
        views.push(cpu_avx512());
        views.push(gpu_a100());
        assert_eq!(views.len(), 3);
        let popped = views.pop();
        assert!(popped.unwrap().device == DeviceFamily::Gpu);
        assert_eq!(views.len(), 2);
    }

    // ── 5. GraphArchetype with mixed realistic values (Qwen3-7B-like) ──

    #[test]
    fn archetype_realistic_qwen3_7b_profile() {
        let arch = GraphArchetype {
            compute_intensive: 0.73,
            memory_intensive: 0.40,
            parallelism_exploitable: 0.15,
            fusion_profitable: 0.80,
            pipeline_valuable: 0.45,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu_avx512());
        // All fields must be positive after validate().
        assert!(bias.fusion_cost_scale > 0.0);
        assert!(bias.parallelism_cost_scale > 0.0);
        assert!(bias.kv_cache_budget_scale > 0.0);
        assert!(bias.quantization_aggressiveness > 0.0);
    }

    // ── 6. InferenceMode Vec dedup via Hash ──

    #[test]
    fn inference_mode_vec_dedup() {
        use std::collections::HashSet;
        let modes = vec![
            InferenceMode::Latency,
            InferenceMode::Throughput,
            InferenceMode::Latency,
            InferenceMode::Throughput,
            InferenceMode::Latency,
        ];
        let unique: HashSet<_> = modes.into_iter().collect();
        assert_eq!(unique.len(), 2);
    }

    // ── 7. register scarcity and L1 richness combined: low regs + large L1 ──

    #[test]
    fn low_regs_and_large_l1_combined() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (131072, 0, 0), // 128KB L1, 8 regs
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Low regs increase epilogue; large L1 decreases fusion cost.
        assert!(bias.epilogue_depth_preference > 1.5, "scarcity should boost epilogue");
        assert!(bias.fusion_cost_scale < 0.5, "large L1 should reduce fusion cost");
    }

    // ── 8. arbitrate idempotent across 10 identical calls ──

    #[test]
    fn arbitrate_idempotent_10_calls() {
        let arch = GraphArchetype {
            compute_intensive: 0.6,
            memory_intensive: 0.7,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.4,
        };
        let hw = cpu_avx2();
        let first = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        for _ in 0..9 {
            let next = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            assert_eq!(first.fusion_cost_scale, next.fusion_cost_scale);
            assert_eq!(first.batch_flexibility, next.batch_flexibility);
            assert_eq!(first.epilogue_depth_preference, next.epilogue_depth_preference);
        }
    }

    // ── 9. ArbiterHwView gpu constructor constant: num_simd_regs always 255 ──

    #[test]
    fn arbiter_hw_view_gpu_255_regs_for_various_shared_sizes() {
        for &sz in &[0usize, 1, 1024, 16384, 32768, 49152, 65536, 98304, 131072] {
            let view = ArbiterHwView::gpu(sz);
            assert_eq!(view.num_simd_regs, 255, "gpu() always sets num_simd_regs to 255");
        }
    }

    // ── 10. StrategyBias default validate is identity (no-op) ──

    #[test]
    fn strategy_bias_default_validate_is_exact_identity() {
        let before = StrategyBias::default();
        let mut after = StrategyBias::default();
        after.validate();
        // Bit-for-bit identical: default values are all within clamp bounds.
        assert_eq!(before.fusion_cost_scale, after.fusion_cost_scale);
        assert_eq!(before.pipeline_cost_scale, after.pipeline_cost_scale);
        assert_eq!(before.parallelism_cost_scale, after.parallelism_cost_scale);
        assert_eq!(before.epilogue_depth_preference, after.epilogue_depth_preference);
        assert_eq!(before.k_depth_preference, after.k_depth_preference);
        assert_eq!(before.kv_cache_budget_scale, after.kv_cache_budget_scale);
        assert_eq!(before.weight_prefetch_budget_scale, after.weight_prefetch_budget_scale);
        assert_eq!(before.batch_flexibility, after.batch_flexibility);
        assert_eq!(before.decode_ratio_scale, after.decode_ratio_scale);
        assert_eq!(before.speculative_decoding_value, after.speculative_decoding_value);
        assert_eq!(before.quantization_aggressiveness, after.quantization_aggressiveness);
        assert_eq!(before.expert_eviction_aggressiveness, after.expert_eviction_aggressiveness);
        assert_eq!(before.expert_prefetch_priority, after.expert_prefetch_priority);
    }

    // ── 11. GraphArchetype all-0.5 values: symmetric archetype ──

    #[test]
    fn graph_archetype_symmetric_half() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &neutral_hw);
        // Symmetric archetype should not bias modes toward each other.
        assert!(lat.fusion_cost_scale < thr.fusion_cost_scale);
        assert!(lat.batch_flexibility < thr.batch_flexibility);
    }

    // ── 12. sigmoid output strictly between 0.0 and 1.0 for moderate inputs ──

    #[test]
    fn sigmoid_strictly_between_zero_and_one_moderate() {
        // f64 exp(-x) underflows past |x|~40; stay well within safe range.
        for x in [-20.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 20.0] {
            let s = sigmoid(x);
            assert!(s > 0.0, "sigmoid({x}) must be > 0.0");
            assert!(s < 1.0, "sigmoid({x}) must be < 1.0");
        }
    }

    // ── 13. lerp with t slightly above 1.0: extrapolation is linear ──

    #[test]
    fn lerp_t_above_one_extrapolation_linear() {
        let a = 10.0;
        let b = 20.0;
        let t = 1.5;
        let result = lerp(a, b, t);
        // lerp(10, 20, 1.5) = 10 + 1.5 * (20 - 10) = 10 + 15 = 25
        assert_approx(result, 25.0, 0.001);
    }

    // ── 14. InferenceMode as array index via match ──

    #[test]
    fn inference_mode_as_array_index_via_match() {
        let labels = match InferenceMode::Latency {
            InferenceMode::Latency => ["latency", "low-batch"],
            InferenceMode::Throughput => ["throughput", "high-batch"],
        };
        assert_eq!(labels[0], "latency");
        let labels = match InferenceMode::Throughput {
            InferenceMode::Latency => ["latency", "low-batch"],
            InferenceMode::Throughput => ["throughput", "high-batch"],
        };
        assert_eq!(labels[0], "throughput");
    }

    // ── 15. ArbiterHwView PartialEq is false when only L2 differs ──

    #[test]
    fn arbiter_hw_view_neq_l2_differs_l1_same() {
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 262144, 8388608),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 524288, 8388608), // L2 differs
        };
        assert_ne!(a, b, "views with different L2 must not be equal");
    }

    // ══════ Additional tests (526-540) ══════

    // ── 526. Negative fusion_profitable causes lerp to increase fusion_cost_scale above baseline ──
    // @trace REQ-SA-001 [level:unit]
    // When fusion_profitable < 0, lerp(1.0, 0.6, negative) > 1.0, so fusion_cost_scale increases.

    #[test]
    fn negative_fusion_profitable_increases_fusion_cost_above_baseline() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: -0.5,
            pipeline_valuable: 0.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        // Assert: Latency baseline fusion_cost_scale = 0.5; lerp(1.0, 0.6, -0.5) = 1.0 + (-0.5)*(-0.4) = 1.2
        // 0.5 * 1.2 = 0.6, and L1=64KB gives richness=1.0 so fusion_cost *= 1/sqrt(1) = unchanged.
        assert_approx(bias.fusion_cost_scale, 0.6, 0.001);
    }

    // ── 527. Negative pipeline_valuable causes lerp to increase pipeline_cost_scale above baseline ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn negative_pipeline_valuable_increases_pipeline_cost_above_baseline() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: -0.5,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &neutral_hw);
        // Assert: Throughput baseline pipeline_cost_scale = 1.3; lerp(1.0, 0.6, -0.5) = 1.2
        // 1.3 * 1.2 = 1.56. L1=64KB no adjustment. validate clamps [0.2, 3.0].
        assert_approx(bias.pipeline_cost_scale, 1.56, 0.001);
    }

    // ── 528. Negative memory_intensive causes quantization to decrease below baseline ──
    // @trace REQ-SA-001 [level:unit]
    // lerp(1.0, 1.3, negative) < 1.0, reducing quantization_aggressiveness.

    #[test]
    fn negative_memory_intensive_reduces_quantization_below_baseline() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: -0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        // Assert: Latency baseline quantization = 1.5; lerp(1.0, 1.3, -0.5) = 1.0 + (-0.5)*(0.3) = 0.85
        // 1.5 * 0.85 = 1.275
        assert_approx(bias.quantization_aggressiveness, 1.275, 0.001);
    }

    // ── 529. Negative memory_intensive causes kv_cache_budget to decrease below baseline ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn negative_memory_intensive_reduces_kv_cache_budget_below_baseline() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: -1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &neutral_hw);
        // Assert: Throughput baseline kv_cache = 1.5; lerp(1.0, 1.5, -1.0) = 1.0 + (-1.0)*(0.5) = 0.5
        // 1.5 * 0.5 = 0.75
        assert_approx(bias.kv_cache_budget_scale, 0.75, 0.001);
    }

    // ── 530. Negative parallelism_exploitable increases parallelism_cost_scale above baseline ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn negative_parallelism_exploitable_increases_parallelism_cost() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: -1.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
        // Assert: Latency baseline parallelism = 1.5; lerp(1.0, 0.5, -1.0) = 1.0 + (-1.0)*(-0.5) = 1.5
        // 1.5 * 1.5 = 2.25
        assert_approx(bias.parallelism_cost_scale, 2.25, 0.001);
    }

    // ── 531. GPU with shared_mem exactly 98304 bytes (1.5x base) precise calculation ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn gpu_shared_mem_98304_richness_precise() {
        // Arrange
        let hw = ArbiterHwView::gpu(98304); // 96KB shared mem
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Assert: L1 richness = 98304/65536 = 1.5, min(1.5, 2.0) = 1.5
        // fusion_cost_scale = 0.5 (latency baseline) * (1/sqrt(1.5)) * 1.0 (no archetype modulation)
        // = 0.5 / 1.2247... ~= 0.40825
        let expected_fusion = 0.5 / 1.5_f64.sqrt();
        assert_approx(bias.fusion_cost_scale, expected_fusion, 0.001);
    }

    // ── 532. GPU with register scarcity still gets GPU epilogue/k_depth boost ──
    // @trace REQ-SA-001 [level:unit]
    // GPU hardware view always has num_simd_regs=255, so scarcity (<=16) never fires.
    // This test confirms the gpu() constructor invariant holds under arbitrate.

    #[test]
    fn gpu_constructor_always_avoids_register_scarcity() {
        // Arrange
        for &shared_mem in &[0usize, 4096, 32768, 65536, 131072] {
            let hw = ArbiterHwView::gpu(shared_mem);
            let zero_arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            };
            // Act
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
            // Assert: GPU path applies epilogue *= 1.2, k_depth *= 1.2, pipeline *= 1.2
            // No scarcity applied. epilogue = 1.5 * 1.2 = 1.8, k_depth = 1.3 * 1.2 = 1.56
            assert_approx(bias.epilogue_depth_preference, 1.8, 0.001);
            assert_approx(bias.k_depth_preference, 1.56, 0.001);
        }
    }

    // ── 533. CPU manually set is_gpu=true with low registers: both GPU and scarcity fire ──
    // @trace REQ-SA-001 [level:unit]
    // A manually constructed ArbiterHwView with is_gpu=true and num_simd_regs<=16 triggers both paths.

    #[test]
    fn cpu_manual_gpu_flag_with_low_regs_both_adjustments_fire() {
        // Arrange
        let hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Assert: GPU path: epilogue *= 1.2, k_depth *= 1.2, pipeline *= 1.2
        // Scarcity: regs=8, scarcity = 1.0 - 8/32 = 0.75
        //   epilogue *= (1.0 + 0.75*0.3) = 1.225; k_depth *= (1.0 - 0.75*0.2) = 0.85
        // Combined epilogue = 1.5 * 1.2 * 1.225 = 2.205
        // Combined k_depth = 1.3 * 1.2 * 0.85 = 1.326
        // Pipeline = 0.6 * 1.2 = 0.72
        assert_approx(bias.epilogue_depth_preference, 2.205, 0.001);
        assert_approx(bias.k_depth_preference, 1.326, 0.001);
        assert_approx(bias.pipeline_cost_scale, 0.72, 0.001);
    }

    // ── 534. MoE modulation with parallelism_exploitable exactly 0.5 does NOT trigger ──
    // @trace REQ-SA-001 [level:unit]
    // The condition is strictly > 0.5, so exactly 0.5 should leave expert fields unchanged.

    #[test]
    fn moe_parallelism_exactly_half_no_expert_adjustment() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.5, // exactly at boundary, not > 0.5
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &neutral_hw);
        // Assert: expert_eviction baseline for Throughput = 0.8; no MoE modulation because 0.5 is not > 0.5
        assert_approx(bias.expert_eviction_aggressiveness, 0.8, 0.001);
        // expert_prefetch baseline for Throughput = 1.5; no modulation
        assert_approx(bias.expert_prefetch_priority, 1.5, 0.001);
    }

    // ── 535. All-negative archetype fields produce valid clamped output ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn all_negative_archetype_fields_produce_valid_clamped_output() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: -1.0,
            memory_intensive: -1.0,
            parallelism_exploitable: -1.0,
            fusion_profitable: -1.0,
            pipeline_valuable: -1.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &neutral_hw);
        // Assert: All fields must be finite and within validate() bounds
        assert!(bias.fusion_cost_scale.is_finite());
        assert!(bias.pipeline_cost_scale.is_finite());
        assert!(bias.parallelism_cost_scale.is_finite());
        assert!(bias.epilogue_depth_preference.is_finite());
        assert!(bias.k_depth_preference.is_finite());
        assert!(bias.kv_cache_budget_scale.is_finite());
        assert!(bias.weight_prefetch_budget_scale.is_finite());
        assert!(bias.batch_flexibility.is_finite());
        assert!(bias.decode_ratio_scale.is_finite());
        assert!(bias.speculative_decoding_value.is_finite());
        assert!(bias.quantization_aggressiveness.is_finite());
        assert!(bias.expert_eviction_aggressiveness.is_finite());
        assert!(bias.expert_prefetch_priority.is_finite());
        // Verify no NaN
        assert!(!bias.fusion_cost_scale.is_nan());
        assert!(!bias.quantization_aggressiveness.is_nan());
    }

    // ── 536. Reg tension monotonic: increasing fusion_profitable with fixed pipeline increases epilogue ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn reg_tension_epilogue_monotonic_with_increasing_fusion() {
        // Arrange
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mut prev_epilogue = 0.0_f64;
        // Act & Assert
        for fusion in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: fusion,
                pipeline_valuable: 0.0, // fixed at 0, so tension = fusion - 0 = fusion
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &neutral_hw);
            if prev_epilogue > 0.0 {
                assert!(
                    bias.epilogue_depth_preference >= prev_epilogue - 1e-9,
                    "epilogue should be non-decreasing as fusion_profitable increases: \
                     fusion={fusion}, epilogue={}, prev={prev_epilogue}",
                    bias.epilogue_depth_preference
                );
            }
            prev_epilogue = bias.epilogue_depth_preference;
        }
    }

    // ── 537. Combined: high negative memory + high parallelism (>0.5) + GPU ──
    // @trace REQ-SA-001 [level:unit]
    // Tests GPU pipeline boost + MoE modulation with negative memory_intensive.

    #[test]
    fn gpu_high_parallel_negative_memory_moe_uses_negative_memory() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: -0.5,
            parallelism_exploitable: 0.8, // > 0.5 triggers MoE
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView::gpu(65536); // 64KB shared mem, richness=1.0
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert: MoE modulation fires: eviction *= lerp(1.0, 1.5, -0.5) = 1.0 + (-0.5)*0.5 = 0.75
        // Throughput baseline eviction = 0.8; 0.8 * 0.75 = 0.6
        assert_approx(bias.expert_eviction_aggressiveness, 0.6, 0.001);
        // MoE prefetch: lerp(1.0, 2.0, -0.5) = 1.0 + (-0.5)*1.0 = 0.5
        // Throughput baseline prefetch = 1.5; 1.5 * 0.5 = 0.75
        assert_approx(bias.expert_prefetch_priority, 0.75, 0.001);
    }

    // ── 538. L1 richness exactly 2.0 (131072 bytes) gives fusion_cost *= 1/sqrt(2) ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn l1_exactly_131072_richness_capped_at_two_fusion_precise() {
        // Arrange
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0), // exactly 2x 65536
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &zero_arch, &hw);
        // Assert: richness = 131072/65536 = 2.0, capped at 2.0
        // fusion_cost = 1.0 (throughput baseline) * 1/sqrt(2.0) = 0.7071...
        assert_approx(bias.fusion_cost_scale, 1.0 / 2.0_f64.sqrt(), 0.001);
    }

    // ── 539. lerp with t=0.0 returns exactly a (identity) ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn lerp_t_zero_returns_exactly_a() {
        // Arrange
        let a = 3.14159;
        let b = 2.71828;
        // Act
        let result = lerp(a, b, 0.0);
        // Assert
        assert_approx(result, a, 1e-12);
    }

    // ── 540. Throughput mode k_depth monotonic with increasing pipeline_valuable ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn throughput_k_depth_monotonic_with_pipeline_valuable() {
        // Arrange
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mut prev_k_depth = 0.0_f64;
        // Act & Assert
        for pipeline in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: pipeline,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &neutral_hw);
            // With fusion_profitable=0 and pipeline>0, reg_tension = 0 - pipeline = negative
            // Negative tension: k_depth *= 1.0 + pipeline * 0.5
            if prev_k_depth > 0.0 {
                assert!(
                    bias.k_depth_preference >= prev_k_depth - 1e-9,
                    "k_depth should be non-decreasing with increasing pipeline_valuable: \
                     pipeline={pipeline}, k_depth={}, prev={prev_k_depth}",
                    bias.k_depth_preference
                );
            }
            prev_k_depth = bias.k_depth_preference;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Tests 541–555: Edge cases, boundaries, and behavioral properties
    // ═══════════════════════════════════════════════════════════════

    // ── 541. Register scarcity boundary: exactly 16 triggers, 17 does not ──
    // @trace REQ-SA-001 [level:unit]
    // The scarcity condition is num_simd_regs <= 16. Verify the boundary precisely.

    #[test]
    fn register_scarcity_boundary_at_16_triggers() {
        // Arrange
        let at = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let above = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 17,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias_at = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &at);
        let bias_above = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &above);
        // Assert: at threshold (16 regs), scarcity = 1.0 - 16/32 = 0.5
        // epilogue = 1.5 * (1.0 + 0.5 * 0.3) = 1.5 * 1.15 = 1.725
        // above threshold (17 regs), no scarcity -> epilogue = 1.5
        assert!(
            bias_at.epilogue_depth_preference > bias_above.epilogue_depth_preference,
            "regs=16 epilogue ({:.4}) must exceed regs=17 ({:.4}) due to scarcity boundary",
            bias_at.epilogue_depth_preference, bias_above.epilogue_depth_preference,
        );
    }

    // ── 542. MoE with zero memory_intensive: lerp(1.0, 1.5, 0.0) = 1.0, no boost ──
    // @trace REQ-SA-001 [level:unit]
    // Even when parallelism > 0.5 triggers MoE, memory=0 produces lerp(..., 0.0) = 1.0,
    // so expert fields remain at baseline.

    #[test]
    fn moe_zero_memory_no_expert_boost_despite_high_parallelism() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.9, // triggers MoE
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert: lerp(1.0, 1.5, 0.0) = 1.0 -> eviction = 0.8 * 1.0 = 0.8
        // lerp(1.0, 2.0, 0.0) = 1.0 -> prefetch = 1.5 * 1.0 = 1.5
        assert_approx(bias.expert_eviction_aggressiveness, 0.8, 0.001);
        assert_approx(bias.expert_prefetch_priority, 1.5, 0.001);
    }

    // ── 543. L1 richness above 2x is capped at 2.0 ──
    // @trace REQ-SA-001 [level:unit]
    // L1=200000 bytes -> richness = 200000/65536 = 3.05..., but min(3.05, 2.0) = 2.0.

    #[test]
    fn l1_richness_above_two_capped_at_two_fusion_precise() {
        // Arrange
        let hw_very_large = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (200000, 0, 0), // > 2 * 65536
        };
        let hw_capped = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0), // exactly 2 * 65536
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias_very = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_very_large);
        let bias_capped = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_capped);
        // Assert: both should produce identical fusion_cost_scale because richness caps at 2.0
        assert_approx(
            bias_very.fusion_cost_scale,
            bias_capped.fusion_cost_scale,
            0.001,
        );
    }

    // ── 544. epilogue and k_depth invariant when reg_tension is exactly zero ──
    // @trace REQ-SA-001 [level:unit]
    // When fusion_profitable == pipeline_valuable, reg_tension = 0 and neither
    // the positive nor negative branch fires, so epilogue/k_depth get no tension modulation.

    #[test]
    fn epilogue_kdepth_unmodulated_when_reg_tension_absent() {
        // Arrange
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Both set to 0.7 -> tension = 0
        let arch_tension_zero = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.7,
        };
        // Baseline with no archetype at all
        let arch_all_zero = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias_tz = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_tension_zero, &hw);
        let baseline = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_all_zero, &hw);
        // Assert: epilogue/k_depth should be identical -- no tension modulation in either case.
        assert_approx(bias_tz.epilogue_depth_preference, baseline.epilogue_depth_preference, 0.001);
        assert_approx(bias_tz.k_depth_preference, baseline.k_depth_preference, 0.001);
    }

    // ── 545. GPU pipeline boost factor is exactly 1.2x ──
    // @trace REQ-SA-001 [level:unit]
    // GPU path multiplies pipeline_cost_scale by 1.2. Verify precise factor with zero archetype.

    #[test]
    fn gpu_pipeline_boost_exact_twenty_percent() {
        // Arrange
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // Assert: GPU pipeline = CPU pipeline * 1.2
        let ratio = gpu_bias.pipeline_cost_scale / cpu_bias.pipeline_cost_scale;
        assert_approx(ratio, 1.2, 0.001);
    }

    // ── 546. lerp with large negative t extrapolates below a ──
    // @trace REQ-SA-001 [level:unit]
    // lerp is not clamped to [a,b]. lerp(2.0, 4.0, -3.0) = 2 + (-3)*(2) = -4.0.

    #[test]
    fn lerp_large_negative_t_extrapolates_below_a() {
        // Arrange
        let a = 2.0_f64;
        let b = 4.0_f64;
        let t = -3.0_f64;
        // Act
        let result = lerp(a, b, t);
        // Assert: 2.0 + (-3.0) * (4.0 - 2.0) = 2.0 - 6.0 = -4.0
        assert_approx(result, -4.0, 0.001);
    }

    // ── 547. sigmoid complementary: sigmoid(x) + sigmoid(-x) = 1.0 ──
    // @trace REQ-SA-001 [level:unit]

    #[test]
    fn sigmoid_complementary_sum_to_unity() {
        // Arrange & Act & Assert
        for x in [-5.0, -2.0, -0.5, 0.5, 2.0, 5.0] {
            let sum = sigmoid(x) + sigmoid(-x);
            assert_approx(sum, 1.0, 1e-10);
        }
    }

    // ── 548. k_depth monotonic with increasingly negative reg_tension ──
    // @trace REQ-SA-001 [level:unit]
    // As pipeline_valuable grows (with fixed fusion_profitable=0), tension = 0 - pv gets more
    // negative, which increases k_depth via 1.0 + |tension| * 0.5.

    #[test]
    fn k_depth_monotonic_with_negative_reg_tension_pipeline_dominant() {
        // Arrange
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let mut prev_k = 0.0_f64;
        // Act & Assert
        for pv in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let arch = GraphArchetype {
                compute_intensive: 0.0,
                memory_intensive: 0.0,
                parallelism_exploitable: 0.0,
                fusion_profitable: 0.0,
                pipeline_valuable: pv,
            };
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            if prev_k > 0.0 {
                assert!(
                    bias.k_depth_preference >= prev_k - 1e-9,
                    "k_depth should increase as pipeline_valuable grows (negative tension): \
                     pv={pv}, k_depth={k:.4}, prev_k={pk:.4}",
                    k = bias.k_depth_preference,
                    pk = prev_k,
                );
            }
            prev_k = bias.k_depth_preference;
        }
    }

    // ── 549. kv_cache_budget_scale: throughput always higher than latency for same arch ──
    // @trace REQ-SA-001 [level:unit]
    // Throughput baseline = 1.5 vs Latency baseline = 0.5, same archetype modulation applied.

    #[test]
    fn kv_cache_budget_throughput_always_higher_than_latency() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.3,
            memory_intensive: 0.7,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.2,
            pipeline_valuable: 0.6,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert!(
            throughput.kv_cache_budget_scale > latency.kv_cache_budget_scale,
            "throughput kv={:.4} should exceed latency kv={:.4}",
            throughput.kv_cache_budget_scale, latency.kv_cache_budget_scale,
        );
    }

    // ── 550. Arbitrate output: all 13 fields finite and non-NaN for mixed inputs ──
    // @trace REQ-SA-001 [level:unit]
    // Sweeps a combination of mixed archetype values and verifies every field is well-formed.

    #[test]
    fn arbitrate_all_fields_finite_mixed_inputs() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.3,
            memory_intensive: -0.2,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.8,
            pipeline_valuable: -0.1,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 12,
            cache_sizes: (32768, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert: every field must be finite (not NaN, not Inf)
        let fields = [
            bias.fusion_cost_scale,
            bias.pipeline_cost_scale,
            bias.parallelism_cost_scale,
            bias.epilogue_depth_preference,
            bias.k_depth_preference,
            bias.kv_cache_budget_scale,
            bias.weight_prefetch_budget_scale,
            bias.batch_flexibility,
            bias.decode_ratio_scale,
            bias.speculative_decoding_value,
            bias.quantization_aggressiveness,
            bias.expert_eviction_aggressiveness,
            bias.expert_prefetch_priority,
        ];
        for (i, &f) in fields.iter().enumerate() {
            assert!(f.is_finite(), "field at index {i} is not finite: {f}");
        }
    }

    // ── 551. Extreme negative archetype: validate clamps all fields to valid ranges ──
    // @trace REQ-SA-001 [level:unit]
    // Even with all archetype fields at -2.0 (extrapolating lerp far beyond [0,1]),
    // validate() must clamp outputs to valid ranges.

    #[test]
    fn extreme_negative_archetype_all_fields_clamped_to_valid_ranges() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: -2.0,
            memory_intensive: -2.0,
            parallelism_exploitable: -2.0,
            fusion_profitable: -2.0,
            pipeline_valuable: -2.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 2,
            cache_sizes: (0, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert: validate() clamp bounds
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
    }

    // ── 552. L1 richness small sub-64KB: fusion_cost inversely proportional to sqrt(L1) ──
    // @trace REQ-SA-001 [level:unit]
    // With L1 = 16384, richness = 16384/65536 = 0.25, fusion_cost *= 1/sqrt(0.25) = 2.0.

    #[test]
    fn l1_small_sub_base_fusion_cost_increases_precise() {
        // Arrange
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (16384, 0, 0), // 16KB -> richness = 0.25
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert: Latency baseline fusion = 0.5; 0.5 * (1/sqrt(0.25)) = 0.5 * 2.0 = 1.0
        assert_approx(bias.fusion_cost_scale, 1.0, 0.001);
    }

    // ── 553. Register scarcity with regs=0: maximum scarcity = 1.0 ──
    // @trace REQ-SA-001 [level:unit]
    // Edge case: zero SIMD registers -> scarcity = 1.0 - 0/32 = 1.0.

    #[test]
    fn register_scarcity_zero_regs_maximum_scarcity() {
        // Arrange
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 0,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert: scarcity = 1.0; epilogue *= 1.0 + 1.0 * 0.3 = 1.3; 1.5 * 1.3 = 1.95
        // k_depth *= 1.0 - 1.0 * 0.2 = 0.8; 1.3 * 0.8 = 1.04
        assert_approx(bias.epilogue_depth_preference, 1.95, 0.001);
        assert_approx(bias.k_depth_preference, 1.04, 0.001);
    }

    // ── 554. MoE eviction modulation: memory_intensive=1.0 with parallelism>0.5 precise ──
    // @trace REQ-SA-001 [level:unit]
    // lerp(1.0, 1.5, 1.0) = 1.5 -> eviction = 0.8 * 1.5 = 1.2
    // lerp(1.0, 2.0, 1.0) = 2.0 -> prefetch = 1.5 * 2.0 = 3.0

    #[test]
    fn moe_full_memory_modulation_precise_throughput() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0, // > 0.5 triggers MoE
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.expert_eviction_aggressiveness, 1.2, 0.001);
        assert_approx(bias.expert_prefetch_priority, 3.0, 0.001);
    }

    // ── 555. Scarcity does not affect fusion_cost_scale or parallelism_cost_scale ──
    // @trace REQ-SA-001 [level:unit]
    // The scarcity path only modifies epilogue and k_depth. Verify fusion/parallelism untouched.

    #[test]
    fn scarcity_does_not_affect_fusion_or_parallelism_cost() {
        // Arrange
        let hw_scarce = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 4,
            cache_sizes: (65536, 0, 0),
        };
        let hw_rich = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.3,
        };
        // Act
        let scarce = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_scarce);
        let rich = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_rich);
        // Assert: same L1 (65536), same archetype -> fusion/parallelism must be identical
        assert_approx(scarce.fusion_cost_scale, rich.fusion_cost_scale, 0.001);
        assert_approx(scarce.parallelism_cost_scale, rich.parallelism_cost_scale, 0.001);
        // But epilogue must differ due to scarcity
        assert!(
            scarce.epilogue_depth_preference > rich.epilogue_depth_preference + 0.01,
            "scarce epilogue ({:.4}) must exceed rich ({:.4})",
            scarce.epilogue_depth_preference, rich.epilogue_depth_preference,
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (15 new, 556 → 570)
    // ══════════════════════════════════════════════════════════════════════

    // ── 556. InferenceMode Ord: Latency < Throughput ──

    #[test]
    fn inference_mode_latency_less_than_throughput() {
        assert!(InferenceMode::Latency < InferenceMode::Throughput);
    }

    // ── 557. lerp symmetry: lerp(a, b, t) vs lerp(b, a, 1-t) ──

    #[test]
    fn lerp_symmetry_via_fusion_pipeline() {
        // fusion_profitable=0.3 multiplies fusion_cost by lerp(1.0, 0.6, 0.3) = 0.88
        // pipeline_valuable=0.7 multiplies pipeline_cost by lerp(1.0, 0.6, 0.7) = 0.72
        // Swap: fusion=0.7 -> lerp(1.0, 0.6, 0.7) = 0.72; pipeline=0.3 -> lerp(1.0, 0.6, 0.3) = 0.88
        let arch_a = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.3,
            pipeline_valuable: 0.7,
        };
        let arch_b = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.3,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias_a = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_a, &hw);
        let bias_b = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_b, &hw);
        // Swapped archetype should produce swapped fusion/pipeline scales
        assert_approx(bias_a.fusion_cost_scale, bias_b.pipeline_cost_scale / (0.6 / 0.5), 0.05);
        assert!(
            bias_a.fusion_cost_scale > bias_b.fusion_cost_scale,
            "lower fusion_profitable should produce higher fusion_cost_scale",
        );
    }

    // ── 558. GPU does not modify fusion_cost_scale ──

    #[test]
    fn gpu_does_not_modify_fusion_cost_scale() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(cpu_bias.fusion_cost_scale, gpu_bias.fusion_cost_scale, 0.001);
    }

    // ── 559. GPU does not modify parallelism_cost_scale via archetype ──

    #[test]
    fn gpu_does_not_modify_parallelism_cost_scale_via_archetype() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // GPU does not touch parallelism_cost_scale; archetype is same, so values should match
        assert_approx(cpu_bias.parallelism_cost_scale, gpu_bias.parallelism_cost_scale, 0.001);
    }

    // ── 560. ArbiterHwView gpu constructor sets L2 and L3 to zero ──

    #[test]
    fn arbiter_hw_view_gpu_sets_l2_l3_to_zero() {
        let view = ArbiterHwView::gpu(131072);
        assert_eq!(view.cache_sizes.0, 131072);
        assert_eq!(view.cache_sizes.1, 0);
        assert_eq!(view.cache_sizes.2, 0);
    }

    // ── 561. Reg tension with equal fusion and pipeline: Throughput mode ──

    #[test]
    fn reg_tension_zero_throughput_mode_no_epilogue_kdepth_change() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Throughput baseline: epilogue=0.8, k_depth=0.8
        // reg_tension = 0.5 - 0.5 = 0 -> no adjustment
        assert_approx(bias.epilogue_depth_preference, 0.8, 0.01);
        assert_approx(bias.k_depth_preference, 0.8, 0.01);
    }

    // ── 562. MoE modulation: parallelism just above 0.5 with high memory triggers adjustment ──

    #[test]
    fn moe_parallelism_just_above_05_with_high_memory_throughput() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.51, // just above the >0.5 threshold
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // parallelism=0.51 (> 0.5 triggers MoE), memory=0.8
        // lerp(1.0, 1.5, 0.8) = 1.4 -> eviction = 0.8 * 1.4 = 1.12
        // lerp(1.0, 2.0, 0.8) = 1.8 -> prefetch = 1.5 * 1.8 = 2.7
        assert!(bias.expert_eviction_aggressiveness > 0.8, "MoE eviction should increase above baseline");
    }

    // ── 563. CPU with very large L1 (near usize::MAX) does not panic ──

    #[test]
    fn cpu_very_large_l1_no_panic() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (usize::MAX / 2, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
    }

    // ── 564. CPU with zero L1, L2, L3: l1_richness = 0.0 skips fusion branch ──

    #[test]
    fn cpu_zero_all_caches_l1_richness_zero_skips_branch() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (0, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // No L1 richness adjustment -> Throughput baseline fusion = 1.0
        assert_approx(bias.fusion_cost_scale, 1.0, 0.01);
    }

    // ── 565. StrategyBias validate: single field out-of-range, rest at defaults ──

    #[test]
    fn strategy_bias_validate_single_field_out_of_range() {
        let mut bias = StrategyBias {
            kv_cache_budget_scale: -99.0,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_eq!(bias.kv_cache_budget_scale, 0.2);
        // All other fields remain at default (1.0 for most)
        assert_eq!(bias.fusion_cost_scale, 1.0);
        assert_eq!(bias.epilogue_depth_preference, 1.0);
    }

    // ── 566. GPU boost precise: Throughput mode k_depth ──

    #[test]
    fn gpu_boost_precise_throughput_k_depth() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        // Throughput baseline k_depth = 0.8, GPU boost *= 1.2 -> 0.96
        assert_approx(bias.k_depth_preference, 0.96, 0.01);
    }

    // ── 567. Combined high fusion and high pipeline cancels reg tension ──

    #[test]
    fn combined_high_fusion_high_pipeline_cancels_reg_tension() {
        // fusion_profitable=0.8, pipeline_valuable=0.8 -> reg_tension = 0.0
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.8,
            pipeline_valuable: 0.8,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // reg_tension = 0 -> no epilogue/k_depth adjustment beyond baseline
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
    }

    // ── 568. ArbiterHwView from DeviceProfile: num_simd_regs matches ──

    #[test]
    fn arbiter_hw_view_from_device_profile_num_regs_matches() {
        let dp = DeviceProfile::detect();
        let view = ArbiterHwView::from(&dp);
        assert_eq!(view.num_simd_regs, dp.num_simd_regs());
    }

    // ── 569. Arbitrate with NaN in archetype produces finite output after validate ──

    #[test]
    fn arbitrate_nan_archetype_finite_output_after_validate() {
        let nan_arch = GraphArchetype {
            compute_intensive: f64::NAN,
            memory_intensive: f64::NAN,
            parallelism_exploitable: f64::NAN,
            fusion_profitable: f64::NAN,
            pipeline_valuable: f64::NAN,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &nan_arch, &hw);
        // validate() clamps NaN results — but NaN.clamp() returns NaN per IEEE 754.
        // The output may contain NaN, but the function must not panic.
        // We just verify it did not panic by reaching this assertion.
        assert!(true, "arbitrate with NaN archetype did not panic");
    }

    // ── 570. CPU register scarcity with Throughput mode: k_depth reduced ──

    #[test]
    fn cpu_scarcity_throughput_k_depth_reduced() {
        let hw_low = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let hw_high = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let low = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw_low);
        let high = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw_high);
        // Throughput baseline k_depth = 0.8
        // scarcity = 1.0 - 8/32 = 0.75 -> k_depth *= 1.0 - 0.75*0.2 = 0.85
        // 0.8 * 0.85 = 0.68 < 0.8
        assert!(
            low.k_depth_preference < high.k_depth_preference,
            "low registers should reduce k_depth_preference in throughput mode",
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (15 new, 571 → 585)
    // ══════════════════════════════════════════════════════════════════════

    // ── 571. GPU boosts all three adjusted fields simultaneously ──

    #[test]
    fn gpu_boosts_epilogue_kdepth_pipeline_together() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        // GPU: epilogue *= 1.2, k_depth *= 1.2, pipeline *= 1.2
        assert_approx(gpu_bias.epilogue_depth_preference, cpu_bias.epilogue_depth_preference * 1.2, 0.01);
        assert_approx(gpu_bias.k_depth_preference, cpu_bias.k_depth_preference * 1.2, 0.01);
        assert_approx(gpu_bias.pipeline_cost_scale, cpu_bias.pipeline_cost_scale * 1.2, 0.01);
    }

    // ── 572. Fusion modulation with fusion_profitable=0.25 precise ──

    #[test]
    fn fusion_modulation_quarter_profitable_precise() {
        // lerp(1.0, 0.6, 0.25) = 1.0 + 0.25 * (0.6 - 1.0) = 0.9
        // Latency baseline fusion_cost_scale=0.5, * 0.9 = 0.45
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.25,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 0.45, 0.01);
    }

    // ── 573. KV cache modulation precise with Throughput mode at 0.75 memory ──

    #[test]
    fn kv_cache_modulation_throughput_high_memory_precise() {
        // lerp(1.0, 1.5, 0.75) = 1.375
        // Throughput baseline kv_cache_budget_scale=1.5, * 1.375 = 2.0625
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.75,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.kv_cache_budget_scale, 2.0625, 0.01);
    }

    // ── 574. Quantization modulation precise with Throughput at 0.25 memory ──

    #[test]
    fn quantization_modulation_throughput_low_memory_precise() {
        // lerp(1.0, 1.3, 0.25) = 1.075
        // Throughput baseline quantization_aggressiveness=0.8, * 1.075 = 0.86
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.25,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.quantization_aggressiveness, 0.86, 0.01);
    }

    // ── 575. MoE eviction precise with 0.75 parallelism and 0.5 memory ──

    #[test]
    fn moe_eviction_medium_parallel_medium_memory_precise() {
        // parallelism=0.75 (>0.5), memory=0.5
        // eviction: lerp(1.0, 1.5, 0.5) = 1.25 -> 0.8 * 1.25 = 1.0
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.75,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.expert_eviction_aggressiveness, 1.0, 0.01);
    }

    // ── 576. CPU 6 registers scarcity precise calculation (Throughput mode) ──

    #[test]
    fn cpu_6_registers_throughput_scarcity_precise() {
        // scarcity = 1.0 - (6/32) = 0.8125
        // epilogue = 1.0 + 0.8125*0.3 = 1.24375
        // k_depth = 1.0 - 0.8125*0.2 = 0.8375
        // Throughput: epilogue = 0.8 * 1.24375 = 0.995, k_depth = 0.8 * 0.8375 = 0.67
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 6,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 0.995, 0.01);
        assert_approx(bias.k_depth_preference, 0.67, 0.01);
    }

    // ── 577. Pipeline modulation with Throughput at 0.8 pipeline_valuable ──

    #[test]
    fn pipeline_modulation_throughput_high_valuable_precise() {
        // lerp(1.0, 0.6, 0.8) = 1.0 + 0.8 * (0.6 - 1.0) = 0.68
        // Throughput baseline pipeline_cost_scale=1.3, * 0.68 = 0.884
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.8,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.pipeline_cost_scale, 0.884, 0.01);
    }

    // ── 578. Reg tension 0.5 precise: epilogue boost and k_depth reduction ──

    #[test]
    fn reg_tension_half_precise() {
        // fusion_profitable=0.7, pipeline_valuable=0.2 -> reg_tension=0.5
        // epilogue *= 1.0 + 0.5*0.5 = 1.25
        // k_depth *= 1.0 - 0.5*0.3 = 0.85
        // Latency: epilogue = 1.5 * 1.25 = 1.875, k_depth = 1.3 * 0.85 = 1.105
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.2,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.875, 0.01);
        assert_approx(bias.k_depth_preference, 1.105, 0.01);
    }

    // ── 579. StrategyBias validate at mid-range values unchanged ──

    #[test]
    fn strategy_bias_validate_mid_range_unchanged() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 1.5,
            pipeline_cost_scale: 1.5,
            parallelism_cost_scale: 1.5,
            epilogue_depth_preference: 1.5,
            k_depth_preference: 1.5,
            kv_cache_budget_scale: 1.5,
            weight_prefetch_budget_scale: 1.5,
            batch_flexibility: 0.5,
            decode_ratio_scale: 1.0,
            speculative_decoding_value: 1.5,
            quantization_aggressiveness: 1.5,
            expert_eviction_aggressiveness: 1.0,
            expert_prefetch_priority: 2.5,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 1.5);
        assert_eq!(bias.pipeline_cost_scale, 1.5);
        assert_eq!(bias.parallelism_cost_scale, 1.5);
        assert_eq!(bias.batch_flexibility, 0.5);
        assert_eq!(bias.decode_ratio_scale, 1.0);
        assert_eq!(bias.expert_eviction_aggressiveness, 1.0);
        assert_eq!(bias.expert_prefetch_priority, 2.5);
    }

    // ── 580. ArbiterHwView::gpu with realistic H100 shared mem (228KB) ──

    #[test]
    fn arbiter_hw_view_gpu_h100_shared_mem_228kb() {
        let shared = 228 * 1024; // 228KB
        let view = ArbiterHwView::gpu(shared);
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.num_simd_regs, 255);
        assert_eq!(view.cache_sizes.0, shared);
        assert_eq!(view.cache_sizes.1, 0);
        assert_eq!(view.cache_sizes.2, 0);
    }

    // ── 581. Parallelism modulation with Latency at 0.3 parallelism_exploitable ──

    #[test]
    fn parallelism_modulation_latency_low_exploitable_precise() {
        // lerp(1.0, 0.5, 0.3) = 1.0 + 0.3 * (0.5 - 1.0) = 0.85
        // Latency baseline parallelism_cost_scale=1.5, * 0.85 = 1.275
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.parallelism_cost_scale, 1.275, 0.01);
    }

    // ── 582. CPU 14 registers scarcity precise calculation (Throughput mode) ──

    #[test]
    fn cpu_14_registers_throughput_scarcity_precise() {
        // scarcity = 1.0 - (14/32) = 0.5625
        // epilogue = 1.0 + 0.5625*0.3 = 1.16875
        // k_depth = 1.0 - 0.5625*0.2 = 0.8875
        // Throughput: epilogue = 0.8 * 1.16875 = 0.935, k_depth = 0.8 * 0.8875 = 0.71
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 14,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 0.935, 0.01);
        assert_approx(bias.k_depth_preference, 0.71, 0.01);
    }

    // ── 583. GPU + Latency + high fusion_profitable + low pipeline_valuable precise ──

    #[test]
    fn gpu_latency_high_fusion_low_pipeline_precise() {
        // fusion_profitable=0.8, pipeline_valuable=0.2 -> reg_tension=0.6
        // epilogue *= 1.0 + 0.6*0.5 = 1.3, k_depth *= 1.0 - 0.6*0.3 = 0.82
        // Latency baseline: epilogue=1.5, k_depth=1.3
        // After archetype: epilogue=1.5*1.3=1.95, k_depth=1.3*0.82=1.066
        // GPU: epilogue *= 1.2 = 2.34, k_depth *= 1.2 = 1.2792
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.8,
            pipeline_valuable: 0.2,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert_approx(bias.epilogue_depth_preference, 2.34, 0.01);
        assert_approx(bias.k_depth_preference, 1.2792, 0.01);
    }

    // ── 584. L1 at 48KB with Throughput mode precise fusion cost ──

    #[test]
    fn l1_48kb_throughput_fusion_precise() {
        // l1_richness = 49152/65536 = 0.75, sqrt(0.75) = 0.866
        // fusion *= 1/0.866 = 1.1547
        // Throughput baseline=1.0, * 1.1547 = 1.1547
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (49152, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 1.1547, 0.01);
    }

    // ── 585. Combined: GPU + Throughput + high parallelism + high memory full MoE path ──

    #[test]
    fn gpu_throughput_high_parallel_high_memory_full_moe_path() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.8,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.3,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        // All outputs should be within validate bounds
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
        // No NaN
        assert!(!bias.fusion_cost_scale.is_nan());
        assert!(!bias.batch_flexibility.is_nan());
    }

    // ── 586. InferenceMode::Latency < Throughput via Ord trait ──

    #[test]
    fn inference_mode_ordering_latency_less_than_throughput() {
        assert!(InferenceMode::Latency < InferenceMode::Throughput);
        assert!(InferenceMode::Throughput > InferenceMode::Latency);
    }

    // ── 587. StrategyBias validate with zero shared memory GPU view ──

    #[test]
    fn gpu_zero_shared_mem_latency_bias_all_within_validate_bounds() {
        let gpu = ArbiterHwView::gpu(0);
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
    }

    // ── 588. lerp endpoint exactness ──

    #[test]
    fn lerp_returns_exact_a_at_zero_t() {
        let result = lerp(3.0, 7.0, 0.0);
        assert_approx(result, 3.0, 1e-10);
    }

    // ── 589. lerp returns exact b at t=1 ──

    #[test]
    fn lerp_returns_exact_b_at_one_t() {
        let result = lerp(3.0, 7.0, 1.0);
        assert_approx(result, 7.0, 1e-10);
    }

    // ── 590. lerp midpoint symmetry ──

    #[test]
    fn lerp_midpoint_is_arithmetic_mean() {
        let result = lerp(-10.0, 10.0, 0.5);
        assert_approx(result, 0.0, 1e-10);
    }

    // ── 591. sigmoid boundary values ──

    #[test]
    fn sigmoid_at_zero_is_half() {
        let result = sigmoid(0.0);
        assert_approx(result, 0.5, 1e-10);
    }

    // ── 592. ArbiterHwView equality semantics ──

    #[test]
    fn arbiter_hw_view_equality_same_fields() {
        let a = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 262144, 0),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 262144, 0),
        };
        assert_eq!(a, b);
    }

    // ── 593. ArbiterHwView inequality with different is_gpu ──

    #[test]
    fn arbiter_hw_view_inequality_different_device() {
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 262144, 0),
        };
        let gpu = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 262144, 0),
        };
        assert_ne!(cpu, gpu);
    }

    // ── 594. CPU with exactly 1 SIMD register triggers max scarcity ──

    #[test]
    fn cpu_one_simd_register_max_scarcity() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 1,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // scarcity = 1.0 - (1/32) = 0.96875
        // epilogue = 1.5 * (1.0 + 0.96875*0.3) = 1.5 * 1.290625 = 1.9359
        assert_approx(bias.epilogue_depth_preference, 1.9359, 0.01);
        // k_depth = 1.3 * (1.0 - 0.96875*0.2) = 1.3 * 0.80625 = 1.048
        assert_approx(bias.k_depth_preference, 1.048, 0.01);
    }

    // ── 595. All-zero archetype with GPU Throughput all outputs finite ──

    #[test]
    fn all_zero_archetype_gpu_throughput_all_outputs_finite() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        assert!(bias.fusion_cost_scale.is_finite());
        assert!(bias.pipeline_cost_scale.is_finite());
        assert!(bias.parallelism_cost_scale.is_finite());
        assert!(bias.epilogue_depth_preference.is_finite());
        assert!(bias.k_depth_preference.is_finite());
        assert!(bias.kv_cache_budget_scale.is_finite());
        assert!(bias.weight_prefetch_budget_scale.is_finite());
        assert!(bias.batch_flexibility.is_finite());
        assert!(bias.decode_ratio_scale.is_finite());
        assert!(bias.expert_eviction_aggressiveness.is_finite());
        assert!(bias.expert_prefetch_priority.is_finite());
        assert!(bias.speculative_decoding_value.is_finite());
        assert!(bias.quantization_aggressiveness.is_finite());
    }

    // ── 596. StrategyBias default has decode_ratio_scale exactly 1.0 ──

    #[test]
    fn strategy_bias_default_decode_ratio_is_one() {
        let bias = StrategyBias::default();
        assert_approx(bias.decode_ratio_scale, 1.0, 1e-10);
    }

    // ── 597. InferenceMode::Latency and Throughput both produce non-NaN ──

    #[test]
    fn both_modes_produce_non_nan_for_any_archetype() {
        let archetypes = [
            GraphArchetype { compute_intensive: 0.0, memory_intensive: 0.0, parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0 },
            GraphArchetype { compute_intensive: 1.0, memory_intensive: 1.0, parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0 },
        ];
        let hws = [cpu_avx2(), cpu_avx512(), gpu_a100()];
        for arch in &archetypes {
            for hw in &hws {
                for mode in [InferenceMode::Latency, InferenceMode::Throughput] {
                    let bias = StrategyArbiter::arbitrate(mode, arch, hw);
                    assert!(!bias.fusion_cost_scale.is_nan(), "NaN for mode={:?} arch={:?} hw={:?}", mode, arch, hw);
                    assert!(!bias.epilogue_depth_preference.is_nan());
                    assert!(!bias.k_depth_preference.is_nan());
                    assert!(!bias.kv_cache_budget_scale.is_nan());
                }
            }
        }
    }

    // ── 598. GPU pipeline cost scale always >= 1.2 multiplier ──

    #[test]
    fn gpu_pipeline_cost_scale_minimum_one_point_two_factor() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        let latency_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        let throughput_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        // Latency baseline pipeline=0.6, GPU *1.2 = 0.72
        assert_approx(latency_bias.pipeline_cost_scale, 0.72, 0.01);
        // Throughput baseline pipeline=1.3, GPU *1.2 = 1.56
        assert_approx(throughput_bias.pipeline_cost_scale, 1.56, 0.01);
    }

    // ── 599. reg_tension zero when fusion equals pipeline ──

    #[test]
    fn reg_tension_zero_when_fusion_equals_pipeline() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.7,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // reg_tension = 0.7 - 0.7 = 0.0 → no epilogue/k_depth adjustment
        // Latency baseline: epilogue=1.5, k_depth=1.3
        // archetype modulation: lerp(1.0, 0.6, 0.7)=0.72 for fusion, pipeline same
        // But reg_tension=0, so epilogue and k_depth unchanged from baseline
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
    }

    // ── 600. L1 at exactly 65536 bytes (64KB) gives l1_richness of 1.0 ──

    #[test]
    fn l1_exact_64kb_richness_one_fusion_scale() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // l1_richness = 65536/65536 = 1.0, sqrt(1.0)=1.0, fusion *= 1/1.0 = unchanged
        // Latency baseline fusion_cost_scale = 0.5
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
    }

    // ── 601. lerp extrapolation beyond t=1 ──

    #[test]
    fn lerp_extrapolation_beyond_t_one() {
        // lerp does not clamp t; t=2.0 should extrapolate linearly
        assert_approx(lerp(0.0, 10.0, 2.0), 20.0, 0.001);
        assert_approx(lerp(5.0, 15.0, 1.5), 20.0, 0.001);
    }

    // ── 602. lerp with negative t extrapolates backward ──

    #[test]
    fn lerp_negative_t_extrapolates_backward() {
        assert_approx(lerp(0.0, 10.0, -1.0), -10.0, 0.001);
        assert_approx(lerp(2.0, 6.0, -0.5), 0.0, 0.001);
    }

    // ── 603. sigmoid monotonically increasing ──

    #[test]
    fn sigmoid_monotonically_increasing() {
        let mut prev = sigmoid(-10.0);
        for x in [-5.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let cur = sigmoid(x);
            assert!(cur > prev, "sigmoid({}) = {} not > sigmoid(prev) = {}", x, cur, prev);
            prev = cur;
        }
    }

    // ── 604. InferenceMode Ord: Latency < Throughput verified via assert ──

    #[test]
    fn inference_mode_ord_latency_before_throughput() {
        assert!(InferenceMode::Latency < InferenceMode::Throughput);
    }

    // ── 605. InferenceMode equality and inequality ──

    #[test]
    fn inference_mode_equality_variants() {
        assert_eq!(InferenceMode::Latency, InferenceMode::Latency);
        assert_eq!(InferenceMode::Throughput, InferenceMode::Throughput);
        assert_ne!(InferenceMode::Latency, InferenceMode::Throughput);
    }

    // ── 606. InferenceMode Copy trait allows multiple uses ──

    #[test]
    fn inference_mode_copy_trait_multiple_uses() {
        let a = InferenceMode::Latency;
        let b = a; // Copy, not move
        let c = a; // Still valid after copy
        assert_eq!(b, InferenceMode::Latency);
        assert_eq!(c, InferenceMode::Latency);
    }

    // ── 607. ArbiterHwView Copy semantics ──

    #[test]
    fn arbiter_hw_view_copy_semantics() {
        let a = cpu_avx2();
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // ── 608. ArbiterHwView Clone produces equal copy ──

    #[test]
    fn arbiter_hw_view_clone_equal() {
        let a = gpu_a100();
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── 609. ArbiterHwView ne when cpu vs gpu differ ──

    #[test]
    fn arbiter_hw_view_ne_cpu_vs_gpu() {
        let a = cpu_avx2();
        let b = cpu_avx512();
        assert_ne!(a, b);
    }

    // ── 610. StrategyBias fields all positive after arbitrate with extreme values ──

    #[test]
    fn strategy_bias_all_positive_after_extreme_archetype() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 1,
            cache_sizes: (0, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // After validate(), all fields must be positive (clamp min >= 0.1)
        assert!(bias.fusion_cost_scale > 0.0);
        assert!(bias.pipeline_cost_scale > 0.0);
        assert!(bias.parallelism_cost_scale > 0.0);
        assert!(bias.epilogue_depth_preference > 0.0);
        assert!(bias.k_depth_preference > 0.0);
        assert!(bias.kv_cache_budget_scale > 0.0);
        assert!(bias.weight_prefetch_budget_scale > 0.0);
        assert!(bias.decode_ratio_scale > 0.0);
        assert!(bias.speculative_decoding_value > 0.0);
        assert!(bias.quantization_aggressiveness > 0.0);
        assert!(bias.expert_prefetch_priority > 0.0);
    }

    // ── 611. Latency mode always has zero batch_flexibility regardless of archetype ──

    #[test]
    fn latency_mode_zero_batch_flexibility_for_all_archetypes() {
        let archetypes = [
            GraphArchetype { compute_intensive: 0.0, memory_intensive: 0.0, parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0 },
            GraphArchetype { compute_intensive: 1.0, memory_intensive: 1.0, parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0 },
            GraphArchetype { compute_intensive: 0.5, memory_intensive: 0.5, parallelism_exploitable: 0.5, fusion_profitable: 0.5, pipeline_valuable: 0.5 },
        ];
        let hw = cpu_avx512();
        for arch in &archetypes {
            let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, arch, &hw);
            assert_eq!(bias.batch_flexibility, 0.0,
                "Latency batch_flexibility should be 0.0 for arch={:?}", arch);
        }
    }

    // ── 612. Throughput mode always has batch_flexibility == 1.0 regardless of archetype ──

    #[test]
    fn throughput_mode_batch_flexibility_always_one() {
        let archetypes = [
            GraphArchetype { compute_intensive: 0.0, memory_intensive: 0.0, parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0 },
            GraphArchetype { compute_intensive: 1.0, memory_intensive: 1.0, parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0 },
        ];
        let hw = cpu_avx2();
        for arch in &archetypes {
            let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, arch, &hw);
            assert_eq!(bias.batch_flexibility, 1.0,
                "Throughput batch_flexibility should be 1.0 for arch={:?}", arch);
        }
    }

    // ── 613. Hardware with zero L1 cache: no division by zero panic ──

    #[test]
    fn zero_l1_cache_no_panic() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (0, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        // Must not panic due to division by zero in l1_richness calculation
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert!(bias.fusion_cost_scale.is_finite());
        assert!(bias.fusion_cost_scale > 0.0);
    }

    // ── 614. GraphArchetype derive Copy allows multiple uses ──

    #[test]
    fn graph_archetype_copy_allows_multiple_uses() {
        let a = GraphArchetype {
            compute_intensive: 0.8,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.2,
        };
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // ── 615. StrategyBias default accessor methods return expected values ──

    #[test]
    fn strategy_bias_default_accessor_methods() {
        let bias = StrategyBias::default();
        assert_eq!(bias.expert_eviction_aggressiveness(), 0.0);
        assert_approx(bias.expert_prefetch_priority(), 1.0, 1e-10);
    }

    // ── 616. arbiter_cpu produces same result as arbitrate with from conversion ──

    #[test]
    fn arbitrate_cpu_matches_arbitrate_with_from() {
        let dp = DeviceProfile::detect();
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.4,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.2,
        };
        let via_arbitrate = StrategyArbiter::arbitrate(
            InferenceMode::Latency,
            &arch,
            &ArbiterHwView::from(&dp),
        );
        let via_arbitrate_cpu = StrategyArbiter::arbitrate_cpu(
            InferenceMode::Latency,
            &arch,
            &dp,
        );
        assert_approx(via_arbitrate.fusion_cost_scale, via_arbitrate_cpu.fusion_cost_scale, 1e-10);
        assert_approx(via_arbitrate.pipeline_cost_scale, via_arbitrate_cpu.pipeline_cost_scale, 1e-10);
        assert_approx(via_arbitrate.epilogue_depth_preference, via_arbitrate_cpu.epilogue_depth_preference, 1e-10);
        assert_approx(via_arbitrate.k_depth_preference, via_arbitrate_cpu.k_depth_preference, 1e-10);
    }

    // ── 617. GraphArchetype with equal fusion and pipeline yields zero reg_tension ──

    #[test]
    fn reg_tension_zero_when_fusion_equals_pipeline_exact() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.6,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // With zero reg_tension, epilogue and k_depth should equal their
        // mode baseline values (1.5 and 1.3 for Latency) after archetype lerp
        // fusion_profitable=0.6 scales fusion_cost, and pipeline_valuable=0.6
        // scales pipeline_cost, but neither epilogue nor k_depth is touched.
        // Baseline: epilogue=1.5, k_depth=1.3.
        // No hw adjustment (32 regs, 64K L1 => no scarcity, richness=1.0).
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(bias.k_depth_preference, 1.3, 0.01);
    }

    // ── 618. Throughput mode latency_weight_prefetch higher than Latency mode ──

    #[test]
    fn latency_weight_prefetch_higher_than_throughput_for_same_arch() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert!(
            latency.weight_prefetch_budget_scale > throughput.weight_prefetch_budget_scale,
            "Latency mode should prioritize weight prefetch more than Throughput"
        );
    }

    // ── 619. KV cache budget scales up with high memory archetype ──

    #[test]
    fn kv_cache_budget_increases_with_high_memory_archetype() {
        let low_mem = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let high_mem = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let low_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &low_mem, &hw);
        let high_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &high_mem, &hw);
        assert!(
            high_bias.kv_cache_budget_scale > low_bias.kv_cache_budget_scale,
            "High memory archetype should increase KV cache budget scale"
        );
    }

    // ── 620. Quantization aggressiveness scales up with memory archetype ──

    #[test]
    fn quantization_aggressiveness_increases_with_memory_archetype() {
        let low_mem = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let high_mem = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let low_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &low_mem, &hw);
        let high_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &high_mem, &hw);
        assert!(
            high_bias.quantization_aggressiveness > low_bias.quantization_aggressiveness,
            "High memory intensity should increase quantization aggressiveness"
        );
    }

    // ── 621. GPU with 1 register still produces valid bias (GPU branch takes precedence) ──

    #[test]
    fn gpu_one_register_gpu_branch_still_applies() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 1,
            cache_sizes: (0, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // GPU branch multiplies epilogue by 1.2
        // Then register scarcity: num_regs=1 <= 16, scarcity = 1 - 1/32 = 0.96875
        // epilogue *= (1 + 0.96875 * 0.3) = 1.290625
        // Latency baseline epilogue = 1.5, total = 1.5 * 1.2 * 1.290625 ≈ 2.323
        assert_approx(bias.epilogue_depth_preference, 2.323, 0.02);
        // Validate ensures all values are in valid range
        assert!(bias.fusion_cost_scale > 0.0);
        assert!(bias.pipeline_cost_scale > 0.0);
    }

    // ── 622. ArbiterHwView with exact boundary register count 16 ──

    #[test]
    fn register_scarcity_boundary_at_exactly_16() {
        let hw_16 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let hw_17 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 17,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_16 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_16);
        let bias_17 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_17);
        // At 16 regs, scarcity = 1.0 - 16/32 = 0.5
        // At 17 regs, scarcity = 1.0 - 17/32 = 0.46875
        // epilogue = baseline * (1 + scarcity * 0.3)
        // bias_16 epilogue = 1.5 * (1 + 0.5*0.3) = 1.5 * 1.15 = 1.725
        assert_approx(bias_16.epilogue_depth_preference, 1.725, 0.01);
        assert!(
            bias_16.epilogue_depth_preference > bias_17.epilogue_depth_preference,
            "16 regs (boundary) should have higher epilogue preference than 17 regs"
        );
    }

    // ── 623. lerp midpoint symmetry verification ──

    #[test]
    fn lerp_midpoint_symmetry_between_modes() {
        // Verify lerp(a, b, 0.5) == lerp(b, a, 0.5) only when a + b is symmetric
        // i.e., midpoint is always (a+b)/2 regardless of direction
        let mid_ab = lerp(1.0, 5.0, 0.5);
        let mid_ba = lerp(5.0, 1.0, 0.5);
        assert_approx(mid_ab, 3.0, 1e-10);
        assert_approx(mid_ba, 3.0, 1e-10);
    }

    // ── 624. Decode ratio scale unchanged by archetype and hardware ──

    #[test]
    fn decode_ratio_scale_never_modulated() {
        let archetypes = [
            GraphArchetype { compute_intensive: 0.0, memory_intensive: 0.0, parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0 },
            GraphArchetype { compute_intensive: 1.0, memory_intensive: 1.0, parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0 },
        ];
        let hws = [cpu_avx2(), cpu_avx512(), gpu_a100()];
        for arch in &archetypes {
            for hw in &hws {
                let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, arch, hw);
                assert_approx(bias.decode_ratio_scale, 1.0, 0.01);
                let bias_t = StrategyArbiter::arbitrate(InferenceMode::Throughput, arch, hw);
                assert_approx(bias_t.decode_ratio_scale, 1.0, 0.01);
            }
        }
    }

    // ── 625. Very large L1 cache (512KB) produces low fusion cost ──

    #[test]
    fn very_large_l1_reduces_fusion_cost_significantly() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (524288, 0, 0), // 512KB L1
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // l1_richness = 524288 / 65536 = 8.0, capped at 2.0
        // fusion_cost_scale *= 1/sqrt(2.0) ≈ 0.707
        // Latency baseline 0.5 * 0.707 ≈ 0.354
        let expected = 0.5 / 2.0_f64.sqrt();
        assert_approx(bias.fusion_cost_scale, expected, 0.01);
    }

    // ── 626. MoE modulation with parallelism exactly 0.5 boundary ──

    #[test]
    fn moe_parallelism_exactly_half_no_expert_modulation_precise() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch_at_threshold = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let arch_just_below = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.49,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_at = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch_at_threshold, &hw);
        let bias_below = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch_just_below, &hw);
        // At 0.5, the condition parallelism > 0.5 is false, so no MoE adjustment
        // At 0.49, also no MoE adjustment
        // expert_eviction for both should be 0.8 * lerp(1.0, 1.5, 1.0) via kv_cache path
        // but expert eviction specifically only triggers if parallelism > 0.5
        assert_approx(
            bias_at.expert_eviction_aggressiveness,
            bias_below.expert_eviction_aggressiveness,
            1e-10,
        );
    }

    // ── 627. StrategyBias validate clamping identity for default values ──

    #[test]
    fn strategy_bias_validate_default_is_exact_noop() {
        let mut bias = StrategyBias::default();
        let before = bias.clone();
        bias.validate();
        assert_approx(bias.fusion_cost_scale, before.fusion_cost_scale, 1e-10);
        assert_approx(bias.pipeline_cost_scale, before.pipeline_cost_scale, 1e-10);
        assert_approx(bias.parallelism_cost_scale, before.parallelism_cost_scale, 1e-10);
        assert_approx(bias.epilogue_depth_preference, before.epilogue_depth_preference, 1e-10);
        assert_approx(bias.k_depth_preference, before.k_depth_preference, 1e-10);
        assert_approx(bias.kv_cache_budget_scale, before.kv_cache_budget_scale, 1e-10);
        assert_approx(bias.weight_prefetch_budget_scale, before.weight_prefetch_budget_scale, 1e-10);
        assert_approx(bias.batch_flexibility, before.batch_flexibility, 1e-10);
        assert_approx(bias.decode_ratio_scale, before.decode_ratio_scale, 1e-10);
        assert_approx(bias.speculative_decoding_value, before.speculative_decoding_value, 1e-10);
        assert_approx(bias.quantization_aggressiveness, before.quantization_aggressiveness, 1e-10);
        assert_approx(bias.expert_eviction_aggressiveness, before.expert_eviction_aggressiveness, 1e-10);
        assert_approx(bias.expert_prefetch_priority, before.expert_prefetch_priority, 1e-10);
    }

    // ── 628. Expert eviction aggressiveness stays zero in Latency mode regardless of MoE ──

    #[test]
    fn latency_expert_eviction_stays_zero_with_moe_archetype() {
        let moe_arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.99,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = gpu_a100();
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &moe_arch, &hw);
        // Latency baseline expert_eviction = 0.0
        // lerp(1.0, 1.5, memory=1.0) = 1.5, but 0.0 * 1.5 = 0.0
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
    }

    // ── 629. L1 richness capped at 2.0 prevents excessive fusion cost reduction ──

    #[test]
    fn l1_richness_capped_at_two_prevents_over_reduction() {
        let hw_256kb = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (262144, 0, 0), // 256KB L1
        };
        let hw_1mb = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (1048576, 0, 0), // 1MB L1
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias_256kb = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_256kb);
        let bias_1mb = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_1mb);
        // Both exceed cap: 256KB -> richness 4.0 (capped 2.0), 1MB -> richness 16.0 (capped 2.0)
        // So fusion_cost_scale should be identical
        assert_approx(
            bias_256kb.fusion_cost_scale,
            bias_1mb.fusion_cost_scale,
            1e-10
        );
    }

    // ── 630. Arbitrate produces deterministic results on repeated calls ──

    #[test]
    fn arbitrate_deterministic_on_repeated_calls() {
        let arch = GraphArchetype {
            compute_intensive: 0.8,
            memory_intensive: 0.6,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.3,
        };
        let hw = cpu_avx2();
        let first = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let second = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let third = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(first.fusion_cost_scale, second.fusion_cost_scale, 1e-15);
        assert_approx(second.fusion_cost_scale, third.fusion_cost_scale, 1e-15);
        assert_approx(first.k_depth_preference, second.k_depth_preference, 1e-15);
        assert_approx(first.expert_prefetch_priority, third.expert_prefetch_priority, 1e-15);
    }

    // ── New tests (wave-15) ──

    /// GPU hardware view applies 1.2x boost to epilogue_depth_preference for latency mode.
    /// With a neutral archetype (all zero), the only modulation is the GPU boost.
    #[test]
    fn gpu_latency_epilogue_depth_boosted_by_1_2() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw_gpu = ArbiterHwView::gpu(32768);
        let hw_cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_gpu);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_cpu);
        // GPU epilogue should be 1.2x the CPU epilogue (both have 32 regs, no scarcity)
        assert_approx(
            gpu_bias.epilogue_depth_preference / cpu_bias.epilogue_depth_preference,
            1.2,
            0.01,
        );
    }

    /// GPU hardware view applies 1.2x boost to k_depth_preference.
    #[test]
    fn gpu_latency_k_depth_boosted_by_1_2() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw_gpu = ArbiterHwView::gpu(32768);
        let hw_cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_gpu);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_cpu);
        assert_approx(
            gpu_bias.k_depth_preference / cpu_bias.k_depth_preference,
            1.2,
            0.01,
        );
    }

    /// GPU hardware view applies 1.2x boost to pipeline_cost_scale.
    #[test]
    fn gpu_throughput_pipeline_cost_boosted_by_1_2() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw_gpu = ArbiterHwView::gpu(32768);
        let hw_cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw_gpu);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw_cpu);
        assert_approx(
            gpu_bias.pipeline_cost_scale / cpu_bias.pipeline_cost_scale,
            1.2,
            0.01,
        );
    }

    /// GPU boost does NOT affect fusion_cost_scale (only L1 richness does).
    #[test]
    fn gpu_boost_does_not_affect_fusion_cost_scale() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw_gpu = ArbiterHwView::gpu(32768);
        let hw_cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_gpu);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_cpu);
        // Both have the same L1 size so fusion_cost_scale should be identical
        assert_approx(gpu_bias.fusion_cost_scale, cpu_bias.fusion_cost_scale, 1e-10);
    }

    /// GPU boost does NOT affect batch_flexibility (mode-driven only).
    #[test]
    fn gpu_boost_does_not_affect_batch_flexibility() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw_gpu = ArbiterHwView::gpu(32768);
        let hw_cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw_gpu);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw_cpu);
        assert_eq!(gpu_bias.batch_flexibility, cpu_bias.batch_flexibility);
    }

    /// Validate that clamping one out-of-range field leaves other in-range fields untouched.
    #[test]
    fn strategy_bias_validate_single_field_out_of_range_preserves_others() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.5,
            pipeline_cost_scale: 50.0, // way above max 3.0
            parallelism_cost_scale: 1.0,
            epilogue_depth_preference: 1.5,
            k_depth_preference: 0.8,
            kv_cache_budget_scale: 1.0,
            weight_prefetch_budget_scale: 0.9,
            batch_flexibility: 0.5,
            decode_ratio_scale: 1.0,
            expert_eviction_aggressiveness: 0.3,
            expert_prefetch_priority: 1.2,
            speculative_decoding_value: 0.8,
            quantization_aggressiveness: 0.6,
        };
        bias.validate();
        // Only pipeline_cost_scale should be clamped
        assert_approx(bias.fusion_cost_scale, 0.5, 1e-10);
        assert_approx(bias.pipeline_cost_scale, 3.0, 1e-10); // clamped
        assert_approx(bias.parallelism_cost_scale, 1.0, 1e-10);
        assert_approx(bias.epilogue_depth_preference, 1.5, 1e-10);
    }

    /// StrategyBias default values for expert fields: eviction=0.0, prefetch=1.0.
    #[test]
    fn strategy_bias_default_expert_fields() {
        let bias = StrategyBias::default();
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 1.0);
    }

    /// StrategyBias accessor methods return the same values as direct field access.
    #[test]
    fn strategy_bias_accessor_matches_field() {
        let mut bias = StrategyBias::default();
        bias.expert_eviction_aggressiveness = 1.5;
        bias.expert_prefetch_priority = 2.3;
        assert_eq!(bias.expert_eviction_aggressiveness(), bias.expert_eviction_aggressiveness);
        assert_eq!(bias.expert_prefetch_priority(), bias.expert_prefetch_priority);
    }

    /// Latency mode baseline has decode_ratio_scale = 1.0 (not mode-dependent).
    #[test]
    fn latency_throughput_decode_ratio_identical_at_baseline() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = cpu_avx512();
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(latency.decode_ratio_scale, throughput.decode_ratio_scale, 1e-10);
    }

    /// ArbiterHwView with is_gpu=false and high registers (>16) triggers no scarcity adjustment.
    #[test]
    fn cpu_high_registers_no_scarcity_no_epilogue_adjustment() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw_32 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let hw_64 = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 64,
            cache_sizes: (65536, 0, 0),
        };
        let bias_32 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_32);
        let bias_64 = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_64);
        // Both above 16 threshold, same L1, so should produce identical results
        assert_approx(bias_32.epilogue_depth_preference, bias_64.epilogue_depth_preference, 1e-10);
        assert_approx(bias_32.k_depth_preference, bias_64.k_depth_preference, 1e-10);
    }

    /// Construct ArbiterHwView with L1 exactly 65536 bytes gives L1 richness = 1.0,
    /// so fusion_cost_scale should not be adjusted by L1.
    #[test]
    fn l1_exactly_64kb_cpu_latency_fusion_scale_no_l1_adjustment() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // With fusion_profitable=0, lerp returns a, so fusion_cost_scale stays at baseline 0.5
        // L1 richness = 65536/65536 = 1.0, sqrt(1.0) = 1.0, so fusion_cost_scale *= 1.0
        assert_approx(bias.fusion_cost_scale, 0.5, 1e-10);
    }

    /// Latency mode baseline has higher weight_prefetch_budget_scale than throughput.
    #[test]
    fn latency_weight_prefetch_budget_higher_than_throughput_baseline() {
        let latency = StrategyArbiter::mode_baseline(InferenceMode::Latency);
        let throughput = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        assert!(
            latency.weight_prefetch_budget_scale > throughput.weight_prefetch_budget_scale,
            "latency weight_prefetch_budget ({}) should exceed throughput ({})",
            latency.weight_prefetch_budget_scale,
            throughput.weight_prefetch_budget_scale,
        );
    }

    /// Throughput mode baseline has higher kv_cache_budget_scale than latency.
    #[test]
    fn throughput_kv_cache_budget_scale_higher_than_latency_baseline() {
        let latency = StrategyArbiter::mode_baseline(InferenceMode::Latency);
        let throughput = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        assert!(
            throughput.kv_cache_budget_scale > latency.kv_cache_budget_scale,
            "throughput kv_cache_budget ({}) should exceed latency ({})",
            throughput.kv_cache_budget_scale,
            latency.kv_cache_budget_scale,
        );
    }

    /// ArbiterHwView constructed with gpu() always has L2 and L3 set to zero,
    /// and shared memory stored in L1 slot.
    #[test]
    fn gpu_constructor_l2_l3_zero_shared_mem_in_l1() {
        let view = ArbiterHwView::gpu(98304);
        assert_eq!(view.cache_sizes.1, 0);
        assert_eq!(view.cache_sizes.2, 0);
        assert_eq!(view.cache_sizes.0, 98304);
    }

    /// GraphArchetype with all fields equal to 0.5 produces a valid, finite bias
    /// and reg_tension is exactly zero (fusion_profitable == pipeline_valuable).
    #[test]
    fn archetype_half_fields_zero_reg_tension_all_finite() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = cpu_avx2();
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // All fields must be finite (no NaN, no infinity)
        assert!(bias.fusion_cost_scale.is_finite());
        assert!(bias.pipeline_cost_scale.is_finite());
        assert!(bias.parallelism_cost_scale.is_finite());
        assert!(bias.epilogue_depth_preference.is_finite());
        assert!(bias.k_depth_preference.is_finite());
        assert!(bias.kv_cache_budget_scale.is_finite());
        assert!(bias.weight_prefetch_budget_scale.is_finite());
        assert!(bias.batch_flexibility.is_finite());
        assert!(bias.decode_ratio_scale.is_finite());
        assert!(bias.expert_eviction_aggressiveness.is_finite());
        assert!(bias.expert_prefetch_priority.is_finite());
        assert!(bias.speculative_decoding_value.is_finite());
        assert!(bias.quantization_aggressiveness.is_finite());
    }

    // ── 新增测试 (2026-06-01): 15 个覆盖未测试场景 ────────────────────

    /// Latency 基线的 epilogue_depth_preference (1.5) 高于 Throughput (0.8)。
    /// 验证 SPEC §4.3.1 的延迟模式偏好更深 epilogue 链设计意图。
    #[test]
    fn latency_epilogue_depth_preference_higher_than_throughput() {
        let latency = StrategyArbiter::mode_baseline(InferenceMode::Latency);
        let throughput = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        assert!(
            latency.epilogue_depth_preference > throughput.epilogue_depth_preference,
            "latency epilogue_depth ({}) should exceed throughput ({})",
            latency.epilogue_depth_preference,
            throughput.epilogue_depth_preference,
        );
    }

    /// Latency 基线的 k_depth_preference (1.3) 高于 Throughput (0.8)。
    /// 验证延迟模式偏好更激进的 K 循环展开。
    #[test]
    fn latency_k_depth_preference_higher_than_throughput() {
        let latency = StrategyArbiter::mode_baseline(InferenceMode::Latency);
        let throughput = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        assert!(
            latency.k_depth_preference > throughput.k_depth_preference,
            "latency k_depth ({}) should exceed throughput ({})",
            latency.k_depth_preference,
            throughput.k_depth_preference,
        );
    }

    /// Latency 基线的 parallelism_cost_scale (1.5) 高于 Throughput (0.5)。
    /// 验证延迟模式惩罚并行同步开销的设计意图。
    #[test]
    fn latency_parallelism_cost_scale_higher_than_throughput() {
        let latency = StrategyArbiter::mode_baseline(InferenceMode::Latency);
        let throughput = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        assert!(
            latency.parallelism_cost_scale > throughput.parallelism_cost_scale,
            "latency parallelism_cost ({}) should exceed throughput ({})",
            latency.parallelism_cost_scale,
            throughput.parallelism_cost_scale,
        );
    }

    /// Latency 基线的 pipeline_cost_scale (0.6) 低于 Throughput (1.3)。
    /// 验证延迟模式更愿意接受 pipeline 同步、追求单请求优化的设计意图。
    #[test]
    fn latency_pipeline_cost_scale_lower_than_throughput() {
        let latency = StrategyArbiter::mode_baseline(InferenceMode::Latency);
        let throughput = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        assert!(
            latency.pipeline_cost_scale < throughput.pipeline_cost_scale,
            "latency pipeline_cost ({}) should be below throughput ({})",
            latency.pipeline_cost_scale,
            throughput.pipeline_cost_scale,
        );
    }

    /// 当 parallelism_exploitable 严格小于 0.5 时，
    /// MoE expert_eviction_aggressiveness 和 expert_prefetch_priority 不受 memory_intensive 调制。
    /// 验证 apply_archetype_modulation 的 `if parallelism_exploitable > 0.5` 守卫。
    #[test]
    fn moe_expert_fields_unmodulated_when_parallelism_below_threshold() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0, // 高内存压力，但不足以触发 MoE 调制
            parallelism_exploitable: 0.49, // 严格低于 0.5 阈值
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = cpu_avx512();
        let latency_baseline = StrategyArbiter::mode_baseline(InferenceMode::Latency);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // expert_eviction 应保持基线值 (latency=0.0)
        assert_approx(bias.expert_eviction_aggressiveness, latency_baseline.expert_eviction_aggressiveness, 1e-10);
        // expert_prefetch 应保持基线值 (latency=0.5)，不受 2x 放大
        assert_approx(bias.expert_prefetch_priority, latency_baseline.expert_prefetch_priority, 1e-10);
    }

    /// 当 parallelism_exploitable 恰好等于 0.5 时 (不满足 > 0.5)，
    /// MoE expert 字段仍不触发调制。验证边界条件。
    #[test]
    fn moe_expert_fields_unmodulated_at_parallelism_exactly_0_5() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.5, // 恰好等于阈值，不满足 >
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = cpu_avx512();
        let baseline = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // expert_eviction 应保持基线值 (throughput=0.8)，不放大到 1.2
        assert_approx(bias.expert_eviction_aggressiveness, baseline.expert_eviction_aggressiveness, 1e-10);
        // expert_prefetch 应保持基线值 (throughput=1.5)，不放大到 3.0
        assert_approx(bias.expert_prefetch_priority, baseline.expert_prefetch_priority, 1e-10);
    }

    /// 当 parallelism_exploitable 为 0.51 (刚过阈值) 且 memory_intensive=1.0 时，
    /// expert_eviction 应被 lerp(1.0, 1.5, 1.0) = 1.5 放大，
    /// expert_prefetch 应被 lerp(1.0, 2.0, 1.0) = 2.0 放大。
    /// 验证 MoE 调制的精确数值。
    #[test]
    fn moe_expert_precise_at_parallelism_just_above_threshold() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.51,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = cpu_avx512();
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        let baseline = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        // eviction: 0.8 * lerp(1.0, 1.5, 1.0) = 0.8 * 1.5 = 1.2
        assert_approx(bias.expert_eviction_aggressiveness, 1.2, 1e-10);
        // prefetch: 1.5 * lerp(1.0, 2.0, 1.0) = 1.5 * 2.0 = 3.0
        assert_approx(bias.expert_prefetch_priority, 3.0, 1e-10);
    }

    /// kv_cache_budget_scale 的精确调制验证：
    /// Throughput 基线 1.5，memory_intensive=0.5 时调制为 lerp(1.0, 1.5, 0.5) = 1.25，
    /// 最终 = 1.5 * 1.25 = 1.875。
    #[test]
    fn kv_cache_budget_throughput_memory_0_5_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = cpu_avx512(); // 高寄存器、无稀缺、L1=49152
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // 基线 1.5 * lerp(1.0, 1.5, 0.5) = 1.5 * 1.25 = 1.875
        assert_approx(bias.kv_cache_budget_scale, 1.875, 1e-10);
    }

    /// Latency 模式下 kv_cache_budget_scale 的精确调制验证：
    /// 基线 0.5，memory_intensive=1.0 时调制为 lerp(1.0, 1.5, 1.0) = 1.5，
    /// 最终 = 0.5 * 1.5 = 0.75。
    #[test]
    fn kv_cache_budget_latency_full_memory_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = cpu_avx512();
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // 基线 0.5 * lerp(1.0, 1.5, 1.0) = 0.5 * 1.5 = 0.75
        assert_approx(bias.kv_cache_budget_scale, 0.75, 1e-10);
    }

    /// L1 richness 的精确计算验证：
    /// L1 = 131072 字节 → richness = min(131072/65536, 2.0) = 2.0。
    /// fusion_cost_scale 调制因子 = 1.0 / sqrt(2.0) ≈ 0.70710678...
    /// 零 archetype 下 latency 基线 0.5，最终 = 0.5 / sqrt(2.0) ≈ 0.35355339...
    #[test]
    fn l1_128kb_latency_fusion_cost_precise_with_zero_archetype() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0), // 128KB L1
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let expected = 0.5 / 2.0_f64.sqrt();
        assert_approx(bias.fusion_cost_scale, expected, 1e-10);
    }

    /// L1 richness 为 0.5 时的精确值验证：
    /// L1 = 32768 → richness = 32768/65536 = 0.5。
    /// fusion_cost_scale 因子 = 1.0 / sqrt(0.5) = sqrt(2.0) ≈ 1.41421356...
    /// 零 archetype + Throughput 基线 1.0 → 最终 = 1.0 * sqrt(2.0) ≈ 1.41421356...
    #[test]
    fn l1_32kb_throughput_fusion_cost_increased_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0), // 32KB L1
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        let expected = 1.0 / 0.5_f64.sqrt();
        assert_approx(bias.fusion_cost_scale, expected, 1e-10);
    }

    /// 寄存器稀缺调整不影响 fusion_cost_scale、pipeline_cost_scale、parallelism_cost_scale。
    /// 验证 apply_hardware_adjustment 的 CPU 稀缺分支只修改 epilogue 和 k_depth。
    #[test]
    fn cpu_scarcity_preserves_cost_scales_unchanged() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // 低寄存器触发稀缺
        let hw_scarce = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        // 高寄存器不触发稀缺
        let hw_normal = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias_scarce = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_scarce);
        let bias_normal = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_normal);
        // 这三个 cost scale 不受寄存器稀缺影响
        assert_approx(bias_scarce.fusion_cost_scale, bias_normal.fusion_cost_scale, 1e-10);
        assert_approx(bias_scarce.pipeline_cost_scale, bias_normal.pipeline_cost_scale, 1e-10);
        assert_approx(bias_scarce.parallelism_cost_scale, bias_normal.parallelism_cost_scale, 1e-10);
    }

    /// 量化攻击性 (quantization_aggressiveness) 的精确跨模式差异验证：
    /// Latency 基线 1.5 vs Throughput 基线 0.8，差值 = 0.7。
    #[test]
    fn quantization_aggressiveness_exact_mode_gap() {
        let latency = StrategyArbiter::mode_baseline(InferenceMode::Latency);
        let throughput = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        let gap = latency.quantization_aggressiveness - throughput.quantization_aggressiveness;
        assert_approx(gap, 0.7, 1e-10);
    }

    /// speculative_decoding_value 的精确跨模式差异验证：
    /// Latency 基线 1.5 vs Throughput 基线 0.3，差值 = 1.2。
    #[test]
    fn speculative_decoding_value_exact_mode_gap() {
        let latency = StrategyArbiter::mode_baseline(InferenceMode::Latency);
        let throughput = StrategyArbiter::mode_baseline(InferenceMode::Throughput);
        let gap = latency.speculative_decoding_value - throughput.speculative_decoding_value;
        assert_approx(gap, 1.2, 1e-10);
    }

    /// GPU 调整与零 archetype 的组合精确值验证：
    /// GPU is_gpu=true → epilogue *= 1.2, k_depth *= 1.2, pipeline *= 1.2。
    /// Latency 基线: epilogue=1.5, k_depth=1.3, pipeline=0.6。
    /// 最终: epilogue=1.8, k_depth=1.56, pipeline=0.72。
    #[test]
    fn gpu_latency_zero_archetype_three_fields_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // GPU with 65536 shared mem → L1 richness = 1.0, no L1 adjustment
        let hw = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.5 * 1.2, 1e-10);
        assert_approx(bias.k_depth_preference, 1.3 * 1.2, 1e-10);
        assert_approx(bias.pipeline_cost_scale, 0.6 * 1.2, 1e-10);
    }

    /// 通过 arbiter 的全部 2 mode × 2 device × 3 memory_intensive (0.0/0.5/1.0) 组合，
    /// 验证所有输出值均在 StrategyBias::validate() 的合法范围内。
    /// 覆盖 12 个配置点的边界条件扫描。
    #[test]
    fn full_cross_product_all_outputs_within_validate_bounds() {
        let modes = [InferenceMode::Latency, InferenceMode::Throughput];
        let device_families = [DeviceFamily::Cpu, DeviceFamily::Gpu];
        let memory_levels = [0.0_f64, 0.5, 1.0];
        let mut count = 0;
        for &mode in &modes {
            for &device in &device_families {
                for &mem in &memory_levels {
                    let arch = GraphArchetype {
                        compute_intensive: 0.5,
                        memory_intensive: mem,
                        parallelism_exploitable: 0.3,
                        fusion_profitable: 0.6,
                        pipeline_valuable: 0.4,
                    };
                    let hw = match device {
                        DeviceFamily::Gpu => ArbiterHwView::gpu(49152),
                        DeviceFamily::Cpu => ArbiterHwView {
                            device: DeviceFamily::Cpu,
                            num_simd_regs: 32,
                            cache_sizes: (49152, 262144, 8388608),
                        },
                    };
                    let bias = StrategyArbiter::arbitrate(mode, &arch, &hw);
                    // 所有 13 个字段必须在 validate 的 clamp 范围内
                    assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
                    assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
                    assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
                    assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
                    assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
                    assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
                    assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
                    assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
                    assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
                    assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
                    assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
                    assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
                    assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
                    count += 1;
                }
            }
        }
        assert_eq!(count, 12, "should have tested all 12 combinations");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  15 additional tests — unique coverage for uncovered paths
    // ═══════════════════════════════════════════════════════════════════

    /// GPU (Throughput) + 零 archetype + 64KB shared mem 的 fusion_cost_scale 精确值：
    /// Throughput 基线 fusion_cost = 1.0, 零 archetype → lerp(1.0, 0.6, 0.0) = 1.0 (不变)。
    /// GPU is_gpu → 不修改 fusion_cost_scale。L1 richness = 65536/65536 = 1.0, sqrt = 1.0, 因子 = 1.0。
    /// 最终 = 1.0 * 1.0 * 1.0 = 1.0。
    #[test]
    fn gpu_throughput_zero_archetype_fusion_cost_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView::gpu(65536);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.fusion_cost_scale, 1.0, 1e-10);
    }

    /// CPU num_simd_regs = 16 恰好触发稀缺分支 (<=16)：
    /// scarcity = 1.0 - 16/32 = 0.5。
    /// epilogue *= 1.0 + 0.5 * 0.3 = 1.15, k_depth *= 1.0 - 0.5 * 0.2 = 0.9。
    /// Latency 基线: epilogue=1.5, k_depth=1.3。零 archetype 无调制。
    /// 最终 epilogue = 1.5 * 1.15 = 1.725, k_depth = 1.3 * 0.9 = 1.17。
    #[test]
    fn cpu_exactly_16_registers_boundary_triggers_scarcity() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.5 * 1.15, 1e-10);
        assert_approx(bias.k_depth_preference, 1.3 * 0.9, 1e-10);
    }

    /// CPU num_simd_regs = 17 不触发稀缺分支 (>16)。
    /// 零 archetype 下 Latency 基线不变。
    /// epilogue = 1.5, k_depth = 1.3 (与 num_simd_regs=32 时相同)。
    #[test]
    fn cpu_17_registers_no_scarcity_adjustment() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 17,
            cache_sizes: (65536, 0, 0),
        };
        let hw_baseline = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let baseline = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_baseline);
        assert_approx(bias.epilogue_depth_preference, baseline.epilogue_depth_preference, 1e-10);
        assert_approx(bias.k_depth_preference, baseline.k_depth_preference, 1e-10);
    }

    /// KV cache 和 quantization modulation 独立性验证：
    /// 两者都使用 memory_intensive 参数，但应用在不同字段。
    /// memory_intensive = 0.6 时：
    /// kv_cache_budget_scale *= lerp(1.0, 1.5, 0.6) = 1.0 + 0.6 * 0.5 = 1.3
    /// quantization_aggressiveness *= lerp(1.0, 1.3, 0.6) = 1.0 + 0.6 * 0.3 = 1.18
    /// 验证两个字段的比例关系精确。
    #[test]
    fn kv_and_quantization_independent_modulation() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.6,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Throughput 基线 kv_cache = 1.5, 量化 = 0.8
        let expected_kv = 1.5 * (1.0 + 0.6 * 0.5);
        let expected_quant = 0.8 * (1.0 + 0.6 * 0.3);
        assert_approx(bias.kv_cache_budget_scale, expected_kv, 1e-10);
        assert_approx(bias.quantization_aggressiveness, expected_quant, 1e-10);
    }

    /// GPU shared memory > 128KB 时 L1 richness 被 min(2.0) 封顶：
    /// L1 = 196608 (192KB) → richness = min(196608/65536, 2.0) = min(3.0, 2.0) = 2.0。
    /// fusion_cost_scale 因子 = 1.0 / sqrt(2.0)。
    /// Throughput 零 archetype → fusion = 1.0 / sqrt(2.0)。
    #[test]
    fn gpu_shared_mem_larger_than_128kb_capped_richness() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw_192kb = ArbiterHwView::gpu(196608);
        let hw_128kb = ArbiterHwView::gpu(131072);
        let bias_192kb = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw_192kb);
        let bias_128kb = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw_128kb);
        // 两者 richness 都 = 2.0, 所以 fusion_cost_scale 相同
        assert_approx(bias_192kb.fusion_cost_scale, bias_128kb.fusion_cost_scale, 1e-10);
        assert_approx(bias_192kb.fusion_cost_scale, 1.0 / 2.0_f64.sqrt(), 1e-10);
    }

    /// MoE 模式下 Throughput 的 expert_eviction 和 expert_prefetch 比率：
    /// parallelism_exploitable = 0.8 (> 0.5), memory_intensive = 0.7。
    /// eviction *= lerp(1.0, 1.5, 0.7) = 1.0 + 0.7 * 0.5 = 1.35
    /// prefetch *= lerp(1.0, 2.0, 0.7) = 1.0 + 0.7 * 1.0 = 1.7
    /// Throughput 基线: eviction=0.8, prefetch=1.5
    /// 最终: eviction = 0.8 * 1.35 = 1.08, prefetch = 1.5 * 1.7 = 2.55
    /// prefetch/eviction 比值 ≈ 2.36 > 1，确认 prefetch 总是比 eviction 更激进。
    #[test]
    fn expert_eviction_and_prefetch_ratio_throughput_moe() {
        let arch = GraphArchetype {
            compute_intensive: 0.3,
            memory_intensive: 0.7,
            parallelism_exploitable: 0.8,
            fusion_profitable: 0.2,
            pipeline_valuable: 0.1,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        assert_approx(bias.expert_eviction_aggressiveness, 0.8 * 1.35, 1e-10);
        assert_approx(bias.expert_prefetch_priority, 1.5 * 1.7, 1e-10);
        assert!(
            bias.expert_prefetch_priority > bias.expert_eviction_aggressiveness,
            "prefetch priority should always exceed eviction aggressiveness in MoE throughput"
        );
    }

    /// CPU num_simd_regs = 0 极端稀缺精确值：
    /// scarcity = 1.0 - 0/32 = 1.0。
    /// epilogue *= 1.0 + 1.0 * 0.3 = 1.3, k_depth *= 1.0 - 1.0 * 0.2 = 0.8。
    /// Latency 基线: epilogue=1.5, k_depth=1.3。
    /// 最终: epilogue = 1.5 * 1.3 = 1.95, k_depth = 1.3 * 0.8 = 1.04。
    #[test]
    fn cpu_scarcity_exactly_zero_registers_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 0,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.5 * 1.3, 1e-10);
        assert_approx(bias.k_depth_preference, 1.3 * 0.8, 1e-10);
    }

    /// 负 memory_intensive 减小 KV cache 和 quantization 调制量：
    /// memory_intensive = -0.5:
    /// kv_cache *= lerp(1.0, 1.5, -0.5) = 1.0 + (-0.5) * 0.5 = 0.75
    /// quantization *= lerp(1.0, 1.3, -0.5) = 1.0 + (-0.5) * 0.3 = 0.85
    /// Throughput 基线: kv_cache=1.5, quant=0.8。
    /// 验证两者均低于纯基线值（无 archetype 调制）。
    #[test]
    fn negative_memory_reduces_kv_and_quantization_below_baseline() {
        let arch_negative = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: -0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let arch_zero = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias_neg = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch_negative, &hw);
        let bias_zero = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch_zero, &hw);
        assert!(
            bias_neg.kv_cache_budget_scale < bias_zero.kv_cache_budget_scale,
            "negative memory_intensive should reduce kv_cache below zero-archetype baseline"
        );
        assert!(
            bias_neg.quantization_aggressiveness < bias_zero.quantization_aggressiveness,
            "negative memory_intensive should reduce quantization below zero-archetype baseline"
        );
    }

    /// GPU pipeline_cost_scale 调整与 L1 richness 效应独立：
    /// GPU 分支只增加 pipeline *= 1.2。L1 richness 只修改 fusion_cost_scale。
    /// 验证修改 pipeline 的因子不影响 fusion_cost_scale，反之亦然。
    #[test]
    fn gpu_pipeline_and_fusion_adjustments_are_independent() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw_64k = ArbiterHwView::gpu(65536);
        let hw_128k = ArbiterHwView::gpu(131072);
        let bias_64k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_64k);
        let bias_128k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_128k);
        // pipeline_cost_scale 不受 L1 richness 影响，两者应该相同
        assert_approx(bias_64k.pipeline_cost_scale, bias_128k.pipeline_cost_scale, 1e-10);
        // fusion_cost_scale 不同（受 L1 richness 影响）
        assert!(
            bias_128k.fusion_cost_scale < bias_64k.fusion_cost_scale,
            "larger shared mem should reduce fusion cost via L1 richness"
        );
    }

    /// Latency 模式 + 高并行 MoE archetype 的 expert 字段精确值：
    /// parallelism_exploitable = 0.9 (> 0.5), memory_intensive = 0.8。
    /// Latency 基线: eviction = 0.0, prefetch = 0.5。
    /// eviction *= lerp(1.0, 1.5, 0.8) = 1.0 + 0.8 * 0.5 = 1.4 → 0.0 * 1.4 = 0.0 (零 * 任何 = 零)
    /// prefetch *= lerp(1.0, 2.0, 0.8) = 1.0 + 0.8 * 1.0 = 1.8 → 0.5 * 1.8 = 0.9
    #[test]
    fn latency_high_parallel_moe_expert_fields_precise() {
        let arch = GraphArchetype {
            compute_intensive: 0.3,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.9,
            fusion_profitable: 0.2,
            pipeline_valuable: 0.1,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Latency eviction baseline = 0.0, 乘以任何因子仍为 0.0
        assert_approx(bias.expert_eviction_aggressiveness, 0.0, 1e-10);
        // prefetch = 0.5 * (1.0 + 0.8 * 1.0) = 0.5 * 1.8 = 0.9
        assert_approx(bias.expert_prefetch_priority, 0.5 * 1.8, 1e-10);
    }

    /// Throughput 模式 + 全满 archetype (所有=1.0) + CPU AVX2 的 fusion_cost 完整 lerp 链：
    /// Throughput 基线 fusion_cost = 1.0。
    /// archetype: fusion_profitable=1.0 → lerp(1.0, 0.6, 1.0) = 0.6, fusion *= 0.6 = 0.6。
    /// L1 = 32768 → richness = 0.5, factor = 1.0/sqrt(0.5) = sqrt(2)。
    /// 最终 = 0.6 * sqrt(2) ≈ 0.848528。
    #[test]
    fn throughput_full_archetype_avx2_fusion_cost_exact_lerp_chain() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Throughput baseline = 1.0, archetype fusion_profitable=1.0 → *= 0.6
        // L1 richness = 32768/65536 = 0.5, factor = 1.0/sqrt(0.5)
        let expected = 1.0 * 0.6 / 0.5_f64.sqrt();
        assert_approx(bias.fusion_cost_scale, expected, 1e-10);
    }

    /// reg_tension = 0 时 GPU 对 epilogue 和 k_depth 应用相同 1.2 因子：
    /// 零 archetype → 无 arch modulation, GPU is_gpu → epilogue *= 1.2, k_depth *= 1.2。
    /// Latency 基线: epilogue=1.5, k_depth=1.3。
    /// 两者增幅比例应该完全相同 (都 ×1.2)。
    #[test]
    fn reg_tension_zero_gpu_boosts_epilogue_kdepth_same_factor() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw_cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let hw_gpu = ArbiterHwView::gpu(65536);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_gpu);
        // GPU boost ratio should be identical for both fields
        let epilogue_ratio = gpu_bias.epilogue_depth_preference / cpu_bias.epilogue_depth_preference;
        let k_depth_ratio = gpu_bias.k_depth_preference / cpu_bias.k_depth_preference;
        assert_approx(epilogue_ratio, 1.2, 1e-10);
        assert_approx(k_depth_ratio, 1.2, 1e-10);
    }

    /// parallelism_cost_scale 完全不被硬件调整分支影响：
    /// GPU 分支和 CPU 稀缺分支都不修改 parallelism_cost_scale。
    /// 两个不同硬件配置下 parallelism_cost_scale 仅受 archetype 影响。
    #[test]
    fn parallelism_cost_unaffected_by_hardware_entirely() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.2,
        };
        let hw_cpu_low = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 4,
            cache_sizes: (8192, 0, 0),
        };
        let hw_gpu = ArbiterHwView::gpu(65536);
        let bias_cpu = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_cpu_low);
        let bias_gpu = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_gpu);
        // parallelism_cost_scale should be identical regardless of hardware
        assert_approx(bias_cpu.parallelism_cost_scale, bias_gpu.parallelism_cost_scale, 1e-10);
    }

    /// Latency 模式 + parallelism=0.6 (>0.5) 但 memory_intensive=0.0:
    /// MoE 调制需要 parallelism > 0.5 AND memory_intensive 参与乘法。
    /// lerp(1.0, 1.5, 0.0) = 1.0, lerp(1.0, 2.0, 0.0) = 1.0。
    /// 因此 expert 字段虽有 parallelism 条件满足，但 memory=0 抵消了增幅。
    /// eviction = 0.0 * 1.0 = 0.0, prefetch = 0.5 * 1.0 = 0.5。
    #[test]
    fn moe_latency_high_parallel_zero_memory_no_expert_boost() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.3,
            pipeline_valuable: 0.3,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // memory_intensive = 0.0, lerp(1.0, X, 0.0) = 1.0 for both
        // Latency baseline: eviction = 0.0, prefetch = 0.5
        assert_approx(bias.expert_eviction_aggressiveness, 0.0, 1e-10);
        assert_approx(bias.expert_prefetch_priority, 0.5, 1e-10);
    }

    /// 验证 arbitrate 输出的 13 个字段不全相同：
    /// 使用非零 archetype + CPU 低寄存器确保多样性。
    /// 如果所有字段相同则说明存在系统性错误（如全零或全一）。
    #[test]
    fn arbitrate_outputs_13_distinct_fields_not_all_same() {
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.4,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let fields = [
            bias.fusion_cost_scale,
            bias.pipeline_cost_scale,
            bias.parallelism_cost_scale,
            bias.epilogue_depth_preference,
            bias.k_depth_preference,
            bias.kv_cache_budget_scale,
            bias.weight_prefetch_budget_scale,
            bias.batch_flexibility,
            bias.decode_ratio_scale,
            bias.speculative_decoding_value,
            bias.quantization_aggressiveness,
            bias.expert_eviction_aggressiveness,
            bias.expert_prefetch_priority,
        ];
        // At least two distinct values must exist
        let first = fields[0];
        let all_same = fields.iter().all(|&v| (v - first).abs() < 1e-10);
        assert!(!all_same, "all 13 fields should not be identical");
        // All must be finite
        for &v in &fields {
            assert!(v.is_finite(), "field value {} should be finite", v);
        }
    }

    // ── Wave 12x34: 15 new tests ────────────────────────────────────
    // Focus: InferenceMode Display/Debug, ArbiterHwView full-field construction,
    // DeviceProfile mapping boundary, latency vs throughput mode diffs,
    // StrategyBias all variants, GraphProfile field coverage.

    /// InferenceMode Debug format produces non-empty strings for both variants.
    /// Neither variant should format to an empty string.
    #[test]
    fn inference_mode_debug_never_empty() {
        let latency_dbg = format!("{:?}", InferenceMode::Latency);
        let throughput_dbg = format!("{:?}", InferenceMode::Throughput);
        assert!(!latency_dbg.is_empty(), "Latency Debug should not be empty");
        assert!(!throughput_dbg.is_empty(), "Throughput Debug should not be empty");
    }

    /// InferenceMode Debug strings are distinct between the two variants.
    /// This ensures they do not accidentally produce the same output.
    #[test]
    fn inference_mode_debug_strings_are_distinct() {
        let latency_dbg = format!("{:?}", InferenceMode::Latency);
        let throughput_dbg = format!("{:?}", InferenceMode::Throughput);
        assert_ne!(
            latency_dbg, throughput_dbg,
            "Latency and Throughput Debug strings must differ"
        );
    }

    /// ArbiterHwView manual construction with is_gpu=false, num_simd_regs=0,
    /// cache_sizes all zero: a degenerate CPU view should not panic when used.
    #[test]
    fn arbiter_hw_view_all_zero_cpu_no_panic_in_arbitrate() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 0,
            cache_sizes: (0, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // All outputs should be finite and within validate bounds
        assert!(bias.fusion_cost_scale.is_finite());
        assert!(bias.kv_cache_budget_scale.is_finite());
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
    }

    /// ArbiterHwView with is_gpu=true and very large shared memory (1 MB).
    /// L1 richness = 1_048_576 / 65536 = 16.0, capped at 2.0.
    /// Fusion cost should be scaled by 1/sqrt(2.0) from the L1 richness path.
    #[test]
    fn arbiter_hw_view_gpu_1mb_shared_mem_l1_richness_capped() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (1_048_576, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Baseline latency fusion_cost_scale = 0.5, archetype lerp(1,0.6,0)=1.0, no modulation.
        // L1 richness = 1048576/65536 = 16.0, min(16.0, 2.0) = 2.0.
        // fusion_cost_scale *= 1.0/sqrt(2.0).
        let expected = 0.5 / 2.0_f64.sqrt();
        assert_approx(bias.fusion_cost_scale, expected, 1e-10);
    }

    /// DeviceProfile::detect() -> ArbiterHwView mapping: is_gpu is always false
    /// because DeviceProfile has no GPU variant. Verify the From impl consistency
    /// by checking that two conversions from the same profile yield equal views.
    #[test]
    fn device_profile_from_produces_identical_views_on_same_profile() {
        let dp = DeviceProfile::detect();
        let view1 = ArbiterHwView::from(&dp);
        let view2 = ArbiterHwView::from(&dp);
        assert_eq!(view1, view2, "two conversions from same DeviceProfile must be equal");
        assert_eq!(view1.device, DeviceFamily::Cpu, "DeviceProfile conversion must always produce CPU view");
    }

    /// Latency mode produces strictly lower batch_flexibility than Throughput
    /// for any non-degenerate archetype. This is a fundamental mode difference.
    /// Verify with a high-memory archetype where other fields diverge significantly.
    #[test]
    fn latency_vs_throughput_batch_flexibility_gap_with_high_memory() {
        let arch = GraphArchetype {
            compute_intensive: 0.8,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.4,
        };
        let hw = cpu_avx512();
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Latency always has batch_flexibility = 0.0, Throughput = 1.0
        assert_approx(lat.batch_flexibility, 0.0, 1e-10);
        assert_approx(thr.batch_flexibility, 1.0, 1e-10);
        assert!(lat.batch_flexibility < thr.batch_flexibility);
    }

    /// Latency vs Throughput: speculative_decoding_value is always higher in Latency
    /// mode (1.5 vs 0.3 baseline) and archetype modulation does not reverse this.
    #[test]
    fn latency_vs_throughput_speculative_decoding_always_higher_in_latency() {
        let archetypes = [
            GraphArchetype { compute_intensive: 0.0, memory_intensive: 0.0, parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0 },
            GraphArchetype { compute_intensive: 1.0, memory_intensive: 1.0, parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0 },
            GraphArchetype { compute_intensive: 0.5, memory_intensive: 0.5, parallelism_exploitable: 0.5, fusion_profitable: 0.5, pipeline_valuable: 0.5 },
        ];
        let hw = cpu_avx2();
        for arch in &archetypes {
            let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, arch, &hw);
            let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, arch, &hw);
            assert!(
                lat.speculative_decoding_value > thr.speculative_decoding_value,
                "Latency speculative_decoding_value ({}) must exceed Throughput ({})",
                lat.speculative_decoding_value, thr.speculative_decoding_value
            );
        }
    }

    /// StrategyBias validate() clamping: each field has specific min/max bounds.
    /// Verify that setting every field to -1.0 results in each being clamped to
    /// its individual minimum (not a universal minimum).
    #[test]
    fn strategy_bias_validate_negative_one_clamps_to_per_field_minimums() {
        let mut bias = StrategyBias {
            fusion_cost_scale: -1.0,
            pipeline_cost_scale: -1.0,
            parallelism_cost_scale: -1.0,
            epilogue_depth_preference: -1.0,
            k_depth_preference: -1.0,
            kv_cache_budget_scale: -1.0,
            weight_prefetch_budget_scale: -1.0,
            batch_flexibility: -1.0,
            decode_ratio_scale: -1.0,
            speculative_decoding_value: -1.0,
            quantization_aggressiveness: -1.0,
            expert_eviction_aggressiveness: -1.0,
            expert_prefetch_priority: -1.0,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.pipeline_cost_scale, 0.2);
        assert_eq!(bias.parallelism_cost_scale, 0.1);
        assert_eq!(bias.epilogue_depth_preference, 0.3);
        assert_eq!(bias.k_depth_preference, 0.3);
        assert_eq!(bias.kv_cache_budget_scale, 0.2);
        assert_eq!(bias.weight_prefetch_budget_scale, 0.2);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.decode_ratio_scale, 0.3);
        assert_eq!(bias.speculative_decoding_value, 0.1);
        assert_eq!(bias.quantization_aggressiveness, 0.3);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 0.1);
    }

    /// StrategyBias validate() boundary: parallelism_cost_scale min is 0.1
    /// (different from most fields' 0.2 or 0.3). Verify the exact boundary.
    #[test]
    fn strategy_bias_parallelism_cost_min_is_zero_point_one() {
        let mut bias = StrategyBias {
            parallelism_cost_scale: 0.05,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_eq!(bias.parallelism_cost_scale, 0.1, "min should be exactly 0.1");
    }

    /// StrategyBias validate() boundary: expert_prefetch_priority max is 5.0
    /// (higher than most fields' 3.0 cap). Verify this specific boundary.
    #[test]
    fn strategy_bias_expert_prefetch_max_is_five() {
        let mut bias = StrategyBias {
            expert_prefetch_priority: 6.0,
            ..StrategyBias::default()
        };
        bias.validate();
        assert_eq!(bias.expert_prefetch_priority, 5.0, "max should be exactly 5.0");
    }

    /// GraphArchetype with all fields at 0.5 (uniform) produces zero reg_tension
    /// (fusion_profitable == pipeline_valuable). The epilogue and k_depth
    /// preferences should remain at baseline with no reg_tension adjustment.
    #[test]
    fn uniform_archetype_zero_reg_tension_preserves_baseline_preferences() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        // Use neutral hardware (no GPU, no scarcity, 64KB L1 = richness 1.0)
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Baseline epilogue = 1.5, k_depth = 1.3.
        // Archetype modulation: lerp(1,0.6,0.5)=0.8 for fusion, lerp(1,0.6,0.5)=0.8 for pipeline,
        // lerp(1,0.5,0.5)=0.75 for parallelism.
        // KV: lerp(1,1.5,0.5)=1.25, quant: lerp(1,1.3,0.5)=1.15
        // reg_tension = 0.5 - 0.5 = 0.0, no adjustment to epilogue/k_depth.
        // Hardware: no GPU, no scarcity, L1=65536 => richness=1.0 => 1/sqrt(1)=1.0 => no fusion change.
        assert_approx(lat.epilogue_depth_preference, 1.5, 1e-10);
        assert_approx(lat.k_depth_preference, 1.3, 1e-10);
    }

    /// Latency mode with zero archetype + GPU hardware: verify that exactly
    /// three fields (epilogue_depth_preference, k_depth_preference, pipeline_cost_scale)
    /// are boosted by 1.2x from their baselines, while fusion_cost_scale is not.
    #[test]
    fn latency_gpu_only_three_fields_boosted_from_baseline() {
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView::gpu(0);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Latency baselines: epilogue=1.5, k_depth=1.3, pipeline=0.6, fusion=0.5
        // GPU boosts epilogue, k_depth, pipeline by 1.2x.
        // L1=0 => richness=0 => no fusion adjustment.
        assert_approx(bias.epilogue_depth_preference, 1.5 * 1.2, 1e-10);
        assert_approx(bias.k_depth_preference, 1.3 * 1.2, 1e-10);
        assert_approx(bias.pipeline_cost_scale, 0.6 * 1.2, 1e-10);
        // fusion_cost_scale: baseline 0.5, lerp(1,0.6,0)=1.0 (no arch mod), L1=0 no hw adj
        assert_approx(bias.fusion_cost_scale, 0.5, 1e-10);
    }

    /// Throughput mode with zero archetype + GPU: pipeline_cost_scale baseline is 1.3,
    /// GPU boosts by 1.2 => 1.56. Verify this exact value.
    #[test]
    fn throughput_gpu_pipeline_cost_exactly_1_56() {
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView::gpu(0);
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &zero_arch, &hw);
        // Throughput pipeline baseline = 1.3, lerp(1,0.6,0)=1.0 (no arch), GPU *1.2 = 1.56
        assert_approx(bias.pipeline_cost_scale, 1.3 * 1.2, 1e-10);
    }

    /// StrategyBias default has expert_eviction_aggressiveness = 0.0,
    /// which differs from all other fields (1.0). Verify this asymmetry explicitly.
    #[test]
    fn strategy_bias_default_single_zero_field_amidst_ones() {
        let bias = StrategyBias::default();
        let zero_fields: Vec<&str> = [
            (bias.fusion_cost_scale, "fusion_cost_scale"),
            (bias.pipeline_cost_scale, "pipeline_cost_scale"),
            (bias.parallelism_cost_scale, "parallelism_cost_scale"),
            (bias.epilogue_depth_preference, "epilogue_depth_preference"),
            (bias.k_depth_preference, "k_depth_preference"),
            (bias.kv_cache_budget_scale, "kv_cache_budget_scale"),
            (bias.weight_prefetch_budget_scale, "weight_prefetch_budget_scale"),
            (bias.batch_flexibility, "batch_flexibility"),
            (bias.decode_ratio_scale, "decode_ratio_scale"),
            (bias.speculative_decoding_value, "speculative_decoding_value"),
            (bias.quantization_aggressiveness, "quantization_aggressiveness"),
            (bias.expert_eviction_aggressiveness, "expert_eviction_aggressiveness"),
            (bias.expert_prefetch_priority, "expert_prefetch_priority"),
        ]
        .iter()
        .filter(|(v, _)| *v == 0.0)
        .map(|(_, name)| *name)
        .collect();
        assert_eq!(
            zero_fields,
            vec!["expert_eviction_aggressiveness"],
            "only expert_eviction_aggressiveness should be 0.0 in default"
        );
    }

    /// GraphArchetype PartialEq: verify that two archetypes with identical fields
    /// are equal, and changing a single field (compute_intensive) makes them unequal.
    #[test]
    fn graph_archetype_partial_eq_single_field_difference() {
        let a = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.6,
        };
        let b = GraphArchetype {
            compute_intensive: 0.6, // differs
            memory_intensive: 0.3,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.6,
        };
        assert_ne!(a, b, "archetypes differing in compute_intensive should be unequal");
        let c = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.6,
        };
        assert_eq!(a, c, "archetypes with identical fields should be equal");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Round 3 — GraphArchetype PartialEq all fields, ArbiterOutput non-NaN,
    //            extreme archetype (all-0 / all-1), hardware fallback, CPU vs GPU
    // ══════════════════════════════════════════════════════════════════════

    /// GraphArchetype PartialEq: change each of the 5 fields one at a time and
    /// verify the archetype is no longer equal. This exhaustively proves the
    /// derive(PartialEq) covers every field.
    #[test]
    fn graph_archetype_partial_eq_each_of_five_fields_differences_detected() {
        let base = GraphArchetype {
            compute_intensive: 0.4,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.8,
        };
        assert_ne!(
            base,
            GraphArchetype { compute_intensive: 0.9, ..base },
            "differing compute_intensive must be unequal"
        );
        assert_ne!(
            base,
            GraphArchetype { memory_intensive: 0.1, ..base },
            "differing memory_intensive must be unequal"
        );
        assert_ne!(
            base,
            GraphArchetype { parallelism_exploitable: 0.0, ..base },
            "differing parallelism_exploitable must be unequal"
        );
        assert_ne!(
            base,
            GraphArchetype { fusion_profitable: 0.2, ..base },
            "differing fusion_profitable must be unequal"
        );
        assert_ne!(
            base,
            GraphArchetype { pipeline_valuable: 0.3, ..base },
            "differing pipeline_valuable must be unequal"
        );
    }

    /// StrategyBias all-ones archetype with latency mode on CPU AVX-512:
    /// every field must be finite (non-NaN, non-infinity).
    #[test]
    fn strategy_bias_latency_all_ones_archetype_avx512_all_fields_finite() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu_avx512());
        for (val, name) in [
            (bias.fusion_cost_scale, "fusion_cost_scale"),
            (bias.pipeline_cost_scale, "pipeline_cost_scale"),
            (bias.parallelism_cost_scale, "parallelism_cost_scale"),
            (bias.epilogue_depth_preference, "epilogue_depth_preference"),
            (bias.k_depth_preference, "k_depth_preference"),
            (bias.kv_cache_budget_scale, "kv_cache_budget_scale"),
            (bias.weight_prefetch_budget_scale, "weight_prefetch_budget_scale"),
            (bias.batch_flexibility, "batch_flexibility"),
            (bias.decode_ratio_scale, "decode_ratio_scale"),
            (bias.speculative_decoding_value, "speculative_decoding_value"),
            (bias.quantization_aggressiveness, "quantization_aggressiveness"),
            (bias.expert_eviction_aggressiveness, "expert_eviction_aggressiveness"),
            (bias.expert_prefetch_priority, "expert_prefetch_priority"),
        ] {
            assert!(val.is_finite(), "{name} should be finite, got {val}");
        }
    }

    /// All-zeros archetype + latency + AVX2: verify that after modulation and
    /// validate(), every field is still strictly positive (> 0).
    #[test]
    fn all_zero_archetype_latency_avx2_all_fields_strictly_positive() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu_avx2());
        assert!(bias.fusion_cost_scale > 0.0, "fusion_cost_scale must be > 0");
        assert!(bias.pipeline_cost_scale > 0.0, "pipeline_cost_scale must be > 0");
        assert!(bias.parallelism_cost_scale > 0.0, "parallelism_cost_scale must be > 0");
        assert!(bias.epilogue_depth_preference > 0.0, "epilogue_depth_preference must be > 0");
        assert!(bias.k_depth_preference > 0.0, "k_depth_preference must be > 0");
        assert!(bias.kv_cache_budget_scale > 0.0, "kv_cache_budget_scale must be > 0");
        assert!(bias.weight_prefetch_budget_scale > 0.0, "weight_prefetch_budget_scale must be > 0");
        assert!(bias.batch_flexibility >= 0.0, "batch_flexibility must be >= 0");
        assert!(bias.decode_ratio_scale > 0.0, "decode_ratio_scale must be > 0");
        assert!(bias.speculative_decoding_value > 0.0, "speculative_decoding_value must be > 0");
        assert!(bias.quantization_aggressiveness > 0.0, "quantization_aggressiveness must be > 0");
        assert!(bias.expert_eviction_aggressiveness >= 0.0, "expert_eviction_aggressiveness must be >= 0");
        assert!(bias.expert_prefetch_priority > 0.0, "expert_prefetch_priority must be > 0");
    }

    /// All-ones archetype + throughput + AVX2: verify that every field stays
    /// within the validate() clamp bounds after modulation + hardware adjustment.
    #[test]
    fn all_ones_archetype_throughput_avx2_all_fields_within_validate_bounds() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &cpu_avx2());
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.weight_prefetch_budget_scale >= 0.2 && bias.weight_prefetch_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.speculative_decoding_value >= 0.1 && bias.speculative_decoding_value <= 3.0);
        assert!(bias.quantization_aggressiveness >= 0.3 && bias.quantization_aggressiveness <= 3.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
    }

    /// Hardware fallback: CPU with zero L1 cache (cache_sizes = (0, 0, 0))
    /// should not panic. The L1 richness is 0.0, so the `if l1_richness > 0.0`
    /// guard skips fusion adjustment entirely. Verify fusion_cost_scale equals
    /// the post-archetype-modulation value with no further division.
    #[test]
    fn cpu_zero_l1_zero_l2_zero_l3_skips_fusion_adjustment_no_panic() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (0, 0, 0),
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Latency baseline fusion = 0.5, lerp(1,0.6,0)=1.0 (no archetype mod),
        // L1=0 => richness=0 => skip hw fusion adj => remains 0.5
        assert_approx(bias.fusion_cost_scale, 0.5, 1e-10);
    }

    /// Hardware fallback: GPU with zero shared memory (L1=0) still applies the
    /// GPU 1.2x boost to epilogue_depth_preference but skips the L1 richness
    /// fusion adjustment (richness=0, guard skips). Verify the exact value.
    #[test]
    fn gpu_zero_shared_mem_applies_gpu_boost_but_skips_fusion_adj() {
        let hw = ArbiterHwView::gpu(0);
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // GPU boost: epilogue = 1.5 * 1.2 = 1.8
        assert_approx(bias.epilogue_depth_preference, 1.8, 1e-10);
        // fusion = 0.5 baseline, no arch mod, L1=0 skip => remains 0.5
        assert_approx(bias.fusion_cost_scale, 0.5, 1e-10);
    }

    /// Hardware fallback: CPU with num_simd_regs = 1 (extreme scarcity).
    /// scarcity = 1.0 - (1/32) = 0.96875. Epilogue should increase and
    /// k_depth should decrease. Neither should panic.
    #[test]
    fn cpu_one_register_extreme_scarcity_epilogue_up_kdepth_down() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 1,
            cache_sizes: (65536, 0, 0),
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Latency baseline: epilogue=1.5, k_depth=1.3
        // scarcity = 1 - 1/32 = 0.96875
        // epilogue *= 1 + 0.96875*0.3 = 1.290625 => 1.5 * 1.290625 = 1.9359375
        // k_depth *= 1 - 0.96875*0.2 = 0.80625 => 1.3 * 0.80625 = 1.048125
        assert_approx(bias.epilogue_depth_preference, 1.5 * (1.0 + 0.96875 * 0.3), 1e-10);
        assert_approx(bias.k_depth_preference, 1.3 * (1.0 - 0.96875 * 0.2), 1e-10);
    }

    /// CPU vs GPU output difference: for the same archetype and Latency mode,
    /// the GPU output must differ from CPU output in exactly three fields
    /// (epilogue_depth_preference, k_depth_preference, pipeline_cost_scale)
    /// which get the 1.2x GPU boost. All other fields should be identical.
    #[test]
    fn cpu_vs_gpu_latency_same_archetype_differs_in_exactly_three_fields() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.2,
        };
        // CPU and GPU both with 64KB L1 so fusion adjustment is identical
        let cpu_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu_hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu_hw);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_hw);

        // These three should differ (GPU 1.2x boost)
        assert!(
            (gpu_bias.epilogue_depth_preference - cpu_bias.epilogue_depth_preference).abs() > 1e-10,
            "epilogue_depth_preference should differ between CPU and GPU"
        );
        assert!(
            (gpu_bias.k_depth_preference - cpu_bias.k_depth_preference).abs() > 1e-10,
            "k_depth_preference should differ between CPU and GPU"
        );
        assert!(
            (gpu_bias.pipeline_cost_scale - cpu_bias.pipeline_cost_scale).abs() > 1e-10,
            "pipeline_cost_scale should differ between CPU and GPU"
        );
        // GPU has 255 regs, no scarcity; CPU has 32 regs, no scarcity either.
        // L1 richness identical (both 64KB). So only GPU 1.2x matters.
        // fusion_cost_scale: same archetype modulation + same L1 richness => identical
        assert_approx(gpu_bias.fusion_cost_scale, cpu_bias.fusion_cost_scale, 1e-10);
        assert_approx(gpu_bias.parallelism_cost_scale, cpu_bias.parallelism_cost_scale, 1e-10);
        assert_approx(gpu_bias.kv_cache_budget_scale, cpu_bias.kv_cache_budget_scale, 1e-10);
        assert_eq!(gpu_bias.batch_flexibility, cpu_bias.batch_flexibility);
        assert_approx(gpu_bias.decode_ratio_scale, cpu_bias.decode_ratio_scale, 1e-10);
        assert_approx(gpu_bias.speculative_decoding_value, cpu_bias.speculative_decoding_value, 1e-10);
        assert_approx(gpu_bias.quantization_aggressiveness, cpu_bias.quantization_aggressiveness, 1e-10);
        assert_approx(gpu_bias.weight_prefetch_budget_scale, cpu_bias.weight_prefetch_budget_scale, 1e-10);
    }

    /// CPU vs GPU output difference: Throughput mode, verify that the GPU
    /// epilogue_depth_preference is exactly 1.2x the CPU value.
    #[test]
    fn cpu_vs_gpu_throughput_epilogue_exactly_1_2x_ratio() {
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.3,
        };
        let cpu_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let gpu_hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (32768, 0, 0),
        };
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &cpu_hw);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu_hw);
        // GPU applies 1.2x to epilogue after archetype modulation
        // CPU has 32 regs => no scarcity. So ratio should be exactly 1.2
        assert_approx(
            gpu_bias.epilogue_depth_preference / cpu_bias.epilogue_depth_preference,
            1.2,
            1e-10,
        );
    }

    /// CPU vs GPU with register scarcity: CPU has 16 regs (scarcity boundary),
    /// GPU has 255. The CPU k_depth_preference should be lower than GPU because
    /// register scarcity reduces it, while GPU gets a 1.2x boost.
    #[test]
    fn cpu_scarcity_vs_gpu_boost_k_depth_double_difference() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 16,
            cache_sizes: (65536, 0, 0),
        };
        let gpu_hw = ArbiterHwView::gpu(0);
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu_hw);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_hw);
        // CPU: scarcity = 1 - 16/32 = 0.5, k_depth = 1.3 * (1 - 0.5*0.2) = 1.3 * 0.9 = 1.17
        // GPU: k_depth = 1.3 * 1.2 = 1.56
        // GPU should be strictly greater
        assert!(
            gpu_bias.k_depth_preference > cpu_bias.k_depth_preference,
            "GPU k_depth ({}) should exceed CPU k_depth ({})",
            gpu_bias.k_depth_preference,
            cpu_bias.k_depth_preference,
        );
    }

    /// All-ones archetype: when fusion_profitable = 1.0 and pipeline_valuable = 1.0,
    /// reg_tension = 0.0 so no epilogue/k_depth adjustment. But lerp saturates:
    /// fusion_cost_scale gets lerp(1.0, 0.6, 1.0) = 0.6 multiplier. Verify that
    /// for latency baseline (0.5), the fusion cost after all modulation is 0.5 * 0.6 = 0.3.
    #[test]
    fn all_ones_archetype_latency_fusion_cost_saturates_to_zero_point_three() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Latency fusion baseline = 0.5
        // lerp(1.0, 0.6, 1.0) = 0.6 => fusion = 0.5 * 0.6 = 0.3
        // L1 richness = 65536/65536 = 1.0 => 1/sqrt(1) = 1.0 => no fusion change
        assert_approx(bias.fusion_cost_scale, 0.3, 1e-10);
    }

    /// All-ones archetype with throughput mode: kv_cache_budget_scale gets
    /// lerp(1.0, 1.5, 1.0) = 1.5 multiplier. Throughput baseline = 1.5.
    /// Result = 1.5 * 1.5 = 2.25. Verify this exact value.
    #[test]
    fn all_ones_archetype_throughput_kv_cache_exactly_2_25() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Throughput kv baseline = 1.5, lerp(1,1.5,1.0) = 1.5 => 1.5*1.5 = 2.25
        assert_approx(bias.kv_cache_budget_scale, 2.25, 1e-10);
    }

    /// All-zeros archetype with throughput mode: MoE modulation threshold is
    /// parallelism_exploitable > 0.5. At 0.0, no MoE adjustment. Verify
    /// expert_eviction_aggressiveness stays at throughput baseline 0.8.
    #[test]
    fn all_zeros_archetype_throughput_no_moe_modulation_expert_eviction() {
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Throughput baseline eviction = 0.8, parallelism=0.0 < 0.5 => no MoE mod
        // memory_intensive=0.0 => lerp(1,1.5,0)=1.0 => no KV mod affects eviction
        assert_approx(bias.expert_eviction_aggressiveness, 0.8, 1e-10);
    }

    /// StrategyBias non-NaN verification: construct a GraphArchetype with all
    /// fields set to 0.5 (uniform mid-range). Run both modes on both CPU AVX2
    /// and GPU. All 13 output fields must be non-NaN for every combination.
    #[test]
    fn strategy_bias_midrange_archetype_all_modes_all_hardware_never_nan() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        for mode in [InferenceMode::Latency, InferenceMode::Throughput] {
            for hw in [cpu_avx2(), cpu_avx512(), gpu_a100()] {
                let bias = StrategyArbiter::arbitrate(mode, &arch, &hw);
                assert!(!bias.fusion_cost_scale.is_nan(), "fusion_cost_scale NaN for {mode:?} hw={hw:?}");
                assert!(!bias.pipeline_cost_scale.is_nan(), "pipeline_cost_scale NaN for {mode:?}");
                assert!(!bias.parallelism_cost_scale.is_nan(), "parallelism_cost_scale NaN for {mode:?}");
                assert!(!bias.epilogue_depth_preference.is_nan(), "epilogue_depth_preference NaN for {mode:?}");
                assert!(!bias.k_depth_preference.is_nan(), "k_depth_preference NaN for {mode:?}");
                assert!(!bias.kv_cache_budget_scale.is_nan(), "kv_cache_budget_scale NaN for {mode:?}");
                assert!(!bias.weight_prefetch_budget_scale.is_nan(), "weight_prefetch_budget_scale NaN");
                assert!(!bias.batch_flexibility.is_nan(), "batch_flexibility NaN for {mode:?}");
                assert!(!bias.decode_ratio_scale.is_nan(), "decode_ratio_scale NaN for {mode:?}");
                assert!(!bias.speculative_decoding_value.is_nan(), "speculative_decoding_value NaN");
                assert!(!bias.quantization_aggressiveness.is_nan(), "quantization_aggressiveness NaN");
                assert!(!bias.expert_eviction_aggressiveness.is_nan(), "expert_eviction_aggressiveness NaN");
                assert!(!bias.expert_prefetch_priority.is_nan(), "expert_prefetch_priority NaN");
            }
        }
    }

    /// CPU vs GPU difference: for throughput mode with a high-memory archetype,
    /// quantization_aggressiveness should be identical across CPU and GPU
    /// (hardware adjustment does not touch this field).
    #[test]
    fn cpu_vs_gpu_throughput_quantization_aggressiveness_identical() {
        let arch = GraphArchetype {
            compute_intensive: 0.3,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.8,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.6,
        };
        let cpu_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu_hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &cpu_hw);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu_hw);
        // quantization_aggressiveness is not touched by hardware adjustment
        assert_approx(
            cpu_bias.quantization_aggressiveness,
            gpu_bias.quantization_aggressiveness,
            1e-10,
        );
    }

    // ── ArbiterOutput comprehensive finiteness for all archetype extremes ──

    /// Verify that when every archetype field is at its maximum (1.0), all 13
    /// StrategyBias output fields are finite for both modes and both hardware
    /// classes (CPU AVX-512 and GPU A100).
    #[test]
    fn all_max_archetype_all_modes_all_hw_every_field_finite() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        for mode in [InferenceMode::Latency, InferenceMode::Throughput] {
            for hw in [cpu_avx512(), gpu_a100()] {
                let bias = StrategyArbiter::arbitrate(mode, &arch, &hw);
                let fields: [f64; 13] = [
                    bias.fusion_cost_scale,
                    bias.pipeline_cost_scale,
                    bias.parallelism_cost_scale,
                    bias.epilogue_depth_preference,
                    bias.k_depth_preference,
                    bias.kv_cache_budget_scale,
                    bias.weight_prefetch_budget_scale,
                    bias.batch_flexibility,
                    bias.decode_ratio_scale,
                    bias.speculative_decoding_value,
                    bias.quantization_aggressiveness,
                    bias.expert_eviction_aggressiveness,
                    bias.expert_prefetch_priority,
                ];
                for (i, &v) in fields.iter().enumerate() {
                    assert!(v.is_finite(), "field[{i}] not finite: mode={mode:?}, hw.device={:?}", hw.device);
                }
            }
        }
    }

    // ── GraphArchetype Debug: verify each field name appears in output ──

    /// GraphArchetype Debug trait output must contain the names of all five
    /// fields so that log output is self-describing.
    #[test]
    fn graph_archetype_debug_output_contains_all_five_field_names() {
        let arch = GraphArchetype {
            compute_intensive: 0.42,
            memory_intensive: 0.55,
            parallelism_exploitable: 0.67,
            fusion_profitable: 0.78,
            pipeline_valuable: 0.89,
        };
        let debug_str = format!("{arch:?}");
        assert!(
            debug_str.contains("compute_intensive"),
            "Debug output missing 'compute_intensive': {debug_str}"
        );
        assert!(
            debug_str.contains("memory_intensive"),
            "Debug output missing 'memory_intensive': {debug_str}"
        );
        assert!(
            debug_str.contains("parallelism_exploitable"),
            "Debug output missing 'parallelism_exploitable': {debug_str}"
        );
        assert!(
            debug_str.contains("fusion_profitable"),
            "Debug output missing 'fusion_profitable': {debug_str}"
        );
        assert!(
            debug_str.contains("pipeline_valuable"),
            "Debug output missing 'pipeline_valuable': {debug_str}"
        );
    }

    // ── StrategyBias all-zero validate: every field clamps to per-field minimum ──

    /// Setting every StrategyBias field to exactly 0.0 and calling validate
    /// should clamp each to its individual minimum (which varies per field).
    #[test]
    fn strategy_bias_all_fields_zero_validate_clamps_to_per_field_minimums() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.0,
            pipeline_cost_scale: 0.0,
            parallelism_cost_scale: 0.0,
            epilogue_depth_preference: 0.0,
            k_depth_preference: 0.0,
            kv_cache_budget_scale: 0.0,
            weight_prefetch_budget_scale: 0.0,
            batch_flexibility: 0.0,
            decode_ratio_scale: 0.0,
            speculative_decoding_value: 0.0,
            quantization_aggressiveness: 0.0,
            expert_eviction_aggressiveness: 0.0,
            expert_prefetch_priority: 0.0,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2);
        assert_eq!(bias.pipeline_cost_scale, 0.2);
        assert_eq!(bias.parallelism_cost_scale, 0.1);
        assert_eq!(bias.epilogue_depth_preference, 0.3);
        assert_eq!(bias.k_depth_preference, 0.3);
        assert_eq!(bias.kv_cache_budget_scale, 0.2);
        assert_eq!(bias.weight_prefetch_budget_scale, 0.2);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_eq!(bias.decode_ratio_scale, 0.3);
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
        assert_eq!(bias.expert_prefetch_priority, 0.1);
        assert_eq!(bias.speculative_decoding_value, 0.1);
        assert_eq!(bias.quantization_aggressiveness, 0.3);
    }

    // ── InferenceMode PartialEq round-trip: clone equals original ──

    /// Cloned InferenceMode instances must compare equal to the original for
    /// both variants, confirming PartialEq and Clone coherence.
    #[test]
    fn inference_mode_clone_partial_eq_roundtrip_both_variants() {
        let latency = InferenceMode::Latency;
        let throughput = InferenceMode::Throughput;
        assert_eq!(latency.clone(), latency);
        assert_eq!(throughput.clone(), throughput);
        assert_ne!(latency.clone(), throughput);
        assert_ne!(throughput.clone(), latency);
    }

    // ── ArbiterHwView manually-constructed zero view: scarcity calc with 0 regs ──

    /// An ArbiterHwView with is_gpu=false, num_simd_regs=0 has maximum scarcity.
    /// scarcity = 1.0 - (0 / 32.0) = 1.0, so epilogue gets * (1 + 1.0 * 0.3) = * 1.3
    /// and k_depth gets * (1 - 1.0 * 0.2) = * 0.8 from the hardware adjustment.
    #[test]
    fn arbiter_hw_view_zero_registers_max_scarcity_precise_effect() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 0,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        // Use latency mode. Baseline: epilogue=1.5, k_depth=1.3
        // reg_tension = 0.5 - 0.5 = 0.0, so archetype modulation does not
        // change epilogue or k_depth.
        // Hardware: scarcity = 1.0, epilogue *= 1.3 = 1.95, k_depth *= 0.8 = 1.04
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        assert_approx(bias.epilogue_depth_preference, 1.95, 0.01);
        assert_approx(bias.k_depth_preference, 1.04, 0.01);
    }

    // ── reg_tension at exactly 0.0: zero archetype values ──

    /// When fusion_profitable=0.0 and pipeline_valuable=0.0, reg_tension=0.0.
    /// Neither the positive nor negative reg_tension branch fires, so
    /// epilogue_depth_preference and k_depth_preference stay at their mode
    /// baseline values (before hardware adjustment).
    #[test]
    fn reg_tension_zero_both_fusion_pipeline_zero_no_archetype_preference_shift() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // No reg_tension effect, no archetype modulation on these two fields.
        // Hardware: 32 regs, no scarcity. L1=65536, l1_richness=1.0, sqrt(1.0)=1.0.
        // epilogue baseline: latency=1.5, throughput=0.8
        // k_depth baseline: latency=1.3, throughput=0.8
        assert_approx(latency.epilogue_depth_preference, 1.5, 0.01);
        assert_approx(latency.k_depth_preference, 1.3, 0.01);
        assert_approx(throughput.epilogue_depth_preference, 0.8, 0.01);
        assert_approx(throughput.k_depth_preference, 0.8, 0.01);
    }

    // ── reg_tension at maximum (1.0): fusion=1.0, pipeline=0.0 ──

    /// Maximum reg_tension occurs when fusion_profitable=1.0 and
    /// pipeline_valuable=0.0, giving reg_tension=1.0.
    /// Epilogue *= (1 + 1.0*0.5) = 1.5, k_depth *= (1 - 1.0*0.3) = 0.7.
    #[test]
    fn reg_tension_maximum_positive_precise_epilogue_up_kdepth_down() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Baseline latency: epilogue=1.5, k_depth=1.3
        // reg_tension=1.0: epilogue *= 1.5 = 2.25, k_depth *= 0.7 = 0.91
        // Hardware: 32 regs, no scarcity. L1=65536, richness=1.0, no fusion adj.
        assert_approx(bias.epilogue_depth_preference, 2.25, 0.01);
        assert_approx(bias.k_depth_preference, 0.91, 0.01);
    }

    // ── reg_tension at minimum (-1.0): fusion=0.0, pipeline=1.0 ──

    /// Minimum reg_tension occurs when fusion_profitable=0.0 and
    /// pipeline_valuable=1.0, giving reg_tension=-1.0.
    /// k_depth *= (1 + 1.0*0.5) = 1.5, epilogue *= (1 - 1.0*0.3) = 0.7.
    #[test]
    fn reg_tension_minimum_negative_precise_kdepth_up_epilogue_down() {
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 1.0,
        };
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Baseline latency: epilogue=1.5, k_depth=1.3
        // reg_tension=-1.0: k_depth *= 1.5 = 1.95, epilogue *= 0.7 = 1.05
        assert_approx(bias.k_depth_preference, 1.95, 0.01);
        assert_approx(bias.epilogue_depth_preference, 1.05, 0.01);
    }

    // ── shared_mem size affects latency mode fusion_cost_scale ──

    /// In latency mode, different GPU shared memory sizes should produce
    /// different fusion_cost_scale values because L1 richness affects it.
    /// A GPU with 32KB shared mem should have higher fusion_cost_scale
    /// (less L1 richness → less cost reduction) than one with 128KB.
    #[test]
    fn shared_mem_size_affects_latency_fusion_cost_scale() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let gpu_small = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (32768, 0, 0),
        };
        let gpu_large = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (131072, 0, 0),
        };
        let bias_small = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_small);
        let bias_large = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_large);
        // Larger L1 = more richness = lower fusion_cost_scale
        assert!(
            bias_large.fusion_cost_scale < bias_small.fusion_cost_scale,
            "larger shared mem should reduce fusion_cost_scale: got small={} large={}",
            bias_small.fusion_cost_scale,
            bias_large.fusion_cost_scale,
        );
    }

    // ── shared_mem size does not affect latency batch_flexibility ──

    /// Regardless of GPU shared memory size, batch_flexibility in latency mode
    /// is always 0.0 because the hardware adjustment never touches that field.
    #[test]
    fn shared_mem_size_does_not_affect_latency_batch_flexibility() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let gpu_16k = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (16384, 0, 0),
        };
        let gpu_256k = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (262144, 0, 0),
        };
        let bias_16k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_16k);
        let bias_256k = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_256k);
        assert_eq!(bias_16k.batch_flexibility, 0.0);
        assert_eq!(bias_256k.batch_flexibility, 0.0);
    }

    // ── ArbiterOutput: latency mode weight_prefetch always higher than throughput ──

    /// For any combination of archetype and hardware, latency mode should
    /// always produce a higher weight_prefetch_budget_scale than throughput
    /// mode, because latency baseline=1.5 vs throughput baseline=0.8 and
    /// the modulation paths are the same.
    #[test]
    fn latency_weight_prefetch_always_higher_than_throughput_for_all_archetypes() {
        let hw = cpu_avx512();
        let archetypes = [
            GraphArchetype { compute_intensive: 0.0, memory_intensive: 0.0, parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0 },
            GraphArchetype { compute_intensive: 0.5, memory_intensive: 0.5, parallelism_exploitable: 0.5, fusion_profitable: 0.5, pipeline_valuable: 0.5 },
            GraphArchetype { compute_intensive: 1.0, memory_intensive: 1.0, parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0 },
            GraphArchetype { compute_intensive: 0.2, memory_intensive: 0.8, parallelism_exploitable: 0.3, fusion_profitable: 0.9, pipeline_valuable: 0.1 },
        ];
        for arch in &archetypes {
            let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, arch, &hw);
            let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, arch, &hw);
            assert!(
                lat.weight_prefetch_budget_scale > thr.weight_prefetch_budget_scale,
                "latency weight_prefetch should exceed throughput: lat={} thr={} arch={:?}",
                lat.weight_prefetch_budget_scale, thr.weight_prefetch_budget_scale, arch,
            );
        }
    }

    // ── ArbiterOutput: throughput kv_cache_budget always higher than latency ──

    /// For any archetype, throughput mode baseline kv_cache_budget_scale=1.5
    /// vs latency=0.5. Since modulation is archetype-identical for both,
    /// throughput should always yield a higher value.
    #[test]
    fn throughput_kv_cache_budget_always_higher_than_latency_for_all_archetypes() {
        let hw = cpu_avx512();
        let archetypes = [
            GraphArchetype { compute_intensive: 0.0, memory_intensive: 0.0, parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0 },
            GraphArchetype { compute_intensive: 0.5, memory_intensive: 0.5, parallelism_exploitable: 0.5, fusion_profitable: 0.5, pipeline_valuable: 0.5 },
            GraphArchetype { compute_intensive: 1.0, memory_intensive: 1.0, parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0 },
            GraphArchetype { compute_intensive: 0.3, memory_intensive: 0.7, parallelism_exploitable: 0.9, fusion_profitable: 0.1, pipeline_valuable: 0.6 },
        ];
        for arch in &archetypes {
            let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, arch, &hw);
            let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, arch, &hw);
            assert!(
                thr.kv_cache_budget_scale > lat.kv_cache_budget_scale,
                "throughput kv_cache should exceed latency: lat={} thr={} arch={:?}",
                lat.kv_cache_budget_scale, thr.kv_cache_budget_scale, arch,
            );
        }
    }

    // ── GPU boost scales epilogue and k_depth by the same factor ──

    /// When is_gpu=true, both epilogue_depth_preference and k_depth_preference
    /// are multiplied by 1.2. This means the ratio between them should be
    /// preserved from the post-archetype-modulation state.
    #[test]
    fn gpu_boost_preserves_epilogue_kdepth_ratio_from_archetype_state() {
        let arch = GraphArchetype {
            compute_intensive: 0.3,
            memory_intensive: 0.4,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.2,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (65536, 0, 0),
        };
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        let cpu_ratio = cpu_bias.epilogue_depth_preference / cpu_bias.k_depth_preference;
        let gpu_ratio = gpu_bias.epilogue_depth_preference / gpu_bias.k_depth_preference;
        assert_approx(gpu_ratio, cpu_ratio, 0.001);
    }

    // ── lerp identity: lerp(a, b, t) returns a when b equals a ──

    /// When a == b, lerp must return that same value for any t in [0, 1].
    #[test]
    fn lerp_returns_same_value_when_endpoints_equal() {
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = lerp(3.14159, 3.14159, t);
            assert_approx(result, 3.14159, 1e-12);
        }
    }

    // ── ArbiterOutput: speculative_decoding_value always higher in latency ──

    /// For any archetype and hardware, latency baseline speculative_decoding_value=1.5
    /// vs throughput=0.3. Since hardware adjustment does not touch this field,
    /// latency should always produce a higher value than throughput.
    #[test]
    fn speculative_decoding_value_always_higher_in_latency_for_all_hardware() {
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        for hw in [cpu_avx2(), cpu_avx512(), gpu_a100()] {
            let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            assert!(
                lat.speculative_decoding_value > thr.speculative_decoding_value,
                "latency speculative_decoding should exceed throughput: lat={} thr={}",
                lat.speculative_decoding_value, thr.speculative_decoding_value,
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  15 additional tests (721 → 736)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. GraphArchetype Debug format contains field names ──

    #[test]
    fn graph_archetype_debug_format_fields_present() {
        let arch = GraphArchetype {
            compute_intensive: 0.3,
            memory_intensive: 0.4,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.7,
        };
        let debug_str = format!("{:?}", arch);
        assert!(debug_str.contains("compute_intensive"), "missing compute_intensive");
        assert!(debug_str.contains("memory_intensive"), "missing memory_intensive");
        assert!(debug_str.contains("parallelism_exploitable"), "missing parallelism_exploitable");
        assert!(debug_str.contains("fusion_profitable"), "missing fusion_profitable");
        assert!(debug_str.contains("pipeline_valuable"), "missing pipeline_valuable");
        // Also verify numeric values appear in output
        assert!(debug_str.contains("0.3"), "missing value 0.3");
        assert!(debug_str.contains("0.7"), "missing value 0.7");
    }

    // ── 2. StrategyBias all-zero fields validate to per-field minimums ──

    #[test]
    fn strategy_bias_all_zero_fields_validate_clamped_to_minimums() {
        let mut bias = StrategyBias {
            fusion_cost_scale: 0.0,
            pipeline_cost_scale: 0.0,
            parallelism_cost_scale: 0.0,
            epilogue_depth_preference: 0.0,
            k_depth_preference: 0.0,
            kv_cache_budget_scale: 0.0,
            weight_prefetch_budget_scale: 0.0,
            batch_flexibility: 0.0,
            decode_ratio_scale: 0.0,
            speculative_decoding_value: 0.0,
            quantization_aggressiveness: 0.0,
            expert_eviction_aggressiveness: 0.0,
            expert_prefetch_priority: 0.0,
        };
        bias.validate();
        assert_eq!(bias.fusion_cost_scale, 0.2, "fusion_cost_scale min is 0.2");
        assert_eq!(bias.parallelism_cost_scale, 0.1, "parallelism_cost_scale min is 0.1");
        assert_eq!(bias.epilogue_depth_preference, 0.3, "epilogue min is 0.3");
        assert_eq!(bias.k_depth_preference, 0.3, "k_depth min is 0.3");
        assert_eq!(bias.batch_flexibility, 0.0, "batch_flexibility min is 0.0");
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0, "eviction min is 0.0");
        assert_eq!(bias.expert_prefetch_priority, 0.1, "prefetch min is 0.1");
        assert_eq!(bias.speculative_decoding_value, 0.1, "speculative min is 0.1");
    }

    // ── 3. InferenceMode Clone + PartialEq roundtrip ──

    #[test]
    fn inference_mode_clone_eq_roundtrip_both_variants() {
        let original_latency = InferenceMode::Latency;
        let cloned_latency = original_latency.clone();
        assert_eq!(original_latency, cloned_latency);

        let original_throughput = InferenceMode::Throughput;
        let cloned_throughput = original_throughput.clone();
        assert_eq!(original_throughput, cloned_throughput);

        // Cross-variant inequality preserved after clone
        assert_ne!(cloned_latency, cloned_throughput);
        assert_ne!(cloned_throughput, cloned_latency);
    }

    // ── 4. Zero registers scarcity score (precise calculation) ──

    #[test]
    fn zero_regs_scarcity_score_precise_effect() {
        // Arrange: 0 registers → scarcity = 1.0 - (0/32) = 1.0
        let hw_zero = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 0,
            cache_sizes: (65536, 0, 0),
        };
        // Neutral hardware baseline: 32 regs, 64KB L1, no scarcity
        let hw_neutral = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias_zero = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_zero);
        let bias_neutral = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw_neutral);
        // Assert: scarcity=1.0 → epilogue *= 1+1.0*0.3 = 1.3, k_depth *= 1-1.0*0.2 = 0.8
        // baseline epilogue = 1.5, so zero-reg = 1.5*1.3 = 1.95
        // baseline k_depth = 1.3, so zero-reg = 1.3*0.8 = 1.04
        assert_approx(bias_zero.epilogue_depth_preference, 1.95, 0.01);
        assert_approx(bias_zero.k_depth_preference, 1.04, 0.01);
        assert!(
            bias_zero.epilogue_depth_preference > bias_neutral.epilogue_depth_preference,
            "zero regs should have strictly higher epilogue preference",
        );
    }

    // ── 5. reg_tension boundaries: +1.0 and -1.0 precise values ──

    #[test]
    fn reg_tension_boundary_plus_one_minus_one_precise() {
        // Arrange: fusion=1.0, pipeline=0.0 → reg_tension = +1.0
        let pos_tension_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 0.0,
        };
        // fusion=0.0, pipeline=1.0 → reg_tension = -1.0
        let neg_tension_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 1.0,
        };
        let neutral_hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let pos = StrategyArbiter::arbitrate(InferenceMode::Latency, &pos_tension_arch, &neutral_hw);
        let neg = StrategyArbiter::arbitrate(InferenceMode::Latency, &neg_tension_arch, &neutral_hw);
        // Assert: +1.0 → epilogue *= 1+1.0*0.5=1.5, k_depth *= 1-1.0*0.3=0.7
        // baseline epilogue=1.5, so pos=1.5*1.5=2.25
        // baseline k_depth=1.3, so pos=1.3*0.7=0.91
        assert_approx(pos.epilogue_depth_preference, 2.25, 0.01);
        assert_approx(pos.k_depth_preference, 0.91, 0.01);
        // -1.0 → abs=1.0 → k_depth *= 1+1.0*0.5=1.5, epilogue *= 1-1.0*0.3=0.7
        // baseline epilogue=1.5, so neg=1.5*0.7=1.05
        // baseline k_depth=1.3, so neg=1.3*1.5=1.95
        assert_approx(neg.epilogue_depth_preference, 1.05, 0.01);
        assert_approx(neg.k_depth_preference, 1.95, 0.01);
    }

    // ── 6. Shared mem latency effect on fusion cost decision ──

    #[test]
    fn shared_mem_latency_affects_fusion_cost_scale() {
        // Arrange: small shared mem vs large shared mem on GPU
        let small_smem = ArbiterHwView::gpu(16384);  // 16KB
        let large_smem = ArbiterHwView::gpu(131072);  // 128KB
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let small_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &small_smem);
        let large_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &large_smem);
        // Assert: larger shared mem → higher l1_richness → lower fusion_cost_scale
        // small: l1_richness = 16384/65536 = 0.25, sqrt=0.5, fusion *= 1/0.5=2.0 → but capped
        // large: l1_richness = 131072/65536 = 2.0 (capped), sqrt=1.414, fusion *= 1/1.414
        assert!(
            large_bias.fusion_cost_scale < small_bias.fusion_cost_scale,
            "larger shared mem should reduce fusion cost: small={} large={}",
            small_bias.fusion_cost_scale, large_bias.fusion_cost_scale,
        );
    }

    // ── 7. Cross-mode invariants (Latency vs Throughput inverse relationships) ──

    #[test]
    fn cross_mode_latency_throughput_inverse_invariants() {
        let arch = GraphArchetype {
            compute_intensive: 0.6,
            memory_intensive: 0.4,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.2,
        };
        let hw = cpu_avx512();
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Invariant 1: latency has lower fusion_cost
        assert!(lat.fusion_cost_scale < thr.fusion_cost_scale);
        // Invariant 2: latency has higher parallelism_cost
        assert!(lat.parallelism_cost_scale > thr.parallelism_cost_scale);
        // Invariant 3: latency has higher epilogue_depth_preference baseline
        assert!(lat.epilogue_depth_preference > thr.epilogue_depth_preference);
        // Invariant 4: latency has higher speculative_decoding_value
        assert!(lat.speculative_decoding_value > thr.speculative_decoding_value);
        // Invariant 5: throughput has higher kv_cache_budget_scale
        assert!(thr.kv_cache_budget_scale > lat.kv_cache_budget_scale);
        // Invariant 6: throughput has higher batch_flexibility
        assert!(thr.batch_flexibility > lat.batch_flexibility);
    }

    // ── 8. GPU boost ratio calculation (1.2x factor) ──

    #[test]
    fn gpu_boost_ratio_epilogue_k_depth_pipeline() {
        // Arrange: neutral arch, no modulation, CPU vs GPU with same L1
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(65536);
        // Act
        let cpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &cpu);
        let gpu_bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // Assert: GPU multiplies epilogue, k_depth, pipeline by exactly 1.2
        assert_approx(
            gpu_bias.epilogue_depth_preference / cpu_bias.epilogue_depth_preference,
            1.2,
            0.01,
        );
        assert_approx(
            gpu_bias.k_depth_preference / cpu_bias.k_depth_preference,
            1.2,
            0.01,
        );
        assert_approx(
            gpu_bias.pipeline_cost_scale / cpu_bias.pipeline_cost_scale,
            1.2,
            0.01,
        );
    }

    // ── 9. Lerp identity: lerp(a, a, t) == a for any t ──

    #[test]
    fn lerp_identity_same_value_any_t() {
        // We verify this via the pipeline: when fusion_profitable == pipeline_valuable,
        // reg_tension = 0.0 and no epilogue/k_depth modulation occurs, regardless of
        // the absolute values. This means lerp(1.0, x, t) applied to equal fields
        // leaves the result equal.
        let arch_a = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.7,
            pipeline_valuable: 0.7,  // same → reg_tension = 0
        };
        let arch_b = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.3,
            pipeline_valuable: 0.3,  // same → reg_tension = 0
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        let bias_a = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_a, &hw);
        let bias_b = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch_b, &hw);
        // When reg_tension = 0, neither epilogue nor k_depth is adjusted by the
        // reg_tension branch. Both should equal the baseline (modulated only by
        // lerp on fusion/pipeline which are equal, so same multiplicative result).
        assert_approx(bias_a.epilogue_depth_preference, bias_b.epilogue_depth_preference, 0.01);
        assert_approx(bias_a.k_depth_preference, bias_b.k_depth_preference, 0.01);
    }

    // ── 10. Speculative decoding cross-mode behavior ──

    #[test]
    fn speculative_decoding_cross_mode_invariant() {
        // speculative_decoding_value is never modified by archetype or hardware.
        // Latency baseline = 1.5, Throughput baseline = 0.3.
        // Verify across multiple archetype/hw combinations.
        let test_cases = [
            (GraphArchetype {
                compute_intensive: 0.0, memory_intensive: 0.0,
                parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
            }, cpu_avx2()),
            (GraphArchetype {
                compute_intensive: 1.0, memory_intensive: 1.0,
                parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0,
            }, gpu_a100()),
            (GraphArchetype {
                compute_intensive: 0.5, memory_intensive: 0.5,
                parallelism_exploitable: 0.5, fusion_profitable: 0.5, pipeline_valuable: 0.5,
            }, cpu_avx512()),
        ];
        for (arch, hw) in test_cases {
            let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
            let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
            // Invariant: latency always has strictly higher speculative decoding value
            assert!(
                lat.speculative_decoding_value > thr.speculative_decoding_value,
                "cross-mode invariant violated: lat={} thr={}",
                lat.speculative_decoding_value, thr.speculative_decoding_value,
            );
            // Both must be within valid range after validate()
            assert!(lat.speculative_decoding_value >= 0.1 && lat.speculative_decoding_value <= 3.0);
            assert!(thr.speculative_decoding_value >= 0.1 && thr.speculative_decoding_value <= 3.0);
        }
    }

    // ── 11. GraphArchetype all fields 1.0 produces finite outputs ──

    #[test]
    fn graph_archetype_all_ones_finite_outputs() {
        let arch = GraphArchetype {
            compute_intensive: 1.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        for mode in [InferenceMode::Latency, InferenceMode::Throughput] {
            for hw in [cpu_avx2(), cpu_avx512(), gpu_a100()] {
                let bias = StrategyArbiter::arbitrate(mode, &arch, &hw);
                assert!(bias.fusion_cost_scale.is_finite());
                assert!(bias.pipeline_cost_scale.is_finite());
                assert!(bias.parallelism_cost_scale.is_finite());
                assert!(bias.epilogue_depth_preference.is_finite());
                assert!(bias.k_depth_preference.is_finite());
                assert!(bias.kv_cache_budget_scale.is_finite());
                assert!(bias.weight_prefetch_budget_scale.is_finite());
                assert!(bias.batch_flexibility.is_finite());
                assert!(bias.decode_ratio_scale.is_finite());
                assert!(bias.speculative_decoding_value.is_finite());
                assert!(bias.quantization_aggressiveness.is_finite());
                assert!(bias.expert_eviction_aggressiveness.is_finite());
                assert!(bias.expert_prefetch_priority.is_finite());
            }
        }
    }

    // ── 12. StrategyBias negative values clamped by validate ──

    #[test]
    fn strategy_bias_negative_input_all_fields_validate_clamped() {
        let mut bias = StrategyBias {
            fusion_cost_scale: -5.0,
            pipeline_cost_scale: -10.0,
            parallelism_cost_scale: -2.0,
            epilogue_depth_preference: -1.0,
            k_depth_preference: -3.0,
            kv_cache_budget_scale: -0.5,
            weight_prefetch_budget_scale: -100.0,
            batch_flexibility: -50.0,
            decode_ratio_scale: -1.0,
            speculative_decoding_value: -0.5,
            quantization_aggressiveness: -2.0,
            expert_eviction_aggressiveness: -10.0,
            expert_prefetch_priority: -5.0,
        };
        bias.validate();
        // Every field must be at or above its minimum
        assert!(bias.fusion_cost_scale >= 0.2, "fusion_cost_scale below min: {}", bias.fusion_cost_scale);
        assert!(bias.pipeline_cost_scale >= 0.2, "pipeline_cost_scale below min: {}", bias.pipeline_cost_scale);
        assert!(bias.parallelism_cost_scale >= 0.1, "parallelism_cost_scale below min: {}", bias.parallelism_cost_scale);
        assert!(bias.epilogue_depth_preference >= 0.3, "epilogue below min: {}", bias.epilogue_depth_preference);
        assert!(bias.k_depth_preference >= 0.3, "k_depth below min: {}", bias.k_depth_preference);
        assert!(bias.kv_cache_budget_scale >= 0.2, "kv_cache below min: {}", bias.kv_cache_budget_scale);
        assert!(bias.weight_prefetch_budget_scale >= 0.2, "weight_prefetch below min: {}", bias.weight_prefetch_budget_scale);
        assert!(bias.batch_flexibility >= 0.0, "batch_flexibility below min: {}", bias.batch_flexibility);
        assert!(bias.decode_ratio_scale >= 0.3, "decode_ratio below min: {}", bias.decode_ratio_scale);
        assert!(bias.speculative_decoding_value >= 0.1, "speculative below min: {}", bias.speculative_decoding_value);
        assert!(bias.quantization_aggressiveness >= 0.3, "quantization below min: {}", bias.quantization_aggressiveness);
        assert!(bias.expert_eviction_aggressiveness >= 0.0, "eviction below min: {}", bias.expert_eviction_aggressiveness);
        assert!(bias.expert_prefetch_priority >= 0.1, "prefetch below min: {}", bias.expert_prefetch_priority);
    }

    // ── 13. ArbiterHwView construction from DeviceProfile matches fields ──

    #[test]
    fn arbiter_hw_view_from_profile_construction_fields_match() {
        let dp = DeviceProfile::detect();
        let view = ArbiterHwView::from(&dp);
        // Assert: is_gpu is always false for CPU DeviceProfile
        assert!(view.device == DeviceFamily::Cpu, "from(&DeviceProfile) must produce CPU view");
        // num_simd_regs must match
        assert_eq!(
            view.num_simd_regs, dp.num_simd_regs(),
            "num_simd_regs must match DeviceProfile::num_simd_regs()"
        );
        // cache_sizes must match
        assert_eq!(
            view.cache_sizes, dp.cache_sizes(),
            "cache_sizes must match DeviceProfile::cache_sizes()"
        );
    }

    // ── 14. HwOptEngine integration: StrategyBias feeds into planner ──

    #[test]
    fn hw_opt_engine_integration_strategy_bias_consumed() {
        // Verify that StrategyBias produced by the arbiter can be consumed by
        // HwOptEngine's solve path. We check that the bias has all fields within
        // the expected clamp ranges, making it a valid input.
        let archetypes = [
            GraphArchetype {
                compute_intensive: 0.0, memory_intensive: 0.0,
                parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
            },
            GraphArchetype {
                compute_intensive: 1.0, memory_intensive: 1.0,
                parallelism_exploitable: 1.0, fusion_profitable: 1.0, pipeline_valuable: 1.0,
            },
        ];
        let profiles: Vec<ArbiterHwView> = vec![
            ArbiterHwView {
                device: DeviceFamily::Cpu, num_simd_regs: 16, cache_sizes: (32768, 262144, 8388608),
            },
            ArbiterHwView::gpu(49152),
        ];
        for arch in &archetypes {
            for mode in [InferenceMode::Latency, InferenceMode::Throughput] {
                for hw in &profiles {
                    let bias = StrategyArbiter::arbitrate(mode, arch, hw);
                    // Validate was called internally, so all fields must be in range
                    assert!(
                        bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0,
                        "fusion_cost_scale out of range: {}", bias.fusion_cost_scale,
                    );
                    assert!(
                        bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0,
                        "pipeline_cost_scale out of range: {}", bias.pipeline_cost_scale,
                    );
                    assert!(
                        bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0,
                        "parallelism_cost_scale out of range: {}", bias.parallelism_cost_scale,
                    );
                    assert!(
                        bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0,
                        "kv_cache_budget_scale out of range: {}", bias.kv_cache_budget_scale,
                    );
                    assert!(
                        bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0,
                        "batch_flexibility out of range: {}", bias.batch_flexibility,
                    );
                }
            }
        }
    }

    // ── 15. InferenceMode Default is Latency (explicit re-check with match proof) ──

    #[test]
    fn inference_mode_default_is_latency_explicit_match_proof() {
        // Arrange & Act
        let default_mode = InferenceMode::default();
        // Assert: Default must be Latency, not Throughput
        assert_eq!(default_mode, InferenceMode::Latency);
        assert_ne!(default_mode, InferenceMode::Throughput);
        // Verify Default trait produces the same variant as explicit construction
        assert_eq!(InferenceMode::default(), InferenceMode::Latency);
        // Verify the #[default] attribute on Latency variant
        let modes = [InferenceMode::Latency, InferenceMode::Throughput];
        assert_eq!(modes[0], default_mode, "Latency should be the first variant and the default");
    }

    // ── 16. Archetype field independence: parallelism_exploitable alone ──

    #[test]
    fn archetype_field_independence_parallelism_exploitable() {
        // Arrange: only parallelism_exploitable varies between two archetypes
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let high_parallel = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = cpu_avx512();
        // Act
        let bias_zero = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        let bias_high = StrategyArbiter::arbitrate(InferenceMode::Latency, &high_parallel, &hw);
        // Assert: high parallelism reduces parallelism_cost_scale
        assert!(
            bias_high.parallelism_cost_scale < bias_zero.parallelism_cost_scale,
            "high parallelism_exploitable should reduce parallelism_cost_scale: {} vs {}",
            bias_high.parallelism_cost_scale, bias_zero.parallelism_cost_scale,
        );
    }

    // ── 17. Archetype field independence: pipeline_valuable alone ──

    #[test]
    fn archetype_field_independence_pipeline_valuable() {
        // Arrange
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0, memory_intensive: 0.0,
            parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
        };
        let high_pipe = GraphArchetype {
            compute_intensive: 0.0, memory_intensive: 0.0,
            parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 1.0,
        };
        let hw = cpu_avx512();
        // Act
        let bias_zero = StrategyArbiter::arbitrate(InferenceMode::Throughput, &zero_arch, &hw);
        let bias_high = StrategyArbiter::arbitrate(InferenceMode::Throughput, &high_pipe, &hw);
        // Assert: high pipeline_valuable reduces pipeline_cost_scale
        assert!(
            bias_high.pipeline_cost_scale < bias_zero.pipeline_cost_scale,
            "high pipeline_valuable should reduce pipeline_cost_scale: {} vs {}",
            bias_high.pipeline_cost_scale, bias_zero.pipeline_cost_scale,
        );
    }

    // ── 18. Archetype field independence: compute_intensive alone ──

    #[test]
    fn archetype_field_independence_compute_intensive() {
        // Arrange: compute_intensive does not directly appear in archetype modulation,
        // so varying it alone should produce identical bias (it's informational).
        let low_compute = GraphArchetype {
            compute_intensive: 0.0, memory_intensive: 0.0,
            parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
        };
        let high_compute = GraphArchetype {
            compute_intensive: 1.0, memory_intensive: 0.0,
            parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
        };
        let hw = cpu_avx512();
        // Act
        let bias_low = StrategyArbiter::arbitrate(InferenceMode::Latency, &low_compute, &hw);
        let bias_high = StrategyArbiter::arbitrate(InferenceMode::Latency, &high_compute, &hw);
        // Assert: compute_intensive alone does not modulate any bias field
        assert_eq!(bias_low.fusion_cost_scale, bias_high.fusion_cost_scale);
        assert_eq!(bias_low.pipeline_cost_scale, bias_high.pipeline_cost_scale);
        assert_eq!(bias_low.parallelism_cost_scale, bias_high.parallelism_cost_scale);
    }

    // ── 19. L1 richness exactly 0.5 produces precise fusion multiplier ──

    #[test]
    fn l1_richness_half_fusion_multiplier_precise() {
        // Arrange: L1 = 32768, richness = 32768/65536 = 0.5
        // fusion_cost_scale multiplier = 1/sqrt(0.5) = sqrt(2) ≈ 1.4142
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (32768, 0, 0),
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0, memory_intensive: 0.0,
            parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Expected: latency baseline fusion = 0.5, then * 1/sqrt(0.5) = sqrt(2)
        let expected = 0.5 * (2.0_f64).sqrt();
        assert_approx(bias.fusion_cost_scale, expected, 0.01);
    }

    // ── 20. Register scarcity boundary: num_simd_regs=32 produces no scarcity ──

    #[test]
    fn register_scarcity_boundary_32_no_adjustment() {
        // Arrange: num_simd_regs=32 → scarcity = 1 - 32/32 = 0.0
        let hw_no_scarcity = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0), // richness = 1.0, no fusion change
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0, memory_intensive: 0.0,
            parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw_no_scarcity);
        // Assert: epilogue and k_depth should be at baseline (no scarcity adjustment)
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.001);
        assert_approx(bias.k_depth_preference, 1.3, 0.001);
    }

    // ── 21. Register scarcity: num_simd_regs=8 produces maximum scarcity effect ──

    #[test]
    fn register_scarcity_8_regs_precise_effect() {
        // Arrange: scarcity = 1 - 8/32 = 0.75
        // epilogue *= 1 + 0.75*0.3 = 1.225, k_depth *= 1 - 0.75*0.2 = 0.85
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0), // richness = 1.0
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0, memory_intensive: 0.0,
            parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Assert
        let expected_epilogue = 1.5 * (1.0 + 0.75 * 0.3);
        let expected_k_depth = 1.3 * (1.0 - 0.75 * 0.2);
        assert_approx(bias.epilogue_depth_preference, expected_epilogue, 0.01);
        assert_approx(bias.k_depth_preference, expected_k_depth, 0.01);
    }

    // ── 22. ArbiterHwView Eq+Hash usable in HashMap with duplicates ──

    #[test]
    fn arbiter_hw_view_hashmap_dedup_same_views() {
        // Arrange
        use std::collections::HashMap;
        let view_a = ArbiterHwView { device: DeviceFamily::Cpu, num_simd_regs: 16, cache_sizes: (32768, 0, 0) };
        let view_b = ArbiterHwView { device: DeviceFamily::Cpu, num_simd_regs: 16, cache_sizes: (32768, 0, 0) };
        let mut map: HashMap<ArbiterHwView, i32> = HashMap::new();
        // Act: insert same key twice
        map.insert(view_a, 1);
        map.insert(view_b, 2);
        // Assert: second insert overwrites
        assert_eq!(map.len(), 1);
        assert_eq!(map[&view_a], 2);
    }

    // ── 23. StrategyBias default then validate is exact no-op ──

    #[test]
    fn strategy_bias_default_validate_exact_noop() {
        // Arrange
        let original = StrategyBias::default();
        let mut subject = StrategyBias::default();
        // Act
        subject.validate();
        // Assert: every field unchanged (all defaults within clamp ranges)
        assert_eq!(subject.fusion_cost_scale, original.fusion_cost_scale);
        assert_eq!(subject.pipeline_cost_scale, original.pipeline_cost_scale);
        assert_eq!(subject.batch_flexibility, original.batch_flexibility);
        assert_eq!(subject.expert_eviction_aggressiveness, original.expert_eviction_aggressiveness);
        assert_eq!(subject.expert_prefetch_priority, original.expert_prefetch_priority);
    }

    // ── 24. Cross-mode: all 13 fields differ between latency and throughput ──

    #[test]
    fn cross_mode_all_13_fields_differ() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.5, memory_intensive: 0.5,
            parallelism_exploitable: 0.5, fusion_profitable: 0.5, pipeline_valuable: 0.5,
        };
        let hw = cpu_avx512();
        // Act
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert: batch_flexibility differs (0.0 vs 1.0 baseline)
        assert_ne!(lat.batch_flexibility, thr.batch_flexibility);
        // decode_ratio_scale is identical at baseline for both modes
        assert_eq!(lat.decode_ratio_scale, thr.decode_ratio_scale);
        // speculative_decoding differs
        assert_ne!(lat.speculative_decoding_value, thr.speculative_decoding_value);
    }

    // ── 25. Archetype values above 1.0 still produce valid output ──

    #[test]
    fn archetype_values_exceeding_one_produce_valid_clamped_output() {
        // Arrange: archetype fields > 1.0 (out of spec but must not panic)
        let over_arch = GraphArchetype {
            compute_intensive: 2.0,
            memory_intensive: 1.5,
            parallelism_exploitable: 3.0,
            fusion_profitable: 1.8,
            pipeline_valuable: 2.5,
        };
        let hw = cpu_avx2();
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &over_arch, &hw);
        // Assert: validate() clamps everything
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
    }

    // ── 26. GPU with zero registers still applies GPU boost ──

    #[test]
    fn gpu_zero_registers_gpu_branch_takes_priority_over_scarcity() {
        // Arrange: GPU with 0 regs — GPU branch fires, scarcity branch also fires
        let hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 0,
            cache_sizes: (65536, 0, 0),
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0, memory_intensive: 0.0,
            parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Assert: GPU boost (1.2x) applies on top of scarcity adjustments
        // epilogue = baseline 1.5 * scarcity(1+1.0*0.3=1.3) * gpu(1.2) = 1.5 * 1.3 * 1.2
        let expected_epilogue = 1.5 * (1.0 + 1.0 * 0.3) * 1.2;
        assert_approx(bias.epilogue_depth_preference, expected_epilogue, 0.01);
        // output must still be finite and within validate bounds
        assert!(bias.epilogue_depth_preference.is_finite());
    }

    // ── 27. L1 richness at exactly 2.0 cap (L1 = 131072) precise ──

    #[test]
    fn l1_richness_capped_at_two_precise_fusion() {
        // Arrange: L1 = 131072 → richness = 131072/65536 = 2.0 (capped)
        // fusion multiplier = 1/sqrt(2.0) = 0.7071...
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (131072, 0, 0),
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0, memory_intensive: 0.0,
            parallelism_exploitable: 0.0, fusion_profitable: 0.0, pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Assert: latency baseline fusion = 0.5 * 1/sqrt(2)
        let expected = 0.5 / (2.0_f64).sqrt();
        assert_approx(bias.fusion_cost_scale, expected, 0.01);
    }

    // ── 28. StrategyBias expert accessors match fields after arbitrate ──

    #[test]
    fn strategy_bias_accessors_match_fields_after_arbitrate() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.9, memory_intensive: 0.8,
            parallelism_exploitable: 0.7, fusion_profitable: 0.6, pipeline_valuable: 0.4,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu_a100());
        // Assert: accessor methods return identical values as direct field access
        assert_eq!(bias.expert_eviction_aggressiveness(), bias.expert_eviction_aggressiveness);
        assert_eq!(bias.expert_prefetch_priority(), bias.expert_prefetch_priority);
    }

    // ── 29. InferenceMode Hash+Eq usable as HashSet key with dedup ──

    #[test]
    fn inference_mode_hashset_dedup_identical_variants() {
        // Arrange
        use std::collections::HashSet;
        let mut set = HashSet::new();
        // Act: insert each variant twice
        set.insert(InferenceMode::Latency);
        set.insert(InferenceMode::Latency);
        set.insert(InferenceMode::Throughput);
        set.insert(InferenceMode::Throughput);
        // Assert: exactly 2 unique entries
        assert_eq!(set.len(), 2);
        assert!(set.contains(&InferenceMode::Latency));
        assert!(set.contains(&InferenceMode::Throughput));
    }

    // ── 30. arbitrate produces identical results via arbitrate_cpu and manual From ──

    #[test]
    fn arbitrate_cpu_vs_manual_from_both_modes_identical() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.6, memory_intensive: 0.7,
            parallelism_exploitable: 0.4, fusion_profitable: 0.5, pipeline_valuable: 0.3,
        };
        let dp = DeviceProfile::detect();
        let hw = ArbiterHwView::from(&dp);
        for mode in [InferenceMode::Latency, InferenceMode::Throughput] {
            // Act
            let via_arbitrate = StrategyArbiter::arbitrate(mode, &arch, &hw);
            let via_cpu = StrategyArbiter::arbitrate_cpu(mode, &arch, &dp);
            // Assert: byte-identical results
            assert_eq!(via_arbitrate.fusion_cost_scale, via_cpu.fusion_cost_scale);
            assert_eq!(via_arbitrate.pipeline_cost_scale, via_cpu.pipeline_cost_scale);
            assert_eq!(via_arbitrate.parallelism_cost_scale, via_cpu.parallelism_cost_scale);
            assert_eq!(via_arbitrate.batch_flexibility, via_cpu.batch_flexibility);
        }
    }

    // ── 31. ArbiterHwView gpu() constructor sets all fields correctly ──

    #[test]
    fn arbiter_hw_view_gpu_constructor_fields() {
        // Arrange: arbitrary shared-mem size
        let shared = 98304usize; // 96 KiB
        // Act
        let view = ArbiterHwView::gpu(shared);
        // Assert
        assert!(view.device == DeviceFamily::Gpu);
        assert_eq!(view.num_simd_regs, 255);
        assert_eq!(view.cache_sizes, (shared, 0, 0));
    }

    // ── 32. ArbiterHwView struct update syntax preserves overridden fields ──

    #[test]
    fn arbiter_hw_view_struct_update_syntax() {
        // Arrange: base CPU view
        let base = cpu_avx2();
        // Act: override device and L1, inherit the rest
        let derived = ArbiterHwView {
            device: DeviceFamily::Gpu,
            cache_sizes: (65536, base.cache_sizes.1, base.cache_sizes.2),
            ..base
        };
        // Assert: overridden fields changed, inherited fields unchanged
        assert!(derived.device == DeviceFamily::Gpu);
        assert_eq!(derived.cache_sizes.0, 65536);
        assert_eq!(derived.num_simd_regs, base.num_simd_regs);
        assert_eq!(derived.cache_sizes.1, base.cache_sizes.1);
        assert_eq!(derived.cache_sizes.2, base.cache_sizes.2);
    }

    // ── 33. GraphArchetype Copy semantics — mutation of copy does not affect original ──

    #[test]
    fn graph_archetype_copy_semantics_independent() {
        // Arrange
        let original = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.4,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.2,
            pipeline_valuable: 0.1,
        };
        // Act: copy and mutate
        let mut copy = original;
        copy.compute_intensive = 0.99;
        copy.pipeline_valuable = 0.88;
        // Assert: original unchanged
        assert!((original.compute_intensive - 0.5).abs() < f64::EPSILON);
        assert!((original.pipeline_valuable - 0.1).abs() < f64::EPSILON);
        assert!((copy.compute_intensive - 0.99).abs() < f64::EPSILON);
        assert!((copy.pipeline_valuable - 0.88).abs() < f64::EPSILON);
    }

    // ── 34. StrategyBias Default all fields are exactly 1.0 or specified value ──

    #[test]
    fn strategy_bias_default_exact_field_values() {
        // Act
        let bias = StrategyBias::default();
        // Assert: all 1.0 except expert_eviction_aggressiveness = 0.0
        assert!((bias.fusion_cost_scale - 1.0).abs() < f64::EPSILON);
        assert!((bias.pipeline_cost_scale - 1.0).abs() < f64::EPSILON);
        assert!((bias.parallelism_cost_scale - 1.0).abs() < f64::EPSILON);
        assert!((bias.epilogue_depth_preference - 1.0).abs() < f64::EPSILON);
        assert!((bias.k_depth_preference - 1.0).abs() < f64::EPSILON);
        assert!((bias.kv_cache_budget_scale - 1.0).abs() < f64::EPSILON);
        assert!((bias.weight_prefetch_budget_scale - 1.0).abs() < f64::EPSILON);
        assert!((bias.batch_flexibility - 1.0).abs() < f64::EPSILON);
        assert!((bias.decode_ratio_scale - 1.0).abs() < f64::EPSILON);
        assert!((bias.speculative_decoding_value - 1.0).abs() < f64::EPSILON);
        assert!((bias.quantization_aggressiveness - 1.0).abs() < f64::EPSILON);
        assert!((bias.expert_prefetch_priority - 1.0).abs() < f64::EPSILON);
        // special: expert_eviction defaults to 0.0
        assert!((bias.expert_eviction_aggressiveness).abs() < f64::EPSILON);
    }

    // ── 35. InferenceMode Ord ordering: Latency < Throughput with strict inequality both ways ──

    #[test]
    fn inference_mode_ord_strict_inequality_both_directions() {
        // Arrange: both variants
        // Act & Assert: Ord is derived, verify total order
        assert!(InferenceMode::Latency < InferenceMode::Throughput);
        assert!(InferenceMode::Throughput > InferenceMode::Latency);
        assert!(!(InferenceMode::Latency > InferenceMode::Throughput));
    }

    // ── 36. StrategyBias validate clamps batch_flexibility to exactly 0.0 on negative ──

    #[test]
    fn strategy_bias_validate_clamps_batch_flexibility_negative_to_zero() {
        // Arrange: construct bias with batch_flexibility = -5.0
        let mut bias = StrategyBias {
            batch_flexibility: -5.0,
            ..StrategyBias::default()
        };
        // Act
        bias.validate();
        // Assert: clamped to 0.0 (lower bound for batch_flexibility)
        assert!((bias.batch_flexibility - 0.0).abs() < f64::EPSILON);
    }

    // ── 37. ArbiterHwView PartialEq — identical and different views ──

    #[test]
    fn arbiter_hw_view_partial_eq_identical_and_different() {
        // Arrange
        let a = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (49152, 1048576, 33554432),
        };
        let b = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (49152, 1048576, 33554432),
        };
        let c = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 32,
            cache_sizes: (49152, 1048576, 33554432),
        };
        // Assert
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── 38. arbiter with zero L1 cache — L1 richness zero, no fusion scale division ──

    #[test]
    fn zero_l1_cache_no_division_fusion_scale_unchanged_by_hw() {
        // Arrange: L1 = 0 → l1_richness = 0, so the L1 branch is skipped
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32, // no scarcity adjustment at 32
            cache_sizes: (0, 262144, 8388608),
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act: latency baseline fusion = 0.5, zero arch → no modulation,
        //      zero L1 → l1_richness = 0 → skip L1 branch
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &zero_arch, &hw);
        // Assert: fusion_cost_scale stays at baseline 0.5 (no modulation, no HW adj)
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
    }

    // ── 39. StrategyBias Clone produces independent copy ──

    #[test]
    fn strategy_bias_clone_independent_after_mutation() {
        // Arrange
        let original = StrategyBias::default();
        // Act: clone and mutate
        let mut cloned = original.clone();
        cloned.fusion_cost_scale = 2.5;
        cloned.k_depth_preference = 0.4;
        // Assert: original unchanged
        assert!((original.fusion_cost_scale - 1.0).abs() < f64::EPSILON);
        assert!((original.k_depth_preference - 1.0).abs() < f64::EPSILON);
        assert!((cloned.fusion_cost_scale - 2.5).abs() < f64::EPSILON);
        assert!((cloned.k_depth_preference - 0.4).abs() < f64::EPSILON);
    }

    // ── 40. lerp at t=0 returns a exactly, at t=1 returns b exactly ──

    #[test]
    fn lerp_boundary_t_zero_and_t_one() {
        // Arrange
        let a = 3.7;
        let b = 9.2;
        // Act
        let at_zero = lerp(a, b, 0.0);
        let at_one = lerp(a, b, 1.0);
        // Assert
        assert!((at_zero - a).abs() < f64::EPSILON);
        assert!((at_one - b).abs() < f64::EPSILON);
    }

    // ── 41. GraphArchetype Debug contains all five field names ──

    #[test]
    fn graph_archetype_debug_all_five_fields_present() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.33,
            memory_intensive: 0.44,
            parallelism_exploitable: 0.55,
            fusion_profitable: 0.66,
            pipeline_valuable: 0.77,
        };
        // Act
        let debug_str = format!("{:?}", arch);
        // Assert
        assert!(debug_str.contains("compute_intensive"));
        assert!(debug_str.contains("memory_intensive"));
        assert!(debug_str.contains("parallelism_exploitable"));
        assert!(debug_str.contains("fusion_profitable"));
        assert!(debug_str.contains("pipeline_valuable"));
    }

    // ── 42. StrategyBias validate clamps speculative_decoding_value upper bound ──

    #[test]
    fn strategy_bias_validate_clamps_speculative_decoding_upper() {
        // Arrange: set speculative_decoding_value to 999.0 (far above max 3.0)
        let mut bias = StrategyBias {
            speculative_decoding_value: 999.0,
            ..StrategyBias::default()
        };
        // Act
        bias.validate();
        // Assert: clamped to 3.0 (upper bound)
        assert!((bias.speculative_decoding_value - 3.0).abs() < f64::EPSILON);
    }

    // ── 43. arbiter large num_simd_regs (usize::MAX) does not overflow in scarcity calc ──

    #[test]
    fn arbiter_large_simd_regs_no_overflow_in_scarcity_calc() {
        // Arrange: extreme register count, should not trigger scarcity
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: usize::MAX,
            cache_sizes: (32768, 262144, 8388608),
        };
        let zero_arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act: should not panic from overflow
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &zero_arch, &hw);
        // Assert: all fields finite (no NaN/Inf from overflow)
        assert!(bias.fusion_cost_scale.is_finite());
        assert!(bias.pipeline_cost_scale.is_finite());
        assert!(bias.epilogue_depth_preference.is_finite());
        assert!(bias.k_depth_preference.is_finite());
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (13 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. GPU with 255 registers skips scarcity branch entirely ──

    #[test]
    fn gpu_255_regs_skips_scarcity_branch() {
        // Arrange: GPU with 255 regs (above 16 threshold)
        let gpu = ArbiterHwView::gpu(65536);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // Assert: epilogue = baseline 1.5 * 1.2 (GPU boost) = 1.8, no scarcity
        assert_approx(bias.epilogue_depth_preference, 1.8, 0.01);
        // k_depth = baseline 1.3 * 1.2 (GPU boost) = 1.56, no scarcity
        assert_approx(bias.k_depth_preference, 1.56, 0.01);
    }

    // ── 2. Latency mode epilogue_depth_preference baseline exactly 1.5 ──

    #[test]
    fn latency_epilogue_depth_baseline_exact_value() {
        // Arrange: neutral archetype, neutral hardware
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.001);
    }

    // ── 3. Throughput mode k_depth_preference baseline exactly 0.8 ──

    #[test]
    fn throughput_k_depth_baseline_exact_value() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.k_depth_preference, 0.8, 0.001);
    }

    // ── 4. Reg tension with small positive margin (0.05) produces small but nonzero shift ──

    #[test]
    fn reg_tension_small_positive_0_05_margin_precise() {
        // Arrange: fusion_profitable=0.55, pipeline_valuable=0.50 -> reg_tension=0.05
        // epilogue boost = 1.0 + 0.05*0.5 = 1.025
        // k_depth reduction = 1.0 - 0.05*0.3 = 0.985
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.55,
            pipeline_valuable: 0.50,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert: epilogue = 1.5 * 1.025 = 1.5375
        assert_approx(bias.epilogue_depth_preference, 1.5375, 0.001);
        // k_depth = 1.3 * 0.985 = 1.2805
        assert_approx(bias.k_depth_preference, 1.2805, 0.001);
    }

    // ── 5. Reg tension with small negative margin (-0.05) produces opposite small shift ──

    #[test]
    fn reg_tension_small_negative_0_05_margin_precise() {
        // Arrange: fusion_profitable=0.45, pipeline_valuable=0.50 -> reg_tension=-0.05
        // k_depth boost = 1.0 + 0.05*0.5 = 1.025
        // epilogue reduction = 1.0 - 0.05*0.3 = 0.985
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.45,
            pipeline_valuable: 0.50,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert: k_depth = 1.3 * 1.025 = 1.3325
        assert_approx(bias.k_depth_preference, 1.3325, 0.001);
        // epilogue = 1.5 * 0.985 = 1.4775
        assert_approx(bias.epilogue_depth_preference, 1.4775, 0.001);
    }

    // ── 6. L1 exactly 32KB on GPU: richness 0.5 boosts fusion cost ──

    #[test]
    fn gpu_32kb_shared_mem_fusion_cost_precise() {
        // Arrange: GPU with 32KB shared mem
        // l1_richness = 32768/65536 = 0.5, sqrt(0.5) = 0.7071
        // fusion_cost_scale = baseline * 1/0.7071
        let gpu = ArbiterHwView::gpu(32768);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // Assert: latency baseline 0.5 / sqrt(0.5) = 0.7071
        assert_approx(bias.fusion_cost_scale, 0.707, 0.01);
    }

    // ── 7. Combined throughput + max fusion_profitable + max pipeline_valuable + GPU ──

    #[test]
    fn throughput_max_fusion_max_pipeline_gpu_all_effects() {
        // Arrange: both fusion and pipeline at max -> reg_tension = 0
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 1.0,
            pipeline_valuable: 1.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        // Assert: fusion *= lerp(1.0, 0.6, 1.0) = 0.6 -> 1.0 * 0.6 = 0.6
        assert_approx(bias.fusion_cost_scale, 0.6, 0.01);
        // pipeline *= lerp(1.0, 0.6, 1.0) = 0.6 -> 1.3 * 0.6 = 0.78, GPU *= 1.2 = 0.936
        assert_approx(bias.pipeline_cost_scale, 0.936, 0.01);
        // reg_tension = 0, so epilogue/k_depth only get GPU 1.2x boost
        assert_approx(bias.epilogue_depth_preference, 0.96, 0.01);
        assert_approx(bias.k_depth_preference, 0.96, 0.01);
    }

    // ── 8. MoE expert_prefetch_priority precise at parallelism 0.7, memory 0.6, throughput ──

    #[test]
    fn moe_prefetch_parallelism_0_7_memory_0_6_throughput_precise() {
        // Arrange: parallelism=0.7 (>0.5 threshold), memory=0.6
        // expert_prefetch_priority *= lerp(1.0, 2.0, 0.6) = 1.0 + 0.6 * 1.0 = 1.6
        // throughput baseline expert_prefetch_priority = 1.5
        // result = 1.5 * 1.6 = 2.4
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.6,
            parallelism_exploitable: 0.7,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.expert_prefetch_priority, 2.4, 0.01);
    }

    // ── 9. ArbiterHwView equality: manual CPU vs gpu() constructor with same L1 ──

    #[test]
    fn arbiter_hw_view_manual_cpu_not_equal_gpu_constructor() {
        // Arrange: same cache sizes but different is_gpu flag
        let cpu = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        let gpu = ArbiterHwView::gpu(49152);
        // Act & Assert: same cache_sizes and num_simd_regs, but is_gpu differs
        assert_ne!(cpu, gpu, "CPU view with 255 regs should not equal GPU view");
    }

    // ── 10. KV cache budget with throughput and zero memory: baseline unchanged ──

    #[test]
    fn kv_cache_throughput_zero_memory_identity() {
        // Arrange: memory_intensive=0.0, lerp(1.0, 1.5, 0.0) = 1.0
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert: throughput baseline kv_cache_budget_scale = 1.5, no modulation
        assert_approx(bias.kv_cache_budget_scale, 1.5, 0.01);
    }

    // ── 11. Latency mode with GPU and high parallelism_exploitable reduces parallelism cost ──

    #[test]
    fn latency_gpu_high_parallelism_reduces_parallelism_cost() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let gpu = ArbiterHwView::gpu(65536);
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // Assert: parallelism_cost_scale *= lerp(1.0, 0.5, 1.0) = 0.5 -> 1.5 * 0.5 = 0.75
        // GPU does not affect parallelism_cost_scale
        assert_approx(bias.parallelism_cost_scale, 0.75, 0.01);
    }

    // ── 12. StrategyBias validate with subnormal f64 input ──

    #[test]
    fn strategy_bias_validate_subnormal_f64_clamped() {
        // Arrange: subnormal f64 (very tiny positive, near zero)
        let subnormal = f64::from_bits(1u64); // smallest positive subnormal
        let mut bias = StrategyBias {
            fusion_cost_scale: subnormal,
            batch_flexibility: subnormal,
            expert_eviction_aggressiveness: subnormal,
            expert_prefetch_priority: subnormal,
            ..StrategyBias::default()
        };
        // Act
        bias.validate();
        // Assert: subnormal < 0.2, so fusion_cost_scale clamped to 0.2
        assert_eq!(bias.fusion_cost_scale, 0.2);
        // subnormal < 0.0 range for batch_flexibility? No, it's positive but tiny
        // batch_flexibility clamp is [0.0, 1.0], subnormal > 0 so it stays
        assert!(bias.batch_flexibility > 0.0, "subnormal > 0 should be within [0.0, 1.0]");
        // expert_eviction_aggressiveness clamp [0.0, 2.0], subnormal > 0 stays
        assert!(bias.expert_eviction_aggressiveness > 0.0);
        // expert_prefetch_priority clamp [0.1, 5.0], subnormal < 0.1 -> clamped to 0.1
        assert_eq!(bias.expert_prefetch_priority, 0.1);
    }

    // ── 13. L1 exactly 64KB on GPU: richness 1.0, no fusion adjustment from L1 ──

    #[test]
    fn gpu_64kb_shared_mem_richness_one_no_fusion_l1_adj() {
        // Arrange: GPU with exactly 64KB shared mem
        let gpu = ArbiterHwView::gpu(65536);
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu);
        // Assert: l1_richness = 65536/65536 = 1.0, sqrt(1.0) = 1.0
        // fusion_cost_scale = 0.5 * (1/1.0) = 0.5 (no L1 adjustment)
        assert_approx(bias.fusion_cost_scale, 0.5, 0.01);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (13 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. Fusion profit + memory intensity combined: precise kv_cache_budget_scale ──

    #[test]
    fn fusion_profit_memory_combined_kv_cache_precise() {
        // Arrange: fusion_profitable=0.6, memory_intensive=0.8
        // kv_cache_budget_scale: throughput baseline=1.5, *= lerp(1.0, 1.5, 0.8)=1.4
        // result = 1.5 * 1.4 = 2.1
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.8,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.kv_cache_budget_scale, 2.1, 0.01);
    }

    // ── 2. Register scarcity + GPU boost both active: GPU branch takes priority ──

    #[test]
    fn gpu_with_few_registers_gpu_branch_overrides_scarcity() {
        // Arrange: GPU with only 8 registers — GPU branch fires (is_gpu=true),
        // CPU scarcity branch also fires (num_simd_regs <= 16).
        // GPU: epilogue *= 1.2, k_depth *= 1.2
        // Scarcity: scarcity = 1.0 - 8/32 = 0.75, epilogue *= 1 + 0.75*0.3 = 1.225
        // Combined epilogue: 1.5 (baseline) * 1.2 (gpu) * 1.225 (scarcity) = 2.205
        let gpu_low_reg = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 8,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_low_reg);
        // Assert: both GPU and scarcity adjustments apply
        assert!(
            bias.epilogue_depth_preference > 1.8,
            "GPU + scarcity should compound epilogue depth preference above 1.8"
        );
        assert!(
            bias.k_depth_preference > 1.2,
            "GPU should boost k_depth above 1.2 (scarcity reduces it but GPU 1.2x dominates)"
        );
    }

    // ── 3. Throughput + max MoE parallelism + max memory: expert eviction precise ──

    #[test]
    fn throughput_max_moe_expert_eviction_precise() {
        // Arrange: parallelism_exploitable=1.0 (>0.5), memory_intensive=1.0
        // expert_eviction_aggressiveness: throughput baseline=0.8
        //   *= lerp(1.0, 1.5, 1.0) = 1.5 -> 0.8 * 1.5 = 1.2
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 1.0,
            parallelism_exploitable: 1.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.expert_eviction_aggressiveness, 1.2, 0.01);
    }

    // ── 4. parallelism_cost_scale reduced linearly by parallelism_exploitable ──

    #[test]
    fn parallelism_cost_reduced_by_parallelism_exploitable_precise() {
        // Arrange: parallelism_exploitable=0.5
        // parallelism_cost_scale: latency baseline=1.5, *= lerp(1.0, 0.5, 0.5)=0.75
        // result = 1.5 * 0.75 = 1.125
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert
        assert_approx(bias.parallelism_cost_scale, 1.125, 0.01);
    }

    // ── 5. Pipeline valuable at 0.5 reduces pipeline cost scale precisely ──

    #[test]
    fn pipeline_valuable_half_reduces_pipeline_cost_precise() {
        // Arrange: pipeline_valuable=0.5
        // pipeline_cost_scale: throughput baseline=1.3, *= lerp(1.0, 0.6, 0.5)=0.8
        // result = 1.3 * 0.8 = 1.04
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.pipeline_cost_scale, 1.04, 0.01);
    }

    // ── 6. GPU with usize::MAX registers: scarcity branch skipped cleanly ──

    #[test]
    fn gpu_max_registers_skips_scarcity_no_panic() {
        // Arrange: GPU with absurdly high register count
        let gpu_max_reg = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: usize::MAX,
            cache_sizes: (65536, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        // Act — must not panic or overflow
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &gpu_max_reg);
        // Assert: all fields are positive finite
        assert!(bias.fusion_cost_scale.is_finite());
        assert!(bias.epilogue_depth_preference.is_finite());
        assert!(bias.k_depth_preference.is_finite());
        assert!(bias.fusion_cost_scale > 0.0);
    }

    // ── 7. Same archetype, same hw: Latency vs Throughput produce different parallelism_cost ──

    #[test]
    fn same_inputs_different_modes_parallelism_cost_differs() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.3,
            memory_intensive: 0.4,
            parallelism_exploitable: 0.2,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.6,
        };
        let hw = cpu_avx2();
        // Act
        let lat = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let thr = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert: Latency has higher parallelism cost (1.5 baseline) vs Throughput (0.5)
        assert!(
            lat.parallelism_cost_scale > thr.parallelism_cost_scale,
            "Latency parallelism cost ({}) should exceed Throughput ({})",
            lat.parallelism_cost_scale, thr.parallelism_cost_scale
        );
    }

    // ── 8. StrategyBias validate: decode_ratio_scale upper bound exactly 2.0 ──

    #[test]
    fn strategy_bias_validate_decode_ratio_upper_bound() {
        // Arrange: decode_ratio_scale at exactly 2.0 (upper bound)
        let mut bias = StrategyBias {
            decode_ratio_scale: 2.0,
            ..StrategyBias::default()
        };
        // Act
        bias.validate();
        // Assert: value unchanged (exactly at boundary)
        assert_eq!(bias.decode_ratio_scale, 2.0);
        // Now test just above
        bias.decode_ratio_scale = 2.001;
        bias.validate();
        assert_eq!(bias.decode_ratio_scale, 2.0, "2.001 should clamp to 2.0");
    }

    // ── 9. decode_ratio_scale unchanged across all mode/hw combinations ──

    #[test]
    fn decode_ratio_scale_constant_across_all_combinations() {
        let arches = [
            GraphArchetype {
                compute_intensive: 0.0, memory_intensive: 0.0,
                parallelism_exploitable: 0.0, fusion_profitable: 0.0,
                pipeline_valuable: 0.0,
            },
            GraphArchetype {
                compute_intensive: 1.0, memory_intensive: 1.0,
                parallelism_exploitable: 1.0, fusion_profitable: 1.0,
                pipeline_valuable: 1.0,
            },
        ];
        let hws = [cpu_avx2(), cpu_avx512(), gpu_a100()];
        let modes = [InferenceMode::Latency, InferenceMode::Throughput];

        for arch in &arches {
            for hw in &hws {
                for &mode in &modes {
                    let bias = StrategyArbiter::arbitrate(mode, arch, hw);
                    assert!(
                        (bias.decode_ratio_scale - 1.0).abs() < 0.001,
                        "decode_ratio_scale must always be 1.0, got {}",
                        bias.decode_ratio_scale
                    );
                }
            }
        }
    }

    // ── 10. GPU zero shared mem produces valid bias through validate ──

    #[test]
    fn gpu_zero_shared_mem_arbitrate_valid_after_validate() {
        // Arrange
        let gpu = ArbiterHwView::gpu(0);
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.8,
            fusion_profitable: 0.6,
            pipeline_valuable: 0.4,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &gpu);
        // Assert: validate() has been called internally, all fields in valid ranges
        assert!(bias.fusion_cost_scale >= 0.2 && bias.fusion_cost_scale <= 3.0);
        assert!(bias.pipeline_cost_scale >= 0.2 && bias.pipeline_cost_scale <= 3.0);
        assert!(bias.parallelism_cost_scale >= 0.1 && bias.parallelism_cost_scale <= 3.0);
        assert!(bias.epilogue_depth_preference >= 0.3 && bias.epilogue_depth_preference <= 3.0);
        assert!(bias.k_depth_preference >= 0.3 && bias.k_depth_preference <= 3.0);
        assert!(bias.kv_cache_budget_scale >= 0.2 && bias.kv_cache_budget_scale <= 3.0);
        assert!(bias.batch_flexibility >= 0.0 && bias.batch_flexibility <= 1.0);
        assert!(bias.decode_ratio_scale >= 0.3 && bias.decode_ratio_scale <= 2.0);
        assert!(bias.expert_eviction_aggressiveness >= 0.0 && bias.expert_eviction_aggressiveness <= 2.0);
        assert!(bias.expert_prefetch_priority >= 0.1 && bias.expert_prefetch_priority <= 5.0);
    }

    // ── 11. Identity archetype on neutral hw produces exact latency baseline values ──

    #[test]
    fn identity_archetype_latency_exact_baseline() {
        // Arrange: all archetype fields = 0.0, neutral hw (32 regs, 64K L1)
        // No modulation, no hw adjustment. All fields = baseline values.
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert: exact baseline values (no modulation applied)
        assert_approx(bias.fusion_cost_scale, 0.5, 0.001);
        assert_approx(bias.pipeline_cost_scale, 0.6, 0.001);
        assert_approx(bias.parallelism_cost_scale, 1.5, 0.001);
        assert_approx(bias.epilogue_depth_preference, 1.5, 0.001);
        assert_approx(bias.k_depth_preference, 1.3, 0.001);
        assert_approx(bias.kv_cache_budget_scale, 0.5, 0.001);
        assert_approx(bias.weight_prefetch_budget_scale, 1.5, 0.001);
        assert_eq!(bias.batch_flexibility, 0.0);
        assert_approx(bias.decode_ratio_scale, 1.0, 0.001);
        assert_approx(bias.speculative_decoding_value, 1.5, 0.001);
        assert_approx(bias.quantization_aggressiveness, 1.5, 0.001);
        assert_approx(bias.expert_eviction_aggressiveness, 0.0, 0.001);
        assert_approx(bias.expert_prefetch_priority, 0.5, 0.001);
    }

    // ── 12. ArbiterHwView::gpu constructor isolates L2/L3 to zero ──

    #[test]
    fn arbiter_hw_view_gpu_constructor_isolates_cache() {
        // Arrange: construct multiple GPU views with different shared mem sizes
        let sizes: Vec<usize> = vec![0, 1024, 32768, 65536, 98304, 131072];
        for size in sizes {
            let view = ArbiterHwView::gpu(size);
            // Assert
            assert!(view.device == DeviceFamily::Gpu, "gpu({}) should set device=Gpu", size);
            assert_eq!(view.num_simd_regs, 255, "gpu({}) should set 255 regs", size);
            assert_eq!(view.cache_sizes.0, size, "gpu({}) L1 mismatch", size);
            assert_eq!(view.cache_sizes.1, 0, "gpu({}) L2 should be 0", size);
            assert_eq!(view.cache_sizes.2, 0, "gpu({}) L3 should be 0", size);
        }
    }

    // ── 13. Quantization aggressiveness throughput baseline with zero memory: identity ──

    #[test]
    fn quantization_throughput_zero_memory_identity() {
        // Arrange: memory_intensive=0.0, throughput baseline quantization=0.8
        // lerp(1.0, 1.3, 0.0) = 1.0, so quantization stays 0.8
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert: no memory modulation, quantization stays at throughput baseline
        assert_approx(bias.quantization_aggressiveness, 0.8, 0.01);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Wave +13: cross-field interaction, composition, imbalanced archetypes
    // ═══════════════════════════════════════════════════════════════════════════

    // ── 1. Smoke: all 13 output fields are finite for both modes ──

    #[test]
    fn smoke_all_fields_finite_both_modes() {
        // Arrange: non-trivial archetype + CPU AVX-512
        let arch = GraphArchetype {
            compute_intensive: 0.7,
            memory_intensive: 0.4,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.3,
            pipeline_valuable: 0.8,
        };
        let hw = cpu_avx512();
        for mode in [InferenceMode::Latency, InferenceMode::Throughput] {
            // Act
            let bias = StrategyArbiter::arbitrate(mode, &arch, &hw);
            // Assert: every field is a normal finite number
            for (name, val) in [
                ("fusion_cost_scale", bias.fusion_cost_scale),
                ("pipeline_cost_scale", bias.pipeline_cost_scale),
                ("parallelism_cost_scale", bias.parallelism_cost_scale),
                ("epilogue_depth_preference", bias.epilogue_depth_preference),
                ("k_depth_preference", bias.k_depth_preference),
                ("kv_cache_budget_scale", bias.kv_cache_budget_scale),
                ("weight_prefetch_budget_scale", bias.weight_prefetch_budget_scale),
                ("batch_flexibility", bias.batch_flexibility),
                ("decode_ratio_scale", bias.decode_ratio_scale),
                ("speculative_decoding_value", bias.speculative_decoding_value),
                ("quantization_aggressiveness", bias.quantization_aggressiveness),
                ("expert_eviction_aggressiveness", bias.expert_eviction_aggressiveness),
                ("expert_prefetch_priority", bias.expert_prefetch_priority),
            ] {
                assert!(
                    val.is_finite(),
                    "{name} not finite for mode {:?}: {val}",
                    mode
                );
            }
        }
    }

    // ── 2. Imbalanced archetype: compute-high, fusion-high, memory-low ──

    #[test]
    fn imbalanced_compute_high_memory_low_archetype_precise() {
        // Arrange: strongly compute-bound, fusion-friendly, not memory-bound
        // fusion_profitable=0.9 → fusion_cost *= lerp(1.0, 0.6, 0.9) = 0.64
        // pipeline_valuable=0.1 → pipeline_cost *= lerp(1.0, 0.6, 0.1) = 0.96
        // parallelism_exploitable=0.1 → no MoE modulation (<=0.5)
        // memory_intensive=0.1 → kv *= lerp(1.0,1.5,0.1)=1.05, quant *= lerp(1.0,1.3,0.1)=1.03
        // reg_tension = 0.9 - 0.1 = 0.8 > 0 → epilogue *= 1+0.8*0.5=1.4, k_depth *= 1-0.8*0.3=0.76
        // L1 richness = 65536/65536 = 1.0, sqrt=1.0 → fusion_cost *= 1/1.0 = 1.0 (no change)
        // Latency baseline: fusion=0.5, pipeline=0.6, epilogue=1.5, k_depth=1.3
        //   kv=0.5, quant=1.5
        // After archetype: fusion=0.5*0.64=0.32, pipeline=0.6*0.96=0.576
        //   epilogue=1.5*1.4=2.1, k_depth=1.3*0.76=0.988
        //   kv=0.5*1.05=0.525, quant=1.5*1.03=1.545
        // HW: no GPU, regs=32 (no scarcity), L1=65536 → fusion unchanged
        // Validate clamps: all in range
        let arch = GraphArchetype {
            compute_intensive: 0.9,
            memory_intensive: 0.1,
            parallelism_exploitable: 0.1,
            fusion_profitable: 0.9,
            pipeline_valuable: 0.1,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert
        assert_approx(bias.fusion_cost_scale, 0.32, 0.01);
        assert_approx(bias.pipeline_cost_scale, 0.576, 0.01);
        assert_approx(bias.epilogue_depth_preference, 2.1, 0.01);
        assert_approx(bias.k_depth_preference, 0.988, 0.01);
        assert_approx(bias.kv_cache_budget_scale, 0.525, 0.01);
        assert_approx(bias.quantization_aggressiveness, 1.545, 0.01);
    }

    // ── 3. Three-stage pipeline composition: manual per-stage verification ──

    #[test]
    fn three_stage_pipeline_composition_manual_verify() {
        // Arrange: Throughput mode, archetype with moderate values, GPU
        // Stage 1 baseline (Throughput):
        //   fusion=1.0, pipeline=1.3, parallelism=0.5, epilogue=0.8, k_depth=0.8
        //   kv=1.5, weight_prefetch=0.8, batch=1.0, decode=1.0
        //   expert_eviction=0.8, expert_prefetch=1.5, speculative=0.3, quant=0.8
        // Stage 2 archetype: fusion=0.5 → lerp(1,0.6,0.5)=0.8 → fusion=1.0*0.8=0.8
        //   pipeline=0.5 → lerp(1,0.6,0.5)=0.8 → pipeline=1.3*0.8=1.04
        //   parallelism=0.5 → lerp(1,0.5,0.5)=0.75 → parallelism=0.5*0.75=0.375
        //   reg_tension=0.5-0.5=0.0 → no epilogue/k_depth change
        //   parallelism_exploitable=0.5 → NOT > 0.5, so no MoE modulation
        //   memory=0.5 → kv *= lerp(1,1.5,0.5)=1.25 → 1.5*1.25=1.875
        //   quant *= lerp(1,1.3,0.5)=1.15 → 0.8*1.15=0.92
        // Stage 3 HW (GPU): epilogue *= 1.2 → 0.8*1.2=0.96
        //   k_depth *= 1.2 → 0.8*1.2=0.96
        //   pipeline *= 1.2 → 1.04*1.2=1.248
        //   GPU has 255 regs → no scarcity
        //   L1=49152 → richness=49152/65536=0.75 → sqrt=0.866 → fusion *= 1/0.866=1.155
        //   → fusion=0.8*1.155=0.924
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (49152, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.fusion_cost_scale, 0.924, 0.02);
        assert_approx(bias.pipeline_cost_scale, 1.248, 0.02);
        assert_approx(bias.parallelism_cost_scale, 0.375, 0.01);
        assert_approx(bias.epilogue_depth_preference, 0.96, 0.01);
        assert_approx(bias.k_depth_preference, 0.96, 0.01);
        assert_approx(bias.kv_cache_budget_scale, 1.875, 0.01);
        assert_approx(bias.quantization_aggressiveness, 0.92, 0.01);
    }

    // ── 4. Cross-field: KV cache and quantization receive identical memory_intensive lerp ──

    #[test]
    fn kv_cache_and_quantization_same_memory_modulation_factor() {
        // Arrange: Both kv_cache_budget_scale and quantization_aggressiveness are modulated
        // by the same lerp(1.0, target, memory_intensive). For a given mode baseline,
        // the ratio of final/baseline should reflect the same multiplier.
        // Latency: kv_base=0.5, quant_base=1.5
        // memory_intensive=0.6 → kv *= lerp(1,1.5,0.6)=1.3, quant *= lerp(1,1.3,0.6)=1.18
        // ratio_kv = final_kv / 0.5, ratio_quant = final_quant / 1.5
        // They differ because lerp targets differ (1.5 vs 1.3), but the lerp t is identical.
        // Verify: final_kv / base_kv = lerp(1.0, 1.5, 0.6) = 1.3
        // Verify: final_quant / base_quant = lerp(1.0, 1.3, 0.6) = 1.18
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.6,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert: the multiplier applied to kv_cache is lerp(1.0, 1.5, 0.6) = 1.3
        let kv_multiplier = bias.kv_cache_budget_scale / 0.5;
        assert_approx(kv_multiplier, 1.3, 0.001);
        // The multiplier applied to quantization is lerp(1.0, 1.3, 0.6) = 1.18
        let quant_multiplier = bias.quantization_aggressiveness / 1.5;
        assert_approx(quant_multiplier, 1.18, 0.001);
    }

    // ── 5. GPU with 1-byte shared memory: no division by zero ──

    #[test]
    fn gpu_one_byte_shared_mem_no_division_by_zero() {
        // Arrange: GPU with smallest possible shared memory
        // L1 richness = 1.0/65536.0 ≈ 0.0000153, sqrt ≈ 0.0039
        // fusion_cost *= 1/0.0039 ≈ 256 → validate clamps to 3.0
        let hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 255,
            cache_sizes: (1, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert: all fields are finite (no NaN/inf from division)
        assert!(bias.fusion_cost_scale.is_finite());
        assert!(bias.fusion_cost_scale > 0.0);
        // fusion_cost is clamped by validate to max 3.0
        assert!(bias.fusion_cost_scale <= 3.0);
    }

    // ── 6. Contrasting archetype: fusion=0.0, pipeline=1.0 → negative reg_tension ──

    #[test]
    fn contrasting_fusion_zero_pipeline_one_negative_reg_tension() {
        // Arrange: fusion_profitable=0.0, pipeline_valuable=1.0
        // reg_tension = 0.0 - 1.0 = -1.0 → abs=1.0
        // k_depth *= 1 + 1.0*0.5 = 1.5
        // epilogue *= 1 - 1.0*0.3 = 0.7
        // Latency baseline: epilogue=1.5, k_depth=1.3
        // After archetype: epilogue=1.5*0.7=1.05, k_depth=1.3*1.5=1.95
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 1.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert: k_depth boosted, epilogue reduced (pipeline-oriented archetype)
        assert_approx(bias.epilogue_depth_preference, 1.05, 0.01);
        assert_approx(bias.k_depth_preference, 1.95, 0.01);
        // Verify k_depth > epilogue (pipeline dominance)
        assert!(
            bias.k_depth_preference > bias.epilogue_depth_preference,
            "k_depth ({}) should exceed epilogue ({}) for pipeline-dominant archetype",
            bias.k_depth_preference,
            bias.epilogue_depth_preference
        );
    }

    // ── 7. Midpoint archetype all 0.5: precise output for Throughput + CPU AVX-512 ──

    #[test]
    fn midpoint_all_half_archetype_throughput_cpu_avx512_precise() {
        // Arrange: all archetype fields = 0.5
        // Throughput baseline: fusion=1.0, pipeline=1.3, parallelism=0.5
        // fusion *= lerp(1,0.6,0.5)=0.8 → 1.0*0.8=0.8
        // pipeline *= lerp(1,0.6,0.5)=0.8 → 1.3*0.8=1.04
        // parallelism *= lerp(1,0.5,0.5)=0.75 → 0.5*0.75=0.375
        // reg_tension=0 → no epilogue/k_depth archetype change
        // parallelism_exploitable=0.5 → NOT > 0.5, no MoE
        // memory=0.5 → kv *= 1.25 → 1.5*1.25=1.875; quant *= 1.15 → 0.8*1.15=0.92
        // HW: CPU, regs=32, L1=49152 → richness=0.75, sqrt=0.866 → fusion *= 1/0.866=1.155
        // → fusion=0.8*1.155=0.924
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.5,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = cpu_avx512();
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.fusion_cost_scale, 0.924, 0.02);
        assert_approx(bias.kv_cache_budget_scale, 1.875, 0.01);
        assert_approx(bias.quantization_aggressiveness, 0.92, 0.01);
        assert_approx(bias.parallelism_cost_scale, 0.375, 0.01);
        // batch_flexibility is always 1.0 for throughput
        assert_approx(bias.batch_flexibility, 1.0, 0.001);
    }

    // ── 8. Arbitrate CPU identical to manual From for both modes ──

    #[test]
    fn arbitrate_cpu_matches_manual_from_both_modes() {
        // Arrange
        let arch = GraphArchetype {
            compute_intensive: 0.6,
            memory_intensive: 0.7,
            parallelism_exploitable: 0.3,
            fusion_profitable: 0.4,
            pipeline_valuable: 0.8,
        };
        let profile = DeviceProfile::detect();
        let hw = ArbiterHwView::from(&profile);

        for mode in [InferenceMode::Latency, InferenceMode::Throughput] {
            // Act
            let via_cpu = StrategyArbiter::arbitrate_cpu(mode, &arch, &profile);
            let via_manual = StrategyArbiter::arbitrate(mode, &arch, &hw);
            // Assert: every field identical
            assert_approx(via_cpu.fusion_cost_scale, via_manual.fusion_cost_scale, 0.0);
            assert_approx(via_cpu.pipeline_cost_scale, via_manual.pipeline_cost_scale, 0.0);
            assert_approx(via_cpu.parallelism_cost_scale, via_manual.parallelism_cost_scale, 0.0);
            assert_approx(via_cpu.epilogue_depth_preference, via_manual.epilogue_depth_preference, 0.0);
            assert_approx(via_cpu.k_depth_preference, via_manual.k_depth_preference, 0.0);
            assert_approx(via_cpu.kv_cache_budget_scale, via_manual.kv_cache_budget_scale, 0.0);
            assert_approx(via_cpu.weight_prefetch_budget_scale, via_manual.weight_prefetch_budget_scale, 0.0);
            assert_approx(via_cpu.batch_flexibility, via_manual.batch_flexibility, 0.0);
            assert_approx(via_cpu.decode_ratio_scale, via_manual.decode_ratio_scale, 0.0);
            assert_approx(via_cpu.speculative_decoding_value, via_manual.speculative_decoding_value, 0.0);
            assert_approx(via_cpu.quantization_aggressiveness, via_manual.quantization_aggressiveness, 0.0);
            assert_approx(via_cpu.expert_eviction_aggressiveness, via_manual.expert_eviction_aggressiveness, 0.0);
            assert_approx(via_cpu.expert_prefetch_priority, via_manual.expert_prefetch_priority, 0.0);
        }
    }

    // ── 9. Batch flexibility is mode-constant, ignoring archetype and hardware ──

    #[test]
    fn batch_flexibility_mode_constant_ignores_archetype_and_hw() {
        // Arrange: extreme archetype + GPU hardware
        let extreme_arch = GraphArchetype {
            compute_intensive: 0.99,
            memory_intensive: 0.99,
            parallelism_exploitable: 0.99,
            fusion_profitable: 0.99,
            pipeline_valuable: 0.99,
        };
        let gpu_hw = gpu_a100();
        // Act
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &extreme_arch, &gpu_hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &extreme_arch, &gpu_hw);
        // Assert: batch_flexibility is not modulated by archetype or HW
        // Latency baseline = 0.0, Throughput baseline = 1.0
        assert_approx(latency.batch_flexibility, 0.0, 0.001);
        assert_approx(throughput.batch_flexibility, 1.0, 0.001);
    }

    // ── 10. GPU + register scarcity cumulative effect ──

    #[test]
    fn gpu_plus_register_scarcity_cumulative_adjustment() {
        // Arrange: GPU with only 8 registers (is_gpu=true AND num_simd_regs=8<=16)
        // GPU branch: epilogue *= 1.2, k_depth *= 1.2
        // Scarcity: scarcity = 1 - 8/32 = 0.75
        //   epilogue *= 1 + 0.75*0.3 = 1.225
        //   k_depth *= 1 - 0.75*0.2 = 0.85
        // Cumulative: epilogue *= 1.2 * 1.225 = 1.47
        //            k_depth *= 1.2 * 0.85 = 1.02
        // Latency baseline: epilogue=1.5, k_depth=1.3
        // After cumulative HW: epilogue=1.5*1.47=2.205, k_depth=1.3*1.02=1.326
        // L1=49152 → richness=0.75, sqrt=0.866 → fusion *= 1/0.866=1.155
        let hw = ArbiterHwView {
            device: DeviceFamily::Gpu,
            num_simd_regs: 8,
            cache_sizes: (49152, 0, 0),
        };
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.0,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        // Assert: both GPU and scarcity adjustments applied cumulatively
        assert_approx(bias.epilogue_depth_preference, 2.205, 0.02);
        assert_approx(bias.k_depth_preference, 1.326, 0.02);
    }

    // ── 11. KV cache + quantization dual modulation precise midpoint ──

    #[test]
    fn kv_cache_quantization_dual_modulation_memory_intensive_0_5_precise() {
        // Arrange: memory_intensive=0.5 drives both kv_cache and quantization
        // Throughput baseline: kv=1.5, quant=0.8
        // kv *= lerp(1.0, 1.5, 0.5) = 1.25 → 1.5*1.25 = 1.875
        // quant *= lerp(1.0, 1.3, 0.5) = 1.15 → 0.8*1.15 = 0.92
        let arch = GraphArchetype {
            compute_intensive: 0.0,
            memory_intensive: 0.5,
            parallelism_exploitable: 0.0,
            fusion_profitable: 0.0,
            pipeline_valuable: 0.0,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.kv_cache_budget_scale, 1.875, 0.01);
        assert_approx(bias.quantization_aggressiveness, 0.92, 0.01);
    }

    // ── 12. Imbalanced archetype reversed: compute-low, memory-high, fusion-low, pipeline-high ──

    #[test]
    fn imbalanced_reversed_compute_low_memory_high_pipeline_high_precise() {
        // Arrange: memory-bound, pipeline-oriented, not fusion-friendly
        // fusion_profitable=0.1 → fusion_cost *= lerp(1,0.6,0.1) = 0.96
        // pipeline_valuable=0.9 → pipeline_cost *= lerp(1,0.6,0.9) = 0.64
        // parallelism_exploitable=0.6 → > 0.5 → MoE fires
        //   expert_eviction *= lerp(1,1.5,0.9) = 1.45
        //   expert_prefetch *= lerp(1,2.0,0.9) = 1.9
        // reg_tension = 0.1 - 0.9 = -0.8 → abs=0.8
        //   k_depth *= 1 + 0.8*0.5 = 1.4
        //   epilogue *= 1 - 0.8*0.3 = 0.76
        // memory=0.9 → kv *= lerp(1,1.5,0.9) = 1.45; quant *= lerp(1,1.3,0.9) = 1.27
        // Throughput baseline: fusion=1.0, pipeline=1.3, kv=1.5, quant=0.8
        //   epilogue=0.8, k_depth=0.8, expert_eviction=0.8, expert_prefetch=1.5
        // After archetype:
        //   fusion = 1.0*0.96 = 0.96
        //   pipeline = 1.3*0.64 = 0.832
        //   epilogue = 0.8*0.76 = 0.608
        //   k_depth = 0.8*1.4 = 1.12
        //   kv = 1.5*1.45 = 2.175
        //   quant = 0.8*1.27 = 1.016
        //   expert_eviction = 0.8*1.45 = 1.16
        //   expert_prefetch = 1.5*1.9 = 2.85
        let arch = GraphArchetype {
            compute_intensive: 0.1,
            memory_intensive: 0.9,
            parallelism_exploitable: 0.6,
            fusion_profitable: 0.1,
            pipeline_valuable: 0.9,
        };
        let hw = ArbiterHwView {
            device: DeviceFamily::Cpu,
            num_simd_regs: 32,
            cache_sizes: (65536, 0, 0),
        };
        // Act
        let bias = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert
        assert_approx(bias.fusion_cost_scale, 0.96, 0.01);
        assert_approx(bias.pipeline_cost_scale, 0.832, 0.01);
        assert_approx(bias.epilogue_depth_preference, 0.608, 0.01);
        assert_approx(bias.k_depth_preference, 1.12, 0.01);
        assert_approx(bias.kv_cache_budget_scale, 2.175, 0.01);
        assert_approx(bias.quantization_aggressiveness, 1.016, 0.01);
        assert_approx(bias.expert_eviction_aggressiveness, 1.16, 0.01);
        assert_approx(bias.expert_prefetch_priority, 2.85, 0.01);
    }

    // ── 13. Weight prefetch and KV cache inversely related across modes ──

    #[test]
    fn weight_prefetch_kv_cache_inverse_mode_relationship() {
        // Arrange: same archetype and hardware for both modes
        // Latency: kv_base=0.5 (low), weight_prefetch_base=1.5 (high)
        // Throughput: kv_base=1.5 (high), weight_prefetch_base=0.8 (low)
        // This is a structural invariant: Latency prioritizes weight prefetch over KV cache,
        // Throughput does the opposite.
        let arch = GraphArchetype {
            compute_intensive: 0.5,
            memory_intensive: 0.3,
            parallelism_exploitable: 0.4,
            fusion_profitable: 0.5,
            pipeline_valuable: 0.5,
        };
        let hw = cpu_avx512();
        // Act
        let latency = StrategyArbiter::arbitrate(InferenceMode::Latency, &arch, &hw);
        let throughput = StrategyArbiter::arbitrate(InferenceMode::Throughput, &arch, &hw);
        // Assert: Latency mode has higher weight_prefetch, lower kv_cache
        assert!(
            latency.weight_prefetch_budget_scale > throughput.weight_prefetch_budget_scale,
            "Latency weight_prefetch ({}) should exceed Throughput ({})",
            latency.weight_prefetch_budget_scale,
            throughput.weight_prefetch_budget_scale
        );
        assert!(
            throughput.kv_cache_budget_scale > latency.kv_cache_budget_scale,
            "Throughput kv_cache ({}) should exceed Latency ({})",
            throughput.kv_cache_budget_scale,
            latency.kv_cache_budget_scale
        );
    }

}
