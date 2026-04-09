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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceMode {
    /// Extreme single-request latency: batch=1, all resources serve one request.
    Latency,
    /// Extreme throughput: maximize tokens/second/dollar.
    Throughput,
}

impl Default for InferenceMode {
    fn default() -> Self {
        Self::Latency
    }
}

// ── Hardware view for the arbiter ───────────────────────────────────────────

/// Minimal hardware view consumed by the arbiter.
///
/// `DeviceProfile` from gllm-kernels is CPU-only (no GPU variant). This thin
/// view lets callers describe both CPU and GPU targets with the three
/// properties the arbiter actually reads.
#[derive(Debug, Clone, Copy)]
pub struct ArbiterHwView {
    pub is_gpu: bool,
    pub num_simd_regs: usize,
    /// (L1, L2, L3) in bytes. For GPUs, L1 maps to shared memory.
    pub cache_sizes: (usize, usize, usize),
}

impl From<&DeviceProfile> for ArbiterHwView {
    fn from(p: &DeviceProfile) -> Self {
        Self {
            is_gpu: false,
            num_simd_regs: p.num_simd_regs(),
            cache_sizes: p.cache_sizes(),
        }
    }
}

impl ArbiterHwView {
    /// Construct a GPU hardware view with the given shared-memory size.
    pub fn gpu(shared_mem_bytes: usize) -> Self {
        Self {
            is_gpu: true,
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
        if hw.is_gpu {
            bias.epilogue_depth_preference *= 1.2;
            bias.k_depth_preference *= 1.2;
            bias.pipeline_cost_scale *= 1.2;
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

#[inline]
#[allow(dead_code)]
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
            is_gpu: false,
            num_simd_regs: 16,
            cache_sizes: (32768, 262144, 8388608),
        }
    }

    fn cpu_avx512() -> ArbiterHwView {
        ArbiterHwView {
            is_gpu: false,
            num_simd_regs: 32,
            cache_sizes: (49152, 1048576, 33554432),
        }
    }

    fn gpu_a100() -> ArbiterHwView {
        ArbiterHwView {
            is_gpu: true,
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
}
