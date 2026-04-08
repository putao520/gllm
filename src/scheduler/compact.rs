//! Compact Decision Model (REQ-SCHED-020, SPEC/02-ARCHITECTURE.md §10.6.3)
//!
//! Implements the cost-benefit analysis for Compact→Execute→Scatter (§9.1).
//! Compact is a GEMM-level optimization that re-packs active SIMD lanes to
//! eliminate wasted compute. It must NOT be applied to memory-bound ops (Attention).

use crate::scheduler::chunked_prefill::BatchManifest;

/// Configuration for the compact decision model.
#[derive(Debug, Clone)]
pub struct CompactConfig {
    /// Waste ratio threshold to trigger compact (SPEC: 0.25 = 25%)
    pub waste_threshold: f32,
    /// Minimum active elements to justify compact overhead
    pub min_active_count: usize,
    /// Estimated cycles per element for compact+scatter (hardware-specific)
    pub cycles_per_element: f32,
    /// Peak FLOPS / peak memory bandwidth ratio (hardware-specific)
    /// Higher = compute-bound → compact saves more FLOPS relative to memory cost
    pub flops_to_mem_ratio: f32,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self {
            waste_threshold: 0.25,
            min_active_count: 4,
            cycles_per_element: 2.0, // ~2 cache line accesses per element (compact + scatter)
            flops_to_mem_ratio: 20.0, // Typical for modern GPUs (compute >> memory bandwidth)
        }
    }
}

/// Result of a compact decision evaluation.
#[derive(Debug, Clone)]
pub struct CompactDecision {
    /// Whether compact should be triggered
    pub should_compact: bool,
    /// Computed waste ratio [0.0, 1.0]
    pub waste_ratio: f32,
    /// Number of active elements
    pub active_count: usize,
    /// Total elements (batch size)
    pub total_count: usize,
    /// Why the decision was made (for logging/debugging)
    pub reason: CompactReason,
}

/// Reason for the compact decision.
#[derive(Debug, Clone, PartialEq)]
pub enum CompactReason {
    /// Compact triggered: waste exceeds threshold and cost-benefit is positive
    Triggered { saved_flops_ratio: f32, cost_ratio: f32 },
    /// Not triggered: waste below threshold
    BelowThreshold { waste_ratio: f32, threshold: f32 },
    /// Not triggered: too few active elements
    TooFewActive { active: usize, min: usize },
    /// Not triggered: compact cost exceeds FLOPS savings
    CostExceedsBenefit { cost_ratio: f32, saved_ratio: f32 },
    /// Not triggered: empty batch
    EmptyBatch,
}

/// Op-level compact eligibility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpKind {
    /// GEMM (QKV, FFN, lm_head) — compute-bound, compact eligible
    Gemm,
    /// Attention (QK^T, softmax, AV) — memory-bound, compact NOT eligible
    Attention,
    /// Normalization (RmsNorm, LayerNorm) — memory-bound, compact NOT eligible
    Norm,
    /// Elementwise (SiLU, residual add) — memory-bound, compact NOT eligible
    Elementwise,
}

impl OpKind {
    /// Whether compact is eligible for this op type.
    ///
    /// SPEC §10.6.3: "禁止在 Attention op 上做 compact
    /// (attention 是 memory-bound，compact 无法节省 memory bandwidth，
    /// 反而增加数据搬移)"
    pub fn is_compact_eligible(&self) -> bool {
        matches!(self, OpKind::Gemm)
    }

    /// Whether this op is compute-bound (compact can save FLOPS).
    pub fn is_compute_bound(&self) -> bool {
        matches!(self, OpKind::Gemm)
    }
}

/// Evaluate whether compact should be triggered for a given manifest and op.
///
/// This is the core decision function per SPEC §10.6.3:
/// 1. waste_ratio > threshold (25%)
/// 2. active_count >= min_compact_threshold (4)
/// 3. op is compute-bound (GEMM only)
/// 4. compact_cost < saved_flops × flops_to_mem_ratio
pub fn evaluate_compact(
    manifest: &BatchManifest,
    op: OpKind,
    config: &CompactConfig,
) -> CompactDecision {
    let total = manifest.slots.len();

    if total == 0 {
        return CompactDecision {
            should_compact: false,
            waste_ratio: 0.0,
            active_count: 0,
            total_count: 0,
            reason: CompactReason::EmptyBatch,
        };
    }

    // Count active elements (slots with non-zero tokens)
    let active = manifest.slots.iter().filter(|s| s.token_end > s.token_start).count();
    let waste_ratio = if total > 0 {
        (total - active) as f32 / total as f32
    } else {
        0.0
    };

    // Guard 1: Too few active elements
    if active < config.min_active_count {
        return CompactDecision {
            should_compact: false,
            waste_ratio,
            active_count: active,
            total_count: total,
            reason: CompactReason::TooFewActive {
                active,
                min: config.min_active_count,
            },
        };
    }

    // Guard 2: Waste below threshold
    if waste_ratio <= config.waste_threshold {
        return CompactDecision {
            should_compact: false,
            waste_ratio,
            active_count: active,
            total_count: total,
            reason: CompactReason::BelowThreshold {
                waste_ratio,
                threshold: config.waste_threshold,
            },
        };
    }

    // Guard 3: Op must be compute-bound (GEMM)
    if !op.is_compact_eligible() {
        return CompactDecision {
            should_compact: false,
            waste_ratio,
            active_count: active,
            total_count: total,
            reason: CompactReason::CostExceedsBenefit {
                cost_ratio: f32::INFINITY,
                saved_ratio: 0.0,
            },
        };
    }

    // Cost-benefit analysis
    // compact_cost = 2 × active × cycles_per_element (compact + scatter)
    // saved_flops = waste_ratio × total_flops
    // decision = compact_cost_cycles < saved_flops_cycles × flops_to_mem_ratio
    let cost_ratio = config.cycles_per_element * 2.0; // Per-element cost relative to compute
    let saved_flops_ratio = waste_ratio; // Fraction of FLOPS saved

    if cost_ratio < saved_flops_ratio * config.flops_to_mem_ratio {
        CompactDecision {
            should_compact: true,
            waste_ratio,
            active_count: active,
            total_count: total,
            reason: CompactReason::Triggered {
                saved_flops_ratio,
                cost_ratio,
            },
        }
    } else {
        CompactDecision {
            should_compact: false,
            waste_ratio,
            active_count: active,
            total_count: total,
            reason: CompactReason::CostExceedsBenefit {
                cost_ratio,
                saved_ratio: saved_flops_ratio,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::chunked_prefill::{BatchSlot, BatchManifest, SlotType};

    fn make_manifest(total: usize, active: usize) -> BatchManifest {
        let mut slots = Vec::new();
        for i in 0..active {
            slots.push(BatchSlot {
                request_id: i as u64,
                slot_type: SlotType::Decode,
                token_start: i,
                token_end: i + 1,
                compact_target: i as i32,
            });
        }
        for i in active..total {
            slots.push(BatchSlot {
                request_id: i as u64,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 0,
                compact_target: -1,
            });
        }
        let waste = if total > 0 { (total - active) as f32 / total as f32 } else { 0.0 };
        BatchManifest {
            slots,
            total_tokens: total,
            decode_tokens: active,
            prefill_tokens: 0,
            compact_required: waste > 0.25,
            waste_ratio: waste,
        }
    }

    #[test]
    fn test_compact_triggered_high_waste_gemm() {
        let manifest = make_manifest(8, 5); // 37.5% waste
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::Triggered { .. }));
    }

    #[test]
    fn test_compact_not_triggered_low_waste() {
        let manifest = make_manifest(8, 7); // 12.5% waste
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }));
    }

    #[test]
    fn test_compact_not_triggered_attention() {
        let manifest = make_manifest(8, 5); // 37.5% waste but attention op
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Attention, &config);
        assert!(!decision.should_compact);
    }

    #[test]
    fn test_compact_not_triggered_too_few_active() {
        let manifest = make_manifest(8, 2); // Only 2 active
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::TooFewActive { .. }));
    }

    #[test]
    fn test_compact_empty_batch() {
        let manifest = BatchManifest {
            slots: Vec::new(),
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::EmptyBatch));
    }
}
