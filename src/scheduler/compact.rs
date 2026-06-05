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

    // ---- Additional tests ----

    #[test]
    fn compact_config_default_values() {
        let c = CompactConfig::default();
        assert!((c.waste_threshold - 0.25).abs() < 1e-5);
        assert_eq!(c.min_active_count, 4);
        assert!((c.cycles_per_element - 2.0).abs() < 1e-5);
        assert!((c.flops_to_mem_ratio - 20.0).abs() < 1e-5);
    }

    #[test]
    fn compact_config_clone() {
        let c = CompactConfig::default();
        let c2 = c.clone();
        assert!((c2.waste_threshold - c.waste_threshold).abs() < 1e-5);
    }

    #[test]
    fn op_kind_is_compact_eligible() {
        assert!(OpKind::Gemm.is_compact_eligible());
        assert!(!OpKind::Attention.is_compact_eligible());
        assert!(!OpKind::Norm.is_compact_eligible());
        assert!(!OpKind::Elementwise.is_compact_eligible());
    }

    #[test]
    fn op_kind_is_compute_bound() {
        assert!(OpKind::Gemm.is_compute_bound());
        assert!(!OpKind::Attention.is_compute_bound());
        assert!(!OpKind::Norm.is_compute_bound());
        assert!(!OpKind::Elementwise.is_compute_bound());
    }

    #[test]
    fn op_kind_equality() {
        assert_eq!(OpKind::Gemm, OpKind::Gemm);
        assert_ne!(OpKind::Gemm, OpKind::Attention);
    }

    #[test]
    fn op_kind_copy_clone() {
        let op = OpKind::Gemm;
        let op2 = op;
        assert_eq!(op, op2);
        let op3 = op.clone();
        assert_eq!(op3, OpKind::Gemm);
    }

    #[test]
    fn compact_reason_equality() {
        assert_eq!(
            CompactReason::EmptyBatch,
            CompactReason::EmptyBatch,
        );
        assert_ne!(
            CompactReason::EmptyBatch,
            CompactReason::BelowThreshold { waste_ratio: 0.0, threshold: 0.25 },
        );
    }

    #[test]
    fn compact_decision_clone() {
        let d = CompactDecision {
            should_compact: true,
            waste_ratio: 0.5,
            active_count: 4,
            total_count: 8,
            reason: CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 },
        };
        let d2 = d.clone();
        assert!(d2.should_compact);
        assert_eq!(d2.active_count, 4);
    }

    #[test]
    fn compact_not_triggered_norm_op() {
        let manifest = make_manifest(8, 5);
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Norm, &config);
        assert!(!decision.should_compact);
    }

    #[test]
    fn compact_not_triggered_elementwise_op() {
        let manifest = make_manifest(8, 5);
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Elementwise, &config);
        assert!(!decision.should_compact);
    }

    #[test]
    fn compact_all_active_no_waste() {
        let manifest = make_manifest(8, 8); // 0% waste
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
        assert!((decision.waste_ratio).abs() < 1e-5);
    }

    #[test]
    fn compact_exactly_at_threshold() {
        // 25% waste exactly at threshold → should NOT trigger (<= check)
        let manifest = make_manifest(8, 6); // 25% waste
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
    }

    #[test]
    fn compact_just_above_threshold() {
        // 26% waste just above threshold
        // Need 0.26 * 20 = 5.2 > 4.0 cost → should trigger
        let manifest = make_manifest(100, 74); // 26% waste
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(decision.should_compact);
    }

    #[test]
    fn compact_cost_exceeds_benefit() {
        // High cycles_per_element makes compact too expensive
        let manifest = make_manifest(8, 5); // 37.5% waste
        let config = CompactConfig {
            cycles_per_element: 100.0, // very expensive
            ..CompactConfig::default()
        };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    #[test]
    fn compact_low_flops_to_mem_ratio() {
        let manifest = make_manifest(8, 5); // 37.5% waste
        let config = CompactConfig {
            flops_to_mem_ratio: 1.0, // low compute advantage
            ..CompactConfig::default()
        };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // cost = 4.0, saved = 0.375 * 1.0 = 0.375 → cost > saved
        assert!(!decision.should_compact);
    }

    #[test]
    fn compact_decision_fields_accuracy() {
        let manifest = make_manifest(10, 6); // 40% waste
        let config = CompactConfig {
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert_eq!(decision.total_count, 10);
        assert_eq!(decision.active_count, 6);
        assert!((decision.waste_ratio - 0.4).abs() < 1e-5);
    }

    #[test]
    fn compact_min_active_count_zero_allowed() {
        let manifest = make_manifest(4, 1); // 75% waste but only 1 active
        let config = CompactConfig {
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(decision.should_compact);
    }

    // ---- Additional 17 tests ----

    // -- CompactConfig construction and field access --

    #[test]
    fn compact_config_manual_construction() {
        let config = CompactConfig {
            waste_threshold: 0.5,
            min_active_count: 10,
            cycles_per_element: 3.0,
            flops_to_mem_ratio: 50.0,
        };
        assert!((config.waste_threshold - 0.5).abs() < 1e-5);
        assert_eq!(config.min_active_count, 10);
        assert!((config.cycles_per_element - 3.0).abs() < 1e-5);
        assert!((config.flops_to_mem_ratio - 50.0).abs() < 1e-5);
    }

    #[test]
    fn compact_config_zero_threshold() {
        let config = CompactConfig {
            waste_threshold: 0.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(4, 2); // 50% waste, threshold 0%
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // waste_ratio (0.5) > threshold (0.0) and cost-benefit passes
        assert!(decision.should_compact);
    }

    #[test]
    fn compact_config_zero_flops_to_mem_ratio() {
        let config = CompactConfig {
            flops_to_mem_ratio: 0.0,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 4); // 50% waste
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // cost_ratio (4.0) < saved_ratio (0.5) * 0.0 = 0.0 → cost > benefit
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    #[test]
    fn compact_config_max_threshold() {
        // threshold = 1.0 means only 100% waste triggers (all inactive)
        let config = CompactConfig {
            waste_threshold: 1.0,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 4); // 50% waste, but threshold is 100%
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }));
    }

    #[test]
    fn compact_config_clone_independence() {
        let mut config = CompactConfig::default();
        let cloned = config.clone();
        config.waste_threshold = 0.99;
        // Cloned value should retain original
        assert!((cloned.waste_threshold - 0.25).abs() < 1e-5);
        assert!((config.waste_threshold - 0.99).abs() < 1e-5);
    }

    // -- CompactReason variants and traits --

    #[test]
    fn compact_reason_triggered_variant_fields() {
        let reason = CompactReason::Triggered {
            saved_flops_ratio: 0.375,
            cost_ratio: 4.0,
        };
        if let CompactReason::Triggered { saved_flops_ratio, cost_ratio } = reason {
            assert!((saved_flops_ratio - 0.375).abs() < 1e-5);
            assert!((cost_ratio - 4.0).abs() < 1e-5);
        } else {
            panic!("Expected Triggered variant");
        }
    }

    #[test]
    fn compact_reason_below_threshold_variant_fields() {
        let reason = CompactReason::BelowThreshold {
            waste_ratio: 0.2,
            threshold: 0.25,
        };
        if let CompactReason::BelowThreshold { waste_ratio, threshold } = reason {
            assert!((waste_ratio - 0.2).abs() < 1e-5);
            assert!((threshold - 0.25).abs() < 1e-5);
        } else {
            panic!("Expected BelowThreshold variant");
        }
    }

    #[test]
    fn compact_reason_too_few_active_variant_fields() {
        let reason = CompactReason::TooFewActive { active: 2, min: 4 };
        if let CompactReason::TooFewActive { active, min } = reason {
            assert_eq!(active, 2);
            assert_eq!(min, 4);
        } else {
            panic!("Expected TooFewActive variant");
        }
    }

    #[test]
    fn compact_reason_cost_exceeds_benefit_variant_fields() {
        let reason = CompactReason::CostExceedsBenefit {
            cost_ratio: 8.0,
            saved_ratio: 0.3,
        };
        if let CompactReason::CostExceedsBenefit { cost_ratio, saved_ratio } = reason {
            assert!((cost_ratio - 8.0).abs() < 1e-5);
            assert!((saved_ratio - 0.3).abs() < 1e-5);
        } else {
            panic!("Expected CostExceedsBenefit variant");
        }
    }

    #[test]
    fn compact_reason_clone_preserves_variant() {
        let reasons = vec![
            CompactReason::EmptyBatch,
            CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 },
            CompactReason::BelowThreshold { waste_ratio: 0.1, threshold: 0.25 },
            CompactReason::TooFewActive { active: 1, min: 4 },
            CompactReason::CostExceedsBenefit { cost_ratio: 10.0, saved_ratio: 0.2 },
        ];
        for reason in &reasons {
            let cloned = reason.clone();
            assert_eq!(*reason, cloned);
        }
    }

    // -- OpKind variants exhaustive coverage --

    #[test]
    fn op_kind_all_variants_copy() {
        let variants = [OpKind::Gemm, OpKind::Attention, OpKind::Norm, OpKind::Elementwise];
        // Copy: moving does not invalidate original
        for v in &variants {
            let copied = *v;
            assert_eq!(*v, copied);
        }
    }

    #[test]
    fn op_kind_all_variants_clone() {
        let variants = [OpKind::Gemm, OpKind::Attention, OpKind::Norm, OpKind::Elementwise];
        for v in &variants {
            let cloned = v.clone();
            assert_eq!(*v, cloned);
        }
    }

    #[test]
    fn op_kind_all_variants_inequality() {
        let variants = [OpKind::Gemm, OpKind::Attention, OpKind::Norm, OpKind::Elementwise];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "{:?} should not equal {:?}", a, b);
                }
            }
        }
    }

    // -- Debug trait verification --

    #[test]
    fn compact_config_debug_output() {
        let config = CompactConfig {
            waste_threshold: 0.25,
            min_active_count: 4,
            cycles_per_element: 2.0,
            flops_to_mem_ratio: 20.0,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("CompactConfig"));
        assert!(debug_str.contains("waste_threshold"));
        assert!(debug_str.contains("min_active_count"));
    }

    #[test]
    fn compact_decision_debug_output() {
        let decision = CompactDecision {
            should_compact: true,
            waste_ratio: 0.5,
            active_count: 4,
            total_count: 8,
            reason: CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 },
        };
        let debug_str = format!("{:?}", decision);
        assert!(debug_str.contains("CompactDecision"));
        assert!(debug_str.contains("should_compact"));
        assert!(debug_str.contains("waste_ratio"));
    }

    #[test]
    fn compact_reason_debug_output_all_variants() {
        let empty = format!("{:?}", CompactReason::EmptyBatch);
        assert!(empty.contains("EmptyBatch"));

        let triggered = format!("{:?}", CompactReason::Triggered {
            saved_flops_ratio: 0.5, cost_ratio: 4.0
        });
        assert!(triggered.contains("Triggered"));

        let below = format!("{:?}", CompactReason::BelowThreshold {
            waste_ratio: 0.1, threshold: 0.25
        });
        assert!(below.contains("BelowThreshold"));

        let few = format!("{:?}", CompactReason::TooFewActive { active: 1, min: 4 });
        assert!(few.contains("TooFewActive"));

        let cost = format!("{:?}", CompactReason::CostExceedsBenefit {
            cost_ratio: 10.0, saved_ratio: 0.2
        });
        assert!(cost.contains("CostExceedsBenefit"));
    }

    #[test]
    fn op_kind_debug_output_all_variants() {
        assert!(format!("{:?}", OpKind::Gemm).contains("Gemm"));
        assert!(format!("{:?}", OpKind::Attention).contains("Attention"));
        assert!(format!("{:?}", OpKind::Norm).contains("Norm"));
        assert!(format!("{:?}", OpKind::Elementwise).contains("Elementwise"));
    }

    // ============================================================
    // 30 additional tests — covering untested behaviors & edge cases
    // ============================================================

    // -- Manifest with mixed SlotType (PrefillChunk + Decode) --

    #[test]
    fn compact_manifest_with_prefill_chunk_slots() {
        // Arrange: mix of Decode and PrefillChunk slots, enough active to pass min threshold
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 5, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 3, compact_target: 3 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 6, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 7, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 10, decode_tokens: 2, prefill_tokens: 8,
            compact_required: true, waste_ratio: 0.5,
        };
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 4 active out of 8 = 50% waste, slot type doesn't matter
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 4);
        assert_eq!(decision.total_count, 8);
    }

    #[test]
    fn compact_manifest_all_prefill_chunks_high_waste() {
        // Arrange: all slots are PrefillChunk, enough active to pass min threshold
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 10, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 10, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 10, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 10, compact_target: 3 },
            BatchSlot { request_id: 4, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 6, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 7, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 40, decode_tokens: 0, prefill_tokens: 40,
            compact_required: true, waste_ratio: 0.5,
        };
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 4 active out of 8, slot type does not affect evaluate_compact logic
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 4);
    }

    // -- Single-slot manifest --

    #[test]
    fn compact_single_slot_active() {
        // Arrange: 1 active slot, 0% waste, but 1 < min_active_count(4)
        let manifest = make_manifest(1, 1);
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 1 active < min_active_count(4) → TooFewActive (checked before waste)
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::TooFewActive { .. }));
    }

    #[test]
    fn compact_single_slot_inactive() {
        // Arrange: 1 slot, inactive (token_start == token_end)
        let manifest = make_manifest(1, 0);
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 0 active < min_active_count(4) → TooFewActive
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::TooFewActive { .. }));
    }

    // -- Zero-active scenarios --

    #[test]
    fn compact_all_slots_inactive() {
        // Arrange: 8 slots, 0 active → 100% waste but 0 active
        let manifest = make_manifest(8, 0);
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: TooFewActive guard fires before waste check
        assert!(!decision.should_compact);
        assert_eq!(decision.active_count, 0);
        assert!((decision.waste_ratio - 1.0).abs() < 1e-5);
    }

    #[test]
    fn compact_all_inactive_min_active_zero() {
        // Arrange: 8 slots, 0 active, min_active_count = 0
        let manifest = make_manifest(8, 0);
        let config = CompactConfig { min_active_count: 0, ..CompactConfig::default() };

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: min_active_count=0 bypasses TooFewActive, but 100% waste > threshold
        // cost = 4.0, saved = 1.0 * 20.0 = 20.0 → triggers
        assert!(decision.should_compact);
        assert!((decision.waste_ratio - 1.0).abs() < 1e-5);
    }

    // -- Large batch --

    #[test]
    fn compact_large_batch_high_waste() {
        // Arrange: 256 slots, 64 active → 75% waste
        let manifest = make_manifest(256, 64);
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 75% waste, GEMM → should trigger
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 64);
        assert_eq!(decision.total_count, 256);
    }

    #[test]
    fn compact_large_batch_low_waste() {
        // Arrange: 256 slots, 250 active → ~2.3% waste
        let manifest = make_manifest(256, 250);
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: low waste → BelowThreshold
        assert!(!decision.should_compact);
        assert!(decision.waste_ratio < 0.05);
    }

    // -- Varying token ranges (not just 0 or 1) --

    #[test]
    fn compact_slots_with_varying_token_ranges() {
        // Arrange: active determined by token_end > token_start, not token count
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 100, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 50, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 5, token_end: 5, compact_target: -1 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 150, decode_tokens: 100, prefill_tokens: 50,
            compact_required: true, waste_ratio: 0.667,
        };
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 2 active out of 6, but 2 < min_active_count(4)
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::TooFewActive { .. }));
        assert_eq!(decision.active_count, 2);
    }

    // -- Exact min_active_count boundary --

    #[test]
    fn compact_exactly_at_min_active_count() {
        // Arrange: 4 active = exactly min_active_count, but high waste
        let manifest = make_manifest(16, 4); // 75% waste
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: active(4) >= min(4) passes, 75% waste > 25% threshold passes
        assert!(decision.should_compact);
    }

    #[test]
    fn compact_one_below_min_active_count() {
        // Arrange: 3 active = just below min_active_count(4), high waste
        let manifest = make_manifest(16, 3); // 81.25% waste
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: TooFewActive despite very high waste
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::TooFewActive { active: 3, min: 4 }));
    }

    // -- Cost-benefit boundary --

    #[test]
    fn compact_cost_benefit_exact_boundary() {
        // Arrange: cost = cycles*2 = 4.0, saved*waste*ratio needs to be > 4.0
        // waste=0.375, ratio=20.0 → saved_ratio*20.0 = 7.5 > 4.0 → triggers
        // But with cycles_per_element = 10.0 → cost = 20.0, 0.375*20.0=7.5 < 20.0
        let manifest = make_manifest(8, 5); // 37.5% waste
        let config = CompactConfig {
            cycles_per_element: 10.0,
            flops_to_mem_ratio: 20.0,
            ..CompactConfig::default()
        };

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: cost (20.0) > saved (7.5) → CostExceedsBenefit
        assert!(!decision.should_compact);
        if let CompactReason::CostExceedsBenefit { cost_ratio, saved_ratio } = decision.reason {
            assert!((cost_ratio - 20.0).abs() < 1e-3);
            assert!((saved_ratio - 0.375).abs() < 1e-3);
        } else {
            panic!("Expected CostExceedsBenefit, got {:?}", decision.reason);
        }
    }

    #[test]
    fn compact_cost_benefit_just_passes() {
        // Arrange: waste=0.5, ratio=10.0, cycles=2.0
        // cost = 4.0, saved = 0.5 * 10.0 = 5.0 → 4.0 < 5.0 → triggers
        let manifest = make_manifest(8, 4); // 50% waste
        let config = CompactConfig {
            cycles_per_element: 2.0,
            flops_to_mem_ratio: 10.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert
        assert!(decision.should_compact);
    }

    // -- Non-GEMM ops: verify infinite cost for ineligible ops --

    #[test]
    fn compact_attention_infinite_cost() {
        // Arrange: Attention op → ineligible → CostExceedsBenefit with infinity
        let manifest = make_manifest(8, 5); // high waste
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Attention, &config);

        // Assert: cost_ratio = infinity
        if let CompactReason::CostExceedsBenefit { cost_ratio, saved_ratio } = decision.reason {
            assert!(cost_ratio.is_infinite() && cost_ratio.is_sign_positive());
            assert!((saved_ratio - 0.0).abs() < 1e-5);
        } else {
            panic!("Expected CostExceedsBenefit for Attention, got {:?}", decision.reason);
        }
    }

    #[test]
    fn compact_norm_infinite_cost() {
        let manifest = make_manifest(8, 5);
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Norm, &config);
        if let CompactReason::CostExceedsBenefit { cost_ratio, .. } = decision.reason {
            assert!(cost_ratio.is_infinite());
        } else {
            panic!("Expected CostExceedsBenefit for Norm");
        }
    }

    #[test]
    fn compact_elementwise_infinite_cost() {
        let manifest = make_manifest(8, 5);
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Elementwise, &config);
        if let CompactReason::CostExceedsBenefit { cost_ratio, .. } = decision.reason {
            assert!(cost_ratio.is_infinite());
        } else {
            panic!("Expected CostExceedsBenefit for Elementwise");
        }
    }

    // -- Triggered decision: verify saved_flops_ratio == waste_ratio --

    #[test]
    fn compact_triggered_saved_equals_waste() {
        // Arrange
        let manifest = make_manifest(10, 6); // 40% waste
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: saved_flops_ratio == waste_ratio
        if let CompactReason::Triggered { saved_flops_ratio, .. } = decision.reason {
            assert!((saved_flops_ratio - decision.waste_ratio).abs() < 1e-5);
        } else {
            panic!("Expected Triggered");
        }
    }

    // -- Custom config combinations --

    #[test]
    fn compact_custom_low_threshold_high_ratio() {
        // Arrange: threshold=0.1, flops_to_mem_ratio=100 → easy to trigger
        let config = CompactConfig {
            waste_threshold: 0.1,
            flops_to_mem_ratio: 100.0,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 6); // 25% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 25% > 10% threshold, cost=4.0, saved=0.25*100=25.0 → triggers
        assert!(decision.should_compact);
    }

    #[test]
    fn compact_custom_high_threshold_normal_ratio() {
        // Arrange: threshold=0.9, 8 active out of 10 → 20% waste, passes min_active
        let config = CompactConfig {
            waste_threshold: 0.9,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(10, 8); // 20% waste, 8 >= min_active

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 20% < 90% threshold → BelowThreshold
        assert!(!decision.should_compact);
        if let CompactReason::BelowThreshold { waste_ratio, threshold } = decision.reason {
            assert!((waste_ratio - 0.2).abs() < 1e-5);
            assert!((threshold - 0.9).abs() < 1e-5);
        } else {
            panic!("Expected BelowThreshold");
        }
    }

    #[test]
    fn compact_custom_very_high_cycles_per_element() {
        // Arrange: extreme cycles cost
        let config = CompactConfig {
            cycles_per_element: 10000.0,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(100, 10); // 90% waste, 10 active >= 4

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: cost = 20000, saved = 0.9 * 20 = 18 → cost >> saved
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    // -- Guard ordering verification: TooFewActive fires before BelowThreshold --

    #[test]
    fn compact_guard_order_few_active_before_threshold() {
        // Arrange: both TooFewActive and BelowThreshold conditions met
        let manifest = make_manifest(100, 2); // 98% waste, only 2 active
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: TooFewActive checked first
        assert!(matches!(decision.reason, CompactReason::TooFewActive { .. }));
    }

    // -- Guard ordering: BelowThreshold fires before compute-bound check --

    #[test]
    fn compact_guard_order_threshold_before_eligibility() {
        // Arrange: BelowThreshold but with Attention (would also reject)
        let manifest = make_manifest(8, 7); // 12.5% waste < 25%
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Attention, &config);

        // Assert: BelowThreshold fires before compute-bound check
        assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }));
    }

    // -- Verify decision fields consistency for all non-triggered reasons --

    #[test]
    fn compact_decision_waste_ratio_matches_computation() {
        // Arrange: 3 active out of 12
        let manifest = make_manifest(12, 3);
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: waste = (12-3)/12 = 0.75
        let expected_waste = (12 - 3) as f32 / 12.0f32;
        assert!((decision.waste_ratio - expected_waste).abs() < 1e-5);
        assert_eq!(decision.total_count, 12);
        assert_eq!(decision.active_count, 3);
    }

    // -- CompactDecision fields always populated for every reason variant --

    #[test]
    fn compact_decision_fields_populated_for_triggered() {
        let manifest = make_manifest(8, 4);
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };
        let d = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(d.should_compact);
        assert_eq!(d.active_count, 4);
        assert_eq!(d.total_count, 8);
        assert!(d.waste_ratio > 0.0);
    }

    #[test]
    fn compact_decision_fields_populated_for_empty() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0, decode_tokens: 0, prefill_tokens: 0,
            compact_required: false, waste_ratio: 0.0,
        };
        let d = evaluate_compact(&manifest, OpKind::Gemm, &CompactConfig::default());
        assert!(!d.should_compact);
        assert_eq!(d.active_count, 0);
        assert_eq!(d.total_count, 0);
        assert!((d.waste_ratio).abs() < 1e-5);
    }

    // -- Direct CompactDecision construction with each reason variant --

    #[test]
    fn compact_decision_construct_triggered() {
        let decision = CompactDecision {
            should_compact: true,
            waste_ratio: 0.6,
            active_count: 4,
            total_count: 10,
            reason: CompactReason::Triggered { saved_flops_ratio: 0.6, cost_ratio: 4.0 },
        };
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 4);
        assert_eq!(decision.total_count, 10);
    }

    #[test]
    fn compact_decision_construct_below_threshold() {
        let decision = CompactDecision {
            should_compact: false,
            waste_ratio: 0.1,
            active_count: 7,
            total_count: 8,
            reason: CompactReason::BelowThreshold { waste_ratio: 0.1, threshold: 0.25 },
        };
        assert!(!decision.should_compact);
    }

    #[test]
    fn compact_decision_construct_too_few_active() {
        let decision = CompactDecision {
            should_compact: false,
            waste_ratio: 0.5,
            active_count: 2,
            total_count: 4,
            reason: CompactReason::TooFewActive { active: 2, min: 4 },
        };
        assert!(!decision.should_compact);
        assert_eq!(decision.active_count, 2);
    }

    #[test]
    fn compact_decision_construct_empty_batch() {
        let decision = CompactDecision {
            should_compact: false,
            waste_ratio: 0.0,
            active_count: 0,
            total_count: 0,
            reason: CompactReason::EmptyBatch,
        };
        assert!(!decision.should_compact);
        assert_eq!(decision.total_count, 0);
    }

    // -- Verify make_manifest helper correctness --

    #[test]
    fn helper_make_manifest_active_slots_have_nonzero_range() {
        let manifest = make_manifest(8, 5);
        let active_slots: Vec<_> = manifest.slots.iter().filter(|s| s.token_end > s.token_start).collect();
        assert_eq!(active_slots.len(), 5);
        for slot in &active_slots {
            assert!(slot.token_end > slot.token_start);
        }
    }

    #[test]
    fn helper_make_manifest_inactive_slots_have_zero_range() {
        let manifest = make_manifest(8, 5);
        let inactive_slots: Vec<_> = manifest.slots.iter().filter(|s| s.token_end <= s.token_start).collect();
        assert_eq!(inactive_slots.len(), 3);
        for slot in &inactive_slots {
            assert_eq!(slot.token_end, slot.token_start);
        }
    }

    // -- Slots with zero-length but non-zero start (token_start == token_end > 0) --

    #[test]
    fn compact_slots_zero_length_nonzero_start_counted_inactive() {
        // Arrange: slot with token_start=5, token_end=5 → zero-length → inactive
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 5, token_end: 5, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 5, token_end: 5, compact_target: -1 },
            BatchSlot { request_id: 6, slot_type: SlotType::Decode, token_start: 5, token_end: 5, compact_target: -1 },
            BatchSlot { request_id: 7, slot_type: SlotType::Decode, token_start: 5, token_end: 5, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 4, decode_tokens: 4, prefill_tokens: 0,
            compact_required: true, waste_ratio: 0.5,
        };
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 4 active out of 8, 50% waste → should trigger
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 4);
    }

    // -- evaluate_compact is pure: same input → same output --

    #[test]
    fn compact_pure_function_same_input_same_output() {
        let manifest = make_manifest(10, 6);
        let config = CompactConfig::default();

        let d1 = evaluate_compact(&manifest, OpKind::Gemm, &config);
        let d2 = evaluate_compact(&manifest, OpKind::Gemm, &config);

        assert_eq!(d1.should_compact, d2.should_compact);
        assert!((d1.waste_ratio - d2.waste_ratio).abs() < 1e-10);
        assert_eq!(d1.active_count, d2.active_count);
        assert_eq!(d1.total_count, d2.total_count);
        assert_eq!(d1.reason, d2.reason);
    }

    // -- CompactConfig with very low cycles_per_element --

    #[test]
    fn compact_very_low_cycles_per_element() {
        // Arrange: cycles=0.01 → cost=0.02, almost always triggers
        let config = CompactConfig {
            cycles_per_element: 0.01,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(100, 99); // 1% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 1% > 0.25% is false → BelowThreshold (waste must exceed threshold)
        // Actually 0.01 > 0.25 is false → BelowThreshold
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }));
    }

    #[test]
    fn compact_very_low_cycles_with_high_waste() {
        // Arrange: cycles=0.01, waste=50% → cost=0.02, saved=0.5*20=10 → triggers
        let config = CompactConfig {
            cycles_per_element: 0.01,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(10, 5);

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert
        assert!(decision.should_compact);
    }

    // -- Verify compact_target field is not used by evaluate_compact --

    #[test]
    fn compact_ignores_compact_target_field() {
        // Arrange: all compact_target = 999 (non-standard), but active by token range
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 999 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 999 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 999 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 999 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: 999 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: 999 },
            BatchSlot { request_id: 6, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: 999 },
            BatchSlot { request_id: 7, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: 999 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 4, decode_tokens: 4, prefill_tokens: 0,
            compact_required: true, waste_ratio: 0.5,
        };
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: evaluate_compact uses token_end > token_start, not compact_target
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 4);
    }

    // -- BatchManifest fields (compact_required, waste_ratio) are informational --

    #[test]
    fn compact_manifest_fields_not_used_by_evaluate() {
        // Arrange: manifest says compact_required=false, low waste_ratio,
        // but actual slot data says 50% waste
        let manifest = BatchManifest {
            slots: vec![
                BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
                BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
                BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
                BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
                BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
                BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
                BatchSlot { request_id: 6, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
                BatchSlot { request_id: 7, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            ],
            total_tokens: 4,
            decode_tokens: 4,
            prefill_tokens: 0,
            compact_required: false, // misleading
            waste_ratio: 0.0,        // misleading
        };
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: evaluate_compact computes waste from slots, not from manifest.waste_ratio
        assert!(decision.should_compact);
        assert!((decision.waste_ratio - 0.5).abs() < 1e-5);
    }

    // -- Exhaustive OpKind × (trigger/not-trigger) matrix --

    #[test]
    fn compact_all_op_kinds_with_minimal_active() {
        let manifest = make_manifest(8, 4);
        let config = CompactConfig { min_active_count: 4, ..CompactConfig::default() };

        // GEMM should be the only op that can trigger
        let gemm_decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        let attn_decision = evaluate_compact(&manifest, OpKind::Attention, &config);
        let norm_decision = evaluate_compact(&manifest, OpKind::Norm, &config);
        let elem_decision = evaluate_compact(&manifest, OpKind::Elementwise, &config);

        assert!(gemm_decision.should_compact, "GEMM should trigger with 50% waste");
        assert!(!attn_decision.should_compact, "Attention should never trigger");
        assert!(!norm_decision.should_compact, "Norm should never trigger");
        assert!(!elem_decision.should_compact, "Elementwise should never trigger");
    }

    // -- Triggered reason cost_ratio matches 2*cycles_per_element --

    #[test]
    fn compact_triggered_cost_ratio_formula() {
        let config = CompactConfig {
            cycles_per_element: 3.5,
            min_active_count: 1,
            flops_to_mem_ratio: 100.0, // ensure trigger
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 4); // 50% waste

        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        if let CompactReason::Triggered { cost_ratio, .. } = decision.reason {
            assert!((cost_ratio - 7.0).abs() < 1e-5, "cost_ratio should be 2 * 3.5 = 7.0");
        } else {
            panic!("Expected Triggered");
        }
    }

    // -- BelowThreshold reason carries correct threshold from config --

    #[test]
    fn compact_below_threshold_reason_carries_config_threshold() {
        let config = CompactConfig {
            waste_threshold: 0.42,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(10, 8); // 20% waste < 42%

        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        if let CompactReason::BelowThreshold { waste_ratio, threshold } = decision.reason {
            assert!((waste_ratio - 0.2).abs() < 1e-5);
            assert!((threshold - 0.42).abs() < 1e-5);
        } else {
            panic!("Expected BelowThreshold");
        }
    }

    // -- TooFewActive reason carries correct min from config --

    #[test]
    fn compact_too_few_active_reason_carries_config_min() {
        let config = CompactConfig {
            min_active_count: 7,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(10, 6); // 6 active < 7 min

        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        if let CompactReason::TooFewActive { active, min } = decision.reason {
            assert_eq!(active, 6);
            assert_eq!(min, 7);
        } else {
            panic!("Expected TooFewActive");
        }
    }

    // ============================================================
    // 55 additional tests — edge cases, property checks, coverage
    // ============================================================

    // -- Waste ratio computation for various sizes --

    #[test]
    fn compact_waste_ratio_two_slots_one_active() {
        let manifest = make_manifest(2, 1);
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // 1 active out of 2 → 50% waste
        assert!((decision.waste_ratio - 0.5).abs() < 1e-5);
        assert_eq!(decision.active_count, 1);
        assert_eq!(decision.total_count, 2);
    }

    #[test]
    fn compact_waste_ratio_three_slots_two_active() {
        let manifest = make_manifest(3, 2);
        let config = CompactConfig { min_active_count: 2, ..CompactConfig::default() };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // 2 active out of 3 → 33.3% waste
        let expected = 1.0 / 3.0;
        assert!((decision.waste_ratio - expected).abs() < 1e-5);
    }

    #[test]
    fn compact_waste_ratio_uneven_large_batch() {
        // 100 slots, 33 active → 67% waste
        let manifest = make_manifest(100, 33);
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        let expected = 67.0 / 100.0;
        assert!((decision.waste_ratio - expected).abs() < 1e-5);
        assert_eq!(decision.active_count, 33);
    }

    // -- CompactDecision clone preserves all fields for every reason --

    #[test]
    fn compact_decision_clone_triggered() {
        let d = CompactDecision {
            should_compact: true,
            waste_ratio: 0.6,
            active_count: 4,
            total_count: 10,
            reason: CompactReason::Triggered { saved_flops_ratio: 0.6, cost_ratio: 4.0 },
        };
        let d2 = d.clone();
        assert_eq!(d2.should_compact, d.should_compact);
        assert_eq!(d2.active_count, d.active_count);
        assert_eq!(d2.total_count, d.total_count);
        assert!((d2.waste_ratio - d.waste_ratio).abs() < 1e-5);
        assert_eq!(d2.reason, d.reason);
    }

    #[test]
    fn compact_decision_clone_below_threshold() {
        let d = CompactDecision {
            should_compact: false,
            waste_ratio: 0.1,
            active_count: 7,
            total_count: 8,
            reason: CompactReason::BelowThreshold { waste_ratio: 0.1, threshold: 0.25 },
        };
        let d2 = d.clone();
        assert!(!d2.should_compact);
        assert_eq!(d2.reason, d.reason);
    }

    #[test]
    fn compact_decision_clone_cost_exceeds() {
        let d = CompactDecision {
            should_compact: false,
            waste_ratio: 0.4,
            active_count: 6,
            total_count: 10,
            reason: CompactReason::CostExceedsBenefit { cost_ratio: 100.0, saved_ratio: 0.4 },
        };
        let d2 = d.clone();
        assert_eq!(d2.reason, d.reason);
    }

    // -- CompactReason equality for all variant combinations --

    #[test]
    fn compact_reason_triggered_equality_same_fields() {
        let r1 = CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 };
        let r2 = CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 };
        assert_eq!(r1, r2);
    }

    #[test]
    fn compact_reason_triggered_inequality_different_fields() {
        let r1 = CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 };
        let r2 = CompactReason::Triggered { saved_flops_ratio: 0.6, cost_ratio: 4.0 };
        assert_ne!(r1, r2);
    }

    #[test]
    fn compact_reason_below_threshold_equality() {
        let r1 = CompactReason::BelowThreshold { waste_ratio: 0.2, threshold: 0.25 };
        let r2 = CompactReason::BelowThreshold { waste_ratio: 0.2, threshold: 0.25 };
        assert_eq!(r1, r2);
    }

    #[test]
    fn compact_reason_below_threshold_inequality() {
        let r1 = CompactReason::BelowThreshold { waste_ratio: 0.2, threshold: 0.25 };
        let r2 = CompactReason::BelowThreshold { waste_ratio: 0.3, threshold: 0.25 };
        assert_ne!(r1, r2);
    }

    #[test]
    fn compact_reason_too_few_active_equality() {
        let r1 = CompactReason::TooFewActive { active: 3, min: 4 };
        let r2 = CompactReason::TooFewActive { active: 3, min: 4 };
        assert_eq!(r1, r2);
    }

    #[test]
    fn compact_reason_cost_exceeds_equality() {
        let r1 = CompactReason::CostExceedsBenefit { cost_ratio: 8.0, saved_ratio: 0.3 };
        let r2 = CompactReason::CostExceedsBenefit { cost_ratio: 8.0, saved_ratio: 0.3 };
        assert_eq!(r1, r2);
    }

    #[test]
    fn compact_reason_cross_variant_inequality() {
        let triggered = CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 };
        let below = CompactReason::BelowThreshold { waste_ratio: 0.5, threshold: 0.25 };
        let few = CompactReason::TooFewActive { active: 4, min: 4 };
        let cost = CompactReason::CostExceedsBenefit { cost_ratio: 4.0, saved_ratio: 0.5 };
        let empty = CompactReason::EmptyBatch;
        // All different variants are unequal even with same numeric values
        assert_ne!(triggered, below);
        assert_ne!(triggered, few);
        assert_ne!(triggered, cost);
        assert_ne!(triggered, empty);
        assert_ne!(below, few);
        assert_ne!(below, cost);
        assert_ne!(below, empty);
        assert_ne!(few, cost);
        assert_ne!(few, empty);
        assert_ne!(cost, empty);
    }

    // -- Non-GEMM ops with all-active batch --

    #[test]
    fn compact_attention_all_active_no_waste() {
        let manifest = make_manifest(8, 8);
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Attention, &config);
        assert!(!decision.should_compact);
        assert!(decision.waste_ratio.abs() < 1e-5);
    }

    #[test]
    fn compact_norm_high_waste_still_rejected() {
        // Even 90% waste, Norm should reject
        let manifest = make_manifest(10, 1);
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };
        let decision = evaluate_compact(&manifest, OpKind::Norm, &config);
        assert!(!decision.should_compact);
    }

    #[test]
    fn compact_elementwise_high_waste_still_rejected() {
        let manifest = make_manifest(10, 1);
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };
        let decision = evaluate_compact(&manifest, OpKind::Elementwise, &config);
        assert!(!decision.should_compact);
    }

    // -- Empty batch with non-GEMM ops --

    #[test]
    fn compact_empty_batch_attention() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let decision = evaluate_compact(&manifest, OpKind::Attention, &CompactConfig::default());
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::EmptyBatch));
    }

    #[test]
    fn compact_empty_batch_norm() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let decision = evaluate_compact(&manifest, OpKind::Norm, &CompactConfig::default());
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::EmptyBatch));
    }

    #[test]
    fn compact_empty_batch_elementwise() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let decision = evaluate_compact(&manifest, OpKind::Elementwise, &CompactConfig::default());
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::EmptyBatch));
    }

    // -- Slots with token_start > 0 but token_end > token_start (active) --

    #[test]
    fn compact_slots_nonzero_start_active() {
        // Arrange: slots where token_start > 0 but still active
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 10, token_end: 20, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 5, token_end: 15, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 100, token_end: 200, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 120,
            decode_tokens: 120,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.25,
        };
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // 3 active out of 4 = 25% waste, <= threshold → BelowThreshold
        assert!(!decision.should_compact);
        assert_eq!(decision.active_count, 3);
    }

    // -- Slots with equal start/end (zero range) are inactive --

    #[test]
    fn compact_slots_all_zero_range_all_inactive() {
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 5, token_end: 5, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 10, token_end: 10, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 100, token_end: 100, compact_target: 3 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 1.0,
        };
        let config = CompactConfig { min_active_count: 0, ..CompactConfig::default() };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert_eq!(decision.active_count, 0);
        assert!((decision.waste_ratio - 1.0).abs() < 1e-5);
    }

    // -- Large request_id values don't affect decision --

    #[test]
    fn compact_large_request_ids_ignored() {
        let slots = vec![
            BatchSlot { request_id: u64::MAX, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            BatchSlot { request_id: 999999, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 4,
            decode_tokens: 4,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.2,
        };
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // 4 active out of 5 = 20% waste, <= 25% threshold
        assert!(!decision.should_compact);
        assert_eq!(decision.active_count, 4);
    }

    // -- Sequential calls are independent (no state mutation) --

    #[test]
    fn compact_sequential_calls_independent() {
        let config = CompactConfig::default();
        // First call with high waste
        let manifest1 = make_manifest(8, 4);
        let d1 = evaluate_compact(&manifest1, OpKind::Gemm, &config);

        // Second call with low waste
        let manifest2 = make_manifest(8, 8);
        let d2 = evaluate_compact(&manifest2, OpKind::Gemm, &config);

        assert!(d1.should_compact);
        assert!(!d2.should_compact);

        // Third call: same as first → same result
        let d3 = evaluate_compact(&manifest1, OpKind::Gemm, &config);
        assert_eq!(d3.should_compact, d1.should_compact);
        assert!((d3.waste_ratio - d1.waste_ratio).abs() < 1e-10);
    }

    // -- CompactConfig with very high flops_to_mem_ratio makes trigger easy --

    #[test]
    fn compact_high_flops_to_mem_ratio_triggers_easily() {
        let config = CompactConfig {
            flops_to_mem_ratio: 10000.0,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 5); // 37.5% waste
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // cost = 4.0, saved = 0.375 * 10000 = 3750 → triggers
        assert!(decision.should_compact);
    }

    // -- Very small batch (2 slots) with 1 active --

    #[test]
    fn compact_two_slots_one_active() {
        let manifest = make_manifest(2, 1);
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // 50% waste, 1 active >= 1 min
        // cost = 4.0, saved = 0.5 * 20.0 = 10.0 → triggers
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 1);
        assert_eq!(decision.total_count, 2);
    }

    // -- Batch where active == total (no waste) with all op kinds --

    #[test]
    fn compact_no_waste_attention() {
        let manifest = make_manifest(4, 4);
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Attention, &config);
        assert!(!decision.should_compact);
        assert!(decision.waste_ratio.abs() < 1e-5);
    }

    #[test]
    fn compact_no_waste_norm() {
        let manifest = make_manifest(4, 4);
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Norm, &config);
        assert!(!decision.should_compact);
    }

    #[test]
    fn compact_no_waste_elementwise() {
        let manifest = make_manifest(4, 4);
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Elementwise, &config);
        assert!(!decision.should_compact);
    }

    // -- cost_ratio is always 2 * cycles_per_element (verify formula) --

    #[test]
    fn compact_cost_ratio_equals_double_cycles_when_triggered() {
        for cycles in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let config = CompactConfig {
                cycles_per_element: cycles,
                min_active_count: 1,
                flops_to_mem_ratio: 10000.0, // ensure trigger
                ..CompactConfig::default()
            };
            let manifest = make_manifest(8, 4); // 50% waste
            let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
            if let CompactReason::Triggered { cost_ratio, .. } = decision.reason {
                assert!(
                    (cost_ratio - 2.0 * cycles).abs() < 1e-3,
                    "cost_ratio {} != 2 * cycles {}",
                    cost_ratio,
                    cycles
                );
            } else {
                panic!("Expected Triggered for cycles={}", cycles);
            }
        }
    }

    // -- Waste ratio is always in [0, 1] range --

    #[test]
    fn compact_waste_ratio_always_in_range() {
        let config = CompactConfig { min_active_count: 0, ..CompactConfig::default() };
        for active in 0..=10usize {
            let manifest = make_manifest(10, active);
            let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
            assert!(
                decision.waste_ratio >= 0.0 && decision.waste_ratio <= 1.0,
                "waste_ratio {} out of [0,1] for active={}",
                decision.waste_ratio,
                active,
            );
        }
    }

    // -- active_count <= total_count always --

    #[test]
    fn compact_active_never_exceeds_total() {
        let config = CompactConfig::default();
        for total in 1..=20usize {
            for active in 0..=total {
                let manifest = make_manifest(total, active);
                let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
                assert!(
                    decision.active_count <= decision.total_count,
                    "active {} > total {}",
                    decision.active_count,
                    decision.total_count,
                );
            }
        }
    }

    // -- BelowThreshold with waste exactly 0 --

    #[test]
    fn compact_zero_waste_below_threshold() {
        let manifest = make_manifest(8, 8); // 0% waste
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // 0% <= 25% → BelowThreshold
        assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }));
        if let CompactReason::BelowThreshold { waste_ratio, .. } = decision.reason {
            assert!(waste_ratio.abs() < 1e-5);
        }
    }

    // -- Triggered decision: verify should_compact consistency --

    #[test]
    fn compact_triggered_implies_positive_waste() {
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };
        // Any triggered decision must have waste_ratio > threshold
        for total in [4, 8, 16, 32, 64] {
            for active in 1..total {
                let manifest = make_manifest(total, active);
                let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
                if decision.should_compact {
                    assert!(
                        decision.waste_ratio > config.waste_threshold,
                        "Triggered but waste {} <= threshold {}",
                        decision.waste_ratio,
                        config.waste_threshold,
                    );
                }
            }
        }
    }

    // -- CompactReason Debug output contains variant names --

    #[test]
    fn compact_reason_empty_batch_debug_name() {
        let s = format!("{:?}", CompactReason::EmptyBatch);
        assert_eq!(s, "EmptyBatch");
    }

    #[test]
    fn compact_reason_too_few_active_debug_contains_fields() {
        let s = format!("{:?}", CompactReason::TooFewActive { active: 3, min: 5 });
        assert!(s.contains("active"));
        assert!(s.contains("min"));
    }

    // -- OpKind: is_compute_bound mirrors is_compact_eligible --

    #[test]
    fn op_kind_compute_bound_matches_eligible() {
        for op in [OpKind::Gemm, OpKind::Attention, OpKind::Norm, OpKind::Elementwise] {
            assert_eq!(
                op.is_compute_bound(),
                op.is_compact_eligible(),
                "Mismatch for {:?}",
                op,
            );
        }
    }

    // -- CompactDecision with CostExceedsBenefit from real evaluate_compact --

    #[test]
    fn compact_cost_exceeds_carries_correct_saved_ratio() {
        // Arrange: 50% waste, but cost > benefit
        let config = CompactConfig {
            cycles_per_element: 100.0, // cost = 200
            flops_to_mem_ratio: 1.0,   // saved = 0.5
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 4);
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        if let CompactReason::CostExceedsBenefit { cost_ratio, saved_ratio } = decision.reason {
            assert!((cost_ratio - 200.0).abs() < 1e-3);
            assert!((saved_ratio - 0.5).abs() < 1e-3);
        } else {
            panic!("Expected CostExceedsBenefit");
        }
    }

    // -- BatchManifest compact_required does not affect evaluate_compact --

    #[test]
    fn compact_manifest_compact_required_true_but_low_waste() {
        // Arrange: compact_required=true but actual waste from slots is low
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 4 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 5 },
            BatchSlot { request_id: 6, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 6 },
            BatchSlot { request_id: 7, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 7 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 8,
            decode_tokens: 8,
            prefill_tokens: 0,
            compact_required: true, // misleading
            waste_ratio: 0.99,     // misleading
        };
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // All 8 slots active → 0% actual waste → BelowThreshold
        assert!(!decision.should_compact);
        assert!(decision.waste_ratio.abs() < 1e-5);
    }

    // -- Config with threshold just below actual waste --

    #[test]
    fn compact_threshold_just_below_waste() {
        // 3/8 = 37.5% waste. threshold = 37.4% → just passes
        let config = CompactConfig {
            waste_threshold: 0.374,
            min_active_count: 1,
            flops_to_mem_ratio: 20.0,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 5); // 37.5% waste
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(decision.should_compact);
    }

    // -- Config with threshold just above actual waste --

    #[test]
    fn compact_threshold_just_above_waste() {
        // 3/8 = 37.5% waste. threshold = 37.6% → just fails
        let config = CompactConfig {
            waste_threshold: 0.376,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 5);
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }));
    }

    // -- Guard ordering: empty batch fires before all other guards --

    #[test]
    fn compact_guard_order_empty_batch_first() {
        let config = CompactConfig {
            waste_threshold: 0.0,
            min_active_count: 0,
            ..CompactConfig::default()
        };
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // EmptyBatch should fire even though all guards are relaxed
        assert!(matches!(decision.reason, CompactReason::EmptyBatch));
    }

    // -- Guard ordering: compute-bound check fires before cost-benefit --

    #[test]
    fn compact_guard_order_compute_bound_before_cost_benefit() {
        // Arrange: enough active, high waste, but Attention (not compute-bound)
        let manifest = make_manifest(8, 4); // 50% waste
        let config = CompactConfig {
            min_active_count: 1,
            waste_threshold: 0.0,
            ..CompactConfig::default()
        };
        let decision = evaluate_compact(&manifest, OpKind::Attention, &config);
        // Should get CostExceedsBenefit (from compute-bound check), not a cost-benefit analysis
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    // -- verify all 4 OpKind variants produce distinct Debug output --

    #[test]
    fn op_kind_debug_outputs_are_distinct() {
        let outputs: Vec<String> = [
            format!("{:?}", OpKind::Gemm),
            format!("{:?}", OpKind::Attention),
            format!("{:?}", OpKind::Norm),
            format!("{:?}", OpKind::Elementwise),
        ].to_vec();
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                assert_ne!(outputs[i], outputs[j], "Debug output collision: {} vs {}", outputs[i], outputs[j]);
            }
        }
    }

    // -- CompactConfig: default is Clone-consistent --

    #[test]
    fn compact_config_default_clone_consistency() {
        let d1 = CompactConfig::default();
        let d2 = CompactConfig::default();
        assert!((d1.waste_threshold - d2.waste_threshold).abs() < 1e-10);
        assert_eq!(d1.min_active_count, d2.min_active_count);
        assert!((d1.cycles_per_element - d2.cycles_per_element).abs() < 1e-10);
        assert!((d1.flops_to_mem_ratio - d2.flops_to_mem_ratio).abs() < 1e-10);
    }

    // -- evaluate_compact with extreme large batch sizes --

    #[test]
    fn compact_very_large_batch_mostly_active() {
        let manifest = make_manifest(1024, 1020); // ~0.4% waste
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
        assert!(decision.waste_ratio < 0.01);
    }

    #[test]
    fn compact_very_large_batch_mostly_inactive() {
        let manifest = make_manifest(1024, 10); // ~99% waste
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(decision.should_compact);
        assert!(decision.waste_ratio > 0.98);
    }

    // -- Make helper: compact_target for inactive slots is -1 --

    #[test]
    fn helper_make_manifest_inactive_compact_target() {
        let manifest = make_manifest(8, 5);
        let inactive: Vec<_> = manifest.slots.iter().filter(|s| s.token_end <= s.token_start).collect();
        assert_eq!(inactive.len(), 3);
        for slot in &inactive {
            assert_eq!(slot.compact_target, -1);
        }
    }

    // -- Make helper: request_ids are sequential --

    #[test]
    fn helper_make_manifest_request_ids_sequential() {
        let manifest = make_manifest(5, 3);
        for (i, slot) in manifest.slots.iter().enumerate() {
            assert_eq!(slot.request_id, i as u64);
        }
    }

    // -- Make helper: slot_type is always Decode --

    #[test]
    fn helper_make_manifest_slot_type_decode() {
        let manifest = make_manifest(6, 4);
        for slot in &manifest.slots {
            assert_eq!(slot.slot_type, SlotType::Decode);
        }
    }

    // -- Make helper: active slots compact_target matches index --

    #[test]
    fn helper_make_manifest_active_compact_target() {
        let manifest = make_manifest(8, 5);
        let active: Vec<_> = manifest.slots.iter().filter(|s| s.token_end > s.token_start).collect();
        for (i, slot) in active.iter().enumerate() {
            assert_eq!(slot.compact_target, i as i32);
        }
    }

    // -- CompactDecision total_count == manifest.slots.len() always --

    #[test]
    fn compact_total_count_equals_manifest_slots_len() {
        let config = CompactConfig::default();
        for size in [1, 2, 4, 8, 16, 32] {
            for active in 0..=size {
                let manifest = make_manifest(size, active);
                let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
                assert_eq!(decision.total_count, manifest.slots.len());
            }
        }
    }

    // -- BatchManifest with prefill_tokens > 0 but all active --

    #[test]
    fn compact_manifest_with_prefill_tokens_all_active() {
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 128, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 256, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 64, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 32, compact_target: 3 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 480,
            decode_tokens: 0,
            prefill_tokens: 480,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let config = CompactConfig::default();
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!(!decision.should_compact);
        assert!(decision.waste_ratio.abs() < 1e-5);
        assert_eq!(decision.active_count, 4);
    }

    // -- CompactConfig with min_active_count larger than batch size --

    #[test]
    fn compact_min_active_larger_than_batch() {
        let manifest = make_manifest(4, 4); // all active
        let config = CompactConfig {
            min_active_count: 100,
            ..CompactConfig::default()
        };
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // 4 active < 100 min → TooFewActive
        assert!(matches!(decision.reason, CompactReason::TooFewActive { active: 4, min: 100 }));
    }

    // -- CompactConfig fields are accessible pub fields --

    #[test]
    fn compact_config_fields_publicly_accessible() {
        let mut config = CompactConfig::default();
        config.waste_threshold = 0.1;
        config.min_active_count = 2;
        config.cycles_per_element = 5.0;
        config.flops_to_mem_ratio = 30.0;
        assert!((config.waste_threshold - 0.1).abs() < 1e-5);
        assert_eq!(config.min_active_count, 2);
        assert!((config.cycles_per_element - 5.0).abs() < 1e-5);
        assert!((config.flops_to_mem_ratio - 30.0).abs() < 1e-5);
    }

    // -- CompactDecision fields are accessible pub fields --

    #[test]
    fn compact_decision_fields_publicly_accessible() {
        let mut decision = CompactDecision {
            should_compact: false,
            waste_ratio: 0.0,
            active_count: 0,
            total_count: 0,
            reason: CompactReason::EmptyBatch,
        };
        decision.should_compact = true;
        decision.waste_ratio = 0.75;
        decision.active_count = 10;
        decision.total_count = 40;
        decision.reason = CompactReason::Triggered { saved_flops_ratio: 0.75, cost_ratio: 4.0 };
        assert!(decision.should_compact);
        assert!((decision.waste_ratio - 0.75).abs() < 1e-5);
        assert_eq!(decision.active_count, 10);
        assert_eq!(decision.total_count, 40);
    }

    // -- BatchSlot fields are publicly accessible --

    #[test]
    fn batch_slot_fields_publicly_accessible() {
        let slot = BatchSlot {
            request_id: 42,
            slot_type: SlotType::Decode,
            token_start: 10,
            token_end: 20,
            compact_target: 5,
        };
        assert_eq!(slot.request_id, 42);
        assert_eq!(slot.slot_type, SlotType::Decode);
        assert_eq!(slot.token_start, 10);
        assert_eq!(slot.token_end, 20);
        assert_eq!(slot.compact_target, 5);
    }

    // -- SlotType variant checks --

    #[test]
    fn slot_type_decode_and_prefill_chunk_distinct() {
        assert_ne!(SlotType::Decode, SlotType::PrefillChunk);
    }

    #[test]
    fn slot_type_debug_output() {
        assert!(format!("{:?}", SlotType::Decode).contains("Decode"));
        assert!(format!("{:?}", SlotType::PrefillChunk).contains("PrefillChunk"));
    }

    // -- CompactDecision clone independence --

    #[test]
    fn compact_decision_clone_independence() {
        let mut d = CompactDecision {
            should_compact: true,
            waste_ratio: 0.5,
            active_count: 4,
            total_count: 8,
            reason: CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 },
        };
        let d2 = d.clone();
        d.should_compact = false;
        d.waste_ratio = 0.0;
        assert!(d2.should_compact);
        assert!((d2.waste_ratio - 0.5).abs() < 1e-5);
    }

    // -- evaluate_compact does not modify input manifest --

    #[test]
    fn compact_does_not_modify_manifest() {
        let manifest = make_manifest(8, 4);
        let slots_len_before = manifest.slots.len();
        let total_tokens_before = manifest.total_tokens;
        let decode_tokens_before = manifest.decode_tokens;
        let _ = evaluate_compact(&manifest, OpKind::Gemm, &CompactConfig::default());
        assert_eq!(manifest.slots.len(), slots_len_before);
        assert_eq!(manifest.total_tokens, total_tokens_before);
        assert_eq!(manifest.decode_tokens, decode_tokens_before);
    }

    // -- evaluate_compact does not modify input config --

    #[test]
    fn compact_does_not_modify_config() {
        let config = CompactConfig {
            waste_threshold: 0.33,
            min_active_count: 7,
            cycles_per_element: 1.5,
            flops_to_mem_ratio: 25.0,
        };
        let manifest = make_manifest(8, 4);
        let _ = evaluate_compact(&manifest, OpKind::Gemm, &config);
        assert!((config.waste_threshold - 0.33).abs() < 1e-5);
        assert_eq!(config.min_active_count, 7);
        assert!((config.cycles_per_element - 1.5).abs() < 1e-5);
        assert!((config.flops_to_mem_ratio - 25.0).abs() < 1e-5);
    }

    // ============================================================
    // 12 additional tests — uncovered gaps
    // ============================================================

    // -- Direct CompactDecision construction with CostExceedsBenefit (missing variant) --

    #[test]
    fn compact_decision_construct_cost_exceeds_benefit() {
        let decision = CompactDecision {
            should_compact: false,
            waste_ratio: 0.4,
            active_count: 6,
            total_count: 10,
            reason: CompactReason::CostExceedsBenefit { cost_ratio: 200.0, saved_ratio: 0.4 },
        };
        assert!(!decision.should_compact);
        assert_eq!(decision.active_count, 6);
        assert_eq!(decision.total_count, 10);
        if let CompactReason::CostExceedsBenefit { cost_ratio, saved_ratio } = decision.reason {
            assert!((cost_ratio - 200.0).abs() < 1e-5);
            assert!((saved_ratio - 0.4).abs() < 1e-5);
        } else {
            panic!("Expected CostExceedsBenefit");
        }
    }

    // -- CompactReason CostExceedsBenefit inequality (missing from existing equality tests) --

    #[test]
    fn compact_reason_cost_exceeds_benefit_inequality() {
        let r1 = CompactReason::CostExceedsBenefit { cost_ratio: 8.0, saved_ratio: 0.3 };
        let r2 = CompactReason::CostExceedsBenefit { cost_ratio: 8.0, saved_ratio: 0.5 };
        let r3 = CompactReason::CostExceedsBenefit { cost_ratio: 9.0, saved_ratio: 0.3 };
        assert_ne!(r1, r2, "Different saved_ratio should be unequal");
        assert_ne!(r1, r3, "Different cost_ratio should be unequal");
    }

    // -- CompactReason TooFewActive inequality (missing from existing equality tests) --

    #[test]
    fn compact_reason_too_few_active_inequality() {
        let r1 = CompactReason::TooFewActive { active: 3, min: 4 };
        let r2 = CompactReason::TooFewActive { active: 2, min: 4 };
        let r3 = CompactReason::TooFewActive { active: 3, min: 5 };
        assert_ne!(r1, r2, "Different active should be unequal");
        assert_ne!(r1, r3, "Different min should be unequal");
    }

    // -- CompactConfig with all fields set to zero --

    #[test]
    fn compact_config_all_fields_zero() {
        let config = CompactConfig {
            waste_threshold: 0.0,
            min_active_count: 0,
            cycles_per_element: 0.0,
            flops_to_mem_ratio: 0.0,
        };
        assert!((config.waste_threshold).abs() < 1e-10);
        assert_eq!(config.min_active_count, 0);
        assert!((config.cycles_per_element).abs() < 1e-10);
        assert!((config.flops_to_mem_ratio).abs() < 1e-10);
    }

    // -- CompactConfig Debug output contains all 4 field names --

    #[test]
    fn compact_config_debug_contains_all_field_names() {
        let debug = format!("{:?}", CompactConfig::default());
        assert!(debug.contains("waste_threshold"), "missing waste_threshold");
        assert!(debug.contains("min_active_count"), "missing min_active_count");
        assert!(debug.contains("cycles_per_element"), "missing cycles_per_element");
        assert!(debug.contains("flops_to_mem_ratio"), "missing flops_to_mem_ratio");
    }

    // -- BatchSlot with PrefillChunk type field access --

    #[test]
    fn batch_slot_prefill_chunk_type_access() {
        let slot = BatchSlot {
            request_id: 99,
            slot_type: SlotType::PrefillChunk,
            token_start: 0,
            token_end: 512,
            compact_target: 7,
        };
        assert_eq!(slot.request_id, 99);
        assert_eq!(slot.slot_type, SlotType::PrefillChunk);
        assert_eq!(slot.token_start, 0);
        assert_eq!(slot.token_end, 512);
        assert_eq!(slot.compact_target, 7);
    }

    // -- CompactDecision allows structurally inconsistent should_compact + reason --

    #[test]
    fn compact_decision_inconsistent_should_compact_false_with_triggered() {
        let decision = CompactDecision {
            should_compact: false,
            waste_ratio: 0.5,
            active_count: 4,
            total_count: 8,
            reason: CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 },
        };
        // Struct allows it; should_compact and reason are independent fields
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::Triggered { .. }));
    }

    // -- Negative waste_threshold: any positive waste exceeds it --

    // -- Negative cycles_per_element: cost becomes negative, always < saved --

    // -- Negative flops_to_mem_ratio: saved becomes negative, cost > saved --

    #[test]
    fn compact_negative_flops_to_mem_ratio_never_triggers() {
        let config = CompactConfig {
            flops_to_mem_ratio: -10.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 4); // 50% waste
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // cost = 4.0, saved = 0.5 * (-10.0) = -5.0 → 4.0 < -5.0 is false
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    // -- All 4 OpKind variants produce valid decisions with non-empty manifest --

    #[test]
    fn compact_all_ops_produce_valid_decision() {
        let manifest = make_manifest(8, 5);
        let config = CompactConfig::default();
        let ops = [OpKind::Gemm, OpKind::Attention, OpKind::Norm, OpKind::Elementwise];
        for op in ops {
            let d = evaluate_compact(&manifest, op, &config);
            assert_eq!(d.total_count, 8);
            assert_eq!(d.active_count, 5);
            assert!(d.waste_ratio >= 0.0 && d.waste_ratio <= 1.0);
        }
    }

    // -- BatchManifest fields are publicly accessible --

    #[test]
    fn batch_manifest_fields_publicly_accessible() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 42,
            decode_tokens: 30,
            prefill_tokens: 12,
            compact_required: true,
            waste_ratio: 0.7,
        };
        assert!(manifest.slots.is_empty());
        assert_eq!(manifest.total_tokens, 42);
        assert_eq!(manifest.decode_tokens, 30);
        assert_eq!(manifest.prefill_tokens, 12);
        assert!(manifest.compact_required);
        assert!((manifest.waste_ratio - 0.7).abs() < 1e-5);
    }

    // ============================================================
    // 13 additional tests — remaining uncovered gaps
    // ============================================================

    // -- Negative waste_threshold: any positive waste exceeds it --

    #[test]
    fn compact_negative_threshold_any_waste_exceeds() {
        // Arrange: negative threshold means even tiny waste triggers threshold check
        let config = CompactConfig {
            waste_threshold: -0.1,
            min_active_count: 4,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 7); // 12.5% actual waste
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: waste_ratio(0.125) > threshold(-0.1) passes, cost-benefit decides
        // cost=4.0, saved=0.125*20=2.5 → 4.0 < 2.5 is false → CostExceedsBenefit
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    // -- CompactDecision Debug output contains reason field name --

    #[test]
    fn compact_decision_debug_contains_reason_field() {
        let decision = CompactDecision {
            should_compact: false,
            waste_ratio: 0.0,
            active_count: 0,
            total_count: 0,
            reason: CompactReason::EmptyBatch,
        };
        let debug = format!("{:?}", decision);
        assert!(debug.contains("reason"), "Debug output missing 'reason' field");
    }

    // -- CompactReason Triggered variant Debug format contains numeric fields --

    #[test]
    fn compact_reason_triggered_debug_shows_float_fields() {
        let reason = CompactReason::Triggered {
            saved_flops_ratio: 0.375,
            cost_ratio: 4.0,
        };
        let debug = format!("{:?}", reason);
        assert!(debug.contains("saved_flops_ratio"), "Missing saved_flops_ratio in Debug");
        assert!(debug.contains("cost_ratio"), "Missing cost_ratio in Debug");
    }

    // -- OpKind is_compact_eligible false for all non-GEMM via iteration --

    #[test]
    fn op_kind_only_gemm_is_compact_eligible() {
        let non_gemms = [OpKind::Attention, OpKind::Norm, OpKind::Elementwise];
        for op in &non_gemms {
            assert!(!op.is_compact_eligible(), "{:?} should not be compact-eligible", op);
            assert!(!op.is_compute_bound(), "{:?} should not be compute-bound", op);
        }
    }

    // -- Single slot with min_active_count=0 and inactive --

    #[test]
    fn compact_single_slot_inactive_min_active_zero() {
        // Arrange: 1 inactive slot, min_active_count=0
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 5, token_end: 5, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 0, decode_tokens: 0, prefill_tokens: 0,
            compact_required: false, waste_ratio: 1.0,
        };
        let config = CompactConfig { min_active_count: 0, ..CompactConfig::default() };
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: 0 active, but min=0 → passes. 100% waste > 25% threshold, cost=4 < 1.0*20=20 → triggers
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 0);
        assert!((decision.waste_ratio - 1.0).abs() < 1e-5);
    }

    // -- CompactConfig clone preserves all 4 fields precisely --

    #[test]
    fn compact_config_clone_preserves_all_four_fields() {
        let config = CompactConfig {
            waste_threshold: 0.33,
            min_active_count: 7,
            cycles_per_element: 1.5,
            flops_to_mem_ratio: 25.0,
        };
        let cloned = config.clone();
        assert!((cloned.waste_threshold - 0.33).abs() < 1e-10);
        assert_eq!(cloned.min_active_count, 7);
        assert!((cloned.cycles_per_element - 1.5).abs() < 1e-10);
        assert!((cloned.flops_to_mem_ratio - 25.0).abs() < 1e-10);
    }

    // -- CompactDecision default-initialized fields can be read --

    #[test]
    fn compact_decision_zero_initialized_fields() {
        // Arrange: construct with zero-ish values
        let decision = CompactDecision {
            should_compact: false,
            waste_ratio: 0.0,
            active_count: 0,
            total_count: 0,
            reason: CompactReason::EmptyBatch,
        };
        // Assert: all fields accessible and match
        assert!(!decision.should_compact);
        assert!((decision.waste_ratio - 0.0).abs() < 1e-10);
        assert_eq!(decision.active_count, 0);
        assert_eq!(decision.total_count, 0);
    }

    // -- Cost-benefit exact equality: cost_ratio == saved * ratio → NOT triggered (< check) --

    #[test]
    fn compact_cost_benefit_exact_equality_not_triggered() {
        // Arrange: make cost == saved * flops_to_mem_ratio exactly
        // cost = cycles * 2 = 4.0, saved = waste_ratio * flops_to_mem_ratio
        // waste = 0.5 (4 of 8), flops_to_mem = 8.0 → saved = 0.5 * 8.0 = 4.0 == cost
        let config = CompactConfig {
            cycles_per_element: 2.0,
            flops_to_mem_ratio: 8.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 4); // 50% waste
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: cost(4.0) < saved(4.0) is false → not triggered
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    // -- Active slot with negative compact_target still counted as active --

    #[test]
    fn compact_active_slot_negative_compact_target_still_active() {
        // Arrange: active slots with negative compact_target
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: -5 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: -99 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: -1 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: -1 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 40, decode_tokens: 40, prefill_tokens: 0,
            compact_required: true, waste_ratio: 0.333,
        };
        let config = CompactConfig::default();
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: 4 active out of 6, compact_target value does not affect counting
        assert_eq!(decision.active_count, 4);
    }

    // -- Multiple zero-length slots with different start values all counted inactive --

    #[test]
    fn compact_zero_length_slots_varied_starts_all_inactive() {
        // Arrange: slots with token_start == token_end at different values
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 1, token_end: 1, compact_target: -1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 100, token_end: 100, compact_target: -1 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: u32::MAX as usize, token_end: u32::MAX as usize, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 0, decode_tokens: 0, prefill_tokens: 0,
            compact_required: false, waste_ratio: 1.0,
        };
        let config = CompactConfig { min_active_count: 0, ..CompactConfig::default() };
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: all 4 have token_end == token_start → 0 active
        assert_eq!(decision.active_count, 0);
        assert_eq!(decision.total_count, 4);
        assert!((decision.waste_ratio - 1.0).abs() < 1e-5);
    }

    // -- evaluate_compact sequential with different ops on same manifest --

    #[test]
    fn compact_same_manifest_different_ops_independent() {
        let manifest = make_manifest(8, 4); // 50% waste
        let config = CompactConfig::default();
        let d_gemm = evaluate_compact(&manifest, OpKind::Gemm, &config);
        let d_attn = evaluate_compact(&manifest, OpKind::Attention, &config);
        // Assert: same waste ratio regardless of op
        assert!((d_gemm.waste_ratio - d_attn.waste_ratio).abs() < 1e-10);
        assert_eq!(d_gemm.active_count, d_attn.active_count);
        assert_eq!(d_gemm.total_count, d_attn.total_count);
        // GEMM triggers, Attention does not
        assert!(d_gemm.should_compact);
        assert!(!d_attn.should_compact);
    }

    // -- All CompactReason variants are Clone + PartialEq round-trip --

    #[test]
    fn compact_reason_all_variants_clone_partial_eq_roundtrip() {
        let reasons = vec![
            CompactReason::EmptyBatch,
            CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 },
            CompactReason::BelowThreshold { waste_ratio: 0.1, threshold: 0.25 },
            CompactReason::TooFewActive { active: 3, min: 4 },
            CompactReason::CostExceedsBenefit { cost_ratio: 10.0, saved_ratio: 0.2 },
        ];
        // Verify Clone + PartialEq round-trip for each variant
        for original in &reasons {
            let cloned = original.clone();
            assert_eq!(*original, cloned, "Clone round-trip failed for {:?}", original);
        }
        // Verify all pairwise unequal (5 variants, each distinct)
        for i in 0..reasons.len() {
            for j in (i + 1)..reasons.len() {
                assert_ne!(reasons[i], reasons[j], "{:?} == {:?}", reasons[i], reasons[j]);
            }
        }
    }

    // -- BelowThreshold reason with threshold=0.0 and waste=0.0 (exact boundary) --

    #[test]
    fn compact_below_threshold_with_zero_threshold_and_zero_waste() {
        // Arrange: threshold=0.0, waste=0.0 → 0.0 <= 0.0 → BelowThreshold
        let config = CompactConfig {
            waste_threshold: 0.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(4, 4); // 0% waste
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: waste(0.0) <= threshold(0.0) → BelowThreshold
        assert!(!decision.should_compact);
        if let CompactReason::BelowThreshold { waste_ratio, threshold } = decision.reason {
            assert!(waste_ratio.abs() < 1e-5);
            assert!(threshold.abs() < 1e-5);
        } else {
            panic!("Expected BelowThreshold, got {:?}", decision.reason);
        }
    }

    // ============================================================
    // 13 additional tests — remaining edge cases
    // ============================================================

    // -- Negative cycles_per_element: cost becomes negative, always triggers cost-benefit --

    #[test]
    fn compact_negative_cycles_per_element_always_triggers() {
        // Arrange: negative cycles makes cost negative → always < saved * ratio
        let config = CompactConfig {
            cycles_per_element: -5.0,
            waste_threshold: 0.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 7); // 12.5% waste
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: cost = -10.0 < 0.125 * 20.0 = 2.5 → triggers
        assert!(decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::Triggered { .. }));
    }

    // -- CompactDecision with should_compact=true but non-Triggered reason (struct allows it) --

    #[test]
    fn compact_decision_true_with_empty_batch_reason() {
        // Arrange: structurally inconsistent but allowed by the type system
        let decision = CompactDecision {
            should_compact: true,
            waste_ratio: 0.0,
            active_count: 0,
            total_count: 0,
            reason: CompactReason::EmptyBatch,
        };
        // Assert: fields are independently readable
        assert!(decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::EmptyBatch));
        assert_eq!(decision.active_count, 0);
    }

    // -- CompactReason BelowThreshold Debug output contains field names --

    #[test]
    fn compact_reason_below_threshold_debug_shows_fields() {
        let reason = CompactReason::BelowThreshold {
            waste_ratio: 0.15,
            threshold: 0.25,
        };
        let debug = format!("{:?}", reason);
        assert!(debug.contains("waste_ratio"), "Missing waste_ratio in Debug");
        assert!(debug.contains("threshold"), "Missing threshold in Debug");
        assert!(debug.contains("BelowThreshold"));
    }

    // -- CompactReason CostExceedsBenefit Debug output contains field names --

    #[test]
    fn compact_reason_cost_exceeds_benefit_debug_shows_fields() {
        let reason = CompactReason::CostExceedsBenefit {
            cost_ratio: 200.0,
            saved_ratio: 0.375,
        };
        let debug = format!("{:?}", reason);
        assert!(debug.contains("cost_ratio"), "Missing cost_ratio in Debug");
        assert!(debug.contains("saved_ratio"), "Missing saved_ratio in Debug");
        assert!(debug.contains("CostExceedsBenefit"));
    }

    // -- Very large total_count field matches manifest slots length --

    #[test]
    fn compact_large_total_count_consistent() {
        // Arrange: 2048 slots, 512 active → 75% waste
        let manifest = make_manifest(2048, 512);
        let config = CompactConfig::default();
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: total_count == slots.len()
        assert_eq!(decision.total_count, 2048);
        assert_eq!(decision.active_count, 512);
        assert!(decision.should_compact);
    }

    // -- Config partially modified from default preserves unmodified fields --

    #[test]
    fn compact_config_partial_override_preserves_defaults() {
        // Arrange: override only waste_threshold, rest stays default
        let config = CompactConfig {
            waste_threshold: 0.5,
            ..CompactConfig::default()
        };
        // Assert: only waste_threshold changed
        assert!((config.waste_threshold - 0.5).abs() < 1e-5);
        assert_eq!(config.min_active_count, 4);
        assert!((config.cycles_per_element - 2.0).abs() < 1e-5);
        assert!((config.flops_to_mem_ratio - 20.0).abs() < 1e-5);
    }

    // -- Multiple sequential evaluate_compact calls do not mutate config --

    #[test]
    fn compact_multiple_calls_preserve_config() {
        // Arrange
        let config = CompactConfig {
            waste_threshold: 0.3,
            min_active_count: 5,
            cycles_per_element: 1.8,
            flops_to_mem_ratio: 15.0,
        };
        // Act: evaluate multiple times
        let m1 = make_manifest(16, 4);
        let _d1 = evaluate_compact(&m1, OpKind::Gemm, &config);
        let m2 = make_manifest(8, 7);
        let _d2 = evaluate_compact(&m2, OpKind::Attention, &config);
        let m3 = make_manifest(100, 50);
        let _d3 = evaluate_compact(&m3, OpKind::Gemm, &config);
        // Assert: config unchanged after all calls
        assert!((config.waste_threshold - 0.3).abs() < 1e-10);
        assert_eq!(config.min_active_count, 5);
        assert!((config.cycles_per_element - 1.8).abs() < 1e-10);
        assert!((config.flops_to_mem_ratio - 15.0).abs() < 1e-10);
    }

    // -- OpKind Copy semantics: assignment creates independent copy --

    #[test]
    fn op_kind_copy_assignment_independence() {
        // Arrange
        let op1 = OpKind::Gemm;
        let mut op2 = op1;
        // Act: reassign op2
        op2 = OpKind::Attention;
        // Assert: op1 unaffected
        assert_eq!(op1, OpKind::Gemm);
        assert_eq!(op2, OpKind::Attention);
    }

    // -- evaluate_compact with waste_threshold just barely positive and zero waste --

    #[test]
    fn compact_tiny_positive_threshold_zero_waste_below() {
        // Arrange: threshold = 0.0001, waste = 0.0 → 0.0 <= 0.0001
        let config = CompactConfig {
            waste_threshold: 0.0001,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(4, 4); // 0% waste
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: zero waste still below any positive threshold
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }));
    }

    // -- Manifest with single active slot and min_active_count=1 triggers with 50% waste --

    #[test]
    fn compact_single_active_min_one_triggers() {
        // Arrange: 2 slots, 1 active = 50% waste, min=1
        let config = CompactConfig {
            waste_threshold: 0.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(2, 1);
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: 50% > 0% threshold, cost=4.0 < 0.5*20=10.0 → triggers
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 1);
        assert_eq!(decision.total_count, 2);
    }

    // -- Manifest where all slots are active but total_tokens disagrees --

    #[test]
    fn compact_all_active_manifest_tokens_mismatch() {
        // Arrange: 4 active slots but total_tokens says 0 (intentionally wrong metadata)
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 0, // misleading
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.99,
        };
        let config = CompactConfig::default();
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: evaluate_compact uses slots, not total_tokens → 4/4 active = 0% waste
        assert!(!decision.should_compact);
        assert!(decision.waste_ratio.abs() < 1e-5);
        assert_eq!(decision.active_count, 4);
    }

    // -- CompactDecision clone for CostExceedsBenefit preserves all fields --

    #[test]
    fn compact_decision_clone_cost_exceeds_preserves_fields() {
        // Arrange
        let d = CompactDecision {
            should_compact: false,
            waste_ratio: 0.6,
            active_count: 4,
            total_count: 10,
            reason: CompactReason::CostExceedsBenefit {
                cost_ratio: 100.0,
                saved_ratio: 0.6,
            },
        };
        // Act
        let d2 = d.clone();
        // Assert
        assert_eq!(d2.should_compact, d.should_compact);
        assert!((d2.waste_ratio - d.waste_ratio).abs() < 1e-10);
        assert_eq!(d2.active_count, d.active_count);
        assert_eq!(d2.total_count, d.total_count);
        assert_eq!(d2.reason, d.reason);
    }

    // -- SlotType Clone and PartialEq round-trip --

    #[test]
    fn slot_type_clone_and_equality() {
        // Arrange
        let decode = SlotType::Decode;
        let prefill = SlotType::PrefillChunk;
        // Act
        let decode2 = decode.clone();
        let prefill2 = prefill.clone();
        // Assert
        assert_eq!(decode, decode2);
        assert_eq!(prefill, prefill2);
        assert_ne!(decode, prefill);
    }

    // ============================================================
    // 13 additional tests — uncovered edge cases and coverage gaps
    // ============================================================

    // -- OpKind has exactly 4 variants --

    #[test]
    fn op_kind_variant_count_is_four() {
        // Arrange: enumerate all known OpKind variants
        let all_variants = [OpKind::Gemm, OpKind::Attention, OpKind::Norm, OpKind::Elementwise];
        // Assert: exactly 4 distinct variants
        assert_eq!(all_variants.len(), 4);
        for (i, a) in all_variants.iter().enumerate() {
            for (j, b) in all_variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Duplicate variant at indices {} and {}", i, j);
                }
            }
        }
    }

    // -- CompactReason has exactly 5 variants --

    #[test]
    fn compact_reason_variant_count_is_five() {
        // Arrange: enumerate all known CompactReason variants
        let all_variants: Vec<CompactReason> = vec![
            CompactReason::EmptyBatch,
            CompactReason::Triggered { saved_flops_ratio: 0.0, cost_ratio: 0.0 },
            CompactReason::BelowThreshold { waste_ratio: 0.0, threshold: 0.0 },
            CompactReason::TooFewActive { active: 0, min: 0 },
            CompactReason::CostExceedsBenefit { cost_ratio: 0.0, saved_ratio: 0.0 },
        ];
        // Assert: exactly 5 distinct variants
        assert_eq!(all_variants.len(), 5);
        for (i, a) in all_variants.iter().enumerate() {
            for (j, b) in all_variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Duplicate variant at indices {} and {}", i, j);
                }
            }
        }
    }

    // -- CompactReason::Triggered inequality via different cost_ratio --

    #[test]
    fn compact_reason_triggered_inequality_different_cost() {
        let r1 = CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 4.0 };
        let r2 = CompactReason::Triggered { saved_flops_ratio: 0.5, cost_ratio: 5.0 };
        assert_ne!(r1, r2, "Different cost_ratio should make Triggered variants unequal");
    }

    // -- BatchSlot where token_start > token_end (inverted range still inactive) --

    #[test]
    fn compact_slot_inverted_range_counted_inactive() {
        // Arrange: slot with token_start > token_end — should be inactive since !(end > start)
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 10, token_end: 5, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 100, token_end: 50, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 3, decode_tokens: 3, prefill_tokens: 0,
            compact_required: true, waste_ratio: 0.4,
        };
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: only 3 active (slots with end > start), 2 inactive (inverted range)
        assert_eq!(decision.active_count, 3);
        assert_eq!(decision.total_count, 5);
        assert!((decision.waste_ratio - 0.4).abs() < 1e-5);
    }

    // -- Empty batch with GEMM op (explicit coverage of Gemm + EmptyBatch combo) --

    #[test]
    fn compact_empty_batch_gemm_explicit() {
        // Arrange
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0, decode_tokens: 0, prefill_tokens: 0,
            compact_required: false, waste_ratio: 0.0,
        };
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &CompactConfig::default());
        // Assert: EmptyBatch reason even for GEMM op
        assert!(matches!(decision.reason, CompactReason::EmptyBatch));
        assert!(!decision.should_compact);
        assert_eq!(decision.active_count, 0);
        assert_eq!(decision.total_count, 0);
    }

    // -- Waste ratio depends on ratio, not absolute count (different sizes, same ratio) --

    #[test]
    fn compact_waste_ratio_same_for_scaled_batches() {
        // Arrange: 2 batches with same waste ratio but different sizes
        let manifest_small = make_manifest(4, 2); // 50% waste
        let manifest_large = make_manifest(100, 50); // 50% waste
        let config = CompactConfig { min_active_count: 1, ..CompactConfig::default() };
        // Act
        let d_small = evaluate_compact(&manifest_small, OpKind::Gemm, &config);
        let d_large = evaluate_compact(&manifest_large, OpKind::Gemm, &config);
        // Assert: same waste ratio regardless of batch size
        assert!((d_small.waste_ratio - d_large.waste_ratio).abs() < 1e-10);
        // Both should trigger (same waste, same cost-benefit)
        assert_eq!(d_small.should_compact, d_large.should_compact);
    }

    // -- Float precision: waste ratio when active does not divide total evenly --

    #[test]
    fn compact_waste_ratio_float_precision_odd_division() {
        // Arrange: 7 active out of 11 = 4/11 ≈ 0.363636...
        let manifest = make_manifest(11, 7);
        let config = CompactConfig::default();
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: waste = (11-7)/11 = 4/11
        let expected = 4.0 / 11.0;
        assert!((decision.waste_ratio - expected).abs() < 1e-6);
        assert_eq!(decision.active_count, 7);
        assert_eq!(decision.total_count, 11);
    }

    // -- Non-GEMM op with zero flops_to_mem_ratio still returns CostExceedsBenefit (compute-bound guard) --

    #[test]
    fn compact_attention_zero_flops_ratio_returns_cost_exceeds() {
        // Arrange: Attention with zero flops_to_mem_ratio — compute-bound check fires first
        let manifest = make_manifest(8, 4);
        let config = CompactConfig {
            flops_to_mem_ratio: 0.0,
            min_active_count: 1,
            waste_threshold: 0.0,
            ..CompactConfig::default()
        };
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Attention, &config);
        // Assert: compute-bound guard sets cost_ratio=inf, saved_ratio=0.0
        if let CompactReason::CostExceedsBenefit { cost_ratio, saved_ratio } = decision.reason {
            assert!(cost_ratio.is_infinite() && cost_ratio.is_sign_positive());
            assert!((saved_ratio - 0.0).abs() < 1e-5);
        } else {
            panic!("Expected CostExceedsBenefit for Attention, got {:?}", decision.reason);
        }
    }

    // -- Inactive PrefillChunk slot counted same as inactive Decode slot --

    #[test]
    fn compact_inactive_prefill_chunk_same_as_inactive_decode() {
        // Arrange: 4 active Decode + 4 inactive PrefillChunk
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
            BatchSlot { request_id: 4, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 6, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 7, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots, total_tokens: 4, decode_tokens: 4, prefill_tokens: 0,
            compact_required: true, waste_ratio: 0.5,
        };
        let config = CompactConfig::default();
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: slot_type does not affect counting — 4 active out of 8
        assert_eq!(decision.active_count, 4);
        assert_eq!(decision.total_count, 8);
        assert!((decision.waste_ratio - 0.5).abs() < 1e-5);
    }

    // -- Manifest with negative waste_ratio field (informational, ignored by evaluate_compact) --

    #[test]
    fn compact_manifest_negative_waste_ratio_field_ignored() {
        // Arrange: manifest.waste_ratio is negative but actual slot data shows 50% waste
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 6, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 7, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 4,
            decode_tokens: 4,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: -0.5, // misleading negative value
        };
        let config = CompactConfig::default();
        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);
        // Assert: evaluate_compact computes waste from slots, ignoring negative field
        assert!((decision.waste_ratio - 0.5).abs() < 1e-5);
        assert!(decision.should_compact);
    }

    // -- CompactDecision with should_compact=true but BelowThreshold reason (struct inconsistency) --

    #[test]
    fn compact_decision_true_with_below_threshold_reason() {
        // Arrange: structurally inconsistent but allowed by the type system
        let decision = CompactDecision {
            should_compact: true,
            waste_ratio: 0.5,
            active_count: 4,
            total_count: 8,
            reason: CompactReason::BelowThreshold { waste_ratio: 0.1, threshold: 0.25 },
        };
        // Assert: fields are independently readable
        assert!(decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }));
        assert_eq!(decision.active_count, 4);
    }

    // -- CompactReason::EmptyBatch clone produces equal value --

    #[test]
    fn compact_reason_empty_batch_clone_equal() {
        let reason = CompactReason::EmptyBatch;
        let cloned = reason.clone();
        assert_eq!(reason, cloned);
    }

    // -- CompactConfig with all identical fields constructed from two separate defaults --

    #[test]
    fn compact_config_two_defaults_are_equal() {
        // Arrange & Act
        let config1 = CompactConfig {
            waste_threshold: 0.25,
            min_active_count: 4,
            cycles_per_element: 2.0,
            flops_to_mem_ratio: 20.0,
        };
        let config2 = CompactConfig {
            waste_threshold: 0.25,
            min_active_count: 4,
            cycles_per_element: 2.0,
            flops_to_mem_ratio: 20.0,
        };
        // Assert: field-by-field equality
        assert!((config1.waste_threshold - config2.waste_threshold).abs() < 1e-10);
        assert_eq!(config1.min_active_count, config2.min_active_count);
        assert!((config1.cycles_per_element - config2.cycles_per_element).abs() < 1e-10);
        assert!((config1.flops_to_mem_ratio - config2.flops_to_mem_ratio).abs() < 1e-10);
    }

    // ============================================================
    // 13 additional tests — NaN, extreme values, structural edge cases
    // ============================================================

    // -- NaN cycles_per_element: cost becomes NaN, NaN < anything is false → CostExceedsBenefit --

    #[test]
    fn compact_nan_cycles_per_element_never_triggers() {
        // Arrange: NaN cycles makes cost_ratio NaN; NaN < X is always false
        let config = CompactConfig {
            cycles_per_element: f32::NAN,
            waste_threshold: 0.0,
            min_active_count: 1,
            flops_to_mem_ratio: 10000.0,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 4); // 50% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: NaN < anything is false → CostExceedsBenefit
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    // -- NaN waste_threshold: waste_ratio <= NaN is false → skips BelowThreshold, goes to cost-benefit --

    #[test]
    fn compact_nan_threshold_skips_below_threshold_goes_to_cost_benefit() {
        // Arrange: NaN threshold; waste_ratio <= NaN is false → BelowThreshold guard skipped
        // Then cost-benefit decides: cost=4.0, saved=0.75*20.0=15.0 → 4.0 < 15.0 → triggers
        let config = CompactConfig {
            waste_threshold: f32::NAN,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 2); // 75% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: NaN threshold bypasses BelowThreshold; cost-benefit passes → triggers
        assert!(decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::Triggered { .. }));
    }

    // -- NaN flops_to_mem_ratio: saved * NaN = NaN, NaN < NaN is false → CostExceedsBenefit --

    #[test]
    fn compact_nan_flops_to_mem_ratio_never_triggers() {
        // Arrange: NaN flops_to_mem_ratio makes saved side NaN
        let config = CompactConfig {
            flops_to_mem_ratio: f32::NAN,
            waste_threshold: 0.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 4); // 50% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: cost(4.0) < NaN → false → CostExceedsBenefit
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    // -- Very large batch: 4096 slots, all active → 0% waste --

    #[test]
    fn compact_very_large_batch_all_active() {
        // Arrange: 4096 slots, all active
        let manifest = make_manifest(4096, 4096);
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 0% waste → BelowThreshold
        assert!(!decision.should_compact);
        assert_eq!(decision.active_count, 4096);
        assert_eq!(decision.total_count, 4096);
        assert!(decision.waste_ratio.abs() < 1e-5);
    }

    // -- evaluate_compact with all 4 OpKind variants all not triggered on low waste --

    #[test]
    fn compact_all_op_kinds_below_threshold_consistent() {
        // Arrange: low waste batch
        let manifest = make_manifest(8, 7); // 12.5% waste
        let config = CompactConfig::default();

        // Act & Assert: all ops should return BelowThreshold
        for op in [OpKind::Gemm, OpKind::Attention, OpKind::Norm, OpKind::Elementwise] {
            let decision = evaluate_compact(&manifest, op, &config);
            assert!(!decision.should_compact, "{:?} should not trigger on 12.5% waste", op);
            assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }),
                "{:?} expected BelowThreshold, got {:?}", op, decision.reason);
            // All ops see the same waste ratio and counts
            assert!((decision.waste_ratio - 0.125).abs() < 1e-5);
            assert_eq!(decision.active_count, 7);
            assert_eq!(decision.total_count, 8);
        }
    }

    // -- Cost-benefit just barely passes: cost < saved * ratio by a tiny margin --

    #[test]
    fn compact_cost_benefit_barely_passes() {
        // Arrange: craft values so cost < saved * ratio by a small margin
        // cost = cycles * 2, saved = waste * flops_to_mem_ratio
        // Want: 2*3.99 = 7.98 < 0.4 * 20.0 = 8.0
        let config = CompactConfig {
            cycles_per_element: 3.99,
            flops_to_mem_ratio: 20.0,
            waste_threshold: 0.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(10, 6); // 40% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 7.98 < 8.0 → triggers
        assert!(decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::Triggered { .. }));
    }

    // -- make_manifest with total=0, active=0 produces valid empty manifest --

    #[test]
    fn helper_make_manifest_zero_total_zero_active() {
        // Act
        let manifest = make_manifest(0, 0);

        // Assert: empty manifest
        assert!(manifest.slots.is_empty());
        assert_eq!(manifest.total_tokens, 0);
        assert_eq!(manifest.decode_tokens, 0);
        assert_eq!(manifest.prefill_tokens, 0);
    }

    // -- CompactReason::Triggered with saved_flops_ratio=0.0 (struct permits it) --

    #[test]
    fn compact_reason_triggered_zero_saved_flops() {
        // Arrange: construct directly (evaluate_compact won't produce this naturally)
        let reason = CompactReason::Triggered {
            saved_flops_ratio: 0.0,
            cost_ratio: 0.0,
        };

        // Assert: struct permits zero values
        if let CompactReason::Triggered { saved_flops_ratio, cost_ratio } = reason {
            assert!(saved_flops_ratio.abs() < 1e-10);
            assert!(cost_ratio.abs() < 1e-10);
        } else {
            panic!("Expected Triggered variant");
        }
    }

    // -- CompactDecision clone then mutate reason field independently --

    #[test]
    fn compact_decision_clone_reason_independence() {
        // Arrange
        let d = CompactDecision {
            should_compact: true,
            waste_ratio: 0.5,
            active_count: 4,
            total_count: 8,
            reason: CompactReason::Triggered {
                saved_flops_ratio: 0.5,
                cost_ratio: 4.0,
            },
        };
        let d2 = d.clone();

        // Assert: cloned reason matches original
        assert_eq!(d2.reason, d.reason);

        // Further verify by destructuring both
        match (&d.reason, &d2.reason) {
            (
                CompactReason::Triggered { saved_flops_ratio: s1, cost_ratio: c1 },
                CompactReason::Triggered { saved_flops_ratio: s2, cost_ratio: c2 },
            ) => {
                assert!((s1 - s2).abs() < 1e-10);
                assert!((c1 - c2).abs() < 1e-10);
            }
            _ => panic!("Both should be Triggered"),
        }
    }

    // -- CompactConfig with f32::EPSILON as waste_threshold --

    #[test]
    fn compact_threshold_f32_epsilon() {
        // Arrange: threshold = f32::EPSILON ≈ 1.19e-7
        let config = CompactConfig {
            waste_threshold: f32::EPSILON,
            min_active_count: 1,
            flops_to_mem_ratio: 100.0,
            ..CompactConfig::default()
        };
        // 50% waste >> epsilon threshold
        let manifest = make_manifest(8, 4);

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 0.5 >> EPSILON → passes threshold; cost=4.0, saved=0.5*100=50 → triggers
        assert!(decision.should_compact);
    }

    // -- evaluate_compact triggers with extremely small waste but zero threshold and high ratio --

    #[test]
    fn compact_tiny_waste_zero_threshold_high_ratio() {
        // Arrange: 1 inactive out of 256 ≈ 0.39% waste, but threshold=0 and very high ratio
        let config = CompactConfig {
            waste_threshold: 0.0,
            min_active_count: 1,
            flops_to_mem_ratio: 10000.0,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(256, 255); // 1/256 ≈ 0.39% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: waste > 0 > threshold, cost=4.0, saved=(1/256)*10000≈39.06 → triggers
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 255);
        assert_eq!(decision.total_count, 256);
    }

    // -- BatchSlot with usize::MAX as request_id field accessibility --

    #[test]
    fn batch_slot_usize_max_request_id() {
        // Arrange: construct slot with max request_id
        let slot = BatchSlot {
            request_id: u64::MAX,
            slot_type: SlotType::PrefillChunk,
            token_start: 0,
            token_end: 1000,
            compact_target: i32::MAX,
        };

        // Assert: all fields readable at extreme values
        assert_eq!(slot.request_id, u64::MAX);
        assert_eq!(slot.slot_type, SlotType::PrefillChunk);
        assert_eq!(slot.token_start, 0);
        assert_eq!(slot.token_end, 1000);
        assert_eq!(slot.compact_target, i32::MAX);
    }

    // -- evaluate_compact with zero threshold and zero min_active on empty batch: EmptyBatch fires first --

    #[test]
    fn compact_all_guards_relaxed_empty_batch_still_empty() {
        // Arrange: relax all guards
        let config = CompactConfig {
            waste_threshold: 0.0,
            min_active_count: 0,
            cycles_per_element: 0.0,
            flops_to_mem_ratio: f32::MAX,
        };
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: EmptyBatch is the first guard; always fires for empty slots
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::EmptyBatch));
        assert_eq!(decision.active_count, 0);
        assert_eq!(decision.total_count, 0);
    }

    // ============================================================
    // 10 additional tests — BatchManifest::should_compact, extreme values, edge cases
    // ============================================================

    // -- BatchManifest::should_compact uses compact_target >= 0, not token range --

    #[test]
    fn manifest_should_compact_uses_compact_target_not_token_range() {
        // Arrange: use ChunkedPrefillConfig with same defaults as CompactConfig
        use crate::scheduler::chunked_prefill::ChunkedPrefillConfig;
        let cp_config = ChunkedPrefillConfig::default();
        // Slots: all have active token ranges BUT compact_target = -1 → not counted as active
        // by BatchManifest::should_compact (which checks compact_target >= 0)
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 100, compact_target: -1 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 100, compact_target: -1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 100, compact_target: -1 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 100, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 400,
            decode_tokens: 400,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.5, // 50% waste
        };

        // Act
        let result = manifest.should_compact(&cp_config);

        // Assert: compact_target all -1 → 0 active → does not trigger
        // even though all slots have active token ranges
        assert!(!result);
    }

    // -- BatchManifest::should_compact triggers when compact_target >= 0 meets threshold --

    #[test]
    fn manifest_should_compact_triggers_with_positive_compact_targets() {
        // Arrange
        use crate::scheduler::chunked_prefill::ChunkedPrefillConfig;
        let cp_config = ChunkedPrefillConfig::default();
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 6, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 7, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 4,
            decode_tokens: 4,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.5, // 50% > 25% threshold
        };

        // Act
        let result = manifest.should_compact(&cp_config);

        // Assert: 4 slots with compact_target >= 0, waste_ratio=0.5 > 0.25 → triggers
        assert!(result);
    }

    // -- evaluate_compact and BatchManifest::should_compact diverge when compact_target disagrees with token range --

    #[test]
    fn evaluate_compact_vs_manifest_should_compact_semantic_difference() {
        // Arrange: slot with active token range but compact_target = -1
        use crate::scheduler::chunked_prefill::ChunkedPrefillConfig;
        let cp_config = ChunkedPrefillConfig::default();
        let compact_config = CompactConfig::default();

        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: -1 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: -1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: -1 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: -1 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 6, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 7, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 40,
            decode_tokens: 40,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.5,
        };

        // Act
        let eval_result = evaluate_compact(&manifest, OpKind::Gemm, &compact_config);
        let manifest_result = manifest.should_compact(&cp_config);

        // Assert: evaluate_compact counts 4 active (token_end > token_start)
        // BatchManifest::should_compact counts 0 active (compact_target >= 0)
        // They disagree because they use different active-counting semantics
        assert_eq!(eval_result.active_count, 4, "evaluate_compact should count 4 active by token range");
        assert!(!manifest_result, "BatchManifest::should_compact should see 0 active by compact_target");
    }

    // -- f32::INFINITY as waste_threshold: nothing can exceed infinity --

    #[test]
    fn compact_infinity_threshold_never_triggers() {
        // Arrange: threshold = +inf, any finite waste <= inf → BelowThreshold
        let config = CompactConfig {
            waste_threshold: f32::INFINITY,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 1); // 87.5% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 0.875 <= +inf → BelowThreshold
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::BelowThreshold { .. }));
    }

    // -- All slots share the same request_id: evaluate_compact does not care --

    #[test]
    fn compact_all_slots_same_request_id() {
        // Arrange: 8 slots all with request_id = 42, 4 active, 4 inactive
        let slots = vec![
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 3 },
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 4,
            decode_tokens: 4,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.5,
        };
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: request_id duplication does not affect counting
        assert_eq!(decision.active_count, 4);
        assert_eq!(decision.total_count, 8);
        assert!((decision.waste_ratio - 0.5).abs() < 1e-5);
        assert!(decision.should_compact);
    }

    // -- f32::MAX as cycles_per_element: cost is enormous, never triggers cost-benefit --

    #[test]
    fn compact_f32_max_cycles_per_element_never_triggers_cost_benefit() {
        // Arrange: cycles = f32::MAX → cost = 2 * f32::MAX = +inf (overflow)
        let config = CompactConfig {
            cycles_per_element: f32::MAX,
            waste_threshold: 0.0,
            min_active_count: 1,
            flops_to_mem_ratio: f32::MAX,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 4); // 50% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 2 * f32::MAX overflows to +inf; inf < anything is false → CostExceedsBenefit
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    // -- Slots with token_end = usize::MAX (extremely large token range) still counted as active --

    #[test]
    fn compact_slots_with_usize_max_token_end_active() {
        // Arrange: slots with extremely large token ranges
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: usize::MAX, compact_target: 0 },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: usize::MAX, compact_target: 1 },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: usize::MAX, compact_target: 2 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: usize::MAX, compact_target: 3 },
            BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
            BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: usize::MAX,
            decode_tokens: usize::MAX,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.333,
        };
        let config = CompactConfig::default();

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 4 active out of 6 (33.3% waste), token range magnitude does not matter
        assert_eq!(decision.active_count, 4);
        assert_eq!(decision.total_count, 6);
        let expected_waste = 2.0 / 6.0;
        assert!((decision.waste_ratio - expected_waste).abs() < 1e-5);
    }

    // -- f32::NEG_INFINITY as waste_threshold: all waste exceeds it --

    #[test]
    fn compact_neg_infinity_threshold_all_waste_exceeds() {
        // Arrange: threshold = -inf, any positive waste > -inf → passes threshold guard
        let config = CompactConfig {
            waste_threshold: f32::NEG_INFINITY,
            min_active_count: 1,
            flops_to_mem_ratio: 20.0,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 7); // 12.5% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: 0.125 > -inf passes threshold; cost=4.0, saved=0.125*20=2.5 → 4.0 < 2.5 false
        // → CostExceedsBenefit
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. }));
    }

    // -- Slots with compact_target = i32::MIN are inactive by BatchManifest::should_compact --

    #[test]
    fn manifest_should_compact_i32_min_compact_target_inactive() {
        // Arrange: slots with compact_target = i32::MIN (< 0 → inactive for should_compact)
        use crate::scheduler::chunked_prefill::ChunkedPrefillConfig;
        let cp_config = ChunkedPrefillConfig::default();
        let slots = vec![
            BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 100, compact_target: i32::MIN },
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 100, compact_target: i32::MIN },
            BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 100, compact_target: i32::MIN },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 100, compact_target: i32::MIN },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 400,
            decode_tokens: 400,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.5,
        };

        // Act
        let result = manifest.should_compact(&cp_config);

        // Assert: i32::MIN < 0 → 0 active slots counted → does not trigger
        assert!(!result);
    }

    // -- f32::MIN (most negative finite) as cycles_per_element triggers easily --

    #[test]
    fn compact_f32_min_cycles_per_element_triggers() {
        // Arrange: cycles = f32::MIN ≈ -3.4e38 → cost = 2 * f32::MIN (very large negative)
        let config = CompactConfig {
            cycles_per_element: f32::MIN,
            waste_threshold: 0.0,
            min_active_count: 1,
            ..CompactConfig::default()
        };
        let manifest = make_manifest(8, 7); // 12.5% waste

        // Act
        let decision = evaluate_compact(&manifest, OpKind::Gemm, &config);

        // Assert: cost = very large negative < 0.125 * 20 = 2.5 → triggers
        assert!(decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::Triggered { .. }));
    }
}
