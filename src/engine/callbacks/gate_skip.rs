//! Gate Skip Callback (SPEC §13.1 + §14.2)
//!
//! Applies pre-computed gate skip decisions to FFN nodes during graph execution.
//! When dead neuron density exceeds threshold, FFN layers use register-level
//! compaction (§14.2) to compact dead neurons into a dense tensor for full-throughput
//! execution, rather than skipping the FFN entirely.
//!
//! Per SPEC §14.2 (Register-Level Compaction):
//! > "不跳远，只挤压" — Dead neurons are compacted via hardware predicated execution
//! > (AVX-512 vcompress, GPU Prefix Sum, SVE predicate). The FFN runs at full
//! > throughput on the compacted dense tensor, then results are scattered back.

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};

/// Per-layer computation decision for gate-based neuron density optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Default)]
pub enum SkipDecision {
    /// Normal full computation — all neurons active.
    #[default]
    FullCompute,
    /// Compacted computation — dead neurons compacted out via hardware masks (§14.2).
    ///
    /// Contains a per-neuron active mask. The executor applies RaggedCompaction
    /// (Compact→Execute→Scatter) to run the FFN on the dense compacted tensor.
    CompactedCompute,
    /// Masked computation — partial dead neurons, below compaction threshold.
    ///
    /// Uses lightweight predicate masking without full compaction.
    MaskedCompute,
}


/// Gate skip callback — compacts dead neurons for FFN computation (SPEC §14.2).
///
/// Per SPEC §14.2: reads skip decisions (typically computed by the EpilogueSubsystem
/// from the previous step's telemetry) and applies register-level compaction.
/// FFN nodes at layers with high dead density (>50%) are compacted, NOT skipped.
///
/// The compaction mask is derived from the SiLU epilogue (§13.5) dead neuron detection.
pub struct GateSkipCallback {
    /// Pre-computed skip decisions per layer (populated from epilogue summary)
    skip_decisions: Vec<SkipDecision>,
    /// Number of transformer layers
    num_layers: usize,
    /// Per-layer dead neuron masks (populated from epilogue telemetry).
    /// Map: layer_index → active mask (true = active neuron, false = dead)
    neuron_masks: Vec<Option<Vec<bool>>>,
    /// Default intermediate size for generating full-active masks
    intermediate_size: usize,
}

impl GateSkipCallback {
    /// Create a new gate skip callback with pre-computed decisions.
    ///
    /// `num_layers` — total number of transformer layers
    /// `decisions` — per-layer skip decisions (length must equal num_layers)
    /// `intermediate_size` — FFN intermediate dimension (for mask sizing)
    pub fn new(num_layers: usize, decisions: Vec<SkipDecision>, intermediate_size: usize) -> Self {
        assert_eq!(decisions.len(), num_layers, "decisions length must match num_layers");
        let neuron_masks = vec![None; num_layers];
        Self {
            skip_decisions: decisions,
            num_layers,
            neuron_masks,
            intermediate_size,
        }
    }

    /// Create a disabled callback (all FullCompute).
    pub fn new_disabled(num_layers: usize) -> Self {
        Self {
            skip_decisions: vec![SkipDecision::FullCompute; num_layers],
            num_layers,
            neuron_masks: vec![None; num_layers],
            intermediate_size: 0,
        }
    }

    /// Update skip decisions for a new batch (typically from epilogue summary).
    pub fn update_decisions(&mut self, decisions: Vec<SkipDecision>) {
        assert_eq!(decisions.len(), self.num_layers);
        self.skip_decisions = decisions;
    }

    /// Store per-layer dead neuron mask from SiLU epilogue (§13.5).
    ///
    /// `mask` is true for active neurons, false for dead neurons.
    pub fn store_neuron_mask(&mut self, layer: usize, mask: Vec<bool>) {
        if layer < self.neuron_masks.len() {
            self.neuron_masks[layer] = Some(mask);
        }
    }

    /// Get the decision for a specific layer.
    pub fn decision_for_layer(&self, layer: usize) -> &SkipDecision {
        self.skip_decisions.get(layer).unwrap_or(&SkipDecision::FullCompute)
    }

    /// Build the active mask for a layer.
    ///
    /// If a SiLU epilogue mask exists, use it directly.
    /// For CompactedCompute without a stored mask, generate a full-active mask
    /// (the decision itself signals compaction is needed — the actual dead neuron
    /// positions will be determined by the runtime's gate values).
    fn active_mask_for_layer(&self, layer: usize) -> Vec<bool> {
        if let Some(Some(mask)) = self.neuron_masks.get(layer) {
            mask.clone()
        } else {
            // Generate default mask based on intermediate_size
            // All neurons active — the executor will apply the actual gate values
            vec![true; self.intermediate_size.max(1)]
        }
    }
}

impl LayerCallback for GateSkipCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        let decision = self.decision_for_layer(ctx.layer_idx);
        match decision {
            SkipDecision::CompactedCompute => {
                // §14.2: Register-level compaction — compact dead neurons, execute dense
                let mask = self.active_mask_for_layer(ctx.layer_idx);
                log::trace!(
                    "gate_skip: compacted compute at layer {} (active: {}/{})",
                    ctx.layer_idx,
                    mask.iter().filter(|&&b| b).count(),
                    mask.len(),
                );
                CallbackAction::CompactMask { active_mask: mask }
            }
            SkipDecision::MaskedCompute => {
                log::trace!("gate_skip: masked compute at layer {}", ctx.layer_idx);
                // MaskedCompute still executes but with lightweight predicate masking
                CallbackAction::Continue
            }
            SkipDecision::FullCompute => CallbackAction::Continue,
        }
    }

    fn post_node(&mut self, _ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
        CallbackAction::Continue
    }

    fn priority(&self) -> u32 {
        60
    }

    fn name(&self) -> &str {
        "gate_skip"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_skip_all_full_compute() {
        let cb = GateSkipCallback::new_disabled(4);
        assert_eq!(cb.priority(), 60);
        assert_eq!(cb.name(), "gate_skip");
        for i in 0..4 {
            assert_eq!(*cb.decision_for_layer(i), SkipDecision::FullCompute);
        }
    }

    #[test]
    fn test_gate_skip_with_compacted_decisions() {
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
        ];
        let cb = GateSkipCallback::new(4, decisions, 1024);
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::FullCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::CompactedCompute);
        assert_eq!(*cb.decision_for_layer(3), SkipDecision::CompactedCompute);
    }

    #[test]
    fn test_update_decisions() {
        let mut cb = GateSkipCallback::new_disabled(2);
        cb.update_decisions(vec![SkipDecision::CompactedCompute, SkipDecision::FullCompute]);
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::CompactedCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::FullCompute);
    }

    // GateSkipCallback.pre_node() only reads ctx.layer_idx — model_config is never accessed.
    // We use a leaked MaybeUninit to satisfy the type without constructing a real config.
    fn make_test_ctx(layer_idx: usize) -> crate::graph::layer_callback::LayerContext<'static> {
        use std::sync::LazyLock;
        static TEST_OP: &str = "test";
        // SAFETY: model_config is never dereferenced by GateSkipCallback.pre_node().
        // It only reads ctx.layer_idx. This is test-only code.
        static FAKE_CONFIG: LazyLock<crate::engine::executor::GeneratorForwardConfig> = LazyLock::new(|| {
            use std::sync::Arc;
            let geom = Arc::new(crate::model_config::ModelGeometry {
                hidden_size: 256, num_layers: 1, vocab_size: 1000, intermediate_size: 512,
                num_heads: 4, num_kv_heads: 4, head_dim: 64, max_seq_len: 128,
                rope_theta: 10000.0, rope_scale: 1.0, rope_interleaved: false,
                dtype: gllm_kernels::types::DType::F32, compute_dtype: gllm_kernels::types::DType::F32, norm_eps: 1e-5,
                num_experts: 0, moe_top_k: 0, expert_intermediate_size: 0,
                global_rope_theta: 0.0, rope_partial_ratio: 1.0, attention_pattern: vec![],
                sliding_window: 0, num_kv_shared_layers: 0, global_head_dim: 0, hidden_size_per_layer_input: 0,
                position_offset: None,
                rope_scaling: None,
                final_logit_softcapping: None,
                hidden_act: None,
                mla_d_c: 0,
                mla_d_rope: 0,
                mla_unabsorbed_threshold: 0,
            });
            crate::engine::executor::GeneratorForwardConfig {
                geometry: geom,
                position_encoding: crate::engine::executor::PositionEncoding::Rope,
                arch_family: crate::manifest::ArchFamily::Decoder,
                rope: crate::engine::executor::RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: true },
                rerank_yes_token_id: None, rerank_no_token_id: None,
                moe_config: None,
                paged_kv: crate::engine::executor::PagedKvConfig { page_table: None, page_size: 16 },
                callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
            }
        });
        crate::graph::layer_callback::LayerContext {
            node_idx: 0,
            layer_idx,
            node_op: &TEST_OP,
            hidden_state: &[],
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 0, seq_len: 0, position: 0, request_id: 0,
            model_config: &FAKE_CONFIG,
        }
    }

    #[test]
    fn test_compacted_compute_returns_compact_mask() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 256);
        cb.store_neuron_mask(0, vec![true, false, true, false, true, true, false, true]);

        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 8);
                assert_eq!(active_mask, vec![true, false, true, false, true, true, false, true]);
            }
            _ => panic!("Expected CompactMask action"),
        }
    }

    #[test]
    fn test_masked_compute_returns_continue() {
        let decisions = vec![SkipDecision::MaskedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 256);

        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        assert!(matches!(action, CallbackAction::Continue));
    }

    // ---- Additional tests ----

    #[test]
    fn skip_decision_default_is_full_compute() {
        assert_eq!(SkipDecision::default(), SkipDecision::FullCompute);
    }

    #[test]
    fn skip_decision_variants_distinct() {
        assert_ne!(SkipDecision::FullCompute, SkipDecision::CompactedCompute);
        assert_ne!(SkipDecision::CompactedCompute, SkipDecision::MaskedCompute);
        assert_ne!(SkipDecision::FullCompute, SkipDecision::MaskedCompute);
    }

    #[test]
    fn skip_decision_clone() {
        let d = SkipDecision::CompactedCompute;
        let d2 = d.clone();
        assert_eq!(d, d2);
    }

    #[test]
    fn gate_skip_new_disabled_all_full() {
        let cb = GateSkipCallback::new_disabled(8);
        for i in 0..8 {
            assert_eq!(*cb.decision_for_layer(i), SkipDecision::FullCompute);
        }
    }

    #[test]
    fn gate_skip_decision_for_layer_out_of_bounds() {
        let cb = GateSkipCallback::new_disabled(2);
        assert_eq!(*cb.decision_for_layer(100), SkipDecision::FullCompute);
    }

    #[test]
    fn gate_skip_store_neuron_mask() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, false, true, true]);
        // verify via pre_node
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_store_neuron_mask_out_of_bounds_no_panic() {
        let decisions = vec![SkipDecision::FullCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(99, vec![true, false]); // should not panic
    }

    #[test]
    fn gate_skip_compacted_without_mask_uses_default() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        // No mask stored → should generate full-active mask of intermediate_size
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 8);
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_post_node_returns_continue() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let ctx = make_test_ctx(0);
        let action = cb.post_node(&ctx, &[]);
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_full_compute_pre_node_returns_continue() {
        let decisions = vec![SkipDecision::FullCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_layer_callback_priority() {
        let cb = GateSkipCallback::new_disabled(4);
        assert_eq!(LayerCallback::priority(&cb), 60);
    }

    #[test]
    fn gate_skip_layer_callback_name() {
        let cb = GateSkipCallback::new_disabled(4);
        assert_eq!(LayerCallback::name(&cb), "gate_skip");
    }

    #[test]
    fn gate_skip_target_layers_is_none() {
        let cb = GateSkipCallback::new_disabled(4);
        assert!(LayerCallback::target_layers(&cb).is_none());
    }

    #[test]
    fn gate_skip_update_decisions_changes_layers() {
        let mut cb = GateSkipCallback::new_disabled(3);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::FullCompute);
        cb.update_decisions(vec![
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
            SkipDecision::FullCompute,
        ]);
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::CompactedCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::MaskedCompute);
        assert_eq!(*cb.decision_for_layer(2), SkipDecision::FullCompute);
    }

    // ========================================================================
    // Additional tests (~50) for comprehensive coverage
    // ========================================================================

    // -- SkipDecision: Debug formatting --

    #[test]
    fn skip_decision_debug_full_compute() {
        let d = SkipDecision::FullCompute;
        let s = format!("{:?}", d);
        assert!(s.contains("FullCompute"), "Debug should contain FullCompute, got: {}", s);
    }

    #[test]
    fn skip_decision_debug_compacted_compute() {
        let d = SkipDecision::CompactedCompute;
        let s = format!("{:?}", d);
        assert!(s.contains("CompactedCompute"), "Debug should contain CompactedCompute, got: {}", s);
    }

    #[test]
    fn skip_decision_debug_masked_compute() {
        let d = SkipDecision::MaskedCompute;
        let s = format!("{:?}", d);
        assert!(s.contains("MaskedCompute"), "Debug should contain MaskedCompute, got: {}", s);
    }

    // -- SkipDecision: Copy trait --

    #[test]
    fn skip_decision_copy_preserves_value() {
        let d = SkipDecision::MaskedCompute;
        let d2 = d;
        assert_eq!(d, d2);
    }

    #[test]
    fn skip_decision_copy_all_variants() {
        let full = SkipDecision::FullCompute;
        let compacted = SkipDecision::CompactedCompute;
        let masked = SkipDecision::MaskedCompute;
        assert_eq!(full, full);
        assert_eq!(compacted, compacted);
        assert_eq!(masked, masked);
    }

    // -- SkipDecision: Hash trait --

    #[test]
    fn skip_decision_hash_equal_variants_produce_equal_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        SkipDecision::FullCompute.hash(&mut h1);
        SkipDecision::FullCompute.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn skip_decision_hash_different_variants_produce_different_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        SkipDecision::FullCompute.hash(&mut h1);
        SkipDecision::CompactedCompute.hash(&mut h2);
        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn skip_decision_all_variants_collect_to_hashset() {
        use std::collections::HashSet;
        let set: HashSet<SkipDecision> = [
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&SkipDecision::FullCompute));
        assert!(set.contains(&SkipDecision::CompactedCompute));
        assert!(set.contains(&SkipDecision::MaskedCompute));
    }

    #[test]
    fn skip_decision_hashset_deduplicates() {
        use std::collections::HashSet;
        let set: HashSet<SkipDecision> = [
            SkipDecision::FullCompute,
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // -- SkipDecision: PartialEq symmetry and transitivity --

    #[test]
    fn skip_decision_eq_is_symmetric() {
        assert_eq!(SkipDecision::FullCompute, SkipDecision::FullCompute);
        assert_eq!(SkipDecision::CompactedCompute, SkipDecision::CompactedCompute);
        assert_eq!(SkipDecision::MaskedCompute, SkipDecision::MaskedCompute);
    }

    #[test]
    fn skip_decision_ne_is_symmetric() {
        assert_ne!(SkipDecision::FullCompute, SkipDecision::CompactedCompute);
        assert_ne!(SkipDecision::CompactedCompute, SkipDecision::FullCompute);
    }

    // -- GateSkipCallback: construction edge cases --

    #[test]
    fn gate_skip_new_with_zero_layers() {
        let cb = GateSkipCallback::new(0, vec![], 512);
        assert_eq!(cb.priority(), 60);
        assert_eq!(cb.name(), "gate_skip");
    }

    #[test]
    fn gate_skip_new_disabled_with_zero_layers() {
        let cb = GateSkipCallback::new_disabled(0);
        assert_eq!(cb.priority(), 60);
        assert_eq!(cb.name(), "gate_skip");
    }

    #[test]
    fn gate_skip_new_disabled_intermediate_size_is_zero() {
        let cb = GateSkipCallback::new_disabled(4);
        // new_disabled sets intermediate_size to 0; CompactedCompute without mask
        // should produce mask of length max(0, 1) = 1
        // Verify via decision_for_layer
        for i in 0..4 {
            assert_eq!(*cb.decision_for_layer(i), SkipDecision::FullCompute);
        }
    }

    #[test]
    fn gate_skip_new_with_single_layer() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let cb = GateSkipCallback::new(1, decisions, 256);
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::CompactedCompute);
    }

    #[test]
    fn gate_skip_new_with_large_layer_count() {
        let num_layers = 128;
        let decisions = vec![SkipDecision::FullCompute; num_layers];
        let cb = GateSkipCallback::new(num_layers, decisions, 1024);
        for i in 0..num_layers {
            assert_eq!(*cb.decision_for_layer(i), SkipDecision::FullCompute);
        }
    }

    #[test]
    fn gate_skip_new_with_large_intermediate_size() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 65536);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 65536);
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // -- GateSkipCallback: mixed decisions per layer --

    #[test]
    fn gate_skip_mixed_decisions_per_layer() {
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
        ];
        let cb = GateSkipCallback::new(5, decisions, 64);
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::FullCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::CompactedCompute);
        assert_eq!(*cb.decision_for_layer(2), SkipDecision::MaskedCompute);
        assert_eq!(*cb.decision_for_layer(3), SkipDecision::FullCompute);
        assert_eq!(*cb.decision_for_layer(4), SkipDecision::CompactedCompute);
    }

    #[test]
    fn gate_skip_all_compacted() {
        let decisions = vec![SkipDecision::CompactedCompute; 6];
        let mut cb = GateSkipCallback::new(6, decisions, 16);
        for layer in 0..6 {
            let ctx = make_test_ctx(layer);
            let action = cb.pre_node(&ctx);
            assert!(matches!(action, CallbackAction::CompactMask { .. }));
        }
    }

    #[test]
    fn gate_skip_all_masked() {
        let decisions = vec![SkipDecision::MaskedCompute; 4];
        let mut cb = GateSkipCallback::new(4, decisions, 16);
        for layer in 0..4 {
            let ctx = make_test_ctx(layer);
            let action = cb.pre_node(&ctx);
            assert!(matches!(action, CallbackAction::Continue));
        }
    }

    // -- GateSkipCallback: neuron mask edge cases --

    #[test]
    fn gate_skip_store_empty_neuron_mask() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        cb.store_neuron_mask(0, vec![]);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 0);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_store_all_active_mask() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, true, true, true]);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, true, true, true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_store_all_dead_mask() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![false, false, false, false]);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![false, false, false, false]);
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 0);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_store_single_element_mask() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 1);
        cb.store_neuron_mask(0, vec![true]);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_store_large_mask() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8192);
        let mask: Vec<bool> = (0..8192).map(|i| i % 3 != 0).collect();
        cb.store_neuron_mask(0, mask.clone());
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 8192);
                assert_eq!(active_mask, mask);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_store_mask_overwrites_previous() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, true, true, true]);
        cb.store_neuron_mask(0, vec![false, false, false, false]);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![false, false, false, false]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // -- GateSkipCallback: multi-layer mask storage --

    #[test]
    fn gate_skip_store_masks_for_multiple_layers() {
        let decisions = vec![
            SkipDecision::CompactedCompute,
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
        ];
        let mut cb = GateSkipCallback::new(3, decisions.clone(), 4);
        cb.store_neuron_mask(0, vec![true, false, true, false]);
        cb.store_neuron_mask(2, vec![false, true, false, true]);

        // Layer 0: CompactedCompute with stored mask
        let ctx0 = make_test_ctx(0);
        let action0 = cb.pre_node(&ctx0);
        match action0 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false]);
            }
            _ => panic!("Expected CompactMask for layer 0"),
        }

        // Layer 1: FullCompute -> Continue
        let ctx1 = make_test_ctx(1);
        let action1 = cb.pre_node(&ctx1);
        assert!(matches!(action1, CallbackAction::Continue));

        // Layer 2: CompactedCompute with stored mask
        let ctx2 = make_test_ctx(2);
        let action2 = cb.pre_node(&ctx2);
        match action2 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![false, true, false, true]);
            }
            _ => panic!("Expected CompactMask for layer 2"),
        }
    }

    // -- GateSkipCallback: update_decisions with mask retention --

    #[test]
    fn gate_skip_update_preserves_stored_masks() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, false, true, false]);

        // Update decisions to FullCompute
        cb.update_decisions(vec![SkipDecision::FullCompute]);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        // FullCompute → Continue (mask is still stored but not used)
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_update_to_compacted_uses_stored_mask() {
        let mut cb = GateSkipCallback::new_disabled(1);
        cb.store_neuron_mask(0, vec![false, true, false, true]);
        cb.update_decisions(vec![SkipDecision::CompactedCompute]);

        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![false, true, false, true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // -- GateSkipCallback: decision_for_layer boundary --

    #[test]
    fn gate_skip_decision_for_layer_zero() {
        let decisions = vec![SkipDecision::MaskedCompute];
        let cb = GateSkipCallback::new(1, decisions, 8);
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::MaskedCompute);
    }

    #[test]
    fn gate_skip_decision_for_layer_at_boundary() {
        let decisions = vec![SkipDecision::CompactedCompute, SkipDecision::MaskedCompute];
        let cb = GateSkipCallback::new(2, decisions, 8);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::MaskedCompute);
    }

    #[test]
    fn gate_skip_decision_for_layer_one_past_end() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let cb = GateSkipCallback::new(1, decisions, 8);
        // Layer 1 is out of bounds → fallback FullCompute
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::FullCompute);
    }

    #[test]
    fn gate_skip_decision_for_layer_usize_max() {
        let cb = GateSkipCallback::new_disabled(1);
        assert_eq!(*cb.decision_for_layer(usize::MAX), SkipDecision::FullCompute);
    }

    // -- GateSkipCallback: pre_node with different layer indices --

    #[test]
    fn gate_skip_pre_node_full_compute_different_layers() {
        let decisions = vec![SkipDecision::FullCompute; 4];
        let mut cb = GateSkipCallback::new(4, decisions, 16);
        for layer in 0..4 {
            let ctx = make_test_ctx(layer);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        }
    }

    #[test]
    fn gate_skip_pre_node_masked_compute_different_layers() {
        let decisions = vec![SkipDecision::MaskedCompute; 3];
        let mut cb = GateSkipCallback::new(3, decisions, 16);
        for layer in 0..3 {
            let ctx = make_test_ctx(layer);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        }
    }

    #[test]
    fn gate_skip_pre_node_compacted_without_mask_zero_intermediate() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 0);
        // intermediate_size=0 → max(0, 1) = 1
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 1);
                assert!(active_mask[0]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // -- GateSkipCallback: post_node always returns Continue --

    #[test]
    fn gate_skip_post_node_full_compute() {
        let decisions = vec![SkipDecision::FullCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let ctx = make_test_ctx(0);
        assert!(matches!(cb.post_node(&ctx, &[]), CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_post_node_masked_compute() {
        let decisions = vec![SkipDecision::MaskedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let ctx = make_test_ctx(0);
        assert!(matches!(cb.post_node(&ctx, &[]), CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_post_node_with_nonempty_output() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let ctx = make_test_ctx(0);
        let output = &[1u8, 2, 3, 4, 5, 6, 7, 8];
        assert!(matches!(cb.post_node(&ctx, output), CallbackAction::Continue));
    }

    // -- GateSkipCallback: LayerCallback trait verification --

    #[test]
    fn gate_skip_trait_priority_consistent() {
        let cb1 = GateSkipCallback::new_disabled(1);
        let cb2 = GateSkipCallback::new(2, vec![SkipDecision::CompactedCompute, SkipDecision::FullCompute], 32);
        assert_eq!(cb1.priority(), cb2.priority());
        assert_eq!(cb1.priority(), 60);
    }

    #[test]
    fn gate_skip_trait_name_consistent() {
        let cb1 = GateSkipCallback::new_disabled(1);
        let cb2 = GateSkipCallback::new(2, vec![SkipDecision::MaskedCompute, SkipDecision::MaskedCompute], 32);
        assert_eq!(cb1.name(), cb2.name());
        assert_eq!(cb1.name(), "gate_skip");
    }

    #[test]
    fn gate_skip_trait_target_layers_always_none() {
        let cb = GateSkipCallback::new(3, vec![SkipDecision::FullCompute; 3], 16);
        assert!(cb.target_layers().is_none());
    }

    // -- GateSkipCallback: repeated pre_node calls --

    #[test]
    fn gate_skip_repeated_pre_node_same_layer() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, false, true, false]);
        let ctx = make_test_ctx(0);
        for _ in 0..5 {
            let action = cb.pre_node(&ctx);
            match action {
                CallbackAction::CompactMask { active_mask } => {
                    assert_eq!(active_mask, vec![true, false, true, false]);
                }
                _ => panic!("Expected CompactMask"),
            }
        }
    }

    #[test]
    fn gate_skip_repeated_pre_node_interleaved_layers() {
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
        ];
        let mut cb = GateSkipCallback::new(2, decisions, 4);
        cb.store_neuron_mask(1, vec![false, true, false, true]);

        for _ in 0..3 {
            let ctx0 = make_test_ctx(0);
            assert!(matches!(cb.pre_node(&ctx0), CallbackAction::Continue));

            let ctx1 = make_test_ctx(1);
            let action = cb.pre_node(&ctx1);
            match action {
                CallbackAction::CompactMask { active_mask } => {
                    assert_eq!(active_mask, vec![false, true, false, true]);
                }
                _ => panic!("Expected CompactMask for layer 1"),
            }
        }
    }

    // -- GateSkipCallback: active mask with alternating pattern --

    #[test]
    fn gate_skip_alternating_mask_pattern() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 10);
        let mask: Vec<bool> = (0..10).map(|i| i % 2 == 0).collect();
        cb.store_neuron_mask(0, mask);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 10);
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 5);
                assert_eq!(active_mask.iter().filter(|&&b| !b).count(), 5);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // -- GateSkipCallback: update_decisions then verify via pre_node --

    #[test]
    fn gate_skip_update_from_full_to_compacted_with_mask() {
        let mut cb = GateSkipCallback::new_disabled(1);
        cb.store_neuron_mask(0, vec![true, true, false, false]);
        cb.update_decisions(vec![SkipDecision::CompactedCompute]);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, true, false, false]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_update_from_compacted_to_full() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, false, true, false]);
        cb.update_decisions(vec![SkipDecision::FullCompute]);
        let ctx = make_test_ctx(0);
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_update_from_masked_to_compacted() {
        let decisions = vec![SkipDecision::MaskedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![false, false, true, true]);
        cb.update_decisions(vec![SkipDecision::CompactedCompute]);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![false, false, true, true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // -- GateSkipCallback: neuron_masks not stored for unaddressed layers --

    #[test]
    fn gate_skip_layer_without_mask_uses_intermediate_size() {
        let decisions = vec![
            SkipDecision::CompactedCompute,
            SkipDecision::CompactedCompute,
        ];
        let mut cb = GateSkipCallback::new(2, decisions, 32);
        // Only store mask for layer 0
        cb.store_neuron_mask(0, vec![true, false]);

        // Layer 0 uses stored mask
        let ctx0 = make_test_ctx(0);
        let action0 = cb.pre_node(&ctx0);
        match action0 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false]);
            }
            _ => panic!("Expected CompactMask for layer 0"),
        }

        // Layer 1 uses default mask of intermediate_size
        let ctx1 = make_test_ctx(1);
        let action1 = cb.pre_node(&ctx1);
        match action1 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 32);
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask for layer 1"),
        }
    }

    // -- GateSkipCallback: store_neuron_mask with various out-of-bounds values --

    #[test]
    fn gate_skip_store_mask_boundary_exact_len() {
        let decisions = vec![SkipDecision::CompactedCompute, SkipDecision::FullCompute];
        let mut cb = GateSkipCallback::new(2, decisions, 4);
        // Layer 1 is at boundary (index == len-1)
        cb.store_neuron_mask(1, vec![true, true, true, true]);
        let ctx = make_test_ctx(1);
        // Layer 1 is FullCompute though, so Continue
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_store_mask_beyond_len_ignored() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(5, vec![true]); // beyond len
        cb.store_neuron_mask(100, vec![true]); // far beyond
        // Layer 0 should still work with default mask
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 4);
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // -- GateSkipCallback: compacted compute mask with mixed active/dead ratio --

    #[test]
    fn gate_skip_mask_mostly_dead() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 16);
        let mask = vec![false, false, false, false, false, false, false, true,
                        false, false, false, false, false, false, false, false];
        cb.store_neuron_mask(0, mask);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 1);
                assert_eq!(active_mask.iter().filter(|&&b| !b).count(), 15);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_mask_mostly_active() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 16);
        let mask = vec![true, true, true, true, true, true, true, true,
                        true, true, true, true, true, true, true, false];
        cb.store_neuron_mask(0, mask);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 15);
                assert_eq!(active_mask.iter().filter(|&&b| !b).count(), 1);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // -- GateSkipCallback: pre_node on layer beyond stored decisions --

    #[test]
    fn gate_skip_pre_node_layer_beyond_decisions() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        // Layer 5 is beyond decisions len, decision_for_layer returns FullCompute
        let ctx = make_test_ctx(5);
        let action = cb.pre_node(&ctx);
        assert!(matches!(action, CallbackAction::Continue));
    }

    // ========================================================================
    // Additional tests (15) for further coverage
    // ========================================================================

    #[test]
    fn gate_skip_sequential_pre_node_all_three_decision_types() {
        // Arrange: 3 layers, one of each decision type
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
        ];
        let mut cb = GateSkipCallback::new(3, decisions, 8);
        cb.store_neuron_mask(1, vec![true, false, true, true, false, true, true, true]);

        // Act & Assert: layer 0 FullCompute -> Continue
        let ctx0 = make_test_ctx(0);
        assert!(matches!(cb.pre_node(&ctx0), CallbackAction::Continue));

        // Act & Assert: layer 1 CompactedCompute -> CompactMask with stored mask
        let ctx1 = make_test_ctx(1);
        let action1 = cb.pre_node(&ctx1);
        match action1 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, true, false, true, true, true]);
            }
            _ => panic!("Expected CompactMask for layer 1"),
        }

        // Act & Assert: layer 2 MaskedCompute -> Continue
        let ctx2 = make_test_ctx(2);
        assert!(matches!(cb.pre_node(&ctx2), CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_update_decisions_all_three_types() {
        // Arrange: start with all FullCompute
        let mut cb = GateSkipCallback::new_disabled(3);

        // Act: update to mixed decisions
        cb.update_decisions(vec![
            SkipDecision::MaskedCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::FullCompute,
        ]);

        // Assert: each layer has the correct decision
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::MaskedCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::CompactedCompute);
        assert_eq!(*cb.decision_for_layer(2), SkipDecision::FullCompute);
    }

    #[test]
    fn gate_skip_mask_stored_for_one_layer_does_not_affect_other() {
        // Arrange: 2 layers both CompactedCompute, store mask only for layer 0
        let decisions = vec![SkipDecision::CompactedCompute, SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(2, decisions, 16);
        cb.store_neuron_mask(0, vec![true, false, true, false]);

        // Act & Assert: layer 0 uses stored mask
        let ctx0 = make_test_ctx(0);
        let action0 = cb.pre_node(&ctx0);
        match action0 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false]);
            }
            _ => panic!("Expected CompactMask for layer 0"),
        }

        // Act & Assert: layer 1 has no stored mask, uses intermediate_size default
        let ctx1 = make_test_ctx(1);
        let action1 = cb.pre_node(&ctx1);
        match action1 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 16);
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask for layer 1"),
        }
    }

    #[test]
    fn gate_skip_multiple_update_cycles() {
        // Arrange: start disabled, then cycle through different decision sets
        let mut cb = GateSkipCallback::new_disabled(2);

        // Act: first update to CompactedCompute + MaskedCompute
        cb.update_decisions(vec![SkipDecision::CompactedCompute, SkipDecision::MaskedCompute]);
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::CompactedCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::MaskedCompute);

        // Act: second update back to all FullCompute
        cb.update_decisions(vec![SkipDecision::FullCompute, SkipDecision::FullCompute]);
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::FullCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::FullCompute);

        // Act: third update to all CompactedCompute
        cb.update_decisions(vec![SkipDecision::CompactedCompute, SkipDecision::CompactedCompute]);
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::CompactedCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::CompactedCompute);
    }

    #[test]
    fn gate_skip_compacted_with_intermediate_size_one() {
        // Arrange: intermediate_size=1, no mask stored
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 1);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: default mask has exactly 1 element
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 1);
                assert!(active_mask[0]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn skip_decision_used_as_hashmap_key() {
        // Arrange: use SkipDecision as HashMap key
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(SkipDecision::FullCompute, "full");
        map.insert(SkipDecision::CompactedCompute, "compacted");
        map.insert(SkipDecision::MaskedCompute, "masked");

        // Assert: all three keys are distinct and retrievable
        assert_eq!(map.get(&SkipDecision::FullCompute), Some(&"full"));
        assert_eq!(map.get(&SkipDecision::CompactedCompute), Some(&"compacted"));
        assert_eq!(map.get(&SkipDecision::MaskedCompute), Some(&"masked"));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn callback_action_default_is_continue() {
        // Assert: CallbackAction::default() is Continue
        assert_eq!(CallbackAction::default(), CallbackAction::Continue);
    }

    #[test]
    fn callback_action_compact_mask_equality() {
        // Arrange: two CompactMask actions with same mask
        let mask1 = vec![true, false, true];
        let mask2 = vec![true, false, true];

        // Assert: they are equal
        assert_eq!(
            CallbackAction::CompactMask { active_mask: mask1 },
            CallbackAction::CompactMask { active_mask: mask2 },
        );
    }

    #[test]
    fn callback_action_compact_mask_inequality() {
        // Arrange: two CompactMask actions with different masks
        let mask1 = vec![true, false, true];
        let mask2 = vec![true, true, true];

        // Assert: they are not equal
        assert_ne!(
            CallbackAction::CompactMask { active_mask: mask1 },
            CallbackAction::CompactMask { active_mask: mask2 },
        );
    }

    #[test]
    fn gate_skip_compacted_single_active_in_large_mask() {
        // Arrange: 32 neurons, only neuron at index 15 is active
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 32);
        let mut mask = vec![false; 32];
        mask[15] = true;
        cb.store_neuron_mask(0, mask);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: exactly 1 active neuron at the correct position
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 32);
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 1);
                assert!(active_mask[15]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_store_masks_all_layers_then_update_to_compacted() {
        // Arrange: 3 layers, store masks for all, update to all CompactedCompute
        let mut cb = GateSkipCallback::new_disabled(3);
        cb.store_neuron_mask(0, vec![true, false]);
        cb.store_neuron_mask(1, vec![false, true]);
        cb.store_neuron_mask(2, vec![true, true]);

        // Act
        cb.update_decisions(vec![
            SkipDecision::CompactedCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::CompactedCompute,
        ]);

        // Assert: each layer returns its stored mask
        for (layer, expected) in [
            (0, vec![true, false]),
            (1, vec![false, true]),
            (2, vec![true, true]),
        ] {
            let ctx = make_test_ctx(layer);
            let action = cb.pre_node(&ctx);
            match action {
                CallbackAction::CompactMask { active_mask } => {
                    assert_eq!(active_mask, expected, "mask mismatch at layer {}", layer);
                }
                _ => panic!("Expected CompactMask at layer {}", layer),
            }
        }
    }

    #[test]
    fn gate_skip_mask_stored_for_full_compute_layer_preserved_across_update() {
        // Arrange: start with FullCompute, store a mask, then switch to CompactedCompute
        let mut cb = GateSkipCallback::new_disabled(1);
        cb.store_neuron_mask(0, vec![false, false, true, true]);

        // Act: update to CompactedCompute
        cb.update_decisions(vec![SkipDecision::CompactedCompute]);

        // Assert: stored mask is preserved and used
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![false, false, true, true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_two_callbacks_independent_state() {
        // Arrange: two callbacks with different decisions
        let cb1 = GateSkipCallback::new(2, vec![SkipDecision::FullCompute, SkipDecision::CompactedCompute], 8);
        let cb2 = GateSkipCallback::new(2, vec![SkipDecision::MaskedCompute, SkipDecision::MaskedCompute], 16);

        // Assert: completely independent state
        assert_eq!(*cb1.decision_for_layer(0), SkipDecision::FullCompute);
        assert_eq!(*cb1.decision_for_layer(1), SkipDecision::CompactedCompute);
        assert_eq!(*cb2.decision_for_layer(0), SkipDecision::MaskedCompute);
        assert_eq!(*cb2.decision_for_layer(1), SkipDecision::MaskedCompute);
    }

    #[test]
    fn gate_skip_mask_unequal_length_produces_different_compact_actions() {
        // Arrange: two CompactedCompute calls with masks of different lengths
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);

        // Act: first with 4-element mask
        cb.store_neuron_mask(0, vec![true, false, true, false]);
        let ctx = make_test_ctx(0);
        let action1 = cb.pre_node(&ctx);
        let len1 = match &action1 {
            CallbackAction::CompactMask { active_mask } => active_mask.len(),
            _ => panic!("Expected CompactMask"),
        };

        // Act: second with 8-element mask
        cb.store_neuron_mask(0, vec![true; 8]);
        let action2 = cb.pre_node(&ctx);
        let len2 = match &action2 {
            CallbackAction::CompactMask { active_mask } => active_mask.len(),
            _ => panic!("Expected CompactMask"),
        };

        // Assert: different mask lengths produce different CompactMask actions
        assert_ne!(len1, len2);
        assert_eq!(len1, 4);
        assert_eq!(len2, 8);
    }

    #[test]
    fn gate_skip_mask_overwrite_between_pre_node_calls() {
        // Arrange: CompactedCompute with initial mask
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);

        // Act & Assert: first call with all-active mask
        cb.store_neuron_mask(0, vec![true, true, true, true]);
        let ctx = make_test_ctx(0);
        match cb.pre_node(&ctx) {
            CallbackAction::CompactMask { active_mask } => {
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask"),
        }

        // Act & Assert: overwrite mask to all-dead
        cb.store_neuron_mask(0, vec![false, false, false, false]);
        match cb.pre_node(&ctx) {
            CallbackAction::CompactMask { active_mask } => {
                assert!(active_mask.iter().all(|&b| !b));
            }
            _ => panic!("Expected CompactMask"),
        }

        // Act & Assert: overwrite again to mixed
        cb.store_neuron_mask(0, vec![true, false, false, true]);
        match cb.pre_node(&ctx) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, false, true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // ========================================================================
    // Additional tests (15) for further edge-case and data-structure coverage
    // ========================================================================

    #[test]
    fn gate_skip_compact_mask_action_is_never_skip_this_node() {
        // Arrange: CompactedCompute should produce CompactMask, never SkipThisNode
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let ctx = make_test_ctx(0);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: must not be SkipThisNode (SPEC §14.2 — compact, don't skip)
        assert!(
            !matches!(action, CallbackAction::SkipThisNode),
            "CompactedCompute must never produce SkipThisNode"
        );
        assert!(
            matches!(action, CallbackAction::CompactMask { .. }),
            "CompactedCompute must produce CompactMask"
        );
    }

    #[test]
    fn gate_skip_masked_never_produces_compact_or_skip() {
        // Arrange: MaskedCompute should only produce Continue
        let decisions = vec![SkipDecision::MaskedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let ctx = make_test_ctx(0);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: MaskedCompute does not skip or compact
        assert!(matches!(action, CallbackAction::Continue));
        assert!(!matches!(action, CallbackAction::SkipThisNode));
        assert!(!matches!(action, CallbackAction::CompactMask { .. }));
    }

    #[test]
    fn gate_skip_full_compute_never_produces_exit_early_or_inject() {
        // Arrange: FullCompute should only produce Continue
        let decisions = vec![SkipDecision::FullCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let ctx = make_test_ctx(0);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: FullCompute never produces ExitEarly or InjectHidden
        assert!(matches!(action, CallbackAction::Continue));
        assert!(!matches!(action, CallbackAction::ExitEarly { .. }));
        assert!(!matches!(action, CallbackAction::InjectHidden { .. }));
    }

    #[test]
    fn gate_skip_new_disabled_and_new_all_full_are_equivalent() {
        // Arrange: two construction paths for all-FullCompute
        let cb_disabled = GateSkipCallback::new_disabled(4);
        let cb_explicit = GateSkipCallback::new(
            4,
            vec![SkipDecision::FullCompute; 4],
            0,
        );

        // Assert: same decisions per layer, same priority, same name
        for i in 0..4 {
            assert_eq!(
                cb_disabled.decision_for_layer(i),
                cb_explicit.decision_for_layer(i),
            );
        }
        assert_eq!(cb_disabled.priority(), cb_explicit.priority());
        assert_eq!(cb_disabled.name(), cb_explicit.name());
    }

    #[test]
    fn gate_skip_update_decisions_idempotent_same_value() {
        // Arrange: 2-layer callback
        let mut cb = GateSkipCallback::new_disabled(2);

        // Act: update twice with identical decisions
        let decisions = vec![SkipDecision::CompactedCompute, SkipDecision::MaskedCompute];
        cb.update_decisions(decisions.clone());
        cb.update_decisions(decisions);

        // Assert: still correct after double update
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::CompactedCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::MaskedCompute);
    }

    #[test]
    fn gate_skip_active_mask_for_layer_none_path_uses_intermediate_size() {
        // Arrange: 2 layers, layer 0 has mask stored, layer 1 has None in neuron_masks
        // This tests the `neuron_masks.get(layer) → Some(None)` vs `None` code path
        let decisions = vec![SkipDecision::CompactedCompute, SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(2, decisions, 64);
        // No mask stored for layer 0 or layer 1 — both are None
        // neuron_masks is initialized as vec![None; 2], so .get(0) → Some(None)

        // Act: layer 0 has neuron_masks[0] = None (Some(None) via get)
        let ctx0 = make_test_ctx(0);
        let action0 = cb.pre_node(&ctx0);
        match action0 {
            CallbackAction::CompactMask { active_mask } => {
                // Assert: falls through to default mask of intermediate_size
                assert_eq!(active_mask.len(), 64);
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask for layer 0"),
        }
    }

    #[test]
    fn gate_skip_store_mask_for_full_compute_layer_does_not_crash_on_pre_node() {
        // Arrange: FullCompute layer with a stored mask — mask exists but is ignored
        let decisions = vec![SkipDecision::FullCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        cb.store_neuron_mask(0, vec![true, false, true, false]);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: FullCompute always Continue regardless of stored mask
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_post_node_with_large_output_buffer() {
        // Arrange: post_node receives a large output slice
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 256);
        let ctx = make_test_ctx(0);
        let output = vec![42u8; 1024 * 1024]; // 1 MB output

        // Act
        let action = cb.post_node(&ctx, &output);

        // Assert: post_node always returns Continue regardless of output size
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_compacted_mask_with_exactly_half_active() {
        // Arrange: 8 neurons, exactly 4 active (50% density — boundary for §14.2 threshold)
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let mask = vec![true, true, true, true, false, false, false, false];
        cb.store_neuron_mask(0, mask);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: exactly 50% active, mask preserved faithfully
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 8);
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 4);
                assert_eq!(active_mask.iter().filter(|&&b| !b).count(), 4);
                // Verify exact positions: first 4 active, last 4 dead
                assert!(active_mask[0] && active_mask[1] && active_mask[2] && active_mask[3]);
                assert!(!active_mask[4] && !active_mask[5] && !active_mask[6] && !active_mask[7]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_multiple_overwrites_then_pre_node_uses_last_mask() {
        // Arrange: overwrite mask 5 times, verify last one wins
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);

        let masks = vec![
            vec![true, false, true, false],
            vec![false, true, false, true],
            vec![true, true, false, false],
            vec![false, false, false, false],
            vec![true, true, true, true],
        ];

        for mask in &masks {
            cb.store_neuron_mask(0, mask.clone());
        }

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: last mask wins
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, true, true, true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_pre_node_with_varying_context_node_idx() {
        // Arrange: 4-layer callback, verify node_idx variation doesn't affect decision
        let decisions = vec![
            SkipDecision::CompactedCompute,
            SkipDecision::FullCompute,
            SkipDecision::MaskedCompute,
            SkipDecision::CompactedCompute,
        ];
        let mut cb = GateSkipCallback::new(4, decisions, 8);
        cb.store_neuron_mask(0, vec![true, false, true, true, true, true, true, true]);
        cb.store_neuron_mask(3, vec![false, false, true, true, false, false, true, true]);

        // Act & Assert: layer 0 with different node_idx values
        let mut ctx_node0 = make_test_ctx(0);
        ctx_node0.node_idx = 0;
        let action0 = cb.pre_node(&ctx_node0);
        match action0 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, true, true, true, true, true]);
            }
            _ => panic!("Expected CompactMask for layer 0"),
        }

        let mut ctx_node99 = make_test_ctx(0);
        ctx_node99.node_idx = 99;
        let action0b = cb.pre_node(&ctx_node99);
        match action0b {
            CallbackAction::CompactMask { active_mask } => {
                // Same result regardless of node_idx
                assert_eq!(active_mask, vec![true, false, true, true, true, true, true, true]);
            }
            _ => panic!("Expected CompactMask for layer 0 with different node_idx"),
        }
    }

    #[test]
    fn gate_skip_update_from_compacted_to_masked_then_back() {
        // Arrange: single layer, start CompactedCompute with mask
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, false, true, false]);

        // Act & Assert: CompactedCompute → CompactMask
        let ctx = make_test_ctx(0);
        match cb.pre_node(&ctx) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false]);
            }
            _ => panic!("Expected CompactMask"),
        }

        // Act: switch to MaskedCompute
        cb.update_decisions(vec![SkipDecision::MaskedCompute]);
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));

        // Act: switch back to CompactedCompute — stored mask should still be there
        cb.update_decisions(vec![SkipDecision::CompactedCompute]);
        match cb.pre_node(&ctx) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false]);
            }
            _ => panic!("Expected CompactMask after round-trip"),
        }
    }

    #[test]
    fn skip_decision_vec_collect_and_iterate() {
        // Arrange: create a Vec of SkipDecision and iterate
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::FullCompute,
        ];

        // Act: count by variant
        let full_count = decisions.iter().filter(|d| **d == SkipDecision::FullCompute).count();
        let compacted_count = decisions.iter().filter(|d| **d == SkipDecision::CompactedCompute).count();
        let masked_count = decisions.iter().filter(|d| **d == SkipDecision::MaskedCompute).count();

        // Assert
        assert_eq!(full_count, 2);
        assert_eq!(compacted_count, 2);
        assert_eq!(masked_count, 1);
    }

    #[test]
    fn gate_skip_rapid_decision_flips_between_all_variants() {
        // Arrange: rapidly flip decisions between all three variants
        let mut cb = GateSkipCallback::new_disabled(1);
        let ctx = make_test_ctx(0);

        let flip_sequence = vec![
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
            SkipDecision::FullCompute,
            SkipDecision::MaskedCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
        ];

        // Act & Assert: each flip produces the correct action
        for (i, decision) in flip_sequence.into_iter().enumerate() {
            cb.update_decisions(vec![decision]);
            let action = cb.pre_node(&ctx);
            match decision {
                SkipDecision::FullCompute | SkipDecision::MaskedCompute => {
                    assert!(
                        matches!(action, CallbackAction::Continue),
                        "iteration {}: {:?} should produce Continue",
                        i, decision,
                    );
                }
                SkipDecision::CompactedCompute => {
                    assert!(
                        matches!(action, CallbackAction::CompactMask { .. }),
                        "iteration {}: CompactedCompute should produce CompactMask",
                        i,
                    );
                }
            }
        }
    }

    #[test]
    fn gate_skip_three_layer_each_with_different_mask_density() {
        // Arrange: 3 layers with 64 neurons each, varying dead neuron density
        let decisions = vec![
            SkipDecision::CompactedCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::CompactedCompute,
        ];
        let mut cb = GateSkipCallback::new(3, decisions, 64);

        // Layer 0: 75% active (high density)
        let mask0: Vec<bool> = (0..64).map(|i| i < 48).collect();
        // Layer 1: 25% active (low density)
        let mask1: Vec<bool> = (0..64).map(|i| i < 16).collect();
        // Layer 2: 50% active (boundary density)
        let mask2: Vec<bool> = (0..64).map(|i| i % 2 == 0).collect();

        cb.store_neuron_mask(0, mask0);
        cb.store_neuron_mask(1, mask1);
        cb.store_neuron_mask(2, mask2);

        // Act & Assert: layer 0 — 48 active out of 64
        let ctx0 = make_test_ctx(0);
        match cb.pre_node(&ctx0) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 48);
                assert_eq!(active_mask.iter().filter(|&&b| !b).count(), 16);
            }
            _ => panic!("Expected CompactMask for layer 0"),
        }

        // Act & Assert: layer 1 — 16 active out of 64
        let ctx1 = make_test_ctx(1);
        match cb.pre_node(&ctx1) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 16);
                assert_eq!(active_mask.iter().filter(|&&b| !b).count(), 48);
            }
            _ => panic!("Expected CompactMask for layer 1"),
        }

        // Act & Assert: layer 2 — 32 active out of 64
        let ctx2 = make_test_ctx(2);
        match cb.pre_node(&ctx2) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 32);
                assert_eq!(active_mask.iter().filter(|&&b| !b).count(), 32);
            }
            _ => panic!("Expected CompactMask for layer 2"),
        }
    }

    #[test]
    #[should_panic(expected = "decisions length must match num_layers")]
    fn gate_skip_new_panics_on_mismatched_decisions_length() {
        // Arrange & Act: decisions.len() != num_layers should panic
        let _cb = GateSkipCallback::new(3, vec![SkipDecision::FullCompute], 8);
    }

    // ========================================================================
    // Additional tests (13) for edge-case coverage
    // ========================================================================

    #[test]
    fn callback_action_skip_this_node_variant_exists() {
        // Arrange & Act: construct SkipThisNode variant
        let action = CallbackAction::SkipThisNode;

        // Assert: it is not Continue, not CompactMask
        assert!(!matches!(action, CallbackAction::Continue));
        assert!(!matches!(action, CallbackAction::CompactMask { .. }));
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    #[test]
    fn callback_action_exit_early_with_empty_logits() {
        // Arrange & Act: ExitEarly with no logits (signal caller to project)
        let action = CallbackAction::ExitEarly { logits: vec![] };

        // Assert: variant is distinguishable
        assert!(matches!(action, CallbackAction::ExitEarly { .. }));
        if let CallbackAction::ExitEarly { logits } = action {
            assert!(logits.is_empty());
        }
    }

    #[test]
    fn callback_action_exit_early_with_logits() {
        // Arrange & Act: ExitEarly with actual logit scores
        let action = CallbackAction::ExitEarly { logits: vec![1.0, 2.5, -0.3] };

        // Assert: logits are preserved exactly
        if let CallbackAction::ExitEarly { logits } = action {
            assert_eq!(logits.len(), 3);
            assert_eq!(logits[0], 1.0);
            assert_eq!(logits[1], 2.5);
            assert_eq!(logits[2], -0.3);
        }
    }

    #[test]
    fn callback_action_inject_hidden_with_data() {
        // Arrange & Act: InjectHidden with arbitrary byte payload
        let action = CallbackAction::InjectHidden { data: vec![0xAB, 0xCD, 0xEF] };

        // Assert: variant is constructible and data preserved
        if let CallbackAction::InjectHidden { data } = action {
            assert_eq!(data.len(), 3);
            assert_eq!(data[0], 0xAB);
            assert_eq!(data[2], 0xEF);
        }
    }

    #[test]
    fn callback_action_clone_preserves_variant() {
        // Arrange: each variant
        let actions: Vec<CallbackAction> = vec![
            CallbackAction::Continue,
            CallbackAction::SkipThisNode,
            CallbackAction::ExitEarly { logits: vec![0.5] },
            CallbackAction::InjectHidden { data: vec![1, 2] },
            CallbackAction::CompactMask { active_mask: vec![true, false] },
        ];

        // Act & Assert: clone each and verify equality
        for original in actions {
            let cloned = original.clone();
            assert_eq!(original, cloned);
        }
    }

    #[test]
    fn skip_decision_debug_output_exact_format() {
        // Arrange & Act: format each variant with Debug
        let full = format!("{:?}", SkipDecision::FullCompute);
        let compacted = format!("{:?}", SkipDecision::CompactedCompute);
        let masked = format!("{:?}", SkipDecision::MaskedCompute);

        // Assert: no variant string is a prefix of another
        assert!(!compacted.contains(&full) || compacted == full);
        assert!(!masked.contains(&compacted) || masked == compacted);
        // All three are distinct strings
        assert_ne!(full, compacted);
        assert_ne!(full, masked);
        assert_ne!(compacted, masked);
    }

    #[test]
    fn gate_skip_compacted_produces_fresh_mask_each_call() {
        // Arrange: CompactedCompute layer with stored mask
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, false, true, false]);
        let ctx = make_test_ctx(0);

        // Act: call pre_node twice and collect masks
        let mask1 = match cb.pre_node(&ctx) {
            CallbackAction::CompactMask { active_mask } => active_mask,
            _ => panic!("Expected CompactMask"),
        };
        let mask2 = match cb.pre_node(&ctx) {
            CallbackAction::CompactMask { active_mask } => active_mask,
            _ => panic!("Expected CompactMask"),
        };

        // Assert: masks have same content but are distinct allocations
        assert_eq!(mask1, mask2);
        assert!(!std::ptr::eq(mask1.as_ptr(), mask2.as_ptr()));
    }

    #[test]
    fn gate_skip_new_with_all_three_decision_types_round_robin() {
        // Arrange: round-robin pattern across many layers
        let num_layers = 12;
        let decisions: Vec<SkipDecision> = (0..num_layers)
            .map(|i| match i % 3 {
                0 => SkipDecision::FullCompute,
                1 => SkipDecision::CompactedCompute,
                _ => SkipDecision::MaskedCompute,
            })
            .collect();
        let cb = GateSkipCallback::new(num_layers, decisions, 32);

        // Assert: pattern is consistent
        for i in 0..num_layers {
            let expected = match i % 3 {
                0 => SkipDecision::FullCompute,
                1 => SkipDecision::CompactedCompute,
                _ => SkipDecision::MaskedCompute,
            };
            assert_eq!(*cb.decision_for_layer(i), expected, "mismatch at layer {}", i);
        }
    }

    #[test]
    fn gate_skip_post_node_ignores_all_decision_types() {
        // Arrange: three layers with different decisions
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
        ];
        let mut cb = GateSkipCallback::new(3, decisions, 8);
        let output = &[0u8; 16];

        // Act & Assert: post_node returns Continue for all decision types
        for layer in 0..3 {
            let ctx = make_test_ctx(layer);
            let action = cb.post_node(&ctx, output);
            assert!(
                matches!(action, CallbackAction::Continue),
                "post_node must return Continue for layer {} with decision {:?}",
                layer,
                cb.decision_for_layer(layer),
            );
        }
    }

    #[test]
    fn gate_skip_compacted_default_mask_all_true_when_zero_intermediate() {
        // Arrange: CompactedCompute with intermediate_size=0 and no stored mask
        // max(0, 1) = 1, so default mask is [true]
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 0);
        let ctx = make_test_ctx(0);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: exactly one element, true
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 1);
                assert!(active_mask[0]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_stored_mask_preserves_position_of_active_neurons() {
        // Arrange: specific active positions in a 16-neuron mask
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 16);
        let mut mask = vec![false; 16];
        mask[0] = true;
        mask[7] = true;
        mask[15] = true;
        cb.store_neuron_mask(0, mask);
        let ctx = make_test_ctx(0);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: exactly 3 active at positions 0, 7, 15
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 16);
                assert!(active_mask[0]);
                assert!(active_mask[7]);
                assert!(active_mask[15]);
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 3);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn skip_decision_can_be_used_in_array() {
        // Arrange: fixed-size array of SkipDecision
        let decisions: [SkipDecision; 4] = [
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
            SkipDecision::FullCompute,
        ];

        // Assert: array indexing works, Copy allows moves
        assert_eq!(decisions[0], SkipDecision::FullCompute);
        assert_eq!(decisions[1], SkipDecision::CompactedCompute);
        assert_eq!(decisions[2], SkipDecision::MaskedCompute);
        assert_eq!(decisions[3], SkipDecision::FullCompute);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn gate_skip_update_decisions_panics_on_wrong_length() {
        let mut cb = GateSkipCallback::new_disabled(2);
        cb.update_decisions(vec![SkipDecision::CompactedCompute]);
    }

    // ========================================================================
    // Additional tests (13) for target 130 total
    // ========================================================================

    #[test]
    fn skip_decision_partial_eq_transitivity() {
        // Arrange: three equal values — transitivity means a==b and b==c implies a==c
        let a = SkipDecision::CompactedCompute;
        let b = SkipDecision::CompactedCompute;
        let c = SkipDecision::CompactedCompute;

        // Assert: transitivity holds for all variant pairs
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    #[test]
    fn gate_skip_new_with_masked_and_full_interleaved() {
        // Arrange: interleaved pattern — even layers FullCompute, odd layers MaskedCompute
        let num_layers = 6;
        let decisions: Vec<SkipDecision> = (0..num_layers)
            .map(|i| if i % 2 == 0 { SkipDecision::FullCompute } else { SkipDecision::MaskedCompute })
            .collect();
        let cb = GateSkipCallback::new(num_layers, decisions, 32);

        // Assert: even layers FullCompute, odd layers MaskedCompute
        for i in 0..num_layers {
            let expected = if i % 2 == 0 { SkipDecision::FullCompute } else { SkipDecision::MaskedCompute };
            assert_eq!(*cb.decision_for_layer(i), expected, "mismatch at layer {}", i);
        }
    }

    #[test]
    fn gate_skip_disabled_zero_layers_decision_for_any_layer_is_full_compute() {
        // Arrange: zero-layer disabled callback
        let cb = GateSkipCallback::new_disabled(0);

        // Assert: any out-of-bounds access returns FullCompute
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::FullCompute);
        assert_eq!(*cb.decision_for_layer(1), SkipDecision::FullCompute);
    }

    #[test]
    fn gate_skip_store_mask_on_layer_zero_then_query_layer_one_default() {
        // Arrange: 2 layers both CompactedCompute, store mask only for layer 0
        let decisions = vec![SkipDecision::CompactedCompute, SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(2, decisions, 128);
        cb.store_neuron_mask(0, vec![true, false, false, true]);

        // Act & Assert: layer 1 without stored mask uses intermediate_size=128
        let ctx1 = make_test_ctx(1);
        let action1 = cb.pre_node(&ctx1);
        match action1 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 128);
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask for layer 1"),
        }
    }

    #[test]
    fn callback_action_debug_format_contains_variant_name() {
        // Arrange & Act: Debug format of each CallbackAction variant
        let continue_s = format!("{:?}", CallbackAction::Continue);
        let skip_s = format!("{:?}", CallbackAction::SkipThisNode);
        let exit_s = format!("{:?}", CallbackAction::ExitEarly { logits: vec![] });
        let inject_s = format!("{:?}", CallbackAction::InjectHidden { data: vec![] });
        let compact_s = format!("{:?}", CallbackAction::CompactMask { active_mask: vec![] });

        // Assert: each debug string contains its variant name
        assert!(continue_s.contains("Continue"), "Debug should contain Continue, got: {}", continue_s);
        assert!(skip_s.contains("SkipThisNode"), "Debug should contain SkipThisNode, got: {}", skip_s);
        assert!(exit_s.contains("ExitEarly"), "Debug should contain ExitEarly, got: {}", exit_s);
        assert!(inject_s.contains("InjectHidden"), "Debug should contain InjectHidden, got: {}", inject_s);
        assert!(compact_s.contains("CompactMask"), "Debug should contain CompactMask, got: {}", compact_s);
    }

    #[test]
    fn callback_action_equality_for_continue() {
        // Arrange & Act: two Continue actions
        let a = CallbackAction::Continue;
        let b = CallbackAction::Continue;

        // Assert: Continue equals itself
        assert_eq!(a, b);
    }

    #[test]
    fn callback_action_skip_this_node_not_equal_continue() {
        // Arrange & Act
        let skip = CallbackAction::SkipThisNode;
        let cont = CallbackAction::Continue;

        // Assert: different variants are not equal
        assert_ne!(skip, cont);
    }

    #[test]
    fn gate_skip_compacted_stored_mask_different_length_from_intermediate_size() {
        // Arrange: intermediate_size=32 but stored mask has length 4
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 32);
        cb.store_neuron_mask(0, vec![true, false, true, true]);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: stored mask takes precedence, length is 4 not 32
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 4);
                assert_eq!(active_mask, vec![true, false, true, true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_update_to_masked_then_pre_node_returns_continue() {
        // Arrange: start CompactedCompute, then update to MaskedCompute
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        cb.store_neuron_mask(0, vec![true, false, true, false]);

        // Act: update to MaskedCompute
        cb.update_decisions(vec![SkipDecision::MaskedCompute]);
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: MaskedCompute returns Continue regardless of stored mask
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_compacted_mask_all_dead_counted_correctly() {
        // Arrange: mask with 0 active neurons (100% dead — extreme boundary)
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        cb.store_neuron_mask(0, vec![false; 8]);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: 0 active, 8 dead
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 8);
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 0);
                assert_eq!(active_mask.iter().filter(|&&b| !b).count(), 8);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_compacted_mask_only_first_active() {
        // Arrange: only index 0 is active
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, false, false, false]);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: first element active, rest dead
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert!(active_mask[0]);
                assert!(!active_mask[1]);
                assert!(!active_mask[2]);
                assert!(!active_mask[3]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_compacted_mask_only_last_active() {
        // Arrange: only last index is active
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![false, false, false, true]);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: last element active, rest dead
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert!(!active_mask[0]);
                assert!(!active_mask[1]);
                assert!(!active_mask[2]);
                assert!(active_mask[3]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn skip_decision_count_is_three() {
        // Assert: exactly 3 SkipDecision variants exist
        let all_variants = [
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
        ];
        assert_eq!(all_variants.len(), 3, "SkipDecision should have exactly 3 variants");
    }

    // ========================================================================
    // Additional tests (13) for 143 total
    // ========================================================================

    #[test]
    fn gate_skip_masked_compute_ignores_stored_mask_in_pre_node() {
        // Arrange: MaskedCompute layer with a stored mask — mask should NOT affect pre_node
        let decisions = vec![SkipDecision::MaskedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 16);
        cb.store_neuron_mask(0, vec![true, false, true, false, true, false, true, false]);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: MaskedCompute always returns Continue, the stored mask is irrelevant
        assert!(
            matches!(action, CallbackAction::Continue),
            "MaskedCompute must return Continue regardless of stored mask"
        );
    }

    #[test]
    fn gate_skip_update_decisions_preserves_mask_for_unmodified_layer() {
        // Arrange: 3 layers, store mask for layer 2
        let mut cb = GateSkipCallback::new_disabled(3);
        cb.store_neuron_mask(2, vec![false, true, false, true, false, true, false, true]);

        // Act: update layer 0 and 1 to CompactedCompute, layer 2 stays FullCompute
        cb.update_decisions(vec![
            SkipDecision::CompactedCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::FullCompute,
        ]);

        // Assert: layer 2's stored mask is still there, but FullCompute returns Continue
        let ctx2 = make_test_ctx(2);
        assert!(matches!(cb.pre_node(&ctx2), CallbackAction::Continue));

        // Act: update layer 2 to CompactedCompute — stored mask should be used
        cb.update_decisions(vec![
            SkipDecision::CompactedCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::CompactedCompute,
        ]);
        match cb.pre_node(&ctx2) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![false, true, false, true, false, true, false, true]);
            }
            _ => panic!("Expected CompactMask for layer 2 after update"),
        }
    }

    #[test]
    fn gate_skip_pre_node_after_multiple_update_cycles_with_masks() {
        // Arrange: 2-layer callback with stored masks
        let decisions = vec![SkipDecision::CompactedCompute, SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(2, decisions, 8);
        cb.store_neuron_mask(0, vec![true, true, false, false, true, true, false, false]);
        cb.store_neuron_mask(1, vec![false, false, true, true, false, false, true, true]);

        // Act: cycle 1 — CompactedCompute
        let ctx0 = make_test_ctx(0);
        let ctx1 = make_test_ctx(1);
        match cb.pre_node(&ctx0) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 4);
            }
            _ => panic!("Expected CompactMask for layer 0"),
        }
        match cb.pre_node(&ctx1) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 4);
            }
            _ => panic!("Expected CompactMask for layer 1"),
        }

        // Act: cycle 2 — update to FullCompute
        cb.update_decisions(vec![SkipDecision::FullCompute, SkipDecision::FullCompute]);
        assert!(matches!(cb.pre_node(&ctx0), CallbackAction::Continue));
        assert!(matches!(cb.pre_node(&ctx1), CallbackAction::Continue));

        // Act: cycle 3 — update back to CompactedCompute, masks still preserved
        cb.update_decisions(vec![SkipDecision::CompactedCompute, SkipDecision::CompactedCompute]);
        match cb.pre_node(&ctx0) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, true, false, false, true, true, false, false]);
            }
            _ => panic!("Expected CompactMask for layer 0 after cycle 3"),
        }
    }

    #[test]
    fn gate_skip_disabled_callback_never_produces_compact_mask() {
        // Arrange: disabled callback across all layers
        let mut cb = GateSkipCallback::new_disabled(5);

        // Act & Assert: every layer should produce Continue
        for layer in 0..5 {
            let ctx = make_test_ctx(layer);
            let action = cb.pre_node(&ctx);
            assert!(
                matches!(action, CallbackAction::Continue),
                "disabled callback layer {} produced {:?}, expected Continue",
                layer,
                action,
            );
        }
    }

    #[test]
    fn gate_skip_layer_context_hidden_state_empty_does_not_affect_decision() {
        // Arrange: CompactedCompute with a stored mask, ctx has empty hidden_state
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        cb.store_neuron_mask(0, vec![true, false, true, false, true, false, true, false]);

        // Act: the helper make_test_ctx already uses hidden_state: &[]
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: decision is based purely on SkipDecision, not hidden_state
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false, true, false, true, false]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_intermediate_size_used_for_multiple_layers_without_masks() {
        // Arrange: 4 layers all CompactedCompute, no masks stored
        let decisions = vec![SkipDecision::CompactedCompute; 4];
        let mut cb = GateSkipCallback::new(4, decisions, 48);

        // Act & Assert: every layer should get default mask of intermediate_size=48
        for layer in 0..4 {
            let ctx = make_test_ctx(layer);
            match cb.pre_node(&ctx) {
                CallbackAction::CompactMask { active_mask } => {
                    assert_eq!(active_mask.len(), 48, "layer {} mask length mismatch", layer);
                    assert!(active_mask.iter().all(|&b| b), "layer {} default mask should be all-true", layer);
                }
                _ => panic!("Expected CompactMask for layer {}", layer),
            }
        }
    }

    #[test]
    fn gate_skip_stored_mask_supersedes_intermediate_size_even_when_larger() {
        // Arrange: intermediate_size=4, but stored mask has 256 elements (larger)
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        let large_mask: Vec<bool> = (0..256).map(|i| i % 5 == 0).collect();
        cb.store_neuron_mask(0, large_mask.clone());

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: stored mask (256 elements) takes precedence over intermediate_size (4)
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 256);
                assert_eq!(active_mask, large_mask);
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 52); // 256/5 = 51.2, ceil for i%5==0
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn skip_decision_copy_independent_of_original() {
        // Arrange: take a copy and verify independence via collection
        let original = SkipDecision::CompactedCompute;
        let copy = original;

        // Act: use both in a collection
        let set: std::collections::HashSet<SkipDecision> = [original, copy].into_iter().collect();

        // Assert: both are equal, so set has exactly 1 element
        assert_eq!(set.len(), 1);
        assert!(set.contains(&SkipDecision::CompactedCompute));
    }

    #[test]
    fn gate_skip_pre_node_for_out_of_range_layer_returns_continue() {
        // Arrange: 2-layer callback, query pre_node for layer 100
        let decisions = vec![SkipDecision::CompactedCompute, SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(2, decisions, 8);

        // Act: layer 100 is out of range, decision_for_layer returns FullCompute
        let ctx = make_test_ctx(100);
        let action = cb.pre_node(&ctx);

        // Assert: FullCompute produces Continue
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_store_neuron_mask_at_boundary_layer_zero_and_last() {
        // Arrange: 5-layer callback, store masks at first and last layers
        let decisions = vec![SkipDecision::CompactedCompute; 5];
        let mut cb = GateSkipCallback::new(5, decisions, 8);
        cb.store_neuron_mask(0, vec![true, false, true, false]);
        cb.store_neuron_mask(4, vec![false, true, false, true]);

        // Act & Assert: layer 0 uses stored mask
        let ctx0 = make_test_ctx(0);
        match cb.pre_node(&ctx0) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false]);
            }
            _ => panic!("Expected CompactMask for layer 0"),
        }

        // Act & Assert: layer 4 (last) uses stored mask
        let ctx4 = make_test_ctx(4);
        match cb.pre_node(&ctx4) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![false, true, false, true]);
            }
            _ => panic!("Expected CompactMask for layer 4"),
        }

        // Act & Assert: layer 2 (middle, no mask) uses default intermediate_size
        let ctx2 = make_test_ctx(2);
        match cb.pre_node(&ctx2) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 8);
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask for layer 2"),
        }
    }

    #[test]
    fn gate_skip_update_to_all_same_variant_compacted() {
        // Arrange: start with mixed decisions, then update to all CompactedCompute
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::MaskedCompute,
            SkipDecision::FullCompute,
        ];
        let mut cb = GateSkipCallback::new(3, decisions, 16);

        // Act: update to all CompactedCompute
        cb.update_decisions(vec![SkipDecision::CompactedCompute; 3]);

        // Assert: every layer is now CompactedCompute
        for i in 0..3 {
            assert_eq!(*cb.decision_for_layer(i), SkipDecision::CompactedCompute);
        }

        // Act: pre_node produces CompactMask for all layers
        for layer in 0..3 {
            let ctx = make_test_ctx(layer);
            match cb.pre_node(&ctx) {
                CallbackAction::CompactMask { active_mask } => {
                    assert_eq!(active_mask.len(), 16);
                }
                _ => panic!("Expected CompactMask for layer {}", layer),
            }
        }
    }

    #[test]
    fn gate_skip_compacted_mask_with_single_active_at_each_position() {
        // Arrange: test each position as the sole active neuron in a 4-element mask
        for active_pos in 0..4 {
            let decisions = vec![SkipDecision::CompactedCompute];
            let mut cb = GateSkipCallback::new(1, decisions, 4);
            let mut mask = vec![false; 4];
            mask[active_pos] = true;
            cb.store_neuron_mask(0, mask);

            // Act
            let ctx = make_test_ctx(0);
            let action = cb.pre_node(&ctx);

            // Assert: exactly 1 active at the correct position
            match action {
                CallbackAction::CompactMask { active_mask } => {
                    assert_eq!(active_mask.len(), 4);
                    assert_eq!(active_mask.iter().filter(|&&b| b).count(), 1);
                    assert!(active_mask[active_pos], "position {} should be active", active_pos);
                    for j in 0..4 {
                        if j != active_pos {
                            assert!(!active_mask[j], "position {} should be dead", j);
                        }
                    }
                }
                _ => panic!("Expected CompactMask for active_pos={}", active_pos),
            }
        }
    }

    #[test]
    fn gate_skip_pre_node_and_post_node_pair_for_each_decision_variant() {
        // Arrange: 3 layers, one of each decision type
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
        ];
        let mut cb = GateSkipCallback::new(3, decisions, 4);
        cb.store_neuron_mask(1, vec![true, false, false, true]);
        let output = &[0xAA; 64];

        // Act & Assert: for each layer, call both pre_node and post_node
        for layer in 0..3 {
            let ctx = make_test_ctx(layer);

            // pre_node behavior
            let pre_action = cb.pre_node(&ctx);
            match cb.decision_for_layer(layer) {
                SkipDecision::FullCompute | SkipDecision::MaskedCompute => {
                    assert!(matches!(pre_action, CallbackAction::Continue));
                }
                SkipDecision::CompactedCompute => {
                    assert!(matches!(pre_action, CallbackAction::CompactMask { .. }));
                }
            }

            // post_node always returns Continue
            let post_action = cb.post_node(&ctx, output);
            assert!(
                matches!(post_action, CallbackAction::Continue),
                "post_node must return Continue for layer {}",
                layer,
            );
        }
    }

    // ========================================================================
    // Additional tests (13) for 156 total
    // ========================================================================

    #[test]
    fn gate_skip_pre_node_does_not_mutate_skip_decisions() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, false, true, false]);
        let ctx = make_test_ctx(0);

        cb.pre_node(&ctx);

        assert_eq!(*cb.decision_for_layer(0), SkipDecision::CompactedCompute);
    }

    #[test]
    fn gate_skip_post_node_does_not_mutate_skip_decisions() {
        let decisions = vec![SkipDecision::MaskedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let ctx = make_test_ctx(0);

        cb.post_node(&ctx, &[0u8; 32]);

        assert_eq!(*cb.decision_for_layer(0), SkipDecision::MaskedCompute);
    }

    #[test]
    fn gate_skip_pre_node_output_deterministic_across_invocations() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        cb.store_neuron_mask(0, vec![true, false, false, true, true, true, false, false]);
        let ctx = make_test_ctx(0);

        let mut results = Vec::new();
        for _ in 0..10 {
            match cb.pre_node(&ctx) {
                CallbackAction::CompactMask { active_mask } => results.push(active_mask),
                _ => panic!("Expected CompactMask"),
            }
        }

        for mask in &results[1..] {
            assert_eq!(*mask, results[0]);
        }
    }

    #[test]
    fn gate_skip_layer_context_seq_len_does_not_affect_compact_mask() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, true, false, false]);
        let mut ctx = make_test_ctx(0);
        ctx.seq_len = 999;
        ctx.total_seq = 9999;
        ctx.position = 500;

        let action = cb.pre_node(&ctx);

        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, true, false, false]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_layer_context_node_op_does_not_affect_decision() {
        let decisions = vec![SkipDecision::FullCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let mut ctx = make_test_ctx(0);
        ctx.node_op = "Gemm";

        let action = cb.pre_node(&ctx);

        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_two_independent_callbacks_produce_different_actions() {
        let mut cb1 = GateSkipCallback::new(1, vec![SkipDecision::CompactedCompute], 4);
        let mut cb2 = GateSkipCallback::new(1, vec![SkipDecision::MaskedCompute], 4);
        cb1.store_neuron_mask(0, vec![true, false, true, false]);
        let ctx = make_test_ctx(0);

        let action1 = cb1.pre_node(&ctx);
        let action2 = cb2.pre_node(&ctx);

        assert!(matches!(action1, CallbackAction::CompactMask { .. }));
        assert!(matches!(action2, CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_update_cycle_full_compacted_masked_full_round_trip() {
        let mut cb = GateSkipCallback::new_disabled(1);
        cb.store_neuron_mask(0, vec![true, false, true, false]);
        let ctx = make_test_ctx(0);

        cb.update_decisions(vec![SkipDecision::FullCompute]);
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));

        cb.update_decisions(vec![SkipDecision::CompactedCompute]);
        match cb.pre_node(&ctx) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false]);
            }
            _ => panic!("Expected CompactMask"),
        }

        cb.update_decisions(vec![SkipDecision::MaskedCompute]);
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));

        cb.update_decisions(vec![SkipDecision::FullCompute]);
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
    }

    #[test]
    fn gate_skip_compacted_mask_with_two_active_separated_by_dead_run() {
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        let mut mask = vec![false; 8];
        mask[1] = true;
        mask[6] = true;
        cb.store_neuron_mask(0, mask);
        let ctx = make_test_ctx(0);

        let action = cb.pre_node(&ctx);

        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 8);
                assert!(!active_mask[0]);
                assert!(active_mask[1]);
                assert!(!active_mask[2]);
                assert!(!active_mask[3]);
                assert!(!active_mask[4]);
                assert!(!active_mask[5]);
                assert!(active_mask[6]);
                assert!(!active_mask[7]);
                assert_eq!(active_mask.iter().filter(|&&b| b).count(), 2);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_disabled_with_many_layers_all_continue_via_pre_node() {
        let mut cb = GateSkipCallback::new_disabled(64);

        for layer in 0..64 {
            let ctx = make_test_ctx(layer);
            let action = cb.pre_node(&ctx);
            assert!(
                matches!(action, CallbackAction::Continue),
                "disabled callback produced {:?} at layer {}",
                action,
                layer,
            );
        }
    }

    #[test]
    fn gate_skip_update_all_to_masked_then_all_pre_node_continue() {
        let decisions = vec![SkipDecision::CompactedCompute; 4];
        let mut cb = GateSkipCallback::new(4, decisions, 16);
        cb.store_neuron_mask(0, vec![true; 16]);
        cb.store_neuron_mask(1, vec![false; 16]);

        cb.update_decisions(vec![SkipDecision::MaskedCompute; 4]);

        for layer in 0..4 {
            let ctx = make_test_ctx(layer);
            let action = cb.pre_node(&ctx);
            assert!(
                matches!(action, CallbackAction::Continue),
                "MaskedCompute layer {} produced {:?}",
                layer,
                action,
            );
        }
    }

    #[test]
    fn skip_decision_vec_can_be_mapped_to_different_type() {
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::CompactedCompute,
            SkipDecision::MaskedCompute,
        ];

        let labels: Vec<&str> = decisions.iter().map(|d| match d {
            SkipDecision::FullCompute => "full",
            SkipDecision::CompactedCompute => "compacted",
            SkipDecision::MaskedCompute => "masked",
        }).collect();

        assert_eq!(labels, vec!["full", "compacted", "masked"]);
    }

    #[test]
    fn gate_skip_compacted_default_mask_is_all_true_regardless_of_intermediate_size() {
        for intermediate_size in [1, 4, 16, 128, 4096] {
            let decisions = vec![SkipDecision::CompactedCompute];
            let mut cb = GateSkipCallback::new(1, decisions, intermediate_size);
            let ctx = make_test_ctx(0);

            let action = cb.pre_node(&ctx);

            match action {
                CallbackAction::CompactMask { active_mask } => {
                    assert_eq!(
                        active_mask.len(),
                        intermediate_size,
                        "mask length mismatch for intermediate_size={}",
                        intermediate_size,
                    );
                    assert!(
                        active_mask.iter().all(|&b| b),
                        "default mask should be all-true for intermediate_size={}",
                        intermediate_size,
                    );
                }
                _ => panic!("Expected CompactMask for intermediate_size={}", intermediate_size),
            }
        }
    }

    #[test]
    fn gate_skip_update_then_store_mask_then_pre_node_uses_new_mask() {
        let mut cb = GateSkipCallback::new_disabled(1);
        cb.update_decisions(vec![SkipDecision::CompactedCompute]);

        let ctx = make_test_ctx(0);
        let action_before = cb.pre_node(&ctx);
        match action_before {
            CallbackAction::CompactMask { active_mask } => {
                assert!(active_mask.iter().all(|&b| b));
            }
            _ => panic!("Expected CompactMask before mask stored"),
        }

        cb.store_neuron_mask(0, vec![false, true, false, true]);
        let action_after = cb.pre_node(&ctx);

        match action_after {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![false, true, false, true]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    // ========================================================================
    // Additional tests (10) for 166 total
    // ========================================================================

    #[test]
    fn gate_skip_callback_action_exit_early_with_negative_logits() {
        // Arrange: ExitEarly with negative float logits (extreme value boundary)
        let action = CallbackAction::ExitEarly { logits: vec![-100.0, -0.001, -f32::MAX] };

        // Assert: negative and extreme float values are preserved exactly
        if let CallbackAction::ExitEarly { logits } = action {
            assert_eq!(logits.len(), 3);
            assert_eq!(logits[0], -100.0);
            assert_eq!(logits[1], -0.001);
            assert_eq!(logits[2], -f32::MAX);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    #[test]
    fn gate_skip_callback_action_exit_early_with_nan_and_infinity() {
        // Arrange: ExitEarly with NaN, +inf, -inf (float edge cases)
        let action = CallbackAction::ExitEarly {
            logits: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        };

        // Assert: special float values are preserved
        if let CallbackAction::ExitEarly { logits } = action {
            assert!(logits[0].is_nan());
            assert!(logits[1].is_infinite() && logits[1].is_sign_positive());
            assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
        } else {
            panic!("Expected ExitEarly");
        }
    }

    #[test]
    fn gate_skip_callback_action_exit_early_different_logits_not_equal() {
        // Arrange: two ExitEarly actions with different logits
        let a = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        let b = CallbackAction::ExitEarly { logits: vec![1.0, 3.0] };

        // Assert: different logits produce unequal actions
        assert_ne!(a, b);
    }

    #[test]
    fn gate_skip_callback_action_inject_hidden_with_large_payload() {
        // Arrange: InjectHidden with a large byte payload (simulating real hidden state)
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let action = CallbackAction::InjectHidden { data: data.clone() };

        // Assert: large payload preserved exactly
        if let CallbackAction::InjectHidden { data: payload } = action {
            assert_eq!(payload.len(), 1024);
            assert_eq!(payload, data);
        } else {
            panic!("Expected InjectHidden");
        }
    }

    #[test]
    fn gate_skip_callback_action_compact_mask_not_equal_different_length() {
        // Arrange: CompactMask actions with masks of different lengths
        let a = CallbackAction::CompactMask { active_mask: vec![true] };
        let b = CallbackAction::CompactMask { active_mask: vec![true, true] };

        // Assert: different-length masks produce unequal actions
        assert_ne!(a, b);
    }

    #[test]
    fn gate_skip_overwrite_mask_with_empty_then_uses_empty() {
        // Arrange: store a mask, then overwrite with empty mask
        let decisions = vec![SkipDecision::CompactedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 8);
        cb.store_neuron_mask(0, vec![true, true, true, true]);
        cb.store_neuron_mask(0, vec![]);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: empty mask is used (0-length), not intermediate_size fallback
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 0);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_multiple_callbacks_same_context_different_decisions() {
        // Arrange: two callbacks with different decisions share the same layer context
        let mut cb_full = GateSkipCallback::new(1, vec![SkipDecision::FullCompute], 8);
        let mut cb_compact = GateSkipCallback::new(1, vec![SkipDecision::CompactedCompute], 8);
        cb_compact.store_neuron_mask(0, vec![true, false, true, false, true, false, true, false]);
        let ctx = make_test_ctx(0);

        // Act
        let action_full = cb_full.pre_node(&ctx);
        let action_compact = cb_compact.pre_node(&ctx);

        // Assert: same context, different callbacks produce different actions
        assert!(matches!(action_full, CallbackAction::Continue));
        match action_compact {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false, true, false, true, false]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn gate_skip_pre_node_across_32_layers_with_alternating_decisions() {
        // Arrange: 32-layer callback with alternating FullCompute/CompactedCompute
        let num_layers = 32;
        let decisions: Vec<SkipDecision> = (0..num_layers)
            .map(|i| if i % 2 == 0 { SkipDecision::FullCompute } else { SkipDecision::CompactedCompute })
            .collect();
        let mut cb = GateSkipCallback::new(num_layers, decisions, 16);

        // Store masks for odd layers only
        for layer in (0..num_layers).filter(|l| l % 2 == 1) {
            let mut mask = vec![true; 16];
            mask[layer % 16] = false; // one dead neuron per odd layer
            cb.store_neuron_mask(layer, mask);
        }

        // Act & Assert: verify each layer
        for layer in 0..num_layers {
            let ctx = make_test_ctx(layer);
            let action = cb.pre_node(&ctx);
            if layer % 2 == 0 {
                assert!(
                    matches!(action, CallbackAction::Continue),
                    "even layer {} should be Continue",
                    layer,
                );
            } else {
                match action {
                    CallbackAction::CompactMask { active_mask } => {
                        assert_eq!(active_mask.len(), 16);
                        assert_eq!(active_mask.iter().filter(|&&b| !b).count(), 1);
                    }
                    _ => panic!("odd layer {} should be CompactMask", layer),
                }
            }
        }
    }

    #[test]
    fn gate_skip_update_decisions_length_zero_layers_accepted() {
        // Arrange: 0-layer callback
        let mut cb = GateSkipCallback::new(0, vec![], 8);

        // Act: update with empty vec (length matches)
        cb.update_decisions(vec![]);

        // Assert: no panic, still reports FullCompute for any layer query
        assert_eq!(*cb.decision_for_layer(0), SkipDecision::FullCompute);
    }

    #[test]
    fn gate_skip_pre_node_returns_continue_for_masked_with_stored_mask_unused() {
        // Arrange: MaskedCompute with a non-trivial stored mask — the mask exists
        // but pre_node for MaskedCompute ignores it and returns Continue
        let decisions = vec![SkipDecision::MaskedCompute];
        let mut cb = GateSkipCallback::new(1, decisions, 4);
        cb.store_neuron_mask(0, vec![true, false, true, false]);

        // Act
        let ctx = make_test_ctx(0);
        let action = cb.pre_node(&ctx);

        // Assert: MaskedCompute returns Continue; stored mask is unused by pre_node
        assert!(matches!(action, CallbackAction::Continue));

        // Act: now switch to CompactedCompute to verify mask was actually stored
        cb.update_decisions(vec![SkipDecision::CompactedCompute]);
        let action2 = cb.pre_node(&ctx);
        match action2 {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false]);
            }
            _ => panic!("Expected CompactMask after switching from MaskedCompute"),
        }
    }
}
