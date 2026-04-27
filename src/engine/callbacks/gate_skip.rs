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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipDecision {
    /// Normal full computation — all neurons active.
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

impl Default for SkipDecision {
    fn default() -> Self {
        Self::FullCompute
    }
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
        use crate::graph::types::{FusedOp, AtomicOp};
        static TEST_OP: LazyLock<FusedOp> = LazyLock::new(|| FusedOp::Atomic(AtomicOp { op_type: "test".to_string() }));
        // SAFETY: model_config is never dereferenced by GateSkipCallback.pre_node().
        // It only reads ctx.layer_idx. This is test-only code.
        static FAKE_CONFIG: LazyLock<crate::engine::executor::GeneratorForwardConfig> = LazyLock::new(|| {
            use std::sync::Arc;
            let geom = Arc::new(crate::model_config::ModelGeometry {
                hidden_size: 256, num_layers: 1, vocab_size: 1000, intermediate_size: 512,
                num_heads: 4, num_kv_heads: 4, head_dim: 64, max_seq_len: 128,
                rope_theta: 10000.0, rope_scale: 1.0, rope_interleaved: false,
                dtype: gllm_kernels::types::DType::F32, norm_eps: 1e-5,
                num_experts: 0, moe_top_k: 0, expert_intermediate_size: 0,
                global_rope_theta: 0.0, rope_partial_ratio: 1.0, attention_pattern: vec![],
                sliding_window: 0, num_kv_shared_layers: 0, global_head_dim: 0, hidden_size_per_layer_input: 0,
                position_offset: None,
                rope_scaling: None,
                final_logit_softcapping: None,
                hidden_act: None,
            });
            crate::engine::executor::GeneratorForwardConfig {
                geometry: geom,
                position_encoding: crate::engine::executor::PositionEncoding::Rope,
                arch_family: crate::manifest::ArchFamily::Decoder,
                rope: crate::engine::executor::RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: true },
                rerank_yes_token_id: None, rerank_no_token_id: None,
                moe_config: None,
                paged_kv: crate::engine::executor::PagedKvConfig { page_table: None, page_size: 16 },
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
                graph_executor_ptr: std::ptr::null_mut(),
                callback_chain_ptr: std::ptr::null_mut(),
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
}
