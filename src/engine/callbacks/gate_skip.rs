//! Gate Skip Callback (SPEC §13.1)
//!
//! Applies pre-computed gate skip decisions to FFN nodes during graph execution.
//! When dead neuron density exceeds threshold, FFN layers can be safely
//! skipped without quality degradation.

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};

/// Per-layer skip decision for gate skip optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipDecision {
    /// Normal full computation.
    FullCompute,
    /// Skip the FFN entirely.
    Skip,
    /// Compute with reduced precision (partial mask).
    MaskedCompute,
}

impl Default for SkipDecision {
    fn default() -> Self {
        Self::FullCompute
    }
}

/// Gate skip callback — skips FFN computation when dead neuron density is high.
///
/// Per SPEC §13.1: reads skip decisions (typically computed by the EpilogueSubsystem
/// from the previous step's telemetry) and applies them to the current forward pass.
/// FFN nodes at layers with high dead density (>50%) are skipped entirely.
pub struct GateSkipCallback {
    /// Pre-computed skip decisions per layer (populated from epilogue summary)
    skip_decisions: Vec<SkipDecision>,
    /// Number of transformer layers
    num_layers: usize,
}

impl GateSkipCallback {
    /// Create a new gate skip callback with pre-computed decisions.
    ///
    /// `num_layers` — total number of transformer layers
    /// `decisions` — per-layer skip decisions (length must equal num_layers)
    pub fn new(num_layers: usize, decisions: Vec<SkipDecision>) -> Self {
        assert_eq!(decisions.len(), num_layers, "decisions length must match num_layers");
        Self {
            skip_decisions: decisions,
            num_layers,
        }
    }

    /// Create a disabled callback (all FullCompute).
    pub fn new_disabled(num_layers: usize) -> Self {
        Self {
            skip_decisions: vec![SkipDecision::FullCompute; num_layers],
            num_layers,
        }
    }

    /// Update skip decisions for a new batch (typically from epilogue summary).
    pub fn update_decisions(&mut self, decisions: Vec<SkipDecision>) {
        assert_eq!(decisions.len(), self.num_layers);
        self.skip_decisions = decisions;
    }

    /// Get the decision for a specific layer.
    pub fn decision_for_layer(&self, layer: usize) -> SkipDecision {
        self.skip_decisions.get(layer).copied().unwrap_or(SkipDecision::FullCompute)
    }
}

impl LayerCallback for GateSkipCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        let decision = self.decision_for_layer(ctx.layer_idx);
        match decision {
            SkipDecision::Skip => {
                log::trace!("gate_skip: skipping FFN at layer {}", ctx.layer_idx);
                CallbackAction::SkipThisNode
            }
            SkipDecision::MaskedCompute => {
                log::trace!("gate_skip: masked compute at layer {}", ctx.layer_idx);
                // MaskedCompute still executes but with reduced work — not a skip
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
            assert_eq!(cb.decision_for_layer(i), SkipDecision::FullCompute);
        }
    }

    #[test]
    fn test_gate_skip_with_skip_decisions() {
        let decisions = vec![
            SkipDecision::FullCompute,
            SkipDecision::Skip,
            SkipDecision::FullCompute,
            SkipDecision::Skip,
        ];
        let cb = GateSkipCallback::new(4, decisions);
        assert_eq!(cb.decision_for_layer(0), SkipDecision::FullCompute);
        assert_eq!(cb.decision_for_layer(1), SkipDecision::Skip);
        assert_eq!(cb.decision_for_layer(3), SkipDecision::Skip);
    }

    #[test]
    fn test_update_decisions() {
        let mut cb = GateSkipCallback::new_disabled(2);
        cb.update_decisions(vec![SkipDecision::Skip, SkipDecision::FullCompute]);
        assert_eq!(cb.decision_for_layer(0), SkipDecision::Skip);
        assert_eq!(cb.decision_for_layer(1), SkipDecision::FullCompute);
    }
}
