//! Knowledge Inject Callback (SPEC §8.1)
//!
//! Integrates knowledge injection into the graph node loop.
//! At the target layer, injects pre-materialized knowledge vectors
//! (Late Fusion, LoRA, or Frozen KV) into the hidden state.

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
use crate::knowledge::{InjectionKind, LayerTarget, MaterializedPayload};

/// Knowledge inject callback — injects materialized knowledge at a target layer.
///
/// Per SPEC §8.1: supports three injection types:
/// - `FrozenKvChunk`: zero-copy page table insertion
/// - `LateFusionVector`: feature vector injection via residual
/// - `DynamicLoRA`: LoRA weight patching
pub struct KnowledgeInjectCallback {
    /// Target layer index (resolved from LayerTarget)
    target_layer_idx: usize,
    /// Pre-materialized payload to inject
    payload: MaterializedPayload,
    /// Whether injection has been applied (one-shot)
    injected: bool,
    /// Stored target layers slice for target_layers() return
    target_layer_vec: Vec<usize>,
}

impl KnowledgeInjectCallback {
    /// Create a new knowledge inject callback.
    ///
    /// `payload` — pre-materialized knowledge payload
    /// `total_layers` — total number of transformer layers
    pub fn new(payload: MaterializedPayload, total_layers: usize) -> Self {
        // Default to MidSemantic for injection target
        let target = LayerTarget::MidSemantic;
        let target_layer_idx = target.to_physical_layer(total_layers);
        let target_layer_vec = vec![target_layer_idx];
        Self {
            target_layer_idx,
            target_layer_vec,
            payload,
            injected: false,
        }
    }

    /// Create with explicit target layer.
    pub fn with_target(payload: MaterializedPayload, target: LayerTarget, total_layers: usize) -> Self {
        let target_layer_idx = target.to_physical_layer(total_layers);
        let target_layer_vec = vec![target_layer_idx];
        Self {
            target_layer_idx,
            target_layer_vec,
            payload,
            injected: false,
        }
    }

    /// Get the injection kind.
    pub fn injection_kind(&self) -> InjectionKind {
        self.payload.kind
    }

    /// Whether injection has been applied.
    pub fn is_injected(&self) -> bool {
        self.injected
    }
}

impl LayerCallback for KnowledgeInjectCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        if self.injected || ctx.layer_idx != self.target_layer_idx {
            return CallbackAction::Continue;
        }

        match self.payload.kind {
            InjectionKind::LateFusionVector => {
                if self.payload.data.is_empty() {
                    log::warn!("knowledge_inject: LateFusionVector payload is empty, skipping");
                    self.injected = true;
                    return CallbackAction::Continue;
                }
                log::trace!(
                    "knowledge_inject: injecting LateFusionVector at layer {} ({} bytes)",
                    ctx.layer_idx,
                    self.payload.data.len(),
                );
                self.injected = true;
                CallbackAction::InjectHidden { data: self.payload.data.clone() }
            }
            InjectionKind::FrozenKvChunk => {
                // Frozen KV injection operates on the KV cache page table,
                // not on the hidden state. The callback just signals that
                // injection should happen — the executor handles page table ops.
                log::trace!(
                    "knowledge_inject: FrozenKvChunk at layer {} ({} bytes, {} pages)",
                    ctx.layer_idx,
                    self.payload.data.len(),
                    self.payload.shape.len(),
                );
                self.injected = true;
                // No hidden state modification — KV injection is transparent
                CallbackAction::Continue
            }
            InjectionKind::DynamicLoRA => {
                // LoRA injection patches model weights, not hidden state.
                // The callback signals the layer, the executor handles weight swap.
                log::trace!(
                    "knowledge_inject: DynamicLoRA at layer {}",
                    ctx.layer_idx,
                );
                self.injected = true;
                CallbackAction::Continue
            }
        }
    }

    fn post_node(&mut self, _ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
        CallbackAction::Continue
    }

    fn priority(&self) -> u32 {
        90
    }

    fn target_layers(&self) -> Option<&[usize]> {
        Some(&self.target_layer_vec)
    }

    fn name(&self) -> &str {
        "knowledge_inject"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_payload(kind: InjectionKind, data: Vec<u8>) -> MaterializedPayload {
        MaterializedPayload {
            kind,
            data,
            shape: vec![256],
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_knowledge_inject_late_fusion() {
        let payload = make_payload(InjectionKind::LateFusionVector, vec![42u8; 1024]);
        let cb = KnowledgeInjectCallback::new(payload, 32);
        assert_eq!(cb.priority(), 90);
        assert_eq!(cb.name(), "knowledge_inject");
        assert_eq!(cb.injection_kind(), InjectionKind::LateFusionVector);
        assert!(!cb.is_injected());
    }

    #[test]
    fn test_knowledge_inject_with_target() {
        let payload = make_payload(InjectionKind::DynamicLoRA, vec![]);
        let cb = KnowledgeInjectCallback::with_target(
            payload,
            LayerTarget::ShallowSyntax,
            12,
        );
        assert_eq!(cb.injection_kind(), InjectionKind::DynamicLoRA);
        // ShallowSyntax at 12 layers = 0.125 * 12 = 1.5 → 1
        assert_eq!(cb.target_layer_idx, 1);
    }

    #[test]
    fn test_knowledge_inject_frozen_kv() {
        let payload = make_payload(InjectionKind::FrozenKvChunk, vec![0u8; 8192]);
        let cb = KnowledgeInjectCallback::new(payload, 32);
        assert_eq!(cb.injection_kind(), InjectionKind::FrozenKvChunk);
    }
}
