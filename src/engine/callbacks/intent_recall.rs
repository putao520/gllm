//! Intent Recall Callback (SPEC §16.3)
//!
//! Integrates intent extraction into the graph node loop.
//! At the target semantic layer, extracts the hidden state as an intent embedding
//! and stores it for downstream use (e.g., routing, classification).

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
use crate::intent::IntentConfig;
use crate::knowledge::LayerTarget;

/// Intent recall callback — extracts intent embedding at a target layer.
///
/// Per SPEC §16.3: "Intent Recall" — at a configured layer, the hidden state
/// is captured as a semantic intent embedding. This embedding can be used for
/// request routing, intent classification, or as a retrieval key.
pub struct IntentRecallCallback {
    /// Intent extraction configuration
    config: IntentConfig,
    /// Target layer index (resolved from LayerTarget)
    target_layer_idx: usize,
    /// Stored target layers slice for target_layers() return
    target_layer_vec: Vec<usize>,
    /// Last extracted embedding (stored for retrieval after the forward pass)
    last_embedding: Vec<f32>,
    /// The request ID for which the embedding was extracted
    last_request_id: u64,
}

impl IntentRecallCallback {
    /// Create a new intent recall callback.
    ///
    /// `config` — intent extraction configuration with target layer
    /// `total_layers` — total number of transformer layers
    pub fn new(config: IntentConfig, total_layers: usize) -> Self {
        let target_layer_idx = config.target.to_physical_layer(total_layers);
        let target_layer_vec = vec![target_layer_idx];
        Self {
            config,
            target_layer_idx,
            target_layer_vec,
            last_embedding: Vec::new(),
            last_request_id: 0,
        }
    }

    /// Get the last extracted intent embedding.
    pub fn last_embedding(&self) -> &[f32] {
        &self.last_embedding
    }

    /// Get the request ID for which the embedding was extracted.
    pub fn last_request_id(&self) -> u64 {
        self.last_request_id
    }

    /// Take the extracted embedding, clearing internal storage.
    pub fn take_embedding(&mut self) -> Vec<f32> {
        self.last_embedding.drain(..).collect()
    }

    /// Convert byte slice to f32 slice (little-endian).
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl LayerCallback for IntentRecallCallback {
    fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
        CallbackAction::Continue
    }

    fn post_node(&mut self, ctx: &LayerContext, output: &[u8]) -> CallbackAction {
        if ctx.layer_idx != self.target_layer_idx {
            return CallbackAction::Continue;
        }

        // Extract hidden state as intent embedding
        self.last_embedding = Self::bytes_to_f32(output);
        self.last_request_id = ctx.request_id;

        log::trace!(
            "intent_recall: extracted embedding at layer {} ({} dims, request_id={})",
            ctx.layer_idx,
            self.last_embedding.len(),
            ctx.request_id,
        );

        CallbackAction::Continue
    }

    fn priority(&self) -> u32 {
        30
    }

    fn target_layers(&self) -> Option<&[usize]> {
        Some(&self.target_layer_vec)
    }

    fn name(&self) -> &str {
        "intent_recall"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_recall_callback_creation() {
        let config = IntentConfig::new(LayerTarget::MidSemantic);
        let cb = IntentRecallCallback::new(config, 32);
        assert_eq!(cb.priority(), 30);
        assert_eq!(cb.name(), "intent_recall");
        assert_eq!(cb.target_layer_idx, 16); // MidSemantic at 32 layers
        assert!(cb.last_embedding().is_empty());
    }

    #[test]
    fn test_intent_recall_target_layers() {
        let config = IntentConfig::new(LayerTarget::DeepLogic);
        let cb = IntentRecallCallback::new(config, 12);
        assert_eq!(cb.target_layers(), Some(&[10usize][..])); // 0.875 * 12 = 10.5 → 10
    }

    #[test]
    fn test_take_embedding() {
        let config = IntentConfig::default();
        let mut cb = IntentRecallCallback::new(config, 12);
        cb.last_embedding = vec![1.0, 2.0, 3.0];
        let taken = cb.take_embedding();
        assert_eq!(taken, vec![1.0, 2.0, 3.0]);
        assert!(cb.last_embedding().is_empty());
    }
}
