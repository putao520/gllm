//! RAG Inject Callback (SPEC §16.1)
//!
//! Integrates `LateFusionRag` into the graph node loop.
//! At the configured fusion layer, retrieves relevant documents and
//! fuses them into the hidden state via residual connection injection.

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
use crate::rag::LateFusionRag;

/// RAG inject callback — retrieves and fuses external knowledge at a target layer.
///
/// Per SPEC §16.1: "Late-Fusion RAG" — retrieval results are injected into
/// the residual stream at a specific layer, providing context without
/// modifying the model weights.
pub struct RagInjectCallback {
    /// The RAG system for retrieval and fusion
    rag: LateFusionRag,
    /// Cached injection data (computed once from retrieve + fuse)
    cached_injection: Option<Vec<u8>>,
    /// Stored target layers slice for target_layers() return
    target_layer_vec: Vec<usize>,
}

impl RagInjectCallback {
    /// Create a new RAG inject callback.
    ///
    /// `rag` — pre-configured LateFusionRag with retrieval_db and fusion params
    pub fn new(rag: LateFusionRag) -> Self {
        let target_layer_vec = vec![rag.fusion_layer];
        Self {
            rag,
            cached_injection: None,
            target_layer_vec,
        }
    }

    /// Get a reference to the underlying RAG system.
    pub fn rag(&self) -> &LateFusionRag {
        &self.rag
    }

    /// Convert f32 slice to bytes (little-endian).
    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Convert bytes to f32 slice (little-endian).
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl LayerCallback for RagInjectCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        if ctx.layer_idx != self.rag.fusion_layer {
            return CallbackAction::Continue;
        }

        if self.rag.retrieval_db.is_empty() {
            return CallbackAction::Continue;
        }

        // Extract hidden state as f32, retrieve, and fuse
        let mut hidden = Self::bytes_to_f32(ctx.hidden_state);
        self.rag.fuse_at_residual(&mut hidden, ctx.layer_idx);

        log::trace!(
            "rag_inject: fusing at layer {} ({} docs, weight={:.3})",
            ctx.layer_idx,
            self.rag.retrieval_db.len(),
            self.rag.fusion_weight,
        );

        // Return modified hidden state as InjectHidden
        let injected = Self::f32_to_bytes(&hidden);
        self.cached_injection = Some(injected.clone());
        CallbackAction::InjectHidden { data: injected }
    }

    fn post_node(&mut self, _ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
        CallbackAction::Continue
    }

    fn priority(&self) -> u32 {
        80
    }

    fn target_layers(&self) -> Option<&[usize]> {
        Some(&self.target_layer_vec)
    }

    fn name(&self) -> &str {
        "rag_inject"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_inject_callback_creation() {
        let rag = LateFusionRag::new(4);
        let cb = RagInjectCallback::new(rag);
        assert_eq!(cb.priority(), 80);
        assert_eq!(cb.name(), "rag_inject");
        assert_eq!(cb.rag().fusion_layer, 4);
    }

    #[test]
    fn test_rag_inject_target_layers() {
        let rag = LateFusionRag::new(7);
        let cb = RagInjectCallback::new(rag);
        assert_eq!(cb.target_layers(), Some(&[7usize][..]));
    }

    #[test]
    fn test_f32_bytes_roundtrip() {
        let original = vec![1.0f32, -2.5, 0.0, 3.14];
        let bytes = RagInjectCallback::f32_to_bytes(&original);
        let restored = RagInjectCallback::bytes_to_f32(&bytes);
        assert_eq!(original.len(), restored.len());
        for (a, b) in original.iter().zip(&restored) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
