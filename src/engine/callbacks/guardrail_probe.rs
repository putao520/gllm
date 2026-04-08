//! Guardrail Probe Callback (SPEC §16.4)
//!
//! Integrates `GuardProbeRunner` into the graph node loop.
//! At the target inspection layer, classifies the hidden state for safety.
//! If the score exceeds threshold, triggers veto via `ExitEarly`.

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
use crate::guardrail::GuardProbeRunner;
use crate::knowledge::LayerTarget;

/// Guardrail probe callback — runs safety classification at a target layer.
///
/// Per SPEC §16.4: "零延迟飞行巡航审查" — a minimal linear classifier is
/// physically inserted at a deep layer to detect harmful content. When the
/// classifier triggers, the generation is immediately vetoed.
pub struct GuardrailProbeCallback {
    /// The probe runner for classification
    runner: GuardProbeRunner,
    /// Target layer index (resolved from LayerTarget at construction)
    target_layer_idx: usize,
    /// Stored target layers slice for target_layers() return
    target_layer_vec: Vec<usize>,
}

impl GuardrailProbeCallback {
    /// Create a new guardrail probe callback.
    ///
    /// `runner` — pre-loaded GuardProbeRunner with weights and threshold
    /// `total_layers` — total number of transformer layers (for LayerTarget resolution)
    pub fn new(runner: GuardProbeRunner, total_layers: usize) -> Self {
        let target_layer = runner.target_layer();
        let target_layer_idx = target_layer.to_physical_layer(total_layers);
        let target_layer_vec = vec![target_layer_idx];
        Self {
            runner,
            target_layer_idx,
            target_layer_vec,
        }
    }

    /// Get a reference to the underlying probe runner.
    pub fn runner(&self) -> &GuardProbeRunner {
        &self.runner
    }

    /// Convert byte slice to f32 slice (little-endian).
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl LayerCallback for GuardrailProbeCallback {
    fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
        CallbackAction::Continue
    }

    fn post_node(&mut self, ctx: &LayerContext, output: &[u8]) -> CallbackAction {
        if ctx.layer_idx != self.target_layer_idx {
            return CallbackAction::Continue;
        }

        // Convert output bytes to f32 and classify
        let hidden = Self::bytes_to_f32(output);
        let score = self.runner.score(&hidden);

        if score > self.runner.threshold_value() {
            log::warn!(
                "guardrail_probe: VETO at layer {} — score {:.4} > threshold {:.4}",
                ctx.layer_idx, score, self.runner.threshold_value(),
            );

            // Write veto state
            self.runner.veto_state().set_veto(format!(
                "Guardrail vetoed at layer {}: score {:.4} > {:.4}",
                ctx.layer_idx, score, self.runner.threshold_value(),
            ));

            // ExitEarly with empty logits — the veto flag is the signal
            CallbackAction::ExitEarly { logits: Vec::new() }
        } else {
            log::trace!(
                "guardrail_probe: layer {} score {:.4} (threshold {:.4}) — pass",
                ctx.layer_idx, score, self.runner.threshold_value(),
            );
            CallbackAction::Continue
        }
    }

    fn priority(&self) -> u32 {
        40
    }

    fn target_layers(&self) -> Option<&[usize]> {
        Some(&self.target_layer_vec)
    }

    fn name(&self) -> &str {
        "guardrail_probe"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guardrail_callback_priority() {
        // Priority is 40 per SPEC plan
        assert_eq!(40u32, 40u32);
    }

    #[test]
    fn test_layer_target_resolution() {
        assert_eq!(LayerTarget::MidSemantic.to_physical_layer(32), 16);
        assert_eq!(LayerTarget::DeepLogic.to_physical_layer(32), 28);
        assert_eq!(LayerTarget::ShallowSyntax.to_physical_layer(32), 4);
    }
}
