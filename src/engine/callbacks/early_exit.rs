//! Early Exit Callback (SPEC §16.2)
//!
//! Integrates `EarlyExitController` into the graph node loop.
//! At designated exit points (golden ratio layers), evaluates cosine similarity
//! and energy delta of residual connections to determine if early termination
//! is safe. When confidence exceeds threshold, returns `ExitEarly`.

use crate::early_exit::{EarlyExitConfig, EarlyExitController, EarlyExitDecision};
use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};

/// Early exit callback — monitors residual convergence at golden-ratio exit points.
///
/// Per SPEC §16.2: "任意层数据召回与高维截断"
/// When the residual between consecutive layers converges (high cosine similarity
/// + low energy delta), remaining layers contribute negligibly and can be skipped.
pub struct EarlyExitCallback {
    controller: EarlyExitController,
    /// Cached last-layer hidden state for cosine similarity computation
    prev_hidden: Vec<f32>,
}

impl EarlyExitCallback {
    /// Create a new early exit callback.
    ///
    /// `config` — early exit thresholds and enabling flag
    /// `total_layers` — total number of transformer layers in the model
    pub fn new(config: EarlyExitConfig, total_layers: usize) -> Self {
        let controller = EarlyExitController::new(config, total_layers);
        Self {
            controller,
            prev_hidden: Vec::new(),
        }
    }

    /// Get a reference to the underlying controller.
    pub fn controller(&self) -> &EarlyExitController {
        &self.controller
    }

    /// Compute cosine similarity between two f32 slices.
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 0.0;
        }
        let dot: f32 = a[..len].iter().zip(&b[..len]).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Compute energy ratio ‖b‖ / ‖a‖ (delta_rho).
    fn energy_ratio(a: &[f32], b: &[f32]) -> f32 {
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 {
            1.0
        } else {
            norm_b / norm_a
        }
    }

    /// Convert byte slice to f32 slice (little-endian).
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl LayerCallback for EarlyExitCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        // Store current hidden state for post_node comparison
        self.prev_hidden = Self::bytes_to_f32(ctx.hidden_state);
        CallbackAction::Continue
    }

    fn post_node(&mut self, ctx: &LayerContext, output: &[u8]) -> CallbackAction {
        let current = Self::bytes_to_f32(output);

        let cosine_sim = Self::cosine_sim(&self.prev_hidden, &current);
        let delta_rho = Self::energy_ratio(&self.prev_hidden, &current);

        match self.controller.check_layer(ctx.layer_idx, cosine_sim, delta_rho) {
            EarlyExitDecision::Exit { confidence, .. } => {
                log::debug!(
                    "early_exit: layer={} confidence={:.4} cos={:.4} delta_rho={:.4}",
                    ctx.layer_idx, confidence, cosine_sim, delta_rho,
                );
                // Return ExitEarly with empty logits — caller should project
                // current hidden_state through lm_head to get actual logits
                CallbackAction::ExitEarly { logits: Vec::new() }
            }
            _ => CallbackAction::Continue,
        }
    }

    fn priority(&self) -> u32 {
        50
    }

    fn target_layers(&self) -> Option<&[usize]> {
        // Only trigger at exit points — but exit points are dynamic,
        // so we check inside post_node() and return None (all layers) here.
        None
    }

    fn name(&self) -> &str {
        "early_exit"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::layer_callback::CallbackChain;

    #[test]
    fn test_early_exit_callback_disabled() {
        let config = EarlyExitConfig::default(); // enabled = false
        let cb = EarlyExitCallback::new(config, 32);
        assert_eq!(cb.controller().config().enabled, false);
        assert_eq!(cb.priority(), 50);
        assert_eq!(cb.name(), "early_exit");
    }

    #[test]
    fn test_cosine_sim_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = EarlyExitCallback::cosine_sim(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_energy_ratio() {
        let a = vec![3.0, 4.0]; // norm = 5
        let b = vec![6.0, 8.0]; // norm = 10
        let ratio = EarlyExitCallback::energy_ratio(&a, &b);
        assert!((ratio - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_bytes_to_f32() {
        let bytes: Vec<u8> = 1.0f32.to_le_bytes().iter()
            .chain(2.0f32.to_le_bytes().iter())
            .copied()
            .collect();
        let result = EarlyExitCallback::bytes_to_f32(&bytes);
        assert_eq!(result, vec![1.0, 2.0]);
    }
}
