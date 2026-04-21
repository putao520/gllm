//! Guardrail Probe Callback — in-flight safety classification.
//!
//! SSOT: `SPEC/GUARDRAIL.md §2-§5`, priority=40 per SPEC/05-OPTIMIZATIONS.md §8.
//!
//! At the configured anchor layer's `post_node` hook, extract the last-token
//! hidden state, compute `score = sigmoid(W · h + b)`, and dispatch based on
//! `SafetyPolicy`:
//!
//! - `HaltAndVeto`: score > threshold → trigger veto flag + `ExitEarly { logits: vec![] }`
//! - `LogOnly`: log score, return `Continue`
//! - `SampleDowngrade`: write `min_temperature` to shared state, return `Continue`
//!
//! The `GuardrailSharedState` is held in an `Arc` so the callback (writer)
//! and the `GuardrailAttachment` returned to users (reader) observe the
//! same state across threads.

use std::sync::Arc;

use gllm_kernels::types::DType;
use half::{bf16, f16};

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
use crate::guardrail::{GuardProbeWeights, GuardrailSharedState, SafetyPolicy};

/// Priority of Guardrail Probe in the CallbackChain (SPEC/05-OPTIMIZATIONS.md §8).
pub const GUARDRAIL_PROBE_PRIORITY: u32 = 40;

/// Runtime GuardrailProbe callback.
///
/// Invoked by `FusedGraphExecutor::run_with_callbacks` at the anchor layer's
/// `post_node` hook. Reads the last token hidden state, runs a linear
/// classifier, and either triggers a veto (HaltAndVeto) or records telemetry.
pub struct GuardrailProbeCallback {
    /// Layer this probe fires at. Used both for `target_layers()` filtering
    /// and for the callback name.
    target_layer: usize,
    /// Linear classifier weights (`score = sigmoid(w · h + b)`).
    weights: GuardProbeWeights,
    /// Safety policy — determines callback action on score.
    policy: SafetyPolicy,
    /// Hidden size of the model (used to slice last-token row from the
    /// full `[seq_len, hidden_size]` hidden buffer).
    hidden_size: usize,
    /// Shared runtime state observed by the `GuardrailAttachment`.
    shared: Arc<GuardrailSharedState>,
    /// Human-readable probe identifier (for log messages / veto reasons).
    probe_name: String,
    /// Target layer list (single-element vec), exposed via `target_layers()`.
    layers_filter: [usize; 1],
}

impl GuardrailProbeCallback {
    pub fn new(
        target_layer: usize,
        weights: GuardProbeWeights,
        policy: SafetyPolicy,
        hidden_size: usize,
        shared: Arc<GuardrailSharedState>,
        probe_name: String,
    ) -> Self {
        Self {
            target_layer,
            weights,
            policy,
            hidden_size,
            shared,
            probe_name,
            layers_filter: [target_layer],
        }
    }

    /// Shared state (test hook).
    #[cfg(test)]
    pub fn shared(&self) -> &Arc<GuardrailSharedState> {
        &self.shared
    }
}

impl LayerCallback for GuardrailProbeCallback {
    fn post_node(&mut self, ctx: &LayerContext, output: &[u8]) -> CallbackAction {
        // layer filter — only fire at the anchor layer.
        if ctx.layer_idx != self.target_layer {
            return CallbackAction::Continue;
        }
        // Respect already-vetoed state (avoid double firing if callback runs
        // multiple nodes within the same layer).
        if self
            .shared
            .vetoed
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return CallbackAction::Continue;
        }

        let dtype = ctx.model_config.geometry.dtype;
        let last_hidden = match extract_last_token_hidden(output, self.hidden_size, dtype) {
            Ok(v) => v,
            Err(_) => return CallbackAction::Continue,
        };

        let score = self.weights.score(&last_hidden);
        self.shared.record_score(score);

        match self.policy {
            SafetyPolicy::HaltAndVeto { threshold } => {
                if score > threshold {
                    let reason = format!(
                        "Guardrail '{}' vetoed: score {:.4} > threshold {:.4}",
                        self.probe_name, score, threshold
                    );
                    log::warn!("{reason}");
                    self.shared.trigger_veto(reason);
                    // SPEC/GUARDRAIL.md §5.1 step 2-3: emit `ExitEarly` with
                    // empty logits; `FusedGraphExecutor` returns empty outputs;
                    // backend surfaces this as an empty hidden vector; Client
                    // reads `GuardrailAttachment::is_vetoed()`.
                    CallbackAction::ExitEarly { logits: Vec::new() }
                } else {
                    CallbackAction::Continue
                }
            }
            SafetyPolicy::LogOnly => {
                log::debug!(
                    "Guardrail '{}' score={:.4} (log-only)",
                    self.probe_name,
                    score
                );
                CallbackAction::Continue
            }
            SafetyPolicy::SampleDowngrade {
                min_temperature,
            } => {
                // Record min_temperature unconditionally — upper sampler may
                // weight it against other signals. Logging helps audit.
                self.shared.record_downgrade(min_temperature);
                log::debug!(
                    "Guardrail '{}' score={:.4} → sample downgrade min_temperature={:.3}",
                    self.probe_name,
                    score,
                    min_temperature
                );
                CallbackAction::Continue
            }
        }
    }

    fn priority(&self) -> u32 {
        GUARDRAIL_PROBE_PRIORITY
    }

    fn target_layers(&self) -> Option<&[usize]> {
        Some(&self.layers_filter)
    }

    fn name(&self) -> &str {
        "GuardrailProbe"
    }
}

// ============================================================================
// dtype-aware hidden state decoding (mirrors semantic_gatekeeper/callback.rs)
// ============================================================================

#[derive(Debug)]
enum ExtractError {
    Truncated,
    Overflow,
    UnsupportedDtype,
}

fn extract_last_token_hidden(
    hidden_bytes: &[u8],
    hidden_size: usize,
    dtype: DType,
) -> Result<Vec<f32>, ExtractError> {
    let elem_bytes = dtype.size_bytes();
    let row_bytes = hidden_size
        .checked_mul(elem_bytes)
        .ok_or(ExtractError::Overflow)?;
    if row_bytes == 0 || hidden_bytes.len() < row_bytes {
        return Err(ExtractError::Truncated);
    }
    let last_start = hidden_bytes.len() - row_bytes;
    decode_row(&hidden_bytes[last_start..], hidden_size, dtype)
}

fn decode_row(row: &[u8], hidden_size: usize, dtype: DType) -> Result<Vec<f32>, ExtractError> {
    let elem_bytes = dtype.size_bytes();
    if row.len() != hidden_size * elem_bytes {
        return Err(ExtractError::Truncated);
    }
    let mut out = Vec::with_capacity(hidden_size);
    match dtype {
        DType::F32 => {
            for i in 0..hidden_size {
                let off = i * 4;
                out.push(f32::from_le_bytes([
                    row[off],
                    row[off + 1],
                    row[off + 2],
                    row[off + 3],
                ]));
            }
        }
        DType::F16 => {
            for i in 0..hidden_size {
                let off = i * 2;
                out.push(f16::from_le_bytes([row[off], row[off + 1]]).to_f32());
            }
        }
        DType::BF16 => {
            for i in 0..hidden_size {
                let off = i * 2;
                out.push(bf16::from_le_bytes([row[off], row[off + 1]]).to_f32());
            }
        }
        _ => return Err(ExtractError::UnsupportedDtype),
    }
    Ok(out)
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guardrail::GuardProbeWeights;

    fn make_shared() -> Arc<GuardrailSharedState> {
        Arc::new(GuardrailSharedState::new())
    }

    #[test]
    fn callback_metadata_priority_and_name() {
        let shared = make_shared();
        let cb = GuardrailProbeCallback::new(
            3,
            GuardProbeWeights { weight: vec![0.0; 4], bias: 0.0 },
            SafetyPolicy::LogOnly,
            4,
            shared,
            "test".to_string(),
        );
        assert_eq!(cb.priority(), GUARDRAIL_PROBE_PRIORITY);
        assert_eq!(cb.name(), "GuardrailProbe");
        assert_eq!(cb.target_layers(), Some(&[3usize][..]));
    }

    #[test]
    fn extract_last_token_f32_returns_last_row() {
        // seq_len=3, hidden_size=4
        let hidden: Vec<u8> = [
            1.0f32, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0,    // row 1
            9.0, 10.0, 11.0, 12.0, // row 2 (last)
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
        let last = extract_last_token_hidden(&hidden, 4, DType::F32).unwrap();
        assert_eq!(last, vec![9.0, 10.0, 11.0, 12.0]);
    }
}
