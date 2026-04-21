//! Mid-Layer Encode Callback — captures hidden state at a target layer and
//! terminates the forward pass early.
//!
//! SSOT: `SPEC/HEAD-ROUTING.md §5 Mid-layer Encode 协议`,
//! `SPEC/INTENT.md §3`.
//!
//! Registered by `Client::encode_to_layer` / `Client::encode_intent` onto a
//! fresh `CallbackChain` passed to `FusedGraphExecutor::run_with_callbacks`.
//! On `post_node` of the first node whose derived `layer_idx` equals the
//! target, the callback returns `CallbackAction::ExitEarly { logits: <hidden bytes as f32> }`,
//! causing the executor to short-circuit and return the hidden state via the
//! `"logits"` output key. The caller then reshapes `[seq_len, hidden_size]`
//! and applies the requested `PoolMode`.

use gllm_kernels::types::DType;
use half::{bf16, f16};

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};

/// Priority for mid-layer encode (higher than Early Exit=50, lower than SG=90).
pub const MID_LAYER_ENCODE_PRIORITY: u32 = 55;

/// Mid-layer encode callback.
pub struct MidLayerEncodeCallback {
    target_layer: usize,
    layers_filter: [usize; 1],
}

impl MidLayerEncodeCallback {
    pub fn new(target_layer: usize) -> Self {
        Self {
            target_layer,
            layers_filter: [target_layer],
        }
    }
}

impl LayerCallback for MidLayerEncodeCallback {
    fn post_node(&mut self, ctx: &LayerContext, output: &[u8]) -> CallbackAction {
        if ctx.layer_idx != self.target_layer {
            return CallbackAction::Continue;
        }
        // Decode full `[seq_len, hidden_size]` hidden state into f32 and ship
        // via ExitEarly. The executor packages `logits` bytes back into the
        // returned HashMap under the `"logits"` key.
        let dtype = ctx.model_config.geometry.dtype;
        let hidden = match decode_hidden(output, dtype) {
            Some(v) => v,
            None => return CallbackAction::Continue,
        };
        CallbackAction::ExitEarly { logits: hidden }
    }

    fn priority(&self) -> u32 {
        MID_LAYER_ENCODE_PRIORITY
    }

    fn target_layers(&self) -> Option<&[usize]> {
        Some(&self.layers_filter)
    }

    fn name(&self) -> &str {
        "MidLayerEncode"
    }
}

/// Decode a dtype-aware hidden buffer into flat f32. Returns `None` when
/// dtype is unsupported for mid-layer encoding (e.g. integer tensors).
fn decode_hidden(bytes: &[u8], dtype: DType) -> Option<Vec<f32>> {
    let elem_bytes = dtype.size_bytes();
    if elem_bytes == 0 || bytes.len() % elem_bytes != 0 {
        return None;
    }
    let n = bytes.len() / elem_bytes;
    let mut out = Vec::with_capacity(n);
    match dtype {
        DType::F32 => {
            for i in 0..n {
                let off = i * 4;
                out.push(f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]));
            }
        }
        DType::F16 => {
            for i in 0..n {
                let off = i * 2;
                out.push(f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32());
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let off = i * 2;
                out.push(bf16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32());
            }
        }
        _ => return None,
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_metadata() {
        let cb = MidLayerEncodeCallback::new(7);
        assert_eq!(cb.priority(), MID_LAYER_ENCODE_PRIORITY);
        assert_eq!(cb.name(), "MidLayerEncode");
        assert_eq!(cb.target_layers(), Some(&[7usize][..]));
    }

    #[test]
    fn decode_f32_roundtrip() {
        let src: Vec<u8> = [1.0f32, -2.5, 3.14]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = decode_hidden(&src, DType::F32).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] + 2.5).abs() < 1e-6);
        assert!((out[2] - 3.14).abs() < 1e-5);
    }

    #[test]
    fn decode_bad_length_returns_none() {
        let bytes = vec![0u8; 3]; // not divisible by 4
        assert!(decode_hidden(&bytes, DType::F32).is_none());
    }
}
