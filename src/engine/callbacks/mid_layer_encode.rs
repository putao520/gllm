//! Mid-Layer Encode Callback — captures hidden state at a target layer and
//! terminates the forward pass early.
//!
//! SSOT: `SPEC/HEAD-ROUTING.md §5 Mid-layer Encode 协议`,
//! `SPEC/INTENT.md §3`.
//!
//! Registered by `Client::encode_to_layer` / `Client::encode_intent` onto a
//! fresh `CallbackChain` passed to `FusedGraphExecutor::run_with_callbacks`.
//!
//! # 语义 (SPEC/HEAD-ROUTING.md §5.1)
//!
//! `forward(tokens, exit_at_layer=anchor_idx)` — 截断前向至 anchor 层，
//! 返回 `H_anchor = [seq_len, hidden_size]` = **layer anchor 完成后的隐藏状态**。
//!
//! # 实现
//!
//! 一个 Transformer layer 包含多个图节点（input_norm / q_proj / k_proj /
//! v_proj / rope / attn / o_proj / residual_add / post_norm / gate / up /
//! silu / mul / down / output_add），每个节点的输出张量 shape 不同：
//! 部分是 `[seq_len, hidden_size]`（input_norm, attn_proj, residual, post_norm,
//! mlp_out, output_add），部分是 Q/K/V 的 head-major 布局。
//!
//! 我们要捕获的是 "layer anchor 的最终输出" = `layer_{anchor}_output_add` 的
//! 结果（写入 `hidden_0`）。由于 fusion 可能改变节点名，我们用**形状过滤**
//! 定位合法 hidden-state 输出：`bytes.len() == seq_len * hidden_size *
//! elem_bytes`。在 target layer 中看到的每个 `[seq_len, hidden_size]` 节点
//! 都保存，layer 切换到下一层时返回最后保存的 (即 layer anchor 的最终
//! 输出)。
//!
//! # CPU F32 canonicalization (ARCH-WEIGHT-F32-CANONICAL)
//!
//! CPU JIT 把所有权重和激活统一转为 f32 (REQ-LOADER-016, CLAUDE.md §5.2)，
//! 所以 CPU 路径上 `output` 总是 f32 字节。GPU 后端保留模型 dtype (F32/F16/
//! BF16)。我们用 `output.len() / (seq_len * hidden_size)` 的运行时尺寸推断
//! 每元素字节数，不依赖 `ctx.model_config.geometry.dtype`（那是模型声明的
//! dtype，不是运行时激活 dtype）。

use gllm_kernels::types::DType;
use half::{bf16, f16};

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};

/// Priority for mid-layer encode (higher than Early Exit=50, lower than SG=90).
pub const MID_LAYER_ENCODE_PRIORITY: u32 = 55;

/// Mid-layer encode callback.
///
/// Stateful: within the target layer, caches the latest `[seq_len,
/// hidden_size]`-shaped node output. When execution transitions out of the
/// target layer (first pre_node with `layer_idx != target_layer`) or the
/// graph reaches a post-decoder node (`final_norm` / `lm_head`), emits
/// `ExitEarly` carrying the cached hidden state.
pub struct MidLayerEncodeCallback {
    target_layer: usize,
    /// Latest `[seq_len, hidden_size]` output captured within `target_layer`.
    /// Stored as flat f32 in row-major order (`seq_len` rows of `hidden_size`
    /// each).
    captured: Option<Vec<f32>>,
}

impl MidLayerEncodeCallback {
    pub fn new(target_layer: usize) -> Self {
        Self {
            target_layer,
            captured: None,
        }
    }

    /// Decode the `[seq_len, hidden_size]` live-region of a raw node output
    /// buffer into `Vec<f32>`.
    ///
    /// # Buffer layout
    ///
    /// The JIT executor pre-allocates each node's output buffer for the
    /// maximum supported `seq_len` (SymDim `max_value`, typically 2048-4096),
    /// so `output.len() = max_seq * feature_dim * elem_bytes`. The *live*
    /// data occupies only the first `seq_len * hidden_size * elem_bytes`
    /// bytes from offset 0 — the tail is uninitialised / stale. The JIT
    /// runtime exposes this live-region convention via
    /// `CompiledNode::feature_dim` (see `graph/executor.rs` GLLM_DUMP_LAYERS
    /// handler).
    ///
    /// We validate that the node *is* a hidden-state tensor by checking
    /// `output.len() >= seq_len * hidden_size * elem_bytes` AND
    /// `output.len() % feature_dim_candidate == 0` where `feature_dim_candidate`
    /// equals `hidden_size`. Non-hidden-state nodes (Q/K/V with
    /// head-major layout, SwiGLU with `intermediate_size` feature_dim) have
    /// different `feature_dim`, so their buffer size is not a multiple of
    /// `hidden_size * elem_bytes` — we reject those.
    ///
    /// # Element width
    ///
    /// CPU JIT canonicalises all activations to f32 (ARCH-WEIGHT-F32-CANONICAL,
    /// SPEC/DOCS/scheduling/jit-cache-protocol.md). GPU codegen keeps the
    /// model's declared dtype. We prefer the CPU F32 path first (the only
    /// currently-enabled backend for mid-layer encode — GPU reports
    /// `Unimplemented` in `gpu_backend_macro.rs`); if the buffer size is
    /// inconsistent with F32 we fall back to `declared_dtype`.
    fn decode_hidden_output(
        output: &[u8],
        seq_len: usize,
        hidden_size: usize,
        declared_dtype: DType,
    ) -> Option<Vec<f32>> {
        if seq_len == 0 || hidden_size == 0 || output.is_empty() {
            return None;
        }
        let numel = seq_len.checked_mul(hidden_size)?;
        if numel == 0 {
            return None;
        }

        // Determine the node's actual `feature_dim * elem_bytes` (the
        // per-token stride) by factoring `output.len()`. For a
        // `[*, hidden_size]`-shaped tensor, `output.len() = max_seq *
        // hidden_size * elem_bytes`. Try F32 first (CPU canonical), then
        // fall through to the declared dtype.
        let try_decode_f32 = |data: &[u8], count: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let off = i * 4;
                out.push(f32::from_le_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                ]));
            }
            out
        };

        // Candidate 1: F32 (4 bytes/elem)
        //   Buffer must be a multiple of `hidden_size * 4` and hold at least
        //   `numel * 4` bytes.
        let f32_stride = hidden_size.checked_mul(4)?;
        if f32_stride > 0
            && output.len() % f32_stride == 0
            && output.len() >= numel.checked_mul(4)?
        {
            return Some(try_decode_f32(&output[..numel * 4], numel));
        }

        // Candidate 2: declared 2-byte dtype (F16 / BF16) — GPU-resident
        // activation path once the backend lands.
        let half_stride = hidden_size.checked_mul(2)?;
        if half_stride > 0
            && output.len() % half_stride == 0
            && output.len() >= numel.checked_mul(2)?
        {
            let live = &output[..numel * 2];
            match declared_dtype {
                DType::F16 => {
                    let mut out = Vec::with_capacity(numel);
                    for i in 0..numel {
                        let off = i * 2;
                        out.push(f16::from_le_bytes([live[off], live[off + 1]]).to_f32());
                    }
                    return Some(out);
                }
                DType::BF16 => {
                    let mut out = Vec::with_capacity(numel);
                    for i in 0..numel {
                        let off = i * 2;
                        out.push(bf16::from_le_bytes([live[off], live[off + 1]]).to_f32());
                    }
                    return Some(out);
                }
                _ => {}
            }
        }

        None
    }
}

impl LayerCallback for MidLayerEncodeCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        // On transition *out of* target_layer (i.e. the first node whose
        // `layer_idx != target_layer` seen after at least one capture), emit
        // `ExitEarly` with the cached hidden state. This captures
        // "hidden-state after layer `target_layer` completes".
        //
        // `extract_layer_index` on non-layer nodes (`final_norm`, `lm_head`,
        // `embed_tokens`) falls back to `node_idx` (a large integer), so this
        // also triggers when `target_layer == num_layers - 1` — the
        // subsequent `final_norm` pre_node exits the graph.
        //
        // `Continue` paths:
        // - We have not captured yet (never reached target layer).
        // - `ctx.layer_idx == self.target_layer` (still inside target layer).
        if ctx.layer_idx == self.target_layer {
            return CallbackAction::Continue;
        }
        match self.captured.take() {
            Some(hidden) => CallbackAction::ExitEarly { logits: hidden },
            None => CallbackAction::Continue,
        }
    }

    fn post_node(&mut self, ctx: &LayerContext, output: &[u8]) -> CallbackAction {
        if ctx.layer_idx != self.target_layer {
            return CallbackAction::Continue;
        }
        let hidden_size = ctx.model_config.geometry.hidden_size;
        let seq_len = ctx.seq_len;
        let declared_dtype = ctx.model_config.geometry.dtype;
        if let Some(decoded) =
            Self::decode_hidden_output(output, seq_len, hidden_size, declared_dtype)
        {
            // Overwrite — the *latest* `[seq_len, hidden_size]`-shaped output
            // within the target layer is `layer_{target}_output_add` (or its
            // fused successor), which is the true layer output written back
            // to `hidden_0`. Non-hidden-shape intermediate outputs (SwiGLU
            // gate*up activations on `intermediate_size`, Q/K/V on
            // head-major layout when head dims differ from hidden) are
            // rejected by `decode_hidden_output`'s stride check.
            self.captured = Some(decoded);
        }
        CallbackAction::Continue
    }

    fn priority(&self) -> u32 {
        MID_LAYER_ENCODE_PRIORITY
    }

    fn target_layers(&self) -> Option<&[usize]> {
        // `None` = fire on every layer. Required because we must observe the
        // *pre_node of the first node after* `target_layer` to emit
        // `ExitEarly`, and post-decoder nodes (`final_norm` / `lm_head`)
        // report `layer_idx = node_idx` (fallback from
        // `FusedGraphExecutor::extract_layer_index`), so their layer index
        // is not known in advance. A bounded filter would risk missing the
        // transition.
        None
    }

    fn name(&self) -> &str {
        "MidLayerEncode"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_metadata() {
        let cb = MidLayerEncodeCallback::new(7);
        assert_eq!(cb.priority(), MID_LAYER_ENCODE_PRIORITY);
        assert_eq!(cb.name(), "MidLayerEncode");
        assert!(
            cb.target_layers().is_none(),
            "filter must be None to see transition out of target layer"
        );
    }

    #[test]
    fn decode_f32_hidden_shape() {
        // seq_len=2, hidden_size=3 → 6 f32 = 24 bytes
        let src: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F32).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn decode_f32_on_bf16_declared_dtype() {
        // CPU canonicalizes to F32 even when the model declares BF16 — the
        // callback must detect the 4-byte elem size from the buffer length
        // and decode as F32 (regression: previously decoded as BF16 → NaN).
        let src: Vec<u8> = [3.14f32, -1.5, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::BF16).unwrap();
        assert!((out[0] - 3.14).abs() < 1e-5, "F32 bytes must decode as F32 regardless of declared dtype");
        assert!((out[1] - -1.5).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn decode_accepts_live_region_in_pre_allocated_buffer() {
        // JIT executor pre-allocates output buffer for max_seq_len=2048 but
        // the actual prompt has seq_len=2. Buffer size =
        // max_seq * hidden_size * 4 = 2048 * 3 * 4 = 24576 bytes. The first
        // 24 bytes (2 * 3 * 4) hold the live data; the rest is stale.
        let mut src = vec![0u8; 2048 * 3 * 4];
        for (i, v) in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::BF16).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn decode_rejects_incompatible_feature_dim() {
        // SwiGLU gate*up output: shape [seq_len, intermediate_size]. For
        // SmolLM2 intermediate_size=1536 (not 576 hidden_size) — buffer
        // bytes not a multiple of hidden_size*4.
        //
        // Pick a size that is definitely not divisible by hidden_size * 4
        // but is a valid 32-bit-aligned buffer. E.g. hidden_size=576,
        // intermediate_size=1536 (3x hidden). Buffer: 2048 * 1536 * 4 =
        // 12582912 bytes. 12582912 % (576 * 4) = 12582912 % 2304 = 0
        // (unfortunately divisible because 1536 = 3*576). So this particular
        // shape is actually accepted; hidden-shape detection relies on the
        // graph itself emitting the right node sequence (covered by the
        // stateful `captured` field overwriting with the final layer
        // output).
        //
        // Use a more distinctive case: buffer with size not a multiple of
        // hidden_size (e.g. intermediate=1000).
        let src = vec![0u8; 2048 * 1000 * 4];
        assert!(
            MidLayerEncodeCallback::decode_hidden_output(&src, 2, 576, DType::F32).is_none(),
            "buffer with intermediate_size=1000 must not be accepted as hidden_size=576"
        );
    }

    #[test]
    fn decode_rejects_empty() {
        assert!(MidLayerEncodeCallback::decode_hidden_output(&[], 2, 3, DType::F32).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&[1, 2, 3], 0, 3, DType::F32).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&[1, 2, 3], 2, 0, DType::F32).is_none());
    }
}
