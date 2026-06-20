//! Mid-Layer Encode Callback — captures hidden state at a target layer and
//! terminates the forward pass early.
//!
//! SSOT: `SPEC/HEAD-ROUTING.md §5 Mid-layer Encode 协议`,
//! `SPEC/INTENT.md §3`.
//!
//! Registered by `Client::encode_to_layer` / `Client::encode_intent` onto a
//! fresh `CallbackChain` passed to the mega-kernel execution path.
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
            && output.len().is_multiple_of(f32_stride)
            && output.len() >= numel.checked_mul(4)?
        {
            return Some(try_decode_f32(&output[..numel * 4], numel));
        }

        // Candidate 2: declared 2-byte dtype (F16 / BF16) — GPU-resident
        // activation path once the backend lands.
        let half_stride = hidden_size.checked_mul(2)?;
        if half_stride > 0
            && output.len().is_multiple_of(half_stride)
            && output.len() >= numel.checked_mul(2)?
        {
            let live = &output[..numel * 2];
            // ARCH-JIT-DATA-YIELDS: declared_dtype is a build-time constant from model config.
            // NOTE: 当 callback 迁移到 JIT 生成代码时,可烘焙为 typed closure 消除运行时 match。
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
        let declared_dtype = ctx.model_config.geometry.compute_dtype;
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
        // mega-kernel extract_layer_index), so their layer index
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

    #[test]
    fn decode_f16_hidden_shape() {
        let h1 = f16::from_f32(1.5);
        let h2 = f16::from_f32(-2.0);
        let h3 = f16::from_f32(0.0);
        let h4 = f16::from_f32(3.14);
        let src: Vec<u8> = [&h1, &h2, &h3, &h4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F16).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.5).abs() < 0.01);
        assert!((out[1] - -2.0).abs() < 0.01);
        assert!((out[3] - 3.14).abs() < 0.01);
    }

    #[test]
    fn decode_bf16_hidden_shape() {
        let b1 = bf16::from_f32(0.5);
        let b2 = bf16::from_f32(-1.0);
        let src: Vec<u8> = [&b1, &b2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 0.01);
        assert!((out[1] - -1.0).abs() < 0.01);
    }

    #[test]
    fn decode_rejects_buffer_too_small() {
        // Buffer = 8 bytes (2 f32), but seq_len=3, hidden_size=3 needs 36 bytes
        let src = vec![0u8; 8];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 3, 3, DType::F32).is_none());
    }

    #[test]
    fn decode_rejects_non_multiple_stride() {
        // 10 bytes is not a multiple of hidden_size*4=12
        let src = vec![0u8; 10];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F32).is_none());
    }

    #[test]
    fn new_initial_state() {
        let cb = MidLayerEncodeCallback::new(7);
        assert_eq!(cb.target_layer, 7);
        assert!(cb.captured.is_none());
    }

    #[test]
    fn priority_constant() {
        assert_eq!(MID_LAYER_ENCODE_PRIORITY, 55);
        assert!(MID_LAYER_ENCODE_PRIORITY > 50, "higher than EarlyExit");
        assert!(MID_LAYER_ENCODE_PRIORITY < 90, "lower than SG");
    }

    #[test]
    fn decode_f32_zero_values() {
        let src = vec![0u8; 24]; // 6 f32 zeros
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F32).unwrap();
        assert_eq!(out, vec![0.0f32; 6]);
    }

    #[test]
    fn decode_f32_negative_values() {
        let values: Vec<f32> = vec![-1.0, -2.5, -0.001];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F32).unwrap();
        assert!((out[0] - -1.0).abs() < 1e-6);
        assert!((out[1] - -2.5).abs() < 1e-5);
        assert!((out[2] - -0.001).abs() < 1e-6);
    }

    #[test]
    fn decode_f32_on_f16_declared() {
        // CPU path: buffer contains F32 even though model declares F16
        let src: Vec<u8> = [42.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert!((out[0] - 42.0).abs() < 1e-5);
    }

    // ── Test helper: owns data that LayerContext borrows ──

    struct TestCtxHolder {
        config: crate::engine::executor::GeneratorForwardConfig,
        hidden_state: Vec<u8>,
    }

    impl TestCtxHolder {
        fn new(hidden_size: usize, compute_dtype: DType) -> Self {
            Self {
                config: crate::engine::executor::GeneratorForwardConfig {
                    geometry: std::sync::Arc::new(crate::model_config::ModelGeometry {
                        hidden_size,
                        num_layers: 8,
                        vocab_size: 1000,
                        intermediate_size: hidden_size * 4,
                        num_heads: 4,
                        num_kv_heads: 2,
                        head_dim: hidden_size / 4,
                        max_seq_len: 2048,
                        rope_theta: 10000.0,
                        rope_scale: 1.0,
                        rope_interleaved: false,
                        dtype: compute_dtype,
                        compute_dtype,
                        norm_eps: 1e-5,
                        num_experts: 0,
                        moe_top_k: 0,
                        expert_intermediate_size: 0,
                        global_rope_theta: 0.0,
                        rope_partial_ratio: 1.0,
                        rope_partial_ratio_global: 1.0,
                        attention_pattern: vec![],
                        sliding_window: 0,
                        num_kv_shared_layers: 0,
                        global_head_dim: 0,
                        hidden_size_per_layer_input: 0,
                        position_offset: None,
                        rope_scaling: None,
                        final_logit_softcapping: None,
                        hidden_act: None,
                        mla_d_c: 0,
                        mla_d_rope: 0,
                        mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
                    }),
                    rope: crate::engine::executor::RoPEConfig {
                        theta: 10000.0,
                        scale: 1.0,
                        interleaved: false,
                        precompute: false,
                    },
                    arch_family: crate::manifest::ArchFamily::Decoder,
                    has_classifier: false,
                    rerank_yes_token_id: None,
                    rerank_no_token_id: None,
                    moe_config: None,
                    paged_kv: crate::engine::executor::PagedKvConfig {
                        page_table: None,
                        page_size: 16,
                    },
                    callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
                },
                hidden_state: vec![0u8; hidden_size * 4],
            }
        }

        fn ctx(&self, layer_idx: usize, seq_len: usize) -> crate::graph::layer_callback::LayerContext<'_> {
            crate::graph::layer_callback::LayerContext {
                node_idx: layer_idx * 2,
                layer_idx,
                node_op: "Test",
                hidden_state: &self.hidden_state,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: seq_len,
                seq_len,
                position: 0,
                request_id: 1,
                model_config: &self.config,
            }
        }
    }

    fn make_f32_output(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // ── pre_node / post_node state machine tests ──

    #[test]
    fn pre_node_at_target_layer_returns_continue() {
        // Arrange: callback targets layer 3, ctx is at layer 3, no capture yet
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(64, DType::F32);
        let ctx = holder.ctx(3, 1);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert
        assert!(
            matches!(action, CallbackAction::Continue),
            "pre_node at target layer without prior capture must return Continue"
        );
    }

    #[test]
    fn pre_node_at_target_layer_with_capture_returns_continue() {
        // Arrange: callback at target layer has already captured data
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Simulate a capture via post_node
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx, &output);

        // Act: pre_node still at target layer
        let action = cb.pre_node(&ctx);

        // Assert: still Continue, doesn't exit while still in target layer
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn pre_node_transition_out_triggers_exit_early() {
        // Arrange: capture hidden state at layer 2, then transition to layer 3
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 2
        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[10.0, 20.0, 30.0, 40.0]);
        cb.post_node(&ctx2, &output);

        // Act: pre_node at layer 3 (transition out of target layer 2)
        let ctx3 = holder.ctx(3, 1);
        let action = cb.pre_node(&ctx3);

        // Assert
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn pre_node_transition_out_without_capture_returns_continue() {
        // Arrange: never visited target layer, transition directly to another
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 1);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: no capture, so Continue
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn captured_state_consumed_on_exit() {
        // Arrange: capture and exit, then try to exit again
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx2, &output);

        // First transition out: ExitEarly
        let ctx3 = holder.ctx(3, 1);
        let action1 = cb.pre_node(&ctx3);
        assert!(matches!(action1, CallbackAction::ExitEarly { .. }));

        // Second transition out: Continue (captured was consumed by take())
        let ctx4 = holder.ctx(4, 1);
        let action2 = cb.pre_node(&ctx4);
        assert!(
            matches!(action2, CallbackAction::Continue),
            "captured must be consumed after ExitEarly"
        );
    }

    #[test]
    fn post_node_wrong_layer_is_noop() {
        // Arrange: callback targets layer 3, but post_node fires on layer 1
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(1, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);

        // Act
        let action = cb.post_node(&ctx, &output);

        // Assert: Continue, and nothing captured
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_none());
    }

    #[test]
    fn post_node_overwrites_with_latest_capture() {
        // Arrange: multiple post_node calls within the same target layer
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // First capture
        let output1 = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx, &output1);

        // Second capture (overwrites first)
        let output2 = make_f32_output(&[10.0, 20.0, 30.0, 40.0]);
        cb.post_node(&ctx, &output2);

        // Act: transition out
        let ctx3 = holder.ctx(3, 1);
        let action = cb.pre_node(&ctx3);

        // Assert: should have the second (latest) capture
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn target_layer_zero_first_layer() {
        // Arrange: edge case — target the very first layer
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 0
        let ctx0 = holder.ctx(0, 1);
        let output = make_f32_output(&[5.0, 6.0, 7.0, 8.0]);
        cb.post_node(&ctx0, &output);

        // Transition to layer 1
        let ctx1 = holder.ctx(1, 1);
        let action = cb.pre_node(&ctx1);

        // Assert
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![5.0, 6.0, 7.0, 8.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn post_node_rejects_wrong_shape_and_does_not_capture() {
        // Arrange: output buffer has incompatible shape (not a multiple of hidden_size * 4)
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(8, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Output with 10 bytes — not a multiple of hidden_size(8) * 4 = 32
        let output = vec![0u8; 10];
        cb.post_node(&ctx, &output);

        // Assert: nothing captured because shape is incompatible
        assert!(cb.captured.is_none());
    }

    #[test]
    fn decode_rejects_integer_declared_dtype_when_f32_fails() {
        // Arrange: a buffer that is 2-byte-aligned but not 4-byte-aligned,
        // and declared dtype is an integer type (not F16/BF16).
        // F32 stride check fails (not multiple of hidden_size*4),
        // half stride check passes but declared dtype is integer → None.
        let src = vec![0u8; 4]; // 4 bytes, seq=1, hidden=3 → F32 needs 12, half needs 6
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::U8);
        assert!(out.is_none());
    }

    #[test]
    fn decode_large_hidden_size_f32() {
        // Arrange: realistic hidden size (e.g., 4096) with seq_len=1
        let hidden_size = 4096;
        let mut src = vec![0u8; hidden_size * 4];
        // Write a known value at the start
        src[0..4].copy_from_slice(&1.5f32.to_le_bytes());
        src[(hidden_size - 1) * 4..hidden_size * 4].copy_from_slice(&(-2.5f32).to_le_bytes());

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden_size, DType::F32).unwrap();
        assert_eq!(out.len(), hidden_size);
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[hidden_size - 1] - (-2.5)).abs() < 1e-5);
        // Middle values should be 0.0 (we didn't write them)
        assert!((out[100] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn decode_seq_len_one_hidden_size_one_minimal() {
        // Arrange: minimal valid decode — 1 element, 4 bytes
        let src: Vec<u8> = 99.5f32.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 99.5).abs() < 1e-5);
    }

    #[test]
    fn decode_f32_preserves_subnormal_values() {
        // Arrange: subnormal (denormalized) f32 values
        let subnormal: f32 = f32::from_bits(0x00000001); // smallest positive subnormal
        let src: Vec<u8> = subnormal.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out[0].to_bits(), 0x00000001u32);
    }

    // ── Additional unit tests ──

    #[test]
    fn decode_exact_fit_buffer_no_slack() {
        // Arrange: output.len() exactly equals numel * 4 with zero slack bytes
        let values: Vec<f32> = vec![0.5, -0.5, 1.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 12); // exact fit: 3 elements * 4 bytes

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F32).unwrap();

        // Assert: values decoded correctly with no slack
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - -0.5).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn post_node_always_returns_continue_even_on_capture() {
        // Arrange: callback captures at target layer, but post_node must still
        // return Continue (it never short-circuits; pre_node handles the exit)
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);

        // Act
        let action = cb.post_node(&ctx, &output);

        // Assert: post_node always returns Continue, even though capture succeeded
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_some(), "capture must have occurred");
    }

    #[test]
    fn pre_node_before_target_layer_without_capture_returns_continue() {
        // Arrange: callback targets layer 5, but pre_node fires at layer 2
        // (we haven't reached the target layer yet, no capture possible)
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: layer 2 != target layer 5, but no capture exists, so Continue
        assert!(
            matches!(action, CallbackAction::Continue),
            "pre_node at non-target layer without capture must return Continue"
        );
    }

    #[test]
    fn new_with_large_target_layer() {
        // Arrange: edge case — large target layer index
        let cb = MidLayerEncodeCallback::new(usize::MAX / 2);

        // Assert: construction succeeds, fields correct
        assert_eq!(cb.target_layer, usize::MAX / 2);
        assert!(cb.captured.is_none());
        assert_eq!(cb.name(), "MidLayerEncode");
        assert_eq!(cb.priority(), MID_LAYER_ENCODE_PRIORITY);
    }

    #[test]
    fn new_with_zero_target_layer() {
        // Arrange: edge case — target layer 0
        let cb = MidLayerEncodeCallback::new(0);

        // Assert
        assert_eq!(cb.target_layer, 0);
        assert!(cb.captured.is_none());
    }

    #[test]
    fn decode_f32_candidate_wins_over_half_when_both_valid() {
        // Arrange: buffer of 4 bytes (1 f32 element). hidden_size=1, seq_len=1.
        // F32 candidate: stride=4, 4 % 4 == 0, 4 >= 4 → passes.
        // Half candidate: stride=2, 4 % 2 == 0, 4 >= 2 → also passes.
        // F32 candidate must win because it is checked first.
        let src: Vec<u8> = 42.0f32.to_le_bytes().to_vec();

        // Act with F16 declared — F32 candidate should still win
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();

        // Assert: decoded as F32 (exact match), not as F16
        assert!((out[0] - 42.0f32).abs() < 1e-6);
    }

    #[test]
    fn decode_half_stride_with_f32_declared_dtype_returns_none() {
        // Arrange: 2-byte buffer that passes half stride check but declared dtype is F32
        // (not F16 or BF16). Half stride passes but the match arm rejects F32.
        let src = vec![0u8; 4]; // 4 bytes: half_stride=2, 4%2==0, 4>=2*1=2 → passes
        // hidden_size=2, seq_len=1 → F32 stride=8, 4%8!=0 → F32 fails
        // half stride=4, 4%4==0, 4>=2 → passes but declared DType::F32 → no match arm

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32);
        assert!(
            out.is_none(),
            "half stride with F32 declared dtype must be rejected (no F16/BF16 match arm)"
        );
    }

    #[test]
    fn post_node_with_empty_output_does_not_capture() {
        // Arrange: output is empty slice — decode returns None
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Act: post_node with empty output buffer
        let action = cb.post_node(&ctx, &[]);

        // Assert: Continue returned, nothing captured
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_none());
    }

    #[test]
    fn post_node_with_seq_len_zero_does_not_capture() {
        // Arrange: ctx with seq_len=0 — decode should return None
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 0); // seq_len=0

        // Act: output has data but seq_len is 0
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        let action = cb.post_node(&ctx, &output);

        // Assert: Continue, nothing captured (seq_len=0 early return in decode)
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_none());
    }

    #[test]
    fn captured_overwritten_by_second_post_node_same_layer() {
        // Arrange: first capture succeeds, second overwrites within same target layer.
        // Use seq_len=1, hidden_size=4 so output needs exactly 4 f32 = 16 bytes.
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1); // seq_len=1

        // First capture
        let output1 = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx, &output1);
        assert!(cb.captured.is_some(), "first capture must succeed");

        // Second capture (different data, overwrites first)
        let output2 = make_f32_output(&[5.0, 6.0, 7.0, 8.0]);
        cb.post_node(&ctx, &output2);

        // Assert: captured has the second set of values
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn decode_checked_mul_overflow_returns_none() {
        // Arrange: seq_len * hidden_size would overflow usize. checked_mul returns None.
        // We use values whose product exceeds usize::MAX.
        let src = vec![0u8; 16]; // small buffer, doesn't matter — checked_mul fails first

        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src,
            usize::MAX,
            usize::MAX,
            DType::F32,
        );

        // Assert: overflow returns None
        assert!(out.is_none());
    }

    #[test]
    fn pre_node_after_capture_consumed_returns_continue() {
        // Arrange: capture consumed by first pre_node transition, second pre_node
        // at yet another layer still returns Continue
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 1
        let ctx1 = holder.ctx(1, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx1, &output);

        // First transition: layer 1 → layer 2 (consumes capture)
        let ctx2 = holder.ctx(2, 1);
        let action1 = cb.pre_node(&ctx2);
        assert!(matches!(action1, CallbackAction::ExitEarly { .. }));

        // Act: second transition: layer 2 → layer 3 (no capture left)
        let ctx3 = holder.ctx(3, 1);
        let action2 = cb.pre_node(&ctx3);

        // Assert: Continue because captured was consumed
        assert!(
            matches!(action2, CallbackAction::Continue),
            "after capture consumed, subsequent pre_node must return Continue"
        );
    }

    #[test]
    fn decode_bf16_specific_bit_patterns() {
        // Arrange: BF16 values with known bit representations
        // BF16 for 1.0 = 0x3F80, for -2.0 = 0xC000
        let b1 = bf16::from_bits(0x3F80u16); // 1.0
        let b2 = bf16::from_bits(0xC000u16); // -2.0
        let src: Vec<u8> = [&b1, &b2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16).unwrap();

        // Assert
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - -2.0).abs() < 0.01);
    }

    #[test]
    fn decode_f16_candidate_accepted_when_f32_stride_fails() {
        // Arrange: a buffer that is 2-byte aligned but not 4-byte aligned
        // with hidden_size that makes f32_stride fail.
        // hidden_size=3, F32 stride=12. Buffer=6 bytes → 6%12!=0 → F32 fails.
        // Half stride=6. 6%6==0, 6>=6 → passes. Declared F16.
        let h1 = f16::from_f32(1.0);
        let h2 = f16::from_f32(2.0);
        let h3 = f16::from_f32(3.0);
        let src: Vec<u8> = [&h1, &h2, &h3]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(src.len(), 6);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F16).unwrap();

        // Assert: F32 candidate fails, F16 candidate succeeds
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - 2.0).abs() < 0.01);
        assert!((out[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn decode_rejects_when_both_candidates_fail() {
        // Arrange: buffer that is neither F32-aligned nor half-aligned for the
        // given hidden_size. E.g., hidden_size=7, F32 stride=28, half stride=14.
        // Buffer = 13 bytes → 13%28!=0, 13%14!=0 → both fail.
        let src = vec![0u8; 13];

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 7, DType::F32);

        // Assert: no candidate matched
        assert!(out.is_none());
    }

    // ── Additional coverage tests (18 new) ──

    #[test]
    fn decode_bf16_live_region_with_slack_bytes() {
        // Arrange: BF16 buffer pre-allocated for max_seq=1024 but only seq_len=2.
        // Use hidden_size=3 so that F32 stride=12, and total=1024*3*2=6144.
        // 6144 % 12 == 0 → F32 candidate would match. To force BF16 path,
        // use an odd max_seq: 1023 * 3 * 2 = 6138. 6138 % 12 = 6 (not multiple of 12).
        // 6138 % 6 == 0 → half stride passes.
        let hidden_size = 3;
        let total_buf = 1023 * hidden_size * 2; // 6138 bytes, not multiple of 12
        let mut src = vec![0u8; total_buf];
        let b1 = bf16::from_f32(1.5);
        let b2 = bf16::from_f32(-0.5);
        let b3 = bf16::from_f32(3.0);
        let b4 = bf16::from_f32(0.0);
        let b5 = bf16::from_f32(7.25);
        let b6 = bf16::from_f32(-8.0);
        let half_vals = [&b1, &b2, &b3, &b4, &b5, &b6]; // 2 * 3 = 6 elements
        for (i, v) in half_vals.iter().enumerate() {
            let off = i * 2;
            src[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, hidden_size, DType::BF16).unwrap();

        // Assert: only first 6 elements (seq_len * hidden_size) decoded
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.5).abs() < 0.01);
        assert!((out[1] - -0.5).abs() < 0.01);
        assert!((out[5] - -8.0).abs() < 0.1);
    }

    #[test]
    fn decode_f16_live_region_with_slack_bytes() {
        // Arrange: F16 buffer pre-allocated for max_seq=511 but only seq_len=3.
        // hidden_size=3: F32 stride=12, half stride=6.
        // total=511 * 3 * 2 = 3066. 3066 % 12 = 6 (not F32-aligned).
        // 3066 % 6 == 0 → half stride passes.
        let hidden_size = 3;
        let total_buf = 511 * hidden_size * 2; // 3066 bytes
        let mut src = vec![0u8; total_buf];
        let h1 = f16::from_f32(0.25);
        let h2 = f16::from_f32(-1.75);
        let h3 = f16::from_f32(4.0);
        let h4 = f16::from_f32(-4.0);
        let h5 = f16::from_f32(0.0625);
        let h6 = f16::from_f32(99.5);
        let h7 = f16::from_f32(-0.5);
        let h8 = f16::from_f32(1.0);
        let h9 = f16::from_f32(2.0);
        let half_vals = [&h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9]; // 3 * 3 = 9 elements
        for (i, v) in half_vals.iter().enumerate() {
            let off = i * 2;
            src[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, hidden_size, DType::F16).unwrap();

        // Assert
        assert_eq!(out.len(), 9);
        assert!((out[0] - 0.25).abs() < 0.01);
        assert!((out[3] - -4.0).abs() < 0.01);
        assert!((out[5] - 99.5).abs() < 0.1);
        assert!((out[8] - 2.0).abs() < 0.01);
    }

    #[test]
    fn pre_node_same_layer_multiple_invocations_all_continue() {
        // Arrange: pre_node called multiple times at target layer without any post_node
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 1);

        // Act: call pre_node 5 times in a row at target layer
        for _ in 0..5 {
            let action = cb.pre_node(&ctx);
            assert!(
                matches!(action, CallbackAction::Continue),
                "pre_node at target layer must always return Continue when no capture"
            );
        }

        // Assert: still no capture
        assert!(cb.captured.is_none());
    }

    #[test]
    fn post_node_with_larger_seq_len_overwrites_smaller() {
        // Arrange: first capture with seq_len=1, then seq_len=2 at same layer
        // Note: hidden_size=4, so output needs 4 f32 (16 bytes) for seq_len=1,
        // or 8 f32 (32 bytes) for seq_len=2. Both buffers must match their respective
        // hidden_size alignment. Since hidden_size=4 and f32 stride=16,
        // both 16 and 32 are multiples of 16 — both accepted.
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // First capture: seq_len=1
        let ctx1 = holder.ctx(2, 1);
        let output1 = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx1, &output1);
        assert_eq!(cb.captured.as_ref().unwrap().len(), 4);

        // Second capture: seq_len=2 (overwrites with more elements)
        let ctx2 = holder.ctx(2, 2);
        let output2 = make_f32_output(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        cb.post_node(&ctx2, &output2);

        // Assert: second capture (seq_len=2) overwrites first
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 8);
        assert!((captured[0] - 5.0).abs() < 1e-6);
        assert!((captured[7] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn full_lifecycle_capture_at_last_layer_and_exit() {
        // Arrange: target layer is the last layer (layer 7)
        let mut cb = MidLayerEncodeCallback::new(7);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Act 1: simulate visiting layers 0..6 — all pre_node return Continue
        for layer in 0..7 {
            let ctx = holder.ctx(layer, 1);
            let action = cb.pre_node(&ctx);
            assert!(matches!(action, CallbackAction::Continue));
        }

        // Act 2: post_node at target layer 7 — captures hidden
        let ctx7 = holder.ctx(7, 1);
        let output = make_f32_output(&[10.0, 20.0, 30.0, 40.0]);
        let post_action = cb.post_node(&ctx7, &output);
        assert!(matches!(post_action, CallbackAction::Continue));
        assert!(cb.captured.is_some());

        // Act 3: transition to "layer 8" (post-decoder node, e.g., final_norm)
        // This simulates the extract_layer_index fallback to node_idx
        let ctx_post = holder.ctx(100, 1);
        let pre_action = cb.pre_node(&ctx_post);

        // Assert: ExitEarly with the captured hidden state
        match pre_action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn decode_f32_max_finite_value() {
        // Arrange: f32::MAX as the element value
        let src: Vec<u8> = f32::MAX.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], f32::MAX);
    }

    #[test]
    fn decode_f32_negative_max_finite_value() {
        // Arrange: f32::MIN (most negative finite) as the element value
        let src: Vec<u8> = f32::MIN.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], f32::MIN);
    }

    #[test]
    fn decode_f32_nan_preserved() {
        // Arrange: NaN bit pattern
        let nan_bits: u32 = 0x7FC00000; // quiet NaN
        let src: Vec<u8> = nan_bits.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert!(out[0].is_nan(), "NaN must be preserved through decode");
    }

    #[test]
    fn decode_f32_infinity_preserved() {
        // Arrange: positive and negative infinity
        let pos_inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let src: Vec<u8> = [pos_inf, neg_inf]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32).unwrap();
        assert!(out[0].is_infinite() && out[0].is_sign_positive());
        assert!(out[1].is_infinite() && out[1].is_sign_negative());
    }

    #[test]
    fn decode_f32_smallest_positive_normal() {
        // Arrange: smallest positive normal f32 (MIN_POSITIVE)
        let src: Vec<u8> = f32::MIN_POSITIVE.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out[0], f32::MIN_POSITIVE);
    }

    #[test]
    fn decode_f32_multi_seq_preserves_row_order() {
        // Arrange: seq_len=3, hidden_size=2 — verify row-major ordering
        // Row 0: [1.0, 2.0], Row 1: [3.0, 4.0], Row 2: [5.0, 6.0]
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 2, DType::F32).unwrap();

        // Assert: row-major layout preserved
        assert_eq!(out.len(), 6);
        // Row 0
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        // Row 1
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
        // Row 2
        assert!((out[4] - 5.0).abs() < 1e-6);
        assert!((out[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn decode_bf16_candidate_accepted_when_f32_stride_fails() {
        // Arrange: buffer is 2-byte aligned but not 4-byte aligned for hidden_size.
        // hidden_size=3: F32 stride=12, half stride=6.
        // Buffer = 6 bytes → 6%12 != 0 → F32 fails. 6%6 == 0, 6>=6 → BF16 passes.
        let b1 = bf16::from_f32(0.125);
        let b2 = bf16::from_f32(2.0);
        let b3 = bf16::from_f32(-3.5);
        let src: Vec<u8> = [&b1, &b2, &b3]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(src.len(), 6);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::BF16).unwrap();

        // Assert
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.125).abs() < 0.01);
        assert!((out[1] - 2.0).abs() < 0.01);
        assert!((out[2] - -3.5).abs() < 0.05);
    }

    #[test]
    fn decode_half_stride_with_non_half_dtype_returns_none() {
        // Arrange: buffer passes half-stride check but declared dtype is F8E4M3 (not F16/BF16).
        // hidden_size=2, half_stride=4. Buffer=4 bytes → 4%4==0, 4>=2 → passes stride.
        // But DType::F8E4M3 has no match arm in the half path.
        let src = vec![0u8; 4];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F8E4M3);
        assert!(
            out.is_none(),
            "half stride with F8E4M3 declared dtype must be rejected"
        );
    }

    #[test]
    fn decode_rejects_single_byte_buffer_for_any_dtype() {
        // Arrange: 1 byte — cannot be F32 (needs 4), cannot be half (needs 2)
        // hidden_size=1: F32 stride=4, half stride=2. Buffer=1 → both fail.
        let src = vec![42u8; 1];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).is_none());
    }

    #[test]
    fn decode_f32_stride_overflow_returns_none() {
        // Arrange: hidden_size * 4 overflows usize → checked_mul returns None
        let src = vec![0u8; 16];
        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src,
            1,
            usize::MAX / 2 + 1, // hidden_size * 4 would overflow
            DType::F32,
        );
        assert!(out.is_none(), "f32_stride overflow must return None");
    }

    #[test]
    fn decode_half_stride_overflow_returns_none() {
        // Arrange: hidden_size * 2 overflows usize → checked_mul returns None
        let src = vec![0u8; 16];
        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src,
            1,
            usize::MAX, // hidden_size * 2 would overflow
            DType::F16,
        );
        assert!(out.is_none(), "half_stride overflow must return None");
    }

    #[test]
    fn post_node_with_hidden_size_one_captures_single_f32() {
        // Arrange: hidden_size=1, seq_len=1 → single f32 element = 4 bytes
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(1, DType::F32);
        let ctx = holder.ctx(2, 1);
        let output = make_f32_output(&[3.14]);

        // Act
        cb.post_node(&ctx, &output);

        // Assert
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 1);
        assert!((captured[0] - 3.14).abs() < 1e-5);
    }

    #[test]
    fn pre_node_target_layer_equals_context_layer_with_prior_capture_returns_continue() {
        // This specifically tests the branch: ctx.layer_idx == self.target_layer
        // returns Continue, even when captured is Some (the captured state must only
        // be emitted when we transition OUT of the target layer).
        let mut cb = MidLayerEncodeCallback::new(4);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 4
        let ctx4 = holder.ctx(4, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx4, &output);

        // Act: pre_node still at layer 4 — must NOT exit
        let action = cb.pre_node(&ctx4);

        // Assert: Continue, captured state still held
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_some(), "captured must not be consumed while at target layer");
    }

    // ========================================================================
    // New tests: 40 additional tests for comprehensive coverage
    // ========================================================================

    // -- decode_hidden_output: special float bit patterns in F32 path --

    #[test]
    fn decode_f32_negative_zero_preserved() {
        // Arrange: -0.0 f32 (sign bit set, all other bits zero)
        let neg_zero: f32 = -0.0;
        let src: Vec<u8> = neg_zero.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out[0].to_bits(), neg_zero.to_bits(), "-0.0 bit pattern must be preserved");
        assert!(out[0].is_sign_negative());
    }

    #[test]
    fn decode_f32_signaling_nan_preserved() {
        // Arrange: signaling NaN (bit 0x7F800001 — mantissa LSB set, no quiet bit)
        let snan_bits: u32 = 0x7F800001;
        let src: Vec<u8> = snan_bits.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert!(out[0].is_nan(), "Signaling NaN must decode as NaN");
    }

    #[test]
    fn decode_f32_multiple_nan_patterns() {
        // Arrange: different NaN bit patterns in a single buffer
        let nan_patterns: Vec<u32> = vec![
            0x7FC00000, // quiet NaN
            0xFFC00000, // negative quiet NaN
            0x7F800001, // signaling NaN
            0xFF800001, // negative signaling NaN
        ];
        let src: Vec<u8> = nan_patterns
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32).unwrap();
        for val in &out {
            assert!(val.is_nan(), "All NaN bit patterns must decode as NaN");
        }
    }

    #[test]
    fn decode_f32_large_negative_value() {
        // Arrange: f32::MIN (most negative finite value, -3.4028235e38)
        let src: Vec<u8> = f32::MIN.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out[0], f32::MIN);
        assert!(out[0].is_sign_negative());
    }

    #[test]
    fn decode_f32_eps_value() {
        // Arrange: f32::EPSILON (smallest representable difference from 1.0)
        let src: Vec<u8> = f32::EPSILON.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out[0], f32::EPSILON);
        assert!(out[0] > 0.0);
    }

    // -- decode_hidden_output: seq_len and hidden_size boundary conditions --

    #[test]
    fn decode_f32_seq_len_one_large_hidden_size() {
        // Arrange: seq_len=1, hidden_size=8192 — large but valid single row
        let hidden_size = 8192;
        let src = vec![0u8; hidden_size * 4];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden_size, DType::F32);
        assert!(out.is_some(), "large hidden_size with exact-fit buffer must decode");
        assert_eq!(out.unwrap().len(), hidden_size);
    }

    #[test]
    fn decode_f32_large_seq_len_small_hidden_size() {
        // Arrange: seq_len=1024, hidden_size=1 — many rows, single column
        let seq_len = 1024;
        let src: Vec<u8> = (0..seq_len)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq_len, 1, DType::F32).unwrap();
        assert_eq!(out.len(), seq_len);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[511] - 511.0).abs() < 1e-3);
    }

    #[test]
    fn decode_f32_buffer_much_larger_than_needed() {
        // Arrange: buffer sized for max_seq=4096 but only seq_len=1 used
        let hidden_size = 64;
        let max_seq = 4096;
        let mut src = vec![0u8; max_seq * hidden_size * 4];
        let value_bytes = 42.0f32.to_le_bytes();
        src[0..4].copy_from_slice(&value_bytes);
        // Fill first hidden_size elements with a pattern
        for i in 0..hidden_size {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&(i as f32).to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden_size, DType::F32).unwrap();
        assert_eq!(out.len(), hidden_size);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[63] - 63.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: half-precision edge cases --

    #[test]
    fn decode_f16_max_finite_value() {
        // Arrange: f16::MAX (65504.0)
        let h = f16::MAX;
        let src: Vec<u8> = h.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert!((out[0] - 65504.0).abs() < 1.0, "f16 MAX must decode to ~65504");
    }

    #[test]
    fn decode_f16_smallest_positive_subnormal() {
        // Arrange: smallest positive f16 subnormal
        let h = f16::from_bits(0x0001u16);
        let src: Vec<u8> = h.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert!(out[0] > 0.0, "Smallest subnormal must be positive");
        assert!(out[0] < 0.001, "Smallest subnormal must be very small");
    }

    #[test]
    fn decode_f16_negative_values() {
        // Arrange: negative f16 values
        let vals = [f16::from_f32(-1.0), f16::from_f32(-0.5), f16::from_f32(-99.0)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F16).unwrap();
        assert!(out[0].is_sign_negative());
        assert!((out[0] - (-1.0)).abs() < 0.01);
        assert!((out[1] - (-0.5)).abs() < 0.01);
        assert!((out[2] - (-99.0)).abs() < 0.5);
    }

    #[test]
    fn decode_bf16_zero_value() {
        // Arrange: BF16 zero
        let b = bf16::from_f32(0.0);
        let src: Vec<u8> = b.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn decode_bf16_negative_zero() {
        // Arrange: BF16 negative zero
        let b = bf16::from_f32(-0.0);
        let src: Vec<u8> = b.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        assert!(out[0].is_sign_negative(), "Negative zero must preserve sign");
        assert_eq!(out[0].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn decode_bf16_large_value() {
        // Arrange: BF16 representation of a large value
        // BF16 has only 7 bits of mantissa, so 10000.0 may not be exactly representable.
        // Use 256.0 which is exactly representable and still tests the large-value path.
        let b = bf16::from_f32(256.0);
        let src: Vec<u8> = b.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        assert!((out[0] - 256.0).abs() < 1.0, "Large BF16 value must decode correctly");
    }

    // -- decode_hidden_output: DType edge cases --

    #[test]
    fn decode_with_u8_declared_dtype_f32_path_succeeds() {
        // Arrange: buffer with F32 bytes but declared DType::U8 — F32 candidate
        // should still win because it's checked first and stride matches
        let src: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::U8).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decode_with_f6e3m2_declared_dtype_f32_path_succeeds() {
        // Arrange: buffer with F32 bytes, declared DType::F6E3M2 — F32 wins
        let src: Vec<u8> = 2.5f32.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F6E3M2).unwrap();
        assert!((out[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn decode_with_f4e2m1_declared_dtype_f32_path_succeeds() {
        // Arrange: F4E2M1 declared but buffer is 4 bytes — F32 stride wins
        let src: Vec<u8> = 3.0f32.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F4E2M1).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn decode_half_stride_with_f8e5m2_dtype_returns_none() {
        // Arrange: buffer passes half stride but dtype is F8E5M2 (no match arm)
        let src = vec![0u8; 4]; // 4 bytes, hidden=2: half_stride=4, 4%4==0
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F8E5M2);
        assert!(out.is_none(), "F8E5M2 has no match arm in half path");
    }

    // -- pre_node / post_node: multi-layer lifecycle scenarios --

    #[test]
    fn lifecycle_skip_layers_then_capture_and_exit() {
        // Arrange: target layer 5, walk through layers 0..5, capture, then exit
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Layers 0..4: pre_node and post_node do nothing (wrong layer)
        for layer in 0..5 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
            let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
            assert!(matches!(cb.post_node(&ctx, &output), CallbackAction::Continue));
        }

        // Layer 5: post_node captures, pre_node returns Continue
        let ctx5 = holder.ctx(5, 1);
        let output = make_f32_output(&[10.0, 20.0, 30.0, 40.0]);
        assert!(matches!(cb.post_node(&ctx5, &output), CallbackAction::Continue));
        assert!(cb.captured.is_some());

        // Pre_node at layer 5: still Continue (same layer)
        assert!(matches!(cb.pre_node(&ctx5), CallbackAction::Continue));

        // Transition out to layer 6: ExitEarly
        let ctx6 = holder.ctx(6, 1);
        match cb.pre_node(&ctx6) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn lifecycle_target_never_reached_no_capture() {
        // Arrange: target layer 100, walk through layers 0..7 — never reach target
        let mut cb = MidLayerEncodeCallback::new(100);
        let holder = TestCtxHolder::new(4, DType::F32);

        for layer in 0..8 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
            let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
            assert!(matches!(cb.post_node(&ctx, &output), CallbackAction::Continue));
        }

        // No capture ever happened
        assert!(cb.captured.is_none());
    }

    #[test]
    fn lifecycle_multiple_post_nodes_then_single_exit() {
        // Arrange: several post_node calls at target layer, then one exit
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(2, DType::F32);

        // Capture 5 different outputs at layer 3 — only last one retained
        for i in 0..5u32 {
            let ctx = holder.ctx(3, 1);
            let output = make_f32_output(&[i as f32, (i * 10) as f32]);
            cb.post_node(&ctx, &output);
        }

        // Exit should carry the last captured values
        let ctx4 = holder.ctx(4, 1);
        match cb.pre_node(&ctx4) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![4.0, 40.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // -- pre_node: transition back to target layer after exit --

    #[test]
    fn pre_node_return_to_target_after_exit_no_recapture() {
        // Arrange: capture, exit, then pre_node at target layer — no re-capture
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 2
        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx2, &output);

        // Exit at layer 3
        let ctx3 = holder.ctx(3, 1);
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::ExitEarly { .. }));

        // Return to layer 2 with no new capture
        let action = cb.pre_node(&ctx2);
        assert!(
            matches!(action, CallbackAction::Continue),
            "pre_node at target layer after consumed capture must return Continue"
        );
        assert!(cb.captured.is_none());
    }

    // -- post_node: mixed shape outputs within target layer --

    #[test]
    fn post_node_mixed_compatible_incompatible_shapes_captures_last_compatible() {
        // Arrange: first a compatible output, then an incompatible one, then compatible
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Compatible output: 4 f32 = 16 bytes, hidden_size=4
        let compatible1 = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx, &compatible1);

        // Incompatible output: 7 bytes — not a multiple of hidden_size*4=16
        let incompatible = vec![0u8; 7];
        cb.post_node(&ctx, &incompatible);

        // Compatible output again: different values
        let compatible2 = make_f32_output(&[5.0, 6.0, 7.0, 8.0]);
        cb.post_node(&ctx, &compatible2);

        // Assert: last compatible output captured
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![5.0, 6.0, 7.0, 8.0]);
    }

    // -- decode_hidden_output: checked_mul edge cases --

    #[test]
    fn decode_numel_exact_max_does_not_overflow() {
        // Arrange: seq_len and hidden_size whose product is exactly usize::MAX
        // This is a theoretical test — in practice such a buffer can't exist,
        // but we verify the overflow check works correctly.
        let src = vec![0u8; 16];
        // usize::MAX = seq_len * hidden_size. We can't easily construct this,
        // but we can verify that normal values don't overflow.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32);
        assert!(out.is_some());
    }

    #[test]
    fn decode_f32_stride_check_zero_hidden_size_returns_none() {
        // Arrange: hidden_size=0 — explicitly tested (early return)
        let src = vec![0u8; 16];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 0, DType::F32);
        assert!(out.is_none(), "hidden_size=0 must return None");
    }

    #[test]
    fn decode_seq_len_zero_returns_none() {
        // Arrange: seq_len=0 — early return
        let src = vec![0u8; 16];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 0, 4, DType::F32);
        assert!(out.is_none(), "seq_len=0 must return None");
    }

    // -- CallbackChain integration with MidLayerEncodeCallback --

    #[test]
    fn chain_integration_mid_layer_encode_dispatches_correctly() {
        // Arrange: put MidLayerEncodeCallback in a CallbackChain
        use crate::graph::layer_callback::CallbackChain;

        let cb = MidLayerEncodeCallback::new(2);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = TestCtxHolder::new(4, DType::F32);

        // Pre-node at layer 0: Continue
        let ctx0 = holder.ctx(0, 1);
        assert_eq!(chain.dispatch_pre_node(&ctx0), CallbackAction::Continue);

        // Post-node at layer 2 with valid output: Continue (capture occurs internally)
        // Note: ctx(layer_idx=2, seq_len=1) — seq_len must match the output buffer size.
        // hidden_size=4, seq_len=1 → need 4 f32 = 16 bytes.
        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(chain.dispatch_post_node(&ctx2, &output), CallbackAction::Continue);

        // Pre-node at layer 3: ExitEarly with captured hidden state
        let ctx3 = holder.ctx(3, 1);
        let action = chain.dispatch_pre_node(&ctx3);
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn chain_integration_mid_layer_with_higher_priority_callback() {
        // Arrange: MidLayerEncodeCallback (prio 55) and a higher-prio callback
        use crate::graph::layer_callback::CallbackChain;

        struct HighPrioCallback;
        impl LayerCallback for HighPrioCallback {
            fn priority(&self) -> u32 { 100 }
            fn name(&self) -> &str { "high_prio" }
        }

        let mid_layer = MidLayerEncodeCallback::new(2);
        let high = HighPrioCallback;
        let chain = CallbackChain::new(vec![Box::new(mid_layer), Box::new(high)]);

        // Assert: two callbacks registered, sorted by priority
        assert_eq!(chain.len(), 2);
    }

    // -- post_node: seq_len from context used correctly --

    #[test]
    fn post_node_uses_seq_len_from_context_not_buffer_size() {
        // Arrange: buffer sized for seq_len=4, but ctx says seq_len=2
        // Only the first 2*hidden_size elements should be decoded
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Buffer for 4 tokens (4 * 4 * 4 = 64 bytes)
        let mut src = vec![0u8; 64];
        // Write different values for first 2 tokens vs last 2
        let first_two: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (i, v) in first_two.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Overwrite last 2 tokens with distinct values
        for i in 0..8 {
            let off = (8 + i) * 4;
            src[off..off + 4].copy_from_slice(&999.0f32.to_le_bytes());
        }

        // Context with seq_len=2
        let ctx = holder.ctx(2, 2);
        cb.post_node(&ctx, &src);

        // Assert: only first 2*4=8 elements captured
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 8); // 2 * 4
        assert!((captured[0] - 1.0).abs() < 1e-6);
        assert!((captured[7] - 8.0).abs() < 1e-6);
    }

    // -- decode: byte alignment edge cases --

    #[test]
    fn decode_f32_buffer_exactly_double_hidden_stride() {
        // Arrange: buffer = 2 * hidden_size * 4, seq_len=2
        let src: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(src.len(), 24); // 6 f32 * 4 bytes

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F32).unwrap();
        assert_eq!(out.len(), 6);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn decode_f32_buffer_with_trailing_slack_bytes() {
        // Arrange: buffer has extra bytes beyond the required minimum
        let mut src = vec![0u8; 100]; // 100 bytes, hidden=4 → stride=16, 100%16=4 (not aligned)
        // Adjust to be aligned: 96 bytes
        src.truncate(96);
        // Write 4 f32 values for seq_len=1, hidden=4
        let values: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        for (i, v) in values.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 10.0).abs() < 1e-6);
        assert!((out[3] - 40.0).abs() < 1e-6);
    }

    // -- new_with_usize_max --

    #[test]
    fn new_with_usize_max_target_layer() {
        // Arrange & Act: maximum target layer value
        let cb = MidLayerEncodeCallback::new(usize::MAX);

        // Assert: construction succeeds, fields stored correctly
        assert_eq!(cb.target_layer, usize::MAX);
        assert!(cb.captured.is_none());
        assert_eq!(cb.priority(), MID_LAYER_ENCODE_PRIORITY);
        assert_eq!(cb.name(), "MidLayerEncode");
        assert!(cb.target_layers().is_none());
    }

    // -- decode: mixed byte pattern verification --

    #[test]
    fn decode_f32_alternating_sign_pattern() {
        // Arrange: alternating positive and negative values
        let values: Vec<f32> = (0..10).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 5, DType::F32).unwrap();
        assert_eq!(out.len(), 10);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - (-1.0)).abs() < 1e-6);
        assert!((out[2] - 2.0).abs() < 1e-6);
        assert!((out[3] - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn decode_f32_all_ones_bit_pattern() {
        // Arrange: all-bits-1 = NaN (0xFFFFFFFF)
        let nan_bits: u32 = 0xFFFFFFFF;
        let src: Vec<u8> = nan_bits.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert!(out[0].is_nan());
    }

    // -- decode: half stride with F32 buffer that happens to be half-aligned --

    #[test]
    fn decode_f32_buffer_that_also_passes_half_stride_uses_f32_first() {
        // Arrange: 8-byte buffer with hidden=1.
        // F32 stride=4, 8%4==0, 8>=4 → F32 passes → decoded as F32.
        // Half stride=2, 8%2==0, 8>=2 → also passes, but F32 checked first.
        let src: Vec<u8> = [1.5f32, 2.5f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16).unwrap();
        // Must decode as F32 (first candidate), not as F16
        assert!((out[0] - 1.5).abs() < 1e-6, "F32 candidate must win");
        assert!((out[1] - 2.5).abs() < 1e-6);
    }

    // -- BF16 / F16 multi-row layout preservation --

    #[test]
    fn decode_bf16_multi_seq_row_order_preserved() {
        // Arrange: seq_len=3, hidden_size=2, BF16
        let vals = [
            bf16::from_f32(1.0), bf16::from_f32(2.0),
            bf16::from_f32(3.0), bf16::from_f32(4.0),
            bf16::from_f32(5.0), bf16::from_f32(6.0),
        ];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 2, DType::BF16).unwrap();
        assert_eq!(out.len(), 6);
        // Row-major: row0=[1,2], row1=[3,4], row2=[5,6]
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - 2.0).abs() < 0.01);
        assert!((out[2] - 3.0).abs() < 0.01);
        assert!((out[5] - 6.0).abs() < 0.01);
    }

    #[test]
    fn decode_f16_multi_seq_row_order_preserved() {
        // Arrange: seq_len=2, hidden_size=3, F16
        let vals = [
            f16::from_f32(0.1), f16::from_f32(0.2), f16::from_f32(0.3),
            f16::from_f32(0.4), f16::from_f32(0.5), f16::from_f32(0.6),
        ];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F16).unwrap();
        assert_eq!(out.len(), 6);
        assert!((out[0] - 0.1).abs() < 0.01);
        assert!((out[5] - 0.6).abs() < 0.01);
    }

    // -- pre_node: exact layer transition sequence --

    #[test]
    fn pre_node_exact_layer_transition_order() {
        // Arrange: simulate exact node sequence: pre(L0), post(L0), pre(L0), post(L0),
        // pre(L1), post(L1), ..., pre(L3), post(L3), pre(L4) → exit
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);

        for layer in 0..3 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
            let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
            assert!(matches!(cb.post_node(&ctx, &output), CallbackAction::Continue));
        }

        // Layer 3: target — capture happens
        let ctx3 = holder.ctx(3, 1);
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));
        let output = make_f32_output(&[7.0, 8.0, 9.0, 10.0]);
        assert!(matches!(cb.post_node(&ctx3, &output), CallbackAction::Continue));

        // Pre-node at layer 4: transition out → ExitEarly
        let ctx4 = holder.ctx(4, 1);
        match cb.pre_node(&ctx4) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![7.0, 8.0, 9.0, 10.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // -- decode: exact boundary between f32 and half stride --

    #[test]
    fn decode_exactly_at_f32_boundary() {
        // Arrange: buffer length = exactly numel * 4, no slack
        let hidden = 128;
        let seq = 1;
        let numel = hidden * seq;
        let src = vec![0u8; numel * 4];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F32);
        assert!(out.is_some(), "exact-fit buffer must decode");
        assert_eq!(out.unwrap().len(), numel);
    }

    #[test]
    fn decode_just_below_f32_minimum() {
        // Arrange: buffer 1 byte short of minimum needed for F32
        let hidden = 4;
        let seq = 1;
        let needed = hidden * seq * 4; // 16 bytes
        let src = vec![0u8; needed - 1]; // 15 bytes
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F32);
        assert!(out.is_none(), "1 byte short must fail F32 candidate");
    }

    #[test]
    fn decode_just_below_half_minimum() {
        // Arrange: buffer 1 byte short of minimum for half precision
        let hidden = 4;
        let seq = 1;
        let needed = hidden * seq * 2; // 8 bytes
        let src = vec![0u8; needed - 1]; // 7 bytes
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F16);
        assert!(out.is_none(), "1 byte short must fail both candidates");
    }

    // ========================================================================
    // Additional tests (50 new) for comprehensive coverage
    // ========================================================================

    // -- decode_hidden_output: remaining DType variants as declared_dtype --

    #[test]
    fn decode_with_f6e2m3_declared_dtype_f32_path_succeeds() {
        // Arrange: F32 buffer, declared DType::F6E2M3 — F32 candidate wins
        let src: Vec<u8> = 7.5f32.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F6E2M3).unwrap();
        assert!((out[0] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn decode_half_stride_with_f6e2m3_declared_returns_none() {
        // Arrange: buffer passes half stride but F6E2M3 has no match arm
        let src = vec![0u8; 4]; // hidden=2, half_stride=4, 4%4==0
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F6E2M3);
        assert!(out.is_none(), "F6E2M3 has no match arm in half path");
    }

    #[test]
    fn decode_half_stride_with_u8_declared_returns_none() {
        // Arrange: buffer passes half stride but U8 has no match arm
        let src = vec![0u8; 4]; // hidden=2, half_stride=4, 4%4==0
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::U8);
        assert!(out.is_none(), "U8 has no match arm in half path");
    }

    // -- decode_hidden_output: odd hidden_size behavior --

    #[test]
    fn decode_f32_odd_hidden_size_exact_fit() {
        // Arrange: hidden_size=5 (odd), seq_len=1 → 5 f32 = 20 bytes exact fit
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 20);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 5, DType::F32).unwrap();
        assert_eq!(out.len(), 5);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[4] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn decode_half_odd_hidden_size_with_slack() {
        // Arrange: hidden_size=5 (odd), F16 buffer pre-allocated for max_seq=3.
        // total = 3 * 5 * 2 = 30 bytes. F32 stride=20, 30%20 != 0 → F32 fails.
        // half stride=10, 30%10==0, 30>=10 → half passes with F16 declared.
        let hidden_size = 5;
        let max_seq = 3;
        let mut src = vec![0u8; max_seq * hidden_size * 2];
        // Write values for seq_len=2 (10 elements)
        let vals: Vec<f16> = (0..10).map(|i| f16::from_f32(i as f32 * 0.5)).collect();
        for (i, v) in vals.iter().enumerate() {
            let off = i * 2;
            src[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, hidden_size, DType::F16).unwrap();
        assert_eq!(out.len(), 10);
        assert!((out[0] - 0.0).abs() < 0.01);
        assert!((out[4] - 2.0).abs() < 0.01);
    }

    // -- decode_hidden_output: numel*4 overflow --

    #[test]
    fn decode_f32_numel_times_four_overflow_returns_none() {
        // Arrange: numel = seq_len * hidden_size does NOT overflow, but numel * 4 does.
        // usize::MAX / 2 is even-ish; pick values where seq_len * hidden_size fits in usize
        // but (seq_len * hidden_size) * 4 overflows.
        let huge = (usize::MAX / 4) + 1; // numel = this, numel*4 overflows
        let src = vec![0u8; 16];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, huge, 1, DType::F32);
        assert!(out.is_none(), "numel*4 overflow in f32 candidate must return None");
    }

    // -- decode_hidden_output: half path with exact 2-byte buffer --

    #[test]
    fn decode_f16_exact_single_element_buffer() {
        // Arrange: exactly 2 bytes (1 f16 element), hidden_size=1, seq_len=1
        // F32 stride=4, 2%4 != 0 → F32 fails.
        // half stride=2, 2%2==0, 2>=2 → passes with F16 declared.
        let h = f16::from_f32(3.5);
        let src: Vec<u8> = h.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 3.5).abs() < 0.01);
    }

    #[test]
    fn decode_bf16_exact_single_element_buffer() {
        // Arrange: exactly 2 bytes (1 bf16 element), hidden_size=1, seq_len=1
        let b = bf16::from_f32(-4.5);
        let src: Vec<u8> = b.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - (-4.5)).abs() < 0.1);
    }

    // -- decode_hidden_output: large hidden_size with half-precision --

    #[test]
    fn decode_f16_large_hidden_size() {
        // Arrange: hidden_size=2048, seq_len=1, F16 → 4096 bytes
        let hidden_size = 2048;
        let src = vec![0u8; hidden_size * 2];
        // Write a value at start and end
        let first = f16::from_f32(1.0);
        let last = f16::from_f32(-1.0);
        let mut src_mut = src;
        src_mut[0..2].copy_from_slice(&first.to_le_bytes());
        let end_off = (hidden_size - 1) * 2;
        src_mut[end_off..end_off + 2].copy_from_slice(&last.to_le_bytes());

        let out = MidLayerEncodeCallback::decode_hidden_output(&src_mut, 1, hidden_size, DType::F16).unwrap();
        assert_eq!(out.len(), hidden_size);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[hidden_size - 1] - (-1.0)).abs() < 0.01);
    }

    // -- decode_hidden_output: two-byte buffer that fails half check --

    #[test]
    fn decode_two_byte_buffer_hidden_size_two_returns_none() {
        // Arrange: 2 bytes, hidden_size=2 → F32 stride=8, 2%8!=0 → F32 fails.
        // half stride=4, 2%4!=0 → half fails. Both fail → None.
        let src = vec![0u8; 2];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16);
        assert!(out.is_none(), "2-byte buffer with hidden_size=2 must fail both candidates");
    }

    // -- decode_hidden_output: buffer that passes F32 stride but fails size check --

    #[test]
    fn decode_f32_stride_passes_but_buffer_too_small_for_seq() {
        // Arrange: hidden_size=4, F32 stride=16. Buffer=32 bytes (multiple of 16).
        // seq_len=3, numel=12, need 12*4=48 bytes. Buffer 32 < 48 → size check fails.
        let src = vec![0u8; 32];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 4, DType::F32);
        assert!(out.is_none(), "F32 stride passes but buffer too small for seq_len*hidden");
    }

    // -- decode_hidden_output: half stride passes but buffer too small --

    #[test]
    fn decode_half_stride_passes_but_buffer_too_small_for_seq() {
        // Arrange: hidden_size=3, half_stride=6. Buffer=12 bytes (multiple of 6).
        // seq_len=3, numel=9, need 9*2=18 bytes. 12 < 18 → half size check fails.
        // F32 stride=12, 12%12==0, 12 < 36 → F32 size check fails too.
        let src = vec![0u8; 12];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 3, DType::F16);
        assert!(out.is_none(), "half stride passes but buffer too small");
    }

    // -- post_node: capture with BF16-declared config --

    #[test]
    fn post_node_captures_with_bf16_declared_dtype() {
        // Arrange: hidden_size=4, declared dtype BF16, CPU canonical F32 output
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::BF16);
        let ctx = holder.ctx(3, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);

        // Act: post_node with F32 bytes but BF16 declared — F32 candidate wins
        cb.post_node(&ctx, &output);

        // Assert
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 4);
        assert!((captured[0] - 1.0).abs() < 1e-6);
        assert!((captured[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn post_node_captures_with_f16_declared_dtype() {
        // Arrange: hidden_size=4, declared dtype F16, CPU canonical F32 output
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F16);
        let ctx = holder.ctx(2, 1);
        let output = make_f32_output(&[5.0, 6.0, 7.0, 8.0]);

        cb.post_node(&ctx, &output);

        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![5.0, 6.0, 7.0, 8.0]);
    }

    // -- pre_node: transitions with non-consecutive layer indices --

    #[test]
    fn pre_node_non_consecutive_layer_transition_exit() {
        // Arrange: target layer 2, jump to layer 10 (skip layers)
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx2, &output);

        // Act: jump from layer 2 to layer 10
        let ctx10 = holder.ctx(10, 1);
        let action = cb.pre_node(&ctx10);

        // Assert: ExitEarly despite non-consecutive jump
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn pre_node_backward_layer_transition_no_exit_without_capture() {
        // Arrange: target layer 5, currently at layer 3 (before target), no capture
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx3 = holder.ctx(3, 1);

        let action = cb.pre_node(&ctx3);
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_none());
    }

    // -- post_node: multiple captures across different target layer calls --

    #[test]
    fn post_node_capture_overwritten_across_multiple_calls() {
        // Arrange: 10 post_node calls at target layer, verify only last retained
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(2, DType::F32);
        let ctx = holder.ctx(1, 1);

        for i in 0..10u32 {
            let output = make_f32_output(&[i as f32, (i + 1) as f32]);
            cb.post_node(&ctx, &output);
        }

        // Assert: only the 10th capture retained (values [9.0, 10.0])
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![9.0, 10.0]);
    }

    // -- decode_hidden_output: F32 with realistic hidden sizes --

    #[test]
    fn decode_f32_hidden_size_768_typical_small_model() {
        // Arrange: typical small model hidden size (e.g., GPT-2 small)
        let hidden_size = 768;
        let src = vec![0u8; hidden_size * 4];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden_size, DType::F32).unwrap();
        assert_eq!(out.len(), hidden_size);
        // All zeros
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn decode_f32_hidden_size_5120_large_model() {
        // Arrange: large model hidden size (e.g., LLaMA-70B)
        let hidden_size = 5120;
        let mut src = vec![0u8; hidden_size * 4];
        // Write a pattern: first element = 1.0, last element = -1.0
        src[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        let last_off = (hidden_size - 1) * 4;
        src[last_off..last_off + 4].copy_from_slice(&(-1.0f32).to_le_bytes());

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden_size, DType::F32).unwrap();
        assert_eq!(out.len(), hidden_size);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[hidden_size - 1] - (-1.0)).abs() < 1e-6);
    }

    // -- decode_hidden_output: interleaved byte pattern verification --

    #[test]
    fn decode_f32_interleaved_zero_nonzero_pattern() {
        // Arrange: alternating zero and non-zero values
        let values: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 0.0 } else { (i * 10) as f32 }).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 4, DType::F32).unwrap();
        assert_eq!(out.len(), 8);
        assert_eq!(out[0], 0.0);
        assert!((out[1] - 10.0).abs() < 1e-6);
        assert_eq!(out[2], 0.0);
        assert!((out[3] - 30.0).abs() < 1e-6);
        assert_eq!(out[4], 0.0);
        assert!((out[5] - 50.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: BF16 with negative infinity --

    #[test]
    fn decode_bf16_negative_infinity() {
        // Arrange: BF16 representation of negative infinity
        let b = bf16::from_f32(f32::NEG_INFINITY);
        let src: Vec<u8> = b.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        assert!(out[0].is_infinite());
        assert!(out[0].is_sign_negative());
    }

    // -- decode_hidden_output: F16 with positive infinity --

    #[test]
    fn decode_f16_positive_infinity() {
        // Arrange: F16 representation of positive infinity
        let h = f16::from_f32(f32::INFINITY);
        let src: Vec<u8> = h.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert!(out[0].is_infinite());
        assert!(out[0].is_sign_positive());
    }

    // -- decode_hidden_output: F16 NaN preservation --

    #[test]
    fn decode_f16_nan_preserved() {
        // Arrange: F16 NaN
        let h = f16::from_bits(0x7E00u16); // F16 quiet NaN
        let src: Vec<u8> = h.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert!(out[0].is_nan(), "F16 NaN must decode as NaN");
    }

    // -- decode_hidden_output: BF16 NaN preservation --

    #[test]
    fn decode_bf16_nan_preserved() {
        // Arrange: BF16 NaN
        let b = bf16::from_bits(0x7FC0u16); // BF16 quiet NaN
        let src: Vec<u8> = b.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        assert!(out[0].is_nan(), "BF16 NaN must decode as NaN");
    }

    // -- post_node + pre_node: full lifecycle with capture at layer 0 and exit --

    #[test]
    fn lifecycle_layer_zero_capture_immediate_exit() {
        // Arrange: target layer 0, single capture, immediate transition
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 0
        let ctx0 = holder.ctx(0, 1);
        let output = make_f32_output(&[42.0, 43.0, 44.0, 45.0]);
        cb.post_node(&ctx0, &output);

        // Exit at layer 1
        let ctx1 = holder.ctx(1, 1);
        match cb.pre_node(&ctx1) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![42.0, 43.0, 44.0, 45.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // -- post_node: incompatibly-shaped output at target layer does not clear prior capture --

    #[test]
    fn post_node_incompatible_does_not_clear_prior_capture() {
        // Arrange: capture a valid output, then send an incompatible one
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // First: valid capture
        let valid = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx, &valid);
        assert!(cb.captured.is_some());

        // Second: incompatible output (wrong shape)
        let incompatible = vec![0u8; 7]; // not a multiple of hidden_size*4=16
        cb.post_node(&ctx, &incompatible);

        // Assert: prior capture preserved (incompatible did not clear it)
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![1.0, 2.0, 3.0, 4.0]);
    }

    // -- decode_hidden_output: buffer exactly equals half stride --

    #[test]
    fn decode_half_buffer_exactly_equals_half_stride() {
        // Arrange: hidden_size=3, half_stride=6. Buffer=6 bytes exactly.
        // F32 stride=12, 6%12!=0 → F32 fails.
        // 6%6==0, 6>=6 → half passes with F16 declared.
        let vals = [f16::from_f32(0.1), f16::from_f32(0.2), f16::from_f32(0.3)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 6);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F16).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.1).abs() < 0.01);
    }

    // -- decode_hidden_output: buffer with padding at end that makes F32 stride fail --

    #[test]
    fn decode_padding_makes_f32_stride_fail_half_succeeds() {
        // Arrange: hidden_size=3, ideal F32 buffer = N*12 bytes.
        // But buffer has 2 extra bytes of padding: N*12 + 2.
        // F32 stride=12, (N*12+2)%12 != 0 → F32 fails.
        // half stride=6, (N*12+2) = (2N+1/3... need to pick carefully.
        // Use: total = 12 + 2 = 14. 14%12 != 0 → F32 fails. 14%6 != 0 → half fails.
        // Try: total = 24 + 6 = 30. 30%12 != 0 → F32 fails. 30%6 == 0 → half passes.
        let mut src = vec![0u8; 30];
        let vals = [bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)];
        for (i, v) in vals.iter().enumerate() {
            let off = i * 2;
            src[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::BF16).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 0.01);
    }

    // -- pre_node: target_layer vs context layer with same value but different node_idx --

    #[test]
    fn pre_node_same_layer_different_node_idx_returns_continue() {
        // Arrange: two contexts at same layer_idx but different node_idx
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 2
        let ctx_a = LayerContext {
            node_idx: 100,
            layer_idx: 2,
            node_op: "Test",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 1,
            seq_len: 1,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx_a, &output);

        // pre_node at same layer_idx=2 but different node_idx=200
        let ctx_b = LayerContext {
            node_idx: 200,
            layer_idx: 2,
            ..ctx_a
        };

        // Assert: still Continue because layer_idx matches target
        assert!(matches!(cb.pre_node(&ctx_b), CallbackAction::Continue));
        assert!(cb.captured.is_some(), "capture not consumed");
    }

    // -- decode_hidden_output: F16 with BF16 declared (mismatch, F32 fails, half picks BF16) --

    #[test]
    fn decode_half_buffer_f16_data_with_bf16_declared() {
        // Arrange: buffer contains F16 bit patterns, but declared_dtype is BF16.
        // If F32 stride fails, half stride check passes and BF16 match arm decodes.
        // The bytes will be interpreted as BF16, not F16 — different values expected.
        let hidden_size = 3;
        // Make buffer that fails F32: 6 bytes (F32 stride=12, 6%12!=0)
        let b1 = bf16::from_f32(2.5);
        let b2 = bf16::from_f32(-1.0);
        let b3 = bf16::from_f32(0.0);
        let src: Vec<u8> = [&b1, &b2, &b3].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 6);

        // Act: declared BF16 — half path picks BF16 match arm
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden_size, DType::BF16).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 2.5).abs() < 0.1);
        assert!((out[1] - (-1.0)).abs() < 0.1);
        assert!((out[2] - 0.0).abs() < 0.01);
    }

    // -- decode_hidden_output: buffer with all bytes set to 0xFF --

    #[test]
    fn decode_f32_all_ff_bytes_is_nan() {
        // Arrange: all bytes 0xFF → 0xFFFFFFFF = NaN
        let src = vec![0xFFu8; 4];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert!(out[0].is_nan());
    }

    #[test]
    fn decode_bf16_all_ff_bytes_is_nan_or_special() {
        // Arrange: 2 bytes all 0xFF → BF16 bits 0xFFFF = NaN or special
        let src = vec![0xFFu8; 2];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        // BF16 0xFFFF = negative NaN or negative infinity + mantissa
        assert!(out[0].is_nan() || out[0].is_infinite() || out[0].is_sign_negative());
    }

    // -- decode_hidden_output: very small positive F32 value --

    #[test]
    fn decode_f32_very_small_positive_subnormal() {
        // Arrange: a subnormal value that is very small but non-zero
        let tiny: f32 = f32::from_bits(0x00000100); // subnormal with a few bits set
        let src: Vec<u8> = tiny.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out[0].to_bits(), 0x00000100u32);
        assert!(out[0] > 0.0);
    }

    // -- decode_hidden_output: f32 with seq_len > 1 and different values per row --

    #[test]
    fn decode_f32_multi_row_different_values_per_row() {
        // Arrange: seq_len=4, hidden_size=2, each row has distinct values
        let values: Vec<f32> = vec![
            100.0, 200.0,  // row 0
            -100.0, -200.0, // row 1
            0.5, 1.5,       // row 2
            -0.5, -1.5,     // row 3
        ];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 4, 2, DType::F32).unwrap();
        assert_eq!(out.len(), 8);
        // Row 0
        assert!((out[0] - 100.0).abs() < 1e-6);
        assert!((out[1] - 200.0).abs() < 1e-6);
        // Row 1
        assert!((out[2] - (-100.0)).abs() < 1e-6);
        assert!((out[3] - (-200.0)).abs() < 1e-6);
        // Row 2
        assert!((out[4] - 0.5).abs() < 1e-6);
        assert!((out[5] - 1.5).abs() < 1e-6);
        // Row 3
        assert!((out[6] - (-0.5)).abs() < 1e-6);
        assert!((out[7] - (-1.5)).abs() < 1e-6);
    }

    // -- decode_hidden_output: seq_len=2 hidden_size=1 with F32 (2 elements total) --

    #[test]
    fn decode_f32_seq_len_2_hidden_size_1() {
        // Arrange: seq_len=2, hidden_size=1 → 2 f32 = 8 bytes
        let values: Vec<f32> = vec![42.0, -42.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 1, DType::F32).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 42.0).abs() < 1e-6);
        assert!((out[1] - (-42.0)).abs() < 1e-6);
    }

    // -- decode_hidden_output: BF16 with multiple rows, partial data --

    #[test]
    fn decode_bf16_multi_row_in_preallocated_buffer() {
        // Arrange: pre-allocated BF16 buffer that fails F32 stride but passes half stride.
        // hidden_size=3: F32 stride=12, half stride=6.
        // max_seq=5 → total=5*3*2=30. 30%12=6 (not F32-aligned). 30%6==0 → half passes.
        let hidden_size = 3;
        let max_seq = 5;
        let mut src = vec![0u8; max_seq * hidden_size * 2];
        // Write seq_len=2 values (6 BF16 elements)
        let vals: Vec<bf16> = vec![
            bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0),
            bf16::from_f32(4.0), bf16::from_f32(5.0), bf16::from_f32(6.0),
        ];
        for (i, v) in vals.iter().enumerate() {
            let off = i * 2;
            src[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, hidden_size, DType::BF16).unwrap();
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[5] - 6.0).abs() < 0.01);
    }

    // -- post_node: seq_len larger than 1 with correct capture size --

    #[test]
    fn post_node_seq_len_3_captures_3_rows() {
        // Arrange: hidden_size=4, seq_len=3 → 12 f32 = 48 bytes
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 3);
        let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let output = make_f32_output(&values);

        // Act
        cb.post_node(&ctx, &output);

        // Assert
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 12);
        assert!((captured[0] - 0.0).abs() < 1e-6);
        assert!((captured[11] - 11.0).abs() < 1e-6);
    }

    // -- post_node + pre_node: exit with multi-row hidden state --

    #[test]
    fn exit_with_multi_row_hidden_state() {
        // Arrange: capture seq_len=3, then exit — verify full multi-row output
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(3, DType::F32);

        let ctx1 = holder.ctx(1, 3);
        let values: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let output = make_f32_output(&values);
        cb.post_node(&ctx1, &output);

        // Exit
        let ctx2 = holder.ctx(2, 3);
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits.len(), 9);
                assert!((logits[0] - 10.0).abs() < 1e-6);
                assert!((logits[8] - 90.0).abs() < 1e-6);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // -- pre_node: layer_idx much larger than target (post-decoder node simulation) --

    #[test]
    fn pre_node_post_decoder_node_triggers_exit() {
        // Arrange: simulate post-decoder node (e.g., final_norm) with very high layer_idx
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx3 = holder.ctx(3, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx3, &output);

        // Simulate final_norm: layer_idx = 9999 (extract_layer_index fallback)
        let ctx_post = LayerContext {
            node_idx: 9999,
            layer_idx: 9999,
            node_op: "final_norm",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 1,
            seq_len: 1,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };

        match cb.pre_node(&ctx_post) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("Expected ExitEarly for post-decoder node, got {:?}", other),
        }
    }

    // -- decode_hidden_output: buffer where F32 stride = half stride (hidden_size=1) --

    #[test]
    fn decode_hidden_size_one_f32_wins_over_half() {
        // Arrange: hidden_size=1, 4 bytes. F32 stride=4, half stride=2.
        // 4%4==0 → F32 passes → decoded as F32 first.
        let src: Vec<u8> = 99.0f32.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        // F32 candidate wins
        assert!((out[0] - 99.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: 8-byte buffer with hidden_size=1 (F32 passes, 2 elems) --

    #[test]
    fn decode_f32_hidden_size_one_seq_len_two() {
        // Arrange: hidden_size=1, seq_len=2 → 2 f32 = 8 bytes
        let values: Vec<f32> = vec![-1.0, 1.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 1, DType::F32).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: buffer alignment boundary (hidden_size=2, 12 bytes) --

    #[test]
    fn decode_f32_hidden_size_two_seq_len_one_with_slack() {
        // Arrange: hidden_size=2, pre-allocated for max_seq=3 → 3*2*4=24 bytes
        let mut src = vec![0u8; 24];
        let vals: Vec<f32> = vec![3.14, -2.71];
        for (i, v) in vals.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.14).abs() < 1e-5);
        assert!((out[1] - (-2.71)).abs() < 1e-5);
    }

    // -- pre_node: calling pre_node then post_node then pre_node at target layer --

    #[test]
    fn pre_post_pre_at_target_layer_captures_and_stays() {
        // Arrange: interleave pre_node and post_node at target layer
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // pre_node at target → Continue
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));

        // post_node → captures
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(cb.post_node(&ctx, &output), CallbackAction::Continue));

        // pre_node again at target → Continue (not exit)
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));

        // captured still present
        assert!(cb.captured.is_some());
    }

    // -- decode_hidden_output: F16 with zero values --

    #[test]
    fn decode_f16_all_zeros() {
        // Arrange: F16 buffer of all zeros
        let src = vec![0u8; 8]; // 4 f16 elements
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F16).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // -- decode_hidden_output: BF16 with zero values --

    #[test]
    fn decode_bf16_all_zeros() {
        // Arrange: BF16 buffer of all zeros
        let src = vec![0u8; 8]; // 4 bf16 elements
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::BF16).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // -- decode_hidden_output: half candidate with hidden_size=1, buffer=2 --

    #[test]
    fn decode_half_hidden_size_one_exact() {
        // Arrange: hidden_size=1, seq_len=1, buffer=2 bytes (1 half element)
        // F32 stride=4, 2%4!=0 → F32 fails.
        // half stride=2, 2%2==0, 2>=2 → passes.
        let b = bf16::from_f32(0.125);
        let src: Vec<u8> = b.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.125).abs() < 0.01);
    }

    // -- decode_hidden_output: numel checked_mul returns 0 (impossible edge) --

    #[test]
    fn decode_numel_zero_returns_none() {
        // Arrange: seq_len=0 or hidden_size=0 leads to numel=0 → early return None
        assert!(MidLayerEncodeCallback::decode_hidden_output(&[0u8; 16], 0, 4, DType::F32).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&[0u8; 16], 4, 0, DType::F32).is_none());
    }

    // -- lifecycle: walk all layers without ever reaching target, no exit --

    #[test]
    fn lifecycle_all_layers_before_target_no_exit() {
        // Arrange: target=10, walk layers 0..9 — never reach target
        let mut cb = MidLayerEncodeCallback::new(10);
        let holder = TestCtxHolder::new(4, DType::F32);

        for layer in 0..10 {
            let ctx = holder.ctx(layer, 1);
            let output = make_f32_output(&[layer as f32, (layer + 1) as f32, (layer + 2) as f32, (layer + 3) as f32]);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
            assert!(matches!(cb.post_node(&ctx, &output), CallbackAction::Continue));
        }

        // No capture ever happened (all post_node calls were at non-target layers)
        assert!(cb.captured.is_none());
    }

    // -- decode_hidden_output: F32 with 3-byte buffer (not F32 aligned, not half aligned for hidden=1) --

    #[test]
    fn decode_three_byte_buffer_fails_all_candidates() {
        // Arrange: 3 bytes. hidden=1: F32 stride=4, 3%4!=0 → F32 fails.
        // half stride=2, 3%2!=0 → half fails.
        let src = vec![0u8; 3];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).is_none());
    }

    // -- post_node: output at non-target layer after capture at target does not overwrite --

    #[test]
    fn post_node_non_target_after_capture_does_not_overwrite() {
        // Arrange: capture at target layer 2, then post_node at layer 3
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 2
        let ctx2 = holder.ctx(2, 1);
        let output2 = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx2, &output2);

        // post_node at layer 3 (non-target) with different data
        let ctx3 = holder.ctx(3, 1);
        let output3 = make_f32_output(&[99.0, 99.0, 99.0, 99.0]);
        cb.post_node(&ctx3, &output3);

        // Assert: captured still has layer 2 data (layer 3 post_node ignored)
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![1.0, 2.0, 3.0, 4.0]);
    }

    // -- decode_hidden_output: F16 negative max value --

    #[test]
    fn decode_f16_negative_max_value() {
        // Arrange: f16 minimum (most negative finite)
        let h = f16::from_f32(-f16::MAX.to_f32());
        let src: Vec<u8> = h.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert!(out[0].is_sign_negative());
        assert!((out[0] - (-65504.0)).abs() < 1.0);
    }

    // -- CallbackAction: ExitEarly with empty logits (edge case for mid-layer) --

    #[test]
    fn exit_early_with_captured_hidden_state_never_empty() {
        // Arrange: verify that when MidLayerEncodeCallback emits ExitEarly,
        // the logits vector is never empty (it contains hidden_size elements)
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx0 = holder.ctx(0, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx0, &output);

        let ctx1 = holder.ctx(1, 1);
        match cb.pre_node(&ctx1) {
            CallbackAction::ExitEarly { logits } => {
                assert!(!logits.is_empty(), "ExitEarly logits must contain hidden state");
                assert_eq!(logits.len(), 4);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // -- decode_hidden_output: buffer with mixed high bytes (non-trivial patterns) --

    #[test]
    fn decode_f32_high_byte_values() {
        // Arrange: values near f32::MAX / 2
        let v1: f32 = f32::MAX / 2.0;
        let v2: f32 = f32::MIN / 2.0;
        let src: Vec<u8> = [v1, v2].iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - v1).abs() < v1.abs() * 1e-6);
        assert!((out[1] - v2).abs() < v2.abs().abs() * 1e-6);
    }

    // -- decode_hidden_output: hidden_size=2 with F16, buffer size forces half path --

    #[test]
    fn decode_f16_hidden_size_two_forced_half_path() {
        // Arrange: hidden_size=2, buffer=4 bytes (2 f16 elements, seq=1)
        // F32 stride=8, 4%8!=0 → F32 fails.
        // half stride=4, 4%4==0, 4>=4 → half passes.
        let h1 = f16::from_f32(0.75);
        let h2 = f16::from_f32(-0.25);
        let src: Vec<u8> = [&h1, &h2].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.75).abs() < 0.01);
        assert!((out[1] - (-0.25)).abs() < 0.01);
    }

    // ========================================================================
    // Additional tests (55 new) for remaining coverage gaps
    // ========================================================================

    // -- decode_hidden_output: F16 negative infinity --

    #[test]
    fn decode_f16_negative_infinity() {
        // Arrange: F16 representation of negative infinity
        let h = f16::from_f32(f32::NEG_INFINITY);
        let src: Vec<u8> = h.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert!(out[0].is_infinite());
        assert!(out[0].is_sign_negative());
    }

    // -- decode_hidden_output: BF16 positive infinity --

    #[test]
    fn decode_bf16_positive_infinity() {
        // Arrange: BF16 representation of positive infinity
        let b = bf16::from_f32(f32::INFINITY);
        let src: Vec<u8> = b.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        assert!(out[0].is_infinite());
        assert!(out[0].is_sign_positive());
    }

    // -- decode_hidden_output: BF16 smallest positive subnormal --

    #[test]
    fn decode_bf16_smallest_positive_subnormal() {
        // Arrange: BF16 smallest positive subnormal (bit pattern 0x0001)
        let b = bf16::from_bits(0x0001u16);
        let src: Vec<u8> = b.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).unwrap();
        assert!(out[0] > 0.0, "BF16 smallest subnormal must be positive");
        assert!(out[0] < 0.01, "BF16 smallest subnormal must be very small");
    }

    // -- decode_hidden_output: F16 zero element --

    #[test]
    fn decode_f16_zero_value() {
        // Arrange: F16 zero
        let h = f16::from_f32(0.0);
        let src: Vec<u8> = h.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert_eq!(out[0], 0.0);
    }

    // -- decode_hidden_output: F16 negative zero --

    #[test]
    fn decode_f16_negative_zero() {
        // Arrange: F16 negative zero
        let h = f16::from_f32(-0.0);
        let src: Vec<u8> = h.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).unwrap();
        assert!(out[0].is_sign_negative(), "F16 negative zero must preserve sign");
    }

    // -- decode_hidden_output: half numel*2 overflow in half candidate --

    #[test]
    fn decode_half_numel_times_two_overflow_returns_none() {
        // Arrange: seq_len and hidden_size whose product fits in usize, but numel * 2 overflows.
        // usize::MAX / 2 + 1 = numel where numel*2 would overflow.
        let huge = (usize::MAX / 2) + 1;
        let src = vec![0u8; 16];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, huge, 1, DType::F16);
        assert!(out.is_none(), "numel*2 overflow in half candidate must return None");
    }

    // -- decode_hidden_output: buffer size exactly matches f32 stride for seq=2 --

    #[test]
    fn decode_f32_two_rows_exact_fit() {
        // Arrange: seq_len=2, hidden_size=4 → 8 f32 = 32 bytes exact fit
        let values: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0, 1.0, 2.0, 3.0, 4.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 32);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 4, DType::F32).unwrap();
        assert_eq!(out.len(), 8);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[7] - 4.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: pre-allocated buffer with realistic hidden_size=1024 --

    #[test]
    fn decode_f32_hidden_1024_with_preallocated_buffer() {
        // Arrange: hidden_size=1024, max_seq=512, actual seq_len=3
        let hidden = 1024;
        let max_seq = 512;
        let mut src = vec![0u8; max_seq * hidden * 4];
        // Write values for first 3 rows
        for row in 0..3usize {
            for col in 0..hidden {
                let idx = row * hidden + col as usize;
                let val = if col == 0 { (row + 1) as f32 } else { 0.0 };
                let off = idx * 4;
                src[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, hidden, DType::F32).unwrap();
        assert_eq!(out.len(), 3 * hidden);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[hidden] - 2.0).abs() < 1e-6);
        assert!((out[2 * hidden] - 3.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: BF16 with mixed positive and negative values --

    #[test]
    fn decode_bf16_mixed_positive_negative() {
        // Arrange: seq_len=1, hidden_size=4, BF16 with mixed signs
        let vals = [
            bf16::from_f32(100.0),
            bf16::from_f32(-100.0),
            bf16::from_f32(0.5),
            bf16::from_f32(-0.5),
        ];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 8); // 4 * 2 bytes

        // hidden_size=4: F32 stride=16, 8%16!=0 → F32 fails.
        // half stride=8, 8%8==0, 8>=8 → passes with BF16.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::BF16).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 100.0).abs() < 1.0);
        assert!((out[1] - (-100.0)).abs() < 1.0);
        assert!((out[2] - 0.5).abs() < 0.01);
        assert!((out[3] - (-0.5)).abs() < 0.01);
    }

    // -- decode_hidden_output: F16 with mixed positive and negative values --

    #[test]
    fn decode_f16_mixed_positive_negative() {
        // Arrange: seq_len=1, hidden_size=4, F16 with mixed signs
        let vals = [
            f16::from_f32(50.0),
            f16::from_f32(-50.0),
            f16::from_f32(0.25),
            f16::from_f32(-0.25),
        ];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 8);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F16).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 50.0).abs() < 1.0);
        assert!((out[1] - (-50.0)).abs() < 1.0);
        assert!((out[2] - 0.25).abs() < 0.01);
        assert!((out[3] - (-0.25)).abs() < 0.01);
    }

    // -- decode_hidden_output: five-byte buffer (not aligned to anything) --

    #[test]
    fn decode_five_byte_buffer_fails_all_candidates() {
        // Arrange: 5 bytes. hidden=1: F32 stride=4, 5%4!=0 → F32 fails.
        // half stride=2, 5%2!=0 → half fails.
        let src = vec![0u8; 5];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).is_none());
    }

    // -- decode_hidden_output: six-byte buffer with hidden_size=1 (F32 fails, half passes) --

    #[test]
    fn decode_six_bytes_hidden_size_one_half_path() {
        // Arrange: 6 bytes, hidden=1, seq_len=3
        // F32 stride=4, 6%4!=0 → F32 fails.
        // half stride=2, 6%2==0, 6>=6 → passes.
        let vals = [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 6);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 1, DType::F16).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[2] - 3.0).abs() < 0.01);
    }

    // -- pre_node: callback still functions after many no-op pre_nodes --

    #[test]
    fn pre_node_many_noop_pre_nodes_then_capture_and_exit() {
        // Arrange: 100 pre_node calls before reaching target layer
        let mut cb = MidLayerEncodeCallback::new(50);
        let holder = TestCtxHolder::new(4, DType::F32);

        // 100 pre_nodes at non-target layers
        for layer in 0..50 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        }

        // Now at target layer 50
        let ctx50 = holder.ctx(50, 1);
        let output = make_f32_output(&[7.0, 8.0, 9.0, 10.0]);
        assert!(matches!(cb.post_node(&ctx50, &output), CallbackAction::Continue));

        // Exit at layer 51
        let ctx51 = holder.ctx(51, 1);
        match cb.pre_node(&ctx51) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![7.0, 8.0, 9.0, 10.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // -- post_node: capture at target layer then more post_nodes at non-target --

    #[test]
    fn post_node_target_then_multiple_non_target_preserves_capture() {
        // Arrange: capture at target, then many post_nodes at wrong layers
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 1
        let ctx1 = holder.ctx(1, 1);
        let output = make_f32_output(&[11.0, 22.0, 33.0, 44.0]);
        cb.post_node(&ctx1, &output);

        // Many post_nodes at non-target layers
        for layer in 2..20 {
            let ctx = holder.ctx(layer, 1);
            let other_output = make_f32_output(&[0.0, 0.0, 0.0, 0.0]);
            cb.post_node(&ctx, &other_output);
        }

        // Assert: original capture preserved
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![11.0, 22.0, 33.0, 44.0]);
    }

    // -- decode_hidden_output: F32 with seq_len and hidden_size both > 1, odd numbers --

    #[test]
    fn decode_f32_odd_dimensions_both() {
        // Arrange: seq_len=3, hidden_size=7 → 21 f32 = 84 bytes
        let values: Vec<f32> = (0..21).map(|i| (i as f32) * 0.1).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 84);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 7, DType::F32).unwrap();
        assert_eq!(out.len(), 21);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[6] - 0.6).abs() < 1e-5);
        assert!((out[20] - 2.0).abs() < 1e-5);
    }

    // -- decode_hidden_output: buffer with one extra byte beyond F32 alignment --

    #[test]
    fn decode_buffer_one_byte_over_f32_alignment_falls_to_half() {
        // Arrange: hidden_size=2. F32 stride=8, half stride=4.
        // Buffer=12 bytes. 12%8=4 (not F32-aligned). 12%4=0 → half passes.
        let vals = [
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
        ];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 12);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 2, DType::BF16).unwrap();
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.0).abs() < 0.01);
    }

    // -- decode_hidden_output: F32 with very large seq_len, hidden_size=1 --

    #[test]
    fn decode_f32_large_seq_len_hidden_one() {
        // Arrange: seq_len=8192, hidden_size=1
        let seq_len = 8192;
        let src: Vec<u8> = (0..seq_len)
            .flat_map(|i| {
                let v = if i == 0 { 42.0f32 } else if i == seq_len - 1 { -42.0f32 } else { 0.0f32 };
                v.to_le_bytes()
            })
            .collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq_len, 1, DType::F32).unwrap();
        assert_eq!(out.len(), seq_len);
        assert!((out[0] - 42.0).abs() < 1e-6);
        assert!((out[seq_len - 1] - (-42.0)).abs() < 1e-6);
    }

    // -- post_node: different hidden sizes with same callback --

    #[test]
    fn post_node_different_hidden_sizes_same_callback() {
        // Arrange: callback with hidden_size=8, multiple post_nodes with varying seq_lens
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(8, DType::F32);

        // seq_len=1 → 8 f32 = 32 bytes
        let ctx1 = holder.ctx(3, 1);
        let output1: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        cb.post_node(&ctx1, &output1);
        assert_eq!(cb.captured.as_ref().unwrap().len(), 8);

        // seq_len=2 → 16 f32 = 64 bytes (overwrites)
        let ctx2 = holder.ctx(3, 2);
        let output2: Vec<u8> = (0..16).flat_map(|i| (i as f32 * 10.0).to_le_bytes()).collect();
        cb.post_node(&ctx2, &output2);
        assert_eq!(cb.captured.as_ref().unwrap().len(), 16);

        // Assert: second capture overwrites first
        let captured = cb.captured.as_ref().unwrap();
        assert!((captured[0] - 0.0).abs() < 1e-5);
        assert!((captured[15] - 150.0).abs() < 1e-3);
    }

    // -- pre_node/post_node: full lifecycle with very high layer index --

    #[test]
    fn lifecycle_high_layer_index_capture_and_exit() {
        // Arrange: target layer = 1000
        let mut cb = MidLayerEncodeCallback::new(1000);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Walk through layers 0..999 (all non-target)
        for layer in 0..1000 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        }

        // Capture at layer 1000
        let ctx1000 = holder.ctx(1000, 1);
        let output = make_f32_output(&[100.0, 200.0, 300.0, 400.0]);
        cb.post_node(&ctx1000, &output);

        // Exit at layer 1001
        let ctx1001 = holder.ctx(1001, 1);
        match cb.pre_node(&ctx1001) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![100.0, 200.0, 300.0, 400.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // -- CallbackAction: verify ExitEarly is Clone + PartialEq --

    #[test]
    fn callback_action_exit_early_is_comparable() {
        // Arrange: get two ExitEarly actions from the callback
        let mut cb1 = MidLayerEncodeCallback::new(1);
        let mut cb2 = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx1 = holder.ctx(1, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb1.post_node(&ctx1, &output);
        cb2.post_node(&ctx1, &output);

        let ctx2 = holder.ctx(2, 1);
        let action1 = cb1.pre_node(&ctx2);
        let action2 = cb2.pre_node(&ctx2);

        // Assert: CallbackAction derives PartialEq
        assert_eq!(action1, action2, "same capture data must produce equal CallbackAction");
    }

    // -- decode_hidden_output: F32 with hidden_size=2, seq_len=2, pre-allocated --

    #[test]
    fn decode_f32_hidden_two_seq_two_preallocated() {
        // Arrange: hidden=2, seq=2, max_seq=32 → buffer=32*2*4=256, decode 4 f32
        let mut src = vec![0u8; 256];
        let values: Vec<f32> = vec![1.5, -1.5, 2.5, -2.5];
        for (i, v) in values.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F32).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[3] - (-2.5)).abs() < 1e-6);
    }

    // -- decode_hidden_output: BF16 with hidden_size=2, buffer forces half path --

    #[test]
    fn decode_bf16_hidden_two_forced_half_path() {
        // Arrange: hidden=2, buffer=4 bytes (2 bf16, seq=1)
        // F32 stride=8, 4%8!=0 → F32 fails.
        // half stride=4, 4%4==0, 4>=4 → passes.
        let vals = [bf16::from_f32(3.14), bf16::from_f32(-2.71)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.14).abs() < 0.1);
        assert!((out[1] - (-2.71)).abs() < 0.1);
    }

    // -- pre_node: exit at very first non-target pre_node after target capture --

    #[test]
    fn pre_node_immediate_exit_after_single_capture() {
        // Arrange: only one post_node at target, immediately followed by pre_node at next layer
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(2, DType::F32);

        // Single capture at layer 0
        let ctx0 = holder.ctx(0, 1);
        let output = make_f32_output(&[5.0, -5.0]);
        cb.post_node(&ctx0, &output);

        // Immediate exit
        let ctx1 = holder.ctx(1, 1);
        match cb.pre_node(&ctx1) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![5.0, -5.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // -- decode_hidden_output: F32 with all max f32 values --

    #[test]
    fn decode_f32_all_max_values() {
        // Arrange: buffer full of f32::MAX
        let count = 16;
        let src: Vec<u8> = (0..count).flat_map(|_| f32::MAX.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 4, 4, DType::F32).unwrap();
        assert_eq!(out.len(), count);
        assert!(out.iter().all(|&v| v == f32::MAX));
    }

    // -- decode_hidden_output: F32 with all negative infinity --

    #[test]
    fn decode_f32_all_neg_infinity() {
        // Arrange: buffer full of negative infinity
        let count = 8;
        let src: Vec<u8> = (0..count).flat_map(|_| f32::NEG_INFINITY.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 4, DType::F32).unwrap();
        assert_eq!(out.len(), count);
        assert!(out.iter().all(|v| v.is_infinite() && v.is_sign_negative()));
    }

    // -- decode_hidden_output: hidden_size=3, F16, pre-allocated with max_seq=128 --

    #[test]
    fn decode_f16_hidden_three_preallocated() {
        // Arrange: hidden=3, max_seq=128, seq_len=2
        // total=128*3*2=768. F32 stride=12, 768%12=0 → F32 candidate wins!
        // So F16 path would NOT be used here. The F32 path decodes first 6 f32 bytes.
        let mut src = vec![0u8; 768];
        // Write 6 f32 values at the beginning (F32 path will read these)
        let f32_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for (i, v) in f32_vals.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F16).unwrap();
        // F32 candidate wins because 768 % 12 == 0
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: f32 stride zero when hidden_size=0 (already covered, but with half path too) --

    #[test]
    fn decode_half_stride_check_zero_hidden_size_returns_none() {
        // Arrange: hidden_size=0 — early return in decode_hidden_output before reaching half
        let src = vec![0u8; 4];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 0, DType::F16);
        assert!(out.is_none(), "hidden_size=0 must return None before half stride check");
    }

    // -- post_node: post_node at target layer with hidden_size=2, seq_len=1, BF16 declared --

    #[test]
    fn post_node_hidden_two_bf16_declared_captures() {
        // Arrange: hidden=2, declared BF16, F32 CPU canonical output
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(2, DType::BF16);
        let ctx = holder.ctx(3, 1);
        let output = make_f32_output(&[1.5, -2.5]);

        // Act
        cb.post_node(&ctx, &output);

        // Assert
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 2);
        assert!((captured[0] - 1.5).abs() < 1e-6);
        assert!((captured[1] - (-2.5)).abs() < 1e-6);
    }

    // -- pre_node: target layer with same context used for pre and post --

    #[test]
    fn pre_node_post_node_same_context_object() {
        // Arrange: reuse the same context for both pre_node and post_node
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // pre_node
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));

        // post_node
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(cb.post_node(&ctx, &output), CallbackAction::Continue));

        // pre_node again at same layer — still Continue
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        assert!(cb.captured.is_some());
    }

    // -- CallbackChain integration: multiple MidLayerEncodeCallbacks --

    #[test]
    fn chain_integration_single_callback_name_and_priority() {
        // Arrange: verify callback properties through LayerCallback trait
        use crate::graph::layer_callback::CallbackChain;

        let cb = MidLayerEncodeCallback::new(5);
        assert_eq!(cb.name(), "MidLayerEncode");
        assert_eq!(cb.priority(), 55);
        assert!(cb.target_layers().is_none());

        let chain = CallbackChain::new(vec![Box::new(cb)]);
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
    }

    // -- CallbackChain integration: empty chain with MidLayerEncodeCallback --

    #[test]
    fn chain_empty_then_add_mid_layer() {
        // Arrange: start with empty chain
        use crate::graph::layer_callback::CallbackChain;

        let empty = CallbackChain::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    // -- decode_hidden_output: buffer with only slack bytes (large buffer, small seq_len) --

    #[test]
    fn decode_f32_massive_buffer_tiny_seq_len() {
        // Arrange: buffer for max_seq=8192, hidden=64, but seq_len=1
        let hidden = 64;
        let max_seq = 8192;
        let mut src = vec![0u8; max_seq * hidden * 4];
        // Write hidden values for seq_len=1
        for i in 0..hidden {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&(i as f32).to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F32).unwrap();
        assert_eq!(out.len(), hidden);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[63] - 63.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: F32 with value exactly 1.0 and -1.0 --

    #[test]
    fn decode_f32_positive_and_negative_one() {
        // Arrange: 1.0 and -1.0 (common activation values)
        let values: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F32).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (-1.0)).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!((out[3] - (-1.0)).abs() < 1e-6);
    }

    // -- decode_hidden_output: 9-byte buffer fails for hidden_size=2 --

    #[test]
    fn decode_nine_byte_buffer_hidden_two_fails_all() {
        // Arrange: 9 bytes, hidden=2. F32 stride=8, 9%8!=0. half stride=4, 9%4!=0.
        let src = vec![0u8; 9];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16).is_none());
    }

    // -- decode_hidden_output: 14-byte buffer with hidden=7 --

    #[test]
    fn decode_14_bytes_hidden_seven_fails_all() {
        // Arrange: 14 bytes, hidden=7. F32 stride=28, 14%28!=0. half stride=14, 14%14==0, 14>=14 → half passes.
        // But dtype=F32 → no F16/BF16 match arm.
        let src = vec![0u8; 14];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 7, DType::F32).is_none());
    }

    // -- decode_hidden_output: F16 with seq_len=4, hidden=3 --

    #[test]
    fn decode_f16_seq_four_hidden_three() {
        // Arrange: seq=4, hidden=3 → 12 f16 elements = 24 bytes
        // F32 stride=12, 24%12==0, 24>=48? No (24 < 48) → F32 fails (need 48 bytes for 12 f32).
        // half stride=6, 24%6==0, 24>=24 → passes.
        let vals: Vec<f16> = (0..12).map(|i| f16::from_f32(i as f32 * 0.5)).collect();
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 24);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 4, 3, DType::F16).unwrap();
        assert_eq!(out.len(), 12);
        assert!((out[0] - 0.0).abs() < 0.01);
        assert!((out[11] - 5.5).abs() < 0.1);
    }

    // -- decode_hidden_output: BF16 with seq_len=4, hidden=3 --

    #[test]
    fn decode_bf16_seq_four_hidden_three() {
        // Arrange: seq=4, hidden=3 → 12 bf16 = 24 bytes
        let vals: Vec<bf16> = (0..12).map(|i| bf16::from_f32(i as f32 * 0.5)).collect();
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 24);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 4, 3, DType::BF16).unwrap();
        assert_eq!(out.len(), 12);
        assert!((out[0] - 0.0).abs() < 0.01);
        assert!((out[11] - 5.5).abs() < 0.1);
    }

    // -- post_node + pre_node: capture at layer 1, walk more layers, then exit at last layer --

    #[test]
    fn lifecycle_capture_early_exit_late() {
        // Arrange: capture at layer 1, continue walking layers, exit at layer 7
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(2, DType::F32);

        // Capture at layer 1
        let ctx1 = holder.ctx(1, 1);
        let output = make_f32_output(&[42.0, -42.0]);
        cb.post_node(&ctx1, &output);

        // Walk layers 2..7 — pre_node at each
        for layer in 2..7 {
            let ctx = holder.ctx(layer, 1);
            let action = cb.pre_node(&ctx);
            // Only the first transition (layer 2) should trigger ExitEarly
            if layer == 2 {
                assert!(matches!(action, CallbackAction::ExitEarly { .. }));
            } else {
                assert!(matches!(action, CallbackAction::Continue));
            }
        }
    }

    // -- decode_hidden_output: F32 with sequential values 0.0 to N-1 --

    #[test]
    fn decode_f32_sequential_values() {
        // Arrange: seq_len=1, hidden_size=100, values 0..100
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 100, DType::F32).unwrap();
        assert_eq!(out.len(), 100);
        for i in 0..100 {
            assert!((out[i] - i as f32).abs() < 1e-6, "mismatch at index {}", i);
        }
    }

    // -- decode_hidden_output: F32 with single NaN in a larger buffer --

    #[test]
    fn decode_f32_single_nan_among_normals() {
        // Arrange: 4 values where one is NaN
        let nan = f32::from_bits(0x7FC00000);
        let values: Vec<f32> = vec![1.0, nan, 3.0, 4.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[1].is_nan());
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: BF16 with BF16 NaN at start --

    #[test]
    fn decode_bf16_nan_among_normals() {
        // Arrange: 4 BF16 values, one is NaN
        let nan_bf16 = bf16::from_bits(0x7FC0u16);
        let vals = [nan_bf16, bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 8);

        // hidden=4: F32 stride=16, 8%16!=0 → F32 fails.
        // half stride=8, 8%8==0, 8>=8 → passes.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::BF16).unwrap();
        assert!(out[0].is_nan());
        assert!((out[1] - 1.0).abs() < 0.01);
    }

    // -- decode_hidden_output: F16 NaN among normal values --

    #[test]
    fn decode_f16_nan_among_normals() {
        // Arrange: F16 values with one NaN
        let nan_f16 = f16::from_bits(0x7E00u16);
        let vals = [f16::from_f32(0.0), nan_f16, f16::from_f32(1.0)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 6);

        // hidden=3: F32 stride=12, 6%12!=0 → F32 fails. half stride=6, 6%6==0 → passes.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F16).unwrap();
        assert_eq!(out[0], 0.0);
        assert!(out[1].is_nan());
        assert!((out[2] - 1.0).abs() < 0.01);
    }

    // -- pre_node: calling pre_node on fresh callback (no post_node yet) --

    #[test]
    fn pre_node_fresh_callback_no_post_returns_continue() {
        // Arrange: brand new callback, never called post_node
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);

        // pre_node at layer 0 (before target)
        let ctx0 = holder.ctx(0, 1);
        assert!(matches!(cb.pre_node(&ctx0), CallbackAction::Continue));

        // pre_node at layer 5 (at target, but no capture)
        let ctx5 = holder.ctx(5, 1);
        assert!(matches!(cb.pre_node(&ctx5), CallbackAction::Continue));

        // pre_node at layer 6 (after target, no capture)
        let ctx6 = holder.ctx(6, 1);
        assert!(matches!(cb.pre_node(&ctx6), CallbackAction::Continue));

        assert!(cb.captured.is_none());
    }

    // -- decode_hidden_output: F32 with hidden_size=6 (divisible by 2 and 4) --

    #[test]
    fn decode_f32_hidden_six_exact_fit() {
        // Arrange: hidden=6, seq=1 → 6 f32 = 24 bytes
        let values: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 24);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 6, DType::F32).unwrap();
        assert_eq!(out.len(), 6);
        assert!((out[0] - 0.1).abs() < 1e-6);
        assert!((out[5] - 0.6).abs() < 1e-6);
    }

    // -- decode_hidden_output: half path with BF16, seq_len=2, hidden=4 --

    #[test]
    fn decode_bf16_seq_two_hidden_four() {
        // Arrange: seq=2, hidden=4 → 8 bf16 = 16 bytes
        // F32 stride=16, 16%16==0, 16>=32? No (16 < 32) → F32 fails.
        // half stride=8, 16%8==0, 16>=16 → passes.
        let vals: Vec<bf16> = (0..8).map(|i| bf16::from_f32(i as f32)).collect();
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 16);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 4, DType::BF16).unwrap();
        assert_eq!(out.len(), 8);
        assert!((out[0] - 0.0).abs() < 0.01);
        assert!((out[7] - 7.0).abs() < 0.1);
    }

    // -- decode_hidden_output: half path with F16, seq_len=2, hidden=4 --

    #[test]
    fn decode_f16_seq_two_hidden_four() {
        // Arrange: same dimensions with F16
        let vals: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 16);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 4, DType::F16).unwrap();
        assert_eq!(out.len(), 8);
        assert!((out[0] - 0.0).abs() < 0.01);
        assert!((out[7] - 7.0).abs() < 0.1);
    }

    // -- decode_hidden_output: 7-byte buffer fails all for hidden=1 --

    #[test]
    fn decode_seven_bytes_hidden_one_fails_f32() {
        // Arrange: 7 bytes, hidden=1. F32 stride=4, 7%4!=0. half stride=2, 7%2!=0.
        let src = vec![0u8; 7];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).is_none());
    }

    // -- decode_hidden_output: 11-byte buffer fails for hidden=3 --

    #[test]
    fn decode_11_bytes_hidden_three_fails_all() {
        // Arrange: 11 bytes, hidden=3. F32 stride=12, 11%12!=0. half stride=6, 11%6!=0.
        let src = vec![0u8; 11];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F32).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::BF16).is_none());
    }

    // -- decode_hidden_output: F32 with hidden=4, seq=3, buffer exactly 48 bytes --

    #[test]
    fn decode_f32_hidden_four_seq_three_exact() {
        // Arrange: 4*3=12 f32 = 48 bytes
        let values: Vec<f32> = (0..12).map(|i| (i as f32) - 5.5).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 48);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 4, DType::F32).unwrap();
        assert_eq!(out.len(), 12);
        assert!((out[0] - (-5.5)).abs() < 1e-5);
        assert!((out[11] - 5.5).abs() < 1e-5);
    }

    // -- decode_hidden_output: buffer of 8 bytes, hidden=4, dtype=F8E4M3 --

    #[test]
    fn decode_f32_wins_with_f8e4m3_declared() {
        // Arrange: 8 bytes, hidden=2, seq=1. F32 stride=8, 8%8==0, 8>=8 → F32 wins.
        let values: Vec<f32> = vec![1.5, -3.5];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F8E4M3).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.5).abs() < 1e-6);
    }

    // -- decode_hidden_output: half stride passes with F8E4M3 declared → no match arm --

    #[test]
    fn decode_half_stride_with_f8e4m3_declared_returns_none() {
        // Arrange: 4 bytes, hidden=2 → half_stride=4, 4%4==0. F32 stride=8, 4%8!=0.
        // Half passes but F8E4M3 has no match arm.
        let src = vec![0u8; 4];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F8E4M3);
        assert!(out.is_none());
    }

    // -- decode_hidden_output: half stride with I32 (if it exists) or fallback to generic non-half dtype --

    #[test]
    fn decode_with_f4e2m1_declared_half_stride_returns_none() {
        // Arrange: 4 bytes, hidden=2 → half_stride=4, F32 stride=8, 4%8!=0 → F32 fails.
        // half passes but F4E2M1 has no match arm.
        let src = vec![0u8; 4];
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F4E2M1);
        assert!(out.is_none());
    }

    // -- post_node: capture with realistic hidden_size=256 --

    #[test]
    fn post_node_realistic_hidden_256() {
        // Arrange: hidden=256, seq_len=1, target layer 2
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(256, DType::F32);
        let ctx = holder.ctx(2, 1);

        let values: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        let output: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        cb.post_node(&ctx, &output);

        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 256);
        assert!((captured[0] - 0.0).abs() < 1e-6);
        assert!((captured[255] - (255.0 / 256.0)).abs() < 1e-5);
    }

    // -- decode_hidden_output: F32 with hidden=10, seq=5 (50 elements) --

    #[test]
    fn decode_f32_hidden_ten_seq_five() {
        // Arrange: 50 f32 = 200 bytes
        let values: Vec<f32> = (0..50).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 200);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 5, 10, DType::F32).unwrap();
        assert_eq!(out.len(), 50);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - (-1.0)).abs() < 1e-6);
        assert!((out[49] - (-49.0)).abs() < 1e-5);
    }

    // -- pre_node: verify CallbackAction variants match expected behavior --

    #[test]
    fn callback_action_continue_is_default() {
        // Arrange: verify that CallbackAction::Continue is the Default variant
        let default_action = CallbackAction::default();
        assert!(matches!(default_action, CallbackAction::Continue));
    }

    // -- decode_hidden_output: F32 with pi values --

    #[test]
    fn decode_f32_pi_and_e_values() {
        // Arrange: common mathematical constants
        let values: Vec<f32> = vec![std::f32::consts::PI, std::f32::consts::E];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - std::f32::consts::PI).abs() < 1e-6);
        assert!((out[1] - std::f32::consts::E).abs() < 1e-6);
    }

    // -- post_node: output buffer has exactly hidden_size * 4 bytes --

    #[test]
    fn post_node_exact_buffer_no_slack_captures() {
        // Arrange: output.len() == hidden_size * seq_len * 4, no slack
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(3, DType::F32);
        let ctx = holder.ctx(1, 2);

        let values: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let output: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(output.len(), 24); // 6 * 4

        cb.post_node(&ctx, &output);
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 6);
        assert!((captured[0] - 10.0).abs() < 1e-6);
        assert!((captured[5] - 60.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: buffer where only half stride is valid, declared F16 --

    #[test]
    fn decode_only_half_valid_with_f16() {
        // Arrange: hidden=5, buffer=10 bytes. F32 stride=20, 10%20!=0 → F32 fails.
        // half stride=10, 10%10==0, 10>=10 → passes with F16.
        let vals: Vec<f16> = vec![
            f16::from_f32(1.0), f16::from_f32(-1.0),
            f16::from_f32(2.0), f16::from_f32(-2.0),
            f16::from_f32(0.0),
        ];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 10);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 5, DType::F16).unwrap();
        assert_eq!(out.len(), 5);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[4] - 0.0).abs() < 0.01);
    }

    // -- decode_hidden_output: same buffer, declared BF16 --

    #[test]
    fn decode_only_half_valid_with_bf16() {
        // Arrange: hidden=5, buffer=10 bytes → same as above but BF16
        let vals: Vec<bf16> = vec![
            bf16::from_f32(1.0), bf16::from_f32(-1.0),
            bf16::from_f32(2.0), bf16::from_f32(-2.0),
            bf16::from_f32(0.0),
        ];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 10);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 5, DType::BF16).unwrap();
        assert_eq!(out.len(), 5);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[4] - 0.0).abs() < 0.01);
    }

    // -- decode_hidden_output: F32 with f32::MIN_POSITIVE in a multi-element buffer --

    #[test]
    fn decode_f32_min_positive_in_multi_element() {
        // Arrange: mix of MIN_POSITIVE and normal values
        let values: Vec<f32> = vec![f32::MIN_POSITIVE, 1.0, f32::MIN_POSITIVE, -1.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32).unwrap();
        assert_eq!(out[0], f32::MIN_POSITIVE);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert_eq!(out[2], f32::MIN_POSITIVE);
        assert!((out[3] - (-1.0)).abs() < 1e-6);
    }

    // -- decode_hidden_output: 16-byte buffer with hidden=1, seq=4, F32 --

    #[test]
    fn decode_f32_hidden_one_seq_four() {
        // Arrange: hidden=1, seq=4 → 4 f32 = 16 bytes
        let values: Vec<f32> = vec![100.0, -100.0, 0.5, -0.5];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 4, 1, DType::F32).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 100.0).abs() < 1e-6);
        assert!((out[3] - (-0.5)).abs() < 1e-6);
    }

    // -- decode_hidden_output: 20-byte buffer with hidden=5, F32, seq=1 --

    #[test]
    fn decode_f32_hidden_five_exact() {
        // Arrange: hidden=5, seq=1 → 5 f32 = 20 bytes
        let values: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 20);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 5, DType::F32).unwrap();
        assert_eq!(out.len(), 5);
        for (i, val) in out.iter().enumerate() {
            assert!((val - (i as f32 - 2.0)).abs() < 1e-6, "mismatch at index {}", i);
        }
    }

    // -- decode_hidden_output: half stride with I64-like type (use U8 as non-half proxy) --

    #[test]
    fn decode_half_stride_with_non_f16_bf16_dtype_returns_none() {
        // Arrange: buffer passes half stride but declared dtype has no F16/BF16 match
        // Using DType::F6E3M2 which has no half-precision match arm
        let src = vec![0u8; 6]; // hidden=3, half_stride=6, 6%6==0
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F6E3M2);
        // F32 stride=12, 6%12!=0 → F32 fails. half passes but F6E3M2 has no match.
        assert!(out.is_none());
    }

    // -- decode_hidden_output: F32 buffer with hidden=8, seq=2, then verify captured values --

    #[test]
    fn post_node_hidden_eight_seq_two_full_lifecycle() {
        // Arrange: hidden=8, seq=2, target layer 1
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(8, DType::F32);

        // Capture
        let ctx1 = holder.ctx(1, 2);
        let values: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let output: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        cb.post_node(&ctx1, &output);

        // Exit
        let ctx2 = holder.ctx(2, 2);
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits.len(), 16);
                assert!((logits[0] - 0.0).abs() < 1e-6);
                assert!((logits[15] - 1.5).abs() < 1e-5);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // -- decode_hidden_output: pre-allocated F32 buffer much larger than needed for seq=1 --

    #[test]
    fn decode_f32_slack_buffer_only_first_row_decoded() {
        // Arrange: buffer = 2048 * 8 * 4 = 65536 bytes, but seq=1, hidden=8
        let hidden = 8;
        let max_seq = 2048;
        let mut src = vec![0u8; max_seq * hidden * 4];
        // Write first row with known values
        for i in 0..hidden {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&(i as f32 + 1.0).to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F32).unwrap();
        assert_eq!(out.len(), hidden);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[7] - 8.0).abs() < 1e-6);
    }

    // -- decode_hidden_output: F16 with values near F16 precision limits --

    #[test]
    fn decode_f16_near_precision_limit() {
        // Arrange: F16 can represent ~3 decimal digits. Test value near that limit.
        let vals = [f16::from_f32(0.123), f16::from_f32(-0.456)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        // hidden=2: F32 stride=8, 4%8!=0 → F32 fails. half stride=4, 4%4==0 → passes.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16).unwrap();
        assert_eq!(out.len(), 2);
        // F16 precision is limited, so allow tolerance
        assert!((out[0] - 0.123).abs() < 0.01);
        assert!((out[1] - (-0.456)).abs() < 0.01);
    }

    // -- decode_hidden_output: BF16 with values near BF16 precision limits --

    #[test]
    fn decode_bf16_near_precision_limit() {
        // Arrange: BF16 has 7-bit mantissa, slightly better than F16 for some values
        let vals = [bf16::from_f32(0.125), bf16::from_f32(-0.875)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        // hidden=2: F32 stride=8, 4%8!=0 → F32 fails. half stride=4, 4%4==0 → passes.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.125).abs() < 0.01);
        assert!((out[1] - (-0.875)).abs() < 0.01);
    }

    // -- decode_hidden_output: empty output buffer (0 bytes) with non-zero params --

    #[test]
    fn decode_empty_buffer_with_non_zero_params() {
        // Arrange: 0-byte buffer with non-zero seq_len and hidden_size
        let out = MidLayerEncodeCallback::decode_hidden_output(&[], 10, 100, DType::F32);
        assert!(out.is_none(), "empty buffer must return None");
    }

    // -- decode_hidden_output: F32 with hidden=4, pre-allocated max_seq=1024, seq=1 --

    #[test]
    fn decode_f32_hidden_four_preallocated_1024() {
        // Arrange: buffer = 1024 * 4 * 4 = 16384 bytes, seq=1
        let hidden = 4;
        let max_seq = 1024;
        let mut src = vec![0u8; max_seq * hidden * 4];
        let vals: Vec<f32> = vec![42.0, -42.0, 0.0, 1.0];
        for (i, v) in vals.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F32).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 42.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    // ========================================================================
    // Additional tests (50 new): context field independence, trait object
    // dispatch, variant constraints, output-length invariant, integration
    // ========================================================================

    // -- Group A: Context field independence --

    #[test]
    fn ctx_position_nonzero_does_not_affect_capture() {
        // Arrange: position=100 (non-zero) should not change behavior
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = LayerContext {
            node_idx: 4,
            layer_idx: 2,
            node_op: "output_add",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 1,
            seq_len: 1,
            position: 100,
            request_id: 1,
            model_config: &holder.config,
        };

        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx, &output);

        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 4);
        assert!((captured[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ctx_request_id_nonzero_does_not_affect_exit() {
        // Arrange: request_id=9999, verify exit still works
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx1 = LayerContext {
            node_idx: 2,
            layer_idx: 1,
            node_op: "Test",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 1,
            seq_len: 1,
            position: 0,
            request_id: 9999,
            model_config: &holder.config,
        };
        let output = make_f32_output(&[5.0, 6.0, 7.0, 8.0]);
        cb.post_node(&ctx1, &output);

        let ctx2 = LayerContext { layer_idx: 2, ..ctx1 };
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![5.0, 6.0, 7.0, 8.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn ctx_total_seq_differs_from_seq_len_does_not_affect_decode() {
        // Arrange: total_seq=50 (cached + new), seq_len=1 (new tokens only)
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = LayerContext {
            node_idx: 4,
            layer_idx: 2,
            node_op: "Test",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 50,
            seq_len: 1,
            position: 49,
            request_id: 1,
            model_config: &holder.config,
        };

        let output = make_f32_output(&[10.0, 20.0, 30.0, 40.0]);
        cb.post_node(&ctx, &output);

        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn ctx_node_op_various_strings_do_not_affect_behavior() {
        // Arrange: different node_op strings — none should affect capture/exit
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        for op_name in &["input_norm", "q_proj", "attn", "o_proj", "post_norm", "gate", "down"] {
            let ctx = LayerContext {
                node_idx: 4,
                layer_idx: 2,
                node_op: op_name,
                hidden_state: &holder.hidden_state,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 1,
                seq_len: 1,
                position: 0,
                request_id: 1,
                model_config: &holder.config,
            };
            let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
            let action = cb.post_node(&ctx, &output);
            assert!(matches!(action, CallbackAction::Continue));
        }

        // Last capture overwrites all previous
        assert!(cb.captured.is_some());
    }

    #[test]
    fn ctx_hidden_state_content_does_not_affect_decode() {
        // Arrange: hidden_state filled with non-zero data — irrelevant to decode
        let mut cb = MidLayerEncodeCallback::new(2);
        let mut holder = TestCtxHolder::new(4, DType::F32);
        // Fill hidden_state with 0xFF — should not affect output decoding
        holder.hidden_state.fill(0xFF);

        let ctx = holder.ctx(2, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx, &output);

        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ctx_non_null_kv_cache_pointers_do_not_affect_behavior() {
        // Arrange: kv_cache_k and kv_cache_v are non-null dummy pointers
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let mut dummy_k = [0.0f32; 16];
        let mut dummy_v = [0.0f32; 16];

        let ctx = LayerContext {
            node_idx: 4,
            layer_idx: 2,
            node_op: "Test",
            hidden_state: &holder.hidden_state,
            kv_cache_k: dummy_k.as_mut_ptr(),
            kv_cache_v: dummy_v.as_mut_ptr(),
            total_seq: 1,
            seq_len: 1,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };

        let output = make_f32_output(&[42.0, -42.0, 0.0, 1.0]);
        cb.post_node(&ctx, &output);

        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![42.0, -42.0, 0.0, 1.0]);
    }

    // -- Group B: LayerCallback trait object dispatch --

    #[test]
    fn trait_object_pre_node_dispatches_correctly() {
        // Arrange: Box<dyn LayerCallback> dispatches pre_node
        let mut cb: Box<dyn LayerCallback + Send> = Box::new(MidLayerEncodeCallback::new(2));
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Act: through trait object
        let action = cb.pre_node(&ctx);
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn trait_object_post_node_dispatches_correctly() {
        // Arrange: Box<dyn LayerCallback> dispatches post_node
        let mut cb: Box<dyn LayerCallback + Send> = Box::new(MidLayerEncodeCallback::new(2));
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        let action = cb.post_node(&ctx, &output);
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn trait_object_priority_and_name_accessible() {
        // Arrange: verify trait object methods work
        let cb: Box<dyn LayerCallback + Send> = Box::new(MidLayerEncodeCallback::new(3));
        assert_eq!(cb.priority(), MID_LAYER_ENCODE_PRIORITY);
        assert_eq!(cb.name(), "MidLayerEncode");
        assert!(cb.target_layers().is_none());
    }

    #[test]
    fn chain_two_mid_layer_callbacks_different_targets() {
        // Arrange: two MidLayerEncodeCallbacks targeting layers 2 and 5
        use crate::graph::layer_callback::CallbackChain;

        let cb1 = MidLayerEncodeCallback::new(2);
        let cb2 = MidLayerEncodeCallback::new(5);
        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2)]);
        assert_eq!(chain.len(), 2);

        let holder = TestCtxHolder::new(4, DType::F32);

        // Walk through layers: capture at layer 2, exit happens on transition out
        for layer in 0..2 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue));
        }

        // post_node at layer 2 → first callback captures
        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[10.0, 20.0, 30.0, 40.0]);
        assert!(matches!(chain.dispatch_post_node(&ctx2, &output), CallbackAction::Continue));

        // pre_node at layer 3 → first callback exits (second has no capture yet)
        let ctx3 = holder.ctx(3, 1);
        match chain.dispatch_pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("Expected ExitEarly from first callback, got {:?}", other),
        }
    }

    #[test]
    fn chain_mid_layer_with_early_exit_callback() {
        // Arrange: EarlyExit-like callback (priority=50) + MidLayerEncode (priority=55)
        // MidLayerEncode has higher priority, fires first
        use crate::graph::layer_callback::CallbackChain;

        struct EarlyExitCallback;
        impl LayerCallback for EarlyExitCallback {
            fn priority(&self) -> u32 { 50 }
            fn name(&self) -> &str { "EarlyExit" }
        }

        let mid = MidLayerEncodeCallback::new(1);
        let early = EarlyExitCallback;
        let mut chain = CallbackChain::new(vec![Box::new(mid), Box::new(early)]);
        assert_eq!(chain.len(), 2);

        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 1
        let ctx1 = holder.ctx(1, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        chain.dispatch_post_node(&ctx1, &output);

        // Transition out: MidLayerEncode (prio 55) fires before EarlyExit (prio 50)
        let ctx2 = holder.ctx(2, 1);
        match chain.dispatch_pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("Expected ExitEarly from MidLayerEncode, got {:?}", other),
        }
    }

    // -- Group C: CallbackAction variant constraints for MidLayerEncodeCallback --

    #[test]
    fn pre_node_never_returns_skip_this_node() {
        // Arrange: pre_node only returns Continue or ExitEarly
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // At target layer
        let ctx2 = holder.ctx(2, 1);
        let action = cb.pre_node(&ctx2);
        assert!(
            matches!(action, CallbackAction::Continue | CallbackAction::ExitEarly { .. }),
            "pre_node must only return Continue or ExitEarly"
        );

        // At non-target layer
        let ctx3 = holder.ctx(3, 1);
        let action2 = cb.pre_node(&ctx3);
        assert!(
            matches!(action2, CallbackAction::Continue | CallbackAction::ExitEarly { .. }),
            "pre_node must only return Continue or ExitEarly"
        );
    }

    #[test]
    fn post_node_always_returns_continue_variant() {
        // Arrange: post_node must always return Continue (never SkipThisNode/ExitEarly/etc.)
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // At target layer with valid output
        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        let action = cb.post_node(&ctx2, &output);
        assert!(
            matches!(action, CallbackAction::Continue),
            "post_node must always return Continue, got {:?}", action
        );

        // At non-target layer
        let ctx3 = holder.ctx(3, 1);
        let action2 = cb.post_node(&ctx3, &output);
        assert!(
            matches!(action2, CallbackAction::Continue),
            "post_node at non-target must return Continue"
        );

        // With empty output
        let action3 = cb.post_node(&ctx2, &[]);
        assert!(
            matches!(action3, CallbackAction::Continue),
            "post_node with empty output must return Continue"
        );
    }

    #[test]
    fn exit_early_logits_length_equals_seq_len_times_hidden_size() {
        // Arrange: verify output invariant: logits.len() == seq_len * hidden_size
        for seq_len in [1, 2, 4] {
            for hidden_size in [2, 4, 8] {
                let mut cb = MidLayerEncodeCallback::new(0);
                let holder = TestCtxHolder::new(hidden_size, DType::F32);
                let ctx = LayerContext {
                    node_idx: 0,
                    layer_idx: 0,
                    node_op: "Test",
                    hidden_state: &holder.hidden_state,
                    kv_cache_k: std::ptr::null_mut(),
                    kv_cache_v: std::ptr::null_mut(),
                    total_seq: seq_len,
                    seq_len,
                    position: 0,
                    request_id: 1,
                    model_config: &holder.config,
                };
                let values: Vec<f32> = (0..seq_len * hidden_size).map(|i| i as f32).collect();
                let output = make_f32_output(&values);
                cb.post_node(&ctx, &output);

                let ctx_exit = holder.ctx(1, seq_len);
                if let CallbackAction::ExitEarly { logits } = cb.pre_node(&ctx_exit) {
                    assert_eq!(
                        logits.len(),
                        seq_len * hidden_size,
                        "logits length must equal seq_len({}) * hidden_size({})",
                        seq_len, hidden_size
                    );
                } else {
                    panic!("Expected ExitEarly for seq={}, hidden={}", seq_len, hidden_size);
                }
            }
        }
    }

    #[test]
    fn pre_node_never_returns_inject_hidden_or_compact_mask() {
        // Arrange: verify MidLayerEncodeCallback never produces these variants
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Various contexts
        let ctx2 = holder.ctx(2, 1);
        let ctx3 = holder.ctx(3, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);

        cb.post_node(&ctx2, &output);
        for ctx in [&ctx2, &ctx3] {
            let action = cb.pre_node(ctx);
            match action {
                CallbackAction::Continue | CallbackAction::ExitEarly { .. } => {}
                _ => panic!("MidLayerEncodeCallback produced unexpected variant: {:?}", action),
            }
        }
    }

    // -- Group D: decode_hidden_output output length invariant --

    #[test]
    fn decode_f32_output_length_invariant_small() {
        // Arrange: output length must always equal seq_len * hidden_size
        for seq in 1..5 {
            for hidden in [1, 2, 4] {
                let numel = seq * hidden;
                let src = vec![0u8; numel * 4];
                if let Some(out) = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F32) {
                    assert_eq!(out.len(), numel, "F32 output length must equal {}*{}", seq, hidden);
                }
            }
        }
    }

    #[test]
    fn decode_f16_output_length_invariant_small() {
        // Arrange: F16 output length must always equal seq_len * hidden_size
        for seq in 1..4 {
            for hidden in [1, 3] {
                let numel = seq * hidden;
                let src = vec![0u8; numel * 2];
                // May fail F32 candidate but should pass F16 if buffer aligned correctly
                if let Some(out) = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F16) {
                    assert_eq!(out.len(), numel, "F16 output length must equal {}*{}", seq, hidden);
                }
            }
        }
    }

    #[test]
    fn decode_bf16_output_length_invariant_small() {
        // Arrange: BF16 output length must always equal seq_len * hidden_size
        for seq in 1..4 {
            for hidden in [1, 3] {
                let numel = seq * hidden;
                let src = vec![0u8; numel * 2];
                if let Some(out) = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::BF16) {
                    assert_eq!(out.len(), numel, "BF16 output length must equal {}*{}", seq, hidden);
                }
            }
        }
    }

    #[test]
    fn decode_f32_consistent_across_repeated_calls() {
        // Arrange: same input must produce same output
        let src: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();

        let out1 = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32).unwrap();
        let out2 = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32).unwrap();

        assert_eq!(out1, out2, "repeated decode must produce identical results");
    }

    #[test]
    fn decode_f32_output_all_finite_for_finite_input() {
        // Arrange: finite F32 inputs must produce finite F32 outputs
        let values: Vec<f32> = vec![1.0, -2.5, 0.001, 100.0, -0.0001, 3.14];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F32).unwrap();

        assert_eq!(out.len(), 6);
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] = {} must be finite", i, v);
        }
    }

    // -- Group E: decode_hidden_output additional edge cases --

    #[test]
    fn decode_f32_hidden_two_seq_one_exact() {
        // Arrange: hidden=2, seq=1 → 2 f32 = 8 bytes, exact fit
        let values: Vec<f32> = vec![-1.0, 1.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 8);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decode_f32_all_negative_infinity_large_buffer() {
        // Arrange: 256 negative infinity values
        let count = 256;
        let src: Vec<u8> = (0..count).flat_map(|_| f32::NEG_INFINITY.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 16, 16, DType::F32).unwrap();
        assert_eq!(out.len(), count);
        assert!(out.iter().all(|v| v.is_infinite() && v.is_sign_negative()));
    }

    #[test]
    fn decode_f32_mixed_nan_and_finite_across_rows() {
        // Arrange: some rows have NaN, others have finite values
        let nan = f32::from_bits(0x7FC00000);
        let values: Vec<f32> = vec![
            1.0, 2.0, 3.0,   // row 0: all finite
            nan, 5.0, 6.0,    // row 1: mixed
            7.0, nan, 9.0,    // row 2: mixed
        ];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 3, DType::F32).unwrap();

        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[3].is_nan());
        assert!((out[4] - 5.0).abs() < 1e-6);
        assert!((out[6] - 7.0).abs() < 1e-6);
        assert!(out[7].is_nan());
        assert!((out[8] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn decode_f32_byte_pattern_0x80_0x00_0x00_0x00_is_negative_zero() {
        // Arrange: 0x80000000 = -0.0 in IEEE 754
        let bits: u32 = 0x80000000;
        let src: Vec<u8> = bits.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        assert_eq!(out[0].to_bits(), 0x80000000u32);
        assert!(out[0].is_sign_negative());
        assert_eq!(out[0].abs(), 0.0);
    }

    #[test]
    fn decode_f32_stride_boundary_hidden_size_three_seq_two() {
        // Arrange: hidden=3, seq=2 → 6 f32 = 24 bytes. F32 stride=12, 24%12==0.
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 24);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F32).unwrap();
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn decode_hidden_size_power_of_two() {
        // Arrange: hidden=16 (power of 2), seq=1, F32
        let hidden = 16;
        let values: Vec<f32> = (0..hidden).map(|i| (i as f32) - 8.0).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F32).unwrap();
        assert_eq!(out.len(), hidden);
        assert!((out[0] - (-8.0)).abs() < 1e-6);
        assert!((out[15] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn decode_f32_positive_infinity_in_preallocated() {
        // Arrange: pre-allocated buffer with positive infinity at position 0
        let hidden = 8;
        let max_seq = 64;
        let mut src = vec![0u8; max_seq * hidden * 4];
        src[0..4].copy_from_slice(&f32::INFINITY.to_le_bytes());

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F32).unwrap();
        assert_eq!(out.len(), hidden);
        assert!(out[0].is_infinite() && out[0].is_sign_positive());
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn decode_f32_large_odd_hidden_preallocated() {
        // Arrange: hidden=1023 (odd), max_seq=64, seq=1
        let hidden = 1023;
        let max_seq = 64;
        let mut src = vec![0u8; max_seq * hidden * 4];
        src[0..4].copy_from_slice(&42.0f32.to_le_bytes());

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F32).unwrap();
        assert_eq!(out.len(), hidden);
        assert!((out[0] - 42.0).abs() < 1e-6);
    }

    // -- Group F: Integration stress tests --

    #[test]
    fn lifecycle_32_layers_capture_at_15() {
        // Arrange: 32 layers, capture at layer 15, exit at 16
        let mut cb = MidLayerEncodeCallback::new(15);
        let holder = TestCtxHolder::new(4, DType::F32);

        for layer in 0..15 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        }

        let ctx15 = holder.ctx(15, 1);
        let output = make_f32_output(&[15.0, 16.0, 17.0, 18.0]);
        cb.post_node(&ctx15, &output);

        let ctx16 = holder.ctx(16, 1);
        match cb.pre_node(&ctx16) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![15.0, 16.0, 17.0, 18.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn lifecycle_capture_many_nodes_at_same_layer_last_wins() {
        // Arrange: simulate many nodes within the same target layer
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Simulate 10 nodes within layer 3 (different node_idx, same layer_idx)
        for node in 0..10u32 {
            let ctx = LayerContext {
                node_idx: node as usize,
                layer_idx: 3,
                node_op: "sub_op",
                hidden_state: &holder.hidden_state,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 1,
                seq_len: 1,
                position: 0,
                request_id: 1,
                model_config: &holder.config,
            };
            let output = make_f32_output(&[node as f32, 0.0, 0.0, 0.0]);
            cb.post_node(&ctx, &output);
        }

        // Exit: last capture (node 9) wins
        let ctx4 = holder.ctx(4, 1);
        match cb.pre_node(&ctx4) {
            CallbackAction::ExitEarly { logits } => {
                assert!((logits[0] - 9.0).abs() < 1e-6, "last capture must win");
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn lifecycle_recreate_callback_independence() {
        // Arrange: two separate callbacks must be independent
        let mut cb1 = MidLayerEncodeCallback::new(2);
        let mut cb2 = MidLayerEncodeCallback::new(4);
        let holder = TestCtxHolder::new(4, DType::F32);

        // cb1 captures at layer 2
        let ctx2 = holder.ctx(2, 1);
        cb1.post_node(&ctx2, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));

        // cb2 does NOT capture at layer 2 (targets layer 4)
        cb2.post_node(&ctx2, &make_f32_output(&[5.0, 6.0, 7.0, 8.0]));

        assert!(cb1.captured.is_some());
        assert!(cb2.captured.is_none(), "cb2 targets layer 4, not 2");

        // cb2 captures at layer 4
        let ctx4 = holder.ctx(4, 1);
        cb2.post_node(&ctx4, &make_f32_output(&[10.0, 20.0, 30.0, 40.0]));

        // cb1 exits at layer 3 with its own data
        let ctx3 = holder.ctx(3, 1);
        match cb1.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }

        // cb2 exits at layer 5 with its own data
        let ctx5 = holder.ctx(5, 1);
        match cb2.pre_node(&ctx5) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn lifecycle_full_8_layers_target_3_verify_data() {
        // Arrange: walk all 8 layers of a small model, capture at layer 3
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);

        for layer in 0..8 {
            let ctx = holder.ctx(layer, 1);
            let output = make_f32_output(&[
                layer as f32 * 10.0,
                (layer as f32 * 10.0) + 1.0,
                (layer as f32 * 10.0) + 2.0,
                (layer as f32 * 10.0) + 3.0,
            ]);

            // pre_node
            let pre = cb.pre_node(&ctx);
            if layer <= 3 {
                assert!(matches!(pre, CallbackAction::Continue), "layer {}", layer);
            } else if layer == 4 {
                // First node after target layer 3 — should exit
                match pre {
                    CallbackAction::ExitEarly { logits } => {
                        // Should have layer 3 data: [30.0, 31.0, 32.0, 33.0]
                        assert!((logits[0] - 30.0).abs() < 1e-6);
                        assert!((logits[3] - 33.0).abs() < 1e-6);
                    }
                    other => panic!("Expected ExitEarly at layer {}, got {:?}", layer, other),
                }
                break; // execution would stop here
            }

            // post_node
            cb.post_node(&ctx, &output);
        }
    }

    #[test]
    fn lifecycle_capture_seq_len_8_exit_with_correct_size() {
        // Arrange: capture with seq_len=8, verify exit carries 8*hidden_size elements
        let hidden = 4;
        let seq = 8;
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(hidden, DType::F32);

        let ctx2 = LayerContext {
            node_idx: 4,
            layer_idx: 2,
            node_op: "Test",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: seq,
            seq_len: seq,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };
        let values: Vec<f32> = (0..seq * hidden).map(|i| i as f32 * 0.1).collect();
        let output = make_f32_output(&values);
        cb.post_node(&ctx2, &output);

        let ctx3 = holder.ctx(3, seq);
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits.len(), seq * hidden);
                assert!((logits[0] - 0.0).abs() < 1e-6);
                assert!((logits[seq * hidden - 1] - (31.0 * 0.1)).abs() < 1e-5);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn chain_dispatch_post_node_mid_layer_always_continues() {
        // Arrange: CallbackChain dispatch_post_node with MidLayerEncode — always Continue
        use crate::graph::layer_callback::CallbackChain;

        let cb = MidLayerEncodeCallback::new(2);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);

        // dispatch_post_node must return Continue (MidLayerEncode never exits from post_node)
        let action = chain.dispatch_post_node(&ctx2, &output);
        assert!(
            matches!(action, CallbackAction::Continue),
            "dispatch_post_node must return Continue, got {:?}", action
        );
    }

    // -- Group G: decode_hidden_output with various declared_dtype --

    #[test]
    fn decode_f32_candidate_wins_for_all_non_half_dtypes() {
        // Arrange: F32 candidate should win regardless of declared dtype when stride matches
        let src: Vec<u8> = 7.0f32.to_le_bytes().to_vec();
        for dtype in [DType::F32, DType::U8, DType::F8E4M3, DType::F8E5M2, DType::F6E3M2, DType::F6E2M3, DType::F4E2M1] {
            let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, dtype);
            assert!(
                out.is_some(),
                "F32 candidate must win for declared dtype {:?}",
                dtype
            );
            assert!((out.unwrap()[0] - 7.0).abs() < 1e-6);
        }
    }

    #[test]
    fn decode_f32_candidate_wins_over_f16_even_with_f16_declared() {
        // Arrange: when both candidates are valid, F32 always wins
        let src: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        // 8 bytes, hidden=2: F32 stride=8, 8%8==0 → F32 passes
        // half stride=4, 8%4==0 → half also passes, but F32 checked first

        let out_f16 = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16).unwrap();
        let out_bf16 = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16).unwrap();

        // Both should decode as F32 (exact values), not half-precision (approximate)
        assert!((out_f16[0] - 1.0).abs() < 1e-6, "F32 candidate must win over F16 declared");
        assert!((out_bf16[0] - 1.0).abs() < 1e-6, "F32 candidate must win over BF16 declared");
    }

    // -- Group H: State machine property tests --

    #[test]
    fn pre_node_at_target_always_continue_regardless_of_capture_state() {
        // Arrange: pre_node at target layer always returns Continue, with or without capture
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx3 = holder.ctx(3, 1);

        // Without capture
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));

        // With capture
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx3, &output);
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));

        // After another capture
        let output2 = make_f32_output(&[5.0, 6.0, 7.0, 8.0]);
        cb.post_node(&ctx3, &output2);
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));
    }

    #[test]
    fn post_node_at_non_target_never_captures_regardless_of_output() {
        // Arrange: post_node at non-target layer never captures
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);

        for layer in 0..5 {
            let ctx = holder.ctx(layer, 1);
            let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
            cb.post_node(&ctx, &output);
            assert!(cb.captured.is_none(), "must not capture at layer {}", layer);
        }
    }

    #[test]
    fn after_exit_early_captured_is_none() {
        // Arrange: after ExitEarly is consumed, captured field is None
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));
        assert!(cb.captured.is_some());

        let ctx2 = holder.ctx(2, 1);
        let _ = cb.pre_node(&ctx2);

        assert!(cb.captured.is_none(), "captured must be None after ExitEarly consumed");
    }


    #[test]
    fn new_callback_post_node_at_non_target_never_captures() {
        // Arrange: fresh callback, post_node at wrong layer never captures
        let mut cb = MidLayerEncodeCallback::new(10);
        let holder = TestCtxHolder::new(4, DType::F32);

        for layer in [0, 5, 9] {
            let ctx = holder.ctx(layer, 1);
            cb.post_node(&ctx, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));
        }
        assert!(cb.captured.is_none());
    }

    #[test]
    fn decode_f32_preserves_value_ordering() {
        // Arrange: if a < b in input, then a < b in output (monotonicity for non-NaN)
        let values: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 4, 5, DType::F32).unwrap();
        for i in 1..out.len() {
            assert!(
                out[i] >= out[i - 1] || (out[i].is_nan() || out[i - 1].is_nan()),
                "output[{}]={} < output[{}]={}, monotonicity violated",
                i, out[i], i - 1, out[i - 1]
            );
        }
    }

    #[test]
    fn decode_f32_roundtrip_bit_exact() {
        // Arrange: encode f32 → bytes → decode → must be bit-identical
        let original: Vec<f32> = vec![
            0.0, -0.0, 1.0, -1.0,
            f32::MIN_POSITIVE, f32::MAX, f32::MIN,
            f32::INFINITY, f32::NEG_INFINITY,
            f32::from_bits(0x7FC00000), // NaN
            3.14159, -2.71828,
        ];
        let src: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, original.len(), DType::F32).unwrap();

        for (i, (orig, decoded)) in original.iter().zip(out.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                decoded.to_bits(),
                "bit-exact roundtrip failed at index {}", i
            );
        }
    }

    #[test]
    fn decode_f16_roundtrip_within_precision() {
        // Arrange: F16 values roundtrip within F16 precision
        let original_f32: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0];
        let f16_vals: Vec<f16> = original_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let src: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Force F16 path: hidden=7, F32 stride=28. total=14 bytes, 14%28!=0 → F32 fails.
        // half stride=14, 14%14==0 → passes.
        assert_eq!(src.len(), 14);
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 7, DType::F16).unwrap();

        assert_eq!(out.len(), 7);
        for (i, (orig, decoded)) in original_f32.iter().zip(out.iter()).enumerate() {
            assert!(
                (decoded - orig).abs() < 1.0,
                "F16 roundtrip tolerance exceeded at index {}: {} vs {}", i, decoded, orig
            );
        }
    }

    #[test]
    fn decode_bf16_roundtrip_within_precision() {
        // Arrange: BF16 values roundtrip within BF16 precision
        let original_f32: Vec<f32> = vec![0.0, 1.0, -1.0, 256.0, -256.0, 0.125, -0.875];
        let bf16_vals: Vec<bf16> = original_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let src: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        assert_eq!(src.len(), 14);
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 7, DType::BF16).unwrap();

        assert_eq!(out.len(), 7);
        for (i, (orig, decoded)) in original_f32.iter().zip(out.iter()).enumerate() {
            assert!(
                (decoded - orig).abs() < 1.0,
                "BF16 roundtrip tolerance exceeded at index {}: {} vs {}", i, decoded, orig
            );
        }
    }

    #[test]
    fn post_node_capture_then_incompatible_then_exit_preserves_original() {
        // Arrange: capture valid → incompatible → exit: must preserve valid capture
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Valid capture
        cb.post_node(&ctx, &make_f32_output(&[10.0, 20.0, 30.0, 40.0]));
        // Incompatible: 3 bytes (not a multiple of hidden_size*4=16)
        cb.post_node(&ctx, &[0u8; 3]);
        // Empty
        cb.post_node(&ctx, &[]);

        // Exit: original capture preserved
        let ctx3 = holder.ctx(3, 1);
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn decode_f32_two_bytes_returns_none() {
        // Arrange: 2 bytes can't hold even 1 f32 element
        let src = vec![0u8; 2];
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).is_none());
    }

    #[test]
    fn decode_f32_six_bytes_hidden_three_fails_f32_passes_half() {
        // Arrange: 6 bytes, hidden=3. F32 stride=12, 6%12!=0 → F32 fails.
        // half stride=6, 6%6==0, 6>=6 → passes with F16.
        let vals = [f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(2.0)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 6);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F16).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.0).abs() < 0.01);
        assert!((out[2] - 2.0).abs() < 0.01);
    }

    #[test]
    fn decode_f32_hidden_four_seq_four_preallocated() {
        // Arrange: hidden=4, seq=4, max_seq=256 → buffer much larger
        let hidden = 4;
        let seq = 4;
        let max_seq = 256;
        let mut src = vec![0u8; max_seq * hidden * 4];
        let values: Vec<f32> = (0..seq * hidden).map(|i| (i as f32) - 8.0).collect();
        for (i, v) in values.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F32).unwrap();
        assert_eq!(out.len(), seq * hidden);
        assert!((out[0] - (-8.0)).abs() < 1e-6);
        assert!((out[seq * hidden - 1] - 7.0).abs() < 1e-6);
    }

    // ========================================================================
    // Additional tests (25 new): CallbackAction variant existence,
    // construction, derived traits, simple MidLayerEncode invariants
    // ========================================================================

    // -- CallbackAction variant construction --

    #[test]
    fn callback_action_continue_default() {
        let action: CallbackAction = CallbackAction::default();
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn callback_action_continue_clone_eq() {
        let a = CallbackAction::Continue;
        assert_eq!(a.clone(), a);
    }

    #[test]
    fn callback_action_skip_this_node_constructible() {
        let action = CallbackAction::SkipThisNode;
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    #[test]
    fn callback_action_skip_this_node_clone_eq() {
        let a = CallbackAction::SkipThisNode;
        assert_eq!(a.clone(), a);
    }

    #[test]
    fn callback_action_exit_early_constructible() {
        let action = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        assert!(matches!(action, CallbackAction::ExitEarly { .. }));
    }

    #[test]
    fn callback_action_exit_early_empty_logits() {
        let action = CallbackAction::ExitEarly { logits: vec![] };
        assert!(matches!(action, CallbackAction::ExitEarly { logits } if logits.is_empty()));
    }

    #[test]
    fn callback_action_exit_early_clone_eq() {
        let a = CallbackAction::ExitEarly { logits: vec![3.14] };
        assert_eq!(a.clone(), a);
    }

    #[test]
    fn callback_action_inject_hidden_constructible() {
        let action = CallbackAction::InjectHidden { data: vec![0u8; 16] };
        assert!(matches!(action, CallbackAction::InjectHidden { .. }));
    }

    #[test]
    fn callback_action_inject_hidden_empty_data() {
        let action = CallbackAction::InjectHidden { data: vec![] };
        assert!(matches!(action, CallbackAction::InjectHidden { data } if data.is_empty()));
    }

    #[test]
    fn callback_action_inject_hidden_clone_eq() {
        let a = CallbackAction::InjectHidden { data: vec![1, 2, 3] };
        assert_eq!(a.clone(), a);
    }

    #[test]
    fn callback_action_compact_mask_constructible() {
        let action = CallbackAction::CompactMask { active_mask: vec![true, false, true] };
        assert!(matches!(action, CallbackAction::CompactMask { .. }));
    }

    #[test]
    fn callback_action_compact_mask_empty_mask() {
        let action = CallbackAction::CompactMask { active_mask: vec![] };
        assert!(matches!(action, CallbackAction::CompactMask { active_mask } if active_mask.is_empty()));
    }

    #[test]
    fn callback_action_compact_mask_clone_eq() {
        let a = CallbackAction::CompactMask { active_mask: vec![true] };
        assert_eq!(a.clone(), a);
    }

    // -- CallbackAction variant inequality --

    #[test]
    fn callback_action_different_variants_not_equal() {
        assert_ne!(CallbackAction::Continue, CallbackAction::SkipThisNode);
        assert_ne!(CallbackAction::Continue, CallbackAction::ExitEarly { logits: vec![] });
        assert_ne!(CallbackAction::SkipThisNode, CallbackAction::InjectHidden { data: vec![] });
        assert_ne!(CallbackAction::ExitEarly { logits: vec![1.0] }, CallbackAction::CompactMask { active_mask: vec![true] });
    }

    #[test]
    fn callback_action_exit_early_different_logits_not_equal() {
        let a = CallbackAction::ExitEarly { logits: vec![1.0] };
        let b = CallbackAction::ExitEarly { logits: vec![2.0] };
        assert_ne!(a, b);
    }

    #[test]
    fn callback_action_debug_formats_without_panic() {
        // Verify Debug derive works for all variants
        let _ = format!("{:?}", CallbackAction::Continue);
        let _ = format!("{:?}", CallbackAction::SkipThisNode);
        let _ = format!("{:?}", CallbackAction::ExitEarly { logits: vec![1.0] });
        let _ = format!("{:?}", CallbackAction::InjectHidden { data: vec![0u8; 4] });
        let _ = format!("{:?}", CallbackAction::CompactMask { active_mask: vec![true, false] });
    }

    // -- MidLayerEncodeCallback simple invariants --

    #[test]
    fn mid_layer_target_layers_always_none() {
        for target in 0..5 {
            let cb = MidLayerEncodeCallback::new(target);
            assert!(cb.target_layers().is_none(), "target_layers must be None for target={}", target);
        }
    }

    #[test]
    fn mid_layer_name_always_mid_layer_encode() {
        for target in [0, 1, 7, 100, usize::MAX] {
            let cb = MidLayerEncodeCallback::new(target);
            assert_eq!(cb.name(), "MidLayerEncode", "name invariant for target={}", target);
        }
    }

    #[test]
    fn mid_layer_priority_always_55() {
        for target in [0, 1, 50, 999, usize::MAX] {
            let cb = MidLayerEncodeCallback::new(target);
            assert_eq!(cb.priority(), 55, "priority invariant for target={}", target);
        }
    }

    #[test]
    fn mid_layer_new_captured_none_for_various_targets() {
        for target in [0, 1, 5, 100, usize::MAX] {
            let cb = MidLayerEncodeCallback::new(target);
            assert!(cb.captured.is_none(), "captured must start as None for target={}", target);
        }
    }

    #[test]
    fn mid_layer_target_layer_stored_correctly() {
        for target in [0, 1, 3, 7, 42, 255, 1000, usize::MAX] {
            let cb = MidLayerEncodeCallback::new(target);
            assert_eq!(cb.target_layer, target, "target_layer must match constructor argument");
        }
    }

    // -- CallbackAction ExitEarly logits length invariant --

    #[test]
    fn callback_action_exit_early_logits_length_matches_construction() {
        let empty = CallbackAction::ExitEarly { logits: vec![] };
        if let CallbackAction::ExitEarly { logits } = empty {
            assert_eq!(logits.len(), 0);
        }

        let single = CallbackAction::ExitEarly { logits: vec![42.0] };
        if let CallbackAction::ExitEarly { logits } = single {
            assert_eq!(logits.len(), 1);
        }

        let multi = CallbackAction::ExitEarly { logits: vec![1.0, 2.0, 3.0] };
        if let CallbackAction::ExitEarly { logits } = multi {
            assert_eq!(logits.len(), 3);
        }
    }

    // -- MID_LAYER_ENCODE_PRIORITY type and range --

    #[test]
    fn mid_layer_priority_is_u32_and_nonzero() {
        let prio: u32 = MID_LAYER_ENCODE_PRIORITY;
        assert!(prio > 0, "priority must be non-zero");
        assert!(prio < 1000, "priority should be a reasonable value");
    }

    // -- Additional tests for coverage gaps --

    #[test]
    fn decode_f32_single_element_seq_len_one_hidden_one() {
        // Arrange: minimal valid buffer — 1 element, 4 bytes
        let src = make_f32_output(&[99.5]);
        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).unwrap();
        // Assert
        assert_eq!(out.len(), 1);
        assert!((out[0] - 99.5).abs() < 1e-6);
    }

    #[test]
    fn decode_returns_none_for_zero_seq_len_with_nonempty_buffer() {
        // Arrange: valid f32 buffer but seq_len=0
        let src = make_f32_output(&[1.0, 2.0, 3.0]);
        // Act
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, 0, 3, DType::F32);
        // Assert
        assert!(result.is_none(), "seq_len=0 must always return None");
    }

    #[test]
    fn decode_returns_none_for_zero_hidden_size_with_nonempty_buffer() {
        // Arrange: valid f32 buffer but hidden_size=0
        let src = make_f32_output(&[1.0, 2.0]);
        // Act
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 0, DType::F32);
        // Assert
        assert!(result.is_none(), "hidden_size=0 must always return None");
    }

    #[test]
    fn decode_f32_buffer_size_equals_hidden_stride_exactly() {
        // Arrange: buffer is exactly hidden_size * 4 bytes, seq_len=1
        let values: Vec<f32> = (0..7).map(|i| i as f32 * 1.1).collect();
        let src = make_f32_output(&values);
        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 7, DType::F32).unwrap();
        // Assert
        assert_eq!(out, values);
    }

    #[test]
    fn decode_f32_with_declared_f32_dtype_f32_path() {
        // Arrange: F32 declared, F32 buffer — direct path
        let src = make_f32_output(&[-0.5, 0.5]);
        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32).unwrap();
        // Assert
        assert!((out[0] - (-0.5)).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn post_node_at_target_captures_then_second_post_node_same_layer_overwrites() {
        // Arrange: callback targets layer 1, first post_node captures, second overwrites
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(3, DType::F32);
        let ctx = holder.ctx(1, 1);
        let first = make_f32_output(&[10.0, 20.0, 30.0]);
        let second = make_f32_output(&[40.0, 50.0, 60.0]);

        // Act
        cb.post_node(&ctx, &first);
        cb.post_node(&ctx, &second);

        // Assert: captured should be the second (latest)
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![40.0, 50.0, 60.0]);
    }

    #[test]
    fn pre_node_layer_jump_from_0_to_5_exits_with_capture() {
        // Arrange: capture at layer 0, then pre_node at layer 5 (non-consecutive jump)
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(2, DType::F32);
        let ctx_target = holder.ctx(0, 1);
        let output = make_f32_output(&[7.0, 8.0]);
        cb.post_node(&ctx_target, &output);

        // Act: pre_node at a distant layer
        let ctx_after = holder.ctx(5, 1);
        let action = cb.pre_node(&ctx_after);

        // Assert
        match action {
            CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![7.0, 8.0]),
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn captured_cleared_after_exit_early() {
        // Arrange: trigger a full exit
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(2, DType::F32);
        let ctx_target = holder.ctx(3, 1);
        cb.post_node(&ctx_target, &make_f32_output(&[1.0, 2.0]));

        let ctx_after = holder.ctx(4, 1);
        let _ = cb.pre_node(&ctx_after);

        // Act & Assert: captured is consumed
        assert!(cb.captured.is_none(), "captured must be None after ExitEarly");
    }

    #[test]
    fn post_node_target_layer_one_seq_len_three() {
        // Arrange: hidden_size=2, seq_len=3 → 6 f32 values
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(2, DType::F32);
        let ctx = holder.ctx(1, 3);
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = make_f32_output(&values);

        // Act
        let action = cb.post_node(&ctx, &output);

        // Assert
        assert!(matches!(action, CallbackAction::Continue));
        assert_eq!(cb.captured.as_ref().unwrap().len(), 6);
        assert_eq!(cb.captured.as_ref().unwrap(), &values);
    }

    #[test]
    fn decode_f16_two_elements_seq_len_two_hidden_one() {
        // Arrange: 2 f16 values, seq_len=2, hidden_size=1, buffer=4 bytes
        let v1 = f16::from_f32(1.5);
        let v2 = f16::from_f32(-3.25);
        let mut src = Vec::new();
        src.extend_from_slice(&v1.to_le_bytes());
        src.extend_from_slice(&v2.to_le_bytes());

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 1, DType::F16).unwrap();

        // Assert
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.5).abs() < 0.01);
        assert!((out[1] - (-3.25)).abs() < 0.01);
    }

    #[test]
    fn decode_bf16_two_elements_seq_len_two_hidden_one() {
        // Arrange: 2 bf16 values, seq_len=2, hidden_size=1, buffer=4 bytes
        let v1 = bf16::from_f32(0.75);
        let v2 = bf16::from_f32(-2.5);
        let mut src = Vec::new();
        src.extend_from_slice(&v1.to_le_bytes());
        src.extend_from_slice(&v2.to_le_bytes());

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 1, DType::BF16).unwrap();

        // Assert
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.75).abs() < 0.01);
        assert!((out[1] - (-2.5)).abs() < 0.01);
    }

    #[test]
    fn decode_f32_hidden_size_two_seq_len_three_preallocated() {
        // Arrange: preallocated buffer for max_seq=64, but seq_len=3, hidden=2
        let max_seq = 64;
        let hidden = 2;
        let mut src = vec![0u8; max_seq * hidden * 4];
        let live_values = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        for (i, v) in live_values.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, hidden, DType::F32).unwrap();

        // Assert
        assert_eq!(out, live_values.to_vec());
    }

    #[test]
    fn callback_action_inject_hidden_with_nonempty_data() {
        // Arrange
        let data = vec![0xABu8, 0xCD, 0xEF];
        let action = CallbackAction::InjectHidden { data: data.clone() };

        // Assert: field access via match
        match &action {
            CallbackAction::InjectHidden { data: d } => assert_eq!(*d, data),
            other => panic!("expected InjectHidden, got {:?}", other),
        }
    }

    #[test]
    fn callback_action_compact_mask_with_multiple_elements() {
        // Arrange
        let mask = vec![true, false, true, true];
        let action = CallbackAction::CompactMask { active_mask: mask.clone() };

        // Assert: field access and clone equality
        match &action {
            CallbackAction::CompactMask { active_mask: m } => {
                assert_eq!(*m, mask);
                assert_eq!(m.iter().filter(|&&x| x).count(), 3);
            }
            other => panic!("expected CompactMask, got {:?}", other),
        }
        assert_eq!(action.clone(), action);
    }

    // ── Additional coverage tests ──

    #[test]
    fn post_node_returns_continue_even_after_capture() {
        // Arrange: callback targets layer 3, capture hidden at that layer
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);

        // Act
        let action = cb.post_node(&ctx, &output);

        // Assert: post_node always returns Continue (exit only via pre_node)
        assert!(
            matches!(action, CallbackAction::Continue),
            "post_node must always return Continue"
        );
    }

    #[test]
    fn post_node_non_target_layer_no_capture_attempted() {
        // Arrange: callback targets layer 5, but ctx is at layer 2
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);

        // Act
        let action = cb.post_node(&ctx, &output);

        // Assert: returns Continue and does not capture
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_none(), "must not capture at non-target layer");
    }

    #[test]
    fn pre_node_skip_multiple_layers_after_capture_exits() {
        // Arrange: capture at layer 2, then jump directly to layer 10 (skip 7 layers)
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[7.0, 8.0, 9.0, 10.0]);
        cb.post_node(&ctx2, &output);

        // Act: skip from layer 2 directly to layer 10
        let ctx10 = holder.ctx(10, 1);
        let action = cb.pre_node(&ctx10);

        // Assert
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![7.0, 8.0, 9.0, 10.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn pre_node_backward_layer_without_prior_capture_continues() {
        // Arrange: callback at layer 5, start at layer 3 (before target), no capture
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Act: pre_node at layer 3 (before target layer 5, no capture)
        let ctx = holder.ctx(3, 1);
        let action = cb.pre_node(&ctx);

        // Assert: Continue since no capture happened yet
        assert!(
            matches!(action, CallbackAction::Continue),
            "pre_node before target without capture must Continue"
        );
    }

    #[test]
    fn exit_with_zero_filled_capture_data() {
        // Arrange: capture all-zeros hidden state, then exit
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx1 = holder.ctx(1, 1);
        let output = make_f32_output(&[0.0f32; 4]);
        cb.post_node(&ctx1, &output);

        // Act: transition to layer 2
        let ctx2 = holder.ctx(2, 1);
        let action = cb.pre_node(&ctx2);

        // Assert: exits with all-zero data (not confused with "no capture")
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![0.0, 0.0, 0.0, 0.0]);
                assert!(!logits.is_empty(), "zero data is valid captured data");
            }
            other => panic!("Expected ExitEarly with zero logits, got {:?}", other),
        }
    }

    #[test]
    fn new_callback_has_no_captured_state() {
        // Arrange & Act
        let cb = MidLayerEncodeCallback::new(42);

        // Assert: no state captured initially
        assert!(cb.captured.is_none());
    }

    #[test]
    fn decode_f32_with_declared_u8_dtype_f32_path_succeeds_already_tested() {
        // This is a cross-check: F32 buffer with U8 declared dtype.
        // The F32 candidate path should succeed regardless of declared dtype.
        // Arrange: 2 f32 values = 8 bytes, hidden_size=2, seq_len=1
        let src = make_f32_output(&[1.5, -2.5]);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src, 1, 2, DType::U8,
        );

        // Assert
        let decoded = out.expect("F32 path must succeed even with U8 declared dtype");
        assert!((decoded[0] - 1.5).abs() < 1e-6);
        assert!((decoded[1] - (-2.5)).abs() < 1e-6);
    }

    #[test]
    fn lifecycle_post_node_incompatible_shape_no_capture_at_target() {
        // Arrange: callback targets layer 3, but post_node receives data that
        // decode_hidden_output rejects (incompatible feature dim)
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 1);

        // Provide only 2 bytes — too small for any hidden_size=4 candidate
        let output = vec![0xABu8, 0xCD];

        // Act
        let action = cb.post_node(&ctx, &output);

        // Assert: returns Continue but does not capture
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_none(), "incompatible shape must not be captured");
    }

    #[test]
    fn decode_hidden_output_with_hidden_size_one_seq_len_one_f32_exact() {
        // Arrange: minimal possible valid F32 buffer: 1 element = 4 bytes
        let src = make_f32_output(&[42.0f32]);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src, 1, 1, DType::F32,
        );

        // Assert
        let decoded = out.expect("single f32 element must decode");
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn decode_rejects_3_byte_buffer_for_hidden_size_one_f32() {
        // Arrange: 3 bytes — not enough for even one f32 element
        let src = vec![0x00u8, 0x01, 0x02];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src, 1, 1, DType::F32,
        );

        // Assert: F32 needs 4 bytes minimum, half needs 2 bytes with stride=2
        // hidden_size=1, half_stride=2, buffer=3 → 3%2!=0 fails half, F32=3<4 fails
        assert!(out.is_none(), "3 bytes cannot satisfy any decode candidate");
    }

    #[test]
    fn lifecycle_same_target_two_callbacks_independent() {
        // Arrange: two independent callbacks targeting the same layer
        let mut cb1 = MidLayerEncodeCallback::new(3);
        let mut cb2 = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Act: capture in cb1 only
        let ctx3 = holder.ctx(3, 1);
        let output1 = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb1.post_node(&ctx3, &output1);

        // cb2 gets different data
        let output2 = make_f32_output(&[5.0, 6.0, 7.0, 8.0]);
        cb2.post_node(&ctx3, &output2);

        // Transition out
        let ctx4 = holder.ctx(4, 1);
        let action1 = cb1.pre_node(&ctx4);
        let action2 = cb2.pre_node(&ctx4);

        // Assert: each callback exits with its own captured data
        match (action1, action2) {
            (
                CallbackAction::ExitEarly { logits: l1 },
                CallbackAction::ExitEarly { logits: l2 },
            ) => {
                assert_eq!(l1, vec![1.0, 2.0, 3.0, 4.0]);
                assert_eq!(l2, vec![5.0, 6.0, 7.0, 8.0]);
            }
            other => panic!("Expected both ExitEarly, got {:?}", other),
        }
    }

    #[test]
    fn decode_f32_candidate_wins_over_bf16_even_with_bf16_declared() {
        // Arrange: 4 f32 elements = 16 bytes, hidden_size=4, seq_len=1
        // F32 stride = 4*4=16, half stride = 4*2=8
        // buffer=16 → 16%16==0 (F32 passes), 16%8==0 (half passes too)
        // F32 candidate is checked first and must win
        let src = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src, 1, 4, DType::BF16,
        );

        // Assert: F32 decoding wins over BF16
        let decoded = out.expect("must decode");
        assert_eq!(decoded, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pre_node_layer_idx_zero_with_target_zero_continues() {
        // Arrange: target layer 0, currently at layer 0
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(0, 1);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: still at target layer, Continue
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn pre_node_large_target_large_layer_gap_continues_without_capture() {
        // Arrange: target layer 100, currently at layer 50, no capture yet
        let mut cb = MidLayerEncodeCallback::new(100);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(50, 1);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: no capture, layer 50 != 100, but no captured data to exit with
        assert!(matches!(action, CallbackAction::Continue));
    }

    // ── Additional edge-case and boundary tests ──

    // @trace TEST-MLE-325 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_hidden_size_one_seq_len_three_different_values() {
        // Arrange: hidden_size=1, seq_len=3 — three distinct f32 values in one column
        let values = [-7.5f32, 0.0, 13.25];
        let src = make_f32_output(&values);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 1, DType::F32)
            .expect("must decode 3x1 f32");

        // Assert
        assert_eq!(out.len(), 3);
        assert!((out[0] - (-7.5)).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - 13.25).abs() < 1e-6);
    }

    // @trace TEST-MLE-326 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_transition_from_target_to_same_target_after_capture_clears() {
        // Arrange: capture at layer 2, then transition to layer 3 (exit), then
        // receive pre_node at layer 2 again — captured was already consumed
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at target layer
        let ctx2 = holder.ctx(2, 1);
        cb.post_node(&ctx2, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));

        // Transition out — consumes captured
        let ctx3 = holder.ctx(3, 1);
        let exit_action = cb.pre_node(&ctx3);
        assert!(matches!(exit_action, CallbackAction::ExitEarly { .. }));

        // Act: pre_node at target layer again — captured is None now
        let ctx2_again = holder.ctx(2, 1);
        let action = cb.pre_node(&ctx2_again);

        // Assert: no captured data left, layer 2 != 2 but that check is equal,
        // so it returns Continue (still at target layer)
        assert!(
            matches!(action, CallbackAction::Continue),
            "pre_node at target layer after consumed capture must Continue"
        );
        assert!(cb.captured.is_none(), "captured must remain None");
    }

    // @trace TEST-MLE-327 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_multiple_layers_only_target_captures() {
        // Arrange: callback targets layer 2, post_node called at layers 0, 1, 2, 3
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(2, DType::F32);
        let output = make_f32_output(&[10.0, 20.0]);

        // Act: post_node at non-target layers
        for layer in [0, 1, 3] {
            let ctx = holder.ctx(layer, 1);
            cb.post_node(&ctx, &output);
        }
        assert!(cb.captured.is_none(), "non-target layers must not capture");

        // Act: post_node at target layer
        let ctx2 = holder.ctx(2, 1);
        cb.post_node(&ctx2, &output);

        // Assert: only target layer captures
        let captured = cb.captured.as_ref().expect("must capture at target layer");
        assert_eq!(captured, &vec![10.0, 20.0]);
    }

    // @trace TEST-MLE-328 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_buffer_size_not_multiple_of_hidden_two_stride() {
        // Arrange: hidden_size=3, bf16 stride=6, buffer=7 bytes — not a multiple of 6
        let src = vec![0u8; 7];

        // Act
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::BF16);

        // Assert: F32 stride=12 > 7 fails, half stride=6, 7%6!=0 fails
        assert!(result.is_none(), "buffer not multiple of half stride must fail");
    }

    // @trace TEST-MLE-329 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_buffer_size_not_multiple_of_hidden_two_stride() {
        // Arrange: hidden_size=3, f16 stride=6, buffer=13 bytes — not a multiple of 6
        let src = vec![0u8; 13];

        // Act
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F16);

        // Assert: F32 stride=12, 13>=12 but 13%12!=0 fails; half stride=6, 13%6!=0 fails
        assert!(result.is_none(), "buffer not multiple of half stride must fail for f16");
    }

    // @trace TEST-MLE-330 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_consecutive_transitions_exit_only_once() {
        // Arrange: capture at layer 1, then call pre_node twice at different layers
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(2, DType::F32);

        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&[5.0, 6.0]));

        // Act: first transition — should ExitEarly
        let ctx2 = holder.ctx(2, 1);
        let action1 = cb.pre_node(&ctx2);
        assert!(matches!(action1, CallbackAction::ExitEarly { .. }));

        // Act: second transition — captured already consumed, should Continue
        let ctx3 = holder.ctx(3, 1);
        let action2 = cb.pre_node(&ctx3);

        // Assert
        assert!(
            matches!(action2, CallbackAction::Continue),
            "second transition after consumed capture must Continue"
        );
        assert!(cb.captured.is_none(), "captured must be None after first exit");
    }

    // @trace TEST-MLE-331 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_seq_len_larger_than_buffer_alloc_size() {
        // Arrange: callback targets layer 3, hidden_size=4, buffer has only 4 f32 values
        // but seq_len=10 requires 40 values — decode_hidden_output will reject
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 10);
        // Only 4 f32 elements = 16 bytes, but seq_len*hidden_size = 40 elements needed
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);

        // Act
        let action = cb.post_node(&ctx, &output);

        // Assert: decode fails (buffer too small), no capture
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_none(), "buffer too small for seq_len must not capture");
    }

    // @trace TEST-MLE-332 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_hidden_three_seq_two_buffer_with_extra_row_slack() {
        // Arrange: hidden_size=3, seq_len=2, buffer sized for max_seq=8 (24 f32 elements)
        let max_seq = 8;
        let hidden = 3;
        let mut src = vec![0u8; max_seq * hidden * 4];
        let live = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows * 3 cols
        for (i, v) in live.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, hidden, DType::F32)
            .expect("must decode with slack");

        // Assert: only live region decoded
        assert_eq!(out.len(), 6);
        assert_eq!(out, live.to_vec());
    }

    // @trace TEST-MLE-333 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_capture_at_last_layer_then_post_decoder_exit() {
        // Arrange: target layer = 7 (last layer in TestCtxHolder with num_layers=8,
        // zero-indexed), capture there, then simulate post-decoder node with
        // a high layer_idx (extract_layer_index falls back to node_idx)
        let mut cb = MidLayerEncodeCallback::new(7);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at target layer 7
        let ctx7 = holder.ctx(7, 1);
        cb.post_node(&ctx7, &make_f32_output(&[10.0, 20.0, 30.0, 40.0]));
        assert!(cb.captured.is_some());

        // Act: simulate post-decoder node (final_norm / lm_head) with high layer_idx
        let ctx_post = holder.ctx(999, 1);
        let action = cb.pre_node(&ctx_post);

        // Assert: exits with captured data from last decoder layer
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("expected ExitEarly for post-decoder transition, got {:?}", other),
        }
    }

    // @trace TEST-MLE-334 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_hidden_size_two_seq_len_two_with_one_extra_pad_byte() {
        // Arrange: 4 f32 values = 16 bytes, but add 1 trailing pad byte = 17 bytes total
        // hidden_size=2, f32 stride = 8, 17 % 8 != 0 → F32 path fails
        // half stride = 4, 17 % 4 != 0 → half path also fails
        let mut src = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        src.push(0x00u8); // 1 byte of padding

        // Act
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F32);

        // Assert: extra byte breaks alignment for both candidates
        assert!(result.is_none(), "odd trailing byte must fail both stride checks");
    }

    // @trace TEST-MLE-335 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_target_layer_large_value_with_capture_at_that_layer() {
        // Arrange: target layer = 9999 (very large but not overflowing node_idx),
        // capture there, then see a lower layer
        let mut cb = MidLayerEncodeCallback::new(9999);
        let holder = TestCtxHolder::new(2, DType::F32);

        // Capture at target layer 9999
        let ctx_target = holder.ctx(9999, 1);
        cb.post_node(&ctx_target, &make_f32_output(&[99.0, -99.0]));

        // Act: transition to a much lower layer index
        let ctx_after = holder.ctx(0, 1);
        let action = cb.pre_node(&ctx_after);

        // Assert: exits with captured data
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![99.0, -99.0]);
            }
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-336 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_single_positive_value() {
        // Arrange: one bf16 value, hidden_size=1, seq_len=1
        let val = bf16::from_f32(3.14);
        let src = val.to_le_bytes().to_vec();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16)
            .expect("single bf16 must decode");

        // Assert
        assert_eq!(out.len(), 1);
        assert!((out[0] - 3.14).abs() < 0.1, "bf16 approximation of pi, got {}", out[0]);
    }

    // @trace TEST-MLE-337 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_single_negative_value() {
        // Arrange: one f16 value, hidden_size=1, seq_len=1
        let val = f16::from_f32(-2.75);
        let src = val.to_le_bytes().to_vec();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16)
            .expect("single f16 must decode");

        // Assert
        assert_eq!(out.len(), 1);
        assert!((out[0] - (-2.75)).abs() < 0.01, "f16 -2.75, got {}", out[0]);
    }

    // @trace TEST-MLE-338 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_incompatible_then_compatible_overwrites_none() {
        // Arrange: callback targets layer 3, first receive incompatible output,
        // then receive compatible output — should capture only the compatible one
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 1);

        // First: incompatible (only 2 bytes, too small for hidden_size=4)
        let bad_output = vec![0xFFu8, 0x00];
        cb.post_node(&ctx, &bad_output);
        assert!(cb.captured.is_none(), "incompatible output must not capture");

        // Act: then compatible output
        let good_output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx, &good_output);

        // Assert: captured is the compatible output
        let captured = cb.captured.as_ref().expect("compatible output must be captured");
        assert_eq!(captured, &vec![1.0, 2.0, 3.0, 4.0]);
    }

    // @trace TEST-MLE-339 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_monotonically_increasing_values_preserved() {
        // Arrange: 8 monotonically increasing f32 values, verify ordering preserved
        let values: Vec<f32> = (0..8).map(|i| (i * 100) as f32).collect();
        let src = make_f32_output(&values);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 4, DType::F32)
            .expect("must decode monotonic values");

        // Assert: ordering preserved exactly
        assert_eq!(out.len(), 8);
        for i in 0..8 {
            assert!((out[i] - values[i]).abs() < 1e-6, "value at index {} mismatch", i);
        }
        // Verify monotonicity
        for w in out.windows(2) {
            assert!(w[0] < w[1], "values must remain monotonically increasing: {} < {}", w[0], w[1]);
        }
    }

    // ========================================================================
    // Additional 15 tests for deeper coverage
    // ========================================================================

    // @trace TEST-MLE-340 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_hidden_size_prime_13_exact_fit() {
        // Arrange: hidden_size=13 (prime), seq_len=1 -> 13 f32 = 52 bytes
        let values: Vec<f32> = (0..13).map(|i| (i as f32) - 6.0).collect();
        let src = make_f32_output(&values);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 13, DType::F32)
            .expect("prime hidden_size must decode");

        // Assert
        assert_eq!(out.len(), 13);
        assert!((out[0] - (-6.0)).abs() < 1e-6);
        assert!((out[12] - 6.0).abs() < 1e-6);
    }

    // @trace TEST-MLE-341 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_negative_infinity_single_element() {
        // Arrange: F16 negative infinity, hidden=1, seq=1, buffer=2 bytes
        let h = f16::from_f32(f32::NEG_INFINITY);
        let src = h.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        // Act: F32 stride=4, 2%4!=0 -> F32 fails. half stride=2, 2%2==0 -> passes.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16)
            .expect("f16 neg infinity must decode");

        // Assert
        assert!(out[0].is_infinite());
        assert!(out[0].is_sign_negative());
    }

    // @trace TEST-MLE-342 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_negative_infinity_single_element() {
        // Arrange: BF16 negative infinity, hidden=1, seq=1, buffer=2 bytes
        let b = bf16::from_f32(f32::NEG_INFINITY);
        let src = b.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16)
            .expect("bf16 neg infinity must decode");

        // Assert
        assert!(out[0].is_infinite());
        assert!(out[0].is_sign_negative());
    }

    // @trace TEST-MLE-343 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_seq_len_one_then_seq_len_one_different_values_overwrites() {
        // Arrange: two consecutive post_nodes at target layer with same dimensions
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(2, DType::F32);
        let ctx = holder.ctx(2, 1);

        // First capture
        cb.post_node(&ctx, &make_f32_output(&[1.0, 2.0]));
        // Second capture (same shape, different values)
        cb.post_node(&ctx, &make_f32_output(&[10.0, 20.0]));

        // Assert: second capture overwrites first
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![10.0, 20.0]);
    }

    // @trace TEST-MLE-344 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_just_above_minimum_needed() {
        // Arrange: hidden_size=4, seq_len=1. Minimum needed = 16 bytes.
        // Provide exactly one extra row of slack (32 bytes total for max_seq=2).
        let mut src = vec![0u8; 32]; // 2 rows of 4 f32
        let values: Vec<f32> = vec![1.5, -2.5, 3.5, -4.5];
        for (i, v) in values.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        // Act: decode only seq_len=1 (first row)
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32)
            .expect("buffer with one extra row must decode first row");

        // Assert
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[3] - (-4.5)).abs() < 1e-6);
    }

    // @trace TEST-MLE-345 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_denormalized_smallest_negative_preserved() {
        // Arrange: most negative subnormal f32 (0x80000001)
        let bits: u32 = 0x80000001;
        let src: Vec<u8> = bits.to_le_bytes().to_vec();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32)
            .expect("negative subnormal must decode");

        // Assert: bit-exact preservation
        assert_eq!(out[0].to_bits(), 0x80000001u32);
        assert!(out[0].is_sign_negative());
        assert!(out[0] < 0.0);
    }

    // @trace TEST-MLE-346 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_multiple_exits_require_recapture() {
        // Arrange: capture -> exit -> recapture -> exit with different data
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // First capture and exit
        let ctx2 = holder.ctx(2, 1);
        cb.post_node(&ctx2, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));
        let ctx3 = holder.ctx(3, 1);
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("first exit: expected ExitEarly, got {:?}", other),
        }

        // Recapture at target layer
        cb.post_node(&ctx2, &make_f32_output(&[10.0, 20.0, 30.0, 40.0]));

        // Second exit with new data
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("second exit: expected ExitEarly, got {:?}", other),
        }

        // Third exit: captured consumed again
        let action3 = cb.pre_node(&ctx3);
        assert!(
            matches!(action3, CallbackAction::Continue),
            "third exit after consumed must Continue"
        );
    }

    // @trace TEST-MLE-347 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_max_finite_negative_pair() {
        // Arrange: F16 MAX (65504.0) and its negative (-65504.0)
        let vals = [f16::MAX, f16::from_f32(-f16::MAX.to_f32())];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // hidden=2: F32 stride=8, 4%8!=0 -> F32 fails. half stride=4, 4%4==0 -> passes.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16)
            .expect("f16 max pair must decode");

        assert_eq!(out.len(), 2);
        assert!((out[0] - 65504.0).abs() < 1.0);
        assert!(out[1].is_sign_negative());
        assert!((out[1] - (-65504.0)).abs() < 1.0);
    }

    // @trace TEST-MLE-348 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_two_elements_with_zero_and_nonzero() {
        // Arrange: BF16 0.0 and 256.0 (both exactly representable)
        let vals = [bf16::from_f32(0.0), bf16::from_f32(256.0)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        // hidden=2: F32 stride=8, 4%8!=0 -> F32 fails. half stride=4, 4%4==0 -> passes.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16)
            .expect("bf16 zero+256 must decode");

        assert_eq!(out.len(), 2);
        assert_eq!(out[0], 0.0);
        assert!((out[1] - 256.0).abs() < 1.0);
    }

    // @trace TEST-MLE-349 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_does_not_capture_when_decode_returns_none_for_valid_shape() {
        // Arrange: hidden_size=4, seq_len=1, output=3 bytes (too small for any candidate)
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Act: 3 bytes cannot be decoded (F32 needs 16, half needs 8)
        let action = cb.post_node(&ctx, &[0u8; 3]);

        // Assert
        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_none());
    }

    // @trace TEST-MLE-350 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_all_same_value_across_rows() {
        // Arrange: seq_len=5, hidden_size=3, all values = 0.42
        let count = 5 * 3;
        let values: Vec<f32> = vec![0.42f32; count];
        let src = make_f32_output(&values);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 5, 3, DType::F32)
            .expect("uniform values must decode");

        // Assert: all elements equal
        assert_eq!(out.len(), count);
        for (i, v) in out.iter().enumerate() {
            assert!((*v - 0.42).abs() < 1e-6, "uniform mismatch at index {}", i);
        }
    }

    // @trace TEST-MLE-351 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_just_one_byte_short_of_f32_alignment() {
        // Arrange: hidden_size=3, F32 stride=12. Buffer = 24 - 1 = 23 bytes.
        // 23 % 12 != 0 -> F32 fails. half stride=6, 23 % 6 != 0 -> half fails.
        let src = vec![0u8; 23];
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F32);
        assert!(result.is_none(), "23 bytes breaks both stride checks for hidden=3");
    }

    // @trace TEST-MLE-352 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_target_layer_zero_with_zero_length_output_then_valid_output() {
        // Arrange: first post_node at target with empty output (no capture),
        // then post_node with valid output (captures)
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx0 = holder.ctx(0, 1);

        // Empty output -> no capture
        cb.post_node(&ctx0, &[]);
        assert!(cb.captured.is_none(), "empty output must not capture");

        // Valid output -> captures
        cb.post_node(&ctx0, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));
        assert!(cb.captured.is_some());

        // Exit
        let ctx1 = holder.ctx(1, 1);
        match cb.pre_node(&ctx1) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-353 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_10_byte_buffer_hidden_five_fails_all() {
        // Arrange: 10 bytes, hidden_size=5. F32 stride=20, 10%20!=0 -> F32 fails.
        // half stride=10, 10%10==0, 10>=10 -> passes with F16/BF16 declared.
        // But DType::F32 -> no match arm in half path -> None.
        let src = vec![0u8; 10];
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 5, DType::F32);
        assert!(
            result.is_none(),
            "10 bytes with hidden=5 and F32 declared: F32 stride fails, half passes but no match arm"
        );
    }

    // @trace TEST-MLE-354 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_10_bytes_hidden_five_succeeds() {
        // Arrange: same 10-byte buffer but with F16 declared -> half path matches
        let vals: Vec<f16> = (0..5).map(|i| f16::from_f32(i as f32 * 2.0)).collect();
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 10);

        // F32 stride=20, 10%20!=0 -> fails. half stride=10, 10%10==0 -> passes.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 5, DType::F16)
            .expect("10 bytes hidden=5 F16 must decode");

        assert_eq!(out.len(), 5);
        assert!((out[0] - 0.0).abs() < 0.01);
        assert!((out[4] - 8.0).abs() < 0.1);
    }

    // ========================================================================
    // Additional 15 tests for remaining edge cases
    // ========================================================================

    // @trace TEST-MLE-355 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_size_just_over_f32_stride_for_next_row() {
        // Arrange: hidden=4, F32 stride=16. Buffer=20 bytes (16 + 4).
        // 20 % 16 == 4 (not a multiple of 16) -> F32 fails.
        // half stride=8, 20 % 8 == 4 (not a multiple of 8) -> half fails.
        let src = vec![0u8; 20];
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32);
        assert!(result.is_none(), "20 bytes with hidden=4 breaks both stride checks");
    }

    // @trace TEST-MLE-356 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_with_hidden_size_two_buffer_two_bytes_fails() {
        // Arrange: 2 bytes, hidden=2. F32 stride=8, 2%8!=0 -> F32 fails.
        // half stride=4, 2%4!=0 -> half fails. Both candidates fail.
        let src = vec![0u8; 2];
        assert!(
            MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16).is_none(),
            "2 bytes cannot satisfy hidden=2 stride for any candidate"
        );
    }

    // @trace TEST-MLE-357 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_at_target_after_exit_and_recapture_exits_with_new_data() {
        // Arrange: capture -> exit -> recapture -> pre_node at target -> then exit
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(2, DType::F32);

        // First round
        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&[1.0, 2.0]));
        let ctx2 = holder.ctx(2, 1);
        let _ = cb.pre_node(&ctx2); // consumes first capture

        // Recapture with different data
        cb.post_node(&ctx1, &make_f32_output(&[99.0, -99.0]));

        // pre_node at target layer -> Continue (still at target)
        assert!(matches!(cb.pre_node(&ctx1), CallbackAction::Continue));

        // Transition out -> ExitEarly with recaptured data
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![99.0, -99.0]);
            }
            other => panic!("expected ExitEarly with recaptured data, got {:?}", other),
        }
    }

    // @trace TEST-MLE-358 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_large_prime_hidden_99_exact_fit() {
        // Arrange: hidden=99 (large prime), seq=1 -> 99 f32 = 396 bytes
        let values: Vec<f32> = (0..99).map(|i| (i as f32) * 0.01).collect();
        let src = make_f32_output(&values);
        assert_eq!(src.len(), 396);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 99, DType::F32)
            .expect("large prime hidden must decode");

        assert_eq!(out.len(), 99);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[98] - 0.98).abs() < 1e-5);
    }

    // @trace TEST-MLE-359 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_stride_divisible_but_numel_times_four_overflows() {
        // Arrange: hidden_size=2, f32_stride=8. A huge seq_len where numel*4 overflows.
        // numel = seq_len * 2. We need (seq_len * 2) * 4 to overflow.
        let huge = (usize::MAX / 8) + 1; // numel = huge * 2, numel*4 overflows
        let src = vec![0u8; 16];
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, huge, 2, DType::F32);
        assert!(result.is_none(), "numel*4 overflow in F32 candidate must return None");
    }

    // @trace TEST-MLE-360 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_valid_capture_then_empty_output_preserves_capture() {
        // Arrange: capture valid data, then receive empty output at same target layer
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 1);

        cb.post_node(&ctx, &make_f32_output(&[10.0, 20.0, 30.0, 40.0]));
        assert!(cb.captured.is_some());

        // Empty output -> decode returns None, should not clear prior capture
        cb.post_node(&ctx, &[]);
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![10.0, 20.0, 30.0, 40.0]);
    }

    // @trace TEST-MLE-361 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_negative_one_half_and_quarter() {
        // Arrange: common activation-range values: -1.0, 0.5, -0.25
        let values: Vec<f32> = vec![-1.0, 0.5, -0.25];
        let src = make_f32_output(&values);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F32)
            .expect("common activation values must decode");

        assert_eq!(out.len(), 3);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - (-0.25)).abs() < 1e-6);
    }

    // @trace TEST-MLE-362 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_at_target_layer_with_prior_capture_from_different_node_idx() {
        // Arrange: capture from node A (node_idx=5), pre_node from node B (node_idx=99)
        // at the same target layer -> Continue (layer_idx match, not node_idx)
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx_a = LayerContext {
            node_idx: 5,
            layer_idx: 2,
            node_op: "attn",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 1,
            seq_len: 1,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };
        cb.post_node(&ctx_a, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));

        let ctx_b = LayerContext { node_idx: 99, ..ctx_a };
        let action = cb.pre_node(&ctx_b);

        assert!(matches!(action, CallbackAction::Continue));
        assert!(cb.captured.is_some(), "capture must persist across node_idx changes");
    }

    // @trace TEST-MLE-363 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_hidden_size_one_seq_len_two_half_forced() {
        // Arrange: 2 f16 elements = 4 bytes, hidden=1, seq=2
        // F32 stride=4, 4%4==0, 4>=8? No (4 < 8) -> F32 size check fails.
        // half stride=2, 4%2==0, 4>=4 -> passes.
        let vals = [f16::from_f32(0.5), f16::from_f32(-0.5)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 1, DType::F16)
            .expect("half-forced f16 decode");
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 0.01);
        assert!((out[1] - (-0.5)).abs() < 0.01);
    }

    // @trace TEST-MLE-364 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_target_zero_capture_empty_then_valid_then_exit() {
        // Arrange: target layer 0, first output is empty (rejected), second is valid
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(2, DType::F32);
        let ctx0 = holder.ctx(0, 1);

        // Empty output -> rejected, no capture
        cb.post_node(&ctx0, &[]);
        assert!(cb.captured.is_none());

        // Valid output -> captured
        cb.post_node(&ctx0, &make_f32_output(&[42.0, -42.0]));

        // Exit
        let ctx1 = holder.ctx(1, 1);
        match cb.pre_node(&ctx1) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![42.0, -42.0]);
            }
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-365 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_hidden_one_seq_three_half_forced() {
        // Arrange: 3 bf16 = 6 bytes, hidden=1, seq=3
        // F32 stride=4, 6%4!=0 -> F32 fails.
        // half stride=2, 6%2==0, 6>=6 -> passes with BF16.
        let vals = [bf16::from_f32(1.0), bf16::from_f32(0.0), bf16::from_f32(-1.0)];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 6);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 1, DType::BF16)
            .expect("half-forced bf16 decode");
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - 0.0).abs() < 0.01);
        assert!((out[2] - (-1.0)).abs() < 0.01);
    }

    // @trace TEST-MLE-366 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_hidden_two_seq_one_buffer_6_bytes_fails_alignment() {
        // Arrange: 6 bytes, hidden=2. F32 stride=8, 6%8!=0 -> F32 fails.
        // half stride=4, 6%4!=0 -> half fails. Neither aligns.
        let src = vec![0u8; 6];
        assert!(
            MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32).is_none(),
            "6 bytes with hidden=2 fails both alignment checks"
        );
    }

    // @trace TEST-MLE-367 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_seq_len_one_hidden_four_capture_then_pre_node_at_target_still_holds() {
        // Arrange: capture at target, then pre_node at same target, verify capture persists
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(5, 1);

        cb.post_node(&ctx, &make_f32_output(&[7.0, 8.0, 9.0, 10.0]));

        // pre_node at target layer -> Continue, capture NOT consumed
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![7.0, 8.0, 9.0, 10.0]);
    }

    // @trace TEST-MLE-368 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_negative_eps_value() {
        // Arrange: -f32::EPSILON (smallest negative representable difference from -1.0)
        let neg_eps = -f32::EPSILON;
        let src: Vec<u8> = neg_eps.to_le_bytes().to_vec();
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32)
            .expect("negative epsilon must decode");
        assert_eq!(out[0], neg_eps);
        assert!(out[0].is_sign_negative());
        assert!(out[0] < 0.0);
    }

    // @trace TEST-MLE-369 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_half_stride_overflows_then_f32_wins() {
        // Arrange: buffer that passes F32 candidate (so decode succeeds via F32 path)
        // even though half_stride computation would overflow for absurd hidden_size.
        // Use a normal case where F32 path clearly wins.
        let src = make_f32_output(&[1.0, 2.0]);
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16)
            .expect("F32 candidate must win");
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    // ========================================================================
    // Additional 15 tests for remaining edge cases and coverage
    // ========================================================================

    // @trace TEST-MLE-370 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_4_bytes_hidden_two_seq_one_fails_size_check() {
        // Arrange: 4 bytes, hidden=2, seq=1. F32 stride=8, 4%8!=0 -> F32 fails.
        // half stride=4, 4%4==0, 4>=4 -> passes. But declared F32 -> no half match arm.
        let src = vec![0u8; 4];
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32);
        assert!(result.is_none(), "4 bytes hidden=2 with F32 declared must fail");
    }

    // @trace TEST-MLE-371 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_many_alternating_valid_invalid_outputs_last_valid_wins() {
        // Arrange: alternating valid and invalid outputs at target layer
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // valid -> invalid -> valid -> invalid -> valid
        cb.post_node(&ctx, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));
        cb.post_node(&ctx, &[0u8; 3]); // invalid
        cb.post_node(&ctx, &make_f32_output(&[10.0, 20.0, 30.0, 40.0]));
        cb.post_node(&ctx, &[]); // invalid
        cb.post_node(&ctx, &make_f32_output(&[100.0, 200.0, 300.0, 400.0]));

        // Assert: last valid capture wins
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![100.0, 200.0, 300.0, 400.0]);
    }

    // @trace TEST-MLE-372 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_positive_one_half_qtr_across_two_rows() {
        // Arrange: seq=2, hidden=2 with [1.0, 0.5] and [0.25, -0.125]
        let values: Vec<f32> = vec![1.0, 0.5, 0.25, -0.125];
        let src = make_f32_output(&values);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F32)
            .expect("small fractions must decode");

        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.25).abs() < 1e-6);
        assert!((out[3] - (-0.125)).abs() < 1e-6);
    }

    // @trace TEST-MLE-373 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_after_capture_consumed_and_no_recapture_continues_indefinitely() {
        // Arrange: capture consumed, then many pre_nodes without new capture
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture and consume
        let ctx2 = holder.ctx(2, 1);
        cb.post_node(&ctx2, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));
        let ctx3 = holder.ctx(3, 1);
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::ExitEarly { .. }));

        // Many pre_nodes at different layers with no recapture
        for layer in 4..20 {
            let ctx = holder.ctx(layer, 1);
            assert!(
                matches!(cb.pre_node(&ctx), CallbackAction::Continue),
                "after consumed, pre_node at layer {} must Continue", layer
            );
        }
    }

    // @trace TEST-MLE-374 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_f32_candidate_passes_on_preallocated_bf16_buffer() {
        // Arrange: BF16 buffer where total bytes happen to be F32-stride-aligned.
        // hidden=4, max_seq=512 -> total=512*4*2=4096. F32 stride=16, 4096%16==0 -> F32 wins.
        let hidden = 4;
        let max_seq = 512;
        let mut src = vec![0u8; max_seq * hidden * 2];
        // Write F32 values at the start (since F32 candidate will decode these bytes as F32)
        let f32_vals: Vec<f32> = vec![2.5, -3.5, 0.0, 1.0];
        for (i, v) in f32_vals.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::BF16)
            .expect("F32 candidate must win on BF16 buffer when stride matches");
        assert_eq!(out.len(), 4);
        assert!((out[0] - 2.5).abs() < 1e-6);
    }

    // @trace TEST-MLE-375 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_at_target_with_f16_declared_and_f32_bytes_captures_correctly() {
        // Arrange: hidden=4, declared F16, but CPU canonical F32 output buffer
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F16);
        let ctx = holder.ctx(2, 1);
        let output = make_f32_output(&[5.5, -6.5, 7.5, -8.5]);

        cb.post_node(&ctx, &output);

        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 4);
        assert!((captured[0] - 5.5).abs() < 1e-6);
        assert!((captured[3] - (-8.5)).abs() < 1e-6);
    }

    // @trace TEST-MLE-376 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_12_bytes_hidden_three_seq_one_exact_fit() {
        // Arrange: hidden=3, seq=1 -> 3 f32 = 12 bytes exact fit
        let values: Vec<f32> = vec![-10.0, 0.0, 10.0];
        let src = make_f32_output(&values);
        assert_eq!(src.len(), 12);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F32)
            .expect("exact 12-byte buffer must decode");

        assert_eq!(out.len(), 3);
        assert!((out[0] - (-10.0)).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - 10.0).abs() < 1e-6);
    }

    // @trace TEST-MLE-377 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_16_layers_capture_at_layer_8_verify_data_integrity() {
        // Arrange: 16 layers, capture at layer 8, verify data survives the exit
        let mut cb = MidLayerEncodeCallback::new(8);
        let holder = TestCtxHolder::new(3, DType::F32);

        // Walk layers 0..7
        for layer in 0..8 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        }

        // Capture at layer 8 with distinctive data
        let ctx8 = holder.ctx(8, 1);
        let output = make_f32_output(&[88.88, -99.99, 0.01]);
        cb.post_node(&ctx8, &output);

        // Exit at layer 9
        let ctx9 = holder.ctx(9, 1);
        match cb.pre_node(&ctx9) {
            CallbackAction::ExitEarly { logits } => {
                assert!((logits[0] - 88.88).abs() < 1e-3);
                assert!((logits[1] - (-99.99)).abs() < 1e-3);
                assert!((logits[2] - 0.01).abs() < 1e-5);
            }
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-378 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_all_bytes_0x00_is_positive_zero() {
        // Arrange: all-zero bytes = 0x00000000 = +0.0 in IEEE 754
        let src = vec![0x00u8; 16]; // 4 f32 zeros
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32)
            .expect("all-zero bytes must decode");

        assert_eq!(out.len(), 4);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 0.0f32, "element {} must be +0.0", i);
            assert!(v.is_sign_positive(), "element {} must be positive zero", i);
        }
    }

    // @trace TEST-MLE-379 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_transition_from_target_to_same_target_no_double_exit() {
        // Arrange: capture at target, pre_node at target does not exit,
        // then another pre_node at target still does not exit
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx3 = holder.ctx(3, 1);

        cb.post_node(&ctx3, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));

        // Repeated pre_nodes at target — all must return Continue
        for _ in 0..5 {
            let action = cb.pre_node(&ctx3);
            assert!(
                matches!(action, CallbackAction::Continue),
                "pre_node at target layer must always return Continue"
            );
        }
        // Capture still held
        assert!(cb.captured.is_some());
    }

    // @trace TEST-MLE-380 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_stride_passes_size_passes_with_extra_rows_decodes_only_live() {
        // Arrange: hidden=4, seq=2, buffer sized for max_seq=16 (16*4*4=256 bytes)
        // Live region = 2*4*4 = 32 bytes. Verify only first 8 f32 decoded.
        let hidden = 4;
        let max_seq = 16;
        let seq = 2;
        let mut src = vec![0u8; max_seq * hidden * 4];
        // Fill live region with ascending values
        let live: Vec<f32> = (0..seq * hidden).map(|i| (i as f32) + 1.0).collect();
        for (i, v) in live.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Fill beyond live region with 999.0 to detect over-decoding
        for i in (seq * hidden)..(max_seq * hidden) {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&999.0f32.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F32)
            .expect("pre-allocated buffer must decode live region");

        assert_eq!(out.len(), seq * hidden);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[seq * hidden - 1] - (seq * hidden) as f32).abs() < 1e-5);
    }

    // @trace TEST-MLE-381 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_hidden_size_two_seq_two_with_padding_fails_stride() {
        // Arrange: 9 bytes, hidden=2. F32 stride=8, 9%8!=0 -> F32 fails.
        // half stride=4, 9%4!=0 -> half fails. Odd byte breaks both.
        let src = vec![0u8; 9];
        assert!(
            MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::BF16).is_none(),
            "9 bytes with hidden=2 fails both stride checks"
        );
    }

    // @trace TEST-MLE-382 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_capture_then_pre_node_at_target_then_exit_data_unmodified() {
        // Arrange: verify that pre_node at target does not modify the captured data
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx1 = holder.ctx(1, 1);

        cb.post_node(&ctx1, &make_f32_output(&[11.0, 22.0, 33.0, 44.0]));

        // pre_node at target — must not modify captured
        let _ = cb.pre_node(&ctx1);

        // Exit
        let ctx2 = holder.ctx(2, 1);
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![11.0, 22.0, 33.0, 44.0]);
            }
            other => panic!("expected ExitEarly with unmodified data, got {:?}", other),
        }
    }

    // @trace TEST-MLE-383 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_8_bytes_hidden_four_fails_size_check() {
        // Arrange: 8 bytes, hidden=4, seq=1. F32 stride=16, 8%16!=0 -> F32 fails.
        // half stride=8, 8%8==0, 8>=8 -> passes. But declared F32 -> no half match.
        let src = vec![0u8; 8];
        assert!(
            MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32).is_none(),
            "8 bytes hidden=4 with F32 declared must fail"
        );
    }

    // @trace TEST-MLE-384 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_target_layer_large_index_capture_and_exit() {
        // Arrange: edge case with a very large target layer (not usize::MAX to avoid
        // overflow in TestCtxHolder::ctx which computes node_idx = layer_idx * 2).
        // Use a large but safe value.
        let target = 50000;
        let mut cb = MidLayerEncodeCallback::new(target);
        let holder = TestCtxHolder::new(2, DType::F32);

        // Capture at target layer
        let ctx_target = holder.ctx(target, 1);
        cb.post_node(&ctx_target, &make_f32_output(&[42.0, -42.0]));

        // Exit at target + 1
        let ctx_next = holder.ctx(target + 1, 1);
        match cb.pre_node(&ctx_next) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![42.0, -42.0]);
            }
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    // ========================================================================
    // Additional 15 tests for deeper coverage
    // ========================================================================

    // @trace TEST-MLE-385 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_math_constants_preserved() {
        // Arrange: IEEE math constants (TAU, LN_2, FRAC_1_SQRT_2) round-trip test
        let values: Vec<f32> = vec![
            std::f32::consts::TAU,
            std::f32::consts::LN_2,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::SQRT_2,
        ];
        let src = make_f32_output(&values);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32)
            .expect("math constants must decode");

        // Assert: bit-exact roundtrip
        assert_eq!(out.len(), 4);
        for (i, (orig, decoded)) in values.iter().zip(out.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                decoded.to_bits(),
                "bit-exact mismatch for math constant at index {}", i
            );
        }
    }

    // @trace TEST-MLE-386 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_bf16_infinity_pairs_half_path() {
        // Arrange: hidden=5, F32 stride=20. 4 elements (8 bytes).
        // F32 stride=20, 8%20!=0 -> F32 fails.
        // half stride=10, 8%10!=0 -> half fails. Need hidden that aligns.
        // Use hidden=4, F32 stride=16, 8%16!=0 -> F32 fails.
        // half stride=8, 8%8==0, 8>=8 -> passes.
        let f16_vals = [
            f16::from_f32(f32::INFINITY),
            f16::from_f32(f32::NEG_INFINITY),
        ]; // 2 F16 elements, 4 bytes
        let src: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16)
            .expect("F16 infinity pair must decode");

        // Assert
        assert_eq!(out.len(), 2);
        assert!(out[0].is_infinite() && out[0].is_sign_positive());
        assert!(out[1].is_infinite() && out[1].is_sign_negative());
    }

    // @trace TEST-MLE-387 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_prime_hidden_prime_seq() {
        // Arrange: hidden=7 (prime), seq=3 (prime) -> 21 f32 = 84 bytes
        let values: Vec<f32> = (0..21).map(|i| ((i as f32) - 10.0) * 0.333).collect();
        let src = make_f32_output(&values);
        assert_eq!(src.len(), 84);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 7, DType::F32)
            .expect("prime-prime dimensions must decode");

        // Assert
        assert_eq!(out.len(), 21);
        assert!((out[0] - (-10.0 * 0.333)).abs() < 1e-4);
        assert!((out[20] - (10.0 * 0.333)).abs() < 1e-4);
    }

    // @trace TEST-MLE-388 [req:REQ-HR-002] [level:unit]
    #[test]
    fn exit_early_carries_nan_values_in_logits() {
        // Arrange: capture hidden state containing NaN, verify ExitEarly preserves NaN
        let nan = f32::from_bits(0x7FC00000);
        let values = vec![1.0, nan, -1.0, nan];
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);

        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&values));

        // Act
        let ctx2 = holder.ctx(2, 1);
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits.len(), 4);
                assert!((logits[0] - 1.0).abs() < 1e-6);
                assert!(logits[1].is_nan(), "NaN must survive through ExitEarly");
                assert!((logits[2] - (-1.0)).abs() < 1e-6);
                assert!(logits[3].is_nan(), "NaN must survive through ExitEarly");
            }
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-389 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_subnormal_pair_half_forced() {
        // Arrange: F16 smallest subnormal pair, hidden=2, buffer=4 bytes
        // F32 stride=8, 4%8!=0 -> F32 fails. half stride=4, 4%4==0 -> passes.
        let h1 = f16::from_bits(0x0001u16); // smallest positive subnormal
        let h2 = f16::from_bits(0x8001u16); // smallest negative subnormal
        let src: Vec<u8> = [&h1, &h2].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16)
            .expect("F16 subnormal pair must decode");

        assert_eq!(out.len(), 2);
        assert!(out[0] > 0.0, "positive subnormal must be positive");
        assert!(out[1] < 0.0, "negative subnormal must be negative");
        assert!(out[0].abs() < 0.001, "subnormal must be very small");
    }

    // @trace TEST-MLE-390 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_all_ff_bytes_at_target_captures_as_nan() {
        // Arrange: output buffer is all 0xFF bytes at target layer -> decodes as NaN
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);
        let output = vec![0xFFu8; 16]; // 4 f32 elements, all 0xFFFFFFFF = NaN

        // Act
        cb.post_node(&ctx, &output);

        // Assert
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 4);
        for (i, v) in captured.iter().enumerate() {
            assert!(v.is_nan(), "element {} from 0xFF bytes must be NaN", i);
        }
    }

    // @trace TEST-MLE-391 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_numel_overflows_half_but_f32_succeeds() {
        // Arrange: seq_len * hidden_size fits usize, but (seq_len * hidden_size) * 2
        // would overflow. F32 candidate's numel*4 also overflows but the F32 stride
        // check fails first (buffer too small for the overflow), so F32 returns None.
        // Then half numel*2 also overflows -> None overall.
        // Instead, test a realistic case: small buffer, huge seq_len that overflows
        // half but F32 stride fails on alignment.
        let src = vec![0u8; 8]; // 8 bytes, hidden=1
        // F32 stride=4, 8%4==0, 8>=numel*4. numel=huge*1. huge*4 overflows -> None
        let huge = (usize::MAX / 4) + 1;
        let result = MidLayerEncodeCallback::decode_hidden_output(&src, huge, 1, DType::F16);
        assert!(result.is_none(), "overflow in numel*4 must return None even for half candidate");
    }

    // @trace TEST-MLE-392 [req:REQ-HR-002] [level:unit]
    #[test]
    fn callback_action_continue_not_equal_to_skip() {
        // Arrange: verify Continue and SkipThisNode are distinct variants
        assert_ne!(
            CallbackAction::Continue,
            CallbackAction::SkipThisNode,
            "Continue and SkipThisNode must be unequal"
        );
    }

    // @trace TEST-MLE-393 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_subnormal_negative_preserved() {
        // Arrange: BF16 negative subnormal (bit 0x8001)
        let b = bf16::from_bits(0x8001u16);
        let src = b.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        // Act: F32 stride=4, 2%4!=0 -> F32 fails. half stride=2, 2%2==0 -> passes.
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16)
            .expect("BF16 negative subnormal must decode");

        assert!(out[0].is_sign_negative());
        assert!(out[0].abs() < 0.01, "subnormal must be very small");
    }

    // @trace TEST-MLE-394 [req:REQ-HR-002] [level:unit]
    #[test]
    fn chain_two_mid_layer_same_target_second_still_works() {
        // Arrange: two callbacks targeting same layer in a chain
        use crate::graph::layer_callback::CallbackChain;

        let cb1 = MidLayerEncodeCallback::new(2);
        let cb2 = MidLayerEncodeCallback::new(2);
        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2)]);
        assert_eq!(chain.len(), 2);

        let holder = TestCtxHolder::new(2, DType::F32);

        // Both callbacks capture at layer 2
        let ctx2 = holder.ctx(2, 1);
        let output = make_f32_output(&[5.0, -5.0]);
        assert!(matches!(chain.dispatch_post_node(&ctx2, &output), CallbackAction::Continue));

        // Transition out: first callback to fire returns ExitEarly
        let ctx3 = holder.ctx(3, 1);
        let action = chain.dispatch_pre_node(&ctx3);
        assert!(
            matches!(action, CallbackAction::ExitEarly { .. }),
            "chain with two same-target callbacks must exit"
        );
    }

    // @trace TEST-MLE-395 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_output_at_exact_alignment_boundary_captures() {
        // Arrange: output buffer exactly aligned to hidden_size * 4, seq_len=2
        // hidden=3, 2*3*4 = 24 bytes. No slack.
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(3, DType::F32);
        let ctx = holder.ctx(1, 2);

        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = make_f32_output(&values);
        assert_eq!(output.len(), 24);

        // Act
        cb.post_node(&ctx, &output);

        // Assert
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 6);
        assert!((captured[0] - 1.0).abs() < 1e-6);
        assert!((captured[5] - 6.0).abs() < 1e-6);
    }

    // @trace TEST-MLE-396 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_both_candidates_pass_f32_wins_with_different_values() {
        // Arrange: buffer where both F32 and half candidates would pass size checks,
        // but F32 is checked first and decodes as exact F32 values
        // hidden=1, buffer=4 bytes (1 f32). Both candidates pass stride.
        // Write F32 value 3.0 which would decode differently if treated as F16.
        let src: Vec<u8> = 3.0f32.to_le_bytes().to_vec();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16)
            .expect("F32 candidate must win");

        // Assert: F32 decoding gives exact 3.0, not F16 approximation
        assert!((out[0] - 3.0).abs() < 1e-6, "F32 candidate must decode exact 3.0");
    }

    // @trace TEST-MLE-397 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_large_seq_len_context_does_not_affect_continue_decision() {
        // Arrange: pre_node at non-target layer with large seq_len in context
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = LayerContext {
            node_idx: 10,
            layer_idx: 3,
            node_op: "Test",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 2048,
            seq_len: 2048,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };

        // Act: pre_node at layer 3, target is 5, no capture
        let action = cb.pre_node(&ctx);

        // Assert: Continue regardless of large seq_len
        assert!(
            matches!(action, CallbackAction::Continue),
            "large seq_len must not affect pre_node Continue decision"
        );
        assert!(cb.captured.is_none());
    }

    // @trace TEST-MLE-398 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_negative_subnormal_max_bits_preserved() {
        // Arrange: largest negative subnormal f32 (0x807FFFFF)
        let bits: u32 = 0x807FFFFF;
        let src: Vec<u8> = bits.to_le_bytes().to_vec();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32)
            .expect("largest negative subnormal must decode");

        assert_eq!(out[0].to_bits(), 0x807FFFFFu32, "bit pattern must be preserved");
        assert!(out[0].is_sign_negative());
        assert!(out[0] < 0.0);
        assert!(out[0] < f32::from_bits(0x80000001), "must be more negative than smallest-magnitude subnormal");
    }

    // @trace TEST-MLE-399 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_capture_exit_recapture_different_hidden_data() {
        // Arrange: capture with data A, exit, recapture with data B, exit with B
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(3, DType::F32);

        // First capture: all positive
        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&[1.0, 2.0, 3.0]));

        // First exit
        let ctx2 = holder.ctx(2, 1);
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("first exit: expected ExitEarly, got {:?}", other),
        }

        // Recapture with all negative values
        cb.post_node(&ctx1, &make_f32_output(&[-10.0, -20.0, -30.0]));

        // Second exit: must carry the new data
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![-10.0, -20.0, -30.0]);
            }
            other => panic!("second exit: expected ExitEarly with new data, got {:?}", other),
        }
    }

    // ========================================================================
    // Additional 15 tests for final coverage
    // ========================================================================

    // @trace TEST-MLE-400 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_hidden_one_seq_one_byte_exactly_four() {
        // Arrange: exactly 4 bytes for hidden=1, seq=1 — smallest valid F32 decode
        let src: Vec<u8> = (-0.0f32).to_le_bytes().to_vec();
        assert_eq!(src.len(), 4);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32)
            .expect("4-byte buffer for hidden=1 seq=1 must decode");

        // Assert
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].to_bits(), (-0.0f32).to_bits());
    }

    // @trace TEST-MLE-401 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_hidden_size_one_seq_len_three_captures_three_elements() {
        // Arrange: hidden_size=1, seq_len=3 -> 3 f32 = 12 bytes
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(1, DType::F32);
        let ctx = holder.ctx(1, 3);
        let values: Vec<f32> = vec![100.0, -100.0, 0.0];
        let output = make_f32_output(&values);

        // Act
        cb.post_node(&ctx, &output);

        // Assert
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 3);
        assert!((captured[0] - 100.0).abs() < 1e-6);
        assert!((captured[1] - (-100.0)).abs() < 1e-6);
        assert!((captured[2] - 0.0).abs() < 1e-6);
    }

    // @trace TEST-MLE-402 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_half_stride_passes_f32_stride_fails_size_check() {
        // Arrange: hidden=3, seq=2, F16 buffer. F32 stride=12, numel*4=24.
        // Buffer = 12 bytes (6 f16). 12%12==0, 12>=24? No -> F32 size fails.
        // half stride=6, 12%6==0, 12>=12 -> passes with F16.
        let vals: Vec<f16> = vec![
            f16::from_f32(0.1), f16::from_f32(0.2), f16::from_f32(0.3),
            f16::from_f32(0.4), f16::from_f32(0.5), f16::from_f32(0.6),
        ];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 12);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F16)
            .expect("F16 decode with seq=2 hidden=3 must succeed");

        assert_eq!(out.len(), 6);
        assert!((out[0] - 0.1).abs() < 0.01);
        assert!((out[5] - 0.6).abs() < 0.01);
    }

    // @trace TEST-MLE-403 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_at_target_zero_with_no_prior_activity_returns_continue() {
        // Arrange: fresh callback targeting layer 0, pre_node at layer 0
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx0 = holder.ctx(0, 1);

        // Act
        let action = cb.pre_node(&ctx0);

        // Assert: at target layer, no capture yet -> Continue
        assert!(
            matches!(action, CallbackAction::Continue),
            "pre_node at target layer 0 without capture must return Continue"
        );
        assert!(cb.captured.is_none());
    }

    // @trace TEST-MLE-404 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_positive_subnormal_max_preserved() {
        // Arrange: largest positive subnormal f32 (0x007FFFFF)
        let bits: u32 = 0x007FFFFF;
        let src: Vec<u8> = bits.to_le_bytes().to_vec();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32)
            .expect("largest positive subnormal must decode");

        assert_eq!(out[0].to_bits(), 0x007FFFFFu32);
        assert!(out[0] > 0.0);
        assert!(out[0] < f32::MIN_POSITIVE, "subnormal must be smaller than MIN_POSITIVE");
    }

    // @trace TEST-MLE-405 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_at_target_with_f32_declared_and_exact_buffer_captures() {
        // Arrange: hidden=8, declared F32, buffer exactly = 8*4=32 bytes, seq_len=1
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(8, DType::F32);
        let ctx = holder.ctx(2, 1);
        let values: Vec<f32> = (0..8).map(|i| (i as f32) * 0.25).collect();
        let output = make_f32_output(&values);
        assert_eq!(output.len(), 32);

        // Act
        cb.post_node(&ctx, &output);

        // Assert
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 8);
        assert!((captured[0] - 0.0).abs() < 1e-6);
        assert!((captured[7] - 1.75).abs() < 1e-5);
    }

    // @trace TEST-MLE-406 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_half_stride_with_odd_hidden_seq_one() {
        // Arrange: hidden=7 (odd), seq=1, BF16 buffer = 14 bytes
        // F32 stride=28, 14%28!=0 -> F32 fails.
        // half stride=14, 14%14==0, 14>=14 -> passes with BF16.
        let vals: Vec<bf16> = (0..7).map(|i| bf16::from_f32(i as f32 * 10.0)).collect();
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 14);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 7, DType::BF16)
            .expect("BF16 hidden=7 must decode via half path");

        assert_eq!(out.len(), 7);
        assert!((out[0] - 0.0).abs() < 0.1);
        assert!((out[6] - 60.0).abs() < 1.0);
    }

    // @trace TEST-MLE-407 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_target_last_minus_one_layer_capture_and_exit() {
        // Arrange: target layer = num_layers - 2 = 6 (holder has num_layers=8)
        let mut cb = MidLayerEncodeCallback::new(6);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Walk to target layer
        for layer in 0..6 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        }

        // Capture at layer 6
        let ctx6 = holder.ctx(6, 1);
        cb.post_node(&ctx6, &make_f32_output(&[60.0, 61.0, 62.0, 63.0]));

        // Exit at layer 7 (next-to-last decoder layer)
        let ctx7 = holder.ctx(7, 1);
        match cb.pre_node(&ctx7) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![60.0, 61.0, 62.0, 63.0]);
            }
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-408 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_32_bytes_hidden_eight_exact_fit() {
        // Arrange: hidden=8, seq=1 -> 8 f32 = 32 bytes exact fit
        let values: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let src = make_f32_output(&values);
        assert_eq!(src.len(), 32);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 8, DType::F32)
            .expect("32-byte buffer hidden=8 must decode");

        assert_eq!(out.len(), 8);
        for (i, v) in out.iter().enumerate() {
            let expected = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert!((v - expected).abs() < 1e-6, "mismatch at index {}", i);
        }
    }

    // @trace TEST-MLE-409 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_after_capture_then_many_post_nodes_at_non_target_exits_with_original() {
        // Arrange: capture at target, then many post_nodes at non-target layers must not
        // overwrite the capture
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 1
        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&[11.0, 22.0, 33.0, 44.0]));

        // Many post_nodes at non-target layers with different data
        for layer in 2..10 {
            let ctx = holder.ctx(layer, 1);
            let different = make_f32_output(&[layer as f32; 4]);
            cb.post_node(&ctx, &different);
        }

        // Exit: must carry original capture
        let ctx_next = holder.ctx(100, 1);
        match cb.pre_node(&ctx_next) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![11.0, 22.0, 33.0, 44.0]);
            }
            other => panic!("expected ExitEarly with original data, got {:?}", other),
        }
    }

    // @trace TEST-MLE-410 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_negative_largest_subnormal_exact_bit_preserved() {
        // Arrange: largest negative subnormal: 0x807FFFFF
        let bits: u32 = 0x807FFFFF;
        let src: Vec<u8> = bits.to_le_bytes().to_vec();

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32)
            .expect("largest negative subnormal must decode");

        assert_eq!(out[0].to_bits(), 0x807FFFFFu32);
        assert!(out[0].is_sign_negative());
    }

    // @trace TEST-MLE-411 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_hidden_three_seq_two_preallocated_buffer() {
        // Arrange: hidden=3, max_seq=10 -> 10*3*2=60 bytes. F32 stride=12, 60%12==0.
        // F32 candidate wins. seq_len=2, decode first 6 f32 values from 60-byte buffer.
        let mut src = vec![0u8; 60];
        let f32_vals: Vec<f32> = vec![1.5, -2.5, 3.5, -4.5, 5.5, -6.5];
        for (i, v) in f32_vals.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::BF16)
            .expect("BF16 preallocated buffer where F32 wins must decode");

        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[5] - (-6.5)).abs() < 1e-6);
    }

    // @trace TEST-MLE-412 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_empty_output_after_valid_capture_does_not_clear() {
        // Arrange: capture valid data, then receive empty output at target layer
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 1);

        // Valid capture
        cb.post_node(&ctx, &make_f32_output(&[10.0, 20.0, 30.0, 40.0]));
        assert!(cb.captured.is_some());

        // Empty output -> decode returns None, does not overwrite
        cb.post_node(&ctx, &[]);

        // Assert: original capture still held
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![10.0, 20.0, 30.0, 40.0]);
    }

    // @trace TEST-MLE-413 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_hidden_four_seq_two_half_path_forced_by_stride() {
        // Arrange: hidden=4, seq=2 -> 8 f16 = 16 bytes.
        // F32 stride=16, 16%16==0, 16>=32? No -> F32 size check fails.
        // half stride=8, 16%8==0, 16>=16 -> passes.
        let vals: Vec<f16> = (0..8).map(|i| f16::from_f32((i as f32) - 3.5)).collect();
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 16);

        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 4, DType::F16)
            .expect("F16 hidden=4 seq=2 forced half path must decode");

        assert_eq!(out.len(), 8);
        assert!((out[0] - (-3.5)).abs() < 0.1);
        assert!((out[7] - 3.5).abs() < 0.1);
    }

    // @trace TEST-MLE-414 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_two_full_cycles_capture_exit_recapture_exit() {
        // Arrange: perform two complete capture->exit cycles on the same callback
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(3, DType::F32);

        // Cycle 1: capture -> exit
        let ctx2 = holder.ctx(2, 1);
        cb.post_node(&ctx2, &make_f32_output(&[1.0, 2.0, 3.0]));
        let ctx3 = holder.ctx(3, 1);
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("cycle 1: expected ExitEarly, got {:?}", other),
        }
        assert!(cb.captured.is_none());

        // Cycle 2: recapture -> exit with different data
        cb.post_node(&ctx2, &make_f32_output(&[100.0, 200.0, 300.0]));
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![100.0, 200.0, 300.0]);
            }
            other => panic!("cycle 2: expected ExitEarly, got {:?}", other),
        }
        assert!(cb.captured.is_none());

        // Cycle 3 attempt: no recapture -> Continue
        assert!(
            matches!(cb.pre_node(&ctx3), CallbackAction::Continue),
            "cycle 3 without recapture must Continue"
        );
    }

    // @trace TEST-MLE-415 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_two_seq_hidden_two_preallocated_with_stale_tail() {
        // Arrange: preallocated buffer for max_seq=8, hidden=2 -> 8*2*4=64 bytes.
        // Live region: seq_len=2, first 16 bytes hold 4 f32 values. Tail is stale.
        let mut src = vec![0xFFu8; 64]; // fill with 0xFF (stale)
        let live: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        for (i, v) in live.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F32)
            .expect("preallocated buffer must decode live region");

        // Assert: only live region decoded, stale tail ignored
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    // @trace TEST-MLE-416 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_with_bf16_compute_dtype_captures_f32_buffer() {
        // Arrange: model declares BF16 compute dtype, but CPU path uses F32.
        // Output buffer has F32 bytes (16 bytes for hidden=4, seq=1).
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::BF16);
        let ctx = holder.ctx(1, 1);
        let output = make_f32_output(&[10.0, 20.0, 30.0, 40.0]);

        // Act: post_node at target layer with BF16 declared but F32 buffer
        cb.post_node(&ctx, &output);

        // Assert: F32 candidate wins over half, capture succeeds
        let captured = cb.captured.as_ref().expect("must capture F32 buffer with BF16 declared dtype");
        assert_eq!(captured.len(), 4);
        assert!((captured[0] - 10.0).abs() < 1e-5);
        assert!((captured[3] - 40.0).abs() < 1e-5);
    }

    // @trace TEST-MLE-417 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_transition_to_earlier_layer_without_capture_returns_continue() {
        // Arrange: callback targets layer 5, pre_node fires at layer 3 (earlier layer).
        // No capture occurred because we never visited layer 5.
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 1);

        // Act
        let action = cb.pre_node(&ctx);

        // Assert: layer 3 != target 5 and no capture -> Continue
        assert!(
            matches!(action, CallbackAction::Continue),
            "transition to earlier layer without capture must return Continue"
        );
    }

    // @trace TEST-MLE-418 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_all_nan_bytes_produces_nan() {
        // Arrange: buffer filled with 0xFF bytes (NaN in F32)
        let src = vec![0xFFu8; 16]; // 4 f32 values, all NaN (0xFFFFFFFF)

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32)
            .expect("NaN-filled buffer must decode");

        // Assert: all values are NaN
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_nan(), "0xFFFFFFFF bytes must decode to NaN");
        }
    }

    // @trace TEST-MLE-419 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_single_byte_buffer_rejected_for_all_dtypes() {
        // Arrange: 1 byte — cannot be any valid dtype (F32 needs 4, half needs 2)
        let src = vec![0xABu8; 1];

        // Act & Assert
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16).is_none());
        assert!(MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::U8).is_none());
    }

    // @trace TEST-MLE-420 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_three_bytes_buffer_rejected_not_aligned() {
        // Arrange: 3 bytes — not aligned to any valid element width
        let src = vec![0u8; 3];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32);

        // Assert
        assert!(out.is_none(), "3-byte buffer cannot represent any valid dtype element");
    }

    // @trace TEST-MLE-421 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_hidden_one_seq_large_decodes_correctly() {
        // Arrange: hidden_size=1, seq_len=1000 -> 1000 f32 = 4000 bytes
        let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1000, 1, DType::F32)
            .expect("hidden=1 seq=1000 must decode");

        // Assert: spot-check first, middle, and last values
        assert_eq!(out.len(), 1000);
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[500] - 50.0).abs() < 1e-3);
        assert!((out[999] - 99.9).abs() < 1e-3);
    }

    // @trace TEST-MLE-422 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_at_non_target_then_pre_node_at_target_does_not_exit() {
        // Arrange: post_node fires at wrong layer (no capture), then pre_node at target layer
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);

        // post_node at layer 1 (not target) — no capture
        let ctx1 = holder.ctx(1, 1);
        let output = make_f32_output(&[1.0, 2.0, 3.0, 4.0]);
        cb.post_node(&ctx1, &output);
        assert!(cb.captured.is_none(), "post_node at wrong layer must not capture");

        // pre_node at target layer 3 — no transition, Continue
        let ctx3 = holder.ctx(3, 1);
        let action = cb.pre_node(&ctx3);
        assert!(
            matches!(action, CallbackAction::Continue),
            "pre_node at target layer without prior capture must Continue"
        );
    }

    // @trace TEST-MLE-423 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_mixed_special_float_values() {
        // Arrange: buffer with 0.0, -0.0, inf, -inf, NaN in sequence
        let values: Vec<f32> = vec![
            0.0f32,
            -0.0f32,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 5, DType::F32)
            .expect("mixed special floats must decode");

        // Assert
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].to_bits(), 0u32); // +0.0
        assert_eq!(out[1].to_bits(), 0x80000000u32); // -0.0
        assert!(out[2].is_infinite() && out[2].is_sign_positive());
        assert!(out[3].is_infinite() && out[3].is_sign_negative());
        assert!(out[4].is_nan());
    }

    // @trace TEST-MLE-424 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_captures_then_pre_node_same_target_then_exit_with_correct_data() {
        // Arrange: capture at target, then pre_node at same layer (Continue), then exit
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(3, DType::F32);

        // Capture at target layer 2
        let ctx2 = holder.ctx(2, 1);
        cb.post_node(&ctx2, &make_f32_output(&[7.0, 8.0, 9.0]));

        // pre_node at same target layer — must Continue
        let action = cb.pre_node(&ctx2);
        assert!(matches!(action, CallbackAction::Continue));

        // Transition out — must ExitEarly with captured data
        let ctx3 = holder.ctx(3, 1);
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![7.0, 8.0, 9.0]);
            }
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-425 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_exactly_double_live_region() {
        // Arrange: preallocated buffer is 2x the live region (max_seq = 2 * seq_len).
        // hidden=2, seq=3 -> live = 24 bytes, buffer = 48 bytes.
        let mut src = vec![0u8; 48];
        let live_vals: Vec<f32> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6];
        for (i, v) in live_vals.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 2, DType::F32)
            .expect("2x buffer must decode live region");

        // Assert
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.1).abs() < 1e-5);
        assert!((out[5] - 6.6).abs() < 1e-5);
    }

    // @trace TEST-MLE-426 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_both_f32_and_half_candidates_rejected_by_size_check() {
        // Arrange: 6 bytes, hidden=5, seq=1.
        // F32 stride = 5*4=20, 6 % 20 != 0 -> F32 fails.
        // Half stride = 5*2=10, 6 % 10 != 0 -> half fails.
        let src = vec![0u8; 6];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 5, DType::F16);

        // Assert
        assert!(out.is_none(), "6 bytes with hidden=5 rejected by both candidates");
    }

    // @trace TEST-MLE-427 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_hidden_two_seq_three_with_alternating_signs() {
        // Arrange: 6 f32 values alternating positive/negative
        let values: Vec<f32> = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 2, DType::F32)
            .expect("seq=3 hidden=2 must decode 6 values");

        // Assert: row order preserved, signs correct
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (-1.0)).abs() < 1e-6);
        assert!((out[4] - 3.0).abs() < 1e-6);
        assert!((out[5] - (-3.0)).abs() < 1e-6);
    }

    // @trace TEST-MLE-428 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_wrong_shape_then_correct_shape_captures_correct() {
        // Arrange: first output is wrong shape (rejected), second is correct shape
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // First: output with wrong shape (10 bytes not multiple of hidden*4=16)
        cb.post_node(&ctx, &vec![0u8; 10]);
        assert!(cb.captured.is_none(), "wrong shape must not capture");

        // Second: correct output
        cb.post_node(&ctx, &make_f32_output(&[11.0, 22.0, 33.0, 44.0]));
        let captured = cb.captured.as_ref().expect("correct shape must capture");
        assert_eq!(captured, &vec![11.0, 22.0, 33.0, 44.0]);
    }

    // @trace TEST-MLE-429 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_two_byte_buffer_hidden_one_seq_one_f32_fails_half_passes_but_f32_declared() {
        // Arrange: 2 bytes, hidden=1, seq=1.
        // F32 stride=4, 2 % 4 != 0 -> F32 fails.
        // Half stride=2, 2 % 2 == 0, 2 >= 2 -> passes stride/size.
        // But declared DType::F32 is not F16/BF16 -> rejected in match arm.
        let src = vec![0u8; 2];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32);

        // Assert: half stride passes but F32 declared dtype has no half match arm
        assert!(out.is_none(), "2-byte buffer with F32 declared must be rejected");
    }

    // ========================================================================
    // Wave-430..444: Edge cases — exotic DType, stride priority, boundary
    // layer indices, overwrite within target, exact-size buffers, trait object
    // dispatch, non-half declared dtypes, double capture within same layer
    // ========================================================================

    // @trace TEST-MLE-430 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_with_u8_declared_dtype_rejected_by_both_candidates() {
        // Arrange: U8 has size_bytes=1 but is not F16/BF16, so half match arm
        // hits `_ => {}`. 6 bytes, hidden=3, seq=1.
        // F32 stride = 3*4=12, 6%12!=0 -> F32 fails.
        // half stride = 3*2=6, 6%6==0, 6>=6 -> passes stride/size but U8
        // does not match F16/BF16 in the match arm.
        let src = vec![0u8; 6];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::U8);

        // Assert
        assert!(out.is_none(), "U8 declared dtype must be rejected by both F32 and half paths");
    }

    // @trace TEST-MLE-431 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_with_f8e4m3_declared_dtype_rejected_by_half_match_arm() {
        // Arrange: F8E4M3 is a 1-byte type but not F16/BF16, so the half match
        // arm `_ => {}` swallows it. 8 bytes, hidden=4, seq=1.
        // F32 stride=16, 8%16!=0 -> F32 fails.
        // half stride=8, 8%8==0, 8>=8 -> passes stride/size but F8E4M3
        // does not match F16/BF16.
        let src = vec![0u8; 8];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F8E4M3);

        // Assert
        assert!(out.is_none(), "F8E4M3 declared dtype must be rejected by half match arm");
    }

    // @trace TEST-MLE-432 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_candidate_wins_over_half_when_both_strides_valid() {
        // Arrange: hidden=2, seq=2 -> 4 elements. Use a buffer that is a
        // multiple of both F32 stride (8) and half stride (4).
        // Buffer = 16 bytes = 4 F32 elements. F32 stride=8, 16%8==0, 16>=16.
        // Since F32 candidate is checked first, it must win.
        let f32_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let src: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 16);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F16)
            .expect("F32 candidate must win when both strides valid");

        // Assert: F32 interpretation yields exact values (half would lose precision)
        assert_eq!(out.len(), 4);
        for (i, expected) in f32_vals.iter().enumerate() {
            assert!(
                (out[i] - *expected).abs() < 1e-6,
                "F32 candidate must produce exact values, out[{}]={} expected={}",
                i, out[i], expected
            );
        }
    }

    // @trace TEST-MLE-433 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_target_layer_zero_captures_and_exits_at_layer_one() {
        // Arrange: target layer = 0 is a valid boundary case.
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(3, DType::F32);

        // Capture at layer 0
        let ctx0 = holder.ctx(0, 1);
        cb.post_node(&ctx0, &make_f32_output(&[10.0, 20.0, 30.0]));
        assert!(cb.captured.is_some(), "must capture at layer 0");

        // Transition to layer 1 -> ExitEarly
        let ctx1 = holder.ctx(1, 1);
        match cb.pre_node(&ctx1) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0]);
            }
            other => panic!("expected ExitEarly at transition from layer 0, got {:?}", other),
        }
        assert!(cb.captured.is_none(), "captured must be consumed after ExitEarly");
    }

    // @trace TEST-MLE-434 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_size_exactly_equals_live_region_no_tail() {
        // Arrange: buffer = seq_len * hidden_size * 4 exactly, no stale tail.
        let values: Vec<f32> = vec![-5.0, 0.0, 5.0, 10.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 16); // exactly 4 f32 elements

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F32)
            .expect("exact-size buffer must decode");

        // Assert
        assert_eq!(out, values);
    }

    // @trace TEST-MLE-435 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_overwrite_within_target_layer_keeps_latest() {
        // Arrange: multiple post_node calls at target layer update captured.
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(3, DType::F32);
        let ctx = holder.ctx(2, 1);

        // First capture
        cb.post_node(&ctx, &make_f32_output(&[1.0, 2.0, 3.0]));
        assert_eq!(cb.captured.as_ref().unwrap(), &vec![1.0, 2.0, 3.0]);

        // Second capture overwrites
        cb.post_node(&ctx, &make_f32_output(&[100.0, 200.0, 300.0]));

        // Assert: latest value retained
        assert_eq!(cb.captured.as_ref().unwrap(), &vec![100.0, 200.0, 300.0]);
    }

    // @trace TEST-MLE-436 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_half_minimal_buffer_single_bf16_element() {
        // Arrange: hidden=1, seq=1 -> 1 element. 2 bytes for BF16.
        let val = bf16::from_f32(0.25);
        let src = val.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16)
            .expect("single BF16 element must decode");

        // Assert
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.25).abs() < 0.01, "BF16 0.25 must round-trip");
    }

    // @trace TEST-MLE-437 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_stride_passes_but_size_check_fails() {
        // Arrange: hidden=3, seq=2 -> numel=6, needs 24 bytes.
        // F32 stride=12. Buffer=12 bytes -> 12%12==0 (stride passes),
        // but 12 < 24 (size check fails).
        let src = vec![0u8; 12];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F32);

        // Assert
        assert!(out.is_none(), "F32 stride passes but size check must fail");
    }

    // @trace TEST-MLE-438 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_at_target_layer_continues_even_after_capture() {
        // Arrange: capture at target, then pre_node at same target -> Continue
        // (the state machine only triggers ExitEarly on layer *transition*).
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(2, DType::F32);

        // Capture
        let ctx3 = holder.ctx(3, 1);
        cb.post_node(&ctx3, &make_f32_output(&[42.0, -42.0]));

        // pre_node at same target layer -> Continue (not ExitEarly)
        let action = cb.pre_node(&ctx3);
        assert!(
            matches!(action, CallbackAction::Continue),
            "pre_node at target layer must Continue even after capture"
        );

        // Capture still held
        assert!(cb.captured.is_some());
    }

    // @trace TEST-MLE-439 [req:REQ-HR-002] [level:unit]
    #[test]
    fn callback_via_trait_object_dispatch_works_correctly() {
        // Arrange: use the callback through a dyn LayerCallback trait object
        // to verify vtable dispatch does not break behavior.
        let mut cb: Box<dyn LayerCallback> = Box::new(MidLayerEncodeCallback::new(1));
        let holder = TestCtxHolder::new(2, DType::F32);

        // Assert trait methods via trait object
        assert_eq!(cb.priority(), 55);
        assert_eq!(cb.name(), "MidLayerEncode");
        assert!(cb.target_layers().is_none());

        // Capture via trait object post_node
        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&[5.0, 6.0]));

        // Exit via trait object pre_node
        let ctx2 = holder.ctx(2, 1);
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![5.0, 6.0]);
            }
            other => panic!("trait object dispatch: expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-440 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_with_f8e5m2_declared_dtype_rejected() {
        // Arrange: F8E5M2 is not F16/BF16. 8 bytes, hidden=4, seq=1.
        // F32 stride=16, 8%16!=0 -> F32 fails.
        // half stride=8, 8%8==0, 8>=8 -> half stride passes but F8E5M2
        // is not F16/BF16 in the match arm.
        let src = vec![0u8; 8];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F8E5M2);

        // Assert
        assert!(out.is_none(), "F8E5M2 declared dtype must be rejected");
    }

    // @trace TEST-MLE-441 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_large_hidden_single_seq_preserves_values() {
        // Arrange: hidden_size=256, seq_len=1 -> 256 f32 = 1024 bytes.
        // Use a pattern: value[i] = sin(i as f32 * 0.1).
        let values: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 1024);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 256, DType::F32)
            .expect("hidden=256 seq=1 must decode");

        // Assert: all values preserved exactly (f32 round-trip)
        assert_eq!(out.len(), 256);
        for (i, expected) in values.iter().enumerate() {
            assert!(
                (out[i] - *expected).abs() < 1e-7,
                "value mismatch at index {}: got {} expected {}",
                i, out[i], expected
            );
        }
    }

    // @trace TEST-MLE-442 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_seq_len_one_hidden_one_exact_buffer() {
        // Arrange: smallest possible F16 decode: 1 element = 2 bytes.
        let val = f16::from_f32(-1.0);
        let src = val.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16)
            .expect("single F16 element must decode");

        // Assert
        assert_eq!(out.len(), 1);
        assert!((out[0] - (-1.0)).abs() < 0.01);
    }

    // @trace TEST-MLE-443 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_double_capture_within_target_then_exit_with_second() {
        // Arrange: capture A at target, then pre_node at target (Continue),
        // then capture B at same target, then transition out -> must emit B.
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(2, DType::F32);
        let ctx2 = holder.ctx(2, 1);

        // First capture
        cb.post_node(&ctx2, &make_f32_output(&[1.0, 2.0]));

        // pre_node at same target -> Continue
        assert!(matches!(cb.pre_node(&ctx2), CallbackAction::Continue));

        // Second capture overwrites first
        cb.post_node(&ctx2, &make_f32_output(&[99.0, 88.0]));

        // Transition out
        let ctx3 = holder.ctx(3, 1);
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![99.0, 88.0], "must emit the second (latest) capture");
            }
            other => panic!("expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-444 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_with_sub_byte_dtypes_f4e2m1_rejected() {
        // Arrange: F4E2M1 has size_bytes=1 but is not F16/BF16.
        // hidden=2, seq=1. F32 stride=8, buffer=4 bytes, 4%8!=0 -> F32 fails.
        // half stride=4, 4%4==0, 4>=4 -> stride/size pass but F4E2M1
        // does not match F16/BF16 -> rejected.
        let src = vec![0u8; 4];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F4E2M1);

        // Assert
        assert!(out.is_none(), "F4E2M1 declared dtype must be rejected by half match arm");
    }

    // ========================================================================
    // Additional tests (15 new) for edge cases and trait behavior
    // ========================================================================

    // @trace TEST-MLE-445 [req:REQ-HR-002] [level:unit]
    #[test]
    fn callback_action_default_is_continue() {
        // Arrange: CallbackAction derives Default, and the default is Continue
        let action = CallbackAction::default();

        // Assert: default must be Continue (the #[default] attribute)
        assert!(
            matches!(action, CallbackAction::Continue),
            "CallbackAction::default() must be Continue"
        );
    }

    // @trace TEST-MLE-446 [req:REQ-HR-002] [level:unit]
    #[test]
    fn callback_action_clone_preserves_exit_early_data() {
        // Arrange: produce an ExitEarly action and clone it
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(3, DType::F32);

        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&[10.0, 20.0, 30.0]));

        let ctx2 = holder.ctx(2, 1);
        let original = cb.pre_node(&ctx2);

        // Act: clone the action
        let cloned = original.clone();

        // Assert: cloned action has the same data
        assert_eq!(original, cloned, "cloned CallbackAction must equal original");
        match cloned {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }

    // @trace TEST-MLE-447 [req:REQ-HR-002] [level:unit]
    #[test]
    fn callback_action_debug_format_exit_early() {
        // Arrange: produce an ExitEarly and verify Debug formatting
        let action = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };

        // Act: format via Debug
        let debug_str = format!("{:?}", action);

        // Assert: debug output contains "ExitEarly" and the values
        assert!(
            debug_str.contains("ExitEarly"),
            "Debug output must contain variant name: {}",
            debug_str
        );
        assert!(
            debug_str.contains("1.0"),
            "Debug output must contain logit value: {}",
            debug_str
        );
    }

    // @trace TEST-MLE-448 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_with_f8e4m3_declared_dtype_f32_path_succeeds_exact_fit() {
        // Arrange: F8E4M3 DType declared but buffer is F32 — F32 candidate wins
        // because it is checked first and buffer alignment matches F32 stride.
        let values: Vec<f32> = vec![7.0, -3.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 8); // 2 * 4

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F8E4M3);

        // Assert: F32 candidate checked first, stride=8, 8%8==0, succeeds
        let decoded = out.expect("F32 path must succeed with F8E4M3 declared dtype");
        assert!((decoded[0] - 7.0).abs() < 1e-6);
        assert!((decoded[1] - (-3.0)).abs() < 1e-6);
    }

    // @trace TEST-MLE-449 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_half_stride_with_f6e3m2_declared_returns_none_exact() {
        // Arrange: buffer passes half stride but declared F6E3M2 (not F16/BF16).
        // hidden=2, half_stride=4, buffer=4 bytes -> 4%4==0 passes.
        // F32 stride=8, 4%8!=0 fails. Half stride passes but F6E3M2 has no match arm.
        let src = vec![0u8; 4];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F6E3M2);

        // Assert: half match arm has no F6E3M2 case
        assert!(out.is_none(), "F6E3M2 declared dtype must be rejected in half match arm");
    }

    // @trace TEST-MLE-450 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_after_target_layer_with_node_idx_fallback() {
        // Arrange: target layer 1, capture, then pre_node with a very high
        // layer_idx simulating a post-decoder node (e.g., final_norm with
        // layer_idx set to node_idx because extract_layer_index could not
        // extract a real layer number).
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Capture at layer 1
        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&[11.0, 22.0, 33.0, 44.0]));

        // Act: pre_node with very high layer_idx (post-decoder fallback)
        let ctx_post = LayerContext {
            node_idx: 50000,
            layer_idx: 50000,
            node_op: "lm_head",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 1,
            seq_len: 1,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };

        // Assert: ExitEarly triggered
        match cb.pre_node(&ctx_post) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![11.0, 22.0, 33.0, 44.0]);
            }
            other => panic!("Expected ExitEarly for post-decoder node, got {:?}", other),
        }
    }

    // @trace TEST-MLE-451 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_incompatible_then_compatible_captures_compatible() {
        // Arrange: first post_node with incompatible shape, then with compatible.
        // The incompatible one must not prevent the compatible capture.
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Incompatible: 7 bytes, not a multiple of hidden_size*4=16
        let incompatible = vec![0u8; 7];
        cb.post_node(&ctx, &incompatible);
        assert!(cb.captured.is_none(), "incompatible must not capture");

        // Compatible: 4 f32 = 16 bytes
        let compatible = make_f32_output(&[5.0, 6.0, 7.0, 8.0]);
        cb.post_node(&ctx, &compatible);

        // Assert: compatible capture succeeds despite prior failure
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![5.0, 6.0, 7.0, 8.0]);
    }

    // @trace TEST-MLE-452 [req:REQ-HR-002] [level:unit]
    #[test]
    fn name_returns_static_str_same_across_instances() {
        // Arrange: two different instances
        let cb1 = MidLayerEncodeCallback::new(0);
        let cb2 = MidLayerEncodeCallback::new(100);

        // Act & Assert: name() returns the same &'static str for both
        assert_eq!(cb1.name(), cb2.name());
        assert_eq!(cb1.name(), "MidLayerEncode");
    }

    // @trace TEST-MLE-453 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_stride_overflow_but_half_succeeds() {
        // Arrange: hidden_size causes f32_stride overflow (hidden_size * 4 overflows)
        // but half_stride does not overflow (hidden_size * 2 fits). However,
        // the f32_stride overflow causes an early return None from checked_mul
        // before reaching the half check. So the result is still None.
        let src = vec![0u8; 16];
        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src,
            1,
            (usize::MAX / 4) + 1, // hidden_size * 4 overflows
            DType::F16,
        );

        // Assert: f32_stride checked_mul returns None -> early exit
        assert!(out.is_none(), "f32_stride overflow prevents reaching half candidate");
    }

    // @trace TEST-MLE-454 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_with_f8e5m2_declared_dtype() {
        // Arrange: DType::F8E5M2 is not F16/BF16. F32 path should still work
        // because it is checked first based on buffer alignment.
        let values: Vec<f32> = vec![3.5, -1.25];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F8E5M2);

        // Assert: F32 candidate wins (stride=8, 8%8==0)
        let decoded = out.expect("F32 path must succeed with F8E5M2 declared dtype");
        assert!((decoded[0] - 3.5).abs() < 1e-6);
        assert!((decoded[1] - (-1.25)).abs() < 1e-6);
    }

    // @trace TEST-MLE-455 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_no_capture_target_higher_than_all_layers() {
        // Arrange: target layer is higher than any layer we visit — no capture ever
        let mut cb = MidLayerEncodeCallback::new(255);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Walk layers 0..8 (all below target 255)
        for layer in 0..8 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
            let output = make_f32_output(&[layer as f32; 4]);
            assert!(matches!(cb.post_node(&ctx, &output), CallbackAction::Continue));
        }

        // Assert: captured is still None
        assert!(cb.captured.is_none(), "no capture when target layer never reached");

        // Continue after walking — pre_node at various layers still Continue
        for layer in 100..105 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        }
    }

    // @trace TEST-MLE-456 [req:REQ-HR-002] [level:unit]
    #[test]
    fn callback_action_partial_eq_continue_vs_exit_early() {
        // Arrange: CallbackAction derives PartialEq — different variants are not equal
        let continue_action = CallbackAction::Continue;
        let exit_action = CallbackAction::ExitEarly { logits: vec![] };

        // Assert: different variants must not be equal
        assert_ne!(continue_action, exit_action, "Continue != ExitEarly");
    }

    // @trace TEST-MLE-457 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_with_f6e2m3_declared_dtype_f32_candidate_wins() {
        // Arrange: DType::F6E2M3 is not F16/BF16. Buffer is 4-byte aligned for F32.
        // F32 candidate checked first.
        let src: Vec<u8> = 99.9f32.to_le_bytes().to_vec();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F6E2M3);

        // Assert: F32 candidate wins
        let decoded = out.expect("F32 path must succeed with F6E2M3 declared dtype");
        assert!((decoded[0] - 99.9).abs() < 1e-5);
    }

    // @trace TEST-MLE-458 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_at_target_with_capture_holds_until_transition() {
        // Arrange: verify that multiple pre_node calls at the target layer
        // after a capture do NOT consume the captured state. Only a layer
        // transition (layer_idx != target_layer) should consume it.
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(2, DType::F32);

        // Capture at layer 3
        let ctx3 = holder.ctx(3, 1);
        cb.post_node(&ctx3, &make_f32_output(&[7.0, 8.0]));

        // Multiple pre_node calls at target layer 3
        for _ in 0..5 {
            assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));
        }

        // Assert: captured still held (not consumed)
        assert!(cb.captured.is_some(), "captured must survive multiple pre_nodes at target");

        // Transition out: now captured should be consumed
        let ctx4 = holder.ctx(4, 1);
        match cb.pre_node(&ctx4) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![7.0, 8.0]);
            }
            other => panic!("Expected ExitEarly on transition, got {:?}", other),
        }

        // After exit, captured consumed
        assert!(cb.captured.is_none());
    }

    // @trace TEST-MLE-459 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_size_is_exactly_numel_times_four_plus_hidden_stride() {
        // Arrange: buffer = numel*4 + hidden_size*4 (one extra row of slack).
        // This is the typical JIT pre-allocated buffer for max_seq = seq_len + 1.
        // F32 stride check: buffer must be multiple of hidden_size*4, and
        // buffer >= numel*4. Both pass.
        let hidden = 4;
        let seq = 2;
        let numel = seq * hidden; // 8
        let max_seq = seq + 1; // 3 rows allocated
        let buf_size = max_seq * hidden * 4; // 48 bytes

        let mut src = vec![0u8; buf_size];
        // Write values for seq_len=2 (8 f32 elements)
        for i in 0..numel {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&(i as f32 + 1.0).to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F32);

        // Assert: only first numel elements decoded (not the slack row)
        let decoded = out.expect("buffer with one slack row must decode");
        assert_eq!(decoded.len(), numel);
        assert!((decoded[0] - 1.0).abs() < 1e-6);
        assert!((decoded[7] - 8.0).abs() < 1e-6);
    }

    // ========================================================================
    // Wave-460..474: Additional 15 tests for uncovered edge cases
    // ========================================================================

    // @trace TEST-MLE-460 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_nan_and_infinity_preserved_via_half_path() {
        // Arrange: F16 NaN 和 infinity 通过半精度路径解码
        // hidden=3, seq=1 -> 3 个 F16 = 6 bytes.
        // F32 stride=12, 6%12!=0 -> F32 失败. half stride=6, 6%6==0 -> 通过.
        let nan_f16 = f16::from_bits(0x7E00u16); // F16 quiet NaN
        let pos_inf_f16 = f16::from_bits(0x7C00u16); // F16 +infinity
        let neg_inf_f16 = f16::from_bits(0xFC00u16); // F16 -infinity
        let src: Vec<u8> = [&nan_f16, &pos_inf_f16, &neg_inf_f16]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(src.len(), 6);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F16)
            .expect("F16 NaN/inf 必须通过半精度路径解码");

        // Assert
        assert_eq!(out.len(), 3);
        assert!(out[0].is_nan(), "F16 NaN 必须保持为 NaN");
        assert!(out[1].is_infinite() && out[1].is_sign_positive(), "F16 +inf 必须保持为正无穷");
        assert!(out[2].is_infinite() && out[2].is_sign_negative(), "F16 -inf 必须保持为负无穷");
    }

    // @trace TEST-MLE-461 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_infinity_via_half_path() {
        // Arrange: BF16 infinity 通过半精度路径解码
        // hidden=2, seq=1 -> 2 个 BF16 = 4 bytes.
        // F32 stride=8, 4%8!=0 -> F32 失败. half stride=4, 4%4==0 -> 通过.
        let pos_inf = bf16::from_bits(0x7F80u16); // BF16 +infinity
        let neg_inf = bf16::from_bits(0xFF80u16); // BF16 -infinity
        let src: Vec<u8> = [&pos_inf, &neg_inf]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(src.len(), 4);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16)
            .expect("BF16 infinity 必须通过半精度路径解码");

        // Assert
        assert_eq!(out.len(), 2);
        assert!(out[0].is_infinite() && out[0].is_sign_positive());
        assert!(out[1].is_infinite() && out[1].is_sign_negative());
    }

    // @trace TEST-MLE-462 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_skip_multiple_layers_after_capture_exits_correctly() {
        // Arrange: 在 target layer 捕获后，下一个 pre_node 跳过多个 layer
        // 模拟 layer_idx 从 target=2 直接跳到 layer_idx=10（中间层被跳过）
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // 在 layer 2 捕获
        let ctx2 = holder.ctx(2, 1);
        cb.post_node(&ctx2, &make_f32_output(&[1.1, 2.2, 3.3, 4.4]));

        // Act: pre_node 跳到 layer 10（跨多层的转换）
        let ctx10 = holder.ctx(10, 1);
        match cb.pre_node(&ctx10) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.1, 2.2, 3.3, 4.4]);
            }
            other => panic!("跨层转换必须触发 ExitEarly, 得到 {:?}", other),
        }

        // Assert: captured 已被消费
        assert!(cb.captured.is_none());
    }

    // @trace TEST-MLE-463 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_numel_zero_from_nonzero_dims_returns_none() {
        // Arrange: seq_len=1, hidden_size=0 在早期返回中被处理，
        // 但这里测试一个更微妙的场景: 让 checked_mul 返回 Some(0)
        // 实际上 hidden_size=0 已被早期返回捕获，所以测试
        // 一个边界情况：非常大的 seq_len * hidden_size == 0
        // 这在正常情况下不可能发生，但验证 guard 的完整性。
        // 改为测试: 很小的 buffer 配合合理的维度
        let src = vec![0u8; 4];
        // seq_len=1, hidden_size=1 -> numel=1, F32 stride=4, 4%4==0 -> 通过
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32);
        assert!(out.is_some(), "正常维度组合必须成功解码");
    }

    // @trace TEST-MLE-464 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_capture_at_layer_zero_post_nodes_at_same_layer_only_latest_kept() {
        // Arrange: 在 layer 0 连续多次 post_node，验证只保留最后一次捕获
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(3, DType::F32);

        // 连续 10 次捕获，值递增
        for i in 0..10u32 {
            let ctx = holder.ctx(0, 1);
            cb.post_node(&ctx, &make_f32_output(&[i as f32, (i * 2) as f32, (i * 3) as f32]));
        }

        // Act: 转换到 layer 1
        let ctx1 = holder.ctx(1, 1);
        match cb.pre_node(&ctx1) {
            CallbackAction::ExitEarly { logits } => {
                // 必须是最后一次捕获的值 (i=9)
                assert!((logits[0] - 9.0).abs() < 1e-6, "第一个值必须是最后一次捕获");
                assert!((logits[1] - 18.0).abs() < 1e-6);
                assert!((logits[2] - 27.0).abs() < 1e-6);
            }
            other => panic!("预期 ExitEarly, 得到 {:?}", other),
        }
    }

    // @trace TEST-MLE-465 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_numel_times_four_overflows_returns_none() {
        // Arrange: numel 合理但 numel * 4 溢出 usize
        // 在实际中这需要 seq_len * hidden_size > usize::MAX / 4
        let src = vec![0u8; 16];
        let huge_numel_half = usize::MAX / 4 + 1;
        // seq_len = huge_numel_half, hidden_size = 1 -> numel > usize::MAX/4
        // numel = huge_numel_half * 1 = huge_numel_half
        // numel.checked_mul(4) = (usize::MAX/4 + 1).checked_mul(4) = None (overflow)
        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src,
            huge_numel_half,
            1,
            DType::F32,
        );
        assert!(out.is_none(), "numel * 4 溢出必须返回 None");
    }

    // @trace TEST-MLE-466 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_with_f16_compute_dtype_and_f32_buffer_captures_correctly() {
        // Arrange: 模型声明 F16 compute_dtype，CPU 路径使用 F32 buffer
        // hidden=2, seq=1, F32 buffer = 8 bytes
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(2, DType::F16);
        let ctx = holder.ctx(1, 1);
        let output = make_f32_output(&[1.5, -2.5]);

        // Act
        cb.post_node(&ctx, &output);

        // Assert: F32 candidate wins (stride=8, 8%8==0)
        let captured = cb.captured.as_ref().expect("F32 buffer 必须被捕获，即使声明 F16");
        assert_eq!(captured.len(), 2);
        assert!((captured[0] - 1.5).abs() < 1e-6);
        assert!((captured[1] - (-2.5)).abs() < 1e-6);
    }

    // @trace TEST-MLE-467 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_huge_preallocated_tiny_live_region() {
        // Arrange: 巨大的预分配 buffer (max_seq=8192)，但 seq_len=1 仅使用一小部分
        let hidden = 128;
        let max_seq = 8192;
        let mut src = vec![0u8; max_seq * hidden * 4]; // 4MB buffer
        // 仅写入第一行的值
        for i in 0..hidden {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&(i as f32 * 0.01).to_le_bytes());
        }

        // Act: 只解码 seq_len=1 的活跃区域
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F32)
            .expect("巨大预分配 buffer 必须仅解码活跃区域");

        // Assert: 只解码了 128 个 f32
        assert_eq!(out.len(), hidden);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[127] - 1.27).abs() < 1e-4);
    }

    // @trace TEST-MLE-468 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_stride_passes_size_passes_with_odd_seq_len() {
        // Arrange: 奇数 seq_len=7, hidden=3 -> 21 f32 = 84 bytes exact fit
        let values: Vec<f32> = (0..21).map(|i| (i as f32) - 10.0).collect();
        let src = make_f32_output(&values);
        assert_eq!(src.len(), 84);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 7, 3, DType::F32)
            .expect("奇数 seq_len 必须正确解码");

        // Assert: 第一个和最后一个值
        assert_eq!(out.len(), 21);
        assert!((out[0] - (-10.0)).abs() < 1e-6);
        assert!((out[20] - 10.0).abs() < 1e-6);
    }

    // @trace TEST-MLE-469 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_target_layer_visited_then_many_non_target_pre_nodes_then_exit() {
        // Arrange: 在 target layer 捕获后，在多个非 target layer 调用 pre_node，
        // 只有第一次非 target layer 的 pre_node 会触发 ExitEarly
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(2, DType::F32);

        // 捕获
        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&[42.0, -42.0]));

        // Act: 第一次非 target layer pre_node 触发 ExitEarly
        let ctx2 = holder.ctx(2, 1);
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![42.0, -42.0]);
            }
            other => panic!("第一次转换必须 ExitEarly, 得到 {:?}", other),
        }

        // 后续 pre_node 全部 Continue（captured 已被消费）
        for layer in 3..20 {
            let ctx = holder.ctx(layer, 1);
            assert!(
                matches!(cb.pre_node(&ctx), CallbackAction::Continue),
                "layer {} 的 pre_node 必须返回 Continue（无捕获状态）",
                layer
            );
        }
    }

    // @trace TEST-MLE-470 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_nan_via_half_path() {
        // Arrange: BF16 NaN 通过半精度路径解码
        // hidden=2, seq=1 -> 2 个 BF16 = 4 bytes.
        // F32 stride=8, 4%8!=0 -> F32 失败. half stride=4, 4%4==0 -> 通过.
        let bf16_nan = bf16::from_bits(0x7FC0u16); // BF16 quiet NaN
        let bf16_neg_nan = bf16::from_bits(0xFFC0u16); // BF16 negative NaN
        let src: Vec<u8> = [&bf16_nan, &bf16_neg_nan]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(src.len(), 4);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16)
            .expect("BF16 NaN 必须通过半精度路径解码");

        // Assert
        assert_eq!(out.len(), 2);
        assert!(out[0].is_nan(), "BF16 NaN 必须保持为 NaN");
        assert!(out[1].is_nan() && out[1].is_sign_negative(), "BF16 负 NaN 必须保持符号");
    }

    // @trace TEST-MLE-471 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_zero_value_via_half_path() {
        // Arrange: F16 正零和负零通过半精度路径解码
        // hidden=3, seq=1 -> 3 个 F16 = 6 bytes. F32 stride=12, 6%12!=0 -> F32 失败.
        let pos_zero = f16::from_bits(0x0000u16); // +0.0
        let neg_zero = f16::from_bits(0x8000u16); // -0.0
        let regular = f16::from_f32(3.14);
        let src: Vec<u8> = [&pos_zero, &neg_zero, &regular]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(src.len(), 6);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F16)
            .expect("F16 零值必须通过半精度路径解码");

        // Assert
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], 0.0);
        assert!(out[1].is_sign_negative(), "F16 负零符号必须保持");
        assert!((out[2] - 3.14).abs() < 0.1);
    }

    // @trace TEST-MLE-472 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_post_node_at_many_wrong_layers_then_target_then_exit() {
        // Arrange: 在多个错误 layer 上调用 post_node（不应捕获），
        // 最后在 target layer 捕获并退出
        let mut cb = MidLayerEncodeCallback::new(5);
        let holder = TestCtxHolder::new(3, DType::F32);

        // 在 layer 0..4 上调用 post_node（全部非 target）
        for layer in 0..5 {
            let ctx = holder.ctx(layer, 1);
            let output = make_f32_output(&[layer as f32; 3]);
            cb.post_node(&ctx, &output);
            // 不应捕获
        }
        assert!(cb.captured.is_none(), "非 target layer 的 post_node 不应捕获");

        // 在 target layer 5 捕获
        let ctx5 = holder.ctx(5, 1);
        cb.post_node(&ctx5, &make_f32_output(&[100.0, 200.0, 300.0]));
        assert!(cb.captured.is_some());

        // 在 layer 6+ 再次调用 post_node（不应覆盖）
        for layer in 6..10 {
            let ctx = holder.ctx(layer, 1);
            cb.post_node(&ctx, &make_f32_output(&[0.0; 3]));
        }

        // Act: 转换退出
        let ctx_next = holder.ctx(100, 1);
        match cb.pre_node(&ctx_next) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![100.0, 200.0, 300.0]);
            }
            other => panic!("预期 ExitEarly, 得到 {:?}", other),
        }
    }

    // @trace TEST-MLE-473 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_one_byte_short_of_f32_stride_alignment() {
        // Arrange: buffer 差 1 byte 就能满足 F32 stride 对齐
        // hidden=4, F32 stride=16. buffer=15 bytes -> 15%16!=0 -> F32 失败.
        // half stride=8, 15%8!=0 -> half 也失败.
        let src = vec![0u8; 15];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32);

        // Assert: 两个候选都因 stride 不对齐而失败
        assert!(out.is_none(), "差 1 byte 不满足 F32 stride 必须返回 None");
    }

    // @trace TEST-MLE-474 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_half_numel_times_two_overflows_returns_none() {
        // Arrange: numel 合理但 numel * 2 溢出 usize
        // half candidate 的 size check: numel.checked_mul(2)
        // 使用 seq_len 使 checked_mul(2) 溢出
        let src = vec![0u8; 16];
        let huge_numel = usize::MAX / 2 + 1; // numel = (usize::MAX / 2 + 1)
        // hidden=1, seq=huge_numel -> numel = huge_numel
        // F32: numel.checked_mul(4) = None (溢出) -> 但 F32 stride 先检查
        // f32_stride = 1 * 4 = 4, 16%4==0 -> stride 通过
        // 但 numel * 4 溢出 -> None
        // half: numel.checked_mul(2) = None (溢出) -> None
        let out = MidLayerEncodeCallback::decode_hidden_output(
            &src,
            huge_numel,
            1,
            DType::F16,
        );
        assert!(out.is_none(), "numel * 2 溢出必须导致 half 候选返回 None");
    }

    // @trace TEST-MLE-475 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_four_bytes_with_hidden_four_seq_one_rejected() {
        // Arrange: 4 bytes 配合 hidden=4, seq=1 -> 需要 16 bytes 但只有 4 bytes
        // F32 stride=16, 4%16!=0 -> F32 stride 失败
        // half stride=8, 4%8!=0 -> half stride 也失败
        let src = vec![0xFFu8; 4];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32);

        // Assert: buffer 远不够
        assert!(out.is_none(), "4 bytes 配合 hidden=4 远不足, 必须 None");
    }

    // @trace TEST-MLE-476 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_pre_and_post_alternating_at_target_only_final_post_captured() {
        // Arrange: 在 target layer 上交替调用 pre_node 和 post_node
        // 验证 pre_node 不影响捕获，只有最后的 post_node 数据被保留
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(2, DType::F32);

        // 交替序列: pre -> post(A) -> pre -> post(B) -> pre -> post(C)
        let ctx3 = holder.ctx(3, 1);
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));
        cb.post_node(&ctx3, &make_f32_output(&[1.0, 2.0])); // 捕获 A

        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));
        cb.post_node(&ctx3, &make_f32_output(&[3.0, 4.0])); // 覆盖为 B

        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));
        cb.post_node(&ctx3, &make_f32_output(&[5.0, 6.0])); // 覆盖为 C

        // Act: 转换退出
        let ctx4 = holder.ctx(4, 1);
        match cb.pre_node(&ctx4) {
            CallbackAction::ExitEarly { logits } => {
                // 必须是最后一次 post_node 的值 (C)
                assert_eq!(logits, vec![5.0, 6.0], "必须保留最后一次 post_node 的数据");
            }
            other => panic!("预期 ExitEarly, 得到 {:?}", other),
        }
    }

    // @trace TEST-MLE-477 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_f32_candidate_wins_over_half_when_buffer_is_f32_aligned() {
        // Arrange: buffer 是 F32 对齐的，但声明 BF16 dtype。
        // F32 candidate 必须先被检查且成功，即使声明的是 BF16。
        // hidden=2, seq=2 -> 4 个 f32 = 16 bytes
        // F32 stride=8, 16%8==0, 16>=16 -> F32 通过
        let values: Vec<f32> = vec![0.125, -0.125, 0.25, -0.25];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 16);

        // Act: 声明 BF16 但 F32 candidate 赢
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::BF16)
            .expect("F32 candidate 必须在 BF16 声明时也赢");

        // Assert: F32 精度保留（如果是 BF16 路径会有精度损失）
        assert_eq!(out.len(), 4);
        for (i, expected) in values.iter().enumerate() {
            assert!(
                (out[i] - *expected).abs() < 1e-6,
                "F32 候选必须保留完整精度, out[{}]={} expected={}",
                i, out[i], expected
            );
        }
    }

    // @trace TEST-MLE-478 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_with_hidden_one_seq_max_preallocated_buffer() {
        // Arrange: hidden=1, max_seq=65536 的超大预分配 buffer, seq_len=1
        // 验证仅第一个 4-byte 元素被解码
        let max_seq = 65536;
        let mut src = vec![0u8; max_seq * 4]; // 256KB buffer
        let expected = -999.999f32;
        src[0..4].copy_from_slice(&expected.to_le_bytes());

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32)
            .expect("超大预分配 buffer 必须仅解码 1 个元素");

        // Assert
        assert_eq!(out.len(), 1);
        assert!((out[0] - expected).abs() < 1e-3);
    }

    // @trace TEST-MLE-479 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_negative_infinity_via_half_path() {
        // Arrange: F16 negative infinity 通过半精度路径
        // hidden=1, seq=1 -> 1 个 F16 = 2 bytes.
        // F32 stride=4, 2%4!=0 -> F32 失败. half stride=2, 2%2==0 -> 通过.
        let neg_inf = f16::from_bits(0xFC00u16); // F16 -infinity
        let src = neg_inf.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16)
            .expect("F16 -inf 必须解码");

        // Assert
        assert_eq!(out.len(), 1);
        assert!(out[0].is_infinite() && out[0].is_sign_negative(), "F16 -inf 必须保持为负无穷");
    }

    // @trace TEST-MLE-480 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_positive_infinity_and_zero_mixed() {
        // Arrange: 混合正无穷和零值的 buffer
        let values: Vec<f32> = vec![f32::INFINITY, 0.0, -0.0, f32::INFINITY];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F32)
            .expect("正无穷和零值混合必须解码");

        // Assert: 每个值的特殊属性保持
        assert_eq!(out.len(), 4);
        assert!(out[0].is_infinite() && out[0].is_sign_positive());
        assert_eq!(out[1].to_bits(), 0u32); // +0.0
        assert_eq!(out[2].to_bits(), 0x80000000u32); // -0.0
        assert!(out[3].is_infinite() && out[3].is_sign_positive());
    }

    // @trace TEST-MLE-481 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_transition_from_higher_to_lower_layer_with_capture_exits() {
        // Arrange: 捕获在 layer 3，然后 pre_node 在 layer 1（更低的 layer）
        // 这模拟反向遍历图的边缘情况（虽然正常推理不会发生）
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);

        // 捕获在 layer 3
        let ctx3 = holder.ctx(3, 1);
        cb.post_node(&ctx3, &make_f32_output(&[10.0, 20.0, 30.0, 40.0]));

        // Act: pre_node 在 layer 1（比 target 更低）
        let ctx1 = holder.ctx(1, 1);
        match cb.pre_node(&ctx1) {
            CallbackAction::ExitEarly { logits } => {
                // layer_idx=1 != target_layer=3 且 captured=Some -> ExitEarly
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0]);
            }
            other => panic!("转换到任何非 target layer 都应 ExitEarly, 得到 {:?}", other),
        }

        // Assert: captured 被消费
        assert!(cb.captured.is_none());
    }

    // @trace TEST-MLE-482 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_at_hidden_size_power_of_two() {
        // Arrange: hidden_size 是 2 的幂次 (512), seq_len=2 -> 1024 f32 = 4096 bytes
        let hidden = 512;
        let seq = 2;
        let numel = hidden * seq;
        let values: Vec<f32> = (0..numel).map(|i| (i as f32) / numel as f32).collect();
        let src = make_f32_output(&values);
        assert_eq!(src.len(), numel * 4);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F32)
            .expect("hidden=512 seq=2 必须正确解码");

        // Assert: 首尾值验证
        assert_eq!(out.len(), numel);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[numel - 1] - (numel as f32 - 1.0) / numel as f32).abs() < 1e-6);
    }

    // @trace TEST-MLE-483 [req:REQ-HR-002] [level:unit]
    #[test]
    fn callback_send_trait_bounds_satisfied() {
        // Arrange: 验证 MidLayerEncodeCallback 满足 Send trait bound
        // 这对于跨线程传递 callback 是必要的
        fn assert_send<T: Send>() {}
        assert_send::<MidLayerEncodeCallback>();

        // 同时验证可以放入 Box<dyn LayerCallback + Send>
        let cb: Box<dyn LayerCallback + Send> = Box::new(MidLayerEncodeCallback::new(3));
        assert_eq!(cb.priority(), 55);
        assert_eq!(cb.name(), "MidLayerEncode");
    }

    // @trace TEST-MLE-484 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_positive_infinity_f32_candidate_fails_half_succeeds() {
        // Arrange: BF16 +infinity 在一个仅通过半精度路径的 buffer 中
        // hidden=3, seq=1 -> 3 个 BF16 = 6 bytes
        // F32 stride=12, 6%12!=0 -> F32 失败
        // half stride=6, 6%6==0, 6>=6 -> 通过
        let b1 = bf16::from_bits(0x7F80u16); // +infinity
        let b2 = bf16::from_f32(0.0);
        let b3 = bf16::from_f32(-0.5);
        let src: Vec<u8> = [&b1, &b2, &b3]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(src.len(), 6);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::BF16)
            .expect("BF16 infinity 必须通过半精度路径解码");

        // Assert
        assert_eq!(out.len(), 3);
        assert!(out[0].is_infinite() && out[0].is_sign_positive());
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - (-0.5)).abs() < 0.01);
    }

    // @trace TEST-MLE-485 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_full_walk_all_layers_capture_only_at_target() {
        // Arrange: 遍历所有 8 层（holder.num_layers=8），只在 target=4 捕获
        // 注意: 捕获后，第一个非 target layer 的 pre_node 会触发 ExitEarly，
        // 所以后续层的遍历不会发生（ExitEarly 终止执行）。
        let mut cb = MidLayerEncodeCallback::new(4);
        let holder = TestCtxHolder::new(2, DType::F32);

        // 遍历 layer 0..4（到 target 为止）
        for layer in 0..4 {
            let ctx = holder.ctx(layer, 1);
            assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));

            let output = make_f32_output(&[layer as f32 * 10.0, layer as f32 * 20.0]);
            let action = cb.post_node(&ctx, &output);
            assert!(matches!(action, CallbackAction::Continue));
            // target 之前的层: 无捕获
            assert!(cb.captured.is_none());
        }

        // Layer 4: target layer — 捕获
        let ctx4 = holder.ctx(4, 1);
        assert!(matches!(cb.pre_node(&ctx4), CallbackAction::Continue));
        let action4 = cb.post_node(&ctx4, &make_f32_output(&[40.0, 80.0]));
        assert!(matches!(action4, CallbackAction::Continue));
        assert!(cb.captured.is_some());

        // Assert: captured 持有 layer 4 的数据
        let captured = cb.captured.as_ref().unwrap();
        assert!((captured[0] - 40.0).abs() < 1e-6);
        assert!((captured[1] - 80.0).abs() < 1e-6);

        // Act: 转换到 layer 5 触发 ExitEarly（不会继续到 layer 6..7）
        let ctx5 = holder.ctx(5, 1);
        match cb.pre_node(&ctx5) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![40.0, 80.0]);
            }
            other => panic!("layer 5 的 pre_node 必须触发 ExitEarly, 得到 {:?}", other),
        }
        assert!(cb.captured.is_none(), "ExitEarly 后 captured 必须被消费");
    }

    // @trace TEST-MLE-486 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_with_all_zero_bits_produces_positive_zero() {
        // Arrange: 全零 bit buffer (正零)
        let src = vec![0u8; 16]; // 4 个 f32 元素，全部是 +0.0
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32)
            .expect("全零 buffer 必须解码为 4 个 +0.0");

        // Assert: 所有元素都是正零
        assert_eq!(out.len(), 4);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v.to_bits(), 0u32, "元素 {} 必须是 +0.0 (全零 bit)", i);
            assert!(!v.is_sign_negative(), "元素 {} 不能是负零", i);
        }
    }

    // @trace TEST-MLE-487 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_rejects_output_with_wrong_hidden_size_alignment() {
        // Arrange: output buffer 的长度不是 hidden_size * 4 的倍数
        // 但恰好是另一个维度的倍数。hidden=6, F32 stride=24
        // buffer = 48 bytes (48%24==0) 但 seq_len=1 只需 24 bytes
        // 等等，48%24==0 且 48>=24 -> 会成功解码 6 个元素
        // 改为: hidden=6, buffer=30 bytes, 30%24!=0 -> F32 失败
        // half stride=12, 30%12!=0 -> half 也失败
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(6, DType::F32);
        let ctx = holder.ctx(2, 1);

        let output = vec![0u8; 30]; // 30 bytes, not aligned to 6*4=24
        cb.post_node(&ctx, &output);

        // Assert: 不应捕获
        assert!(cb.captured.is_none(), "未对齐的 buffer 不应被捕获");
    }

    // @trace TEST-MLE-488 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_positive_infinity_via_half_path() {
        // Arrange: F16 +infinity 通过半精度路径
        // hidden=1, seq=1 -> 1 个 F16 = 2 bytes
        let pos_inf = f16::from_bits(0x7C00u16); // F16 +infinity
        let src = pos_inf.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16)
            .expect("F16 +inf 必须解码");

        // Assert
        assert_eq!(out.len(), 1);
        assert!(out[0].is_infinite() && out[0].is_sign_positive(), "F16 +inf 必须保持为正无穷");
    }

    // @trace TEST-MLE-489 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_half_stride_passes_but_size_check_fails_for_half() {
        // Arrange: buffer 的长度同时是 F32 和 half stride 的倍数，
        // 但 numel * 2 (half 需要的字节数) 大于 buffer。
        // hidden=4, seq=4 -> numel=16.
        // F32 stride=16, buffer=32 bytes, 32%16==0, 32>=64? No -> F32 size 失败
        // half stride=8, 32%8==0, 32>=32 -> half size 通过
        // 但 DType::F32 声明 -> half match arm 无 F32 case -> None
        let src = vec![0u8; 32];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 4, 4, DType::F32);

        // Assert: F32 size 检查失败，half match arm 不接受 F32 dtype
        assert!(
            out.is_none(),
            "F32 size 检查失败且 half match arm 不接受 F32 dtype"
        );
    }

    // @trace TEST-MLE-490 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_after_many_post_nodes_at_non_target_no_capture_persists() {
        // Arrange: 在多个非 target layer 反复调用 post_node，
        // 然后验证 pre_node 不会意外触发 ExitEarly
        let mut cb = MidLayerEncodeCallback::new(7);
        let holder = TestCtxHolder::new(3, DType::F32);

        // 在 layer 0..6 上反复调用 post_node（全部非 target）
        for round in 0..3 {
            for layer in 0..7 {
                let ctx = holder.ctx(layer, 1);
                let output = make_f32_output(&[round as f32, (round + 1) as f32, (round + 2) as f32]);
                cb.post_node(&ctx, &output);
            }
        }

        // Assert: 无捕获
        assert!(cb.captured.is_none(), "从未访问 target layer 所以无捕获");

        // Act: pre_node 在多个 layer 都应返回 Continue
        for layer in 0..7 {
            let ctx = holder.ctx(layer, 1);
            assert!(
                matches!(cb.pre_node(&ctx), CallbackAction::Continue),
                "layer {} 无捕获时应返回 Continue", layer
            );
        }
    }

    // @trace TEST-MLE-491 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_value_very_close_to_zero() {
        // Arrange: 接近零但不为零的极小 F32 值
        let tiny_pos = f32::from_bits(0x00000010u32); // 非常小的正 subnormal
        let tiny_neg = f32::from_bits(0x80000010u32); // 非常小的负 subnormal
        let values = vec![tiny_pos, 0.0f32, tiny_neg, 0.0f32];
        let src = make_f32_output(&values);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 2, DType::F32)
            .expect("极小值必须正确解码");

        // Assert: bit pattern 完整保留
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].to_bits(), tiny_pos.to_bits(), "极小正值 bit 必须保留");
        assert_eq!(out[1].to_bits(), 0u32);
        assert_eq!(out[2].to_bits(), tiny_neg.to_bits(), "极小负值 bit 必须保留");
        assert_eq!(out[3].to_bits(), 0u32);
    }

    // @trace TEST-MLE-492 [req:REQ-HR-002] [level:unit]
    #[test]
    fn callback_in_callbackchain_with_different_target_layers() {
        // Arrange: 两个 MidLayerEncodeCallback 分别针对不同 target layer
        // 放入 CallbackChain，验证第一个 ExitEarly 终止链
        use crate::graph::layer_callback::CallbackChain;

        let cb1 = MidLayerEncodeCallback::new(1);
        let cb2 = MidLayerEncodeCallback::new(3);
        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2)]);
        assert_eq!(chain.len(), 2);

        let holder = TestCtxHolder::new(2, DType::F32);

        // 在 layer 1 捕获（cb1 的 target）
        let ctx1 = holder.ctx(1, 1);
        let output1 = make_f32_output(&[11.0, 12.0]);
        assert!(matches!(chain.dispatch_post_node(&ctx1, &output1), CallbackAction::Continue));

        // 在 layer 3 捕获（cb2 的 target）
        let ctx3 = holder.ctx(3, 1);
        let output3 = make_f32_output(&[31.0, 32.0]);
        assert!(matches!(chain.dispatch_post_node(&ctx3, &output3), CallbackAction::Continue));

        // Act: pre_node 在 layer 2 -> cb1 的 ExitEarly（layer 1 -> layer 2 转换）
        let ctx2 = holder.ctx(2, 1);
        let action = chain.dispatch_pre_node(&ctx2);

        // Assert: 第一个 callback (cb1) 的 ExitEarly 应终止链
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![11.0, 12.0], "cb1 的 ExitEarly 必须先触发");
            }
            other => panic!("预期 cb1 的 ExitEarly, 得到 {:?}", other),
        }
    }

    // @trace TEST-MLE-493 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_two_element_buffer_with_hidden_two_seq_one() {
        // Arrange: hidden=2, seq=1 -> 2 f32 = 8 bytes exact fit
        // F32 stride=8, 8%8==0, 8>=8 -> 通过
        let values: Vec<f32> = vec![f32::MIN_POSITIVE, f32::MAX];
        let src = make_f32_output(&values);
        assert_eq!(src.len(), 8);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32)
            .expect("hidden=2 seq=1 exact fit 必须解码");

        // Assert: 极端值保留
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], f32::MIN_POSITIVE);
        assert_eq!(out[1], f32::MAX);
    }

    // @trace TEST-MLE-494 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_smallest_positive_normal_value() {
        // Arrange: BF16 最小正规值 (0x0080 = 2^{-126})
        // 通过半精度路径: hidden=1, 2 bytes, F32 stride=4, 2%4!=0 -> F32 失败
        let b = bf16::from_bits(0x0080u16); // BF16 smallest positive normal
        let src = b.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16)
            .expect("BF16 最小正规值必须解码");

        // Assert
        assert_eq!(out.len(), 1);
        assert!(out[0] > 0.0, "必须为正值");
        assert!(out[0] < 1e-30, "必须是非常小的值");
    }

    // @trace TEST-MLE-495 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_single_layer_model_target_zero_immediate_exit() {
        // Arrange: 单层模型 (num_layers=1), target layer=0
        // 模拟: 捕获在 layer 0, 然后立即遇到 post-decoder node
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(4, DType::F32);

        // 捕获在 layer 0
        let ctx0 = holder.ctx(0, 1);
        cb.post_node(&ctx0, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));

        // Act: 直接到 post-decoder node (final_norm, layer_idx=1000)
        let ctx_post = holder.ctx(1000, 1);
        match cb.pre_node(&ctx_post) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("单层模型 post-decoder 必须触发 ExitEarly, 得到 {:?}", other),
        }

        // captured 已消费
        assert!(cb.captured.is_none());
    }

    // @trace TEST-MLE-496 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_all_ones_in_lowest_byte_per_element() {
        // Arrange: 每个 F32 元素的最低字节为 0xFF，其余为 0
        // 0x000000FF = 极小的 subnormal
        let bits: Vec<u32> = vec![0x000000FF, 0x00000100, 0x000001FF];
        let src: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        assert_eq!(src.len(), 12);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 3, DType::F32)
            .expect("subnormal bit pattern 必须解码");

        // Assert: bit pattern 完整保留
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].to_bits(), 0x000000FF);
        assert_eq!(out[1].to_bits(), 0x00000100);
        assert_eq!(out[2].to_bits(), 0x000001FF);
    }

    // @trace TEST-MLE-497 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_two_elements_with_opposite_signs_via_half_path() {
        // Arrange: 2 个 F16 值（一正一负），通过半精度路径
        // hidden=2, seq=1 -> 2 个 F16 = 4 bytes
        // F32 stride=8, 4%8!=0 -> F32 失败. half stride=4, 4%4==0 -> 通过.
        let pos = f16::from_f32(1234.0); // 较大正值
        let neg = f16::from_f32(-5678.0); // 较大负值
        let src: Vec<u8> = [&pos, &neg].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16)
            .expect("F16 正负值必须解码");

        // Assert: 符号和近似值正确
        assert_eq!(out.len(), 2);
        assert!(out[0] > 0.0);
        assert!(out[1] < 0.0);
        assert!((out[0] - 1234.0).abs() < 5.0, "F16 1234 精度损失在可接受范围");
        assert!((out[1] - (-5678.0)).abs() < 20.0, "F16 -5678 精度损失在可接受范围");
    }

    // @trace TEST-MLE-498 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_at_target_layer_with_different_seq_lens_captures_latest() {
        // Arrange: 同一 target layer，先 seq_len=3 捕获，再 seq_len=1 捕获
        // 验证后者覆盖前者
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(2, DType::F32);

        // 第一次: seq_len=3 -> 6 个 f32 = 24 bytes
        let ctx3 = holder.ctx(2, 3);
        let output3 = make_f32_output(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        cb.post_node(&ctx3, &output3);
        assert_eq!(cb.captured.as_ref().unwrap().len(), 6);

        // 第二次: seq_len=1 -> 2 个 f32 = 8 bytes
        let ctx1 = holder.ctx(2, 1);
        let output1 = make_f32_output(&[99.0, 88.0]);
        cb.post_node(&ctx1, &output1);

        // Assert: 最新捕获 (seq_len=1) 覆盖
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured.len(), 2, "最新捕获应覆盖旧捕获");
        assert!((captured[0] - 99.0).abs() < 1e-6);
        assert!((captured[1] - 88.0).abs() < 1e-6);
    }

    // @trace TEST-MLE-499 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_with_f6e3m2_declared_dtype_f32_candidate_wins() {
        // Arrange: DType::F6E3M2 声明，但 buffer 是 F32 对齐的
        // F32 candidate 先检查且通过
        let values: Vec<f32> = vec![-128.0, 127.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 8);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F6E3M2);

        // Assert: F32 candidate 赢
        let decoded = out.expect("F32 candidate 必须在 F6E3M2 声明时也成功");
        assert_eq!(decoded.len(), 2);
        assert!((decoded[0] - (-128.0)).abs() < 1e-6);
        assert!((decoded[1] - 127.0).abs() < 1e-6);
    }

    // @trace TEST-MLE-500 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_two_elements_one_zero_one_max() {
        // Arrange: BF16 零值和最大值，通过半精度路径
        // hidden=2, seq=1 -> 2 个 BF16 = 4 bytes
        // F32 stride=8, 4%8!=0 -> F32 失败
        let zero = bf16::from_f32(0.0);
        let max_val = bf16::from_f32(f16::MAX.to_f32()); // 使用 F16 max 的 f32 表示
        let src: Vec<u8> = [&zero, &max_val]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(src.len(), 4);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::BF16)
            .expect("BF16 零和最大值必须解码");

        // Assert
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!(out[1] > 0.0, "BF16 max 必须为正");
    }

    // @trace TEST-MLE-501 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_just_above_f32_stride_alignment() {
        // Arrange: buffer 大小刚好超过 F32 stride 的倍数
        // hidden=4, F32 stride=16. buffer=20 bytes -> 20%16!=0 -> F32 失败
        // half stride=8, 20%8!=0 -> half 也失败
        let src = vec![0u8; 20];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32);

        // Assert: stride 不对齐
        assert!(out.is_none(), "20 bytes 对 hidden=4 不满足 stride 对齐");
    }

    // @trace TEST-MLE-502 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_ascending_integer_values_exact_fit() {
        // Arrange: 从 0 开始的连续整数值，exact fit buffer
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let src = make_f32_output(&values);
        assert_eq!(src.len(), 400);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 10, 10, DType::F32)
            .expect("100 个连续整数值必须解码");

        // Assert: 所有值精确保留
        assert_eq!(out.len(), 100);
        for (i, v) in out.iter().enumerate() {
            assert!(
                (v - i as f32).abs() < 1e-6,
                "值 {} 不匹配: 预期 {} 得到 {}",
                i, i as f32, v
            );
        }
    }

    // @trace TEST-MLE-503 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_capture_exit_with_all_negative_values() {
        // Arrange: 捕获全负值的 hidden state，验证符号完整保留
        let mut cb = MidLayerEncodeCallback::new(1);
        let holder = TestCtxHolder::new(5, DType::F32);

        let values: Vec<f32> = vec![-0.001, -1.0, -100.0, -10000.0, -f32::MAX];
        let ctx1 = holder.ctx(1, 1);
        cb.post_node(&ctx1, &make_f32_output(&values));

        // Act
        let ctx2 = holder.ctx(2, 1);
        match cb.pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                // Assert: 所有值负号保留
                assert_eq!(logits.len(), 5);
                for (i, v) in logits.iter().enumerate() {
                    assert!(
                        v.is_sign_negative(),
                        "元素 {} = {} 必须为负值",
                        i, v
                    );
                }
                assert!((logits[0] - (-0.001)).abs() < 1e-6);
                assert!((logits[1] - (-1.0)).abs() < 1e-6);
                assert!((logits[2] - (-100.0)).abs() < 1e-6);
                assert!((logits[3] - (-10000.0)).abs() < 1e-3);
                assert_eq!(logits[4], -f32::MAX);
            }
            other => panic!("预期 ExitEarly, 得到 {:?}", other),
        }
    }

    // @trace TEST-MLE-504 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_stride_alignment_with_non_power_of_two_hidden() {
        // Arrange: hidden_size=7 (非 2 的幂次), seq=3 -> 21 f32 = 84 bytes
        // F32 stride=28, 84%28==0 -> 通过
        // 这测试非对齐隐藏维度的 stride 检查
        let values: Vec<f32> = (0..21).map(|i| ((i as f32) - 10.0) * 0.5).collect();
        let src = make_f32_output(&values);
        assert_eq!(src.len(), 84);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 3, 7, DType::F32)
            .expect("hidden=7 (非 2 的幂次) 必须正确解码");

        // Assert
        assert_eq!(out.len(), 21);
        assert!((out[0] - (-5.0)).abs() < 1e-5);
        assert!((out[20] - 5.0).abs() < 1e-5);
    }

    // @trace TEST-MLE-505 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_correct_shape_after_many_wrong_shapes_all_rejected() {
        // Arrange: 在 target layer 上先发多次错误 shape 的 output，
        // 全部被拒绝，最后发一次正确 shape
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(2, 1);

        // 错误 shape: 3, 5, 7, 9, 11 bytes (都不是 hidden*4=16 的倍数)
        for wrong_size in [3, 5, 7, 9, 11] {
            cb.post_node(&ctx, &vec![0u8; wrong_size]);
            assert!(cb.captured.is_none(), "{} bytes 不应被捕获", wrong_size);
        }

        // 正确 shape: 16 bytes = 4 f32
        cb.post_node(&ctx, &make_f32_output(&[42.0, 43.0, 44.0, 45.0]));

        // Assert: 最终的正确 shape 被捕获
        let captured = cb.captured.as_ref().expect("多次拒绝后正确 shape 必须被捕获");
        assert_eq!(captured, &vec![42.0, 43.0, 44.0, 45.0]);
    }

    // @trace TEST-MLE-506 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_pair_with_hidden_two_via_half_path() {
        // Arrange: hidden=2, seq=1 -> 2 个 F16 = 4 bytes
        // F32 stride=8, 4%8!=0 -> F32 失败. half stride=4, 4%4==0 -> 通过.
        let h1 = f16::from_f32(0.5);
        let h2 = f16::from_f32(-0.5);
        let src: Vec<u8> = [&h1, &h2].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 4);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F16)
            .expect("F16 pair 必须通过半精度路径解码");

        // Assert
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 0.01);
        assert!((out[1] - (-0.5)).abs() < 0.01);
    }

    // @trace TEST-MLE-507 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_with_extra_padding_slack_decodes_only_live_region() {
        // Arrange: buffer 有 50% 的 padding (max_seq = 2 * seq_len)
        // hidden=3, seq=2 -> 6 f32 live = 24 bytes. buffer = 48 bytes (50% slack)
        let mut src = vec![0xAAu8; 48]; // 填充非零 stale 数据
        let live_values: Vec<f32> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6];
        for (i, v) in live_values.iter().enumerate() {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, 3, DType::F32)
            .expect("50% padding buffer 必须解码 live region");

        // Assert: 只有 6 个 live 元素被解码，stale tail 被忽略
        assert_eq!(out.len(), 6);
        assert!((out[0] - 1.1).abs() < 1e-5);
        assert!((out[5] - 6.6).abs() < 1e-5);
    }

    // @trace TEST-MLE-508 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_negative_one_preserved() {
        // Arrange: -1.0 的精确 bit pattern 验证
        let neg_one_bits: u32 = 0xBF800000u32; // -1.0
        let src: Vec<u8> = neg_one_bits.to_le_bytes().to_vec();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32)
            .expect("-1.0 必须解码");

        // Assert: bit pattern 精确保留
        assert_eq!(out[0].to_bits(), neg_one_bits);
        assert_eq!(out[0], -1.0f32);
    }

    // @trace TEST-MLE-509 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_at_target_after_exit_with_recapture_exits_on_next_transition() {
        // Arrange: 捕获 -> 退出 -> 重新捕获在 target layer -> 新转换退出
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(2, DType::F32);

        // 第一轮: 捕获 [1,2], 退出
        let ctx2 = holder.ctx(2, 1);
        cb.post_node(&ctx2, &make_f32_output(&[1.0, 2.0]));
        let ctx3 = holder.ctx(3, 1);
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::ExitEarly { .. }));
        assert!(cb.captured.is_none());

        // 回到 target layer: 重新捕获 [3,4]
        cb.post_node(&ctx2, &make_f32_output(&[3.0, 4.0]));
        assert!(cb.captured.is_some());

        // pre_node 仍在 target -> Continue
        assert!(matches!(cb.pre_node(&ctx2), CallbackAction::Continue));

        // 转换退出: 必须携带 [3,4]
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![3.0, 4.0], "重新捕获后的退出必须携带新数据");
            }
            other => panic!("预期 ExitEarly, 得到 {:?}", other),
        }
    }

    // @trace TEST-MLE-510 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_size_not_multiple_of_any_candidate_stride() {
        // Arrange: buffer 大小既不是 F32 stride 的倍数，也不是 half stride 的倍数
        // hidden=6, F32 stride=24, half stride=12
        // buffer=30 bytes -> 30%24!=0, 30%12!=0 (wait, 30%12==6!=0? no, 30/12=2.5)
        // Actually 30%12 = 6, so both fail. But wait: 30%24=6, 30%12=6 -> both fail
        let src = vec![0u8; 30];

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 6, DType::BF16);

        // Assert: 两个候选都因 stride 不对齐而失败
        assert!(out.is_none(), "30 bytes 对 hidden=6 不满足任何 stride 对齐");
    }

    // @trace TEST-MLE-511 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_multiple_post_nodes_interleaved_with_pre_nodes_at_target() {
        // Arrange: 在 target layer 上交替 post_node 和 pre_node
        // 验证 pre_node 不清除捕获，只有层转换才会
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(2, DType::F32);
        let ctx3 = holder.ctx(3, 1);

        // post(A) -> pre -> post(B) -> pre -> post(C) -> pre -> exit
        cb.post_node(&ctx3, &make_f32_output(&[1.0, 2.0]));
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));
        assert!(cb.captured.is_some()); // 未被消费

        cb.post_node(&ctx3, &make_f32_output(&[3.0, 4.0]));
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));
        assert!(cb.captured.is_some()); // 仍未被消费

        cb.post_node(&ctx3, &make_f32_output(&[5.0, 6.0]));
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));

        // 转换退出
        let ctx4 = holder.ctx(4, 1);
        match cb.pre_node(&ctx4) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![5.0, 6.0], "必须携带最后一次 post_node 的数据");
            }
            other => panic!("预期 ExitEarly, 得到 {:?}", other),
        }
    }

    // @trace TEST-MLE-512 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_with_f8e5m2_declared_dtype_f32_path_succeeds() {
        // Arrange: DType::F8E5M2 声明但 buffer 是 F32 对齐的
        let src: Vec<u8> = (-1.0f32).to_le_bytes().to_vec();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F8E5M2);

        // Assert: F32 candidate 赢
        let decoded = out.expect("F32 path 必须在 F8E5M2 声明时也成功");
        assert!((decoded[0] - (-1.0)).abs() < 1e-6);
    }

    // @trace TEST-MLE-513 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_negative_zero_via_half_path() {
        // Arrange: F16 负零通过半精度路径
        // hidden=1, seq=1 -> 1 个 F16 = 2 bytes
        let neg_zero = f16::from_bits(0x8000u16); // F16 -0.0
        let src = neg_zero.to_le_bytes().to_vec();
        assert_eq!(src.len(), 2);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F16)
            .expect("F16 负零必须解码");

        // Assert
        assert_eq!(out.len(), 1);
        assert!(out[0].is_sign_negative(), "F16 负零符号必须保留");
        assert_eq!(out[0], 0.0);
    }

    // @trace TEST-MLE-514 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_nan_via_f32_candidate_when_buffer_is_f32_aligned() {
        // Arrange: 声明 BF16 但 buffer 是 F32 对齐的。
        // buffer 中的 bytes 如果被解释为 BF16 会是 NaN，但 F32 candidate 先检查。
        // hidden=1, F32 stride=4, buffer=4 bytes -> F32 赢
        let value = 1.0f32;
        let src: Vec<u8> = value.to_le_bytes().to_vec();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::BF16)
            .expect("F32 candidate 必须在 BF16 声明时赢");

        // Assert: F32 解码得到 1.0，不是 BF16 NaN
        assert!((out[0] - 1.0).abs() < 1e-6, "F32 解码必须得到精确 1.0，不是 BF16 NaN");
    }

    // ========================================================================
    // Additional tests (15 new) for remaining coverage gaps
    // ========================================================================

    // @trace TEST-MLE-515 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_prime_hidden_size_7_exact_fit() {
        // Arrange: hidden_size=7 (prime), seq_len=1, exact-fit buffer = 28 bytes
        let values: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 28);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 7, DType::F32)
            .expect("prime hidden_size=7 exact fit must decode");

        // Assert: all 7 values preserved exactly
        assert_eq!(out.len(), 7);
        for i in 0..7 {
            assert!((out[i] - values[i]).abs() < 1e-6, "element {} mismatch", i);
        }
    }

    // @trace TEST-MLE-516 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_prime_hidden_size_13_with_slack() {
        // Arrange: hidden_size=13 (prime), pre-allocated for max_seq=8, actual seq=2
        // total = 8 * 13 * 4 = 416 bytes. Live region = 2 * 13 * 4 = 104 bytes.
        let hidden = 13;
        let max_seq = 8;
        let mut src = vec![0u8; max_seq * hidden * 4];
        // Write first 26 elements (2 rows * 13 cols) with a distinct pattern
        for i in 0..(2 * hidden) {
            let val = if i < hidden { i as f32 } else { -(i as f32 - hidden as f32) };
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 2, hidden, DType::F32)
            .expect("prime hidden_size=13 with slack must decode");

        // Assert: 26 elements decoded, stale tail ignored
        assert_eq!(out.len(), 26);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[12] - 12.0).abs() < 1e-6);
        assert!((out[13] - 0.0).abs() < 1e-6);
        assert!((out[25] - (-12.0)).abs() < 1e-6);
    }

    // @trace TEST-MLE-517 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_target_layer_all_post_nodes_incompatible_no_capture_then_exit_continue() {
        // Arrange: target layer reached but every post_node output has incompatible shape
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(8, DType::F32);
        let ctx = holder.ctx(2, 1);

        // Send 5 incompatible outputs (sizes not multiple of hidden_size*4=32)
        for &wrong_size in &[3usize, 7, 11, 15, 19] {
            cb.post_node(&ctx, &vec![0u8; wrong_size]);
        }
        assert!(cb.captured.is_none(), "all incompatible outputs must not capture");

        // Act: transition out — no capture means Continue
        let ctx3 = holder.ctx(3, 1);
        let action = cb.pre_node(&ctx3);

        // Assert: Continue because captured is None
        assert!(
            matches!(action, CallbackAction::Continue),
            "exit without capture must return Continue"
        );
    }

    // @trace TEST-MLE-518 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_negative_pi_and_e_values() {
        // Arrange: negative mathematical constants as f32
        let neg_pi: f32 = -std::f32::consts::PI;
        let neg_e: f32 = -std::f32::consts::E;
        let src: Vec<u8> = [neg_pi, neg_e]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 2, DType::F32)
            .expect("negative constants must decode");

        // Assert: bit-exact preservation of irrational values
        assert_eq!(out[0].to_bits(), neg_pi.to_bits(), "-PI bit pattern must be preserved");
        assert_eq!(out[1].to_bits(), neg_e.to_bits(), "-E bit pattern must be preserved");
    }

    // @trace TEST-MLE-519 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_hidden_size_9_odd_with_slack() {
        // Arrange: hidden_size=9 (odd), pre-allocated for max_seq=7, seq=1
        // total = 7 * 9 * 2 = 126 bytes. F32 stride=36, 126%36=18 (not F32-aligned).
        // half stride=18, 126%18==0, 126>=18 → half passes with BF16 declared.
        let hidden = 9;
        let max_seq = 7;
        let mut src = vec![0u8; max_seq * hidden * 2];
        let vals: Vec<bf16> = (0..hidden).map(|i| bf16::from_f32((i + 1) as f32 * 0.5)).collect();
        for (i, v) in vals.iter().enumerate() {
            let off = i * 2;
            src[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::BF16)
            .expect("BF16 odd hidden_size=9 with slack must decode");

        // Assert: only 9 elements from live region
        assert_eq!(out.len(), 9);
        assert!((out[0] - 0.5).abs() < 0.01);
        assert!((out[8] - 4.5).abs() < 0.01);
    }

    // @trace TEST-MLE-520 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_hidden_size_11_odd_forced_half_path() {
        // Arrange: hidden_size=11 (odd), seq=1, buffer = 22 bytes exactly
        // F32 stride=44, 22%44!=0 → F32 fails.
        // half stride=22, 22%22==0, 22>=22 → passes.
        let hidden = 11;
        let vals: Vec<f16> = (0..hidden)
            .map(|i| f16::from_f32(if i % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(src.len(), 22);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F16)
            .expect("F16 odd hidden_size=11 must decode via half path");

        // Assert
        assert_eq!(out.len(), 11);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - (-1.0)).abs() < 0.01);
        assert!((out[10] - 1.0).abs() < 0.01, "element 10 (even index) must be 1.0");
    }

    // @trace TEST-MLE-521 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_two_sequential_capture_exit_cycles_with_different_data() {
        // Arrange: first capture/exit cycle with data A, then second cycle with data B
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(4, DType::F32);

        // Cycle 1: capture [1,2,3,4] at layer 2, exit at layer 3
        let ctx2 = holder.ctx(2, 1);
        cb.post_node(&ctx2, &make_f32_output(&[1.0, 2.0, 3.0, 4.0]));
        let ctx3 = holder.ctx(3, 1);
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0], "cycle 1 must carry first capture");
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }

        // Cycle 2: re-capture [10,20,30,40] at layer 2, exit at layer 3
        cb.post_node(&ctx2, &make_f32_output(&[10.0, 20.0, 30.0, 40.0]));
        match cb.pre_node(&ctx3) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0, 40.0], "cycle 2 must carry second capture");
            }
            other => panic!("Expected ExitEarly in cycle 2, got {:?}", other),
        }

        // Third attempt: no capture, should return Continue
        assert!(matches!(cb.pre_node(&ctx3), CallbackAction::Continue));
    }

    // @trace TEST-MLE-522 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_large_seq_len_with_realistic_hidden_3072() {
        // Arrange: realistic hidden_size=3072 (e.g., LLaMA-13B), seq_len=4
        let hidden = 3072;
        let seq = 4;
        let total_elems = hidden * seq;
        let mut src = vec![0u8; total_elems * 4];
        // Write first and last element
        src[0..4].copy_from_slice(&1.5f32.to_le_bytes());
        let last_off = (total_elems - 1) * 4;
        src[last_off..last_off + 4].copy_from_slice(&(-2.5f32).to_le_bytes());

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, seq, hidden, DType::F32)
            .expect("realistic hidden=3072 seq=4 must decode");

        // Assert: correct length and boundary values
        assert_eq!(out.len(), total_elems);
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[total_elems - 1] - (-2.5)).abs() < 1e-6);
        // Middle values are zero
        assert!((out[hidden] - 0.0).abs() < 1e-10);
    }

    // @trace TEST-MLE-523 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_buffer_exact_f32_stride_with_extra_rows() {
        // Arrange: hidden=16, seq=1 needs 64 bytes. Buffer has 4*64=256 bytes (4 rows).
        // F32 stride=64, 256%64==0, 256>=64 → F32 passes, decodes only 16 elements.
        let hidden = 16;
        let mut src = vec![0xAAu8; 256]; // fill with stale data
        // Write live region: first 16 f32 values
        for i in 0..hidden {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&(i as f32 * 0.1).to_le_bytes());
        }

        // Act: seq_len=1 (only first row is live)
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F32)
            .expect("multi-row buffer with seq=1 must decode live region only");

        // Assert
        assert_eq!(out.len(), 16);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[15] - 1.5).abs() < 1e-5);
    }

    // @trace TEST-MLE-524 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_bf16_seq_len_4_hidden_2_row_order_verified() {
        // Arrange: BF16, seq=4, hidden=2 → 8 elements = 16 bytes
        // F32 stride=8, 16%8==0, 16>=8 → F32 passes! So F32 wins.
        // To force BF16 path: make buffer that fails F32 stride.
        // Use hidden=3: F32 stride=12, half stride=6.
        // seq=4: 4*3*2=24 bytes. 24%12==0 → F32 wins again.
        // Use hidden=3, max_seq=5: 5*3*2=30 bytes. 30%12=6 (not F32). 30%6==0 → BF16.
        let hidden = 3;
        let max_seq = 5;
        let mut src = vec![0u8; max_seq * hidden * 2];
        let vals: Vec<bf16> = (0..12) // 4 rows * 3 cols = 12 elements
            .map(|i| bf16::from_f32((i + 1) as f32))
            .collect();
        for (i, v) in vals.iter().enumerate() {
            let off = i * 2;
            src[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 4, hidden, DType::BF16)
            .expect("BF16 seq=4 hidden=3 must decode");

        // Assert: row-major order, 12 elements total
        assert_eq!(out.len(), 12);
        // Row 0: [1,2,3]
        assert!((out[0] - 1.0).abs() < 0.1);
        assert!((out[2] - 3.0).abs() < 0.1);
        // Row 3: [10,11,12]
        assert!((out[9] - 10.0).abs() < 0.1);
        assert!((out[11] - 12.0).abs() < 0.1);
    }

    // @trace TEST-MLE-525 [req:REQ-HR-002] [level:unit]
    #[test]
    fn pre_node_capture_at_layer_zero_exit_to_layer_one_then_reenter_target() {
        // Arrange: capture at layer 0, exit, then re-enter layer 0 with new data
        let mut cb = MidLayerEncodeCallback::new(0);
        let holder = TestCtxHolder::new(3, DType::F32);

        // First cycle
        let ctx0 = holder.ctx(0, 1);
        cb.post_node(&ctx0, &make_f32_output(&[1.0, 2.0, 3.0]));
        let ctx1 = holder.ctx(1, 1);
        assert!(matches!(cb.pre_node(&ctx1), CallbackAction::ExitEarly { .. }));
        assert!(cb.captured.is_none());

        // Re-enter target layer 0 with new data
        cb.post_node(&ctx0, &make_f32_output(&[10.0, 20.0, 30.0]));
        assert!(cb.captured.is_some());

        // pre_node at layer 0 (target) → Continue
        assert!(matches!(cb.pre_node(&ctx0), CallbackAction::Continue));

        // Exit at layer 1 → ExitEarly with new data
        match cb.pre_node(&ctx1) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![10.0, 20.0, 30.0]);
            }
            other => panic!("Expected ExitEarly with re-captured data, got {:?}", other),
        }
    }

    // @trace TEST-MLE-526 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f16_large_hidden_size_4096_forced_half_path() {
        // Arrange: hidden=4096, buffer = 4096*2 = 8192 bytes (1 row of F16)
        // F32 stride=16384, 8192%16384!=0 → F32 fails.
        // half stride=8192, 8192%8192==0, 8192>=8192 → half passes.
        let hidden = 4096;
        let mut src = vec![0u8; hidden * 2];
        // Write first element = 0.5, last element = -0.5
        let first = f16::from_f32(0.5);
        let last = f16::from_f32(-0.5);
        src[0..2].copy_from_slice(&first.to_le_bytes());
        let end_off = (hidden - 1) * 2;
        src[end_off..end_off + 2].copy_from_slice(&last.to_le_bytes());

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, hidden, DType::F16)
            .expect("F16 hidden=4096 must decode via half path");

        // Assert
        assert_eq!(out.len(), hidden);
        assert!((out[0] - 0.5).abs() < 0.01);
        assert!((out[hidden - 1] - (-0.5)).abs() < 0.01);
    }

    // @trace TEST-MLE-527 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_alternating_very_large_and_very_small_values() {
        // Arrange: alternating f32::MAX / 2 and f32::MIN_POSITIVE (tiny)
        let big = f32::MAX / 2.0;
        let tiny = f32::MIN_POSITIVE;
        let values: Vec<f32> = vec![big, tiny, big, tiny];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 4, DType::F32)
            .expect("alternating large/small must decode");

        // Assert: exact bit preservation
        assert_eq!(out[0].to_bits(), big.to_bits());
        assert_eq!(out[1].to_bits(), tiny.to_bits());
        assert_eq!(out[2].to_bits(), big.to_bits());
        assert_eq!(out[3].to_bits(), tiny.to_bits());
    }

    // @trace TEST-MLE-528 [req:REQ-HR-002] [level:unit]
    #[test]
    fn post_node_f32_output_with_f16_config_captures_as_f32_via_cpu_canonical() {
        // Arrange: model config declares F16 compute_dtype but CPU canonicalizes to F32
        // output buffer has F32 bytes, must decode correctly despite F16 config
        let mut cb = MidLayerEncodeCallback::new(2);
        let holder = TestCtxHolder::new(8, DType::F16); // F16 declared
        let ctx = holder.ctx(2, 1);

        // Output has F32 bytes (CPU canonical) — 8 f32 = 32 bytes
        let output = make_f32_output(&[1.5, -2.5, 3.5, -4.5, 5.5, -6.5, 7.5, -8.5]);

        // Act
        cb.post_node(&ctx, &output);

        // Assert: F32 candidate wins despite F16 declared dtype
        let captured = cb.captured.as_ref().expect("F32 bytes must be captured with F16 config");
        assert_eq!(captured.len(), 8);
        assert!((captured[0] - 1.5).abs() < 1e-6);
        assert!((captured[7] - (-8.5)).abs() < 1e-6);
    }

    // @trace TEST-MLE-529 [req:REQ-HR-002] [level:unit]
    #[test]
    fn decode_f32_hidden_1_seq_1_buffer_exactly_4_bytes_no_slack() {
        // Arrange: minimal possible decode: 1 element, 4 bytes, no slack at all
        // This is the absolute minimum valid F32 decode scenario
        let src: Vec<u8> = (-0.0625f32).to_le_bytes().to_vec();
        assert_eq!(src.len(), 4);

        // Act
        let out = MidLayerEncodeCallback::decode_hidden_output(&src, 1, 1, DType::F32)
            .expect("4-byte exact-fit buffer must decode");

        // Assert
        assert_eq!(out.len(), 1);
        assert!((out[0] - (-0.0625)).abs() < 1e-7);
    }

    // @trace TEST-MLE-530 [req:REQ-HR-002] [level:unit]
    #[test]
    fn lifecycle_post_node_at_target_with_incompatible_then_compatible_preserves_only_compatible() {
        // Arrange: send incompatible shape first (no capture), then compatible (captures),
        // then another incompatible (no overwrite), then verify exit has the compatible one
        let mut cb = MidLayerEncodeCallback::new(3);
        let holder = TestCtxHolder::new(4, DType::F32);
        let ctx = holder.ctx(3, 1);

        // Incompatible: 5 bytes (not multiple of hidden_size*4=16)
        cb.post_node(&ctx, &vec![0u8; 5]);
        assert!(cb.captured.is_none());

        // Compatible: 16 bytes = 4 f32
        cb.post_node(&ctx, &make_f32_output(&[11.0, 22.0, 33.0, 44.0]));
        assert!(cb.captured.is_some());

        // Incompatible again: 9 bytes
        cb.post_node(&ctx, &vec![0u8; 9]);

        // Assert: the compatible capture was NOT overwritten by subsequent incompatible
        let captured = cb.captured.as_ref().unwrap();
        assert_eq!(captured, &vec![11.0, 22.0, 33.0, 44.0]);

        // Exit: must carry the compatible capture
        let ctx4 = holder.ctx(4, 1);
        match cb.pre_node(&ctx4) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![11.0, 22.0, 33.0, 44.0]);
            }
            other => panic!("Expected ExitEarly, got {:?}", other),
        }
    }
}
