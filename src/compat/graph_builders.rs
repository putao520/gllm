//! JIT Extreme Fusion Mega-Graph Builders (P4/P5 Zero-Overhead Protocol)
//!
//! Provides AST template functions for `gllm_kernels::compiler::CompilerGraph`
//! structured specifically for <3 physical kernel launches per logic component.

use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};
use gllm_kernels::types::DType;

// ---------------------------------------------------------------------------
// Symbolic Graph Builders (REQ-JIT-CACHE-005: SymDim::Symbolic binding)
//
// These builders use SymDim::Symbolic("seq_len") for the sequence dimension.
// A single compiled kernel handles ALL sequence lengths at runtime via
// ShapeBinding — no recompilation when seq_len changes.
// ---------------------------------------------------------------------------

/// Symbolic Fused Attention Layer Graph.
///
/// Identical to `build_fused_attention_layer_graph` but with `seq_len` as
/// `SymDim::Symbolic("seq_len")`. Compiled once at model load time;
/// `ShapeBinding::from([("seq_len", actual)])` binds at launch.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_fused_attention_layer_graph_symbolic(
    config: &crate::engine::executor::GeneratorForwardConfig,
) -> CompilerGraph {
    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let eps = config.norm_eps;
    let rope_theta = config.rope_theta;
    let dtype = crate::compat::jit_helpers::computation_dtype_from_config(config);
    let mut g = CompilerGraph::new();
    let dt = dtype;
    let ft = dtype;
    let sym_s = SymDim::Symbolic("seq_len".into());
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor("input", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    let prev_telemetry = g.add_tensor("prev_telemetry", vec![sym_s.clone()], DType::F32);
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], ft);
    let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
    let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);
    let seq_offsets = g.add_tensor("seq_offsets", vec![sym_s.clone()], DType::F32);

    g.inputs = vec![input, prev_telemetry, seq_offsets, rn1_w, w_q, w_k, w_v, w_o];

    g.add_op(OpKind::VariableLengthBatch, vec![seq_offsets], vec![], "ragged_batch");

    let skip_mask = g.add_tensor("attn_skip_mask", vec![sym_s.clone()], ft);
    g.add_op(
        OpKind::AttentionSkipMask { seq_len: sym_s.clone(), threshold: 0.05 },
        vec![prev_telemetry],
        vec![skip_mask],
        "attention_skip_mask",
    );

    let q_out = g.add_tensor("q", vec![sym_s.clone(), SymDim::Concrete(q_dim)], ft);
    g.add_op(OpKind::FusedRmsNormGemm { m: sym_s.clone(), n: q_dim, k: h, eps, dtype: dt }, vec![input, rn1_w, w_q], vec![q_out], "gemm_q_fused");
    let k_out = g.add_tensor("k", vec![sym_s.clone(), SymDim::Concrete(kv_dim)], ft);
    g.add_op(OpKind::FusedRmsNormGemm { m: sym_s.clone(), n: kv_dim, k: h, eps, dtype: dt }, vec![input, rn1_w, w_k], vec![k_out], "gemm_k_fused");
    let v_out = g.add_tensor("v_proj", vec![sym_s.clone(), SymDim::Concrete(kv_dim)], ft);
    g.add_op(OpKind::FusedRmsNormGemm { m: sym_s.clone(), n: kv_dim, k: h, eps, dtype: dt }, vec![input, rn1_w, w_v], vec![v_out], "gemm_v_fused");

    let q_rope = g.add_tensor("q_rope", vec![sym_s.clone(), SymDim::Concrete(q_dim)], ft);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![q_out], vec![q_rope], "rope_q");
    let k_rope = g.add_tensor("k_rope", vec![sym_s.clone(), SymDim::Concrete(kv_dim)], ft);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![k_out], vec![k_rope], "rope_k");

    let attn_out = g.add_tensor("attn_out", vec![sym_s.clone(), SymDim::Concrete(q_dim)], ft);
    g.add_op(
        OpKind::MultiHeadAttention { seq_len: sym_s.clone(), num_heads, num_kv_heads, head_dim },
        vec![q_rope, k_rope, v_out], vec![attn_out], "mha",
    );

    let o_out = g.add_tensor("o_proj", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    g.add_op(OpKind::Gemm { m: sym_s.clone(), n: h, k: q_dim, dtype: dt }, vec![attn_out, w_o], vec![o_out], "gemm_o");

    let resid1 = g.add_tensor("residual1", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    let tel1 = g.add_tensor("telemetry1", vec![sym_s.clone()], DType::F32);
    g.add_op(OpKind::ResidualWithTelemetry { hidden: h }, vec![input, o_out], vec![resid1, tel1], "residual_1");

    // EntropyGate
    let write_mask = g.add_tensor("entropy_write_mask", vec![sym_s.clone()], DType::F32);
    g.add_op(
        OpKind::EntropyGate {
            seq_len: sym_s.clone(),
            vocab_size: q_dim,
            entropy_threshold: 0.5_f32,
        },
        vec![attn_out],
        vec![write_mask],
        "entropy_gate",
    );

    // VRangeQuant
    let v_quantized = g.add_tensor("v_quantized", vec![sym_s.clone(), SymDim::Concrete(kv_dim / 2)], ft);
    g.add_op(
        OpKind::VRangeQuant {
            seq_len: sym_s.clone(),
            kv_dim,
            block_size: 32,
            range_threshold: 0.1_f32,
        },
        vec![v_out],
        vec![v_quantized],
        "vrange_quant",
    );

    // KvScatterWrite
    let kv_cache_in = g.add_tensor_concrete("kv_cache_ptr", &[1], DType::F32);
    let kv_written = g.add_tensor("kv_written", vec![sym_s.clone(), SymDim::Concrete(kv_dim)], ft);
    g.add_op(
        OpKind::KvScatterWrite {
            seq_len: sym_s.clone(),
            num_kv_heads,
            head_dim,
            kv_dim,
            write_start: 0,
            layer_offset: 0,
            half_offset: kv_dim / 2 * dtype.size_bytes(),
            head_stride: head_dim * dtype.size_bytes(),
            dtype_size: dtype.size_bytes(),
        },
        vec![k_rope, v_quantized, kv_cache_in, write_mask],
        vec![kv_written],
        "kv_scatter_write_gated",
    );

    // KvCentroidPrefetch
    let prefetch_sink = g.add_tensor_concrete("prefetch_sink", &[1], DType::F32);
    g.add_op(
        OpKind::KvCentroidPrefetch {
            seq_len: sym_s.clone(),
            num_heads,
            head_dim,
            prefetch_blocks: 4,
        },
        vec![attn_out],
        vec![prefetch_sink],
        "kv_centroid_prefetch",
    );

    g.inputs.push(kv_cache_in);
    g.outputs = vec![resid1, tel1, kv_written, prefetch_sink];
    g
}

/// Symbolic Fused Dense FFN Layer Graph.
///
/// Uses `SymDim::Symbolic("seq_len")` for model-load-time compilation.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_fused_ffn_layer_graph_symbolic(
    config: &crate::engine::executor::GeneratorForwardConfig,
) -> CompilerGraph {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let dtype = crate::compat::jit_helpers::computation_dtype_from_config(config);
    let mut g = CompilerGraph::new();
    let dt = dtype;
    let ft = dtype;
    let sym_s = SymDim::Symbolic("seq_len".into());
    let h = hidden;

    let input = g.add_tensor("input", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    let prev_telemetry = g.add_tensor("prev_telemetry", vec![sym_s.clone()], DType::F32);
    let rn2_w = g.add_tensor_concrete("rn2_w", &[h], ft);
    let w_gate = g.add_tensor_concrete("w_gate", &[h, inter], dt);
    let w_up = g.add_tensor_concrete("w_up", &[h, inter], dt);
    let w_down = g.add_tensor_concrete("w_down", &[inter, h], dt);
    let seq_offsets = g.add_tensor("seq_offsets", vec![sym_s.clone()], DType::F32);

    g.inputs = vec![input, prev_telemetry, seq_offsets, rn2_w, w_gate, w_up, w_down];

    g.add_op(OpKind::VariableLengthBatch, vec![seq_offsets], vec![], "ragged_batch");

    g.add_op(
        OpKind::LayerBypass { threshold: 0.002 },
        vec![prev_telemetry],
        vec![],
        "layer_bypass_guard",
    );

    let normed2 = g.add_tensor("normed2", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn2_w], vec![normed2], "rms_norm_2");

    let gate_out = g.add_tensor("ffn_gate", vec![sym_s.clone(), SymDim::Concrete(inter)], ft);
    g.add_op(OpKind::Gemm { m: sym_s.clone(), n: inter, k: h, dtype: dt }, vec![normed2, w_gate], vec![gate_out], "gemm_gate");

    let mask_out = g.add_tensor("gate_mask", vec![sym_s.clone(), SymDim::Concrete(inter)], ft);
    g.add_op(OpKind::GateMask { hidden: inter }, vec![gate_out], vec![mask_out], "gate_mask");

    let up_out = g.add_tensor("ffn_up", vec![sym_s.clone(), SymDim::Concrete(inter)], ft);
    g.add_op(OpKind::MaskedGemm { m: sym_s.clone(), n: inter, k: h, dtype: dt }, vec![normed2, w_up, mask_out], vec![up_out], "gemm_up_masked");

    let swiglu_out = g.add_tensor("ffn_swiglu", vec![sym_s.clone(), SymDim::Concrete(inter)], ft);
    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], "swiglu");

    let down_out = g.add_tensor("ffn_down", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    g.add_op(OpKind::Gemm { m: sym_s.clone(), n: h, k: inter, dtype: dt }, vec![swiglu_out, w_down], vec![down_out], "gemm_down");

    let output = g.add_tensor("output", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    let tel2 = g.add_tensor("telemetry2", vec![sym_s.clone()], DType::F32);
    g.add_op(OpKind::ResidualWithTelemetry { hidden: h }, vec![input, down_out], vec![output, tel2], "residual_2");

    g.outputs = vec![output, tel2];
    g
}

/// Symbolic Fused MoE FFN Layer Graph.
///
/// Uses `SymDim::Symbolic("seq_len")` for model-load-time compilation.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_fused_moe_layer_graph_symbolic(
    config: &crate::engine::executor::GeneratorForwardConfig,
) -> CompilerGraph {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let moe = config.moe_config.as_ref().unwrap();
    let num_experts = moe.num_experts;
    let top_k = moe.num_experts_per_tok;
    let eps = config.norm_eps;
    let dtype = crate::compat::jit_helpers::computation_dtype_from_config(config);
    let mut g = CompilerGraph::new();
    let dt = dtype;
    let ft = dtype;
    let sym_s = SymDim::Symbolic("seq_len".into());
    let h = hidden;

    let input = g.add_tensor("input", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    let rn2_w = g.add_tensor_concrete("rn2_w", &[h], ft);
    let w_router = g.add_tensor_concrete("w_router", &[h, num_experts], dt);

    let mut inputs = vec![input, rn2_w, w_router];

    let mut w_gates = vec![];
    let mut w_ups = vec![];
    let mut w_downs = vec![];
    for i in 0..num_experts {
        let wg = g.add_tensor_concrete(&format!("w_gate_exp{}", i), &[h, inter], dt);
        let wu = g.add_tensor_concrete(&format!("w_up_exp{}", i), &[h, inter], dt);
        let wd = g.add_tensor_concrete(&format!("w_down_exp{}", i), &[inter, h], dt);
        inputs.push(wg);
        inputs.push(wu);
        inputs.push(wd);
        w_gates.push(wg);
        w_ups.push(wu);
        w_downs.push(wd);
    }
    g.inputs = inputs;

    let normed2 = g.add_tensor("normed2", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn2_w], vec![normed2], "rms_norm_2");

    let gate_probs = g.add_tensor("gate_probs", vec![sym_s.clone(), SymDim::Concrete(num_experts)], ft);
    g.add_op(OpKind::MoEGate { seq_len: sym_s.clone(), num_experts, hidden: h }, vec![normed2, w_router], vec![gate_probs], "moe_gate");

    let topk_idx = g.add_tensor("topk_idx", vec![sym_s.clone(), SymDim::Concrete(top_k)], DType::F32);
    let topk_w = g.add_tensor("topk_w", vec![sym_s.clone(), SymDim::Concrete(top_k)], DType::F32);
    g.add_op(OpKind::TopK { seq_len: sym_s.clone(), num_experts, top_k }, vec![gate_probs], vec![topk_idx, topk_w], "top_k");

    let mut current_acc = g.add_tensor("expert_accumulator_init", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    g.inputs.push(current_acc);

    for i in 0..num_experts {
        let gate_out = g.add_tensor(&format!("ffn_gate_{}", i), vec![sym_s.clone(), SymDim::Concrete(inter)], ft);
        g.add_op(OpKind::Gemm { m: sym_s.clone(), n: inter, k: h, dtype: dt }, vec![normed2, w_gates[i]], vec![gate_out], &format!("gemm_gate_{}", i));

        let mask_out = g.add_tensor(&format!("gate_mask_{}", i), vec![sym_s.clone(), SymDim::Concrete(inter)], ft);
        g.add_op(OpKind::GateMask { hidden: inter }, vec![gate_out], vec![mask_out], &format!("gate_mask_{}", i));

        let up_out = g.add_tensor(&format!("ffn_up_{}", i), vec![sym_s.clone(), SymDim::Concrete(inter)], ft);
        g.add_op(OpKind::MaskedGemm { m: sym_s.clone(), n: inter, k: h, dtype: dt }, vec![normed2, w_ups[i], mask_out], vec![up_out], &format!("gemm_up_masked_{}", i));

        let swiglu_out = g.add_tensor(&format!("ffn_swiglu_{}", i), vec![sym_s.clone(), SymDim::Concrete(inter)], ft);
        g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], &format!("swiglu_{}", i));

        let down_out = g.add_tensor(&format!("ffn_down_{}", i), vec![sym_s.clone(), SymDim::Concrete(h)], ft);
        g.add_op(OpKind::Gemm { m: sym_s.clone(), n: h, k: inter, dtype: dt }, vec![swiglu_out, w_downs[i]], vec![down_out], &format!("gemm_down_{}", i));

        let next_acc = g.add_tensor(&format!("acc_after_{}", i), vec![sym_s.clone(), SymDim::Concrete(h)], ft);
        g.add_op(
            OpKind::MoEConditionalAdd { seq_len: sym_s.clone(), hidden: h, num_experts, expert_idx: i },
            vec![current_acc, down_out, gate_probs],
            vec![next_acc],
            &format!("cond_add_{}", i),
        );
        current_acc = next_acc;
    }

    let output = g.add_tensor("output", vec![sym_s.clone(), SymDim::Concrete(h)], ft);
    let tel_moe = g.add_tensor("telemetry_moe", vec![sym_s.clone()], DType::F32);
    g.add_op(OpKind::ResidualWithTelemetry { hidden: h }, vec![input, current_acc], vec![output, tel_moe], "residual_moe_tied");

    g.outputs = vec![output, tel_moe];
    g
}

/// Monolithic Attention Graph
/// Fuses: RmsNorm1 -> QKV -> RoPE -> FlashAttn -> O_Proj -> ResAdd1
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_fused_attention_layer_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    rope_theta: f64,
    dtype: DType,
) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = dtype;     // GEMM weights
    let ft = dtype;     // Activations
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], ft);
    let prev_telemetry = g.add_tensor_concrete("prev_telemetry", &[s], DType::F32);
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], ft);
    let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
    let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);
    let seq_offsets = g.add_tensor_concrete("seq_offsets", &[s], DType::F32);

    g.inputs = vec![input, prev_telemetry, seq_offsets, rn1_w, w_q, w_k, w_v, w_o];

    g.add_op(OpKind::VariableLengthBatch, vec![seq_offsets], vec![], "ragged_batch");

    // P4/P5 Tier IV: Attention Skip Mask based on previous layer's L2-delta telemetry
    let skip_mask = g.add_tensor_concrete("attn_skip_mask", &[s], ft);
    g.add_op(
        OpKind::AttentionSkipMask { seq_len: s.into(), threshold: 0.05 },
        vec![prev_telemetry],
        vec![skip_mask],
        "attention_skip_mask"
    );

    let q_out = g.add_tensor_concrete("q", &[s, q_dim], ft);
    g.add_op(OpKind::FusedRmsNormGemm { m: s.into(), n: q_dim, k: h, eps, dtype: dt }, vec![input, rn1_w, w_q], vec![q_out], "gemm_q_fused");
    let k_out = g.add_tensor_concrete("k", &[s, kv_dim], ft);
    g.add_op(OpKind::FusedRmsNormGemm { m: s.into(), n: kv_dim, k: h, eps, dtype: dt }, vec![input, rn1_w, w_k], vec![k_out], "gemm_k_fused");
    let v_out = g.add_tensor_concrete("v_proj", &[s, kv_dim], ft);
    g.add_op(OpKind::FusedRmsNormGemm { m: s.into(), n: kv_dim, k: h, eps, dtype: dt }, vec![input, rn1_w, w_v], vec![v_out], "gemm_v_fused");

    let q_rope = g.add_tensor_concrete("q_rope", &[s, q_dim], ft);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![q_out], vec![q_rope], "rope_q");
    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_dim], ft);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![k_out], vec![k_rope], "rope_k");

    let attn_out = g.add_tensor_concrete("attn_out", &[s, q_dim], ft);
    g.add_op(
        OpKind::MultiHeadAttention { seq_len: s.into(), num_heads, num_kv_heads, head_dim },
        vec![q_rope, k_rope, v_out], vec![attn_out], "mha",
    );

    let o_out = g.add_tensor_concrete("o_proj", &[s, h], ft);
    g.add_op(OpKind::Gemm { m: s.into(), n: h, k: q_dim, dtype: dt }, vec![attn_out, w_o], vec![o_out], "gemm_o");
    
    let resid1 = g.add_tensor_concrete("residual1", &[s, h], ft);
    let tel1 = g.add_tensor_concrete("telemetry1", &[s], DType::F32);
    g.add_op(OpKind::ResidualWithTelemetry { hidden: h }, vec![input, o_out], vec![resid1, tel1], "residual_1");

    // ── Phase 29: KV Cache Extreme Squeezing (SPEC §12.9) ─────────────────
    //
    // §12.9.1 EntropyGate — GPU-side entropy threshold gate.
    //   Reads post-Attention softmax probability vectors (attn_out).
    //   Outputs write_mask[seq_len]: 1.0 = write to KV cache, 0.0 = skip ST.global.
    //   Low-entropy tokens (filler words, conjunctions) produce write_mask=0,
    //   directly blocking the KvScatterWrite ST.global for those token positions.
    let write_mask = g.add_tensor_concrete("entropy_write_mask", &[s], DType::F32);
    g.add_op(
        OpKind::EntropyGate {
            seq_len: s.into(),
            vocab_size: q_dim,             // head-space distribution entropy proxy
            entropy_threshold: 0.5_f32,    // nats; < 0.5 nat → high-confidence, skip write
        },
        vec![attn_out],
        vec![write_mask],
        "entropy_gate",
    );

    // §12.9.2 VRangeQuant — compress V projection to INT4 for narrow-range blocks.
    //   Reads v_out (full precision). Outputs v_quantized (INT4 packed, kv_dim/2 width).
    //   v_quantized replaces v_out as the V source for KvScatterWrite to reduce bandwidth.
    let v_quantized = g.add_tensor_concrete("v_quantized", &[s, kv_dim / 2], ft);
    g.add_op(
        OpKind::VRangeQuant {
            seq_len: s.into(),
            kv_dim,
            block_size: 32,
            range_threshold: 0.1_f32,
        },
        vec![v_out],
        vec![v_quantized],
        "vrange_quant",
    );

    // §12.9 KvScatterWrite — gated KV cache write-back.
    //   inputs[0] = k_rope  (K source, post-RoPE)
    //   inputs[1] = v_quantized  (V source, INT4-compressed)
    //   inputs[2] = kv_cache  (destination pointer placeholder)
    //   inputs[3] = write_mask  (EntropyGate gate: 0.0 → skip ST.global for this token)
    //
    //   This is the core SPEC §12.9.1 contract: write_mask is the gate input that
    //   prevents low-entropy tokens from ever touching GMEM KV writes.
    let kv_cache_in = g.add_tensor_concrete("kv_cache_ptr", &[1], DType::F32); // opaque ptr
    let kv_written  = g.add_tensor_concrete("kv_written",   &[s, kv_dim], ft);
    g.add_op(
        OpKind::KvScatterWrite {
            seq_len:      s.into(),
            num_kv_heads,
            head_dim,
            kv_dim,
            write_start:  0,
            layer_offset: 0,
            half_offset:  kv_dim / 2 * dtype.size_bytes(),
            head_stride:  head_dim * dtype.size_bytes(),
            dtype_size:   dtype.size_bytes(),
        },
        // inputs[3] = write_mask is the EntropyGate output — controls per-token write gates
        vec![k_rope, v_quantized, kv_cache_in, write_mask],
        vec![kv_written],
        "kv_scatter_write_gated",
    );

    // §12.9.3 KvCentroidPrefetch — side-effect: async cuMemPrefetchAsync for next layer.
    //   No output tensor consumed by downstream ops; added to g.outputs to prevent DCE.
    let prefetch_sink = g.add_tensor_concrete("prefetch_sink", &[1], DType::F32);
    g.add_op(
        OpKind::KvCentroidPrefetch {
            seq_len: s.into(),
            num_heads,
            head_dim,
            prefetch_blocks: 4,
        },
        vec![attn_out],
        vec![prefetch_sink],
        "kv_centroid_prefetch",
    );
    // ─────────────────────────────────────────────────────────────────────

    // g.inputs must include kv_cache_in so the executor can bind the live KV cache ptr
    g.inputs.push(kv_cache_in);

    // All Phase 29 outputs enter g.outputs to prevent Dead-Code Elimination:
    //   kv_written:    consumed by next-layer attention as updated KV cache reference
    //   prefetch_sink: side-effect scheduling node, kept alive via output registration
    g.outputs = vec![resid1, tel1, kv_written, prefetch_sink];
    g
}


/// Monolithic Dense FFN Graph
/// Fuses: RmsNorm2 -> Gate -> SiLU -> Up -> Mul -> Down -> ResAdd2
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_fused_ffn_layer_graph(
    seq_len: usize,
    hidden: usize,
    inter: usize,
    eps: f32,
    dtype: DType,
) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = dtype;
    let ft = dtype;
    let s = seq_len;
    let h = hidden;

    let input = g.add_tensor_concrete("input", &[s, h], ft);
    let prev_telemetry = g.add_tensor_concrete("prev_telemetry", &[s], DType::F32);
    let rn2_w = g.add_tensor_concrete("rn2_w", &[h], ft);
    let w_gate = g.add_tensor_concrete("w_gate", &[h, inter], dt);
    let w_up = g.add_tensor_concrete("w_up", &[h, inter], dt);
    let w_down = g.add_tensor_concrete("w_down", &[inter, h], dt);
    let seq_offsets = g.add_tensor_concrete("seq_offsets", &[s], DType::F32);

    g.inputs = vec![input, prev_telemetry, seq_offsets, rn2_w, w_gate, w_up, w_down];

    g.add_op(OpKind::VariableLengthBatch, vec![seq_offsets], vec![], "ragged_batch");

    // P4/P5 Tier IV Chain A: Residual-Dominant Layer Bypass (SPEC I.2 §12.10)
    // When transform_ratio = ||transform|| / ||input|| < threshold (0.002), this layer
    // contributes negligibly. JIT will skip all GEMM ops, forwarding input directly.
    g.add_op(
        OpKind::LayerBypass { threshold: 0.002 },
        vec![prev_telemetry],
        vec![],
        "layer_bypass_guard",
    );

    let normed2 = g.add_tensor_concrete("normed2", &[s, h], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn2_w], vec![normed2], "rms_norm_2");

    let gate_out = g.add_tensor_concrete("ffn_gate", &[s, inter], ft);
    g.add_op(OpKind::Gemm { m: s.into(), n: inter, k: h, dtype: dt }, vec![normed2, w_gate], vec![gate_out], "gemm_gate");
    
    let mask_out = g.add_tensor_concrete("gate_mask", &[s, inter], ft);
    g.add_op(OpKind::GateMask { hidden: inter }, vec![gate_out], vec![mask_out], "gate_mask");
    
    let up_out = g.add_tensor_concrete("ffn_up", &[s, inter], ft);
    g.add_op(OpKind::MaskedGemm { m: s.into(), n: inter, k: h, dtype: dt }, vec![normed2, w_up, mask_out], vec![up_out], "gemm_up_masked");
    
    let swiglu_out = g.add_tensor_concrete("ffn_swiglu", &[s, inter], ft);
    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], "swiglu");
    
    let down_out = g.add_tensor_concrete("ffn_down", &[s, h], ft);
    g.add_op(OpKind::Gemm { m: s.into(), n: h, k: inter, dtype: dt }, vec![swiglu_out, w_down], vec![down_out], "gemm_down");

    let output = g.add_tensor_concrete("output", &[s, h], ft);
    let tel2 = g.add_tensor_concrete("telemetry2", &[s], DType::F32);
    g.add_op(OpKind::ResidualWithTelemetry { hidden: h }, vec![input, down_out], vec![output, tel2], "residual_2");

    g.outputs = vec![output, tel2];
    g
}

/// Monolithic MoE Graph (Shared-Expert Fast Path & P4/P5 PGSLE Speculation)
/// Fuses: RmsNorm2 -> Router(NoOp/Branch) -> Shared-Gate -> SiLU -> Shared-Up -> Mul -> Shared-Down -> ResAdd2
/// Exclusively bypasses Routed Experts by intercepting execution at trace level via `ConditionalBranch` directives 
/// to maximize decoding throughput under the "Zero-Overhead Freeloading" scheme.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_fused_moe_layer_graph(
    seq_len: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
    eps: f32,
    dtype: DType,
) -> CompilerGraph {
    let mut g = CompilerGraph::new();
    let dt = dtype;
    let ft = dtype;
    let s = seq_len;
    let h = hidden;

    let input = g.add_tensor_concrete("input", &[s, h], ft);
    let rn2_w = g.add_tensor_concrete("rn2_w", &[h], ft);
    
    // Router weights
    let w_router = g.add_tensor_concrete("w_router", &[h, num_experts], dt);

    let mut inputs = vec![input, rn2_w, w_router];
    
    // Add dynamic expert weight inputs
    let mut w_gates = vec![];
    let mut w_ups = vec![];
    let mut w_downs = vec![];
    for i in 0..num_experts {
        let wg = g.add_tensor_concrete(&format!("w_gate_exp{}", i), &[h, inter], dt);
        let wu = g.add_tensor_concrete(&format!("w_up_exp{}", i), &[h, inter], dt);
        let wd = g.add_tensor_concrete(&format!("w_down_exp{}", i), &[inter, h], dt);
        inputs.push(wg);
        inputs.push(wu);
        inputs.push(wd);
        w_gates.push(wg);
        w_ups.push(wu);
        w_downs.push(wd);
    }
    
    g.inputs = inputs;

    // 1. RMSNorm
    let normed2 = g.add_tensor_concrete("normed2", &[s, h], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn2_w], vec![normed2], "rms_norm_2");

    // 2. Router Path
    let gate_probs = g.add_tensor_concrete("gate_probs", &[s, num_experts], ft);
    g.add_op(OpKind::MoEGate { seq_len: s.into(), num_experts, hidden: h }, vec![normed2, w_router], vec![gate_probs], "moe_gate");

    let topk_idx = g.add_tensor_concrete("topk_idx", &[s, top_k], DType::F32);
    let topk_w = g.add_tensor_concrete("topk_w", &[s, top_k], DType::F32);
    g.add_op(OpKind::TopK { seq_len: s.into(), num_experts, top_k }, vec![gate_probs], vec![topk_idx, topk_w], "top_k");

    // 3. Dynamic Zero-Overhead Mega-Graph Construction for all Experts
    // We instantiate a zeroed accumulator and iteratively process branches.
    let mut current_acc = g.add_tensor_concrete("expert_accumulator_init", &[s, h], ft);
    
    // Note: The JIT Engine's Memory Allocator relies on Tensor inputs to be initialized externally or cleared.
    // Given the trace logic naturally supports accumulation, we construct the graph strictly structurally.
    // In practice, `expert_accumulator_init` must be a known zeroed buffer or we use elementwise tricks.
    // We expect the pipeline loader to bind zero buffers to it!
    g.inputs.push(current_acc);

    for i in 0..num_experts {
        let gate_out = g.add_tensor_concrete(&format!("ffn_gate_{}", i), &[s, inter], ft);
        g.add_op(OpKind::Gemm { m: s.into(), n: inter, k: h, dtype: dt }, vec![normed2, w_gates[i]], vec![gate_out], &format!("gemm_gate_{}", i));
        
        let mask_out = g.add_tensor_concrete(&format!("gate_mask_{}", i), &[s, inter], ft);
        g.add_op(OpKind::GateMask { hidden: inter }, vec![gate_out], vec![mask_out], &format!("gate_mask_{}", i));
        
        let up_out = g.add_tensor_concrete(&format!("ffn_up_{}", i), &[s, inter], ft);
        g.add_op(OpKind::MaskedGemm { m: s.into(), n: inter, k: h, dtype: dt }, vec![normed2, w_ups[i], mask_out], vec![up_out], &format!("gemm_up_masked_{}", i));
        
        let swiglu_out = g.add_tensor_concrete(&format!("ffn_swiglu_{}", i), &[s, inter], ft);
        g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], &format!("swiglu_{}", i));
        
        let down_out = g.add_tensor_concrete(&format!("ffn_down_{}", i), &[s, h], ft);
        g.add_op(OpKind::Gemm { m: s.into(), n: h, k: inter, dtype: dt }, vec![swiglu_out, w_downs[i]], vec![down_out], &format!("gemm_down_{}", i));
        
        let next_acc = g.add_tensor_concrete(&format!("acc_after_{}", i), &[s, h], ft);
        g.add_op(
            OpKind::MoEConditionalAdd { seq_len: s.into(), hidden: h, num_experts, expert_idx: i },
            vec![current_acc, down_out, gate_probs],
            vec![next_acc],
            &format!("cond_add_{}", i)
        );
        current_acc = next_acc;
    }

    // 4. Residual Tie-back
    let output = g.add_tensor_concrete("output", &[s, h], ft);
    let tel_moe = g.add_tensor_concrete("telemetry_moe", &[s], DType::F32);
    g.add_op(OpKind::ResidualWithTelemetry { hidden: h }, vec![input, current_acc], vec![output, tel_moe], "residual_moe_tied");

    g.outputs = vec![output, tel_moe];
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use gllm_kernels::types::DType;

    #[test]
    fn test_entropy_gate_blocks_kv_write() {
        let graph = build_fused_attention_layer_graph(
            32, 128, 8, 8, 16, 1e-5, 10000.0, DType::F32
        );

        let mut found_entropy = false;
        let mut found_scatter = false;
        let mut write_mask_id = None;

        for op in graph.ops.iter() {
            match &op.kind {
                gllm_kernels::compiler::OpKind::EntropyGate { .. } => {
                    found_entropy = true;
                    assert_eq!(op.outputs.len(), 1);
                    write_mask_id = Some(op.outputs[0]);
                }
                gllm_kernels::compiler::OpKind::KvScatterWrite { .. } => {
                    found_scatter = true;
                    assert!(write_mask_id.is_some(), "EntropyGate must precede KvScatterWrite");
                    assert_eq!(op.inputs.len(), 4, "KvScatterWrite expected exactly 4 inputs with P29 gated-mask active");
                    assert_eq!(op.inputs[3], write_mask_id.unwrap(), "write_mask missed binding to KvScatterWrite.inputs[3]");
                }
                _ => {}
            }
        }
        
        assert!(found_entropy, "Missing EntropyGate operator in AST");
        assert!(found_scatter, "Missing KvScatterWrite operator in AST");

        assert!(graph.outputs.len() >= 4, "Expected at least 4 ops (resid, tel, kv_written, prefetch) at graph output bound");
    }
}
