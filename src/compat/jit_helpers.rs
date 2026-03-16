//! JIT graph builders and execution helpers extracted from decoder_forward.rs.
//!
//! These functions build `CompilerGraph` instances and execute compiled layers,
//! shared across CPU decoder forward, embedding forward, and rerank forward paths.

use super::cpu_backend::CpuBackend;
use super::Element;
use crate::engine::executor::{BackendError as BE, KvCacheHandle};

// ---------------------------------------------------------------------------
// Weight packing helper (shared by all JIT execute_* functions)
// ---------------------------------------------------------------------------

/// Pack multiple f32 weight slices into a contiguous byte buffer.
pub(crate) fn pack_weights(slices: &[&[f32]]) -> Vec<u8> {
    let total_bytes: usize = slices.iter().map(|s| s.len() * 4).sum();
    let mut buf = vec![0u8; total_bytes];
    let mut offset = 0;
    for slice in slices {
        let bytes = slice.len() * 4;
        buf[offset..offset + bytes].copy_from_slice(unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes)
        });
        offset += bytes;
    }
    buf
}

// ---------------------------------------------------------------------------
// Decoder layer graph
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for a single decoder layer (pre-norm, RMSNorm + SwiGLU).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_decoder_layer_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
    rope_theta: f64,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor("input", vec![s, h], dt);
    let w_q = g.add_tensor("w_q", vec![h, q_dim], dt);
    let w_k = g.add_tensor("w_k", vec![h, kv_dim], dt);
    let w_v = g.add_tensor("w_v", vec![h, kv_dim], dt);
    let w_o = g.add_tensor("w_o", vec![q_dim, h], dt);
    let rn1_w = g.add_tensor("rn1_w", vec![h], dt);
    let w_gate = g.add_tensor("w_gate", vec![h, inter], dt);
    let w_up = g.add_tensor("w_up", vec![h, inter], dt);
    let w_down = g.add_tensor("w_down", vec![inter, h], dt);
    let rn2_w = g.add_tensor("rn2_w", vec![h], dt);

    g.inputs = vec![
        input, w_q, w_k, w_v, w_o, rn1_w,
        w_gate, w_up, w_down, rn2_w,
    ];

    // Pre-attention RMSNorm
    let normed1 = g.add_tensor("normed1", vec![s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed1], "rms_norm_1");

    // Q/K/V Projections
    let q_out = g.add_tensor("q", vec![s, q_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: q_dim, k: h }, vec![normed1, w_q], vec![q_out], "gemm_q");
    let k_out = g.add_tensor("k", vec![s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h }, vec![normed1, w_k], vec![k_out], "gemm_k");
    let v_out = g.add_tensor("v", vec![s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h }, vec![normed1, w_v], vec![v_out], "gemm_v");

    // RoPE
    let q_rope = g.add_tensor("q_rope", vec![s, q_dim], dt);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![q_out], vec![q_rope], "rope_q");
    let k_rope = g.add_tensor("k_rope", vec![s, kv_dim], dt);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![k_out], vec![k_rope], "rope_k");

    // Multi-Head Attention
    let attn_out = g.add_tensor("attn_out", vec![s, q_dim], dt);
    g.add_op(
        OpKind::MultiHeadAttention { seq_len: s, num_heads, head_dim },
        vec![q_rope, k_rope, v_out], vec![attn_out], "mha",
    );

    // Output projection + Residual 1
    let o_out = g.add_tensor("o_proj", vec![s, h], dt);
    g.add_op(OpKind::Gemm { m: s, n: h, k: q_dim }, vec![attn_out, w_o], vec![o_out], "gemm_o");
    let resid1 = g.add_tensor("residual1", vec![s, h], dt);
    g.add_op(OpKind::Residual, vec![input, o_out], vec![resid1], "residual_1");

    // Pre-FFN RMSNorm
    let normed2 = g.add_tensor("normed2", vec![s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![resid1, rn2_w], vec![normed2], "rms_norm_2");

    // SwiGLU FFN
    let gate_out = g.add_tensor("ffn_gate", vec![s, inter], dt);
    g.add_op(OpKind::Gemm { m: s, n: inter, k: h }, vec![normed2, w_gate], vec![gate_out], "gemm_gate");
    let up_out = g.add_tensor("ffn_up", vec![s, inter], dt);
    g.add_op(OpKind::Gemm { m: s, n: inter, k: h }, vec![normed2, w_up], vec![up_out], "gemm_up");
    let swiglu_out = g.add_tensor("ffn_swiglu", vec![s, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], "swiglu");
    let down_out = g.add_tensor("ffn_down", vec![s, h], dt);
    g.add_op(OpKind::Gemm { m: s, n: h, k: inter }, vec![swiglu_out, w_down], vec![down_out], "gemm_down");

    // Residual 2
    let output = g.add_tensor("output", vec![s, h], dt);
    g.add_op(OpKind::Residual, vec![resid1, down_out], vec![output], "residual_2");

    g.outputs = vec![output];
    g
}

/// Execute a JIT-compiled decoder layer.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_decoder_layer(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    q_w: &[f32], k_w: &[f32], v_w: &[f32], o_w: &[f32], rn1_w: &[f32],
    gate_w: &[f32], up_w: &[f32], down_w: &[f32], rn2_w: &[f32],
    positions: &[u32],
    seq_len: usize,
    output: &mut [f32],
) {
    let weights_buf = pack_weights(&[q_w, k_w, v_w, o_w, rn1_w, gate_w, up_w, down_w, rn2_w]);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
}

// ---------------------------------------------------------------------------
// KV projection graph (for incremental decode KV cache update)
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for KV projection only: RmsNorm → K Gemm → K RoPE.
///
/// V projection is computed via a separate JIT graph (build_v_projection_graph).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_kv_projection_graph(
    seq_len: usize,
    hidden: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    rope_theta: f64,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor("input", vec![s, h], dt);
    let rn1_w = g.add_tensor("rn1_w", vec![h], dt);
    let w_k = g.add_tensor("w_k", vec![h, kv_dim], dt);
    g.inputs = vec![input, rn1_w, w_k];

    let normed = g.add_tensor("normed", vec![s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed], "rms_norm_kv");

    let k_out = g.add_tensor("k", vec![s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h }, vec![normed, w_k], vec![k_out], "gemm_k");

    let k_rope = g.add_tensor("k_rope", vec![s, kv_dim], dt);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![k_out], vec![k_rope], "rope_k");

    g.outputs = vec![k_rope];
    g
}

/// Build a CompilerGraph for V projection only: RmsNorm → V Gemm (no RoPE).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_v_projection_graph(
    seq_len: usize,
    hidden: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor("input", vec![s, h], dt);
    let rn1_w = g.add_tensor("rn1_w", vec![h], dt);
    let w_v = g.add_tensor("w_v", vec![h, kv_dim], dt);
    g.inputs = vec![input, rn1_w, w_v];

    let normed = g.add_tensor("normed", vec![s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed], "rms_norm_v");

    let v_out = g.add_tensor("v_proj", vec![s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h }, vec![normed, w_v], vec![v_out], "gemm_v");

    g.outputs = vec![v_out];
    g
}

/// Execute a JIT-compiled KV projection graph.
///
/// Returns (k_rope, v_proj) both as [seq_len, kv_dim].
/// K uses the pre-compiled graph (RmsNorm → K Gemm → RoPE).
/// V uses a separately compiled graph (RmsNorm → V Gemm).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_kv_projection(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    rn1_w: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    positions: &[u32],
    seq_len: usize,
    hidden: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let kv_dim = num_kv_heads * head_dim;

    // JIT graph for K projection (RmsNorm → Gemm → RoPE)
    let weights_buf = pack_weights(&[rn1_w, k_w]);
    let mut k_rope = vec![0.0f32; seq_len * kv_dim];
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1, seq_len,
            k_rope.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }

    // V projection via JIT: RmsNorm → V Gemm (no RoPE on V)
    let v_graph = build_v_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps);
    let mut v_compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let v_compiled = v_compiler.compile_graph(&v_graph)
        .expect("JIT compile v_projection failed");

    let v_weights_buf = pack_weights(&[rn1_w, v_w]);
    let mut v_proj = vec![0.0f32; seq_len * kv_dim];
    let mut v_scratchpad = vec![0u8; v_compiled.scratchpad_bytes];

    unsafe {
        v_compiled.execute(
            hidden_state.as_ptr() as *const u8,
            v_weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            v_proj.as_mut_ptr() as *mut u8,
            v_scratchpad.as_mut_ptr(),
        );
    }

    (k_rope, v_proj)
}

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------

/// Write pre-computed K/V data into the KV cache buffer.
pub(crate) fn write_kv_to_cache<E: Element>(
    backend: &CpuBackend<E>,
    handle: KvCacheHandle,
    layer: usize,
    k_data: &[f32],
    v_data: &[f32],
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(), BE> {
    let kv_dim = num_kv_heads * head_dim;
    let mut store = backend.kv_store().lock().map_err(|e| {
        BE::Cpu(format!("KV store lock poisoned: {e}"))
    })?;

    let buffer = store.get_mut(&handle.0).ok_or_else(|| {
        BE::Cpu(format!("KV cache handle {} not found", handle.0))
    })?;

    let write_start = if layer == 0 { buffer.seq_len } else { buffer.seq_len.saturating_sub(seq_len) };
    let max_seq = buffer.max_seq_len;

    if write_start + seq_len > max_seq {
        return Err(BE::Cpu(format!(
            "KV cache overflow: write_start={write_start} + seq_len={seq_len} > max_seq_len={max_seq}"
        )));
    }

    for h in 0..num_kv_heads {
        let layer_head_base = (layer * num_kv_heads + h) * max_seq * head_dim;
        for s in 0..seq_len {
            let cache_offset = layer_head_base + (write_start + s) * head_dim;
            let proj_offset = s * kv_dim + h * head_dim;
            buffer.k[cache_offset..cache_offset + head_dim]
                .copy_from_slice(&k_data[proj_offset..proj_offset + head_dim]);
            buffer.v[cache_offset..cache_offset + head_dim]
                .copy_from_slice(&v_data[proj_offset..proj_offset + head_dim]);
        }
    }

    if layer == 0 {
        buffer.seq_len = (buffer.seq_len + seq_len).min(max_seq);
    }

    log::debug!(
        "write_kv_to_cache: layer={layer}, wrote {seq_len} tokens at pos {write_start}, total_seq={}",
        if layer == 0 { write_start + seq_len } else { buffer.seq_len }
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// lm_head graph (final RMSNorm + projection → logits)
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for the final RMSNorm + lm_head projection.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_lm_head_graph(
    seq_len: usize,
    hidden: usize,
    vocab_size: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let input = g.add_tensor("input", vec![seq_len, hidden], dt);
    let norm_w = g.add_tensor("norm_w", vec![hidden], dt);
    let lm_w = g.add_tensor("lm_w", vec![hidden, vocab_size], dt);
    g.inputs = vec![input, norm_w, lm_w];

    let normed = g.add_tensor("normed", vec![seq_len, hidden], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, norm_w], vec![normed], "final_rms_norm");

    let logits = g.add_tensor("logits", vec![seq_len, vocab_size], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n: vocab_size, k: hidden },
        vec![normed, lm_w], vec![logits], "lm_head",
    );

    g.outputs = vec![logits];
    g
}

/// Execute the JIT-compiled lm_head.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_lm_head(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    norm_w: &[f32],
    lm_w: &[f32],
    seq_len: usize,
    output: &mut [f32],
) {
    let weights_buf = pack_weights(&[norm_w, lm_w]);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
}

// ---------------------------------------------------------------------------
// Final norm graph (RMSNorm only, for embedding/rerank forward)
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for final RMSNorm only (no lm_head projection).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_final_norm_graph(
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let input = g.add_tensor("input", vec![seq_len, hidden], dt);
    let norm_w = g.add_tensor("norm_w", vec![hidden], dt);
    g.inputs = vec![input, norm_w];

    let normed = g.add_tensor("normed", vec![seq_len, hidden], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, norm_w], vec![normed], "final_rms_norm");

    g.outputs = vec![normed];
    g
}

/// Execute JIT-compiled final norm (RMSNorm only).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_final_norm(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    norm_w: &[f32],
    seq_len: usize,
    output: &mut [f32],
) {
    let weights_buf = pack_weights(&[norm_w]);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
}

// ---------------------------------------------------------------------------
// MoE pre-attention graph: RmsNorm → Q/K/V Gemm → RoPE(Q) → RoPE(K)
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for MoE pre-attention: RmsNorm → Q/K/V GEMM → RoPE.
/// Output: q_rope[seq_len, q_dim] (via scratchpad segmentation in execute).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_moe_pre_attention_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    rope_theta: f64,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor("input", vec![s, h], dt);
    let rn1_w = g.add_tensor("rn1_w", vec![h], dt);
    let w_q = g.add_tensor("w_q", vec![h, q_dim], dt);
    let w_k = g.add_tensor("w_k", vec![h, kv_dim], dt);
    let w_v = g.add_tensor("w_v", vec![h, kv_dim], dt);
    g.inputs = vec![input, rn1_w, w_q, w_k, w_v];

    let normed = g.add_tensor("normed", vec![s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed], "rms_norm_1");

    let q_out = g.add_tensor("q", vec![s, q_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: q_dim, k: h }, vec![normed, w_q], vec![q_out], "gemm_q");
    let k_out = g.add_tensor("k", vec![s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h }, vec![normed, w_k], vec![k_out], "gemm_k");
    let v_out = g.add_tensor("v", vec![s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h }, vec![normed, w_v], vec![v_out], "gemm_v");

    let q_rope = g.add_tensor("q_rope", vec![s, q_dim], dt);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![q_out], vec![q_rope], "rope_q");
    let k_rope = g.add_tensor("k_rope", vec![s, kv_dim], dt);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![k_out], vec![k_rope], "rope_k");

    // Output: q_rope (primary output). k_rope and v_out are extracted from scratchpad.
    g.outputs = vec![q_rope];
    g
}

/// Execute MoE pre-attention graph. Returns (q_rope, k_rope, v_proj).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[allow(dead_code)]
pub(crate) fn execute_moe_pre_attention(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    rn1_w: &[f32],
    q_w: &[f32], k_w: &[f32], v_w: &[f32],
    positions: &[u32],
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let weights_buf = pack_weights(&[rn1_w, q_w, k_w, v_w]);
    let mut q_rope = vec![0.0f32; seq_len * q_dim];
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1, seq_len,
            q_rope.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }

    // k_rope and v_proj via separate JIT graphs (single-output ABI limitation)
    // K: RmsNorm → K Gemm → RoPE
    let k_graph = build_kv_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps, 10000.0);
    let mut k_compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let k_compiled = k_compiler.compile_graph(&k_graph)
        .expect("JIT compile k_projection failed");
    let k_weights_buf = pack_weights(&[rn1_w, k_w]);
    let mut k_proj = vec![0.0f32; seq_len * kv_dim];
    let mut k_scratch = vec![0u8; k_compiled.scratchpad_bytes];
    unsafe {
        k_compiled.execute(
            hidden_state.as_ptr() as *const u8,
            k_weights_buf.as_ptr(),
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1, seq_len,
            k_proj.as_mut_ptr() as *mut u8,
            k_scratch.as_mut_ptr(),
        );
    }

    // V: RmsNorm → V Gemm (no RoPE)
    let v_graph = build_v_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps);
    let mut v_compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let v_compiled = v_compiler.compile_graph(&v_graph)
        .expect("JIT compile v_projection failed");
    let v_weights_buf = pack_weights(&[rn1_w, v_w]);
    let mut v_proj = vec![0.0f32; seq_len * kv_dim];
    let mut v_scratch = vec![0u8; v_compiled.scratchpad_bytes];
    unsafe {
        v_compiled.execute(
            hidden_state.as_ptr() as *const u8,
            v_weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            v_proj.as_mut_ptr() as *mut u8,
            v_scratch.as_mut_ptr(),
        );
    }

    (q_rope, k_proj, v_proj)
}

// ---------------------------------------------------------------------------
// Post-attention graph: O Gemm → Residual → RmsNorm
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for post-attention: O_proj → Residual → RmsNorm2.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_post_attention_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;

    let attn_out = g.add_tensor("attn_out", vec![s, q_dim], dt);
    let w_o = g.add_tensor("w_o", vec![q_dim, h], dt);
    let residual_in = g.add_tensor("residual_in", vec![s, h], dt);
    let rn2_w = g.add_tensor("rn2_w", vec![h], dt);
    g.inputs = vec![attn_out, w_o, residual_in, rn2_w];

    let o_out = g.add_tensor("o_proj", vec![s, h], dt);
    g.add_op(OpKind::Gemm { m: s, n: h, k: q_dim }, vec![attn_out, w_o], vec![o_out], "gemm_o");

    let resid1 = g.add_tensor("residual1", vec![s, h], dt);
    g.add_op(OpKind::Residual, vec![residual_in, o_out], vec![resid1], "residual_1");

    let normed2 = g.add_tensor("normed2", vec![s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![resid1, rn2_w], vec![normed2], "rms_norm_2");

    g.outputs = vec![normed2];
    g
}

// ---------------------------------------------------------------------------
// Cached GQA Attention graph (for incremental decode)
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for cached GQA attention.
/// Q[seq_len, q_dim] × K_cache[total_seq, kv_dim] → softmax(causal) → × V_cache → out[seq_len, q_dim]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_cached_gqa_graph(
    seq_len: usize,
    total_seq: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q_in = g.add_tensor("q", vec![seq_len, q_dim], dt);
    let k_cache = g.add_tensor("k_cache", vec![total_seq, kv_dim], dt);
    let v_cache = g.add_tensor("v_cache", vec![total_seq, kv_dim], dt);
    g.inputs = vec![q_in, k_cache, v_cache];

    let attn_out = g.add_tensor("attn_out", vec![seq_len, q_dim + 1], dt); // +1 for sparsity
    g.add_op(
        OpKind::CachedGQA { seq_len, total_seq, num_heads, num_kv_heads, head_dim },
        vec![q_in, k_cache, v_cache], vec![attn_out], "cached_gqa",
    );

    g.outputs = vec![attn_out];
    g
}

/// Execute JIT-compiled cached GQA attention. Returns (attn_out, sparsity).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_cached_gqa(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, f32) {
    let q_dim = num_heads * head_dim;
    let out_size = seq_len * q_dim + 1; // +1 for sparsity stat
    let mut output = vec![0.0f32; out_size];
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    let weights_buf = pack_weights(&[k_cache, v_cache]);

    unsafe {
        compiled.execute(
            q.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }

    let sparsity = output[seq_len * q_dim];
    output.truncate(seq_len * q_dim);
    (output, sparsity)
}

// ---------------------------------------------------------------------------
// MoE FFN via JIT (all stages JIT-compiled)
// ---------------------------------------------------------------------------

/// Execute MoE FFN: JIT routing (gate → topk) + JIT expert FFN + JIT weighted combine.
///
/// All stages use JIT-compiled graphs. Zero scalar runtime calls.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_moe_ffn_jit(
    normed2: &[f32],
    router_w: &[f32],
    expert_weights: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    shared_expert: Option<&(Vec<f32>, Vec<f32>, Vec<f32>)>,
    seq_len: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<Vec<f32>, crate::engine::executor::BackendError> {
    use crate::engine::executor::BackendError as BE;

    // Step 1: MoE routing via JIT (MoEGate → TopK)
    let routing_graph = build_moe_routing_graph(seq_len, hidden, num_experts, top_k);
    let mut routing_compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let routing_compiled = routing_compiler.compile_graph(&routing_graph).map_err(|e| {
        BE::Other(format!("MoE routing JIT failed: {e}"))
    })?;

    let routing_weights = pack_weights(&[router_w]);
    // Output: [seq_len * top_k * 2] — first half indices (u32 as f32 bits), second half weights
    let routing_out_size = seq_len * top_k * 2;
    let mut routing_out = vec![0.0f32; routing_out_size];
    let mut routing_scratch = vec![0u8; routing_compiled.scratchpad_bytes];

    unsafe {
        routing_compiled.execute(
            normed2.as_ptr() as *const u8,
            routing_weights.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            routing_out.as_mut_ptr() as *mut u8,
            routing_scratch.as_mut_ptr(),
        );
    }

    // Parse routing output: indices (u32 stored as f32 bits) + weights
    let indices_raw = &routing_out[..seq_len * top_k];
    let weights_raw = &routing_out[seq_len * top_k..];

    let mut indices = vec![0usize; seq_len * top_k];
    let mut weights = vec![0.0f32; seq_len * top_k];
    for i in 0..seq_len * top_k {
        indices[i] = indices_raw[i].to_bits() as usize;
        weights[i] = weights_raw[i];
    }

    // Step 2: Expert FFN via JIT (compile once, execute per expert)
    let ffn_graph = build_expert_ffn_graph(seq_len, hidden, inter);
    let mut ffn_compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let ffn_compiled = ffn_compiler.compile_graph(&ffn_graph).map_err(|e| {
        BE::Other(format!("MoE expert FFN JIT failed: {e}"))
    })?;

    // Collect which experts are needed
    let mut needed_experts: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for &idx in &indices { needed_experts.insert(idx); }

    // Run each needed expert
    let mut expert_outputs: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();
    for &expert_idx in &needed_experts {
        let (ref gate_w, ref up_w, ref down_w) = expert_weights[expert_idx];
        let expert_weight_buf = pack_weights(&[gate_w, up_w, down_w]);
        let mut expert_out = vec![0.0f32; seq_len * hidden];
        let mut scratch = vec![0u8; ffn_compiled.scratchpad_bytes];
        unsafe {
            ffn_compiled.execute(
                normed2.as_ptr() as *const u8,
                expert_weight_buf.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1, seq_len,
                expert_out.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }
        expert_outputs.insert(expert_idx, expert_out);
    }

    // Step 3: Weighted combine via JIT (WeightedSum)
    let combine_graph = build_moe_combine_graph(seq_len, hidden, top_k);
    let mut combine_compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let combine_compiled = combine_compiler.compile_graph(&combine_graph).map_err(|e| {
        BE::Other(format!("MoE combine JIT failed: {e}"))
    })?;

    // Build flat expert_outputs buffer: [top_k, seq_len, hidden]
    // We need to lay out the selected experts' outputs contiguously.
    // The WeightedSum op expects expert_outputs indexed by the indices array.
    // Build a flat buffer with all num_experts slots (only needed ones populated).
    let expert_buf_size = num_experts * seq_len * hidden;
    let mut expert_flat = vec![0.0f32; expert_buf_size];
    for (&expert_idx, expert_out) in &expert_outputs {
        let base = expert_idx * seq_len * hidden;
        expert_flat[base..base + seq_len * hidden].copy_from_slice(expert_out);
    }

    // Prepare indices as f32 bits (already in routing_out format) and weights
    let indices_f32: Vec<f32> = indices.iter().map(|&i| f32::from_bits(i as u32)).collect();

    // Pack: expert_outputs, indices, weights
    let combine_input = pack_weights(&[&expert_flat]);
    let combine_weights = pack_weights(&[&indices_f32, &weights]);
    let mut output = vec![0.0f32; seq_len * hidden];
    let mut combine_scratch = vec![0u8; combine_compiled.scratchpad_bytes];

    unsafe {
        combine_compiled.execute(
            combine_input.as_ptr(),
            combine_weights.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            combine_scratch.as_mut_ptr(),
        );
    }

    // Shared expert (if present)
    if let Some((ref sg, ref su, ref sd)) = shared_expert {
        let shared_buf = pack_weights(&[sg, su, sd]);
        let mut shared_out = vec![0.0f32; seq_len * hidden];
        let mut scratch = vec![0u8; ffn_compiled.scratchpad_bytes];
        unsafe {
            ffn_compiled.execute(
                normed2.as_ptr() as *const u8,
                shared_buf.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1, seq_len,
                shared_out.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }
        for i in 0..seq_len * hidden {
            output[i] += shared_out[i];
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// MoE routing graph: MoEGate → TopK
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for MoE routing: gate → topk.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_moe_routing_graph(
    seq_len: usize,
    hidden: usize,
    num_experts: usize,
    top_k: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;

    let normed2 = g.add_tensor("normed2", vec![s, hidden], dt);
    let router_w = g.add_tensor("router_w", vec![hidden, num_experts], dt);
    g.inputs = vec![normed2, router_w];

    let gate_probs = g.add_tensor("gate_probs", vec![s, num_experts], dt);
    g.add_op(
        OpKind::MoEGate { seq_len: s, num_experts, hidden },
        vec![normed2, router_w], vec![gate_probs], "moe_gate",
    );

    let topk_out = g.add_tensor("topk_out", vec![s, top_k * 2], dt);
    g.add_op(
        OpKind::TopK { seq_len: s, num_experts, top_k },
        vec![gate_probs], vec![topk_out], "topk",
    );

    g.outputs = vec![topk_out];
    g
}

// ---------------------------------------------------------------------------
// Expert FFN graph (SwiGLU, reusable per expert)
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for a single expert FFN: Gate Gemm → Up Gemm → SwiGLU → Down Gemm.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_expert_ffn_graph(
    seq_len: usize,
    hidden: usize,
    inter: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;

    let input = g.add_tensor("input", vec![s, hidden], dt);
    let w_gate = g.add_tensor("w_gate", vec![hidden, inter], dt);
    let w_up = g.add_tensor("w_up", vec![hidden, inter], dt);
    let w_down = g.add_tensor("w_down", vec![inter, hidden], dt);
    g.inputs = vec![input, w_gate, w_up, w_down];

    let gate_out = g.add_tensor("gate", vec![s, inter], dt);
    g.add_op(OpKind::Gemm { m: s, n: inter, k: hidden }, vec![input, w_gate], vec![gate_out], "gemm_gate");
    let up_out = g.add_tensor("up", vec![s, inter], dt);
    g.add_op(OpKind::Gemm { m: s, n: inter, k: hidden }, vec![input, w_up], vec![up_out], "gemm_up");
    let swiglu_out = g.add_tensor("swiglu", vec![s, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], "swiglu");
    let down_out = g.add_tensor("down", vec![s, hidden], dt);
    g.add_op(OpKind::Gemm { m: s, n: hidden, k: inter }, vec![swiglu_out, w_down], vec![down_out], "gemm_down");

    g.outputs = vec![down_out];
    g
}

// ---------------------------------------------------------------------------
// MoE combine graph: WeightedSum
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for MoE combine: weighted sum of expert outputs.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_moe_combine_graph(
    seq_len: usize,
    hidden: usize,
    top_k: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;

    let expert_outputs = g.add_tensor("expert_outputs", vec![top_k, s, hidden], dt);
    let indices = g.add_tensor("indices", vec![s, top_k], dt);
    let weights = g.add_tensor("weights", vec![s, top_k], dt);
    g.inputs = vec![expert_outputs, indices, weights];

    let output = g.add_tensor("output", vec![s, hidden], dt);
    g.add_op(
        OpKind::WeightedSum { seq_len: s, hidden, top_k },
        vec![expert_outputs, indices, weights], vec![output], "weighted_sum",
    );

    g.outputs = vec![output];
    g
}

// ---------------------------------------------------------------------------
// JIT KV cache update (replaces scalar update_kv_cache)
// ---------------------------------------------------------------------------

/// Compute K/V projections via JIT and write to KV cache (for MoE layers).
///
/// Replaces the scalar `update_kv_cache` — uses `build_kv_projection_graph` JIT path
/// for RmsNorm → K Gemm → RoPE, and scalar only for V Gemm (no RoPE on V).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn update_kv_cache_jit<E: Element>(
    backend: &CpuBackend<E>,
    handle: KvCacheHandle,
    layer: usize,
    hidden_state: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    rn1_w: &[f32],
    positions: &[u32],
    seq_len: usize,
    hidden: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    rope_theta: f64,
) -> Result<(), BE> {
    let kv_graph = build_kv_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta);
    let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let compiled = compiler.compile_graph(&kv_graph)
        .map_err(|e| BE::Cpu(format!("JIT compile kv_projection failed: {e}")))?;

    let (k_rope, v_proj) = execute_kv_projection(
        &compiled, hidden_state, rn1_w, k_w, v_w, positions,
        seq_len, hidden, num_kv_heads, head_dim, eps,
    );

    write_kv_to_cache(backend, handle, layer, &k_rope, &v_proj, seq_len, num_kv_heads, head_dim)
}

