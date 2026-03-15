//! JIT graph builders and execution helpers extracted from decoder_forward.rs.
//!
//! These functions build `CompilerGraph` instances and execute compiled layers,
//! shared across CPU decoder forward, embedding forward, and rerank forward paths.

use super::cpu_backend::CpuBackend;
use super::scalar_ops::{scalar_gemm, scalar_rms_norm, scalar_rope};
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
/// V projection is computed separately via scalar GEMM (no Concat OpKind).
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

/// Execute a JIT-compiled KV projection graph.
///
/// Returns (k_rope, v_proj) both as [seq_len, kv_dim].
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

    // V projection via scalar: RmsNorm → V Gemm (no RoPE on V)
    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);
    let mut v_proj = vec![0.0f32; seq_len * kv_dim];
    scalar_gemm(&normed, v_w, &mut v_proj, seq_len, kv_dim, hidden);

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

/// Compute K/V projections via scalar and write to KV cache (for MoE layers).
pub(crate) fn update_kv_cache<E: Element>(
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
    let kv_dim = num_kv_heads * head_dim;

    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);

    let mut k_proj = vec![0.0f32; seq_len * kv_dim];
    scalar_gemm(&normed, k_w, &mut k_proj, seq_len, kv_dim, hidden);

    let mut v_proj = vec![0.0f32; seq_len * kv_dim];
    scalar_gemm(&normed, v_w, &mut v_proj, seq_len, kv_dim, hidden);

    scalar_rope(&mut k_proj, positions, head_dim, rope_theta);

    write_kv_to_cache(backend, handle, layer, &k_proj, &v_proj, seq_len, num_kv_heads, head_dim)
}
