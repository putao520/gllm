//! JIT graph builders and execution helpers extracted from decoder_forward.rs.
//!
//! These functions build `CompilerGraph` instances and execute compiled layers,
//! shared across CPU decoder forward, embedding forward, and rerank forward paths.

use super::cpu_backend::CpuBackend;
use super::Element;
use crate::engine::executor::{BackendError as BE, KvCacheHandle};
use gllm_kernels::types::DType;

/// Convert ModelConfig.dtype_size (bytes per element) to gllm-kernels DType.
///
/// 2-byte storage → F16 (native half-precision JIT path).
/// 4-byte storage → F32.
/// All other sizes → F32 (safe default).
#[inline]
pub(crate) fn computation_dtype(dtype_size: usize) -> DType {
    match dtype_size {
        2 => DType::F16,
        _ => DType::F32,
    }
}

/// Derive computation DType from a `GeneratorForwardConfig`.
///
/// Checks the `dtype` string first (distinguishes BF16 from F16, both 2 bytes),
/// then falls back to `dtype_size`.
#[inline]
pub(crate) fn computation_dtype_from_config(
    config: &crate::engine::executor::GeneratorForwardConfig,
) -> DType {
    match config.dtype.as_str() {
        "bf16" | "bfloat16" => DType::BF16,
        "f16" | "float16" | "fp16" => DType::F16,
        _ => computation_dtype(config.dtype_size),
    }
}

/// Convert a `gllm_kernels::types::DType` to the `crate::compat::DType` used in `ModelArchKey`.
#[inline]
pub(crate) fn kernels_dtype_to_compat(dt: DType) -> crate::compat::DType {
    match dt {
        DType::F16 => crate::compat::DType::F16,
        DType::BF16 => crate::compat::DType::BF16,
        DType::F32 => crate::compat::DType::F32,
    }
}

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
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], dt);
    let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
    let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], dt);
    let w_gate = g.add_tensor_concrete("w_gate", &[h, inter], dt);
    let w_up = g.add_tensor_concrete("w_up", &[h, inter], dt);
    let w_down = g.add_tensor_concrete("w_down", &[inter, h], dt);
    let rn2_w = g.add_tensor_concrete("rn2_w", &[h], dt);

    g.inputs = vec![
        input, w_q, w_k, w_v, w_o, rn1_w,
        w_gate, w_up, w_down, rn2_w,
    ];

    // Pre-attention RMSNorm
    let normed1 = g.add_tensor_concrete("normed1", &[s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed1], "rms_norm_1");

    // Q/K/V Projections
    let q_out = g.add_tensor_concrete("q", &[s, q_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: q_dim, k: h, dtype: dt }, vec![normed1, w_q], vec![q_out], "gemm_q");
    let k_out = g.add_tensor_concrete("k", &[s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h, dtype: dt }, vec![normed1, w_k], vec![k_out], "gemm_k");
    let v_out = g.add_tensor_concrete("v", &[s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h, dtype: dt }, vec![normed1, w_v], vec![v_out], "gemm_v");

    // RoPE
    let q_rope = g.add_tensor_concrete("q_rope", &[s, q_dim], dt);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![q_out], vec![q_rope], "rope_q");
    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_dim], dt);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![k_out], vec![k_rope], "rope_k");

    // Multi-Head Attention
    let attn_out = g.add_tensor_concrete("attn_out", &[s, q_dim], dt);
    g.add_op(
        OpKind::MultiHeadAttention { seq_len: s, num_heads, head_dim },
        vec![q_rope, k_rope, v_out], vec![attn_out], "mha",
    );

    // Output projection + Residual 1
    let o_out = g.add_tensor_concrete("o_proj", &[s, h], dt);
    g.add_op(OpKind::Gemm { m: s, n: h, k: q_dim, dtype: dt }, vec![attn_out, w_o], vec![o_out], "gemm_o");
    let resid1 = g.add_tensor_concrete("residual1", &[s, h], dt);
    g.add_op(OpKind::Residual, vec![input, o_out], vec![resid1], "residual_1");

    // Pre-FFN RMSNorm
    let normed2 = g.add_tensor_concrete("normed2", &[s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![resid1, rn2_w], vec![normed2], "rms_norm_2");

    // SwiGLU FFN
    let gate_out = g.add_tensor_concrete("ffn_gate", &[s, inter], dt);
    g.add_op(OpKind::Gemm { m: s, n: inter, k: h, dtype: dt }, vec![normed2, w_gate], vec![gate_out], "gemm_gate");
    let up_out = g.add_tensor_concrete("ffn_up", &[s, inter], dt);
    g.add_op(OpKind::Gemm { m: s, n: inter, k: h, dtype: dt }, vec![normed2, w_up], vec![up_out], "gemm_up");
    let swiglu_out = g.add_tensor_concrete("ffn_swiglu", &[s, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], "swiglu");
    let down_out = g.add_tensor_concrete("ffn_down", &[s, h], dt);
    g.add_op(OpKind::Gemm { m: s, n: h, k: inter, dtype: dt }, vec![swiglu_out, w_down], vec![down_out], "gemm_down");

    // Residual 2
    let output = g.add_tensor_concrete("output", &[s, h], dt);
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
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;
    let h = hidden;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], dt);
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], dt);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
    g.inputs = vec![input, rn1_w, w_k];

    let normed = g.add_tensor_concrete("normed", &[s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed], "rms_norm_kv");

    let k_out = g.add_tensor_concrete("k", &[s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h, dtype: dt }, vec![normed, w_k], vec![k_out], "gemm_k");

    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_dim], dt);
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
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;
    let h = hidden;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], dt);
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], dt);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
    g.inputs = vec![input, rn1_w, w_v];

    let normed = g.add_tensor_concrete("normed", &[s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed], "rms_norm_v");

    let v_out = g.add_tensor_concrete("v_proj", &[s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h, dtype: dt }, vec![normed, w_v], vec![v_out], "gemm_v");

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
    let v_graph = build_v_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps, DType::F32);
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
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;

    let input = g.add_tensor_concrete("input", &[seq_len, hidden], dt);
    let norm_w = g.add_tensor_concrete("norm_w", &[hidden], dt);
    let lm_w = g.add_tensor_concrete("lm_w", &[hidden, vocab_size], dt);
    g.inputs = vec![input, norm_w, lm_w];

    let normed = g.add_tensor_concrete("normed", &[seq_len, hidden], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, norm_w], vec![normed], "final_rms_norm");

    let logits = g.add_tensor_concrete("logits", &[seq_len, vocab_size], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n: vocab_size, k: hidden, dtype: dt },
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
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;

    let input = g.add_tensor_concrete("input", &[seq_len, hidden], dt);
    let norm_w = g.add_tensor_concrete("norm_w", &[hidden], dt);
    g.inputs = vec![input, norm_w];

    let normed = g.add_tensor_concrete("normed", &[seq_len, hidden], dt);
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
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], dt);
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], dt);
    let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
    g.inputs = vec![input, rn1_w, w_q, w_k, w_v];

    let normed = g.add_tensor_concrete("normed", &[s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed], "rms_norm_1");

    let q_out = g.add_tensor_concrete("q", &[s, q_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: q_dim, k: h, dtype: dt }, vec![normed, w_q], vec![q_out], "gemm_q");
    let k_out = g.add_tensor_concrete("k", &[s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h, dtype: dt }, vec![normed, w_k], vec![k_out], "gemm_k");
    let v_out = g.add_tensor_concrete("v", &[s, kv_dim], dt);
    g.add_op(OpKind::Gemm { m: s, n: kv_dim, k: h, dtype: dt }, vec![normed, w_v], vec![v_out], "gemm_v");

    let q_rope = g.add_tensor_concrete("q_rope", &[s, q_dim], dt);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![q_out], vec![q_rope], "rope_q");
    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_dim], dt);
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
    let k_graph = build_kv_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps, 10000.0, DType::F32);
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
    let v_graph = build_v_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps, DType::F32);
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
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;

    let attn_out = g.add_tensor_concrete("attn_out", &[s, q_dim], dt);
    let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);
    let residual_in = g.add_tensor_concrete("residual_in", &[s, h], dt);
    let rn2_w = g.add_tensor_concrete("rn2_w", &[h], dt);
    g.inputs = vec![attn_out, w_o, residual_in, rn2_w];

    let o_out = g.add_tensor_concrete("o_proj", &[s, h], dt);
    g.add_op(OpKind::Gemm { m: s, n: h, k: q_dim, dtype: dt }, vec![attn_out, w_o], vec![o_out], "gemm_o");

    let resid1 = g.add_tensor_concrete("residual1", &[s, h], dt);
    g.add_op(OpKind::Residual, vec![residual_in, o_out], vec![resid1], "residual_1");

    let normed2 = g.add_tensor_concrete("normed2", &[s, h], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![resid1, rn2_w], vec![normed2], "rms_norm_2");

    g.outputs = vec![normed2];
    g
}

// ---------------------------------------------------------------------------
// Cached GQA Attention graph (for incremental decode)
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for cached GQA attention.
/// Q[seq_len, q_dim] × K_cache[total_seq, kv_dim] → softmax(causal) → × V_cache → out[seq_len, q_dim]
///
/// `total_seq` is Concrete: full JIT loop-unrolling/tiling optimizations are preserved.
/// The caller (DecodeCachedJit) caches compiled layers per unique total_seq value,
/// so each value is compiled exactly once and reused on subsequent steps.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_cached_gqa_graph(
    seq_len: usize,
    total_seq: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::compiler::codegen::attention_strategy::select_attention_strategy;
    use gllm_kernels::dispatch::DeviceProfile;

    let profile = DeviceProfile::detect();
    let strategy = select_attention_strategy(
        seq_len, total_seq, head_dim, num_heads,
        dtype, &profile, None, None,
    );

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q_in = g.add_tensor_concrete("q", &[seq_len, q_dim], dt);
    let k_cache = g.add_tensor_concrete("k_cache", &[total_seq, kv_dim], dt);
    let v_cache = g.add_tensor_concrete("v_cache", &[total_seq, kv_dim], dt);
    g.inputs = vec![q_in, k_cache, v_cache];

    let attn_out = g.add_tensor_concrete("attn_out", &[seq_len, q_dim + 1], dt); // +1 for sparsity
    g.add_op(
        OpKind::CachedGQA { seq_len, total_seq, num_heads, num_kv_heads, head_dim, strategy, kv_dtype: dt },
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

    use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
    use crate::compat::DType as CompatDType;

    let arch_key = ModelArchKey {
        arch_name: "moe_ffn".to_string(),
        hidden_size: hidden,
        num_heads: 0,
        num_kv_heads: 0,
        head_dim: 0,
        dtype: CompatDType::F32,
    };

    // Step 1: MoE routing via JIT (MoEGate → TopK) — cached
    let routing_key = JitCacheKey {
        arch: arch_key.clone(),
        graph: GraphType::MoeFfnRouting { num_experts, top_k },
    };
    let routing_compiled = global_jit_cache()
        .get_or_compile(routing_key, || {
            let graph = build_moe_routing_graph(seq_len, hidden, num_experts, top_k, DType::F32);
            let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
            compiler.compile_graph(&graph).map_err(|e| format!("MoE routing JIT failed: {e}"))
        })
        .map_err(|e| BE::Other(e))?;

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

    // Step 2: Expert FFN via JIT (compile once, execute per expert) — cached
    let ffn_key = JitCacheKey {
        arch: arch_key.clone(),
        graph: GraphType::MoeFfnExpert { inter_size: inter },
    };
    let ffn_compiled = global_jit_cache()
        .get_or_compile(ffn_key, || {
            let graph = build_expert_ffn_graph(seq_len, hidden, inter, DType::F32);
            let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
            compiler.compile_graph(&graph).map_err(|e| format!("MoE expert FFN JIT failed: {e}"))
        })
        .map_err(|e| BE::Other(e))?;

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

    // Step 3: Weighted combine via JIT (WeightedSum) — cached
    let combine_key = JitCacheKey {
        arch: arch_key,
        graph: GraphType::MoeFfnCombine { top_k },
    };
    let combine_compiled = global_jit_cache()
        .get_or_compile(combine_key, || {
            let graph = build_moe_combine_graph(seq_len, hidden, top_k, DType::F32);
            let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
            compiler.compile_graph(&graph).map_err(|e| format!("MoE combine JIT failed: {e}"))
        })
        .map_err(|e| BE::Other(e))?;

    // Build compact expert_outputs buffer: [top_k, seq_len, hidden]
    // Layout: slot 0 = first selected expert's output, slot 1 = second, etc.
    // Indices are remapped to local 0..top_k offsets for WeightedSum.
    let expert_buf_size = top_k * seq_len * hidden;
    let mut expert_flat = vec![0.0f32; expert_buf_size];
    let mut local_indices = vec![0usize; seq_len * top_k];

    for s in 0..seq_len {
        for t in 0..top_k {
            let global_idx = indices[s * top_k + t];
            let expert_out = expert_outputs.get(&global_idx).expect("missing expert output");
            let src_base = s * hidden;
            let dst_slot = t;
            let dst_base = dst_slot * seq_len * hidden + s * hidden;
            expert_flat[dst_base..dst_base + hidden]
                .copy_from_slice(&expert_out[src_base..src_base + hidden]);
            local_indices[s * top_k + t] = t;
        }
    }

    let indices_f32: Vec<f32> = local_indices.iter().map(|&i| f32::from_bits(i as u32)).collect();

    // Pack: expert_outputs (input), indices + weights (weights)
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
// SwiGLU activation JIT helper (for quantized FFN path)
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for SwiGLU activation only: gate * silu(gate) * up.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn build_swiglu_graph(seq_len: usize, inter: usize) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let gate = g.add_tensor_concrete("gate", &[seq_len, inter], dt);
    let up = g.add_tensor_concrete("up", &[seq_len, inter], dt);
    g.inputs = vec![gate, up];

    let out = g.add_tensor_concrete("swiglu_out", &[seq_len, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate, up], vec![out], "swiglu");
    g.outputs = vec![out];
    g
}

/// Execute SwiGLU activation via JIT (cached).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn jit_swiglu(
    gate: &[f32],
    up: &[f32],
    seq_len: usize,
    inter: usize,
) -> Result<Vec<f32>, String> {
    use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
    use crate::compat::DType as GllmDType;

    let key = JitCacheKey {
        arch: ModelArchKey {
            arch_name: "swiglu".to_string(),
            hidden_size: inter,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            dtype: GllmDType::F32,
        },
        graph: GraphType::SwiGluActivation { inter_size: inter },
    };

    let compiled = global_jit_cache().get_or_compile(key, || {
        let graph = build_swiglu_graph(seq_len, inter);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        compiler.compile_graph(&graph).map_err(|e| format!("SwiGLU JIT failed: {e}"))
    })?;

    let gate_bytes = gate.len() * 4;
    let up_bytes = up.len() * 4;
    let mut input_buf = vec![0u8; gate_bytes + up_bytes];
    unsafe {
        std::ptr::copy_nonoverlapping(gate.as_ptr() as *const u8, input_buf.as_mut_ptr(), gate_bytes);
        std::ptr::copy_nonoverlapping(up.as_ptr() as *const u8, input_buf.as_mut_ptr().add(gate_bytes), up_bytes);
    }

    let mut output = vec![0.0f32; seq_len * inter];
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
    unsafe {
        compiled.execute(
            input_buf.as_ptr(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
    Ok(output)
}


/// Build a CompilerGraph for MoE routing: gate → topk.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_moe_routing_graph(
    seq_len: usize,
    hidden: usize,
    num_experts: usize,
    top_k: usize,
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;

    let normed2 = g.add_tensor_concrete("normed2", &[s, hidden], dt);
    let router_w = g.add_tensor_concrete("router_w", &[hidden, num_experts], dt);
    g.inputs = vec![normed2, router_w];

    let gate_probs = g.add_tensor_concrete("gate_probs", &[s, num_experts], dt);
    g.add_op(
        OpKind::MoEGate { seq_len: s, num_experts, hidden },
        vec![normed2, router_w], vec![gate_probs], "moe_gate",
    );

    let topk_out = g.add_tensor_concrete("topk_out", &[s, top_k * 2], dt);
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
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;

    let input = g.add_tensor_concrete("input", &[s, hidden], dt);
    let w_gate = g.add_tensor_concrete("w_gate", &[hidden, inter], dt);
    let w_up = g.add_tensor_concrete("w_up", &[hidden, inter], dt);
    let w_down = g.add_tensor_concrete("w_down", &[inter, hidden], dt);
    g.inputs = vec![input, w_gate, w_up, w_down];

    let gate_out = g.add_tensor_concrete("gate", &[s, inter], dt);
    g.add_op(OpKind::Gemm { m: s, n: inter, k: hidden, dtype: dt }, vec![input, w_gate], vec![gate_out], "gemm_gate");
    let up_out = g.add_tensor_concrete("up", &[s, inter], dt);
    g.add_op(OpKind::Gemm { m: s, n: inter, k: hidden, dtype: dt }, vec![input, w_up], vec![up_out], "gemm_up");
    let swiglu_out = g.add_tensor_concrete("swiglu", &[s, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], "swiglu");
    let down_out = g.add_tensor_concrete("down", &[s, hidden], dt);
    g.add_op(OpKind::Gemm { m: s, n: hidden, k: inter, dtype: dt }, vec![swiglu_out, w_down], vec![down_out], "gemm_down");

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
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;

    let expert_outputs = g.add_tensor_concrete("expert_outputs", &[top_k, s, hidden], dt);
    let indices = g.add_tensor_concrete("indices", &[s, top_k], dt);
    let weights = g.add_tensor_concrete("weights", &[s, top_k], dt);
    g.inputs = vec![expert_outputs, indices, weights];

    let output = g.add_tensor_concrete("output", &[s, hidden], dt);
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
    let kv_graph = build_kv_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta, DType::F32);
    let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let compiled = compiler.compile_graph(&kv_graph)
        .map_err(|e| BE::Cpu(format!("JIT compile kv_projection failed: {e}")))?;

    let (k_rope, v_proj) = execute_kv_projection(
        &compiled, hidden_state, rn1_w, k_w, v_w, positions,
        seq_len, hidden, num_kv_heads, head_dim, eps,
    );

    write_kv_to_cache(backend, handle, layer, &k_rope, &v_proj, seq_len, num_kv_heads, head_dim)
}

// ---------------------------------------------------------------------------
// JIT F32 GEMM (replaces scalar_gemm in all runtime paths)
// ---------------------------------------------------------------------------

/// Perform F32 GEMM via JIT compilation: output[m, n] = input[m, k] @ weight[k, n].
///
/// Builds a single-op CompilerGraph with `OpKind::Gemm`, compiles to native SIMD
/// (AVX2/AVX-512/NEON/SVE based on DeviceProfile), and executes.
/// DType is passed through to the JIT compiler for future F16/BF16 support.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn jit_gemm(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    dtype: gllm_kernels::types::DType,
) -> Result<(), String> {
    use gllm_kernels::compiler::{CompilerGraph, OpKind, InferenceCompiler};

    let mut g = CompilerGraph::new();

    let a = g.add_tensor_concrete("input", &[m, k], dtype);
    let b = g.add_tensor_concrete("weight", &[k, n], dtype);
    g.inputs = vec![a, b];

    let c = g.add_tensor_concrete("output", &[m, n], dtype);
    g.add_op(OpKind::Gemm { m, n, k, dtype }, vec![a, b], vec![c], "gemm");
    g.outputs = vec![c];

    let mut compiler = InferenceCompiler::new();
    let compiled = compiler.compile_graph(&g).map_err(|e| format!("{e:?}"))?;

    let weights_buf = pack_weights(&[weight]);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            input.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, m,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GPT-2 JIT graph builders
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for GPT-2 LayerNorm + fused QKV GemmBias.
///
/// Inputs (weight blob order): ln_w[hidden], ln_b[hidden], qkv_w[hidden, 3*hidden], qkv_b[3*hidden]
/// Output: qkv[seq_len, 3*hidden]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_gpt2_ln_qkv_graph(
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;
    let dt = DType::F32;
    let qkv_dim = 3 * hidden;
    let mut g = CompilerGraph::new();
    let input  = g.add_tensor_concrete("input",  &[seq_len, hidden],  dt);
    let ln_w   = g.add_tensor_concrete("ln_w",   &[hidden],           dt);
    let ln_b   = g.add_tensor_concrete("ln_b",   &[hidden],           dt);
    let qkv_w  = g.add_tensor_concrete("qkv_w",  &[hidden, qkv_dim],  dt);
    let qkv_b  = g.add_tensor_concrete("qkv_b",  &[qkv_dim],          dt);
    g.inputs = vec![input, ln_w, ln_b, qkv_w, qkv_b];
    let normed  = g.add_tensor_concrete("normed",  &[seq_len, hidden],  dt);
    g.add_op(OpKind::LayerNorm { eps }, vec![input, ln_w, ln_b], vec![normed], "ln1");
    let qkv_out = g.add_tensor_concrete("qkv", &[seq_len, qkv_dim], dt);
    g.add_op(
        OpKind::GemmBias { m: seq_len, n: qkv_dim, k: hidden, dtype: dt },
        vec![normed, qkv_w, qkv_b], vec![qkv_out], "gemm_qkv",
    );
    g.outputs = vec![qkv_out];
    g
}

/// Execute GPT-2 LayerNorm + fused QKV GemmBias. Returns qkv[seq_len, 3*hidden].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_gpt2_ln_qkv(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    ln_w: &[f32], ln_b: &[f32],
    qkv_w: &[f32], qkv_b: &[f32],
    seq_len: usize, hidden: usize,
) -> Vec<f32> {
    let qkv_dim = 3 * hidden;
    let mut output = vec![0.0f32; seq_len * qkv_dim];
    let weights_buf = pack_weights(&[ln_w, ln_b, qkv_w, qkv_b]);
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
    output
}

/// Build a CompilerGraph for GPT-2 O projection GemmBias + residual add.
///
/// Inputs: attn_out[seq_len, q_dim], o_w[q_dim, hidden], o_b[hidden], residual[seq_len, hidden]
/// Output: out[seq_len, hidden]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_gpt2_o_proj_graph(
    seq_len: usize,
    hidden: usize,
    q_dim: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;
    let dt = DType::F32;
    let mut g = CompilerGraph::new();
    let attn_out = g.add_tensor_concrete("attn_out", &[seq_len, q_dim],  dt);
    let o_w      = g.add_tensor_concrete("o_w",      &[q_dim, hidden],   dt);
    let o_b      = g.add_tensor_concrete("o_b",      &[hidden],          dt);
    let residual = g.add_tensor_concrete("residual", &[seq_len, hidden], dt);
    g.inputs = vec![attn_out, o_w, o_b, residual];
    let o_out = g.add_tensor_concrete("o_proj", &[seq_len, hidden], dt);
    g.add_op(
        OpKind::GemmBias { m: seq_len, n: hidden, k: q_dim, dtype: dt },
        vec![attn_out, o_w, o_b], vec![o_out], "gemm_o",
    );
    let out = g.add_tensor_concrete("out", &[seq_len, hidden], dt);
    g.add_op(OpKind::Residual, vec![residual, o_out], vec![out], "residual_1");
    g.outputs = vec![out];
    g
}

/// Execute GPT-2 O projection + residual. Returns hidden[seq_len, hidden].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_gpt2_o_proj(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    attn_out: &[f32],
    o_w: &[f32], o_b: &[f32],
    residual: &[f32],
    seq_len: usize, hidden: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * hidden];
    // ABI: activation = attn_out; weights blob = [o_w, o_b, residual]
    let weights_buf = pack_weights(&[o_w, o_b, residual]);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
    unsafe {
        compiled.execute(
            attn_out.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
    output
}

/// Build a CompilerGraph for GPT-2 LayerNorm2 + MLP (c_fc GemmBias + Gelu + c_proj GemmBias) + residual.
///
/// Inputs: hidden[seq_len, hidden], ln2_w[hidden], ln2_b[hidden],
///         fc_w[hidden, inter], fc_b[inter], proj_w[inter, hidden], proj_b[hidden],
///         residual[seq_len, hidden]
/// Output: out[seq_len, hidden]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_gpt2_ln_mlp_graph(
    seq_len: usize,
    hidden: usize,
    inter: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;
    let dt = DType::F32;
    let mut g = CompilerGraph::new();
    let input    = g.add_tensor_concrete("input",    &[seq_len, hidden], dt);
    let ln2_w    = g.add_tensor_concrete("ln2_w",    &[hidden],          dt);
    let ln2_b    = g.add_tensor_concrete("ln2_b",    &[hidden],          dt);
    let fc_w     = g.add_tensor_concrete("fc_w",     &[hidden, inter],   dt);
    let fc_b     = g.add_tensor_concrete("fc_b",     &[inter],           dt);
    let proj_w   = g.add_tensor_concrete("proj_w",   &[inter, hidden],   dt);
    let proj_b   = g.add_tensor_concrete("proj_b",   &[hidden],          dt);
    let residual = g.add_tensor_concrete("residual", &[seq_len, hidden], dt);
    g.inputs = vec![input, ln2_w, ln2_b, fc_w, fc_b, proj_w, proj_b, residual];
    let normed2 = g.add_tensor_concrete("normed2", &[seq_len, hidden], dt);
    g.add_op(OpKind::LayerNorm { eps }, vec![input, ln2_w, ln2_b], vec![normed2], "ln2");
    let fc_out = g.add_tensor_concrete("fc_out", &[seq_len, inter], dt);
    g.add_op(
        OpKind::GemmBias { m: seq_len, n: inter, k: hidden, dtype: dt },
        vec![normed2, fc_w, fc_b], vec![fc_out], "gemm_fc",
    );
    let gelu_out = g.add_tensor_concrete("gelu_out", &[seq_len, inter], dt);
    g.add_op(OpKind::Gelu, vec![fc_out], vec![gelu_out], "gelu");
    let proj_out = g.add_tensor_concrete("proj_out", &[seq_len, hidden], dt);
    g.add_op(
        OpKind::GemmBias { m: seq_len, n: hidden, k: inter, dtype: dt },
        vec![gelu_out, proj_w, proj_b], vec![proj_out], "gemm_proj",
    );
    let out = g.add_tensor_concrete("out", &[seq_len, hidden], dt);
    g.add_op(OpKind::Residual, vec![residual, proj_out], vec![out], "residual_2");
    g.outputs = vec![out];
    g
}

/// Execute GPT-2 LayerNorm2 + MLP + residual. Returns hidden[seq_len, hidden].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_gpt2_ln_mlp(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    ln2_w: &[f32], ln2_b: &[f32],
    fc_w: &[f32], fc_b: &[f32],
    proj_w: &[f32], proj_b: &[f32],
    residual: &[f32],
    seq_len: usize, hidden: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * hidden];
    let weights_buf = pack_weights(&[ln2_w, ln2_b, fc_w, fc_b, proj_w, proj_b, residual]);
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
    output
}

/// Build a CompilerGraph for GPT-2 final LayerNorm + lm_head (tied embedding, no bias).
///
/// `embed_w` must be pre-transposed to [hidden, vocab_size] before packing.
/// Inputs: hidden[seq_len, hidden], ln_f_w[hidden], ln_f_b[hidden], embed_w_t[hidden, vocab]
/// Output: logits[seq_len, vocab]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_gpt2_final_ln_lm_head_graph(
    seq_len: usize,
    hidden: usize,
    vocab_size: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;
    let dt = DType::F32;
    let mut g = CompilerGraph::new();
    let input   = g.add_tensor_concrete("input",   &[seq_len, hidden],    dt);
    let ln_f_w  = g.add_tensor_concrete("ln_f_w",  &[hidden],             dt);
    let ln_f_b  = g.add_tensor_concrete("ln_f_b",  &[hidden],             dt);
    let embed_w = g.add_tensor_concrete("embed_w", &[hidden, vocab_size], dt);
    g.inputs = vec![input, ln_f_w, ln_f_b, embed_w];
    let normed = g.add_tensor_concrete("normed", &[seq_len, hidden], dt);
    g.add_op(OpKind::LayerNorm { eps }, vec![input, ln_f_w, ln_f_b], vec![normed], "ln_f");
    let logits = g.add_tensor_concrete("logits", &[seq_len, vocab_size], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n: vocab_size, k: hidden, dtype: dt },
        vec![normed, embed_w], vec![logits], "lm_head",
    );
    g.outputs = vec![logits];
    g
}

/// Execute GPT-2 final LayerNorm + lm_head.
/// `embed_w_t` must be pre-transposed to [hidden, vocab_size].
/// Returns logits[seq_len, vocab_size].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_gpt2_final_ln_lm_head(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    ln_f_w: &[f32], ln_f_b: &[f32],
    embed_w_t: &[f32],
    seq_len: usize, vocab_size: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * vocab_size];
    let weights_buf = pack_weights(&[ln_f_w, ln_f_b, embed_w_t]);
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
    output
}


// ---------------------------------------------------------------------------
// JIT LayerNorm (with bias) — for embedding LayerNorm in BERT/GPT-2
// ---------------------------------------------------------------------------

/// Build a JIT graph for LayerNorm with bias: out = layernorm(x, gamma, beta).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_layer_norm_graph(
    seq_len: usize,
    hidden: usize,
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    let mut g = CompilerGraph::new();
    let x = g.add_tensor_concrete("x", &[seq_len, hidden], dtype);
    let gamma = g.add_tensor_concrete("gamma", &[hidden], dtype);
    let beta = g.add_tensor_concrete("beta", &[hidden], dtype);
    g.inputs = vec![x, gamma, beta];
    let out = g.add_tensor_concrete("out", &[seq_len, hidden], dtype);
    g.add_op(
        OpKind::LayerNorm { eps: 1e-5 },
        vec![x, gamma, beta], vec![out], "layer_norm",
    );
    g.outputs = vec![out];
    g
}

/// Execute JIT LayerNorm with bias over [seq_len, hidden] input.
/// Uses global_jit_cache to avoid recompilation across calls.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn jit_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    seq_len: usize,
    hidden: usize,
) -> Result<Vec<f32>, String> {
    use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
    use crate::compat::DType as GllmDType;
    use gllm_kernels::types::DType;

    let key = JitCacheKey {
        arch: ModelArchKey {
            arch_name: "layer_norm".to_string(),
            hidden_size: hidden,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            dtype: GllmDType::F32,
        },
        graph: GraphType::Norm2, // reuse Norm2 slot for LayerNorm
    };

    let compiled = global_jit_cache().get_or_compile(key, || {
        let graph = build_layer_norm_graph(seq_len, hidden, DType::F32);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        compiler.compile_graph(&graph).map_err(|e| e.to_string())
    })?;

    let mut output = vec![0.0f32; seq_len * hidden];
    let weights_buf = pack_weights(&[gamma, beta]);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
    unsafe {
        compiled.execute(
            input.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// JIT L2 Normalize — for embedding output normalization
// ---------------------------------------------------------------------------

/// Build a JIT graph for L2 normalization: out = x / ||x||_2.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_l2_norm_graph(
    seq_len: usize,
    hidden: usize,
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    let mut g = CompilerGraph::new();
    let x = g.add_tensor_concrete("x", &[seq_len, hidden], dtype);
    g.inputs = vec![x];
    let out = g.add_tensor_concrete("out", &[seq_len, hidden], dtype);
    g.add_op(
        OpKind::L2Normalize { hidden: hidden },
        vec![x], vec![out], "l2_norm",
    );
    g.outputs = vec![out];
    g
}

/// Execute JIT L2 normalization over [seq_len, hidden] input.
/// Uses global_jit_cache to avoid recompilation across calls.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn jit_l2_normalize(
    input: &[f32],
    seq_len: usize,
    hidden: usize,
) -> Result<Vec<f32>, String> {
    use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
    use crate::compat::DType as GllmDType;
    use gllm_kernels::types::DType;

    let key = JitCacheKey {
        arch: ModelArchKey {
            arch_name: "l2_normalize".to_string(),
            hidden_size: hidden,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            dtype: GllmDType::F32,
        },
        graph: GraphType::Norm2,
    };

    let compiled = global_jit_cache().get_or_compile(key, || {
        let graph = build_l2_norm_graph(seq_len, hidden, DType::F32);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        compiler.compile_graph(&graph).map_err(|e| e.to_string())
    })?;

    let mut output = vec![0.0f32; seq_len * hidden];
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
    unsafe {
        compiled.execute(
            input.as_ptr() as *const u8,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Residual add JIT helper
// ---------------------------------------------------------------------------

/// Execute element-wise add via JIT (cached): output = a + b.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn jit_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
    use crate::compat::DType as GllmDType;
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let numel = a.len();
    let key = JitCacheKey {
        arch: ModelArchKey {
            arch_name: "residual_add".to_string(),
            hidden_size: numel,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            dtype: GllmDType::F32,
        },
        graph: GraphType::ResidualAdd { numel },
    };

    let compiled = global_jit_cache().get_or_compile(key, || {
        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let x = g.add_tensor_concrete("a", &[numel], dt);
        let y = g.add_tensor_concrete("b", &[numel], dt);
        g.inputs = vec![x, y];
        let out = g.add_tensor_concrete("out", &[numel], dt);
        g.add_op(OpKind::Add, vec![x, y], vec![out], "add");
        g.outputs = vec![out];
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        compiler.compile_graph(&g).map_err(|e| format!("residual add JIT failed: {e}"))
    })?;

    let a_bytes = numel * 4;
    let b_bytes = numel * 4;
    let mut input_buf = vec![0u8; a_bytes + b_bytes];
    unsafe {
        std::ptr::copy_nonoverlapping(a.as_ptr() as *const u8, input_buf.as_mut_ptr(), a_bytes);
        std::ptr::copy_nonoverlapping(b.as_ptr() as *const u8, input_buf.as_mut_ptr().add(a_bytes), b_bytes);
    }

    let mut output = vec![0.0f32; numel];
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
    unsafe {
        compiled.execute(
            input_buf.as_ptr(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, numel,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Mean pool JIT helper
// ---------------------------------------------------------------------------

/// Execute mean pooling via JIT (cached): average [seq_len, hidden] → [hidden].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn jit_mean_pool(
    input: &[f32],
    seq_len: usize,
    hidden: usize,
) -> Result<Vec<f32>, String> {
    use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
    use crate::compat::DType as GllmDType;

    let key = JitCacheKey {
        arch: ModelArchKey {
            arch_name: "mean_pool".to_string(),
            hidden_size: hidden,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            dtype: GllmDType::F32,
        },
        graph: GraphType::BertMeanPool,
    };

    let compiled = global_jit_cache().get_or_compile(key, || {
        let graph = super::bert_forward::build_mean_pool_graph(seq_len, hidden);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        compiler.compile_graph(&graph).map_err(|e| format!("mean pool JIT failed: {e}"))
    })?;

    let mut output = vec![0.0f32; hidden];
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
    unsafe {
        compiled.execute(
            input.as_ptr() as *const u8,
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
    Ok(output)
}
