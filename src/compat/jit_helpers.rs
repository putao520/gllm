//! JIT graph builders and execution helpers extracted from decoder_forward.rs.
//!
//! These functions build `CompilerGraph` instances and execute compiled layers,
//! shared across CPU decoder forward, embedding forward, and rerank forward paths.

use super::cpu_backend::CpuBackend;
use super::Element;
use crate::engine::executor::{BackendError as BE, KvCacheHandle};
use gllm_kernels::types::DType;

// ---------------------------------------------------------------------------
// ARCH-DTYPE-FULLCHAIN-ORCH: TypedBuffer — Vec<u8> + DType wrapper
// ---------------------------------------------------------------------------

/// A buffer of typed elements stored as raw bytes.
/// Replaces `Vec<f32>` in forward pass for dtype-agnostic computation.
pub(crate) struct TypedBuffer {
    pub data: Vec<u8>,
    pub dtype: DType,
    /// Number of logical elements (not bytes).
    pub len: usize,
}

impl TypedBuffer {
    /// Allocate a zero-initialized buffer for `len` elements of `dtype`.
    pub fn zeros(len: usize, dtype: DType) -> Self {
        Self {
            data: vec![0u8; len * dtype.size_bytes()],
            dtype,
            len,
        }
    }

    /// Raw byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] { &self.data }

    /// Mutable raw byte slice.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] { &mut self.data }

    /// Byte length.
    #[inline]
    pub fn byte_len(&self) -> usize { self.data.len() }

    /// Reinterpret as `&[f32]`. Panics if dtype != F32.
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "as_f32 called on non-F32 TypedBuffer");
        bytes_as_f32(&self.data)
    }

    /// Reinterpret as `&mut [f32]`. Panics if dtype != F32.
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.dtype, DType::F32, "as_f32_mut called on non-F32 TypedBuffer");
        let len = self.data.len() / 4;
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f32, len) }
    }

    /// Convert to Vec<f32>. Zero-copy for F32, element-wise conversion for F16/BF16.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        typed_bytes_to_f32(&self.data, self.dtype)
    }

    /// Write f32 data into this buffer, converting to the buffer's dtype.
    pub fn copy_from_f32(&mut self, src: &[f32]) {
        assert_eq!(src.len(), self.len, "copy_from_f32: length mismatch");
        let converted = super::weight_helpers::f32_to_typed_bytes(src, self.dtype);
        self.data.copy_from_slice(&converted);
    }

    /// Create from existing f32 slice, converting to target dtype.
    pub fn from_f32_slice(src: &[f32], dtype: DType) -> Self {
        let data = super::weight_helpers::f32_to_typed_bytes(src, dtype);
        Self { data, dtype, len: src.len() }
    }

    /// Create from raw bytes (no conversion). Caller must ensure bytes match dtype.
    pub fn from_raw(data: Vec<u8>, dtype: DType) -> Self {
        let len = data.len() / dtype.size_bytes();
        Self { data, dtype, len }
    }

    /// Copy bytes from another TypedBuffer or byte slice into this buffer.
    pub fn copy_from_bytes(&mut self, src: &[u8]) {
        assert_eq!(src.len(), self.data.len(), "copy_from_bytes: length mismatch");
        self.data.copy_from_slice(src);
    }
}

/// Zero-copy reinterpret `&[f32]` as `&[u8]`.
#[inline]
pub(crate) fn f32_as_bytes(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// Zero-copy reinterpret `&mut [f32]` as `&mut [u8]`.
#[inline]
pub(crate) fn f32_as_bytes_mut(s: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}

/// Zero-copy reinterpret `&[u8]` as `&[f32]`.
#[inline]
pub(crate) fn bytes_as_f32(s: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len() / std::mem::size_of::<f32>()) }
}

/// Convert typed bytes (F16/BF16/F32) to Vec<f32>.
/// F32: zero-copy reinterpret. F16/BF16: element-wise conversion.
pub(crate) fn typed_bytes_to_f32(data: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytes_as_f32(data).to_vec(),
        DType::F16 => {
            data.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        DType::BF16 => {
            data.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        _ => bytes_as_f32(data).to_vec(),
    }
}

/// Derive computation DType from a `GeneratorForwardConfig`.
///
/// Returns the model's native dtype — F16/BF16/F32.
/// ARCH-DTYPE-ADAPTIVE: 禁止硬编码返回 F32。
#[inline]
pub(crate) fn computation_dtype_from_config(
    config: &crate::engine::executor::GeneratorForwardConfig,
) -> DType {
    config.dtype()
}

/// Convert a `gllm_kernels::types::DType` to the `crate::compat::DType` used in `ModelArchKey`.
#[inline]
pub(crate) fn kernels_dtype_to_compat(dt: DType) -> crate::compat::DType {
    match dt {
        DType::F16 => crate::compat::DType::F16,
        DType::BF16 => crate::compat::DType::BF16,
        DType::F32 => crate::compat::DType::F32,
        DType::U8
        | DType::F8E4M3
        | DType::F8E5M2
        | DType::F6E3M2
        | DType::F6E2M3
        | DType::F4E2M1 => crate::compat::DType::U8,
    }
}

// ---------------------------------------------------------------------------
// Weight packing helper (shared by all JIT execute_* functions)
// ---------------------------------------------------------------------------

/// Pack multiple f32 weight slices into a contiguous byte buffer (F32 快捷方式).
pub(crate) fn pack_weights(slices: &[&[f32]]) -> Vec<u8> {
    pack_weights_typed(slices, DType::F32)
}

/// Pack multiple f32 weight slices into a contiguous byte buffer, converting to target dtype.
/// ARCH-DTYPE-ADAPTIVE: GEMM weights 按模型 dtype pack，Norm weights 按 F32 pack。
pub(crate) fn pack_weights_typed(slices: &[&[f32]], dtype: DType) -> Vec<u8> {
    let elem_bytes = dtype.size_bytes();
    let total_bytes: usize = slices.iter().map(|s| s.len() * elem_bytes).sum();
    let mut buf = vec![0u8; total_bytes];
    let mut offset = 0;
    for slice in slices {
        let bytes = slice.len() * elem_bytes;
        match dtype {
            DType::U8
            | DType::F8E4M3
            | DType::F8E5M2
            | DType::F6E3M2
            | DType::F6E2M3
            | DType::F4E2M1 => panic!("sub-byte/U8 dtype unsupported for weight packing"),
            DType::F32 => {
                buf[offset..offset + bytes].copy_from_slice(unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes)
                });
            }
            DType::F16 => {
                for (i, &val) in slice.iter().enumerate() {
                    let h = half::f16::from_f32(val);
                    let hb = h.to_le_bytes();
                    buf[offset + i * 2] = hb[0];
                    buf[offset + i * 2 + 1] = hb[1];
                }
            }
            DType::BF16 => {
                for (i, &val) in slice.iter().enumerate() {
                    let h = half::bf16::from_f32(val);
                    let hb = h.to_le_bytes();
                    buf[offset + i * 2] = hb[0];
                    buf[offset + i * 2 + 1] = hb[1];
                }
            }
        }
        offset += bytes;
    }
    buf
}

/// Pack weight slices with per-tensor dtype (mixed dtype graphs).
/// Each tuple is (slice, dtype) — GEMM weights use model dtype, norm weights use F32.
/// ARCH-DTYPE-FULLCHAIN-ORCH: accepts &[u8] instead of &[f32].
pub(crate) fn pack_weights_multi(slices_with_dtypes: &[(&[u8], DType)]) -> Vec<u8> {
    let total_bytes: usize = slices_with_dtypes.iter()
        .map(|(s, dt)| s.len() / dt.size_bytes() * dt.size_bytes())
        .sum();
    let mut buf = vec![0u8; total_bytes];
    let mut offset = 0;
    for &(slice, _dtype) in slices_with_dtypes {
        let bytes = slice.len();
        // Direct byte copy — slice is already in target dtype
        buf[offset..offset + bytes].copy_from_slice(slice);
        offset += bytes;
    }
    buf
}

// ---------------------------------------------------------------------------
// Decoder layer graph
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for a single decoder layer (pre-norm, RMSNorm + SwiGLU).
///
/// Delegates to `CompilerGraph::decoder_layer()` from gllm-kernels for graph
/// structure, then overrides `g.inputs` to include weight tensors in the order
/// expected by `execute_jit_decoder_layer` (CPU JIT path packs weights into
/// a contiguous buffer indexed by input position).
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
    rope_partial: f32,
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::CompilerGraph;

    let mut g = CompilerGraph::decoder_layer(
        seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta, rope_partial, dtype,
    );

    // Override inputs: CPU JIT path needs all weight tensors in g.inputs so
    // that execute_jit_decoder_layer can pack them into a contiguous buffer.
    // The decoder_layer() builder creates tensors named:
    //   input(0), attn_norm_w(1), w_q(2), w_k(3), w_v(4), w_o(5),
    //   ffn_norm_w(6), w_gate(7), w_up(8), w_down(9)
    // CPU path expects order: input, w_q, w_k, w_v, w_o, rn1_w, w_gate, w_up, w_down, rn2_w
    let find_tensor = |name: &str| -> gllm_kernels::compiler::TensorId {
        g.tensors.iter()
            .find(|t| t.name == name)
            .map(|t| t.id)
            .unwrap_or_else(|| panic!("tensor '{}' not found in decoder_layer graph", name))
    };

    let input = find_tensor("input");
    let w_q = find_tensor("w_q");
    let w_k = find_tensor("w_k");
    let w_v = find_tensor("w_v");
    let w_o = find_tensor("w_o");
    let rn1_w = find_tensor("attn_norm_w");
    let w_gate = find_tensor("w_gate");
    let w_up = find_tensor("w_up");
    let w_down = find_tensor("w_down");
    let rn2_w = find_tensor("ffn_norm_w");

    g.inputs = vec![input, w_q, w_k, w_v, w_o, rn1_w, w_gate, w_up, w_down, rn2_w];
    g
}

/// Execute a JIT-compiled decoder layer.
/// ARCH-DTYPE-FULLCHAIN-ORCH: accepts &[u8] for all data, dtype via parameter.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_decoder_layer(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[u8],
    q_w: &[u8], k_w: &[u8], v_w: &[u8], o_w: &[u8], rn1_w: &[u8],
    gate_w: &[u8], up_w: &[u8], down_w: &[u8], rn2_w: &[u8],
    positions: &[u32],
    seq_len: usize,
    output: &mut [u8],
    dtype: DType,
) {
    let ft = dtype; // norm weight dtype
    let weights_buf = pack_weights_multi(&[
        (q_w, dtype), (k_w, dtype), (v_w, dtype), (o_w, dtype), (rn1_w, ft),
        (gate_w, dtype), (up_w, dtype), (down_w, dtype), (rn2_w, ft),
    ]);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr(),
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr(),
            scratchpad.as_mut_ptr(),
        );
    }
}

// KV projection is handled by GraphExecutor (REQ-JIT-CACHE-001).
// Legacy build_kv_projection_graph / build_v_projection_graph / execute_kv_projection
// removed — operator-level JIT superseded by full-layer fusion graphs.

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------

/// Write typed KV data (any dtype) into the KV cache buffer.
/// ARCH-DTYPE-FULLCHAIN-ORCH: accepts &[u8] + src_dtype instead of &[f32].
pub(crate) fn write_kv_to_cache_typed<E: Element>(
    backend: &CpuBackend<E>,
    handle: KvCacheHandle,
    layer: usize,
    k_data: &[u8],
    v_data: &[u8],
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    src_dtype: DType,
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
    let cache_eb = buffer.elem_bytes;
    let src_eb = src_dtype.size_bytes();

    if write_start + seq_len > max_seq {
        return Err(BE::Cpu(format!(
            "KV cache overflow: write_start={write_start} + seq_len={seq_len} > max_seq_len={max_seq}"
        )));
    }

    for h in 0..num_kv_heads {
        let layer_head_base = (layer * num_kv_heads + h) * max_seq * head_dim;
        for s in 0..seq_len {
            let cache_offset = (layer_head_base + (write_start + s) * head_dim) * cache_eb;
            let proj_offset = (s * kv_dim + h * head_dim) * src_eb;

            if cache_eb == src_eb {
                // Same dtype: direct byte copy
                buffer.k[cache_offset..cache_offset + head_dim * cache_eb]
                    .copy_from_slice(&k_data[proj_offset..proj_offset + head_dim * src_eb]);
                buffer.v[cache_offset..cache_offset + head_dim * cache_eb]
                    .copy_from_slice(&v_data[proj_offset..proj_offset + head_dim * src_eb]);
            } else {
                // Cross-dtype: convert element by element via f32 intermediate
                for d in 0..head_dim {
                    let s_off = proj_offset + d * src_eb;
                    let d_off = cache_offset + d * cache_eb;
                    // Read source element as f32
                    let val = match src_dtype {
                        DType::F32 => f32::from_le_bytes([
                            k_data[s_off], k_data[s_off+1], k_data[s_off+2], k_data[s_off+3]
                        ]),
                        DType::F16 => half::f16::from_le_bytes([
                            k_data[s_off], k_data[s_off+1]
                        ]).to_f32(),
                        DType::BF16 => half::bf16::from_le_bytes([
                            k_data[s_off], k_data[s_off+1]
                        ]).to_f32(),
                        DType::U8
                        | DType::F8E4M3
                        | DType::F8E5M2
                        | DType::F6E3M2
                        | DType::F6E2M3
                        | DType::F4E2M1 => panic!("sub-byte/U8 dtype unsupported for KV cache cross-dtype conversion"),
                    };
                    let v_val = match src_dtype {
                        DType::F32 => f32::from_le_bytes([
                            v_data[s_off], v_data[s_off+1], v_data[s_off+2], v_data[s_off+3]
                        ]),
                        DType::F16 => half::f16::from_le_bytes([
                            v_data[s_off], v_data[s_off+1]
                        ]).to_f32(),
                        DType::BF16 => half::bf16::from_le_bytes([
                            v_data[s_off], v_data[s_off+1]
                        ]).to_f32(),
                        DType::U8
                        | DType::F8E4M3
                        | DType::F8E5M2
                        | DType::F6E3M2
                        | DType::F6E2M3
                        | DType::F4E2M1 => panic!("sub-byte/U8 dtype unsupported for KV cache cross-dtype conversion"),
                    };
                    // Write to cache in cache dtype
                    match cache_eb {
                        4 => {
                            buffer.k[d_off..d_off+4].copy_from_slice(&val.to_le_bytes());
                            buffer.v[d_off..d_off+4].copy_from_slice(&v_val.to_le_bytes());
                        }
                        2 => {
                            let kh = half::f16::from_f32(val).to_le_bytes();
                            buffer.k[d_off..d_off+2].copy_from_slice(&kh);
                            let vh = half::f16::from_f32(v_val).to_le_bytes();
                            buffer.v[d_off..d_off+2].copy_from_slice(&vh);
                        }
                        _ => {
                            return Err(BE::Cpu(format!("unsupported cache elem_bytes: {cache_eb}")));
                        }
                    }
                }
            }
        }
    }

    if layer == 0 {
        buffer.seq_len = (buffer.seq_len + seq_len).min(max_seq);
    }

    log::debug!(
        "write_kv_to_cache_typed: layer={layer}, wrote {seq_len} tokens at pos {write_start}, src_dtype={src_dtype:?}, total_seq={}",
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
    let dt = dtype;           // GEMM weight dtype (lm_w)
    let ft = dtype;

    let input = g.add_tensor_concrete("input", &[seq_len, hidden], ft);
    let norm_w = g.add_tensor_concrete("norm_w", &[hidden], ft);
    let lm_w = g.add_tensor_concrete("lm_w", &[hidden, vocab_size], dt);
    g.inputs = vec![input, norm_w, lm_w];

    let normed = g.add_tensor_concrete("normed", &[seq_len, hidden], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, norm_w], vec![normed], "final_rms_norm");

    let logits = g.add_tensor_concrete("logits", &[seq_len, vocab_size], ft);
    g.add_op(
        OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(seq_len), n: vocab_size, k: hidden, dtype: dt, trans_b: false },
        vec![normed, lm_w], vec![logits], "lm_head",
    );

    g.outputs = vec![logits];
    g
}

/// Execute the JIT-compiled lm_head.
/// ARCH-DTYPE-FULLCHAIN-ORCH: accepts &[u8] for all data.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_lm_head(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[u8],
    norm_w: &[u8],
    lm_w: &[u8],
    seq_len: usize,
    output: &mut [u8],
    dtype: DType,
) {
    let weights_buf = pack_weights_multi(&[(norm_w, dtype), (lm_w, dtype)]);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr(),
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr(),
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
    let ft = dtype; // norm weights and activations

    let input = g.add_tensor_concrete("input", &[seq_len, hidden], ft);
    let norm_w = g.add_tensor_concrete("norm_w", &[hidden], ft);
    g.inputs = vec![input, norm_w];

    let normed = g.add_tensor_concrete("normed", &[seq_len, hidden], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, norm_w], vec![normed], "final_rms_norm");

    g.outputs = vec![normed];
    g
}

/// Execute JIT-compiled final norm (RMSNorm only).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_final_norm(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[u8],
    norm_w: &[u8],
    seq_len: usize,
    output: &mut [u8],
) {
    let weights_buf = norm_w.to_vec();
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr(),
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            output.as_mut_ptr(),
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
    rope_partial: f32,
    dtype: DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;           // GEMM weight dtype
    let ft = dtype;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], ft);
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], ft);
    let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
    g.inputs = vec![input, rn1_w, w_q, w_k, w_v];

    let normed = g.add_tensor_concrete("normed", &[s, h], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed], "rms_norm_1");

    let q_out = g.add_tensor_concrete("q", &[s, q_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: q_dim, k: h, dtype: dt, trans_b: false }, vec![normed, w_q], vec![q_out], "gemm_q");
    let k_out = g.add_tensor_concrete("k", &[s, kv_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: kv_dim, k: h, dtype: dt, trans_b: false }, vec![normed, w_k], vec![k_out], "gemm_k");
    let v_out = g.add_tensor_concrete("v", &[s, kv_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: kv_dim, k: h, dtype: dt, trans_b: false }, vec![normed, w_v], vec![v_out], "gemm_v");

    let q_rope = g.add_tensor_concrete("q_rope", &[s, q_dim], ft);
    g.add_op(OpKind::RoPE { num_heads, head_dim, theta: rope_theta, partial: rope_partial, rope_scaling: None }, vec![q_out], vec![q_rope], "rope_q");
    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_dim], ft);
    g.add_op(OpKind::RoPE { num_heads: num_kv_heads, head_dim, theta: rope_theta, partial: rope_partial, rope_scaling: None }, vec![k_out], vec![k_rope], "rope_k");

    // Output: q_rope (primary output). k_rope and v_out are extracted from scratchpad.
    g.outputs = vec![q_rope];
    g
}

/// Execute MoE pre-attention graph. Returns (q_rope, k_rope, v_proj).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[allow(dead_code)]
pub(crate) fn execute_moe_pre_attention(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[u8],
    rn1_w: &[u8],
    q_w: &[u8], k_w: &[u8], v_w: &[u8],
    positions: &[u32],
    seq_len: usize,
    _hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    _eps: f32,
    dtype: DType,
) -> Result<(TypedBuffer, TypedBuffer, TypedBuffer), String> {
    let q_dim = num_heads * head_dim;
    let _kv_dim = num_kv_heads * head_dim;

    let weights_buf = pack_weights_multi(&[(rn1_w, dtype), (q_w, dtype), (k_w, dtype), (v_w, dtype)]);
    let mut q_rope = TypedBuffer::zeros(seq_len * q_dim, dtype);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr(),
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1, seq_len,
            q_rope.as_bytes_mut().as_mut_ptr(),
            scratchpad.as_mut_ptr(),
        );
    }

    // NOTE: The original implementation used runtime JIT compilation (InferenceCompiler::new())
// for K/V projections, which violates REQ-JIT-CACHE-001 (all compilation must happen at
// model load time). Since this function is unused (dead code), the runtime compilation
// paths have been removed. If this function is needed in the future, it must be refactored
// to use pre-compiled graphs from the model load phase.

    // Return an error to prevent accidental use of the old runtime-JIT pattern
    Err("execute_moe_pre_attention: runtime JIT compilation removed (REQ-JIT-CACHE-001). \
         Use pre-compiled graphs from the GraphExecutor instead.".to_string())
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
    let dt = dtype;           // GEMM weight dtype
    let ft = dtype;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;

    let attn_out = g.add_tensor_concrete("attn_out", &[s, q_dim], ft);
    let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);
    let residual_in = g.add_tensor_concrete("residual_in", &[s, h], ft);
    let rn2_w = g.add_tensor_concrete("rn2_w", &[h], ft);
    g.inputs = vec![attn_out, w_o, residual_in, rn2_w];

    let o_out = g.add_tensor_concrete("o_proj", &[s, h], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: h, k: q_dim, dtype: dt, trans_b: false }, vec![attn_out, w_o], vec![o_out], "gemm_o");

    let resid1 = g.add_tensor_concrete("residual1", &[s, h], ft);
    g.add_op(OpKind::Residual, vec![residual_in, o_out], vec![resid1], "residual_1");

    let normed2 = g.add_tensor_concrete("normed2", &[s, h], ft);
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
    use gllm_kernels::compiler::graph::AttentionStrategy;

    // Derive attention strategy from HwOptEngine's AttentionPlan (§10.7)
    let exec_plan = gllm_kernels::compiler::planner::global_execution_plan();
    let strategy = match exec_plan.attention_plan.variant {
        gllm_kernels::compiler::planner::AttentionVariant::FA4BlockScaled
        | gllm_kernels::compiler::planner::AttentionVariant::FA3Pipeline
        | gllm_kernels::compiler::planner::AttentionVariant::FA2Tiled => {
            AttentionStrategy::FlashV2 {
                block_m: exec_plan.attention_plan.tile_q,
                block_n: exec_plan.attention_plan.tile_kv,
            }
        }
        _ => {
            if total_seq > 1024 {
                AttentionStrategy::FlashV2 { block_m: 64, block_n: 64 }
            } else {
                AttentionStrategy::Naive
            }
        }
    };

    let mut g = CompilerGraph::new();
    let ft = dtype;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q_in = g.add_tensor_concrete("q", &[seq_len, q_dim], ft);
    let k_cache = g.add_tensor_concrete("k_cache", &[total_seq, kv_dim], ft);
    let v_cache = g.add_tensor_concrete("v_cache", &[total_seq, kv_dim], ft);
    g.inputs = vec![q_in, k_cache, v_cache];

    let attn_out = g.add_tensor_concrete("attn_out", &[seq_len, q_dim + 1], ft); // +1 for sparsity
    g.add_op(
        OpKind::CachedGQA { seq_len, total_seq, num_heads, num_kv_heads, head_dim, strategy, kv_dtype: dtype },
        vec![q_in, k_cache, v_cache], vec![attn_out], "cached_gqa",
    );

    g.outputs = vec![attn_out];
    g
}




// ---------------------------------------------------------------------------
// JIT F32 GEMM (replaces scalar_gemm in all runtime paths)
// ---------------------------------------------------------------------------

/// Perform F32 GEMM via JIT compilation: output[m, n] = input[m, k] @ weight[k, n].
///
/// **DEPRECATED**: This function used runtime JIT compilation (InferenceCompiler::new()),
/// which violates REQ-JIT-CACHE-001. All compilation must happen at model load time.
///
/// Use the GraphExecutor's pre-compiled GEMM kernels instead.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn jit_gemm(
    _input: &[f32],
    _weight: &[f32],
    _output: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
    _dtype: gllm_kernels::types::DType,
) -> Result<(), String> {
    Err("jit_gemm: runtime JIT compilation removed (REQ-JIT-CACHE-001). \
         Use pre-compiled GEMM kernels from the GraphExecutor.".to_string())
}

pub fn cpu_fingerprint() -> u64 {
    let mut bits: u64 = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2")       { bits |= 1 << 0; }
        if std::is_x86_feature_detected!("avx512f")    { bits |= 1 << 1; }
        if std::is_x86_feature_detected!("avx512bf16") { bits |= 1 << 2; }
        if std::is_x86_feature_detected!("fma")        { bits |= 1 << 4; }
    }
    #[cfg(target_arch = "aarch64")]
    {
        bits |= 1 << 8;
    }
    bits
}
