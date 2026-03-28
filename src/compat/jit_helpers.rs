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
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * std::mem::size_of::<f32>()) }
}

/// Zero-copy reinterpret `&mut [f32]` as `&mut [u8]`.
#[inline]
pub(crate) fn f32_as_bytes_mut(s: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, s.len() * std::mem::size_of::<f32>()) }
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

/// Convert ModelConfig.dtype_size (bytes per element) to gllm-kernels DType.
///
/// 2-byte storage → F16 or BF16 (based on dtype string).
/// 4-byte storage → F32.
/// All other sizes → F32 (safe default).
#[inline]
pub(crate) fn computation_dtype(dtype_size: usize) -> DType {
    match dtype_size {
        2 => DType::F16,  // Default to F16 for 2-byte types
        _ => DType::F32,
    }
}

/// Derive computation DType from a `GeneratorForwardConfig` with BF16 support.
///
/// Returns the model's native dtype — F16/BF16/F32 based on dtype_size and dtype string.
/// ARCH-DTYPE-ADAPTIVE: 禁止硬编码返回 F32。
#[inline]
pub(crate) fn computation_dtype_from_config(
    config: &crate::engine::executor::GeneratorForwardConfig,
) -> DType {
    match config.dtype_size {
        2 => {
            // Distinguish F16 vs BF16 using dtype string
            if config.dtype.to_lowercase().contains("bf16") {
                DType::BF16
            } else {
                DType::F16
            }
        }
        _ => DType::F32,
    }
}

/// Convert a `gllm_kernels::types::DType` to the `crate::compat::DType` used in `ModelArchKey`.
#[inline]
pub(crate) fn kernels_dtype_to_compat(dt: DType) -> crate::compat::DType {
    match dt {
        DType::F16 => crate::compat::DType::F16,
        DType::BF16 => crate::compat::DType::BF16,
        DType::F32 => crate::compat::DType::F32,
        gllm_kernels::types::DType::U8 => panic!("U8 unsupported as compat DataType"),
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
            DType::U8 => panic!("U8 unsupported"),
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
pub(crate) fn pack_weights_multi(slices_with_dtypes: &[(&[f32], DType)]) -> Vec<u8> {
    let total_bytes: usize = slices_with_dtypes.iter()
        .map(|(s, dt)| s.len() * dt.size_bytes())
        .sum();
    let mut buf = vec![0u8; total_bytes];
    let mut offset = 0;
    for &(slice, dtype) in slices_with_dtypes {
        let bytes = slice.len() * dtype.size_bytes();
        match dtype {
            DType::U8 => panic!("U8 unsupported"),
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

// ---------------------------------------------------------------------------
// Decoder layer graph
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for a single decoder layer (pre-norm, RMSNorm + SwiGLU).
/// ARCH-DTYPE-ADAPTIVE: GEMM weights use model dtype `dt`, norm weights use F32,
/// activations/intermediates use F32 (accumulator precision).
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
    let dt = dtype;           // GEMM weight dtype (model native)
    let ft = dtype;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Input activation: F32
    let input = g.add_tensor_concrete("input", &[s, h], ft);
    // GEMM weights: model dtype
    let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
    let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);
    // Norm weights: F32
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], ft);
    // GEMM weights: model dtype
    let w_gate = g.add_tensor_concrete("w_gate", &[h, inter], dt);
    let w_up = g.add_tensor_concrete("w_up", &[h, inter], dt);
    let w_down = g.add_tensor_concrete("w_down", &[inter, h], dt);
    // Norm weights: F32
    let rn2_w = g.add_tensor_concrete("rn2_w", &[h], ft);

    g.inputs = vec![
        input, w_q, w_k, w_v, w_o, rn1_w,
        w_gate, w_up, w_down, rn2_w,
    ];

    // Pre-attention RMSNorm (activation F32)
    let normed1 = g.add_tensor_concrete("normed1", &[s, h], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed1], "rms_norm_1");

    // Q/K/V Projections (GEMM: F32 activation × dt weight → F32 output)
    let q_out = g.add_tensor_concrete("q", &[s, q_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: q_dim, k: h, dtype: dt }, vec![normed1, w_q], vec![q_out], "gemm_q");
    let k_out = g.add_tensor_concrete("k", &[s, kv_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: kv_dim, k: h, dtype: dt }, vec![normed1, w_k], vec![k_out], "gemm_k");
    let v_out = g.add_tensor_concrete("v", &[s, kv_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: kv_dim, k: h, dtype: dt }, vec![normed1, w_v], vec![v_out], "gemm_v");

    // RoPE (F32 activation)
    let q_rope = g.add_tensor_concrete("q_rope", &[s, q_dim], ft);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![q_out], vec![q_rope], "rope_q");
    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_dim], ft);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![k_out], vec![k_rope], "rope_k");

    // Multi-Head Attention (F32 activation)
    let attn_out = g.add_tensor_concrete("attn_out", &[s, q_dim], ft);
    g.add_op(
        OpKind::MultiHeadAttention { seq_len: gllm_kernels::compiler::SymDim::Concrete(s), num_heads, num_kv_heads, head_dim },
        vec![q_rope, k_rope, v_out], vec![attn_out], "mha",
    );

    // Output projection + Residual 1
    let o_out = g.add_tensor_concrete("o_proj", &[s, h], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: h, k: q_dim, dtype: dt }, vec![attn_out, w_o], vec![o_out], "gemm_o");
    let resid1 = g.add_tensor_concrete("residual1", &[s, h], ft);
    g.add_op(OpKind::Residual, vec![input, o_out], vec![resid1], "residual_1");

    // Pre-FFN RMSNorm (F32)
    let normed2 = g.add_tensor_concrete("normed2", &[s, h], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![resid1, rn2_w], vec![normed2], "rms_norm_2");

    // SwiGLU FFN
    let gate_out = g.add_tensor_concrete("ffn_gate", &[s, inter], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: inter, k: h, dtype: dt }, vec![normed2, w_gate], vec![gate_out], "gemm_gate");
    let up_out = g.add_tensor_concrete("ffn_up", &[s, inter], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: inter, k: h, dtype: dt }, vec![normed2, w_up], vec![up_out], "gemm_up");
    let swiglu_out = g.add_tensor_concrete("ffn_swiglu", &[s, inter], ft);
    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], "swiglu");
    let down_out = g.add_tensor_concrete("ffn_down", &[s, h], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: h, k: inter, dtype: dt }, vec![swiglu_out, w_down], vec![down_out], "gemm_down");

    // Residual 2
    let output = g.add_tensor_concrete("output", &[s, h], ft);
    g.add_op(OpKind::Residual, vec![resid1, down_out], vec![output], "residual_2");

    g.outputs = vec![output];
    g
}

/// Execute a JIT-compiled decoder layer.
/// ARCH-DTYPE-ADAPTIVE: GEMM weights packed at model dtype, norm weights at F32.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_decoder_layer(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    q_w: &[f32], k_w: &[f32], v_w: &[f32], o_w: &[f32], rn1_w: &[f32],
    gate_w: &[f32], up_w: &[f32], down_w: &[f32], rn2_w: &[f32],
    positions: &[u32],
    seq_len: usize,
    output: &mut [f32],
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
    let dt = dtype;           // GEMM weight dtype
    let ft = dtype;
    let s = seq_len;
    let h = hidden;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], ft);
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], ft);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_dim], dt);
    g.inputs = vec![input, rn1_w, w_k];

    let normed = g.add_tensor_concrete("normed", &[s, h], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed], "rms_norm_kv");

    let k_out = g.add_tensor_concrete("k", &[s, kv_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: kv_dim, k: h, dtype: dt }, vec![normed, w_k], vec![k_out], "gemm_k");

    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_dim], ft);
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
    let dt = dtype;           // GEMM weight dtype
    let ft = dtype;
    let s = seq_len;
    let h = hidden;
    let kv_dim = num_kv_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], ft);
    let rn1_w = g.add_tensor_concrete("rn1_w", &[h], ft);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_dim], dt);
    g.inputs = vec![input, rn1_w, w_v];

    let normed = g.add_tensor_concrete("normed", &[s, h], ft);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1_w], vec![normed], "rms_norm_v");

    let v_out = g.add_tensor_concrete("v_proj", &[s, kv_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: kv_dim, k: h, dtype: dt }, vec![normed, w_v], vec![v_out], "gemm_v");

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
    v_compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    rn1_w: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    positions: &[u32],
    seq_len: usize,
    _hidden: usize,
    num_kv_heads: usize,
    head_dim: usize,
    _eps: f32,
    dtype: DType,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let kv_dim = num_kv_heads * head_dim;

    // JIT graph for K projection (RmsNorm → Gemm → RoPE)
    let weights_buf = pack_weights_multi(&[(rn1_w, dtype), (k_w, dtype)]);
    let mut k_rope = TypedBuffer::zeros(seq_len * kv_dim, dtype);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1, seq_len,
            k_rope.as_bytes_mut().as_mut_ptr(),
            scratchpad.as_mut_ptr(),
        );
    }

    // V projection via JIT: RmsNorm → V Gemm (no RoPE on V)
    let v_weights_buf = pack_weights_multi(&[(rn1_w, dtype), (v_w, dtype)]);
    let mut v_proj = TypedBuffer::zeros(seq_len * kv_dim, dtype);
    let mut v_scratchpad = vec![0u8; v_compiled.scratchpad_bytes];

    unsafe {
        v_compiled.execute(
            hidden_state.as_ptr() as *const u8,
            v_weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            v_proj.as_bytes_mut().as_mut_ptr(),
            v_scratchpad.as_mut_ptr(),
        );
    }

    Ok((k_rope.to_f32_vec(), v_proj.to_f32_vec()))
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
    let eb = buffer.elem_bytes;

    if write_start + seq_len > max_seq {
        return Err(BE::Cpu(format!(
            "KV cache overflow: write_start={write_start} + seq_len={seq_len} > max_seq_len={max_seq}"
        )));
    }

    // Convert f32 source data to bytes and write into typed KV cache
    let k_bytes = f32_as_bytes(k_data);
    let v_bytes = f32_as_bytes(v_data);
    let src_eb = std::mem::size_of::<f32>();

    for h in 0..num_kv_heads {
        let layer_head_base = (layer * num_kv_heads + h) * max_seq * head_dim;
        for s in 0..seq_len {
            let cache_offset = (layer_head_base + (write_start + s) * head_dim) * eb;
            let proj_offset = (s * kv_dim + h * head_dim) * src_eb;
            if eb == src_eb {
                // F32 cache: direct byte copy
                buffer.k[cache_offset..cache_offset + head_dim * eb]
                    .copy_from_slice(&k_bytes[proj_offset..proj_offset + head_dim * src_eb]);
                buffer.v[cache_offset..cache_offset + head_dim * eb]
                    .copy_from_slice(&v_bytes[proj_offset..proj_offset + head_dim * src_eb]);
            } else {
                // F16/BF16 cache: convert f32 → target dtype per element
                let k_src = &k_data[s * kv_dim + h * head_dim..s * kv_dim + h * head_dim + head_dim];
                let v_src = &v_data[s * kv_dim + h * head_dim..s * kv_dim + h * head_dim + head_dim];
                for d in 0..head_dim {
                    let dst = cache_offset + d * eb;
                    if eb == 2 {
                        // Use cache_dtype to distinguish F16 vs BF16
                        match buffer.cache_dtype {
                            DType::BF16 => {
                                let kh = half::bf16::from_f32(k_src[d]).to_le_bytes();
                                buffer.k[dst..dst + 2].copy_from_slice(&kh);
                                let vh = half::bf16::from_f32(v_src[d]).to_le_bytes();
                                buffer.v[dst..dst + 2].copy_from_slice(&vh);
                            }
                            _ => {
                                let kh = half::f16::from_f32(k_src[d]).to_le_bytes();
                                buffer.k[dst..dst + 2].copy_from_slice(&kh);
                                let vh = half::f16::from_f32(v_src[d]).to_le_bytes();
                                buffer.v[dst..dst + 2].copy_from_slice(&vh);
                            }
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
        "write_kv_to_cache: layer={layer}, wrote {seq_len} tokens at pos {write_start}, total_seq={}",
        if layer == 0 { write_start + seq_len } else { buffer.seq_len }
    );

    Ok(())
}

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
                        gllm_kernels::types::DType::U8 => panic!("U8 unsupported"),
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
                        gllm_kernels::types::DType::U8 => panic!("U8 unsupported"),
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
        OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(seq_len), n: vocab_size, k: hidden, dtype: dt },
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
    dtype: DType,
) {
    let weights_buf = pack_weights_multi(&[(norm_w, dtype), (lm_w, dtype)]);
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
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: q_dim, k: h, dtype: dt }, vec![normed, w_q], vec![q_out], "gemm_q");
    let k_out = g.add_tensor_concrete("k", &[s, kv_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: kv_dim, k: h, dtype: dt }, vec![normed, w_k], vec![k_out], "gemm_k");
    let v_out = g.add_tensor_concrete("v", &[s, kv_dim], ft);
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: kv_dim, k: h, dtype: dt }, vec![normed, w_v], vec![v_out], "gemm_v");

    let q_rope = g.add_tensor_concrete("q_rope", &[s, q_dim], ft);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![q_out], vec![q_rope], "rope_q");
    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_dim], ft);
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
    dtype: DType,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let weights_buf = pack_weights_multi(&[(rn1_w, dtype), (q_w, dtype), (k_w, dtype), (v_w, dtype)]);
    let mut q_rope = TypedBuffer::zeros(seq_len * q_dim, dtype);
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr(),
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1, seq_len,
            q_rope.as_bytes_mut().as_mut_ptr(),
            scratchpad.as_mut_ptr(),
        );
    }

    // k_rope and v_proj via separate JIT graphs (single-output ABI limitation)
    // K: RmsNorm → K Gemm → RoPE
    let k_graph = build_kv_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps, 10000.0, dtype);
    let mut k_compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let k_compiled = k_compiler.compile_graph(&k_graph)
        .map_err(|e| format!("JIT compile k_projection failed: {e}"))?;
    let k_weights_buf = pack_weights_multi(&[(rn1_w, dtype), (k_w, dtype)]);
    let mut k_proj = TypedBuffer::zeros(seq_len * kv_dim, dtype);
    let mut k_scratch = vec![0u8; k_compiled.scratchpad_bytes];
    unsafe {
        k_compiled.execute(
            hidden_state.as_ptr() as *const u8,
            k_weights_buf.as_ptr(),
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1, seq_len,
            k_proj.as_bytes_mut().as_mut_ptr(),
            k_scratch.as_mut_ptr(),
        );
    }

    // V: RmsNorm → V Gemm (no RoPE)
    let v_graph = build_v_projection_graph(seq_len, hidden, num_kv_heads, head_dim, eps, dtype);
    let mut v_compiler = gllm_kernels::compiler::InferenceCompiler::new();
    let v_compiled = v_compiler.compile_graph(&v_graph)
        .map_err(|e| format!("JIT compile v_projection failed: {e}"))?;
    let v_weights_buf = pack_weights_multi(&[(rn1_w, dtype), (v_w, dtype)]);
    let mut v_proj = TypedBuffer::zeros(seq_len * kv_dim, dtype);
    let mut v_scratch = vec![0u8; v_compiled.scratchpad_bytes];
    unsafe {
        v_compiled.execute(
            hidden_state.as_ptr() as *const u8,
            v_weights_buf.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1, seq_len,
            v_proj.as_bytes_mut().as_mut_ptr(),
            v_scratch.as_mut_ptr(),
        );
    }

    Ok((q_rope.to_f32_vec(), k_proj.to_f32_vec(), v_proj.to_f32_vec()))
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
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(s), n: h, k: q_dim, dtype: dt }, vec![attn_out, w_o], vec![o_out], "gemm_o");

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
    use gllm_kernels::compiler::codegen::attention_strategy::select_attention_strategy;
    use gllm_kernels::dispatch::DeviceProfile;

    let profile = DeviceProfile::detect();
    let strategy = select_attention_strategy(
        seq_len, total_seq, head_dim, num_heads,
        dtype, &profile, None, None,
    );

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
        OpKind::CachedGQA { seq_len: gllm_kernels::compiler::SymDim::Concrete(seq_len), total_seq: gllm_kernels::compiler::SymDim::Concrete(total_seq), num_heads, num_kv_heads, head_dim, strategy, kv_dtype: dtype },
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
    g.add_op(OpKind::Gemm { m: gllm_kernels::compiler::SymDim::Concrete(m), n, k, dtype }, vec![a, b], vec![c], "gemm");
    g.outputs = vec![c];

    let mut compiler = InferenceCompiler::new();
    let compiled = compiler.compile_graph(&g).map_err(|e| format!("{e:?}"))?;

    let weights_buf = pack_weights_typed(&[weight], dtype);
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
