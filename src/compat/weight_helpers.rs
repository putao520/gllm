use super::backend_trait::{self, Backend};
use super::cpu_backend::CpuBackend;
use super::Element;
use crate::engine::executor::BackendError as BE;
use crate::loader::QuantizedTensor;
use gllm_kernels::quant::QuantType;
use gllm_kernels::types::DType;

/// Reinterpret a slice of Element as &[f32]. Only valid when E is f32.
#[allow(dead_code)]
pub(crate) fn as_f32_slice<E: Element>(data: &[E]) -> &[f32] {
    debug_assert_eq!(
        std::mem::size_of::<E>(), std::mem::size_of::<f32>(),
        "as_f32_slice requires E to be f32-sized (got {} bytes)", std::mem::size_of::<E>()
    );
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) }
}

/// Derive DType from Element type.
/// Returns error for unsupported element types.
pub(crate) fn element_to_dtype<E: Element>() -> Result<DType, BE> {
    let elem_id = E::ELEM_ID;
    match elem_id {
        0 => Ok(DType::F32),  // f32
        1 => Ok(DType::F16),  // f16
        2 => Ok(DType::BF16), // bf16
        _ => Err(BE::Other(format!("Unsupported element type with ELEM_ID: {}", elem_id))),
    }
}

/// Internal helper: dequantize a quantized tensor to f32 Vec.
/// Handles GGUF row-padding for block quantization.
fn _dequantize_to_f32<E: Element>(
    qt: &QuantizedTensor,
    backend: &CpuBackend<E>,
) -> Result<Vec<f32>, BE> {
    let n: usize = qt.shape.iter().product();
    let mut out = vec![0.0f32; n];
    let blk_bytes = qt.quant_type.block_bytes();
    let blk_elems = qt.quant_type.block_size();

    // GGUF pads each row (innermost dim) to the block boundary.
    // When ne0 % block_size != 0, we must dequantize row-by-row,
    // skipping the padding blocks at the end of each row.
    // After GGUF→HF shape reversal, innermost dim is shape[last] (not shape[0]).
    // tensor_nbytes uses original GGUF order where shape[0] = innermost,
    // but qt.shape has been reversed to HF order where shape[last] = innermost.
    let ne0 = if qt.shape.is_empty() { n } else { qt.shape[qt.shape.len() - 1] };
    let blocks_per_row = ne0.div_ceil(blk_elems);
    let row_data_bytes = blocks_per_row * blk_bytes;
    let needs_row_dequant = ne0 % blk_elems != 0 && qt.shape.len() >= 2;

    if needs_row_dequant {
        let n_rows: usize = qt.shape[..qt.shape.len() - 1].iter().product();
        let mut out_off = 0;
        let mut data_off = 0;
        // Temp buffer for one full row (including padding elements)
        let row_elems_padded = blocks_per_row * blk_elems;
        let mut row_buf = vec![0.0f32; row_elems_padded];
        for _row in 0..n_rows {
            let row_data = &qt.data[data_off..data_off + row_data_bytes];
            backend.dequantize(row_data, &mut row_buf, qt.quant_type)?;
            // Copy only the valid (non-padding) elements
            out[out_off..out_off + ne0].copy_from_slice(&row_buf[..ne0]);
            out_off += ne0;
            data_off += row_data_bytes;
        }
    } else {
        backend.dequantize(&qt.data, &mut out, qt.quant_type)?;
    }

    Ok(out)
}

/// Dequantize a QuantizedTensor to Vec<f32>. Public wrapper for executor use.
pub(crate) fn dequantize_weight_to_f32<E: Element>(
    qt: &crate::loader::QuantizedTensor,
    backend: &crate::compat::CpuBackend<E>,
) -> Result<Vec<f32>, crate::compat::BackendError> {
    _dequantize_to_f32(qt, backend)
}

/// Try to get bias data as Vec<f32>. Returns zeros if not found.
#[allow(dead_code)]
pub(crate) fn get_bias_data<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    names: &[impl AsRef<str>],
    size: usize,
) -> Vec<f32> {
    for name in names {
        if let Some(t) = weights.get_tensor(name.as_ref()) {
            return as_f32_slice(t.as_slice()).to_vec();
        }
    }
    vec![0.0; size]
}

/// Get bias data as (Vec<u8>, DType). Returns zeros if not found.
pub(crate) fn get_bias_data_typed<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    names: &[impl AsRef<str>],
    size: usize,
) -> (Vec<u8>, DType) {
    let dtype = match element_to_dtype::<E>() {
        Ok(d) => d,
        Err(_) => DType::F32, // Fallback for unsupported types
    };

    for name in names {
        if let Some(t) = weights.get_tensor(name.as_ref()) {
            let e_slice: &[E] = t.as_slice();
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    e_slice.as_ptr() as *const u8,
                    std::mem::size_of_val(e_slice),
                )
            };
            return (bytes.to_vec(), dtype);
        }
    }

    // Return zero-initialized buffer
    let elem_bytes = dtype.size_bytes();
    (vec![0u8; size * elem_bytes], dtype)
}

/// Detect whether linear-layer weights need transposing before GEMM.
///
/// Returns `true` when the dequantized flat array stores weights in
/// `[out_dim, in_dim]` row-major order (SafeTensors / PyTorch convention AND
/// GGUF convention — GGUF shape `[ne0=in, ne1=out]` is ne1 rows of ne0 cols,
/// i.e. `[out, in]` in memory).
///
/// Returns `false` only for genuine ONNX layout where the flat array is
/// already `[in_dim, out_dim]`.
#[allow(dead_code)]
pub(crate) fn needs_weight_transpose<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
) -> bool {
    // Build name_map from available tensor names for format-agnostic probing.
    let all_names = weights.available_names();
    let name_map = crate::loader::name_map::TensorNameMap::build_from_names(&all_names, false);

    // GGUF models use blk.* names — they always need transpose because
    // GGUF shape [ne0, ne1] stores ne1 rows of ne0 elements, giving
    // [out_dim, in_dim] in memory (same as SafeTensors).
    const GGUF_PROBE: &str = "blk.0.ffn_up.weight";
    if weights.tensor_shape(GGUF_PROBE).is_some() {
        return true;
    }

    // Probe decoder gate_proj via canonical name → external name lookup.
    if let Some(ext) = name_map.to_external("L0.gate_proj") {
        if let Some(shape) = weights.tensor_shape(ext) {
            if shape.len() == 2 && shape[0] != shape[1] {
                return shape[0] > shape[1];
            }
        }
    }
    if let Some(ext) = name_map.to_external("L0.up_proj") {
        if let Some(shape) = weights.tensor_shape(ext) {
            if shape.len() == 2 && shape[0] != shape[1] {
                return shape[0] > shape[1];
            }
        }
    }

    // Probe encoder FFN up_proj via canonical name.
    if let Some(ext) = name_map.to_external("L0.up_proj") {
        if let Some(shape) = weights.tensor_shape(ext) {
            if shape.len() == 2 && shape[0] != shape[1] {
                // SafeTensors: [out_dim(1536), in_dim(384)] → first > second → true
                // ONNX:        [in_dim(384), out_dim(1536)] → first < second → false
                return shape[0] > shape[1];
            }
        }
    }

    // Fallback: check if any embedding weight exists (SafeTensors naming).
    if let Some(ext) = name_map.to_external("embed") {
        weights.get_tensor(ext).is_some() || weights.get_quantized(ext).is_some()
    } else {
        false
    }
}

/// Try to get tensor data as (Vec<u8>, DType). Returns None if not found.
#[allow(dead_code)]
pub(crate) fn try_get_typed_data<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    backend: &CpuBackend<E>,
    names: &[impl AsRef<str>],
) -> Option<(Vec<u8>, DType)> {
    get_typed_data(weights, backend, names).ok()
}

/// Transpose a row-major matrix [rows, cols] → [cols, rows].
/// Adaptively handles 2:4 structure sparsity where physical dimension is halved.
#[allow(dead_code)]
pub(crate) fn transpose_f32(data: &[f32], rows: usize, _cols_expected: usize) -> Vec<f32> {
    let actual_cols = data.len() / rows;
    assert_eq!(data.len(), rows * actual_cols, "data length must be a multiple of rows");
    let mut out = vec![0.0f32; rows * actual_cols];
    for r in 0..rows {
        for c in 0..actual_cols {
            out[c * rows + r] = data[r * actual_cols + c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Quantized linear acceleration
// ---------------------------------------------------------------------------

/// Result of looking up a weight: either quantized raw blocks or dequantized f32.
#[allow(dead_code)]
pub(crate) enum WeightData {
    /// Quantized weight: raw block bytes, quant type, and shape [out_dim, in_dim].
    Quantized {
        data: Vec<u8>,
        quant_type: QuantType,
        out_dim: usize,
        in_dim: usize,
    },
    /// Dequantized f32 weight (native f32 tensor or already dequantized).
    F32(Vec<f32>),
}

/// Look up a weight tensor, returning quantized data when available.
///
/// For quantized (GGUF) weights with 2D shape, returns `WeightData::Quantized`
/// so the caller can dispatch to `quantized_matmul` directly, skipping the
/// expensive dequantize + transpose path.
///
/// For native f32 tensors or 1D quantized tensors (e.g. norm weights),
/// returns `WeightData::F32`.
#[allow(dead_code)]
pub(crate) fn get_weight_data<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    backend: &CpuBackend<E>,
    names: &[impl AsRef<str>],
) -> Result<WeightData, BE> {
    for name in names {
        let name = name.as_ref();
        if let Some(t) = weights.get_tensor(name) {
            let data = as_f32_slice(t.as_slice()).to_vec();
            return Ok(WeightData::F32(data));
        }
        if let Some(qt) = weights.get_quantized(name) {
            if qt.shape.len() >= 2 {
                // Shape is HF order [out_dim, in_dim] (reversed from GGUF at TensorProvider boundary).
                // Memory layout: out_dim rows of in_dim elements (quantized blocks).
                let out_dim = qt.shape[0];
                let in_dim = qt.shape[qt.shape.len() - 1];
                return Ok(WeightData::Quantized {
                    data: qt.data.clone(),
                    quant_type: qt.quant_type,
                    out_dim,
                    in_dim,
                });
            }
            // 1D quantized tensor (norm weights etc.): dequantize to f32
            return Ok(WeightData::F32(_dequantize_to_f32(qt, backend)?));
        }
    }
    let name_strs: Vec<&str> = names.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("tensor not found: {:?}", name_strs)))
}

/// Extract the f32 data from a WeightData, dequantizing if necessary.
/// Used for weights that cannot go through quantized_matmul (e.g. norm weights,
/// or weights needed for non-GEMM operations like KV cache update).
#[allow(dead_code)]
pub(crate) fn weight_data_to_f32<E: Element>(
    weight: &WeightData,
    backend: &CpuBackend<E>,
    transpose_weights: bool,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, BE> {
    match weight {
        WeightData::F32(data) => {
            if transpose_weights && data.len() == out_dim * in_dim {
                Ok(transpose_f32(data, out_dim, in_dim))
            } else {
                Ok(data.clone())
            }
        }
        WeightData::Quantized { data, quant_type, out_dim: w_out, in_dim: w_in } => {
            // Dequantize then transpose
            let total = w_out * w_in;
            let mut f32_data = vec![0.0f32; total];
            let blk_bytes = quant_type.block_bytes();
            let blk_elems = quant_type.block_size();
            let blocks_per_row = w_in.div_ceil(blk_elems);
            let row_data_bytes = blocks_per_row * blk_bytes;
            let needs_row_dequant = w_in % blk_elems != 0;

            if needs_row_dequant {
                let row_elems_padded = blocks_per_row * blk_elems;
                let mut row_buf = vec![0.0f32; row_elems_padded];
                let mut out_off = 0;
                let mut data_off = 0;
                for _row in 0..*w_out {
                    let row_data = &data[data_off..data_off + row_data_bytes];
                    backend.dequantize(row_data, &mut row_buf, *quant_type)?;
                    f32_data[out_off..out_off + *w_in].copy_from_slice(&row_buf[..*w_in]);
                    out_off += w_in;
                    data_off += row_data_bytes;
                }
            } else {
                backend.dequantize(data, &mut f32_data, *quant_type)?;
            }

            if transpose_weights {
                Ok(transpose_f32(&f32_data, *w_out, *w_in))
            } else {
                Ok(f32_data)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ARCH-DTYPE-FULLCHAIN-ORCH: Typed (Vec<u8> + DType) variants
// ---------------------------------------------------------------------------

/// Convert f32 slice to typed bytes in target dtype.
pub(crate) fn f32_to_typed_bytes(data: &[f32], dtype: DType) -> Vec<u8> {
    let eb = dtype.size_bytes();
    let mut out = vec![0u8; data.len() * eb];
    match dtype {
        DType::F32 => {
            let n = out.len();
            out.copy_from_slice(unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, n)
            });
        }
        DType::F16 => {
            for (i, &v) in data.iter().enumerate() {
                let h = half::f16::from_f32(v);
                let b = h.to_le_bytes();
                out[i * 2] = b[0];
                out[i * 2 + 1] = b[1];
            }
        }
        DType::BF16 => {
            for (i, &v) in data.iter().enumerate() {
                let h = half::bf16::from_f32(v);
                let b = h.to_le_bytes();
                out[i * 2] = b[0];
                out[i * 2 + 1] = b[1];
            }
        }
        DType::U8
        | DType::F8E4M3
        | DType::F8E5M2
        | DType::F6E3M2
        | DType::F6E2M3
        | DType::F4E2M1 => {
            for (i, &v) in data.iter().enumerate() {
                out[i] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Transpose a row-major matrix stored as typed bytes [rows, cols] → [cols, rows].
#[allow(dead_code)]
pub(crate) fn transpose_typed(data: &[u8], rows: usize, cols: usize, dtype: DType) -> Vec<u8> {
    let eb = dtype.size_bytes();
    assert_eq!(data.len(), rows * cols * eb);
    let mut out = vec![0u8; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * eb;
            let dst = (c * rows + r) * eb;
            out[dst..dst + eb].copy_from_slice(&data[src..src + eb]);
        }
    }
    out
}

/// Load tensor data, returning (bytes, dtype).
///
/// Derives dtype from the Element type parameter E:
/// - f32 → DType::F32
/// - f16 → DType::F16
/// - bf16 → DType::BF16
///
/// For native tensors: returns raw bytes in Element's dtype.
/// For quantized tensors: dequantizes to f32, then converts to Element's dtype.
pub(crate) fn get_typed_data<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    backend: &CpuBackend<E>,
    names: &[impl AsRef<str>],
) -> Result<(Vec<u8>, DType), BE> {
    let dtype = element_to_dtype::<E>()?;

    // Try native tensor first (raw bytes in Element's dtype)
    for name in names {
        let name = name.as_ref();
        if let Some(t) = weights.get_tensor(name) {
            let e_slice: &[E] = t.as_slice();
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    e_slice.as_ptr() as *const u8,
                    std::mem::size_of_val(e_slice),
                )
            };
            return Ok((bytes.to_vec(), dtype));
        }
    }

    // Try quantized tensor (dequantize to f32, then convert to Element's dtype)
    for name in names {
        let name = name.as_ref();
        if let Some(qt) = weights.get_quantized(name) {
            let f32_data = _dequantize_to_f32(qt, backend)?;
            return Ok((f32_to_typed_bytes(&f32_data, dtype), dtype));
        }
    }

    let name_strs: Vec<&str> = names.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("tensor not found: {:?}", name_strs)))
}

/// Convert WeightData to typed bytes in target dtype.
///
/// Quantized weights are dequantized then converted.
/// F32 weights are converted directly.
#[allow(dead_code)]
pub(crate) fn weight_data_to_typed<E: Element>(
    weight: &WeightData,
    backend: &CpuBackend<E>,
    transpose_weights: bool,
    out_dim: usize,
    in_dim: usize,
    dtype: DType,
) -> Result<Vec<u8>, BE> {
    let f32_data = weight_data_to_f32(weight, backend, transpose_weights, out_dim, in_dim)?;
    Ok(f32_to_typed_bytes(&f32_data, dtype))
}

/// Pack typed byte slices into a contiguous buffer (no dtype conversion).
///
/// All slices must already be in the correct dtype. This is the typed equivalent
/// of `pack_weights()` but operates on raw bytes instead of `&[f32]`.
#[allow(dead_code)]
pub(crate) fn pack_typed_byte_slices(slices: &[&[u8]]) -> Vec<u8> {
    let total: usize = slices.iter().map(|s| s.len()).sum();
    let mut buf = vec![0u8; total];
    let mut off = 0;
    for s in slices {
        buf[off..off + s.len()].copy_from_slice(s);
        off += s.len();
    }
    buf
}
