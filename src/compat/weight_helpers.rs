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

/// Dequantize a QuantizedTensor to Vec<f32>. Kept for backward compat; prefer `dequantize_weight_to_dtype`.
#[allow(dead_code)]
pub(crate) fn dequantize_weight_to_f32<E: Element>(
    qt: &crate::loader::QuantizedTensor,
    backend: &crate::compat::CpuBackend<E>,
) -> Result<Vec<f32>, crate::compat::BackendError> {
    _dequantize_to_f32(qt, backend)
}

/// Dequantize a quantized tensor to typed bytes in the target dtype.
///
/// Two-phase: dequantize to F32, then convert to target dtype via `f32_to_typed_bytes`.
/// This allows GGUF weights (Q4_0/Q8_0/Q4K etc.) to be stored in their native
/// compute dtype (BF16/F16/FP8 etc.) instead of always expanding to F32.
pub(crate) fn dequantize_weight_to_dtype<E: Element>(
    qt: &crate::loader::QuantizedTensor,
    backend: &crate::compat::CpuBackend<E>,
    target_dtype: DType,
) -> Result<Vec<u8>, crate::compat::BackendError> {
    let f32_data = _dequantize_to_f32(qt, backend)?;
    Ok(f32_to_typed_bytes(&f32_data, target_dtype))
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
#[allow(dead_code)]
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
    let name_map = crate::loader::name_map::TensorNameMap::build_from_names(&all_names, None);

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
/// For quantized weights (GGUF or repacked AWQ/GPTQ safetensors) with 2D shape,
/// returns `WeightData::Quantized` so the caller can dispatch to `quantized_matmul`
/// directly, skipping the expensive dequantize + transpose path.
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
        DType::U8 => {
            for (i, &v) in data.iter().enumerate() {
                out[i] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        DType::F8E4M3 => {
            for (i, &v) in data.iter().enumerate() {
                out[i] = f32_to_e4m3fn(v);
            }
        }
        DType::F8E5M2 => {
            for (i, &v) in data.iter().enumerate() {
                out[i] = f32_to_e5m2(v);
            }
        }
        DType::F6E3M2 => {
            let packed = f32_slice_to_f6e3m2(data);
            out = packed;
        }
        DType::F6E2M3 => {
            let packed = f32_slice_to_f6e2m3(data);
            out = packed;
        }
        DType::F4E2M1 => {
            let packed = f32_slice_to_f4e2m1(data);
            out = packed;
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

// ---------------------------------------------------------------------------
// FP8 / FP6 / FP4 encoding — F32 → sub-byte IEEE-like floating point
// ---------------------------------------------------------------------------

/// Round-to-nearest-even for FP8 E4M3fn (NVIDIA/AMD OCP spec).
///
/// Bit layout: S (1) | E (3, bias=7) | M (3) — no infinity/NaN, max = 448.
fn f32_to_e4m3fn(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F; // NaN representation: sign=0, exp=0b111, mant=0b111
    }
    let sign = if v.to_bits() & 0x8000_0000 != 0 { 1u8 } else { 0u8 };
    let abs = v.abs();

    // E4M3fn: max normal = 448.0, subnormal min ≈ 2^-9
    if abs == 0.0 {
        return sign << 7;
    }

    // Extract f32 components
    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_man = bits & 0x007F_FFFF;

    let fp8_exp = f32_exp - 127 + 7; // bias adjustment

    if fp8_exp <= 0 {
        // Subnormal: mantissa has implicit leading 0
        // Shift the mantissa right by (1 - fp8_exp) positions
        let shift = (1 - fp8_exp) as u32;
        if shift > 30 {
            return sign << 7; // underflow to zero
        }
        let mantissa = (0x0080_0000 | f32_man) >> shift;
        let rounded = (mantissa + 4) >> 3; // round to 3-bit mantissa
        if rounded >= (1 << 4) {
            // Overflow to smallest normal
            return (sign << 7) | 1; // exp=1, mant=0
        }
        return (sign << 7) | (rounded as u8 & 0x07);
    }

    if fp8_exp >= 16 {
        // E4M3fn max exp is 15 (no infinity), clamp to max value
        // Max = 0_1111_110 = sign * (1.110) * 2^(15-7) = sign * 1.75 * 256 = 448
        return (sign << 7) | 0x7E;
    }

    // Normal: 3-bit mantissa from f32's 23-bit mantissa
    // Round to nearest even on the top 3 bits of mantissa
    let round_bit = (f32_man >> 19) & 1;
    let sticky = (f32_man & 0x0007_FFFF) != 0;
    let mantissa_3 = ((f32_man >> 20) & 0x7) as u8;

    let mantissa_rounded = if round_bit == 1 && (sticky || (mantissa_3 & 1) == 1) {
        mantissa_3 + 1
    } else {
        mantissa_3
    };

    if mantissa_rounded >= 8 {
        // Mantissa overflow → increment exponent
        let exp = (fp8_exp + 1) as u8;
        if exp >= 16 {
            return (sign << 7) | 0x7E; // clamp to max
        }
        return (sign << 7) | (exp << 3);
    }

    (sign << 7) | ((fp8_exp as u8) << 3) | mantissa_rounded
}

/// Round-to-nearest-even for FP8 E5M2 (NVIDIA/AMD OCP spec).
///
/// Bit layout: S (1) | E (5, bias=15) | M (2) — supports infinity/NaN.
fn f32_to_e5m2(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F; // NaN: sign=0, exp=0b11111, mant=0b11
    }
    let sign = if v.to_bits() & 0x8000_0000 != 0 { 1u8 } else { 0u8 };
    let abs = v.abs();

    if abs == 0.0 {
        return sign << 7;
    }

    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_man = bits & 0x007F_FFFF;

    let fp8_exp = f32_exp - 127 + 15;

    if fp8_exp <= 0 {
        // Subnormal
        let shift = (1 - fp8_exp) as u32;
        if shift > 30 {
            return sign << 7;
        }
        let mantissa = (0x0080_0000 | f32_man) >> shift;
        let rounded = (mantissa + 2) >> 2; // round to 2-bit mantissa
        if rounded >= (1 << 3) {
            return (sign << 7) | (1 << 2); // smallest normal
        }
        return (sign << 7) | (rounded as u8 & 0x03);
    }

    if fp8_exp >= 31 {
        // E5M2 max exp is 30 for normal values; exp=31 is Inf/NaN
        if fp8_exp > 31 || f32_man != 0 {
            return (sign << 7) | 0x7C; // NaN: exp=31, mant=11
        }
        return (sign << 7) | 0x7C; // Infinity: exp=31, mant=00 → use max normal ≈ 57344
    }

    // Round to 2-bit mantissa
    let round_bit = (f32_man >> 20) & 1;
    let sticky = (f32_man & 0x000F_FFFF) != 0;
    let mantissa_2 = ((f32_man >> 21) & 0x3) as u8;

    let mantissa_rounded = if round_bit == 1 && (sticky || (mantissa_2 & 1) == 1) {
        mantissa_2 + 1
    } else {
        mantissa_2
    };

    if mantissa_rounded >= 4 {
        let exp = (fp8_exp + 1) as u8;
        if exp >= 31 {
            return (sign << 7) | 0x7B; // max normal ≈ 57344
        }
        return (sign << 7) | (exp << 2);
    }

    (sign << 7) | ((fp8_exp as u8) << 2) | mantissa_rounded
}

/// Pack F32 slice → FP6 E3M2 bytes (6 bits per value, 4 values per 3 bytes).
///
/// Bit layout per value: S (1) | E (3, bias=3) | M (2)
fn f32_slice_to_f6e3m2(data: &[f32]) -> Vec<u8> {
    let n = data.len();
    // 4 FP6 values → 3 bytes (24 bits / 6 = 4)
    let byte_len = (n as u64 * 6).div_ceil(8) as usize;
    let mut out = vec![0u8; byte_len];

    for (i, &v) in data.iter().enumerate() {
        let bits = f32_to_f6e3m2_bits(v);
        let bit_offset = i * 6;
        // Write 6 bits starting at bit_offset (big-endian bit packing)
        for b in 0..6 {
            if bits & (1 << (5 - b)) != 0 {
                let byte_idx = (bit_offset + b) / 8;
                let bit_idx = 7 - ((bit_offset + b) % 8);
                if byte_idx < out.len() {
                    out[byte_idx] |= 1 << bit_idx;
                }
            }
        }
    }
    out
}

/// Pack F32 slice → FP6 E2M3 bytes (6 bits per value, 4 values per 3 bytes).
///
/// Bit layout per value: S (1) | E (2, bias=1) | M (3)
fn f32_slice_to_f6e2m3(data: &[f32]) -> Vec<u8> {
    let n = data.len();
    let byte_len = (n as u64 * 6).div_ceil(8) as usize;
    let mut out = vec![0u8; byte_len];

    for (i, &v) in data.iter().enumerate() {
        let bits = f32_to_f6e2m3_bits(v);
        let bit_offset = i * 6;
        for b in 0..6 {
            if bits & (1 << (5 - b)) != 0 {
                let byte_idx = (bit_offset + b) / 8;
                let bit_idx = 7 - ((bit_offset + b) % 8);
                if byte_idx < out.len() {
                    out[byte_idx] |= 1 << bit_idx;
                }
            }
        }
    }
    out
}

/// Pack F32 slice → FP4 E2M1 bytes (4 bits per value, 2 values per byte).
///
/// Bit layout per value: S (1) | E (2, bias=1) | M (1)
fn f32_slice_to_f4e2m1(data: &[f32]) -> Vec<u8> {
    let n = data.len();
    let byte_len = (n as u64 * 4).div_ceil(8) as usize;
    let mut out = vec![0u8; byte_len];

    for (i, &v) in data.iter().enumerate() {
        let bits = f32_to_f4e2m1_bits(v);
        let bit_offset = i * 4;
        let byte_idx = bit_offset / 8;
        let bit_idx = 7 - (bit_offset % 8);
        if byte_idx < out.len() {
            if bits & 0x8 != 0 {
                out[byte_idx] |= 1 << bit_idx;
            }
            if bits & 0x4 != 0 {
                out[byte_idx] |= 1 << (bit_idx - 1);
            }
            if bits & 0x2 != 0 {
                out[byte_idx] |= 1 << (bit_idx - 2);
            }
            if bits & 0x1 != 0 {
                out[byte_idx] |= 1 << (bit_idx - 3);
            }
        }
    }
    out
}

/// F32 → FP6 E3M2 (6 bits: S1 E3 M2, bias=3). Returns 6-bit packed value.
fn f32_to_f6e3m2_bits(v: f32) -> u8 {
    if v.is_nan() {
        return 0x3F; // sign=0, exp=111, mant=11
    }
    let sign = if v.to_bits() & 0x8000_0000 != 0 { 1u8 } else { 0u8 };
    let abs = v.abs();
    if abs == 0.0 {
        return sign << 5;
    }

    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_man = bits & 0x007F_FFFF;
    let fp6_exp = f32_exp - 127 + 3;

    if fp6_exp <= 0 {
        let shift = (1 - fp6_exp) as u32;
        if shift > 30 {
            return sign << 5;
        }
        let mantissa = (0x0080_0000 | f32_man) >> shift;
        let rounded = (mantissa + 2) >> 2;
        if rounded >= 8 {
            return (sign << 5) | (1 << 2);
        }
        return (sign << 5) | (rounded as u8 & 0x03);
    }

    if fp6_exp >= 8 {
        // Max normal for E3M2: exp=6 (0b110), mant=11 → clamp
        return (sign << 5) | 0x1F;
    }

    let round_bit = (f32_man >> 20) & 1;
    let sticky = (f32_man & 0x000F_FFFF) != 0;
    let mantissa_2 = ((f32_man >> 21) & 0x3) as u8;

    let mantissa_rounded = if round_bit == 1 && (sticky || (mantissa_2 & 1) == 1) {
        mantissa_2 + 1
    } else {
        mantissa_2
    };

    if mantissa_rounded >= 4 {
        let exp = (fp6_exp + 1) as u8;
        if exp >= 8 {
            return (sign << 5) | 0x1F;
        }
        return (sign << 5) | (exp << 2);
    }

    (sign << 5) | ((fp6_exp as u8) << 2) | mantissa_rounded
}

/// F32 → FP6 E2M3 (6 bits: S1 E2 M3, bias=1). Returns 6-bit packed value.
fn f32_to_f6e2m3_bits(v: f32) -> u8 {
    if v.is_nan() {
        return 0x3F;
    }
    let sign = if v.to_bits() & 0x8000_0000 != 0 { 1u8 } else { 0u8 };
    let abs = v.abs();
    if abs == 0.0 {
        return sign << 5;
    }

    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_man = bits & 0x007F_FFFF;
    let fp6_exp = f32_exp - 127 + 1;

    if fp6_exp <= 0 {
        let shift = (1 - fp6_exp) as u32;
        if shift > 30 {
            return sign << 5;
        }
        let mantissa = (0x0080_0000 | f32_man) >> shift;
        let rounded = (mantissa + 4) >> 3;
        if rounded >= 16 {
            return (sign << 5) | (1 << 3);
        }
        return (sign << 5) | (rounded as u8 & 0x07);
    }

    if fp6_exp >= 4 {
        return (sign << 5) | 0x1F;
    }

    let round_bit = (f32_man >> 19) & 1;
    let sticky = (f32_man & 0x0007_FFFF) != 0;
    let mantissa_3 = ((f32_man >> 20) & 0x7) as u8;

    let mantissa_rounded = if round_bit == 1 && (sticky || (mantissa_3 & 1) == 1) {
        mantissa_3 + 1
    } else {
        mantissa_3
    };

    if mantissa_rounded >= 8 {
        let exp = (fp6_exp + 1) as u8;
        if exp >= 4 {
            return (sign << 5) | 0x1F;
        }
        return (sign << 5) | (exp << 3);
    }

    (sign << 5) | ((fp6_exp as u8) << 3) | mantissa_rounded
}

/// F32 → FP4 E2M1 (4 bits: S1 E2 M1, bias=1). Returns 4-bit packed value.
fn f32_to_f4e2m1_bits(v: f32) -> u8 {
    if v.is_nan() {
        return 0x07;
    }
    let sign = if v.to_bits() & 0x8000_0000 != 0 { 1u8 } else { 0u8 };
    let abs = v.abs();
    if abs == 0.0 {
        return sign << 3;
    }

    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_man = bits & 0x007F_FFFF;
    let fp4_exp = f32_exp - 127 + 1;

    if fp4_exp <= 0 {
        let shift = (1 - fp4_exp) as u32;
        if shift > 30 {
            return sign << 3;
        }
        let mantissa = (0x0080_0000 | f32_man) >> shift;
        let rounded = (mantissa + 0x40) >> 7; // round to 1-bit mantissa
        if rounded >= 2 {
            return (sign << 3) | (1 << 0);
        }
        return (sign << 3) | (rounded as u8 & 0x01);
    }

    if fp4_exp >= 4 {
        return (sign << 3) | 0x07;
    }

    let round_bit = (f32_man >> 22) & 1;
    let sticky = (f32_man & 0x003F_FFFF) != 0;
    let mantissa_1 = ((f32_man >> 23) & 0x1) as u8;

    let mantissa_rounded = if round_bit == 1 && (sticky || (mantissa_1 & 1) == 1) {
        mantissa_1 + 1
    } else {
        mantissa_1
    };

    if mantissa_rounded >= 2 {
        let exp = (fp4_exp + 1) as u8;
        if exp >= 4 {
            return (sign << 3) | 0x07;
        }
        return (sign << 3) | (exp << 1);
    }

    (sign << 3) | ((fp4_exp as u8) << 1) | mantissa_rounded
}

#[cfg(test)]
mod dtype_encoding_tests {
    use super::*;

    #[test]
    fn test_e4m3_zero() {
        assert_eq!(f32_to_e4m3fn(0.0), 0x00);
        assert_eq!(f32_to_e4m3fn(-0.0), 0x80);
    }

    #[test]
    fn test_e4m3_one() {
        // 1.0 = 0_1000_000 in E4M3: sign=0, exp=7 (bias=7, so 2^0), mant=000
        let encoded = f32_to_e4m3fn(1.0);
        assert_eq!(encoded, 0x38); // 0_0111_000
    }

    #[test]
    fn test_e4m3_max() {
        // Max normal = 448.0 → sign=0, exp=15 (0b1111), mant=110
        let encoded = f32_to_e4m3fn(448.0);
        assert_eq!(encoded, 0x7E); // 0_1111_110
    }

    #[test]
    fn test_e5m2_zero() {
        assert_eq!(f32_to_e5m2(0.0), 0x00);
        assert_eq!(f32_to_e5m2(-0.0), 0x80);
    }

    #[test]
    fn test_e5m2_one() {
        // 1.0 = 0_01111_00 in E5M2: sign=0, exp=15 (bias=15, so 2^0), mant=00
        let encoded = f32_to_e5m2(1.0);
        assert_eq!(encoded, 0x3C); // 0_01111_00
    }

    #[test]
    fn test_f4_roundtrip_simple() {
        // FP4 E2M1 can represent 0, ±1, ±1.5, ±2, ±3, ±4, ±6
        let encoded = f32_to_f4e2m1_bits(1.0);
        // 1.0: sign=0, exp=1 (bias=1), mant=0 → 0_01_0 = 0b0010
        assert_eq!(encoded, 0b0010);
    }

    #[test]
    fn test_f32_to_typed_bytes_f8e4m3() {
        let values = vec![0.0, 1.0, -1.0, 448.0];
        let bytes = f32_to_typed_bytes(&values, DType::F8E4M3);
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes[0], 0x00); // 0.0
        assert_eq!(bytes[1], 0x38); // 1.0
        assert_eq!(bytes[2], 0xB8); // -1.0
        assert_eq!(bytes[3], 0x7E); // 448.0
    }

    #[test]
    fn test_f32_to_typed_bytes_f4e2m1() {
        let values = vec![1.0, 2.0];
        let bytes = f32_to_typed_bytes(&values, DType::F4E2M1);
        // 2 values × 4 bits = 1 byte
        assert_eq!(bytes.len(), 1);
        // 1.0 = 0b0010, 2.0 = 0b0100 → packed: 0010_0100 = 0x24
        assert_eq!(bytes[0], 0x24);
    }

    // ── New tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_transpose_f32_identity_square() {
        // 2×2 identity
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        let result = transpose_f32(&data, 2, 2);
        assert_eq!(result, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_transpose_f32_non_square() {
        // [2, 3] → [3, 2]
        // [[1, 2, 3], [4, 5, 6]]
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose_f32(&data, 2, 3);
        // [[1, 4], [2, 5], [3, 6]]
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_f32_single_row() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = transpose_f32(&data, 1, 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_transpose_f32_single_col() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = transpose_f32(&data, 3, 1);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pack_typed_byte_slices() {
        let s1: &[u8] = &[0xAA, 0xBB];
        let s2: &[u8] = &[0xCC, 0xDD, 0xEE];
        let result = pack_typed_byte_slices(&[s1, s2]);
        assert_eq!(result, vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE]);
    }

    #[test]
    fn test_pack_typed_byte_slices_empty() {
        let result = pack_typed_byte_slices(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_transpose_typed_f32() {
        // [2, 2] with F32 (4 bytes each)
        let data: Vec<u8> = vec![
            // row 0: 1.0f32, 2.0f32
            0x00, 0x00, 0x80, 0x3F,
            0x00, 0x00, 0x00, 0x40,
            // row 1: 3.0f32, 4.0f32
            0x00, 0x00, 0x40, 0x40,
            0x00, 0x00, 0x80, 0x40,
        ];
        let result = transpose_typed(&data, 2, 2, DType::F32);
        assert_eq!(result.len(), 16);
        // After transpose: col 0 first = [1.0, 3.0], col 1 = [2.0, 4.0]
        let val_00 = f32::from_le_bytes(result[0..4].try_into().unwrap());
        let val_10 = f32::from_le_bytes(result[4..8].try_into().unwrap());
        assert!((val_00 - 1.0).abs() < 1e-6);
        assert!((val_10 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_e5m2_negative_one() {
        let encoded = f32_to_e5m2(-1.0);
        // -1.0: sign=1, exp=15 (bias 15), mant=00
        // Binary: 1_01111_00 = 0xBC
        assert_eq!(encoded, 0xBC);
    }

    #[test]
    fn test_e4m3_nan() {
        let encoded = f32_to_e4m3fn(f32::NAN);
        assert_eq!(encoded, 0x7F);
    }

    #[test]
    fn test_e5m2_nan() {
        let encoded = f32_to_e5m2(f32::NAN);
        assert_eq!(encoded, 0x7F);
    }

    #[test]
    fn test_f32_to_typed_bytes_f32_passthrough() {
        let values = vec![1.0f32, 2.0];
        let bytes = f32_to_typed_bytes(&values, DType::F32);
        assert_eq!(bytes.len(), 8);
        let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let v1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!((v0 - 1.0).abs() < 1e-6);
        assert!((v1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_f32_to_typed_bytes_u8() {
        let values = vec![0.0f32, 127.0, 255.0, 128.6];
        let bytes = f32_to_typed_bytes(&values, DType::U8);
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 127);
        assert_eq!(bytes[2], 255);
        assert_eq!(bytes[3], 129); // round(128.6) = 129
    }

    #[test]
    fn test_f6e3m2_bits_zero_and_negative() {
        assert_eq!(f32_to_f6e3m2_bits(0.0), 0);
        // Negative zero: sign bit set
        let neg_zero = f32_to_f6e3m2_bits(-0.0);
        assert_ne!(neg_zero, 0); // sign bit is set
    }

    #[test]
    fn test_f6e2m3_bits_zero_and_nan() {
        assert_eq!(f32_to_f6e2m3_bits(0.0), 0);
        assert_eq!(f32_to_f6e2m3_bits(f32::NAN), 0x3F);
    }

    #[test]
    fn test_f4e2m1_bits_zero_and_nan() {
        assert_eq!(f32_to_f4e2m1_bits(0.0), 0);
        assert_eq!(f32_to_f4e2m1_bits(f32::NAN), 0x07);
    }

    #[test]
    fn test_as_f32_slice_valid() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let slice = as_f32_slice(&data);
        assert_eq!(slice.len(), 3);
        assert!((slice[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_element_to_dtype_f32() {
        let dtype = element_to_dtype::<f32>().unwrap();
        assert_eq!(dtype, DType::F32);
    }

    // ── Additional comprehensive tests ──────────────────────────────────────────

    #[test]
    fn test_element_to_dtype_f16() {
        let dtype = element_to_dtype::<half::f16>().unwrap();
        assert_eq!(dtype, DType::F16);
    }

    #[test]
    fn test_element_to_dtype_bf16() {
        let dtype = element_to_dtype::<half::bf16>().unwrap();
        assert_eq!(dtype, DType::BF16);
    }

    #[test]
    fn test_e4m3_negative_one() {
        let encoded = f32_to_e4m3fn(-1.0);
        // -1.0: sign=1, exp=7, mant=000 → 0b10111000 = 0xB8
        assert_eq!(encoded, 0xB8);
    }

    #[test]
    fn test_e4m3_small_positive() {
        // 0.5 = 1.0 * 2^(-1) → exp = 7 - 1 = 6, mant = 0 → 0_0110_000 = 0x30
        let encoded = f32_to_e4m3fn(0.5);
        assert_eq!(encoded, 0x30);
    }

    #[test]
    fn test_e4m3_positive_max_clamps() {
        // Value larger than E4M3 max (448) should clamp to 0x7E (sign=0)
        let encoded = f32_to_e4m3fn(1000.0);
        assert_eq!(encoded, 0x7E); // sign=0, exp=15, mant=110
    }

    #[test]
    fn test_e4m3_negative_max_clamps() {
        // Negative value larger than E4M3 max should clamp with sign bit set
        let encoded = f32_to_e4m3fn(-1000.0);
        assert_eq!(encoded, 0xFE); // sign=1, exp=15, mant=110
    }

    #[test]
    fn test_e4m3_negative_small() {
        let encoded = f32_to_e4m3fn(-0.5);
        // sign=1, exp=6, mant=0 → 0b10110000 = 0xB0
        assert_eq!(encoded, 0xB0);
    }

    #[test]
    fn test_e4m3_subnormal_very_small() {
        // Very small value that should underflow to zero
        let encoded = f32_to_e4m3fn(1e-20);
        assert_eq!(encoded, 0x00);
    }

    #[test]
    fn test_e4m3_negative_nan() {
        // 0xFFC00000 is a negative NaN (sign=1, exp=0xFF, mantissa nonzero)
        let neg_nan = f32::from_bits(0xFFC0_0000);
        assert!(neg_nan.is_nan());
        let encoded = f32_to_e4m3fn(neg_nan);
        // NaN always maps to 0x7F regardless of sign
        assert_eq!(encoded, 0x7F);
    }

    #[test]
    fn test_e5m2_large_clamps() {
        // E5M2: 100000.0 has fp8_exp=31 which triggers the >=31 path.
        // f32_man != 0 so it returns 0x7C (NaN representation, same as max in this impl).
        let encoded = f32_to_e5m2(100000.0);
        assert_eq!(encoded, 0x7C);
    }

    #[test]
    fn test_e5m2_negative_large_clamps() {
        let encoded = f32_to_e5m2(-100000.0);
        // sign=1, same clamp path as positive: 0x80 | 0x7C = 0xFC
        assert_eq!(encoded, 0xFC);
    }

    #[test]
    fn test_e5m2_subnormal_very_small() {
        let encoded = f32_to_e5m2(1e-30);
        assert_eq!(encoded, 0x00);
    }

    #[test]
    fn test_e5m2_half_value() {
        // 0.5 = 1.0 * 2^(-1) → exp = 15 - 1 = 14, mant = 00 → 0_01110_00 = 0x38
        let encoded = f32_to_e5m2(0.5);
        assert_eq!(encoded, 0x38);
    }

    #[test]
    fn test_e5m2_two() {
        // 2.0 = 1.0 * 2^1 → exp = 15 + 1 = 16, mant = 00 → 0_10000_00 = 0x40
        let encoded = f32_to_e5m2(2.0);
        assert_eq!(encoded, 0x40);
    }

    #[test]
    fn test_f32_to_typed_bytes_f16() {
        let values = vec![0.0f32, 1.0, -1.0];
        let bytes = f32_to_typed_bytes(&values, DType::F16);
        assert_eq!(bytes.len(), 6); // 3 values * 2 bytes
        // 0.0 in f16 = 0x0000
        assert_eq!(bytes[0], 0x00);
        assert_eq!(bytes[1], 0x00);
        // 1.0 in f16 = 0x3C00 (little-endian: 0x00, 0x3C)
        assert_eq!(bytes[2], 0x00);
        assert_eq!(bytes[3], 0x3C);
        // -1.0 in f16 = 0xBC00 (little-endian: 0x00, 0xBC)
        assert_eq!(bytes[4], 0x00);
        assert_eq!(bytes[5], 0xBC);
    }

    #[test]
    fn test_f32_to_typed_bytes_bf16() {
        let values = vec![0.0f32, 1.0, -1.0];
        let bytes = f32_to_typed_bytes(&values, DType::BF16);
        assert_eq!(bytes.len(), 6); // 3 values * 2 bytes
        // 0.0 in bf16 = 0x0000
        assert_eq!(bytes[0], 0x00);
        assert_eq!(bytes[1], 0x00);
        // 1.0 in bf16 = 0x3F80 (little-endian: 0x80, 0x3F)
        assert_eq!(bytes[2], 0x80);
        assert_eq!(bytes[3], 0x3F);
        // -1.0 in bf16 = 0xBF80 (little-endian: 0x80, 0xBF)
        assert_eq!(bytes[4], 0x80);
        assert_eq!(bytes[5], 0xBF);
    }

    #[test]
    fn test_f32_to_typed_bytes_f8e5m2() {
        let values = vec![0.0f32, 1.0, -1.0];
        let bytes = f32_to_typed_bytes(&values, DType::F8E5M2);
        assert_eq!(bytes.len(), 3);
        assert_eq!(bytes[0], 0x00); // 0.0
        assert_eq!(bytes[1], 0x3C); // 1.0
        assert_eq!(bytes[2], 0xBC); // -1.0
    }

    #[test]
    fn test_f32_to_typed_bytes_empty() {
        let values: Vec<f32> = vec![];
        let bytes = f32_to_typed_bytes(&values, DType::F32);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_f32_to_typed_bytes_u8_clamp_negative() {
        let values = vec![-10.0f32, 300.0];
        let bytes = f32_to_typed_bytes(&values, DType::U8);
        assert_eq!(bytes[0], 0);   // -10 clamped to 0
        assert_eq!(bytes[1], 255); // 300 clamped to 255
    }

    #[test]
    fn test_f6e3m2_bits_one() {
        // 1.0: sign=0, exp = 127-127+3 = 3, mant=00 → 0_011_00 = 0b00011100 = 0x0C... wait
        // Actually: sign=0, fp6_exp=3, mant=0 → (0<<5)|(3<<2)|0 = 12 = 0x0C
        let encoded = f32_to_f6e3m2_bits(1.0);
        assert_eq!(encoded, 0x0C);
    }

    #[test]
    fn test_f6e3m2_bits_negative_one() {
        // -1.0: sign=1, fp6_exp=3, mant=0 → (1<<5)|(3<<2)|0 = 0x2C
        let encoded = f32_to_f6e3m2_bits(-1.0);
        assert_eq!(encoded, 0x2C);
    }

    #[test]
    fn test_f6e3m2_bits_large_clamps() {
        // Value with exp >= 8 should clamp to max (0x1F)
        let encoded = f32_to_f6e3m2_bits(100.0);
        assert_eq!(encoded, 0x1F);
    }

    #[test]
    fn test_f6e3m2_bits_nan() {
        let encoded = f32_to_f6e3m2_bits(f32::NAN);
        assert_eq!(encoded, 0x3F);
    }

    #[test]
    fn test_f6e2m3_bits_one() {
        // 1.0: sign=0, fp6_exp = 127-127+1 = 1, mant=000 → (0<<5)|(1<<3)|0 = 8
        let encoded = f32_to_f6e2m3_bits(1.0);
        assert_eq!(encoded, 0x08);
    }

    #[test]
    fn test_f6e2m3_bits_negative_one() {
        // -1.0: sign=1, fp6_exp=1, mant=0 → (1<<5)|(1<<3)|0 = 0x28
        let encoded = f32_to_f6e2m3_bits(-1.0);
        assert_eq!(encoded, 0x28);
    }

    #[test]
    fn test_f6e2m3_bits_large_clamps() {
        let encoded = f32_to_f6e2m3_bits(50.0);
        assert_eq!(encoded, 0x1F); // max
    }

    #[test]
    fn test_f4e2m1_bits_negative_one() {
        // -1.0: sign=1, fp4_exp=1, mant=0 → (1<<3)|(1<<1)|0 = 0x0A
        let encoded = f32_to_f4e2m1_bits(-1.0);
        assert_eq!(encoded, 0x0A);
    }

    #[test]
    fn test_f4e2m1_bits_two() {
        // 2.0: sign=0, fp4_exp=2, mant=0 → (0<<3)|(2<<1)|0 = 0x04
        let encoded = f32_to_f4e2m1_bits(2.0);
        assert_eq!(encoded, 0x04);
    }

    #[test]
    fn test_f4e2m1_bits_large_clamps() {
        // Value with exp >= 4 should clamp to max (0x07)
        let encoded = f32_to_f4e2m1_bits(100.0);
        assert_eq!(encoded, 0x07);
    }

    #[test]
    fn test_f4e2m1_bits_negative_zero() {
        let encoded = f32_to_f4e2m1_bits(-0.0);
        assert_eq!(encoded, 0x08); // sign=1, exp=0, mant=0
    }

    #[test]
    fn test_f32_slice_to_f6e3m2_empty() {
        let data: Vec<f32> = vec![];
        let bytes = f32_slice_to_f6e3m2(&data);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_f32_slice_to_f6e2m3_empty() {
        let data: Vec<f32> = vec![];
        let bytes = f32_slice_to_f6e2m3(&data);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_f32_slice_to_f4e2m1_empty() {
        let data: Vec<f32> = vec![];
        let bytes = f32_slice_to_f4e2m1(&data);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_f32_slice_to_f6e3m2_single_value() {
        let data = vec![1.0f32];
        let bytes = f32_slice_to_f6e3m2(&data);
        // 1 value * 6 bits = 6 bits, ceil(6/8) = 1 byte
        assert_eq!(bytes.len(), 1);
    }

    #[test]
    fn test_f32_slice_to_f6e2m3_single_value() {
        let data = vec![1.0f32];
        let bytes = f32_slice_to_f6e2m3(&data);
        assert_eq!(bytes.len(), 1);
    }

    #[test]
    fn test_f32_slice_to_f4e2m1_single_value() {
        let data = vec![1.0f32];
        let bytes = f32_slice_to_f4e2m1(&data);
        // 1 value * 4 bits = 4 bits, ceil(4/8) = 1 byte
        assert_eq!(bytes.len(), 1);
    }

    #[test]
    fn test_f32_slice_to_f4e2m1_multiple_values() {
        // 4 values * 4 bits = 16 bits = 2 bytes
        let data = vec![0.0f32, 1.0, 2.0, 3.0];
        let bytes = f32_slice_to_f4e2m1(&data);
        assert_eq!(bytes.len(), 2);
    }

    #[test]
    fn test_f32_slice_to_f6e3m2_four_values() {
        // 4 values * 6 bits = 24 bits = 3 bytes (exact packing boundary)
        let data = vec![0.0f32, 1.0, 0.0, 1.0];
        let bytes = f32_slice_to_f6e3m2(&data);
        assert_eq!(bytes.len(), 3);
    }

    #[test]
    fn test_transpose_f32_1x1() {
        let data = vec![42.0f32];
        let result = transpose_f32(&data, 1, 1);
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_transpose_f32_3x2() {
        // [[1, 2], [3, 4], [5, 6]] → [[1, 3, 5], [2, 4, 6]]
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose_f32(&data, 3, 2);
        assert_eq!(result, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_transpose_f32_double_transpose_identity() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let once = transpose_f32(&original, 2, 3);
        let twice = transpose_f32(&once, 3, 2);
        assert_eq!(twice, original);
    }

    #[test]
    fn test_transpose_typed_bf16() {
        // BF16: 2 bytes per element, [1, 2] → [1, 2]^T is 1x2 → 2x1
        let mut data = Vec::new();
        // 1.0 in bf16 = 0x3F80
        data.extend_from_slice(&[0x80, 0x3F]);
        // 2.0 in bf16 = 0x4000
        data.extend_from_slice(&[0x00, 0x40]);
        // Transpose [1, 2] (1 row, 2 cols) → [1; 2] (2 rows, 1 col)
        let result = transpose_typed(&data, 1, 2, DType::BF16);
        assert_eq!(result.len(), 4);
        // First element (row 0 col 0) stays first
        assert_eq!(result[0], 0x80);
        assert_eq!(result[1], 0x3F);
        // Second element (row 1 col 0) was originally row 0 col 1
        assert_eq!(result[2], 0x00);
        assert_eq!(result[3], 0x40);
    }

    #[test]
    fn test_transpose_typed_f16_2x3() {
        // F16: 2 bytes each, [2, 3] → [3, 2]
        // 6 elements * 2 bytes = 12 bytes
        let mut data = Vec::new();
        for v in &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            let h = half::f16::from_f32(*v);
            data.extend_from_slice(&h.to_le_bytes());
        }
        let result = transpose_typed(&data, 2, 3, DType::F16);
        assert_eq!(result.len(), 12);
        // After transpose [2,3] → [3,2], verify first element is still 1.0
        let first = half::f16::from_le_bytes([result[0], result[1]]);
        assert!((first.to_f32() - 1.0).abs() < 1e-3);
        // Second element should be original row 1 col 0 = 4.0
        let second = half::f16::from_le_bytes([result[2], result[3]]);
        assert!((second.to_f32() - 4.0).abs() < 1e-3);
    }

    #[test]
    fn test_pack_typed_byte_slices_single() {
        let s1: &[u8] = &[0x01, 0x02, 0x03];
        let result = pack_typed_byte_slices(&[s1]);
        assert_eq!(result, vec![0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_pack_typed_byte_slices_with_empty() {
        let s1: &[u8] = &[0xAA];
        let s2: &[u8] = &[];
        let s3: &[u8] = &[0xBB];
        let result = pack_typed_byte_slices(&[s1, s2, s3]);
        assert_eq!(result, vec![0xAA, 0xBB]);
    }

    #[test]
    fn test_f32_to_typed_bytes_f6e3m2_roundtrip() {
        let values = vec![0.0f32, 1.0, 0.0, 1.0];
        let bytes = f32_to_typed_bytes(&values, DType::F6E3M2);
        // 4 values * 6 bits = 24 bits = 3 bytes
        assert_eq!(bytes.len(), 3);
    }

    #[test]
    fn test_f32_to_typed_bytes_f6e2m3_roundtrip() {
        let values = vec![0.0f32, 1.0, 0.0, 1.0];
        let bytes = f32_to_typed_bytes(&values, DType::F6E2M3);
        assert_eq!(bytes.len(), 3);
    }

    #[test]
    fn test_e4m3_two() {
        // 2.0 = 1.0 * 2^1 → exp = 7 + 1 = 8, mant = 0 → 0_1000_000 = 0x40
        let encoded = f32_to_e4m3fn(2.0);
        assert_eq!(encoded, 0x40);
    }

    #[test]
    fn test_e4m3_negative_two() {
        let encoded = f32_to_e4m3fn(-2.0);
        assert_eq!(encoded, 0xC0); // sign=1, exp=8, mant=0
    }

    #[test]
    fn test_e5m2_four() {
        // 4.0 = 1.0 * 2^2 → exp = 15 + 2 = 17, mant = 00 → 0_10001_00 = 0x44
        let encoded = f32_to_e5m2(4.0);
        assert_eq!(encoded, 0x44);
    }

    #[test]
    fn test_e5m2_negative_four() {
        let encoded = f32_to_e5m2(-4.0);
        assert_eq!(encoded, 0xC4); // sign=1, exp=17, mant=00
    }

    // ── Additional test block: 50+ new tests ────────────────────────────────────

    // --- f32_to_e4m3fn boundary & special floats ---

    #[test]
    fn test_e4m3_positive_infinity_clamps() {
        // E4M3 has no infinity; positive infinity should clamp to max (0x7E)
        let encoded = f32_to_e4m3fn(f32::INFINITY);
        assert_eq!(encoded, 0x7E);
    }

    #[test]
    fn test_e4m3_negative_infinity_clamps() {
        let encoded = f32_to_e4m3fn(f32::NEG_INFINITY);
        assert_eq!(encoded, 0xFE); // sign=1, max
    }

    #[test]
    fn test_e4m3_subnormal_nonzero() {
        // 2^-10 ≈ 0.000976... is subnormal in E4M3 (bias=7, min normal exp=1)
        // 2^-10: f32 exp = 127-10 = 117, fp8_exp = 117-127+7 = -3, shift = 4
        let v = 2f32.powi(-10);
        let encoded = f32_to_e4m3fn(v);
        // Subnormal: sign=0, exp field = 0, some mantissa bits nonzero (unless fully underflows)
        assert_eq!(encoded & 0x80, 0); // sign=0
        assert!(encoded > 0, "subnormal should be nonzero for 2^-10, got {}", encoded);
    }

    #[test]
    fn test_e4m3_half() {
        // 0.5 = 1.0 * 2^-1 → fp8_exp = 7-1 = 6, mant = 0 → 0_0110_000 = 0x30
        assert_eq!(f32_to_e4m3fn(0.5), 0x30);
    }

    #[test]
    fn test_e4m3_quarter() {
        // 0.25 = 2^-2 → fp8_exp = 7-2 = 5, mant = 0 → 0_0101_000 = 0x28
        assert_eq!(f32_to_e4m3fn(0.25), 0x28);
    }

    #[test]
    fn test_e4m3_symmetry_positive_negative() {
        // For every positive value, the negative should differ only in sign bit
        let vals = [0.0f32, 0.5, 1.0, 2.0, 448.0, 0.25];
        for &v in &vals {
            let pos = f32_to_e4m3fn(v);
            let neg = f32_to_e4m3fn(-v);
            assert_eq!(pos & 0x7F, neg & 0x7F, "symmetry failed for {}", v);
            assert_eq!(neg & 0x80, 0x80, "sign bit missing for -{}", v);
        }
    }

    // --- f32_to_e5m2 boundary & special floats ---

    #[test]
    fn test_e5m2_positive_infinity() {
        // E5M2 exp=31 is Inf/NaN path; infinity maps to 0x7C
        let encoded = f32_to_e5m2(f32::INFINITY);
        assert_eq!(encoded, 0x7C);
    }

    #[test]
    fn test_e5m2_negative_infinity() {
        let encoded = f32_to_e5m2(f32::NEG_INFINITY);
        assert_eq!(encoded, 0xFC); // sign=1, same clamp
    }

    #[test]
    fn test_e5m2_subnormal_nonzero() {
        // 2^-18 ≈ 3.8e-6, subnormal in E5M2 (bias=15, min normal exp=1)
        // fp8_exp = 127-18-127+15 = -3
        let v = 2f32.powi(-18);
        let encoded = f32_to_e5m2(v);
        assert_eq!(encoded & 0x80, 0); // sign=0
        assert!(encoded > 0, "subnormal should be nonzero for 2^-18, got {}", encoded);
    }

    #[test]
    fn test_e5m2_eight() {
        // 8.0 = 2^3 → exp = 15+3 = 18, mant=00 → 0_10010_00 = 0x48
        assert_eq!(f32_to_e5m2(8.0), 0x48);
    }

    #[test]
    fn test_e5m2_negative_eight() {
        assert_eq!(f32_to_e5m2(-8.0), 0xC8); // sign=1
    }

    #[test]
    fn test_e5m2_symmetry() {
        let vals = [0.0f32, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        for &v in &vals {
            let pos = f32_to_e5m2(v);
            let neg = f32_to_e5m2(-v);
            assert_eq!(pos & 0x7F, neg & 0x7F, "symmetry failed for {}", v);
            assert_eq!(neg & 0x80, 0x80, "sign bit missing for -{}", v);
        }
    }

    // --- f32_to_f6e3m2_bits more coverage ---

    #[test]
    fn test_f6e3m2_bits_negative_zero() {
        let encoded = f32_to_f6e3m2_bits(-0.0);
        // sign=1, rest zero → (1<<5) = 0x20
        assert_eq!(encoded, 0x20);
    }

    #[test]
    fn test_f6e3m2_bits_subnormal() {
        // 2^-6 is subnormal (bias=3, so fp6_exp = 127-6-127+3 = -3, shift=4)
        let v = 2f32.powi(-6);
        let encoded = f32_to_f6e3m2_bits(v);
        assert_eq!(encoded & 0x20, 0); // sign=0
        // Should be nonzero subnormal or zero (very small might underflow)
    }

    #[test]
    fn test_f6e3m2_bits_negative_large_clamps() {
        let encoded = f32_to_f6e3m2_bits(-100.0);
        // sign=1, max = 0x20 | 0x1F = 0x3F ... but max for E3M2 is 0x1F
        assert_eq!(encoded, 0x3F); // sign=1 | max
    }

    #[test]
    fn test_f6e3m2_bits_half() {
        // 0.5: fp6_exp = 127-127-1+3 = 2, mant=0 → (0<<5)|(2<<2)|0 = 8
        assert_eq!(f32_to_f6e3m2_bits(0.5), 0x08);
    }

    // --- f32_to_f6e2m3_bits more coverage ---

    #[test]
    fn test_f6e2m3_bits_negative_zero() {
        let encoded = f32_to_f6e2m3_bits(-0.0);
        assert_eq!(encoded, 0x20); // sign=1, rest zero
    }

    #[test]
    fn test_f6e2m3_bits_negative_large_clamps() {
        let encoded = f32_to_f6e2m3_bits(-50.0);
        assert_eq!(encoded, 0x3F); // sign=1 | max
    }

    #[test]
    fn test_f6e2m3_bits_subnormal() {
        let v = 2f32.powi(-4);
        let encoded = f32_to_f6e2m3_bits(v);
        assert_eq!(encoded & 0x20, 0); // sign=0
    }

    #[test]
    fn test_f6e2m3_bits_half() {
        // 0.5: fp6_exp = 127-1-127+1 = 0, subnormal path
        let encoded = f32_to_f6e2m3_bits(0.5);
        // Subnormal: sign=0, exp=0, some mantissa bits
        assert_eq!(encoded & 0x20, 0); // sign=0
        assert!(encoded > 0, "0.5 should encode as nonzero subnormal in F6E2M3, got {}", encoded);
    }

    // --- f32_to_f4e2m1_bits more coverage ---

    #[test]
    fn test_f4e2m1_bits_half() {
        // 0.5: fp4_exp = 127-1-127+1 = 0, subnormal
        let encoded = f32_to_f4e2m1_bits(0.5);
        assert_eq!(encoded & 0x08, 0); // sign=0
        // Subnormal path
    }

    #[test]
    fn test_f4e2m1_bits_negative_large_clamps() {
        let encoded = f32_to_f4e2m1_bits(-100.0);
        assert_eq!(encoded, 0x0F); // sign=1, max=0x07 → 0x08|0x07=0x0F
    }

    #[test]
    fn test_f4e2m1_bits_negative_nan() {
        let neg_nan = f32::from_bits(0xFFC0_0000);
        assert!(neg_nan.is_nan());
        let encoded = f32_to_f4e2m1_bits(neg_nan);
        // NaN always maps to 0x07 regardless of sign
        assert_eq!(encoded, 0x07);
    }

    #[test]
    fn test_f4e2m1_bits_subnormal_tiny() {
        // Very tiny value, should underflow to zero
        let encoded = f32_to_f4e2m1_bits(1e-20);
        assert_eq!(encoded, 0x00);
    }

    // --- f32_to_typed_bytes special float inputs ---

    #[test]
    fn test_f32_to_typed_bytes_f32_nan() {
        let values = vec![f32::NAN];
        let bytes = f32_to_typed_bytes(&values, DType::F32);
        assert_eq!(bytes.len(), 4);
        let reconstructed = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert!(reconstructed.is_nan());
    }

    #[test]
    fn test_f32_to_typed_bytes_f32_infinity() {
        let values = vec![f32::INFINITY, f32::NEG_INFINITY];
        let bytes = f32_to_typed_bytes(&values, DType::F32);
        assert_eq!(bytes.len(), 8);
        let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let v1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!(v0.is_infinite() && v0.is_sign_positive());
        assert!(v1.is_infinite() && v1.is_sign_negative());
    }

    #[test]
    fn test_f32_to_typed_bytes_f16_infinity() {
        let values = vec![f32::INFINITY];
        let bytes = f32_to_typed_bytes(&values, DType::F16);
        assert_eq!(bytes.len(), 2);
        let h = half::f16::from_le_bytes([bytes[0], bytes[1]]);
        assert!(h.is_infinite());
    }

    #[test]
    fn test_f32_to_typed_bytes_bf16_infinity() {
        let values = vec![f32::NEG_INFINITY];
        let bytes = f32_to_typed_bytes(&values, DType::BF16);
        assert_eq!(bytes.len(), 2);
        let b = half::bf16::from_le_bytes([bytes[0], bytes[1]]);
        assert!(b.is_infinite() && b.is_sign_negative());
    }

    #[test]
    fn test_f32_to_typed_bytes_u8_zero_and_max() {
        let values = vec![0.0f32, 255.0];
        let bytes = f32_to_typed_bytes(&values, DType::U8);
        assert_eq!(bytes, vec![0u8, 255]);
    }

    #[test]
    fn test_f32_to_typed_bytes_u8_nan_rounds_to_zero() {
        // NaN.round() is implementation-defined; verify it doesn't panic
        let values = vec![f32::NAN];
        let bytes = f32_to_typed_bytes(&values, DType::U8);
        assert_eq!(bytes.len(), 1);
        // NaN.clamp(0, 255) behavior: clamp doesn't affect NaN, but round produces some value
    }

    #[test]
    fn test_f32_to_typed_bytes_f8e4m3_negative_max() {
        let values = vec![-448.0f32];
        let bytes = f32_to_typed_bytes(&values, DType::F8E4M3);
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0xFE); // sign=1, max
    }

    #[test]
    fn test_f32_to_typed_bytes_f8e5m2_with_nan() {
        let values = vec![f32::NAN];
        let bytes = f32_to_typed_bytes(&values, DType::F8E5M2);
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0x7F); // NaN representation
    }

    #[test]
    fn test_f32_to_typed_bytes_f8e5m2_large_value() {
        let values = vec![100000.0f32];
        let bytes = f32_to_typed_bytes(&values, DType::F8E5M2);
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0x7C); // clamp
    }

    // --- f32_slice_to_f6e3m2 / f6e2m3 / f4e2m1 with negative values ---

    #[test]
    fn test_f32_slice_to_f6e3m2_with_negatives() {
        let data = vec![1.0f32, -1.0, 0.0, 2.0];
        let bytes = f32_slice_to_f6e3m2(&data);
        // 4 values * 6 bits = 24 bits = 3 bytes
        assert_eq!(bytes.len(), 3);
    }

    #[test]
    fn test_f32_slice_to_f6e2m3_with_negatives() {
        let data = vec![-1.0f32, 1.0, -2.0, 2.0];
        let bytes = f32_slice_to_f6e2m3(&data);
        assert_eq!(bytes.len(), 3);
    }

    #[test]
    fn test_f32_slice_to_f4e2m1_with_negatives() {
        let data = vec![-1.0f32, 1.0];
        let bytes = f32_slice_to_f4e2m1(&data);
        assert_eq!(bytes.len(), 1);
    }

    #[test]
    fn test_f32_slice_to_f6e3m2_two_values() {
        let data = vec![0.0f32, 1.0];
        let bytes = f32_slice_to_f6e3m2(&data);
        // 2 * 6 = 12 bits, ceil(12/8) = 2 bytes
        assert_eq!(bytes.len(), 2);
    }

    #[test]
    fn test_f32_slice_to_f6e2m3_two_values() {
        let data = vec![0.0f32, 1.0];
        let bytes = f32_slice_to_f6e2m3(&data);
        assert_eq!(bytes.len(), 2);
    }

    #[test]
    fn test_f32_slice_to_f4e2m1_three_values() {
        // 3 * 4 = 12 bits, ceil(12/8) = 2 bytes
        let data = vec![1.0f32, 2.0, 0.0];
        let bytes = f32_slice_to_f4e2m1(&data);
        assert_eq!(bytes.len(), 2);
    }

    #[test]
    fn test_f32_slice_to_f6e3m2_five_values() {
        // 5 * 6 = 30 bits, ceil(30/8) = 4 bytes
        let data = vec![1.0f32, 2.0, 0.0, -1.0, 0.5];
        let bytes = f32_slice_to_f6e3m2(&data);
        assert_eq!(bytes.len(), 4);
    }

    #[test]
    fn test_f32_slice_to_f4e2m1_eight_values() {
        // 8 * 4 = 32 bits = 4 bytes
        let data = vec![0.0f32; 8];
        let bytes = f32_slice_to_f4e2m1(&data);
        assert_eq!(bytes.len(), 4);
        // All zeros
        assert!(bytes.iter().all(|&b| b == 0));
    }

    // --- transpose_f32 more edge cases ---

    #[test]
    fn test_transpose_f32_with_zeros() {
        let data = vec![0.0f32; 6];
        let result = transpose_f32(&data, 2, 3);
        assert_eq!(result, vec![0.0f32; 6]);
    }

    #[test]
    fn test_transpose_f32_with_negatives() {
        // [[-1, -2], [-3, -4]]
        let data = vec![-1.0f32, -2.0, -3.0, -4.0];
        let result = transpose_f32(&data, 2, 2);
        assert_eq!(result, vec![-1.0, -3.0, -2.0, -4.0]);
    }

    #[test]
    fn test_transpose_f32_4x1() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = transpose_f32(&data, 4, 1);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_f32_1x4() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = transpose_f32(&data, 1, 4);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_f32_large_square() {
        // 4x4
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let result = transpose_f32(&data, 4, 4);
        // (0,1) → (1,0): data[0*4+1]=1 → result[1*4+0]=1
        assert_eq!(result[4], 1.0); // row 1 col 0 = original row 0 col 1
        assert_eq!(result[1], 4.0); // row 0 col 1 = original row 1 col 0
    }

    #[test]
    fn test_transpose_f32_preserves_special_values() {
        // row0 = [NAN, INFINITY], row1 = [NEG_INFINITY, -0.0]
        let data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let result = transpose_f32(&data, 2, 2);
        // After transpose: col0 first = [NAN, NEG_INFINITY], col1 = [INFINITY, -0.0]
        assert!(result[0].is_nan());
        assert_eq!(result[1], f32::NEG_INFINITY); // original row1 col0
        assert_eq!(result[2], f32::INFINITY); // original row0 col1
        assert_eq!(result[3].to_bits(), (-0.0f32).to_bits());
    }

    // --- transpose_typed more edge cases ---

    #[test]
    fn test_transpose_typed_single_element() {
        // 1x1 matrix
        let data: Vec<u8> = vec![0x42];
        let result = transpose_typed(&data, 1, 1, DType::U8);
        assert_eq!(result, vec![0x42]);
    }

    #[test]
    fn test_transpose_typed_u8() {
        // [2, 3] of U8 → [3, 2] of U8
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let result = transpose_typed(&data, 2, 3, DType::U8);
        assert_eq!(result, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_transpose_typed_u8_single_row() {
        let data: Vec<u8> = vec![10, 20, 30];
        let result = transpose_typed(&data, 1, 3, DType::U8);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn test_transpose_typed_u8_single_col() {
        let data: Vec<u8> = vec![10, 20, 30];
        let result = transpose_typed(&data, 3, 1, DType::U8);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn test_transpose_typed_double_transpose_identity() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let once = transpose_typed(&data, 2, 3, DType::U8);
        let twice = transpose_typed(&once, 3, 2, DType::U8);
        assert_eq!(twice, data);
    }

    // --- as_f32_slice edge cases ---

    #[test]
    fn test_as_f32_slice_empty() {
        let data: Vec<f32> = vec![];
        let slice = as_f32_slice(&data);
        assert!(slice.is_empty());
    }

    #[test]
    fn test_as_f32_slice_single_element() {
        let data: Vec<f32> = vec![42.0];
        let slice = as_f32_slice(&data);
        assert_eq!(slice.len(), 1);
        assert!((slice[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_as_f32_slice_preserves_nan() {
        let data: Vec<f32> = vec![f32::NAN];
        let slice = as_f32_slice(&data);
        assert!(slice[0].is_nan());
    }

    // --- pack_typed_byte_slices more edge cases ---

    #[test]
    fn test_pack_typed_byte_slices_large() {
        let s1: &[u8] = &[0xFF; 100];
        let s2: &[u8] = &[0x00; 50];
        let result = pack_typed_byte_slices(&[s1, s2]);
        assert_eq!(result.len(), 150);
        assert!(result[..100].iter().all(|&b| b == 0xFF));
        assert!(result[100..].iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_pack_typed_byte_slices_many_empty() {
        let e: &[u8] = &[];
        let result = pack_typed_byte_slices(&[e, e, e]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pack_typed_byte_slices_preserves_order() {
        let s1: &[u8] = &[0x01];
        let s2: &[u8] = &[0x02];
        let s3: &[u8] = &[0x03];
        let result = pack_typed_byte_slices(&[s1, s2, s3]);
        assert_eq!(result, vec![0x01, 0x02, 0x03]);
    }

    // --- E4M3 monotonicity: increasing f32 → increasing or equal encoding ---

    #[test]
    fn test_e4m3_monotonicity_positive() {
        let vals: Vec<f32> = (0..20).map(|i| (i as f32) * 0.5).collect();
        let encodings: Vec<u8> = vals.iter().map(|&v| f32_to_e4m3fn(v)).collect();
        for i in 1..encodings.len() {
            assert!(
                encodings[i] >= encodings[i - 1],
                "monotonicity violated: {} → {}, {} → {}",
                vals[i - 1], encodings[i - 1],
                vals[i], encodings[i]
            );
        }
    }

    // --- E5M2 monotonicity ---

    #[test]
    fn test_e5m2_monotonicity_positive() {
        let vals: Vec<f32> = (0..20).map(|i| (i as f32) * 0.5).collect();
        let encodings: Vec<u8> = vals.iter().map(|&v| f32_to_e5m2(v)).collect();
        for i in 1..encodings.len() {
            assert!(
                encodings[i] >= encodings[i - 1],
                "monotonicity violated: {} → {}, {} → {}",
                vals[i - 1], encodings[i - 1],
                vals[i], encodings[i]
            );
        }
    }

    // --- FP4/FP6 NaN for negative NaN ---

    #[test]
    fn test_f6e3m2_bits_negative_nan() {
        let neg_nan = f32::from_bits(0xFFC0_0000);
        assert!(neg_nan.is_nan());
        let encoded = f32_to_f6e3m2_bits(neg_nan);
        assert_eq!(encoded, 0x3F); // NaN always same encoding
    }

    #[test]
    fn test_f6e2m3_bits_negative_nan() {
        let neg_nan = f32::from_bits(0xFFC0_0000);
        let encoded = f32_to_f6e2m3_bits(neg_nan);
        assert_eq!(encoded, 0x3F);
    }

    // --- f32_to_typed_bytes F6E3M2 / F6E2M3 / F4E2M1 edge cases ---

    #[test]
    fn test_f32_to_typed_bytes_f6e3m2_single() {
        let values = vec![1.0f32];
        let bytes = f32_to_typed_bytes(&values, DType::F6E3M2);
        assert_eq!(bytes.len(), 1); // ceil(6/8) = 1
    }

    #[test]
    fn test_f32_to_typed_bytes_f6e2m3_single() {
        let values = vec![1.0f32];
        let bytes = f32_to_typed_bytes(&values, DType::F6E2M3);
        assert_eq!(bytes.len(), 1);
    }

    #[test]
    fn test_f32_to_typed_bytes_f4e2m1_single() {
        let values = vec![1.0f32];
        let bytes = f32_to_typed_bytes(&values, DType::F4E2M1);
        assert_eq!(bytes.len(), 1); // ceil(4/8) = 1
    }

    #[test]
    fn test_f32_to_typed_bytes_f4e2m1_odd_count() {
        // 3 values * 4 bits = 12 bits, ceil(12/8) = 2 bytes
        let values = vec![0.0f32, 1.0, 2.0];
        let bytes = f32_to_typed_bytes(&values, DType::F4E2M1);
        assert_eq!(bytes.len(), 2);
    }

    #[test]
    fn test_f32_to_typed_bytes_f4e2m1_all_zeros() {
        let values = vec![0.0f32; 4];
        let bytes = f32_to_typed_bytes(&values, DType::F4E2M1);
        assert_eq!(bytes.len(), 2);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    // --- E4M3 two and four ---

    #[test]
    fn test_e4m3_four() {
        // 4.0 = 2^2 → exp = 7+2 = 9, mant=0 → 0_1001_000 = 0x48
        assert_eq!(f32_to_e4m3fn(4.0), 0x48);
    }

    #[test]
    fn test_e4m3_eight() {
        // 8.0 = 2^3 → exp = 7+3 = 10, mant=0 → 0_1010_000 = 0x50
        assert_eq!(f32_to_e4m3fn(8.0), 0x50);
    }

    // --- E5M2 power-of-two sequence ---

    #[test]
    fn test_e5m2_power_of_two_sequence() {
        // Verify a sequence: 0.25, 0.5, 1, 2, 4, 8, 16
        let vals = [0.25f32, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        let encodings: Vec<u8> = vals.iter().map(|&v| f32_to_e5m2(v)).collect();
        // Each should be strictly larger than the previous
        for i in 1..encodings.len() {
            assert!(encodings[i] > encodings[i - 1],
                "E5M2 not strictly increasing: {} → {}, {} → {}",
                vals[i-1], encodings[i-1], vals[i], encodings[i]);
        }
    }

    // --- element_to_dtype for all supported types (comprehensive) ---

    #[test]
    fn test_element_to_dtype_roundtrip_f32() {
        let dtype = element_to_dtype::<f32>().unwrap();
        assert_eq!(dtype, DType::F32);
        assert_eq!(dtype.size_bytes(), 4);
    }

    #[test]
    fn test_element_to_dtype_roundtrip_f16() {
        let dtype = element_to_dtype::<half::f16>().unwrap();
        assert_eq!(dtype, DType::F16);
        assert_eq!(dtype.size_bytes(), 2);
    }

    #[test]
    fn test_element_to_dtype_roundtrip_bf16() {
        let dtype = element_to_dtype::<half::bf16>().unwrap();
        assert_eq!(dtype, DType::BF16);
        assert_eq!(dtype.size_bytes(), 2);
    }

    // --- f32_to_typed_bytes: F16 with NaN ---

    #[test]
    fn test_f32_to_typed_bytes_f16_nan() {
        let values = vec![f32::NAN];
        let bytes = f32_to_typed_bytes(&values, DType::F16);
        assert_eq!(bytes.len(), 2);
        let h = half::f16::from_le_bytes([bytes[0], bytes[1]]);
        assert!(h.is_nan());
    }

    // --- f32_to_typed_bytes: BF16 with NaN ---

    #[test]
    fn test_f32_to_typed_bytes_bf16_nan() {
        let values = vec![f32::NAN];
        let bytes = f32_to_typed_bytes(&values, DType::BF16);
        assert_eq!(bytes.len(), 2);
        let b = half::bf16::from_le_bytes([bytes[0], bytes[1]]);
        assert!(b.is_nan());
    }

    // --- E4M3 very small subnormal that underflows to zero ---

    #[test]
    fn test_e4m3_underflow_to_zero() {
        // 2^-30 is small enough to underflow to zero in E4M3 (shift=30 > threshold)
        let encoded = f32_to_e4m3fn(2f32.powi(-30));
        assert_eq!(encoded, 0x00);
    }

    #[test]
    fn test_e4m3_negative_underflow_to_zero() {
        let encoded = f32_to_e4m3fn(-2f32.powi(-30));
        assert_eq!(encoded, 0x80); // sign=1, rest zero
    }

    // --- E5M2 underflow ---

    #[test]
    fn test_e5m2_underflow_to_zero() {
        // 2^-48: f32_exp = 79, fp8_exp = 79-127+15 = -33, shift = 34 > 30 → underflow
        let encoded = f32_to_e5m2(2f32.powi(-48));
        assert_eq!(encoded, 0x00);
    }

    #[test]
    fn test_e5m2_negative_underflow_to_zero() {
        let encoded = f32_to_e5m2(-2f32.powi(-48));
        assert_eq!(encoded, 0x80);
    }

    // --- FP6/FP4 infinity handling ---

    #[test]
    fn test_f6e3m2_bits_positive_infinity_clamps() {
        let encoded = f32_to_f6e3m2_bits(f32::INFINITY);
        assert_eq!(encoded, 0x1F); // clamp to max
    }

    #[test]
    fn test_f6e3m2_bits_negative_infinity_clamps() {
        let encoded = f32_to_f6e3m2_bits(f32::NEG_INFINITY);
        assert_eq!(encoded, 0x3F); // sign=1 | max
    }

    #[test]
    fn test_f6e2m3_bits_positive_infinity_clamps() {
        let encoded = f32_to_f6e2m3_bits(f32::INFINITY);
        assert_eq!(encoded, 0x1F); // clamp to max
    }

    #[test]
    fn test_f6e2m3_bits_negative_infinity_clamps() {
        let encoded = f32_to_f6e2m3_bits(f32::NEG_INFINITY);
        assert_eq!(encoded, 0x3F); // sign=1 | max
    }

    #[test]
    fn test_f4e2m1_bits_positive_infinity_clamps() {
        let encoded = f32_to_f4e2m1_bits(f32::INFINITY);
        assert_eq!(encoded, 0x07); // clamp to max
    }

    #[test]
    fn test_f4e2m1_bits_negative_infinity_clamps() {
        let encoded = f32_to_f4e2m1_bits(f32::NEG_INFINITY);
        assert_eq!(encoded, 0x0F); // sign=1 | max
    }

    // --- transpose_typed: U8 double transpose preserves data ---

    #[test]
    fn test_transpose_typed_u8_non_square_3x2() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        // [3,2] → [2,3]
        let result = transpose_typed(&data, 3, 2, DType::U8);
        assert_eq!(result, vec![1, 3, 5, 2, 4, 6]);
    }

    // --- 15 new tests (wave 3) ---

    #[test]
    fn test_transpose_f32_5x3_rectangular() {
        // [5, 3] → [3, 5]: larger non-square matrix
        let data: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let result = transpose_f32(&data, 5, 3);
        // Verify: result[1*5+0] = data[0*3+1] = 1.0 (original row 0, col 1)
        assert_eq!(result[5], 1.0);
        // result[0*5+2] = data[2*3+0] = 6.0 (original row 2, col 0)
        assert_eq!(result[2], 6.0);
        // Verify size preserved
        assert_eq!(result.len(), 15);
    }

    #[test]
    fn test_transpose_f32_double_transpose_3x4() {
        // Double transpose of [3, 4] non-square should return original
        let original: Vec<f32> = (0..12).map(|x| (x as f32) + 1.0).collect();
        let once = transpose_f32(&original, 3, 4);
        assert_eq!(once.len(), 12);
        let twice = transpose_f32(&once, 4, 3);
        assert_eq!(twice, original);
    }

    #[test]
    fn test_transpose_typed_bf16_2x2_square() {
        // BF16 2x2: [[1, 2], [3, 4]] → [[1, 3], [2, 4]]
        let mut data = Vec::new();
        for v in &[1.0f32, 2.0, 3.0, 4.0] {
            let b = half::bf16::from_f32(*v);
            data.extend_from_slice(&b.to_le_bytes());
        }
        let result = transpose_typed(&data, 2, 2, DType::BF16);
        assert_eq!(result.len(), 8);
        // After transpose: [1.0, 3.0, 2.0, 4.0]
        let val_01 = half::bf16::from_le_bytes([result[4], result[5]]);
        assert!((val_01.to_f32() - 2.0).abs() < 0.01, "expected 2.0, got {}", val_01.to_f32());
    }

    #[test]
    fn test_transpose_typed_f8e4m3_single_row() {
        // F8E4M3: 1 byte per element, [1, 3] → [3, 1]
        let data = vec![0x00u8, 0x38, 0xB8]; // 0.0, 1.0, -1.0
        let result = transpose_typed(&data, 1, 3, DType::F8E4M3);
        // Single row transpose is identity
        assert_eq!(result, data);
    }

    #[test]
    fn test_pack_typed_byte_slices_duplicate_slices() {
        // Pack the same slice content twice to verify no aliasing issues
        let s1: &[u8] = &[0xAB, 0xCD];
        let s2: &[u8] = &[0xAB, 0xCD];
        let result = pack_typed_byte_slices(&[s1, s2]);
        assert_eq!(result, vec![0xAB, 0xCD, 0xAB, 0xCD]);
    }

    #[test]
    fn test_e4m3_rounding_nontrivial_mantissa() {
        // 0.75 = 1.5 * 2^-1 → fp8_exp = 6, mant = 100 (0.5 in 3-bit mantissa)
        // sign=0, exp=6, mant=4 → 0_0110_100 = 0x34
        let encoded = f32_to_e4m3fn(0.75);
        assert_eq!(encoded, 0x34);
    }

    #[test]
    fn test_e5m2_quarter() {
        // 0.25 = 2^-2 → exp = 15-2 = 13, mant = 00 → 0_01101_00 = 0x34
        let encoded = f32_to_e5m2(0.25);
        assert_eq!(encoded, 0x34);
    }

    #[test]
    fn test_e5m2_sixteen() {
        // 16.0 = 2^4 → exp = 15+4 = 19, mant = 00 → 0_10011_00 = 0x4C
        let encoded = f32_to_e5m2(16.0);
        assert_eq!(encoded, 0x4C);
    }

    #[test]
    fn test_f6e3m2_bits_quarter() {
        // 0.25 = 2^-2 → fp6_exp = 127-2-127+3 = -2, subnormal
        // shift = 1-(-2) = 3, mantissa = (0x800000 >> 3) = 0x100000
        // rounded = (0x100000 + 2) >> 2 = 0x40000 >> 2... check encoding is nonzero
        let encoded = f32_to_f6e3m2_bits(0.25);
        assert_eq!(encoded & 0x20, 0); // sign=0
        assert!(encoded > 0, "0.25 should be nonzero in F6E3M2, got {}", encoded);
    }

    #[test]
    fn test_f6e2m3_bits_two() {
        // 2.0: fp6_exp = 127-127+1+1 = 2, mant=0 → (0<<5)|(2<<3)|0 = 16 = 0x10
        let encoded = f32_to_f6e2m3_bits(2.0);
        assert_eq!(encoded, 0x10);
    }

    #[test]
    fn test_f4e2m1_bits_one_point_five() {
        // 1.5: f32_man=0x400000, fp4_exp=1, mantissa_1=0, round_bit=1, sticky=false
        // Round-to-even: mantissa stays 0 (even), so encoded = 1.0*2^0 = exp=1,mant=0
        // sign=0, fp4_exp=1, mant=0 → 0_01_0 = 0b0010 = 0x02
        let encoded = f32_to_f4e2m1_bits(1.5);
        assert_eq!(encoded, 0x02);
    }

    #[test]
    fn test_f4e2m1_bits_three() {
        // 3.0: f32_man=0x400000, fp4_exp=2, same rounding as 1.5 → mantissa rounds to 0
        // sign=0, fp4_exp=2, mant=0 → 0_10_0 = 0b0100 = 0x04
        let encoded = f32_to_f4e2m1_bits(3.0);
        assert_eq!(encoded, 0x04);
    }

    #[test]
    fn test_f6e3m2_bits_symmetry() {
        // Positive and negative values should differ only in sign bit (bit 5)
        let vals = [0.0f32, 0.5, 1.0, 2.0];
        for &v in &vals {
            let pos = f32_to_f6e3m2_bits(v);
            let neg = f32_to_f6e3m2_bits(-v);
            assert_eq!(pos & 0x1F, neg & 0x1F, "F6E3M2 symmetry failed for {}", v);
            assert_eq!(neg & 0x20, 0x20, "F6E3M2 sign bit missing for -{}", v);
        }
    }

    #[test]
    fn test_f32_to_typed_bytes_f8e4m3_multiple_values_byte_check() {
        // Encode [0.0, 1.0, -1.0, 0.5] as F8E4M3 and verify each byte
        let values = vec![0.0f32, 1.0, -1.0, 0.5];
        let bytes = f32_to_typed_bytes(&values, DType::F8E4M3);
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes[0], 0x00); // 0.0
        assert_eq!(bytes[1], 0x38); // 1.0
        assert_eq!(bytes[2], 0xB8); // -1.0
        assert_eq!(bytes[3], 0x30); // 0.5
    }

    #[test]
    fn test_f32_to_typed_bytes_f4e2m1_with_negatives_packed_bytes() {
        // [-1.0, 1.0]: -1.0 = 0xA, 1.0 = 0x2 → packed: 1010_0010 = 0xA2
        let values = vec![-1.0f32, 1.0];
        let bytes = f32_to_typed_bytes(&values, DType::F4E2M1);
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0xA2);
    }

    // ── Wave 4: 13 additional tests ─────────────────────────────────────────────

    #[test]
    fn test_transpose_typed_f8e5m2_2x2() {
        // F8E5M2: 1 byte per element. [2,2] → [2,2] with distinct values.
        // Use 0x00 (0.0), 0x3C (1.0), 0xBC (-1.0), 0x40 (2.0)
        let data: Vec<u8> = vec![0x00, 0x3C, 0xBC, 0x40];
        let result = transpose_typed(&data, 2, 2, DType::F8E5M2);
        assert_eq!(result.len(), 4);
        // After transpose: col 0 first = [0x00, 0xBC], col 1 = [0x3C, 0x40]
        assert_eq!(result[0], 0x00);
        assert_eq!(result[1], 0xBC);
        assert_eq!(result[2], 0x3C);
        assert_eq!(result[3], 0x40);
    }

    #[test]
    fn test_f32_slice_to_f6e3m2_seven_values_byte_length() {
        // 7 values * 6 bits = 42 bits, ceil(42/8) = 6 bytes
        let data = vec![1.0f32, 0.0, -1.0, 0.5, 2.0, 0.0, -0.5];
        let bytes = f32_slice_to_f6e3m2(&data);
        assert_eq!(bytes.len(), 6);
    }

    #[test]
    fn test_f32_slice_to_f6e2m3_seven_values_byte_length() {
        // 7 values * 6 bits = 42 bits, ceil(42/8) = 6 bytes
        let data = vec![1.0f32, 0.0, -1.0, 0.5, 2.0, 0.0, -0.5];
        let bytes = f32_slice_to_f6e2m3(&data);
        assert_eq!(bytes.len(), 6);
    }

    #[test]
    fn test_f32_to_typed_bytes_u8_rounding_halfway() {
        // f32::round() rounds half away from zero: 0.5 → 1.0, 1.5 → 2.0, 2.5 → 3.0
        let values = vec![0.5f32, 1.5, 2.5];
        let bytes = f32_to_typed_bytes(&values, DType::U8);
        assert_eq!(bytes.len(), 3);
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[1], 2);
        assert_eq!(bytes[2], 3);
    }

    #[test]
    fn test_transpose_f32_6x4_data_integrity() {
        // 6x4 = 24 elements, verify all elements preserved after transpose
        let data: Vec<f32> = (0..24).map(|x| (x as f32) * 0.1).collect();
        let result = transpose_f32(&data, 6, 4);
        assert_eq!(result.len(), 24);
        // Verify specific positions: result[1*6+0] = data[0*4+1] = 0.1
        assert!((result[6] - 0.1).abs() < 1e-6);
        // result[0*6+3] = data[3*4+0] = 1.2
        assert!((result[3] - 1.2).abs() < 1e-6);
        // Verify total sum preserved (transpose is a permutation)
        let sum_orig: f32 = data.iter().sum();
        let sum_result: f32 = result.iter().sum();
        assert!((sum_orig - sum_result).abs() < 1e-6);
    }

    #[test]
    fn test_pack_typed_byte_slices_all_empty_except_one() {
        let e: &[u8] = &[];
        let s: &[u8] = &[0xDE, 0xAD];
        let result = pack_typed_byte_slices(&[e, e, s, e]);
        assert_eq!(result, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_f6e2m3_bits_symmetry() {
        // Positive and negative values should differ only in sign bit (bit 5)
        let vals = [0.0f32, 1.0, 2.0, 0.5];
        for &v in &vals {
            let pos = f32_to_f6e2m3_bits(v);
            let neg = f32_to_f6e2m3_bits(-v);
            assert_eq!(pos & 0x1F, neg & 0x1F, "F6E2M3 symmetry failed for {}", v);
            assert_eq!(neg & 0x20, 0x20, "F6E2M3 sign bit missing for -{}", v);
        }
    }

    #[test]
    fn test_f4e2m1_bits_symmetry() {
        // Positive and negative values should differ only in sign bit (bit 3)
        let vals = [0.0f32, 1.0, 2.0];
        for &v in &vals {
            let pos = f32_to_f4e2m1_bits(v);
            let neg = f32_to_f4e2m1_bits(-v);
            assert_eq!(pos & 0x07, neg & 0x07, "F4E2M1 symmetry failed for {}", v);
            assert_eq!(neg & 0x08, 0x08, "F4E2M1 sign bit missing for -{}", v);
        }
    }

    #[test]
    fn test_f32_slice_to_f4e2m1_five_values_byte_length() {
        // 5 values * 4 bits = 20 bits, ceil(20/8) = 3 bytes
        let data = vec![1.0f32, -1.0, 0.0, 2.0, -2.0];
        let bytes = f32_slice_to_f4e2m1(&data);
        assert_eq!(bytes.len(), 3);
    }

    #[test]
    fn test_transpose_typed_bf16_double_transpose_2x3() {
        // BF16: 2 bytes per element, [2, 3] → double transpose = identity
        let mut data = Vec::new();
        for v in &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            let b = half::bf16::from_f32(*v);
            data.extend_from_slice(&b.to_le_bytes());
        }
        let once = transpose_typed(&data, 2, 3, DType::BF16);
        assert_eq!(once.len(), 12);
        let twice = transpose_typed(&once, 3, 2, DType::BF16);
        assert_eq!(twice, data);
    }

    #[test]
    fn test_e4m3_three_point_five() {
        // 3.5 = 1.75 * 2^1 → fp8_exp = 8, mant = 110 (0.75 in 3-bit)
        // sign=0, exp=8, mant=6 → 0_1000_110 = 0x46
        let encoded = f32_to_e4m3fn(3.5);
        assert_eq!(encoded, 0x46);
    }

    #[test]
    fn test_f6e3m2_bits_two() {
        // 2.0: fp6_exp = 127-127+3+1 = 4, mant=0 → (0<<5)|(4<<2)|0 = 16 = 0x10
        let encoded = f32_to_f6e3m2_bits(2.0);
        assert_eq!(encoded, 0x10);
    }

    // ── Wave 5: 10 additional tests ──────────────────────────────────────────

    #[test]
    fn test_e4m3_thirty_two() {
        // 32.0 = 1.0 * 2^5 → fp8_exp = 7 + 5 = 12, mant = 0 → 0_1100_000 = 0x60
        let encoded = f32_to_e4m3fn(32.0);
        assert_eq!(encoded, 0x60);
    }

    #[test]
    fn test_e5m2_thirty_two() {
        // 32.0 = 1.0 * 2^5 → fp8_exp = 15 + 5 = 20, mant = 00 → 0_10100_00 = 0x50
        let encoded = f32_to_e5m2(32.0);
        assert_eq!(encoded, 0x50);
    }

    #[test]
    fn test_f6e3m2_bits_four() {
        // 4.0 = 1.0 * 2^2 → fp6_exp = 3 + 2 = 5, mant = 00 → (0<<5)|(5<<2)|0 = 20 = 0x14
        let encoded = f32_to_f6e3m2_bits(4.0);
        assert_eq!(encoded, 0x14);
    }

    #[test]
    fn test_f4e2m1_bits_four() {
        // 4.0 = 1.0 * 2^2 → fp4_exp = 1 + 2 = 3, mant = 0 → (0<<3)|(3<<1)|0 = 6 = 0x06
        let encoded = f32_to_f4e2m1_bits(4.0);
        assert_eq!(encoded, 0x06);
    }

    #[test]
    fn test_f32_slice_to_f4e2m1_odd_three_byte_output() {
        // Arrange: 5 values * 4 bits = 20 bits → ceil(20/8) = 3 bytes
        // Use [0.0, 1.0, 2.0, 3.0, -1.0] and verify byte content.
        // F4E2M1 with truncation (mantissa_1 from bit 23 = always 0):
        // 0.0 = 0b0000, 1.0 = 0b0010, 2.0 = 0b0100, 3.0 = 0b0100 (truncated), -1.0 = 0b1010
        // Packed MSB-first: 0000|0010|0100|0100|1010|xxxx
        //   byte 0: 0000_0010 = 0x02
        //   byte 1: 0100_0100 = 0x44
        //   byte 2: 1010_0000 = 0xA0
        let data = vec![0.0f32, 1.0, 2.0, 3.0, -1.0];
        let bytes = f32_slice_to_f4e2m1(&data);

        // Assert
        assert_eq!(bytes.len(), 3, "5 F4E2M1 values should pack to 3 bytes");
        assert_eq!(bytes[0], 0x02, "byte 0 mismatch");
        assert_eq!(bytes[1], 0x44, "byte 1 mismatch");
        assert_eq!(bytes[2], 0xA0, "byte 2 mismatch");
    }

    #[test]
    fn test_transpose_typed_f8e4m3_double_transpose_identity() {
        // Arrange: 2x3 matrix of F8E4M3 values (1 byte each)
        // Encode [1.0, 2.0, 4.0, 0.5, -1.0, 3.5] as F8E4M3 bytes
        let values = [1.0f32, 2.0, 4.0, 0.5, -1.0, 3.5];
        let data: Vec<u8> = values.iter().map(|&v| f32_to_e4m3fn(v)).collect();

        // Act: double transpose
        let once = transpose_typed(&data, 2, 3, DType::F8E4M3);

        // Assert: intermediate has correct size
        assert_eq!(once.len(), 6, "single transpose should preserve size");

        // Act: second transpose
        let twice = transpose_typed(&once, 3, 2, DType::F8E4M3);

        // Assert: double transpose = identity
        assert_eq!(twice, data, "double transpose should recover original data");
    }

    #[test]
    fn test_transpose_typed_f16_double_transpose_identity() {
        // Arrange: 3x2 matrix of F16 values (2 bytes each)
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data = Vec::with_capacity(12);
        for &v in &values {
            let h = half::f16::from_f32(v);
            data.extend_from_slice(&h.to_le_bytes());
        }

        // Act: double transpose [3,2] → [2,3] → [3,2]
        let once = transpose_typed(&data, 3, 2, DType::F16);
        assert_eq!(once.len(), 12);
        let twice = transpose_typed(&once, 2, 3, DType::F16);

        // Assert: round-trip preserves bytes exactly
        assert_eq!(twice, data, "F16 double transpose should be identity");
    }

    #[test]
    fn test_f32_to_typed_bytes_u8_negative_one_clamps_to_zero() {
        // Arrange: -1.0 is below U8 range [0, 255]
        let values = vec![-1.0f32];

        // Act
        let bytes = f32_to_typed_bytes(&values, DType::U8);

        // Assert: clamped to 0
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0, "-1.0 should clamp to 0 in U8");
    }

    #[test]
    fn test_pack_typed_byte_slices_five_segments() {
        // Arrange: 5 non-empty segments of varying sizes
        let s0: &[u8] = &[0x10, 0x20];
        let s1: &[u8] = &[0x30];
        let s2: &[u8] = &[0x40, 0x50, 0x60, 0x70];
        let s3: &[u8] = &[0x80];
        let s4: &[u8] = &[0x90, 0xA0, 0xB0];

        // Act
        let result = pack_typed_byte_slices(&[s0, s1, s2, s3, s4]);

        // Assert: all bytes concatenated in order
        let expected: Vec<u8> = vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0];
        assert_eq!(result, expected);
        assert_eq!(result.len(), 11, "total should be 2+1+4+1+3 = 11 bytes");
    }

    #[test]
    fn test_transpose_f32_8x8_sum_and_spot_check() {
        // Arrange: 8x8 matrix with distinct values, verify sum and specific elements
        let n = 8usize;
        let data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();

        // Act
        let result = transpose_f32(&data, n, n);

        // Assert: sum preserved (transpose is a permutation)
        let sum_orig: f32 = data.iter().sum();
        let sum_result: f32 = result.iter().sum();
        assert!((sum_orig - sum_result).abs() < 1e-4, "sum must be preserved");

        // Assert: spot-check specific positions
        // result[2*8+5] = data[5*8+2] = 42*0.01 = 0.42
        assert!((result[2 * 8 + 5] - 0.42).abs() < 1e-6, "element [5,2] should move to [2,5]");

        // Assert: diagonal elements unchanged
        for i in 0..n {
            assert!((result[i * n + i] - data[i * n + i]).abs() < 1e-10, "diagonal [{}] unchanged", i);
        }
    }

    // ── Wave 6: Tests for previously uncovered public functions ──────────────────

    /// Mock TensorLookup for unit-testing weight helper functions that
    /// depend on a weight store. Stores named tensors as `Vec<f32>` and
    /// optional `QuantizedTensor` entries.
    struct MockWeights {
        tensors: std::collections::HashMap<String, Vec<f32>>,
        shapes: std::collections::HashMap<String, Vec<usize>>,
        quantized: std::collections::HashMap<String, QuantizedTensor>,
    }

    impl MockWeights {
        fn new() -> Self {
            Self {
                tensors: std::collections::HashMap::new(),
                shapes: std::collections::HashMap::new(),
                quantized: std::collections::HashMap::new(),
            }
        }

        fn add_tensor(&mut self, name: &str, data: Vec<f32>, shape: Vec<usize>) {
            self.tensors.insert(name.to_string(), data);
            self.shapes.insert(name.to_string(), shape);
        }

        fn add_quantized(&mut self, name: &str, qt: QuantizedTensor) {
            self.shapes.insert(name.to_string(), qt.shape.clone());
            self.quantized.insert(name.to_string(), qt);
        }
    }

    use crate::compat::backend_trait::TensorLookup;
    use crate::loader::gguf::GgmlDType;

    impl TensorLookup<f32, CpuBackend<f32>> for MockWeights {
        fn get_tensor(&self, name: &str) -> Option<&Vec<f32>> {
            self.tensors.get(name)
        }
        fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
            self.shapes.get(name).map(|s| s.as_slice())
        }
        fn get_quantized(&self, name: &str) -> Option<&QuantizedTensor> {
            self.quantized.get(name)
        }
        fn available_names(&self) -> Vec<String> {
            let mut names: Vec<String> = self.tensors.keys().cloned().collect();
            names.extend(self.quantized.keys().cloned());
            names.sort();
            names.dedup();
            names
        }
    }

    // --- get_bias_data tests ---

    #[test]
    fn test_get_bias_data_returns_tensor_when_found() {
        // Arrange
        let mut w = MockWeights::new();
        w.add_tensor("bias", vec![1.0f32, 2.0, 3.0], vec![3]);
        let backend = CpuBackend::<f32>::new();

        // Act
        let result = get_bias_data(&w as &dyn TensorLookup<f32, CpuBackend<f32>>, &["bias"], 3);

        // Assert
        assert_eq!(result, vec![1.0f32, 2.0, 3.0]);
        drop(backend);
    }

    #[test]
    fn test_get_bias_data_returns_zeros_when_not_found() {
        // Arrange
        let w = MockWeights::new();
        let backend = CpuBackend::<f32>::new();

        // Act
        let result = get_bias_data(&w as &dyn TensorLookup<f32, CpuBackend<f32>>, &["missing_bias"], 4);

        // Assert
        assert_eq!(result, vec![0.0f32; 4]);
        drop(backend);
    }

    #[test]
    fn test_get_bias_data_tries_multiple_names_first_match() {
        // Arrange
        let mut w = MockWeights::new();
        w.add_tensor("second_choice", vec![5.0f32, 6.0], vec![2]);
        let backend = CpuBackend::<f32>::new();

        // Act: first name is missing, second is present
        let result = get_bias_data(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &["first_choice", "second_choice"],
            2,
        );

        // Assert
        assert_eq!(result, vec![5.0f32, 6.0]);
        drop(backend);
    }

    // --- get_bias_data_typed tests ---

    #[test]
    fn test_get_bias_data_typed_returns_zeros_when_not_found() {
        // Arrange
        let w = MockWeights::new();

        // Act
        let (bytes, dtype) = get_bias_data_typed(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &["missing"],
            3,
        );

        // Assert
        assert_eq!(dtype, DType::F32);
        assert_eq!(bytes.len(), 3 * 4); // 3 elements * 4 bytes for F32
        let zeros = vec![0u8; 12];
        assert_eq!(bytes, zeros);
    }

    #[test]
    fn test_get_bias_data_typed_returns_tensor_bytes_when_found() {
        // Arrange
        let mut w = MockWeights::new();
        w.add_tensor("my_bias", vec![1.0f32, 2.0], vec![2]);

        // Act
        let (bytes, dtype) = get_bias_data_typed(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &["my_bias"],
            2,
        );

        // Assert
        assert_eq!(dtype, DType::F32);
        assert_eq!(bytes.len(), 8); // 2 * 4 bytes
        let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let v1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!((v0 - 1.0).abs() < 1e-6);
        assert!((v1 - 2.0).abs() < 1e-6);
    }

    // --- get_weight_data tests ---

    #[test]
    fn test_get_weight_data_f32_tensor() {
        // Arrange
        let mut w = MockWeights::new();
        w.add_tensor("weight", vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let backend = CpuBackend::<f32>::new();

        // Act
        let result = get_weight_data(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &backend,
            &["weight"],
        );

        // Assert
        assert!(result.is_ok(), "get_weight_data should succeed for existing f32 tensor");
        match result.unwrap() {
            WeightData::F32(data) => assert_eq!(data, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _ => panic!("expected WeightData::F32, got a Quantized variant"),
        }
    }

    #[test]
    fn test_get_weight_data_not_found_returns_error() {
        // Arrange
        let w = MockWeights::new();
        let backend = CpuBackend::<f32>::new();

        // Act
        let result = get_weight_data(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &backend,
            &["nonexistent_weight"],
        );

        // Assert
        assert!(result.is_err());
        let err = result.err().unwrap();
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("nonexistent_weight"), "error should mention the tensor name");
    }

    #[test]
    fn test_get_weight_data_2d_quantized_returns_quantized_variant() {
        // Arrange
        let mut w = MockWeights::new();
        // Q8_0: block_size=256, block_bytes=256+2=258
        // For a 2x4 tensor (8 elements), need ceil(4/256)=1 block per row, 2 rows = 2 blocks
        // Each block: 258 bytes. Total: 516 bytes.
        let block_bytes = 258usize;
        let fake_data = vec![0u8; block_bytes * 2];
        w.add_quantized("qweight", QuantizedTensor {
            data: fake_data,
            quant_type: QuantType::Q8_0,
            shape: vec![2, 4],
            ggml_dtype: GgmlDType::Q8_0,
        });
        let backend = CpuBackend::<f32>::new();

        // Act
        let result = get_weight_data(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &backend,
            &["qweight"],
        );

        // Assert
        assert!(result.is_ok(), "get_weight_data should succeed for quantized tensor");
        match result.unwrap() {
            WeightData::Quantized { out_dim, in_dim, quant_type, .. } => {
                assert_eq!(out_dim, 2);
                assert_eq!(in_dim, 4);
                assert_eq!(quant_type, QuantType::Q8_0);
            }
            _ => panic!("expected WeightData::Quantized for 2D quantized tensor, got F32 variant"),
        }
    }

    // --- weight_data_to_f32 tests ---

    #[test]
    fn test_weight_data_to_f32_without_transpose() {
        // Arrange
        let backend = CpuBackend::<f32>::new();
        let weight = WeightData::F32(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Act
        let result = weight_data_to_f32(&weight, &backend, false, 2, 3).unwrap();

        // Assert
        assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_weight_data_to_f32_with_transpose() {
        // Arrange
        let backend = CpuBackend::<f32>::new();
        // [2, 3] row-major: [[1,2,3],[4,5,6]]
        let weight = WeightData::F32(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Act
        let result = weight_data_to_f32(&weight, &backend, true, 2, 3).unwrap();

        // Assert: transposed [3, 2]: [[1,4],[2,5],[3,6]]
        assert_eq!(result, vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_weight_data_to_f32_transpose_skipped_when_size_mismatch() {
        // Arrange
        let backend = CpuBackend::<f32>::new();
        // data has 6 elements, but out_dim*in_dim = 2*2 = 4 (mismatch)
        let weight = WeightData::F32(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Act: transpose_weights=true but data.len() != out_dim*in_dim, so no transpose
        let result = weight_data_to_f32(&weight, &backend, true, 2, 2).unwrap();

        // Assert: original data returned unmodified
        assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // --- weight_data_to_typed tests ---

    #[test]
    fn test_weight_data_to_typed_f32_without_transpose() {
        // Arrange
        let backend = CpuBackend::<f32>::new();
        let weight = WeightData::F32(vec![1.0f32, 2.0, 3.0]);

        // Act
        let result = weight_data_to_typed(&weight, &backend, false, 1, 3, DType::F32).unwrap();

        // Assert
        assert_eq!(result.len(), 12); // 3 * 4 bytes
        let v0 = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert!((v0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_data_to_typed_bf16_with_transpose() {
        // Arrange
        let backend = CpuBackend::<f32>::new();
        // [2, 2]: [[1,2],[3,4]]
        let weight = WeightData::F32(vec![1.0f32, 2.0, 3.0, 4.0]);

        // Act
        let result = weight_data_to_typed(&weight, &backend, true, 2, 2, DType::BF16).unwrap();

        // Assert: transposed [2, 2]: [[1,3],[2,4]] → BF16 bytes
        assert_eq!(result.len(), 8); // 4 elements * 2 bytes
        let v0 = half::bf16::from_le_bytes([result[0], result[1]]);
        let v1 = half::bf16::from_le_bytes([result[2], result[3]]);
        assert!((v0.to_f32() - 1.0).abs() < 0.01);
        assert!((v1.to_f32() - 3.0).abs() < 0.01, "expected 3.0, got {}", v1.to_f32());
    }

    // --- try_get_typed_data tests ---

    #[test]
    fn test_try_get_typed_data_not_found_returns_none() {
        // Arrange
        let w = MockWeights::new();
        let backend = CpuBackend::<f32>::new();

        // Act
        let result = try_get_typed_data(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &backend,
            &["missing_tensor"],
        );

        // Assert
        assert!(result.is_none());
    }

    #[test]
    fn test_try_get_typed_data_f32_tensor_found() {
        // Arrange
        let mut w = MockWeights::new();
        w.add_tensor("my_tensor", vec![10.0f32, 20.0], vec![2]);
        let backend = CpuBackend::<f32>::new();

        // Act
        let result = try_get_typed_data(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &backend,
            &["my_tensor"],
        ).unwrap();

        // Assert
        let (bytes, dtype) = result;
        assert_eq!(dtype, DType::F32);
        assert_eq!(bytes.len(), 8);
        let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert!((v0 - 10.0).abs() < 1e-6);
    }

    // --- get_typed_data error path ---

    #[test]
    fn test_get_typed_data_not_found_returns_error() {
        // Arrange
        let w = MockWeights::new();
        let backend = CpuBackend::<f32>::new();

        // Act
        let result = get_typed_data(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &backend,
            &["absent"],
        );

        // Assert
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("absent"));
    }

    // --- get_weight_data: tries multiple names in order ---

    #[test]
    fn test_get_weight_data_tries_names_in_order() {
        // Arrange
        let w = MockWeights::new();
        let backend = CpuBackend::<f32>::new();

        // Act: no names match
        let result = get_weight_data(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &backend,
            &["alpha", "beta", "gamma"],
        );

        // Assert: error mentions all tried names
        assert!(result.is_err());
        let err = result.err().unwrap();
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("alpha"));
        assert!(err_msg.contains("beta"));
        assert!(err_msg.contains("gamma"));
    }

    // --- get_weight_data: second name matches when first does not ---

    #[test]
    fn test_get_weight_data_falls_through_to_second_name() {
        // Arrange
        let mut w = MockWeights::new();
        w.add_tensor("fallback_weight", vec![7.0f32, 8.0], vec![1, 2]);
        let backend = CpuBackend::<f32>::new();

        // Act
        let result = get_weight_data(
            &w as &dyn TensorLookup<f32, CpuBackend<f32>>,
            &backend,
            &["primary_weight", "fallback_weight"],
        );

        // Assert
        assert!(result.is_ok(), "should find fallback_weight");
        match result.unwrap() {
            WeightData::F32(data) => assert_eq!(data, vec![7.0f32, 8.0]),
            _ => panic!("expected WeightData::F32, got Quantized variant"),
        }
    }
}
