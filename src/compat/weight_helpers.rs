use super::backend_trait::{self, Backend};
use super::cpu_backend::CpuBackend;
use super::Element;
use crate::engine::executor::BackendError as BE;

/// Reinterpret a slice of Element as &[f32]. Only valid when E is f32.
pub(crate) fn as_f32_slice<E: Element>(data: &[E]) -> &[f32] {
    debug_assert_eq!(
        std::mem::size_of::<E>(), std::mem::size_of::<f32>(),
        "as_f32_slice requires E to be f32-sized (got {} bytes)", std::mem::size_of::<E>()
    );
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) }
}

/// Try to get tensor data as an owned Vec<f32>. Tries each name in order.
/// Handles both native (f32) tensors and quantized (GGUF) tensors.
pub(crate) fn get_f32_data<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    backend: &CpuBackend<E>,
    names: &[impl AsRef<str>],
) -> Result<Vec<f32>, BE> {
    for name in names {
        let name = name.as_ref();
        if let Some(t) = weights.get_tensor(name) {
            let data = as_f32_slice(t.as_slice()).to_vec();
            return Ok(data);
        }
        if let Some(qt) = weights.get_quantized(name) {
            let n: usize = qt.shape.iter().product();
            let mut out = vec![0.0f32; n];
            let blk_bytes = qt.quant_type.block_bytes();
            let blk_elems = qt.quant_type.block_size();

            // GGUF pads each row (innermost dim) to the block boundary.
            // When ne0 % block_size != 0, we must dequantize row-by-row,
            // skipping the padding blocks at the end of each row.
            // GGUF shape is [ne0, ne1, ...] where ne0 is the innermost dim (first element).
            let ne0 = if qt.shape.is_empty() { n } else { qt.shape[0] };
            let blocks_per_row = ne0.div_ceil(blk_elems);
            let row_data_bytes = blocks_per_row * blk_bytes;
            let needs_row_dequant = ne0 % blk_elems != 0 && qt.shape.len() >= 2;

            if needs_row_dequant {
                let n_rows: usize = qt.shape[1..].iter().product();
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

            return Ok(out);
        }
    }
    let name_strs: Vec<&str> = names.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("tensor not found: {:?}", name_strs)))
}

/// Try to get bias data as Vec<f32>. Returns zeros if not found.
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

/// Detect whether linear-layer weights need transposing before GEMM.
///
/// Returns `true` when the dequantized flat array stores weights in
/// `[out_dim, in_dim]` row-major order (SafeTensors / PyTorch convention AND
/// GGUF convention — GGUF shape `[ne0=in, ne1=out]` is ne1 rows of ne0 cols,
/// i.e. `[out, in]` in memory).
///
/// Returns `false` only for genuine ONNX layout where the flat array is
/// already `[in_dim, out_dim]`.
pub(crate) fn needs_weight_transpose<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
) -> bool {
    // GGUF models use blk.* names — they always need transpose because
    // GGUF shape [ne0, ne1] stores ne1 rows of ne0 elements, giving
    // [out_dim, in_dim] in memory (same as SafeTensors).
    const GGUF_PROBE: &str = "blk.0.ffn_up.weight";
    if weights.tensor_shape(GGUF_PROBE).is_some() {
        return true;
    }

    // Probe decoder (Llama/Qwen/Mistral) style weights: self_attn.q_proj.weight
    // SafeTensors shape: [out_dim, in_dim] → [num_heads*head_dim, hidden] → first < second for GQA
    // but for q_proj with hidden=576, heads=9, head_dim=64: out=576, in=576 (square, skip)
    // Use mlp.gate_proj.weight: [inter_dim, hidden] → inter > hidden → true for SafeTensors
    let decoder_probes: &[&str] = &[
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "transformer.h.0.mlp.c_fc.weight",
    ];
    for name in decoder_probes {
        if let Some(shape) = weights.tensor_shape(name) {
            if shape.len() == 2 && shape[0] != shape[1] {
                // SafeTensors: [out_dim, in_dim] → gate_proj [inter, hidden], inter > hidden → true
                // If shape[0] > shape[1]: out_dim > in_dim → SafeTensors row-major → needs transpose
                return shape[0] > shape[1];
            }
        }
    }

    // Probe a non-square SafeTensors / ONNX weight (BERT style).
    let probe_names = crate::weight_names::layer_aliases(0, "intermediate.dense.weight", None);
    for name in &probe_names {
        if let Some(shape) = weights.tensor_shape(name) {
            if shape.len() == 2 && shape[0] != shape[1] {
                // SafeTensors: [out_dim(1536), in_dim(384)] → first > second → true
                // ONNX:        [in_dim(384), out_dim(1536)] → first < second → false
                return shape[0] > shape[1];
            }
        }
    }
    // Fallback: name-existence heuristic (original behaviour).
    // SafeTensors models use "embeddings.word_embeddings.weight" style names.
    crate::weight_names::has_any_embedding_weight(|n| {
        weights.get_tensor(n).is_some() || weights.get_quantized(n).is_some()
    })
}

/// Try to get tensor data as Vec<f32>. Returns None if not found (instead of Err).
pub(crate) fn try_get_f32_data<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    backend: &CpuBackend<E>,
    names: &[impl AsRef<str>],
) -> Option<Vec<f32>> {
    get_f32_data(weights, backend, names).ok()
}

/// Transpose a row-major matrix [rows, cols] → [cols, rows].
pub(crate) fn transpose_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Quantized linear acceleration
// ---------------------------------------------------------------------------

/// Result of looking up a weight: either quantized raw blocks or dequantized f32.
pub(crate) enum WeightData {
    /// Quantized weight: raw block bytes, quant type, and shape [out_dim, in_dim].
    Quantized {
        data: Vec<u8>,
        quant_type: backend_trait::QuantType,
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
                // GGUF shape: [ne0=in_dim, ne1=out_dim]
                // Memory layout: out_dim rows of in_dim elements (quantized blocks)
                let in_dim = qt.shape[0];
                let out_dim = qt.shape[1];
                return Ok(WeightData::Quantized {
                    data: qt.data.clone(),
                    quant_type: qt.quant_type,
                    out_dim,
                    in_dim,
                });
            }
            // 1D quantized tensor (norm weights etc.): dequantize to f32
            return Ok(WeightData::F32(get_f32_data(weights, backend, &[name])?));
        }
    }
    let name_strs: Vec<&str> = names.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("tensor not found: {:?}", name_strs)))
}

/// Perform a linear projection: output = input @ W^T, where W may be quantized.
///
/// - `input`: `[seq_len, in_dim]` row-major f32
/// - `output`: `[seq_len, out_dim]` row-major f32
///
/// When `weight` is `WeightData::Quantized`, calls `backend.quantized_matmul()`
/// directly on the raw block data, skipping dequantization and transpose.
///
/// When `weight` is `WeightData::F32`, uses JIT-compiled GEMM via
/// `jit_gemm` for hardware-optimal SIMD execution.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantized_linear<E: Element>(
    backend: &CpuBackend<E>,
    input: &[f32],
    weight: &WeightData,
    output: &mut [f32],
    seq_len: usize,
    out_dim: usize,
    in_dim: usize,
    transpose_weights: bool,
) -> Result<(), BE> {
    match weight {
        WeightData::Quantized { data, quant_type, out_dim: w_out, in_dim: w_in } => {
            if *w_out != out_dim || *w_in != in_dim {
                return Err(BE::Other(format!(
                    "quantized weight shape mismatch: expected [{}, {}], got [{}, {}]",
                    out_dim, in_dim, w_out, w_in
                )));
            }

            // quantized_matmul convention (gllm-kernels):
            //   weight_blocks: m rows of k quantized elements
            //   input: [n, k] row-major
            //   output: [m, n] row-major
            //   output[j, i] = dot(weight_row_j[k], input_row_i[k])
            //
            // m=out_dim, k=in_dim, n=seq_len
            // Result is [out_dim, seq_len], we need [seq_len, out_dim] → transpose
            let input_e: &[E] = unsafe {
                std::slice::from_raw_parts(input.as_ptr() as *const E, input.len())
            };
            let mut qmm_out = vec![E::from_f32(0.0); out_dim * seq_len];
            backend.quantized_matmul(
                data,
                input_e,
                &mut qmm_out,
                *quant_type,
                out_dim,
                seq_len,
                in_dim,
            )?;

            // Transpose [out_dim, seq_len] → [seq_len, out_dim]
            for s in 0..seq_len {
                for d in 0..out_dim {
                    output[s * out_dim + d] = qmm_out[d * seq_len + s].to_f32();
                }
            }
            Ok(())
        }
        WeightData::F32(w_data) => {
            let w: std::borrow::Cow<'_, [f32]> = if transpose_weights {
                std::borrow::Cow::Owned(transpose_f32(w_data, out_dim, in_dim))
            } else {
                std::borrow::Cow::Borrowed(w_data)
            };
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            {
                super::jit_helpers::jit_gemm(input, &w, output, seq_len, out_dim, in_dim, gllm_kernels::types::DType::F32)
                    .map_err(|e| BE::Other(format!("JIT GEMM failed: {e}")))?;
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                compile_error!("F32 GEMM requires JIT support (x86_64 or aarch64)");
            }
            Ok(())
        }
    }
}

/// Extract the f32 data from a WeightData, dequantizing if necessary.
/// Used for weights that cannot go through quantized_matmul (e.g. norm weights,
/// or weights needed for non-GEMM operations like KV cache update).
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
