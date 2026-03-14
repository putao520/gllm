use super::backend_trait::{self, Backend};
use super::cpu_backend::CpuBackend;
use super::Element;
use crate::engine::executor::BackendError as BE;

/// Reinterpret a slice of Element as &[f32]. Only valid when E is f32.
pub(crate) fn as_f32_slice<E: Element>(data: &[E]) -> &[f32] {
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
            let blocks_per_row = (ne0 + blk_elems - 1) / blk_elems;
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

    // Probe a non-square SafeTensors / ONNX weight.
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
