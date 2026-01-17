//! Quantized matmul kernels for CPU, with a backend abstraction for future GPU kernels.

use half::f16;

use gllm_kernels::KernelDispatcher;
use std::sync::OnceLock;

const Q4_BLOCK: usize = 32;
const Q4_PACKED: usize = 16;
const AWQ_PACKED: usize = 8;
const Q4_OUT_BLOCK: usize = 32;
const AWQ_OUT_BLOCK: usize = 16;

/// Lightweight view of an input matrix in row-major order.
#[derive(Clone, Copy)]
pub struct MatmulInput<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

impl<'a> MatmulInput<'a> {
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }
}

/// Quantized backend abstraction for CPU/GPU implementations.
pub trait QuantizedBackend {
    fn q4_matmul(input: MatmulInput<'_>, qweight: &[u8], scales: &[f16]) -> Vec<f32>;
    fn awq_matmul(
        input: MatmulInput<'_>,
        qweight: &[u32],
        scales: &[f16],
        zeros: &[u32],
        group_size: usize,
    ) -> Vec<f32>;
}

/// Optimized CPU backend for quantized matmul.
pub struct CpuQuantizedBackend;

/// GPU-accelerated backend for quantized matmul (falls back to CPU for now).
pub struct GpuQuantizedBackend;

/// Default backend uses GPU dispatcher for hardware detection.
pub type DefaultQuantizedBackend = GpuQuantizedBackend;

fn quantized_dispatcher() -> &'static KernelDispatcher {
    static DISPATCHER: OnceLock<KernelDispatcher> = OnceLock::new();
    DISPATCHER.get_or_init(KernelDispatcher::new)
}

impl QuantizedBackend for CpuQuantizedBackend {
    fn q4_matmul(input: MatmulInput<'_>, qweight: &[u8], scales: &[f16]) -> Vec<f32> {
        if input.rows == 0 || input.cols == 0 {
            return Vec::new();
        }
        if input.cols % Q4_BLOCK != 0 {
            return Vec::new();
        }
        let blocks_per_row = input.cols / Q4_BLOCK;
        if blocks_per_row == 0 || scales.len() % blocks_per_row != 0 {
            return Vec::new();
        }
        let out_features = scales.len() / blocks_per_row;
        let expected_qweight = out_features * blocks_per_row * Q4_PACKED;
        if qweight.len() != expected_qweight {
            return Vec::new();
        }

        let mut output = vec![0.0f32; input.rows * out_features];
        let mut dequant_block = vec![0.0f32; Q4_OUT_BLOCK * Q4_BLOCK];

        for out_start in (0..out_features).step_by(Q4_OUT_BLOCK) {
            let out_end = (out_start + Q4_OUT_BLOCK).min(out_features);
            let out_block = out_end - out_start;

            for block in 0..blocks_per_row {
                let in_offset = block * Q4_BLOCK;

                for ob in 0..out_block {
                    let out_row = out_start + ob;
                    let scale = scales[out_row * blocks_per_row + block].to_f32();
                    let q_offset = (out_row * blocks_per_row + block) * Q4_PACKED;
                    let qbytes = &qweight[q_offset..q_offset + Q4_PACKED];
                    let slice = &mut dequant_block[ob * Q4_BLOCK..(ob + 1) * Q4_BLOCK];
                    decode_q4_0_block(qbytes, scale, slice);
                }

                for row in 0..input.rows {
                    let in_row_start = row * input.cols + in_offset;
                    let in_slice =
                        &input.data[in_row_start..in_row_start + Q4_BLOCK];
                    let out_base = row * out_features + out_start;

                    for ob in 0..out_block {
                        let w = &dequant_block[ob * Q4_BLOCK..(ob + 1) * Q4_BLOCK];
                        output[out_base + ob] += dot32(in_slice, w);
                    }
                }
            }
        }

        output
    }

    fn awq_matmul(
        input: MatmulInput<'_>,
        qweight: &[u32],
        scales: &[f16],
        zeros: &[u32],
        group_size: usize,
    ) -> Vec<f32> {
        if input.rows == 0 || input.cols == 0 || group_size == 0 {
            return Vec::new();
        }
        if input.cols % group_size != 0 || group_size % AWQ_PACKED != 0 {
            return Vec::new();
        }
        if input.cols % AWQ_PACKED != 0 {
            return Vec::new();
        }

        let group_count = input.cols / group_size;
        if group_count == 0 || scales.len() % group_count != 0 {
            return Vec::new();
        }
        let out_features = scales.len() / group_count;
        if out_features == 0 || out_features % AWQ_PACKED != 0 {
            return Vec::new();
        }

        let out_blocks = out_features / AWQ_PACKED;
        let expected_qweight = (input.cols / AWQ_PACKED) * out_features;
        if qweight.len() != expected_qweight || zeros.len() != group_count * out_blocks {
            return Vec::new();
        }

        let mut output = vec![0.0f32; input.rows * out_features];
        let mut dequant_block = vec![0.0f32; AWQ_OUT_BLOCK * AWQ_PACKED];
        let mut scale_cache = vec![0.0f32; AWQ_OUT_BLOCK];
        let mut zero_cache = vec![0.0f32; AWQ_OUT_BLOCK];

        for out_start in (0..out_features).step_by(AWQ_OUT_BLOCK) {
            let out_end = (out_start + AWQ_OUT_BLOCK).min(out_features);
            let out_block = out_end - out_start;

            for group in 0..group_count {
                let group_in_start = group * group_size;
                let group_block_start = group_in_start / AWQ_PACKED;
                let blocks_in_group = group_size / AWQ_PACKED;

                for ob in 0..out_block {
                    let out_row = out_start + ob;
                    scale_cache[ob] = scales[group * out_features + out_row].to_f32();
                    let zero_word = zeros[group * out_blocks + out_row / AWQ_PACKED];
                    zero_cache[ob] = unpack_int4(zero_word, out_row % AWQ_PACKED) as f32;
                }

                for blk in 0..blocks_in_group {
                    let in_block = group_block_start + blk;
                    let in_offset = group_in_start + blk * AWQ_PACKED;

                    for ob in 0..out_block {
                        let out_row = out_start + ob;
                        let qword = qweight[in_block * out_features + out_row];
                        let scale = scale_cache[ob];
                        let zero = zero_cache[ob];
                        let base = ob * AWQ_PACKED;

                        dequant_block[base] = (unpack_int4(qword, 0) as f32 - zero) * scale;
                        dequant_block[base + 1] =
                            (unpack_int4(qword, 1) as f32 - zero) * scale;
                        dequant_block[base + 2] =
                            (unpack_int4(qword, 2) as f32 - zero) * scale;
                        dequant_block[base + 3] =
                            (unpack_int4(qword, 3) as f32 - zero) * scale;
                        dequant_block[base + 4] =
                            (unpack_int4(qword, 4) as f32 - zero) * scale;
                        dequant_block[base + 5] =
                            (unpack_int4(qword, 5) as f32 - zero) * scale;
                        dequant_block[base + 6] =
                            (unpack_int4(qword, 6) as f32 - zero) * scale;
                        dequant_block[base + 7] =
                            (unpack_int4(qword, 7) as f32 - zero) * scale;
                    }

                    for row in 0..input.rows {
                        let in_row_start = row * input.cols + in_offset;
                        let in_slice =
                            &input.data[in_row_start..in_row_start + AWQ_PACKED];
                        let out_base = row * out_features + out_start;

                        for ob in 0..out_block {
                            let w = &dequant_block[ob * AWQ_PACKED..(ob + 1) * AWQ_PACKED];
                            output[out_base + ob] += dot8(in_slice, w);
                        }
                    }
                }
            }
        }

        output
    }
}

impl QuantizedBackend for GpuQuantizedBackend {
    fn q4_matmul(input: MatmulInput<'_>, qweight: &[u8], scales: &[f16]) -> Vec<f32> {
        let _ = quantized_dispatcher().backend();
        CpuQuantizedBackend::q4_matmul(input, qweight, scales)
    }

    fn awq_matmul(
        input: MatmulInput<'_>,
        qweight: &[u32],
        scales: &[f16],
        zeros: &[u32],
        group_size: usize,
    ) -> Vec<f32> {
        let _ = quantized_dispatcher().backend();
        CpuQuantizedBackend::awq_matmul(input, qweight, scales, zeros, group_size)
    }
}

#[inline(always)]
fn decode_q4_0_block(qbytes: &[u8], scale: f32, output: &mut [f32]) {
    debug_assert_eq!(qbytes.len(), Q4_PACKED);
    debug_assert_eq!(output.len(), Q4_BLOCK);
    let mut idx = 0;
    for &byte in qbytes {
        let lo = (byte & 0x0F) as i8 - 8;
        let hi = ((byte >> 4) & 0x0F) as i8 - 8;
        output[idx] = lo as f32 * scale;
        output[idx + 1] = hi as f32 * scale;
        idx += 2;
    }
}

#[inline(always)]
fn unpack_int4(packed: u32, idx: usize) -> i8 {
    ((packed >> (idx * 4)) & 0xF) as i8
}

#[inline(always)]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(a.len() >= AWQ_PACKED && b.len() >= AWQ_PACKED);
    a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
        + a[3] * b[3]
        + a[4] * b[4]
        + a[5] * b[5]
        + a[6] * b[6]
        + a[7] * b[7]
}

#[inline(always)]
fn dot32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(a.len() >= Q4_BLOCK && b.len() >= Q4_BLOCK);
    dot8(&a[0..AWQ_PACKED], &b[0..AWQ_PACKED])
        + dot8(&a[8..16], &b[8..16])
        + dot8(&a[16..24], &b[16..24])
        + dot8(&a[24..32], &b[24..32])
}
