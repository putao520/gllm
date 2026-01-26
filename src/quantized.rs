//! Quantized tensor support for GGML/GGUF data types.
//!
//! This module provides the minimal building blocks for loading GGUF quantized
//! tensors and dequantizing them into f32 for inference. The implementation is
//! intentionally CPU-only and pure Rust to match the gllm design constraints.

use crate::types::{Error, Result};
use gllm_kernels::backend::{Backend, BackendImpl};
use gllm_kernels::linear_forward;
use gllm_kernels::quantized::{AwqWeight, Q4_0Block, Q8_0Block};
use half::f16;

/// GGML data types used by GGUF tensors.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgmlDType {
    /// 32-bit floating point.
    F32 = 0,
    /// 16-bit floating point.
    F16 = 1,
    /// 4-bit quantization with a single scale per block.
    Q4_0 = 2,
    /// 4-bit quantization with scale + offset per block.
    Q4_1 = 3,
    /// 5-bit quantization with a single scale per block.
    Q5_0 = 6,
    /// 5-bit quantization with scale + offset per block.
    Q5_1 = 7,
    /// 8-bit quantization with a single scale per block.
    Q8_0 = 8,
    /// 8-bit quantization with scale + offset per block.
    Q8_1 = 9,
    /// 2-bit K-quantization.
    Q2_K = 10,
    /// 3-bit K-quantization (small).
    Q3_K_S = 11,
    /// 3-bit K-quantization (medium).
    Q3_K_M = 12,
    /// 3-bit K-quantization (large).
    Q3_K_L = 13,
    /// 4-bit K-quantization (small).
    Q4_K_S = 14,
    /// 4-bit K-quantization (medium).
    Q4_K_M = 15,
    /// 5-bit K-quantization (small).
    Q5_K_S = 16,
    /// 5-bit K-quantization (medium).
    Q5_K_M = 17,
    /// 6-bit K-quantization.
    Q6_K = 18,
}

impl GgmlDType {
    /// Parse a GGML data type from its raw integer value.
    pub fn from_u32(value: u32) -> Result<Self> {
        let dtype = match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K_S,
            12 => Self::Q3_K_M,
            13 => Self::Q3_K_L,
            14 => Self::Q4_K_S,
            15 => Self::Q4_K_M,
            16 => Self::Q5_K_S,
            17 => Self::Q5_K_M,
            18 => Self::Q6_K,
            _ => {
                return Err(Error::LoadError(format!(
                    "Unsupported GGML dtype value: {value}"
                )))
            }
        };
        Ok(dtype)
    }

    /// Number of elements represented by a single quantized block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1 => 32,
            Self::Q2_K
            | Self::Q3_K_S
            | Self::Q3_K_M
            | Self::Q3_K_L
            | Self::Q4_K_S
            | Self::Q4_K_M
            | Self::Q5_K_S
            | Self::Q5_K_M
            | Self::Q6_K => 256,
        }
    }

    /// Byte size of a single quantized block.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2_K => 84,
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L => 110,
            Self::Q4_K_S | Self::Q4_K_M => 144,
            Self::Q5_K_S | Self::Q5_K_M => 176,
            Self::Q6_K => 210,
        }
    }
}

/// Quantized tensor container for GGUF data.
pub struct QTensor {
    /// Raw packed bytes from the GGUF tensor payload.
    pub data: Vec<u8>,
    /// GGML dtype describing how to interpret the bytes.
    pub dtype: GgmlDType,
    /// Tensor dimensions in row-major order.
    pub shape: Vec<usize>,
}

impl QTensor {
    /// Dequantize into a flat f32 buffer (row-major).
    pub fn dequantize(&self) -> Vec<f32> {
        match self.dtype {
            GgmlDType::F32 => self.dequant_f32(),
            GgmlDType::F16 => self.dequant_f16(),
            GgmlDType::Q4_0 => self.dequant_q4_0(),
            GgmlDType::Q3_K_S | GgmlDType::Q3_K_M | GgmlDType::Q3_K_L => self.dequant_q3_k(),
            GgmlDType::Q4_K_S | GgmlDType::Q4_K_M => self.dequant_q4_k(),
            GgmlDType::Q8_0 => self.dequant_q8_0(),
            _ => unimplemented!("GGML dtype {:?} dequantization is not implemented", self.dtype),
        }
    }

    /// Convenience helper for the total element count implied by shape.
    pub fn element_count(&self) -> usize {
        self.shape.iter().copied().product()
    }

    /// Decode raw f32 bytes.
    fn dequant_f32(&self) -> Vec<f32> {
        self.data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Decode raw f16 bytes into f32 values.
    fn dequant_f16(&self) -> Vec<f32> {
        self.data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f16::from_bits(bits).to_f32()
            })
            .collect()
    }

    /// Q4_0: block_size=32, each block = f16 scale + 16 packed bytes.
    fn dequant_q4_0(&self) -> Vec<f32> {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 2 + 16; // f16 scale + 16 packed bytes
        let num_blocks = self.data.len() / BLOCK_BYTES;
        let mut output = Vec::with_capacity(num_blocks * BLOCK_SIZE);

        for block in self.data.chunks_exact(BLOCK_BYTES) {
            let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for i in 0..16 {
                let byte = block[2 + i];
                let v0 = (byte & 0x0F) as i8 - 8;
                let v1 = ((byte >> 4) & 0x0F) as i8 - 8;
                output.push(v0 as f32 * scale);
                output.push(v1 as f32 * scale);
            }
        }
        output
    }

    /// Q8_0: block_size=32, each block = f16 scale + 32 signed bytes.
    fn dequant_q8_0(&self) -> Vec<f32> {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 2 + 32;
        let num_blocks = self.data.len() / BLOCK_BYTES;
        let mut output = Vec::with_capacity(num_blocks * BLOCK_SIZE);

        for block in self.data.chunks_exact(BLOCK_BYTES) {
            let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for i in 0..32 {
                let v = block[2 + i] as i8;
                output.push(v as f32 * scale);
            }
        }
        output
    }

    /// Q3_K_S/Q3_K_M/Q3_K_L: block_size=256, each block = 110 bytes.
    /// Layout: hmask[32] + qs[64] + scales[12] + d[2] (f16)
    /// 
    /// Each element is 3 bits: 2 bits from qs + 1 bit from hmask.
    /// The scales array encodes 16 scale values for 16 sub-blocks of 16 elements.
    fn dequant_q3_k(&self) -> Vec<f32> {
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 110; // 32 + 64 + 12 + 2
        let num_blocks = self.data.len() / BLOCK_BYTES;
        let mut output = Vec::with_capacity(num_blocks * QK_K);

        for block in self.data.chunks_exact(BLOCK_BYTES) {
            // Block layout: hmask[0..32] + qs[32..96] + scales[96..108] + d[108..110]
            let hmask = &block[0..32];
            let qs = &block[32..96];
            let scales_raw = &block[96..108];
            let d = f16::from_bits(u16::from_le_bytes([block[108], block[109]])).to_f32();

            // Decode 16 scale values from 12 packed bytes
            let mut scales = [0i8; 16];
            for i in 0..8 {
                scales[i] = ((scales_raw[i] & 0x0F) as i8) - 8;
                scales[i + 8] = ((scales_raw[i] >> 4) as i8) - 8;
            }
            // High bits of scales from bytes 8-11
            for i in 0..4 {
                let b = scales_raw[8 + i];
                scales[i] = (scales[i] & 0x0F) | (((b & 0x03) as i8) << 4);
                scales[i + 4] = (scales[i + 4] & 0x0F) | ((((b >> 2) & 0x03) as i8) << 4);
                scales[i + 8] = (scales[i + 8] & 0x0F) | ((((b >> 4) & 0x03) as i8) << 4);
                scales[i + 12] = (scales[i + 12] & 0x0F) | ((((b >> 6) & 0x03) as i8) << 4);
            }

            // Dequantize 256 elements
            let mut m = 1u8;
            let mut is = 0usize;
            let mut qs_idx = 0usize;

            for n in 0..QK_K {
                let l = n % 256;
                let j = l % 128;
                let qb = j / 2;

                // Extract 2-bit value from qs
                let q2 = if j < 64 {
                    (qs[qb] >> ((l % 2) * 4)) & 0x03
                } else {
                    (qs[qb] >> (((l % 2) * 4) + 2)) & 0x03
                };

                // Extract high bit from hmask
                let hbit = if (hmask[j % 32] & m) != 0 { 4u8 } else { 0u8 };

                // Combine to get 3-bit value
                let q3 = (q2 | hbit) as i8 - 4;

                // Apply scale
                let scale_idx = is;
                let value = d * (scales[scale_idx] as f32) * (q3 as f32);
                output.push(value);

                // Update indices
                if (n + 1) % 16 == 0 {
                    is += 1;
                    if is >= 16 {
                        is = 0;
                    }
                }
                if (n + 1) % 32 == 0 {
                    qs_idx += 32;
                }
                if (n + 1) % 128 == 0 {
                    m <<= 1;
                    if m == 0 {
                        m = 1;
                    }
                }
            }
        }
        output
    }


    /// Q4_K_S/Q4_K_M: block_size=256, block layout matches ggml's BlockQ4K.
    fn dequant_q4_k(&self) -> Vec<f32> {
        const QK_K: usize = 256;
        const K_SCALE_SIZE: usize = 12;
        const BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + (QK_K / 2);
        let num_blocks = self.data.len() / BLOCK_BYTES;
        let mut output = Vec::with_capacity(num_blocks * QK_K);

        for block in self.data.chunks_exact(BLOCK_BYTES) {
            let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            let scales = &block[4..4 + K_SCALE_SIZE];
            let qs = &block[4 + K_SCALE_SIZE..];
            let mut scale_index = 0usize;

            for j in (0..QK_K).step_by(64) {
                let q = &qs[j / 2..j / 2 + 32];
                let (sc1, m1) = get_scale_min_k4(scale_index, scales);
                let d1 = d * sc1 as f32;
                let m1 = dmin * m1 as f32;
                let (sc2, m2) = get_scale_min_k4(scale_index + 1, scales);
                let d2 = d * sc2 as f32;
                let m2 = dmin * m2 as f32;

                for &qv in q {
                    output.push(d1 * (qv & 0xF) as f32 - m1);
                }
                for &qv in q {
                    output.push(d2 * (qv >> 4) as f32 - m2);
                }
                scale_index += 2;
            }
        }
        output
    }
}

/// Decode packed scale/min values for Q4_K blocks.
fn get_scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        let d = scales[index] & 63;
        let m = scales[index + 4] & 63;
        (d, m)
    } else {
        let d = (scales[index + 4] & 0xF) | ((scales[index - 4] >> 6) << 4);
        let m = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
        (d, m)
    }
}

/// Supported quantized weight payloads for native matmul kernels.
#[derive(Clone, Debug)]
pub enum QuantizedWeight {
    Q4 {
        q_weight: Vec<u8>,
        scales: Vec<f16>,
        shape: [usize; 2],
    },
    Q8 {
        q_weight: Vec<i8>,
        scales: Vec<f16>,
        shape: [usize; 2],
    },
    Awq {
        weight: AwqWeight,
        shape: [usize; 2],
    },
}

impl QuantizedWeight {
    pub fn from_qtensor(tensor: &QTensor) -> Result<Self> {
        let shape = shape_2d(&tensor.shape)?;
        match tensor.dtype {
            GgmlDType::Q4_0 => {
                let blocks = parse_q4_0_blocks(tensor, shape)?;
                let (q_weight, scales) = pack_q4_blocks(&blocks);
                Ok(Self::Q4 {
                    q_weight,
                    scales,
                    shape,
                })
            }
            GgmlDType::Q8_0 => {
                let blocks = parse_q8_0_blocks(tensor, shape)?;
                let (q_weight, scales) = pack_q8_blocks(&blocks);
                Ok(Self::Q8 {
                    q_weight,
                    scales,
                    shape,
                })
            }
            _ => Err(Error::LoadError(format!(
                "GGML dtype {:?} is not supported for native quantized weights",
                tensor.dtype
            ))),
        }
    }

    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::Q4 { shape, .. } | Self::Q8 { shape, .. } | Self::Awq { shape, .. } => *shape,
        }
    }

    pub fn in_features(&self) -> usize {
        self.shape()[1]
    }

    pub fn out_features(&self) -> usize {
        self.shape()[0]
    }
}

/// Native quantized Linear layer using backend quantized matmul kernels.
#[derive(Clone, Debug)]
pub struct NativeQLinear {
    weight: QuantizedWeight,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl NativeQLinear {
    pub fn new(weight: QuantizedWeight, bias: Option<Vec<f32>>) -> Result<Self> {
        let [out_features, in_features] = weight.shape();
        if let Some(ref bias_vec) = bias {
            if bias_vec.len() != out_features {
                return Err(Error::LoadError(
                    "NativeQLinear bias length does not match out_features".into(),
                ));
            }
        }
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
        backend: &BackendImpl,
    ) -> Result<()> {
        if input.len() != batch * self.in_features {
            return Err(Error::InferenceError(
                "NativeQLinear input length mismatch".into(),
            ));
        }
        if output.len() != batch * self.out_features {
            return Err(Error::InferenceError(
                "NativeQLinear output buffer size mismatch".into(),
            ));
        }

        let mut result = match &self.weight {
            QuantizedWeight::Q4 {
                q_weight,
                scales,
                ..
            } => backend
                .q4_matmul(
                    input,
                    q_weight,
                    scales,
                    batch,
                    self.out_features,
                    self.in_features,
                )
                .map_err(Error::InferenceError)?,
            QuantizedWeight::Q8 {
                q_weight,
                scales,
                ..
            } => backend
                .q8_matmul(
                    input,
                    q_weight,
                    scales,
                    batch,
                    self.out_features,
                    self.in_features,
                )
                .map_err(Error::InferenceError)?,
            QuantizedWeight::Awq { weight, .. } => backend
                .awq_matmul(
                    input,
                    &weight.qweight,
                    &weight.qzeros,
                    &weight.scales,
                    batch,
                    self.out_features,
                    self.in_features,
                    weight.group_size,
                )
                .map_err(Error::InferenceError)?,
        };

        if let Some(ref bias) = self.bias {
            for row in 0..batch {
                let row_offset = row * self.out_features;
                for col in 0..self.out_features {
                    result[row_offset + col] += bias[col];
                }
            }
        }

        output.copy_from_slice(&result);
        Ok(())
    }

    pub fn forward_alloc(
        &self,
        input: &[f32],
        batch: usize,
        backend: &BackendImpl,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; batch * self.out_features];
        self.forward(input, &mut output, batch, backend)?;
        Ok(output)
    }
}

fn shape_2d(shape: &[usize]) -> Result<[usize; 2]> {
    match shape {
        [out, inn] => Ok([*out, *inn]),
        _ => Err(Error::LoadError(
            "Expected 2D tensor shape for quantized weights".into(),
        )),
    }
}

fn parse_q4_0_blocks(tensor: &QTensor, shape: [usize; 2]) -> Result<Vec<Q4_0Block>> {
    const BLOCK_BYTES: usize = 18;
    const BLOCK_SIZE: usize = 32;
    if shape[1] % BLOCK_SIZE != 0 {
        return Err(Error::LoadError(
            "Q4_0 in_features must be a multiple of 32".into(),
        ));
    }
    let blocks_per_row = shape[1] / BLOCK_SIZE;
    let expected_blocks = shape[0]
        .checked_mul(blocks_per_row)
        .ok_or_else(|| Error::LoadError("Q4_0 block count overflow".into()))?;
    if tensor.data.len() != expected_blocks * BLOCK_BYTES {
        return Err(Error::LoadError(
            "Q4_0 data length does not match tensor shape".into(),
        ));
    }
    let mut blocks = Vec::with_capacity(expected_blocks);
    for block in tensor.data.chunks_exact(BLOCK_BYTES) {
        let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]]));
        let mut qs = [0u8; 16];
        qs.copy_from_slice(&block[2..18]);
        blocks.push(Q4_0Block { scale, qs });
    }
    Ok(blocks)
}

fn parse_q8_0_blocks(tensor: &QTensor, shape: [usize; 2]) -> Result<Vec<Q8_0Block>> {
    const BLOCK_BYTES: usize = 34;
    const BLOCK_SIZE: usize = 32;
    if shape[1] % BLOCK_SIZE != 0 {
        return Err(Error::LoadError(
            "Q8_0 in_features must be a multiple of 32".into(),
        ));
    }
    let blocks_per_row = shape[1] / BLOCK_SIZE;
    let expected_blocks = shape[0]
        .checked_mul(blocks_per_row)
        .ok_or_else(|| Error::LoadError("Q8_0 block count overflow".into()))?;
    if tensor.data.len() != expected_blocks * BLOCK_BYTES {
        return Err(Error::LoadError(
            "Q8_0 data length does not match tensor shape".into(),
        ));
    }
    let mut blocks = Vec::with_capacity(expected_blocks);
    for block in tensor.data.chunks_exact(BLOCK_BYTES) {
        let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]]));
        let mut qs = [0i8; 32];
        for (idx, value) in qs.iter_mut().enumerate() {
            *value = block[2 + idx] as i8;
        }
        blocks.push(Q8_0Block { scale, qs });
    }
    Ok(blocks)
}

fn pack_q4_blocks(blocks: &[Q4_0Block]) -> (Vec<u8>, Vec<f16>) {
    let mut q_weight = Vec::with_capacity(blocks.len() * 16);
    let mut scales = Vec::with_capacity(blocks.len());
    for block in blocks {
        scales.push(block.scale);
        q_weight.extend_from_slice(&block.qs);
    }
    (q_weight, scales)
}

fn pack_q8_blocks(blocks: &[Q8_0Block]) -> (Vec<i8>, Vec<f16>) {
    let mut q_weight = Vec::with_capacity(blocks.len() * 32);
    let mut scales = Vec::with_capacity(blocks.len());
    for block in blocks {
        scales.push(block.scale);
        q_weight.extend_from_slice(&block.qs);
    }
    (q_weight, scales)
}

pub struct QLinear {
    /// Quantized weight matrix.
    weight: QTensor,
    /// Cached dequantized weights.
    dequantized_weight: std::sync::OnceLock<Vec<f32>>,
    /// Optional bias vector.
    bias: Option<Vec<f32>>,
    /// Input feature dimension.
    in_features: usize,
    /// Output feature dimension.
    out_features: usize,
}

impl QLinear {
    /// Create a quantized linear layer from a GGUF tensor.
    pub fn new(weight: QTensor, bias: Option<Vec<f32>>) -> Self {
        let (out_features, in_features) = match weight.shape.as_slice() {
            [out, inn] => (*out, *inn),
            _ => (0, 0),
        };
        Self {
            weight,
            dequantized_weight: std::sync::OnceLock::new(),
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass using pre-allocated output buffer.
    pub fn forward(&self, input: &[f32], output: &mut [f32], batch: usize) -> Result<()> {
        if input.len() != batch * self.in_features {
            return Err(Error::InferenceError(
                "QLinear input length mismatch".into(),
            ));
        }
        if output.len() != batch * self.out_features {
            return Err(Error::InferenceError(
                "QLinear output buffer size mismatch".into(),
            ));
        }

        // Use cached dequantized weights
        let weight_data = self.dequantized_weight.get_or_init(|| {
            self.weight.dequantize()
        });

        linear_forward(
            input,
            weight_data,
            self.bias.as_deref(),
            output,
            batch,
            self.in_features,
            self.out_features,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_dequant() {
        let scale = f16::from_f32(0.5);
        let mut data = Vec::new();
        data.extend_from_slice(&scale.to_bits().to_le_bytes());
        data.extend(std::iter::repeat(0u8).take(16));

        let tensor = QTensor {
            data,
            dtype: GgmlDType::Q4_0,
            shape: vec![32],
        };
        let out = tensor.dequantize();
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|v| (*v - -4.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_q8_0_dequant() {
        let scale = f16::from_f32(1.0);
        let mut data = Vec::new();
        data.extend_from_slice(&scale.to_bits().to_le_bytes());
        for i in 0..32u8 {
            data.push(i);
        }

        let tensor = QTensor {
            data,
            dtype: GgmlDType::Q8_0,
            shape: vec![32],
        };
        let out = tensor.dequantize();
        let expected: Vec<f32> = (0..32).map(|v| v as f32).collect();
        assert_eq!(out, expected);
    }
}
