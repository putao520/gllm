//! Quantized tensor support for GGML/GGUF data types.
//!
//! This module provides the minimal building blocks for loading GGUF quantized
//! tensors and dequantizing them into f32 for inference. The implementation is
//! intentionally CPU-only and pure Rust to match the gllm design constraints.

use crate::quantized_ops::{DefaultQuantizedBackend, MatmulInput, QuantizedBackend};
use crate::types::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
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

struct Q4Cache {
    qweight: Vec<u8>,
    scales: Vec<f16>,
}

/// Quantized linear layer that dequantizes weights on-the-fly.
pub struct QLinear<B: Backend> {
    /// Quantized weight matrix.
    weight: QTensor,
    /// Cached Q4_0 scales and packed values.
    q4_cache: Option<Q4Cache>,
    /// Optional bias vector.
    bias: Option<Tensor<B, 1>>,
    /// Input feature dimension.
    in_features: usize,
    /// Output feature dimension.
    out_features: usize,
    /// Device to place the dequantized tensor on.
    device: B::Device,
}

fn build_q4_cache(weight: &QTensor, in_features: usize, out_features: usize) -> Option<Q4Cache> {
    if weight.dtype != GgmlDType::Q4_0 || in_features == 0 || out_features == 0 {
        return None;
    }
    if in_features % 32 != 0 {
        return None;
    }
    const BLOCK_BYTES: usize = 18;
    const QBYTES: usize = 16;
    if weight.data.len() % BLOCK_BYTES != 0 {
        return None;
    }
    let blocks_per_row = in_features / 32;
    let expected_blocks = out_features * blocks_per_row;
    let actual_blocks = weight.data.len() / BLOCK_BYTES;
    if expected_blocks != actual_blocks {
        return None;
    }

    let mut scales = Vec::with_capacity(actual_blocks);
    let mut qweight = Vec::with_capacity(actual_blocks * QBYTES);

    for block in weight.data.chunks_exact(BLOCK_BYTES) {
        let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]]));
        scales.push(scale);
        qweight.extend_from_slice(&block[2..]);
    }

    Some(Q4Cache { qweight, scales })
}

impl<B: Backend> QLinear<B> {
    /// Create a quantized linear layer from a GGUF tensor.
    pub fn new(weight: QTensor, bias: Option<Tensor<B, 1>>, device: &B::Device) -> Self {
        let (out_features, in_features) = match weight.shape.as_slice() {
            [out, inn] => (*out, *inn),
            _ => (0, 0),
        };
        let q4_cache = build_q4_cache(&weight, in_features, out_features);
        Self {
            weight,
            q4_cache,
            bias,
            in_features,
            out_features,
            device: device.clone(),
        }
    }

    /// Forward pass using the optimized quantized matmul when available.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_optimized(input)
    }

    /// Optimized forward pass using block dequantization + matmul.
    pub fn forward_optimized(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        if self.weight.dtype != GgmlDType::Q4_0 {
            return self.forward_naive(input);
        }
        let Some(cache) = &self.q4_cache else {
            return self.forward_naive(input);
        };

        let [batch, in_features] = input.dims();
        if in_features != self.in_features {
            return self.forward_naive(input);
        }

        let input_data = match input.clone().into_data().into_vec::<f32>() {
            Ok(data) => data,
            Err(_) => return self.forward_naive(input),
        };
        if input_data.len() != batch * in_features {
            return self.forward_naive(input);
        }

        let output_data = DefaultQuantizedBackend::q4_matmul(
            MatmulInput::new(&input_data, batch, in_features),
            &cache.qweight,
            &cache.scales,
        );
        if output_data.len() != batch * self.out_features {
            return self.forward_naive(input);
        }

        let output = Tensor::from_data(
            TensorData::new(output_data, [batch, self.out_features]),
            &self.device,
        );

        if let Some(bias) = &self.bias {
            output + bias.clone().unsqueeze()
        } else {
            output
        }
    }

    fn forward_naive(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Dequantize weight on every call to keep the storage quantized.
        let weight_data = self.weight.dequantize();
        let weight = Tensor::from_data(
            TensorData::new(weight_data, [self.out_features, self.in_features]),
            &self.device,
        )
        .transpose(); // Burn expects [in, out].

        let output = input.matmul(weight);

        if let Some(bias) = &self.bias {
            output + bias.clone().unsqueeze()
        } else {
            output
        }
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
