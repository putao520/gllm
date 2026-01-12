//! AWQ (Activation-aware Weight Quantization) support.
//!
//! AWQ stores INT4 weights in a packed layout alongside per-group scales and
//! zero-points. This module provides minimal utilities for decoding the packed
//! representation and running dequantize + matmul in pure Rust.

use crate::quantized_ops::{DefaultQuantizedBackend, MatmulInput, QuantizedBackend};
use crate::types::{Error, Result};
use crate::weight_loader::{RawTensor, WeightLoader};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use half::f16;
use safetensors::Dtype;

/// AWQ packed weight representation.
pub struct AwqWeight {
    /// INT4 quantized values (packed: 8 x 4-bit values per u32).
    pub qweight: Vec<u32>,
    /// Per-group scales (f16).
    pub scales: Vec<f16>,
    /// Zero points (packed: 8 x 4-bit values per u32).
    pub zeros: Vec<u32>,
    /// Group size along the input dimension (typically 128).
    pub group_size: usize,
    /// Tensor shape as [out_features, in_features].
    pub shape: [usize; 2],
}

impl AwqWeight {
    /// Load AWQ weights from a safetensors loader using the given prefix.
    pub fn from_safetensors(loader: &WeightLoader, prefix: &str) -> Result<Self> {
        let qweight_raw = loader.load_raw_tensor(&format!("{prefix}.qweight"))?;
        let scales_raw = loader.load_raw_tensor(&format!("{prefix}.scales"))?;
        let zeros_raw = loader.load_raw_tensor(&format!("{prefix}.zeros"))?;

        let qweight = parse_u32_tensor(&qweight_raw, "qweight")?;
        let zeros = parse_u32_tensor(&zeros_raw, "zeros")?;
        let scales = parse_scales(&scales_raw)?;

        let (in_features, out_features) = parse_qweight_shape(&qweight_raw)?;
        let group_size = parse_group_size(&scales_raw, in_features, out_features)?;
        validate_zeros_shape(&zeros_raw, group_size, in_features, out_features)?;

        Ok(Self {
            qweight,
            scales,
            zeros,
            group_size,
            shape: [out_features, in_features],
        })
    }

    /// Unpack a 4-bit value from a packed u32.
    fn unpack_int4(&self, packed: u32, idx: usize) -> i8 {
        ((packed >> (idx * 4)) & 0xF) as i8
    }

    /// Dequantize to a full f32 matrix in [out, in] order.
    pub fn dequantize(&self) -> Vec<f32> {
        let [out_features, in_features] = self.shape;
        let out_blocks = out_features / 8;
        let mut output = vec![0.0f32; out_features * in_features];

        for out in 0..out_features {
            let out_block = out / 8;
            let out_offset = out % 8;
            for inp in 0..in_features {
                let qword = self.qweight[(inp / 8) * out_features + out];
                let qval = self.unpack_int4(qword, inp % 8);
                let group = inp / self.group_size;
                let zword = self.zeros[group * out_blocks + out_block];
                let zero = self.unpack_int4(zword, out_offset);
                let scale = self.scales[group * out_features + out].to_f32();
                output[out * in_features + inp] = (qval - zero) as f32 * scale;
            }
        }

        output
    }

    /// Dequantize and run matmul on the provided device.
    pub fn matmul<B: Backend>(&self, input: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 2> {
        self.matmul_optimized(input, device)
    }

    /// Optimized AWQ matmul using block dequantization.
    pub fn matmul_optimized<B: Backend>(
        &self,
        input: Tensor<B, 2>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let [batch, in_features] = input.dims();
        if in_features != self.shape[1] {
            return self.matmul_naive(input, device);
        }

        let input_data = match input.clone().into_data().into_vec::<f32>() {
            Ok(data) => data,
            Err(_) => return self.matmul_naive(input, device),
        };
        if input_data.len() != batch * in_features {
            return self.matmul_naive(input, device);
        }

        let output_data = DefaultQuantizedBackend::awq_matmul(
            MatmulInput::new(&input_data, batch, in_features),
            &self.qweight,
            &self.scales,
            &self.zeros,
            self.group_size,
        );
        if output_data.len() != batch * self.shape[0] {
            return self.matmul_naive(input, device);
        }

        Tensor::from_data(
            TensorData::new(output_data, [batch, self.shape[0]]),
            device,
        )
    }

    fn matmul_naive<B: Backend>(&self, input: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 2> {
        let weight_data = self.dequantize();
        let [out_features, in_features] = self.shape;
        let weight = Tensor::from_data(
            TensorData::new(weight_data, [out_features, in_features]),
            device,
        )
        .transpose();
        input.matmul(weight)
    }
}

/// Linear layer wrapper that uses AWQ weights.
pub struct AwqLinear<B: Backend> {
    /// Packed AWQ weights.
    pub weight: AwqWeight,
    /// Optional bias vector.
    pub bias: Option<Tensor<B, 1>>,
    /// Device used for dequantization.
    pub device: B::Device,
}

impl<B: Backend> AwqLinear<B> {
    /// Forward pass using AWQ dequantization + matmul.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.weight.matmul(input, &self.device);
        if let Some(bias) = &self.bias {
            output + bias.clone().unsqueeze()
        } else {
            output
        }
    }
}

/// Parse a packed u32 tensor from raw safetensors data.
fn parse_u32_tensor(raw: &RawTensor, label: &str) -> Result<Vec<u32>> {
    match raw.dtype {
        Dtype::I32 | Dtype::U32 => {}
        _ => {
            return Err(Error::LoadError(format!(
                "AWQ {label} must be int32/uint32, got {:?}",
                raw.dtype
            )))
        }
    }
    if raw.data.len() % 4 != 0 {
        return Err(Error::LoadError(format!(
            "AWQ {label} byte length is not divisible by 4"
        )));
    }
    Ok(raw
        .data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Parse AWQ scales as f16 values.
fn parse_scales(raw: &RawTensor) -> Result<Vec<f16>> {
    match raw.dtype {
        Dtype::F16 => {
            if raw.data.len() % 2 != 0 {
                return Err(Error::LoadError(
                    "AWQ scales byte length is not divisible by 2".into(),
                ));
            }
            Ok(raw
                .data
                .chunks_exact(2)
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect())
        }
        Dtype::F32 => {
            if raw.data.len() % 4 != 0 {
                return Err(Error::LoadError(
                    "AWQ scales byte length is not divisible by 4".into(),
                ));
            }
            Ok(raw
                .data
                .chunks_exact(4)
                .map(|chunk| {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f16::from_f32(value)
                })
                .collect())
        }
        _ => Err(Error::LoadError(format!(
            "AWQ scales must be f16/f32, got {:?}",
            raw.dtype
        ))),
    }
}

/// Derive input/output dimensions from qweight shape.
fn parse_qweight_shape(raw: &RawTensor) -> Result<(usize, usize)> {
    if raw.shape.len() != 2 {
        return Err(Error::LoadError(
            "AWQ qweight must be a 2D tensor".into(),
        ));
    }
    let in_packed = raw.shape[0];
    let out_features = raw.shape[1];
    let in_features = in_packed * 8;
    if in_features == 0 || out_features == 0 {
        return Err(Error::LoadError(
            "AWQ qweight shape has zero-sized dimension".into(),
        ));
    }
    Ok((in_features, out_features))
}

/// Compute group size from scales shape and validate compatibility.
fn parse_group_size(raw: &RawTensor, in_features: usize, out_features: usize) -> Result<usize> {
    if raw.shape.len() != 2 {
        return Err(Error::LoadError(
            "AWQ scales must be a 2D tensor".into(),
        ));
    }
    let group_count = raw.shape[0];
    let scale_out = raw.shape[1];
    if scale_out != out_features {
        return Err(Error::LoadError(
            "AWQ scales shape does not match out_features".into(),
        ));
    }
    if group_count == 0 {
        return Err(Error::LoadError(
            "AWQ scales shape has zero group count".into(),
        ));
    }
    if in_features % group_count != 0 {
        return Err(Error::LoadError(
            "AWQ scales shape is incompatible with in_features".into(),
        ));
    }
    Ok(in_features / group_count)
}

/// Validate zeros tensor shape against AWQ expectations.
fn validate_zeros_shape(
    raw: &RawTensor,
    group_size: usize,
    in_features: usize,
    out_features: usize,
) -> Result<()> {
    if raw.shape.len() != 2 {
        return Err(Error::LoadError(
            "AWQ zeros must be a 2D tensor".into(),
        ));
    }
    if out_features % 8 != 0 {
        return Err(Error::LoadError(
            "AWQ out_features must be divisible by 8".into(),
        ));
    }
    let group_count = in_features / group_size;
    if raw.shape[0] != group_count || raw.shape[1] * 8 != out_features {
        return Err(Error::LoadError(
            "AWQ zeros shape does not match expected dimensions".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_awq_unpack_int4() {
        let weight = AwqWeight {
            qweight: vec![],
            scales: vec![],
            zeros: vec![],
            group_size: 1,
            shape: [1, 1],
        };
        let packed = 0xFEDCBA98u32;
        assert_eq!(weight.unpack_int4(packed, 0), 0x8);
        assert_eq!(weight.unpack_int4(packed, 1), 0x9);
        assert_eq!(weight.unpack_int4(packed, 7), 0xF);
    }
}
