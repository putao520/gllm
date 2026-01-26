//! AWQ (Activation-aware Weight Quantization) support.
//!
//! This module focuses on loading AWQ INT4 weights and validating their layout
//! for native quantized matmul kernels.

use crate::parallel_parser::TensorLoader;
use crate::types::{Error, Result};
use crate::weight_loader::RawTensor;
use gllm_kernels::quantized::AwqWeight;
use half::f16;
use safetensors::Dtype;

/// Loaded AWQ weight payload with shape metadata.
#[derive(Clone, Debug)]
pub struct AwqQuantizedWeight {
    pub weight: AwqWeight,
    pub shape: [usize; 2],
}

impl AwqQuantizedWeight {
    /// Load AWQ weights from a tensor loader using the given prefix.
    pub fn from_safetensors<L: TensorLoader>(loader: &L, prefix: &str) -> Result<Self> {
        let qweight_raw = loader.load_raw_tensor(&format!("{prefix}.qweight"))?;
        let scales_raw = loader.load_raw_tensor(&format!("{prefix}.scales"))?;
        let qzeros_raw = load_qzeros(loader, prefix)?;

        let qweight = parse_u32_tensor(&qweight_raw, "qweight")?;
        let qzeros = parse_u32_tensor(&qzeros_raw, "qzeros")?;
        let scales = parse_scales(&scales_raw)?;

        let (out_features, in_features, packed_out) = parse_qweight_shape(&qweight_raw)?;
        let group_size = parse_group_size(&scales_raw, in_features, out_features)?;
        validate_qzeros_shape(&qzeros_raw, packed_out, in_features, out_features)?;

        Ok(Self {
            weight: AwqWeight {
                qweight,
                qzeros,
                scales,
                group_size,
            },
            shape: [out_features, in_features],
        })
    }
}

fn load_qzeros<L: TensorLoader>(loader: &L, prefix: &str) -> Result<RawTensor> {
    let qzeros_name = format!("{prefix}.qzeros");
    if loader.has_tensor(&qzeros_name) {
        return loader.load_raw_tensor(&qzeros_name);
    }
    let zeros_name = format!("{prefix}.zeros");
    if loader.has_tensor(&zeros_name) {
        return loader.load_raw_tensor(&zeros_name);
    }
    Err(Error::LoadError(format!(
        "AWQ {prefix} is missing qzeros/zeros tensor"
    )))
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
fn parse_qweight_shape(raw: &RawTensor) -> Result<(usize, usize, usize)> {
    if raw.shape.len() != 2 {
        return Err(Error::LoadError(
            "AWQ qweight must be a 2D tensor".into(),
        ));
    }
    let packed_out = raw.shape[0];
    let in_features = raw.shape[1];
    if packed_out == 0 || in_features == 0 {
        return Err(Error::LoadError(
            "AWQ qweight shape has zero-sized dimension".into(),
        ));
    }
    Ok((packed_out * 8, in_features, packed_out))
}

/// Compute group size from scales shape and validate compatibility.
fn parse_group_size(raw: &RawTensor, in_features: usize, out_features: usize) -> Result<usize> {
    if raw.shape.len() != 2 {
        return Err(Error::LoadError(
            "AWQ scales must be a 2D tensor".into(),
        ));
    }
    let scale_out = raw.shape[0];
    let group_count = raw.shape[1];
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

/// Validate qzeros tensor shape against AWQ expectations.
fn validate_qzeros_shape(
    raw: &RawTensor,
    packed_out: usize,
    in_features: usize,
    out_features: usize,
) -> Result<()> {
    if raw.shape.len() != 2 {
        return Err(Error::LoadError(
            "AWQ qzeros must be a 2D tensor".into(),
        ));
    }
    if out_features % 8 != 0 {
        return Err(Error::LoadError(
            "AWQ out_features must be divisible by 8".into(),
        ));
    }
    let group_count = raw.shape[1];
    if in_features % group_count != 0 {
        return Err(Error::LoadError(
            "AWQ qzeros shape is incompatible with in_features".into(),
        ));
    }
    if raw.shape[0] != packed_out {
        return Err(Error::LoadError(
            "AWQ qzeros shape does not match packed out_features".into(),
        ));
    }
    Ok(())
}
