use std::path::Path;

/// Placeholder for packed quantized bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedBits {
    Int1,
    Int2,
    Int4,
}

/// Extended DType that includes quantized types not in gllm-kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    U8,
    PackedU8(PackedBits),
}

impl DType {
    /// Size in bytes per element.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::U8 => 1,
            Self::PackedU8(_) => 1,
        }
    }
}

use gllm_kernels::quant::QuantType;

use super::gguf::{GgmlDType, GgufError, GgufReader, TensorInfo};

/// Maps GGUF parser's GgmlDType to gllm-kernels' QuantType for kernel dispatch.
/// Returns `None` for native float/integer types (F32, F16, BF16, F64, I8, I16, I32, I64).
pub fn ggml_dtype_to_quant_type(dtype: GgmlDType) -> Option<QuantType> {
    match dtype {
        // K-Quant family
        GgmlDType::Q2_K => Some(QuantType::Q2K),
        GgmlDType::Q3_K => Some(QuantType::Q3K),
        GgmlDType::Q4_K => Some(QuantType::Q4K),
        GgmlDType::Q5_K => Some(QuantType::Q5K),
        GgmlDType::Q6_K => Some(QuantType::Q6K),
        GgmlDType::Q8_K => Some(QuantType::Q8K),
        // Classic GGML family
        GgmlDType::Q4_0 => Some(QuantType::Q4_0),
        GgmlDType::Q4_1 => Some(QuantType::Q4_1),
        GgmlDType::Q5_0 => Some(QuantType::Q5_0),
        GgmlDType::Q5_1 => Some(QuantType::Q5_1),
        GgmlDType::Q8_0 => Some(QuantType::Q8_0),
        GgmlDType::Q8_1 => Some(QuantType::Q8_1),
        // IQ family
        GgmlDType::IQ1_S => Some(QuantType::IQ1S),
        GgmlDType::IQ1_M => Some(QuantType::IQ1M),
        GgmlDType::IQ2_XXS => Some(QuantType::IQ2XXS),
        GgmlDType::IQ2_XS => Some(QuantType::IQ2XS),
        GgmlDType::IQ2_S => Some(QuantType::IQ2S),
        GgmlDType::IQ3_XXS => Some(QuantType::IQ3XXS),
        GgmlDType::IQ3_S => Some(QuantType::IQ3S),
        GgmlDType::IQ4_NL => Some(QuantType::IQ4NL),
        GgmlDType::IQ4_XS => Some(QuantType::IQ4XS),
        // Native float/integer types — not quantized
        GgmlDType::F32
        | GgmlDType::F16
        | GgmlDType::BF16
        | GgmlDType::F64
        | GgmlDType::I8
        | GgmlDType::I16
        | GgmlDType::I32
        | GgmlDType::I64 => None,
        // Exotic types — not yet mapped to kernels
        GgmlDType::TQ1_0 | GgmlDType::TQ2_0 | GgmlDType::MXFP4 => None,
    }
}

#[derive(Debug, Clone)]
pub struct KernelTensorView<'a> {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

#[derive(Debug)]
pub struct GgufAdapter {
    reader: GgufReader,
}

impl GgufAdapter {
    pub fn new(reader: GgufReader) -> Self {
        Self { reader }
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        Ok(Self {
            reader: GgufReader::open(path)?,
        })
    }

    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        self.reader.tensor_info(name)
    }

    pub fn tensor_for_kernel(&self, name: &str) -> Result<KernelTensorView<'_>, GgufError> {
        let info = self.reader.tensor_info(name)?;
        let dtype = map_dtype(info.dtype)?;
        let data = self.reader.tensor_bytes(name)?;

        let mut shape = Vec::with_capacity(info.shape.len());
        for &dim in &info.shape {
            let dim = usize::try_from(dim)
                .map_err(|_| GgufError::ParseError("tensor shape overflows usize".to_string()))?;
            shape.push(dim);
        }

        Ok(KernelTensorView { dtype, shape, data })
    }
}

pub fn map_dtype(dtype: GgmlDType) -> Result<DType, GgufError> {
    let mapped = match dtype {
        GgmlDType::F32 => DType::F32,
        GgmlDType::F16 => DType::F16,
        GgmlDType::BF16 => DType::BF16,

        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q4_K
        | GgmlDType::IQ4_NL
        | GgmlDType::IQ4_XS
        | GgmlDType::MXFP4 => DType::PackedU8(PackedBits::Int4),

        GgmlDType::Q2_K
        | GgmlDType::IQ2_XXS
        | GgmlDType::IQ2_XS
        | GgmlDType::IQ2_S
        | GgmlDType::TQ2_0 => DType::PackedU8(PackedBits::Int2),

        GgmlDType::IQ1_S | GgmlDType::IQ1_M | GgmlDType::TQ1_0 => DType::PackedU8(PackedBits::Int1),

        GgmlDType::Q8_0 | GgmlDType::Q8_1 | GgmlDType::Q8_K | GgmlDType::I8 => DType::U8,

        GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q5_K
        | GgmlDType::Q6_K
        | GgmlDType::Q3_K
        | GgmlDType::IQ3_XXS
        | GgmlDType::IQ3_S
        | GgmlDType::I16
        | GgmlDType::I32
        | GgmlDType::I64
        | GgmlDType::F64 => return Err(GgufError::UnsupportedType(dtype)),
    };

    Ok(mapped)
}
