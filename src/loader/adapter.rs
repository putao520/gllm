use std::path::Path;

use gllm_kernels::{DType, PackedBits};

use super::gguf::{GgmlDType, GgufError, GgufReader, TensorInfo};

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
