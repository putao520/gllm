use std::sync::Arc;

use thiserror::Error;

pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// GGUF v3 is the current production format supported by gllm parser.
/// Earlier/newer versions must be handled explicitly in `GgufReader`.
pub const GGUF_SUPPORTED_VERSION: u32 = 3;
/// GGML K-quantized blocks have fixed width 256 by format definition.
/// Source: SPEC DATA-GGUF-DTYPE table (Q2_K/Q3_K/... block size = 256).
const QK_K: usize = 256;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufValueType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(GgufError::InvalidValueType(other)),
        }
    }
}

#[repr(u32)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
}

impl GgmlDType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
        }
    }

    pub const fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    pub const fn block_size(self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64
            | Self::BF16 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL
            | Self::MXFP4 => 32,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => QK_K,
        }
    }

    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 50,
            Self::IQ4_NL => 18,
            Self::IQ3_S => 110,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 136,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::F64 => 8,
            Self::IQ1_M => 56,
            Self::BF16 => 2,
            Self::TQ1_0 => 54,
            Self::TQ2_0 => 66,
            Self::MXFP4 => 17,
        }
    }

    pub const fn all() -> &'static [Self] {
        &[
            Self::F32,
            Self::F16,
            Self::Q4_0,
            Self::Q4_1,
            Self::Q5_0,
            Self::Q5_1,
            Self::Q8_0,
            Self::Q8_1,
            Self::Q2_K,
            Self::Q3_K,
            Self::Q4_K,
            Self::Q5_K,
            Self::Q6_K,
            Self::Q8_K,
            Self::IQ2_XXS,
            Self::IQ2_XS,
            Self::IQ3_XXS,
            Self::IQ1_S,
            Self::IQ4_NL,
            Self::IQ3_S,
            Self::IQ2_S,
            Self::IQ4_XS,
            Self::I8,
            Self::I16,
            Self::I32,
            Self::I64,
            Self::F64,
            Self::IQ1_M,
            Self::BF16,
            Self::TQ1_0,
            Self::TQ2_0,
            Self::MXFP4,
        ]
    }
}

impl TryFrom<u32> for GgmlDType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            39 => Ok(Self::MXFP4),
            other => Err(GgufError::InvalidDType(other)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(Arc<str>),
    Array(GgufArray),
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Int8(v) => u64::try_from(*v).ok(),
            Self::Uint16(v) => Some(*v as u64),
            Self::Int16(v) => u64::try_from(*v).ok(),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int32(v) => u64::try_from(*v).ok(),
            Self::Uint64(v) => Some(*v),
            Self::Int64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&GgufArray> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufArray {
    pub item_type: GgufValueType,
    pub items: Vec<GgufValue>,
}

impl GgufArray {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("Invalid GGUF magic: 0x{0:08x}")]
    InvalidMagic(u32),

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Invalid GGUF value type: {0}")]
    InvalidValueType(u32),

    #[error("Invalid GGML dtype: {0}")]
    InvalidDType(u32),

    #[error("Missing metadata: {0}")]
    MissingMetadata(String),

    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Tensor out of bounds: {0}")]
    TensorOutOfBounds(String),

    #[error("Unsupported type: {0:?}")]
    UnsupportedType(GgmlDType),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

pub fn tensor_nbytes(dtype: GgmlDType, shape: &[u64]) -> Result<usize, GgufError> {
    if shape.is_empty() {
        return Ok(0);
    }

    let block_size = dtype.block_size();
    let bytes_per_block = dtype.block_bytes();

    // GGUF pads the innermost dimension (ne[0] = shape[0]) to the block boundary,
    // matching llama.cpp's ggml_row_size(type, ne[0]) * ne[1] * ne[2] * ...
    let ne0 = usize::try_from(shape[0])
        .map_err(|_| GgufError::ParseError("tensor dimension overflows usize".to_string()))?;

    let blocks_per_row = ne0.div_ceil(block_size);
    let row_bytes = blocks_per_row
        .checked_mul(bytes_per_block)
        .ok_or_else(|| GgufError::ParseError("row byte size overflow".to_string()))?;

    // Multiply by all outer dimensions (shape[1], shape[2], ...)
    let mut total = row_bytes;
    for &dim in &shape[1..] {
        let dim = usize::try_from(dim)
            .map_err(|_| GgufError::ParseError("tensor dimension overflows usize".to_string()))?;
        total = total
            .checked_mul(dim)
            .ok_or_else(|| GgufError::ParseError("tensor byte size overflow".to_string()))?;
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TEST-GGUF-005: 量化类型识别 — 所有 GgmlDType 变体可从 u32 正确解析
    #[test]
    fn test_gguf_005_quant_type_recognition() {
        let cases: &[(u32, GgmlDType)] = &[
            (0, GgmlDType::F32),
            (1, GgmlDType::F16),
            (2, GgmlDType::Q4_0),
            (3, GgmlDType::Q4_1),
            (6, GgmlDType::Q5_0),
            (7, GgmlDType::Q5_1),
            (8, GgmlDType::Q8_0),
            (9, GgmlDType::Q8_1),
            (10, GgmlDType::Q2_K),
            (11, GgmlDType::Q3_K),
            (12, GgmlDType::Q4_K),
            (13, GgmlDType::Q5_K),
            (14, GgmlDType::Q6_K),
            (15, GgmlDType::Q8_K),
            (16, GgmlDType::IQ2_XXS),
            (17, GgmlDType::IQ2_XS),
            (18, GgmlDType::IQ3_XXS),
            (19, GgmlDType::IQ1_S),
            (20, GgmlDType::IQ4_NL),
            (21, GgmlDType::IQ3_S),
            (22, GgmlDType::IQ2_S),
            (23, GgmlDType::IQ4_XS),
            (24, GgmlDType::I8),
            (25, GgmlDType::I16),
            (26, GgmlDType::I32),
            (27, GgmlDType::I64),
            (28, GgmlDType::F64),
            (29, GgmlDType::IQ1_M),
            (30, GgmlDType::BF16),
            (34, GgmlDType::TQ1_0),
            (35, GgmlDType::TQ2_0),
            (39, GgmlDType::MXFP4),
        ];

        for &(raw, expected) in cases {
            let parsed = GgmlDType::try_from(raw)
                .unwrap_or_else(|_| panic!("GgmlDType::try_from({raw}) failed"));
            assert_eq!(parsed, expected, "dtype mismatch for raw={raw}");
            // as_str() must not panic
            let _ = parsed.as_str();
        }

        // 未知类型必须返回 Err
        assert!(GgmlDType::try_from(4).is_err());
        assert!(GgmlDType::try_from(5).is_err());
        assert!(GgmlDType::try_from(99).is_err());
    }
}
