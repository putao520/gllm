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
    AWQ4 = 50,
    GPTQ4 = 51,
    SQUEEZE = 52,
    NVFP4 = 53,
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
            Self::AWQ4 => "AWQ4",
            Self::GPTQ4 => "GPTQ4",
            Self::SQUEEZE => "SQUEEZE",
            Self::NVFP4 => "NVFP4",
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
            // AWQ4/GPTQ4: group_size=128, 72 bytes per group
            Self::AWQ4 | Self::GPTQ4 => 128,
            // SqueezeLLM: 256-element block, 3-bit + per-block scale, 130 bytes
            Self::SQUEEZE => QK_K,
            // NVFP4: 64-element block, 4×UE4M3 sub-block scales + 32 bytes E2M1, 36 bytes
            Self::NVFP4 => 64,
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
            Self::Q8_1 => 36,  // d(f16,2B) + s(f16,2B) + qs[i8;32](32B) = 36
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
            Self::AWQ4 => 72,
            Self::GPTQ4 => 72,
            // SqueezeLLM: 256 elements × 3-bit + 2-byte F16 scale = 96 + 2 + 32(LUT placeholder) = 130 bytes per block
            Self::SQUEEZE => 130,
            // NVFP4: 4 bytes UE4M3 sub-block scales + 32 bytes packed E2M1 = 36 bytes per 64-element block
            Self::NVFP4 => 36,
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
            Self::AWQ4,
            Self::GPTQ4,
            Self::SQUEEZE,
            Self::NVFP4,
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
            50 => Ok(Self::AWQ4),
            51 => Ok(Self::GPTQ4),
            52 => Ok(Self::SQUEEZE),
            53 => Ok(Self::NVFP4),
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
                .unwrap_or_else(|_| panic!("GgmlDType::try_from({raw}) failed")); // LEGAL: 测试代码中的 panic 是合理的
            assert_eq!(parsed, expected, "dtype mismatch for raw={raw}");
            // as_str() must not panic
            let _ = parsed.as_str();
        }

        // 未知类型必须返回 Err
        assert!(GgmlDType::try_from(4).is_err());
        assert!(GgmlDType::try_from(5).is_err());
        assert!(GgmlDType::try_from(99).is_err());
    }

    // ── GgufValueType ────────────────────────────────────────────────────

    #[test]
    fn gguf_value_type_try_from_all() {
        let cases: &[(u32, GgufValueType)] = &[
            (0, GgufValueType::Uint8), (1, GgufValueType::Int8),
            (2, GgufValueType::Uint16), (3, GgufValueType::Int16),
            (4, GgufValueType::Uint32), (5, GgufValueType::Int32),
            (6, GgufValueType::Float32), (7, GgufValueType::Bool),
            (8, GgufValueType::String), (9, GgufValueType::Array),
            (10, GgufValueType::Uint64), (11, GgufValueType::Int64),
            (12, GgufValueType::Float64),
        ];
        for &(raw, expected) in cases {
            assert_eq!(GgufValueType::try_from(raw).unwrap(), expected, "value type mismatch for raw={raw}");
        }
    }

    #[test]
    fn gguf_value_type_try_from_invalid() {
        assert!(GgufValueType::try_from(13).is_err());
        assert!(GgufValueType::try_from(255).is_err());
    }

    // ── GgmlDType methods ────────────────────────────────────────────────

    #[test]
    fn ggml_dtype_is_quantized() {
        assert!(!GgmlDType::F32.is_quantized());
        assert!(!GgmlDType::F16.is_quantized());
        assert!(!GgmlDType::BF16.is_quantized());
        assert!(!GgmlDType::F64.is_quantized());
        assert!(!GgmlDType::I8.is_quantized());
        assert!(!GgmlDType::I16.is_quantized());
        assert!(!GgmlDType::I32.is_quantized());
        assert!(!GgmlDType::I64.is_quantized());

        assert!(GgmlDType::Q4_0.is_quantized());
        assert!(GgmlDType::Q8_0.is_quantized());
        assert!(GgmlDType::Q2_K.is_quantized());
        assert!(GgmlDType::IQ2_XXS.is_quantized());
        assert!(GgmlDType::AWQ4.is_quantized());
        assert!(GgmlDType::GPTQ4.is_quantized());
        assert!(GgmlDType::SQUEEZE.is_quantized());
        assert!(GgmlDType::NVFP4.is_quantized());
        assert!(GgmlDType::MXFP4.is_quantized());
    }

    #[test]
    fn ggml_dtype_block_size_non_quantized() {
        assert_eq!(GgmlDType::F32.block_size(), 1);
        assert_eq!(GgmlDType::F16.block_size(), 1);
        assert_eq!(GgmlDType::BF16.block_size(), 1);
        assert_eq!(GgmlDType::I8.block_size(), 1);
        assert_eq!(GgmlDType::I32.block_size(), 1);
    }

    #[test]
    fn ggml_dtype_block_size_quantized() {
        // Standard block-32 types
        assert_eq!(GgmlDType::Q4_0.block_size(), 32);
        assert_eq!(GgmlDType::Q4_1.block_size(), 32);
        assert_eq!(GgmlDType::Q5_0.block_size(), 32);
        assert_eq!(GgmlDType::Q5_1.block_size(), 32);
        assert_eq!(GgmlDType::Q8_0.block_size(), 32);
        assert_eq!(GgmlDType::Q8_1.block_size(), 32);
        assert_eq!(GgmlDType::IQ4_NL.block_size(), 32);
        assert_eq!(GgmlDType::MXFP4.block_size(), 32);

        // K-quant block-256
        assert_eq!(GgmlDType::Q2_K.block_size(), 256);
        assert_eq!(GgmlDType::Q4_K.block_size(), 256);
        assert_eq!(GgmlDType::Q8_K.block_size(), 256);
        assert_eq!(GgmlDType::IQ1_S.block_size(), 256);

        // Modern formats
        assert_eq!(GgmlDType::AWQ4.block_size(), 128);
        assert_eq!(GgmlDType::GPTQ4.block_size(), 128);
        assert_eq!(GgmlDType::NVFP4.block_size(), 64);
        assert_eq!(GgmlDType::SQUEEZE.block_size(), 256);
    }

    #[test]
    fn ggml_dtype_block_bytes_consistency() {
        // block_bytes must be > 0 for all types
        for &dtype in GgmlDType::all() {
            assert!(dtype.block_bytes() > 0, "{dtype:?} has block_bytes=0");
        }
    }

    #[test]
    fn ggml_dtype_block_bytes_specific() {
        assert_eq!(GgmlDType::F32.block_bytes(), 4);
        assert_eq!(GgmlDType::F16.block_bytes(), 2);
        assert_eq!(GgmlDType::BF16.block_bytes(), 2);
        assert_eq!(GgmlDType::Q4_0.block_bytes(), 18);
        assert_eq!(GgmlDType::Q4_1.block_bytes(), 20);
        assert_eq!(GgmlDType::Q8_0.block_bytes(), 34);
        assert_eq!(GgmlDType::Q2_K.block_bytes(), 84);
        assert_eq!(GgmlDType::Q4_K.block_bytes(), 144);
        assert_eq!(GgmlDType::Q8_K.block_bytes(), 292);
        assert_eq!(GgmlDType::AWQ4.block_bytes(), 72);
        assert_eq!(GgmlDType::GPTQ4.block_bytes(), 72);
        assert_eq!(GgmlDType::NVFP4.block_bytes(), 36);
        assert_eq!(GgmlDType::SQUEEZE.block_bytes(), 130);
        assert_eq!(GgmlDType::MXFP4.block_bytes(), 17);
    }

    #[test]
    fn ggml_dtype_all_list_completeness() {
        let all = GgmlDType::all();
        // Must contain all major format families
        assert!(all.contains(&GgmlDType::F32));
        assert!(all.contains(&GgmlDType::F16));
        assert!(all.contains(&GgmlDType::BF16));
        assert!(all.contains(&GgmlDType::Q4_0));
        assert!(all.contains(&GgmlDType::Q8_0));
        assert!(all.contains(&GgmlDType::Q2_K));
        assert!(all.contains(&GgmlDType::IQ2_XXS));
        assert!(all.contains(&GgmlDType::AWQ4));
        assert!(all.contains(&GgmlDType::GPTQ4));
        assert!(all.contains(&GgmlDType::SQUEEZE));
        assert!(all.contains(&GgmlDType::NVFP4));
        assert!(all.contains(&GgmlDType::MXFP4));
        // No duplicates
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "duplicate in all(): {:?} at [{i}] and [{j}]", all[i]);
            }
        }
    }

    #[test]
    fn ggml_dtype_as_str_all() {
        for &dtype in GgmlDType::all() {
            let s = dtype.as_str();
            assert!(!s.is_empty(), "{dtype:?} has empty as_str()");
        }
    }

    // ── GgmlDType TryFrom extended ───────────────────────────────────────

    #[test]
    fn ggml_dtype_try_from_awq_gptq_squeeze_nvfp4() {
        assert_eq!(GgmlDType::try_from(50).unwrap(), GgmlDType::AWQ4);
        assert_eq!(GgmlDType::try_from(51).unwrap(), GgmlDType::GPTQ4);
        assert_eq!(GgmlDType::try_from(52).unwrap(), GgmlDType::SQUEEZE);
        assert_eq!(GgmlDType::try_from(53).unwrap(), GgmlDType::NVFP4);
        assert_eq!(GgmlDType::try_from(39).unwrap(), GgmlDType::MXFP4);
        assert_eq!(GgmlDType::try_from(30).unwrap(), GgmlDType::BF16);
    }

    // ── GgufValue accessors ──────────────────────────────────────────────

    #[test]
    fn gguf_value_as_u64_integer_types() {
        assert_eq!(GgufValue::Uint8(42).as_u64(), Some(42));
        assert_eq!(GgufValue::Int8(-1).as_u64(), None);
        assert_eq!(GgufValue::Uint16(1000).as_u64(), Some(1000));
        assert_eq!(GgufValue::Int16(100).as_u64(), Some(100));
        assert_eq!(GgufValue::Uint32(100_000).as_u64(), Some(100_000));
        assert_eq!(GgufValue::Int32(-5).as_u64(), None);
        assert_eq!(GgufValue::Uint64(u64::MAX).as_u64(), Some(u64::MAX));
        assert_eq!(GgufValue::Int64(12345).as_u64(), Some(12345));
    }

    #[test]
    fn gguf_value_as_u64_non_integer_returns_none() {
        assert_eq!(GgufValue::Float32(1.0).as_u64(), None);
        assert_eq!(GgufValue::Float64(2.0).as_u64(), None);
        assert_eq!(GgufValue::Bool(true).as_u64(), None);
        assert_eq!(GgufValue::String("hello".into()).as_u64(), None);
    }

    #[test]
    fn gguf_value_as_f32() {
        assert_eq!(GgufValue::Float32(3.14).as_f32(), Some(3.14f32));
        assert_eq!(GgufValue::Float64(2.718).as_f32(), Some(2.718f64 as f32));
        assert_eq!(GgufValue::Uint32(1).as_f32(), None);
    }

    #[test]
    fn gguf_value_as_bool() {
        assert_eq!(GgufValue::Bool(true).as_bool(), Some(true));
        assert_eq!(GgufValue::Bool(false).as_bool(), Some(false));
        assert_eq!(GgufValue::Uint8(1).as_bool(), None);
    }

    #[test]
    fn gguf_value_as_str() {
        let val = GgufValue::String("llama".into());
        assert_eq!(val.as_str(), Some("llama"));
        assert_eq!(GgufValue::Uint32(0).as_str(), None);
    }

    #[test]
    fn gguf_value_as_array() {
        let arr = GgufArray { item_type: GgufValueType::Uint32, items: vec![] };
        assert!(GgufValue::Array(arr).as_array().is_some());
        assert!(GgufValue::Uint32(0).as_array().is_none());
    }

    // ── GgufArray ────────────────────────────────────────────────────────

    #[test]
    fn gguf_array_len_and_empty() {
        let empty = GgufArray { item_type: GgufValueType::Uint32, items: vec![] };
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let arr = GgufArray {
            item_type: GgufValueType::Float32,
            items: vec![GgufValue::Float32(1.0), GgufValue::Float32(2.0)],
        };
        assert!(!arr.is_empty());
        assert_eq!(arr.len(), 2);
    }

    // ── tensor_nbytes ────────────────────────────────────────────────────

    #[test]
    fn tensor_nbytes_f32() {
        assert_eq!(tensor_nbytes(GgmlDType::F32, &[4]).unwrap(), 16);
        assert_eq!(tensor_nbytes(GgmlDType::F32, &[2, 3]).unwrap(), 24);
    }

    #[test]
    fn tensor_nbytes_f16() {
        assert_eq!(tensor_nbytes(GgmlDType::F16, &[8]).unwrap(), 16);
    }

    #[test]
    fn tensor_nbytes_q4_0() {
        // block_size=32, block_bytes=18, shape=[32] → 1 block × 18 = 18
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[32]).unwrap(), 18);
        // shape=[64] → 2 blocks × 18 = 36
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[64]).unwrap(), 36);
    }

    #[test]
    fn tensor_nbytes_q4_0_padding() {
        // shape=[33] → ceil(33/32)=2 blocks × 18 = 36 (padded)
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[33]).unwrap(), 36);
    }

    #[test]
    fn tensor_nbytes_empty_shape() {
        assert_eq!(tensor_nbytes(GgmlDType::F32, &[]).unwrap(), 0);
    }

    #[test]
    fn tensor_nbytes_multidim() {
        // Q4_0, shape=[64, 2] → 2 blocks × 18 × 2 = 72
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[64, 2]).unwrap(), 72);
    }

    #[test]
    fn tensor_nbytes_awq4() {
        // block_size=128, block_bytes=72, shape=[128] → 72
        assert_eq!(tensor_nbytes(GgmlDType::AWQ4, &[128]).unwrap(), 72);
    }

    #[test]
    fn tensor_nbytes_nvfp4() {
        // block_size=64, block_bytes=36, shape=[64] → 36
        assert_eq!(tensor_nbytes(GgmlDType::NVFP4, &[64]).unwrap(), 36);
    }

    // ── GgufError display ────────────────────────────────────────────────

    #[test]
    fn gguf_error_display_variants() {
        let e = GgufError::InvalidMagic(0x1234);
        assert!(e.to_string().contains("0x00001234"));

        let e = GgufError::UnsupportedVersion(99);
        assert!(e.to_string().contains("99"));

        let e = GgufError::InvalidValueType(13);
        assert!(e.to_string().contains("13"));

        let e = GgufError::InvalidDType(77);
        assert!(e.to_string().contains("77"));

        let e = GgufError::MissingMetadata("key".into());
        assert!(e.to_string().contains("key"));

        let e = GgufError::TensorNotFound("weight".into());
        assert!(e.to_string().contains("weight"));
    }

    #[test]
    fn gguf_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof");
        let gguf_err: GgufError = io_err.into();
        assert!(matches!(gguf_err, GgufError::Io(_)));
    }

    #[test]
    fn gguf_error_from_utf8() {
        let utf8_err = std::str::from_utf8(b"\xff\xfe").unwrap_err();
        let gguf_err: GgufError = utf8_err.into();
        assert!(matches!(gguf_err, GgufError::Utf8(_)));
    }

    // ── Trait derivations: GgufValueType ─────────────────────────────────

    #[test]
    fn gguf_value_type_traits_clone_copy_eq_hash() {
        let a = GgufValueType::Float32;
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b); // PartialEq
        assert_eq!(a, c);

        // Hash: insert into HashSet
        let mut set = std::collections::HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
        assert!(set.contains(&c));
    }

    #[test]
    fn gguf_value_type_debug_format() {
        let v = GgufValueType::String;
        let debug = format!("{v:?}");
        assert!(debug.contains("String"), "Debug should contain variant name");
    }

    // ── Trait derivations: GgmlDType ─────────────────────────────────────

    #[test]
    fn ggml_dtype_traits_clone_copy_eq_hash() {
        let a = GgmlDType::Q4_0;
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);

        let mut set = std::collections::HashSet::new();
        set.insert(GgmlDType::F32);
        set.insert(GgmlDType::F16);
        set.insert(GgmlDType::F32); // duplicate, should not increase size
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn ggml_dtype_debug_format() {
        let d = GgmlDType::IQ3_XXS;
        let debug = format!("{d:?}");
        assert!(debug.contains("IQ3_XXS"), "Debug should contain variant name");
    }

    // ── GgmlDType discriminant values ────────────────────────────────────

    #[test]
    fn ggml_dtype_discriminant_values() {
        assert_eq!(GgmlDType::F32 as u32, 0);
        assert_eq!(GgmlDType::F16 as u32, 1);
        assert_eq!(GgmlDType::Q4_0 as u32, 2);
        assert_eq!(GgmlDType::Q4_1 as u32, 3);
        assert_eq!(GgmlDType::Q5_0 as u32, 6);
        assert_eq!(GgmlDType::Q5_1 as u32, 7);
        assert_eq!(GgmlDType::Q8_0 as u32, 8);
        assert_eq!(GgmlDType::Q8_1 as u32, 9);
        assert_eq!(GgmlDType::Q2_K as u32, 10);
        assert_eq!(GgmlDType::Q3_K as u32, 11);
        assert_eq!(GgmlDType::Q4_K as u32, 12);
        assert_eq!(GgmlDType::Q5_K as u32, 13);
        assert_eq!(GgmlDType::Q6_K as u32, 14);
        assert_eq!(GgmlDType::Q8_K as u32, 15);
        assert_eq!(GgmlDType::BF16 as u32, 30);
        assert_eq!(GgmlDType::TQ1_0 as u32, 34);
        assert_eq!(GgmlDType::TQ2_0 as u32, 35);
        assert_eq!(GgmlDType::MXFP4 as u32, 39);
        assert_eq!(GgmlDType::AWQ4 as u32, 50);
        assert_eq!(GgmlDType::GPTQ4 as u32, 51);
        assert_eq!(GgmlDType::SQUEEZE as u32, 52);
        assert_eq!(GgmlDType::NVFP4 as u32, 53);
    }

    // ── GgmlDType as_str correctness per variant ─────────────────────────

    #[test]
    fn ggml_dtype_as_str_correctness() {
        assert_eq!(GgmlDType::F32.as_str(), "F32");
        assert_eq!(GgmlDType::F16.as_str(), "F16");
        assert_eq!(GgmlDType::BF16.as_str(), "BF16");
        assert_eq!(GgmlDType::Q4_0.as_str(), "Q4_0");
        assert_eq!(GgmlDType::Q4_1.as_str(), "Q4_1");
        assert_eq!(GgmlDType::Q5_0.as_str(), "Q5_0");
        assert_eq!(GgmlDType::Q5_1.as_str(), "Q5_1");
        assert_eq!(GgmlDType::Q8_0.as_str(), "Q8_0");
        assert_eq!(GgmlDType::Q8_1.as_str(), "Q8_1");
        assert_eq!(GgmlDType::Q2_K.as_str(), "Q2_K");
        assert_eq!(GgmlDType::Q3_K.as_str(), "Q3_K");
        assert_eq!(GgmlDType::Q4_K.as_str(), "Q4_K");
        assert_eq!(GgmlDType::Q5_K.as_str(), "Q5_K");
        assert_eq!(GgmlDType::Q6_K.as_str(), "Q6_K");
        assert_eq!(GgmlDType::Q8_K.as_str(), "Q8_K");
        assert_eq!(GgmlDType::IQ2_XXS.as_str(), "IQ2_XXS");
        assert_eq!(GgmlDType::IQ2_XS.as_str(), "IQ2_XS");
        assert_eq!(GgmlDType::IQ3_XXS.as_str(), "IQ3_XXS");
        assert_eq!(GgmlDType::IQ1_S.as_str(), "IQ1_S");
        assert_eq!(GgmlDType::IQ4_NL.as_str(), "IQ4_NL");
        assert_eq!(GgmlDType::IQ3_S.as_str(), "IQ3_S");
        assert_eq!(GgmlDType::IQ2_S.as_str(), "IQ2_S");
        assert_eq!(GgmlDType::IQ4_XS.as_str(), "IQ4_XS");
        assert_eq!(GgmlDType::I8.as_str(), "I8");
        assert_eq!(GgmlDType::I16.as_str(), "I16");
        assert_eq!(GgmlDType::I32.as_str(), "I32");
        assert_eq!(GgmlDType::I64.as_str(), "I64");
        assert_eq!(GgmlDType::F64.as_str(), "F64");
        assert_eq!(GgmlDType::IQ1_M.as_str(), "IQ1_M");
        assert_eq!(GgmlDType::TQ1_0.as_str(), "TQ1_0");
        assert_eq!(GgmlDType::TQ2_0.as_str(), "TQ2_0");
        assert_eq!(GgmlDType::MXFP4.as_str(), "MXFP4");
        assert_eq!(GgmlDType::AWQ4.as_str(), "AWQ4");
        assert_eq!(GgmlDType::GPTQ4.as_str(), "GPTQ4");
        assert_eq!(GgmlDType::SQUEEZE.as_str(), "SQUEEZE");
        assert_eq!(GgmlDType::NVFP4.as_str(), "NVFP4");
    }

    // ── GgmlDType::all() length matches enum variant count ───────────────

    #[test]
    fn ggml_dtype_all_count_matches_variants() {
        // GgmlDType has 36 variants (F32..NVFP4). all() must enumerate all of them.
        assert_eq!(GgmlDType::all().len(), 36, "all() must list every GgmlDType variant");
    }

    // ── GgmlDType is_quantized for TQ/MX types ───────────────────────────

    #[test]
    fn ggml_dtype_is_quantized_tq_mx_types() {
        assert!(GgmlDType::TQ1_0.is_quantized());
        assert!(GgmlDType::TQ2_0.is_quantized());
        assert!(GgmlDType::MXFP4.is_quantized());
    }

    // ── GgmlDType block_size for specific remaining types ────────────────

    #[test]
    fn ggml_dtype_block_size_i64_f64() {
        assert_eq!(GgmlDType::I64.block_size(), 1);
        assert_eq!(GgmlDType::F64.block_size(), 1);
    }

    #[test]
    fn ggml_dtype_block_size_tq_types() {
        assert_eq!(GgmlDType::TQ1_0.block_size(), 256);
        assert_eq!(GgmlDType::TQ2_0.block_size(), 256);
    }

    // ── GgmlDType block_bytes for remaining types ────────────────────────

    #[test]
    fn ggml_dtype_block_bytes_integer_types() {
        assert_eq!(GgmlDType::I8.block_bytes(), 1);
        assert_eq!(GgmlDType::I16.block_bytes(), 2);
        assert_eq!(GgmlDType::I32.block_bytes(), 4);
        assert_eq!(GgmlDType::I64.block_bytes(), 8);
        assert_eq!(GgmlDType::F64.block_bytes(), 8);
    }

    #[test]
    fn ggml_dtype_block_bytes_k_quant_family() {
        assert_eq!(GgmlDType::Q3_K.block_bytes(), 110);
        assert_eq!(GgmlDType::Q5_K.block_bytes(), 176);
        assert_eq!(GgmlDType::Q6_K.block_bytes(), 210);
    }

    #[test]
    fn ggml_dtype_block_bytes_iq_family() {
        assert_eq!(GgmlDType::IQ2_XXS.block_bytes(), 66);
        assert_eq!(GgmlDType::IQ2_XS.block_bytes(), 74);
        assert_eq!(GgmlDType::IQ3_XXS.block_bytes(), 98);
        assert_eq!(GgmlDType::IQ1_S.block_bytes(), 50);
        assert_eq!(GgmlDType::IQ3_S.block_bytes(), 110);
        assert_eq!(GgmlDType::IQ2_S.block_bytes(), 82);
        assert_eq!(GgmlDType::IQ4_XS.block_bytes(), 136);
        assert_eq!(GgmlDType::IQ1_M.block_bytes(), 56);
    }

    #[test]
    fn ggml_dtype_block_bytes_tq_mxfp4() {
        assert_eq!(GgmlDType::TQ1_0.block_bytes(), 54);
        assert_eq!(GgmlDType::TQ2_0.block_bytes(), 66);
        assert_eq!(GgmlDType::MXFP4.block_bytes(), 17);
    }

    // ── GgufValue Debug and Clone ────────────────────────────────────────

    #[test]
    fn gguf_value_debug_and_clone() {
        let v = GgufValue::Float32(1.5);
        let debug = format!("{v:?}");
        assert!(debug.contains("Float32"), "Debug should contain variant name");

        let cloned = v.clone();
        assert_eq!(v.as_f32(), cloned.as_f32());

        let s = GgufValue::String("test".into());
        let s_cloned = s.clone();
        assert_eq!(s.as_str(), s_cloned.as_str());
    }

    // ── GgufValue as_u64 edge cases ──────────────────────────────────────

    #[test]
    fn gguf_value_as_u64_boundary_values() {
        // Int8 positive boundary
        assert_eq!(GgufValue::Int8(i8::MAX).as_u64(), Some(i8::MAX as u64));
        // Int8 negative → None
        assert_eq!(GgufValue::Int8(-1).as_u64(), None);
        // Int16 positive
        assert_eq!(GgufValue::Int16(i16::MAX).as_u64(), Some(i16::MAX as u64));
        // Int32 positive boundary
        assert_eq!(GgufValue::Int32(i32::MAX).as_u64(), Some(i32::MAX as u64));
        // Int64 positive
        assert_eq!(GgufValue::Int64(i64::MAX).as_u64(), Some(i64::MAX as u64));
        // Int64 negative → None
        assert_eq!(GgufValue::Int64(-1).as_u64(), None);
        // Uint8 max
        assert_eq!(GgufValue::Uint8(u8::MAX).as_u64(), Some(u8::MAX as u64));
        // Uint16 max
        assert_eq!(GgufValue::Uint16(u16::MAX).as_u64(), Some(u16::MAX as u64));
        // Uint32 max
        assert_eq!(GgufValue::Uint32(u32::MAX).as_u64(), Some(u32::MAX as u64));
    }

    // ── GgufValue as_f32 non-float returns None ──────────────────────────

    #[test]
    fn gguf_value_as_f32_non_float_returns_none() {
        assert_eq!(GgufValue::Uint8(1).as_f32(), None);
        assert_eq!(GgufValue::Int32(0).as_f32(), None);
        assert_eq!(GgufValue::Bool(true).as_f32(), None);
        assert_eq!(GgufValue::String("1.0".into()).as_f32(), None);
        let arr = GgufArray { item_type: GgufValueType::Float32, items: vec![] };
        assert_eq!(GgufValue::Array(arr).as_f32(), None);
    }

    // ── GgufArray with various item types ────────────────────────────────

    #[test]
    fn gguf_array_mixed_item_types() {
        let arr = GgufArray {
            item_type: GgufValueType::Uint32,
            items: vec![
                GgufValue::Uint32(10),
                GgufValue::Uint32(20),
                GgufValue::Uint32(30),
            ],
        };
        assert_eq!(arr.len(), 3);
        assert!(!arr.is_empty());
        assert_eq!(arr.item_type, GgufValueType::Uint32);
    }

    #[test]
    fn gguf_array_debug_format() {
        let arr = GgufArray {
            item_type: GgufValueType::String,
            items: vec![GgufValue::String("hello".into())],
        };
        let debug = format!("{arr:?}");
        assert!(debug.contains("item_type") || debug.contains("GgufArray"));
    }

    // ── tensor_nbytes for more types ─────────────────────────────────────

    #[test]
    fn tensor_nbytes_bf16() {
        assert_eq!(tensor_nbytes(GgmlDType::BF16, &[16]).unwrap(), 32);
        assert_eq!(tensor_nbytes(GgmlDType::BF16, &[3, 4]).unwrap(), 24);
    }

    #[test]
    fn tensor_nbytes_i8() {
        assert_eq!(tensor_nbytes(GgmlDType::I8, &[100]).unwrap(), 100);
        assert_eq!(tensor_nbytes(GgmlDType::I8, &[10, 5]).unwrap(), 50);
    }

    #[test]
    fn tensor_nbytes_i32() {
        assert_eq!(tensor_nbytes(GgmlDType::I32, &[8]).unwrap(), 32);
    }

    #[test]
    fn tensor_nbytes_i64() {
        assert_eq!(tensor_nbytes(GgmlDType::I64, &[4]).unwrap(), 32);
        assert_eq!(tensor_nbytes(GgmlDType::F64, &[4]).unwrap(), 32);
    }

    #[test]
    fn tensor_nbytes_q2_k() {
        // block_size=256, block_bytes=84
        // shape=[256] → 1 * 84 = 84
        assert_eq!(tensor_nbytes(GgmlDType::Q2_K, &[256]).unwrap(), 84);
        // shape=[512] → 2 * 84 = 168
        assert_eq!(tensor_nbytes(GgmlDType::Q2_K, &[512]).unwrap(), 168);
        // shape=[257] → ceil(257/256)=2 * 84 = 168 (padded)
        assert_eq!(tensor_nbytes(GgmlDType::Q2_K, &[257]).unwrap(), 168);
    }

    #[test]
    fn tensor_nbytes_q8_k() {
        // block_size=256, block_bytes=292
        assert_eq!(tensor_nbytes(GgmlDType::Q8_K, &[256]).unwrap(), 292);
        assert_eq!(tensor_nbytes(GgmlDType::Q8_K, &[256, 3]).unwrap(), 876);
    }

    #[test]
    fn tensor_nbytes_squeeze() {
        // block_size=256, block_bytes=130
        assert_eq!(tensor_nbytes(GgmlDType::SQUEEZE, &[256]).unwrap(), 130);
        // shape=[512] → 2 * 130 = 260
        assert_eq!(tensor_nbytes(GgmlDType::SQUEEZE, &[512]).unwrap(), 260);
    }

    #[test]
    fn tensor_nbytes_gptq4() {
        // block_size=128, block_bytes=72
        assert_eq!(tensor_nbytes(GgmlDType::GPTQ4, &[128]).unwrap(), 72);
        // shape=[256] → 2 * 72 = 144
        assert_eq!(tensor_nbytes(GgmlDType::GPTQ4, &[256]).unwrap(), 144);
    }

    #[test]
    fn tensor_nbytes_mxfp4() {
        // block_size=32, block_bytes=17
        assert_eq!(tensor_nbytes(GgmlDType::MXFP4, &[32]).unwrap(), 17);
        // shape=[64] → 2 * 17 = 34
        assert_eq!(tensor_nbytes(GgmlDType::MXFP4, &[64]).unwrap(), 34);
    }

    #[test]
    fn tensor_nbytes_tq1_0() {
        // block_size=256, block_bytes=54
        assert_eq!(tensor_nbytes(GgmlDType::TQ1_0, &[256]).unwrap(), 54);
        assert_eq!(tensor_nbytes(GgmlDType::TQ1_0, &[512, 2]).unwrap(), 216);
    }

    #[test]
    fn tensor_nbytes_tq2_0() {
        // block_size=256, block_bytes=66
        assert_eq!(tensor_nbytes(GgmlDType::TQ2_0, &[256]).unwrap(), 66);
    }

    #[test]
    fn tensor_nbytes_shape_with_zero_outer_dim() {
        // shape=[32, 0] → row_bytes=18 * 0 = 0
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[32, 0]).unwrap(), 0);
    }

    #[test]
    fn tensor_nbytes_quantized_padding_non_aligned_inner_dim() {
        // Q4_0 block_size=32: shape=[1] → ceil(1/32)=1 block * 18 = 18
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[1]).unwrap(), 18);
        // shape=[31] → ceil(31/32)=1 block * 18 = 18
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[31]).unwrap(), 18);
        // shape=[33] → ceil(33/32)=2 blocks * 18 = 36
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[33]).unwrap(), 36);
    }

    #[test]
    fn tensor_nbytes_3d_tensor() {
        // F32, shape=[2, 3, 4] → 2*4 * 3 * 4 = 96
        assert_eq!(tensor_nbytes(GgmlDType::F32, &[2, 3, 4]).unwrap(), 96);
        // Q4_0, shape=[64, 2, 3] → (2 blocks * 18) * 2 * 3 = 216
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[64, 2, 3]).unwrap(), 216);
    }

    // ── GgufError remaining display variants ─────────────────────────────

    #[test]
    fn gguf_error_display_remaining_variants() {
        let e = GgufError::InvalidMetadata("bad value".into());
        assert!(e.to_string().contains("bad value"));

        let e = GgufError::TensorOutOfBounds("offset 999".into());
        assert!(e.to_string().contains("offset 999"));

        let e = GgufError::UnsupportedType(GgmlDType::Q4_0);
        assert!(e.to_string().contains("Q4_0"));

        let e = GgufError::ParseError("overflow".into());
        assert!(e.to_string().contains("overflow"));
    }

    #[test]
    fn gguf_error_source_chain() {
        use std::error::Error;

        // Io error source
        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof");
        let gguf_err = GgufError::Io(io_err);
        assert!(gguf_err.source().is_some());

        // Utf8 error source
        let utf8_err = std::str::from_utf8(b"\xff").unwrap_err();
        let gguf_err = GgufError::Utf8(utf8_err);
        assert!(gguf_err.source().is_some());

        // Non-source variants return None
        let gguf_err = GgufError::MissingMetadata("key".into());
        assert!(gguf_err.source().is_none());
    }

    // ── GgmlDType roundtrip: discriminant → TryFrom ─────────────────────

    #[test]
    fn ggml_dtype_roundtrip_all_variants() {
        for &dtype in GgmlDType::all() {
            let disc = dtype as u32;
            let recovered = GgmlDType::try_from(disc)
                .unwrap_or_else(|_| panic!("roundtrip failed for {dtype:?} (disc={disc})"));
            assert_eq!(dtype, recovered, "roundtrip mismatch for {dtype:?}");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NEW TESTS — 47 additional tests below this line
    // ═══════════════════════════════════════════════════════════════════════

    // ── Constants ────────────────────────────────────────────────────────

    #[test]
    fn gguf_magic_value_is_gguf_ascii() {
        // GGUF magic 0x46554747 = "GGUF" in little-endian
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
        let bytes = GGUF_MAGIC.to_le_bytes();
        assert_eq!(&bytes, b"GGUF");
    }

    #[test]
    fn gguf_supported_version_is_3() {
        assert_eq!(GGUF_SUPPORTED_VERSION, 3);
    }

    #[test]
    fn qk_k_is_256() {
        // QK_K is the GGML K-quant block width
        assert_eq!(QK_K, 256);
    }

    // ── GgufValueType discriminant values ────────────────────────────────

    #[test]
    fn gguf_value_type_discriminant_values() {
        assert_eq!(GgufValueType::Uint8 as u32, 0);
        assert_eq!(GgufValueType::Int8 as u32, 1);
        assert_eq!(GgufValueType::Uint16 as u32, 2);
        assert_eq!(GgufValueType::Int16 as u32, 3);
        assert_eq!(GgufValueType::Uint32 as u32, 4);
        assert_eq!(GgufValueType::Int32 as u32, 5);
        assert_eq!(GgufValueType::Float32 as u32, 6);
        assert_eq!(GgufValueType::Bool as u32, 7);
        assert_eq!(GgufValueType::String as u32, 8);
        assert_eq!(GgufValueType::Array as u32, 9);
        assert_eq!(GgufValueType::Uint64 as u32, 10);
        assert_eq!(GgufValueType::Int64 as u32, 11);
        assert_eq!(GgufValueType::Float64 as u32, 12);
    }

    // ── GgufValueType TryFrom boundary: 0 and max valid ─────────────────

    #[test]
    fn gguf_value_type_try_from_zero() {
        assert_eq!(GgufValueType::try_from(0).unwrap(), GgufValueType::Uint8);
    }

    #[test]
    fn gguf_value_type_try_from_max_valid() {
        assert_eq!(GgufValueType::try_from(12).unwrap(), GgufValueType::Float64);
    }

    #[test]
    fn gguf_value_type_try_from_u32_max() {
        assert!(GgufValueType::try_from(u32::MAX).is_err());
    }

    // ── GgmlDType TryFrom gaps (holes in the discriminant space) ─────────

    #[test]
    fn ggml_dtype_try_from_hole_4() {
        assert!(GgmlDType::try_from(4).is_err());
    }

    #[test]
    fn ggml_dtype_try_from_hole_5() {
        assert!(GgmlDType::try_from(5).is_err());
    }

    #[test]
    fn ggml_dtype_try_from_holes_31_to_33() {
        // Values 31-33 are between BF16(30) and TQ1_0(34), must all fail
        assert!(GgmlDType::try_from(31).is_err());
        assert!(GgmlDType::try_from(32).is_err());
        assert!(GgmlDType::try_from(33).is_err());
    }

    #[test]
    fn ggml_dtype_try_from_holes_36_to_38() {
        // Values 36-38 between TQ2_0(35) and MXFP4(39)
        assert!(GgmlDType::try_from(36).is_err());
        assert!(GgmlDType::try_from(37).is_err());
        assert!(GgmlDType::try_from(38).is_err());
    }

    #[test]
    fn ggml_dtype_try_from_holes_40_to_49() {
        // Values 40-49 between MXFP4(39) and AWQ4(50)
        for v in 40..50 {
            assert!(GgmlDType::try_from(v).is_err(), "expected err for value {v}");
        }
    }

    #[test]
    fn ggml_dtype_try_from_beyond_nvfp4() {
        assert!(GgmlDType::try_from(54).is_err());
        assert!(GgmlDType::try_from(100).is_err());
        assert!(GgmlDType::try_from(u32::MAX).is_err());
    }

    // ── GgufValue as_f32 special floats ──────────────────────────────────

    #[test]
    fn gguf_value_as_f32_nan() {
        let v = GgufValue::Float32(f32::NAN);
        let result = v.as_f32();
        assert!(result.is_some());
        assert!(result.unwrap().is_nan());
    }

    #[test]
    fn gguf_value_as_f32_infinity() {
        assert_eq!(GgufValue::Float32(f32::INFINITY).as_f32(), Some(f32::INFINITY));
        assert_eq!(GgufValue::Float32(f32::NEG_INFINITY).as_f32(), Some(f32::NEG_INFINITY));
    }

    #[test]
    fn gguf_value_as_f32_zero() {
        assert_eq!(GgufValue::Float32(0.0f32).as_f32(), Some(0.0f32));
        let neg_zero = GgufValue::Float32(-0.0f32).as_f32().unwrap();
        assert!(neg_zero.is_sign_negative());
    }

    #[test]
    fn gguf_value_as_f32_from_f64_precision() {
        // f64 to f32 conversion loses precision for large values
        let v = GgufValue::Float64(f64::MAX);
        let result = v.as_f32();
        assert!(result.is_some());
        assert!(result.unwrap().is_infinite()); // f64::MAX overflows f32 to infinity
    }

    #[test]
    fn gguf_value_as_f64_special_values() {
        let v = GgufValue::Float64(f64::NAN);
        let result = v.as_f32();
        assert!(result.is_some());
        assert!(result.unwrap().is_nan());

        assert_eq!(GgufValue::Float64(f64::INFINITY).as_f32(), Some(f64::INFINITY as f32));
        assert_eq!(GgufValue::Float64(f64::NEG_INFINITY).as_f32(), Some(f64::NEG_INFINITY as f32));
    }

    // ── GgufValue as_u64 zero values ─────────────────────────────────────

    #[test]
    fn gguf_value_as_u64_zero_signed_types() {
        assert_eq!(GgufValue::Int8(0).as_u64(), Some(0));
        assert_eq!(GgufValue::Int16(0).as_u64(), Some(0));
        assert_eq!(GgufValue::Int32(0).as_u64(), Some(0));
        assert_eq!(GgufValue::Int64(0).as_u64(), Some(0));
    }

    #[test]
    fn gguf_value_as_u64_zero_unsigned_types() {
        assert_eq!(GgufValue::Uint8(0).as_u64(), Some(0));
        assert_eq!(GgufValue::Uint16(0).as_u64(), Some(0));
        assert_eq!(GgufValue::Uint32(0).as_u64(), Some(0));
        assert_eq!(GgufValue::Uint64(0).as_u64(), Some(0));
    }

    // ── GgufValue as_bool edge cases ─────────────────────────────────────

    #[test]
    fn gguf_value_as_bool_non_bool_returns_none() {
        assert_eq!(GgufValue::Uint8(1).as_bool(), None);
        assert_eq!(GgufValue::Int32(0).as_bool(), None);
        assert_eq!(GgufValue::Float32(1.0).as_bool(), None);
        assert_eq!(GgufValue::String("true".into()).as_bool(), None);
    }

    // ── GgufValue String edge cases ──────────────────────────────────────

    #[test]
    fn gguf_value_as_str_empty_string() {
        let v = GgufValue::String("".into());
        assert_eq!(v.as_str(), Some(""));
    }

    #[test]
    fn gguf_value_as_str_unicode() {
        let v = GgufValue::String("模型权重".into());
        assert_eq!(v.as_str(), Some("模型权重"));
    }

    #[test]
    fn gguf_value_as_str_non_string_returns_none() {
        assert_eq!(GgufValue::Float32(1.0).as_str(), None);
        assert_eq!(GgufValue::Bool(true).as_str(), None);
        assert_eq!(GgufValue::Uint64(0).as_str(), None);
    }

    // ── GgufValue as_array non-array returns None ────────────────────────

    #[test]
    fn gguf_value_as_array_non_array_returns_none() {
        assert!(GgufValue::Float32(0.0).as_array().is_none());
        assert!(GgufValue::String("[]".into()).as_array().is_none());
        assert!(GgufValue::Uint64(0).as_array().is_none());
        assert!(GgufValue::Bool(false).as_array().is_none());
    }

    // ── GgufValue Clone with nested Array ────────────────────────────────

    #[test]
    fn gguf_value_clone_array_with_items() {
        let inner = GgufArray {
            item_type: GgufValueType::Int32,
            items: vec![GgufValue::Int32(10), GgufValue::Int32(20)],
        };
        let original = GgufValue::Array(inner);
        let cloned = original.clone();

        let arr = cloned.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.items[0].as_u64(), Some(10));
        assert_eq!(arr.items[1].as_u64(), Some(20));
    }

    // ── GgufValue Debug format for each variant ──────────────────────────

    #[test]
    fn gguf_value_debug_format_all_variants() {
        assert!(format!("{:?}", GgufValue::Uint8(42)).contains("Uint8"));
        assert!(format!("{:?}", GgufValue::Int8(-1)).contains("Int8"));
        assert!(format!("{:?}", GgufValue::Uint16(100)).contains("Uint16"));
        assert!(format!("{:?}", GgufValue::Int16(-100)).contains("Int16"));
        assert!(format!("{:?}", GgufValue::Uint32(1000)).contains("Uint32"));
        assert!(format!("{:?}", GgufValue::Int32(-1000)).contains("Int32"));
        assert!(format!("{:?}", GgufValue::Uint64(999)).contains("Uint64"));
        assert!(format!("{:?}", GgufValue::Int64(-999)).contains("Int64"));
        assert!(format!("{:?}", GgufValue::Float32(1.5)).contains("Float32"));
        assert!(format!("{:?}", GgufValue::Float64(2.5)).contains("Float64"));
        assert!(format!("{:?}", GgufValue::Bool(true)).contains("Bool"));
        assert!(format!("{:?}", GgufValue::String("x".into())).contains("String"));
    }

    // ── GgufArray Clone produces independent copy ────────────────────────

    #[test]
    fn gguf_array_clone_independent() {
        let arr = GgufArray {
            item_type: GgufValueType::Float64,
            items: vec![GgufValue::Float64(3.14)],
        };
        let cloned = arr.clone();
        assert_eq!(cloned.item_type, GgufValueType::Float64);
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.is_empty(), false);
    }

    // ── GgufArray with string items ──────────────────────────────────────

    #[test]
    fn gguf_array_with_string_items() {
        let arr = GgufArray {
            item_type: GgufValueType::String,
            items: vec![
                GgufValue::String("alpha".into()),
                GgufValue::String("beta".into()),
            ],
        };
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.items[0].as_str(), Some("alpha"));
        assert_eq!(arr.items[1].as_str(), Some("beta"));
    }

    // ── tensor_nbytes shape [1] for various types ────────────────────────

    #[test]
    fn tensor_nbytes_shape_one_f32() {
        assert_eq!(tensor_nbytes(GgmlDType::F32, &[1]).unwrap(), 4);
    }

    #[test]
    fn tensor_nbytes_shape_one_f16() {
        assert_eq!(tensor_nbytes(GgmlDType::F16, &[1]).unwrap(), 2);
    }

    #[test]
    fn tensor_nbytes_shape_one_bf16() {
        assert_eq!(tensor_nbytes(GgmlDType::BF16, &[1]).unwrap(), 2);
    }

    #[test]
    fn tensor_nbytes_shape_one_i8() {
        assert_eq!(tensor_nbytes(GgmlDType::I8, &[1]).unwrap(), 1);
    }

    #[test]
    fn tensor_nbytes_shape_one_q8_0() {
        // block_size=32, block_bytes=34: ceil(1/32)=1 * 34 = 34
        assert_eq!(tensor_nbytes(GgmlDType::Q8_0, &[1]).unwrap(), 34);
    }

    // ── tensor_nbytes for Q5_0, Q5_1, Q8_1, IQ4_NL ─────────────────────

    #[test]
    fn tensor_nbytes_q5_0() {
        // block_size=32, block_bytes=22
        assert_eq!(tensor_nbytes(GgmlDType::Q5_0, &[32]).unwrap(), 22);
        assert_eq!(tensor_nbytes(GgmlDType::Q5_0, &[64]).unwrap(), 44);
    }

    #[test]
    fn tensor_nbytes_q5_1() {
        // block_size=32, block_bytes=24
        assert_eq!(tensor_nbytes(GgmlDType::Q5_1, &[32]).unwrap(), 24);
    }

    #[test]
    fn tensor_nbytes_q8_1() {
        // block_size=32, block_bytes=36
        assert_eq!(tensor_nbytes(GgmlDType::Q8_1, &[32]).unwrap(), 36);
    }

    #[test]
    fn tensor_nbytes_iq4_nl() {
        // block_size=32, block_bytes=18
        assert_eq!(tensor_nbytes(GgmlDType::IQ4_NL, &[32]).unwrap(), 18);
    }

    // ── tensor_nbytes overflow detection ─────────────────────────────────

    #[test]
    fn tensor_nbytes_dimension_overflow_u64_max() {
        let result = tensor_nbytes(GgmlDType::F32, &[u64::MAX]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("overflows") || msg.contains("overflow"));
    }

    // ── tensor_nbytes large multi-dim ────────────────────────────────────

    #[test]
    fn tensor_nbytes_large_4d_tensor() {
        // F32, shape=[2, 3, 4, 5] = 120 elements * 4 bytes = 480
        assert_eq!(tensor_nbytes(GgmlDType::F32, &[2, 3, 4, 5]).unwrap(), 480);
    }

    // ── tensor_nbytes for IQ types ───────────────────────────────────────

    #[test]
    fn tensor_nbytes_iq2_xxs() {
        // block_size=256, block_bytes=66
        assert_eq!(tensor_nbytes(GgmlDType::IQ2_XXS, &[256]).unwrap(), 66);
    }

    #[test]
    fn tensor_nbytes_iq1_s() {
        // block_size=256, block_bytes=50
        assert_eq!(tensor_nbytes(GgmlDType::IQ1_S, &[256]).unwrap(), 50);
    }

    #[test]
    fn tensor_nbytes_iq1_m() {
        // block_size=256, block_bytes=56
        assert_eq!(tensor_nbytes(GgmlDType::IQ1_M, &[256]).unwrap(), 56);
    }

    // ── GgufError display InvalidMagic with actual GGUF_MAGIC ────────────

    #[test]
    fn gguf_error_display_invalid_magic_with_gguf_magic() {
        let e = GgufError::InvalidMagic(GGUF_MAGIC);
        let s = e.to_string();
        assert!(s.contains("0x46554747"));
    }

    // ── GgufError display UnsupportedVersion with v2 ─────────────────────

    #[test]
    fn gguf_error_display_unsupported_version_v2() {
        let e = GgufError::UnsupportedVersion(2);
        assert!(e.to_string().contains("2"));
    }

    // ── GgmlDType::all() contains every single variant ──────────────────

    #[test]
    fn ggml_dtype_all_contains_every_variant() {
        let all = GgmlDType::all();
        // Exhaustive check against every known discriminant
        let expected_variants: Vec<GgmlDType> = vec![
            GgmlDType::F32, GgmlDType::F16, GgmlDType::Q4_0, GgmlDType::Q4_1,
            GgmlDType::Q5_0, GgmlDType::Q5_1, GgmlDType::Q8_0, GgmlDType::Q8_1,
            GgmlDType::Q2_K, GgmlDType::Q3_K, GgmlDType::Q4_K, GgmlDType::Q5_K,
            GgmlDType::Q6_K, GgmlDType::Q8_K,
            GgmlDType::IQ2_XXS, GgmlDType::IQ2_XS, GgmlDType::IQ3_XXS,
            GgmlDType::IQ1_S, GgmlDType::IQ4_NL, GgmlDType::IQ3_S,
            GgmlDType::IQ2_S, GgmlDType::IQ4_XS,
            GgmlDType::I8, GgmlDType::I16, GgmlDType::I32, GgmlDType::I64,
            GgmlDType::F64, GgmlDType::IQ1_M, GgmlDType::BF16,
            GgmlDType::TQ1_0, GgmlDType::TQ2_0, GgmlDType::MXFP4,
            GgmlDType::AWQ4, GgmlDType::GPTQ4, GgmlDType::SQUEEZE, GgmlDType::NVFP4,
        ];
        assert_eq!(all.len(), expected_variants.len());
        for variant in &expected_variants {
            assert!(all.contains(variant), "all() missing variant {:?}", variant);
        }
    }

    // ── GgmlDType block_size/block_bytes consistency ─────────────────────

    #[test]
    fn ggml_dtype_block_bytes_positive_for_all() {
        for &dtype in GgmlDType::all() {
            assert!(dtype.block_bytes() > 0, "{dtype:?} has block_bytes=0");
            assert!(dtype.block_size() > 0, "{dtype:?} has block_size=0");
        }
    }

    // ── GgmlDType non-quantized types are exactly the 8 expected ─────────

    #[test]
    fn ggml_dtype_non_quantized_set_is_complete() {
        let non_quantized: Vec<GgmlDType> = GgmlDType::all()
            .iter()
            .filter(|d| !d.is_quantized())
            .copied()
            .collect();
        assert_eq!(non_quantized.len(), 8);
        assert!(non_quantized.contains(&GgmlDType::F32));
        assert!(non_quantized.contains(&GgmlDType::F16));
        assert!(non_quantized.contains(&GgmlDType::BF16));
        assert!(non_quantized.contains(&GgmlDType::F64));
        assert!(non_quantized.contains(&GgmlDType::I8));
        assert!(non_quantized.contains(&GgmlDType::I16));
        assert!(non_quantized.contains(&GgmlDType::I32));
        assert!(non_quantized.contains(&GgmlDType::I64));
    }

    // ── GgmlDType block_size == 1 implies block_bytes == element size ────

    #[test]
    fn ggml_dtype_block_size_one_matches_element_size() {
        assert_eq!(GgmlDType::F32.block_bytes(), 4);
        assert_eq!(GgmlDType::F16.block_bytes(), 2);
        assert_eq!(GgmlDType::BF16.block_bytes(), 2);
        assert_eq!(GgmlDType::I8.block_bytes(), 1);
        assert_eq!(GgmlDType::I16.block_bytes(), 2);
        assert_eq!(GgmlDType::I32.block_bytes(), 4);
        assert_eq!(GgmlDType::I64.block_bytes(), 8);
        assert_eq!(GgmlDType::F64.block_bytes(), 8);
    }

    // ── tensor_nbytes single dimension various types ─────────────────────

    #[test]
    fn tensor_nbytes_f64() {
        assert_eq!(tensor_nbytes(GgmlDType::F64, &[4]).unwrap(), 32);
        assert_eq!(tensor_nbytes(GgmlDType::F64, &[1]).unwrap(), 8);
    }

    // ── GgufValue Uint64 max as_u64 roundtrip ────────────────────────────

    #[test]
    fn gguf_value_uint64_max_roundtrip() {
        let v = GgufValue::Uint64(u64::MAX);
        assert_eq!(v.as_u64(), Some(u64::MAX));
    }

    // ── GgufValue Int8 MIN is negative, as_u64 returns None ──────────────

    #[test]
    fn gguf_value_int8_min_as_u64_none() {
        assert_eq!(GgufValue::Int8(i8::MIN).as_u64(), None);
        assert_eq!(GgufValue::Int16(i16::MIN).as_u64(), None);
        assert_eq!(GgufValue::Int32(i32::MIN).as_u64(), None);
        assert_eq!(GgufValue::Int64(i64::MIN).as_u64(), None);
    }

    // ── tensor_nbytes Q4_K multidimensional ──────────────────────────────

    #[test]
    fn tensor_nbytes_q4_k_multidim() {
        // block_size=256, block_bytes=144
        // shape=[256, 4] → 1 * 144 * 4 = 576
        assert_eq!(tensor_nbytes(GgmlDType::Q4_K, &[256, 4]).unwrap(), 576);
    }

    // ── tensor_nbytes Q3_K padding ───────────────────────────────────────

    #[test]
    fn tensor_nbytes_q3_k_padding() {
        // block_size=256, block_bytes=110
        // shape=[1] → ceil(1/256)=1 * 110 = 110
        assert_eq!(tensor_nbytes(GgmlDType::Q3_K, &[1]).unwrap(), 110);
        // shape=[257] → ceil(257/256)=2 * 110 = 220
        assert_eq!(tensor_nbytes(GgmlDType::Q3_K, &[257]).unwrap(), 220);
    }

    // ── tensor_nbytes AWQ4 non-aligned dimension ────────────────────────

    #[test]
    fn tensor_nbytes_awq4_non_aligned() {
        // block_size=128, block_bytes=72
        // shape=[1] → ceil(1/128)=1 * 72 = 72
        assert_eq!(tensor_nbytes(GgmlDType::AWQ4, &[1]).unwrap(), 72);
        // shape=[129] → ceil(129/128)=2 * 72 = 144
        assert_eq!(tensor_nbytes(GgmlDType::AWQ4, &[129]).unwrap(), 144);
    }

    // ── tensor_nbytes NVFP4 non-aligned dimension ────────────────────────

    #[test]
    fn tensor_nbytes_nvfp4_non_aligned() {
        // block_size=64, block_bytes=36
        // shape=[1] → ceil(1/64)=1 * 36 = 36
        assert_eq!(tensor_nbytes(GgmlDType::NVFP4, &[1]).unwrap(), 36);
        // shape=[65] → ceil(65/64)=2 * 36 = 72
        assert_eq!(tensor_nbytes(GgmlDType::NVFP4, &[65]).unwrap(), 72);
    }

    // ── GgmlDType as_str returns expected for all IQ variants ────────────

    #[test]
    fn ggml_dtype_as_str_iq_variants() {
        assert_eq!(GgmlDType::IQ2_XXS.as_str(), "IQ2_XXS");
        assert_eq!(GgmlDType::IQ2_XS.as_str(), "IQ2_XS");
        assert_eq!(GgmlDType::IQ3_XXS.as_str(), "IQ3_XXS");
        assert_eq!(GgmlDType::IQ1_S.as_str(), "IQ1_S");
        assert_eq!(GgmlDType::IQ4_NL.as_str(), "IQ4_NL");
        assert_eq!(GgmlDType::IQ3_S.as_str(), "IQ3_S");
        assert_eq!(GgmlDType::IQ2_S.as_str(), "IQ2_S");
        assert_eq!(GgmlDType::IQ4_XS.as_str(), "IQ4_XS");
        assert_eq!(GgmlDType::IQ1_M.as_str(), "IQ1_M");
    }

    // ── GgufArray is_empty consistent with len == 0 ──────────────────────

    #[test]
    fn gguf_array_is_empty_consistent_with_len() {
        let empty = GgufArray { item_type: GgufValueType::Bool, items: vec![] };
        assert_eq!(empty.is_empty(), empty.len() == 0);

        let non_empty = GgufArray {
            item_type: GgufValueType::Bool,
            items: vec![GgufValue::Bool(true)],
        };
        assert_eq!(non_empty.is_empty(), non_empty.len() == 0);
        assert_eq!(non_empty.len(), 1);
    }

    // ── GgufValue Float32 smallest positive subnormal ────────────────────

    #[test]
    fn gguf_value_as_f32_subnormal() {
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        let v = GgufValue::Float32(subnormal);
        assert_eq!(v.as_f32(), Some(subnormal));
        assert!(subnormal > 0.0);
        assert!(subnormal.is_subnormal());
    }

    // ── GgufError Io variant preserves error kind ────────────────────────

    #[test]
    fn gguf_error_io_preserves_kind() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let gguf_err: GgufError = io_err.into();
        match &gguf_err {
            GgufError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
            other => panic!("expected Io variant, got {:?}", other),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // WAVE 13 — 40 additional tests
    // ═══════════════════════════════════════════════════════════════════════

    // ── GgmlDType block_size for remaining K-quant variants ──────────────

    #[test]
    fn ggml_dtype_block_size_k_quant_remaining() {
        assert_eq!(GgmlDType::Q3_K.block_size(), 256);
        assert_eq!(GgmlDType::Q5_K.block_size(), 256);
        assert_eq!(GgmlDType::Q6_K.block_size(), 256);
        assert_eq!(GgmlDType::IQ2_XS.block_size(), 256);
        assert_eq!(GgmlDType::IQ3_S.block_size(), 256);
        assert_eq!(GgmlDType::IQ2_S.block_size(), 256);
        assert_eq!(GgmlDType::IQ4_XS.block_size(), 256);
        assert_eq!(GgmlDType::IQ1_M.block_size(), 256);
    }

    #[test]
    fn ggml_dtype_block_size_i16() {
        assert_eq!(GgmlDType::I16.block_size(), 1);
    }

    #[test]
    fn ggml_dtype_quantized_compression_ratio_property() {
        let f32_bytes_per_elem = GgmlDType::F32.block_bytes() as f64 / GgmlDType::F32.block_size() as f64;
        for &dtype in GgmlDType::all() {
            if dtype.is_quantized() {
                let ratio = dtype.block_bytes() as f64 / dtype.block_size() as f64;
                assert!(
                    ratio < f32_bytes_per_elem,
                    "{dtype:?} has bytes/elem={ratio} >= F32 bytes/elem={f32_bytes_per_elem}",
                );
            }
        }
    }

    #[test]
    fn ggml_dtype_as_str_all_unique() {
        let all = GgmlDType::all();
        let mut seen = std::collections::HashSet::new();
        for &dtype in all {
            let s = dtype.as_str();
            assert!(seen.insert(s), "duplicate as_str value: {s} from {dtype:?}");
        }
    }

    #[test]
    fn ggml_dtype_as_str_is_non_empty_and_ascii() {
        for &dtype in GgmlDType::all() {
            let s = dtype.as_str();
            assert!(!s.is_empty(), "{dtype:?} as_str is empty");
            assert!(s.is_ascii(), "{dtype:?} as_str is not ASCII: {s}");
        }
    }

    #[test]
    fn ggml_dtype_all_sorted_by_discriminant() {
        let all = GgmlDType::all();
        for window in all.windows(2) {
            assert!(
                (window[0] as u32) < (window[1] as u32),
                "all() not sorted: {:?} (disc={}) should precede {:?} (disc={})",
                window[0], window[0] as u32, window[1], window[1] as u32,
            );
        }
    }

    #[test]
    fn ggml_dtype_hash_consistent() {
        let mut set1 = std::collections::HashSet::new();
        let mut set2 = std::collections::HashSet::new();
        for &dtype in GgmlDType::all() {
            set1.insert(dtype);
            set2.insert(dtype);
        }
        assert_eq!(set1, set2);
    }

    // ── tensor_nbytes for remaining IQ types ─────────────────────────────

    #[test]
    fn tensor_nbytes_iq3_xxs() {
        assert_eq!(tensor_nbytes(GgmlDType::IQ3_XXS, &[256]).unwrap(), 98);
        assert_eq!(tensor_nbytes(GgmlDType::IQ3_XXS, &[512]).unwrap(), 196);
    }

    #[test]
    fn tensor_nbytes_iq2_xs() {
        assert_eq!(tensor_nbytes(GgmlDType::IQ2_XS, &[256]).unwrap(), 74);
        assert_eq!(tensor_nbytes(GgmlDType::IQ2_XS, &[512, 2]).unwrap(), 296);
    }

    #[test]
    fn tensor_nbytes_iq3_s() {
        assert_eq!(tensor_nbytes(GgmlDType::IQ3_S, &[256]).unwrap(), 110);
        assert_eq!(tensor_nbytes(GgmlDType::IQ3_S, &[257]).unwrap(), 220);
    }

    #[test]
    fn tensor_nbytes_iq2_s() {
        assert_eq!(tensor_nbytes(GgmlDType::IQ2_S, &[256]).unwrap(), 82);
        assert_eq!(tensor_nbytes(GgmlDType::IQ2_S, &[256, 3]).unwrap(), 246);
    }

    #[test]
    fn tensor_nbytes_iq4_xs() {
        assert_eq!(tensor_nbytes(GgmlDType::IQ4_XS, &[256]).unwrap(), 136);
        assert_eq!(tensor_nbytes(GgmlDType::IQ4_XS, &[512]).unwrap(), 272);
    }

    #[test]
    fn tensor_nbytes_iq4_xs_multidim() {
        assert_eq!(tensor_nbytes(GgmlDType::IQ4_XS, &[256, 2, 3]).unwrap(), 816);
    }

    #[test]
    fn tensor_nbytes_q5_k() {
        assert_eq!(tensor_nbytes(GgmlDType::Q5_K, &[256]).unwrap(), 176);
        assert_eq!(tensor_nbytes(GgmlDType::Q5_K, &[512]).unwrap(), 352);
    }

    #[test]
    fn tensor_nbytes_q6_k() {
        assert_eq!(tensor_nbytes(GgmlDType::Q6_K, &[256]).unwrap(), 210);
        assert_eq!(tensor_nbytes(GgmlDType::Q6_K, &[256, 4]).unwrap(), 840);
    }

    #[test]
    fn tensor_nbytes_i16() {
        assert_eq!(tensor_nbytes(GgmlDType::I16, &[10]).unwrap(), 20);
        assert_eq!(tensor_nbytes(GgmlDType::I16, &[5, 6]).unwrap(), 60);
    }

    #[test]
    fn tensor_nbytes_q5_0_padding() {
        assert_eq!(tensor_nbytes(GgmlDType::Q5_0, &[1]).unwrap(), 22);
        assert_eq!(tensor_nbytes(GgmlDType::Q5_0, &[33]).unwrap(), 44);
    }

    #[test]
    fn tensor_nbytes_q5_1_padding() {
        assert_eq!(tensor_nbytes(GgmlDType::Q5_1, &[1]).unwrap(), 24);
        assert_eq!(tensor_nbytes(GgmlDType::Q5_1, &[33]).unwrap(), 48);
    }

    #[test]
    fn tensor_nbytes_q8_0_padding() {
        assert_eq!(tensor_nbytes(GgmlDType::Q8_0, &[1]).unwrap(), 34);
        assert_eq!(tensor_nbytes(GgmlDType::Q8_0, &[33]).unwrap(), 68);
    }

    #[test]
    fn tensor_nbytes_q8_1_padding() {
        assert_eq!(tensor_nbytes(GgmlDType::Q8_1, &[1]).unwrap(), 36);
        assert_eq!(tensor_nbytes(GgmlDType::Q8_1, &[33]).unwrap(), 72);
    }

    #[test]
    fn tensor_nbytes_iq4_nl_padding() {
        assert_eq!(tensor_nbytes(GgmlDType::IQ4_NL, &[1]).unwrap(), 18);
        assert_eq!(tensor_nbytes(GgmlDType::IQ4_NL, &[33]).unwrap(), 36);
    }

    #[test]
    fn tensor_nbytes_5d_tensor() {
        // F32, shape=[2, 1, 2, 1, 2] = 8 elements * 4 bytes = 32
        assert_eq!(tensor_nbytes(GgmlDType::F32, &[2, 1, 2, 1, 2]).unwrap(), 32);
    }

    #[test]
    fn tensor_nbytes_shape_multiple_zero_dims() {
        assert_eq!(tensor_nbytes(GgmlDType::F32, &[4, 0, 3]).unwrap(), 0);
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[32, 0, 0]).unwrap(), 0);
    }

    #[test]
    fn tensor_nbytes_bf16_multidim() {
        assert_eq!(tensor_nbytes(GgmlDType::BF16, &[128, 3]).unwrap(), 768);
    }

    #[test]
    fn tensor_nbytes_q4_0_large_shape() {
        let result = tensor_nbytes(GgmlDType::Q4_0, &[1024, 1024]).unwrap();
        // ceil(1024/32)=32 blocks * 18 * 1024 = 589824
        assert_eq!(result, 589_824);
    }

    // ── GgufValue as_u64 Int16 specific boundary ────────────────────────

    #[test]
    fn gguf_value_as_u64_int16_boundary() {
        assert_eq!(GgufValue::Int16(i16::MAX).as_u64(), Some(i16::MAX as u64));
        assert_eq!(GgufValue::Int16(-1).as_u64(), None);
        assert_eq!(GgufValue::Int16(0).as_u64(), Some(0));
    }

    #[test]
    fn gguf_value_as_u64_all_unsigned_max() {
        assert_eq!(GgufValue::Uint8(u8::MAX).as_u64(), Some(u8::MAX as u64));
        assert_eq!(GgufValue::Uint16(u16::MAX).as_u64(), Some(u16::MAX as u64));
        assert_eq!(GgufValue::Uint32(u32::MAX).as_u64(), Some(u32::MAX as u64));
        assert_eq!(GgufValue::Uint64(u64::MAX).as_u64(), Some(u64::MAX));
    }

    // ── GgufValue as_f32 from f64 normal precision ──────────────────────

    #[test]
    fn gguf_value_as_f32_from_f64_normal_value() {
        let v = GgufValue::Float64(1.5);
        let result = v.as_f32().unwrap();
        assert!((result - 1.5f32).abs() < f32::EPSILON);
    }

    // ── GgufArray with bool and mixed items ─────────────────────────────

    #[test]
    fn gguf_array_with_bool_items() {
        let arr = GgufArray {
            item_type: GgufValueType::Bool,
            items: vec![GgufValue::Bool(true), GgufValue::Bool(false)],
        };
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.items[0].as_bool(), Some(true));
        assert_eq!(arr.items[1].as_bool(), Some(false));
    }

    #[test]
    fn gguf_array_with_int_types() {
        let arr = GgufArray {
            item_type: GgufValueType::Int64,
            items: vec![
                GgufValue::Int64(100),
                GgufValue::Int64(-50),
            ],
        };
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.items[0].as_u64(), Some(100));
        assert_eq!(arr.items[1].as_u64(), None);
    }

    #[test]
    fn gguf_array_with_float_items() {
        let arr = GgufArray {
            item_type: GgufValueType::Float64,
            items: vec![GgufValue::Float64(1.1), GgufValue::Float64(2.2)],
        };
        assert_eq!(arr.len(), 2);
        assert!(arr.items[0].as_f32().unwrap() > 1.0f32);
        assert!(arr.items[1].as_f32().unwrap() > 2.0f32);
    }

    #[test]
    fn gguf_array_len_after_clone_matches() {
        let arr = GgufArray {
            item_type: GgufValueType::Uint8,
            items: vec![GgufValue::Uint8(1), GgufValue::Uint8(2), GgufValue::Uint8(3)],
        };
        let cloned = arr.clone();
        assert_eq!(arr.len(), cloned.len());
        assert_eq!(arr.is_empty(), cloned.is_empty());
    }

    #[test]
    fn gguf_value_array_with_nested_empty_array() {
        let inner = GgufArray { item_type: GgufValueType::Uint32, items: vec![] };
        let outer = GgufValue::Array(GgufArray {
            item_type: GgufValueType::Array,
            items: vec![GgufValue::Array(inner.clone())],
        });
        let outer_arr = outer.as_array().unwrap();
        assert_eq!(outer_arr.len(), 1);
        let inner_val = &outer_arr.items[0];
        let inner_arr = inner_val.as_array().unwrap();
        assert!(inner_arr.is_empty());
    }

    // ── GgufValue Clone works for every variant ─────────────────────────

    #[test]
    fn gguf_value_clone_all_variants() {
        let values = vec![
            GgufValue::Uint8(42),
            GgufValue::Int8(-1),
            GgufValue::Uint16(1000),
            GgufValue::Int16(-100),
            GgufValue::Uint32(100_000),
            GgufValue::Int32(-5),
            GgufValue::Uint64(u64::MAX),
            GgufValue::Int64(i64::MAX),
            GgufValue::Float32(3.14),
            GgufValue::Float64(2.718),
            GgufValue::Bool(true),
            GgufValue::String("test".into()),
            GgufValue::Array(GgufArray { item_type: GgufValueType::Uint32, items: vec![] }),
        ];
        for v in &values {
            let cloned = v.clone();
            assert_eq!(v.as_u64().is_some(), cloned.as_u64().is_some());
            assert_eq!(v.as_f32().is_some(), cloned.as_f32().is_some());
            assert_eq!(v.as_bool().is_some(), cloned.as_bool().is_some());
            assert_eq!(v.as_str().is_some(), cloned.as_str().is_some());
            assert_eq!(v.as_array().is_some(), cloned.as_array().is_some());
        }
    }

    // ── GgufValueType hash consistency ──────────────────────────────────

    #[test]
    fn gguf_value_type_hash_consistent() {
        let a = GgufValueType::Float32;
        let b = GgufValueType::Float32;
        let mut map = std::collections::HashMap::new();
        map.insert(a, 1);
        assert_eq!(map.get(&b), Some(&1));
    }

    // ── GgufError is Send + Sync ────────────────────────────────────────

    #[test]
    fn gguf_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GgufError>();
    }

    // ── GgufError Debug format ──────────────────────────────────────────

    #[test]
    fn gguf_error_debug_format() {
        let e = GgufError::MissingMetadata("test_key".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("MissingMetadata"), "Debug should contain variant name");
    }

    // ── tensor_nbytes single element for all non-quantized ──────────────

    #[test]
    fn tensor_nbytes_single_element_all_non_quantized() {
        assert_eq!(tensor_nbytes(GgmlDType::F32, &[1]).unwrap(), 4);
        assert_eq!(tensor_nbytes(GgmlDType::F16, &[1]).unwrap(), 2);
        assert_eq!(tensor_nbytes(GgmlDType::BF16, &[1]).unwrap(), 2);
        assert_eq!(tensor_nbytes(GgmlDType::I8, &[1]).unwrap(), 1);
        assert_eq!(tensor_nbytes(GgmlDType::I16, &[1]).unwrap(), 2);
        assert_eq!(tensor_nbytes(GgmlDType::I32, &[1]).unwrap(), 4);
        assert_eq!(tensor_nbytes(GgmlDType::I64, &[1]).unwrap(), 8);
        assert_eq!(tensor_nbytes(GgmlDType::F64, &[1]).unwrap(), 8);
    }

    // ── GgmlDType block_bytes vs bytes_per_element for block-1 types ───

    #[test]
    fn ggml_dtype_non_quantized_bytes_per_element() {
        for &dtype in GgmlDType::all() {
            if !dtype.is_quantized() {
                let elem_bytes = dtype.block_bytes() / dtype.block_size();
                assert!(
                    (1..=8).contains(&elem_bytes),
                    "{dtype:?} has unexpected elem_bytes={elem_bytes}",
                );
            }
        }
    }

    // ── GgufValue as_f32 from f64 small value preserved ─────────────────

    #[test]
    fn gguf_value_as_f32_from_f64_small_value() {
        let v = GgufValue::Float64(0.25);
        let result = v.as_f32().unwrap();
        assert!((result - 0.25f32).abs() < f32::EPSILON);
    }

    // ── tensor_nbytes SQUEEZE multidim ──────────────────────────────────

    #[test]
    fn tensor_nbytes_squeeze_multidim() {
        // block_size=256, block_bytes=130
        // shape=[256, 2] → 130 * 2 = 260
        assert_eq!(tensor_nbytes(GgmlDType::SQUEEZE, &[256, 2]).unwrap(), 260);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // WAVE 14 — 15 additional tests: type_size, GgufValueType Hash/Copy/Debug,
    // string edge cases, unknown types, roundtrip, coverage gaps
    // ═══════════════════════════════════════════════════════════════════════

    // ── GgmlDType type_size (bytes per element) for all quantized ────────
    // @trace TEST-GGUF-TYPESIZE [req:REQ-GGUF-DTYPE] [level:unit]

    #[test]
    fn ggml_dtype_type_size_all_quantized() {
        // type_size = block_bytes / block_size = bytes per element
        // Block-32 types (8)
        assert!((GgmlDType::Q4_0.block_bytes() as f64 / GgmlDType::Q4_0.block_size() as f64 - 0.5625).abs() < f64::EPSILON);
        assert!((GgmlDType::Q4_1.block_bytes() as f64 / GgmlDType::Q4_1.block_size() as f64 - 0.625).abs() < f64::EPSILON);
        assert!((GgmlDType::Q5_0.block_bytes() as f64 / GgmlDType::Q5_0.block_size() as f64 - 0.6875).abs() < f64::EPSILON);
        assert!((GgmlDType::Q5_1.block_bytes() as f64 / GgmlDType::Q5_1.block_size() as f64 - 0.75).abs() < f64::EPSILON);
        assert!((GgmlDType::Q8_0.block_bytes() as f64 / GgmlDType::Q8_0.block_size() as f64 - 1.0625).abs() < f64::EPSILON);
        assert!((GgmlDType::Q8_1.block_bytes() as f64 / GgmlDType::Q8_1.block_size() as f64 - 1.125).abs() < f64::EPSILON);
        assert!((GgmlDType::IQ4_NL.block_bytes() as f64 / GgmlDType::IQ4_NL.block_size() as f64 - 0.5625).abs() < f64::EPSILON);
        assert!((GgmlDType::MXFP4.block_bytes() as f64 / GgmlDType::MXFP4.block_size() as f64 - 0.53125).abs() < f64::EPSILON);
        // K-quant block-256 types (16)
        assert!((GgmlDType::Q2_K.block_bytes() as f64 / GgmlDType::Q2_K.block_size() as f64 - 0.328125).abs() < f64::EPSILON);
        assert!((GgmlDType::Q3_K.block_bytes() as f64 / GgmlDType::Q3_K.block_size() as f64 - 0.4296875).abs() < f64::EPSILON);
        assert!((GgmlDType::Q4_K.block_bytes() as f64 / GgmlDType::Q4_K.block_size() as f64 - 0.5625).abs() < f64::EPSILON);
        assert!((GgmlDType::Q5_K.block_bytes() as f64 / GgmlDType::Q5_K.block_size() as f64 - 0.6875).abs() < f64::EPSILON);
        assert!((GgmlDType::Q6_K.block_bytes() as f64 / GgmlDType::Q6_K.block_size() as f64 - 0.8203125).abs() < f64::EPSILON);
        assert!((GgmlDType::Q8_K.block_bytes() as f64 / GgmlDType::Q8_K.block_size() as f64 - 1.140625).abs() < f64::EPSILON);
        assert!((GgmlDType::TQ1_0.block_bytes() as f64 / GgmlDType::TQ1_0.block_size() as f64 - 0.2109375).abs() < f64::EPSILON);
        assert!((GgmlDType::TQ2_0.block_bytes() as f64 / GgmlDType::TQ2_0.block_size() as f64 - 0.2578125).abs() < f64::EPSILON);
        // IQ block-256 types (9)
        assert!((GgmlDType::IQ2_XXS.block_bytes() as f64 / GgmlDType::IQ2_XXS.block_size() as f64 - 0.2578125).abs() < f64::EPSILON);
        assert!((GgmlDType::IQ2_XS.block_bytes() as f64 / GgmlDType::IQ2_XS.block_size() as f64 - 0.2890625).abs() < f64::EPSILON);
        assert!((GgmlDType::IQ3_XXS.block_bytes() as f64 / GgmlDType::IQ3_XXS.block_size() as f64 - 0.3828125).abs() < f64::EPSILON);
        assert!((GgmlDType::IQ1_S.block_bytes() as f64 / GgmlDType::IQ1_S.block_size() as f64 - 0.1953125).abs() < f64::EPSILON);
        assert!((GgmlDType::IQ3_S.block_bytes() as f64 / GgmlDType::IQ3_S.block_size() as f64 - 0.4296875).abs() < f64::EPSILON);
        assert!((GgmlDType::IQ2_S.block_bytes() as f64 / GgmlDType::IQ2_S.block_size() as f64 - 0.3203125).abs() < f64::EPSILON);
        assert!((GgmlDType::IQ4_XS.block_bytes() as f64 / GgmlDType::IQ4_XS.block_size() as f64 - 0.53125).abs() < f64::EPSILON);
        assert!((GgmlDType::IQ1_M.block_bytes() as f64 / GgmlDType::IQ1_M.block_size() as f64 - 0.21875).abs() < f64::EPSILON);
        assert!((GgmlDType::SQUEEZE.block_bytes() as f64 / GgmlDType::SQUEEZE.block_size() as f64 - 0.5078125).abs() < f64::EPSILON);
        // Modern formats
        assert!((GgmlDType::AWQ4.block_bytes() as f64 / GgmlDType::AWQ4.block_size() as f64 - 0.5625).abs() < f64::EPSILON);
        assert!((GgmlDType::GPTQ4.block_bytes() as f64 / GgmlDType::GPTQ4.block_size() as f64 - 0.5625).abs() < f64::EPSILON);
        assert!((GgmlDType::NVFP4.block_bytes() as f64 / GgmlDType::NVFP4.block_size() as f64 - 0.5625).abs() < f64::EPSILON);
    }

    // ── GgmlDType type_size for all non-quantized ─────────────────────
    // @trace TEST-GGUF-TYPESIZE [req:REQ-GGUF-DTYPE] [level:unit]

    #[test]
    fn ggml_dtype_type_size_all_non_quantized() {
        // For block_size=1 types, type_size == block_bytes
        assert_eq!(GgmlDType::F32.block_bytes(), 4);
        assert_eq!(GgmlDType::F16.block_bytes(), 2);
        assert_eq!(GgmlDType::BF16.block_bytes(), 2);
        assert_eq!(GgmlDType::F64.block_bytes(), 8);
        assert_eq!(GgmlDType::I8.block_bytes(), 1);
        assert_eq!(GgmlDType::I16.block_bytes(), 2);
        assert_eq!(GgmlDType::I32.block_bytes(), 4);
        assert_eq!(GgmlDType::I64.block_bytes(), 8);
    }

    // ── GgmlDType is_quantized exhaustive ─────────────────────────────
    // @trace TEST-GGUF-QUANT [req:REQ-GGUF-DTYPE] [level:unit]

    #[test]
    fn ggml_dtype_is_quantized_exhaustive() {
        // Non-quantized (8 of 36)
        assert!(!GgmlDType::F32.is_quantized());
        assert!(!GgmlDType::F16.is_quantized());
        assert!(!GgmlDType::BF16.is_quantized());
        assert!(!GgmlDType::F64.is_quantized());
        assert!(!GgmlDType::I8.is_quantized());
        assert!(!GgmlDType::I16.is_quantized());
        assert!(!GgmlDType::I32.is_quantized());
        assert!(!GgmlDType::I64.is_quantized());
        // Quantized (28 of 36)
        assert!(GgmlDType::Q4_0.is_quantized());
        assert!(GgmlDType::Q4_1.is_quantized());
        assert!(GgmlDType::Q5_0.is_quantized());
        assert!(GgmlDType::Q5_1.is_quantized());
        assert!(GgmlDType::Q8_0.is_quantized());
        assert!(GgmlDType::Q8_1.is_quantized());
        assert!(GgmlDType::Q2_K.is_quantized());
        assert!(GgmlDType::Q3_K.is_quantized());
        assert!(GgmlDType::Q4_K.is_quantized());
        assert!(GgmlDType::Q5_K.is_quantized());
        assert!(GgmlDType::Q6_K.is_quantized());
        assert!(GgmlDType::Q8_K.is_quantized());
        assert!(GgmlDType::IQ2_XXS.is_quantized());
        assert!(GgmlDType::IQ2_XS.is_quantized());
        assert!(GgmlDType::IQ3_XXS.is_quantized());
        assert!(GgmlDType::IQ1_S.is_quantized());
        assert!(GgmlDType::IQ4_NL.is_quantized());
        assert!(GgmlDType::IQ3_S.is_quantized());
        assert!(GgmlDType::IQ2_S.is_quantized());
        assert!(GgmlDType::IQ4_XS.is_quantized());
        assert!(GgmlDType::IQ1_M.is_quantized());
        assert!(GgmlDType::TQ1_0.is_quantized());
        assert!(GgmlDType::TQ2_0.is_quantized());
        assert!(GgmlDType::MXFP4.is_quantized());
        assert!(GgmlDType::AWQ4.is_quantized());
        assert!(GgmlDType::GPTQ4.is_quantized());
        assert!(GgmlDType::SQUEEZE.is_quantized());
        assert!(GgmlDType::NVFP4.is_quantized());
    }

    // ── GgmlDType block_size power-of-two property ────────────────────
    // @trace TEST-GGUF-DTYPE [req:REQ-GGUF-DTYPE] [level:unit]

    #[test]
    fn ggml_dtype_block_size_power_of_two_for_all() {
        for &dtype in GgmlDType::all() {
            let bs = dtype.block_size();
            assert!(bs.is_power_of_two(), "{dtype:?} block_size={bs} is not a power of 2");
        }
    }

    // ── GgufValueType: all 13 variants hash distinct ──────────────────
    // @trace TEST-GGUF-VALUETYPE [req:REQ-GGUF-VALUE] [level:unit]

    #[test]
    fn gguf_value_type_hash_all_distinct() {
        let variants: [GgufValueType; 13] = [
            GgufValueType::Uint8, GgufValueType::Int8, GgufValueType::Uint16,
            GgufValueType::Int16, GgufValueType::Uint32, GgufValueType::Int32,
            GgufValueType::Float32, GgufValueType::Bool, GgufValueType::String,
            GgufValueType::Array, GgufValueType::Uint64, GgufValueType::Int64,
            GgufValueType::Float64,
        ];
        let mut set = std::collections::HashSet::new();
        for v in variants {
            assert!(set.insert(v), "GgufValueType variant {v:?} produced duplicate hash");
        }
        assert_eq!(set.len(), 13);
    }

    // ── GgufValueType: Copy trait demonstrated ────────────────────────
    // @trace TEST-GGUF-VALUETYPE [req:REQ-GGUF-VALUE] [level:unit]

    #[test]
    fn gguf_value_type_copy_trait_demonstrated() {
        let a = GgufValueType::Float64;
        let b = a; // Copy (a is still usable after this assignment)
        let c = a; // Copy again — would fail if !Copy
        assert_eq!(a, b);
        assert_eq!(b, c);
        // Verify all 13 variants are Copy-compatible by stacking assignments
        let v0 = GgufValueType::Uint8;
        let v1 = GgufValueType::Int8;
        let v2 = GgufValueType::Uint16;
        let v3 = GgufValueType::Int16;
        let v4 = GgufValueType::Uint32;
        let v5 = GgufValueType::Int32;
        let v6 = GgufValueType::Float32;
        let v7 = GgufValueType::Bool;
        let v8 = GgufValueType::String;
        let v9 = GgufValueType::Array;
        let v10 = GgufValueType::Uint64;
        let v11 = GgufValueType::Int64;
        let v12 = GgufValueType::Float64;
        // All originals still accessible after copies
        let _ = (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12);
        // Copies from originals
        let c0 = v0; let c1 = v1; let c2 = v2; let c3 = v3;
        let _ = (c0, c1, c2, c3);
    }

    // ── GgufValueType: Debug output for all 13 variants ───────────────
    // @trace TEST-GGUF-VALUETYPE [req:REQ-GGUF-VALUE] [level:unit]

    #[test]
    fn gguf_value_type_debug_all_variants() {
        assert_eq!(format!("{:?}", GgufValueType::Uint8), "Uint8");
        assert_eq!(format!("{:?}", GgufValueType::Int8), "Int8");
        assert_eq!(format!("{:?}", GgufValueType::Uint16), "Uint16");
        assert_eq!(format!("{:?}", GgufValueType::Int16), "Int16");
        assert_eq!(format!("{:?}", GgufValueType::Uint32), "Uint32");
        assert_eq!(format!("{:?}", GgufValueType::Int32), "Int32");
        assert_eq!(format!("{:?}", GgufValueType::Float32), "Float32");
        assert_eq!(format!("{:?}", GgufValueType::Bool), "Bool");
        assert_eq!(format!("{:?}", GgufValueType::String), "String");
        assert_eq!(format!("{:?}", GgufValueType::Array), "Array");
        assert_eq!(format!("{:?}", GgufValueType::Uint64), "Uint64");
        assert_eq!(format!("{:?}", GgufValueType::Int64), "Int64");
        assert_eq!(format!("{:?}", GgufValueType::Float64), "Float64");
    }

    // ── GgufValueType roundtrip: type→u32→type for all 13 ────────────
    // @trace TEST-GGUF-VALUETYPE [req:REQ-GGUF-VALUE] [level:unit]

    #[test]
    fn gguf_value_type_roundtrip_all() {
        let cases: [(u32, GgufValueType); 13] = [
            (0, GgufValueType::Uint8), (1, GgufValueType::Int8),
            (2, GgufValueType::Uint16), (3, GgufValueType::Int16),
            (4, GgufValueType::Uint32), (5, GgufValueType::Int32),
            (6, GgufValueType::Float32), (7, GgufValueType::Bool),
            (8, GgufValueType::String), (9, GgufValueType::Array),
            (10, GgufValueType::Uint64), (11, GgufValueType::Int64),
            (12, GgufValueType::Float64),
        ];
        for (disc, variant) in cases {
            let recovered = GgufValueType::try_from(disc)
                .unwrap_or_else(|_| panic!("roundtrip failed for {variant:?} (disc={disc})"));
            assert_eq!(variant, recovered, "roundtrip mismatch for disc={disc}");
            assert_eq!(recovered as u32, disc, "discriminant mismatch for {variant:?}");
        }
    }

    // ── GgufValueType unknown values beyond valid range ───────────────
    // @trace TEST-GGUF-VALUETYPE [req:REQ-GGUF-VALUE] [level:unit]

    #[test]
    fn gguf_value_type_try_from_unknown_range() {
        // Test a systematic range of unknown IDs, not just boundary values
        for id in 13..=255 {
            assert!(
                GgufValueType::try_from(id).is_err(),
                "expected Err for unknown GgufValueType id={id}"
            );
        }
        // u32::MAX must fail
        assert!(GgufValueType::try_from(u32::MAX).is_err());
    }

    // ── GgmlDType TryFrom systematic gap boundaries ──────────────────
    // @trace TEST-GGUF-DTYPE [req:REQ-GGUF-DTYPE] [level:unit]

    #[test]
    fn ggml_dtype_try_from_systematic_gap_boundaries() {
        // All known gap regions in the discriminant space must produce Err.
        // Gap 1: IDs 4-5 (between Q4_1(3) and Q5_0(6))
        for id in 4..=5 { assert!(GgmlDType::try_from(id).is_err(), "expected err for {id}"); }
        // Gap 2: IDs 31-33 (between BF16(30) and TQ1_0(34))
        for id in 31..=33 { assert!(GgmlDType::try_from(id).is_err(), "expected err for {id}"); }
        // Gap 3: IDs 36-38 (between TQ2_0(35) and MXFP4(39))
        for id in 36..=38 { assert!(GgmlDType::try_from(id).is_err(), "expected err for {id}"); }
        // Gap 4: IDs 40-49 (between MXFP4(39) and AWQ4(50))
        for id in 40..=49 { assert!(GgmlDType::try_from(id).is_err(), "expected err for {id}"); }
        // Beyond NVFP4(53): IDs 54-63
        for id in 54..=63 { assert!(GgmlDType::try_from(id).is_err(), "expected err for {id}"); }
        // Boundary: u32::MAX
        assert!(GgmlDType::try_from(u32::MAX).is_err());
    }

    // ── GgufValue String edge cases ──────────────────────────────────
    // @trace TEST-GGUF-STRING [req:REQ-GGUF-VALUE] [level:unit]

    #[test]
    fn gguf_value_as_str_very_long() {
        // 10,000 character string
        let long = "a".repeat(10_000);
        let v = GgufValue::String(long.into());
        let s = v.as_str().unwrap();
        assert_eq!(s.len(), 10_000);
        assert!(s.chars().all(|c| c == 'a'));
    }

    #[test]
    fn gguf_value_as_str_whitespace_only() {
        let v = GgufValue::String("   \t\n\r  ".into());
        assert_eq!(v.as_str(), Some("   \t\n\r  "));
    }

    #[test]
    fn gguf_value_as_str_null_byte() {
        let v = GgufValue::String("abc\0def".into());
        assert_eq!(v.as_str(), Some("abc\0def"));
        assert_eq!(v.as_str().unwrap().len(), 7);
    }

    #[test]
    fn gguf_value_as_str_unicode_mixed() {
        let mixed = "Hello 世界 🌍 αβγ 模型权重 🔥 \u{00E9}\u{0301}";
        let v = GgufValue::String(mixed.into());
        let s = v.as_str().unwrap();
        assert_eq!(s, mixed);
        // Verify roundtrip character count
        assert_eq!(s.chars().count(), 24);
    }

    // ── tensor_nbytes Q4_1 (coverage gap) ─────────────────────────────
    // @trace TEST-GGUF-NBYTES [req:REQ-GGUF-DTYPE] [level:unit]

    #[test]
    fn tensor_nbytes_q4_1_basic_and_padding() {
        // block_size=32, block_bytes=20
        // shape=[32] → 1 * 20 = 20
        assert_eq!(tensor_nbytes(GgmlDType::Q4_1, &[32]).unwrap(), 20);
        // shape=[64] → 2 * 20 = 40
        assert_eq!(tensor_nbytes(GgmlDType::Q4_1, &[64]).unwrap(), 40);
        // shape=[1] → ceil(1/32)=1 * 20 = 20
        assert_eq!(tensor_nbytes(GgmlDType::Q4_1, &[1]).unwrap(), 20);
        // shape=[33] → ceil(33/32)=2 * 20 = 40
        assert_eq!(tensor_nbytes(GgmlDType::Q4_1, &[33]).unwrap(), 40);
        // multidim: [32, 3] → 20 * 3 = 60
        assert_eq!(tensor_nbytes(GgmlDType::Q4_1, &[32, 3]).unwrap(), 60);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 13 additional tests — edge cases, boundary conditions, coverage gaps
    // ═══════════════════════════════════════════════════════════════════════

    // ── GgufValue as_f32 negative zero from f64 ─────────────────────────

    #[test]
    fn gguf_value_as_f32_negative_zero_from_f64() {
        // Arrange: Float64 containing -0.0
        let v = GgufValue::Float64(-0.0f64);
        // Act
        let result = v.as_f32().unwrap();
        // Assert: negative zero sign is preserved through f64→f32 conversion
        assert!(result.is_sign_negative());
        assert_eq!(result, 0.0f32);
    }

    // ── GgufValue as_f32 smallest positive normal f32 ──────────────────

    #[test]
    fn gguf_value_as_f32_smallest_positive_normal() {
        // Arrange: Float32 with the smallest positive normal value
        let min_normal = f32::MIN_POSITIVE;
        let v = GgufValue::Float32(min_normal);
        // Act
        let result = v.as_f32();
        // Assert: exact preservation
        assert_eq!(result, Some(min_normal));
        assert!(!min_normal.is_subnormal());
    }

    // ── GgufValue Int8 boundary just above MIN ─────────────────────────

    #[test]
    fn gguf_value_int8_boundary_above_min() {
        // Arrange: Int8 at i8::MIN + 1 — still negative, as_u64 should return None
        let v = GgufValue::Int8(i8::MIN + 1);
        // Act
        let result = v.as_u64();
        // Assert: any negative value, even one above MIN, must return None
        assert_eq!(result, None);
    }

    // ── GgufValue String Arc<str> clone shares data ────────────────────

    #[test]
    fn gguf_value_string_arc_clone_shares_data() {
        // Arrange: a long string to make Arc sharing observable
        let long_str: Arc<str> = "x".repeat(1000).into();
        let v = GgufValue::String(long_str.clone());
        // Act: clone the GgufValue
        let cloned = v.clone();
        // Assert: both original and cloned return the same string content
        assert_eq!(v.as_str(), cloned.as_str());
        assert_eq!(v.as_str().unwrap().len(), 1000);
    }

    // ── GgufArray single item is_empty vs len consistency ───────────────

    #[test]
    fn gguf_array_single_item_consistency() {
        // Arrange: array with exactly one item
        let arr = GgufArray {
            item_type: GgufValueType::Int32,
            items: vec![GgufValue::Int32(42)],
        };
        // Act & Assert: len == 1, is_empty == false, invariant holds
        assert_eq!(arr.len(), 1);
        assert!(!arr.is_empty());
        assert_eq!(arr.is_empty(), arr.len() == 0);
    }

    // ── GgufArray clone preserves item_type ────────────────────────────

    #[test]
    fn gguf_array_clone_preserves_item_type() {
        // Arrange: array with a specific item_type
        let arr = GgufArray {
            item_type: GgufValueType::Int64,
            items: vec![GgufValue::Int64(-1)],
        };
        // Act
        let cloned = arr.clone();
        // Assert: item_type is preserved exactly
        assert_eq!(cloned.item_type, GgufValueType::Int64);
        assert_eq!(arr.item_type, cloned.item_type);
    }

    // ── tensor_nbytes GPTQ4 non-aligned dimension ──────────────────────

    #[test]
    fn tensor_nbytes_gptq4_non_aligned() {
        // Arrange: GPTQ4 with block_size=128, block_bytes=72
        // Act & Assert: shape=[1] → ceil(1/128)=1 * 72 = 72
        assert_eq!(tensor_nbytes(GgmlDType::GPTQ4, &[1]).unwrap(), 72);
        // shape=[129] → ceil(129/128)=2 * 72 = 144
        assert_eq!(tensor_nbytes(GgmlDType::GPTQ4, &[129]).unwrap(), 144);
        // shape=[256, 2] → 2 * 72 * 2 = 288
        assert_eq!(tensor_nbytes(GgmlDType::GPTQ4, &[256, 2]).unwrap(), 288);
    }

    // ── tensor_nbytes Q4_0 with outer dimension 1 ──────────────────────

    #[test]
    fn tensor_nbytes_q4_0_outer_dim_one() {
        // Arrange: Q4_0 block_size=32, block_bytes=18
        // Act & Assert: shape=[32, 1] → 1 block * 18 * 1 = 18
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[32, 1]).unwrap(), 18);
        // shape=[64, 1, 1] → 2 * 18 * 1 * 1 = 36
        assert_eq!(tensor_nbytes(GgmlDType::Q4_0, &[64, 1, 1]).unwrap(), 36);
    }

    // ── tensor_nbytes large but non-overflowing shape ───────────────────

    #[test]
    fn tensor_nbytes_large_non_overflow_shape() {
        // Arrange: I8, shape=[1000, 1000] = 1,000,000 bytes (well within usize)
        // Act
        let result = tensor_nbytes(GgmlDType::I8, &[1000, 1000]).unwrap();
        // Assert
        assert_eq!(result, 1_000_000);
    }

    // ── GgufError InvalidValueType contains numeric value ───────────────

    #[test]
    fn gguf_error_invalid_value_type_contains_number() {
        // Arrange: construct with a specific numeric ID
        let e = GgufError::InvalidValueType(42);
        // Act
        let msg = e.to_string();
        // Assert: the error message includes "42"
        assert!(msg.contains("42"), "error message should contain '42', got: {msg}");
    }

    // ── GgmlDType::all() strictly increasing discriminants ──────────────

    #[test]
    fn ggml_dtype_all_strictly_increasing_discriminants() {
        // Arrange: get all() slice
        let all = GgmlDType::all();
        // Act & Assert: every consecutive pair has strictly increasing discriminant
        for i in 0..(all.len() - 1) {
            let disc_a = all[i] as u32;
            let disc_b = all[i + 1] as u32;
            assert!(
                disc_a < disc_b,
                "all()[{i}]={:?} (disc={disc_a}) not strictly before all()[{}]={:?} (disc={disc_b})",
                all[i], i + 1, all[i + 1],
            );
        }
    }

    // ── GgmlDType block_bytes divisible by block_size for block-1 types ─

    #[test]
    fn ggml_dtype_block_bytes_equals_elem_size_for_block_size_one() {
        // Arrange/Act/Assert: for all block_size==1 types, block_bytes should be 1,2,4,or 8
        let block_one_types: Vec<GgmlDType> = GgmlDType::all()
            .iter()
            .filter(|d| d.block_size() == 1)
            .copied()
            .collect();
        // There should be exactly 8 block_size==1 types
        assert_eq!(block_one_types.len(), 8);
        // Each should have block_bytes matching a primitive element size
        for &dtype in &block_one_types {
            let bb = dtype.block_bytes();
            assert!(
                matches!(bb, 1 | 2 | 4 | 8),
                "{dtype:?} has block_size=1 but block_bytes={bb} (expected 1,2,4,or 8)",
            );
        }
    }

    // ── GgufValue as_u64 Uint8 zero returns Some(0) ────────────────────

    #[test]
    fn gguf_value_uint8_zero_as_u64() {
        // Arrange
        let v = GgufValue::Uint8(0);
        // Act
        let result = v.as_u64();
        // Assert: zero is a valid non-negative integer, must return Some(0)
        assert_eq!(result, Some(0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 10 additional tests — overflow, Arc semantics, coverage gaps
    // ═══════════════════════════════════════════════════════════════════════

    // ── tensor_nbytes: outer dimension overflow on multiplication ────────

    #[test]
    fn tensor_nbytes_outer_dim_overflow_checked_mul() {
        // Arrange: F32 with shape=[2, u64::MAX]. row_bytes=8, 8 * u64::MAX overflows usize.
        // Act
        let result = tensor_nbytes(GgmlDType::F32, &[2, u64::MAX]);
        // Assert: must return Err, not panic or silently wrap
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("overflow"),
            "error message should mention overflow, got: {msg}"
        );
    }

    // ── GgufValue String Arc<str> clone is reference-counted ─────────────

    #[test]
    fn gguf_value_string_arc_strong_count_increases_on_clone() {
        // Arrange: a string value backed by Arc<str>
        let s: Arc<str> = "shared_data".into();
        let v = GgufValue::String(s.clone());
        let count_before = Arc::strong_count(&s);
        // Act: clone the GgufValue (which clones the inner Arc<str>)
        let cloned = v.clone();
        let count_after = Arc::strong_count(&s);
        // Assert: both the original and clone are valid, and ref count increased
        assert_eq!(v.as_str(), cloned.as_str());
        assert_eq!(count_after, count_before + 1);
    }

    // ── tensor_nbytes TQ2_0 multidimensional ─────────────────────────────

    #[test]
    fn tensor_nbytes_tq2_0_multidim() {
        // Arrange: TQ2_0 with block_size=256, block_bytes=66
        // Act & Assert: shape=[256, 3] → 1 block * 66 * 3 = 198
        assert_eq!(tensor_nbytes(GgmlDType::TQ2_0, &[256, 3]).unwrap(), 198);
        // shape=[512, 2, 2] → 2 * 66 * 2 * 2 = 528
        assert_eq!(tensor_nbytes(GgmlDType::TQ2_0, &[512, 2, 2]).unwrap(), 528);
    }

    // ── GgufError source() returns None for all non-chained variants ─────

    #[test]
    fn gguf_error_source_none_for_non_chained_variants() {
        use std::error::Error;
        // Arrange: each non-chained GgufError variant
        let variants: Vec<GgufError> = vec![
            GgufError::InvalidMagic(0),
            GgufError::UnsupportedVersion(1),
            GgufError::InvalidValueType(2),
            GgufError::InvalidDType(3),
            GgufError::MissingMetadata("k".into()),
            GgufError::InvalidMetadata("v".into()),
            GgufError::TensorNotFound("t".into()),
            GgufError::TensorOutOfBounds("o".into()),
            GgufError::UnsupportedType(GgmlDType::F32),
            GgufError::ParseError("p".into()),
        ];
        // Act & Assert: every non-chained variant must have source() == None
        for (i, err) in variants.into_iter().enumerate() {
            assert!(
                err.source().is_none(),
                "variant at index {i} should have no source, but returned Some"
            );
        }
    }

    // ── GgufValue nested array clone produces independent items ──────────

    #[test]
    fn gguf_value_nested_array_clone_independence() {
        // Arrange: outer array containing an inner array with data
        let inner = GgufArray {
            item_type: GgufValueType::Float32,
            items: vec![GgufValue::Float32(1.0), GgufValue::Float32(2.0)],
        };
        let outer = GgufValue::Array(GgufArray {
            item_type: GgufValueType::Array,
            items: vec![GgufValue::Array(inner)],
        });
        // Act: clone the outer value
        let cloned = outer.clone();
        // Assert: cloned array has same structure and values
        let outer_arr = outer.as_array().unwrap();
        let cloned_arr = cloned.as_array().unwrap();
        assert_eq!(outer_arr.len(), cloned_arr.len());
        let outer_inner = outer_arr.items[0].as_array().unwrap();
        let cloned_inner = cloned_arr.items[0].as_array().unwrap();
        assert_eq!(outer_inner.len(), cloned_inner.len());
        assert_eq!(
            outer_inner.items[0].as_f32(),
            cloned_inner.items[0].as_f32()
        );
    }

    // ── tensor_nbytes MXFP4 non-aligned dimension padding ────────────────

    #[test]
    fn tensor_nbytes_mxfp4_non_aligned_padding() {
        // Arrange: MXFP4 with block_size=32, block_bytes=17
        // Act & Assert: shape=[1] → ceil(1/32)=1 * 17 = 17 (heavily padded)
        assert_eq!(tensor_nbytes(GgmlDType::MXFP4, &[1]).unwrap(), 17);
        // shape=[31] → ceil(31/32)=1 * 17 = 17
        assert_eq!(tensor_nbytes(GgmlDType::MXFP4, &[31]).unwrap(), 17);
        // shape=[33] → ceil(33/32)=2 * 17 = 34
        assert_eq!(tensor_nbytes(GgmlDType::MXFP4, &[33]).unwrap(), 34);
        // shape=[33, 2] → 34 * 2 = 68
        assert_eq!(tensor_nbytes(GgmlDType::MXFP4, &[33, 2]).unwrap(), 68);
    }

    // ── GgmlDType all quantized types have strictly less than 4 bytes/elem

    #[test]
    fn ggml_dtype_quantized_bytes_per_elem_under_f32() {
        // Arrange: F32 is 4 bytes/elem. All quantized types must be strictly less.
        // Act & Assert
        for &dtype in GgmlDType::all() {
            if dtype.is_quantized() {
                let bytes_per_elem = dtype.block_bytes() as f64 / dtype.block_size() as f64;
                assert!(
                    bytes_per_elem < 4.0,
                    "{dtype:?} has bytes/elem={bytes_per_elem} >= 4.0 (F32 equivalent)",
                );
            }
        }
    }

    // ── GgufValue Array accessor returns correct items ───────────────────

    #[test]
    fn gguf_value_array_accessor_items_match() {
        // Arrange: an array with heterogeneous-typed GgufValue items
        let arr = GgufArray {
            item_type: GgufValueType::Uint32,
            items: vec![
                GgufValue::Uint32(100),
                GgufValue::Uint32(200),
                GgufValue::Uint32(300),
            ],
        };
        let wrapper = GgufValue::Array(arr);
        // Act
        let accessed = wrapper.as_array().unwrap();
        // Assert: all items are accessible and correct via their own accessors
        assert_eq!(accessed.len(), 3);
        assert_eq!(accessed.items[0].as_u64(), Some(100));
        assert_eq!(accessed.items[1].as_u64(), Some(200));
        assert_eq!(accessed.items[2].as_u64(), Some(300));
        // item_type field matches
        assert_eq!(accessed.item_type, GgufValueType::Uint32);
    }

    // ── tensor_nbytes large shape[0] that does not overflow usize ─────────

    #[test]
    fn tensor_nbytes_large_shape0_no_overflow() {
        // Arrange: I8 with shape=[500_000_000]. 500M bytes fits in usize on 64-bit.
        // Act
        let result = tensor_nbytes(GgmlDType::I8, &[500_000_000]);
        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 500_000_000);
    }

    // ── GgufError InvalidDType error message contains numeric value ──────

    #[test]
    fn gguf_error_invalid_dtype_error_message_contains_number() {
        // Arrange: construct InvalidDType with a specific unknown discriminant
        let err = GgufError::InvalidDType(999);
        // Act
        let msg = err.to_string();
        // Assert: the display message includes the numeric value "999"
        assert!(
            msg.contains("999"),
            "InvalidDType(999) display should contain '999', got: {msg}"
        );
    }
}
