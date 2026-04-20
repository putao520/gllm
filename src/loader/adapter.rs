use std::path::Path;

/// Placeholder for packed quantized bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedBits {
    Int1,
    Int2,
    Int4,
}

/// StorageFormat — 适配层物理源格式扩展（文件中的原始物理存储格式，JIT 根据 QuantType 生成对应内核）
///
/// 这是 API-GGUF-ADAPTER 的核心类型，表示 GGUF 文件中的原始物理存储格式。
/// 量化内核分派通过 `ggml_dtype_to_quant_type()` 映射到 `QuantType`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageFormat {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// Brain floating point (16-bit)
    BF16,
    /// 8-bit unsigned integer
    U8,
    /// Packed quantized bits (Int1/Int2/Int4)
    PackedU8(PackedBits),
}

/// DType 是 StorageFormat 的别名（向后兼容）
pub type DType = StorageFormat;

impl StorageFormat {
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
        // MXFP4 (OCP Microscaling FP4). The SafeTensors loader rewires
        // `<prefix>_blocks` + `<prefix>_scales` pairs (OpenAI gpt-oss layout)
        // into GGUF-style interleaved bytes before upload, so the
        // compat::cpu_backend::dequantize `Mxfp4` path handles them uniformly
        // regardless of source format. Block size is fixed to the OCP standard
        // 32 (matches both GGUF type 39 and gpt-oss packaging).
        GgmlDType::MXFP4 => Some(QuantType::Mxfp4 { block_size: 32 }),
        // Exotic GGUF types not yet mapped to kernels.
        GgmlDType::TQ1_0 | GgmlDType::TQ2_0 => None,
    }
}

/// KernelTensorView — 零拷贝张量视图
///
/// 用于将 GGUF Tensor 转换为 gllm-kernels 格式。
/// 生命周期绑定到 GGUF reader，确保零拷贝安全性。
#[derive(Debug, Clone)]
pub struct KernelTensorView<'a> {
    /// 物理层映射结构（文件中的原始物理存储格式）
    pub storage_format: StorageFormat,
    /// 张量形状
    pub shape: Vec<usize>,
    /// 生命周期绑定的字节切片（零拷贝）
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
        let storage_format = map_storage_format(info.dtype)?;
        let data = self.reader.tensor_bytes(name)?;

        let mut shape = Vec::with_capacity(info.shape.len());
        for &dim in &info.shape {
            let dim = usize::try_from(dim)
                .map_err(|_| GgufError::ParseError("tensor shape overflows usize".to_string()))?;
            shape.push(dim);
        }

        Ok(KernelTensorView { storage_format, shape, data })
    }
}

/// 将 GGUF 的 GgmlDType 映射到适配层 StorageFormat
///
/// 这是物理存储格式的映射，JIT 根据 QuantType 生成对应内核。
/// 量化内核分派通过 `ggml_dtype_to_quant_type()` 映射到 `QuantType`。
///
/// # 映射表
///
/// | GGUF 类型 | StorageFormat | QuantType |
/// |-----------|---------------|-----------|
/// | F32 | F32 | None |
/// | F16 | F16 | None |
/// | BF16 | BF16 | None |
/// | Q4_0/Q4_1/Q4_K/IQ4_NL/IQ4_XS/MXFP4 | PackedU8(Int4) | Q4_0/Q4_1/Q4K/IQ4NL/IQ4XS/Mxfp4{32} |
/// | Q2_K/IQ2_XXS/IQ2_XS/IQ2_S/TQ2_0 | PackedU8(Int2) | Q2K/IQ2XXS/IQ2XS/IQ2S/— |
/// | IQ1_S/IQ1_M/TQ1_0 | PackedU8(Int1) | IQ1S/IQ1M/— |
/// | Q8_0/Q8_1/Q8_K/I8 | U8 | Q8_0/Q8_1/Q8K/— |
/// | Q3_K/Q5_0/Q5_1/Q5_K/Q6_K/IQ3_XXS/IQ3_S | UnsupportedType | Q3K/Q5_0/Q5_1/Q5K/Q6K/IQ3XXS/IQ3S |
pub fn map_storage_format(dtype: GgmlDType) -> Result<StorageFormat, GgufError> {
    let mapped = match dtype {
        GgmlDType::F32 => StorageFormat::F32,
        GgmlDType::F16 => StorageFormat::F16,
        GgmlDType::BF16 => StorageFormat::BF16,

        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q4_K
        | GgmlDType::IQ4_NL
        | GgmlDType::IQ4_XS
        | GgmlDType::MXFP4 => StorageFormat::PackedU8(PackedBits::Int4),

        GgmlDType::Q2_K
        | GgmlDType::IQ2_XXS
        | GgmlDType::IQ2_XS
        | GgmlDType::IQ2_S
        | GgmlDType::TQ2_0 => StorageFormat::PackedU8(PackedBits::Int2),

        GgmlDType::IQ1_S | GgmlDType::IQ1_M | GgmlDType::TQ1_0 => StorageFormat::PackedU8(PackedBits::Int1),

        GgmlDType::Q8_0 | GgmlDType::Q8_1 | GgmlDType::Q8_K | GgmlDType::I8 => StorageFormat::U8,

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::gguf::GgmlDType;

    /// TEST-GGUF-010: 量化类型映射测试 — GgmlDType → StorageFormat 正确映射
    #[test]
    fn test_gguf_010_quant_type_mapping() {
        // P0 类型
        assert_eq!(map_storage_format(GgmlDType::F32).unwrap(), StorageFormat::F32);
        assert_eq!(map_storage_format(GgmlDType::F16).unwrap(), StorageFormat::F16);
        assert_eq!(map_storage_format(GgmlDType::BF16).unwrap(), StorageFormat::BF16);
        assert_eq!(map_storage_format(GgmlDType::Q4_0).unwrap(), StorageFormat::PackedU8(PackedBits::Int4));
        assert_eq!(map_storage_format(GgmlDType::Q8_0).unwrap(), StorageFormat::U8);

        // P1 类型
        assert_eq!(map_storage_format(GgmlDType::Q4_K).unwrap(), StorageFormat::PackedU8(PackedBits::Int4));
        assert_eq!(map_storage_format(GgmlDType::Q8_K).unwrap(), StorageFormat::U8);

        // P2 类型
        assert_eq!(map_storage_format(GgmlDType::Q2_K).unwrap(), StorageFormat::PackedU8(PackedBits::Int2));

        // IQ 系列
        assert_eq!(map_storage_format(GgmlDType::IQ4_NL).unwrap(), StorageFormat::PackedU8(PackedBits::Int4));
        assert_eq!(map_storage_format(GgmlDType::IQ2_XXS).unwrap(), StorageFormat::PackedU8(PackedBits::Int2));
        assert_eq!(map_storage_format(GgmlDType::IQ1_S).unwrap(), StorageFormat::PackedU8(PackedBits::Int1));

        // 不支持的类型返回 Err
        assert!(map_storage_format(GgmlDType::Q5_0).is_err());
        assert!(map_storage_format(GgmlDType::Q6_K).is_err());
        assert!(map_storage_format(GgmlDType::Q3_K).is_err());
        assert!(map_storage_format(GgmlDType::F64).is_err());
    }

    /// TEST-GGUF-011: 泛型约束验证 — GGUF 解析器返回原始字节 + 类型标识符
    /// 适配层负责类型映射，不依赖 GGUF 内部类型
    #[test]
    fn test_gguf_011_adapter_type_isolation() {
        // map_storage_format 是纯函数，不依赖 GGUF 文件 I/O
        // 验证所有 StorageFormat 变体可以被正确构造和比较
        let types = [
            StorageFormat::F32,
            StorageFormat::F16,
            StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int1),
            StorageFormat::PackedU8(PackedBits::Int2),
            StorageFormat::PackedU8(PackedBits::Int4),
        ];
        // 每种类型与自身相等
        for t in &types {
            assert_eq!(t, t);
        }
        // F32 与 F16 不相等
        assert_ne!(StorageFormat::F32, StorageFormat::F16);
        // Int1 与 Int4 不相等
        assert_ne!(
            StorageFormat::PackedU8(PackedBits::Int1),
            StorageFormat::PackedU8(PackedBits::Int4)
        );
    }
}
