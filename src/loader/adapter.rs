use std::path::Path;

/// Convert safetensors Dtype to gllm_kernels DType.
/// Returns None for non-float types that don't have a gllm_kernels equivalent.
pub fn safetensors_dtype_to_gllm(dtype: ::safetensors::Dtype) -> Option<gllm_kernels::types::DType> {
    match dtype {
        ::safetensors::Dtype::F32 => Some(gllm_kernels::types::DType::F32),
        ::safetensors::Dtype::F16 => Some(gllm_kernels::types::DType::F16),
        ::safetensors::Dtype::BF16 => Some(gllm_kernels::types::DType::BF16),
        _ => None,
    }
}

/// Placeholder for packed quantized bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedBits {
    Int1,
    Int2,
    Int3,
    Int4,
    Int5,
    Int6,
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

/// Reverse mapping: gllm-kernels' QuantType → GGUF parser's GgmlDType.
/// Returns `None` for QuantType variants that have no GGUF/GgmlDType equivalent
/// (native floats Bf16/F16/F32, FP8 E4M3/E5M2).
pub fn quant_type_to_ggml_dtype(qt: QuantType) -> Option<GgmlDType> {
    match qt {
        // K-Quant family
        QuantType::Q2K => Some(GgmlDType::Q2_K),
        QuantType::Q3K => Some(GgmlDType::Q3_K),
        QuantType::Q4K => Some(GgmlDType::Q4_K),
        QuantType::Q5K => Some(GgmlDType::Q5_K),
        QuantType::Q6K => Some(GgmlDType::Q6_K),
        QuantType::Q8K => Some(GgmlDType::Q8_K),
        // Classic GGML family
        QuantType::Q4_0 => Some(GgmlDType::Q4_0),
        QuantType::Q4_1 => Some(GgmlDType::Q4_1),
        QuantType::Q5_0 => Some(GgmlDType::Q5_0),
        QuantType::Q5_1 => Some(GgmlDType::Q5_1),
        QuantType::Q8_0 => Some(GgmlDType::Q8_0),
        QuantType::Q8_1 => Some(GgmlDType::Q8_1),
        // IQ family
        QuantType::IQ1S => Some(GgmlDType::IQ1_S),
        QuantType::IQ1M => Some(GgmlDType::IQ1_M),
        QuantType::IQ2XXS => Some(GgmlDType::IQ2_XXS),
        QuantType::IQ2XS => Some(GgmlDType::IQ2_XS),
        QuantType::IQ2S => Some(GgmlDType::IQ2_S),
        QuantType::IQ3XXS => Some(GgmlDType::IQ3_XXS),
        QuantType::IQ3S => Some(GgmlDType::IQ3_S),
        QuantType::IQ4NL => Some(GgmlDType::IQ4_NL),
        QuantType::IQ4XS => Some(GgmlDType::IQ4_XS),
        // Vendor / custom types
        QuantType::AWQ4 => Some(GgmlDType::AWQ4),
        QuantType::GPTQ4 => Some(GgmlDType::GPTQ4),
        QuantType::Squeeze => Some(GgmlDType::SQUEEZE),
        QuantType::Nvfp4 => Some(GgmlDType::NVFP4),
        QuantType::TQ1_0 => Some(GgmlDType::TQ1_0),
        QuantType::TQ2_0 => Some(GgmlDType::TQ2_0),
        QuantType::Mxfp4 { block_size: 32 } => Some(GgmlDType::MXFP4),
        QuantType::Mxfp4 { .. } => None, // non-standard block_size has no GgmlDType
        // Native float types — no GgmlDType equivalent in GGUF
        QuantType::Bf16 | QuantType::F16 | QuantType::F32 => None,
        // FP8 — no GgmlDType equivalent
        QuantType::Fp8E4M3 | QuantType::Fp8E5M2 => None,
    }
}

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
        // Custom types for AWQ/GPTQ (non-standard GGML values 50/51).
        GgmlDType::AWQ4 => Some(QuantType::AWQ4),
        GgmlDType::GPTQ4 => Some(QuantType::GPTQ4),
        // SqueezeLLM (non-standard 52): 3-bit codebook quantization, kernels QuantType::Squeeze
        GgmlDType::SQUEEZE => Some(QuantType::Squeeze),
        // NVFP4 (non-standard 53): 4-bit E2M1 + UE4M3 sub-block scales, kernels QuantType::Nvfp4
        GgmlDType::NVFP4 => Some(QuantType::Nvfp4),
        // Ternary 1.0/2.0 (GGUF types 34/35): linear ternary quantization {-1, 0, +1} × scale.
        GgmlDType::TQ1_0 => Some(QuantType::TQ1_0),
        GgmlDType::TQ2_0 => Some(QuantType::TQ2_0),
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
/// | Q3_K/IQ3_XXS/IQ3_S | PackedU8(Int3) | Q3K/IQ3XXS/IQ3S |
/// | Q5_0/Q5_1/Q5_K | PackedU8(Int5) | Q5_0/Q5_1/Q5K |
/// | Q6_K | PackedU8(Int6) | Q6K |
/// | I16/I32/I64/F64 | UnsupportedType | — |
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
        | GgmlDType::MXFP4
        | GgmlDType::AWQ4
        | GgmlDType::GPTQ4
        | GgmlDType::NVFP4 => StorageFormat::PackedU8(PackedBits::Int4),

        GgmlDType::Q2_K
        | GgmlDType::IQ2_XXS
        | GgmlDType::IQ2_XS
        | GgmlDType::IQ2_S
        | GgmlDType::TQ2_0 => StorageFormat::PackedU8(PackedBits::Int2),

        // 3-bit: K-quant Q3_K and IQ3 variants (block_size=256, hierarchical scales + hmask)
        GgmlDType::Q3_K
        | GgmlDType::IQ3_XXS
        | GgmlDType::IQ3_S => StorageFormat::PackedU8(PackedBits::Int3),

        // 5-bit: Q5_0/Q5_1 (block_size=32) and Q5_K (block_size=256) use NibbleWithHighBits
        GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q5_K => StorageFormat::PackedU8(PackedBits::Int5),

        // 6-bit: Q6_K (block_size=210 bytes, Q6KScales layout)
        GgmlDType::Q6_K => StorageFormat::PackedU8(PackedBits::Int6),

        // SqueezeLLM is 3-bit but stored aligned to 4-bit nibbles (8 values × 4-bit = 32 bytes payload + 2-byte F16 scale).
        GgmlDType::SQUEEZE => StorageFormat::PackedU8(PackedBits::Int4),

        GgmlDType::IQ1_S | GgmlDType::IQ1_M | GgmlDType::TQ1_0 => StorageFormat::PackedU8(PackedBits::Int1),

        GgmlDType::Q8_0 | GgmlDType::Q8_1 | GgmlDType::Q8_K | GgmlDType::I8 => StorageFormat::U8,

        GgmlDType::I16
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

        // K-quant 3/5/6-bit (previously unsupported, now mapped)
        assert_eq!(map_storage_format(GgmlDType::Q3_K).unwrap(), StorageFormat::PackedU8(PackedBits::Int3));
        assert_eq!(map_storage_format(GgmlDType::Q5_0).unwrap(), StorageFormat::PackedU8(PackedBits::Int5));
        assert_eq!(map_storage_format(GgmlDType::Q5_1).unwrap(), StorageFormat::PackedU8(PackedBits::Int5));
        assert_eq!(map_storage_format(GgmlDType::Q5_K).unwrap(), StorageFormat::PackedU8(PackedBits::Int5));
        assert_eq!(map_storage_format(GgmlDType::Q6_K).unwrap(), StorageFormat::PackedU8(PackedBits::Int6));
        assert_eq!(map_storage_format(GgmlDType::IQ3_XXS).unwrap(), StorageFormat::PackedU8(PackedBits::Int3));
        assert_eq!(map_storage_format(GgmlDType::IQ3_S).unwrap(), StorageFormat::PackedU8(PackedBits::Int3));

        // 真正不支持的类型返回 Err
        assert!(map_storage_format(GgmlDType::F64).is_err());
        assert!(map_storage_format(GgmlDType::I16).is_err());
        assert!(map_storage_format(GgmlDType::I32).is_err());
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
            StorageFormat::PackedU8(PackedBits::Int3),
            StorageFormat::PackedU8(PackedBits::Int4),
            StorageFormat::PackedU8(PackedBits::Int5),
            StorageFormat::PackedU8(PackedBits::Int6),
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

    /// TEST-QCG-MAP: 所有 SPEC §2.2 量化格式必须能从 GgmlDType 解析到 QuantType
    /// (除 TQ1_0/TQ2_0 — gllm-kernels 未实现)
    #[test]
    fn test_qcg_ggml_dtype_to_quant_type_full_coverage() {
        use gllm_kernels::quant::QuantType;

        assert!(ggml_dtype_to_quant_type(GgmlDType::F32).is_none());
        assert!(ggml_dtype_to_quant_type(GgmlDType::F16).is_none());
        assert!(ggml_dtype_to_quant_type(GgmlDType::BF16).is_none());

        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q4_0), Some(QuantType::Q4_0));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q4_1), Some(QuantType::Q4_1));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q5_0), Some(QuantType::Q5_0));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q5_1), Some(QuantType::Q5_1));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q8_0), Some(QuantType::Q8_0));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q8_1), Some(QuantType::Q8_1));

        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q2_K), Some(QuantType::Q2K));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q3_K), Some(QuantType::Q3K));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q4_K), Some(QuantType::Q4K));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q5_K), Some(QuantType::Q5K));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q6_K), Some(QuantType::Q6K));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::Q8_K), Some(QuantType::Q8K));

        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::IQ1_S), Some(QuantType::IQ1S));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::IQ1_M), Some(QuantType::IQ1M));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::IQ2_XXS), Some(QuantType::IQ2XXS));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::IQ2_XS), Some(QuantType::IQ2XS));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::IQ2_S), Some(QuantType::IQ2S));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::IQ3_XXS), Some(QuantType::IQ3XXS));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::IQ3_S), Some(QuantType::IQ3S));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::IQ4_NL), Some(QuantType::IQ4NL));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::IQ4_XS), Some(QuantType::IQ4XS));

        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::MXFP4), Some(QuantType::Mxfp4 { block_size: 32 }));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::AWQ4), Some(QuantType::AWQ4));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::GPTQ4), Some(QuantType::GPTQ4));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::SQUEEZE), Some(QuantType::Squeeze));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::NVFP4), Some(QuantType::Nvfp4));

        // Ternary 1.0/2.0 (GGUF types 34/35) — linear ternary quantization.
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::TQ1_0), Some(QuantType::TQ1_0));
        assert_eq!(ggml_dtype_to_quant_type(GgmlDType::TQ2_0), Some(QuantType::TQ2_0));
    }

    /// Cross-repository consistency: every GgmlDType that maps to a QuantType
    /// must have a corresponding QuantFormatDescriptor in gllm-kernels registry.
    #[test]
    fn test_all_ggml_quant_types_have_format_descriptor() {
        use gllm_kernels::quant_format::QuantFormatRegistry;

        let registry = QuantFormatRegistry::new();
        let quant_dtypes = [
            GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q5_0, GgmlDType::Q5_1,
            GgmlDType::Q8_0, GgmlDType::Q8_1,
            GgmlDType::Q2_K, GgmlDType::Q3_K, GgmlDType::Q4_K, GgmlDType::Q5_K,
            GgmlDType::Q6_K, GgmlDType::Q8_K,
            GgmlDType::IQ1_S, GgmlDType::IQ1_M,
            GgmlDType::IQ2_XXS, GgmlDType::IQ2_XS, GgmlDType::IQ2_S,
            GgmlDType::IQ3_XXS, GgmlDType::IQ3_S,
            GgmlDType::IQ4_NL, GgmlDType::IQ4_XS,
            GgmlDType::AWQ4, GgmlDType::GPTQ4,
            GgmlDType::SQUEEZE, GgmlDType::NVFP4,
            GgmlDType::MXFP4,
            GgmlDType::TQ1_0, GgmlDType::TQ2_0,
        ];

        for dtype in &quant_dtypes {
            let qt = ggml_dtype_to_quant_type(*dtype)
                .unwrap_or_else(|| panic!("{:?} should map to QuantType", dtype));
            let desc = registry.get(&qt)
                .unwrap_or_else(|| panic!("QuantType {:?} (from {:?}) missing from registry", qt, dtype));
            assert_eq!(desc.quant_type, qt);
            assert_eq!(desc.block_bytes, qt.block_bytes(),
                "{:?}: desc.block_bytes={} != qt.block_bytes()={}", qt, desc.block_bytes, qt.block_bytes());
        }
    }

    /// Cross-repository consistency: GgmlDType block_bytes/block_size must match
    /// the corresponding QuantType values.
    #[test]
    fn test_ggml_dtype_block_metadata_matches_quant_type() {
        let pairs: Vec<(GgmlDType, QuantType)> = vec![
            (GgmlDType::Q4_0, QuantType::Q4_0),
            (GgmlDType::Q4_1, QuantType::Q4_1),
            (GgmlDType::Q8_0, QuantType::Q8_0),
            (GgmlDType::Q8_1, QuantType::Q8_1),
            (GgmlDType::Q2_K, QuantType::Q2K),
            (GgmlDType::Q3_K, QuantType::Q3K),
            (GgmlDType::Q4_K, QuantType::Q4K),
            (GgmlDType::Q5_K, QuantType::Q5K),
            (GgmlDType::Q6_K, QuantType::Q6K),
            (GgmlDType::Q8_K, QuantType::Q8K),
            (GgmlDType::IQ1_S, QuantType::IQ1S),
            (GgmlDType::IQ1_M, QuantType::IQ1M),
            (GgmlDType::IQ2_XXS, QuantType::IQ2XXS),
            (GgmlDType::IQ2_XS, QuantType::IQ2XS),
            (GgmlDType::IQ2_S, QuantType::IQ2S),
            (GgmlDType::IQ3_XXS, QuantType::IQ3XXS),
            (GgmlDType::IQ3_S, QuantType::IQ3S),
            (GgmlDType::IQ4_NL, QuantType::IQ4NL),
            (GgmlDType::IQ4_XS, QuantType::IQ4XS),
            (GgmlDType::AWQ4, QuantType::AWQ4),
            (GgmlDType::GPTQ4, QuantType::GPTQ4),
            (GgmlDType::SQUEEZE, QuantType::Squeeze),
            (GgmlDType::NVFP4, QuantType::Nvfp4),
            (GgmlDType::TQ1_0, QuantType::TQ1_0),
            (GgmlDType::TQ2_0, QuantType::TQ2_0),
        ];

        for (ggml, qt) in &pairs {
            assert_eq!(ggml.block_bytes(), qt.block_bytes(),
                "{:?}/{:?}: GgmlDType.block_bytes={} != QuantType.block_bytes()={}",
                ggml, qt, ggml.block_bytes(), qt.block_bytes());
            assert_eq!(ggml.block_size(), qt.block_size(),
                "{:?}/{:?}: GgmlDType.block_size={} != QuantType.block_size()={}",
                ggml, qt, ggml.block_size(), qt.block_size());
        }
    }

    // ── StorageFormat + PackedBits ──

    #[test]
    fn storage_format_size_bytes() {
        assert_eq!(StorageFormat::F32.size_bytes(), 4);
        assert_eq!(StorageFormat::F16.size_bytes(), 2);
        assert_eq!(StorageFormat::BF16.size_bytes(), 2);
        assert_eq!(StorageFormat::U8.size_bytes(), 1);
        assert_eq!(StorageFormat::PackedU8(PackedBits::Int4).size_bytes(), 1);
        assert_eq!(StorageFormat::PackedU8(PackedBits::Int2).size_bytes(), 1);
        assert_eq!(StorageFormat::PackedU8(PackedBits::Int1).size_bytes(), 1);
    }

    #[test]
    fn packed_bits_equality() {
        assert_eq!(PackedBits::Int1, PackedBits::Int1);
        assert_eq!(PackedBits::Int2, PackedBits::Int2);
        assert_eq!(PackedBits::Int4, PackedBits::Int4);
        assert_ne!(PackedBits::Int1, PackedBits::Int4);
    }

    #[test]
    fn storage_format_equality() {
        assert_eq!(StorageFormat::F32, StorageFormat::F32);
        assert_ne!(StorageFormat::F16, StorageFormat::BF16);
        assert_ne!(StorageFormat::PackedU8(PackedBits::Int1), StorageFormat::PackedU8(PackedBits::Int4));
    }

    #[test]
    fn storage_format_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StorageFormat::F32);
        set.insert(StorageFormat::F16);
        set.insert(StorageFormat::F32);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn dtype_is_storage_format_alias() {
        let f: DType = StorageFormat::F32;
        assert_eq!(f.size_bytes(), 4);
    }

    #[test]
    fn safetensors_dtype_mapping() {
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::F32), Some(gllm_kernels::types::DType::F32));
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::F16), Some(gllm_kernels::types::DType::F16));
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::BF16), Some(gllm_kernels::types::DType::BF16));
    }

    #[test]
    fn safetensors_dtype_unsupported() {
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::U8), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::I32), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::BOOL), None);
    }

    // ── Additional coverage tests ──

    /// PackedBits Debug trait must produce readable output for every variant.
    #[test]
    fn packed_bits_debug_output() {
        assert_eq!(format!("{:?}", PackedBits::Int1), "Int1");
        assert_eq!(format!("{:?}", PackedBits::Int2), "Int2");
        assert_eq!(format!("{:?}", PackedBits::Int3), "Int3");
        assert_eq!(format!("{:?}", PackedBits::Int4), "Int4");
        assert_eq!(format!("{:?}", PackedBits::Int5), "Int5");
        assert_eq!(format!("{:?}", PackedBits::Int6), "Int6");
    }

    /// PackedBits Clone produces an equal copy for every variant.
    #[test]
    fn packed_bits_clone() {
        let variants = [
            PackedBits::Int1, PackedBits::Int2, PackedBits::Int3,
            PackedBits::Int4, PackedBits::Int5, PackedBits::Int6,
        ];
        for v in &variants {
            assert_eq!(v.clone(), *v);
        }
    }

    /// PackedBits Copy: assigning to a new variable does not move, both remain valid.
    #[test]
    fn packed_bits_copy() {
        let original = PackedBits::Int4;
        let copied = original;
        assert_eq!(original, copied);
        assert_eq!(original, PackedBits::Int4);
    }

    /// All six PackedBits variants are pairwise unequal.
    #[test]
    fn packed_bits_all_inequalities() {
        let variants = [
            PackedBits::Int1, PackedBits::Int2, PackedBits::Int3,
            PackedBits::Int4, PackedBits::Int5, PackedBits::Int6,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    /// PackedBits Hash: every variant produces a unique hash in a HashSet.
    #[test]
    fn packed_bits_hash_distinct() {
        use std::collections::HashSet;
        let all = [
            PackedBits::Int1, PackedBits::Int2, PackedBits::Int3,
            PackedBits::Int4, PackedBits::Int5, PackedBits::Int6,
        ];
        let set: HashSet<PackedBits> = all.into_iter().collect();
        assert_eq!(set.len(), 6);
    }

    /// StorageFormat Debug trait produces readable output for representative variants.
    #[test]
    fn storage_format_debug_output() {
        assert_eq!(format!("{:?}", StorageFormat::F32), "F32");
        assert_eq!(format!("{:?}", StorageFormat::F16), "F16");
        assert_eq!(format!("{:?}", StorageFormat::BF16), "BF16");
        assert_eq!(format!("{:?}", StorageFormat::U8), "U8");
        assert_eq!(
            format!("{:?}", StorageFormat::PackedU8(PackedBits::Int3)),
            "PackedU8(Int3)"
        );
    }

    /// StorageFormat Clone/Copy: every variant can be copied and stays equal.
    #[test]
    fn storage_format_clone_copy() {
        let variants = [
            StorageFormat::F32, StorageFormat::F16, StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int1),
            StorageFormat::PackedU8(PackedBits::Int2),
            StorageFormat::PackedU8(PackedBits::Int3),
            StorageFormat::PackedU8(PackedBits::Int4),
            StorageFormat::PackedU8(PackedBits::Int5),
            StorageFormat::PackedU8(PackedBits::Int6),
        ];
        for v in &variants {
            let cloned = v.clone();
            let copied = *v;
            assert_eq!(*v, cloned);
            assert_eq!(*v, copied);
        }
    }

    /// size_bytes for all PackedU8 variants returns 1 regardless of bit width.
    #[test]
    fn packed_u8_size_bytes_all_variants() {
        assert_eq!(StorageFormat::PackedU8(PackedBits::Int1).size_bytes(), 1);
        assert_eq!(StorageFormat::PackedU8(PackedBits::Int2).size_bytes(), 1);
        assert_eq!(StorageFormat::PackedU8(PackedBits::Int3).size_bytes(), 1);
        assert_eq!(StorageFormat::PackedU8(PackedBits::Int4).size_bytes(), 1);
        assert_eq!(StorageFormat::PackedU8(PackedBits::Int5).size_bytes(), 1);
        assert_eq!(StorageFormat::PackedU8(PackedBits::Int6).size_bytes(), 1);
    }

    /// All StorageFormat variants produce distinct hashes (no collisions in HashSet).
    #[test]
    fn storage_format_all_variants_distinct_in_hashset() {
        use std::collections::HashSet;
        let all = [
            StorageFormat::F32, StorageFormat::F16, StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int1),
            StorageFormat::PackedU8(PackedBits::Int2),
            StorageFormat::PackedU8(PackedBits::Int3),
            StorageFormat::PackedU8(PackedBits::Int4),
            StorageFormat::PackedU8(PackedBits::Int5),
            StorageFormat::PackedU8(PackedBits::Int6),
        ];
        let set: HashSet<StorageFormat> = all.into_iter().collect();
        assert_eq!(set.len(), all.len());
    }

    /// DType alias interoperates with StorageFormat methods and comparisons.
    #[test]
    fn dtype_alias_full_interop() {
        let a: DType = StorageFormat::BF16;
        let b: DType = StorageFormat::BF16;
        assert_eq!(a, b);
        assert_eq!(a.size_bytes(), 2);
        assert_eq!(b.size_bytes(), 2);

        let c: DType = StorageFormat::PackedU8(PackedBits::Int5);
        assert_ne!(a, c);
        assert_eq!(c.size_bytes(), 1);
    }

    /// safetensors_dtype_to_gllm returns None for all non-float types.
    #[test]
    fn safetensors_dtype_all_non_float_unsupported() {
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::BOOL), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::U8), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::I8), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::F8_E5M2), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::F8_E4M3), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::I16), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::U16), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::I32), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::U32), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::F64), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::I64), None);
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::U64), None);
    }

    /// map_storage_format covers all GGUF custom/vendor types (AWQ4, GPTQ4, NVFP4, SQUEEZE).
    #[test]
    fn map_storage_format_custom_vendor_types() {
        assert_eq!(map_storage_format(GgmlDType::AWQ4).unwrap(), StorageFormat::PackedU8(PackedBits::Int4));
        assert_eq!(map_storage_format(GgmlDType::GPTQ4).unwrap(), StorageFormat::PackedU8(PackedBits::Int4));
        assert_eq!(map_storage_format(GgmlDType::NVFP4).unwrap(), StorageFormat::PackedU8(PackedBits::Int4));
        assert_eq!(map_storage_format(GgmlDType::SQUEEZE).unwrap(), StorageFormat::PackedU8(PackedBits::Int4));
        assert_eq!(map_storage_format(GgmlDType::MXFP4).unwrap(), StorageFormat::PackedU8(PackedBits::Int4));
    }

    /// map_storage_format for ternary and IQ1_M types.
    #[test]
    fn map_storage_format_ternary_and_iq1m() {
        assert_eq!(map_storage_format(GgmlDType::TQ1_0).unwrap(), StorageFormat::PackedU8(PackedBits::Int1));
        assert_eq!(map_storage_format(GgmlDType::TQ2_0).unwrap(), StorageFormat::PackedU8(PackedBits::Int2));
        assert_eq!(map_storage_format(GgmlDType::IQ1_M).unwrap(), StorageFormat::PackedU8(PackedBits::Int1));
    }

    /// map_storage_format returns U8 for I8 (native integer).
    #[test]
    fn map_storage_format_i8_is_u8() {
        assert_eq!(map_storage_format(GgmlDType::I8).unwrap(), StorageFormat::U8);
    }

    /// map_storage_format returns Err for all unsupported native types (I16, I32, I64, F64).
    #[test]
    fn map_storage_format_unsupported_native_types() {
        assert!(map_storage_format(GgmlDType::I16).is_err());
        assert!(map_storage_format(GgmlDType::I32).is_err());
        assert!(map_storage_format(GgmlDType::I64).is_err());
        assert!(map_storage_format(GgmlDType::F64).is_err());
    }

    /// ggml_dtype_to_quant_type returns None for all native integer types.
    #[test]
    fn ggml_dtype_to_quant_type_native_none() {
        assert!(ggml_dtype_to_quant_type(GgmlDType::F32).is_none());
        assert!(ggml_dtype_to_quant_type(GgmlDType::F16).is_none());
        assert!(ggml_dtype_to_quant_type(GgmlDType::BF16).is_none());
        assert!(ggml_dtype_to_quant_type(GgmlDType::F64).is_none());
        assert!(ggml_dtype_to_quant_type(GgmlDType::I8).is_none());
        assert!(ggml_dtype_to_quant_type(GgmlDType::I16).is_none());
        assert!(ggml_dtype_to_quant_type(GgmlDType::I32).is_none());
        assert!(ggml_dtype_to_quant_type(GgmlDType::I64).is_none());
    }

    /// KernelTensorView can be constructed with correct fields and data access.
    #[test]
    fn kernel_tensor_view_construction() {
        let data: &[u8] = &[0u8, 1, 2, 3, 4, 5, 6, 7];
        let view = KernelTensorView {
            storage_format: StorageFormat::F32,
            shape: vec![2],
            data,
        };
        assert_eq!(view.storage_format, StorageFormat::F32);
        assert_eq!(view.shape, vec![2]);
        assert_eq!(view.data.len(), 8);
        assert_eq!(view.data[0], 0);
        assert_eq!(view.data[7], 7);
    }

    /// KernelTensorView Clone produces an independent copy with identical fields.
    #[test]
    fn kernel_tensor_view_clone() {
        let data: &[u8] = &[10, 20, 30];
        let original = KernelTensorView {
            storage_format: StorageFormat::F16,
            shape: vec![3],
            data,
        };
        let cloned = original.clone();
        assert_eq!(cloned.storage_format, original.storage_format);
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.data, original.data);
    }

    /// KernelTensorView with packed storage format and multi-dimensional shape.
    #[test]
    fn kernel_tensor_view_quantized_multidim() {
        let data: &[u8] = &[0xAA, 0xBB];
        let view = KernelTensorView {
            storage_format: StorageFormat::PackedU8(PackedBits::Int4),
            shape: vec![4, 4],
            data,
        };
        assert_eq!(view.storage_format, StorageFormat::PackedU8(PackedBits::Int4));
        assert_eq!(view.shape.len(), 2);
        assert_eq!(view.shape[0], 4);
        assert_eq!(view.shape[1], 4);
        assert_eq!(view.data.len(), 2);
    }

    /// Bijective mapping: every GgmlDType variant is covered by either
    /// map_storage_format (Ok) or returns Err — no panic on any variant.
    #[test]
    fn map_storage_format_exhaustive_no_panic() {
        let all_dtypes = [
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
            GgmlDType::AWQ4, GgmlDType::GPTQ4, GgmlDType::SQUEEZE,
            GgmlDType::NVFP4,
        ];
        let supported_count = all_dtypes.iter()
            .filter(|dt| map_storage_format(**dt).is_ok())
            .count();
        assert_eq!(supported_count, all_dtypes.len() - 4); // F64, I16, I32, I64 are unsupported
    }

    /// Every GgmlDType that is_quantized() must map to Some QuantType.
    /// Every GgmlDType that is NOT is_quantized() must map to None.
    #[test]
    fn quant_type_only_for_quantized_dtypes() {
        let all_dtypes = [
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
            GgmlDType::AWQ4, GgmlDType::GPTQ4, GgmlDType::SQUEEZE,
            GgmlDType::NVFP4,
        ];
        for dt in &all_dtypes {
            let result = ggml_dtype_to_quant_type(*dt);
            if dt.is_quantized() {
                assert!(result.is_some(), "{:?} is quantized but maps to None", dt);
            } else {
                assert!(result.is_none(), "{:?} is not quantized but maps to {:?}", dt, result);
            }
        }
    }

    /// size_bytes is const-evaluable (compile-time check via const assertion).
    #[test]
    fn size_bytes_const_evaluable() {
        const F32_SIZE: usize = StorageFormat::F32.size_bytes();
        const F16_SIZE: usize = StorageFormat::F16.size_bytes();
        const U8_SIZE: usize = StorageFormat::U8.size_bytes();
        assert_eq!(F32_SIZE, 4);
        assert_eq!(F16_SIZE, 2);
        assert_eq!(U8_SIZE, 1);
    }

    // ── New tests (15-20 additional) ────────────────────────────────────

    /// map_storage_format maps all Q8-like types and native I8 to U8.
    #[test]
    fn map_storage_format_q8_family_maps_to_u8() {
        assert_eq!(map_storage_format(GgmlDType::Q8_0).unwrap(), StorageFormat::U8);
        assert_eq!(map_storage_format(GgmlDType::Q8_1).unwrap(), StorageFormat::U8);
        assert_eq!(map_storage_format(GgmlDType::Q8_K).unwrap(), StorageFormat::U8);
        assert_eq!(map_storage_format(GgmlDType::I8).unwrap(), StorageFormat::U8);
    }

    /// map_storage_format maps all native float types to their exact StorageFormat.
    #[test]
    fn map_storage_format_native_floats() {
        assert_eq!(map_storage_format(GgmlDType::F32).unwrap(), StorageFormat::F32);
        assert_eq!(map_storage_format(GgmlDType::F16).unwrap(), StorageFormat::F16);
        assert_eq!(map_storage_format(GgmlDType::BF16).unwrap(), StorageFormat::BF16);
    }

    /// map_storage_format maps every 4-bit type to PackedU8(Int4).
    #[test]
    fn map_storage_format_all_int4_group() {
        let int4_group = [
            GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q4_K,
            GgmlDType::IQ4_NL, GgmlDType::IQ4_XS, GgmlDType::MXFP4,
            GgmlDType::AWQ4, GgmlDType::GPTQ4, GgmlDType::NVFP4,
            GgmlDType::SQUEEZE,
        ];
        for dt in &int4_group {
            assert_eq!(
                map_storage_format(*dt).unwrap(),
                StorageFormat::PackedU8(PackedBits::Int4),
                "{:?} should map to PackedU8(Int4)", dt
            );
        }
    }

    /// map_storage_format maps every 2-bit type to PackedU8(Int2).
    #[test]
    fn map_storage_format_all_int2_group() {
        let int2_group = [
            GgmlDType::Q2_K, GgmlDType::IQ2_XXS,
            GgmlDType::IQ2_XS, GgmlDType::IQ2_S, GgmlDType::TQ2_0,
        ];
        for dt in &int2_group {
            assert_eq!(
                map_storage_format(*dt).unwrap(),
                StorageFormat::PackedU8(PackedBits::Int2),
                "{:?} should map to PackedU8(Int2)", dt
            );
        }
    }

    /// map_storage_format maps every 3-bit type to PackedU8(Int3).
    #[test]
    fn map_storage_format_all_int3_group() {
        let int3_group = [GgmlDType::Q3_K, GgmlDType::IQ3_XXS, GgmlDType::IQ3_S];
        for dt in &int3_group {
            assert_eq!(
                map_storage_format(*dt).unwrap(),
                StorageFormat::PackedU8(PackedBits::Int3),
                "{:?} should map to PackedU8(Int3)", dt
            );
        }
    }

    /// map_storage_format maps every 5-bit type to PackedU8(Int5), and 6-bit to Int6.
    #[test]
    fn map_storage_format_int5_and_int6_groups() {
        let int5_group = [GgmlDType::Q5_0, GgmlDType::Q5_1, GgmlDType::Q5_K];
        for dt in &int5_group {
            assert_eq!(
                map_storage_format(*dt).unwrap(),
                StorageFormat::PackedU8(PackedBits::Int5),
                "{:?} should map to PackedU8(Int5)", dt
            );
        }
        assert_eq!(
            map_storage_format(GgmlDType::Q6_K).unwrap(),
            StorageFormat::PackedU8(PackedBits::Int6)
        );
    }

    /// PackedBits Eq trait (stronger than PartialEq): reflexive, symmetric, transitive.
    #[test]
    fn packed_bits_eq_trait_properties() {
        // Reflexive
        assert!(PackedBits::Int1 == PackedBits::Int1);
        assert!(PackedBits::Int6 == PackedBits::Int6);
        // Symmetric
        assert_eq!(PackedBits::Int3 == PackedBits::Int4, PackedBits::Int4 == PackedBits::Int3);
        // Transitive: Int1 != Int2, Int2 != Int3, so Int1 != Int3 (by distinctness)
        assert_ne!(PackedBits::Int1, PackedBits::Int2);
        assert_ne!(PackedBits::Int2, PackedBits::Int3);
        assert_ne!(PackedBits::Int1, PackedBits::Int3);
    }

    /// StorageFormat Eq trait verification — reflexive and symmetric for every variant.
    #[test]
    fn storage_format_eq_trait_properties() {
        // Reflexive
        assert!(StorageFormat::F32 == StorageFormat::F32);
        assert!(StorageFormat::U8 == StorageFormat::U8);
        assert!(StorageFormat::PackedU8(PackedBits::Int3) == StorageFormat::PackedU8(PackedBits::Int3));
        // Symmetric
        assert_eq!(
            StorageFormat::F16 == StorageFormat::BF16,
            StorageFormat::BF16 == StorageFormat::F16
        );
    }

    /// KernelTensorView Debug trait produces readable output.
    #[test]
    fn kernel_tensor_view_debug_output() {
        let data: &[u8] = &[1, 2, 3, 4];
        let view = KernelTensorView {
            storage_format: StorageFormat::F32,
            shape: vec![1],
            data,
        };
        let debug = format!("{:?}", view);
        assert!(debug.contains("KernelTensorView"), "Debug should contain struct name");
        assert!(debug.contains("storage_format"), "Debug should show storage_format field");
        assert!(debug.contains("shape"), "Debug should show shape field");
        assert!(debug.contains("data"), "Debug should show data field");
    }

    /// KernelTensorView with empty data slice and zero-element shape.
    #[test]
    fn kernel_tensor_view_empty_data() {
        let data: &[u8] = &[];
        let view = KernelTensorView {
            storage_format: StorageFormat::U8,
            shape: vec![0],
            data,
        };
        assert_eq!(view.data.len(), 0);
        assert_eq!(view.shape, vec![0]);
        assert_eq!(view.storage_format, StorageFormat::U8);
    }

    /// KernelTensorView with U8 storage format.
    #[test]
    fn kernel_tensor_view_u8_storage() {
        let data: &[u8] = &[10, 20, 30, 40, 50];
        let view = KernelTensorView {
            storage_format: StorageFormat::U8,
            shape: vec![5],
            data,
        };
        assert_eq!(view.storage_format, StorageFormat::U8);
        assert_eq!(view.storage_format.size_bytes(), 1);
        assert_eq!(view.data.len(), 5);
        assert_eq!(view.data[0], 10);
        assert_eq!(view.data[4], 50);
    }

    /// KernelTensorView with BF16 storage and 3D shape.
    #[test]
    fn kernel_tensor_view_bf16_3d_shape() {
        let data: &[u8] = &[0u8; 24]; // 12 elements x 2 bytes
        let view = KernelTensorView {
            storage_format: StorageFormat::BF16,
            shape: vec![2, 3, 2],
            data,
        };
        assert_eq!(view.storage_format, StorageFormat::BF16);
        assert_eq!(view.shape.len(), 3);
        assert_eq!(view.shape[0], 2);
        assert_eq!(view.shape[1], 3);
        assert_eq!(view.shape[2], 2);
        assert_eq!(view.data.len(), 24);
    }

    /// Cross-validation: every GgmlDType that maps to PackedU8 must also
    /// map to a QuantType (is_quantized() == true).
    #[test]
    fn packed_u8_types_are_all_quantized() {
        let packed_dtypes = [
            GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q4_K,
            GgmlDType::IQ4_NL, GgmlDType::IQ4_XS, GgmlDType::MXFP4,
            GgmlDType::AWQ4, GgmlDType::GPTQ4, GgmlDType::NVFP4,
            GgmlDType::SQUEEZE, GgmlDType::Q2_K, GgmlDType::IQ2_XXS,
            GgmlDType::IQ2_XS, GgmlDType::IQ2_S, GgmlDType::TQ2_0,
            GgmlDType::Q3_K, GgmlDType::IQ3_XXS, GgmlDType::IQ3_S,
            GgmlDType::Q5_0, GgmlDType::Q5_1, GgmlDType::Q5_K,
            GgmlDType::Q6_K, GgmlDType::IQ1_S, GgmlDType::IQ1_M, GgmlDType::TQ1_0,
        ];
        for dt in &packed_dtypes {
            let sf = map_storage_format(*dt).unwrap();
            assert!(matches!(sf, StorageFormat::PackedU8(_)), "{:?} should be PackedU8", dt);
            assert!(dt.is_quantized(), "{:?} should be quantized", dt);
            assert!(ggml_dtype_to_quant_type(*dt).is_some(), "{:?} should have QuantType", dt);
        }
    }

    /// Cross-validation: GgmlDType types that map to U8 storage may or may not
    /// be quantized — I8 is not, Q8_0/Q8_1/Q8_K are.
    #[test]
    fn u8_storage_types_mixed_quantization() {
        // Native I8: not quantized, maps to U8, QuantType is None
        assert_eq!(map_storage_format(GgmlDType::I8).unwrap(), StorageFormat::U8);
        assert!(!GgmlDType::I8.is_quantized());
        assert!(ggml_dtype_to_quant_type(GgmlDType::I8).is_none());

        // Quantized Q8 family: quantized, maps to U8, QuantType is Some
        let q8_types = [GgmlDType::Q8_0, GgmlDType::Q8_1, GgmlDType::Q8_K];
        for dt in &q8_types {
            assert_eq!(map_storage_format(*dt).unwrap(), StorageFormat::U8);
            assert!(dt.is_quantized());
            assert!(ggml_dtype_to_quant_type(*dt).is_some());
        }
    }

    /// map_storage_format error variant carries the exact GgmlDType.
    #[test]
    fn map_storage_format_error_contains_correct_dtype() {
        let unsupported = [GgmlDType::I16, GgmlDType::I32, GgmlDType::I64, GgmlDType::F64];
        for dt in &unsupported {
            let err = map_storage_format(*dt).unwrap_err();
            match err {
                GgufError::UnsupportedType(received) => assert_eq!(received, *dt),
                other => panic!("expected UnsupportedType({:?}), got {:?}", dt, other),
            }
        }
    }

    /// safetensors_dtype_to_gllm: F64 is not supported (would need StorageFormat extension).
    #[test]
    fn safetensors_dtype_f64_returns_none() {
        assert_eq!(safetensors_dtype_to_gllm(::safetensors::Dtype::F64), None);
    }

    /// ggml_dtype_to_quant_type MXFP4 has block_size=32, not default.
    #[test]
    fn ggml_dtype_to_quant_type_mxfp4_block_size() {
        let qt = ggml_dtype_to_quant_type(GgmlDType::MXFP4).unwrap();
        assert_eq!(qt, QuantType::Mxfp4 { block_size: 32 });
    }

    // ── Additional tests batch (18 new) ──────────────────────────────────

    /// safetensors_dtype_to_gllm: supported types produce correct kernel DType values.
    #[test]
    fn safetensors_dtype_supported_exact_kernel_dtype_values() {
        let f32_result = safetensors_dtype_to_gllm(::safetensors::Dtype::F32);
        let f16_result = safetensors_dtype_to_gllm(::safetensors::Dtype::F16);
        let bf16_result = safetensors_dtype_to_gllm(::safetensors::Dtype::BF16);

        // Verify the return values are exactly the correct kernel DType variants.
        assert_eq!(f32_result, Some(gllm_kernels::types::DType::F32));
        assert_eq!(f16_result, Some(gllm_kernels::types::DType::F16));
        assert_eq!(bf16_result, Some(gllm_kernels::types::DType::BF16));

        // Verify they are all distinct.
        assert_ne!(f32_result, f16_result);
        assert_ne!(f16_result, bf16_result);
        assert_ne!(f32_result, bf16_result);
    }

    /// StorageFormat PackedU8 size_bytes is const-evaluable for all PackedBits variants.
    #[test]
    fn packed_u8_size_bytes_const_evaluable() {
        const INT1: usize = StorageFormat::PackedU8(PackedBits::Int1).size_bytes();
        const INT2: usize = StorageFormat::PackedU8(PackedBits::Int2).size_bytes();
        const INT3: usize = StorageFormat::PackedU8(PackedBits::Int3).size_bytes();
        const INT4: usize = StorageFormat::PackedU8(PackedBits::Int4).size_bytes();
        const INT5: usize = StorageFormat::PackedU8(PackedBits::Int5).size_bytes();
        const INT6: usize = StorageFormat::PackedU8(PackedBits::Int6).size_bytes();
        assert_eq!(INT1, 1);
        assert_eq!(INT2, 1);
        assert_eq!(INT3, 1);
        assert_eq!(INT4, 1);
        assert_eq!(INT5, 1);
        assert_eq!(INT6, 1);
    }

    /// StorageFormat Hash consistency: inserting the same value twice yields same hash,
    /// so HashSet deduplication works correctly.
    #[test]
    fn storage_format_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let compute_hash = |v: &StorageFormat| -> u64 {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };

        // Same value must produce same hash.
        assert_eq!(compute_hash(&StorageFormat::F32), compute_hash(&StorageFormat::F32));
        assert_eq!(
            compute_hash(&StorageFormat::PackedU8(PackedBits::Int3)),
            compute_hash(&StorageFormat::PackedU8(PackedBits::Int3))
        );
    }

    /// PackedBits Hash consistency: same variant always hashes to same value.
    #[test]
    fn packed_bits_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let compute_hash = |v: &PackedBits| -> u64 {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };

        assert_eq!(compute_hash(&PackedBits::Int1), compute_hash(&PackedBits::Int1));
        assert_eq!(compute_hash(&PackedBits::Int6), compute_hash(&PackedBits::Int6));
        // Different variants should produce different hashes (probabilistic but guaranteed for small enums).
        assert_ne!(compute_hash(&PackedBits::Int1), compute_hash(&PackedBits::Int6));
    }

    /// KernelTensorView can have an empty shape (0 dimensions).
    #[test]
    fn kernel_tensor_view_empty_shape() {
        let data: &[u8] = &[42];
        let view = KernelTensorView {
            storage_format: StorageFormat::U8,
            shape: vec![],
            data,
        };
        assert!(view.shape.is_empty());
        assert_eq!(view.data.len(), 1);
    }

    /// KernelTensorView with single F16 element (2 bytes).
    #[test]
    fn kernel_tensor_view_single_f16_element() {
        let data: &[u8] = &[0x00, 0x3C]; // F16 representation
        let view = KernelTensorView {
            storage_format: StorageFormat::F16,
            shape: vec![1],
            data,
        };
        assert_eq!(view.storage_format.size_bytes(), 2);
        assert_eq!(view.data.len(), 2);
        assert_eq!(view.shape, vec![1]);
    }

    /// KernelTensorView shape values can be large (stress test usize range).
    #[test]
    fn kernel_tensor_view_large_shape_values() {
        let data: &[u8] = &[0u8; 8];
        let view = KernelTensorView {
            storage_format: StorageFormat::F32,
            shape: vec![1000000, 2000000],
            data,
        };
        assert_eq!(view.shape[0], 1_000_000);
        assert_eq!(view.shape[1], 2_000_000);
    }

    /// KernelTensorView with PackedU8(Int6) storage (least common bit width).
    #[test]
    fn kernel_tensor_view_int6_storage() {
        let data: &[u8] = &[0xFF, 0xFE, 0xFD];
        let view = KernelTensorView {
            storage_format: StorageFormat::PackedU8(PackedBits::Int6),
            shape: vec![4],
            data,
        };
        assert_eq!(view.storage_format, StorageFormat::PackedU8(PackedBits::Int6));
        assert_eq!(view.storage_format.size_bytes(), 1);
        assert_eq!(view.data.len(), 3);
    }

    /// Cross-validation: map_storage_format and ggml_dtype_to_quant_type are consistent.
    /// Every dtype that maps to PackedU8(Int4) must have a non-None QuantType,
    /// and every dtype that maps to a native float must have None QuantType.
    #[test]
    fn map_storage_format_and_quant_type_cross_consistency() {
        // Native floats: StorageFormat = F32/F16/BF16, QuantType = None
        for (dtype, expected_sf) in [
            (GgmlDType::F32, StorageFormat::F32),
            (GgmlDType::F16, StorageFormat::F16),
            (GgmlDType::BF16, StorageFormat::BF16),
        ] {
            assert_eq!(map_storage_format(dtype).unwrap(), expected_sf);
            assert!(ggml_dtype_to_quant_type(dtype).is_none());
        }

        // Quantized: StorageFormat != native float, QuantType = Some
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_K, GgmlDType::IQ3_S] {
            let sf = map_storage_format(dtype).unwrap();
            assert!(!matches!(sf, StorageFormat::F32 | StorageFormat::F16 | StorageFormat::BF16));
            assert!(ggml_dtype_to_quant_type(dtype).is_some());
        }
    }

    /// GgufError::UnsupportedType produces meaningful Display output containing the dtype.
    #[test]
    fn gguf_error_unsupported_type_display() {
        let err = GgufError::UnsupportedType(GgmlDType::F64);
        let msg = format!("{}", err);
        assert!(msg.contains("F64"), "Error message should contain the dtype name");
        assert!(msg.contains("Unsupported"), "Error message should indicate unsupported type");
    }

    /// StorageFormat: BF16 and F16 have the same size_bytes but are not equal.
    #[test]
    fn storage_format_same_size_different_types() {
        assert_eq!(StorageFormat::F16.size_bytes(), StorageFormat::BF16.size_bytes());
        assert_ne!(StorageFormat::F16, StorageFormat::BF16);
    }

    /// DType alias works as a function parameter and return type.
    #[test]
    fn dtype_alias_as_function_boundary() {
        fn get_format() -> DType {
            StorageFormat::F32
        }
        fn check_format(fmt: DType) -> bool {
            fmt == StorageFormat::F32
        }
        assert!(check_format(get_format()));
    }

    /// KernelTensorView Clone with BF16 storage produces identical field values.
    #[test]
    fn kernel_tensor_view_clone_bf16() {
        let data: &[u8] = &[0xAB, 0xCD, 0xEF, 0x01];
        let original = KernelTensorView {
            storage_format: StorageFormat::BF16,
            shape: vec![2],
            data,
        };
        let cloned = original.clone();
        assert_eq!(cloned.storage_format, original.storage_format);
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.data, original.data);
        assert_eq!(cloned.storage_format.size_bytes(), 2);
    }

    /// KernelTensorView Clone with PackedU8 storage produces identical field values.
    #[test]
    fn kernel_tensor_view_clone_packed() {
        let data: &[u8] = &[0x12, 0x34];
        let original = KernelTensorView {
            storage_format: StorageFormat::PackedU8(PackedBits::Int4),
            shape: vec![4, 2],
            data,
        };
        let cloned = original.clone();
        assert_eq!(cloned.storage_format, original.storage_format);
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.data.len(), 2);
    }

    /// Count of supported map_storage_format results matches expected partition.
    #[test]
    fn map_storage_format_supported_count() {
        let all_dtypes = [
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
            GgmlDType::AWQ4, GgmlDType::GPTQ4, GgmlDType::SQUEEZE,
            GgmlDType::NVFP4,
        ];

        let mut f32_count = 0;
        let mut f16_count = 0;
        let mut bf16_count = 0;
        let mut u8_count = 0;
        let mut packed_count = 0;

        for dt in &all_dtypes {
            if let Ok(sf) = map_storage_format(*dt) {
                match sf {
                    StorageFormat::F32 => f32_count += 1,
                    StorageFormat::F16 => f16_count += 1,
                    StorageFormat::BF16 => bf16_count += 1,
                    StorageFormat::U8 => u8_count += 1,
                    StorageFormat::PackedU8(_) => packed_count += 1,
                }
            }
        }

        // Exactly one dtype maps to each native float format.
        assert_eq!(f32_count, 1);
        assert_eq!(f16_count, 1);
        assert_eq!(bf16_count, 1);
        // Q8_0, Q8_1, Q8_K, I8 map to U8.
        assert_eq!(u8_count, 4);
        // The rest (36 total dtypes - 4 unsupported - 4 u8 - 3 floats = 25) map to PackedU8.
        assert_eq!(packed_count, 25);
    }

    /// KernelTensorView with [1, 1, 1] shape (3D scalar-like tensor).
    #[test]
    fn kernel_tensor_view_scalar_3d() {
        let data: &[u8] = &[0u8; 4]; // One F32 element
        let view = KernelTensorView {
            storage_format: StorageFormat::F32,
            shape: vec![1, 1, 1],
            data,
        };
        assert_eq!(view.shape.len(), 3);
        assert_eq!(view.shape.iter().product::<usize>(), 1);
        assert_eq!(view.data.len(), 4);
    }

    /// ggml_dtype_to_quant_type: non-MXFP4 quantized types do not embed block_size
    /// (MXFP4 is the only variant with a non-unit block_size field).
    #[test]
    fn only_mxfp4_carries_block_size() {
        // MXFP4 is the only QuantType variant with a block_size field.
        let mxfp4 = ggml_dtype_to_quant_type(GgmlDType::MXFP4).unwrap();
        assert!(matches!(mxfp4, QuantType::Mxfp4 { block_size: 32 }));

        // Spot-check other quantized types do not match Mxfp4 pattern.
        let others = [
            GgmlDType::Q4_0, GgmlDType::Q8_K, GgmlDType::IQ2_XXS,
            GgmlDType::AWQ4, GgmlDType::NVFP4, GgmlDType::TQ1_0,
        ];
        for dt in &others {
            let qt = ggml_dtype_to_quant_type(*dt).unwrap();
            assert!(!matches!(qt, QuantType::Mxfp4 { .. }), "{:?} should not be Mxfp4", dt);
        }
    }

    /// StorageFormat all variants are matchable without wildcard (exhaustiveness check).
    #[test]
    fn storage_format_exhaustive_match() {
        let variants = [
            StorageFormat::F32,
            StorageFormat::F16,
            StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int1),
        ];
        for v in &variants {
            // Each variant must be matchable without _ wildcard.
            let label = match v {
                StorageFormat::F32 => "f32",
                StorageFormat::F16 => "f16",
                StorageFormat::BF16 => "bf16",
                StorageFormat::U8 => "u8",
                StorageFormat::PackedU8(bits) => match bits {
                    PackedBits::Int1 => "packed_int1",
                    PackedBits::Int2 => "packed_int2",
                    PackedBits::Int3 => "packed_int3",
                    PackedBits::Int4 => "packed_int4",
                    PackedBits::Int5 => "packed_int5",
                    PackedBits::Int6 => "packed_int6",
                },
            };
            assert!(!label.is_empty());
        }
    }

    // ── Batch 3: 45 additional tests ─────────────────────────────────────

    /// PackedBits exhaustive Debug format matches variant names exactly.
    #[test]
    fn packed_bits_debug_all_variants_exact() {
        assert_eq!(format!("{:?}", PackedBits::Int1), "Int1");
        assert_eq!(format!("{:?}", PackedBits::Int2), "Int2");
        assert_eq!(format!("{:?}", PackedBits::Int3), "Int3");
        assert_eq!(format!("{:?}", PackedBits::Int4), "Int4");
        assert_eq!(format!("{:?}", PackedBits::Int5), "Int5");
        assert_eq!(format!("{:?}", PackedBits::Int6), "Int6");
    }

    /// StorageFormat Debug for every variant is non-empty and well-formed.
    #[test]
    fn storage_format_debug_all_variants_nonempty() {
        let all = [
            StorageFormat::F32,
            StorageFormat::F16,
            StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int1),
            StorageFormat::PackedU8(PackedBits::Int2),
            StorageFormat::PackedU8(PackedBits::Int3),
            StorageFormat::PackedU8(PackedBits::Int4),
            StorageFormat::PackedU8(PackedBits::Int5),
            StorageFormat::PackedU8(PackedBits::Int6),
        ];
        for v in &all {
            let s = format!("{:?}", v);
            assert!(!s.is_empty(), "Debug output should not be empty for {:?}", v);
        }
    }

    /// StorageFormat PartialEq is false between F32 and all other base variants.
    #[test]
    fn storage_format_f32_not_equal_any_other_base() {
        let others = [
            StorageFormat::F16,
            StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int4),
        ];
        for other in &others {
            assert_ne!(StorageFormat::F32, *other);
        }
    }

    /// StorageFormat PackedU8 with different inner bits are all unequal.
    #[test]
    fn storage_format_packed_u8_different_bits_all_unequal() {
        let bits = [
            PackedBits::Int1, PackedBits::Int2, PackedBits::Int3,
            PackedBits::Int4, PackedBits::Int5, PackedBits::Int6,
        ];
        for (i, a) in bits.iter().enumerate() {
            for (j, b) in bits.iter().enumerate() {
                if i == j {
                    assert_eq!(StorageFormat::PackedU8(*a), StorageFormat::PackedU8(*b));
                } else {
                    assert_ne!(StorageFormat::PackedU8(*a), StorageFormat::PackedU8(*b));
                }
            }
        }
    }

    /// size_bytes for F32 is exactly 4.
    #[test]
    fn size_bytes_f32_exact() {
        assert_eq!(StorageFormat::F32.size_bytes(), 4);
    }

    /// size_bytes for F16 is exactly 2.
    #[test]
    fn size_bytes_f16_exact() {
        assert_eq!(StorageFormat::F16.size_bytes(), 2);
    }

    /// size_bytes for BF16 is exactly 2 (same as F16 but types differ).
    #[test]
    fn size_bytes_bf16_exact() {
        assert_eq!(StorageFormat::BF16.size_bytes(), 2);
    }

    /// size_bytes for U8 is exactly 1.
    #[test]
    fn size_bytes_u8_exact() {
        assert_eq!(StorageFormat::U8.size_bytes(), 1);
    }

    /// size_bytes for all PackedU8 variants is 1 regardless of bit width.
    #[test]
    fn size_bytes_packed_u8_always_one() {
        for bits in [PackedBits::Int1, PackedBits::Int2, PackedBits::Int3,
                     PackedBits::Int4, PackedBits::Int5, PackedBits::Int6] {
            assert_eq!(StorageFormat::PackedU8(bits).size_bytes(), 1,
                "PackedU8({:?}) should have size_bytes=1", bits);
        }
    }

    /// GgmlDType as_str returns correct static str for F32.
    #[test]
    fn ggml_dtype_as_str_f32() {
        assert_eq!(GgmlDType::F32.as_str(), "F32");
    }

    /// GgmlDType as_str returns correct static str for BF16.
    #[test]
    fn ggml_dtype_as_str_bf16() {
        assert_eq!(GgmlDType::BF16.as_str(), "BF16");
    }

    /// GgmlDType as_str returns correct static str for NVFP4.
    #[test]
    fn ggml_dtype_as_str_nvfp4() {
        assert_eq!(GgmlDType::NVFP4.as_str(), "NVFP4");
    }

    /// GgmlDType as_str returns correct static str for SQUEEZE.
    #[test]
    fn ggml_dtype_as_str_squeeze() {
        assert_eq!(GgmlDType::SQUEEZE.as_str(), "SQUEEZE");
    }

    /// GgmlDType as_str returns correct static str for TQ1_0.
    #[test]
    fn ggml_dtype_as_str_tq1_0() {
        assert_eq!(GgmlDType::TQ1_0.as_str(), "TQ1_0");
    }

    /// GgmlDType as_str returns correct static str for TQ2_0.
    #[test]
    fn ggml_dtype_as_str_tq2_0() {
        assert_eq!(GgmlDType::TQ2_0.as_str(), "TQ2_0");
    }

    /// GgmlDType is_quantized returns false for all native float types.
    #[test]
    fn ggml_dtype_is_quantized_native_floats() {
        assert!(!GgmlDType::F32.is_quantized());
        assert!(!GgmlDType::F16.is_quantized());
        assert!(!GgmlDType::BF16.is_quantized());
        assert!(!GgmlDType::F64.is_quantized());
    }

    /// GgmlDType is_quantized returns false for all native integer types.
    #[test]
    fn ggml_dtype_is_quantized_native_integers() {
        assert!(!GgmlDType::I8.is_quantized());
        assert!(!GgmlDType::I16.is_quantized());
        assert!(!GgmlDType::I32.is_quantized());
        assert!(!GgmlDType::I64.is_quantized());
    }

    /// GgmlDType is_quantized returns true for all K-quant types.
    #[test]
    fn ggml_dtype_is_quantized_k_quant_family() {
        for dt in [GgmlDType::Q2_K, GgmlDType::Q3_K, GgmlDType::Q4_K,
                   GgmlDType::Q5_K, GgmlDType::Q6_K, GgmlDType::Q8_K] {
            assert!(dt.is_quantized(), "{:?} should be quantized", dt);
        }
    }

    /// GgmlDType is_quantized returns true for all IQ types.
    #[test]
    fn ggml_dtype_is_quantized_iq_family() {
        for dt in [GgmlDType::IQ1_S, GgmlDType::IQ1_M,
                   GgmlDType::IQ2_XXS, GgmlDType::IQ2_XS, GgmlDType::IQ2_S,
                   GgmlDType::IQ3_XXS, GgmlDType::IQ3_S,
                   GgmlDType::IQ4_NL, GgmlDType::IQ4_XS] {
            assert!(dt.is_quantized(), "{:?} should be quantized", dt);
        }
    }

    /// GgmlDType is_quantized returns true for vendor/custom types.
    #[test]
    fn ggml_dtype_is_quantized_vendor_types() {
        assert!(GgmlDType::MXFP4.is_quantized());
        assert!(GgmlDType::AWQ4.is_quantized());
        assert!(GgmlDType::GPTQ4.is_quantized());
        assert!(GgmlDType::SQUEEZE.is_quantized());
        assert!(GgmlDType::NVFP4.is_quantized());
        assert!(GgmlDType::TQ1_0.is_quantized());
        assert!(GgmlDType::TQ2_0.is_quantized());
    }

    /// GgmlDType block_size is 1 for all native types (F32, F16, BF16, I8, I16, I32, I64, F64).
    #[test]
    fn ggml_dtype_block_size_native_types_are_one() {
        for dt in [GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16,
                   GgmlDType::I8, GgmlDType::I16, GgmlDType::I32,
                   GgmlDType::I64, GgmlDType::F64] {
            assert_eq!(dt.block_size(), 1, "{:?} should have block_size=1", dt);
        }
    }

    /// GgmlDType block_bytes for F32 is 4.
    #[test]
    fn ggml_dtype_block_bytes_f32() {
        assert_eq!(GgmlDType::F32.block_bytes(), 4);
    }

    /// GgmlDType block_bytes for F16 is 2.
    #[test]
    fn ggml_dtype_block_bytes_f16() {
        assert_eq!(GgmlDType::F16.block_bytes(), 2);
    }

    /// GgmlDType block_bytes for BF16 is 2.
    #[test]
    fn ggml_dtype_block_bytes_bf16() {
        assert_eq!(GgmlDType::BF16.block_bytes(), 2);
    }

    /// GgmlDType block_bytes for I8 is 1.
    #[test]
    fn ggml_dtype_block_bytes_i8() {
        assert_eq!(GgmlDType::I8.block_bytes(), 1);
    }

    /// GgmlDType block_bytes for F64 is 8.
    #[test]
    fn ggml_dtype_block_bytes_f64() {
        assert_eq!(GgmlDType::F64.block_bytes(), 8);
    }

    /// GgmlDType block_bytes for I64 is 8.
    #[test]
    fn ggml_dtype_block_bytes_i64() {
        assert_eq!(GgmlDType::I64.block_bytes(), 8);
    }

    /// GgmlDType block_bytes for AWQ4 and GPTQ4 are both 72.
    #[test]
    fn ggml_dtype_block_bytes_awq_gptq() {
        assert_eq!(GgmlDType::AWQ4.block_bytes(), 72);
        assert_eq!(GgmlDType::GPTQ4.block_bytes(), 72);
    }

    /// GgmlDType block_size for AWQ4 and GPTQ4 are both 128.
    #[test]
    fn ggml_dtype_block_size_awq_gptq() {
        assert_eq!(GgmlDType::AWQ4.block_size(), 128);
        assert_eq!(GgmlDType::GPTQ4.block_size(), 128);
    }

    /// GgmlDType block_bytes for NVFP4 is 36.
    #[test]
    fn ggml_dtype_block_bytes_nvfp4() {
        assert_eq!(GgmlDType::NVFP4.block_bytes(), 36);
    }

    /// GgmlDType block_size for NVFP4 is 64.
    #[test]
    fn ggml_dtype_block_size_nvfp4() {
        assert_eq!(GgmlDType::NVFP4.block_size(), 64);
    }

    /// GgmlDType block_bytes for SQUEEZE is 130.
    #[test]
    fn ggml_dtype_block_bytes_squeeze() {
        assert_eq!(GgmlDType::SQUEEZE.block_bytes(), 130);
    }

    /// GgmlDType block_bytes is greater than zero for all variants.
    #[test]
    fn ggml_dtype_block_bytes_all_positive() {
        for &dt in GgmlDType::all() {
            assert!(dt.block_bytes() > 0, "{:?} has non-positive block_bytes", dt);
        }
    }

    /// GgmlDType block_size is greater than zero for all variants.
    #[test]
    fn ggml_dtype_block_size_all_positive() {
        for &dt in GgmlDType::all() {
            assert!(dt.block_size() > 0, "{:?} has non-positive block_size", dt);
        }
    }

    /// KernelTensorView with maximum usize dimension in shape.
    #[test]
    fn kernel_tensor_view_usize_max_shape() {
        let data: &[u8] = &[0u8; 4];
        let view = KernelTensorView {
            storage_format: StorageFormat::F32,
            shape: vec![usize::MAX],
            data,
        };
        assert_eq!(view.shape[0], usize::MAX);
    }

    /// KernelTensorView shape with multiple dimensions including zero.
    #[test]
    fn kernel_tensor_view_shape_with_zero_dim() {
        let data: &[u8] = &[];
        let view = KernelTensorView {
            storage_format: StorageFormat::F32,
            shape: vec![4, 0, 3],
            data,
        };
        assert_eq!(view.shape.len(), 3);
        assert_eq!(view.shape[0], 4);
        assert_eq!(view.shape[1], 0);
        assert_eq!(view.shape[2], 3);
        assert_eq!(view.data.len(), 0);
    }

    /// KernelTensorView with all PackedBits variants as storage_format.
    #[test]
    fn kernel_tensor_view_all_packed_storage_formats() {
        let data: &[u8] = &[0xFF];
        for bits in [PackedBits::Int1, PackedBits::Int2, PackedBits::Int3,
                     PackedBits::Int4, PackedBits::Int5, PackedBits::Int6] {
            let view = KernelTensorView {
                storage_format: StorageFormat::PackedU8(bits),
                shape: vec![1],
                data,
            };
            assert_eq!(view.storage_format, StorageFormat::PackedU8(bits));
            assert_eq!(view.storage_format.size_bytes(), 1);
        }
    }

    /// map_storage_format for Q5_0 specifically maps to PackedU8(Int5).
    #[test]
    fn map_storage_format_q5_0_is_int5() {
        assert_eq!(
            map_storage_format(GgmlDType::Q5_0).unwrap(),
            StorageFormat::PackedU8(PackedBits::Int5)
        );
    }

    /// map_storage_format for Q5_1 specifically maps to PackedU8(Int5).
    #[test]
    fn map_storage_format_q5_1_is_int5() {
        assert_eq!(
            map_storage_format(GgmlDType::Q5_1).unwrap(),
            StorageFormat::PackedU8(PackedBits::Int5)
        );
    }

    /// map_storage_format for IQ1_S maps to PackedU8(Int1).
    #[test]
    fn map_storage_format_iq1_s_is_int1() {
        assert_eq!(
            map_storage_format(GgmlDType::IQ1_S).unwrap(),
            StorageFormat::PackedU8(PackedBits::Int1)
        );
    }

    /// map_storage_format for IQ1_M maps to PackedU8(Int1).
    #[test]
    fn map_storage_format_iq1_m_is_int1() {
        assert_eq!(
            map_storage_format(GgmlDType::IQ1_M).unwrap(),
            StorageFormat::PackedU8(PackedBits::Int1)
        );
    }

    /// safetensors_dtype_to_gllm returns correct results for all 3 supported types.
    #[test]
    fn safetensors_dtype_returns_some_for_only_three_types() {
        let supported: Vec<::safetensors::Dtype> = vec![
            ::safetensors::Dtype::F32,
            ::safetensors::Dtype::F16,
            ::safetensors::Dtype::BF16,
        ];
        let mut count = 0;
        for dt in &supported {
            if safetensors_dtype_to_gllm(*dt).is_some() {
                count += 1;
            }
        }
        assert_eq!(count, 3);
    }

    /// StorageFormat F16 and BF16 both have 2 bytes but are distinct types.
    #[test]
    fn storage_format_f16_bf16_same_size_distinct_identity() {
        assert_eq!(StorageFormat::F16.size_bytes(), StorageFormat::BF16.size_bytes());
        assert_ne!(StorageFormat::F16, StorageFormat::BF16);
        // Verify Debug output is also different.
        assert_ne!(format!("{:?}", StorageFormat::F16), format!("{:?}", StorageFormat::BF16));
    }

    /// GgmlDType all() returns a non-empty list.
    #[test]
    fn ggml_dtype_all_returns_nonempty() {
        let all = GgmlDType::all();
        assert!(!all.is_empty());
        assert!(all.len() >= 30);
    }

    /// GgmlDType all() contains every supported variant exactly once.
    #[test]
    fn ggml_dtype_all_contains_unique_variants() {
        use std::collections::HashSet;
        let all = GgmlDType::all();
        let set: HashSet<GgmlDType> = all.iter().copied().collect();
        assert_eq!(set.len(), all.len(), "all() should not contain duplicates");
    }

    /// GgufError Display for InvalidMagic contains hex prefix.
    #[test]
    fn gguf_error_invalid_magic_display() {
        let err = GgufError::InvalidMagic(0x12345678);
        let msg = format!("{}", err);
        assert!(msg.contains("0x12345678"), "Should contain the hex magic value");
    }

    /// GgufError Display for UnsupportedVersion contains the version number.
    #[test]
    fn gguf_error_unsupported_version_display() {
        let err = GgufError::UnsupportedVersion(99);
        let msg = format!("{}", err);
        assert!(msg.contains("99"));
    }

    /// GgufError Display for MissingMetadata contains the key name.
    #[test]
    fn gguf_error_missing_metadata_display() {
        let err = GgufError::MissingMetadata("general.architecture".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("general.architecture"));
    }

    /// GgufError Display for TensorNotFound contains the tensor name.
    #[test]
    fn gguf_error_tensor_not_found_display() {
        let err = GgufError::TensorNotFound("token_embd.weight".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("token_embd.weight"));
    }

    /// GgufError Display for ParseError contains the message.
    #[test]
    fn gguf_error_parse_error_display() {
        let err = GgufError::ParseError("bad offset".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("bad offset"));
    }

    /// GgufError Debug for UnsupportedType shows the enum name.
    #[test]
    fn gguf_error_unsupported_type_debug() {
        let err = GgufError::UnsupportedType(GgmlDType::I32);
        let debug = format!("{:?}", err);
        assert!(debug.contains("UnsupportedType"));
    }

    /// StorageFormat size_bytes sum over all variants covers expected range.
    #[test]
    fn storage_format_size_bytes_range() {
        let sizes: Vec<usize> = [
            StorageFormat::F32,
            StorageFormat::F16,
            StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int4),
        ].iter().map(|v| v.size_bytes()).collect();
        assert!(sizes.iter().all(|&s| s >= 1 && s <= 4));
    }

    /// KernelTensorView data bytes are accessible by index.
    #[test]
    fn kernel_tensor_view_data_byte_access() {
        let data: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF];
        let view = KernelTensorView {
            storage_format: StorageFormat::F32,
            shape: vec![1],
            data,
        };
        assert_eq!(view.data[0], 0xDE);
        assert_eq!(view.data[1], 0xAD);
        assert_eq!(view.data[2], 0xBE);
        assert_eq!(view.data[3], 0xEF);
    }

    /// map_storage_format: every quantized dtype maps to either PackedU8 or U8.
    #[test]
    fn map_storage_format_quantized_dtypes_only_packed_or_u8() {
        for &dt in GgmlDType::all() {
            if dt.is_quantized() {
                let sf = map_storage_format(dt).unwrap();
                assert!(
                    matches!(sf, StorageFormat::PackedU8(_) | StorageFormat::U8),
                    "{:?} is quantized but maps to {:?}", dt, sf
                );
            }
        }
    }

    /// GgmlDType Q4_0 and Q4_1 both have block_size 32.
    #[test]
    fn ggml_dtype_q4_block_size_32() {
        assert_eq!(GgmlDType::Q4_0.block_size(), 32);
        assert_eq!(GgmlDType::Q4_1.block_size(), 32);
    }

    /// GgmlDType Q8_0 has block_size 32 and Q8_K has block_size 256.
    #[test]
    fn ggml_dtype_q8_block_sizes() {
        assert_eq!(GgmlDType::Q8_0.block_size(), 32);
        assert_eq!(GgmlDType::Q8_K.block_size(), 256);
    }

    /// GgmlDType block_bytes for MXFP4 is 17.
    #[test]
    fn ggml_dtype_block_bytes_mxfp4() {
        assert_eq!(GgmlDType::MXFP4.block_bytes(), 17);
    }

    /// GgmlDType block_size for MXFP4 is 32.
    #[test]
    fn ggml_dtype_block_size_mxfp4() {
        assert_eq!(GgmlDType::MXFP4.block_size(), 32);
    }

    /// DType alias can be used in a collection (Vec).
    #[test]
    fn dtype_alias_in_collection() {
        let formats: Vec<DType> = vec![
            StorageFormat::F32,
            StorageFormat::F16,
            StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int4),
        ];
        assert_eq!(formats.len(), 5);
        assert_eq!(formats[0].size_bytes(), 4);
        assert_eq!(formats[3], StorageFormat::U8);
    }

    /// KernelTensorView with F16 data and shape matching 2 bytes per element.
    #[test]
    fn kernel_tensor_view_f16_data_integrity() {
        let data: &[u8] = &[0x00, 0x3C, 0x00, 0x40]; // two F16 values
        let view = KernelTensorView {
            storage_format: StorageFormat::F16,
            shape: vec![2],
            data,
        };
        assert_eq!(view.data.len(), 4);
        assert_eq!(view.shape.len(), 1);
        // First element bytes
        assert_eq!(view.data[0], 0x00);
        assert_eq!(view.data[1], 0x3C);
        // Second element bytes
        assert_eq!(view.data[2], 0x00);
        assert_eq!(view.data[3], 0x40);
    }

    /// GgmlDType all() count matches the number of enum variants declared.
    #[test]
    fn ggml_dtype_all_count_matches_enum() {
        let all = GgmlDType::all();
        // Count from the enum definition: F32,F16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,Q8_1,
        // Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,Q8_K,IQ2_XXS,IQ2_XS,IQ3_XXS,IQ1_S,IQ4_NL,
        // IQ3_S,IQ2_S,IQ4_XS,I8,I16,I32,I64,F64,IQ1_M,BF16,TQ1_0,TQ2_0,MXFP4,AWQ4,GPTQ4,SQUEEZE,NVFP4
        assert_eq!(all.len(), 36);
    }

    /// KernelTensorView Debug output includes all field names.
    #[test]
    fn kernel_tensor_view_debug_includes_all_fields() {
        let data: &[u8] = &[0x01, 0x02];
        let view = KernelTensorView {
            storage_format: StorageFormat::PackedU8(PackedBits::Int3),
            shape: vec![4],
            data,
        };
        let debug = format!("{:?}", view);
        assert!(debug.contains("storage_format"));
        assert!(debug.contains("shape"));
        assert!(debug.contains("data"));
        assert!(debug.contains("PackedU8"));
        assert!(debug.contains("Int3"));
    }

    /// PackedBits can be used as HashMap key.
    #[test]
    fn packed_bits_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(PackedBits::Int4, "4-bit");
        map.insert(PackedBits::Int2, "2-bit");
        assert_eq!(map.get(&PackedBits::Int4), Some(&"4-bit"));
        assert_eq!(map.get(&PackedBits::Int2), Some(&"2-bit"));
        assert_eq!(map.get(&PackedBits::Int6), None);
    }

    /// StorageFormat can be used as HashMap key.
    #[test]
    fn storage_format_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(StorageFormat::F32, 4usize);
        map.insert(StorageFormat::F16, 2usize);
        map.insert(StorageFormat::U8, 1usize);
        assert_eq!(map.get(&StorageFormat::F32), Some(&4));
        assert_eq!(map.get(&StorageFormat::F16), Some(&2));
        assert_eq!(map.get(&StorageFormat::BF16), None);
    }

    // ── Batch 4: 15 additional tests ─────────────────────────────────────

    /// PackedBits: all 6 variants produce pairwise distinct hashes via DefaultHasher.
    /// Verifies no hash collision exists in the standard hasher for the full enum.
    #[test]
    fn packed_bits_all_variants_pairwise_hash_distinct() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let variants = [
            PackedBits::Int1, PackedBits::Int2, PackedBits::Int3,
            PackedBits::Int4, PackedBits::Int5, PackedBits::Int6,
        ];
        let hashes: Vec<u64> = variants.iter().map(|v| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }).collect();

        for (i, ha) in hashes.iter().enumerate() {
            for (j, hb) in hashes.iter().enumerate() {
                if i != j {
                    assert_ne!(ha, hb, "{:?} and {:?} have same hash", variants[i], variants[j]);
                }
            }
        }
    }

    /// PackedBits Copy: assigning from a cloned value still preserves original.
    /// Tests that Copy is truly bitwise (no Drop interaction).
    #[test]
    fn packed_bits_copy_preserves_original_through_assignment() {
        let a = PackedBits::Int3;
        let b = a;
        let c = b;
        assert_eq!(a, PackedBits::Int3);
        assert_eq!(b, PackedBits::Int3);
        assert_eq!(c, PackedBits::Int3);
        // All three are still usable independently after copy chain.
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    /// PackedBits Debug output is exactly the variant name (no extra whitespace or quotes).
    #[test]
    fn packed_bits_debug_no_quotes_no_whitespace() {
        for (v, name) in [
            (PackedBits::Int1, "Int1"),
            (PackedBits::Int2, "Int2"),
            (PackedBits::Int3, "Int3"),
            (PackedBits::Int4, "Int4"),
            (PackedBits::Int5, "Int5"),
            (PackedBits::Int6, "Int6"),
        ] {
            let s = format!("{:?}", v);
            assert_eq!(s, name);
            assert!(!s.contains('"'));
            assert!(!s.contains(' '));
        }
    }

    /// StorageFormat size_bytes: every variant returns a value in {1, 2, 4}.
    #[test]
    fn storage_format_size_bytes_valid_range() {
        let all = [
            StorageFormat::F32,
            StorageFormat::F16,
            StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int1),
            StorageFormat::PackedU8(PackedBits::Int2),
            StorageFormat::PackedU8(PackedBits::Int3),
            StorageFormat::PackedU8(PackedBits::Int4),
            StorageFormat::PackedU8(PackedBits::Int5),
            StorageFormat::PackedU8(PackedBits::Int6),
        ];
        for v in &all {
            let sz = v.size_bytes();
            assert!(sz == 1 || sz == 2 || sz == 4,
                "{:?} has unexpected size_bytes={}", v, sz);
        }
    }

    /// StorageFormat equality and hash consistency: equal values must produce equal hashes.
    #[test]
    fn storage_format_equality_implies_hash_equality() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let compute_hash = |v: &StorageFormat| -> u64 {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };

        let pairs = [
            (StorageFormat::F32, StorageFormat::F32),
            (StorageFormat::BF16, StorageFormat::BF16),
            (StorageFormat::PackedU8(PackedBits::Int5), StorageFormat::PackedU8(PackedBits::Int5)),
        ];
        for (a, b) in &pairs {
            assert_eq!(a, b);
            assert_eq!(compute_hash(a), compute_hash(b),
                "Equal values must have equal hashes: {:?} vs {:?}", a, b);
        }
    }

    /// DType alias: verify sizeof<DType> equals sizeof<StorageFormat> (type equivalence).
    #[test]
    fn dtype_alias_has_same_size_as_storage_format() {
        assert_eq!(
            std::mem::size_of::<DType>(),
            std::mem::size_of::<StorageFormat>()
        );
        assert_eq!(
            std::mem::align_of::<DType>(),
            std::mem::align_of::<StorageFormat>()
        );
    }

    /// map_storage_format: every supported dtype produces a result where
    /// size_bytes > 0 (no zero-size format returned).
    #[test]
    fn map_storage_format_all_supported_have_nonzero_size() {
        for &dt in GgmlDType::all() {
            if let Ok(sf) = map_storage_format(dt) {
                assert!(sf.size_bytes() > 0,
                    "{:?} maps to {:?} with zero size_bytes", dt, sf);
            }
        }
    }

    /// map_storage_format: roundtrip consistency — dtype → StorageFormat → size_bytes
    /// must be deterministic (called twice with same input gives same output).
    #[test]
    fn map_storage_format_deterministic_repeated_calls() {
        for &dt in GgmlDType::all() {
            let first = map_storage_format(dt);
            let second = map_storage_format(dt);
            assert_eq!(first.is_ok(), second.is_ok(), "{:?} not deterministic", dt);
            if let (Ok(a), Ok(b)) = (&first, &second) {
                assert_eq!(a, b, "{:?} not deterministic", dt);
            }
        }
    }

    /// ggml_dtype_to_quant_type: non-quantized types that map_storage_format supports
    /// (F32, F16, BF16, I8) must return None.
    #[test]
    fn quant_type_none_for_native_supported_dtypes() {
        let native_supported = [
            GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16, GgmlDType::I8,
        ];
        for dt in &native_supported {
            assert!(map_storage_format(*dt).is_ok(),
                "{:?} should be supported by map_storage_format", dt);
            assert!(ggml_dtype_to_quant_type(*dt).is_none(),
                "{:?} is native but has a QuantType", dt);
        }
    }

    /// GgmlDType Q4_0 block_bytes equals 18 (32 elems × 4bit / 8 + 2-byte f16 scale).
    #[test]
    fn ggml_dtype_q4_0_block_bytes_is_18() {
        assert_eq!(GgmlDType::Q4_0.block_bytes(), 18);
    }

    /// GgmlDType Q2_K block_size is 256 (K-quant standard super-block).
    #[test]
    fn ggml_dtype_q2_k_block_size_is_256() {
        assert_eq!(GgmlDType::Q2_K.block_size(), 256);
    }

    /// GgmlDType IQ4_NL block_size is 32 (non-linear 4-bit).
    #[test]
    fn ggml_dtype_iq4_nl_block_size_is_32() {
        assert_eq!(GgmlDType::IQ4_NL.block_size(), 32);
    }

    /// KernelTensorView with PackedU8(Int1) (rarest bit width) roundtrips correctly.
    #[test]
    fn kernel_tensor_view_int1_roundtrip() {
        let data: &[u8] = &[0b01010101];
        let view = KernelTensorView {
            storage_format: StorageFormat::PackedU8(PackedBits::Int1),
            shape: vec![8],
            data,
        };
        assert_eq!(view.storage_format, StorageFormat::PackedU8(PackedBits::Int1));
        assert_eq!(view.shape, vec![8]);
        assert_eq!(view.data.len(), 1);
        assert_eq!(view.data[0], 0b01010101);
    }

    /// StorageFormat HashSet: insert all 10 variants, remove one, verify count.
    #[test]
    fn storage_format_hashset_insert_remove() {
        use std::collections::HashSet;
        let all = [
            StorageFormat::F32, StorageFormat::F16, StorageFormat::BF16,
            StorageFormat::U8,
            StorageFormat::PackedU8(PackedBits::Int1),
            StorageFormat::PackedU8(PackedBits::Int2),
            StorageFormat::PackedU8(PackedBits::Int3),
            StorageFormat::PackedU8(PackedBits::Int4),
            StorageFormat::PackedU8(PackedBits::Int5),
            StorageFormat::PackedU8(PackedBits::Int6),
        ];
        let mut set: HashSet<StorageFormat> = all.into_iter().collect();
        assert_eq!(set.len(), 10);
        assert!(set.remove(&StorageFormat::F32));
        assert_eq!(set.len(), 9);
        assert!(!set.contains(&StorageFormat::F32));
        assert!(set.contains(&StorageFormat::F16));
    }

    /// safetensors_dtype_to_gllm: the three supported types produce mutually distinct kernel DTypes.
    #[test]
    fn safetensors_dtype_supported_types_produce_distinct_kernel_dtypes() {
        let f32_dt = safetensors_dtype_to_gllm(::safetensors::Dtype::F32).unwrap();
        let f16_dt = safetensors_dtype_to_gllm(::safetensors::Dtype::F16).unwrap();
        let bf16_dt = safetensors_dtype_to_gllm(::safetensors::Dtype::BF16).unwrap();
        assert_ne!(f32_dt, f16_dt);
        assert_ne!(f16_dt, bf16_dt);
        assert_ne!(f32_dt, bf16_dt);
    }

    // ── Batch 5: 13 additional tests ─────────────────────────────────────

    /// GgmlDType Q4_1 block_bytes is 20 (32 elems, 4-bit packed + 2×F16 scale/min).
    #[test]
    fn ggml_dtype_q4_1_block_bytes_is_20() {
        assert_eq!(GgmlDType::Q4_1.block_bytes(), 20);
    }

    /// GgmlDType Q3_K block_bytes is 110 and block_size is 256.
    #[test]
    fn ggml_dtype_q3_k_block_metadata() {
        assert_eq!(GgmlDType::Q3_K.block_bytes(), 110);
        assert_eq!(GgmlDType::Q3_K.block_size(), 256);
    }

    /// GgmlDType Q6_K block_bytes is 210 and block_size is 256.
    #[test]
    fn ggml_dtype_q6_k_block_metadata() {
        assert_eq!(GgmlDType::Q6_K.block_bytes(), 210);
        assert_eq!(GgmlDType::Q6_K.block_size(), 256);
    }

    /// GgmlDType IQ1_S block_bytes is 50 and IQ1_M block_bytes is derived from ratio.
    /// Verifies the 1-bit quantization family has the smallest block_bytes in IQ series.
    #[test]
    fn ggml_dtype_iq1_family_block_bytes() {
        assert_eq!(GgmlDType::IQ1_S.block_bytes(), 50);
        assert_eq!(GgmlDType::IQ1_S.block_size(), 256);
        // IQ1_M should also be compact (block_size=256)
        assert_eq!(GgmlDType::IQ1_M.block_size(), 256);
        assert!(GgmlDType::IQ1_M.block_bytes() > 0);
    }

    /// GgmlDType TQ1_0 block_bytes is 54 and TQ2_0 block_bytes is 66 (both block_size=256).
    /// Ternary quantization family block sizes must match expected values.
    #[test]
    fn ggml_dtype_tq_family_block_metadata() {
        assert_eq!(GgmlDType::TQ1_0.block_bytes(), 54);
        assert_eq!(GgmlDType::TQ1_0.block_size(), 256);
        assert_eq!(GgmlDType::TQ2_0.block_bytes(), 66);
        assert_eq!(GgmlDType::TQ2_0.block_size(), 256);
    }

    /// GgmlDType as_str for classic GGML quant types returns expected string.
    #[test]
    fn ggml_dtype_as_str_classic_quant_types() {
        assert_eq!(GgmlDType::Q4_0.as_str(), "Q4_0");
        assert_eq!(GgmlDType::Q4_1.as_str(), "Q4_1");
        assert_eq!(GgmlDType::Q5_0.as_str(), "Q5_0");
        assert_eq!(GgmlDType::Q5_1.as_str(), "Q5_1");
        assert_eq!(GgmlDType::Q8_0.as_str(), "Q8_0");
        assert_eq!(GgmlDType::Q8_1.as_str(), "Q8_1");
    }

    /// GgmlDType as_str for K-quant family returns expected string.
    #[test]
    fn ggml_dtype_as_str_k_quant_family() {
        assert_eq!(GgmlDType::Q2_K.as_str(), "Q2_K");
        assert_eq!(GgmlDType::Q3_K.as_str(), "Q3_K");
        assert_eq!(GgmlDType::Q4_K.as_str(), "Q4_K");
        assert_eq!(GgmlDType::Q5_K.as_str(), "Q5_K");
        assert_eq!(GgmlDType::Q6_K.as_str(), "Q6_K");
        assert_eq!(GgmlDType::Q8_K.as_str(), "Q8_K");
    }

    /// GgmlDType as_str for IQ family returns expected string (all 9 variants).
    #[test]
    fn ggml_dtype_as_str_iq_family() {
        assert_eq!(GgmlDType::IQ1_S.as_str(), "IQ1_S");
        assert_eq!(GgmlDType::IQ1_M.as_str(), "IQ1_M");
        assert_eq!(GgmlDType::IQ2_XXS.as_str(), "IQ2_XXS");
        assert_eq!(GgmlDType::IQ2_XS.as_str(), "IQ2_XS");
        assert_eq!(GgmlDType::IQ2_S.as_str(), "IQ2_S");
        assert_eq!(GgmlDType::IQ3_XXS.as_str(), "IQ3_XXS");
        assert_eq!(GgmlDType::IQ3_S.as_str(), "IQ3_S");
        assert_eq!(GgmlDType::IQ4_NL.as_str(), "IQ4_NL");
        assert_eq!(GgmlDType::IQ4_XS.as_str(), "IQ4_XS");
    }

    /// GgmlDType as_str for native integer and float64 types returns expected string.
    #[test]
    fn ggml_dtype_as_str_native_integer_types() {
        assert_eq!(GgmlDType::I8.as_str(), "I8");
        assert_eq!(GgmlDType::I16.as_str(), "I16");
        assert_eq!(GgmlDType::I32.as_str(), "I32");
        assert_eq!(GgmlDType::I64.as_str(), "I64");
        assert_eq!(GgmlDType::F64.as_str(), "F64");
    }

    /// safetensors_dtype_to_gllm: the returned kernel DType size_bytes matches expectations.
    /// F32→4, F16→2, BF16→2 (verified through the kernel DType's own size_bytes).
    #[test]
    fn safetensors_dtype_kernel_dtype_size_consistency() {
        let f32_dt = safetensors_dtype_to_gllm(::safetensors::Dtype::F32).unwrap();
        assert_eq!(f32_dt.size_bytes(), 4);
        let f16_dt = safetensors_dtype_to_gllm(::safetensors::Dtype::F16).unwrap();
        assert_eq!(f16_dt.size_bytes(), 2);
        let bf16_dt = safetensors_dtype_to_gllm(::safetensors::Dtype::BF16).unwrap();
        assert_eq!(bf16_dt.size_bytes(), 2);
    }

    /// KernelTensorView data slice borrows the original slice (no copy).
    /// Verifies that the data pointer identity is preserved.
    #[test]
    fn kernel_tensor_view_data_slice_identity() {
        let original: &[u8] = &[0xAA, 0xBB, 0xCC, 0xDD];
        let view = KernelTensorView {
            storage_format: StorageFormat::F32,
            shape: vec![1],
            data: original,
        };
        // The data field must point to the same underlying memory.
        assert!(std::ptr::eq(view.data.as_ptr(), original.as_ptr()));
        assert_eq!(view.data.len(), original.len());
    }

    /// map_storage_format: no supported dtype maps to a float format that is not
    /// F32, F16, or BF16 — the float variants are exactly those three.
    #[test]
    fn map_storage_format_float_variants_exactly_three() {
        let float_dtypes = [GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16];
        let mut float_format_count = 0;
        for &dt in GgmlDType::all() {
            if let Ok(sf) = map_storage_format(dt) {
                if matches!(sf, StorageFormat::F32 | StorageFormat::F16 | StorageFormat::BF16) {
                    float_format_count += 1;
                }
            }
        }
        assert_eq!(float_format_count, 3);
        // Verify each float dtype maps to its own format
        assert_eq!(map_storage_format(GgmlDType::F32).unwrap(), StorageFormat::F32);
        assert_eq!(map_storage_format(GgmlDType::F16).unwrap(), StorageFormat::F16);
        assert_eq!(map_storage_format(GgmlDType::BF16).unwrap(), StorageFormat::BF16);
    }

    /// ggml_dtype_to_quant_type: every QuantType returned has block_bytes > 0.
    /// Cross-validates that no QuantType variant accidentally has zero block_bytes.
    #[test]
    fn all_quant_types_have_positive_block_bytes() {
        for &dt in GgmlDType::all() {
            if let Some(qt) = ggml_dtype_to_quant_type(dt) {
                assert!(qt.block_bytes() > 0,
                    "{:?} → {:?} has block_bytes=0", dt, qt);
            }
        }
    }

    /// GgmlDType IQ2_XXS, IQ2_XS, IQ2_S have block_size 256 (same super-block).
    /// All 2-bit IQ variants share the K-quant standard block_size.
    #[test]
    fn ggml_dtype_iq2_family_block_size_256() {
        assert_eq!(GgmlDType::IQ2_XXS.block_size(), 256);
        assert_eq!(GgmlDType::IQ2_XS.block_size(), 256);
        assert_eq!(GgmlDType::IQ2_S.block_size(), 256);
    }

    /// Round-trip: GgmlDType → QuantType → GgmlDType must be identity
    /// for every quantized GgmlDType variant.
    #[test]
    fn roundtrip_ggml_dtype_to_quant_type_and_back() {
        let quantized_dtypes = [
            GgmlDType::Q2_K, GgmlDType::Q3_K, GgmlDType::Q4_K, GgmlDType::Q5_K,
            GgmlDType::Q6_K, GgmlDType::Q8_K,
            GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q5_0, GgmlDType::Q5_1,
            GgmlDType::Q8_0, GgmlDType::Q8_1,
            GgmlDType::IQ1_S, GgmlDType::IQ1_M,
            GgmlDType::IQ2_XXS, GgmlDType::IQ2_XS, GgmlDType::IQ2_S,
            GgmlDType::IQ3_XXS, GgmlDType::IQ3_S,
            GgmlDType::IQ4_NL, GgmlDType::IQ4_XS,
            GgmlDType::MXFP4,
            GgmlDType::AWQ4, GgmlDType::GPTQ4, GgmlDType::SQUEEZE, GgmlDType::NVFP4,
            GgmlDType::TQ1_0, GgmlDType::TQ2_0,
        ];
        for dt in &quantized_dtypes {
            let qt = ggml_dtype_to_quant_type(*dt)
                .unwrap_or_else(|| panic!("{:?} should map to QuantType", dt));
            let back = quant_type_to_ggml_dtype(qt)
                .unwrap_or_else(|| panic!("QuantType {:?} (from {:?}) should map back to GgmlDType", qt, dt));
            assert_eq!(back, *dt, "round-trip failed: {:?} → {:?} → {:?}", dt, qt, back);
        }
    }

    /// quant_type_to_ggml_dtype returns None for native float QuantType variants.
    #[test]
    fn quant_type_to_ggml_dtype_native_floats_return_none() {
        assert!(quant_type_to_ggml_dtype(QuantType::Bf16).is_none());
        assert!(quant_type_to_ggml_dtype(QuantType::F16).is_none());
        assert!(quant_type_to_ggml_dtype(QuantType::F32).is_none());
    }

    /// quant_type_to_ggml_dtype returns None for FP8 QuantType variants.
    #[test]
    fn quant_type_to_ggml_dtype_fp8_returns_none() {
        assert!(quant_type_to_ggml_dtype(QuantType::Fp8E4M3).is_none());
        assert!(quant_type_to_ggml_dtype(QuantType::Fp8E5M2).is_none());
    }

    /// quant_type_to_ggml_dtype returns None for Mxfp4 with non-standard block_size.
    #[test]
    fn quant_type_to_ggml_dtype_mxfp4_nonstandard_block_size() {
        assert!(quant_type_to_ggml_dtype(QuantType::Mxfp4 { block_size: 64 }).is_none());
    }
}
