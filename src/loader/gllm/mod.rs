//! gllm 原生模型权重格式 (.gllm) 解析器。
//!
//! SPEC: `SPEC/36-GLLM-WEIGHT-FORMAT.md`
//!
//! # 文件布局
//!
//! ```text
//! Header (64B) → Tensor Directory (64B × N) → String Table → Metadata (MessagePack) → Data
//! ```
//!
//! 量化权重以量化格式原生存储（AWQ4/GPTQ4/NVFP4/FP8 等），
//! 推理时通过 GEMM prologue (QuantGather) on-the-fly 解量化。

pub mod convert;
pub mod quant_encode;
mod reader;
mod types;
pub mod writer;

pub use convert::{ConvertOptions, ConvertResult, convert_gguf_fp16_to_gllm, convert_gguf_to_gllm, convert_safetensors_to_gllm};
pub use reader::{GllmModelParams, GllmReader};
pub use types::{GllmError, GllmHeader, GllmTensorEntry};
pub use writer::{GllmWriter, TensorEntry, build_metadata, quant_type_to_u8};

/// Magic number: "GLLM" in ASCII (little-endian).
pub const GLLM_MAGIC: u32 = 0x4D4C4C47; // 'G','L','L','M'
/// Current supported format version.
pub const GLLM_VERSION: u32 = 1;
/// Header size in bytes.
pub const HEADER_SIZE: usize = 64;
/// Tensor directory entry size in bytes.
pub const TENSOR_ENTRY_SIZE: usize = 72;
