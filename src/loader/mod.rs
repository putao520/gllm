//! Layer 2/3: Loader (HF + SafeTensors + fused splits).
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 4 个片段):
//! - `fragments/types.inc.rs`         — QuantizedTensor, LoaderConfig, LoaderError, TensorProvider 等
//! - `fragments/loader_impl.inc.rs`   — Loader struct + impl (核心加载逻辑)
//! - `fragments/upload_convert.inc.rs` — TensorSlice, convert_tensor_to_f32, parallel 转换, LoadedWeights
//! - `fragments/tests.inc.rs`         — 测试模块

// unused imports removed
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ::safetensors::Dtype;
use crate::compat::backend_trait::{Backend, Element};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::manifest::{ModelManifest, TensorRole, EMPTY_FILE_MAP};
use crate::loader::gguf::GgmlDType;

// Re-export modules
pub mod adapter; // GGUF tensor adapter (KernelTensorView)
pub mod downloader;
pub mod format_detector;
pub mod gguf;
pub mod gllm;     // .gllm native weight format (SPEC 36)
pub mod hf_hub;
pub mod modelscope;
pub mod name_map;
pub mod onnx;
pub mod parallel;
pub mod pytorch;
pub mod safetensors;
pub mod weight_compress; // COMP12: Weight page compression (SPEC 22 §6)
#[cfg(feature = "nccl")]
pub mod weight_shard; // TP Weight Sharding (REQ-DIST-004)
pub mod weight_tier;

pub use downloader::{ModelScopeDownloader, ProgressBar};
pub use gguf::GgufReader as GgufLoader;
pub use hf_hub::HfHubClient;
pub use modelscope::ModelScopeClient;
pub use onnx::OnnxLoader;
pub use parallel::ParallelLoader;
pub use safetensors::SafeTensorsLoader;

use gllm_kernels::quant::QuantType;
pub use adapter::ggml_dtype_to_quant_type;
pub use pytorch::{PytorchLoader, PytorchLoaderConfig};

// Re-export quantization metadata types (defined later in this file)
// Note: CompanionConfig and QuantizationMetadata are already public below

include!("fragments/types.inc.rs");
include!("fragments/loader_impl.inc.rs");
include!("fragments/upload_convert.inc.rs");

#[cfg(test)]
include!("fragments/tests.inc.rs");
