//! Model config loader (metadata/tensor-driven, no config.json fallback).
#![allow(clippy::manual_checked_ops)]
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 6 个片段):
//! - `model_config_fragments/types.inc.rs`         — MlaConfig, ModelGeometry, ModelConfig struct
//! - `model_config_fragments/config_impl.inc.rs`   — impl ModelConfig (from_loader, from_gguf, from_value)
//! - `model_config_fragments/tensor_derive.inc.rs`  — TensorDerivedConfig + tensor-driven config 推导
//! - `model_config_fragments/helpers.inc.rs`        — 独立 helper 函数
//! - `model_config_fragments/field_registry.inc.rs` — FieldDef 注册表 + normalize_text_config + apply_field_registry (BCE-040)
//! - `model_config_fragments/tests.inc.rs`          — 测试模块

use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

use crate::loader::{
    gguf::GgufReader as GgufLoader, match_tensor_role, Loader, TensorMeta, TensorProvider,
    WeightFormat,
};
use crate::manifest::{ModelManifest, TensorRole};
use gllm_kernels::types::DType;

include!("model_config_fragments/types.inc.rs");
include!("model_config_fragments/config_impl.inc.rs");
include!("model_config_fragments/tensor_derive.inc.rs");
include!("model_config_fragments/helpers.inc.rs");
include!("model_config_fragments/field_registry.inc.rs");

#[cfg(test)]
include!("model_config_fragments/tests.inc.rs");
