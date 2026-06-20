//! Tensor-name-driven graph builder.
//!
//! Automatically derives a complete CompilerGraph from tensor names + shapes,
//! eliminating the need for YAML architecture templates.
//!
//! Architecture: `tensor names → role index → ArchitectureFeatures → CompilerGraph`
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 3 个片段):
//! - `auto_graph_fragments/types.inc.rs`       — ArchitectureFeatures, analyze_architecture, GraphBuildError, helpers
//! - `auto_graph_fragments/build_graph.inc.rs` — build_compiler_graph (核心图构建)
//! - `auto_graph_fragments/tests.inc.rs`       — 测试模块
// @trace REQ-FATOP-013 from_op_kind 翻译器已物理删除 — BUILD 阶段直接构造 Op
// @trace REQ-FATOP-023 OpKind enum 物理删除 — Op 单 IR 收敛
// @trace REQ-FATOP-024 [entity:Op] add_op(Op) 统一 API — OpKind 参数删除

use std::collections::HashMap;

use gllm_kernels::compiler::graph::{
    AttentionGeometry, AttentionMask, AttentionSpec, CompilerGraph, DualRopeSpec, GemmSpec, MlaSpec,
    NormSpec, Op, QuantGemmSpec, RopeSpec, SinksSpec, SymDim, TensorId,
};
use gllm_kernels::compiler::mega_kernel_abi::BusinessConfig;
use gllm_kernels::types::DType;

use crate::manifest::TensorRole;
use crate::model_config::{ArchHints, HiddenAct};
use super::resolve::ResolvedConfig;

include!("auto_graph_fragments/types.inc.rs");
include!("auto_graph_fragments/build_graph.inc.rs");

#[cfg(test)]
include!("auto_graph_fragments/tests.inc.rs");
