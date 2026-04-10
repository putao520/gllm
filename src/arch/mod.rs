//! 架构模板系统 (REQ-ARCH-001 ~ REQ-ARCH-005)
//!
//! 本模块实现 YAML 架构模板 → OnnxGraph 的转换，替代硬编码 adapter。
//!
//! ## 设计目标
//! - Ω1: 100% 从模型文件推导配置
//! - Ω3: 统一到 OnnxGraph 表示
//!
//! ## 模块结构
//! - `template`: YAML 模板解析和类型定义
//! - `registry`: 架构模板注册表
//! - `resolve`: 占位符解析和配置推导

mod registry;
mod resolve;
mod template;

pub use registry::{
    get_template, register_builtin_templates,
    resolve_template, resolve_template_name, resolve_family,
    resolve_moe_router, is_valid_template,
    ArchRegistry,
};
pub use resolve::{resolve_config, ResolvedConfig};
pub use template::{ArchTemplate, GraphNode, NodeDef, RepeatBlock};

/// Build a `FusedGraphExecutor` from a registered YAML template name.
///
/// Steps:
/// 1. Look up the template in the global registry (calls `register_builtin_templates`
///    if not yet initialised).
/// 2. Expand the template with `config` → `OnnxGraph`.
/// 3. Run graph optimisation passes + JIT-compile every node.
///
/// `seq_len` and `hidden` are the concrete shape dimensions used for JIT
/// compilation (they can be representative values; symbolic dims are resolved
/// at runtime via `ShapeBinding`).
///
/// Returns `Err` if the template is unknown, expansion fails, or JIT
/// compilation fails for any node.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
pub fn build_executor_from_yaml(
    arch_name: &str,
    config: &ResolvedConfig,
    seq_len: usize,
    hidden: usize,
    dtype: gllm_kernels::types::DType,
    model_id: &str,
    backend: &str,
    cache: &crate::compat::artifact_cache::ArtifactCache,
    arch_family: crate::manifest::ArchFamily,
) -> Result<crate::graph::executor::FusedGraphExecutor, crate::graph::executor::ExecutorError> {
    register_builtin_templates();

    let template = get_template(arch_name).ok_or_else(|| {
        crate::graph::executor::ExecutorError::CompilationFailed(format!(
            "unknown architecture template: '{arch_name}'"
        ))
    })?;

    let onnx_graph = template.to_onnx_graph(config).map_err(|e| {
        crate::graph::executor::ExecutorError::CompilationFailed(format!(
            "template expansion for '{arch_name}': {e}"
        ))
    })?;

    let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
        hidden_size: config.hidden_size,
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        num_layers: config.num_hidden_layers,
        vocab_size: config.vocab_size,
        intermediate_size: config.intermediate_size.unwrap_or(config.hidden_size * 4),
        max_seq_len: 4096,
        rope_theta: config.rope_theta,
        rope_scale: 1.0,
        rope_interleaved: false,
        dtype,
        norm_eps: 1e-5,
        num_experts: 0,
        moe_top_k: 0,
        expert_intermediate_size: 0,
        global_rope_theta: 0.0,
        rope_partial_ratio: 1.0,
        attention_pattern: vec![],
        sliding_window: 0,
        num_kv_shared_layers: 0,
        global_head_dim: 0,
        hidden_size_per_layer_input: 0,
    });
    let ctx = crate::graph::optimizer::OptimizationContext {
        geometry,
        arch_family,
        ..Default::default()
    };

    crate::graph::executor::FusedGraphExecutor::from_graph_with_cache(
        onnx_graph, seq_len, hidden, dtype, model_id, backend, cache, ctx
    )
}

/// Build an **uncompiled** `FusedGraphExecutor` from a YAML template.
///
/// Only runs template expansion + graph optimisation. Does NOT JIT-compile.
/// Caller must populate weight shapes and then call `compile_with_cache()`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
pub fn build_uncompiled_executor_from_yaml(
    arch_name: &str,
    config: &ResolvedConfig,
    dtype: gllm_kernels::types::DType,
    arch_family: crate::manifest::ArchFamily,
) -> Result<crate::graph::executor::FusedGraphExecutor, crate::graph::executor::ExecutorError> {
    register_builtin_templates();

    let template = get_template(arch_name).ok_or_else(|| {
        crate::graph::executor::ExecutorError::CompilationFailed(format!(
            "unknown architecture template: '{arch_name}'"
        ))
    })?;

    let onnx_graph = template.to_onnx_graph(config).map_err(|e| {
        crate::graph::executor::ExecutorError::CompilationFailed(format!(
            "template expansion for '{arch_name}': {e}"
        ))
    })?;

    let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
        hidden_size: config.hidden_size,
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        num_layers: config.num_hidden_layers,
        vocab_size: config.vocab_size,
        intermediate_size: config.intermediate_size.unwrap_or(config.hidden_size * 4),
        max_seq_len: 4096,
        rope_theta: config.rope_theta,
        rope_scale: 1.0,
        rope_interleaved: false,
        dtype,
        norm_eps: 1e-5,
        num_experts: 0,
        moe_top_k: 0,
        expert_intermediate_size: 0,
        global_rope_theta: 0.0,
        rope_partial_ratio: 1.0,
        attention_pattern: vec![],
        sliding_window: 0,
        num_kv_shared_layers: 0,
        global_head_dim: 0,
        hidden_size_per_layer_input: 0,
    });
    let ctx = crate::graph::optimizer::OptimizationContext {
        geometry,
        arch_family,
        ..Default::default()
    };

    crate::graph::executor::FusedGraphExecutor::from_graph_optimized(onnx_graph, ctx)
}
