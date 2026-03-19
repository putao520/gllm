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

pub use registry::{get_template, get_template_by_arch, register_builtin_templates, ArchRegistry};
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

    crate::graph::executor::FusedGraphExecutor::from_graph(onnx_graph, seq_len, hidden)
}
