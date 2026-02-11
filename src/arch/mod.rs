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

pub use registry::{get_template, register_builtin_templates, ArchRegistry};
pub use resolve::{resolve_config, ResolvedConfig};
pub use template::{ArchTemplate, GraphNode, NodeDef, RepeatBlock};
