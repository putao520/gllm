//! DAG 图处理模块
//!
//! 本模块提供图基础设施和回调链支持。
//! 所有模型统一通过 Mega-Kernel 路径执行 (REQ-UGS-005)。

pub mod profile;
pub mod types;
pub mod layer_callback;
