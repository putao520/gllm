//! DAG 图处理模块 (REQ-OPT-001 ~ REQ-OPT-005, REQ-EXEC-002)
//!
//! 本模块实现 OnnxGraph 的优化和执行。
//!
//! ## 设计目标
//! - Ω4: 动态 DAG 优化，最大化性能
//!
//! ## 模块结构
//! - `types`: 扩展图类型定义
//! - `optimizer`: 优化器管道和 Pass
//! - `executor`: FusedGraph 执行器

pub mod executor;
pub mod optimizer;
pub mod types;

pub use executor::{
    ExecutionContext, ExecutionError, ExecutionOp, ExecutionPlan, FusedGraphExecutor,
};
pub use optimizer::{GraphOptimizer, OptimizationContext, OptimizationPass};
pub use types::{FusedGraph, FusedNode, FusedOp};
