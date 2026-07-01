//! 架构推导系统
//!
//! - `auto_graph`: Tensor-name-driven graph builder（从 tensor names 自动推导 CompilerGraph）
//! - `resolve`: 配置推导
//! - `registry`: Architecture token 别名查找表

pub mod auto_graph;
pub mod c1_v2_graph;
pub mod c1_v2_tracker;
pub mod intent_tracker_graph;
mod registry;
pub mod resolve;

pub use registry::{
    resolve_template_name, resolve_family,
    resolve_moe_router, is_valid_template,
};
pub use resolve::ResolvedConfig;
