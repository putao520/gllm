//! 优化器模块 (REQ-OPT-001 ~ REQ-OPT-004)

mod dead_code;
mod hardware_fusion;
mod pass;
mod pattern_fusion;

pub use pass::{OptimizationContext, OptimizationPass};

use crate::loader::onnx::OnnxGraph;

use super::types::{AtomicOp, FusedGraph, FusedNode, FusedOp};

/// 图优化器 (REQ-OPT-004)
#[derive(Debug)]
pub struct GraphOptimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
    context: OptimizationContext,
}

impl GraphOptimizer {
    /// 创建默认优化器（包含所有内置 Pass）
    pub fn new(context: OptimizationContext) -> Self {
        let mut optimizer = Self {
            passes: Vec::new(),
            context,
        };
        optimizer.register_builtin_passes();
        optimizer
    }

    /// 创建空优化器（无 Pass）
    pub fn empty(context: OptimizationContext) -> Self {
        Self {
            passes: Vec::new(),
            context,
        }
    }

    /// 注册优化 Pass
    pub fn register_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
        self.passes.sort_by_key(|p| p.priority());
    }

    /// 注册内置 Pass
    fn register_builtin_passes(&mut self) {
        self.register_pass(Box::new(pattern_fusion::FlashAttentionFusionPass));
        self.register_pass(Box::new(pattern_fusion::FusedQkvRopeFusionPass));
        self.register_pass(Box::new(pattern_fusion::SwiGLUFusionPass));
        self.register_pass(Box::new(pattern_fusion::FusedRMSLinearFusionPass));
        self.register_pass(Box::new(hardware_fusion::HardwareFusionPass));
        self.register_pass(Box::new(dead_code::DeadCodeEliminationPass));
    }

    /// 执行优化
    pub fn optimize(&self, graph: &OnnxGraph) -> Result<FusedGraph, OptimizeError> {
        // 1. 转换为中间表示
        let mut fused = self.convert_to_fused(graph)?;
        let original_count = fused.nodes.len();
        fused.stats.original_nodes = original_count;

        // 2. 执行所有 Pass
        for pass in &self.passes {
            if pass.enabled(&self.context) {
                fused = pass.run(fused, &self.context)?;
            }
        }

        // 3. 更新统计
        fused.stats.optimized_nodes = fused.nodes.len();

        Ok(fused)
    }

    /// 将 OnnxGraph 转换为 FusedGraph
    fn convert_to_fused(&self, graph: &OnnxGraph) -> Result<FusedGraph, OptimizeError> {
        let mut fused = FusedGraph::new();

        // 复制输入输出
        fused.inputs = graph.inputs.iter().map(|i| i.name.clone()).collect();
        fused.outputs = graph.outputs.iter().map(|o| o.name.clone()).collect();

        // 转换节点
        for node in &graph.nodes {
            let fused_node = FusedNode {
                name: node.name.clone(),
                op: FusedOp::Atomic(AtomicOp::new(&node.op_type)),
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                attributes: std::collections::HashMap::new(),
            };
            fused.nodes.push(fused_node);
        }

        for (name, tensor) in &graph.initializers {
            fused.weight_bindings.insert(
                name.clone(),
                crate::graph::types::WeightBinding {
                    source_name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                },
            );
        }

        Ok(fused)
    }

    /// 获取已注册的 Pass 名称
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.name()).collect()
    }
}

/// 优化错误
#[derive(Debug, thiserror::Error)]
pub enum OptimizeError {
    #[error("Pass '{pass}' failed: {reason}")]
    PassFailed { pass: String, reason: String },
    #[error("Invalid graph: {0}")]
    InvalidGraph(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_registers_builtin_passes() {
        let ctx = OptimizationContext::default();
        let optimizer = GraphOptimizer::new(ctx);
        let names = optimizer.pass_names();
        assert!(names.contains(&"SwiGLUFusion"));
        assert!(names.contains(&"FlashAttentionFusion"));
        assert!(names.contains(&"DeadCodeElimination"));
    }

    #[test]
    fn empty_optimizer_has_no_passes() {
        let ctx = OptimizationContext::default();
        let optimizer = GraphOptimizer::empty(ctx);
        assert!(optimizer.pass_names().is_empty());
    }
}
