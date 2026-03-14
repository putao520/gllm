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

/// 优化 OnnxGraph，返回融合后的 FusedGraph。
///
/// 按优先级执行所有内置 pass:
/// 1. ConstantFolding (0)
/// 2. FlashAttentionFusion (10)
/// 3. GQAFusion (14)
/// 4. FusedQkvRopeFusion (15)
/// 5. SwiGLUFusion (20)
/// 6. MoERoutingFusion (22)
/// 7. FusedRMSLinearFusion (25)
/// 8. HardwareFusion (40)
/// 9. DeadCodeElimination (100)
pub fn optimize(
    graph: &crate::loader::onnx::OnnxGraph,
    ctx: &OptimizationContext,
) -> Result<FusedGraph, optimizer::OptimizeError> {
    let optimizer = GraphOptimizer::new(ctx.clone());
    optimizer.optimize(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::onnx::{OnnxGraph, OnnxNode, OnnxValueInfo};
    use std::collections::HashMap;

    fn make_onnx_node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> OnnxNode {
        OnnxNode {
            name: name.to_string(),
            op_type: op_type.to_string(),
            domain: String::new(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attributes: HashMap::new(),
        }
    }

    fn make_value_info(name: &str) -> OnnxValueInfo {
        OnnxValueInfo {
            name: name.to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        }
    }

    #[test]
    fn test_optimization_pipeline_runs_all_passes() {
        // Build a graph with a fusable SwiGLU pattern + dead code
        let graph = OnnxGraph {
            name: "test".to_string(),
            doc_string: String::new(),
            nodes: vec![
                // SwiGLU pattern: gate -> up -> silu -> mul
                make_onnx_node("layer_0_gate", "MatMul", &["x"], &["gate_out"]),
                make_onnx_node("layer_0_up", "MatMul", &["x"], &["up_out"]),
                make_onnx_node("layer_0_silu", "SiLU", &["gate_out"], &["silu_out"]),
                make_onnx_node("layer_0_mul", "Mul", &["silu_out", "up_out"], &["output"]),
                // Dead code: not connected to output
                make_onnx_node("dead_node", "MatMul", &["x"], &["dead_out"]),
            ],
            inputs: vec![make_value_info("x")],
            outputs: vec![make_value_info("output")],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };

        let ctx = OptimizationContext::default();
        let result = optimize(&graph, &ctx).unwrap();

        // SwiGLU fusion should have fired
        assert_eq!(result.stats.swiglu_fusions, 1);
        // Dead code should have been eliminated
        assert_eq!(result.stats.dead_code_eliminated, 1);
        // Original: 5 nodes, after SwiGLU: 4 -> 2 (1 fused + 1 dead), after DCE: 1
        assert_eq!(result.stats.original_nodes, 5);
        assert_eq!(result.node_count(), 1);
        assert!(matches!(result.nodes[0].op, FusedOp::SwiGLU(_)));
    }

    #[test]
    fn test_no_naive_1to1_translation() {
        // Build a graph with Q/K/V + RoPE pattern
        let graph = OnnxGraph {
            name: "test_qkv_rope".to_string(),
            doc_string: String::new(),
            nodes: vec![
                make_onnx_node("layer_0_q_proj", "MatMul", &["x"], &["q"]),
                make_onnx_node("layer_0_k_proj", "MatMul", &["x"], &["k"]),
                make_onnx_node("layer_0_v_proj", "MatMul", &["x"], &["v"]),
                make_onnx_node("layer_0_rope", "RotaryEmbedding", &["q", "k"], &["output"]),
            ],
            inputs: vec![make_value_info("x")],
            outputs: vec![make_value_info("output")],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };

        let ctx = OptimizationContext::default();
        let result = optimize(&graph, &ctx).unwrap();

        // The 4 atomic nodes should be fused into 1 FusedQkvRope node
        assert_eq!(result.stats.qkv_rope_fusions, 1);
        assert_eq!(result.stats.original_nodes, 4);
        assert_eq!(result.node_count(), 1);
        assert!(matches!(result.nodes[0].op, FusedOp::FusedQkvRope(_)));

        // Verify no atomic MatMul/RotaryEmbedding nodes remain (no naive 1:1 translation)
        for node in &result.nodes {
            match &node.op {
                FusedOp::Atomic(op) => {
                    assert!(
                        op.op_type != "MatMul" && op.op_type != "RotaryEmbedding",
                        "Naive 1:1 translation detected: atomic {} node should have been fused",
                        op.op_type
                    );
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_optimize_convenience_fn_matches_manual() {
        let graph = OnnxGraph {
            name: "test".to_string(),
            doc_string: String::new(),
            nodes: vec![
                make_onnx_node("n0", "Add", &["a", "b"], &["output"]),
            ],
            inputs: vec![make_value_info("a"), make_value_info("b")],
            outputs: vec![make_value_info("output")],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };

        let ctx = OptimizationContext::default();
        let convenience = optimize(&graph, &ctx).unwrap();
        let manual = GraphOptimizer::new(ctx).optimize(&graph).unwrap();

        assert_eq!(convenience.nodes.len(), manual.nodes.len());
        assert_eq!(convenience.stats, manual.stats);
    }
}
