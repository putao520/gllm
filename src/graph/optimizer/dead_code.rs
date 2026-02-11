//! 死代码消除 Pass

use super::pass::{OptimizationContext, OptimizationPass};
use super::OptimizeError;
use crate::graph::types::FusedGraph;

use std::collections::HashSet;

/// 死代码消除 Pass
#[derive(Debug, Clone, Copy)]
pub struct DeadCodeEliminationPass;

impl OptimizationPass for DeadCodeEliminationPass {
    fn name(&self) -> &'static str {
        "DeadCodeElimination"
    }

    fn run(
        &self,
        mut graph: FusedGraph,
        _ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        // 收集所有输出节点的名称
        let mut live_outputs: HashSet<String> = graph.outputs.iter().cloned().collect();

        // 反向遍历，标记活跃节点
        let mut changed = true;

        while changed {
            changed = false;
            for node in graph.nodes.iter().rev() {
                let is_live = node.outputs.iter().any(|o| live_outputs.contains(o));
                if is_live {
                    // 将该节点的输入加入活跃集合
                    for input in &node.inputs {
                        if live_outputs.insert(input.clone()) {
                            changed = true;
                        }
                    }
                }
            }
        }

        // 过滤只保留活跃节点
        let original_count = graph.nodes.len();
        graph
            .nodes
            .retain(|node| node.outputs.iter().any(|o| live_outputs.contains(o)));

        let eliminated = original_count - graph.nodes.len();
        graph.stats.dead_code_eliminated = eliminated;

        Ok(graph)
    }

    fn enabled(&self, _ctx: &OptimizationContext) -> bool {
        true // 始终启用
    }

    fn priority(&self) -> i32 {
        -100 // 最后执行
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::{AtomicOp, FusedNode, FusedOp};

    fn make_node(name: &str, inputs: &[&str], outputs: &[&str]) -> FusedNode {
        FusedNode {
            name: name.to_string(),
            op: FusedOp::Atomic(AtomicOp::new("MatMul")),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attributes: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn eliminates_unused_nodes() {
        let mut graph = FusedGraph::new();
        graph.inputs = vec!["input".to_string()];
        graph.outputs = vec!["output".to_string()];

        // 活跃路径: input -> a -> output
        graph.nodes.push(make_node("op_a", &["input"], &["a"]));
        graph
            .nodes
            .push(make_node("op_output", &["a"], &["output"]));

        // 死代码: input -> dead -> dead_out
        graph
            .nodes
            .push(make_node("op_dead", &["input"], &["dead_out"]));

        let pass = DeadCodeEliminationPass;
        let ctx = OptimizationContext::default();
        let result = pass.run(graph, &ctx).unwrap();

        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.stats.dead_code_eliminated, 1);
    }

    #[test]
    fn keeps_all_if_no_dead_code() {
        let mut graph = FusedGraph::new();
        graph.inputs = vec!["input".to_string()];
        graph.outputs = vec!["output".to_string()];

        graph.nodes.push(make_node("op_a", &["input"], &["a"]));
        graph
            .nodes
            .push(make_node("op_output", &["a"], &["output"]));

        let pass = DeadCodeEliminationPass;
        let ctx = OptimizationContext::default();
        let result = pass.run(graph, &ctx).unwrap();

        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.stats.dead_code_eliminated, 0);
    }
}
