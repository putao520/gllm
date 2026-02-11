//! FusedGraph 执行器 (REQ-EXEC-002)

use std::collections::{HashMap, HashSet};

use super::types::{FusedGraph, FusedOp};

/// FusedGraph 执行错误
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("Missing input tensor: {0}")]
    MissingInput(String),
    #[error("Missing weight tensor: {0}")]
    MissingWeight(String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),
    #[error("Backend error: {0}")]
    Backend(String),
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}

/// 执行上下文 - 保存中间张量名称和状态
#[derive(Debug, Default)]
pub struct ExecutionContext {
    pub computed: Vec<String>,
    pub outputs: Vec<String>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_computed(&mut self, name: String) {
        self.computed.push(name);
    }

    pub fn is_computed(&self, name: &str) -> bool {
        self.computed.iter().any(|n| n == name)
    }
}

/// FusedGraph 执行计划
#[derive(Debug)]
pub struct ExecutionPlan {
    pub operations: Vec<ExecutionOp>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// 执行操作
#[derive(Debug, Clone)]
pub enum ExecutionOp {
    FlashAttention {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    SwiGLU {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    RoPE {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        rope_theta: f64,
    },
    FusedQkvRope {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    FusedRMSLinear {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    Atomic {
        name: String,
        op_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
}

impl ExecutionPlan {
    pub fn from_fused_graph(graph: &FusedGraph) -> Self {
        let mut operations = Vec::new();

        for node in &graph.nodes {
            let op = match &node.op {
                FusedOp::FlashAttention(config) => ExecutionOp::FlashAttention {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                },
                FusedOp::SwiGLU(_config) => ExecutionOp::SwiGLU {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
                FusedOp::RoPE(config) => ExecutionOp::RoPE {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                    rope_theta: config.rope_theta,
                },
                FusedOp::FusedQkvRope(_config) => ExecutionOp::FusedQkvRope {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
                FusedOp::FusedRMSLinear(_config) => ExecutionOp::FusedRMSLinear {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
                FusedOp::Atomic(atomic) => ExecutionOp::Atomic {
                    name: node.name.clone(),
                    op_type: atomic.op_type.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
            };
            operations.push(op);
        }

        Self {
            operations,
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
        }
    }

    pub fn op_count(&self) -> usize {
        self.operations.len()
    }

    pub fn fused_op_count(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| !matches!(op, ExecutionOp::Atomic { .. }))
            .count()
    }
}

/// FusedGraph 执行器。
///
/// 当前实现为图调度与依赖检查，计算由后端算子层负责。
#[derive(Debug)]
pub struct FusedGraphExecutor {
    graph: FusedGraph,
}

impl FusedGraphExecutor {
    pub fn new(graph: FusedGraph) -> Self {
        Self { graph }
    }

    /// 运行融合图。
    ///
    /// 返回每个图输出名对应的占位张量（由后端执行阶段填充真实数据）。
    pub fn run(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
    ) -> Result<HashMap<String, Vec<u8>>, ExecutionError> {
        let mut available: HashSet<String> = inputs.keys().cloned().collect();

        for weight_name in self.graph.weight_bindings.keys() {
            available.insert(weight_name.clone());
        }

        for node in &self.graph.nodes {
            for input in &node.inputs {
                if input.is_empty() || available.contains(input) {
                    continue;
                }
                if self.graph.inputs.iter().any(|name| name == input) {
                    return Err(ExecutionError::MissingInput(input.clone()));
                }
                return Err(ExecutionError::MissingWeight(input.clone()));
            }

            match &node.op {
                FusedOp::FlashAttention(_)
                | FusedOp::SwiGLU(_)
                | FusedOp::RoPE(_)
                | FusedOp::FusedQkvRope(_)
                | FusedOp::FusedRMSLinear(_)
                | FusedOp::Atomic(_) => {}
            }

            for output in &node.outputs {
                if !output.is_empty() {
                    available.insert(output.clone());
                }
            }
        }

        let mut out = HashMap::new();
        for output in &self.graph.outputs {
            if !available.contains(output) {
                return Err(ExecutionError::MissingInput(output.clone()));
            }
            out.insert(output.clone(), Vec::new());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::{AtomicOp, FusedNode, OptimizationStats};

    #[test]
    fn execution_context_tracks_computed() {
        let mut ctx = ExecutionContext::new();
        assert!(!ctx.is_computed("hidden_0"));
        ctx.mark_computed("hidden_0".to_string());
        assert!(ctx.is_computed("hidden_0"));
    }

    #[test]
    fn execution_plan_from_empty_graph() {
        let graph = FusedGraph {
            nodes: vec![],
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            weight_bindings: HashMap::new(),
            stats: OptimizationStats::default(),
        };
        let plan = ExecutionPlan::from_fused_graph(&graph);
        assert_eq!(plan.op_count(), 0);
        assert_eq!(plan.inputs, vec!["input".to_string()]);
    }

    #[test]
    fn fused_executor_runs_graph_dependencies() {
        let graph = FusedGraph {
            nodes: vec![FusedNode {
                name: "node0".to_string(),
                op: FusedOp::Atomic(AtomicOp::new("Add")),
                inputs: vec!["x".to_string(), "w".to_string()],
                outputs: vec!["y".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            weight_bindings: HashMap::from([(
                "w".to_string(),
                crate::graph::types::WeightBinding {
                    source_name: "w".to_string(),
                    shape: vec![1],
                    dtype: safetensors::Dtype::F32,
                },
            )]),
            stats: OptimizationStats::default(),
        };

        let executor = FusedGraphExecutor::new(graph);
        let outputs = executor
            .run(&HashMap::from([("x".to_string(), vec![0u8; 4])]))
            .unwrap();
        assert!(outputs.contains_key("y"));
    }
}
