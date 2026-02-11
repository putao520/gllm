use gllm::arch::{ArchTemplate, ResolvedConfig};
use gllm::graph::executor::FusedGraphExecutor;
use gllm::graph::optimizer::{GraphOptimizer, OptimizationContext};
use gllm::graph::types::{FusedGraph, FusedNode, FusedOp};
use gllm::loader::{TensorMeta, TensorProvider};
use safetensors::Dtype;
use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Debug)]
struct MockProvider {
    tensors: Vec<TensorMeta>,
}

impl TensorProvider for MockProvider {
    fn tensor_info(&self, name: &str) -> Option<TensorMeta> {
        self.tensors.iter().find(|t| t.name == name).cloned()
    }

    fn iter_tensors(&self) -> impl Iterator<Item = TensorMeta> {
        self.tensors.clone().into_iter()
    }

    fn load_tensor_data(&self, _name: &str) -> gllm::loader::Result<Cow<'_, [u8]>> {
        Ok(Cow::Borrowed(&[]))
    }
}

#[test]
fn dag_refactor_template_to_graph_and_optimizer() {
    let yaml = r#"
name: qwen3_test
graph:
  inputs:
    - name: input_ids
      dtype: int64
  outputs:
    - name: logits
      dtype: f16
  nodes:
    - name: q_proj
      op_type: MatMul
      inputs: [x, q_w]
      outputs: [q]
    - name: k_proj
      op_type: MatMul
      inputs: [x, k_w]
      outputs: [k]
    - name: v_proj
      op_type: MatMul
      inputs: [x, v_w]
      outputs: [v]
    - name: rope
      op_type: RotaryEmbedding
      inputs: [q, k]
      outputs: [qk]
"#;
    let template = ArchTemplate::from_yaml(yaml).unwrap();
    let mut cfg = ResolvedConfig::default();
    cfg.num_hidden_layers = 1;
    cfg.hidden_size = 128;
    cfg.num_attention_heads = 8;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 16;
    cfg.vocab_size = 1000;
    cfg.dtype = "f16".to_string();

    let graph = template.to_onnx_graph(&cfg).unwrap();
    let optimizer = GraphOptimizer::new(OptimizationContext::cuda((9, 0)));
    let fused = optimizer.optimize(&graph).unwrap();
    assert!(fused
        .nodes
        .iter()
        .any(|n| matches!(n.op, FusedOp::FusedQkvRope(_))));
}

#[test]
fn dag_refactor_bind_weights_and_execute() {
    let mut graph = FusedGraph::new();
    graph.inputs = vec!["x".to_string()];
    graph.outputs = vec!["y".to_string()];
    graph.nodes.push(
        FusedNode::new(
            "add",
            FusedOp::Atomic(gllm::graph::types::AtomicOp::new("Add")),
        )
        .with_inputs(vec!["x".to_string(), "w".to_string()])
        .with_outputs(vec!["y".to_string()]),
    );

    let provider = MockProvider {
        tensors: vec![TensorMeta {
            name: "w".to_string(),
            shape: vec![1],
            dtype: Dtype::F32,
        }],
    };
    assert_eq!(graph.bind_weights(&provider), 1);
    assert!(graph.weight_bindings.contains_key("w"));

    let executor = FusedGraphExecutor::new(graph);
    let outputs = executor
        .run(&HashMap::from([("x".to_string(), vec![0u8; 4])]))
        .unwrap();
    assert!(outputs.contains_key("y"));
}
