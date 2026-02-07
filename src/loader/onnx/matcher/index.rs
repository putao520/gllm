use std::collections::{HashMap, HashSet};

use super::super::attributes::OnnxAttributeValue;
use super::super::model::{OnnxGraph, OnnxNode};
use super::{AtomicOp, FusedKernel, FusedOp};

pub(super) struct GraphIndex {
    producers: HashMap<String, usize>,
    consumers: HashMap<String, Vec<usize>>,
}

impl GraphIndex {
    pub(super) fn new(graph: &OnnxGraph) -> Self {
        let mut producers: HashMap<String, usize> = HashMap::new();
        let mut consumers: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, node) in graph.nodes.iter().enumerate() {
            for output in &node.outputs {
                if output.is_empty() {
                    continue;
                }
                producers.insert(output.clone(), idx);
            }
            for input in &node.inputs {
                if input.is_empty() {
                    continue;
                }
                consumers.entry(input.clone()).or_default().push(idx);
            }
        }
        Self {
            producers,
            consumers,
        }
    }

    pub(super) fn producer(&self, value: &str) -> Option<usize> {
        self.producers.get(value).copied()
    }

    pub(super) fn consumers(&self, value: &str) -> Option<&[usize]> {
        self.consumers.get(value).map(Vec::as_slice)
    }
}

pub(super) fn collect_atomic_ops(graph: &OnnxGraph, consumed: &HashSet<usize>) -> Vec<FusedOp> {
    let mut ops = Vec::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        if consumed.contains(&id) {
            continue;
        }
        ops.push(FusedOp {
            kind: FusedKernel::Atomic(AtomicOp {
                op_type: node.op_type.clone(),
                domain: node.domain.clone(),
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
            }),
            nodes: vec![id],
        });
    }
    ops
}

pub(super) fn single_consumer(index: &GraphIndex, value: &str) -> Option<usize> {
    let consumers = index.consumers(value)?;
    if consumers.len() == 1 {
        Some(consumers[0])
    } else {
        None
    }
}

pub(super) fn other_input(node: &OnnxNode, known: &str) -> Option<String> {
    if node.inputs.len() < 2 {
        return None;
    }
    if node.inputs[0] == known {
        return Some(node.inputs[1].clone());
    }
    if node.inputs[1] == known {
        return Some(node.inputs[0].clone());
    }
    None
}

pub(super) fn is_linear_value(graph: &OnnxGraph, index: &GraphIndex, value: &str) -> bool {
    let Some(node_id) = index.producer(value) else {
        return false;
    };
    let node = &graph.nodes[node_id];
    if node.op_type == "MatMul" || node.op_type == "Gemm" {
        return true;
    }
    if node.op_type != "Add" {
        return false;
    }
    let left = node.inputs.first().map(String::as_str);
    let right = node.inputs.get(1).map(String::as_str);
    match (left, right) {
        (Some(a), Some(b)) => {
            (is_matmul_or_gemm(graph, index, a) && is_constant_value(graph, index, b))
                || (is_matmul_or_gemm(graph, index, b) && is_constant_value(graph, index, a))
        }
        _ => false,
    }
}

pub(super) fn is_matmul_or_gemm(graph: &OnnxGraph, index: &GraphIndex, value: &str) -> bool {
    let Some(node_id) = index.producer(value) else {
        return false;
    };
    let op = graph.nodes[node_id].op_type.as_str();
    op == "MatMul" || op == "Gemm"
}

pub(super) fn is_constant_value(graph: &OnnxGraph, index: &GraphIndex, value: &str) -> bool {
    graph.initializers.contains_key(value) || is_constant_node(graph, index, value)
}

pub(super) fn constant_scalar(graph: &OnnxGraph, index: &GraphIndex, value: &str) -> Option<f32> {
    if let Some(tensor) = graph.initializers.get(value) {
        return tensor.scalar_f32();
    }
    let node_id = index.producer(value)?;
    let node = &graph.nodes[node_id];
    if node.op_type != "Constant" {
        return None;
    }
    extract_constant_attr(node, "value")
        .or_else(|| extract_constant_attr(node, "value_float"))
        .or_else(|| extract_constant_attr(node, "value_int"))
        .or_else(|| extract_constant_attr(node, "value_floats"))
        .or_else(|| extract_constant_attr(node, "value_ints"))
}

fn is_constant_node(graph: &OnnxGraph, index: &GraphIndex, value: &str) -> bool {
    let Some(node_id) = index.producer(value) else {
        return false;
    };
    graph.nodes[node_id].op_type == "Constant"
}

fn extract_constant_attr(node: &OnnxNode, name: &str) -> Option<f32> {
    let attr = node.attributes.get(name)?;
    match &attr.value {
        OnnxAttributeValue::Tensor(tensor) => tensor.scalar_f32(),
        OnnxAttributeValue::Float(value) => Some(*value),
        OnnxAttributeValue::Int(value) => Some(*value as f32),
        OnnxAttributeValue::Floats(values) => values.first().copied(),
        OnnxAttributeValue::Ints(values) => values.first().map(|value| *value as f32),
        _ => None,
    }
}
