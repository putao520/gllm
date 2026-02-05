use std::collections::HashSet;

use super::index::{is_constant_value, GraphIndex};
use super::{FusedKernel, FusedOp, FusedQkvRopeSpec, RopeSpec};
use super::super::model::{OnnxGraph, OnnxNode};
use super::super::Result;

pub(super) fn match_rope(
    graph: &OnnxGraph,
    index: &GraphIndex,
    consumed: &mut HashSet<usize>,
) -> Result<Vec<FusedOp>> {
    let mut ops = Vec::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        if consumed.contains(&id) || !is_rope_op(node) {
            continue;
        }
        let input = match node.inputs.get(0) {
            Some(value) => value.clone(),
            None => continue,
        };
        let output = match node.outputs.get(0) {
            Some(value) => value.clone(),
            None => continue,
        };
        let attrs = rope_attrs(node);
        if let Some((spec, extra)) = fused_qkv_rope_candidate(graph, index, &input, &output, attrs) {
            let mut nodes = vec![id];
            nodes.extend(extra);
            for node_id in &nodes {
                consumed.insert(*node_id);
            }
            ops.push(FusedOp {
                kind: FusedKernel::FusedQkvRope(spec),
                nodes,
            });
            continue;
        }
        consumed.insert(id);
        ops.push(FusedOp {
            kind: FusedKernel::Rope(RopeSpec {
                input,
                output,
                rotary_dim: attrs.rotary_dim,
                base: attrs.base,
                scale: attrs.scale,
                interleaved: attrs.interleaved,
            }),
            nodes: vec![id],
        });
    }
    Ok(ops)
}

fn fused_qkv_rope_candidate(
    graph: &OnnxGraph,
    index: &GraphIndex,
    rope_input: &str,
    rope_output: &str,
    attrs: RopeAttrs,
) -> Option<(FusedQkvRopeSpec, Vec<usize>)> {
    let source = linear_qkv_source(graph, index, rope_input)?;
    if !is_fused_qkv_weight(graph, &source.weight) {
        return None;
    }
    let mut nodes = source.extra_nodes;
    nodes.push(source.matmul_id);
    Some((
        FusedQkvRopeSpec {
            input: source.input,
            weight: source.weight,
            bias: source.bias,
            output: rope_output.to_string(),
            rotary_dim: attrs.rotary_dim,
            base: attrs.base,
            scale: attrs.scale,
            interleaved: attrs.interleaved,
        },
        nodes,
    ))
}

fn linear_qkv_source(
    graph: &OnnxGraph,
    index: &GraphIndex,
    value: &str,
) -> Option<LinearSource> {
    let producer = index.producer(value)?;
    let node = &graph.nodes[producer];
    if node.op_type == "Add" && node.inputs.len() >= 2 {
        let left = node.inputs.get(0).map(String::as_str)?;
        let right = node.inputs.get(1).map(String::as_str)?;
        if let Some(source) = linear_qkv_source(graph, index, left) {
            if is_constant_value(graph, index, right) {
                return Some(source.with_bias(right.to_string(), producer));
            }
        }
        if let Some(source) = linear_qkv_source(graph, index, right) {
            if is_constant_value(graph, index, left) {
                return Some(source.with_bias(left.to_string(), producer));
            }
        }
        return None;
    }
    if node.op_type != "MatMul" && node.op_type != "Gemm" {
        return None;
    }
    if node.inputs.len() < 2 {
        return None;
    }
    let (input, weight) = if graph.initializers.contains_key(&node.inputs[0]) {
        (node.inputs[1].clone(), node.inputs[0].clone())
    } else if graph.initializers.contains_key(&node.inputs[1]) {
        (node.inputs[0].clone(), node.inputs[1].clone())
    } else {
        return None;
    };
    Some(LinearSource {
        matmul_id: producer,
        input,
        weight,
        bias: None,
        extra_nodes: Vec::new(),
    })
}

fn is_fused_qkv_weight(graph: &OnnxGraph, name: &str) -> bool {
    let Some(tensor) = graph.initializers.get(name) else {
        return false;
    };
    if tensor.shape.len() != 2 {
        return false;
    }
    tensor.shape.iter().any(|dim| dim % 3 == 0)
}

fn is_rope_op(node: &OnnxNode) -> bool {
    matches!(node.op_type.as_str(), "RotaryEmbedding" | "RoPE")
}

fn rope_attrs(node: &OnnxNode) -> RopeAttrs {
    RopeAttrs {
        rotary_dim: attr_i64(node, "rotary_dim"),
        base: attr_f32(node, "base"),
        scale: attr_f32(node, "scale"),
        interleaved: attr_i64(node, "interleaved"),
    }
}

fn attr_i64(node: &OnnxNode, name: &str) -> Option<i64> {
    let attr = node.attributes.get(name)?;
    match &attr.value {
        super::super::attributes::OnnxAttributeValue::Int(value) => Some(*value),
        super::super::attributes::OnnxAttributeValue::Ints(values) => values.first().copied(),
        _ => None,
    }
}

fn attr_f32(node: &OnnxNode, name: &str) -> Option<f32> {
    let attr = node.attributes.get(name)?;
    match &attr.value {
        super::super::attributes::OnnxAttributeValue::Float(value) => Some(*value),
        super::super::attributes::OnnxAttributeValue::Floats(values) => values.first().copied(),
        super::super::attributes::OnnxAttributeValue::Int(value) => Some(*value as f32),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct RopeAttrs {
    rotary_dim: Option<i64>,
    base: Option<f32>,
    scale: Option<f32>,
    interleaved: Option<i64>,
}

struct LinearSource {
    matmul_id: usize,
    input: String,
    weight: String,
    bias: Option<String>,
    extra_nodes: Vec<usize>,
}

impl LinearSource {
    fn with_bias(mut self, bias: String, add_node: usize) -> Self {
        self.bias = Some(bias);
        self.extra_nodes.push(add_node);
        self
    }
}
