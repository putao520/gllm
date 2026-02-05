use std::collections::HashSet;

use super::index::{constant_scalar, other_input, single_consumer, GraphIndex};
use super::{FlashAttentionSpec, FusedKernel, FusedOp};
use super::super::model::{OnnxGraph, OnnxNode};
use super::super::Result;

pub(super) fn match_attention(
    graph: &OnnxGraph,
    index: &GraphIndex,
    consumed: &mut HashSet<usize>,
) -> Result<Vec<FusedOp>> {
    let mut ops = Vec::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        if consumed.contains(&id) || !is_attention_op(node) {
            continue;
        }
        if let Some(spec) = flash_from_attention_node(node) {
            consumed.insert(id);
            ops.push(FusedOp {
                kind: FusedKernel::FlashAttention(spec),
                nodes: vec![id],
            });
        }
    }
    for (id, node) in graph.nodes.iter().enumerate() {
        if consumed.contains(&id) || node.op_type != "Softmax" {
            continue;
        }
        if let Some((spec, node_ids)) = flash_from_softmax(graph, index, id, consumed) {
            for node_id in &node_ids {
                consumed.insert(*node_id);
            }
            ops.push(FusedOp {
                kind: FusedKernel::FlashAttention(spec),
                nodes: node_ids,
            });
        }
    }
    Ok(ops)
}

fn flash_from_attention_node(node: &OnnxNode) -> Option<FlashAttentionSpec> {
    let q = node.inputs.get(0)?.clone();
    let k = node.inputs.get(1)?.clone();
    let v = node.inputs.get(2)?.clone();
    let output = node.outputs.get(0)?.clone();
    let causal = attr_i64(node, "unidirectional").map(|value| value != 0);
    let scale = attr_f32(node, "scale");
    Some(FlashAttentionSpec {
        q,
        k,
        v,
        output,
        scale,
        causal,
    })
}

fn flash_from_softmax(
    graph: &OnnxGraph,
    index: &GraphIndex,
    softmax_id: usize,
    consumed: &HashSet<usize>,
) -> Option<(FlashAttentionSpec, Vec<usize>)> {
    let softmax = &graph.nodes[softmax_id];
    let score_input = softmax.inputs.get(0)?.clone();
    let (score_value, scale, mut extra_nodes) = unwrap_scale(graph, index, &score_input);
    let score_node_id = index.producer(&score_value)?;
    if consumed.contains(&score_node_id) {
        return None;
    }
    let score_node = &graph.nodes[score_node_id];
    if score_node.op_type != "MatMul" {
        return None;
    }
    let q = score_node.inputs.get(0)?.clone();
    let k = score_node.inputs.get(1)?.clone();
    let softmax_out = softmax.outputs.get(0)?.clone();
    let out_node_id = single_consumer(index, &softmax_out)?;
    if consumed.contains(&out_node_id) {
        return None;
    }
    let out_node = &graph.nodes[out_node_id];
    if out_node.op_type != "MatMul" {
        return None;
    }
    let v = other_input(out_node, &softmax_out)?;
    let output = out_node.outputs.get(0)?.clone();
    extra_nodes.push(score_node_id);
    extra_nodes.push(softmax_id);
    extra_nodes.push(out_node_id);
    Some((
        FlashAttentionSpec {
            q,
            k,
            v,
            output,
            scale,
            causal: None,
        },
        extra_nodes,
    ))
}

fn unwrap_scale(
    graph: &OnnxGraph,
    index: &GraphIndex,
    value: &str,
) -> (String, Option<f32>, Vec<usize>) {
    let Some(node_id) = index.producer(value) else {
        return (value.to_string(), None, Vec::new());
    };
    let node = &graph.nodes[node_id];
    if node.op_type != "Mul" && node.op_type != "Div" {
        return (value.to_string(), None, Vec::new());
    }
    if node.inputs.len() < 2 {
        return (value.to_string(), None, Vec::new());
    }
    if let Some(scale) = constant_scalar(graph, index, &node.inputs[1]) {
        return (
            node.inputs[0].clone(),
            adjust_scale(node.op_type.as_str(), scale),
            vec![node_id],
        );
    }
    if let Some(scale) = constant_scalar(graph, index, &node.inputs[0]) {
        return (
            node.inputs[1].clone(),
            adjust_scale(node.op_type.as_str(), scale),
            vec![node_id],
        );
    }
    (value.to_string(), None, Vec::new())
}

fn adjust_scale(op: &str, scale: f32) -> Option<f32> {
    if scale == 0.0 {
        return None;
    }
    if op == "Div" {
        return Some(1.0 / scale);
    }
    Some(scale)
}

fn is_attention_op(node: &OnnxNode) -> bool {
    matches!(
        node.op_type.as_str(),
        "Attention" | "MultiHeadAttention" | "ScaledDotProductAttention"
    )
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
