use std::collections::HashSet;

use super::super::model::{OnnxGraph, OnnxNode};
use super::super::Result;
use super::index::{is_linear_value, GraphIndex};
use super::{FusedKernel, FusedOp, SwiGluSpec};

pub(super) fn match_swiglu(
    graph: &OnnxGraph,
    index: &GraphIndex,
    consumed: &mut HashSet<usize>,
) -> Result<Vec<FusedOp>> {
    let mut ops = Vec::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        if consumed.contains(&id) || node.op_type != "Mul" {
            continue;
        }
        let Some((silu_id, gate_input, up_input)) = find_swiglu_inputs(graph, index, node) else {
            continue;
        };
        if consumed.contains(&silu_id) || !is_linear_value(graph, index, &gate_input) {
            continue;
        }
        if !is_linear_value(graph, index, &up_input) {
            continue;
        }
        let output = match node.outputs.get(0) {
            Some(value) => value.clone(),
            None => continue,
        };
        consumed.insert(id);
        consumed.insert(silu_id);
        ops.push(FusedOp {
            kind: FusedKernel::SwiGlu(SwiGluSpec {
                gate: gate_input,
                up: up_input,
                output,
            }),
            nodes: vec![id, silu_id],
        });
    }
    Ok(ops)
}

fn find_swiglu_inputs(
    graph: &OnnxGraph,
    index: &GraphIndex,
    node: &OnnxNode,
) -> Option<(usize, String, String)> {
    let first_input = node.inputs.get(0)?;
    let second_input = node.inputs.get(1)?;
    let first = silu_source(graph, index, first_input, second_input);
    let second = silu_source(graph, index, second_input, first_input);
    first.or(second)
}

fn silu_source(
    graph: &OnnxGraph,
    index: &GraphIndex,
    silu_output: &str,
    other_input: &str,
) -> Option<(usize, String, String)> {
    let silu_id = index.producer(silu_output)?;
    let silu_node = &graph.nodes[silu_id];
    if !is_silu_op(silu_node) {
        return None;
    }
    let gate_input = silu_node.inputs.get(0)?.clone();
    Some((silu_id, gate_input, other_input.to_string()))
}

fn is_silu_op(node: &OnnxNode) -> bool {
    node.op_type == "Silu" || node.op_type == "SiLU"
}
