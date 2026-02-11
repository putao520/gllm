//! 常量折叠 Pass (REQ-OPT-008)

use std::collections::HashMap;

use super::pass::{OptimizationContext, OptimizationPass};
use super::OptimizeError;
use crate::graph::types::{FusedGraph, FusedNode, FusedOp, WeightBinding};

#[derive(Debug)]
pub struct ConstantFoldingPass;

impl OptimizationPass for ConstantFoldingPass {
    fn name(&self) -> &'static str {
        "ConstantFolding"
    }

    fn run(
        &self,
        mut graph: FusedGraph,
        _ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError> {
        let mut constants = graph.weight_bindings.clone();
        let mut replacements: HashMap<String, WeightBinding> = HashMap::new();

        let mut kept_nodes = Vec::with_capacity(graph.nodes.len());
        let mut folded_count = 0usize;

        for node in graph.nodes {
            if !is_atomic_foldable(&node) {
                kept_nodes.push(node);
                continue;
            }
            if node.inputs.is_empty() || node.outputs.is_empty() {
                kept_nodes.push(node);
                continue;
            }
            if !node.inputs.iter().all(|input| constants.contains_key(input)) {
                kept_nodes.push(node);
                continue;
            }

            if let Some(binding) = try_fold_node(&node, &constants) {
                let out_name = node.outputs[0].clone();
                constants.insert(out_name.clone(), binding.clone());
                replacements.insert(out_name, binding);
                folded_count += 1;
            } else {
                kept_nodes.push(node);
            }
        }

        graph.nodes = kept_nodes;
        graph.weight_bindings.extend(replacements);
        graph.stats.constant_folded_nodes = folded_count;

        Ok(graph)
    }

    fn priority(&self) -> i32 {
        0
    }
}

fn is_atomic_foldable(node: &FusedNode) -> bool {
    matches!(
        &node.op,
        FusedOp::Atomic(op)
            if matches!(
                op.op_type.as_str(),
                "Add" | "Mul" | "Div" | "Reshape" | "Transpose"
            )
    )
}

fn try_fold_node(node: &FusedNode, constants: &HashMap<String, WeightBinding>) -> Option<WeightBinding> {
    let FusedOp::Atomic(op) = &node.op else {
        return None;
    };

    match op.op_type.as_str() {
        "Add" | "Mul" | "Div" => fold_binary_numeric(node, constants, op.op_type.as_str()),
        "Reshape" => fold_reshape(node, constants),
        "Transpose" => fold_transpose(node, constants),
        _ => None,
    }
}

fn fold_binary_numeric(
    node: &FusedNode,
    constants: &HashMap<String, WeightBinding>,
    op_type: &str,
) -> Option<WeightBinding> {
    if node.inputs.len() < 2 {
        return None;
    }
    let lhs = constants.get(&node.inputs[0])?;
    let rhs = constants.get(&node.inputs[1])?;
    let lhs_data = lhs.data.as_ref()?;
    let rhs_data = rhs.data.as_ref()?;

    if lhs.dtype != safetensors::Dtype::F32 || rhs.dtype != safetensors::Dtype::F32 {
        return None;
    }
    if lhs.shape != rhs.shape {
        return None;
    }
    if lhs_data.len() != rhs_data.len() || lhs_data.len() % 4 != 0 {
        return None;
    }

    let mut out = vec![0u8; lhs_data.len()];
    for idx in (0..lhs_data.len()).step_by(4) {
        let a = f32::from_le_bytes(lhs_data[idx..idx + 4].try_into().ok()?);
        let b = f32::from_le_bytes(rhs_data[idx..idx + 4].try_into().ok()?);
        let value = match op_type {
            "Add" => a + b,
            "Mul" => a * b,
            "Div" => {
                if b == 0.0 {
                    return None;
                }
                a / b
            }
            _ => return None,
        };
        out[idx..idx + 4].copy_from_slice(&value.to_le_bytes());
    }

    Some(WeightBinding {
        source_name: node.outputs[0].clone(),
        shape: lhs.shape.clone(),
        dtype: lhs.dtype,
        data: Some(out),
    })
}

fn fold_reshape(
    node: &FusedNode,
    constants: &HashMap<String, WeightBinding>,
) -> Option<WeightBinding> {
    if node.inputs.len() < 2 {
        return None;
    }
    let src = constants.get(&node.inputs[0])?;
    let src_data = src.data.clone()?;
    let shape_binding = constants.get(&node.inputs[1])?;
    let shape_values = decode_shape_values(shape_binding)?;
    let target_shape = resolve_reshape_shape(&src.shape, &shape_values)?;

    Some(WeightBinding {
        source_name: node.outputs[0].clone(),
        shape: target_shape,
        dtype: src.dtype,
        data: Some(src_data),
    })
}

fn fold_transpose(
    node: &FusedNode,
    constants: &HashMap<String, WeightBinding>,
) -> Option<WeightBinding> {
    let src = constants.get(node.inputs.first()?)?;
    let src_data = src.data.as_ref()?;
    let rank = src.shape.len();
    if rank == 0 {
        return Some(WeightBinding {
            source_name: node.outputs[0].clone(),
            shape: vec![],
            dtype: src.dtype,
            data: Some(src_data.clone()),
        });
    }

    let perm = match node.attributes.get("perm") {
        Some(crate::graph::types::AttrValue::Ints(values)) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                out.push(usize::try_from(*value).ok()?);
            }
            out
        }
        _ => (0..rank).rev().collect(),
    };

    if perm.len() != rank {
        return None;
    }
    let mut seen = vec![false; rank];
    for &axis in &perm {
        if axis >= rank || seen[axis] {
            return None;
        }
        seen[axis] = true;
    }

    let element_size = src.dtype.size();
    if src_data.len() % element_size != 0 {
        return None;
    }

    let out_shape: Vec<usize> = perm.iter().map(|&axis| src.shape[axis]).collect();
    let element_count = if src.shape.is_empty() {
        1
    } else {
        src.shape.iter().product()
    };
    if element_count * element_size != src_data.len() {
        return None;
    }

    let in_strides = compute_strides(&src.shape);
    let out_strides = compute_strides(&out_shape);
    let mut out_data = vec![0u8; src_data.len()];

    for out_linear in 0..element_count {
        let out_coord = linear_to_coord(out_linear, &out_shape, &out_strides);
        let mut in_coord = vec![0usize; rank];
        for (out_axis, &in_axis) in perm.iter().enumerate() {
            in_coord[in_axis] = out_coord[out_axis];
        }
        let in_linear = coord_to_linear(&in_coord, &in_strides);

        let in_offset = in_linear * element_size;
        let out_offset = out_linear * element_size;
        out_data[out_offset..out_offset + element_size]
            .copy_from_slice(&src_data[in_offset..in_offset + element_size]);
    }

    Some(WeightBinding {
        source_name: node.outputs[0].clone(),
        shape: out_shape,
        dtype: src.dtype,
        data: Some(out_data),
    })
}

fn decode_shape_values(shape: &WeightBinding) -> Option<Vec<i64>> {
    let data = shape.data.as_ref()?;
    match shape.dtype {
        safetensors::Dtype::I64 => {
            if data.len() % 8 != 0 {
                return None;
            }
            let mut out = Vec::with_capacity(data.len() / 8);
            for chunk in data.chunks_exact(8) {
                out.push(i64::from_le_bytes(chunk.try_into().ok()?));
            }
            Some(out)
        }
        safetensors::Dtype::I32 => {
            if data.len() % 4 != 0 {
                return None;
            }
            let mut out = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                out.push(i32::from_le_bytes(chunk.try_into().ok()?) as i64);
            }
            Some(out)
        }
        _ => None,
    }
}

fn resolve_reshape_shape(input_shape: &[usize], target_spec: &[i64]) -> Option<Vec<usize>> {
    if target_spec.is_empty() {
        return Some(vec![]);
    }
    let mut out = Vec::with_capacity(target_spec.len());
    let mut infer_at = None;

    for (idx, &dim) in target_spec.iter().enumerate() {
        match dim {
            -1 => {
                if infer_at.is_some() {
                    return None;
                }
                infer_at = Some(idx);
                out.push(1);
            }
            0 => {
                let copied = *input_shape.get(idx)?;
                out.push(copied);
            }
            d if d > 0 => out.push(usize::try_from(d).ok()?),
            _ => return None,
        }
    }

    let in_elements = if input_shape.is_empty() {
        1usize
    } else {
        input_shape.iter().product()
    };

    let known_product = if out.is_empty() {
        1usize
    } else {
        out.iter().product()
    };

    if let Some(index) = infer_at {
        if known_product == 0 || !in_elements.is_multiple_of(known_product) {
            return None;
        }
        out[index] = in_elements / known_product;
    }

    let out_elements = if out.is_empty() {
        1usize
    } else {
        out.iter().product()
    };
    if out_elements != in_elements {
        return None;
    }

    Some(out)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn linear_to_coord(linear: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut remaining = linear;
    let mut coord = vec![0usize; shape.len()];
    for axis in 0..shape.len() {
        let stride = strides[axis];
        coord[axis] = remaining / stride;
        remaining %= stride;
    }
    coord
}

fn coord_to_linear(coord: &[usize], strides: &[usize]) -> usize {
    coord.iter().zip(strides.iter()).map(|(c, s)| c * s).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::{AtomicOp, FusedNode, OptimizationStats};

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn folds_add_node() {
        let node = FusedNode {
            name: "add0".to_string(),
            op: FusedOp::Atomic(AtomicOp::new("Add")),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            attributes: HashMap::new(),
        };

        let graph = FusedGraph {
            nodes: vec![node],
            inputs: vec![],
            outputs: vec!["c".to_string()],
            weight_bindings: HashMap::from([
                (
                    "a".to_string(),
                    WeightBinding {
                        source_name: "a".to_string(),
                        shape: vec![2],
                        dtype: safetensors::Dtype::F32,
                        data: Some(f32_bytes(&[1.0, 2.0])),
                    },
                ),
                (
                    "b".to_string(),
                    WeightBinding {
                        source_name: "b".to_string(),
                        shape: vec![2],
                        dtype: safetensors::Dtype::F32,
                        data: Some(f32_bytes(&[3.0, 4.0])),
                    },
                ),
            ]),
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
            stats: OptimizationStats::default(),
        };

        let pass = ConstantFoldingPass;
        let out = pass.run(graph, &OptimizationContext::default()).unwrap();

        assert_eq!(out.stats.constant_folded_nodes, 1);
        assert!(out.nodes.is_empty());

        let folded = out.weight_bindings.get("c").expect("folded binding");
        let bytes = folded.data.as_ref().expect("folded data");
        let got: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(got, vec![4.0, 6.0]);
    }

    #[test]
    fn folds_reshape_node() {
        let node = FusedNode {
            name: "reshape0".to_string(),
            op: FusedOp::Atomic(AtomicOp::new("Reshape")),
            inputs: vec!["x".to_string(), "shape".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };

        let graph = FusedGraph {
            nodes: vec![node],
            inputs: vec![],
            outputs: vec!["y".to_string()],
            weight_bindings: HashMap::from([
                (
                    "x".to_string(),
                    WeightBinding {
                        source_name: "x".to_string(),
                        shape: vec![2, 2],
                        dtype: safetensors::Dtype::F32,
                        data: Some(f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
                    },
                ),
                (
                    "shape".to_string(),
                    WeightBinding {
                        source_name: "shape".to_string(),
                        shape: vec![2],
                        dtype: safetensors::Dtype::I64,
                        data: Some(vec![4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
                    },
                ),
            ]),
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
            stats: OptimizationStats::default(),
        };

        let pass = ConstantFoldingPass;
        let out = pass.run(graph, &OptimizationContext::default()).unwrap();
        let folded = out.weight_bindings.get("y").expect("folded binding");
        assert_eq!(folded.shape, vec![4, 1]);
    }

    #[test]
    fn folds_transpose_node() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "perm".to_string(),
            crate::graph::types::AttrValue::Ints(vec![1, 0]),
        );
        let node = FusedNode {
            name: "transpose0".to_string(),
            op: FusedOp::Atomic(AtomicOp::new("Transpose")),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: attrs,
        };

        let graph = FusedGraph {
            nodes: vec![node],
            inputs: vec![],
            outputs: vec!["y".to_string()],
            weight_bindings: HashMap::from([(
                "x".to_string(),
                WeightBinding {
                    source_name: "x".to_string(),
                    shape: vec![2, 3],
                    dtype: safetensors::Dtype::F32,
                    data: Some(f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
                },
            )]),
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
            stats: OptimizationStats::default(),
        };

        let pass = ConstantFoldingPass;
        let out = pass.run(graph, &OptimizationContext::default()).unwrap();
        let folded = out.weight_bindings.get("y").expect("folded binding");
        assert_eq!(folded.shape, vec![3, 2]);
        let bytes = folded.data.as_ref().expect("folded data");
        let got: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(got, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
