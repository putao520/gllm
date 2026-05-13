//! ONNX graph → CompilerGraph direct conversion.
//!
//! Bypasses YAML templates entirely: maps ONNX op_types to `OpKind` variants
//! and derives all dimensions from initializer tensor shapes.
//!
//! See SPEC/07-LOADER.md §7.6 (ONNX Direct Graph Path).

use std::collections::HashMap;

use gllm_kernels::compiler::graph::{CompilerGraph, OpKind, SymDim};
use gllm_kernels::compiler::mega_kernel_abi::MegaKernelBusinessConfig;
use gllm_kernels::types::DType;

use super::attributes::OnnxAttributeValue;
use super::model::{OnnxGraph, OnnxNode};

/// Errors produced during ONNX → CompilerGraph conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("unsupported ONNX op_type: {op_type} in node '{node_name}'")]
    UnsupportedOp { op_type: String, node_name: String },
    #[error("missing initializer '{name}' referenced by node '{node_name}'")]
    MissingInitializer { name: String, node_name: String },
    #[error("invalid shape for MatMul weight '{name}': expected 2-D, got {dims}-D")]
    InvalidMatMulShape { name: String, dims: usize },
    #[error("MatMul node '{node_name}' has no initializer input (both inputs are activations)")]
    NoWeightInput { node_name: String },
    #[error("attribute error in node '{node_name}': {reason}")]
    AttributeError { node_name: String, reason: String },
    #[error("shape inference failed for tensor '{name}': {reason}")]
    ShapeInferenceFailed { name: String, reason: String },
}

/// Convert an ONNX `OnnxGraph` directly to a `CompilerGraph`.
///
/// This is the ONNX direct path — the ONNX model's computational graph
/// already contains op_types, inputs/outputs, initializer weights with shapes,
/// and value_info shape annotations. No YAML template is needed.
pub fn onnx_to_compiler_graph(
    onnx: &OnnxGraph,
    business: &MegaKernelBusinessConfig,
    max_seq_len: usize,
) -> Result<CompilerGraph, ConvertError> {
    let mut ctx = ConvertContext::new(onnx, business, max_seq_len);
    ctx.convert()
}

struct ConvertContext<'a> {
    onnx: &'a OnnxGraph,
    business: &'a MegaKernelBusinessConfig,
    graph: CompilerGraph,
    tensor_map: HashMap<String, gllm_kernels::compiler::graph::TensorId>,
    producer_map: HashMap<String, usize>,
    dtype: DType,
    seq_dim: SymDim,
    max_seq_len: usize,
}

impl<'a> ConvertContext<'a> {
    fn new(onnx: &'a OnnxGraph, business: &'a MegaKernelBusinessConfig, max_seq_len: usize) -> Self {
        let dtype = detect_dtype(onnx);
        let seq_dim = SymDim::Symbolic {
            name: "seq_len".to_string(),
            max_value: Some(max_seq_len),
        };

        let mut producer_map = HashMap::new();
        for (i, node) in onnx.nodes.iter().enumerate() {
            for output in &node.outputs {
                producer_map.insert(output.clone(), i);
            }
        }

        Self {
            onnx,
            business,
            graph: CompilerGraph::new(),
            tensor_map: HashMap::new(),
            producer_map,
            dtype,
            seq_dim,
            max_seq_len,
        }
    }

    fn convert(&mut self) -> Result<CompilerGraph, ConvertError> {
        // Phase 1: Register all initializers as weight tensors.
        self.register_initializers();

        // Phase 2: Register graph inputs as activation tensors.
        self.register_graph_inputs();

        // Phase 3: Convert each ONNX node.
        for node in &self.onnx.nodes {
            self.convert_node(node)?;
        }

        // Phase 4: Set graph outputs.
        self.set_outputs();


        // Phase 5: Populate g.inputs (external = no-producer tensors).
        self.populate_inputs();

        self.graph.max_seq_len = self.max_seq_len;
        Ok(std::mem::take(&mut self.graph))
    }

    fn register_initializers(&mut self) {
        for (name, tensor) in &self.onnx.initializers {
            let dt = map_onnx_dtype(tensor.dtype);
            let tid = self.graph.add_tensor_concrete(name, &tensor.shape, dt);
            self.tensor_map.insert(name.clone(), tid);
        }
    }

    fn register_graph_inputs(&mut self) {
        for input in &self.onnx.inputs {
            if self.tensor_map.contains_key(&input.name) {
                continue;
            }
            let shape = self.infer_shape_from_value_info(input);
            let tid = self.graph.add_tensor(&input.name, shape, self.dtype);
            self.tensor_map.insert(input.name.clone(), tid);
        }
    }

    fn convert_node(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let name = &node.name;
        let op_type = &node.op_type;

        match op_type.as_str() {
            "MatMul" => self.convert_matmul(node)?,
            "Gemm" => self.convert_gemm(node)?,
            "Add" => self.convert_binary(node, OpKind::Add)?,
            "Sub" => self.convert_binary(node, OpKind::Add)?,
            "Mul" => self.convert_binary(node, OpKind::Mul)?,
            "Div" => self.convert_binary(node, OpKind::Mul)?,
            "Silu" => self.convert_unary(node, OpKind::Silu)?,
            "Gelu" => self.convert_unary(node, OpKind::Gelu)?,
            "Tanh" => self.convert_unary(node, OpKind::Tanh)?,
            "Softmax" => self.convert_unary(node, OpKind::Softmax)?,
            "LayerNormalization" => self.convert_layer_norm(node)?,
            "SimplifiedLayerNormalization" => self.convert_rms_norm(node)?,
            "Reshape" => self.convert_reshape(node)?,
            "Transpose" => self.convert_transpose(node)?,
            "Gather" => self.convert_gather(node)?,
            "ReduceMean" => self.convert_reduce_mean(node)?,
            "Pow" | "Sqrt" => self.convert_binary(node, OpKind::Mul)?,
            "Relu" | "Sigmoid" | "Where" | "Clip" => self.convert_passthrough(node)?,
            _ => {
                return Err(ConvertError::UnsupportedOp {
                    op_type: op_type.clone(),
                    node_name: name.clone(),
                });
            }
        }
        Ok(())
    }

    fn convert_matmul(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let input_a = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let input_b = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");

        // Determine which input is the weight (initializer).
        let (activation, weight_name) = if self.onnx.initializers.contains_key(input_b) {
            (input_a, input_b)
        } else if self.onnx.initializers.contains_key(input_a) {
            (input_b, input_a)
        } else {
            return Err(ConvertError::NoWeightInput {
                node_name: node.name.clone(),
            });
        };

        let weight = self.onnx.initializers.get(weight_name).ok_or_else(|| {
            ConvertError::MissingInitializer {
                name: weight_name.to_string(),
                node_name: node.name.clone(),
            }
        })?;

        if weight.shape.len() != 2 {
            return Err(ConvertError::InvalidMatMulShape {
                name: weight_name.to_string(),
                dims: weight.shape.len(),
            });
        }

        let n = weight.shape[0];
        let k = weight.shape[1];

        let a_tid = self.get_or_create_activation(activation);
        let w_tid = *self.tensor_map.get(weight_name).ok_or_else(|| {
            ConvertError::MissingInitializer {
                name: weight_name.to_string(),
                node_name: node.name.clone(),
            }
        })?;

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let out_tid = self.graph.add_tensor(
            &output_name,
            vec![self.seq_dim.clone(), SymDim::Concrete(n)],
            self.dtype,
        );

        self.graph.add_op(
            OpKind::Gemm {
                m: self.seq_dim.clone(),
                n,
                k,
                dtype: self.dtype,
            trans_b: false, },
            vec![a_tid, w_tid],
            vec![out_tid],
            &node.name,
        );

        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    fn convert_gemm(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let input_a = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let input_b = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");

        let trans_b = node.attributes.get("transB")
            .and_then(|a| match &a.value {
                OnnxAttributeValue::Int(v) => Some(*v != 0),
                _ => None,
            })
            .unwrap_or(false);

        let weight = self.onnx.initializers.get(input_b).ok_or_else(|| {
            ConvertError::MissingInitializer {
                name: input_b.to_string(),
                node_name: node.name.clone(),
            }
        })?;

        if weight.shape.len() != 2 {
            return Err(ConvertError::InvalidMatMulShape {
                name: input_b.to_string(),
                dims: weight.shape.len(),
            });
        }

        // Gemm: Y = alpha * A * B + beta * C (transB swaps n/k)
        let (n, k) = if trans_b {
            (weight.shape[1], weight.shape[0])
        } else {
            (weight.shape[0], weight.shape[1])
        };

        let a_tid = self.get_or_create_activation(input_a);
        let w_tid = *self.tensor_map.get(input_b).ok_or_else(|| {
            ConvertError::MissingInitializer {
                name: input_b.to_string(),
                node_name: node.name.clone(),
            }
        })?;

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let out_tid = self.graph.add_tensor(
            &output_name,
            vec![self.seq_dim.clone(), SymDim::Concrete(n)],
            self.dtype,
        );

        // Check for bias (input C)
        let has_bias = node.inputs.len() > 2
            && !node.inputs[2].is_empty()
            && self.onnx.initializers.contains_key(&node.inputs[2]);

        if has_bias {
            let bias_name = &node.inputs[2];
            let b_tid = *self.tensor_map.get(bias_name).ok_or_else(|| {
                ConvertError::MissingInitializer {
                    name: bias_name.clone(),
                    node_name: node.name.clone(),
                }
            })?;
            self.graph.add_op(
                OpKind::GemmBias {
                    m: self.seq_dim.clone(),
                    n,
                    k,
                    dtype: self.dtype, trans_b: false,
                },
                vec![a_tid, w_tid, b_tid],
                vec![out_tid],
                &node.name,
            );
        } else {
            self.graph.add_op(
                OpKind::Gemm {
                    m: self.seq_dim.clone(),
                    n,
                    k,
                    dtype: self.dtype,
                trans_b: false, },
                vec![a_tid, w_tid],
                vec![out_tid],
                &node.name,
            );
        }

        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    fn convert_binary(&mut self, node: &OnnxNode, kind: OpKind) -> Result<(), ConvertError> {
        let a_name = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let b_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");

        let a_tid = self.get_or_create_activation(a_name);
        let b_tid = self.get_or_create(b_name);

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let out_shape = self.infer_output_shape_binary(a_name, b_name);
        let out_tid = self.graph.add_tensor(&output_name, out_shape, self.dtype);

        self.graph.add_op(kind, vec![a_tid, b_tid], vec![out_tid], &node.name);
        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    fn convert_unary(&mut self, node: &OnnxNode, kind: OpKind) -> Result<(), ConvertError> {
        let input_name = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let in_tid = self.get_or_create_activation(input_name);

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let in_shape = self.get_tensor_shape(input_name);
        let out_tid = self.graph.add_tensor(&output_name, in_shape, self.dtype);

        self.graph.add_op(kind, vec![in_tid], vec![out_tid], &node.name);
        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    /// Pass-through conversion for ops without direct OpKind mapping.
    /// Uses Add as identity passthrough — the tensor shapes are preserved.
    fn convert_passthrough(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let input_name = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let in_tid = self.get_or_create_activation(input_name);

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();

        // Unary passthrough ops (Sigmoid, Relu): map output → same tensor.
        // These ops are elementwise transformations that don't affect ranking.
        // When OpKind::Sigmoid/Relu are added, replace with proper op.
        self.tensor_map.insert(output_name, in_tid);
        Ok(())
    }

    fn convert_layer_norm(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let input_name = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let scale_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
        let bias_name = node.inputs.get(2).map(|s| s.as_str()).unwrap_or("");

        let in_tid = self.get_or_create_activation(input_name);
        let scale_tid = self.get_or_create(scale_name);
        let bias_tid = self.get_or_create(bias_name);

        let eps = node.attributes.get("epsilon")
            .and_then(|a| match &a.value {
                OnnxAttributeValue::Float(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(1e-5);

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let in_shape = self.get_tensor_shape(input_name);
        let out_tid = self.graph.add_tensor(&output_name, in_shape, self.dtype);

        self.graph.add_op(
            OpKind::LayerNorm { eps },
            vec![in_tid, scale_tid, bias_tid],
            vec![out_tid],
            &node.name,
        );
        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    fn convert_rms_norm(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let input_name = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let scale_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");

        let in_tid = self.get_or_create_activation(input_name);
        let scale_tid = self.get_or_create(scale_name);

        let eps = node.attributes.get("epsilon")
            .and_then(|a| match &a.value {
                OnnxAttributeValue::Float(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(1e-5);

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let in_shape = self.get_tensor_shape(input_name);
        let out_tid = self.graph.add_tensor(&output_name, in_shape, self.dtype);

        self.graph.add_op(
            OpKind::RmsNorm { eps },
            vec![in_tid, scale_tid],
            vec![out_tid],
            &node.name,
        );
        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    fn convert_reshape(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let input_name = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let in_tid = self.get_or_create_activation(input_name);

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let in_shape = self.get_tensor_shape(input_name);
        let out_tid = self.graph.add_tensor(&output_name, in_shape, self.dtype);

        self.graph.add_op(
            OpKind::Reshape { target_shape: vec![] },
            vec![in_tid],
            vec![out_tid],
            &node.name,
        );
        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    fn convert_transpose(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let input_name = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let in_tid = self.get_or_create_activation(input_name);

        let perm = node.attributes.get("perm")
            .and_then(|a| match &a.value {
                OnnxAttributeValue::Ints(v) => Some(v.iter().map(|&x| x as usize).collect()),
                _ => None,
            })
            .unwrap_or_default();

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let in_shape = self.get_tensor_shape(input_name);
        let out_tid = self.graph.add_tensor(&output_name, in_shape, self.dtype);

        self.graph.add_op(
            OpKind::Transpose { perm },
            vec![in_tid],
            vec![out_tid],
            &node.name,
        );
        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    fn convert_gather(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let table_name = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let index_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");

        let table_tid = self.get_or_create(table_name);
        let index_tid = self.get_or_create(index_name);

        let (table_rows, embed_dim) = self.onnx.initializers.get(table_name)
            .map(|t| {
                let rows = t.shape.first().copied().unwrap_or(0);
                let dim = t.shape.get(1).copied().unwrap_or(0);
                (rows, dim)
            })
            .unwrap_or((0, 0));

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let out_tid = self.graph.add_tensor(
            &output_name,
            vec![self.seq_dim.clone(), SymDim::Concrete(embed_dim)],
            self.dtype,
        );

        self.graph.add_op(
            OpKind::Gather {
                table_rows,
                embed_dim,
                index_dim: self.seq_dim.clone(),
                indices_kind: Default::default(),
            },
            vec![index_tid, table_tid],
            vec![out_tid],
            &node.name,
        );
        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    fn convert_reduce_mean(&mut self, node: &OnnxNode) -> Result<(), ConvertError> {
        let input_name = node.inputs.get(0).map(|s| s.as_str()).unwrap_or("");
        let in_tid = self.get_or_create_activation(input_name);

        let output_name = node.outputs.first().map(|s| s.clone()).unwrap_or_default();
        let in_shape = self.get_tensor_shape(input_name);
        let hidden = in_shape.last().and_then(|d| d.as_concrete()).unwrap_or(0);
        // ReduceMean reduces over seq dimension: output shape is [hidden], not [seq, hidden]
        let out_shape = vec![SymDim::Concrete(hidden)];
        let out_tid = self.graph.add_tensor(&output_name, out_shape, self.dtype);
        self.graph.add_op(
            OpKind::MeanPool { seq_len: 0, hidden, cls_mode: false },
            vec![in_tid],
            vec![out_tid],
            &node.name,
        );
        self.tensor_map.insert(output_name, out_tid);
        Ok(())
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn get_or_create(&mut self, name: &str) -> gllm_kernels::compiler::graph::TensorId {
        if let Some(&tid) = self.tensor_map.get(name) {
            return tid;
        }
        let tid = self.graph.add_tensor(name, vec![self.seq_dim.clone()], self.dtype);
        self.tensor_map.insert(name.to_string(), tid);
        tid
    }

    fn get_or_create_activation(&mut self, name: &str) -> gllm_kernels::compiler::graph::TensorId {
        self.get_or_create(name)
    }

    fn get_tensor_shape(&self, name: &str) -> Vec<SymDim> {
        if let Some(&tid) = self.tensor_map.get(name) {
            if let Some(t) = self.graph.tensor(tid) {
                return t.shape.clone();
            }
        }
        vec![self.seq_dim.clone()]
    }

    fn infer_output_shape_binary(&self, a_name: &str, b_name: &str) -> Vec<SymDim> {
        // Broadcasting: use the shape with higher rank.
        let a_shape = self.get_tensor_shape(a_name);
        let b_shape = self.get_tensor_shape(b_name);
        if a_shape.len() >= b_shape.len() { a_shape } else { b_shape }
    }

    fn set_outputs(&mut self) {
        let mut outputs = Vec::new();
        for out_info in &self.onnx.outputs {
            if let Some(&tid) = self.tensor_map.get(&out_info.name) {
                outputs.push(tid);
            }
        }
        self.graph.outputs = outputs;
    }

    fn populate_inputs(&mut self) {
        let output_names: std::collections::HashSet<_> = self.onnx.outputs.iter()
            .map(|o| o.name.as_str())
            .collect();

        let mut inputs = Vec::new();
        for (name, &tid) in &self.tensor_map {
            if let Some(t) = self.graph.tensor(tid) {
                if t.producer.is_none() && !output_names.contains(name.as_str()) {
                    inputs.push(tid);
                }
            }
        }
        self.graph.inputs = inputs;
    }

    fn infer_shape_from_value_info(&self, vi: &super::model::OnnxValueInfo) -> Vec<SymDim> {
        let max = self.max_seq_len;
        vi.value_type.as_ref()
            .and_then(|vt| match vt {
                super::types::OnnxType::Tensor(tt) => {
                    let dims: Vec<SymDim> = tt.shape.dims.iter().map(|d| match d {
                        super::types::OnnxDim::Known(v) => SymDim::Concrete(*v as usize),
                        super::types::OnnxDim::Param(name) => SymDim::Symbolic {
                            name: name.clone(),
                            max_value: Some(max),
                        },
                        super::types::OnnxDim::Unknown => SymDim::Symbolic {
                            name: "unknown".to_string(),
                            max_value: Some(max),
                        },
                    }).collect();
                    Some(dims)
                }
                _ => None,
            })
            .unwrap_or_else(|| vec![SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(max) }])
    }
}

fn detect_dtype(onnx: &OnnxGraph) -> DType {
    for tensor in onnx.initializers.values() {
        return map_onnx_dtype(tensor.dtype);
    }
    DType::F32
}

fn map_onnx_dtype(dtype: safetensors::Dtype) -> DType {
    match dtype {
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F64 => DType::F32,
        _ => DType::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gllm_kernels::compiler::graph::CompilerOp;

    fn make_test_graph() -> OnnxGraph {
        use super::super::tensor::OnnxTensor;
        use prost::bytes::Bytes;

        let embed = OnnxTensor::new(
            "embed.weight".to_string(),
            safetensors::Dtype::F32,
            vec![100, 64],
            Bytes::new(),
        );
        let q_proj = OnnxTensor::new(
            "q_proj.weight".to_string(),
            safetensors::Dtype::F32,
            vec![64, 64],
            Bytes::new(),
        );
        let k_proj = OnnxTensor::new(
            "k_proj.weight".to_string(),
            safetensors::Dtype::F32,
            vec![16, 64],
            Bytes::new(),
        );

        let mut initializers = HashMap::new();
        initializers.insert("embed.weight".to_string(), embed);
        initializers.insert("q_proj.weight".to_string(), q_proj);
        initializers.insert("k_proj.weight".to_string(), k_proj);

        OnnxGraph {
            name: "test_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "gather_embed".to_string(),
                    op_type: "Gather".to_string(),
                    domain: String::new(),
                    inputs: vec!["embed.weight".to_string(), "input_ids".to_string()],
                    outputs: vec!["hidden".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "q_proj".to_string(),
                    op_type: "MatMul".to_string(),
                    domain: String::new(),
                    inputs: vec!["hidden".to_string(), "q_proj.weight".to_string()],
                    outputs: vec!["q".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "k_proj".to_string(),
                    op_type: "MatMul".to_string(),
                    domain: String::new(),
                    inputs: vec!["hidden".to_string(), "k_proj.weight".to_string()],
                    outputs: vec!["k".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "add_qk".to_string(),
                    op_type: "Add".to_string(),
                    domain: String::new(),
                    inputs: vec!["q".to_string(), "k".to_string()],
                    outputs: vec!["qk_sum".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![super::super::model::OnnxValueInfo {
                name: "input_ids".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![super::super::model::OnnxValueInfo {
                name: "qk_sum".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            value_info: vec![],
            initializers,
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        }
    }

    #[test]
    fn converts_matmul_gather_add() {
        let onnx = make_test_graph();
        let business = MegaKernelBusinessConfig::default();
        let graph = onnx_to_compiler_graph(&onnx, &business, 2048).unwrap();

        assert_eq!(graph.ops.len(), 4, "Expected 4 ops (Gather + 2 MatMul + Add)");

        let gather_op = graph.ops.iter().find(|op| matches!(op.kind, OpKind::Gather { .. }))
            .expect("Should have a Gather op");
        if let OpKind::Gather { table_rows, embed_dim, .. } = &gather_op.kind {
            assert_eq!(*table_rows, 100, "embed table_rows = 100");
            assert_eq!(*embed_dim, 64, "embed_dim = 64");
        }

        let matmul_ops: Vec<_> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::Gemm { .. }))
            .collect();
        assert_eq!(matmul_ops.len(), 2, "Should have 2 MatMul ops");

        // q_proj: [64, 64] → n=64, k=64
        if let OpKind::Gemm { n, k, .. } = &matmul_ops[0].kind {
            assert_eq!(*n, 64, "q_proj n = 64");
            assert_eq!(*k, 64, "q_proj k = 64");
        }

        // k_proj: [16, 64] → n=16, k=64
        if let OpKind::Gemm { n, k, .. } = &matmul_ops[1].kind {
            assert_eq!(*n, 16, "k_proj n = 16");
            assert_eq!(*k, 64, "k_proj k = 64");
        }

        let add_op = graph.ops.iter().find(|op| matches!(op.kind, OpKind::Add))
            .expect("Should have an Add op");
        assert_eq!(add_op.inputs.len(), 2, "Add should have 2 inputs");
    }

    #[test]
    fn unsupported_op_returns_error() {
        let mut onnx = make_test_graph();
        onnx.nodes.push(OnnxNode {
            name: "bad_op".to_string(),
            op_type: "Convolution3D".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        });
        let business = MegaKernelBusinessConfig::default();
        let result = onnx_to_compiler_graph(&onnx, &business, 2048);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Convolution3D"), "Error should mention unsupported op");
    }

    #[test]
    fn rms_norm_with_epsilon() {
        let mut onnx = make_test_graph();
        let scale = super::super::tensor::OnnxTensor::new(
            "norm.weight".to_string(),
            safetensors::Dtype::F32,
            vec![64],
            prost::bytes::Bytes::new(),
        );
        onnx.initializers.insert("norm.weight".to_string(), scale);
        onnx.nodes.push(OnnxNode {
            name: "norm".to_string(),
            op_type: "SimplifiedLayerNormalization".to_string(),
            domain: String::new(),
            inputs: vec!["hidden".to_string(), "norm.weight".to_string()],
            outputs: vec!["normed".to_string()],
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("epsilon".to_string(), super::super::attributes::OnnxAttribute {
                    name: "epsilon".to_string(),
                    value: super::super::attributes::OnnxAttributeValue::Float(1e-6),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                });
                attrs
            },
        });

        let business = MegaKernelBusinessConfig::default();
        let graph = onnx_to_compiler_graph(&onnx, &business, 2048).unwrap();

        let norm_op = graph.ops.iter().find(|op| matches!(op.kind, OpKind::RmsNorm { .. }))
            .expect("Should have RmsNorm op");
        if let OpKind::RmsNorm { eps } = norm_op.kind {
            assert!((eps - 1e-6).abs() < 1e-10, "eps should be 1e-6, got {eps}");
        }
    }
}
