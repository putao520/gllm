//! 优化器模块 (REQ-OPT-001 ~ REQ-OPT-004)

mod dead_code;
mod constant_folding;
mod hardware_fusion;
mod pass;
mod pattern_fusion;

pub use pass::{OptimizationContext, OptimizationPass};

use crate::loader::onnx::OnnxGraph;
use crate::loader::onnx::{OnnxAttributeValue, OnnxSparseFormat, OnnxTensor};

use super::types::{
    AtomicOp, AttrValue, FusedGraph, FusedNode, FusedOp, QuantizationInfo, SparseFormat,
    SparseTensorBinding, TensorAttrValue,
};

/// 图优化器 (REQ-OPT-004)
#[derive(Debug)]
pub struct GraphOptimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
    context: OptimizationContext,
}

impl GraphOptimizer {
    /// 创建默认优化器（包含所有内置 Pass）
    pub fn new(context: OptimizationContext) -> Self {
        let mut optimizer = Self {
            passes: Vec::new(),
            context,
        };
        optimizer.register_builtin_passes();
        optimizer
    }

    /// 创建空优化器（无 Pass）
    pub fn empty(context: OptimizationContext) -> Self {
        Self {
            passes: Vec::new(),
            context,
        }
    }

    /// 注册优化 Pass
    pub fn register_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
        self.passes.sort_by_key(|p| p.priority());
    }

    /// 注册内置 Pass
    fn register_builtin_passes(&mut self) {
        self.register_pass(Box::new(constant_folding::ConstantFoldingPass));
        self.register_pass(Box::new(pattern_fusion::FlashAttentionFusionPass));
        self.register_pass(Box::new(pattern_fusion::GQAFusionPass));
        self.register_pass(Box::new(pattern_fusion::FusedQkvRopeFusionPass));
        self.register_pass(Box::new(pattern_fusion::SwiGLUFusionPass));
        self.register_pass(Box::new(pattern_fusion::FusedRMSLinearFusionPass));
        self.register_pass(Box::new(pattern_fusion::MoERoutingFusionPass));
        self.register_pass(Box::new(hardware_fusion::HardwareFusionPass));
        self.register_pass(Box::new(dead_code::DeadCodeEliminationPass));
    }

    /// 执行优化
    pub fn optimize(&self, graph: &OnnxGraph) -> Result<FusedGraph, OptimizeError> {
        // 1. 转换为中间表示
        let mut fused = self.convert_to_fused(graph)?;
        let original_count = fused.nodes.len();
        fused.stats.original_nodes = original_count;

        // 2. 执行所有 Pass
        for pass in &self.passes {
            if pass.enabled(&self.context) {
                fused = pass.run(fused, &self.context)?;
            }
        }

        // 3. 更新统计
        fused.stats.optimized_nodes = fused.nodes.len();

        Ok(fused)
    }

    /// 将 OnnxGraph 转换为 FusedGraph
    fn convert_to_fused(&self, graph: &OnnxGraph) -> Result<FusedGraph, OptimizeError> {
        let mut fused = FusedGraph::new();

        // 复制输入输出
        fused.inputs = graph.inputs.iter().map(|i| i.name.clone()).collect();
        fused.outputs = graph.outputs.iter().map(|o| o.name.clone()).collect();

        // 转换节点
        for node in &graph.nodes {
            let fused_node = FusedNode {
                name: node.name.clone(),
                op: FusedOp::Atomic(AtomicOp::new(&node.op_type)),
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                attributes: convert_attributes(&node.attributes, self),
            };
            fused.nodes.push(fused_node);
        }

        for (name, tensor) in &graph.initializers {
            fused.weight_bindings.insert(
                name.clone(),
                crate::graph::types::WeightBinding {
                    source_name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                    data: Some(tensor.raw_data().to_vec()),
                },
            );
        }

        for sparse in &graph.sparse_initializers {
            let format = match sparse.format {
                OnnxSparseFormat::Coo => SparseFormat::Coo,
                OnnxSparseFormat::Csr => SparseFormat::Csr,
                OnnxSparseFormat::Csc => SparseFormat::Csc,
            };
            let values_name = sparse.values.name.clone();
            let indices_name = sparse.indices.name.clone();
            fused.sparse_tensors.insert(
                values_name.clone(),
                SparseTensorBinding {
                    format,
                    indices: indices_name.clone(),
                    values: values_name.clone(),
                    shape: sparse.dims.clone(),
                },
            );
            fused.weight_bindings.insert(
                values_name,
                crate::graph::types::WeightBinding {
                    source_name: sparse.values.name.clone(),
                    shape: sparse.values.shape.clone(),
                    dtype: sparse.values.dtype,
                    data: Some(sparse.values.raw_data().to_vec()),
                },
            );
            fused.weight_bindings.insert(
                indices_name,
                crate::graph::types::WeightBinding {
                    source_name: sparse.indices.name.clone(),
                    shape: sparse.indices.shape.clone(),
                    dtype: sparse.indices.dtype,
                    data: Some(sparse.indices.raw_data().to_vec()),
                },
            );
        }

        for annotation in &graph.quantization_annotation {
            let scale = annotation
                .quant_param_tensor_names
                .get("SCALE_TENSOR")
                .and_then(|name| graph.initializers.get(name))
                .and_then(OnnxTensor::scalar_f32)
                .unwrap_or(1.0);
            let zero_point = annotation
                .quant_param_tensor_names
                .get("ZERO_POINT_TENSOR")
                .and_then(|name| graph.initializers.get(name))
                .and_then(OnnxTensor::scalar_i64)
                .unwrap_or(0);
            let axis = annotation
                .quant_param_tensor_names
                .get("AXIS")
                .and_then(|v| v.parse::<i32>().ok());
            fused.quantization_info.insert(
                annotation.tensor_name.clone(),
                QuantizationInfo {
                    scale,
                    zero_point,
                    axis,
                },
            );
        }

        Ok(fused)
    }

    /// 获取已注册的 Pass 名称
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.name()).collect()
    }
}

fn convert_attributes(
    attrs: &std::collections::HashMap<String, crate::loader::onnx::OnnxAttribute>,
    optimizer: &GraphOptimizer,
) -> std::collections::HashMap<String, AttrValue> {
    let mut out = std::collections::HashMap::new();
    for (name, attr) in attrs {
        let converted = match &attr.value {
            OnnxAttributeValue::Int(value) => Some(AttrValue::Int(*value)),
            OnnxAttributeValue::Float(value) => Some(AttrValue::Float(*value)),
            OnnxAttributeValue::String(value) => Some(AttrValue::String(value.clone())),
            OnnxAttributeValue::Ints(values) => Some(AttrValue::Ints(values.clone())),
            OnnxAttributeValue::Floats(values) => Some(AttrValue::Floats(values.clone())),
            OnnxAttributeValue::Strings(values) => Some(AttrValue::Strings(values.clone())),
            OnnxAttributeValue::Tensor(tensor) => Some(AttrValue::Tensor(TensorAttrValue {
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                data: tensor.raw_data().to_vec(),
            })),
            // Subgraph support for If/Loop/Scan control flow operators
            OnnxAttributeValue::Graph(subgraph) => {
                optimizer.convert_to_fused(subgraph)
                    .ok()
                    .map(|g| AttrValue::Graph(Box::new(g)))
            }
            OnnxAttributeValue::Graphs(subgraphs) => {
                let converted_graphs: Result<Vec<_>, _> = subgraphs
                    .iter()
                    .map(|g| optimizer.convert_to_fused(g))
                    .collect();
                converted_graphs.ok().map(AttrValue::Graphs)
            }
            _ => None,
        };
        if let Some(value) = converted {
            out.insert(name.clone(), value);
        }
    }
    out
}

/// 优化错误
#[derive(Debug, thiserror::Error)]
pub enum OptimizeError {
    #[error("Pass '{pass}' failed: {reason}")]
    PassFailed { pass: String, reason: String },
    #[error("Invalid graph: {0}")]
    InvalidGraph(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::onnx::{
        OnnxAttribute, OnnxAttributeValue, OnnxNode, OnnxQuantizationAnnotation, OnnxSparseFormat,
    };
    use prost::bytes::Bytes;
    use safetensors::Dtype;
    use std::collections::HashMap;

    fn scalar_f32_tensor(name: &str, value: f32) -> crate::loader::onnx::OnnxTensor {
        crate::loader::onnx::OnnxTensor::new(
            name.to_string(),
            Dtype::F32,
            vec![],
            Bytes::copy_from_slice(&value.to_le_bytes()),
        )
    }

    fn scalar_i64_tensor(name: &str, value: i64) -> crate::loader::onnx::OnnxTensor {
        crate::loader::onnx::OnnxTensor::new(
            name.to_string(),
            Dtype::I64,
            vec![],
            Bytes::copy_from_slice(&value.to_le_bytes()),
        )
    }

    #[test]
    fn optimizer_registers_builtin_passes() {
        let ctx = OptimizationContext::default();
        let optimizer = GraphOptimizer::new(ctx);
        let names = optimizer.pass_names();
        assert!(names.contains(&"ConstantFolding"));
        assert!(names.contains(&"SwiGLUFusion"));
        assert!(names.contains(&"GQAFusion"));
        assert!(names.contains(&"MoERoutingFusion"));
        assert!(names.contains(&"FlashAttentionFusion"));
        assert!(names.contains(&"DeadCodeElimination"));
    }

    #[test]
    fn empty_optimizer_has_no_passes() {
        let ctx = OptimizationContext::default();
        let optimizer = GraphOptimizer::empty(ctx);
        assert!(optimizer.pass_names().is_empty());
    }

    #[test]
    fn convert_to_fused_propagates_node_attributes() {
        let node = OnnxNode {
            name: "n0".to_string(),
            op_type: "TopK".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::from([
                (
                    "k".to_string(),
                    OnnxAttribute {
                        name: "k".to_string(),
                        value: OnnxAttributeValue::Int(2),
                        doc_string: String::new(),
                        ref_attr_name: None,
                        attr_type: None,
                    },
                ),
                (
                    "labels".to_string(),
                    OnnxAttribute {
                        name: "labels".to_string(),
                        value: OnnxAttributeValue::Strings(vec!["a".to_string(), "b".to_string()]),
                        doc_string: String::new(),
                        ref_attr_name: None,
                        attr_type: None,
                    },
                ),
            ]),
        };

        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![node],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };

        let optimizer = GraphOptimizer::empty(OptimizationContext::default());
        let fused = optimizer.convert_to_fused(&graph).unwrap();

        assert!(matches!(
            fused.nodes[0].attributes.get("k"),
            Some(AttrValue::Int(2))
        ));
        assert!(matches!(
            fused.nodes[0].attributes.get("labels"),
            Some(AttrValue::Strings(values)) if values.len() == 2
        ));
    }

    #[test]
    fn convert_to_fused_propagates_sparse_and_quantization() {
        let mut initializers = HashMap::new();
        initializers.insert("scale_t".to_string(), scalar_f32_tensor("scale_t", 0.125));
        initializers.insert("zp_t".to_string(), scalar_i64_tensor("zp_t", 3));

        let sparse = crate::loader::onnx::OnnxSparseTensor {
            values: crate::loader::onnx::OnnxTensor::new(
                "w_sparse".to_string(),
                Dtype::F32,
                vec![2],
                Bytes::copy_from_slice(&[0, 0, 128, 63, 0, 0, 0, 64]),
            ),
            indices: crate::loader::onnx::OnnxTensor::new(
                "w_sparse_idx".to_string(),
                Dtype::I64,
                vec![2],
                Bytes::copy_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            ),
            dims: vec![4],
            format: OnnxSparseFormat::Coo,
        };

        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers,
            sparse_initializers: vec![sparse],
            quantization_annotation: vec![OnnxQuantizationAnnotation {
                tensor_name: "w_sparse".to_string(),
                quant_param_tensor_names: HashMap::from([
                    ("SCALE_TENSOR".to_string(), "scale_t".to_string()),
                    ("ZERO_POINT_TENSOR".to_string(), "zp_t".to_string()),
                ]),
                scale: Some(0.125),
                zero_point: Some(3),
                axis: None,
            }],
            metadata_props: HashMap::new(),
        };

        let optimizer = GraphOptimizer::empty(OptimizationContext::default());
        let fused = optimizer.convert_to_fused(&graph).unwrap();

        assert!(fused.sparse_tensors.contains_key("w_sparse"));
        assert!(fused.weight_bindings.contains_key("w_sparse_idx"));
        let q = fused
            .quantization_info
            .get("w_sparse")
            .expect("quantization info");
        assert!((q.scale - 0.125).abs() < f32::EPSILON);
        assert_eq!(q.zero_point, 3);
    }
}
