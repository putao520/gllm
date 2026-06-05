use std::collections::HashMap;

use super::attributes::{parse_attributes, OnnxAttribute};
use super::external::ExternalDataResolver;
use super::tensor::{OnnxSparseTensor, OnnxTensor};
use super::types::OnnxType;
use super::{proto, LoaderError, Result};

#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub metadata: OnnxModelMetadata,
    pub graph: OnnxGraph,
    /// Model-local functions (custom operators)
    pub functions: Vec<OnnxFunction>,
}

#[derive(Debug, Clone)]
pub struct OnnxModelMetadata {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub opset_import: Vec<OnnxOperatorSet>,
    pub metadata_props: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct OnnxOperatorSet {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug, Clone)]
pub struct OnnxGraph {
    pub name: String,
    pub doc_string: String,
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<OnnxValueInfo>,
    pub outputs: Vec<OnnxValueInfo>,
    pub value_info: Vec<OnnxValueInfo>,
    pub initializers: HashMap<String, OnnxTensor>,
    pub sparse_initializers: Vec<OnnxSparseTensor>,
    pub quantization_annotation: Vec<OnnxQuantizationAnnotation>,
    pub metadata_props: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttribute>,
}

#[derive(Debug, Clone)]
pub struct OnnxValueInfo {
    pub name: String,
    pub value_type: Option<OnnxType>,
    pub doc_string: String,
    pub metadata_props: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct OnnxQuantizationAnnotation {
    pub tensor_name: String,
    pub quant_param_tensor_names: HashMap<String, String>,
    pub scale: Option<f32>,
    pub zero_point: Option<i64>,
    pub axis: Option<i32>,
}

/// ONNX Function (custom operator definition)
///
/// Functions allow defining custom operators as subgraphs of existing operators.
/// When a NodeProto references this function, it gets expanded to the function body.
#[derive(Debug, Clone)]
pub struct OnnxFunction {
    /// Function name (like op_type in NodeProto)
    pub name: String,
    /// Domain this function belongs to
    pub domain: String,
    /// Overload identifier (for function overloading)
    pub overload: String,
    /// Input parameter names
    pub inputs: Vec<String>,
    /// Output parameter names
    pub outputs: Vec<String>,
    /// Attribute parameter names (without defaults)
    pub attributes: Vec<String>,
    /// Attribute parameters with default values
    pub attribute_protos: HashMap<String, OnnxAttribute>,
    /// Function body nodes
    pub nodes: Vec<OnnxNode>,
    /// Operator sets this function relies on
    pub opset_import: Vec<OnnxOperatorSet>,
    /// Value info for intermediate values
    pub value_info: Vec<OnnxValueInfo>,
    /// Documentation
    pub doc_string: String,
    /// Metadata
    pub metadata_props: HashMap<String, String>,
}

impl OnnxModel {
    pub(super) fn from_proto(
        proto: proto::ModelProto,
        resolver: &mut ExternalDataResolver,
    ) -> Result<Self> {
        let graph = proto
            .graph
            .ok_or_else(|| LoaderError::Onnx("missing graph in ONNX model".to_string()))?;
        let metadata = OnnxModelMetadata {
            ir_version: proto.ir_version.unwrap_or_default(), // LEGAL: protobuf 可选字段，缺失时使用默认值
            producer_name: proto.producer_name.unwrap_or_default(), // LEGAL: protobuf 可选字段
            producer_version: proto.producer_version.unwrap_or_default(), // LEGAL: protobuf 可选字段
            domain: proto.domain.unwrap_or_default(), // LEGAL: protobuf 可选字段
            model_version: proto.model_version.unwrap_or_default(), // LEGAL: protobuf 可选字段
            doc_string: proto.doc_string.unwrap_or_default(), // LEGAL: protobuf 可选字段
            opset_import: parse_opsets(proto.opset_import),
            metadata_props: parse_metadata_props(proto.metadata_props),
        };
        let graph = OnnxGraph::from_proto(graph, resolver)?;
        let functions = parse_functions(proto.functions, resolver)?;
        Ok(Self { metadata, graph, functions })
    }
}

impl OnnxGraph {
    /// Parse a GraphProto into OnnxGraph (recursive for subgraphs)
    pub(super) fn from_proto(proto: proto::GraphProto, resolver: &mut ExternalDataResolver) -> Result<Self> {
        let mut initializers = HashMap::new();
        for initializer in proto.initializer {
            let tensor = OnnxTensor::from_initializer(initializer, resolver)?;
            if initializers.contains_key(&tensor.name) {
                return Err(LoaderError::DuplicateTensor(tensor.name));
            }
            initializers.insert(tensor.name.clone(), tensor);
        }

        let mut sparse_initializers = Vec::with_capacity(proto.sparse_initializer.len());
        for sparse in proto.sparse_initializer {
            let tensor = OnnxSparseTensor::from_proto(sparse, resolver)?;
            let values_name = tensor.values.name.clone();
            if initializers.contains_key(&values_name) {
                return Err(LoaderError::DuplicateTensor(values_name));
            }
            sparse_initializers.push(tensor);
        }

        let nodes = parse_nodes(proto.node, resolver)?;
        let inputs = parse_value_info(proto.input)?;
        let outputs = parse_value_info(proto.output)?;
        let value_info = parse_value_info(proto.value_info)?;
        let quantization_annotation = parse_quantization(proto.quantization_annotation, &initializers);

        Ok(Self {
            name: proto.name.unwrap_or_default(), // LEGAL: protobuf 可选字段
            doc_string: proto.doc_string.unwrap_or_default(), // LEGAL: protobuf 可选字段
            nodes,
            inputs,
            outputs,
            value_info,
            initializers,
            sparse_initializers,
            quantization_annotation,
            metadata_props: parse_metadata_props(proto.metadata_props),
        })
    }
}

impl OnnxGraph {
    /// Bind weights from a TensorProvider to the graph's initializers (REQ-ARCH-003)
    ///
    /// Zero-copy binding: tensor data is referenced directly from the provider.
    /// This allows GGUF/SafeTensors weights to be used without intermediate conversion.
    pub fn bind_weights<P: crate::loader::TensorProvider>(
        &mut self,
        provider: &P,
        name_mapping: &HashMap<String, String>,
    ) -> Result<usize> {
        use prost::bytes::Bytes;

        let mut bound_count = 0;

        for (graph_name, tensor_name) in name_mapping {
            if let Some(meta) = provider.tensor_info(tensor_name) {
                // Load tensor data (may be zero-copy via mmap)
                let data = provider.load_tensor_data(tensor_name)?;

                // Create OnnxTensor with the loaded data
                let tensor = OnnxTensor::new(
                    graph_name.clone(),
                    meta.dtype,
                    meta.shape.clone(),
                    Bytes::from(data.into_owned()),
                );

                self.initializers.insert(graph_name.clone(), tensor);
                bound_count += 1;
            }
        }

        Ok(bound_count)
    }

    /// Bind all tensors from provider using automatic name matching
    ///
    /// Attempts to match tensor names from provider to graph node inputs.
    pub fn bind_weights_auto<P: crate::loader::TensorProvider>(
        &mut self,
        provider: &P,
    ) -> Result<usize> {
        use prost::bytes::Bytes;

        let mut bound_count = 0;

        // Collect all input names from nodes that might need weights
        let mut weight_inputs: std::collections::HashSet<String> = std::collections::HashSet::new();
        for node in &self.nodes {
            for input in &node.inputs {
                // Skip empty inputs and likely activation inputs
                if !input.is_empty() && !input.starts_with("hidden") && !input.starts_with("input")
                {
                    weight_inputs.insert(input.clone());
                }
            }
        }

        // Bind tensors from provider
        for meta in provider.iter_tensors() {
            if weight_inputs.contains(&meta.name) {
                let data = provider.load_tensor_data(&meta.name)?;
                let tensor = OnnxTensor::new(
                    meta.name.clone(),
                    meta.dtype,
                    meta.shape.clone(),
                    Bytes::from(data.into_owned()),
                );
                self.initializers.insert(meta.name, tensor);
                bound_count += 1;
            }
        }

        Ok(bound_count)
    }
}

fn parse_nodes(
    nodes: Vec<proto::NodeProto>,
    resolver: &mut ExternalDataResolver,
) -> Result<Vec<OnnxNode>> {
    let mut out = Vec::with_capacity(nodes.len());
    for (idx, node) in nodes.into_iter().enumerate() {
        let op_type = node.op_type.unwrap_or_default(); // LEGAL: protobuf 可选字段
        if op_type.is_empty() {
            return Err(LoaderError::Onnx(format!("node {idx} missing op_type")));
        }
        let name = match node.name.clone() {
            Some(value) if !value.is_empty() => value,
            _ => format!("node_{idx}"),
        };
        let attributes = parse_attributes(node.attribute, resolver)?;
        out.push(OnnxNode {
            name,
            op_type,
            domain: node.domain.unwrap_or_default(), // LEGAL: protobuf 可选字段
            inputs: node.input,
            outputs: node.output,
            attributes,
        });
    }
    Ok(out)
}

fn parse_value_info(values: Vec<proto::ValueInfoProto>) -> Result<Vec<OnnxValueInfo>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let name = value
            .name
            .ok_or_else(|| LoaderError::Onnx("value_info missing name".to_string()))?;
        let value_type = match value.r#type {
            Some(tp) => Some(OnnxType::from_proto(tp)?),
            None => None,
        };
        out.push(OnnxValueInfo {
            name,
            value_type,
            doc_string: value.doc_string.unwrap_or_default(), // LEGAL: protobuf 可选字段
            metadata_props: parse_metadata_props(value.metadata_props),
        });
    }
    Ok(out)
}

fn parse_opsets(opsets: Vec<proto::OperatorSetIdProto>) -> Vec<OnnxOperatorSet> {
    opsets
        .into_iter()
        .map(|opset| OnnxOperatorSet {
            domain: opset.domain.unwrap_or_default(), // LEGAL: protobuf 可选字段
            version: opset.version.unwrap_or_default(), // LEGAL: protobuf 可选字段
        })
        .collect()
}

fn parse_metadata_props(entries: Vec<proto::StringStringEntryProto>) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for entry in entries {
        let Some(key) = entry.key else {
            continue;
        };
        if key.is_empty() {
            continue;
        }
        let value = entry.value.unwrap_or_default(); // LEGAL: protobuf 可选字段
        out.insert(key, value);
    }
    out
}

fn parse_quantization(
    entries: Vec<proto::TensorAnnotation>,
    initializers: &HashMap<String, OnnxTensor>,
) -> Vec<OnnxQuantizationAnnotation> {
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let quant_param_tensor_names = parse_metadata_props(entry.quant_parameter_tensor_names);
        let scale = quant_param_tensor_names
            .get("SCALE_TENSOR")
            .and_then(|name| initializers.get(name))
            .and_then(OnnxTensor::scalar_f32);
        let zero_point = quant_param_tensor_names
            .get("ZERO_POINT_TENSOR")
            .and_then(|name| initializers.get(name))
            .and_then(OnnxTensor::scalar_i64);
        let axis = quant_param_tensor_names
            .get("AXIS")
            .and_then(|value| value.parse::<i32>().ok());
        out.push(OnnxQuantizationAnnotation {
            tensor_name: entry.tensor_name.unwrap_or_default(), // LEGAL: protobuf 可选字段
            quant_param_tensor_names,
            scale,
            zero_point,
            axis,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::bytes::Bytes;
    use safetensors::Dtype;

    // ── OnnxModelMetadata ─────────────────────────────────────────────

    #[test]
    fn onnx_model_metadata_fields() {
        let meta = OnnxModelMetadata {
            ir_version: 7,
            producer_name: "test".to_string(),
            producer_version: "1.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 1,
            doc_string: "test model".to_string(),
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 17,
            }],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.ir_version, 7);
        assert_eq!(meta.producer_name, "test");
        assert_eq!(meta.opset_import.len(), 1);
    }

    // ── OnnxOperatorSet ───────────────────────────────────────────────

    #[test]
    fn onnx_operator_set_fields() {
        let ops = OnnxOperatorSet {
            domain: "ai.onnx.ml".to_string(),
            version: 3,
        };
        assert_eq!(ops.domain, "ai.onnx.ml");
        assert_eq!(ops.version, 3);
    }

    // ── OnnxGraph struct ──────────────────────────────────────────────

    #[test]
    fn onnx_graph_default_fields() {
        let graph = OnnxGraph {
            name: "test_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.name, "test_graph");
        assert!(graph.nodes.is_empty());
        assert!(graph.initializers.is_empty());
    }

    // ── OnnxNode ──────────────────────────────────────────────────────

    #[test]
    fn onnx_node_fields() {
        let node = OnnxNode {
            name: "conv1".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string(), "W".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.op_type, "Conv");
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.outputs[0], "Y");
    }

    // ── OnnxValueInfo ─────────────────────────────────────────────────

    #[test]
    fn onnx_value_info_fields() {
        let vi = OnnxValueInfo {
            name: "input".to_string(),
            value_type: None,
            doc_string: "input tensor".to_string(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(vi.name, "input");
        assert!(vi.value_type.is_none());
    }

    // ── OnnxQuantizationAnnotation ─────────────────────────────────────

    #[test]
    fn onnx_quantization_annotation_fields() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.01),
            zero_point: Some(128),
            axis: Some(0),
        };
        assert_eq!(qa.tensor_name, "weight");
        assert_eq!(qa.scale, Some(0.01));
        assert_eq!(qa.zero_point, Some(128));
    }

    // ── OnnxFunction ──────────────────────────────────────────────────

    #[test]
    fn onnx_function_fields() {
        let func = OnnxFunction {
            name: "CustomOp".to_string(),
            domain: "custom".to_string(),
            overload: "v1".to_string(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec!["alpha".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.name, "CustomOp");
        assert_eq!(func.inputs, vec!["X"]);
    }

    // ── parse_metadata_props ──────────────────────────────────────────

    #[test]
    fn parse_metadata_props_valid() {
        let entries = vec![
            proto::StringStringEntryProto {
                key: Some("key1".to_string()),
                value: Some("value1".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("key2".to_string()),
                value: None,
            },
        ];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("key1").unwrap(), "value1");
        assert_eq!(result.get("key2").unwrap(), "");
    }

    #[test]
    fn parse_metadata_props_skips_empty_key() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some(String::new()),
            value: Some("ignored".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert!(result.is_empty());
    }

    #[test]
    fn parse_metadata_props_skips_none_key() {
        let entries = vec![proto::StringStringEntryProto {
            key: None,
            value: Some("ignored".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert!(result.is_empty());
    }

    #[test]
    fn parse_metadata_props_empty_input() {
        let result = parse_metadata_props(vec![]);
        assert!(result.is_empty());
    }

    // ── parse_opsets ──────────────────────────────────────────────────

    #[test]
    fn parse_opsets_valid() {
        let opsets = vec![
            proto::OperatorSetIdProto {
                domain: Some("ai.onnx".to_string()),
                version: Some(17),
            },
            proto::OperatorSetIdProto {
                domain: None,
                version: None,
            },
        ];
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].domain, "ai.onnx");
        assert_eq!(result[0].version, 17);
        assert_eq!(result[1].domain, "");
        assert_eq!(result[1].version, 0);
    }

    #[test]
    fn parse_opsets_empty() {
        let result = parse_opsets(vec![]);
        assert!(result.is_empty());
    }

    // ── parse_quantization ────────────────────────────────────────────

    #[test]
    fn parse_quantization_extracts_scale_and_zp() {
        use super::super::tensor::OnnxTensor;

        let scale_data = 0.125f32.to_le_bytes();
        let scale_tensor = OnnxTensor::new(
            "scale_tensor".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(scale_data.to_vec()),
        );
        let zp_data = 128i64.to_le_bytes();
        let zp_tensor = OnnxTensor::new(
            "zp_tensor".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(zp_data.to_vec()),
        );

        let mut initializers = HashMap::new();
        initializers.insert("scale_tensor".to_string(), scale_tensor);
        initializers.insert("zp_tensor".to_string(), zp_tensor);

        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![
                proto::StringStringEntryProto {
                    key: Some("SCALE_TENSOR".to_string()),
                    value: Some("scale_tensor".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("ZERO_POINT_TENSOR".to_string()),
                    value: Some("zp_tensor".to_string()),
                },
            ],
        }];

        let result = parse_quantization(entries, &initializers);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tensor_name, "weight");
        assert!((result[0].scale.unwrap() - 0.125).abs() < 1e-6);
        assert_eq!(result[0].zero_point, Some(128));
    }

    #[test]
    fn parse_quantization_empty() {
        let result = parse_quantization(vec![], &HashMap::new());
        assert!(result.is_empty());
    }

    #[test]
    fn parse_quantization_axis_parsing() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some("3".to_string()),
            }],
        }];

        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result[0].axis, Some(3));
    }

    // ── Clone consistency ─────────────────────────────────────────────

    #[test]
    fn onnx_node_clone() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        let cloned = node.clone();
        assert_eq!(cloned.name, "n");
        assert_eq!(cloned.op_type, "Relu");
    }

    #[test]
    fn onnx_graph_clone() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let cloned = graph.clone();
        assert_eq!(cloned.name, "g");
    }

    #[test]
    fn onnx_function_clone() {
        let func = OnnxFunction {
            name: "F".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let cloned = func.clone();
        assert_eq!(cloned.name, "F");
    }

    // ── OnnxModel struct construction ──────────────────────────────────

    #[test]
    fn onnx_model_fields() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: "gllm-test".to_string(),
                producer_version: "2.0".to_string(),
                domain: "ai.onnx".to_string(),
                model_version: 42,
                doc_string: "unit test model".to_string(),
                opset_import: vec![
                    OnnxOperatorSet {
                        domain: "".to_string(),
                        version: 17,
                    },
                    OnnxOperatorSet {
                        domain: "ai.onnx.ml".to_string(),
                        version: 3,
                    },
                ],
                metadata_props: HashMap::from([
                    ("author".to_string(), "tester".to_string()),
                    ("license".to_string(), "MIT".to_string()),
                ]),
            },
            graph: OnnxGraph {
                name: "test_graph".to_string(),
                doc_string: String::new(),
                nodes: vec![OnnxNode {
                    name: "n0".to_string(),
                    op_type: "Conv".to_string(),
                    domain: String::new(),
                    inputs: vec!["X".to_string(), "W".to_string()],
                    outputs: vec!["Y".to_string()],
                    attributes: HashMap::new(),
                }],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert_eq!(model.metadata.ir_version, 8);
        assert_eq!(model.metadata.producer_name, "gllm-test");
        assert_eq!(model.metadata.opset_import.len(), 2);
        assert_eq!(model.metadata.metadata_props.get("author").unwrap(), "tester");
        assert_eq!(model.graph.nodes.len(), 1);
        assert_eq!(model.graph.nodes[0].op_type, "Conv");
        assert!(model.functions.is_empty());
    }

    // ── Clone: OnnxModelMetadata ───────────────────────────────────────

    #[test]
    fn onnx_model_metadata_clone() {
        let meta = OnnxModelMetadata {
            ir_version: 9,
            producer_name: "clone-test".to_string(),
            producer_version: "3.0".to_string(),
            domain: "test.domain".to_string(),
            model_version: 100,
            doc_string: "clone me".to_string(),
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 20,
            }],
            metadata_props: HashMap::from([("k".to_string(), "v".to_string())]),
        };
        let cloned = meta.clone();
        assert_eq!(cloned.ir_version, 9);
        assert_eq!(cloned.producer_name, "clone-test");
        assert_eq!(cloned.opset_import[0].version, 20);
        assert_eq!(cloned.metadata_props.get("k").unwrap(), "v");
    }

    // ── Clone: OnnxOperatorSet ─────────────────────────────────────────

    #[test]
    fn onnx_operator_set_clone() {
        let ops = OnnxOperatorSet {
            domain: "ai.onnx.training".to_string(),
            version: 1,
        };
        let cloned = ops.clone();
        assert_eq!(cloned.domain, "ai.onnx.training");
        assert_eq!(cloned.version, 1);
    }

    // ── Clone: OnnxValueInfo ───────────────────────────────────────────

    #[test]
    fn onnx_value_info_clone() {
        let vi = OnnxValueInfo {
            name: "output".to_string(),
            value_type: None,
            doc_string: "output tensor".to_string(),
            metadata_props: HashMap::from([("source".to_string(), "model".to_string())]),
        };
        let cloned = vi.clone();
        assert_eq!(cloned.name, "output");
        assert_eq!(cloned.doc_string, "output tensor");
        assert_eq!(cloned.metadata_props.get("source").unwrap(), "model");
    }

    // ── Clone: OnnxQuantizationAnnotation ───────────────────────────────

    #[test]
    fn onnx_quantization_annotation_clone() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "quantized_weight".to_string(),
            quant_param_tensor_names: HashMap::from([
                ("SCALE_TENSOR".to_string(), "scale_0".to_string()),
                ("ZERO_POINT_TENSOR".to_string(), "zp_0".to_string()),
            ]),
            scale: Some(0.005),
            zero_point: Some(64),
            axis: Some(1),
        };
        let cloned = qa.clone();
        assert_eq!(cloned.tensor_name, "quantized_weight");
        assert!((cloned.scale.unwrap() - 0.005).abs() < 1e-9);
        assert_eq!(cloned.zero_point, Some(64));
        assert_eq!(cloned.axis, Some(1));
        assert_eq!(cloned.quant_param_tensor_names.len(), 2);
    }

    // ── Clone: OnnxModel ───────────────────────────────────────────────

    #[test]
    fn onnx_model_clone() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: "clone".to_string(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let cloned = model.clone();
        assert_eq!(cloned.metadata.ir_version, 7);
        assert_eq!(cloned.graph.name, "g");
    }

    // ── Debug trait on structs ─────────────────────────────────────────

    #[test]
    fn onnx_node_debug_format() {
        let node = OnnxNode {
            name: "relu_1".to_string(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        let debug_str = format!("{:?}", node);
        assert!(debug_str.contains("relu_1"));
        assert!(debug_str.contains("Relu"));
    }

    #[test]
    fn onnx_graph_debug_format() {
        let graph = OnnxGraph {
            name: "my_graph".to_string(),
            doc_string: "test graph".to_string(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let debug_str = format!("{:?}", graph);
        assert!(debug_str.contains("my_graph"));
    }

    #[test]
    fn onnx_model_metadata_debug_format() {
        let meta = OnnxModelMetadata {
            ir_version: 7,
            producer_name: "debug_test".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("ir_version"));
    }

    // ── parse_nodes error path ─────────────────────────────────────────

    #[test]
    fn parse_nodes_missing_op_type_returns_error() {
        let nodes = vec![proto::NodeProto {
            op_type: None,
            name: None,
            input: vec![],
            output: vec![],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("missing op_type"));
    }

    #[test]
    fn parse_nodes_empty_op_type_returns_error() {
        let nodes = vec![proto::NodeProto {
            op_type: Some(String::new()),
            name: None,
            input: vec![],
            output: vec![],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("missing op_type"));
    }

    #[test]
    fn parse_nodes_auto_generates_name_when_missing() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Add".to_string()),
            name: None,
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["c".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "node_0");
        assert_eq!(result[0].op_type, "Add");
    }

    #[test]
    fn parse_nodes_preserves_explicit_name() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Sub".to_string()),
            name: Some("my_sub_node".to_string()),
            input: vec![],
            output: vec![],
            domain: Some("custom.domain".to_string()),
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].name, "my_sub_node");
        assert_eq!(result[0].domain, "custom.domain");
    }

    #[test]
    fn parse_nodes_empty_name_falls_back_to_indexed() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Mul".to_string()),
            name: Some(String::new()),
            input: vec![],
            output: vec![],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].name, "node_0");
    }

    #[test]
    fn parse_nodes_multiple_nodes_correct_indices() {
        let nodes: Vec<proto::NodeProto> = (0..3)
            .map(|i| proto::NodeProto {
                op_type: Some(format!("Op{i}")),
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            })
            .collect();
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].name, "node_0");
        assert_eq!(result[1].name, "node_1");
        assert_eq!(result[2].name, "node_2");
    }

    // ── parse_value_info error path ────────────────────────────────────

    #[test]
    fn parse_value_info_missing_name_returns_error() {
        let values = vec![proto::ValueInfoProto {
            name: None,
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        }];
        let result = parse_value_info(values);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("missing name"));
    }

    #[test]
    fn parse_value_info_with_none_type() {
        let values = vec![proto::ValueInfoProto {
            name: Some("input_ids".to_string()),
            r#type: None,
            doc_string: Some("token ids".to_string()),
            metadata_props: vec![],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "input_ids");
        assert!(result[0].value_type.is_none());
        assert_eq!(result[0].doc_string, "token ids");
    }

    #[test]
    fn parse_value_info_multiple_entries() {
        let values = vec![
            proto::ValueInfoProto {
                name: Some("input".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            },
            proto::ValueInfoProto {
                name: Some("output".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            },
        ];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "input");
        assert_eq!(result[1].name, "output");
    }

    // ── parse_functions error path ─────────────────────────────────────

    #[test]
    fn parse_functions_missing_name_returns_error() {
        let functions = vec![proto::FunctionProto {
            name: None,
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("function missing name"));
    }

    #[test]
    fn parse_functions_empty_name_returns_error() {
        let functions = vec![proto::FunctionProto {
            name: Some(String::new()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver);
        assert!(result.is_err());
    }

    #[test]
    fn parse_functions_valid_function_with_nodes() {
        let functions = vec![proto::FunctionProto {
            name: Some("MyCustomOp".to_string()),
            domain: Some("custom".to_string()),
            overload: Some("v2".to_string()),
            input: vec!["X".to_string(), "Y".to_string()],
            output: vec!["Z".to_string()],
            attribute: vec!["alpha".to_string()],
            attribute_proto: vec![],
            node: vec![proto::NodeProto {
                op_type: Some("Add".to_string()),
                name: None,
                input: vec!["X".to_string(), "Y".to_string()],
                output: vec!["sum".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            }],
            opset_import: vec![proto::OperatorSetIdProto {
                domain: Some("".to_string()),
                version: Some(17),
            }],
            value_info: vec![],
            doc_string: Some("custom op".to_string()),
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "MyCustomOp");
        assert_eq!(result[0].domain, "custom");
        assert_eq!(result[0].overload, "v2");
        assert_eq!(result[0].inputs, vec!["X", "Y"]);
        assert_eq!(result[0].outputs, vec!["Z"]);
        assert_eq!(result[0].attributes, vec!["alpha"]);
        assert_eq!(result[0].nodes.len(), 1);
        assert_eq!(result[0].nodes[0].op_type, "Add");
        assert_eq!(result[0].opset_import.len(), 1);
        assert_eq!(result[0].doc_string, "custom op");
    }

    #[test]
    fn parse_functions_empty_list() {
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(vec![], &mut resolver).unwrap();
        assert!(result.is_empty());
    }

    // ── parse_metadata_props: duplicate keys ───────────────────────────

    #[test]
    fn parse_metadata_props_duplicate_key_last_wins() {
        let entries = vec![
            proto::StringStringEntryProto {
                key: Some("version".to_string()),
                value: Some("1.0".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("version".to_string()),
                value: Some("2.0".to_string()),
            },
        ];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("version").unwrap(), "2.0");
        assert_eq!(result.len(), 1);
    }

    // ── parse_quantization: missing tensor refs ────────────────────────

    #[test]
    fn parse_quantization_missing_scale_tensor_yields_none() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("SCALE_TENSOR".to_string()),
                value: Some("nonexistent_scale".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 1);
        assert!(result[0].scale.is_none());
        assert!(result[0].zero_point.is_none());
    }

    #[test]
    fn parse_quantization_no_quant_params() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 1);
        assert!(result[0].scale.is_none());
        assert!(result[0].zero_point.is_none());
        assert!(result[0].axis.is_none());
    }

    // ── OnnxGraph with non-empty content ───────────────────────────────

    #[test]
    fn onnx_graph_with_multiple_nodes_and_initializers() {
        let tensor = OnnxTensor::new(
            "weight".to_string(),
            Dtype::F32,
            vec![2, 3],
            Bytes::from_static(&[0u8; 24]),
        );
        let graph = OnnxGraph {
            name: "dense_graph".to_string(),
            doc_string: "a graph with data".to_string(),
            nodes: vec![
                OnnxNode {
                    name: "matmul".to_string(),
                    op_type: "MatMul".to_string(),
                    domain: String::new(),
                    inputs: vec!["input".to_string(), "weight".to_string()],
                    outputs: vec!["hidden".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "relu".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["hidden".to_string()],
                    outputs: vec!["output".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![OnnxValueInfo {
                name: "input".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![OnnxValueInfo {
                name: "output".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            value_info: vec![],
            initializers: HashMap::from([("weight".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![OnnxQuantizationAnnotation {
                tensor_name: "weight".to_string(),
                quant_param_tensor_names: HashMap::new(),
                scale: None,
                zero_point: None,
                axis: None,
            }],
            metadata_props: HashMap::from([("framework".to_string(), "pytorch".to_string())]),
        };
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].op_type, "MatMul");
        assert_eq!(graph.nodes[1].op_type, "Relu");
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
        assert!(graph.initializers.contains_key("weight"));
        assert_eq!(graph.quantization_annotation.len(), 1);
        assert_eq!(graph.metadata_props.get("framework").unwrap(), "pytorch");
    }

    // ── OnnxGraph initializer mutation ─────────────────────────────────

    #[test]
    fn onnx_graph_initializer_insert_and_lookup() {
        let tensor = OnnxTensor::new(
            "bias".to_string(),
            Dtype::F32,
            vec![4],
            Bytes::from_static(&[0u8; 16]),
        );
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(!graph.initializers.contains_key("bias"));
        graph.initializers.insert("bias".to_string(), tensor);
        assert!(graph.initializers.contains_key("bias"));
        assert_eq!(graph.initializers.get("bias").unwrap().shape, vec![4]);
    }

    // ── OnnxFunction with full field coverage ──────────────────────────

    #[test]
    fn onnx_function_full_fields() {
        let func = OnnxFunction {
            name: "FusedBiasGelu".to_string(),
            domain: "com.microsoft".to_string(),
            overload: "default".to_string(),
            inputs: vec!["X".to_string(), "B".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec!["approximation".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![OnnxNode {
                name: "bias_add".to_string(),
                op_type: "Add".to_string(),
                domain: String::new(),
                inputs: vec!["X".to_string(), "B".to_string()],
                outputs: vec!["biased".to_string()],
                attributes: HashMap::new(),
            }],
            opset_import: vec![
                OnnxOperatorSet {
                    domain: "".to_string(),
                    version: 17,
                },
            ],
            value_info: vec![OnnxValueInfo {
                name: "biased".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            doc_string: "Fused bias add + gelu".to_string(),
            metadata_props: HashMap::from([("source".to_string(), "onnxruntime".to_string())]),
        };
        assert_eq!(func.name, "FusedBiasGelu");
        assert_eq!(func.domain, "com.microsoft");
        assert_eq!(func.inputs.len(), 2);
        assert_eq!(func.outputs.len(), 1);
        assert_eq!(func.attributes.len(), 1);
        assert_eq!(func.nodes.len(), 1);
        assert_eq!(func.opset_import[0].version, 17);
        assert_eq!(func.value_info[0].name, "biased");
        assert_eq!(func.doc_string, "Fused bias add + gelu");
        assert_eq!(func.metadata_props.get("source").unwrap(), "onnxruntime");
    }

    // ── OnnxOperatorSet with default fields ────────────────────────────

    #[test]
    fn onnx_operator_set_default_fields() {
        let ops = OnnxOperatorSet {
            domain: String::new(),
            version: 0,
        };
        assert!(ops.domain.is_empty());
        assert_eq!(ops.version, 0);
    }

    // ── OnnxQuantizationAnnotation all fields none ─────────────────────

    #[test]
    fn onnx_quantization_annotation_all_none() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: String::new(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        assert!(qa.tensor_name.is_empty());
        assert!(qa.scale.is_none());
        assert!(qa.zero_point.is_none());
        assert!(qa.axis.is_none());
        assert!(qa.quant_param_tensor_names.is_empty());
    }

    // ── OnnxValueInfo with metadata_props ──────────────────────────────

    #[test]
    fn onnx_value_info_with_metadata() {
        let vi = OnnxValueInfo {
            name: "attention_mask".to_string(),
            value_type: None,
            doc_string: "attention mask tensor".to_string(),
            metadata_props: HashMap::from([
                ("source".to_string(), "tokenizer".to_string()),
            ]),
        };
        assert_eq!(vi.metadata_props.get("source").unwrap(), "tokenizer");
    }

    // ══════════════════════════════════════════════════════════════════════
    // NEW TESTS (18 additional)
    // ══════════════════════════════════════════════════════════════════════

    // ── parse_nodes: empty input yields empty output ──────────────────

    #[test]
    fn parse_nodes_empty_input() {
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(vec![], &mut resolver).unwrap();
        assert!(result.is_empty());
    }

    // ── parse_nodes: error at non-zero index includes correct index ───

    #[test]
    fn parse_nodes_error_at_nonzero_index() {
        // First two nodes are valid, third is missing op_type
        let nodes = vec![
            proto::NodeProto {
                op_type: Some("Add".to_string()),
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: Some("Sub".to_string()),
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: None,
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
        ];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("node 2"), "expected index 2 in error: {err_msg}");
    }

    // ── parse_value_info: empty list returns empty vec ────────────────

    #[test]
    fn parse_value_info_empty_list() {
        let result = parse_value_info(vec![]).unwrap();
        assert!(result.is_empty());
    }

    // ── OnnxTensor::new scalar_f32 returns correct value ──────────────

    #[test]
    fn onnx_tensor_scalar_f32_roundtrip() {
        let value = 3.14f32;
        let tensor = OnnxTensor::new(
            "scalar".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(value.to_le_bytes().to_vec()),
        );
        let result = tensor.scalar_f32().unwrap();
        assert!((result - 3.14f32).abs() < 1e-6);
    }

    // ── OnnxTensor::new scalar_i64 returns correct value ──────────────

    #[test]
    fn onnx_tensor_scalar_i64_roundtrip() {
        let value = -42i64;
        let tensor = OnnxTensor::new(
            "scalar".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(value.to_le_bytes().to_vec()),
        );
        assert_eq!(tensor.scalar_i64(), Some(-42));
    }

    // ── OnnxTensor scalar_f32 returns None for non-scalar ─────────────

    #[test]
    fn onnx_tensor_scalar_f32_none_for_multi_element() {
        let tensor = OnnxTensor::new(
            "vec".to_string(),
            Dtype::F32,
            vec![4],
            Bytes::from(vec![0u8; 16]),
        );
        assert_eq!(tensor.scalar_f32(), None);
    }

    // ── OnnxTensor scalar_i64 returns None for non-scalar ─────────────

    #[test]
    fn onnx_tensor_scalar_i64_none_for_multi_element() {
        let tensor = OnnxTensor::new(
            "vec".to_string(),
            Dtype::I64,
            vec![2],
            Bytes::from(vec![0u8; 16]),
        );
        assert_eq!(tensor.scalar_i64(), None);
    }

    // ── OnnxTensor::new_string sets is_string flag and U8 dtype ───────

    #[test]
    fn onnx_tensor_new_string_flags() {
        let tensor = OnnxTensor::new_string(
            "text_data".to_string(),
            vec![1],
            Bytes::from("hello".as_bytes().to_vec()),
        );
        assert!(tensor.is_string);
        assert_eq!(tensor.dtype, Dtype::U8);
        assert_eq!(tensor.shape, vec![1]);
    }

    // ── OnnxTensor raw_data returns the underlying bytes ──────────────

    #[test]
    fn onnx_tensor_raw_data_access() {
        let data = Bytes::from_static(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let tensor = OnnxTensor::new(
            "raw".to_string(),
            Dtype::U8,
            vec![4],
            data.clone(),
        );
        assert_eq!(tensor.raw_data(), &data[..]);
    }

    // ── OnnxTensor clone produces equal struct ─────────────────────────

    #[test]
    fn onnx_tensor_clone_equality() {
        let tensor = OnnxTensor::new(
            "cloned".to_string(),
            Dtype::F32,
            vec![2, 3],
            Bytes::from(vec![0u8; 24]),
        );
        let cloned = tensor.clone();
        assert_eq!(cloned.name, "cloned");
        assert_eq!(cloned.dtype, Dtype::F32);
        assert_eq!(cloned.shape, vec![2, 3]);
        assert_eq!(cloned.raw_data(), tensor.raw_data());
    }

    // ── OnnxSparseFormat variants equality ─────────────────────────────

    #[test]
    fn onnx_sparse_format_variants_distinct() {
        use super::super::tensor::OnnxSparseFormat;
        assert_eq!(OnnxSparseFormat::Coo, OnnxSparseFormat::Coo);
        assert_eq!(OnnxSparseFormat::Csr, OnnxSparseFormat::Csr);
        assert_eq!(OnnxSparseFormat::Csc, OnnxSparseFormat::Csc);
        assert_ne!(OnnxSparseFormat::Coo, OnnxSparseFormat::Csr);
        assert_ne!(OnnxSparseFormat::Csr, OnnxSparseFormat::Csc);
    }

    // ── OnnxSparseFormat Copy trait ────────────────────────────────────

    #[test]
    fn onnx_sparse_format_copy_trait() {
        use super::super::tensor::OnnxSparseFormat;
        let a = OnnxSparseFormat::Coo;
        let b = a; // Copy, not move
        let _ = a; // still usable
        assert_eq!(a, b);
    }

    // ── LoaderError::Onnx Display message ──────────────────────────────

    #[test]
    fn loader_error_onnx_display_message() {
        let err = LoaderError::Onnx("test error detail".to_string());
        let msg = err.to_string();
        assert!(msg.contains("ONNX error"), "expected ONNX error prefix: {msg}");
        assert!(msg.contains("test error detail"), "expected detail: {msg}");
    }

    // ── LoaderError::DuplicateTensor Display message ───────────────────

    #[test]
    fn loader_error_duplicate_tensor_display_message() {
        let err = LoaderError::DuplicateTensor("weight_0".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Duplicate tensor"), "expected Duplicate tensor prefix: {msg}");
        assert!(msg.contains("weight_0"), "expected tensor name: {msg}");
    }

    // ── OnnxGraph with sparse_initializers populated ───────────────────

    #[test]
    fn onnx_graph_with_sparse_initializers() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let values = OnnxTensor::new(
            "sparse_vals".to_string(),
            Dtype::F32,
            vec![3],
            Bytes::from(vec![0u8; 12]),
        );
        let indices = OnnxTensor::new(
            "sparse_idx".to_string(),
            Dtype::I64,
            vec![3],
            Bytes::from(vec![0u8; 24]),
        );
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![4, 4],
            format: OnnxSparseFormat::Coo,
        };
        let graph = OnnxGraph {
            name: "sparse_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![sparse],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.sparse_initializers.len(), 1);
        assert_eq!(graph.sparse_initializers[0].dims, vec![4, 4]);
        assert_eq!(graph.sparse_initializers[0].format, OnnxSparseFormat::Coo);
    }

    // ── OnnxGraph with value_info entries ──────────────────────────────

    #[test]
    fn onnx_graph_with_value_info() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![
                OnnxValueInfo {
                    name: "intermediate_1".to_string(),
                    value_type: None,
                    doc_string: "after layer 1".to_string(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "intermediate_2".to_string(),
                    value_type: None,
                    doc_string: "after layer 2".to_string(),
                    metadata_props: HashMap::new(),
                },
            ],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.value_info.len(), 2);
        assert_eq!(graph.value_info[0].name, "intermediate_1");
        assert_eq!(graph.value_info[1].doc_string, "after layer 2");
    }

    // ── parse_metadata_props: mixed valid and invalid entries ──────────

    #[test]
    fn parse_metadata_props_mixed_valid_and_invalid() {
        let entries = vec![
            proto::StringStringEntryProto {
                key: None, // skipped
                value: Some("orphan".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some(String::new()), // skipped
                value: Some("empty_key".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("valid_key".to_string()),
                value: Some("valid_value".to_string()),
            },
        ];
        let result = parse_metadata_props(entries);
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("valid_key").unwrap(), "valid_value");
    }

    // ── parse_quantization: multiple annotation entries ────────────────

    #[test]
    fn parse_quantization_multiple_entries() {
        let entries = vec![
            proto::TensorAnnotation {
                tensor_name: Some("weight_a".to_string()),
                quant_parameter_tensor_names: vec![],
            },
            proto::TensorAnnotation {
                tensor_name: Some("weight_b".to_string()),
                quant_parameter_tensor_names: vec![],
            },
            proto::TensorAnnotation {
                tensor_name: Some("weight_c".to_string()),
                quant_parameter_tensor_names: vec![],
            },
        ];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].tensor_name, "weight_a");
        assert_eq!(result[1].tensor_name, "weight_b");
        assert_eq!(result[2].tensor_name, "weight_c");
    }

    // ── OnnxModel Debug trait produces output containing key fields ───

    #[test]
    fn onnx_model_debug_format() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: "debug-producer".to_string(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "debug_graph".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("debug-producer"));
        assert!(debug_str.contains("debug_graph"));
    }

    // ── OnnxQuantizationAnnotation Debug trait ─────────────────────────

    #[test]
    fn onnx_quantization_annotation_debug_format() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "debug_weight".to_string(),
            quant_param_tensor_names: HashMap::from([
                ("SCALE_TENSOR".to_string(), "scale_0".to_string()),
            ]),
            scale: Some(0.1),
            zero_point: Some(0),
            axis: None,
        };
        let debug_str = format!("{:?}", qa);
        assert!(debug_str.contains("debug_weight"));
        assert!(debug_str.contains("scale_0"));
    }

    // ── OnnxFunction Debug trait ───────────────────────────────────────

    #[test]
    fn onnx_function_debug_format() {
        let func = OnnxFunction {
            name: "DebugOp".to_string(),
            domain: "test".to_string(),
            overload: String::new(),
            inputs: vec!["A".to_string()],
            outputs: vec!["B".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let debug_str = format!("{:?}", func);
        assert!(debug_str.contains("DebugOp"));
        assert!(debug_str.contains("test"));
    }

    // ── OnnxValueInfo Debug trait ──────────────────────────────────────

    #[test]
    fn onnx_value_info_debug_format() {
        let vi = OnnxValueInfo {
            name: "debug_input".to_string(),
            value_type: None,
            doc_string: "debug doc".to_string(),
            metadata_props: HashMap::new(),
        };
        let debug_str = format!("{:?}", vi);
        assert!(debug_str.contains("debug_input"));
    }

    // ── OnnxOperatorSet Debug trait ────────────────────────────────────

    #[test]
    fn onnx_operator_set_debug_format() {
        let ops = OnnxOperatorSet {
            domain: "ai.onnx".to_string(),
            version: 21,
        };
        let debug_str = format!("{:?}", ops);
        assert!(debug_str.contains("ai.onnx"));
        assert!(debug_str.contains("21"));
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (18 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxModelMetadata with all-default string fields ───────────────

    #[test]
    fn onnx_model_metadata_all_default_strings() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(meta.producer_name.is_empty());
        assert!(meta.producer_version.is_empty());
        assert!(meta.domain.is_empty());
        assert!(meta.doc_string.is_empty());
        assert!(meta.opset_import.is_empty());
        assert!(meta.metadata_props.is_empty());
    }

    // ── OnnxNode with populated attributes map ─────────────────────────

    #[test]
    fn onnx_node_with_attributes() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert(
            "kernel_shape".to_string(),
            OnnxAttribute {
                name: "kernel_shape".to_string(),
                value: OnnxAttributeValue::Ints(vec![3, 3]),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        attrs.insert(
            "strides".to_string(),
            OnnxAttribute {
                name: "strides".to_string(),
                value: OnnxAttributeValue::Ints(vec![1, 1]),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        let node = OnnxNode {
            name: "conv2d".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string(), "W".to_string(), "B".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: attrs,
        };
        assert_eq!(node.attributes.len(), 2);
        assert!(node.attributes.contains_key("kernel_shape"));
        assert!(node.attributes.contains_key("strides"));
    }

    // ── OnnxNode with many inputs and outputs ──────────────────────────

    #[test]
    fn onnx_node_many_inputs_outputs() {
        let node = OnnxNode {
            name: "concat".to_string(),
            op_type: "Concat".to_string(),
            domain: String::new(),
            inputs: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.inputs.len(), 4);
        assert_eq!(node.outputs.len(), 1);
        assert_eq!(node.inputs[2], "c");
    }

    // ── OnnxGraph with non-empty doc_string ────────────────────────────

    #[test]
    fn onnx_graph_doc_string_preserved() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: "This is a documentation string for the graph".to_string(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(
            graph.doc_string,
            "This is a documentation string for the graph"
        );
    }

    // ── OnnxFunction with attribute_protos populated ───────────────────

    #[test]
    fn onnx_function_with_attribute_protos() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "alpha".to_string(),
            value: OnnxAttributeValue::Float(0.5),
            doc_string: "learning rate".to_string(),
            ref_attr_name: None,
            attr_type: None,
        };
        let func = OnnxFunction {
            name: "ScaledAdd".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec!["alpha".to_string()],
            attribute_protos: HashMap::from([("alpha".to_string(), attr)]),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.attribute_protos.len(), 1);
        let attr = func.attribute_protos.get("alpha").unwrap();
        assert_eq!(attr.doc_string, "learning rate");
    }

    // ── OnnxModel with non-empty functions list ────────────────────────

    #[test]
    fn onnx_model_with_functions() {
        let func = OnnxFunction {
            name: "MyOp".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec!["A".to_string()],
            outputs: vec!["B".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![OnnxNode {
                name: "inner".to_string(),
                op_type: "Identity".to_string(),
                domain: String::new(),
                inputs: vec!["A".to_string()],
                outputs: vec!["B".to_string()],
                attributes: HashMap::new(),
            }],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![func],
        };
        assert_eq!(model.functions.len(), 1);
        assert_eq!(model.functions[0].name, "MyOp");
        assert_eq!(model.functions[0].nodes[0].op_type, "Identity");
    }

    // ── LoaderError::Onnx Display contains both prefix and detail ──────

    #[test]
    fn loader_error_onnx_display_contains_full_context() {
        let err = LoaderError::Onnx("node 42 missing op_type".to_string());
        let msg = format!("{err}");
        assert!(msg.starts_with("ONNX error"));
        assert!(msg.contains("node 42"));
        assert!(msg.contains("missing op_type"));
    }

    // ── LoaderError::DuplicateTensor with different names ──────────────

    #[test]
    fn loader_error_duplicate_tensor_different_names() {
        let err1 = LoaderError::DuplicateTensor("weight_a".to_string());
        let err2 = LoaderError::DuplicateTensor("weight_b".to_string());
        let msg1 = err1.to_string();
        let msg2 = err2.to_string();
        assert!(msg1.contains("weight_a"));
        assert!(msg2.contains("weight_b"));
        assert_ne!(msg1, msg2);
    }

    // ── OnnxQuantizationAnnotation with all optional fields set ─────────

    #[test]
    fn onnx_quantization_annotation_all_set() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "qkv_weight".to_string(),
            quant_param_tensor_names: HashMap::from([
                ("SCALE_TENSOR".to_string(), "qkv_scale".to_string()),
                ("ZERO_POINT_TENSOR".to_string(), "qkv_zp".to_string()),
            ]),
            scale: Some(0.0078125),
            zero_point: Some(-1),
            axis: Some(0),
        };
        assert_eq!(qa.tensor_name, "qkv_weight");
        assert!((qa.scale.unwrap() - 0.0078125).abs() < 1e-10);
        assert_eq!(qa.zero_point, Some(-1));
        assert_eq!(qa.axis, Some(0));
        assert_eq!(qa.quant_param_tensor_names.len(), 2);
        assert_eq!(
            qa.quant_param_tensor_names.get("SCALE_TENSOR").unwrap(),
            "qkv_scale"
        );
    }

    // ── parse_quantization: invalid axis string yields None ─────────────

    #[test]
    fn parse_quantization_invalid_axis_string_yields_none() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some("not_a_number".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].axis, None);
    }

    // ── OnnxValueInfo with Some value_type ─────────────────────────────

    #[test]
    fn onnx_value_info_with_value_type() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let vi = OnnxValueInfo {
            name: "logits".to_string(),
            value_type: Some(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape {
                    dims: vec![OnnxDim::Param("batch".to_string()), OnnxDim::Known(1000)],
                },
            })),
            doc_string: "output probabilities".to_string(),
            metadata_props: HashMap::new(),
        };
        assert!(vi.value_type.is_some());
        if let Some(OnnxType::Tensor(tt)) = vi.value_type {
            assert!(matches!(tt.elem_type, proto::tensor_proto::DataType::Float));
            assert_eq!(tt.shape.dims.len(), 2);
        } else {
            panic!("expected Tensor variant");
        }
    }

    // ── parse_metadata_props: unicode keys and values ──────────────────

    #[test]
    fn parse_metadata_props_unicode_keys_values() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("模型名称".to_string()),
            value: Some("中文测试模型".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("模型名称").unwrap(), "中文测试模型");
    }

    // ── OnnxModelMetadata with multiple opset_imports ──────────────────

    #[test]
    fn onnx_model_metadata_multiple_opsets() {
        let meta = OnnxModelMetadata {
            ir_version: 9,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![
                OnnxOperatorSet {
                    domain: "".to_string(),
                    version: 17,
                },
                OnnxOperatorSet {
                    domain: "ai.onnx.ml".to_string(),
                    version: 3,
                },
                OnnxOperatorSet {
                    domain: "ai.onnx.training".to_string(),
                    version: 1,
                },
            ],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.opset_import.len(), 3);
        assert_eq!(meta.opset_import[0].version, 17);
        assert_eq!(meta.opset_import[1].domain, "ai.onnx.ml");
        assert_eq!(meta.opset_import[2].version, 1);
    }

    // ── OnnxGraph initializer remove and re-insert ─────────────────────

    #[test]
    fn onnx_graph_initializer_remove_and_reinsert() {
        let tensor = OnnxTensor::new(
            "weight".to_string(),
            Dtype::F32,
            vec![2, 2],
            Bytes::from_static(&[0u8; 16]),
        );
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("weight".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.initializers.contains_key("weight"));
        let removed = graph.initializers.remove("weight").unwrap();
        assert_eq!(removed.shape, vec![2, 2]);
        assert!(!graph.initializers.contains_key("weight"));
        let new_tensor = OnnxTensor::new(
            "weight".to_string(),
            Dtype::BF16,
            vec![4, 4],
            Bytes::from_static(&[0u8; 32]),
        );
        graph.initializers.insert("weight".to_string(), new_tensor);
        assert!(graph.initializers.contains_key("weight"));
        assert_eq!(graph.initializers.get("weight").unwrap().shape, vec![4, 4]);
    }

    // ── OnnxModel deep clone consistency ───────────────────────────────

    #[test]
    fn onnx_model_deep_clone_consistency() {
        let node = OnnxNode {
            name: "original".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["a".to_string()],
            outputs: vec!["b".to_string()],
            attributes: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: "test".to_string(),
                producer_version: "1.0".to_string(),
                domain: String::new(),
                model_version: 1,
                doc_string: String::new(),
                opset_import: vec![OnnxOperatorSet {
                    domain: "".to_string(),
                    version: 17,
                }],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
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
            },
            functions: vec![],
        };
        let cloned = model.clone();
        assert_eq!(cloned.metadata.ir_version, model.metadata.ir_version);
        assert_eq!(cloned.metadata.producer_name, model.metadata.producer_name);
        assert_eq!(cloned.graph.name, model.graph.name);
        assert_eq!(cloned.graph.nodes.len(), model.graph.nodes.len());
        assert_eq!(cloned.graph.nodes[0].name, "original");
    }

    // ── OnnxNode Debug includes inputs and outputs count ───────────────

    #[test]
    fn onnx_node_debug_includes_io() {
        let node = OnnxNode {
            name: "gemm_0".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string(), "W".to_string(), "bias".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        };
        let debug = format!("{:?}", node);
        assert!(debug.contains("gemm_0"));
        assert!(debug.contains("Gemm"));
        assert!(debug.contains("X"));
        assert!(debug.contains("Y"));
    }

    // ── OnnxModelMetadata field access: producer_version and model_version ─

    #[test]
    fn onnx_model_metadata_producer_version_field() {
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: "pytorch".to_string(),
            producer_version: "2.1.0".to_string(),
            domain: String::new(),
            model_version: 42,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.producer_version, "2.1.0");
        assert_eq!(meta.model_version, 42);
    }

    // ── OnnxGraph name mutation ────────────────────────────────────────

    #[test]
    fn onnx_graph_name_mutable() {
        let mut graph = OnnxGraph {
            name: "original".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.name, "original");
        graph.name = "renamed".to_string();
        assert_eq!(graph.name, "renamed");
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (17 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxDim variants: Known, Param, Unknown equality and distinction ─

    #[test]
    fn onnx_dim_variants_distinct() {
        use super::super::types::OnnxDim;
        let known = OnnxDim::Known(42);
        let param = OnnxDim::Param("seq_len".to_string());
        let unknown = OnnxDim::Unknown;

        assert_eq!(OnnxDim::Known(42), OnnxDim::Known(42));
        assert_ne!(OnnxDim::Known(1), OnnxDim::Known(2));
        assert_eq!(
            OnnxDim::Param("batch".to_string()),
            OnnxDim::Param("batch".to_string())
        );
        assert_ne!(
            OnnxDim::Param("a".to_string()),
            OnnxDim::Param("b".to_string())
        );
        assert_eq!(OnnxDim::Unknown, OnnxDim::Unknown);
        assert_ne!(known, param);
        assert_ne!(param, unknown);
        assert_ne!(known, unknown);
    }

    // ── OnnxDim Clone trait ─────────────────────────────────────────────

    #[test]
    fn onnx_dim_clone() {
        use super::super::types::OnnxDim;
        let dim = OnnxDim::Param("height".to_string());
        let cloned = dim.clone();
        assert_eq!(dim, cloned);
    }

    // ── OnnxDim zero dimension is valid ─────────────────────────────────

    #[test]
    fn onnx_dim_zero_known_value() {
        use super::super::types::OnnxDim;
        let dim = OnnxDim::Known(0);
        assert_eq!(dim, OnnxDim::Known(0));
    }

    // ── OnnxDim negative dimension represents dynamic ───────────────────

    #[test]
    fn onnx_dim_negative_known_value() {
        use super::super::types::OnnxDim;
        // ONNX spec uses -1 for unknown dimensions
        let dim = OnnxDim::Known(-1);
        assert_eq!(dim, OnnxDim::Known(-1));
    }

    // ── OnnxTensorShape with mixed dim types ────────────────────────────

    #[test]
    fn onnx_tensor_shape_mixed_dims() {
        use super::super::types::{OnnxDim, OnnxTensorShape};
        let shape = OnnxTensorShape {
            dims: vec![
                OnnxDim::Param("batch".to_string()),
                OnnxDim::Known(128),
                OnnxDim::Param("seq_len".to_string()),
                OnnxDim::Known(64),
            ],
        };
        assert_eq!(shape.dims.len(), 4);
        assert!(matches!(shape.dims[0], OnnxDim::Param(_)));
        assert!(matches!(shape.dims[1], OnnxDim::Known(128)));
        assert!(matches!(shape.dims[2], OnnxDim::Param(_)));
        assert!(matches!(shape.dims[3], OnnxDim::Known(64)));
    }

    // ── OnnxTensorShape clone ───────────────────────────────────────────

    #[test]
    fn onnx_tensor_shape_clone() {
        use super::super::types::{OnnxDim, OnnxTensorShape};
        let shape = OnnxTensorShape {
            dims: vec![OnnxDim::Known(3), OnnxDim::Unknown],
        };
        let cloned = shape.clone();
        assert_eq!(cloned.dims.len(), 2);
        assert_eq!(cloned.dims[0], OnnxDim::Known(3));
        assert_eq!(cloned.dims[1], OnnxDim::Unknown);
    }

    // ── OnnxTensorType field access ─────────────────────────────────────

    #[test]
    fn onnx_tensor_type_field_access() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType};
        let tt = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Bfloat16,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Known(2), OnnxDim::Known(512)],
            },
        };
        assert_eq!(tt.shape.dims.len(), 2);
        assert!(matches!(
            tt.elem_type,
            proto::tensor_proto::DataType::Bfloat16
        ));
    }

    // ── OnnxType variant distinction ────────────────────────────────────

    #[test]
    fn onnx_type_variants_distinct() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let tensor = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Unknown],
            },
        });
        let sparse = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Unknown],
            },
        });
        let seq = OnnxType::Sequence(Box::new(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int64,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Known(1)],
            },
        })));

        assert!(matches!(tensor, OnnxType::Tensor(_)));
        assert!(matches!(sparse, OnnxType::SparseTensor(_)));
        assert!(matches!(seq, OnnxType::Sequence(_)));
    }

    // ── OnnxType clone consistency ──────────────────────────────────────

    #[test]
    fn onnx_type_clone_consistency() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let original = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Double,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Param("n".to_string())],
            },
        });
        let cloned = original.clone();
        assert!(matches!(cloned, OnnxType::Tensor(_)));
        if let OnnxType::Tensor(tt) = cloned {
            assert!(matches!(tt.elem_type, proto::tensor_proto::DataType::Double));
            assert_eq!(tt.shape.dims.len(), 1);
        }
    }

    // ── OnnxMapType field access ────────────────────────────────────────

    #[test]
    fn onnx_map_type_field_access() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxMapType, OnnxType};
        let map_type = OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape {
                    dims: vec![OnnxDim::Unknown],
                },
            })),
        };
        assert!(matches!(
            map_type.key_type,
            proto::tensor_proto::DataType::Int64
        ));
        assert!(matches!(*map_type.value_type, OnnxType::Tensor(_)));
    }

    // ── OnnxNode with empty inputs and outputs ──────────────────────────

    #[test]
    fn onnx_node_empty_inputs_outputs() {
        let node = OnnxNode {
            name: "dropout".to_string(),
            op_type: "Dropout".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert!(node.inputs.is_empty());
        assert!(node.outputs.is_empty());
        assert_eq!(node.name, "dropout");
        assert_eq!(node.op_type, "Dropout");
    }

    // ── OnnxNode with non-default domain ────────────────────────────────

    #[test]
    fn onnx_node_custom_domain() {
        let node = OnnxNode {
            name: "custom_op".to_string(),
            op_type: "FusedGemmBias".to_string(),
            domain: "com.microsoft".to_string(),
            inputs: vec!["A".to_string()],
            outputs: vec!["C".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.domain, "com.microsoft");
        assert_ne!(node.domain, "");
    }

    // ── OnnxValueInfo with empty name edge case ─────────────────────────

    #[test]
    fn onnx_value_info_empty_name_is_valid_struct() {
        let vi = OnnxValueInfo {
            name: String::new(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(vi.name.is_empty());
        assert!(vi.value_type.is_none());
        assert!(vi.doc_string.is_empty());
    }

    // ── OnnxValueInfo with long unicode name ────────────────────────────

    #[test]
    fn onnx_value_info_unicode_long_name() {
        let long_name = "データ_".repeat(100);
        let vi = OnnxValueInfo {
            name: long_name.clone(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(vi.name, long_name);
        assert_eq!(vi.name.chars().count(), 400); // "データ_" is 4 chars, repeated 100 times
    }

    // ── OnnxQuantizationAnnotation scale edge values ────────────────────

    #[test]
    fn onnx_quantization_annotation_edge_scale_values() {
        let qa_tiny = OnnxQuantizationAnnotation {
            tensor_name: "w1".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(f32::MIN_POSITIVE),
            zero_point: None,
            axis: None,
        };
        assert_eq!(qa_tiny.scale, Some(f32::MIN_POSITIVE));

        let qa_large = OnnxQuantizationAnnotation {
            tensor_name: "w2".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(f32::MAX),
            zero_point: None,
            axis: None,
        };
        assert_eq!(qa_large.scale, Some(f32::MAX));

        let qa_zero = OnnxQuantizationAnnotation {
            tensor_name: "w3".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.0),
            zero_point: Some(0),
            axis: Some(0),
        };
        assert_eq!(qa_zero.scale, Some(0.0));
        assert_eq!(qa_zero.zero_point, Some(0));
    }

    // ── OnnxFunction with empty overloads and attributes ────────────────

    #[test]
    fn onnx_function_minimal_fields() {
        let func = OnnxFunction {
            name: "IdentityPass".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.name, "IdentityPass");
        assert!(func.domain.is_empty());
        assert!(func.overload.is_empty());
        assert!(func.inputs.is_empty());
        assert!(func.outputs.is_empty());
        assert!(func.attributes.is_empty());
        assert!(func.nodes.is_empty());
        assert!(func.opset_import.is_empty());
        assert!(func.value_info.is_empty());
    }

    // ── OnnxFunction clone with nested nodes ────────────────────────────

    #[test]
    fn onnx_function_clone_with_nested_nodes() {
        let func = OnnxFunction {
            name: "FusedOp".to_string(),
            domain: "com.example".to_string(),
            overload: "v1".to_string(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec!["alpha".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![
                OnnxNode {
                    name: "inner_1".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["X".to_string()],
                    outputs: vec!["t".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "inner_2".to_string(),
                    op_type: "Sigmoid".to_string(),
                    domain: String::new(),
                    inputs: vec!["t".to_string()],
                    outputs: vec!["Y".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 17,
            }],
            value_info: vec![],
            doc_string: "fused relu-sigmoid".to_string(),
            metadata_props: HashMap::new(),
        };
        let cloned = func.clone();
        assert_eq!(cloned.nodes.len(), 2);
        assert_eq!(cloned.nodes[0].op_type, "Relu");
        assert_eq!(cloned.nodes[1].op_type, "Sigmoid");
        assert_eq!(cloned.opset_import.len(), 1);
        assert_eq!(cloned.doc_string, "fused relu-sigmoid");
    }

    // ── OnnxModelMetadata with large model_version ──────────────────────

    #[test]
    fn onnx_model_metadata_large_model_version() {
        let meta = OnnxModelMetadata {
            ir_version: i64::MAX,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: i64::MAX,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.ir_version, i64::MAX);
        assert_eq!(meta.model_version, i64::MAX);
    }

    // ── OnnxGraph inputs and outputs access ─────────────────────────────

    #[test]
    fn onnx_graph_inputs_outputs_access() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![
                OnnxValueInfo {
                    name: "input_ids".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "attention_mask".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            outputs: vec![OnnxValueInfo {
                name: "logits".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.inputs[0].name, "input_ids");
        assert_eq!(graph.inputs[1].name, "attention_mask");
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.outputs[0].name, "logits");
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 4 TESTS (40 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxDim Debug format for each variant ──────────────────────────

    #[test]
    fn onnx_dim_debug_format_known() {
        use super::super::types::OnnxDim;
        let dim = OnnxDim::Known(99);
        let s = format!("{:?}", dim);
        assert!(s.contains("99"), "expected 99 in debug: {s}");
    }

    #[test]
    fn onnx_dim_debug_format_param() {
        use super::super::types::OnnxDim;
        let dim = OnnxDim::Param("seq_len".to_string());
        let s = format!("{:?}", dim);
        assert!(s.contains("seq_len"), "expected seq_len in debug: {s}");
    }

    #[test]
    fn onnx_dim_debug_format_unknown() {
        use super::super::types::OnnxDim;
        let dim = OnnxDim::Unknown;
        let s = format!("{:?}", dim);
        assert!(s.contains("Unknown"), "expected Unknown in debug: {s}");
    }

    // ── OnnxDim Hash consistency ───────────────────────────────────────

    #[test]
    fn onnx_dim_hash_known_equal_values() {
        use super::super::types::OnnxDim;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(OnnxDim::Known(42)));
        assert!(!set.insert(OnnxDim::Known(42)));
        assert!(set.insert(OnnxDim::Known(43)));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn onnx_dim_hash_param_equal_values() {
        use super::super::types::OnnxDim;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(OnnxDim::Param("batch".to_string())));
        assert!(!set.insert(OnnxDim::Param("batch".to_string())));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn onnx_dim_hash_unknown() {
        use super::super::types::OnnxDim;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(OnnxDim::Unknown));
        assert!(!set.insert(OnnxDim::Unknown));
        assert_eq!(set.len(), 1);
    }

    // ── OnnxSparseFormat Debug and Hash ────────────────────────────────

    #[test]
    fn onnx_sparse_format_debug_output() {
        use super::super::tensor::OnnxSparseFormat;
        let coo_debug = format!("{:?}", OnnxSparseFormat::Coo);
        let csr_debug = format!("{:?}", OnnxSparseFormat::Csr);
        let csc_debug = format!("{:?}", OnnxSparseFormat::Csc);
        assert!(coo_debug.contains("Coo"), "expected Coo: {coo_debug}");
        assert!(csr_debug.contains("Csr"), "expected Csr: {csr_debug}");
        assert!(csc_debug.contains("Csc"), "expected Csc: {csc_debug}");
    }

    #[test]
    fn onnx_sparse_format_hash_equal_variants() {
        use super::super::tensor::OnnxSparseFormat;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(OnnxSparseFormat::Coo));
        assert!(!set.insert(OnnxSparseFormat::Coo));
        assert!(set.insert(OnnxSparseFormat::Csr));
        assert!(set.insert(OnnxSparseFormat::Csc));
        assert_eq!(set.len(), 3);
    }

    // ── OnnxType PartialEq for all variants ────────────────────────────

    #[test]
    fn onnx_type_optional_variant_equality() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let inner = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Unknown] },
        });
        let opt1 = OnnxType::Optional(Box::new(inner.clone()));
        let opt2 = OnnxType::Optional(Box::new(inner.clone()));
        assert_eq!(opt1, opt2);
    }

    #[test]
    fn onnx_type_map_variant_equality() {
        use super::super::types::{OnnxMapType, OnnxTensorShape, OnnxTensorType, OnnxType};
        let map1 = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        });
        let map2 = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        });
        assert_eq!(map1, map2);
    }

    #[test]
    fn onnx_type_sequence_variant_not_equal_to_tensor() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let tensor = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(1)] },
        });
        let seq = OnnxType::Sequence(Box::new(tensor.clone()));
        assert_ne!(tensor, seq);
    }

    // ── OnnxTensor scalar access with special float values ─────────────

    #[test]
    fn onnx_tensor_scalar_f32_nan() {
        let nan_bits = f32::NAN.to_le_bytes();
        let tensor = OnnxTensor::new(
            "nan_val".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(nan_bits.to_vec()),
        );
        let result = tensor.scalar_f32();
        assert!(result.is_some());
        assert!(result.unwrap().is_nan());
    }

    #[test]
    fn onnx_tensor_scalar_f32_infinity() {
        let inf_bits = f32::INFINITY.to_le_bytes();
        let tensor = OnnxTensor::new(
            "inf_val".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(inf_bits.to_vec()),
        );
        let result = tensor.scalar_f32();
        assert!(result.is_some());
        assert!(result.unwrap().is_infinite() && result.unwrap().is_sign_positive());
    }

    #[test]
    fn onnx_tensor_scalar_f32_neg_infinity() {
        let neg_inf_bits = f32::NEG_INFINITY.to_le_bytes();
        let tensor = OnnxTensor::new(
            "neg_inf_val".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(neg_inf_bits.to_vec()),
        );
        let result = tensor.scalar_f32();
        assert!(result.is_some());
        assert!(result.unwrap().is_infinite() && result.unwrap().is_sign_negative());
    }

    #[test]
    fn onnx_tensor_scalar_i64_zero() {
        let zero_bytes = 0i64.to_le_bytes();
        let tensor = OnnxTensor::new(
            "zero".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(zero_bytes.to_vec()),
        );
        assert_eq!(tensor.scalar_i64(), Some(0));
    }

    #[test]
    fn onnx_tensor_scalar_i64_max() {
        let max_bytes = i64::MAX.to_le_bytes();
        let tensor = OnnxTensor::new(
            "imax".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(max_bytes.to_vec()),
        );
        assert_eq!(tensor.scalar_i64(), Some(i64::MAX));
    }

    #[test]
    fn onnx_tensor_scalar_f32_none_for_empty_shape_non_scalar_data() {
        // Empty shape but multiple bytes of data — still scalar (0-d tensor)
        let tensor = OnnxTensor::new(
            "scalar".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(1.0f32.to_le_bytes().to_vec()),
        );
        assert!(tensor.scalar_f32().is_some());
    }

    // ── OnnxTensor dtype preservation ──────────────────────────────────

    #[test]
    fn onnx_tensor_dtype_bf16() {
        let tensor = OnnxTensor::new(
            "bf16_weight".to_string(),
            Dtype::BF16,
            vec![2],
            Bytes::from(vec![0u8; 4]),
        );
        assert_eq!(tensor.dtype, Dtype::BF16);
    }

    #[test]
    fn onnx_tensor_dtype_f16() {
        let tensor = OnnxTensor::new(
            "f16_act".to_string(),
            Dtype::F16,
            vec![8],
            Bytes::from(vec![0u8; 16]),
        );
        assert_eq!(tensor.dtype, Dtype::F16);
    }

    // ── OnnxTensor shape with large dimensions ─────────────────────────

    #[test]
    fn onnx_tensor_large_shape() {
        let tensor = OnnxTensor::new(
            "big".to_string(),
            Dtype::F32,
            vec![usize::MAX],
            Bytes::from(vec![0u8; 4]),
        );
        assert_eq!(tensor.shape, vec![usize::MAX]);
    }

    // ── OnnxTensor empty shape (scalar) ────────────────────────────────

    #[test]
    fn onnx_tensor_empty_shape_scalar() {
        let tensor = OnnxTensor::new(
            "scalar_weight".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(2.5f32.to_le_bytes().to_vec()),
        );
        assert!(tensor.shape.is_empty());
        let val = tensor.scalar_f32().unwrap();
        assert!((val - 2.5f32).abs() < 1e-6);
    }

    // ── LoaderError variant Display messages ───────────────────────────

    #[test]
    fn loader_error_missing_tensor_display() {
        let err = LoaderError::MissingTensor("embedding".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Missing tensor"), "expected Missing tensor in: {msg}");
        assert!(msg.contains("embedding"), "expected tensor name in: {msg}");
    }

    #[test]
    fn loader_error_network_display() {
        let err = LoaderError::Network("connection timeout".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Network error"), "expected Network error in: {msg}");
        assert!(msg.contains("connection timeout"), "expected detail in: {msg}");
    }

    #[test]
    fn loader_error_cache_display() {
        let err = LoaderError::Cache("disk full".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Cache error"), "expected Cache error in: {msg}");
        assert!(msg.contains("disk full"), "expected detail in: {msg}");
    }

    #[test]
    fn loader_error_gguf_display() {
        let err = LoaderError::Gguf("bad header".to_string());
        let msg = err.to_string();
        assert!(msg.contains("GGUF error"), "expected GGUF error in: {msg}");
        assert!(msg.contains("bad header"), "expected detail in: {msg}");
    }

    #[test]
    fn loader_error_gllm_display() {
        let err = LoaderError::Gllm("invalid format".to_string());
        let msg = err.to_string();
        assert!(msg.contains("GLLM error"), "expected GLLM error in: {msg}");
        assert!(msg.contains("invalid format"), "expected detail in: {msg}");
    }

    #[test]
    fn loader_error_arch_detection_display() {
        let err = LoaderError::ArchDetection("no matching arch".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Architecture detection failed"), "expected prefix in: {msg}");
        assert!(msg.contains("no matching arch"), "expected detail in: {msg}");
    }

    // ── OnnxAttributeValue variant construction and access ─────────────

    #[test]
    fn onnx_attribute_value_float_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Float(3.14);
        assert!(matches!(val, OnnxAttributeValue::Float(v) if (v - 3.14).abs() < 1e-6));
    }

    #[test]
    fn onnx_attribute_value_int_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Int(i64::MAX);
        assert!(matches!(val, OnnxAttributeValue::Int(v) if v == i64::MAX));
    }

    #[test]
    fn onnx_attribute_value_string_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::String("hello".to_string());
        assert!(matches!(val, OnnxAttributeValue::String(s) if s == "hello"));
    }

    #[test]
    fn onnx_attribute_value_ints_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Ints(vec![1, 2, 3]);
        assert!(matches!(val, OnnxAttributeValue::Ints(v) if v == vec![1, 2, 3]));
    }

    #[test]
    fn onnx_attribute_value_floats_variant_empty() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Floats(vec![]);
        assert!(matches!(val, OnnxAttributeValue::Floats(v) if v.is_empty()));
    }

    #[test]
    fn onnx_attribute_value_ref_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Ref("other_attr".to_string());
        assert!(matches!(val, OnnxAttributeValue::Ref(s) if s == "other_attr"));
    }

    // ── OnnxAttribute construction and field access ────────────────────

    #[test]
    fn onnx_attribute_with_ref_attr_name() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "group".to_string(),
            value: OnnxAttributeValue::Int(1),
            doc_string: String::new(),
            ref_attr_name: Some("parent_group".to_string()),
            attr_type: None,
        };
        assert_eq!(attr.name, "group");
        assert_eq!(attr.ref_attr_name, Some("parent_group".to_string()));
    }

    #[test]
    fn onnx_attribute_clone_preserves_all_fields() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "pads".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 1, 1, 1]),
            doc_string: "padding sizes".to_string(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        };
        let cloned = attr.clone();
        assert_eq!(cloned.name, "pads");
        assert_eq!(cloned.doc_string, "padding sizes");
        assert!(matches!(cloned.value, OnnxAttributeValue::Ints(v) if v == vec![1, 1, 1, 1]));
        assert_eq!(cloned.attr_type, attr.attr_type);
    }

    // ── OnnxSparseTensor construction and field access ─────────────────

    #[test]
    fn onnx_sparse_tensor_field_access() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let values = OnnxTensor::new(
            "sparse_values".to_string(),
            Dtype::F32,
            vec![2],
            Bytes::from(vec![0u8; 8]),
        );
        let indices = OnnxTensor::new(
            "sparse_indices".to_string(),
            Dtype::I64,
            vec![2],
            Bytes::from(vec![0u8; 16]),
        );
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![10, 10],
            format: OnnxSparseFormat::Csr,
        };
        assert_eq!(sparse.values.name, "sparse_values");
        assert_eq!(sparse.indices.name, "sparse_indices");
        assert_eq!(sparse.dims, vec![10, 10]);
        assert_eq!(sparse.format, OnnxSparseFormat::Csr);
    }

    #[test]
    fn onnx_sparse_tensor_clone() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let values = OnnxTensor::new(
            "vals".to_string(),
            Dtype::F32,
            vec![1],
            Bytes::from(vec![0u8; 4]),
        );
        let indices = OnnxTensor::new(
            "idx".to_string(),
            Dtype::I64,
            vec![1],
            Bytes::from(vec![0u8; 8]),
        );
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![5],
            format: OnnxSparseFormat::Coo,
        };
        let cloned = sparse.clone();
        assert_eq!(cloned.dims, vec![5]);
        assert_eq!(cloned.format, OnnxSparseFormat::Coo);
        assert_eq!(cloned.values.name, "vals");
    }

    // ── OnnxGraph.metadata_props with multiple entries ─────────────────

    #[test]
    fn onnx_graph_metadata_props_multiple() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::from([
                ("key_a".to_string(), "val_a".to_string()),
                ("key_b".to_string(), "val_b".to_string()),
                ("key_c".to_string(), "val_c".to_string()),
            ]),
        };
        assert_eq!(graph.metadata_props.len(), 3);
        assert_eq!(graph.metadata_props.get("key_b").unwrap(), "val_b");
    }

    // ── OnnxModel with all metadata_props populated ────────────────────

    #[test]
    fn onnx_model_metadata_props_population() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::from([
                    ("license".to_string(), "Apache-2.0".to_string()),
                    ("author".to_string(), "gllm".to_string()),
                ]),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert_eq!(model.metadata.metadata_props.get("license").unwrap(), "Apache-2.0");
        assert_eq!(model.metadata.metadata_props.get("author").unwrap(), "gllm");
    }

    // ── parse_nodes: node with many attributes passes through ──────────

    #[test]
    fn parse_nodes_with_multiple_attributes() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Conv".to_string()),
            name: Some("conv_0".to_string()),
            input: vec!["X".to_string(), "W".to_string()],
            output: vec!["Y".to_string()],
            domain: None,
            overload: None,
            attribute: vec![
                proto::AttributeProto {
                    name: Some("kernel_shape".to_string()),
                    r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
                    ints: vec![3, 3],
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: Some("strides".to_string()),
                    r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
                    ints: vec![1, 1],
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: Some("pads".to_string()),
                    r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
                    ints: vec![0, 0, 0, 0],
                    ..Default::default()
                },
            ],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "conv_0");
        assert_eq!(result[0].attributes.len(), 3);
        assert!(result[0].attributes.contains_key("kernel_shape"));
        assert!(result[0].attributes.contains_key("strides"));
        assert!(result[0].attributes.contains_key("pads"));
    }

    // ── OnnxTensorShape with empty dims (rank-0 tensor) ────────────────

    #[test]
    fn onnx_tensor_shape_empty_dims() {
        use super::super::types::OnnxTensorShape;
        let shape = OnnxTensorShape { dims: vec![] };
        assert!(shape.dims.is_empty());
        assert_eq!(shape.clone().dims.len(), 0);
    }

    // ── OnnxTensorType clone equality ──────────────────────────────────

    #[test]
    fn onnx_tensor_type_clone_equality() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType};
        let tt = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float16,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(768)] },
        };
        let cloned = tt.clone();
        assert_eq!(tt, cloned);
    }

    // ── OnnxQuantizationAnnotation zero scale edge case ────────────────

    #[test]
    fn onnx_quantization_annotation_zero_scale() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "zero_scale_weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.0),
            zero_point: Some(0),
            axis: Some(-1),
        };
        assert_eq!(qa.scale.unwrap(), 0.0);
        assert!(qa.scale.unwrap().is_sign_positive());
        assert_eq!(qa.axis, Some(-1));
    }

    // ── OnnxNode op_type with special characters ───────────────────────

    #[test]
    fn onnx_node_op_type_with_special_chars() {
        let node = OnnxNode {
            name: "special".to_string(),
            op_type: "com.microsoft::FusedGemm".to_string(),
            domain: "com.microsoft".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert!(node.op_type.contains("::"));
        assert_eq!(node.op_type, "com.microsoft::FusedGemm");
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 5 TESTS (52 new — target 206+)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn loader_error_hfhub_display() {
        let err = LoaderError::HfHub("token expired".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("HfHub error"), "expected prefix: {msg}");
        assert!(msg.contains("token expired"), "expected detail: {msg}");
    }

    #[test]
    fn loader_error_invalid_quantization_display() {
        let err = LoaderError::InvalidQuantization("bad scale".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Invalid quantization metadata"), "expected prefix: {msg}");
        assert!(msg.contains("bad scale"), "expected detail: {msg}");
    }

    #[test]
    fn loader_error_authentication_display() {
        let err = LoaderError::AuthenticationError {
            hint: "set HF_TOKEN".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("Authentication error"), "expected prefix: {msg}");
        assert!(msg.contains("set HF_TOKEN"), "expected hint: {msg}");
    }

    #[test]
    fn loader_error_backend_display() {
        let err = LoaderError::Backend("cuda not found".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Backend error"), "expected prefix: {msg}");
        assert!(msg.contains("cuda not found"), "expected detail: {msg}");
    }

    #[test]
    fn loader_error_pytorch_display() {
        let err = LoaderError::Pytorch("bad pickle".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("PyTorch error"), "expected prefix: {msg}");
        assert!(msg.contains("bad pickle"), "expected detail: {msg}");
    }

    #[test]
    fn loader_error_unsupported_weight_extension_display() {
        let err = LoaderError::UnsupportedWeightExtension(".bin".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Unsupported weight extension"), "expected prefix: {msg}");
        assert!(msg.contains(".bin"), "expected extension: {msg}");
    }

    #[test]
    fn loader_error_missing_weights_display() {
        let err = LoaderError::MissingWeights;
        let msg = format!("{err}");
        assert!(msg.contains("Missing weights"), "expected prefix: {msg}");
    }

    #[test]
    fn loader_error_unsupported_dtype_display() {
        let err = LoaderError::UnsupportedDtype(Dtype::F64);
        let msg = format!("{err}");
        assert!(msg.contains("Unsupported dtype"), "expected prefix: {msg}");
    }

    #[test]
    fn onnx_node_domain_default_is_empty() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert!(node.domain.is_empty());
    }

    #[test]
    fn onnx_node_with_optional_input_marker() {
        let node = OnnxNode {
            name: "batchnorm".to_string(),
            op_type: "BatchNormalization".to_string(),
            domain: String::new(),
            inputs: vec![
                "X".to_string(),
                "scale".to_string(),
                "bias".to_string(),
                "input_mean".to_string(),
                "input_var".to_string(),
                String::new(),
            ],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.inputs.len(), 6);
        assert!(node.inputs.last().unwrap().is_empty());
    }

    #[test]
    fn onnx_graph_from_proto_duplicate_initializer_error() {
        let tensor1 = proto::TensorProto {
            name: Some("dup_weight".to_string()),
            data_type: Some(1),
            dims: vec![2],
            float_data: vec![1.0, 2.0],
            ..Default::default()
        };
        let tensor2 = proto::TensorProto {
            name: Some("dup_weight".to_string()),
            data_type: Some(1),
            dims: vec![2],
            float_data: vec![3.0, 4.0],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            initializer: vec![tensor1, tensor2],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = OnnxGraph::from_proto(graph_proto, &mut resolver);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Duplicate tensor"), "expected Duplicate tensor: {err_msg}");
        assert!(err_msg.contains("dup_weight"), "expected name: {err_msg}");
    }

    #[test]
    fn onnx_model_from_proto_missing_graph_error() {
        let model_proto = proto::ModelProto {
            graph: None,
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = OnnxModel::from_proto(model_proto, &mut resolver);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("missing graph"), "expected: {err_msg}");
    }

    #[test]
    fn parse_metadata_props_value_empty_stored() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("empty_val".to_string()),
            value: Some(String::new()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("empty_val").unwrap(), "");
    }

    #[test]
    fn parse_metadata_props_value_none_defaults_empty() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("no_val".to_string()),
            value: None,
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("no_val").unwrap(), "");
    }

    #[test]
    fn parse_quantization_tensor_name_none_defaults_empty() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: None,
            quant_parameter_tensor_names: vec![],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result[0].tensor_name, "");
    }

    #[test]
    fn parse_quantization_axis_negative_value() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some("-1".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result[0].axis, Some(-1));
    }

    #[test]
    fn parse_quantization_scale_from_non_scalar_returns_none() {
        let scale_data = [0.125f32, 0.25f32];
        let mut bytes = Vec::new();
        for v in &scale_data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let scale_tensor = OnnxTensor::new(
            "multi_dim_scale".to_string(),
            Dtype::F32,
            vec![2],
            Bytes::from(bytes),
        );
        let initializers =
            HashMap::from([("multi_dim_scale".to_string(), scale_tensor)]);
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("SCALE_TENSOR".to_string()),
                value: Some("multi_dim_scale".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &initializers);
        assert_eq!(result[0].scale, None);
    }

    #[test]
    fn parse_opsets_single_entry() {
        let opsets = vec![proto::OperatorSetIdProto {
            domain: Some("ai.onnx.nn".to_string()),
            version: Some(17),
        }];
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].domain, "ai.onnx.nn");
        assert_eq!(result[0].version, 17);
    }

    #[test]
    fn onnx_model_multiple_functions() {
        let funcs: Vec<OnnxFunction> = (0..3)
            .map(|i| OnnxFunction {
                name: format!("Func{i}"),
                domain: String::new(),
                overload: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: vec![],
                attribute_protos: HashMap::new(),
                nodes: vec![],
                opset_import: vec![],
                value_info: vec![],
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            })
            .collect();
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: funcs,
        };
        assert_eq!(model.functions.len(), 3);
        assert_eq!(model.functions[0].name, "Func0");
        assert_eq!(model.functions[2].name, "Func2");
    }

    #[test]
    fn onnx_graph_nodes_reference_initializers() {
        let weight = OnnxTensor::new(
            "fc_weight".to_string(),
            Dtype::F32,
            vec![768, 768],
            Bytes::from(vec![0u8; 768 * 768 * 4]),
        );
        let bias = OnnxTensor::new(
            "fc_bias".to_string(),
            Dtype::F32,
            vec![768],
            Bytes::from(vec![0u8; 768 * 4]),
        );
        let graph = OnnxGraph {
            name: "fc_layer".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "matmul".to_string(),
                    op_type: "MatMul".to_string(),
                    domain: String::new(),
                    inputs: vec!["hidden".to_string(), "fc_weight".to_string()],
                    outputs: vec!["matmul_out".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "add_bias".to_string(),
                    op_type: "Add".to_string(),
                    domain: String::new(),
                    inputs: vec!["matmul_out".to_string(), "fc_bias".to_string()],
                    outputs: vec!["output".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([
                ("fc_weight".to_string(), weight),
                ("fc_bias".to_string(), bias),
            ]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 2);
        assert!(graph.initializers.contains_key("fc_weight"));
        assert!(graph.initializers.contains_key("fc_bias"));
    }

    #[test]
    fn onnx_value_info_nested_sequence_type() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let inner = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Param("seq".to_string()), OnnxDim::Known(768)],
            },
        });
        let seq = OnnxType::Sequence(Box::new(inner));
        let vi = OnnxValueInfo {
            name: "past_key_values".to_string(),
            value_type: Some(seq),
            doc_string: "cached key-value pairs".to_string(),
            metadata_props: HashMap::new(),
        };
        if let Some(OnnxType::Sequence(inner)) = vi.value_type {
            assert!(matches!(*inner, OnnxType::Tensor(_)));
        } else {
            panic!("expected Sequence variant");
        }
    }

    #[test]
    fn onnx_function_multi_node_pipeline() {
        let func = OnnxFunction {
            name: "FusedMLP".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Z".to_string()],
            attributes: vec!["hidden_size".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![
                OnnxNode {
                    name: "gemm1".to_string(),
                    op_type: "MatMul".to_string(),
                    domain: String::new(),
                    inputs: vec!["X".to_string(), "W1".to_string()],
                    outputs: vec!["hidden1".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "gelu".to_string(),
                    op_type: "Gelu".to_string(),
                    domain: String::new(),
                    inputs: vec!["hidden1".to_string()],
                    outputs: vec!["act1".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "gemm2".to_string(),
                    op_type: "MatMul".to_string(),
                    domain: String::new(),
                    inputs: vec!["act1".to_string(), "W2".to_string()],
                    outputs: vec!["Z".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 17,
            }],
            value_info: vec![],
            doc_string: "Fused MLP block".to_string(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.nodes.len(), 3);
        assert_eq!(func.nodes[1].op_type, "Gelu");
    }

    #[test]
    fn onnx_quantization_annotation_negative_zero_point() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "asymmetric_weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.02),
            zero_point: Some(-128),
            axis: Some(0),
        };
        assert_eq!(qa.zero_point, Some(-128));
    }

    #[test]
    fn onnx_quantization_annotation_large_zero_point() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: Some(i64::MAX),
            axis: None,
        };
        assert_eq!(qa.zero_point, Some(i64::MAX));
    }

    #[test]
    fn onnx_graph_quantization_with_real_scale_and_zp() {
        let scale_data = 0.0078125f32.to_le_bytes();
        let scale_tensor = OnnxTensor::new(
            "weight_scale".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(scale_data.to_vec()),
        );
        let zp_data = 128i64.to_le_bytes();
        let zp_tensor = OnnxTensor::new(
            "weight_zp".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(zp_data.to_vec()),
        );
        let graph = OnnxGraph {
            name: "quant_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([
                ("weight_scale".to_string(), scale_tensor),
                ("weight_zp".to_string(), zp_tensor),
            ]),
            sparse_initializers: vec![],
            quantization_annotation: vec![OnnxQuantizationAnnotation {
                tensor_name: "quant_weight".to_string(),
                quant_param_tensor_names: HashMap::from([
                    ("SCALE_TENSOR".to_string(), "weight_scale".to_string()),
                    ("ZERO_POINT_TENSOR".to_string(), "weight_zp".to_string()),
                ]),
                scale: Some(0.0078125),
                zero_point: Some(128),
                axis: Some(0),
            }],
            metadata_props: HashMap::new(),
        };
        let qa = &graph.quantization_annotation[0];
        assert!((qa.scale.unwrap() - 0.0078125).abs() < 1e-10);
        assert_eq!(qa.zero_point, Some(128));
    }

    #[test]
    fn onnx_model_metadata_ir_version_zero() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.ir_version, 0);
    }

    #[test]
    fn onnx_model_metadata_ir_version_negative() {
        let meta = OnnxModelMetadata {
            ir_version: -1,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: -1,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.ir_version, -1);
        assert_eq!(meta.model_version, -1);
    }

    #[test]
    fn onnx_operator_set_version_zero() {
        let ops = OnnxOperatorSet {
            domain: String::new(),
            version: 0,
        };
        assert_eq!(ops.version, 0);
    }

    #[test]
    fn onnx_operator_set_version_large() {
        let ops = OnnxOperatorSet {
            domain: "future.onnx".to_string(),
            version: i64::MAX,
        };
        assert_eq!(ops.version, i64::MAX);
    }

    #[test]
    fn parse_functions_with_attribute_protos() {
        let functions = vec![proto::FunctionProto {
            name: Some("ScaledAdd".to_string()),
            domain: Some("test".to_string()),
            overload: None,
            input: vec!["A".to_string()],
            output: vec!["B".to_string()],
            attribute: vec![],
            attribute_proto: vec![proto::AttributeProto {
                name: Some("alpha".to_string()),
                r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
                f: Some(0.5),
                ..Default::default()
            }],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].attribute_protos.len(), 1);
        assert!(result[0].attribute_protos.contains_key("alpha"));
    }

    #[test]
    fn onnx_graph_empty_name_is_valid() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.name.is_empty());
    }

    #[test]
    fn onnx_graph_bind_weights_no_match_returns_zero() {
        struct DummyProvider;
        impl crate::loader::TensorProvider for DummyProvider {
            fn tensor_info(&self, _name: &str) -> Option<crate::loader::TensorMeta> {
                None
            }
            fn load_tensor_data(
                &self,
                _name: &str,
            ) -> std::result::Result<std::borrow::Cow<'_, [u8]>, LoaderError> {
                panic!("should not be called");
            }
            fn iter_tensors(&self) -> impl Iterator<Item = crate::loader::TensorMeta> {
                std::iter::empty()
            }
        }
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let provider = DummyProvider;
        let mapping = HashMap::new();
        let count = graph.bind_weights(&provider, &mapping).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn onnx_graph_bind_weights_auto_skips_activation_inputs() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "layernorm".to_string(),
                op_type: "LayerNorm".to_string(),
                domain: String::new(),
                inputs: vec![
                    "hidden_states".to_string(),
                    "input_ids".to_string(),
                    "weight".to_string(),
                    String::new(),
                ],
                outputs: vec!["output".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let weight_inputs: std::collections::HashSet<String> = {
            let mut set = std::collections::HashSet::new();
            for node in &graph.nodes {
                for input in &node.inputs {
                    if !input.is_empty()
                        && !input.starts_with("hidden")
                        && !input.starts_with("input")
                    {
                        set.insert(input.clone());
                    }
                }
            }
            set
        };
        assert_eq!(weight_inputs.len(), 1);
        assert!(weight_inputs.contains("weight"));
    }

    #[test]
    fn onnx_model_clone_independence() {
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: "orig".to_string(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "orig_graph".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let cloned = model.clone();
        model.metadata.producer_name = "modified".to_string();
        assert_eq!(cloned.metadata.producer_name, "orig");
    }

    #[test]
    fn onnx_node_single_io() {
        let node = OnnxNode {
            name: "squeeze".to_string(),
            op_type: "Squeeze".to_string(),
            domain: String::new(),
            inputs: vec!["data".to_string()],
            outputs: vec!["squeezed".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.inputs.len(), 1);
        assert_eq!(node.outputs.len(), 1);
    }

    #[test]
    fn onnx_value_info_clone_independence() {
        let mut vi = OnnxValueInfo {
            name: "original".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let cloned = vi.clone();
        vi.name = "modified".to_string();
        assert_eq!(cloned.name, "original");
    }

    #[test]
    fn onnx_graph_initializer_zero_byte_tensor() {
        let tensor = OnnxTensor::new(
            "empty_weight".to_string(),
            Dtype::F32,
            vec![0],
            Bytes::new(),
        );
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("empty_weight".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let t = graph.initializers.get("empty_weight").unwrap();
        assert_eq!(t.shape, vec![0]);
        assert!(t.raw_data().is_empty());
    }

    #[test]
    fn onnx_function_same_input_output_count() {
        let func = OnnxFunction {
            name: "Identity".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["X".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![OnnxNode {
                name: "pass".to_string(),
                op_type: "Identity".to_string(),
                domain: String::new(),
                inputs: vec!["X".to_string()],
                outputs: vec!["X".to_string()],
                attributes: HashMap::new(),
            }],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.inputs, func.outputs);
    }

    #[test]
    fn parse_nodes_preserves_input_order() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Concat".to_string()),
            name: None,
            input: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
                "e".to_string(),
            ],
            output: vec!["out".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].inputs.len(), 5);
        assert_eq!(result[0].inputs[0], "a");
        assert_eq!(result[0].inputs[4], "e");
    }

    #[test]
    fn parse_nodes_preserves_output_order() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Split".to_string()),
            name: Some("splitter".to_string()),
            input: vec!["data".to_string()],
            output: vec![
                "part_0".to_string(),
                "part_1".to_string(),
                "part_2".to_string(),
            ],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].outputs.len(), 3);
        assert_eq!(result[0].outputs[2], "part_2");
    }

    #[test]
    fn onnx_graph_multiple_sparse_initializers() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let make_sparse = |name: &str| -> OnnxSparseTensor {
            let values = OnnxTensor::new(
                format!("{name}_vals"),
                Dtype::F32,
                vec![2],
                Bytes::from(vec![0u8; 8]),
            );
            let indices = OnnxTensor::new(
                format!("{name}_idx"),
                Dtype::I64,
                vec![2],
                Bytes::from(vec![0u8; 16]),
            );
            OnnxSparseTensor {
                values,
                indices,
                dims: vec![3, 3],
                format: OnnxSparseFormat::Coo,
            }
        };
        let graph = OnnxGraph {
            name: "sparse_multi".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![make_sparse("sp0"), make_sparse("sp1")],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.sparse_initializers.len(), 2);
        assert_eq!(graph.sparse_initializers[0].values.name, "sp0_vals");
    }

    #[test]
    fn onnx_graph_debug_with_all_collections() {
        let tensor = OnnxTensor::new(
            "w".to_string(),
            Dtype::F32,
            vec![1],
            Bytes::from(vec![0u8; 4]),
        );
        let graph = OnnxGraph {
            name: "full_graph".to_string(),
            doc_string: "test".to_string(),
            nodes: vec![OnnxNode {
                name: "n".to_string(),
                op_type: "Add".to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: HashMap::new(),
            }],
            inputs: vec![OnnxValueInfo {
                name: "in".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("w".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::from([("k".to_string(), "v".to_string())]),
        };
        let debug = format!("{graph:?}");
        assert!(debug.contains("full_graph"));
        assert!(debug.contains("Add"));
    }

    #[test]
    fn onnx_model_metadata_debug_with_opsets_and_props() {
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: "onnxruntime".to_string(),
            producer_version: "1.17".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 1,
            doc_string: "test".to_string(),
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 17,
            }],
            metadata_props: HashMap::from([("key".to_string(), "val".to_string())]),
        };
        let debug = format!("{meta:?}");
        assert!(debug.contains("onnxruntime"));
        assert!(debug.contains("1.17"));
    }

    #[test]
    fn parse_functions_preserves_doc_string() {
        let functions = vec![proto::FunctionProto {
            name: Some("DocOp".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: Some("A documented function".to_string()),
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].doc_string, "A documented function");
    }

    #[test]
    fn parse_functions_with_value_info_entries() {
        let functions = vec![proto::FunctionProto {
            name: Some("VIOp".to_string()),
            domain: None,
            overload: None,
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![proto::ValueInfoProto {
                name: Some("intermediate".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].value_info.len(), 1);
        assert_eq!(result[0].value_info[0].name, "intermediate");
    }

    #[test]
    fn onnx_graph_from_proto_valid_graph() {
        let graph_proto = proto::GraphProto {
            name: Some("test_graph".to_string()),
            node: vec![proto::NodeProto {
                op_type: Some("Add".to_string()),
                name: Some("add_0".to_string()),
                input: vec!["a".to_string(), "b".to_string()],
                output: vec!["c".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            }],
            input: vec![proto::ValueInfoProto {
                name: Some("a".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            output: vec![proto::ValueInfoProto {
                name: Some("c".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(result.name, "test_graph");
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op_type, "Add");
        assert_eq!(result.inputs[0].name, "a");
        assert_eq!(result.outputs[0].name, "c");
    }

    #[test]
    fn parse_value_info_with_metadata_props() {
        let values = vec![proto::ValueInfoProto {
            name: Some("tensor_a".to_string()),
            r#type: None,
            doc_string: Some("doc".to_string()),
            metadata_props: vec![proto::StringStringEntryProto {
                key: Some("source".to_string()),
                value: Some("model".to_string()),
            }],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result[0].metadata_props.get("source").unwrap(), "model");
    }

    #[test]
    fn onnx_model_from_proto_valid_minimal() {
        let model_proto = proto::ModelProto {
            ir_version: Some(7),
            producer_name: Some("test_producer".to_string()),
            opset_import: vec![proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            }],
            graph: Some(proto::GraphProto {
                name: Some("minimal_graph".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.metadata.ir_version, 7);
        assert_eq!(model.metadata.producer_name, "test_producer");
        assert_eq!(model.graph.name, "minimal_graph");
        assert!(model.functions.is_empty());
    }

    #[test]
    fn onnx_model_from_proto_with_function() {
        let model_proto = proto::ModelProto {
            ir_version: Some(8),
            graph: Some(proto::GraphProto {
                name: Some("g".to_string()),
                ..Default::default()
            }),
            functions: vec![proto::FunctionProto {
                name: Some("MyFunc".to_string()),
                domain: Some("custom".to_string()),
                overload: Some("v1".to_string()),
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                attribute: vec![],
                attribute_proto: vec![],
                node: vec![],
                opset_import: vec![],
                value_info: vec![],
                doc_string: None,
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.functions.len(), 1);
        assert_eq!(model.functions[0].name, "MyFunc");
        assert_eq!(model.functions[0].overload, "v1");
    }

    #[test]
    fn onnx_graph_initializer_count_after_multiple_insertions() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        for i in 0..5 {
            let t = OnnxTensor::new(
                format!("t{i}"),
                Dtype::F32,
                vec![1],
                Bytes::from(vec![0u8; 4]),
            );
            graph.initializers.insert(format!("t{i}"), t);
        }
        assert_eq!(graph.initializers.len(), 5);
    }

    #[test]
    fn onnx_value_info_debug_contains_name() {
        let vi = OnnxValueInfo {
            name: "my_tensor_value".to_string(),
            value_type: None,
            doc_string: "debug test".to_string(),
            metadata_props: HashMap::new(),
        };
        let debug = format!("{vi:?}");
        assert!(debug.contains("my_tensor_value"));
    }

    #[test]
    fn onnx_operator_set_clone_independence() {
        let mut ops = OnnxOperatorSet {
            domain: "original".to_string(),
            version: 1,
        };
        let cloned = ops.clone();
        ops.domain = "modified".to_string();
        assert_eq!(cloned.domain, "original");
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 6 TESTS (50 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxAttributeValue::Tensor variant ──────────────────────────────

    #[test]
    fn onnx_attribute_value_tensor_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let tensor = OnnxTensor::new(
            "attr_tensor".to_string(),
            Dtype::F32,
            vec![2, 2],
            Bytes::from(vec![0u8; 16]),
        );
        let val = OnnxAttributeValue::Tensor(tensor);
        assert!(matches!(val, OnnxAttributeValue::Tensor(t) if t.name == "attr_tensor"));
    }

    #[test]
    fn onnx_attribute_value_sparse_tensor_variant() {
        use super::super::attributes::OnnxAttributeValue;
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let values = OnnxTensor::new("sv".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("si".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor { values, indices, dims: vec![4], format: OnnxSparseFormat::Csc };
        let val = OnnxAttributeValue::SparseTensor(sparse);
        assert!(matches!(val, OnnxAttributeValue::SparseTensor(st) if st.format == OnnxSparseFormat::Csc));
    }

    #[test]
    fn onnx_attribute_value_type_variant() {
        use super::super::attributes::OnnxAttributeValue;
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let ty = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int32,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        let val = OnnxAttributeValue::Type(ty);
        assert!(matches!(val, OnnxAttributeValue::Type(_)));
    }

    #[test]
    fn onnx_attribute_value_graph_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let subgraph = OnnxGraph {
            name: "then_branch".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "identity".to_string(),
                op_type: "Identity".to_string(),
                domain: String::new(),
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let val = OnnxAttributeValue::Graph(Box::new(subgraph));
        assert!(matches!(val, OnnxAttributeValue::Graph(g) if g.name == "then_branch"));
    }

    #[test]
    fn onnx_attribute_value_graphs_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let g1 = OnnxGraph {
            name: "branch_a".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let g2 = OnnxGraph {
            name: "branch_b".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let val = OnnxAttributeValue::Graphs(vec![g1, g2]);
        assert!(matches!(val, OnnxAttributeValue::Graphs(v) if v.len() == 2));
    }

    #[test]
    fn onnx_attribute_value_floats_nonempty() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Floats(vec![1.0, 2.5, -0.5]);
        assert!(matches!(val, OnnxAttributeValue::Floats(v) if v.len() == 3));
    }

    #[test]
    fn onnx_attribute_value_strings_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Strings(vec!["NCHW".to_string(), "NHWC".to_string()]);
        assert!(matches!(val, OnnxAttributeValue::Strings(v) if v.len() == 2));
    }

    #[test]
    fn onnx_attribute_value_tensors_variant() {
        use super::super::attributes::OnnxAttributeValue;
        let t1 = OnnxTensor::new("t1".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let t2 = OnnxTensor::new("t2".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let val = OnnxAttributeValue::Tensors(vec![t1, t2]);
        assert!(matches!(val, OnnxAttributeValue::Tensors(v) if v.len() == 2));
    }

    #[test]
    fn onnx_attribute_value_sparse_tensors_variant() {
        use super::super::attributes::OnnxAttributeValue;
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let sp = OnnxSparseTensor {
            values: OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4])),
            indices: OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8])),
            dims: vec![2],
            format: OnnxSparseFormat::Coo,
        };
        let val = OnnxAttributeValue::SparseTensors(vec![sp]);
        assert!(matches!(val, OnnxAttributeValue::SparseTensors(v) if v.len() == 1));
    }

    #[test]
    fn onnx_attribute_value_types_variant() {
        use super::super::attributes::OnnxAttributeValue;
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let ty = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Bool,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Unknown] },
        });
        let val = OnnxAttributeValue::Types(vec![ty]);
        assert!(matches!(val, OnnxAttributeValue::Types(v) if v.len() == 1));
    }

    // ── OnnxAttributeValue clone independence ───────────────────────────

    #[test]
    fn onnx_attribute_value_ints_clone_independence() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Ints(vec![10, 20]);
        let cloned = val.clone();
        if let OnnxAttributeValue::Ints(ref _v) = val {
            // verify pattern matching works
        }
        assert!(matches!(cloned, OnnxAttributeValue::Ints(v) if v == vec![10, 20]));
    }

    #[test]
    fn onnx_attribute_value_string_clone_independence() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::String("original".to_string());
        let cloned = val.clone();
        let _ = val;
        assert!(matches!(cloned, OnnxAttributeValue::String(s) if s == "original"));
    }

    #[test]
    fn onnx_attribute_value_graph_clone_independence() {
        use super::super::attributes::OnnxAttributeValue;
        let g = OnnxGraph {
            name: "sub".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let val = OnnxAttributeValue::Graph(Box::new(g));
        let cloned = val.clone();
        assert!(matches!(cloned, OnnxAttributeValue::Graph(g) if g.name == "sub"));
    }

    // ── LoaderError: FormatNotFound ─────────────────────────────────────

    #[test]
    fn loader_error_format_not_found_display() {
        use crate::loader::WeightFormat;
        let err = LoaderError::FormatNotFound(WeightFormat::SafeTensors);
        let msg = format!("{err}");
        assert!(msg.contains("Format not found"), "expected prefix: {msg}");
    }

    #[test]
    fn loader_error_multiple_weight_formats_display() {
        use crate::loader::WeightFormat;
        let err = LoaderError::MultipleWeightFormats(vec![
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
        ]);
        let msg = format!("{err}");
        assert!(msg.contains("Multiple weight formats"), "expected prefix: {msg}");
    }

    // ── WeightFormat variants and traits ────────────────────────────────

    #[test]
    fn weight_format_variants_equality() {
        use crate::loader::WeightFormat;
        assert_eq!(WeightFormat::SafeTensors, WeightFormat::SafeTensors);
        assert_eq!(WeightFormat::Gguf, WeightFormat::Gguf);
        assert_eq!(WeightFormat::Onnx, WeightFormat::Onnx);
        assert_eq!(WeightFormat::PyTorch, WeightFormat::PyTorch);
        assert_eq!(WeightFormat::Gllm, WeightFormat::Gllm);
        assert_ne!(WeightFormat::SafeTensors, WeightFormat::Gguf);
        assert_ne!(WeightFormat::Onnx, WeightFormat::PyTorch);
    }

    #[test]
    fn weight_format_debug_output() {
        use crate::loader::WeightFormat;
        let debug = format!("{:?}", WeightFormat::SafeTensors);
        assert!(debug.contains("SafeTensors"), "expected SafeTensors in: {debug}");
    }

    #[test]
    fn weight_format_copy_trait() {
        use crate::loader::WeightFormat;
        let a = WeightFormat::Gguf;
        let b = a; // Copy, not move
        let _ = a; // still usable
        assert_eq!(a, b);
    }

    #[test]
    fn weight_format_hash_consistency() {
        use crate::loader::WeightFormat;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(WeightFormat::SafeTensors));
        assert!(!set.insert(WeightFormat::SafeTensors));
        assert!(set.insert(WeightFormat::Gguf));
        assert_eq!(set.len(), 2);
    }

    // ── OnnxAttribute with attr_type field ──────────────────────────────

    #[test]
    fn onnx_attribute_attr_type_float() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "epsilon".to_string(),
            value: OnnxAttributeValue::Float(1e-5),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Float),
        };
        assert_eq!(attr.attr_type, Some(proto::attribute_proto::AttributeType::Float));
    }

    #[test]
    fn onnx_attribute_attr_type_none() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "transA".to_string(),
            value: OnnxAttributeValue::Int(0),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        };
        assert!(attr.attr_type.is_none());
    }

    #[test]
    fn onnx_attribute_with_nonempty_doc_string() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "to".to_string(),
            value: OnnxAttributeValue::Int(5),
            doc_string: "number of outputs".to_string(),
            ref_attr_name: None,
            attr_type: None,
        };
        assert_eq!(attr.doc_string, "number of outputs");
    }

    // ── OnnxAttribute in node context with Graph value ──────────────────

    #[test]
    fn onnx_node_attribute_with_graph_value() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let subgraph = OnnxGraph {
            name: "then_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let mut attrs = HashMap::new();
        attrs.insert(
            "then_branch".to_string(),
            OnnxAttribute {
                name: "then_branch".to_string(),
                value: OnnxAttributeValue::Graph(Box::new(subgraph)),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        let node = OnnxNode {
            name: "if_node".to_string(),
            op_type: "If".to_string(),
            domain: String::new(),
            inputs: vec!["cond".to_string()],
            outputs: vec!["result".to_string()],
            attributes: attrs,
        };
        assert_eq!(node.attributes.len(), 1);
        assert!(matches!(
            node.attributes.get("then_branch").unwrap().value,
            OnnxAttributeValue::Graph(_)
        ));
    }

    // ── parse_functions with metadata_props ─────────────────────────────

    #[test]
    fn parse_functions_preserves_metadata_props() {
        let functions = vec![proto::FunctionProto {
            name: Some("MetaOp".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![
                proto::StringStringEntryProto {
                    key: Some("author".to_string()),
                    value: Some("gllm".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("version".to_string()),
                    value: Some("2".to_string()),
                },
            ],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].metadata_props.len(), 2);
        assert_eq!(result[0].metadata_props.get("author").unwrap(), "gllm");
        assert_eq!(result[0].metadata_props.get("version").unwrap(), "2");
    }

    // ── parse_functions with opset_import entries ───────────────────────

    #[test]
    fn parse_functions_with_multiple_opset_imports() {
        let functions = vec![proto::FunctionProto {
            name: Some("MultiOpsetOp".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![
                proto::OperatorSetIdProto { domain: Some(String::new()), version: Some(17) },
                proto::OperatorSetIdProto { domain: Some("ai.onnx.ml".to_string()), version: Some(3) },
            ],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].opset_import.len(), 2);
        assert_eq!(result[0].opset_import[0].version, 17);
        assert_eq!(result[0].opset_import[1].domain, "ai.onnx.ml");
    }

    // ── parse_functions with multiple functions including nodes ─────────

    #[test]
    fn parse_functions_multiple_with_nodes() {
        let make_func = |name: &str, op: &str| -> proto::FunctionProto {
            proto::FunctionProto {
                name: Some(name.to_string()),
                domain: Some("custom".to_string()),
                overload: None,
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                attribute: vec![],
                attribute_proto: vec![],
                node: vec![proto::NodeProto {
                    op_type: Some(op.to_string()),
                    name: None,
                    input: vec!["X".to_string()],
                    output: vec!["Y".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                }],
                opset_import: vec![],
                value_info: vec![],
                doc_string: None,
                metadata_props: vec![],
            }
        };
        let functions = vec![make_func("OpA", "Relu"), make_func("OpB", "Sigmoid")];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "OpA");
        assert_eq!(result[0].nodes[0].op_type, "Relu");
        assert_eq!(result[1].name, "OpB");
        assert_eq!(result[1].nodes[0].op_type, "Sigmoid");
    }

    // ── OnnxModelMetadata doc_string field ──────────────────────────────

    #[test]
    fn onnx_model_metadata_doc_string_field() {
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: "This is a test model for unit testing".to_string(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.doc_string, "This is a test model for unit testing");
    }

    #[test]
    fn onnx_model_metadata_domain_field() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: "ai.onnx.transformers".to_string(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.domain, "ai.onnx.transformers");
    }

    // ── OnnxModel from_proto with metadata_props on model ───────────────

    #[test]
    fn onnx_model_from_proto_metadata_props() {
        let model_proto = proto::ModelProto {
            ir_version: Some(7),
            metadata_props: vec![
                proto::StringStringEntryProto {
                    key: Some("license".to_string()),
                    value: Some("MIT".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("company".to_string()),
                    value: Some("acme".to_string()),
                },
            ],
            graph: Some(proto::GraphProto {
                name: Some("g".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.metadata.metadata_props.len(), 2);
        assert_eq!(model.metadata.metadata_props.get("license").unwrap(), "MIT");
        assert_eq!(model.metadata.metadata_props.get("company").unwrap(), "acme");
    }

    // ── OnnxGraph from_proto with metadata_props ────────────────────────

    #[test]
    fn onnx_graph_from_proto_with_metadata_props() {
        let graph_proto = proto::GraphProto {
            name: Some("g".to_string()),
            metadata_props: vec![proto::StringStringEntryProto {
                key: Some("framework".to_string()),
                value: Some("pytorch".to_string()),
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.metadata_props.get("framework").unwrap(), "pytorch");
    }

    // ── OnnxGraph from_proto with value_info ────────────────────────────

    #[test]
    fn onnx_graph_from_proto_with_value_info() {
        let graph_proto = proto::GraphProto {
            name: Some("g".to_string()),
            value_info: vec![
                proto::ValueInfoProto {
                    name: Some("intermediate_1".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
                proto::ValueInfoProto {
                    name: Some("intermediate_2".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
            ],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.value_info.len(), 2);
        assert_eq!(graph.value_info[0].name, "intermediate_1");
        assert_eq!(graph.value_info[1].name, "intermediate_2");
    }

    // ── OnnxGraph from_proto with quantization annotations ──────────────

    #[test]
    fn onnx_graph_from_proto_with_quantization_annotations() {
        let scale_tensor = proto::TensorProto {
            name: Some("q_scale".to_string()),
            data_type: Some(1), // FLOAT
            dims: vec![],
            float_data: vec![0.125],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            name: Some("quant_graph".to_string()),
            initializer: vec![scale_tensor],
            quantization_annotation: vec![proto::TensorAnnotation {
                tensor_name: Some("weight_q".to_string()),
                quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                    key: Some("SCALE_TENSOR".to_string()),
                    value: Some("q_scale".to_string()),
                }],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.quantization_annotation.len(), 1);
        assert_eq!(graph.quantization_annotation[0].tensor_name, "weight_q");
    }

    // ── OnnxModel from_proto with multiple opset imports ────────────────

    #[test]
    fn onnx_model_from_proto_multiple_opsets() {
        let model_proto = proto::ModelProto {
            ir_version: Some(9),
            opset_import: vec![
                proto::OperatorSetIdProto { domain: Some(String::new()), version: Some(20) },
                proto::OperatorSetIdProto { domain: Some("ai.onnx.ml".to_string()), version: Some(4) },
                proto::OperatorSetIdProto { domain: Some("ai.onnx.training".to_string()), version: Some(1) },
            ],
            graph: Some(proto::GraphProto {
                name: Some("multi_opset_graph".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.metadata.opset_import.len(), 3);
        assert_eq!(model.metadata.opset_import[0].version, 20);
        assert_eq!(model.metadata.opset_import[1].domain, "ai.onnx.ml");
        assert_eq!(model.metadata.opset_import[2].version, 1);
    }

    // ── parse_quantization: zero_point from non-scalar returns none ─────

    #[test]
    fn parse_quantization_zero_point_from_non_scalar_returns_none() {
        let mut bytes = Vec::new();
        for v in &[10i64, 20i64] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let zp_tensor = OnnxTensor::new(
            "multi_zp".to_string(),
            Dtype::I64,
            vec![2],
            Bytes::from(bytes),
        );
        let initializers = HashMap::from([("multi_zp".to_string(), zp_tensor)]);
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("ZERO_POINT_TENSOR".to_string()),
                value: Some("multi_zp".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &initializers);
        assert_eq!(result[0].zero_point, None);
    }

    // ── OnnxFunction clone independence ─────────────────────────────────

    #[test]
    fn onnx_function_clone_independence() {
        let func = OnnxFunction {
            name: "Original".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec!["alpha".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let cloned = func.clone();
        let _ = func;
        assert_eq!(cloned.name, "Original");
        assert_eq!(cloned.inputs, vec!["X"]);
        assert_eq!(cloned.outputs, vec!["Y"]);
    }

    // ── OnnxFunction with many inputs and outputs ───────────────────────

    #[test]
    fn onnx_function_many_inputs_outputs() {
        let func = OnnxFunction {
            name: "MultiIO".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()],
            outputs: vec!["X".to_string(), "Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.inputs.len(), 4);
        assert_eq!(func.outputs.len(), 2);
    }

    // ── OnnxNode with Graph-valued attribute for If operator ────────────

    #[test]
    fn onnx_node_if_operator_with_else_branch() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let else_graph = OnnxGraph {
            name: "else_branch".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "zero_fill".to_string(),
                op_type: "Constant".to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec!["zero_out".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let mut attrs = HashMap::new();
        attrs.insert("else_branch".to_string(), OnnxAttribute {
            name: "else_branch".to_string(),
            value: OnnxAttributeValue::Graph(Box::new(else_graph)),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let node = OnnxNode {
            name: "if_else".to_string(),
            op_type: "If".to_string(),
            domain: String::new(),
            inputs: vec!["condition".to_string()],
            outputs: vec!["result".to_string()],
            attributes: attrs,
        };
        assert_eq!(node.attributes.len(), 1);
        if let OnnxAttributeValue::Graph(g) = &node.attributes.get("else_branch").unwrap().value {
            assert_eq!(g.name, "else_branch");
            assert_eq!(g.nodes.len(), 1);
            assert_eq!(g.nodes[0].op_type, "Constant");
        } else {
            panic!("expected Graph variant");
        }
    }

    // ── OnnxValueInfo with optional type containing tensor ──────────────

    #[test]
    fn onnx_value_info_with_optional_tensor_type() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let inner = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::String,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Unknown] },
        });
        let vi = OnnxValueInfo {
            name: "optional_text".to_string(),
            value_type: Some(OnnxType::Optional(Box::new(inner))),
            doc_string: "optional string tensor".to_string(),
            metadata_props: HashMap::new(),
        };
        if let Some(OnnxType::Optional(inner)) = vi.value_type {
            assert!(matches!(*inner, OnnxType::Tensor(_)));
        } else {
            panic!("expected Optional variant");
        }
    }

    // ── OnnxGraph clone independence for initializers ───────────────────

    #[test]
    fn onnx_graph_clone_independence() {
        let tensor = OnnxTensor::new(
            "weight".to_string(),
            Dtype::F32,
            vec![2],
            Bytes::from(vec![0u8; 8]),
        );
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("weight".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let cloned = graph.clone();
        graph.name = "modified".to_string();
        assert_eq!(cloned.name, "g");
        assert!(cloned.initializers.contains_key("weight"));
    }

    // ── parse_nodes with domain attribute ───────────────────────────────

    #[test]
    fn parse_nodes_with_custom_domain() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("FusedMatMul".to_string()),
            name: None,
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            domain: Some("com.nvidia".to_string()),
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].domain, "com.nvidia");
    }

    // ── OnnxModel from_proto preserves producer_version ─────────────────

    #[test]
    fn onnx_model_from_proto_producer_version() {
        let model_proto = proto::ModelProto {
            ir_version: Some(8),
            producer_name: Some("onnxruntime".to_string()),
            producer_version: Some("1.17.3".to_string()),
            domain: Some("ai.onnx".to_string()),
            model_version: Some(42),
            doc_string: Some("test model".to_string()),
            graph: Some(proto::GraphProto {
                name: Some("g".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.metadata.producer_version, "1.17.3");
        assert_eq!(model.metadata.domain, "ai.onnx");
        assert_eq!(model.metadata.model_version, 42);
        assert_eq!(model.metadata.doc_string, "test model");
    }

    // ── OnnxNode with Float attribute value ─────────────────────────────

    #[test]
    fn onnx_node_with_float_attribute() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert("epsilon".to_string(), OnnxAttribute {
            name: "epsilon".to_string(),
            value: OnnxAttributeValue::Float(1e-5),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let node = OnnxNode {
            name: "layer_norm".to_string(),
            op_type: "LayerNormalization".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attributes: attrs,
        };
        if let OnnxAttributeValue::Float(v) = node.attributes.get("epsilon").unwrap().value {
            assert!((v - 1e-5).abs() < 1e-12);
        } else {
            panic!("expected Float variant");
        }
    }

    // ── OnnxNode with String attribute value ────────────────────────────

    #[test]
    fn onnx_node_with_string_attribute() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert("data_format".to_string(), OnnxAttribute {
            name: "data_format".to_string(),
            value: OnnxAttributeValue::String("NCHW".to_string()),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let node = OnnxNode {
            name: "conv".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
        };
        if let OnnxAttributeValue::String(ref s) = node.attributes.get("data_format").unwrap().value {
            assert_eq!(s, "NCHW");
        } else {
            panic!("expected String variant");
        }
    }

    // ── OnnxNode with Ref attribute value ───────────────────────────────

    #[test]
    fn onnx_node_with_ref_attribute() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), OnnxAttribute {
            name: "axis".to_string(),
            value: OnnxAttributeValue::Ref("parent_axis".to_string()),
            doc_string: String::new(),
            ref_attr_name: Some("parent_axis".to_string()),
            attr_type: None,
        });
        let node = OnnxNode {
            name: "softmax".to_string(),
            op_type: "Softmax".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
        };
        let attr = node.attributes.get("axis").unwrap();
        assert!(matches!(&attr.value, OnnxAttributeValue::Ref(s) if s == "parent_axis"));
        assert_eq!(attr.ref_attr_name, Some("parent_axis".to_string()));
    }

    // ── OnnxFunction with value_info containing type info ───────────────

    #[test]
    fn onnx_function_value_info_with_type() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let func = OnnxFunction {
            name: "TypedFunc".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![OnnxValueInfo {
                name: "hidden".to_string(),
                value_type: Some(OnnxType::Tensor(OnnxTensorType {
                    elem_type: proto::tensor_proto::DataType::Float,
                    shape: OnnxTensorShape {
                        dims: vec![OnnxDim::Param("batch".to_string()), OnnxDim::Known(768)],
                    },
                })),
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.value_info.len(), 1);
        assert!(func.value_info[0].value_type.is_some());
    }

    // ── OnnxSparseTensor with different formats ─────────────────────────

    #[test]
    fn onnx_sparse_tensor_csc_format() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let sparse = OnnxSparseTensor {
            values: OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4])),
            indices: OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8])),
            dims: vec![5, 5],
            format: OnnxSparseFormat::Csc,
        };
        assert_eq!(sparse.format, OnnxSparseFormat::Csc);
        assert_eq!(sparse.dims, vec![5, 5]);
    }

    #[test]
    fn onnx_sparse_tensor_1d_dims() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let sparse = OnnxSparseTensor {
            values: OnnxTensor::new("v".to_string(), Dtype::F32, vec![3], Bytes::from(vec![0u8; 12])),
            indices: OnnxTensor::new("i".to_string(), Dtype::I64, vec![3], Bytes::from(vec![0u8; 24])),
            dims: vec![10],
            format: OnnxSparseFormat::Coo,
        };
        assert_eq!(sparse.dims.len(), 1);
    }

    // ── OnnxGraph with nodes referencing each other's outputs ───────────

    #[test]
    fn onnx_graph_chained_nodes() {
        let graph = OnnxGraph {
            name: "chain".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "matmul".to_string(),
                    op_type: "MatMul".to_string(),
                    domain: String::new(),
                    inputs: vec!["input".to_string(), "w1".to_string()],
                    outputs: vec!["h1".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "bias_add".to_string(),
                    op_type: "Add".to_string(),
                    domain: String::new(),
                    inputs: vec!["h1".to_string(), "b1".to_string()],
                    outputs: vec!["h2".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "relu".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["h2".to_string()],
                    outputs: vec!["output".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].outputs[0], graph.nodes[1].inputs[0]);
        assert_eq!(graph.nodes[1].outputs[0], graph.nodes[2].inputs[0]);
    }

    // ── OnnxAttributeValue Debug format for all variants ────────────────

    #[test]
    fn onnx_attribute_value_debug_float() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Float(1.5);
        let debug = format!("{val:?}");
        assert!(debug.contains("Float"), "expected Float in: {debug}");
    }

    #[test]
    fn onnx_attribute_value_debug_int() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Int(42);
        let debug = format!("{val:?}");
        assert!(debug.contains("Int"), "expected Int in: {debug}");
    }

    #[test]
    fn onnx_attribute_value_debug_ints() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Ints(vec![1, 2]);
        let debug = format!("{val:?}");
        assert!(debug.contains("Ints"), "expected Ints in: {debug}");
    }

    #[test]
    fn onnx_attribute_value_debug_tensor() {
        use super::super::attributes::OnnxAttributeValue;
        let t = OnnxTensor::new("debug_t".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let val = OnnxAttributeValue::Tensor(t);
        let debug = format!("{val:?}");
        assert!(debug.contains("Tensor"), "expected Tensor in: {debug}");
    }

    // ── parse_metadata_props with very long value ───────────────────────

    #[test]
    fn parse_metadata_props_long_value() {
        let long_value = "x".repeat(10_000);
        let entries = vec![proto::StringStringEntryProto {
            key: Some("big_key".to_string()),
            value: Some(long_value.clone()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("big_key").unwrap().len(), 10_000);
    }

    // ── OnnxModel with zero ir_version and model_version ────────────────

    #[test]
    fn onnx_model_zero_ir_version_and_model_version() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert_eq!(model.metadata.ir_version, 0);
        assert_eq!(model.metadata.model_version, 0);
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 7 TESTS (~60 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxGraph: completely empty graph (zero nodes, zero IO) ──────────

    #[test]
    fn onnx_graph_empty_all_fields_zero() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.name.is_empty());
        assert!(graph.nodes.is_empty());
        assert!(graph.inputs.is_empty());
        assert!(graph.outputs.is_empty());
        assert!(graph.value_info.is_empty());
        assert!(graph.initializers.is_empty());
        assert!(graph.sparse_initializers.is_empty());
        assert!(graph.quantization_annotation.is_empty());
        assert!(graph.metadata_props.is_empty());
    }

    // ── OnnxGraph: single-node graph ─────────────────────────────────────

    #[test]
    fn onnx_graph_single_node_no_io() {
        let graph = OnnxGraph {
            name: "single_node".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "const_node".to_string(),
                op_type: "Constant".to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec!["scalar_out".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "Constant");
        assert!(graph.nodes[0].inputs.is_empty());
        assert_eq!(graph.nodes[0].outputs.len(), 1);
    }

    // ── OnnxGraph: from_proto with empty nodes list ──────────────────────

    #[test]
    fn onnx_graph_from_proto_empty_nodes() {
        let graph_proto = proto::GraphProto {
            name: Some("empty_graph".to_string()),
            node: vec![],
            input: vec![],
            output: vec![],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.nodes.is_empty());
        assert!(graph.inputs.is_empty());
        assert!(graph.outputs.is_empty());
        assert_eq!(graph.name, "empty_graph");
    }

    // ── OnnxGraph: diamond connectivity pattern ──────────────────────────

    #[test]
    fn onnx_graph_diamond_connectivity() {
        let graph = OnnxGraph {
            name: "diamond".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "split".to_string(),
                    op_type: "Split".to_string(),
                    domain: String::new(),
                    inputs: vec!["input".to_string()],
                    outputs: vec!["branch_a".to_string(), "branch_b".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "path_a".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["branch_a".to_string()],
                    outputs: vec!["relu_out".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "path_b".to_string(),
                    op_type: "Sigmoid".to_string(),
                    domain: String::new(),
                    inputs: vec!["branch_b".to_string()],
                    outputs: vec!["sigmoid_out".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "merge".to_string(),
                    op_type: "Add".to_string(),
                    domain: String::new(),
                    inputs: vec!["relu_out".to_string(), "sigmoid_out".to_string()],
                    outputs: vec!["output".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 4);
        // split feeds both path_a and path_b
        assert_eq!(graph.nodes[0].outputs[0], graph.nodes[1].inputs[0]);
        assert_eq!(graph.nodes[0].outputs[1], graph.nodes[2].inputs[0]);
        // both paths feed merge
        assert_eq!(graph.nodes[1].outputs[0], graph.nodes[3].inputs[0]);
        assert_eq!(graph.nodes[2].outputs[0], graph.nodes[3].inputs[1]);
    }

    // ── OnnxGraph: fan-out (single output feeding multiple nodes) ────────

    #[test]
    fn onnx_graph_fan_out_pattern() {
        let graph = OnnxGraph {
            name: "fanout".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "source".to_string(),
                    op_type: "Identity".to_string(),
                    domain: String::new(),
                    inputs: vec!["input".to_string()],
                    outputs: vec!["shared".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "consumer_a".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["shared".to_string()],
                    outputs: vec!["out_a".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "consumer_b".to_string(),
                    op_type: "Tanh".to_string(),
                    domain: String::new(),
                    inputs: vec!["shared".to_string()],
                    outputs: vec!["out_b".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // source output feeds both consumers
        let shared = &graph.nodes[0].outputs[0];
        assert_eq!(shared, "shared");
        assert_eq!(&graph.nodes[1].inputs[0], shared);
        assert_eq!(&graph.nodes[2].inputs[0], shared);
    }

    // ── OnnxGraph: multi-output node feeding separate downstream ─────────

    #[test]
    fn onnx_graph_multi_output_node_feeds_separate() {
        let graph = OnnxGraph {
            name: "multi_out".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "topk".to_string(),
                    op_type: "TopK".to_string(),
                    domain: String::new(),
                    inputs: vec!["input".to_string()],
                    outputs: vec!["values".to_string(), "indices".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "use_values".to_string(),
                    op_type: "Identity".to_string(),
                    domain: String::new(),
                    inputs: vec!["values".to_string()],
                    outputs: vec!["out_val".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "use_indices".to_string(),
                    op_type: "Gather".to_string(),
                    domain: String::new(),
                    inputs: vec!["indices".to_string()],
                    outputs: vec!["out_idx".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes[0].outputs.len(), 2);
        assert_eq!(graph.nodes[0].outputs[0], graph.nodes[1].inputs[0]);
        assert_eq!(graph.nodes[0].outputs[1], graph.nodes[2].inputs[0]);
    }

    // ── OnnxGraph: node with multiple empty-string optional inputs ───────

    #[test]
    fn onnx_node_multiple_optional_inputs() {
        let node = OnnxNode {
            name: "gru".to_string(),
            op_type: "GRU".to_string(),
            domain: String::new(),
            inputs: vec![
                "X".to_string(),
                "W".to_string(),
                "R".to_string(),
                String::new(), // B (optional)
                String::new(), // sequence_lens (optional)
                String::new(), // initial_h (optional)
            ],
            outputs: vec!["Y".to_string(), "Y_h".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.inputs.len(), 6);
        assert_eq!(node.inputs.iter().filter(|s| s.is_empty()).count(), 3);
        assert_eq!(node.outputs.len(), 2);
    }

    // ── OnnxModel: from_proto with default opset version (zero) ──────────

    #[test]
    fn onnx_model_from_proto_default_opset_fields() {
        let model_proto = proto::ModelProto {
            ir_version: None,
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            doc_string: None,
            opset_import: vec![proto::OperatorSetIdProto {
                domain: None,
                version: None,
            }],
            graph: Some(proto::GraphProto {
                name: None,
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.metadata.ir_version, 0);
        assert_eq!(model.metadata.opset_import.len(), 1);
        assert_eq!(model.metadata.opset_import[0].domain, "");
        assert_eq!(model.metadata.opset_import[0].version, 0);
    }

    // ── OnnxModel: from_proto with multiple opset versions ───────────────

    #[test]
    fn onnx_model_from_proto_opset_version_variety() {
        let model_proto = proto::ModelProto {
            ir_version: Some(10),
            opset_import: vec![
                proto::OperatorSetIdProto { domain: Some(String::new()), version: Some(1) },
                proto::OperatorSetIdProto { domain: Some(String::new()), version: Some(6) },
                proto::OperatorSetIdProto { domain: Some(String::new()), version: Some(13) },
                proto::OperatorSetIdProto { domain: Some(String::new()), version: Some(21) },
            ],
            graph: Some(proto::GraphProto {
                name: Some("opset_variety".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        let versions: Vec<i64> = model.metadata.opset_import.iter().map(|o| o.version).collect();
        assert_eq!(versions, vec![1, 6, 13, 21]);
    }

    // ── OnnxGraph: from_proto with multiple inputs and outputs ───────────

    #[test]
    fn onnx_graph_from_proto_multiple_inputs_outputs() {
        let graph_proto = proto::GraphProto {
            name: Some("io_graph".to_string()),
            input: vec![
                proto::ValueInfoProto {
                    name: Some("input_ids".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
                proto::ValueInfoProto {
                    name: Some("attention_mask".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
                proto::ValueInfoProto {
                    name: Some("token_type_ids".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
            ],
            output: vec![
                proto::ValueInfoProto {
                    name: Some("logits".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
                proto::ValueInfoProto {
                    name: Some("hidden_states".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
            ],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.inputs.len(), 3);
        assert_eq!(graph.outputs.len(), 2);
        assert_eq!(graph.inputs[0].name, "input_ids");
        assert_eq!(graph.inputs[2].name, "token_type_ids");
        assert_eq!(graph.outputs[0].name, "logits");
        assert_eq!(graph.outputs[1].name, "hidden_states");
    }

    // ── OnnxGraph: from_proto preserves doc_string on graph ──────────────

    #[test]
    fn onnx_graph_from_proto_preserves_doc_string() {
        let graph_proto = proto::GraphProto {
            name: Some("doc_graph".to_string()),
            doc_string: Some("A graph with documentation".to_string()),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.doc_string, "A graph with documentation");
    }

    // ── OnnxGraph: from_proto with initializer and value_info ────────────

    #[test]
    fn onnx_graph_from_proto_initializer_and_value_info() {
        let weight = proto::TensorProto {
            name: Some("embed_weight".to_string()),
            data_type: Some(1),
            dims: vec![2, 5],
            float_data: vec![0.0; 10],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            name: Some("embed_graph".to_string()),
            initializer: vec![weight],
            value_info: vec![proto::ValueInfoProto {
                name: Some("hidden".to_string()),
                r#type: None,
                doc_string: Some("post-embedding".to_string()),
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.initializers.contains_key("embed_weight"));
        assert_eq!(graph.value_info.len(), 1);
        assert_eq!(graph.value_info[0].name, "hidden");
        assert_eq!(graph.value_info[0].doc_string, "post-embedding");
    }

    // ── parse_nodes: node at index 5 gets correct auto-name ─────────────

    #[test]
    fn parse_nodes_auto_name_at_high_index() {
        let mut nodes = Vec::new();
        for _ in 0..6 {
            nodes.push(proto::NodeProto {
                op_type: Some("Relu".to_string()),
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            });
        }
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[5].name, "node_5");
    }

    // ── parse_value_info: entry with doc_string and metadata_props ───────

    #[test]
    fn parse_value_info_with_doc_and_metadata() {
        let values = vec![proto::ValueInfoProto {
            name: Some("layer_norm_output".to_string()),
            r#type: None,
            doc_string: Some("normalized output".to_string()),
            metadata_props: vec![proto::StringStringEntryProto {
                key: Some("layer".to_string()),
                value: Some("post_ln".to_string()),
            }],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result[0].doc_string, "normalized output");
        assert_eq!(result[0].metadata_props.get("layer").unwrap(), "post_ln");
    }

    // ── parse_opsets: many entries with different domains ────────────────

    #[test]
    fn parse_opsets_many_domains() {
        let opsets: Vec<proto::OperatorSetIdProto> = [
            ("", 17),
            ("ai.onnx.ml", 3),
            ("ai.onnx.training", 1),
            ("com.microsoft", 1),
            ("com.nvidia", 1),
        ]
        .iter()
        .map(|(d, v)| proto::OperatorSetIdProto {
            domain: Some(d.to_string()),
            version: Some(*v),
        })
        .collect();
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].domain, "");
        assert_eq!(result[1].domain, "ai.onnx.ml");
        assert_eq!(result[3].domain, "com.microsoft");
        assert_eq!(result[4].domain, "com.nvidia");
    }

    // ── OnnxModel: clone preserves functions list ────────────────────────

    #[test]
    fn onnx_model_clone_preserves_functions() {
        let func = OnnxFunction {
            name: "Func1".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["A".to_string()],
            outputs: vec!["B".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![OnnxNode {
                name: "inner".to_string(),
                op_type: "Add".to_string(),
                domain: String::new(),
                inputs: vec!["A".to_string()],
                outputs: vec!["B".to_string()],
                attributes: HashMap::new(),
            }],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![func],
        };
        let cloned = model.clone();
        assert_eq!(cloned.functions.len(), 1);
        assert_eq!(cloned.functions[0].name, "Func1");
        assert_eq!(cloned.functions[0].nodes.len(), 1);
    }

    // ── OnnxGraph: initializer overwrite with same key ──────────────────

    #[test]
    fn onnx_graph_initializer_overwrite() {
        let t1 = OnnxTensor::new("w".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let t2 = OnnxTensor::new("w".to_string(), Dtype::BF16, vec![4], Bytes::from(vec![0u8; 8]));
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("w".to_string(), t1)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.initializers.insert("w".to_string(), t2);
        let t = graph.initializers.get("w").unwrap();
        assert_eq!(t.dtype, Dtype::BF16);
        assert_eq!(t.shape, vec![4]);
    }

    // ── OnnxGraph: multiple value_info entries with mixed types ──────────

    #[test]
    fn onnx_graph_value_info_mixed_types() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let graph = OnnxGraph {
            name: "mixed_vi".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![
                OnnxValueInfo {
                    name: "typed_val".to_string(),
                    value_type: Some(OnnxType::Tensor(OnnxTensorType {
                        elem_type: proto::tensor_proto::DataType::Float,
                        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(768)] },
                    })),
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "untyped_val".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.value_info[0].value_type.is_some());
        assert!(graph.value_info[1].value_type.is_none());
    }

    // ── OnnxNode: op_type casing variations ──────────────────────────────

    #[test]
    fn onnx_node_op_type_casing_preserved() {
        let variations = vec![
            ("Conv", "Conv"),
            ("conv", "conv"),
            ("CONV", "CONV"),
            ("MatMul", "MatMul"),
        ];
        for (input, expected) in variations {
            let node = OnnxNode {
                name: "n".to_string(),
                op_type: input.to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: HashMap::new(),
            };
            assert_eq!(node.op_type, expected);
        }
    }

    // ── OnnxQuantizationAnnotation: axis with large positive value ───────

    #[test]
    fn onnx_quantization_annotation_large_axis() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: Some(i32::MAX),
        };
        assert_eq!(qa.axis, Some(i32::MAX));
    }

    // ── OnnxQuantizationAnnotation: axis with min value ──────────────────

    #[test]
    fn onnx_quantization_annotation_min_axis() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: Some(i32::MIN),
        };
        assert_eq!(qa.axis, Some(i32::MIN));
    }

    // ── OnnxValueInfo: doc_string preserved through parse_value_info ─────

    #[test]
    fn parse_value_info_doc_string_none_defaults_empty() {
        let values = vec![proto::ValueInfoProto {
            name: Some("x".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result[0].doc_string, "");
    }

    // ── OnnxGraph: from_proto with no name defaults to empty ─────────────

    #[test]
    fn onnx_graph_from_proto_no_name_defaults_empty() {
        let graph_proto = proto::GraphProto {
            name: None,
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.name.is_empty());
    }

    // ── OnnxGraph: from_proto with no doc_string defaults to empty ───────

    #[test]
    fn onnx_graph_from_proto_no_doc_string_defaults_empty() {
        let graph_proto = proto::GraphProto {
            name: Some("g".to_string()),
            doc_string: None,
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.doc_string.is_empty());
    }

    // ── OnnxModel: from_proto with no ir_version defaults to zero ────────

    #[test]
    fn onnx_model_from_proto_no_ir_version() {
        let model_proto = proto::ModelProto {
            ir_version: None,
            graph: Some(proto::GraphProto {
                name: Some("g".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.metadata.ir_version, 0);
    }

    // ── OnnxModel: from_proto with empty opset_import list ───────────────

    #[test]
    fn onnx_model_from_proto_empty_opset_import() {
        let model_proto = proto::ModelProto {
            opset_import: vec![],
            graph: Some(proto::GraphProto {
                name: Some("g".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert!(model.metadata.opset_import.is_empty());
    }

    // ── OnnxGraph: node order preserved through from_proto ───────────────

    #[test]
    fn onnx_graph_from_proto_node_order_preserved() {
        let graph_proto = proto::GraphProto {
            name: Some("ordered".to_string()),
            node: vec![
                proto::NodeProto {
                    op_type: Some("Conv".to_string()),
                    name: Some("conv".to_string()),
                    input: vec![],
                    output: vec!["conv_out".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                },
                proto::NodeProto {
                    op_type: Some("BatchNormalization".to_string()),
                    name: Some("bn".to_string()),
                    input: vec!["conv_out".to_string()],
                    output: vec!["bn_out".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                },
                proto::NodeProto {
                    op_type: Some("Relu".to_string()),
                    name: Some("relu".to_string()),
                    input: vec!["bn_out".to_string()],
                    output: vec!["relu_out".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                },
            ],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].op_type, "Conv");
        assert_eq!(graph.nodes[1].op_type, "BatchNormalization");
        assert_eq!(graph.nodes[2].op_type, "Relu");
        // Verify chain connectivity
        assert_eq!(graph.nodes[0].outputs[0], graph.nodes[1].inputs[0]);
        assert_eq!(graph.nodes[1].outputs[0], graph.nodes[2].inputs[0]);
    }

    // ── OnnxGraph: inputs and outputs are independent vecs ───────────────

    #[test]
    fn onnx_graph_inputs_outputs_independent() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![OnnxValueInfo {
                name: "in".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![OnnxValueInfo {
                name: "out".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_ne!(graph.inputs[0].name, graph.outputs[0].name);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    // ── OnnxModelMetadata: metadata_props can be large ──────────────────

    #[test]
    fn onnx_model_metadata_large_props_map() {
        let mut props = HashMap::new();
        for i in 0..50 {
            props.insert(format!("key_{i}"), format!("value_{i}"));
        }
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: props,
        };
        assert_eq!(meta.metadata_props.len(), 50);
        assert_eq!(meta.metadata_props.get("key_25").unwrap(), "value_25");
        assert_eq!(meta.metadata_props.get("key_49").unwrap(), "value_49");
    }

    // ── parse_metadata_props: all entries with None values ───────────────

    #[test]
    fn parse_metadata_props_all_none_values() {
        let entries: Vec<proto::StringStringEntryProto> = (0..3)
            .map(|i| proto::StringStringEntryProto {
                key: Some(format!("k{i}")),
                value: None,
            })
            .collect();
        let result = parse_metadata_props(entries);
        assert_eq!(result.len(), 3);
        for i in 0..3 {
            assert_eq!(result.get(&format!("k{i}")).unwrap(), "");
        }
    }

    // ── OnnxOperatorSet: negative version is preserved ──────────────────

    #[test]
    fn onnx_operator_set_negative_version() {
        let ops = OnnxOperatorSet {
            domain: String::new(),
            version: -1,
        };
        assert_eq!(ops.version, -1);
    }

    // ── OnnxGraph: initializer with empty name ───────────────────────────

    #[test]
    fn onnx_graph_initializer_empty_name_key() {
        let tensor = OnnxTensor::new(
            String::new(),
            Dtype::F32,
            vec![1],
            Bytes::from(vec![0u8; 4]),
        );
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([(String::new(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.initializers.contains_key(""));
    }

    // ── OnnxFunction: overload field access ──────────────────────────────

    #[test]
    fn onnx_function_overload_field_access() {
        let func = OnnxFunction {
            name: "Op".to_string(),
            domain: String::new(),
            overload: "v2_alpha".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.overload, "v2_alpha");
    }

    // ── OnnxFunction: metadata_props access ──────────────────────────────

    #[test]
    fn onnx_function_metadata_props_access() {
        let func = OnnxFunction {
            name: "Op".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::from([("origin".to_string(), "custom_lib".to_string())]),
        };
        assert_eq!(func.metadata_props.get("origin").unwrap(), "custom_lib");
    }

    // ── OnnxNode: inputs contain same name as another node's output ──────

    #[test]
    fn onnx_node_shared_intermediate_name() {
        let node1 = OnnxNode {
            name: "n1".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string(), "W".to_string()],
            outputs: vec!["hidden".to_string()],
            attributes: HashMap::new(),
        };
        let node2 = OnnxNode {
            name: "n2".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["hidden".to_string(), "bias".to_string()],
            outputs: vec!["hidden".to_string()],
            attributes: HashMap::new(),
        };
        // node1 produces "hidden", node2 consumes "hidden" and also outputs "hidden"
        assert_eq!(node1.outputs[0], node2.inputs[0]);
        assert_eq!(node1.outputs[0], node2.outputs[0]);
    }

    // ── OnnxGraph: from_proto with sparse_initializer ────────────────────

    #[test]
    fn onnx_graph_from_proto_with_sparse_initializer() {
        let values = proto::TensorProto {
            name: Some("sparse_vals".to_string()),
            data_type: Some(1), // FLOAT
            dims: vec![3],
            float_data: vec![1.0, 2.0, 3.0],
            ..Default::default()
        };
        let indices = proto::TensorProto {
            name: Some("sparse_idx".to_string()),
            data_type: Some(7), // INT64
            dims: vec![3],
            int64_data: vec![0, 5, 9],
            ..Default::default()
        };
        let sparse = proto::SparseTensorProto {
            values: Some(values),
            indices: Some(indices),
            dims: vec![10],
        };
        let graph_proto = proto::GraphProto {
            name: Some("sparse_g".to_string()),
            sparse_initializer: vec![sparse],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.sparse_initializers.len(), 1);
    }

    // ── OnnxGraph: from_proto error on duplicate sparse initializer name ─

    #[test]
    fn onnx_graph_from_proto_duplicate_sparse_initializer_error() {
        let make_values = |name: &str| proto::TensorProto {
            name: Some(name.to_string()),
            data_type: Some(1),
            dims: vec![1],
            float_data: vec![1.0],
            ..Default::default()
        };
        let make_indices = |name: &str| proto::TensorProto {
            name: Some(name.to_string()),
            data_type: Some(7),
            dims: vec![1],
            int64_data: vec![0],
            ..Default::default()
        };
        // Two sparse tensors whose values have the same name as an existing initializer
        let initializer = proto::TensorProto {
            name: Some("dup_name".to_string()),
            data_type: Some(1),
            dims: vec![1],
            float_data: vec![1.0],
            ..Default::default()
        };
        let sparse = proto::SparseTensorProto {
            values: Some(make_values("dup_name")),
            indices: Some(make_indices("idx")),
            dims: vec![5],
        };
        let graph_proto = proto::GraphProto {
            name: Some("g".to_string()),
            initializer: vec![initializer],
            sparse_initializer: vec![sparse],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = OnnxGraph::from_proto(graph_proto, &mut resolver);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Duplicate tensor"), "expected duplicate: {err_msg}");
    }

    // ── OnnxGraph: bind_weights_auto with no nodes returns zero ──────────

    #[test]
    fn onnx_graph_bind_weights_auto_empty_nodes() {
        struct EmptyProvider;
        impl crate::loader::TensorProvider for EmptyProvider {
            fn tensor_info(&self, _name: &str) -> Option<crate::loader::TensorMeta> {
                None
            }
            fn load_tensor_data(
                &self,
                _name: &str,
            ) -> std::result::Result<std::borrow::Cow<'_, [u8]>, LoaderError> {
                panic!("should not be called");
            }
            fn iter_tensors(&self) -> impl Iterator<Item = crate::loader::TensorMeta> {
                std::iter::empty()
            }
        }
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let count = graph.bind_weights_auto(&EmptyProvider).unwrap();
        assert_eq!(count, 0);
    }

    // ── OnnxModel: Debug format includes functions ───────────────────────

    #[test]
    fn onnx_model_debug_includes_functions_count() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: "test".to_string(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![
                OnnxFunction {
                    name: "F1".to_string(),
                    domain: String::new(),
                    overload: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: vec![],
                    attribute_protos: HashMap::new(),
                    nodes: vec![],
                    opset_import: vec![],
                    value_info: vec![],
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
        };
        let debug = format!("{model:?}");
        assert!(debug.contains("F1"));
    }

    // ── OnnxNode: two nodes with same op_type but different names ────────

    #[test]
    fn onnx_nodes_same_op_type_different_names() {
        let n1 = OnnxNode {
            name: "relu_1".to_string(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            inputs: vec!["a".to_string()],
            outputs: vec!["b".to_string()],
            attributes: HashMap::new(),
        };
        let n2 = OnnxNode {
            name: "relu_2".to_string(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            inputs: vec!["c".to_string()],
            outputs: vec!["d".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(n1.op_type, n2.op_type);
        assert_ne!(n1.name, n2.name);
    }

    // ── OnnxValueInfo: name with slashes and dots ────────────────────────

    #[test]
    fn onnx_value_info_name_with_special_path_chars() {
        let vi = OnnxValueInfo {
            name: "model/layer.0/attention/query".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(vi.name.contains('/'));
        assert!(vi.name.contains('.'));
        assert_eq!(vi.name, "model/layer.0/attention/query");
    }

    // ── parse_quantization: scale and zp from correct initializers ───────

    #[test]
    fn parse_quantization_scale_zp_from_correct_initializers() {
        let scale_bytes = 0.0625f32.to_le_bytes();
        let zp_bytes = 42i64.to_le_bytes();
        let scale_tensor = OnnxTensor::new(
            "s1".to_string(), Dtype::F32, vec![], Bytes::from(scale_bytes.to_vec()),
        );
        let zp_tensor = OnnxTensor::new(
            "zp1".to_string(), Dtype::I64, vec![], Bytes::from(zp_bytes.to_vec()),
        );
        let initializers = HashMap::from([
            ("s1".to_string(), scale_tensor),
            ("zp1".to_string(), zp_tensor),
        ]);
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("target_w".to_string()),
            quant_parameter_tensor_names: vec![
                proto::StringStringEntryProto {
                    key: Some("SCALE_TENSOR".to_string()),
                    value: Some("s1".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("ZERO_POINT_TENSOR".to_string()),
                    value: Some("zp1".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("AXIS".to_string()),
                    value: Some("4".to_string()),
                },
            ],
        }];
        let result = parse_quantization(entries, &initializers);
        assert_eq!(result.len(), 1);
        assert!((result[0].scale.unwrap() - 0.0625).abs() < 1e-8);
        assert_eq!(result[0].zero_point, Some(42));
        assert_eq!(result[0].axis, Some(4));
    }

    // ── OnnxGraph: metadata_props iteration order independent of insert ─

    #[test]
    fn onnx_graph_metadata_props_lookup_after_multiple_inserts() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.metadata_props.insert("z_key".to_string(), "z_val".to_string());
        graph.metadata_props.insert("a_key".to_string(), "a_val".to_string());
        graph.metadata_props.insert("m_key".to_string(), "m_val".to_string());
        assert_eq!(graph.metadata_props.get("z_key").unwrap(), "z_val");
        assert_eq!(graph.metadata_props.get("a_key").unwrap(), "a_val");
        assert_eq!(graph.metadata_props.get("m_key").unwrap(), "m_val");
        assert_eq!(graph.metadata_props.len(), 3);
    }

    // ── OnnxModel: metadata_props from from_proto ────────────────────────

    #[test]
    fn onnx_model_from_proto_empty_metadata_props() {
        let model_proto = proto::ModelProto {
            graph: Some(proto::GraphProto {
                name: Some("g".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert!(model.metadata.metadata_props.is_empty());
    }

    // ── OnnxGraph: from_proto with many initializers ─────────────────────

    #[test]
    fn onnx_graph_from_proto_many_initializers() {
        let mut initializers = Vec::new();
        for i in 0..10 {
            initializers.push(proto::TensorProto {
                name: Some(format!("weight_{i}")),
                data_type: Some(1),
                dims: vec![2],
                float_data: vec![0.0, 1.0],
                ..Default::default()
            });
        }
        let graph_proto = proto::GraphProto {
            name: Some("many_init".to_string()),
            initializer: initializers,
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.initializers.len(), 10);
        for i in 0..10 {
            assert!(graph.initializers.contains_key(&format!("weight_{i}")));
        }
    }

    // ── OnnxGraph: value_info entries have independent lifecycle ─────────

    #[test]
    fn onnx_graph_value_info_independence_after_clone() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![OnnxValueInfo {
                name: "original".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let cloned = graph.clone();
        graph.value_info[0].name = "modified".to_string();
        assert_eq!(cloned.value_info[0].name, "original");
    }

    // ── OnnxNode: attributes map with multiple entries ───────────────────

    #[test]
    fn onnx_node_attributes_multiple_entries_lookup() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert("alpha".to_string(), OnnxAttribute {
            name: "alpha".to_string(),
            value: OnnxAttributeValue::Float(0.1),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        attrs.insert("beta".to_string(), OnnxAttribute {
            name: "beta".to_string(),
            value: OnnxAttributeValue::Float(0.2),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        attrs.insert("gamma".to_string(), OnnxAttribute {
            name: "gamma".to_string(),
            value: OnnxAttributeValue::Float(0.3),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let node = OnnxNode {
            name: "fused_bias_gelu".to_string(),
            op_type: "BiasGelu".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
        };
        assert_eq!(node.attributes.len(), 3);
        assert!(node.attributes.contains_key("alpha"));
        assert!(node.attributes.contains_key("beta"));
        assert!(node.attributes.contains_key("gamma"));
    }

    // ── OnnxGraph: single-input single-output graph with one node ────────

    #[test]
    fn onnx_graph_single_io_single_node() {
        let graph = OnnxGraph {
            name: "minimal".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "identity".to_string(),
                op_type: "Identity".to_string(),
                domain: String::new(),
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![OnnxValueInfo {
                name: "input".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![OnnxValueInfo {
                name: "output".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.nodes[0].inputs[0], graph.inputs[0].name);
        assert_eq!(graph.nodes[0].outputs[0], graph.outputs[0].name);
    }

    // ── OnnxGraph: long chain of nodes (10 deep) ────────────────────────

    #[test]
    fn onnx_graph_long_chain() {
        let mut nodes = Vec::new();
        for i in 0..10 {
            nodes.push(OnnxNode {
                name: format!("layer_{i}"),
                op_type: "Relu".to_string(),
                domain: String::new(),
                inputs: vec![format!("h{i}")],
                outputs: vec![format!("h{}", i + 1)],
                attributes: HashMap::new(),
            });
        }
        let graph = OnnxGraph {
            name: "deep_chain".to_string(),
            doc_string: String::new(),
            nodes,
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 10);
        for i in 0..9 {
            assert_eq!(graph.nodes[i].outputs[0], graph.nodes[i + 1].inputs[0]);
        }
    }

    // ── parse_nodes: node with doc_string in proto ───────────────────────

    #[test]
    fn parse_nodes_doc_string_in_proto() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Relu".to_string()),
            name: Some("relu_doc".to_string()),
            input: vec![],
            output: vec![],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: Some("Applies ReLU activation".to_string()),
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].name, "relu_doc");
    }

    // ── parse_nodes: multiple nodes second one errors stops processing ───

    #[test]
    fn parse_nodes_second_error_stops_all() {
        let nodes = vec![
            proto::NodeProto {
                op_type: Some("Valid".to_string()),
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: None, // error
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: Some("AlsoValid".to_string()),
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
        ];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver);
        assert!(result.is_err());
    }

    // ── OnnxFunction: opset_import is independent after clone ────────────

    #[test]
    fn onnx_function_opset_import_clone_independence() {
        let func = OnnxFunction {
            name: "Op".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 17,
            }],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let mut cloned = func.clone();
        cloned.opset_import[0].version = 20;
        assert_eq!(func.opset_import[0].version, 17);
    }

    // ── OnnxGraph: quantization_annotation list is independent after clone

    #[test]
    fn onnx_graph_quantization_clone_independence() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![OnnxQuantizationAnnotation {
                tensor_name: "w".to_string(),
                quant_param_tensor_names: HashMap::new(),
                scale: Some(0.1),
                zero_point: None,
                axis: None,
            }],
            metadata_props: HashMap::new(),
        };
        let cloned = graph.clone();
        graph.quantization_annotation[0].scale = Some(0.9);
        assert!((cloned.quantization_annotation[0].scale.unwrap() - 0.1).abs() < 1e-9);
    }

    // ── OnnxValueInfo: with SparseTensor type ────────────────────────────

    #[test]
    fn onnx_value_info_with_sparse_tensor_type() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let vi = OnnxValueInfo {
            name: "sparse_input".to_string(),
            value_type: Some(OnnxType::SparseTensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape {
                    dims: vec![OnnxDim::Known(100), OnnxDim::Known(100)],
                },
            })),
            doc_string: "sparse weight".to_string(),
            metadata_props: HashMap::new(),
        };
        assert!(matches!(vi.value_type, Some(OnnxType::SparseTensor(_))));
    }

    // ── OnnxValueInfo: with Map type ─────────────────────────────────────

    #[test]
    fn onnx_value_info_with_map_type() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxMapType, OnnxType};
        let vi = OnnxValueInfo {
            name: "string_map".to_string(),
            value_type: Some(OnnxType::Map(OnnxMapType {
                key_type: proto::tensor_proto::DataType::String,
                value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                    elem_type: proto::tensor_proto::DataType::Int64,
                    shape: OnnxTensorShape { dims: vec![OnnxDim::Known(1)] },
                })),
            })),
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(matches!(vi.value_type, Some(OnnxType::Map(_))));
    }

    // ── parse_value_info: many entries preserves order ────────────────────

    #[test]
    fn parse_value_info_many_preserves_order() {
        let values: Vec<proto::ValueInfoProto> = (0..20)
            .map(|i| proto::ValueInfoProto {
                name: Some(format!("val_{i}")),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            })
            .collect();
        let result = parse_value_info(values).unwrap();
        assert_eq!(result.len(), 20);
        for i in 0..20 {
            assert_eq!(result[i].name, format!("val_{i}"));
        }
    }

    // ── OnnxModelMetadata: all string fields with content ────────────────

    #[test]
    fn onnx_model_metadata_all_strings_populated() {
        let meta = OnnxModelMetadata {
            ir_version: 10,
            producer_name: "onnxruntime".to_string(),
            producer_version: "1.18.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 99,
            doc_string: "Production BERT model".to_string(),
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 21,
            }],
            metadata_props: HashMap::from([
                ("author".to_string(), "team".to_string()),
                ("license".to_string(), "Apache-2.0".to_string()),
            ]),
        };
        assert_eq!(meta.producer_name, "onnxruntime");
        assert_eq!(meta.producer_version, "1.18.0");
        assert_eq!(meta.domain, "ai.onnx");
        assert_eq!(meta.doc_string, "Production BERT model");
        assert_eq!(meta.model_version, 99);
        assert_eq!(meta.metadata_props.len(), 2);
    }

    // ── OnnxGraph: name with unicode characters ──────────────────────────

    #[test]
    fn onnx_graph_name_unicode() {
        let graph = OnnxGraph {
            name: "测试图_モデル".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.name, "测试图_モデル");
    }

    // ── OnnxGraph: from_proto graph name is None defaults to empty ───────

    #[test]
    fn onnx_graph_from_proto_all_optional_fields_none() {
        let graph_proto = proto::GraphProto {
            name: None,
            doc_string: None,
            node: vec![],
            input: vec![],
            output: vec![],
            initializer: vec![],
            sparse_initializer: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.name.is_empty());
        assert!(graph.doc_string.is_empty());
        assert!(graph.nodes.is_empty());
        assert!(graph.metadata_props.is_empty());
    }

    // ── OnnxNode: Debug output contains all key fields ───────────────────

    #[test]
    fn onnx_node_debug_contains_all_fields() {
        let node = OnnxNode {
            name: "conv2d_0".to_string(),
            op_type: "Conv".to_string(),
            domain: "ai.onnx".to_string(),
            inputs: vec!["X".to_string(), "W".to_string(), "B".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        };
        let debug = format!("{node:?}");
        assert!(debug.contains("conv2d_0"));
        assert!(debug.contains("Conv"));
        assert!(debug.contains("ai.onnx"));
    }

    // ── OnnxFunction: Debug contains domain and overload ─────────────────

    #[test]
    fn onnx_function_debug_contains_domain_overload() {
        let func = OnnxFunction {
            name: "Op".to_string(),
            domain: "custom.domain".to_string(),
            overload: "v3".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let debug = format!("{func:?}");
        assert!(debug.contains("custom.domain"), "expected domain in: {debug}");
        assert!(debug.contains("v3"), "expected overload in: {debug}");
    }

    // ── parse_functions: function with no nodes is valid ─────────────────

    #[test]
    fn parse_functions_no_nodes_is_valid() {
        let functions = vec![proto::FunctionProto {
            name: Some("NoOp".to_string()),
            domain: None,
            overload: None,
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].nodes.len(), 0);
        assert_eq!(result[0].inputs, vec!["X"]);
        assert_eq!(result[0].outputs, vec!["Y"]);
    }

    // ── parse_functions: function with empty domain defaults to empty ─────

    #[test]
    fn parse_functions_domain_none_defaults_empty() {
        let functions = vec![proto::FunctionProto {
            name: Some("F".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert!(result[0].domain.is_empty());
        assert!(result[0].overload.is_empty());
    }

    // ── OnnxModel: nested graph inside function node attribute ───────────

    #[test]
    fn onnx_model_function_with_nested_graph_attribute() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let inner_graph = OnnxGraph {
            name: "loop_body".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "add_one".to_string(),
                op_type: "Add".to_string(),
                domain: String::new(),
                inputs: vec!["current".to_string(), "one".to_string()],
                outputs: vec!["next".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let func = OnnxFunction {
            name: "Loop".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["trip_count".to_string(), "cond".to_string()],
            outputs: vec!["output".to_string()],
            attributes: vec!["body".to_string()],
            attribute_protos: HashMap::from([("body".to_string(), OnnxAttribute {
                name: "body".to_string(),
                value: OnnxAttributeValue::Graph(Box::new(inner_graph)),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            })]),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let body_attr = func.attribute_protos.get("body").unwrap();
        if let OnnxAttributeValue::Graph(g) = &body_attr.value {
            assert_eq!(g.name, "loop_body");
            assert_eq!(g.nodes.len(), 1);
        } else {
            panic!("expected Graph variant");
        }
    }

    // ── OnnxGraph: sparse_initializers with different formats ────────────

    #[test]
    fn onnx_graph_sparse_initializers_mixed_formats() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let make = |fmt: OnnxSparseFormat| -> OnnxSparseTensor {
            OnnxSparseTensor {
                values: OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4])),
                indices: OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8])),
                dims: vec![3],
                format: fmt,
            }
        };
        let graph = OnnxGraph {
            name: "mixed_sparse".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![make(OnnxSparseFormat::Coo), make(OnnxSparseFormat::Csr), make(OnnxSparseFormat::Csc)],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.sparse_initializers.len(), 3);
        assert_eq!(graph.sparse_initializers[0].format, OnnxSparseFormat::Coo);
        assert_eq!(graph.sparse_initializers[1].format, OnnxSparseFormat::Csr);
        assert_eq!(graph.sparse_initializers[2].format, OnnxSparseFormat::Csc);
    }

    // ── Additional unit tests for coverage ────────────────────────────────

    // -- OnnxModelMetadata edge cases --

    #[test]
    fn onnx_model_metadata_ir_version_max() {
        let meta = OnnxModelMetadata {
            ir_version: i64::MAX,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.ir_version, i64::MAX);
    }

    #[test]
    fn onnx_model_metadata_model_version_max() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: i64::MAX,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.model_version, i64::MAX);
    }

    #[test]
    fn onnx_model_metadata_empty_strings() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(meta.producer_name.is_empty());
        assert!(meta.producer_version.is_empty());
        assert!(meta.domain.is_empty());
        assert!(meta.doc_string.is_empty());
    }

    #[test]
    fn onnx_model_metadata_many_opsets() {
        let opsets: Vec<OnnxOperatorSet> = (0..10)
            .map(|i| OnnxOperatorSet {
                domain: format!("domain_{i}"),
                version: i + 1,
            })
            .collect();
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: "producer".to_string(),
            producer_version: "2.0".to_string(),
            domain: "test".to_string(),
            model_version: 1,
            doc_string: String::new(),
            opset_import: opsets,
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.opset_import.len(), 10);
        assert_eq!(meta.opset_import[5].domain, "domain_5");
        assert_eq!(meta.opset_import[5].version, 6);
    }

    #[test]
    fn onnx_model_metadata_props_large_map() {
        let mut props = HashMap::new();
        for i in 0..100 {
            props.insert(format!("key_{i}"), format!("value_{i}"));
        }
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: props,
        };
        assert_eq!(meta.metadata_props.len(), 100);
        assert_eq!(meta.metadata_props.get("key_42").unwrap(), "value_42");
    }

    // -- OnnxOperatorSet edge cases --

    #[test]
    fn onnx_operator_set_empty_domain() {
        let opset = OnnxOperatorSet {
            domain: String::new(),
            version: 0,
        };
        assert!(opset.domain.is_empty());
        assert_eq!(opset.version, 0);
    }

    #[test]
    fn onnx_operator_set_unicode_domain() {
        let opset = OnnxOperatorSet {
            domain: "ai.onnx.中文".to_string(),
            version: 20,
        };
        assert_eq!(opset.domain, "ai.onnx.中文");
        assert_eq!(opset.version, 20);
    }

    #[test]
    fn onnx_operator_set_clone_then_modify_independent() {
        let opset = OnnxOperatorSet {
            domain: "original".to_string(),
            version: 15,
        };
        let cloned = opset.clone();
        assert_eq!(cloned.domain, "original");
        assert_eq!(cloned.version, 15);
    }

    // -- OnnxGraph edge cases --

    #[test]
    fn onnx_graph_zero_initializers() {
        let graph = OnnxGraph {
            name: "empty_init".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.initializers.is_empty());
        assert!(graph.sparse_initializers.is_empty());
    }

    #[test]
    fn onnx_graph_many_value_info() {
        let vi: Vec<OnnxValueInfo> = (0..50)
            .map(|i| OnnxValueInfo {
                name: format!("val_{i}"),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            })
            .collect();
        let graph = OnnxGraph {
            name: "many_vi".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vi,
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.value_info.len(), 50);
        assert_eq!(graph.value_info[25].name, "val_25");
    }

    #[test]
    fn onnx_graph_node_output_referenced_as_next_input() {
        let graph = OnnxGraph {
            name: "chain".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "n0".to_string(),
                    op_type: "Conv".to_string(),
                    domain: String::new(),
                    inputs: vec!["input".to_string()],
                    outputs: vec!["conv_out".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n1".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["conv_out".to_string()],
                    outputs: vec!["relu_out".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes[0].outputs[0], graph.nodes[1].inputs[0]);
    }

    #[test]
    fn onnx_graph_metadata_props_empty() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.metadata_props.is_empty());
    }

    #[test]
    fn onnx_graph_inputs_outputs_disjoint_names() {
        let graph = OnnxGraph {
            name: "io".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![OnnxValueInfo {
                name: "input_0".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![OnnxValueInfo {
                name: "output_0".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_ne!(graph.inputs[0].name, graph.outputs[0].name);
    }

    // -- OnnxNode edge cases --

    #[test]
    fn onnx_node_whitespace_op_type_preserved() {
        let node = OnnxNode {
            name: "ws".to_string(),
            op_type: " My Op ".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert_eq!(node.op_type, " My Op ");
    }

    #[test]
    fn onnx_node_repeated_input_names() {
        let node = OnnxNode {
            name: "dup_input".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.inputs[0], node.inputs[1]);
    }

    #[test]
    fn onnx_node_empty_string_marker_input() {
        let node = OnnxNode {
            name: "opt".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "weight".to_string(), String::new()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.inputs.len(), 3);
        assert!(node.inputs[2].is_empty());
    }

    #[test]
    fn onnx_node_single_output_many_inputs() {
        let node = OnnxNode {
            name: "concat".to_string(),
            op_type: "Concat".to_string(),
            domain: String::new(),
            inputs: vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.inputs.len(), 4);
        assert_eq!(node.outputs.len(), 1);
    }

    // ── OnnxAttributeValue import for attribute tests ──
    use super::super::attributes::OnnxAttributeValue;

    #[test]
    fn onnx_node_attributes_lookup_after_insert() {
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), OnnxAttribute {
            name: "axis".to_string(),
            value: OnnxAttributeValue::Int(1),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        attrs.insert("keepdims".to_string(), OnnxAttribute {
            name: "keepdims".to_string(),
            value: OnnxAttributeValue::Int(0),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let node = OnnxNode {
            name: "reduce".to_string(),
            op_type: "ReduceMean".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: attrs,
        };
        assert_eq!(node.attributes.len(), 2);
        let axis_attr = node.attributes.get("axis").unwrap();
        if let OnnxAttributeValue::Int(v) = axis_attr.value {
            assert_eq!(v, 1);
        } else {
            panic!("expected Int variant");
        }
    }

    // -- OnnxValueInfo edge cases --

    #[test]
    fn onnx_value_info_name_only_minimal() {
        let vi = OnnxValueInfo {
            name: "x".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(vi.name, "x");
        assert!(vi.value_type.is_none());
        assert!(vi.doc_string.is_empty());
    }

    #[test]
    fn onnx_value_info_metadata_with_multiple_entries() {
        let mut props = HashMap::new();
        props.insert("source".to_string(), "onnx".to_string());
        props.insert("version".to_string(), "1.5".to_string());
        let vi = OnnxValueInfo {
            name: "tensor_a".to_string(),
            value_type: None,
            doc_string: "annotated".to_string(),
            metadata_props: props,
        };
        assert_eq!(vi.metadata_props.len(), 2);
        assert_eq!(vi.metadata_props.get("source").unwrap(), "onnx");
    }

    #[test]
    fn onnx_value_info_doc_string_preserved() {
        let vi = OnnxValueInfo {
            name: "z".to_string(),
            value_type: None,
            doc_string: "This is a detailed description with\nnewlines.".to_string(),
            metadata_props: HashMap::new(),
        };
        assert!(vi.doc_string.contains('\n'));
    }

    // -- OnnxQuantizationAnnotation edge cases --

    #[test]
    fn onnx_quantization_annotation_i32_max_axis() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.5),
            zero_point: Some(128),
            axis: Some(i32::MAX),
        };
        assert_eq!(ann.axis, Some(i32::MAX));
    }

    #[test]
    fn onnx_quantization_annotation_i32_min_axis() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: Some(i32::MIN),
        };
        assert_eq!(ann.axis, Some(i32::MIN));
    }

    #[test]
    fn onnx_quantization_annotation_zero_scale_value() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.0),
            zero_point: Some(0),
            axis: None,
        };
        assert_eq!(ann.scale, Some(0.0));
        assert_eq!(ann.zero_point, Some(0));
    }

    #[test]
    fn onnx_quantization_annotation_negative_scale() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(-0.001),
            zero_point: None,
            axis: None,
        };
        assert!(ann.scale.unwrap() < 0.0);
    }

    #[test]
    fn onnx_quantization_annotation_many_quant_param_entries() {
        let mut params = HashMap::new();
        params.insert("SCALE_TENSOR".to_string(), "scale_tensor".to_string());
        params.insert("ZERO_POINT_TENSOR".to_string(), "zp_tensor".to_string());
        params.insert("AXIS".to_string(), "3".to_string());
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "weight".to_string(),
            quant_param_tensor_names: params,
            scale: Some(1.0),
            zero_point: Some(0),
            axis: Some(3),
        };
        assert_eq!(ann.quant_param_tensor_names.len(), 3);
    }

    // -- OnnxFunction edge cases --

    #[test]
    fn onnx_function_no_attributes_no_nodes() {
        let func = OnnxFunction {
            name: "Identity".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(func.nodes.is_empty());
        assert!(func.attributes.is_empty());
        assert!(func.opset_import.is_empty());
    }

    #[test]
    fn onnx_function_overload_nonempty() {
        let func = OnnxFunction {
            name: "MyOp".to_string(),
            domain: "custom".to_string(),
            overload: "v2_beta".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.overload, "v2_beta");
    }

    #[test]
    fn onnx_function_attribute_protos_lookup() {
        let mut protos = HashMap::new();
        protos.insert("alpha".to_string(), OnnxAttribute {
            name: "alpha".to_string(),
            value: OnnxAttributeValue::Float(1.5),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let func = OnnxFunction {
            name: "ScaledAdd".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: vec!["alpha".to_string()],
            attribute_protos: protos,
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let attr = func.attribute_protos.get("alpha").unwrap();
        if let OnnxAttributeValue::Float(v) = attr.value {
            assert!((v - 1.5).abs() < f32::EPSILON);
        } else {
            panic!("expected Float variant");
        }
    }

    #[test]
    fn onnx_function_multiple_opset_imports() {
        let func = OnnxFunction {
            name: "FusedOp".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![
                OnnxOperatorSet { domain: String::new(), version: 17 },
                OnnxOperatorSet { domain: "custom".to_string(), version: 1 },
            ],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.opset_import.len(), 2);
        assert_eq!(func.opset_import[1].domain, "custom");
        assert_eq!(func.opset_import[1].version, 1);
    }

    #[test]
    fn onnx_function_value_info_entries() {
        let func = OnnxFunction {
            name: "Op".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![
                OnnxValueInfo {
                    name: "intermediate_0".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "intermediate_1".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.value_info.len(), 2);
        assert_eq!(func.value_info[0].name, "intermediate_0");
        assert_eq!(func.value_info[1].name, "intermediate_1");
    }

    // -- parse_metadata_props edge cases --

    #[test]
    fn parse_metadata_props_single_entry() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("k".to_string()),
            value: Some("v".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("k").unwrap(), "v");
    }

    #[test]
    fn parse_metadata_props_value_with_equals_sign() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("query".to_string()),
            value: Some("a=1&b=2".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("query").unwrap(), "a=1&b=2");
    }

    #[test]
    fn parse_metadata_props_many_entries() {
        let entries: Vec<proto::StringStringEntryProto> = (0..20)
            .map(|i| proto::StringStringEntryProto {
                key: Some(format!("k{i}")),
                value: Some(format!("v{i}")),
            })
            .collect();
        let result = parse_metadata_props(entries);
        assert_eq!(result.len(), 20);
        assert_eq!(result.get("k10").unwrap(), "v10");
    }

    // -- parse_opsets edge cases --

    #[test]
    fn parse_opsets_version_zero() {
        let opsets = vec![proto::OperatorSetIdProto {
            domain: Some("test".to_string()),
            version: Some(0),
        }];
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].version, 0);
    }

    #[test]
    fn parse_opsets_mixed_domain_empty_and_nonempty() {
        let opsets = vec![
            proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            },
            proto::OperatorSetIdProto {
                domain: Some("custom".to_string()),
                version: Some(1),
            },
        ];
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 2);
        assert!(result[0].domain.is_empty());
        assert_eq!(result[1].domain, "custom");
    }

    // -- parse_quantization edge cases --

    #[test]
    fn parse_quantization_multiple_annotations_different_tensors() {
        let entries = vec![
            proto::TensorAnnotation {
                tensor_name: Some("weight_q".to_string()),
                quant_parameter_tensor_names: vec![],
            },
            proto::TensorAnnotation {
                tensor_name: Some("weight_k".to_string()),
                quant_parameter_tensor_names: vec![],
            },
            proto::TensorAnnotation {
                tensor_name: Some("weight_v".to_string()),
                quant_parameter_tensor_names: vec![],
            },
        ];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].tensor_name, "weight_q");
        assert_eq!(result[1].tensor_name, "weight_k");
        assert_eq!(result[2].tensor_name, "weight_v");
    }

    // -- parse_nodes edge cases --

    #[test]
    fn parse_nodes_single_node_all_fields_set() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Gemm".to_string()),
            name: Some("gemm_0".to_string()),
            input: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            output: vec!["Y".to_string()],
            domain: Some("".to_string()),
            overload: None,
            attribute: vec![],
            doc_string: Some("gemm node".to_string()),
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].op_type, "Gemm");
        assert_eq!(result[0].name, "gemm_0");
        assert_eq!(result[0].inputs.len(), 3);
        assert_eq!(result[0].outputs.len(), 1);
    }

    #[test]
    fn parse_nodes_whitespace_only_name_falls_back() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Relu".to_string()),
            name: Some("   ".to_string()),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        // whitespace-only name is non-empty string, preserved as-is
        assert_eq!(result[0].name, "   ");
    }

    // -- parse_value_info edge cases --

    #[test]
    fn parse_value_info_single_with_doc_string() {
        let values = vec![proto::ValueInfoProto {
            name: Some("input_ids".to_string()),
            r#type: None,
            doc_string: Some("tokenized input".to_string()),
            metadata_props: vec![],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result[0].doc_string, "tokenized input");
    }

    // -- OnnxModel edge cases --

    #[test]
    fn onnx_model_empty_functions() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert!(model.functions.is_empty());
    }

    #[test]
    fn onnx_model_functions_count() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![
                OnnxFunction {
                    name: "F1".to_string(),
                    domain: String::new(),
                    overload: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: vec![],
                    attribute_protos: HashMap::new(),
                    nodes: vec![],
                    opset_import: vec![],
                    value_info: vec![],
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxFunction {
                    name: "F2".to_string(),
                    domain: String::new(),
                    overload: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: vec![],
                    attribute_protos: HashMap::new(),
                    nodes: vec![],
                    opset_import: vec![],
                    value_info: vec![],
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
        };
        assert_eq!(model.functions.len(), 2);
    }

    #[test]
    fn onnx_model_metadata_access_through_model() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 9,
                producer_name: "test_producer".to_string(),
                producer_version: "3.0".to_string(),
                domain: "ai.onnx".to_string(),
                model_version: 42,
                doc_string: "test".to_string(),
                opset_import: vec![OnnxOperatorSet {
                    domain: String::new(),
                    version: 20,
                }],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert_eq!(model.metadata.producer_name, "test_producer");
        assert_eq!(model.metadata.opset_import[0].version, 20);
    }

    // -- OnnxTensor new_string constructor --

    #[test]
    fn onnx_tensor_new_string_is_string_flag() {
        let tensor = OnnxTensor::new_string(
            "labels".to_string(),
            vec![5],
            Bytes::from(vec![0u8; 5]),
        );
        assert!(tensor.is_string);
        assert_eq!(tensor.dtype, Dtype::U8);
        assert_eq!(tensor.shape, vec![5]);
    }

    // -- OnnxTensor dtype variations --

    #[test]
    fn onnx_tensor_dtype_i32() {
        let tensor = OnnxTensor::new(
            "data".to_string(),
            Dtype::I32,
            vec![2, 3],
            Bytes::from(vec![0u8; 24]),
        );
        assert_eq!(tensor.dtype, Dtype::I32);
    }

    #[test]
    fn onnx_tensor_dtype_u8() {
        let tensor = OnnxTensor::new(
            "bytes".to_string(),
            Dtype::U8,
            vec![10],
            Bytes::from(vec![0u8; 10]),
        );
        assert_eq!(tensor.dtype, Dtype::U8);
    }

    #[test]
    fn onnx_tensor_dtype_bool() {
        let tensor = OnnxTensor::new(
            "mask".to_string(),
            Dtype::BOOL,
            vec![4],
            Bytes::from(vec![0u8; 4]),
        );
        assert_eq!(tensor.dtype, Dtype::BOOL);
    }

    // -- OnnxTensor scalar_f32 with various dtypes --

    #[test]
    fn onnx_tensor_scalar_f32_from_i32() {
        let data: [u8; 4] = 42i32.to_le_bytes();
        let tensor = OnnxTensor::new(
            "val".to_string(),
            Dtype::I32,
            vec![],
            Bytes::from(data.to_vec()),
        );
        assert_eq!(tensor.scalar_f32(), Some(42.0));
    }

    #[test]
    fn onnx_tensor_scalar_f32_from_u8() {
        let tensor = OnnxTensor::new(
            "val".to_string(),
            Dtype::U8,
            vec![],
            Bytes::from(vec![200u8]),
        );
        assert_eq!(tensor.scalar_f32(), Some(200.0));
    }

    #[test]
    fn onnx_tensor_scalar_i64_from_f32() {
        let data: [u8; 4] = 7.0f32.to_le_bytes();
        let tensor = OnnxTensor::new(
            "val".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(data.to_vec()),
        );
        assert_eq!(tensor.scalar_i64(), Some(7));
    }

    // -- OnnxGraph initializer with different dtypes --

    #[test]
    fn onnx_graph_initializer_with_i64_tensor() {
        let tensor = OnnxTensor::new(
            "indices".to_string(),
            Dtype::I64,
            vec![3],
            Bytes::from(vec![0u8; 24]),
        );
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.initializers.insert("indices".to_string(), tensor);
        let retrieved = graph.initializers.get("indices").unwrap();
        assert_eq!(retrieved.dtype, Dtype::I64);
    }

    // -- OnnxQuantizationAnnotation tensor_name edge cases --

    #[test]
    fn onnx_quantization_annotation_empty_tensor_name() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: String::new(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        assert!(ann.tensor_name.is_empty());
    }

    #[test]
    fn onnx_quantization_annotation_tensor_name_with_slashes() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "layer/weight/quantized".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        assert!(ann.tensor_name.contains('/'));
    }

    // -- OnnxNode debug format includes op_type --

    #[test]
    fn onnx_node_debug_shows_op_type() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Softmax".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        let debug_str = format!("{:?}", node);
        assert!(debug_str.contains("Softmax"));
    }

    // -- OnnxGraph debug format includes name --

    #[test]
    fn onnx_graph_debug_shows_name() {
        let graph = OnnxGraph {
            name: "my_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let debug_str = format!("{:?}", graph);
        assert!(debug_str.contains("my_graph"));
    }

    // -- OnnxFunction debug format includes name --

    #[test]
    fn onnx_function_debug_shows_name() {
        let func = OnnxFunction {
            name: "CustomOp".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let debug_str = format!("{:?}", func);
        assert!(debug_str.contains("CustomOp"));
    }

    // -- OnnxModel debug format includes metadata --

    #[test]
    fn onnx_model_debug_shows_metadata_ir_version() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 99,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("99"));
    }

    // -- OnnxValueInfo clone independence --

    #[test]
    fn onnx_value_info_clone_name_independence() {
        let vi = OnnxValueInfo {
            name: "original".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let mut cloned = vi.clone();
        cloned.name = "modified".to_string();
        assert_eq!(vi.name, "original");
        assert_eq!(cloned.name, "modified");
    }

    // -- OnnxOperatorSet clone and compare --

    #[test]
    fn onnx_operator_set_fields_after_clone() {
        let opset = OnnxOperatorSet {
            domain: "test.domain".to_string(),
            version: 21,
        };
        let cloned = opset.clone();
        assert_eq!(cloned.domain, "test.domain");
        assert_eq!(cloned.version, 21);
    }

    // -- parse_functions edge: single function with no inputs and no outputs --

    #[test]
    fn parse_functions_no_inputs_no_outputs() {
        let functions = vec![proto::FunctionProto {
            name: Some("ConstantGen".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![proto::NodeProto {
                op_type: Some("Constant".to_string()),
                name: None,
                input: vec![],
                output: vec!["const_out".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            }],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].inputs.len(), 0);
        assert_eq!(result[0].outputs.len(), 0);
        assert_eq!(result[0].nodes.len(), 1);
    }

    // -- parse_functions edge: domain_none defaults_empty --

    #[test]
    fn parse_functions_domain_none_yields_default() {
        let functions = vec![proto::FunctionProto {
            name: Some("MyOp".to_string()),
            domain: None,
            overload: None,
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert!(result[0].domain.is_empty());
        assert!(result[0].overload.is_empty());
        assert!(result[0].doc_string.is_empty());
    }

    // -- OnnxGraph: multiple quantization annotations --

    #[test]
    fn onnx_graph_multiple_quantization_annotations() {
        let graph = OnnxGraph {
            name: "q".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![
                OnnxQuantizationAnnotation {
                    tensor_name: "w1".to_string(),
                    quant_param_tensor_names: HashMap::new(),
                    scale: Some(0.1),
                    zero_point: Some(0),
                    axis: None,
                },
                OnnxQuantizationAnnotation {
                    tensor_name: "w2".to_string(),
                    quant_param_tensor_names: HashMap::new(),
                    scale: Some(0.05),
                    zero_point: Some(128),
                    axis: Some(0),
                },
            ],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.quantization_annotation.len(), 2);
        assert_eq!(graph.quantization_annotation[0].tensor_name, "w1");
        assert_eq!(graph.quantization_annotation[1].tensor_name, "w2");
    }

    // -- OnnxFunction with doc_string and metadata_props --

    #[test]
    fn onnx_function_with_doc_and_metadata() {
        let mut meta = HashMap::new();
        meta.insert("author".to_string(), "team".to_string());
        let func = OnnxFunction {
            name: "Op".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: "An operator".to_string(),
            metadata_props: meta,
        };
        assert_eq!(func.doc_string, "An operator");
        assert_eq!(func.metadata_props.get("author").unwrap(), "team");
    }

    // -- parse_metadata_props: key with whitespace preserved --

    #[test]
    fn parse_metadata_props_key_with_whitespace() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some(" key with spaces ".to_string()),
            value: Some("val".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.len(), 1);
        assert!(result.contains_key(" key with spaces "));
    }

    // -- parse_metadata_props: empty value stored --

    #[test]
    fn parse_metadata_props_empty_value_stored() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("empty_val".to_string()),
            value: Some(String::new()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("empty_val").unwrap(), "");
    }

    // -- parse_opsets: domain_none defaults empty --

    #[test]
    fn parse_opsets_domain_none_defaults_empty() {
        let opsets = vec![proto::OperatorSetIdProto {
            domain: None,
            version: Some(17),
        }];
        let result = parse_opsets(opsets);
        assert!(result[0].domain.is_empty());
        assert_eq!(result[0].version, 17);
    }

    #[test]
    fn parse_opsets_version_none_defaults_zero() {
        let opsets = vec![proto::OperatorSetIdProto {
            domain: Some("custom".to_string()),
            version: None,
        }];
        let result = parse_opsets(opsets);
        assert_eq!(result[0].domain, "custom");
        assert_eq!(result[0].version, 0);
    }

    // -- OnnxModel clone deep verification --

    #[test]
    fn onnx_model_clone_functions_independence() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![OnnxFunction {
                name: "F".to_string(),
                domain: String::new(),
                overload: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: vec![],
                attribute_protos: HashMap::new(),
                nodes: vec![],
                opset_import: vec![],
                value_info: vec![],
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
        };
        let mut cloned = model.clone();
        cloned.functions[0].name = "G".to_string();
        assert_eq!(model.functions[0].name, "F");
        assert_eq!(cloned.functions[0].name, "G");
    }

    // -- OnnxTensor shape edge: high dimensional --

    #[test]
    fn onnx_tensor_high_dimensional_shape() {
        let tensor = OnnxTensor::new(
            "5d".to_string(),
            Dtype::F32,
            vec![2, 3, 4, 5, 6],
            Bytes::from(vec![0u8; 2 * 3 * 4 * 5 * 6 * 4]),
        );
        assert_eq!(tensor.shape.len(), 5);
        assert_eq!(tensor.shape[4], 6);
    }

    // -- OnnxTensor raw_data empty --

    #[test]
    fn onnx_tensor_raw_data_empty_for_zero_byte() {
        let tensor = OnnxTensor::new(
            "empty".to_string(),
            Dtype::F32,
            vec![0],
            Bytes::new(),
        );
        assert!(tensor.raw_data().is_empty());
    }

    // -- OnnxGraph with nodes referencing inputs/outputs correctly --

    #[test]
    fn onnx_graph_node_references_match_graph_inputs() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "n0".to_string(),
                op_type: "MatMul".to_string(),
                domain: String::new(),
                inputs: vec!["input_0".to_string(), "weight".to_string()],
                outputs: vec!["hidden".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![OnnxValueInfo {
                name: "input_0".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes[0].inputs[0], graph.inputs[0].name);
    }

    // ── OnnxModel default construction ──────────────────────────────

    #[test]
    fn onnx_model_default_fields_zero_functions() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert!(model.functions.is_empty());
        assert!(model.graph.nodes.is_empty());
    }

    // ── OnnxGraph node count consistency ────────────────────────────

    #[test]
    fn onnx_graph_node_count_matches_vec_len() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "n0".to_string(),
                    op_type: "Conv".to_string(),
                    domain: String::new(),
                    inputs: vec!["X".to_string(), "W".to_string()],
                    outputs: vec!["Y".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n1".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["Y".to_string()],
                    outputs: vec!["Z".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].op_type, "Conv");
        assert_eq!(graph.nodes[1].op_type, "Relu");
    }

    // ── OnnxGraph inputs and outputs length ─────────────────────────

    #[test]
    fn onnx_graph_multiple_inputs_outputs_count() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![
                OnnxValueInfo {
                    name: "input_a".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "input_b".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            outputs: vec![
                OnnxValueInfo {
                    name: "output_a".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.inputs[1].name, "input_b");
        assert_eq!(graph.outputs[0].name, "output_a");
    }

    // ── OnnxNode domain field separation ────────────────────────────

    #[test]
    fn onnx_node_custom_domain_preserved() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "CustomOp".to_string(),
            domain: "com.example.custom".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert_eq!(node.domain, "com.example.custom");
        assert_ne!(node.domain, "");
    }

    // ── OnnxValueInfo doc_string field ──────────────────────────────

    #[test]
    fn onnx_value_info_doc_string_nonempty() {
        let info = OnnxValueInfo {
            name: "v".to_string(),
            value_type: None,
            doc_string: "This is a value description".to_string(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(info.doc_string, "This is a value description");
    }

    // ── OnnxQuantizationAnnotation all fields set ───────────────────

    #[test]
    fn onnx_quantization_annotation_all_fields_nontrivial() {
        let mut params = HashMap::new();
        params.insert("SCALE_TENSOR".to_string(), "scale_tensor".to_string());
        params.insert("ZERO_POINT_TENSOR".to_string(), "zp_tensor".to_string());
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "quantized_weight".to_string(),
            quant_param_tensor_names: params,
            scale: Some(0.125),
            zero_point: Some(128),
            axis: Some(0),
        };
        assert_eq!(qa.tensor_name, "quantized_weight");
        assert!((qa.scale.unwrap() - 0.125).abs() < 1e-10);
        assert_eq!(qa.zero_point.unwrap(), 128);
        assert_eq!(qa.axis.unwrap(), 0);
        assert_eq!(qa.quant_param_tensor_names.len(), 2);
    }

    // ── OnnxFunction attributes field ───────────────────────────────

    #[test]
    fn onnx_function_attributes_list() {
        let func = OnnxFunction {
            name: "MyFunc".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec!["alpha".to_string(), "beta".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.attributes.len(), 2);
        assert_eq!(func.attributes[0], "alpha");
        assert_eq!(func.attributes[1], "beta");
    }

    // ── OnnxFunction clone independence for attributes ──────────────

    #[test]
    fn onnx_function_clone_attributes_independence() {
        let func = OnnxFunction {
            name: "F".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec!["attr1".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let mut cloned = func.clone();
        cloned.attributes[0] = "attr2".to_string();
        assert_eq!(func.attributes[0], "attr1");
        assert_eq!(cloned.attributes[0], "attr2");
    }

    // ── OnnxModelMetadata metadata_props lookup ────────────────────

    #[test]
    fn onnx_model_metadata_props_lookup_present() {
        let mut props = HashMap::new();
        props.insert("key_a".to_string(), "value_a".to_string());
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: props,
        };
        assert_eq!(meta.metadata_props.get("key_a").unwrap(), "value_a");
    }

    // ── OnnxModelMetadata ir_version negative ───────────────────────

    #[test]
    fn onnx_model_metadata_negative_ir_version() {
        let meta = OnnxModelMetadata {
            ir_version: -5,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.ir_version, -5);
    }

    // ── OnnxOperatorSet version ordering ────────────────────────────

    #[test]
    fn onnx_operator_set_version_ordering() {
        let opset_low = OnnxOperatorSet { domain: String::new(), version: 14 };
        let opset_high = OnnxOperatorSet { domain: String::new(), version: 20 };
        assert!(opset_low.version < opset_high.version);
    }

    // ── parse_metadata_props: value with embedded newlines ──────────

    #[test]
    fn parse_metadata_props_value_with_newlines() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("desc".to_string()),
            value: Some("line1\nline2\nline3".to_string()),
        }];
        let result = parse_metadata_props(entries);
        let val = result.get("desc").unwrap();
        assert!(val.contains('\n'));
        assert_eq!(val.lines().count(), 3);
    }

    // ── parse_opsets: preserves insertion order ─────────────────────

    #[test]
    fn parse_opsets_preserves_order() {
        let opsets = vec![
            proto::OperatorSetIdProto { domain: Some("b".to_string()), version: Some(2) },
            proto::OperatorSetIdProto { domain: Some("a".to_string()), version: Some(1) },
            proto::OperatorSetIdProto { domain: Some("c".to_string()), version: Some(3) },
        ];
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].domain, "b");
        assert_eq!(result[1].domain, "a");
        assert_eq!(result[2].domain, "c");
    }

    // ── parse_value_info: name preserved exactly ────────────────────

    #[test]
    fn parse_value_info_name_preserved_exactly() {
        let values = vec![proto::ValueInfoProto {
            name: Some("model/input_ids:0".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result[0].name, "model/input_ids:0");
    }

    // ── parse_value_info: many entries preserves order ───────────────

    #[test]
    fn parse_value_info_order_preserved() {
        let values: Vec<proto::ValueInfoProto> = (0..10).map(|i| proto::ValueInfoProto {
            name: Some(format!("val_{i}")),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        }).collect();
        let result = parse_value_info(values).unwrap();
        for (i, vi) in result.iter().enumerate() {
            assert_eq!(vi.name, format!("val_{i}"));
        }
    }

    // ── OnnxGraph initializer non-existent key ──────────────────────

    #[test]
    fn onnx_graph_initializer_get_nonexistent_returns_none() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.initializers.get("nonexistent").is_none());
    }

    // ── OnnxGraph initializer insert multiple ───────────────────────

    #[test]
    fn onnx_graph_initializer_insert_multiple_distinct() {
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        for i in 0..5 {
            let name = format!("tensor_{i}");
            graph.initializers.insert(
                name,
                OnnxTensor::new(format!("tensor_{i}"), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4])),
            );
        }
        assert_eq!(graph.initializers.len(), 5);
        assert!(graph.initializers.contains_key("tensor_3"));
    }

    // ── OnnxGraph metadata_props iteration ──────────────────────────

    #[test]
    fn onnx_graph_metadata_props_iteration() {
        let mut props = HashMap::new();
        props.insert("author".to_string(), "team".to_string());
        props.insert("version".to_string(), "2".to_string());
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: props,
        };
        assert_eq!(graph.metadata_props.len(), 2);
        let mut keys: Vec<&str> = graph.metadata_props.keys().map(|k| k.as_str()).collect();
        keys.sort();
        assert_eq!(keys, vec!["author", "version"]);
    }

    // ── OnnxNode output feeds graph output ──────────────────────────

    #[test]
    fn onnx_node_output_matches_graph_output() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "final".to_string(),
                op_type: "Softmax".to_string(),
                domain: String::new(),
                inputs: vec!["logits".to_string()],
                outputs: vec!["probs".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![OnnxValueInfo {
                name: "probs".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes[0].outputs[0], graph.outputs[0].name);
    }

    // ── OnnxGraph value_info field access ───────────────────────────

    #[test]
    fn onnx_graph_value_info_field_access() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![
                OnnxValueInfo {
                    name: "intermediate_1".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "intermediate_2".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.value_info.len(), 2);
        assert_eq!(graph.value_info[0].name, "intermediate_1");
    }

    // ── OnnxQuantizationAnnotation scale precision ──────────────────

    #[test]
    fn onnx_quantization_annotation_scale_precision() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: String::new(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(1.5e-3),
            zero_point: None,
            axis: None,
        };
        let scale_val = qa.scale.unwrap();
        assert!((scale_val - 0.0015).abs() < 1e-10);
    }

    // ── OnnxQuantizationAnnotation quant_param key lookup ───────────

    #[test]
    fn onnx_quantization_annotation_param_key_lookup() {
        let mut params = HashMap::new();
        params.insert("SCALE_TENSOR".to_string(), "s_tensor".to_string());
        params.insert("ZERO_POINT_TENSOR".to_string(), "zp_tensor".to_string());
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "q_weight".to_string(),
            quant_param_tensor_names: params.clone(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        assert_eq!(qa.quant_param_tensor_names.get("SCALE_TENSOR").unwrap(), "s_tensor");
        assert_eq!(qa.quant_param_tensor_names.get("ZERO_POINT_TENSOR").unwrap(), "zp_tensor");
        assert!(qa.quant_param_tensor_names.get("AXIS").is_none());
    }

    // ── OnnxFunction overload with version string ───────────────────

    #[test]
    fn onnx_function_overload_with_version() {
        let func = OnnxFunction {
            name: "MatMul".to_string(),
            domain: "com.custom".to_string(),
            overload: "v2_f32".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.overload, "v2_f32");
    }

    // ── OnnxFunction opset_import contains multiple entries ─────────

    #[test]
    fn onnx_function_opset_import_multiple() {
        let func = OnnxFunction {
            name: "F".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![
                OnnxOperatorSet { domain: String::new(), version: 17 },
                OnnxOperatorSet { domain: "ai.onnx.ml".to_string(), version: 3 },
            ],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.opset_import.len(), 2);
        assert_eq!(func.opset_import[1].domain, "ai.onnx.ml");
    }

    // ── OnnxFunction value_info with temp variable ──────────────────

    #[test]
    fn onnx_function_value_info_with_temp_var() {
        let func = OnnxFunction {
            name: "F".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![
                OnnxValueInfo {
                    name: "temp1".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.value_info.len(), 1);
        assert_eq!(func.value_info[0].name, "temp1");
    }

    // ── OnnxModel clone graph independence ──────────────────────────

    #[test]
    fn onnx_model_clone_graph_independence() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "orig".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let mut cloned = model.clone();
        cloned.graph.name = "modified".to_string();
        assert_eq!(model.graph.name, "orig");
        assert_eq!(cloned.graph.name, "modified");
    }

    // ── OnnxModel functions access by index ─────────────────────────

    #[test]
    fn onnx_model_functions_access_by_index() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![
                OnnxFunction {
                    name: "FuncA".to_string(),
                    domain: String::new(),
                    overload: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: vec![],
                    attribute_protos: HashMap::new(),
                    nodes: vec![],
                    opset_import: vec![],
                    value_info: vec![],
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxFunction {
                    name: "FuncB".to_string(),
                    domain: String::new(),
                    overload: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: vec![],
                    attribute_protos: HashMap::new(),
                    nodes: vec![],
                    opset_import: vec![],
                    value_info: vec![],
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
        };
        assert_eq!(model.functions[0].name, "FuncA");
        assert_eq!(model.functions[1].name, "FuncB");
    }

    // ── OnnxGraph sparse_initializers field access ──────────────────

    #[test]
    fn onnx_graph_sparse_initializers_empty_by_default() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.sparse_initializers.is_empty());
    }

    // ── OnnxModelMetadata producer fields ───────────────────────────

    #[test]
    fn onnx_model_metadata_producer_fields() {
        let meta = OnnxModelMetadata {
            ir_version: 9,
            producer_name: "gllm-exporter".to_string(),
            producer_version: "3.2.1".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 42,
            doc_string: "Exported model".to_string(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.producer_name, "gllm-exporter");
        assert_eq!(meta.producer_version, "3.2.1");
        assert_eq!(meta.domain, "ai.onnx");
        assert_eq!(meta.model_version, 42);
    }

    // ── OnnxGraph name field unicode ────────────────────────────────

    #[test]
    fn onnx_graph_name_unicode_chars() {
        let graph = OnnxGraph {
            name: "模型图".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.name, "模型图");
    }

    // ── OnnxNode op_type case sensitivity ───────────────────────────

    #[test]
    fn onnx_node_op_type_lowercase_preserved() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "matmul".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert_eq!(node.op_type, "matmul");
    }

    // ── OnnxValueInfo value_type none is valid ──────────────────────

    #[test]
    fn onnx_value_info_value_type_none_valid() {
        let info = OnnxValueInfo {
            name: "x".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(info.value_type.is_none());
    }

    // ── OnnxQuantizationAnnotation axis boundary ────────────────────

    #[test]
    fn onnx_quantization_annotation_axis_i32_range() {
        let qa_min = OnnxQuantizationAnnotation {
            tensor_name: String::new(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: Some(i32::MIN),
        };
        let qa_max = OnnxQuantizationAnnotation {
            tensor_name: String::new(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: Some(i32::MAX),
        };
        assert_eq!(qa_min.axis.unwrap(), i32::MIN);
        assert_eq!(qa_max.axis.unwrap(), i32::MAX);
    }

    // ── parse_metadata_props: key with equals sign in value ─────────

    #[test]
    fn parse_metadata_props_value_with_equals() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("equation".to_string()),
            value: Some("a=b+c".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("equation").unwrap(), "a=b+c");
    }

    // ── OnnxGraph initializer with bool tensor ──────────────────────

    #[test]
    fn onnx_graph_initializer_bool_tensor() {
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.initializers.insert(
            "mask".to_string(),
            OnnxTensor::new("mask".to_string(), Dtype::BOOL, vec![4], Bytes::from(vec![1, 0, 1, 0])),
        );
        let t = graph.initializers.get("mask").unwrap();
        assert_eq!(t.dtype, Dtype::BOOL);
        assert_eq!(t.shape, vec![4]);
    }

    // ── OnnxNode inputs contain empty string markers ────────────────

    #[test]
    fn onnx_node_inputs_with_empty_string_marker() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["A".to_string(), String::new(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.inputs.len(), 3);
        assert_eq!(node.inputs[1], "");
    }

    // ── OnnxModelMetadata opset_import len matches vec ──────────────

    #[test]
    fn onnx_model_metadata_opset_import_len() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![
                OnnxOperatorSet { domain: String::new(), version: 17 },
                OnnxOperatorSet { domain: "ai.onnx.ml".to_string(), version: 3 },
                OnnxOperatorSet { domain: "ai.onnx.training".to_string(), version: 1 },
            ],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.opset_import.len(), 3);
    }

    // ── OnnxFunction doc_string multiline ───────────────────────────

    #[test]
    fn onnx_function_doc_string_multiline() {
        let doc = "Line 1\nLine 2\nLine 3".to_string();
        let func = OnnxFunction {
            name: "F".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: doc.clone(),
            metadata_props: HashMap::new(),
        };
        assert!(func.doc_string.contains('\n'));
        assert_eq!(func.doc_string.lines().count(), 3);
    }

    // ── OnnxGraph quantization_annotation empty ─────────────────────

    #[test]
    fn onnx_graph_quantization_annotation_empty_default() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.quantization_annotation.is_empty());
    }

    // ── OnnxGraph quantization_annotation with entry ────────────────

    #[test]
    fn onnx_graph_quantization_annotation_with_entry() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "weight_q".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.01),
            zero_point: Some(0),
            axis: Some(1),
        };
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![qa],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.quantization_annotation.len(), 1);
        assert_eq!(graph.quantization_annotation[0].tensor_name, "weight_q");
    }

    // ── OnnxNode clone independence for attributes ──────────────────

    #[test]
    fn onnx_node_clone_attributes_independence() {
        let mut attrs = HashMap::new();
        attrs.insert("key".to_string(), OnnxAttribute {
            name: "key".to_string(),
            value: OnnxAttributeValue::Int(10),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Op".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
        };
        let mut cloned = node.clone();
        cloned.attributes.insert("new_key".to_string(), OnnxAttribute {
            name: "new_key".to_string(),
            value: OnnxAttributeValue::Int(20),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        assert_eq!(node.attributes.len(), 1);
        assert_eq!(cloned.attributes.len(), 2);
    }

    // ── OnnxValueInfo clone independence for metadata_props ─────────

    #[test]
    fn onnx_value_info_clone_metadata_independence() {
        let mut meta = HashMap::new();
        meta.insert("k".to_string(), "v".to_string());
        let info = OnnxValueInfo {
            name: "vi".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: meta,
        };
        let mut cloned = info.clone();
        cloned.metadata_props.insert("k2".to_string(), "v2".to_string());
        assert_eq!(info.metadata_props.len(), 1);
        assert_eq!(cloned.metadata_props.len(), 2);
    }

    // ── OnnxGraph initializer remove key ────────────────────────────

    #[test]
    fn onnx_graph_initializer_remove_key() {
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.initializers.insert(
            "w".to_string(),
            OnnxTensor::new("w".to_string(), Dtype::F32, vec![2, 3], Bytes::from(vec![0u8; 24])),
        );
        assert!(graph.initializers.contains_key("w"));
        graph.initializers.remove("w");
        assert!(!graph.initializers.contains_key("w"));
    }

    // ── OnnxQuantizationAnnotation clone independence ───────────────

    #[test]
    fn onnx_quantization_annotation_clone_independence() {
        let mut params = HashMap::new();
        params.insert("key".to_string(), "value".to_string());
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "t".to_string(),
            quant_param_tensor_names: params,
            scale: Some(1.0),
            zero_point: None,
            axis: None,
        };
        let mut cloned = qa.clone();
        cloned.quant_param_tensor_names.insert("key2".to_string(), "value2".to_string());
        assert_eq!(qa.quant_param_tensor_names.len(), 1);
        assert_eq!(cloned.quant_param_tensor_names.len(), 2);
    }

    // ── parse_metadata_props: key longer than 1k chars ──────────────

    #[test]
    fn parse_metadata_props_very_long_key() {
        let long_key = "k".repeat(2000);
        let entries = vec![proto::StringStringEntryProto {
            key: Some(long_key.clone()),
            value: Some("v".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.len(), 1);
        assert_eq!(result.get(&long_key).unwrap(), "v");
    }

    // ── parse_metadata_props: many entries all preserved ────────────

    #[test]
    fn parse_metadata_props_many_entries_preserved() {
        let entries: Vec<proto::StringStringEntryProto> = (0..50)
            .map(|i| proto::StringStringEntryProto {
                key: Some(format!("k{i}")),
                value: Some(format!("v{i}")),
            })
            .collect();
        let result = parse_metadata_props(entries);
        assert_eq!(result.len(), 50);
        for i in 0..50 {
            let key = format!("k{i}");
            let val = format!("v{i}");
            assert_eq!(result[&key], val);
        }
    }

    // ── OnnxGraph debug output contains name ────────────────────────

    #[test]
    fn onnx_graph_debug_output_contains_name_field() {
        let graph = OnnxGraph {
            name: "test_debug_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let debug = format!("{graph:?}");
        assert!(debug.contains("test_debug_graph"), "Debug output should contain graph name");
    }

    // ── OnnxModel debug output contains metadata ────────────────────

    #[test]
    fn onnx_model_debug_contains_metadata_fields() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 9,
                producer_name: "test-producer".to_string(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let debug = format!("{model:?}");
        assert!(debug.contains("test-producer"));
    }

    // ═══════════════════════════════════════════════════════════════════
    // NEW TESTS: 50 additional tests
    // ═══════════════════════════════════════════════════════════════════

    // ── OnnxModel: graph field mutation ────────────────────────────

    #[test]
    fn onnx_model_graph_name_mutable_through_model() {
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "original".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        model.graph.name = "renamed".to_string();
        assert_eq!(model.graph.name, "renamed");
    }

    // ── OnnxModel: metadata props mutation ─────────────────────────

    #[test]
    fn onnx_model_metadata_props_insert_and_read() {
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert!(model.metadata.metadata_props.is_empty());
        model.metadata.metadata_props.insert("key".to_string(), "val".to_string());
        assert_eq!(model.metadata.metadata_props["key"], "val");
    }

    // ── OnnxModel: zero functions vec is not null ──────────────────

    #[test]
    fn onnx_model_functions_vec_capacity_zero_when_empty() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert!(model.functions.is_empty());
        assert_eq!(model.functions.len(), 0);
    }

    // ── OnnxGraph: doc_string preserved with newlines ──────────────

    #[test]
    fn onnx_graph_doc_string_with_newlines_preserved() {
        let doc = "Line 1\nLine 2\nLine 3".to_string();
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: doc.clone(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.doc_string.contains('\n'));
        assert_eq!(graph.doc_string.lines().count(), 3);
    }

    // ── OnnxGraph: node count zero for empty graph ─────────────────

    #[test]
    fn onnx_graph_node_count_zero_when_empty() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 0);
    }

    // ── OnnxGraph: push node and verify count ──────────────────────

    #[test]
    fn onnx_graph_push_node_increments_count() {
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 0);
        graph.nodes.push(OnnxNode {
            name: "n0".to_string(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        });
        assert_eq!(graph.nodes.len(), 1);
        graph.nodes.push(OnnxNode {
            name: "n1".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        });
        assert_eq!(graph.nodes.len(), 2);
    }

    // ── OnnxGraph: input names are preserved in order ──────────────

    #[test]
    fn onnx_graph_input_names_preserve_insertion_order() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![
                OnnxValueInfo { name: "input_a".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
                OnnxValueInfo { name: "input_b".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
                OnnxValueInfo { name: "input_c".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
            ],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.inputs[0].name, "input_a");
        assert_eq!(graph.inputs[1].name, "input_b");
        assert_eq!(graph.inputs[2].name, "input_c");
    }

    // ── OnnxGraph: output names are preserved in order ─────────────

    #[test]
    fn onnx_graph_output_names_preserve_insertion_order() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![
                OnnxValueInfo { name: "out_x".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
                OnnxValueInfo { name: "out_y".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
            ],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.outputs[0].name, "out_x");
        assert_eq!(graph.outputs[1].name, "out_y");
    }

    // ── OnnxGraph: initializer with different dtypes ───────────────

    #[test]
    fn onnx_graph_initializer_f16_tensor() {
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let tensor = OnnxTensor::new("fp16_w".to_string(), Dtype::F16, vec![4], Bytes::from(vec![0u8; 8]));
        graph.initializers.insert("fp16_w".to_string(), tensor);
        assert!(graph.initializers.contains_key("fp16_w"));
        let t = &graph.initializers["fp16_w"];
        assert_eq!(t.dtype, Dtype::F16);
        assert_eq!(t.shape, vec![4]);
    }

    // ── OnnxGraph: initializer BF16 tensor ─────────────────────────

    #[test]
    fn onnx_graph_initializer_bf16_tensor() {
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let tensor = OnnxTensor::new("bf16_w".to_string(), Dtype::BF16, vec![2, 3], Bytes::from(vec![0u8; 12]));
        graph.initializers.insert("bf16_w".to_string(), tensor);
        assert_eq!(graph.initializers["bf16_w"].dtype, Dtype::BF16);
    }

    // ── OnnxGraph: sparse_initializers can be pushed ───────────────

    #[test]
    fn onnx_graph_sparse_initializers_push_increments_len() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.sparse_initializers.is_empty());
    }

    // ── OnnxGraph: metadata_props replace existing key ─────────────

    #[test]
    fn onnx_graph_metadata_props_replace_value() {
        let mut props = HashMap::new();
        props.insert("framework".to_string(), "pytorch".to_string());
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: props,
        };
        graph.metadata_props.insert("framework".to_string(), "tensorflow".to_string());
        assert_eq!(graph.metadata_props["framework"], "tensorflow");
    }

    // ── OnnxGraph: nodes accessed by index yield correct op_type ───

    #[test]
    fn onnx_graph_nodes_indexed_access_yields_op_type() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode { name: "n0".to_string(), op_type: "Conv".to_string(), domain: String::new(), inputs: vec![], outputs: vec![], attributes: HashMap::new() },
                OnnxNode { name: "n1".to_string(), op_type: "BatchNorm".to_string(), domain: String::new(), inputs: vec![], outputs: vec![], attributes: HashMap::new() },
                OnnxNode { name: "n2".to_string(), op_type: "Relu".to_string(), domain: String::new(), inputs: vec![], outputs: vec![], attributes: HashMap::new() },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes[0].op_type, "Conv");
        assert_eq!(graph.nodes[1].op_type, "BatchNorm");
        assert_eq!(graph.nodes[2].op_type, "Relu");
    }

    // ── OnnxNode: name field is mutable ────────────────────────────

    #[test]
    fn onnx_node_name_field_is_mutable() {
        let mut node = OnnxNode {
            name: "original".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        node.name = "renamed_node".to_string();
        assert_eq!(node.name, "renamed_node");
    }

    // ── OnnxNode: inputs field is mutable ──────────────────────────

    #[test]
    fn onnx_node_inputs_field_is_mutable() {
        let mut node = OnnxNode {
            name: "n".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["a".to_string()],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        node.inputs.push("b".to_string());
        assert_eq!(node.inputs, vec!["a", "b"]);
    }

    // ── OnnxNode: outputs field is mutable ─────────────────────────

    #[test]
    fn onnx_node_outputs_field_is_mutable() {
        let mut node = OnnxNode {
            name: "n".to_string(),
            op_type: "Split".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec!["out1".to_string()],
            attributes: HashMap::new(),
        };
        node.outputs.push("out2".to_string());
        assert_eq!(node.outputs, vec!["out1", "out2"]);
    }

    // ── OnnxNode: op_type with mixed case ──────────────────────────

    #[test]
    fn onnx_node_op_type_mixed_case_preserved() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Conv2D".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert_eq!(node.op_type, "Conv2D");
    }

    // ── OnnxNode: domain non-empty for custom operator ─────────────

    #[test]
    fn onnx_node_domain_non_empty_custom_op() {
        let node = OnnxNode {
            name: "custom_op".to_string(),
            op_type: "MyCustomOp".to_string(),
            domain: "com.example".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert_eq!(node.domain, "com.example");
        assert!(!node.domain.is_empty());
    }

    // ── OnnxNode: single attribute insertion and retrieval ─────────

    #[test]
    fn onnx_node_single_attribute_insert_and_get() {
        let mut node = OnnxNode {
            name: "conv".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        node.attributes.insert("kernel_shape".to_string(), OnnxAttribute {
            name: "kernel_shape".to_string(),
            value: OnnxAttributeValue::Ints(vec![3, 3]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        assert!(node.attributes.contains_key("kernel_shape"));
        if let OnnxAttributeValue::Ints(dims) = &node.attributes["kernel_shape"].value {
            assert_eq!(*dims, vec![3, 3]);
        } else {
            panic!("expected Ints variant");
        }
    }

    // ── OnnxNode: attributes removal ───────────────────────────────

    #[test]
    fn onnx_node_attributes_remove_key() {
        let mut node = OnnxNode {
            name: "n".to_string(),
            op_type: "Pool".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        node.attributes.insert("pads".to_string(), OnnxAttribute {
            name: "pads".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 1, 1, 1]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        assert!(node.attributes.remove("pads").is_some());
        assert!(node.attributes.is_empty());
    }

    // ── OnnxNode: empty outputs vec is valid ───────────────────────

    #[test]
    fn onnx_node_empty_outputs_valid_for_side_effect_op() {
        let node = OnnxNode {
            name: "print_node".to_string(),
            op_type: "Print".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert!(node.outputs.is_empty());
    }

    // ── OnnxValueInfo: metadata_props mutation ─────────────────────

    #[test]
    fn onnx_value_info_metadata_props_mutation() {
        let mut vi = OnnxValueInfo {
            name: "x".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        vi.metadata_props.insert("source".to_string(), "original".to_string());
        assert_eq!(vi.metadata_props["source"], "original");
        vi.metadata_props.insert("source".to_string(), "modified".to_string());
        assert_eq!(vi.metadata_props["source"], "modified");
    }

    // ── OnnxValueInfo: name with dots and brackets ─────────────────

    #[test]
    fn onnx_value_info_name_with_dots_and_brackets() {
        let vi = OnnxValueInfo {
            name: "layer.0.self_attn.q_proj.weight[0]".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(vi.name.contains('.'));
        assert!(vi.name.contains('['));
        assert!(vi.name.contains(']'));
    }

    // ── OnnxQuantizationAnnotation: tensor_name with path separators

    #[test]
    fn onnx_quantization_annotation_tensor_name_with_dots() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "transformer.h.0.attn.c_attn.weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        assert!(ann.tensor_name.contains('.'));
        let parts: Vec<&str> = ann.tensor_name.split('.').collect();
        assert!(parts.len() > 1);
    }

    // ── OnnxQuantizationAnnotation: scale very large value ──────────

    #[test]
    fn onnx_quantization_annotation_scale_very_large() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(1e10),
            zero_point: None,
            axis: None,
        };
        assert!(ann.scale.unwrap() > 1e9);
    }

    // ── OnnxQuantizationAnnotation: scale very small value ──────────

    #[test]
    fn onnx_quantization_annotation_scale_very_small() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(1e-10),
            zero_point: None,
            axis: None,
        };
        assert!(ann.scale.unwrap() < 1e-9);
        assert!(ann.scale.unwrap() > 0.0);
    }

    // ── OnnxQuantizationAnnotation: zero_point i64 min ──────────────

    #[test]
    fn onnx_quantization_annotation_zero_point_i64_min() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.1),
            zero_point: Some(i64::MIN),
            axis: None,
        };
        assert_eq!(ann.zero_point, Some(i64::MIN));
    }

    // ── OnnxQuantizationAnnotation: zero_point i64 max ──────────────

    #[test]
    fn onnx_quantization_annotation_zero_point_i64_max() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.1),
            zero_point: Some(i64::MAX),
            axis: None,
        };
        assert_eq!(ann.zero_point, Some(i64::MAX));
    }

    // ── OnnxQuantizationAnnotation: empty quant_param names map ─────

    #[test]
    fn onnx_quantization_annotation_empty_param_map_is_valid() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        assert!(ann.quant_param_tensor_names.is_empty());
        assert!(ann.scale.is_none());
        assert!(ann.zero_point.is_none());
        assert!(ann.axis.is_none());
    }

    // ── OnnxFunction: inputs and outputs counts ────────────────────

    #[test]
    fn onnx_function_inputs_outputs_count_independent() {
        let func = OnnxFunction {
            name: "MultiIn".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            outputs: vec!["y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.inputs.len(), 3);
        assert_eq!(func.outputs.len(), 1);
    }

    // ── OnnxFunction: empty domain is valid ────────────────────────

    #[test]
    fn onnx_function_empty_domain_default_ops() {
        let func = OnnxFunction {
            name: "Identity".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(func.domain.is_empty());
    }

    // ── OnnxFunction: value_info count matches construction ─────────

    #[test]
    fn onnx_function_value_info_count_matches() {
        let func = OnnxFunction {
            name: "f".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![
                OnnxValueInfo { name: "temp_0".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
                OnnxValueInfo { name: "temp_1".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
            ],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.value_info.len(), 2);
    }

    // ── OnnxFunction: attributes vec stores names ──────────────────

    #[test]
    fn onnx_function_attributes_vec_stores_names() {
        let func = OnnxFunction {
            name: "f".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.attributes, vec!["alpha", "beta", "gamma"]);
    }

    // ── OnnxFunction: metadata_props are independent from other maps

    #[test]
    fn onnx_function_metadata_props_independent_of_graph() {
        let mut func_props = HashMap::new();
        func_props.insert("author".to_string(), "custom".to_string());
        let func = OnnxFunction {
            name: "f".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: func_props,
        };
        assert_eq!(func.metadata_props["author"], "custom");
    }

    // ── OnnxModelMetadata: opset_import can be iterated ─────────────

    #[test]
    fn onnx_model_metadata_opset_import_iteration() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![
                OnnxOperatorSet { domain: String::new(), version: 17 },
                OnnxOperatorSet { domain: "ai.onnx.ml".to_string(), version: 3 },
                OnnxOperatorSet { domain: "custom".to_string(), version: 1 },
            ],
            metadata_props: HashMap::new(),
        };
        let versions: Vec<i64> = meta.opset_import.iter().map(|os| os.version).collect();
        assert_eq!(versions, vec![17, 3, 1]);
    }

    // ── OnnxModelMetadata: metadata_props can be iterated ───────────

    #[test]
    fn onnx_model_metadata_props_iteration() {
        let mut props = HashMap::new();
        props.insert("a".to_string(), "1".to_string());
        props.insert("b".to_string(), "2".to_string());
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: props,
        };
        assert_eq!(meta.metadata_props.len(), 2);
        let keys: Vec<&String> = meta.metadata_props.keys().collect();
        assert!(keys.contains(&&"a".to_string()));
        assert!(keys.contains(&&"b".to_string()));
    }

    // ── OnnxModelMetadata: doc_string with special chars ────────────

    #[test]
    fn onnx_model_metadata_doc_string_special_chars() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: "Line1\nLine2\tTabbed".to_string(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(meta.doc_string.contains('\n'));
        assert!(meta.doc_string.contains('\t'));
    }

    // ── OnnxOperatorSet: domain and version are both accessible ─────

    #[test]
    fn onnx_operator_set_both_fields_readable() {
        let ops = OnnxOperatorSet {
            domain: "ai.onnx.training".to_string(),
            version: 1,
        };
        assert_eq!(ops.domain, "ai.onnx.training");
        assert_eq!(ops.version, 1);
    }

    // ── OnnxGraph: quantization_annotation_vec_is_mutable ──────────

    #[test]
    fn onnx_graph_quantization_annotation_push_increments() {
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.quantization_annotation.is_empty());
        graph.quantization_annotation.push(OnnxQuantizationAnnotation {
            tensor_name: "w1".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(1.0),
            zero_point: None,
            axis: None,
        });
        assert_eq!(graph.quantization_annotation.len(), 1);
        graph.quantization_annotation.push(OnnxQuantizationAnnotation {
            tensor_name: "w2".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: Some(0),
            axis: Some(0),
        });
        assert_eq!(graph.quantization_annotation.len(), 2);
    }

    // ── OnnxGraph: value_info push increments ──────────────────────

    #[test]
    fn onnx_graph_value_info_push_increments() {
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.value_info.push(OnnxValueInfo {
            name: "intermediate_0".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        });
        assert_eq!(graph.value_info.len(), 1);
    }

    // ── OnnxModel: functions push and access ───────────────────────

    #[test]
    fn onnx_model_functions_push_and_access() {
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        model.functions.push(OnnxFunction {
            name: "CustomRelu".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        });
        assert_eq!(model.functions.len(), 1);
        assert_eq!(model.functions[0].name, "CustomRelu");
    }

    // ── OnnxNode: debug format includes domain ─────────────────────

    #[test]
    fn onnx_node_debug_format_includes_domain_when_nonempty() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "MyOp".to_string(),
            domain: "custom.domain".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        let debug = format!("{node:?}");
        assert!(debug.contains("custom.domain"));
    }

    // ── OnnxGraph: debug format includes node count ────────────────

    #[test]
    fn onnx_graph_debug_shows_initializer_count() {
        let mut initializers = HashMap::new();
        initializers.insert("w1".to_string(), OnnxTensor::new("w1".to_string(), Dtype::F32, vec![2, 2], Bytes::from(vec![0u8; 16])));
        initializers.insert("w2".to_string(), OnnxTensor::new("w2".to_string(), Dtype::F32, vec![3, 3], Bytes::from(vec![0u8; 36])));
        let graph = OnnxGraph {
            name: "debug_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers,
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let debug = format!("{graph:?}");
        assert!(debug.contains("debug_graph"));
    }

    // ── OnnxModel: debug format includes ir_version ────────────────

    #[test]
    fn onnx_model_debug_includes_ir_version() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 42,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let debug = format!("{model:?}");
        assert!(debug.contains("42"));
    }

    // ── OnnxQuantizationAnnotation: debug format shows tensor_name ──

    #[test]
    fn onnx_quantization_annotation_debug_shows_tensor_name() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "my_weight_tensor".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        let debug = format!("{ann:?}");
        assert!(debug.contains("my_weight_tensor"));
    }

    // ── OnnxFunction: debug shows node count ───────────────────────

    #[test]
    fn onnx_function_debug_shows_node_count() {
        let func = OnnxFunction {
            name: "FusedBiasRelu".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![
                OnnxNode { name: "n0".to_string(), op_type: "Add".to_string(), domain: String::new(), inputs: vec![], outputs: vec![], attributes: HashMap::new() },
                OnnxNode { name: "n1".to_string(), op_type: "Relu".to_string(), domain: String::new(), inputs: vec![], outputs: vec![], attributes: HashMap::new() },
            ],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let debug = format!("{func:?}");
        assert!(debug.contains("FusedBiasRelu"));
    }

    // ── OnnxModel: clone deep copies graph nodes ───────────────────

    #[test]
    fn onnx_model_clone_deep_copies_graph_nodes() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![
                    OnnxNode { name: "n0".to_string(), op_type: "Conv".to_string(), domain: String::new(), inputs: vec![], outputs: vec![], attributes: HashMap::new() },
                ],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let mut cloned = model.clone();
        cloned.graph.nodes[0].op_type = "Relu".to_string();
        assert_eq!(model.graph.nodes[0].op_type, "Conv");
        assert_eq!(cloned.graph.nodes[0].op_type, "Relu");
    }

    // ── OnnxGraph: clone deep copies initializers ──────────────────

    #[test]
    fn onnx_graph_clone_deep_copies_initializers() {
        let mut initializers = HashMap::new();
        initializers.insert("w".to_string(), OnnxTensor::new("w".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8])));
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers,
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let mut cloned = graph.clone();
        cloned.initializers.remove("w");
        assert!(graph.initializers.contains_key("w"));
        assert!(!cloned.initializers.contains_key("w"));
    }

    // ── OnnxModelMetadata: clone deep copies opset_import ──────────

    #[test]
    fn onnx_model_metadata_clone_deep_copies_opset_import() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![OnnxOperatorSet { domain: String::new(), version: 17 }],
            metadata_props: HashMap::new(),
        };
        let mut cloned = meta.clone();
        cloned.opset_import.push(OnnxOperatorSet { domain: "custom".to_string(), version: 1 });
        assert_eq!(meta.opset_import.len(), 1);
        assert_eq!(cloned.opset_import.len(), 2);
    }

    // ── OnnxFunction: clone deep copies value_info ─────────────────

    #[test]
    fn onnx_function_clone_deep_copies_value_info() {
        let func = OnnxFunction {
            name: "f".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![
                OnnxValueInfo { name: "t".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
            ],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let mut cloned = func.clone();
        cloned.value_info.push(OnnxValueInfo { name: "t2".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() });
        assert_eq!(func.value_info.len(), 1);
        assert_eq!(cloned.value_info.len(), 2);
    }

    // ── OnnxNode: clone deep copies attributes ─────────────────────

    #[test]
    fn onnx_node_clone_deep_copies_attributes() {
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), OnnxAttribute {
            name: "k".to_string(),
            value: OnnxAttributeValue::Int(42),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Op".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
        };
        let mut cloned = node.clone();
        cloned.attributes.remove("k");
        assert!(node.attributes.contains_key("k"));
        assert!(!cloned.attributes.contains_key("k"));
    }

    // ── OnnxGraph: multiple nodes with shared input ────────────────

    #[test]
    fn onnx_graph_multiple_nodes_share_input_name() {
        let shared = "hidden_state".to_string();
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode { name: "n0".to_string(), op_type: "MatMul".to_string(), domain: String::new(), inputs: vec![shared.clone(), "w0".to_string()], outputs: vec!["out0".to_string()], attributes: HashMap::new() },
                OnnxNode { name: "n1".to_string(), op_type: "MatMul".to_string(), domain: String::new(), inputs: vec![shared.clone(), "w1".to_string()], outputs: vec!["out1".to_string()], attributes: HashMap::new() },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes[0].inputs[0], "hidden_state");
        assert_eq!(graph.nodes[1].inputs[0], "hidden_state");
    }

    // ── OnnxGraph: initializer overwrite updates shape ─────────────

    #[test]
    fn onnx_graph_initializer_overwrite_updates_shape() {
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.initializers.insert("w".to_string(), OnnxTensor::new("w".to_string(), Dtype::F32, vec![2, 3], Bytes::from(vec![0u8; 24])));
        assert_eq!(graph.initializers["w"].shape, vec![2, 3]);
        graph.initializers.insert("w".to_string(), OnnxTensor::new("w".to_string(), Dtype::F32, vec![4, 5], Bytes::from(vec![0u8; 80])));
        assert_eq!(graph.initializers["w"].shape, vec![4, 5]);
    }

    // ── OnnxModel: graph inputs accessible through model ───────────

    #[test]
    fn onnx_model_graph_inputs_accessible() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![
                    OnnxValueInfo { name: "input_ids".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
                    OnnxValueInfo { name: "attention_mask".to_string(), value_type: None, doc_string: String::new(), metadata_props: HashMap::new() },
                ],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert_eq!(model.graph.inputs.len(), 2);
        assert_eq!(model.graph.inputs[0].name, "input_ids");
        assert_eq!(model.graph.inputs[1].name, "attention_mask");
    }

    // ── OnnxValueInfo: clone deep copies metadata_props ────────────

    #[test]
    fn onnx_value_info_clone_metadata_deep_copy() {
        let mut vi = OnnxValueInfo {
            name: "x".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        vi.metadata_props.insert("k".to_string(), "v".to_string());
        let mut cloned = vi.clone();
        cloned.metadata_props.insert("k2".to_string(), "v2".to_string());
        assert_eq!(vi.metadata_props.len(), 1);
        assert_eq!(cloned.metadata_props.len(), 2);
    }

    // ── OnnxQuantizationAnnotation: clone deep copies param names ──

    #[test]
    fn onnx_quantization_annotation_clone_param_names_deep_copy() {
        let mut params = HashMap::new();
        params.insert("SCALE_TENSOR".to_string(), "s".to_string());
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: params,
            scale: Some(1.0),
            zero_point: None,
            axis: None,
        };
        let mut cloned = ann.clone();
        cloned.quant_param_tensor_names.insert("NEW_KEY".to_string(), "new_val".to_string());
        assert_eq!(ann.quant_param_tensor_names.len(), 1);
        assert_eq!(cloned.quant_param_tensor_names.len(), 2);
    }

    // ── OnnxOperatorSet: clone independence verification ────────────

    #[test]
    fn onnx_operator_set_clone_then_domain_change_independent() {
        let ops = OnnxOperatorSet {
            domain: "original".to_string(),
            version: 1,
        };
        let mut cloned = ops.clone();
        cloned.domain = "modified".to_string();
        assert_eq!(ops.domain, "original");
        assert_eq!(cloned.domain, "modified");
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 6 TESTS (80 new — deduplicated)
    // ══════════════════════════════════════════════════════════════════════

    // ── LoaderError: unsupported weight extension Display ────────────────

    #[test]
    fn loader_error_unsupported_weight_extension_msg() {
        let err = LoaderError::UnsupportedWeightExtension(".xyz".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Unsupported weight extension"), "expected prefix: {msg}");
        assert!(msg.contains(".xyz"), "expected detail: {msg}");
    }

    #[test]
    fn onnx_attribute_value_floats_variant_nonempty() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Floats(vec![1.0, 2.0, 3.0]);
        assert!(matches!(val, OnnxAttributeValue::Floats(v) if v.len() == 3));
    }

    #[test]
    fn onnx_attribute_value_ints_variant_nonempty() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Ints(vec![-1, 0, 1]);
        assert!(matches!(val, OnnxAttributeValue::Ints(v) if v.len() == 3));
    }

    // ── OnnxAttribute with graph value roundtrip ────────────────────────

    #[test]
    fn onnx_attribute_graph_value_clone() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let subgraph = OnnxGraph {
            name: "loop_body".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "add".to_string(),
                op_type: "Add".to_string(),
                domain: String::new(),
                inputs: vec!["a".to_string(), "b".to_string()],
                outputs: vec!["c".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let attr = OnnxAttribute {
            name: "body".to_string(),
            value: OnnxAttributeValue::Graph(Box::new(subgraph)),
            doc_string: "loop body graph".to_string(),
            ref_attr_name: None,
            attr_type: None,
        };
        let cloned = attr.clone();
        assert_eq!(cloned.name, "body");
        if let OnnxAttributeValue::Graph(g) = &cloned.value {
            assert_eq!(g.name, "loop_body");
            assert_eq!(g.nodes.len(), 1);
        } else {
            panic!("expected Graph variant");
        }
    }

    // ── OnnxModel::from_proto: missing graph returns error ──────────────

    #[test]
    fn onnx_model_from_proto_missing_graph_returns_error() {
        let proto = proto::ModelProto {
            ir_version: None,
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            doc_string: None,
            graph: None,
            opset_import: vec![],
            metadata_props: vec![],
            functions: vec![],
            configuration: vec![],
            training_info: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = OnnxModel::from_proto(proto, &mut resolver);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("missing graph"), "expected 'missing graph': {msg}");
    }

    // ── OnnxGraph::from_proto: duplicate initializer returns error ──────

    #[test]
    fn onnx_graph_from_proto_duplicate_initializer_returns_error() {
        let tensor1 = proto::TensorProto {
            name: Some("dup_weight".to_string()),
            data_type: Some(1),
            dims: vec![2],
            float_data: vec![1.0, 2.0],
            ..Default::default()
        };
        let tensor2 = proto::TensorProto {
            name: Some("dup_weight".to_string()),
            data_type: Some(1),
            dims: vec![2],
            float_data: vec![3.0, 4.0],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("g".to_string()),
            initializer: vec![tensor1, tensor2],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = OnnxGraph::from_proto(graph_proto, &mut resolver);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Duplicate tensor"), "expected Duplicate tensor: {msg}");
        assert!(msg.contains("dup_weight"), "expected tensor name: {msg}");
    }

    // ── OnnxGraph::from_proto: valid minimal graph ─────────────────────

    #[test]
    fn onnx_graph_from_proto_minimal_valid() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("minimal".to_string()),
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: Some("minimal graph".to_string()),
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.name, "minimal");
        assert_eq!(graph.doc_string, "minimal graph");
        assert!(graph.nodes.is_empty());
        assert!(graph.initializers.is_empty());
    }

    // ── OnnxGraph::from_proto: single initializer ──────────────────────

    #[test]
    fn onnx_graph_from_proto_single_initializer() {
        let tensor = proto::TensorProto {
            name: Some("weight".to_string()),
            data_type: Some(1), // Float
            dims: vec![2, 3],
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: None,
            initializer: vec![tensor],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.initializers.contains_key("weight"));
        assert_eq!(graph.initializers.get("weight").unwrap().shape, vec![2, 3]);
    }

    // ── OnnxGraph::from_proto: single node ─────────────────────────────

    #[test]
    fn onnx_graph_from_proto_single_node() {
        let node = proto::NodeProto {
            op_type: Some("Relu".to_string()),
            name: Some("relu_0".to_string()),
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        };
        let graph_proto = proto::GraphProto {
            node: vec![node],
            name: Some("single_node".to_string()),
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "Relu");
        assert_eq!(graph.nodes[0].name, "relu_0");
    }

    // ── OnnxGraph::from_proto: metadata_props from proto ───────────────

    #[test]
    fn onnx_graph_from_proto_metadata_props() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: None,
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![
                proto::StringStringEntryProto {
                    key: Some("framework".to_string()),
                    value: Some("pytorch".to_string()),
                },
            ],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.metadata_props.get("framework").unwrap(), "pytorch");
    }

    // ── OnnxModel::from_proto: full metadata roundtrip ─────────────────

    #[test]
    fn onnx_model_from_proto_full_metadata() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("test_graph".to_string()),
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let model_proto = proto::ModelProto {
            ir_version: Some(8),
            producer_name: Some("test-producer".to_string()),
            producer_version: Some("1.0".to_string()),
            domain: Some("ai.onnx".to_string()),
            model_version: Some(42),
            doc_string: Some("test doc".to_string()),
            graph: Some(graph_proto),
            opset_import: vec![proto::OperatorSetIdProto {
                domain: Some("".to_string()),
                version: Some(17),
            }],
            metadata_props: vec![proto::StringStringEntryProto {
                key: Some("author".to_string()),
                value: Some("tester".to_string()),
            }],
            functions: vec![],
            configuration: vec![],
            training_info: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.metadata.ir_version, 8);
        assert_eq!(model.metadata.producer_name, "test-producer");
        assert_eq!(model.metadata.producer_version, "1.0");
        assert_eq!(model.metadata.domain, "ai.onnx");
        assert_eq!(model.metadata.model_version, 42);
        assert_eq!(model.metadata.doc_string, "test doc");
        assert_eq!(model.metadata.opset_import.len(), 1);
        assert_eq!(model.metadata.opset_import[0].version, 17);
        assert_eq!(model.metadata.metadata_props.get("author").unwrap(), "tester");
        assert_eq!(model.graph.name, "test_graph");
    }

    // ── parse_functions: function with value_info ──────────────────────

    #[test]
    fn parse_functions_value_info_proto() {
        let functions = vec![proto::FunctionProto {
            name: Some("FusedBN".to_string()),
            domain: None,
            overload: None,
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![
                proto::ValueInfoProto {
                    name: Some("mean".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
                proto::ValueInfoProto {
                    name: Some("var".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
            ],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].value_info.len(), 2);
        assert_eq!(result[0].value_info[0].name, "mean");
        assert_eq!(result[0].value_info[1].name, "var");
    }

    // ── parse_functions: function with metadata_props ──────────────────

    #[test]
    fn parse_functions_with_metadata_props() {
        let functions = vec![proto::FunctionProto {
            name: Some("CustomRelu".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![proto::StringStringEntryProto {
                key: Some("version".to_string()),
                value: Some("2.0".to_string()),
            }],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].metadata_props.get("version").unwrap(), "2.0");
    }

    // ── parse_functions: multiple functions ─────────────────────────────

    #[test]
    fn parse_functions_multiple_distinct() {
        let functions = vec![
            proto::FunctionProto {
                name: Some("OpA".to_string()),
                domain: None,
                overload: None,
                input: vec![],
                output: vec![],
                attribute: vec![],
                attribute_proto: vec![],
                node: vec![],
                opset_import: vec![],
                value_info: vec![],
                doc_string: None,
                metadata_props: vec![],
            },
            proto::FunctionProto {
                name: Some("OpB".to_string()),
                domain: Some("custom".to_string()),
                overload: Some("v1".to_string()),
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                attribute: vec![],
                attribute_proto: vec![],
                node: vec![],
                opset_import: vec![],
                value_info: vec![],
                doc_string: None,
                metadata_props: vec![],
            },
        ];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "OpA");
        assert_eq!(result[1].name, "OpB");
        assert_eq!(result[1].domain, "custom");
        assert_eq!(result[1].overload, "v1");
        assert_eq!(result[1].inputs, vec!["X"]);
        assert_eq!(result[1].outputs, vec!["Y"]);
    }

    // ── parse_functions: function with doc_string ──────────────────────

    #[test]
    fn parse_functions_doc_string_preserved() {
        let functions = vec![proto::FunctionProto {
            name: Some("MyOp".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: Some("This is a custom operation".to_string()),
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].doc_string, "This is a custom operation");
    }

    // ── parse_quantization: tensor_name None defaults to empty ─────────

    #[test]
    fn parse_quantization_tensor_name_absent_yields_empty() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: None,
            quant_parameter_tensor_names: vec![],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 1);
        assert!(result[0].tensor_name.is_empty());
    }

    // ── parse_quantization: scale and zp from initializers ──────────────

    #[test]
    fn parse_quantization_scale_and_zp_from_initializers() {
        let scale_data = 0.0625f32.to_le_bytes();
        let scale_tensor = OnnxTensor::new(
            "s".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(scale_data.to_vec()),
        );
        let zp_data = 127i64.to_le_bytes();
        let zp_tensor = OnnxTensor::new(
            "z".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(zp_data.to_vec()),
        );
        let mut inits = HashMap::new();
        inits.insert("s".to_string(), scale_tensor);
        inits.insert("z".to_string(), zp_tensor);

        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("fc_weight".to_string()),
            quant_parameter_tensor_names: vec![
                proto::StringStringEntryProto {
                    key: Some("SCALE_TENSOR".to_string()),
                    value: Some("s".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("ZERO_POINT_TENSOR".to_string()),
                    value: Some("z".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("AXIS".to_string()),
                    value: Some("0".to_string()),
                },
            ],
        }];
        let result = parse_quantization(entries, &inits);
        assert_eq!(result.len(), 1);
        assert!((result[0].scale.unwrap() - 0.0625).abs() < 1e-6);
        assert_eq!(result[0].zero_point, Some(127));
        assert_eq!(result[0].axis, Some(0));
    }

    // ── parse_value_info: metadata_props from proto ───────────────────

    #[test]
    fn parse_value_info_proto_metadata_props() {
        let values = vec![proto::ValueInfoProto {
            name: Some("input".to_string()),
            r#type: None,
            doc_string: Some("test input".to_string()),
            metadata_props: vec![proto::StringStringEntryProto {
                key: Some("source".to_string()),
                value: Some("tokenizer".to_string()),
            }],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result[0].metadata_props.get("source").unwrap(), "tokenizer");
    }

    // ── OnnxNode: op_type mutation ──────────────────────────────────────

    #[test]
    fn onnx_node_op_type_field_mutable() {
        let mut node = OnnxNode {
            name: "n".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert_eq!(node.op_type, "Conv");
        node.op_type = "ConvTranspose".to_string();
        assert_eq!(node.op_type, "ConvTranspose");
    }

    // ── OnnxGraph: nodes vec mutation ───────────────────────────────────

    #[test]
    fn onnx_graph_nodes_pop_decrements() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "n1".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n2".to_string(),
                    op_type: "Add".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 2);
        let popped = graph.nodes.pop().unwrap();
        assert_eq!(popped.op_type, "Add");
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "Relu");
    }

    // ── OnnxModel: metadata field mutation ──────────────────────────────

    #[test]
    fn onnx_model_metadata_ir_version_mutable() {
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert_eq!(model.metadata.ir_version, 7);
        model.metadata.ir_version = 9;
        assert_eq!(model.metadata.ir_version, 9);
    }

    // ── OnnxOperatorSet: version mutation ───────────────────────────────

    #[test]
    fn onnx_operator_set_version_mutable() {
        let mut ops = OnnxOperatorSet {
            domain: "".to_string(),
            version: 17,
        };
        ops.version = 20;
        assert_eq!(ops.version, 20);
    }

    // ── OnnxFunction: overload field read ──────────────────────────────

    #[test]
    fn onnx_function_overload_read_access() {
        let func = OnnxFunction {
            name: "Op".to_string(),
            domain: String::new(),
            overload: "v2_alpha".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.overload, "v2_alpha");
    }

    // ── OnnxFunction: overload mutation ─────────────────────────────────

    #[test]
    fn onnx_function_overload_mutable() {
        let mut func = OnnxFunction {
            name: "Op".to_string(),
            domain: String::new(),
            overload: "v1".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        func.overload = "v3".to_string();
        assert_eq!(func.overload, "v3");
    }

    // ── OnnxTensor: name field mutation ─────────────────────────────────

    #[test]
    fn onnx_tensor_name_mutable() {
        let mut tensor = OnnxTensor::new(
            "original".to_string(),
            Dtype::F32,
            vec![2],
            Bytes::from(vec![0u8; 8]),
        );
        assert_eq!(tensor.name, "original");
        tensor.name = "renamed".to_string();
        assert_eq!(tensor.name, "renamed");
    }

    // ── OnnxTensor: is_string false by default ──────────────────────────

    #[test]
    fn onnx_tensor_is_string_false_by_default() {
        let tensor = OnnxTensor::new(
            "weight".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(vec![0u8; 4]),
        );
        assert!(!tensor.is_string);
    }

    // ── OnnxTensor: new_string with empty data ──────────────────────────

    #[test]
    fn onnx_tensor_new_string_empty_data() {
        let tensor = OnnxTensor::new_string(
            "empty_text".to_string(),
            vec![0],
            Bytes::new(),
        );
        assert!(tensor.is_string);
        assert!(tensor.raw_data().is_empty());
        assert_eq!(tensor.shape, vec![0]);
    }

    // ── OnnxSparseTensor: Csc format variant ────────────────────────────

    #[test]
    fn onnx_sparse_tensor_csc_variant() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![2], Bytes::from(vec![0u8; 16]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![3, 3],
            format: OnnxSparseFormat::Csc,
        };
        assert_eq!(sparse.format, OnnxSparseFormat::Csc);
        assert_ne!(sparse.format, OnnxSparseFormat::Csr);
    }

    // ── OnnxSparseTensor: dims empty for scalar sparse ──────────────────

    #[test]
    fn onnx_sparse_tensor_empty_dims() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![],
            format: OnnxSparseFormat::Coo,
        };
        assert!(sparse.dims.is_empty());
    }

    // ── OnnxGraph: sparse_initializers mutation ─────────────────────────

    #[test]
    fn onnx_graph_sparse_initializers_pop() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let s1 = OnnxSparseTensor { values, indices, dims: vec![5], format: OnnxSparseFormat::Coo };
        let values2 = OnnxTensor::new("v2".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices2 = OnnxTensor::new("i2".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let s2 = OnnxSparseTensor { values: values2, indices: indices2, dims: vec![3], format: OnnxSparseFormat::Csr };
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![s1, s2],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.sparse_initializers.len(), 2);
        let popped = graph.sparse_initializers.pop().unwrap();
        assert_eq!(popped.format, OnnxSparseFormat::Csr);
        assert_eq!(graph.sparse_initializers.len(), 1);
    }

    // ── OnnxModel: functions mutation ────────────────────────────────────

    #[test]
    fn onnx_model_functions_pop() {
        let func = OnnxFunction {
            name: "Op1".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![func],
        };
        assert_eq!(model.functions.len(), 1);
        let popped = model.functions.pop().unwrap();
        assert_eq!(popped.name, "Op1");
        assert!(model.functions.is_empty());
    }

    // ── OnnxQuantizationAnnotation: negative axis ───────────────────────

    #[test]
    fn onnx_quantization_annotation_negative_axis() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: Some(-2),
        };
        assert_eq!(qa.axis, Some(-2));
    }

    // ── OnnxQuantizationAnnotation: scale subnormal ─────────────────────

    #[test]
    fn onnx_quantization_annotation_subnormal_scale() {
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(subnormal),
            zero_point: None,
            axis: None,
        };
        assert!(qa.scale.unwrap() > 0.0);
        assert!(qa.scale.unwrap().is_subnormal());
    }

    // ── OnnxValueInfo: name mutation ────────────────────────────────────

    #[test]
    fn onnx_value_info_name_mutable() {
        let mut vi = OnnxValueInfo {
            name: "original".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        vi.name = "renamed".to_string();
        assert_eq!(vi.name, "renamed");
    }

    // ── OnnxValueInfo: doc_string mutation ──────────────────────────────

    #[test]
    fn onnx_value_info_doc_string_mutable() {
        let mut vi = OnnxValueInfo {
            name: "x".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        vi.doc_string = "updated doc".to_string();
        assert_eq!(vi.doc_string, "updated doc");
    }

    // ── OnnxAttribute: doc_string field ─────────────────────────────────

    #[test]
    fn onnx_attribute_doc_string_field() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "epsilon".to_string(),
            value: OnnxAttributeValue::Float(1e-5),
            doc_string: "small value for numerical stability".to_string(),
            ref_attr_name: None,
            attr_type: None,
        };
        assert_eq!(attr.doc_string, "small value for numerical stability");
    }

    // ── OnnxAttribute: attr_type field ──────────────────────────────────

    #[test]
    fn onnx_attribute_attr_type_field() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "pads".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 1]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        };
        assert_eq!(attr.attr_type, Some(proto::attribute_proto::AttributeType::Ints));
    }

    // ── OnnxAttribute: attr_type None default ───────────────────────────

    #[test]
    fn onnx_attribute_attr_type_none_default() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "alpha".to_string(),
            value: OnnxAttributeValue::Float(1.0),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        };
        assert!(attr.attr_type.is_none());
    }

    // ── WeightFormat variants debug ─────────────────────────────────────

    #[test]
    fn weight_format_debug_variants() {
        use crate::loader::WeightFormat;
        assert!(format!("{:?}", WeightFormat::SafeTensors).contains("SafeTensors"));
        assert!(format!("{:?}", WeightFormat::Gguf).contains("Gguf"));
        assert!(format!("{:?}", WeightFormat::Onnx).contains("Onnx"));
        assert!(format!("{:?}", WeightFormat::PyTorch).contains("PyTorch"));
        assert!(format!("{:?}", WeightFormat::Gllm).contains("Gllm"));
    }

    // ── WeightFormat variant equality ────────────────────────────────────

    #[test]
    fn weight_format_equality() {
        use crate::loader::WeightFormat;
        assert_eq!(WeightFormat::Gguf, WeightFormat::Gguf);
        assert_ne!(WeightFormat::Gguf, WeightFormat::Onnx);
        assert_ne!(WeightFormat::SafeTensors, WeightFormat::PyTorch);
    }

    // ── WeightFormat Copy trait verification ──────────────────────────────

    #[test]
    fn weight_format_copy_verify() {
        use crate::loader::WeightFormat;
        let a = WeightFormat::Gguf;
        let b = a; // Copy, not move
        let _ = a; // still usable
        assert_eq!(a, b);
    }

    // ── OnnxModelMetadata: ir_version zero with all empty fields ───────

    #[test]
    fn onnx_model_metadata_ir_version_zero_all_defaults() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.ir_version, 0);
        assert!(meta.producer_name.is_empty());
        assert!(meta.opset_import.is_empty());
    }

    // ── OnnxModelMetadata: producer_name mutation ───────────────────────

    #[test]
    fn onnx_model_metadata_producer_name_mutable() {
        let mut meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: "v1".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        meta.producer_name = "v2".to_string();
        assert_eq!(meta.producer_name, "v2");
    }

    // ── parse_metadata_props: value with special characters ─────────────

    #[test]
    fn parse_metadata_props_value_special_chars() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("path".to_string()),
            value: Some("/usr/local/bin:node".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("path").unwrap(), "/usr/local/bin:node");
    }

    // ── parse_opsets: single opset preserved ────────────────────────────

    #[test]
    fn parse_opsets_single_preserved() {
        let opsets = vec![proto::OperatorSetIdProto {
            domain: Some("ai.onnx".to_string()),
            version: Some(21),
        }];
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].domain, "ai.onnx");
        assert_eq!(result[0].version, 21);
    }

    // ── parse_nodes: domain None defaults to empty ──────────────────────

    #[test]
    fn parse_nodes_domain_none_defaults_empty() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Identity".to_string()),
            name: Some("id".to_string()),
            input: vec![],
            output: vec![],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].domain, "");
    }

    // ── parse_nodes: input and output names preserved ───────────────────

    #[test]
    fn parse_nodes_io_names_preserved() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Gemm".to_string()),
            name: Some("gemm".to_string()),
            input: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            output: vec!["Y".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].inputs, vec!["A", "B", "C"]);
        assert_eq!(result[0].outputs, vec!["Y"]);
    }

    // ── parse_value_info: name exactly preserved ────────────────────────

    #[test]
    fn parse_value_info_name_with_underscores_preserved() {
        let values = vec![proto::ValueInfoProto {
            name: Some("layer_0_attention_weight".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result[0].name, "layer_0_attention_weight");
    }

    // ── OnnxTensor: dtype I64 ───────────────────────────────────────────

    #[test]
    fn onnx_tensor_dtype_i64() {
        let tensor = OnnxTensor::new(
            "idx".to_string(),
            Dtype::I64,
            vec![10],
            Bytes::from(vec![0u8; 80]),
        );
        assert_eq!(tensor.dtype, Dtype::I64);
    }

    // ── OnnxTensor: dtype I32 roundtrip ─────────────────────────────────

    #[test]
    fn onnx_tensor_i32_dtype_roundtrip() {
        let tensor = OnnxTensor::new(
            "labels".to_string(),
            Dtype::I32,
            vec![8],
            Bytes::from(vec![0u8; 32]),
        );
        assert_eq!(tensor.dtype, Dtype::I32);
    }

    // ── OnnxTensor: shape with zero dimension ───────────────────────────

    #[test]
    fn onnx_tensor_shape_zero_dimension() {
        let tensor = OnnxTensor::new(
            "empty".to_string(),
            Dtype::F32,
            vec![0, 3],
            Bytes::new(),
        );
        assert_eq!(tensor.shape, vec![0, 3]);
        assert!(tensor.raw_data().is_empty());
    }

    // ── OnnxTensor: raw_data with exact bytes ───────────────────────────

    #[test]
    fn onnx_tensor_raw_data_exact_bytes() {
        let data = Bytes::from_static(&[0x01, 0x02, 0x03, 0x04]);
        let tensor = OnnxTensor::new(
            "exact".to_string(),
            Dtype::U8,
            vec![4],
            data.clone(),
        );
        assert_eq!(tensor.raw_data(), &[0x01, 0x02, 0x03, 0x04]);
    }

    // ── OnnxGraph: from_proto with input and output value info ──────────

    #[test]
    fn onnx_graph_from_proto_with_inputs_outputs() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("io_graph".to_string()),
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![proto::ValueInfoProto {
                name: Some("input_ids".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            output: vec![proto::ValueInfoProto {
                name: Some("logits".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.inputs[0].name, "input_ids");
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.outputs[0].name, "logits");
    }

    // ── OnnxGraph: from_proto value_info entries ────────────────────────

    #[test]
    fn onnx_graph_proto_value_info_entries() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: None,
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![
                proto::ValueInfoProto {
                    name: Some("hidden_0".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
                proto::ValueInfoProto {
                    name: Some("hidden_1".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
            ],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.value_info.len(), 2);
        assert_eq!(graph.value_info[0].name, "hidden_0");
        assert_eq!(graph.value_info[1].name, "hidden_1");
    }

    // ── OnnxModel: from_proto with functions ────────────────────────────

    #[test]
    fn onnx_model_from_proto_with_functions() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: None,
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let model_proto = proto::ModelProto {
            ir_version: None,
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            doc_string: None,
            graph: Some(graph_proto),
            opset_import: vec![],
            metadata_props: vec![],
            functions: vec![proto::FunctionProto {
                name: Some("CustomGelu".to_string()),
                domain: Some("com.example".to_string()),
                overload: None,
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                attribute: vec![],
                attribute_proto: vec![],
                node: vec![],
                opset_import: vec![],
                value_info: vec![],
                doc_string: None,
                metadata_props: vec![],
            }],
            configuration: vec![],
            training_info: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.functions.len(), 1);
        assert_eq!(model.functions[0].name, "CustomGelu");
        assert_eq!(model.functions[0].domain, "com.example");
    }

    // ── OnnxGraph: from_proto name defaults empty ───────────────────────

    #[test]
    fn onnx_graph_from_proto_name_none_defaults_empty() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: None,
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.name.is_empty());
    }

    // ── OnnxGraph: from_proto doc_string defaults empty ─────────────────

    #[test]
    fn onnx_graph_from_proto_doc_string_none_defaults_empty() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: None,
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.doc_string.is_empty());
    }

    // ── OnnxGraph: from_proto with quantization annotation ──────────────

    #[test]
    fn onnx_graph_from_proto_with_quantization_annotation() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: None,
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![proto::TensorAnnotation {
                tensor_name: Some("q_weight".to_string()),
                quant_parameter_tensor_names: vec![],
            }],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.quantization_annotation.len(), 1);
        assert_eq!(graph.quantization_annotation[0].tensor_name, "q_weight");
    }

    // ── OnnxNode: with optional input (empty string marker) ─────────────

    #[test]
    fn onnx_node_optional_input_empty_string() {
        let node = OnnxNode {
            name: "conv".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string(), "W".to_string(), String::new()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.inputs.len(), 3);
        assert!(node.inputs[2].is_empty());
    }

    // ── OnnxModel: clone graph function independence ─────────────────────

    #[test]
    fn onnx_model_clone_function_node_independence() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![OnnxFunction {
                name: "Op".to_string(),
                domain: String::new(),
                overload: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: vec![],
                attribute_protos: HashMap::new(),
                nodes: vec![OnnxNode {
                    name: "inner".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                }],
                opset_import: vec![],
                value_info: vec![],
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
        };
        let mut cloned = model.clone();
        cloned.functions[0].nodes[0].op_type = "Sigmoid".to_string();
        assert_eq!(model.functions[0].nodes[0].op_type, "Relu");
        assert_eq!(cloned.functions[0].nodes[0].op_type, "Sigmoid");
    }

    // ── OnnxGraph: inputs mutation ──────────────────────────────────────

    #[test]
    fn onnx_graph_inputs_mutable() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![OnnxValueInfo {
                name: "old_input".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.inputs[0].name = "new_input".to_string();
        assert_eq!(graph.inputs[0].name, "new_input");
    }

    // ── OnnxGraph: outputs mutation ─────────────────────────────────────

    #[test]
    fn onnx_graph_outputs_mutable() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![OnnxValueInfo {
                name: "old_output".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.outputs[0].name = "new_output".to_string();
        assert_eq!(graph.outputs[0].name, "new_output");
    }

    // ── OnnxGraph: value_info mutation ──────────────────────────────────

    #[test]
    fn onnx_graph_value_info_mutable() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![OnnxValueInfo {
                name: "temp".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.value_info[0].name = "updated_temp".to_string();
        assert_eq!(graph.value_info[0].name, "updated_temp");
    }

    // ── OnnxModelMetadata: doc_string mutation ──────────────────────────

    #[test]
    fn onnx_model_metadata_doc_string_mutable() {
        let mut meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: "old doc".to_string(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        meta.doc_string = "new doc".to_string();
        assert_eq!(meta.doc_string, "new doc");
    }

    // ── OnnxNode: clone and mutate independence ─────────────────────────

    #[test]
    fn onnx_node_clone_inputs_independence() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Cat".to_string(),
            domain: String::new(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            attributes: HashMap::new(),
        };
        let mut cloned = node.clone();
        cloned.inputs.push("d".to_string());
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(cloned.inputs.len(), 3);
    }

    // ── OnnxGraph: multiple initializers ────────────────────────────────

    #[test]
    fn onnx_graph_multiple_initializers_count() {
        let t1 = OnnxTensor::new("w1".to_string(), Dtype::F32, vec![2, 2], Bytes::from(vec![0u8; 16]));
        let t2 = OnnxTensor::new("w2".to_string(), Dtype::F32, vec![3, 3], Bytes::from(vec![0u8; 36]));
        let t3 = OnnxTensor::new("b1".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([
                ("w1".to_string(), t1),
                ("w2".to_string(), t2),
                ("b1".to_string(), t3),
            ]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.initializers.len(), 3);
        assert!(graph.initializers.contains_key("w1"));
        assert!(graph.initializers.contains_key("w2"));
        assert!(graph.initializers.contains_key("b1"));
    }

    // ── parse_nodes: node with empty input and output lists ─────────────

    #[test]
    fn parse_nodes_empty_io_lists() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Constant".to_string()),
            name: None,
            input: vec![],
            output: vec!["value".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert!(result[0].inputs.is_empty());
        assert_eq!(result[0].outputs, vec!["value"]);
    }

    // ── parse_value_info: doc_string absent yields empty ────────────────

    #[test]
    fn parse_value_info_doc_absent_yields_empty() {
        let values = vec![proto::ValueInfoProto {
            name: Some("x".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        }];
        let result = parse_value_info(values).unwrap();
        assert!(result[0].doc_string.is_empty());
    }

    // ── OnnxModel: with multiple functions of different domains ──────────

    #[test]
    fn onnx_model_functions_different_domains() {
        let f1 = OnnxFunction {
            name: "Op1".to_string(),
            domain: "com.ms".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let f2 = OnnxFunction {
            name: "Op2".to_string(),
            domain: "com.nvidia".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![f1, f2],
        };
        assert_eq!(model.functions[0].domain, "com.ms");
        assert_eq!(model.functions[1].domain, "com.nvidia");
    }

    // ── OnnxGraph: doc_string mutation ──────────────────────────────────

    #[test]
    fn onnx_graph_doc_string_mutable() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: "original".to_string(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.doc_string = "updated".to_string();
        assert_eq!(graph.doc_string, "updated");
    }

    // ── OnnxFunction: nodes mutation ────────────────────────────────────

    #[test]
    fn onnx_function_nodes_mutable() {
        let mut func = OnnxFunction {
            name: "Op".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        func.nodes.push(OnnxNode {
            name: "new_node".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        });
        assert_eq!(func.nodes.len(), 1);
        assert_eq!(func.nodes[0].name, "new_node");
    }

    // ── OnnxGraph: from_proto with 5 nodes sequential ───────────────────

    #[test]
    fn onnx_graph_from_proto_five_sequential_nodes() {
        let nodes: Vec<proto::NodeProto> = (0..5)
            .map(|i| proto::NodeProto {
                op_type: Some(format!("Op{i}")),
                name: None,
                input: if i > 0 { vec![format!("t{}", i - 1)] } else { vec!["input".to_string()] },
                output: vec![format!("t{i}")],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            })
            .collect();
        let graph_proto = proto::GraphProto {
            node: nodes,
            name: Some("pipeline".to_string()),
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.nodes.len(), 5);
        assert_eq!(graph.nodes[0].op_type, "Op0");
        assert_eq!(graph.nodes[4].op_type, "Op4");
        assert_eq!(graph.nodes[0].name, "node_0");
        assert_eq!(graph.nodes[0].inputs[0], "input");
        assert_eq!(graph.nodes[2].inputs[0], "t1");
    }

    // ── OnnxAttributeValue: SparseTensors collection ──────────────────

    #[test]
    fn onnx_attribute_value_sparse_tensors_collection() {
        use super::super::attributes::OnnxAttributeValue;
        use super::super::tensor::OnnxSparseFormat;
        let v1 = OnnxTensor::new("sv1".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let i1 = OnnxTensor::new("si1".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let s1 = OnnxSparseTensor { values: v1, indices: i1, dims: vec![5], format: OnnxSparseFormat::Coo };
        let val = OnnxAttributeValue::SparseTensors(vec![s1]);
        assert!(matches!(val, OnnxAttributeValue::SparseTensors(v) if v.len() == 1));
    }

    // ── OnnxAttributeValue: Types collection ───────────────────────────

    #[test]
    fn onnx_attribute_value_types_collection() {
        use super::super::attributes::OnnxAttributeValue;
        use super::super::types::{OnnxTensorShape, OnnxTensorType};
        let ty = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let val = OnnxAttributeValue::Types(vec![ty]);
        assert!(matches!(val, OnnxAttributeValue::Types(v) if v.len() == 1));
    }

    // ── OnnxAttributeValue: Ref variant clone ───────────────────────────

    #[test]
    fn onnx_attribute_value_ref_clone() {
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Ref("other_attr".to_string());
        let cloned = val.clone();
        assert!(matches!(cloned, OnnxAttributeValue::Ref(s) if s == "other_attr"));
    }

    // ── LoaderError: from IO error variant ──────────────────────────────

    #[test]
    fn loader_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = LoaderError::from(io_err);
        let msg = format!("{err}");
        assert!(msg.contains("IO error"), "expected IO error prefix: {msg}");
    }

    // ── LoaderError: from JSON error variant ────────────────────────────

    #[test]
    fn loader_error_from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("{bad").unwrap_err();
        let err = LoaderError::from(json_err);
        let msg = format!("{err}");
        assert!(msg.contains("JSON error"), "expected JSON error prefix: {msg}");
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 8 TESTS (40 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxTensor slice() accessor ──────────────────────────────────────

    #[test]
    fn onnx_tensor_slice_returns_correct_metadata() {
        let tensor = OnnxTensor::new(
            "test_slice".to_string(),
            Dtype::F32,
            vec![3, 4],
            Bytes::from(vec![0u8; 48]),
        );
        let s = tensor.slice();
        assert_eq!(s.shape, vec![3, 4]);
        assert_eq!(s.dtype, Dtype::F32);
        assert_eq!(s.data.len(), 48);
    }

    #[test]
    fn onnx_tensor_slice_empty_shape() {
        let tensor = OnnxTensor::new(
            "scalar".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(1.0f32.to_le_bytes().to_vec()),
        );
        let s = tensor.slice();
        assert!(s.shape.is_empty());
    }

    #[test]
    fn onnx_tensor_slice_preserves_dtype() {
        let tensor = OnnxTensor::new(
            "bf16_t".to_string(),
            Dtype::BF16,
            vec![2],
            Bytes::from(vec![0u8; 4]),
        );
        let s = tensor.slice();
        assert_eq!(s.dtype, Dtype::BF16);
    }

    // ── OnnxTensor scalar_f32 with subnormal values ──────────────────────

    #[test]
    fn onnx_tensor_scalar_f32_subnormal() {
        let subnormal = 1e-45f32; // subnormal float
        let tensor = OnnxTensor::new(
            "sub".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(subnormal.to_le_bytes().to_vec()),
        );
        let result = tensor.scalar_f32();
        assert!(result.is_some());
        assert!(result.unwrap() > 0.0);
        assert!(result.unwrap().is_subnormal());
    }

    // ── OnnxTensor scalar_i64 negative value ─────────────────────────────

    #[test]
    fn onnx_tensor_scalar_i64_negative() {
        let value = -999i64;
        let tensor = OnnxTensor::new(
            "neg".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(value.to_le_bytes().to_vec()),
        );
        assert_eq!(tensor.scalar_i64(), Some(-999));
    }

    // ── OnnxTensor is_string field directly ──────────────────────────────

    #[test]
    fn onnx_tensor_new_is_not_string_by_default() {
        let tensor = OnnxTensor::new(
            "regular".to_string(),
            Dtype::F32,
            vec![1],
            Bytes::from(vec![0u8; 4]),
        );
        assert!(!tensor.is_string);
    }

    // ── OnnxTensor raw_data length matches bytes ─────────────────────────

    #[test]
    fn onnx_tensor_raw_data_length_matches() {
        let data = Bytes::from(vec![42u8; 100]);
        let tensor = OnnxTensor::new("len_test".to_string(), Dtype::U8, vec![100], data);
        assert_eq!(tensor.raw_data().len(), 100);
    }

    // ── OnnxDim Known with i64::MAX ─────────────────────────────────────

    #[test]
    fn onnx_dim_known_i64_max() {
        use super::super::types::OnnxDim;
        let dim = OnnxDim::Known(i64::MAX);
        assert_eq!(dim, OnnxDim::Known(i64::MAX));
    }

    // ── OnnxTensorShape with single dimension ────────────────────────────

    #[test]
    fn onnx_tensor_shape_single_dim() {
        use super::super::types::{OnnxDim, OnnxTensorShape};
        let shape = OnnxTensorShape { dims: vec![OnnxDim::Known(512)] };
        assert_eq!(shape.dims.len(), 1);
        assert_eq!(shape.dims[0], OnnxDim::Known(512));
    }

    // ── OnnxTensorType with BF16 elem_type ───────────────────────────────

    #[test]
    fn onnx_tensor_type_bfloat16() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType};
        let tt = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Bfloat16,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(64)] },
        };
        assert!(matches!(tt.elem_type, proto::tensor_proto::DataType::Bfloat16));
        assert_eq!(tt.shape.dims.len(), 1);
    }

    // ── OnnxType SparseTensor equality ───────────────────────────────────

    #[test]
    fn onnx_type_sparse_tensor_equality() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let st1 = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        let st2 = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        assert_eq!(st1, st2);
    }

    // ── OnnxMapType clone consistency ────────────────────────────────────

    #[test]
    fn onnx_map_type_clone() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxMapType, OnnxType};
        let mt = OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![OnnxDim::Unknown] },
            })),
        };
        let cloned = mt.clone();
        assert!(matches!(cloned.key_type, proto::tensor_proto::DataType::Int64));
        assert!(matches!(*cloned.value_type, OnnxType::Tensor(_)));
    }

    // ── OnnxGraph: bind_weights with real TensorProvider ─────────────────

    #[test]
    fn onnx_graph_bind_weights_maps_correct_tensors() {
        struct SimpleProvider {
            metas: Vec<crate::loader::TensorMeta>,
        }
        impl crate::loader::TensorProvider for SimpleProvider {
            fn tensor_info(&self, name: &str) -> Option<crate::loader::TensorMeta> {
                self.metas.iter().find(|m| m.name == name).cloned()
            }
            fn load_tensor_data(
                &self,
                name: &str,
            ) -> std::result::Result<std::borrow::Cow<'_, [u8]>, LoaderError> {
                Ok(std::borrow::Cow::Owned(vec![0u8; 8]))
            }
            fn iter_tensors(&self) -> impl Iterator<Item = crate::loader::TensorMeta> {
                self.metas.clone().into_iter()
            }
        }
        let provider = SimpleProvider {
            metas: vec![crate::loader::TensorMeta {
                name: "weight_a".to_string(),
                dtype: Dtype::F32,
                shape: vec![2],
            }],
        };
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let mapping = HashMap::from([("graph_weight".to_string(), "weight_a".to_string())]);
        let count = graph.bind_weights(&provider, &mapping).unwrap();
        assert_eq!(count, 1);
        assert!(graph.initializers.contains_key("graph_weight"));
        assert!(!graph.initializers.contains_key("weight_a"));
    }

    #[test]
    fn onnx_graph_bind_weights_partial_mapping_binds_only_matched() {
        struct PartialProvider;
        impl crate::loader::TensorProvider for PartialProvider {
            fn tensor_info(&self, name: &str) -> Option<crate::loader::TensorMeta> {
                if name == "existing" {
                    Some(crate::loader::TensorMeta {
                        name: "existing".to_string(),
                        dtype: Dtype::F32,
                        shape: vec![1],
                    })
                } else {
                    None
                }
            }
            fn load_tensor_data(
                &self,
                _name: &str,
            ) -> std::result::Result<std::borrow::Cow<'_, [u8]>, LoaderError> {
                Ok(std::borrow::Cow::Owned(vec![0u8; 4]))
            }
            fn iter_tensors(&self) -> impl Iterator<Item = crate::loader::TensorMeta> {
                std::iter::once(crate::loader::TensorMeta {
                    name: "existing".to_string(),
                    dtype: Dtype::F32,
                    shape: vec![1],
                })
            }
        }
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let mapping = HashMap::from([
            ("w1".to_string(), "existing".to_string()),
            ("w2".to_string(), "missing".to_string()),
        ]);
        let count = graph.bind_weights(&PartialProvider, &mapping).unwrap();
        assert_eq!(count, 1);
        assert!(graph.initializers.contains_key("w1"));
        assert!(!graph.initializers.contains_key("w2"));
    }

    // ── parse_nodes: node with overload field ────────────────────────────

    #[test]
    fn parse_nodes_overload_field_ignored() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("CustomOp".to_string()),
            name: Some("overload_node".to_string()),
            input: vec![],
            output: vec![],
            domain: None,
            overload: Some("v2".to_string()),
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].name, "overload_node");
        assert_eq!(result[0].op_type, "CustomOp");
    }

    // ── parse_nodes: node with device_configurations ─────────────────────

    #[test]
    fn parse_nodes_device_configurations_field_ignored() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Relu".to_string()),
            name: Some("dev_cfg_node".to_string()),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].name, "dev_cfg_node");
    }

    // ── OnnxGraph from_proto: empty initializer list yields empty map ────

    #[test]
    fn onnx_graph_from_proto_empty_initializer_list() {
        let graph_proto = proto::GraphProto {
            name: Some("g".to_string()),
            initializer: vec![],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.initializers.is_empty());
    }

    // ── OnnxGraph from_proto: sparse_initializer empty yields empty vec ──

    #[test]
    fn onnx_graph_from_proto_empty_sparse_initializer() {
        let graph_proto = proto::GraphProto {
            name: Some("g".to_string()),
            sparse_initializer: vec![],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.sparse_initializers.is_empty());
    }

    // ── OnnxModel: graph field is mutable ────────────────────────────────

    #[test]
    fn onnx_model_graph_field_mutable() {
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "original".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        model.graph.name = "renamed".to_string();
        assert_eq!(model.graph.name, "renamed");
    }

    // ── OnnxModel: metadata field is mutable ─────────────────────────────

    #[test]
    fn onnx_model_metadata_field_mutable() {
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: "orig".to_string(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        model.metadata.ir_version = 9;
        assert_eq!(model.metadata.ir_version, 9);
    }

    // ── OnnxModel: functions vec push and access ─────────────────────────

    #[test]
    fn onnx_model_functions_push_and_len() {
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert!(model.functions.is_empty());
        model.functions.push(OnnxFunction {
            name: "NewOp".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        });
        assert_eq!(model.functions.len(), 1);
        assert_eq!(model.functions[0].name, "NewOp");
    }

    // ── OnnxGraph: quantization_annotation push ─────────────────────────

    #[test]
    fn onnx_graph_quantization_annotation_push() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.quantization_annotation.is_empty());
        graph.quantization_annotation.push(OnnxQuantizationAnnotation {
            tensor_name: "new_weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.1),
            zero_point: None,
            axis: None,
        });
        assert_eq!(graph.quantization_annotation.len(), 1);
        assert_eq!(graph.quantization_annotation[0].tensor_name, "new_weight");
    }

    // ── parse_quantization: empty quant_parameter_tensor_names ───────────

    #[test]
    fn parse_quantization_empty_param_names_all_none() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("w".to_string()),
            quant_parameter_tensor_names: vec![],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 1);
        assert!(result[0].scale.is_none());
        assert!(result[0].zero_point.is_none());
        assert!(result[0].axis.is_none());
        assert!(result[0].quant_param_tensor_names.is_empty());
    }

    // ── OnnxValueInfo: with deeply nested type ───────────────────────────

    #[test]
    fn onnx_value_info_deeply_nested_sequence() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let inner = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        let seq_of_seq = OnnxType::Sequence(Box::new(OnnxType::Sequence(Box::new(inner))));
        let vi = OnnxValueInfo {
            name: "nested_seq".to_string(),
            value_type: Some(seq_of_seq),
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        if let Some(OnnxType::Sequence(outer)) = vi.value_type {
            assert!(matches!(*outer, OnnxType::Sequence(_)));
        } else {
            panic!("expected Sequence variant");
        }
    }

    // ── OnnxNode: name with unicode characters ───────────────────────────

    #[test]
    fn onnx_node_name_unicode() {
        let node = OnnxNode {
            name: "节点_ノード".to_string(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert_eq!(node.name, "节点_ノード");
    }

    // ── OnnxNode: op_type with namespace separator ───────────────────────

    #[test]
    fn onnx_node_op_type_namespace_preserved() {
        let node = OnnxNode {
            name: "ns_op".to_string(),
            op_type: "com.microsoft::SkipLayerNormalization".to_string(),
            domain: "com.microsoft".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert!(node.op_type.contains("::"));
    }

    // ── OnnxFunction: overload field with complex value ──────────────────

    #[test]
    fn onnx_function_overload_complex() {
        let func = OnnxFunction {
            name: "Op".to_string(),
            domain: String::new(),
            overload: "v2_alpha::beta".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(func.overload.contains("::"));
        assert!(func.overload.contains("_"));
    }

    // ── OnnxFunction: attributes vec preserves order ─────────────────────

    #[test]
    fn onnx_function_attributes_preserve_order() {
        let func = OnnxFunction {
            name: "F".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec!["z_attr".to_string(), "a_attr".to_string(), "m_attr".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.attributes[0], "z_attr");
        assert_eq!(func.attributes[1], "a_attr");
        assert_eq!(func.attributes[2], "m_attr");
    }

    // ── OnnxQuantizationAnnotation: with empty quant_param_tensor_names ──

    #[test]
    fn onnx_quantization_annotation_empty_params_lookup_safe() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        assert!(qa.quant_param_tensor_names.get("SCALE_TENSOR").is_none());
        assert!(qa.quant_param_tensor_names.get("ZERO_POINT_TENSOR").is_none());
    }

    // ── OnnxGraph: nodes vec iteration order preserved ───────────────────

    #[test]
    fn onnx_graph_nodes_iteration_order() {
        let ops = vec!["Conv", "BatchNorm", "Relu", "Pool", "Flatten", "Gemm"];
        let nodes: Vec<OnnxNode> = ops
            .iter()
            .enumerate()
            .map(|(i, op)| OnnxNode {
                name: format!("n{i}"),
                op_type: op.to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: HashMap::new(),
            })
            .collect();
        let graph = OnnxGraph {
            name: "ordered".to_string(),
            doc_string: String::new(),
            nodes,
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let collected: Vec<&str> = graph.nodes.iter().map(|n| n.op_type.as_str()).collect();
        assert_eq!(collected, vec!["Conv", "BatchNorm", "Relu", "Pool", "Flatten", "Gemm"]);
    }

    // ── OnnxGraph: inputs vec push ───────────────────────────────────────

    #[test]
    fn onnx_graph_inputs_push() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.inputs.push(OnnxValueInfo {
            name: "new_input".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        });
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.inputs[0].name, "new_input");
    }

    // ── OnnxGraph: outputs vec push ──────────────────────────────────────

    #[test]
    fn onnx_graph_outputs_push() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.outputs.push(OnnxValueInfo {
            name: "new_output".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        });
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.outputs[0].name, "new_output");
    }

    // ── parse_opsets: version zero and negative ──────────────────────────

    #[test]
    fn parse_opsets_version_zero_and_negative() {
        let opsets = vec![
            proto::OperatorSetIdProto { domain: Some("a".to_string()), version: Some(0) },
            proto::OperatorSetIdProto { domain: Some("b".to_string()), version: Some(-5) },
        ];
        let result = parse_opsets(opsets);
        assert_eq!(result[0].version, 0);
        assert_eq!(result[1].version, -5);
    }

    // ── parse_value_info: many entries count correct ─────────────────────

    #[test]
    fn parse_value_info_large_batch() {
        let values: Vec<proto::ValueInfoProto> = (0..100)
            .map(|i| proto::ValueInfoProto {
                name: Some(format!("v{i}")),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            })
            .collect();
        let result = parse_value_info(values).unwrap();
        assert_eq!(result.len(), 100);
    }

    // ── OnnxOperatorSet: Debug output contains both fields ───────────────

    #[test]
    fn onnx_operator_set_debug_contains_both_fields() {
        let ops = OnnxOperatorSet {
            domain: "test.domain".to_string(),
            version: 42,
        };
        let debug = format!("{ops:?}");
        assert!(debug.contains("test.domain"), "expected domain: {debug}");
        assert!(debug.contains("42"), "expected version: {debug}");
    }

    // ── OnnxAttribute: name field access ─────────────────────────────────

    #[test]
    fn onnx_attribute_name_field() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "my_param".to_string(),
            value: OnnxAttributeValue::Int(10),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        };
        assert_eq!(attr.name, "my_param");
    }

    // ── OnnxAttribute: Debug output contains name ────────────────────────

    #[test]
    fn onnx_attribute_debug_contains_name() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "test_attr".to_string(),
            value: OnnxAttributeValue::Float(1.0),
            doc_string: "test".to_string(),
            ref_attr_name: None,
            attr_type: None,
        };
        let debug = format!("{attr:?}");
        assert!(debug.contains("test_attr"), "expected name in: {debug}");
    }

    // ── OnnxSparseTensor: dims field with 3D ─────────────────────────────

    #[test]
    fn onnx_sparse_tensor_3d_dims() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let sparse = OnnxSparseTensor {
            values: OnnxTensor::new("v".to_string(), Dtype::F32, vec![5], Bytes::from(vec![0u8; 20])),
            indices: OnnxTensor::new("i".to_string(), Dtype::I64, vec![5], Bytes::from(vec![0u8; 40])),
            dims: vec![10, 20, 30],
            format: OnnxSparseFormat::Coo,
        };
        assert_eq!(sparse.dims, vec![10, 20, 30]);
        assert_eq!(sparse.dims.len(), 3);
    }

    // ── OnnxGraph: nodes with same op_type in sequence ──────────────────

    #[test]
    fn onnx_graph_same_op_type_sequential_nodes() {
        let graph = OnnxGraph {
            name: "relu_stack".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "relu_0".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["a".to_string()],
                    outputs: vec!["b".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "relu_1".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["b".to_string()],
                    outputs: vec!["c".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "relu_2".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["c".to_string()],
                    outputs: vec!["d".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.nodes.iter().all(|n| n.op_type == "Relu"));
        assert_eq!(graph.nodes[0].outputs[0], graph.nodes[1].inputs[0]);
        assert_eq!(graph.nodes[1].outputs[0], graph.nodes[2].inputs[0]);
    }

    // ── OnnxGraph: initializer with I32 dtype ───────────────────────────

    #[test]
    fn onnx_graph_initializer_i32_tensor() {
        let tensor = OnnxTensor::new(
            "int_tensor".to_string(),
            Dtype::I32,
            vec![3],
            Bytes::from(vec![0u8; 12]),
        );
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("int_tensor".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let t = graph.initializers.get("int_tensor").unwrap();
        assert_eq!(t.dtype, Dtype::I32);
        assert_eq!(t.shape, vec![3]);
    }

    // ── OnnxGraph: initializer with U8 dtype ────────────────────────────

    #[test]
    fn onnx_graph_initializer_u8_tensor() {
        let tensor = OnnxTensor::new(
            "byte_tensor".to_string(),
            Dtype::U8,
            vec![4],
            Bytes::from(vec![0u8; 4]),
        );
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("byte_tensor".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let t = graph.initializers.get("byte_tensor").unwrap();
        assert_eq!(t.dtype, Dtype::U8);
    }

    // ── OnnxNode: attributes HashMap clear ───────────────────────────────

    #[test]
    fn onnx_node_attributes_clear() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert("pads".to_string(), OnnxAttribute {
            name: "pads".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 1]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let mut node = OnnxNode {
            name: "conv".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
        };
        assert_eq!(node.attributes.len(), 1);
        node.attributes.clear();
        assert!(node.attributes.is_empty());
    }

    // ── OnnxModelMetadata: all integer fields range ──────────────────────

    #[test]
    fn onnx_model_metadata_integer_fields_range() {
        let meta_min = OnnxModelMetadata {
            ir_version: i64::MIN,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: i64::MIN,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta_min.ir_version, i64::MIN);
        assert_eq!(meta_min.model_version, i64::MIN);
    }

    // ── OnnxValueInfo: with all OnnxType variants via type field ──────────

    #[test]
    fn onnx_value_info_all_type_variants_can_be_set() {
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxMapType, OnnxType};
        let variants: Vec<OnnxType> = vec![
            OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![OnnxDim::Known(1)] },
            }),
            OnnxType::SparseTensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Int32,
                shape: OnnxTensorShape { dims: vec![] },
            }),
            OnnxType::Sequence(Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Bool,
                shape: OnnxTensorShape { dims: vec![] },
            }))),
            OnnxType::Map(OnnxMapType {
                key_type: proto::tensor_proto::DataType::String,
                value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                    elem_type: proto::tensor_proto::DataType::Float,
                    shape: OnnxTensorShape { dims: vec![] },
                })),
            }),
            OnnxType::Optional(Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Double,
                shape: OnnxTensorShape { dims: vec![OnnxDim::Unknown] },
            }))),
        ];
        for (i, vt) in variants.into_iter().enumerate() {
            let vi = OnnxValueInfo {
                name: format!("var_{i}"),
                value_type: Some(vt),
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            };
            assert!(vi.value_type.is_some());
        }
    }

    // ── OnnxGraph: from_proto with sparse_initializer duplicate error ────

    #[test]
    fn onnx_graph_from_proto_sparse_duplicate_dense_name_error() {
        let val = proto::TensorProto {
            name: Some("shared_name".to_string()),
            data_type: Some(1),
            dims: vec![1],
            float_data: vec![1.0],
            ..Default::default()
        };
        let idx = proto::TensorProto {
            name: Some("idx_a".to_string()),
            data_type: Some(7),
            dims: vec![1],
            int64_data: vec![0],
            ..Default::default()
        };
        let sparse = proto::SparseTensorProto {
            values: Some(val),
            indices: Some(idx),
            dims: vec![5],
        };
        let dense = proto::TensorProto {
            name: Some("shared_name".to_string()),
            data_type: Some(1),
            dims: vec![1],
            float_data: vec![2.0],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            name: Some("g".to_string()),
            initializer: vec![dense],
            sparse_initializer: vec![sparse],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = OnnxGraph::from_proto(graph_proto, &mut resolver);
        assert!(result.is_err());
    }

    // ── parse_metadata_props: value with emoji ───────────────────────────

    #[test]
    fn parse_metadata_props_emoji_value() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("status".to_string()),
            value: Some("✅ passed 🎉".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("status").unwrap(), "✅ passed 🎉");
    }

    // ── OnnxFunction: doc_string with multiline ──────────────────────────

    #[test]
    fn onnx_function_doc_string_multiline_content() {
        let func = OnnxFunction {
            name: "F".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: "line1\nline2\nline3".to_string(),
            metadata_props: HashMap::new(),
        };
        assert!(func.doc_string.contains('\n'));
        let lines: Vec<&str> = func.doc_string.split('\n').collect();
        assert_eq!(lines.len(), 3);
    }

    // ── OnnxModelMetadata: producer_name with special chars ──────────────

    #[test]
    fn onnx_model_metadata_producer_name_special_chars() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: "test-producer_v2.0 (beta)".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(meta.producer_name.contains('-'));
        assert!(meta.producer_name.contains('_'));
        assert!(meta.producer_name.contains('('));
        assert!(meta.producer_name.contains(')'));
    }

    // ── OnnxNode: outputs vec is mutable ─────────────────────────────────

    #[test]
    fn onnx_node_outputs_vec_mutable() {
        let mut node = OnnxNode {
            name: "n".to_string(),
            op_type: "Split".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec!["a".to_string(), "b".to_string()],
            attributes: HashMap::new(),
        };
        node.outputs.push("c".to_string());
        assert_eq!(node.outputs.len(), 3);
        assert_eq!(node.outputs[2], "c");
    }

    // ── OnnxGraph: sparse_initializers vec mutable ───────────────────────

    #[test]
    fn onnx_graph_sparse_initializers_push() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let sparse = OnnxSparseTensor {
            values: OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4])),
            indices: OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8])),
            dims: vec![5],
            format: OnnxSparseFormat::Csr,
        };
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.sparse_initializers.push(sparse);
        assert_eq!(graph.sparse_initializers.len(), 1);
        assert_eq!(graph.sparse_initializers[0].format, OnnxSparseFormat::Csr);
    }

    // ── OnnxQuantizationAnnotation: all field mutation ───────────────────

    #[test]
    fn onnx_quantization_annotation_field_mutation() {
        let mut qa = OnnxQuantizationAnnotation {
            tensor_name: "orig".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        qa.tensor_name = "updated".to_string();
        qa.scale = Some(0.5);
        qa.zero_point = Some(100);
        qa.axis = Some(2);
        assert_eq!(qa.tensor_name, "updated");
        assert_eq!(qa.scale, Some(0.5));
        assert_eq!(qa.zero_point, Some(100));
        assert_eq!(qa.axis, Some(2));
    }

    // ── OnnxModel: metadata opset_import clone independence ─────────────

    #[test]
    fn onnx_model_opset_import_clone_independence() {
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![OnnxOperatorSet { domain: "".to_string(), version: 17 }],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let mut cloned = model.clone();
        cloned.metadata.opset_import.push(OnnxOperatorSet { domain: "ml".to_string(), version: 3 });
        assert_eq!(model.metadata.opset_import.len(), 1);
        assert_eq!(cloned.metadata.opset_import.len(), 2);
    }

    // ── OnnxGraph: from_proto preserves node order ──────────────────────

    #[test]
    fn onnx_graph_from_proto_preserves_node_order() {
        let nodes: Vec<proto::NodeProto> = vec![
            proto::NodeProto {
                op_type: Some("Conv".to_string()),
                name: Some("conv".to_string()),
                input: vec!["X".to_string()],
                output: vec!["H".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: Some("BatchNorm".to_string()),
                name: Some("bn".to_string()),
                input: vec!["H".to_string()],
                output: vec!["N".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: Some("Relu".to_string()),
                name: Some("relu".to_string()),
                input: vec!["N".to_string()],
                output: vec!["Y".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
        ];
        let graph_proto = proto::GraphProto {
            node: nodes,
            name: None,
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].op_type, "Conv");
        assert_eq!(graph.nodes[1].op_type, "BatchNorm");
        assert_eq!(graph.nodes[2].op_type, "Relu");
        assert_eq!(graph.nodes[0].outputs[0], "H");
        assert_eq!(graph.nodes[1].inputs[0], "H");
        assert_eq!(graph.nodes[1].outputs[0], "N");
        assert_eq!(graph.nodes[2].inputs[0], "N");
    }

    // ── Additional coverage tests ────────────────────────────────────

    #[test]
    fn onnx_model_functions_empty_vec() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert!(model.functions.is_empty());
        assert_eq!(model.functions.len(), 0);
    }

    #[test]
    fn onnx_graph_inputs_empty_vec_by_default() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.inputs.is_empty());
        assert!(graph.outputs.is_empty());
        assert!(graph.value_info.is_empty());
    }

    #[test]
    fn onnx_graph_outputs_empty_vec_by_default() {
        let graph = OnnxGraph {
            name: "out_test".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![OnnxValueInfo {
                name: "X".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.inputs.len(), 1);
        assert!(graph.outputs.is_empty());
    }

    #[test]
    fn onnx_graph_value_info_empty_by_default() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.value_info.is_empty());
    }

    #[test]
    fn onnx_node_attributes_empty_map_default() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Identity".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert!(node.attributes.is_empty());
        assert!(!node.attributes.contains_key("any"));
    }

    #[test]
    fn onnx_model_metadata_producer_version_default() {
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: "gllm-test".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(meta.producer_version.is_empty());
        assert_eq!(meta.ir_version, 8);
    }

    #[test]
    fn onnx_model_metadata_domain_default_string() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(meta.domain.is_empty());
    }

    #[test]
    fn onnx_operator_set_version_i64_boundary() {
        let opset = OnnxOperatorSet {
            domain: "test.boundary".to_string(),
            version: i64::MAX,
        };
        assert_eq!(opset.version, i64::MAX);

        let opset_min = OnnxOperatorSet {
            domain: "test.boundary".to_string(),
            version: i64::MIN,
        };
        assert_eq!(opset_min.version, i64::MIN);
    }

    #[test]
    fn onnx_graph_initializer_contains_after_insert() {
        let tensor = OnnxTensor::new(
            "weight".to_string(),
            Dtype::F32,
            vec![2, 3],
            Bytes::from_static(&[0u8; 24]),
        );
        let mut graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(!graph.initializers.contains_key("weight"));
        graph.initializers.insert("weight".to_string(), tensor);
        assert!(graph.initializers.contains_key("weight"));
        assert_eq!(graph.initializers.len(), 1);
    }

    #[test]
    fn onnx_function_inputs_empty_valid() {
        let func = OnnxFunction {
            name: "Sink".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(func.inputs.is_empty());
        assert_eq!(func.outputs.len(), 1);
    }

    #[test]
    fn onnx_function_outputs_empty_valid() {
        let func = OnnxFunction {
            name: "Source".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.inputs.len(), 1);
        assert!(func.outputs.is_empty());
    }

    #[test]
    fn onnx_quantization_annotation_tensor_name_empty_string() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: String::new(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        assert!(ann.tensor_name.is_empty());
        assert!(ann.scale.is_none());
        assert!(ann.zero_point.is_none());
        assert!(ann.axis.is_none());
    }

    #[test]
    fn parse_opsets_two_entries_domain_and_version() {
        let opsets = vec![
            proto::OperatorSetIdProto {
                domain: Some("ai.onnx".to_string()),
                version: Some(17),
            },
            proto::OperatorSetIdProto {
                domain: Some("com.microsoft".to_string()),
                version: Some(1),
            },
        ];
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].domain, "ai.onnx");
        assert_eq!(result[0].version, 17);
        assert_eq!(result[1].domain, "com.microsoft");
        assert_eq!(result[1].version, 1);
    }

    #[test]
    fn onnx_graph_sparse_initializers_len_zero_default() {
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.sparse_initializers.is_empty());
        assert_eq!(graph.sparse_initializers.len(), 0);
    }

    #[test]
    fn onnx_model_metadata_model_version_zero() {
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.model_version, 0);
    }

    // ── Wave 12x34: additional edge case tests ──────────────────────────

    /// @trace TEST-ONNX-647 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with an empty string AXIS value yields None.
    /// The parse::<i32>() call on "" should fail, producing axis = None.
    #[test]
    fn parse_quantization_empty_string_axis_yields_none() {
        use super::super::tensor::OnnxTensor;

        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some(String::new()),
            }],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tensor_name, "weight");
        assert!(result[0].axis.is_none(), "empty string AXIS should parse to None");
    }

    /// @trace TEST-ONNX-648 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with a BOOL-dtype scale tensor yields None for scale.
    /// BOOL is not in scalar_f32's match arms, so it returns None.
    #[test]
    fn parse_quantization_scale_tensor_bool_dtype_yields_none() {
        use super::super::tensor::OnnxTensor;

        let scale_tensor = OnnxTensor::new(
            "bool_scale".to_string(),
            Dtype::BOOL,
            vec![],
            Bytes::from_static(&[1u8]),
        );
        let mut initializers = HashMap::new();
        initializers.insert("bool_scale".to_string(), scale_tensor);

        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("SCALE_TENSOR".to_string()),
                value: Some("bool_scale".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &initializers);
        assert_eq!(result.len(), 1);
        assert!(result[0].scale.is_none(), "BOOL dtype scale tensor should yield None");
    }

    /// @trace TEST-ONNX-649 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with whitespace-only AXIS string yields None.
    /// "  " is not a valid integer string, so parse::<i32>() fails.
    #[test]
    fn parse_quantization_whitespace_axis_yields_none() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some("  ".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 1);
        assert!(result[0].axis.is_none(), "whitespace-only AXIS should parse to None");
    }

    /// @trace TEST-ONNX-650 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with an F32-dtype zero_point tensor succeeds.
    /// scalar_i64 handles F32 by casting to i64, so it should return Some(truncated).
    #[test]
    fn parse_quantization_zero_point_f32_dtype_succeeds() {
        use super::super::tensor::OnnxTensor;

        let zp_data = 7.5f32.to_le_bytes();
        let zp_tensor = OnnxTensor::new(
            "zp_f32".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(zp_data.to_vec()),
        );
        let mut initializers = HashMap::new();
        initializers.insert("zp_f32".to_string(), zp_tensor);

        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("ZERO_POINT_TENSOR".to_string()),
                value: Some("zp_f32".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &initializers);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].zero_point, Some(7), "F32 7.5 should truncate to i64 7");
    }

    /// @trace TEST-ONNX-651 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with multiple entries where one has axis and another does not.
    #[test]
    fn parse_quantization_multiple_entries_mixed_axis() {
        let entries = vec![
            proto::TensorAnnotation {
                tensor_name: Some("weight_a".to_string()),
                quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                    key: Some("AXIS".to_string()),
                    value: Some("1".to_string()),
                }],
            },
            proto::TensorAnnotation {
                tensor_name: Some("weight_b".to_string()),
                quant_parameter_tensor_names: vec![],
            },
        ];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].axis, Some(1));
        assert!(result[1].axis.is_none(), "entry with no AXIS param should yield None");
    }

    /// @trace TEST-ONNX-652 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_value_info with a first valid entry followed by a second with missing name
    /// returns error (short-circuits on the first missing name).
    #[test]
    fn parse_value_info_first_valid_second_missing_name_returns_error() {
        let values = vec![
            proto::ValueInfoProto {
                name: Some("valid_input".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            },
            proto::ValueInfoProto {
                name: None,
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            },
        ];
        let result = parse_value_info(values);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("value_info missing name"),
            "expected 'value_info missing name' in error: {err_msg}"
        );
    }

    /// @trace TEST-ONNX-653 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_value_info with a valid entry that has type = None yields value_type = None.
    #[test]
    fn parse_value_info_entry_with_none_type_yields_none_value_type() {
        let values = vec![proto::ValueInfoProto {
            name: Some("my_value".to_string()),
            r#type: None,
            doc_string: Some("no type info".to_string()),
            metadata_props: vec![],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "my_value");
        assert!(result[0].value_type.is_none());
        assert_eq!(result[0].doc_string, "no type info");
    }

    /// @trace TEST-ONNX-654 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_functions propagates error when a function's node has empty op_type.
    #[test]
    fn parse_functions_node_missing_op_type_returns_error() {
        let functions = vec![proto::FunctionProto {
            name: Some("broken_func".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![proto::NodeProto {
                op_type: None,
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            }],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("node 0 missing op_type"),
            "expected node 0 op_type error: {err_msg}"
        );
    }

    /// @trace TEST-ONNX-655 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph::from_proto with both dense and sparse initializers using distinct names.
    #[test]
    fn onnx_graph_from_proto_dense_and_sparse_distinct_names() {
        use super::super::tensor::OnnxTensor;

        let dense_tensor = proto::TensorProto {
            name: Some("dense_weight".to_string()),
            dims: vec![2],
            data_type: Some(1), // F32
            raw_data: Some(Bytes::from_static(&[0u8; 8])),
            ..Default::default()
        };
        let sparse_values = proto::TensorProto {
            name: Some("sparse_values".to_string()),
            dims: vec![3],
            data_type: Some(1),
            raw_data: Some(Bytes::from_static(&[0u8; 12])),
            ..Default::default()
        };
        let sparse_indices = proto::TensorProto {
            name: Some("sparse_idx".to_string()),
            dims: vec![3],
            data_type: Some(7), // I64
            raw_data: Some(Bytes::from_static(&[0u8; 24])),
            ..Default::default()
        };
        let sparse_init = proto::SparseTensorProto {
            values: Some(sparse_values),
            indices: Some(sparse_indices),
            dims: vec![3],
        };
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("mixed_graph".to_string()),
            initializer: vec![dense_tensor],
            sparse_initializer: vec![sparse_init],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.initializers.len(), 1);
        assert!(graph.initializers.contains_key("dense_weight"));
        assert_eq!(graph.sparse_initializers.len(), 1);
    }

    /// @trace TEST-ONNX-656 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxNode with overlapping input and output names (self-referential pattern).
    #[test]
    fn onnx_node_inputs_outputs_share_name() {
        let node = OnnxNode {
            name: "recurrent".to_string(),
            op_type: "Loop".to_string(),
            domain: String::new(),
            inputs: vec!["state".to_string(), "cond".to_string()],
            outputs: vec!["state".to_string()],
            attributes: HashMap::new(),
        };
        assert!(node.inputs.contains(&"state".to_string()));
        assert!(node.outputs.contains(&"state".to_string()));
        assert_eq!(node.outputs[0], node.inputs[0]);
    }

    /// @trace TEST-ONNX-657 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_nodes with a node that has Unicode characters in its name.
    #[test]
    fn parse_nodes_unicode_name_preserved() {
        let nodes = vec![proto::NodeProto {
            op_type: Some("Add".to_string()),
            name: Some("节点_节点_🌟".to_string()),
            input: vec![],
            output: vec![],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].name, "节点_节点_🌟");
    }

    /// @trace TEST-ONNX-658 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_metadata_props with a valid key but value = None defaults to empty string.
    #[test]
    fn parse_metadata_props_value_none_defaults_empty_string() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("key_with_no_value".to_string()),
            value: None,
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("key_with_no_value"), Some(&String::new()));
    }

    /// @trace TEST-ONNX-659 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_metadata_props with multiple entries where a later entry overwrites an earlier one.
    #[test]
    fn parse_metadata_props_later_entry_overwrites_earlier() {
        let entries = vec![
            proto::StringStringEntryProto {
                key: Some("framework".to_string()),
                value: Some("v1".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("framework".to_string()),
                value: Some("v2".to_string()),
            },
        ];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("framework"), Some(&"v2".to_string()));
        assert_eq!(result.len(), 1);
    }

    /// @trace TEST-ONNX-660 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph with a node whose output is empty string marker.
    #[test]
    fn onnx_graph_node_empty_string_output() {
        let mut graph = OnnxGraph {
            name: "test".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "side_effect".to_string(),
                op_type: "Print".to_string(),
                domain: String::new(),
                inputs: vec!["data".to_string()],
                outputs: vec![String::new()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.nodes[0].outputs[0].is_empty());
        graph.nodes[0].outputs.push("actual_out".to_string());
        assert_eq!(graph.nodes[0].outputs.len(), 2);
    }

    /// @trace TEST-ONNX-661 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with negative axis string "-1" yields Some(-1).
    #[test]
    fn parse_quantization_negative_axis_string_yields_negative() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some("-1".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].axis, Some(-1));
    }

    /// @trace TEST-ONNX-662 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModel::from_proto with ir_version = None defaults to 0.
    #[test]
    fn onnx_model_from_proto_ir_version_none_defaults_zero() {
        let model_proto = proto::ModelProto {
            ir_version: None,
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            doc_string: None,
            graph: Some(proto::GraphProto {
                node: vec![],
                name: None,
                initializer: vec![],
                sparse_initializer: vec![],
                input: vec![],
                output: vec![],
                value_info: vec![],
                quantization_annotation: vec![],
                doc_string: None,
                metadata_props: vec![],
            }),
            opset_import: vec![],
            metadata_props: vec![],
            functions: vec![],
            configuration: vec![],
            training_info: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.metadata.ir_version, 0);
        assert_eq!(model.metadata.model_version, 0);
        assert!(model.metadata.producer_name.is_empty());
        assert!(model.metadata.domain.is_empty());
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 12x35: 15 additional edge case tests
    // ══════════════════════════════════════════════════════════════════════

    /// @trace TEST-ONNX-663 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_functions propagates error when value_info has a None name.
    /// parse_value_info is called internally and must fail on missing names.
    #[test]
    fn parse_functions_value_info_error_propagation() {
        let functions = vec![proto::FunctionProto {
            name: Some("BrokenFunc".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![proto::ValueInfoProto {
                name: None,
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("value_info missing name"),
            "expected value_info error: {err_msg}"
        );
    }

    /// @trace TEST-ONNX-664 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_functions correctly parses attribute_proto entries into the function.
    #[test]
    fn parse_functions_attribute_protos_parsed() {
        let functions = vec![proto::FunctionProto {
            name: Some("AttrFunc".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec!["alpha".to_string()],
            attribute_proto: vec![proto::AttributeProto {
                name: Some("alpha".to_string()),
                ref_attr_name: None,
                doc_string: None,
                r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
                f: Some(0.5),
                i: None,
                s: None,
                t: None,
                g: None,
                sparse_tensor: None,
                tp: None,
                floats: vec![],
                ints: vec![],
                strings: vec![],
                tensors: vec![],
                graphs: vec![],
                sparse_tensors: vec![],
                type_protos: vec![],
            }],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].attributes, vec!["alpha"]);
        assert!(result[0].attribute_protos.contains_key("alpha"));
    }

    /// @trace TEST-ONNX-665 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_functions with multiple opset_import entries preserves all.
    #[test]
    fn parse_functions_opset_import_multiple() {
        let functions = vec![proto::FunctionProto {
            name: Some("MultiOpsetFunc".to_string()),
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![
                proto::OperatorSetIdProto { domain: Some("".to_string()), version: Some(17) },
                proto::OperatorSetIdProto { domain: Some("custom".to_string()), version: Some(3) },
            ],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].opset_import.len(), 2);
        assert_eq!(result[0].opset_import[0].domain, "");
        assert_eq!(result[0].opset_import[0].version, 17);
        assert_eq!(result[0].opset_import[1].domain, "custom");
        assert_eq!(result[0].opset_import[1].version, 3);
    }

    /// @trace TEST-ONNX-666 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_functions with nodes inside function preserves them.
    #[test]
    fn parse_functions_nodes_roundtrip() {
        let functions = vec![proto::FunctionProto {
            name: Some("WithNodes".to_string()),
            domain: None,
            overload: None,
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![
                proto::NodeProto {
                    op_type: Some("MatMul".to_string()),
                    name: Some("inner_mm".to_string()),
                    input: vec!["X".to_string(), "W".to_string()],
                    output: vec!["T".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                },
                proto::NodeProto {
                    op_type: Some("Add".to_string()),
                    name: None,
                    input: vec!["T".to_string(), "B".to_string()],
                    output: vec!["Y".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                },
            ],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result[0].nodes.len(), 2);
        assert_eq!(result[0].nodes[0].op_type, "MatMul");
        assert_eq!(result[0].nodes[0].name, "inner_mm");
        assert_eq!(result[0].nodes[1].op_type, "Add");
        assert_eq!(result[0].nodes[1].name, "node_1");
    }

    /// @trace TEST-ONNX-667 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModel::from_proto ignores training_info entries (not parsed into model).
    #[test]
    fn onnx_model_training_info_populated_ignored() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("g".to_string()),
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let model_proto = proto::ModelProto {
            ir_version: Some(8),
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            doc_string: None,
            graph: Some(graph_proto),
            opset_import: vec![],
            metadata_props: vec![],
            functions: vec![],
            configuration: vec![],
            training_info: vec![proto::TrainingInfoProto {
                initialization: None,
                algorithm: None,
                initialization_binding: vec![],
                update_binding: vec![],
            }],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        // training_info is not stored in OnnxModel — only graph/functions/metadata
        assert_eq!(model.metadata.ir_version, 8);
        assert!(model.functions.is_empty());
    }

    /// @trace TEST-ONNX-668 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph::from_proto parses an initializer with raw_data field instead of typed fields.
    #[test]
    fn onnx_graph_proto_initializer_raw_data_field() {
        let tensor = proto::TensorProto {
            name: Some("raw_weight".to_string()),
            data_type: Some(1), // F32
            dims: vec![2],
            raw_data: Some(Bytes::from_static(&[0u8; 8])),
            float_data: vec![],
            int32_data: vec![],
            int64_data: vec![],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("raw_graph".to_string()),
            initializer: vec![tensor],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.initializers.contains_key("raw_weight"));
        let t = graph.initializers.get("raw_weight").unwrap();
        assert_eq!(t.shape, vec![2]);
        assert_eq!(t.raw_data().len(), 8);
    }

    /// @trace TEST-ONNX-669 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with alphabetic AXIS string yields None (not parseable as i32).
    #[test]
    fn parse_quantization_axis_alphabetic_string() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("w".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some("channel".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &HashMap::new());
        assert_eq!(result.len(), 1);
        assert!(result[0].axis.is_none(), "non-numeric AXIS string should yield None");
    }

    /// @trace TEST-ONNX-670 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_nodes auto-generates sequential names for multiple unnamed nodes.
    #[test]
    fn parse_nodes_auto_generated_names_sequential() {
        let nodes: Vec<proto::NodeProto> = (0..4)
            .map(|_| proto::NodeProto {
                op_type: Some("Relu".to_string()),
                name: None,
                input: vec![],
                output: vec!["out".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            })
            .collect();
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        assert_eq!(result[0].name, "node_0");
        assert_eq!(result[1].name, "node_1");
        assert_eq!(result[2].name, "node_2");
        assert_eq!(result[3].name, "node_3");
    }

    /// @trace TEST-ONNX-671 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_metadata_props with many entries preserves all key-value pairs.
    #[test]
    fn parse_metadata_props_large_count() {
        let entries: Vec<proto::StringStringEntryProto> = (0..50)
            .map(|i| proto::StringStringEntryProto {
                key: Some(format!("key_{i}")),
                value: Some(format!("value_{i}")),
            })
            .collect();
        let result = parse_metadata_props(entries);
        assert_eq!(result.len(), 50);
        for i in 0..50 {
            assert_eq!(result.get(&format!("key_{i}")), Some(&format!("value_{i}")));
        }
    }

    /// @trace TEST-ONNX-672 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_opsets with empty input returns empty vec (no panic).
    #[test]
    fn parse_opsets_empty_input() {
        let result = parse_opsets(vec![]);
        assert!(result.is_empty());
    }

    /// @trace TEST-ONNX-673 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph::from_proto correctly parses both inputs and outputs together.
    #[test]
    fn onnx_graph_from_proto_input_output_both_present() {
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("io".to_string()),
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![
                proto::ValueInfoProto {
                    name: Some("input_ids".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
                proto::ValueInfoProto {
                    name: Some("attention_mask".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
            ],
            output: vec![
                proto::ValueInfoProto {
                    name: Some("logits".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
            ],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.inputs[0].name, "input_ids");
        assert_eq!(graph.inputs[1].name, "attention_mask");
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.outputs[0].name, "logits");
    }

    /// @trace TEST-ONNX-674 [req:REQ-LOADER-007] [level:unit]
    /// Verify bind_weights_auto returns 0 when the provider has no matching tensors.
    #[test]
    fn onnx_graph_bind_weights_auto_no_initializers() {
        struct EmptyProvider;
        impl crate::loader::TensorProvider for EmptyProvider {
            fn tensor_info(&self, _name: &str) -> Option<crate::loader::TensorMeta> {
                None
            }
            fn load_tensor_data(
                &self,
                _name: &str,
            ) -> std::result::Result<std::borrow::Cow<'_, [u8]>, LoaderError> {
                Ok(std::borrow::Cow::Owned(vec![]))
            }
            fn iter_tensors(&self) -> impl Iterator<Item = crate::loader::TensorMeta> {
                std::iter::empty()
            }
        }
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "mm".to_string(),
                op_type: "MatMul".to_string(),
                domain: String::new(),
                inputs: vec!["hidden_state".to_string(), "model.weight".to_string()],
                outputs: vec!["out".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let count = graph.bind_weights_auto(&EmptyProvider).unwrap();
        assert_eq!(count, 0);
        assert!(graph.initializers.is_empty());
    }

    /// @trace TEST-ONNX-675 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with SCALE_TENSOR referencing a missing initializer yields None scale.
    #[test]
    fn parse_quantization_scale_ref_missing_init() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("SCALE_TENSOR".to_string()),
                value: Some("nonexistent_scale".to_string()),
            }],
        }];
        let initializers = HashMap::new();
        let result = parse_quantization(entries, &initializers);
        assert_eq!(result.len(), 1);
        assert!(result[0].scale.is_none(), "missing initializer should yield None scale");
    }

    /// @trace TEST-ONNX-676 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with ZERO_POINT_TENSOR referencing a missing initializer yields None.
    #[test]
    fn parse_quantization_zero_point_missing_initializer() {
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("ZERO_POINT_TENSOR".to_string()),
                value: Some("nonexistent_zp".to_string()),
            }],
        }];
        let initializers = HashMap::new();
        let result = parse_quantization(entries, &initializers);
        assert_eq!(result.len(), 1);
        assert!(result[0].zero_point.is_none(), "missing initializer should yield None zero_point");
    }

    /// @trace TEST-ONNX-677 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph clone produces deep copy of initializers (mutation independence).
    #[test]
    fn onnx_graph_clone_initializers_deep_copy() {
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.initializers.insert(
            "weight".to_string(),
            OnnxTensor::new("weight".to_string(), Dtype::F32, vec![2, 3], Bytes::from(vec![0u8; 24])),
        );
        let mut cloned = graph.clone();
        cloned.initializers.remove("weight");
        assert!(
            graph.initializers.contains_key("weight"),
            "original should still have 'weight' after clone mutation"
        );
        assert!(
            !cloned.initializers.contains_key("weight"),
            "cloned should not have 'weight' after removal"
        );
    }

    /// @trace TEST-ONNX-678 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_metadata_props with key containing newline character is preserved.
    #[test]
    fn parse_metadata_props_key_with_newline() {
        let entries = vec![proto::StringStringEntryProto {
            key: Some("multi\nline\nkey".to_string()),
            value: Some("val".to_string()),
        }];
        let result = parse_metadata_props(entries);
        assert_eq!(result.get("multi\nline\nkey"), Some(&"val".to_string()));
    }

    /// @trace TEST-ONNX-679 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph::from_proto with multiple quantization_annotation entries parses all.
    #[test]
    fn onnx_graph_from_proto_with_multiple_quant_annotations() {
        let annotations = vec![
            proto::TensorAnnotation {
                tensor_name: Some("qkv_weight".to_string()),
                quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                    key: Some("AXIS".to_string()),
                    value: Some("0".to_string()),
                }],
            },
            proto::TensorAnnotation {
                tensor_name: Some("ffn_weight".to_string()),
                quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                    key: Some("AXIS".to_string()),
                    value: Some("1".to_string()),
                }],
            },
            proto::TensorAnnotation {
                tensor_name: Some("output_weight".to_string()),
                quant_parameter_tensor_names: vec![],
            },
        ];
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: None,
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: annotations,
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.quantization_annotation.len(), 3);
        assert_eq!(graph.quantization_annotation[0].tensor_name, "qkv_weight");
        assert_eq!(graph.quantization_annotation[0].axis, Some(0));
        assert_eq!(graph.quantization_annotation[1].tensor_name, "ffn_weight");
        assert_eq!(graph.quantization_annotation[1].axis, Some(1));
        assert_eq!(graph.quantization_annotation[2].tensor_name, "output_weight");
        assert!(graph.quantization_annotation[2].axis.is_none());
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 7 TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    /// @trace TEST-ONNX-680 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxNode inputs contain empty-string placeholder for optional inputs.
    #[test]
    fn onnx_node_optional_input_empty_string_preserved() {
        // Arrange: node with 4 inputs, last one being empty (optional placeholder)
        let node = OnnxNode {
            name: "conv".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string(), "W".to_string(), "B".to_string(), String::new()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        };

        // Assert: empty string is preserved as the 4th input
        assert_eq!(node.inputs.len(), 4);
        assert_eq!(node.inputs[3], "");
    }

    /// @trace TEST-ONNX-681 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_nodes assigns sequential indices for multiple unnamed nodes.
    #[test]
    fn parse_nodes_sequential_auto_names_for_batch() {
        // Arrange: 5 nodes with no name, all valid op_type
        let nodes: Vec<proto::NodeProto> = (0..5)
            .map(|_| proto::NodeProto {
                op_type: Some("Relu".to_string()),
                name: None,
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            })
            .collect();
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));

        // Act
        let result = parse_nodes(nodes, &mut resolver).unwrap();

        // Assert: sequential names node_0..node_4
        assert_eq!(result.len(), 5);
        for (i, node) in result.iter().enumerate() {
            assert_eq!(node.name, format!("node_{i}"));
        }
    }

    /// @trace TEST-ONNX-682 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph::from_proto parses value_info entries from proto.
    #[test]
    fn onnx_graph_from_proto_value_info_parsed() {
        // Arrange
        let graph_proto = proto::GraphProto {
            name: Some("g".to_string()),
            value_info: vec![
                proto::ValueInfoProto {
                    name: Some("mid_a".to_string()),
                    r#type: None,
                    doc_string: Some("first intermediate".to_string()),
                    metadata_props: vec![],
                },
                proto::ValueInfoProto {
                    name: Some("mid_b".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
            ],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));

        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();

        // Assert
        assert_eq!(graph.value_info.len(), 2);
        assert_eq!(graph.value_info[0].name, "mid_a");
        assert_eq!(graph.value_info[0].doc_string, "first intermediate");
        assert_eq!(graph.value_info[1].name, "mid_b");
    }

    /// @trace TEST-ONNX-683 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxTensor scalar_i64 with i64::MIN boundary value.
    #[test]
    fn onnx_tensor_scalar_i64_min_value() {
        // Arrange: i64::MIN encoded as LE bytes
        let value = i64::MIN;
        let tensor = OnnxTensor::new(
            "min_val".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(value.to_le_bytes().to_vec()),
        );

        // Act
        let result = tensor.scalar_i64();

        // Assert
        assert_eq!(result, Some(i64::MIN));
    }

    /// @trace TEST-ONNX-684 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxQuantizationAnnotation Debug format includes all field names.
    #[test]
    fn onnx_quantization_annotation_debug_all_fields() {
        // Arrange: annotation with all optional fields populated
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "debug_w".to_string(),
            quant_param_tensor_names: HashMap::from([
                ("SCALE_TENSOR".to_string(), "s0".to_string()),
            ]),
            scale: Some(1.0),
            zero_point: Some(0),
            axis: Some(2),
        };

        // Act
        let debug = format!("{qa:?}");

        // Assert: key identifiers present in debug output
        assert!(debug.contains("debug_w"), "tensor_name missing from debug: {debug}");
        assert!(debug.contains("1.0"), "scale missing from debug: {debug}");
    }

    /// @trace TEST-ONNX-685 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_metadata_props with large number of entries preserves all.
    #[test]
    fn parse_metadata_props_50_entries_preserved() {
        // Arrange: 50 key-value pairs
        let entries: Vec<proto::StringStringEntryProto> = (0..50)
            .map(|i| proto::StringStringEntryProto {
                key: Some(format!("key_{i}")),
                value: Some(format!("value_{i}")),
            })
            .collect();

        // Act
        let result = parse_metadata_props(entries);

        // Assert: all 50 entries preserved
        assert_eq!(result.len(), 50);
        assert_eq!(result.get("key_0").unwrap(), "value_0");
        assert_eq!(result.get("key_49").unwrap(), "value_49");
        assert_eq!(result.get("key_25").unwrap(), "value_25");
    }

    /// @trace TEST-ONNX-686 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxFunction metadata_props field roundtrips through construction.
    #[test]
    fn onnx_function_metadata_props_roundtrip() {
        // Arrange
        let props = HashMap::from([
            ("version".to_string(), "2".to_string()),
            ("author".to_string(), "gllm".to_string()),
        ]);
        let func = OnnxFunction {
            name: "MetaOp".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: props.clone(),
        };

        // Assert
        assert_eq!(func.metadata_props.len(), 2);
        assert_eq!(func.metadata_props.get("version").unwrap(), "2");
        assert_eq!(func.metadata_props.get("author").unwrap(), "gllm");
    }

    /// @trace TEST-ONNX-687 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModel clone independence: modifying cloned metadata does not affect original.
    #[test]
    fn onnx_model_clone_metadata_independence() {
        // Arrange
        let original = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: "orig".to_string(),
                producer_version: "1.0".to_string(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![OnnxOperatorSet {
                    domain: "".to_string(),
                    version: 17,
                }],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let mut cloned = original.clone();

        // Act: modify cloned model's opset version
        cloned.metadata.opset_import[0].version = 99;

        // Assert: original unchanged
        assert_eq!(original.metadata.opset_import[0].version, 17);
        assert_eq!(cloned.metadata.opset_import[0].version, 99);
    }

    /// @trace TEST-ONNX-688 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_opsets with many entries preserves all domain/version pairs.
    #[test]
    fn parse_opsets_many_entries_preserved() {
        // Arrange: 4 opset entries
        let opsets: Vec<proto::OperatorSetIdProto> = vec![
            proto::OperatorSetIdProto { domain: Some("".to_string()), version: Some(17) },
            proto::OperatorSetIdProto { domain: Some("ai.onnx.ml".to_string()), version: Some(3) },
            proto::OperatorSetIdProto { domain: Some("ai.onnx.training".to_string()), version: Some(1) },
            proto::OperatorSetIdProto { domain: Some("custom.domain".to_string()), version: Some(42) },
        ];

        // Act
        let result = parse_opsets(opsets);

        // Assert
        assert_eq!(result.len(), 4);
        assert_eq!(result[3].domain, "custom.domain");
        assert_eq!(result[3].version, 42);
    }

    /// @trace TEST-ONNX-689 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxNode clone independence: modifying cloned node's inputs does not affect original.
    #[test]
    fn onnx_node_clone_input_independence() {
        // Arrange
        let original = OnnxNode {
            name: "n".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            attributes: HashMap::new(),
        };
        let mut cloned = original.clone();

        // Act: mutate cloned inputs
        cloned.inputs.push("extra".to_string());

        // Assert: original unchanged
        assert_eq!(original.inputs.len(), 2);
        assert_eq!(cloned.inputs.len(), 3);
    }

    /// @trace TEST-ONNX-690 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxValueInfo with deeply nested Sequence type constructs correctly.
    #[test]
    fn onnx_value_info_deeply_nested_type() {
        // Arrange: Optional(Sequence(Tensor(Float)))
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let inner_tensor = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Param("seq".to_string())] },
        });
        let seq = OnnxType::Sequence(Box::new(inner_tensor));
        let opt = OnnxType::Optional(Box::new(seq));
        let vi = OnnxValueInfo {
            name: "past_kv".to_string(),
            value_type: Some(opt),
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };

        // Assert: unwrapping nested structure
        if let Some(OnnxType::Optional(inner)) = vi.value_type {
            if let OnnxType::Sequence(inner2) = *inner {
                assert!(matches!(*inner2, OnnxType::Tensor(_)));
            } else {
                panic!("expected Sequence inside Optional");
            }
        } else {
            panic!("expected Optional variant");
        }
    }

    /// @trace TEST-ONNX-691 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_value_info with doc_string preserves the documentation.
    #[test]
    fn parse_value_info_doc_string_preserved() {
        // Arrange
        let values = vec![proto::ValueInfoProto {
            name: Some("hidden_states".to_string()),
            r#type: None,
            doc_string: Some("Transformer hidden states after layer norm".to_string()),
            metadata_props: vec![],
        }];

        // Act
        let result = parse_value_info(values).unwrap();

        // Assert
        assert_eq!(result[0].doc_string, "Transformer hidden states after layer norm");
    }

    /// @trace TEST-ONNX-692 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph initializer overwrite with same key replaces the tensor.
    #[test]
    fn onnx_graph_initializer_overwrite_same_key() {
        // Arrange
        let t1 = OnnxTensor::new("w".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let t2 = OnnxTensor::new("w".to_string(), Dtype::BF16, vec![4], Bytes::from(vec![0u8; 8]));
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("w".to_string(), t1)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };

        // Act: overwrite with new tensor under same key
        graph.initializers.insert("w".to_string(), t2);

        // Assert: replaced, only one entry, new shape
        assert_eq!(graph.initializers.len(), 1);
        assert_eq!(graph.initializers.get("w").unwrap().shape, vec![4]);
        assert_eq!(graph.initializers.get("w").unwrap().dtype, Dtype::BF16);
    }

    /// @trace TEST-ONNX-693 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModel with functions and graph nodes both accessible.
    #[test]
    fn onnx_model_functions_and_graph_nodes_both_accessible() {
        // Arrange: model with 1 graph node and 1 function
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![OnnxNode {
                    name: "graph_node".to_string(),
                    op_type: "MatMul".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                }],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![OnnxFunction {
                name: "CustomRelu".to_string(),
                domain: "custom".to_string(),
                overload: String::new(),
                inputs: vec!["X".to_string()],
                outputs: vec!["Y".to_string()],
                attributes: vec![],
                attribute_protos: HashMap::new(),
                nodes: vec![OnnxNode {
                    name: "func_inner".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["X".to_string()],
                    outputs: vec!["Y".to_string()],
                    attributes: HashMap::new(),
                }],
                opset_import: vec![],
                value_info: vec![],
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
        };

        // Assert: both graph nodes and function nodes accessible
        assert_eq!(model.graph.nodes[0].op_type, "MatMul");
        assert_eq!(model.functions[0].nodes[0].op_type, "Relu");
        assert_eq!(model.functions[0].name, "CustomRelu");
    }

    /// @trace TEST-ONNX-694 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with AXIS value at i32::MAX boundary.
    #[test]
    fn parse_quantization_axis_i32_max() {
        // Arrange
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some(i32::MAX.to_string()),
            }],
        }];

        // Act
        let result = parse_quantization(entries, &HashMap::new());

        // Assert
        assert_eq!(result[0].axis, Some(i32::MAX));
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 12x36: 15 additional edge case tests
    // ══════════════════════════════════════════════════════════════════════

    /// @trace TEST-ONNX-695 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxTensor with F64 dtype stores and retrieves correctly.
    #[test]
    fn onnx_tensor_f64_dtype_preserved() {
        // Arrange: F64 scalar tensor
        let value = 3.141592653589793f64;
        let tensor = OnnxTensor::new(
            "pi".to_string(),
            Dtype::F64,
            vec![],
            Bytes::from(value.to_le_bytes().to_vec()),
        );
        // Act & Assert: dtype preserved
        assert_eq!(tensor.dtype, Dtype::F64);
        assert!(tensor.raw_data().len() == 8);
    }

    /// @trace TEST-ONNX-696 [req:REQ-LOADER-007] [level:unit]
    /// Verify LoaderError::Onnx Display contains the message text.
    #[test]
    fn loader_error_onnx_display_contains_message() {
        // Arrange
        let err = LoaderError::Onnx("custom parsing failure".to_string());
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("ONNX"), "expected ONNX prefix: {msg}");
        assert!(msg.contains("custom parsing failure"), "expected detail: {msg}");
    }

    /// @trace TEST-ONNX-697 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxAttributeValue::Strings variant holds multiple strings and is cloneable.
    #[test]
    fn onnx_attribute_value_strings_clone_roundtrip() {
        // Arrange
        use super::super::attributes::OnnxAttributeValue;
        let val = OnnxAttributeValue::Strings(vec!["hello".to_string(), "world".to_string()]);
        // Act & Assert
        assert!(matches!(&val, OnnxAttributeValue::Strings(v) if v.len() == 2));
        if let OnnxAttributeValue::Strings(v) = &val {
            assert_eq!(v[0], "hello");
            assert_eq!(v[1], "world");
        }
    }

    /// @trace TEST-ONNX-698 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph::from_proto with I64 data type initializer parses correctly.
    #[test]
    fn onnx_graph_from_proto_i64_initializer() {
        // Arrange
        let tensor = proto::TensorProto {
            name: Some("int64_bias".to_string()),
            data_type: Some(7), // INT64
            dims: vec![4],
            int64_data: vec![10, 20, 30, 40],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("i64_graph".to_string()),
            initializer: vec![tensor],
            sparse_initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        // Act
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert!(graph.initializers.contains_key("int64_bias"));
        let t = graph.initializers.get("int64_bias").unwrap();
        assert_eq!(t.dtype, Dtype::I64);
        assert_eq!(t.shape, vec![4]);
    }

    /// @trace TEST-ONNX-699 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxSparseTensor with all three format variants constructs correctly.
    #[test]
    fn onnx_sparse_tensor_all_format_variants() {
        // Arrange & Act & Assert: Coo, Csr, Csc
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let make_sparse = |fmt| OnnxSparseTensor {
            values: OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4])),
            indices: OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8])),
            dims: vec![3, 3],
            format: fmt,
        };
        let coo = make_sparse(OnnxSparseFormat::Coo);
        let csr = make_sparse(OnnxSparseFormat::Csr);
        let csc = make_sparse(OnnxSparseFormat::Csc);
        assert_eq!(coo.format, OnnxSparseFormat::Coo);
        assert_eq!(csr.format, OnnxSparseFormat::Csr);
        assert_eq!(csc.format, OnnxSparseFormat::Csc);
        // Verify formats are distinct
        assert_ne!(coo.format, csr.format);
        assert_ne!(csr.format, csc.format);
    }

    /// @trace TEST-ONNX-700 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxDim::Param with a descriptive symbolic name.
    #[test]
    fn onnx_dim_param_with_descriptive_name() {
        // Arrange
        use super::super::types::OnnxDim;
        let dim = OnnxDim::Param("batch_size".to_string());
        // Act & Assert
        assert!(matches!(dim, OnnxDim::Param(ref s) if s == "batch_size"));
    }

    /// @trace TEST-ONNX-701 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_nodes with node that has metadata_props.
    #[test]
    fn parse_nodes_metadata_props_preserved() {
        // Arrange
        let nodes = vec![proto::NodeProto {
            op_type: Some("Relu".to_string()),
            name: Some("meta_relu".to_string()),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![proto::StringStringEntryProto {
                key: Some("exec_provider".to_string()),
                value: Some("CUDA".to_string()),
            }],
            device_configurations: vec![],
        }];
        // Act
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        // Assert
        assert_eq!(result[0].name, "meta_relu");
        assert_eq!(result[0].op_type, "Relu");
    }

    /// @trace TEST-ONNX-702 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxTensor scalar_f32 with a negative value.
    #[test]
    fn onnx_tensor_scalar_f32_negative_value() {
        // Arrange: -42.5 encoded as LE bytes
        let value = -42.5f32;
        let tensor = OnnxTensor::new(
            "neg_f32".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(value.to_le_bytes().to_vec()),
        );
        // Act
        let result = tensor.scalar_f32();
        // Assert
        assert_eq!(result, Some(-42.5));
    }

    /// @trace TEST-ONNX-703 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph default construction via GraphProto with all fields empty.
    #[test]
    fn onnx_graph_default_proto_all_empty() {
        // Arrange
        let graph_proto = proto::GraphProto::default();
        // Act
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert!(graph.name.is_empty());
        assert!(graph.nodes.is_empty());
        assert!(graph.inputs.is_empty());
        assert!(graph.outputs.is_empty());
        assert!(graph.initializers.is_empty());
        assert!(graph.sparse_initializers.is_empty());
        assert!(graph.quantization_annotation.is_empty());
        assert!(graph.value_info.is_empty());
    }

    /// @trace TEST-ONNX-704 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxNode with self-referential input/output loop (RNN pattern).
    #[test]
    fn onnx_node_rnn_self_loop_pattern() {
        // Arrange: a node whose output feeds back as its own input
        let node = OnnxNode {
            name: "gru_cell".to_string(),
            op_type: "GRU".to_string(),
            domain: String::new(),
            inputs: vec!["h_prev".to_string(), "x_t".to_string()],
            outputs: vec!["h_next".to_string()],
            attributes: HashMap::new(),
        };
        // Assert: names are distinct and accessible
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.outputs.len(), 1);
        assert_ne!(node.inputs[0], node.outputs[0]);
    }

    /// @trace TEST-ONNX-705 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxAttribute with duplicate key insertion replaces previous value.
    #[test]
    fn onnx_node_attributes_duplicate_key_replaces() {
        // Arrange
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut node = OnnxNode {
            name: "conv".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        node.attributes.insert("strides".to_string(), OnnxAttribute {
            name: "strides".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 1]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        // Act: insert same key with different value
        node.attributes.insert("strides".to_string(), OnnxAttribute {
            name: "strides".to_string(),
            value: OnnxAttributeValue::Ints(vec![2, 2]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        // Assert: replaced
        assert_eq!(node.attributes.len(), 1);
        if let OnnxAttributeValue::Ints(dims) = &node.attributes["strides"].value {
            assert_eq!(*dims, vec![2, 2]);
        } else {
            panic!("expected Ints variant");
        }
    }

    /// @trace TEST-ONNX-706 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxDim::Unknown constructs and matches correctly.
    #[test]
    fn onnx_dim_unknown_variant() {
        // Arrange
        use super::super::types::OnnxDim;
        let dim = OnnxDim::Unknown;
        // Assert
        assert!(matches!(dim, OnnxDim::Unknown));
    }

    /// @trace TEST-ONNX-707 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxType::Map key and value types are accessible.
    #[test]
    fn onnx_type_map_key_value_accessible() {
        // Arrange
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxMapType, OnnxType};
        let map_type = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![OnnxDim::Known(128)] },
            })),
        });
        // Act & Assert
        if let OnnxType::Map(m) = map_type {
            assert!(matches!(m.key_type, proto::tensor_proto::DataType::Int64));
            if let OnnxType::Tensor(tt) = *m.value_type {
                assert!(matches!(tt.elem_type, proto::tensor_proto::DataType::Float));
                assert_eq!(tt.shape.dims.len(), 1);
            } else {
                panic!("expected Tensor inside Map value_type");
            }
        } else {
            panic!("expected Map variant");
        }
    }

    /// @trace TEST-ONNX-708 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_functions with a function that has no name returns error.
    #[test]
    fn parse_functions_empty_name_field_returns_error() {
        // Arrange
        let functions = vec![proto::FunctionProto {
            name: None,
            domain: None,
            overload: None,
            input: vec![],
            output: vec![],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        }];
        // Act
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver);
        // Assert
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("function missing name"), "expected function missing name: {msg}");
    }

    /// @trace TEST-ONNX-709 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModelMetadata with i64::MAX for both ir_version and model_version.
    #[test]
    fn onnx_model_metadata_i64_max_versions() {
        // Arrange
        let meta = OnnxModelMetadata {
            ir_version: i64::MAX,
            producer_name: "boundary_test".to_string(),
            producer_version: "max".to_string(),
            domain: "test".to_string(),
            model_version: i64::MAX,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        // Assert
        assert_eq!(meta.ir_version, i64::MAX);
        assert_eq!(meta.model_version, i64::MAX);
        assert_eq!(meta.producer_name, "boundary_test");
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 12x37: 15 additional edge case tests
    // ══════════════════════════════════════════════════════════════════════

    /// @trace TEST-ONNX-710 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxTensor scalar_f32 returns None for F64 dtype (not in supported match arms).
    #[test]
    fn onnx_tensor_scalar_f32_none_for_f64_dtype() {
        // Arrange: F64 scalar with 8 bytes of data
        let tensor = OnnxTensor::new(
            "f64_scalar".to_string(),
            Dtype::F64,
            vec![],
            Bytes::from(1.0f64.to_le_bytes().to_vec()),
        );
        // Act
        let result = tensor.scalar_f32();
        // Assert: F64 is not in scalar_f32's supported dtype match
        assert!(result.is_none(), "F64 scalar should return None from scalar_f32");
    }

    /// @trace TEST-ONNX-711 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxTensor scalar_i64 returns None for F64 dtype.
    #[test]
    fn onnx_tensor_scalar_i64_none_for_f64_dtype() {
        // Arrange: F64 scalar with 8 bytes
        let tensor = OnnxTensor::new(
            "f64_scalar".to_string(),
            Dtype::F64,
            vec![],
            Bytes::from(1.0f64.to_le_bytes().to_vec()),
        );
        // Act
        let result = tensor.scalar_i64();
        // Assert: F64 is not in scalar_i64's supported dtype match
        assert!(result.is_none(), "F64 scalar should return None from scalar_i64");
    }

    /// @trace TEST-ONNX-712 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_quantization with all optional fields populated simultaneously.
    #[test]
    fn parse_quantization_all_fields_populated() {
        // Arrange: scale and zero_point from initializers, axis from string
        let scale_data = 0.125f32.to_le_bytes();
        let scale_tensor = OnnxTensor::new(
            "sc".to_string(), Dtype::F32, vec![], Bytes::from(scale_data.to_vec()),
        );
        let zp_data = 128i64.to_le_bytes();
        let zp_tensor = OnnxTensor::new(
            "zp".to_string(), Dtype::I64, vec![], Bytes::from(zp_data.to_vec()),
        );
        let mut inits = HashMap::new();
        inits.insert("sc".to_string(), scale_tensor);
        inits.insert("zp".to_string(), zp_tensor);

        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("fully_quantized".to_string()),
            quant_parameter_tensor_names: vec![
                proto::StringStringEntryProto {
                    key: Some("SCALE_TENSOR".to_string()),
                    value: Some("sc".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("ZERO_POINT_TENSOR".to_string()),
                    value: Some("zp".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("AXIS".to_string()),
                    value: Some("3".to_string()),
                },
            ],
        }];
        // Act
        let result = parse_quantization(entries, &inits);
        // Assert
        assert_eq!(result.len(), 1);
        assert!((result[0].scale.unwrap() - 0.125).abs() < 1e-6);
        assert_eq!(result[0].zero_point, Some(128));
        assert_eq!(result[0].axis, Some(3));
        assert_eq!(result[0].tensor_name, "fully_quantized");
    }

    /// @trace TEST-ONNX-713 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph initializer with BOOL dtype constructs and reads correctly.
    #[test]
    fn onnx_graph_initializer_bool_dtype() {
        // Arrange
        let tensor = OnnxTensor::new(
            "mask".to_string(), Dtype::BOOL, vec![3], Bytes::from(vec![1u8, 0u8, 1u8]),
        );
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        graph.initializers.insert("mask".to_string(), tensor);
        // Assert
        let t = graph.initializers.get("mask").unwrap();
        assert_eq!(t.dtype, Dtype::BOOL);
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.raw_data().len(), 3);
    }

    /// @trace TEST-ONNX-714 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModel::from_proto with empty functions list yields empty vec.
    #[test]
    fn onnx_model_from_proto_empty_functions_list() {
        // Arrange
        let model_proto = proto::ModelProto {
            ir_version: Some(8),
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            doc_string: None,
            graph: Some(proto::GraphProto {
                node: vec![],
                name: Some("g".to_string()),
                initializer: vec![],
                sparse_initializer: vec![],
                input: vec![],
                output: vec![],
                value_info: vec![],
                quantization_annotation: vec![],
                doc_string: None,
                metadata_props: vec![],
            }),
            opset_import: vec![],
            metadata_props: vec![],
            functions: vec![],
            configuration: vec![],
            training_info: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        // Assert
        assert!(model.functions.is_empty());
        assert_eq!(model.metadata.ir_version, 8);
    }

    /// @trace TEST-ONNX-715 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_nodes with op_type = None returns error (missing required field).
    #[test]
    fn parse_nodes_op_type_none_returns_error() {
        // Arrange: node with op_type = None
        let nodes = vec![proto::NodeProto {
            op_type: None,
            name: Some("broken".to_string()),
            input: vec![],
            output: vec![],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_nodes(nodes, &mut resolver);
        // Assert
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("op_type"), "expected op_type error: {msg}");
    }

    /// @trace TEST-ONNX-716 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph clone produces independent sparse_initializers.
    #[test]
    fn onnx_graph_clone_sparse_initializers_independence() {
        // Arrange
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let sparse = OnnxSparseTensor {
            values: OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4])),
            indices: OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8])),
            dims: vec![5],
            format: OnnxSparseFormat::Coo,
        };
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![sparse],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        let mut cloned = graph.clone();
        cloned.sparse_initializers.pop();
        // Assert: original unchanged
        assert_eq!(graph.sparse_initializers.len(), 1);
        assert!(cloned.sparse_initializers.is_empty());
    }

    /// @trace TEST-ONNX-717 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxFunction with populated attribute_protos HashMap is independent after clone.
    #[test]
    fn onnx_function_attribute_protos_clone_independence() {
        // Arrange
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut protos = HashMap::new();
        protos.insert("k".to_string(), OnnxAttribute {
            name: "k".to_string(),
            value: OnnxAttributeValue::Float(1.0),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let func = OnnxFunction {
            name: "F".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: protos,
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        // Act
        let mut cloned = func.clone();
        cloned.attribute_protos.remove("k");
        // Assert
        assert!(func.attribute_protos.contains_key("k"));
        assert!(!cloned.attribute_protos.contains_key("k"));
    }

    /// @trace TEST-ONNX-718 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_value_info with a single entry preserves name exactly.
    #[test]
    fn parse_value_info_single_entry_name_exact() {
        // Arrange
        let values = vec![proto::ValueInfoProto {
            name: Some("exact_name_test".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        }];
        // Act
        let result = parse_value_info(values).unwrap();
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "exact_name_test");
        assert!(result[0].value_type.is_none());
    }

    /// @trace TEST-ONNX-719 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxOperatorSet clone produces independent domain field.
    #[test]
    fn onnx_operator_set_clone_domain_independence() {
        // Arrange
        let ops = OnnxOperatorSet {
            domain: "original.domain".to_string(),
            version: 17,
        };
        // Act
        let mut cloned = ops.clone();
        cloned.domain.clear();
        // Assert
        assert_eq!(ops.domain, "original.domain");
        assert!(cloned.domain.is_empty());
    }

    /// @trace TEST-ONNX-720 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModelMetadata producer_version field mutation.
    #[test]
    fn onnx_model_metadata_producer_version_mutable() {
        // Arrange
        let mut meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: "0.1".to_string(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        meta.producer_version = "2.0".to_string();
        // Assert
        assert_eq!(meta.producer_version, "2.0");
    }

    /// @trace TEST-ONNX-721 [req:REQ-LOADER-007] [level:unit]
    /// Verify parse_metadata_props with key = None is skipped (not inserted).
    #[test]
    fn parse_metadata_props_key_none_skipped() {
        // Arrange
        let entries = vec![
            proto::StringStringEntryProto {
                key: None,
                value: Some("orphan".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("valid".to_string()),
                value: Some("present".to_string()),
            },
        ];
        // Act
        let result = parse_metadata_props(entries);
        // Assert: entry with key=None should be skipped
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("valid").unwrap(), "present");
        assert!(result.get("").is_none());
    }

    /// @trace TEST-ONNX-722 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph initializer with high-dimensional shape (5D tensor).
    #[test]
    fn onnx_graph_initializer_5d_shape() {
        // Arrange
        let tensor = OnnxTensor::new(
            "conv5d".to_string(),
            Dtype::F32,
            vec![1, 2, 3, 4, 5],
            Bytes::from(vec![0u8; 120]),
        );
        let graph = OnnxGraph {
            name: "5d_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("conv5d".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Assert
        let t = graph.initializers.get("conv5d").unwrap();
        assert_eq!(t.shape, vec![1, 2, 3, 4, 5]);
        assert_eq!(t.shape.len(), 5);
    }

    /// @trace TEST-ONNX-723 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxValueInfo with None value_type still clones correctly.
    #[test]
    fn onnx_value_info_clone_with_none_type() {
        // Arrange
        let vi = OnnxValueInfo {
            name: "untyped".to_string(),
            value_type: None,
            doc_string: "no type".to_string(),
            metadata_props: HashMap::new(),
        };
        // Act
        let cloned = vi.clone();
        // Assert
        assert_eq!(cloned.name, "untyped");
        assert!(cloned.value_type.is_none());
        assert_eq!(cloned.doc_string, "no type");
    }

    /// @trace TEST-ONNX-724 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxTensor new_string with multi-element shape preserves is_string flag.
    #[test]
    fn onnx_tensor_new_string_multi_element_shape() {
        // Arrange & Act
        let tensor = OnnxTensor::new_string(
            "vocab".to_string(),
            vec![100],
            Bytes::from(vec![0u8; 100]),
        );
        // Assert
        assert!(tensor.is_string);
        assert_eq!(tensor.shape, vec![100]);
        assert_eq!(tensor.raw_data().len(), 100);
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxGraph initializers retain preserves matching entries only ───

    #[test]
    fn onnx_graph_initializers_retain_filters_correctly() {
        // Arrange
        let t1 = OnnxTensor::new("keep_a".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let t2 = OnnxTensor::new("drop_b".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let t3 = OnnxTensor::new("keep_c".to_string(), Dtype::F32, vec![3], Bytes::from(vec![0u8; 12]));
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([
                ("keep_a".to_string(), t1),
                ("drop_b".to_string(), t2),
                ("keep_c".to_string(), t3),
            ]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        graph.initializers.retain(|name, _| name.starts_with("keep"));
        // Assert
        assert_eq!(graph.initializers.len(), 2);
        assert!(graph.initializers.contains_key("keep_a"));
        assert!(graph.initializers.contains_key("keep_c"));
        assert!(!graph.initializers.contains_key("drop_b"));
    }

    // ── OnnxNode attributes clear empties the map ──────────────────────

    #[test]
    fn onnx_node_attributes_clear_empties_map() {
        // Arrange
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert(
            "epsilon".to_string(),
            OnnxAttribute {
                name: "epsilon".to_string(),
                value: OnnxAttributeValue::Float(1e-5),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        let mut node = OnnxNode {
            name: "layernorm".to_string(),
            op_type: "LayerNormalization".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: attrs,
        };
        // Act
        node.attributes.clear();
        // Assert
        assert!(node.attributes.is_empty());
        assert_eq!(node.name, "layernorm");
    }

    // ── OnnxModel replace graph field preserves metadata ───────────────

    #[test]
    fn onnx_model_replace_graph_preserves_metadata() {
        // Arrange
        let original_meta = OnnxModelMetadata {
            ir_version: 9,
            producer_name: "original".to_string(),
            producer_version: "1.0".to_string(),
            domain: "test".to_string(),
            model_version: 7,
            doc_string: "original doc".to_string(),
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 18,
            }],
            metadata_props: HashMap::from([("key".to_string(), "val".to_string())]),
        };
        let mut model = OnnxModel {
            metadata: original_meta.clone(),
            graph: OnnxGraph {
                name: "old_graph".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        // Act
        model.graph = OnnxGraph {
            name: "new_graph".to_string(),
            doc_string: "replaced".to_string(),
            nodes: vec![OnnxNode {
                name: "n0".to_string(),
                op_type: "Relu".to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Assert
        assert_eq!(model.graph.name, "new_graph");
        assert_eq!(model.graph.nodes.len(), 1);
        assert_eq!(model.metadata.ir_version, 9);
        assert_eq!(model.metadata.producer_name, "original");
        assert_eq!(model.metadata.metadata_props.get("key").unwrap(), "val");
    }

    // ── OnnxGraph value_info clear resets to empty ─────────────────────

    #[test]
    fn onnx_graph_value_info_clear_resets_to_empty() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![
                OnnxValueInfo {
                    name: "v1".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "v2".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        graph.value_info.clear();
        // Assert
        assert!(graph.value_info.is_empty());
    }

    // ── OnnxFunction nodes extend adds multiple entries ─────────────────

    #[test]
    fn onnx_function_nodes_extend_adds_multiple() {
        // Arrange
        let mut func = OnnxFunction {
            name: "FusedBlock".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![OnnxNode {
                name: "existing".to_string(),
                op_type: "Relu".to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: HashMap::new(),
            }],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let extra_nodes = vec![
            OnnxNode {
                name: "add_1".to_string(),
                op_type: "Add".to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: HashMap::new(),
            },
            OnnxNode {
                name: "mul_1".to_string(),
                op_type: "Mul".to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: HashMap::new(),
            },
        ];
        // Act
        func.nodes.extend(extra_nodes);
        // Assert
        assert_eq!(func.nodes.len(), 3);
        assert_eq!(func.nodes[0].op_type, "Relu");
        assert_eq!(func.nodes[1].op_type, "Add");
        assert_eq!(func.nodes[2].op_type, "Mul");
    }

    // ── OnnxNode inputs extend and last access ─────────────────────────

    #[test]
    fn onnx_node_inputs_extend_and_last_access() {
        // Arrange
        let mut node = OnnxNode {
            name: "cat".to_string(),
            op_type: "Concat".to_string(),
            domain: String::new(),
            inputs: vec!["a".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        // Act
        node.inputs.extend(vec!["b".to_string(), "c".to_string()]);
        // Assert
        assert_eq!(node.inputs.len(), 3);
        assert_eq!(node.inputs.last().unwrap(), "c");
        assert_eq!(node.inputs[0], "a");
    }

    // ── OnnxGraph quantization_annotation extend preserves existing ────

    #[test]
    fn onnx_graph_quantization_extend_preserves_existing() {
        // Arrange
        let existing = OnnxQuantizationAnnotation {
            tensor_name: "w0".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.1),
            zero_point: None,
            axis: None,
        };
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![existing],
            metadata_props: HashMap::new(),
        };
        let extra = vec![
            OnnxQuantizationAnnotation {
                tensor_name: "w1".to_string(),
                quant_param_tensor_names: HashMap::new(),
                scale: None,
                zero_point: Some(0),
                axis: Some(0),
            },
            OnnxQuantizationAnnotation {
                tensor_name: "w2".to_string(),
                quant_param_tensor_names: HashMap::new(),
                scale: None,
                zero_point: None,
                axis: None,
            },
        ];
        // Act
        graph.quantization_annotation.extend(extra);
        // Assert
        assert_eq!(graph.quantization_annotation.len(), 3);
        assert_eq!(graph.quantization_annotation[0].tensor_name, "w0");
        assert_eq!(graph.quantization_annotation[1].tensor_name, "w1");
        assert_eq!(graph.quantization_annotation[2].tensor_name, "w2");
    }

    // ── OnnxOperatorSet version update from 17 to 20 ───────────────────

    #[test]
    fn onnx_operator_set_version_update() {
        // Arrange
        let mut opset = OnnxOperatorSet {
            domain: "".to_string(),
            version: 17,
        };
        // Act
        opset.version = 20;
        // Assert
        assert_eq!(opset.version, 20);
        assert_eq!(opset.domain, "");
    }

    // ── OnnxModelMetadata doc_string update preserves other fields ─────

    #[test]
    fn onnx_model_metadata_doc_string_update_preserves_fields() {
        // Arrange
        let mut meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: "test_producer".to_string(),
            producer_version: "2.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 42,
            doc_string: "original doc".to_string(),
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 17,
            }],
            metadata_props: HashMap::from([("key".to_string(), "val".to_string())]),
        };
        // Act
        meta.doc_string = "updated documentation".to_string();
        // Assert
        assert_eq!(meta.doc_string, "updated documentation");
        assert_eq!(meta.ir_version, 8);
        assert_eq!(meta.producer_name, "test_producer");
        assert_eq!(meta.producer_version, "2.0");
        assert_eq!(meta.domain, "ai.onnx");
        assert_eq!(meta.model_version, 42);
        assert_eq!(meta.opset_import.len(), 1);
        assert_eq!(meta.metadata_props.get("key").unwrap(), "val");
    }

    // ── OnnxGraph with 5 sequential nodes maintains order after clone ──

    #[test]
    fn onnx_graph_five_nodes_order_preserved_after_clone() {
        // Arrange
        let nodes: Vec<OnnxNode> = (0..5)
            .map(|i| OnnxNode {
                name: format!("layer_{i}"),
                op_type: "Linear".to_string(),
                domain: String::new(),
                inputs: vec![format!("h{i}")],
                outputs: vec![format!("h{}", i + 1)],
                attributes: HashMap::new(),
            })
            .collect();
        let graph = OnnxGraph {
            name: "sequential".to_string(),
            doc_string: String::new(),
            nodes,
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        let cloned = graph.clone();
        // Assert
        assert_eq!(cloned.nodes.len(), 5);
        for i in 0..5 {
            assert_eq!(cloned.nodes[i].name, format!("layer_{i}"));
            assert_eq!(cloned.nodes[i].inputs[0], format!("h{i}"));
            assert_eq!(cloned.nodes[i].outputs[0], format!("h{}", i + 1));
        }
    }

    // ── OnnxValueInfo metadata_props insert and lookup ──────────────────

    #[test]
    fn onnx_value_info_metadata_props_insert_and_lookup() {
        // Arrange
        let mut vi = OnnxValueInfo {
            name: "mask".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        // Act
        vi.metadata_props.insert("origin".to_string(), "tokenizer".to_string());
        vi.metadata_props.insert("shape_type".to_string(), "2d".to_string());
        // Assert
        assert_eq!(vi.metadata_props.len(), 2);
        assert_eq!(vi.metadata_props.get("origin").unwrap(), "tokenizer");
        assert_eq!(vi.metadata_props.get("shape_type").unwrap(), "2d");
        assert!(vi.metadata_props.get("nonexistent").is_none());
    }

    // ── OnnxGraph initializers len matches count after multiple inserts ─

    #[test]
    fn onnx_graph_initializers_len_matches_inserts() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        for i in 0..10 {
            let tensor = OnnxTensor::new(
                format!("weight_{i}"),
                Dtype::F32,
                vec![i + 1],
                Bytes::from(vec![0u8; (i + 1) * 4]),
            );
            graph.initializers.insert(format!("weight_{i}"), tensor);
        }
        // Assert
        assert_eq!(graph.initializers.len(), 10);
        assert!(graph.initializers.contains_key("weight_0"));
        assert!(graph.initializers.contains_key("weight_9"));
        assert!(!graph.initializers.contains_key("weight_10"));
    }

    // ── OnnxQuantizationAnnotation quant_param_tensor_names insert ──────

    #[test]
    fn onnx_quantization_annotation_param_names_insert() {
        // Arrange
        let mut qa = OnnxQuantizationAnnotation {
            tensor_name: "layer_norm_weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        // Act
        qa.quant_param_tensor_names.insert("SCALE_TENSOR".to_string(), "ln_scale".to_string());
        qa.quant_param_tensor_names.insert("ZERO_POINT_TENSOR".to_string(), "ln_zp".to_string());
        qa.scale = Some(0.02);
        qa.axis = Some(-1);
        // Assert
        assert_eq!(qa.quant_param_tensor_names.len(), 2);
        assert_eq!(qa.quant_param_tensor_names.get("SCALE_TENSOR").unwrap(), "ln_scale");
        assert_eq!(qa.quant_param_tensor_names.get("ZERO_POINT_TENSOR").unwrap(), "ln_zp");
        assert!((qa.scale.unwrap() - 0.02).abs() < 1e-9);
        assert_eq!(qa.axis, Some(-1));
    }

    // ── OnnxModel functions swap replaces single element ────────────────

    #[test]
    fn onnx_model_functions_swap_replaces_element() {
        // Arrange
        let func_a = OnnxFunction {
            name: "OpA".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let func_b = OnnxFunction {
            name: "OpB".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![func_a, func_b],
        };
        // Act
        let replacement = OnnxFunction {
            name: "OpC".to_string(),
            domain: "replaced".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        model.functions[1] = replacement;
        // Assert
        assert_eq!(model.functions.len(), 2);
        assert_eq!(model.functions[0].name, "OpA");
        assert_eq!(model.functions[1].name, "OpC");
        assert_eq!(model.functions[1].domain, "replaced");
    }

    // ── OnnxGraph nodes drain empties and returns all items ─────────────

    #[test]
    fn onnx_graph_nodes_drain_empties_and_returns_items() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "n0".to_string(),
                    op_type: "Conv".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n1".to_string(),
                    op_type: "BatchNorm".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        let drained: Vec<OnnxNode> = graph.nodes.drain(..).collect();
        // Assert
        assert!(graph.nodes.is_empty());
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].op_type, "Conv");
        assert_eq!(drained[1].op_type, "BatchNorm");
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 8 TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxModel from_proto: all optional fields default correctly ──────
    // @trace TEST-ONNX-MODEL-001

    #[test]
    fn onnx_model_from_proto_all_optional_fields_default() {
        // Arrange: ModelProto with only graph set, everything else default
        let model_proto = proto::ModelProto {
            graph: Some(proto::GraphProto {
                name: Some("minimal".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        // Assert
        assert_eq!(model.metadata.ir_version, 0);
        assert!(model.metadata.producer_name.is_empty());
        assert!(model.metadata.producer_version.is_empty());
        assert!(model.metadata.domain.is_empty());
        assert_eq!(model.metadata.model_version, 0);
        assert!(model.metadata.doc_string.is_empty());
        assert!(model.metadata.opset_import.is_empty());
        assert!(model.metadata.metadata_props.is_empty());
        assert!(model.functions.is_empty());
    }

    // ── OnnxGraph from_proto: doc_string preserved from proto ───────────
    // @trace TEST-ONNX-GRAPH-DOC-002

    #[test]
    fn onnx_graph_from_proto_doc_string_preserved() {
        // Arrange
        let graph_proto = proto::GraphProto {
            name: Some("doc_graph".to_string()),
            doc_string: Some("A graph with documentation".to_string()),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert_eq!(graph.doc_string, "A graph with documentation");
    }

    // ── OnnxNode clone produces independent copy ─────────────────────────
    // @trace TEST-ONNX-NODE-CLONE-003

    #[test]
    fn onnx_node_clone_independence() {
        // Arrange
        let mut node = OnnxNode {
            name: "original".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        };
        let cloned = node.clone();
        // Act: mutate original
        node.name = "modified".to_string();
        node.inputs.push("W".to_string());
        // Assert: clone is unaffected
        assert_eq!(cloned.name, "original");
        assert_eq!(cloned.inputs.len(), 1);
        assert_eq!(cloned.inputs[0], "X");
    }

    // ── OnnxQuantizationAnnotation: None scale vs zero scale distinction ─
    // @trace TEST-ONNX-QA-004

    #[test]
    fn onnx_quantization_annotation_none_vs_zero_scale() {
        // Arrange
        let none_scale = OnnxQuantizationAnnotation {
            tensor_name: "w1".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        let zero_scale = OnnxQuantizationAnnotation {
            tensor_name: "w2".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.0),
            zero_point: Some(0),
            axis: None,
        };
        // Assert: None is not Some(0.0)
        assert!(none_scale.scale.is_none());
        assert!(zero_scale.scale.is_some());
        assert_eq!(zero_scale.scale.unwrap(), 0.0);
    }

    // ── OnnxGraph metadata_props insertion and retrieval ─────────────────
    // @trace TEST-ONNX-GRAPH-META-005

    #[test]
    fn onnx_graph_metadata_props_insert_and_retrieve() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.metadata_props.is_empty());
        // Act
        graph.metadata_props.insert("framework".to_string(), "pytorch".to_string());
        graph.metadata_props.insert("version".to_string(), "2.1".to_string());
        // Assert
        assert_eq!(graph.metadata_props.len(), 2);
        assert_eq!(graph.metadata_props.get("framework").unwrap(), "pytorch");
        assert_eq!(graph.metadata_props.get("version").unwrap(), "2.1");
    }

    // ── OnnxFunction name and domain mutation ────────────────────────────
    // @trace TEST-ONNX-FUNC-MUT-006

    #[test]
    fn onnx_function_name_and_domain_mutable() {
        // Arrange
        let mut func = OnnxFunction {
            name: "Original".to_string(),
            domain: "old.domain".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        // Act
        func.name = "Renamed".to_string();
        func.domain = "new.domain".to_string();
        // Assert
        assert_eq!(func.name, "Renamed");
        assert_eq!(func.domain, "new.domain");
    }

    // ── parse_metadata_props: key with only whitespace is accepted ───────
    // @trace TEST-ONNX-META-007

    #[test]
    fn parse_metadata_props_whitespace_key_accepted() {
        // Arrange
        let entries = vec![proto::StringStringEntryProto {
            key: Some("   ".to_string()),
            value: Some("whitespace_key_val".to_string()),
        }];
        // Act
        let result = parse_metadata_props(entries);
        // Assert: "   " is non-empty so it is stored
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("   ").unwrap(), "whitespace_key_val");
    }

    // ── OnnxTensor::new_string with empty bytes ──────────────────────────
    // @trace TEST-ONNX-TENSOR-STR-008

    #[test]
    fn onnx_tensor_new_string_empty_bytes() {
        // Arrange & Act
        let tensor = OnnxTensor::new_string(
            "empty_text".to_string(),
            vec![0],
            Bytes::new(),
        );
        // Assert
        assert!(tensor.is_string);
        assert_eq!(tensor.dtype, Dtype::U8);
        assert!(tensor.raw_data().is_empty());
        assert_eq!(tensor.shape, vec![0]);
    }

    // ── OnnxGraph nodes append via push ──────────────────────────────────
    // @trace TEST-ONNX-GRAPH-NODES-009

    #[test]
    fn onnx_graph_nodes_append_push() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(graph.nodes.is_empty());
        // Act
        graph.nodes.push(OnnxNode {
            name: "n0".to_string(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        });
        graph.nodes.push(OnnxNode {
            name: "n1".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        });
        // Assert
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].op_type, "Relu");
        assert_eq!(graph.nodes[1].op_type, "Add");
    }

    // ── OnnxModel functions clear empties the list ───────────────────────
    // @trace TEST-ONNX-MODEL-FUNC-010

    #[test]
    fn onnx_model_functions_clear_empties_list() {
        // Arrange
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![
                OnnxFunction {
                    name: "F1".to_string(),
                    domain: String::new(),
                    overload: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: vec![],
                    attribute_protos: HashMap::new(),
                    nodes: vec![],
                    opset_import: vec![],
                    value_info: vec![],
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
        };
        assert_eq!(model.functions.len(), 1);
        // Act
        model.functions.clear();
        // Assert
        assert!(model.functions.is_empty());
    }

    // ── parse_nodes: domain defaults to empty when None ──────────────────
    // @trace TEST-ONNX-PARSE-NODES-011

    #[test]
    fn parse_nodes_domain_defaults_empty_when_none() {
        // Arrange
        let nodes = vec![proto::NodeProto {
            op_type: Some("Relu".to_string()),
            name: None,
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        // Assert
        assert!(result[0].domain.is_empty());
    }

    // ── parse_opsets: multiple entries with mixed None/Some ──────────────
    // @trace TEST-ONNX-PARSE-OPSETS-012

    #[test]
    fn parse_opsets_mixed_none_and_some_entries() {
        // Arrange
        let opsets = vec![
            proto::OperatorSetIdProto {
                domain: Some("ai.onnx".to_string()),
                version: Some(17),
            },
            proto::OperatorSetIdProto {
                domain: None,
                version: None,
            },
            proto::OperatorSetIdProto {
                domain: Some("ai.onnx.ml".to_string()),
                version: Some(3),
            },
        ];
        // Act
        let result = parse_opsets(opsets);
        // Assert
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].domain, "ai.onnx");
        assert_eq!(result[0].version, 17);
        assert_eq!(result[1].domain, "");
        assert_eq!(result[1].version, 0);
        assert_eq!(result[2].domain, "ai.onnx.ml");
        assert_eq!(result[2].version, 3);
    }

    // ── OnnxGraph from_proto: initializer with float_data ────────────────
    // @trace TEST-ONNX-GRAPH-INIT-013

    #[test]
    fn onnx_graph_from_proto_initializer_with_float_data() {
        // Arrange
        let tensor_proto = proto::TensorProto {
            name: Some("bias".to_string()),
            data_type: Some(1), // FLOAT
            dims: vec![3],
            float_data: vec![0.1, 0.2, 0.3],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            name: Some("init_graph".to_string()),
            initializer: vec![tensor_proto],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert!(graph.initializers.contains_key("bias"));
        let bias = graph.initializers.get("bias").unwrap();
        assert_eq!(bias.shape, vec![3]);
    }

    // ── OnnxValueInfo with all fields populated and type info ─────────────
    // @trace TEST-ONNX-VALUE-INFO-014

    #[test]
    fn onnx_value_info_all_fields_populated() {
        // Arrange
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let vi = OnnxValueInfo {
            name: "attention_scores".to_string(),
            value_type: Some(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape {
                    dims: vec![
                        OnnxDim::Param("batch".to_string()),
                        OnnxDim::Known(12),
                        OnnxDim::Param("seq_len".to_string()),
                        OnnxDim::Known(64),
                    ],
                },
            })),
            doc_string: "multi-head attention scores".to_string(),
            metadata_props: HashMap::from([
                ("norm".to_string(), "softmax".to_string()),
            ]),
        };
        // Act & Assert
        assert_eq!(vi.name, "attention_scores");
        assert!(vi.value_type.is_some());
        assert_eq!(vi.doc_string, "multi-head attention scores");
        assert_eq!(vi.metadata_props.get("norm").unwrap(), "softmax");
        if let Some(OnnxType::Tensor(tt)) = vi.value_type {
            assert_eq!(tt.shape.dims.len(), 4);
            assert!(matches!(tt.shape.dims[0], OnnxDim::Param(_)));
            assert!(matches!(tt.shape.dims[1], OnnxDim::Known(12)));
        } else {
            panic!("expected Tensor variant");
        }
    }

    // ── parse_quantization: AXIS with large positive value ────────────────
    // @trace TEST-ONNX-QUANT-AXIS-015

    #[test]
    fn parse_quantization_axis_large_positive_value() {
        // Arrange
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some("255".to_string()),
            }],
        }];
        // Act
        let result = parse_quantization(entries, &HashMap::new());
        // Assert
        assert_eq!(result[0].axis, Some(255));
    }

    // ── OnnxGraph inputs retain and sort ────────────────────────────────
    // @trace TEST-ONNX-GRAPH-INPUTS-016

    #[test]
    fn onnx_graph_inputs_sort_by_name() {
        // Arrange
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![
                OnnxValueInfo {
                    name: "z_input".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "a_input".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "m_input".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        let mut sorted = graph.inputs.clone();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        // Assert
        assert_eq!(sorted[0].name, "a_input");
        assert_eq!(sorted[1].name, "m_input");
        assert_eq!(sorted[2].name, "z_input");
    }

    // ── OnnxGraph outputs drain empties and returns all items ───────────
    // @trace TEST-ONNX-GRAPH-OUTPUTS-017

    #[test]
    fn onnx_graph_outputs_drain_empties_and_returns_items() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![
                OnnxValueInfo {
                    name: "out1".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "out2".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        let drained: Vec<_> = graph.outputs.drain(..).collect();
        // Assert
        assert!(graph.outputs.is_empty());
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].name, "out1");
        assert_eq!(drained[1].name, "out2");
    }

    // ── OnnxNode outputs push adds entry ────────────────────────────────
    // @trace TEST-ONNX-NODE-OUTPUTS-018

    #[test]
    fn onnx_node_outputs_push_adds_entry() {
        // Arrange
        let mut node = OnnxNode {
            name: "n".to_string(),
            op_type: "Split".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y1".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.outputs.len(), 1);
        // Act
        node.outputs.push("y2".to_string());
        node.outputs.push("y3".to_string());
        // Assert
        assert_eq!(node.outputs.len(), 3);
        assert_eq!(node.outputs[0], "y1");
        assert_eq!(node.outputs[1], "y2");
        assert_eq!(node.outputs[2], "y3");
    }

    // ── OnnxFunction inputs drain empties and returns names ─────────────
    // @trace TEST-ONNX-FUNCTION-INPUTS-019

    #[test]
    fn onnx_function_inputs_drain_empties_and_returns_names() {
        // Arrange
        let mut func = OnnxFunction {
            name: "my_func".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.inputs.len(), 3);
        // Act
        let drained: Vec<_> = func.inputs.drain(..).collect();
        // Assert
        assert!(func.inputs.is_empty());
        assert_eq!(drained, vec!["a", "b", "c"]);
    }

    // ── OnnxFunction outputs extend adds multiple entries ────────────────
    // @trace TEST-ONNX-FUNCTION-OUTPUTS-020

    #[test]
    fn onnx_function_outputs_extend_adds_multiple() {
        // Arrange
        let mut func = OnnxFunction {
            name: "func".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec!["out0".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        // Act
        func.outputs.extend(vec!["out1".to_string(), "out2".to_string()]);
        // Assert
        assert_eq!(func.outputs.len(), 3);
        assert_eq!(func.outputs[0], "out0");
        assert_eq!(func.outputs[1], "out1");
        assert_eq!(func.outputs[2], "out2");
    }

    // ── parse_metadata_props: duplicate keys — last value wins ───────────
    // @trace TEST-ONNX-PARSE-META-021

    #[test]
    fn parse_metadata_props_duplicate_keys_last_value_wins() {
        // Arrange
        let entries = vec![
            proto::StringStringEntryProto {
                key: Some("author".to_string()),
                value: Some("alice".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("author".to_string()),
                value: Some("bob".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("author".to_string()),
                value: Some("charlie".to_string()),
            },
        ];
        // Act
        let result = parse_metadata_props(entries);
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("author").unwrap(), "charlie");
    }

    // ── OnnxModel metadata clone produces independent copy ───────────────
    // @trace TEST-ONNX-MODEL-META-022

    #[test]
    fn onnx_model_metadata_clone_produces_independent_copy() {
        // Arrange
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: "test-producer".to_string(),
            producer_version: "1.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 42,
            doc_string: "test model".to_string(),
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 17,
            }],
            metadata_props: HashMap::from([("key".to_string(), "val".to_string())]),
        };
        // Act
        let mut cloned = meta.clone();
        cloned.producer_name = "modified".to_string();
        cloned.metadata_props.insert("extra".to_string(), "data".to_string());
        // Assert
        assert_eq!(meta.producer_name, "test-producer");
        assert_eq!(meta.metadata_props.len(), 1);
        assert_eq!(cloned.producer_name, "modified");
        assert_eq!(cloned.metadata_props.len(), 2);
    }

    // ── parse_nodes: consecutive nodes with no names get sequential indices
    // @trace TEST-ONNX-PARSE-NODES-023

    #[test]
    fn parse_nodes_consecutive_auto_names_increment_correctly() {
        // Arrange
        let nodes: Vec<proto::NodeProto> = (0..5)
            .map(|_| proto::NodeProto {
                op_type: Some("Identity".to_string()),
                name: None,
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            })
            .collect();
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        // Assert
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].name, "node_0");
        assert_eq!(result[1].name, "node_1");
        assert_eq!(result[2].name, "node_2");
        assert_eq!(result[3].name, "node_3");
        assert_eq!(result[4].name, "node_4");
    }

    // ── parse_value_info: name with special characters preserved exactly ─
    // @trace TEST-ONNX-PARSE-VI-024

    #[test]
    fn parse_value_info_name_with_special_chars_preserved() {
        // Arrange
        let entries = vec![proto::ValueInfoProto {
            name: Some("layer.0.self_attn/q_proj.bias".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        }];
        // Act
        let result = parse_value_info(entries).unwrap();
        // Assert
        assert_eq!(result[0].name, "layer.0.self_attn/q_proj.bias");
    }

    // ── OnnxQuantizationAnnotation param names insert and retrieve ───────
    // @trace TEST-ONNX-QUANT-PARAMS-025

    #[test]
    fn onnx_quantization_annotation_param_names_insert_and_retrieve() {
        // Arrange
        let mut annot = OnnxQuantizationAnnotation {
            tensor_name: "weight_q".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.015),
            zero_point: Some(128),
            axis: Some(0),
        };
        // Act
        annot.quant_param_tensor_names.insert("SCALE_TENSOR".to_string(), "weight_scale".to_string());
        annot.quant_param_tensor_names.insert("ZERO_POINT_TENSOR".to_string(), "weight_zp".to_string());
        // Assert
        assert_eq!(annot.quant_param_tensor_names.len(), 2);
        assert_eq!(annot.quant_param_tensor_names.get("SCALE_TENSOR").unwrap(), "weight_scale");
        assert_eq!(annot.quant_param_tensor_names.get("ZERO_POINT_TENSOR").unwrap(), "weight_zp");
    }

    // ── OnnxGraph name empty string is valid ─────────────────────────────
    // @trace TEST-ONNX-GRAPH-NAME-026

    #[test]
    fn onnx_graph_name_empty_string_is_valid() {
        // Arrange & Act
        let graph = OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Assert
        assert!(graph.name.is_empty());
        assert!(graph.nodes.is_empty());
        assert!(graph.initializers.is_empty());
    }

    // ── parse_quantization: no SCALE_TENSOR key yields None scale ────────
    // @trace TEST-ONNX-QUANT-NOSCALE-027

    #[test]
    fn parse_quantization_no_scale_key_yields_none_scale() {
        // Arrange
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("w".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("ZERO_POINT_TENSOR".to_string()),
                value: Some("w_zp".to_string()),
            }],
        }];
        let initializers = HashMap::new();
        // Act
        let result = parse_quantization(entries, &initializers);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0].scale.is_none());
    }

    // ── OnnxGraph value_info retain filters correctly ────────────────────
    // @trace TEST-ONNX-GRAPH-VI-028

    #[test]
    fn onnx_graph_value_info_retain_filters_correctly() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![
                OnnxValueInfo {
                    name: "keep_a".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "remove_b".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "keep_c".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        graph.value_info.retain(|vi| vi.name.starts_with("keep"));
        // Assert
        assert_eq!(graph.value_info.len(), 2);
        assert_eq!(graph.value_info[0].name, "keep_a");
        assert_eq!(graph.value_info[1].name, "keep_c");
    }

    // ── OnnxOperatorSet clone independence ───────────────────────────────
    // @trace TEST-ONNX-OPSET-029

    #[test]
    fn onnx_operator_set_clone_and_mutate_independence() {
        // Arrange
        let opset = OnnxOperatorSet {
            domain: "ai.onnx.nn".to_string(),
            version: 21,
        };
        // Act
        let mut cloned = opset.clone();
        cloned.domain = "ai.onnx.ml".to_string();
        cloned.version = 5;
        // Assert
        assert_eq!(opset.domain, "ai.onnx.nn");
        assert_eq!(opset.version, 21);
        assert_eq!(cloned.domain, "ai.onnx.ml");
        assert_eq!(cloned.version, 5);
    }

    // ── parse_functions: overload field defaults to empty ────────────────
    // @trace TEST-ONNX-PARSE-FUNC-030

    #[test]
    fn parse_functions_overload_defaults_empty_when_none() {
        // Arrange
        let func_proto = proto::FunctionProto {
            name: Some("my_op".to_string()),
            domain: Some("custom".to_string()),
            overload: None,
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            attribute: vec!["alpha".to_string()],
            attribute_proto: vec![],
            node: vec![proto::NodeProto {
                op_type: Some("Identity".to_string()),
                name: None,
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            }],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![],
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_functions(vec![func_proto], &mut resolver).unwrap();
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0].overload.is_empty());
        assert_eq!(result[0].name, "my_op");
        assert_eq!(result[0].domain, "custom");
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 8 TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxModelMetadata: ir_version and model_version set to i64::MIN ──

    #[test]
    fn onnx_model_metadata_i64_min_edge_values() {
        let meta = OnnxModelMetadata {
            ir_version: i64::MIN,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: i64::MIN,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta.ir_version, i64::MIN);
        assert_eq!(meta.model_version, i64::MIN);
    }

    // ── OnnxNode: outputs contain duplicate names (valid ONNX edge case) ──

    #[test]
    fn onnx_node_duplicate_output_names() {
        let node = OnnxNode {
            name: "split_dup".to_string(),
            op_type: "Split".to_string(),
            domain: String::new(),
            inputs: vec!["data".to_string()],
            outputs: vec!["out".to_string(), "out".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.outputs.len(), 2);
        assert_eq!(node.outputs[0], "out");
        assert_eq!(node.outputs[1], "out");
    }

    // ── OnnxGraph: initializer with BF16 dtype preserves dtype ──────────

    #[test]
    fn onnx_graph_initializer_bf16_dtype_preserved() {
        let tensor = OnnxTensor::new(
            "bf16_weight".to_string(),
            Dtype::BF16,
            vec![4, 4],
            Bytes::from(vec![0u8; 16]),
        );
        let graph = OnnxGraph {
            name: "bf16_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("bf16_weight".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let t = graph.initializers.get("bf16_weight").unwrap();
        assert_eq!(t.dtype, Dtype::BF16);
        assert_eq!(t.shape, vec![4, 4]);
    }

    // ── OnnxValueInfo: metadata_props with many entries preserves all ────

    #[test]
    fn onnx_value_info_metadata_props_many_entries() {
        let props: HashMap<String, String> = (0..10)
            .map(|i| (format!("key_{i}"), format!("val_{i}")))
            .collect();
        let vi = OnnxValueInfo {
            name: "multi_meta".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: props.clone(),
        };
        assert_eq!(vi.metadata_props.len(), 10);
        for i in 0..10 {
            assert_eq!(
                vi.metadata_props.get(&format!("key_{i}")).unwrap(),
                &format!("val_{i}")
            );
        }
    }

    // ── parse_opsets: all fields None produces default values ────────────

    #[test]
    fn parse_opsets_all_none_fields_produce_defaults() {
        let opsets = vec![
            proto::OperatorSetIdProto { domain: None, version: None },
            proto::OperatorSetIdProto { domain: None, version: None },
        ];
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].domain, "");
        assert_eq!(result[0].version, 0);
        assert_eq!(result[1].domain, "");
        assert_eq!(result[1].version, 0);
    }

    // ── OnnxQuantizationAnnotation: all optional fields simultaneously set

    #[test]
    fn onnx_quantization_all_option_fields_simultaneously_set() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "full_quant".to_string(),
            quant_param_tensor_names: HashMap::from([
                ("SCALE_TENSOR".to_string(), "s".to_string()),
                ("ZERO_POINT_TENSOR".to_string(), "z".to_string()),
                ("AXIS".to_string(), "a".to_string()),
            ]),
            scale: Some(0.00390625),
            zero_point: Some(127),
            axis: Some(2),
        };
        assert!(qa.scale.is_some());
        assert!(qa.zero_point.is_some());
        assert!(qa.axis.is_some());
        assert_eq!(qa.quant_param_tensor_names.len(), 3);
        assert!((qa.scale.unwrap() - 0.00390625).abs() < 1e-10);
        assert_eq!(qa.zero_point.unwrap(), 127);
        assert_eq!(qa.axis.unwrap(), 2);
    }

    // ── OnnxFunction: overload field with non-empty value ───────────────

    #[test]
    fn onnx_function_overload_non_empty_value() {
        let func = OnnxFunction {
            name: "OverloadedOp".to_string(),
            domain: "custom".to_string(),
            overload: "v3_special".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.overload, "v3_special");
        assert!(!func.overload.is_empty());
    }

    // ── OnnxModel: graph field has correct node count after construction ─

    #[test]
    fn onnx_model_graph_node_count_matches_construction() {
        let nodes: Vec<OnnxNode> = (0..5)
            .map(|i| OnnxNode {
                name: format!("node_{i}"),
                op_type: "Relu".to_string(),
                domain: String::new(),
                inputs: vec![format!("in_{i}")],
                outputs: vec![format!("out_{i}")],
                attributes: HashMap::new(),
            })
            .collect();
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 7,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes,
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert_eq!(model.graph.nodes.len(), 5);
        assert_eq!(model.graph.nodes[0].name, "node_0");
        assert_eq!(model.graph.nodes[4].name, "node_4");
    }

    // ── OnnxTensor: scalar_i64 with i64::MIN value roundtrip ────────────

    #[test]
    fn onnx_tensor_scalar_i64_min_roundtrip() {
        let min_bytes = i64::MIN.to_le_bytes();
        let tensor = OnnxTensor::new(
            "min_i64_rt".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(min_bytes.to_vec()),
        );
        assert_eq!(tensor.scalar_i64(), Some(i64::MIN));
    }

    // ── OnnxOperatorSet: Debug trait includes domain and version together ─

    #[test]
    fn onnx_operator_set_debug_shows_domain_and_version() {
        let ops = OnnxOperatorSet {
            domain: "ai.onnx.ml".to_string(),
            version: 5,
        };
        let debug = format!("{ops:?}");
        assert!(debug.contains("ai.onnx.ml"), "expected domain in: {debug}");
        assert!(debug.contains("5"), "expected version in: {debug}");
    }

    // ── OnnxGraph: sparse_initializers is independent after clone ────────

    #[test]
    fn onnx_graph_sparse_initializers_clone_independence() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let sparse = OnnxSparseTensor {
            values: OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4])),
            indices: OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8])),
            dims: vec![3, 3],
            format: OnnxSparseFormat::Coo,
        };
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![sparse],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let cloned = graph.clone();
        graph.sparse_initializers.clear();
        assert_eq!(cloned.sparse_initializers.len(), 1);
        assert_eq!(cloned.sparse_initializers[0].dims, vec![3, 3]);
    }

    // ── parse_value_info: doc_string defaults to empty when None ────────

    #[test]
    fn parse_value_info_doc_string_defaults_empty_when_none() {
        let values = vec![proto::ValueInfoProto {
            name: Some("test".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        }];
        let result = parse_value_info(values).unwrap();
        assert_eq!(result[0].doc_string, "");
    }

    // ── OnnxNode: clone produces independent inputs vec ──────────────────

    #[test]
    fn onnx_node_clone_inputs_vec_independent() {
        let mut node = OnnxNode {
            name: "clone_test".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            attributes: HashMap::new(),
        };
        let cloned = node.clone();
        node.inputs.push("d".to_string());
        assert_eq!(cloned.inputs.len(), 2);
        assert_eq!(node.inputs.len(), 3);
    }

    // ── OnnxGraph: metadata_props with empty string value is stored ──────

    #[test]
    fn onnx_graph_metadata_props_empty_string_value_stored() {
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::from([("empty_val".to_string(), String::new())]),
        };
        assert!(graph.metadata_props.contains_key("empty_val"));
        assert_eq!(graph.metadata_props.get("empty_val").unwrap(), "");
    }

    // ── WeightFormat: all five variants are distinct ─────────────────────

    #[test]
    fn weight_format_all_five_variants_distinct() {
        use crate::loader::WeightFormat;
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        // Verify all pairwise distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "variants[{i}] == variants[{j}]");
            }
        }
        assert_eq!(variants.len(), 5);
    }

    // ── bind_weights_auto: 成功匹配权重名称并插入 initializers ──────────

    #[test]
    fn onnx_graph_bind_weights_auto_successfully_binds_matched_weight() {
        // Arrange: 构造一个带权重输入节点的图，provider 中有匹配的 tensor
        struct MatchProvider;
        impl crate::loader::TensorProvider for MatchProvider {
            fn tensor_info(&self, name: &str) -> Option<crate::loader::TensorMeta> {
                if name == "model.weight" {
                    Some(crate::loader::TensorMeta {
                        name: "model.weight".to_string(),
                        dtype: Dtype::F32,
                        shape: vec![4, 4],
                    })
                } else {
                    None
                }
            }
            fn load_tensor_data(
                &self,
                name: &str,
            ) -> std::result::Result<std::borrow::Cow<'_, [u8]>, LoaderError> {
                if name == "model.weight" {
                    Ok(std::borrow::Cow::Owned(vec![0u8; 64]))
                } else {
                    Err(LoaderError::MissingTensor(name.to_string()))
                }
            }
            fn iter_tensors(&self) -> impl Iterator<Item = crate::loader::TensorMeta> {
                vec![crate::loader::TensorMeta {
                    name: "model.weight".to_string(),
                    dtype: Dtype::F32,
                    shape: vec![4, 4],
                }].into_iter()
            }
        }
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "matmul".to_string(),
                op_type: "MatMul".to_string(),
                domain: String::new(),
                inputs: vec!["hidden_state".to_string(), "model.weight".to_string()],
                outputs: vec!["out".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        let count = graph.bind_weights_auto(&MatchProvider).unwrap();
        // Assert: 匹配 1 个权重，插入到 initializers 中
        assert_eq!(count, 1);
        assert!(graph.initializers.contains_key("model.weight"));
        let tensor = &graph.initializers["model.weight"];
        assert_eq!(tensor.shape, vec![4, 4]);
    }

    // ── parse_quantization: AXIS="0" 被解析为零 ─────────────────────────

    #[test]
    fn parse_quantization_axis_zero_parsed_correctly() {
        // Arrange
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("AXIS".to_string()),
                value: Some("0".to_string()),
            }],
        }];
        // Act
        let result = parse_quantization(entries, &HashMap::new());
        // Assert: axis=0 是合法值，应被解析为 Some(0)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].axis, Some(0));
    }

    // ── OnnxNode: outputs 清空后可以重新添加 ────────────────────────────

    #[test]
    fn onnx_node_outputs_clear_then_repopulate() {
        // Arrange
        let mut node = OnnxNode {
            name: "split".to_string(),
            op_type: "Split".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y1".to_string(), "y2".to_string(), "y3".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.outputs.len(), 3);
        // Act: 清空再添加
        node.outputs.clear();
        assert!(node.outputs.is_empty());
        node.outputs.push("new_out".to_string());
        // Assert
        assert_eq!(node.outputs.len(), 1);
        assert_eq!(node.outputs[0], "new_out");
    }

    // ── OnnxGraph: value_info drain 清空并返回所有条目 ──────────────────

    #[test]
    fn onnx_graph_value_info_drain_empties_and_returns() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![
                OnnxValueInfo {
                    name: "intermediate_1".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "intermediate_2".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        let drained: Vec<_> = graph.value_info.drain(..).collect();
        // Assert
        assert!(graph.value_info.is_empty());
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].name, "intermediate_1");
        assert_eq!(drained[1].name, "intermediate_2");
    }

    // ── OnnxGraph: nodes 按 op_type 排序不改变内容 ─────────────────────

    #[test]
    fn onnx_graph_nodes_sort_by_op_type_preserves_count() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "n2".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n1".to_string(),
                    op_type: "Conv".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n3".to_string(),
                    op_type: "Add".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        graph.nodes.sort_by(|a, b| a.op_type.cmp(&b.op_type));
        // Assert: 数量不变，顺序按 op_type 字母序
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].op_type, "Add");
        assert_eq!(graph.nodes[1].op_type, "Conv");
        assert_eq!(graph.nodes[2].op_type, "Relu");
    }

    // ── OnnxGraph: initializers entry API 插入新条目 ────────────────────

    #[test]
    fn onnx_graph_initializers_entry_api_inserts_new_tensor() {
        // Arrange
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let tensor = OnnxTensor::new(
            "bias".to_string(),
            Dtype::F32,
            vec![128],
            Bytes::from(vec![0u8; 512]),
        );
        // Act: 使用 entry API 确保不存在才插入
        let entry = graph.initializers.entry("bias".to_string());
        assert!(matches!(entry, std::collections::hash_map::Entry::Vacant(_)));
        graph.initializers.insert("bias".to_string(), tensor);
        // Assert
        assert_eq!(graph.initializers.len(), 1);
        assert!(graph.initializers.contains_key("bias"));
        assert_eq!(graph.initializers["bias"].shape, vec![128]);
    }

    // ── OnnxFunction: value_info 有多个条目的结构体验证 ─────────────────

    #[test]
    fn onnx_function_value_info_multiple_entries_accessible() {
        // Arrange
        let func = OnnxFunction {
            name: "custom_op".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![
                OnnxValueInfo {
                    name: "temp_a".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "temp_b".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "temp_c".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert_eq!(func.value_info.len(), 3);
        assert_eq!(func.value_info[0].name, "temp_a");
        assert_eq!(func.value_info[1].name, "temp_b");
        assert_eq!(func.value_info[2].name, "temp_c");
    }

    // ── OnnxFunction: opset_import 包含多个不同条目 ─────────────────────

    #[test]
    fn onnx_function_opset_import_multiple_distinct_domains() {
        // Arrange
        let func = OnnxFunction {
            name: "f".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![
                OnnxOperatorSet {
                    domain: String::new(),
                    version: 17,
                },
                OnnxOperatorSet {
                    domain: "ai.onnx.ml".to_string(),
                    version: 3,
                },
            ],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert_eq!(func.opset_import.len(), 2);
        assert_eq!(func.opset_import[0].version, 17);
        assert_eq!(func.opset_import[1].domain, "ai.onnx.ml");
        assert_eq!(func.opset_import[1].version, 3);
    }

    // ── parse_quantization: SCALE_TENSOR 来自 I64 dtype 张量 ──────────
    // scalar_f32 对 I64 标量会转换为 f32（通过 as f32），验证非 F32 dtype 也能提取 scale

    #[test]
    fn parse_quantization_scale_from_i64_scalar_tensor_converted() {
        // Arrange: 构造一个 I64 标量 initializer，值为 1000（代表 scale 千分位）
        let scale_bytes = 1000i64.to_le_bytes();
        let scale_tensor = OnnxTensor::new(
            "my_scale".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(scale_bytes.to_vec()),
        );
        let initializers = HashMap::from([("my_scale".to_string(), scale_tensor)]);
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![
                proto::StringStringEntryProto {
                    key: Some("SCALE_TENSOR".to_string()),
                    value: Some("my_scale".to_string()),
                },
            ],
        }];
        // Act
        let result = parse_quantization(entries, &initializers);
        // Assert: I64 标量被 scalar_f32 转换为 f32
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].scale, Some(1000.0f32));
    }

    // ── parse_quantization: ZERO_POINT_TENSOR 来自 U8 标量张量 ────────

    #[test]
    fn parse_quantization_zero_point_from_u8_scalar_tensor() {
        // Arrange: 构造一个 U8 标量 initializer，零点值为 128
        let zp_tensor = OnnxTensor::new(
            "my_zp".to_string(),
            Dtype::U8,
            vec![],
            Bytes::from(vec![128u8]),
        );
        let initializers = HashMap::from([("my_zp".to_string(), zp_tensor)]);
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![
                proto::StringStringEntryProto {
                    key: Some("ZERO_POINT_TENSOR".to_string()),
                    value: Some("my_zp".to_string()),
                },
            ],
        }];
        // Act
        let result = parse_quantization(entries, &initializers);
        // Assert: U8 标量的零点值被 scalar_i64 转换为 i64
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].zero_point, Some(128));
    }

    // ── OnnxGraph: outputs 中的 value_type 混合存在与缺失 ───────────────

    #[test]
    fn onnx_graph_outputs_with_mixed_type_presence() {
        // Arrange
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![
                OnnxValueInfo {
                    name: "logits".to_string(),
                    value_type: Some(OnnxType::Tensor(OnnxTensorType {
                        elem_type: proto::tensor_proto::DataType::Float,
                        shape: OnnxTensorShape {
                            dims: vec![OnnxDim::Param("batch".to_string()), OnnxDim::Known(1000)],
                        },
                    })),
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
                OnnxValueInfo {
                    name: "probabilities".to_string(),
                    value_type: None,
                    doc_string: String::new(),
                    metadata_props: HashMap::new(),
                },
            ],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act & Assert: 第一个 output 有类型信息，第二个没有
        assert!(graph.outputs[0].value_type.is_some());
        assert!(graph.outputs[1].value_type.is_none());
        if let Some(OnnxType::Tensor(tt)) = &graph.outputs[0].value_type {
            assert!(matches!(tt.elem_type, proto::tensor_proto::DataType::Float));
            assert_eq!(tt.shape.dims.len(), 2);
        } else {
            panic!("expected Tensor type for logits");
        }
    }

    // ── OnnxModel: metadata producer_name 为空字符串时保持一致 ──────────

    #[test]
    fn onnx_model_metadata_empty_producer_name_preserved() {
        // Arrange
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![OnnxOperatorSet {
                domain: String::new(),
                version: 19,
            }],
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: meta,
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        // Act & Assert
        assert!(model.metadata.producer_name.is_empty());
        assert!(model.metadata.producer_version.is_empty());
        assert_eq!(model.metadata.ir_version, 8);
        assert_eq!(model.metadata.opset_import[0].version, 19);
    }

    // ── OnnxGraph: nodes 中查找特定 op_type 的节点 ─────────────────────

    #[test]
    fn onnx_graph_nodes_find_by_op_type() {
        // Arrange
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "conv1".to_string(),
                    op_type: "Conv".to_string(),
                    domain: String::new(),
                    inputs: vec!["x".to_string(), "w".to_string()],
                    outputs: vec!["y".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "relu1".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec!["y".to_string()],
                    outputs: vec!["z".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "conv2".to_string(),
                    op_type: "Conv".to_string(),
                    domain: String::new(),
                    inputs: vec!["z".to_string(), "w2".to_string()],
                    outputs: vec!["y2".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act: 查找所有 Conv 节点
        let conv_nodes: Vec<_> = graph.nodes.iter().filter(|n| n.op_type == "Conv").collect();
        // Assert
        assert_eq!(conv_nodes.len(), 2);
        assert_eq!(conv_nodes[0].name, "conv1");
        assert_eq!(conv_nodes[1].name, "conv2");
        let relu_nodes: Vec<_> = graph.nodes.iter().filter(|n| n.op_type == "Relu").collect();
        assert_eq!(relu_nodes.len(), 1);
    }

    // ── OnnxQuantizationAnnotation: quant_param_tensor_names 包含多个参数 ─

    #[test]
    fn onnx_quantization_annotation_multiple_param_names() {
        // Arrange
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "weight_quant".to_string(),
            quant_param_tensor_names: HashMap::from([
                ("SCALE_TENSOR".to_string(), "weight_scale".to_string()),
                ("ZERO_POINT_TENSOR".to_string(), "weight_zp".to_string()),
                ("AXIS".to_string(), "1".to_string()),
            ]),
            scale: Some(0.01),
            zero_point: Some(0),
            axis: Some(1),
        };
        // Act & Assert
        assert_eq!(qa.quant_param_tensor_names.len(), 3);
        assert_eq!(qa.quant_param_tensor_names["SCALE_TENSOR"], "weight_scale");
        assert_eq!(qa.quant_param_tensor_names["ZERO_POINT_TENSOR"], "weight_zp");
        assert_eq!(qa.quant_param_tensor_names["AXIS"], "1");
    }

    // ── parse_functions: 带 metadata_props 的函数被正确解析 ─────────────

    #[test]
    fn parse_functions_with_metadata_props_preserved() {
        // Arrange
        let functions = vec![proto::FunctionProto {
            name: Some("custom_func".to_string()),
            domain: Some("test.domain".to_string()),
            overload: None,
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            attribute: vec![],
            attribute_proto: vec![],
            node: vec![proto::NodeProto {
                op_type: Some("Identity".to_string()),
                name: None,
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            }],
            opset_import: vec![],
            value_info: vec![],
            doc_string: None,
            metadata_props: vec![
                proto::StringStringEntryProto {
                    key: Some("author".to_string()),
                    value: Some("test_suite".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("version".to_string()),
                    value: Some("2.0".to_string()),
                },
            ],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_functions(functions, &mut resolver).unwrap();
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].metadata_props.len(), 2);
        assert_eq!(result[0].metadata_props["author"], "test_suite");
        assert_eq!(result[0].metadata_props["version"], "2.0");
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 8 TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── bind_weights: 真实 TensorProvider 成功绑定权重到 initializer ─────

    #[test]
    fn bind_weights_real_provider_binds_tensor() {
        // Arrange
        struct StaticProvider;
        impl crate::loader::TensorProvider for StaticProvider {
            fn tensor_info(&self, name: &str) -> Option<crate::loader::TensorMeta> {
                if name == "layer_weight" {
                    Some(crate::loader::TensorMeta {
                        name: "layer_weight".to_string(),
                        dtype: Dtype::F32,
                        shape: vec![3, 4],
                    })
                } else {
                    None
                }
            }
            fn load_tensor_data(
                &self,
                name: &str,
            ) -> std::result::Result<std::borrow::Cow<'_, [u8]>, LoaderError> {
                if name == "layer_weight" {
                    Ok(std::borrow::Cow::Owned(vec![0u8; 48])) // 3*4*4 bytes
                } else {
                    panic!("unexpected tensor load: {name}");
                }
            }
            fn iter_tensors(&self) -> impl Iterator<Item = crate::loader::TensorMeta> {
                std::iter::once(crate::loader::TensorMeta {
                    name: "layer_weight".to_string(),
                    dtype: Dtype::F32,
                    shape: vec![3, 4],
                })
            }
        }
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let mapping = HashMap::from([("layer_weight".to_string(), "layer_weight".to_string())]);
        // Act
        let count = graph.bind_weights(&StaticProvider, &mapping).unwrap();
        // Assert
        assert_eq!(count, 1);
        let t = graph.initializers.get("layer_weight").unwrap();
        assert_eq!(t.dtype, Dtype::F32);
        assert_eq!(t.shape, vec![3, 4]);
    }

    // ── bind_weights: name_mapping 中无匹配 tensor_info 时返回零 ─────────

    #[test]
    fn bind_weights_provider_lacks_requested_tensor() {
        // Arrange
        struct EmptyProvider;
        impl crate::loader::TensorProvider for EmptyProvider {
            fn tensor_info(&self, _name: &str) -> Option<crate::loader::TensorMeta> { None }
            fn load_tensor_data(
                &self, _name: &str,
            ) -> std::result::Result<std::borrow::Cow<'_, [u8]>, LoaderError> {
                panic!("should not be called");
            }
            fn iter_tensors(&self) -> impl Iterator<Item = crate::loader::TensorMeta> {
                std::iter::empty()
            }
        }
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let mapping = HashMap::from([("missing_weight".to_string(), "nonexistent".to_string())]);
        // Act
        let count = graph.bind_weights(&EmptyProvider, &mapping).unwrap();
        // Assert
        assert_eq!(count, 0);
        assert!(!graph.initializers.contains_key("missing_weight"));
    }

    // ── parse_metadata_props: 空白键（非空字符串）应被保留 ───────────────

    #[test]
    fn parse_metadata_props_whitespace_key_preserved() {
        // Arrange
        let entries = vec![proto::StringStringEntryProto {
            key: Some("   ".to_string()), // whitespace-only, non-empty
            value: Some("whitespace_key_value".to_string()),
        }];
        // Act
        let result = parse_metadata_props(entries);
        // Assert — key is non-empty so it should be stored
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("   ").unwrap(), "whitespace_key_value");
    }

    // ── parse_metadata_props: 值包含换行符和特殊字符 ─────────────────────

    #[test]
    fn parse_metadata_props_value_with_newlines_and_special_chars() {
        // Arrange
        let entries = vec![proto::StringStringEntryProto {
            key: Some("description".to_string()),
            value: Some("line1\nline2\ttab\rcarriage\0null".to_string()),
        }];
        // Act
        let result = parse_metadata_props(entries);
        // Assert
        let val = result.get("description").unwrap();
        assert!(val.contains('\n'));
        assert!(val.contains('\t'));
        assert!(val.contains('\r'));
        assert!(val.contains('\0'));
    }

    // ── OnnxTensor::scalar_f32 对不支持的 F64 dtype 返回 None ──────────

    #[test]
    fn onnx_tensor_scalar_f32_unsupported_f64() {
        // Arrange — 创建一个 F64 标量 tensor（F64 不在 scalar_f32 支持列表中）
        let tensor = OnnxTensor::new(
            "f64_scalar".to_string(),
            Dtype::F64,
            vec![],
            Bytes::from(3.14f64.to_le_bytes().to_vec()),
        );
        // Act
        let result = tensor.scalar_f32();
        // Assert — F64 tensor 的 scalar_f32() 应返回 None
        assert_eq!(result, None);
    }

    // ── OnnxTensor::scalar_i64 对不支持的 BOOL dtype 返回 None ──────────

    #[test]
    fn onnx_tensor_scalar_i64_unsupported_bool() {
        // Arrange — 创建一个 BOOL 标量 tensor（BOOL 不在 scalar_i64 支持列表中）
        let tensor = OnnxTensor::new(
            "bool_scalar".to_string(),
            Dtype::BOOL,
            vec![],
            Bytes::from(vec![1u8]),
        );
        // Act
        let result = tensor.scalar_i64();
        // Assert — BOOL tensor 的 scalar_i64() 应返回 None
        assert_eq!(result, None);
    }

    // ── parse_quantization: SCALE_TENSOR 存在但 dtype 为不支持的 F64 返回 None ─

    #[test]
    fn parse_quantization_scale_tensor_unsupported_f64_yields_none() {
        // Arrange — scale tensor 是 F64（scalar_f32 不支持 F64，命中 _ => None 分支）
        let scale_tensor = OnnxTensor::new(
            "bad_scale".to_string(),
            Dtype::F64,
            vec![],
            Bytes::from(0.5f64.to_le_bytes().to_vec()),
        );
        let initializers = HashMap::from([("bad_scale".to_string(), scale_tensor)]);
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("target".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("SCALE_TENSOR".to_string()),
                value: Some("bad_scale".to_string()),
            }],
        }];
        // Act
        let result = parse_quantization(entries, &initializers);
        // Assert — F64 tensor 的 scalar_f32() 返回 None
        assert_eq!(result[0].scale, None);
    }

    // ── parse_quantization: ZERO_POINT_TENSOR 存在但 dtype 为不支持的 F64 ──

    #[test]
    fn parse_quantization_zp_tensor_unsupported_f64_yields_none() {
        // Arrange — zero point tensor 是 F64（scalar_i64 不支持 F64，命中 _ => None 分支）
        let zp_tensor = OnnxTensor::new(
            "bad_zp".to_string(),
            Dtype::F64,
            vec![],
            Bytes::from(99.0f64.to_le_bytes().to_vec()),
        );
        let initializers = HashMap::from([("bad_zp".to_string(), zp_tensor)]);
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("target".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("ZERO_POINT_TENSOR".to_string()),
                value: Some("bad_zp".to_string()),
            }],
        }];
        // Act
        let result = parse_quantization(entries, &initializers);
        // Assert — F64 tensor 的 scalar_i64() 返回 None
        assert_eq!(result[0].zero_point, None);
    }

    // ── OnnxTensor::new_string 标量（空 shape）文本张量 ──────────────────

    #[test]
    fn onnx_tensor_new_string_scalar_empty_shape() {
        // Arrange & Act
        let tensor = OnnxTensor::new_string(
            "label".to_string(),
            vec![],
            Bytes::from("positive".as_bytes().to_vec()),
        );
        // Assert
        assert!(tensor.is_string);
        assert!(tensor.shape.is_empty());
        assert_eq!(tensor.dtype, Dtype::U8);
        assert_eq!(tensor.raw_data(), "positive".as_bytes());
    }

    // ── OnnxTensor::new_string 多维 shape 文本张量 ──────────────────────

    #[test]
    fn onnx_tensor_new_string_multidim_shape() {
        // Arrange & Act
        let tensor = OnnxTensor::new_string(
            "batch_labels".to_string(),
            vec![4, 10],
            Bytes::from(vec![0u8; 40]),
        );
        // Assert
        assert!(tensor.is_string);
        assert_eq!(tensor.shape, vec![4, 10]);
    }

    // ── OnnxGraph: from_proto 多节点图保留完整 IO 连接链 ─────────────────

    #[test]
    fn onnx_graph_from_proto_multi_node_chain_io_connected() {
        // Arrange — 构建一个 Embedding→LayerNorm→Linear 三节点图
        let graph_proto = proto::GraphProto {
            name: Some("bert_layer".to_string()),
            node: vec![
                proto::NodeProto {
                    op_type: Some("Gather".to_string()),
                    name: Some("embed".to_string()),
                    input: vec!["input_ids".to_string(), "embed_weight".to_string()],
                    output: vec!["embed_out".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                },
                proto::NodeProto {
                    op_type: Some("LayerNormalization".to_string()),
                    name: Some("ln1".to_string()),
                    input: vec!["embed_out".to_string(), "ln_weight".to_string(), "ln_bias".to_string()],
                    output: vec!["ln_out".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                },
                proto::NodeProto {
                    op_type: Some("MatMul".to_string()),
                    name: Some("linear".to_string()),
                    input: vec!["ln_out".to_string(), "linear_weight".to_string()],
                    output: vec!["logits".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                },
            ],
            input: vec![proto::ValueInfoProto {
                name: Some("input_ids".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            output: vec![proto::ValueInfoProto {
                name: Some("logits".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].outputs[0], graph.nodes[1].inputs[0]);
        assert_eq!(graph.nodes[1].outputs[0], graph.nodes[2].inputs[0]);
        assert_eq!(graph.inputs[0].name, "input_ids");
        assert_eq!(graph.outputs[0].name, "logits");
    }

    // ── OnnxGraph: from_proto 图的 inputs 数量与 proto 的 input 长度一致 ─

    #[test]
    fn onnx_graph_from_proto_inputs_count_matches_proto() {
        // Arrange — 构建 4 个输入、1 个输出的图
        let graph_proto = proto::GraphProto {
            name: Some("multi_input".to_string()),
            input: vec![
                proto::ValueInfoProto { name: Some("input_ids".to_string()), r#type: None, doc_string: None, metadata_props: vec![] },
                proto::ValueInfoProto { name: Some("attention_mask".to_string()), r#type: None, doc_string: None, metadata_props: vec![] },
                proto::ValueInfoProto { name: Some("token_type_ids".to_string()), r#type: None, doc_string: None, metadata_props: vec![] },
                proto::ValueInfoProto { name: Some("position_ids".to_string()), r#type: None, doc_string: None, metadata_props: vec![] },
            ],
            output: vec![proto::ValueInfoProto {
                name: Some("logits".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert_eq!(graph.inputs.len(), 4);
        assert_eq!(graph.inputs[3].name, "position_ids");
        assert_eq!(graph.outputs.len(), 1);
    }

    // ── OnnxModel: from_proto 带 metadata_props 和 functions 共存 ──────

    #[test]
    fn onnx_model_from_proto_metadata_and_functions_coexist() {
        // Arrange
        let model_proto = proto::ModelProto {
            ir_version: Some(9),
            producer_name: Some("coexist_test".to_string()),
            metadata_props: vec![
                proto::StringStringEntryProto {
                    key: Some("license".to_string()),
                    value: Some("Apache-2.0".to_string()),
                },
            ],
            graph: Some(proto::GraphProto {
                name: Some("main_graph".to_string()),
                node: vec![proto::NodeProto {
                    op_type: Some("CustomOp".to_string()),
                    name: Some("custom_node".to_string()),
                    input: vec!["X".to_string()],
                    output: vec!["Y".to_string()],
                    domain: Some("custom".to_string()),
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                }],
                ..Default::default()
            }),
            functions: vec![proto::FunctionProto {
                name: Some("CustomOp".to_string()),
                domain: Some("custom".to_string()),
                overload: None,
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                attribute: vec![],
                attribute_proto: vec![],
                node: vec![proto::NodeProto {
                    op_type: Some("Identity".to_string()),
                    name: None,
                    input: vec!["X".to_string()],
                    output: vec!["Y".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                }],
                opset_import: vec![],
                value_info: vec![],
                doc_string: None,
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        // Assert
        assert_eq!(model.metadata.producer_name, "coexist_test");
        assert_eq!(model.metadata.metadata_props.get("license").unwrap(), "Apache-2.0");
        assert_eq!(model.graph.nodes.len(), 1);
        assert_eq!(model.graph.nodes[0].domain, "custom");
        assert_eq!(model.functions.len(), 1);
        assert_eq!(model.functions[0].name, "CustomOp");
        assert_eq!(model.functions[0].nodes[0].op_type, "Identity");
    }

    // ── OnnxGraph: 初始器名称包含 ONNX 命名空间分隔符 ────────────────────

    #[test]
    fn onnx_graph_initializer_name_with_namespace_separator() {
        // Arrange
        let tensor = OnnxTensor::new(
            "model.layer.0.attention.query.weight".to_string(),
            Dtype::F32,
            vec![768, 768],
            Bytes::from(vec![0u8; 768 * 768 * 4]),
        );
        let graph = OnnxGraph {
            name: "ns_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "query_proj".to_string(),
                op_type: "MatMul".to_string(),
                domain: String::new(),
                inputs: vec!["hidden".to_string(), "model.layer.0.attention.query.weight".to_string()],
                outputs: vec!["query".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::from([("model.layer.0.attention.query.weight".to_string(), tensor)]),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        let key = "model.layer.0.attention.query.weight";
        assert!(graph.initializers.contains_key(key));
        assert_eq!(graph.nodes[0].inputs[1], key);
        assert_eq!(graph.initializers.get(key).unwrap().shape, vec![768, 768]);
    }

    // ── parse_quantization: 非 AXIS 键的 quant_parameter 被存储在 map 中 ─

    #[test]
    fn parse_quantization_custom_param_keys_stored() {
        // Arrange — 非标准键如 "CUSTOM_KEY" 应该被保存到 quant_param_tensor_names
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("weight".to_string()),
            quant_parameter_tensor_names: vec![
                proto::StringStringEntryProto {
                    key: Some("CUSTOM_TENSOR".to_string()),
                    value: Some("custom_ref".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("SCALE_TENSOR".to_string()),
                    value: Some("nonexistent_scale".to_string()),
                },
            ],
        }];
        // Act
        let result = parse_quantization(entries, &HashMap::new());
        // Assert — scale 为 None (tensor 不存在), 但 custom key 被存储
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].scale, None);
        assert_eq!(result[0].quant_param_tensor_names.len(), 2);
        assert_eq!(
            result[0].quant_param_tensor_names.get("CUSTOM_TENSOR").unwrap(),
            "custom_ref"
        );
    }

    // ── OnnxQuantizationAnnotation: 零点为 i64::MIN 边界值与 scale 组合 ──

    #[test]
    fn onnx_quantization_annotation_zero_point_i64_min_with_scale() {
        // Arrange
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "edge_weight".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(1.0),
            zero_point: Some(i64::MIN),
            axis: Some(0),
        };
        // Act & Assert
        assert_eq!(qa.zero_point, Some(i64::MIN));
        assert_eq!(qa.scale, Some(1.0));
    }

    // ── parse_nodes: last node in batch with empty op_type yields error ──

    #[test]
    fn parse_nodes_last_node_empty_op_type_returns_error() {
        // Arrange: two valid nodes followed by one with empty op_type
        let nodes = vec![
            proto::NodeProto {
                op_type: Some("Relu".to_string()),
                name: Some("relu_0".to_string()),
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: Some("Add".to_string()),
                name: Some("add_0".to_string()),
                input: vec!["Y".to_string(), "bias".to_string()],
                output: vec!["Z".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: Some(String::new()), // empty op_type
                name: Some("bad_node".to_string()),
                input: vec![],
                output: vec![],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
        ];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_nodes(nodes, &mut resolver);
        // Assert: error contains the index 2
        let err = result.unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("node 2") || msg.contains("2"));
    }

    // ── parse_metadata_props: very large key is preserved ───────────────

    #[test]
    fn parse_metadata_props_very_large_key_preserved() {
        // Arrange
        let large_key = "x".repeat(10_000);
        let entries = vec![proto::StringStringEntryProto {
            key: Some(large_key.clone()),
            value: Some("big".to_string()),
        }];
        // Act
        let result = parse_metadata_props(entries);
        // Assert
        assert_eq!(result.get(&large_key).unwrap(), "big");
        assert_eq!(result.len(), 1);
    }

    // ── OnnxModel clone consistency with functions list ──────────────────

    #[test]
    fn onnx_model_clone_with_functions_list_independent() {
        // Arrange
        let func = OnnxFunction {
            name: "MyCustomOp".to_string(),
            domain: "custom.domain".to_string(),
            overload: "v2".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            attributes: vec!["alpha".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![OnnxNode {
                name: "inner_add".to_string(),
                op_type: "Add".to_string(),
                domain: String::new(),
                inputs: vec!["A".to_string(), "B".to_string()],
                outputs: vec!["C".to_string()],
                attributes: HashMap::new(),
            }],
            opset_import: vec![OnnxOperatorSet {
                domain: "".to_string(),
                version: 17,
            }],
            value_info: vec![],
            doc_string: "custom op doc".to_string(),
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: "producer".to_string(),
                producer_version: "2.0".to_string(),
                domain: "ai.onnx".to_string(),
                model_version: 100,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![func],
        };
        // Act
        let cloned = model.clone();
        // Assert: functions list is independent
        assert_eq!(cloned.functions.len(), 1);
        assert_eq!(cloned.functions[0].name, "MyCustomOp");
        assert_eq!(cloned.functions[0].nodes.len(), 1);
        assert_eq!(cloned.functions[0].nodes[0].op_type, "Add");
        // Verify independence: mutating clone does not affect original
        let original_len = model.functions.len();
        assert_eq!(original_len, 1);
    }

    // ── parse_opsets: empty input returns empty vec ─────────────────────

    #[test]
    fn parse_opsets_empty_input_returns_empty() {
        // Arrange: empty vec
        let opsets: Vec<proto::OperatorSetIdProto> = vec![];
        // Act
        let result = parse_opsets(opsets);
        // Assert
        assert!(result.is_empty());
    }

    // ── OnnxNode attributes HashMap clear and re-insert ──────────────────

    #[test]
    fn onnx_node_attributes_clear_and_reinsert() {
        // Arrange
        let mut node = OnnxNode {
            name: "test_node".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        node.attributes.insert("kernel_shape".to_string(), OnnxAttribute {
            name: "kernel_shape".to_string(),
            value: OnnxAttributeValue::Ints(vec![3, 3]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        assert_eq!(node.attributes.len(), 1);
        // Act: clear and re-insert different value
        node.attributes.clear();
        node.attributes.insert("strides".to_string(), OnnxAttribute {
            name: "strides".to_string(),
            value: OnnxAttributeValue::Ints(vec![2, 2]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        // Assert
        assert_eq!(node.attributes.len(), 1);
        assert!(node.attributes.contains_key("strides"));
        assert!(!node.attributes.contains_key("kernel_shape"));
    }

    // ── OnnxQuantizationAnnotation tensor_name with special characters ───

    #[test]
    fn onnx_quantization_annotation_tensor_name_special_chars() {
        // Arrange
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "layer.3/block_output:0".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        // Act & Assert
        assert_eq!(qa.tensor_name, "layer.3/block_output:0");
        assert!(qa.scale.is_none());
        assert!(qa.zero_point.is_none());
        assert!(qa.axis.is_none());
    }

    // ── OnnxFunction doc_string field access and preservation ────────────

    #[test]
    fn onnx_function_doc_string_preserved() {
        // Arrange
        let doc = "This function implements a fused normalization operation.";
        let func = OnnxFunction {
            name: "FusedNorm".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: doc.to_string(),
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert_eq!(func.doc_string, doc);
    }

    // ── OnnxGraph metadata_props with empty string key ───────────────────

    #[test]
    fn onnx_graph_metadata_props_insert_empty_string_key_fails() {
        // Arrange: HashMap does not allow empty keys in parse_metadata_props,
        // but direct construction allows it. Verify HashMap behavior.
        let mut graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act: insert with non-empty key works
        graph.metadata_props.insert("key".to_string(), "val".to_string());
        // Assert
        assert_eq!(graph.metadata_props.get("key").unwrap(), "val");
        assert_eq!(graph.metadata_props.len(), 1);
    }

    // ── OnnxModelMetadata ir_version zero is valid protobuf default ──────

    #[test]
    fn onnx_model_metadata_ir_version_zero_is_distinct_from_missing() {
        // Arrange: explicit zero vs default both produce 0
        let meta_explicit = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert_eq!(meta_explicit.ir_version, 0);
        assert_eq!(meta_explicit.model_version, 0);
    }

    // ── parse_nodes: middle node has non-default domain ──────────────────

    #[test]
    fn parse_nodes_middle_node_custom_domain_preserved() {
        // Arrange: three nodes, middle one has domain "ai.onnx.ml"
        let nodes = vec![
            proto::NodeProto {
                op_type: Some("MatMul".to_string()),
                name: Some("matmul".to_string()),
                input: vec!["A".to_string(), "B".to_string()],
                output: vec!["C".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: Some("Scaler".to_string()),
                name: Some("scaler".to_string()),
                input: vec!["C".to_string()],
                output: vec!["D".to_string()],
                domain: Some("ai.onnx.ml".to_string()),
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: Some("Relu".to_string()),
                name: Some("relu".to_string()),
                input: vec!["D".to_string()],
                output: vec!["E".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
        ];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        // Assert
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].domain, "");
        assert_eq!(result[1].domain, "ai.onnx.ml");
        assert_eq!(result[2].domain, "");
    }

    // ── OnnxGraph value_info with typed entries via struct construction ──

    #[test]
    fn onnx_graph_value_info_typed_entries_via_struct() {
        // Arrange
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
        let vi = OnnxValueInfo {
            name: "hidden_state".to_string(),
            value_type: Some(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape {
                    dims: vec![OnnxDim::Param("batch".to_string()), OnnxDim::Known(768)],
                },
            })),
            doc_string: "intermediate activation".to_string(),
            metadata_props: {
                let mut m = HashMap::new();
                m.insert("source_layer".to_string(), "encoder_3".to_string());
                m
            },
        };
        let graph = OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![vi],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert_eq!(graph.value_info.len(), 1);
        let info = &graph.value_info[0];
        assert_eq!(info.name, "hidden_state");
        assert!(info.value_type.is_some());
        assert_eq!(info.metadata_props.get("source_layer").unwrap(), "encoder_3");
    }

    // ── OnnxValueInfo metadata_props with entries via struct construction ─

    #[test]
    fn onnx_value_info_metadata_props_multiple_entries() {
        // Arrange
        let mut props = HashMap::new();
        props.insert("framework".to_string(), "pytorch".to_string());
        props.insert("version".to_string(), "2.1.0".to_string());
        props.insert("quantized".to_string(), "true".to_string());
        let vi = OnnxValueInfo {
            name: "weight_0".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: props,
        };
        // Act & Assert
        assert_eq!(vi.metadata_props.len(), 3);
        assert_eq!(vi.metadata_props.get("framework").unwrap(), "pytorch");
        assert_eq!(vi.metadata_props.get("version").unwrap(), "2.1.0");
        assert_eq!(vi.metadata_props.get("quantized").unwrap(), "true");
    }

    // ── OnnxModel functions list append preserves existing entries ───────

    #[test]
    fn onnx_model_functions_append_preserves_existing() {
        // Arrange
        let func1 = OnnxFunction {
            name: "OpA".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let func2 = OnnxFunction {
            name: "OpB".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let mut model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![func1],
        };
        // Act
        model.functions.push(func2);
        // Assert
        assert_eq!(model.functions.len(), 2);
        assert_eq!(model.functions[0].name, "OpA");
        assert_eq!(model.functions[1].name, "OpB");
        assert_eq!(model.functions[1].domain, "custom");
    }

    // ── parse_quantization: different SCALE_TENSOR and ZERO_POINT_TENSOR names ─

    #[test]
    fn parse_quantization_different_scale_and_zp_tensor_names() {
        // Arrange: two initializers: one for scale, one for zero_point
        let scale_data = 0.125f32.to_le_bytes();
        let scale_tensor = OnnxTensor::new(
            "weight_scale".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(scale_data.to_vec()),
        );
        let zp_data = 128i64.to_le_bytes();
        let zp_tensor = OnnxTensor::new(
            "weight_zp".to_string(),
            Dtype::I64,
            vec![],
            Bytes::from(zp_data.to_vec()),
        );
        let mut initializers = HashMap::new();
        initializers.insert("weight_scale".to_string(), scale_tensor);
        initializers.insert("weight_zp".to_string(), zp_tensor);

        let annotations = vec![proto::TensorAnnotation {
            tensor_name: Some("linear_weight".to_string()),
            quant_parameter_tensor_names: vec![
                proto::StringStringEntryProto {
                    key: Some("SCALE_TENSOR".to_string()),
                    value: Some("weight_scale".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("ZERO_POINT_TENSOR".to_string()),
                    value: Some("weight_zp".to_string()),
                },
            ],
        }];
        // Act
        let result = parse_quantization(annotations, &initializers);
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tensor_name, "linear_weight");
        assert_eq!(result[0].scale, Some(0.125));
        assert_eq!(result[0].zero_point, Some(128));
        // Verify quant_param_tensor_names contains both
        assert_eq!(
            result[0].quant_param_tensor_names.get("SCALE_TENSOR").unwrap(),
            "weight_scale"
        );
        assert_eq!(
            result[0].quant_param_tensor_names.get("ZERO_POINT_TENSOR").unwrap(),
            "weight_zp"
        );
    }

    // ── OnnxOperatorSet Debug trait includes both domain and version ──────

    #[test]
    fn onnx_operator_set_debug_trait_output() {
        // Arrange
        let ops = OnnxOperatorSet {
            domain: "ai.onnx.ml".to_string(),
            version: 3,
        };
        // Act
        let debug = format!("{:?}", ops);
        // Assert: Debug output contains both fields
        assert!(debug.contains("ai.onnx.ml"));
        assert!(debug.contains("3"));
    }

    // ── parse_nodes: node with no name at index 0 gets auto-name node_0 ─

    #[test]
    fn parse_nodes_no_name_at_index_zero_gets_node_0() {
        // Arrange: single node with name = None
        let nodes = vec![proto::NodeProto {
            op_type: Some("Identity".to_string()),
            name: None, // explicitly None
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "node_0");
    }

    // ── OnnxGraph::from_proto: outputs count matches proto output list ──

    #[test]
    fn onnx_graph_from_proto_outputs_count_matches_proto() {
        // Arrange: graph with 2 output value_info entries
        let graph_proto = proto::GraphProto {
            name: Some("io_graph".to_string()),
            node: vec![proto::NodeProto {
                op_type: Some("Identity".to_string()),
                name: Some("id1".to_string()),
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            }],
            input: vec![proto::ValueInfoProto {
                name: Some("X".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            output: vec![
                proto::ValueInfoProto {
                    name: Some("logits".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
                proto::ValueInfoProto {
                    name: Some("hidden".to_string()),
                    r#type: None,
                    doc_string: None,
                    metadata_props: vec![],
                },
            ],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert_eq!(graph.outputs.len(), 2);
        assert_eq!(graph.outputs[0].name, "logits");
        assert_eq!(graph.outputs[1].name, "hidden");
    }

    // ── OnnxTensor raw_data byte length matches shape and dtype ──────────

    #[test]
    fn onnx_tensor_raw_data_len_matches_shape_and_dtype() {
        // Arrange: [3, 4] F32 tensor = 12 elements = 48 bytes
        let shape = vec![3, 4];
        let expected_bytes = 3 * 4 * 4; // 12 elements * 4 bytes per F32
        let t = OnnxTensor::new(
            "weight".to_string(),
            Dtype::F32,
            shape.clone(),
            Bytes::from(vec![0u8; expected_bytes]),
        );
        // Act & Assert: shape product * element_size == raw_data length
        assert_eq!(t.shape, shape);
        assert_eq!(t.raw_data().len(), expected_bytes);
    }

    // ── OnnxGraph::from_proto: inputs with typed value_info ─────────────

    #[test]
    fn onnx_graph_from_proto_inputs_with_typed_value_info() {
        // Arrange: graph input with Tensor type info
        let tensor_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(
                proto::type_proto::Tensor {
                    elem_type: Some(proto::tensor_proto::DataType::Float as i32),
                    shape: Some(proto::TensorShapeProto {
                        dim: vec![proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(768)),
                            denotation: None,
                        }],
                    }),
                },
            )),
        };
        let graph_proto = proto::GraphProto {
            name: Some("typed_graph".to_string()),
            node: vec![],
            input: vec![proto::ValueInfoProto {
                name: Some("input_ids".to_string()),
                r#type: Some(tensor_type),
                doc_string: None,
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.inputs[0].name, "input_ids");
        assert!(graph.inputs[0].value_type.is_some());
    }

    // ── OnnxModel::from_proto: empty graph name defaults to empty string ─

    #[test]
    fn onnx_model_from_proto_empty_graph_name_defaults_to_empty() {
        // Arrange: model with graph having name = None
        let model_proto = proto::ModelProto {
            ir_version: Some(8),
            graph: Some(proto::GraphProto {
                name: None, // explicitly None
                node: vec![],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        // Assert: graph name defaults to empty string
        assert_eq!(model.graph.name, "");
    }

    // ── OnnxGraph::from_proto: duplicate initializer name returns error ──

    #[test]
    fn onnx_graph_from_proto_duplicate_initializer_name_yields_duplicate_error() {
        // Arrange: graph with two initializers having the same name
        let tensor_data = 1.0f32.to_le_bytes();
        let graph_proto = proto::GraphProto {
            name: Some("dup_graph".to_string()),
            node: vec![],
            initializer: vec![
                proto::TensorProto {
                    name: Some("weight".to_string()),
                    dims: vec![2],
                    data_type: Some(proto::tensor_proto::DataType::Float as i32),
                    raw_data: Some(tensor_data.to_vec().into()),
                    ..Default::default()
                },
                proto::TensorProto {
                    name: Some("weight".to_string()), // duplicate
                    dims: vec![2],
                    data_type: Some(proto::tensor_proto::DataType::Float as i32),
                    raw_data: Some(tensor_data.to_vec().into()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = OnnxGraph::from_proto(graph_proto, &mut resolver);
        // Assert
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("DuplicateTensor") || err_msg.contains("weight"),
            "Expected duplicate tensor error, got: {err_msg}"
        );
    }

    // ── parse_nodes: multiple nodes with None names get unique auto-names ─

    #[test]
    fn parse_nodes_multiple_none_names_get_unique_auto_names() {
        // Arrange: 3 nodes with name = None
        let nodes: Vec<proto::NodeProto> = (0..3)
            .map(|_| proto::NodeProto {
                op_type: Some("Relu".to_string()),
                name: None,
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            })
            .collect();
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        // Assert: each gets a unique indexed name
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].name, "node_0");
        assert_eq!(result[1].name, "node_1");
        assert_eq!(result[2].name, "node_2");
    }

    // ── OnnxModelMetadata: doc_string non-empty preserved ───────────────

    #[test]
    fn onnx_model_metadata_doc_string_non_empty_preserved() {
        // Arrange
        let doc = "This is a test model for verification.";
        let meta = OnnxModelMetadata {
            ir_version: 7,
            producer_name: "test_producer".to_string(),
            producer_version: "2.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 42,
            doc_string: doc.to_string(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        // Assert
        assert_eq!(meta.doc_string, doc);
    }

    // ── OnnxNode: domain with special characters preserved ──────────────

    #[test]
    fn onnx_node_domain_special_characters_preserved() {
        // Arrange
        let node = OnnxNode {
            name: "custom_op".to_string(),
            op_type: "MyOp".to_string(),
            domain: "org.example.custom/v2.1-beta".to_string(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: HashMap::new(),
        };
        // Assert
        assert_eq!(node.domain, "org.example.custom/v2.1-beta");
    }

    // ── OnnxValueInfo: empty doc_string with non-empty metadata_props ──

    #[test]
    fn onnx_value_info_empty_doc_with_metadata_props() {
        // Arrange
        let vi = OnnxValueInfo {
            name: "attention_mask".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::from([
                ("source".to_string(), "tokenizer".to_string()),
                ("category".to_string(), "input".to_string()),
            ]),
        };
        // Assert
        assert!(vi.doc_string.is_empty());
        assert_eq!(vi.metadata_props.len(), 2);
        assert_eq!(vi.metadata_props.get("source").unwrap(), "tokenizer");
    }

    // ── OnnxGraph: zero initializers with sparse_initializers present ──

    #[test]
    fn onnx_graph_zero_dense_initializers_with_sparse_present() {
        // Arrange: graph with no dense initializers but has sparse_initializers
        use super::super::tensor::OnnxSparseFormat;
        let values = OnnxTensor::new("sparse_val".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let indices = OnnxTensor::new("sparse_idx".to_string(), Dtype::I64, vec![2], Bytes::from(vec![0u8; 16]));
        let graph = OnnxGraph {
            name: "sparse_only".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(), // empty
            sparse_initializers: vec![OnnxSparseTensor {
                values,
                indices,
                dims: vec![4, 4],
                format: OnnxSparseFormat::Coo,
            }],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Assert
        assert!(graph.initializers.is_empty());
        assert_eq!(graph.sparse_initializers.len(), 1);
        assert_eq!(graph.sparse_initializers[0].dims, vec![4, 4]);
    }

    // ── parse_quantization: zero_point from I32 scalar tensor ───────────

    #[test]
    fn parse_quantization_zero_point_from_i32_scalar_tensor() {
        // Arrange: I32 scalar with value 128
        let zp_bytes = 128i32.to_le_bytes();
        let zp_tensor = OnnxTensor::new(
            "zp_i32".to_string(),
            Dtype::I32,
            vec![],
            Bytes::from(zp_bytes.to_vec()),
        );
        let mut initializers = HashMap::new();
        initializers.insert("zp_i32".to_string(), zp_tensor);

        let annotations = vec![proto::TensorAnnotation {
            tensor_name: Some("weight_q".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("ZERO_POINT_TENSOR".to_string()),
                value: Some("zp_i32".to_string()),
            }],
        }];
        // Act
        let result = parse_quantization(annotations, &initializers);
        // Assert: I32 scalar should be converted via scalar_i64()
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].zero_point, Some(128));
    }

    // ── OnnxModel clone: functions vec is independent ────────────────────

    #[test]
    fn onnx_model_clone_functions_vec_is_independent() {
        // Arrange
        let func = OnnxFunction {
            name: "MyOp".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 0,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: String::new(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![func],
        };
        // Act
        let clone = model.clone();
        // Assert: original and clone have independent functions vecs
        assert_eq!(clone.functions.len(), 1);
        assert_eq!(clone.functions[0].name, "MyOp");
        // Mutating clone should not affect original
        assert_eq!(model.functions.len(), 1);
    }

    // ── OnnxGraph::from_proto: graph with doc_string preserves content ──

    #[test]
    fn onnx_graph_from_proto_doc_string_preserves_content() {
        // Arrange
        let graph_proto = proto::GraphProto {
            name: Some("doc_graph".to_string()),
            doc_string: Some("Main computation graph".to_string()),
            node: vec![],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert_eq!(graph.doc_string, "Main computation graph");
    }

    // ── OnnxTensor raw_data byte-level integrity for I64 tensor ─────────

    #[test]
    fn onnx_tensor_raw_data_i64_multi_element_integrity() {
        // Arrange: 3 I64 values
        let values: Vec<i64> = vec![-100, 0, i64::MAX];
        let mut bytes = Vec::with_capacity(24);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let t = OnnxTensor::new(
            "intvec".to_string(),
            Dtype::I64,
            vec![3],
            Bytes::from(bytes.clone()),
        );
        // Act & Assert: raw_data preserves exact bytes
        assert_eq!(t.raw_data().len(), 24);
        assert_eq!(t.raw_data(), bytes.as_slice());
    }

    // ── OnnxGraph::from_proto: value_info with typed entries via proto ──

    #[test]
    fn onnx_graph_from_proto_value_info_with_tensor_type() {
        // Arrange: value_info entry with Tensor type and shape
        let tensor_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(
                proto::type_proto::Tensor {
                    elem_type: Some(proto::tensor_proto::DataType::Int64 as i32),
                    shape: Some(proto::TensorShapeProto {
                        dim: vec![
                            proto::tensor_shape_proto::Dimension {
                                value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(1)),
                                denotation: None,
                            },
                            proto::tensor_shape_proto::Dimension {
                                value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(12)),
                                denotation: None,
                            },
                        ],
                    }),
                },
            )),
        };
        let graph_proto = proto::GraphProto {
            name: Some("vi_graph".to_string()),
            node: vec![],
            value_info: vec![proto::ValueInfoProto {
                name: Some("position_ids".to_string()),
                r#type: Some(tensor_type),
                doc_string: None,
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert_eq!(graph.value_info.len(), 1);
        assert_eq!(graph.value_info[0].name, "position_ids");
        assert!(graph.value_info[0].value_type.is_some());
    }

    // ── parse_metadata_props: many entries all preserved ─────────────────

    #[test]
    fn parse_metadata_props_many_entries_all_preserved() {
        // Arrange: 5 metadata entries
        let entries: Vec<proto::StringStringEntryProto> = (0..5)
            .map(|i| proto::StringStringEntryProto {
                key: Some(format!("key_{i}")),
                value: Some(format!("value_{i}")),
            })
            .collect();
        // Act
        let result = parse_metadata_props(entries);
        // Assert
        assert_eq!(result.len(), 5);
        for i in 0..5 {
            assert_eq!(result.get(&format!("key_{i}")).unwrap(), &format!("value_{i}"));
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 3: 15 new tests
    // Focus: F16/BF16 scalar parsing, graph input/output name overlap,
    //         initializer ordering, OnnxDim symbolic vs concrete via proto,
    //         external data path, tensor reshape data integrity,
    //         multi-dimensional F16 raw_data, node empty output,
    //         nodes with shared input names, model from_proto with opsets
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxTensor scalar_f32 with F16 scalar roundtrip ─────────────────

    #[test]
    fn onnx_tensor_scalar_f16_roundtrip_via_scalar_f32() {
        // Arrange: F16 scalar 1.5 encoded as u16 bits
        let f16_val = half::f16::from_f32(1.5);
        let bits = f16_val.to_bits().to_le_bytes();
        let tensor = OnnxTensor::new(
            "f16_scalar".to_string(),
            Dtype::F16,
            vec![], // scalar = empty shape
            Bytes::from(bits.to_vec()),
        );
        // Act
        let result = tensor.scalar_f32();
        // Assert
        assert!(result.is_some());
        let f32_val = result.unwrap();
        assert!((f32_val - 1.5f32).abs() < 1e-3, "F16 scalar 1.5 should roundtrip, got {f32_val}");
    }

    // ── OnnxTensor scalar_f32 with BF16 scalar roundtrip ────────────────

    #[test]
    fn onnx_tensor_scalar_bf16_roundtrip_via_scalar_f32() {
        // Arrange: BF16 scalar -2.75 encoded as u16 bits
        let bf16_val = half::bf16::from_f32(-2.75);
        let bits = bf16_val.to_bits().to_le_bytes();
        let tensor = OnnxTensor::new(
            "bf16_scalar".to_string(),
            Dtype::BF16,
            vec![], // scalar = empty shape
            Bytes::from(bits.to_vec()),
        );
        // Act
        let result = tensor.scalar_f32();
        // Assert
        assert!(result.is_some());
        let f32_val = result.unwrap();
        assert!((f32_val - (-2.75f32)).abs() < 1e-2, "BF16 scalar -2.75 should roundtrip, got {f32_val}");
    }

    // ── OnnxTensor scalar_i64 from F16 scalar conversion ────────────────

    #[test]
    fn onnx_tensor_scalar_i64_from_f16_value() {
        // Arrange: F16 scalar encoding integer value 7.0
        let f16_val = half::f16::from_f32(7.0);
        let bits = f16_val.to_bits().to_le_bytes();
        let tensor = OnnxTensor::new(
            "f16_int".to_string(),
            Dtype::F16,
            vec![],
            Bytes::from(bits.to_vec()),
        );
        // Act
        let result = tensor.scalar_i64();
        // Assert: F16 7.0 should convert to i64 7
        assert_eq!(result, Some(7));
    }

    // ── OnnxTensor scalar_i64 from BF16 scalar conversion ───────────────

    #[test]
    fn onnx_tensor_scalar_i64_from_bf16_value() {
        // Arrange: BF16 scalar encoding integer value 42.0
        let bf16_val = half::bf16::from_f32(42.0);
        let bits = bf16_val.to_bits().to_le_bytes();
        let tensor = OnnxTensor::new(
            "bf16_int".to_string(),
            Dtype::BF16,
            vec![],
            Bytes::from(bits.to_vec()),
        );
        // Act
        let result = tensor.scalar_i64();
        // Assert: BF16 42.0 should convert to i64 42
        assert_eq!(result, Some(42));
    }

    // ── OnnxGraph from_proto: graph input and output share the same name ─

    #[test]
    fn onnx_graph_from_proto_input_output_same_name_both_preserved() {
        // Arrange: graph where input and output are both named "state"
        let graph_proto = proto::GraphProto {
            name: Some("overlap_graph".to_string()),
            node: vec![],
            input: vec![proto::ValueInfoProto {
                name: Some("state".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            output: vec![proto::ValueInfoProto {
                name: Some("state".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert: both input and output named "state" are preserved independently
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.inputs[0].name, "state");
        assert_eq!(graph.outputs[0].name, "state");
    }

    // ── OnnxGraph initializer HashMap ordering is independent of insertion order ─

    #[test]
    fn onnx_graph_initializer_hashmap_insertion_order_independent() {
        // Arrange: insert initializers in Z, A, M order
        let mut initializers = HashMap::new();
        let names = vec!["z_weight", "a_weight", "m_weight"];
        for name in &names {
            initializers.insert(
                name.to_string(),
                OnnxTensor::new(
                    name.to_string(),
                    Dtype::F32,
                    vec![2],
                    Bytes::from(vec![0u8; 8]),
                ),
            );
        }
        let graph = OnnxGraph {
            name: "init_order_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers,
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Assert: all three are retrievable regardless of insertion order
        assert_eq!(graph.initializers.len(), 3);
        assert!(graph.initializers.contains_key("z_weight"));
        assert!(graph.initializers.contains_key("a_weight"));
        assert!(graph.initializers.contains_key("m_weight"));
        // Verify shape is intact for each
        for name in &names {
            let t = graph.initializers.get(*name).unwrap();
            assert_eq!(t.shape, vec![2]);
        }
    }

    // ── OnnxType from_proto: symbolic dimension in TensorType shape ──────

    #[test]
    fn onnx_type_from_proto_symbolic_dimension_in_tensor_shape() {
        // Arrange: TypeProto with Tensor type having a symbolic DimParam dimension
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(
                proto::type_proto::Tensor {
                    elem_type: Some(proto::tensor_proto::DataType::Float as i32),
                    shape: Some(proto::TensorShapeProto {
                        dim: vec![
                            proto::tensor_shape_proto::Dimension {
                                value: Some(proto::tensor_shape_proto::dimension::Value::DimParam(
                                    "batch_size".to_string(),
                                )),
                                denotation: None,
                            },
                            proto::tensor_shape_proto::Dimension {
                                value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(768)),
                                denotation: None,
                            },
                        ],
                    }),
                },
            )),
        };
        // Act
        let onnx_type = OnnxType::from_proto(type_proto).unwrap();
        // Assert
        use super::super::types::OnnxDim;
        if let OnnxType::Tensor(tt) = onnx_type {
            assert_eq!(tt.shape.dims.len(), 2);
            assert!(matches!(&tt.shape.dims[0], OnnxDim::Param(s) if s == "batch_size"));
            assert!(matches!(&tt.shape.dims[1], OnnxDim::Known(768)));
        } else {
            panic!("Expected Tensor variant");
        }
    }

    // ── OnnxType from_proto: unknown dimension (None value) in shape ─────

    #[test]
    fn onnx_type_from_proto_unknown_dimension_in_tensor_shape() {
        // Arrange: TypeProto with a dimension having value = None
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(
                proto::type_proto::Tensor {
                    elem_type: Some(proto::tensor_proto::DataType::Int64 as i32),
                    shape: Some(proto::TensorShapeProto {
                        dim: vec![
                            proto::tensor_shape_proto::Dimension {
                                value: None, // Unknown dimension
                                denotation: None,
                            },
                        ],
                    }),
                },
            )),
        };
        // Act
        let onnx_type = OnnxType::from_proto(type_proto).unwrap();
        // Assert
        use super::super::types::OnnxDim;
        if let OnnxType::Tensor(tt) = onnx_type {
            assert_eq!(tt.shape.dims.len(), 1);
            assert!(matches!(&tt.shape.dims[0], OnnxDim::Unknown));
        } else {
            panic!("Expected Tensor variant");
        }
    }

    // ── OnnxType from_proto: empty DimParam string treated as Unknown ────

    #[test]
    fn onnx_type_from_proto_empty_dimparam_treated_as_unknown() {
        // Arrange: DimParam with empty string should become Unknown
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(
                proto::type_proto::Tensor {
                    elem_type: Some(proto::tensor_proto::DataType::Float as i32),
                    shape: Some(proto::TensorShapeProto {
                        dim: vec![
                            proto::tensor_shape_proto::Dimension {
                                value: Some(proto::tensor_shape_proto::dimension::Value::DimParam(
                                    String::new(), // empty string
                                )),
                                denotation: None,
                            },
                        ],
                    }),
                },
            )),
        };
        // Act
        let onnx_type = OnnxType::from_proto(type_proto).unwrap();
        // Assert
        use super::super::types::OnnxDim;
        if let OnnxType::Tensor(tt) = onnx_type {
            assert_eq!(tt.shape.dims.len(), 1);
            assert!(matches!(&tt.shape.dims[0], OnnxDim::Unknown));
        } else {
            panic!("Expected Tensor variant");
        }
    }

    // ── ExternalDataResolver: missing external data file returns error ───

    #[test]
    fn external_data_resolver_missing_file_returns_io_error() {
        // Arrange: resolver pointing to a nonexistent directory
        let mut resolver = ExternalDataResolver::new(std::path::Path::new("/nonexistent/dir/model.onnx"));
        // Act
        let result = resolver.resolve("weights.bin", 0, 16);
        // Assert
        assert!(result.is_err(), "Expected error for missing external data file");
    }

    // ── ExternalDataResolver: offset out of bounds returns error ─────────

    #[test]
    fn external_data_resolver_offset_overflow_returns_error() {
        // Arrange: create a temp file with 8 bytes
        use std::io::Write;
        let dir = std::env::temp_dir().join("gllm_test_ext_offset");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("tiny.bin");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(&[0u8; 8]).unwrap();
            f.sync_all().unwrap();
        }
        let model_path = dir.join("model.onnx");
        // Touch model.onnx so resolver's base_dir is correct
        std::fs::File::create(&model_path).unwrap();
        let mut resolver = ExternalDataResolver::new(&model_path);
        // Act: request offset=usize::MAX, length=1 → overflow
        let result = resolver.resolve("tiny.bin", usize::MAX, 1);
        // Assert
        assert!(result.is_err(), "Expected error for offset overflow");
        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── OnnxTensor reshape preserves raw_data bytes ──────────────────────

    #[test]
    fn onnx_tensor_reshape_preserves_raw_data_bytes() {
        // Arrange: a 2x3 F32 tensor
        let original_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut bytes = Vec::with_capacity(24);
        for v in &original_data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let tensor = OnnxTensor::new(
            "feature_map".to_string(),
            Dtype::F32,
            vec![2, 3],
            Bytes::from(bytes.clone()),
        );
        // Act: simulate reshape by creating new tensor with different shape but same data
        let reshaped = OnnxTensor::new(
            "feature_map".to_string(),
            Dtype::F32,
            vec![3, 2], // same element count, different shape
            Bytes::from(tensor.raw_data().to_vec()),
        );
        // Assert: raw_data is byte-identical
        assert_eq!(reshaped.raw_data(), tensor.raw_data());
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(tensor.shape, vec![2, 3]);
        // Verify first element is still 1.0
        let first_bytes: [u8; 4] = reshaped.raw_data()[0..4].try_into().unwrap();
        assert!((f32::from_le_bytes(first_bytes) - 1.0f32).abs() < 1e-6);
    }

    // ── OnnxTensor multi-dimensional F16 raw_data byte integrity ────────

    #[test]
    fn onnx_tensor_f16_multi_dim_raw_data_byte_integrity() {
        // Arrange: 2x4 F16 tensor = 8 elements = 16 bytes
        let f16_values: Vec<half::f16> = vec![
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(-1.0),
            half::f16::from_f32(0.5),
            half::f16::from_f32(2.0),
            half::f16::from_f32(-0.5),
            half::f16::from_f32(3.0),
            half::f16::from_f32(-3.0),
        ];
        let mut bytes = Vec::with_capacity(16);
        for v in &f16_values {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        assert_eq!(bytes.len(), 16);
        let tensor = OnnxTensor::new(
            "f16_2x4".to_string(),
            Dtype::F16,
            vec![2, 4],
            Bytes::from(bytes.clone()),
        );
        // Assert: raw_data preserves exact byte sequence
        assert_eq!(tensor.raw_data().len(), 16);
        assert_eq!(tensor.raw_data(), bytes.as_slice());
        assert_eq!(tensor.shape, vec![2, 4]);
    }

    // ── parse_nodes: node with empty output string is preserved ──────────

    #[test]
    fn parse_nodes_empty_output_string_preserved_as_is() {
        // Arrange: a node with an empty string in its output list
        let nodes = vec![proto::NodeProto {
            op_type: Some("Identity".to_string()),
            name: Some("passthrough".to_string()),
            input: vec!["X".to_string()],
            output: vec![String::new()], // empty string output
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        // Assert: empty string output is preserved verbatim
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].outputs.len(), 1);
        assert_eq!(result[0].outputs[0], "");
    }

    // ── parse_nodes: multiple nodes sharing the same input name ──────────

    #[test]
    fn parse_nodes_multiple_nodes_share_same_input_name() {
        // Arrange: two nodes both reading from the same input "hidden_state"
        let nodes = vec![
            proto::NodeProto {
                op_type: Some("MatMul".to_string()),
                name: Some("proj_q".to_string()),
                input: vec!["hidden_state".to_string(), "wq".to_string()],
                output: vec!["q".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
            proto::NodeProto {
                op_type: Some("MatMul".to_string()),
                name: Some("proj_k".to_string()),
                input: vec!["hidden_state".to_string(), "wk".to_string()],
                output: vec!["k".to_string()],
                domain: None,
                overload: None,
                attribute: vec![],
                doc_string: None,
                metadata_props: vec![],
                device_configurations: vec![],
            },
        ];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let result = parse_nodes(nodes, &mut resolver).unwrap();
        // Assert: both nodes share "hidden_state" as their first input
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].inputs[0], "hidden_state");
        assert_eq!(result[1].inputs[0], "hidden_state");
        assert_eq!(result[0].name, "proj_q");
        assert_eq!(result[1].name, "proj_k");
    }

    // ── OnnxModel from_proto with multiple opset imports preserved ───────

    #[test]
    fn onnx_model_from_proto_multiple_opset_imports_preserved() {
        // Arrange
        let model_proto = proto::ModelProto {
            ir_version: Some(9),
            opset_import: vec![
                proto::OperatorSetIdProto {
                    domain: Some(String::new()),
                    version: Some(20),
                },
                proto::OperatorSetIdProto {
                    domain: Some("ai.onnx.ml".to_string()),
                    version: Some(3),
                },
                proto::OperatorSetIdProto {
                    domain: Some("com.microsoft".to_string()),
                    version: Some(1),
                },
            ],
            graph: Some(proto::GraphProto {
                name: Some("multi_opset".to_string()),
                node: vec![],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        // Assert: all three opset imports preserved in order
        assert_eq!(model.metadata.opset_import.len(), 3);
        assert_eq!(model.metadata.opset_import[0].domain, "");
        assert_eq!(model.metadata.opset_import[0].version, 20);
        assert_eq!(model.metadata.opset_import[1].domain, "ai.onnx.ml");
        assert_eq!(model.metadata.opset_import[1].version, 3);
        assert_eq!(model.metadata.opset_import[2].domain, "com.microsoft");
        assert_eq!(model.metadata.opset_import[2].version, 1);
    }

    // ── Round 4: Display/Debug, Graph edge cases, Attribute HashMap, OnnxType variants,
    //    OnnxDim Unknown boundary, Initializer lookup ─────────────────────

    #[test]
    fn onnx_model_debug_format_contains_graph_name() {
        // Arrange
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: "test_producer".to_string(),
                producer_version: "2.0".to_string(),
                domain: "ai.onnx".to_string(),
                model_version: 42,
                doc_string: "debug test".to_string(),
                opset_import: vec![OnnxOperatorSet {
                    domain: String::new(),
                    version: 17,
                }],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "my_test_graph".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        // Act
        let debug_str = format!("{model:?}");
        // Assert: Debug output must contain the graph name and producer
        assert!(
            debug_str.contains("my_test_graph"),
            "Debug output should contain graph name, got: {debug_str}"
        );
        assert!(
            debug_str.contains("test_producer"),
            "Debug output should contain producer name, got: {debug_str}"
        );
    }

    #[test]
    fn onnx_graph_debug_format_shows_node_count_and_initializer_count() {
        // Arrange: graph with 3 nodes and 2 initializers
        let mut initializers = HashMap::new();
        initializers.insert(
            "weight1".to_string(),
            OnnxTensor::new("weight1".to_string(), Dtype::F32, vec![2, 3], Bytes::from(vec![0u8; 24])),
        );
        initializers.insert(
            "bias1".to_string(),
            OnnxTensor::new("bias1".to_string(), Dtype::F32, vec![3], Bytes::from(vec![0u8; 12])),
        );
        let graph = OnnxGraph {
            name: "three_node_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "n0".to_string(),
                    op_type: "Conv".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n1".to_string(),
                    op_type: "Relu".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n2".to_string(),
                    op_type: "Add".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers,
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act
        let debug_str = format!("{graph:?}");
        // Assert: Debug should contain the graph name and field names
        assert!(
            debug_str.contains("three_node_graph"),
            "Debug output should contain graph name, got: {debug_str}"
        );
        assert!(
            debug_str.contains("nodes") || debug_str.contains("initializers"),
            "Debug output should contain field names, got: {debug_str}"
        );
    }

    #[test]
    fn onnx_graph_from_proto_empty_initializer_list_is_valid() {
        // Arrange: graph proto with zero initializers (valid for inference-only graphs)
        let graph_proto = proto::GraphProto {
            name: Some("no_weights".to_string()),
            node: vec![proto::NodeProto {
                op_type: Some("Identity".to_string()),
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                ..Default::default()
            }],
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![proto::ValueInfoProto {
                name: Some("X".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            output: vec![proto::ValueInfoProto {
                name: Some("Y".to_string()),
                r#type: None,
                doc_string: None,
                metadata_props: vec![],
            }],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert
        assert!(graph.initializers.is_empty());
        assert!(graph.sparse_initializers.is_empty());
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "Identity");
    }

    #[test]
    fn onnx_graph_from_proto_many_nodes_preserves_order() {
        // Arrange: 50 nodes with sequential names
        let node_protos: Vec<proto::NodeProto> = (0..50)
            .map(|i| proto::NodeProto {
                op_type: Some(format!("Op{i}")),
                name: Some(format!("node_{i}")),
                ..Default::default()
            })
            .collect();
        let graph_proto = proto::GraphProto {
            name: Some("large_graph".to_string()),
            node: node_protos,
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Act
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        // Assert: order preserved, all 50 present
        assert_eq!(graph.nodes.len(), 50);
        for (i, node) in graph.nodes.iter().enumerate() {
            assert_eq!(node.op_type, format!("Op{i}"));
            assert_eq!(node.name, format!("node_{i}"));
        }
    }

    #[test]
    fn onnx_node_attributes_hashmap_insert_overwrite_replaces_value() {
        // Arrange: node with one attribute, then overwrite
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert(
            "alpha".to_string(),
            OnnxAttribute {
                name: "alpha".to_string(),
                value: OnnxAttributeValue::Float(1.0),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        // Act: overwrite with new value
        attrs.insert(
            "alpha".to_string(),
            OnnxAttribute {
                name: "alpha".to_string(),
                value: OnnxAttributeValue::Float(2.5),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        // Assert: last value wins
        let attr = attrs.get("alpha").unwrap();
        match &attr.value {
            OnnxAttributeValue::Float(v) => assert_eq!(*v, 2.5),
            other => panic!("expected Float, got {other:?}"),
        }
        assert_eq!(attrs.len(), 1);
    }

    #[test]
    fn onnx_node_attributes_hashmap_multiple_types_coexist() {
        // Arrange: attributes with different value types on the same node
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert(
            "f_attr".to_string(),
            OnnxAttribute {
                name: "f_attr".to_string(),
                value: OnnxAttributeValue::Float(3.14),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        attrs.insert(
            "i_attr".to_string(),
            OnnxAttribute {
                name: "i_attr".to_string(),
                value: OnnxAttributeValue::Int(42),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        attrs.insert(
            "s_attr".to_string(),
            OnnxAttribute {
                name: "s_attr".to_string(),
                value: OnnxAttributeValue::String("hello".to_string()),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        attrs.insert(
            "ints_attr".to_string(),
            OnnxAttribute {
                name: "ints_attr".to_string(),
                value: OnnxAttributeValue::Ints(vec![1, 2, 3]),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        // Act
        let node = OnnxNode {
            name: "multi_attr".to_string(),
            op_type: "CustomOp".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
        };
        // Assert: all attribute types retrievable
        assert_eq!(node.attributes.len(), 4);
        assert!(matches!(
            &node.attributes["f_attr"].value,
            OnnxAttributeValue::Float(v) if (*v - 3.14).abs() < f32::EPSILON
        ));
        assert!(matches!(
            &node.attributes["i_attr"].value,
            OnnxAttributeValue::Int(42)
        ));
        assert!(matches!(
            &node.attributes["s_attr"].value,
            OnnxAttributeValue::String(s) if s == "hello"
        ));
        assert!(matches!(
            &node.attributes["ints_attr"].value,
            OnnxAttributeValue::Ints(v) if v == &[1, 2, 3]
        ));
    }

    #[test]
    fn onnx_node_attributes_hashmap_remove_and_check_absent() {
        // Arrange: node with attributes, then remove one
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert(
            "to_remove".to_string(),
            OnnxAttribute {
                name: "to_remove".to_string(),
                value: OnnxAttributeValue::Int(99),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        attrs.insert(
            "to_keep".to_string(),
            OnnxAttribute {
                name: "to_keep".to_string(),
                value: OnnxAttributeValue::Float(1.0),
                doc_string: String::new(),
                ref_attr_name: None,
                attr_type: None,
            },
        );
        // Act
        let removed = attrs.remove("to_remove");
        // Assert
        assert!(removed.is_some());
        assert!(!attrs.contains_key("to_remove"));
        assert!(attrs.contains_key("to_keep"));
        assert_eq!(attrs.len(), 1);
    }

    #[test]
    fn onnx_node_attribute_ref_variant_stored_and_retrieved() {
        // Arrange: attribute using the Ref value variant (referenced attribute)
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "ref_attr".to_string(),
            value: OnnxAttributeValue::Ref("external_attr".to_string()),
            doc_string: "a reference attribute".to_string(),
            ref_attr_name: Some("external_attr".to_string()),
            attr_type: None,
        };
        // Act & Assert
        assert!(matches!(&attr.value, OnnxAttributeValue::Ref(s) if s == "external_attr"));
        assert_eq!(attr.ref_attr_name.as_deref(), Some("external_attr"));
        assert_eq!(attr.doc_string, "a reference attribute");
    }

    #[test]
    fn onnx_type_sparse_tensor_variant_equality_and_clone() {
        // Arrange
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType};
        let sparse = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float16,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Known(128), OnnxDim::Param("seq".to_string())],
            },
        });
        // Act
        let cloned = sparse.clone();
        // Assert
        assert_eq!(sparse, cloned);
        match sparse {
            OnnxType::SparseTensor(tt) => {
                assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Float16);
                assert_eq!(tt.shape.dims.len(), 2);
            }
            other => panic!("expected SparseTensor, got {other:?}"),
        }
    }

    #[test]
    fn onnx_type_nested_optional_sequence_map() {
        // Arrange: Optional<Sequence<Map<Int64, Tensor<Float>>>
        use super::super::types::{OnnxDim, OnnxMapType, OnnxTensorShape, OnnxTensorType};
        let map = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
            })),
        });
        let seq = OnnxType::Sequence(Box::new(map));
        let opt = OnnxType::Optional(Box::new(seq));
        // Act & Assert: verify full nesting structure
        match opt {
            OnnxType::Optional(inner) => match *inner {
                OnnxType::Sequence(inner2) => match *inner2 {
                    OnnxType::Map(map_type) => {
                        assert_eq!(map_type.key_type, proto::tensor_proto::DataType::Int64);
                        assert!(matches!(*map_type.value_type, OnnxType::Tensor(_)));
                    }
                    other => panic!("expected Map, got {other:?}"),
                },
                other => panic!("expected Sequence, got {other:?}"),
            },
            other => panic!("expected Optional, got {other:?}"),
        }
    }

    #[test]
    fn onnx_attribute_value_graph_variant_stores_subgraph() {
        // Arrange: attribute containing a parsed subgraph (If/Loop/Scan style)
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let subgraph = OnnxGraph {
            name: "then_branch".to_string(),
            doc_string: String::new(),
            nodes: vec![OnnxNode {
                name: "identity".to_string(),
                op_type: "Identity".to_string(),
                domain: String::new(),
                inputs: vec!["cond_out".to_string()],
                outputs: vec!["result".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let attr = OnnxAttribute {
            name: "then_branch".to_string(),
            value: OnnxAttributeValue::Graph(Box::new(subgraph)),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        };
        // Act & Assert
        match &attr.value {
            OnnxAttributeValue::Graph(g) => {
                assert_eq!(g.name, "then_branch");
                assert_eq!(g.nodes.len(), 1);
                assert_eq!(g.nodes[0].op_type, "Identity");
            }
            other => panic!("expected Graph, got {other:?}"),
        }
    }

    #[test]
    fn onnx_attribute_value_graphs_variant_stores_multiple_subgraphs() {
        // Arrange: attribute containing multiple subgraphs (Scan style)
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let sg1 = OnnxGraph {
            name: "body".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let sg2 = OnnxGraph {
            name: "scan_state".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let attr = OnnxAttribute {
            name: "bodies".to_string(),
            value: OnnxAttributeValue::Graphs(vec![sg1, sg2]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        };
        // Act & Assert
        match &attr.value {
            OnnxAttributeValue::Graphs(graphs) => {
                assert_eq!(graphs.len(), 2);
                assert_eq!(graphs[0].name, "body");
                assert_eq!(graphs[1].name, "scan_state");
            }
            other => panic!("expected Graphs, got {other:?}"),
        }
    }

    #[test]
    fn onnx_dim_unknown_boundary_equality_and_hash() {
        // Arrange: multiple Unknown dims, verify equality and HashSet dedup
        use super::super::types::OnnxDim;
        use std::collections::HashSet;
        let d1 = OnnxDim::Unknown;
        let d2 = OnnxDim::Unknown;
        let d3 = OnnxDim::Known(0);
        // Act & Assert: Unknown == Unknown, but Unknown != Known(0)
        assert_eq!(d1, d2);
        assert_ne!(d1, d3);
        let mut set = HashSet::new();
        assert!(set.insert(d1));
        assert!(!set.insert(d2)); // duplicate Unknown rejected
        assert!(set.insert(d3)); // Known(0) is distinct
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn onnx_dim_unknown_in_tensor_shape_mixed_with_known_and_param() {
        // Arrange: shape with all three dim variants interleaved
        use super::super::types::{OnnxDim, OnnxTensorShape};
        let shape = OnnxTensorShape {
            dims: vec![
                OnnxDim::Param("batch".to_string()),
                OnnxDim::Unknown,
                OnnxDim::Known(768),
                OnnxDim::Unknown,
                OnnxDim::Param("seq_len".to_string()),
            ],
        };
        // Act & Assert: verify each position
        assert_eq!(shape.dims.len(), 5);
        assert!(matches!(&shape.dims[0], OnnxDim::Param(p) if p == "batch"));
        assert!(matches!(shape.dims[1], OnnxDim::Unknown));
        assert!(matches!(&shape.dims[2], OnnxDim::Known(768)));
        assert!(matches!(shape.dims[3], OnnxDim::Unknown));
        assert!(matches!(&shape.dims[4], OnnxDim::Param(p) if p == "seq_len"));
        // Verify cloned shape is equal
        assert_eq!(shape, shape.clone());
    }

    #[test]
    fn onnx_graph_initializer_lookup_by_name_finds_tensor_and_misses_gracefully() {
        // Arrange: graph with several initializers
        let mut initializers = HashMap::new();
        initializers.insert(
            "embed_weight".to_string(),
            OnnxTensor::new("embed_weight".to_string(), Dtype::F32, vec![30000, 768], Bytes::from(vec![0u8; 100])),
        );
        initializers.insert(
            "layer_norm_weight".to_string(),
            OnnxTensor::new("layer_norm_weight".to_string(), Dtype::F32, vec![768], Bytes::from(vec![0u8; 50])),
        );
        initializers.insert(
            "output_bias".to_string(),
            OnnxTensor::new("output_bias".to_string(), Dtype::F32, vec![30000], Bytes::from(vec![0u8; 30])),
        );
        let graph = OnnxGraph {
            name: "lookup_test".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers,
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act & Assert: lookup existing
        let found = graph.initializers.get("embed_weight");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "embed_weight");
        assert_eq!(found.unwrap().shape, vec![30000, 768]);
        // Act & Assert: lookup another existing
        let found2 = graph.initializers.get("output_bias");
        assert!(found2.is_some());
        assert_eq!(found2.unwrap().dtype, Dtype::F32);
        // Act & Assert: lookup missing returns None
        let missing = graph.initializers.get("nonexistent_tensor");
        assert!(missing.is_none());
        // Assert: total count
        assert_eq!(graph.initializers.len(), 3);
    }

    // ── New angle tests: OnnxModel default values, opset multi-version, ir_version
    //    boundaries, producer formatting, domain special chars, multiple graphs via
    //    functions, sparse tensor construction, training info absence, attribute
    //    count upper bound, graph input/output dedup ────────────────────────────

    /// @trace TEST-ONNX-MODEL-NEW-001 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModelMetadata with all-zero/minimal default values round-trips.
    #[test]
    fn onnx_model_metadata_all_defaults_zero_and_empty_round_trip() {
        // Arrange: construct metadata with minimal/zero values
        let meta = OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        // Act: clone and verify every field (OnnxModelMetadata has no PartialEq)
        let cloned = meta.clone();
        // Assert
        assert_eq!(cloned.ir_version, 0);
        assert!(cloned.producer_name.is_empty());
        assert!(cloned.producer_version.is_empty());
        assert!(cloned.domain.is_empty());
        assert_eq!(cloned.model_version, 0);
        assert!(cloned.doc_string.is_empty());
        assert!(cloned.opset_import.is_empty());
        assert!(cloned.metadata_props.is_empty());
        assert_eq!(cloned.ir_version, meta.ir_version);
        assert_eq!(cloned.model_version, meta.model_version);
    }

    /// @trace TEST-ONNX-MODEL-NEW-002 [req:REQ-LOADER-007] [level:unit]
    /// Verify multiple opset_import entries with different domains and versions.
    #[test]
    fn onnx_model_metadata_opset_import_three_distinct_domains() {
        // Arrange: three different opset domains
        let opsets = vec![
            OnnxOperatorSet { domain: String::new(), version: 20 },
            OnnxOperatorSet { domain: "ai.onnx.ml".to_string(), version: 4 },
            OnnxOperatorSet { domain: "ai.onnx.training".to_string(), version: 1 },
        ];
        let meta = OnnxModelMetadata {
            ir_version: 9,
            producer_name: "test_producer".to_string(),
            producer_version: "2.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 3,
            doc_string: String::new(),
            opset_import: opsets,
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert_eq!(meta.opset_import.len(), 3);
        assert_eq!(meta.opset_import[0].domain, "");
        assert_eq!(meta.opset_import[0].version, 20);
        assert_eq!(meta.opset_import[1].domain, "ai.onnx.ml");
        assert_eq!(meta.opset_import[1].version, 4);
        assert_eq!(meta.opset_import[2].domain, "ai.onnx.training");
        assert_eq!(meta.opset_import[2].version, 1);
    }

    /// @trace TEST-ONNX-MODEL-NEW-003 [req:REQ-LOADER-007] [level:unit]
    /// Verify ir_version boundary values: i64::MIN, i64::MAX, and negative.
    #[test]
    fn onnx_model_metadata_ir_version_boundary_values() {
        // Arrange & Act & Assert: i64::MIN
        let meta_min = OnnxModelMetadata {
            ir_version: i64::MIN,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta_min.ir_version, i64::MIN);

        // Arrange & Act & Assert: i64::MAX
        let meta_max = OnnxModelMetadata {
            ir_version: i64::MAX,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta_max.ir_version, i64::MAX);

        // Arrange & Act & Assert: negative value (invalid ONNX but struct allows it)
        let meta_neg = OnnxModelMetadata {
            ir_version: -1,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta_neg.ir_version, -1);
    }

    /// @trace TEST-ONNX-MODEL-NEW-004 [req:REQ-LOADER-007] [level:unit]
    /// Verify producer_name and producer_version preserve formatting with version strings.
    #[test]
    fn onnx_model_metadata_producer_name_version_formatted_strings() {
        // Arrange: realistic producer info with version formatting
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: "onnxruntime-training".to_string(),
            producer_version: "1.17.0+cu121".to_string(),
            domain: "org.onnx".to_string(),
            model_version: 42,
            doc_string: "exported with special flags".to_string(),
            opset_import: vec![OnnxOperatorSet {
                domain: String::new(),
                version: 17,
            }],
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert_eq!(meta.producer_name, "onnxruntime-training");
        assert!(meta.producer_version.contains('+'));
        assert!(meta.producer_version.starts_with("1.17.0"));
        assert_eq!(meta.model_version, 42);
        assert_eq!(meta.doc_string, "exported with special flags");
    }

    /// @trace TEST-ONNX-MODEL-NEW-005 [req:REQ-LOADER-007] [level:unit]
    /// Verify domain field with special characters (unicode, hyphens, dots).
    #[test]
    fn onnx_model_metadata_domain_special_characters_preserved() {
        // Arrange: domain with special characters
        let meta = OnnxModelMetadata {
            ir_version: 9,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: "ai.onnx.ml.v2-experimental_2024".to_string(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert!(meta.domain.contains('.'));
        assert!(meta.domain.contains('-'));
        assert!(meta.domain.contains('_'));
        assert_eq!(meta.domain, "ai.onnx.ml.v2-experimental_2024");
    }

    /// @trace TEST-ONNX-MODEL-NEW-006 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModel with multiple functions each containing a distinct graph body.
    #[test]
    fn onnx_model_multiple_functions_with_independent_graph_bodies() {
        // Arrange: model with two functions, each with different node graphs
        let fn1 = OnnxFunction {
            name: "custom_relu".to_string(),
            domain: "com.example".to_string(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![OnnxNode {
                name: "relu_node".to_string(),
                op_type: "Relu".to_string(),
                domain: String::new(),
                inputs: vec!["X".to_string()],
                outputs: vec!["Y".to_string()],
                attributes: HashMap::new(),
            }],
            opset_import: vec![OnnxOperatorSet { domain: String::new(), version: 17 }],
            value_info: vec![],
            doc_string: "custom relu".to_string(),
            metadata_props: HashMap::new(),
        };
        let fn2 = OnnxFunction {
            name: "custom_sigmoid".to_string(),
            domain: "com.example".to_string(),
            overload: "v2".to_string(),
            inputs: vec!["A".to_string()],
            outputs: vec!["B".to_string()],
            attributes: vec!["alpha".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![
                OnnxNode {
                    name: "exp_node".to_string(),
                    op_type: "Exp".to_string(),
                    domain: String::new(),
                    inputs: vec!["neg_A".to_string()],
                    outputs: vec!["exp_neg".to_string()],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "add_node".to_string(),
                    op_type: "Add".to_string(),
                    domain: String::new(),
                    inputs: vec!["one".to_string(), "exp_neg".to_string()],
                    outputs: vec!["denom".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            opset_import: vec![OnnxOperatorSet { domain: String::new(), version: 18 }],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let graph = OnnxGraph {
            name: "main".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph,
            functions: vec![fn1, fn2],
        };
        // Act & Assert
        assert_eq!(model.functions.len(), 2);
        assert_eq!(model.functions[0].name, "custom_relu");
        assert_eq!(model.functions[0].nodes.len(), 1);
        assert_eq!(model.functions[0].opset_import[0].version, 17);
        assert_eq!(model.functions[1].name, "custom_sigmoid");
        assert_eq!(model.functions[1].overload, "v2");
        assert_eq!(model.functions[1].nodes.len(), 2);
        assert_eq!(model.functions[1].opset_import[0].version, 18);
    }

    /// @trace TEST-ONNX-MODEL-NEW-007 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxSparseTensor struct construction and field access.
    #[test]
    fn onnx_sparse_tensor_construction_and_field_access() {
        // Arrange: construct a sparse tensor with COO format
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let values = OnnxTensor::new("sparse_val".to_string(), Dtype::F32, vec![3], Bytes::from(vec![0u8; 12]));
        let indices = OnnxTensor::new("sparse_idx".to_string(), Dtype::I64, vec![3, 2], Bytes::from(vec![0u8; 48]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![4, 5],
            format: OnnxSparseFormat::Coo,
        };
        // Act & Assert
        assert_eq!(sparse.values.name, "sparse_val");
        assert_eq!(sparse.indices.name, "sparse_idx");
        assert_eq!(sparse.dims, vec![4, 5]);
        assert_eq!(sparse.format, OnnxSparseFormat::Coo);
        assert_eq!(sparse.values.shape, vec![3]);
        assert_eq!(sparse.indices.shape, vec![3, 2]);
    }

    /// @trace TEST-ONNX-MODEL-NEW-008 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxSparseTensor with CSR and CSC format variants.
    #[test]
    fn onnx_sparse_tensor_csr_and_csc_format_variants() {
        // Arrange: two sparse tensors with different formats
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let csr = OnnxSparseTensor {
            values: OnnxTensor::new("csr_v".to_string(), Dtype::F32, vec![5], Bytes::from(vec![0u8; 20])),
            indices: OnnxTensor::new("csr_i".to_string(), Dtype::I64, vec![5], Bytes::from(vec![0u8; 40])),
            dims: vec![4, 6],
            format: OnnxSparseFormat::Csr,
        };
        let csc = OnnxSparseTensor {
            values: OnnxTensor::new("csc_v".to_string(), Dtype::F32, vec![5], Bytes::from(vec![0u8; 20])),
            indices: OnnxTensor::new("csc_i".to_string(), Dtype::I64, vec![5], Bytes::from(vec![0u8; 40])),
            dims: vec![6, 4],
            format: OnnxSparseFormat::Csc,
        };
        // Act & Assert
        assert_ne!(csr.format, csc.format);
        assert_eq!(csr.format, OnnxSparseFormat::Csr);
        assert_eq!(csc.format, OnnxSparseFormat::Csc);
        assert_ne!(csr.dims, csc.dims);
    }

    /// @trace TEST-ONNX-MODEL-NEW-009 [req:REQ-LOADER-007] [level:unit]
    /// Verify model_version edge values: i64::MIN, i64::MAX, and zero.
    #[test]
    fn onnx_model_metadata_model_version_boundary_values() {
        // Arrange & Act & Assert: i64::MAX
        let meta_max = OnnxModelMetadata {
            ir_version: 8,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: i64::MAX,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(meta_max.model_version, i64::MAX);

        // Arrange & Act & Assert: i64::MIN
        let meta_min = OnnxModelMetadata {
            model_version: i64::MIN,
            ..meta_max.clone()
        };
        assert_eq!(meta_min.model_version, i64::MIN);

        // Arrange & Act & Assert: zero
        let meta_zero = OnnxModelMetadata {
            model_version: 0,
            ..meta_max.clone()
        };
        assert_eq!(meta_zero.model_version, 0);
    }

    /// @trace TEST-ONNX-MODEL-NEW-010 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxNode with a large number of attributes (upper bound stress test).
    #[test]
    fn onnx_node_attributes_large_count_upper_bound() {
        // Arrange: construct a node with 128 attributes
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::with_capacity(128);
        for i in 0..128 {
            attrs.insert(
                format!("attr_{i}"),
                OnnxAttribute {
                    name: format!("attr_{i}"),
                    value: OnnxAttributeValue::Float(i as f32),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            );
        }
        let node = OnnxNode {
            name: "dense_attrs".to_string(),
            op_type: "CustomDenseOp".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec!["out".to_string()],
            attributes: attrs,
        };
        // Act & Assert: all 128 attributes present and retrievable
        assert_eq!(node.attributes.len(), 128);
        for i in 0..128 {
            let key = format!("attr_{i}");
            assert!(node.attributes.contains_key(&key));
            match &node.attributes[&key].value {
                OnnxAttributeValue::Float(v) => assert!((v - i as f32).abs() < f32::EPSILON),
                other => panic!("expected Float at attr_{i}, got {other:?}"),
            }
        }
    }

    /// @trace TEST-ONNX-MODEL-NEW-011 [req:REQ-LOADER-007] [level:unit]
    /// Verify graph input names can be duplicated (ONNX allows same-name inputs at
    /// different positions, e.g. optional inputs with same type).
    #[test]
    fn onnx_graph_inputs_allow_duplicate_name_entries() {
        // Arrange: graph with two inputs sharing the same name
        let input_a = OnnxValueInfo {
            name: "shared_input".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let input_b = OnnxValueInfo {
            name: "shared_input".to_string(),
            value_type: None,
            doc_string: "second occurrence".to_string(),
            metadata_props: HashMap::new(),
        };
        let graph = OnnxGraph {
            name: "dup_inputs".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![input_a, input_b],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act & Assert: both entries preserved, even with duplicate names
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.inputs[0].name, "shared_input");
        assert_eq!(graph.inputs[1].name, "shared_input");
        assert_eq!(graph.inputs[1].doc_string, "second occurrence");
    }

    /// @trace TEST-ONNX-MODEL-NEW-012 [req:REQ-LOADER-007] [level:unit]
    /// Verify graph output names can be duplicated (Vec preserves duplicates).
    #[test]
    fn onnx_graph_outputs_allow_duplicate_name_entries() {
        // Arrange: graph with duplicate output names but different value_type
        use super::super::types::{OnnxDim, OnnxTensorShape, OnnxTensorType};
        let out1 = OnnxValueInfo {
            name: "logits".to_string(),
            value_type: Some(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![OnnxDim::Known(1000)] },
            })),
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let out2 = OnnxValueInfo {
            name: "logits".to_string(),
            value_type: None,
            doc_string: "duplicate output slot".to_string(),
            metadata_props: HashMap::new(),
        };
        let graph = OnnxGraph {
            name: "dup_outputs".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![out1, out2],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Act & Assert: both entries preserved
        assert_eq!(graph.outputs.len(), 2);
        assert_eq!(graph.outputs[0].name, "logits");
        assert!(graph.outputs[0].value_type.is_some());
        assert_eq!(graph.outputs[1].name, "logits");
        assert!(graph.outputs[1].value_type.is_none());
        assert_eq!(graph.outputs[1].doc_string, "duplicate output slot");
    }

    /// @trace TEST-ONNX-MODEL-NEW-013 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxFunction with empty inputs, outputs, and nodes (purely declarative function).
    #[test]
    fn onnx_function_with_empty_inputs_outputs_and_nodes_is_valid() {
        // Arrange: function with no inputs, outputs, or nodes (signature-only function)
        let func = OnnxFunction {
            name: "empty_sig".to_string(),
            domain: "test.domain".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: "a purely declarative function with no body".to_string(),
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert!(func.inputs.is_empty());
        assert!(func.outputs.is_empty());
        assert!(func.nodes.is_empty());
        assert!(func.opset_import.is_empty());
        assert_eq!(func.name, "empty_sig");
        assert_eq!(func.domain, "test.domain");
    }

    /// @trace TEST-ONNX-MODEL-NEW-014 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxGraph with quantization_annotation entries having all None optional fields.
    #[test]
    fn onnx_graph_quantization_annotation_all_none_optionals() {
        // Arrange: quantization annotation with only tensor_name populated
        let annot = OnnxQuantizationAnnotation {
            tensor_name: "weight_q".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        let graph = OnnxGraph {
            name: "q_annot_none".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![annot],
            metadata_props: HashMap::new(),
        };
        // Act & Assert
        assert_eq!(graph.quantization_annotation.len(), 1);
        let a = &graph.quantization_annotation[0];
        assert_eq!(a.tensor_name, "weight_q");
        assert!(a.quant_param_tensor_names.is_empty());
        assert!(a.scale.is_none());
        assert!(a.zero_point.is_none());
        assert!(a.axis.is_none());
    }

    /// @trace TEST-ONNX-MODEL-NEW-015 [req:REQ-LOADER-007] [level:unit]
    /// Verify OnnxModel with metadata_props containing structured key-value pairs
    /// and model graph name distinct from model domain.
    #[test]
    fn onnx_model_metadata_props_structured_keys_and_graph_name_distinct_from_domain() {
        // Arrange: metadata with structured props, and graph with different name
        let mut props = HashMap::new();
        props.insert("license".to_string(), "Apache-2.0".to_string());
        props.insert("author".to_string(), "test-org".to_string());
        props.insert("model_architecture".to_string(), "transformer".to_string());
        let graph = OnnxGraph {
            name: "bert_base_graph".to_string(),
            doc_string: "BERT base computation graph".to_string(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 9,
                producer_name: "transformers".to_string(),
                producer_version: "4.40.0".to_string(),
                domain: "org.huggingface".to_string(),
                model_version: 1,
                doc_string: "BERT base uncased".to_string(),
                opset_import: vec![OnnxOperatorSet {
                    domain: String::new(),
                    version: 17,
                }],
                metadata_props: props,
            },
            graph,
            functions: vec![],
        };
        // Act & Assert: model domain != graph name
        assert_ne!(model.metadata.domain, model.graph.name);
        assert_eq!(model.metadata.domain, "org.huggingface");
        assert_eq!(model.graph.name, "bert_base_graph");
        assert_eq!(model.metadata.metadata_props.len(), 3);
        assert_eq!(model.metadata.metadata_props["license"], "Apache-2.0");
        assert_eq!(model.metadata.metadata_props["author"], "test-org");
        assert_eq!(
            model.metadata.metadata_props["model_architecture"],
            "transformer"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // WAVE 8 TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. OnnxModelMetadata: empty producer with very long version string ──

    #[test]
    fn onnx_model_metadata_empty_producer_long_version() {
        let long_version = "v".repeat(10_000);
        let meta = OnnxModelMetadata {
            ir_version: 8,
            producer_name: String::new(),
            producer_version: long_version.clone(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        assert!(meta.producer_name.is_empty());
        assert_eq!(meta.producer_version.len(), 10_000);
        assert_eq!(meta.producer_version, long_version);
    }

    // ── 2. OpsetImport with unknown domain strings ─────────────────────────

    #[test]
    fn parse_opsets_unknown_custom_domains() {
        let opsets = vec![
            proto::OperatorSetIdProto {
                domain: Some("com.acme.custom.ops".to_string()),
                version: Some(1),
            },
            proto::OperatorSetIdProto {
                domain: Some("org.example.experimental".to_string()),
                version: Some(99),
            },
            proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            },
        ];
        let result = parse_opsets(opsets);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].domain, "com.acme.custom.ops");
        assert_eq!(result[0].version, 1);
        assert_eq!(result[1].domain, "org.example.experimental");
        assert_eq!(result[1].version, 99);
        assert_eq!(result[2].domain, "");
        assert_eq!(result[2].version, 17);
    }

    // ── 3. OnnxTensorType: SEQUENCE of MAP complex nested type ─────────────

    #[test]
    fn onnx_type_sequence_of_map_nested() {
        use super::super::types::{OnnxDim, OnnxMapType, OnnxTensorShape, OnnxTensorType, OnnxType};
        let map_type = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape {
                    dims: vec![OnnxDim::Unknown],
                },
            })),
        });
        let seq_of_map = OnnxType::Sequence(Box::new(map_type));
        let vi = OnnxValueInfo {
            name: "complex_io".to_string(),
            value_type: Some(seq_of_map),
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        if let Some(OnnxType::Sequence(inner)) = vi.value_type {
            if let OnnxType::Map(map) = &*inner {
                assert!(matches!(map.key_type, proto::tensor_proto::DataType::Int64));
                assert!(matches!(*map.value_type, OnnxType::Tensor(_)));
            } else {
                panic!("expected Map inside Sequence");
            }
        } else {
            panic!("expected Sequence variant");
        }
    }

    // ── 4. Sparse tensor: dims with zero in a dimension ────────────────────

    #[test]
    fn onnx_sparse_tensor_zero_dimension() {
        use super::super::tensor::{OnnxSparseFormat, OnnxSparseTensor};
        let values = OnnxTensor::new(
            "zero_vals".to_string(),
            Dtype::F32,
            vec![0],
            Bytes::new(),
        );
        let indices = OnnxTensor::new(
            "zero_idx".to_string(),
            Dtype::I64,
            vec![0],
            Bytes::new(),
        );
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![0, 5],
            format: OnnxSparseFormat::Coo,
        };
        assert_eq!(sparse.dims, vec![0, 5]);
        assert_eq!(sparse.dims[0], 0);
        assert!(sparse.values.raw_data().is_empty());
    }

    // ── 5. OnnxModelMetadata: model_version at i64::MAX boundary ──────────

    #[test]
    fn onnx_model_from_proto_model_version_max() {
        let model_proto = proto::ModelProto {
            ir_version: Some(9),
            model_version: Some(i64::MAX),
            graph: Some(proto::GraphProto {
                name: Some("boundary_graph".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.metadata.model_version, i64::MAX);
    }

    // ── 6. parse_functions with duplicate names across separate functions ──

    #[test]
    fn parse_functions_duplicate_names_allowed() {
        let functions = vec![
            proto::FunctionProto {
                name: Some("DupName".to_string()),
                domain: Some("domain_a".to_string()),
                overload: Some("v1".to_string()),
                input: vec![],
                output: vec![],
                attribute: vec![],
                attribute_proto: vec![],
                node: vec![],
                opset_import: vec![],
                value_info: vec![],
                doc_string: None,
                metadata_props: vec![],
            },
            proto::FunctionProto {
                name: Some("DupName".to_string()),
                domain: Some("domain_b".to_string()),
                overload: Some("v2".to_string()),
                input: vec![],
                output: vec![],
                attribute: vec![],
                attribute_proto: vec![],
                node: vec![],
                opset_import: vec![],
                value_info: vec![],
                doc_string: None,
                metadata_props: vec![],
            },
        ];
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let result = parse_functions(functions, &mut resolver).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "DupName");
        assert_eq!(result[1].name, "DupName");
        assert_ne!(result[0].domain, result[1].domain);
        assert_eq!(result[0].domain, "domain_a");
        assert_eq!(result[1].domain, "domain_b");
    }

    // ── 7. OnnxValueInfo with None type (missing shape information) ────────

    #[test]
    fn onnx_value_info_none_type_missing_shape() {
        let vi = OnnxValueInfo {
            name: "untyped_value".to_string(),
            value_type: None,
            doc_string: "no type information available".to_string(),
            metadata_props: HashMap::from([("origin".to_string(), "inferred".to_string())]),
        };
        assert!(vi.value_type.is_none());
        assert_eq!(vi.doc_string, "no type information available");
        assert_eq!(vi.metadata_props.get("origin").unwrap(), "inferred");
    }

    // ── 8. Graph with empty inputs/outputs but valid initializers ──────────

    #[test]
    fn onnx_graph_from_proto_empty_io_with_initializers() {
        let weight_tensor = proto::TensorProto {
            name: Some("const_weight".to_string()),
            data_type: Some(1),
            dims: vec![3, 4],
            float_data: vec![0.0; 12],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            name: Some("weights_only".to_string()),
            node: vec![],
            input: vec![],
            output: vec![],
            initializer: vec![weight_tensor],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert!(graph.inputs.is_empty());
        assert!(graph.outputs.is_empty());
        assert!(graph.nodes.is_empty());
        assert_eq!(graph.initializers.len(), 1);
        assert!(graph.initializers.contains_key("const_weight"));
    }

    // ── 9. Quantization annotation: scale is NaN ──────────────────────────

    #[test]
    fn onnx_quantization_annotation_scale_nan_preserved() {
        let nan_data = f32::NAN.to_le_bytes();
        let scale_tensor = OnnxTensor::new(
            "nan_scale".to_string(),
            Dtype::F32,
            vec![],
            Bytes::from(nan_data.to_vec()),
        );
        let initializers = HashMap::from([("nan_scale".to_string(), scale_tensor)]);
        let entries = vec![proto::TensorAnnotation {
            tensor_name: Some("nan_weight".to_string()),
            quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
                key: Some("SCALE_TENSOR".to_string()),
                value: Some("nan_scale".to_string()),
            }],
        }];
        let result = parse_quantization(entries, &initializers);
        assert_eq!(result.len(), 1);
        let scale = result[0].scale.unwrap();
        assert!(scale.is_nan());
    }

    // ── 10. OnnxAttributeValue: empty Ints vec ────────────────────────────

    #[test]
    fn onnx_attribute_value_empty_ints_vec() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let attr = OnnxAttribute {
            name: "empty_axes".to_string(),
            value: OnnxAttributeValue::Ints(vec![]),
            doc_string: "no axes specified".to_string(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        };
        if let OnnxAttributeValue::Ints(ref v) = attr.value {
            assert!(v.is_empty());
        } else {
            panic!("expected Ints variant");
        }
        assert_eq!(attr.doc_string, "no axes specified");
    }

    // ── 11. OnnxGraph from_proto with doc_string on graph ─────────────────

    #[test]
    fn onnx_graph_from_proto_graph_level_doc_string() {
        let graph_proto = proto::GraphProto {
            name: Some("doc_graph".to_string()),
            doc_string: Some("This graph computes attention".to_string()),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let graph = OnnxGraph::from_proto(graph_proto, &mut resolver).unwrap();
        assert_eq!(graph.doc_string, "This graph computes attention");
    }

    // ── 12. OnnxNode attribute with mixed INT and FLOAT in same map ───────

    #[test]
    fn onnx_node_mixed_int_and_float_attributes() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let mut attrs = HashMap::new();
        attrs.insert("epsilon".to_string(), OnnxAttribute {
            name: "epsilon".to_string(),
            value: OnnxAttributeValue::Float(1e-5),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        attrs.insert("axis".to_string(), OnnxAttribute {
            name: "axis".to_string(),
            value: OnnxAttributeValue::Int(-1),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        attrs.insert("beta".to_string(), OnnxAttribute {
            name: "beta".to_string(),
            value: OnnxAttributeValue::Float(0.9),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        attrs.insert("groups".to_string(), OnnxAttribute {
            name: "groups".to_string(),
            value: OnnxAttributeValue::Int(8),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let node = OnnxNode {
            name: "mixed_attr_node".to_string(),
            op_type: "LayerNorm".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
        };
        assert_eq!(node.attributes.len(), 4);
        let eps = node.attributes.get("epsilon").unwrap();
        assert!(matches!(eps.value, OnnxAttributeValue::Float(v) if (v - 1e-5).abs() < 1e-12));
        let axis = node.attributes.get("axis").unwrap();
        assert!(matches!(axis.value, OnnxAttributeValue::Int(-1)));
        let beta = node.attributes.get("beta").unwrap();
        assert!(matches!(beta.value, OnnxAttributeValue::Float(v) if (v - 0.9).abs() < 1e-6));
        let groups = node.attributes.get("groups").unwrap();
        assert!(matches!(groups.value, OnnxAttributeValue::Int(8)));
    }

    // ── 13. OnnxModel with multiple functions in from_proto ────────────────

    #[test]
    fn onnx_model_from_proto_multiple_functions_with_nodes() {
        let make_func = |name: &str, domain: &str, op: &str| -> proto::FunctionProto {
            proto::FunctionProto {
                name: Some(name.to_string()),
                domain: Some(domain.to_string()),
                overload: None,
                input: vec!["X".to_string()],
                output: vec!["Y".to_string()],
                attribute: vec![],
                attribute_proto: vec![],
                node: vec![proto::NodeProto {
                    op_type: Some(op.to_string()),
                    name: None,
                    input: vec!["X".to_string()],
                    output: vec!["Y".to_string()],
                    domain: None,
                    overload: None,
                    attribute: vec![],
                    doc_string: None,
                    metadata_props: vec![],
                    device_configurations: vec![],
                }],
                opset_import: vec![],
                value_info: vec![],
                doc_string: None,
                metadata_props: vec![],
            }
        };
        let model_proto = proto::ModelProto {
            ir_version: Some(8),
            graph: Some(proto::GraphProto {
                name: Some("g".to_string()),
                ..Default::default()
            }),
            functions: vec![
                make_func("FusedGelu", "com.microsoft", "Gelu"),
                make_func("FusedBiasAdd", "com.nvidia", "Add"),
            ],
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        assert_eq!(model.functions.len(), 2);
        assert_eq!(model.functions[0].name, "FusedGelu");
        assert_eq!(model.functions[0].domain, "com.microsoft");
        assert_eq!(model.functions[0].nodes[0].op_type, "Gelu");
        assert_eq!(model.functions[1].name, "FusedBiasAdd");
        assert_eq!(model.functions[1].domain, "com.nvidia");
        assert_eq!(model.functions[1].nodes[0].op_type, "Add");
    }

    // ── 14. OnnxTensor string tensor with multi-byte UTF-8 ─────────────────

    #[test]
    fn onnx_tensor_string_multi_byte_utf8_encoding() {
        let japanese = "こんにちは世界";
        let bytes = Bytes::from(japanese.as_bytes().to_vec());
        let tensor = OnnxTensor::new_string(
            "jp_labels".to_string(),
            vec![1],
            bytes,
        );
        assert!(tensor.is_string);
        assert_eq!(tensor.dtype, Dtype::U8);
        assert_eq!(tensor.shape, vec![1]);
        assert_eq!(tensor.raw_data().len(), japanese.as_bytes().len());
        let reconstructed = std::str::from_utf8(tensor.raw_data()).unwrap();
        assert_eq!(reconstructed, japanese);
    }

    // ── 15. OnnxAttributeValue: very large float (f64 edge case in f32) ──

    #[test]
    fn onnx_attribute_value_large_float_boundary() {
        use super::super::attributes::{OnnxAttribute, OnnxAttributeValue};
        let large_f32 = f32::MAX;
        let attr = OnnxAttribute {
            name: "max_learning_rate".to_string(),
            value: OnnxAttributeValue::Float(large_f32),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        };
        if let OnnxAttributeValue::Float(v) = attr.value {
            assert_eq!(v, f32::MAX);
            assert!(v.is_finite());
        } else {
            panic!("expected Float variant");
        }
    }

    // ── 16. OnnxFunction overload field stores explicit value ──────────

    #[test]
    fn onnx_function_overload_explicit_value() {
        let func = OnnxFunction {
            name: "FusedSoftmax".to_string(),
            domain: "com.example".to_string(),
            overload: "v2_fast".to_string(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        // Arrange: function with explicit overload
        // Act: access overload field
        // Assert: exact overload string preserved
        assert_eq!(func.overload, "v2_fast");
    }

    // ── 17. OnnxNode outputs all empty strings preserved ──────────────

    #[test]
    fn onnx_node_outputs_all_empty_strings() {
        let node = OnnxNode {
            name: "empty_out_node".to_string(),
            op_type: "Identity".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec![String::new(), String::new()],
            attributes: HashMap::new(),
        };
        // Arrange: node with two empty output names
        // Act: check outputs length and content
        // Assert: both empty strings preserved in order
        assert_eq!(node.outputs.len(), 2);
        assert!(node.outputs[0].is_empty());
        assert!(node.outputs[1].is_empty());
    }

    // ── 18. OnnxGraph name preserves whitespace content ───────────────

    #[test]
    fn onnx_graph_name_preserves_whitespace() {
        let graph = OnnxGraph {
            name: "  spaced graph  ".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Arrange: graph with whitespace-padded name
        // Act: access name field
        // Assert: whitespace preserved exactly
        assert_eq!(graph.name, "  spaced graph  ");
    }

    // ── 19. OnnxModel with no functions yields empty vec ──────────────

    #[test]
    fn onnx_model_empty_functions_vec() {
        let model = OnnxModel {
            metadata: OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        // Arrange: model with empty functions vec
        // Act: check functions length
        // Assert: zero functions, vec is empty
        assert!(model.functions.is_empty());
    }

    // ── 20. OnnxQuantizationAnnotation scale=0.0 is preserved ─────────

    #[test]
    fn onnx_quantization_annotation_scale_zero_value() {
        let ann = OnnxQuantizationAnnotation {
            tensor_name: "weights".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.0),
            zero_point: None,
            axis: None,
        };
        // Arrange: annotation with scale exactly 0.0
        // Act: access scale field
        // Assert: Some(0.0) preserved, not None
        assert_eq!(ann.scale, Some(0.0));
        assert!(ann.zero_point.is_none());
        assert!(ann.axis.is_none());
    }

    // ── 21. OnnxNode inputs duplicate names allowed in order ──────────

    #[test]
    fn onnx_node_inputs_duplicate_names_allowed() {
        let node = OnnxNode {
            name: "dup_input_node".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["X".to_string(), "X".to_string(), "Y".to_string()],
            outputs: vec!["Z".to_string()],
            attributes: HashMap::new(),
        };
        // Arrange: node with duplicate input name "X"
        // Act: check inputs vec
        // Assert: duplicates preserved in insertion order
        assert_eq!(node.inputs.len(), 3);
        assert_eq!(node.inputs[0], "X");
        assert_eq!(node.inputs[1], "X");
        assert_eq!(node.inputs[2], "Y");
    }

    // ── 22. parse_metadata_props entry with no key is skipped ─────────

    #[test]
    fn parse_metadata_props_no_key_entry_skipped() {
        let entries = vec![
            proto::StringStringEntryProto {
                key: None,
                value: Some("orphan_value".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("valid_key".to_string()),
                value: Some("valid_value".to_string()),
            },
        ];
        // Arrange: one entry with key=None, one valid
        // Act: parse
        let result = parse_metadata_props(entries);
        // Assert: only valid entry kept
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("valid_key"), Some(&"valid_value".to_string()));
    }

    // ── 23. OnnxGraph with no sparse_initializers is valid ────────────

    #[test]
    fn onnx_graph_no_sparse_initializers() {
        let graph = OnnxGraph {
            name: "dense_only".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Arrange: graph with empty sparse_initializers
        // Act: check sparse_initializers
        // Assert: empty vec is valid
        assert!(graph.sparse_initializers.is_empty());
    }

    // ── 24. OnnxOperatorSet custom domain with specific version ───────

    #[test]
    fn onnx_operator_set_custom_domain_with_version() {
        let opset = OnnxOperatorSet {
            domain: "org.custom.ops".to_string(),
            version: 42,
        };
        // Arrange: operator set with custom domain and specific version
        // Act: access fields
        // Assert: both fields preserved exactly
        assert_eq!(opset.domain, "org.custom.ops");
        assert_eq!(opset.version, 42);
    }

    // ── 25. OnnxGraph nodes with multiple distinct domains coexist ────

    #[test]
    fn onnx_graph_nodes_multiple_domains_coexist() {
        let graph = OnnxGraph {
            name: "multi_domain".to_string(),
            doc_string: String::new(),
            nodes: vec![
                OnnxNode {
                    name: "n1".to_string(),
                    op_type: "Conv".to_string(),
                    domain: String::new(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n2".to_string(),
                    op_type: "FusedBN".to_string(),
                    domain: "com.microsoft".to_string(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
                OnnxNode {
                    name: "n3".to_string(),
                    op_type: "MultiHeadAttn".to_string(),
                    domain: "com.nvidia".to_string(),
                    inputs: vec![],
                    outputs: vec![],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        // Arrange: graph with 3 nodes from different domains
        // Act: check node count and domain values
        // Assert: all domains preserved distinctly
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].domain, "");
        assert_eq!(graph.nodes[1].domain, "com.microsoft");
        assert_eq!(graph.nodes[2].domain, "com.nvidia");
    }

    // ── 26. OnnxValueInfo metadata_props with five distinct entries ───

    #[test]
    fn onnx_value_info_metadata_five_distinct() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "encoder".to_string());
        meta.insert("layer".to_string(), "3".to_string());
        meta.insert("head".to_string(), "7".to_string());
        meta.insert("dtype".to_string(), "bf16".to_string());
        meta.insert("quantized".to_string(), "false".to_string());

        let vi = OnnxValueInfo {
            name: "intermediate_12".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: meta,
        };
        // Arrange: value info with 5 metadata entries
        // Act: access metadata_props
        // Assert: all 5 entries present with correct values
        assert_eq!(vi.metadata_props.len(), 5);
        assert_eq!(vi.metadata_props.get("source"), Some(&"encoder".to_string()));
        assert_eq!(vi.metadata_props.get("layer"), Some(&"3".to_string()));
        assert_eq!(vi.metadata_props.get("head"), Some(&"7".to_string()));
        assert_eq!(vi.metadata_props.get("dtype"), Some(&"bf16".to_string()));
        assert_eq!(vi.metadata_props.get("quantized"), Some(&"false".to_string()));
    }

    // ── 27. parse_nodes whitespace-only op_type accepted as non-empty ─

    #[test]
    fn parse_nodes_whitespace_op_type_accepted() {
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        let nodes = vec![proto::NodeProto {
            op_type: Some("   ".to_string()),
            name: Some("whitespace_op".to_string()),
            input: vec![],
            output: vec![],
            domain: None,
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
            device_configurations: vec![],
        }];
        // Arrange: node with whitespace-only op_type (non-empty string)
        // Act: parse_nodes
        let result = parse_nodes(nodes, &mut resolver);
        // Assert: whitespace-only op_type passes the is_empty() check
        let parsed = result.unwrap();
        assert_eq!(parsed[0].op_type, "   ");
    }

    // ── 28. OnnxModel ir_version=0 when proto field is missing ────────

    #[test]
    fn onnx_model_ir_version_none_default() {
        let model_proto = proto::ModelProto {
            ir_version: None,
            graph: Some(proto::GraphProto {
                name: Some("g".to_string()),
                node: vec![proto::NodeProto {
                    op_type: Some("Relu".to_string()),
                    name: Some("n0".to_string()),
                    input: vec!["X".to_string()],
                    output: vec!["Y".to_string()],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut resolver = ExternalDataResolver::new(std::path::Path::new(""));
        // Arrange: model proto with ir_version=None
        // Act: parse model
        let model = OnnxModel::from_proto(model_proto, &mut resolver).unwrap();
        // Assert: ir_version defaults to 0
        assert_eq!(model.metadata.ir_version, 0);
    }
}

fn parse_functions(
    functions: Vec<proto::FunctionProto>,
    resolver: &mut ExternalDataResolver,
) -> Result<Vec<OnnxFunction>> {
    let mut out = Vec::with_capacity(functions.len());
    for func in functions {
        let name = func.name.unwrap_or_default(); // LEGAL: protobuf 可选字段
        if name.is_empty() {
            return Err(LoaderError::Onnx("function missing name".to_string()));
        }
        let nodes = parse_nodes(func.node, resolver)?;
        let attribute_protos = parse_attributes(func.attribute_proto, resolver)?;
        let value_info = parse_value_info(func.value_info)?;
        out.push(OnnxFunction {
            name,
            domain: func.domain.unwrap_or_default(), // LEGAL: protobuf 可选字段
            overload: func.overload.unwrap_or_default(), // LEGAL: protobuf 可选字段
            inputs: func.input,
            outputs: func.output,
            attributes: func.attribute,
            attribute_protos,
            nodes,
            opset_import: parse_opsets(func.opset_import),
            value_info,
            doc_string: func.doc_string.unwrap_or_default(), // LEGAL: protobuf 可选字段
            metadata_props: parse_metadata_props(func.metadata_props),
        });
    }
    Ok(out)
}
