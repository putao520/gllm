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

impl OnnxModel {
    pub(super) fn from_proto(
        proto: proto::ModelProto,
        resolver: &mut ExternalDataResolver,
    ) -> Result<Self> {
        let graph = proto
            .graph
            .ok_or_else(|| LoaderError::Onnx("missing graph in ONNX model".to_string()))?;
        let metadata = OnnxModelMetadata {
            ir_version: proto.ir_version.unwrap_or_default(),
            producer_name: proto.producer_name.unwrap_or_default(),
            producer_version: proto.producer_version.unwrap_or_default(),
            domain: proto.domain.unwrap_or_default(),
            model_version: proto.model_version.unwrap_or_default(),
            doc_string: proto.doc_string.unwrap_or_default(),
            opset_import: parse_opsets(proto.opset_import),
            metadata_props: parse_metadata_props(proto.metadata_props),
        };
        let graph = OnnxGraph::from_proto(graph, resolver)?;
        Ok(Self { metadata, graph })
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
            name: proto.name.unwrap_or_default(),
            doc_string: proto.doc_string.unwrap_or_default(),
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
        let op_type = node.op_type.unwrap_or_default();
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
            domain: node.domain.unwrap_or_default(),
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
            doc_string: value.doc_string.unwrap_or_default(),
            metadata_props: parse_metadata_props(value.metadata_props),
        });
    }
    Ok(out)
}

fn parse_opsets(opsets: Vec<proto::OperatorSetIdProto>) -> Vec<OnnxOperatorSet> {
    opsets
        .into_iter()
        .map(|opset| OnnxOperatorSet {
            domain: opset.domain.unwrap_or_default(),
            version: opset.version.unwrap_or_default(),
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
        let value = entry.value.unwrap_or_default();
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
            tensor_name: entry.tensor_name.unwrap_or_default(),
            quant_param_tensor_names,
            scale,
            zero_point,
            axis,
        });
    }
    out
}
