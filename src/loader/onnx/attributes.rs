use std::collections::HashMap;

use super::external::ExternalDataResolver;
use super::model::OnnxGraph;
use super::tensor::{OnnxSparseTensor, OnnxTensor};
use super::types::OnnxType;
use super::{proto, LoaderError, Result};

#[derive(Debug, Clone)]
pub struct OnnxAttribute {
    pub name: String,
    pub value: OnnxAttributeValue,
    pub doc_string: String,
    pub ref_attr_name: Option<String>,
    pub attr_type: Option<proto::attribute_proto::AttributeType>,
}

/// ONNX attribute value with recursive subgraph support.
///
/// Graph/Graphs variants now contain fully parsed `OnnxGraph` structures,
/// enabling control flow operators (If, Loop, Scan) to work correctly.
#[derive(Debug, Clone)]
pub enum OnnxAttributeValue {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(OnnxTensor),
    SparseTensor(OnnxSparseTensor),
    Type(OnnxType),
    /// Parsed subgraph (used by If/Loop/Scan operators)
    Graph(Box<OnnxGraph>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensors(Vec<OnnxTensor>),
    SparseTensors(Vec<OnnxSparseTensor>),
    Types(Vec<OnnxType>),
    /// Multiple parsed subgraphs
    Graphs(Vec<OnnxGraph>),
    Ref(String),
}

pub(super) fn parse_attributes(
    attrs: Vec<proto::AttributeProto>,
    resolver: &mut ExternalDataResolver,
) -> Result<HashMap<String, OnnxAttribute>> {
    let mut out = HashMap::new();
    for attr in attrs {
        let parsed = parse_attribute(attr, resolver)?;
        if out.contains_key(&parsed.name) {
            return Err(LoaderError::Onnx(format!(
                "duplicate attribute {}",
                parsed.name
            )));
        }
        out.insert(parsed.name.clone(), parsed);
    }
    Ok(out)
}

fn parse_attribute(
    attr: proto::AttributeProto,
    resolver: &mut ExternalDataResolver,
) -> Result<OnnxAttribute> {
    let name = attr
        .name
        .clone()
        .ok_or_else(|| LoaderError::Onnx("attribute missing name".to_string()))?;
    let doc_string = attr.doc_string.clone().unwrap_or_default();
    let ref_attr_name = match attr.ref_attr_name.clone() {
        Some(value) if !value.is_empty() => Some(value),
        _ => None,
    };
    if let Some(reference) = ref_attr_name.clone() {
        return Ok(OnnxAttribute {
            name,
            value: OnnxAttributeValue::Ref(reference),
            doc_string,
            ref_attr_name,
            attr_type: parse_attr_type(attr.r#type),
        });
    }
    let value = parse_attribute_value(&attr, &name, resolver)?;
    Ok(OnnxAttribute {
        name,
        value,
        doc_string,
        ref_attr_name,
        attr_type: parse_attr_type(attr.r#type),
    })
}

fn parse_attribute_value(
    attr: &proto::AttributeProto,
    name: &str,
    resolver: &mut ExternalDataResolver,
) -> Result<OnnxAttributeValue> {
    if let Some(attr_type) = parse_attr_type(attr.r#type) {
        return parse_by_type(attr, attr_type, name, resolver);
    }
    parse_by_presence(attr, name, resolver)
}

fn parse_by_type(
    attr: &proto::AttributeProto,
    attr_type: proto::attribute_proto::AttributeType,
    name: &str,
    resolver: &mut ExternalDataResolver,
) -> Result<OnnxAttributeValue> {
    use proto::attribute_proto::AttributeType as AttrType;
    match attr_type {
        AttrType::Float => attr
            .f
            .map(OnnxAttributeValue::Float)
            .ok_or_else(|| missing_attr_value(name, "float")),
        AttrType::Int => attr
            .i
            .map(OnnxAttributeValue::Int)
            .ok_or_else(|| missing_attr_value(name, "int")),
        AttrType::String => {
            let value = attr
                .s
                .as_ref()
                .ok_or_else(|| missing_attr_value(name, "string"))?;
            Ok(OnnxAttributeValue::String(parse_utf8(value, name)?))
        }
        AttrType::Tensor => {
            let tensor = attr
                .t
                .clone()
                .ok_or_else(|| missing_attr_value(name, "tensor"))?;
            Ok(OnnxAttributeValue::Tensor(OnnxTensor::from_attribute(
                tensor, resolver, name,
            )?))
        }
        AttrType::SparseTensor => {
            let tensor = attr
                .sparse_tensor
                .clone()
                .ok_or_else(|| missing_attr_value(name, "sparse_tensor"))?;
            Ok(OnnxAttributeValue::SparseTensor(
                OnnxSparseTensor::from_proto(tensor, resolver)?,
            ))
        }
        AttrType::TypeProto => {
            let tp = attr
                .tp
                .clone()
                .ok_or_else(|| missing_attr_value(name, "type_proto"))?;
            Ok(OnnxAttributeValue::Type(OnnxType::from_proto(tp)?))
        }
        AttrType::Graph => {
            let graph_proto = attr
                .g
                .clone()
                .ok_or_else(|| missing_attr_value(name, "graph"))?;
            let parsed = OnnxGraph::from_proto(graph_proto, resolver)?;
            Ok(OnnxAttributeValue::Graph(Box::new(parsed)))
        }
        AttrType::Floats => Ok(OnnxAttributeValue::Floats(attr.floats.clone())),
        AttrType::Ints => Ok(OnnxAttributeValue::Ints(attr.ints.clone())),
        AttrType::Strings => Ok(OnnxAttributeValue::Strings(parse_strings(
            &attr.strings,
            name,
        )?)),
        AttrType::Tensors => {
            let mut tensors = Vec::with_capacity(attr.tensors.len());
            for (idx, tensor) in attr.tensors.iter().cloned().enumerate() {
                let fallback = format!("{name}_tensor_{idx}");
                tensors.push(OnnxTensor::from_attribute(tensor, resolver, &fallback)?);
            }
            Ok(OnnxAttributeValue::Tensors(tensors))
        }
        AttrType::SparseTensors => {
            let mut tensors = Vec::with_capacity(attr.sparse_tensors.len());
            for tensor in attr.sparse_tensors.iter().cloned() {
                tensors.push(OnnxSparseTensor::from_proto(tensor, resolver)?);
            }
            Ok(OnnxAttributeValue::SparseTensors(tensors))
        }
        AttrType::Graphs => {
            let mut graphs = Vec::with_capacity(attr.graphs.len());
            for graph_proto in attr.graphs.iter().cloned() {
                graphs.push(OnnxGraph::from_proto(graph_proto, resolver)?);
            }
            Ok(OnnxAttributeValue::Graphs(graphs))
        }
        AttrType::TypeProtos => {
            let mut types = Vec::with_capacity(attr.type_protos.len());
            for tp in attr.type_protos.iter().cloned() {
                types.push(OnnxType::from_proto(tp)?);
            }
            Ok(OnnxAttributeValue::Types(types))
        }
        AttrType::Undefined => Err(LoaderError::Onnx(format!(
            "attribute {name} has undefined type"
        ))),
    }
}

fn parse_by_presence(
    attr: &proto::AttributeProto,
    name: &str,
    resolver: &mut ExternalDataResolver,
) -> Result<OnnxAttributeValue> {
    if let Some(value) = attr.f {
        return Ok(OnnxAttributeValue::Float(value));
    }
    if let Some(value) = attr.i {
        return Ok(OnnxAttributeValue::Int(value));
    }
    if let Some(value) = attr.s.as_ref() {
        return Ok(OnnxAttributeValue::String(parse_utf8(value, name)?));
    }
    if let Some(value) = attr.t.clone() {
        return Ok(OnnxAttributeValue::Tensor(OnnxTensor::from_attribute(
            value, resolver, name,
        )?));
    }
    if let Some(graph_proto) = attr.g.clone() {
        let parsed = OnnxGraph::from_proto(graph_proto, resolver)?;
        return Ok(OnnxAttributeValue::Graph(Box::new(parsed)));
    }
    if !attr.floats.is_empty() {
        return Ok(OnnxAttributeValue::Floats(attr.floats.clone()));
    }
    if !attr.ints.is_empty() {
        return Ok(OnnxAttributeValue::Ints(attr.ints.clone()));
    }
    if !attr.strings.is_empty() {
        return Ok(OnnxAttributeValue::Strings(parse_strings(
            &attr.strings,
            name,
        )?));
    }
    if !attr.tensors.is_empty() {
        let mut tensors = Vec::with_capacity(attr.tensors.len());
        for (idx, tensor) in attr.tensors.iter().cloned().enumerate() {
            let fallback = format!("{name}_tensor_{idx}");
            tensors.push(OnnxTensor::from_attribute(tensor, resolver, &fallback)?);
        }
        return Ok(OnnxAttributeValue::Tensors(tensors));
    }
    if !attr.graphs.is_empty() {
        let mut graphs = Vec::with_capacity(attr.graphs.len());
        for graph_proto in attr.graphs.iter().cloned() {
            graphs.push(OnnxGraph::from_proto(graph_proto, resolver)?);
        }
        return Ok(OnnxAttributeValue::Graphs(graphs));
    }
    if let Some(value) = attr.tp.clone() {
        return Ok(OnnxAttributeValue::Type(OnnxType::from_proto(value)?));
    }
    if let Some(value) = attr.sparse_tensor.clone() {
        return Ok(OnnxAttributeValue::SparseTensor(
            OnnxSparseTensor::from_proto(value, resolver)?,
        ));
    }
    if !attr.sparse_tensors.is_empty() {
        let mut tensors = Vec::with_capacity(attr.sparse_tensors.len());
        for tensor in attr.sparse_tensors.iter().cloned() {
            tensors.push(OnnxSparseTensor::from_proto(tensor, resolver)?);
        }
        return Ok(OnnxAttributeValue::SparseTensors(tensors));
    }
    if !attr.type_protos.is_empty() {
        let mut types = Vec::with_capacity(attr.type_protos.len());
        for tp in attr.type_protos.iter().cloned() {
            types.push(OnnxType::from_proto(tp)?);
        }
        return Ok(OnnxAttributeValue::Types(types));
    }
    Err(LoaderError::Onnx(format!("attribute {name} missing value")))
}

fn parse_utf8(value: &[u8], name: &str) -> Result<String> {
    String::from_utf8(value.to_vec())
        .map_err(|_| LoaderError::Onnx(format!("attribute {name} has invalid utf8 string")))
}

fn parse_strings(values: &[Vec<u8>], name: &str) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(parse_utf8(value, name)?);
    }
    Ok(out)
}

fn missing_attr_value(name: &str, kind: &str) -> LoaderError {
    LoaderError::Onnx(format!("attribute {name} missing {kind} value"))
}

fn parse_attr_type(value: Option<i32>) -> Option<proto::attribute_proto::AttributeType> {
    value.and_then(|value| proto::attribute_proto::AttributeType::try_from(value).ok())
}
