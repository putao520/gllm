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
    let doc_string = attr.doc_string.clone().unwrap_or_default(); // LEGAL: protobuf 可选字段
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::onnx::{OnnxDim, OnnxMapType, OnnxTensorShape, OnnxTensorType};

    // ── OnnxAttributeValue variants ───────────────────────────────────

    #[test]
    fn attribute_value_float() {
        let v = OnnxAttributeValue::Float(3.14);
        assert!(matches!(v, OnnxAttributeValue::Float(f) if (f - 3.14).abs() < 0.01));
    }

    #[test]
    fn attribute_value_int() {
        let v = OnnxAttributeValue::Int(-42);
        assert!(matches!(v, OnnxAttributeValue::Int(-42)));
    }

    #[test]
    fn attribute_value_string() {
        let v = OnnxAttributeValue::String("kernel_shape".to_string());
        assert!(matches!(v, OnnxAttributeValue::String(s) if s == "kernel_shape"));
    }

    #[test]
    fn attribute_value_floats() {
        let v = OnnxAttributeValue::Floats(vec![1.0, 2.0, 3.0]);
        assert!(matches!(v, OnnxAttributeValue::Floats(v) if v == vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn attribute_value_ints() {
        let v = OnnxAttributeValue::Ints(vec![3, 1, 4]);
        assert!(matches!(v, OnnxAttributeValue::Ints(v) if v == vec![3, 1, 4]));
    }

    #[test]
    fn attribute_value_strings() {
        let v = OnnxAttributeValue::Strings(vec!["a".to_string(), "b".to_string()]);
        assert!(matches!(v, OnnxAttributeValue::Strings(s) if s.len() == 2));
    }

    #[test]
    fn attribute_value_ref() {
        let v = OnnxAttributeValue::Ref("other_attr".to_string());
        assert!(matches!(v, OnnxAttributeValue::Ref(r) if r == "other_attr"));
    }

    #[test]
    fn attribute_value_types() {
        let v = OnnxAttributeValue::Types(vec![]);
        assert!(matches!(v, OnnxAttributeValue::Types(_)));
    }

    // ── OnnxAttribute struct ──────────────────────────────────────────

    #[test]
    fn attribute_fields() {
        let attr = OnnxAttribute {
            name: "pads".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 1, 1, 1]),
            doc_string: "padding".to_string(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        };
        assert_eq!(attr.name, "pads");
        assert!(attr.ref_attr_name.is_none());
    }

    #[test]
    fn attribute_clone() {
        let attr = OnnxAttribute {
            name: "epsilon".to_string(),
            value: OnnxAttributeValue::Float(1e-5),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        };
        let cloned = attr.clone();
        assert_eq!(cloned.name, "epsilon");
    }

    // ── parse_utf8 ────────────────────────────────────────────────────

    #[test]
    fn parse_utf8_valid() {
        let result = parse_utf8(b"hello", "test").unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn parse_utf8_invalid() {
        let invalid: &[u8] = &[0xFF, 0xFE];
        assert!(parse_utf8(invalid, "bad_attr").is_err());
    }

    // ── parse_strings ─────────────────────────────────────────────────

    #[test]
    fn parse_strings_valid() {
        let input: Vec<Vec<u8>> = vec![b"abc".to_vec(), b"def".to_vec()];
        let result = parse_strings(&input, "test").unwrap();
        assert_eq!(result, vec!["abc", "def"]);
    }

    #[test]
    fn parse_strings_empty() {
        let result: Vec<String> = parse_strings(&[], "test").unwrap();
        assert!(result.is_empty());
    }

    // ── parse_attr_type ───────────────────────────────────────────────

    #[test]
    fn parse_attr_type_valid() {
        let at = parse_attr_type(Some(1));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Float)));
    }

    #[test]
    fn parse_attr_type_none_input() {
        assert!(parse_attr_type(None).is_none());
    }

    #[test]
    fn parse_attr_type_invalid_value() {
        assert!(parse_attr_type(Some(9999)).is_none());
    }

    // ── missing_attr_value ────────────────────────────────────────────

    #[test]
    fn missing_attr_value_message() {
        let err = missing_attr_value("my_attr", "float");
        let msg = format!("{err}");
        assert!(msg.contains("my_attr"));
        assert!(msg.contains("float"));
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn attribute_value_zero_float() {
        let v = OnnxAttributeValue::Float(0.0);
        assert!(matches!(v, OnnxAttributeValue::Float(f) if f == 0.0));
    }

    #[test]
    fn attribute_value_zero_int() {
        let v = OnnxAttributeValue::Int(0);
        assert!(matches!(v, OnnxAttributeValue::Int(0)));
    }

    #[test]
    fn attribute_value_negative_int() {
        let v = OnnxAttributeValue::Int(i64::MIN);
        assert!(matches!(v, OnnxAttributeValue::Int(n) if n == i64::MIN));
    }

    #[test]
    fn attribute_value_empty_string() {
        let v = OnnxAttributeValue::String(String::new());
        assert!(matches!(v, OnnxAttributeValue::String(s) if s.is_empty()));
    }

    #[test]
    fn attribute_value_empty_floats() {
        let v = OnnxAttributeValue::Floats(vec![]);
        assert!(matches!(v, OnnxAttributeValue::Floats(v) if v.is_empty()));
    }

    #[test]
    fn attribute_value_empty_ints() {
        let v = OnnxAttributeValue::Ints(vec![]);
        assert!(matches!(v, OnnxAttributeValue::Ints(v) if v.is_empty()));
    }

    #[test]
    fn attribute_value_empty_strings() {
        let v = OnnxAttributeValue::Strings(vec![]);
        assert!(matches!(v, OnnxAttributeValue::Strings(v) if v.is_empty()));
    }

    #[test]
    fn attribute_value_empty_tensors() {
        let v = OnnxAttributeValue::Tensors(vec![]);
        assert!(matches!(v, OnnxAttributeValue::Tensors(v) if v.is_empty()));
    }

    #[test]
    fn attribute_value_empty_sparse_tensors() {
        let v = OnnxAttributeValue::SparseTensors(vec![]);
        assert!(matches!(v, OnnxAttributeValue::SparseTensors(v) if v.is_empty()));
    }

    #[test]
    fn attribute_value_empty_graphs() {
        let v = OnnxAttributeValue::Graphs(vec![]);
        assert!(matches!(v, OnnxAttributeValue::Graphs(v) if v.is_empty()));
    }

    #[test]
    fn attribute_clone_independent() {
        let attr = OnnxAttribute {
            name: "alpha".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 2, 3]),
            doc_string: "test doc".to_string(),
            ref_attr_name: Some("ref_target".to_string()),
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        };
        let mut cloned = attr.clone();
        // Mutate the clone, verify original is unaffected.
        if let OnnxAttributeValue::Ints(ref mut v) = cloned.value {
            v.push(4);
        }
        assert!(matches!(
            &attr.value,
            OnnxAttributeValue::Ints(v) if v.len() == 3
        ));
        assert!(matches!(
            &cloned.value,
            OnnxAttributeValue::Ints(v) if v.len() == 4
        ));
    }

    #[test]
    fn attribute_debug_format() {
        let attr = OnnxAttribute {
            name: "beta".to_string(),
            value: OnnxAttributeValue::Float(2.5),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Float),
        };
        let debug_str = format!("{attr:?}");
        assert!(debug_str.contains("beta"));
        assert!(debug_str.contains("Float"));
    }

    #[test]
    fn attribute_value_clone_value_semantics() {
        // Verify that cloning an enum variant yields independent data.
        let original = OnnxAttributeValue::Strings(vec!["x".to_string(), "y".to_string()]);
        let cloned = original.clone();
        assert!(matches!(
            cloned,
            OnnxAttributeValue::Strings(s) if s == vec!["x".to_string(), "y".to_string()]
        ));
    }

    #[test]
    fn attribute_with_ref_attr_name() {
        let attr = OnnxAttribute {
            name: "referred".to_string(),
            value: OnnxAttributeValue::Ref("parent_attr".to_string()),
            doc_string: String::new(),
            ref_attr_name: Some("parent_attr".to_string()),
            attr_type: None,
        };
        assert_eq!(attr.ref_attr_name.as_deref(), Some("parent_attr"));
        assert!(matches!(attr.value, OnnxAttributeValue::Ref(r) if r == "parent_attr"));
    }

    #[test]
    fn attribute_with_none_attr_type() {
        let attr = OnnxAttribute {
            name: "no_type".to_string(),
            value: OnnxAttributeValue::Int(7),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        };
        assert!(attr.attr_type.is_none());
    }

    #[test]
    fn parse_attr_type_int_type_code() {
        let at = parse_attr_type(Some(2));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Int)));
    }

    #[test]
    fn parse_attr_type_floats_type_code() {
        let at = parse_attr_type(Some(6));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Floats)));
    }

    #[test]
    fn parse_attr_type_graph_type_code() {
        let at = parse_attr_type(Some(5));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Graph)));
    }

    #[test]
    fn parse_attr_type_undefined_code() {
        let at = parse_attr_type(Some(0));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Undefined)));
    }

    #[test]
    fn parse_utf8_empty_bytes() {
        let result = parse_utf8(b"", "test").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn parse_utf8_cjk_string() {
        let input = "模型权重".as_bytes();
        let result = parse_utf8(input, "test").unwrap();
        assert_eq!(result, "模型权重");
    }

    #[test]
    fn parse_strings_single_element() {
        let input: Vec<Vec<u8>> = vec![b"only_one".to_vec()];
        let result = parse_strings(&input, "test").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "only_one");
    }

    #[test]
    fn parse_strings_with_invalid_member() {
        let input: Vec<Vec<u8>> = vec![b"valid".to_vec(), vec![0xFF, 0xFE]];
        let result = parse_strings(&input, "bad_attr");
        assert!(result.is_err());
    }

    #[test]
    fn missing_attr_value_formats_all_kinds() {
        for kind in &["float", "int", "string", "tensor", "graph"] {
            let err = missing_attr_value("x", kind);
            let msg = format!("{err}");
            assert!(msg.contains("x"), "expected attr name in error for kind={kind}");
            assert!(msg.contains(kind), "expected kind in error for kind={kind}");
        }
    }

    #[test]
    fn loader_error_onnx_display() {
        let err = LoaderError::Onnx("something broke".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("something broke"));
    }

    #[test]
    fn attribute_value_float_nan() {
        let v = OnnxAttributeValue::Float(f32::NAN);
        assert!(matches!(v, OnnxAttributeValue::Float(f) if f.is_nan()));
    }

    #[test]
    fn attribute_value_float_infinity() {
        let v = OnnxAttributeValue::Float(f32::INFINITY);
        assert!(matches!(v, OnnxAttributeValue::Float(f) if f.is_infinite() && f.is_sign_positive()));
    }

    #[test]
    fn attribute_value_int_max() {
        let v = OnnxAttributeValue::Int(i64::MAX);
        assert!(matches!(v, OnnxAttributeValue::Int(n) if n == i64::MAX));
    }

    #[test]
    fn attribute_doc_string_preserved() {
        let attr = OnnxAttribute {
            name: "strides".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 2]),
            doc_string: "Stride values for each spatial dimension.".to_string(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        };
        assert_eq!(attr.doc_string, "Stride values for each spatial dimension.");
    }

    #[test]
    fn attribute_value_debug_variants() {
        // Verify Debug trait produces meaningful output for multiple variants.
        let variants: Vec<OnnxAttributeValue> = vec![
            OnnxAttributeValue::Float(1.0),
            OnnxAttributeValue::Int(42),
            OnnxAttributeValue::String("hello".to_string()),
            OnnxAttributeValue::Floats(vec![1.0, 2.0]),
            OnnxAttributeValue::Ints(vec![10, 20]),
            OnnxAttributeValue::Strings(vec!["a".to_string()]),
            OnnxAttributeValue::Ref("other".to_string()),
            OnnxAttributeValue::Types(vec![]),
            OnnxAttributeValue::Tensors(vec![]),
            OnnxAttributeValue::SparseTensors(vec![]),
            OnnxAttributeValue::Graphs(vec![]),
        ];
        let debug = format!("{variants:?}");
        assert!(debug.contains("Float"));
        assert!(debug.contains("Int"));
        assert!(debug.contains("String"));
        assert!(debug.contains("Ref"));
        assert!(debug.contains("Types"));
    }

    // ── 45 new tests ─────────────────────────────────────────────────

    // --- OnnxAttributeValue: special floats and boundary values ---

    #[test]
    fn attribute_value_float_neg_infinity() {
        let v = OnnxAttributeValue::Float(f32::NEG_INFINITY);
        assert!(matches!(v, OnnxAttributeValue::Float(f) if f.is_infinite() && f.is_sign_negative()));
    }

    #[test]
    fn attribute_value_float_neg_zero() {
        let v = OnnxAttributeValue::Float(-0.0f32);
        assert!(matches!(v, OnnxAttributeValue::Float(f) if f == 0.0 && f.is_sign_negative()));
    }

    #[test]
    fn attribute_value_float_subnormal() {
        let v = OnnxAttributeValue::Float(f32::from_bits(1)); // smallest positive subnormal
        assert!(matches!(v, OnnxAttributeValue::Float(f) if f > 0.0 && f.is_subnormal()));
    }

    #[test]
    fn attribute_value_float_max() {
        let v = OnnxAttributeValue::Float(f32::MAX);
        assert!(matches!(v, OnnxAttributeValue::Float(f) if f == f32::MAX));
    }

    #[test]
    fn attribute_value_float_min_positive() {
        let v = OnnxAttributeValue::Float(f32::MIN_POSITIVE);
        assert!(matches!(v, OnnxAttributeValue::Float(f) if f == f32::MIN_POSITIVE));
    }

    // --- OnnxAttributeValue: Int edge cases ---

    #[test]
    fn attribute_value_int_one() {
        let v = OnnxAttributeValue::Int(1);
        assert!(matches!(v, OnnxAttributeValue::Int(1)));
    }

    #[test]
    fn attribute_value_int_negative_one() {
        let v = OnnxAttributeValue::Int(-1);
        assert!(matches!(v, OnnxAttributeValue::Int(-1)));
    }

    // --- OnnxAttributeValue: String edge cases ---

    #[test]
    fn attribute_value_string_multibyte() {
        let japanese = "こんにちは世界".to_string();
        let v = OnnxAttributeValue::String(japanese.clone());
        assert!(matches!(v, OnnxAttributeValue::String(s) if s == japanese));
    }

    #[test]
    fn attribute_value_string_long() {
        let long = "x".repeat(10000);
        let v = OnnxAttributeValue::String(long.clone());
        assert!(matches!(v, OnnxAttributeValue::String(s) if s.len() == 10000));
    }

    #[test]
    fn attribute_value_string_with_null_bytes() {
        // Rust strings can contain null bytes; verify it stores correctly.
        let with_null = "ab\0cd".to_string();
        let v = OnnxAttributeValue::String(with_null.clone());
        assert!(matches!(v, OnnxAttributeValue::String(s) if s == with_null));
    }

    // --- OnnxAttributeValue: Floats edge cases ---

    #[test]
    fn attribute_value_floats_single_element() {
        let v = OnnxAttributeValue::Floats(vec![42.5]);
        assert!(matches!(v, OnnxAttributeValue::Floats(v) if v == vec![42.5]));
    }

    #[test]
    fn attribute_value_floats_with_nan_and_inf() {
        let v = OnnxAttributeValue::Floats(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0]);
        assert!(matches!(v, OnnxAttributeValue::Floats(f) if f.len() == 4));
    }

    // --- OnnxAttributeValue: Ints edge cases ---

    #[test]
    fn attribute_value_ints_large_set() {
        let vals: Vec<i64> = (0..1000).collect();
        let v = OnnxAttributeValue::Ints(vals.clone());
        assert!(matches!(v, OnnxAttributeValue::Ints(v) if v.len() == 1000));
    }

    #[test]
    fn attribute_value_ints_with_negative() {
        let v = OnnxAttributeValue::Ints(vec![-100, 0, 100]);
        assert!(matches!(v, OnnxAttributeValue::Ints(v) if v[0] == -100 && v[2] == 100));
    }

    // --- OnnxAttributeValue: Strings edge cases ---

    #[test]
    fn attribute_value_strings_with_empty_member() {
        let v = OnnxAttributeValue::Strings(vec![String::new(), "nonempty".to_string(), String::new()]);
        assert!(matches!(v, OnnxAttributeValue::Strings(s) if s.len() == 3 && s[0].is_empty()));
    }

    // --- OnnxAttributeValue: Ref edge cases ---

    #[test]
    fn attribute_value_ref_empty_string() {
        let v = OnnxAttributeValue::Ref(String::new());
        assert!(matches!(v, OnnxAttributeValue::Ref(r) if r.is_empty()));
    }

    #[test]
    fn attribute_value_ref_unicode_name() {
        let v = OnnxAttributeValue::Ref("属性引用".to_string());
        assert!(matches!(v, OnnxAttributeValue::Ref(r) if r == "属性引用"));
    }

    // --- OnnxAttributeValue: clone independence for each vector variant ---

    #[test]
    fn attribute_value_clone_floats_independence() {
        let original = OnnxAttributeValue::Floats(vec![1.0, 2.0]);
        let cloned = original.clone();
        if let OnnxAttributeValue::Floats(v) = cloned {
            assert_eq!(v, vec![1.0, 2.0]);
        } else {
            panic!("expected Floats variant");
        }
    }

    #[test]
    fn attribute_value_clone_ints_independence() {
        let original = OnnxAttributeValue::Ints(vec![10, 20, 30]);
        let mut cloned = original.clone();
        if let OnnxAttributeValue::Ints(ref mut v) = cloned {
            v.push(40);
        }
        // Original must remain unmodified.
        assert!(matches!(&original, OnnxAttributeValue::Ints(v) if v.len() == 3));
    }

    #[test]
    fn attribute_value_clone_strings_independence() {
        let original = OnnxAttributeValue::Strings(vec!["a".to_string()]);
        let mut cloned = original.clone();
        if let OnnxAttributeValue::Strings(ref mut v) = cloned {
            v.push("b".to_string());
        }
        assert!(matches!(&original, OnnxAttributeValue::Strings(v) if v.len() == 1));
    }

    // --- OnnxAttribute struct: comprehensive field tests ---

    #[test]
    fn attribute_default_doc_string_is_empty() {
        let attr = OnnxAttribute {
            name: "test".to_string(),
            value: OnnxAttributeValue::Int(0),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        };
        assert!(attr.doc_string.is_empty());
    }

    #[test]
    fn attribute_name_preserved() {
        let attr = OnnxAttribute {
            name: "kernel_shape".to_string(),
            value: OnnxAttributeValue::Ints(vec![3, 3]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        };
        assert_eq!(attr.name, "kernel_shape");
    }

    #[test]
    fn attribute_with_all_fields_populated() {
        let attr = OnnxAttribute {
            name: "alpha".to_string(),
            value: OnnxAttributeValue::Float(0.1),
            doc_string: "LeakyReLU alpha parameter".to_string(),
            ref_attr_name: Some("parent_alpha".to_string()),
            attr_type: Some(proto::attribute_proto::AttributeType::Float),
        };
        assert_eq!(attr.name, "alpha");
        assert_eq!(attr.doc_string, "LeakyReLU alpha parameter");
        assert_eq!(attr.ref_attr_name.as_deref(), Some("parent_alpha"));
        assert!(matches!(attr.attr_type, Some(proto::attribute_proto::AttributeType::Float)));
    }

    #[test]
    fn attribute_debug_contains_all_field_names() {
        let attr = OnnxAttribute {
            name: "epsilon".to_string(),
            value: OnnxAttributeValue::Float(1e-5),
            doc_string: "small value".to_string(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Float),
        };
        let debug = format!("{attr:?}");
        assert!(debug.contains("epsilon"), "Debug should contain attribute name");
        assert!(debug.contains("Float"), "Debug should contain variant name");
        assert!(debug.contains("small value"), "Debug should contain doc_string");
    }

    // --- parse_utf8: additional edge cases ---

    #[test]
    fn parse_utf8_ascii_only() {
        let result = parse_utf8(b"abcABC123", "test").unwrap();
        assert_eq!(result, "abcABC123");
    }

    #[test]
    fn parse_utf8_single_byte() {
        let result = parse_utf8(b"x", "test").unwrap();
        assert_eq!(result, "x");
    }

    #[test]
    fn parse_utf8_error_contains_attr_name() {
        let invalid: &[u8] = &[0xFF];
        let result = parse_utf8(invalid, "my_special_attr");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("my_special_attr"), "error message should contain attribute name");
    }

    #[test]
    fn parse_utf8_mixed_valid_invalid() {
        // First byte valid, then invalid continuation.
        let mixed: &[u8] = &[0x61, 0xC2, 0x00]; // 'a' + truncated 2-byte UTF-8
        let result = parse_utf8(mixed, "test");
        // \x00 is valid UTF-8, so this should actually succeed with "a\u{00}\x00"...
        // Actually, 0xC2 followed by 0x00 is invalid: 0xC2 expects 0x80..0xBF continuation.
        assert!(result.is_err());
    }

    #[test]
    fn parse_utf8_emoji() {
        let emoji = "🎉🚀".as_bytes();
        let result = parse_utf8(emoji, "test").unwrap();
        assert_eq!(result, "🎉🚀");
    }

    // --- parse_strings: additional edge cases ---

    #[test]
    fn parse_strings_multiple_valid() {
        let input: Vec<Vec<u8>> = vec![
            b"first".to_vec(),
            b"second".to_vec(),
            b"third".to_vec(),
        ];
        let result = parse_strings(&input, "test").unwrap();
        assert_eq!(result, vec!["first", "second", "third"]);
    }

    #[test]
    fn parse_strings_error_propagates_attr_name() {
        let input: Vec<Vec<u8>> = vec![vec![0xFF]];
        let result = parse_strings(&input, "broken_attr");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("broken_attr"));
    }

    #[test]
    fn parse_strings_preserves_empty_strings() {
        let input: Vec<Vec<u8>> = vec![b"".to_vec(), b"nonempty".to_vec(), b"".to_vec()];
        let result = parse_strings(&input, "test").unwrap();
        assert_eq!(result.len(), 3);
        assert!(result[0].is_empty());
        assert_eq!(result[1], "nonempty");
        assert!(result[2].is_empty());
    }

    // --- parse_attr_type: all valid AttributeType codes ---

    #[test]
    fn parse_attr_type_string_code() {
        let at = parse_attr_type(Some(3));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::String)));
    }

    #[test]
    fn parse_attr_type_tensor_code() {
        let at = parse_attr_type(Some(4));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Tensor)));
    }

    #[test]
    fn parse_attr_type_ints_code() {
        let at = parse_attr_type(Some(7));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Ints)));
    }

    #[test]
    fn parse_attr_type_strings_code() {
        let at = parse_attr_type(Some(8));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Strings)));
    }

    #[test]
    fn parse_attr_type_tensors_code() {
        let at = parse_attr_type(Some(9));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Tensors)));
    }

    #[test]
    fn parse_attr_type_graphs_code() {
        let at = parse_attr_type(Some(10));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::Graphs)));
    }

    #[test]
    fn parse_attr_type_sparse_tensor_code() {
        let at = parse_attr_type(Some(11));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::SparseTensor)));
    }

    #[test]
    fn parse_attr_type_sparse_tensors_code() {
        let at = parse_attr_type(Some(12));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::SparseTensors)));
    }

    #[test]
    fn parse_attr_type_type_proto_code() {
        let at = parse_attr_type(Some(13));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::TypeProto)));
    }

    #[test]
    fn parse_attr_type_type_protos_code() {
        let at = parse_attr_type(Some(14));
        assert!(matches!(at, Some(proto::attribute_proto::AttributeType::TypeProtos)));
    }

    #[test]
    fn parse_attr_type_negative_value() {
        let at = parse_attr_type(Some(-1));
        assert!(at.is_none(), "negative i32 should map to no valid AttributeType");
    }

    #[test]
    fn parse_attr_type_large_value() {
        let at = parse_attr_type(Some(i32::MAX));
        assert!(at.is_none(), "i32::MAX should map to no valid AttributeType");
    }

    // --- missing_attr_value: additional coverage ---

    #[test]
    fn missing_attr_value_with_empty_name() {
        let err = missing_attr_value("", "tensor");
        let msg = format!("{err}");
        assert!(msg.contains("tensor"));
    }

    #[test]
    fn missing_attr_value_with_long_name() {
        let long_name = "a".repeat(500);
        let err = missing_attr_value(&long_name, "graph");
        let msg = format!("{err}");
        assert!(msg.contains(&long_name));
    }

    // --- LoaderError::Onnx Display ---

    #[test]
    fn loader_error_onnx_empty_message() {
        let err = LoaderError::Onnx(String::new());
        let msg = format!("{err}");
        // Verify formatting completes without panic
        let _ = msg;
    }

    #[test]
    fn loader_error_onnx_unicode_message() {
        let err = LoaderError::Onnx("错误：模型加载失败".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("错误：模型加载失败"));
    }

    // --- OnnxAttributeValue: Debug format for each variant individually ---

    #[test]
    fn debug_format_float_variant() {
        let v = OnnxAttributeValue::Float(1.5);
        let s = format!("{v:?}");
        assert!(s.contains("Float"), "Debug should contain 'Float', got: {s}");
    }

    #[test]
    fn debug_format_int_variant() {
        let v = OnnxAttributeValue::Int(99);
        let s = format!("{v:?}");
        assert!(s.contains("Int"), "Debug should contain 'Int', got: {s}");
    }

    #[test]
    fn debug_format_string_variant() {
        let v = OnnxAttributeValue::String("test_value".to_string());
        let s = format!("{v:?}");
        assert!(s.contains("String"), "Debug should contain 'String', got: {s}");
    }

    #[test]
    fn debug_format_floats_variant() {
        let v = OnnxAttributeValue::Floats(vec![1.0, 2.0, 3.0]);
        let s = format!("{v:?}");
        assert!(s.contains("Floats"), "Debug should contain 'Floats', got: {s}");
    }

    #[test]
    fn debug_format_ints_variant() {
        let v = OnnxAttributeValue::Ints(vec![5, 6, 7]);
        let s = format!("{v:?}");
        assert!(s.contains("Ints"), "Debug should contain 'Ints', got: {s}");
    }

    #[test]
    fn debug_format_strings_variant() {
        let v = OnnxAttributeValue::Strings(vec!["hello".to_string()]);
        let s = format!("{v:?}");
        assert!(s.contains("Strings"), "Debug should contain 'Strings', got: {s}");
    }

    #[test]
    fn debug_format_tensors_variant() {
        let v = OnnxAttributeValue::Tensors(vec![]);
        let s = format!("{v:?}");
        assert!(s.contains("Tensors"), "Debug should contain 'Tensors', got: {s}");
    }

    #[test]
    fn debug_format_sparse_tensors_variant() {
        let v = OnnxAttributeValue::SparseTensors(vec![]);
        let s = format!("{v:?}");
        assert!(s.contains("SparseTensors"), "Debug should contain 'SparseTensors', got: {s}");
    }

    #[test]
    fn debug_format_types_variant() {
        let v = OnnxAttributeValue::Types(vec![]);
        let s = format!("{v:?}");
        assert!(s.contains("Types"), "Debug should contain 'Types', got: {s}");
    }

    #[test]
    fn debug_format_graphs_variant() {
        let v = OnnxAttributeValue::Graphs(vec![]);
        let s = format!("{v:?}");
        assert!(s.contains("Graphs"), "Debug should contain 'Graphs', got: {s}");
    }

    #[test]
    fn debug_format_ref_variant() {
        let v = OnnxAttributeValue::Ref("target".to_string());
        let s = format!("{v:?}");
        assert!(s.contains("Ref"), "Debug should contain 'Ref', got: {s}");
    }

    // ── 40 new tests: parse_attributes / parse_attribute / parse_by_type / parse_by_presence ─

    /// Helper: create an ExternalDataResolver pointing at a temp dir (no real files needed for
    /// non-tensor attribute tests).
    fn make_resolver() -> ExternalDataResolver {
        ExternalDataResolver::new(std::path::Path::new("/tmp/__gllm_test_onnx_attr__/model.onnx"))
    }

    /// Helper: build a minimal AttributeProto with a name.
    fn make_attr_proto(name: &str) -> proto::AttributeProto {
        proto::AttributeProto {
            name: Some(name.to_string()),
            ..Default::default()
        }
    }

    // --- parse_attributes ---

    #[test]
    fn parse_attributes_empty_list() {
        let mut resolver = make_resolver();
        let result = parse_attributes(vec![], &mut resolver).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn parse_attributes_single_float_attr() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            name: Some("alpha".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
            f: Some(0.25),
            ..Default::default()
        };
        let result = parse_attributes(vec![attr], &mut resolver).unwrap();
        assert_eq!(result.len(), 1);
        let parsed = &result["alpha"];
        assert!(matches!(&parsed.value, OnnxAttributeValue::Float(v) if (*v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn parse_attributes_duplicate_name_returns_error() {
        let mut resolver = make_resolver();
        let a1 = proto::AttributeProto {
            name: Some("x".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(1),
            ..Default::default()
        };
        let a2 = proto::AttributeProto {
            name: Some("x".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(2),
            ..Default::default()
        };
        let err = parse_attributes(vec![a1, a2], &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("duplicate"), "expected 'duplicate' in error, got: {msg}");
    }

    #[test]
    fn parse_attributes_multiple_distinct_names() {
        let mut resolver = make_resolver();
        let a1 = proto::AttributeProto {
            name: Some("a".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(10),
            ..Default::default()
        };
        let a2 = proto::AttributeProto {
            name: Some("b".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
            f: Some(3.14),
            ..Default::default()
        };
        let a3 = proto::AttributeProto {
            name: Some("c".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::String as i32),
            s: Some(b"hello".to_vec()),
            ..Default::default()
        };
        let result = parse_attributes(vec![a1, a2, a3], &mut resolver).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.contains_key("a"));
        assert!(result.contains_key("b"));
        assert!(result.contains_key("c"));
    }

    #[test]
    fn parse_attributes_missing_name_returns_error() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            name: None,
            i: Some(42),
            ..Default::default()
        };
        let err = parse_attributes(vec![attr], &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("missing name"), "expected 'missing name' in error, got: {msg}");
    }

    // --- parse_attribute: ref_attr_name path ---

    #[test]
    fn parse_attribute_ref_attr_name_non_empty() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            name: Some("my_attr".to_string()),
            ref_attr_name: Some("target_attr".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            ..Default::default()
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert_eq!(result.name, "my_attr");
        assert!(matches!(&result.value, OnnxAttributeValue::Ref(r) if r == "target_attr"));
        assert_eq!(result.ref_attr_name.as_deref(), Some("target_attr"));
    }

    #[test]
    fn parse_attribute_ref_attr_name_empty_falls_through() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            name: Some("my_attr".to_string()),
            ref_attr_name: Some(String::new()),
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(99),
            ..Default::default()
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Int(99)));
    }

    #[test]
    fn parse_attribute_ref_attr_name_none_falls_through() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            name: Some("my_attr".to_string()),
            ref_attr_name: None,
            r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
            f: Some(1.5),
            ..Default::default()
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Float(v) if (*v - 1.5).abs() < 1e-6));
    }

    #[test]
    fn parse_attribute_doc_string_preserved_from_proto() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            name: Some("desc_attr".to_string()),
            doc_string: Some("A documented attribute".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(0),
            ..Default::default()
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert_eq!(result.doc_string, "A documented attribute");
    }

    #[test]
    fn parse_attribute_doc_string_default_when_missing() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            name: Some("no_doc".to_string()),
            doc_string: None,
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(1),
            ..Default::default()
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(result.doc_string.is_empty());
    }

    // --- parse_by_type: each AttributeType branch ---

    #[test]
    fn parse_by_type_float_present() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
            f: Some(2.718),
            ..make_attr_proto("f")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Float(v) if (*v - 2.718).abs() < 0.001));
    }

    #[test]
    fn parse_by_type_float_missing_value() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
            f: None,
            ..make_attr_proto("f_missing")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("f_missing") && msg.contains("float"), "got: {msg}");
    }

    #[test]
    fn parse_by_type_int_present() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(-999),
            ..make_attr_proto("i")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Int(-999)));
    }

    #[test]
    fn parse_by_type_int_missing_value() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            ..make_attr_proto("i_missing")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("i_missing"), "got: {msg}");
    }

    #[test]
    fn parse_by_type_string_present() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::String as i32),
            s: Some(b"pads".to_vec()),
            ..make_attr_proto("s")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::String(s) if s == "pads"));
    }

    #[test]
    fn parse_by_type_string_missing_value() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::String as i32),
            s: None,
            ..make_attr_proto("s_missing")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("s_missing"), "got: {msg}");
    }

    #[test]
    fn parse_by_type_string_invalid_utf8() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::String as i32),
            s: Some(vec![0xFF, 0xFE]),
            ..make_attr_proto("bad_utf8")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("invalid utf8"), "got: {msg}");
    }

    #[test]
    fn parse_by_type_floats_present() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Floats as i32),
            floats: vec![1.0, 2.0, 3.0],
            ..make_attr_proto("floats")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Floats(v) if v.len() == 3));
    }

    #[test]
    fn parse_by_type_floats_empty_vec() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Floats as i32),
            floats: vec![],
            ..make_attr_proto("empty_floats")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Floats(v) if v.is_empty()));
    }

    #[test]
    fn parse_by_type_ints_present() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
            ints: vec![1, 2, 3, 4],
            ..make_attr_proto("ints")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Ints(v) if *v == vec![1i64, 2, 3, 4]));
    }

    #[test]
    fn parse_by_type_strings_present() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Strings as i32),
            strings: vec![b"foo".to_vec(), b"bar".to_vec()],
            ..make_attr_proto("strs")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Strings(v) if v.len() == 2));
    }

    #[test]
    fn parse_by_type_strings_invalid_member() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Strings as i32),
            strings: vec![b"ok".to_vec(), vec![0xFF]],
            ..make_attr_proto("bad_strs")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("invalid utf8"), "got: {msg}");
    }

    #[test]
    fn parse_by_type_undefined_returns_error() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Undefined as i32),
            ..make_attr_proto("undef_type")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("undefined type"), "got: {msg}");
    }

    // --- parse_by_presence: fallback when attr type is None ---

    #[test]
    fn parse_by_presence_float() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            f: Some(1.23),
            ..make_attr_proto("pf")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Float(v) if (*v - 1.23).abs() < 0.01));
    }

    #[test]
    fn parse_by_presence_int() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            i: Some(42),
            ..make_attr_proto("pi")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Int(42)));
    }

    #[test]
    fn parse_by_presence_string() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            s: Some(b"value".to_vec()),
            ..make_attr_proto("ps")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::String(s) if s == "value"));
    }

    #[test]
    fn parse_by_presence_floats() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            floats: vec![10.0, 20.0],
            ..make_attr_proto("pfs")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Floats(v) if v.len() == 2));
    }

    #[test]
    fn parse_by_presence_ints() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            ints: vec![5, 6, 7],
            ..make_attr_proto("pis")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Ints(v) if *v == vec![5i64, 6, 7]));
    }

    #[test]
    fn parse_by_presence_strings() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            strings: vec![b"a".to_vec(), b"b".to_vec()],
            ..make_attr_proto("pss")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Strings(v) if v.len() == 2));
    }

    #[test]
    fn parse_by_presence_empty_floats_skipped() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            floats: vec![],
            i: Some(7),
            ..make_attr_proto("empty_fs")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Empty floats is skipped, falls through to int
        assert!(matches!(&result.value, OnnxAttributeValue::Int(7)));
    }

    #[test]
    fn parse_by_presence_no_value_returns_error() {
        let mut resolver = make_resolver();
        let attr = make_attr_proto("nothing");
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("missing value"), "got: {msg}");
    }

    #[test]
    fn parse_by_presence_string_invalid_utf8() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            s: Some(vec![0xC2, 0x00]),
            ..make_attr_proto("bad_ps")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("invalid utf8"), "got: {msg}");
    }

    // --- attr_type preservation ---

    #[test]
    fn attr_type_preserved_from_proto() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            name: Some("typed".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
            ints: vec![1, 2],
            ..Default::default()
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert_eq!(
            result.attr_type,
            Some(proto::attribute_proto::AttributeType::Ints)
        );
    }

    #[test]
    fn attr_type_none_when_proto_type_absent() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            name: Some("untyped".to_string()),
            i: Some(1),
            ..Default::default()
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(result.attr_type.is_none());
    }

    // --- parse_by_presence: graph/tensor/type precedence ---

    #[test]
    fn parse_by_presence_graph_with_empty_graph_proto() {
        let mut resolver = make_resolver();
        let graph_proto = proto::GraphProto {
            node: vec![],
            name: Some("empty_graph".to_string()),
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            g: Some(graph_proto),
            ..make_attr_proto("pg")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Graph(_)));
    }

    #[test]
    fn parse_by_presence_graphs_multiple() {
        let mut resolver = make_resolver();
        let g1 = proto::GraphProto { name: Some("g1".to_string()), ..Default::default() };
        let g2 = proto::GraphProto { name: Some("g2".to_string()), ..Default::default() };
        let attr = proto::AttributeProto {
            graphs: vec![g1, g2],
            ..make_attr_proto("pgs")
        };
        let result = parse_attribute(attr, &mut resolver).unwrap();
        assert!(matches!(&result.value, OnnxAttributeValue::Graphs(v) if v.len() == 2));
    }

    // --- OnnxAttributeValue: Tensor/SparseTensor variant round-trip via parse_by_type ---

    #[test]
    fn parse_by_type_tensor_missing_value_returns_error() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Tensor as i32),
            t: None,
            ..make_attr_proto("t_missing")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("t_missing"), "got: {msg}");
    }

    #[test]
    fn parse_by_type_sparse_tensor_missing_value_returns_error() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::SparseTensor as i32),
            sparse_tensor: None,
            ..make_attr_proto("sp_missing")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("sp_missing"), "got: {msg}");
    }

    #[test]
    fn parse_by_type_type_proto_missing_value_returns_error() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::TypeProto as i32),
            tp: None,
            ..make_attr_proto("tp_missing")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("tp_missing"), "got: {msg}");
    }

    #[test]
    fn parse_by_type_graph_missing_value_returns_error() {
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Graph as i32),
            g: None,
            ..make_attr_proto("g_missing")
        };
        let err = parse_attribute(attr, &mut resolver).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("g_missing"), "got: {msg}");
    }

    // --- OnnxAttributeValue: Type variant with public types ---

    #[test]
    fn attribute_value_type_variant_constructible() {
        let tp = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let v = OnnxAttributeValue::Type(tp);
        let s = format!("{v:?}");
        assert!(s.contains("Type"), "Debug should contain 'Type', got: {s}");
    }

    #[test]
    fn attribute_value_types_vec_constructible() {
        let tp = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int64,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(3)] },
        });
        let v = OnnxAttributeValue::Types(vec![tp]);
        let s = format!("{v:?}");
        assert!(s.contains("Types"), "Debug should contain 'Types', got: {s}");
    }

    #[test]
    fn attribute_value_type_sequence_variant() {
        let inner = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let seq = OnnxType::Sequence(Box::new(inner));
        let v = OnnxAttributeValue::Type(seq);
        let s = format!("{v:?}");
        assert!(s.contains("Sequence"), "Debug should contain 'Sequence', got: {s}");
    }

    #[test]
    fn attribute_value_type_map_variant() {
        let val = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let map = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(val),
        });
        let v = OnnxAttributeValue::Type(map);
        let s = format!("{v:?}");
        assert!(s.contains("Map"), "Debug should contain 'Map', got: {s}");
    }

    #[test]
    fn attribute_value_type_optional_variant() {
        let inner = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Double,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Unknown] },
        });
        let opt = OnnxType::Optional(Box::new(inner));
        let v = OnnxAttributeValue::Type(opt);
        let s = format!("{v:?}");
        assert!(s.contains("Optional"), "Debug should contain 'Optional', got: {s}");
    }

    #[test]
    fn attribute_value_type_sparse_tensor_variant() {
        let sp = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        let v = OnnxAttributeValue::Type(sp);
        let s = format!("{v:?}");
        assert!(s.contains("SparseTensor"), "Debug should contain 'SparseTensor', got: {s}");
    }

    // ── 15 additional tests ────────────────────────────────────────────

    // --- parse_by_presence: precedence (float wins over int when both set) ---

    #[test]
    fn parse_by_presence_float_takes_precedence_over_int() {
        // Arrange: attribute with both f and i set, no type field.
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            f: Some(2.5),
            i: Some(99),
            ..make_attr_proto("prec")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: float is checked first in parse_by_presence, so Float wins.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Float(v) if (*v - 2.5).abs() < 0.01),
            "expected Float(2.5) when both f and i are set"
        );
    }

    // --- parse_by_presence: tensor path ---

    #[test]
    fn parse_by_presence_tensor_with_minimal_proto() {
        // Arrange: attribute with a tensor proto set, no type field.
        let mut resolver = make_resolver();
        let tensor_proto = proto::TensorProto {
            name: Some("t_attr".to_string()),
            data_type: Some(proto::tensor_proto::DataType::Float as i32),
            dims: vec![2],
            float_data: vec![1.0, 2.0],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            t: Some(tensor_proto),
            ..make_attr_proto("pt")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::Tensor(_)),
            "expected Tensor variant"
        );
    }

    // --- parse_by_presence: sparse_tensor single ---

    #[test]
    fn parse_by_presence_sparse_tensor_single() {
        // Arrange: attribute with sparse_tensor set (no type field).
        let mut resolver = make_resolver();
        let values = proto::TensorProto {
            name: Some("sp_vals".to_string()),
            data_type: Some(proto::tensor_proto::DataType::Float as i32),
            float_data: vec![5.0],
            dims: vec![1],
            ..Default::default()
        };
        let indices = proto::TensorProto {
            name: Some("sp_idxs".to_string()),
            data_type: Some(proto::tensor_proto::DataType::Int64 as i32),
            int64_data: vec![0],
            dims: vec![1],
            ..Default::default()
        };
        let sparse = proto::SparseTensorProto {
            values: Some(values),
            indices: Some(indices),
            dims: vec![1],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            sparse_tensor: Some(sparse),
            ..make_attr_proto("pst")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::SparseTensor(_)),
            "expected SparseTensor variant"
        );
    }

    // --- parse_by_presence: tensors list ---

    #[test]
    fn parse_by_presence_tensors_list() {
        // Arrange: attribute with multiple tensors, no type field.
        let mut resolver = make_resolver();
        let t1 = proto::TensorProto {
            name: Some("t1".to_string()),
            data_type: Some(proto::tensor_proto::DataType::Float as i32),
            dims: vec![1],
            float_data: vec![3.0],
            ..Default::default()
        };
        let t2 = proto::TensorProto {
            name: Some("t2".to_string()),
            data_type: Some(proto::tensor_proto::DataType::Float as i32),
            dims: vec![1],
            float_data: vec![4.0],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            tensors: vec![t1, t2],
            ..make_attr_proto("pts")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::Tensors(v) if v.len() == 2),
            "expected Tensors with 2 elements"
        );
    }

    // --- parse_by_presence: sparse_tensors list ---

    #[test]
    fn parse_by_presence_sparse_tensors_list() {
        // Arrange: attribute with multiple sparse tensors, no type field.
        let mut resolver = make_resolver();
        let make_sparse = |val: f32, idx: usize| -> proto::SparseTensorProto {
            let values = proto::TensorProto {
                name: Some(format!("sp_vals_{idx}")),
                data_type: Some(proto::tensor_proto::DataType::Float as i32),
                float_data: vec![val],
                dims: vec![1],
                ..Default::default()
            };
            let indices = proto::TensorProto {
                name: Some(format!("sp_idxs_{idx}")),
                data_type: Some(proto::tensor_proto::DataType::Int64 as i32),
                int64_data: vec![0],
                dims: vec![1],
                ..Default::default()
            };
            proto::SparseTensorProto {
                values: Some(values),
                indices: Some(indices),
                dims: vec![1],
                ..Default::default()
            }
        };
        let attr = proto::AttributeProto {
            sparse_tensors: vec![make_sparse(1.0, 0), make_sparse(2.0, 1)],
            ..make_attr_proto("psps")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::SparseTensors(v) if v.len() == 2),
            "expected SparseTensors with 2 elements"
        );
    }

    // --- parse_by_presence: tp (single TypeProto) ---

    #[test]
    fn parse_by_presence_type_proto_single() {
        // Arrange: attribute with tp (TypeProto) set, no type field.
        let mut resolver = make_resolver();
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(proto::tensor_proto::DataType::Float as i32),
                shape: Some(proto::TensorShapeProto {
                    dim: vec![proto::tensor_shape_proto::Dimension {
                        denotation: None,
                        value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(3)),
                    }],
                }),
            })),
        };
        let attr = proto::AttributeProto {
            tp: Some(type_proto),
            ..make_attr_proto("ptp")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::Type(_)),
            "expected Type variant"
        );
    }

    // --- parse_by_presence: type_protos list ---

    #[test]
    fn parse_by_presence_type_protos_list() {
        // Arrange: attribute with multiple type_protos, no type field.
        let mut resolver = make_resolver();
        let make_tp = |dt: i32| -> proto::TypeProto {
            proto::TypeProto {
                denotation: None,
                value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                    elem_type: Some(dt),
                    shape: None,
                })),
            }
        };
        let attr = proto::AttributeProto {
            type_protos: vec![
                make_tp(proto::tensor_proto::DataType::Float as i32),
                make_tp(proto::tensor_proto::DataType::Int64 as i32),
            ],
            ..make_attr_proto("ptps")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::Types(v) if v.len() == 2),
            "expected Types with 2 elements"
        );
    }

    // --- parse_by_type: tensors with actual tensor protos ---

    #[test]
    fn parse_by_type_tensors_with_values() {
        // Arrange: attribute with typed Tensors, containing actual tensor protos.
        let mut resolver = make_resolver();
        let t = proto::TensorProto {
            name: Some("typed_t".to_string()),
            data_type: Some(proto::tensor_proto::DataType::Float as i32),
            dims: vec![3],
            float_data: vec![10.0, 20.0, 30.0],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Tensors as i32),
            tensors: vec![t],
            ..make_attr_proto("typed_tensors")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::Tensors(v) if v.len() == 1),
            "expected Tensors with 1 element"
        );
    }

    // --- parse_by_type: graphs with multiple graph protos ---

    #[test]
    fn parse_by_type_graphs_multiple() {
        // Arrange: attribute with typed Graphs, containing 3 graph protos.
        let mut resolver = make_resolver();
        let g1 = proto::GraphProto { name: Some("then".to_string()), ..Default::default() };
        let g2 = proto::GraphProto { name: Some("else".to_string()), ..Default::default() };
        let g3 = proto::GraphProto { name: Some("merge".to_string()), ..Default::default() };
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Graphs as i32),
            graphs: vec![g1, g2, g3],
            ..make_attr_proto("typed_graphs")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::Graphs(v) if v.len() == 3),
            "expected Graphs with 3 elements"
        );
    }

    // --- parse_by_type: type_protos with values ---

    #[test]
    fn parse_by_type_type_protos_with_values() {
        // Arrange: attribute with typed TypeProtos.
        let mut resolver = make_resolver();
        let make_tp = |dt: i32| -> proto::TypeProto {
            proto::TypeProto {
                denotation: None,
                value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                    elem_type: Some(dt),
                    shape: None,
                })),
            }
        };
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::TypeProtos as i32),
            type_protos: vec![
                make_tp(proto::tensor_proto::DataType::Double as i32),
                make_tp(proto::tensor_proto::DataType::Bool as i32),
            ],
            ..make_attr_proto("typed_tps")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::Types(v) if v.len() == 2),
            "expected Types with 2 elements"
        );
    }

    // --- parse_by_type: ints empty vec is valid (returns Ints([])) ---

    #[test]
    fn parse_by_type_ints_empty_vec() {
        // Arrange: attribute typed as Ints but with empty ints list.
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
            ints: vec![],
            ..make_attr_proto("empty_ints_typed")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: Ints(vec![]) is valid, unlike scalar Int which requires a value.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Ints(v) if v.is_empty()),
            "expected empty Ints"
        );
    }

    // --- parse_by_type: graph with actual nodes ---

    #[test]
    fn parse_by_type_graph_with_node() {
        // Arrange: attribute typed as Graph, with a single node inside.
        let mut resolver = make_resolver();
        let node = proto::NodeProto {
            name: Some("add_node".to_string()),
            op_type: Some("Add".to_string()),
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            node: vec![node],
            name: Some("subgraph".to_string()),
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Graph as i32),
            g: Some(graph_proto),
            ..make_attr_proto("typed_graph")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        match &result.value {
            OnnxAttributeValue::Graph(g) => {
                assert_eq!(g.nodes.len(), 1, "subgraph should have 1 node");
                assert_eq!(g.nodes[0].op_type, "Add");
            }
            other => panic!("expected Graph variant, got {:?}", other),
        }
    }

    // --- parse_by_type: sparse_tensors with actual sparse protos ---

    #[test]
    fn parse_by_type_sparse_tensors_with_values() {
        // Arrange: attribute typed as SparseTensors with 2 sparse tensor protos.
        let mut resolver = make_resolver();
        let make_sparse = |val: f32, idx: usize| -> proto::SparseTensorProto {
            let values = proto::TensorProto {
                name: Some(format!("typed_sp_vals_{idx}")),
                data_type: Some(proto::tensor_proto::DataType::Float as i32),
                float_data: vec![val],
                dims: vec![1],
                ..Default::default()
            };
            let indices = proto::TensorProto {
                name: Some(format!("typed_sp_idxs_{idx}")),
                data_type: Some(proto::tensor_proto::DataType::Int64 as i32),
                int64_data: vec![0],
                dims: vec![1],
                ..Default::default()
            };
            proto::SparseTensorProto {
                values: Some(values),
                indices: Some(indices),
                dims: vec![1],
                ..Default::default()
            }
        };
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::SparseTensors as i32),
            sparse_tensors: vec![make_sparse(7.0, 0), make_sparse(8.0, 1)],
            ..make_attr_proto("typed_sps")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::SparseTensors(v) if v.len() == 2),
            "expected SparseTensors with 2 elements"
        );
    }

    // --- parse_by_type: strings empty vec is valid ---

    #[test]
    fn parse_by_type_strings_empty_vec() {
        // Arrange: attribute typed as Strings with empty list.
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Strings as i32),
            strings: vec![],
            ..make_attr_proto("empty_strs_typed")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: Strings(vec![]) is valid (empty repeated field).
        assert!(
            matches!(&result.value, OnnxAttributeValue::Strings(v) if v.is_empty()),
            "expected empty Strings"
        );
    }

    // --- attribute debug format includes ref_attr_name and attr_type ---

    #[test]
    fn attribute_debug_includes_ref_attr_name_field() {
        // Arrange
        let attr = OnnxAttribute {
            name: "ref_test".to_string(),
            value: OnnxAttributeValue::Ref("target".to_string()),
            doc_string: String::new(),
            ref_attr_name: Some("target".to_string()),
            attr_type: None,
        };
        // Act
        let debug = format!("{attr:?}");
        // Assert: Debug output should contain the ref_attr_name value.
        assert!(
            debug.contains("ref_attr_name"),
            "Debug should contain 'ref_attr_name' field, got: {debug}"
        );
        assert!(
            debug.contains("target"),
            "Debug should contain the ref target value, got: {debug}"
        );
    }

    #[test]
    fn attribute_debug_includes_attr_type_field() {
        // Arrange
        let attr = OnnxAttribute {
            name: "typed_debug".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 2]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        };
        // Act
        let debug = format!("{attr:?}");
        // Assert: Debug output should contain the attr_type field.
        assert!(
            debug.contains("attr_type"),
            "Debug should contain 'attr_type' field, got: {debug}"
        );
    }

    // ── 13 additional tests ──────────────────────────────────────────

    // --- parse_by_type: Tensor with actual tensor proto ---

    #[test]
    fn parse_by_type_tensor_with_float_data() {
        // Arrange: attribute typed as Tensor with a valid float tensor proto.
        let mut resolver = make_resolver();
        let tensor_proto = proto::TensorProto {
            name: Some("attr_tensor".to_string()),
            data_type: Some(proto::tensor_proto::DataType::Float as i32),
            dims: vec![4],
            float_data: vec![1.0, 2.0, 3.0, 4.0],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Tensor as i32),
            t: Some(tensor_proto),
            ..make_attr_proto("typed_tensor")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::Tensor(_)),
            "expected Tensor variant"
        );
    }

    // --- parse_by_type: SparseTensor with valid proto ---

    #[test]
    fn parse_by_type_sparse_tensor_with_valid_proto() {
        // Arrange: attribute typed as SparseTensor with a valid sparse proto.
        let mut resolver = make_resolver();
        let values = proto::TensorProto {
            name: Some("vals".to_string()),
            data_type: Some(proto::tensor_proto::DataType::Float as i32),
            float_data: vec![10.0, 20.0],
            dims: vec![2],
            ..Default::default()
        };
        let indices = proto::TensorProto {
            name: Some("idxs".to_string()),
            data_type: Some(proto::tensor_proto::DataType::Int64 as i32),
            int64_data: vec![0, 5],
            dims: vec![2],
            ..Default::default()
        };
        let sparse = proto::SparseTensorProto {
            values: Some(values),
            indices: Some(indices),
            dims: vec![10],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::SparseTensor as i32),
            sparse_tensor: Some(sparse),
            ..make_attr_proto("typed_sparse")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        assert!(
            matches!(&result.value, OnnxAttributeValue::SparseTensor(_)),
            "expected SparseTensor variant"
        );
    }

    // --- parse_by_type: TypeProto with valid tensor type ---

    #[test]
    fn parse_by_type_type_proto_with_tensor_type() {
        // Arrange: attribute typed as TypeProto with a valid tensor type.
        let mut resolver = make_resolver();
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(proto::tensor_proto::DataType::Int32 as i32),
                shape: Some(proto::TensorShapeProto {
                    dim: vec![
                        proto::tensor_shape_proto::Dimension {
                            denotation: None,
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(2)),
                        },
                        proto::tensor_shape_proto::Dimension {
                            denotation: None,
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(3)),
                        },
                    ],
                }),
            })),
        };
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::TypeProto as i32),
            tp: Some(type_proto),
            ..make_attr_proto("typed_type_proto")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        match &result.value {
            OnnxAttributeValue::Type(ot) => {
                let s = format!("{ot:?}");
                assert!(s.contains("Tensor"), "expected Tensor type, got: {s}");
            }
            other => panic!("expected Type variant, got {:?}", other),
        }
    }

    // --- parse_by_presence: empty ints skipped, falls through to float ---

    #[test]
    fn parse_by_presence_empty_ints_skipped() {
        // Arrange: empty ints list with a float present, no type field.
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            ints: vec![],
            f: Some(3.14),
            ..make_attr_proto("empty_ints_fall")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: empty ints is skipped, falls through to float.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Float(v) if (*v - 3.14).abs() < 0.01),
            "expected Float(3.14) when ints is empty"
        );
    }

    // --- parse_by_presence: empty strings skipped, falls through to ints ---

    #[test]
    fn parse_by_presence_empty_strings_skipped() {
        // Arrange: empty strings list with ints present, no type field.
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            strings: vec![],
            ints: vec![7, 8, 9],
            ..make_attr_proto("empty_strs_fall")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: empty strings is skipped, falls through to ints.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Ints(v) if *v == vec![7i64, 8, 9]),
            "expected Ints([7, 8, 9]) when strings is empty"
        );
    }

    // --- parse_by_presence: empty tensors skipped, falls through to graphs ---

    #[test]
    fn parse_by_presence_empty_tensors_skipped() {
        // Arrange: empty tensors list with graphs present, no type field.
        let mut resolver = make_resolver();
        let g = proto::GraphProto {
            name: Some("fallback_graph".to_string()),
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            tensors: vec![],
            graphs: vec![g],
            ..make_attr_proto("empty_tensors_fall")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: empty tensors is skipped, falls through to graphs.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Graphs(v) if v.len() == 1),
            "expected Graphs with 1 element when tensors is empty"
        );
    }

    // --- parse_by_presence: empty graphs skipped, falls through to type_proto ---

    #[test]
    fn parse_by_presence_empty_graphs_skipped() {
        // Arrange: empty graphs list with tp present, no type field.
        let mut resolver = make_resolver();
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(proto::tensor_proto::DataType::Bool as i32),
                shape: None,
            })),
        };
        let attr = proto::AttributeProto {
            graphs: vec![],
            tp: Some(type_proto),
            ..make_attr_proto("empty_graphs_fall")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: empty graphs is skipped, falls through to type_proto.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Type(_)),
            "expected Type variant when graphs is empty"
        );
    }

    // --- parse_by_presence: empty sparse_tensors skipped, falls through to type_protos ---

    #[test]
    fn parse_by_presence_empty_sparse_tensors_skipped() {
        // Arrange: empty sparse_tensors list with type_protos present, no type field.
        let mut resolver = make_resolver();
        let make_tp = |dt: i32| -> proto::TypeProto {
            proto::TypeProto {
                denotation: None,
                value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                    elem_type: Some(dt),
                    shape: None,
                })),
            }
        };
        let attr = proto::AttributeProto {
            sparse_tensors: vec![],
            type_protos: vec![make_tp(proto::tensor_proto::DataType::Float as i32)],
            ..make_attr_proto("empty_sp_fall")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: empty sparse_tensors is skipped, falls through to type_protos.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Types(v) if v.len() == 1),
            "expected Types with 1 element when sparse_tensors is empty"
        );
    }

    // --- parse_by_type: float zero value is not confused with missing ---

    #[test]
    fn parse_by_type_float_zero_is_valid() {
        // Arrange: attribute typed as Float with value 0.0.
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
            f: Some(0.0),
            ..make_attr_proto("zero_float")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: 0.0 is a valid float, not treated as missing.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Float(v) if *v == 0.0),
            "expected Float(0.0)"
        );
    }

    // --- parse_by_type: int zero value is not confused with missing ---

    #[test]
    fn parse_by_type_int_zero_is_valid() {
        // Arrange: attribute typed as Int with value 0.
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(0),
            ..make_attr_proto("zero_int")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: 0 is a valid int, not treated as missing.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Int(0)),
            "expected Int(0)"
        );
    }

    // --- parse_by_presence: string takes precedence over float when string is set ---

    #[test]
    fn parse_by_presence_float_wins_over_string_due_to_order() {
        // Arrange: both f and s are set, no type field.
        // parse_by_presence checks f first, then i, then s.
        let mut resolver = make_resolver();
        let attr = proto::AttributeProto {
            f: Some(1.0),
            s: Some(b"should_not_win".to_vec()),
            ..make_attr_proto("prec_f_s")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert: float is checked before string in parse_by_presence.
        assert!(
            matches!(&result.value, OnnxAttributeValue::Float(v) if *v == 1.0),
            "expected Float(1.0) due to precedence order"
        );
    }

    // --- parse_by_type: Graph variant (Box<OnnxGraph>) contains correct node count ---

    #[test]
    fn parse_by_type_graph_single_node_preserves_details() {
        // Arrange: graph with one Relu node.
        let mut resolver = make_resolver();
        let node = proto::NodeProto {
            name: Some("relu_0".to_string()),
            op_type: Some("Relu".to_string()),
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            ..Default::default()
        };
        let graph_proto = proto::GraphProto {
            node: vec![node],
            name: Some("relu_graph".to_string()),
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            r#type: Some(proto::attribute_proto::AttributeType::Graph as i32),
            g: Some(graph_proto),
            ..make_attr_proto("graph_details")
        };
        // Act
        let result = parse_attribute(attr, &mut resolver).unwrap();
        // Assert
        match &result.value {
            OnnxAttributeValue::Graph(g) => {
                assert_eq!(g.nodes.len(), 1);
                assert_eq!(g.nodes[0].op_type, "Relu");
                assert_eq!(g.nodes[0].inputs.len(), 1);
                assert_eq!(g.nodes[0].outputs.len(), 1);
            }
            other => panic!("expected Graph variant, got {:?}", other),
        }
    }

    // --- parse_attributes: large number of attributes all preserved ---

    #[test]
    fn parse_attributes_many_attributes_all_accessible() {
        // Arrange: 50 int attributes with distinct names.
        let mut resolver = make_resolver();
        let attrs: Vec<proto::AttributeProto> = (0..50)
            .map(|i| proto::AttributeProto {
                name: Some(format!("attr_{i}")),
                r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
                i: Some(i as i64),
                ..Default::default()
            })
            .collect();
        // Act
        let result = parse_attributes(attrs, &mut resolver).unwrap();
        // Assert: all 50 attributes are accessible by name.
        assert_eq!(result.len(), 50);
        for i in 0..50 {
            let key = format!("attr_{i}");
            let parsed = &result[&key];
            assert!(
                matches!(&parsed.value, OnnxAttributeValue::Int(v) if *v == i as i64),
                "expected Int({i}) for attr_{i}"
            );
        }
    }
}
