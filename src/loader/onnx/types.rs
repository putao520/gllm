use super::{proto, LoaderError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum OnnxType {
    Tensor(OnnxTensorType),
    SparseTensor(OnnxTensorType),
    Sequence(Box<OnnxType>),
    Map(OnnxMapType),
    Optional(Box<OnnxType>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnnxTensorType {
    pub elem_type: proto::tensor_proto::DataType,
    pub shape: OnnxTensorShape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnnxTensorShape {
    pub dims: Vec<OnnxDim>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OnnxDim {
    Known(i64),
    Param(String),
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnnxMapType {
    pub key_type: proto::tensor_proto::DataType,
    pub value_type: Box<OnnxType>,
}

impl OnnxType {
    pub fn from_proto(proto: proto::TypeProto) -> Result<Self> {
        let value = proto
            .value
            .ok_or_else(|| LoaderError::Onnx("unsupported or empty TypeProto".to_string()))?;
        use proto::type_proto::Value;
        match value {
            Value::TensorType(tensor) => {
                Ok(OnnxType::Tensor(OnnxTensorType::from_tensor_proto(tensor)?))
            }
            Value::SparseTensorType(sparse) => Ok(OnnxType::SparseTensor(
                OnnxTensorType::from_sparse_proto(sparse)?,
            )),
            Value::SequenceType(sequence) => {
                let elem = sequence.elem_type.ok_or_else(|| {
                    LoaderError::Onnx("sequence type missing elem_type".to_string())
                })?;
                Ok(OnnxType::Sequence(Box::new(OnnxType::from_proto(*elem)?)))
            }
            Value::MapType(map) => {
                let value = map
                    .value_type
                    .ok_or_else(|| LoaderError::Onnx("map type missing value_type".to_string()))?;
                let key_type = parse_data_type(
                    map.key_type.ok_or_else(|| {
                        LoaderError::Onnx("map type missing key_type".to_string())
                    })?,
                    "map",
                )?;
                Ok(OnnxType::Map(OnnxMapType {
                    key_type,
                    value_type: Box::new(OnnxType::from_proto(*value)?),
                }))
            }
            Value::OptionalType(optional) => {
                let elem = optional.elem_type.ok_or_else(|| {
                    LoaderError::Onnx("optional type missing elem_type".to_string())
                })?;
                Ok(OnnxType::Optional(Box::new(OnnxType::from_proto(*elem)?)))
            }
        }
    }
}

impl OnnxTensorType {
    fn from_tensor_proto(proto: proto::type_proto::Tensor) -> Result<Self> {
        let elem_type = parse_data_type(
            proto
                .elem_type
                .ok_or_else(|| LoaderError::Onnx("tensor type missing elem_type".to_string()))?,
            "tensor",
        )?;
        let shape = proto
            .shape
            .map(OnnxTensorShape::from_proto)
            .transpose()?
            .unwrap_or_else(|| OnnxTensorShape { dims: Vec::new() }); // LEGAL: shape 缺失时使用空维度
        Ok(Self { elem_type, shape })
    }

    fn from_sparse_proto(proto: proto::type_proto::SparseTensor) -> Result<Self> {
        let elem_type = parse_data_type(
            proto.elem_type.ok_or_else(|| {
                LoaderError::Onnx("sparse tensor type missing elem_type".to_string())
            })?,
            "sparse_tensor",
        )?;
        let shape = proto
            .shape
            .map(OnnxTensorShape::from_proto)
            .transpose()?
            .unwrap_or_else(|| OnnxTensorShape { dims: Vec::new() }); // LEGAL: shape 缺失时使用空维度
        Ok(Self { elem_type, shape })
    }
}

impl OnnxTensorShape {
    fn from_proto(proto: proto::TensorShapeProto) -> Result<Self> {
        let mut dims = Vec::with_capacity(proto.dim.len());
        for dim in proto.dim {
            dims.push(OnnxDim::from_proto(dim)?);
        }
        Ok(Self { dims })
    }
}

impl OnnxDim {
    fn from_proto(proto: proto::tensor_shape_proto::Dimension) -> Result<Self> {
        let Some(value) = proto.value else {
            return Ok(OnnxDim::Unknown);
        };
        match value {
            proto::tensor_shape_proto::dimension::Value::DimValue(value) => {
                Ok(OnnxDim::Known(value))
            }
            proto::tensor_shape_proto::dimension::Value::DimParam(param) => {
                if param.is_empty() {
                    Ok(OnnxDim::Unknown)
                } else {
                    Ok(OnnxDim::Param(param))
                }
            }
        }
    }
}

fn parse_data_type(value: i32, context: &str) -> Result<proto::tensor_proto::DataType> {
    proto::tensor_proto::DataType::try_from(value)
        .map_err(|_| LoaderError::Onnx(format!("unsupported data_type {value} in {context}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_dim_known() {
        let dim = OnnxDim::Known(128);
        assert!(matches!(dim, OnnxDim::Known(128)));
    }

    #[test]
    fn onnx_dim_param() {
        let dim = OnnxDim::Param("batch_size".to_string());
        assert!(matches!(dim, OnnxDim::Param(p) if p == "batch_size"));
    }

    #[test]
    fn onnx_dim_unknown() {
        let dim = OnnxDim::Unknown;
        assert!(matches!(dim, OnnxDim::Unknown));
    }

    #[test]
    fn onnx_tensor_shape_empty() {
        let shape = OnnxTensorShape { dims: vec![] };
        assert!(shape.dims.is_empty());
    }

    #[test]
    fn onnx_tensor_shape_mixed_dims() {
        let shape = OnnxTensorShape {
            dims: vec![
                OnnxDim::Param("batch".to_string()),
                OnnxDim::Known(256),
                OnnxDim::Unknown,
            ],
        };
        assert_eq!(shape.dims.len(), 3);
        assert!(matches!(&shape.dims[0], OnnxDim::Param(p) if p == "batch"));
        assert!(matches!(&shape.dims[1], OnnxDim::Known(256)));
        assert!(matches!(&shape.dims[2], OnnxDim::Unknown));
    }

    #[test]
    fn onnx_type_tensor_variant() {
        let ty = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(3), OnnxDim::Known(4)] },
        });
        assert!(matches!(ty, OnnxType::Tensor(_)));
    }

    #[test]
    fn onnx_type_sparse_tensor() {
        let ty = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int64,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        assert!(matches!(ty, OnnxType::SparseTensor(_)));
    }

    #[test]
    fn onnx_type_sequence() {
        let inner = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let ty = OnnxType::Sequence(Box::new(inner));
        assert!(matches!(ty, OnnxType::Sequence(_)));
    }

    #[test]
    fn onnx_type_map() {
        let val_ty = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::String,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let ty = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(val_ty),
        });
        assert!(matches!(ty, OnnxType::Map(_)));
    }

    #[test]
    fn onnx_type_optional() {
        let inner = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Double,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(1)] },
        });
        let ty = OnnxType::Optional(Box::new(inner));
        assert!(matches!(ty, OnnxType::Optional(_)));
    }

    #[test]
    fn onnx_map_type_fields() {
        let map = OnnxMapType {
            key_type: proto::tensor_proto::DataType::String,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Int32,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        };
        assert!(matches!(map.key_type, proto::tensor_proto::DataType::String));
    }

    #[test]
    fn onnx_tensor_type_clone() {
        let tt = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float16,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(768)] },
        };
        let cloned = tt.clone();
        assert!(matches!(cloned.elem_type, proto::tensor_proto::DataType::Float16));
    }

    #[test]
    fn onnx_dim_from_proto_known() {
        let proto_dim = proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(64)),
            denotation: None,
        };
        let dim = OnnxDim::from_proto(proto_dim).unwrap();
        assert!(matches!(dim, OnnxDim::Known(64)));
    }

    #[test]
    fn onnx_dim_from_proto_param() {
        let proto_dim = proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam("seq_len".to_string())),
            denotation: None,
        };
        let dim = OnnxDim::from_proto(proto_dim).unwrap();
        assert!(matches!(dim, OnnxDim::Param(p) if p == "seq_len"));
    }

    #[test]
    fn onnx_dim_from_proto_empty_param_is_unknown() {
        let proto_dim = proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam(String::new())),
            denotation: None,
        };
        let dim = OnnxDim::from_proto(proto_dim).unwrap();
        assert!(matches!(dim, OnnxDim::Unknown));
    }

    #[test]
    fn onnx_dim_from_proto_no_value_is_unknown() {
        let proto_dim = proto::tensor_shape_proto::Dimension { value: None, denotation: None };
        let dim = OnnxDim::from_proto(proto_dim).unwrap();
        assert!(matches!(dim, OnnxDim::Unknown));
    }

    #[test]
    fn parse_data_type_valid() {
        let dt = parse_data_type(1, "test").unwrap();
        assert!(matches!(dt, proto::tensor_proto::DataType::Float));
    }

    #[test]
    fn parse_data_type_invalid() {
        let result = parse_data_type(9999, "test_context");
        assert!(result.is_err());
    }

    // --- Trait tests ---

    #[test]
    fn onnx_dim_debug_format_known() {
        let dim = OnnxDim::Known(42);
        let debug_str = format!("{dim:?}");
        assert!(debug_str.contains("Known"), "Debug output should contain 'Known'");
    }

    #[test]
    fn onnx_dim_debug_format_param() {
        let dim = OnnxDim::Param("N".to_string());
        let debug_str = format!("{dim:?}");
        assert!(debug_str.contains("Param"), "Debug output should contain 'Param'");
    }

    #[test]
    fn onnx_dim_debug_format_unknown() {
        let dim = OnnxDim::Unknown;
        let debug_str = format!("{dim:?}");
        assert!(debug_str.contains("Unknown"), "Debug output should contain 'Unknown'");
    }

    #[test]
    fn onnx_dim_clone_equals() {
        let dim = OnnxDim::Known(100);
        let cloned = dim.clone();
        assert_eq!(dim, cloned);
    }

    #[test]
    fn onnx_dim_hash_equal_values_match() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(OnnxDim::Known(7)));
        assert!(!set.insert(OnnxDim::Known(7)));
        assert!(set.insert(OnnxDim::Known(8)));
        assert!(set.insert(OnnxDim::Unknown));
        assert!(!set.insert(OnnxDim::Unknown));
        assert!(set.insert(OnnxDim::Param("batch".to_string())));
        assert!(!set.insert(OnnxDim::Param("batch".to_string())));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn onnx_dim_eq_reflexive() {
        let dim = OnnxDim::Known(0);
        assert_eq!(dim, dim);
    }

    #[test]
    fn onnx_dim_not_equal_different_variants() {
        assert_ne!(OnnxDim::Known(1), OnnxDim::Unknown);
        assert_ne!(OnnxDim::Known(1), OnnxDim::Param("1".to_string()));
        assert_ne!(OnnxDim::Unknown, OnnxDim::Param("x".to_string()));
    }

    #[test]
    fn onnx_tensor_shape_clone() {
        let shape = OnnxTensorShape {
            dims: vec![OnnxDim::Known(3), OnnxDim::Param("N".to_string())],
        };
        let cloned = shape.clone();
        assert_eq!(shape, cloned);
    }

    #[test]
    fn onnx_tensor_shape_debug() {
        let shape = OnnxTensorShape {
            dims: vec![OnnxDim::Known(1)],
        };
        let debug_str = format!("{shape:?}");
        assert!(debug_str.contains("dims"));
    }

    #[test]
    fn onnx_tensor_type_clone_equals() {
        let tt = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Known(10), OnnxDim::Known(20)],
            },
        };
        let cloned = tt.clone();
        assert_eq!(tt, cloned);
    }

    #[test]
    fn onnx_type_clone_all_variants() {
        let tensor_ty = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        assert_eq!(tensor_ty, tensor_ty.clone());

        let sparse_ty = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int64,
            shape: OnnxTensorShape { dims: vec![] },
        });
        assert_eq!(sparse_ty, sparse_ty.clone());

        let seq_ty = OnnxType::Sequence(Box::new(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        })));
        assert_eq!(seq_ty, seq_ty.clone());

        let map_ty = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        });
        assert_eq!(map_ty, map_ty.clone());

        let opt_ty = OnnxType::Optional(Box::new(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        })));
        assert_eq!(opt_ty, opt_ty.clone());
    }

    #[test]
    fn onnx_type_not_equal_different_variants() {
        let tensor = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let sparse = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        assert_ne!(tensor, sparse);
    }

    #[test]
    fn onnx_map_type_clone() {
        let map = OnnxMapType {
            key_type: proto::tensor_proto::DataType::String,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Int32,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        };
        assert_eq!(map.key_type, map.clone().key_type);
    }

    // --- Boundary value tests ---

    #[test]
    fn onnx_dim_known_zero() {
        let dim = OnnxDim::Known(0);
        assert!(matches!(dim, OnnxDim::Known(0)));
    }

    #[test]
    fn onnx_dim_known_max() {
        let dim = OnnxDim::Known(i64::MAX);
        assert!(matches!(dim, OnnxDim::Known(v) if v == i64::MAX));
    }

    #[test]
    fn onnx_dim_known_negative() {
        let dim = OnnxDim::Known(-1);
        assert!(matches!(dim, OnnxDim::Known(-1)));
    }

    #[test]
    fn onnx_dim_from_proto_known_zero() {
        let proto_dim = proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(0)),
            denotation: None,
        };
        let dim = OnnxDim::from_proto(proto_dim).unwrap();
        assert!(matches!(dim, OnnxDim::Known(0)));
    }

    #[test]
    fn onnx_dim_from_proto_known_max() {
        let proto_dim = proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(i64::MAX)),
            denotation: None,
        };
        let dim = OnnxDim::from_proto(proto_dim).unwrap();
        assert!(matches!(dim, OnnxDim::Known(v) if v == i64::MAX));
    }

    #[test]
    fn onnx_dim_from_proto_known_negative() {
        let proto_dim = proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(-99)),
            denotation: None,
        };
        let dim = OnnxDim::from_proto(proto_dim).unwrap();
        assert!(matches!(dim, OnnxDim::Known(-99)));
    }

    #[test]
    fn onnx_tensor_shape_single_dim() {
        let shape = OnnxTensorShape {
            dims: vec![OnnxDim::Known(1)],
        };
        assert_eq!(shape.dims.len(), 1);
    }

    // --- Error path tests ---

    #[test]
    fn from_proto_empty_type_proto_is_error() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: None,
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_tensor_missing_elem_type_is_error() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: None,
                shape: None,
            })),
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_tensor_invalid_elem_type_is_error() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(9999),
                shape: None,
            })),
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_sparse_missing_elem_type_is_error() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SparseTensorType(
                proto::type_proto::SparseTensor {
                    elem_type: None,
                    shape: None,
                },
            )),
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_sparse_invalid_elem_type_is_error() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SparseTensorType(
                proto::type_proto::SparseTensor {
                    elem_type: Some(-1),
                    shape: None,
                },
            )),
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_sequence_missing_elem_type_is_error() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SequenceType(Box::new(
                proto::type_proto::Sequence { elem_type: None },
            ))),
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_map_missing_key_type_is_error() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::MapType(Box::new(
                proto::type_proto::Map {
                    key_type: None,
                    value_type: None,
                },
            ))),
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_map_missing_value_type_is_error() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::MapType(Box::new(
                proto::type_proto::Map {
                    key_type: Some(7), // Int64
                    value_type: None,
                },
            ))),
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_map_invalid_key_type_is_error() {
        let inner_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1),
                shape: None,
            })),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::MapType(Box::new(
                proto::type_proto::Map {
                    key_type: Some(9999),
                    value_type: Some(Box::new(inner_type)),
                },
            ))),
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_optional_missing_elem_type_is_error() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::OptionalType(Box::new(
                proto::type_proto::Optional { elem_type: None },
            ))),
        };
        let result = OnnxType::from_proto(type_proto);
        assert!(result.is_err());
    }

    // --- from_proto success path tests ---

    #[test]
    fn from_proto_tensor_with_shape() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1), // Float
                shape: Some(proto::TensorShapeProto {
                    dim: vec![
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(3)),
                            denotation: None,
                        },
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(4)),
                            denotation: None,
                        },
                    ],
                }),
            })),
        };
        let ty = OnnxType::from_proto(type_proto).unwrap();
        match ty {
            OnnxType::Tensor(tt) => {
                assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Float);
                assert_eq!(tt.shape.dims.len(), 2);
                assert!(matches!(&tt.shape.dims[0], OnnxDim::Known(3)));
                assert!(matches!(&tt.shape.dims[1], OnnxDim::Known(4)));
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_tensor_without_shape_is_empty_dims() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1), // Float
                shape: None,
            })),
        };
        let ty = OnnxType::from_proto(type_proto).unwrap();
        match ty {
            OnnxType::Tensor(tt) => {
                assert!(tt.shape.dims.is_empty());
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_sparse_with_shape() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SparseTensorType(
                proto::type_proto::SparseTensor {
                    elem_type: Some(7), // Int64
                    shape: Some(proto::TensorShapeProto {
                        dim: vec![proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(10)),
                            denotation: None,
                        }],
                    }),
                },
            )),
        };
        let ty = OnnxType::from_proto(type_proto).unwrap();
        match ty {
            OnnxType::SparseTensor(tt) => {
                assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Int64);
                assert_eq!(tt.shape.dims.len(), 1);
            }
            other => panic!("expected SparseTensor, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_sequence_of_tensor() {
        let inner_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1),
                shape: None,
            })),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SequenceType(Box::new(
                proto::type_proto::Sequence {
                    elem_type: Some(Box::new(inner_type)),
                },
            ))),
        };
        let ty = OnnxType::from_proto(type_proto).unwrap();
        match ty {
            OnnxType::Sequence(inner) => {
                assert!(matches!(*inner, OnnxType::Tensor(_)));
            }
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_map_valid() {
        let value_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1), // Float
                shape: None,
            })),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::MapType(Box::new(
                proto::type_proto::Map {
                    key_type: Some(8), // String
                    value_type: Some(Box::new(value_type)),
                },
            ))),
        };
        let ty = OnnxType::from_proto(type_proto).unwrap();
        match ty {
            OnnxType::Map(map) => {
                assert_eq!(map.key_type, proto::tensor_proto::DataType::String);
                assert!(matches!(*map.value_type, OnnxType::Tensor(_)));
            }
            other => panic!("expected Map, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_optional_valid() {
        let inner_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(11), // Double
                shape: None,
            })),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::OptionalType(Box::new(
                proto::type_proto::Optional {
                    elem_type: Some(Box::new(inner_type)),
                },
            ))),
        };
        let ty = OnnxType::from_proto(type_proto).unwrap();
        match ty {
            OnnxType::Optional(inner) => {
                assert!(matches!(*inner, OnnxType::Tensor(_)));
            }
            other => panic!("expected Optional, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_tensor_shape_with_mixed_dims() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1),
                shape: Some(proto::TensorShapeProto {
                    dim: vec![
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
                            denotation: None,
                        },
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(256)),
                            denotation: None,
                        },
                        proto::tensor_shape_proto::Dimension { value: None, denotation: None },
                    ],
                }),
            })),
        };
        let ty = OnnxType::from_proto(type_proto).unwrap();
        match ty {
            OnnxType::Tensor(tt) => {
                assert_eq!(tt.shape.dims.len(), 3);
                assert!(matches!(&tt.shape.dims[0], OnnxDim::Param(p) if p == "batch"));
                assert!(matches!(&tt.shape.dims[1], OnnxDim::Known(256)));
                assert!(matches!(&tt.shape.dims[2], OnnxDim::Unknown));
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_tensor_empty_shape() {
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1),
                shape: Some(proto::TensorShapeProto { dim: vec![] }),
            })),
        };
        let ty = OnnxType::from_proto(type_proto).unwrap();
        match ty {
            OnnxType::Tensor(tt) => {
                assert!(tt.shape.dims.is_empty());
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    // --- parse_data_type edge cases ---

    #[test]
    fn parse_data_type_zero_is_undefined() {
        let dt = parse_data_type(0, "test").unwrap();
        assert!(matches!(dt, proto::tensor_proto::DataType::Undefined));
    }

    #[test]
    fn parse_data_type_negative_is_error() {
        let result = parse_data_type(-1, "test");
        assert!(result.is_err());
    }

    #[test]
    fn parse_data_type_max_valid() {
        // Int2 = 26 is the last valid variant
        let dt = parse_data_type(26, "test").unwrap();
        assert!(matches!(dt, proto::tensor_proto::DataType::Int2));
    }

    #[test]
    fn parse_data_type_just_past_valid_is_error() {
        let result = parse_data_type(27, "test");
        assert!(result.is_err());
    }

    #[test]
    fn parse_data_type_i32_max_is_error() {
        let result = parse_data_type(i32::MAX, "test");
        assert!(result.is_err());
    }

    #[test]
    fn parse_data_type_i32_min_is_error() {
        let result = parse_data_type(i32::MIN, "test");
        assert!(result.is_err());
    }

    // --- OnnxTensorShape from_proto ---

    #[test]
    fn onnx_tensor_shape_from_proto_empty() {
        let proto_shape = proto::TensorShapeProto { dim: vec![] };
        let shape = OnnxTensorShape::from_proto(proto_shape).unwrap();
        assert!(shape.dims.is_empty());
    }

    #[test]
    fn onnx_tensor_shape_from_proto_multiple_dims() {
        let proto_shape = proto::TensorShapeProto {
            dim: vec![
                proto::tensor_shape_proto::Dimension {
                    value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(2)),
                    denotation: None,
                },
                proto::tensor_shape_proto::Dimension {
                    value: Some(proto::tensor_shape_proto::dimension::Value::DimParam("W".to_string())),
                    denotation: None,
                },
                proto::tensor_shape_proto::Dimension { value: None, denotation: None },
            ],
        };
        let shape = OnnxTensorShape::from_proto(proto_shape).unwrap();
        assert_eq!(shape.dims.len(), 3);
        assert!(matches!(shape.dims[0], OnnxDim::Known(2)));
        assert!(matches!(&shape.dims[1], OnnxDim::Param(p) if p == "W"));
        assert!(matches!(shape.dims[2], OnnxDim::Unknown));
    }

    // --- Nested type tests ---

    #[test]
    fn onnx_type_nested_sequence() {
        // Sequence<Sequence<Tensor<Float>>>
        let inner_tensor = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let inner_seq = OnnxType::Sequence(Box::new(inner_tensor));
        let outer_seq = OnnxType::Sequence(Box::new(inner_seq));
        match outer_seq {
            OnnxType::Sequence(outer) => match *outer {
                OnnxType::Sequence(inner) => {
                    assert!(matches!(*inner, OnnxType::Tensor(_)));
                }
                other => panic!("expected inner Sequence, got {other:?}"),
            },
            other => panic!("expected outer Sequence, got {other:?}"),
        }
    }

    #[test]
    fn onnx_type_optional_of_map() {
        // Optional<Map<String, Tensor<Float>>>
        let map = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::String,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        });
        let opt = OnnxType::Optional(Box::new(map));
        match opt {
            OnnxType::Optional(inner) => {
                assert!(matches!(*inner, OnnxType::Map(_)));
            }
            other => panic!("expected Optional, got {other:?}"),
        }
    }

    #[test]
    fn onnx_tensor_type_field_access() {
        let tt = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Bfloat16,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Known(1), OnnxDim::Known(128), OnnxDim::Known(768)],
            },
        };
        assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Bfloat16);
        assert_eq!(tt.shape.dims.len(), 3);
        assert!(matches!(&tt.shape.dims[2], OnnxDim::Known(768)));
    }

    #[test]
    fn onnx_map_type_field_access() {
        let val_ty = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Double,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(5)] },
        });
        let map = OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int32,
            value_type: Box::new(val_ty),
        };
        assert_eq!(map.key_type, proto::tensor_proto::DataType::Int32);
        match &*map.value_type {
            OnnxType::Tensor(tt) => {
                assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Double);
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn onnx_dim_param_equality_same_string() {
        let a = OnnxDim::Param("seq_len".to_string());
        let b = OnnxDim::Param("seq_len".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn onnx_dim_param_inequality_different_string() {
        let a = OnnxDim::Param("seq_len".to_string());
        let b = OnnxDim::Param("batch".to_string());
        assert_ne!(a, b);
    }

    // --- Additional tests (15 new) ---

    #[test]
    fn parse_data_type_bool() {
        // Arrange: ONNX Bool data type is value 9
        // Act
        let dt = parse_data_type(9, "test").unwrap();
        // Assert
        assert!(matches!(dt, proto::tensor_proto::DataType::Bool));
    }

    #[test]
    fn parse_data_type_float16() {
        // Arrange: ONNX Float16 data type is value 10
        // Act
        let dt = parse_data_type(10, "test").unwrap();
        // Assert
        assert!(matches!(dt, proto::tensor_proto::DataType::Float16));
    }

    #[test]
    fn parse_data_type_int8() {
        // Arrange: ONNX Int8 data type is value 3
        // Act
        let dt = parse_data_type(3, "test").unwrap();
        // Assert
        assert!(matches!(dt, proto::tensor_proto::DataType::Int8));
    }

    #[test]
    fn parse_data_type_uint8() {
        // Arrange: ONNX Uint8 data type is value 2
        // Act
        let dt = parse_data_type(2, "test").unwrap();
        // Assert
        assert!(matches!(dt, proto::tensor_proto::DataType::Uint8));
    }

    #[test]
    fn parse_data_type_error_contains_context() {
        // Arrange: invalid data type with a specific context string
        // Act
        let err = parse_data_type(9999, "my_custom_context");
        // Assert
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("my_custom_context"),
            "error message should contain the context string, got: {msg}"
        );
    }

    #[test]
    fn onnx_dim_known_min_value() {
        // Arrange: i64::MIN boundary for OnnxDim::Known
        // Act
        let dim = OnnxDim::Known(i64::MIN);
        // Assert
        assert!(matches!(dim, OnnxDim::Known(v) if v == i64::MIN));
    }

    #[test]
    fn onnx_dim_from_proto_known_min_value() {
        // Arrange: proto dimension with i64::MIN value
        let proto_dim = proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(i64::MIN)),
            denotation: None,
        };
        // Act
        let dim = OnnxDim::from_proto(proto_dim).unwrap();
        // Assert
        assert!(matches!(dim, OnnxDim::Known(v) if v == i64::MIN));
    }

    #[test]
    fn onnx_tensor_shape_from_proto_single_dim() {
        // Arrange: proto shape with exactly one dimension
        let proto_shape = proto::TensorShapeProto {
            dim: vec![proto::tensor_shape_proto::Dimension {
                value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(512)),
                denotation: None,
            }],
        };
        // Act
        let shape = OnnxTensorShape::from_proto(proto_shape).unwrap();
        // Assert
        assert_eq!(shape.dims.len(), 1);
        assert!(matches!(shape.dims[0], OnnxDim::Known(512)));
    }

    #[test]
    fn onnx_tensor_type_partial_eq_different_elem_types() {
        // Arrange: two OnnxTensorType with same shape but different elem_type
        let a = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        };
        let b = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Double,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn onnx_tensor_type_partial_eq_different_shapes() {
        // Arrange: two OnnxTensorType with same elem_type but different shape
        let a = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        };
        let b = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(20)] },
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn onnx_tensor_shape_partial_eq_different_lengths() {
        // Arrange: two shapes with different numbers of dimensions
        let a = OnnxTensorShape { dims: vec![OnnxDim::Known(3), OnnxDim::Known(4)] };
        let b = OnnxTensorShape { dims: vec![OnnxDim::Known(3)] };
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn onnx_type_partial_eq_tensor_vs_sparse_same_contents() {
        // Arrange: Tensor and SparseTensor with identical inner contents
        let tensor = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(5)] },
        });
        let sparse = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(5)] },
        });
        // Act & Assert: different outer variants are never equal
        assert_ne!(tensor, sparse);
    }

    #[test]
    fn from_proto_sparse_without_shape_is_empty_dims() {
        // Arrange: sparse tensor proto with no shape field
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SparseTensorType(
                proto::type_proto::SparseTensor {
                    elem_type: Some(1), // Float
                    shape: None,
                },
            )),
        };
        // Act
        let ty = OnnxType::from_proto(type_proto).unwrap();
        // Assert
        match ty {
            OnnxType::SparseTensor(tt) => {
                assert!(tt.shape.dims.is_empty());
            }
            other => panic!("expected SparseTensor, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_sequence_with_invalid_inner_elem_is_error() {
        // Arrange: sequence wrapping a tensor with invalid elem_type
        let inner_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(-5), // invalid
                shape: None,
            })),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SequenceType(Box::new(
                proto::type_proto::Sequence {
                    elem_type: Some(Box::new(inner_type)),
                },
            ))),
        };
        // Act
        let result = OnnxType::from_proto(type_proto);
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn from_proto_optional_of_tensor_with_param_dims() {
        // Arrange: optional wrapping a tensor with symbolic param dims
        let inner_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1), // Float
                shape: Some(proto::TensorShapeProto {
                    dim: vec![
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam(
                                "batch".to_string(),
                            )),
                            denotation: None,
                        },
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam(
                                "height".to_string(),
                            )),
                            denotation: None,
                        },
                    ],
                }),
            })),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::OptionalType(Box::new(
                proto::type_proto::Optional {
                    elem_type: Some(Box::new(inner_type)),
                },
            ))),
        };
        // Act
        let ty = OnnxType::from_proto(type_proto).unwrap();
        // Assert
        match ty {
            OnnxType::Optional(inner) => match *inner {
                OnnxType::Tensor(tt) => {
                    assert_eq!(tt.shape.dims.len(), 2);
                    assert!(matches!(&tt.shape.dims[0], OnnxDim::Param(p) if p == "batch"));
                    assert!(matches!(&tt.shape.dims[1], OnnxDim::Param(p) if p == "height"));
                }
                other => panic!("expected inner Tensor, got {other:?}"),
            },
            other => panic!("expected Optional, got {other:?}"),
        }
    }

    // --- Additional gap-coverage tests (13 new) ---

    #[test]
    fn from_proto_tensor_with_denotation_field_is_ignored() {
        // Arrange: TypeProto with a non-empty denotation string — should be ignored, not error
        let type_proto = proto::TypeProto {
            denotation: Some("data".to_string()),
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1), // Float
                shape: Some(proto::TensorShapeProto {
                    dim: vec![proto::tensor_shape_proto::Dimension {
                        value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(3)),
                        denotation: Some("batch_dim".to_string()),
                    }],
                }),
            })),
        };
        // Act
        let ty = OnnxType::from_proto(type_proto).unwrap();
        // Assert: denotation is silently ignored, tensor still parsed correctly
        match ty {
            OnnxType::Tensor(tt) => {
                assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Float);
                assert_eq!(tt.shape.dims.len(), 1);
                assert!(matches!(&tt.shape.dims[0], OnnxDim::Known(3)));
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_optional_with_invalid_inner_elem_is_error() {
        // Arrange: Optional wrapping a tensor with an invalid elem_type
        let inner_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(9999),
                shape: None,
            })),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::OptionalType(Box::new(
                proto::type_proto::Optional {
                    elem_type: Some(Box::new(inner_type)),
                },
            ))),
        };
        // Act
        let result = OnnxType::from_proto(type_proto);
        // Assert
        assert!(result.is_err(), "optional with invalid inner elem_type should fail");
    }

    #[test]
    fn from_proto_map_with_invalid_value_tensor_elem_is_error() {
        // Arrange: Map with valid key_type but value_type has invalid elem_type
        let inner_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(-10),
                shape: None,
            })),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::MapType(Box::new(
                proto::type_proto::Map {
                    key_type: Some(7), // Int64 — valid key
                    value_type: Some(Box::new(inner_type)),
                },
            ))),
        };
        // Act
        let result = OnnxType::from_proto(type_proto);
        // Assert
        assert!(result.is_err(), "map with invalid value tensor elem_type should fail");
    }

    #[test]
    fn from_proto_sequence_of_sparse_tensor() {
        // Arrange: Sequence wrapping a SparseTensor
        let inner_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SparseTensorType(
                proto::type_proto::SparseTensor {
                    elem_type: Some(1), // Float
                    shape: Some(proto::TensorShapeProto {
                        dim: vec![proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(100)),
                            denotation: None,
                        }],
                    }),
                },
            )),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SequenceType(Box::new(
                proto::type_proto::Sequence {
                    elem_type: Some(Box::new(inner_type)),
                },
            ))),
        };
        // Act
        let ty = OnnxType::from_proto(type_proto).unwrap();
        // Assert
        match ty {
            OnnxType::Sequence(inner) => match *inner {
                OnnxType::SparseTensor(tt) => {
                    assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Float);
                    assert_eq!(tt.shape.dims.len(), 1);
                }
                other => panic!("expected inner SparseTensor, got {other:?}"),
            },
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_deeply_nested_optional_sequence_tensor() {
        // Arrange: Optional<Sequence<Tensor<Float>>>
        let tensor_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1),
                shape: None,
            })),
        };
        let seq_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SequenceType(Box::new(
                proto::type_proto::Sequence {
                    elem_type: Some(Box::new(tensor_type)),
                },
            ))),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::OptionalType(Box::new(
                proto::type_proto::Optional {
                    elem_type: Some(Box::new(seq_type)),
                },
            ))),
        };
        // Act
        let ty = OnnxType::from_proto(type_proto).unwrap();
        // Assert: unwrap Optional -> Sequence -> Tensor
        match ty {
            OnnxType::Optional(outer) => match *outer {
                OnnxType::Sequence(inner) => {
                    assert!(matches!(*inner, OnnxType::Tensor(_)));
                }
                other => panic!("expected inner Sequence, got {other:?}"),
            },
            other => panic!("expected Optional, got {other:?}"),
        }
    }

    #[test]
    fn from_proto_tensor_shape_all_unknown_dims() {
        // Arrange: tensor shape where every dimension has value: None
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1), // Float
                shape: Some(proto::TensorShapeProto {
                    dim: vec![
                        proto::tensor_shape_proto::Dimension { value: None, denotation: None },
                        proto::tensor_shape_proto::Dimension { value: None, denotation: None },
                        proto::tensor_shape_proto::Dimension { value: None, denotation: None },
                    ],
                }),
            })),
        };
        // Act
        let ty = OnnxType::from_proto(type_proto).unwrap();
        // Assert
        match ty {
            OnnxType::Tensor(tt) => {
                assert_eq!(tt.shape.dims.len(), 3);
                assert!(tt.shape.dims.iter().all(|d| matches!(d, OnnxDim::Unknown)));
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn onnx_type_debug_all_variants() {
        // Arrange: one of each OnnxType variant
        let tensor = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(1)] },
        });
        let sparse = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int32,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let seq = OnnxType::Sequence(Box::new(tensor.clone()));
        let map = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::String,
            value_type: Box::new(tensor.clone()),
        });
        let opt = OnnxType::Optional(Box::new(tensor.clone()));
        // Act & Assert: Debug output contains variant names
        assert!(format!("{tensor:?}").contains("Tensor"));
        assert!(format!("{sparse:?}").contains("SparseTensor"));
        assert!(format!("{seq:?}").contains("Sequence"));
        assert!(format!("{map:?}").contains("Map"));
        assert!(format!("{opt:?}").contains("Optional"));
    }

    #[test]
    fn onnx_map_type_debug_format() {
        // Arrange
        let map = OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        };
        // Act
        let debug_str = format!("{map:?}");
        // Assert: debug output contains field names
        assert!(debug_str.contains("key_type"), "OnnxMapType debug should contain 'key_type'");
        assert!(debug_str.contains("value_type"), "OnnxMapType debug should contain 'value_type'");
    }

    #[test]
    fn onnx_tensor_type_partial_eq_different_shape_lengths() {
        // Arrange: two OnnxTensorType with same elem_type but different shape dimension counts
        let a = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(3), OnnxDim::Known(4)] },
        };
        let b = OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Known(3), OnnxDim::Known(4), OnnxDim::Known(5)],
            },
        };
        // Act & Assert
        assert_ne!(a, b, "tensor types with different shape lengths should not be equal");
    }

    #[test]
    fn onnx_map_type_partial_eq_different_key_types() {
        // Arrange: two OnnxMapType with different key_type
        let a = OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        };
        let b = OnnxMapType {
            key_type: proto::tensor_proto::DataType::String,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        };
        // Act & Assert
        assert_ne!(a, b, "maps with different key types should not be equal");
    }

    #[test]
    fn onnx_map_type_partial_eq_different_value_types() {
        // Arrange: two OnnxMapType with same key but different value types
        let a = OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        };
        let b = OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Double,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        };
        // Act & Assert
        assert_ne!(a, b, "maps with different value types should not be equal");
    }

    #[test]
    fn from_proto_map_with_sequence_value_type() {
        // Arrange: Map<String, Sequence<Tensor<Float>>>
        let tensor_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1), // Float
                shape: None,
            })),
        };
        let seq_type = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SequenceType(Box::new(
                proto::type_proto::Sequence {
                    elem_type: Some(Box::new(tensor_type)),
                },
            ))),
        };
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::MapType(Box::new(
                proto::type_proto::Map {
                    key_type: Some(8), // String
                    value_type: Some(Box::new(seq_type)),
                },
            ))),
        };
        // Act
        let ty = OnnxType::from_proto(type_proto).unwrap();
        // Assert
        match ty {
            OnnxType::Map(map) => {
                assert_eq!(map.key_type, proto::tensor_proto::DataType::String);
                match &*map.value_type {
                    OnnxType::Sequence(inner) => {
                        assert!(matches!(**inner, OnnxType::Tensor(_)));
                    }
                    other => panic!("expected Sequence value, got {other:?}"),
                }
            }
            other => panic!("expected Map, got {other:?}"),
        }
    }

    #[test]
    fn parse_data_type_bfloat16() {
        // Arrange: ONNX Bfloat16 data type is value 16
        // Act
        let dt = parse_data_type(16, "test").unwrap();
        // Assert
        assert!(matches!(dt, proto::tensor_proto::DataType::Bfloat16));
    }

    // --- Additional gap-coverage tests (10 new) ---

    #[test]
    fn test_parse_data_type_int16() {
        // Arrange: ONNX Int16 data type is value 5
        // Act
        let dt = parse_data_type(5, "test").unwrap();
        // Assert
        assert!(matches!(dt, proto::tensor_proto::DataType::Int16));
    }

    #[test]
    fn test_parse_data_type_int32() {
        // Arrange: ONNX Int32 data type is value 6
        // Act
        let dt = parse_data_type(6, "test").unwrap();
        // Assert
        assert!(matches!(dt, proto::tensor_proto::DataType::Int32));
    }

    #[test]
    fn test_onnx_dim_param_unicode_string() {
        // Arrange: OnnxDim::Param with unicode characters
        // Act
        let dim = OnnxDim::Param("维度_αβγ".to_string());
        // Assert
        assert!(matches!(&dim, OnnxDim::Param(p) if p == "维度_αβγ"));
        let cloned = dim.clone();
        assert_eq!(dim, cloned);
    }

    #[test]
    fn test_onnx_map_type_eq_same_key_and_value() {
        // Arrange: two OnnxMapType with identical key and value types
        let a = OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        };
        let b = OnnxMapType {
            key_type: proto::tensor_proto::DataType::Int64,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        };
        // Act & Assert
        assert_eq!(a, b, "OnnxMapType with identical key and value types should be equal");
    }

    #[test]
    fn test_onnx_type_sequence_eq_same_inner() {
        // Arrange: two Sequence types wrapping identical inner Tensor types
        let inner_a = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        let inner_b = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        let seq_a = OnnxType::Sequence(Box::new(inner_a));
        let seq_b = OnnxType::Sequence(Box::new(inner_b));
        // Act & Assert
        assert_eq!(seq_a, seq_b, "Sequence types with same inner type should be equal");
    }

    #[test]
    fn test_onnx_type_sequence_neq_different_inner() {
        // Arrange: two Sequence types wrapping different inner types (Float vs Double)
        let inner_a = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let inner_b = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Double,
            shape: OnnxTensorShape { dims: vec![] },
        });
        let seq_a = OnnxType::Sequence(Box::new(inner_a));
        let seq_b = OnnxType::Sequence(Box::new(inner_b));
        // Act & Assert
        assert_ne!(seq_a, seq_b, "Sequence types with different inner types should not be equal");
    }

    #[test]
    fn test_from_proto_sparse_shape_with_mixed_dims() {
        // Arrange: sparse tensor with mixed dimension types (Known + Param + Unknown)
        let type_proto = proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::SparseTensorType(
                proto::type_proto::SparseTensor {
                    elem_type: Some(1), // Float
                    shape: Some(proto::TensorShapeProto {
                        dim: vec![
                            proto::tensor_shape_proto::Dimension {
                                value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(64)),
                                denotation: None,
                            },
                            proto::tensor_shape_proto::Dimension {
                                value: Some(proto::tensor_shape_proto::dimension::Value::DimParam(
                                    "features".to_string(),
                                )),
                                denotation: None,
                            },
                            proto::tensor_shape_proto::Dimension { value: None, denotation: None },
                        ],
                    }),
                },
            )),
        };
        // Act
        let ty = OnnxType::from_proto(type_proto).unwrap();
        // Assert
        match ty {
            OnnxType::SparseTensor(tt) => {
                assert_eq!(tt.shape.dims.len(), 3);
                assert!(matches!(&tt.shape.dims[0], OnnxDim::Known(64)));
                assert!(matches!(&tt.shape.dims[1], OnnxDim::Param(p) if p == "features"));
                assert!(matches!(&tt.shape.dims[2], OnnxDim::Unknown));
            }
            other => panic!("expected SparseTensor, got {other:?}"),
        }
    }

    #[test]
    fn test_from_proto_empty_type_proto_error_message_content() {
        // Arrange: TypeProto with no value field set
        let type_proto = proto::TypeProto {
            denotation: None,
            value: None,
        };
        // Act
        let result = OnnxType::from_proto(type_proto);
        // Assert: error message should describe the problem
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("unsupported") || err_msg.contains("empty"),
            "error message should mention the problem, got: {err_msg}"
        );
    }

    #[test]
    fn test_onnx_tensor_shape_many_dimensions() {
        // Arrange: shape with many dimensions to verify capacity handling
        let dims: Vec<OnnxDim> = (0..100).map(|i| OnnxDim::Known(i)).collect();
        let shape = OnnxTensorShape { dims };
        // Act & Assert
        assert_eq!(shape.dims.len(), 100);
        assert!(matches!(&shape.dims[0], OnnxDim::Known(0)));
        assert!(matches!(&shape.dims[99], OnnxDim::Known(99)));
        // Clone preserves all dims
        let cloned = shape.clone();
        assert_eq!(cloned.dims.len(), 100);
        assert_eq!(shape, cloned);
    }

    #[test]
    fn test_onnx_optional_eq_same_inner() {
        // Arrange: two Optional types wrapping identical inner Tensor types
        let inner_a = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int32,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(5)] },
        });
        let inner_b = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int32,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(5)] },
        });
        let opt_a = OnnxType::Optional(Box::new(inner_a));
        let opt_b = OnnxType::Optional(Box::new(inner_b));
        // Act & Assert
        assert_eq!(opt_a, opt_b, "Optional types with same inner type should be equal");
    }
}
