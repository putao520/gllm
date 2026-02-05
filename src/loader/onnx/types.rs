use super::{proto, LoaderError, Result};

#[derive(Debug, Clone)]
pub enum OnnxType {
    Tensor(OnnxTensorType),
    SparseTensor(OnnxTensorType),
    Sequence(Box<OnnxType>),
    Map(OnnxMapType),
    Optional(Box<OnnxType>),
}

#[derive(Debug, Clone)]
pub struct OnnxTensorType {
    pub elem_type: proto::tensor_proto::DataType,
    pub shape: OnnxTensorShape,
}

#[derive(Debug, Clone)]
pub struct OnnxTensorShape {
    pub dims: Vec<OnnxDim>,
}

#[derive(Debug, Clone)]
pub enum OnnxDim {
    Known(i64),
    Param(String),
    Unknown,
}

#[derive(Debug, Clone)]
pub struct OnnxMapType {
    pub key_type: proto::tensor_proto::DataType,
    pub value_type: Box<OnnxType>,
}

impl OnnxType {
    pub fn from_proto(proto: proto::TypeProto) -> Result<Self> {
        let value = proto.value.ok_or_else(|| {
            LoaderError::Onnx("unsupported or empty TypeProto".to_string())
        })?;
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
                let value = map.value_type.ok_or_else(|| {
                    LoaderError::Onnx("map type missing value_type".to_string())
                })?;
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
            proto.elem_type.ok_or_else(|| {
                LoaderError::Onnx("tensor type missing elem_type".to_string())
            })?,
            "tensor",
        )?;
        let shape = proto
            .shape
            .map(OnnxTensorShape::from_proto)
            .transpose()?
            .unwrap_or_else(|| OnnxTensorShape { dims: Vec::new() });
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
            .unwrap_or_else(|| OnnxTensorShape { dims: Vec::new() });
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

fn parse_data_type(
    value: i32,
    context: &str,
) -> Result<proto::tensor_proto::DataType> {
    proto::tensor_proto::DataType::try_from(value).map_err(|_| {
        LoaderError::Onnx(format!("unsupported data_type {value} in {context}"))
    })
}
