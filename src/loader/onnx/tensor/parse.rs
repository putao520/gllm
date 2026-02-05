use prost::bytes::Bytes;
use safetensors::Dtype;

use super::super::external::ExternalDataResolver;
use super::super::{proto, LoaderError, Result};

pub(super) fn parse_data_type(
    value: i32,
    name: &str,
) -> Result<proto::tensor_proto::DataType> {
    proto::tensor_proto::DataType::try_from(value).map_err(|_| {
        LoaderError::Onnx(format!("unsupported data_type {value} for tensor {name}"))
    })
}

pub(super) fn map_dtype(
    data_type: proto::tensor_proto::DataType,
    name: &str,
) -> Result<Dtype> {
    use proto::tensor_proto::DataType as OnnxType;

    match data_type {
        OnnxType::Float => Ok(Dtype::F32),
        OnnxType::Double => Ok(Dtype::F64),
        OnnxType::Float16 => Ok(Dtype::F16),
        OnnxType::Bfloat16 => Ok(Dtype::BF16),
        OnnxType::Int8 => Ok(Dtype::I8),
        OnnxType::Uint8 => Ok(Dtype::U8),
        OnnxType::Int16 => Ok(Dtype::I16),
        OnnxType::Uint16 => Ok(Dtype::U16),
        OnnxType::Int32 => Ok(Dtype::I32),
        OnnxType::Uint32 => Ok(Dtype::U32),
        OnnxType::Int64 => Ok(Dtype::I64),
        OnnxType::Uint64 => Ok(Dtype::U64),
        OnnxType::Bool => Ok(Dtype::U8),
        _ => Err(LoaderError::Onnx(format!(
            "unsupported data_type {:?} for tensor {name}",
            data_type
        ))),
    }
}

pub(super) fn parse_dims(dims: &[i64], name: &str) -> Result<Vec<usize>> {
    let mut shape = Vec::with_capacity(dims.len());
    for &dim in dims {
        if dim < 0 {
            return Err(LoaderError::Onnx(format!(
                "negative dimension {dim} in tensor {name}"
            )));
        }
        let dim = usize::try_from(dim).map_err(|_| {
            LoaderError::Onnx(format!("dimension overflow {dim} in tensor {name}"))
        })?;
        shape.push(dim);
    }
    Ok(shape)
}

pub(super) fn element_count(shape: &[usize], name: &str) -> Result<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut count: usize = 1;
    for &dim in shape {
        count = count.checked_mul(dim).ok_or_else(|| {
            LoaderError::Onnx(format!("tensor size overflow for {name}"))
        })?;
    }
    Ok(count)
}

pub(super) fn load_external_data(
    resolver: &mut ExternalDataResolver,
    external_data: &[proto::StringStringEntryProto],
    dtype: Dtype,
    element_count: usize,
    name: &str,
) -> Result<Bytes> {
    let entries = external_data_map(external_data);
    let location = entries.get("location").ok_or_else(|| {
        LoaderError::Onnx(format!("external tensor {name} missing location"))
    })?;
    let offset = parse_optional_usize(entries.get("offset"), "offset", name)?
        .unwrap_or(0);
    let length = parse_optional_usize(entries.get("length"), "length", name)?
        .unwrap_or_else(|| dtype.size() * element_count);
    let expected = dtype.size().checked_mul(element_count).ok_or_else(|| {
        LoaderError::Onnx(format!("tensor byte size overflow for {name}"))
    })?;
    if length != expected {
        return Err(LoaderError::Onnx(format!(
            "external tensor {name} length {length} does not match expected {expected}"
        )));
    }
    resolver.resolve(location, offset, length)
}

pub(super) fn slice_to_f32(bytes: &[u8]) -> Option<f32> {
    let mut array = [0u8; 4];
    array.copy_from_slice(bytes);
    Some(f32::from_le_bytes(array))
}

pub(super) fn slice_to_f16(bytes: &[u8]) -> Option<u16> {
    let mut array = [0u8; 2];
    array.copy_from_slice(bytes);
    Some(u16::from_le_bytes(array))
}

pub(super) fn slice_to_i32(bytes: &[u8]) -> Option<i32> {
    let mut array = [0u8; 4];
    array.copy_from_slice(bytes);
    Some(i32::from_le_bytes(array))
}

pub(super) fn slice_to_i64(bytes: &[u8]) -> Option<i64> {
    let mut array = [0u8; 8];
    array.copy_from_slice(bytes);
    Some(i64::from_le_bytes(array))
}

pub(super) fn slice_to_u16(bytes: &[u8]) -> Option<u16> {
    let mut array = [0u8; 2];
    array.copy_from_slice(bytes);
    Some(u16::from_le_bytes(array))
}

pub(super) fn slice_to_u32(bytes: &[u8]) -> Option<u32> {
    let mut array = [0u8; 4];
    array.copy_from_slice(bytes);
    Some(u32::from_le_bytes(array))
}

pub(super) fn slice_to_u64(bytes: &[u8]) -> Option<u64> {
    let mut array = [0u8; 8];
    array.copy_from_slice(bytes);
    Some(u64::from_le_bytes(array))
}

fn external_data_map(
    entries: &[proto::StringStringEntryProto],
) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for entry in entries {
        let Some(key) = entry.key.clone() else {
            continue;
        };
        if key.is_empty() {
            continue;
        }
        let value = entry.value.clone().unwrap_or_default();
        map.insert(key, value);
    }
    map
}

fn parse_optional_usize(
    value: Option<&String>,
    label: &str,
    name: &str,
) -> Result<Option<usize>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let parsed = value.parse::<usize>().map_err(|_| {
        LoaderError::Onnx(format!("invalid external {label} {value} for tensor {name}"))
    })?;
    Ok(Some(parsed))
}
