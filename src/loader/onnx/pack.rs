use prost::bytes::Bytes;
use safetensors::Dtype;

use super::{proto, LoaderError, Result};

pub(super) struct TensorPackInput<'a> {
    pub data_type: proto::tensor_proto::DataType,
    pub dtype: Dtype,
    pub element_count: usize,
    pub raw_data: Bytes,
    pub float_data: Vec<f32>,
    pub int32_data: Vec<i32>,
    pub int64_data: Vec<i64>,
    pub double_data: Vec<f64>,
    pub uint64_data: Vec<u64>,
    pub string_data: Vec<Vec<u8>>,
    pub name: &'a str,
}

pub(super) fn build_tensor_bytes(input: TensorPackInput<'_>) -> Result<Bytes> {
    let TensorPackInput {
        data_type,
        dtype,
        element_count,
        raw_data,
        float_data,
        int32_data,
        int64_data,
        double_data,
        uint64_data,
        string_data,
        name,
    } = input;

    if !raw_data.is_empty() {
        let expected = expected_byte_len(data_type, dtype, element_count, name)?;
        if raw_data.len() != expected {
            return Err(LoaderError::Onnx(format!(
                "tensor {name} raw_data has {} bytes, expected {expected}",
                raw_data.len()
            )));
        }
        return Ok(raw_data);
    }

    use proto::tensor_proto::DataType as OnnxType;
    match data_type {
        OnnxType::Float => pack_f32(float_data, element_count, name),
        OnnxType::Double => pack_f64(double_data, element_count, name),
        OnnxType::Int32 => pack_i32(int32_data, element_count, name),
        OnnxType::Int64 => pack_i64(int64_data, element_count, name),
        OnnxType::Uint64 => pack_u64(uint64_data, element_count, name),
        OnnxType::Uint32 => pack_u32_from_u64(uint64_data, element_count, name),
        OnnxType::Int8 => pack_i8_from_i32(int32_data, element_count, name),
        OnnxType::Uint8 => pack_u8_from_i32(int32_data, element_count, name),
        OnnxType::Int16 => pack_i16_from_i32(int32_data, element_count, name),
        OnnxType::Uint16 => pack_u16_from_i32(int32_data, element_count, name),
        OnnxType::Float16 => pack_f16_bits_from_i32(int32_data, element_count, name),
        OnnxType::Bfloat16 => pack_bf16_bits_from_i32(int32_data, element_count, name),
        OnnxType::Bool => pack_bool_from_i32(int32_data, element_count, name),
        // STRING: serialize as length-prefixed byte sequences
        OnnxType::String => pack_strings(string_data, element_count, name),
        _ => Err(LoaderError::Onnx(format!(
            "tensor {name} missing raw_data for unsupported type {:?}",
            data_type
        ))),
    }
}

/// Calculate expected byte length, handling packed types (INT4/UINT4)
fn expected_byte_len(
    data_type: proto::tensor_proto::DataType,
    dtype: Dtype,
    element_count: usize,
    name: &str,
) -> Result<usize> {
    use proto::tensor_proto::DataType as OnnxType;
    let size = match data_type {
        // INT4/UINT4: 2 elements packed per byte
        OnnxType::Int4 | OnnxType::Uint4 => (element_count + 1) / 2,
        // All other types use safetensors Dtype size
        _ => element_count
            .checked_mul(dtype.size())
            .ok_or_else(|| LoaderError::Onnx(format!("tensor byte size overflow for {name}")))?,
    };
    Ok(size)
}

fn ensure_len(name: &str, expected: usize, actual: usize, field: &str) -> Result<()> {
    if expected != actual {
        return Err(LoaderError::Onnx(format!(
            "tensor {name} has {actual} values in {field}, expected {expected}"
        )));
    }
    Ok(())
}

fn pack_f32(data: Vec<f32>, element_count: usize, name: &str) -> Result<Bytes> {
    ensure_len(name, element_count, data.len(), "float_data")?;
    let mut out = Vec::with_capacity(data.len() * 4);
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Bytes::from(out))
}

fn pack_f64(data: Vec<f64>, element_count: usize, name: &str) -> Result<Bytes> {
    ensure_len(name, element_count, data.len(), "double_data")?;
    let mut out = Vec::with_capacity(data.len() * 8);
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Bytes::from(out))
}

fn pack_i32(data: Vec<i32>, element_count: usize, name: &str) -> Result<Bytes> {
    ensure_len(name, element_count, data.len(), "int32_data")?;
    let mut out = Vec::with_capacity(data.len() * 4);
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Bytes::from(out))
}

fn pack_i64(data: Vec<i64>, element_count: usize, name: &str) -> Result<Bytes> {
    ensure_len(name, element_count, data.len(), "int64_data")?;
    let mut out = Vec::with_capacity(data.len() * 8);
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Bytes::from(out))
}

fn pack_u64(data: Vec<u64>, element_count: usize, name: &str) -> Result<Bytes> {
    ensure_len(name, element_count, data.len(), "uint64_data")?;
    let mut out = Vec::with_capacity(data.len() * 8);
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Bytes::from(out))
}

fn pack_u32_from_u64(data: Vec<u64>, element_count: usize, name: &str) -> Result<Bytes> {
    ensure_len(name, element_count, data.len(), "uint64_data")?;
    let mut out = Vec::with_capacity(data.len() * 4);
    for value in data {
        let value = u32::try_from(value)
            .map_err(|_| LoaderError::Onnx(format!("uint32 overflow in tensor {name}")))?;
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Bytes::from(out))
}

fn pack_i8_from_i32(data: Vec<i32>, element_count: usize, name: &str) -> Result<Bytes> {
    pack_i32_with(data, element_count, name, 1, |value, out| {
        let value = i8::try_from(value)
            .map_err(|_| LoaderError::Onnx(format!("int8 overflow in tensor {name}")))?;
        out.push(value as u8);
        Ok(())
    })
}

fn pack_u8_from_i32(data: Vec<i32>, element_count: usize, name: &str) -> Result<Bytes> {
    pack_i32_with(data, element_count, name, 1, |value, out| {
        let value = u8::try_from(value)
            .map_err(|_| LoaderError::Onnx(format!("uint8 overflow in tensor {name}")))?;
        out.push(value);
        Ok(())
    })
}

fn pack_i16_from_i32(data: Vec<i32>, element_count: usize, name: &str) -> Result<Bytes> {
    pack_i32_with(data, element_count, name, 2, |value, out| {
        let value = i16::try_from(value)
            .map_err(|_| LoaderError::Onnx(format!("int16 overflow in tensor {name}")))?;
        out.extend_from_slice(&value.to_le_bytes());
        Ok(())
    })
}

fn pack_u16_from_i32(data: Vec<i32>, element_count: usize, name: &str) -> Result<Bytes> {
    pack_i32_with(data, element_count, name, 2, |value, out| {
        let value = u16::try_from(value)
            .map_err(|_| LoaderError::Onnx(format!("uint16 overflow in tensor {name}")))?;
        out.extend_from_slice(&value.to_le_bytes());
        Ok(())
    })
}

fn pack_f16_bits_from_i32(data: Vec<i32>, element_count: usize, name: &str) -> Result<Bytes> {
    pack_i32_with(data, element_count, name, 2, |value, out| {
        let value = u16::try_from(value)
            .map_err(|_| LoaderError::Onnx(format!("float16 bits overflow in tensor {name}")))?;
        out.extend_from_slice(&value.to_le_bytes());
        Ok(())
    })
}

fn pack_bf16_bits_from_i32(data: Vec<i32>, element_count: usize, name: &str) -> Result<Bytes> {
    pack_i32_with(data, element_count, name, 2, |value, out| {
        let value = u16::try_from(value)
            .map_err(|_| LoaderError::Onnx(format!("bfloat16 bits overflow in tensor {name}")))?;
        out.extend_from_slice(&value.to_le_bytes());
        Ok(())
    })
}

fn pack_bool_from_i32(data: Vec<i32>, element_count: usize, name: &str) -> Result<Bytes> {
    pack_i32_with(data, element_count, name, 1, |value, out| {
        match value {
            0 => out.push(0),
            1 => out.push(1),
            _ => {
                return Err(LoaderError::Onnx(format!(
                    "invalid bool value {value} in tensor {name}"
                )))
            }
        }
        Ok(())
    })
}

/// Pack STRING data as length-prefixed byte sequences
///
/// Format: for each string, write 4-byte little-endian length followed by UTF-8 bytes
/// This allows reconstruction of individual strings from the raw data.
fn pack_strings(data: Vec<Vec<u8>>, element_count: usize, name: &str) -> Result<Bytes> {
    ensure_len(name, element_count, data.len(), "string_data")?;
    // Estimate capacity: 4 bytes per length + average string length
    let estimated_size: usize = data.iter().map(|s| 4 + s.len()).sum();
    let mut out = Vec::with_capacity(estimated_size);
    for s in data {
        // Write length as 4-byte little-endian
        let len = u32::try_from(s.len())
            .map_err(|_| LoaderError::Onnx(format!("string too long in tensor {name}")))?;
        out.extend_from_slice(&len.to_le_bytes());
        // Write UTF-8 bytes
        out.extend_from_slice(&s);
    }
    Ok(Bytes::from(out))
}

fn pack_i32_with<F>(
    data: Vec<i32>,
    element_count: usize,
    name: &str,
    bytes_per: usize,
    mut write: F,
) -> Result<Bytes>
where
    F: FnMut(i32, &mut Vec<u8>) -> Result<()>,
{
    ensure_len(name, element_count, data.len(), "int32_data")?;
    let mut out = Vec::with_capacity(data.len().saturating_mul(bytes_per));
    for value in data {
        write(value, &mut out)?;
    }
    Ok(Bytes::from(out))
}
