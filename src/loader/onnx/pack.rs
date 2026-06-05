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
        OnnxType::Int4 | OnnxType::Uint4 => element_count.div_ceil(2),
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

#[cfg(test)]
mod tests {
    use super::*;
    use proto::tensor_proto::DataType as OnnxType;

    fn make_input<'a>(
        data_type: OnnxType,
        element_count: usize,
        raw_data: Bytes,
        name: &'a str,
    ) -> TensorPackInput<'a> {
        TensorPackInput {
            data_type,
            dtype: Dtype::F32,
            element_count,
            raw_data,
            float_data: vec![],
            int32_data: vec![],
            int64_data: vec![],
            double_data: vec![],
            uint64_data: vec![],
            string_data: vec![],
            name,
        }
    }

    // ── build_tensor_bytes: raw_data passthrough ──────────────────────

    #[test]
    fn build_tensor_bytes_raw_data_passthrough() {
        let data: Vec<u8> = [1.0f32, 2.0f32].iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let raw = Bytes::from(data);
        let input = make_input(OnnxType::Float, 2, raw.clone(), "test");
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result, raw);
    }

    #[test]
    fn build_tensor_bytes_raw_data_wrong_len() {
        let raw = Bytes::from_static(&[0u8; 4]);
        let input = make_input(OnnxType::Float, 2, raw, "test");
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: packed float_data ─────────────────────────

    #[test]
    fn build_tensor_bytes_float() {
        let mut input = make_input(OnnxType::Float, 3, Bytes::new(), "f");
        input.float_data = vec![1.0, 2.0, 3.0];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 12);
        let v0 = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert!((v0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn build_tensor_bytes_float_wrong_len() {
        let mut input = make_input(OnnxType::Float, 3, Bytes::new(), "f");
        input.float_data = vec![1.0, 2.0];
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: packed int64_data ─────────────────────────

    #[test]
    fn build_tensor_bytes_int64() {
        let mut input = make_input(OnnxType::Int64, 2, Bytes::new(), "i64");
        input.int64_data = vec![100, -200];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 16);
        let v0 = i64::from_le_bytes(result[0..8].try_into().unwrap());
        assert_eq!(v0, 100);
    }

    // ── build_tensor_bytes: packed double_data ────────────────────────

    #[test]
    fn build_tensor_bytes_double() {
        let mut input = make_input(OnnxType::Double, 1, Bytes::new(), "d");
        input.double_data = vec![3.14];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 8);
    }

    // ── build_tensor_bytes: int8 from int32 ───────────────────────────

    #[test]
    fn build_tensor_bytes_int8() {
        let mut input = make_input(OnnxType::Int8, 3, Bytes::new(), "i8");
        input.int32_data = vec![-1, 0, 127];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 255);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 127);
    }

    // ── build_tensor_bytes: uint8 from int32 ──────────────────────────

    #[test]
    fn build_tensor_bytes_uint8() {
        let mut input = make_input(OnnxType::Uint8, 2, Bytes::new(), "u8");
        input.int32_data = vec![0, 255];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 255);
    }

    // ── build_tensor_bytes: bool from int32 ───────────────────────────

    #[test]
    fn build_tensor_bytes_bool() {
        let mut input = make_input(OnnxType::Bool, 3, Bytes::new(), "b");
        input.int32_data = vec![0, 1, 0];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.as_ref(), &[0, 1, 0]);
    }

    #[test]
    fn build_tensor_bytes_bool_invalid() {
        let mut input = make_input(OnnxType::Bool, 1, Bytes::new(), "b");
        input.int32_data = vec![2];
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: uint32 from uint64 ────────────────────────

    #[test]
    fn build_tensor_bytes_uint32() {
        let mut input = make_input(OnnxType::Uint32, 2, Bytes::new(), "u32");
        input.uint64_data = vec![0, 1000];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn build_tensor_bytes_uint32_overflow() {
        let mut input = make_input(OnnxType::Uint32, 1, Bytes::new(), "u32");
        input.uint64_data = vec![u32::MAX as u64 + 1];
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: STRING ────────────────────────────────────

    #[test]
    fn build_tensor_bytes_string() {
        let mut input = make_input(OnnxType::String, 2, Bytes::new(), "s");
        input.string_data = vec![b"hello".to_vec(), b"world".to_vec()];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4 + 5 + 4 + 5);
    }

    // ── expected_byte_len ─────────────────────────────────────────────

    #[test]
    fn expected_byte_len_float() {
        let len = expected_byte_len(OnnxType::Float, Dtype::F32, 4, "t").unwrap();
        assert_eq!(len, 16);
    }

    #[test]
    fn expected_byte_len_int4_packing() {
        let len = expected_byte_len(OnnxType::Int4, Dtype::U8, 5, "t").unwrap();
        assert_eq!(len, 3); // ceil(5/2) = 3
    }

    #[test]
    fn expected_byte_len_int4_even() {
        let len = expected_byte_len(OnnxType::Int4, Dtype::U8, 4, "t").unwrap();
        assert_eq!(len, 2);
    }

    // ── expected_byte_len: Uint4 packing ──────────────────────────────

    #[test]
    fn expected_byte_len_uint4_odd_count() {
        let len = expected_byte_len(OnnxType::Uint4, Dtype::U8, 7, "t").unwrap();
        assert_eq!(len, 4); // ceil(7/2) = 4
    }

    #[test]
    fn expected_byte_len_uint4_even_count() {
        let len = expected_byte_len(OnnxType::Uint4, Dtype::U8, 6, "t").unwrap();
        assert_eq!(len, 3); // 6/2 = 3
    }

    // ── expected_byte_len: overflow ───────────────────────────────────

    #[test]
    fn expected_byte_len_overflow() {
        let result = expected_byte_len(
            OnnxType::Float,
            Dtype::F32,
            usize::MAX,
            "overflow_test",
        );
        assert!(result.is_err());
    }

    // ── expected_byte_len: BF16 / F16 sizes ───────────────────────────

    #[test]
    fn expected_byte_len_bf16() {
        let len = expected_byte_len(OnnxType::Bfloat16, Dtype::BF16, 10, "t").unwrap();
        assert_eq!(len, 20); // 10 * 2 bytes
    }

    #[test]
    fn expected_byte_len_float16() {
        let len = expected_byte_len(OnnxType::Float16, Dtype::F16, 8, "t").unwrap();
        assert_eq!(len, 16); // 8 * 2 bytes
    }

    // ── expected_byte_len: zero elements ──────────────────────────────

    #[test]
    fn expected_byte_len_zero_elements() {
        let len = expected_byte_len(OnnxType::Float, Dtype::F32, 0, "t").unwrap();
        assert_eq!(len, 0);
    }

    #[test]
    fn expected_byte_len_int4_zero_elements() {
        let len = expected_byte_len(OnnxType::Int4, Dtype::U8, 0, "t").unwrap();
        assert_eq!(len, 0);
    }

    // ── ensure_len ────────────────────────────────────────────────────

    #[test]
    fn ensure_len_matching() {
        assert!(ensure_len("t", 5, 5, "field").is_ok());
    }

    #[test]
    fn ensure_len_mismatch() {
        let err = ensure_len("tensor_x", 3, 5, "float_data").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("tensor_x"));
        assert!(msg.contains("float_data"));
        assert!(msg.contains("expected 3"));
        assert!(msg.contains("5"));
    }

    // ── build_tensor_bytes: Int32 ─────────────────────────────────────

    #[test]
    fn build_tensor_bytes_int32() {
        let mut input = make_input(OnnxType::Int32, 3, Bytes::new(), "i32");
        input.int32_data = vec![0, -1, i32::MAX];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 12);
        assert_eq!(i32::from_le_bytes(result[0..4].try_into().unwrap()), 0);
        assert_eq!(i32::from_le_bytes(result[4..8].try_into().unwrap()), -1);
        assert_eq!(
            i32::from_le_bytes(result[8..12].try_into().unwrap()),
            i32::MAX
        );
    }

    #[test]
    fn build_tensor_bytes_int32_wrong_len() {
        let mut input = make_input(OnnxType::Int32, 2, Bytes::new(), "i32");
        input.int32_data = vec![1, 2, 3]; // 3 != 2
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Uint64 ────────────────────────────────────

    #[test]
    fn build_tensor_bytes_uint64() {
        let mut input = make_input(OnnxType::Uint64, 2, Bytes::new(), "u64");
        input.uint64_data = vec![0, u64::MAX];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 16);
        assert_eq!(u64::from_le_bytes(result[0..8].try_into().unwrap()), 0);
        assert_eq!(
            u64::from_le_bytes(result[8..16].try_into().unwrap()),
            u64::MAX
        );
    }

    #[test]
    fn build_tensor_bytes_uint64_wrong_len() {
        let mut input = make_input(OnnxType::Uint64, 3, Bytes::new(), "u64");
        input.uint64_data = vec![1]; // 1 != 3
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Int16 from int32 ──────────────────────────

    #[test]
    fn build_tensor_bytes_int16() {
        let mut input = make_input(OnnxType::Int16, 3, Bytes::new(), "i16");
        input.int32_data = vec![-32768, 0, 32767];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(i16::from_le_bytes(result[0..2].try_into().unwrap()), -32768);
        assert_eq!(i16::from_le_bytes(result[2..4].try_into().unwrap()), 0);
        assert_eq!(i16::from_le_bytes(result[4..6].try_into().unwrap()), 32767);
    }

    #[test]
    fn build_tensor_bytes_int16_overflow() {
        let mut input = make_input(OnnxType::Int16, 1, Bytes::new(), "i16");
        input.int32_data = vec![32768]; // exceeds i16 range
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Uint16 from int32 ─────────────────────────

    #[test]
    fn build_tensor_bytes_uint16() {
        let mut input = make_input(OnnxType::Uint16, 2, Bytes::new(), "u16");
        input.int32_data = vec![0, 65535];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(u16::from_le_bytes(result[0..2].try_into().unwrap()), 0);
        assert_eq!(u16::from_le_bytes(result[2..4].try_into().unwrap()), 65535);
    }

    #[test]
    fn build_tensor_bytes_uint16_overflow() {
        let mut input = make_input(OnnxType::Uint16, 1, Bytes::new(), "u16");
        input.int32_data = vec![65536]; // exceeds u16 range
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Float16 bits from int32 ───────────────────

    #[test]
    fn build_tensor_bytes_float16_bits() {
        let mut input = make_input(OnnxType::Float16, 2, Bytes::new(), "f16");
        input.int32_data = vec![0x3C00, 0x4000]; // 1.0, 2.0 in f16
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(u16::from_le_bytes(result[0..2].try_into().unwrap()), 0x3C00);
        assert_eq!(u16::from_le_bytes(result[2..4].try_into().unwrap()), 0x4000);
    }

    #[test]
    fn build_tensor_bytes_float16_bits_overflow() {
        let mut input = make_input(OnnxType::Float16, 1, Bytes::new(), "f16");
        input.int32_data = vec![65536]; // exceeds u16
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Bfloat16 bits from int32 ──────────────────

    #[test]
    fn build_tensor_bytes_bfloat16_bits() {
        let mut input = make_input(OnnxType::Bfloat16, 2, Bytes::new(), "bf16");
        input.int32_data = vec![0x3F80, 0x4000]; // 1.0, 2.0 in bf16
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(u16::from_le_bytes(result[0..2].try_into().unwrap()), 0x3F80);
        assert_eq!(u16::from_le_bytes(result[2..4].try_into().unwrap()), 0x4000);
    }

    #[test]
    fn build_tensor_bytes_bfloat16_bits_overflow() {
        let mut input = make_input(OnnxType::Bfloat16, 1, Bytes::new(), "bf16");
        input.int32_data = vec![65536]; // exceeds u16
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Int8 overflow ─────────────────────────────

    #[test]
    fn build_tensor_bytes_int8_overflow() {
        let mut input = make_input(OnnxType::Int8, 1, Bytes::new(), "i8");
        input.int32_data = vec![128]; // exceeds i8 max of 127
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Uint8 overflow ────────────────────────────

    #[test]
    fn build_tensor_bytes_uint8_overflow() {
        let mut input = make_input(OnnxType::Uint8, 1, Bytes::new(), "u8");
        input.int32_data = vec![256]; // exceeds u8 max of 255
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: unsupported type without raw_data ─────────

    #[test]
    fn build_tensor_bytes_unsupported_type_no_raw() {
        let input = make_input(OnnxType::Complex64, 1, Bytes::new(), "cx");
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Double wrong length ───────────────────────

    #[test]
    fn build_tensor_bytes_double_wrong_len() {
        let mut input = make_input(OnnxType::Double, 2, Bytes::new(), "d");
        input.double_data = vec![1.0]; // 1 != 2
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Int64 wrong length ────────────────────────

    #[test]
    fn build_tensor_bytes_int64_wrong_len() {
        let mut input = make_input(OnnxType::Int64, 2, Bytes::new(), "i64");
        input.int64_data = vec![]; // 0 != 2
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: String wrong length ───────────────────────

    #[test]
    fn build_tensor_bytes_string_wrong_len() {
        let mut input = make_input(OnnxType::String, 3, Bytes::new(), "s");
        input.string_data = vec![b"a".to_vec(), b"b".to_vec()]; // 2 != 3
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: String empty strings ──────────────────────

    #[test]
    fn build_tensor_bytes_string_empty_strings() {
        let mut input = make_input(OnnxType::String, 2, Bytes::new(), "s");
        input.string_data = vec![vec![], vec![]];
        let result = build_tensor_bytes(input).unwrap();
        // Each empty string = 4 byte length prefix + 0 bytes content = 4 bytes
        assert_eq!(result.len(), 8);
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(result[4..8].try_into().unwrap()), 0);
    }

    // ── build_tensor_bytes: zero element count ────────────────────────

    #[test]
    fn build_tensor_bytes_float_zero_elements() {
        let mut input = make_input(OnnxType::Float, 0, Bytes::new(), "f");
        input.float_data = vec![];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ══════════════════════════════════════════════════════════════════════
    // NEW TESTS (30 additional)
    // ══════════════════════════════════════════════════════════════════════

    // ── Special float values: NaN ──────────────────────────────────────

    #[test]
    fn build_tensor_bytes_float_nan() {
        let mut input = make_input(OnnxType::Float, 1, Bytes::new(), "nan");
        input.float_data = vec![f32::NAN];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4);
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert!(val.is_nan());
    }

    // ── Special float values: positive infinity ────────────────────────

    #[test]
    fn build_tensor_bytes_float_pos_inf() {
        let mut input = make_input(OnnxType::Float, 1, Bytes::new(), "pinf");
        input.float_data = vec![f32::INFINITY];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4);
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    // ── Special float values: negative infinity ────────────────────────

    #[test]
    fn build_tensor_bytes_float_neg_inf() {
        let mut input = make_input(OnnxType::Float, 1, Bytes::new(), "ninf");
        input.float_data = vec![f32::NEG_INFINITY];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4);
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert!(val.is_infinite() && val.is_sign_negative());
    }

    // ── Special float values: negative zero ────────────────────────────

    #[test]
    fn build_tensor_bytes_float_neg_zero() {
        let mut input = make_input(OnnxType::Float, 1, Bytes::new(), "nz");
        input.float_data = vec![-0.0f32];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4);
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert!(val.is_sign_negative());
        assert_eq!(val, 0.0);
    }

    // ── Double: special values (NaN, Inf) ──────────────────────────────

    #[test]
    fn build_tensor_bytes_double_nan() {
        let mut input = make_input(OnnxType::Double, 1, Bytes::new(), "dnan");
        input.double_data = vec![f64::NAN];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 8);
        let val = f64::from_le_bytes(result[0..8].try_into().unwrap());
        assert!(val.is_nan());
    }

    #[test]
    fn build_tensor_bytes_double_infinity() {
        let mut input = make_input(OnnxType::Double, 1, Bytes::new(), "dinf");
        input.double_data = vec![f64::INFINITY];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 8);
        let val = f64::from_le_bytes(result[0..8].try_into().unwrap());
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    // ── Int32: boundary values (MIN, MAX, zero) ───────────────────────

    #[test]
    fn build_tensor_bytes_int32_boundary_values() {
        let mut input = make_input(OnnxType::Int32, 3, Bytes::new(), "i32b");
        input.int32_data = vec![i32::MIN, 0, i32::MAX];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 12);
        assert_eq!(i32::from_le_bytes(result[0..4].try_into().unwrap()), i32::MIN);
        assert_eq!(i32::from_le_bytes(result[4..8].try_into().unwrap()), 0);
        assert_eq!(i32::from_le_bytes(result[8..12].try_into().unwrap()), i32::MAX);
    }

    // ── Int64: boundary values (MIN, MAX) ──────────────────────────────

    #[test]
    fn build_tensor_bytes_int64_boundary_values() {
        let mut input = make_input(OnnxType::Int64, 2, Bytes::new(), "i64b");
        input.int64_data = vec![i64::MIN, i64::MAX];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 16);
        assert_eq!(i64::from_le_bytes(result[0..8].try_into().unwrap()), i64::MIN);
        assert_eq!(i64::from_le_bytes(result[8..16].try_into().unwrap()), i64::MAX);
    }

    // ── Uint64: single element (zero) ──────────────────────────────────

    #[test]
    fn build_tensor_bytes_uint64_single_zero() {
        let mut input = make_input(OnnxType::Uint64, 1, Bytes::new(), "u64z");
        input.uint64_data = vec![0u64];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 8);
        assert_eq!(u64::from_le_bytes(result[0..8].try_into().unwrap()), 0);
    }

    // ── Uint32: boundary values ────────────────────────────────────────

    #[test]
    fn build_tensor_bytes_uint32_boundary() {
        let mut input = make_input(OnnxType::Uint32, 2, Bytes::new(), "u32b");
        input.uint64_data = vec![0u64, u32::MAX as u64];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 8);
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(result[4..8].try_into().unwrap()), u32::MAX);
    }

    // ── Int8: boundary values (MIN=-128, MAX=127) ─────────────────────

    #[test]
    fn build_tensor_bytes_int8_boundary() {
        let mut input = make_input(OnnxType::Int8, 2, Bytes::new(), "i8b");
        input.int32_data = vec![-128, 127];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 128u8); // -128 as u8
        assert_eq!(result[1], 127u8);
    }

    // ── Int8: negative overflow (below -128) ───────────────────────────

    #[test]
    fn build_tensor_bytes_int8_negative_overflow() {
        let mut input = make_input(OnnxType::Int8, 1, Bytes::new(), "i8n");
        input.int32_data = vec![-129]; // below i8 min
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── Uint8: boundary (zero and 255) ─────────────────────────────────

    #[test]
    fn build_tensor_bytes_uint8_boundary() {
        let mut input = make_input(OnnxType::Uint8, 2, Bytes::new(), "u8b");
        input.int32_data = vec![0, 255];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 255);
    }

    // ── Uint8: negative value rejected ─────────────────────────────────

    #[test]
    fn build_tensor_bytes_uint8_negative_rejected() {
        let mut input = make_input(OnnxType::Uint8, 1, Bytes::new(), "u8n");
        input.int32_data = vec![-1];
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── Bool: all valid combinations ───────────────────────────────────

    #[test]
    fn build_tensor_bytes_bool_all_zeros() {
        let mut input = make_input(OnnxType::Bool, 3, Bytes::new(), "bz");
        input.int32_data = vec![0, 0, 0];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.as_ref(), &[0u8, 0, 0]);
    }

    #[test]
    fn build_tensor_bytes_bool_all_ones() {
        let mut input = make_input(OnnxType::Bool, 2, Bytes::new(), "bo");
        input.int32_data = vec![1, 1];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.as_ref(), &[1u8, 1]);
    }

    // ── String: UTF-8 content round-trip ───────────────────────────────

    #[test]
    fn build_tensor_bytes_string_utf8_content() {
        let mut input = make_input(OnnxType::String, 1, Bytes::new(), "sutf8");
        input.string_data = vec!["héllo".as_bytes().to_vec()];
        let result = build_tensor_bytes(input).unwrap();
        let len = u32::from_le_bytes(result[0..4].try_into().unwrap()) as usize;
        assert_eq!(len, "héllo".len()); // multi-byte UTF-8
        assert_eq!(&result[4..4 + len], "héllo".as_bytes());
    }

    // ── String: empty element_count zero ───────────────────────────────

    #[test]
    fn build_tensor_bytes_string_zero_elements() {
        let mut input = make_input(OnnxType::String, 0, Bytes::new(), "s0");
        input.string_data = vec![];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── String: single byte content ────────────────────────────────────

    #[test]
    fn build_tensor_bytes_string_single_byte() {
        let mut input = make_input(OnnxType::String, 1, Bytes::new(), "s1");
        input.string_data = vec![vec![42u8]];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 5); // 4 byte length + 1 byte content
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 1);
        assert_eq!(result[4], 42);
    }

    // ── Int16: negative overflow ───────────────────────────────────────

    #[test]
    fn build_tensor_bytes_int16_negative_overflow() {
        let mut input = make_input(OnnxType::Int16, 1, Bytes::new(), "i16n");
        input.int32_data = vec![-32769]; // below i16 min
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── Uint16: negative value rejected ────────────────────────────────

    #[test]
    fn build_tensor_bytes_uint16_negative_rejected() {
        let mut input = make_input(OnnxType::Uint16, 1, Bytes::new(), "u16n");
        input.int32_data = vec![-1];
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── Float16 bits: zero bits ────────────────────────────────────────

    #[test]
    fn build_tensor_bytes_float16_bits_zero() {
        let mut input = make_input(OnnxType::Float16, 1, Bytes::new(), "f16z");
        input.int32_data = vec![0];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(u16::from_le_bytes(result[0..2].try_into().unwrap()), 0);
    }

    // ── Bfloat16 bits: max u16 value ───────────────────────────────────

    #[test]
    fn build_tensor_bytes_bfloat16_bits_max() {
        let mut input = make_input(OnnxType::Bfloat16, 1, Bytes::new(), "bf16m");
        input.int32_data = vec![65535]; // u16::MAX
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(u16::from_le_bytes(result[0..2].try_into().unwrap()), 65535);
    }

    // ── expected_byte_len: single element ──────────────────────────────

    #[test]
    fn expected_byte_len_single_element_float() {
        let len = expected_byte_len(OnnxType::Float, Dtype::F32, 1, "t").unwrap();
        assert_eq!(len, 4);
    }

    // ── expected_byte_len: Uint64 (8 bytes per element) ────────────────

    #[test]
    fn expected_byte_len_uint64() {
        let len = expected_byte_len(OnnxType::Uint64, Dtype::U64, 3, "t").unwrap();
        assert_eq!(len, 24);
    }

    // ── expected_byte_len: Int4 single element ─────────────────────────

    #[test]
    fn expected_byte_len_int4_single() {
        let len = expected_byte_len(OnnxType::Int4, Dtype::U8, 1, "t").unwrap();
        assert_eq!(len, 1); // ceil(1/2) = 1
    }

    // ── expected_byte_len: Uint4 single element ────────────────────────

    #[test]
    fn expected_byte_len_uint4_single() {
        let len = expected_byte_len(OnnxType::Uint4, Dtype::U8, 1, "t").unwrap();
        assert_eq!(len, 1); // ceil(1/2) = 1
    }

    // ── expected_byte_len: Double (8 bytes per element) ────────────────

    #[test]
    fn expected_byte_len_double() {
        let len = expected_byte_len(OnnxType::Double, Dtype::F64, 2, "t").unwrap();
        assert_eq!(len, 16);
    }

    // ── ensure_len: zero elements matching ─────────────────────────────

    #[test]
    fn ensure_len_zero_matching() {
        assert!(ensure_len("t", 0, 0, "field").is_ok());
    }

    // ── ensure_len: error message contains all context ─────────────────

    #[test]
    fn ensure_len_error_message_content() {
        let err = ensure_len("my_tensor", 10, 3, "int64_data").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("my_tensor"), "should contain tensor name");
        assert!(msg.contains("int64_data"), "should contain field name");
        assert!(msg.contains("expected 10"), "should contain expected count");
        assert!(msg.contains("3"), "should contain actual count");
    }

    // ── build_tensor_bytes: raw_data with INT4 odd packing ────────────

    #[test]
    fn build_tensor_bytes_raw_data_int4_odd() {
        // 5 INT4 elements = ceil(5/2) = 3 bytes
        let raw = Bytes::from_static(&[0xABu8, 0xCD, 0xEF]);
        let input = make_input(OnnxType::Int4, 5, raw.clone(), "int4");
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result, raw);
    }

    // ── build_tensor_bytes: raw_data INT4 wrong length ────────────────

    #[test]
    fn build_tensor_bytes_raw_data_int4_wrong_len() {
        // 5 INT4 elements need 3 bytes, but we give 2
        let raw = Bytes::from_static(&[0xABu8, 0xCD]);
        let input = make_input(OnnxType::Int4, 5, raw, "int4");
        assert!(build_tensor_bytes(input).is_err());
    }

    // ── build_tensor_bytes: Uint32 exactly u32::MAX ───────────────────

    #[test]
    fn build_tensor_bytes_uint32_exact_max() {
        let mut input = make_input(OnnxType::Uint32, 1, Bytes::new(), "u32max");
        input.uint64_data = vec![u32::MAX as u64];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), u32::MAX);
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (45+ new tests)
    // ══════════════════════════════════════════════════════════════════════

    // ── pack_f32: direct testing with subnormal float ──────────────────

    #[test]
    fn pack_f32_subnormal_value() {
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        let result = pack_f32(vec![subnormal], 1, "sub").unwrap();
        assert_eq!(result.len(), 4);
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert_eq!(val.to_bits(), 1u32);
    }

    // ── pack_f32: direct testing with f32::MAX ────────────────────────

    #[test]
    fn pack_f32_max_value() {
        let result = pack_f32(vec![f32::MAX], 1, "fmax").unwrap();
        assert_eq!(result.len(), 4);
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert_eq!(val, f32::MAX);
    }

    // ── pack_f32: direct testing with f32::MIN_POSITIVE ───────────────

    #[test]
    fn pack_f32_min_positive() {
        let result = pack_f32(vec![f32::MIN_POSITIVE], 1, "fminp").unwrap();
        assert_eq!(result.len(), 4);
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert_eq!(val, f32::MIN_POSITIVE);
    }

    // ── pack_f32: wrong length ────────────────────────────────────────

    #[test]
    fn pack_f32_wrong_length() {
        let result = pack_f32(vec![1.0f32, 2.0], 3, "f32wl");
        assert!(result.is_err());
    }

    // ── pack_f32: empty vector ────────────────────────────────────────

    #[test]
    fn pack_f32_empty() {
        let result = pack_f32(vec![], 0, "f32e").unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── pack_f64: direct testing with f64::MAX ────────────────────────

    #[test]
    fn pack_f64_max_value() {
        let result = pack_f64(vec![f64::MAX], 1, "dmax").unwrap();
        assert_eq!(result.len(), 8);
        let val = f64::from_le_bytes(result[0..8].try_into().unwrap());
        assert_eq!(val, f64::MAX);
    }

    // ── pack_f64: direct testing with f64::MIN_POSITIVE ───────────────

    #[test]
    fn pack_f64_min_positive() {
        let result = pack_f64(vec![f64::MIN_POSITIVE], 1, "dminp").unwrap();
        assert_eq!(result.len(), 8);
        let val = f64::from_le_bytes(result[0..8].try_into().unwrap());
        assert_eq!(val, f64::MIN_POSITIVE);
    }

    // ── pack_f64: negative infinity direct ────────────────────────────

    #[test]
    fn pack_f64_neg_infinity() {
        let result = pack_f64(vec![f64::NEG_INFINITY], 1, "dninf").unwrap();
        assert_eq!(result.len(), 8);
        let val = f64::from_le_bytes(result[0..8].try_into().unwrap());
        assert!(val.is_infinite() && val.is_sign_negative());
    }

    // ── pack_f64: wrong length ────────────────────────────────────────

    #[test]
    fn pack_f64_wrong_length() {
        let result = pack_f64(vec![1.0], 2, "d64wl");
        assert!(result.is_err());
    }

    // ── pack_f64: empty vector ────────────────────────────────────────

    #[test]
    fn pack_f64_empty() {
        let result = pack_f64(vec![], 0, "d64e").unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── pack_i32: direct testing with i32::MIN ────────────────────────

    #[test]
    fn pack_i32_min_value() {
        let result = pack_i32(vec![i32::MIN], 1, "imin").unwrap();
        assert_eq!(result.len(), 4);
        let val = i32::from_le_bytes(result[0..4].try_into().unwrap());
        assert_eq!(val, i32::MIN);
    }

    // ── pack_i32: empty vector ────────────────────────────────────────

    #[test]
    fn pack_i32_empty() {
        let result = pack_i32(vec![], 0, "i32e").unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── pack_i32: wrong length ────────────────────────────────────────

    #[test]
    fn pack_i32_wrong_length() {
        let result = pack_i32(vec![1], 0, "i32wl");
        assert!(result.is_err());
    }

    // ── pack_i64: direct testing with i64::MIN ────────────────────────

    #[test]
    fn pack_i64_min_value() {
        let result = pack_i64(vec![i64::MIN], 1, "i64min").unwrap();
        assert_eq!(result.len(), 8);
        let val = i64::from_le_bytes(result[0..8].try_into().unwrap());
        assert_eq!(val, i64::MIN);
    }

    // ── pack_i64: empty vector ────────────────────────────────────────

    #[test]
    fn pack_i64_empty() {
        let result = pack_i64(vec![], 0, "i64e").unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── pack_i64: wrong length ────────────────────────────────────────

    #[test]
    fn pack_i64_wrong_length() {
        let result = pack_i64(vec![1, 2], 1, "i64wl");
        assert!(result.is_err());
    }

    // ── pack_u64: direct testing with u64::MAX ────────────────────────

    #[test]
    fn pack_u64_max_value() {
        let result = pack_u64(vec![u64::MAX], 1, "u64max").unwrap();
        assert_eq!(result.len(), 8);
        let val = u64::from_le_bytes(result[0..8].try_into().unwrap());
        assert_eq!(val, u64::MAX);
    }

    // ── pack_u64: empty vector ────────────────────────────────────────

    #[test]
    fn pack_u64_empty() {
        let result = pack_u64(vec![], 0, "u64e").unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── pack_u64: wrong length ────────────────────────────────────────

    #[test]
    fn pack_u64_wrong_length() {
        let result = pack_u64(vec![], 1, "u64wl");
        assert!(result.is_err());
    }

    // ── pack_u32_from_u64: zero value ─────────────────────────────────

    #[test]
    fn pack_u32_from_u64_zero() {
        let result = pack_u32_from_u64(vec![0u64], 1, "u32z").unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 0);
    }

    // ── pack_u32_from_u64: wrong length ───────────────────────────────

    #[test]
    fn pack_u32_from_u64_wrong_length() {
        let result = pack_u32_from_u64(vec![1, 2], 1, "u32wl");
        assert!(result.is_err());
    }

    // ── pack_u32_from_u64: multiple elements ──────────────────────────

    #[test]
    fn pack_u32_from_u64_multiple() {
        let vals = vec![100u64, 200, 300];
        let result = pack_u32_from_u64(vals.clone(), 3, "u32m").unwrap();
        assert_eq!(result.len(), 12);
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 100);
        assert_eq!(u32::from_le_bytes(result[4..8].try_into().unwrap()), 200);
        assert_eq!(u32::from_le_bytes(result[8..12].try_into().unwrap()), 300);
    }

    // ── pack_strings: direct testing with multiple strings ────────────

    #[test]
    fn pack_strings_multiple_strings() {
        let data = vec![b"ab".to_vec(), b"cde".to_vec(), b"f".to_vec()];
        let result = pack_strings(data, 3, "sm").unwrap();
        // 3 strings: (4+2) + (4+3) + (4+1) = 18 bytes
        assert_eq!(result.len(), 18);
        // First string: len=2, content="ab"
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 2);
        assert_eq!(&result[4..6], b"ab");
        // Second string: len=3, content="cde"
        assert_eq!(u32::from_le_bytes(result[6..10].try_into().unwrap()), 3);
        assert_eq!(&result[10..13], b"cde");
        // Third string: len=1, content="f"
        assert_eq!(u32::from_le_bytes(result[13..17].try_into().unwrap()), 1);
        assert_eq!(&result[17..18], b"f");
    }

    // ── pack_strings: wrong length ────────────────────────────────────

    #[test]
    fn pack_strings_wrong_length() {
        let data = vec![b"a".to_vec()];
        let result = pack_strings(data, 2, "swl");
        assert!(result.is_err());
    }

    // ── pack_strings: empty vector ────────────────────────────────────

    #[test]
    fn pack_strings_empty() {
        let result = pack_strings(vec![], 0, "se").unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── pack_strings: string with binary (non-UTF8) content ───────────

    #[test]
    fn pack_strings_binary_content() {
        let data = vec![vec![0x00, 0xFF, 0xFE]];
        let result = pack_strings(data.clone(), 1, "sbin").unwrap();
        assert_eq!(result.len(), 7); // 4 + 3
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 3);
        assert_eq!(&result[4..7], &[0x00, 0xFF, 0xFE]);
    }

    // ── pack_bool_from_i32: direct wrong length ───────────────────────

    #[test]
    fn pack_bool_wrong_length() {
        let result = pack_bool_from_i32(vec![0, 1], 1, "bwl");
        assert!(result.is_err());
    }

    // ── pack_bool_from_i32: large boolean vector ──────────────────────

    #[test]
    fn pack_bool_large_vector() {
        let data: Vec<i32> = (0..100).map(|i| i % 2).collect();
        let result = pack_bool_from_i32(data, 100, "bl").unwrap();
        assert_eq!(result.len(), 100);
        for i in 0..100 {
            assert_eq!(result[i], (i % 2) as u8);
        }
    }

    // ── pack_i8_from_i32: direct boundary values ─────────────────────

    #[test]
    fn pack_i8_from_i32_min_max() {
        let result = pack_i8_from_i32(vec![-128, 127], 2, "i8mm").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 128u8); // -128 as u8
        assert_eq!(result[1], 127u8);
    }

    // ── pack_i8_from_i32: wrong length ────────────────────────────────

    #[test]
    fn pack_i8_from_i32_wrong_length() {
        let result = pack_i8_from_i32(vec![], 1, "i8wl");
        assert!(result.is_err());
    }

    // ── pack_u8_from_i32: wrong length ────────────────────────────────

    #[test]
    fn pack_u8_from_i32_wrong_length() {
        let result = pack_u8_from_i32(vec![1, 2, 3], 2, "u8wl");
        assert!(result.is_err());
    }

    // ── pack_i16_from_i32: direct wrong length ────────────────────────

    #[test]
    fn pack_i16_from_i32_wrong_length() {
        let result = pack_i16_from_i32(vec![0], 0, "i16wl");
        assert!(result.is_err());
    }

    // ── pack_u16_from_i32: wrong length ───────────────────────────────

    #[test]
    fn pack_u16_from_i32_wrong_length() {
        let result = pack_u16_from_i32(vec![1, 2], 3, "u16wl");
        assert!(result.is_err());
    }

    // ── pack_f16_bits_from_i32: wrong length ──────────────────────────

    #[test]
    fn pack_f16_bits_wrong_length() {
        let result = pack_f16_bits_from_i32(vec![], 1, "f16wl");
        assert!(result.is_err());
    }

    // ── pack_bf16_bits_from_i32: wrong length ─────────────────────────

    #[test]
    fn pack_bf16_bits_wrong_length() {
        let result = pack_bf16_bits_from_i32(vec![0, 1], 1, "bf16wl");
        assert!(result.is_err());
    }

    // ── ensure_len: large count mismatch ──────────────────────────────

    #[test]
    fn ensure_len_large_mismatch() {
        let err = ensure_len("big_tensor", 1_000_000, 0, "float_data").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("big_tensor"));
        assert!(msg.contains("1000000"));
    }

    // ── ensure_len: expected one, actual zero ─────────────────────────

    #[test]
    fn ensure_len_expected_one_actual_zero() {
        let err = ensure_len("t", 1, 0, "field").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expected 1"));
    }

    // ── expected_byte_len: Int32 (4 bytes per element) ────────────────

    #[test]
    fn expected_byte_len_int32() {
        let len = expected_byte_len(OnnxType::Int32, Dtype::I32, 5, "t").unwrap();
        assert_eq!(len, 20);
    }

    // ── expected_byte_len: Int64 (8 bytes per element) ────────────────

    #[test]
    fn expected_byte_len_int64() {
        let len = expected_byte_len(OnnxType::Int64, Dtype::I64, 3, "t").unwrap();
        assert_eq!(len, 24);
    }

    // ── expected_byte_len: Bool (1 byte per element) ──────────────────

    #[test]
    fn expected_byte_len_bool() {
        let len = expected_byte_len(OnnxType::Bool, Dtype::BOOL, 10, "t").unwrap();
        assert_eq!(len, 10);
    }

    // ── expected_byte_len: Int8 (1 byte per element) ──────────────────

    #[test]
    fn expected_byte_len_int8() {
        let len = expected_byte_len(OnnxType::Int8, Dtype::I8, 16, "t").unwrap();
        assert_eq!(len, 16);
    }

    // ── expected_byte_len: Uint8 (1 byte per element) ─────────────────

    #[test]
    fn expected_byte_len_uint8() {
        let len = expected_byte_len(OnnxType::Uint8, Dtype::U8, 32, "t").unwrap();
        assert_eq!(len, 32);
    }

    // ── expected_byte_len: Int4 with large count ──────────────────────

    #[test]
    fn expected_byte_len_int4_large() {
        let len = expected_byte_len(OnnxType::Int4, Dtype::U8, 1000, "t").unwrap();
        assert_eq!(len, 500);
    }

    // ── expected_byte_len: Uint4 with large count ─────────────────────

    #[test]
    fn expected_byte_len_uint4_large() {
        let len = expected_byte_len(OnnxType::Uint4, Dtype::U8, 999, "t").unwrap();
        assert_eq!(len, 500); // ceil(999/2) = 500
    }

    // ── build_tensor_bytes: raw_data exact match with zero bytes ──────

    #[test]
    fn build_tensor_bytes_raw_data_zero_length_valid() {
        let raw = Bytes::new();
        let input = make_input(OnnxType::Float, 0, raw.clone(), "zero");
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── build_tensor_bytes: raw_data with Int32 correct size ──────────

    #[test]
    fn build_tensor_bytes_raw_data_int32_correct() {
        let data: Vec<u8> = [1i32, 2, 3].iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let raw = Bytes::from(data);
        let input = make_input(OnnxType::Int32, 3, raw.clone(), "i32r");
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result, raw);
    }

    // ── build_tensor_bytes: error message for wrong raw_data length ───

    #[test]
    fn build_tensor_bytes_raw_data_error_message() {
        let raw = Bytes::from_static(&[0u8; 3]); // 3 bytes but need 8 for 2 floats
        let input = make_input(OnnxType::Float, 2, raw, "errmsg");
        let err = build_tensor_bytes(input).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("errmsg"));
        assert!(msg.contains("3"));
        assert!(msg.contains("expected"));
    }

    // ── build_tensor_bytes: Int64 single value ────────────────────────

    #[test]
    fn build_tensor_bytes_int64_single_value() {
        let mut input = make_input(OnnxType::Int64, 1, Bytes::new(), "i64s");
        input.int64_data = vec![42i64];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 8);
        assert_eq!(i64::from_le_bytes(result[0..8].try_into().unwrap()), 42);
    }

    // ── build_tensor_bytes: Double exact value round-trip ─────────────

    #[test]
    fn build_tensor_bytes_double_exact_roundtrip() {
        let mut input = make_input(OnnxType::Double, 1, Bytes::new(), "drt");
        input.double_data = vec![std::f64::consts::PI];
        let result = build_tensor_bytes(input).unwrap();
        let val = f64::from_le_bytes(result[0..8].try_into().unwrap());
        assert_eq!(val, std::f64::consts::PI);
    }

    // ── build_tensor_bytes: Float multiple values round-trip ──────────

    #[test]
    fn build_tensor_bytes_float_multiple_roundtrip() {
        let values = vec![0.0f32, -0.0, 1.0, -1.0, f32::MAX, f32::MIN];
        let mut input = make_input(OnnxType::Float, values.len(), Bytes::new(), "frt");
        input.float_data = values.clone();
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), values.len() * 4);
        for (i, expected) in values.iter().enumerate() {
            let val = f32::from_le_bytes(result[i * 4..i * 4 + 4].try_into().unwrap());
            if expected.is_nan() {
                assert!(val.is_nan());
            } else if expected == &0.0 {
                assert_eq!(val.to_bits(), expected.to_bits());
            } else {
                assert_eq!(val, *expected);
            }
        }
    }

    // ── build_tensor_bytes: Bool value 2 is rejected with error ───────

    #[test]
    fn build_tensor_bytes_bool_value_2_error_message() {
        let mut input = make_input(OnnxType::Bool, 1, Bytes::new(), "b2");
        input.int32_data = vec![2];
        let err = build_tensor_bytes(input).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("b2"));
        assert!(msg.contains("2"));
    }

    // ── build_tensor_bytes: unsupported type error message ────────────

    #[test]
    fn build_tensor_bytes_unsupported_type_error_message() {
        let input = make_input(OnnxType::Complex128, 1, Bytes::new(), "cx128");
        let err = build_tensor_bytes(input).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("cx128"));
        assert!(msg.contains("raw_data"));
    }

    // ── build_tensor_bytes: Uint32 error message on overflow ──────────

    #[test]
    fn build_tensor_bytes_uint32_overflow_error_message() {
        let mut input = make_input(OnnxType::Uint32, 1, Bytes::new(), "u32of");
        input.uint64_data = vec![u32::MAX as u64 + 1];
        let err = build_tensor_bytes(input).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("u32of"));
        assert!(msg.contains("overflow"));
    }

    // ── build_tensor_bytes: Int8 overflow error message ───────────────

    #[test]
    fn build_tensor_bytes_int8_overflow_error_message() {
        let mut input = make_input(OnnxType::Int8, 1, Bytes::new(), "i8of");
        input.int32_data = vec![200];
        let err = build_tensor_bytes(input).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("i8of"));
    }

    // ── build_tensor_bytes: Int16 overflow error message ──────────────

    #[test]
    fn build_tensor_bytes_int16_overflow_error_message() {
        let mut input = make_input(OnnxType::Int16, 1, Bytes::new(), "i16of");
        input.int32_data = vec![40000];
        let err = build_tensor_bytes(input).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("i16of"));
    }

    // ── build_tensor_bytes: Int32 single value ────────────────────────

    #[test]
    fn build_tensor_bytes_int32_single_value() {
        let mut input = make_input(OnnxType::Int32, 1, Bytes::new(), "i32s");
        input.int32_data = vec![42];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(i32::from_le_bytes(result[0..4].try_into().unwrap()), 42);
    }

    // ── build_tensor_bytes: Int16 zero value ──────────────────────────

    #[test]
    fn build_tensor_bytes_int16_zero() {
        let mut input = make_input(OnnxType::Int16, 1, Bytes::new(), "i16z");
        input.int32_data = vec![0];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(i16::from_le_bytes(result[0..2].try_into().unwrap()), 0);
    }

    // ── build_tensor_bytes: Uint16 zero value ─────────────────────────

    #[test]
    fn build_tensor_bytes_uint16_zero() {
        let mut input = make_input(OnnxType::Uint16, 1, Bytes::new(), "u16z");
        input.int32_data = vec![0];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(u16::from_le_bytes(result[0..2].try_into().unwrap()), 0);
    }

    // ── pack_i32_with: verifies saturating_mul capacity ───────────────

    #[test]
    fn pack_i32_with_empty_data() {
        let result = pack_i32_with(vec![], 0, "p32e", 4, |_, _| Ok(())).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── pack_i32_with: wrong length ───────────────────────────────────

    #[test]
    fn pack_i32_with_wrong_length() {
        let result = pack_i32_with(vec![1], 2, "p32wl", 4, |_, _| Ok(()));
        assert!(result.is_err());
    }

    // ── pack_i32_with: write callback error propagation ───────────────

    #[test]
    fn pack_i32_with_write_error_propagation() {
        let result = pack_i32_with(vec![1], 1, "p32err", 4, |_, _| {
            Err(LoaderError::Onnx("write failed".to_string()))
        });
        assert!(result.is_err());
    }

    // ── expected_byte_len: Int16 (2 bytes per element) ────────────────

    #[test]
    fn expected_byte_len_int16() {
        let len = expected_byte_len(OnnxType::Int16, Dtype::I16, 7, "t").unwrap();
        assert_eq!(len, 14);
    }

    // ── expected_byte_len: Uint16 (2 bytes per element) ───────────────

    #[test]
    fn expected_byte_len_uint16() {
        let len = expected_byte_len(OnnxType::Uint16, Dtype::U16, 3, "t").unwrap();
        assert_eq!(len, 6);
    }

    // ── build_tensor_bytes: Float zero and normal mixed ───────────────

    #[test]
    fn build_tensor_bytes_float_mixed_values() {
        let mut input = make_input(OnnxType::Float, 4, Bytes::new(), "fmix");
        input.float_data = vec![0.0, 1.0, -1.0, 0.5];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 16);
        let vals: Vec<f32> = (0..4)
            .map(|i| f32::from_le_bytes(result[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], -1.0);
        assert_eq!(vals[3], 0.5);
    }

    // ── build_tensor_bytes: String with CJK multi-byte ───────────────

    #[test]
    fn build_tensor_bytes_string_cjk_content() {
        let cjk = "你好世界".as_bytes().to_vec();
        let cjk_len = cjk.len();
        let mut input = make_input(OnnxType::String, 1, Bytes::new(), "scjk");
        input.string_data = vec![cjk.clone()];
        let result = build_tensor_bytes(input).unwrap();
        let stored_len = u32::from_le_bytes(result[0..4].try_into().unwrap()) as usize;
        assert_eq!(stored_len, cjk_len);
        assert_eq!(&result[4..4 + cjk_len], &cjk[..]);
    }

    // ── build_tensor_bytes: Uint64 zero elements ──────────────────────

    #[test]
    fn build_tensor_bytes_uint64_zero_elements() {
        let mut input = make_input(OnnxType::Uint64, 0, Bytes::new(), "u64z");
        input.uint64_data = vec![];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── build_tensor_bytes: Double zero elements ──────────────────────

    #[test]
    fn build_tensor_bytes_double_zero_elements() {
        let mut input = make_input(OnnxType::Double, 0, Bytes::new(), "dz");
        input.double_data = vec![];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── build_tensor_bytes: Int8 zero elements ────────────────────────

    #[test]
    fn build_tensor_bytes_int8_zero_elements() {
        let mut input = make_input(OnnxType::Int8, 0, Bytes::new(), "i8z");
        input.int32_data = vec![];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── build_tensor_bytes: Uint8 zero elements ───────────────────────

    #[test]
    fn build_tensor_bytes_uint8_zero_elements() {
        let mut input = make_input(OnnxType::Uint8, 0, Bytes::new(), "u8z");
        input.int32_data = vec![];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── build_tensor_bytes: Int16 zero elements ───────────────────────

    #[test]
    fn build_tensor_bytes_int16_zero_elements() {
        let mut input = make_input(OnnxType::Int16, 0, Bytes::new(), "i16z");
        input.int32_data = vec![];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── build_tensor_bytes: Bool zero elements ────────────────────────

    #[test]
    fn build_tensor_bytes_bool_zero_elements() {
        let mut input = make_input(OnnxType::Bool, 0, Bytes::new(), "bz");
        input.int32_data = vec![];
        let result = build_tensor_bytes(input).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ══════════════════════════════════════════════════════════════════════
    // SUPPLEMENTARY TESTS (10 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── pack_i32_with: multiple elements produce correct byte output ───

    #[test]
    fn pack_i32_with_multiple_elements_correct_output() {
        // Arrange: 3 i32 values written as raw little-endian via callback
        let data = vec![0x01020304i32, 0x05060708, 0x0A0B0C0D];
        // Act
        let result = pack_i32_with(data, 3, "multi", 4, |val, out| {
            out.extend_from_slice(&val.to_le_bytes());
            Ok(())
        })
        .unwrap();
        // Assert: 3 elements * 4 bytes = 12 bytes total
        assert_eq!(result.len(), 12);
        assert_eq!(
            i32::from_le_bytes(result[0..4].try_into().unwrap()),
            0x01020304
        );
        assert_eq!(
            i32::from_le_bytes(result[4..8].try_into().unwrap()),
            0x05060708
        );
        assert_eq!(
            i32::from_le_bytes(result[8..12].try_into().unwrap()),
            0x0A0B0C0D
        );
    }

    // ── pack_f64: negative value round-trip ─────────────────────────────

    #[test]
    fn pack_f64_negative_roundtrip() {
        // Arrange: a negative f64 that exercises sign bit
        let val = -std::f64::consts::E;
        // Act
        let result = pack_f64(vec![val], 1, "dneg").unwrap();
        // Assert
        assert_eq!(result.len(), 8);
        let got = f64::from_le_bytes(result[0..8].try_into().unwrap());
        assert_eq!(got.to_bits(), val.to_bits());
    }

    // ── pack_strings: embedded NUL bytes in string content ─────────────

    #[test]
    fn pack_strings_embedded_nul_bytes() {
        // Arrange: string data containing NUL bytes in the middle
        let data = vec![vec![0x41, 0x00, 0x42]];
        // Act
        let result = pack_strings(data.clone(), 1, "nulstr").unwrap();
        // Assert: length prefix = 3, then raw bytes preserved including NUL
        assert_eq!(result.len(), 7);
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 3);
        assert_eq!(&result[4..7], &[0x41, 0x00, 0x42]);
    }

    // ── expected_byte_len: String type uses Dtype size (not div_ceil) ──

    #[test]
    fn expected_byte_len_string_type_uses_dtype_size() {
        // Arrange: String type should hit the non-Int4/Uint4 branch
        // Act
        let len = expected_byte_len(OnnxType::String, Dtype::U8, 10, "strtype").unwrap();
        // Assert: String is not Int4/Uint4, so element_count * dtype.size() = 10*1 = 10
        assert_eq!(len, 10);
    }

    // ── pack_u32_from_u64: zero value boundary ─────────────────────────

    #[test]
    fn pack_u32_from_u64_zero_boundary() {
        // Arrange: u64 value 0 (u32::MIN)
        // Act
        let result = pack_u32_from_u64(vec![0u64], 1, "u32min").unwrap();
        // Assert
        assert_eq!(result.len(), 4);
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 0);
    }

    // ── pack_i32_with: write callback invoked in order ──────────────────

    #[test]
    fn pack_i32_with_callback_ordering() {
        // Arrange: collect the order of values passed to callback
        let mut seen: Vec<i32> = Vec::new();
        let data = vec![10, 20, 30];
        // Act
        let _ = pack_i32_with(data, 3, "order", 1, |val, out| {
            seen.push(val);
            out.push(val as u8);
            Ok(())
        })
        .unwrap();
        // Assert: callback received values in original order
        assert_eq!(seen, vec![10, 20, 30]);
    }

    // ── build_tensor_bytes: raw_data passthrough ignores typed fields ──

    #[test]
    fn build_tensor_bytes_raw_data_ignores_typed_fields() {
        // Arrange: raw_data present but float_data also non-empty (should be ignored)
        let raw_bytes: Vec<u8> = [99.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let raw = Bytes::from(raw_bytes.clone());
        let mut input = make_input(OnnxType::Float, 1, raw.clone(), "raw_pri");
        input.float_data = vec![0.0]; // would produce different output if used
        // Act
        let result = build_tensor_bytes(input).unwrap();
        // Assert: raw_data takes priority, content matches raw exactly
        assert_eq!(result.as_ref(), raw_bytes.as_slice());
    }

    // ── pack_i16_from_i32: multiple values correct byte layout ─────────

    #[test]
    fn pack_i16_from_i32_multiple_values_layout() {
        // Arrange: i16 values at boundaries: min, -1, 0, 1, max
        let data = vec![-32768i32, -1, 0, 1, 32767];
        // Act
        let result = pack_i16_from_i32(data, 5, "i16layout").unwrap();
        // Assert: 5 * 2 = 10 bytes
        assert_eq!(result.len(), 10);
        assert_eq!(i16::from_le_bytes(result[0..2].try_into().unwrap()), -32768);
        assert_eq!(i16::from_le_bytes(result[2..4].try_into().unwrap()), -1);
        assert_eq!(i16::from_le_bytes(result[4..6].try_into().unwrap()), 0);
        assert_eq!(i16::from_le_bytes(result[6..8].try_into().unwrap()), 1);
        assert_eq!(i16::from_le_bytes(result[8..10].try_into().unwrap()), 32767);
    }

    // ── pack_u16_from_i32: multiple values correct byte layout ─────────

    #[test]
    fn pack_u16_from_i32_multiple_values_layout() {
        // Arrange: u16 values at boundaries: 0, 1, 32767, 65535
        let data = vec![0i32, 1, 32767, 65535];
        // Act
        let result = pack_u16_from_i32(data, 4, "u16layout").unwrap();
        // Assert: 4 * 2 = 8 bytes
        assert_eq!(result.len(), 8);
        assert_eq!(u16::from_le_bytes(result[0..2].try_into().unwrap()), 0);
        assert_eq!(u16::from_le_bytes(result[2..4].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(result[4..6].try_into().unwrap()), 32767);
        assert_eq!(u16::from_le_bytes(result[6..8].try_into().unwrap()), 65535);
    }

    // ── build_tensor_bytes: raw_data with Uint4 even packing ───────────

    #[test]
    fn build_tensor_bytes_raw_data_uint4_even_packing() {
        // Arrange: 6 Uint4 elements = 3 bytes packed
        let raw = Bytes::from_static(&[0x12u8, 0x34, 0x56]);
        let input = make_input(OnnxType::Uint4, 6, raw.clone(), "u4raw");
        // Act
        let result = build_tensor_bytes(input).unwrap();
        // Assert: raw_data passes through unchanged
        assert_eq!(result, raw);
    }
}
