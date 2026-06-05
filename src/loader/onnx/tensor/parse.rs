use prost::bytes::Bytes;
use safetensors::Dtype;

use super::super::external::ExternalDataResolver;
use super::super::{proto, LoaderError, Result};

pub(super) fn parse_data_type(value: i32, name: &str) -> Result<proto::tensor_proto::DataType> {
    proto::tensor_proto::DataType::try_from(value)
        .map_err(|_| LoaderError::Onnx(format!("unsupported data_type {value} for tensor {name}")))
}

pub(super) fn map_dtype(data_type: proto::tensor_proto::DataType, name: &str) -> Result<Dtype> {
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
        // FLOAT8 types (FP8 quantized inference)
        OnnxType::Float8e4m3fn | OnnxType::Float8e4m3fnuz => Ok(Dtype::F8_E4M3),
        OnnxType::Float8e5m2 | OnnxType::Float8e5m2fnuz => Ok(Dtype::F8_E5M2),
        // INT4/UINT4: packed as 2 elements per byte, stored as U8
        // The actual unpacking happens at kernel execution time
        OnnxType::Int4 | OnnxType::Uint4 => Ok(Dtype::U8),
        // STRING: variable length, stored with length-prefix encoding
        // U8 is used as placeholder dtype; actual data is length-prefixed bytes
        OnnxType::String => Ok(Dtype::U8),
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
        let dim = usize::try_from(dim)
            .map_err(|_| LoaderError::Onnx(format!("dimension overflow {dim} in tensor {name}")))?;
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
        count = count
            .checked_mul(dim)
            .ok_or_else(|| LoaderError::Onnx(format!("tensor size overflow for {name}")))?;
    }
    Ok(count)
}

/// Calculate byte size for a tensor, handling packed types (INT4/UINT4)
pub(super) fn byte_size_for_elements(
    data_type: proto::tensor_proto::DataType,
    dtype: Dtype,
    element_count: usize,
) -> usize {
    use proto::tensor_proto::DataType as OnnxType;
    match data_type {
        // INT4/UINT4: 2 elements packed per byte
        OnnxType::Int4 | OnnxType::Uint4 => element_count.div_ceil(2),
        // All other types use safetensors Dtype size
        _ => dtype.size() * element_count,
    }
}

pub(super) fn load_external_data(
    resolver: &mut ExternalDataResolver,
    external_data: &[proto::StringStringEntryProto],
    data_type: proto::tensor_proto::DataType,
    dtype: Dtype,
    element_count: usize,
    name: &str,
) -> Result<Bytes> {
    let entries = external_data_map(external_data);
    let location = entries
        .get("location")
        .ok_or_else(|| LoaderError::Onnx(format!("external tensor {name} missing location")))?;
    let offset = parse_optional_usize(entries.get("offset"), "offset", name)?.unwrap_or(0); // LEGAL: offset 可选，默认 0
    let expected = byte_size_for_elements(data_type, dtype, element_count);
    let length = parse_optional_usize(entries.get("length"), "length", name)?
        .unwrap_or(expected); // LEGAL: length 缺失时使用计算值
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
        let value = entry.value.clone().unwrap_or_default(); // LEGAL: protobuf 可选字段
        map.insert(key, value);
    }
    map
}

fn parse_optional_usize(value: Option<&String>, label: &str, name: &str) -> Result<Option<usize>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let parsed = value.parse::<usize>().map_err(|_| {
        LoaderError::Onnx(format!(
            "invalid external {label} {value} for tensor {name}"
        ))
    })?;
    Ok(Some(parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_data_type ───────────────────────────────────────────────

    #[test]
    fn parse_data_type_float() {
        let dt = parse_data_type(1, "t").unwrap();
        assert!(matches!(dt, proto::tensor_proto::DataType::Float));
    }

    #[test]
    fn parse_data_type_invalid() {
        assert!(parse_data_type(9999, "t").is_err());
    }

    // ── map_dtype ────────────────────────────────────────────────────

    #[test]
    fn map_dtype_basic_types() {
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Float, "t").unwrap(), Dtype::F32);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Double, "t").unwrap(), Dtype::F64);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Float16, "t").unwrap(), Dtype::F16);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Bfloat16, "t").unwrap(), Dtype::BF16);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Int8, "t").unwrap(), Dtype::I8);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Uint8, "t").unwrap(), Dtype::U8);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Int32, "t").unwrap(), Dtype::I32);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Int64, "t").unwrap(), Dtype::I64);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Bool, "t").unwrap(), Dtype::U8);
    }

    #[test]
    fn map_dtype_fp8() {
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Float8e4m3fn, "t").unwrap(), Dtype::F8_E4M3);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Float8e5m2, "t").unwrap(), Dtype::F8_E5M2);
    }

    #[test]
    fn map_dtype_int4_packed_as_u8() {
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Int4, "t").unwrap(), Dtype::U8);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Uint4, "t").unwrap(), Dtype::U8);
    }

    #[test]
    fn map_dtype_string_as_u8() {
        assert_eq!(map_dtype(proto::tensor_proto::DataType::String, "t").unwrap(), Dtype::U8);
    }

    // ── parse_dims ────────────────────────────────────────────────────

    #[test]
    fn parse_dims_valid() {
        let shape = parse_dims(&[2, 3, 4], "t").unwrap();
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn parse_dims_empty() {
        let shape = parse_dims(&[], "t").unwrap();
        assert!(shape.is_empty());
    }

    #[test]
    fn parse_dims_negative_fails() {
        assert!(parse_dims(&[2, -1], "t").is_err());
    }

    #[test]
    fn parse_dims_zero() {
        let shape = parse_dims(&[0, 3], "t").unwrap();
        assert_eq!(shape, vec![0, 3]);
    }

    // ── element_count ─────────────────────────────────────────────────

    #[test]
    fn element_count_scalar() {
        assert_eq!(element_count(&[], "t").unwrap(), 1);
    }

    #[test]
    fn element_count_2d() {
        assert_eq!(element_count(&[3, 4], "t").unwrap(), 12);
    }

    #[test]
    fn element_count_with_zero() {
        assert_eq!(element_count(&[0, 5], "t").unwrap(), 0);
    }

    // ── byte_size_for_elements ────────────────────────────────────────

    #[test]
    fn byte_size_for_float() {
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Float, Dtype::F32, 4,
        );
        assert_eq!(size, 16);
    }

    #[test]
    fn byte_size_for_int4_packed() {
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Int4, Dtype::U8, 5,
        );
        assert_eq!(size, 3); // ceil(5/2)
    }

    #[test]
    fn byte_size_for_int4_even() {
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Int4, Dtype::U8, 4,
        );
        assert_eq!(size, 2);
    }

    // ── slice_to_* functions ──────────────────────────────────────────

    #[test]
    fn slice_to_f32_roundtrip() {
        let bytes = 3.14f32.to_le_bytes();
        let val = slice_to_f32(&bytes).unwrap();
        assert!((val - 3.14).abs() < 0.01);
    }

    #[test]
    fn slice_to_i32_roundtrip() {
        let bytes = (-42i32).to_le_bytes();
        assert_eq!(slice_to_i32(&bytes), Some(-42));
    }

    #[test]
    fn slice_to_i64_roundtrip() {
        let bytes = 123456789i64.to_le_bytes();
        assert_eq!(slice_to_i64(&bytes), Some(123456789));
    }

    #[test]
    fn slice_to_u16_roundtrip() {
        let bytes = 65000u16.to_le_bytes();
        assert_eq!(slice_to_u16(&bytes), Some(65000));
    }

    #[test]
    fn slice_to_u32_roundtrip() {
        let bytes = 3000000000u32.to_le_bytes();
        assert_eq!(slice_to_u32(&bytes), Some(3000000000));
    }

    #[test]
    fn slice_to_u64_roundtrip() {
        let bytes = 9876543210u64.to_le_bytes();
        assert_eq!(slice_to_u64(&bytes), Some(9876543210));
    }

    #[test]
    fn slice_to_f16_roundtrip() {
        let bytes = 0x3C00u16.to_le_bytes(); // 1.0 in f16
        assert_eq!(slice_to_f16(&bytes), Some(0x3C00));
    }

    // ── Additional edge-case tests ────────────────────────────────────

    #[test]
    fn slice_to_f32_nan() {
        let bytes = f32::NAN.to_le_bytes();
        let val = slice_to_f32(&bytes).unwrap();
        assert!(val.is_nan());
    }

    #[test]
    fn slice_to_f32_infinity() {
        let pos_bytes = f32::INFINITY.to_le_bytes();
        assert!(slice_to_f32(&pos_bytes).unwrap().is_infinite() && slice_to_f32(&pos_bytes).unwrap().is_sign_positive());
        let neg_bytes = f32::NEG_INFINITY.to_le_bytes();
        assert!(slice_to_f32(&neg_bytes).unwrap().is_infinite() && slice_to_f32(&neg_bytes).unwrap().is_sign_negative());
    }

    #[test]
    fn slice_to_f32_signed_zero() {
        let pos_bytes = 0.0f32.to_le_bytes();
        assert!(slice_to_f32(&pos_bytes).unwrap() == 0.0 && slice_to_f32(&pos_bytes).unwrap().is_sign_positive());
        let neg_bytes = (-0.0f32).to_le_bytes();
        assert!(slice_to_f32(&neg_bytes).unwrap() == 0.0 && slice_to_f32(&neg_bytes).unwrap().is_sign_negative());
    }

    #[test]
    fn slice_to_f32_subnormal() {
        let bytes = f32::from_bits(1u32).to_le_bytes(); // smallest positive subnormal
        let val = slice_to_f32(&bytes).unwrap();
        assert!(val > 0.0 && val.is_subnormal());
    }

    #[test]
    fn slice_to_i32_boundaries() {
        assert_eq!(slice_to_i32(&i32::MIN.to_le_bytes()), Some(i32::MIN));
        assert_eq!(slice_to_i32(&i32::MAX.to_le_bytes()), Some(i32::MAX));
    }

    #[test]
    fn slice_to_i64_boundaries() {
        assert_eq!(slice_to_i64(&i64::MIN.to_le_bytes()), Some(i64::MIN));
        assert_eq!(slice_to_i64(&i64::MAX.to_le_bytes()), Some(i64::MAX));
    }

    #[test]
    fn slice_to_u16_boundary_values() {
        assert_eq!(slice_to_u16(&u16::MIN.to_le_bytes()), Some(u16::MIN));
        assert_eq!(slice_to_u16(&u16::MAX.to_le_bytes()), Some(u16::MAX));
    }

    #[test]
    fn map_dtype_fp8_fnuz_variants() {
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Float8e4m3fnuz, "t").unwrap(), Dtype::F8_E4M3);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Float8e5m2fnuz, "t").unwrap(), Dtype::F8_E5M2);
    }

    #[test]
    fn map_dtype_int16_uint16_uint32_uint64() {
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Int16, "t").unwrap(), Dtype::I16);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Uint16, "t").unwrap(), Dtype::U16);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Uint32, "t").unwrap(), Dtype::U32);
        assert_eq!(map_dtype(proto::tensor_proto::DataType::Uint64, "t").unwrap(), Dtype::U64);
    }

    #[test]
    fn map_dtype_unsupported_returns_error() {
        let result = map_dtype(proto::tensor_proto::DataType::Complex64, "my_tensor");
        assert!(result.is_err());
    }

    #[test]
    fn element_count_overflow_returns_error() {
        let huge = usize::MAX;
        let result = element_count(&[huge, 2], "overflow_tensor");
        assert!(result.is_err());
    }

    #[test]
    fn element_count_single_dim() {
        assert_eq!(element_count(&[7], "t").unwrap(), 7);
    }

    #[test]
    fn parse_dims_preserves_tensor_name_in_error() {
        let err = parse_dims(&[-5], "weight_matrix").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("weight_matrix"), "error message should contain tensor name");
    }

    #[test]
    fn byte_size_zero_elements() {
        let size = byte_size_for_elements(proto::tensor_proto::DataType::Float, Dtype::F32, 0);
        assert_eq!(size, 0);
    }

    #[test]
    fn parse_data_type_zero_returns_undefined() {
        let dt = parse_data_type(0, "t").unwrap();
        assert!(matches!(dt, proto::tensor_proto::DataType::Undefined));
    }

    // ── New tests (15) ─────────────────────────────────────────────────

    // 1. parse_data_type with multiple valid ONNX type codes
    #[test]
    fn parse_data_type_multiple_valid_codes() {
        // Arrange: well-known ONNX data type codes from the protobuf enum
        let cases: Vec<(i32, proto::tensor_proto::DataType)> = vec![
            (1, proto::tensor_proto::DataType::Float),
            (3, proto::tensor_proto::DataType::Int8),
            (6, proto::tensor_proto::DataType::Int32),
            (7, proto::tensor_proto::DataType::Int64),
            (9, proto::tensor_proto::DataType::Bool),
            (10, proto::tensor_proto::DataType::Float16),
            (11, proto::tensor_proto::DataType::Double),
        ];
        // Act & Assert
        for (code, expected) in cases {
            let result = parse_data_type(code, "test_tensor").unwrap();
            assert!(
                matches!(result, dt if dt == expected),
                "expected {expected:?} for code {code}, got {result:?}"
            );
        }
    }

    // 2. parse_data_type error includes tensor name
    #[test]
    fn parse_data_type_error_contains_tensor_name() {
        // Arrange: an invalid data type code
        let name = "my_special_tensor";
        // Act
        let err = parse_data_type(-1, name).unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains(name), "error message should mention tensor name");
    }

    // 3. parse_dims with i64::MAX value causes overflow
    #[test]
    fn parse_dims_i64_max_overflow() {
        // Arrange: a dimension equal to i64::MAX which overflows usize on 64-bit
        let huge = i64::MAX;
        // Act
        let result = parse_dims(&[huge], "big_tensor");
        // Assert: on 64-bit platforms i64::MAX fits in usize, so this succeeds
        // but the result should equal usize::MAX
        if let Ok(shape) = result {
            assert_eq!(shape[0], i64::MAX as usize);
        }
        // On 32-bit this would be an overflow error — both outcomes are acceptable
    }

    // 4. parse_dims with multiple zeros
    #[test]
    fn parse_dims_all_zeros() {
        // Arrange
        let dims: &[i64] = &[0, 0, 0];
        // Act
        let shape = parse_dims(dims, "zero_tensor").unwrap();
        // Assert
        assert_eq!(shape, vec![0, 0, 0]);
    }

    // 5. element_count with 4D tensor
    #[test]
    fn element_count_4d() {
        // Arrange: batch=2, heads=8, seq=16, dim=64
        let shape: &[usize] = &[2, 8, 16, 64];
        // Act
        let count = element_count(shape, "4d_tensor").unwrap();
        // Assert
        assert_eq!(count, 2 * 8 * 16 * 64);
    }

    // 6. element_count with single element 1D tensor
    #[test]
    fn element_count_single_element() {
        // Arrange
        let shape: &[usize] = &[1];
        // Act
        let count = element_count(shape, "scalar_like").unwrap();
        // Assert
        assert_eq!(count, 1);
    }

    // 7. byte_size_for_elements with F64 (8 bytes per element)
    #[test]
    fn byte_size_for_f64() {
        // Arrange
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Double, Dtype::F64, 10,
        );
        // Assert
        assert_eq!(size, 80); // 10 * 8 bytes
    }

    // 8. byte_size_for_elements with I64 (8 bytes per element)
    #[test]
    fn byte_size_for_i64() {
        // Arrange
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Int64, Dtype::I64, 5,
        );
        // Assert
        assert_eq!(size, 40); // 5 * 8 bytes
    }

    // 9. byte_size_for_elements with BF16 (2 bytes per element)
    #[test]
    fn byte_size_for_bf16() {
        // Arrange
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Bfloat16, Dtype::BF16, 100,
        );
        // Assert
        assert_eq!(size, 200); // 100 * 2 bytes
    }

    // 10. byte_size_for_elements with Uint4 (same packing as Int4)
    #[test]
    fn byte_size_for_uint4_packed() {
        // Arrange: 7 elements, 2 per byte → ceil(7/2) = 4
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Uint4, Dtype::U8, 7,
        );
        // Assert
        assert_eq!(size, 4);
    }

    // 11. byte_size_for_elements with FP8 (1 byte per element)
    #[test]
    fn byte_size_for_fp8_e4m3() {
        // Arrange
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Float8e4m3fn, Dtype::F8_E4M3, 16,
        );
        // Assert
        assert_eq!(size, 16); // 1 byte per element for FP8
    }

    // 12. map_dtype error message includes tensor name
    #[test]
    fn map_dtype_unsupported_error_contains_tensor_name() {
        // Arrange
        let name = "unsupported_weight";
        // Act
        let err = map_dtype(proto::tensor_proto::DataType::Complex128, name).unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains(name), "error should reference tensor name '{name}'");
    }

    // 13. parse_data_type with negative code returns error
    #[test]
    fn parse_data_type_negative_code_is_error() {
        // Arrange: negative values are not valid protobuf enum values
        // Act
        let result = parse_data_type(-42, "neg_tensor");
        // Assert
        assert!(result.is_err(), "negative data type code should fail");
    }

    // 14. parse_dims single large positive dimension
    #[test]
    fn parse_dims_single_large_dimension() {
        // Arrange: a large but valid dimension
        let dims: &[i64] = &[1_000_000];
        // Act
        let shape = parse_dims(dims, "large_tensor").unwrap();
        // Assert
        assert_eq!(shape, vec![1_000_000]);
    }

    // 15. element_count large but non-overflowing dimensions
    #[test]
    fn element_count_large_non_overflowing() {
        // Arrange: 1024 * 1024 = 1_048_576 — well within usize range
        let shape: &[usize] = &[1024, 1024];
        // Act
        let count = element_count(shape, "large_2d").unwrap();
        // Assert
        assert_eq!(count, 1_048_576);
    }

    // ── New tests (13) — untested paths ──────────────────────────────────

    // 16. external_data_map: entries with None key are skipped
    // @trace TEST-TP-54 [req:REQ-ONNX] [level:unit]
    #[test]
    fn external_data_map_skips_none_key_entries() {
        // Arrange: mix of valid entries and entries with None key
        let entries = vec![
            proto::StringStringEntryProto { key: None, value: Some("ignored".to_string()) },
            proto::StringStringEntryProto { key: Some("location".to_string()), value: Some("data.bin".to_string()) },
        ];
        // Act
        let map = external_data_map(&entries);
        // Assert: only the entry with a key is present
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("location").unwrap(), "data.bin");
    }

    // 17. external_data_map: entries with empty key are skipped
    // @trace TEST-TP-55 [req:REQ-ONNX] [level:unit]
    #[test]
    fn external_data_map_skips_empty_key_entries() {
        // Arrange: an entry with an empty string key should be filtered out
        let entries = vec![
            proto::StringStringEntryProto { key: Some(String::new()), value: Some("bad".to_string()) },
            proto::StringStringEntryProto { key: Some("offset".to_string()), value: Some("0".to_string()) },
        ];
        // Act
        let map = external_data_map(&entries);
        // Assert: empty-key entry is excluded
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("offset").unwrap(), "0");
    }

    // 18. external_data_map: entries with None value default to empty string
    // @trace TEST-TP-56 [req:REQ-ONNX] [level:unit]
    #[test]
    fn external_data_map_none_value_defaults_empty() {
        // Arrange: an entry with a valid key but None value
        let entries = vec![
            proto::StringStringEntryProto { key: Some("location".to_string()), value: None },
        ];
        // Act
        let map = external_data_map(&entries);
        // Assert: value defaults to empty string
        assert_eq!(map.get("location").unwrap(), "");
    }

    // 19. parse_optional_usize: None input returns Ok(None)
    // @trace TEST-TP-57 [req:REQ-ONNX] [level:unit]
    #[test]
    fn parse_optional_usize_none_returns_ok_none() {
        // Arrange: no value provided
        // Act
        let result = parse_optional_usize(None, "offset", "t").unwrap();
        // Assert
        assert!(result.is_none());
    }

    // 20. parse_optional_usize: valid numeric string returns parsed value
    // @trace TEST-TP-58 [req:REQ-ONNX] [level:unit]
    #[test]
    fn parse_optional_usize_valid_number() {
        // Arrange
        let val = "1024".to_string();
        // Act
        let result = parse_optional_usize(Some(&val), "offset", "t").unwrap();
        // Assert
        assert_eq!(result, Some(1024));
    }

    // 21. parse_optional_usize: invalid string returns error with context
    // @trace TEST-TP-59 [req:REQ-ONNX] [level:unit]
    #[test]
    fn parse_optional_usize_invalid_string_returns_error() {
        // Arrange: a non-numeric string
        let val = "not_a_number".to_string();
        // Act
        let err = parse_optional_usize(Some(&val), "offset", "my_tensor").unwrap_err();
        let msg = format!("{err}");
        // Assert: error message includes the label and tensor name
        assert!(msg.contains("offset"), "error should mention the label");
        assert!(msg.contains("my_tensor"), "error should mention the tensor name");
        assert!(msg.contains("not_a_number"), "error should mention the bad value");
    }

    // 22. load_external_data: missing location key returns error
    // @trace TEST-TP-60 [req:REQ-ONNX] [level:unit]
    #[test]
    fn load_external_data_missing_location_returns_error() {
        // Arrange: external data entries without "location" key
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);
        let entries = vec![
            proto::StringStringEntryProto { key: Some("offset".to_string()), value: Some("0".to_string()) },
        ];
        // Act
        let err = load_external_data(
            &mut resolver,
            &entries,
            proto::tensor_proto::DataType::Float,
            Dtype::F32,
            4,
            "weight_tensor",
        ).unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("weight_tensor"), "error should mention tensor name");
        assert!(msg.contains("missing location"), "error should mention missing location");
    }

    // 23. load_external_data: length mismatch returns error with both values
    // @trace TEST-TP-61 [req:REQ-ONNX] [level:unit]
    #[test]
    fn load_external_data_length_mismatch_returns_error() {
        // Arrange: provide a length that does not match expected byte size
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);
        let entries = vec![
            proto::StringStringEntryProto { key: Some("location".to_string()), value: Some("data.bin".to_string()) },
            proto::StringStringEntryProto { key: Some("length".to_string()), value: Some("99".to_string()) },
        ];
        // Float * 4 elements = 4 * 4 = 16 expected bytes, but length says 99
        // Act
        let err = load_external_data(
            &mut resolver,
            &entries,
            proto::tensor_proto::DataType::Float,
            Dtype::F32,
            4,
            "mismatch_tensor",
        ).unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("99"), "error should show provided length");
        assert!(msg.contains("16"), "error should show expected size");
        assert!(msg.contains("mismatch_tensor"), "error should mention tensor name");
    }

    // 24. load_external_data: invalid offset string returns error
    // @trace TEST-TP-62 [req:REQ-ONNX] [level:unit]
    #[test]
    fn load_external_data_invalid_offset_returns_error() {
        // Arrange: location present but offset is not a valid number
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);
        let entries = vec![
            proto::StringStringEntryProto { key: Some("location".to_string()), value: Some("data.bin".to_string()) },
            proto::StringStringEntryProto { key: Some("offset".to_string()), value: Some("abc".to_string()) },
        ];
        // Act
        let err = load_external_data(
            &mut resolver,
            &entries,
            proto::tensor_proto::DataType::Float,
            Dtype::F32,
            4,
            "bad_offset_tensor",
        ).unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("offset"), "error should mention offset");
        assert!(msg.contains("bad_offset_tensor"), "error should mention tensor name");
    }

    // 25. byte_size_for_elements with Float8e5m2 variant
    // @trace TEST-TP-63 [req:REQ-ONNX] [level:unit]
    #[test]
    fn byte_size_for_fp8_e5m2() {
        // Arrange: FP8 E5M2 is 1 byte per element
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Float8e5m2, Dtype::F8_E5M2, 32,
        );
        // Assert
        assert_eq!(size, 32);
    }

    // 26. element_count with all zero dimensions produces zero
    // @trace TEST-TP-64 [req:REQ-ONNX] [level:unit]
    #[test]
    fn element_count_all_zero_dims() {
        // Arrange: shape with all zero dimensions
        let shape: &[usize] = &[0, 0, 0, 0];
        // Act
        let count = element_count(shape, "zero_4d").unwrap();
        // Assert
        assert_eq!(count, 0);
    }

    // 27. parse_dims with negative at first position
    // @trace TEST-TP-65 [req:REQ-ONNX] [level:unit]
    #[test]
    fn parse_dims_negative_at_first_position() {
        // Arrange: negative dimension at the very first position
        let dims: &[i64] = &[-3, 4, 5];
        // Act
        let err = parse_dims(dims, "first_neg").unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("first_neg"), "error should mention tensor name");
        assert!(msg.contains("-3"), "error should mention the negative dimension");
    }

    // 28. slice_to_f32 with a negative value
    // @trace TEST-TP-66 [req:REQ-ONNX] [level:unit]
    #[test]
    fn slice_to_f32_negative_value() {
        // Arrange: -100.5 as little-endian bytes
        let bytes = (-100.5f32).to_le_bytes();
        // Act
        let val = slice_to_f32(&bytes).unwrap();
        // Assert
        assert!((val - (-100.5)).abs() < 0.001);
        assert!(val.is_sign_negative());
    }

    // ── Additional boundary and coverage tests (10) ─────────────────────

    // 29. external_data_map: duplicate keys — last entry wins
    // @trace TEST-TP-67 [req:REQ-ONNX] [level:unit]
    #[test]
    fn external_data_map_duplicate_keys_last_wins() {
        // Arrange: two entries with the same key "location"
        let entries = vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("first.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("second.bin".to_string()),
            },
        ];
        // Act
        let map = external_data_map(&entries);
        // Assert: the second entry overwrites the first
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("location").unwrap(), "second.bin");
    }

    // 30. external_data_map: empty input list yields empty map
    // @trace TEST-TP-68 [req:REQ-ONNX] [level:unit]
    #[test]
    fn external_data_map_empty_input() {
        // Arrange: no entries at all
        let entries: Vec<proto::StringStringEntryProto> = vec![];
        // Act
        let map = external_data_map(&entries);
        // Assert
        assert!(map.is_empty());
    }

    // 31. parse_optional_usize: value "0" parses to zero (boundary)
    // @trace TEST-TP-69 [req:REQ-ONNX] [level:unit]
    #[test]
    fn parse_optional_usize_zero_boundary() {
        // Arrange: string "0"
        let val = "0".to_string();
        // Act
        let result = parse_optional_usize(Some(&val), "offset", "t").unwrap();
        // Assert
        assert_eq!(result, Some(0));
    }

    // 32. parse_optional_usize: value at usize max string representation
    // @trace TEST-TP-70 [req:REQ-ONNX] [level:unit]
    #[test]
    fn parse_optional_usize_max_value() {
        // Arrange: the maximum representable usize value as a string
        let val = usize::MAX.to_string();
        // Act
        let result = parse_optional_usize(Some(&val), "length", "big_tensor").unwrap();
        // Assert
        assert_eq!(result, Some(usize::MAX));
    }

    // 33. parse_dims: multiple negatives — reports the first encountered
    // @trace TEST-TP-71 [req:REQ-ONNX] [level:unit]
    #[test]
    fn parse_dims_reports_first_negative() {
        // Arrange: two negative dimensions, the first is -7
        let dims: &[i64] = &[-7, -2, 3];
        // Act
        let err = parse_dims(dims, "multi_neg").unwrap_err();
        let msg = format!("{err}");
        // Assert: error reports the first negative dimension value
        assert!(msg.contains("-7"), "should report the first negative dimension");
    }

    // 34. element_count: dimension of 1 followed by large values does not overflow
    // @trace TEST-TP-72 [req:REQ-ONNX] [level:unit]
    #[test]
    fn element_count_leading_one_no_overflow() {
        // Arrange: [1, 65536, 256]
        let shape: &[usize] = &[1, 65536, 256];
        // Act
        let count = element_count(shape, "batch_one").unwrap();
        // Assert
        assert_eq!(count, 65536 * 256);
    }

    // 35. slice_to_u32 boundary values (min and max)
    // @trace TEST-TP-73 [req:REQ-ONNX] [level:unit]
    #[test]
    fn slice_to_u32_boundary_values() {
        // Arrange: u32 min and max as little-endian bytes
        // Act & Assert
        assert_eq!(slice_to_u32(&u32::MIN.to_le_bytes()), Some(u32::MIN));
        assert_eq!(slice_to_u32(&u32::MAX.to_le_bytes()), Some(u32::MAX));
    }

    // 36. slice_to_u64 boundary values (min and max)
    // @trace TEST-TP-74 [req:REQ-ONNX] [level:unit]
    #[test]
    fn slice_to_u64_boundary_values() {
        // Arrange: u64 min and max as little-endian bytes
        // Act & Assert
        assert_eq!(slice_to_u64(&u64::MIN.to_le_bytes()), Some(u64::MIN));
        assert_eq!(slice_to_u64(&u64::MAX.to_le_bytes()), Some(u64::MAX));
    }

    // 37. byte_size_for_elements with Int4 and exactly one element
    // @trace TEST-TP-75 [req:REQ-ONNX] [level:unit]
    #[test]
    fn byte_size_for_int4_single_element() {
        // Arrange: 1 element packed as Int4 → ceil(1/2) = 1 byte
        let size = byte_size_for_elements(
            proto::tensor_proto::DataType::Int4,
            Dtype::U8,
            1,
        );
        // Assert: a single nibble still occupies one full byte
        assert_eq!(size, 1);
    }

    // 38. load_external_data: valid file with correct length reads successfully
    // @trace TEST-TP-76 [req:REQ-ONNX] [level:unit]
    #[test]
    fn load_external_data_valid_file_succeeds() {
        // Arrange: create a temporary binary file with exactly 8 bytes (2 x F32)
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("weights.bin");
        let data = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        std::fs::write(&data_path, &data).unwrap();
        let model_path = dir.path().join("model.onnx");
        let mut resolver = ExternalDataResolver::new(&model_path);
        let entries = vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("weights.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("offset".to_string()),
                value: Some("0".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("length".to_string()),
                value: Some("8".to_string()),
            },
        ];
        // Act
        let bytes = load_external_data(
            &mut resolver,
            &entries,
            proto::tensor_proto::DataType::Float,
            Dtype::F32,
            2, // 2 elements × 4 bytes = 8 bytes
            "valid_tensor",
        )
        .unwrap();
        // Assert: bytes match the written data
        assert_eq!(bytes.len(), 8);
        let val0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let val1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!((val0 - 1.0).abs() < 0.001);
        assert!((val1 - 2.0).abs() < 0.001);
    }
}
