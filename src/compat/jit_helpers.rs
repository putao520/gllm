//! JIT helper utilities shared across inference paths.

use gllm_kernels::types::DType;

/// Zero-copy reinterpret `&[u8]` as `&[f32]`.
#[inline]
fn bytes_as_f32(s: &[u8]) -> &[f32] {
    if s.is_empty() {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len() / std::mem::size_of::<f32>()) }
}

/// Convert typed bytes (F16/BF16/F32) to Vec<f32>.
/// F32: zero-copy reinterpret. F16/BF16: element-wise conversion.
pub(crate) fn typed_bytes_to_f32(data: &[u8], dtype: DType) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }
    match dtype {
        DType::F32 => bytes_as_f32(data).to_vec(),
        DType::F16 => {
            data.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        DType::BF16 => {
            data.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        _ => bytes_as_f32(data).to_vec(),
    }
}

/// Derive computation DType from a `GeneratorForwardConfig`.
///
/// Returns the model's native dtype — F16/BF16/F32.
/// ARCH-DTYPE-ADAPTIVE: 禁止硬编码返回 F32。
#[inline]
#[allow(dead_code)]
pub(crate) fn computation_dtype_from_config(
    config: &crate::engine::executor::GeneratorForwardConfig,
) -> DType {
    config.dtype()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor_types::GeneratorForwardConfig;

    // ── typed_bytes_to_f32: F32 path ──

    #[test]
    fn typed_bytes_to_f32_f32_identity() {
        let values: Vec<f32> = vec![1.0, -2.5, 3.14, 0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        assert_eq!(result, values);
    }

    #[test]
    fn typed_bytes_to_f32_f32_single_element() {
        let value: f32 = 42.0;
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn typed_bytes_to_f32_f32_negative_values() {
        let values: Vec<f32> = vec![-1.0, -100.5, -0.001, -999.999];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        assert_eq!(result, values);
    }

    // ── typed_bytes_to_f32: F16 path ──

    #[test]
    fn typed_bytes_to_f32_f16_roundtrip() {
        let f16_val = half::f16::from_f32(3.5);
        let bytes: Vec<u8> = f16_val.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.5).abs() < 1e-3);
    }

    #[test]
    fn typed_bytes_to_f32_f16_multiple_elements() {
        let f16_vals: Vec<half::f16> = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(-1.0),
        ];
        let bytes: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-3);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn typed_bytes_to_f32_f16_zero() {
        let f16_zero = half::f16::from_f32(0.0);
        let bytes: Vec<u8> = f16_zero.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0.0);
    }

    // ── typed_bytes_to_f32: BF16 path ──

    #[test]
    fn typed_bytes_to_f32_bf16_roundtrip() {
        let bf16_val = half::bf16::from_f32(2.75);
        let bytes: Vec<u8> = bf16_val.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.75).abs() < 0.01);
    }

    #[test]
    fn typed_bytes_to_f32_bf16_multiple_elements() {
        let bf16_vals: Vec<half::bf16> = vec![
            half::bf16::from_f32(0.5),
            half::bf16::from_f32(1.5),
            half::bf16::from_f32(-0.25),
        ];
        let bytes: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 0.01);
        assert!((result[1] - 1.5).abs() < 0.01);
        assert!((result[2] - (-0.25)).abs() < 0.01);
    }

    // ── typed_bytes_to_f32: fallback path (non-F32/F16/BF16 dtypes) ──

    #[test]
    fn typed_bytes_to_f32_u8_falls_back_to_f32_reinterpret() {
        let values: Vec<f32> = vec![1.0, 2.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = typed_bytes_to_f32(&bytes, DType::U8);
        assert_eq!(result, values);
    }

    #[test]
    fn typed_bytes_to_f32_f8e4m3_falls_back_to_f32_reinterpret() {
        let values: Vec<f32> = vec![0.5, -0.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = typed_bytes_to_f32(&bytes, DType::F8E4M3);
        assert_eq!(result, values);
    }

    #[test]
    fn typed_bytes_to_f32_f8e5m2_falls_back_to_f32_reinterpret() {
        let values: Vec<f32> = vec![100.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = typed_bytes_to_f32(&bytes, DType::F8E5M2);
        assert_eq!(result, values);
    }

    // ── typed_bytes_to_f32: truncated input (odd bytes for F16/BF16) ──

    #[test]
    fn typed_bytes_to_f32_f16_ignores_trailing_byte() {
        let f16_val = half::f16::from_f32(7.0);
        let mut bytes: Vec<u8> = f16_val.to_le_bytes().to_vec();
        bytes.push(0xFF);
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 7.0).abs() < 1e-3);
    }

    // ── computation_dtype_from_config ──

    #[test]
    fn computation_dtype_from_config_returns_f32() {
        let config = GeneratorForwardConfig::default_for_test();
        let dtype = computation_dtype_from_config(&config);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn computation_dtype_from_config_matches_config_dtype_accessor() {
        let config = GeneratorForwardConfig::default_for_test();
        let from_helper = computation_dtype_from_config(&config);
        let from_accessor = config.dtype();
        assert_eq!(from_helper, from_accessor);
    }

    // ── typed_bytes_to_f32: F32 preserves subnormal values ──

    #[test]
    fn typed_bytes_to_f32_f32_preserves_subnormal() {
        let value: f32 = 1.5e-38;
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), value.to_bits());
    }

    // ── typed_bytes_to_f32: F16 special values ──

    #[test]
    fn typed_bytes_to_f32_f16_nan() {
        let f16_nan = half::f16::NAN;
        let bytes: Vec<u8> = f16_nan.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_nan(), "expected NaN, got {}", result[0]);
    }

    #[test]
    fn typed_bytes_to_f32_f16_positive_infinity() {
        let f16_inf = half::f16::INFINITY;
        let bytes: Vec<u8> = f16_inf.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_infinite() && result[0].is_sign_positive(),
            "expected +Inf, got {}", result[0]);
    }

    #[test]
    fn typed_bytes_to_f32_f16_negative_infinity() {
        let f16_neg_inf = half::f16::NEG_INFINITY;
        let bytes: Vec<u8> = f16_neg_inf.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_infinite() && result[0].is_sign_negative(),
            "expected -Inf, got {}", result[0]);
    }

    #[test]
    fn typed_bytes_to_f32_f16_large_set() {
        let f16_vals: Vec<half::f16> = (0..64)
            .map(|i| half::f16::from_f32(i as f32 * 0.1))
            .collect();
        let bytes: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        assert_eq!(result.len(), 64);
        for (i, val) in result.iter().enumerate() {
            let expected = i as f32 * 0.1;
            assert!((val - expected).abs() < 0.01,
                "index {}: expected ~{}, got {}", i, expected, val);
        }
    }

    // ── typed_bytes_to_f32: BF16 special values ──

    #[test]
    fn typed_bytes_to_f32_bf16_zero() {
        let bf16_zero = half::bf16::from_f32(0.0);
        let bytes: Vec<u8> = bf16_zero.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn typed_bytes_to_f32_bf16_negative_values() {
        // BF16 has 7-bit mantissa (~3 decimal digits); use values it can
        // represent exactly or within small tolerance.
        let bf16_vals: Vec<half::bf16> = vec![
            half::bf16::from_f32(-10.0),
            half::bf16::from_f32(-0.125),
            half::bf16::from_f32(-3.5),
        ];
        let bytes: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        assert_eq!(result.len(), 3);
        assert!((result[0] - (-10.0)).abs() < 0.05);
        assert!((result[1] - (-0.125)).abs() < 0.005);
        assert!((result[2] - (-3.5)).abs() < 0.05);
    }

    #[test]
    fn typed_bytes_to_f32_bf16_ignores_trailing_byte() {
        let bf16_val = half::bf16::from_f32(5.25);
        let mut bytes: Vec<u8> = bf16_val.to_le_bytes().to_vec();
        bytes.push(0xAB);
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 5.25).abs() < 0.01);
    }

    // ── typed_bytes_to_f32: empty inputs ──

    #[test]
    fn typed_bytes_to_f32_f32_empty() {
        let result = typed_bytes_to_f32(&[], DType::F32);
        assert!(result.is_empty());
    }

    #[test]
    fn typed_bytes_to_f32_f16_empty() {
        let result = typed_bytes_to_f32(&[], DType::F16);
        assert!(result.is_empty());
    }

    #[test]
    fn typed_bytes_to_f32_bf16_empty() {
        let result = typed_bytes_to_f32(&[], DType::BF16);
        assert!(result.is_empty());
    }

    // ── typed_bytes_to_f32: F32 preserves NaN bit pattern ──

    #[test]
    fn typed_bytes_to_f32_f32_preserves_nan() {
        let nan_bits: u32 = 0x7FC00001; // a quiet NaN with payload
        let value = f32::from_bits(nan_bits);
        assert!(value.is_nan());
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_nan());
        assert_eq!(result[0].to_bits(), nan_bits);
    }

    // ── bytes_as_f32: slice reinterpret tests ──
    // Note: bytes_as_f32 is unsafe and requires 4-byte aligned input for
    // correctness. Tests use Vec<u8>-backed data which is heap-aligned.

    #[test]
    fn bytes_as_f32_single_element() {
        let value: f32 = 3.14159;
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        let result = bytes_as_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], value);
    }

    #[test]
    fn bytes_as_f32_multiple_elements() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = bytes_as_f32(&bytes);
        assert_eq!(result.len(), 5);
        assert_eq!(result, values.as_slice());
    }

    #[test]
    fn bytes_as_f32_preserves_negative_and_large() {
        let values: Vec<f32> = vec![-1e10, f32::MAX, f32::MIN];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = bytes_as_f32(&bytes);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], -1e10);
        assert_eq!(result[1], f32::MAX);
        assert_eq!(result[2], f32::MIN);
    }

    #[test]
    fn bytes_as_f32_roundtrip_through_typed_bytes_to_f32() {
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 42.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // bytes_as_f32 gives zero-copy view, typed_bytes_to_f32 copies
        let view = bytes_as_f32(&bytes);
        let copied = typed_bytes_to_f32(&bytes, DType::F32);
        assert_eq!(view, copied.as_slice());
    }

    // ── New tests (TEST-JH-31 through TEST-JH-43) ──

    // @trace TEST-JH-31 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f32_positive_infinity() {
        // Arrange: a single f32 positive infinity encoded as bytes
        let value = f32::INFINITY;
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0].is_infinite() && result[0].is_sign_positive(),
            "expected +Inf, got {}", result[0]);
    }

    // @trace TEST-JH-32 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f32_negative_infinity() {
        // Arrange: a single f32 negative infinity encoded as bytes
        let value = f32::NEG_INFINITY;
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0].is_infinite() && result[0].is_sign_negative(),
            "expected -Inf, got {}", result[0]);
    }

    // @trace TEST-JH-33 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f32_negative_zero() {
        // Arrange: negative zero f32
        let value: f32 = -0.0;
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), value.to_bits(),
            "expected negative zero bit pattern");
    }

    // @trace TEST-JH-34 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f32_short_byte_array_discards_remainder() {
        // Arrange: 5 bytes — one full f32 (4 bytes) + 1 trailing byte
        let value: f32 = 99.5;
        let mut bytes: Vec<u8> = value.to_le_bytes().to_vec();
        bytes.push(0xAB);
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        // Assert: only the first 4 bytes are reinterpreted as one f32
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], value);
    }

    // @trace TEST-JH-35 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f16_negative_zero() {
        // Arrange: F16 negative zero
        let f16_neg_zero = half::f16::from_f32(-0.0);
        assert_ne!(f16_neg_zero.to_bits(), half::f16::from_f32(0.0).to_bits());
        let bytes: Vec<u8> = f16_neg_zero.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0] == 0.0 && result[0].is_sign_negative(),
            "expected negative zero, got {}", result[0]);
    }

    // @trace TEST-JH-36 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f16_single_byte_yields_empty() {
        // Arrange: only 1 byte — not enough for a complete F16 element
        let bytes: Vec<u8> = vec![0x3C];
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        // Assert: chunks_exact(2) discards incomplete trailing chunk
        assert!(result.is_empty());
    }

    // @trace TEST-JH-37 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_bf16_nan() {
        // Arrange: BF16 NaN
        let bf16_nan = half::bf16::NAN;
        let bytes: Vec<u8> = bf16_nan.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0].is_nan(), "expected NaN, got {}", result[0]);
    }

    // @trace TEST-JH-38 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_bf16_positive_infinity() {
        // Arrange: BF16 positive infinity
        let bf16_inf = half::bf16::INFINITY;
        let bytes: Vec<u8> = bf16_inf.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0].is_infinite() && result[0].is_sign_positive(),
            "expected +Inf, got {}", result[0]);
    }

    // @trace TEST-JH-39 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_bf16_negative_infinity() {
        // Arrange: BF16 negative infinity
        let bf16_neg_inf = half::bf16::NEG_INFINITY;
        let bytes: Vec<u8> = bf16_neg_inf.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0].is_infinite() && result[0].is_sign_negative(),
            "expected -Inf, got {}", result[0]);
    }

    // @trace TEST-JH-40 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_bf16_negative_zero() {
        // Arrange: BF16 negative zero
        let bf16_neg_zero = half::bf16::from_f32(-0.0);
        let bytes: Vec<u8> = bf16_neg_zero.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        // Assert
        assert_eq!(result.len(), 1);
        assert!(result[0] == 0.0 && result[0].is_sign_negative(),
            "expected negative zero, got {}", result[0]);
    }

    // @trace TEST-JH-41 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_bf16_large_set() {
        // Arrange: 128 BF16 elements spanning a range of values
        let bf16_vals: Vec<half::bf16> = (0..128)
            .map(|i| half::bf16::from_f32(i as f32 * 0.25 - 16.0))
            .collect();
        let bytes: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        // Assert
        assert_eq!(result.len(), 128);
        for (i, val) in result.iter().enumerate() {
            let expected = i as f32 * 0.25 - 16.0;
            assert!((val - expected).abs() < 0.05,
                "index {}: expected ~{}, got {}", i, expected, val);
        }
    }

    // @trace TEST-JH-42 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f4e2m1_fallback_reinterpret() {
        // Arrange: F4E2M1 (sub-byte) hits the fallback path — reinterpreted as f32 bytes
        let values: Vec<f32> = vec![1.5, -3.0, 0.25];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F4E2M1);
        // Assert: fallback path does zero-copy f32 reinterpret
        assert_eq!(result, values);
    }

    // @trace TEST-JH-43 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f32_preserves_f32_max() {
        // Arrange: f32::MAX — largest representable finite f32
        let value = f32::MAX;
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        // Assert: round-trip preserves the exact bit pattern
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), value.to_bits());
    }

    // ── New tests (TEST-JH-44 through TEST-JH-54) ──

    // @trace TEST-JH-44 [req:REQ-JIT] [level:unit]
    #[test]
    fn bytes_as_f32_empty_slice_yields_empty() {
        // Arrange: zero-length byte slice
        let bytes: &[u8] = &[];
        // Act
        let result = bytes_as_f32(bytes);
        // Assert: 0 bytes / 4 = 0 elements
        assert!(result.is_empty());
    }

    // @trace TEST-JH-45 [req:REQ-JIT] [level:unit]
    #[test]
    fn bytes_as_f32_sub_f32_length_yields_empty() {
        // Arrange: 3 bytes — not enough for a complete f32 (needs 4)
        let bytes: Vec<u8> = vec![0x01, 0x02, 0x03];
        // Act
        let result = bytes_as_f32(&bytes);
        // Assert: integer division 3/4 = 0 elements
        assert!(result.is_empty());
    }

    // @trace TEST-JH-46 [req:REQ-JIT] [level:unit]
    #[test]
    fn bytes_as_f32_length_calculation_matches_byte_count() {
        // Arrange: 20 bytes should yield exactly 5 f32 elements
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(bytes.len(), 20);
        // Act
        let result = bytes_as_f32(&bytes);
        // Assert
        assert_eq!(result.len(), 5);
        assert_eq!(result, values.as_slice());
    }

    // @trace TEST-JH-47 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f6e3m2_fallback_reinterpret() {
        // Arrange: F6E3M2 is a sub-byte dtype that hits the fallback path
        let values: Vec<f32> = vec![2.0, -4.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F6E3M2);
        // Assert: fallback path reinterprets bytes as f32
        assert_eq!(result, values);
    }

    // @trace TEST-JH-48 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f6e2m3_fallback_reinterpret() {
        // Arrange: F6E2M3 is a sub-byte dtype that hits the fallback path
        let values: Vec<f32> = vec![0.125, -0.75];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F6E2M3);
        // Assert: fallback path reinterprets bytes as f32
        assert_eq!(result, values);
    }

    // @trace TEST-JH-49 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_fallback_dtype_empty_input() {
        // Arrange: empty byte slice with U8 dtype (fallback path)
        // Act
        let result = typed_bytes_to_f32(&[], DType::U8);
        // Assert: early return produces empty Vec regardless of dtype
        assert!(result.is_empty());
    }

    // @trace TEST-JH-50 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f16_three_trailing_bytes_yields_one_element() {
        // Arrange: one F16 element (2 bytes) + 3 trailing bytes
        let f16_val = half::f16::from_f32(3.0);
        let mut bytes: Vec<u8> = f16_val.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        assert_eq!(bytes.len(), 5);
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        // Assert: chunks_exact(2) yields 2 elements (bytes[0..2] and bytes[2..4])
        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 1e-3);
    }

    // @trace TEST-JH-51 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_bf16_three_trailing_bytes_yields_one_element() {
        // Arrange: one BF16 element (2 bytes) + 3 trailing bytes
        let bf16_val = half::bf16::from_f32(7.5);
        let mut bytes: Vec<u8> = bf16_val.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0xDD, 0xEE, 0xFF]);
        assert_eq!(bytes.len(), 5);
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::BF16);
        // Assert: chunks_exact(2) yields 2 elements
        assert_eq!(result.len(), 2);
        assert!((result[0] - 7.5).abs() < 0.01);
    }

    // @trace TEST-JH-52 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f32_preserves_subnormal_negative() {
        // Arrange: the most negative subnormal f32 (largest magnitude negative subnormal)
        let value: f32 = f32::from_bits(0x807FFFFFu32); // -max subnormal
        assert!(value.is_subnormal());
        assert!(value.is_sign_negative());
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        // Assert: exact bit-pattern round-trip
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), value.to_bits());
        assert!(result[0].is_subnormal());
    }

    // @trace TEST-JH-53 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f16_max_finite() {
        // Arrange: largest representable finite F16 value (65504.0)
        let f16_max = half::f16::MAX;
        assert_eq!(f16_max.to_f32(), 65504.0);
        let bytes: Vec<u8> = f16_max.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F16);
        // Assert
        assert_eq!(result.len(), 1);
        assert!((result[0] - 65504.0).abs() < 1.0,
            "expected ~65504.0, got {}", result[0]);
    }

    // @trace TEST-JH-54 [req:REQ-JIT] [level:unit]
    #[test]
    fn typed_bytes_to_f32_f32_alternating_sign_nan() {
        // Arrange: a NaN with sign bit set (negative NaN)
        let nan_bits: u32 = 0xFFC00000u32; // negative quiet NaN
        let value = f32::from_bits(nan_bits);
        assert!(value.is_nan());
        assert!(value.is_sign_negative());
        let bytes: Vec<u8> = value.to_le_bytes().to_vec();
        // Act
        let result = typed_bytes_to_f32(&bytes, DType::F32);
        // Assert: NaN preserved with sign bit
        assert_eq!(result.len(), 1);
        assert!(result[0].is_nan());
        assert_eq!(result[0].to_bits(), nan_bits);
    }
}
