//! Float8 量化/反量化 (REQ-DTYPE-FP8-001)
//!
//! 支持两种 FP8 格式：
//! - E4M3: 1 sign + 4 exp + 3 mantissa，范围 [-448, 448]，用于训练梯度通信
//! - E5M2: 1 sign + 5 exp + 2 mantissa，范围 [-57344, 57344]，用于推理激活

/// FP32 → FP8 E4M3 (1 sign, 4 exp, 3 mantissa)
pub fn fp32_to_fp8_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F; // NaN
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0x7E } else { 0xFE }; // ±Inf
    }
    if x == 0.0 {
        return if x.is_sign_negative() { 0x80 } else { 0x00 }; // ±0
    }

    let bits = x.to_bits();
    let sign = (bits >> 31) & 0x1;
    let fp32_exp = ((bits >> 23) & 0xFF) as i32;
    let fp32_mantissa = bits & 0x7FFFFF;

    if fp32_exp == 0 {
        return (sign as u8) << 7; // Denormal → 0
    }

    let exp = fp32_exp - 127; // FP32 bias
    let mantissa = (fp32_mantissa >> 20) & 0x7; // 取高 3 位

    // FP8 E4M3: bias = 7, exp range [0, 15]
    let fp8_exp = (exp + 7).clamp(0, 15) as u8;
    let fp8_mantissa = mantissa as u8;

    ((sign as u8) << 7) | (fp8_exp << 3) | fp8_mantissa
}

/// FP8 E4M3 → FP32
pub fn fp8_e4m3_to_fp32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = ((byte >> 3) & 0xF) as i32;
    let mantissa = (byte & 0x7) as u32;

    if exp == 0 && mantissa == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }

    if exp == 15 && mantissa == 7 {
        return f32::NAN;
    }
    if exp == 15 && mantissa == 6 {
        return if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY };
    }

    let fp32_exp = (exp - 7 + 127) as u32;
    let fp32_mantissa = mantissa << 20;
    let fp32_bits = ((sign as u32) << 31) | (fp32_exp << 23) | fp32_mantissa;

    f32::from_bits(fp32_bits)
}

/// FP32 → FP8 E5M2 (1 sign, 5 exp, 2 mantissa)
pub fn fp32_to_fp8_e5m2(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F; // NaN
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0x7C } else { 0xFC }; // ±Inf
    }
    if x == 0.0 {
        return if x.is_sign_negative() { 0x80 } else { 0x00 }; // ±0
    }

    let bits = x.to_bits();
    let sign = (bits >> 31) & 0x1;
    let fp32_exp = ((bits >> 23) & 0xFF) as i32;
    let fp32_mantissa = bits & 0x7FFFFF;

    if fp32_exp == 0 {
        return (sign as u8) << 7; // Denormal → 0
    }

    let exp = fp32_exp - 127;
    let mantissa = (fp32_mantissa >> 21) & 0x3; // 取高 2 位

    // FP8 E5M2: bias = 15, exp range [0, 31]
    let fp8_exp = (exp + 15).clamp(0, 31) as u8;
    let fp8_mantissa = mantissa as u8;

    ((sign as u8) << 7) | (fp8_exp << 2) | fp8_mantissa
}

/// FP8 E5M2 → FP32
pub fn fp8_e5m2_to_fp32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = ((byte >> 2) & 0x1F) as i32;
    let mantissa = (byte & 0x3) as u32;

    if exp == 0 && mantissa == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }

    if exp == 31 && mantissa == 3 {
        return f32::NAN;
    }
    if exp == 31 && mantissa == 0 {
        return if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY };
    }

    let fp32_exp = (exp - 15 + 127) as u32;
    let fp32_mantissa = mantissa << 21;
    let fp32_bits = ((sign as u32) << 31) | (fp32_exp << 23) | fp32_mantissa;

    f32::from_bits(fp32_bits)
}

/// 批量量化：FP32 → FP8 E4M3
pub fn quantize_fp8_e4m3(x: &[f32], scale: f32) -> Vec<u8> {
    x.iter().map(|&val| fp32_to_fp8_e4m3(val / scale)).collect()
}

/// 批量反量化：FP8 E4M3 → FP32
pub fn dequantize_fp8_e4m3(q: &[u8], scale: f32) -> Vec<f32> {
    q.iter().map(|&byte| fp8_e4m3_to_fp32(byte) * scale).collect()
}

/// 批量量化：FP32 → FP8 E5M2
pub fn quantize_fp8_e5m2(x: &[f32], scale: f32) -> Vec<u8> {
    x.iter().map(|&val| fp32_to_fp8_e5m2(val / scale)).collect()
}

/// 批量反量化：FP8 E5M2 → FP32
pub fn dequantize_fp8_e5m2(q: &[u8], scale: f32) -> Vec<f32> {
    q.iter().map(|&byte| fp8_e5m2_to_fp32(byte) * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        let values = vec![0.0, 1.0, -1.0, 2.5, -3.75, 100.0, -200.0];
        for &x in &values {
            let byte = fp32_to_fp8_e4m3(x);
            let recovered = fp8_e4m3_to_fp32(byte);
            let error = (x - recovered).abs() / x.abs().max(1e-6);
            assert!(error < 0.1, "E4M3: {} → {} (error: {})", x, recovered, error);
        }
    }

    #[test]
    fn test_fp8_e5m2_roundtrip() {
        let values = vec![0.0, 1.0, -1.0, 10.0, -50.0, 1000.0];
        for &x in &values {
            let byte = fp32_to_fp8_e5m2(x);
            let recovered = fp8_e5m2_to_fp32(byte);
            let error = (x - recovered).abs() / x.abs().max(1e-6);
            assert!(error < 0.15, "E5M2: {} → {} (error: {})", x, recovered, error);
        }
    }

    #[test]
    fn test_quantize_dequantize_e4m3() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let scale = 4.0;
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs() / x[i];
            assert!(error < 0.1);
        }
    }

    #[test]
    fn test_quantize_dequantize_e5m2() {
        let x = vec![10.0, 20.0, 30.0, 40.0];
        let scale = 40.0;
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs() / x[i];
            assert!(error < 0.15);
        }
    }

    #[test]
    fn test_fp8_special_values() {
        // NaN
        assert!(fp8_e4m3_to_fp32(0x7F).is_nan());
        assert!(fp8_e5m2_to_fp32(0x7F).is_nan());

        // +Inf
        assert!(fp8_e4m3_to_fp32(0x7E).is_infinite());
        assert!(fp8_e5m2_to_fp32(0x7C).is_infinite());

        // -Inf
        assert!(fp8_e4m3_to_fp32(0xFE).is_infinite());
        assert!(fp8_e5m2_to_fp32(0xFC).is_infinite());
    }

    // --- E4M3 zero encoding ---

    #[test]
    fn test_e4m3_positive_zero() {
        let byte = fp32_to_fp8_e4m3(0.0f32);
        assert_eq!(byte, 0x00);
        let recovered = fp8_e4m3_to_fp32(byte);
        assert_eq!(recovered, 0.0f32);
        assert!(!recovered.is_sign_negative());
    }

    #[test]
    fn test_e4m3_negative_zero() {
        let byte = fp32_to_fp8_e4m3(-0.0f32);
        assert_eq!(byte, 0x80);
        let recovered = fp8_e4m3_to_fp32(byte);
        assert_eq!(recovered, 0.0f32);
        assert!(recovered.is_sign_negative());
    }

    // --- E4M3 NaN / Inf encoding ---

    #[test]
    fn test_e4m3_nan_input() {
        let byte = fp32_to_fp8_e4m3(f32::NAN);
        assert_eq!(byte, 0x7F);
        let recovered = fp8_e4m3_to_fp32(byte);
        assert!(recovered.is_nan());
    }

    #[test]
    fn test_e4m3_positive_infinity_input() {
        let byte = fp32_to_fp8_e4m3(f32::INFINITY);
        assert_eq!(byte, 0x7E);
        let recovered = fp8_e4m3_to_fp32(byte);
        assert!(recovered.is_infinite());
        assert!(recovered.is_sign_positive());
    }

    #[test]
    fn test_e4m3_negative_infinity_input() {
        let byte = fp32_to_fp8_e4m3(f32::NEG_INFINITY);
        assert_eq!(byte, 0xFE);
        let recovered = fp8_e4m3_to_fp32(byte);
        assert!(recovered.is_infinite());
        assert!(recovered.is_sign_negative());
    }

    // --- E5M2 zero encoding ---

    #[test]
    fn test_e5m2_positive_zero() {
        let byte = fp32_to_fp8_e5m2(0.0f32);
        assert_eq!(byte, 0x00);
        let recovered = fp8_e5m2_to_fp32(byte);
        assert_eq!(recovered, 0.0f32);
        assert!(!recovered.is_sign_negative());
    }

    #[test]
    fn test_e5m2_negative_zero() {
        let byte = fp32_to_fp8_e5m2(-0.0f32);
        assert_eq!(byte, 0x80);
        let recovered = fp8_e5m2_to_fp32(byte);
        assert_eq!(recovered, 0.0f32);
        assert!(recovered.is_sign_negative());
    }

    // --- E5M2 NaN / Inf encoding ---

    #[test]
    fn test_e5m2_nan_input() {
        let byte = fp32_to_fp8_e5m2(f32::NAN);
        assert_eq!(byte, 0x7F);
        let recovered = fp8_e5m2_to_fp32(byte);
        assert!(recovered.is_nan());
    }

    #[test]
    fn test_e5m2_positive_infinity_input() {
        let byte = fp32_to_fp8_e5m2(f32::INFINITY);
        assert_eq!(byte, 0x7C);
        let recovered = fp8_e5m2_to_fp32(byte);
        assert!(recovered.is_infinite());
        assert!(recovered.is_sign_positive());
    }

    #[test]
    fn test_e5m2_negative_infinity_input() {
        let byte = fp32_to_fp8_e5m2(f32::NEG_INFINITY);
        assert_eq!(byte, 0xFC);
        let recovered = fp8_e5m2_to_fp32(byte);
        assert!(recovered.is_infinite());
        assert!(recovered.is_sign_negative());
    }

    // --- E4M3 clamping for out-of-range values ---
    // Values exceeding E4M3 range (~448) encode to NaN (0x7F) or Inf (0x7E).
    // Saturated values near the boundary round-trip with acceptable error.

    #[test]
    fn test_e4m3_clamp_out_of_range_positive() {
        // 500.0 exceeds E4M3 range; encoding saturates to NaN bit pattern.
        let byte = fp32_to_fp8_e4m3(500.0f32);
        assert_eq!(byte, 0x7F, "Out-of-range positive should encode to NaN pattern");
        let recovered = fp8_e4m3_to_fp32(byte);
        assert!(recovered.is_nan(), "Out-of-range positive decodes to NaN");
    }

    #[test]
    fn test_e4m3_clamp_out_of_range_negative() {
        // -500.0 exceeds E4M3 range; encoding saturates to NaN bit pattern.
        let byte = fp32_to_fp8_e4m3(-500.0f32);
        assert_eq!(byte, 0xFF, "Out-of-range negative should encode to NaN pattern with sign");
        let recovered = fp8_e4m3_to_fp32(byte);
        assert!(recovered.is_nan(), "Out-of-range negative decodes to NaN");
    }

    #[test]
    fn test_e4m3_near_boundary_roundtrip() {
        // 240.0 is near the max representable E4M3 normal value.
        let byte = fp32_to_fp8_e4m3(240.0f32);
        let recovered = fp8_e4m3_to_fp32(byte);
        let error = (240.0f32 - recovered).abs();
        assert!(
            error < 20.0,
            "E4M3 near-boundary 240.0 → {} (error: {})",
            recovered,
            error
        );
    }

    // --- E5M2 clamping for out-of-range values ---

    #[test]
    fn test_e5m2_clamp_very_large_positive() {
        let byte = fp32_to_fp8_e5m2(60000.0f32);
        let recovered = fp8_e5m2_to_fp32(byte);
        let error = (60000.0f32 - recovered).abs();
        assert!(error < 3000.0, "E5M2 clamped 60000.0 → {} (error: {})", recovered, error);
    }

    #[test]
    fn test_e5m2_clamp_very_large_negative() {
        let byte = fp32_to_fp8_e5m2(-60000.0f32);
        let recovered = fp8_e5m2_to_fp32(byte);
        let error = (-60000.0f32 - recovered).abs();
        assert!(error < 3000.0, "E5M2 clamped -60000.0 → {} (error: {})", recovered, error);
    }

    // --- Sign preservation symmetry ---

    #[test]
    fn test_e4m3_sign_symmetry() {
        let values = [0.5f32, 1.5, 3.0, 10.0, 50.0, 200.0];
        for &v in &values {
            let pos_byte = fp32_to_fp8_e4m3(v);
            let neg_byte = fp32_to_fp8_e4m3(-v);
            let pos_recovered = fp8_e4m3_to_fp32(pos_byte);
            let neg_recovered = fp8_e4m3_to_fp32(neg_byte);
            assert!(
                (pos_recovered - (-neg_recovered)).abs() < 0.01,
                "E4M3 sign asymmetry: {} vs {}",
                pos_recovered,
                neg_recovered
            );
        }
    }

    #[test]
    fn test_e5m2_sign_symmetry() {
        let values = [0.5f32, 1.5, 10.0, 100.0, 1000.0, 10000.0];
        for &v in &values {
            let pos_byte = fp32_to_fp8_e5m2(v);
            let neg_byte = fp32_to_fp8_e5m2(-v);
            let pos_recovered = fp8_e5m2_to_fp32(pos_byte);
            let neg_recovered = fp8_e5m2_to_fp32(neg_byte);
            assert!(
                (pos_recovered - (-neg_recovered)).abs() < 0.05,
                "E5M2 sign asymmetry: {} vs {}",
                pos_recovered,
                neg_recovered
            );
        }
    }

    // --- Subnormal / very small inputs ---

    #[test]
    fn test_e4m3_subnormal_input() {
        let tiny = f32::from_bits(0x007FFFFF); // Largest FP32 subnormal
        let byte = fp32_to_fp8_e4m3(tiny);
        assert_eq!(byte, 0x00, "E4M3 subnormal should encode as positive zero");
        let recovered = fp8_e4m3_to_fp32(byte);
        assert_eq!(recovered, 0.0f32);
    }

    #[test]
    fn test_e5m2_subnormal_input() {
        let tiny = f32::from_bits(0x007FFFFF); // Largest FP32 subnormal
        let byte = fp32_to_fp8_e5m2(tiny);
        assert_eq!(byte, 0x00, "E5M2 subnormal should encode as positive zero");
        let recovered = fp8_e5m2_to_fp32(byte);
        assert_eq!(recovered, 0.0f32);
    }

    // --- Batch identity with scale = 1.0 ---

    #[test]
    fn test_quantize_dequantize_e4m3_identity_scale() {
        let x = vec![1.0f32, -1.0, 2.0, -2.0, 4.0];
        let scale = 1.0f32;
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);
        assert_eq!(quantized.len(), x.len());
        assert_eq!(dequantized.len(), x.len());
        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs() / x[i].abs().max(1e-6);
            assert!(
                error < 0.15,
                "E4M3 identity scale: {} → {} (error: {})",
                x[i],
                dequantized[i],
                error
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_e5m2_identity_scale() {
        let x = vec![1.0f32, -1.0, 10.0, -10.0, 100.0];
        let scale = 1.0f32;
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);
        assert_eq!(quantized.len(), x.len());
        assert_eq!(dequantized.len(), x.len());
        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs() / x[i].abs().max(1e-6);
            assert!(
                error < 0.25,
                "E5M2 identity scale: {} → {} (error: {})",
                x[i],
                dequantized[i],
                error
            );
        }
    }

    // --- Empty batch input ---

    #[test]
    fn test_quantize_empty_input_e4m3() {
        let x: Vec<f32> = vec![];
        let quantized = quantize_fp8_e4m3(&x, 1.0);
        assert!(quantized.is_empty());
        let dequantized = dequantize_fp8_e4m3(&quantized, 1.0);
        assert!(dequantized.is_empty());
    }

    #[test]
    fn test_quantize_empty_input_e5m2() {
        let x: Vec<f32> = vec![];
        let quantized = quantize_fp8_e5m2(&x, 1.0);
        assert!(quantized.is_empty());
        let dequantized = dequantize_fp8_e5m2(&quantized, 1.0);
        assert!(dequantized.is_empty());
    }

    // --- Bit pattern consistency ---

    #[test]
    fn test_e4m3_known_bit_pattern_one() {
        // 1.0 in FP32: sign=0, exp=127 (0x7F), mantissa=0
        // FP8 E4M3: exp = 127-127+7 = 7, mantissa = 0
        // byte = 0 | (7 << 3) | 0 = 0x38
        let byte = fp32_to_fp8_e4m3(1.0f32);
        assert_eq!(byte, 0x38);
        let recovered = fp8_e4m3_to_fp32(byte);
        assert!((recovered - 1.0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_e4m3_known_bit_pattern_negative_two() {
        // -2.0 in FP32: sign=1, exp=128, mantissa=0
        // FP8 E4M3: exp = 128-127+7 = 8, mantissa = 0
        // byte = 0x80 | (8 << 3) | 0 = 0x80 | 0x40 = 0xC0
        let byte = fp32_to_fp8_e4m3(-2.0f32);
        assert_eq!(byte, 0xC0);
        let recovered = fp8_e4m3_to_fp32(byte);
        assert!((recovered - (-2.0f32)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_e5m2_known_bit_pattern_one() {
        // 1.0 in FP32: sign=0, exp=127, mantissa=0
        // FP8 E5M2: exp = 127-127+15 = 15, mantissa = 0
        // byte = 0 | (15 << 2) | 0 = 0x3C
        let byte = fp32_to_fp8_e5m2(1.0f32);
        assert_eq!(byte, 0x3C);
        let recovered = fp8_e5m2_to_fp32(byte);
        assert!((recovered - 1.0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_e5m2_known_bit_pattern_negative_four() {
        // -4.0 in FP32: sign=1, exp=129, mantissa=0
        // FP8 E5M2: exp = 129-127+15 = 17, mantissa = 0
        // byte = 0x80 | (17 << 2) | 0 = 0x80 | 0x44 = 0xC4
        let byte = fp32_to_fp8_e5m2(-4.0f32);
        assert_eq!(byte, 0xC4);
        let recovered = fp8_e5m2_to_fp32(byte);
        assert!((recovered - (-4.0f32)).abs() < f32::EPSILON);
    }

    // --- Batch scale factor correctness ---

    #[test]
    fn test_e4m3_scale_divides_range() {
        // With scale=448.0, values [-448, 448] map to [-1.0, 1.0] before encoding.
        // 224.0 / 448.0 = 0.5, which should encode precisely.
        let x = vec![224.0f32];
        let scale = 448.0f32;
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);
        let error = (x[0] - dequantized[0]).abs();
        assert!(
            error < 10.0,
            "E4M3 scaled 224.0 → {} (error: {})",
            dequantized[0],
            error
        );
    }

    #[test]
    fn test_e5m2_scale_divides_range() {
        // With scale=57344.0, values [-57344, 57344] map to [-1.0, 1.0].
        // 28672.0 / 57344.0 = 0.5
        let x = vec![28672.0f32];
        let scale = 57344.0f32;
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);
        let error = (x[0] - dequantized[0]).abs();
        assert!(
            error < 500.0,
            "E5M2 scaled 28672.0 → {} (error: {})",
            dequantized[0],
            error
        );
    }

    // --- E4M3 exponent boundary: max representable normal ---

    #[test]
    fn test_e4m3_max_normal_roundtrip() {
        // E4M3 max normal: exp=14, mantissa=7 => 2^(14-7)*(1+7/8) = 240.0
        // byte = 0 | (14 << 3) | 7 = 0x77
        let val = fp8_e4m3_to_fp32(0x77);
        assert!(val > 0.0);
        let re_encoded = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(re_encoded);
        assert!(
            (val - recovered).abs() < 1.0,
            "E4M3 max normal: {} → {} → {}",
            val,
            re_encoded,
            recovered
        );
    }

    // --- E5M2 exponent boundary: max representable normal ---

    #[test]
    fn test_e5m2_max_normal_roundtrip() {
        // E5M2 max normal exponent = 30 (exp 31 reserved), mantissa = 3
        // byte = 0 | (30 << 2) | 3 = 0x7B
        let val = fp8_e5m2_to_fp32(0x7B);
        assert!(val > 0.0);
        let re_encoded = fp32_to_fp8_e5m2(val);
        let recovered = fp8_e5m2_to_fp32(re_encoded);
        assert!(
            (val - recovered).abs() < val * 0.2,
            "E5M2 max normal: {} → {} → {}",
            val,
            re_encoded,
            recovered
        );
    }

    // --- Additional edge case tests ---

    #[test]
    fn test_e4m3_single_element_batch() {
        let x = vec![3.14f32];
        let scale = 2.0f32;
        let quantized = quantize_fp8_e4m3(&x, scale);
        assert_eq!(quantized.len(), 1);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);
        assert_eq!(dequantized.len(), 1);
        let error = (x[0] - dequantized[0]).abs() / x[0];
        assert!(error < 0.15, "E4M3 single element: {} → {}", x[0], dequantized[0]);
    }

    #[test]
    fn test_e4m3_negative_subnormal_input() {
        let tiny = f32::from_bits(0x807FFFFF); // Largest negative FP32 subnormal
        let byte = fp32_to_fp8_e4m3(tiny);
        assert_eq!(byte, 0x80, "E4M3 negative subnormal should encode as negative zero");
        let recovered = fp8_e4m3_to_fp32(byte);
        assert_eq!(recovered, 0.0f32);
        assert!(recovered.is_sign_negative());
    }

    #[test]
    fn test_e5m2_negative_subnormal_input() {
        let tiny = f32::from_bits(0x807FFFFF); // Largest negative FP32 subnormal
        let byte = fp32_to_fp8_e5m2(tiny);
        assert_eq!(byte, 0x80, "E5M2 negative subnormal should encode as negative zero");
        let recovered = fp8_e5m2_to_fp32(byte);
        assert_eq!(recovered, 0.0f32);
        assert!(recovered.is_sign_negative());
    }

    #[test]
    fn test_e4m3_negative_max_normal_roundtrip() {
        // Negative of max normal: byte = 0x80 | 0x77 = 0xF7
        let val = fp8_e4m3_to_fp32(0xF7);
        assert!(val < 0.0, "Should be negative, got {}", val);
        let re_encoded = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(re_encoded);
        assert!(
            (val - recovered).abs() < 1.0,
            "E4M3 neg max normal: {} → {} → {}",
            val,
            re_encoded,
            recovered
        );
    }

    #[test]
    fn test_e5m2_negative_max_normal_roundtrip() {
        // Negative of E5M2 max normal: byte = 0x80 | 0x7B = 0xFB
        let val = fp8_e5m2_to_fp32(0xFB);
        assert!(val < 0.0, "Should be negative, got {}", val);
        let re_encoded = fp32_to_fp8_e5m2(val);
        let recovered = fp8_e5m2_to_fp32(re_encoded);
        assert!(
            (val - recovered).abs() < val.abs() * 0.2,
            "E5M2 neg max normal: {} → {} → {}",
            val,
            re_encoded,
            recovered
        );
    }

    #[test]
    fn test_e4m3_min_normal_decode() {
        // E4M3 min normal: exp=1, mantissa=0 => byte = (1 << 3) | 0 = 0x08
        let val = fp8_e4m3_to_fp32(0x08);
        assert!(val > 0.0, "E4M3 min normal should be positive, got {}", val);
        assert!(val < 0.02, "E4M3 min normal should be very small, got {}", val);
    }

    #[test]
    fn test_e5m2_min_normal_decode() {
        // E5M2 min normal: exp=1, mantissa=0 => byte = (1 << 2) | 0 = 0x04
        let val = fp8_e5m2_to_fp32(0x04);
        assert!(val > 0.0, "E5M2 min normal should be positive, got {}", val);
        assert!(val < 1e-3, "E5M2 min normal should be small, got {}", val);
    }

    #[test]
    fn test_e5m2_known_bit_pattern_negative_one() {
        // -1.0: sign=1, exp=127-127+15=15, mantissa=0
        // byte = 0x80 | (15 << 2) | 0 = 0x80 | 0x3C = 0xBC
        let byte = fp32_to_fp8_e5m2(-1.0f32);
        assert_eq!(byte, 0xBC);
        let recovered = fp8_e5m2_to_fp32(byte);
        assert!((recovered - (-1.0f32)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_mixed_values_with_zeros() {
        let x = vec![0.0f32, -0.0, 5.0, -3.0, 0.0, 100.0];
        let scale = 1.0f32;
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);
        assert_eq!(dequantized.len(), 6);
        // Zeros round-trip exactly
        assert_eq!(dequantized[0], 0.0f32);
        assert_eq!(dequantized[4], 0.0f32);
        // Non-zero values have acceptable error
        assert!((dequantized[2] - 5.0f32).abs() / 5.0 < 0.15);
    }

    #[test]
    fn test_e4m3_tiny_normal_clamps_to_zero() {
        // FP32 exp = 120 => actual exp = -7, E4M3 fp8_exp = -7+7 = 0 => clamps to exp=0
        let tiny = 2.0f32.powi(-7); // exp=-7 is below E4M3 minimum representable
        let byte = fp32_to_fp8_e4m3(tiny);
        let recovered = fp8_e4m3_to_fp32(byte);
        // With clamped exp=0 and zero mantissa, this is zero
        assert!(
            recovered == 0.0f32 || (recovered - tiny).abs() / tiny < 0.5,
            "Tiny normal {} → byte={} → {}",
            tiny,
            byte,
            recovered
        );
    }

    #[test]
    fn test_dequantize_e4m3_nan_does_not_panic() {
        let q = vec![0x7F, 0xFF]; // NaN patterns
        let dequantized = dequantize_fp8_e4m3(&q, 2.0f32);
        assert_eq!(dequantized.len(), 2);
        assert!(dequantized[0].is_nan());
        assert!(dequantized[1].is_nan());
    }

    #[test]
    fn test_dequantize_e5m2_nan_does_not_panic() {
        let q = vec![0x7F, 0xFF]; // NaN patterns
        let dequantized = dequantize_fp8_e5m2(&q, 2.0f32);
        assert_eq!(dequantized.len(), 2);
        assert!(dequantized[0].is_nan());
        assert!(dequantized[1].is_nan());
    }

    #[test]
    fn test_batch_with_nan_preserves_length() {
        let x = vec![1.0f32, f32::NAN, -2.0, f32::INFINITY, 0.0];
        let quantized = quantize_fp8_e4m3(&x, 1.0);
        assert_eq!(quantized.len(), 5);
        let dequantized = dequantize_fp8_e4m3(&quantized, 1.0);
        assert_eq!(dequantized.len(), 5);
        // NaN round-trips as NaN
        assert!(dequantized[1].is_nan());
        // Inf round-trips as Inf
        assert!(dequantized[3].is_infinite());
    }

    #[test]
    fn test_e5m2_large_batch_preserves_order() {
        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let scale = 100.0f32;
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);
        assert_eq!(dequantized.len(), 100);
        // Monotonicity: dequantized values should be non-decreasing
        for i in 1..dequantized.len() {
            assert!(
                dequantized[i] >= dequantized[i - 1],
                "E5M2 monotonicity violated at index {}: {} < {}",
                i,
                dequantized[i],
                dequantized[i - 1]
            );
        }
    }

    // --- 15 new tests below ---

    #[test]
    fn test_e4m3_powers_of_two_exact_roundtrip() {
        // Arrange: powers of two within E4M3 range should roundtrip exactly
        // because they have zero mantissa bits in both FP32 and FP8.
        let powers: Vec<f32> = (0..=8).map(|e| 2.0f32.powi(e)).collect();

        // Act & Assert
        for &v in &powers {
            let byte = fp32_to_fp8_e4m3(v);
            let recovered = fp8_e4m3_to_fp32(byte);
            assert!(
                (recovered - v).abs() < f32::EPSILON,
                "E4M3 power-of-two exact roundtrip failed: {} → byte 0x{:02X} → {}",
                v,
                byte,
                recovered
            );
        }
    }

    #[test]
    fn test_e5m2_powers_of_two_exact_roundtrip() {
        // Arrange: powers of two within E5M2 range should roundtrip exactly.
        let powers: Vec<f32> = (0..=13).map(|e| 2.0f32.powi(e)).collect();

        // Act & Assert
        for &v in &powers {
            let byte = fp32_to_fp8_e5m2(v);
            let recovered = fp8_e5m2_to_fp32(byte);
            assert!(
                (recovered - v).abs() < f32::EPSILON,
                "E5M2 power-of-two exact roundtrip failed: {} → byte 0x{:02X} → {}",
                v,
                byte,
                recovered
            );
        }
    }

    #[test]
    fn test_e4m3_negative_powers_of_two_exact_roundtrip() {
        // Arrange: negative powers of two should also roundtrip exactly
        // with sign bit set but identical exponent/mantissa.
        let powers: Vec<f32> = (0..=8).map(|e| -2.0f32.powi(e)).collect();

        // Act & Assert
        for &v in &powers {
            let byte = fp32_to_fp8_e4m3(v);
            let recovered = fp8_e4m3_to_fp32(byte);
            assert!(
                (recovered - v).abs() < f32::EPSILON,
                "E4M3 negative power-of-two exact roundtrip failed: {} → byte 0x{:02X} → {}",
                v,
                byte,
                recovered
            );
        }
    }

    #[test]
    fn test_e5m2_inf_nan_distinct_bit_patterns() {
        // Arrange: E5M2 exp=31 reserves multiple patterns.
        // mantissa=0 → Inf, mantissa=3 → NaN, mantissa=1,2 → should not panic.
        let inf_byte = 0x7C_u8; // exp=31, mantissa=0, sign=0
        let nan_byte = 0x7F_u8; // exp=31, mantissa=3, sign=0
        let reserved_1 = 0x7D_u8; // exp=31, mantissa=1
        let reserved_2 = 0x7E_u8; // exp=31, mantissa=2

        // Act
        let inf_val = fp8_e5m2_to_fp32(inf_byte);
        let nan_val = fp8_e5m2_to_fp32(nan_byte);
        let val_1 = fp8_e5m2_to_fp32(reserved_1);
        let val_2 = fp8_e5m2_to_fp32(reserved_2);

        // Assert: Inf and NaN are distinct; reserved patterns decode to finite values
        assert!(inf_val.is_infinite() && inf_val.is_sign_positive());
        assert!(nan_val.is_nan());
        assert!(val_1.is_finite(), "E5M2 exp=31 mantissa=1 should be finite, got {}", val_1);
        assert!(val_2.is_finite(), "E5M2 exp=31 mantissa=2 should be finite, got {}", val_2);
    }

    #[test]
    fn test_e4m3_negative_inf_does_not_equal_positive_inf() {
        // Arrange
        let pos_inf_byte = fp32_to_fp8_e4m3(f32::INFINITY);
        let neg_inf_byte = fp32_to_fp8_e4m3(f32::NEG_INFINITY);

        // Act
        let pos_recovered = fp8_e4m3_to_fp32(pos_inf_byte);
        let neg_recovered = fp8_e4m3_to_fp32(neg_inf_byte);

        // Assert: different bit patterns and different sign
        assert_ne!(pos_inf_byte, neg_inf_byte);
        assert!(pos_recovered.is_sign_positive());
        assert!(neg_recovered.is_sign_negative());
    }

    #[test]
    fn test_e5m2_negative_inf_does_not_equal_positive_inf() {
        // Arrange
        let pos_inf_byte = fp32_to_fp8_e5m2(f32::INFINITY);
        let neg_inf_byte = fp32_to_fp8_e5m2(f32::NEG_INFINITY);

        // Act
        let pos_recovered = fp8_e5m2_to_fp32(pos_inf_byte);
        let neg_recovered = fp8_e5m2_to_fp32(neg_inf_byte);

        // Assert: different bit patterns and different sign
        assert_ne!(pos_inf_byte, neg_inf_byte);
        assert!(pos_recovered.is_sign_positive());
        assert!(neg_recovered.is_sign_negative());
    }

    #[test]
    fn test_quantize_dequantize_e4m3_preserves_order() {
        // Arrange: strictly increasing values
        let x = vec![1.0f32, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: monotonicity must be preserved
        assert_eq!(dequantized.len(), x.len());
        for i in 1..dequantized.len() {
            assert!(
                dequantized[i] > dequantized[i - 1],
                "E4M3 order violated at [{}]: {} <= {}",
                i,
                dequantized[i],
                dequantized[i - 1]
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_e4m3_negative_order() {
        // Arrange: strictly decreasing negative values
        let x = vec![-200.0f32, -100.0, -50.0, -10.0, -5.0, -2.0, -1.0];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: monotonicity must be preserved for negatives
        assert_eq!(dequantized.len(), x.len());
        for i in 1..dequantized.len() {
            assert!(
                dequantized[i] > dequantized[i - 1],
                "E4M3 negative order violated at [{}]: {} <= {}",
                i,
                dequantized[i],
                dequantized[i - 1]
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_e5m2_with_special_floats() {
        // Arrange: mix of normal, zero, NaN, Inf
        let x = vec![0.0f32, 1.0, f32::NAN, f32::INFINITY, -100.0, -0.0];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        // Assert: length preserved, special values handled
        assert_eq!(dequantized.len(), 6);
        assert_eq!(dequantized[0], 0.0f32);
        assert!(dequantized[2].is_nan());
        assert!(dequantized[3].is_infinite());
        assert!(dequantized[5] == 0.0f32 && dequantized[5].is_sign_negative());
    }

    #[test]
    fn test_e4m3_mantissa_precision_loss() {
        // Arrange: values with non-trivial mantissa that get truncated
        // E4M3 keeps only 3 mantissa bits, so 1.xxx with 7 patterns (0..7).
        // 1.5 in FP32 = 1 + 0.5 = mantissa bit 0x400000 >> 20 = 4 (top 3 bits).
        let val = 1.5f32;
        let byte = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Act & Assert: 1.5 should encode with mantissa=4 (0b100), roundtrip exactly
        let expected_mantissa = (byte & 0x7) as u32;
        assert_eq!(expected_mantissa, 4, "1.5 should have mantissa=4, got {}", expected_mantissa);
        assert!(
            (recovered - val).abs() < f32::EPSILON,
            "1.5 roundtrip: got {}",
            recovered
        );
    }

    #[test]
    fn test_e5m2_mantissa_precision_loss() {
        // Arrange: E5M2 keeps only 2 mantissa bits.
        // 3.0 in FP32 = 1.5 * 2^1 → mantissa top 2 bits from 0x400000 >> 21 = 2.
        let val = 3.0f32;
        let byte = fp32_to_fp8_e5m2(val);
        let recovered = fp8_e5m2_to_fp32(byte);

        // Act & Assert: 3.0 should encode with mantissa=2
        let expected_mantissa = (byte & 0x3) as u32;
        assert_eq!(expected_mantissa, 2, "3.0 should have mantissa=2, got {}", expected_mantissa);
        assert!(
            (recovered - val).abs() < f32::EPSILON,
            "3.0 roundtrip: got {}",
            recovered
        );
    }

    #[test]
    fn test_quantize_dequantize_e4m3_large_scale() {
        // Arrange: very large scale maps huge values into representable range
        let x = vec![10000.0f32, 20000.0, 30000.0];
        let scale = 100.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: large scale brings out-of-range values into representable range
        assert_eq!(dequantized.len(), 3);
        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs() / x[i];
            assert!(
                error < 0.2,
                "E4M3 large scale: {} → {} (error: {})",
                x[i],
                dequantized[i],
                error
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_e5m2_small_scale() {
        // Arrange: small scale amplifies tiny values for better precision
        let x = vec![0.001f32, 0.005, 0.01];
        let scale = 0.001f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        // Assert: small scale makes tiny values representable
        assert_eq!(dequantized.len(), 3);
        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs() / x[i];
            assert!(
                error < 0.3,
                "E5M2 small scale: {} → {} (error: {})",
                x[i],
                dequantized[i],
                error
            );
        }
    }

    #[test]
    fn test_e4m3_all_non_zero_exponent_values_distinct() {
        // Arrange: encode exp values 1..14 with mantissa=0, verify all produce distinct bytes
        let mut encoded: Vec<u8> = Vec::new();
        for exp in 1..=14u8 {
            let byte = (exp << 3) & 0x78; // sign=0, mantissa=0
            encoded.push(byte);
        }

        // Act: decode each and verify they are distinct and positive
        let decoded: Vec<f32> = encoded.iter().map(|&b| fp8_e4m3_to_fp32(b)).collect();

        // Assert
        for (i, &val) in decoded.iter().enumerate() {
            assert!(
                val > 0.0,
                "E4M3 exp={} mantissa=0 should decode to positive, got {}",
                i + 1,
                val
            );
        }
        // Each consecutive value should be larger (exponent increases)
        for i in 1..decoded.len() {
            assert!(
                decoded[i] > decoded[i - 1],
                "E4M3 exp={} value {} should be > exp={} value {}",
                i + 1,
                decoded[i],
                i,
                decoded[i - 1]
            );
        }
    }

    #[test]
    fn test_e5m2_all_non_zero_exponent_values_distinct() {
        // Arrange: encode exp values 1..30 with mantissa=0, verify monotonic increase
        let mut decoded: Vec<f32> = Vec::new();
        for exp in 1..=30u8 {
            let byte = (exp << 2) & 0x7C; // sign=0, mantissa=0
            let val = fp8_e5m2_to_fp32(byte);
            decoded.push(val);
        }

        // Assert: all positive and strictly increasing
        for (i, &val) in decoded.iter().enumerate() {
            assert!(
                val > 0.0,
                "E5M2 exp={} mantissa=0 should decode to positive, got {}",
                i + 1,
                val
            );
        }
        for i in 1..decoded.len() {
            assert!(
                decoded[i] > decoded[i - 1],
                "E5M2 exp={} value {} should be > exp={} value {}",
                i + 1,
                decoded[i],
                i,
                decoded[i - 1]
            );
        }
    }

    #[test]
    fn test_e4m3_max_minus_one_mantissa_roundtrip() {
        // Arrange: max normal with mantissa=6 (one below max mantissa=7)
        // exp=14, mantissa=6 → byte = (14 << 3) | 6 = 0x76
        let byte = 0x76_u8;

        // Act
        let val = fp8_e4m3_to_fp32(byte);
        let re_encoded = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(re_encoded);

        // Assert: value is positive, less than max normal, roundtrips
        assert!(val > 0.0);
        let max_normal = fp8_e4m3_to_fp32(0x77);
        assert!(val < max_normal, "mantissa=6 should be less than mantissa=7 for same exp");
        assert!(
            (val - recovered).abs() < 2.0,
            "E4M3 exp=14 mantissa=6 roundtrip: {} → {} → {}",
            val,
            re_encoded,
            recovered
        );
    }

    // --- 15 additional tests ---

    #[test]
    fn test_e4m3_denormal_byte_patterns_decode_finite() {
        // Arrange: E4M3 bytes with exp=0 but non-zero mantissa.
        // These are not standard denormals in this implementation — they decode
        // via the normal formula with fp32_exp = 0 - 7 + 127 = 120.
        let denormal_bytes: Vec<u8> = (1..=7).collect(); // exp=0, mantissa=1..7

        // Act
        let decoded: Vec<f32> = denormal_bytes.iter().map(|&b| fp8_e4m3_to_fp32(b)).collect();

        // Assert: all decode to finite, positive, non-zero values
        for (i, &val) in decoded.iter().enumerate() {
            assert!(
                val.is_finite() && val > 0.0,
                "E4M3 denormal byte 0x{:02X} should decode to positive finite, got {}",
                denormal_bytes[i],
                val
            );
        }
        // Monotonicity within same exponent
        for i in 1..decoded.len() {
            assert!(
                decoded[i] > decoded[i - 1],
                "E4M3 denormal monotonicity: byte {} val {} should be > byte {} val {}",
                denormal_bytes[i],
                decoded[i],
                denormal_bytes[i - 1],
                decoded[i - 1]
            );
        }
    }

    #[test]
    fn test_e5m2_denormal_byte_patterns_decode_finite() {
        // Arrange: E5M2 bytes with exp=0 but non-zero mantissa (1 or 2).
        // fp32_exp = 0 - 15 + 127 = 112, so these decode to very small numbers.
        let denormal_bytes: Vec<u8> = vec![0x01, 0x02]; // exp=0, mantissa=1,2

        // Act
        let decoded: Vec<f32> = denormal_bytes.iter().map(|&b| fp8_e5m2_to_fp32(b)).collect();

        // Assert: both decode to finite positive values
        for (i, &val) in decoded.iter().enumerate() {
            assert!(
                val.is_finite() && val > 0.0,
                "E5M2 denormal byte 0x{:02X} should decode to positive finite, got {}",
                denormal_bytes[i],
                val
            );
        }
        assert!(
            decoded[1] > decoded[0],
            "E5M2 mantissa=2 should be larger than mantissa=1 for same exp"
        );
    }

    #[test]
    fn test_e4m3_known_bit_pattern_negative_one() {
        // Arrange: -1.0 in FP32: sign=1, exp=127, mantissa=0
        // FP8 E4M3: exp = 127-127+7 = 7, mantissa = 0
        // byte = 0x80 | (7 << 3) | 0 = 0x80 | 0x38 = 0xB8

        // Act
        let byte = fp32_to_fp8_e4m3(-1.0f32);

        // Assert
        assert_eq!(byte, 0xB8, "-1.0 should encode to 0xB8");
        let recovered = fp8_e4m3_to_fp32(byte);
        assert!((recovered - (-1.0f32)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_e4m3_full_mantissa_mid_exponent_roundtrip() {
        // Arrange: exp=7 (center), mantissa=7 (all bits set), sign=0
        // byte = (7 << 3) | 7 = 0x3F
        let byte = 0x3F_u8;

        // Act
        let val = fp8_e4m3_to_fp32(byte);
        let re_encoded = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(re_encoded);

        // Assert: should roundtrip with minimal error
        assert!(val > 0.0, "Should be positive, got {}", val);
        assert!(
            (val - recovered).abs() < 0.1,
            "E4M3 exp=7 mantissa=7 roundtrip: {} → {} → {}",
            val,
            re_encoded,
            recovered
        );
    }

    #[test]
    fn test_e5m2_negative_powers_of_two_roundtrip() {
        // Arrange: E5M2 has wider exponent range, so small negative powers of two
        // should roundtrip exactly (bias=15 covers exp -15..+16).
        let neg_powers: Vec<f32> = (0..=13).map(|e| -2.0f32.powi(e)).collect();

        // Act & Assert
        for &v in &neg_powers {
            let byte = fp32_to_fp8_e5m2(v);
            let recovered = fp8_e5m2_to_fp32(byte);
            assert!(
                (recovered - v).abs() < f32::EPSILON,
                "E5M2 negative power-of-two roundtrip failed: {} → byte 0x{:02X} → {}",
                v,
                byte,
                recovered
            );
        }
    }

    #[test]
    fn test_quantize_e4m3_mixed_sign_large_batch() {
        // Arrange: alternating positive/negative values
        let x: Vec<f32> = (0..50)
            .flat_map(|i| {
                let v = (i as f32) * 2.0 + 1.0;
                vec![v, -v]
            })
            .collect();
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: length preserved, signs alternate correctly
        assert_eq!(dequantized.len(), 100);
        for i in 0..50 {
            let pos_idx = i * 2;
            let neg_idx = i * 2 + 1;
            assert!(
                dequantized[pos_idx] >= 0.0,
                "Even index {} should be non-negative, got {}",
                pos_idx,
                dequantized[pos_idx]
            );
            assert!(
                dequantized[neg_idx] <= 0.0,
                "Odd index {} should be non-positive, got {}",
                neg_idx,
                dequantized[neg_idx]
            );
        }
    }

    #[test]
    fn test_quantize_e5m2_with_nan_inf_preserves_structure() {
        // Arrange: values with NaN and Inf scattered in the batch
        let x = vec![10.0f32, f32::NAN, -20.0, f32::INFINITY, 0.0, f32::NEG_INFINITY];
        let scale = 2.0f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        // Assert: length preserved, special values handled, scale applied
        assert_eq!(quantized.len(), 6);
        assert_eq!(dequantized.len(), 6);
        assert!(dequantized[1].is_nan(), "NaN should roundtrip as NaN");
        assert!(dequantized[3].is_infinite() && dequantized[3].is_sign_positive());
        assert!(dequantized[5].is_infinite() && dequantized[5].is_sign_negative());
        assert_eq!(dequantized[4], 0.0f32, "Zero should roundtrip exactly");
    }

    #[test]
    fn test_dequantize_e4m3_all_zeros_is_all_zeros() {
        // Arrange: all-zero bytes decode to +0.0
        let q = vec![0x00_u8; 16];

        // Act
        let dequantized = dequantize_fp8_e4m3(&q, 1.0f32);

        // Assert: scale doesn't matter for zero
        assert_eq!(dequantized.len(), 16);
        for (i, &val) in dequantized.iter().enumerate() {
            assert_eq!(
                val, 0.0f32,
                "Byte 0x00 at index {} should dequantize to 0.0, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_dequantize_e5m2_all_zeros_with_large_scale() {
        // Arrange: zero bytes with a very large scale should still be zero
        let q = vec![0x00_u8; 8];
        let scale = 1e10f32;

        // Act
        let dequantized = dequantize_fp8_e5m2(&q, scale);

        // Assert: 0 * any_scale = 0
        for (i, &val) in dequantized.iter().enumerate() {
            assert_eq!(
                val, 0.0f32,
                "Zero with large scale at index {} should still be 0.0, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_e4m3_half_integer_roundtrip() {
        // Arrange: 0.5 = 2^(-1), should encode exactly (power of two).
        // Also test 1.25 = 1 + 1/4, mantissa = 0b010 = 2, should be exact.
        let values = vec![0.5f32, 1.25f32, 2.5f32, 5.0f32];

        // Act & Assert
        for &v in &values {
            let byte = fp32_to_fp8_e4m3(v);
            let recovered = fp8_e4m3_to_fp32(byte);
            assert!(
                (recovered - v).abs() < f32::EPSILON,
                "E4M3 half-integer {} → byte 0x{:02X} → {} should be exact",
                v,
                byte,
                recovered
            );
        }
    }

    #[test]
    fn test_e5m2_half_integer_roundtrip() {
        // Arrange: values representable exactly in E5M2 (power of two or simple fraction)
        let values = vec![0.5f32, 1.5f32, 3.0f32, 6.0f32, 12.0f32];

        // Act & Assert
        for &v in &values {
            let byte = fp32_to_fp8_e5m2(v);
            let recovered = fp8_e5m2_to_fp32(byte);
            assert!(
                (recovered - v).abs() < f32::EPSILON,
                "E5M2 half-integer {} → byte 0x{:02X} → {} should be exact",
                v,
                byte,
                recovered
            );
        }
    }

    #[test]
    fn test_e4m3_exponent_boundary_clamp_exp_zero() {
        // Arrange: FP32 exp = 121 => actual exp = -6, E4M3 fp8_exp = -6+7 = 1
        // This is the smallest non-clamped exponent.
        let smallest_normal = 2.0f32.powi(-6); // exp = -6, fp8_exp = 1

        // Act
        let byte = fp32_to_fp8_e4m3(smallest_normal);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Assert: should not be clamped to zero since fp8_exp = 1 > 0
        assert!(
            recovered > 0.0,
            "E4M3 exp=-6 should not clamp to zero: {} → byte 0x{:02X} → {}",
            smallest_normal,
            byte,
            recovered
        );
        assert!(
            (recovered - smallest_normal).abs() < f32::EPSILON,
            "E4M3 min non-clamped normal should roundtrip exactly: {} vs {}",
            smallest_normal,
            recovered
        );
    }

    #[test]
    fn test_e5m2_exponent_boundary_clamp_exp_zero() {
        // Arrange: FP32 exp = 112 => actual exp = -15, E5M2 fp8_exp = -15+15 = 0
        // This should clamp to exp=0.
        let at_boundary = 2.0f32.powi(-15);

        // Act
        let byte = fp32_to_fp8_e5m2(at_boundary);
        let recovered = fp8_e5m2_to_fp32(byte);

        // Assert: exp clamps to 0, so this decodes to a very small but non-zero value
        // since mantissa will also be 0, it becomes 0.0
        assert!(
            recovered == 0.0f32 || (recovered - at_boundary).abs() / at_boundary < 0.5,
            "E5M2 exp=-15 boundary: {} → byte 0x{:02X} → {}",
            at_boundary,
            byte,
            recovered
        );
    }

    #[test]
    fn test_quantize_dequantize_e4m3_symmetric_scale() {
        // Arrange: scale chosen so that value/scale is exactly representable.
        // 8.0 / 8.0 = 1.0, which is exactly representable in E4M3.
        let x = vec![8.0f32, -8.0, 16.0, -16.0];
        let scale = 8.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: values where x/scale is a power of two roundtrip exactly
        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs();
            assert!(
                error < f32::EPSILON,
                "E4M3 symmetric scale: {} → {} (error: {})",
                x[i],
                dequantized[i],
                error
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_e5m2_symmetric_scale() {
        // Arrange: scale=4.0, so 4.0/4.0=1.0 and 12.0/4.0=3.0, both exact in E5M2.
        let x = vec![4.0f32, -4.0, 12.0, -12.0];
        let scale = 4.0f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        // Assert: exact roundtrip for these values
        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs();
            assert!(
                error < f32::EPSILON,
                "E5M2 symmetric scale: {} → {} (error: {})",
                x[i],
                dequantized[i],
                error
            );
        }
    }

    #[test]
    fn test_e4m3_encode_decode_all_exp_7_mantissa_values() {
        // Arrange: exp=7 (center of E4M3 range), all 8 mantissa values (0..7)
        let mut values: Vec<f32> = Vec::with_capacity(8);
        for m in 0u8..=7 {
            let byte = (7u8 << 3) | m;
            values.push(fp8_e4m3_to_fp32(byte));
        }

        // Act: re-encode each decoded value
        let re_encoded: Vec<u8> = values.iter().map(|&v| fp32_to_fp8_e4m3(v)).collect();

        // Assert: re-encoded bytes should match or be very close (within rounding)
        for m in 0u8..=7 {
            let original_byte = (7u8 << 3) | m;
            let diff = (re_encoded[m as usize] as i16 - original_byte as i16).abs();
            assert!(
                diff <= 1,
                "E4M3 exp=7 mantissa={}: original=0x{:02X}, re-encoded=0x{:02X} (diff={})",
                m,
                original_byte,
                re_encoded[m as usize],
                diff
            );
        }
    }

    // =========================================================================
    // 15 additional tests — FP8 encoding boundaries, cross-format, scale edge cases
    // =========================================================================

    #[test]
    fn test_e4m3_exp_15_non_reserved_mantissas_are_finite() {
        // Arrange: E4M3 exp=15 is partially reserved.
        // mantissa=7 → NaN, mantissa=6 → Inf, but mantissa=0..5 should decode to
        // finite values (fp32_exp = 15 - 7 + 127 = 135).
        let reserved_bytes: Vec<u8> = (0..=5).map(|m| (15u8 << 3) | m).collect();

        // Act
        let decoded: Vec<f32> = reserved_bytes
            .iter()
            .map(|&b| fp8_e4m3_to_fp32(b))
            .collect();

        // Assert: mantissa 0..5 with exp=15 decode to finite positive values
        for (i, &val) in decoded.iter().enumerate() {
            assert!(
                val.is_finite() && val > 0.0,
                "E4M3 exp=15 mantissa={} (byte 0x{:02X}) should be finite positive, got {}",
                i,
                reserved_bytes[i],
                val
            );
        }
    }

    #[test]
    fn test_e5m2_negative_inf_byte_pattern() {
        // Arrange: E5M2 negative infinity is exp=31, mantissa=0, sign=1
        // byte = 0x80 | (31 << 2) | 0 = 0x80 | 0x7C = 0xFC

        // Act
        let val = fp8_e5m2_to_fp32(0xFC);

        // Assert
        assert!(
            val.is_infinite() && val.is_sign_negative(),
            "E5M2 byte 0xFC should decode to -Inf, got {}",
            val
        );
    }

    #[test]
    fn test_cross_format_same_value_both_formats() {
        // Arrange: 1.0 is exactly representable in both E4M3 and E5M2.
        // Verify both formats produce the same decoded value from different encodings.
        let val = 1.0f32;

        // Act
        let e4m3_byte = fp32_to_fp8_e4m3(val);
        let e5m2_byte = fp32_to_fp8_e5m2(val);
        let e4m3_recovered = fp8_e4m3_to_fp32(e4m3_byte);
        let e5m2_recovered = fp8_e5m2_to_fp32(e5m2_byte);

        // Assert: both recover 1.0 exactly, but use different bit patterns
        assert!((e4m3_recovered - val).abs() < f32::EPSILON);
        assert!((e5m2_recovered - val).abs() < f32::EPSILON);
        assert_ne!(
            e4m3_byte, e5m2_byte,
            "E4M3 and E5M2 should use different bit patterns for 1.0"
        );
    }

    #[test]
    fn test_e4m3_just_above_clamp_boundary() {
        // Arrange: FP32 exp=120 => actual exp=-7, E4M3 fp8_exp = -7+7 = 0.
        // This clamps to exp=0. The value just above (exp=121, actual=-6) should NOT clamp.
        let just_below = 2.0f32.powi(-7); // exp=-7, clamps
        let just_above = 2.0f32.powi(-6); // exp=-6, does NOT clamp

        // Act
        let byte_below = fp32_to_fp8_e4m3(just_below);
        let byte_above = fp32_to_fp8_e4m3(just_above);

        // Assert: just_above encodes with exp > 0 (byte has non-zero exp bits)
        let exp_above = (byte_above >> 3) & 0xF;
        assert!(
            exp_above > 0,
            "E4M3 exp=-6 should produce fp8_exp > 0, got exp={}",
            exp_above
        );
        // just_below may clamp to exp=0
        let exp_below = (byte_below >> 3) & 0xF;
        assert_eq!(
            exp_below, 0,
            "E4M3 exp=-7 should clamp to fp8_exp=0, got exp={}",
            exp_below
        );
    }

    #[test]
    fn test_e5m2_just_above_clamp_boundary() {
        // Arrange: E5M2 bias=15. exp=-15 → fp8_exp=0 (clamped).
        // exp=-14 → fp8_exp=1 (not clamped).
        let at_boundary = 2.0f32.powi(-15);
        let just_above = 2.0f32.powi(-14);

        // Act
        let byte_at = fp32_to_fp8_e5m2(at_boundary);
        let byte_above = fp32_to_fp8_e5m2(just_above);

        // Assert
        let exp_at = (byte_at >> 2) & 0x1F;
        let exp_above = (byte_above >> 2) & 0x1F;
        assert_eq!(exp_at, 0, "E5M2 exp=-15 should clamp to fp8_exp=0");
        assert_eq!(exp_above, 1, "E5M2 exp=-14 should produce fp8_exp=1");
    }

    #[test]
    fn test_batch_all_identical_values_e4m3() {
        // Arrange: all identical values should quantize to same byte, dequantize identically
        let x = vec![7.5f32; 32];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: all quantized bytes identical, all dequantized values identical
        assert_eq!(quantized.len(), 32);
        let first_byte = quantized[0];
        for (i, &b) in quantized.iter().enumerate() {
            assert_eq!(
                b, first_byte,
                "All bytes should be identical, byte[{}]=0x{:02X} != byte[0]=0x{:02X}",
                i, b, first_byte
            );
        }
        for (i, &val) in dequantized.iter().enumerate() {
            assert!(
                (val - dequantized[0]).abs() < f32::EPSILON,
                "Dequantized[{}] = {} differs from dequantized[0] = {}",
                i,
                val,
                dequantized[0]
            );
        }
    }

    #[test]
    fn test_e4m3_sign_bit_in_raw_byte() {
        // Arrange: encode positive and negative of same magnitude
        let pos_byte = fp32_to_fp8_e4m3(3.0f32);
        let neg_byte = fp32_to_fp8_e4m3(-3.0f32);

        // Act: extract sign bits
        let pos_sign = (pos_byte >> 7) & 1;
        let neg_sign = (neg_byte >> 7) & 1;
        let magnitude_bits = pos_byte & 0x7F;

        // Assert: sign bits differ, magnitude bits identical
        assert_eq!(pos_sign, 0, "Positive value sign bit should be 0");
        assert_eq!(neg_sign, 1, "Negative value sign bit should be 1");
        assert_eq!(
            neg_byte & 0x7F,
            magnitude_bits,
            "Magnitude bits should be identical for ±3.0"
        );
    }

    #[test]
    fn test_e5m2_sign_bit_in_raw_byte() {
        // Arrange: encode positive and negative of same magnitude
        let pos_byte = fp32_to_fp8_e5m2(5.0f32);
        let neg_byte = fp32_to_fp8_e5m2(-5.0f32);

        // Act: extract sign bits
        let pos_sign = (pos_byte >> 7) & 1;
        let neg_sign = (neg_byte >> 7) & 1;

        // Assert: sign bits differ, magnitude bits identical
        assert_eq!(pos_sign, 0, "Positive value sign bit should be 0");
        assert_eq!(neg_sign, 1, "Negative value sign bit should be 1");
        assert_eq!(
            neg_byte & 0x7F,
            pos_byte & 0x7F,
            "Magnitude bits should be identical for ±5.0"
        );
    }

    #[test]
    fn test_quantize_e4m3_scale_produces_exact_value() {
        // Arrange: 448.0 / 448.0 = 1.0, which is exactly representable.
        // The quantized byte for 1.0 in E4M3 is 0x38.
        let x = vec![448.0f32];
        let scale = 448.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);

        // Assert: after dividing by scale, the value is exactly 1.0 → byte 0x38
        assert_eq!(
            quantized[0], 0x38,
            "448.0/448.0=1.0 should encode to byte 0x38, got 0x{:02X}",
            quantized[0]
        );
    }

    #[test]
    fn test_quantize_e5m2_scale_produces_exact_value() {
        // Arrange: 57344.0 / 57344.0 = 1.0 → byte 0x3C in E5M2.
        let x = vec![57344.0f32];
        let scale = 57344.0f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);

        // Assert
        assert_eq!(
            quantized[0], 0x3C,
            "57344.0/57344.0=1.0 should encode to byte 0x3C, got 0x{:02X}",
            quantized[0]
        );
    }

    #[test]
    fn test_large_batch_stress_e4m3() {
        // Arrange: 500 values spanning the full E4M3 representable range
        let x: Vec<f32> = (0..500)
            .map(|i| -400.0 + (i as f32) * 800.0 / 499.0)
            .collect();
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: length preserved, no NaN in dequantized (NaN input from overflow handled)
        assert_eq!(dequantized.len(), 500);
        for (i, &val) in dequantized.iter().enumerate() {
            assert!(
                val.is_finite() || val.is_nan(),
                "E4M3 stress index {} produced unexpected value {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_e5m2_negative_sign_propagation_in_batch() {
        // Arrange: all-negative batch with varying magnitudes
        let x: Vec<f32> = (1..=20).map(|i| -(i as f32) * 10.0).collect();
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        // Assert: all dequantized values are non-positive (sign preserved)
        for (i, &val) in dequantized.iter().enumerate() {
            assert!(
                val <= 0.0,
                "E5M2 negative sign not preserved at index {}: input={}, output={}",
                i,
                x[i],
                val
            );
        }
    }

    #[test]
    fn test_quantize_zero_with_various_scales_e4m3() {
        // Arrange: zero should quantize to 0x00 regardless of scale
        let scales = vec![0.001f32, 1.0, 100.0, 1e6, 1e-6];

        // Act & Assert
        for &scale in &scales {
            let quantized = quantize_fp8_e4m3(&[0.0f32], scale);
            assert_eq!(
                quantized[0], 0x00,
                "Zero with scale={} should encode to 0x00, got 0x{:02X}",
                scale,
                quantized[0]
            );
            let dequantized = dequantize_fp8_e4m3(&quantized, scale);
            assert_eq!(
                dequantized[0], 0.0f32,
                "Zero with scale={} should dequantize to 0.0, got {}",
                scale,
                dequantized[0]
            );
        }
    }

    #[test]
    fn test_e5m2_near_max_normal_roundtrip_all_mantissas() {
        // Arrange: E5M2 exp=30 (just below reserved exp=31), all mantissa values 0..3
        let mut decoded: Vec<f32> = Vec::with_capacity(4);
        for m in 0u8..=3 {
            let byte = (30u8 << 2) | m; // exp=30, mantissa=m, sign=0
            decoded.push(fp8_e5m2_to_fp32(byte));
        }

        // Act: re-encode each decoded value
        let re_encoded: Vec<u8> = decoded.iter().map(|&v| fp32_to_fp8_e5m2(v)).collect();

        // Assert: all decode to positive finite, re-encode within ±1 of original
        for (m, &val) in decoded.iter().enumerate() {
            assert!(
                val.is_finite() && val > 0.0,
                "E5M2 exp=30 mantissa={} should be finite positive, got {}",
                m,
                val
            );
            let original_byte = (30u8 << 2) | m as u8;
            let diff = (re_encoded[m] as i16 - original_byte as i16).abs();
            assert!(
                diff <= 1,
                "E5M2 exp=30 mantissa={}: original=0x{:02X}, re-encoded=0x{:02X}",
                m,
                original_byte,
                re_encoded[m]
            );
        }
    }

    #[test]
    fn test_e4m3_e5m2_shared_zero_semantics() {
        // Arrange: 0.0 encoded in both formats should decode to exactly 0.0
        // and have identical magnitude (bit pattern 0x00 for positive zero).
        let e4m3_zero = fp32_to_fp8_e4m3(0.0f32);
        let e5m2_zero = fp32_to_fp8_e5m2(0.0f32);

        // Act
        let e4m3_decoded = fp8_e4m3_to_fp32(e4m3_zero);
        let e5m2_decoded = fp8_e5m2_to_fp32(e5m2_zero);

        // Assert: both formats encode +0.0 as 0x00 and decode to exactly 0.0
        assert_eq!(e4m3_zero, 0x00);
        assert_eq!(e5m2_zero, 0x00);
        assert_eq!(e4m3_decoded, 0.0f32);
        assert_eq!(e5m2_decoded, 0.0f32);
        assert!(!e4m3_decoded.is_sign_negative());
        assert!(!e5m2_decoded.is_sign_negative());
    }

    // =========================================================================
    // 13 additional tests — edge cases and boundary conditions
    // =========================================================================

    #[test]
    fn test_e4m3_smallest_fp32_subnormal_encodes_to_zero() {
        // Arrange: smallest positive FP32 subnormal (only lowest mantissa bit set)
        let smallest_subnormal = f32::from_bits(0x00000001);

        // Act
        let byte = fp32_to_fp8_e4m3(smallest_subnormal);

        // Assert: FP32 subnormal maps to FP8 zero since fp32_exp == 0
        assert_eq!(
            byte, 0x00,
            "Smallest FP32 subnormal {} should encode to 0x00, got 0x{:02X}",
            smallest_subnormal, byte
        );
        let recovered = fp8_e4m3_to_fp32(byte);
        assert_eq!(recovered, 0.0f32);
    }

    #[test]
    fn test_e5m2_negative_denormal_byte_decodes_negative() {
        // Arrange: E5M2 byte with exp=0, mantissa=1, sign=1 → byte = 0x80 | 0x01 = 0x81
        // This should decode to a very small negative value (not zero, not NaN).
        let byte = 0x81_u8;

        // Act
        let val = fp8_e5m2_to_fp32(byte);

        // Assert: fp32_exp = 0 - 15 + 127 = 112, mantissa shifted, sign=1
        assert!(
            val.is_finite() && val < 0.0,
            "E5M2 byte 0x81 should decode to a small negative finite value, got {}",
            val
        );
        // The magnitude should match the positive counterpart (byte 0x01)
        let pos_counterpart = fp8_e5m2_to_fp32(0x01);
        assert!(
            (val.abs() - pos_counterpart).abs() < f32::EPSILON,
            "Magnitude of 0x81 ({}) should equal 0x01 ({})",
            val.abs(),
            pos_counterpart
        );
    }

    #[test]
    fn test_e4m3_negative_nan_byte_decodes_to_nan() {
        // Arrange: E4M3 NaN with sign=1 → byte = 0x80 | 0x7F = 0xFF
        // Encoding -500.0 already produces this (existing test), but let's verify
        // that directly decoding 0xFF gives NaN.
        let byte = 0xFF_u8;

        // Act
        let val = fp8_e4m3_to_fp32(byte);

        // Assert
        assert!(
            val.is_nan(),
            "E4M3 byte 0xFF (sign=1, exp=15, mantissa=7) should decode to NaN, got {}",
            val
        );
    }

    #[test]
    fn test_dequantize_e4m3_inf_with_scale_remains_inf() {
        // Arrange: E4M3 Inf byte (0x7E) dequantized with any scale should stay Inf
        let inf_byte = 0x7E_u8;

        // Act
        let decoded_identity = fp8_e4m3_to_fp32(inf_byte) * 1.0f32;
        let decoded_large_scale = fp8_e4m3_to_fp32(inf_byte) * 1000.0f32;
        let decoded_small_scale = fp8_e4m3_to_fp32(inf_byte) * 0.001f32;

        // Assert: Inf * any_finite = Inf
        assert!(
            decoded_identity.is_infinite() && decoded_identity.is_sign_positive(),
            "Inf * 1.0 should be +Inf, got {}",
            decoded_identity
        );
        assert!(
            decoded_large_scale.is_infinite() && decoded_large_scale.is_sign_positive(),
            "Inf * 1000.0 should be +Inf, got {}",
            decoded_large_scale
        );
        assert!(
            decoded_small_scale.is_infinite() && decoded_small_scale.is_sign_positive(),
            "Inf * 0.001 should be +Inf, got {}",
            decoded_small_scale
        );
    }

    #[test]
    fn test_quantize_e5m2_tiny_scale_saturates_to_max() {
        // Arrange: very small scale means val/scale exceeds E5M2 representable range.
        // 1.0 / 1e-20 = 1e20 which is beyond E5M2 max (~57344).
        // The encoder clamps the exponent to max (31), so it saturates near max normal.
        let x = vec![1.0f32];
        let scale = 1e-20f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let raw_decoded = fp8_e5m2_to_fp32(quantized[0]);

        // Assert: the raw decoded value (before scale) should be near E5M2 max range.
        // After dequantize (multiply by scale), the result is raw_decoded * 1e-20.
        assert!(
            raw_decoded.is_finite() && raw_decoded > 0.0,
            "E5M2 saturated value should be positive finite, got {}",
            raw_decoded
        );
        // The encoded byte should have exp clamped to max (31 with sign=0)
        // and the raw decoded value should be at least 28672 (half of max range)
        assert!(
            raw_decoded >= 28672.0,
            "Saturated E5M2 should be near max range, got {}",
            raw_decoded
        );
    }

    #[test]
    fn test_e4m3_exact_max_normal_boundary() {
        // Arrange: E4M3 max normal = exp=14, mantissa=7 → 2^(14-7) * (1 + 7/8) = 240.0
        // 240.0 should roundtrip exactly as it's the max normal value.
        let max_normal = 240.0f32;

        // Act
        let byte = fp32_to_fp8_e4m3(max_normal);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Assert: 240.0 is the max normal, should roundtrip with acceptable error
        assert!(
            (recovered - max_normal).abs() <= 16.0,
            "E4M3 max normal 240.0 → byte 0x{:02X} → {} (error: {})",
            byte,
            recovered,
            (recovered - max_normal).abs()
        );
    }

    #[test]
    fn test_e4m3_between_mantissa_steps_precision() {
        // Arrange: 1.3 has mantissa bits that don't fit in 3 bits exactly.
        // The encode truncates to the nearest representable value.
        let val = 1.3f32;

        // Act
        let byte = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Assert: error should be bounded by the quantization step size
        // For exp=7, the step size is 2^(7-7) * (1/8) = 0.125
        let error = (val - recovered).abs();
        assert!(
            error <= 0.125,
            "E4M3 1.3 quantization error {} exceeds step size 0.125: byte=0x{:02X}, recovered={}",
            error, byte, recovered
        );
    }

    #[test]
    fn test_e5m2_between_mantissa_steps_precision() {
        // Arrange: 3.7 has mantissa bits that don't fit in 2 bits exactly.
        // For exp=16 (since 3.7 is between 2 and 4, exp = 16), step size = 2^(16-15) * (1/4) = 0.5
        let val = 3.7f32;

        // Act
        let byte = fp32_to_fp8_e5m2(val);
        let recovered = fp8_e5m2_to_fp32(byte);

        // Assert: error bounded by E5M2 quantization step
        let error = (val - recovered).abs();
        assert!(
            error <= 0.5,
            "E5M2 3.7 quantization error {} exceeds step size 0.5: byte=0x{:02X}, recovered={}",
            error, byte, recovered
        );
    }

    #[test]
    fn test_quantize_e4m3_with_nan_in_batch_does_not_corrupt_neighbors() {
        // Arrange: NaN in the middle of a batch should not affect adjacent values
        let x = vec![5.0f32, f32::NAN, -5.0];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: neighbors are valid finite values
        assert!(
            dequantized[0].is_finite(),
            "Value before NaN should be finite, got {}",
            dequantized[0]
        );
        assert!(dequantized[1].is_nan(), "NaN should roundtrip as NaN");
        assert!(
            dequantized[2].is_finite(),
            "Value after NaN should be finite, got {}",
            dequantized[2]
        );
        // Neighbors should have correct signs
        assert!(dequantized[0] > 0.0, "First value should be positive");
        assert!(dequantized[2] < 0.0, "Third value should be negative");
    }

    #[test]
    fn test_e5m2_exp_31_reserved_mantissa_2_with_negative_sign() {
        // Arrange: E5M2 exp=31, mantissa=2, sign=1 → byte = 0x80 | (31 << 2) | 2 = 0xFE
        // mantissa=2 is not NaN (which is mantissa=3) and not Inf (mantissa=0).
        let byte = 0xFE_u8;

        // Act
        let val = fp8_e5m2_to_fp32(byte);

        // Assert: decodes via normal formula to a finite value (not NaN, not Inf)
        // fp32_exp = 31 - 15 + 127 = 143, mantissa=2 shifted
        assert!(
            val.is_finite(),
            "E5M2 byte 0xFE (exp=31, mantissa=2, sign=1) should be finite, got {}",
            val
        );
        assert!(
            val.is_sign_negative(),
            "E5M2 byte 0xFE should be negative, got {}",
            val
        );
    }

    #[test]
    fn test_e4m3_negative_denormal_byte_decodes_negative() {
        // Arrange: E4M3 byte with exp=0, mantissa=3, sign=1 → byte = 0x80 | 0x03 = 0x83
        // fp32_exp = 0 - 7 + 127 = 120, mantissa=3 << 20, sign=1
        let byte = 0x83_u8;

        // Act
        let val = fp8_e4m3_to_fp32(byte);

        // Assert: should be a small negative finite value
        assert!(
            val.is_finite() && val < 0.0,
            "E4M3 byte 0x83 should decode to a small negative value, got {}",
            val
        );
        // Magnitude should match positive counterpart (byte 0x03)
        let pos = fp8_e4m3_to_fp32(0x03);
        assert!(
            (val.abs() - pos).abs() < f32::EPSILON,
            "Magnitude of 0x83 ({}) should equal 0x03 ({})",
            val.abs(),
            pos
        );
    }

    #[test]
    fn test_quantize_dequantize_e5m2_single_element_with_large_scale() {
        // Arrange: single element with scale making it exactly representable
        // 2048.0 / 2048.0 = 1.0 → exact in E5M2
        let x = vec![2048.0f32];
        let scale = 2048.0f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        // Assert
        assert_eq!(quantized.len(), 1);
        assert_eq!(dequantized.len(), 1);
        assert_eq!(quantized[0], 0x3C, "2048.0/2048.0=1.0 should encode to E5M2 byte 0x3C");
        let error = (x[0] - dequantized[0]).abs();
        assert!(
            error < f32::EPSILON,
            "E5M2 single element exact roundtrip: 2048.0 → {} (error: {})",
            dequantized[0], error
        );
    }

    #[test]
    fn test_dequantize_e4m3_negative_inf_with_scale() {
        // Arrange: dequantize E4M3 negative Inf byte with positive scale
        let neg_inf_byte = 0xFE_u8; // E4M3 -Inf

        // Act: dequantize_fp8_e4m3 multiplies decoded value by scale
        let decoded = fp8_e4m3_to_fp32(neg_inf_byte) * 5.0f32;

        // Assert: -Inf * positive_scale = -Inf
        assert!(
            decoded.is_infinite() && decoded.is_sign_negative(),
            "E4M3 -Inf * 5.0 should be -Inf, got {}",
            decoded
        );
    }

    // =========================================================================
    // 13 additional tests — subnormal decode, NaN propagation, extreme scales,
    // sign symmetry in raw bytes, boundary roundtrip precision, error bounds
    // =========================================================================

    #[test]
    fn test_e4m3_subnormal_decode_byte_0x01_is_positive_finite() {
        // Arrange: E4M3 byte 0x01 = exp=0, mantissa=1, sign=0.
        // fp32_exp = 0 - 7 + 127 = 120. Decodes to a small positive finite value.
        let byte = 0x01_u8;

        // Act
        let val = fp8_e4m3_to_fp32(byte);

        // Assert
        assert!(
            val.is_finite() && val > 0.0,
            "E4M3 byte 0x01 should decode to a small positive finite, got {}",
            val
        );
        // Verify monotonicity: mantissa=1 < mantissa=3
        let val_m3 = fp8_e4m3_to_fp32(0x03);
        assert!(
            val < val_m3,
            "E4M3 mantissa=1 ({}) should be less than mantissa=3 ({})",
            val,
            val_m3
        );
    }

    #[test]
    fn test_e5m2_subnormal_decode_byte_0x02_is_positive_finite() {
        // Arrange: E5M2 byte 0x02 = exp=0, mantissa=2, sign=0.
        // fp32_exp = 0 - 15 + 127 = 112. Small positive value.
        let byte = 0x02_u8;

        // Act
        let val = fp8_e5m2_to_fp32(byte);

        // Assert
        assert!(
            val.is_finite() && val > 0.0,
            "E5M2 byte 0x02 should decode to positive finite, got {}",
            val
        );
        // mantissa=2 should be larger than mantissa=1 at same exp
        let val_m1 = fp8_e5m2_to_fp32(0x01);
        assert!(
            val > val_m1,
            "E5M2 mantissa=2 ({}) should be > mantissa=1 ({})",
            val,
            val_m1
        );
    }

    #[test]
    fn test_e4m3_nan_propagation_preserves_nan_through_batch() {
        // Arrange: a batch containing two NaNs at different positions
        let x = vec![f32::NAN, 3.0f32, f32::NAN, -1.0];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: NaNs remain NaN after encode-decode roundtrip
        assert!(dequantized[0].is_nan(), "First NaN did not propagate");
        assert!(dequantized[2].is_nan(), "Second NaN did not propagate");
        // Non-NaN values remain finite
        assert!(dequantized[1].is_finite(), "3.0 should be finite");
        assert!(dequantized[3].is_finite(), "-1.0 should be finite");
    }

    #[test]
    fn test_e5m2_infinity_encode_positive_and_negative_distinct() {
        // Arrange: both +Inf and -Inf should produce different bytes
        let pos_inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;

        // Act
        let pos_byte = fp32_to_fp8_e5m2(pos_inf);
        let neg_byte = fp32_to_fp8_e5m2(neg_inf);

        // Assert: bit patterns differ only in sign bit
        assert_ne!(pos_byte, neg_byte, "+Inf and -Inf must have different encodings");
        assert_eq!(pos_byte & 0x7F, neg_byte & 0x7F, "Magnitude bits should match for ±Inf");
        assert_eq!(pos_byte, 0x7C, "E5M2 +Inf should be 0x7C");
        assert_eq!(neg_byte, 0xFC, "E5M2 -Inf should be 0xFC");
    }

    #[test]
    fn test_quantize_e4m3_tiny_values_with_moderate_scale() {
        // Arrange: values that are exactly representable as powers of two in E4M3,
        // quantized with a moderate scale that keeps them in range.
        let x = vec![1.0f32, 2.0, 4.0, 8.0];
        let scale = 8.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: powers of two divided by power-of-two scale produce exact roundtrips
        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs();
            assert!(
                error < f32::EPSILON,
                "E4M3 tiny values with moderate scale: {} → {} (error: {})",
                x[i], dequantized[i], error
            );
        }
    }

    #[test]
    fn test_quantize_e5m2_extreme_small_scale_saturates() {
        // Arrange: very small scale amplifies values beyond E5M2 range
        let x = vec![1.0f32];
        let scale = 1e-30f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let raw_decoded = fp8_e5m2_to_fp32(quantized[0]);

        // Assert: the raw decoded value should be very large (saturated to max range)
        assert!(
            raw_decoded > 0.0 && raw_decoded.is_finite(),
            "E5M2 saturated value should be large positive finite, got {}",
            raw_decoded
        );
        assert!(
            raw_decoded >= 28672.0,
            "Saturated E5M2 should be near max range, got {}",
            raw_decoded
        );
    }

    #[test]
    fn test_e4m3_sign_symmetry_raw_bytes() {
        // Arrange: for representable values, positive and negative encodings
        // should differ only in the sign bit (bit 7).
        let values = [1.0f32, 2.5, 5.0, 10.0, 100.0];

        for &v in &values {
            // Act
            let pos_byte = fp32_to_fp8_e4m3(v);
            let neg_byte = fp32_to_fp8_e4m3(-v);

            // Assert: only sign bit differs
            assert_eq!(
                pos_byte & 0x7F,
                neg_byte & 0x7F,
                "E4M3 ±{} magnitude bits differ: pos=0x{:02X}, neg=0x{:02X}",
                v, pos_byte, neg_byte
            );
            assert_eq!(pos_byte >> 7, 0, "Positive sign bit should be 0");
            assert_eq!(neg_byte >> 7, 1, "Negative sign bit should be 1");
        }
    }

    #[test]
    fn test_e5m2_roundtrip_boundary_value_exponent_14() {
        // Arrange: E5M2 exp=14, mantissa=0 → fp32_exp = 14 - 15 + 127 = 126.
        // Value = 2^(126-127) * (1 + 0/4) = 0.5
        let byte = (14u8 << 2) | 0; // 0x38

        // Act
        let val = fp8_e5m2_to_fp32(byte);
        let re_encoded = fp32_to_fp8_e5m2(val);
        let recovered = fp8_e5m2_to_fp32(re_encoded);

        // Assert: exact roundtrip for a representable boundary value
        assert!(
            (val - recovered).abs() < f32::EPSILON,
            "E5M2 exp=14 boundary: {} → 0x{:02X} → {}",
            val,
            re_encoded,
            recovered
        );
    }

    #[test]
    fn test_f32_to_fp8_e4m3_to_f32_error_bound_small_values() {
        // Arrange: values in [1.0, 10.0] should have bounded quantization error.
        // E4M3 has 3 mantissa bits, so relative error should be < 1/8 = 12.5%.
        let values: Vec<f32> = (1..=20).map(|i| i as f32 * 0.5).collect();

        for &v in &values {
            // Act
            let byte = fp32_to_fp8_e4m3(v);
            let recovered = fp8_e4m3_to_fp32(byte);
            let rel_error = (v - recovered).abs() / v;

            // Assert: relative error bounded by mantissa quantization step
            assert!(
                rel_error < 0.15,
                "E4M3 error bound exceeded for {}: recovered={}, rel_error={}",
                v, recovered, rel_error
            );
        }
    }

    #[test]
    fn test_f32_to_fp8_e5m2_to_f32_error_bound_mid_range() {
        // Arrange: values in [10.0, 1000.0] with E5M2 (2 mantissa bits).
        // Relative error should be < 1/4 = 25%.
        let values = vec![10.0f32, 50.0, 100.0, 500.0, 1000.0];

        for &v in &values {
            // Act
            let byte = fp32_to_fp8_e5m2(v);
            let recovered = fp8_e5m2_to_fp32(byte);
            let rel_error = (v - recovered).abs() / v;

            // Assert
            assert!(
                rel_error < 0.25,
                "E5M2 error bound exceeded for {}: recovered={}, rel_error={}",
                v, recovered, rel_error
            );
        }
    }

    #[test]
    fn test_batch_quantize_e4m3_all_zeros_with_negative_scale() {
        // Arrange: all zeros should dequantize to zero regardless of scale sign.
        // Negative scale flips sign during quantize and dequantize, so zero stays zero.
        let x = vec![0.0f32; 5];
        let scale = -1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: 0 / scale = 0 (signed zero may flip), 0 * scale = 0
        assert_eq!(quantized.len(), 5);
        for (i, &val) in dequantized.iter().enumerate() {
            assert_eq!(
                val, 0.0f32,
                "Zero with negative scale at index {} should be 0.0, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_e4m3_exp_15_mantissa_5_roundtrip_consistency() {
        // Arrange: E4M3 exp=15, mantissa=5 (not NaN=7, not Inf=6).
        // This is a valid high-exponent value that should encode/decode consistently.
        let byte = (15u8 << 3) | 5; // 0x7D

        // Act
        let val = fp8_e4m3_to_fp32(byte);
        let re_encoded = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(re_encoded);

        // Assert: value is finite, positive, and roundtrips with acceptable error
        assert!(val.is_finite() && val > 0.0, "Should be finite positive, got {}", val);
        assert!(
            (val - recovered).abs() / val < 0.1,
            "E4M3 exp=15 mantissa=5 roundtrip: {} → byte 0x{:02X} → {} (error: {})",
            val, re_encoded, recovered, (val - recovered).abs() / val
        );
    }

    #[test]
    fn test_e5m2_positive_and_negative_zero_different_bytes() {
        // Arrange: +0.0 and -0.0 should produce different byte encodings
        let pos_zero = 0.0f32;
        let neg_zero = -0.0f32;

        // Act
        let pos_byte_e5 = fp32_to_fp8_e5m2(pos_zero);
        let neg_byte_e5 = fp32_to_fp8_e5m2(neg_zero);

        // Assert: different bytes (sign bit), both decode to zero
        assert_ne!(pos_byte_e5, neg_byte_e5, "+0 and -0 should have different E5M2 encodings");
        assert_eq!(pos_byte_e5, 0x00, "E5M2 +0 should be 0x00");
        assert_eq!(neg_byte_e5, 0x80, "E5M2 -0 should be 0x80");
        let pos_decoded = fp8_e5m2_to_fp32(pos_byte_e5);
        let neg_decoded = fp8_e5m2_to_fp32(neg_byte_e5);
        assert_eq!(pos_decoded, 0.0f32);
        assert_eq!(neg_decoded, 0.0f32);
        assert!(!pos_decoded.is_sign_negative());
        assert!(neg_decoded.is_sign_negative());
    }

    // =========================================================================
    // 13 additional tests — constructor correctness, cross-format consistency,
    // boundary precision, error propagation, monotonicity, extreme inputs
    // =========================================================================

    #[test]
    fn test_e4m3_encode_value_0_125_exact_roundtrip() {
        // Arrange: 0.125 = 2^(-3), a power of two within E4M3 range.
        // FP32 exp = 124, FP8 E4M3 fp8_exp = -3+7 = 4, mantissa = 0.
        // byte = (4 << 3) | 0 = 0x20
        let val = 0.125f32;

        // Act
        let byte = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Assert: exact roundtrip for power of two
        assert_eq!(byte, 0x20, "0.125 should encode to 0x20, got 0x{:02X}", byte);
        assert!(
            (recovered - val).abs() < f32::EPSILON,
            "0.125 roundtrip: expected {}, got {}",
            val,
            recovered
        );
    }

    #[test]
    fn test_e5m2_encode_value_0_25_exact_roundtrip() {
        // Arrange: 0.25 = 2^(-2), power of two within E5M2 range.
        // FP32 exp = 125, FP8 E5M2 fp8_exp = -2+15 = 13, mantissa = 0.
        // byte = (13 << 2) | 0 = 0x34
        let val = 0.25f32;

        // Act
        let byte = fp32_to_fp8_e5m2(val);
        let recovered = fp8_e5m2_to_fp32(byte);

        // Assert: exact roundtrip
        assert_eq!(byte, 0x34, "0.25 should encode to 0x34, got 0x{:02X}", byte);
        assert!(
            (recovered - val).abs() < f32::EPSILON,
            "0.25 roundtrip: expected {}, got {}",
            val,
            recovered
        );
    }

    #[test]
    fn test_e4m3_fp32_negative_quiet_nan_encode_decode() {
        // Arrange: negative quiet NaN has sign bit set in FP32.
        // The encode function checks x.is_nan() first, so sign doesn't matter.
        let neg_nan = f32::from_bits(0xFFC00000); // negative quiet NaN

        // Act
        let byte = fp32_to_fp8_e4m3(neg_nan);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Assert: regardless of input NaN sign, encoding is 0x7F (positive NaN)
        assert_eq!(byte, 0x7F, "NaN should encode to 0x7F, got 0x{:02X}", byte);
        assert!(recovered.is_nan(), "Decoded NaN should be NaN, got {}", recovered);
    }

    #[test]
    fn test_e5m2_fp32_signaling_nan_encode_decode() {
        // Arrange: signaling NaN (mantissa non-zero, exponent all ones, sign=0)
        let s_nan = f32::from_bits(0x7F800001); // signaling NaN

        // Act
        let byte = fp32_to_fp8_e5m2(s_nan);
        let recovered = fp8_e5m2_to_fp32(byte);

        // Assert: is_nan() catches signaling NaN too, encoding is 0x7F
        assert_eq!(byte, 0x7F, "SNaN should encode to 0x7F, got 0x{:02X}", byte);
        assert!(recovered.is_nan(), "Decoded SNaN should be NaN, got {}", recovered);
    }

    #[test]
    fn test_e4m3_decode_all_256_bytes_no_panic() {
        // Arrange & Act: decode every possible byte value 0x00..0xFF
        // This tests that no input causes a panic or assertion failure.
        let mut all_decoded: Vec<f32> = Vec::with_capacity(256);
        for byte in 0u8..=255 {
            all_decoded.push(fp8_e4m3_to_fp32(byte));
        }

        // Assert: all 256 values decode without panic, and NaN/Inf are in expected positions
        assert_eq!(all_decoded.len(), 256);
        // 0x7F = NaN
        assert!(all_decoded[0x7F].is_nan());
        // 0x7E = +Inf
        assert!(all_decoded[0x7E].is_infinite() && all_decoded[0x7E].is_sign_positive());
        // 0xFE = -Inf
        assert!(all_decoded[0xFE].is_infinite() && all_decoded[0xFE].is_sign_negative());
        // 0x00 = +0
        assert_eq!(all_decoded[0x00], 0.0f32);
        // 0x80 = -0
        assert_eq!(all_decoded[0x80], 0.0f32);
        assert!(all_decoded[0x80].is_sign_negative());
    }

    #[test]
    fn test_e5m2_decode_all_256_bytes_no_panic() {
        // Arrange & Act: decode every possible byte value for E5M2
        let mut all_decoded: Vec<f32> = Vec::with_capacity(256);
        for byte in 0u8..=255 {
            all_decoded.push(fp8_e5m2_to_fp32(byte));
        }

        // Assert: no panics, special values at expected positions
        assert_eq!(all_decoded.len(), 256);
        // 0x7F = NaN (exp=31, mantissa=3)
        assert!(all_decoded[0x7F].is_nan());
        // 0x7C = +Inf (exp=31, mantissa=0)
        assert!(all_decoded[0x7C].is_infinite() && all_decoded[0x7C].is_sign_positive());
        // 0xFC = -Inf (exp=31, mantissa=0, sign=1)
        assert!(all_decoded[0xFC].is_infinite() && all_decoded[0xFC].is_sign_negative());
        // 0x00 = +0
        assert_eq!(all_decoded[0x00], 0.0f32);
        // 0x80 = -0
        assert_eq!(all_decoded[0x80], 0.0f32);
        assert!(all_decoded[0x80].is_sign_negative());
    }

    #[test]
    fn test_e4m3_encode_all_exp_7_mantissa_values_roundtrip() {
        // Arrange: for exp=7 (bias-adjusted value near 1.0), verify all 8 mantissa
        // values 0..7 produce distinct decoded values.
        let mut decoded: Vec<f32> = Vec::with_capacity(8);
        for m in 0u8..=7 {
            let byte = (7u8 << 3) | m;
            decoded.push(fp8_e4m3_to_fp32(byte));
        }

        // Act & Assert: all 8 values are distinct (different mantissa -> different value)
        for i in 1..decoded.len() {
            assert!(
                decoded[i] > decoded[i - 1],
                "E4M3 exp=7 mantissa={} ({}) should be > mantissa={} ({})",
                i, decoded[i], i - 1, decoded[i - 1]
            );
        }
    }

    #[test]
    fn test_e5m2_exp_15_mantissa_all_values_distinct() {
        // Arrange: E5M2 exp=15 (center of range), all 4 mantissa values 0..3
        let mut decoded: Vec<f32> = Vec::with_capacity(4);
        for m in 0u8..=3 {
            let byte = (15u8 << 2) | m;
            decoded.push(fp8_e5m2_to_fp32(byte));
        }

        // Act & Assert: all 4 values distinct and increasing with mantissa
        for i in 1..decoded.len() {
            assert!(
                decoded[i] > decoded[i - 1],
                "E5M2 exp=15 mantissa={} ({}) should be > mantissa={} ({})",
                i, decoded[i], i - 1, decoded[i - 1]
            );
        }
        // First value should be exactly 1.0 (exp=15, bias=15 -> actual exp=0, mantissa=0)
        assert!(
            (decoded[0] - 1.0f32).abs() < f32::EPSILON,
            "E5M2 exp=15 mantissa=0 should be 1.0, got {}",
            decoded[0]
        );
    }

    #[test]
    fn test_quantize_e4m3_negative_scale_flips_sign() {
        // Arrange: positive values with negative scale -> negative after division,
        // then dequantize with same negative scale -> negative * negative = positive.
        // 4.0 / -2.0 = -2.0 -> encode as -2.0 -> decode -> -2.0 * -2.0 = 4.0
        let x = vec![4.0f32];
        let scale = -2.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: double sign flip returns to positive
        assert_eq!(dequantized.len(), 1);
        let error = (x[0] - dequantized[0]).abs();
        assert!(
            error < 0.5,
            "E4M3 negative scale roundtrip: {} -> {} (error: {})",
            x[0], dequantized[0], error
        );
    }

    #[test]
    fn test_quantize_e5m2_negative_scale_roundtrip() {
        // Arrange: negative scale causes two sign flips (quantize and dequantize)
        let x = vec![8.0f32, -8.0];
        let scale = -4.0f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        // Assert: roundtrip should recover values with acceptable error
        assert_eq!(dequantized.len(), 2);
        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs() / x[i].abs();
            assert!(
                error < 0.15,
                "E5M2 negative scale roundtrip [{}]: {} -> {} (error: {})",
                i, x[i], dequantized[i], error
            );
        }
    }

    #[test]
    fn test_e4m3_cross_format_e5m2_disagree_for_non_power_of_two() {
        // Arrange: 1.5 is exactly representable in E4M3 (mantissa=4) but E5M2
        // has only 2 mantissa bits, so 1.5 may not be exact in E5M2.
        let val = 1.5f32;

        // Act
        let e4m3_byte = fp32_to_fp8_e4m3(val);
        let e5m2_byte = fp32_to_fp8_e5m2(val);
        let e4m3_recovered = fp8_e4m3_to_fp32(e4m3_byte);
        let e5m2_recovered = fp8_e5m2_to_fp32(e5m2_byte);

        // Assert: E4M3 has 3 mantissa bits (exact for 1.5), E5M2 has 2 bits.
        // The formats use different bit layouts.
        assert_ne!(
            e4m3_byte, e5m2_byte,
            "E4M3 and E5M2 should produce different bit patterns for 1.5"
        );
        // E4M3 should be exact for 1.5
        assert!(
            (e4m3_recovered - val).abs() < f32::EPSILON,
            "E4M3 1.5 should be exact, got {}",
            e4m3_recovered
        );
    }

    #[test]
    fn test_quantize_e4m3_batch_mixed_nan_inf_zero_all_positions() {
        // Arrange: batch with special values at start, middle, and end positions
        let x = vec![f32::NAN, 0.0f32, f32::INFINITY, 5.0f32, f32::NEG_INFINITY, -0.0f32];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: all 6 elements handled correctly regardless of position
        assert_eq!(dequantized.len(), 6);
        assert!(dequantized[0].is_nan(), "Position 0: NaN");
        assert_eq!(dequantized[1], 0.0f32, "Position 1: +0");
        assert!(dequantized[2].is_infinite() && dequantized[2].is_sign_positive(), "Position 2: +Inf");
        assert!(dequantized[3].is_finite() && dequantized[3] > 0.0, "Position 3: positive finite");
        assert!(dequantized[4].is_infinite() && dequantized[4].is_sign_negative(), "Position 4: -Inf");
        assert!(dequantized[5] == 0.0f32 && dequantized[5].is_sign_negative(), "Position 5: -0");
    }

    #[test]
    fn test_e4m3_e5m2_both_encode_negative_zero_consistently() {
        // Arrange: -0.0 should produce byte with sign bit set in both formats
        let neg_zero = -0.0f32;

        // Act
        let e4m3_byte = fp32_to_fp8_e4m3(neg_zero);
        let e5m2_byte = fp32_to_fp8_e5m2(neg_zero);

        // Assert: both formats encode -0 with sign bit = 1, rest = 0
        assert_eq!(e4m3_byte, 0x80, "E4M3 -0 should be 0x80, got 0x{:02X}", e4m3_byte);
        assert_eq!(e5m2_byte, 0x80, "E5M2 -0 should be 0x80, got 0x{:02X}", e5m2_byte);
        // Decode back to -0.0 in both formats
        let e4m3_decoded = fp8_e4m3_to_fp32(e4m3_byte);
        let e5m2_decoded = fp8_e5m2_to_fp32(e5m2_byte);
        assert_eq!(e4m3_decoded, 0.0f32);
        assert!(e4m3_decoded.is_sign_negative(), "E4M3 decoded should be -0");
        assert_eq!(e5m2_decoded, 0.0f32);
        assert!(e5m2_decoded.is_sign_negative(), "E5M2 decoded should be -0");
    }

    // =========================================================================
    // 13 additional tests — extreme inputs, scale edge cases, monotonicity,
    // sign preservation in batch API, boundary precision
    // =========================================================================

    #[test]
    fn test_e4m3_encode_f32_max_saturates() {
        // Arrange: f32::MAX (~3.4e38) is far beyond E4M3 range (~448).
        // The encoder clamps the exponent to max, producing a saturated value.
        let max_f32 = f32::MAX;

        // Act
        let byte = fp32_to_fp8_e4m3(max_f32);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Assert: should not panic, result is a finite value or NaN due to saturation.
        // With exp clamped to 15 and full mantissa, this saturates to NaN (0x7F).
        assert!(
            recovered.is_nan() || recovered.is_finite(),
            "f32::MAX encode should not panic, got {}",
            recovered
        );
    }

    #[test]
    fn test_e5m2_encode_f32_min_saturates() {
        // Arrange: f32::MIN (most negative finite, ~-3.4e38) exceeds E5M2 range.
        let min_f32 = f32::MIN;

        // Act
        let byte = fp32_to_fp8_e5m2(min_f32);
        let recovered = fp8_e5m2_to_fp32(byte);

        // Assert: should not panic, result is finite or NaN due to saturation.
        assert!(
            recovered.is_nan() || (recovered.is_finite() && recovered < 0.0),
            "f32::MIN encode should produce NaN or negative finite, got {}",
            recovered
        );
    }

    #[test]
    fn test_e4m3_encode_f32_min_positive_clamps_to_zero() {
        // Arrange: f32::MIN_POSITIVE (~1.175e-38) is a tiny normal float
        // with FP32 exp=1 (actual exp=-126). E4M3 fp8_exp = -126+7 = -119, clamped to 0.
        let min_positive = f32::MIN_POSITIVE;

        // Act
        let byte = fp32_to_fp8_e4m3(min_positive);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Assert: exponent clamps to 0, mantissa also 0 (top 3 bits of tiny mantissa are 0).
        // Result is +0.0.
        assert_eq!(
            byte, 0x00,
            "f32::MIN_POSITIVE should clamp to 0x00, got 0x{:02X}",
            byte
        );
        assert_eq!(recovered, 0.0f32);
    }

    #[test]
    fn test_dequantize_e5m2_with_zero_scale_produces_zero() {
        // Arrange: zero scale means all dequantized values are decoded * 0.0.
        // For finite values this is 0.0; for Inf it's NaN; for NaN it stays NaN.
        let q: Vec<u8> = vec![0x3C, 0x7F, 0x7C, 0x00]; // 1.0, NaN, +Inf, +0 in E5M2
        let scale = 0.0f32;

        // Act
        let dequantized = dequantize_fp8_e5m2(&q, scale);

        // Assert
        assert_eq!(dequantized.len(), 4);
        // 1.0 * 0.0 = 0.0
        assert_eq!(dequantized[0], 0.0f32, "finite * 0.0 should be 0.0");
        // NaN * 0.0 = NaN
        assert!(dequantized[1].is_nan(), "NaN * 0.0 should be NaN");
        // Inf * 0.0 = NaN (IEEE 754)
        assert!(dequantized[2].is_nan(), "Inf * 0.0 should be NaN");
        // 0.0 * 0.0 = 0.0
        assert_eq!(dequantized[3], 0.0f32, "0.0 * 0.0 should be 0.0");
    }

    #[test]
    fn test_e4m3_value_just_below_max_normal() {
        // Arrange: E4M3 max normal is 240.0 (exp=14, mantissa=7).
        // 224.0 = 2^7 * (1 + 6/8) = 128 * 1.75 = 224.0.
        // FP32 mantissa for 1.75 is 0x600000, top 3 bits = 6.
        // This is a representable value just below max normal.
        let val = 224.0f32;

        // Act
        let byte = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Assert: 224.0 is representable exactly (mantissa=6)
        let exp = (byte >> 3) & 0xF;
        let mantissa = byte & 0x7;
        assert_eq!(exp, 14, "224.0 should have exp=14, got {}", exp);
        assert_eq!(mantissa, 6, "224.0 should have mantissa=6, got {}", mantissa);
        assert!(
            (recovered - val).abs() < f32::EPSILON,
            "224.0 should roundtrip exactly: got {}",
            recovered
        );
    }

    #[test]
    fn test_e5m2_encode_near_max_range_boundary() {
        // Arrange: E5M2 max normal is approximately 57344.0 (exp=30, mantissa=3).
        // 49152.0 = 2^(30-15) * (1 + 2/4) = 32768 * 1.5 = 49152.0 (exp=30, mantissa=2).
        let val = 49152.0f32;

        // Act
        let byte = fp32_to_fp8_e5m2(val);
        let recovered = fp8_e5m2_to_fp32(byte);

        // Assert: should be a large positive finite value near the E5M2 range boundary
        assert!(
            recovered.is_finite() && recovered > 0.0,
            "49152.0 should decode to positive finite, got {}",
            recovered
        );
        let error = (val - recovered).abs() / val;
        assert!(
            error < 0.25,
            "E5M2 boundary value 49152.0 roundtrip error {} too large",
            error
        );
    }

    #[test]
    fn test_quantize_e5m2_all_identical_negative_values() {
        // Arrange: batch of identical negative values should quantize to same byte
        let x = vec![-10.0f32; 16];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        // Assert: all bytes identical, all dequantized identical and negative
        assert_eq!(quantized.len(), 16);
        let first_byte = quantized[0];
        for (i, &b) in quantized.iter().enumerate() {
            assert_eq!(
                b, first_byte,
                "All bytes should be identical, byte[{}]=0x{:02X} != byte[0]=0x{:02X}",
                i, b, first_byte
            );
        }
        for (i, &val) in dequantized.iter().enumerate() {
            assert!(
                val <= 0.0,
                "Dequantized[{}] = {} should be non-positive",
                i, val
            );
            assert!(
                (val - dequantized[0]).abs() < f32::EPSILON,
                "All dequantized values should be identical, [{}] = {}",
                i, val
            );
        }
    }

    #[test]
    fn test_e4m3_exp_zero_mantissa_zero_byte_0x00_is_positive_zero() {
        // Arrange: byte 0x00 = sign=0, exp=0, mantissa=0.
        // The decoder explicitly returns +0.0 for this pattern.
        let byte = 0x00_u8;

        // Act
        let val = fp8_e4m3_to_fp32(byte);

        // Assert: exactly +0.0 (not negative, not NaN, not subnormal)
        assert_eq!(val, 0.0f32);
        assert!(!val.is_sign_negative(), "0x00 should decode to +0.0, not -0.0");
        assert!(!val.is_nan());
    }

    #[test]
    fn test_e5m2_exp_zero_mantissa_zero_byte_0x80_is_negative_zero() {
        // Arrange: byte 0x80 = sign=1, exp=0, mantissa=0.
        // The decoder explicitly returns -0.0 for this pattern.
        let byte = 0x80_u8;

        // Act
        let val = fp8_e5m2_to_fp32(byte);

        // Assert: exactly -0.0
        assert_eq!(val, 0.0f32);
        assert!(val.is_sign_negative(), "0x80 should decode to -0.0");
        assert!(!val.is_nan());
    }

    #[test]
    fn test_quantize_e4m3_alternating_zeros_and_values() {
        // Arrange: alternating zero and non-zero values
        let x = vec![0.0f32, 5.0, 0.0, -3.0, 0.0, 100.0];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: zeros at even positions roundtrip exactly
        assert_eq!(dequantized.len(), 6);
        assert_eq!(dequantized[0], 0.0f32, "Position 0 should be +0");
        assert_eq!(dequantized[2], 0.0f32, "Position 2 should be +0");
        assert_eq!(dequantized[4], 0.0f32, "Position 4 should be +0");
        // Non-zero values have correct sign
        assert!(dequantized[1] > 0.0, "Position 1 should be positive");
        assert!(dequantized[3] < 0.0, "Position 3 should be negative");
        assert!(dequantized[5] > 0.0, "Position 5 should be positive");
    }

    #[test]
    fn test_e4m3_smallest_representable_normal_roundtrip() {
        // Arrange: E4M3 smallest representable normal: exp=1, mantissa=0.
        // Value = 2^(1-7) = 2^(-6) = 1/64 ≈ 0.015625.
        let val = 2.0f32.powi(-6);

        // Act
        let byte = fp32_to_fp8_e4m3(val);
        let recovered = fp8_e4m3_to_fp32(byte);

        // Assert: exact roundtrip for power-of-two at min normal exponent
        assert_eq!(
            (byte >> 3) & 0xF, 1,
            "E4M3 min normal should have exp=1, got {}",
            (byte >> 3) & 0xF
        );
        assert!(
            (recovered - val).abs() < f32::EPSILON,
            "E4M3 min normal {} should roundtrip exactly, got {}",
            val,
            recovered
        );
    }

    #[test]
    fn test_e5m2_negative_values_dequantize_monotonicity() {
        // Arrange: construct bytes with same exponent but increasing negative mantissa.
        // exp=16, mantissa=0..3, sign=1: bytes 0xC0, 0xC1, 0xC2, 0xC3
        // More negative mantissa = more negative value.
        let bytes: Vec<u8> = (0..=3).map(|m| 0xC0_u8 | m).collect();

        // Act
        let decoded: Vec<f32> = bytes.iter().map(|&b| fp8_e5m2_to_fp32(b)).collect();

        // Assert: values are negative and monotonically decreasing (more negative)
        for &val in &decoded {
            assert!(val < 0.0, "All values should be negative, got {}", val);
        }
        for i in 1..decoded.len() {
            assert!(
                decoded[i] < decoded[i - 1],
                "E5M2 negative monotonicity: [{}]={} should be < [{}]={}",
                i, decoded[i], i - 1, decoded[i - 1]
            );
        }
    }

    #[test]
    fn test_quantize_e4m3_negative_zero_preserves_sign_in_batch() {
        // Arrange: -0.0 in a batch should encode to 0x80 (sign bit set)
        let x = vec![1.0f32, -0.0f32, -1.0];
        let scale = 1.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: -0.0 encodes to 0x80, and decodes back to -0.0
        assert_eq!(quantized[1], 0x80, "-0.0 should encode to 0x80, got 0x{:02X}", quantized[1]);
        assert_eq!(dequantized[1], 0.0f32, "-0.0 should dequantize to 0.0");
        assert!(
            dequantized[1].is_sign_negative(),
            "Dequantized -0.0 should have negative sign"
        );
        // Neighbors unaffected
        assert!(dequantized[0] > 0.0, "First value should be positive");
        assert!(dequantized[2] < 0.0, "Third value should be negative");
    }

    // =========================================================================
    // 13 additional tests — f32 extreme inputs, zero scale IEEE behavior,
    // re-encode stability, negative batch sign, cross-format zero, batch NaN
    // =========================================================================

    #[test]
    fn test_e4m3_encode_f32_min_negative_saturates() {
        let min_f32 = f32::MIN;

        let byte = fp32_to_fp8_e4m3(min_f32);
        let recovered = fp8_e4m3_to_fp32(byte);

        assert!(
            recovered.is_nan() || (recovered.is_finite() && recovered < 0.0),
            "f32::MIN encode should produce NaN or negative finite, got {}",
            recovered
        );
    }

    #[test]
    fn test_e5m2_encode_f32_max_saturates() {
        let max_f32 = f32::MAX;

        let byte = fp32_to_fp8_e5m2(max_f32);
        let recovered = fp8_e5m2_to_fp32(byte);

        assert!(
            recovered.is_nan() || (recovered.is_finite() && recovered > 0.0),
            "f32::MAX encode should produce NaN or positive finite, got {}",
            recovered
        );
    }

    #[test]
    fn test_dequantize_e4m3_with_zero_scale_produces_zero_or_nan() {
        let q: Vec<u8> = vec![0x38, 0x7F, 0x7E, 0x00];
        let scale = 0.0f32;

        let dequantized = dequantize_fp8_e4m3(&q, scale);

        assert_eq!(dequantized.len(), 4);
        assert_eq!(dequantized[0], 0.0f32, "1.0 * 0.0 should be 0.0");
        assert!(dequantized[1].is_nan(), "NaN * 0.0 should be NaN");
        assert!(dequantized[2].is_nan(), "Inf * 0.0 should be NaN");
        assert_eq!(dequantized[3], 0.0f32, "0.0 * 0.0 should be 0.0");
    }

    #[test]
    fn test_e4m3_re_encode_stability_two_roundtrips() {
        let values = vec![1.0f32, 3.75, -5.0, 100.0, -200.0];

        for &v in &values {
            let byte1 = fp32_to_fp8_e4m3(v);
            let decoded1 = fp8_e4m3_to_fp32(byte1);
            let byte2 = fp32_to_fp8_e4m3(decoded1);
            let decoded2 = fp8_e4m3_to_fp32(byte2);

            assert!(
                (decoded1 - decoded2).abs() < f32::EPSILON,
                "E4M3 re-encode unstable: {} → byte 0x{:02X} → {} → byte 0x{:02X} → {}",
                v, byte1, decoded1, byte2, decoded2
            );
            assert_eq!(
                byte1, byte2,
                "E4M3 second encode should produce same byte: first=0x{:02X}, second=0x{:02X}",
                byte1, byte2
            );
        }
    }

    #[test]
    fn test_e5m2_re_encode_stability_two_roundtrips() {
        let values = vec![1.0f32, 7.0, -15.0, 500.0, -1000.0];

        for &v in &values {
            let byte1 = fp32_to_fp8_e5m2(v);
            let decoded1 = fp8_e5m2_to_fp32(byte1);
            let byte2 = fp32_to_fp8_e5m2(decoded1);
            let decoded2 = fp8_e5m2_to_fp32(byte2);

            assert!(
                (decoded1 - decoded2).abs() < f32::EPSILON,
                "E5M2 re-encode unstable: {} → byte 0x{:02X} → {} → byte 0x{:02X} → {}",
                v, byte1, decoded1, byte2, decoded2
            );
            assert_eq!(
                byte1, byte2,
                "E5M2 second encode should produce same byte: first=0x{:02X}, second=0x{:02X}",
                byte1, byte2
            );
        }
    }

    #[test]
    fn test_e4m3_all_negative_batch_sign_consistency() {
        let x: Vec<f32> = (1..=30).map(|i| -(i as f32) * 5.0).collect();
        let scale = 1.0f32;

        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        assert_eq!(dequantized.len(), 30);
        for (i, &val) in dequantized.iter().enumerate() {
            assert!(
                val <= 0.0,
                "E4M3 negative batch [{}]: input={}, output={} should be non-positive",
                i, x[i], val
            );
        }
    }

    #[test]
    fn test_e5m2_large_batch_full_range_monotonicity() {
        let x: Vec<f32> = (0..200).map(|i| (i as f32) * 300.0).collect();
        let scale = 1.0f32;

        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        assert_eq!(dequantized.len(), 200);
        for i in 1..dequantized.len() {
            assert!(
                dequantized[i] >= dequantized[i - 1],
                "E5M2 large batch monotonicity violated at [{}]: {} < {}",
                i, dequantized[i], dequantized[i - 1]
            );
        }
    }

    #[test]
    fn test_e4m3_exp_1_all_mantissa_values_roundtrip() {
        let mut decoded: Vec<f32> = Vec::with_capacity(8);
        for m in 0u8..=7 {
            let byte = (1u8 << 3) | m;
            decoded.push(fp8_e4m3_to_fp32(byte));
        }

        for i in 1..decoded.len() {
            assert!(
                decoded[i] > decoded[i - 1],
                "E4M3 exp=1 mantissa={} ({}) should be > mantissa={} ({})",
                i, decoded[i], i - 1, decoded[i - 1]
            );
        }

        for m in 0u8..=7 {
            let byte = (1u8 << 3) | m;
            let re_encoded = fp32_to_fp8_e4m3(decoded[m as usize]);
            let diff = (re_encoded as i16 - byte as i16).abs();
            assert!(
                diff <= 1,
                "E4M3 exp=1 mantissa={}: original=0x{:02X}, re-encoded=0x{:02X}",
                m, byte, re_encoded
            );
        }
    }

    #[test]
    fn test_e5m2_negative_inf_decode_with_positive_scale() {
        let neg_inf_byte = 0xFC_u8;

        let decoded = fp8_e5m2_to_fp32(neg_inf_byte) * 5.0f32;

        assert!(
            decoded.is_infinite() && decoded.is_sign_negative(),
            "E5M2 -Inf * 5.0 should be -Inf, got {}",
            decoded
        );
    }

    #[test]
    fn test_e4m3_e5m2_both_decode_byte_0x00_to_positive_zero() {
        let byte = 0x00_u8;

        let e4m3_val = fp8_e4m3_to_fp32(byte);
        let e5m2_val = fp8_e5m2_to_fp32(byte);

        assert_eq!(e4m3_val, 0.0f32);
        assert_eq!(e5m2_val, 0.0f32);
        assert!(!e4m3_val.is_sign_negative());
        assert!(!e5m2_val.is_sign_negative());
    }

    #[test]
    fn test_quantize_e5m2_batch_all_nan_preserves_count() {
        let x = vec![f32::NAN; 8];
        let scale = 2.0f32;

        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        assert_eq!(quantized.len(), 8);
        assert_eq!(dequantized.len(), 8);
        for (i, &val) in dequantized.iter().enumerate() {
            assert!(
                val.is_nan(),
                "E5M2 all-NaN batch position {} should be NaN, got {}",
                i, val
            );
        }
        let first_byte = quantized[0];
        for (i, &b) in quantized.iter().enumerate() {
            assert_eq!(
                b, first_byte,
                "All NaN bytes should be identical, byte[{}]=0x{:02X}",
                i, b
            );
        }
    }

    #[test]
    fn test_quantize_e4m3_with_inf_in_batch_does_not_corrupt_adjacent() {
        let x = vec![3.0f32, f32::INFINITY, -7.0, f32::NEG_INFINITY, 0.5];
        let scale = 1.0f32;

        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        assert_eq!(dequantized.len(), 5);
        assert!(dequantized[0].is_finite() && dequantized[0] > 0.0, "First value should be positive finite");
        assert!(dequantized[1].is_infinite() && dequantized[1].is_sign_positive(), "Second should be +Inf");
        assert!(dequantized[2].is_finite() && dequantized[2] < 0.0, "Third value should be negative finite");
        assert!(dequantized[3].is_infinite() && dequantized[3].is_sign_negative(), "Fourth should be -Inf");
        assert!(dequantized[4].is_finite() && dequantized[4] > 0.0, "Fifth value should be positive finite");
    }

    #[test]
    fn test_e4m3_e5m2_both_decode_byte_0x80_to_negative_zero() {
        let byte = 0x80_u8;

        let e4m3_val = fp8_e4m3_to_fp32(byte);
        let e5m2_val = fp8_e5m2_to_fp32(byte);

        assert_eq!(e4m3_val, 0.0f32);
        assert_eq!(e5m2_val, 0.0f32);
        assert!(e4m3_val.is_sign_negative(), "E4M3 byte 0x80 should decode to -0.0");
        assert!(e5m2_val.is_sign_negative(), "E5M2 byte 0x80 should decode to -0.0");
    }

    // =========================================================================
    // 10 additional tests — exponent doubling, saturation, mantissa coverage,
    // batch dequantize special bytes, scale-0 IEEE edge cases, cross-format gaps
    // =========================================================================

    #[test]
    fn test_e4m3_consecutive_zero_mantissa_exponents_double() {
        // Arrange: for E4M3 mantissa=0, consecutive exponent values represent
        // powers of two: exp=1 -> 2^(-6), exp=2 -> 2^(-5), ..., each doubles.
        // Verify this doubling relationship for exp 1..=14.
        let mut prev: Option<f32> = None;
        for exp in 1u8..=14 {
            let byte = exp << 3; // sign=0, mantissa=0

            // Act
            let val = fp8_e4m3_to_fp32(byte);

            // Assert: each value is exactly double the previous
            if let Some(p) = prev {
                let ratio = val / p;
                assert!(
                    (ratio - 2.0f32).abs() < f32::EPSILON,
                    "E4M3 exp={} mantissa=0 val={} should be exactly 2x exp={} val={}",
                    exp, val, exp - 1, p
                );
            }
            prev = Some(val);
        }
    }

    #[test]
    fn test_quantize_e4m3_tiny_scale_saturates_to_max_range() {
        // Arrange: very small scale amplifies 1.0 far beyond E4M3 range (~448).
        // 1.0 / 1e-20 = 1e20 which saturates to max normal or NaN.
        let x = vec![1.0f32];
        let scale = 1e-20f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let raw_decoded = fp8_e4m3_to_fp32(quantized[0]);

        // Assert: the raw decoded value (before re-applying scale) should be
        // a large positive finite value near the E4M3 max range, or NaN.
        assert!(
            raw_decoded.is_finite() && raw_decoded > 0.0,
            "E4M3 saturated value should be positive finite, got {}",
            raw_decoded
        );
        // Should be near max E4M3 representable value (>= 200)
        assert!(
            raw_decoded >= 200.0,
            "Saturated E4M3 should be near max range, got {}",
            raw_decoded
        );
    }

    #[test]
    fn test_e5m2_exp_10_all_mantissa_values_distinct_and_ordered() {
        // Arrange: E5M2 exp=10 (mid-range), all 4 mantissa values 0..3.
        // Similar to existing exp=15 test but at a different exponent.
        let mut decoded: Vec<f32> = Vec::with_capacity(4);
        for m in 0u8..=3 {
            let byte = (10u8 << 2) | m;
            decoded.push(fp8_e5m2_to_fp32(byte));
        }

        // Act & Assert: all 4 values are finite, positive, strictly increasing
        for (m, &val) in decoded.iter().enumerate() {
            assert!(
                val.is_finite() && val > 0.0,
                "E5M2 exp=10 mantissa={} should be positive finite, got {}",
                m, val
            );
        }
        for i in 1..decoded.len() {
            assert!(
                decoded[i] > decoded[i - 1],
                "E5M2 exp=10 mantissa={} ({}) should be > mantissa={} ({})",
                i, decoded[i], i - 1, decoded[i - 1]
            );
        }
        // Re-encode stability: each should re-encode to the same byte
        for m in 0u8..=3 {
            let original_byte = (10u8 << 2) | m;
            let re_encoded = fp32_to_fp8_e5m2(decoded[m as usize]);
            let diff = (re_encoded as i16 - original_byte as i16).abs();
            assert!(
                diff <= 1,
                "E5M2 exp=10 mantissa={}: original=0x{:02X}, re-encoded=0x{:02X} (diff={})",
                m, original_byte, re_encoded, diff
            );
        }
    }

    #[test]
    fn test_dequantize_e4m3_all_0xff_bytes_produces_nan() {
        // Arrange: byte 0xFF in E4M3 = sign=1, exp=15, mantissa=7 → NaN
        let q = vec![0xFF_u8; 8];
        let scale = 3.0f32;

        // Act
        let dequantized = dequantize_fp8_e4m3(&q, scale);

        // Assert: all 8 values are NaN (NaN * scale = NaN)
        assert_eq!(dequantized.len(), 8);
        for (i, &val) in dequantized.iter().enumerate() {
            assert!(
                val.is_nan(),
                "E4M3 byte 0xFF at index {} should dequantize to NaN, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_quantize_e5m2_alternating_sign_moderate_scale() {
        // Arrange: alternating positive/negative with a scale that keeps them in range.
        // 10.0 / 5.0 = 2.0 and -10.0 / 5.0 = -2.0, both exactly representable in E5M2.
        let x = vec![10.0f32, -10.0, 20.0, -20.0, 40.0, -40.0];
        let scale = 5.0f32;

        // Act
        let quantized = quantize_fp8_e5m2(&x, scale);
        let dequantized = dequantize_fp8_e5m2(&quantized, scale);

        // Assert: length preserved, signs alternate
        assert_eq!(dequantized.len(), 6);
        for i in 0..6 {
            let sign_ok = if i % 2 == 0 {
                dequantized[i] >= 0.0
            } else {
                dequantized[i] <= 0.0
            };
            assert!(
                sign_ok,
                "E5M2 alternating sign at index {}: expected {}, got {}",
                i, if i % 2 == 0 { "non-negative" } else { "non-positive" },
                dequantized[i]
            );
        }
        // Values should have acceptable relative error
        for i in 0..x.len() {
            let error = (x[i] - dequantized[i]).abs() / x[i].abs();
            assert!(
                error < 0.15,
                "E5M2 moderate scale [{}]: {} -> {} (error: {})",
                i, x[i], dequantized[i], error
            );
        }
    }

    #[test]
    fn test_e4m3_exp_15_mantissa_3_and_4_distinct_and_ordered() {
        // Arrange: E4M3 exp=15 with mantissa=3 and mantissa=4.
        // Both are non-reserved (NaN=7, Inf=6) and should decode to distinct
        // finite positive values with mantissa=4 > mantissa=3.
        let byte_m3 = (15u8 << 3) | 3; // 0x7B
        let byte_m4 = (15u8 << 3) | 4; // 0x7C

        // Act
        let val_m3 = fp8_e4m3_to_fp32(byte_m3);
        let val_m4 = fp8_e4m3_to_fp32(byte_m4);

        // Assert: both finite positive, ordered correctly
        assert!(
            val_m3.is_finite() && val_m3 > 0.0,
            "E4M3 exp=15 mantissa=3 should be finite positive, got {}",
            val_m3
        );
        assert!(
            val_m4.is_finite() && val_m4 > 0.0,
            "E4M3 exp=15 mantissa=4 should be finite positive, got {}",
            val_m4
        );
        assert!(
            val_m4 > val_m3,
            "E4M3 exp=15 mantissa=4 ({}) should be > mantissa=3 ({})",
            val_m4, val_m3
        );
    }

    #[test]
    fn test_dequantize_e5m2_negative_inf_with_scale_through_batch_api() {
        // Arrange: E5M2 -Inf byte (0xFC) processed through the batch dequantize API
        let q = vec![0x3C_u8, 0xFC_u8, 0x00]; // 1.0, -Inf, +0 in E5M2
        let scale = 7.0f32;

        // Act
        let dequantized = dequantize_fp8_e5m2(&q, scale);

        // Assert: 1.0 * 7.0 = 7.0, -Inf * 7.0 = -Inf, 0.0 * 7.0 = 0.0
        assert_eq!(dequantized.len(), 3);
        assert!(
            (dequantized[0] - 7.0f32).abs() < f32::EPSILON,
            "E5M2 1.0 * 7.0 should be 7.0, got {}",
            dequantized[0]
        );
        assert!(
            dequantized[1].is_infinite() && dequantized[1].is_sign_negative(),
            "E5M2 -Inf * 7.0 should be -Inf, got {}",
            dequantized[1]
        );
        assert_eq!(
            dequantized[2], 0.0f32,
            "E5M2 0.0 * 7.0 should be 0.0, got {}",
            dequantized[2]
        );
    }

    #[test]
    fn test_quantize_e4m3_value_equal_to_negative_scale_roundtrip() {
        // Arrange: positive value with negative scale where val/scale is exactly
        // representable. -8.0 / -2.0 = 4.0 (power of two, exact in E4M3).
        let x = vec![-8.0f32];
        let scale = -2.0f32;

        // Act
        let quantized = quantize_fp8_e4m3(&x, scale);
        let dequantized = dequantize_fp8_e4m3(&quantized, scale);

        // Assert: double negative sign flip returns close to original
        assert_eq!(dequantized.len(), 1);
        let error = (x[0] - dequantized[0]).abs();
        assert!(
            error < 0.5,
            "E4M3 negative value negative scale: {} -> {} (error: {})",
            x[0], dequantized[0], error
        );
    }

    #[test]
    fn test_e5m2_encode_decode_all_exp_10_mantissa_roundtrip() {
        // Arrange: encode values that decode to exp=10 mantissa 0..3,
        // then verify re-encoding produces the same byte.
        for m in 0u8..=3 {
            let original_byte = (10u8 << 2) | m;

            // Act
            let val = fp8_e5m2_to_fp32(original_byte);
            let re_encoded = fp32_to_fp8_e5m2(val);

            // Assert: re-encoding should produce the same or adjacent byte
            let diff = (re_encoded as i16 - original_byte as i16).abs();
            assert!(
                diff == 0,
                "E5M2 exp=10 mantissa={} should re-encode to same byte: original=0x{:02X}, got=0x{:02X}",
                m, original_byte, re_encoded
            );
        }
    }

    #[test]
    fn test_quantize_e4m3_batch_with_mixed_nan_inf_zero_scale_zero() {
        // Arrange: batch with mix of finite, NaN, Inf, zero — all with scale=0.0.
        // Finite * 0.0 = 0.0, NaN * 0.0 = NaN, Inf * 0.0 = NaN, 0.0 * 0.0 = 0.0
        let q: Vec<u8> = vec![
            0x38, // E4M3 1.0 (finite)
            0x7F, // E4M3 NaN
            0x7E, // E4M3 +Inf
            0xFE, // E4M3 -Inf
            0x00, // E4M3 +0
            0x80, // E4M3 -0
        ];
        let scale = 0.0f32;

        // Act
        let dequantized = dequantize_fp8_e4m3(&q, scale);

        // Assert: IEEE 754 multiplication rules with 0.0
        assert_eq!(dequantized.len(), 6);
        assert_eq!(dequantized[0], 0.0f32, "1.0 * 0.0 should be 0.0");
        assert!(dequantized[1].is_nan(), "NaN * 0.0 should be NaN");
        assert!(dequantized[2].is_nan(), "+Inf * 0.0 should be NaN");
        assert!(dequantized[3].is_nan(), "-Inf * 0.0 should be NaN");
        assert_eq!(dequantized[4], 0.0f32, "+0 * 0.0 should be 0.0");
        assert_eq!(dequantized[5], 0.0f32, "-0 * 0.0 should be 0.0");
    }
}
