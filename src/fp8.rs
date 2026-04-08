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
        return ((sign as u8) << 7); // Denormal → 0
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
        return ((sign as u8) << 7); // Denormal → 0
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
}
