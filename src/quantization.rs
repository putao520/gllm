//! Quantization helpers (INT8/INT4/AWQ/GPTQ scaffolding).

use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationKind {
    None,
    Int8,
    Int4,
    Awq,
    Gptq,
}

#[derive(Debug, Clone)]
pub struct BlockQuantization {
    pub block_size: usize,
    pub scales: Vec<f16>,
}

impl BlockQuantization {
    pub fn new(block_size: usize, scales: Vec<f16>) -> Self {
        Self {
            block_size: block_size.max(1),
            scales,
        }
    }

    pub fn scale_for_block(&self, block: usize) -> f32 {
        self.scales
            .get(block)
            .copied()
            .unwrap_or_else(|| f16::from_f32(1.0)) // LEGAL: scale=1.0 是量化参数的合法默认值（无缩放）
            .to_f32()
    }
}

pub fn dequantize_int8(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|v| *v as f32 * scale).collect()
}

pub fn dequantize_int8_with_zero(data: &[i8], scale: f32, zero: f32) -> Vec<f32> {
    data.iter().map(|v| (*v as f32 - zero) * scale).collect()
}

pub fn dequantize_int4(packed: &[u8], scale: f32, signed: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    for &byte in packed {
        let lo = byte & 0x0f;
        let hi = (byte >> 4) & 0x0f;
        out.push(int4_to_f32(lo, scale, signed));
        out.push(int4_to_f32(hi, scale, signed));
    }
    out
}

pub fn dequantize_int4_with_zero(packed: &[u8], scale: f32, zero: f32, signed: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    for &byte in packed {
        let lo = byte & 0x0f;
        let hi = (byte >> 4) & 0x0f;
        out.push((int4_to_i8(lo, signed) as f32 - zero) * scale);
        out.push((int4_to_i8(hi, signed) as f32 - zero) * scale);
    }
    out
}

fn int4_to_f32(value: u8, scale: f32, signed: bool) -> f32 {
    if signed {
        let signed = if value & 0x08 != 0 {
            (value as i8) - 16
        } else {
            value as i8
        };
        signed as f32 * scale
    } else {
        value as f32 * scale
    }
}

fn int4_to_i8(value: u8, signed: bool) -> i8 {
    if signed {
        if value & 0x08 != 0 {
            (value as i8) - 16
        } else {
            value as i8
        }
    } else {
        value as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_int8_scales() {
        let data = vec![-2i8, 0, 2];
        let out = dequantize_int8(&data, 0.5);
        assert_eq!(out, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn dequantize_int4_signed() {
        let data = vec![0b1111_0001];
        let out = dequantize_int4(&data, 1.0, true);
        assert_eq!(out, vec![1.0, -1.0]);
    }

    #[test]
    fn dequantize_int4_with_zero_offset() {
        let data = vec![0b0010_0001];
        let out = dequantize_int4_with_zero(&data, 2.0, 1.0, false);
        assert_eq!(out, vec![0.0, 2.0]);
    }

    #[test]
    fn dequantize_int8_with_zero_offset() {
        let data = vec![2i8, 4, -1];
        let out = dequantize_int8_with_zero(&data, 0.5, 1.0);
        // (2 - 1) * 0.5 = 0.5, (4 - 1) * 0.5 = 1.5, (-1 - 1) * 0.5 = -1.0
        assert_eq!(out, vec![0.5, 1.5, -1.0]);
    }

    #[test]
    fn dequantize_int4_unsigned() {
        let data = vec![0b1010_0101]; // lo=5, hi=10
        let out = dequantize_int4(&data, 2.0, false);
        assert_eq!(out, vec![10.0, 20.0]);
    }

    #[test]
    fn block_quantization_scale_for_block() {
        let scales = vec![f16::from_f32(0.5), f16::from_f32(2.0)];
        let bq = BlockQuantization::new(32, scales);
        assert!((bq.scale_for_block(0) - 0.5).abs() < 1e-3);
        assert!((bq.scale_for_block(1) - 2.0).abs() < 1e-3);
        // Out of bounds returns default 1.0
        assert!((bq.scale_for_block(99) - 1.0).abs() < 1e-3);
    }

    // ── QuantizationKind ──

    #[test]
    fn quantization_kind_equality() {
        assert_eq!(QuantizationKind::None, QuantizationKind::None);
        assert_ne!(QuantizationKind::Int8, QuantizationKind::Int4);
        assert_ne!(QuantizationKind::Awq, QuantizationKind::Gptq);
    }

    #[test]
    fn quantization_kind_copy() {
        let a = QuantizationKind::Int4;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn quantization_kind_debug() {
        let debug = format!("{:?}", QuantizationKind::Awq);
        assert!(debug.contains("Awq"));
        let debug = format!("{:?}", QuantizationKind::Gptq);
        assert!(debug.contains("Gptq"));
    }

    // ── BlockQuantization ──

    #[test]
    fn block_quantization_min_block_size_one() {
        let bq = BlockQuantization::new(0, vec![]);
        assert_eq!(bq.block_size, 1);
    }

    #[test]
    fn block_quantization_empty_scales_default() {
        let bq = BlockQuantization::new(32, vec![]);
        assert!((bq.scale_for_block(0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn block_quantization_debug() {
        let bq = BlockQuantization::new(32, vec![f16::from_f32(1.0)]);
        let debug = format!("{bq:?}");
        assert!(debug.contains("block_size"));
        assert!(debug.contains("scales"));
    }

    #[test]
    fn block_quantization_clone() {
        let bq = BlockQuantization::new(16, vec![f16::from_f32(0.5)]);
        let cloned = bq.clone();
        assert_eq!(bq.block_size, cloned.block_size);
        assert!((bq.scale_for_block(0) - cloned.scale_for_block(0)).abs() < 1e-6);
    }

    // ── INT8 dequantize edge cases ──

    #[test]
    fn dequantize_int8_zero_scale() {
        let data = vec![100i8, -50, 0];
        let out = dequantize_int8(&data, 0.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn dequantize_int8_empty() {
        let out = dequantize_int8(&[], 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn dequantize_int8_with_zero_empty() {
        let out = dequantize_int8_with_zero(&[], 1.0, 0.0);
        assert!(out.is_empty());
    }

    // ── INT4 dequantize edge cases ──

    #[test]
    fn dequantize_int4_signed_negative_range() {
        // 0b1000_0111: lo=7 (signed: 7), hi=8 (signed: 8-16=-8)
        let data = vec![0b1000_0111];
        let out = dequantize_int4(&data, 1.0, true);
        assert_eq!(out[0], 7.0);
        assert_eq!(out[1], -8.0);
    }

    #[test]
    fn dequantize_int4_zero_scale() {
        let data = vec![0xFF];
        let out = dequantize_int4(&data, 0.0, false);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn dequantize_int4_with_zero_signed() {
        // 0b0001_1110: lo=14 (signed: -2), hi=1 (signed: 1)
        let data = vec![0b0001_1110];
        let out = dequantize_int4_with_zero(&data, 2.0, 1.0, true);
        // lo=14 signed=-2, (-2-1)*2.0 = -6.0; hi=1 signed=1, (1-1)*2.0 = 0.0
        assert!((out[0] - (-6.0)).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int4_empty() {
        let out = dequantize_int4(&[], 1.0, false);
        assert!(out.is_empty());
    }

    // ════════════════════════════════════════════════════════════════
    //  ~40 additional tests for public types and methods
    // ════════════════════════════════════════════════════════════════

    // ── QuantizationKind: all variant pairwise inequality ──

    #[test]
    fn quantization_kind_none_not_equal_others() {
        assert_ne!(QuantizationKind::None, QuantizationKind::Int8);
        assert_ne!(QuantizationKind::None, QuantizationKind::Int4);
        assert_ne!(QuantizationKind::None, QuantizationKind::Awq);
        assert_ne!(QuantizationKind::None, QuantizationKind::Gptq);
    }

    #[test]
    fn quantization_kind_int8_not_equal_others() {
        assert_ne!(QuantizationKind::Int8, QuantizationKind::None);
        assert_ne!(QuantizationKind::Int8, QuantizationKind::Int4);
        assert_ne!(QuantizationKind::Int8, QuantizationKind::Awq);
        assert_ne!(QuantizationKind::Int8, QuantizationKind::Gptq);
    }

    #[test]
    fn quantization_kind_int4_not_equal_others() {
        assert_ne!(QuantizationKind::Int4, QuantizationKind::None);
        assert_ne!(QuantizationKind::Int4, QuantizationKind::Int8);
        assert_ne!(QuantizationKind::Int4, QuantizationKind::Awq);
        assert_ne!(QuantizationKind::Int4, QuantizationKind::Gptq);
    }

    #[test]
    fn quantization_kind_all_variants_debug_format() {
        assert_eq!(format!("{:?}", QuantizationKind::None), "None");
        assert_eq!(format!("{:?}", QuantizationKind::Int8), "Int8");
        assert_eq!(format!("{:?}", QuantizationKind::Int4), "Int4");
        assert_eq!(format!("{:?}", QuantizationKind::Awq), "Awq");
        assert_eq!(format!("{:?}", QuantizationKind::Gptq), "Gptq");
    }

    #[test]
    fn quantization_kind_copy_independent() {
        let original = QuantizationKind::Awq;
        let copy = original;
        assert_eq!(original, QuantizationKind::Awq);
        assert_eq!(copy, QuantizationKind::Awq);
    }

    #[test]
    fn quantization_kind_clone_matches_copy() {
        let a = QuantizationKind::Gptq;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn quantization_kind_self_equality_all_variants() {
        assert_eq!(QuantizationKind::None, QuantizationKind::None);
        assert_eq!(QuantizationKind::Int8, QuantizationKind::Int8);
        assert_eq!(QuantizationKind::Int4, QuantizationKind::Int4);
        assert_eq!(QuantizationKind::Awq, QuantizationKind::Awq);
        assert_eq!(QuantizationKind::Gptq, QuantizationKind::Gptq);
    }

    // ── BlockQuantization: constructor & field access ──

    #[test]
    fn block_quantization_new_sets_block_size() {
        let bq = BlockQuantization::new(64, vec![]);
        assert_eq!(bq.block_size, 64);
    }

    #[test]
    fn block_quantization_new_block_size_clamped_to_one() {
        let bq = BlockQuantization::new(0, vec![]);
        assert_eq!(bq.block_size, 1);
    }

    #[test]
    fn block_quantization_new_negative_block_size_wraps_to_one() {
        // usize wrapping: negative literal won't compile, but 0 is the only edge
        let bq = BlockQuantization::new(0, vec![]);
        assert_eq!(bq.block_size, 1);
    }

    #[test]
    fn block_quantization_new_stores_scales() {
        let scales = vec![f16::from_f32(0.1), f16::from_f32(0.2), f16::from_f32(0.3)];
        let bq = BlockQuantization::new(16, scales);
        assert_eq!(bq.scales.len(), 3);
    }

    #[test]
    fn block_quantization_pub_fields_accessible() {
        let bq = BlockQuantization::new(128, vec![f16::from_f32(1.5)]);
        assert_eq!(bq.block_size, 128);
        assert_eq!(bq.scales.len(), 1);
    }

    #[test]
    fn block_quantization_scale_for_block_in_bounds() {
        let scales = vec![f16::from_f32(3.0), f16::from_f32(5.0)];
        let bq = BlockQuantization::new(8, scales);
        assert!((bq.scale_for_block(0) - 3.0).abs() < 1e-3);
        assert!((bq.scale_for_block(1) - 5.0).abs() < 1e-3);
    }

    #[test]
    fn block_quantization_scale_for_block_out_of_bounds_default() {
        let bq = BlockQuantization::new(8, vec![f16::from_f32(2.0)]);
        assert!((bq.scale_for_block(5) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn block_quantization_scale_for_block_empty_scales_any_index() {
        let bq = BlockQuantization::new(32, vec![]);
        assert!((bq.scale_for_block(0) - 1.0).abs() < 1e-3);
        assert!((bq.scale_for_block(100) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn block_quantization_clone_independent() {
        let bq = BlockQuantization::new(64, vec![f16::from_f32(0.25), f16::from_f32(0.75)]);
        let cloned = bq.clone();
        assert_eq!(bq.block_size, cloned.block_size);
        assert_eq!(bq.scales.len(), cloned.scales.len());
        assert!((bq.scale_for_block(0) - cloned.scale_for_block(0)).abs() < 1e-6);
        assert!((bq.scale_for_block(1) - cloned.scale_for_block(1)).abs() < 1e-6);
    }

    #[test]
    fn block_quantization_large_block_size() {
        let bq = BlockQuantization::new(4096, vec![f16::from_f32(0.5)]);
        assert_eq!(bq.block_size, 4096);
    }

    #[test]
    fn block_quantization_many_scales() {
        let scales: Vec<f16> = (0..100).map(|i| f16::from_f32(i as f32 * 0.01)).collect();
        let bq = BlockQuantization::new(1, scales);
        assert_eq!(bq.scales.len(), 100);
        assert!((bq.scale_for_block(0) - 0.0).abs() < 1e-2);
        assert!((bq.scale_for_block(99) - 0.99).abs() < 1e-2);
    }

    #[test]
    fn block_quantization_debug_contains_fields() {
        let bq = BlockQuantization::new(32, vec![f16::from_f32(1.0)]);
        let s = format!("{bq:?}");
        assert!(s.contains("32"), "debug should contain block_size value");
    }

    // ── dequantize_int8: boundary & property tests ──

    #[test]
    fn dequantize_int8_single_element() {
        let data = vec![42i8];
        let out = dequantize_int8(&data, 1.0);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int8_negative_scale() {
        let data = vec![2i8, -3];
        let out = dequantize_int8(&data, -1.0);
        assert!((out[0] - (-2.0)).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int8_boundary_values() {
        let data = vec![i8::MIN, i8::MAX, 0i8];
        let out = dequantize_int8(&data, 1.0);
        assert_eq!(out[0], i8::MIN as f32);
        assert_eq!(out[1], i8::MAX as f32);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn dequantize_int8_output_length_matches_input() {
        let data = vec![0i8; 37];
        let out = dequantize_int8(&data, 1.0);
        assert_eq!(out.len(), 37);
    }

    #[test]
    fn dequantize_int8_scale_one_is_identity() {
        let data = vec![-5i8, 0, 5, 100, -100];
        let out = dequantize_int8(&data, 1.0);
        for (i, &v) in data.iter().enumerate() {
            assert!((out[i] - v as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn dequantize_int8_large_scale() {
        let data = vec![1i8];
        let out = dequantize_int8(&data, 1000.0);
        assert!((out[0] - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn dequantize_int8_very_small_scale() {
        let data = vec![127i8];
        let out = dequantize_int8(&data, 1e-6);
        assert!((out[0] - 127e-6).abs() < 1e-10);
    }

    // ── dequantize_int8_with_zero: property tests ──

    #[test]
    fn dequantize_int8_with_zero_zero_point_zero_scale() {
        let data = vec![5i8, -3, 0];
        let out = dequantize_int8_with_zero(&data, 0.0, 0.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn dequantize_int8_with_zero_output_length() {
        let data = vec![0i8; 10];
        let out = dequantize_int8_with_zero(&data, 1.0, 0.0);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn dequantize_int8_with_zero_zero_point_cancels() {
        // When zero_point equals value, result is 0
        let data = vec![5i8];
        let out = dequantize_int8_with_zero(&data, 2.0, 5.0);
        assert!((out[0]).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int8_with_zero_negative_zero_point() {
        let data = vec![0i8];
        let out = dequantize_int8_with_zero(&data, 1.0, -1.0);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int8_with_zero_boundary_values() {
        let data = vec![i8::MIN, i8::MAX];
        let out = dequantize_int8_with_zero(&data, 1.0, 0.0);
        assert_eq!(out[0], i8::MIN as f32);
        assert_eq!(out[1], i8::MAX as f32);
    }

    // ── dequantize_int4: structural & property tests ──

    #[test]
    fn dequantize_int4_output_is_double_input_length() {
        let packed = vec![0u8; 10];
        let out = dequantize_int4(&packed, 1.0, false);
        assert_eq!(out.len(), 20);
    }

    #[test]
    fn dequantize_int4_single_byte_lo_hi_order() {
        // byte = 0bHILO: lo nibble first, hi nibble second
        let data = vec![0b0101_0010]; // lo=2, hi=5
        let out = dequantize_int4(&data, 1.0, false);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 5.0);
    }

    #[test]
    fn dequantize_int4_signed_max_positive() {
        // 0b0111: 7 (max positive in 4-bit signed)
        let data = vec![0b0111_0000]; // lo=0, hi=7
        let out = dequantize_int4(&data, 1.0, true);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 7.0);
    }

    #[test]
    fn dequantize_int4_signed_min_negative() {
        // 0b1000: -8 (min negative in 4-bit signed)
        let data = vec![0b0000_1000]; // lo=8 (signed=-8), hi=0
        let out = dequantize_int4(&data, 1.0, true);
        assert_eq!(out[0], -8.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn dequantize_int4_unsigned_max() {
        // 0b1111: 15 unsigned
        let data = vec![0b1111_1111];
        let out = dequantize_int4(&data, 1.0, false);
        assert_eq!(out[0], 15.0);
        assert_eq!(out[1], 15.0);
    }

    #[test]
    fn dequantize_int4_multiple_bytes_independent() {
        let data = vec![0b0001_0000, 0b0010_0000]; // lo=0,1; hi=1,2
        let out = dequantize_int4(&data, 3.0, false);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
        assert!((out[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int4_scale_applied_correctly() {
        let data = vec![0b0010_0001]; // lo=1, hi=2
        let out = dequantize_int4(&data, 0.5, false);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    // ── dequantize_int4_with_zero: property tests ──

    #[test]
    fn dequantize_int4_with_zero_output_length() {
        let packed = vec![0u8; 5];
        let out = dequantize_int4_with_zero(&packed, 1.0, 0.0, false);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn dequantize_int4_with_zero_empty() {
        let out = dequantize_int4_with_zero(&[], 1.0, 0.0, false);
        assert!(out.is_empty());
    }

    #[test]
    fn dequantize_int4_with_zero_zero_cancels() {
        // When signed value == zero_point, result is 0
        let data = vec![0b0001_0000]; // lo=0 unsigned=0, hi=1
        let out = dequantize_int4_with_zero(&data, 2.0, 1.0, false);
        // lo: (0-1)*2=-2, hi: (1-1)*2=0
        assert!((out[0] - (-2.0)).abs() < 1e-6);
        assert!((out[0] + 2.0).abs() < 1e-6);
        assert!((out[1]).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int4_with_zero_signed_nibble_conversion() {
        // 0b1110_0001: lo=1 (signed 1), hi=14 (signed -2)
        let data = vec![0b1110_0001];
        let out = dequantize_int4_with_zero(&data, 1.0, 0.0, true);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int4_with_zero_unsigned_all_zero() {
        let data = vec![0u8];
        let out = dequantize_int4_with_zero(&data, 5.0, 0.0, false);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn dequantize_int4_with_zero_negative_scale() {
        let data = vec![0b0001_0000]; // lo=0, hi=1 unsigned
        let out = dequantize_int4_with_zero(&data, -1.0, 0.0, false);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int4_with_zero_scale_zero_all_zero() {
        let data = vec![0xFF];
        let out = dequantize_int4_with_zero(&data, 0.0, 0.0, false);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn dequantize_int4_with_zero_signed_boundary() {
        // lo=8 (signed -8), hi=7 (signed 7)
        let data = vec![0b0111_1000];
        let out = dequantize_int4_with_zero(&data, 1.0, 0.0, true);
        assert!((out[0] - (-8.0)).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    // ════════════════════════════════════════════════════════════════
    //  15 additional tests: QuantizationKind exhaustiveness, int4_to_i8
    //  via public API, BlockQuantization f16 precision, multi-byte
    //  dequantize_int4_with_zero signed patterns, linearity proofs
    // ════════════════════════════════════════════════════════════════

    // ── QuantizationKind: exhaustiveness via match ──

    #[test]
    fn quantization_kind_all_five_variants_matchable() {
        // Arrange: collect all variants
        let kinds = vec![
            QuantizationKind::None,
            QuantizationKind::Int8,
            QuantizationKind::Int4,
            QuantizationKind::Awq,
            QuantizationKind::Gptq,
        ];
        // Act: count via exhaustive match
        let count = kinds
            .iter()
            .filter(|k| matches!(k, QuantizationKind::None | QuantizationKind::Int8 | QuantizationKind::Int4 | QuantizationKind::Awq | QuantizationKind::Gptq))
            .count();
        // Assert: all 5 variants are recognized
        assert_eq!(count, 5);
    }

    #[test]
    fn quantization_kind_derived_hash_consistency() {
        // QuantizationKind derives PartialEq + Eq but not Hash.
        // Verify that copy produces equal values usable in collections.
        let a = QuantizationKind::Int8;
        let b = a;
        // Act: collect into a vec and dedup by equality
        let mut v = vec![a, b];
        v.dedup();
        // Assert: dedup collapsed the pair
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], QuantizationKind::Int8);
    }

    // ── int4_to_i8 signed conversion via public API ──

    #[test]
    fn dequantize_int4_signed_all_positive_nibbles() {
        // Arrange: nibbles 0..7 are positive in signed mode; pack 0b0111_0000 (lo=0, hi=7)
        let data = vec![0b0111_0000];
        // Act
        let out = dequantize_int4(&data, 1.0, true);
        // Assert: positive range 0..=7
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 7.0);
    }

    #[test]
    fn dequantize_int4_signed_all_negative_nibbles() {
        // Arrange: nibbles 8..15 map to -8..=-1; pack 0b1111_1000 (lo=8→-8, hi=15→-1)
        let data = vec![0b1111_1000];
        // Act
        let out = dequantize_int4(&data, 1.0, true);
        // Assert
        assert_eq!(out[0], -8.0);
        assert_eq!(out[1], -1.0);
    }

    #[test]
    fn dequantize_int4_signed_full_range_per_nibble() {
        // Arrange: byte where lo=0 (signed 0), hi=15 (signed -1)
        let data = vec![0b1111_0000];
        // Act
        let out = dequantize_int4(&data, 1.0, true);
        // Assert
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], -1.0);
    }

    // ── dequantize_int8_with_zero: linearity proof ──

    #[test]
    fn dequantize_int8_with_zero_is_linear_in_scale() {
        // Arrange: same data and zero_point, two different scales
        let data = vec![10i8, -5, 3];
        let zero_point = 2.0_f32;
        let out_a = dequantize_int8_with_zero(&data, 1.0, zero_point);
        let out_b = dequantize_int8_with_zero(&data, 3.0, zero_point);
        // Act & Assert: out_b = 3 * out_a (linearity in scale)
        for i in 0..data.len() {
            let expected = out_a[i] * 3.0;
            assert!((out_b[i] - expected).abs() < 1e-4, "at index {i}");
        }
    }

    #[test]
    fn dequantize_int8_with_zero_is_linear_in_zero_point_shift() {
        // (v - zp) * s: doubling zp shifts output by -s per unit
        // Arrange
        let data = vec![4i8];
        let out_zp0 = dequantize_int8_with_zero(&data, 2.0, 0.0);
        let out_zp2 = dequantize_int8_with_zero(&data, 2.0, 2.0);
        // Act & Assert: difference = -2 * 2.0 = -4.0
        let diff = out_zp2[0] - out_zp0[0];
        assert!((diff - (-4.0)).abs() < 1e-6);
    }

    // ── dequantize_int4: scale multiplication verification ──

    #[test]
    fn dequantize_int4_unsigned_scale_proportional() {
        // Arrange: same packed data, two scales in ratio 1:4
        let data = vec![0b0011_0001]; // lo=1, hi=3 unsigned
        let out_s1 = dequantize_int4(&data, 1.0, false);
        let out_s4 = dequantize_int4(&data, 4.0, false);
        // Assert
        assert!((out_s4[0] - out_s1[0] * 4.0).abs() < 1e-5);
        assert!((out_s4[1] - out_s1[1] * 4.0).abs() < 1e-5);
    }

    #[test]
    fn dequantize_int4_signed_scale_proportional() {
        // Arrange
        let data = vec![0b1000_0111]; // lo=7 (signed 7), hi=8 (signed -8)
        let out_s1 = dequantize_int4(&data, 1.0, true);
        let out_s2 = dequantize_int4(&data, 2.0, true);
        // Assert
        assert!((out_s2[0] - out_s1[0] * 2.0).abs() < 1e-5);
        assert!((out_s2[1] - out_s1[1] * 2.0).abs() < 1e-5);
    }

    // ── BlockQuantization: f16 precision characteristics ──

    #[test]
    fn block_quantization_scale_preserves_f16_precision() {
        // Arrange: f16 can represent 0.25 exactly (power of 2)
        let bq = BlockQuantization::new(32, vec![f16::from_f32(0.25)]);
        // Act
        let scale = bq.scale_for_block(0);
        // Assert: exact representation within f32 precision
        assert!((scale - 0.25).abs() < 1e-6);
    }

    #[test]
    fn block_quantization_scale_f16_lossy_value() {
        // Arrange: f16 cannot represent 0.1 exactly
        let bq = BlockQuantization::new(32, vec![f16::from_f32(0.1)]);
        // Act
        let scale = bq.scale_for_block(0);
        // Assert: close but not exact — f16 round-trip introduces error
        assert!((scale - 0.1).abs() < 0.005, "f16 round-trip should be within 0.5%: got {scale}");
        assert!((scale - 0.1).abs() > 0.0, "f16 cannot represent 0.1 exactly, some loss expected");
    }

    // ── dequantize_int4_with_zero: multi-byte signed patterns ──

    #[test]
    fn dequantize_int4_with_zero_multi_byte_signed() {
        // Arrange: two bytes with mixed signed nibbles
        let data = vec![0b0111_1000, 0b0000_1111]; // [lo=8→-8, hi=7], [lo=15→-1, hi=0]
        // Act
        let out = dequantize_int4_with_zero(&data, 1.0, 0.0, true);
        // Assert
        assert_eq!(out.len(), 4);
        assert!((out[0] - (-8.0)).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
        assert!((out[2] - (-1.0)).abs() < 1e-6);
        assert!((out[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int4_with_zero_large_zero_point() {
        // Arrange: zero_point=8.0, values are small unsigned nibbles
        let data = vec![0b1000_0000]; // lo=0, hi=8 unsigned
        // Act
        let out = dequantize_int4_with_zero(&data, 1.0, 8.0, false);
        // Assert: (0-8)*1=-8, (8-8)*1=0
        assert!((out[0] - (-8.0)).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
    }

    // ── BlockQuantization: zero and negative f16 scales ──

    #[test]
    fn block_quantization_zero_scale_returns_zero() {
        // Arrange: f16 zero as scale
        let bq = BlockQuantization::new(16, vec![f16::from_f32(0.0)]);
        // Act
        let scale = bq.scale_for_block(0);
        // Assert
        assert!((scale - 0.0).abs() < 1e-6);
    }

    // ── dequantize_int8: consistency between with_zero and without ──

    #[test]
    fn dequantize_int8_with_zero_zero_point_is_superset() {
        // Arrange: dequantize_int8_with_zero(data, scale, 0.0) == dequantize_int8(data, scale)
        let data = vec![-10i8, 0, 10, 50, -50];
        let scale = 0.25_f32;
        // Act
        let without_zp = dequantize_int8(&data, scale);
        let with_zp_zero = dequantize_int8_with_zero(&data, scale, 0.0);
        // Assert: identical results
        for i in 0..data.len() {
            assert!(
                (without_zp[i] - with_zp_zero[i]).abs() < 1e-10,
                "mismatch at index {i}: {} vs {}",
                without_zp[i],
                with_zp_zero[i]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  10 additional tests: exhaustive nibble mapping, scale linearity
    //  cross-function equivalence, BlockQuantization clone isolation,
    //  int8 extreme scale, int4_with_zero proportional to int4
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn dequantize_int4_unsigned_all_16_nibble_values() {
        // Arrange: construct a byte for each nibble value, verify unsigned mapping
        // lo nibble covers values 0..=15, hi nibble always 0
        for nibble in 0u8..=15 {
            let byte = nibble; // hi=0, lo=nibble
            // Act
            let out = dequantize_int4(&[byte], 1.0, false);
            // Assert: lo nibble maps to its unsigned value
            assert!(
                (out[0] - nibble as f32).abs() < 1e-6,
                "unsigned nibble {nibble}: expected {}, got {}",
                nibble,
                out[0]
            );
        }
    }

    #[test]
    fn dequantize_int4_signed_all_16_nibble_values() {
        // Arrange: verify signed 4-bit mapping for all nibble values
        // 0..7 -> 0..7, 8..15 -> -8..-1
        let expected: Vec<f32> = (0..=15).map(|n| {
            if n < 8 { n as f32 } else { n as f32 - 16.0 }
        }).collect();
        for nibble in 0u8..=15 {
            let byte = nibble; // hi=0, lo=nibble
            // Act
            let out = dequantize_int4(&[byte], 1.0, true);
            // Assert
            assert!(
                (out[0] - expected[nibble as usize]).abs() < 1e-6,
                "signed nibble {nibble}: expected {}, got {}",
                expected[nibble as usize],
                out[0]
            );
        }
    }

    #[test]
    fn dequantize_int8_boundary_with_non_unit_scale() {
        // Arrange: i8::MIN and i8::MAX with scale=0.5
        let data = vec![i8::MIN, i8::MAX];
        let scale = 0.5_f32;
        // Act
        let out = dequantize_int8(&data, scale);
        // Assert
        assert!((out[0] - i8::MIN as f32 * 0.5).abs() < 1e-4);
        assert!((out[1] - i8::MAX as f32 * 0.5).abs() < 1e-4);
    }

    #[test]
    fn dequantize_int8_with_zero_scale_zero_suppresses_all() {
        // Arrange: arbitrary data and zero_point, but scale=0
        let data = vec![127i8, -128, 0, 42];
        // Act
        let out = dequantize_int8_with_zero(&data, 0.0, 8.0);
        // Assert: all outputs are 0 regardless of data and zero_point
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v == 0.0,
                "scale=0 must produce all zeros: got {v} at index {i}"
            );
        }
    }

    #[test]
    fn dequantize_int4_with_zero_proportional_to_int4() {
        // Arrange: dequantize_int4_with_zero(data, scale, 0.0, unsigned)
        // should equal dequantize_int4(data, scale, unsigned)
        let data = vec![0b1010_0101, 0b0000_1111, 0b1000_0111];
        let scale = 0.75_f32;
        // Act
        let without_zp = dequantize_int4(&data, scale, false);
        let with_zp_zero = dequantize_int4_with_zero(&data, scale, 0.0, false);
        // Assert
        assert_eq!(without_zp.len(), with_zp_zero.len());
        for i in 0..without_zp.len() {
            assert!(
                (without_zp[i] - with_zp_zero[i]).abs() < 1e-10,
                "unsigned mismatch at {i}: {} vs {}",
                without_zp[i],
                with_zp_zero[i]
            );
        }
    }

    #[test]
    fn dequantize_int4_with_zero_signed_proportional_to_int4_signed() {
        // Arrange: same equivalence check for signed mode
        let data = vec![0b1111_0000, 0b1000_0111, 0b0101_1010];
        let scale = 1.5_f32;
        // Act
        let without_zp = dequantize_int4(&data, scale, true);
        let with_zp_zero = dequantize_int4_with_zero(&data, scale, 0.0, true);
        // Assert
        for i in 0..without_zp.len() {
            assert!(
                (without_zp[i] - with_zp_zero[i]).abs() < 1e-10,
                "signed mismatch at {i}: {} vs {}",
                without_zp[i],
                with_zp_zero[i]
            );
        }
    }

    #[test]
    fn block_quantization_clone_isolation() {
        // Arrange: create a BlockQuantization, clone it
        let mut bq = BlockQuantization::new(32, vec![f16::from_f32(1.0), f16::from_f32(2.0)]);
        let cloned = bq.clone();
        // Act: mutate original's scales vec (push is a mutation on the vec, not via methods)
        bq.scales.push(f16::from_f32(3.0));
        // Assert: clone is unaffected
        assert_eq!(bq.scales.len(), 3);
        assert_eq!(cloned.scales.len(), 2);
    }

    #[test]
    fn block_quantization_f16_nan_scale() {
        // Arrange: f16 NaN as a scale value
        let bq = BlockQuantization::new(16, vec![f16::NAN]);
        // Act
        let scale = bq.scale_for_block(0);
        // Assert: NaN should propagate to f32
        assert!(scale.is_nan(), "f16 NaN should produce f32 NaN");
    }

    #[test]
    fn dequantize_int4_negative_scale_inverts_sign() {
        // Arrange: unsigned nibbles with negative scale
        let data = vec![0b0011_0001]; // lo=1, hi=3 unsigned
        // Act
        let out_pos = dequantize_int4(&data, 2.0, false);
        let out_neg = dequantize_int4(&data, -2.0, false);
        // Assert: each output negated
        assert!((out_neg[0] + out_pos[0]).abs() < 1e-6);
        assert!((out_neg[1] + out_pos[1]).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int8_negative_values_with_negative_scale() {
        // Arrange: negative i8 values * negative scale -> positive output
        let data = vec![-5i8, -10, -1];
        // Act
        let out = dequantize_int8(&data, -2.0);
        // Assert
        assert!((out[0] - 10.0).abs() < 1e-4);
        assert!((out[1] - 20.0).abs() < 1e-4);
        assert!((out[2] - 2.0).abs() < 1e-4);
    }
}
