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
            .unwrap_or_else(|| f16::from_f32(1.0))
            .to_f32()
    }
}

pub fn dequantize_int8(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|v| *v as f32 * scale).collect()
}

pub fn dequantize_int8_with_zero(data: &[i8], scale: f32, zero: f32) -> Vec<f32> {
    data.iter()
        .map(|v| (*v as f32 - zero) * scale)
        .collect()
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
}
