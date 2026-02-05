//! Naming rules parser for GGUF and ONNX files.

use std::path::Path;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufQuantization {
    Q4_0,
    Q8_0,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
    F16,
    F32,
}

impl GgufQuantization {
    pub fn as_str(&self) -> &'static str {
        match self {
            GgufQuantization::Q4_0 => "Q4_0",
            GgufQuantization::Q8_0 => "Q8_0",
            GgufQuantization::Q4_K_M => "Q4_K_M",
            GgufQuantization::Q5_K_S => "Q5_K_S",
            GgufQuantization::Q5_K_M => "Q5_K_M",
            GgufQuantization::Q6_K => "Q6_K",
            GgufQuantization::F16 => "F16",
            GgufQuantization::F32 => "F32",
        }
    }

    pub fn preference_rank(&self) -> u8 {
        match self {
            GgufQuantization::F32 => 8,
            GgufQuantization::F16 => 7,
            GgufQuantization::Q8_0 => 6,
            GgufQuantization::Q6_K => 5,
            GgufQuantization::Q5_K_M => 4,
            GgufQuantization::Q5_K_S => 3,
            GgufQuantization::Q4_K_M => 2,
            GgufQuantization::Q4_0 => 1,
        }
    }

    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            GgufQuantization::Q4_0
                | GgufQuantization::Q8_0
                | GgufQuantization::F16
                | GgufQuantization::F32
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnnxPrecision {
    Fp32,
    Fp16,
    Int8,
    Uint8,
    Q4,
    Quantized,
}

impl OnnxPrecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            OnnxPrecision::Fp32 => "FP32",
            OnnxPrecision::Fp16 => "FP16",
            OnnxPrecision::Int8 => "INT8",
            OnnxPrecision::Uint8 => "UINT8",
            OnnxPrecision::Q4 => "Q4",
            OnnxPrecision::Quantized => "QUANTIZED",
        }
    }

    pub fn preference_rank(&self) -> u8 {
        match self {
            OnnxPrecision::Fp32 => 6,
            OnnxPrecision::Fp16 => 5,
            OnnxPrecision::Int8 => 4,
            OnnxPrecision::Uint8 => 3,
            OnnxPrecision::Q4 => 2,
            OnnxPrecision::Quantized => 1,
        }
    }
}

const GGUF_TOKENS: &[(&str, GgufQuantization)] = &[
    ("Q4_K_M", GgufQuantization::Q4_K_M),
    ("Q5_K_S", GgufQuantization::Q5_K_S),
    ("Q5_K_M", GgufQuantization::Q5_K_M),
    ("Q6_K", GgufQuantization::Q6_K),
    ("Q8_0", GgufQuantization::Q8_0),
    ("Q4_0", GgufQuantization::Q4_0),
    ("FP16", GgufQuantization::F16),
    ("F16", GgufQuantization::F16),
    ("FP32", GgufQuantization::F32),
    ("F32", GgufQuantization::F32),
];

const ONNX_TOKENS: &[(&str, OnnxPrecision)] = &[
    ("FP32", OnnxPrecision::Fp32),
    ("FLOAT32", OnnxPrecision::Fp32),
    ("F32", OnnxPrecision::Fp32),
    ("FP16", OnnxPrecision::Fp16),
    ("FLOAT16", OnnxPrecision::Fp16),
    ("F16", OnnxPrecision::Fp16),
    ("INT8", OnnxPrecision::Int8),
    ("UINT8", OnnxPrecision::Uint8),
    ("Q4", OnnxPrecision::Q4),
    ("QUANTIZED", OnnxPrecision::Quantized),
];

pub fn parse_gguf_quantization(name: &str) -> Option<GgufQuantization> {
    let stem = file_stem(name)?;
    let normalized = normalize_token(&stem);
    for (token, quant) in GGUF_TOKENS {
        if normalized.contains(token) {
            return Some(*quant);
        }
    }
    None
}

pub fn parse_onnx_precision(name: &str) -> Option<OnnxPrecision> {
    if !ends_with_ext(name, "onnx") {
        return None;
    }
    let stem = file_stem(name)?;
    let normalized = normalize_token(&stem);
    for (token, precision) in ONNX_TOKENS {
        if normalized.contains(token) {
            return Some(*precision);
        }
    }
    Some(OnnxPrecision::Fp32)
}

pub fn gguf_candidate_rank(name: &str) -> Option<(u8, u8)> {
    if !ends_with_ext(name, "gguf") {
        return None;
    }
    let quant = parse_gguf_quantization(name);
    let supported = quant.map(|q| q.is_supported() as u8).unwrap_or(0);
    let rank = quant.map(|q| q.preference_rank()).unwrap_or(0);
    Some((supported, rank))
}

pub fn onnx_candidate_rank(name: &str) -> Option<(u8, u8)> {
    if !ends_with_ext(name, "onnx") {
        return None;
    }
    let precision = parse_onnx_precision(name).unwrap_or(OnnxPrecision::Fp32);
    let in_onnx_dir = if is_in_onnx_dir(name) { 1 } else { 0 };
    Some((in_onnx_dir, precision.preference_rank()))
}

fn file_stem(name: &str) -> Option<String> {
    let path = Path::new(name);
    path.file_stem().map(|s| s.to_string_lossy().to_string())
}

fn normalize_token(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            '-' | '.' | ' ' => '_',
            _ => c,
        })
        .collect::<String>()
        .to_ascii_uppercase()
}

fn ends_with_ext(name: &str, ext: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

fn is_in_onnx_dir(name: &str) -> bool {
    let normalized = name.replace('\\', "/");
    normalized.starts_with("onnx/") || normalized.contains("/onnx/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gguf_quantization_variants() {
        assert_eq!(
            parse_gguf_quantization("SmolLM-135M-Instruct-Q4_0.gguf"),
            Some(GgufQuantization::Q4_0)
        );
        assert_eq!(
            parse_gguf_quantization("SmolLM-135M-Instruct-f16.gguf"),
            Some(GgufQuantization::F16)
        );
        assert_eq!(
            parse_gguf_quantization("model.Q4_K_M.gguf"),
            Some(GgufQuantization::Q4_K_M)
        );
    }

    #[test]
    fn parse_onnx_precision_variants() {
        assert_eq!(
            parse_onnx_precision("onnx/model_fp16.onnx"),
            Some(OnnxPrecision::Fp16)
        );
        assert_eq!(
            parse_onnx_precision("onnx/model.onnx"),
            Some(OnnxPrecision::Fp32)
        );
        assert_eq!(
            parse_onnx_precision("model_int8.onnx"),
            Some(OnnxPrecision::Int8)
        );
    }

    #[test]
    fn candidate_rankers_respect_dirs() {
        let rank = onnx_candidate_rank("onnx/model_fp16.onnx").unwrap();
        assert_eq!(rank.0, 1);
        let rank = onnx_candidate_rank("model_fp16.onnx").unwrap();
        assert_eq!(rank.0, 0);
    }
}
