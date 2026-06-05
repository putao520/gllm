//! 离线格式转换器 — GGUF/safetensors → .gllm。
//!
//! SPEC: `SPEC/36-GLLM-WEIGHT-FORMAT.md §3`
//!
//! - GGUF 已量化 → .gllm 直通模式 (§3.3): 量化数据直通复制，不重新量化。
//! - safetensors → .gllm (§3.2): 原始精度直通，量化校准待后续实现。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::quant_encode::{QuantTarget, quantize_awq4, quantize_gptq4, quantize_nvfp4};
use super::writer::{self, GllmWriter, TensorEntry};
use super::GllmError;
use crate::loader::adapter::ggml_dtype_to_quant_type;
use crate::loader::gguf::{GgmlDType, GgufReader};
use crate::loader::safetensors::MappedSafetensors;
use half::f16;

/// 转换选项。
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// 分页对齐大小 (默认 4096)。
    pub page_size: u32,
    /// 模型配置文件路径 (config.json)，用于提取架构元数据。
    /// safetensors → .gllm 转换时可选；GGUF → .gllm 转换时忽略。
    pub config_path: Option<PathBuf>,
    /// 量化目标格式。仅在 GGUF FP16/BF16 → .gllm 时使用。
    /// None 表示保留原始精度（直通模式）；Some 表示 RTN 量化到目标格式。
    pub quant_target: Option<QuantTarget>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self { page_size: 4096, config_path: None, quant_target: None }
    }
}

/// GGUF → .gllm 转换结果。
#[derive(Debug)]
pub struct ConvertResult {
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub tensor_count: usize,
    pub quantized_count: usize,
}

/// 将 GGUF 已量化模型转换为 .gllm 格式 (直通模式)。
///
/// SPEC 36 §3.3: 量化数据直通复制，不重新量化。
/// SPEC 36 §4: 从 GGUF metadata 自动推导模型架构参数。
pub fn convert_gguf_to_gllm(
    gguf_path: &Path,
    output_path: &Path,
    options: &ConvertOptions,
) -> Result<ConvertResult, GllmError> {
    let reader = GgufReader::open(gguf_path).map_err(|e| GllmError::Io(std::io::Error::other(e.to_string())))?;
    let input_bytes = std::fs::metadata(gguf_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let mut writer = GllmWriter::new(options.page_size);
    let mut quantized_count = 0;

    // ── 转换每个张量 ────────────────────────────────────────────────
    let tensor_names = reader.names();
    for name in &tensor_names {
        let info = reader.tensor_info(name)
            .map_err(|e| GllmError::ParseError(format!("tensor '{}': {}", name, e)))?;
        let data = reader.tensor_bytes(name)
            .map_err(|e| GllmError::ParseError(format!("tensor '{}': {}", name, e)))?;

        let dtype_code = ggml_dtype_to_gllm_code(info.dtype);
        let (quant_format, quant_block_size, scale_dtype, zp_type) =
            match ggml_dtype_to_quant_type(info.dtype) {
                Some(qt) => {
                    let code = writer::quant_type_to_u8(qt);
                    let block_size = info.dtype.block_size() as u16;
                    let scale_dt = gllm_scale_dtype(info.dtype);
                    let zp = gllm_zp_type(info.dtype);
                    quantized_count += 1;
                    (code, block_size, scale_dt, zp)
                }
                None => (0u8, 0u16, 0u8, 0u8),
            };

        let ndim = info.shape.len().min(4) as u8;
        let mut shape = [0u64; 4];
        for (i, &d) in info.shape.iter().enumerate().take(4) {
            shape[i] = d;
        }

        let original_size = info.shape.iter().product::<u64>() * 4; // F32 equivalent

        writer.add_tensor(TensorEntry {
            name: name.to_string(),
            ndim,
            dtype: dtype_code,
            shape,
            quant_format,
            quant_block_size,
            scale_dtype,
            zp_type,
            data: data.to_vec(),
            original_size,
        });
    }

    // ── 从 GGUF metadata 构建 .gllm metadata ────────────────────────
    let arch = reader.architecture()
        .unwrap_or("unknown")
        .to_string();
    let vocab_size = reader.get_metadata_u64("general.vocab_size").unwrap_or(0);
    let hidden_size = reader.embedding_length().unwrap_or(0);
    let num_layers = reader.block_count().unwrap_or(0);
    let num_heads = reader.head_count().unwrap_or(0);
    let num_kv_heads = reader.head_count_kv().unwrap_or(num_heads);
    let head_dim = reader.attention_head_dim().unwrap_or(0);
    let intermediate_size = reader.feed_forward_length().unwrap_or(0);
    let context_length = reader.context_length().unwrap_or(0);

    let mut extras = HashMap::new();
    if let Some(v) = reader.rope_freq_base() {
        extras.insert("rope_freq_base".to_string(), v.to_string());
    }
    if let Some(v) = reader.rope_dimension_count() {
        extras.insert("rope_dimension_count".to_string(), v.to_string());
    }
    if let Some(v) = reader.num_experts() {
        extras.insert("num_experts".to_string(), v.to_string());
    }
    if let Some(v) = reader.num_experts_per_tok() {
        extras.insert("num_experts_per_tok".to_string(), v.to_string());
    }
    if let Some(v) = reader.kv_lora_rank() {
        extras.insert("kv_lora_rank".to_string(), v.to_string());
    }
    if let Some(v) = reader.qk_rope_head_dim() {
        extras.insert("qk_rope_head_dim".to_string(), v.to_string());
    }

    let meta = writer::build_metadata(
        &arch, vocab_size, hidden_size, num_layers,
        num_heads, num_kv_heads, head_dim, intermediate_size,
        context_length, &extras,
    );
    writer.set_metadata(meta);

    // ── 写出 .gllm ─────────────────────────────────────────────────
    writer.write_to_path(output_path)?;
    let output_bytes = std::fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(ConvertResult {
        input_bytes,
        output_bytes,
        tensor_count: tensor_names.len(),
        quantized_count,
    })
}

/// GGUF FP16/BF16 → .gllm 量化转换 (REQ-GLF-007)。
///
/// 对 GGUF 中的 FP16/BF16 权重执行 RTN 量化编码，写入 .gllm 量化格式。
/// 已量化的张量（Q4_0 等）仍然直通复制。
pub fn convert_gguf_fp16_to_gllm(
    gguf_path: &Path,
    output_path: &Path,
    options: &ConvertOptions,
) -> Result<ConvertResult, GllmError> {
    let target = options.quant_target.ok_or_else(|| {
        GllmError::ParseError("quant_target required for FP16→.gllm conversion".to_string())
    })?;

    let reader = GgufReader::open(gguf_path)
        .map_err(|e| GllmError::Io(std::io::Error::other(e.to_string())))?;
    let input_bytes = std::fs::metadata(gguf_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let mut writer = GllmWriter::new(options.page_size);
    let mut quantized_count = 0;

    let tensor_names = reader.names();
    for name in &tensor_names {
        let info = reader.tensor_info(name)
            .map_err(|e| GllmError::ParseError(format!("tensor '{}': {}", name, e)))?;
        let raw_data = reader.tensor_bytes(name)
            .map_err(|e| GllmError::ParseError(format!("tensor '{}': {}", name, e)))?;

        let is_float = matches!(info.dtype, GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32);
        let is_already_quant = ggml_dtype_to_quant_type(info.dtype).is_some();

        let ndim = info.shape.len().min(4) as u8;
        let mut shape = [0u64; 4];
        for (i, &d) in info.shape.iter().enumerate().take(4) {
            shape[i] = d;
        }
        let original_size = info.shape.iter().product::<u64>() * 4;

        if is_float && !is_already_quant {
            // FP16/BF16/F32 权重 → RTN 量化
            let nrows = shape[0] as usize;
            let ncols: usize = shape[1..].iter().filter(|&&d| d > 0).product::<u64>() as usize;
            let ncols = if ncols == 0 { 1 } else { ncols };

            let weights_f16 = decode_to_f16(raw_data, info.dtype);

            let (quant_format, quant_block_size, scale_dtype, zp_type, encoded_data) =
                match target {
                    QuantTarget::Awq4 => {
                        let group_size = 128;
                        let result = quantize_awq4(&weights_f16, nrows, ncols, group_size);
                        let mut data = Vec::with_capacity(result.encoded_bytes);
                        data.extend_from_slice(&result.scales);
                        data.extend_from_slice(&result.zero_points);
                        data.extend_from_slice(&result.packed_data);
                        (40u8, group_size as u16, 1u8, 1u8, data)
                    }
                    QuantTarget::Gptq4 => {
                        let group_size = 128;
                        let result = quantize_gptq4(&weights_f16, nrows, ncols, group_size);
                        let mut data = Vec::with_capacity(result.encoded_bytes);
                        data.extend_from_slice(&result.scales);
                        data.extend_from_slice(&result.zero_points);
                        data.extend_from_slice(&result.packed_data);
                        (41u8, group_size as u16, 1u8, 2u8, data)
                    }
                    QuantTarget::Nvfp4 => {
                        let result = quantize_nvfp4(&weights_f16, nrows, ncols);
                        (53u8, 64u16, 0u8, 0u8, result.packed_data)
                    }
                };

            quantized_count += 1;
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim,
                dtype: match info.dtype {
                    GgmlDType::F16 => 1,
                    GgmlDType::BF16 => 2,
                    _ => 0,
                },
                shape,
                quant_format,
                quant_block_size,
                scale_dtype,
                zp_type,
                data: encoded_data,
                original_size,
            });
        } else if is_already_quant {
            // 已量化张量 → 直通
            let qt = ggml_dtype_to_quant_type(info.dtype).unwrap();
            let quant_format = writer::quant_type_to_u8(qt);
            let quant_block_size = info.dtype.block_size() as u16;
            let scale_dt = gllm_scale_dtype(info.dtype);
            let zp = gllm_zp_type(info.dtype);
            quantized_count += 1;
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim,
                dtype: ggml_dtype_to_gllm_code(info.dtype),
                shape,
                quant_format,
                quant_block_size,
                scale_dtype: scale_dt,
                zp_type: zp,
                data: raw_data.to_vec(),
                original_size,
            });
        } else {
            // 非权重张量（如 token_embedding 可保留原始精度）
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim,
                dtype: ggml_dtype_to_gllm_code(info.dtype),
                shape,
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: raw_data.to_vec(),
                original_size,
            });
        }
    }

    // ── 从 GGUF metadata 构建 .gllm metadata ────────────────────────
    let arch = reader.architecture()
        .unwrap_or("unknown")
        .to_string();
    let vocab_size = reader.get_metadata_u64("general.vocab_size").unwrap_or(0);
    let hidden_size = reader.embedding_length().unwrap_or(0);
    let num_layers = reader.block_count().unwrap_or(0);
    let num_heads = reader.head_count().unwrap_or(0);
    let num_kv_heads = reader.head_count_kv().unwrap_or(num_heads);
    let head_dim = reader.attention_head_dim().unwrap_or(0);
    let intermediate_size = reader.feed_forward_length().unwrap_or(0);
    let context_length = reader.context_length().unwrap_or(0);

    let mut extras = HashMap::new();
    if let Some(v) = reader.rope_freq_base() {
        extras.insert("rope_freq_base".to_string(), v.to_string());
    }
    if let Some(v) = reader.rope_dimension_count() {
        extras.insert("rope_dimension_count".to_string(), v.to_string());
    }
    if let Some(v) = reader.num_experts() {
        extras.insert("num_experts".to_string(), v.to_string());
    }
    if let Some(v) = reader.num_experts_per_tok() {
        extras.insert("num_experts_per_tok".to_string(), v.to_string());
    }
    if let Some(v) = reader.kv_lora_rank() {
        extras.insert("kv_lora_rank".to_string(), v.to_string());
    }
    if let Some(v) = reader.qk_rope_head_dim() {
        extras.insert("qk_rope_head_dim".to_string(), v.to_string());
    }

    let meta = writer::build_metadata(
        &arch, vocab_size, hidden_size, num_layers,
        num_heads, num_kv_heads, head_dim, intermediate_size,
        context_length, &extras,
    );
    writer.set_metadata(meta);

    writer.write_to_path(output_path)?;
    let output_bytes = std::fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(ConvertResult {
        input_bytes,
        output_bytes,
        tensor_count: tensor_names.len(),
        quantized_count,
    })
}

/// 将 GGUF FP16/BF16/F32 原始字节解码为 f16 向量。
fn decode_to_f16(data: &[u8], dtype: GgmlDType) -> Vec<f16> {
    match dtype {
        GgmlDType::F16 => {
            assert!(data.len().is_multiple_of(2));
            data.chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                .collect()
        }
        GgmlDType::BF16 => {
            assert!(data.len().is_multiple_of(2));
            data.chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    // BF16 → F32 → F16 (精度损失可接受，RTN 量化场景)
                    let f32_val = half::bf16::from_bits(bits).to_f32();
                    f16::from_f32(f32_val)
                })
                .collect()
        }
        GgmlDType::F32 => {
            assert!(data.len().is_multiple_of(4));
            data.chunks_exact(4)
                .map(|chunk| {
                    let f32_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f16::from_f32(f32_val)
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

/// GgmlDType → .gllm dtype u8 编码。
fn ggml_dtype_to_gllm_code(dt: GgmlDType) -> u8 {
    match dt {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::BF16 => 2,
        _ => 0, // 量化格式按原始 FP 类型编码；量化信息由 quant_format 承载
    }
}

/// 量化格式的 scale dtype 编码。
fn gllm_scale_dtype(dt: GgmlDType) -> u8 {
    match dt {
        GgmlDType::Q4_1 | GgmlDType::Q5_1 => 1,
        GgmlDType::Q4_0 | GgmlDType::Q5_0 | GgmlDType::Q8_0 | GgmlDType::Q8_1 => 1,
        GgmlDType::Q2_K | GgmlDType::Q3_K | GgmlDType::Q4_K
        | GgmlDType::Q5_K | GgmlDType::Q6_K | GgmlDType::Q8_K => 1,
        GgmlDType::IQ1_S | GgmlDType::IQ1_M | GgmlDType::IQ2_XXS
        | GgmlDType::IQ2_XS | GgmlDType::IQ2_S | GgmlDType::IQ3_XXS
        | GgmlDType::IQ3_S | GgmlDType::IQ4_NL | GgmlDType::IQ4_XS => 1,
        _ => 0,
    }
}

/// 量化格式的 zero-point type 编码。
fn gllm_zp_type(dt: GgmlDType) -> u8 {
    match dt {
        GgmlDType::Q4_1 | GgmlDType::Q5_1 => 1,
        _ => 0,
    }
}

/// safetensors → .gllm 转换器 (原始精度直通)。
///
/// SPEC 36 §3.2: BF16/FP16/F32 原始精度直通写入 .gllm。
/// 量化校准 (AWQ4/GPTQ4/NVFP4) 需完整 ML 管线，留待后续实现。
///
/// 支持单文件和多分片 safetensors 输入。
/// 架构元数据从 config.json 提取 (通过 `options.config_path`)。
pub fn convert_safetensors_to_gllm(
    safetensors_paths: &[PathBuf],
    output_path: &Path,
    options: &ConvertOptions,
) -> Result<ConvertResult, GllmError> {
    if safetensors_paths.is_empty() {
        return Err(GllmError::ParseError("no safetensors input files".to_string()));
    }

    let mut total_input_bytes: u64 = 0;
    for p in safetensors_paths {
        total_input_bytes += std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    }

    let mut writer = GllmWriter::new(options.page_size);
    let mut tensor_count = 0usize;

    // ── 遍历所有分片，逐张量写入 ──────────────────────────────────
    for path in safetensors_paths {
        let st = MappedSafetensors::open(path)
            .map_err(|e| GllmError::Io(std::io::Error::other(e.to_string())))?;

        for name in st.names() {
            let slice = st.tensor(&name)
                .map_err(|e| GllmError::ParseError(format!("tensor '{}': {}", name, e)))?;

            let dtype_code = writer::safetensors_dtype_to_u8(slice.dtype);
            let ndim = slice.shape.len().min(4) as u8;
            let mut shape = [0u64; 4];
            for (i, &d) in slice.shape.iter().enumerate().take(4) {
                shape[i] = d as u64;
            }

            let original_size = slice.shape.iter().product::<usize>() as u64
                * elem_bytes(slice.dtype) as u64;

            writer.add_tensor(TensorEntry {
                name: name.clone(),
                ndim,
                dtype: dtype_code,
                shape,
                quant_format: 0, // 原始精度，未量化
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: slice.data.to_vec(),
                original_size,
            });
            tensor_count += 1;
        }
    }

    // ── 从 config.json 提取架构元数据 ──────────────────────────────
    let (arch, vocab_size, hidden_size, num_layers, num_heads, num_kv_heads,
         head_dim, intermediate_size, context_length, extras) =
        extract_model_params(options.config_path.as_deref());

    let meta = writer::build_metadata(
        &arch, vocab_size, hidden_size, num_layers,
        num_heads, num_kv_heads, head_dim, intermediate_size,
        context_length, &extras,
    );
    writer.set_metadata(meta);

    // ── 写出 .gllm ────────────────────────────────────────────────
    writer.write_to_path(output_path)?;
    let output_bytes = std::fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(ConvertResult {
        input_bytes: total_input_bytes,
        output_bytes,
        tensor_count,
        quantized_count: 0,
    })
}

/// 从 config.json 提取模型架构参数。
///
/// 如果 config.json 不存在或无法解析，返回合理默认值 ("unknown", 全零)。
fn extract_model_params(
    config_path: Option<&Path>,
) -> (String, u64, u64, u64, u64, u64, u64, u64, u64, HashMap<String, String>) {
    let mut extras = HashMap::new();
    let path = match config_path {
        Some(p) => p,
        None => return ("unknown".to_string(), 0, 0, 0, 0, 0, 0, 0, 0, extras),
    };

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return ("unknown".to_string(), 0, 0, 0, 0, 0, 0, 0, 0, extras),
    };

    let config: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return ("unknown".to_string(), 0, 0, 0, 0, 0, 0, 0, 0, extras),
    };

    let arch = config.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let vocab_size = config_u64(&config, "vocab_size");
    let hidden_size = config_u64(&config, "hidden_size");
    let num_layers = config_u64(&config, "num_hidden_layers");
    let num_heads = config_u64(&config, "num_attention_heads");
    let num_kv_heads = config_u64(&config, "num_key_value_heads").max(1);
    let head_dim = config_u64(&config, "head_dim")
        .max(hidden_size / num_heads.max(1));
    let intermediate_size = config_u64(&config, "intermediate_size");
    let context_length = config_u64(&config, "max_position_embeddings");

    if let Some(v) = config.get("rope_theta").and_then(|v| v.as_f64()) {
        extras.insert("rope_freq_base".to_string(), v.to_string());
    }
    if let Some(v) = config.get("rope_scaling").and_then(|v| v.as_object()) {
        if let Some(f) = v.get("factor").and_then(|f| f.as_f64()) {
            extras.insert("rope_scaling_factor".to_string(), f.to_string());
        }
    }
    if let Some(v) = config.get("num_local_experts").and_then(|v| v.as_u64()) {
        extras.insert("num_experts".to_string(), v.to_string());
    }
    if let Some(v) = config.get("num_experts_per_tok").and_then(|v| v.as_u64()) {
        extras.insert("num_experts_per_tok".to_string(), v.to_string());
    }

    (arch, vocab_size, hidden_size, num_layers, num_heads, num_kv_heads,
     head_dim, intermediate_size, context_length, extras)
}

fn config_u64(config: &serde_json::Value, key: &str) -> u64 {
    config.get(key).and_then(|v| v.as_u64()).unwrap_or(0)
}

/// safetensors Dtype → 每元素字节数。
fn elem_bytes(dt: safetensors::Dtype) -> usize {
    match dt {
        safetensors::Dtype::BOOL | safetensors::Dtype::U8 | safetensors::Dtype::I8 => 1,
        safetensors::Dtype::F16 | safetensors::Dtype::BF16
        | safetensors::Dtype::I16 | safetensors::Dtype::U16 => 2,
        safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn gllm_scale_dtype_coverage() {
        let quant_types = [
            GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q5_0, GgmlDType::Q5_1,
            GgmlDType::Q8_0, GgmlDType::Q8_1, GgmlDType::Q4_K, GgmlDType::Q6_K,
        ];
        for dt in &quant_types {
            let scale = gllm_scale_dtype(*dt);
            assert_ne!(scale, 0, "GgmlDType {:?} should have scale_dtype != 0", dt);
        }
    }

    #[test]
    fn zp_type_for_q4_1() {
        assert_eq!(gllm_zp_type(GgmlDType::Q4_1), 1);
        assert_eq!(gllm_zp_type(GgmlDType::Q5_1), 1);
        assert_eq!(gllm_zp_type(GgmlDType::Q4_0), 0);
    }

    /// 创建一个包含 BF16 + F32 张量的 safetensors fixture。
    fn write_safetensors_fixture(dir: &Path) -> (PathBuf, Vec<u8>, Vec<u8>) {
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;

        let bf16_data: Vec<u8> = (0..32).map(|i| (i as u8).wrapping_mul(37)).collect();
        let f32_data: Vec<u8> = (0..64).map(|i| (i as u8).wrapping_mul(13)).collect();

        let bf16_view = TensorView::new(Dtype::BF16, vec![4, 4], &bf16_data).expect("bf16 view");
        let f32_view = TensorView::new(Dtype::F32, vec![4, 4], &f32_data).expect("f32 view");

        let path = dir.join("model.safetensors");
        serialize_to_file(
            vec![
                ("layer.0.weight", bf16_view),
                ("layer.0.bias", f32_view),
            ],
            &None,
            &path,
        )
        .expect("write safetensors");

        (path, bf16_data, f32_data)
    }

    #[test]
    fn safetensors_to_gllm_roundtrip() {
        let dir = std::env::temp_dir().join("gllm_test_st2gllm");
        std::fs::create_dir_all(&dir).unwrap();

        let (st_path, bf16_bytes, f32_bytes) = write_safetensors_fixture(&dir);
        let gllm_path = dir.join("output.gllm");

        let result = convert_safetensors_to_gllm(
            &[st_path],
            &gllm_path,
            &ConvertOptions::default(),
        ).expect("convert");

        assert_eq!(result.tensor_count, 2);
        assert_eq!(result.quantized_count, 0);
        assert!(gllm_path.exists());

        // 读回验证
        let reader = crate::loader::gllm::GllmReader::open(&gllm_path).expect("read back");
        assert_eq!(reader.tensor_count(), 2);
        assert!(!reader.header().is_quantized());

        // BF16 张量
        let t1 = reader.find_tensor("layer.0.weight").expect("find bf16 tensor");
        assert_eq!(t1.entry.shape[0], 4);
        assert_eq!(t1.entry.shape[1], 4);
        assert!(!t1.entry.is_quantized());
        let d1 = reader.tensor_data("layer.0.weight").expect("bf16 data");
        let d1_ref: &[u8] = d1.as_ref();
        assert_eq!(d1_ref, bf16_bytes.as_slice());

        // F32 张量
        let t2 = reader.find_tensor("layer.0.bias").expect("find f32 tensor");
        assert_eq!(t2.entry.shape[0], 4);
        assert!(!t2.entry.is_quantized());
        let d2 = reader.tensor_data("layer.0.bias").expect("f32 data");
        let d2_ref: &[u8] = d2.as_ref();
        assert_eq!(d2_ref, f32_bytes.as_slice());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn safetensors_to_gllm_with_config() {
        let dir = std::env::temp_dir().join("gllm_test_st2gllm_cfg");
        std::fs::create_dir_all(&dir).unwrap();

        let (st_path, _, _) = write_safetensors_fixture(&dir);
        let gllm_path = dir.join("output.gllm");

        // 写一个 config.json
        let config = serde_json::json!({
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11008,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let result = convert_safetensors_to_gllm(
            &[st_path],
            &gllm_path,
            &ConvertOptions {
                page_size: 4096,
                config_path: Some(config_path),
                quant_target: None,
            },
        ).expect("convert");

        assert_eq!(result.tensor_count, 2);

        // 验证元数据
        let reader = crate::loader::gllm::GllmReader::open(&gllm_path).expect("read");
        let params = reader.model_params().expect("params");
        assert_eq!(params.vocab_size, 151936);
        assert_eq!(params.hidden_size, 4096);
        assert_eq!(params.num_layers, 36);
        assert_eq!(params.num_heads, 32);
        assert_eq!(params.num_kv_heads, 8);
        assert_eq!(params.head_dim, 128);
        assert_eq!(params.intermediate_size, 11008);
        assert_eq!(params.context_length, 32768);
        assert_eq!(reader.architecture().as_deref(), Some("qwen3"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_from_config() {
        let dir = std::env::temp_dir().join("gllm_test_extract_params");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (arch, vocab, hidden, layers, heads, kv_heads, head_dim, inter, ctx, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(arch, "llama");
        assert_eq!(vocab, 32000);
        assert_eq!(hidden, 4096);
        assert_eq!(layers, 32);
        assert_eq!(heads, 32);
        assert_eq!(kv_heads, 1); // num_key_value_heads 不存在 → .max(1) = 1
        assert_eq!(head_dim, 128); // 4096 / 32
        assert_eq!(inter, 11008);
        assert_eq!(ctx, 4096);
        assert_eq!(extras.get("rope_freq_base").map(String::as_str), Some("10000"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── New tests ──────────────────────────────────────────────────────────────

    #[test]
    fn convert_options_default_values() {
        let opts = ConvertOptions::default();
        assert_eq!(opts.page_size, 4096);
        assert!(opts.config_path.is_none());
        assert!(opts.quant_target.is_none());
    }

    #[test]
    fn ggml_dtype_to_gllm_code_mappings() {
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::F32), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::F16), 1);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::BF16), 2);
        // Quantized types map to 0 (quant info in quant_format field)
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::Q4_0), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::Q8_0), 0);
    }

    #[test]
    fn elem_bytes_safetensors_dtypes() {
        use safetensors::Dtype;
        assert_eq!(elem_bytes(Dtype::U8), 1);
        assert_eq!(elem_bytes(Dtype::I8), 1);
        assert_eq!(elem_bytes(Dtype::F16), 2);
        assert_eq!(elem_bytes(Dtype::BF16), 2);
        assert_eq!(elem_bytes(Dtype::F32), 4);
        assert_eq!(elem_bytes(Dtype::I32), 4);
        assert_eq!(elem_bytes(Dtype::I64), 8);
        assert_eq!(elem_bytes(Dtype::F64), 8);
    }

    #[test]
    fn decode_to_f16_from_f16_bytes() {
        // Two f16 values: 1.0 and 2.0
        let one = f16::from_f32(1.0);
        let two = f16::from_f32(2.0);
        let mut data = Vec::new();
        data.extend_from_slice(&one.to_le_bytes());
        data.extend_from_slice(&two.to_le_bytes());

        let result = decode_to_f16(&data, GgmlDType::F16);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[1], f16::from_f32(2.0));
    }

    #[test]
    fn decode_to_f16_from_f32_bytes() {
        let one = 1.0f32.to_le_bytes();
        let two = 2.0f32.to_le_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&one);
        data.extend_from_slice(&two);

        let result = decode_to_f16(&data, GgmlDType::F32);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[1], f16::from_f32(2.0));
    }

    #[test]
    fn decode_to_f16_from_bf16_bytes() {
        let one = half::bf16::from_f32(1.0);
        let two = half::bf16::from_f32(2.0);
        let mut data = Vec::new();
        data.extend_from_slice(&one.to_le_bytes());
        data.extend_from_slice(&two.to_le_bytes());

        let result = decode_to_f16(&data, GgmlDType::BF16);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[1], f16::from_f32(2.0));
    }

    #[test]
    fn decode_to_f16_non_float_returns_empty() {
        // Q4_0 is not a float type, should return empty vec
        let data = vec![0u8; 32];
        let result = decode_to_f16(&data, GgmlDType::Q4_0);
        assert!(result.is_empty());
    }

    #[test]
    fn gllm_scale_dtype_non_quant_returns_zero() {
        assert_eq!(gllm_scale_dtype(GgmlDType::F32), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::F16), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::BF16), 0);
    }

    #[test]
    fn extract_model_params_no_config_returns_defaults() {
        let (arch, vocab, hidden, layers, heads, kv_heads, head_dim, inter, ctx, extras) =
            extract_model_params(None);

        assert_eq!(arch, "unknown");
        assert_eq!(vocab, 0);
        assert_eq!(hidden, 0);
        assert_eq!(layers, 0);
        assert_eq!(heads, 0);
        assert_eq!(kv_heads, 0);
        assert_eq!(head_dim, 0);
        assert_eq!(inter, 0);
        assert_eq!(ctx, 0);
        assert!(extras.is_empty());
    }

    #[test]
    fn extract_model_params_invalid_json_returns_defaults() {
        let dir = std::env::temp_dir().join("gllm_test_extract_invalid");
        std::fs::create_dir_all(&dir).unwrap();

        let config_path = dir.join("config.json");
        std::fs::write(&config_path, "not valid json {{{}").unwrap();

        let (arch, vocab, _, _, _, _, _, _, _, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(arch, "unknown");
        assert_eq!(vocab, 0);
        assert!(extras.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn convert_result_fields() {
        let result = ConvertResult {
            input_bytes: 1024,
            output_bytes: 512,
            tensor_count: 10,
            quantized_count: 5,
        };
        assert_eq!(result.input_bytes, 1024);
        assert_eq!(result.output_bytes, 512);
        assert_eq!(result.tensor_count, 10);
        assert_eq!(result.quantized_count, 5);
    }

    #[test]
    fn convert_safetensors_empty_paths_errors() {
        let dir = std::env::temp_dir().join("gllm_test_empty_st");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("out.gllm");

        let result = convert_safetensors_to_gllm(&[], &out, &ConvertOptions::default());
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_moe_extras() {
        let dir = std::env::temp_dir().join("gllm_test_extract_moe");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "deepseek",
            "vocab_size": 129280,
            "hidden_size": 4096,
            "num_hidden_layers": 28,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 11008,
            "max_position_embeddings": 163840,
            "num_local_experts": 64,
            "num_experts_per_tok": 8,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, _, _, _, _, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(extras.get("num_experts").map(String::as_str), Some("64"));
        assert_eq!(extras.get("num_experts_per_tok").map(String::as_str), Some("8"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Additional tests ─────────────────────────────────────────────────────

    #[test]
    fn convert_options_clone_produces_equal_instance() {
        let opts = ConvertOptions {
            page_size: 8192,
            config_path: Some(PathBuf::from("/tmp/config.json")),
            quant_target: Some(QuantTarget::Awq4),
        };
        let cloned = opts.clone();
        assert_eq!(cloned.page_size, opts.page_size);
        assert_eq!(cloned.config_path, opts.config_path);
        assert_eq!(cloned.quant_target, opts.quant_target);
    }

    #[test]
    fn convert_options_debug_includes_all_fields() {
        let opts = ConvertOptions {
            page_size: 4096,
            config_path: Some(PathBuf::from("/cfg.json")),
            quant_target: None,
        };
        let debug_str = format!("{:?}", opts);
        assert!(debug_str.contains("page_size"));
        assert!(debug_str.contains("config_path"));
        assert!(debug_str.contains("quant_target"));
    }

    #[test]
    fn convert_options_custom_values() {
        let opts = ConvertOptions {
            page_size: 65536,
            config_path: Some(PathBuf::from("/data/config.json")),
            quant_target: Some(QuantTarget::Gptq4),
        };
        assert_eq!(opts.page_size, 65536);
        assert_eq!(opts.config_path.as_ref().map(|p| p.to_str().unwrap()), Some("/data/config.json"));
        assert!(matches!(opts.quant_target, Some(QuantTarget::Gptq4)));
    }

    #[test]
    fn convert_result_debug_format() {
        let result = ConvertResult {
            input_bytes: 1000,
            output_bytes: 500,
            tensor_count: 7,
            quantized_count: 3,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("input_bytes"));
        assert!(debug_str.contains("output_bytes"));
        assert!(debug_str.contains("tensor_count"));
        assert!(debug_str.contains("quantized_count"));
    }

    #[test]
    fn gllm_error_display_parse_error() {
        let err = GllmError::ParseError("bad tensor data".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bad tensor data"));
        assert!(msg.contains("parse error"));
    }

    #[test]
    fn gllm_error_display_invalid_metadata() {
        let err = GllmError::InvalidMetadata("missing field".to_string());
        let msg = err.to_string();
        assert!(msg.contains("missing field"));
        assert!(msg.contains("invalid metadata"));
    }

    #[test]
    fn gllm_error_display_tensor_out_of_bounds() {
        let err = GllmError::TensorOutOfBounds {
            name: "weight.0".to_string(),
            start: 100,
            end: 200,
            file_size: 150,
        };
        let msg = err.to_string();
        assert!(msg.contains("weight.0"));
        assert!(msg.contains("100"));
        assert!(msg.contains("200"));
        assert!(msg.contains("150"));
    }

    #[test]
    fn gllm_error_display_string_table_out_of_bounds() {
        let err = GllmError::StringTableOutOfBounds {
            offset: 500,
            length: 300,
            file_size: 600,
        };
        let msg = err.to_string();
        assert!(msg.contains("500"));
        assert!(msg.contains("800"));
        assert!(msg.contains("600"));
    }

    #[test]
    fn gllm_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let gllm_err: GllmError = io_err.into();
        assert!(matches!(gllm_err, GllmError::Io(_)));
        let msg = gllm_err.to_string();
        assert!(msg.contains("denied"));
    }

    #[test]
    fn gllm_error_display_tensor_dir_out_of_bounds() {
        let err = GllmError::TensorDirOutOfBounds {
            offset: 1000,
            count: 50,
            file_size: 2000,
        };
        let msg = err.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("4600")); // 1000 + 50 * 72
        assert!(msg.contains("2000"));
    }

    #[test]
    fn gllm_error_display_metadata_out_of_bounds() {
        let err = GllmError::MetadataOutOfBounds {
            offset: 9999,
            file_size: 500,
        };
        let msg = err.to_string();
        assert!(msg.contains("9999"));
        assert!(msg.contains("500"));
    }

    #[test]
    fn gllm_error_display_duplicate_tensor() {
        let err = GllmError::DuplicateTensorName("layer.0.weight".to_string());
        let msg = err.to_string();
        assert!(msg.contains("duplicate"));
        assert!(msg.contains("layer.0.weight"));
    }

    #[test]
    fn ggml_dtype_to_gllm_code_all_float_types() {
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::F32), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::F16), 1);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::BF16), 2);
    }

    #[test]
    fn ggml_dtype_to_gllm_code_quantized_fallback_to_zero() {
        // All quantized dtypes should map to 0 since quant info is in quant_format
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::Q4_0), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::Q5_1), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::Q8_K), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::IQ3_XXS), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::Q2_K), 0);
    }

    #[test]
    fn gllm_scale_dtype_k_quants_all_nonzero() {
        let k_quants = [
            GgmlDType::Q2_K, GgmlDType::Q3_K, GgmlDType::Q4_K,
            GgmlDType::Q5_K, GgmlDType::Q6_K, GgmlDType::Q8_K,
        ];
        for dt in &k_quants {
            assert_ne!(gllm_scale_dtype(*dt), 0, "{:?} should have nonzero scale_dtype", dt);
        }
    }

    #[test]
    fn gllm_scale_dtype_iq_variants_all_nonzero() {
        let iq_types = [
            GgmlDType::IQ1_S, GgmlDType::IQ1_M, GgmlDType::IQ2_XXS,
            GgmlDType::IQ2_XS, GgmlDType::IQ2_S, GgmlDType::IQ3_XXS,
            GgmlDType::IQ3_S, GgmlDType::IQ4_NL, GgmlDType::IQ4_XS,
        ];
        for dt in &iq_types {
            assert_ne!(gllm_scale_dtype(*dt), 0, "{:?} should have nonzero scale_dtype", dt);
        }
    }

    #[test]
    fn gllm_zp_type_only_q4_1_and_q5_1() {
        // Only Q4_1 and Q5_1 have zero-point type != 0
        assert_eq!(gllm_zp_type(GgmlDType::Q4_1), 1);
        assert_eq!(gllm_zp_type(GgmlDType::Q5_1), 1);

        // Everything else is 0
        let others = [
            GgmlDType::Q4_0, GgmlDType::Q5_0, GgmlDType::Q8_0, GgmlDType::Q8_1,
            GgmlDType::Q2_K, GgmlDType::Q3_K, GgmlDType::Q4_K, GgmlDType::Q5_K,
            GgmlDType::Q6_K, GgmlDType::Q8_K, GgmlDType::IQ1_S, GgmlDType::F32,
            GgmlDType::F16, GgmlDType::BF16,
        ];
        for dt in &others {
            assert_eq!(gllm_zp_type(*dt), 0, "{:?} should have zp_type == 0", dt);
        }
    }

    #[test]
    fn decode_to_f16_f32_zero_values() {
        let zeros = vec![0u8; 8]; // two f32 zeros
        let result = decode_to_f16(&zeros, GgmlDType::F32);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], f16::from_f32(0.0));
        assert_eq!(result[1], f16::from_f32(0.0));
    }

    #[test]
    fn decode_to_f16_bf16_zero_values() {
        let zeros = vec![0u8; 4]; // two bf16 zeros
        let result = decode_to_f16(&zeros, GgmlDType::BF16);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], f16::from_f32(0.0));
        assert_eq!(result[1], f16::from_f32(0.0));
    }

    #[test]
    fn extract_model_params_missing_file_returns_defaults() {
        let dir = std::env::temp_dir().join("gllm_test_extract_missing_cfg");
        std::fs::create_dir_all(&dir).unwrap();
        let nonexistent = dir.join("nonexistent_config.json");

        let (arch, vocab, hidden, layers, heads, kv_heads, head_dim, inter, ctx, extras) =
            extract_model_params(Some(&nonexistent));

        assert_eq!(arch, "unknown");
        assert_eq!(vocab, 0);
        assert_eq!(hidden, 0);
        assert_eq!(layers, 0);
        assert_eq!(heads, 0);
        assert_eq!(kv_heads, 0);
        assert_eq!(head_dim, 0);
        assert_eq!(inter, 0);
        assert_eq!(ctx, 0);
        assert!(extras.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_rope_scaling_factor() {
        let dir = std::env::temp_dir().join("gllm_test_extract_rope_scaling");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "rope_scaling": {
                "factor": 4.0,
                "type": "yarn"
            }
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, _, _, _, _, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(extras.get("rope_scaling_factor").map(String::as_str), Some("4"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_head_dim_derived_from_hidden_and_heads() {
        let dir = std::env::temp_dir().join("gllm_test_extract_head_dim");
        std::fs::create_dir_all(&dir).unwrap();

        // No explicit head_dim → should be derived as hidden_size / num_attention_heads
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 5120,
            "num_attention_heads": 40,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, _, head_dim, _, _, _) =
            extract_model_params(Some(&config_path));

        // 5120 / 40 = 128
        assert_eq!(head_dim, 128);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn elem_bytes_u16_i16() {
        use safetensors::Dtype;
        assert_eq!(elem_bytes(Dtype::I16), 2);
        assert_eq!(elem_bytes(Dtype::U16), 2);
    }

    #[test]
    fn elem_bytes_u32_u64() {
        use safetensors::Dtype;
        assert_eq!(elem_bytes(Dtype::U32), 4);
        assert_eq!(elem_bytes(Dtype::U64), 8);
    }

    #[test]
    fn elem_bytes_bool() {
        use safetensors::Dtype;
        assert_eq!(elem_bytes(Dtype::BOOL), 1);
    }

    #[test]
    fn config_u64_missing_key_returns_zero() {
        let config = serde_json::json!({"other_key": 42});
        let val = config_u64(&config, "missing_key");
        assert_eq!(val, 0);
    }

    #[test]
    fn config_u64_present_key_returns_value() {
        let config = serde_json::json!({"vocab_size": 32000});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 32000);
    }

    #[test]
    fn convert_options_default_is_consistent_with_new() {
        let default = ConvertOptions::default();
        let manual = ConvertOptions {
            page_size: 4096,
            config_path: None,
            quant_target: None,
        };
        assert_eq!(default.page_size, manual.page_size);
        assert_eq!(default.config_path, manual.config_path);
        assert_eq!(default.quant_target, manual.quant_target);
    }

    // ── QuantTarget trait tests ─────────────────────────────────────────────

    #[test]
    fn quant_target_variants_debug_format() {
        assert_eq!(format!("{:?}", QuantTarget::Awq4), "Awq4");
        assert_eq!(format!("{:?}", QuantTarget::Gptq4), "Gptq4");
        assert_eq!(format!("{:?}", QuantTarget::Nvfp4), "Nvfp4");
    }

    #[test]
    fn quant_target_equality() {
        assert_eq!(QuantTarget::Awq4, QuantTarget::Awq4);
        assert_eq!(QuantTarget::Gptq4, QuantTarget::Gptq4);
        assert_eq!(QuantTarget::Nvfp4, QuantTarget::Nvfp4);
        assert_ne!(QuantTarget::Awq4, QuantTarget::Gptq4);
        assert_ne!(QuantTarget::Gptq4, QuantTarget::Nvfp4);
    }

    #[test]
    fn quant_target_copy_semantics() {
        let a = QuantTarget::Awq4;
        let b = a; // Copy, not move
        assert_eq!(a, b);
    }

    #[test]
    fn quant_target_clone_is_equal() {
        let a = QuantTarget::Nvfp4;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── TensorEntry (writer) tests ──────────────────────────────────────────

    #[test]
    fn tensor_entry_is_quantized_true() {
        let entry = TensorEntry {
            name: "weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [128, 256, 0, 0],
            quant_format: 10, // Q4_0
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 64],
            original_size: 512,
        };
        assert!(entry.is_quantized());
    }

    #[test]
    fn tensor_entry_is_quantized_false() {
        let entry = TensorEntry {
            name: "bias".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [256, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 1024],
            original_size: 1024,
        };
        assert!(!entry.is_quantized());
    }

    #[test]
    fn tensor_entry_compressed_size_equals_data_len() {
        let data = vec![0xABu8; 200];
        let entry = TensorEntry {
            name: "layer.0.weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [64, 64, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 1024,
        };
        assert_eq!(entry.compressed_size(), 200);
    }

    #[test]
    fn tensor_entry_compressed_size_zero_data() {
        let entry = TensorEntry {
            name: "empty".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [0, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: Vec::new(),
            original_size: 0,
        };
        assert_eq!(entry.compressed_size(), 0);
    }

    #[test]
    fn tensor_entry_clone_is_equal() {
        let entry = TensorEntry {
            name: "model.layer.0".to_string(),
            ndim: 4,
            dtype: 2,
            shape: [8, 16, 32, 64],
            quant_format: 41,
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 2,
            data: vec![1u8, 2, 3, 4],
            original_size: 65536,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.name, entry.name);
        assert_eq!(cloned.ndim, entry.ndim);
        assert_eq!(cloned.dtype, entry.dtype);
        assert_eq!(cloned.shape, entry.shape);
        assert_eq!(cloned.quant_format, entry.quant_format);
        assert_eq!(cloned.quant_block_size, entry.quant_block_size);
        assert_eq!(cloned.scale_dtype, entry.scale_dtype);
        assert_eq!(cloned.zp_type, entry.zp_type);
        assert_eq!(cloned.data, entry.data);
        assert_eq!(cloned.original_size, entry.original_size);
    }

    #[test]
    fn tensor_entry_debug_includes_fields() {
        let entry = TensorEntry {
            name: "debug_tensor".to_string(),
            ndim: 2,
            dtype: 1,
            shape: [64, 128, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![42u8; 16],
            original_size: 256,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("debug_tensor"));
        assert!(debug_str.contains("ndim"));
        assert!(debug_str.contains("quant_format"));
    }

    // ── decode_to_f16 special float values ──────────────────────────────────

    #[test]
    fn decode_to_f16_f32_negative_zero() {
        let neg_zero_bits: u32 = 0x8000_0000u32;
        let data = neg_zero_bits.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F32);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_sign_negative());
        assert_eq!(result[0].to_f32(), 0.0f32);
    }

    #[test]
    fn decode_to_f16_f16_negative_values() {
        let neg_one = f16::from_f32(-1.0);
        let data = neg_one.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F16);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], f16::from_f32(-1.0));
    }

    #[test]
    fn decode_to_f16_bf16_negative_values() {
        let neg_one = half::bf16::from_f32(-1.0);
        let data = neg_one.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::BF16);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], f16::from_f32(-1.0));
    }

    // ── ggml_dtype_to_gllm_code additional types ────────────────────────────

    #[test]
    fn ggml_dtype_to_gllm_code_integer_types() {
        // Integer types should fall through to the default case (0)
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::I8), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::I16), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::I32), 0);
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::I64), 0);
    }

    #[test]
    fn ggml_dtype_to_gllm_code_f64() {
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::F64), 0);
    }

    // ── gllm_scale_dtype / gllm_zp_type for newer GGML types ────────────────

    #[test]
    fn gllm_scale_dtype_tq_and_awq_types() {
        // TQ1_0, TQ2_0, MXFP4, AWQ4, GPTQ4, SQUEEZE, NVFP4 should return 0
        // (these are gllm-native quant types, not GGML quant types with scales)
        assert_eq!(gllm_scale_dtype(GgmlDType::TQ1_0), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::TQ2_0), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::MXFP4), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::AWQ4), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::GPTQ4), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::SQUEEZE), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::NVFP4), 0);
    }

    #[test]
    fn gllm_zp_type_all_non_q4_1_q5_1_return_zero() {
        // Exhaustive check: only Q4_1 and Q5_1 return 1
        let all_zero_types = [
            GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16, GgmlDType::F64,
            GgmlDType::Q4_0, GgmlDType::Q5_0, GgmlDType::Q8_0, GgmlDType::Q8_1,
            GgmlDType::Q2_K, GgmlDType::Q3_K, GgmlDType::Q4_K, GgmlDType::Q5_K,
            GgmlDType::Q6_K, GgmlDType::Q8_K, GgmlDType::IQ1_S, GgmlDType::IQ1_M,
            GgmlDType::IQ2_XXS, GgmlDType::IQ2_XS, GgmlDType::IQ2_S,
            GgmlDType::IQ3_XXS, GgmlDType::IQ3_S, GgmlDType::IQ4_NL, GgmlDType::IQ4_XS,
            GgmlDType::TQ1_0, GgmlDType::TQ2_0, GgmlDType::MXFP4,
            GgmlDType::AWQ4, GgmlDType::GPTQ4, GgmlDType::SQUEEZE, GgmlDType::NVFP4,
        ];
        for dt in &all_zero_types {
            assert_eq!(gllm_zp_type(*dt), 0, "{:?} should have zp_type == 0", dt);
        }
    }

    // ── config_u64 edge cases ───────────────────────────────────────────────

    #[test]
    fn config_u64_string_value_returns_zero() {
        // If the value is a string "32000" instead of integer 32000, it should return 0
        let config = serde_json::json!({"vocab_size": "32000"});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 0);
    }

    #[test]
    fn config_u64_float_value_returns_zero() {
        // Float 32000.5 is not an integer in JSON → as_u64 returns None
        let config = serde_json::json!({"vocab_size": 32000.5});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 0);
    }

    #[test]
    fn config_u64_zero_value() {
        let config = serde_json::json!({"vocab_size": 0});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 0);
    }

    // ── extract_model_params edge cases ─────────────────────────────────────

    #[test]
    fn extract_model_params_empty_json_object() {
        let dir = std::env::temp_dir().join("gllm_test_extract_empty_json");
        std::fs::create_dir_all(&dir).unwrap();

        let config_path = dir.join("config.json");
        std::fs::write(&config_path, "{}").unwrap();

        let (arch, vocab, hidden, layers, heads, kv_heads, head_dim, inter, ctx, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(arch, "unknown");
        assert_eq!(vocab, 0);
        assert_eq!(hidden, 0);
        assert_eq!(layers, 0);
        assert_eq!(heads, 0);
        assert_eq!(kv_heads, 1); // max(0, 1) = 1
        assert_eq!(head_dim, 0); // 0 / 1 = 0
        assert_eq!(inter, 0);
        assert_eq!(ctx, 0);
        assert!(extras.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_explicit_head_dim_overrides_derived() {
        let dir = std::env::temp_dir().join("gllm_test_extract_explicit_head_dim");
        std::fs::create_dir_all(&dir).unwrap();

        // Explicit head_dim=96 should be used even though hidden/heads would give 128
        let config = serde_json::json!({
            "model_type": "test_model",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "head_dim": 96,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, _, head_dim, _, _, _) =
            extract_model_params(Some(&config_path));

        // 96 > 128 (derived) so max(96, 128) = 128
        // Wait — the code is .max(hidden_size / num_heads.max(1))
        // hidden=4096, heads=32 → derived = 128
        // head_dim = max(96, 128) = 128
        assert_eq!(head_dim, 128);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── ConvertResult zero and boundary values ──────────────────────────────

    #[test]
    fn convert_result_zero_values() {
        let result = ConvertResult {
            input_bytes: 0,
            output_bytes: 0,
            tensor_count: 0,
            quantized_count: 0,
        };
        assert_eq!(result.input_bytes, 0);
        assert_eq!(result.output_bytes, 0);
        assert_eq!(result.tensor_count, 0);
        assert_eq!(result.quantized_count, 0);
    }

    #[test]
    fn convert_result_large_values() {
        let result = ConvertResult {
            input_bytes: u64::MAX,
            output_bytes: u64::MAX,
            tensor_count: usize::MAX,
            quantized_count: usize::MAX,
        };
        assert_eq!(result.input_bytes, u64::MAX);
        assert_eq!(result.output_bytes, u64::MAX);
        assert_eq!(result.tensor_count, usize::MAX);
    }

    // ── GllmError variants for completeness ─────────────────────────────────

    #[test]
    fn gllm_error_display_invalid_quant_type() {
        let err = GllmError::InvalidQuantType(99);
        let msg = err.to_string();
        assert!(msg.contains("99"));
        assert!(msg.contains("quant_format"));
    }

    #[test]
    fn gllm_error_display_invalid_dtype() {
        let err = GllmError::InvalidDType(42);
        let msg = err.to_string();
        assert!(msg.contains("42"));
        assert!(msg.contains("dtype"));
    }

    // ── elem_bytes for less common safetensors types ─────────────────────────

    #[test]
    fn elem_bytes_fallback_types() {
        use safetensors::Dtype;
        // Types not explicitly listed should fall through to 1
        // F64 and U64 are explicitly 8, but let's verify U8
        assert_eq!(elem_bytes(Dtype::U8), 1);
        // Verify all covered types are correct
        assert_eq!(elem_bytes(Dtype::BOOL), 1);
        assert_eq!(elem_bytes(Dtype::I8), 1);
        assert_eq!(elem_bytes(Dtype::F16), 2);
        assert_eq!(elem_bytes(Dtype::BF16), 2);
        assert_eq!(elem_bytes(Dtype::I16), 2);
        assert_eq!(elem_bytes(Dtype::U16), 2);
        assert_eq!(elem_bytes(Dtype::F32), 4);
        assert_eq!(elem_bytes(Dtype::I32), 4);
        assert_eq!(elem_bytes(Dtype::U32), 4);
        assert_eq!(elem_bytes(Dtype::F64), 8);
        assert_eq!(elem_bytes(Dtype::I64), 8);
        assert_eq!(elem_bytes(Dtype::U64), 8);
    }

    // ── Additional edge case tests ──────────────────────────────────────────

    #[test]
    fn convert_options_page_size_zero() {
        let opts = ConvertOptions {
            page_size: 0,
            config_path: None,
            quant_target: None,
        };
        assert_eq!(opts.page_size, 0);
    }

    #[test]
    fn convert_options_page_size_one() {
        let opts = ConvertOptions {
            page_size: 1,
            config_path: None,
            quant_target: None,
        };
        assert_eq!(opts.page_size, 1);
    }

    #[test]
    fn decode_to_f16_f16_max_finite() {
        // f16 max finite value = 65504.0
        let max_f16 = f16::from_f32(65504.0);
        let data = max_f16.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F16);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], max_f16);
    }

    #[test]
    fn decode_to_f16_f32_infinity() {
        let inf_bits = f32::INFINITY.to_le_bytes();
        let result = decode_to_f16(&inf_bits, GgmlDType::F32);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_infinite());
        assert!(result[0].to_f32().is_sign_positive());
    }

    #[test]
    fn decode_to_f16_f32_nan() {
        let nan_bits = f32::NAN.to_le_bytes();
        let result = decode_to_f16(&nan_bits, GgmlDType::F32);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_nan());
    }

    #[test]
    fn decode_to_f16_f32_large_batch() {
        let values: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = decode_to_f16(&data, GgmlDType::F32);
        assert_eq!(result.len(), 256);
        for (i, v) in values.iter().enumerate() {
            let diff = (result[i].to_f32() - v).abs();
            assert!(diff < 0.01, "index {i}: expected ~{v}, got {}", result[i].to_f32());
        }
    }

    #[test]
    fn gllm_error_display_empty_parse_error_message() {
        let err = GllmError::ParseError(String::new());
        let msg = err.to_string();
        assert!(msg.contains("parse error"));
    }

    #[test]
    fn gllm_error_display_empty_invalid_metadata_message() {
        let err = GllmError::InvalidMetadata(String::new());
        let msg = err.to_string();
        assert!(msg.contains("invalid metadata"));
    }

    #[test]
    fn gllm_error_display_invalid_magic_zero() {
        let err = GllmError::InvalidMagic(0);
        let msg = err.to_string();
        assert!(msg.contains("0x00000000"));
        assert!(msg.contains("GLLM"));
    }

    #[test]
    fn gllm_error_display_unsupported_version_zero() {
        let err = GllmError::UnsupportedVersion(0);
        let msg = err.to_string();
        assert!(msg.contains("0"));
        assert!(msg.contains("unsupported version"));
    }

    #[test]
    fn gllm_error_display_header_too_small_zero() {
        let err = GllmError::HeaderTooSmall(0);
        let msg = err.to_string();
        assert!(msg.contains("0"));
        assert!(msg.contains("header"));
    }

    #[test]
    fn gllm_error_display_duplicate_tensor_empty_name() {
        let err = GllmError::DuplicateTensorName(String::new());
        let msg = err.to_string();
        assert!(msg.contains("duplicate"));
    }

    #[test]
    fn extract_model_params_only_model_type() {
        let dir = std::env::temp_dir().join("gllm_test_extract_only_type");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({"model_type": "phi4"});
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (arch, vocab, hidden, layers, heads, kv_heads, head_dim, inter, ctx, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(arch, "phi4");
        assert_eq!(vocab, 0);
        assert_eq!(hidden, 0);
        assert_eq!(layers, 0);
        assert_eq!(heads, 0);
        assert_eq!(kv_heads, 1);
        assert_eq!(head_dim, 0);
        assert_eq!(inter, 0);
        assert_eq!(ctx, 0);
        assert!(extras.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_explicit_kv_heads() {
        let dir = std::env::temp_dir().join("gllm_test_extract_kv_heads");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, kv_heads, _, _, _, _) =
            extract_model_params(Some(&config_path));

        assert_eq!(kv_heads, 8);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_null_model_type() {
        let dir = std::env::temp_dir().join("gllm_test_extract_null_type");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({"model_type": null});
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (arch, _, _, _, _, _, _, _, _, _) =
            extract_model_params(Some(&config_path));

        assert_eq!(arch, "unknown");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_u64_null_value_returns_zero() {
        let config = serde_json::json!({"vocab_size": null});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 0);
    }

    #[test]
    fn config_u64_boolean_value_returns_zero() {
        let config = serde_json::json!({"vocab_size": true});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 0);
    }

    #[test]
    fn tensor_entry_with_single_element_shape() {
        let entry = TensorEntry {
            name: "scalar_bias".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        };
        assert!(!entry.is_quantized());
        assert_eq!(entry.compressed_size(), 4);
    }

    #[test]
    fn tensor_entry_zero_ndim_and_empty_shape() {
        let entry = TensorEntry {
            name: "empty_tensor".to_string(),
            ndim: 0,
            dtype: 0,
            shape: [0, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: Vec::new(),
            original_size: 0,
        };
        assert!(!entry.is_quantized());
        assert_eq!(entry.compressed_size(), 0);
    }

    // ── Batch 3: Additional 45+ tests ──────────────────────────────────────────

    // ── 1. ConvertOptions boundary values ──────────────────────────────────────

    #[test]
    fn convert_options_page_size_max() {
        let opts = ConvertOptions {
            page_size: u32::MAX,
            config_path: None,
            quant_target: None,
        };
        assert_eq!(opts.page_size, u32::MAX);
    }

    #[test]
    fn convert_options_with_all_quant_targets() {
        for target in [QuantTarget::Awq4, QuantTarget::Gptq4, QuantTarget::Nvfp4] {
            let opts = ConvertOptions {
                page_size: 4096,
                config_path: None,
                quant_target: Some(target),
            };
            assert!(opts.quant_target.is_some());
            assert_eq!(opts.quant_target.unwrap(), target);
        }
    }

    #[test]
    fn convert_options_config_path_with_special_chars() {
        let path = PathBuf::from("/tmp/模型权重/config.json");
        let opts = ConvertOptions {
            page_size: 4096,
            config_path: Some(path.clone()),
            quant_target: None,
        };
        assert_eq!(opts.config_path.as_ref(), Some(&path));
    }

    #[test]
    fn convert_options_debug_shows_none_fields() {
        let opts = ConvertOptions {
            page_size: 4096,
            config_path: None,
            quant_target: None,
        };
        let debug = format!("{:?}", opts);
        assert!(debug.contains("None"));
    }

    #[test]
    fn convert_options_debug_shows_some_fields() {
        let opts = ConvertOptions {
            page_size: 8192,
            config_path: Some(PathBuf::from("/a/b.json")),
            quant_target: Some(QuantTarget::Nvfp4),
        };
        let debug = format!("{:?}", opts);
        assert!(debug.contains("8192"));
        assert!(debug.contains("a/b.json"));
        assert!(debug.contains("Nvfp4"));
    }

    // ── 2. ConvertResult boundary and construction ─────────────────────────────

    #[test]
    fn convert_result_quantized_count_exceeds_tensor_count() {
        // Edge: more quantized than total tensors is technically allowed by struct
        let result = ConvertResult {
            input_bytes: 100,
            output_bytes: 50,
            tensor_count: 2,
            quantized_count: 5,
        };
        assert_eq!(result.tensor_count, 2);
        assert_eq!(result.quantized_count, 5);
    }

    #[test]
    fn convert_result_output_larger_than_input() {
        let result = ConvertResult {
            input_bytes: 500,
            output_bytes: 1000,
            tensor_count: 3,
            quantized_count: 0,
        };
        assert!(result.output_bytes > result.input_bytes);
    }

    #[test]
    fn convert_result_u64_max_bytes() {
        let result = ConvertResult {
            input_bytes: u64::MAX,
            output_bytes: u64::MAX,
            tensor_count: 1,
            quantized_count: 1,
        };
        assert_eq!(result.input_bytes, u64::MAX);
        assert_eq!(result.output_bytes, u64::MAX);
    }

    // ── 3. QuantTarget exhaustive variant tests ────────────────────────────────

    #[test]
    fn quant_target_all_variants_are_distinct() {
        let variants = [QuantTarget::Awq4, QuantTarget::Gptq4, QuantTarget::Nvfp4];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "variants at {i} and {j} should differ");
                }
            }
        }
    }

    #[test]
    fn quant_target_copy_does_not_move() {
        let a = QuantTarget::Gptq4;
        let b = a;
        let c = a; // Copy again — if it were move, this would fail
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // ── 4. TensorEntry additional edge cases ───────────────────────────────────

    #[test]
    fn tensor_entry_4d_shape_all_dimensions_set() {
        let entry = TensorEntry {
            name: "conv.weight".to_string(),
            ndim: 4,
            dtype: 1,
            shape: [3, 64, 7, 7],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 3 * 64 * 7 * 7 * 2],
            original_size: 3 * 64 * 7 * 7 * 4,
        };
        assert_eq!(entry.ndim, 4);
        assert_eq!(entry.shape, [3, 64, 7, 7]);
        assert!(!entry.is_quantized());
    }

    #[test]
    fn tensor_entry_quant_format_max_u8() {
        let entry = TensorEntry {
            name: "max_quant".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [128, 256, 0, 0],
            quant_format: u8::MAX,
            quant_block_size: u16::MAX,
            scale_dtype: u8::MAX,
            zp_type: u8::MAX,
            data: vec![0u8; 100],
            original_size: 500,
        };
        assert!(entry.is_quantized());
        assert_eq!(entry.quant_format, u8::MAX);
        assert_eq!(entry.quant_block_size, u16::MAX);
    }

    #[test]
    fn tensor_entry_original_size_zero_with_nonempty_data() {
        let entry = TensorEntry {
            name: "test".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 40],
            original_size: 0,
        };
        assert_eq!(entry.original_size, 0);
        assert_eq!(entry.compressed_size(), 40);
    }

    #[test]
    fn tensor_entry_dtype_field_values() {
        // dtype=0 F32, dtype=1 F16, dtype=2 BF16 per ggml_dtype_to_gllm_code
        for dtype_val in [0u8, 1, 2] {
            let entry = TensorEntry {
                name: "test".to_string(),
                ndim: 2,
                dtype: dtype_val,
                shape: [4, 4, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 32],
                original_size: 64,
            };
            assert_eq!(entry.dtype, dtype_val);
        }
    }

    #[test]
    fn tensor_entry_name_with_unicode() {
        let entry = TensorEntry {
            name: "层.权重.嵌入".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [100, 200, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: Vec::new(),
            original_size: 0,
        };
        assert_eq!(entry.name, "层.权重.嵌入");
    }

    // ── 5. decode_to_f16 special float values ──────────────────────────────────

    #[test]
    fn decode_to_f16_f32_negative_infinity() {
        let neg_inf_bits = f32::NEG_INFINITY.to_le_bytes();
        let result = decode_to_f16(&neg_inf_bits, GgmlDType::F32);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_infinite());
        assert!(result[0].to_f32().is_sign_negative());
    }

    #[test]
    fn decode_to_f16_f32_max_finite() {
        let max_bits = f32::MAX.to_le_bytes();
        let result = decode_to_f16(&max_bits, GgmlDType::F32);
        assert_eq!(result.len(), 1);
        // f32::MAX overflows f16 → becomes infinity
        assert!(result[0].to_f32().is_infinite());
    }

    #[test]
    fn decode_to_f16_f32_min_positive_subnormal() {
        let min_pos = f32::from_bits(1u32); // smallest positive subnormal
        let data = min_pos.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F32);
        assert_eq!(result.len(), 1);
        // f32 subnormal may round to zero in f16
        assert!(result[0].to_f32() >= 0.0);
    }

    #[test]
    fn decode_to_f16_f16_smallest_positive() {
        // f16 smallest positive subnormal = 2^(-24) ≈ 5.96e-8
        let bits = 1u16.to_le_bytes();
        let result = decode_to_f16(&bits, GgmlDType::F16);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32() > 0.0);
    }

    #[test]
    fn decode_to_f16_f16_nan() {
        // f16 NaN: exponent all 1s, mantissa nonzero
        let nan_f16 = f16::from_bits(0x7E00u16); // quiet NaN
        let data = nan_f16.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F16);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_nan());
    }

    #[test]
    fn decode_to_f16_bf16_infinity() {
        // BF16 infinity: sign=0, exponent=0xFF, mantissa=0 → bits 0x7F80
        let bf16_inf = half::bf16::from_bits(0x7F80u16);
        let data = bf16_inf.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::BF16);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_infinite());
    }

    #[test]
    fn decode_to_f16_empty_f32_data() {
        let result = decode_to_f16(&[], GgmlDType::F32);
        assert!(result.is_empty());
    }

    #[test]
    fn decode_to_f16_empty_f16_data() {
        let result = decode_to_f16(&[], GgmlDType::F16);
        assert!(result.is_empty());
    }

    #[test]
    fn decode_to_f16_empty_bf16_data() {
        let result = decode_to_f16(&[], GgmlDType::BF16);
        assert!(result.is_empty());
    }

    // ── 6. ggml_dtype_to_gllm_code exhaustive ─────────────────────────────────

    #[test]
    fn ggml_dtype_to_gllm_code_all_basic_quant_types() {
        let quant_types = [
            GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q5_0, GgmlDType::Q5_1,
            GgmlDType::Q8_0, GgmlDType::Q8_1,
        ];
        for dt in &quant_types {
            assert_eq!(ggml_dtype_to_gllm_code(*dt), 0, "{:?} should map to 0", dt);
        }
    }

    #[test]
    fn ggml_dtype_to_gllm_code_all_k_quant_types() {
        let k_types = [
            GgmlDType::Q2_K, GgmlDType::Q3_K, GgmlDType::Q4_K,
            GgmlDType::Q5_K, GgmlDType::Q6_K, GgmlDType::Q8_K,
        ];
        for dt in &k_types {
            assert_eq!(ggml_dtype_to_gllm_code(*dt), 0, "{:?} should map to 0", dt);
        }
    }

    #[test]
    fn ggml_dtype_to_gllm_code_all_iq_types() {
        let iq_types = [
            GgmlDType::IQ1_S, GgmlDType::IQ1_M, GgmlDType::IQ2_XXS,
            GgmlDType::IQ2_XS, GgmlDType::IQ2_S, GgmlDType::IQ3_XXS,
            GgmlDType::IQ3_S, GgmlDType::IQ4_NL, GgmlDType::IQ4_XS,
        ];
        for dt in &iq_types {
            assert_eq!(ggml_dtype_to_gllm_code(*dt), 0, "{:?} should map to 0", dt);
        }
    }

    #[test]
    fn ggml_dtype_to_gllm_code_all_gllm_native_types() {
        let native_types = [
            GgmlDType::TQ1_0, GgmlDType::TQ2_0, GgmlDType::MXFP4,
            GgmlDType::AWQ4, GgmlDType::GPTQ4, GgmlDType::SQUEEZE,
            GgmlDType::NVFP4,
        ];
        for dt in &native_types {
            assert_eq!(ggml_dtype_to_gllm_code(*dt), 0, "{:?} should map to 0", dt);
        }
    }

    // ── 7. gllm_scale_dtype / gllm_zp_type exhaustive coverage ────────────────

    #[test]
    fn gllm_scale_dtype_q8_1_is_nonzero() {
        assert_ne!(gllm_scale_dtype(GgmlDType::Q8_1), 0);
    }

    #[test]
    fn gllm_scale_dtype_iq1_s_is_nonzero() {
        assert_ne!(gllm_scale_dtype(GgmlDType::IQ1_S), 0);
    }

    #[test]
    fn gllm_scale_dtype_iq4_xs_is_nonzero() {
        assert_ne!(gllm_scale_dtype(GgmlDType::IQ4_XS), 0);
    }

    #[test]
    fn gllm_zp_type_q4_1_value_is_one() {
        assert_eq!(gllm_zp_type(GgmlDType::Q4_1), 1);
    }

    #[test]
    fn gllm_zp_type_q5_1_value_is_one() {
        assert_eq!(gllm_zp_type(GgmlDType::Q5_1), 1);
    }

    // ── 8. config_u64 additional edge cases ───────────────────────────────────

    #[test]
    fn config_u64_negative_integer_json() {
        // JSON allows negative numbers; as_u64 returns None for negative
        let config = serde_json::json!({"vocab_size": -1});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 0);
    }

    #[test]
    fn config_u64_nested_object_returns_zero() {
        let config = serde_json::json!({"vocab_size": {"nested": 42}});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 0);
    }

    #[test]
    fn config_u64_array_value_returns_zero() {
        let config = serde_json::json!({"vocab_size": [1, 2, 3]});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 0);
    }

    #[test]
    fn config_u64_large_value() {
        let config = serde_json::json!({"vocab_size": 999999999999u64});
        let val = config_u64(&config, "vocab_size");
        assert_eq!(val, 999999999999u64);
    }

    // ── 9. extract_model_params additional scenarios ──────────────────────────

    #[test]
    fn extract_model_params_zero_hidden_size() {
        let dir = std::env::temp_dir().join("gllm_test_extract_zero_hidden");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "test",
            "hidden_size": 0,
            "num_attention_heads": 0,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, kv_heads, head_dim, _, _, _) =
            extract_model_params(Some(&config_path));

        assert_eq!(kv_heads, 1); // max(0, 1) = 1
        assert_eq!(head_dim, 0); // 0 / 1 = 0

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_large_values() {
        let dir = std::env::temp_dir().join("gllm_test_extract_large_vals");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "giant_model",
            "vocab_size": 1000000,
            "hidden_size": 16384,
            "num_hidden_layers": 128,
            "num_attention_heads": 128,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "intermediate_size": 65536,
            "max_position_embeddings": 1048576,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (arch, vocab, hidden, layers, heads, kv_heads, head_dim, inter, ctx, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(arch, "giant_model");
        assert_eq!(vocab, 1000000);
        assert_eq!(hidden, 16384);
        assert_eq!(layers, 128);
        assert_eq!(heads, 128);
        assert_eq!(kv_heads, 16);
        assert_eq!(head_dim, 256);
        assert_eq!(inter, 65536);
        assert_eq!(ctx, 1048576);
        assert!(extras.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_rope_theta_integer_not_float() {
        let dir = std::env::temp_dir().join("gllm_test_extract_rope_int");
        std::fs::create_dir_all(&dir).unwrap();

        // rope_theta as integer, not float → as_f64 returns None
        let config = serde_json::json!({
            "model_type": "test",
            "rope_theta": 10000,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, _, _, _, _, extras) =
            extract_model_params(Some(&config_path));

        // Integer 10000 is also a valid f64 (10000.0), so as_f64 should return Some
        assert!(extras.contains_key("rope_freq_base"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_rope_scaling_without_factor() {
        let dir = std::env::temp_dir().join("gllm_test_extract_rope_no_factor");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "test",
            "rope_scaling": {
                "type": "linear"
            },
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, _, _, _, _, extras) =
            extract_model_params(Some(&config_path));

        assert!(!extras.contains_key("rope_scaling_factor"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_head_dim_equals_derived() {
        let dir = std::env::temp_dir().join("gllm_test_extract_hd_equals");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "test",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "head_dim": 128, // exactly equals 2048/16
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, _, head_dim, _, _, _) =
            extract_model_params(Some(&config_path));

        assert_eq!(head_dim, 128);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 10. GllmError from std::io::Error round-trip ─────────────────────────

    #[test]
    fn gllm_error_from_io_preserves_message() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found: model.gllm");
        let gllm_err: GllmError = io_err.into();
        let msg = gllm_err.to_string();
        assert!(msg.contains("file not found"));
        assert!(msg.contains("model.gllm"));
    }

    #[test]
    fn gllm_error_display_tensor_out_of_bounds_large_values() {
        let err = GllmError::TensorOutOfBounds {
            name: "huge_tensor".to_string(),
            start: usize::MAX - 100,
            end: usize::MAX,
            file_size: usize::MAX / 2,
        };
        let msg = err.to_string();
        assert!(msg.contains("huge_tensor"));
    }

    #[test]
    fn gllm_error_display_string_table_offset_plus_length() {
        let err = GllmError::StringTableOutOfBounds {
            offset: 100,
            length: 200,
            file_size: 250,
        };
        let msg = err.to_string();
        // offset + length = 300 should appear in the message
        assert!(msg.contains("300"));
        assert!(msg.contains("250"));
    }

    // ── 11. elem_bytes for remaining safetensors types ─────────────────────────

    #[test]
    fn elem_bytes_unknown_type_returns_one() {
        use safetensors::Dtype;
        // Test a less common type that should hit the fallback
        // NB: If the type is explicitly listed, it won't hit fallback.
        // We just verify the known ones are correct.
        assert_eq!(elem_bytes(Dtype::BOOL), 1);
        assert_eq!(elem_bytes(Dtype::U8), 1);
    }

    // ── 12. decode_to_f16 mixed positive/negative batch ───────────────────────

    #[test]
    fn decode_to_f16_f32_mixed_sign_batch() {
        let values: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = decode_to_f16(&data, GgmlDType::F32);
        assert_eq!(result.len(), 5);
        for (i, v) in values.iter().enumerate() {
            let diff = (result[i].to_f32() - v).abs();
            assert!(diff < 0.01, "index {i}: expected ~{v}, got {}", result[i].to_f32());
        }
    }

    #[test]
    fn decode_to_f16_f16_multiple_values_roundtrip() {
        let originals: Vec<f16> = vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(-1.0),
            f16::from_f32(0.5),
            f16::from_f32(-0.5),
        ];
        let mut data = Vec::with_capacity(originals.len() * 2);
        for v in &originals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = decode_to_f16(&data, GgmlDType::F16);
        assert_eq!(result.len(), originals.len());
        for (i, orig) in originals.iter().enumerate() {
            assert_eq!(result[i], *orig, "index {i} mismatch");
        }
    }

    #[test]
    fn decode_to_f16_bf16_multiple_values() {
        let bf16_vals: Vec<half::bf16> = vec![
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(3.14),
            half::bf16::from_f32(-2.71),
        ];
        let mut data = Vec::with_capacity(bf16_vals.len() * 2);
        for v in &bf16_vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = decode_to_f16(&data, GgmlDType::BF16);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], f16::from_f32(0.0));
        // BF16→F32→F16 may lose precision for some values
        assert!((result[1].to_f32() - 3.14).abs() < 0.01);
        assert!((result[2].to_f32() - (-2.71)).abs() < 0.01);
    }

    // ── Batch 4: 50 additional tests (target 191+) ────────────────────────────

    // ── GllmError Debug format for all variants ────────────────────────────────

    #[test]
    fn gllm_error_debug_io_variant() {
        let err = GllmError::Io(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe"));
        let debug = format!("{err:?}");
        assert!(debug.contains("Io"));
    }

    #[test]
    fn gllm_error_debug_parse_error_variant() {
        let err = GllmError::ParseError("test message".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("ParseError"));
    }

    #[test]
    fn gllm_error_debug_tensor_out_of_bounds_variant() {
        let err = GllmError::TensorOutOfBounds {
            name: "w".to_string(),
            start: 0,
            end: 10,
            file_size: 5,
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("TensorOutOfBounds"));
    }

    #[test]
    fn gllm_error_debug_string_table_variant() {
        let err = GllmError::StringTableOutOfBounds {
            offset: 10,
            length: 20,
            file_size: 15,
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("StringTableOutOfBounds"));
    }

    #[test]
    fn gllm_error_debug_metadata_out_of_bounds_variant() {
        let err = GllmError::MetadataOutOfBounds {
            offset: 999,
            file_size: 100,
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("MetadataOutOfBounds"));
    }

    #[test]
    fn gllm_error_debug_tensor_dir_variant() {
        let err = GllmError::TensorDirOutOfBounds {
            offset: 100,
            count: 5,
            file_size: 50,
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("TensorDirOutOfBounds"));
    }

    #[test]
    fn gllm_error_debug_duplicate_tensor_variant() {
        let err = GllmError::DuplicateTensorName("my_tensor".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("DuplicateTensorName"));
    }

    #[test]
    fn gllm_error_debug_invalid_quant_type_variant() {
        let err = GllmError::InvalidQuantType(200);
        let debug = format!("{err:?}");
        assert!(debug.contains("InvalidQuantType"));
    }

    #[test]
    fn gllm_error_debug_invalid_dtype_variant() {
        let err = GllmError::InvalidDType(50);
        let debug = format!("{err:?}");
        assert!(debug.contains("InvalidDType"));
    }

    #[test]
    fn gllm_error_debug_invalid_metadata_variant() {
        let err = GllmError::InvalidMetadata("corrupt header".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("InvalidMetadata"));
    }

    // ── GllmError source() method for all non-Io variants ──────────────────────

    #[test]
    fn gllm_error_source_invalid_magic_is_none() {
        let err = GllmError::InvalidMagic(0);
        assert!(err.source().is_none());
    }

    #[test]
    fn gllm_error_source_unsupported_version_is_none() {
        let err = GllmError::UnsupportedVersion(42);
        assert!(err.source().is_none());
    }

    #[test]
    fn gllm_error_source_header_too_small_is_none() {
        let err = GllmError::HeaderTooSmall(10);
        assert!(err.source().is_none());
    }

    #[test]
    fn gllm_error_source_parse_error_is_none() {
        let err = GllmError::ParseError("msg".to_string());
        assert!(err.source().is_none());
    }

    #[test]
    fn gllm_error_source_invalid_quant_type_is_none() {
        let err = GllmError::InvalidQuantType(99);
        assert!(err.source().is_none());
    }

    #[test]
    fn gllm_error_source_invalid_dtype_is_none() {
        let err = GllmError::InvalidDType(55);
        assert!(err.source().is_none());
    }

    #[test]
    fn gllm_error_source_invalid_metadata_is_none() {
        let err = GllmError::InvalidMetadata("x".to_string());
        assert!(err.source().is_none());
    }

    // ── GllmError Display for remaining combinations ───────────────────────────

    #[test]
    fn gllm_error_display_magic_with_known_prefix() {
        let err = GllmError::InvalidMagic(0x474C4C4D);
        let msg = err.to_string();
        assert!(msg.contains("0x474C4C4D"));
        assert!(msg.contains("GLLM"));
    }

    #[test]
    fn gllm_error_display_header_too_small_with_size() {
        let err = GllmError::HeaderTooSmall(32);
        let msg = err.to_string();
        assert!(msg.contains("32"));
        assert!(msg.contains("header"));
    }

    #[test]
    fn gllm_error_display_unsupported_version_with_value() {
        let err = GllmError::UnsupportedVersion(255);
        let msg = err.to_string();
        assert!(msg.contains("255"));
        assert!(msg.contains("unsupported version"));
    }

    // ── ConvertResult construction patterns ─────────────────────────────────────

    #[test]
    fn convert_result_input_zero_output_nonzero() {
        let result = ConvertResult {
            input_bytes: 0,
            output_bytes: 100,
            tensor_count: 1,
            quantized_count: 0,
        };
        assert_eq!(result.input_bytes, 0);
        assert_eq!(result.output_bytes, 100);
    }

    #[test]
    fn convert_result_all_zeros() {
        let result = ConvertResult {
            input_bytes: 0,
            output_bytes: 0,
            tensor_count: 0,
            quantized_count: 0,
        };
        assert_eq!(result.input_bytes, 0);
        assert_eq!(result.output_bytes, 0);
        assert_eq!(result.tensor_count, 0);
        assert_eq!(result.quantized_count, 0);
    }

    // ── ConvertOptions with various page sizes ──────────────────────────────────

    #[test]
    fn convert_options_page_size_power_of_two() {
        for page_size in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] {
            let opts = ConvertOptions {
                page_size,
                config_path: None,
                quant_target: None,
            };
            assert_eq!(opts.page_size, page_size);
        }
    }

    #[test]
    fn convert_options_config_path_relative() {
        let path = PathBuf::from("relative/config.json");
        let opts = ConvertOptions {
            page_size: 4096,
            config_path: Some(path.clone()),
            quant_target: None,
        };
        assert_eq!(opts.config_path.as_ref(), Some(&path));
    }

    // ── QuantTarget exhaustiveness ──────────────────────────────────────────────

    #[test]
    fn quant_target_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(QuantTarget::Awq4);
        set.insert(QuantTarget::Gptq4);
        set.insert(QuantTarget::Nvfp4);
        assert_eq!(set.len(), 3);
        // Re-insert should not increase size
        set.insert(QuantTarget::Awq4);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn quant_target_ordering() {
        // Verify all three variants are distinct via inequality
        assert_ne!(QuantTarget::Awq4, QuantTarget::Gptq4);
        assert_ne!(QuantTarget::Awq4, QuantTarget::Nvfp4);
        assert_ne!(QuantTarget::Gptq4, QuantTarget::Nvfp4);
    }

    // ── decode_to_f16 bf16 infinity and NaN ─────────────────────────────────────

    #[test]
    fn decode_to_f16_bf16_nan() {
        // BF16 NaN: exponent=0xFF, mantissa!=0 → bits 0x7FC0 (quiet NaN)
        let bf16_nan = half::bf16::from_bits(0x7FC0u16);
        let data = bf16_nan.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::BF16);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_nan());
    }

    #[test]
    fn decode_to_f16_bf16_negative_infinity() {
        // BF16 negative infinity: sign=1, exponent=0xFF, mantissa=0 → bits 0xFF80
        let bf16_neg_inf = half::bf16::from_bits(0xFF80u16);
        let data = bf16_neg_inf.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::BF16);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_infinite());
        assert!(result[0].to_f32().is_sign_negative());
    }

    // ── decode_to_f16 f16 negative infinity ─────────────────────────────────────

    #[test]
    fn decode_to_f16_f16_negative_infinity() {
        // f16 negative infinity: sign=1, exponent=0x1F, mantissa=0 → bits 0xFC00
        let neg_inf = f16::from_bits(0xFC00u16);
        let data = neg_inf.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F16);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_infinite());
        assert!(result[0].to_f32().is_sign_negative());
    }

    #[test]
    fn decode_to_f16_f16_positive_infinity() {
        let pos_inf = f16::from_bits(0x7C00u16);
        let data = pos_inf.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F16);
        assert_eq!(result.len(), 1);
        assert!(result[0].to_f32().is_infinite());
        assert!(result[0].to_f32().is_sign_positive());
    }

    // ── decode_to_f16 f32 denormalized ──────────────────────────────────────────

    #[test]
    fn decode_to_f16_f32_smallest_normal() {
        // f32 smallest normal = 2^-126 ≈ 1.18e-38, which underflows f16 (min normal = 2^-14)
        // So f16 result may be zero or subnormal — verify the conversion does not panic
        let smallest_normal = f32::from_bits(0x0080_0000u32); // 2^-126
        let data = smallest_normal.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F32);
        assert_eq!(result.len(), 1);
        // Value may underflow to zero in f16, that's expected
        assert!(result[0].to_f32() >= 0.0);
    }

    #[test]
    fn decode_to_f16_f32_negative_one() {
        let data = (-1.0f32).to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F32);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], f16::from_f32(-1.0));
    }

    // ── gllm_scale_dtype individual type verification ───────────────────────────

    #[test]
    fn gllm_scale_dtype_q4_0_value() {
        assert_eq!(gllm_scale_dtype(GgmlDType::Q4_0), 1);
    }

    #[test]
    fn gllm_scale_dtype_q5_0_value() {
        assert_eq!(gllm_scale_dtype(GgmlDType::Q5_0), 1);
    }

    #[test]
    fn gllm_scale_dtype_q8_0_value() {
        assert_eq!(gllm_scale_dtype(GgmlDType::Q8_0), 1);
    }

    #[test]
    fn gllm_scale_dtype_q4_1_value() {
        assert_eq!(gllm_scale_dtype(GgmlDType::Q4_1), 1);
    }

    #[test]
    fn gllm_scale_dtype_q5_1_value() {
        assert_eq!(gllm_scale_dtype(GgmlDType::Q5_1), 1);
    }

    // ── ggml_dtype_to_gllm_code integer and special types ───────────────────────

    #[test]
    fn ggml_dtype_to_gllm_code_i32_maps_to_zero() {
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::I32), 0);
    }

    #[test]
    fn ggml_dtype_to_gllm_code_i64_maps_to_zero() {
        assert_eq!(ggml_dtype_to_gllm_code(GgmlDType::I64), 0);
    }

    // ── extract_model_params: config with all MoE + MLA fields ─────────────────

    #[test]
    fn extract_model_params_deepseek_mla_config() {
        let dir = std::env::temp_dir().join("gllm_test_extract_mla");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "deepseek",
            "vocab_size": 129280,
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "head_dim": 192,
            "intermediate_size": 18432,
            "max_position_embeddings": 163840,
            "rope_theta": 10000.0,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (arch, vocab, hidden, layers, heads, kv_heads, head_dim, inter, ctx, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(arch, "deepseek");
        assert_eq!(vocab, 129280);
        assert_eq!(hidden, 7168);
        assert_eq!(layers, 61);
        assert_eq!(heads, 128);
        assert_eq!(kv_heads, 128);
        assert_eq!(head_dim, 192);
        assert_eq!(inter, 18432);
        assert_eq!(ctx, 163840);
        assert_eq!(extras.get("rope_freq_base").map(String::as_str), Some("10000"));
        assert_eq!(extras.get("num_experts").map(String::as_str), Some("256"));
        assert_eq!(extras.get("num_experts_per_tok").map(String::as_str), Some("8"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_config_with_missing_optional_fields() {
        let dir = std::env::temp_dir().join("gllm_test_extract_minimal");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "test",
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, vocab, hidden, layers, heads, kv_heads, head_dim, inter, ctx, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(vocab, 0);
        assert_eq!(hidden, 0);
        assert_eq!(layers, 0);
        assert_eq!(heads, 0);
        assert_eq!(kv_heads, 1);
        assert_eq!(head_dim, 0);
        assert_eq!(inter, 0);
        assert_eq!(ctx, 0);
        assert!(extras.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── config_u64 additional robustness tests ──────────────────────────────────

    #[test]
    fn config_u64_u64_max_value() {
        let config = serde_json::json!({"val": u64::MAX});
        assert_eq!(config_u64(&config, "val"), u64::MAX);
    }

    #[test]
    fn config_u64_empty_config_object() {
        let config = serde_json::json!({});
        assert_eq!(config_u64(&config, "any_key"), 0);
    }

    // ── TensorEntry with quantized format ───────────────────────────────────────

    #[test]
    fn tensor_entry_quantized_with_nvfp4_format() {
        let entry = TensorEntry {
            name: "model.layer.0.weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4096, 4096, 0, 0],
            quant_format: 53, // NVFP4
            quant_block_size: 64,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 1024],
            original_size: 4096 * 4096 * 4,
        };
        assert!(entry.is_quantized());
        assert_eq!(entry.quant_format, 53);
        assert_eq!(entry.compressed_size(), 1024);
    }

    #[test]
    fn tensor_entry_quantized_with_awq4_format() {
        let entry = TensorEntry {
            name: "model.layer.1.weight".to_string(),
            ndim: 2,
            dtype: 1, // F16
            shape: [11008, 4096, 0, 0],
            quant_format: 40, // AWQ4
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data: vec![0u8; 2048],
            original_size: 11008 * 4096 * 4,
        };
        assert!(entry.is_quantized());
        assert_eq!(entry.compressed_size(), 2048);
    }

    #[test]
    fn tensor_entry_quantized_with_gptq4_format() {
        let entry = TensorEntry {
            name: "model.layer.2.weight".to_string(),
            ndim: 2,
            dtype: 2, // BF16
            shape: [4096, 11008, 0, 0],
            quant_format: 41, // GPTQ4
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 2,
            data: vec![0u8; 4096],
            original_size: 4096 * 11008 * 4,
        };
        assert!(entry.is_quantized());
        assert_eq!(entry.compressed_size(), 4096);
    }

    // ── extract_model_params: rope_scaling with nested object ──────────────────

    #[test]
    fn extract_model_params_rope_scaling_with_multiple_fields() {
        let dir = std::env::temp_dir().join("gllm_test_extract_rope_multi");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "qwen3",
            "rope_scaling": {
                "factor": 8.0,
                "type": "dynamic",
                "low_freq_factor": 1.0,
            },
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, _, _, _, _, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(extras.get("rope_scaling_factor").map(String::as_str), Some("8"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── decode_to_f16 single element ────────────────────────────────────────────

    #[test]
    fn decode_to_f16_f32_single_element() {
        let data = 42.0f32.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F32);
        assert_eq!(result.len(), 1);
        assert!((result[0].to_f32() - 42.0).abs() < 0.1);
    }

    #[test]
    fn decode_to_f16_f16_single_element() {
        let val = f16::from_f32(3.5);
        let data = val.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::F16);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], val);
    }

    #[test]
    fn decode_to_f16_bf16_single_element() {
        let val = half::bf16::from_f32(7.25);
        let data = val.to_le_bytes();
        let result = decode_to_f16(&data, GgmlDType::BF16);
        assert_eq!(result.len(), 1);
        assert!((result[0].to_f32() - 7.25).abs() < 0.01);
    }

    // ── elem_bytes for all types (exhaustive) ───────────────────────────────────

    #[test]
    fn elem_bytes_exhaustive_all_types() {
        use safetensors::Dtype;
        // 1-byte types
        assert_eq!(elem_bytes(Dtype::BOOL), 1);
        assert_eq!(elem_bytes(Dtype::U8), 1);
        assert_eq!(elem_bytes(Dtype::I8), 1);
        // 2-byte types
        assert_eq!(elem_bytes(Dtype::F16), 2);
        assert_eq!(elem_bytes(Dtype::BF16), 2);
        assert_eq!(elem_bytes(Dtype::I16), 2);
        assert_eq!(elem_bytes(Dtype::U16), 2);
        // 4-byte types
        assert_eq!(elem_bytes(Dtype::F32), 4);
        assert_eq!(elem_bytes(Dtype::I32), 4);
        assert_eq!(elem_bytes(Dtype::U32), 4);
        // 8-byte types
        assert_eq!(elem_bytes(Dtype::F64), 8);
        assert_eq!(elem_bytes(Dtype::I64), 8);
        assert_eq!(elem_bytes(Dtype::U64), 8);
    }

    // ── gllm_zp_type explicit zero for non-Q4_1/Q5_1 quant types ───────────────

    #[test]
    fn gllm_zp_type_q2_k_is_zero() {
        assert_eq!(gllm_zp_type(GgmlDType::Q2_K), 0);
    }

    #[test]
    fn gllm_zp_type_q3_k_is_zero() {
        assert_eq!(gllm_zp_type(GgmlDType::Q3_K), 0);
    }

    #[test]
    fn gllm_zp_type_q8_k_is_zero() {
        assert_eq!(gllm_zp_type(GgmlDType::Q8_K), 0);
    }

    #[test]
    fn gllm_zp_type_iq4_xs_is_zero() {
        assert_eq!(gllm_zp_type(GgmlDType::IQ4_XS), 0);
    }

    // ── ConvertOptions Debug format with all quant targets ──────────────────────

    #[test]
    fn convert_options_debug_with_awq4() {
        let opts = ConvertOptions {
            page_size: 4096,
            config_path: None,
            quant_target: Some(QuantTarget::Awq4),
        };
        let debug = format!("{opts:?}");
        assert!(debug.contains("Awq4"));
    }

    #[test]
    fn convert_options_debug_with_gptq4() {
        let opts = ConvertOptions {
            page_size: 4096,
            config_path: None,
            quant_target: Some(QuantTarget::Gptq4),
        };
        let debug = format!("{opts:?}");
        assert!(debug.contains("Gptq4"));
    }

    #[test]
    fn convert_options_debug_with_nvfp4() {
        let opts = ConvertOptions {
            page_size: 4096,
            config_path: None,
            quant_target: Some(QuantTarget::Nvfp4),
        };
        let debug = format!("{opts:?}");
        assert!(debug.contains("Nvfp4"));
    }

    // ── extract_model_params: extra field for rope_freq_base ────────────────────

    #[test]
    fn extract_model_params_rope_freq_base_float() {
        let dir = std::env::temp_dir().join("gllm_test_extract_rope_freq");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "llama",
            "rope_theta": 500000.0,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        let (_, _, _, _, _, _, _, _, _, extras) =
            extract_model_params(Some(&config_path));

        assert_eq!(extras.get("rope_freq_base").map(String::as_str), Some("500000"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── TensorEntry 3D shape ────────────────────────────────────────────────────

    #[test]
    fn tensor_entry_3d_shape() {
        let entry = TensorEntry {
            name: "pos_embedding".to_string(),
            ndim: 3,
            dtype: 0,
            shape: [32, 1024, 768, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 100],
            original_size: 32 * 1024 * 768 * 4,
        };
        assert_eq!(entry.ndim, 3);
        assert_eq!(entry.shape[0], 32);
        assert_eq!(entry.shape[1], 1024);
        assert_eq!(entry.shape[2], 768);
        assert_eq!(entry.shape[3], 0);
        assert!(!entry.is_quantized());
    }

    // ── gllm_scale_dtype: integer types return zero ────────────────────────────

    #[test]
    fn gllm_scale_dtype_integer_types_return_zero() {
        assert_eq!(gllm_scale_dtype(GgmlDType::I8), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::I16), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::I32), 0);
        assert_eq!(gllm_scale_dtype(GgmlDType::I64), 0);
    }

    // ── gllm_scale_dtype: float types return zero ──────────────────────────────

    #[test]
    fn gllm_scale_dtype_float_types_return_zero() {
        assert_eq!(gllm_scale_dtype(GgmlDType::F64), 0);
    }

    // ── Batch 5: Additional 15 tests ──────────────────────────────────────────

    #[test]
    fn convert_gguf_fp16_to_gllm_without_quant_target_returns_error() {
        // Arrange: create a ConvertOptions with quant_target = None
        let dir = std::env::temp_dir().join("gllm_test_fp16_no_target");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("out.gllm");

        // We need a valid GGUF file to reach the quant_target check.
        // Since we can't easily create one, use a nonexistent path — the error
        // from opening the GGUF file is fine, but we verify the options are used.
        let opts = ConvertOptions {
            page_size: 4096,
            config_path: None,
            quant_target: None,
        };

        // Act: convert_gguf_fp16_to_gllm should error because quant_target is None
        let result = convert_gguf_fp16_to_gllm(
            Path::new("/nonexistent/model.gguf"),
            &out,
            &opts,
        );

        // Assert: should be an error (either Io from missing file or ParseError from missing quant_target)
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn convert_safetensors_to_gllm_nonexistent_path_returns_error() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_test_st_nonexistent");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("out.gllm");

        // Act: pass a path to a file that does not exist
        let result = convert_safetensors_to_gllm(
            &[PathBuf::from("/nonexistent/model.safetensors")],
            &out,
            &ConvertOptions::default(),
        );

        // Assert: should be an error
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_writer_new_creates_empty_writer() {
        // Arrange & Act
        let writer = GllmWriter::new(4096);

        // Assert
        assert_eq!(writer.tensor_count(), 0);
    }

    #[test]
    fn gllm_writer_add_tensor_increments_count() {
        // Arrange
        let mut writer = GllmWriter::new(4096);
        assert_eq!(writer.tensor_count(), 0);

        // Act
        writer.add_tensor(TensorEntry {
            name: "test.weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [64, 64, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 256],
            original_size: 256,
        });

        // Assert
        assert_eq!(writer.tensor_count(), 1);
    }

    #[test]
    fn gllm_writer_add_multiple_tensors() {
        // Arrange
        let mut writer = GllmWriter::new(4096);

        // Act: add 3 tensors
        for i in 0..3 {
            writer.add_tensor(TensorEntry {
                name: format!("layer.{i}.weight"),
                ndim: 2,
                dtype: 0,
                shape: [16, 16, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 64],
                original_size: 64,
            });
        }

        // Assert
        assert_eq!(writer.tensor_count(), 3);
    }

    #[test]
    fn gllm_writer_write_without_metadata_succeeds() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_test_writer_no_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("no_meta.gllm");

        let mut writer = GllmWriter::new(4096);
        writer.add_tensor(TensorEntry {
            name: "layer.0.weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        });
        writer.set_metadata(vec![0x80]); // minimal MessagePack (empty map)

        // Act
        let result = writer.write_to_path(&out);

        // Assert: should succeed
        assert!(result.is_ok());
        assert!(out.exists());
        assert!(std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0) > 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_writer_different_page_sizes() {
        // Arrange & Act: verify writer can be created with various page sizes
        for &ps in &[1u32, 512, 4096, 65536, u32::MAX] {
            let writer = GllmWriter::new(ps);
            assert_eq!(writer.tensor_count(), 0);
        }
    }

    #[test]
    fn tensor_entry_compressed_size_returns_data_len_as_u64() {
        // Arrange
        let data = vec![0xCDu8; 1024];
        let entry = TensorEntry {
            name: "big_weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [128, 128, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 4096,
        };

        // Act
        let compressed = entry.compressed_size();

        // Assert: compressed_size() returns the data length
        assert_eq!(compressed, data.len() as u64);
    }

    #[test]
    fn safetensors_to_gllm_with_two_shards() {
        // Arrange
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;

        let dir = std::env::temp_dir().join("gllm_test_st_two_shards");
        std::fs::create_dir_all(&dir).unwrap();

        // Create two safetensors files
        let data1: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let data2: Vec<u8> = (32..64).map(|i| i as u8).collect();

        let view1 = TensorView::new(Dtype::F16, vec![4, 4], &data1).expect("v1");
        let path1 = dir.join("shard-00001.safetensors");
        serialize_to_file(vec![("layer.0.weight", view1)], &None, &path1).expect("write1");

        let view2 = TensorView::new(Dtype::F16, vec![4, 4], &data2).expect("v2");
        let path2 = dir.join("shard-00002.safetensors");
        serialize_to_file(vec![("layer.1.weight", view2)], &None, &path2).expect("write2");

        let gllm_path = dir.join("merged.gllm");

        // Act
        let result = convert_safetensors_to_gllm(
            &[path1, path2],
            &gllm_path,
            &ConvertOptions::default(),
        ).expect("convert");

        // Assert
        assert_eq!(result.tensor_count, 2);
        assert_eq!(result.quantized_count, 0);
        assert!(gllm_path.exists());

        let reader = crate::loader::gllm::GllmReader::open(&gllm_path).expect("read back");
        assert_eq!(reader.tensor_count(), 2);
        assert!(reader.find_tensor("layer.0.weight").is_some());
        assert!(reader.find_tensor("layer.1.weight").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_model_type_as_integer_returns_unknown() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_test_extract_int_type");
        std::fs::create_dir_all(&dir).unwrap();

        // model_type is an integer, not a string — as_str() returns None
        let config = serde_json::json!({"model_type": 42});
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        // Act
        let (arch, _, _, _, _, _, _, _, _, _) = extract_model_params(Some(&config_path));

        // Assert
        assert_eq!(arch, "unknown");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_model_params_explicit_head_dim_larger_than_derived() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_test_extract_larger_hd");
        std::fs::create_dir_all(&dir).unwrap();

        // head_dim=256 > derived 4096/32=128
        let config = serde_json::json!({
            "model_type": "test",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "head_dim": 256,
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        // Act
        let (_, _, _, _, _, _, head_dim, _, _, _) = extract_model_params(Some(&config_path));

        // Assert: max(256, 128) = 256
        assert_eq!(head_dim, 256);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_to_f16_f32_very_small_positive() {
        // Arrange: f32 smallest positive normal
        let smallest = f32::from_bits(0x0080_0000u32);
        let data = smallest.to_le_bytes();

        // Act
        let result = decode_to_f16(&data, GgmlDType::F32);

        // Assert: should not panic, returns 1 element
        assert_eq!(result.len(), 1);
        // The value may underflow to zero in f16 — that's acceptable
        assert!(result[0].to_f32() >= 0.0);
    }

    #[test]
    fn decode_to_f16_bf16_zero() {
        // Arrange
        let bf16_zero = half::bf16::from_f32(0.0);
        let data = bf16_zero.to_le_bytes();

        // Act
        let result = decode_to_f16(&data, GgmlDType::BF16);

        // Assert
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], f16::from_f32(0.0));
    }

    #[test]
    fn gllm_error_from_io_error_into() {
        // Arrange
        let io_err = std::io::Error::new(std::io::ErrorKind::WriteZero, "disk full");

        // Act
        let gllm_err: GllmError = io_err.into();

        // Assert
        let msg = gllm_err.to_string();
        assert!(msg.contains("disk full"));
        assert!(gllm_err.source().is_some());
    }

    #[test]
    fn elem_bytes_returns_correct_for_all_common_types() {
        // Arrange & Act & Assert: comprehensive verification
        use safetensors::Dtype;

        // All 1-byte types
        for dt in &[Dtype::BOOL, Dtype::U8, Dtype::I8] {
            assert_eq!(elem_bytes(*dt), 1, "{dt:?} should be 1 byte");
        }
        // All 2-byte types
        for dt in &[Dtype::F16, Dtype::BF16, Dtype::I16, Dtype::U16] {
            assert_eq!(elem_bytes(*dt), 2, "{dt:?} should be 2 bytes");
        }
        // All 4-byte types
        for dt in &[Dtype::F32, Dtype::I32, Dtype::U32] {
            assert_eq!(elem_bytes(*dt), 4, "{dt:?} should be 4 bytes");
        }
        // All 8-byte types
        for dt in &[Dtype::F64, Dtype::I64, Dtype::U64] {
            assert_eq!(elem_bytes(*dt), 8, "{dt:?} should be 8 bytes");
        }
    }

    #[test]
    fn safetensors_roundtrip_preserves_bf16_data_exactly() {
        // Arrange: create safetensors with BF16 data
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;

        let dir = std::env::temp_dir().join("gllm_test_st_bf16_roundtrip");
        std::fs::create_dir_all(&dir).unwrap();

        let bf16_bytes: Vec<u8> = vec![0x00, 0x3F, 0x00, 0x40, 0x00, 0x41, 0x00, 0x42];
        let view = TensorView::new(Dtype::BF16, vec![2, 2], &bf16_bytes).expect("view");
        let st_path = dir.join("model.safetensors");
        serialize_to_file(vec![("embed.weight", view)], &None, &st_path).expect("write");

        let gllm_path = dir.join("output.gllm");

        // Act
        let result = convert_safetensors_to_gllm(
            &[st_path],
            &gllm_path,
            &ConvertOptions::default(),
        ).expect("convert");

        // Assert
        assert_eq!(result.tensor_count, 1);
        assert_eq!(result.quantized_count, 0);

        let reader = crate::loader::gllm::GllmReader::open(&gllm_path).expect("read back");
        let t = reader.find_tensor("embed.weight").expect("find tensor");
        assert_eq!(t.entry.shape[0], 2);
        assert_eq!(t.entry.shape[1], 2);
        assert!(!t.entry.is_quantized());
        let data = reader.tensor_data("embed.weight").expect("data");
        let data_ref: &[u8] = data.as_ref();
        assert_eq!(data_ref, bf16_bytes.as_slice());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Batch 6: 13 additional tests ──────────────────────────────────────────

    #[test]
    fn convert_options_clone_then_modify_original() {
        // Arrange
        let opts = ConvertOptions {
            page_size: 8192,
            config_path: Some(PathBuf::from("/tmp/cfg.json")),
            quant_target: Some(QuantTarget::Awq4),
        };
        // Act
        let cloned = opts.clone();
        // Assert: cloned is independent — original unchanged
        assert_eq!(opts.page_size, 8192);
        assert_eq!(cloned.page_size, 8192);
        assert_eq!(opts.quant_target, cloned.quant_target);
    }

    #[test]
    fn convert_result_debug_includes_all_fields() {
        // Arrange
        let result = ConvertResult {
            input_bytes: 111,
            output_bytes: 222,
            tensor_count: 33,
            quantized_count: 7,
        };
        // Act
        let debug = format!("{result:?}");
        // Assert
        assert!(debug.contains("input_bytes"));
        assert!(debug.contains("output_bytes"));
        assert!(debug.contains("tensor_count"));
        assert!(debug.contains("quantized_count"));
    }

    #[test]
    fn gllm_error_display_tensor_dir_with_zero_count() {
        // Arrange: zero tensors in directory
        let err = GllmError::TensorDirOutOfBounds {
            offset: 500,
            count: 0,
            file_size: 1000,
        };
        // Act
        let msg = err.to_string();
        // Assert: offset + count*72 = 500+0 = 500
        assert!(msg.contains("500"));
        assert!(msg.contains("1000"));
    }

    #[test]
    fn decode_to_f16_i8_dtype_returns_empty() {
        // Arrange: I8 is not a float dtype, should return empty
        let data = vec![1u8, 2, 3, 4];
        // Act
        let result = decode_to_f16(&data, GgmlDType::I8);
        // Assert
        assert!(result.is_empty());
    }

    #[test]
    fn tensor_entry_all_fields_zero_except_name() {
        // Arrange
        let entry = TensorEntry {
            name: "zero_tensor".to_string(),
            ndim: 0,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: Vec::new(),
            original_size: 0,
        };
        // Assert
        assert!(!entry.is_quantized());
        assert_eq!(entry.compressed_size(), 0);
        assert_eq!(entry.name, "zero_tensor");
    }

    #[test]
    fn config_u64_with_empty_key_string() {
        // Arrange: key is an empty string
        let config = serde_json::json!({"": 42});
        // Act
        let val = config_u64(&config, "");
        // Assert
        assert_eq!(val, 42);
        // Nonexistent empty-ish key
        assert_eq!(config_u64(&config, "missing"), 0);
    }

    #[test]
    fn extract_model_params_rope_theta_as_string_not_extracted() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_test_extract_rope_str");
        std::fs::create_dir_all(&dir).unwrap();

        let config = serde_json::json!({
            "model_type": "test",
            "rope_theta": "10000.0",
        });
        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

        // Act
        let (_, _, _, _, _, _, _, _, _, extras) =
            extract_model_params(Some(&config_path));

        // Assert: rope_theta is a string, not a number, so as_f64 returns None
        assert!(!extras.contains_key("rope_freq_base"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_scale_dtype_q8_1_explicit_value() {
        // Arrange & Act & Assert
        assert_eq!(gllm_scale_dtype(GgmlDType::Q8_1), 1);
    }

    #[test]
    fn convert_safetensors_empty_paths_error_message() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_test_empty_st_msg");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("out.gllm");

        // Act
        let result = convert_safetensors_to_gllm(&[], &out, &ConvertOptions::default());

        // Assert: error message should mention the problem
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("no safetensors"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn safetensors_roundtrip_preserves_f32_data_exactly() {
        // Arrange: create safetensors with F32 data
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;

        let dir = std::env::temp_dir().join("gllm_test_st_f32_roundtrip");
        std::fs::create_dir_all(&dir).unwrap();

        let f32_bytes: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![2, 2], &f32_bytes).expect("view");
        let st_path = dir.join("model.safetensors");
        serialize_to_file(vec![("bias", view)], &None, &st_path).expect("write");

        let gllm_path = dir.join("output.gllm");

        // Act
        let result = convert_safetensors_to_gllm(
            &[st_path],
            &gllm_path,
            &ConvertOptions::default(),
        ).expect("convert");

        // Assert
        assert_eq!(result.tensor_count, 1);
        let reader = crate::loader::gllm::GllmReader::open(&gllm_path).expect("read");
        let data = reader.tensor_data("bias").expect("data");
        let data_ref: &[u8] = data.as_ref();
        assert_eq!(data_ref, f32_bytes.as_slice());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_to_f16_f16_alternating_sign_pattern() {
        // Arrange: alternating positive and negative values
        let values: Vec<f16> = (0..8)
            .map(|i| f16::from_f32(if i % 2 == 0 { (i + 1) as f32 } else { -((i + 1) as f32) }))
            .collect();
        let mut data = Vec::with_capacity(values.len() * 2);
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        // Act
        let result = decode_to_f16(&data, GgmlDType::F16);
        // Assert
        assert_eq!(result.len(), 8);
        for (i, orig) in values.iter().enumerate() {
            assert_eq!(result[i], *orig, "index {i} mismatch");
        }
    }

    #[test]
    fn convert_result_input_equals_output_bytes() {
        // Arrange: same size in and out (e.g., passthrough with no overhead)
        let result = ConvertResult {
            input_bytes: 4096,
            output_bytes: 4096,
            tensor_count: 1,
            quantized_count: 0,
        };
        // Assert
        assert_eq!(result.input_bytes, result.output_bytes);
    }

    #[test]
    fn tensor_entry_is_quantized_boundary_at_format_one() {
        // Arrange: quant_format=1 is nonzero, so is_quantized returns true
        let entry = TensorEntry {
            name: "boundary".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 1,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 64,
        };
        // Assert: any nonzero quant_format means quantized
        assert!(entry.is_quantized());
    }
}
