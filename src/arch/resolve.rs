//! 占位符解析和配置推导 (REQ-ARCH-005)
//!
//! 从 GGUF metadata 或 SafeTensors 张量形状推导配置值。

use std::collections::HashMap;

use crate::loader::gguf::GgufReader;
use crate::loader::TensorProvider;
use crate::manifest::TensorRole;

use super::template::{ArchTemplate, ConfigValue};

/// 解析后的配置
#[derive(Debug, Clone, Default)]
pub struct ResolvedConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub dtype: String,
    /// 额外的自定义配置
    pub extra: HashMap<String, i64>,
}

impl ResolvedConfig {
    /// 获取配置值（整数）
    pub fn get_int(&self, key: &str) -> Option<i64> {
        match key {
            "num_hidden_layers" => Some(self.num_hidden_layers as i64),
            "hidden_size" => Some(self.hidden_size as i64),
            "num_attention_heads" => Some(self.num_attention_heads as i64),
            "num_key_value_heads" => Some(self.num_key_value_heads as i64),
            "head_dim" => Some(self.head_dim as i64),
            "intermediate_size" => self.intermediate_size.map(|v| v as i64),
            "vocab_size" => Some(self.vocab_size as i64),
            _ => self.extra.get(key).copied(),
        }
    }

    /// 获取配置值（浮点）
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match key {
            "rope_theta" => Some(self.rope_theta),
            _ => None,
        }
    }

    /// 获取配置值（字符串）
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match key {
            "dtype" => Some(&self.dtype),
            _ => None,
        }
    }
}

/// 配置推导错误
#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("Missing required config: {0}")]
    MissingConfig(String),
    #[error("Failed to derive {key} from tensors: {reason}")]
    DerivationFailed { key: String, reason: String },
    #[error("Inconsistent config: {0}")]
    Inconsistent(String),
}

/// 从模板和数据源解析配置
pub fn resolve_config<P: TensorProvider>(
    template: &ArchTemplate,
    provider: &P,
    gguf: Option<&GgufReader>,
) -> Result<ResolvedConfig, ResolveError> {
    let mut config = ResolvedConfig::default();

    // 优先从 GGUF metadata 获取
    if let Some(reader) = gguf {
        resolve_from_gguf(&mut config, reader)?;
    }

    // 从张量形状补充/验证
    resolve_from_tensors(&mut config, provider)?;

    // 处理模板中的配置覆盖
    for (key, value) in &template.config {
        if !value.is_placeholder() {
            // 直接值覆盖
            match value {
                ConfigValue::Direct(v) => {
                    config.extra.insert(key.clone(), *v);
                }
                ConfigValue::Float(v) => {
                    if key == "rope_theta" {
                        config.rope_theta = *v;
                    }
                }
                ConfigValue::String(_) => {}
            }
        }
    }

    // 计算派生值
    if config.head_dim == 0 && config.hidden_size > 0 && config.num_attention_heads > 0 {
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }

    // 验证必需字段
    validate_config(&config)?;

    Ok(config)
}

/// 从 GGUF metadata 解析
fn resolve_from_gguf(config: &mut ResolvedConfig, reader: &GgufReader) -> Result<(), ResolveError> {
    // 尝试常见的 GGUF metadata 键
    // llama.* 系列（Qwen, Llama, Mistral 等通用）
    if let Some(v) = reader.get_metadata_u64("llama.block_count") {
        config.num_hidden_layers = v as usize;
    }
    if let Some(v) = reader.get_metadata_u64("llama.embedding_length") {
        config.hidden_size = v as usize;
    }
    if let Some(v) = reader.get_metadata_u64("llama.attention.head_count") {
        config.num_attention_heads = v as usize;
    }
    if let Some(v) = reader.get_metadata_u64("llama.attention.head_count_kv") {
        config.num_key_value_heads = v as usize;
    }
    if let Some(v) = reader.get_metadata_u64("llama.feed_forward_length") {
        config.intermediate_size = Some(v as usize);
    }
    if let Some(v) = reader.get_metadata_f32("llama.rope.freq_base") {
        config.rope_theta = v as f64;
    }

    // qwen2.* 系列
    if config.num_hidden_layers == 0 {
        if let Some(v) = reader.get_metadata_u64("qwen2.block_count") {
            config.num_hidden_layers = v as usize;
        }
    }

    Ok(())
}

/// 从张量形状推导配置
fn resolve_from_tensors<P: TensorProvider>(
    config: &mut ResolvedConfig,
    provider: &P,
) -> Result<(), ResolveError> {
    // 收集张量信息
    let mut max_layer_idx = 0usize;
    let mut embed_shape: Option<Vec<usize>> = None;
    let mut q_proj_shape: Option<Vec<usize>> = None;
    let mut k_proj_shape: Option<Vec<usize>> = None;
    let mut detected_dtype = None;

    for tensor in provider.iter_tensors() {
        let name = &tensor.name;
        let shape = &tensor.shape;

        // 检测层数
        if let Some((_, Some(layer_idx))) = crate::loader::match_tensor_role(name) {
            max_layer_idx = max_layer_idx.max(layer_idx + 1);
        }

        // 匹配角色
        if let Some((role, _)) = crate::loader::match_tensor_role(name) {
            match role {
                TensorRole::Embedding => {
                    embed_shape = Some(shape.clone());
                    if detected_dtype.is_none() {
                        detected_dtype = Some(tensor.dtype);
                    }
                }
                TensorRole::AttentionQuery => {
                    q_proj_shape = Some(shape.clone());
                }
                TensorRole::AttentionKey => {
                    k_proj_shape = Some(shape.clone());
                }
                _ => {}
            }
        }
    }

    // 从 embed 推导 vocab_size 和 hidden_size
    if let Some(shape) = embed_shape {
        if shape.len() >= 2 {
            if config.vocab_size == 0 {
                config.vocab_size = shape[0];
            }
            if config.hidden_size == 0 {
                config.hidden_size = shape[1];
            }
        }
    }

    // 从 q_proj 推导 num_attention_heads
    if let Some(shape) = q_proj_shape {
        if shape.len() >= 2 && config.hidden_size > 0 {
            let q_out_dim = shape[0];
            // q_proj: [num_heads * head_dim, hidden_size]
            if config.num_attention_heads == 0 && config.head_dim > 0 {
                config.num_attention_heads = q_out_dim / config.head_dim;
            }
        }
    }

    // 从 k_proj 推导 num_key_value_heads
    if let Some(shape) = k_proj_shape {
        if shape.len() >= 2 && config.head_dim > 0 {
            let k_out_dim = shape[0];
            if config.num_key_value_heads == 0 {
                config.num_key_value_heads = k_out_dim / config.head_dim;
            }
        }
    }

    // 层数
    if config.num_hidden_layers == 0 && max_layer_idx > 0 {
        config.num_hidden_layers = max_layer_idx;
    }

    // dtype
    if let Some(dtype) = detected_dtype {
        config.dtype = match dtype {
            safetensors::Dtype::F32 => "f32".to_string(),
            safetensors::Dtype::F16 => "f16".to_string(),
            safetensors::Dtype::BF16 => "bf16".to_string(),
            _ => "f32".to_string(),
        };
    }

    Ok(())
}

/// 验证配置完整性
fn validate_config(config: &ResolvedConfig) -> Result<(), ResolveError> {
    if config.num_hidden_layers == 0 {
        return Err(ResolveError::MissingConfig(
            "num_hidden_layers".to_string(),
        ));
    }
    if config.hidden_size == 0 {
        return Err(ResolveError::MissingConfig("hidden_size".to_string()));
    }
    if config.num_attention_heads == 0 {
        return Err(ResolveError::MissingConfig(
            "num_attention_heads".to_string(),
        ));
    }
    if config.vocab_size == 0 {
        return Err(ResolveError::MissingConfig("vocab_size".to_string()));
    }
    Ok(())
}

/// 替换字符串中的占位符
pub fn substitute_placeholders(template: &str, config: &ResolvedConfig) -> String {
    let mut result = template.to_string();

    // 整数占位符
    for (key, getter) in [
        ("num_hidden_layers", config.num_hidden_layers),
        ("hidden_size", config.hidden_size),
        ("num_attention_heads", config.num_attention_heads),
        ("num_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("num_kv_heads", config.num_key_value_heads),
        ("head_dim", config.head_dim),
        ("vocab_size", config.vocab_size),
    ] {
        let placeholder = format!("${{{}}}", key);
        result = result.replace(&placeholder, &getter.to_string());
    }

    // intermediate_size (可选)
    if let Some(v) = config.intermediate_size {
        result = result.replace("${intermediate_size}", &v.to_string());
    }

    // 浮点占位符
    result = result.replace("${rope_theta}", &config.rope_theta.to_string());

    // 字符串占位符
    result = result.replace("${dtype}", &config.dtype);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_placeholders_works() {
        let config = ResolvedConfig {
            num_hidden_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: Some(11008),
            vocab_size: 32000,
            rope_theta: 10000.0,
            dtype: "f16".to_string(),
            extra: HashMap::new(),
        };

        let template = "layers: ${num_hidden_layers}, hidden: ${hidden_size}, dtype: ${dtype}";
        let result = substitute_placeholders(template, &config);
        assert_eq!(result, "layers: 32, hidden: 4096, dtype: f16");
    }

    #[test]
    fn config_get_methods() {
        let config = ResolvedConfig {
            num_hidden_layers: 24,
            hidden_size: 2048,
            rope_theta: 500000.0,
            dtype: "bf16".to_string(),
            ..Default::default()
        };

        assert_eq!(config.get_int("num_hidden_layers"), Some(24));
        assert_eq!(config.get_float("rope_theta"), Some(500000.0));
        assert_eq!(config.get_str("dtype"), Some("bf16"));
    }
}
