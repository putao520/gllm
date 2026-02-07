//! Model config loader (config.json).

use std::path::Path;

use serde_json::Value;
use thiserror::Error;

use crate::loader::Loader;
use crate::manifest::ModelManifest;

#[derive(Debug, Error)]
pub enum ModelConfigError {
    #[error("config.json not found in model files")]
    MissingConfig,
    #[error("invalid or incomplete config.json: {0}")]
    InvalidConfig(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type ModelConfigResult<T> = std::result::Result<T, ModelConfigError>;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub rope_interleaved: bool,
    pub kv_cache_block_size: usize,
    pub head_dim: usize,
    pub dtype_size: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

impl ModelConfig {
    pub fn from_loader(manifest: &ModelManifest, loader: &mut Loader) -> ModelConfigResult<Self> {
        // 先提取路径，避免借用冲突
        let path = loader
            .config_path()
            .ok_or(ModelConfigError::MissingConfig)?
            .to_path_buf();
        Self::from_path_with_loader(manifest, &path, loader)
    }

    pub fn from_path(manifest: &ModelManifest, path: &Path) -> ModelConfigResult<Self> {
        let bytes = std::fs::read(path)?;
        let value: Value = serde_json::from_slice(&bytes)?;
        Self::from_value(manifest, &value, None)
    }

    /// Ω1: 从实际权重中检测 dtype 大小
    ///
    /// 优先使用权重文件中的实际 dtype，而非 config.json 中的声明
    pub fn from_path_with_loader(
        manifest: &ModelManifest,
        path: &Path,
        loader: &mut Loader,
    ) -> ModelConfigResult<Self> {
        // 先克隆需要的数据，避免借用冲突
        let manifest_clone = manifest.clone();
        let bytes = std::fs::read(path)?;
        let value: Value = serde_json::from_slice(&bytes)?;

        // 检测权重 dtype（需要可变借用）
        let weight_dtype_size = loader.detect_weight_dtype_size().unwrap_or(None);

        Self::from_value(&manifest_clone, &value, weight_dtype_size)
    }

    pub fn from_value(
        manifest: &ModelManifest,
        value: &Value,
        weight_dtype_size: Option<usize>,
    ) -> ModelConfigResult<Self> {
        let hidden_size = require_usize(value, &["hidden_size", "n_embd", "d_model"])?;
        let num_attention_heads =
            require_usize(value, &["num_attention_heads", "n_head", "num_heads"])?;
        let num_key_value_heads =
            find_usize(value, &["num_key_value_heads", "num_kv_heads", "n_kv_head"])
                .unwrap_or(num_attention_heads);
        let num_hidden_layers =
            require_usize(value, &["num_hidden_layers", "n_layer", "num_layers"])?;
        let vocab_size = require_usize(value, &["vocab_size"])?;

        let max_position_embeddings = find_usize(
            value,
            &[
                "max_position_embeddings",
                "max_seq_len",
                "max_sequence_length",
                "seq_length",
                "n_positions",
            ],
        )
        .unwrap_or(0);

        let max_position_embeddings = manifest
            .max_context_override
            .unwrap_or(max_position_embeddings);

        if max_position_embeddings == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "missing max_position_embeddings".to_string(),
            ));
        }

        // Ω1: rope_theta 必须从模型配置或 manifest 中读取，不再使用硬编码默认值
        // 对于 Embedding 模型（没有 attention），rope_theta 可以为 0
        let rope_theta = if let Some(override_value) = manifest.rope_base_override {
            override_value
        } else {
            find_f32(value, &["rope_theta", "rope_base", "rope_base_value"]).unwrap_or(0.0)
        };

        // RoPE 缩放系数优先读取 metadata；缺失时保持无缩放 (1.0)。
        let rope_scale = rope_scaling_factor(value)
            .or_else(|| find_f32(value, &["rope_scale", "rope_factor"]))
            .unwrap_or(1.0);
        if rope_scale <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scale must be positive".to_string(),
            ));
        }

        let rope_interleaved = find_bool(
            value,
            &[
                "rope_interleaved",
                "rotary_interleaved",
                "interleaved_rotary",
            ],
        )
        .unwrap_or(false);

        // Ω1: head_dim 从 config.json 读取，或使用标准公式计算
        // 注意：Embedding 模型可能没有 num_attention_heads，此时 head_dim = 0
        let head_dim = if num_attention_heads > 0 {
            find_usize(value, &["head_dim", "kv_channels"])
                .unwrap_or(hidden_size / num_attention_heads)
        } else {
            // Embedding 模型没有 attention heads
            0
        };

        if num_attention_heads > 0 && head_dim == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "invalid head_dim for non-embedding model".to_string(),
            ));
        }

        let kv_cache_block_size = find_usize(
            value,
            &["kv_cache_block_size", "kv_block_size", "page_size"],
        )
        .unwrap_or_else(|| head_dim.max(num_key_value_heads));
        if kv_cache_block_size == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "invalid kv_cache_block_size".to_string(),
            ));
        }

        // Ω1: dtype 大小优先从实际权重中读取，而非 config.json
        // 如果权重检测失败，再尝试从配置文件读取
        let dtype_size = match weight_dtype_size {
            Some(size) => size,
            None => dtype_size_from_config(value).ok_or_else(|| {
                ModelConfigError::InvalidConfig(
                    "无法确定模型的 dtype 大小，config.json 中缺少 torch_dtype 字段".to_string(),
                )
            })?,
        };

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            rope_scale,
            rope_interleaved,
            kv_cache_block_size,
            head_dim,
            dtype_size,
            bos_token_id: find_u32(value, &["bos_token_id"]),
            eos_token_id: find_u32(value, &["eos_token_id"]),
            pad_token_id: find_u32(value, &["pad_token_id"]),
        })
    }
}

fn require_usize(value: &Value, keys: &[&str]) -> ModelConfigResult<usize> {
    find_usize(value, keys).ok_or_else(|| ModelConfigError::InvalidConfig(keys[0].to_string()))
}

fn find_usize(value: &Value, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
    })
}

fn find_u32(value: &Value, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(|v| v.as_u64())
            .and_then(|v| u32::try_from(v).ok())
    })
}

fn find_f32(value: &Value, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(|v| v.as_f64().map(|num| num as f32))
    })
}

fn find_bool(value: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|v| {
            v.as_bool().or_else(|| {
                v.as_u64().and_then(|num| match num {
                    0 => Some(false),
                    1 => Some(true),
                    _ => None,
                })
            })
        })
    })
}

fn rope_scaling_factor(value: &Value) -> Option<f32> {
    let scaling = value.get("rope_scaling")?;
    scaling
        .get("factor")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
}

fn dtype_size_from_config(value: &Value) -> Option<usize> {
    let dtype = value.get("torch_dtype").or_else(|| value.get("dtype"))?;
    let dtype = dtype.as_str()?.to_ascii_lowercase();
    if dtype.contains("float32") || dtype.contains("fp32") {
        Some(4)
    } else if dtype.contains("float16")
        || dtype.contains("fp16")
        || dtype.contains("bfloat16")
        || dtype.contains("bf16")
    {
        Some(2)
    } else {
        None
    }
}
