//! Model config loader (GGUF metadata first, config.json fallback).

use std::path::Path;

use serde_json::Value;
use thiserror::Error;

use crate::loader::{GgufLoader, Loader, WeightFormat};
use crate::manifest::ModelManifest;

#[derive(Debug, Error)]
pub enum ModelConfigError {
    #[error("config.json not found in model files")]
    MissingConfig,
    #[error("GGUF metadata unavailable and config.json not found: {0}")]
    MissingConfigAndMetadata(String),
    #[error("invalid or incomplete config.json: {0}")]
    InvalidConfig(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type ModelConfigResult<T> = std::result::Result<T, ModelConfigError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RopeScalingType {
    Linear,
    Dynamic,
    Yarn,
    LongRope,
    NtkAware,
    Llama3,
    Unknown(String),
}

impl RopeScalingType {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "linear" => Self::Linear,
            "dynamic" | "dynamic_ntk" | "ntk" => Self::Dynamic,
            "ntk_aware" | "ntk-aware" => Self::NtkAware,
            "yarn" => Self::Yarn,
            "longrope" | "long_rope" => Self::LongRope,
            "llama3" | "llama_3" => Self::Llama3,
            other => Self::Unknown(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RopeScalingConfig {
    pub scaling_type: Option<RopeScalingType>,
    pub rope_type: Option<String>,
    pub factor: Option<f32>,
    pub factors: Option<Vec<f32>>,
    pub base: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub ext_factor: Option<f32>,
    pub attn_factor: Option<f32>,
    pub beta_fast: Option<f32>,
    pub beta_slow: Option<f32>,
}

impl RopeScalingConfig {
    fn has_any_value(&self) -> bool {
        self.scaling_type.is_some()
            || self.rope_type.is_some()
            || self.factor.is_some()
            || self.factors.is_some()
            || self.base.is_some()
            || self.original_max_position_embeddings.is_some()
            || self.ext_factor.is_some()
            || self.attn_factor.is_some()
            || self.beta_fast.is_some()
            || self.beta_slow.is_some()
    }

    fn runtime_factor(&self) -> Option<f32> {
        self.factor.filter(|v| v.is_finite() && *v > 0.0)
    }
}

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
    pub rope_scaling: Option<RopeScalingConfig>,
    pub kv_cache_block_size: usize,
    pub head_dim: usize,
    pub dtype_size: usize,
    pub use_cache: Option<bool>,
    pub tie_word_embeddings: Option<bool>,
    pub attention_dropout: Option<f32>,
    pub hidden_act: Option<String>,
    pub layer_norm_epsilon: Option<f32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

impl ModelConfig {
    pub fn from_loader(manifest: &ModelManifest, loader: &mut Loader) -> ModelConfigResult<Self> {
        let mut gguf_metadata_error: Option<ModelConfigError> = None;

        if loader.weight_format() == WeightFormat::Gguf {
            match Self::from_gguf_loader(manifest, loader) {
                Ok(config) => return Ok(config),
                Err(err) => gguf_metadata_error = Some(err),
            }
        }

        if let Some(path) = loader.config_path().map(|path| path.to_path_buf()) {
            return Self::from_path_with_loader(manifest, &path, loader);
        }

        if let Some(err) = gguf_metadata_error {
            return Err(ModelConfigError::MissingConfigAndMetadata(err.to_string()));
        }

        Err(ModelConfigError::MissingConfig)
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

    fn from_gguf_loader(manifest: &ModelManifest, loader: &mut Loader) -> ModelConfigResult<Self> {
        let (
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rope_theta,
            rope_scale,
            rope_interleaved,
            rope_scaling,
            head_dim,
            attention_dropout,
            hidden_act,
            layer_norm_epsilon,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        ) = {
            let reader = loader.gguf_reader().map_err(|err| {
                ModelConfigError::InvalidConfig(format!("failed to load GGUF metadata: {err}"))
            })?;
            let arch = reader.architecture().map_err(|err| {
                ModelConfigError::InvalidConfig(format!(
                    "missing GGUF architecture metadata: {err}"
                ))
            })?;

            let vocab_size_key = format!("{arch}.vocab_size");
            let vocab_size =
                require_gguf_usize(reader.get_metadata_u64(&vocab_size_key), &vocab_size_key)?;
            let hidden_size = require_gguf_usize(reader.embedding_length(), "embedding_length")?;
            let num_hidden_layers = require_gguf_usize(reader.block_count(), "block_count")?;
            let num_attention_heads =
                require_gguf_usize(reader.head_count(), "attention.head_count")?;
            let num_key_value_heads =
                require_gguf_usize(reader.head_count_kv(), "attention.head_count_kv")?;
            let max_position_embeddings =
                require_gguf_usize(reader.context_length(), "context_length")?;
            let rope_scaling = rope_scaling_from_gguf(reader, arch)?;

            let rope_theta = rope_scaling
                .as_ref()
                .and_then(|cfg| cfg.base)
                .or_else(|| require_gguf_f32(reader.rope_freq_base(), "rope.freq_base").ok())
                .unwrap_or(0.0);

            let rope_scale = gguf_arch_f32(reader, arch, "rope.scale")
                .or_else(|| {
                    rope_scaling
                        .as_ref()
                        .and_then(RopeScalingConfig::runtime_factor)
                })
                .unwrap_or(1.0);
            if !rope_scale.is_finite() || rope_scale <= 0.0 {
                return Err(ModelConfigError::InvalidConfig(
                    "GGUF metadata field invalid: rope.scale".to_string(),
                ));
            }

            let rope_interleaved =
                gguf_arch_bool(reader, arch, "rope.interleaved").unwrap_or(false);

            let head_dim = if let Some(value) = gguf_arch_usize(reader, arch, "attention.head_dim")
            {
                value
            } else if let Some(value) = reader.rope_dimension_count() {
                usize::try_from(value).map_err(|_| {
                    ModelConfigError::InvalidConfig(
                        "GGUF metadata field overflow: rope.dimension_count".to_string(),
                    )
                })?
            } else {
                if num_attention_heads == 0 || hidden_size % num_attention_heads != 0 {
                    return Err(ModelConfigError::InvalidConfig(
                        "cannot derive head_dim from GGUF metadata".to_string(),
                    ));
                }
                hidden_size / num_attention_heads
            };

            if num_attention_heads > 0 && head_dim == 0 {
                return Err(ModelConfigError::InvalidConfig(
                    "invalid head_dim in GGUF metadata".to_string(),
                ));
            }

            let attention_dropout =
                gguf_arch_f32(reader, arch, "attention.dropout").filter(|v| v.is_finite());
            let hidden_act = gguf_arch_str(reader, arch, "feed_forward.activation")
                .or_else(|| gguf_arch_str(reader, arch, "hidden_act"))
                .map(|v| v.to_string());
            let layer_norm_epsilon = gguf_arch_f32(reader, arch, "layer_norm_epsilon")
                .or_else(|| gguf_arch_f32(reader, arch, "layer_norm_rms_epsilon"))
                .or_else(|| gguf_arch_f32(reader, arch, "attention.layer_norm_rms_epsilon"))
                .filter(|v| v.is_finite() && *v > 0.0);

            (
                vocab_size,
                hidden_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                max_position_embeddings,
                rope_theta,
                rope_scale,
                rope_interleaved,
                rope_scaling,
                head_dim,
                attention_dropout,
                hidden_act,
                layer_norm_epsilon,
                reader.bos_token_id(),
                reader.eos_token_id(),
                reader
                    .get_metadata_u64("tokenizer.ggml.padding_token_id")
                    .and_then(|v| u32::try_from(v).ok()),
            )
        };

        let max_position_embeddings = manifest
            .max_context_override
            .unwrap_or(max_position_embeddings);
        if max_position_embeddings == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "missing max_position_embeddings".to_string(),
            ));
        }

        let rope_theta = manifest.rope_base_override.unwrap_or(rope_theta);

        let kv_cache_block_size = head_dim.max(num_key_value_heads);
        if kv_cache_block_size == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "invalid kv_cache_block_size".to_string(),
            ));
        }

        let dtype_size = loader
            .detect_weight_dtype_size()
            .map_err(|err| {
                ModelConfigError::InvalidConfig(format!(
                    "failed to detect model dtype from weights: {err}"
                ))
            })?
            .ok_or_else(|| {
                ModelConfigError::InvalidConfig(
                    "unable to determine model dtype size from GGUF tensors".to_string(),
                )
            })?;

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
            rope_scaling,
            kv_cache_block_size,
            head_dim,
            dtype_size,
            use_cache: None,
            tie_word_embeddings: None,
            attention_dropout,
            hidden_act,
            layer_norm_epsilon,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    pub fn from_value(
        manifest: &ModelManifest,
        value: &Value,
        weight_dtype_size: Option<usize>,
    ) -> ModelConfigResult<Self> {
        let hidden_size = require_usize(value, &["hidden_size", "n_embd", "d_model"])?;
        let num_attention_heads =
            require_usize(value, &["num_attention_heads", "n_head", "num_heads"])?;
        let num_key_value_heads = find_usize(
            value,
            &[
                "num_key_value_heads",
                "num_kv_heads",
                "n_kv_head",
                "attention.num_key_value_heads",
            ],
        )
        .unwrap_or(num_attention_heads);
        let num_hidden_layers =
            require_usize(value, &["num_hidden_layers", "n_layer", "num_layers"])?;
        let vocab_size = require_usize(value, &["vocab_size"])?;
        let rope_scaling = rope_scaling_from_json(value)?;

        let max_position_embeddings = find_usize(
            value,
            &[
                "max_position_embeddings",
                "max_seq_len",
                "max_sequence_length",
                "seq_length",
                "n_positions",
                "rope_scaling.original_max_position_embeddings",
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
            rope_scaling
                .as_ref()
                .and_then(|cfg| cfg.base)
                .or_else(|| find_f32(value, &["rope_theta", "rope_base", "rope_base_value"]))
                .unwrap_or(0.0)
        };

        // RoPE 缩放系数优先读取完整 rope_scaling 对象；缺失时保持无缩放 (1.0)。
        let rope_scale = find_f32(value, &["rope_scale", "rope_factor", "rope.scaling.factor"])
            .or_else(|| {
                rope_scaling
                    .as_ref()
                    .and_then(RopeScalingConfig::runtime_factor)
            })
            .unwrap_or(1.0);
        if !rope_scale.is_finite() || rope_scale <= 0.0 {
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
                "rope_scaling.interleaved",
            ],
        )
        .unwrap_or(false);

        // Ω1: head_dim 从 config.json 读取，或使用标准公式计算
        // 注意：Embedding 模型可能没有 num_attention_heads，此时 head_dim = 0
        let head_dim = if num_attention_heads > 0 {
            find_usize(value, &["attention.head_dim", "head_dim", "kv_channels"])
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

        let attention_dropout = find_f32(value, &["attention_dropout", "attention.dropout"])
            .filter(|v| v.is_finite() && *v >= 0.0);
        let layer_norm_epsilon = find_f32(
            value,
            &[
                "layer_norm_epsilon",
                "layer_norm_eps",
                "rms_norm_eps",
                "norm_epsilon",
            ],
        )
        .filter(|v| v.is_finite() && *v > 0.0);

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
            rope_scaling,
            kv_cache_block_size,
            head_dim,
            dtype_size,
            use_cache: find_bool(value, &["use_cache"]),
            tie_word_embeddings: find_bool(value, &["tie_word_embeddings"]),
            attention_dropout,
            hidden_act: find_string(value, &["hidden_act", "hidden_activation"]),
            layer_norm_epsilon,
            bos_token_id: find_u32(value, &["bos_token_id"]),
            eos_token_id: find_u32(value, &["eos_token_id"]),
            pad_token_id: find_u32(value, &["pad_token_id"]),
        })
    }
}

fn require_usize(value: &Value, keys: &[&str]) -> ModelConfigResult<usize> {
    find_usize(value, keys).ok_or_else(|| ModelConfigError::InvalidConfig(keys[0].to_string()))
}

fn require_gguf_usize(value: Option<u64>, field: &str) -> ModelConfigResult<usize> {
    let value = value.ok_or_else(|| {
        ModelConfigError::InvalidConfig(format!("missing GGUF metadata field: {field}"))
    })?;
    usize::try_from(value).map_err(|_| {
        ModelConfigError::InvalidConfig(format!("GGUF metadata field overflow: {field}"))
    })
}

fn require_gguf_f32(value: Option<f32>, field: &str) -> ModelConfigResult<f32> {
    value.filter(|v| v.is_finite()).ok_or_else(|| {
        ModelConfigError::InvalidConfig(format!("missing GGUF metadata field: {field}"))
    })
}

fn find_value<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| value_at_path(value, key))
}

fn value_at_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for segment in path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

fn find_usize(value: &Value, keys: &[&str]) -> Option<usize> {
    find_value(value, keys)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

fn find_u32(value: &Value, keys: &[&str]) -> Option<u32> {
    find_value(value, keys)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
}

fn find_f32(value: &Value, keys: &[&str]) -> Option<f32> {
    find_value(value, keys).and_then(|v| v.as_f64().map(|num| num as f32))
}

fn find_f32_array(value: &Value, keys: &[&str]) -> Option<Vec<f32>> {
    let values = find_value(value, keys)?.as_array()?;
    let mut out = Vec::with_capacity(values.len());
    for item in values {
        let value = item.as_f64()? as f32;
        if !value.is_finite() {
            return None;
        }
        out.push(value);
    }
    Some(out)
}

fn find_string(value: &Value, keys: &[&str]) -> Option<String> {
    find_value(value, keys)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn find_bool(value: &Value, keys: &[&str]) -> Option<bool> {
    find_value(value, keys).and_then(|v| {
        v.as_bool().or_else(|| {
            v.as_u64().and_then(|num| match num {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            })
        })
    })
}

fn rope_scaling_from_json(value: &Value) -> ModelConfigResult<Option<RopeScalingConfig>> {
    let mut config = RopeScalingConfig::default();

    if let Some(scaling) = value.get("rope_scaling") {
        if let Some(obj) = scaling.as_object() {
            if let Some(raw) = obj
                .get("type")
                .or_else(|| obj.get("scaling_type"))
                .and_then(Value::as_str)
            {
                config.scaling_type = Some(RopeScalingType::parse(raw));
            }
            config.rope_type = obj
                .get("rope_type")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            config.factor = obj.get("factor").and_then(|v| v.as_f64().map(|n| n as f32));
            let parse_array = |key: &str| -> ModelConfigResult<Option<Vec<f32>>> {
                obj.get(key)
                    .and_then(Value::as_array)
                    .map(|values| to_f32_array(values.as_slice()))
                    .transpose()
            };
            config.factors = parse_array("factors")?
                .or_else(|| parse_array("short_factor").ok().flatten())
                .or_else(|| parse_array("long_factor").ok().flatten());
            config.base = obj
                .get("base")
                .and_then(|v| v.as_f64().map(|n| n as f32))
                .filter(|v| v.is_finite() && *v > 0.0);
            config.original_max_position_embeddings = obj
                .get("original_max_position_embeddings")
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok());
            config.ext_factor = obj
                .get("ext_factor")
                .and_then(|v| v.as_f64().map(|n| n as f32));
            config.attn_factor = obj
                .get("attn_factor")
                .and_then(|v| v.as_f64().map(|n| n as f32));
            config.beta_fast = obj
                .get("beta_fast")
                .and_then(|v| v.as_f64().map(|n| n as f32));
            config.beta_slow = obj
                .get("beta_slow")
                .and_then(|v| v.as_f64().map(|n| n as f32));
        } else if let Some(factor) = scaling.as_f64().map(|n| n as f32) {
            config.factor = Some(factor);
        } else if let Some(raw) = scaling.as_str() {
            config.scaling_type = Some(RopeScalingType::parse(raw));
        }
    }

    if config.scaling_type.is_none() {
        config.scaling_type = find_string(value, &["rope_type", "rope_scaling_type"])
            .map(|v| RopeScalingType::parse(&v));
    }
    if config.factor.is_none() {
        config.factor = find_f32(value, &["rope_scaling.factor"]);
    }
    if config.factors.is_none() {
        config.factors = find_f32_array(value, &["rope_scaling.factors"]);
    }
    if config.base.is_none() {
        config.base = find_f32(value, &["rope_scaling.base", "rope_base", "rope_theta"]);
    }
    if config.original_max_position_embeddings.is_none() {
        config.original_max_position_embeddings =
            find_usize(value, &["rope_scaling.original_max_position_embeddings"]);
    }

    if let Some(factor) = config.factor {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factor must be positive".to_string(),
            ));
        }
    }
    if let Some(factors) = &config.factors {
        if factors.is_empty() || factors.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factors must contain positive finite values".to_string(),
            ));
        }
    }

    if config.has_any_value() {
        Ok(Some(config))
    } else {
        Ok(None)
    }
}

fn to_f32_array(values: &[Value]) -> ModelConfigResult<Vec<f32>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let Some(v) = value.as_f64().map(|n| n as f32) else {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factors must be numeric".to_string(),
            ));
        };
        if !v.is_finite() {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factors must be finite".to_string(),
            ));
        }
        out.push(v);
    }
    Ok(out)
}

fn rope_scaling_from_gguf(
    reader: &GgufLoader,
    arch: &str,
) -> ModelConfigResult<Option<RopeScalingConfig>> {
    let mut config = RopeScalingConfig {
        scaling_type: gguf_arch_str(reader, arch, "rope.scaling.type")
            .or_else(|| gguf_arch_str(reader, arch, "rope.scaling"))
            .map(RopeScalingType::parse),
        rope_type: gguf_arch_str(reader, arch, "rope.type")
            .or_else(|| gguf_arch_str(reader, arch, "rope.scaling.rope_type"))
            .map(|v| v.to_string()),
        factor: gguf_arch_f32(reader, arch, "rope.scaling.factor"),
        factors: gguf_arch_array_f32(reader, arch, "rope.scaling.factors")
            .or_else(|| gguf_arch_array_f32(reader, arch, "rope.scaling.short_factor"))
            .or_else(|| gguf_arch_array_f32(reader, arch, "rope.scaling.long_factor")),
        base: gguf_arch_f32(reader, arch, "rope.scaling.base")
            .or_else(|| gguf_arch_f32(reader, arch, "rope.freq_base")),
        original_max_position_embeddings: gguf_arch_usize(
            reader,
            arch,
            "rope.scaling.original_max_position_embeddings",
        )
        .or_else(|| gguf_arch_usize(reader, arch, "rope.scaling.original_context_length")),
        ext_factor: gguf_arch_f32(reader, arch, "rope.ext_factor"),
        attn_factor: gguf_arch_f32(reader, arch, "rope.attn_factor"),
        beta_fast: gguf_arch_f32(reader, arch, "rope.beta_fast"),
        beta_slow: gguf_arch_f32(reader, arch, "rope.beta_slow"),
    };

    if let Some(factor) = config.factor {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "GGUF metadata field invalid: rope.scaling.factor".to_string(),
            ));
        }
    }

    if let Some(factors) = &mut config.factors {
        if factors.is_empty() || factors.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(ModelConfigError::InvalidConfig(
                "GGUF metadata field invalid: rope.scaling.factors".to_string(),
            ));
        }
    }

    if config.has_any_value() {
        Ok(Some(config))
    } else {
        Ok(None)
    }
}

fn gguf_arch_key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

fn gguf_arch_u64(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<u64> {
    let key = gguf_arch_key(arch, suffix);
    reader.get_metadata_u64(&key)
}

fn gguf_arch_usize(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<usize> {
    gguf_arch_u64(reader, arch, suffix).and_then(|v| usize::try_from(v).ok())
}

fn gguf_arch_f32(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<f32> {
    let key = gguf_arch_key(arch, suffix);
    reader.get_metadata_f32(&key)
}

fn gguf_arch_str<'a>(reader: &'a GgufLoader, arch: &str, suffix: &str) -> Option<&'a str> {
    let key = gguf_arch_key(arch, suffix);
    reader.get_metadata_str(&key)
}

fn gguf_arch_bool(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<bool> {
    let key = gguf_arch_key(arch, suffix);
    let value = reader.get(&key)?;
    value.as_bool().or_else(|| value.as_u64().map(|v| v != 0))
}

fn gguf_arch_array_f32(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<Vec<f32>> {
    let key = gguf_arch_key(arch, suffix);
    let array = reader.get_metadata_array(&key)?;
    let mut out = Vec::with_capacity(array.items.len());
    for item in &array.items {
        let value = item.as_f32()?;
        if !value.is_finite() {
            return None;
        }
        out.push(value);
    }
    Some(out)
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
