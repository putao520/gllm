//! config.json parsing and architecture resolution.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use serde_json::Value;
use thiserror::Error;

use crate::manifest::{
    FileMap, ModelArchitecture, ModelKind, ModelManifest, TensorNamingRule, EMPTY_FILE_MAP,
};

use super::{CacheLayout, HfHubClient, LoaderConfig, LoaderError, ModelScopeClient, ModelSource};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config.json not found for model: {model_id}")]
    MissingConfig { model_id: String },
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct ConfigFiles {
    pub config_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub source: ModelSource,
}

pub fn download_config_files(
    model_id: &str,
    config: &LoaderConfig,
    file_map: FileMap,
) -> Result<ConfigFiles, ConfigError> {
    let cache = CacheLayout::new(config.cache_dir.clone())?;
    cache.ensure()?;

    let result = download_from_source(model_id, config.source, &cache, file_map);

    if config.enable_fallback {
        result.or_else(|err| {
            if should_fallback(&err) {
                let fallback = super::fallback_source(config.source);
                download_from_source(model_id, fallback, &cache, file_map)
            } else {
                Err(err)
            }
        })
    } else {
        result
    }
}

pub fn load_config_value(path: &Path) -> Result<Value, ConfigError> {
    let bytes = std::fs::read(path)?;
    let value = serde_json::from_slice(&bytes)?;
    Ok(value)
}

pub fn manifest_from_config(
    model_id: &str,
    config: &Value,
    kind: ModelKind,
) -> Result<ModelManifest, ConfigError> {
    let arch = resolve_architecture(config)?;
    let tensor_rules = tensor_rules_for_arch(arch);

    Ok(ModelManifest {
        model_id: Cow::Owned(model_id.to_string()),
        file_map: EMPTY_FILE_MAP,
        arch,
        tensor_rules,
        kind,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
    })
}

pub fn resolve_architecture(config: &Value) -> Result<ModelArchitecture, ConfigError> {
    let mut candidates = Vec::new();

    if let Some(architectures) = config.get("architectures") {
        collect_architectures(architectures, &mut candidates);
    }

    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
        candidates.push(model_type.to_string());
    }

    for candidate in &candidates {
        if let Some(arch) = map_architecture_token(candidate) {
            return Ok(arch);
        }
    }

    // 🚨 禁止基于 Model ID 推断架构 (Ω1: 真实性原则)
    // 必须使用模型自身提供的 metadata (config.json 或 GGUF general.architecture)

    if candidates.is_empty() {
        return Err(ConfigError::UnsupportedArchitecture(
            "missing architectures/model_type".to_string(),
        ));
    }

    Err(ConfigError::UnsupportedArchitecture(candidates.join(", ")))
}

pub fn tensor_rules_for_arch(arch: ModelArchitecture) -> TensorNamingRule {
    match arch {
        ModelArchitecture::Qwen2_5 => TensorNamingRule::Llama4,
        ModelArchitecture::Qwen3 => TensorNamingRule::Qwen3,
        ModelArchitecture::Qwen3MoE => TensorNamingRule::Qwen3,
        ModelArchitecture::Llama4 => TensorNamingRule::Llama4,
        ModelArchitecture::Mistral3 => TensorNamingRule::Mistral3,
        ModelArchitecture::Ministral => TensorNamingRule::Ministral,
        ModelArchitecture::GLM4 => TensorNamingRule::GLM4,
        ModelArchitecture::GLM5 => TensorNamingRule::GLM5,
        ModelArchitecture::GPT2Next => TensorNamingRule::GPT2Next,
        ModelArchitecture::Phi4 => TensorNamingRule::Phi4,
        ModelArchitecture::Gemma2 => TensorNamingRule::Gemma2,
        ModelArchitecture::XlmR => TensorNamingRule::XlmR,
        ModelArchitecture::XlmRNext => TensorNamingRule::XlmRNext,
    }
}

fn download_from_source(
    model_id: &str,
    source: ModelSource,
    cache: &CacheLayout,
    file_map: FileMap,
) -> Result<ConfigFiles, ConfigError> {
    match source {
        ModelSource::HuggingFace => {
            let hf = HfHubClient::new(cache.hf_cache_dir())?;
            let config_path = hf
                .download_config_file(model_id, file_map)
                .map_err(|err| map_missing_config(model_id, err))?;
            let tokenizer_path = hf.download_tokenizer_file(model_id, file_map).ok();
            Ok(ConfigFiles {
                config_path,
                tokenizer_path,
                source,
            })
        }
        ModelSource::ModelScope => {
            let ms = ModelScopeClient::new(cache.modelscope_cache_dir())?;
            let config_path = ms
                .download_config_file(model_id, file_map)
                .map_err(|err| map_missing_config(model_id, err))?;
            let tokenizer_path = ms.download_tokenizer_file(model_id, file_map).ok();
            Ok(ConfigFiles {
                config_path,
                tokenizer_path,
                source,
            })
        }
    }
}

fn map_missing_config(model_id: &str, err: LoaderError) -> ConfigError {
    match err {
        LoaderError::MissingWeights => ConfigError::MissingConfig {
            model_id: model_id.to_string(),
        },
        other => ConfigError::Loader(other),
    }
}

fn should_fallback(err: &ConfigError) -> bool {
    match err {
        ConfigError::MissingConfig { .. } => true,
        ConfigError::Loader(loader) => super::is_recoverable_error(loader),
        _ => false,
    }
}

fn collect_architectures(value: &Value, out: &mut Vec<String>) {
    if let Some(array) = value.as_array() {
        for item in array {
            if let Some(name) = item.as_str() {
                out.push(name.to_string());
                continue;
            }
            if let Some(obj) = item.as_object() {
                if let Some(name) = obj.get("type").and_then(|v| v.as_str()) {
                    out.push(name.to_string());
                }
                if let Some(name) = obj.get("architecture").and_then(|v| v.as_str()) {
                    out.push(name.to_string());
                }
            }
        }
    } else if let Some(name) = value.as_str() {
        out.push(name.to_string());
    }
}

fn map_architecture_token(token: &str) -> Option<ModelArchitecture> {
    match normalize_architecture_token(token).as_str() {
        "ministral" | "ministralforcausallm" => Some(ModelArchitecture::Ministral),
        "mistral" | "mistralforcausallm" => Some(ModelArchitecture::Mistral3),
        "qwen3_moe" | "qwen3moe" | "qwen3moeforcausallm" => Some(ModelArchitecture::Qwen3MoE),
        "qwen3" | "qwen3forcausallm" => Some(ModelArchitecture::Qwen3),
        "qwen2_5" | "qwen2_5forcausallm" => Some(ModelArchitecture::Qwen2_5),
        "qwen2" | "qwen2forcausallm" => Some(ModelArchitecture::Qwen2_5),
        "llama" | "llamaforcausallm" => Some(ModelArchitecture::Llama4),
        "phi3" | "phi3forcausallm" | "phi4" | "phi4forcausallm" => Some(ModelArchitecture::Phi4),
        "gemma" | "gemmaforcausallm" | "gemma2" | "gemma2forcausallm" => {
            Some(ModelArchitecture::Gemma2)
        }
        "glm5" | "glm5forcausallm" => Some(ModelArchitecture::GLM5),
        "glm4" | "glm4forcausallm" | "chatglm" | "chatglmforcausallm" => {
            Some(ModelArchitecture::GLM4)
        }
        "glm" | "glmforcausallm" => Some(ModelArchitecture::GLM5),
        "gpt2" | "gpt2lmheadmodel" | "gpt_oss" | "gptoss" => Some(ModelArchitecture::GPT2Next),
        "xlm_roberta" | "xlm_roberta_model" | "xlmr" | "roberta" | "bert" => {
            Some(ModelArchitecture::XlmR)
        }
        _ => None,
    }
}

fn normalize_architecture_token(token: &str) -> String {
    token
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
        .map(|ch| match ch {
            '-' | '.' => '_',
            _ => ch.to_ascii_lowercase(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_architecture_token_uses_exact_normalized_matching() {
        assert_eq!(
            map_architecture_token("LlamaForCausalLM"),
            Some(ModelArchitecture::Llama4)
        );
        assert_eq!(
            map_architecture_token("Qwen2ForCausalLM"),
            Some(ModelArchitecture::Qwen2_5)
        );
        assert_eq!(
            map_architecture_token("Qwen2.5ForCausalLM"),
            Some(ModelArchitecture::Qwen2_5)
        );
        assert_eq!(
            map_architecture_token("MistralForCausalLM"),
            Some(ModelArchitecture::Mistral3)
        );
        assert_eq!(
            map_architecture_token("Gemma2ForCausalLM"),
            Some(ModelArchitecture::Gemma2)
        );
        assert_eq!(map_architecture_token("custom-llama-adapter"), None);
    }
}
