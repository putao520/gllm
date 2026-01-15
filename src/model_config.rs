use crate::model_presets::model_defaults;
use crate::types::{Error, Result};
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs;
use std::path::Path;

fn de_usize_or_zero<'de, D>(deserializer: D) -> std::result::Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<usize>::deserialize(deserializer)?.unwrap_or(0))
}

fn merge_json(base: Value, override_val: Value) -> Value {
    match (base, override_val) {
        (Value::Object(mut base_map), Value::Object(override_map)) => {
            for (key, value) in override_map {
                base_map.insert(key, value);
            }
            Value::Object(base_map)
        }
        (_, override_map) if !override_map.is_null() => override_map,
        (base_map, _) => base_map,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigWarning {
    HighDropout(&'static str, f32),
    ShortContext(usize),
    MissingTypeVocabSize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigAutoFix {
    SetHiddenSize(usize),
    SetAttentionHeads(usize),
    SetIntermediateSize(usize),
    SetLayerNormEps(f64),
    SetHiddenAct(String),
    SetDropout(&'static str, f32),
    SetTypeVocabSize(usize),
    SetPadTokenId(i64),
    SetModelType(String),
    SetMaxPositionEmbeddings(usize),
    SetMaxBatchSize(usize),
    SetMemoryLimit(usize),
    SetGpuMemoryFraction(f32),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValidationReport {
    pub warnings: Vec<ConfigWarning>,
    pub auto_fixes: Vec<ConfigAutoFix>,
}

/// HuggingFace BERT-style model configuration from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default, deserialize_with = "de_usize_or_zero")]
    pub hidden_size: usize,
    #[serde(default, deserialize_with = "de_usize_or_zero")]
    pub num_hidden_layers: usize,
    #[serde(default, deserialize_with = "de_usize_or_zero")]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default, deserialize_with = "de_usize_or_zero")]
    pub vocab_size: usize,
    #[serde(default, deserialize_with = "de_usize_or_zero")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub type_vocab_size: Option<usize>,
    #[serde(default)]
    pub attention_probs_dropout_prob: Option<f32>,
    #[serde(default)]
    pub hidden_dropout_prob: Option<f32>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default, alias = "n_routed_experts", alias = "num_local_experts")]
    pub num_experts: Option<usize>,
    #[serde(default, alias = "num_experts_per_token", alias = "top_k")]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub max_batch_size: Option<usize>,
    #[serde(default)]
    pub memory_limit_mb: Option<usize>,
    #[serde(default)]
    pub gpu_memory_fraction: Option<f32>,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub initializer_range: Option<f32>,
    #[serde(default)]
    pub layer_norm_eps: Option<f64>,
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_scaling: Option<Value>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_cache: Option<bool>,
    #[serde(default)]
    pub position_embedding_type: Option<String>,
    #[serde(default)]
    pub pooler_hidden_act: Option<String>,
    #[serde(default)]
    pub pooler_dropout: Option<f32>,
    #[serde(default)]
    pub pooling_type: Option<String>,
    #[serde(default)]
    pub num_labels: Option<usize>,
    #[serde(default)]
    pub classifier_dropout: Option<f32>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub is_decoder: Option<bool>,
    #[serde(default)]
    pub cross_attention_hidden_size: Option<usize>,
    #[serde(default)]
    pub pad_token_id: Option<i64>,
    #[serde(default)]
    pub bos_token_id: Option<i64>,
    #[serde(default)]
    pub eos_token_id: Option<i64>,
    /// Engram conditional memory configuration (DeepSeek-V4+)
    #[serde(default)]
    pub engram_enabled: Option<bool>,
    /// Engram N-gram size (typically 2-4)
    #[serde(default)]
    pub engram_ngram_size: Option<usize>,
    /// Engram number of buckets in embedding table
    #[serde(default)]
    pub engram_num_buckets: Option<usize>,
    /// Engram embedding dimension (should match hidden_size)
    #[serde(default)]
    pub engram_embedding_dim: Option<usize>,
    /// Engram output scaling factor
    #[serde(default)]
    pub engram_scale: Option<f32>,
    #[serde(default)]
    pub extra: Value,
}

impl ModelConfig {
    pub fn from_file(config_path: &Path) -> Result<Self> {
        let content = fs::read_to_string(config_path)
            .map_err(|e| Error::LoadError(format!("Failed to read config.json: {}", e)))?;
        let mut config: ModelConfig = serde_json::from_str(&content)
            .map_err(|e| Error::LoadError(format!("Failed to parse config.json: {}", e)))?;
        if config.extra.is_null() {
            config.extra = Value::Object(Map::new());
        }
        Ok(config)
    }

    pub fn load(repo_id: &str, config_path: Option<&Path>) -> Result<(Self, ValidationReport)> {
        let mut resolved = model_defaults(repo_id);
        if let Some(path) = config_path {
            if path.exists() {
                let user = ModelConfig::from_file(path)?;
                resolved = resolved.merge(user);
            }
        }

        let mut report = ValidationReport::default();
        report.auto_fixes.extend(resolved.apply_defaults(repo_id));
        report.warnings.extend(resolved.validate(repo_id)?);
        Ok((resolved, report))
    }

    fn merge(mut self, override_config: ModelConfig) -> ModelConfig {
        if override_config.hidden_size != 0 {
            self.hidden_size = override_config.hidden_size;
        }
        if override_config.num_hidden_layers != 0 {
            self.num_hidden_layers = override_config.num_hidden_layers;
        }
        if override_config.num_attention_heads != 0 {
            self.num_attention_heads = override_config.num_attention_heads;
        }
        if override_config.vocab_size != 0 {
            self.vocab_size = override_config.vocab_size;
        }
        if override_config.max_position_embeddings != 0 {
            self.max_position_embeddings = override_config.max_position_embeddings;
        }

        macro_rules! merge_opt {
            ($field:ident) => {
                if override_config.$field.is_some() {
                    self.$field = override_config.$field;
                }
            };
        }

        merge_opt!(architectures);
        merge_opt!(model_type);
        merge_opt!(type_vocab_size);
        merge_opt!(attention_probs_dropout_prob);
        merge_opt!(hidden_dropout_prob);
        merge_opt!(intermediate_size);
        merge_opt!(num_experts);
        merge_opt!(num_experts_per_tok);
        merge_opt!(n_shared_experts);
        merge_opt!(moe_intermediate_size);
        merge_opt!(max_batch_size);
        merge_opt!(memory_limit_mb);
        merge_opt!(gpu_memory_fraction);
        merge_opt!(hidden_act);
        merge_opt!(initializer_range);
        merge_opt!(layer_norm_eps);
        merge_opt!(rms_norm_eps);
        merge_opt!(rope_theta);
        merge_opt!(rope_scaling);
        merge_opt!(sliding_window);
        merge_opt!(num_key_value_heads);
        merge_opt!(head_dim);
        merge_opt!(use_cache);
        merge_opt!(position_embedding_type);
        merge_opt!(pooler_hidden_act);
        merge_opt!(pooler_dropout);
        merge_opt!(pooling_type);
        merge_opt!(num_labels);
        merge_opt!(classifier_dropout);
        merge_opt!(tie_word_embeddings);
        merge_opt!(is_decoder);
        merge_opt!(cross_attention_hidden_size);
        merge_opt!(pad_token_id);
        merge_opt!(bos_token_id);
        merge_opt!(eos_token_id);
        merge_opt!(engram_enabled);
        merge_opt!(engram_ngram_size);
        merge_opt!(engram_num_buckets);
        merge_opt!(engram_embedding_dim);
        merge_opt!(engram_scale);

        self.extra = merge_json(self.extra, override_config.extra);
        self
    }

    fn is_roberta_like(&self) -> bool {
        let model_type_matches = self
            .model_type
            .as_ref()
            .map_or(false, |t| t.contains("roberta"));
        let arch_matches = self.architectures.as_ref().map_or(false, |a| {
            a.iter()
                .any(|arch| arch.contains("Roberta") || arch.contains("XLMRoberta"))
        });
        model_type_matches || arch_matches
    }

    fn apply_defaults(&mut self, model_id: &str) -> Vec<ConfigAutoFix> {
        let mut fixes = Vec::new();
        if self.hidden_size == 0 {
            self.hidden_size = 768;
            fixes.push(ConfigAutoFix::SetHiddenSize(self.hidden_size));
        }
        if self.num_hidden_layers == 0 {
            self.num_hidden_layers = 12;
            fixes.push(ConfigAutoFix::SetModelType(format!(
                "{model_id} layers set to 12"
            )));
        }
        if self.num_attention_heads == 0 {
            self.num_attention_heads = 12;
            fixes.push(ConfigAutoFix::SetAttentionHeads(self.num_attention_heads));
        }
        if self.max_position_embeddings == 0 {
            self.max_position_embeddings = 512;
            fixes.push(ConfigAutoFix::SetMaxPositionEmbeddings(
                self.max_position_embeddings,
            ));
        }
        if self.vocab_size == 0 {
            self.vocab_size = 30522;
            fixes.push(ConfigAutoFix::SetModelType(format!(
                "{model_id} vocab_size set to default"
            )));
        }
        if self.intermediate_size.is_none() {
            let value = self.hidden_size.saturating_mul(4);
            self.intermediate_size = Some(value);
            fixes.push(ConfigAutoFix::SetIntermediateSize(value));
        }
        if self.hidden_act.is_none() {
            self.hidden_act = Some("gelu".to_string());
            fixes.push(ConfigAutoFix::SetHiddenAct("gelu".to_string()));
        }

        let default_layer_norm = if self.is_roberta_like() { 1e-5 } else { 1e-12 };
        if self.layer_norm_eps.is_none() && self.rms_norm_eps.is_none() {
            self.layer_norm_eps = Some(default_layer_norm);
            fixes.push(ConfigAutoFix::SetLayerNormEps(default_layer_norm));
        }

        if self.attention_probs_dropout_prob.is_none() {
            self.attention_probs_dropout_prob = Some(0.1);
            fixes.push(ConfigAutoFix::SetDropout(
                "attention_probs_dropout_prob",
                0.1,
            ));
        }
        if self.hidden_dropout_prob.is_none() {
            self.hidden_dropout_prob = Some(0.1);
            fixes.push(ConfigAutoFix::SetDropout("hidden_dropout_prob", 0.1));
        }
        if self.classifier_dropout.is_none() && self.pooler_dropout.is_some() {
            self.classifier_dropout = self.pooler_dropout;
            fixes.push(ConfigAutoFix::SetDropout(
                "classifier_dropout",
                self.pooler_dropout.unwrap(),
            ));
        }

        if self.model_type.is_none() {
            let default_type = if self.is_roberta_like() {
                "roberta"
            } else {
                "bert"
            };
            self.model_type = Some(default_type.to_string());
            fixes.push(ConfigAutoFix::SetModelType(default_type.to_string()));
        }

        if self.type_vocab_size.is_none() {
            let value = if self.is_roberta_like() { 1 } else { 2 };
            self.type_vocab_size = Some(value);
            fixes.push(ConfigAutoFix::SetTypeVocabSize(value));
        }

        if self.pad_token_id.is_none() {
            let value = if self.is_roberta_like() { 1 } else { 0 };
            self.pad_token_id = Some(value);
            fixes.push(ConfigAutoFix::SetPadTokenId(value));
        }

        if self.max_batch_size.is_none() {
            self.max_batch_size = Some(8);
            fixes.push(ConfigAutoFix::SetMaxBatchSize(8));
        }
        if self.memory_limit_mb.is_none() {
            self.memory_limit_mb = Some(512);
            fixes.push(ConfigAutoFix::SetMemoryLimit(512));
        }
        if self.gpu_memory_fraction.is_none() {
            self.gpu_memory_fraction = Some(1.0);
            fixes.push(ConfigAutoFix::SetGpuMemoryFraction(1.0));
        }

        fixes
    }

    #[allow(dead_code)]
    pub fn get_embedding_dim(&self) -> usize {
        self.hidden_size
    }

    pub fn is_cross_encoder(&self) -> bool {
        self.architectures.as_ref().map_or(false, |a| {
            a.iter().any(|arch| {
                arch.contains("SequenceClassification") || arch.contains("CrossEncoder")
            })
        }) || self
            .model_type
            .as_ref()
            .map_or(false, |t| t.contains("cross-encoder"))
            || self.num_labels == Some(1)
            || self
                .pooling_type
                .as_ref()
                .map_or(false, |p| p.eq_ignore_ascii_case("cls"))
    }

    pub fn validate(&self, model_id: &str) -> Result<Vec<ConfigWarning>> {
        if self.hidden_size == 0 {
            return Err(Error::InvalidConfig(format!(
                "hidden_size is missing for model {model_id}; ensure config.json includes hidden_size or rely on built-in presets"
            )));
        }
        if self.num_hidden_layers == 0 {
            return Err(Error::InvalidConfig(format!(
                "num_hidden_layers is missing for model {model_id}"
            )));
        }
        if self.num_attention_heads == 0 {
            return Err(Error::InvalidConfig(format!(
                "num_attention_heads is missing for model {model_id}"
            )));
        }
        if self.vocab_size == 0 {
            return Err(Error::InvalidConfig(format!(
                "vocab_size is missing for model {model_id}"
            )));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(Error::InvalidConfig(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({}) for model {}",
                self.hidden_size, self.num_attention_heads, model_id
            )));
        }
        if self.intermediate_size.unwrap_or(0) == 0 {
            return Err(Error::InvalidConfig(format!(
                "intermediate_size is missing for model {model_id}"
            )));
        }
        if self.type_vocab_size.unwrap_or(0) == 0 {
            return Err(Error::InvalidConfig(
                "type_vocab_size must be greater than 0".to_string(),
            ));
        }

        let mut warnings = Vec::new();
        if let Some(dropout) = self.hidden_dropout_prob {
            if dropout > 0.5 {
                warnings.push(ConfigWarning::HighDropout("hidden_dropout_prob", dropout));
            }
        }
        if self.max_position_embeddings < 128 {
            warnings.push(ConfigWarning::ShortContext(self.max_position_embeddings));
        }
        if self.type_vocab_size.is_none() {
            warnings.push(ConfigWarning::MissingTypeVocabSize);
        }

        Ok(warnings)
    }

    /// Check if Engram conditional memory is enabled for this model.
    pub fn is_engram_enabled(&self) -> bool {
        self.engram_enabled.unwrap_or(false)
    }

    /// Get Engram configuration if enabled.
    /// Returns (ngram_size, num_buckets, embedding_dim, scale).
    pub fn engram_config(&self) -> Option<(usize, usize, usize, f32)> {
        if !self.is_engram_enabled() {
            return None;
        }
        let ngram_size = self.engram_ngram_size.unwrap_or(3);
        let num_buckets = self.engram_num_buckets.unwrap_or(1 << 20); // 1M default
        let embedding_dim = self.engram_embedding_dim.unwrap_or(self.hidden_size);
        let scale = self.engram_scale.unwrap_or(1.0);
        Some((ngram_size, num_buckets, embedding_dim, scale))
    }

    #[allow(dead_code)]
    pub fn get_model_defaults(repo_id: &str) -> Self {
        model_defaults(repo_id)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            architectures: None,
            model_type: None,
            hidden_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
            attention_probs_dropout_prob: None,
            hidden_dropout_prob: None,
            intermediate_size: None,
            num_experts: None,
            num_experts_per_tok: None,
            n_shared_experts: None,
            moe_intermediate_size: None,
            max_batch_size: None,
            memory_limit_mb: None,
            gpu_memory_fraction: None,
            hidden_act: None,
            initializer_range: None,
            layer_norm_eps: None,
            rms_norm_eps: None,
            rope_theta: None,
            rope_scaling: None,
            sliding_window: None,
            num_key_value_heads: None,
            head_dim: None,
            use_cache: Some(true),
            position_embedding_type: Some("absolute".to_string()),
            pooler_hidden_act: None,
            pooler_dropout: None,
            pooling_type: None,
            num_labels: None,
            classifier_dropout: None,
            tie_word_embeddings: Some(true),
            is_decoder: Some(false),
            cross_attention_hidden_size: None,
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            type_vocab_size: None,
            engram_enabled: None,
            engram_ngram_size: None,
            engram_num_buckets: None,
            engram_embedding_dim: None,
            engram_scale: None,
            extra: Value::Object(Map::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;
    use crate::model_presets::model_defaults;
    use crate::types::Error;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn load_merges_user_overrides() {
        let dir = tempdir().expect("temp dir");
        let config_path = dir.path().join("config.json");
        fs::write(
            &config_path,
            r#"{
                "hidden_size": 256,
                "num_hidden_layers": 6,
                "num_attention_heads": 8,
                "vocab_size": 123,
                "max_position_embeddings": 128
            }"#,
        )
        .expect("write config");

        let (cfg, _) = ModelConfig::load("baai/bge-small-en-v1.5", Some(config_path.as_path()))
            .expect("load config");
        assert_eq!(cfg.hidden_size, 256);
        assert_eq!(cfg.num_hidden_layers, 6);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.vocab_size, 123);
        assert_eq!(cfg.max_position_embeddings, 128);
        assert_eq!(cfg.type_vocab_size, Some(2));
        assert_eq!(cfg.pad_token_id, Some(0));
    }

    #[test]
    fn validate_rejects_invalid_head_ratio() {
        let mut cfg = model_defaults("baai/bge-small-en-v1.5");
        cfg.num_attention_heads = 10;
        let err = cfg.validate("baai/bge-small-en-v1.5").unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
        let msg = err.to_string();
        assert!(msg.contains("divisible"));
        assert!(msg.contains("baai/bge-small-en-v1.5"));
    }
}
