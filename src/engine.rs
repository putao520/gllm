use crate::decoder_model::DecoderModel;
use crate::dynamic_bert::{DynamicBertModel, DynamicCrossEncoder};
use crate::generation::{GenerationConfig, GenerationOptions, GenerationOutput};
use crate::generator_engine::GeneratorEngine;
use crate::model_config::ModelConfig;
use crate::registry::{Architecture, ModelInfo, ModelType, Quantization};
use crate::tensor::Matrix;
use crate::types::{Device, Error, Result};
use gllm_kernels::backend::auto_select_static;
use gllm_kernels::{detect_backend, BackendType};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

/// Cached backend detection result (OnceLock for single initialization).
/// This avoids repeated GPU capability probing which can be expensive.
static CACHED_BACKEND_TYPE: OnceLock<BackendType> = OnceLock::new();

/// Get cached backend type, detecting only once on first call.
#[inline]
fn get_cached_backend_type() -> BackendType {
    *CACHED_BACKEND_TYPE.get_or_init(detect_backend)
}

pub(crate) const MAX_SEQ_LEN: usize = 512;

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    pad_token: Option<Value>,
    pad_token_id: Option<i64>,
    eos_token: Option<Value>,
    eos_token_id: Option<i64>,
    vocab_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SpecialTokensMap {
    pad_token: Option<Value>,
    eos_token: Option<Value>,
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub(crate) fn find_model_file(model_dir: &Path, quantization: &Quantization) -> Option<PathBuf> {
    match quantization {
        Quantization::GGUF => {
            if let Ok(entries) = fs::read_dir(model_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path
                        .extension()
                        .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                        .unwrap_or(false)
                    {
                        return Some(path);
                    }
                }
            }
            None
        }
        _ => {
            let shard_index = model_dir.join("model.safetensors.index.json");
            if shard_index.exists() {
                return Some(shard_index);
            }
            let safetensors = model_dir.join("model.safetensors");
            safetensors.exists().then_some(safetensors)
        }
    }
}

fn token_value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(token) => Some(token.clone()),
        Value::Object(map) => map
            .get("content")
            .and_then(Value::as_str)
            .map(|token| token.to_string()),
        _ => None,
    }
}

fn resolve_token_id(
    token: &str,
    tokenizer: Option<&Tokenizer>,
    vocab_map: Option<&HashMap<String, u32>>,
) -> Option<i64> {
    if let Some(tokenizer) = tokenizer {
        if let Some(id) = tokenizer.token_to_id(token) {
            return Some(id as i64);
        }
    }
    vocab_map.and_then(|vocab| vocab.get(token).map(|id| *id as i64))
}

fn resolve_pad_id(
    tokenizer: Option<&Tokenizer>,
    config: Option<&TokenizerConfig>,
    special_tokens: Option<&SpecialTokensMap>,
    vocab_map: Option<&HashMap<String, u32>>,
) -> i64 {
    if let Some(tokenizer) = tokenizer {
        if let Some(padding) = tokenizer.get_padding() {
            return padding.pad_id as i64;
        }
    }

    if let Some(config) = config {
        if let Some(pad_id) = config.pad_token_id {
            return pad_id;
        }
        if let Some(pad_token) = config
            .pad_token
            .as_ref()
            .and_then(token_value_to_string)
        {
            if let Some(pad_id) = resolve_token_id(&pad_token, tokenizer, vocab_map) {
                return pad_id;
            }
        }
        if let Some(eos_id) = config.eos_token_id {
            return eos_id;
        }
        if let Some(eos_token) = config
            .eos_token
            .as_ref()
            .and_then(token_value_to_string)
        {
            if let Some(eos_id) = resolve_token_id(&eos_token, tokenizer, vocab_map) {
                return eos_id;
            }
        }
    }

    if let Some(special_tokens) = special_tokens {
        if let Some(pad_token) = special_tokens
            .pad_token
            .as_ref()
            .and_then(token_value_to_string)
        {
            if let Some(pad_id) = resolve_token_id(&pad_token, tokenizer, vocab_map) {
                return pad_id;
            }
        }
        if let Some(eos_token) = special_tokens
            .eos_token
            .as_ref()
            .and_then(token_value_to_string)
        {
            if let Some(eos_id) = resolve_token_id(&eos_token, tokenizer, vocab_map) {
                return eos_id;
            }
        }
    }

    0
}

/// Lightweight tokenizer wrapper supporting both HF tokenizers and a deterministic fallback.
#[derive(Clone)]
pub struct TokenizerAdapter {
    inner: TokenizerAdapterInner,
}

#[derive(Clone)]
enum TokenizerAdapterInner {
    Tokenizer { tokenizer: Tokenizer, pad_id: i64 },
    Fallback { vocab_size: usize, pad_id: i64 },
}

impl TokenizerAdapter {
    /// Load tokenizer from model directory if available.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer_config = read_json::<TokenizerConfig>(&dir.join("tokenizer_config.json"));
        let special_tokens = read_json::<SpecialTokensMap>(&dir.join("special_tokens_map.json"));
        let vocab_map = read_json::<HashMap<String, u32>>(&dir.join("vocab.json"));

        if tokenizer_path.exists() {
            if let Ok(tokenizer) = Tokenizer::from_file(&tokenizer_path) {
                let pad_id = resolve_pad_id(
                    Some(&tokenizer),
                    tokenizer_config.as_ref(),
                    special_tokens.as_ref(),
                    vocab_map.as_ref(),
                );
                return Ok(Self {
                    inner: TokenizerAdapterInner::Tokenizer { tokenizer, pad_id },
                });
            }
        }

        let vocab_size = vocab_map
            .as_ref()
            .map(|vocab| vocab.len())
            .or_else(|| tokenizer_config.as_ref().and_then(|config| config.vocab_size))
            .unwrap_or(32_000);
        let pad_id = resolve_pad_id(
            None,
            tokenizer_config.as_ref(),
            special_tokens.as_ref(),
            vocab_map.as_ref(),
        );

        Ok(Self {
            inner: TokenizerAdapterInner::Fallback { vocab_size, pad_id },
        })
    }

    /// Encode a single string into token ids.
    pub fn encode(&self, text: &str, max_len: usize) -> (Vec<i64>, usize) {
        let limit = max_len.max(1);
        match &self.inner {
            TokenizerAdapterInner::Tokenizer { tokenizer, .. } => {
                if let Ok(encoding) = tokenizer.encode(text, true) {
                    let mut ids: Vec<i64> =
                        encoding.get_ids().iter().map(|id| *id as i64).collect();
                    let used = ids.len();
                    ids.truncate(limit);
                    return (self.pad(ids, limit), used);
                }
                self.fallback_encode(text, limit, tokenizer.get_vocab_size(true))
            }
            TokenizerAdapterInner::Fallback { vocab_size, .. } => {
                self.fallback_encode(text, limit, *vocab_size)
            }
        }
    }

    /// Encode a string into token ids without padding.
    pub fn encode_unpadded(&self, text: &str, max_len: usize) -> Vec<i64> {
        let limit = max_len.max(1);
        match &self.inner {
            TokenizerAdapterInner::Tokenizer { tokenizer, .. } => {
                if let Ok(encoding) = tokenizer.encode(text, true) {
                    let mut ids: Vec<i64> =
                        encoding.get_ids().iter().map(|id| *id as i64).collect();
                    ids.truncate(limit);
                    if ids.is_empty() {
                        ids.push(self.pad_id());
                    }
                    return ids;
                }
                self.fallback_encode_unpadded(text, limit, tokenizer.get_vocab_size(true))
            }
            TokenizerAdapterInner::Fallback { vocab_size, .. } => {
                self.fallback_encode_unpadded(text, limit, *vocab_size)
            }
        }
    }

    /// Decode token ids into a string.
    pub fn decode(&self, tokens: &[i64]) -> String {
        match &self.inner {
            TokenizerAdapterInner::Tokenizer { tokenizer, .. } => {
                let ids: Vec<u32> = tokens
                    .iter()
                    .filter_map(|id| u32::try_from(*id).ok())
                    .collect();
                if let Ok(text) = tokenizer.decode(&ids, true) {
                    return text;
                }
            }
            TokenizerAdapterInner::Fallback { .. } => {}
        }

        tokens
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Encode a query and document pair by concatenating tokens with a separator.
    pub fn encode_pair(&self, query: &str, document: &str, max_len: usize) -> (Vec<i64>, usize) {
        let half = max_len / 2;
        let (mut query_tokens, used_query) = self.encode(query, half.max(1));
        let (mut doc_tokens, used_doc) = self.encode(document, max_len - query_tokens.len());

        let mut merged = Vec::with_capacity(max_len);
        merged.push(101); // CLS token id
        merged.append(&mut query_tokens);
        merged.push(102); // SEP token id
        merged.append(&mut doc_tokens);
        merged.truncate(max_len);

        let used = used_query + used_doc + 2;
        (self.pad(merged, max_len), used)
    }

    /// Vocabulary size for embedding tables.
    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        // Reserve one additional slot for padding.
        let size = match &self.inner {
            TokenizerAdapterInner::Tokenizer { tokenizer, .. } => tokenizer.get_vocab_size(true),
            TokenizerAdapterInner::Fallback { vocab_size, .. } => *vocab_size,
        };
        size + 1
    }

    fn pad(&self, mut ids: Vec<i64>, len: usize) -> Vec<i64> {
        if ids.len() < len {
            ids.resize(len, self.pad_id());
        }
        if ids.is_empty() {
            ids.push(self.pad_id());
        }
        ids
    }

    fn pad_id(&self) -> i64 {
        match &self.inner {
            TokenizerAdapterInner::Tokenizer { pad_id, .. } => *pad_id,
            TokenizerAdapterInner::Fallback { pad_id, .. } => *pad_id,
        }
    }

    fn fallback_encode(&self, text: &str, limit: usize, vocab_size: usize) -> (Vec<i64>, usize) {
        let vocab_size = vocab_size.max(1);
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            ids.push(((hash % vocab_size as u64) as i64).max(1));
            if ids.len() >= limit {
                break;
            }
        }
        let used = ids.len();
        (self.pad(ids, limit), used)
    }

    fn fallback_encode_unpadded(&self, text: &str, limit: usize, vocab_size: usize) -> Vec<i64> {
        let vocab_size = vocab_size.max(1);
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            ids.push(((hash % vocab_size as u64) as i64).max(1));
            if ids.len() >= limit {
                break;
            }
        }
        if ids.is_empty() {
            ids.push(self.pad_id());
        }
        ids
    }
}

pub(crate) enum EmbeddingModel {
    Bert(DynamicBertModel),
    Decoder(DecoderModel),
}

fn pooled_rows(pooled: Matrix) -> Vec<Vec<f32>> {
    pooled
        .data
        .chunks(pooled.cols)
        .map(|row| row.to_vec())
        .collect()
}

impl EmbeddingModel {
    fn forward(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self {
            EmbeddingModel::Bert(model) => {
                let hidden_states = model.forward(tokens)?;
                let pooled = model.pool_hidden_states(&hidden_states, tokens);
                Ok(pooled_rows(pooled))
            }
            EmbeddingModel::Decoder(model) => {
                let hidden_states = model.forward(tokens)?;
                let pooled = model.pool_hidden_states(&hidden_states, tokens);
                Ok(pooled_rows(pooled))
            }
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            EmbeddingModel::Bert(model) => model.hidden_size(),
            EmbeddingModel::Decoder(model) => model.hidden_size(),
        }
    }
}

pub(crate) struct EmbeddingEngine {
    model: EmbeddingModel,
}

impl EmbeddingEngine {
    pub fn new(model_dir: &Path, info: &ModelInfo) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let repo_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .replace("--", "/");
        let config_file = config_path.exists().then_some(config_path.as_path());
        let (config, _) = ModelConfig::load(&repo_name, config_file)?;

        let model_path = find_model_file(model_dir, &info.quantization);

        // Create backend ONCE at engine level
        let backend = auto_select_static();

        let model = match info.architecture {
            Architecture::Bert
            | Architecture::CrossEncoder
            | Architecture::Qwen3Embedding
            | Architecture::Qwen3Reranker
            | Architecture::JinaV4
            | Architecture::JinaRerankerV3
            | Architecture::NVIDIANemotron
            | Architecture::Gemma3n
            | Architecture::GLM4 => {
                let mut model = DynamicBertModel::new(config.clone(), backend)?;
                if let Some(model_path) = model_path.as_ref() {
                    model.load_safetensors(model_path)?;
                }
                EmbeddingModel::Bert(model)
            }
            _ => {
                let mut model = DecoderModel::new(config.clone(), backend)?;
                if let Some(model_path) = model_path.as_ref() {
                    model.load_safetensors(model_path)?;
                }
                EmbeddingModel::Decoder(model)
            }
        };

        Ok(Self { model })
    }

    pub fn embed(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Err(Error::InferenceError(
                "At least one input is required".into(),
            ));
        }
        self.model.forward(tokens)
    }

    pub fn hidden_size(&self) -> usize {
        self.model.hidden_size()
    }
}

#[derive(Clone)]
pub(crate) struct RerankEngine {
    model: DynamicCrossEncoder,
}

impl RerankEngine {
    pub fn new(model_dir: &Path, info: &ModelInfo) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let repo_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .replace("--", "/");
        let config_file = config_path.exists().then_some(config_path.as_path());
        let (config, _) = ModelConfig::load(&repo_name, config_file)?;

        // Create backend ONCE at engine level
        let backend = auto_select_static();

        let mut model = DynamicCrossEncoder::new(config, backend)?;
        if let Some(model_path) = find_model_file(model_dir, &info.quantization) {
            model.load_safetensors(&model_path)?;
        }

        Ok(Self { model })
    }

    pub fn score(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::InferenceError(
                "At least one query-document pair is required".into(),
            ));
        }
        self.model.score(tokens)
    }
}

/// Backend-specific engine bundle.
pub(crate) struct EngineBackend {
    embedding: Option<EmbeddingEngine>,
    rerank: Option<RerankEngine>,
    generator: Option<GeneratorEngine>,
}

impl EngineBackend {
    pub fn run_embeddings(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self.embedding.as_ref() {
            Some(embedding) => embedding.embed(tokens),
            None => Err(Error::InvalidConfig(
                "Embeddings are not supported for generator models".into(),
            )),
        }
    }

    pub fn run_rerank(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        match self.rerank.as_ref() {
            Some(rerank) => rerank.score(tokens),
            None => Err(Error::InvalidConfig(
                "Rerank is not supported for generator models".into(),
            )),
        }
    }

    #[allow(dead_code)]
    pub fn run_generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
    ) -> Result<GenerationOutput> {
        self.run_generate_with_options(prompt_ids, config, tokenizer, &GenerationOptions::default())
    }

    pub fn run_generate_with_options(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
        options: &GenerationOptions,
    ) -> Result<GenerationOutput> {
        match self.generator.as_ref() {
            Some(generator) => generator.generate(prompt_ids, config, tokenizer, options),
            None => Err(Error::InvalidConfig(
                "Generation is not supported for this model".into(),
            )),
        }
    }

    pub fn max_position_embeddings(&self) -> Option<usize> {
        self.generator
            .as_ref()
            .map(|generator| generator.max_position_embeddings())
    }
}

/// Build an embedding-only backend according to device preference.
pub(crate) fn build_embedding_backend(
    info: &ModelInfo,
    model_dir: &PathBuf,
    device: &Device,
) -> Result<EmbeddingEngine> {
    if should_use_gpu_for_model(model_dir, device) {
        let detected = get_cached_backend_type();
        if !matches!(detected, BackendType::Cpu) {
            let engine = EmbeddingEngine::new(model_dir, info)?;
            return Ok(engine);
        }
    }

    // Fallback to CPU
    let embedding = EmbeddingEngine::new(model_dir, info)?;
    Ok(embedding)
}

/// Build a reranking-only backend according to device preference.
pub(crate) fn build_rerank_backend(
    info: &ModelInfo,
    model_dir: &PathBuf,
    device: &Device,
) -> Result<RerankEngine> {
    if should_use_gpu_for_model(model_dir, device) {
        let detected = get_cached_backend_type();
        if !matches!(detected, BackendType::Cpu) {
            let engine = RerankEngine::new(model_dir, info)?;
            return Ok(engine);
        }
    }

    // Fallback to CPU
    let rerank = RerankEngine::new(model_dir, info)?;
    Ok(rerank)
}

fn should_use_gpu_for_model(model_dir: &Path, device: &Device) -> bool {
    if matches!(device, Device::Gpu(_)) {
        return true;
    }
    if matches!(device, Device::Cpu) {
        return false;
    }

    // Device::Auto logic: check model parameters
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return true;
    }

    let repo_name = model_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    if let Ok((config, _)) = ModelConfig::load(repo_name, Some(&config_path)) {
        let params = estimate_model_params(&config);
        // Threshold: 80M parameters. Below this, CPU is usually faster for small batches
        // and saves GPU memory.
        if params < 800_000_000 {
            log::debug!("Auto-routing small model ({} params) to CPU", params);
            return false;
        }
    }

    true
}

fn estimate_model_params(config: &ModelConfig) -> usize {
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let layers = config.num_hidden_layers;

    // Estimate parameters:
    // Embeddings: vocab * hidden
    // Layers: layers * 12 * hidden^2 (approximate for Transformer: 4*h^2 attention + 8*h^2 MLP)
    // This is a rough order-of-magnitude estimation.
    let embedding_params = vocab * hidden;
    let layer_params = layers * 12 * hidden * hidden;

    embedding_params + layer_params
}

/// Build a generator backend according to device preference.
pub(crate) fn build_generator_backend(
    info: &ModelInfo,
    model_dir: &PathBuf,
    device: &Device,
) -> Result<EngineBackend> {
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let detected = get_cached_backend_type();
        if !matches!(detected, BackendType::Cpu) {
            let engine = GeneratorEngine::new(model_dir, info)?;
            return Ok(EngineBackend {
                embedding: None,
                rerank: None,
                generator: Some(engine),
            });
        }
    }

    if matches!(device, Device::Cpu | Device::Auto) {
        let generator = GeneratorEngine::new(model_dir, info)?;
        return Ok(EngineBackend {
            embedding: None,
            rerank: None,
            generator: Some(generator),
        });
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}

/// Build an engine backend according to device preference.
pub(crate) fn build_backend(
    info: &ModelInfo,
    model_dir: &PathBuf,
    device: &Device,
) -> Result<EngineBackend> {
    if matches!(info.model_type, ModelType::Generator) {
        return build_generator_backend(info, model_dir, device);
    }

    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let detected = get_cached_backend_type();
        if !matches!(detected, BackendType::Cpu) {
            let embedding = EmbeddingEngine::new(model_dir, info)?;
            let rerank = RerankEngine::new(model_dir, info)?;
            return Ok(EngineBackend {
                embedding: Some(embedding),
                rerank: Some(rerank),
                generator: None,
            });
        }
    }

    if matches!(device, Device::Cpu | Device::Auto) {
        let embedding = EmbeddingEngine::new(model_dir, info)?;
        let rerank = RerankEngine::new(model_dir, info)?;
        return Ok(EngineBackend {
            embedding: Some(embedding),
            rerank: Some(rerank),
            generator: None,
        });
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelManager;
    use crate::types::ClientConfig;

    #[test]
    fn test_real_model_full_routing_small() {
        // Use a very small model as requested: all-MiniLM-L6-v2 (~22M params)
        // This ensures fast "full real" testing without waiting for gigabytes.
        let config = ClientConfig::default();
        let manager = ModelManager::new(config);
        
        let model_id = "sentence-transformers/all-MiniLM-L6-v2";
        println!("Preparing model {}...", model_id);
        
        let artifacts = manager.prepare(model_id).expect("Model preparation failed");
        
        // Test Routing Logic
        let use_gpu = should_use_gpu_for_model(&artifacts.model_dir, &Device::Auto);
        assert_eq!(use_gpu, false, "MiniLM (22M) should route to CPU");
    }

    #[test]
    fn test_real_model_full_routing_large() {
        // Model: Qwen2.5-1.5B (~1.0B Params)
        // Expected: Route to GPU (> 800M)
        let config = ClientConfig::default();
        let manager = ModelManager::new(config);
        
        let model_id = "Qwen/Qwen2.5-1.5B-Instruct";
        println!("Preparing model {}...", model_id);
        
        let artifacts = manager.prepare(model_id).expect("Model preparation failed");
        
        let use_gpu = should_use_gpu_for_model(&artifacts.model_dir, &Device::Auto);
        assert_eq!(use_gpu, true, "Qwen 1.5B (Large) should route to GPU");
    }
}
