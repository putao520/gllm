use crate::decoder_model::DecoderModel;
use crate::dynamic_bert::{DynamicBertModel, DynamicCrossEncoder};
use crate::generation::{GenerationConfig, GenerationOutput};
use crate::generator_engine::GeneratorEngine;
use crate::model_config::ModelConfig;
use crate::registry::{Architecture, ModelInfo, ModelType, Quantization};
use crate::types::{Device, Error, Result};
use gllm_kernels::{detect_backend, BackendType};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

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
pub(crate) struct TokenizerAdapter {
    tokenizer: Option<Tokenizer>,
    vocab_size: usize,
    pad_id: i64,
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
                let vocab_size = tokenizer.get_vocab_size(true);
                let pad_id = resolve_pad_id(
                    Some(&tokenizer),
                    tokenizer_config.as_ref(),
                    special_tokens.as_ref(),
                    vocab_map.as_ref(),
                );
                return Ok(Self {
                    tokenizer: Some(tokenizer),
                    vocab_size,
                    pad_id,
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
            tokenizer: None,
            vocab_size,
            pad_id,
        })
    }

    /// Encode a single string into token ids.
    pub fn encode(&self, text: &str, max_len: usize) -> (Vec<i64>, usize) {
        let limit = max_len.max(1);
        if let Some(tokenizer) = &self.tokenizer {
            if let Ok(encoding) = tokenizer.encode(text, true) {
                let mut ids: Vec<i64> = encoding.get_ids().iter().map(|id| *id as i64).collect();
                let used = ids.len();
                ids.truncate(limit);
                return (self.pad(ids, limit), used);
            }
        }

        // Fallback: hash-based tokenization per word for determinism.
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            ids.push(((hash % self.vocab_size as u64) as i64).max(1));
            if ids.len() >= limit {
                break;
            }
        }
        let used = ids.len();
        (self.pad(ids, limit), used)
    }

    /// Encode a string into token ids without padding.
    pub fn encode_unpadded(&self, text: &str, max_len: usize) -> Vec<i64> {
        let limit = max_len.max(1);
        if let Some(tokenizer) = &self.tokenizer {
            if let Ok(encoding) = tokenizer.encode(text, true) {
                let mut ids: Vec<i64> = encoding.get_ids().iter().map(|id| *id as i64).collect();
                ids.truncate(limit);
                if ids.is_empty() {
                    ids.push(self.pad_id);
                }
                return ids;
            }
        }

        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            ids.push(((hash % self.vocab_size as u64) as i64).max(1));
            if ids.len() >= limit {
                break;
            }
        }
        if ids.is_empty() {
            ids.push(self.pad_id);
        }
        ids
    }

    /// Decode token ids into a string.
    pub fn decode(&self, tokens: &[i64]) -> String {
        if let Some(tokenizer) = &self.tokenizer {
            let ids: Vec<u32> = tokens
                .iter()
                .filter_map(|id| u32::try_from(*id).ok())
                .collect();
            if let Ok(text) = tokenizer.decode(&ids, true) {
                return text;
            }
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
        self.vocab_size + 1
    }

    fn pad(&self, mut ids: Vec<i64>, len: usize) -> Vec<i64> {
        if ids.len() < len {
            ids.resize(len, self.pad_id);
        }
        if ids.is_empty() {
            ids.push(self.pad_id);
        }
        ids
    }
}

#[derive(Clone)]
enum EmbeddingModel {
    Encoder(DynamicBertModel),
    Decoder(DecoderModel),
}

impl EmbeddingModel {
    fn forward(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self {
            EmbeddingModel::Encoder(model) => {
                let hidden_states = model.forward(tokens)?;
                let pooled = model.pool_hidden_states(&hidden_states, tokens);
                Ok(pooled
                    .data
                    .chunks(pooled.cols)
                    .map(|row| row.to_vec())
                    .collect())
            }
            EmbeddingModel::Decoder(model) => {
                let hidden_states = model.forward(tokens)?;
                let pooled = model.pool_hidden_states(&hidden_states, tokens);
                Ok(pooled
                    .data
                    .chunks(pooled.cols)
                    .map(|row| row.to_vec())
                    .collect())
            }
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            EmbeddingModel::Encoder(model) => model.hidden_size(),
            EmbeddingModel::Decoder(model) => model.hidden_size(),
        }
    }
}

#[derive(Clone)]
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

        let mut model = match info.architecture {
            Architecture::Bert
            | Architecture::CrossEncoder
            | Architecture::Qwen3Embedding
            | Architecture::Qwen3Reranker
            | Architecture::JinaV4
            | Architecture::JinaRerankerV3
            | Architecture::NVIDIANemotron
            | Architecture::Gemma3n
            | Architecture::GLM4 => EmbeddingModel::Encoder(DynamicBertModel::new(config.clone())?),
            _ => EmbeddingModel::Decoder(DecoderModel::new(config.clone())?),
        };

        if let Some(model_path) = find_model_file(model_dir, &info.quantization) {
            match &mut model {
                EmbeddingModel::Encoder(enc) => enc.load_safetensors(&model_path)?,
                EmbeddingModel::Decoder(dec) => dec.load_safetensors(&model_path)?,
            }
        }

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

        let mut model = DynamicCrossEncoder::new(config)?;
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
pub(crate) enum EngineBackend {
    Gpu { embedding: EmbeddingEngine, rerank: RerankEngine },
    Cpu { embedding: EmbeddingEngine, rerank: RerankEngine },
    GeneratorGpu { generator: GeneratorEngine },
    GeneratorCpu { generator: GeneratorEngine },
}

impl EngineBackend {
    pub fn run_embeddings(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self {
            EngineBackend::Gpu { embedding, .. } => embedding.embed(tokens),
            EngineBackend::Cpu { embedding, .. } => embedding.embed(tokens),
            EngineBackend::GeneratorGpu { .. } | EngineBackend::GeneratorCpu { .. } => {
                Err(Error::InvalidConfig(
                    "Embeddings are not supported for generator models".into(),
                ))
            }
        }
    }

    pub fn run_rerank(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        match self {
            EngineBackend::Gpu { rerank, .. } => rerank.score(tokens),
            EngineBackend::Cpu { rerank, .. } => rerank.score(tokens),
            EngineBackend::GeneratorGpu { .. } | EngineBackend::GeneratorCpu { .. } => {
                Err(Error::InvalidConfig(
                    "Rerank is not supported for generator models".into(),
                ))
            }
        }
    }

    pub fn run_generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
    ) -> Result<GenerationOutput> {
        match self {
            EngineBackend::GeneratorGpu { generator } => {
                generator.generate(prompt_ids, config, tokenizer)
            }
            EngineBackend::GeneratorCpu { generator } => {
                generator.generate(prompt_ids, config, tokenizer)
            }
            _ => Err(Error::InvalidConfig(
                "Generation is not supported for this model".into(),
            )),
        }
    }

    pub fn max_position_embeddings(&self) -> Option<usize> {
        match self {
            EngineBackend::GeneratorGpu { generator } | EngineBackend::GeneratorCpu { generator } => {
                Some(generator.max_position_embeddings())
            }
            _ => None,
        }
    }
}

/// Embedding-only engine backend.
pub(crate) enum EmbeddingBackend {
    Gpu(EmbeddingEngine),
    Cpu(EmbeddingEngine),
}

impl EmbeddingBackend {
    pub fn embed(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self {
            EmbeddingBackend::Gpu(engine) => engine.embed(tokens),
            EmbeddingBackend::Cpu(engine) => engine.embed(tokens),
        }
    }
}

/// Reranking-only engine backend.
pub(crate) enum RerankingBackend {
    Gpu(RerankEngine),
    Cpu(RerankEngine),
}

impl RerankingBackend {
    pub fn score(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        match self {
            RerankingBackend::Gpu(engine) => engine.score(tokens),
            RerankingBackend::Cpu(engine) => engine.score(tokens),
        }
    }
}

/// Build an embedding-only backend according to device preference.
pub(crate) fn build_embedding_backend(
    info: &ModelInfo,
    model_dir: &PathBuf,
    device: &Device,
) -> Result<EmbeddingBackend> {
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let detected = detect_backend();
        if !matches!(detected, BackendType::Cpu) {
            let engine = EmbeddingEngine::new(model_dir, info)?;
            return Ok(EmbeddingBackend::Gpu(engine));
        }
    }

    if matches!(device, Device::Cpu | Device::Auto) {
        let embedding = EmbeddingEngine::new(model_dir, info)?;
        return Ok(EmbeddingBackend::Cpu(embedding));
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}

/// Build a reranking-only backend according to device preference.
pub(crate) fn build_rerank_backend(
    info: &ModelInfo,
    model_dir: &PathBuf,
    device: &Device,
) -> Result<RerankingBackend> {
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let detected = detect_backend();
        if !matches!(detected, BackendType::Cpu) {
            let engine = RerankEngine::new(model_dir, info)?;
            return Ok(RerankingBackend::Gpu(engine));
        }
    }

    if matches!(device, Device::Cpu | Device::Auto) {
        let rerank = RerankEngine::new(model_dir, info)?;
        return Ok(RerankingBackend::Cpu(rerank));
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}

/// Build a generator backend according to device preference.
pub(crate) fn build_generator_backend(
    info: &ModelInfo,
    model_dir: &PathBuf,
    device: &Device,
) -> Result<EngineBackend> {
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let detected = detect_backend();
        if !matches!(detected, BackendType::Cpu) {
            let engine = GeneratorEngine::new(model_dir, info)?;
            return Ok(EngineBackend::GeneratorGpu { generator: engine });
        }
    }

    if matches!(device, Device::Cpu | Device::Auto) {
        let generator = GeneratorEngine::new(model_dir, info)?;
        return Ok(EngineBackend::GeneratorCpu { generator });
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
        let detected = detect_backend();
        if !matches!(detected, BackendType::Cpu) {
            let embedding = EmbeddingEngine::new(model_dir, info)?;
            let rerank = RerankEngine::new(model_dir, info)?;
            return Ok(EngineBackend::Gpu { embedding, rerank });
        }
    }

    if matches!(device, Device::Cpu | Device::Auto) {
        let embedding = EmbeddingEngine::new(model_dir, info)?;
        let rerank = RerankEngine::new(model_dir, info)?;
        return Ok(EngineBackend::Cpu { embedding, rerank });
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}
