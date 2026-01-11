use crate::decoder_model::DecoderModel;
use crate::dynamic_bert::{DynamicBertModel, DynamicCrossEncoder};
use crate::generation::{GenerationConfig, GenerationOutput};
use crate::generator_engine::GeneratorEngine;
use crate::model_config::ModelConfig;
use crate::registry::{Architecture, ModelInfo, ModelType};
use crate::types::{Device, Error, Result};
use burn::backend::{Candle, NdArray, Wgpu};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use tokenizers::Tokenizer;

pub(crate) const MAX_SEQ_LEN: usize = 512;

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    pad_token: Option<Value>,
    pad_token_id: Option<i64>,
    eos_token: Option<Value>,
    eos_token_id: Option<i64>,
    bos_token: Option<Value>,
    bos_token_id: Option<i64>,
    unk_token: Option<Value>,
    unk_token_id: Option<i64>,
    vocab_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SpecialTokensMap {
    pad_token: Option<Value>,
    eos_token: Option<Value>,
    bos_token: Option<Value>,
    unk_token: Option<Value>,
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
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
enum EmbeddingModel<B: Backend> {
    Encoder(DynamicBertModel<B>),
    Decoder(DecoderModel<B>),
}

impl<B: Backend> EmbeddingModel<B> {
    fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 2>> {
        match self {
            EmbeddingModel::Encoder(model) => {
                let hidden_states = model.forward(input_ids)?;
                Ok(model.pool_hidden_states(hidden_states))
            }
            EmbeddingModel::Decoder(model) => {
                let hidden_states = model.forward(input_ids.clone())?;
                model.pool_hidden_states(hidden_states, input_ids)
            }
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            EmbeddingModel::Encoder(model) => model.hidden_size(),
            EmbeddingModel::Decoder(model) => model.hidden_size(),
        }
    }

    fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        match self {
            EmbeddingModel::Encoder(model) => model.load_safetensors(safetensors_path),
            EmbeddingModel::Decoder(model) => model.load_safetensors(safetensors_path),
        }
    }
}

/// Embedding encoder built with dynamic BERT model.
#[derive(Clone)]
pub(crate) struct EmbeddingEngine<B: Backend> {
    model: EmbeddingModel<B>,
    device: B::Device,
    pad_id: i64,
}

impl<B: Backend> EmbeddingEngine<B> {
    pub fn new(device: B::Device, model_dir: &Path, info: &ModelInfo) -> Result<Self> {
        // Load model configuration
        let config_path = model_dir.join("config.json");
        let repo_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .replace("--", "/");
        let config_file = config_path.exists().then_some(config_path.as_path());
        let (config, _) = ModelConfig::load(&repo_name, config_file)?;
        let pad_id = config.pad_token_id.unwrap_or(0);

        let mut model = match info.architecture {
            Architecture::Qwen2Embedding | Architecture::MistralEmbedding => {
                EmbeddingModel::Decoder(DecoderModel::new(&device, config)?)
            }
            _ => EmbeddingModel::Encoder(DynamicBertModel::new(&device, config)?),
        };

        // Load weights if available
        let safetensors_path = model_dir.join("model.safetensors");
        if safetensors_path.exists() {
            model.load_safetensors(&safetensors_path)?;
        }

        Ok(Self {
            model,
            device,
            pad_id,
        })
    }

    pub fn embed(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Err(Error::InferenceError(
                "At least one text is required for embeddings".into(),
            ));
        }

        let batch = tokens.len();
        let seq_len = tokens.iter().map(|t| t.len()).max().unwrap_or(1);

        // Pad all token sequences to same length
        let mut flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
        for item in tokens {
            let mut padded = item.clone();
            padded.resize(seq_len, self.pad_id);
            flat.extend_from_slice(&padded);
        }

        // Create input tensor
        let token_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [batch, seq_len]), &self.device);

        // Forward pass through embedding model
        let pooled = self.model.forward(token_tensor)?;

        let data = pooled
            .into_data()
            .into_vec::<f32>()
            .map_err(|err| Error::InferenceError(err.to_string()))?;

        let hidden_size = self.model.hidden_size();
        let mut embeddings = Vec::with_capacity(batch);
        for chunk in data.chunks(hidden_size) {
            let norm = chunk.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            embeddings.push(chunk.iter().map(|v| v / norm).collect());
        }

        Ok(embeddings)
    }

}

/// Cross-encoder rerank engine with dynamic BERT model.
#[derive(Clone)]
pub(crate) struct RerankEngine<B: Backend> {
    model: DynamicCrossEncoder<B>,
    device: B::Device,
}

impl<B: Backend> RerankEngine<B> {
    pub fn new(device: B::Device, model_dir: &Path) -> Result<Self> {
        // Load model configuration
        let config_path = model_dir.join("config.json");
        let repo_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .replace("--", "/");
        let config_file = config_path.exists().then_some(config_path.as_path());
        let (config, _) = ModelConfig::load(&repo_name, config_file)?;

        // Ensure this is configured as a cross-encoder
        let mut config = config;
        if !config.is_cross_encoder() {
            // Force cross-encoder configuration for rerankers
            config.num_labels = Some(1);
            config.classifier_dropout = Some(0.1);
        }

        // Create dynamic cross-encoder model
        let mut model = DynamicCrossEncoder::new(&device, config)?;

        // Load weights if available
        let safetensors_path = model_dir.join("model.safetensors");
        if safetensors_path.exists() {
            model.load_safetensors(&safetensors_path)?;
        }

        Ok(Self { model, device })
    }

    pub fn score(&self, pairs: &[Vec<i64>]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Err(Error::InferenceError(
                "At least one query-document pair is required".into(),
            ));
        }

        let batch = pairs.len();
        let seq_len = pairs.iter().map(|t| t.len()).max().unwrap_or(1);

        // Pad all token sequences to same length
        let mut flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
        for pair in pairs {
            let mut padded = pair.clone();
            padded.resize(seq_len, 0); // pad with 0 (PAD token)
            flat.extend_from_slice(&padded);
        }

        // Create input tensor
        let token_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [batch, seq_len]), &self.device);

        // Forward pass through cross-encoder with sigmoid
        let scores = self.model.forward_with_sigmoid(token_tensor)?;

        // Convert to vector and extract scalar values
        let data = scores
            .into_data()
            .into_vec::<f32>()
            .map_err(|err| Error::InferenceError(err.to_string()))?;

        Ok(data)
    }
}

/// Backend-specific engine bundle.
/// Priority: Wgpu (GPU) -> Candle (CPU with BLAS) -> NdArray (pure Rust CPU)
pub(crate) enum EngineBackend {
    /// GPU backend using WebGPU
    Wgpu {
        embedding: EmbeddingEngine<Wgpu<f32>>,
        rerank: RerankEngine<Wgpu<f32>>,
    },
    /// CPU backend with BLAS acceleration (OpenBLAS/MKL/Accelerate)
    Candle {
        embedding: EmbeddingEngine<Candle<f32, i64>>,
        rerank: RerankEngine<Candle<f32, i64>>,
    },
    /// Pure Rust CPU backend (no external dependencies)
    NdArray {
        embedding: EmbeddingEngine<NdArray<f32>>,
        rerank: RerankEngine<NdArray<f32>>,
    },
    /// Generator backend using WebGPU
    GeneratorWgpu {
        generator: GeneratorEngine<Wgpu<f32>>,
    },
    /// Generator backend using BLAS-accelerated CPU
    GeneratorCandle {
        generator: GeneratorEngine<Candle<f32, i64>>,
    },
    /// Generator backend using pure Rust CPU
    GeneratorNdArray {
        generator: GeneratorEngine<NdArray<f32>>,
    },
}

impl EngineBackend {
    pub fn run_embeddings(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self {
            EngineBackend::Wgpu { embedding, .. } => embedding.embed(tokens),
            EngineBackend::Candle { embedding, .. } => embedding.embed(tokens),
            EngineBackend::NdArray { embedding, .. } => embedding.embed(tokens),
            EngineBackend::GeneratorWgpu { .. }
            | EngineBackend::GeneratorCandle { .. }
            | EngineBackend::GeneratorNdArray { .. } => Err(Error::InvalidConfig(
                "Embeddings are not supported for generator models".into(),
            )),
        }
    }

    pub fn run_rerank(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        match self {
            EngineBackend::Wgpu { rerank, .. } => rerank.score(tokens),
            EngineBackend::Candle { rerank, .. } => rerank.score(tokens),
            EngineBackend::NdArray { rerank, .. } => rerank.score(tokens),
            EngineBackend::GeneratorWgpu { .. }
            | EngineBackend::GeneratorCandle { .. }
            | EngineBackend::GeneratorNdArray { .. } => Err(Error::InvalidConfig(
                "Rerank is not supported for generator models".into(),
            )),
        }
    }

    pub fn run_generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
    ) -> Result<GenerationOutput> {
        match self {
            EngineBackend::GeneratorWgpu { generator } => {
                generator.generate(prompt_ids, config, tokenizer)
            }
            EngineBackend::GeneratorCandle { generator } => {
                generator.generate(prompt_ids, config, tokenizer)
            }
            EngineBackend::GeneratorNdArray { generator } => {
                generator.generate(prompt_ids, config, tokenizer)
            }
            _ => Err(Error::InvalidConfig(
                "Generation is not supported for this model".into(),
            )),
        }
    }

    pub fn max_position_embeddings(&self) -> Option<usize> {
        match self {
            EngineBackend::GeneratorWgpu { generator } => {
                Some(generator.max_position_embeddings())
            }
            EngineBackend::GeneratorCandle { generator } => {
                Some(generator.max_position_embeddings())
            }
            EngineBackend::GeneratorNdArray { generator } => {
                Some(generator.max_position_embeddings())
            }
            _ => None,
        }
    }
}

impl Drop for EngineBackend {
    fn drop(&mut self) {
        // Explicit cleanup for wgpu backend to avoid SIGSEGV on exit
        // Uses retry logic with increasing delays to ensure GPU cleanup threads complete
        if matches!(self, EngineBackend::Wgpu { .. } | EngineBackend::GeneratorWgpu { .. }) {
            log::debug!("Starting wgpu EngineBackend cleanup (retry strategy)");

            // Retry strategy: Multiple shorter sleeps instead of one long sleep
            // This allows GPU cleanup threads to checkpoint progress multiple times
            let retry_attempts = 5;
            let delays = [50, 100, 150, 200, 300]; // milliseconds per attempt

            for (attempt, &delay_ms) in delays.iter().enumerate() {
                log::debug!("wgpu cleanup attempt {}/{}: {}ms delay",
                    attempt + 1, retry_attempts, delay_ms);
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }

            log::debug!("wgpu EngineBackend cleanup completed (total delay: 800ms)");
        }
    }
}

/// Embedding-only engine backend.
/// Priority: Wgpu (GPU) -> Candle (CPU with BLAS) -> NdArray (pure Rust CPU)
pub(crate) enum EmbeddingBackend {
    Wgpu(EmbeddingEngine<Wgpu<f32>>),
    Candle(EmbeddingEngine<Candle<f32, i64>>),
    NdArray(EmbeddingEngine<NdArray<f32>>),
}

impl EmbeddingBackend {
    pub fn embed(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self {
            EmbeddingBackend::Wgpu(engine) => engine.embed(tokens),
            EmbeddingBackend::Candle(engine) => engine.embed(tokens),
            EmbeddingBackend::NdArray(engine) => engine.embed(tokens),
        }
    }
}

/// Reranking-only engine backend.
/// Priority: Wgpu (GPU) -> Candle (CPU with BLAS) -> NdArray (pure Rust CPU)
pub(crate) enum RerankingBackend {
    Wgpu(RerankEngine<Wgpu<f32>>),
    Candle(RerankEngine<Candle<f32, i64>>),
    NdArray(RerankEngine<NdArray<f32>>),
}

impl RerankingBackend {
    pub fn score(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        match self {
            RerankingBackend::Wgpu(engine) => engine.score(tokens),
            RerankingBackend::Candle(engine) => engine.score(tokens),
            RerankingBackend::NdArray(engine) => engine.score(tokens),
        }
    }
}

/// Build an embedding-only backend according to device preference.
/// Priority: Wgpu (GPU) -> Candle (CPU with BLAS) -> NdArray (pure Rust CPU)
pub(crate) fn build_embedding_backend(
    info: &ModelInfo,
    model_dir: &std::path::PathBuf,
    device: &Device,
) -> Result<EmbeddingBackend> {
    use crate::gpu_capabilities::GpuCapabilities;

    let prefer_cpu = std::env::var("GLLM_TEST_MODE").is_ok();

    // Test mode: use pure Rust NdArray backend
    if prefer_cpu {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let embedding = EmbeddingEngine::<NdArray<f32>>::new(ndarray_device, model_dir, info)?;
        return Ok(EmbeddingBackend::NdArray(embedding));
    }

    // Priority 1: Try GPU (Wgpu)
    // CRITICAL: Pre-check GPU availability using wgpu-detect feature.
    // This detects OOM errors BEFORE calling burn/cubecl, which would panic
    // with .unwrap() in cubecl-wgpu. The panic can't be caught when panic=abort.
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        // GpuCapabilities::detect() actually calls request_device() to verify GPU works
        let gpu_caps = GpuCapabilities::detect();

        if !gpu_caps.is_gpu_available() {
            log::warn!(
                "GPU (Wgpu) pre-check failed: {} not usable, skipping GPU initialization",
                gpu_caps.name
            );
        } else {
            // GPU is available, proceed with initialization
            let init = std::panic::catch_unwind(|| {
                let wgpu_device = <Wgpu<f32> as Backend>::Device::default();
                EmbeddingEngine::<Wgpu<f32>>::new(wgpu_device, model_dir, info)
            });

            match init {
                Ok(Ok(engine)) => {
                    // Step 2: Verify GPU compute actually works by running a test embedding
                    // This catches "Encoder is invalid" errors that only occur during real GPU work
                    let compute_test = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        // Run a minimal embedding to verify GPU compute works
                        let test_tokens = vec![vec![101i64, 7592, 102]]; // [CLS] hello [SEP]
                        engine.embed(&test_tokens)
                    }));

                    match compute_test {
                        Ok(Ok(_)) => return Ok(EmbeddingBackend::Wgpu(engine)),
                        Ok(Err(e)) => log::warn!("GPU (Wgpu) compute test failed: {}, falling back to CPU", e),
                        Err(_) => log::warn!("GPU (Wgpu) compute test panicked (Encoder invalid?), falling back to CPU"),
                    }
                }
                Ok(Err(e)) => log::warn!("GPU (Wgpu) initialization failed (likely OOM or config): {}, falling back to CPU", e),
                Err(_) => log::warn!("GPU (Wgpu) initialization panicked (likely OOM), falling back to CPU"),
            }
        }
    }

    // Priority 2: Try Candle (CPU with BLAS acceleration)
    if matches!(device, Device::Cpu | Device::Auto) {
        let candle_init = std::panic::catch_unwind(|| {
            let candle_device = <Candle<f32, i64> as Backend>::Device::default();
            EmbeddingEngine::<Candle<f32, i64>>::new(candle_device, model_dir, info).ok()
        });

        if let Ok(Some(engine)) = candle_init {
            return Ok(EmbeddingBackend::Candle(engine));
        }
    }

    // Priority 3: Fallback to NdArray (pure Rust, no external dependencies)
    if matches!(device, Device::Cpu | Device::Auto) {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let embedding = EmbeddingEngine::<NdArray<f32>>::new(ndarray_device, model_dir, info)?;
        return Ok(EmbeddingBackend::NdArray(embedding));
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}

/// Build a reranking-only backend according to device preference.
/// Priority: Wgpu (GPU) -> Candle (CPU with BLAS) -> NdArray (pure Rust CPU)
pub(crate) fn build_rerank_backend(
    model_dir: &std::path::PathBuf,
    device: &Device,
) -> Result<RerankingBackend> {
    use crate::gpu_capabilities::GpuCapabilities;

    let prefer_cpu = std::env::var("GLLM_TEST_MODE").is_ok();

    // Test mode: use pure Rust NdArray backend
    if prefer_cpu {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let rerank = RerankEngine::<NdArray<f32>>::new(ndarray_device, model_dir)?;
        return Ok(RerankingBackend::NdArray(rerank));
    }

    // Priority 1: Try GPU (Wgpu)
    // CRITICAL: Pre-check GPU availability to avoid panic in cubecl-wgpu
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let gpu_caps = GpuCapabilities::detect();

        if !gpu_caps.is_gpu_available() {
            log::warn!(
                "GPU (Wgpu) pre-check failed for reranker: {} not usable, skipping GPU",
                gpu_caps.name
            );
        } else {
            let init = std::panic::catch_unwind(|| {
                let wgpu_device = <Wgpu<f32> as Backend>::Device::default();
                RerankEngine::<Wgpu<f32>>::new(wgpu_device, model_dir).ok()
            });

            if let Ok(Some(engine)) = init {
                // Verify GPU compute works by running a test score
                let compute_test = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // [CLS] query [SEP] doc [SEP] - a simple pair for testing
                    let test_pairs = vec![vec![101i64, 7592, 102, 2088, 102]];
                    engine.score(&test_pairs)
                }));

                match compute_test {
                    Ok(Ok(_)) => return Ok(RerankingBackend::Wgpu(engine)),
                    Ok(Err(e)) => log::warn!("GPU (Wgpu) rerank compute test failed: {}, falling back to CPU", e),
                    Err(_) => log::warn!("GPU (Wgpu) rerank compute test panicked, falling back to CPU"),
                }
            }
        }
    }

    // Priority 2: Try Candle (CPU with BLAS acceleration)
    if matches!(device, Device::Cpu | Device::Auto) {
        let candle_init = std::panic::catch_unwind(|| {
            let candle_device = <Candle<f32, i64> as Backend>::Device::default();
            RerankEngine::<Candle<f32, i64>>::new(candle_device, model_dir).ok()
        });

        if let Ok(Some(engine)) = candle_init {
            return Ok(RerankingBackend::Candle(engine));
        }
    }

    // Priority 3: Fallback to NdArray (pure Rust, no external dependencies)
    if matches!(device, Device::Cpu | Device::Auto) {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let rerank = RerankEngine::<NdArray<f32>>::new(ndarray_device, model_dir)?;
        return Ok(RerankingBackend::NdArray(rerank));
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}

/// Build a generator backend according to device preference.
/// Priority: Wgpu (GPU) -> Candle (CPU with BLAS) -> NdArray (pure Rust CPU)
pub(crate) fn build_generator_backend(
    info: &ModelInfo,
    model_dir: &std::path::PathBuf,
    device: &Device,
) -> Result<EngineBackend> {
    use crate::gpu_capabilities::GpuCapabilities;

    let prefer_cpu = std::env::var("GLLM_TEST_MODE").is_ok();

    if prefer_cpu {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let generator = GeneratorEngine::<NdArray<f32>>::new(ndarray_device, model_dir, info)?;
        return Ok(EngineBackend::GeneratorNdArray { generator });
    }

    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let gpu_caps = GpuCapabilities::detect();

        if !gpu_caps.is_gpu_available() {
            log::warn!(
                "GPU (Wgpu) pre-check failed for generator: {} not usable, skipping GPU",
                gpu_caps.name
            );
        } else {
            let init = std::panic::catch_unwind(|| {
                let wgpu_device = <Wgpu<f32> as Backend>::Device::default();
                GeneratorEngine::<Wgpu<f32>>::new(wgpu_device, model_dir, info).ok()
            });

            if let Ok(Some(engine)) = init {
                return Ok(EngineBackend::GeneratorWgpu { generator: engine });
            }
        }
    }

    if matches!(device, Device::Cpu | Device::Auto) {
        let candle_init = std::panic::catch_unwind(|| {
            let candle_device = <Candle<f32, i64> as Backend>::Device::default();
            GeneratorEngine::<Candle<f32, i64>>::new(candle_device, model_dir, info).ok()
        });

        if let Ok(Some(engine)) = candle_init {
            return Ok(EngineBackend::GeneratorCandle { generator: engine });
        }
    }

    if matches!(device, Device::Cpu | Device::Auto) {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let generator = GeneratorEngine::<NdArray<f32>>::new(ndarray_device, model_dir, info)?;
        return Ok(EngineBackend::GeneratorNdArray { generator });
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}

/// Build an engine backend according to device preference.
/// Priority: Wgpu (GPU) -> Candle (CPU with BLAS) -> NdArray (pure Rust CPU)
pub(crate) fn build_backend(
    info: &ModelInfo,
    model_dir: &std::path::PathBuf,
    device: &Device,
) -> Result<EngineBackend> {
    let prefer_cpu = std::env::var("GLLM_TEST_MODE").is_ok();

    if matches!(info.model_type, ModelType::Generator) {
        return build_generator_backend(info, model_dir, device);
    }

    // Test mode: use pure Rust NdArray backend
    if prefer_cpu {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let embedding =
            EmbeddingEngine::<NdArray<f32>>::new(ndarray_device.clone(), model_dir, info)?;
        let rerank = RerankEngine::<NdArray<f32>>::new(ndarray_device, model_dir)?;
        return Ok(EngineBackend::NdArray { embedding, rerank });
    }

    // Priority 1: Try GPU (Wgpu)
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let init = std::panic::catch_unwind(|| {
            let wgpu_device = <Wgpu<f32> as Backend>::Device::default();
            let embedding =
                EmbeddingEngine::<Wgpu<f32>>::new(wgpu_device.clone(), model_dir, info);
            let rerank = RerankEngine::<Wgpu<f32>>::new(wgpu_device, model_dir);

            match (embedding, rerank) {
                (Ok(embedding), Ok(rerank)) => Some(EngineBackend::Wgpu { embedding, rerank }),
                _ => None,
            }
        });

        if let Ok(Some(engine)) = init {
            return Ok(engine);
        }
    }

    // Priority 2: Try Candle (CPU with BLAS acceleration)
    if matches!(device, Device::Cpu | Device::Auto) {
        let candle_init = std::panic::catch_unwind(|| {
            let candle_device = <Candle<f32, i64> as Backend>::Device::default();
            let embedding =
                EmbeddingEngine::<Candle<f32, i64>>::new(candle_device.clone(), model_dir, info);
            let rerank = RerankEngine::<Candle<f32, i64>>::new(candle_device, model_dir);

            match (embedding, rerank) {
                (Ok(embedding), Ok(rerank)) => Some(EngineBackend::Candle { embedding, rerank }),
                _ => None,
            }
        });

        if let Ok(Some(engine)) = candle_init {
            return Ok(engine);
        }
    }

    // Priority 3: Fallback to NdArray (pure Rust, no external dependencies)
    if matches!(device, Device::Cpu | Device::Auto) {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let embedding =
            EmbeddingEngine::<NdArray<f32>>::new(ndarray_device.clone(), model_dir, info)?;
        let rerank = RerankEngine::<NdArray<f32>>::new(ndarray_device, model_dir)?;
        return Ok(EngineBackend::NdArray { embedding, rerank });
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}
