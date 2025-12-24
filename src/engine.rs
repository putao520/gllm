use crate::dynamic_bert::{DynamicBertModel, DynamicCrossEncoder};
use crate::model_config::ModelConfig;
use crate::registry::ModelInfo;
use crate::types::{Device, Error, Result};
use burn::backend::{Candle, NdArray, Wgpu};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use tokenizers::Tokenizer;

pub(crate) const MAX_SEQ_LEN: usize = 512;

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
        if tokenizer_path.exists() {
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|err| Error::LoadError(err.to_string()))?;
            let pad_id = tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);
            let vocab_size = tokenizer.get_vocab_size(true);
            return Ok(Self {
                tokenizer: Some(tokenizer),
                vocab_size,
                pad_id: pad_id as i64,
            });
        }

        Ok(Self {
            tokenizer: None,
            vocab_size: 32_000,
            pad_id: 0,
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

/// Embedding encoder built with dynamic BERT model.
#[derive(Clone)]
pub(crate) struct EmbeddingEngine<B: Backend> {
    model: DynamicBertModel<B>,
    device: B::Device,
}

impl<B: Backend> EmbeddingEngine<B> {
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

        // Create dynamic BERT model
        let mut model = DynamicBertModel::new(&device, config)?;

        // Load weights if available
        let safetensors_path = model_dir.join("model.safetensors");
        if safetensors_path.exists() {
            model.load_safetensors(&safetensors_path)?;
        }

        Ok(Self { model, device })
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
            padded.resize(seq_len, 0); // pad with 0 (PAD token)
            flat.extend_from_slice(&padded);
        }

        // Create input tensor
        let token_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [batch, seq_len]), &self.device);

        // Forward pass through BERT
        let hidden_states = self.model.forward(token_tensor)?;
        let pooled = self.model.pool_hidden_states(hidden_states);

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
}

impl EngineBackend {
    pub fn run_embeddings(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self {
            EngineBackend::Wgpu { embedding, .. } => embedding.embed(tokens),
            EngineBackend::Candle { embedding, .. } => embedding.embed(tokens),
            EngineBackend::NdArray { embedding, .. } => embedding.embed(tokens),
        }
    }

    pub fn run_rerank(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        match self {
            EngineBackend::Wgpu { rerank, .. } => rerank.score(tokens),
            EngineBackend::Candle { rerank, .. } => rerank.score(tokens),
            EngineBackend::NdArray { rerank, .. } => rerank.score(tokens),
        }
    }

}

impl Drop for EngineBackend {
    fn drop(&mut self) {
        // Explicit cleanup for wgpu backend to avoid SIGSEGV on exit
        // Uses retry logic with increasing delays to ensure GPU cleanup threads complete
        if matches!(self, EngineBackend::Wgpu { .. }) {
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
    model_dir: &std::path::PathBuf,
    device: &Device,
) -> Result<EmbeddingBackend> {
    let prefer_cpu = std::env::var("GLLM_TEST_MODE").is_ok();

    // Test mode: use pure Rust NdArray backend
    if prefer_cpu {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let embedding = EmbeddingEngine::<NdArray<f32>>::new(ndarray_device, model_dir)?;
        return Ok(EmbeddingBackend::NdArray(embedding));
    }

    // Priority 1: Try GPU (Wgpu)
    // Supports concurrent model loading; if VRAM is insufficient, initialization will fail
    // or panic, triggering the fallback to CPU (Candle).
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let init = std::panic::catch_unwind(|| {
            let wgpu_device = <Wgpu<f32> as Backend>::Device::default();
            EmbeddingEngine::<Wgpu<f32>>::new(wgpu_device, model_dir)
        });

        match init {
            Ok(Ok(engine)) => return Ok(EmbeddingBackend::Wgpu(engine)),
            Ok(Err(e)) => log::warn!("GPU (Wgpu) initialization failed (likely OOM or config): {}, falling back to CPU", e),
            Err(_) => log::warn!("GPU (Wgpu) initialization panicked (likely OOM), falling back to CPU"),
        }
    }

    // Priority 2: Try Candle (CPU with BLAS acceleration)
    if matches!(device, Device::Cpu | Device::Auto) {
        let candle_init = std::panic::catch_unwind(|| {
            let candle_device = <Candle<f32, i64> as Backend>::Device::default();
            EmbeddingEngine::<Candle<f32, i64>>::new(candle_device, model_dir).ok()
        });

        if let Ok(Some(engine)) = candle_init {
            return Ok(EmbeddingBackend::Candle(engine));
        }
    }

    // Priority 3: Fallback to NdArray (pure Rust, no external dependencies)
    if matches!(device, Device::Cpu | Device::Auto) {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let embedding = EmbeddingEngine::<NdArray<f32>>::new(ndarray_device, model_dir)?;
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
    let prefer_cpu = std::env::var("GLLM_TEST_MODE").is_ok();

    // Test mode: use pure Rust NdArray backend
    if prefer_cpu {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let rerank = RerankEngine::<NdArray<f32>>::new(ndarray_device, model_dir)?;
        return Ok(RerankingBackend::NdArray(rerank));
    }

    // Priority 1: Try GPU (Wgpu)
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let init = std::panic::catch_unwind(|| {
            let wgpu_device = <Wgpu<f32> as Backend>::Device::default();
            RerankEngine::<Wgpu<f32>>::new(wgpu_device, model_dir).ok()
        });

        if let Ok(Some(engine)) = init {
            return Ok(RerankingBackend::Wgpu(engine));
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

/// Build an engine backend according to device preference.
/// Priority: Wgpu (GPU) -> Candle (CPU with BLAS) -> NdArray (pure Rust CPU)
pub(crate) fn build_backend(
    _info: &ModelInfo,
    model_dir: &std::path::PathBuf,
    device: &Device,
) -> Result<EngineBackend> {
    let prefer_cpu = std::env::var("GLLM_TEST_MODE").is_ok();

    // Test mode: use pure Rust NdArray backend
    if prefer_cpu {
        let ndarray_device = <NdArray<f32> as Backend>::Device::default();
        let embedding = EmbeddingEngine::<NdArray<f32>>::new(ndarray_device.clone(), model_dir)?;
        let rerank = RerankEngine::<NdArray<f32>>::new(ndarray_device, model_dir)?;
        return Ok(EngineBackend::NdArray { embedding, rerank });
    }

    // Priority 1: Try GPU (Wgpu)
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let init = std::panic::catch_unwind(|| {
            let wgpu_device = <Wgpu<f32> as Backend>::Device::default();
            let embedding = EmbeddingEngine::<Wgpu<f32>>::new(wgpu_device.clone(), model_dir);
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
            let embedding = EmbeddingEngine::<Candle<f32, i64>>::new(candle_device.clone(), model_dir);
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
            EmbeddingEngine::<NdArray<f32>>::new(ndarray_device.clone(), model_dir)?;
        let rerank = RerankEngine::<NdArray<f32>>::new(ndarray_device, model_dir)?;
        return Ok(EngineBackend::NdArray { embedding, rerank });
    }

    Err(Error::InvalidConfig(
        "No compatible backend available".into(),
    ))
}
