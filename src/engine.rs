use crate::model::default_hidden_size;
use crate::registry::ModelInfo;
use crate::types::{Device, Error, Result};
#[cfg(feature = "cpu")]
use burn::backend::NdArray;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Sigmoid,
};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use tokenizers::Tokenizer;

pub(crate) const MAX_SEQ_LEN: usize = 128;
const EMBEDDING_OUTPUT: usize = 128;

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
        merged.push(1); // CLS-like token for separation.
        merged.append(&mut query_tokens);
        merged.push(2); // SEP-like token.
        merged.append(&mut doc_tokens);
        merged.truncate(max_len);

        let used = used_query + used_doc + 2;
        (self.pad(merged, max_len), used)
    }

    /// Vocabulary size for embedding tables.
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

/// Embedding encoder built with Burn primitives.
#[derive(Clone)]
pub(crate) struct EmbeddingEngine<B: Backend> {
    embedding: Embedding<B>,
    attention: MultiHeadAttention<B>,
    projection: Linear<B>,
    norm: LayerNorm<B>,
    output_dim: usize,
    device: B::Device,
}

impl<B: Backend> EmbeddingEngine<B> {
    pub fn new(device: B::Device, vocab_size: usize, hidden: usize, output_dim: usize) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, hidden).init(&device);
        let attention = MultiHeadAttentionConfig::new(hidden, 4)
            .with_dropout(0.05)
            .init(&device);
        let norm = LayerNormConfig::new(hidden).init(&device);
        let projection = LinearConfig::new(hidden, output_dim).init(&device);
        B::seed(&device, 42);

        Self {
            embedding,
            attention,
            projection,
            norm,
            output_dim,
            device,
        }
    }

    pub fn embed(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Err(Error::InferenceError(
                "At least one text is required for embeddings".into(),
            ));
        }
        let batch = tokens.len();
        let seq_len = tokens.iter().map(|t| t.len()).max().unwrap_or(1);
        let mut flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
        for item in tokens {
            if item.len() != seq_len {
                return Err(Error::InvalidConfig("Mismatched token lengths".into()));
            }
            flat.extend_from_slice(item);
        }

        let token_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [batch, seq_len]), &self.device);
        let embedded = self.embedding.forward(token_tensor);
        let attn_out = self.attention.forward(MhaInput::self_attn(embedded));
        let normalized = self.norm.forward(attn_out.context);
        let pooled = normalized.mean_dim(1);
        let projected = self.projection.forward(pooled);
        let data = projected
            .into_data()
            .into_vec::<f32>()
            .map_err(|err| Error::InferenceError(err.to_string()))?;

        let mut result = Vec::with_capacity(batch);
        for chunk in data.chunks(self.output_dim) {
            let norm = chunk.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            result.push(chunk.iter().map(|v| v / norm).collect());
        }
        Ok(result)
    }
}

/// Cross-encoder rerank engine with simple attention + scoring head.
#[derive(Clone)]
pub(crate) struct RerankEngine<B: Backend> {
    embedding: Embedding<B>,
    attention: MultiHeadAttention<B>,
    norm: LayerNorm<B>,
    scorer: Linear<B>,
    sigmoid: Sigmoid,
    device: B::Device,
}

impl<B: Backend> RerankEngine<B> {
    pub fn new(device: B::Device, vocab_size: usize, hidden: usize) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, hidden).init(&device);
        let attention = MultiHeadAttentionConfig::new(hidden, 2)
            .with_dropout(0.1)
            .init(&device);
        let norm = LayerNormConfig::new(hidden).init(&device);
        let scorer = LinearConfig::new(hidden, 1).init(&device);
        B::seed(&device, 7);

        Self {
            embedding,
            attention,
            norm,
            scorer,
            sigmoid: Sigmoid::new(),
            device,
        }
    }

    pub fn score(&self, pairs: &[Vec<i64>]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Err(Error::InferenceError(
                "At least one query-document pair is required".into(),
            ));
        }
        let batch = pairs.len();
        let seq_len = pairs.iter().map(|t| t.len()).max().unwrap_or(1);
        let mut flat = Vec::with_capacity(batch * seq_len);
        for pair in pairs {
            if pair.len() != seq_len {
                return Err(Error::InvalidConfig(
                    "Rerank token sequences must be equal length".into(),
                ));
            }
            flat.extend_from_slice(pair);
        }

        let token_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [batch, seq_len]), &self.device);
        let embedded = self.embedding.forward(token_tensor);
        let attn_out = self.attention.forward(MhaInput::self_attn(embedded));
        let normalized = self.norm.forward(attn_out.context);
        let pooled = normalized.mean_dim(1);
        let logits = self.scorer.forward(pooled);
        let scores = self.sigmoid.forward(logits);

        // Flatten the [batch, 1] tensor to [batch] by extracting all values
        let data = scores.into_data();
        let values: Vec<f32> = data
            .into_vec::<f32>()
            .map_err(|err| Error::InferenceError(err.to_string()))?;
        Ok(values)
    }
}

/// Backend-specific engine bundle.
pub(crate) enum EngineBackend {
    #[cfg(feature = "wgpu")]
    Wgpu {
        embedding: EmbeddingEngine<Wgpu<f32>>,
        rerank: RerankEngine<Wgpu<f32>>,
    },
    #[cfg(feature = "cpu")]
    Cpu {
        embedding: EmbeddingEngine<NdArray<f32>>,
        rerank: RerankEngine<NdArray<f32>>,
    },
}

impl EngineBackend {
    pub fn run_embeddings(&self, tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        match self {
            #[cfg(feature = "wgpu")]
            EngineBackend::Wgpu { embedding, .. } => embedding.embed(tokens),
            #[cfg(feature = "cpu")]
            EngineBackend::Cpu { embedding, .. } => embedding.embed(tokens),
        }
    }

    pub fn run_rerank(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        match self {
            #[cfg(feature = "wgpu")]
            EngineBackend::Wgpu { rerank, .. } => rerank.score(tokens),
            #[cfg(feature = "cpu")]
            EngineBackend::Cpu { rerank, .. } => rerank.score(tokens),
        }
    }
}

/// Build an engine backend according to device preference and available features.
pub(crate) fn build_backend(
    info: &ModelInfo,
    tokenizer: &TokenizerAdapter,
    device: &Device,
) -> Result<EngineBackend> {
    let hidden = default_hidden_size(info.architecture);

    #[cfg(feature = "wgpu")]
    if matches!(device, Device::Gpu(_) | Device::Auto) {
        let init = std::panic::catch_unwind(|| {
            let wgpu_device = <Wgpu<f32> as Backend>::Device::default();
            let embedding = EmbeddingEngine::<Wgpu<f32>>::new(
                wgpu_device.clone(),
                tokenizer.vocab_size(),
                hidden,
                EMBEDDING_OUTPUT,
            );
            let rerank =
                RerankEngine::<Wgpu<f32>>::new(wgpu_device, tokenizer.vocab_size(), hidden);
            EngineBackend::Wgpu { embedding, rerank }
        });
        if let Ok(engine) = init {
            return Ok(engine);
        }
    }

    #[cfg(feature = "cpu")]
    {
        if matches!(device, Device::Cpu | Device::Auto) {
            let ndarray_device = <NdArray<f32> as Backend>::Device::default();
            let embedding = EmbeddingEngine::<NdArray<f32>>::new(
                ndarray_device.clone(),
                tokenizer.vocab_size(),
                hidden,
                EMBEDDING_OUTPUT,
            );
            let rerank =
                RerankEngine::<NdArray<f32>>::new(ndarray_device, tokenizer.vocab_size(), hidden);
            return Ok(EngineBackend::Cpu { embedding, rerank });
        }
    }

    Err(Error::InvalidConfig(
        "No compatible backend available; enable `wgpu` or `cpu` feature".into(),
    ))
}
