use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Total tokens processed.
    pub total_tokens: usize,
}

/// Input for GraphCodeBERT containing source code and data flow graph info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphCodeInput {
    /// Source code text.
    pub code: String,
    /// Position indices mapping (if using custom positions).
    pub position_ids: Option<Vec<i64>>,
    /// Data flow graph attention mask (allows focusing on data dependencies).
    pub dfg_mask: Option<Vec<Vec<i64>>>,
}

/// Single embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Index of the input text.
    pub index: usize,
    /// Embedding vector values.
    pub embedding: Vec<f32>,
}

/// Embedding response containing all vectors and usage stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The list of embedding vectors.
    pub embeddings: Vec<Embedding>,
    /// Usage statistics for the request.
    pub usage: Usage,
}

/// Single rerank result entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Index of the document in the original list.
    pub index: usize,
    /// Relevance score after reranking.
    pub score: f32,
    /// Optional original document text.
    pub document: Option<String>,
}

/// Rerank response containing sorted results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Sorted rerank results.
    pub results: Vec<RerankResult>,
}

/// Device selection for inference.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum Device {
    /// Auto-select best available backend (prefer GPU).
    #[default]
    Auto,
    /// Specific GPU index (backend dependent).
    Gpu(usize),
    /// Force CPU backend.
    Cpu,
}

/// Client configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Directory where models are stored locally.
    pub models_dir: PathBuf,
    /// Desired device for inference.
    pub device: Device,
}

impl Default for ClientConfig {
    fn default() -> Self {
        let models_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".gllm")
            .join("models");
        Self {
            models_dir,
            device: Device::Auto,
        }
    }
}

impl ClientConfig {
    /// Returns the model directory as a path reference.
    pub fn model_dir(&self) -> &Path {
        &self.models_dir
    }
}

/// gllm error type.
#[derive(Debug, Error)]
pub enum Error {
    /// Model not found in registry or local storage.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Model download failed.
    #[error("Failed to download model: {0}")]
    DownloadError(String),

    /// Model loading failed.
    #[error("Failed to load model: {0}")]
    LoadError(String),

    /// Inference execution error.
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// Out of memory error (GPU VRAM or system memory exhausted).
    /// This error triggers automatic fallback to CPU backend.
    #[error("Out of memory: {0}")]
    Oom(String),

    /// Invalid configuration provided.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Wrapped IO error.
    #[error("IO error: {0}")]
    Io(std::io::Error),

    /// Internal error (e.g., task join error in async operations).
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl Error {
    /// Check if this error indicates an out-of-memory condition.
    pub fn is_oom(&self) -> bool {
        match self {
            Error::Oom(_) => true,
            Error::InferenceError(msg) | Error::InternalError(msg) => {
                let msg_lower = msg.to_lowercase();
                msg_lower.contains("oom")
                    || msg_lower.contains("out of memory")
                    || msg_lower.contains("allocate")
                    || msg_lower.contains("memory")
                    || msg_lower.contains("buffer")
            }
            _ => false,
        }
    }
}

impl Clone for Error {
    fn clone(&self) -> Self {
        match self {
            Error::ModelNotFound(s) => Error::ModelNotFound(s.clone()),
            Error::DownloadError(s) => Error::DownloadError(s.clone()),
            Error::LoadError(s) => Error::LoadError(s.clone()),
            Error::InferenceError(s) => Error::InferenceError(s.clone()),
            Error::Oom(s) => Error::Oom(s.clone()),
            Error::InvalidConfig(s) => Error::InvalidConfig(s.clone()),
            Error::Io(io_err) => Error::Io(std::io::Error::new(io_err.kind(), io_err.to_string())),
            Error::InternalError(s) => Error::InternalError(s.clone()),
        }
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::LoadError(s)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

/// Result alias for gllm operations.
pub type Result<T> = std::result::Result<T, Error>;
