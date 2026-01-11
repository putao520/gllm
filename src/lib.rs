//! gllm: Pure Rust embeddings and rerank library built on Burn.
//!
//! ## Features
//!
//! - **Embedding**: Text-to-vector conversion using BERT-based models
//! - **Reranking**: Cross-encoder based document reranking
//! - **Runtime Fallback**: Automatic GPU→CPU fallback on memory errors
//! - **Multi-backend**: GPU (Wgpu), CPU+BLAS (Candle), Pure Rust (NdArray)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use gllm::{FallbackEmbedder, Device};
//!
//! // Create embedder with automatic GPU→CPU fallback
//! let embedder = FallbackEmbedder::new("bge-small-en").await?;
//!
//! // Embed text - automatically falls back to CPU if GPU OOMs
//! let vector = embedder.embed("Hello world").await?;
//! ```

mod bert_variants;
pub mod causal_attention;
mod client;
pub mod decoder_layer;
pub mod decoder_model;
mod dynamic_bert;
mod embeddings;
mod engine;
mod fallback;
pub mod generation;
mod generator_engine;
pub mod generator_model;
mod gpu_capabilities;
mod handle;
pub mod kv_cache;
mod model;
mod model_config;
mod model_presets;
mod performance_optimizer;
mod pooling;
mod registry;
mod rerank;
mod rope;
pub mod rms_norm;
pub mod sampler;
mod types;

pub use client::Client;
pub use embeddings::EmbeddingsBuilder;
pub use fallback::FallbackEmbedder;
pub use generation::{FinishReason, GenerationBuilder, GenerationConfig, GenerationOutput};
pub use gpu_capabilities::{GpuCapabilities, GpuType};
pub use handle::{EmbedderHandle, RerankerHandle};
pub use registry::{Architecture, ModelInfo, ModelRegistry, ModelType, Quantization};
pub use rerank::RerankBuilder;
pub use types::{
    ClientConfig, Device, Embedding, EmbeddingResponse, Error, GraphCodeInput, RerankResponse, RerankResult,
    Result, Usage,
};
