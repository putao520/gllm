//! gllm: Pure Rust embeddings and rerank library built on Burn.

mod bert_variants;
mod client;
mod dynamic_bert;
mod embeddings;
mod engine;
mod handle;
mod model;
mod model_config;
mod model_presets;
mod performance_optimizer;
mod pooling;
mod registry;
mod rerank;
mod types;

pub use client::Client;
pub use embeddings::EmbeddingsBuilder;
pub use handle::{EmbedderHandle, RerankerHandle};
pub use registry::{Architecture, ModelInfo, ModelRegistry, ModelType};
pub use rerank::RerankBuilder;
pub use types::{
    ClientConfig, Device, Embedding, EmbeddingResponse, Error, GraphCodeInput, RerankResponse, RerankResult,
    Result, Usage,
};
