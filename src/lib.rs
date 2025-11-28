//! gllm: Pure Rust embeddings and rerank library built on Burn.

mod bert_variants;
mod client;
mod dynamic_bert;
mod embeddings;
mod engine;
mod model;
mod model_config;
mod model_presets;
mod performance_optimizer;
mod pooling;
mod registry;
mod rerank;
mod types;

#[cfg(feature = "async")]
pub use client::AsyncClient;
pub use client::Client;
#[cfg(feature = "async")]
pub use embeddings::AsyncEmbeddingsBuilder;
pub use embeddings::EmbeddingsBuilder;
pub use registry::{Architecture, ModelInfo, ModelRegistry, ModelType};
#[cfg(feature = "async")]
pub use rerank::AsyncRerankBuilder;
pub use rerank::RerankBuilder;
pub use types::{
    ClientConfig, Device, Embedding, EmbeddingResponse, Error, RerankResponse, RerankResult,
    Result, Usage,
};
