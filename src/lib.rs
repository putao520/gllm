//! gllm: Pure Rust embeddings and rerank library built on Burn.

mod client;
mod embeddings;
mod engine;
mod model;
mod registry;
mod rerank;
mod types;

#[cfg(feature = "async")]
pub use client::AsyncClient;
pub use client::Client;
#[cfg(feature = "async")]
pub use embeddings::AsyncEmbeddingsBuilder;
pub use embeddings::EmbeddingsBuilder;
pub use registry::{Architecture, ModelType};
#[cfg(feature = "async")]
pub use rerank::AsyncRerankBuilder;
pub use rerank::RerankBuilder;
pub use types::{
    ClientConfig, Device, Embedding, EmbeddingResponse, Error, RerankResponse, RerankResult,
    Result, Usage,
};
