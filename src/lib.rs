//! gllm - inference client library (skeleton).

pub mod manifest;

pub mod arch;
pub mod backend;
pub mod engine;
pub mod loader;
pub mod scheduler;

pub mod client;
pub mod embeddings;
pub mod generation;
pub mod kv_cache;
pub mod model_config;
pub mod quantization;
pub mod rerank;
pub mod tokenizer;
pub mod weight_loader;

pub use manifest::{
    FileMap, MoEConfig, ModelArchitecture, ModelKind, ModelManifest, RouterType, TensorNamingRule,
};

// Re-export for convenience
pub use backend::{detect_backend, BackendType};
pub use client::{AsyncClient, Client};
pub use manifest::EMPTY_FILE_MAP;
pub use tokenizer::{TokenizerError, TokenizerHandle};
