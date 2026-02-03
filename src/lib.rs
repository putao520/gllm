//! gllm - inference client library (skeleton).

pub mod manifest;
pub mod registry;

pub mod adapter;
pub mod engine;
pub mod loader;
pub mod backend;
pub mod scheduler;

pub mod client;
pub mod embeddings;
pub mod rerank;
pub mod generation;
pub mod kv_cache;
pub mod weight_loader;
pub mod model_config;
pub mod tokenizer;
pub mod quantization;

pub use manifest::{
    FileMap, KnownModel, ModelArchitecture, ModelManifest, MoEConfig, RouterType,
    TensorNamingRule,
};

// Re-export EMPTY_FILE_MAP for convenience
pub use manifest::EMPTY_FILE_MAP;
pub use client::{AsyncClient, Client};
pub use adapter::{Message, Role};
pub use tokenizer::{TokenizerError, TokenizerHandle};
pub use backend::{BackendType, detect_backend};
