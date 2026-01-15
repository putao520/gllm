//! gllm: Pure Rust embeddings and rerank library built on Burn.
//!
//! ## Features
//!
//! - **Embedding**: Text-to-vector conversion using BERT-based models
//! - **Reranking**: Cross-encoder based document reranking
//! - **Runtime Fallback**: Automatic GPU→CPU fallback on memory errors
//! - **Multi-backend**: CUDA → ROCm → Metal → WGPU → CPU (auto-detect priority)
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

pub mod attention;
pub mod distributed;
mod bert_variants;
pub mod causal_attention;
pub mod flash_attention;
mod client;
pub mod decoder_layer;
pub mod decoder_model;
mod dynamic_bert;
mod embeddings;
pub mod engram;
mod engine;
mod fallback;
pub mod generation;
mod generator_engine;
pub mod generator_model;
#[cfg(feature = "quantized")]
pub mod gguf;
#[cfg(feature = "quantized")]
pub mod quantized;
#[cfg(feature = "quantized")]
pub mod awq;
#[cfg(feature = "quantized")]
pub mod quantized_ops;
pub mod moe_decoder_layer;
pub mod moe_generator_model;
pub mod moe_layer;
mod handle;
pub mod kv_cache;
#[cfg(feature = "paged-attention")]
pub mod paged_attention;
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
pub mod weight_loader;

pub use client::Client;
pub use embeddings::EmbeddingsBuilder;
pub use fallback::FallbackEmbedder;
pub use generation::{FinishReason, GenerationBuilder, GenerationConfig, GenerationOutput};
// Re-export backend detection from gllm-kernels
pub use gllm_kernels::runtime_detection::{BackendType, BackendDetectionResult, DeviceInfo, GpuCapabilities};
pub use gllm_kernels::{detect_backend, redetect_backend};
pub use handle::{EmbedderHandle, RerankerHandle};
pub use registry::{Architecture, ModelInfo, ModelRegistry, ModelType, Quantization};
pub use rerank::RerankBuilder;
pub use types::{
    ClientConfig, Device, Embedding, EmbeddingResponse, Error, GraphCodeInput, RerankResponse, RerankResult,
    Result, Usage,
};
// Engram conditional memory exports
pub use engram::{EngramModule, SharedEngram};

// Embedding quantization and fast similarity search exports from gllm-kernels
pub use gllm_kernels::{
    // Binary Quantization (1-bit, 32x compression, POPCNT similarity)
    BinaryIpConfig, pack_binary_f32, binary_ip_hamming, binary_ip_hamming_simd, binary_ip_asymmetric,
    // Int8 Quantization (4x compression, near-lossless)
    Int8DotConfig, quantize_to_int8, int8_dot_product, int8_dot_product_unrolled,
    // Int4 Packed Quantization (8x compression)
    Int4PackedConfig, pack_int4, unpack_int4, quantize_to_int4_packed, int4_packed_dot_product,
    // Matryoshka dimension truncation (runtime dim selection)
    MatryoshkaConfig, matryoshka_truncate, select_matryoshka_dim,
    // Three-stage rerank pipeline
    RerankPipelineConfig, RerankResult as KernelRerankResult, rerank_binary_stage, rerank_int8_stage,
};
