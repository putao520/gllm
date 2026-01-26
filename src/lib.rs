//! gllm: Pure Rust embeddings and rerank library built on gllm-kernels.
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

pub mod distributed;
mod bert_variants;
mod client;
pub mod decoder_model;
mod dynamic_bert;
mod embeddings;
pub mod engram;
mod engine;
mod fallback;
pub mod generation;
mod generator_engine;
pub mod hooks;
pub mod gguf;
pub mod quantized;
pub mod awq;
mod handle;
#[cfg(feature = "paged-attention")]
pub mod paged_attention;
mod model;
mod model_config;
mod model_presets;
mod performance_optimizer;
mod pooling;
mod registry;
mod rerank;
mod tensor;
pub mod sampler;
mod types;
pub mod parallel_parser;
pub mod weight_loader;

// 基于新 Backend trait API 的核心模块
pub mod kv_cache;
pub mod rms_norm;
pub mod causal_attention;
pub mod decoder_layer;
pub mod generator_model;
pub mod moe_layer;
pub mod moe_generator_model;

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
pub use parallel_parser::{LoadConfig, LoadProgress, ProgressStage};
pub use types::{
    ClientConfig, Device, Embedding, EmbeddingResponse, Error, GraphCodeInput, RerankResponse, RerankResult,
    Result, Usage,
};
// Engram conditional memory exports
pub use engram::{EngramModule, SharedEngram};

// Inference hooks exports
pub use hooks::{InferenceHook, HookManager, LastHiddenStateCollector, CollectedState};

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

// Raw slice-based operations from gllm-kernels
pub use gllm_kernels::{
    // RoPE (Rotary Position Embedding) - raw slice API
    RoPEConfig as KernelRoPEConfig, rope_precompute, rope_apply, rope_apply_inplace,
    // Sampling operations - raw slice API
    SamplingConfig as KernelSamplingConfig, TopKResult,
    topk, apply_temperature, softmax_1d, apply_top_p, sample_tokens, argmax,
    // MoE (Mixture-of-Experts) routing - raw slice API
    MoERoutingConfig as KernelMoERoutingConfig, MoERoutingResult,
    moe_route, compute_routing_logits, compute_expert_load, compute_load_balance_loss,
};
