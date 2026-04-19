//! gllm - inference client library (skeleton).

pub mod compat;

pub mod guardrail;
pub mod manifest;

pub mod arch;
pub mod backend;
pub mod engine;
pub mod graph;
pub mod loader;
pub mod scheduler;

pub mod client;
pub mod embeddings;
pub mod generation;
pub mod intent;
pub mod knowledge;
pub mod kv_cache;
pub mod model_config;
pub mod quantization;
pub mod classify;
pub mod rerank;
pub mod tokenizer;
pub mod weight_loader;
pub mod weight_names;
pub mod static_compression;

pub mod ffi;

pub mod jit;
pub mod sensors;
pub mod moe;
pub mod speculative;
pub mod early_exit;
pub mod fp8;
pub mod rag;
pub mod routing;
pub mod prefetch;

pub use manifest::{
    ArchFamily, FileMap, MoEConfig, ModelKind, ModelManifest, RouterType,
};

// GGUF Loader API (API-GGUF)
pub use loader::gguf::{GgufReader, TensorInfo, GgufError, GgmlDType, TensorSlice, GgufValueType};

// GGUF Adapter API (API-GGUF-ADAPTER)
pub use loader::adapter::{GgufAdapter, KernelTensorView, StorageFormat, PackedBits};

// Re-export for convenience
pub use backend::{detect_backend, BackendType};
pub use client::{Client, GllmError, ModelInfo};
pub use generation::{
    GenerationChunk, GenerationResponse, GenerationStream,
    GenerationHook, HookDecision, MediaInput, ThresholdHook,
};
pub use compat::multimodal::{
    EncoderMedia, MediaKind, MultimodalContext, MultimodalEncoded, MultimodalEncoder,
    MultimodalTokenIds, RoutedSequence,
};
pub use guardrail::{GuardProbeError, GuardProbeRunner};
pub use embeddings::{Embedding, EmbeddingsResponse, RagResponse};
pub use classify::{ClassifyResponse, ClassificationResult};
pub use rerank::{RerankResponse, RerankResult};
pub use intent::{
    GuardProbe, GuardrailAttachment, IntentConfig, IntentEncoding,
    IntentError, SafetyPolicy, SafetyPolicyConfig,
};
pub use knowledge::{
    KnowledgeSource, HitRateTracker, InjectionKind, InjectionScheduler, KnowledgeDataSource,
    KnowledgeError, KnowledgeInjectionConfig, KnowledgeInjectionResult, KvSideloadManager,
    LayerTarget, MaterializedPayload,
};
pub use manifest::EMPTY_FILE_MAP;
pub use tokenizer::{TokenizerError, TokenizerHandle};

pub use engine::arbiter::{ArbiterHwView, GraphArchetype, InferenceMode, StrategyArbiter, StrategyBias};
pub use graph::profile::{AttentionKind, FfnKind, GraphProfile, GraphProfiler, ResidualKind};
