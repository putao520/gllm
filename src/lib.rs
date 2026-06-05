//! gllm - inference client library (skeleton).

pub mod compat;

pub mod manifest;

pub mod arch;
pub mod backend;
pub mod engine;
pub mod graph;
pub mod loader;
pub mod scheduler;

pub mod client;
pub mod cot_reasoner;
pub mod embeddings;
pub mod generation;
pub mod guardrail;
pub mod head_routing;
pub mod intent;
pub mod intent_tracker;
pub mod kv_cache;
pub mod model_config;
pub mod quantization;
pub mod classify;
pub mod rerank;
pub mod semantic_gatekeeper;
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
pub use client::{Client, GllmError, ModelInfo, MtpGenerationResponse, MtpStepInfo};
pub use generation::{
    GenerationChunk, GenerationResponse, GenerationStream,
    GenerationHook, HookDecision, MediaInput, ThresholdHook,
};
pub use compat::multimodal::{
    EncoderMedia, MediaKind, MultimodalContext, MultimodalEncoded, MultimodalEncoder,
    MultimodalTokenIds, RoutedSequence,
};
pub use compat::audio_forward::{
    audio_encode, mel_spectrogram, AudioConfig, AudioTensorLookup, InMemoryAudioWeights,
    UsmConformerEncoder,
};
pub use cot_reasoner::{
    ReasoningBuilder, ReasoningError, ReasoningMode, ReasoningResponse, ReasoningStepHook,
    ReasoningStopReason, ReasoningTemplate, StepAction, StepContext, StepKnowledge, StepResult,
    DEFAULT_STOP_PATTERNS,
};
pub use embeddings::{Embedding, EmbeddingsResponse, RagResponse};
pub use classify::{ClassifyResponse, ClassificationResult};
pub use rerank::{RerankResponse, RerankResult};
pub use semantic_gatekeeper::{
    AstContext, AstSentinel, KnowledgeEntry, KnowledgeProvider, RetrieveContext,
    SemanticGatekeeperCallback, SemanticGatekeeperConfig, SemanticGatekeeperError, SemanticLevel,
    TokenizerLookup, DEFAULT_LEVEL_DESCRIPTORS,
};
pub use head_routing::{
    ClassifyBinaryConfig, ClassifyMultiwayConfig, HeadRoutingError, LayerAnchor, PoolMode,
};
pub use guardrail::{
    GuardProbe, GuardProbeWeights, GuardrailAttachment, GuardrailError, SafetyPolicy,
};
pub use intent::{IntentEncoding, IntentError};
pub use intent_tracker::{
    classify_conversation_turn, Classification, IntentTracker, TaskType, TrackerError,
    TrackerTurnInput,
};
pub use manifest::EMPTY_FILE_MAP;
pub use tokenizer::{TokenizerError, TokenizerHandle};

pub use engine::arbiter::{ArbiterHwView, GraphArchetype, InferenceMode, StrategyArbiter, StrategyBias};
pub use graph::profile::{AttentionKind, FfnKind, GraphProfile, GraphProfiler, ResidualKind};
