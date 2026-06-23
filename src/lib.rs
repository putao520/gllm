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
pub use client::{Client, AsyncClient, GllmError, ModelInfo, MtpGenerationResponse, MtpStepInfo};
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
pub use embeddings::{EmbedConfig, Embedding, EmbeddingsBuilder, EmbeddingsResponse, RagResponse};
pub use classify::{ClassifyResponse, ClassificationResult};
pub use rerank::{RerankBuilder, RerankResponse, RerankResult};
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

pub use engine::arbiter::{ArbiterHwView, DeviceFamily, GraphArchetype, InferenceMode, StrategyArbiter, StrategyBias, StrategyBiasResolver};
pub use engine::intent_bias::{IntentBias, OverlapHint, ScenarioHint};
pub use engine::distributed_config::{
    AllToAllStrategy, CommCompressHint, ExpertPlacement, KvDistMode, NodeRole, PdDisaggMode,
};
#[cfg(feature = "nccl")]
pub use engine::distributed_config::{
    CommConfig, CommHandleWrapper, DistributedConfig, DistributedConfigError,
    KvDistributionConfig, MoeDistributedConfig, ParallelConfig, PdDisaggConfig,
};
#[cfg(feature = "nccl")]
pub use engine::pipeline::config::{PipelineConfig, PipelineConfigError};
#[cfg(feature = "nccl")]
pub use engine::pipeline::topology::{Topology2D, Topology2DError, Topology3D, Topology3DError};
#[cfg(feature = "nccl")]
pub use engine::pipeline::micro_batch::{
    MicroBatch, MicroBatchScheduler, MicroBatchSchedulerError,
    InterleavedScheduler, InterleavedSchedulerError, InterleavedScheduleStep, SchedulePhase,
};
#[cfg(feature = "nccl")]
pub use engine::pipeline::activation_xfer::{
    ActivationTransport, ActivationTransportError, ActivationDirection,
};
#[cfg(feature = "nccl")]
pub use engine::pipeline::scheduler::{
    PipelineOp, MicroBatchStrategy, PipelineScheduler, PipelineSchedulerError,
    CommComputeOverlap, CommComputeOverlapError, StreamKind, StreamAssignment,
    PipelineKvCacheManager, PipelineKvCacheManagerError,
};
#[cfg(feature = "nccl")]
pub use engine::pipeline::interleaved::{
    Interleaved1F1B, Interleaved1F1BError, ScheduleComparison,
};
#[cfg(feature = "nccl")]
pub use loader::weight_shard::{
    is_shared_weight, extract_layer_index, should_load_weight_for_stage, filter_weights_by_stage,
};
pub use engine::batch_executor::{GenerateRequest, GenerateResult};
pub use graph::profile::{AttentionKind, FfnKind, GraphProfile, GraphProfiler, ResidualKind};
