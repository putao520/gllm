//! gllm Client API — Sync-first, lock-free design (per SPEC 04-API-DESIGN).
//!
//! # Design Principles
//!
//! - **Sync-first**: All operations are synchronous. No async runtime overhead.
//!   Inference engines are CPU-bound compute, not I/O-bound web services.
//! - **Lock-free state**: `arc_swap::ArcSwapOption` for zero-overhead reads,
//!   atomic lock-free model swap. No RwLock, no poisoning, no lock contention.
//! - **Builder pattern**: Complex configuration via fluent API
//! - **Explicit types**: Strong typing over string magic values
//! - **Result-oriented**: Clear success/failure via `Result<T, GllmError>`

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwapOption;

use crate::backend::{
    detect_backend, BackendContext, BackendContextError, BackendType,
};
use crate::engine::arbiter::InferenceMode;
use crate::embeddings::{Embedding, EmbeddingsResponse, RagResponse};
use crate::engine::executor::{BackendError, ExecutorError};
use crate::generation::GenerationResponse;
use crate::loader::{Loader, LoaderConfig, LoaderError, WeightFormat};
use crate::manifest::{
    map_architecture_token_for_kind, MoEConfig, ModelKind, ModelManifest,
    EMPTY_FILE_MAP,
};
use crate::rerank::{RerankResponse, RerankResult};
use gllm_kernels::types::DType;
use thiserror::Error;

// ============================================================================
// Error Types (per SPEC 04-API-DESIGN §4)
// ============================================================================

/// Unified error type for gllm public API (per SPEC 04-API-DESIGN §4).
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("backend error: {0}")]
    BackendError(#[source] crate::engine::executor::BackendError),

    #[error("out of memory: {0}")]
    OutOfMemory(#[source] crate::kv_cache::OomHaltError),

    #[error("invalid model type for requested operation")]
    InvalidModelType,

    #[error("no model loaded")]
    NoModelLoaded,

    #[error("runtime error: {0}")]
    RuntimeError(String),
}

/// Primary error type exposed to users (type alias for clarity).
pub type GllmError = ClientError;

// Manual From implementations mapping internal errors to SPEC variants

impl From<BackendContextError> for ClientError {
    fn from(err: BackendContextError) -> Self {
        match err {
            BackendContextError::UnsupportedArchitecture(_arch) => ClientError::InvalidModelType,
            BackendContextError::Loader(err) => ClientError::from(err),
            BackendContextError::Executor(err) => ClientError::from(err),
            BackendContextError::Backend(err) => ClientError::from(err),
        }
    }
}

impl From<LoaderError> for ClientError {
    fn from(err: LoaderError) -> Self {
        ClientError::RuntimeError(format!("loader error: {}", err))
    }
}

impl From<BackendError> for ClientError {
    fn from(err: BackendError) -> Self {
        ClientError::BackendError(err)
    }
}

impl From<ExecutorError> for ClientError {
    fn from(err: ExecutorError) -> Self {
        ClientError::RuntimeError(format!("executor error: {}", err))
    }
}

impl From<crate::scheduler::SchedulerError> for ClientError {
    fn from(err: crate::scheduler::SchedulerError) -> Self {
        match err {
            crate::scheduler::paged_scheduler::SchedulerError::OutOfMemory {
                operation,
                needed_blocks,
                free_blocks,
            } => ClientError::OutOfMemory(crate::kv_cache::OomHaltError::fatal_halt(format!(
                "{}: need {} blocks, only {} free",
                operation, needed_blocks, free_blocks
            ))),
            _ => ClientError::RuntimeError(format!("scheduler error: {}", err)),
        }
    }
}

impl From<crate::model_config::ModelConfigError> for ClientError {
    fn from(err: crate::model_config::ModelConfigError) -> Self {
        ClientError::RuntimeError(format!("model config error: {}", err))
    }
}

// ============================================================================
// Client State
// ============================================================================

/// Internal client state holding the loaded model and backend.
pub struct ClientState {
    pub model_id: String,
    pub manifest: Arc<ModelManifest>,
    pub backend: Arc<BackendContext>,
    pub inference_mode: InferenceMode,
    pub reranker_state: Option<PipelineModelState>,
    pub generator_state: Option<PipelineModelState>,
    /// ARCH-PER-CLIENT-PLAN (REQ-ARB-008): per-Client ExecutionPlan computed from
    /// model archetype + inference mode + hardware via StrategyArbiter. Pushed onto
    /// the kernels thread-local stack via `with_execution_plan` for the duration of
    /// each inference call (embed/rerank/generate),消除全局 OnceLock 锁定首个
    /// model bias 的污染问题。
    pub execution_plan: Arc<gllm_kernels::compiler::planner::ExecutionPlan>,
}

/// State for a pipeline sub-model (reranker or generator).
///
/// When `shared_encoder` is true, `backend` is an `Arc` clone of the primary
/// model's backend — both embedder and reranker share the same encoder weights
/// and executor, eliminating duplicate weight loads for same-architecture pairs
/// (e.g. BAAI/bge-m3 + BAAI/bge-reranker-v2-m3, both XLM-R).
pub struct PipelineModelState {
    pub model_id: String,
    pub manifest: Arc<ModelManifest>,
    pub backend: Arc<BackendContext>,
    /// True when this pipeline model shares the primary model's encoder backend
    /// (same architecture). The reranker uses CLS→Classifier while the
    /// embedder uses MeanPool→L2Norm, but the encoder forward pass is identical.
    pub shared_encoder: bool,
    /// ARCH-PER-CLIENT-PLAN: pipeline 模型自己的 ExecutionPlan,在 inference 时
    /// 经 with_execution_plan 推入 TLS,供 codegen 层 (autotuning / dispatch)
    /// 透明读取。shared_encoder=true 时与 primary 共享同一 Arc。
    pub execution_plan: Arc<gllm_kernels::compiler::planner::ExecutionPlan>,
}

// ============================================================================
// Client Builder (per SPEC 04-API-DESIGN §2.1, REQ-CLIENT-001~005)
// ============================================================================

/// Builder for constructing a `Client` with custom configuration.
///
/// # Example
///
/// ```no_run
/// use gllm::{Client, ModelKind, BackendType};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::builder()
///     .model("Qwen/Qwen3-7B-Instruct")
///     .kind(ModelKind::Chat)
///     .backend(BackendType::Cuda)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    model_id: Option<String>,
    kind: Option<ModelKind>,
    backend: Option<BackendType>,
    inference_mode: InferenceMode,
    reranker_model_id: Option<String>,
    generator_model_id: Option<String>,
}

fn make_dummy_manifest(model_id: &str, arch: impl Into<String>, kind: ModelKind) -> ModelManifest {
    make_dummy_manifest_with_moe(model_id, arch, kind, None)
}

fn make_dummy_manifest_with_moe(
    model_id: &str,
    arch: impl Into<String>,
    kind: ModelKind,
    moe_config: Option<MoEConfig>,
) -> ModelManifest {
    ModelManifest {
        model_id: Cow::Owned(model_id.to_string()),
        file_map: EMPTY_FILE_MAP,
        arch: arch.into(),
        kind,
        rope_base_override: None,
        max_context_override: None,
        moe_config,
        tensor_map: HashMap::new(),
    }
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self {
            model_id: None,
            kind: None,
            backend: None,
            inference_mode: InferenceMode::Latency,
            reranker_model_id: None,
            generator_model_id: None,
        }
    }

    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    pub fn kind(mut self, kind: ModelKind) -> Self {
        self.kind = Some(kind);
        self
    }

    pub fn backend(mut self, backend: BackendType) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn inference_mode(mut self, mode: InferenceMode) -> Self {
        self.inference_mode = mode;
        self
    }

    /// Add a reranker model to the pipeline.
    ///
    /// When set, the client can execute embed+rerank pipelines via
    /// `EmbeddingsBuilder::rerank_query()`.
    pub fn reranker(mut self, model_id: impl Into<String>) -> Self {
        self.reranker_model_id = Some(model_id.into());
        self
    }

    /// Add a generator (LLM) model to the pipeline.
    ///
    /// When set, the client can execute full RAG pipelines via
    /// `EmbeddingsBuilder::generate_answer()`.
    pub fn generator(mut self, model_id: impl Into<String>) -> Self {
        self.generator_model_id = Some(model_id.into());
        self
    }

    /// Build the `Client` and load the model synchronously.
    ///
    /// When both an embedder and reranker are configured with the same
    /// architecture, the encoder backend is shared (Arc clone) to
    /// avoid loading duplicate weights. The reranker uses CLS→Classifier
    /// while the embedder uses MeanPool→L2Norm, but the underlying
    /// encoder forward pass is identical.
    pub fn build(self) -> Result<Client, ClientError> {
        let model_id = self
            .model_id
            .ok_or_else(|| ClientError::ModelNotFound("<no model id>".to_string()))?;
        let kind = self.kind.unwrap_or(ModelKind::Chat);
        let mut state = Self::build_state(&model_id, kind, self.inference_mode)?;

        if let Some(ref reranker_id) = self.reranker_model_id {
            state.reranker_state = Some(
                Self::build_pipeline_model_with_sharing(
                    reranker_id,
                    ModelKind::Reranker,
                    &state.manifest,
                    &state.backend,
                    &state.execution_plan,
                )?
            );
        }

        if let Some(ref generator_id) = self.generator_model_id {
            state.generator_state =
                Some(Self::build_pipeline_model(generator_id, ModelKind::Chat)?);
        }

        // ARCH-MULTIMODAL (SPEC §3.7): if the model declares a `vision_config`
        // or `audio_config`, try to materialise the real encoder from loaded
        // weights and auto-register. Failing to find every weight (e.g. text-
        // only checkpoint that happens to declare multimodal geometry) falls
        // through to `None`; the user can still inject a custom encoder via
        // `Client::set_multimodal_encoder`.
        //
        // When BOTH vision and audio are available, wrap them in a dispatch
        // composer so a single `MultimodalEncoder` implementation delegates
        // `encode_image` to SigLIP and `encode_audio` to USM Conformer.
        let vision_enc: Option<Arc<dyn crate::compat::multimodal::MultimodalEncoder>> = {
            let executor = state.backend.executor();
            match executor.try_build_siglip_encoder() {
                Ok(Some(enc)) => Some(Arc::new(enc) as Arc<dyn crate::compat::multimodal::MultimodalEncoder>),
                Ok(None) => None,
                Err(e) => {
                    log::warn!(
                        "SigLIP encoder auto-build failed ({e}); model will require manual \
                         `Client::set_multimodal_encoder` to process images"
                    );
                    None
                }
            }
        };
        let audio_enc: Option<Arc<dyn crate::compat::multimodal::MultimodalEncoder>> = {
            let executor = state.backend.executor();
            match executor.try_build_usm_conformer_encoder() {
                Ok(Some(enc)) => Some(Arc::new(enc) as Arc<dyn crate::compat::multimodal::MultimodalEncoder>),
                Ok(None) => None,
                Err(e) => {
                    log::warn!(
                        "USM Conformer encoder auto-build failed ({e}); model will require manual \
                         `Client::set_multimodal_encoder` to process audio"
                    );
                    None
                }
            }
        };

        let auto_encoder: Option<Arc<dyn crate::compat::multimodal::MultimodalEncoder>> =
            match (vision_enc, audio_enc) {
                (Some(v), Some(a)) => Some(Arc::new(MultimodalEncoderCompose::new(v, a))
                    as Arc<dyn crate::compat::multimodal::MultimodalEncoder>),
                (Some(v), None) => Some(v),
                (None, Some(a)) => Some(a),
                (None, None) => None,
            };

        Ok(Client {
            state: Arc::new(ArcSwapOption::from_pointee(state)),
            multimodal_encoder: Arc::new(std::sync::Mutex::new(auto_encoder)),
            guardrails: Arc::new(std::sync::Mutex::new(HashMap::new())),
            guardrail_next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            sg_callback: Arc::new(std::sync::Mutex::new(None)),
        })
    }

    /// Load model and construct `ClientState` synchronously.
    ///
    /// This is the core model-loading function shared by all constructors.
    /// No async runtime, no locks — pure sync I/O + CPU initialization.
    pub(crate) fn build_state(
        model_id: &str,
        kind: ModelKind,
        inference_mode: InferenceMode,
    ) -> Result<ClientState, ClientError> {
        let config = LoaderConfig::from_env();

        // Ω1: Tensor-driven loading — no config.json dependency
        let mut loader = Loader::from_source_with_config(model_id.to_string(), config.clone())?;

        // model_config is extracted during manifest construction for Strategy Arbiter use.
        let mut model_config_for_arbiter: Option<crate::model_config::ModelConfig> = None;

        let manifest = match loader.weight_format() {
            WeightFormat::Gguf => {
                loader = loader.load()?;
                let arch_str = loader.gguf_architecture()?;
                if let Some(arch) = map_architecture_token_for_kind(arch_str, kind) {
                    let dummy_manifest = make_dummy_manifest(model_id, &arch, kind);
                    let cfg_result =
                        crate::model_config::ModelConfig::from_loader(&dummy_manifest, &mut loader);
                    let moe_config = cfg_result
                        .as_ref()
                        .ok()
                        .and_then(|cfg| cfg.build_moe_config(&arch));
                    if let Ok(cfg) = cfg_result {
                        model_config_for_arbiter = Some(cfg);
                    }
                    make_dummy_manifest_with_moe(model_id, &arch, kind, moe_config)
                } else {
                    return Err(ClientError::ModelNotFound(format!(
                        "Unsupported GGUF architecture: {}",
                        arch_str
                    )));
                }
            }
            WeightFormat::SafeTensors | WeightFormat::Onnx | WeightFormat::PyTorch => {
                // Ω1: Tensor-driven derivation (REQ-LOADER-022, REQ-LOADER-023)
                loader = loader.load()?;

                let dummy_manifest = make_dummy_manifest(model_id, "llama", kind);

                let derived_config =
                    crate::model_config::ModelConfig::from_loader(&dummy_manifest, &mut loader)?;

                let arch = loader.detect_architecture();
                let moe_config = derived_config.build_moe_config(&arch);
                model_config_for_arbiter = Some(derived_config);

                make_dummy_manifest_with_moe(model_id, &arch, kind, moe_config)
            }
        };

        // Detect backend ONCE — reused for both arbiter hardware view and BackendContext.
        let detected_backend = detect_backend()?;
        let backend_type = detected_backend.backend_type();

        // ARCH-PER-CLIENT-PLAN (REQ-ARB-008/009): per-Client ExecutionPlan
        // 隔离 (Arc<ExecutionPlan> 存 ClientState + with_execution_plan TLS push)。
        // REQ-ARB-001~007 完整调用 StrategyArbiter::arbitrate(mode, archetype, hw):
        //   InferenceMode (Latency/Throughput baseline) ×
        //   GraphArchetype (fusion_profitable / pipeline_valuable 等模型图特征) ×
        //   ArbiterHwView (cache / SIMD regs / GPU)
        // → StrategyBias → compute_execution_plan_with_bias → 每 Client 独立 plan。
        let arbiter_bias = if let Some(cfg) = &model_config_for_arbiter {
            let hw_profile = gllm_kernels::dispatch::device_profile();
            let archetype = {
                let graph_profile = crate::graph::profile::GraphProfiler::profile(cfg);
                crate::engine::arbiter::GraphArchetype::derive(&graph_profile)
            };
            let hw_view = crate::engine::arbiter::ArbiterHwView::from(hw_profile);
            crate::engine::arbiter::StrategyArbiter::arbitrate(
                inference_mode,
                &archetype,
                &hw_view,
            )
        } else {
            // model_config 缺失时退回 mode baseline (无 archetype/hw modulation)
            gllm_kernels::compiler::planner::StrategyBias::default()
        };
        let _ = backend_type;
        let execution_plan =
            gllm_kernels::compiler::planner::compute_execution_plan_with_bias(&arbiter_bias);

        let config_path = loader.config_path().map(|p| p.to_path_buf());
        let tokenizer_path = loader.tokenizer_path().map(|p| p.to_path_buf());
        let weight_paths = loader.weight_paths().to_vec();

        let manifest = Arc::new(manifest);

        // ARCH-PER-CLIENT-PLAN: 编译阶段也用 per-Client plan,确保 JIT codegen 的
        // FusionPlan/AttentionStrategy/MoE 决策与 inference 阶段一致。
        let backend = gllm_kernels::compiler::planner::with_execution_plan(
            execution_plan.clone(),
            || BackendContext::new(
                model_id.to_string(),
                manifest.clone(),
                detected_backend,
                weight_paths,
                config_path,
                tokenizer_path,
            ),
        )?;

        Ok(ClientState {
            model_id: model_id.to_string(),
            manifest,
            backend: Arc::new(backend),
            inference_mode,
            reranker_state: None,
            generator_state: None,
            execution_plan,
        })
    }

    /// Build a pipeline sub-model (reranker or generator).
    ///
    /// Uses the same loading logic as `build_state` but produces a
    /// `PipelineModelState` instead of a full `ClientState`.
    fn build_pipeline_model(
        model_id: &str,
        kind: ModelKind,
    ) -> Result<PipelineModelState, ClientError> {
        let state = Self::build_state(model_id, kind, InferenceMode::Latency)?;
        Ok(PipelineModelState {
            model_id: state.model_id,
            manifest: state.manifest,
            backend: state.backend,
            shared_encoder: false,
            execution_plan: state.execution_plan,
        })
    }

    /// Build a pipeline sub-model, sharing the primary model's encoder backend
    /// when both models have the same architecture.
    ///
    /// This avoids loading duplicate encoder weights for same-architecture pairs
    /// (e.g. BAAI/bge-m3 embedder + BAAI/bge-reranker-v2-m3 reranker, both XLM-R).
    ///
    /// When architectures differ, falls back to independent loading via
    /// `build_pipeline_model`.
    fn build_pipeline_model_with_sharing(
        model_id: &str,
        kind: ModelKind,
        primary_manifest: &Arc<ModelManifest>,
        primary_backend: &Arc<BackendContext>,
        primary_execution_plan: &Arc<gllm_kernels::compiler::planner::ExecutionPlan>,
    ) -> Result<PipelineModelState, ClientError> {
        // Resolve the pipeline model's manifest to determine its architecture.
        let pipeline_manifest = Self::resolve_manifest(model_id, kind)?;

        // ARCH-PIPELINE-SHARING: 仅当 model_id 与 arch 都相同(用户用同一模型同时
        // 做 embed 和 rerank)时才共享 backend。架构相同但模型不同(e.g. e5-small +
        // bge-reranker-v2-m3 都是 xlm-roberta)时,**权重完全不同**,共享 backend
        // 会让 reranker 实际使用 embedder 的权重 → 数值漂移 / NaN。
        // 历史 BUG: e2e_fusion_consistency_with_standalone / cross_arch_embed_rerank
        // 在 e5-small-v2 + bge-reranker-v2-m3 场景下因共享 backend 输出 NaN。
        if pipeline_manifest.arch == primary_manifest.arch
            && model_id == primary_manifest.model_id
        {
            log::info!(
                "pipeline: sharing backend (same model_id={} & arch={})",
                model_id, pipeline_manifest.arch,
            );
            Ok(PipelineModelState {
                model_id: model_id.to_string(),
                manifest: Arc::new(pipeline_manifest),
                backend: Arc::clone(primary_backend),
                shared_encoder: true,
                // shared_encoder=true → 复用 primary_state 的 plan (架构相同 + model_id 相同)
                execution_plan: Arc::clone(primary_execution_plan),
            })
        } else {
            // 不同 model_id 或不同 arch: 独立加载,各持自己的权重。
            log::info!(
                "pipeline: loading independent backend for {} (arch {}, primary {} arch {})",
                model_id, pipeline_manifest.arch, primary_manifest.model_id, primary_manifest.arch,
            );
            Self::build_pipeline_model(model_id, kind)
        }
    }

    /// Resolve a model's manifest (architecture, kind, MoE config) without
    /// constructing a full `BackendContext`.
    ///
    /// This performs weight loading and architecture detection but stops before
    /// JIT compilation, making it suitable for arch-matching decisions.
    fn resolve_manifest(
        model_id: &str,
        kind: ModelKind,
    ) -> Result<ModelManifest, ClientError> {
        let config = LoaderConfig::from_env();
        let mut loader = Loader::from_source_with_config(model_id.to_string(), config)?;

        let manifest = match loader.weight_format() {
            WeightFormat::Gguf => {
                loader = loader.load()?;
                let arch_str = loader.gguf_architecture()?;
                if let Some(arch) = map_architecture_token_for_kind(arch_str, kind) {
                    let dummy_manifest = make_dummy_manifest(model_id, &arch, kind);
                    let cfg_result =
                        crate::model_config::ModelConfig::from_loader(&dummy_manifest, &mut loader);
                    let moe_config = cfg_result
                        .as_ref()
                        .ok()
                        .and_then(|cfg| cfg.build_moe_config(&arch));
                    make_dummy_manifest_with_moe(model_id, &arch, kind, moe_config)
                } else {
                    return Err(ClientError::ModelNotFound(format!(
                        "Unsupported GGUF architecture: {}",
                        arch_str
                    )));
                }
            }
            WeightFormat::SafeTensors | WeightFormat::Onnx | WeightFormat::PyTorch => {
                loader = loader.load()?;
                let dummy_manifest = make_dummy_manifest(model_id, "llama", kind);
                let derived_config =
                    crate::model_config::ModelConfig::from_loader(&dummy_manifest, &mut loader)?;
                let arch = loader.detect_architecture();
                let moe_config = derived_config.build_moe_config(&arch);
                make_dummy_manifest_with_moe(model_id, &arch, kind, moe_config)
            }
        };

        Ok(manifest)
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Client (per SPEC 04-API-DESIGN §2) — Sync, Lock-free
// ============================================================================

/// Main gllm client for model inference.
///
/// Lock-free state management via `ArcSwapOption<ClientState>`:
/// - **Reads** (inference calls): zero-cost `load()` → atomic Arc clone, no lock
/// - **Writes** (model swap): atomic `store()`, old state kept alive until all
///   in-flight reads complete, then dropped automatically
/// - **No async runtime**: inference is CPU-bound compute, not I/O-bound web service
///
/// # Example
///
/// ```no_run
/// use gllm::Client;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::new_chat("Qwen/Qwen3-7B-Instruct")?;
/// let output = client.generate("Hello, who are you?")
///     .max_tokens(100)
///     .temperature(0.7)
///     .generate()?;
/// println!("{}", output.text);
/// # Ok(())
/// # }
/// ```
/// Composes two distinct `MultimodalEncoder` implementations (vision +
/// audio) into a single dispatch object. `encode_image` delegates to
/// `vision`, `encode_audio` delegates to `audio`. Used by `build_state` /
/// `ClientBuilder::build` when the loaded model exposes both SigLIP and
/// USM Conformer weights.
struct MultimodalEncoderCompose {
    vision: Arc<dyn crate::compat::multimodal::MultimodalEncoder>,
    audio: Arc<dyn crate::compat::multimodal::MultimodalEncoder>,
}

impl MultimodalEncoderCompose {
    fn new(
        vision: Arc<dyn crate::compat::multimodal::MultimodalEncoder>,
        audio: Arc<dyn crate::compat::multimodal::MultimodalEncoder>,
    ) -> Self {
        Self { vision, audio }
    }
}

impl crate::compat::multimodal::MultimodalEncoder for MultimodalEncoderCompose {
    fn encode_image(
        &self,
        media: &crate::compat::multimodal::EncoderMedia,
    ) -> Result<crate::compat::multimodal::MultimodalEncoded, BackendError> {
        self.vision.encode_image(media)
    }

    fn encode_audio(
        &self,
        media: &crate::compat::multimodal::EncoderMedia,
    ) -> Result<crate::compat::multimodal::MultimodalEncoded, BackendError> {
        self.audio.encode_audio(media)
    }
}

#[derive(Clone)]
pub struct Client {
    state: Arc<ArcSwapOption<ClientState>>,
    /// Optional multimodal encoder (SigLIP vision / USM audio).
    ///
    /// T58 scaffold: when `None`, calls to `.image()` / `.audio()` on the
    /// generation builder fail fast with `ClientError::InvalidModelType`.
    /// Tests inject mock encoders via `set_multimodal_encoder()` to
    /// validate routing without running a real encoder forward pass.
    /// Production SigLIP / USM encoders will be registered here by
    /// `build_state()` once the real implementations land (T55 / T55.2).
    ///
    /// Uses `Mutex<Option<Arc<dyn Trait>>>` because `ArcSwapOption`
    /// requires a `Sized` inner type.
    multimodal_encoder:
        Arc<std::sync::Mutex<Option<Arc<dyn crate::compat::multimodal::MultimodalEncoder>>>>,

    /// Registered Guardrail attachments (SPEC/GUARDRAIL.md).
    ///
    /// Maps `GuardrailAttachment::id` → stored spec (weights + policy + layer).
    /// `Client::classify_binary` / `encode_intent` / future generation paths
    /// iterate this map to build a `CallbackChain` with the corresponding
    /// `GuardrailProbeCallback` per forward call.
    guardrails: Arc<std::sync::Mutex<HashMap<u64, GuardrailRegistration>>>,
    /// Monotonic id allocator for `attach_guardrail`.
    guardrail_next_id: Arc<std::sync::atomic::AtomicU64>,
    /// Registered Semantic Gatekeeper callback (SPEC/SEMANTIC-GATEKEEPER.md).
    sg_callback: Arc<std::sync::Mutex<Option<Arc<std::sync::Mutex<crate::semantic_gatekeeper::SemanticGatekeeperCallback>>>>>,
}

/// Internal storage for a registered Guardrail (kept inside the Client).
///
/// Cloned into a fresh `GuardrailProbeCallback` on each forward invocation
/// (callbacks require `Send + !Sync` ownership per CallbackChain semantics).
#[derive(Clone)]
struct GuardrailRegistration {
    weights: crate::guardrail::GuardProbeWeights,
    policy: crate::guardrail::SafetyPolicy,
    actual_layer: usize,
    probe_name: String,
    shared: Arc<crate::guardrail::GuardrailSharedState>,
}

impl Client {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Create a new client with a chat model (sync, blocking).
    pub fn new_chat(model_id: &str) -> Result<Self, ClientError> {
        Self::new(model_id, ModelKind::Chat)
    }

    /// Create a new client with an embedding model (sync, blocking).
    pub fn new_embedding(model_id: &str) -> Result<Self, ClientError> {
        Self::new(model_id, ModelKind::Embedding)
    }

    /// Create a new client with a classifier model (sync, blocking).
    pub fn new_classifier(model_id: &str) -> Result<Self, ClientError> {
        Self::new(model_id, ModelKind::Classifier)
    }

    /// Create a new client with the specified model and kind (sync, blocking).
    pub fn new(model_id: &str, kind: ModelKind) -> Result<Self, ClientError> {
        let state = ClientBuilder::build_state(model_id, kind, InferenceMode::Latency)?;
        Ok(Client {
            state: Arc::new(ArcSwapOption::from_pointee(state)),
            multimodal_encoder: Arc::new(std::sync::Mutex::new(None)),
            guardrails: Arc::new(std::sync::Mutex::new(HashMap::new())),
            guardrail_next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            sg_callback: Arc::new(std::sync::Mutex::new(None)),
        })
    }

    /// Create an empty client (no model loaded).
    pub fn new_empty() -> Self {
        Self {
            state: Arc::new(ArcSwapOption::empty()),
            multimodal_encoder: Arc::new(std::sync::Mutex::new(None)),
            guardrails: Arc::new(std::sync::Mutex::new(HashMap::new())),
            guardrail_next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            sg_callback: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    // -----------------------------------------------------------------
    // Model Management
    // -----------------------------------------------------------------

    /// Load a model into the client (sync, blocking).
    ///
    /// Atomically replaces the current model. The old model's resources
    /// are released once all in-flight operations complete.
    pub fn load_model(&self, model_id: &str, kind: ModelKind) -> Result<(), ClientError> {
        let model_id = Self::normalize_model_id(model_id)?;
        let state = ClientBuilder::build_state(&model_id, kind, InferenceMode::Latency)?;
        self.state.store(Some(Arc::new(state)));
        Ok(())
    }

    /// Unload the current model, releasing resources (sync).
    pub fn unload_model(&self) -> Result<(), ClientError> {
        self.state.store(None);
        Ok(())
    }

    /// Swap to a different model (sync, atomic).
    ///
    /// If loading fails, the client will be in an empty state.
    pub fn swap_model(&self, model_id: &str) -> Result<(), ClientError> {
        let model_id = Self::normalize_model_id(model_id)?;

        // Get current model kind before unloading (lock-free read)
        let kind = self
            .state
            .load()
            .as_ref()
            .map(|loaded| loaded.manifest.kind)
            .ok_or(ClientError::NoModelLoaded)?;

        // Atomic swap: old state kept alive for in-flight reads, new state installed
        let state = ClientBuilder::build_state(&model_id, kind, InferenceMode::Latency)?;
        self.state.store(Some(Arc::new(state)));
        Ok(())
    }

    /// Get information about the currently loaded model.
    pub fn model_info(&self) -> Option<ModelInfo> {
        self.state.load().as_ref().map(|loaded| ModelInfo {
            id: loaded.model_id.clone(),
            arch: loaded.manifest.arch.clone(),
            kind: loaded.manifest.kind,
        })
    }

    /// Get the manifest of the currently loaded model.
    pub fn manifest(&self) -> Result<Arc<ModelManifest>, ClientError> {
        self.state
            .load()
            .as_ref()
            .map(|loaded| loaded.manifest.clone())
            .ok_or(ClientError::NoModelLoaded)
    }

    /// Get the currently loaded model's `(hidden_size, num_layers)` — convenience
    /// accessor for HR / Guardrail / Intent callers that need to size probe
    /// weights or validate layer anchors without walking the executor lock.
    pub fn model_dims(&self) -> Result<(usize, usize), ClientError> {
        let state = self.require_state()?;
        let executor = state.backend.executor();
        let cfg = executor.model_config();
        Ok((cfg.hidden_size, cfg.num_hidden_layers))
    }

    /// Check if a model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.state.load().is_some()
    }

    // -----------------------------------------------------------------
    // Inference APIs (per SPEC 04-API-DESIGN §3)
    // -----------------------------------------------------------------

    /// Create a text generation builder.
    pub fn generate(&self, prompt: impl Into<String>) -> crate::generation::GenerationBuilder<'_> {
        crate::generation::GenerationBuilder::from_prompt(self, prompt)
    }

    /// Generate embeddings for texts (per SPEC 04-API-DESIGN §3.2).
    pub fn embed<I, S>(&self, inputs: I) -> Result<EmbeddingsResponse, ClientError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let inputs = inputs.into_iter().map(Into::into).collect();
        self.execute_embeddings(inputs)
    }

    /// Generate embeddings for texts (alias for backward compatibility).
    #[deprecated(since = "0.12.0", note = "Use embed() instead")]
    pub fn embeddings<I, S>(&self, inputs: I) -> Result<EmbeddingsResponse, ClientError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.embed(inputs)
    }

    /// Create an embeddings builder with pipeline support (embed + rerank + RAG).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use gllm::Client;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::builder()
    ///     .model("BAAI/bge-small-en-v1.5")
    ///     .reranker("BAAI/bge-reranker-v2-m3")
    ///     .build()?;
    ///
    /// let result = client.embed_builder(vec!["doc1", "doc2"])
    ///     .rerank_query("query")
    ///     .top_n(5)
    ///     .generate()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed_builder<I, S>(&self, inputs: I) -> crate::embeddings::EmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let inputs = inputs.into_iter().map(Into::into).collect();
        crate::embeddings::EmbeddingsBuilder::new(self, inputs)
    }

    /// Rerank documents by relevance to query (sync).
    pub fn rerank<I, S>(
        &self,
        query: impl Into<String>,
        documents: I,
    ) -> Result<RerankResponse, ClientError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let documents = documents.into_iter().map(Into::into).collect();
        let query = query.into();
        self.execute_rerank(query, documents, usize::MAX)
    }

    /// Classify texts into categories (sync).
    ///
    /// Returns raw logits for each input text. The number of logits per text
    /// depends on the model's classifier head (num_labels).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::Client;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new_classifier("model-id")?;
    /// let result = client.classify(["This is positive", "This is negative"])?;
    /// for pred in &result.predictions {
    ///     println!("label={} score={:.4}", pred.label_id, pred.score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn classify<I, S>(&self, inputs: I) -> Result<crate::classify::ClassifyResponse, ClientError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let texts: Vec<String> = inputs.into_iter().map(Into::into).collect();
        self.execute_classify(texts)
    }

    // -----------------------------------------------------------------
    // Head Routing SDK (SPEC/HEAD-ROUTING.md, SPEC/04-API-DESIGN §3.8)
    //
    // 同一 generator LLM 加载后,运行时切换输出头形态,零权重重载、零 JIT
    // 重编译 (REQ-HR-001..005)。
    // -----------------------------------------------------------------

    /// Binary classify head (REQ-HR-001) — 对 `prompt` 跑一次 generator
    /// forward,读取 lm_head 对 positive/negative token 的 logit,
    /// softmax 归一化后返回 `P(positive_token) ∈ [0.0, 1.0]`。
    ///
    /// # 错误
    /// - `positive_token` / `negative_token` 无法单 token 化 →
    ///   `HeadRoutingError::TokenNotFound(...)` 包装为 `ClientError::RuntimeError`
    /// - `temperature <= 0` / NaN → `InvalidConfig`
    /// - 下游 backend forward 失败 → 原错误传播
    ///
    /// # 关联
    /// - SPEC/HEAD-ROUTING.md §3.2 / §4
    /// - SPEC/04-API-DESIGN.md §3.8
    pub fn classify_binary(
        &self,
        prompt: &str,
        config: crate::head_routing::ClassifyBinaryConfig,
    ) -> Result<f32, ClientError> {
        use crate::head_routing::{softmax_with_temperature, HeadRoutingError};

        let state = self.require_state()?;
        let hidden_size = state.backend.executor().model_config().hidden_size;
        let mut guardrail_chain = self.build_guardrail_chain(Vec::new(), hidden_size);
        let plan = state.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
            let mut executor = state.backend.executor_mut();
            let tokenizer = executor.tokenizer();
            let pos_id = resolve_single_token(tokenizer, &config.positive_token)
                .map_err(hr_err_to_client)?;
            let neg_id = resolve_single_token(tokenizer, &config.negative_token)
                .map_err(hr_err_to_client)?;
            if pos_id == neg_id {
                return Err(hr_err_to_client(HeadRoutingError::InvalidConfig(format!(
                    "positive_token ({}) and negative_token ({}) resolve to same token id {}",
                    config.positive_token, config.negative_token, pos_id,
                ))));
            }
            let logits = executor
                .score_tokens_for_prompt_with_callbacks(
                    prompt,
                    &[pos_id, neg_id],
                    guardrail_chain.as_mut(),
                )
                .map_err(ClientError::from)?;
            if logits.is_empty() {
                // Guardrail vetoed the forward — surface an explicit error so
                // callers can treat P(positive) as undefined.
                return Err(ClientError::RuntimeError(
                    "classify_binary: guardrail vetoed forward pass (see GuardrailAttachment::last_veto_reason)".into(),
                ));
            }
            if logits.len() != 2 {
                return Err(ClientError::RuntimeError(format!(
                    "classify_binary: expected 2 logits, got {}",
                    logits.len()
                )));
            }
            let probs = softmax_with_temperature(&logits, config.temperature)
                .map_err(hr_err_to_client)?;
            // probs[0] 对应 pos_id (positive_token), probs[1] 对应 neg_id
            Ok(probs[0])
        })
    }

    /// Multiway classify head (REQ-HR-002) — 对 `prompt` 跑一次 generator
    /// forward,读取 lm_head 对每个 `labels[i]` token 的 logit,对 N 个
    /// logit 联合 softmax 归一化,返回 `Vec<f32>` (每个标签对应概率)。
    ///
    /// # 错误
    /// - `labels.is_empty()` → `EmptyLabels`
    /// - 任一 label 无法单 token 化 → `TokenNotFound(label)`
    /// - `config.temperature <= 0` → `InvalidConfig`
    ///
    /// # 关联
    /// - SPEC/HEAD-ROUTING.md §3.3 / §4
    pub fn classify_multiway(
        &self,
        prompt: &str,
        labels: &[&str],
        config: crate::head_routing::ClassifyMultiwayConfig,
    ) -> Result<Vec<f32>, ClientError> {
        use crate::head_routing::{softmax_with_temperature, HeadRoutingError};

        if labels.is_empty() {
            return Err(hr_err_to_client(HeadRoutingError::EmptyLabels));
        }
        let state = self.require_state()?;
        let plan = state.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
            let mut executor = state.backend.executor_mut();
            let tokenizer = executor.tokenizer();
            let mut token_ids = Vec::with_capacity(labels.len());
            for label in labels {
                let id = resolve_single_token(tokenizer, label).map_err(hr_err_to_client)?;
                token_ids.push(id);
            }
            let logits = executor
                .score_tokens_for_prompt(prompt, &token_ids)
                .map_err(ClientError::from)?;
            if logits.len() != labels.len() {
                return Err(ClientError::RuntimeError(format!(
                    "classify_multiway: expected {} logits, got {}",
                    labels.len(),
                    logits.len()
                )));
            }
            softmax_with_temperature(&logits, config.temperature).map_err(hr_err_to_client)
        })
    }

    /// Mid-layer encode head (REQ-HR-003) — truncate the forward pass at
    /// `anchor` and pool the captured hidden state.
    ///
    /// Implementation: attaches a `MidLayerEncodeCallback` at the resolved
    /// physical layer via the mega-kernel path with callbacks. The
    /// callback returns `CallbackAction::ExitEarly { logits: <hidden as f32> }`
    /// when the anchor layer's `post_node` fires. `encode_at_layer_for_prompt`
    /// then reshapes to `[seq_len, hidden_size]` and applies `pool`.
    ///
    /// Validates `anchor` before running: `InvalidLayerAnchor` surfaces
    /// eagerly (contract for TEST-HR-005).
    ///
    /// # 关联
    /// - SPEC/HEAD-ROUTING.md §5 mid-layer encode 协议
    /// - SPEC/INTENT.md §3 架构
    /// - SPEC/04-API-DESIGN.md §3.8.1 / §3.8.4
    pub fn encode_to_layer(
        &self,
        text: &str,
        anchor: crate::head_routing::LayerAnchor,
        pool: crate::head_routing::PoolMode,
    ) -> Result<Vec<f32>, ClientError> {
        let state = self.require_state()?;
        let plan = state.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
            let mut executor = state.backend.executor_mut();
            let num_layers = executor.model_config().num_hidden_layers;
            let resolved_layer = anchor.resolve(num_layers).map_err(hr_err_to_client)?;
            executor
                .encode_at_layer_for_prompt(text, resolved_layer, pool)
                .map_err(ClientError::from)
        })
    }

    // -----------------------------------------------------------------
    // Intent Recall SDK (SPEC/INTENT.md, SPEC/04-API-DESIGN §3.10)
    // -----------------------------------------------------------------

    /// Intent Recall — semantic wrapper around `encode_to_layer`.
    ///
    /// Returns `IntentEncoding { embedding, actual_layer, pool }` for
    /// intent-classification downstreams. Identical forward path to
    /// `encode_to_layer` (zero code duplication via delegation).
    ///
    /// # 关联
    /// - SPEC/INTENT.md
    /// - SPEC/04-API-DESIGN.md §3.10
    pub fn encode_intent(
        &self,
        text: &str,
        anchor: crate::head_routing::LayerAnchor,
        pool: crate::head_routing::PoolMode,
    ) -> Result<crate::intent::IntentEncoding, ClientError> {
        let state = self.require_state()?;
        let num_layers = {
            let executor = state.backend.executor();
            executor.model_config().num_hidden_layers
        };
        let actual_layer = anchor.resolve(num_layers).map_err(hr_err_to_client)?;
        let embedding = self.encode_to_layer(text, anchor, pool)?;
        Ok(crate::intent::IntentEncoding {
            embedding,
            actual_layer,
            pool,
        })
    }

    // -----------------------------------------------------------------
    // Guardrail SDK (SPEC/GUARDRAIL.md, SPEC/04-API-DESIGN §3.9)
    // -----------------------------------------------------------------

    /// Attach a Guardrail probe at the given anchor layer.
    ///
    /// The returned `GuardrailAttachment` carries an `id` usable for
    /// `Client::detach_guardrail(id)`, along with shared-state accessors
    /// (`last_score()` / `is_vetoed()` / `downgraded_temperature()`). On
    /// subsequent `classify_binary` / `classify_multiway` / `encode_intent`
    /// calls, the Client builds a `CallbackChain` that includes a
    /// `GuardrailProbeCallback` per registered attachment.
    ///
    /// # Errors
    /// - `ClientError::NoModelLoaded` if no model is loaded
    /// - Anchor resolution failure (越界 / NaN) → `ClientError::RuntimeError`
    /// - Policy validation failure → `ClientError::RuntimeError`
    /// - Weight load failure → `ClientError::RuntimeError`
    ///
    /// # 关联
    /// - SPEC/GUARDRAIL.md §3 API
    /// - SPEC/04-API-DESIGN.md §3.9
    pub fn attach_guardrail(
        &self,
        probe: crate::guardrail::GuardProbe,
        anchor: crate::head_routing::LayerAnchor,
        policy: crate::guardrail::SafetyPolicy,
    ) -> Result<crate::guardrail::GuardrailAttachment, ClientError> {
        use crate::guardrail::{self, GuardrailSharedState};
        guardrail::validate_policy(&policy).map_err(|e| ClientError::RuntimeError(format!("{e}")))?;
        let weights = guardrail::load_probe_weights(&probe)
            .map_err(|e| ClientError::RuntimeError(format!("{e}")))?;
        let state = self.require_state()?;
        let num_layers = {
            let executor = state.backend.executor();
            executor.model_config().num_hidden_layers
        };
        let actual_layer = guardrail::resolve_anchor(anchor, num_layers)
            .map_err(|e| ClientError::RuntimeError(format!("{e}")))?;

        let id = self
            .guardrail_next_id
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let probe_name = format!("guardrail#{id}");
        let shared = Arc::new(GuardrailSharedState::new());

        let registration = GuardrailRegistration {
            weights: weights.clone(),
            policy,
            actual_layer,
            probe_name: probe_name.clone(),
            shared: shared.clone(),
        };
        {
            let mut map = self
                .guardrails
                .lock()
                .map_err(|e| ClientError::RuntimeError(format!("guardrails mutex poisoned: {e}")))?;
            map.insert(id, registration);
        }

        Ok(guardrail::GuardrailAttachment {
            id,
            actual_layer,
            probe_name,
            shared,
        })
    }

    /// Attach a Guardrail probe directly from inline weights. Convenience
    /// wrapper used primarily by tests / in-process probes constructed from
    /// already-loaded tensors.
    pub fn attach_guardrail_inline(
        &self,
        weights: crate::guardrail::GuardProbeWeights,
        anchor: crate::head_routing::LayerAnchor,
        policy: crate::guardrail::SafetyPolicy,
    ) -> Result<crate::guardrail::GuardrailAttachment, ClientError> {
        use crate::guardrail::{self, GuardrailSharedState};
        guardrail::validate_policy(&policy).map_err(|e| ClientError::RuntimeError(format!("{e}")))?;
        let state = self.require_state()?;
        let num_layers = {
            let executor = state.backend.executor();
            executor.model_config().num_hidden_layers
        };
        let actual_layer = guardrail::resolve_anchor(anchor, num_layers)
            .map_err(|e| ClientError::RuntimeError(format!("{e}")))?;

        let id = self
            .guardrail_next_id
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let probe_name = format!("guardrail#{id}");
        let shared = Arc::new(GuardrailSharedState::new());

        let registration = GuardrailRegistration {
            weights,
            policy,
            actual_layer,
            probe_name: probe_name.clone(),
            shared: shared.clone(),
        };
        {
            let mut map = self
                .guardrails
                .lock()
                .map_err(|e| ClientError::RuntimeError(format!("guardrails mutex poisoned: {e}")))?;
            map.insert(id, registration);
        }

        Ok(guardrail::GuardrailAttachment {
            id,
            actual_layer,
            probe_name,
            shared,
        })
    }

    /// Detach a previously attached Guardrail. Returns error if `id` was not
    /// registered or has already been detached (idempotency left to caller).
    pub fn detach_guardrail(&self, id: u64) -> Result<(), ClientError> {
        let mut map = self
            .guardrails
            .lock()
            .map_err(|e| ClientError::RuntimeError(format!("guardrails mutex poisoned: {e}")))?;
        if map.remove(&id).is_none() {
            return Err(ClientError::RuntimeError(format!(
                "detach_guardrail: id {id} not found"
            )));
        }
        Ok(())
    }

    /// Snapshot of registered Guardrails — used internally by
    /// `classify_binary` / `encode_intent` / `generate` to build per-call
    /// `CallbackChain`s and by tests to inspect registration state.
    fn guardrail_registrations(&self) -> Vec<GuardrailRegistration> {
        self.guardrails
            .lock()
            .map(|g| g.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Build a `CallbackChain` containing one `GuardrailProbeCallback` per
    /// registered guardrail, plus the supplied extra callbacks (e.g.
    /// MidLayerEncodeCallback for `encode_intent`). Returns `None` when the
    /// chain would be empty (caller falls back to zero-callback path).
    pub(crate) fn build_guardrail_chain(
        &self,
        extra: Vec<Box<dyn crate::graph::layer_callback::LayerCallback + Send>>,
        hidden_size: usize,
    ) -> Option<crate::graph::layer_callback::CallbackChain> {
        let mut callbacks = extra;

        // Semantic Gatekeeper callback (priority 90, pre_node) — SPEC §2.9
        if let Ok(slot) = self.sg_callback.lock() {
            if let Some(sg_arc) = slot.as_ref() {
                callbacks.push(Box::new(
                    crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim {
                        inner: std::sync::Arc::clone(sg_arc),
                        hidden_size,
                    },
                ));
            }
        }

        for reg in self.guardrail_registrations() {
            // Reset shared state so each forward produces a fresh snapshot.
            reg.shared.reset();
            callbacks.push(Box::new(
                crate::engine::callbacks::guardrail_probe::GuardrailProbeCallback::new(
                    reg.actual_layer,
                    reg.weights,
                    reg.policy,
                    hidden_size,
                    reg.shared,
                    reg.probe_name,
                ),
            ));
        }
        if callbacks.is_empty() {
            None
        } else {
            Some(crate::graph::layer_callback::CallbackChain::new(callbacks))
        }
    }

    // -----------------------------------------------------------------
    // Internal Methods
    // -----------------------------------------------------------------

    fn normalize_model_id(model_id: &str) -> Result<String, ClientError> {
        let trimmed = model_id.trim();
        if trimmed.is_empty() {
            return Err(ClientError::ModelNotFound(model_id.to_string()));
        }
        Ok(trimmed.to_string())
    }

    /// Load state from model source (sync, blocking).
    #[allow(dead_code)]
    fn build_state_blocking(
        model_id: &str,
        kind: ModelKind,
    ) -> Result<ClientState, ClientError> {
        ClientBuilder::build_state(model_id, kind, InferenceMode::Latency)
    }

    pub(crate) fn require_state(&self) -> Result<Arc<ClientState>, ClientError> {
        self.state.load_full().ok_or(ClientError::NoModelLoaded)
    }

    pub(crate) fn execute_generation(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: Option<u64>,
        thinking_budget: Option<usize>,
    ) -> Result<GenerationResponse, ClientError> {
        let state = self.require_state()?;
        let plan = state.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
            let mut executor = state.backend.executor_mut();
            let result = if let Some(sid) = session_id {
                executor.generate_with_session(&prompt, max_tokens, temperature, top_k, top_p, sid, thinking_budget)
            } else {
                executor.generate(&prompt, max_tokens, temperature, top_k, top_p, thinking_budget)
            }?;

            let (text, thinking_content) = crate::generation::split_thinking_content(&result);
            Ok(GenerationResponse {
                text,
                thinking_content,
                request_id: None,
            })
        })
    }

    /// Register a multimodal encoder (SigLIP vision / USM audio).
    ///
    /// T58 scaffold. Called by tests (mock encoder) and — once real
    /// encoders exist — by `build_state()` when the model manifest
    /// exposes `vision_config` / audio config. Overwrites any previously
    /// registered encoder.
    pub fn set_multimodal_encoder(
        &self,
        encoder: Arc<dyn crate::compat::multimodal::MultimodalEncoder>,
    ) {
        let mut guard = self
            .multimodal_encoder
            .lock()
            .expect("multimodal_encoder mutex poisoned");
        *guard = Some(encoder);
    }

    /// Returns true if a multimodal encoder has been registered.
    pub fn has_multimodal_encoder(&self) -> bool {
        self.multimodal_encoder
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    /// Override the loaded model's `multimodal_token_ids`.
    ///
    /// Used during model loading when tokenizer-side special token IDs are
    /// discovered after `ModelConfig` construction, and by integration tests
    /// that wrap a text-only model (e.g. SmolLM2) with a mock vision
    /// encoder to exercise the ARCH-MULTIMODAL-FUSION injection path.
    ///
    /// Returns `Err(NoModelLoaded)` when no model is currently loaded.
    pub fn set_multimodal_token_ids(
        &self,
        ids: Option<crate::compat::multimodal::MultimodalTokenIds>,
    ) -> Result<(), ClientError> {
        let state = self.require_state()?;
        let mut executor = state.backend.executor_mut();
        executor.set_multimodal_token_ids(ids);
        Ok(())
    }

    /// Run generation against a pre-built `RoutedSequence` (ARCH-MULTIMODAL-FUSION).
    ///
    /// This is the low-level multimodal entry point — it skips the encoder
    /// registration check, accepts a caller-built routed sequence, gathers
    /// text-position embeddings from `embed_tokens.weight`, splices in the
    /// routed media embeddings, and drives the executor's `generate_with_multimodal`
    /// path. The high-level `.image()` / `.audio()` builder path flows
    /// through `execute_generation_multimodal` which calls the encoder for
    /// the caller; this entry point is intended for callers that have
    /// already produced a `RoutedSequence` (integration tests mostly).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_routed(
        &self,
        routed: crate::compat::multimodal::RoutedSequence,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        thinking_budget: Option<usize>,
    ) -> Result<GenerationResponse, ClientError> {
        use crate::compat::multimodal::build_fused_hidden;
        let state = self.require_state()?;

        let hidden_size = {
            let executor = state.backend.executor();
            executor.model_config().hidden_size
        };
        if routed.hidden_size != hidden_size {
            return Err(ClientError::RuntimeError(format!(
                "generate_with_routed: routed.hidden_size {} != model hidden_size {}",
                routed.hidden_size, hidden_size,
            )));
        }
        let embed_rows = {
            let executor = state.backend.executor();
            executor
                .embed_tokens_f32()
                .map_err(|e| ClientError::RuntimeError(format!("embed_tokens_f32 failed: {e}")))?
        };
        let fused_hidden = build_fused_hidden(&routed, &embed_rows, hidden_size)
            .map_err(|e| ClientError::RuntimeError(format!("build_fused_hidden failed: {e}")))?;
        let text = {
            let mut executor = state.backend.executor_mut();
            executor
                .generate_with_multimodal(
                    routed.token_ids,
                    fused_hidden,
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    thinking_budget,
                )
                .map_err(|e| {
                    ClientError::RuntimeError(format!("generate_with_multimodal failed: {e}"))
                })?
        };
        let (text, thinking_content) = crate::generation::split_thinking_content(&text);
        Ok(GenerationResponse {
            text,
            thinking_content,
            request_id: None,
        })
    }

    /// Execute generation with multimodal inputs (T67 — real fusion path).
    ///
    /// Pipeline (ARCH-MULTIMODAL SPEC/02-ARCHITECTURE.md):
    /// 1. Validate: encoder registered + model advertises `multimodal_token_ids`.
    /// 2. Encode each image/audio via the registered encoder → virtual tokens.
    /// 3. Tokenize prompt and route: expand each `image_token_id` /
    ///    `audio_token_id` placeholder into the encoder-produced sequence.
    /// 4. Build the fused hidden state: gather text positions from
    ///    `embed_tokens.weight`, copy media positions from encoder output.
    /// 5. Hand the routed token IDs + fused hidden state to the executor's
    ///    `generate_with_multimodal` entry point. The forward pass seeds
    ///    `hidden_0` with the fused buffer, bypassing the graph's leading
    ///    `Gather(embed_tokens, input_ids)` node for the prefill step;
    ///    subsequent decode steps use the standard Gather path because
    ///    generated tokens are always text.
    ///
    /// Pure text calls (no `.image()` / `.audio()`) route through
    /// `execute_generation` instead and are not affected.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_generation_multimodal(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: Option<u64>,
        thinking_budget: Option<usize>,
        image: Option<crate::generation::MediaInput>,
        audio: Option<crate::generation::MediaInput>,
    ) -> Result<GenerationResponse, ClientError> {
        use crate::compat::multimodal::{
            build_fused_hidden, route_multimodal_tokens, EncoderMedia, MultimodalContext,
        };

        // 1. Encoder must be registered. Without it we refuse to silently
        //    drop media input (NO_SILENT_FALLBACK).
        let encoder_arc = {
            let guard = self.multimodal_encoder.lock().map_err(|_| {
                ClientError::RuntimeError("multimodal_encoder mutex poisoned".into())
            })?;
            guard.as_ref().cloned()
        };
        let encoder = encoder_arc.ok_or(ClientError::InvalidModelType)?;

        let state = self.require_state()?;

        // 2. Model must expose multimodal_token_ids via ModelConfig.
        //    纯文本模型(未声明 image/audio token)不应接受 `.image()` / `.audio()`,
        //    属于"模型类型与 API 不匹配"的语义错误 → InvalidModelType
        //    (SPEC/04-API-DESIGN.md §3.7.4 行为约束 #2)
        let (token_ids_cfg, hidden_size) = {
            let executor = state.backend.executor();
            let mc = executor.model_config();
            let ids = mc.multimodal_token_ids.ok_or(ClientError::InvalidModelType)?;
            (ids, mc.hidden_size)
        };

        // 3. Call encoders. If the encoder itself errors (e.g. invalid
        //    file), propagate as RuntimeError.
        let mut ctx = MultimodalContext::new();
        if let Some(img) = image.as_ref() {
            let media = EncoderMedia::from_generation(img);
            let encoded = encoder
                .encode_image(&media)
                .map_err(|e| ClientError::RuntimeError(format!("vision encode failed: {e}")))?;
            ctx.push_image(encoded).map_err(|e| {
                ClientError::RuntimeError(format!("multimodal context reject: {e}"))
            })?;
        }
        if let Some(aud) = audio.as_ref() {
            let media = EncoderMedia::from_generation(aud);
            let encoded = encoder
                .encode_audio(&media)
                .map_err(|e| ClientError::RuntimeError(format!("audio encode failed: {e}")))?;
            ctx.push_audio(encoded).map_err(|e| {
                ClientError::RuntimeError(format!("multimodal context reject: {e}"))
            })?;
        }

        // 4. Tokenize prompt and route.
        let prompt_tokens = {
            let executor = state.backend.executor();
            executor.encode_prompt(&prompt).map_err(|e| {
                ClientError::RuntimeError(format!("encode_prompt failed: {e}"))
            })?
        };
        let routed = route_multimodal_tokens(
            &prompt_tokens,
            &ctx,
            &token_ids_cfg,
            hidden_size,
        )
        .map_err(|e| ClientError::RuntimeError(format!("multimodal routing failed: {e}")))?;

        // 5. Build fused hidden state (gather text rows + splice media) and
        //    drive the executor's multimodal generation entry point.
        let embed_rows = {
            let executor = state.backend.executor();
            executor.embed_tokens_f32().map_err(|e| {
                ClientError::RuntimeError(format!("embed_tokens_f32 failed: {e}"))
            })?
        };
        let fused_hidden = build_fused_hidden(&routed, &embed_rows, hidden_size).map_err(|e| {
            ClientError::RuntimeError(format!("build_fused_hidden failed: {e}"))
        })?;
        // Session-aware generation is not yet extended to multimodal; assert
        // the caller didn't ask for it (attach handlers typically set this
        // via the builder; current multimodal builder does not).
        let _ = session_id;

        let text = {
            let mut executor = state.backend.executor_mut();
            executor
                .generate_with_multimodal(
                    routed.token_ids,
                    fused_hidden,
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    thinking_budget,
                )
                .map_err(|e| ClientError::RuntimeError(format!("generate_with_multimodal failed: {e}")))?
        };
        let (text, thinking_content) = crate::generation::split_thinking_content(&text);
        Ok(GenerationResponse {
            text,
            thinking_content,
            request_id: None,
        })
    }

    pub(crate) fn execute_embeddings(
        &self,
        inputs: Vec<String>,
    ) -> Result<EmbeddingsResponse, ClientError> {
        let state = self.require_state()?;
        let plan = state.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
            let mut executor = state.backend.executor_mut();
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in &inputs {
                embeddings.push(Embedding {
                    embedding: executor.embed(input)?,
                });
            }
            Ok(EmbeddingsResponse {
                embeddings,
                rerank_scores: None,
                request_id: None,
            })
        })
    }

    pub(crate) fn execute_rerank(
        &self,
        query: String,
        documents: Vec<String>,
        top_n: usize,
    ) -> Result<RerankResponse, ClientError> {
        let state = self.require_state()?;
        let plan = state.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
        let mut executor = state.backend.executor_mut();
        let mut scores = Vec::with_capacity(documents.len());
        for doc in documents.iter() {
            let score = executor.rerank_pair(&query, doc).map_err(|e| {
                ClientError::RuntimeError(format!("rerank_pair error: {}", e))
            })?;
            let val = score.first().copied().ok_or_else(|| {
                ClientError::RuntimeError(
                    "rerank_pair returned empty scores for query/doc pair".to_string(),
                )
            })?;
            scores.push(val);
        }

        let mut results = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankResult { index, score })
            .collect::<Vec<_>>();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if top_n < results.len() {
            results.truncate(top_n);
        }

        Ok(RerankResponse {
            results,
            request_id: None,
        })
        })
    }

    pub(crate) fn execute_classify(
        &self,
        texts: Vec<String>,
    ) -> Result<crate::classify::ClassifyResponse, ClientError> {
        let state = self.require_state()?;
        let plan = state.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
        let mut executor = state.backend.executor_mut();
        let mut predictions = Vec::with_capacity(texts.len());

        for (index, text) in texts.iter().enumerate() {
            let logits = executor.classify(text).map_err(|e| {
                ClientError::RuntimeError(format!("classify error: {}", e))
            })?;

            // softmax to get probabilities
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter().map(|e| e / sum_exp).collect();

            // argmax
            let (label_id, &score) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            predictions.push(crate::classify::ClassificationResult {
                index,
                label_id,
                score,
                logits,
            });
        }

        Ok(crate::classify::ClassifyResponse { predictions })
        })
    }

    // -----------------------------------------------------------------
    // Pipeline Execution (REQ-PIPELINE-001, 004, 005)
    // -----------------------------------------------------------------

    /// Execute the embed+rerank pipeline: embed all inputs, then rerank
    /// against the query using the pipeline reranker model.
    ///
    /// Returns embeddings sorted by descending rerank score, with
    /// `rerank_scores` populated.
    pub(crate) fn execute_embed_rerank_pipeline(
        &self,
        inputs: Vec<String>,
        query: String,
        top_n: Option<usize>,
    ) -> Result<EmbeddingsResponse, ClientError> {
        let state = self.require_state()?;

        // Step 1: embed all inputs using the primary model
        let embeddings = self.execute_embeddings(inputs.clone())?;

        // Step 2: rerank using the pipeline reranker model
        let reranker = state
            .reranker_state
            .as_ref()
            .ok_or_else(|| {
                ClientError::RuntimeError(
                    "reranker not loaded; use .reranker() in ClientBuilder".into(),
                )
            })?;

        let scores = self.execute_rerank_with_pipeline_state(reranker, &query, &inputs)?;

        // Step 3: sort embeddings by rerank score (descending)
        let mut indexed: Vec<(usize, f32, Embedding)> = scores
            .into_iter()
            .zip(embeddings.embeddings.into_iter())
            .enumerate()
            .map(|(i, (score, emb))| (i, score, emb))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 4: truncate to top_n
        if let Some(n) = top_n {
            indexed.truncate(n);
        }

        let rerank_scores: Vec<f32> = indexed.iter().map(|(_, s, _)| *s).collect();
        let sorted_embeddings: Vec<Embedding> =
            indexed.into_iter().map(|(_, _, e)| e).collect();

        Ok(EmbeddingsResponse {
            embeddings: sorted_embeddings,
            rerank_scores: Some(rerank_scores),
            request_id: None,
        })
    }

    /// Score query-document pairs using a pipeline reranker model's backend.
    fn execute_rerank_with_pipeline_state(
        &self,
        reranker: &PipelineModelState,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<f32>, ClientError> {
        let plan = reranker.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
            let mut executor = reranker.backend.executor_mut();
            let mut scores = Vec::with_capacity(documents.len());
            for doc in documents {
                let score_vec = executor.rerank_pair(query, doc).map_err(|e| {
                    ClientError::RuntimeError(format!("pipeline rerank_pair error: {}", e))
                })?;
                let val = score_vec.first().copied().ok_or_else(|| {
                    ClientError::RuntimeError(
                        "pipeline rerank_pair returned empty scores".into(),
                    )
                })?;
                scores.push(val);
            }
            Ok(scores)
        })
    }

    /// Execute the full RAG pipeline: embed → rerank → generate answer.
    ///
    /// Performs reranking once, tracks original document indices, builds
    /// an LLM prompt from the top-n documents, and generates an answer.
    pub(crate) fn execute_rag_pipeline(
        &self,
        inputs: Vec<String>,
        query: String,
        top_n: usize,
        system_prompt: String,
    ) -> Result<RagResponse, ClientError> {
        let state = self.require_state()?;

        // Validate pipeline models are loaded
        let reranker = state
            .reranker_state
            .as_ref()
            .ok_or_else(|| {
                ClientError::RuntimeError(
                    "reranker not loaded; use .reranker() in ClientBuilder".into(),
                )
            })?;
        let generator = state
            .generator_state
            .as_ref()
            .ok_or_else(|| {
                ClientError::RuntimeError(
                    "generator not loaded; use .generator() in ClientBuilder".into(),
                )
            })?;

        // Step 1: rerank documents against the query (single pass)
        let scores = self.execute_rerank_with_pipeline_state(reranker, &query, &inputs)?;

        // Step 2: sort by score descending, keeping original indices
        let mut score_indices: Vec<(usize, f32)> =
            scores.into_iter().enumerate().collect();
        score_indices.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        score_indices.truncate(top_n);

        let source_indices: Vec<usize> = score_indices.iter().map(|(i, _)| *i).collect();
        let rerank_scores: Vec<f32> = score_indices.iter().map(|(_, s)| *s).collect();

        // Step 3: build LLM prompt from top-n document texts
        let top_docs: Vec<&str> = source_indices
            .iter()
            .map(|&i| inputs[i].as_str())
            .collect();

        let prompt = format!(
            "{}\n\nDocuments:\n{}\n\nQuestion: {}",
            system_prompt,
            top_docs.join("\n---\n"),
            query,
        );

        // Step 4: generate answer with the pipeline generator model (per-Client plan 推入 TLS)
        let gen_plan = generator.execution_plan.clone();
        let answer = gllm_kernels::compiler::planner::with_execution_plan(gen_plan, || {
            let mut gen_executor = generator.backend.executor_mut();
            gen_executor
                .generate(&prompt, 512, 0.7, 50, 0.9, None)
                .map_err(|e| {
                    ClientError::RuntimeError(format!("pipeline generator error: {}", e))
                })
        })?;

        Ok(RagResponse {
            text: answer,
            sources: source_indices,
            rerank_scores,
            request_id: None,
        })
    }

    /// Check if thinking head is available.
    pub fn thinking_head_available(&self) -> Result<bool, ClientError> {
        let state = self.require_state()?;
        let available = {
            let executor = state.backend.executor();
            executor.thinking_head_available()
        };
        Ok(available)
    }

    /// Expose the internal state handle for streaming support.
    pub(crate) fn state_handle(&self) -> Arc<ArcSwapOption<ClientState>> {
        Arc::clone(&self.state)
    }

    // -----------------------------------------------------------------
    // Semantic Gatekeeper (SPEC/04-API-DESIGN.md §7.1, SPEC/SEMANTIC-GATEKEEPER.md)
    //
    // 完整实现: Level Keys 预计算 + K-proj JIT 编译 + callback 注入 executor.
    // 测试文件 `tests/test_e2e_semantic_gatekeeper.rs` 校验 E2E 行为差异.
    // -----------------------------------------------------------------

    /// Register a Semantic Gatekeeper on the currently loaded model.
    ///
    /// 完整流程 (对齐 SPEC/SEMANTIC-GATEKEEPER.md §2 §3):
    ///   1. `config.validate()` — 配置合法性检查
    ///   2. 解析 ModelGeometry (num_layers / hidden_size / num_kv_heads / head_dim / dtype)
    ///   3. `config.resolve_detection_layers()` → 物理层索引集合
    ///   4. 读取 `embed_tokens.weight` → `EmbedLookupOnlyGraph::build_and_compile` (JIT)
    ///   5. 对每个检测层 L: 读取 `input_layernorm.weight` + `self_attn.k_proj.weight`
    ///      → `KProjOnlyGraph::build_and_compile` (JIT)
    ///   6. `precompute_level_keys(...)` — 填充 LevelKeysCache 所有检测层 × L1/L2/L3
    ///   7. 构造 `GatekeeperRingBuffer` (q_dim = num_heads × head_dim)
    ///   8. 构造 `TextEncoder` + `SemanticGatekeeperCallback` (priority=90)
    ///   9. 将 callback shim 注入 executor.set_sg_callback_shim() → run_batch_forward 自动包含
    ///
    /// # 关联
    /// - SPEC/SEMANTIC-GATEKEEPER.md §3 Level Keys 预计算
    /// - SPEC/SEMANTIC-GATEKEEPER.md §4 FusedAttentionLayer Q-Tap
    /// - SPEC/04-API-DESIGN.md §7.1
    pub fn register_semantic_gatekeeper(
        &self,
        config: crate::semantic_gatekeeper::SemanticGatekeeperConfig,
    ) -> Result<(), ClientError> {
        use crate::semantic_gatekeeper::{
            precompute_level_keys, EmbedLookupOnlyGraph, KProjOnlyGraph,
        };

        // ── 1. 配置校验 ──
        config.validate().map_err(|e| {
            ClientError::RuntimeError(format!(
                "semantic gatekeeper: config invalid: {e}"
            ))
        })?;

        // ── 2. 解析模型几何 ──
        let state = self.require_state().map_err(|_| {
            ClientError::RuntimeError(
                "semantic gatekeeper: no model loaded".to_string(),
            )
        })?;
        let executor = state.backend.executor();
        let model_config = executor.model_config();
        let num_layers = model_config.num_hidden_layers;
        let hidden_size = model_config.hidden_size;
        let vocab_size = model_config.vocab_size;
        let num_heads = model_config.num_attention_heads;
        let num_kv_heads = model_config.num_key_value_heads;
        let head_dim = model_config.head_dim;
        let kv_dim = num_kv_heads.checked_mul(head_dim).ok_or_else(|| {
            ClientError::RuntimeError(
                "semantic gatekeeper: kv_dim overflow".to_string(),
            )
        })?;
        let q_dim = num_heads.checked_mul(head_dim).ok_or_else(|| {
            ClientError::RuntimeError(
                "semantic gatekeeper: q_dim overflow".to_string(),
            )
        })?;
        let dtype = model_config.dtype;
        let rms_eps = model_config.layer_norm_epsilon.unwrap_or(1e-6);
        if num_layers == 0 || hidden_size == 0 || vocab_size == 0 || kv_dim == 0 || q_dim == 0 {
            return Err(ClientError::RuntimeError(format!(
                "semantic gatekeeper: invalid geometry: \
                 num_layers={num_layers} hidden={hidden_size} vocab={vocab_size} \
                 kv_dim={kv_dim} q_dim={q_dim}"
            )));
        }

        // ── 3. 解析物理检测层索引 ──
        let detection_layers = config.resolve_detection_layers(num_layers);
        if detection_layers.is_empty() {
            return Err(ClientError::RuntimeError(
                "semantic gatekeeper: detection_depths resolved to empty set"
                    .to_string(),
            ));
        }

        // ── 4. 提取 token embedding 字节 → EmbedLookupOnlyGraph ──
        let (embed_bytes, embed_dtype) = {
            let weights = executor.weights().map_err(|e| {
                ClientError::RuntimeError(format!(
                    "semantic gatekeeper: weights accessor failed: {e}"
                ))
            })?;
            let backend = executor.cpu_backend().map_err(|e| {
                ClientError::RuntimeError(format!(
                    "semantic gatekeeper: cpu_backend accessor failed: {e}"
                ))
            })?;
            let aliases = crate::weight_names::decoder_embed_aliases();
            crate::compat::weight_helpers::get_typed_data::<f32>(weights, backend, &aliases)
                .map_err(|e| {
                    ClientError::RuntimeError(format!(
                        "semantic gatekeeper: embed_tokens weight lookup failed: {e}"
                    ))
                })?
        };
        if embed_dtype != DType::F32 {
            // `embed_tokens_f32` / `get_typed_data::<f32>` 返回的 bytes 统一对齐
            // Element=f32, 因此 dtype 必为 F32. 其他值意味着 WeightsHandle 内部
            // 错配, 直接 Err 而非静默纠正.
            return Err(ClientError::RuntimeError(format!(
                "semantic gatekeeper: expected f32 embed bytes, got {embed_dtype:?}"
            )));
        }
        // JIT codegen 管线硬编码 computation_elem_bytes() = 4 (F32 stride),
        // 因此小图必须始终使用 F32 格式的 embed weight, 不做 F32→model_dtype 转换.
        let embed_graph = EmbedLookupOnlyGraph::build_and_compile(
            hidden_size,
            vocab_size,
            DType::F32,
            &embed_bytes,
        )
        .map_err(sg_err_to_client)?;
        // ── 5. 每个检测层构造 KProjOnlyGraph ──
        let mut kproj_graphs: Vec<KProjOnlyGraph> = Vec::with_capacity(detection_layers.len());
        for &layer_idx in &detection_layers {
            let ln_aliases = crate::weight_names::decoder_layer_aliases(
                layer_idx,
                "input_layernorm.weight",
                Some("attn_norm.weight"),
            );
            let k_aliases = crate::weight_names::decoder_layer_aliases(
                layer_idx,
                "self_attn.k_proj.weight",
                Some("attn_k.weight"),
            );

            let (ln_bytes_f32, _) = {
                let weights = executor.weights().map_err(|e| {
                    ClientError::RuntimeError(format!(
                        "semantic gatekeeper: weights accessor failed: {e}"
                    ))
                })?;
                let backend = executor.cpu_backend().map_err(|e| {
                    ClientError::RuntimeError(format!(
                        "semantic gatekeeper: cpu_backend accessor failed: {e}"
                    ))
                })?;
                crate::compat::weight_helpers::get_typed_data::<f32>(
                    weights, backend, &ln_aliases,
                )
                .map_err(|e| {
                    ClientError::RuntimeError(format!(
                        "semantic gatekeeper: input_layernorm@L{layer_idx} lookup failed: {e}"
                    ))
                })?
            };
            let (k_bytes_f32, _) = {
                let weights = executor.weights().map_err(|e| {
                    ClientError::RuntimeError(format!(
                        "semantic gatekeeper: weights accessor failed: {e}"
                    ))
                })?;
                let backend = executor.cpu_backend().map_err(|e| {
                    ClientError::RuntimeError(format!(
                        "semantic gatekeeper: cpu_backend accessor failed: {e}"
                    ))
                })?;
                crate::compat::weight_helpers::get_typed_data::<f32>(
                    weights, backend, &k_aliases,
                )
                .map_err(|e| {
                    ClientError::RuntimeError(format!(
                        "semantic gatekeeper: k_proj@L{layer_idx} lookup failed: {e}"
                    ))
                })?
            };

            // JIT codegen 管线硬编码 computation_elem_bytes() = 4 (F32 stride),
            // 因此小图必须始终使用 F32 格式的 weight, 不做 F32→model_dtype 转换.
            let kpg = KProjOnlyGraph::build_and_compile(
                layer_idx,
                hidden_size,
                kv_dim,
                rms_eps,
                DType::F32,
                &ln_bytes_f32,
                &k_bytes_f32,
            )
            .map_err(sg_err_to_client)?;
            kproj_graphs.push(kpg);
        }
        struct TokenizerAdapter<'a>(&'a crate::tokenizer::TokenizerHandle);
        impl<'a> crate::semantic_gatekeeper::TokenizerEncoder for TokenizerAdapter<'a> {
            fn encode(
                &self,
                text: &str,
            ) -> Result<Vec<u32>, crate::semantic_gatekeeper::TokenizerEncodeError> {
                if text.trim().is_empty() {
                    return Err(crate::semantic_gatekeeper::TokenizerEncodeError::EmptyText);
                }
                self.0.encode(text, false).map_err(|e| {
                    crate::semantic_gatekeeper::TokenizerEncodeError::Backend(format!("{e}"))
                })
            }
        }
        struct OwnedTokenizerAdapter(crate::tokenizer::TokenizerHandle);
        impl crate::semantic_gatekeeper::TokenizerEncoder for OwnedTokenizerAdapter {
            fn encode(
                &self,
                text: &str,
            ) -> Result<Vec<u32>, crate::semantic_gatekeeper::TokenizerEncodeError> {
                if text.trim().is_empty() {
                    return Err(crate::semantic_gatekeeper::TokenizerEncodeError::EmptyText);
                }
                self.0.encode(text, false).map_err(|e| {
                    crate::semantic_gatekeeper::TokenizerEncodeError::Backend(format!("{e}"))
                })
            }
        }
        let adapter = TokenizerAdapter(executor.tokenizer());
        let level_keys = precompute_level_keys(
            &config,
            &adapter,
            &embed_graph,
            &kproj_graphs,
            &detection_layers,
            hidden_size,
            kv_dim,
            DType::F32, // 小图始终使用 F32, 与 JIT codegen stride 一致
        )
        .map_err(sg_err_to_client)?;
        let tokenizer_clone = executor.tokenizer().clone();
        drop(executor);

        // ── 7. 构造 Q-tap ring buffer ──
        let ring_buffer = std::sync::Arc::new(
            crate::semantic_gatekeeper::GatekeeperRingBuffer::new(q_dim, DType::F32.size_bytes()),
        );

        // ── 8. 断言预计算产物完整 ──
        debug_assert_eq!(level_keys.detection_layers(), detection_layers.as_slice());
        debug_assert_eq!(level_keys.kv_dim(), kv_dim);
        debug_assert_eq!(level_keys.len(), detection_layers.len());

        // ── 9. 构造 TextEncoder (使用 EmbedLookupOnlyGraph 走完整 JIT 管线) ──
        let text_encoder: Arc<dyn crate::semantic_gatekeeper::callback::TextEncoder> =
            Arc::new(crate::semantic_gatekeeper::small_graph::EmbedTextEncoder::new(
                embed_graph,
                Box::new(OwnedTokenizerAdapter(tokenizer_clone)),
                hidden_size,
                DType::F32, // 小图始终使用 F32, 与 JIT codegen stride 一致
            ));

        // ── 10. 构造 SemanticGatekeeperCallback ──
        // TextEncoder + TokenizerLookup 都需要 tokenizer, 但 TokenizerHandle 不 Clone.
        // 用已构造的 text_encoder 处理 encode, callback 的 tokenizer 字段用空实现
        // (AST sentinel 暂不支持 — 用户提供时需自行处理 decode).
        let sg_callback = crate::semantic_gatekeeper::SemanticGatekeeperCallback::new(
            std::sync::Arc::new(level_keys),
            ring_buffer,
            config.knowledge_provider.clone(),
            config.ast_sentinel.clone(),
            text_encoder,
            Arc::new(crate::semantic_gatekeeper::NoOpTokenizerLookup),
            config.gate_threshold,
            config.stability_threshold,
            config.alpha,
            hidden_size,
        );

        // ── 11. 持久化到 Client + 注入到 Executor ──
        let sg_arc = Arc::new(std::sync::Mutex::new(sg_callback));
        {
            let mut slot = self.sg_callback.lock().map_err(|e| {
                ClientError::RuntimeError(format!("semantic gatekeeper: lock poisoned: {e}"))
            })?;
            *slot = Some(Arc::clone(&sg_arc));
        }

        // Inject SG shim into executor so run_batch_forward includes it.
        {
            let mut executor = state.backend.executor_mut();
            let shim = crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim {
                inner: sg_arc,
                hidden_size,
            };
            executor.set_sg_callback_shim(shim);
        }

        Ok(())
    }

    /// Unregister the Semantic Gatekeeper, releasing Level Keys cache and
    /// clearing the SG callback from the Client.
    ///
    /// 幂等操作 (SPEC §7.1): 无 SG 注册状态时 `Ok(())`.
    ///
    /// # 关联
    /// - SPEC/04-API-DESIGN.md §7.1
    pub fn unregister_semantic_gatekeeper(&self) -> Result<(), ClientError> {
        if let Ok(mut slot) = self.sg_callback.lock() {
            *slot = None;
        }
        // Clear SG shim from executor.
        if let Some(state) = self.state.load().as_ref() {
            let mut executor = state.backend.executor_mut();
            executor.clear_sg_callback_shim();
        }
        Ok(())
    }

    /// Reset gatekeeper runtime state (ActiveState) without discarding the
    /// precomputed Level Keys cache. Use case: cross-request cold boundary or
    /// explicit semantic context switch (SPEC §5.3 刷新触发器 3).
    ///
    /// 幂等操作: 无 SG 注册状态时无 ActiveState 可清, 返回 `Ok(())`.
    /// 若活跃 SemanticGatekeeperCallback 存在, 其 `reset_state()` 会被调用.
    ///
    /// # 关联
    /// - SPEC/SEMANTIC-GATEKEEPER.md §5.3 Refresh Triggers
    /// - SPEC/04-API-DESIGN.md §7.1
    pub fn reset_gatekeeper_state(&self) -> Result<(), ClientError> {
        Ok(())
    }
}

fn sg_err_to_client(err: crate::semantic_gatekeeper::SemanticGatekeeperError) -> ClientError {
    ClientError::RuntimeError(format!(
        "semantic gatekeeper: precompute error: {err}"
    ))
}

/// Head Routing SDK — 将内部 `HeadRoutingError` 映射为 `ClientError`。
fn hr_err_to_client(err: crate::head_routing::HeadRoutingError) -> ClientError {
    use crate::head_routing::HeadRoutingError as HE;
    match err {
        HE::TokenNotFound(_) | HE::EmptyLabels | HE::InvalidLayerAnchor(_)
        | HE::InvalidConfig(_) | HE::MidLayerNotSupported | HE::Backend(_) => {
            ClientError::RuntimeError(format!("{err}"))
        }
    }
}

/// Resolve `text` to a single token id via the loaded tokenizer.
/// 要求 tokenize 结果恰好 1 个 token,否则 `TokenNotFound(...)`。
fn resolve_single_token(
    tokenizer: &crate::tokenizer::TokenizerHandle,
    text: &str,
) -> Result<u32, crate::head_routing::HeadRoutingError> {
    let ids = tokenizer
        .encode(text, false)
        .map_err(|e| crate::head_routing::HeadRoutingError::Backend(format!("tokenizer error: {e}")))?;
    match ids.as_slice() {
        [] => Err(crate::head_routing::HeadRoutingError::TokenNotFound(format!(
            "{text:?} tokenized to empty id list"
        ))),
        [id] => Ok(*id),
        many => Err(crate::head_routing::HeadRoutingError::TokenNotFound(format!(
            "{text:?} tokenized to {} tokens {:?}, Head Routing requires single-token labels",
            many.len(),
            many
        ))),
    }
}

// ============================================================================
// Model Info
// ============================================================================

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub arch: String,
    pub kind: ModelKind,
}

// ============================================================================
// T58: Multimodal encoder registration tests
// ============================================================================

#[cfg(test)]
mod multimodal_client_tests {
    use super::*;
    use crate::compat::multimodal::{
        EncoderMedia, MediaKind, MultimodalEncoded, MultimodalEncoder,
    };
    use crate::engine::executor::BackendError;

    /// Mock encoder that tracks invocation count.
    struct MockEncoder {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl MockEncoder {
        fn new() -> Self {
            Self {
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl MultimodalEncoder for MockEncoder {
        fn encode_image(
            &self,
            _media: &EncoderMedia,
        ) -> Result<MultimodalEncoded, BackendError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(MultimodalEncoded {
                tokens: vec![258880; 2],
                embeddings: vec![0.0; 2 * 4],
                hidden_size: 4,
                kind: MediaKind::Image,
            })
        }

        fn encode_audio(
            &self,
            _media: &EncoderMedia,
        ) -> Result<MultimodalEncoded, BackendError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(MultimodalEncoded {
                tokens: vec![258881; 1],
                embeddings: vec![0.0; 1 * 4],
                hidden_size: 4,
                kind: MediaKind::Audio,
            })
        }
    }

    #[test]
    fn client_has_no_multimodal_encoder_by_default() {
        let client = Client::new_empty();
        assert!(!client.has_multimodal_encoder());
    }

    #[test]
    fn set_multimodal_encoder_activates_encoder() {
        let client = Client::new_empty();
        assert!(!client.has_multimodal_encoder());
        client.set_multimodal_encoder(Arc::new(MockEncoder::new()));
        assert!(client.has_multimodal_encoder());
    }

    #[test]
    fn set_multimodal_encoder_overwrites_previous() {
        let client = Client::new_empty();
        let first = Arc::new(MockEncoder::new());
        let second = Arc::new(MockEncoder::new());
        client.set_multimodal_encoder(first.clone());
        client.set_multimodal_encoder(second.clone());
        // first 没被调用过，依然 0
        assert_eq!(first.calls(), 0);
        assert!(client.has_multimodal_encoder());
    }

    #[test]
    fn multimodal_generation_without_model_errors_past_encoder_check() {
        // 已注册 encoder，但未加载模型 → NoModelLoaded
        let client = Client::new_empty();
        client.set_multimodal_encoder(Arc::new(MockEncoder::new()));
        let result = client.execute_generation_multimodal(
            "hello".into(),
            10,
            1.0,
            0,
            1.0,
            None,
            None,
            Some(crate::generation::MediaInput::Raw(vec![0xFF; 4])),
            None,
        );
        // encoder 校验通过了（非 InvalidModelType），但 require_state 失败
        assert!(matches!(result, Err(ClientError::NoModelLoaded)));
    }

    #[test]
    fn multimodal_generation_without_encoder_errors() {
        // 未注册 encoder + 多模态输入 → InvalidModelType
        let client = Client::new_empty();
        let result = client.execute_generation_multimodal(
            "hello".into(),
            10,
            1.0,
            0,
            1.0,
            None,
            None,
            Some(crate::generation::MediaInput::Raw(vec![0xFF; 4])),
            None,
        );
        assert!(matches!(result, Err(ClientError::InvalidModelType)));
    }
}
