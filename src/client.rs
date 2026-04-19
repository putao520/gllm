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
use crate::compat::{forward_to_semantic_layer, layer_target_to_idx};
use crate::embeddings::{Embedding, EmbeddingsResponse, RagResponse};
use crate::engine::executor::{BackendError, ExecutorError};
use crate::generation::GenerationResponse;
use crate::knowledge::LayerTarget;
use crate::loader::{Loader, LoaderConfig, LoaderError, WeightFormat};
use crate::manifest::{
    map_architecture_token_for_kind, MoEConfig, ModelKind, ModelManifest,
    EMPTY_FILE_MAP,
};
use crate::rerank::{RerankResponse, RerankResult};
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
                )?
            );
        }

        if let Some(ref generator_id) = self.generator_model_id {
            state.generator_state =
                Some(Self::build_pipeline_model(generator_id, ModelKind::Chat)?);
        }

        Ok(Client {
            state: Arc::new(ArcSwapOption::from_pointee(state)),
            multimodal_encoder: Arc::new(std::sync::Mutex::new(None)),
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

        // Strategy Arbiter: compute StrategyBias from InferenceMode × GraphArchetype × Hardware.
        // Must happen BEFORE BackendContext::new() which triggers JIT compilation.
        if let Some(ref model_cfg) = model_config_for_arbiter {
            use crate::engine::arbiter::ArbiterHwView;

            let graph_profile = crate::graph::profile::GraphProfiler::profile(model_cfg);
            let archetype = crate::graph::profile::GraphArchetype::derive(&graph_profile);

            // Build hardware view from the already-detected backend type.
            // GPU backends get ArbiterHwView::gpu() for correct bias adjustments
            // (SPEC §4.3.3: epilogue×1.2, k_depth×1.2, pipeline×1.2 on GPU).
            let hw_view = match backend_type {
                BackendType::Cuda | BackendType::Rocm | BackendType::Metal => {
                    ArbiterHwView::gpu(49152)
                }
                BackendType::Cpu => {
                    let profile = gllm_kernels::dispatch::device_profile();
                    ArbiterHwView::from(profile)
                }
            };

            // StrategyArbiter now returns gllm_kernels::compiler::planner::StrategyBias
            // directly — no field-by-field copy needed.
            let arbiter_bias = crate::engine::arbiter::StrategyArbiter::arbitrate(
                inference_mode,
                &archetype,
                &hw_view,
            );

            gllm_kernels::compiler::planner::init_global_execution_plan_with_bias(&arbiter_bias);
        }

        let config_path = loader.config_path().map(|p| p.to_path_buf());
        let tokenizer_path = loader.tokenizer_path().map(|p| p.to_path_buf());
        let weight_paths = loader.weight_paths().to_vec();

        let manifest = Arc::new(manifest);

        let backend = BackendContext::new(
            model_id.to_string(),
            manifest.clone(),
            detected_backend,
            weight_paths,
            config_path,
            tokenizer_path,
        )?;

        Ok(ClientState {
            model_id: model_id.to_string(),
            manifest,
            backend: Arc::new(backend),
            inference_mode,
            reranker_state: None,
            generator_state: None,
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
    ) -> Result<PipelineModelState, ClientError> {
        // Resolve the pipeline model's manifest to determine its architecture.
        let pipeline_manifest = Self::resolve_manifest(model_id, kind)?;

        if pipeline_manifest.arch == primary_manifest.arch {
            // Same architecture: share the primary model's encoder backend.
            // The reranker uses CLS→Classifier while the embedder uses
            // MeanPool→L2Norm, but the encoder forward pass is identical.
            log::info!(
                "pipeline: sharing encoder weights between primary ({}) and pipeline ({}) — architecture {}",
                primary_manifest.model_id, model_id, pipeline_manifest.arch,
            );
            Ok(PipelineModelState {
                model_id: model_id.to_string(),
                manifest: Arc::new(pipeline_manifest),
                backend: Arc::clone(primary_backend),
                shared_encoder: true,
            })
        } else {
            // Different architecture: load independently.
            log::info!(
                "pipeline: loading independent backend for {} (arch {} != primary {})",
                model_id, pipeline_manifest.arch, primary_manifest.arch,
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
        })
    }

    /// Create an empty client (no model loaded).
    pub fn new_empty() -> Self {
        Self {
            state: Arc::new(ArcSwapOption::empty()),
            multimodal_encoder: Arc::new(std::sync::Mutex::new(None)),
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

    /// Execute generation with multimodal inputs (T58 scaffold).
    ///
    /// Pipeline:
    /// 1. Validate: multimodal encoder registered + model advertises
    ///    `multimodal_token_ids`
    /// 2. Encode each image / audio via the registered encoder
    /// 3. Encode the prompt → token IDs
    /// 4. Route tokens: expand each `image_token_id` / `audio_token_id`
    ///    placeholder into the encoder-produced virtual token sequence
    /// 5. Run generation
    ///
    /// The routed embedding sequence is computed and validated here but
    /// not yet consumed by the decoder's embedding layer — embedding-side
    /// fusion is a follow-up task (the executor still gathers text
    /// embeddings for the original prompt). This scaffold guarantees:
    /// - The API round-trips user input (image / audio paths / bytes)
    /// - Token routing produces the correct expanded sequence
    /// - Encoders are consulted and errors surfaced
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
            route_multimodal_tokens, EncoderMedia, MultimodalContext,
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
        let (token_ids_cfg, hidden_size) = {
            let executor = state.backend.executor();
            let mc = executor.model_config();
            let ids = mc.multimodal_token_ids.ok_or_else(|| {
                ClientError::RuntimeError(
                    "model does not advertise multimodal_token_ids (vision_config missing)"
                        .into(),
                )
            })?;
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

        // 5. Run generation. The routed embedding sequence has been
        //    validated; full decoder-side fusion arrives alongside the
        //    real SigLIP / Conformer implementations in a follow-up.
        debug_assert!(
            routed.seq_len() >= prompt_tokens.len(),
            "routing can only grow the sequence"
        );
        let _ = routed; // explicit consume; silence unused-var lint

        let mut executor = state.backend.executor_mut();
        let result = if let Some(sid) = session_id {
            executor.generate_with_session(
                &prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                sid,
                thinking_budget,
            )
        } else {
            executor.generate(
                &prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                thinking_budget,
            )
        }?;

        let (text, thinking_content) = crate::generation::split_thinking_content(&result);
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
    }

    pub(crate) fn execute_rerank(
        &self,
        query: String,
        documents: Vec<String>,
        top_n: usize,
    ) -> Result<RerankResponse, ClientError> {
        let state = self.require_state()?;
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
    }

    pub(crate) fn execute_classify(
        &self,
        texts: Vec<String>,
    ) -> Result<crate::classify::ClassifyResponse, ClientError> {
        let state = self.require_state()?;
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

        // Step 4: generate answer with the pipeline generator model
        let mut gen_executor = generator.backend.executor_mut();
        let answer = gen_executor
            .generate(&prompt, 512, 0.7, 50, 0.9, None)
            .map_err(|e| {
                ClientError::RuntimeError(format!("pipeline generator error: {}", e))
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

    // ========================================================================
    // Knowledge Injection API (per SPEC 04-API-DESIGN §7.2)
    // ========================================================================

    /// Inject knowledge into the model (per SPEC 04-API-DESIGN §7.2).
    ///
    /// Lock-free: reads state via `ArcSwapOption::load()`, which gives an
    /// `Arc<ClientState>` that stays alive for the duration of the operation,
    /// even if another thread swaps the model concurrently.
    pub fn inject_knowledge(
        &self,
        source: crate::knowledge::KnowledgeSource,
        target: crate::knowledge::LayerTarget,
    ) -> Result<crate::knowledge::KnowledgeInjectionResult, ClientError> {
        use crate::knowledge::{InjectionKind, KnowledgeDataSource};

        // Phase 1: read model config (lock-free snapshot), materialize payload
        let state = self.require_state()?;
        let num_layers;
        let engine_ctx;
        {
            let executor = state.backend.executor();
            num_layers = executor.model_config().num_hidden_layers;
            let hidden_size = executor.model_config().hidden_size;
            let kv_page_size = executor.model_config().kv_cache_block_size;
            let num_kv_heads = executor.model_config().num_key_value_heads;
            let max_seq_len = executor.model_config().max_position_embeddings;
            engine_ctx = crate::engine::EngineContext::new(
                num_layers,
                hidden_size,
                kv_page_size,
                num_kv_heads,
                max_seq_len,
            );
        }
        let payload = source.materialize(&engine_ctx)?;
        let actual_layer = target.to_physical_layer(num_layers);
        let data_size_bytes = payload.data.len();

        // Phase 2a: Store payload on executor for KnowledgeInjectCallback (§8.1)
        {
            let mut executor = state.backend.executor_mut();
            executor.set_knowledge_payload(payload.clone());
        }

        // Phase 2b: dispatch by injection kind (all sync, no locks)
        match payload.kind {
            InjectionKind::FrozenKvChunk => {
                let executor = state.backend.executor();
                let backend = executor.cpu_backend().map_err(|e| {
                    ClientError::RuntimeError(format!("cpu_backend: {}", e))
                })?;
                crate::compat::inject_frozen_kv_from_bytes(
                    backend, &payload.data, &payload.shape, actual_layer,
                ).map_err(|e| ClientError::RuntimeError(format!("inject_frozen_kv: {}", e)))?;

                Ok(crate::knowledge::KnowledgeInjectionResult {
                    actual_layer,
                    data_size_bytes,
                })
            }
            InjectionKind::LateFusionVector => {
                let text = std::fs::read_to_string(&source.path).map_err(|e| {
                    ClientError::RuntimeError(format!(
                        "failed to read late fusion source '{}': {}",
                        source.path.display(), e
                    ))
                })?;

                let executor = state.backend.executor();
                let tokens = executor.encode_prompt(&text).map_err(|e| {
                    ClientError::RuntimeError(format!("encode_prompt: {}", e))
                })?;
                let backend = executor.cpu_backend().map_err(|e| {
                    ClientError::RuntimeError(format!("cpu_backend: {}", e))
                })?;
                let weights = executor.weights().map_err(|e| {
                    ClientError::RuntimeError(format!("weights: {}", e))
                })?;
                let config = executor.forward_config().map_err(|e| {
                    ClientError::RuntimeError(format!("forward_config: {}", e))
                })?;

                let embedding = crate::compat::inject_late_fusion(
                    backend, &tokens, weights, &config, target
                ).map_err(|e| ClientError::RuntimeError(format!("inject_late_fusion: {}", e)))?;

                Ok(crate::knowledge::KnowledgeInjectionResult {
                    actual_layer,
                    data_size_bytes: embedding.len() * std::mem::size_of::<f32>(),
                })
            }
            InjectionKind::DynamicLoRA => {
                let loader = crate::loader::safetensors::MappedSafetensors::open(&source.path)
                    .map_err(|e| ClientError::RuntimeError(format!(
                        "failed to open LoRA safetensors '{}': {}",
                        source.path.display(), e
                    )))?;

                let lora_a_tensor = loader.tensor("lora_a.weight")
                    .or_else(|_| loader.tensor("lora_A.weight"))
                    .map_err(|e| ClientError::RuntimeError(format!(
                        "no 'lora_a.weight' tensor in '{}': {}",
                        source.path.display(), e
                    )))?;
                let lora_b_tensor = loader.tensor("lora_b.weight")
                    .or_else(|_| loader.tensor("lora_B.weight"))
                    .map_err(|e| ClientError::RuntimeError(format!(
                        "no 'lora_b.weight' tensor in '{}': {}",
                        source.path.display(), e
                    )))?;

                let lora_a_data = lora_a_tensor.as_f32()
                    .map_err(|e| ClientError::RuntimeError(format!("lora_a not f32: {}", e)))?
                    .into_owned();
                let lora_b_data = lora_b_tensor.as_f32()
                    .map_err(|e| ClientError::RuntimeError(format!("lora_b not f32: {}", e)))?
                    .into_owned();

                let metadata = loader.metadata();
                let layer: usize = metadata
                    .and_then(|m| m.get("layer"))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                let rank: usize = metadata
                    .and_then(|m| m.get("rank"))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| lora_a_data.len().div_ceil(lora_b_data.len().max(1)));
                let alpha: f32 = metadata
                    .and_then(|m| m.get("alpha"))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(rank as f32);
                let target_module = metadata
                    .and_then(|m| m.get("target_module"))
                    .cloned()
                    .unwrap_or_else(|| "q_proj".to_string());

                let mut adapter = crate::compat::LoRAAdapter::new(
                    layer, rank, alpha,
                    lora_a_data.len() / rank.max(1),
                    lora_b_data.len() / rank.max(1),
                    target_module,
                );
                adapter.load_weights(lora_a_data, lora_b_data)
                    .map_err(|e| ClientError::RuntimeError(format!("load_weights: {}", e)))?;

                let executor = state.backend.executor();
                let weights = executor.weights().map_err(|e| {
                    ClientError::RuntimeError(format!("weights: {}", e))
                })?;

                let payload = crate::compat::inject_dynamic_lora(&adapter, weights)
                    .map_err(|e| ClientError::RuntimeError(format!("inject_dynamic_lora: {}", e)))?;

                Ok(crate::knowledge::KnowledgeInjectionResult {
                    actual_layer,
                    data_size_bytes: payload.data.len(),
                })
            }
        }
    }

    // ========================================================================
    // Intent SDK (per SPEC 04-API-DESIGN §7.3, §7.4)
    // ========================================================================

    /// Encode intent (per SPEC 04-API-DESIGN §7.3).
    pub fn encode_intent(
        &self,
        text: &str,
        target: crate::knowledge::LayerTarget,
    ) -> Result<crate::intent::IntentEncoding, ClientError> {
        let state = self.require_state()?;
        let executor = state.backend.executor();
        let tokens = executor.encode_prompt(text)?;
        let num_layers = executor.model_config().num_hidden_layers;
        let actual_layer = layer_target_to_idx(target, num_layers);

        let backend = executor.cpu_backend().map_err(|e| {
            ClientError::RuntimeError(format!("cpu_backend not available: {}", e))
        })?;
        let weights = executor.weights().map_err(|e| {
            ClientError::RuntimeError(format!("weights not available: {}", e))
        })?;
        let config = executor.forward_config().map_err(|e| {
            ClientError::RuntimeError(format!("forward_config not available: {}", e))
        })?;

        let embedding = forward_to_semantic_layer(
            backend, &tokens, weights, &config, target,
        ).map_err(|e| ClientError::RuntimeError(format!("forward_to_semantic_layer failed: {}", e)))?;

        Ok(crate::intent::IntentEncoding {
            embedding,
            actual_layer,
        })
    }

    /// Attach guardrail (per SPEC 04-API-DESIGN §7.4).
    ///
    /// Registers the probe runner into the executor's callback chain
    /// (GuardrailProbeCallback, priority 40) for deep-layer hidden state
    /// classification per SPEC §16.4.
    pub fn attach_guardrail(
        &self,
        probe: crate::intent::GuardProbe,
        target: LayerTarget,
        policy: crate::intent::SafetyPolicy,
    ) -> Result<crate::intent::GuardrailAttachment, ClientError> {
        use crate::guardrail::GuardProbeRunner;

        let state = self.require_state()?;
        let probe_id = match &probe {
            crate::intent::GuardProbe::FromSafetensors { path } => path.clone(),
            crate::intent::GuardProbe::FromModel { model_id } => model_id.clone(),
        };

        let runner = GuardProbeRunner::from_policy(probe, target, policy)
            .map_err(|e| ClientError::RuntimeError(format!("failed to create guard probe runner: {}", e)))?;

        let mut executor = state.backend.executor_mut();
        let num_layers = executor.model_config().num_hidden_layers;
        let actual_layer = target.to_physical_layer(num_layers);

        executor.add_guardrail_runner(runner);

        Ok(crate::intent::GuardrailAttachment {
            actual_layer,
            probe_id,
        })
    }

    /// Attach global guardrail (per SPEC 04-API-DESIGN §9.2).
    pub fn attach_guardrail_global(
        &self,
        policy: crate::intent::SafetyPolicyConfig,
    ) -> Result<(), ClientError> {
        use crate::generation::{GenerationHook, HookDecision};

        let state = self.require_state()?;
        if !policy.global_guardrail_enabled {
            return Ok(());
        }

        let threshold = policy.halt_and_veto_threshold;
        struct GlobalGuardrailHook {
            threshold: f32,
        }

        impl GenerationHook for GlobalGuardrailHook {
            fn post_step(&self, logits: &[f32], _generated_tokens: &[u32]) -> HookDecision {
                let max_logit = logits.iter().copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
                let max_prob = 1.0 / exp_sum;

                if max_prob < self.threshold {
                    HookDecision::Terminate
                } else {
                    HookDecision::Continue
                }
            }
        }

        let hook = Box::new(GlobalGuardrailHook { threshold }) as Box<dyn GenerationHook>;
        let executor = state.backend.executor();
        executor.add_hook(hook).map_err(|e| {
            ClientError::RuntimeError(format!("failed to register global guardrail hook: {}", e))
        })?;

        Ok(())
    }

    /// Expose the internal state handle for streaming support.
    pub(crate) fn state_handle(&self) -> Arc<ArcSwapOption<ClientState>> {
        Arc::clone(&self.state)
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
