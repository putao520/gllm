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
use crate::embeddings::{Embedding, EmbeddingsResponse};
use crate::engine::executor::{BackendError, ExecutorError};
use crate::generation::GenerationResponse;
use crate::knowledge::LayerTarget;
use crate::loader::{Loader, LoaderConfig, LoaderError, WeightFormat};
use crate::manifest::{
    map_architecture_token, ModelArchitecture, ModelKind, ModelManifest, EMPTY_FILE_MAP,
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
    pub backend: BackendContext,
    pub inference_mode: InferenceMode,
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
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self {
            model_id: None,
            kind: None,
            backend: None,
            inference_mode: InferenceMode::Latency,
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

    /// Build the `Client` and load the model synchronously.
    pub fn build(self) -> Result<Client, ClientError> {
        let model_id = self
            .model_id
            .ok_or_else(|| ClientError::ModelNotFound("<no model id>".to_string()))?;
        let kind = self.kind.unwrap_or(ModelKind::Chat);
        let state = Self::build_state(&model_id, kind, self.inference_mode)?;
        Ok(Client {
            state: Arc::new(ArcSwapOption::from_pointee(state)),
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
                if let Some(arch) = map_architecture_token(arch_str) {
                    let dummy_manifest = ModelManifest {
                        model_id: Cow::Owned(model_id.to_string()),
                        file_map: EMPTY_FILE_MAP,
                        arch,
                        kind,
                        rope_base_override: None,
                        max_context_override: None,
                        moe_config: None,
                        tensor_map: HashMap::new(),
                    };
                    let cfg_result =
                        crate::model_config::ModelConfig::from_loader(&dummy_manifest, &mut loader);
                    let moe_config = cfg_result
                        .as_ref()
                        .ok()
                        .and_then(|cfg| cfg.build_moe_config(arch));
                    if let Ok(cfg) = cfg_result {
                        model_config_for_arbiter = Some(cfg);
                    }
                    ModelManifest {
                        model_id: Cow::Owned(model_id.to_string()),
                        file_map: EMPTY_FILE_MAP,
                        arch,
                        kind,
                        rope_base_override: None,
                        max_context_override: None,
                        moe_config,
                        tensor_map: HashMap::new(),
                    }
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

                let dummy_manifest = ModelManifest {
                    model_id: Cow::Owned(model_id.to_string()),
                    file_map: EMPTY_FILE_MAP,
                    arch: ModelArchitecture::Llama4,
                    kind,
                    rope_base_override: None,
                    max_context_override: None,
                    moe_config: None,
                    tensor_map: HashMap::new(),
                };

                let derived_config =
                    crate::model_config::ModelConfig::from_loader(&dummy_manifest, &mut loader)?;

                let arch = loader.detect_architecture();
                let moe_config = derived_config.build_moe_config(arch);
                model_config_for_arbiter = Some(derived_config);

                ModelManifest {
                    model_id: Cow::Owned(model_id.to_string()),
                    file_map: EMPTY_FILE_MAP,
                    arch,
                    kind,
                    rope_base_override: None,
                    max_context_override: None,
                    moe_config,
                    tensor_map: HashMap::new(),
                }
            }
        };

        // Strategy Arbiter: compute StrategyBias from InferenceMode × GraphArchetype × Hardware.
        // Must happen BEFORE BackendContext::new() which triggers JIT compilation.
        if let Some(ref model_cfg) = model_config_for_arbiter {
            let graph_profile = crate::graph::profile::GraphProfiler::profile(model_cfg);
            let archetype = crate::graph::profile::GraphArchetype::derive(&graph_profile);

            // Detect actual backend type to build correct hardware view.
            // GPU backends need ArbiterHwView::gpu() for correct bias adjustments
            // (SPEC §4.3.3: epilogue×1.2, k_depth×1.2, pipeline×1.2 on GPU).
            let hw_view = Self::detect_arbiter_hw_view();

            let arbiter_bias = crate::engine::arbiter::StrategyArbiter::arbitrate(
                inference_mode,
                &archetype,
                &hw_view,
            );

            let kernels_bias = gllm_kernels::compiler::planner::StrategyBias {
                fusion_cost_scale: arbiter_bias.fusion_cost_scale,
                pipeline_cost_scale: arbiter_bias.pipeline_cost_scale,
                parallelism_cost_scale: arbiter_bias.parallelism_cost_scale,
                epilogue_depth_preference: arbiter_bias.epilogue_depth_preference,
                k_depth_preference: arbiter_bias.k_depth_preference,
                kv_cache_budget_scale: arbiter_bias.kv_cache_budget_scale,
                weight_prefetch_budget_scale: arbiter_bias.weight_prefetch_budget_scale,
                batch_flexibility: arbiter_bias.batch_flexibility,
                decode_ratio_scale: arbiter_bias.decode_ratio_scale,
                speculative_decoding_value: arbiter_bias.speculative_decoding_value,
                quantization_aggressiveness: arbiter_bias.quantization_aggressiveness,
                expert_eviction_aggressiveness: arbiter_bias.expert_eviction_aggressiveness,
                expert_prefetch_priority: arbiter_bias.expert_prefetch_priority,
            };

            gllm_kernels::compiler::planner::init_global_execution_plan_with_bias(&kernels_bias);
        }

        let config_path = loader.config_path().map(|p| p.to_path_buf());
        let tokenizer_path = loader.tokenizer_path().map(|p| p.to_path_buf());
        let weight_paths = loader.weight_paths().to_vec();

        let manifest = Arc::new(manifest);

        let detected_backend = detect_backend()?;
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
            backend,
            inference_mode,
        })
    }

    /// Detect hardware view for the Strategy Arbiter.
    ///
    /// Probes GPU availability via backend detection (CUDA→ROCm→Metal→CPU).
    /// GPU backends get `ArbiterHwView::gpu()` with typical shared memory size.
    /// CPU backends get the real DeviceProfile values.
    fn detect_arbiter_hw_view() -> crate::engine::arbiter::ArbiterHwView {
        use crate::engine::arbiter::ArbiterHwView;
        use crate::backend::BackendType;

        // Quick probe: which backend would be selected?
        let backend_type = detect_backend()
            .map(|b| b.backend_type())
            .unwrap_or(BackendType::Cpu);

        match backend_type {
            BackendType::Cuda | BackendType::Rocm | BackendType::Metal => {
                // GPU detected. Use typical shared memory size (49152 bytes = 48KB).
                // Exact value doesn't matter for Arbiter — it only drives
                // L1 richness scaling in §4.3.3 and the is_gpu flag.
                ArbiterHwView::gpu(49152)
            }
            BackendType::Cpu => {
                let profile = gllm_kernels::dispatch::device_profile();
                ArbiterHwView::from(profile)
            }
        }
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

    /// Create a new client with the specified model and kind (sync, blocking).
    pub fn new(model_id: &str, kind: ModelKind) -> Result<Self, ClientError> {
        let state = ClientBuilder::build_state(model_id, kind, InferenceMode::Latency)?;
        Ok(Client {
            state: Arc::new(ArcSwapOption::from_pointee(state)),
        })
    }

    /// Create an empty client (no model loaded).
    pub fn new_empty() -> Self {
        Self {
            state: Arc::new(ArcSwapOption::empty()),
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
            arch: loaded.manifest.arch,
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
    ) -> Result<GenerationResponse, ClientError> {
        let state = self.require_state()?;
        let mut executor = state.backend.executor_mut();
        let result = if let Some(sid) = session_id {
            executor.generate_with_session(&prompt, max_tokens, temperature, top_k, top_p, sid)
        } else {
            executor.generate(&prompt, max_tokens, temperature, top_k, top_p)
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

        // Phase 2: dispatch by injection kind (all sync, no locks)
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
    pub fn attach_guardrail(
        &self,
        probe: crate::intent::GuardProbe,
        target: LayerTarget,
        policy: crate::intent::SafetyPolicy,
    ) -> Result<crate::intent::GuardrailAttachment, ClientError> {
        use crate::guardrail::GuardProbeRunner;

        let state = self.require_state()?;
        let executor = state.backend.executor();
        let num_layers = executor.model_config().num_hidden_layers;
        let probe_id = match &probe {
            crate::intent::GuardProbe::FromSafetensors { path } => path.clone(),
            crate::intent::GuardProbe::FromModel { model_id } => model_id.clone(),
        };
        let actual_layer = target.to_physical_layer(num_layers);

        let runner = GuardProbeRunner::from_policy(probe, target, policy)
            .map_err(|e| ClientError::RuntimeError(format!("failed to create guard probe runner: {}", e)))?;

        executor.add_hook(Box::new(runner)).map_err(|e| {
            ClientError::RuntimeError(format!("failed to register guardrail hook: {}", e))
        })?;

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
    pub arch: ModelArchitecture,
    pub kind: ModelKind,
}
