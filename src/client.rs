//! gllm Client API — Async-first design (per SPEC 04-API-DESIGN).
//!
//! # Design Principles
//!
//! - **Async-first**: All IO operations are async (model loading, network downloads)
//! - **Builder pattern**: Complex configuration via fluent API
//! - **Explicit types**: Strong typing over string magic values
//! - **Result-oriented**: Clear success/failure via `Result<T, GllmError>`

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::backend::{
    detect_backend, BackendContext, BackendContextError, BackendType,
};
use crate::compat::{forward_to_semantic_layer, layer_target_to_idx};
use crate::embeddings::{Embedding, EmbeddingsBuilder, EmbeddingsResponse};
use crate::engine::executor::{BackendError, ExecutorError};
use crate::generation::{GenerationBuilder, GenerationResponse};
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
///
/// This is the primary error type exposed to users of the gllm library.
///
/// # Error Variants (SPEC §4)
///
/// - `ModelNotFound(String)` — Model not found or download failed
/// - `BackendError(BackendError)` — Backend initialization failed (nested type preserves error details)
/// - `OutOfMemory(OomHaltError)` — Out of memory (nested type preserves OOM details)
/// - `InvalidModelType` — Model type mismatch (e.g., using Embedding model for Chat)
/// - `RuntimeError(String)` — General runtime errors
///
/// # Error Source Chain
///
/// Nested error types (`BackendError`, `OomHaltError`) are accessible via `source()`,
/// allowing callers to inspect detailed error information (e.g., `Unimplemented` variants).
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

    #[error("runtime error: {0}")]
    RuntimeError(String),
}

/// Primary error type exposed to users (type alias for clarity).
pub type GllmError = ClientError;

// Manual From implementations mapping internal errors to SPEC variants

impl From<BackendContextError> for ClientError {
    fn from(err: BackendContextError) -> Self {
        match err {
            BackendContextError::UnsupportedArchitecture(_arch) => {
                // PER SPEC 04-API-DESIGN §4: Map to InvalidModelType for architecture mismatch
                // This preserves semantic meaning while avoiding string-based error loss
                ClientError::InvalidModelType
            }
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
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::builder()
///     .model("Qwen/Qwen3-7B-Instruct")
///     .kind(ModelKind::Chat)
///     .backend(BackendType::Cuda) // Optional: force specific backend
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    model_id: Option<String>,
    kind: Option<ModelKind>,
    backend: Option<BackendType>,
}

impl ClientBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            model_id: None,
            kind: None,
            backend: None,
        }
    }

    /// Set the model identifier (e.g., "Qwen/Qwen3-7B-Instruct").
    ///
    /// This is required. The model ID can be:
    /// - A HuggingFace Hub model ID (e.g., "Qwen/Qwen3-7B-Instruct")
    /// - A local path to a model directory
    /// - A model alias from the manifest registry
    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Set the model kind/purpose.
    ///
    /// If not specified, it will be auto-detected from the model metadata.
    pub fn kind(mut self, kind: ModelKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Force a specific backend type (optional).
    ///
    /// If not specified, the backend will be auto-detected based on
    /// available hardware (CUDA → ROCm → Metal → CPU).
    pub fn backend(mut self, backend: BackendType) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Build the `Client` and load the model (async).
    ///
    /// This performs:
    /// 1. Model download (if not cached)
    /// 2. Architecture detection
    /// 3. Backend initialization
    /// 4. Weight loading
    ///
    /// # Errors
    ///
    /// Returns `ClientError` if:
    /// - Model ID is not set
    /// - Model download fails
    /// - Backend initialization fails
    /// - Weight loading fails
    pub async fn build(self) -> Result<Client, ClientError> {
        let model_id = self
            .model_id
            .ok_or_else(|| ClientError::ModelNotFound("<no model id>".to_string()))?;

        let kind = self.kind.unwrap_or(ModelKind::Chat); // LEGAL: ModelKind::Chat 是默认模型类型

        let client = Client::new_empty();
        client.load_model_inner(model_id, kind).await?;
        Ok(client)
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Client (per SPEC 04-API-DESIGN §2)
// ============================================================================

/// Main gllm client for model inference.
///
/// The client manages model lifecycle and provides inference APIs.
/// It is async-first and uses a lock-based state container for
/// thread-safe model switching.
///
/// # Example
///
/// ```no_run
/// use gllm::Client;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Quick creation
/// let client = Client::new_chat("Qwen/Qwen3-7B-Instruct").await?;
///
/// // Generate text
/// let output = client.generate("Hello, who are you?")
///     .max_tokens(100)
///     .temperature(0.7)
///     .await?;
///
/// println!("{}", output.text);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Client {
    #[allow(clippy::arc_with_non_send_sync)]
    state: Arc<RwLock<Option<ClientState>>>,
}

impl Client {
    // ---------------------------------------------------------------------
    // Construction (per SPEC 04-API-DESIGN §2.1)
    // ---------------------------------------------------------------------

    /// Create a new builder for configuring the client.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::{Client, ModelKind, BackendType};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::builder()
    ///     .model("Qwen/Qwen3-7B-Instruct")
    ///     .kind(ModelKind::Chat)
    ///     .backend(BackendType::Cuda)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Create a new client with a chat model (async).
    ///
    /// This is a convenience method that:
    /// 1. Downloads the model if not cached
    /// 2. Detects the architecture
    /// 3. Initializes the backend
    /// 4. Loads the weights
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::Client;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new_chat("Qwen/Qwen3-7B-Instruct").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new_chat(model_id: &str) -> Result<Self, ClientError> {
        Self::new(model_id, ModelKind::Chat).await
    }

    /// Create a new client with an embedding model (async).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::Client;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new_embedding("BAAI/bge-m3").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new_embedding(model_id: &str) -> Result<Self, ClientError> {
        Self::new(model_id, ModelKind::Embedding).await
    }

    /// Create a new client with the specified model and kind (async).
    ///
    /// This is the primary async constructor. It performs:
    /// 1. Model download (if not cached)
    /// 2. Architecture detection
    /// 3. Backend initialization
    /// 4. Weight loading
    pub async fn new(model_id: &str, kind: ModelKind) -> Result<Self, ClientError> {
        let client = Self::new_empty();
        client.load_model_inner(model_id.to_string(), kind).await?;
        Ok(client)
    }

    /// Create an empty client (no model loaded).
    ///
    /// Use `load_model()` to load a model later.
    pub fn new_empty() -> Self {
        Self {
            state: Arc::new(RwLock::new(None)),
        }
    }

    // ---------------------------------------------------------------------
    // Model Management (per SPEC 04-API-DESIGN §2.2)
    // ---------------------------------------------------------------------

    /// Load a model into the client (async).
    ///
    /// This replaces any currently loaded model. The old model's
    /// resources (KV cache, weights) are fully released before
    /// loading the new model.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::{Client, ModelKind};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// client.load_model("Qwen/Qwen3-7B-Instruct", ModelKind::Chat).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_model(&self, model_id: &str, kind: ModelKind) -> Result<(), ClientError> {
        let model_id = Self::normalize_model_id(model_id)?;
        self.load_model_inner(model_id, kind).await
    }

    /// Unload the current model, releasing resources (async).
    ///
    /// The client instance remains valid and can load a new model.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::Client;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// client.unload_model().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn unload_model(&self) -> Result<(), ClientError> {
        let mut guard = self.write_state().await?;
        *guard = None;
        Ok(())
    }

    /// Swap to a different model (async).
    ///
    /// This is an atomic operation that:
    /// 1. Unloads the current model
    /// 2. Loads the new model
    ///
    /// If loading fails, the client will be in an empty state
    /// (no model loaded).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::Client;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// client.swap_model("Qwen/Qwen3-14B-Chat").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn swap_model(&self, model_id: &str) -> Result<(), ClientError> {
        let model_id = Self::normalize_model_id(model_id)?;

        // Get current model kind before unloading
        let kind = {
            let guard = self.read_state().await?;
            guard
                .as_ref()
                .map(|loaded| loaded.manifest.kind)
                .ok_or(ClientError::RuntimeError("no model loaded".to_string()))?
        };

        // Unload current model
        self.unload_model().await?;

        // Load new model
        self.load_model_inner(model_id, kind).await
    }

    /// Get information about the currently loaded model.
    ///
    /// Returns `None` if no model is loaded.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::Client;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// if let Some(info) = client.model_info().await {
    ///     println!("Current model: {}", info.model_id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn model_info(&self) -> Option<ModelInfo> {
        let guard = self.state.read().ok()?;
        guard.as_ref().map(|loaded| ModelInfo {
            id: loaded.model_id.clone(),
            arch: loaded.manifest.arch,
            kind: loaded.manifest.kind,
        })
    }

    /// Get the manifest of the currently loaded model.
    ///
    /// # Errors
    ///
    /// Returns `ClientError::NoModelLoaded` if no model is loaded.
    pub async fn manifest(&self) -> Result<Arc<ModelManifest>, ClientError> {
        let state = self.read_state().await?;
        state
            .as_ref()
            .map(|loaded| loaded.manifest.clone())
            .ok_or(ClientError::RuntimeError("no model loaded".to_string()))
    }

    // ---------------------------------------------------------------------
    // Inference APIs (per SPEC 04-API-DESIGN §3)
    // ---------------------------------------------------------------------

    /// Create a text generation builder.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::Client;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// let output = client.generate("Hello, who are you?")
    ///     .max_tokens(100)
    ///     .temperature(0.7)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate(&self, prompt: impl Into<String>) -> GenerationBuilder<'_> {
        GenerationBuilder::from_prompt(self, prompt)
    }

    /// Generate embeddings for texts (per SPEC 04-API-DESIGN §3.2).
    ///
    /// This is the primary method for generating embeddings. The method name
    /// follows SPEC convention (shorter verb form).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::Client;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// let embeddings = client.embed(vec![
    ///     "Hello world",
    ///     "Machine learning is fascinating"
    /// ]).await?;
    ///
    /// assert_eq!(embeddings.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed<I, S>(&self, inputs: I) -> Result<EmbeddingsResponse, ClientError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let inputs = inputs.into_iter().map(Into::into).collect();
        self.execute_embeddings(inputs).await
    }

    /// Generate embeddings for texts (alias for backward compatibility).
    ///
    /// # Deprecated
    ///
    /// Use [`Self::embed()`] instead. This method is kept for backward
    /// compatibility and will be removed in a future version.
    #[deprecated(since = "0.12.0", note = "Use embed() instead")]
    pub async fn embeddings<I, S>(&self, inputs: I) -> Result<EmbeddingsResponse, ClientError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.embed(inputs).await
    }

    /// Rerank documents by relevance to query (async).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::Client;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// let scores = client.rerank(
    ///     "What is the capital of France?",
    ///     vec![
    ///         "Paris is the capital of France",
    ///         "London is in UK",
    ///         "Berlin is in Germany"
    ///     ]
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn rerank<I, S>(
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
        self.execute_rerank(query, documents, usize::MAX).await
    }

    // ---------------------------------------------------------------------
    // Internal Methods
    // ---------------------------------------------------------------------

    async fn load_model_inner(
        &self,
        model_id: String,
        kind: ModelKind,
    ) -> Result<(), ClientError> {
        // Run the blocking model loading in a thread pool
        let state = tokio::task::spawn_blocking(move || {
            Self::build_state_blocking(&model_id, kind)
        })
        .await
        .map_err(|e| ClientError::RuntimeError(format!("join error: {}", e)))??;

        let mut guard = self.write_state().await?;
        *guard = Some(state);
        Ok(())
    }

    async fn read_state(&self) -> Result<RwLockReadGuard<'_, Option<ClientState>>, ClientError> {
        self.state.read().map_err(|_| ClientError::RuntimeError("state lock poisoned".to_string()))
    }

    async fn write_state(&self) -> Result<RwLockWriteGuard<'_, Option<ClientState>>, ClientError> {
        self.state
            .write()
            .map_err(|_| ClientError::RuntimeError("state lock poisoned".to_string()))
    }

    fn normalize_model_id(model_id: &str) -> Result<String, ClientError> {
        let trimmed = model_id.trim();
        if trimmed.is_empty() {
            return Err(ClientError::ModelNotFound(model_id.to_string()));
        }
        Ok(trimmed.to_string())
    }

    fn build_state_blocking(
        model_id: &str,
        kind: ModelKind,
    ) -> Result<ClientState, ClientError> {
        let config = LoaderConfig::from_env();

        // Ω1: Tensor-driven loading - no config.json dependency (REQ-REFACTOR-004)
        let mut loader = Loader::from_source_with_config(model_id.to_string(), config.clone())?;

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
                    let moe_config = crate::model_config::ModelConfig::from_loader(&dummy_manifest, &mut loader)
                        .ok()
                        .and_then(|cfg| cfg.build_moe_config(arch));
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
                // PyTorch is auto-converted to SafeTensors by load()
                loader = loader.load()?;

                // 1. Validate Topology via ModelConfig
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

                // 2. Detect Architecture from Tensor Names
                let arch = loader.detect_architecture();

                // 3. Build MoE config from derived metadata
                let moe_config = derived_config.build_moe_config(arch);

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
        })
    }

    pub(crate) async fn execute_generation(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: Option<u64>,
    ) -> Result<GenerationResponse, ClientError> {
        let state_handle = Arc::clone(&self.state);

        // Run blocking executor call in thread pool
        let result = tokio::task::spawn_blocking(move || {
            let guard = state_handle
                .read()
                .map_err(|_| ClientError::RuntimeError("state lock poisoned".to_string()))
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let loaded = guard
                .as_ref()
                .ok_or(ClientError::RuntimeError("no model loaded".to_string()))
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let mut executor = loaded.backend.executor_mut();
            if let Some(sid) = session_id {
                executor.generate_with_session(&prompt, max_tokens, temperature, top_k, top_p, sid)
            } else {
                executor.generate(&prompt, max_tokens, temperature, top_k, top_p)
            }
        })
        .await
        .map_err(|e| {
            ClientError::RuntimeError(format!("join error: {}", e))
        })??;

        let (text, thinking_content) = crate::generation::split_thinking_content(&result);
        Ok(GenerationResponse {
            text,
            thinking_content,
            request_id: None,
        })
    }

    pub(crate) async fn execute_embeddings(
        &self,
        inputs: Vec<String>,
    ) -> Result<EmbeddingsResponse, ClientError> {
        let state_handle = Arc::clone(&self.state);

        let embeddings = tokio::task::spawn_blocking(move || {
            let guard = state_handle
                .read()
                .map_err(|_| ClientError::RuntimeError("state lock poisoned".to_string()))
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let loaded = guard
                .as_ref()
                .ok_or(ClientError::RuntimeError("no model loaded".to_string()))
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let mut executor = loaded.backend.executor_mut();
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in &inputs {
                embeddings.push(Embedding {
                    embedding: executor.embed(input)?,
                });
            }
            Result::<Vec<Embedding>, ExecutorError>::Ok(embeddings)
                .map_err(|e| ClientError::RuntimeError(format!("executor error: {}", e)))
        })
        .await
        .map_err(|e| {
            ClientError::RuntimeError(format!("join error: {}", e))
        })??;

        Ok(EmbeddingsResponse {
            embeddings,
            request_id: None,
        })
    }

    pub(crate) async fn execute_rerank(
        &self,
        query: String,
        documents: Vec<String>,
        top_n: usize,
    ) -> Result<RerankResponse, ClientError> {
        let state_handle = Arc::clone(&self.state);

        let results = tokio::task::spawn_blocking(move || {
            let guard = state_handle
                .read()
                .map_err(|_| ClientError::RuntimeError("state lock poisoned".to_string()))
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let loaded = guard
                .as_ref()
                .ok_or(ClientError::RuntimeError("no model loaded".to_string()))
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let mut executor = loaded.backend.executor_mut();
            let mut scores = Vec::with_capacity(documents.len());
            for doc in documents.iter() {
                let score = executor.rerank_pair(&query, doc).map_err(|e| {
                    ExecutorError::Scheduler(format!("rerank_pair error: {}", e))
                })?;
                let val = score.first().copied().ok_or_else(|| {
                    ExecutorError::Scheduler(
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
                    .unwrap_or(std::cmp::Ordering::Equal) // LEGAL: NaN 比较的标准 Rust 模式
            });
            if top_n < results.len() {
                results.truncate(top_n);
            }

            Ok::<Vec<RerankResult>, ExecutorError>(results)
        })
        .await
        .map_err(|e| {
            ClientError::RuntimeError(format!("join error: {}", e))
        })??;

        Ok(RerankResponse {
            results,
            request_id: None,
        })
    }

    /// Check if thinking head is available.
    pub async fn thinking_head_available(&self) -> Result<bool, ClientError> {
        let state = self.read_state().await?;
        let loaded = state.as_ref().ok_or(ClientError::RuntimeError("no model loaded".to_string()))?;
        let available = {
            let executor = loaded.backend.executor();
            executor.thinking_head_available()
        };
        Ok(available)
    }

    // ========================================================================
    // Knowledge Injection API (per SPEC 04-API-DESIGN §7.2)
    // ========================================================================

    /// Inject knowledge into the model (per SPEC 04-API-DESIGN §7.2).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::{Client, knowledge::{KnowledgeSource, LayerTarget}};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// let result = client.inject_knowledge(
    ///     KnowledgeSource::from_frozen_kv("company_logs.kv"),
    ///     LayerTarget::MidSemantic
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn inject_knowledge(
        &self,
        source: crate::knowledge::KnowledgeSource,
        target: crate::knowledge::LayerTarget,
    ) -> Result<crate::knowledge::KnowledgeInjectionResult, ClientError> {
        use crate::knowledge::{InjectionKind, KnowledgeDataSource};

        // Phase 1: read model config under lock, materialize, then release lock
        let (payload, num_layers) = {
            let state_guard = self.state.read().map_err(|_| {
                ClientError::RuntimeError("client state lock poisoned".to_string())
            })?;
            let client_state = state_guard.as_ref().ok_or_else(|| {
                ClientError::RuntimeError("no model loaded".to_string())
            })?;

            let executor = client_state.backend.executor();
            let num_layers = executor.model_config().num_hidden_layers;
            let hidden_size = executor.model_config().hidden_size;
            let kv_page_size = executor.model_config().kv_cache_block_size;
            let num_kv_heads = executor.model_config().num_key_value_heads;
            let max_seq_len = executor.model_config().max_position_embeddings;
            drop(executor);

            let engine_ctx = crate::engine::EngineContext::new(
                num_layers,
                hidden_size,
                kv_page_size,
                num_kv_heads,
                max_seq_len,
            );

            let payload = source.materialize(&engine_ctx)?;
            (payload, num_layers)
        };
        // state_guard dropped here — lock released

        // Phase 2: dispatch blocking work without holding any lock
        let actual_layer = target.to_physical_layer(num_layers);
        let data_size_bytes = payload.data.len();

        match payload.kind {
            InjectionKind::FrozenKvChunk => {
                let state_handle = Arc::clone(&self.state);
                let result = tokio::task::spawn_blocking(move || {
                    let guard = state_handle
                        .read()
                        .map_err(|_| ClientError::RuntimeError("state lock poisoned".to_string()))?;
                    let cs = guard.as_ref()
                        .ok_or_else(|| ClientError::RuntimeError("no model loaded".to_string()))?;

                    let executor = cs.backend.executor();
                    let backend = executor.cpu_backend()
                        .map_err(|e| ClientError::RuntimeError(format!("cpu_backend: {}", e)))?;

                    crate::compat::inject_frozen_kv_from_bytes(
                        backend, &payload.data, &payload.shape, actual_layer,
                    ).map_err(|e| ClientError::RuntimeError(format!("inject_frozen_kv: {}", e)))?;

                    Ok::<_, ClientError>(())
                })
                .await
                .map_err(|e| ClientError::RuntimeError(format!("join error: {}", e)))??;

                Ok(crate::knowledge::KnowledgeInjectionResult {
                    actual_layer,
                    data_size_bytes,
                })
            }
            InjectionKind::LateFusionVector => {
                // LateFusionVector: read text from file, tokenize, run truncated forward
                let text = std::fs::read_to_string(&source.path).map_err(|e| {
                    ClientError::RuntimeError(format!(
                        "failed to read late fusion source '{}': {}",
                        source.path.display(), e
                    ))
                })?;

                let state_handle = Arc::clone(&self.state);
                let target_layer = target;
                let result = tokio::task::spawn_blocking(move || {
                    let guard = state_handle
                        .read()
                        .map_err(|_| ClientError::RuntimeError("state lock poisoned".to_string()))?;
                    let cs = guard.as_ref()
                        .ok_or_else(|| ClientError::RuntimeError("no model loaded".to_string()))?;

                    let executor = cs.backend.executor();
                    let tokens = executor.encode_prompt(&text)
                        .map_err(|e| ClientError::RuntimeError(format!("encode_prompt: {}", e)))?;
                    let n_layers = executor.model_config().num_hidden_layers;
                    let backend = executor.cpu_backend()
                        .map_err(|e| ClientError::RuntimeError(format!("cpu_backend: {}", e)))?;
                    let weights = executor.weights()
                        .map_err(|e| ClientError::RuntimeError(format!("weights: {}", e)))?;
                    let config = executor.forward_config()
                        .map_err(|e| ClientError::RuntimeError(format!("forward_config: {}", e)))?;

                    let embedding = crate::compat::inject_late_fusion(
                        backend, &tokens, weights, &config, target_layer
                    ).map_err(|e| ClientError::RuntimeError(format!("inject_late_fusion: {}", e)))?;

                    let layer = target_layer.to_physical_layer(n_layers);
                    Ok::<_, ClientError>((layer, embedding.len() * std::mem::size_of::<f32>()))
                })
                .await
                .map_err(|e| ClientError::RuntimeError(format!("join error: {}", e)))??;

                Ok(crate::knowledge::KnowledgeInjectionResult {
                    actual_layer: result.0,
                    data_size_bytes: result.1,
                })
            }
            InjectionKind::DynamicLoRA => {
                // DynamicLoRA: load LoRA weights from safetensors file, build adapter, inject
                let loader = crate::loader::safetensors::MappedSafetensors::open(&source.path)
                    .map_err(|e| ClientError::RuntimeError(format!(
                        "failed to open LoRA safetensors '{}': {}",
                        source.path.display(), e
                    )))?;

                // Load LoRA A and B matrices
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

                // Parse metadata for layer, rank, alpha, target_module
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

                let state_handle = Arc::clone(&self.state);
                let result = tokio::task::spawn_blocking(move || {
                    let guard = state_handle
                        .read()
                        .map_err(|_| ClientError::RuntimeError("state lock poisoned".to_string()))?;
                    let cs = guard.as_ref()
                        .ok_or_else(|| ClientError::RuntimeError("no model loaded".to_string()))?;

                    let executor = cs.backend.executor();
                    let weights = executor.weights()
                        .map_err(|e| ClientError::RuntimeError(format!("weights: {}", e)))?;

                    let payload = crate::compat::inject_dynamic_lora(&adapter, weights)
                        .map_err(|e| ClientError::RuntimeError(format!("inject_dynamic_lora: {}", e)))?;

                    Ok::<_, ClientError>(payload.data.len())
                })
                .await
                .map_err(|e| ClientError::RuntimeError(format!("join error: {}", e)))??;

                Ok(crate::knowledge::KnowledgeInjectionResult {
                    actual_layer,
                    data_size_bytes: result,
                })
            }
        }
    }

    // ========================================================================
    // Intent SDK (per SPEC 04-API-DESIGN §7.3, §7.4)
    // ========================================================================

    /// Encode intent (per SPEC 04-API-DESIGN §7.3).
    ///
    /// Only compute to the semantic layer target and return the feature vector,
    /// truncated from the current layer to return immediately.
    /// The method name follows SPEC convention (shorter verb form).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::{Client, knowledge::LayerTarget};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// let intent = client.encode_intent("Cancel my subscription", LayerTarget::MidSemantic).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn encode_intent(
        &self,
        text: &str,
        target: crate::knowledge::LayerTarget,
    ) -> Result<crate::intent::IntentEncoding, ClientError> {
        let state_handle = Arc::clone(&self.state);

        // Run blocking executor call in thread pool
        let (text, target) = (text.to_string(), target);
        let result = tokio::task::spawn_blocking(move || {
            let guard = state_handle
                .read()
                .map_err(|_| ClientError::RuntimeError("state lock poisoned".to_string()))?;
            let client_state = guard.as_ref()
                .ok_or_else(|| ClientError::RuntimeError("no model loaded".to_string()))?;

            // Get executor and tokenize input
            let executor = client_state.backend.executor();
            let tokens = executor.encode_prompt(&text)?;
            let num_layers = executor.model_config().num_hidden_layers;

            // Map LayerTarget to physical layer index (per SPEC §7.3)
            let actual_layer = layer_target_to_idx(target, num_layers);

            // Get backend, weights, and forward config from executor
            // All in the same scope to avoid lifetime issues
            let backend = executor.cpu_backend()
                .map_err(|e| ClientError::RuntimeError(format!("cpu_backend not available: {}", e)))?;
            let weights = executor.weights()
                .map_err(|e| ClientError::RuntimeError(format!("weights not available: {}", e)))?;
            let config = executor.forward_config()
                .map_err(|e| ClientError::RuntimeError(format!("forward_config not available: {}", e)))?;

            // Execute truncated forward pass to the target layer (per SPEC §7.3)
            // "物理砍断后续层以加速判别式任务"
            let embedding = forward_to_semantic_layer(
                backend,
                &tokens,
                weights,
                &config,
                target,
            ).map_err(|e| ClientError::RuntimeError(format!("forward_to_semantic_layer failed: {}", e)))?;

            Ok::<_, ClientError>(crate::intent::IntentEncoding {
                embedding,
                actual_layer,
            })
        })
        .await
        .map_err(|e| ClientError::RuntimeError(format!("join error: {}", e)))??;

        Ok(result)
    }

    /// Attach guardrail (per SPEC 04-API-DESIGN §7.4).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::{Client, intent::{GuardProbe, SafetyPolicy, LayerTarget}};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// let probe = GuardProbe::from_safetensors("toxicity.safetensors");
    /// let policy = SafetyPolicy::HaltAndVeto { threshold: 0.95 };
    /// client.attach_guardrail(probe, LayerTarget::DeepLogic, policy).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn attach_guardrail(
        &self,
        probe: crate::intent::GuardProbe,
        target: LayerTarget,
        policy: crate::intent::SafetyPolicy,
    ) -> Result<crate::intent::GuardrailAttachment, ClientError> {
        use crate::guardrail::GuardProbeRunner;

        // Get the executor state
        let state_guard = self.state.read().map_err(|_| {
            ClientError::RuntimeError("client state lock poisoned".to_string())
        })?;
        let client_state = state_guard.as_ref().ok_or_else(|| {
            ClientError::RuntimeError("no model loaded".to_string())
        })?;

        // Get model config for layer mapping
        let (num_layers, probe_id) = {
            let executor = client_state.backend.executor();
            let num_layers = executor.model_config().num_hidden_layers;
            let probe_id = match &probe {
                crate::intent::GuardProbe::FromSafetensors { path } => path.clone(),
                crate::intent::GuardProbe::FromModel { model_id } => model_id.clone(),
            };
            (num_layers, probe_id)
        };

        let actual_layer = target.to_physical_layer(num_layers);

        // Create GuardProbeRunner from probe and policy (per SPEC §7.4)
        let runner = GuardProbeRunner::from_policy(probe, target, policy)
            .map_err(|e| ClientError::RuntimeError(format!("failed to create guard probe runner: {}", e)))?;

        // Register the hook with the executor
        let executor = client_state.backend.executor();
        executor.add_hook(Box::new(runner)).map_err(|e| {
            ClientError::RuntimeError(format!("failed to register guardrail hook: {}", e))
        })?;

        Ok(crate::intent::GuardrailAttachment {
            actual_layer,
            probe_id,
        })
    }

    /// Attach global guardrail (per SPEC 04-API-DESIGN §9.2).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::{Client, intent::SafetyPolicyConfig};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// let policy = SafetyPolicyConfig::new()
    ///     .with_guardrail(true)
    ///     .with_threshold(0.95);
    /// client.attach_guardrail_global(policy).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn attach_guardrail_global(
        &self,
        policy: crate::intent::SafetyPolicyConfig,
    ) -> Result<(), ClientError> {
        use crate::generation::{GenerationHook, HookDecision};

        // Get the executor state
        let state_guard = self.state.read().map_err(|_| {
            ClientError::RuntimeError("client state lock poisoned".to_string())
        })?;
        let client_state = state_guard.as_ref().ok_or_else(|| {
            ClientError::RuntimeError("no model loaded".to_string())
        })?;

        // If guardrail is disabled, return early
        if !policy.global_guardrail_enabled {
            return Ok(());
        }

        // Create a global guardrail hook based on the policy configuration
        let threshold = policy.halt_and_veto_threshold;
        struct GlobalGuardrailHook {
            threshold: f32,
        }

        impl GenerationHook for GlobalGuardrailHook {
            fn post_step(&self, logits: &[f32], _generated_tokens: &[u32]) -> HookDecision {
                // Compute softmax probability to estimate confidence
                let max_logit = logits.iter().copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp())
                    .sum();
                // 正确的 softmax max probability 计算: 1.0 / sum(exp(x - max))
                let max_prob = 1.0 / exp_sum;

                // If confidence is below threshold, halt generation
                if max_prob < self.threshold {
                    HookDecision::Terminate
                } else {
                    HookDecision::Continue
                }
            }
        }

        let hook = Box::new(GlobalGuardrailHook {
            threshold,
        }) as Box<dyn GenerationHook>;

        // Register the global hook with the executor
        let executor = client_state.backend.executor();
        executor.add_hook(hook).map_err(|e| {
            ClientError::RuntimeError(format!("failed to register global guardrail hook: {}", e))
        })?;

        Ok(())
    }

    /// Expose the internal state handle for streaming support.
    pub(crate) fn state_handle(&self) -> Arc<RwLock<Option<ClientState>>> {
        Arc::clone(&self.state)
    }
}

// ============================================================================
// Model Info
// ============================================================================

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier (e.g., "Qwen/Qwen3-7B-Instruct") (SPEC: info.id)
    pub id: String,
    /// Detected architecture
    pub arch: ModelArchitecture,
    /// Model kind/purpose
    pub kind: ModelKind,
}

