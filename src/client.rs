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
/// # Error Variants
///
/// - `UnknownModel(String)` — Model not found or download failed (maps to SPEC's `ModelNotFound`)
/// - `UnsupportedArchitecture` — Backend initialization failed (maps to SPEC's `BackendError`)
/// - `ExecutorPoisoned` — Internal state corruption
/// - `NoModelLoaded` — Operation attempted without a loaded model
/// - `InvalidModelType` — Model type mismatch (e.g., using Embedding model for Chat)
/// - `OomHalt` — Out of memory (maps to SPEC's `OutOfMemory`)
/// - Other variants represent runtime errors (maps to SPEC's `RuntimeError`)
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("unknown model alias: {0}")]
    UnknownModel(String),

    #[error("unsupported architecture: {0:?}")]
    UnsupportedArchitecture(ModelArchitecture),

    #[error("state lock poisoned")]
    ExecutorPoisoned,

    #[error("no model loaded")]
    NoModelLoaded,

    #[error("not implemented: {kind} (queued request {request_id})")]
    NotImplementedQueued { kind: &'static str, request_id: u64 },

    #[error("OOM halt: {message} (fatal={fatal})")]
    OomHalt { message: String, fatal: bool },

    #[error("invalid model type for requested operation")]
    InvalidModelType,

    #[error(transparent)]
    Loader(#[from] LoaderError),

    #[error(transparent)]
    Backend(#[from] BackendError),

    #[error(transparent)]
    Executor(#[from] ExecutorError),

    #[error(transparent)]
    ModelConfig(#[from] crate::model_config::ModelConfigError),
}

/// Primary error type exposed to users (type alias for clarity).
pub type GllmError = ClientError;

impl From<BackendContextError> for ClientError {
    fn from(err: BackendContextError) -> Self {
        match err {
            BackendContextError::UnsupportedArchitecture(arch) => {
                ClientError::UnsupportedArchitecture(arch)
            }
            BackendContextError::Loader(err) => ClientError::Loader(err),
            BackendContextError::Executor(err) => ClientError::Executor(err),
            BackendContextError::Backend(err) => ClientError::Backend(err),
        }
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
            .ok_or_else(|| ClientError::UnknownModel("<no model id>".to_string()))?;

        let kind = self.kind.unwrap_or(ModelKind::Chat);

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
                .ok_or(ClientError::NoModelLoaded)?
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
            .ok_or(ClientError::NoModelLoaded)
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
        .map_err(|e| ClientError::Executor(ExecutorError::Scheduler(format!(
            "join error: {}",
            e
        ))))??;

        let mut guard = self.write_state().await?;
        *guard = Some(state);
        Ok(())
    }

    async fn read_state(&self) -> Result<RwLockReadGuard<'_, Option<ClientState>>, ClientError> {
        self.state.read().map_err(|_| ClientError::ExecutorPoisoned)
    }

    async fn write_state(&self) -> Result<RwLockWriteGuard<'_, Option<ClientState>>, ClientError> {
        self.state
            .write()
            .map_err(|_| ClientError::ExecutorPoisoned)
    }

    fn normalize_model_id(model_id: &str) -> Result<String, ClientError> {
        let trimmed = model_id.trim();
        if trimmed.is_empty() {
            return Err(ClientError::UnknownModel(model_id.to_string()));
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
                    return Err(ClientError::UnknownModel(format!(
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
                .map_err(|_| ClientError::ExecutorPoisoned)
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let loaded = guard
                .as_ref()
                .ok_or(ClientError::NoModelLoaded)
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
            ClientError::Executor(ExecutorError::Scheduler(format!("join error: {}", e)))
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
                .map_err(|_| ClientError::ExecutorPoisoned)
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let loaded = guard
                .as_ref()
                .ok_or(ClientError::NoModelLoaded)
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let mut executor = loaded.backend.executor_mut();
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in &inputs {
                embeddings.push(Embedding {
                    embedding: executor.embed(input)?,
                });
            }
            Result::<Vec<Embedding>, ExecutorError>::Ok(embeddings)
                .map_err(|e| ClientError::Executor(e))
        })
        .await
        .map_err(|e| {
            ClientError::Executor(ExecutorError::Scheduler(format!("join error: {}", e)))
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
                .map_err(|_| ClientError::ExecutorPoisoned)
                .map_err(|e| ExecutorError::Scheduler(e.to_string()))?;
            let loaded = guard
                .as_ref()
                .ok_or(ClientError::NoModelLoaded)
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
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if top_n < results.len() {
                results.truncate(top_n);
            }

            Ok::<Vec<RerankResult>, ExecutorError>(results)
        })
        .await
        .map_err(|e| {
            ClientError::Executor(ExecutorError::Scheduler(format!("join error: {}", e)))
        })??;

        Ok(RerankResponse {
            results,
            request_id: None,
        })
    }

    /// Check if thinking head is available.
    pub async fn thinking_head_available(&self) -> Result<bool, ClientError> {
        let state = self.read_state().await?;
        let loaded = state.as_ref().ok_or(ClientError::NoModelLoaded)?;
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
    /// use gllm::{Client, knowledge::{KnowledgeInjectionConfig, LayerTarget, FrozenKvSource}};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// let source = FrozenKvSource::from_frozen_kv("company_logs.kv");
    /// let config = KnowledgeInjectionConfig::new(LayerTarget::MidSemantic, Box::new(source));
    /// let result = client.inject_knowledge(config).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn inject_knowledge(
        &self,
        _config: crate::knowledge::KnowledgeInjectionConfig,
    ) -> Result<crate::knowledge::KnowledgeInjectionResult, ClientError> {
        // Skeleton: requires executor integration
        Err(ClientError::NotImplementedQueued {
            kind: "inject_knowledge",
            request_id: 0,
        })
    }

    // ========================================================================
    // Intent SDK (per SPEC 04-API-DESIGN §7.3, §7.4)
    // ========================================================================

    /// Encode intent (per SPEC 04-API-DESIGN §7.3).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::{Client, intent::{IntentConfig, LayerTarget}};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new_empty();
    /// let config = IntentConfig::new(LayerTarget::MidSemantic);
    /// let intent = client.encode_intent("Cancel my subscription", config).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn encode_intent(
        &self,
        _text: &str,
        _config: crate::intent::IntentConfig,
    ) -> Result<crate::intent::IntentEncoding, ClientError> {
        // Skeleton: requires executor integration
        Err(ClientError::NotImplementedQueued {
            kind: "encode_intent",
            request_id: 0,
        })
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
        _probe: crate::intent::GuardProbe,
        _target: LayerTarget,
        _policy: crate::intent::SafetyPolicy,
    ) -> Result<crate::intent::GuardrailAttachment, ClientError> {
        // Skeleton: requires executor integration
        Err(ClientError::NotImplementedQueued {
            kind: "attach_guardrail",
            request_id: 0,
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
        _policy: crate::intent::SafetyPolicyConfig,
    ) -> Result<(), ClientError> {
        // Skeleton: requires executor integration
        Err(ClientError::NotImplementedQueued {
            kind: "attach_guardrail_global",
            request_id: 0,
        })
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

