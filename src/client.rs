//! Client API skeleton.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::engine::executor::BackendError;
use thiserror::Error;

use crate::backend::{
    detect_backend, BackendContext, BackendContextError,
};
use crate::embeddings::{Embedding, EmbeddingsBuilder, EmbeddingsResponse};
use crate::engine::executor::ExecutorError;
use crate::generation::{GenerationBuilder, GenerationResponse};
use crate::intent::{
    attach_guardrail, encode_intent, GuardrailAttachment, IntentEncoding, SafetyPolicyConfig,
};
use crate::knowledge::{KnowledgeInjectionConfig, KnowledgeError, LayerTarget};
use crate::loader::{Loader, LoaderConfig, LoaderError, WeightFormat};
use crate::manifest::{
    map_architecture_token, ModelArchitecture, ModelKind, ModelManifest, EMPTY_FILE_MAP,
};
use crate::rerank::{RerankBuilder, RerankResponse, RerankResult};

#[derive(Debug, Error)]
pub enum GllmError {
    #[error("unknown model alias: {0}")]
    UnknownModel(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("invalid model type: {0}")]
    InvalidModelType(String),
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
    #[error(transparent)]
    Loader(#[from] LoaderError),

    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    #[error(transparent)]
    ModelConfig(#[from] crate::model_config::ModelConfigError),
}

/// Unified error type for public API (per SPEC 04-API-DESIGN §4).
///
/// This is the primary error type exposed to users of the gllm library.
///
/// # Error Variants
///
/// - `ModelNotFound(String)` — Model not found or download failed (maps to SPEC's `ModelNotFound`)
/// - `UnknownModel(String)` — Alias for ModelNotFound (backward compatibility)
/// - `InvalidModelType(String)` — Model type mismatch (e.g., using Embedding model for Chat)
/// - `UnsupportedArchitecture` — Backend initialization failed (maps to SPEC's `BackendError`)
/// - `OomHalt` — Out of memory (maps to SPEC's `OutOfMemory`)
/// - Other variants represent runtime errors (maps to SPEC's `RuntimeError`)

pub struct ClientState {
    pub model_id: String,
    pub manifest: Arc<ModelManifest>,
    pub backend: BackendContext,
}

#[derive(Clone)]
pub struct Client {
    #[allow(clippy::arc_with_non_send_sync)]
    state: Arc<RwLock<Option<ClientState>>>,
}

pub struct AsyncClient {
    inner: Client,
}

impl From<BackendContextError> for GllmError {
    fn from(err: BackendContextError) -> Self {
        match err {
            BackendContextError::UnsupportedArchitecture(arch) => {
                GllmError::UnsupportedArchitecture(arch)
            }
            BackendContextError::Loader(err) => GllmError::Loader(err),
            BackendContextError::Executor(err) => GllmError::Executor(err),
            BackendContextError::Backend(err) => GllmError::Backend(err),
        }
    }
}

impl Client {
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(model_id: &str, kind: ModelKind) -> Result<Self, GllmError> {
        let client = Self {
            state: Arc::new(RwLock::new(None)),
        };
        client.load_model(model_id, kind)?;
        Ok(client)
    }

    pub fn new_chat(model_id: &str) -> Result<Self, GllmError> {
        Self::new(model_id, ModelKind::Chat)
    }
    pub fn new_embedding(model_id: &str) -> Result<Self, GllmError> {
        Self::new(model_id, ModelKind::Embedding)
    }

    pub fn manifest(&self) -> Result<Arc<ModelManifest>, GllmError> {
        let state = self.read_state()?;
        state
            .as_ref()
            .map(|loaded| loaded.manifest.clone())
            .ok_or(GllmError::NoModelLoaded)
    }

    pub fn load_model(&self, model_id: &str, kind: ModelKind) -> Result<(), GllmError> {
        let model_id = Self::normalize_model_id(model_id)?;
        let mut guard = self.write_state()?;
        let state = Self::build_state(&model_id, kind)?;
        *guard = Some(state);
        Ok(())
    }

    pub fn unload_model(&self) -> Result<(), GllmError> {
        let mut guard = self.write_state()?;
        *guard = None;
        Ok(())
    }

    pub fn swap_model(&self, model_id: &str) -> Result<(), GllmError> {
        let model_id = Self::normalize_model_id(model_id)?;
        let mut guard = self.write_state()?;
        let kind = guard
            .as_ref()
            .map(|loaded| loaded.manifest.kind)
            .ok_or(GllmError::NoModelLoaded)?;

        *guard = None;
        let state = Self::build_state(&model_id, kind)?;
        *guard = Some(state);
        Ok(())
    }

    pub fn generate(&self, prompt: impl Into<String>) -> GenerationBuilder<'_> {
        GenerationBuilder::from_prompt(self, prompt)
    }

    pub fn embeddings<I, S>(&self, inputs: I) -> EmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let inputs = inputs.into_iter().map(Into::into).collect();
        EmbeddingsBuilder::new(self, inputs)
    }

    pub fn rerank<I, S>(&self, query: impl Into<String>, documents: I) -> RerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let documents = documents.into_iter().map(Into::into).collect();
        RerankBuilder::new(self, query, documents)
    }

    pub fn thinking_head_available(&self) -> Result<bool, GllmError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(GllmError::NoModelLoaded)?;
        let available = {
            let executor = loaded.backend.executor();
            executor.thinking_head_available()
        };
        Ok(available)
    }

    /// Injects knowledge into the model at a specified semantic layer.
    ///
    /// This method implements the Knowledge Injection API (SPEC 04-API-DESIGN.md §7).
    /// It allows external knowledge to be injected into the model's residual stream
    /// at a semantic anchor point (ShallowSyntax, MidSemantic, or DeepLogic).
    ///
    /// # Arguments
    ///
    /// * `config` - Knowledge injection configuration containing the source and target
    ///
    /// # Returns
    ///
    /// * `Result<(), GllmError>` - Success or error
    ///
    /// # Example
    ///
    /// ```ignore
    /// use gllm::{Client, KnowledgeSource, LayerTarget, KnowledgeInjectionConfig};
    ///
    /// let client = Client::new_chat("Qwen/Qwen3-7B-Instruct")?;
    ///
    /// let config = KnowledgeInjectionConfig::new(
    ///     KnowledgeSource::from_text("Company policy document..."),
    ///     LayerTarget::MidSemantic,
    /// );
    ///
    /// client.inject_knowledge(config)?;
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// This is a skeleton implementation. The full implementation would:
    /// 1. Materialize the knowledge source into an engine-ready payload
    /// 2. Determine the physical layer number from the semantic anchor
    /// 3. Register the injection with the engine's injection scheduler
    /// 4. Update the Mega-Kernel launch parameters for affected requests
    pub fn inject_knowledge(&self, config: KnowledgeInjectionConfig) -> Result<(), GllmError> {
        // Verify a model is loaded
        let _state = self.read_state()?;
        let _loaded = _state.as_ref().ok_or(GllmError::NoModelLoaded)?;

        // Materialize the knowledge source
        let _payload = config.materialize().map_err(|e| match e {
            KnowledgeError::EngineNotReady(msg) => GllmError::Executor(ExecutorError::Other(msg)),
            KnowledgeError::MaterializationFailed(msg) => {
                GllmError::Executor(ExecutorError::Other(msg))
            }
            KnowledgeError::InvalidSource(msg) => {
                GllmError::Executor(ExecutorError::Other(format!("Invalid knowledge source: {}", msg)))
            }
            KnowledgeError::VectorDb(msg) => {
                GllmError::Executor(ExecutorError::Other(format!("VectorDB error: {}", msg)))
            }
            KnowledgeError::File(e) => GllmError::Executor(ExecutorError::Other(format!(
                "File error: {}",
                e
            ))),
        })?;

        // Skeleton: In a full implementation, this would:
        // 1. Register the payload with the engine's KvSideloadManager or InjectionScheduler
        // 2. Update the injection routing table for batched requests
        // 3. Trigger JIT recompilation if necessary for the new injection pattern

        Ok(())
    }

    /// Encodes an intent by extracting features at a specified semantic layer.
    ///
    /// This method implements the Multi-Intent Dimensionality Reduction API
    /// (SPEC 04-API-DESIGN.md §7.3). It physically truncates computation at the
    /// specified layer to accelerate discriminative tasks.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    /// * `layer_target` - Semantic anchor point for truncation
    ///
    /// # Returns
    ///
    /// * `Result<IntentEncoding, GllmError>` - Feature vector and metadata
    ///
    /// # Example
    ///
    /// ```ignore
    /// use gllm::{Client, LayerTarget};
    ///
    /// let client = Client::new_chat("Qwen/Qwen3-7B-Instruct")?;
    ///
    /// // Extract features at the mid-semantic layer for intent classification
    /// let intent = client.encode_intent("Cancel my subscription", LayerTarget::MidSemantic)?;
    ///
    /// // The embedding can be used directly with an external lightweight classifier
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// This is a skeleton implementation. The full implementation would:
    /// 1. Map the semantic anchor to a physical layer number based on model topology
    /// 2. Execute forward propagation only up to the target layer
    /// 3. Extract and return the hidden state at that layer
    pub fn encode_intent(&self, text: &str, layer_target: LayerTarget) -> Result<IntentEncoding, GllmError> {
        // Verify a model is loaded
        let _state = self.read_state()?;
        let _loaded = _state.as_ref().ok_or(GllmError::NoModelLoaded)?;

        // Delegate to the intent module
        encode_intent(text, layer_target)
    }

    /// Attaches a safety guardrail probe at a specified semantic layer.
    ///
    /// This method implements the In-Flight Guardrail API
    /// (SPEC 04-API-DESIGN.md §7.4). It mounts a lightweight classifier
    /// in the model's forward pass for zero-latency safety intervention.
    ///
    /// # Arguments
    ///
    /// * `probe_path` - Path to the guard probe weights (safetensors format)
    /// * `layer_target` - Semantic anchor point for probe mounting
    /// * `policy` - Safety policy configuration
    ///
    /// # Returns
    ///
    /// * `Result<GuardrailAttachment, GllmError>` - Probe attachment info
    ///
    /// # Example
    ///
    /// ```ignore
    /// use gllm::{Client, LayerTarget, SafetyPolicyConfig};
    ///
    /// let client = Client::new_chat("Qwen/Qwen3-7B-Instruct")?;
    ///
    /// // Mount a toxicity classifier at the deep logic layer
    /// let attachment = client.attach_guardrail(
    ///     "toxicity_classifier_v1.safetensors",
    ///     LayerTarget::DeepLogic,
    ///     SafetyPolicyConfig::halt_and_veto(0.95),
    /// )?;
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// This is a skeleton implementation. The full implementation would:
    /// 1. Load the probe weights from the specified file
    /// 2. Register the probe with the executor's guardrail registry
    /// 3. Compile the probe integration into the Mega-Kernel launch parameters
    /// 4. Enable hardware-level intervention when the threshold is exceeded
    pub fn attach_guardrail(
        &self,
        probe_path: &str,
        layer_target: LayerTarget,
        policy: SafetyPolicyConfig,
    ) -> Result<GuardrailAttachment, GllmError> {
        // Verify a model is loaded
        let _state = self.read_state()?;
        let _loaded = _state.as_ref().ok_or(GllmError::NoModelLoaded)?;

        // Delegate to the intent module
        attach_guardrail(probe_path, layer_target, policy)
    }

    pub(crate) fn execute_generation(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: Option<u64>,
    ) -> Result<GenerationResponse, GllmError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(GllmError::NoModelLoaded)?;
        // ARCH-ZERO-FALLBACK: Direct executor call, no fallback wrapper.
        // OOM errors propagate as-is via ExecutorError -> GllmError.
        let mut executor = loaded.backend.executor_mut();
        let result = if let Some(sid) = session_id {
            executor.generate_with_session(&prompt, max_tokens, temperature, top_k, top_p, sid)?
        } else {
            executor.generate(&prompt, max_tokens, temperature, top_k, top_p)?
        };
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
    ) -> Result<EmbeddingsResponse, GllmError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(GllmError::NoModelLoaded)?;
        // ARCH-ZERO-FALLBACK: Direct executor call, no fallback wrapper.
        let mut executor = loaded.backend.executor_mut();
        let mut embeddings = Vec::with_capacity(inputs.len());
        for input in &inputs {
            embeddings.push(Embedding { embedding: executor.embed(input)? });
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
    ) -> Result<RerankResponse, GllmError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(GllmError::NoModelLoaded)?;
        // ARCH-ZERO-FALLBACK: Direct executor call, no fallback wrapper.
        let mut executor = loaded.backend.executor_mut();
        let mut scores = Vec::with_capacity(documents.len());
        for doc in documents.iter() {
            let score = executor.rerank_pair(&query, doc)?;
            let val = score.first().copied().ok_or_else(|| {
                GllmError::Backend(BackendError::Cpu(
                    "rerank_pair returned empty scores for query/doc pair".into(),
                ))
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

    fn normalize_model_id(model_id: &str) -> Result<String, GllmError> {
        let trimmed = model_id.trim();
        if trimmed.is_empty() {
            return Err(GllmError::UnknownModel(model_id.to_string()));
        }
        Ok(trimmed.to_string())
    }

    fn build_state(model_id: &str, kind: ModelKind) -> Result<ClientState, GllmError> {
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
                    return Err(GllmError::UnknownModel(format!(
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

    fn read_state(&self) -> Result<RwLockReadGuard<'_, Option<ClientState>>, GllmError> {
        self.state.read().map_err(|_| GllmError::ExecutorPoisoned)
    }

    fn write_state(&self) -> Result<RwLockWriteGuard<'_, Option<ClientState>>, GllmError> {
        self.state
            .write()
            .map_err(|_| GllmError::ExecutorPoisoned)
    }
}

impl AsyncClient {
    pub fn new(model_id: &str, kind: ModelKind) -> Result<Self, GllmError> {
        Ok(Self {
            inner: Client::new(model_id, kind)?,
        })
    }

    pub fn inner(&self) -> &Client {
        &self.inner
    }
    pub fn generate(&self, prompt: impl Into<String>) -> GenerationBuilder<'_> {
        self.inner.generate(prompt)
    }

    pub fn embeddings<I, S>(&self, inputs: I) -> EmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.embeddings(inputs)
    }

    pub fn rerank<I, S>(&self, query: impl Into<String>, documents: I) -> RerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.rerank(query, documents)
    }
}
