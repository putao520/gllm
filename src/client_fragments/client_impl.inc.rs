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

/// Response from MTP-aware generation (REQ-MTP-002).
///
/// Contains the generated text along with detailed MTP statistics
/// including per-step candidate acceptance information.
#[derive(Debug, Clone)]
pub struct MtpGenerationResponse {
    /// Generated text (only verified/committed tokens decoded).
    pub text: String,
    /// Thinking content (for models with thinking heads), if any.
    pub thinking_content: Option<String>,
    /// Total MTP candidates generated across all decode steps.
    pub total_mtp_candidates: usize,
    /// Total MTP candidates accepted (verified correct).
    pub total_mtp_accepted: usize,
    /// Per-step MTP details.
    pub step_details: Vec<MtpStepInfo>,
}

impl MtpGenerationResponse {
    /// Overall MTP acceptance rate (0.0-1.0).
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_mtp_candidates == 0 {
            0.0
        } else {
            self.total_mtp_accepted as f32 / self.total_mtp_candidates as f32
        }
    }

    /// Effective throughput multiplier from MTP.
    ///
    /// Returns how many tokens per decode step were committed on average.
    /// Standard decode = 1.0; MTP with 60% acceptance at depth 2 = 2.2.
    pub fn throughput_multiplier(&self) -> f32 {
        if self.step_details.is_empty() {
            1.0
        } else {
            let total_committed: usize = self.step_details.iter().map(|s| 1 + s.accepted_count).sum();
            total_committed as f32 / self.step_details.len() as f32
        }
    }
}

/// Per-step MTP information in the generation response.
#[derive(Debug, Clone)]
pub struct MtpStepInfo {
    /// Main token for this decode step (always committed).
    pub main_token: u32,
    /// MTP candidate tokens generated for this step.
    pub mtp_candidates: Vec<u32>,
    /// Number of accepted MTP candidates (consecutive from first).
    pub accepted_count: usize,
    /// Whether the main token was the EOS token.
    pub main_token_is_eos: bool,
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
    guardrails: Arc<std::sync::Mutex<HashMap<u64, GuardrailRegistration>>>,
    /// Monotonic id allocator for `attach_guardrail`.
    guardrail_next_id: Arc<std::sync::atomic::AtomicU64>,
    /// Registered Semantic Gatekeeper callback (SPEC/SEMANTIC-GATEKEEPER.md).
    sg_callback: Arc<std::sync::Mutex<Option<Arc<std::sync::Mutex<crate::semantic_gatekeeper::SemanticGatekeeperCallback>>>>>,
    /// Intent Tracker for signal-aware intent classification (SPEC/INTENT-TRACKER.md).
    ///
    /// When set, `generate()` classifies the prompt's intent before generation
    /// and includes the result in the response. When unset, no intent tracking.
    intent_tracker: Arc<std::sync::Mutex<Option<crate::intent_tracker::IntentTracker>>>,
}

/// Internal storage for a registered Guardrail (kept inside the Client).
#[allow(dead_code)]
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

    /// @trace REQ-API-1 [entity:ENT-CLIENT] Client::builder() — Builder 模式入口
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
        let state = ClientBuilder::build_state(model_id, kind, InferenceMode::Latency, None, None, false)?;
        Ok(Client {
            state: Arc::new(ArcSwapOption::from_pointee(state)),
            multimodal_encoder: Arc::new(std::sync::Mutex::new(None)),
            guardrails: Arc::new(std::sync::Mutex::new(HashMap::new())),
            guardrail_next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            sg_callback: Arc::new(std::sync::Mutex::new(None)),
            intent_tracker: Arc::new(std::sync::Mutex::new(None)),
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
            intent_tracker: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    // -----------------------------------------------------------------
    // Model Management
    // -----------------------------------------------------------------

    /// Load a model into the client (sync, blocking).
    ///
    /// Atomically replaces the current model. The old model's resources
    /// are released once all in-flight operations complete.
    ///
    /// @trace REQ-API-1 [entity:ENT-CLIENT] load_model — Builder 构建后加载模型
    pub fn load_model(&self, model_id: &str, kind: ModelKind) -> Result<(), ClientError> {
        let model_id = Self::normalize_model_id(model_id)?;
        let state = ClientBuilder::build_state(&model_id, kind, InferenceMode::Latency, None, None, false)?;
        self.state.store(Some(Arc::new(state)));
        Ok(())
    }

    /// Unload the current model, releasing resources (sync).
    ///
    /// @trace REQ-API-1 [entity:ENT-CLIENT] unload_model — 释放模型资源
    pub fn unload_model(&self) -> Result<(), ClientError> {
        self.state.store(None);
        Ok(())
    }

    /// Swap to a different model (sync, atomic).
    ///
    /// Atomic semantics: if `build_state` fails, the current model remains loaded
    /// (rollback-on-failure). On success, `ArcSwapOption::store` atomically installs
    /// the new state; in-flight reads continue using the old state until they
    /// complete, after which the old model's resources are released automatically.
    ///
    /// @trace REQ-API-1 [entity:ENT-CLIENT] swap_model 原子模型切换
    /// @trace REQ-API-7 [entity:ENT-CLIENT] swap_model 原子操作 + 自动释放旧模型
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
        let state = ClientBuilder::build_state(&model_id, kind, InferenceMode::Latency, None, None, false)?;
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

    /// Encode text to token IDs using the loaded model's tokenizer.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ClientError> {
        let state = self.require_state()?;
        let executor = state.backend.executor();
        executor.encode_prompt(text).map_err(ClientError::from)
    }

    /// Decode token IDs back to text using the loaded model's tokenizer.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, ClientError> {
        let state = self.require_state()?;
        let executor = state.backend.executor();
        executor.decode_tokens(tokens).map_err(ClientError::from)
    }

    // -----------------------------------------------------------------
    // Inference APIs (per SPEC 04-API-DESIGN §3)
    // -----------------------------------------------------------------

    /// Diagnostic: read a row from a weight tensor in the weight blob.
    pub fn diagnostic_weight_row(&self, tensor_name: &str, row: usize, cols: usize) -> Option<Vec<f32>> {
        self.state.load().as_ref()?.backend.executor().diagnostic_weight_row(tensor_name, row, cols)
    }

    /// Diagnostic: return all named weight offsets.
    pub fn diagnostic_weight_offsets(&self) -> Option<Vec<(String, usize)>> {
        self.state.load().as_ref()?.backend.executor().diagnostic_weight_offsets()
    }

    /// Diagnostic: run prefill on prompt tokens and return logits for the last token.
    pub fn diagnostic_prefill_logits(&self, prompt_tokens: &[u32]) -> Option<Vec<f32>> {
        self.state.load().as_ref()?.backend.executor().diagnostic_prefill_logits(prompt_tokens)
    }

    pub fn diagnostic_prefill_scratchpad(&self, prompt_tokens: &[u32]) -> Option<crate::engine::mega_kernel::DiagnosticScratchpad> {
        self.state.load().as_ref()?.backend.executor().diagnostic_prefill_scratchpad(prompt_tokens)
    }

    pub fn diagnostic_forward_only(&self, prompt_tokens: &[u32]) -> Option<Vec<f32>> {
        self.state.load().as_ref()?.backend.executor().diagnostic_forward_only(prompt_tokens)
    }

    /// Create a text generation builder.
    // @trace REQ-API-2 [api:POST /client/generate] Client::generate() entry point — returns GenerationBuilder
    pub fn generate(&self, prompt: impl Into<String>) -> crate::generation::GenerationBuilder<'_> {
        crate::generation::GenerationBuilder::from_prompt(self, prompt)
    }

    /// Generate embeddings for texts (per SPEC 04-API-DESIGN §3.2).
    // @trace REQ-API-3 [api:POST /client/embed] Client::embed() entry point
    pub fn embed<I, S>(&self, inputs: I) -> Result<EmbeddingsResponse, ClientError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let inputs = inputs.into_iter().map(Into::into).collect();
        self.execute_embeddings(inputs, false, None)
    }

    /// Generate embeddings for texts with configuration (per REQ-API-3).
    ///
    /// Supports L2 normalization and dimension truncation via `EmbedConfig`.
    // @trace REQ-API-3 [api:POST /client/embed] Client::embed_with() entry point
    pub fn embed_with<I, S>(&self, inputs: I, config: crate::embeddings::EmbedConfig) -> Result<EmbeddingsResponse, ClientError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let inputs = inputs.into_iter().map(Into::into).collect();
        self.execute_embeddings(inputs, config.normalize, config.dimension)
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
    // @trace REQ-API-4 [api:POST /client/rerank] Client::rerank() — query+documents → RerankResponse
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

    /// Create a rerank builder for document reranking with configuration.
    ///
    /// Supports `top_n()` for Top-K truncation via builder pattern.
    // @trace REQ-API-4 [api:POST /client/rerank] Client::rerank_builder() — returns RerankBuilder with top_n support
    pub fn rerank_builder<I, S>(
        &self,
        query: impl Into<String>,
        documents: I,
    ) -> crate::rerank::RerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let documents = documents.into_iter().map(Into::into).collect();
        crate::rerank::RerankBuilder::new(self, query, documents)
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
                .score_tokens_for_prompt(
                    prompt,
                    &[pos_id, neg_id],
                )
                .map_err(ClientError::from)?;
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
    // Intent Tracker SDK (SPEC/INTENT-TRACKER.md, REQ-SIT-001~009)
    // -----------------------------------------------------------------

    /// Set the Intent Tracker for signal-aware intent classification
    /// (SPEC/INTENT-TRACKER.md, REQ-SIT-007).
    ///
    /// When set, `generate()` will classify the prompt's intent before
    /// generation and include the classification result in the response.
    /// Overwrites any previously registered tracker.
    pub fn set_intent_tracker(
        &self,
        tracker: crate::intent_tracker::IntentTracker,
    ) {
        let mut guard = self
            .intent_tracker
            .lock()
            .expect("intent_tracker mutex poisoned");
        *guard = Some(tracker);
    }

    /// Returns true if an Intent Tracker has been registered.
    pub fn has_intent_tracker(&self) -> bool {
        self.intent_tracker
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    /// Classify the intent of a prompt using the registered Intent Tracker
    /// (SPEC/INTENT-TRACKER.md, REQ-SIT-007).
    ///
    /// Uses `classify_conversation_turn` to embed the text then classify.
    /// Returns the `Classification` result, or `Ok(None)` if no tracker
    /// is registered.
    pub fn classify_intent(
        &self,
        turn_texts: &[&str],
        roles: &[u8],
        signals: &[f32; 11],
    ) -> Result<Option<crate::intent_tracker::Classification>, ClientError> {
        let tracker_arc = {
            let guard = self.intent_tracker.lock().map_err(|_| {
                ClientError::RuntimeError("intent_tracker mutex poisoned".into())
            })?;
            guard.as_ref().cloned()
        };
        let Some(tracker) = tracker_arc else {
            return Ok(None);
        };
        crate::intent_tracker::classify_conversation_turn(self, &tracker, turn_texts, roles, signals)
            .map(Some)
            .map_err(|e| ClientError::RuntimeError(format!("intent classification failed: {e}")))
    }

    // -----------------------------------------------------------------
    // Guardrail SDK (SPEC/GUARDRAIL.md, SPEC/04-API-DESIGN §3.9)
    // -----------------------------------------------------------------

    /// Attach a Guardrail probe at the given anchor layer.
    ///
    /// The returned `GuardrailAttachment` carries an `id` usable for
    /// `Client::detach_guardrail(id)`, along with shared-state accessors
    /// (`last_score()` / `is_vetoed()` / `downgraded_temperature()`).
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
        ClientBuilder::build_state(model_id, kind, InferenceMode::Latency, None, None, false)
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
        // Intent classification (SPEC/INTENT-TRACKER.md, REQ-SIT-007).
        // When an IntentTracker is registered, classify the prompt before
        // generation. The result is attached to GenerationResponse for
        // observability. Classification failure does not block generation.
        let intent_classification = {
            let tracker_opt = self.intent_tracker.lock().ok().and_then(|g| g.as_ref().cloned());
            if let Some(tracker) = tracker_opt {
                let signals = [0.0f32; 11];
                crate::intent_tracker::classify_conversation_turn(
                    self, &tracker, &[&prompt], &[0u8], &signals,
                ).ok()
            } else {
                None
            }
        };

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
                intent_classification,
            })
        })
    }

    /// MTP-aware generation (REQ-MTP-002).
    ///
    /// Generates text using Multi-Token Prediction when available.
    /// Returns both the committed text and detailed MTP statistics
    /// including per-step acceptance rates.
    ///
    /// When MTP is not configured (model has no MTP weights), falls back
    /// to standard generation with zero overhead.
    pub fn generate_with_mtp(
        &self,
        prompt: impl Into<String>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<MtpGenerationResponse, ClientError> {
        let state = self.require_state()?;
        let plan = state.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
            let mut executor = state.backend.executor_mut();
            let mtp_result = executor.generate_with_mtp(
                &prompt.into(), max_tokens, temperature, top_k, top_p,
            )?;

            let text = executor.decode_tokens(&mtp_result.committed_tokens)?;
            let (text, thinking_content) = crate::generation::split_thinking_content(&text);

            Ok(MtpGenerationResponse {
                text,
                thinking_content,
                total_mtp_candidates: mtp_result.total_mtp_candidates,
                total_mtp_accepted: mtp_result.total_mtp_accepted,
                step_details: mtp_result.step_details.into_iter().map(|d| MtpStepInfo {
                    main_token: d.main_token,
                    mtp_candidates: d.mtp_candidates,
                    accepted_count: d.accepted_count,
                    main_token_is_eos: d.main_token_is_eos,
                }).collect(),
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
            intent_classification: None,
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
            intent_classification: None,
        })
    }

    // @trace REQ-API-3 [api:POST /client/embed] execute_embeddings — batch embed with normalize + dimension
    pub(crate) fn execute_embeddings(
        &self,
        inputs: Vec<String>,
        normalize: bool,
        dimension: Option<usize>,
    ) -> Result<EmbeddingsResponse, ClientError> {
        let state = self.require_state()?;
        let plan = state.execution_plan.clone();
        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
            let mut executor = state.backend.executor_mut();
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in &inputs {
                let mut raw = executor.embed(input)?;
                let norm = if normalize {
                    let n = crate::jit::epilogue::compute_l2_norm(&raw);
                    if n > 0.0 {
                        let inv = 1.0 / n;
                        for v in raw.iter_mut() {
                            *v *= inv;
                        }
                    }
                    Some(n)
                } else {
                    None
                };
                if let Some(dim) = dimension {
                    if dim < raw.len() {
                        raw.truncate(dim);
                    }
                }
                embeddings.push(Embedding {
                    embedding: raw,
                    norm,
                });
            }
            Ok(EmbeddingsResponse {
                embeddings,
                rerank_scores: None,
                request_id: None,
            })
        })
    }

    // @trace REQ-API-4 [api:POST /client/rerank] execute_rerank — score, sort descending, truncate top_n
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
    // @trace REQ-API-3 [api:POST /client/embed] execute_embed_rerank_pipeline — embed+rerank with normalize + dimension
    pub(crate) fn execute_embed_rerank_pipeline(
        &self,
        inputs: Vec<String>,
        query: String,
        top_n: Option<usize>,
        normalize: bool,
        dimension: Option<usize>,
    ) -> Result<EmbeddingsResponse, ClientError> {
        let state = self.require_state()?;

        // Step 1: embed all inputs using the primary model
        let embeddings = self.execute_embeddings(inputs.clone(), normalize, dimension)?;

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
            .zip(embeddings.embeddings)
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
            for doc in documents.iter() {
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
            eprintln!("[RERANK-DEBUG] all_scores={:?}", scores);
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
        _normalize: bool,
        _dimension: Option<usize>,
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

    /// Query the runtime status of the weight paging system (SPEC/21 §9).
    ///
    /// Returns `WeightPagingStatus` reflecting the current weight page JIT
    /// injection configuration. When no model is loaded, `enabled` is `false`
    /// and all numeric fields are zero.
    pub fn weight_paging_status(&self) -> WeightPagingStatus {
        let binding = self.state.load();
        let Some(state) = binding.as_ref() else {
            return WeightPagingStatus {
                enabled: false,
                num_pages: 0,
                page_size_bytes: 0,
                prefetch_distance: 0,
            };
        };
        let executor = state.backend.executor();
        let jit_config = executor.weight_page_jit_config();
        WeightPagingStatus {
            enabled: jit_config.enabled,
            num_pages: jit_config.num_pages,
            page_size_bytes: jit_config.page_size_bytes,
            prefetch_distance: jit_config.prefetch_distance,
        }
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
        let kv_dim = if let Some(ref mla) = model_config.mla_config {
            mla.d_c + mla.d_rope
        } else {
            num_kv_heads.checked_mul(head_dim).ok_or_else(|| {
                ClientError::RuntimeError(
                    "semantic gatekeeper: kv_dim overflow".to_string(),
                )
            })?
        };
        let q_dim = num_heads.checked_mul(head_dim).ok_or_else(|| {
            ClientError::RuntimeError(
                "semantic gatekeeper: q_dim overflow".to_string(),
            )
        })?;
        let _dtype = model_config.dtype;
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
        // BCE-20260619-001: Pass model_kind to name_map so Reranker models
        // correctly resolve classifier (not lm_head) if needed.
        let model_kind = self.manifest()?.kind;
        let name_map = {
            let weights = executor.weights().map_err(|e| {
                ClientError::RuntimeError(format!(
                    "semantic gatekeeper: weights accessor failed: {e}"
                ))
            })?;
            weights.name_map_with_kind(Some(model_kind))
        };
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
            let embed_ext = name_map.resolve_external_to_string("embed");
            crate::compat::weight_helpers::get_typed_data::<f32>(weights, backend, &[embed_ext.as_str(), "embed"])
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
            executor.forward_config().map_err(|e| ClientError::RuntimeError(format!("forward_config: {e}")))?.max_seq_len(),
        )
        .map_err(sg_err_to_client)?;
        // ── 5. 每个检测层构造 KProjOnlyGraph ──
        let mut kproj_graphs: Vec<KProjOnlyGraph> = Vec::with_capacity(detection_layers.len());
        for &layer_idx in &detection_layers {
            let ln_ext = name_map.resolve_external_to_string(&format!("L{}.input_norm", layer_idx));
            let ln_cn = format!("L{}.input_norm", layer_idx);
            let k_ext = name_map.resolve_external_to_string(&format!("L{}.k_proj", layer_idx));
            let k_cn = format!("L{}.k_proj", layer_idx);

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
                    weights, backend, &[ln_ext.as_str(), &ln_cn],
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
                    weights, backend, &[k_ext.as_str(), &k_cn],
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
                executor.forward_config().map_err(|e| ClientError::RuntimeError(format!("forward_config: {e}")))?.max_seq_len(),
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

        // ── 7. Reuse pre-created Q-tap ring buffer from Executor ──
        // The ring buffer is pre-created at model load time so the mega-kernel
        // compiles with QTapSTG. We must reuse the same instance because JIT
        // code has its sink_ptr/step_index_ptr baked in.
        let ring_buffer = state.backend.executor().sg_ring_buffer().ok_or_else(|| {
            ClientError::RuntimeError(
                "semantic gatekeeper: executor has no pre-created ring buffer".to_string(),
            )
        })?;
        // Validate geometry consistency (ring buffer was created with the same q_dim).
        if ring_buffer.q_dim() != q_dim {
            return Err(ClientError::RuntimeError(format!(
                "semantic gatekeeper: ring buffer q_dim mismatch: buffer={} expected={}",
                ring_buffer.q_dim(), q_dim,
            )));
        }

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
            let shim = crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim::new(
                sg_arc,
                hidden_size,
            );
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

    // ========================================================================
    // Batch API (SPEC/09 REQ-API-5, SPEC/20 REQ-BCI-008)
    // ========================================================================

    /// Batch generate (SPEC/09 REQ-API-5, SPEC/20 REQ-BCI-008)
    ///
    /// 输入多个 `GenerateRequest`，返回对应的 `GenerateResult` 列表。
    /// M 维度统一架构：batch_size=1 是 batch_size=N 的特例。
    ///
    /// # 注意
    /// 当前版本返回 Unimplemented 错误，完整实现需要 gllm-kernels 侧完成 arg 22 ABI 扩展。
    ///
    /// # 示例
    /// ```no_run
    /// # use gllm::{Client, GenerateRequest};
    /// # let client = Client::new_empty();
    /// let requests = vec![
    ///     GenerateRequest {
    ///         request_id: 1,
    ///         prompt_tokens: vec![1, 2, 3],
    ///         max_new_tokens: 10,
    ///         temperature: 0.8,
    ///         top_k: 0,
    ///         top_p: 0.9,
    ///         session_id: None,
    ///         eos_token_id: 2,
    ///         hook_ctx_ptr: std::ptr::null(),
    ///         callback_table_ptr: std::ptr::null(),
    ///     },
    /// ];
    /// let results = client.generate_batch(&requests).unwrap();
    /// ```
    // @trace REQ-API-5 [entity:Client] [api:POST /client/generate_batch] sync batch generate — M-dimension unified, continuous batching, KV prefix sharing
    pub fn generate_batch(
        &self,
        requests: &[crate::engine::batch_executor::GenerateRequest],
    ) -> Result<Vec<crate::engine::batch_executor::GenerateResult>, ClientError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let state = self.require_state()?;
        let plan = state.execution_plan.clone();

        gllm_kernels::compiler::planner::with_execution_plan(plan, || {
            let mut executor = state.backend.executor_mut();
            executor.generate_batch(requests)
                .map_err(|e| ClientError::RuntimeError(format!("{:?}", e)))
        })
    }

    /// Async version of `generate_batch` (SPEC/09 REQ-API-5, SPEC/20 REQ-BCI-008)
    ///
    /// Offloads the sync batch call to a dedicated thread for non-blocking
    /// async execution. Multiple concurrent calls run in parallel threads,
    /// enabling true concurrent multi-prompt submission.
    // @trace REQ-API-5 [entity:Client] [api:POST /client/generate_batch_async] async batch generate — offloaded to thread, concurrent multi-prompt
    pub async fn generate_batch_async(
        &self,
        requests: &[crate::engine::batch_executor::GenerateRequest],
    ) -> Result<Vec<crate::engine::batch_executor::GenerateResult>, ClientError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let requests = requests.to_vec();
        let this = self.clone();
        std::thread::spawn(move || {
            this.generate_batch(&requests)
        })
        .join()
        .map_err(|_| ClientError::RuntimeError("batch thread panicked".into()))?
    }
}
