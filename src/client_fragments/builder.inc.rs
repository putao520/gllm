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
    compute_dtype: Option<gllm_kernels::types::DType>,
    reranker_model_id: Option<String>,
    generator_model_id: Option<String>,
    gguf_file_filter: Option<String>,
    debug_jit: bool,
    /// Enable weight page JIT injection (SPEC/21 §8).
    weight_paging_enabled: bool,
    /// Intent Tracker for signal-aware intent classification (SPEC/INTENT-TRACKER.md).
    intent_tracker: Option<crate::intent_tracker::IntentTracker>,
    /// User intent bias for strategy resolution (REQ-IB-005).
    /// When set, `build_state()` uses `StrategyBiasResolver::resolve()` instead
    /// of `StrategyArbiter::arbitrate()`, merging user preferences with
    /// auto-derived hardware/model baseline.
    intent_bias: crate::engine::intent_bias::IntentBias,
    /// Distributed inference configuration (REQ-IB-012), nccl feature-gated.
    #[cfg(feature = "nccl")]
    distributed: crate::engine::distributed_config::DistributedConfig,
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
            compute_dtype: None,
            reranker_model_id: None,
            generator_model_id: None,
            gguf_file_filter: None,
            debug_jit: false,
            weight_paging_enabled: false,
            intent_tracker: None,
            intent_bias: crate::engine::intent_bias::IntentBias::default(),
            #[cfg(feature = "nccl")]
            distributed: crate::engine::distributed_config::DistributedConfig::default(),
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

    #[deprecated(
        note = "use .intent(IntentBias{ scenario: ScenarioHint::..., ..Default::default() })"
    )]
    pub fn inference_mode(mut self, mode: InferenceMode) -> Self {
        self.inference_mode = mode;
        self
    }

    /// Set the intent bias for strategy resolution (REQ-IB-005).
    ///
    /// When set, `build_state()` uses `StrategyBiasResolver::resolve()` which
    /// merges user-specified intent preferences (scenario hint, overlap hint,
    /// numeric knobs) with the auto-derived hardware/model baseline, instead
    /// of the plain `StrategyArbiter::arbitrate()` path.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gllm::{Client, ModelKind, ScenarioHint, IntentBias};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::builder()
    ///     .model("Qwen/Qwen3-7B-Instruct")
    ///     .intent(IntentBias {
    ///         scenario: ScenarioHint::LatencyCritical,
    ///         ..Default::default()
    ///     })
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn intent(mut self, bias: crate::engine::intent_bias::IntentBias) -> Self {
        self.intent_bias = bias;
        self
    }

    /// Set the distributed inference configuration (REQ-IB-012).
    ///
    /// Only available when the `nccl` feature is enabled. Controls tensor
    /// parallelism, pipeline parallelism, KV distribution, communication
    /// overlap, and MoE expert placement for multi-GPU/multi-node inference.
    #[cfg(feature = "nccl")]
    pub fn distributed(mut self, config: crate::engine::distributed_config::DistributedConfig) -> Self {
        self.distributed = config;
        self
    }

    /// Override compute dtype for the model.
    ///
    /// When set, weights are dequantized from their storage format then converted
    /// to this dtype before passing to JIT. When unset (default), weights pass
    /// through in their original format (zero-copy).
    pub fn compute_dtype(mut self, dtype: gllm_kernels::types::DType) -> Self {
        self.compute_dtype = Some(dtype);
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

    /// Filter GGUF candidate files by name substring (case-insensitive).
    ///
    /// When set, only GGUF files whose filename contains this substring
    /// are considered. Useful for selecting a specific quantization variant.
    pub fn gguf_file_filter(mut self, filter: impl Into<String>) -> Self {
        self.gguf_file_filter = Some(filter.into());
        self
    }

    /// Enable JIT debug instrumentation (INT3 breakpoints + source map).
    ///
    /// When enabled, the JIT compiler inserts DebugBreakpoint VmInstr at key
    /// positions (after embed, before sampling, etc.) and generates a JitSourceMap.
    /// DAP debuggers can attach and set breakpoints on the INT3 instructions.
    /// Default: false (zero overhead).
    pub fn debug_jit(mut self, enabled: bool) -> Self {
        self.debug_jit = enabled;
        if enabled {
            std::env::set_var("GLLM_DEBUG_JIT", "1");
        }
        self
    }

    /// Enable weight page JIT injection for the mega-kernel (SPEC/21 §8).
    ///
    /// When enabled, the JIT compiler injects page fault detection and prefetch
    /// trigger instructions at weight access points in the generated machine
    /// code. Default: `false`.
    pub fn weight_paging_enabled(mut self, enabled: bool) -> Self {
        self.weight_paging_enabled = enabled;
        self
    }

    /// Attach an Intent Tracker for signal-aware intent classification
    /// (SPEC/INTENT-TRACKER.md, REQ-SIT-001~009).
    ///
    /// When set, `Client::generate()` will classify the input prompt's intent
    /// before generation and include the classification result in the response.
    /// When unset (default), generation proceeds without intent tracking.
    pub fn with_intent_tracker(mut self, tracker: crate::intent_tracker::IntentTracker) -> Self {
        self.intent_tracker = Some(tracker);
        self
    }

    /// Build the `Client` and load the model synchronously.
    ///
    /// When both an embedder and reranker are configured with the same
    /// architecture, the encoder backend is shared (Arc clone) to
    /// avoid loading duplicate weights. The reranker uses CLS→Classifier
    /// while the embedder uses MeanPool→L2Norm, but the underlying
    /// encoder forward pass is identical.
    ///
    /// @trace REQ-API-1 [entity:ENT-CLIENT] ClientBuilder::build() — 链式配置终结,返回 Client 实例
    pub fn build(self) -> Result<Client, ClientError> {
        let model_id = self
            .model_id
            .ok_or_else(|| ClientError::ModelNotFound("<no model id>".to_string()))?;
        let kind = self.kind.unwrap_or(ModelKind::Chat);

        // REQ-DIST-002: pass distributed config to build_state (full propagation chain)
        // @trace REQ-DIST-002 [entity:ENT-DISTRIBUTED-CONFIG] [lifecycle:propagate]
        #[cfg(feature = "nccl")]
        let distributed_config = self.distributed.clone();
        #[cfg(not(feature = "nccl"))]
        let distributed_config = ();

        let mut state = Self::build_state(
            &model_id, kind, self.inference_mode, self.compute_dtype, self.gguf_file_filter.as_deref(),
            self.weight_paging_enabled, &self.intent_bias,
            distributed_config,
        )?;

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
            intent_tracker: Arc::new(std::sync::Mutex::new(self.intent_tracker)),
        })
    }

    /// Load model and construct `ClientState` synchronously.
    ///
    /// This is the core model-loading function shared by all constructors.
    /// No async runtime, no locks — pure sync I/O + CPU initialization.
    ///
    /// `weight_paging_enabled`: when `true`, configures the mega-kernel JIT to
    /// inject page fault detection and prefetch instructions at weight access
    /// points (SPEC/21 §8).
    ///
    /// `distributed_config`: when the `nccl` feature is enabled, this is a
    /// `DistributedConfig` that initializes NCCL communicators for multi-GPU
    /// inference (REQ-DIST-001). Without `nccl`, this is `()`.
    #[cfg(feature = "nccl")]
    pub(crate) fn build_state(
        model_id: &str,
        kind: ModelKind,
        inference_mode: InferenceMode,
        compute_dtype: Option<gllm_kernels::types::DType>,
        gguf_file_filter: Option<&str>,
        weight_paging_enabled: bool,
        intent_bias: &crate::engine::intent_bias::IntentBias,
        distributed_config: crate::engine::distributed_config::DistributedConfig,
    ) -> Result<ClientState, ClientError> {
        Self::build_state_inner(
            model_id, kind, inference_mode, compute_dtype, gguf_file_filter,
            weight_paging_enabled, intent_bias, Some(distributed_config),
        )
    }

    #[cfg(not(feature = "nccl"))]
    pub(crate) fn build_state(
        model_id: &str,
        kind: ModelKind,
        inference_mode: InferenceMode,
        compute_dtype: Option<gllm_kernels::types::DType>,
        gguf_file_filter: Option<&str>,
        weight_paging_enabled: bool,
        intent_bias: &crate::engine::intent_bias::IntentBias,
        _distributed_config: (),
    ) -> Result<ClientState, ClientError> {
        Self::build_state_inner(
            model_id, kind, inference_mode, compute_dtype, gguf_file_filter,
            weight_paging_enabled, intent_bias, None,
        )
    }

    /// Core implementation shared by all feature configurations.
    fn build_state_inner(
        model_id: &str,
        kind: ModelKind,
        inference_mode: InferenceMode,
        compute_dtype: Option<gllm_kernels::types::DType>,
        gguf_file_filter: Option<&str>,
        weight_paging_enabled: bool,
        intent_bias: &crate::engine::intent_bias::IntentBias,
        #[cfg(feature = "nccl")] distributed_config: Option<crate::engine::distributed_config::DistributedConfig>,
        #[cfg(not(feature = "nccl"))] _distributed_config: Option<()>,
    ) -> Result<ClientState, ClientError> {
        let mut config = LoaderConfig::from_env();
        if let Some(filter) = gguf_file_filter {
            config.gguf_file_filter = Some(filter.to_string());
        }

        // Ω1: Tensor-driven loading — no config.json dependency
        let mut loader = Loader::from_source_with_config(model_id.to_string(), config.clone())?;

        // ARCH-WEIGHT-ZERO-COPY: inject user-specified compute_dtype into Loader
        if let Some(dt) = compute_dtype {
            loader = loader.with_compute_dtype(dt);
        }

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
            WeightFormat::SafeTensors | WeightFormat::Onnx | WeightFormat::PyTorch | WeightFormat::Gllm => {
                // Ω1: Tensor-driven derivation (REQ-LOADER-022, REQ-LOADER-023)
                // .gllm metadata includes arch_key + all auto_graph params
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
        // REQ-IB-005: 当 IntentBias 非 default 时走 StrategyBiasResolver::resolve()
        // 三阶段管线 (auto-bias → scenario override → clamp),否则走
        // StrategyArbiter::arbitrate() 纯自动推导路径。
        //   InferenceMode (Latency/Throughput baseline) ×
        //   GraphArchetype (fusion_profitable / pipeline_valuable 等模型图特征) ×
        //   ArbiterHwView (cache / SIMD regs / GPU)
        //   × IntentBias (scenario / overlap / numeric knobs)
        // → StrategyBias → compute_execution_plan_with_bias → 每 Client 独立 plan。
        let arbiter_bias = if let Some(cfg) = &model_config_for_arbiter {
            let hw_profile = gllm_kernels::dispatch::device_profile();
            let archetype = {
                let graph_profile = crate::graph::profile::GraphProfiler::profile(cfg);
                crate::engine::arbiter::GraphArchetype::derive(&graph_profile)
            };
            let hw_view = crate::engine::arbiter::ArbiterHwView::from(hw_profile);
            crate::engine::arbiter::StrategyArbiter::arbitrate_with_bias(
                inference_mode,
                &archetype,
                &hw_view,
                intent_bias,
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

        // ── Weight Paging (SPEC/21 §8 §9) ──
        // If enabled, configure the mega-kernel JIT to inject page fault
        // detection and prefetch trigger instructions at weight access points.
        if weight_paging_enabled {
            let mut executor = backend.executor_mut();
            let wp_config = crate::engine::mega_kernel::WeightPageJitConfig {
                enabled: true,
                // Default page geometry — user-tunable in future ClientConfig.
                num_pages: 1024,
                page_size_bytes: 64 * 1024 * 1024, // 64 MiB
                prefetch_distance: 0,
            };
            executor.set_weight_page_jit_config(wp_config);
        }

        // ── Distributed Infrastructure (REQ-DIST-001, REQ-DIST-002) ──
        // Initialize NCCL communicator if a non-default DistributedConfig is
        // provided. Single-node (world_size==1) skips NCCL init but still
        // records the ParallelConfig. Multi-node creates CommHandleWrapper,
        // stores ParallelConfig, and builds PageRoutingTable.
        //
        // REQ-DIST-002: The full DistributedConfig (parallel, kv_distribution,
        // pd_disagg, comm, moe) is passed to init_distributed() which persists
        // all sub-configs on ModelContextHolder with zero information loss.
        // @trace REQ-DIST-001 [entity:ENT-DIST-COMMHANDLE] [lifecycle:init]
        // @trace REQ-DIST-002 [entity:ENT-DISTRIBUTED-CONFIG] [lifecycle:propagate]
        #[cfg(feature = "nccl")]
        {
            if let Some(dist_cfg) = distributed_config {
                let default_cfg = crate::engine::distributed_config::DistributedConfig::default();
                if dist_cfg != default_cfg {
                    let mut executor = backend.executor_mut();
                    executor.init_distributed(dist_cfg)?;
                }
            }
        }

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
        // Pipeline sub-models use default (single-node) distributed config
        #[cfg(feature = "nccl")]
        let dist_cfg = crate::engine::distributed_config::DistributedConfig::default();
        #[cfg(not(feature = "nccl"))]
        let dist_cfg = ();
        let state = Self::build_state(model_id, kind, InferenceMode::Latency, None, None, false, &crate::engine::intent_bias::IntentBias::default(), dist_cfg)?;
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
            WeightFormat::SafeTensors | WeightFormat::Onnx | WeightFormat::PyTorch | WeightFormat::Gllm => {
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

