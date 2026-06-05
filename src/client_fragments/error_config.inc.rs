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
// Client Configuration
// ============================================================================

/// Page compression configuration (SPEC 22-PAGE-COMPRESSION §9).
///
/// Controls the compression codec and storage tier behavior for KV cache
/// pages and weight pages. When `None`, the system uses default codec
/// Public configuration struct for the gllm Client (SPEC 21-WEIGHT-PAGING §9).
///
/// Controls global Client-level options. Passed to `ClientBuilder` via
/// builder methods and consumed during `build_state()`.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct ClientConfig {
    /// Enable weight paging for mega-kernel JIT (SPEC/21 §8).
    ///
    /// When `true`, the JIT compiler injects page fault detection and prefetch
    /// trigger instructions at weight access points. Default: `false`.
    pub weight_paging_enabled: bool,
}


/// Runtime status of the weight paging system (SPEC 21-WEIGHT-PAGING §9).
///
/// Returned by `Client::weight_paging_status()` to report whether weight
/// paging is active and the configured page geometry.
#[derive(Debug, Clone)]
pub struct WeightPagingStatus {
    /// Whether weight paging is enabled for the current model.
    pub enabled: bool,
    /// Number of page table entries (matches WeightPageTable capacity).
    pub num_pages: usize,
    /// Page size in bytes (matches GlobalMemoryManager page size).
    pub page_size_bytes: usize,
    /// Prefetch distance (0 = no prefetch).
    pub prefetch_distance: usize,
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
