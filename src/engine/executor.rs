//! Executor skeleton.

use std::fmt;

use log;

use crate::compat::backend_trait::{Backend, Element};
use crate::compat::CpuBackend;
use crate::scheduler::types::{PageId, RequestId, StorageKey};
use gllm_kernels::compiler::graph::SYMDIM_MAX_SEQ_LEN;
use gllm_kernels::types::DType;
use thiserror::Error;

/// Cap the geometry-declared `max_position_embeddings` by the JIT compile-time
/// ceiling (`SYMDIM_MAX_SEQ_LEN`) whenever it is used to size / index the
/// runtime KV cache.
///
/// Rationale: modern LLMs (Gemma 4 E2B = 131 072, DeepSeek = 163 840, Kimi =
/// 200 000+) ship with enormous theoretical context windows that would
/// allocate hundreds of gigabytes of CPU KV cache at load time (e.g. Gemma 4
/// E2B: 15 effective layers × 1 KV head × 131 072 × 256 head_dim × 4 B × 2 =
/// 3.84 GB just for the zero-memset, on top of the weight loader!).
///
/// The JIT-emitted attention kernel already bakes `SYMDIM_MAX_SEQ_LEN` as its
/// compile-time upper bound (`emit_multi_head_attention` computes the `V_ptr`
/// offset as `compile_seq_len * kv_dim * 4` where `compile_seq_len ==
/// SYMDIM_MAX_SEQ_LEN`), so the cache stride MUST match this constant. Any
/// larger allocation is wasted memory and any smaller stride produces
/// out-of-bounds reads inside the JIT kernel. Exposing a single SSOT
/// (`effective_kv_max_seq_len`) keeps the cache size, the cache-offset math
/// in `merge_kv_cache_for_decode` / `perform_kv_cache_write`, and the JIT
/// `compile_seq_len` in lock-step.
#[inline]
pub fn effective_kv_max_seq_len(geometry_max_seq_len: usize) -> usize {
    geometry_max_seq_len.min(SYMDIM_MAX_SEQ_LEN)
}

// ---- Engine types (moved from compat) ----

/// Positional encoding variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEncoding {
    None,
    Rope,
}

/// Sampling hyper-parameters for a single request.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// RoPE (Rotary Position Embedding) configuration.
#[derive(Debug, Clone, Copy)]
pub struct RoPEConfig {
    pub theta: f64,
    pub scale: f64,
    pub interleaved: bool,
    pub precompute: bool,
}

/// Attention head geometry.
#[derive(Debug, Clone, Copy)]
pub struct AttentionHeadConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionHeadConfig {
    /// Construct from ModelGeometry (single source of truth).
    pub fn from_geometry(g: &crate::model_config::ModelGeometry) -> Self {
        Self {
            num_heads: g.num_heads,
            num_kv_heads: g.num_kv_heads,
            head_dim: g.head_dim,
        }
    }
}

/// Paged KV cache configuration for GPU decode.
#[derive(Debug, Clone)]
pub struct PagedKvConfig {
    pub page_table: Option<Vec<u32>>,
    pub page_size: usize,
}

/// Static configuration for the generator forward pass.
#[derive(Debug, Clone)]
pub struct GeneratorForwardConfig {
    /// Model geometric constants (Arc-shared single source of truth).
    pub geometry: Arc<crate::model_config::ModelGeometry>,
    /// RoPE config (convenience view, derived from geometry).
    pub rope: RoPEConfig,
    pub position_encoding: PositionEncoding,
    /// Architecture family (Encoder vs Decoder).
    pub arch_family: crate::manifest::ArchFamily,
    /// Token ID for "yes" (used by decoder-based rerankers without a score head).
    pub rerank_yes_token_id: Option<u32>,
    /// Token ID for "no" (used by decoder-based rerankers without a score head).
    pub rerank_no_token_id: Option<u32>,
    /// MoE configuration (None for dense models).
    pub moe_config: Option<crate::manifest::MoEConfig>,
    /// Paged KV cache configuration.
    pub paged_kv: PagedKvConfig,
    /// YAML→JIT graph executor pointer.
    /// Points into the Executor's `graph_executor` Option; null when not available.
    pub graph_executor_ptr: *mut crate::graph::executor::FusedGraphExecutor,
    /// §9-§18: Callback chain pointer for Gate-First Skip / Residual Bypass / Early Exit.
    /// Points into the Executor's callback chain; null when no callbacks active.
    pub callback_chain_ptr: *mut crate::graph::layer_callback::CallbackChain,
}

impl GeneratorForwardConfig {
    /// Backward-compatible accessor: hidden size.
    pub fn hidden_size(&self) -> usize { self.geometry.hidden_size }
    /// Backward-compatible accessor: number of layers.
    pub fn num_layers(&self) -> usize { self.geometry.num_layers }
    /// Backward-compatible accessor: vocabulary size.
    pub fn vocab_size(&self) -> usize { self.geometry.vocab_size }
    /// Backward-compatible accessor: FFN intermediate dimension.
    pub fn intermediate_size(&self) -> usize { self.geometry.intermediate_size }
    /// Backward-compatible accessor: LayerNorm epsilon.
    pub fn norm_eps(&self) -> f32 { self.geometry.norm_eps }
    /// Backward-compatible accessor: model weight dtype.
    pub fn dtype(&self) -> DType { self.geometry.dtype }
    /// Backward-compatible accessor: maximum sequence length, capped by the
    /// JIT compile-time ceiling (`SYMDIM_MAX_SEQ_LEN`). See
    /// `effective_kv_max_seq_len` for why the cap is mandatory.
    pub fn max_seq_len(&self) -> usize {
        effective_kv_max_seq_len(self.geometry.max_seq_len)
    }
    /// Backward-compatible accessor: number of attention heads.
    pub fn num_heads(&self) -> usize { self.geometry.num_heads }
    /// Backward-compatible accessor: number of KV heads.
    pub fn num_kv_heads(&self) -> usize { self.geometry.num_kv_heads }
    /// Backward-compatible accessor: head dimension.
    pub fn head_dim(&self) -> usize { self.geometry.head_dim }
    /// Derive AttentionHeadConfig from geometry.
    pub fn attention(&self) -> AttentionHeadConfig {
        AttentionHeadConfig::from_geometry(&self.geometry)
    }
    /// Backward-compatible accessor: RoPE theta.
    pub fn rope_theta(&self) -> f64 { self.rope.theta }
    /// Backward-compatible accessor: RoPE scale.
    pub fn rope_scale(&self) -> f64 { self.rope.scale }
}

// SAFETY: graph_executor_ptr is only written/read while the Executor's Mutex is held,
// and the Executor itself is not Send across threads without synchronization.
unsafe impl Send for GeneratorForwardConfig {}
unsafe impl Sync for GeneratorForwardConfig {}

impl GeneratorForwardConfig {
    /// Extract attention head geometry as a grouped struct.
    #[allow(dead_code)]
    pub(crate) fn attention_geometry(&self) -> crate::compat::types::AttentionGeometry {
        let q_dim = self.geometry.num_heads * self.geometry.head_dim;
        let kv_dim = self.geometry.num_kv_heads * self.geometry.head_dim;
        crate::compat::types::AttentionGeometry {
            num_heads: self.geometry.num_heads,
            num_kv_heads: self.geometry.num_kv_heads,
            head_dim: self.geometry.head_dim,
            q_dim,
            kv_dim,
            heads_per_group: self.geometry.num_heads / self.geometry.num_kv_heads.max(1),
        }
    }

    /// Extract per-layer dimension constants.
    #[allow(dead_code)]
    pub(crate) fn layer_dims(&self) -> crate::compat::types::LayerDims {
        crate::compat::types::LayerDims {
            hidden: self.geometry.hidden_size,
            inter: self.geometry.intermediate_size,
            eps: self.geometry.norm_eps,
            rope_theta: self.rope.theta,
        }
    }
}

/// KV-cache swap configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SwapConfig {
    pub enable_swap: bool,
    pub swap_threshold: f32,
    pub lru_granularity: usize,
}

/// KV-cache geometry.
#[derive(Debug, Clone)]
#[allow(clippy::derived_hash_with_manual_partial_eq)]
pub struct KvCacheConfig {
    /// Model geometry (provides num_layers, num_kv_heads, head_dim, max_seq_len).
    pub geometry: Arc<crate::model_config::ModelGeometry>,
    /// KV cache element dtype (may differ from geometry.dtype, e.g. CPU forces F32).
    pub kv_dtype: DType,
    pub page_size: usize,
    pub swap_config: Option<SwapConfig>,
}

impl KvCacheConfig {
    /// Bytes per KV cache element.
    pub fn dtype_size(&self) -> usize { self.kv_dtype.size_bytes() }
    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize { self.geometry.num_layers }
    /// Number of KV attention heads.
    pub fn num_heads(&self) -> usize { self.geometry.num_kv_heads }
    /// Dimension of each attention head.
    pub fn head_dim(&self) -> usize { self.geometry.head_dim }
    /// Maximum sequence length, capped by the JIT compile-time ceiling
    /// (`SYMDIM_MAX_SEQ_LEN`). This is the stride used by the KV cache buffer
    /// allocation and by `merge_kv_cache_for_decode` / `perform_kv_cache_write`
    /// offset math — it MUST equal the `compile_seq_len` baked into the JIT
    /// attention kernel. Models with `max_position_embeddings` larger than
    /// `SYMDIM_MAX_SEQ_LEN` (e.g. Gemma 4 E2B's 131 072) would otherwise
    /// allocate multi-GB of zeroed cache that the JIT kernel can't index into.
    pub fn max_seq_len(&self) -> usize {
        effective_kv_max_seq_len(self.geometry.max_seq_len)
    }
    /// SharedKvRef: 后 N 层共享 KV cache.
    pub fn num_kv_shared_layers(&self) -> usize { self.geometry.num_kv_shared_layers }
    /// Per-layer attention pattern (0=sliding, 1=global).
    pub fn attention_pattern(&self) -> &[u8] { &self.geometry.attention_pattern }
}

impl PartialEq for KvCacheConfig {
    fn eq(&self, other: &Self) -> bool {
        self.num_layers() == other.num_layers()
            && self.num_heads() == other.num_heads()
            && self.head_dim() == other.head_dim()
            && self.max_seq_len() == other.max_seq_len()
            && self.kv_dtype == other.kv_dtype
            && self.page_size == other.page_size
            && self.swap_config == other.swap_config
    }
}

/// Errors originating from a compute backend.
#[derive(Debug, Clone)]
pub enum BackendError {
    Cuda(String),
    Hip(String),
    Metal(String),
    Cpu(String),
    Unimplemented(&'static str),
    Other(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::Cuda(msg) => write!(f, "CUDA error: {msg}"),
            BackendError::Hip(msg) => write!(f, "HIP error: {msg}"),
            BackendError::Metal(msg) => write!(f, "Metal error: {msg}"),
            BackendError::Cpu(msg) => write!(f, "CPU error: {msg}"),
            BackendError::Unimplemented(what) => write!(f, "unimplemented: {what}"),
            BackendError::Other(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Opaque handle to an allocated KV-cache buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvCacheHandle(pub u64);

/// Opaque handle to a logits tensor returned by forward.
#[derive(Debug, Clone)]
pub struct LogitsHandle {
    pub data: Vec<f32>,
}

/// Attention mask strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMaskType {
    /// BERT-style bidirectional (no causal mask needed).
    Bidirectional,
    /// GPT-style causal mask.
    Causal,
}

/// Attention topology descriptor.
///
/// Lightweight struct that captures the attention geometry for a model.
/// Constructed from `ModelGeometry` without additional I/O.
#[derive(Debug, Clone)]
pub struct AttentionTopology {
    /// Model geometry (provides num_heads, num_kv_heads, head_dim, max_seq_len).
    pub geometry: Arc<crate::model_config::ModelGeometry>,
    /// Attention mask type (bidirectional vs causal).
    pub mask_type: AttentionMaskType,
}

impl AttentionTopology {
    /// Construct a bidirectional (encoder) topology for BERT-style models
    /// (embedding / reranker).
    pub fn bidirectional(geometry: Arc<crate::model_config::ModelGeometry>) -> Self {
        Self {
            geometry,
            mask_type: AttentionMaskType::Bidirectional,
        }
    }

    /// Construct a causal (decoder) topology for GPT-style generator models.
    pub fn causal(geometry: Arc<crate::model_config::ModelGeometry>) -> Self {
        Self {
            geometry,
            mask_type: AttentionMaskType::Causal,
        }
    }

    /// Legacy compatibility constructor (minimal bidirectional topology).
    pub fn linear() -> Self {
        let geometry = Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 1,
            num_layers: 1,
            vocab_size: 1,
            intermediate_size: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
        });
        Self::bidirectional(geometry)
    }

    /// Number of query attention heads.
    pub fn num_heads(&self) -> usize { self.geometry.num_heads }
    /// Number of KV heads.
    pub fn num_kv_heads(&self) -> usize { self.geometry.num_kv_heads }
    /// Head dimension.
    pub fn head_dim(&self) -> usize { self.geometry.head_dim }
    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize { self.geometry.max_seq_len }
}

/// A single sequence in a batch.
#[derive(Debug, Clone)]
pub struct SequenceInput {
    pub tokens: Vec<u32>,
    pub position: usize,
    pub draft_steps: usize,
    /// Pre-computed fused embedding sequence for this step (multimodal only).
    ///
    /// ARCH-MULTIMODAL-FUSION (SPEC/02-ARCHITECTURE.md):
    /// flat row-major `[tokens.len() * hidden_size]` f32 buffer. When present,
    /// the forward pass seeds `hidden_0` with this buffer and bypasses the
    /// `Gather(embed_tokens, input_ids)` node. Only populated for the prefill
    /// step of multimodal requests; decode steps still use Gather-on-input_ids
    /// because generated tokens are always text.
    ///
    /// Pure text requests leave this as `None` — they follow the standard
    /// Gather path with zero runtime overhead.
    pub fused_hidden: Option<Vec<f32>>,
}

/// Batched input for the forward pass.
#[derive(Debug, Clone)]
pub struct BatchInput {
    pub sequences: Vec<SequenceInput>,
}

use crate::graph::optimizer::{GraphOptimizer, OptimizationContext};
use crate::graph::types::FusedOp;
use crate::kv_cache::{KvCacheDoubleBuffer, KvCacheError, KvCacheSlot, KvCacheState};
use crate::loader::WeightsHandle;
use crate::loader::{Loader, LoaderError, WeightFormat};
use crate::manifest::{ModelKind, ModelManifest};
use crate::model_config::{ModelConfig, ModelConfigError};
use crate::tokenizer::{TokenizerError, TokenizerHandle};
use std::sync::Arc;

use crate::scheduler::batcher::{BatchAction, BatchResult, ContinuousBatcher};
use crate::scheduler::hgal::HGALConfig;
pub use crate::scheduler::types::RequestKind;
use crate::scheduler::types::BatchOrderPolicy;
use crate::scheduler::vllm2024::{AdaptiveChunkPolicy, Scheduler2024Config};
use crate::scheduler::{
    BasicObserver, GlobalMemoryManager, MemoryManagerError, PagedScheduler, PolicyVariant,
    PrefillPlan, ScheduledBatch, Sequence, SessionId, Tier, VirtualPageId,
};
use std::collections::HashMap;

#[derive(Debug)]
pub struct RequestData {
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub sampling_config: SamplingConfig,
    pub is_prefill: bool,
    // kv_cache: KvCacheHandle, // Moved to Scheduler/BlockTable management
    pub max_new_tokens: usize,
    pub finished: bool,
    pub session_id: Option<SessionId>,
    /// Thinking token budget: None = unlimited, Some(0) = disabled, Some(n) = max n tokens.
    pub thinking_budget: Option<usize>,
    /// Multimodal fused embedding (ARCH-MULTIMODAL-FUSION).
    ///
    /// Populated by `enqueue_with_multimodal` when the request carries image
    /// or audio content. Flat row-major `[prompt_tokens.len() * hidden_size]`
    /// f32 buffer: text positions gathered from `embed_tokens.weight`, media
    /// positions copied from the encoder output. Consumed once on the prefill
    /// step and then set to `None` so subsequent decode steps follow the
    /// standard Gather-on-input_ids path.
    pub fused_prefill_hidden: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default)]
struct OnnxFusedKernelStats {
    flash_attention: usize,
    swiglu: usize,
    rope: usize,
    fused_qkv_rope: usize,
    gqa: usize,
    moe_routing: usize,
    fused_rms_linear: usize,
    ple: usize,
    atomic: usize,
}

impl OnnxFusedKernelStats {
    fn fused_total(&self) -> usize {
        self.flash_attention
            + self.swiglu
            + self.rope
            + self.fused_qkv_rope
            + self.gqa
            + self.moe_routing
            + self.fused_rms_linear
            + self.ple
    }

    fn from_fused_graph(fused_graph: &crate::graph::FusedGraph) -> Self {
        let mut stats = Self::default();
        for node in &fused_graph.nodes {
            match &node.op {
                FusedOp::FlashAttention(_) => stats.flash_attention += 1,
                FusedOp::SwiGLU(_) => stats.swiglu += 1,
                FusedOp::RoPE(_) => stats.rope += 1,
                FusedOp::FusedQkvRope(_) => stats.fused_qkv_rope += 1,
                FusedOp::FusedQkvNormRope(_) => stats.fused_qkv_rope += 1,
                FusedOp::GQA(_) => stats.gqa += 1,
                FusedOp::MoERouting(_) => stats.moe_routing += 1,
                FusedOp::FusedRMSLinear(_) => stats.fused_rms_linear += 1,
                FusedOp::PerLayerEmbed(_) => stats.ple += 1,
                FusedOp::Atomic(_) => stats.atomic += 1,
            }
        }
        stats
    }
}

#[derive(Debug, Clone)]
struct OnnxGeneratorPlan {
    fused_kernels: OnnxFusedKernelStats,
    graph_outputs: Vec<String>,
    kv_outputs: Vec<String>,
    execution_order: Vec<OnnxKernelExecutionOp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnnxKernelExecutionOp {
    FlashAttention,
    SwiGlu,
    Rope,
    FusedQkvRope,
    Gqa,
    MoERouting,
    FusedRMSLinear,
    PerLayerEmbed,
    Atomic,
}

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Config(#[from] ModelConfigError),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    KvCache(#[from] KvCacheError),
    #[error(transparent)]
    MemoryManager(#[from] MemoryManagerError),
    #[error("scheduler error: {0}")]
    Scheduler(String),
    #[error("empty prompt tokens")]
    EmptyPrompt,
    #[error("backend returned empty sample")]
    EmptySample,
    #[error("request not found: {request_id}")]
    RequestNotFound { request_id: RequestId },
    #[error("onnx generator plan error: {0}")]
    OnnxPlan(String),
}

pub type ExecutorResult<T> = std::result::Result<T, ExecutorError>;

pub struct Executor<B: Backend<E> + 'static, E: Element = f32> {
    backend: B,
    scheduler: PagedScheduler,
    batcher: ContinuousBatcher,
    observer: BasicObserver,
    policy: PolicyVariant,
    requests: HashMap<RequestId, RequestData>,
    manifest: Arc<ModelManifest>,
    weights: WeightsHandle<B, E>,
    add_special_tokens: bool,
    geometry: Arc<crate::model_config::ModelGeometry>,
    model_config: ModelConfig,
    forward_config: GeneratorForwardConfig,
    kv_cache_config: KvCacheConfig,
    tokenizer: TokenizerHandle,
    kv_cache: Option<KvCacheDoubleBuffer>,
    kv_cache_slot: KvCacheSlot,
    memory_manager: GlobalMemoryManager,
    onnx_generator_plan: Option<OnnxGeneratorPlan>,
    topology: AttentionTopology,

    /// YAML→JIT graph executor (built at load time, used preferentially in forward pass).
    /// None when the architecture template is not registered or JIT compilation fails.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    graph_executor: Option<crate::graph::executor::FusedGraphExecutor>,
    /// Tracks stability telemetries to trigger Re-Fusion (Tier V.3)
    pub profile_accumulator: crate::scheduler::telemetry::ProfileAccumulator,
    /// Generation hooks (guardrails, probes) called after each decode step.
    /// per SPEC 04-API-DESIGN §7.4
    hooks: std::sync::Arc<std::sync::RwLock<Vec<Box<dyn crate::generation::GenerationHook>>>>,
    /// §12.6 系统级硬件拓扑（一次性探测，运行时固定）
    system_topology: crate::sensors::SystemTopology,
    /// §9.2 JIT Director Daemon（后台常驻监控线程）
    jit_director: Option<crate::jit::director::JitDirector>,
    /// §18.1 Epilogue 遥测聚合器
    telemetry_aggregator: crate::jit::epilogue::TelemetryAggregator,
    /// §9.1 Mega-Kernel 执行器（模型加载时编译，推理时优先使用）
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    mega_kernel: Option<super::mega_kernel::MegaKernelExecutor>,
    /// §11 TurboQuant 运行时（KV Cache 量化 + FWHT + RaBitQ）
    turboquant: crate::kv_cache::turboquant::TurboQuantRuntime,
    /// §9-§18 Epilogue 优化子系统（Gate Skip + Sink + Bypass + Prefetch + Spec）
    epilogue_subsystem: crate::jit::epilogue_subsystem::EpilogueSubsystem,
    /// §12.1 Sub-Batch 分发器（空间异构）
    sub_batch_dispatcher: crate::jit::sub_batch::SubBatchDispatcher,
    /// §12.4 Golden Bucket 注册表
    golden_buckets: crate::jit::golden_bucket::GoldenBucketRegistry,
    /// §12.4 SEQ 分布直方图（滑动窗口，供 JIT Director 运行时演化）
    seq_histogram: crate::jit::histogram::SeqHistogram,
    /// §10 Chunked Prefill 交织调度器（BatchManifest 组合 + 自适应 Chunk）
    chunked_prefill_scheduler: crate::scheduler::chunked_prefill::ChunkedPrefillScheduler,
    /// §9.1 Ragged Compaction 执行上下文（Compact→Execute→Scatter 三段式）
    ragged_compaction: crate::jit::ragged::RaggedCompaction,
    /// §15.4 MoE 专家热度管理器（冷专家封杀 + Uncommon Trap）
    moe_thermal: Option<crate::moe::thermal::ExpertThermalManager>,
    /// §15.3 MoE 硬件分发器（专家→GPU/CPU 分配）
    moe_dispatcher: Option<crate::moe::dispatch::MoeHardwareDispatcher>,
    /// §15.2 MoE 专家权重预取调度器
    moe_prefetcher: Option<crate::moe::prefetch::ExpertWeightPrefetcher>,
    /// §17 推测解码引擎（EESD / SAGUARO / Standard）
    spec_decoding: crate::speculative::engine::SpecDecodingState,
    /// §16.1 RAG system (set via set_rag_system)
    rag_system: Option<crate::rag::LateFusionRag>,
    /// §14.4 Hot JMP Patching 管理器（冷专家 NOP/Deopt + 前缀塌缩）
    hot_patch_manager: Option<crate::moe::hot_patch::HotPatchManager>,
    /// §9.3 残差数据总线（Injection/Recall 端口）
    residual_bus: crate::routing::ResidualBus,
    /// §9 Semantic Gatekeeper callback shim (set via set_sg_callback).
    sg_callback_shim: Option<crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim>,
}

/// Backward-compatible type alias for f32 executor.
pub type ExecutorF32<B> = Executor<B, f32>;

impl<B: Backend<E> + 'static, E: Element> Executor<B, E> {
    pub fn from_loader(
        backend: B,
        manifest: Arc<ModelManifest>,
        loader: &mut Loader,
    ) -> ExecutorResult<Self> {
        loader.set_manifest_if_missing(manifest.as_ref());
        let model_config = ModelConfig::from_loader(manifest.as_ref(), loader)?;
        loader.set_tie_word_embeddings_hint(model_config.tie_word_embeddings);
        if manifest.kind == ModelKind::Chat && model_config.use_cache == Some(false) {
            return Err(ExecutorError::Config(ModelConfigError::InvalidConfig(
                "config.use_cache=false is not supported for generator models in current gllm executor"
                    .to_string(),
            )));
        }
        let position_encoding = match manifest.kind {
            // Encoder-style embedding/reranker models (e.g. XLM-R/BERT) usually do not expose RoPE.
            // When rope_theta is absent/invalid, skip positional rotation instead of forcing RoPE.
            ModelKind::Embedding | ModelKind::Reranker | ModelKind::Classifier if model_config.rope_theta <= 0.0 => {
                PositionEncoding::None
            }
            _ => PositionEncoding::Rope,
        };
        // Validate intermediate_size is present before creating geometry
        if model_config.intermediate_size.is_none() {
            return Err(ExecutorError::Config(ModelConfigError::InvalidConfig(
                "model config missing intermediate_size (FFN hidden dimension)".to_string(),
            )));
        }

        // Create geometry ONCE — single source of truth for all model dimensions
        let geometry = Arc::new(crate::model_config::ModelGeometry::from_config(
            &model_config,
            manifest.moe_config,
        ));

        let forward_config = GeneratorForwardConfig {
            geometry: geometry.clone(),
            rope: RoPEConfig {
                theta: geometry.rope_theta,
                scale: geometry.rope_scale,
                interleaved: geometry.rope_interleaved,
                precompute: true,
            },
            position_encoding,
            arch_family: manifest.family(),
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: manifest.moe_config,
            paged_kv: PagedKvConfig {
                page_table: None,
                page_size: model_config.kv_cache_block_size,
            },
            graph_executor_ptr: std::ptr::null_mut(),
            callback_chain_ptr: std::ptr::null_mut(),
        };

        let block_size = model_config.kv_cache_block_size;
        let hgal_config = HGALConfig::default();
        let total_blocks = model_config.max_position_embeddings.div_ceil(block_size);

        let mut scheduler = PagedScheduler::new(total_blocks, block_size, hgal_config);
        scheduler.enable_vllm_2024(Scheduler2024Config {
            enable_2024_optimizations: true,
            ..Scheduler2024Config::default()
        });
        let page_size = scheduler.page_size();

        // CPU backend 只支持 f32 dtype for KV cache
        // 如果模型配置是 f16/bf16，需要强制转换为 f32
        let kv_dtype = if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend<E>>() {
            DType::F32 // CPU backend 只支持 f32
        } else {
            model_config.dtype
        };

        let kv_cache_config = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype,
            page_size,
            swap_config: None,
        };
        let onnx_generator_plan = Self::build_onnx_generator_plan(manifest.as_ref(), loader, &geometry)?;

        let is_moe = geometry.is_moe();
        let tokenizer = TokenizerHandle::from_loader(loader, manifest.kind)?;


        let mut weights = loader.upload_weights(&backend)?;
        let l1_capacity = total_blocks;
        let l2_capacity = total_blocks.saturating_mul(10);
        let l3_capacity = total_blocks.saturating_mul(100);
        let memory_manager =
            GlobalMemoryManager::new_with_capacities(l1_capacity, l2_capacity, l3_capacity);
        let topology = match manifest.kind {
            ModelKind::Chat => AttentionTopology::causal(geometry.clone()),
            ModelKind::Embedding | ModelKind::Reranker | ModelKind::Classifier => AttentionTopology::bidirectional(geometry.clone()),
        };

        // Build YAML→JIT graph executor: 3-phase pipeline
        // Phase 1: template → OnnxGraph → optimize → FusedGraph (uncompiled)
        // Phase 2: inject real weight shapes from loaded tensors (fixes GEMM dimensions)
        // Phase 3: JIT-compile + bind weight pointers
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        let graph_executor = {
            use crate::arch::{build_uncompiled_executor_from_yaml, register_builtin_templates, get_template, ResolvedConfig};
            use crate::compat::backend_trait::TensorLookup;
            register_builtin_templates();
            let resolved = ResolvedConfig::from_geometry(&geometry, std::collections::HashMap::new());
            let hidden = geometry.hidden_size;
            let cache = crate::compat::artifact_cache::ArtifactCache::new(None);
            let jit_backend = "cpu";
            let model_id = &manifest.model_id;

            // Phase 1: build uncompiled executor (template expand + graph optimize, NO JIT)
            // ARCH-TEMPLATE-KIND-OVERRIDE: manifest.arch 是 encoder/decoder 基础模板,
            // ModelKind (Reranker/Embedding) 可能需要专用模板 (如 xlmr → xlmr-reranker
            // 加 classifier head)。先走 KIND 查找, 未命中回落到基础模板。
            let effective_arch = crate::manifest::map_kind_template(&manifest.arch, manifest.kind)
                .unwrap_or_else(|| manifest.arch.clone());
            let uncompiled = match get_template(&effective_arch) {
                Some(tmpl) => {
                    let arch_name = tmpl.name.clone();
                    match build_uncompiled_executor_from_yaml(
                        &arch_name,
                        &resolved,
                        geometry.dtype,
                        manifest.family(),
                    ) {
                        Ok(ge) => Some(ge),
                        Err(e) => {
                            log::error!("Failed to build uncompiled executor for '{}': {}", arch_name, e);
                            None
                        }
                    }
                }
                None => {
                    log::error!("No YAML template found for architecture '{}'", effective_arch);
                    None
                }
            };

            if let Some(mut ge) = uncompiled {
                // Phase 2: inject real weight shapes from loaded tensors
                // This MUST happen BEFORE JIT compilation so that build_node_graph
                // uses correct GEMM dimensions (K, N) instead of defaulting to hidden_size.
                // Inject real weight shapes by trying each format's TensorProvider
                // ARCH-WEIGHT-CANONICAL-LAYOUT: shape_needs_transpose 按格式设置:
                //   SafeTensors/PyTorch: HF 原生 [out, in], 需语义转置到 [K, N]
                //   ONNX: 原生 [K, N] canonical
                //   GGUF: 走独立量化路径, 存储布局与 ONNX 一致
                if let Some(st) = loader.safetensors_ref() {
                    ge.graph_mut().bind_weight_shapes_fuzzy(st, true);
                } else if let Some(onnx) = loader.onnx_ref() {
                    ge.graph_mut().bind_weight_shapes_fuzzy(onnx, false);
                } else if let Some(gguf) = loader.gguf_ref() {
                    // ARCH-WEIGHT-CANONICAL-LAYOUT (GGUF): GGUF stores Linear weights
                    // as [out_features, in_features] (same row-major layout as HF
                    // SafeTensors). Linear weight roles need a metadata swap to
                    // canonical [K=in, N=out] just like SafeTensors. Embedding tables
                    // (Gather role) and 1D norm/bias weights are excluded by
                    // `is_linear_matmul_weight` so they keep their raw [vocab, hidden]
                    // / [hidden] shapes — matching the dequant path below which only
                    // physically transposes 2D Linear weights.
                    ge.graph_mut().bind_weight_shapes_fuzzy(gguf, true);
                }

                // Phase 3a: JIT-compile with correct weight shapes.
                // Use max_seq_len as compile-time M to force BLIS 5-loop path which
                // reads runtime seq_len from [rbp+16] for the M dimension loop.
                // Using seq_len=1 would trigger the direct (fully-unrolled) path
                // ARCH-SYMDIM-NO-CONST-DEGRADE: seq_len 不再是编译时参数。
                // 所有 Symbolic 维度在 JIT 中通过 BoundExpr::Symbolic 从运行时读取。
                // compile() 的 seq_len 参数仅用于 buffer 分配上界（max_for_allocation）。
                // seq_len 参数仅用于 buffer 分配上界
                let max_seq_for_alloc = geometry.max_seq_len;
                let compile_ok = ge.compile_with_cache(max_seq_for_alloc, hidden, geometry.dtype, model_id, jit_backend, &cache)
                    .map_err(|e| { eprintln!("[EXECUTOR-JIT-ERR] JIT compilation failed for '{}': {e}", effective_arch); })
                    .is_ok();

                if compile_ok {
                    // Phase 3b: bind weight pointers for runtime execution
                    let mut expected_names = std::collections::HashSet::new();
                    for key in ge.graph().weight_bindings.keys() {
                        expected_names.insert(key.clone());
                    }
                    let graph_inputs: std::collections::HashSet<String> = ge.graph().inputs.iter().cloned().collect();
                    for node in &ge.graph().nodes {
                        for input_name in &node.inputs {
                            if !graph_inputs.contains(input_name) && input_name.contains('.') {
                                expected_names.insert(input_name.clone());
                            }
                        }
                    }
                    // ARCH-GGUF-DEQUANT-ROLE-AWARE: build a name→needs_transpose
                    // set mirroring `bind_weight_shapes_fuzzy`'s role-based logic
                    // so dequant can decide per canonical_name whether to
                    // physically transpose the f32 payload. Linear MatMul weights
                    // (q/k/v/o/gate/up/down/lm_head) need [out, in] → [in, out];
                    // Gather embedding tables and 1D norm/bias weights must NOT
                    // be transposed.
                    //
                    // Snapshot to an owned `HashSet<String>` so we don't keep
                    // a borrow of `ge` while mutating it via `bind`/`bind_bytes`.
                    let needs_transpose_set: std::collections::HashSet<String> = {
                        let mut role_map: std::collections::HashMap<&str, Vec<(&FusedOp, usize)>> =
                            std::collections::HashMap::new();
                        for node in &ge.graph().nodes {
                            for (i, input) in node.inputs.iter().enumerate() {
                                role_map.entry(input.as_str()).or_default().push((&node.op, i));
                            }
                        }
                        role_map
                            .keys()
                            .filter(|name| {
                                crate::graph::types::is_linear_matmul_weight(name, &role_map)
                            })
                            .map(|name| name.to_string())
                            .collect()
                    };
                    for canonical_name in expected_names {
                        let needs_transpose = needs_transpose_set.contains(&canonical_name);
                        let found = TensorLookup::get_tensor(&weights, &canonical_name)
                            .map(|t| t.as_ref().as_ptr() as *const f32);
                        let found = found.or_else(|| {
                            // Multi-modal models (e.g. Gemma 4 `Gemma4ForConditionalGeneration`)
                            // nest the text decoder weights under `model.language_model.` while
                            // the YAML template names them with the bare `model.` prefix. Include
                            // both add/strip directions for these multi-modal nesting variants.
                            const PREFIXES: &[&str] = &[
                                "roberta.",
                                "bert.",
                                "model.",
                                "encoder.",
                                "transformer.",
                                "language_model.",
                                "model.language_model.",
                            ];
                            for prefix in PREFIXES {
                                if let Some(stripped) = canonical_name.strip_prefix(prefix) {
                                    if let Some(t) = TensorLookup::get_tensor(&weights, stripped) {
                                        return Some(t.as_ref().as_ptr() as *const f32);
                                    }
                                }
                            }
                            for prefix in PREFIXES {
                                let prefixed = format!("{prefix}{canonical_name}");
                                if let Some(t) = TensorLookup::get_tensor(&weights, &prefixed) {
                                    return Some(t.as_ref().as_ptr() as *const f32);
                                }
                            }
                            // tie_word_embeddings: lm_head.weight → model.embed_tokens.weight
                            // (也处理 GGUF weight tying via token_embd.weight)
                            //
                            // ARCH-TIED-EMBED-TRANSPOSE: embed_tokens 作为 Gather 的 table
                            // 被加载为 [vocab, hidden] (HF 原始 layout, shape_needs_transpose=false)。
                            // 但 lm_head 是 MatMul 权重 (shape_needs_transpose=true),
                            // 需要 canonical [hidden, vocab]。
                            // 直接把 embed_tokens 的 ptr 绑给 lm_head.weight 会让 MatMul
                            // 按 [hidden, vocab] 解读 [vocab, hidden] 字节流 → logits 完全错误。
                            // 正确做法: 同一块数据需要两份视图 — 原样给 Gather,
                            // 转置后给 lm_head。这里返回 None 让下方的专门处理分支
                            // 通过 bind_bytes 材化出 transpose 后的字节。
                            if canonical_name == "lm_head.weight" {
                                return None;  // 交给下方 lm_head 专用转置分支处理
                            }
                            // Decoder layer HF → GGUF name translation.
                            // E.g. model.layers.0.self_attn.q_proj.weight → blk.0.attn_q.weight
                            for alias in crate::weight_names::all_decoder_weight_aliases(&canonical_name) {
                                if let Some(t) = TensorLookup::get_tensor(&weights, &alias) {
                                    return Some(t.as_ref().as_ptr() as *const f32);
                                }
                            }
                            // Global decoder weight aliases (final norm, lm_head, embedding).
                            // E.g. model.norm.weight → output_norm.weight (GGUF)
                            //      lm_head.weight   → token_embd.weight (tied embedding)
                            let global_alias_groups: &[&dyn Fn() -> Vec<String>] = &[
                                &crate::weight_names::decoder_final_norm_aliases,
                                &crate::weight_names::lm_head_aliases,
                                &crate::weight_names::decoder_embed_aliases,
                            ];
                            for alias_fn in global_alias_groups {
                                let aliases = alias_fn();
                                if aliases.contains(&canonical_name) {
                                    for alias in &aliases {
                                        if alias != &canonical_name {
                                            if let Some(t) = TensorLookup::get_tensor(&weights, alias) {
                                                return Some(t.as_ref().as_ptr() as *const f32);
                                            }
                                        }
                                    }
                                }
                            }
                            None
                        });
                        // Dequantize a QuantizedTensor to f32 bytes using the backend.
                        //
                        // ARCH-GGUF-DEQUANT-ROLE-AWARE: GGUF/HF store 2D weights as
                        // `[out_features, in_features]` row-major. Linear MatMul
                        // operands need a physical transpose to canonical
                        // `[K=in, N=out]` (matching `format_needs_transpose=true` in
                        // `bind_weight_shapes_fuzzy`). Gather embedding tables MUST
                        // be left as `[vocab, hidden]` row-major so the JIT Gather
                        // reads `table[token_id * embed_dim + j]` correctly.
                        // 1D weights (norm/bias) never get transposed.
                        let dequant = |q_tensor: &crate::loader::QuantizedTensor,
                                       transpose: bool|
                         -> Option<Vec<u8>> {
                            let numel: usize = q_tensor.shape.iter().product();
                            let mut f32_out = vec![0.0f32; numel];
                            backend.dequantize(&q_tensor.data, &mut f32_out, q_tensor.quant_type)
                                .ok()?;
                            let shape = &q_tensor.shape;
                            let f32_to_bytes = |v: &[f32]| -> Vec<u8> {
                                v.iter().flat_map(|f| f.to_le_bytes()).collect()
                            };
                            if transpose && shape.len() == 2 {
                                let rows = shape[0]; // out_features = N
                                let cols = shape[1]; // in_features = K
                                let mut transposed = vec![0.0f32; numel];
                                for r in 0..rows {
                                    for c in 0..cols {
                                        transposed[c * rows + r] = f32_out[r * cols + c];
                                    }
                                }
                                Some(f32_to_bytes(&transposed))
                            } else {
                                Some(f32_to_bytes(&f32_out))
                            }
                        };
                        // ARCH-TIED-EMBED-TRANSPOSE: lm_head.weight 在 tied-embed 情况下
                        // 需从 embed_tokens [vocab, hidden] 转置到 canonical [hidden, vocab]。
                        let lm_head_tied_bytes: Option<Vec<u8>> = if found.is_none()
                            && canonical_name == "lm_head.weight"
                        {
                            const EMBED_ALIASES: &[&str] = &[
                                "model.embed_tokens.weight", "embed_tokens.weight",
                                "transformer.wte.weight", "embeddings.word_embeddings.weight",
                                "token_embd.weight",
                                // Gemma 4 multi-modal nesting
                                "model.language_model.embed_tokens.weight",
                                "language_model.embed_tokens.weight",
                            ];
                            let mut result: Option<Vec<u8>> = None;
                            for alias in EMBED_ALIASES {
                                if let Some(t) = TensorLookup::get_tensor(&weights, alias) {
                                    // E 可能是 f32/f16/bf16; CPU backend 实际是 f32。
                                    // 用 ptr 强转为 &[f32] — 条件: E size == f32 size (CPU 路径).
                                    let data_slice_e: &[E] = t.as_ref();
                                    if std::mem::size_of::<E>() != std::mem::size_of::<f32>() {
                                        continue;
                                    }
                                    let data: &[f32] = unsafe {
                                        std::slice::from_raw_parts(
                                            data_slice_e.as_ptr() as *const f32,
                                            data_slice_e.len(),
                                        )
                                    };
                                    let shape = TensorLookup::tensor_shape(&weights, alias).unwrap_or(&[][..]);
                                    if shape.len() == 2 {
                                        let vocab = shape[0];
                                        let hidden = shape[1];
                                        if data.len() == vocab * hidden {
                                            // Transpose [vocab, hidden] → [hidden, vocab]
                                            let mut transposed = vec![0.0f32; vocab * hidden];
                                            for v in 0..vocab {
                                                for h in 0..hidden {
                                                    transposed[h * vocab + v] = data[v * hidden + h];
                                                }
                                            }
                                            let bytes: Vec<u8> = transposed.iter()
                                                .flat_map(|f| f.to_le_bytes())
                                                .collect();
                                            result = Some(bytes);
                                            break;
                                        }
                                    }
                                }
                            }
                            result
                        } else { None };

                        if let Some(bytes) = lm_head_tied_bytes {
                            ge = ge.bind_bytes(canonical_name.clone(), bytes);
                        } else if let Some(ptr) = found {
                            ge = ge.bind(canonical_name.clone(), ptr);
                        } else if let Some(q_tensor) = TensorLookup::get_quantized(&weights, &canonical_name) {
                            // ARCH-MXFP4-SEPARATE: MXFP4 blocks/scales are raw bytes consumed
                            // directly by JIT MoEDispatchPacked dequant — never host-dequantized.
                            if matches!(q_tensor.quant_type, gllm_kernels::quant::QuantType::Mxfp4 { .. }) {
                                ge = ge.bind_bytes(canonical_name.clone(), q_tensor.data.clone());
                            } else if let Some(f32_bytes) = dequant(q_tensor, needs_transpose) {
                                ge = ge.bind_bytes(canonical_name.clone(), f32_bytes);
                            }
                        } else {
                            // Try decoder layer GGUF aliases for quantized tensors.
                            // E.g. model.layers.0.self_attn.q_proj.weight → blk.0.attn_q.weight
                            let mut bound_alias = false;
                            for alias in crate::weight_names::all_decoder_weight_aliases(&canonical_name) {
                                if let Some(t) = TensorLookup::get_tensor(&weights, &alias) {
                                    ge = ge.bind(canonical_name.clone(), t.as_ref().as_ptr() as *const f32);
                                    bound_alias = true;
                                    break;
                                }
                                if let Some(q_tensor) = TensorLookup::get_quantized(&weights, &alias) {
                                    if matches!(q_tensor.quant_type, gllm_kernels::quant::QuantType::Mxfp4 { .. }) {
                                        ge = ge.bind_bytes(canonical_name.clone(), q_tensor.data.clone());
                                        bound_alias = true;
                                    } else if let Some(f32_bytes) = dequant(q_tensor, needs_transpose) {
                                        ge = ge.bind_bytes(canonical_name.clone(), f32_bytes);
                                        bound_alias = true;
                                    }
                                    break;
                                }
                            }
                            // Try global decoder weight aliases (final norm, lm_head, embedding).
                            // E.g. model.norm.weight → output_norm.weight (GGUF)
                            if !bound_alias {
                                let global_alias_groups: &[&dyn Fn() -> Vec<String>] = &[
                                    &crate::weight_names::decoder_final_norm_aliases,
                                    &crate::weight_names::lm_head_aliases,
                                    &crate::weight_names::decoder_embed_aliases,
                                ];
                                'outer: for alias_fn in global_alias_groups {
                                    let aliases = alias_fn();
                                    if aliases.contains(&canonical_name) {
                                        for alias in &aliases {
                                            if alias != &canonical_name {
                                                if let Some(t) = TensorLookup::get_tensor(&weights, alias) {
                                                    ge = ge.bind(canonical_name.clone(), t.as_ref().as_ptr() as *const f32);
                                                    break 'outer;
                                                }
                                                if let Some(q_tensor) = TensorLookup::get_quantized(&weights, alias) {
                                                    if matches!(q_tensor.quant_type, gllm_kernels::quant::QuantType::Mxfp4 { .. }) {
                                                        ge = ge.bind_bytes(canonical_name.clone(), q_tensor.data.clone());
                                                    } else if let Some(f32_bytes) = dequant(q_tensor, needs_transpose) {
                                                        ge = ge.bind_bytes(canonical_name.clone(), f32_bytes);
                                                    }
                                                    break 'outer;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some(ge)
                } else {
                    None
                }
            } else {
                None
            }
        };

        // §9.1: 编译 true mega-kernel（铁律: 无 fallback，编译失败 = 致命错误）
        // ARCH-MOE-MEGA-KERNEL: MoE 模型的 packed expert weights (blocks/scales/bias/router)
        // 无法映射到 PerLayerWeightLayout 的密集 FFN 槽位。MoE 模型走 per-node
        // FusedGraphExecutor 路径，由 JIT MoEDispatchPacked 直接消费原始字节。
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        let mega_kernel = if is_moe {
            log::info!("executor: MoE model — skipping mega-kernel (uses per-node JIT path)");
            None
        } else {
            use crate::compat::backend_trait::TensorLookup;
            let ge = graph_executor.as_ref()
                .ok_or_else(|| ExecutorError::Backend(BackendError::Other(
                    "graph_executor not available for mega-kernel compilation".into(),
                )))?;

            // Resolve EOS token ID: prefer tokenizer vocab (authoritative) over config.json
            // (many ONNX/Community repos ship config.json with eos_token_id=0).
            let eos_id = tokenizer.eos_token_id()
                .or(model_config.eos_token_id)
                .unwrap_or(2);
            log::info!(
                "mega-kernel eos_id={} (config={:?}, tokenizer={:?})",
                eos_id, model_config.eos_token_id, tokenizer.eos_token_id(),
            );

            let wb = &ge.graph().weight_bindings;
            let mut weight_ptrs: std::collections::HashMap<String, *const u8> = std::collections::HashMap::new();
            let mut weight_sizes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for (name, binding) in wb {
                if let Some(ptr) = binding.ptr {
                    weight_ptrs.insert(name.clone(), ptr as *const u8);
                    let num_elems: usize = binding.shape.iter().product();
                    weight_sizes.insert(name.clone(), num_elems * 4);
                } else if let Some(ref data) = binding.data {
                    weight_ptrs.insert(name.clone(), data.as_ptr());
                    weight_sizes.insert(name.clone(), data.len());
                }
            }

            // Supplement weight bindings that the compat-layer YAML template may not
            // reference but the mega-kernel packing code needs. Two categories:
            // 1. Fused weights (qkv_proj, gate_up_proj) — the template uses separate names
            //    but the safetensors stores a single fused tensor.
            // 2. Separate weights (gate_proj) — the template omits gate_proj for GeGLU
            //    architectures (e.g., Gemma 4 template only references up_proj).
            // The weights HashMap uses original safetensors names (e.g.,
            // model.language_model.layers.X.*), so we try both the normalized
            // name and prefixed variants.
            // IMPORTANT: Linear weights in safetensors are [out, in]. The mega-kernel
            // expects canonical [in, out]. We must transpose supplemented weights.
            const MID_PREFIX_REPLACEMENTS: &[(&str, &str)] = &[
                ("model.layers.", "model.language_model.layers."),
            ];
            // Keep transposed data alive until compile_from_geometry finishes.
            let mut _supplement_owned: Vec<Vec<u8>> = Vec::new();
            // WeightsHandle data is already canonical [in, out] (loader's
            // normalize_linear_weight_layout physically transposes on upload).
            // Just copy bytes — no transpose needed.
            let supplement_weight = |name: &str, owned: &mut Vec<Vec<u8>>| -> Option<(*const u8, usize)> {
                let mut candidates = vec![name.to_string()];
                for (from, to) in MID_PREFIX_REPLACEMENTS {
                    if let Some(idx) = name.find(from) {
                        let mut replaced = name.to_string();
                        replaced.replace_range(idx..idx + from.len(), to);
                        candidates.push(replaced);
                    }
                }
                for lookup in &candidates {
                    if let Some(t) = TensorLookup::get_tensor(&weights, lookup) {
                        let data: &[E] = t.as_ref();
                        let size = data.len() * std::mem::size_of::<E>();
                        let bytes: Vec<u8> = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const u8, size).to_vec()
                        };
                        let ptr = bytes.as_ptr();
                        owned.push(bytes);
                        return Some((ptr, size));
                    }
                }
                None
            };
            for layer_idx in 0..geometry.num_layers {
                // Fused QKV
                let qkv_name = format!("model.layers.{}.self_attn.qkv_proj.weight", layer_idx);
                if !weight_ptrs.contains_key(&qkv_name) {
                    if let Some((ptr, size)) = supplement_weight(&qkv_name, &mut _supplement_owned) {
                        weight_ptrs.insert(qkv_name.clone(), ptr);
                        weight_sizes.insert(qkv_name, size);
                    }
                }
                // Fused gate_up
                let gu_name = format!("model.layers.{}.mlp.gate_up_proj.weight", layer_idx);
                if !weight_ptrs.contains_key(&gu_name) {
                    if let Some((ptr, size)) = supplement_weight(&gu_name, &mut _supplement_owned) {
                        weight_ptrs.insert(gu_name.clone(), ptr);
                        weight_sizes.insert(gu_name, size);
                    }
                }
                // Separate gate_proj (GeGLU architectures like Gemma 4 where the YAML
                // template only references up_proj but the safetensors has gate_proj too).
                let gate_name = format!("model.layers.{}.mlp.gate_proj.weight", layer_idx);
                if !weight_ptrs.contains_key(&gate_name) {
                    if let Some((ptr, size)) = supplement_weight(&gate_name, &mut _supplement_owned) {
                        weight_ptrs.insert(gate_name.clone(), ptr);
                        weight_sizes.insert(gate_name, size);
                    }
                }
            }
            // Detect heterogeneous layers (e.g., Gemma-4 E2B with sliding + full attention).
            // Build HeteroLayerConfig if detected, otherwise compile homogeneous mega-kernel.
            let ref_size_q = weight_sizes.get("model.layers.0.self_attn.q_proj.weight").copied();
            let ref_size_gate = weight_sizes.get("model.layers.0.mlp.gate_proj.weight").copied();
            let mut full_attention_indices: Vec<usize> = Vec::new();
            let mut hetero_q_dim_full: usize = 0;
            let mut hetero_kv_dim_full: usize = 0;
            let mut hetero_intermediate_differs = false;
            if let Some(ref_q) = ref_size_q {
                for layer_idx in 1..geometry.num_layers {
                    let q_key = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
                    if let Some(&sz) = weight_sizes.get(&q_key) {
                        if sz != ref_q {
                            full_attention_indices.push(layer_idx);
                            if hetero_q_dim_full == 0 {
                                let ref_q_dim = ref_q / (geometry.hidden_size * 4);
                                let full_q_dim = sz / (geometry.hidden_size * 4);
                                hetero_q_dim_full = full_q_dim;
                                let k_key = format!("model.layers.{}.self_attn.k_proj.weight", layer_idx);
                                if let Some(&k_sz) = weight_sizes.get(&k_key) {
                                    hetero_kv_dim_full = k_sz / (geometry.hidden_size * 4);
                                }
                            }
                        }
                    }
                    if let Some(ref_gate) = ref_size_gate {
                        let gate_key = format!("model.layers.{}.mlp.gate_proj.weight", layer_idx);
                        if let Some(&sz) = weight_sizes.get(&gate_key) {
                            if sz != ref_gate {
                                hetero_intermediate_differs = true;
                            }
                        }
                    }
                }
            }

            let hetero_config = if !full_attention_indices.is_empty() {
                let num_full = full_attention_indices.len();
                if num_full > 0 {
                    let first_full = full_attention_indices[0];
                    let sliding_per_segment = first_full;
                    let num_segments = num_full;
                    let total_expected = num_segments * (sliding_per_segment + 1);
                    if total_expected == geometry.num_layers {
                        let ref_q_dim = ref_size_q.unwrap_or(0) / (geometry.hidden_size * 4);
                        let sliding_head_dim = ref_q_dim / geometry.num_heads;
                        let full_head_dim = hetero_q_dim_full / geometry.num_heads;
                        let full_kv_dim = hetero_kv_dim_full;
                        let full_num_kv_heads = if full_head_dim > 0 { full_kv_dim / full_head_dim } else { geometry.num_kv_heads };

                        // Detect FFN size transition: scan gate_proj sizes to find where large FFN starts.
                        let ref_gate = ref_size_gate.unwrap_or(0);
                        let small_intermediate = geometry.intermediate_size;
                        let mut large_intermediate = small_intermediate;
                        let mut large_ffn_start_segment = num_segments; // default: no large FFN
                        if hetero_intermediate_differs && ref_gate > 0 {
                            // Find first layer with different gate_proj size
                            let first_large_layer = (1..geometry.num_layers).find(|&l| {
                                let gk = format!("model.layers.{}.mlp.gate_proj.weight", l);
                                weight_sizes.get(&gk).map_or(false, |&s| s != ref_gate)
                            });
                            if let Some(fl) = first_large_layer {
                                let large_gate = weight_sizes.get(&format!("model.layers.{}.mlp.gate_proj.weight", fl)).copied().unwrap_or(ref_gate);
                                large_intermediate = large_gate / (geometry.hidden_size * 4);
                                // Convert layer index to segment index
                                large_ffn_start_segment = fl / (sliding_per_segment + 1);
                            }
                        }

                        log::info!(
                            "executor: heterogeneous layers: {} segments of [{} sliding + 1 full], sliding_head_dim={}, full_head_dim={}, small_ffn={}, large_ffn={}, large_start_seg={}",
                            num_segments, sliding_per_segment, sliding_head_dim, full_head_dim,
                            small_intermediate, large_intermediate, large_ffn_start_segment,
                        );

                        Some(gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig {
                            num_segments,
                            sliding_per_segment,
                            sliding_head_dim,
                            sliding_num_kv_heads: geometry.num_kv_heads,
                            full_head_dim,
                            full_num_kv_heads,
                            full_layer_indices: full_attention_indices,
                            small_intermediate,
                            large_intermediate,
                            large_ffn_start_segment,
                        })
                    } else {
                        log::info!(
                            "executor: heterogeneous layers detected but pattern doesn't match [N+1]×M: total={} expected={} — skipping mega-kernel",
                            geometry.num_layers, total_expected,
                        );
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let arch_for_template = crate::manifest::map_kind_template(&manifest.arch, manifest.kind)
                .unwrap_or_else(|| manifest.arch.clone());
            let template = crate::arch::get_template(&arch_for_template);

            // Build mega-kernel business config from template + geometry.
            use gllm_kernels::compiler::MegaKernelBusinessConfig;
            use gllm_kernels::compiler::mega_kernel_abi::FfnActivation;
            let business_config = MegaKernelBusinessConfig {
                has_head_rms_norm: template.as_ref().map_or(false, |t| t.has_head_rms_norm),
                head_rms_norm_eps: template.as_ref().map_or(1e-6, |t| t.head_rms_norm_eps),
                ffn_activation: match geometry.hidden_act.as_ref().map(|a| a.as_str()).unwrap_or("") {
                    "gelu_new" | "gelu_pytorch_tanh" | "gelu" => FfnActivation::GeGLU,
                    _ => FfnActivation::SwiGLU,
                },
                has_qk_norm: template.as_ref().map_or(false, |t| t.name == "gemma4"),
                has_value_norm: template.as_ref().map_or(false, |t| t.name == "gemma4"),
                value_norm_eps: geometry.norm_eps,
                logit_softcapping: geometry.final_logit_softcapping,
                // Gemma models: embedding scaled by sqrt(hidden_size)
                embedding_scale: if template.as_ref().map_or(false, |t| t.name == "gemma4") {
                    Some((geometry.hidden_size as f32).sqrt())
                } else {
                    None
                },
                ..MegaKernelBusinessConfig::default()
            };

            let mega = if let Some(ref tmpl) = template {
                // REQ-UGS-001/003: YAML template → CompilerGraph direct path.
                // Bypasses graph_builders.rs hand-written decoder_model().
                let resolved = crate::arch::ResolvedConfig::from_geometry(
                    &geometry, std::collections::HashMap::new(),
                );
                match super::mega_kernel::MegaKernelExecutor::compile_from_template(
                    tmpl,
                    &resolved,
                    &geometry,
                    &weight_ptrs,
                    &weight_sizes,
                    eos_id,
                    business_config.clone(),
                    hetero_config.clone(),
                ) {
                    Ok(m) => {
                        log::info!("executor: mega-kernel compiled from YAML template '{}'", tmpl.name);
                        m
                    }
                    Err(e) => {
                        log::warn!("template-based compilation failed ({}), falling back to geometry path", e);
                        super::mega_kernel::MegaKernelExecutor::compile_from_geometry(
                            &geometry,
                            &weight_ptrs,
                            &weight_sizes,
                            eos_id,
                            business_config,
                            hetero_config,
                        ).map_err(|e2| ExecutorError::Backend(BackendError::Other(
                            format!("mega-kernel compilation failed (template: {}, geometry: {})", e, e2),
                        )))?
                    }
                }
            } else {
                super::mega_kernel::MegaKernelExecutor::compile_from_geometry(
                    &geometry,
                    &weight_ptrs,
                    &weight_sizes,
                    eos_id,
                    business_config,
                    hetero_config,
                ).map_err(|e| ExecutorError::Backend(BackendError::Other(
                    format!("mega-kernel compilation failed: {}", e),
                )))?
            };

            log::info!("executor: true mega-kernel compiled");
            Some(mega)
        };

        // MoE subsystem: read global ExecutionPlan once, initialize all three
        // MoE components in a single block.
        // MoE subsystem + JitDirector: read global ExecutionPlan once, initialize all
        // MoE components and the JIT director in a single block.
        let (moe_thermal, moe_dispatcher, moe_prefetcher, jit_director, hot_patch_manager) = if is_moe {
            let exec_plan = gllm_kernels::compiler::planner::global_execution_plan();
            let bias = &exec_plan.strategy_bias;

            let thermal = crate::moe::thermal::ExpertThermalManager::new(geometry.num_experts)
                .with_eviction_aggressiveness(bias.expert_eviction_aggressiveness());

            let route_config = crate::moe::routing::ExpertRouteConfig::new(
                geometry.num_experts,
                geometry.moe_top_k,
            );
            let dispatcher = crate::moe::dispatch::MoeHardwareDispatcher::new(route_config.clone());

            let prefetcher = crate::moe::prefetch::ExpertWeightPrefetcher::new(
                geometry.num_experts,
                geometry.expert_weight_bytes(),
            ).with_prefetch_priority(bias.expert_prefetch_priority());

            let director_config = crate::jit::director::DirectorConfig {
                num_experts: geometry.num_experts,
                ..Default::default()
            };
            let director = crate::jit::director::JitDirector::spawn(director_config);

            let patch_manager = crate::moe::hot_patch::HotPatchManager::new(route_config);

            (Some(thermal), Some(dispatcher), Some(prefetcher), Some(director), Some(patch_manager))
        } else {
            (None, None, None, None, None)
        };

        // §12.6: Derive CompilerConstraints from real SystemTopology (not default)
        // This must happen before Executor construction so we can pass to sub_batch_dispatcher
        // and golden_buckets. We detect topology once here, then move it into the struct.
        let pre_topology = crate::sensors::SystemTopology::detect();
        let compiler_constraints = pre_topology.constraints.clone();

        // §12.4: Run LatencyProfiler to discover real hardware spill points
        let probe_config = crate::jit::profiler::ProbeConfig::for_model(
            geometry.hidden_size,
            geometry.max_seq_len.min(4096),
        );
        let probe_result = match crate::jit::profiler::LatencyProfiler::probe_cpu(&probe_config) {
            Ok(result) => {
                log::info!(
                    "executor: §12.4 LatencyProfiler probe complete — spill_points={:?}, l2_thrash={}",
                    result.spill_points, result.l2_thrash_threshold,
                );
                result
            }
            Err(e) => {
                log::warn!(
                    "executor: §12.4 LatencyProfiler probe failed ({e}), deriving from topology"
                );
                // Derive probe result from topology constraints when micro-benchmark fails
                let (_, l2, _) = pre_topology.profile.cache_sizes();
                let elem_bytes = 4usize;
                let l2_thrash = if l2 > 0 {
                    l2 / (geometry.hidden_size * 2 * elem_bytes).max(1)
                } else {
                    2048
                };
                crate::jit::profiler::ProbeResult {
                    spill_points: vec![l2_thrash / 4, l2_thrash / 2, l2_thrash],
                    smem_cliffs: Vec::new(),
                    l2_thrash_threshold: l2_thrash,
                    device_fingerprint: format!("topology-derived-{}", pre_topology.cpu.core_count),
                    raw_measurements: std::collections::HashMap::new(),
                }
            }
        };

        // ARCH-WEIGHT-BLOB-REMAPPING: graph_executor 编译完成后，收集已预打包到
        // weight_blob 的权重名集合，从 WeightsHandle.tensors 中精确释放这些权重。
        // 只释放 pre_bound_weight_names() 中的权重（FFN 层的 gate/up/down_proj），
        // 保留 attention 层权重（needs_runtime_weight_pack=true，运行时仍从 tensors 读取）。
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        if let Some(ref ge) = graph_executor {
            let safe = ge.pre_bound_weight_names();
            if !safe.is_empty() {
                weights.release_compiled_weights(&safe);
            }
        }

        // §9.1: Detect CompactPlatform from topology for RaggedCompaction
        let compact_platform = crate::jit::ragged::CompactPlatform::detect(
            if pre_topology.has_gpu() { "cuda" } else { "cpu" },
            compiler_constraints.simd_width_bits >= 512,
            compiler_constraints.simd_width_bits == 128 && cfg!(target_arch = "aarch64"),
            if cfg!(target_arch = "aarch64") { compiler_constraints.simd_width_bits / 8 } else { 0 },
            32, // default warp_size
        );

        Ok(Self {
            backend,
            scheduler,
            manifest,
            weights,
            add_special_tokens: true,
            geometry: geometry.clone(),
            model_config,
            forward_config,
            kv_cache_config,
            tokenizer,
            kv_cache: None,
            kv_cache_slot: KvCacheSlot::Front,
            memory_manager,
            onnx_generator_plan,
            batcher: ContinuousBatcher::new().with_chunked(
                crate::scheduler::vllm2024::ChunkedConfig::default(),
            ),
            observer: BasicObserver::new(),
            policy: PolicyVariant::default(),
            requests: HashMap::new(),
            topology,
            graph_executor,
            profile_accumulator: crate::scheduler::telemetry::ProfileAccumulator::new(),
            hooks: std::sync::Arc::new(std::sync::RwLock::new(Vec::new())),
            system_topology: {
                log::info!(
                    "SystemTopology: {} cores, NUMA={}, SIMD={}bit, L2={}KB",
                    pre_topology.cpu.core_count,
                    pre_topology.numa_node_count(),
                    pre_topology.constraints.simd_width_bits,
                    pre_topology.constraints.l2_cache_size / 1024,
                );
                pre_topology
            },
            jit_director,
            telemetry_aggregator: crate::jit::epilogue::TelemetryAggregator::new(),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
            mega_kernel,
            turboquant: {
                // §11: 非 F32 模型自动启用 TurboQuant（FWHT + KV 非对称量化）
                if geometry.dtype != gllm_kernels::types::DType::F32 {
                    log::info!("executor: §11 TurboQuant enabled (dtype={:?}, fwht=true)", geometry.dtype);
                    crate::kv_cache::turboquant::TurboQuantRuntime::new(
                        crate::kv_cache::turboquant::TurboQuantConfig {
                            bits: 4,
                            sink_count: 4,
                            fwht_enabled: true,
                            mode: crate::kv_cache::quant::QuantMode::Deterministic,
                            dual_track_enabled: false,
                        },
                    ).map_err(|e| ExecutorError::Config(
                        ModelConfigError::InvalidConfig(format!("TurboQuant init failed: {e}")),
                    ))?
                } else {
                    crate::kv_cache::turboquant::TurboQuantRuntime::disabled()
                }
            },
            epilogue_subsystem: {
                let config = crate::jit::epilogue_subsystem::EpilogueConfig {
                    num_layers: geometry.num_layers,
                    num_experts: geometry.num_experts,
                    ..Default::default()
                };
                crate::jit::epilogue_subsystem::EpilogueSubsystem::new(config)
            },
            sub_batch_dispatcher: crate::jit::sub_batch::SubBatchDispatcher::new(
                compiler_constraints.clone(),
            ),
            golden_buckets: crate::jit::golden_bucket::GoldenBucketRegistry::from_probe_results(
                &probe_result,
                compiler_constraints.clone(),
            ),
            seq_histogram: crate::jit::histogram::SeqHistogram::new(
                10000, // sliding window: last 10k samples
                geometry.max_seq_len.max(4096),
            ),
            chunked_prefill_scheduler: crate::scheduler::chunked_prefill::ChunkedPrefillScheduler::new(
                crate::scheduler::chunked_prefill::ChunkedPrefillConfig::default(),
            ),
            ragged_compaction: crate::jit::ragged::RaggedCompaction::new(compact_platform),
            // MoE subsystem: single global_execution_plan() read, single expert check.
            moe_thermal,
            moe_dispatcher,
            moe_prefetcher,
            spec_decoding: crate::speculative::engine::SpecDecodingState::new_standard(),
            rag_system: None,
            hot_patch_manager,
            residual_bus: {
                let mut bus = crate::routing::ResidualBus::new(
                    geometry.hidden_size,
                    geometry.num_layers,
                );
                // §9.3: Register standard Injection/Recall ports
                // RAG injection at mid-semantic layer (§16.1)
                let rag_layer = geometry.num_layers / 2;
                bus.register(crate::routing::BusPort::injection(
                    rag_layer,
                    crate::routing::BusPortTag::RagInjection,
                ));
                // Early exit recall at golden-ratio points (§16.2)
                let exit_layer = (geometry.num_layers as f64 * 0.786) as usize;
                bus.register(crate::routing::BusPort::recall(
                    exit_layer.min(geometry.num_layers.saturating_sub(1)),
                    crate::routing::BusPortTag::EarlyExit,
                ));
                // Intent recall at deep semantic layer (§16.3)
                let intent_layer = (geometry.num_layers as f64 * 0.75) as usize;
                bus.register(crate::routing::BusPort::recall(
                    intent_layer.min(geometry.num_layers.saturating_sub(1)),
                    crate::routing::BusPortTag::IntentRecall,
                ));
                // Guardrail injection (§16.4)
                let guard_layer = geometry.num_layers.saturating_sub(2);
                bus.register(crate::routing::BusPort::injection(
                    guard_layer.min(geometry.num_layers.saturating_sub(1)),
                    crate::routing::BusPortTag::Guardrail,
                ));
                log::info!(
                    "executor: §9.3 ResidualBus initialized ({} ports, hidden_size={}, num_layers={})",
                    bus.active_port_count(),
                    geometry.hidden_size,
                    geometry.num_layers,
                );
                bus
            },
            sg_callback_shim: None,
        })
    }

    fn build_onnx_generator_plan(
        manifest: &ModelManifest,
        loader: &mut Loader,
        geometry: &Arc<crate::model_config::ModelGeometry>,
    ) -> ExecutorResult<Option<OnnxGeneratorPlan>> {
        if manifest.kind != ModelKind::Chat || loader.weight_format() != WeightFormat::Onnx {
            return Ok(None);
        }

        let onnx = loader.onnx()?;
        let onnx_graph = onnx.graph();

        // Use the new GraphOptimizer to generate FusedGraph
        let ctx = OptimizationContext {
            geometry: geometry.clone(),
            arch_family: manifest.family(),
            ..Default::default()
        };
        let optimizer = GraphOptimizer::new(ctx);
        let fused_graph = optimizer
            .optimize(onnx_graph)
            .map_err(|e| ExecutorError::OnnxPlan(format!("graph optimization failed: {e}")))?;

        let fused_kernels = OnnxFusedKernelStats::from_fused_graph(&fused_graph);
        let execution_order = fused_graph
            .nodes
            .iter()
            .map(|node| match &node.op {
                FusedOp::FlashAttention(_) => OnnxKernelExecutionOp::FlashAttention,
                FusedOp::SwiGLU(_) => OnnxKernelExecutionOp::SwiGlu,
                FusedOp::RoPE(_) => OnnxKernelExecutionOp::Rope,
                FusedOp::FusedQkvRope(_) => OnnxKernelExecutionOp::FusedQkvRope,
                FusedOp::FusedQkvNormRope(_) => OnnxKernelExecutionOp::FusedQkvRope,
                FusedOp::GQA(_) => OnnxKernelExecutionOp::Gqa,
                FusedOp::MoERouting(_) => OnnxKernelExecutionOp::MoERouting,
                FusedOp::FusedRMSLinear(_) => OnnxKernelExecutionOp::FusedRMSLinear,
                FusedOp::PerLayerEmbed(_) => OnnxKernelExecutionOp::PerLayerEmbed,
                FusedOp::Atomic(_) => OnnxKernelExecutionOp::Atomic,
            })
            .collect::<Vec<_>>();
        if execution_order.is_empty() {
            return Err(ExecutorError::OnnxPlan(
                "graph optimizer produced an empty execution plan".to_string(),
            ));
        }

        let graph_outputs = onnx_graph
            .outputs
            .iter()
            .map(|value| value.name.clone())
            .collect::<Vec<_>>();
        let kv_outputs = extract_onnx_kv_outputs(&graph_outputs);
        if kv_outputs.is_empty() {
            return Err(ExecutorError::OnnxPlan(
                "ONNX graph does not expose identifiable KV cache outputs; \
                 cannot proceed without KV cache (O(n²) fallback is not authorized)".into(),
            ));
        }

        Ok(Some(OnnxGeneratorPlan {
            fused_kernels,
            graph_outputs,
            kv_outputs,
            execution_order,
        }))
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn manifest(&self) -> &ModelManifest {
        self.manifest.as_ref()
    }

    pub fn weights(&self) -> &WeightsHandle<B, E> {
        &self.weights
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Override `model_config.multimodal_token_ids` at runtime.
    ///
    /// This is the single plumbing point for runtime multimodal registration
    /// and is used by (a) loaders that discover tokenizer-level multimodal
    /// IDs post-hoc and (b) integration tests that exercise the fusion path
    /// on models whose config does not originally advertise multimodal IDs
    /// (e.g. wrapping SmolLM2 with a mock vision encoder). Accepts `None`
    /// to clear a previously-set override.
    pub fn set_multimodal_token_ids(
        &mut self,
        ids: Option<crate::compat::multimodal::MultimodalTokenIds>,
    ) {
        self.model_config.multimodal_token_ids = ids;
    }

    /// §12.6 获取系统硬件拓扑
    pub fn system_topology(&self) -> &crate::sensors::SystemTopology {
        &self.system_topology
    }

    /// §12.6 获取 JIT 编译器约束变量
    pub fn compiler_constraints(&self) -> &crate::jit::compiler_constraints::CompilerConstraints {
        &self.system_topology.constraints
    }

    /// §18.1 获取遥测聚合器
    pub fn telemetry(&self) -> &crate::jit::epilogue::TelemetryAggregator {
        &self.telemetry_aggregator
    }

    /// Get forward configuration (per SPEC 04-API-DESIGN §7.3 for encode_intent).
    pub fn forward_config(&self) -> GeneratorForwardConfig {
        self.forward_config.clone()
    }

    /// Expose the loaded `TokenizerHandle` (read-only).
    ///
    /// Consumer: `Client::register_semantic_gatekeeper` uses it to encode
    /// SG level descriptor strings and runtime knowledge text into token ids
    /// (SPEC/SEMANTIC-GATEKEEPER.md §3.1).
    pub fn tokenizer(&self) -> &TokenizerHandle {
        &self.tokenizer
    }

    /// Add a generation hook (guardrail/probe).
    ///
    /// per SPEC 04-API-DESIGN §7.4 — hooks are called after each decode step
    /// and can veto tokens or terminate generation.
    pub fn add_hook(&self, hook: Box<dyn crate::generation::GenerationHook>) -> ExecutorResult<()> {
        let mut hooks = self.hooks.write()
            .map_err(|e| ExecutorError::Scheduler(format!("hooks lock poisoned: {e}")))?;
        hooks.push(hook);
        Ok(())
    }

    /// Remove all hooks with the given type name.
    pub fn remove_hooks_by_type(&self, type_name: &str) -> ExecutorResult<usize> {
        let mut hooks = self.hooks.write()
            .map_err(|e| ExecutorError::Scheduler(format!("hooks lock poisoned: {e}")))?;
        let original_len = hooks.len();
        hooks.retain(|h| std::any::type_name_of_val(&**h) != type_name);
        Ok(original_len - hooks.len())
    }

    /// Clear all generation hooks.
    pub fn clear_hooks(&self) -> ExecutorResult<()> {
        let mut hooks = self.hooks.write()
            .map_err(|e| ExecutorError::Scheduler(format!("hooks lock poisoned: {e}")))?;
        hooks.clear();
        Ok(())
    }

    /// Get the number of active hooks.
    pub fn hook_count(&self) -> usize {
        self.hooks.read()
            .map(|h| h.len())
            .unwrap_or(0) // LEGAL: 锁失败时返回 0（表示无 hooks）
    }

    pub fn allocate_kv_cache(&mut self, config: &KvCacheConfig) -> ExecutorResult<KvCacheHandle> {
        let front = self.backend.alloc_kv_cache(config)?;
        let back = self.backend.alloc_kv_cache(config)?;
        let front = KvCacheState::new(front, config.clone());
        let back = KvCacheState::new(back, config.clone());
        self.kv_cache = Some(KvCacheDoubleBuffer::new(front, back));
        self.kv_cache_slot = KvCacheSlot::Front;
        Ok(self
            .kv_cache
            .as_ref()
            .ok_or_else(|| ExecutorError::Config(ModelConfigError::InvalidConfig(
                "KV cache not initialized".to_string()
            )))?
            .front()
            .handle())
    }

    pub fn kv_cache(&self) -> Option<KvCacheHandle> {
        self.kv_cache
            .as_ref()
            .map(|cache| cache.slot(self.kv_cache_slot).handle())
    }

    pub fn enqueue(&mut self, _kind: RequestKind, prompt: impl Into<String>) -> ExecutorResult<RequestId> {
        let id = self.requests.len() as RequestId + 1;
        let prompt_str = prompt.into();
        let prompt_tokens = self.encode_prompt(&prompt_str)?;
        let sequence = Sequence::new(id, prompt_tokens.clone());

        let request_data = RequestData {
            prompt_tokens,
            output_tokens: Vec::new(),
            sampling_config: SamplingConfig::default(),
            is_prefill: true,
            max_new_tokens: 128, // Default
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };

        self.requests.insert(id, request_data);
        self.batcher.enqueue(sequence);
        Ok(id)
    }

    pub fn enqueue_with_config(
        &mut self,
        _kind: RequestKind,
        prompt: impl Into<String>,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        thinking_budget: Option<usize>,
    ) -> ExecutorResult<RequestId> {
        let id = self.requests.len() as RequestId + 1;
        let prompt_str = prompt.into();
        let prompt_tokens = self.encode_prompt(&prompt_str)?;
        let sequence = Sequence::new(id, prompt_tokens.clone());

        let request_data = RequestData {
            prompt_tokens,
            output_tokens: Vec::new(),
            sampling_config,
            is_prefill: true,
            max_new_tokens,
            finished: false,
            session_id: None,
            thinking_budget,
            fused_prefill_hidden: None,
        };

        self.requests.insert(id, request_data);
        self.batcher.enqueue(sequence);
        Ok(id)
    }


    /// Enqueue a multimodal generation request (ARCH-MULTIMODAL-FUSION).
    ///
    /// Unlike `enqueue_with_config` which tokenizes the prompt string, this
    /// accepts the *already-routed* token sequence (special tokens expanded
    /// into encoder virtual tokens) together with a pre-computed fused hidden
    /// state of shape `[token_ids.len() * hidden_size]`. The first prefill
    /// step seeds the graph's leading `Gather` output with that buffer and
    /// bypasses the standard embedding lookup.
    ///
    /// SPEC: 04-API-DESIGN §3.7, 02-ARCHITECTURE §ARCH-MULTIMODAL-FUSION.
    pub fn enqueue_with_multimodal(
        &mut self,
        _kind: RequestKind,
        token_ids: Vec<u32>,
        fused_hidden: Vec<f32>,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        thinking_budget: Option<usize>,
    ) -> ExecutorResult<RequestId> {
        if token_ids.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let hidden = self.geometry.hidden_size;
        let expected = token_ids.len() * hidden;
        if fused_hidden.len() != expected {
            return Err(ExecutorError::Config(ModelConfigError::InvalidConfig(
                format!(
                    "enqueue_with_multimodal: fused_hidden length {} != token_ids.len()*hidden_size ({}*{})",
                    fused_hidden.len(),
                    token_ids.len(),
                    hidden,
                ),
            )));
        }

        let id = self.requests.len() as RequestId + 1;
        let sequence = Sequence::new(id, token_ids.clone());

        let request_data = RequestData {
            prompt_tokens: token_ids,
            output_tokens: Vec::new(),
            sampling_config,
            is_prefill: true,
            max_new_tokens,
            finished: false,
            session_id: None,
            thinking_budget,
            fused_prefill_hidden: Some(fused_hidden),
        };

        self.requests.insert(id, request_data);
        self.batcher.enqueue(sequence);
        Ok(id)
    }

    /// Register a new session for multi-turn KV cache reuse.
    pub fn register_session(&mut self, session_id: SessionId) {
        self.memory_manager.register_session(session_id);
    }

    /// Attach a session to an existing request for KV cache prefix reuse.
    pub fn set_session_id(
        &mut self,
        request_id: RequestId,
        session_id: SessionId,
    ) -> ExecutorResult<()> {
        let req = self
            .requests
            .get_mut(&request_id)
            .ok_or(ExecutorError::RequestNotFound { request_id })?;
        req.session_id = Some(session_id);
        Ok(())
    }

    /// Enqueue a request with session affinity for multi-turn KV cache reuse.
    pub fn enqueue_with_session(
        &mut self,
        kind: RequestKind,
        prompt: impl Into<String>,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        session_id: SessionId,
        thinking_budget: Option<usize>,
    ) -> ExecutorResult<RequestId> {
        let req_id = self.enqueue_with_config(kind, prompt, max_new_tokens, sampling_config, thinking_budget)?;
        self.set_session_id(req_id, session_id)?;
        Ok(req_id)
    }

    pub fn next_batch(&mut self) -> Option<ScheduledBatch> {
        if !self.batcher.has_pending_work() {
            return None;
        }
        let batch = self.batcher.build_batch(
            &mut self.scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );

        Some(batch)
    }


    pub fn encode_prompt(&self, prompt: &str) -> ExecutorResult<Vec<u32>> {
        let add_special_tokens = self.add_special_tokens;
        Ok(self.tokenizer.encode_prompt(prompt, add_special_tokens)?)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> ExecutorResult<String> {
        Ok(self.tokenizer.decode(tokens, true)?)
    }

    pub fn sample_from_logits(
        &self,
        logits: &LogitsHandle,
        sampling: &SamplingConfig,
    ) -> ExecutorResult<u32> {
        let tokens = self.backend.sample_from_tensor(
            logits,
            &self.topology,
            self.geometry.vocab_size,
            sampling,
        )?;
        tokens.into_iter().next().ok_or(ExecutorError::EmptySample)
    }

    fn run_batch_forward(&mut self, batch_input: &BatchInput) -> ExecutorResult<(Vec<LogitsHandle>, f32, Vec<crate::scheduler::SequenceTelemetry>)> {
        // ARCH-RUST-IS-CODEGEN: mega-kernel 是唯一推理路径
        // generate_with_sampling() 已通过 generate_single_sequence() 处理完整生成循环
        // run_batch_forward() 仅在 step-by-step 连续批处理模式中被调用，
        // 该模式尚未迁移到 mega-kernel（需要 per-step KV cache 管理）

        // 标准路径：逐层执行
        if let Some(plan) = self.onnx_generator_plan.as_ref() {
            if plan.execution_order.is_empty() {
                return Err(ExecutorError::OnnxPlan(
                    "ONNX execution plan is empty".to_string(),
                ));
            }
            let total_kernels = plan.fused_kernels.fused_total() + plan.fused_kernels.atomic;
            if total_kernels != plan.execution_order.len() {
                return Err(ExecutorError::OnnxPlan(
                    "ONNX execution plan kernel accounting mismatch".to_string(),
                ));
            }
            let has_logits_output = plan
                .graph_outputs
                .iter()
                .any(|name| is_onnx_logits_output(name))
                || plan
                    .graph_outputs
                    .len()
                    .saturating_sub(plan.kv_outputs.len())
                    == 1;
            if !has_logits_output {
                return Err(ExecutorError::OnnxPlan(
                    "ONNX execution plan has no logits output".to_string(),
                ));
            }

            // Validate KV cache layer count matches ONNX graph KV outputs
            if !plan.kv_outputs.is_empty() {
                let onnx_layers = onnx_kv_layer_count(&plan.kv_outputs);
                if onnx_layers > 0 && onnx_layers != self.kv_cache_config.num_layers() {
                    log::warn!(
                        "ONNX KV output layers ({}) differs from model config layers ({}); \
                         KV cache managed by compat layer using model config",
                        onnx_layers,
                        self.kv_cache_config.num_layers()
                    );
                }
            }
        }

        let kv_handle = self.active_kv_handle()?;
        let mut kv_caches = vec![kv_handle; batch_input.sequences.len()];
        self.forward_config.graph_executor_ptr = self
            .graph_executor
            .as_mut()
            .map(|ge| ge as *mut _)
            .unwrap_or(std::ptr::null_mut()); // LEGAL: GPU 指针在 CPU 路径下为 null

        // §9-§18: 构建完整 callback chain 并通过 forward_config 传递给 decoder_forward
        let num_layers = self.geometry.num_layers;

        // §13.1 + §14.2 Gate-First Compaction callback
        // Per §14.2: dead neurons trigger register-level compaction, NOT skip.
        let gate_decisions: Vec<crate::engine::callbacks::gate_skip::SkipDecision> = {
            let dead_ratio = self.telemetry_aggregator.dead_neuron_ratio();
            (0..num_layers).map(|_| {
                if dead_ratio > 0.5 {
                    // §14.2: High dead ratio → CompactedCompute (register-level compaction)
                    crate::engine::callbacks::gate_skip::SkipDecision::CompactedCompute
                } else if dead_ratio > 0.3 {
                    crate::engine::callbacks::gate_skip::SkipDecision::MaskedCompute
                } else {
                    crate::engine::callbacks::gate_skip::SkipDecision::FullCompute
                }
            }).collect()
        };
        let gate_skip_cb = crate::engine::callbacks::gate_skip::GateSkipCallback::new(
            num_layers, gate_decisions, self.geometry.intermediate_size,
        );

        // §16.2 Early Exit callback
        let early_exit_cb = crate::engine::callbacks::early_exit::EarlyExitCallback::new(
            crate::early_exit::EarlyExitConfig::default(),
            num_layers,
        );

        let mut callbacks: Vec<Box<dyn crate::graph::layer_callback::LayerCallback + Send>> = Vec::new();

        // §9.3 Residual Bus Bridge callback (priority 95, pre_node + post_node)
        // Bridges all bus-based injection/recall operations into the node loop.
        let bus_bridge = crate::engine::callbacks::ResidualBusBridgeCallback::from_bus(
            &self.residual_bus,
        );
        callbacks.push(Box::new(bus_bridge));

        // §2.9 Semantic Gatekeeper callback (priority 90, pre_node)
        if let Some(ref shim) = self.sg_callback_shim {
            callbacks.push(Box::new(
                crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim {
                    inner: std::sync::Arc::clone(&shim.inner),
                    hidden_size: shim.hidden_size,
                },
            ));
        }

        // §16.1 RAG Inject callback (priority 80, pre_node)
        if let Some(rag) = self.rag_system.take() {
            callbacks.push(Box::new(
                crate::engine::callbacks::RagInjectCallback::new(rag),
            ));
        }

        // §15 MoE Dispatch callback (priority 70, pre_node)
        if let Some(moe_config) = self.forward_config.moe_config {
            let moe_cb = crate::engine::callbacks::moe_dispatch::MoeDispatchCallback::new(
                moe_config.num_experts,
                moe_config.num_experts_per_tok,
                num_layers,
                0, // moe_start_layer: 默认从第 0 层开始
            );
            callbacks.push(Box::new(moe_cb));
        }

        // §13.1 Gate-First Skip callback (priority 60, pre_node)
        callbacks.push(Box::new(gate_skip_cb));

        // §16.2 Early Exit callback (priority 50, post_node)
        callbacks.push(Box::new(early_exit_cb));

        let mut callback_chain = crate::graph::layer_callback::CallbackChain::new(callbacks);
        self.forward_config.callback_chain_ptr = &mut callback_chain as *mut _;

        let result = self.backend.batch_forward_gpu_pure(
            batch_input,
            &self.topology,
            &self.weights,
            &mut kv_caches,
            &self.forward_config,
        );

        // 清除指针（callback_chain 生命周期结束）
        self.forward_config.callback_chain_ptr = std::ptr::null_mut();

        Ok(result?)
    }

    /// Hot-swap the scheduling policy. Takes effect on the next `step()` call.
    pub fn set_policy(&mut self, policy: crate::scheduler::PolicyVariant) {
        self.policy = policy;
    }

    /// §16.1 Set the Late-Fusion RAG system for RagInjectCallback.
    pub fn set_rag_system(&mut self, rag: crate::rag::LateFusionRag) {
        self.rag_system = Some(rag);
    }

    /// §9 Set the Semantic Gatekeeper callback shim for injection into the
    /// per-forward CallbackChain.
    pub fn set_sg_callback_shim(
        &mut self,
        shim: crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim,
    ) {
        self.sg_callback_shim = Some(shim);
    }

    /// Remove the Semantic Gatekeeper callback shim.
    pub fn clear_sg_callback_shim(&mut self) {
        self.sg_callback_shim = None;
    }

    /// Main Engine Step: Continuous Batching
    ///
    /// 1. Schedule next batch (mix of prefill and decode)
    /// 2. Construct batched inputs (strict causal ordering)
    /// 3. Run backend forward
    /// 4. Sample and update request states
    pub fn step(&mut self) -> ExecutorResult<()> {
        self.check_memory_pressure()?;

        // 0. Observability: Capture System State
        let pressure_result = self.backend.get_memory_pressure()
            .map_err(|e| format!("{e}"));
        if let Err(e) = self.observer.update_memory_pressure(pressure_result) {
            log::warn!("executor: update_memory_pressure failed: {e}");
        }
        self.observer.update_kv_fragmentation(self.scheduler.kv_fragmentation_ratio());
        self.observer.update_scheduler_metrics(
            self.batcher.waiting_len(),
            self.batcher.running_len(),
            self.batcher.running_len(),
            self.batcher.mean_context_len(),
        );
        let system_state = self.observer.last_state;

        // JIT Decision: Decide Scheduling Strategy
        let decision = self.policy.decide(&system_state);

        // Schedule — §10.1 交织调度：Decode 优先 + Prefill Chunk 交织
        let interleaved = if !self.batcher.has_pending_work() {
            return Ok(());
        } else {
            self.batcher.build_interleaved_batch(
                &mut self.scheduler,
                decision.max_batch_size,
                decision.admit_new_prefill,
                BatchOrderPolicy::StrictRequestIdOrder,
            )
        };
        let batch = interleaved.inner.clone();

        if interleaved.is_interleaved() {
            log::debug!(
                "executor: interleaved batch — {} decode + {} prefill tokens",
                interleaved.decode_tokens(),
                interleaved.prefill_tokens(),
            );
        }

        // §17.9: 推测解码自适应决策
        let decode_count = interleaved.decode_slots.len();
        let spec_advice = self.spec_decoding.should_speculate(decode_count);
        let spec_enabled = matches!(spec_advice, crate::jit::epilogue::SpecScheduleAdvice::EnableSpec);
        match spec_advice {
            crate::jit::epilogue::SpecScheduleAdvice::EnableSpec => {
                log::debug!("executor: §17.9 speculative decoding ENABLED (acceptance_rate={:.2})",
                    self.spec_decoding.avg_acceptance_rate());
            }
            crate::jit::epilogue::SpecScheduleAdvice::Fallback => {
                log::debug!("executor: §17.9 speculative decoding FALLBACK (low acceptance streak)");
            }
            crate::jit::epilogue::SpecScheduleAdvice::StandardDecode => {}
        }

        if batch.requests.is_empty() {
            return Ok(());
        }

        let mut batch_results = Vec::with_capacity(batch.requests.len());
        let mut sequences = Vec::with_capacity(batch.requests.len());
        let mut request_indices = Vec::with_capacity(batch.requests.len());

        // 3. PrefillPlan + Session Prefix: plan memory for prefill requests
        let page_size = self.scheduler.page_size().max(1);
        // Adaptive chunk sizing (REQ-KV-EXT-001): use runtime load signals
        // instead of the static max_seq_len.
        let adaptive = AdaptiveChunkPolicy::new(
            &Scheduler2024Config::default().chunked,
        );
        let l1_usage = self.memory_manager.tier_usage(Tier::L1);
        let l1_ratio = if l1_usage.capacity > 0 {
            (l1_usage.capacity.saturating_sub(l1_usage.used)) as f32
                / l1_usage.capacity as f32
        } else {
            1.0
        };
        let concurrent = batch.requests.len();
        for &req_id in &batch.requests {
            let (is_prefill, prompt_len, session_id) = {
                let Some(req) = self.requests.get(&req_id) else {
                    continue;
                };
                (req.is_prefill, req.prompt_tokens.len(), req.session_id)
            };
            if !is_prefill {
                continue;
            }

            let chunk_size = adaptive.compute(l1_ratio, concurrent, prompt_len);
            // Plan prefill page allocation strategy and pre-reclaim if pipelined
            let plan = self
                .memory_manager
                .plan_prefill(prompt_len, chunk_size, page_size);
            if let PrefillPlan::Pipelined { l1_pages, .. } = plan {
                if l1_pages > 0 {
                    self.reclaim_memory(l1_pages)?;
                }
            }

            // If request has a session, try to claim cached prefix pages
            if let Some(sid) = session_id {
                let finalized = match self
                    .memory_manager
                    .session_finalized_position(sid)
                {
                    Some(pos) => pos,
                    None => {
                        return Err(ExecutorError::Scheduler(
                            format!("session_finalized_position returned None for session {sid}")
                        ));
                    }
                };
                let prefix_tokens = prompt_len.min(finalized);
                if prefix_tokens > 0 {
                    if let Err(e) = self
                        .memory_manager
                        .claim_session_prefix(sid, req_id, prefix_tokens) {
                        log::warn!("executor: claim_session_prefix failed for session {sid}: {e}");
                    }
                }
            }
        }

        // 4. Prepare Batch — §12.4 Golden Bucket SEQ 映射 + §12.1 Sub-Batch 分类
        let mut shape_map = std::collections::HashMap::new();
        for (_idx, req_id) in batch.requests.iter().enumerate() {
            let seq_len = self.requests.get(req_id)
                .map(|r| if r.is_prefill { r.prompt_tokens.len() } else { 1 })
                .unwrap_or(1);

            // §12.4: 将任意 SEQ 长度映射到黄金尺寸
            let (_golden_idx, golden_size) = self.golden_buckets.collapse(seq_len);
            let golden_seq = golden_size.seq_len;
            if golden_seq != seq_len {
                log::trace!("executor: §12.4 Golden Bucket: seq_len {} → {}", seq_len, golden_seq);
            }

            // §12.1: 根据 Epilogue 遥测对请求进行形状分类
            let dead_ratio = self.telemetry_aggregator.dead_neuron_ratio();
            let delta_rho = self.telemetry_aggregator.residual_delta_rho();
            let is_moe = self.forward_config.moe_config.is_some();
            let shape = self.sub_batch_dispatcher.classify_request(
                dead_ratio, delta_rho, is_moe, 0.0,
            );
            shape_map.insert(*req_id, shape);
        }

        // §12.1: Sub-Batch 分发决策 + §12.4 golden_seq 传递
        let dispatch_plan = if !shape_map.is_empty() {
            use crate::scheduler::chunked_prefill::{BatchManifest, BatchSlot, SlotType};

            // 构建 BatchManifest 供 dispatcher 消费
            let slots: Vec<BatchSlot> = batch.requests.iter().enumerate().map(|(i, &rid)| {
                let seq_len = self.requests.get(&rid)
                    .map(|r| if r.is_prefill { r.prompt_tokens.len() } else { 1 })
                    .unwrap_or(1);
                let (_golden_idx, golden_size) = self.golden_buckets.collapse(seq_len);
                let is_prefill = self.requests.get(&rid).map(|r| r.is_prefill).unwrap_or(false);
                BatchSlot {
                    request_id: rid,
                    slot_type: if is_prefill { SlotType::PrefillChunk } else { SlotType::Decode },
                    token_start: 0,
                    token_end: golden_size.seq_len, // §12.4: 使用黄金尺寸而非原始 seq_len
                    compact_target: i as i32,
                }
            }).collect();

            let total_tokens: usize = slots.iter().map(|s| s.token_end - s.token_start).sum();
            let decode_tokens: usize = slots.iter().filter(|s| matches!(s.slot_type, SlotType::Decode)).map(|s| s.token_end - s.token_start).sum();
            let prefill_tokens = total_tokens - decode_tokens;

            let batch_capacity = batch.requests.len();
            let active_count = slots.len();
            let waste_ratio = if batch_capacity > 0 {
                (batch_capacity.saturating_sub(active_count)) as f32 / batch_capacity as f32
            } else {
                0.0
            };

            let manifest = BatchManifest {
                slots,
                total_tokens,
                decode_tokens,
                prefill_tokens,
                compact_required: waste_ratio > 0.25,
                waste_ratio,
            };

            // §12.1: 实际分发
            let plan = self.sub_batch_dispatcher.dispatch(&manifest, &shape_map);
            if plan.sub_batches.len() > 1 {
                log::info!(
                    "executor: §12.1 Sub-Batch dispatched {} sub-batches ({} orphans, reason={:?})",
                    plan.sub_batches.len(), plan.orphan_count, plan.reason,
                );
            }
            Some(plan)
        } else {
            None
        };

        for (idx, req_id) in batch.requests.into_iter().enumerate() {
            let current_draft_steps = batch.draft_steps.get(idx).copied().unwrap_or(0); // LEGAL: draft_steps=0 表示无 draft tokens
            self.ensure_pages_resident(req_id)?;

            let (tokens, position, fused_hidden) = {
                let Some(req) = self.requests.get_mut(&req_id) else {
                    // Request was removed, skip it
                    continue;
                };
                if req.finished {
                    continue;
                }

                let tokens = if req.is_prefill {
                    req.prompt_tokens.clone()
                } else {
                    req.output_tokens
                        .last()
                        .map(|t| vec![*t])
                        .unwrap_or_default() // LEGAL: 空 output_tokens 时返回空 Vec，语义正确
                };

                let position = if req.is_prefill {
                    0
                } else {
                    req.prompt_tokens.len() + req.output_tokens.len().saturating_sub(1)
                };

                // ARCH-MULTIMODAL-FUSION: consume the pre-computed fused hidden
                // state once on prefill. Decode steps re-enter with None so
                // subsequent generated tokens go through the standard Gather
                // path. `take()` guarantees idempotence (no double-consume).
                let fused = if req.is_prefill {
                    req.fused_prefill_hidden.take()
                } else {
                    None
                };
                (tokens, position, fused)
            };

            if tokens.is_empty() {
                if let Some(req) = self.requests.get_mut(&req_id) {
                    req.finished = true;
                }
                batch_results.push(BatchResult::fail(req_id));
                continue;
            }

            sequences.push(SequenceInput {
                tokens,
                position,
                draft_steps: current_draft_steps,
                fused_hidden,
            });
            request_indices.push(req_id);
        }

        if sequences.is_empty() {
            return Ok(());
        }

        let batch_input = BatchInput { sequences };

        // §15.3: MoE 硬件分发（在 forward 之前决定专家→硬件映射）
        if let Some(ref dispatcher) = self.moe_dispatcher {
            if let Some(ref thermal) = self.moe_thermal {
                let heat_levels: Vec<crate::moe::thermal::ExpertHeatLevel> = (0..dispatcher.config().num_experts)
                    .map(|i| thermal.state(i).map(|s| s.heat_level).unwrap_or(crate::moe::thermal::ExpertHeatLevel::Warm))
                    .collect();
                log::trace!(
                    "executor: §15.3 MoE dispatch ready ({} experts, {} hot)",
                    heat_levels.len(),
                    heat_levels.iter().filter(|h| matches!(h, crate::moe::thermal::ExpertHeatLevel::Hot)).count(),
                );
            }
        }

        // §12.4: 记录 SEQ 长度到直方图（供 JIT Director 运行时演化）
        for seq in &batch_input.sequences {
            let seq_len = seq.tokens.len();
            self.golden_buckets.collapse(seq_len);
            self.seq_histogram.record(seq_len);
        }

        // §10: 更新 ChunkedPrefillScheduler 的运行时状态
        self.chunked_prefill_scheduler.update_l1_ratio(l1_ratio);
        self.chunked_prefill_scheduler.update_concurrency(concurrent);

        // §10.6.3: Compact 决策 — 在 GEMM ops 上评估是否需要 compact
        if let Some(ref plan) = dispatch_plan {
            use crate::scheduler::chunked_prefill::{BatchManifest, BatchSlot, SlotType};
            // Rebuild manifest from dispatch plan for compact evaluation
            let manifest_slots: Vec<BatchSlot> = plan.sub_batches.iter()
                .flat_map(|sb| sb.request_ids.iter().enumerate().map(|(i, &rid)| {
                    let seq_len = self.requests.get(&rid)
                        .map(|r| if r.is_prefill { r.prompt_tokens.len() } else { 1 })
                        .unwrap_or(1);
                    BatchSlot {
                        request_id: rid,
                        slot_type: if seq_len > 1 { SlotType::PrefillChunk } else { SlotType::Decode },
                        token_start: 0,
                        token_end: seq_len,
                        compact_target: i as i32,
                    }
                }))
                .collect();
            let total_tokens: usize = manifest_slots.iter().map(|s| s.token_end - s.token_start).sum();
            let decode_tokens: usize = manifest_slots.iter()
                .filter(|s| matches!(s.slot_type, SlotType::Decode))
                .map(|s| s.token_end - s.token_start).sum();
            let waste = if total_tokens > 0 {
                (batch_input.sequences.len().saturating_sub(manifest_slots.len())) as f32
                    / batch_input.sequences.len().max(1) as f32
            } else {
                0.0
            };
            let compact_manifest = BatchManifest {
                slots: manifest_slots,
                total_tokens,
                decode_tokens,
                prefill_tokens: total_tokens.saturating_sub(decode_tokens),
                compact_required: waste > 0.25,
                waste_ratio: waste,
            };

            let compact_config = crate::scheduler::compact::CompactConfig::default();
            let compact_decision = crate::scheduler::compact::evaluate_compact(
                &compact_manifest,
                crate::scheduler::compact::OpKind::Gemm,
                &compact_config,
            );

            if compact_decision.should_compact {
                log::info!(
                    "executor: §10.6.3 Compact decision TRIGGERED — waste={:.1}%, active={}/{}, reason={:?}",
                    compact_decision.waste_ratio * 100.0,
                    compact_decision.active_count,
                    compact_decision.total_count,
                    compact_decision.reason,
                );
            }

            // §12.2: Apply RaggedCompaction when dispatch plan indicates need
            if plan.needs_ragged_compaction {
                let batch_size = batch_input.sequences.len();
                // Build active mask from dispatch plan sub-batches
                let active_flags = vec![true; batch_size];
                // All requests in the dispatch plan are active; orphans that were
                // merged are already included in sub_batches. The mask is used for
                // future per-sub-batch execution where inactive slots would be
                // requests that were dropped/finished mid-batch.
                let mask = crate::jit::ragged::RequestActiveMask::new(active_flags);
                if self.ragged_compaction.should_compact(&mask) {
                    log::debug!(
                        "executor: §12.2 RaggedCompaction active — waste={:.1}%, batch_size={}",
                        mask.waste_ratio() * 100.0,
                        batch_size,
                    );
                }
            }
        }

        // 4. Run Backend Forward
        let (logits_list, batch_sparsity, batch_telemetry) = self.run_batch_forward(&batch_input)?;

        // §11 TurboQuant: 记录 per-channel scales 供下一步 KV 量化使用
        if self.turboquant.is_enabled() {
            let kv_dim = self.kv_cache_config.num_heads() * self.geometry.head_dim;
            // §11.2: 存储 per-channel scales（从 Epilogue 遥测白嫖）
            let per_ch_scale = self.telemetry_aggregator.per_channel_scale();
            if per_ch_scale > 0.0 {
                for layer in 0..self.kv_cache_config.num_layers() {
                    let scales = vec![per_ch_scale; kv_dim];
                    self.turboquant.store_k_scales(layer, scales);
                }
            }
            // §11.3: 存储 RaBitQ 修正因子（从 Embedding 范数白嫖）
            let embed_norm = self.telemetry_aggregator.embedding_norm();
            if embed_norm > 0.0 {
                for layer in 0..self.kv_cache_config.num_layers() {
                    self.turboquant.store_correction(
                        layer,
                        crate::kv_cache::quant::RabitqCorrection {
                            c0: 0.0,
                            c1: 1.0 / (1.0 + 1.0 / (embed_norm * embed_norm).sqrt()),
                            v_norm: embed_norm,
                        },
                    );
                }
            }
            log::trace!(
                "executor: §11 TurboQuant active (bits={}, fwht={}, scales_stored={})",
                self.turboquant.bits(),
                self.turboquant.fwht_enabled(),
                self.turboquant.get_k_scales(0).is_some(),
            );
        }

        // 5. Process Results
        // Note: batch_forward_gpu_pure must return results in the same order as input sequences
        if logits_list.len() != request_indices.len() || batch_telemetry.len() != request_indices.len() {
            return Err(ExecutorError::Backend(BackendError::Other(format!("Backend returned {} logits and {} telemetries for {} requests", logits_list.len(), batch_telemetry.len(), request_indices.len()))));
        }

        let mut total_tokens = 0;
        for seq in &batch_input.sequences {
            total_tokens += seq.tokens.len();
        }

        // Observer Phase 2: compute batch-average logits entropy (SPEC 07-OBSERVABILITY §2.1)
        let batch_entropy = {
            let mut total = 0.0f32;
            for logits in &logits_list {
                total += shannon_entropy(&logits.data);
            }
            if !logits_list.is_empty() {
                total / logits_list.len() as f32
            } else {
                0.0
            }
        };
        self.observer.update_logits_entropy(batch_entropy);
        self.observer.update_attention_sparsity(batch_sparsity);

        // §9.2 + §9.5: Push telemetry to JIT Director Daemon
        if let Some(ref director) = self.jit_director {
            let shared = director.shared();
            shared.advance_step();
            // Feed expert hit data to Director (§9.2: expert thermal → consensus detection)
            if let Some(ref thermal) = self.moe_thermal {
                for expert_idx in 0..self.geometry.num_experts {
                    if let Some(state) = thermal.state(expert_idx) {
                        if state.hit_count > 0 {
                            shared.record_expert_hit(expert_idx);
                        }
                    }
                }
            }
            // Feed batch-level telemetry to Director via synthetic KvPageHeaders
            for tel in &batch_telemetry {
                let header = crate::kv_cache::KvPageHeader {
                    page_id: 0,
                    ref_count: 1,
                    fragmentation_metric: 0.0,
                    logits_entropy: tel.output_entropy,
                    guard_veto_flag: 0,
                    softmax_max: 0.0,
                    softmax_sharpness: 0.0,
                    residual_delta_rho: tel.transform_ratio,
                    dead_neuron_ratio: 0.0,
                    per_channel_scale: 0.0,
                };
                self.telemetry_aggregator.ingest_from_page_header(&header);
            }
            // Drain consensus events from Director → drive MoE thermal + Hot JMP Patching
            for event in shared.drain_events() {
                match &event {
                    crate::jit::director::ConsensusEvent::ExpertFrozen { expert_idx, zero_hit_steps } => {
                        log::info!("executor: JIT Director detected frozen expert {} (zero hits for {} steps)", expert_idx, zero_hit_steps);
                        // §15.4: 触发冷专家封杀
                        if let Some(ref mut thermal) = self.moe_thermal {
                            if thermal.evict_expert(*expert_idx) {
                                log::info!("executor: Expert {} evicted via thermal manager", expert_idx);
                            }
                        }
                        // §14.4: 生成并执行 Hot JMP Patch（NOP/DeoptJump）
                        if let (Some(ref mut patch_mgr), Some(ref thermal)) =
                            (&mut self.hot_patch_manager, &self.moe_thermal)
                        {
                            let active_requests = self.requests.len();
                            let instructions = patch_mgr.generate_expert_patch_instructions(
                                thermal,
                                self.geometry.num_layers,
                                active_requests,
                            );
                            for instr in &instructions {
                                let result = patch_mgr.apply_patch(instr);
                                if result.success {
                                    log::info!(
                                        "executor: §14.4 Hot JMP Patch applied: {:?} → {:?}",
                                        instr.target, instr.operation,
                                    );
                                }
                            }
                        }
                    }
                    crate::jit::director::ConsensusEvent::AttentionSilent { avg_entropy, duration_steps } => {
                        log::warn!("executor: JIT Director detected attention silence (entropy={:.4}, duration={})", avg_entropy, duration_steps);
                    }
                    crate::jit::director::ConsensusEvent::LayerRedundant { avg_delta_rho, duration_steps } => {
                        log::info!("executor: JIT Director detected redundant layer (delta_rho={:.6}, duration={})", avg_delta_rho, duration_steps);
                    }
                }
            }
        }

        let page_size = self.scheduler.page_size().max(1);
        let mut page_entropies = std::collections::HashMap::new();

        // §9-§18: Epilogue 优化子系统 — 从遥测驱动全链路决策
        let epilogue_summary = self.epilogue_subsystem.ingest_and_decide(&batch_telemetry);
        if epilogue_summary.compact_required {
            log::debug!("executor: Epilogue compact required (waste={:.2}%)", epilogue_summary.waste_ratio * 100.0);
        }

        // §13.1: 将 Epilogue gate_skip 决策写入 forward_config，
        // 供下一步 run_batch_forward() 的 graph_executor callback chain 消费。
        // 每个请求的 gate_skip 决策映射到层级 skip_flags。
        {
            use crate::jit::epilogue::GateSkipDecision;
            let mut skip_layer_count = 0usize;
            let mut bypass_layer_count = 0usize;
            for decision in &epilogue_summary.per_request {
                match decision.gate_skip {
                    GateSkipDecision::Skip => skip_layer_count += 1,
                    GateSkipDecision::MaskedCompute => {}
                    GateSkipDecision::FullCompute => {}
                }
                match decision.bypass_decision {
                    crate::jit::epilogue::ResidualBypassDecision::Bypass => bypass_layer_count += 1,
                    _ => {}
                }
            }
            if skip_layer_count > 0 {
                log::debug!(
                    "executor: §13.1 Gate-First Skip active for {}/{} requests",
                    skip_layer_count, epilogue_summary.per_request.len(),
                );
            }
            if bypass_layer_count > 0 {
                log::debug!(
                    "executor: §13.3 Residual Bypass active for {}/{} requests",
                    bypass_layer_count, epilogue_summary.per_request.len(),
                );
            }

            // §13.2: Centroid Prefetch — 触发异步预取
            for decision in &epilogue_summary.per_request {
                match &decision.prefetch_advice {
                    crate::jit::prefetch::PrefetchAdvice::Forward(distance) => {
                        log::trace!("executor: §13.2 Centroid Prefetch forward {} tokens", distance);
                    }
                    crate::jit::prefetch::PrefetchAdvice::Backward(distance) => {
                        log::trace!("executor: §13.2 Centroid Prefetch backward {} tokens", distance);
                    }
                    crate::jit::prefetch::PrefetchAdvice::Sink(count) => {
                        log::trace!("executor: §13.2 Sink Prefetch {} tokens", count);
                    }
                    crate::jit::prefetch::PrefetchAdvice::None => {}
                }
            }

            // §17.9: 推测解码建议 — 调整 draft_budget
            for (i, decision) in epilogue_summary.per_request.iter().enumerate() {
                if let Some(&req_id) = request_indices.get(i) {
                    match decision.spec_advice {
                        crate::jit::epilogue::SpecScheduleAdvice::EnableSpec => {
                            if let Some(seq) = self.batcher.get_running_mut(req_id) {
                                seq.draft_budget = 8;
                            }
                        }
                        crate::jit::epilogue::SpecScheduleAdvice::Fallback => {
                            if let Some(seq) = self.batcher.get_running_mut(req_id) {
                                seq.draft_budget = 0;
                            }
                        }
                        crate::jit::epilogue::SpecScheduleAdvice::StandardDecode => {}
                    }
                }
            }
        }

        // Processing results loop
        for (i, logits) in logits_list.iter().enumerate() {
            let req_id = request_indices[i];
            let req_telemetry = batch_telemetry[i];
            
            // Tier V.3 Profile-Guided Re-Fusion
            if self.profile_accumulator.record_and_check(0, req_telemetry.transform_ratio) {
                log::info!("executor: ProfileAccumulator triggered Re-Fusion due to high stability.");

            }
            
            // SPEC §12.9.1: Tie telemetry entropy to the current physical page
            let logical_index = batch_input.sequences[i].position / page_size;
            let vpid = crate::scheduler::VirtualPageId::new(req_id, logical_index);
            if let Ok((tier, physical_id)) = self.memory_manager.resolve(vpid) {
                if tier == crate::scheduler::Tier::L1 {
                    page_entropies.insert(physical_id, req_telemetry.output_entropy);
                }
            }

            let sampling_config = self
                .requests
                .get(&req_id)
                .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?
                .sampling_config;
            let next_token = self.sample_from_logits(logits, &sampling_config)?;
            if std::env::var("GLLM_DEBUG_SAMPLING").is_ok() {
                // 诊断 probe: logits.len / argmax / top-3 / mean / var, 方便观察数值退化。
                let l = &logits.data;
                let mut top: [(u32, f32); 3] = [(0, f32::NEG_INFINITY); 3];
                for (i, &v) in l.iter().enumerate() {
                    if v > top[0].1 {
                        top[2] = top[1]; top[1] = top[0]; top[0] = (i as u32, v);
                    } else if v > top[1].1 {
                        top[2] = top[1]; top[1] = (i as u32, v);
                    } else if v > top[2].1 {
                        top[2] = (i as u32, v);
                    }
                }
                let mean: f32 = l.iter().sum::<f32>() / l.len() as f32;
                let var: f32 = l.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / l.len() as f32;
                eprintln!("[SAMPLE req={req_id} logits.len={} next={next_token} \
                           top3=[{},{}],[{},{}],[{},{}] mean={:.4} var={:.4}]",
                    l.len(), top[0].0, top[0].1, top[1].0, top[1].1, top[2].0, top[2].1, mean, var);
            }

            // Generation hooks (guardrails, probes) — per SPEC 04-API-DESIGN §7.4
            // Hooks can veto the current token or terminate generation.
            let hooks_guard = self.hooks.read();
            let hooks_decision = if let Ok(hooks) = &hooks_guard {
                let mut decision = crate::generation::HookDecision::Continue;
                // Get current generated tokens for this request
                let generated_tokens = self
                    .requests
                    .get(&req_id)
                    .map(|req| req.output_tokens.clone())
                    .unwrap_or_default();
                for hook in hooks.iter() {
                    match hook.post_step(&logits.data, &generated_tokens) {
                        crate::generation::HookDecision::Continue => continue,
                        crate::generation::HookDecision::Veto(reason) => {
                            log::debug!("executor: hook vetoed token {} for request {}: {}", next_token, req_id, reason);
                            decision = crate::generation::HookDecision::Veto(reason);
                            break;
                        }
                        crate::generation::HookDecision::Terminate => {
                            log::debug!("executor: hook terminated generation for request {}", req_id);
                            decision = crate::generation::HookDecision::Terminate;
                            break;
                        }
                    }
                }
                decision
            } else {
                crate::generation::HookDecision::Continue
            };
            drop(hooks_guard);

            // Handle hook decision
            match hooks_decision {
                crate::generation::HookDecision::Continue => {
                    // Accept token, continue generation
                }
                crate::generation::HookDecision::Veto(_) => {
                    // Veto: skip this token and mark request as finished
                    // (In a full implementation, we would re-sample; for now, we just terminate)
                    let req = self
                        .requests
                        .get_mut(&req_id)
                        .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
                    req.finished = true;
                    batch_results.push(BatchResult::complete(req_id, None, req_telemetry));
                    continue;
                }
                crate::generation::HookDecision::Terminate => {
                    // Terminate generation immediately
                    let req = self
                        .requests
                        .get_mut(&req_id)
                        .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
                    req.finished = true;
                    batch_results.push(BatchResult::complete(req_id, None, req_telemetry));
                    continue;
                }
            }

            // Update request
            let req = self
                .requests
                .get_mut(&req_id)
                .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
            req.output_tokens.push(next_token);
            req.is_prefill = false;

            // Check finish
            let eos_token = self.model_config.eos_token_id;
            let mut request_finished = false;
            if eos_token.is_some_and(|id| id == next_token)
                || req.output_tokens.len() >= req.max_new_tokens
            {
                req.finished = true;
                request_finished = true;
            }

            if request_finished {
                batch_results.push(BatchResult::complete(req_id, Some(next_token), req_telemetry));
            } else {
                batch_results.push(BatchResult::continue_with_token(req_id, next_token, req_telemetry));
            }
        }

        // §17.1-§17.9: Speculative Decoding Draft→Verify Pipeline (PLD-based)
        //
        // When spec_advice == EnableSpec, run PLD (Prompt Lookup Decoding) to draft
        // additional tokens via n-gram matching, then verify them with a batched
        // forward pass through the full model. Accepted tokens are appended directly
        // to request output, yielding multiple tokens per step.
        let mut spec_extra_tokens = 0usize;
        if spec_enabled && self.spec_decoding.is_active() {
            // Collect decode requests that are still running (got a Continue result)
            let continuing_decode_reqs: Vec<RequestId> = batch_results
                .iter()
                .filter(|r| r.action == BatchAction::Continue)
                .map(|r| r.request_id)
                .collect();

            if !continuing_decode_reqs.is_empty() {
                // Phase A: Draft via PLD for each continuing decode request.
                // We pick the first continuing request's logits to extract top-k
                // for the shared SpecTree (EqSpec I1: all sequences share same topology).
                let first_req_idx = request_indices
                    .iter()
                    .position(|&rid| rid == continuing_decode_reqs[0]);
                if let Some(logits_idx) = first_req_idx {
                    let top_k_tokens = extract_top_k_token_ids(&logits_list[logits_idx].data, 3);

                    // Gather full token context (prompt + output) from the first request
                    let all_tokens: Vec<u32> = {
                        let req = &self.requests[&continuing_decode_reqs[0]];
                        let mut tokens = req.prompt_tokens.clone();
                        tokens.extend_from_slice(&req.output_tokens);
                        tokens
                    };

                    // Build spec tree using n-gram PLD
                    let tree = self.spec_decoding.draft_phase(&top_k_tokens, &all_tokens);
                    let spine_tokens = tree.spine_token_ids();

                    if spine_tokens.len() > 1 {
                        // Phase B: Verify — run full model forward with draft tokens batched.
                        // Build one verify sequence per continuing request, each containing
                        // the spine draft tokens at successive positions after the last token.
                        let mut verify_sequences = Vec::with_capacity(continuing_decode_reqs.len());
                        let mut verify_req_indices = Vec::with_capacity(continuing_decode_reqs.len());
                        for &req_id in &continuing_decode_reqs {
                            let req = &self.requests[&req_id];
                            if req.finished {
                                continue;
                            }
                            let position = req.prompt_tokens.len() + req.output_tokens.len();
                            verify_sequences.push(SequenceInput {
                                tokens: spine_tokens.clone(),
                                position,
                                draft_steps: spine_tokens.len(),
                                fused_hidden: None,
                            });
                            verify_req_indices.push(req_id);
                        }

                        if !verify_sequences.is_empty() {
                            let verify_input = BatchInput {
                                sequences: verify_sequences,
                            };

                            match self.run_batch_forward(&verify_input) {
                                Ok((verify_logits, _verify_sparsity, _verify_telemetry)) => {
                                    // For each verified request, compare greedy argmax from each
                                    // verify position against the draft spine tokens.
                                    let mut seq_results = Vec::with_capacity(verify_req_indices.len());

                                    for (vi, &req_id) in verify_req_indices.iter().enumerate() {
                                        // The verify forward returns logits for each spine token position.
                                        // We take the argmax of each position's logits as the target token.
                                        let target_tokens: Vec<u32> = if vi < verify_logits.len() {
                                            // Single logits per sequence: the model predicts what follows
                                            // the last spine token. We compare the prefix.
                                            vec![argmax_token(&verify_logits[vi].data)]
                                        } else {
                                            Vec::new()
                                        };

                                        let seq_result = crate::speculative::verify::SequenceVerifyResult::verify_spine(
                                            req_id,
                                            &spine_tokens,
                                            &target_tokens,
                                        );

                                        // Append accepted tokens to request output
                                        if seq_result.accepted_count > 0 {
                                            if let Some(req) = self.requests.get_mut(&req_id) {
                                                let eos = self.model_config.eos_token_id;
                                                for &tok in &seq_result.accepted_tokens {
                                                    if req.finished {
                                                        break;
                                                    }
                                                    req.output_tokens.push(tok);
                                                    spec_extra_tokens += 1;
                                                    if eos.is_some_and(|id| id == tok)
                                                        || req.output_tokens.len() >= req.max_new_tokens
                                                    {
                                                        req.finished = true;
                                                    }
                                                }
                                            }
                                        }

                                        seq_results.push(seq_result);
                                    }

                                    // Phase C: Update speculative decoding state with verify results
                                    let verify_result = crate::speculative::verify::VerifyResult::from_sequence_results(seq_results);

                                    // Generate KV commit/rollback instructions (§17.4 I3)
                                    let kv_instructions = crate::speculative::verify::generate_kv_commit_instructions(&verify_result);
                                    for instr in &kv_instructions {
                                        match instr {
                                            crate::speculative::verify::KvCommitInstruction::Commit { request_id, accepted_tokens, .. } => {
                                                log::debug!(
                                                    "executor: §17.4 spec KV commit req={} accepted={} tokens",
                                                    request_id, accepted_tokens.len(),
                                                );
                                            }
                                            crate::speculative::verify::KvCommitInstruction::Rollback { request_id, rejected_count, .. } => {
                                                log::debug!(
                                                    "executor: §17.4 spec KV rollback req={} rejected={} tokens",
                                                    request_id, rejected_count,
                                                );
                                            }
                                        }
                                    }

                                    self.spec_decoding.verify_phase(&verify_result);

                                    log::debug!(
                                        "executor: §17.1 spec decode complete — drafted={}, accepted={}, rate={:.2}",
                                        verify_result.total_draft_tokens,
                                        verify_result.total_accepted_tokens,
                                        verify_result.avg_acceptance_rate,
                                    );
                                }
                                Err(e) => {
                                    log::warn!("executor: §17.1 spec verify forward failed: {}", e);
                                    // On verify failure, we gracefully degrade — the standard decode
                                    // token was already accepted above, so we lose nothing.
                                }
                            }
                        }
                    }
                }
            }
        }

        // Advance KV cache
        {
            let slot = self.kv_cache_slot;
            if let Some(kv_cache) = self.kv_cache.as_mut() {
                let active = kv_cache.slot_mut(slot);
                active.advance(total_tokens + spec_extra_tokens)?;
            }
        }

        // 7. Finalize sessions for completed requests
        for result in &batch_results {
            if !matches!(result.action, BatchAction::Complete | BatchAction::Fail) {
                continue;
            }
            let request_id = result.request_id;
            if let Some(req) = self.requests.get(&request_id) {
                if let Some(sid) = req.session_id {
                    let total_processed = req.prompt_tokens.len() + req.output_tokens.len();
                    self.memory_manager
                        .finalize_session_tokens(sid, total_processed);
                }
            }
            self.release_request_pages(request_id);
        }

        self.batcher
            .update_batch(&mut self.scheduler, batch_results.as_slice());

        // SPEC §12.9.1: CPU-side KV page eviction after each step.
        // page_entropies populated by telemetry pipeline; evict pages with output_entropy < threshold
        let evicted = self.memory_manager.entropy_evict(&page_entropies, 0.1, crate::scheduler::Tier::L1);
        if evicted > 0 {
            log::debug!("entropy_evict: freed {evicted} low-entropy KV pages synchronously after batch step");
        }

        Ok(())
    }

    /// Legacy method for compatibility with tests (e.g. test_alignment)
    /// Wraps the batch API for a single request.
    pub fn forward_step(&mut self, tokens: &[u32]) -> ExecutorResult<LogitsHandle> {
        let seq = SequenceInput {
            tokens: tokens.to_vec(),
            position: 0,
            draft_steps: 0,
            fused_hidden: None,
        };

        let batch_input = BatchInput {
            sequences: vec![seq],
        };

        let mut kv_cache = self.active_kv_handle()?;

        self.forward_config.graph_executor_ptr = self
            .graph_executor
            .as_mut()
            .map(|ge| ge as *mut _)
            .unwrap_or(std::ptr::null_mut()); // LEGAL: GPU 指针在 CPU 路径下为 null
        // Profiling/test execution (run pure forward manually without inserting into metrics stream)
        let (logits_list, _sparsity, _telemetries) = self.backend.batch_forward_gpu_pure(
            &batch_input,
            &self.topology,
            &self.weights,
            std::slice::from_mut(&mut kv_cache),
            &self.forward_config,
        )?;

        // Maintain legacy KV cache state advancement
        if let Some(kv_cache) = self.kv_cache.as_mut() {
            let active = kv_cache.slot_mut(self.kv_cache_slot);
            active.advance(tokens.len())?;
        }

        logits_list
            .into_iter()
            .next()
            .ok_or(ExecutorError::EmptySample)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> ExecutorResult<String> {
        self.generate_with_sampling(prompt, max_tokens, temperature, 0, 1.0, None)
    }

    pub fn generate_with_sampling(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        thinking_budget: Option<usize>,
    ) -> ExecutorResult<String> {
        if prompt.trim().is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(ref mega) = self.mega_kernel {
            let prompt_tokens = self.encode_prompt(&prompt.to_string())?;
            log::debug!("[executor] mega-kernel path: prompt_tokens={:?}, max_tokens={}", prompt_tokens, max_tokens);
            let output_tokens = mega.generate_single_sequence(
                &prompt_tokens,
                max_tokens,
                temperature,
                top_k,
                top_p,
            ).map_err(|e| ExecutorError::Backend(BackendError::Other(
                format!("mega-kernel generate failed: {}", e),
            )))?;
            log::debug!("[executor] mega-kernel output_tokens={:?}", output_tokens);
            let text = self.decode_tokens(&output_tokens)?;
            return Ok(text);
        }

        // Per-node JIT path: MoE models / heterogeneous layers use FusedGraphExecutor
        // with token-by-token generation. Each forward pass is still JIT-compiled
        // machine code — only the autoregressive loop is Rust-orchestrated.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        if self.graph_executor.is_some() {
            return self.generate_per_node(prompt, max_tokens, temperature, top_k, top_p);
        }

        Err(ExecutorError::Backend(BackendError::Other(
            "generate_with_sampling: no execution backend available. \
             mega-kernel not compiled and graph_executor not available.".into(),
        )))
    }

    /// Per-node JIT generation: enqueue + step loop for MoE / heterogeneous-layer models.
    /// Each forward pass runs through JIT-compiled FusedGraphExecutor.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn generate_per_node(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> ExecutorResult<String> {
        // Ensure KV cache is allocated for this model.
        if self.kv_cache.is_none() {
            let kv_config = self.kv_cache_config.clone();
            self.allocate_kv_cache(&kv_config)?;
        }

        let sampling = SamplingConfig { temperature, top_k, top_p };
        let req_id = self.enqueue_with_config(
            crate::scheduler::types::RequestKind::Chat,
            prompt.to_string(),
            max_tokens,
            sampling,
            None,
        )?;

        // Step loop: each call processes one forward + sample for all active requests.
        let max_steps = max_tokens + 1; // prefill + max_tokens decode
        for _ in 0..max_steps {
            if !self.batcher.has_pending_work() {
                break;
            }
            self.step()?;

            // Check if our request finished.
            if let Some(req) = self.requests.get(&req_id) {
                if req.finished {
                    let output_tokens = req.output_tokens.clone();
                    let all_tokens: Vec<u32> = req.prompt_tokens.iter()
                        .chain(&output_tokens).copied().collect();
                    let text = self.decode_tokens(&all_tokens)?;
                    self.requests.remove(&req_id);
                    return Ok(text);
                }
            }
        }

        // Timeout: collect whatever we have.
        let req = self.requests.remove(&req_id)
            .ok_or_else(|| ExecutorError::Backend(BackendError::Other(
                "generate_per_node: request vanished during generation".into(),
            )))?;
        let all_tokens: Vec<u32> = req.prompt_tokens.iter()
            .chain(&req.output_tokens).copied().collect();
        let text = self.decode_tokens(&all_tokens)?;
        Ok(text)
    }

    /// Generate with multimodal-routed inputs (ARCH-MULTIMODAL-FUSION).
    ///
    /// Takes the pre-expanded token sequence and pre-computed fused hidden
    /// state (text positions: gathered from `embed_tokens`; media positions:
    /// encoder output).
    ///
    /// ARCH-RUST-IS-CODEGEN: 需要支持 fused hidden state 注入的 mega-kernel 路径。
    /// 当前 mega-kernel 仅支持纯文本 generate_single_sequence。
    /// TODO: mega-kernel multimodal 支持 (fused hidden state → JIT 注入)
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_multimodal(
        &mut self,
        _token_ids: Vec<u32>,
        _fused_hidden: Vec<f32>,
        _max_tokens: usize,
        _temperature: f32,
        _top_k: usize,
        _top_p: f32,
        _thinking_budget: Option<usize>,
    ) -> ExecutorResult<String> {
        // ARCH-RUST-IS-CODEGEN: mega-kernel 尚未支持 multimodal (fused hidden 注入)
        Err(ExecutorError::Backend(BackendError::Other(
            "generate_with_multimodal: mega-kernel 尚未支持 multimodal 输入。\
             需要扩展 generate_single_sequence 以接受 fused hidden state (ARCH-MULTIMODAL-FUSION)。\
             当前仅支持纯文本路径 generate_with_sampling (ARCH-RUST-IS-CODEGEN)。".into(),
        )))
    }

    /// Generate with session affinity for multi-turn conversation KV cache reuse.
    ///
    /// ARCH-RUST-IS-CODEGEN: 需要支持 session KV cache 复用的 mega-kernel 路径。
    /// 当前 mega-kernel 仅支持单次 generate_single_sequence。
    /// TODO: mega-kernel session 支持 (KV cache 跨轮次复用)
    pub fn generate_with_session(
        &mut self,
        _prompt: &str,
        _max_tokens: usize,
        _temperature: f32,
        _top_k: usize,
        _top_p: f32,
        _session_id: SessionId,
        _thinking_budget: Option<usize>,
    ) -> ExecutorResult<String> {
        // ARCH-RUST-IS-CODEGEN: mega-kernel 尚未支持 session KV cache 复用
        Err(ExecutorError::Backend(BackendError::Other(
            "generate_with_session: mega-kernel 尚未支持 multi-turn session。\
             需要扩展 generate_single_sequence 以支持 KV cache 跨轮次复用 (ARCH-SESSION)。\
             当前仅支持单轮 generate_with_sampling (ARCH-RUST-IS-CODEGEN)。".into(),
        )))
    }

    pub fn embed(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let embedding = {
            self.forward_config.graph_executor_ptr = self
                .graph_executor
                .as_mut()
                .map(|ge| ge as *mut _)
                .unwrap_or(std::ptr::null_mut()); // LEGAL: GPU 指针在 CPU 路径下为 null
            self.backend.embedding_forward_gpu_pure(
                &tokens,
                &self.topology,
                &self.weights,
                &self.forward_config,
            )?
        };
        Ok(embedding)
    }

    pub fn rerank(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        self.forward_config.graph_executor_ptr = self
            .graph_executor
            .as_mut()
            .map(|ge| ge as *mut _)
            .unwrap_or(std::ptr::null_mut()); // LEGAL: GPU 指针在 CPU 路径下为 null
        let scores = self.backend.rerank_forward_gpu_pure(
            &tokens,
            &self.topology,
            &self.weights,
            &self.forward_config,
        )?;
        Ok(scores)
    }

    /// Rerank with proper pair encoding (query + document as separate segments).
    pub fn rerank_pair(&mut self, query: &str, document: &str) -> ExecutorResult<Vec<f32>> {
        let is_decoder = self.forward_config.arch_family == crate::manifest::ArchFamily::Decoder;
        let tokens = if is_decoder {
            // Decoder-based rerankers: caller is responsible for prompt formatting.
            // We just tokenize the concatenated query+document.
            let prompt = format!("{} {}", query, document);
            self.tokenizer.encode(&prompt, self.add_special_tokens)?
        } else {
            self.tokenizer.encode_pair(query, document, self.add_special_tokens)?
        };
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        // For decoder-based rerankers without a score head, resolve yes/no token IDs
        if is_decoder && self.forward_config.rerank_yes_token_id.is_none() {
            if let Ok(yes_ids) = self.tokenizer.encode("yes", false) {
                if let Some(&id) = yes_ids.first() {
                    self.forward_config.rerank_yes_token_id = Some(id);
                }
            }
        }
        if is_decoder && self.forward_config.rerank_no_token_id.is_none() {
            if let Ok(no_ids) = self.tokenizer.encode("no", false) {
                if let Some(&id) = no_ids.first() {
                    self.forward_config.rerank_no_token_id = Some(id);
                }
            }
        }
        self.forward_config.graph_executor_ptr = self
            .graph_executor
            .as_mut()
            .map(|ge| ge as *mut _)
            .unwrap_or(std::ptr::null_mut()); // LEGAL: GPU 指针在 CPU 路径下为 null
        let scores = self.backend.rerank_forward_gpu_pure(
            &tokens,
            &self.topology,
            &self.weights,
            &self.forward_config,
        )?;
        Ok(scores)
    }

    /// Classify text: run encoder/decoder + classifier head, return raw logits.
    pub fn classify(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        self.forward_config.graph_executor_ptr = self
            .graph_executor
            .as_mut()
            .map(|ge| ge as *mut _)
            .unwrap_or(std::ptr::null_mut());
        let logits = self.backend.classify_forward_gpu_pure(
            &tokens,
            &self.topology,
            &self.weights,
            &self.forward_config,
        )?;
        Ok(logits)
    }

    /// Head Routing SDK — 对 `prompt` 跑一次 generator forward,返回
    /// `target_token_ids` 每个 id 对应的**原始 logit** (未经 softmax/温度缩放)。
    ///
    /// 调用者 (Client) 负责应用温度 + softmax 归一化。
    ///
    /// # 关联
    /// - SPEC/HEAD-ROUTING.md §4.2
    /// - SPEC/04-API-DESIGN.md §3.8
    pub fn score_tokens_for_prompt(
        &mut self,
        prompt: &str,
        target_token_ids: &[u32],
    ) -> ExecutorResult<Vec<f32>> {
        self.score_tokens_for_prompt_with_callbacks(prompt, target_token_ids, None)
    }

    /// Like `score_tokens_for_prompt` but threads an optional `CallbackChain`
    /// into the forward pass. Callbacks installed here observe `pre_node` /
    /// `post_node` hooks during the JIT-compiled forward and may trigger
    /// `ExitEarly` (HR encode_to_layer) / veto (Guardrail) semantics.
    ///
    /// The chain is attached via `forward_config.callback_chain_ptr` — the
    /// backend consumes this pointer when calling
    /// `FusedGraphExecutor::run_with_callbacks`.
    pub fn score_tokens_for_prompt_with_callbacks(
        &mut self,
        prompt: &str,
        target_token_ids: &[u32],
        callbacks: Option<&mut crate::graph::layer_callback::CallbackChain>,
    ) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(prompt)?;
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        self.forward_config.graph_executor_ptr = self
            .graph_executor
            .as_mut()
            .map(|ge| ge as *mut _)
            .unwrap_or(std::ptr::null_mut());
        self.forward_config.callback_chain_ptr = callbacks
            .map(|c| c as *mut _)
            .unwrap_or(std::ptr::null_mut());
        let result = self.backend.score_tokens_forward_gpu_pure(
            &tokens,
            target_token_ids,
            &self.topology,
            &self.weights,
            &self.forward_config,
        );
        // Clear the callback pointer so subsequent calls do not re-use a
        // dangling pointer if the caller drops the chain.
        self.forward_config.callback_chain_ptr = std::ptr::null_mut();
        let logits = result?;
        Ok(logits)
    }

    /// Head Routing §3.4 / Intent §3 — truncate the forward pass at
    /// `anchor_layer` via `MidLayerEncodeCallback` and pool the captured
    /// hidden state.
    ///
    /// Returns a `[hidden_size]` vector. `anchor_layer` must already have
    /// been resolved from a `LayerAnchor` (range check happens at Client API).
    pub fn encode_at_layer_for_prompt(
        &mut self,
        prompt: &str,
        anchor_layer: usize,
        pool: crate::head_routing::PoolMode,
    ) -> ExecutorResult<Vec<f32>> {
        use crate::graph::layer_callback::CallbackChain;
        let tokens = self.encode_prompt(prompt)?;
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        self.forward_config.graph_executor_ptr = self
            .graph_executor
            .as_mut()
            .map(|ge| ge as *mut _)
            .unwrap_or(std::ptr::null_mut());

        let mut chain = CallbackChain::new(vec![Box::new(
            crate::engine::callbacks::mid_layer_encode::MidLayerEncodeCallback::new(anchor_layer),
        )]);
        self.forward_config.callback_chain_ptr = &mut chain as *mut _;

        let result = self.backend.encode_at_layer_forward_gpu_pure(
            &tokens,
            anchor_layer,
            &self.topology,
            &self.weights,
            &self.forward_config,
        );
        self.forward_config.callback_chain_ptr = std::ptr::null_mut();
        let hidden = result?;

        let hidden_size = self.forward_config.hidden_size();
        let seq_len = tokens.len();
        if hidden_size == 0 || hidden.len() < seq_len * hidden_size {
            return Err(ExecutorError::Scheduler(format!(
                "encode_at_layer_for_prompt: hidden buffer too small, expected >= {}*{}, got {}",
                seq_len,
                hidden_size,
                hidden.len()
            )));
        }
        pool.apply(&hidden, seq_len, hidden_size).map_err(|e| {
            ExecutorError::Scheduler(format!("encode_at_layer_for_prompt pool failed: {e}"))
        })
    }

    pub fn is_finished(&self, request_id: RequestId) -> bool {
        self.requests
            .get(&request_id)
            .map(|r| r.finished)
            .unwrap_or(false) // LEGAL: 不存在的 request 视为 finished（已释放）
    }

    /// Get a reference to a request's data (for streaming inspection).
    pub fn get_request(&self, request_id: RequestId) -> Option<&RequestData> {
        self.requests.get(&request_id)
    }

    /// Release a finished request: free pages and remove from request map.
    pub fn release_request(&mut self, request_id: RequestId) {
        self.release_request_pages(request_id);
        self.requests.remove(&request_id);
    }

    pub fn get_output(&self, request_id: RequestId) -> ExecutorResult<String> {
        let req = self
            .requests
            .get(&request_id)
            .ok_or(ExecutorError::Scheduler("Request not found".into()))?;
        self.decode_tokens(&req.output_tokens)
    }

    fn ensure_kv_cache(&mut self) -> ExecutorResult<&mut KvCacheDoubleBuffer> {
        let needs_alloc = self.kv_cache.as_ref().is_none_or(|existing| {
            existing.front().config() != self.kv_cache_config
                || existing.back().config() != self.kv_cache_config
        });
        if needs_alloc {
            let front = self.backend.alloc_kv_cache(&self.kv_cache_config)?;
            let back = self.backend.alloc_kv_cache(&self.kv_cache_config)?;
            let front = KvCacheState::new(front, self.kv_cache_config.clone());
            let back = KvCacheState::new(back, self.kv_cache_config.clone());
            self.kv_cache = Some(KvCacheDoubleBuffer::new(front, back));
            self.kv_cache_slot = KvCacheSlot::Front;
        }
        self.kv_cache.as_mut().ok_or_else(|| ExecutorError::Config(ModelConfigError::InvalidConfig(
            "KV cache not available after allocation".to_string()
        )))
    }

    fn active_kv_handle(&mut self) -> ExecutorResult<KvCacheHandle> {
        let slot = self.kv_cache_slot;
        let cache = self.ensure_kv_cache()?;
        Ok(cache.slot(slot).handle())
    }

    /// Swap-out pages to secondary storage (sync).
    pub fn swap_out_pages(
        &mut self,
        page_mappings: &[(PageId, StorageKey)],
    ) -> ExecutorResult<()> {
        let mut handle = self.active_kv_handle()?;
        self.backend.swap_out_pages(&mut handle, page_mappings)?;
        Ok(())
    }


    /// 从 backend 同步页面状态到调度器（集成 get_page_states）。
    pub fn refresh_page_states(&mut self) -> ExecutorResult<()> {
        if self.kv_cache.is_some() {
            let handle = self.active_kv_handle()?;
            let states = self.backend.get_page_states(&handle)?;
            self.scheduler.sync_page_states(&states);
        }
        Ok(())
    }

    fn storage_key_to_page_id(storage_key: StorageKey) -> ExecutorResult<PageId> {
        usize::try_from(storage_key).map_err(|_| {
            ExecutorError::Scheduler("storage key does not fit into page id".to_string())
        })
    }

    fn ensure_l1_page_tracked(&mut self, physical_id: PageId) -> ExecutorResult<()> {
        match self.memory_manager.track_page(Tier::L1, physical_id) {
            Ok(()) => Ok(()),
            Err(MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 }) => {
                self.reclaim_memory(1)?;
                self.memory_manager.track_page(Tier::L1, physical_id)?;
                Ok(())
            }
            Err(err) => Err(err.into()),
        }
    }

    fn release_request_pages(&mut self, request_id: RequestId) {
        for (logical_idx, page_id) in self.scheduler.request_pages(request_id) {
            let virtual_id = VirtualPageId::new(request_id, logical_idx);
            if let Some(location) = self.memory_manager.unmap_virtual_page(virtual_id) {
                if let Err(e) = self
                    .memory_manager
                    .untrack_page(location.tier, location.physical_id) {
                    log::warn!("executor: untrack_page failed for request {request_id}: {e}");
                }
            } else {
                if let Err(e) = self.memory_manager.untrack_page(Tier::L1, page_id) {
                    log::warn!("executor: untrack_page(L1) failed for request {request_id}: {e}");
                }
            }
        }
    }

    fn reclaim_memory(&mut self, required_pages: usize) -> ExecutorResult<()> {
        if required_pages == 0 {
            return Ok(());
        }

        let mut kv_handle = self.active_kv_handle()?;
        loop {
            let usage = self.memory_manager.tier_usage(Tier::L1);
            let free_pages = usage.capacity.saturating_sub(usage.used);
            if free_pages >= required_pages {
                return Ok(());
            }

            let need = required_pages.saturating_sub(free_pages).max(1);
            let victims = self.scheduler.select_victims(need);
            if victims.is_empty() {
                return Ok(());
            }

            let mut victim_ids = Vec::with_capacity(victims.len());
            let mut swap_out_mappings = Vec::new();
            let mut planned_remaps = Vec::new();

            for (request_id, pages) in &victims {
                if pages.is_empty() {
                    continue;
                }
                victim_ids.push(*request_id);
                for (logical_idx, &l1_page_id) in pages.iter().enumerate() {
                    let storage_key = PagedScheduler::storage_key(*request_id, logical_idx)
                        .map_err(|err| ExecutorError::Scheduler(err.to_string()))?;
                    let target_page = Self::storage_key_to_page_id(storage_key)?;
                    let (target_tier, target_page) =
                        match self.memory_manager.track_page(Tier::L2, target_page) {
                            Ok(()) => (Tier::L2, target_page),
                            Err(MemoryManagerError::TierCapacityExceeded { tier: Tier::L2 }) => {
                                self.memory_manager.track_page(Tier::L3, target_page)?;
                                (Tier::L3, target_page)
                            }
                            Err(err) => return Err(err.into()),
                        };
                    let virtual_id = VirtualPageId::new(*request_id, logical_idx);
                    swap_out_mappings.push((l1_page_id, storage_key));
                    planned_remaps.push((virtual_id, l1_page_id, target_tier, target_page));
                }
            }

            if swap_out_mappings.is_empty() {
                return Ok(());
            }

            self.backend
                .swap_out_pages(&mut kv_handle, &swap_out_mappings)?;

            for (virtual_id, l1_page_id, target_tier, target_page) in planned_remaps {
                if let Some(old_location) = self.memory_manager.unmap_virtual_page(virtual_id) {
                    if let Err(e) = self
                        .memory_manager
                        .untrack_page(old_location.tier, old_location.physical_id) {
                        log::warn!("executor: untrack_page failed during reclaim: {e}");
                    }
                } else {
                    if let Err(e) = self.memory_manager.untrack_page(Tier::L1, l1_page_id) {
                        log::warn!("executor: untrack_page(L1) failed during reclaim: {e}");
                    }
                }
                self.memory_manager
                    .bind_virtual_page(virtual_id, target_tier, target_page)?;
            }

            for (request_id, pages) in &victims {
                self.scheduler.on_page_evicted(*request_id, pages);
            }
            self.scheduler
                .free_victims(&victim_ids)
                .map_err(|err| ExecutorError::Scheduler(err.to_string()))?;
        }
    }

    fn ensure_pages_resident(&mut self, request_id: RequestId) -> ExecutorResult<()> {
        if let Some(mappings) = self.scheduler.take_pending_swap_in(request_id) {
            if !mappings.is_empty() {
                let mut kv_handle = self.active_kv_handle()?;
                self.backend.swap_in_pages(&mut kv_handle, &mappings)?;
                let page_indices: Vec<PageId> = mappings
                    .iter()
                    .map(|(physical_id, _)| *physical_id)
                    .collect();

                for (logical_idx, (physical_id, storage_key)) in mappings.into_iter().enumerate() {
                    let virtual_id = VirtualPageId::new(request_id, logical_idx);
                    self.ensure_l1_page_tracked(physical_id)?;

                    let old_location = self.memory_manager.resolve(virtual_id).ok();
                    if old_location.is_some() {
                        self.memory_manager.remap_virtual_page(
                            virtual_id,
                            Tier::L1,
                            physical_id,
                        )?;
                    } else {
                        self.memory_manager
                            .bind_virtual_page(virtual_id, Tier::L1, physical_id)?;
                    }

                    if let Some((tier, page)) = old_location {
                        if tier != Tier::L1 {
                            if let Err(e) = self.memory_manager.untrack_page(tier, page) {
                                log::warn!("executor: untrack_page failed during swap-in: {e}");
                            }
                        }
                    } else {
                        let offload_page = Self::storage_key_to_page_id(storage_key)?;
                        if let Err(e) = self.memory_manager.untrack_page(Tier::L2, offload_page) {
                            log::warn!("executor: untrack_page(L2) failed for offload page: {e}");
                        }
                        if let Err(e) = self.memory_manager.untrack_page(Tier::L3, offload_page) {
                            log::warn!("executor: untrack_page(L3) failed for offload page: {e}");
                        }
                    }
                }

                self.scheduler.on_swap_in(request_id, &page_indices);
            }
        }

        let request_pages = self.scheduler.request_pages(request_id);
        if request_pages.is_empty() {
            return Ok(());
        }

        let mut swap_in_mappings = Vec::new();
        let mut swapped_pages = Vec::new();
        let mut remap_plan = Vec::new();

        for (logical_idx, physical_id) in request_pages {
            let virtual_id = VirtualPageId::new(request_id, logical_idx);
            match self.memory_manager.resolve(virtual_id) {
                Ok((Tier::L1, mapped)) if mapped == physical_id => {}
                Ok((Tier::L1, mapped)) => {
                    self.ensure_l1_page_tracked(physical_id)?;
                    self.memory_manager
                        .remap_virtual_page(virtual_id, Tier::L1, physical_id)?;
                    if let Err(e) = self.memory_manager.untrack_page(Tier::L1, mapped) {
                        log::warn!("executor: untrack_page(L1) failed for remapped page: {e}");
                    }
                }
                Ok((tier @ Tier::L2, offload_id)) | Ok((tier @ Tier::L3, offload_id)) => {
                    self.reclaim_memory(1)?;
                    self.ensure_l1_page_tracked(physical_id)?;
                    let storage_key: StorageKey = offload_id as StorageKey;
                    swap_in_mappings.push((physical_id, storage_key));
                    swapped_pages.push(physical_id);
                    remap_plan.push((virtual_id, tier, offload_id, physical_id));
                }
                Err(MemoryManagerError::UnknownVirtualPage { .. }) => {
                    self.ensure_l1_page_tracked(physical_id)?;
                    self.memory_manager
                        .bind_virtual_page(virtual_id, Tier::L1, physical_id)?;
                }
                Err(err) => return Err(err.into()),
            }
        }

        if !swap_in_mappings.is_empty() {
            let mut kv_handle = self.active_kv_handle()?;
            self.backend
                .swap_in_pages(&mut kv_handle, &swap_in_mappings)?;

            for (virtual_id, old_tier, old_physical_id, new_physical_id) in remap_plan {
                self.memory_manager
                    .remap_virtual_page(virtual_id, Tier::L1, new_physical_id)?;
                if let Err(e) = self.memory_manager.untrack_page(old_tier, old_physical_id) {
                    log::warn!("executor: untrack_page failed during swap-in remap: {e}");
                }
            }
            self.scheduler.on_swap_in(request_id, &swapped_pages);
        }

        Ok(())
    }

    fn check_memory_pressure(&mut self) -> ExecutorResult<()> {
        let Some(ref swap_cfg) = self.kv_cache_config.swap_config else {
            return Ok(());
        };
        if !swap_cfg.enable_swap || self.kv_cache.is_none() {
            return Ok(());
        }

        let threshold = swap_cfg.swap_threshold.clamp(0.0, 1.0);
        let needed_blocks = swap_cfg.lru_granularity.max(1);
        let mut pressure = self.backend.get_memory_pressure()?;
        if pressure <= threshold {
            return Ok(());
        }

        while pressure > threshold {
            self.reclaim_memory(needed_blocks)?;
            let next_pressure = self.backend.get_memory_pressure()?;
            if next_pressure >= pressure {
                break;
            }
            pressure = next_pressure;
        }

        Ok(())
    }
}

fn extract_onnx_kv_outputs(graph_outputs: &[String]) -> Vec<String> {
    graph_outputs
        .iter()
        .filter(|name| is_onnx_kv_output(name))
        .cloned()
        .collect()
}

fn is_onnx_kv_output(name: &str) -> bool {
    let normalized = name
        .trim()
        .to_ascii_lowercase()
        .replace(['/', '.', '-', ' '], "_");
    let has_cache_hint = normalized.contains("past")
        || normalized.contains("present")
        || normalized.contains("cache")
        || normalized.contains("key_values")
        || normalized.contains("kv");
    let has_kv_axis = normalized.contains("key") || normalized.contains("value");
    has_cache_hint && has_kv_axis
}

fn is_onnx_logits_output(name: &str) -> bool {
    let normalized = name
        .trim()
        .to_ascii_lowercase()
        .replace(['/', '.', '-', ' '], "_");
    normalized.contains("logits")
}

/// Extract the number of KV cache layers from ONNX KV output names.
///
/// ONNX models typically expose KV outputs as `present.{layer}.key` and
/// `present.{layer}.value`. Each layer has exactly one key and one value
/// output, so the layer count is `kv_outputs.len() / 2`.
fn onnx_kv_layer_count(kv_outputs: &[String]) -> usize {
    if kv_outputs.is_empty() {
        return 0;
    }
    // Each layer produces a key + value pair
    kv_outputs.len() / 2
}

/// Extract the top-k token IDs from a logits vector, sorted by descending probability.
///
/// Used by §17.3 to seed the SpecTree with adapter top-k candidates.
fn extract_top_k_token_ids(logits: &[f32], k: usize) -> Vec<u32> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Return the token ID with the highest logit (greedy argmax).
///
/// Used by §17.4 to determine the target model's prediction for verification.
fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn shannon_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    let log_sum = exp_sum.ln();
    let mut entropy = 0.0f32;
    for &x in logits {
        let log_p = (x - max) - log_sum;
        let p = log_p.exp();
        if p > 0.0 {
            entropy -= p * log_p;
        }
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_kv_outputs_from_onnx_names() {
        let outputs = vec![
            "logits".to_string(),
            "present.0.key".to_string(),
            "present.0.value".to_string(),
            "present.1.key".to_string(),
            "present.1.value".to_string(),
        ];
        let kv_outputs = extract_onnx_kv_outputs(&outputs);
        assert_eq!(kv_outputs.len(), 4);
        assert!(kv_outputs.iter().all(|name| name.contains("present")));
    }

    #[test]
    fn detects_logits_output_name() {
        assert!(is_onnx_logits_output("logits"));
        assert!(is_onnx_logits_output("model/logits_output"));
        assert!(!is_onnx_logits_output("present.0.key"));
    }

    #[test]
    fn shannon_entropy_uniform_distribution() {
        // Uniform distribution over 4 classes: entropy = ln(4) ≈ 1.386
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let h = super::shannon_entropy(&logits);
        assert!((h - (4.0f32).ln()).abs() < 1e-5, "uniform entropy mismatch: {h}");
    }

    #[test]
    fn shannon_entropy_peaked_distribution() {
        // Peaked distribution: one very high logit
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let h = super::shannon_entropy(&logits);
        assert!(h < 0.01, "peaked entropy should be near zero: {h}");
    }

    #[test]
    fn shannon_entropy_empty() {
        let h = super::shannon_entropy(&[]);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn onnx_kv_layer_count_from_outputs() {
        // 2 layers × (key + value) = 4 outputs → 2 layers
        let kv = vec![
            "present.0.key".to_string(),
            "present.0.value".to_string(),
            "present.1.key".to_string(),
            "present.1.value".to_string(),
        ];
        assert_eq!(super::onnx_kv_layer_count(&kv), 2);
    }

    #[test]
    fn onnx_kv_layer_count_empty() {
        assert_eq!(super::onnx_kv_layer_count(&[]), 0);
    }

    #[test]
    fn onnx_kv_layer_count_odd_outputs() {
        // 3 outputs (malformed) → floor(3/2) = 1
        let kv = vec![
            "present.0.key".to_string(),
            "present.0.value".to_string(),
            "present.1.key".to_string(),
        ];
        assert_eq!(super::onnx_kv_layer_count(&kv), 1);
    }

    #[test]
    fn extract_top_k_basic() {
        let logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let top3 = super::extract_top_k_token_ids(&logits, 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0], 1); // logits[1] = 5.0 is highest
        assert_eq!(top3[1], 3); // logits[3] = 4.0
        assert_eq!(top3[2], 2); // logits[2] = 3.0
    }

    #[test]
    fn extract_top_k_empty() {
        assert!(super::extract_top_k_token_ids(&[], 5).is_empty());
        assert!(super::extract_top_k_token_ids(&[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn extract_top_k_exceeds_len() {
        let logits = vec![3.0, 1.0];
        let top5 = super::extract_top_k_token_ids(&logits, 5);
        assert_eq!(top5.len(), 2);
        assert_eq!(top5[0], 0);
        assert_eq!(top5[1], 1);
    }

    #[test]
    fn argmax_token_basic() {
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        assert_eq!(super::argmax_token(&logits), 3);
    }

    #[test]
    fn argmax_token_single() {
        assert_eq!(super::argmax_token(&[42.0]), 0);
    }
}
