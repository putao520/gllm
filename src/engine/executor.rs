//! Executor skeleton.

use std::fmt;

use log;

use crate::compat::backend_trait::{Backend, Element};
use crate::compat::CpuBackend;
use crate::scheduler::types::{PageId, RequestId, StorageKey};
use thiserror::Error;

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

/// Static configuration for the generator forward pass.
#[derive(Debug, Clone)]
pub struct GeneratorForwardConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rope_scale: f64,
    pub rope_interleaved: bool,
    pub rope_precompute: bool,
    pub position_encoding: PositionEncoding,
    /// Architecture family (Encoder vs Decoder).
    pub arch_family: crate::manifest::ArchFamily,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// LayerNorm epsilon.
    pub norm_eps: f32,
    /// Token ID for "yes" (used by decoder-based rerankers without a score head).
    pub rerank_yes_token_id: Option<u32>,
    /// Token ID for "no" (used by decoder-based rerankers without a score head).
    pub rerank_no_token_id: Option<u32>,
    /// Active kernel execution strategy from JIT scheduler.
    pub kernel_strategy: crate::scheduler::jit_types::KernelStrategy,
    /// MoE configuration (None for dense models).
    pub moe_config: Option<crate::manifest::MoEConfig>,
}

impl GeneratorForwardConfig {
    /// Extract attention head geometry as a grouped struct.
    #[allow(dead_code)]
    pub(crate) fn attention_geometry(&self) -> crate::compat::types::AttentionGeometry {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        crate::compat::types::AttentionGeometry {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            q_dim,
            kv_dim,
            heads_per_group: self.num_heads / self.num_kv_heads.max(1),
        }
    }

    /// Extract per-layer dimension constants.
    #[allow(dead_code)]
    pub(crate) fn layer_dims(&self) -> crate::compat::types::LayerDims {
        crate::compat::types::LayerDims {
            hidden: self.hidden_size,
            inter: self.intermediate_size,
            eps: self.norm_eps,
            rope_theta: self.rope_theta,
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
#[derive(Debug, Clone, PartialEq)]
pub struct KvCacheConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub dtype_size: usize,
    pub page_size: usize,
    pub swap_config: Option<SwapConfig>,
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
/// Constructed from `ModelConfig` without additional I/O.
#[derive(Debug, Clone)]
pub struct AttentionTopology {
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (for GQA; equals `num_heads` for MHA).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Attention mask type (bidirectional vs causal).
    pub mask_type: AttentionMaskType,
    /// Maximum sequence length the model supports.
    pub max_seq_len: usize,
}

impl AttentionTopology {
    /// Construct a bidirectional (encoder) topology for BERT-style models
    /// (embedding / reranker).
    pub fn bidirectional(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            mask_type: AttentionMaskType::Bidirectional,
            max_seq_len,
        }
    }

    /// Construct a causal (decoder) topology for GPT-style generator models.
    pub fn causal(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            mask_type: AttentionMaskType::Causal,
            max_seq_len,
        }
    }

    /// Legacy compatibility constructor (minimal bidirectional topology).
    pub fn linear() -> Self {
        Self::bidirectional(1, 1, 1, 512)
    }
}

/// A single sequence in a batch.
#[derive(Debug, Clone)]
pub struct SequenceInput {
    pub tokens: Vec<u32>,
    pub position: usize,
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
use crate::scheduler::types::{BatchOrderPolicy, RequestKind};
use crate::scheduler::vllm2024::{AdaptiveChunkPolicy, Scheduler2024Config};
use crate::scheduler::{
    BasicObserver, GlobalMemoryManager, MemoryManagerError, PagedScheduler, PolicyVariant,
    PrefillPlan, ScheduledBatch, Sequence, SessionId, Tier, VirtualPageId,
};
use std::collections::HashMap;

#[derive(Debug)]
struct RequestData {
    prompt_tokens: Vec<u32>,
    output_tokens: Vec<u32>,
    sampling_config: SamplingConfig,
    is_prefill: bool,
    // kv_cache: KvCacheHandle, // Moved to Scheduler/BlockTable management
    max_new_tokens: usize,
    finished: bool,
    session_id: Option<SessionId>,
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
    }

    fn from_fused_graph(fused_graph: &crate::graph::FusedGraph) -> Self {
        let mut stats = Self::default();
        for node in &fused_graph.nodes {
            match &node.op {
                FusedOp::FlashAttention(_) => stats.flash_attention += 1,
                FusedOp::SwiGLU(_) => stats.swiglu += 1,
                FusedOp::RoPE(_) => stats.rope += 1,
                FusedOp::FusedQkvRope(_) => stats.fused_qkv_rope += 1,
                FusedOp::GQA(_) => stats.gqa += 1,
                FusedOp::MoERouting(_) => stats.moe_routing += 1,
                FusedOp::FusedRMSLinear(_) => stats.fused_rms_linear += 1,
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
    model_config: ModelConfig,
    forward_config: GeneratorForwardConfig,
    kv_cache_config: KvCacheConfig,
    tokenizer: TokenizerHandle,
    kv_cache: Option<KvCacheDoubleBuffer>,
    kv_cache_slot: KvCacheSlot,
    memory_manager: GlobalMemoryManager,
    onnx_generator_plan: Option<OnnxGeneratorPlan>,
    topology: AttentionTopology,
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
            ModelKind::Embedding | ModelKind::Reranker if model_config.rope_theta <= 0.0 => {
                PositionEncoding::None
            }
            _ => PositionEncoding::Rope,
        };
        let forward_config = GeneratorForwardConfig {
            hidden_size: model_config.hidden_size,
            num_layers: model_config.num_hidden_layers,
            num_heads: model_config.num_attention_heads,
            num_kv_heads: model_config.num_key_value_heads,
            head_dim: model_config.head_dim,
            max_seq_len: model_config.max_position_embeddings,
            vocab_size: model_config.vocab_size,
            rope_theta: model_config.rope_theta as f64,
            rope_scale: model_config.rope_scale as f64,
            rope_interleaved: model_config.rope_interleaved,
            rope_precompute: true,
            position_encoding,
            arch_family: manifest.arch.family(),
            intermediate_size: model_config.intermediate_size.unwrap_or(0),
            norm_eps: model_config.layer_norm_epsilon.unwrap_or(1e-12),
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            kernel_strategy: crate::scheduler::jit_types::KernelStrategy::AccuracyFirst,
            moe_config: manifest.moe_config,
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
        let cpu_dtype_size =
            if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend<E>>() {
                Some(4) // f32
            } else {
                None
            };

        let kv_cache_config = KvCacheConfig {
            num_layers: model_config.num_hidden_layers,
            num_heads: model_config.num_key_value_heads,
            head_dim: model_config.head_dim,
            max_seq_len: model_config.max_position_embeddings,
            dtype_size: cpu_dtype_size.unwrap_or(model_config.dtype_size),
            page_size,
            swap_config: None,
        };
        let onnx_generator_plan = Self::build_onnx_generator_plan(manifest.as_ref(), loader)?;
        let tokenizer = TokenizerHandle::from_loader(loader)?;
        let weights = loader.upload_weights(&backend)?;
        let l1_capacity = total_blocks;
        let l2_capacity = total_blocks.saturating_mul(10);
        let l3_capacity = total_blocks.saturating_mul(100);
        let memory_manager =
            GlobalMemoryManager::new_with_capacities(l1_capacity, l2_capacity, l3_capacity);
        let topology = match manifest.kind {
            ModelKind::Chat => AttentionTopology::causal(
                model_config.num_attention_heads,
                model_config.num_key_value_heads,
                model_config.head_dim,
                model_config.max_position_embeddings,
            ),
            ModelKind::Embedding | ModelKind::Reranker => AttentionTopology::bidirectional(
                model_config.num_attention_heads,
                model_config.num_key_value_heads,
                model_config.head_dim,
                model_config.max_position_embeddings,
            ),
        };
        Ok(Self {
            backend,
            scheduler,
            manifest,
            weights,
            add_special_tokens: true,
            model_config,
            forward_config,
            kv_cache_config,
            tokenizer,
            kv_cache: None,
            kv_cache_slot: KvCacheSlot::Front,
            memory_manager,
            onnx_generator_plan,
            batcher: ContinuousBatcher::new(),
            observer: BasicObserver::new(),
            policy: PolicyVariant::default(),
            requests: HashMap::new(),
            topology,
        })
    }

    fn build_onnx_generator_plan(
        manifest: &ModelManifest,
        loader: &mut Loader,
    ) -> ExecutorResult<Option<OnnxGeneratorPlan>> {
        if manifest.kind != ModelKind::Chat || loader.weight_format() != WeightFormat::Onnx {
            return Ok(None);
        }

        let onnx = loader.onnx()?;
        let onnx_graph = onnx.graph();

        // Use the new GraphOptimizer to generate FusedGraph
        let ctx = OptimizationContext::default();
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
                FusedOp::GQA(_) => OnnxKernelExecutionOp::Gqa,
                FusedOp::MoERouting(_) => OnnxKernelExecutionOp::MoERouting,
                FusedOp::FusedRMSLinear(_) => OnnxKernelExecutionOp::FusedRMSLinear,
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
            log::warn!(
                "ONNX graph does not expose identifiable KV cache outputs; \
                 falling back to no-KV-cache mode (O(n²) per step)"
            );
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
        };

        self.requests.insert(id, request_data);
        self.batcher.enqueue(sequence);
        Ok(id)
    }

    // Deprecated/Modified: enqueue_with_tokens is used by tests, so we adapt it.
    pub fn enqueue_with_tokens(
        &mut self,
        kind: RequestKind,
        prompt: impl Into<String>,
        tokens: usize,
    ) -> RequestId {
        // Adapt for tests that manually specify token count but don't care about actual encoding
        // intentional: test helper, callers don't check enqueue result
        let _ = self.enqueue_with_config(
            kind,
            prompt,
            tokens, // Interpret tokens as max_new_tokens for tests
            SamplingConfig::default(),
        );
        self.requests.len() as RequestId
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
    ) -> ExecutorResult<RequestId> {
        let req_id = self.enqueue_with_config(kind, prompt, max_new_tokens, sampling_config)?;
        self.set_session_id(req_id, session_id)?;
        Ok(req_id)
    }

    pub fn next_batch(&mut self) -> Option<ScheduledBatch> {
        if !self.batcher.has_pending_work() {
            return None;
        }
        Some(self.batcher.build_batch(
            &mut self.scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        ))
    }

    pub fn encode_prompt(&self, prompt: &str) -> ExecutorResult<Vec<u32>> {
        let add_special_tokens = self.add_special_tokens;
        Ok(self.tokenizer.encode(prompt, add_special_tokens)?)
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
            self.model_config.vocab_size,
            sampling,
        )?;
        tokens.into_iter().next().ok_or(ExecutorError::EmptySample)
    }

    fn run_batch_forward(&mut self, batch_input: &BatchInput) -> ExecutorResult<Vec<LogitsHandle>> {
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
                if onnx_layers > 0 && onnx_layers != self.kv_cache_config.num_layers {
                    log::warn!(
                        "ONNX KV output layers ({}) differs from model config layers ({}); \
                         KV cache managed by compat layer using model config",
                        onnx_layers,
                        self.kv_cache_config.num_layers
                    );
                }
            }
        }

        let kv_handle = self.active_kv_handle()?;
        let mut kv_caches = vec![kv_handle; batch_input.sequences.len()];
        Ok(self.backend.batch_forward_gpu_pure(
            batch_input,
            &self.topology,
            &self.weights,
            &mut kv_caches,
            &self.forward_config,
        )?)
    }

    /// Hot-swap the scheduling policy. Takes effect on the next `step()` call.
    pub fn set_policy(&mut self, policy: crate::scheduler::PolicyVariant) {
        self.policy = policy;
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

        // 1. JIT Decision: Decide Scheduling Strategy
        let decision = self.policy.decide(&system_state);
        self.forward_config.kernel_strategy = decision.kernel_strategy;
        if decision.kernel_strategy != crate::scheduler::jit_types::KernelStrategy::AccuracyFirst {
            log::info!("executor: kernel_strategy changed to {:?}", decision.kernel_strategy);
        }

        // 2. Schedule
        // Pass dynamic decision parameters to batcher
        let batch = if !self.batcher.has_pending_work() {
            return Ok(());
        } else {
            self.batcher.build_batch(
                &mut self.scheduler,
                decision.max_batch_size,
                decision.admit_new_prefill,
                BatchOrderPolicy::StrictRequestIdOrder,
            )
        };

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
            self.kv_cache_config.max_seq_len,
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
                let finalized = self
                    .memory_manager
                    .session_finalized_position(sid)
                    .unwrap_or_else(|| {
                        log::warn!("executor: session_finalized_position returned None for session {sid}");
                        0
                    });
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

        // 4. Prepare Batch
        for req_id in batch.requests {
            self.ensure_pages_resident(req_id)?;

            let (tokens, position) = {
                let Some(req) = self.requests.get(&req_id) else {
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
                        .unwrap_or_default()
                };

                let position = if req.is_prefill {
                    0
                } else {
                    req.prompt_tokens.len() + req.output_tokens.len().saturating_sub(1)
                };
                (tokens, position)
            };

            if tokens.is_empty() {
                if let Some(req) = self.requests.get_mut(&req_id) {
                    req.finished = true;
                }
                batch_results.push(BatchResult::fail(req_id));
                continue;
            }

            sequences.push(SequenceInput { tokens, position });
            request_indices.push(req_id);
        }

        if sequences.is_empty() {
            return Ok(());
        }

        let batch_input = BatchInput { sequences };

        // 4. Run Backend Forward
        let logits_list = self.run_batch_forward(&batch_input)?;

        // 5. Process Results
        // Note: batch_forward_gpu_pure must return results in the same order as input sequences
        if logits_list.len() != request_indices.len() {
            return Err(ExecutorError::Backend(BackendError::Cuda(
                "Backend returned mismatched number of logits".to_string(),
            )));
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
        // attention_sparsity: MHA op currently only outputs attn_out, not attention weights.
        // Requires MHA op extension to return weights for actual sparsity measurement.
        // Setting to 0.0 per SPEC "Phase 2 — reserved" positioning.
        self.observer.update_attention_sparsity(0.0);

        // Processing results loop
        for (i, logits) in logits_list.iter().enumerate() {
            let req_id = request_indices[i];
            let sampling_config = self
                .requests
                .get(&req_id)
                .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?
                .sampling_config;
            let next_token = self.sample_from_logits(logits, &sampling_config)?;

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
                batch_results.push(BatchResult::complete(req_id, Some(next_token)));
            } else {
                batch_results.push(BatchResult::continue_with_token(req_id, next_token));
            }
        }

        // Advance KV cache
        {
            let slot = self.kv_cache_slot;
            if let Some(kv_cache) = self.kv_cache.as_mut() {
                let active = kv_cache.slot_mut(slot);
                active.advance(total_tokens)?;
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
        Ok(())
    }

    /// Legacy method for compatibility with tests (e.g. test_alignment)
    /// Wraps the batch API for a single request.
    pub fn forward_step(&mut self, tokens: &[u32]) -> ExecutorResult<LogitsHandle> {
        let seq = SequenceInput {
            tokens: tokens.to_vec(),
            position: 0,
        };

        let batch_input = BatchInput {
            sequences: vec![seq],
        };

        let mut kv_cache = self.active_kv_handle()?;

        let logits_list = self.backend.batch_forward_gpu_pure(
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
        self.generate_with_sampling(prompt, max_tokens, temperature, 0, 1.0)
    }

    pub fn generate_with_sampling(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> ExecutorResult<String> {
        if prompt.trim().is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let sampling_config = SamplingConfig {
            temperature,
            top_k,
            top_p,
        };

        let req_id =
            self.enqueue_with_config(RequestKind::Chat, prompt, max_tokens, sampling_config)?;

        loop {
            self.step()?;

            if let Some(req) = self.requests.get(&req_id) {
                if req.finished {
                    break;
                }
            } else {
                break;
            }
        }

        let req = self
            .requests
            .get(&req_id)
            .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
        let text = self.decode_tokens(&req.output_tokens)?;
        Ok(text)
    }

    /// Generate with session affinity for multi-turn conversation KV cache reuse.
    pub fn generate_with_session(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: SessionId,
    ) -> ExecutorResult<String> {
        if prompt.trim().is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let sampling_config = SamplingConfig {
            temperature,
            top_k,
            top_p,
        };

        let req_id = self.enqueue_with_session(
            RequestKind::Chat,
            prompt,
            max_tokens,
            sampling_config,
            session_id,
        )?;

        loop {
            self.step()?;

            if let Some(req) = self.requests.get(&req_id) {
                if req.finished {
                    break;
                }
            } else {
                break;
            }
        }

        let req = self
            .requests
            .get(&req_id)
            .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
        let text = self.decode_tokens(&req.output_tokens)?;
        Ok(text)
    }

    pub fn embed(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let embedding = self.backend.embedding_forward_gpu_pure(
            &tokens,
            &self.topology,
            &self.weights,
            &self.forward_config,
        )?;
        Ok(embedding)
    }

    pub fn rerank(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
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
            // Decoder-based rerankers (Qwen3-Reranker) expect a chat-template formatted prompt
            let prompt = format!(
                "<|im_start|>system\nJudge whether the Document is relevant to the Query. Output only \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<query>{}</query>\n<document>{}</document><|im_end|>\n<|im_start|>assistant\n",
                query, document
            );
            self.tokenizer.encode(&prompt, false)?
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
        let scores = self.backend.rerank_forward_gpu_pure(
            &tokens,
            &self.topology,
            &self.weights,
            &self.forward_config,
        )?;
        Ok(scores)
    }

    pub fn is_finished(&self, request_id: RequestId) -> bool {
        self.requests
            .get(&request_id)
            .map(|r| r.finished)
            .unwrap_or(false) // If removed, considered finished? Or unknown.
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

    /// 异步友好的 swap-out；目前内部仍为同步调用，提供 async 接口方便集成。
    pub async fn swap_out_pages_async(
        &mut self,
        page_mappings: &[(PageId, StorageKey)],
    ) -> ExecutorResult<()> {
        let mut handle = self.active_kv_handle()?;
        self.backend.swap_out_pages(&mut handle, page_mappings)?;
        Ok(())
    }

    /// 异步友好的 swap-in，完成后通知调度器进入 Warm 保护。
    pub async fn swap_in_pages_async(
        &mut self,
        request_id: RequestId,
        page_mappings: &[(PageId, StorageKey)],
    ) -> ExecutorResult<()> {
        let mut handle = self.active_kv_handle()?;
        self.backend.swap_in_pages(&mut handle, page_mappings)?;
        let page_indices: Vec<PageId> = page_mappings
            .iter()
            .map(|(physical_id, _)| *physical_id)
            .collect();
        self.scheduler.on_swap_in(request_id, &page_indices);
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
}
