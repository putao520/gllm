//! Executor configuration types.
//!
//! Extracted from `executor.rs` for CLAUDE.md file size compliance (≤2000 lines).

use std::fmt;
use std::sync::Arc;

use gllm_kernels::types::DType;
use thiserror::Error;

use crate::kv_cache::KvCacheError;
use crate::loader::LoaderError;
use crate::model_config::ModelConfigError;
use crate::scheduler::types::RequestId;
use crate::scheduler::{MemoryManagerError, SessionId};
use crate::tokenizer::TokenizerError;

/// Maximum sequence length for KV cache sizing.
///
/// Now sourced directly from `geometry.max_seq_len` (model config
/// `max_position_embeddings`) — no longer capped by a hardcoded constant.
/// Buffer allocation uses `CompilerGraph.max_seq_len` which is set from
/// the same source during graph construction.
#[inline]
pub fn effective_kv_max_seq_len(geometry_max_seq_len: usize) -> usize {
    geometry_max_seq_len
}

// ---- Engine types (moved from compat) ----

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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// §9-§18: Callback chain handle for Gate-First Skip / Residual Bypass / Early Exit.
    /// Safe wrapper around the FFI pointer; `set()` before call, `clear()` after.
    pub callback_chain: super::coordinator::callback_slot::CallbackChainHandle,
}

impl GeneratorForwardConfig {
    /// Backward-compatible accessor: hidden size.
    pub fn hidden_size(&self) -> usize {
        self.geometry.hidden_size
    }
    /// Backward-compatible accessor: number of layers.
    pub fn num_layers(&self) -> usize {
        self.geometry.num_layers
    }
    /// Backward-compatible accessor: vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.geometry.vocab_size
    }
    /// Backward-compatible accessor: FFN intermediate dimension.
    pub fn intermediate_size(&self) -> usize {
        self.geometry.intermediate_size
    }
    /// Backward-compatible accessor: LayerNorm epsilon.
    pub fn norm_eps(&self) -> f32 {
        self.geometry.norm_eps
    }
    /// Backward-compatible accessor: model compute dtype.
    pub fn dtype(&self) -> DType {
        self.geometry.compute_dtype
    }
    /// Backward-compatible accessor: maximum sequence length from model config.
    pub fn max_seq_len(&self) -> usize {
        effective_kv_max_seq_len(self.geometry.max_seq_len)
    }
    /// Backward-compatible accessor: number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.geometry.num_heads
    }
    /// Backward-compatible accessor: number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.geometry.num_kv_heads
    }
    /// Backward-compatible accessor: head dimension.
    pub fn head_dim(&self) -> usize {
        self.geometry.head_dim
    }
    /// Derive AttentionHeadConfig from geometry.
    pub fn attention(&self) -> AttentionHeadConfig {
        AttentionHeadConfig::from_geometry(&self.geometry)
    }
    /// Backward-compatible accessor: RoPE theta.
    pub fn rope_theta(&self) -> f64 {
        self.rope.theta
    }
    /// Backward-compatible accessor: RoPE scale.
    pub fn rope_scale(&self) -> f64 {
        self.rope.scale
    }
}


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

    /// Create a minimal default config for use in tests.
    #[cfg(test)]
    pub fn default_for_test() -> Self {
        Self {
            geometry: Arc::new(crate::model_config::ModelGeometry {
                hidden_size: 64,
                num_layers: 4,
                vocab_size: 100,
                intermediate_size: 128,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 16,
                max_seq_len: 512,
                rope_theta: 10000.0,
                rope_scale: 1.0,
                rope_interleaved: false,
                dtype: DType::F32,
                compute_dtype: DType::F32,
                norm_eps: 1e-5,
                num_experts: 0,
                moe_top_k: 0,
                expert_intermediate_size: 0,
                global_rope_theta: 0.0,
                rope_partial_ratio: 1.0,
                rope_partial_ratio_global: 1.0,
                attention_pattern: vec![],
                sliding_window: 0,
                num_kv_shared_layers: 0,
                global_head_dim: 0,
                hidden_size_per_layer_input: 0,
                position_offset: None,
                rope_scaling: None,
                final_logit_softcapping: None,
                hidden_act: None,
                mla_d_c: 0,
                mla_d_rope: 0,
                mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
            }),
            rope: RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: super::coordinator::callback_slot::CallbackChainHandle::new(),
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
    /// KV cache element dtype (may differ from geometry.compute_dtype, e.g. CPU forces F32).
    pub kv_dtype: DType,
    pub page_size: usize,
    pub swap_config: Option<SwapConfig>,
}

impl KvCacheConfig {
    /// Bytes per KV cache element.
    pub fn dtype_size(&self) -> usize {
        self.kv_dtype.size_bytes()
    }
    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.geometry.num_layers
    }
    /// Number of KV attention heads.
    pub fn num_heads(&self) -> usize {
        self.geometry.num_kv_heads
    }
    /// Dimension of each attention head.
    pub fn head_dim(&self) -> usize {
        self.geometry.head_dim
    }
    /// Maximum sequence length for KV cache sizing, from model config.
    /// Buffer allocation uses `CompilerGraph.max_seq_len` set during graph
    /// construction from the same `geometry.max_seq_len` source.
    pub fn max_seq_len(&self) -> usize {
        effective_kv_max_seq_len(self.geometry.max_seq_len)
    }
    /// SharedKvRef: 后 N 层共享 KV cache.
    pub fn num_kv_shared_layers(&self) -> usize {
        self.geometry.num_kv_shared_layers
    }
    /// Per-layer attention pattern (0=sliding, 1=global).
    pub fn attention_pattern(&self) -> &[u8] {
        &self.geometry.attention_pattern
    }
    /// Whether this is an MLA model.
    pub fn is_mla(&self) -> bool {
        self.geometry.is_mla()
    }
    /// Total KV dimension per token per layer.
    /// Standard: num_kv_heads * head_dim. MLA: d_c + d_rope.
    pub fn kv_dim(&self) -> usize {
        self.geometry.kv_dim()
    }
    /// Total bytes for contiguous KV cache at max_seq_len (REQ-PA-007).
    /// = num_layers * 2 * kv_dim * max_seq_len * dtype_size
    pub fn kv_cache_bytes_for_max_seq(&self) -> usize {
        let effective_layers = if self.num_kv_shared_layers() > 0 {
            self.num_layers() - self.num_kv_shared_layers()
        } else {
            self.num_layers()
        };
        effective_layers * 2 * self.kv_dim() * self.max_seq_len() * self.dtype_size()
    }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Construct a bidirectional topology for embedding/reranker models
    /// (无 causal mask, 全 token 互可见).
    pub fn bidirectional(geometry: Arc<crate::model_config::ModelGeometry>) -> Self {
        Self {
            geometry,
            mask_type: AttentionMaskType::Bidirectional,
        }
    }

    /// Construct a causal topology for generator models
    /// (causal mask, 每个 token 只能看到前驱).
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
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        Self::bidirectional(geometry)
    }

    /// Number of query attention heads.
    pub fn num_heads(&self) -> usize {
        self.geometry.num_heads
    }
    /// Number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.geometry.num_kv_heads
    }
    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.geometry.head_dim
    }
    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.geometry.max_seq_len
    }
}

/// A single sequence in a batch.
#[derive(Debug, Clone)]
pub struct SequenceInput {
    pub tokens: Vec<u32>,
    pub position: usize,
    pub draft_steps: usize,
    /// PagedAttention page table for this sequence (flat u32 array).
    ///
    /// Each entry maps a logical block index to a physical page ID.
    /// Populated by the executor from PagedScheduler::get_page_table().
    /// None = contiguous KV access (no paging).
    pub page_table: Option<Vec<u32>>,
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

impl SequenceInput {
    /// Validate page table entries against pool bounds.
    /// Returns Ok(()) if valid, Err with description if any page ID is out of bounds.
    pub fn validate_page_table(&self, total_pages: usize) -> Result<(), String> {
        if let Some(ref pt) = self.page_table {
            for (i, &page_id) in pt.iter().enumerate() {
                if page_id as usize >= total_pages {
                    return Err(format!(
                        "page_table[{}] = {} >= total_pages {} (bounds violation)",
                        i, page_id, total_pages
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Batched input for the forward pass.
#[derive(Debug, Clone)]
pub struct BatchInput {
    pub sequences: Vec<SequenceInput>,
}

#[derive(Debug)]
pub struct RequestData {
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub sampling_config: SamplingConfig,
    pub phase: crate::scheduler::request_state::RequestPhase,
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
    #[error("JIT compilation failed: {0}")]
    Compilation(String),
    #[error("graph expansion failed: {0}")]
    GraphExpansion(String),
    #[error("sequence too long: prompt {prompt_tokens} + max_new {max_new_tokens} = {total} > max_seq_len {max_seq_len}")]
    SequenceTooLong {
        prompt_tokens: usize,
        max_new_tokens: usize,
        total: usize,
        max_seq_len: usize,
    },
}

pub type ExecutorResult<T> = std::result::Result<T, ExecutorError>;

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(max_seq_len: usize, num_layers: usize, num_kv_heads: usize, head_dim: usize, num_kv_shared: usize) -> Arc<crate::model_config::ModelGeometry> {
        Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 64,
            num_layers,
            vocab_size: 100,
            intermediate_size: 128,
            num_heads: 4,
            num_kv_heads,
            head_dim,
            max_seq_len,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: num_kv_shared,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        })
    }

    fn make_config(geo: Arc<crate::model_config::ModelGeometry>, page_size: usize) -> KvCacheConfig {
        KvCacheConfig {
            geometry: geo,
            kv_dtype: DType::F32,
            page_size,
            swap_config: None,
        }
    }

    #[test]
    fn kv_cache_bytes_for_max_seq_basic() {
        // 4 layers, 2 kv_heads, head_dim=16, max_seq=512, F32=4 bytes
        // = 4 * 2 * 2 * 16 * 512 * 4 = 524288 bytes
        let geo = geometry(512, 4, 2, 16, 0);
        let config = make_config(geo, 0);
        assert_eq!(config.kv_cache_bytes_for_max_seq(), 524288);
    }

    #[test]
    fn kv_cache_bytes_with_shared_kv_layers() {
        // 6 layers, 2 shared → effective 4 layers
        let geo = geometry(512, 6, 2, 16, 2);
        let config = make_config(geo, 0);
        // Same as basic: 4 * 2 * 2 * 16 * 512 * 4 = 524288
        assert_eq!(config.kv_cache_bytes_for_max_seq(), 524288);
    }

    #[test]
    fn kv_cache_bytes_no_shared_layers() {
        let geo = geometry(1024, 2, 4, 32, 0);
        let config = make_config(geo, 0);
        // 2 * 2 * 4 * 32 * 1024 * 4 = 2097152
        assert_eq!(config.kv_cache_bytes_for_max_seq(), 2097152);
    }

    #[test]
    fn executor_error_sequence_too_long_display() {
        let err = ExecutorError::SequenceTooLong {
            prompt_tokens: 1000,
            max_new_tokens: 200,
            total: 1200,
            max_seq_len: 1024,
        };
        let msg = format!("{err}");
        assert!(msg.contains("1200"), "should show total: {msg}");
        assert!(msg.contains("1024"), "should show max_seq_len: {msg}");
        assert!(msg.contains("1000"), "should show prompt_tokens: {msg}");
        assert!(msg.contains("200"), "should show max_new_tokens: {msg}");
    }

    #[test]
    fn executor_error_sequence_too_long_fields() {
        let err = ExecutorError::SequenceTooLong {
            prompt_tokens: 500,
            max_new_tokens: 100,
            total: 600,
            max_seq_len: 512,
        };
        match err {
            ExecutorError::SequenceTooLong { prompt_tokens, max_new_tokens, total, max_seq_len } => {
                assert_eq!(prompt_tokens, 500);
                assert_eq!(max_new_tokens, 100);
                assert_eq!(total, 600);
                assert_eq!(max_seq_len, 512);
            }
            _ => panic!("expected SequenceTooLong variant"),
        }
    }
}

