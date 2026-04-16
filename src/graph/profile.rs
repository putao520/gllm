//! Graph profiling and archetype derivation (REQ-ARB-001 ~ REQ-ARB-002).
//!
//! Extracts structural properties from `ModelConfig` into a `GraphProfile`,
//! then derives a `GraphArchetype` with independent [0,1] dimensions that
//! characterize the computational nature of the model graph.

use crate::model_config::{HiddenAct, ModelConfig};

// ── Enums (SPEC §3.2) ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnKind {
    SwiGLU,
    GeGLU,
    ReLU,
    MoESwiGLU,
    MoEGeGLU,
}

impl FfnKind {
    pub fn is_gated(&self) -> bool {
        matches!(
            self,
            Self::SwiGLU | Self::GeGLU | Self::MoESwiGLU | Self::MoEGeGLU
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionKind {
    MHA,
    GQA,
    MQA,
    SlidingWindow { window_size: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResidualKind {
    PreNorm,
    PostNorm,
    DeepNorm,
}

// ── GraphProfile (SPEC §3.1) ───────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GraphProfile {
    pub compute_density: f64,
    pub total_param_bytes: usize,
    pub fusion_opportunity: f64,
    pub avg_epilogue_chain_len: f64,
    pub ffn_kind: FfnKind,
    pub num_experts: usize,
    pub moe_top_k: usize,
    pub moe_layer_ratio: f64,
    pub attention_kind: AttentionKind,
    pub kv_q_head_ratio: f64,
    pub head_dim: usize,
    pub kv_bytes_per_token: usize,
    pub weight_reuse_ratio: f64,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub tied_embeddings: bool,
    pub residual_kind: ResidualKind,
}

// ── GraphArchetype (SPEC §3.4) ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GraphArchetype {
    pub compute_intensive: f64,
    pub memory_intensive: f64,
    pub parallelism_exploitable: f64,
    pub fusion_profitable: f64,
    pub pipeline_valuable: f64,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl GraphArchetype {
    /// Derive archetype dimensions from a profile (SPEC §3.5).
    /// Each dimension is independently clamped to [0, 1] (REQ-ARB-002).
    pub fn derive(profile: &GraphProfile) -> Self {
        // compute_intensive
        let base = sigmoid((profile.hidden_dim as f64 - 2048.0) / 2048.0);
        let boost = profile.avg_epilogue_chain_len / 8.0;
        let penalty = profile.moe_layer_ratio * 0.5;
        let compute_intensive = (base + boost - penalty).clamp(0.0, 1.0);

        // memory_intensive
        let kv_factor =
            sigmoid((profile.kv_bytes_per_token as f64 - 1024.0) / 4096.0);
        let reuse_factor = 1.0 - profile.weight_reuse_ratio;
        let memory_intensive = (kv_factor + reuse_factor).clamp(0.0, 1.0);

        // parallelism_exploitable
        let moe_factor = if profile.num_experts > 0 {
            sigmoid(
                (profile.num_experts as f64 * profile.moe_top_k as f64 - 16.0)
                    / 64.0,
            )
        } else {
            0.0
        };
        let batch_factor = 1.0 - compute_intensive * 0.3;
        let parallelism_exploitable =
            (moe_factor + batch_factor * 0.3).clamp(0.0, 1.0);

        // fusion_profitable
        let epilogue = profile.avg_epilogue_chain_len / 6.0;
        let swiglu = if profile.ffn_kind.is_gated() { 0.3 } else { 0.0 };
        let qkv = if profile.kv_q_head_ratio < 1.0 {
            0.2
        } else {
            0.1
        };
        let norm = if profile.residual_kind == ResidualKind::PreNorm {
            0.2
        } else {
            0.0
        };
        let fusion_profitable = (epilogue + swiglu + qkv + norm).clamp(0.0, 1.0);

        // pipeline_valuable
        let small_gemm =
            sigmoid((2048.0 - profile.hidden_dim as f64) / 2048.0);
        let depth =
            sigmoid((profile.num_layers as f64 - 16.0) / 32.0);
        let memory_bound = memory_intensive * 0.5;
        let pipeline_valuable =
            (small_gemm + depth * 0.3 + memory_bound).clamp(0.0, 1.0);

        Self {
            compute_intensive,
            memory_intensive,
            parallelism_exploitable,
            fusion_profitable,
            pipeline_valuable,
        }
    }
}

// ── GraphProfiler ──────────────────────────────────────────────────

pub struct GraphProfiler;

impl GraphProfiler {
    /// Extract a `GraphProfile` from a `ModelConfig`.
    ///
    /// # Design Note
    ///
    /// SPEC §3.3 specifies graph-level extraction (traversing OnnxGraph nodes/edges).
    /// However, this function runs BEFORE graph construction (which happens inside
    /// BackendContext::new), because StrategyBias must be available before JIT compilation.
    ///
    /// Fields that would ideally come from graph traversal use ModelConfig-based heuristics:
    /// - `compute_density`: 2 × hidden_dim (approximate FLOPs/byte for transformer)
    /// - `fusion_opportunity`: 0.6 for gated FFN, 0.4 otherwise
    /// - `avg_epilogue_chain_len`: 3.0 for SwiGLU/GeGLU (gate+activation+mul), 1.0 for ReLU
    /// - `moe_layer_ratio`: 0.8 for MoE models (industry standard)
    /// - `residual_kind`: PreNorm (all modern models since Llama)
    ///
    /// These heuristics are validated by the 4 golden test vectors in arbiter.rs tests.
    pub fn profile(config: &ModelConfig) -> GraphProfile {
        let hidden_dim = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let num_experts = config.num_experts.unwrap_or(0);
        let moe_top_k = config.num_experts_per_tok.unwrap_or(0);
        let vocab_size = config.vocab_size;
        let head_dim = config.head_dim;
        let tied_embeddings = config.tie_word_embeddings.unwrap_or(false);
        let num_kv_heads = config.num_key_value_heads;
        let num_attn_heads = config.num_attention_heads;

        let attention_kind = if num_kv_heads == 1 {
            AttentionKind::MQA
        } else if num_kv_heads < num_attn_heads {
            AttentionKind::GQA
        } else {
            AttentionKind::MHA
        };

        let kv_q_head_ratio = num_kv_heads as f64 / num_attn_heads.max(1) as f64;

        // 2 tensors (K+V) * num_kv_heads * head_dim * num_layers * 4 bytes (f32)
        let kv_bytes_per_token = 2 * num_kv_heads * head_dim * num_layers * std::mem::size_of::<f32>();

        let is_moe = num_experts > 0;

        let ffn_kind = match config.hidden_act.as_ref() {
            Some(HiddenAct::Silu | HiddenAct::Swish) if is_moe => FfnKind::MoESwiGLU,
            Some(HiddenAct::Silu | HiddenAct::Swish) => FfnKind::SwiGLU,
            Some(HiddenAct::Gelu | HiddenAct::GeluNew | HiddenAct::QuickGelu) if is_moe => {
                FfnKind::MoEGeGLU
            }
            Some(HiddenAct::Gelu | HiddenAct::GeluNew | HiddenAct::QuickGelu) => FfnKind::GeGLU,
            Some(HiddenAct::Relu) => FfnKind::ReLU,
            Some(HiddenAct::Unknown(_)) | None => {
                if is_moe {
                    FfnKind::MoESwiGLU
                } else {
                    FfnKind::SwiGLU
                }
            }
        };

        let residual_kind = ResidualKind::PreNorm;

        let compute_density = 2.0 * hidden_dim as f64;

        let fusion_opportunity = if ffn_kind.is_gated() { 0.6 } else { 0.4 };

        let avg_epilogue_chain_len = if ffn_kind.is_gated() { 3.0 } else { 1.0 };

        let weight_reuse_ratio = if is_moe {
            let intermediate = config.intermediate_size.unwrap_or(hidden_dim * 4) as f64;
            let expert_inter =
                config.expert_intermediate_size.unwrap_or(intermediate as usize) as f64;
            let shared = hidden_dim as f64 * intermediate;
            let expert_total =
                moe_top_k as f64 * expert_inter * hidden_dim as f64;
            let total =
                num_experts as f64 * expert_inter * hidden_dim as f64 + shared;
            if total > 0.0 {
                (shared + expert_total) / total
            } else {
                1.0
            }
        } else {
            1.0
        };

        let moe_layer_ratio = if is_moe { 0.8 } else { 0.0 };

        let intermediate = config.intermediate_size.unwrap_or(hidden_dim * 4);
        // Rough param estimate: embedding + per-layer (attn + ffn) weights
        let total_param_bytes = {
            let embed = vocab_size * hidden_dim;
            let attn_per_layer = hidden_dim * (num_attn_heads * head_dim)
                + hidden_dim * (num_kv_heads * head_dim) * 2
                + (num_attn_heads * head_dim) * hidden_dim;
            let ffn_per_layer = if is_moe {
                let expert_inter = config
                    .expert_intermediate_size
                    .unwrap_or(intermediate);
                num_experts * hidden_dim * expert_inter * 3
            } else {
                hidden_dim * intermediate * 3
            };
            (embed + (attn_per_layer + ffn_per_layer) * num_layers) * 4
        };

        GraphProfile {
            compute_density,
            total_param_bytes,
            fusion_opportunity,
            avg_epilogue_chain_len,
            ffn_kind,
            num_experts,
            moe_top_k,
            moe_layer_ratio,
            attention_kind,
            kv_q_head_ratio,
            head_dim,
            kv_bytes_per_token,
            weight_reuse_ratio,
            num_layers,
            hidden_dim,
            vocab_size,
            tied_embeddings,
            residual_kind,
        }
    }
}
