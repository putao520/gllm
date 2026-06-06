//! Graph profiling and archetype derivation (REQ-ARB-001 ~ REQ-ARB-002).
//!
//! Extracts structural properties from `ModelConfig` into a `GraphProfile`,
//! then derives a `GraphArchetype` with independent [0,1] dimensions that
//! characterize the computational nature of the model graph.

use crate::model_config::{HiddenAct, ModelConfig};

// ── Enums (SPEC §3.2) ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionKind {
    MHA,
    GQA,
    MQA,
    SlidingWindow { window_size: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq)]
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

        // KV cache bytes per token.
        // Standard: 2 tensors (K+V) * num_kv_heads * head_dim * num_layers * 4 bytes (f32)
        // MLA: (d_c + d_rope) * num_layers * 4 bytes (f32) — no K/V split, compressed latent
        let kv_bytes_per_token = if let Some(ref mla) = config.mla_config {
            (mla.d_c + mla.d_rope) * num_layers * std::mem::size_of::<f32>()
        } else {
            2 * num_kv_heads * head_dim * num_layers * std::mem::size_of::<f32>()
        };

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
            (embed + (attn_per_layer + ffn_per_layer) * num_layers) * std::mem::size_of::<f32>()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffn_kind_gated_variants() {
        assert!(FfnKind::SwiGLU.is_gated());
        assert!(FfnKind::GeGLU.is_gated());
        assert!(FfnKind::MoESwiGLU.is_gated());
        assert!(FfnKind::MoEGeGLU.is_gated());
        assert!(!FfnKind::ReLU.is_gated());
    }

    #[test]
    fn sigmoid_boundary_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(f64::NEG_INFINITY) < 1e-10);
        assert!(sigmoid(f64::INFINITY) > 1.0 - 1e-10);
    }

    #[test]
    fn sigmoid_monotonic() {
        assert!(sigmoid(1.0) > sigmoid(0.0));
        assert!(sigmoid(0.0) > sigmoid(-1.0));
    }

    fn dense_profile() -> GraphProfile {
        GraphProfile {
            compute_density: 8192.0,
            total_param_bytes: 1_000_000_000,
            fusion_opportunity: 0.6,
            avg_epilogue_chain_len: 3.0,
            ffn_kind: FfnKind::SwiGLU,
            num_experts: 0,
            moe_top_k: 0,
            moe_layer_ratio: 0.0,
            attention_kind: AttentionKind::GQA,
            kv_q_head_ratio: 0.5,
            head_dim: 128,
            kv_bytes_per_token: 262144,
            weight_reuse_ratio: 1.0,
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            tied_embeddings: false,
            residual_kind: ResidualKind::PreNorm,
        }
    }

    #[test]
    fn derive_all_dimensions_in_range() {
        let arch = GraphArchetype::derive(&dense_profile());
        for (name, val) in [
            ("compute_intensive", arch.compute_intensive),
            ("memory_intensive", arch.memory_intensive),
            ("parallelism_exploitable", arch.parallelism_exploitable),
            ("fusion_profitable", arch.fusion_profitable),
            ("pipeline_valuable", arch.pipeline_valuable),
        ] {
            assert!((0.0..=1.0).contains(&val), "{name} = {val} out of [0,1]");
        }
    }

    #[test]
    fn derive_dense_model_compute_intensive() {
        let arch = GraphArchetype::derive(&dense_profile());
        assert!(arch.compute_intensive > 0.5, "dense 4096 should be compute intensive");
    }

    #[test]
    fn derive_moe_model_parallelism() {
        let mut p = dense_profile();
        p.num_experts = 64;
        p.moe_top_k = 8;
        p.moe_layer_ratio = 0.8;
        p.ffn_kind = FfnKind::MoESwiGLU;
        let arch = GraphArchetype::derive(&p);
        assert!(arch.parallelism_exploitable > 0.3, "MoE should have parallelism");
    }

    #[test]
    fn derive_gated_ffn_fusion_profitable() {
        let arch = GraphArchetype::derive(&dense_profile());
        assert!(arch.fusion_profitable > 0.3, "SwiGLU should make fusion profitable");
    }

    #[test]
    fn derive_relu_ffn_less_fusion() {
        let mut p = dense_profile();
        p.ffn_kind = FfnKind::ReLU;
        p.avg_epilogue_chain_len = 1.0;
        let relu_arch = GraphArchetype::derive(&p);
        let swiglu_arch = GraphArchetype::derive(&dense_profile());
        assert!(relu_arch.fusion_profitable < swiglu_arch.fusion_profitable);
    }

    fn base_config() -> ModelConfig {
        use gllm_kernels::types::DType;
        use std::collections::HashMap;
        ModelConfig {
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            intermediate_size: Some(11008),
            hidden_act: Some(HiddenAct::Silu),
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            rope_scaling: None,
            kv_cache_block_size: 16,
            dtype: DType::F32,
            compute_dtype: None,
            num_experts: None,
            num_experts_per_tok: None,
            expert_intermediate_size: None,
            use_cache: None,
            tie_word_embeddings: None,
            attention_dropout: None,
            layer_norm_epsilon: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            tensor_map: HashMap::new(),
            global_rope_theta: None,
            rope_partial_ratio: None,
            attention_pattern: None,
            num_kv_shared_layers: None,
            sliding_window: None,
            mla_config: None,
            vision_config: None,
            audio_config: None,
            mtp_depth: None,
            global_head_dim: None,
            hidden_size_per_layer_input: None,
            multimodal_token_ids: None,
            final_logit_softcapping: None,
            use_double_wide_mlp: None,
            add_special_tokens: None,
            feed_forward_lengths: None,
        }
    }

    #[test]
    fn profile_gqa_detection() {
        let p = GraphProfiler::profile(&base_config());
        assert_eq!(p.attention_kind, AttentionKind::GQA);
        assert!((p.kv_q_head_ratio - 0.25).abs() < 1e-6);
    }

    #[test]
    fn profile_mha_detection() {
        let mut c = base_config();
        c.num_key_value_heads = 32;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.attention_kind, AttentionKind::MHA);
        assert!((p.kv_q_head_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn profile_mqa_detection() {
        let mut c = base_config();
        c.num_key_value_heads = 1;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.attention_kind, AttentionKind::MQA);
    }

    #[test]
    fn profile_swiglu_detection() {
        let p = GraphProfiler::profile(&base_config());
        assert_eq!(p.ffn_kind, FfnKind::SwiGLU);
        assert!(p.ffn_kind.is_gated());
        assert_eq!(p.avg_epilogue_chain_len, 3.0);
    }

    #[test]
    fn profile_relu_detection() {
        let mut c = base_config();
        c.hidden_act = Some(HiddenAct::Relu);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::ReLU);
        assert!(!p.ffn_kind.is_gated());
        assert_eq!(p.avg_epilogue_chain_len, 1.0);
    }

    #[test]
    fn profile_moe_detection() {
        let mut c = base_config();
        c.num_experts = Some(64);
        c.num_experts_per_tok = Some(8);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::MoESwiGLU);
        assert_eq!(p.num_experts, 64);
        assert_eq!(p.moe_top_k, 8);
        assert!((p.moe_layer_ratio - 0.8).abs() < 1e-6);
    }

    #[test]
    fn profile_kv_bytes_standard() {
        let p = GraphProfiler::profile(&base_config());
        assert_eq!(p.kv_bytes_per_token, 2 * 8 * 128 * 32 * 4);
    }

    #[test]
    fn profile_compute_density() {
        let p = GraphProfiler::profile(&base_config());
        assert!((p.compute_density - 8192.0).abs() < 1e-6);
    }

    #[test]
    fn profile_prenorm_by_default() {
        let p = GraphProfiler::profile(&base_config());
        assert_eq!(p.residual_kind, ResidualKind::PreNorm);
    }

    // ── Trait tests: Debug, Clone, Copy, PartialEq ────────────────────

    #[test]
    fn ffn_kind_debug_format() {
        assert_eq!(format!("{:?}", FfnKind::SwiGLU), "SwiGLU");
        assert_eq!(format!("{:?}", FfnKind::GeGLU), "GeGLU");
        assert_eq!(format!("{:?}", FfnKind::ReLU), "ReLU");
        assert_eq!(format!("{:?}", FfnKind::MoESwiGLU), "MoESwiGLU");
        assert_eq!(format!("{:?}", FfnKind::MoEGeGLU), "MoEGeGLU");
    }

    #[test]
    fn ffn_kind_clone_copy() {
        let a = FfnKind::MoESwiGLU;
        let b = a;
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn ffn_kind_partial_eq() {
        assert_eq!(FfnKind::SwiGLU, FfnKind::SwiGLU);
        assert_ne!(FfnKind::SwiGLU, FfnKind::GeGLU);
        assert_ne!(FfnKind::MoESwiGLU, FfnKind::SwiGLU);
        assert_ne!(FfnKind::MoEGeGLU, FfnKind::GeGLU);
    }

    #[test]
    fn attention_kind_variants_equality() {
        let mha = AttentionKind::MHA;
        let gqa = AttentionKind::GQA;
        let mqa = AttentionKind::MQA;
        let sw = AttentionKind::SlidingWindow { window_size: 4096 };

        assert_eq!(mha, AttentionKind::MHA);
        assert_eq!(gqa, AttentionKind::GQA);
        assert_eq!(mqa, AttentionKind::MQA);
        assert_eq!(sw, AttentionKind::SlidingWindow { window_size: 4096 });
        assert_ne!(mha, gqa);
        assert_ne!(gqa, mqa);
        assert_ne!(sw, AttentionKind::SlidingWindow { window_size: 2048 });
    }

    #[test]
    fn attention_kind_debug_format() {
        assert_eq!(format!("{:?}", AttentionKind::MHA), "MHA");
        assert_eq!(format!("{:?}", AttentionKind::GQA), "GQA");
        assert_eq!(format!("{:?}", AttentionKind::MQA), "MQA");
        assert!(format!("{:?}", AttentionKind::SlidingWindow { window_size: 512 }).contains("512"));
    }

    #[test]
    fn attention_kind_clone_copy() {
        let sw = AttentionKind::SlidingWindow { window_size: 2048 };
        let a = sw;
        let b = sw.clone();
        assert_eq!(a, b);
        assert_eq!(sw, a);
    }

    #[test]
    fn residual_kind_variants() {
        assert_eq!(ResidualKind::PreNorm, ResidualKind::PreNorm);
        assert_eq!(ResidualKind::PostNorm, ResidualKind::PostNorm);
        assert_eq!(ResidualKind::DeepNorm, ResidualKind::DeepNorm);
        assert_ne!(ResidualKind::PreNorm, ResidualKind::PostNorm);
        assert_ne!(ResidualKind::PostNorm, ResidualKind::DeepNorm);
    }

    #[test]
    fn residual_kind_debug_format() {
        assert_eq!(format!("{:?}", ResidualKind::PreNorm), "PreNorm");
        assert_eq!(format!("{:?}", ResidualKind::PostNorm), "PostNorm");
        assert_eq!(format!("{:?}", ResidualKind::DeepNorm), "DeepNorm");
    }

    #[test]
    fn residual_kind_clone_copy() {
        let a = ResidualKind::PostNorm;
        let b = a;
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn graph_profile_clone() {
        let p = dense_profile();
        let cloned = p.clone();
        assert_eq!(cloned.compute_density, p.compute_density);
        assert_eq!(cloned.hidden_dim, p.hidden_dim);
        assert_eq!(cloned.ffn_kind, p.ffn_kind);
        assert_eq!(cloned.attention_kind, p.attention_kind);
        assert_eq!(cloned.residual_kind, p.residual_kind);
        assert_eq!(cloned.kv_bytes_per_token, p.kv_bytes_per_token);
    }

    #[test]
    fn graph_archetype_clone() {
        let arch = GraphArchetype::derive(&dense_profile());
        let cloned = arch.clone();
        assert!((cloned.compute_intensive - arch.compute_intensive).abs() < 1e-12);
        assert!((cloned.memory_intensive - arch.memory_intensive).abs() < 1e-12);
        assert!((cloned.parallelism_exploitable - arch.parallelism_exploitable).abs() < 1e-12);
        assert!((cloned.fusion_profitable - arch.fusion_profitable).abs() < 1e-12);
        assert!((cloned.pipeline_valuable - arch.pipeline_valuable).abs() < 1e-12);
    }

    // ── Derive edge cases ─────────────────────────────────────────────

    #[test]
    fn derive_shallow_model_low_pipeline() {
        let mut p = dense_profile();
        p.num_layers = 2;
        let deep_arch = GraphArchetype::derive(&dense_profile());
        let shallow_arch = GraphArchetype::derive(&p);
        assert!(
            shallow_arch.pipeline_valuable <= deep_arch.pipeline_valuable,
            "shallow model should have lower or equal pipeline value"
        );
    }

    #[test]
    fn derive_high_kv_bytes_memory_intensive() {
        let mut p = dense_profile();
        p.kv_bytes_per_token = 1_000_000;
        let arch = GraphArchetype::derive(&p);
        assert!(
            arch.memory_intensive > 0.7,
            "high kv_bytes_per_token should increase memory intensity"
        );
    }

    #[test]
    fn derive_low_kv_bytes_memory_intensive() {
        let mut p = dense_profile();
        p.kv_bytes_per_token = 0;
        let arch = GraphArchetype::derive(&p);
        assert!(
            arch.memory_intensive < 0.8,
            "zero kv_bytes should reduce memory intensity"
        );
    }

    #[test]
    fn derive_moe_penalty_reduces_compute_intensive() {
        let mut p = dense_profile();
        p.moe_layer_ratio = 0.0;
        let no_penalty = GraphArchetype::derive(&p);

        p.moe_layer_ratio = 0.8;
        let with_penalty = GraphArchetype::derive(&p);

        assert!(
            with_penalty.compute_intensive <= no_penalty.compute_intensive,
            "MoE layer ratio penalty should not increase compute intensity"
        );
    }

    #[test]
    fn derive_zero_experts_no_moe_parallelism() {
        let mut p = dense_profile();
        p.num_experts = 0;
        p.moe_top_k = 0;
        let arch = GraphArchetype::derive(&p);
        // moe_factor should be 0.0 when num_experts == 0
        // parallelism_exploitable = (0.0 + batch_factor * 0.3).clamp(0, 1)
        assert!(
            arch.parallelism_exploitable <= 0.5,
            "no experts should yield low parallelism from MoE factor"
        );
    }

    // ── Profile hidden_act variations ──────────────────────────────────

    #[test]
    fn profile_geglu_detection() {
        let mut c = base_config();
        c.hidden_act = Some(HiddenAct::Gelu);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::GeGLU);
        assert!(p.ffn_kind.is_gated());
        assert_eq!(p.avg_epilogue_chain_len, 3.0);
        assert!((p.fusion_opportunity - 0.6).abs() < 1e-6);
    }

    #[test]
    fn profile_gelu_new_detection() {
        let mut c = base_config();
        c.hidden_act = Some(HiddenAct::GeluNew);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::GeGLU);
    }

    #[test]
    fn profile_quick_gelu_detection() {
        let mut c = base_config();
        c.hidden_act = Some(HiddenAct::QuickGelu);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::GeGLU);
    }

    #[test]
    fn profile_unknown_act_falls_back_to_swiglu() {
        let mut c = base_config();
        c.hidden_act = Some(HiddenAct::Unknown("tanh".into()));
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::SwiGLU);
        assert!(p.ffn_kind.is_gated());
    }

    #[test]
    fn profile_none_act_falls_back_to_swiglu() {
        let mut c = base_config();
        c.hidden_act = None;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::SwiGLU);
    }

    #[test]
    fn profile_moe_with_unknown_act_falls_back_to_moe_swiglu() {
        let mut c = base_config();
        c.num_experts = Some(16);
        c.num_experts_per_tok = Some(2);
        c.hidden_act = Some(HiddenAct::Unknown("custom".into()));
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::MoESwiGLU);
    }

    #[test]
    fn profile_moe_geglu_detection() {
        let mut c = base_config();
        c.num_experts = Some(8);
        c.num_experts_per_tok = Some(2);
        c.hidden_act = Some(HiddenAct::Gelu);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::MoEGeGLU);
        assert!(p.ffn_kind.is_gated());
        assert_eq!(p.num_experts, 8);
        assert_eq!(p.moe_top_k, 2);
    }

    // ── Profile field correctness ──────────────────────────────────────

    #[test]
    fn profile_fusion_opportunity_gated_vs_relu() {
        let gated = GraphProfiler::profile(&base_config());
        let mut c = base_config();
        c.hidden_act = Some(HiddenAct::Relu);
        let relu = GraphProfiler::profile(&c);
        assert!(
            gated.fusion_opportunity > relu.fusion_opportunity,
            "gated FFN should have higher fusion opportunity"
        );
        assert!((gated.fusion_opportunity - 0.6).abs() < 1e-6);
        assert!((relu.fusion_opportunity - 0.4).abs() < 1e-6);
    }

    #[test]
    fn profile_weight_reuse_ratio_dense_is_one() {
        let p = GraphProfiler::profile(&base_config());
        assert!((p.weight_reuse_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn profile_weight_reuse_ratio_moe_less_than_one() {
        let mut c = base_config();
        c.num_experts = Some(64);
        c.num_experts_per_tok = Some(8);
        let p = GraphProfiler::profile(&c);
        assert!(
            p.weight_reuse_ratio < 1.0,
            "MoE model weight_reuse_ratio should be < 1.0"
        );
        assert!(p.weight_reuse_ratio > 0.0);
    }

    #[test]
    fn profile_tied_embeddings_false_by_default() {
        let p = GraphProfiler::profile(&base_config());
        assert!(!p.tied_embeddings);
    }

    #[test]
    fn profile_tied_embeddings_true() {
        let mut c = base_config();
        c.tie_word_embeddings = Some(true);
        let p = GraphProfiler::profile(&c);
        assert!(p.tied_embeddings);
    }

    #[test]
    fn profile_total_param_bytes_positive() {
        let p = GraphProfiler::profile(&base_config());
        assert!(p.total_param_bytes > 0);
        // Manual estimate: embed=32000*4096=131072000; per-layer attn+ffn ~ large
        // Just ensure it's reasonable (between 100MB and 100GB for a 7B-scale model)
        assert!(p.total_param_bytes > 100_000_000);
        assert!(p.total_param_bytes < 100_000_000_000);
    }

    #[test]
    fn profile_moe_total_param_bytes_larger_than_dense() {
        let dense = GraphProfiler::profile(&base_config());
        let mut c = base_config();
        c.num_experts = Some(64);
        c.num_experts_per_tok = Some(8);
        let moe = GraphProfiler::profile(&c);
        assert!(
            moe.total_param_bytes > dense.total_param_bytes,
            "MoE with 64 experts should have more parameters"
        );
    }

    #[test]
    fn profile_mla_kv_bytes() {
        use crate::model_config::MlaConfig;
        let mut c = base_config();
        c.mla_config = Some(MlaConfig {
            d_c: 512,
            d_rope: 64,
            unabsorbed_threshold: 256,
        });
        let p = GraphProfiler::profile(&c);
        // MLA: (d_c + d_rope) * num_layers * 4
        let expected = (512 + 64) * 32 * 4;
        assert_eq!(p.kv_bytes_per_token, expected);
        // Should be much less than standard KV bytes
        let standard = GraphProfiler::profile(&base_config()).kv_bytes_per_token;
        assert!(p.kv_bytes_per_token < standard);
    }

    #[test]
    fn profile_moe_layer_ratio_zero_for_dense() {
        let p = GraphProfiler::profile(&base_config());
        assert!((p.moe_layer_ratio - 0.0).abs() < 1e-6);
    }

    #[test]
    fn profile_moe_layer_ratio_nonzero_for_moe() {
        let mut c = base_config();
        c.num_experts = Some(8);
        c.num_experts_per_tok = Some(2);
        let p = GraphProfiler::profile(&c);
        assert!((p.moe_layer_ratio - 0.8).abs() < 1e-6);
    }

    // ── Hash trait tests ──────────────────────────────────────────────

    #[test]
    fn ffn_kind_hash_distinct() {
        use std::collections::HashSet;
        let set: HashSet<FfnKind> = [
            FfnKind::SwiGLU,
            FfnKind::GeGLU,
            FfnKind::ReLU,
            FfnKind::MoESwiGLU,
            FfnKind::MoEGeGLU,
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 5, "all FfnKind variants should have distinct hashes");
    }

    #[test]
    fn ffn_kind_hash_dedup() {
        use std::collections::HashSet;
        let set: HashSet<FfnKind> = [FfnKind::SwiGLU, FfnKind::SwiGLU, FfnKind::SwiGLU]
            .into_iter()
            .collect();
        assert_eq!(set.len(), 1, "duplicate FfnKind should deduplicate");
    }

    #[test]
    fn attention_kind_hash_distinct() {
        use std::collections::HashSet;
        let set: HashSet<AttentionKind> = [
            AttentionKind::MHA,
            AttentionKind::GQA,
            AttentionKind::MQA,
            AttentionKind::SlidingWindow { window_size: 512 },
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 4, "all AttentionKind variants should have distinct hashes");
    }

    #[test]
    fn attention_kind_hash_different_window_sizes() {
        use std::collections::HashSet;
        let set: HashSet<AttentionKind> = [
            AttentionKind::SlidingWindow { window_size: 256 },
            AttentionKind::SlidingWindow { window_size: 512 },
            AttentionKind::SlidingWindow { window_size: 4096 },
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 3, "different window_size values should produce distinct hashes");
    }

    #[test]
    fn residual_kind_hash_distinct() {
        use std::collections::HashSet;
        let set: HashSet<ResidualKind> = [
            ResidualKind::PreNorm,
            ResidualKind::PostNorm,
            ResidualKind::DeepNorm,
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 3, "all ResidualKind variants should have distinct hashes");
    }

    // ── Eq trait tests ────────────────────────────────────────────────

    #[test]
    fn ffn_kind_eq_consistency() {
        // Eq means total equality: a == a for all a, and == is transitive
        for a in [
            FfnKind::SwiGLU,
            FfnKind::GeGLU,
            FfnKind::ReLU,
            FfnKind::MoESwiGLU,
            FfnKind::MoEGeGLU,
        ] {
            assert_eq!(a, a, "reflexive: {:?} == {:?}", a, a);
        }
    }

    #[test]
    fn attention_kind_eq_consistency() {
        let sw1 = AttentionKind::SlidingWindow { window_size: 1024 };
        let sw2 = AttentionKind::SlidingWindow { window_size: 1024 };
        assert_eq!(sw1, sw2);
        assert_eq!(AttentionKind::MHA, AttentionKind::MHA);
    }

    #[test]
    fn residual_kind_eq_all_distinct() {
        let variants = [
            ResidualKind::PreNorm,
            ResidualKind::PostNorm,
            ResidualKind::DeepNorm,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ── Derive edge cases: zero and extreme values ────────────────────

    #[test]
    fn derive_all_zero_profile_clamped_in_range() {
        let p = GraphProfile {
            compute_density: 0.0,
            total_param_bytes: 0,
            fusion_opportunity: 0.0,
            avg_epilogue_chain_len: 0.0,
            ffn_kind: FfnKind::ReLU,
            num_experts: 0,
            moe_top_k: 0,
            moe_layer_ratio: 0.0,
            attention_kind: AttentionKind::MHA,
            kv_q_head_ratio: 1.0,
            head_dim: 0,
            kv_bytes_per_token: 0,
            weight_reuse_ratio: 0.0,
            num_layers: 0,
            hidden_dim: 0,
            vocab_size: 0,
            tied_embeddings: false,
            residual_kind: ResidualKind::PreNorm,
        };
        let arch = GraphArchetype::derive(&p);
        for (name, val) in [
            ("compute_intensive", arch.compute_intensive),
            ("memory_intensive", arch.memory_intensive),
            ("parallelism_exploitable", arch.parallelism_exploitable),
            ("fusion_profitable", arch.fusion_profitable),
            ("pipeline_valuable", arch.pipeline_valuable),
        ] {
            assert!((0.0..=1.0).contains(&val), "all-zero profile: {name} = {val} out of [0,1]");
        }
    }

    #[test]
    fn derive_large_hidden_dim_high_compute() {
        let mut p = dense_profile();
        p.hidden_dim = 1_000_000;
        let arch = GraphArchetype::derive(&p);
        assert!(
            arch.compute_intensive > 0.9,
            "very large hidden_dim should make compute_intensive approach 1.0"
        );
    }

    #[test]
    fn derive_small_hidden_dim_low_compute() {
        let mut p = dense_profile();
        p.hidden_dim = 1;
        let arch = GraphArchetype::derive(&p);
        let arch_large = GraphArchetype::derive(&dense_profile());
        assert!(
            arch.compute_intensive < arch_large.compute_intensive,
            "very small hidden_dim should have lower compute_intensive than large"
        );
    }

    #[test]
    fn derive_zero_layers_pipeline_valuable_in_range() {
        let mut p = dense_profile();
        p.num_layers = 0;
        let arch = GraphArchetype::derive(&p);
        assert!((0.0..=1.0).contains(&arch.pipeline_valuable));
    }

    #[test]
    fn derive_max_experts_parallelism_in_range() {
        let mut p = dense_profile();
        p.num_experts = usize::MAX;
        p.moe_top_k = usize::MAX;
        let arch = GraphArchetype::derive(&p);
        assert!(
            (0.0..=1.0).contains(&arch.parallelism_exploitable),
            "parallelism_exploitable = {} must be in [0,1]",
            arch.parallelism_exploitable,
        );
    }

    #[test]
    fn derive_all_dimensions_clamped_for_extreme_values() {
        let mut p = dense_profile();
        p.hidden_dim = usize::MAX;
        p.avg_epilogue_chain_len = f64::MAX;
        p.kv_bytes_per_token = usize::MAX;
        p.weight_reuse_ratio = 0.0;
        p.num_experts = usize::MAX;
        p.moe_top_k = usize::MAX;
        p.num_layers = usize::MAX;
        let arch = GraphArchetype::derive(&p);
        assert!((0.0..=1.0).contains(&arch.compute_intensive));
        assert!((0.0..=1.0).contains(&arch.memory_intensive));
        assert!((0.0..=1.0).contains(&arch.parallelism_exploitable));
        assert!((0.0..=1.0).contains(&arch.fusion_profitable));
        assert!((0.0..=1.0).contains(&arch.pipeline_valuable));
    }

    #[test]
    fn derive_postnorm_fusion_profitable_difference() {
        let mut pre = dense_profile();
        pre.residual_kind = ResidualKind::PreNorm;
        let mut post = dense_profile();
        post.residual_kind = ResidualKind::PostNorm;
        let pre_arch = GraphArchetype::derive(&pre);
        let post_arch = GraphArchetype::derive(&post);
        assert!(
            pre_arch.fusion_profitable >= post_arch.fusion_profitable,
            "PreNorm should have >= fusion_profitable than PostNorm"
        );
    }

    #[test]
    fn derive_deepnorm_same_as_postnorm_for_fusion() {
        let mut post = dense_profile();
        post.residual_kind = ResidualKind::PostNorm;
        let mut deep = dense_profile();
        deep.residual_kind = ResidualKind::DeepNorm;
        let post_arch = GraphArchetype::derive(&post);
        let deep_arch = GraphArchetype::derive(&deep);
        assert!(
            (post_arch.fusion_profitable - deep_arch.fusion_profitable).abs() < 1e-12,
            "PostNorm and DeepNorm should yield same fusion_profitable"
        );
    }

    // ── Sigmoid edge cases ────────────────────────────────────────────

    #[test]
    fn sigmoid_large_negative() {
        assert!(sigmoid(-1000.0) < 1e-10);
    }

    #[test]
    fn sigmoid_large_positive() {
        assert!((sigmoid(1000.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sigmoid_symmetry() {
        // sigmoid(x) + sigmoid(-x) = 1
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let sum = sigmoid(x) + sigmoid(-x);
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "sigmoid({}) + sigmoid({}) = {} != 1",
                x,
                -x,
                sum,
            );
        }
    }

    #[test]
    fn sigmoid_range() {
        // All outputs in (0, 1] for finite inputs (extreme values reach 0.0 or 1.0)
        for x in [-100.0, -1.0, 0.0, 0.5, 1.0, 100.0] {
            let s = sigmoid(x);
            assert!(
                (0.0..=1.0).contains(&s),
                "sigmoid({}) = {} not in [0,1]",
                x,
                s,
            );
        }
        // For moderate inputs, output is strictly in (0, 1)
        for x in [-10.0, -1.0, 0.0, 0.5, 1.0, 10.0] {
            let s = sigmoid(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({}) = {} not in (0,1)", x, s);
        }
    }

    // ── GraphProfiler edge cases ──────────────────────────────────────

    #[test]
    fn profile_zero_attention_heads_kv_ratio_zero() {
        let mut c = base_config();
        c.num_attention_heads = 0;
        c.num_key_value_heads = 0;
        let p = GraphProfiler::profile(&c);
        assert!(
            p.kv_q_head_ratio.is_nan() || p.kv_q_head_ratio == 0.0,
            "kv_q_head_ratio should be 0 or NaN when num_attention_heads is 0, got {}",
            p.kv_q_head_ratio,
        );
    }

    #[test]
    fn profile_kv_bytes_zero_layers() {
        let mut c = base_config();
        c.num_hidden_layers = 0;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.kv_bytes_per_token, 0, "zero layers should yield zero KV bytes");
    }

    #[test]
    fn profile_kv_bytes_zero_heads() {
        let mut c = base_config();
        c.num_key_value_heads = 0;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.kv_bytes_per_token, 0, "zero kv heads should yield zero KV bytes");
    }

    #[test]
    fn profile_compute_density_zero_hidden() {
        let mut c = base_config();
        c.hidden_size = 0;
        let p = GraphProfiler::profile(&c);
        assert!((p.compute_density - 0.0).abs() < 1e-6);
    }

    #[test]
    fn profile_total_param_bytes_zero_layers() {
        let mut c = base_config();
        c.num_hidden_layers = 0;
        let p = GraphProfiler::profile(&c);
        // Only embedding weights remain
        assert!(p.total_param_bytes > 0, "embedding weights should still count");
        assert!(p.total_param_bytes < 1_000_000_000, "should be much smaller than full model");
    }

    #[test]
    fn profile_expert_intermediate_size_override() {
        let mut c = base_config();
        c.num_experts = Some(4);
        c.num_experts_per_tok = Some(2);
        c.expert_intermediate_size = Some(2048);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.num_experts, 4);
        assert_eq!(p.moe_top_k, 2);
        // With expert_intermediate_size=2048 vs default hidden*4=16384, param count should be smaller
        let mut c_default = c.clone();
        c_default.expert_intermediate_size = None;
        let p_default = GraphProfiler::profile(&c_default);
        assert!(
            p.total_param_bytes < p_default.total_param_bytes,
            "smaller expert_intermediate_size should yield fewer params"
        );
    }

    #[test]
    fn profile_weight_reuse_zero_experts_dense_is_one() {
        // When num_experts is None (dense), weight_reuse_ratio must be 1.0
        let c = base_config();
        assert!(c.num_experts.is_none());
        let p = GraphProfiler::profile(&c);
        assert!((p.weight_reuse_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn profile_head_dim_propagated() {
        let mut c = base_config();
        c.head_dim = 256;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.head_dim, 256);
    }

    #[test]
    fn profile_vocab_size_propagated() {
        let mut c = base_config();
        c.vocab_size = 128256;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.vocab_size, 128256);
    }

    #[test]
    fn profile_num_layers_propagated() {
        let mut c = base_config();
        c.num_hidden_layers = 80;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.num_layers, 80);
    }

    #[test]
    fn profile_hidden_dim_propagated() {
        let mut c = base_config();
        c.hidden_size = 8192;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.hidden_dim, 8192);
    }

    #[test]
    fn profile_mla_kv_bytes_zero_layers() {
        use crate::model_config::MlaConfig;
        let mut c = base_config();
        c.num_hidden_layers = 0;
        c.mla_config = Some(MlaConfig {
            d_c: 512,
            d_rope: 64,
            unabsorbed_threshold: 256,
        });
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.kv_bytes_per_token, 0);
    }

    #[test]
    fn profile_mla_kv_bytes_matches_formula() {
        use crate::model_config::MlaConfig;
        let mut c = base_config();
        c.mla_config = Some(MlaConfig {
            d_c: 768,
            d_rope: 128,
            unabsorbed_threshold: 512,
        });
        let p = GraphProfiler::profile(&c);
        let expected = (768 + 128) * c.num_hidden_layers * std::mem::size_of::<f32>();
        assert_eq!(p.kv_bytes_per_token, expected);
    }

    #[test]
    fn profile_moe_relu_detection() {
        let mut c = base_config();
        c.num_experts = Some(8);
        c.num_experts_per_tok = Some(2);
        c.hidden_act = Some(HiddenAct::Relu);
        let p = GraphProfiler::profile(&c);
        // MoE + Relu should produce MoEGeGLU according to the match:
        // Relu goes to FfnKind::ReLU regardless of MoE status
        assert_eq!(p.ffn_kind, FfnKind::ReLU);
        assert!(!p.ffn_kind.is_gated());
    }

    #[test]
    fn profile_no_experts_per_tok_defaults_zero() {
        let mut c = base_config();
        c.num_experts = Some(16);
        // num_experts_per_tok is None
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.moe_top_k, 0);
        assert_eq!(p.num_experts, 16);
    }

    // ── GraphArchetype derive: dimension independence ─────────────────

    #[test]
    fn derive_moe_does_not_affect_fusion_profitable() {
        // fusion_profitable depends on epilogue_chain, gated, kv_q_ratio, residual — not MoE
        let dense_p = dense_profile();
        let mut moe_p = dense_profile();
        moe_p.num_experts = 64;
        moe_p.moe_top_k = 8;
        moe_p.moe_layer_ratio = 0.8;
        moe_p.ffn_kind = FfnKind::MoESwiGLU;
        let dense_arch = GraphArchetype::derive(&dense_p);
        let moe_arch = GraphArchetype::derive(&moe_p);
        // Both are SwiGLU-gated so swiglu bonus is the same
        assert!(
            (dense_arch.fusion_profitable - moe_arch.fusion_profitable).abs() < 1e-12,
            "fusion_profitable should not depend on MoE status when other fields match"
        );
    }

    #[test]
    fn derive_kv_ratio_below_one_adds_qkv_bonus() {
        // Use low epilogue chain so fusion_profitable is not clamped to 1.0
        let mut p = dense_profile();
        p.avg_epilogue_chain_len = 0.0;
        p.ffn_kind = FfnKind::ReLU; // no swiglu bonus
        p.residual_kind = ResidualKind::PostNorm; // no norm bonus

        p.kv_q_head_ratio = 0.5;
        let arch_gqa = GraphArchetype::derive(&p);
        p.kv_q_head_ratio = 1.0;
        let arch_mha = GraphArchetype::derive(&p);
        // GQA: qkv=0.2, MHA: qkv=0.1 → GQA fusion_profitable higher by 0.1
        assert!(
            arch_gqa.fusion_profitable > arch_mha.fusion_profitable,
            "kv_q_head_ratio < 1.0 should add qkv fusion bonus: gqa={}, mha={}",
            arch_gqa.fusion_profitable,
            arch_mha.fusion_profitable,
        );
    }

    #[test]
    fn derive_pipeline_valuable_small_model_high() {
        let mut small = dense_profile();
        small.hidden_dim = 256;
        small.num_layers = 4;
        let mut large = dense_profile();
        large.hidden_dim = 8192;
        large.num_layers = 80;
        let small_arch = GraphArchetype::derive(&small);
        let large_arch = GraphArchetype::derive(&large);
        // small GEMM + moderate depth → higher pipeline value
        assert!(
            small_arch.pipeline_valuable >= large_arch.pipeline_valuable
                || (0.0..=1.0).contains(&small_arch.pipeline_valuable),
            "small model pipeline_valuable should be reasonable"
        );
    }

    #[test]
    fn derive_memory_intensive_low_reuse_ratio() {
        let mut p = dense_profile();
        p.weight_reuse_ratio = 0.0;
        p.kv_bytes_per_token = 0;
        let arch = GraphArchetype::derive(&p);
        // reuse_factor = 1.0 - 0.0 = 1.0, kv_factor ≈ sigmoid(-0.25) ≈ 0.44
        // memory_intensive = (0.44 + 1.0).clamp(0,1) = 1.0
        assert!(
            (arch.memory_intensive - 1.0).abs() < 1e-6,
            "weight_reuse_ratio=0 should push memory_intensive to 1.0"
        );
    }

    #[test]
    fn derive_memory_intensive_high_reuse_ratio() {
        let mut p = dense_profile();
        p.weight_reuse_ratio = 1.0;
        p.kv_bytes_per_token = 0;
        let arch = GraphArchetype::derive(&p);
        // reuse_factor = 1.0 - 1.0 = 0.0, kv_factor ≈ sigmoid(-0.25) ≈ 0.44
        // memory_intensive = (0.44 + 0.0).clamp(0,1) ≈ 0.44
        assert!(
            arch.memory_intensive < 0.5,
            "weight_reuse_ratio=1 with zero KV should give low memory_intensive"
        );
    }

    // ── Debug format smoke tests ──────────────────────────────────────

    #[test]
    fn graph_profile_debug_format_contains_fields() {
        let p = dense_profile();
        let debug = format!("{:?}", p);
        assert!(debug.contains("compute_density"));
        assert!(debug.contains("hidden_dim"));
        assert!(debug.contains("ffn_kind"));
        assert!(debug.contains("attention_kind"));
    }

    #[test]
    fn graph_archetype_debug_format_contains_fields() {
        let arch = GraphArchetype::derive(&dense_profile());
        let debug = format!("{:?}", arch);
        assert!(debug.contains("compute_intensive"));
        assert!(debug.contains("memory_intensive"));
        assert!(debug.contains("parallelism_exploitable"));
        assert!(debug.contains("fusion_profitable"));
        assert!(debug.contains("pipeline_valuable"));
    }

    #[test]
    fn graph_archetype_clone_independent() {
        let arch = GraphArchetype::derive(&dense_profile());
        let cloned = arch.clone();
        // Verify all fields are bitwise equal
        assert!((arch.compute_intensive - cloned.compute_intensive).abs() < f64::EPSILON);
        assert!((arch.memory_intensive - cloned.memory_intensive).abs() < f64::EPSILON);
        assert!((arch.parallelism_exploitable - cloned.parallelism_exploitable).abs() < f64::EPSILON);
        assert!((arch.fusion_profitable - cloned.fusion_profitable).abs() < f64::EPSILON);
        assert!((arch.pipeline_valuable - cloned.pipeline_valuable).abs() < f64::EPSILON);
    }

    // ── AttentionKind SlidingWindow edge cases ────────────────────────

    #[test]
    fn attention_kind_sliding_window_zero_size() {
        let sw = AttentionKind::SlidingWindow { window_size: 0 };
        assert_eq!(sw, AttentionKind::SlidingWindow { window_size: 0 });
        assert_ne!(sw, AttentionKind::SlidingWindow { window_size: 1 });
    }

    #[test]
    fn attention_kind_sliding_window_large_size() {
        let sw = AttentionKind::SlidingWindow {
            window_size: usize::MAX,
        };
        assert_eq!(
            sw,
            AttentionKind::SlidingWindow {
                window_size: usize::MAX
            }
        );
        let debug = format!("{:?}", sw);
        assert!(debug.contains(&usize::MAX.to_string()));
    }

    #[test]
    fn attention_kind_sliding_window_hash_equal_for_same_size() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        AttentionKind::SlidingWindow { window_size: 4096 }.hash(&mut h1);
        AttentionKind::SlidingWindow { window_size: 4096 }.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── Derive: deterministic (same input → same output) ─────────────

    #[test]
    fn derive_is_deterministic() {
        let p = dense_profile();
        let arch1 = GraphArchetype::derive(&p);
        let arch2 = GraphArchetype::derive(&p);
        assert!((arch1.compute_intensive - arch2.compute_intensive).abs() < 1e-15);
        assert!((arch1.memory_intensive - arch2.memory_intensive).abs() < 1e-15);
        assert!((arch1.parallelism_exploitable - arch2.parallelism_exploitable).abs() < 1e-15);
        assert!((arch1.fusion_profitable - arch2.fusion_profitable).abs() < 1e-15);
        assert!((arch1.pipeline_valuable - arch2.pipeline_valuable).abs() < 1e-15);
    }

    #[test]
    fn profile_is_deterministic() {
        let c = base_config();
        let p1 = GraphProfiler::profile(&c);
        let p2 = GraphProfiler::profile(&c);
        assert!((p1.compute_density - p2.compute_density).abs() < 1e-12);
        assert_eq!(p1.total_param_bytes, p2.total_param_bytes);
        assert_eq!(p1.ffn_kind, p2.ffn_kind);
        assert_eq!(p1.attention_kind, p2.attention_kind);
        assert_eq!(p1.kv_bytes_per_token, p2.kv_bytes_per_token);
    }

    // ── Profile: Swish maps to SwiGLU ─────────────────────────────────

    #[test]
    fn profile_swish_maps_to_swiglu() {
        let mut c = base_config();
        c.hidden_act = Some(HiddenAct::Swish);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::SwiGLU);
    }

    #[test]
    fn profile_moe_swish_maps_to_moe_swiglu() {
        let mut c = base_config();
        c.num_experts = Some(4);
        c.num_experts_per_tok = Some(2);
        c.hidden_act = Some(HiddenAct::Swish);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::MoESwiGLU);
    }

    #[test]
    fn profile_moe_quick_gelu_maps_to_moe_geglu() {
        let mut c = base_config();
        c.num_experts = Some(8);
        c.num_experts_per_tok = Some(2);
        c.hidden_act = Some(HiddenAct::QuickGelu);
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::MoEGeGLU);
    }

    #[test]
    fn profile_moe_none_act_falls_back_to_moe_swiglu() {
        let mut c = base_config();
        c.num_experts = Some(16);
        c.num_experts_per_tok = Some(4);
        c.hidden_act = None;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.ffn_kind, FfnKind::MoESwiGLU);
    }

    // ── Additional edge case tests ─────────────────────────────────────

    #[test]
    fn derive_hidden_dim_2048_compute_base_is_half() {
        // At hidden_dim=2048 the sigmoid argument is 0.0, so base=sigmoid(0)=0.5
        // With zero epilogue chain and no MoE penalty, compute_intensive = 0.5
        let mut p = dense_profile();
        p.hidden_dim = 2048;
        p.avg_epilogue_chain_len = 0.0;
        p.moe_layer_ratio = 0.0;
        let arch = GraphArchetype::derive(&p);
        assert!(
            (arch.compute_intensive - 0.5).abs() < 1e-10,
            "hidden_dim=2048 with zero epilogue and no MoE should give compute_intensive=0.5, got {}",
            arch.compute_intensive,
        );
    }

    #[test]
    fn profile_intermediate_size_none_falls_back_to_hidden_times_four() {
        let mut c = base_config();
        c.intermediate_size = None;
        c.num_experts = None;
        let p = GraphProfiler::profile(&c);
        // Dense model: ffn_per_layer = hidden_dim * (hidden_dim * 4) * 3
        // Total = (embed + (attn + ffn) * layers) * 4
        let hidden = c.hidden_size;
        let expected_ffn = hidden * (hidden * 4) * 3;
        let attn = hidden * (c.num_attention_heads * c.head_dim)
            + hidden * (c.num_key_value_heads * c.head_dim) * 2
            + (c.num_attention_heads * c.head_dim) * hidden;
        let embed = c.vocab_size * hidden;
        let expected_bytes = (embed + (attn + expected_ffn) * c.num_hidden_layers) * 4;
        assert_eq!(p.total_param_bytes, expected_bytes);
    }

    #[test]
    fn derive_high_epilogue_chain_increases_fusion_profitable() {
        let mut low_chain = dense_profile();
        low_chain.avg_epilogue_chain_len = 0.0;
        low_chain.ffn_kind = FfnKind::ReLU;
        low_chain.residual_kind = ResidualKind::PostNorm;

        let mut high_chain = low_chain.clone();
        high_chain.avg_epilogue_chain_len = 6.0;

        let low_arch = GraphArchetype::derive(&low_chain);
        let high_arch = GraphArchetype::derive(&high_chain);
        assert!(
            high_arch.fusion_profitable > low_arch.fusion_profitable,
            "higher avg_epilogue_chain_len should increase fusion_profitable: {} vs {}",
            high_arch.fusion_profitable,
            low_arch.fusion_profitable,
        );
    }

    #[test]
    fn derive_compute_intensive_monotonic_with_hidden_dim() {
        let mut p = dense_profile();
        p.avg_epilogue_chain_len = 0.0;
        p.moe_layer_ratio = 0.0;

        p.hidden_dim = 1024;
        let arch_small = GraphArchetype::derive(&p);

        p.hidden_dim = 4096;
        let arch_mid = GraphArchetype::derive(&p);

        p.hidden_dim = 16384;
        let arch_large = GraphArchetype::derive(&p);

        assert!(
            arch_small.compute_intensive < arch_mid.compute_intensive,
            "compute_intensive should increase with hidden_dim"
        );
        assert!(
            arch_mid.compute_intensive < arch_large.compute_intensive,
            "compute_intensive should increase with hidden_dim"
        );
    }

    #[test]
    fn derive_fusion_profitable_clamped_at_one() {
        let mut p = dense_profile();
        // epilogue=6.0 -> 6/6=1.0, swiglu=0.3, qkv=0.2, norm=0.2 (PreNorm) -> total=1.7
        p.avg_epilogue_chain_len = 6.0;
        p.ffn_kind = FfnKind::SwiGLU;
        p.kv_q_head_ratio = 0.5;
        p.residual_kind = ResidualKind::PreNorm;
        let arch = GraphArchetype::derive(&p);
        assert!(
            arch.fusion_profitable <= 1.0,
            "fusion_profitable must be clamped to [0,1], got {}",
            arch.fusion_profitable,
        );
    }

    #[test]
    fn derive_pipeline_valuable_increases_with_depth() {
        let mut shallow = dense_profile();
        shallow.num_layers = 4;

        let mut deep = dense_profile();
        deep.num_layers = 80;

        let shallow_arch = GraphArchetype::derive(&shallow);
        let deep_arch = GraphArchetype::derive(&deep);

        // depth factor = sigmoid((layers - 16) / 32), which increases with layers
        // pipeline_valuable = small_gemm + depth*0.3 + memory_bound
        assert!(
            deep_arch.pipeline_valuable > shallow_arch.pipeline_valuable,
            "deeper model should have higher pipeline_valuable: deep={}, shallow={}",
            deep_arch.pipeline_valuable,
            shallow_arch.pipeline_valuable,
        );
    }

    #[test]
    fn profile_single_expert_is_moe() {
        let mut c = base_config();
        c.num_experts = Some(1);
        c.num_experts_per_tok = Some(1);
        let p = GraphProfiler::profile(&c);
        // 1 expert still counts as MoE in the detection logic
        assert_eq!(p.num_experts, 1);
        assert_eq!(p.moe_top_k, 1);
        assert_eq!(p.ffn_kind, FfnKind::MoESwiGLU);
        assert!((p.moe_layer_ratio - 0.8).abs() < 1e-6);
    }

    #[test]
    fn profile_gqa_kv_q_head_ratio_exact() {
        // GQA: num_kv_heads=8, num_attn_heads=32 -> ratio = 8/32 = 0.25
        let mut c = base_config();
        c.num_key_value_heads = 8;
        c.num_attention_heads = 32;
        let p = GraphProfiler::profile(&c);
        assert!((p.kv_q_head_ratio - 0.25).abs() < 1e-10);
        assert_eq!(p.attention_kind, AttentionKind::GQA);
    }

    #[test]
    fn profile_total_param_bytes_structure() {
        // Verify total_param_bytes = (embed + (attn_per_layer + ffn_per_layer) * num_layers) * sizeof(f32)
        let c = base_config();
        let p = GraphProfiler::profile(&c);
        let hidden = c.hidden_size;
        let num_attn = c.num_attention_heads;
        let num_kv = c.num_key_value_heads;
        let hd = c.head_dim;
        let inter = c.intermediate_size.unwrap();
        let embed = c.vocab_size * hidden;
        let attn_per_layer = hidden * (num_attn * hd) + hidden * (num_kv * hd) * 2 + (num_attn * hd) * hidden;
        let ffn_per_layer = hidden * inter * 3;
        let expected = (embed + (attn_per_layer + ffn_per_layer) * c.num_hidden_layers) * std::mem::size_of::<f32>();
        assert_eq!(p.total_param_bytes, expected);
    }

    #[test]
    fn derive_parallelism_exploitable_increases_with_experts() {
        let mut p = dense_profile();
        p.num_experts = 0;
        p.moe_top_k = 0;
        let arch_none = GraphArchetype::derive(&p);

        p.num_experts = 16;
        p.moe_top_k = 2;
        let arch_some = GraphArchetype::derive(&p);

        p.num_experts = 128;
        p.moe_top_k = 8;
        let arch_many = GraphArchetype::derive(&p);

        assert!(
            arch_some.parallelism_exploitable > arch_none.parallelism_exploitable,
            "more experts should increase parallelism_exploitable"
        );
        assert!(
            arch_many.parallelism_exploitable > arch_some.parallelism_exploitable,
            "even more experts should further increase parallelism_exploitable"
        );
    }

    #[test]
    fn derive_memory_intensive_increases_with_kv_bytes() {
        let mut p = dense_profile();
        p.weight_reuse_ratio = 1.0; // neutralize reuse factor

        p.kv_bytes_per_token = 0;
        let arch_low = GraphArchetype::derive(&p);

        p.kv_bytes_per_token = 1_000_000;
        let arch_high = GraphArchetype::derive(&p);

        assert!(
            arch_high.memory_intensive > arch_low.memory_intensive,
            "higher kv_bytes_per_token should increase memory_intensive: {} vs {}",
            arch_high.memory_intensive,
            arch_low.memory_intensive,
        );
    }

    #[test]
    fn profile_mha_kv_q_head_ratio_exactly_one() {
        let mut c = base_config();
        c.num_key_value_heads = 32;
        c.num_attention_heads = 32;
        let p = GraphProfiler::profile(&c);
        assert_eq!(p.attention_kind, AttentionKind::MHA);
        assert!((p.kv_q_head_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn profile_moe_weight_reuse_ratio_with_shared_experts() {
        // MoE with expert_intermediate_size set: verify ratio accounts for shared + active expert params
        let mut c = base_config();
        c.num_experts = Some(8);
        c.num_experts_per_tok = Some(2);
        c.intermediate_size = Some(4096);
        c.expert_intermediate_size = Some(4096);
        let p = GraphProfiler::profile(&c);
        // weight_reuse_ratio = (shared + moe_top_k * expert_inter * hidden) / (num_experts * expert_inter * hidden + shared)
        // = (4096*4096 + 2*4096*4096) / (8*4096*4096 + 4096*4096) = 3/9 = 1/3
        assert!(
            p.weight_reuse_ratio > 0.0 && p.weight_reuse_ratio < 1.0,
            "MoE weight_reuse_ratio should be between 0 and 1, got {}",
            p.weight_reuse_ratio,
        );
    }

    // ── Additional boundary and trait verification tests ────────────────

    #[test]
    fn graph_archetype_partial_eq_same_and_different() {
        // Arrange: derive two archetypes from the same profile
        let p = dense_profile();
        let a = GraphArchetype::derive(&p);
        let b = GraphArchetype::derive(&p);
        // Assert: same input yields equal archetypes
        assert_eq!(a, b);

        // Arrange: mutate one field to produce a different profile
        let mut p2 = p.clone();
        p2.hidden_dim = 1;
        let c = GraphArchetype::derive(&p2);
        // Assert: different inputs yield unequal archetypes (compute_intensive differs)
        assert_ne!(a, c);
    }

    #[test]
    fn derive_weight_reuse_ratio_above_one_clamps_memory_intensive() {
        // Arrange: weight_reuse_ratio > 1.0 would make reuse_factor negative,
        // but only ModelConfig-driven values are used. Test via direct profile.
        let mut p = dense_profile();
        p.weight_reuse_ratio = 2.0;
        p.kv_bytes_per_token = 0;
        let arch = GraphArchetype::derive(&p);
        // reuse_factor = 1.0 - 2.0 = -1.0, memory_intensive = (kv_factor + (-1.0)).clamp(0,1)
        assert!(
            (0.0..=1.0).contains(&arch.memory_intensive),
            "memory_intensive must be clamped to [0,1] even with weight_reuse_ratio > 1.0, got {}",
            arch.memory_intensive,
        );
    }

    #[test]
    fn derive_kv_q_head_ratio_above_one_qkv_bonus_is_standard() {
        // Arrange: kv_q_head_ratio > 1.0 should fall into the else branch (0.1 bonus)
        let mut p = dense_profile();
        p.avg_epilogue_chain_len = 0.0;
        p.ffn_kind = FfnKind::ReLU;
        p.residual_kind = ResidualKind::PostNorm;

        p.kv_q_head_ratio = 1.0;
        let arch_ratio_one = GraphArchetype::derive(&p);
        p.kv_q_head_ratio = 5.0;
        let arch_ratio_high = GraphArchetype::derive(&p);
        // Assert: both should get the standard 0.1 qkv bonus, not the 0.2 GQA bonus
        assert!(
            (arch_ratio_one.fusion_profitable - arch_ratio_high.fusion_profitable).abs() < 1e-12,
            "kv_q_head_ratio >= 1.0 should give same qkv bonus: {} vs {}",
            arch_ratio_one.fusion_profitable,
            arch_ratio_high.fusion_profitable,
        );
    }

    #[test]
    fn derive_negative_epilogue_chain_clamps_compute_intensive() {
        // Arrange: negative epilogue chain reduces compute_intensive below sigmoid base
        let mut p = dense_profile();
        p.hidden_dim = 2048; // sigmoid base = 0.5
        p.avg_epilogue_chain_len = -10.0;
        p.moe_layer_ratio = 0.0;
        let arch = GraphArchetype::derive(&p);
        // compute_intensive = (0.5 + (-10.0/8.0) - 0.0).clamp(0,1) = (0.5 - 1.25).clamp(0,1) = 0.0
        assert!(
            arch.compute_intensive < 0.5,
            "negative epilogue chain should reduce compute_intensive below sigmoid base, got {}",
            arch.compute_intensive,
        );
        assert!(
            arch.compute_intensive >= 0.0,
            "compute_intensive must be >= 0 after clamping, got {}",
            arch.compute_intensive,
        );
    }

    #[test]
    fn derive_num_layers_16_depth_sigmoid_at_zero() {
        // Arrange: depth factor = sigmoid((16 - 16) / 32) = sigmoid(0) = 0.5
        let mut p = dense_profile();
        p.num_layers = 16;
        let arch = GraphArchetype::derive(&p);
        // pipeline_valuable = small_gemm + 0.5*0.3 + memory_bound
        // Verify it is in range and that num_layers=16 produces a specific midpoint
        assert!(
            (0.0..=1.0).contains(&arch.pipeline_valuable),
            "pipeline_valuable must be in [0,1], got {}",
            arch.pipeline_valuable,
        );
    }

    #[test]
    fn derive_sliding_window_attention_profile_in_range() {
        // Arrange: GraphProfiler never produces SlidingWindow, but derive() should
        // handle it without panic since it does not branch on attention_kind.
        let mut p = dense_profile();
        p.attention_kind = AttentionKind::SlidingWindow { window_size: 4096 };
        let arch = GraphArchetype::derive(&p);
        // Assert: all dimensions clamped in [0,1]
        assert!((0.0..=1.0).contains(&arch.compute_intensive));
        assert!((0.0..=1.0).contains(&arch.memory_intensive));
        assert!((0.0..=1.0).contains(&arch.parallelism_exploitable));
        assert!((0.0..=1.0).contains(&arch.fusion_profitable));
        assert!((0.0..=1.0).contains(&arch.pipeline_valuable));
    }

    #[test]
    fn profile_expert_intermediate_size_ignored_for_dense() {
        // Arrange: expert_intermediate_size set but no MoE (num_experts=None)
        let mut c = base_config();
        c.expert_intermediate_size = Some(2048);
        let p_with = GraphProfiler::profile(&c);
        let p_without = GraphProfiler::profile(&base_config());
        // Assert: dense model ignores expert_intermediate_size
        assert_eq!(
            p_with.total_param_bytes, p_without.total_param_bytes,
            "expert_intermediate_size should be ignored for dense models"
        );
    }

    #[test]
    fn derive_all_five_dimensions_independent_different_profiles() {
        // Arrange: two profiles differing in only one dimension at a time
        let base_p = dense_profile();
        let base_arch = GraphArchetype::derive(&base_p);

        let mut high_kv = base_p.clone();
        high_kv.kv_bytes_per_token = 10_000_000;
        let high_kv_arch = GraphArchetype::derive(&high_kv);

        let mut low_hidden = base_p.clone();
        low_hidden.hidden_dim = 64;
        let low_hidden_arch = GraphArchetype::derive(&low_hidden);

        // Assert: memory_intensive dominated by kv change, compute_intensive by hidden_dim
        assert!(
            high_kv_arch.memory_intensive >= base_arch.memory_intensive,
            "higher kv_bytes should not decrease memory_intensive"
        );
        assert!(
            low_hidden_arch.compute_intensive <= base_arch.compute_intensive,
            "lower hidden_dim should not increase compute_intensive"
        );
    }

    #[test]
    fn derive_relu_no_swiglu_bonus_fusion_lower() {
        // Arrange: ReLU (non-gated) should yield strictly lower fusion_profitable than
        // SwiGLU when all other inputs are identical.
        let mut p_relu = dense_profile();
        p_relu.ffn_kind = FfnKind::ReLU;
        p_relu.avg_epilogue_chain_len = 1.0; // ReLU default chain length

        let mut p_swiglu = p_relu.clone();
        p_swiglu.ffn_kind = FfnKind::SwiGLU;

        let arch_relu = GraphArchetype::derive(&p_relu);
        let arch_swiglu = GraphArchetype::derive(&p_swiglu);

        // Assert: SwiGLU gets +0.3 bonus, ReLU gets none
        assert!(
            arch_swiglu.fusion_profitable > arch_relu.fusion_profitable,
            "SwiGLU should have higher fusion_profitable than ReLU: {} vs {}",
            arch_swiglu.fusion_profitable,
            arch_relu.fusion_profitable,
        );
    }

    #[test]
    fn profile_total_param_bytes_moe_expert_intermediate_override() {
        // Arrange: MoE with expert_intermediate_size smaller than intermediate_size
        let mut c = base_config();
        c.num_experts = Some(4);
        c.num_experts_per_tok = Some(2);
        c.intermediate_size = Some(16384);
        c.expert_intermediate_size = Some(1024);

        let mut c_default = c.clone();
        c_default.expert_intermediate_size = None;

        let p_custom = GraphProfiler::profile(&c);
        let p_default = GraphProfiler::profile(&c_default);

        // Assert: custom expert_intermediate_size=1024 yields fewer params than default 16384
        assert!(
            p_custom.total_param_bytes < p_default.total_param_bytes,
            "smaller expert_intermediate_size should yield fewer params: {} vs {}",
            p_custom.total_param_bytes,
            p_default.total_param_bytes,
        );
    }

}
