// ---------------------------------------------------------------------------
// Architecture Features — derived from role index
// ---------------------------------------------------------------------------

/// Weight-topology family — BUILD-stage Strategy Pattern decision.
///
/// This enum classifies model weight topology (which tensors exist) to select
/// the appropriate graph-building strategy. It is NOT a compiler branch:
/// the JIT compiler processes whatever graph it receives, regardless of family.
/// Family drives only the `build_compiler_graph` strategy selection at model
/// load time, determining which ops to emit (e.g., MeanPool for Encoder,
/// Argmax for Decoder).
///
/// Per SPEC/39 §0.1: the compiler emits exactly what the graph contains;
/// family never appears in compiled machine code.
#[derive(Debug, Clone, PartialEq)]
pub enum Family {
    /// Weight topology with OutputHead or FinalNorm — typically causal-attention
    /// generator models (chat, base LLM). Selects generate-loop graph strategy.
    Decoder,
    /// Weight topology without OutputHead — typically bidirectional-attention
    /// encoder models (embedding, reranker, classifier). Selects single-pass
    /// graph strategy (MeanPool/Classify output instead of Argmax loop).
    Encoder,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FfnType {
    SwiGLU,
    GeGLU,
    Standard,
    MoE,
}

/// Architecture features derived purely from tensor name presence/absence.
///
/// Every field is determined by exact role-index lookups — no heuristics.
#[derive(Debug)]
pub struct ArchitectureFeatures {
    /// Weight-topology family — BUILD-stage strategy selector (not a compiler branch).
    /// Determines graph-building strategy: Decoder → generate loop, Encoder → single pass.
    pub family: Family,
    pub num_layers: usize,

    // Attention
    pub has_rope: bool,
    pub has_head_rms_norm: bool,
    pub has_attention_bias: bool,
    pub attention_sinks: bool,
    pub has_qk_norm: bool,
    pub has_value_norm: bool,
    pub has_per_layer_embedding: bool,
    pub hidden_size_per_layer_input: usize,
    /// AltUp (Alternating Updates) number of parallel prediction paths (P).
    /// Gemma 4 E2B/E4B: P=2. Non-AltUp models: P=0.
    pub altup_num_inputs: usize,
    pub has_embedding_scale: bool,

    // Norm
    pub norm_type: NormType,

    // FFN
    pub ffn_type: FfnType,

    // MoE
    pub is_moe: bool,
    pub has_shared_experts: bool,
    pub num_experts: usize,
    pub moe_top_k: usize,

    // MLA (Multi-head Latent Attention)
    pub is_mla: bool,
    pub mla_latent_dim: usize,
    pub mla_rope_dim: usize,
    /// Build un-absorbed path (K/V restore → standard MHA) instead of absorbed.
    /// Used for short prefill where compute is saturated (REQ-MLA-004).
    pub mla_use_unabsorbed: bool,

    // Special
    pub has_classifier: bool,
    /// Post-norm architecture (BERT/XLM-R): norm applied after residual, not before.
    /// Detected by: presence of `TensorRole::EmbedNorm` (embeddings.LayerNorm weight).
    pub is_post_norm: bool,
    /// Causal attention: mask future positions. Post-norm models use bidirectional
    /// (causal=false); pre-norm models use causal (causal=true).
    pub causal: bool,
    /// Absolute position embeddings (BERT): learned `position_embed` tensor.
    /// Decoders use RoPE instead (has_absolute_position_embed=false).
    pub has_absolute_position_embed: bool,

    // ── Heterogeneous layers (Gemma 4 E2B/E4B) ──
    /// True when model has both sliding + global attention AND different FFN sizes across segments.
    /// Detected by: global_head_dim > 0 && global_head_dim != head_dim
    pub is_hetero_layer: bool,
    /// Head dim for sliding attention layers (Gemma 4: 256).
    pub sliding_head_dim: usize,
    /// Number of Q heads for sliding layers (Gemma 4 E2B: 4).
    pub sliding_num_q_heads: usize,
    /// Head dim for global attention layers (Gemma 4: 512).
    pub full_head_dim: usize,
    /// Number of Q heads for full layers (Gemma 4 E2B: 8).
    pub full_num_q_heads: usize,
    /// FFN intermediate size for small-FFN segment group (Gemma 4 E2B: 6144).
    pub small_intermediate: usize,
    /// FFN intermediate size for large-FFN segment group (Gemma 4 E2B: 12288).
    pub large_intermediate: usize,
    /// First segment index (0-based) using large FFN. Segments < this use small FFN.
    pub large_ffn_start_segment: usize,
    /// Number of attention segments. Gemma 4 E2B: 7.
    pub num_segments: usize,
    /// Number of sliding layers per segment (before the global layer). Gemma 4: 4.
    pub sliding_per_segment: usize,
}

/// Analyze architecture features from a tensor role index.
///
/// `role_index`: maps `(TensorRole, layer_idx)` → tensor name
/// `weight_shapes`: maps tensor name → shape
/// `arch_name`: optional canonical architecture name (e.g., "gemma4", "qwen3")
/// `hints`: optional BUILD-stage architecture hints from ModelConfig (REQ-MC-EXT-001..007).
///          When None, features not derivable from tensor topology default to "off".
pub fn analyze_architecture(
    role_index: &HashMap<(TensorRole, Option<usize>), String>,
    weight_shapes: &HashMap<String, Vec<usize>>,
    _arch_name: Option<&str>,
    hints: Option<&ArchHints>,
) -> ArchitectureFeatures {
    // ── Family (BUILD-stage strategy selection — not a compiler branch) ──
    let has_output_head = role_index.contains_key(&(TensorRole::OutputHead, None));
    let has_classifier = role_index.contains_key(&(TensorRole::ClassifierDense, None))
        || role_index.contains_key(&(TensorRole::ClassifierOutProj, None));
    let _has_final_norm = role_index.contains_key(&(TensorRole::FinalNorm, None));

    // Weight-topology heuristic:
    //   OutputHead (lm_head) → Decoder (generate loop, KV cache FromCache)
    //   No OutputHead → Encoder (single pass, KV cache FromTensor)
    //
    //   FinalNorm alone is ambiguous: present in both decoder base models
    //   (e.g. Qwen3-0.6B-Base) and encoder models (e.g. Qwen3-Embedding).
    //   The disambiguator is OutputHead: only decoders have lm_head.
    //   Encoder/embedding/reranker models have FinalNorm (output_norm) but
    //   no OutputHead — they produce embeddings, not tokens.
    //
    //   OE-4 root cause: Qwen3-Embedding has output_norm.weight (FinalNorm)
    //   but no lm_head.weight (OutputHead). The old heuristic treated
    //   FinalNorm alone as Decoder, causing kv_source=FromCache and
    //   MemCopy to NULL kv_cache_ptr → SIGSEGV.
    let family = if has_output_head {
        Family::Decoder
    } else {
        Family::Encoder
    };

    // ── Num layers ──
    let num_layers = role_index.keys()
        .filter_map(|(_, layer_idx)| *layer_idx)
        .max()
        .map(|idx| idx + 1)
        .unwrap_or(0);

    // ── Attention features ──
    let has_head_rms_norm = role_index.contains_key(&(TensorRole::AttentionQNorm, Some(0)));
    let has_attention_bias = weight_shapes.keys().any(|n| {
        let lower = n.to_ascii_lowercase();
        lower.ends_with(".bias") && (lower.contains("q_proj") || lower.contains("k_proj") || lower.contains("v_proj"))
    });
    let attention_sinks = role_index.contains_key(&(TensorRole::AttentionSinks, Some(0)));

    // QkNorm: config-driven (REQ-MC-EXT-001).
    // When hints provide explicit value, use it. Otherwise derive from tensor topology:
    // no AttentionQNorm/KNorm weight tensors present → pure L2 normalization (Gemma 4 style).
    let has_qk_norm = hints.and_then(|h| h.qk_norm).unwrap_or(false);

    // ValueNorm: config-driven (REQ-MC-EXT-002).
    // Default false; Gemma 4 config provides true via ArchHints.
    let has_value_norm = hints.and_then(|h| h.value_norm).unwrap_or(false);

    // Embedding scale: config-driven (REQ-MC-EXT-003).
    // >0 → has_embedding_scale=true.
    let has_embedding_scale = hints
        .and_then(|h| h.embedding_scale_factor)
        .map(|f| f > 0.0)
        .unwrap_or(false);

    // ── PerLayerEmbedding (Gemma 4 E2B/E4B) ──
    // Detection: presence of PLE weight tensors in weight_shapes.
    // Gemma 4 E2B/E4B have `embed_tokens_per_layer` / `per_layer_embedding` tensors;
    // Gemma 4 31B Dense / 26B MoE do not.
    let has_per_layer_embedding = weight_shapes.keys().any(|n| {
        n.contains("embed_tokens_per_layer") || n.contains("per_layer_embedding")
            || n.contains("per_layer_token_embd")
    });
    // hidden_size_per_layer_input: dimension of per-layer input signal.
    // Derived from PLE projection weight shape [hidden_size, hidden_size_per_layer_input].
    let hidden_size_per_layer_input = if has_per_layer_embedding {
        weight_shapes.keys()
            .filter(|n| n.contains("per_layer_projection") || n.contains("per_layer_model_projection")
                || n.contains("per_layer_model_proj"))
            .filter_map(|n| weight_shapes.get(n))
            .filter_map(|shape| shape.last().copied())
            .next()
            .unwrap_or(0)
    } else {
        0
    };

    // ── AltUp (Alternating Updates) — Gemma 4 E2B/E4B ──
    // Detection: presence of `altup.` prefixed weight tensors OR
    // GGUF-style AltUp correction weights (`correction_coefs` / `altup_corrections`).
    // GGUF Gemma 4 E2B has PLE weights but NO AltUp weights — has_altup must be false.
    let has_altup = weight_shapes.keys().any(|n| {
        n.contains("altup.") || n.contains("correction_coefs") || n.contains("altup_corrections")
    });
    let altup_num_inputs = if has_altup {
        // Derive P from correction_coefs.weight shape [P, P] or modality_router.weight [P, H].
        weight_shapes.keys()
            .filter(|n| n.contains("correction_coefs") || n.contains("altup.correction"))
            .filter_map(|n| weight_shapes.get(n))
            .filter_map(|shape| shape.first().copied())
            .next()
            .unwrap_or(2)
    } else {
        0
    };

    // ── Topology-derived: absolute position embeddings (BERT) vs RoPE (decoders) ──
    let has_absolute_position_embed = role_index.contains_key(&(TensorRole::PositionEmbedding, None));

    // ── Topology-derived: post-norm (BERT/XLM-R) vs pre-norm (decoders) ──
    // Detected by presence of EmbedNorm (embeddings.LayerNorm weight).
    // Post-norm: InputNorm applied after attn residual; Pre-norm: InputNorm applied before attention.
    let is_post_norm = role_index.contains_key(&(TensorRole::EmbedNorm, None));

    // ── Topology-derived: causal attention ──
    // Post-norm models (BERT/XLM-R) use bidirectional attention (causal=false).
    // Pre-norm models (decoders) use causal masking (causal=true).
    let causal = !is_post_norm;

    // ── RoPE: pre-norm models with attention keys have RoPE; post-norm models do not ──
    let has_rope = !is_post_norm && role_index.contains_key(&(TensorRole::AttentionKey, Some(0)));

    // ── Norm type ──
    // Check if layer 0 InputNorm has a corresponding bias tensor
    let norm_type = if let Some(norm_name) = role_index.get(&(TensorRole::InputNorm, Some(0))) {
        let bias_name = norm_name.replace(".weight", ".bias");
        if weight_shapes.contains_key(&bias_name) {
            NormType::LayerNorm
        } else {
            NormType::RmsNorm
        }
    } else {
        // Default for post-norm: LayerNorm; for pre-norm: RmsNorm
        if is_post_norm {
            NormType::LayerNorm
        } else {
            NormType::RmsNorm
        }
    };

    // ── FFN type ──
    let has_gate = role_index.contains_key(&(TensorRole::FfnGate, Some(0)));
    let has_up = role_index.contains_key(&(TensorRole::FfnUp, Some(0)));
    let has_down = role_index.contains_key(&(TensorRole::FfnDown, Some(0)));
    let is_moe = role_index.contains_key(&(TensorRole::MoEGate, Some(0)));

    let ffn_type = if is_moe {
        FfnType::MoE
    } else if has_gate && has_up && has_down {
        // REQ-MC-EXT-007: FFN type from hidden_act, not is_gemma4.
        match hints.and_then(|h| h.hidden_act.as_ref()) {
            Some(HiddenAct::GeluNew) | Some(HiddenAct::Gelu) => FfnType::GeGLU,
            _ => FfnType::SwiGLU,  // Silu/Swish/None → SwiGLU (most common default)
        }
    } else if has_gate && has_down {
        // gate_up_proj fused (Phi4 style) + down
        FfnType::SwiGLU
    } else {
        FfnType::Standard
    };

    // ── MoE ──
    let has_shared_experts = role_index.keys().any(|(role, _)| *role == TensorRole::MoESharedExpert);

    // num_experts from MoEGate router weight shape [hidden, num_experts]
    let num_experts = role_index.get(&(TensorRole::MoEGate, Some(0)))
        .and_then(|name| weight_shapes.get(name))
        .map(|shape| if shape.len() >= 2 { shape[1] } else { shape[0] })
        .unwrap_or(0);

    // moe_top_k: not derivable from tensor shapes alone.
    // Default to 2 (Mixtral/DeepSeek common); executor overrides from config.
    let moe_top_k = if num_experts > 0 { 2 } else { 0 };

    // ── MLA (Multi-head Latent Attention) ──
    let is_mla = role_index.keys().any(|(role, _)| {
        matches!(role, TensorRole::MlaKvCompress | TensorRole::MlaKeyAbsorb)
    });
    let mla_latent_dim = if is_mla {
        role_index.get(&(TensorRole::MlaKvCompress, Some(0)))
            .and_then(|name| weight_shapes.get(name))
            .map(|shape| if shape.len() >= 2 { shape[1] } else { shape[0] })
            .unwrap_or(0)
    } else {
        0
    };
    let mla_rope_dim = if is_mla {
        role_index.get(&(TensorRole::MlaRopeKey, Some(0)))
            .and_then(|name| weight_shapes.get(name))
            .map(|shape| if shape.len() >= 2 { shape[1] } else { shape[0] })
            .unwrap_or(0)
    } else {
        0
    };


    ArchitectureFeatures {
        family,
        num_layers,
        has_rope,
        has_head_rms_norm,
        has_attention_bias,
        attention_sinks,
        has_qk_norm,
        has_value_norm,
        has_per_layer_embedding,
        hidden_size_per_layer_input,
        altup_num_inputs,
        has_embedding_scale,
        norm_type,
        ffn_type,
        is_moe,
        has_shared_experts,
        num_experts,
        moe_top_k,
        is_mla,
        mla_latent_dim,
        mla_rope_dim,
        mla_use_unabsorbed: hints.and_then(|h| h.mla_use_unabsorbed).unwrap_or(false),
        has_classifier,
        is_post_norm,
        causal,
        has_absolute_position_embed,
        // Hetero fields — populated after analyze_architecture by caller
        is_hetero_layer: false,
        sliding_head_dim: 0,
        sliding_num_q_heads: 0,
        full_head_dim: 0,
        full_num_q_heads: 0,
        small_intermediate: 0,
        large_intermediate: 0,
        large_ffn_start_segment: 0,
        num_segments: 0,
        sliding_per_segment: 0,
    }
}

// ---------------------------------------------------------------------------
// AutoGraphBuilder — CompilerGraph from features + weight_shapes
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum GraphBuildError {
    MissingTensor(String),
    InvalidDimension(String),
    UnsupportedArchitecture(String),
}

impl std::fmt::Display for GraphBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTensor(s) => write!(f, "missing tensor: {s}"),
            Self::InvalidDimension(s) => write!(f, "invalid dimension: {s}"),
            Self::UnsupportedArchitecture(s) => write!(f, "unsupported architecture: {s}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical name helpers
// ---------------------------------------------------------------------------

