//! Centralized weight name aliases for different model architectures.
//!
//! All architecture-specific naming conventions are defined here.
//! Adding a new architecture (e.g., XLM-RoBERTa) = adding ONE prefix entry.
//! No logic code needs to change.

/// Known architecture prefixes for BERT-family models.
///
/// - `""` (empty): bare names like `embeddings.word_embeddings.weight`
/// - `"bert"`: BERT, MiniLM, etc.
/// - `"model"`: some HuggingFace exports
/// - `"roberta"`: XLM-RoBERTa, BGE-reranker, etc.
const ENCODER_PREFIXES: &[&str] = &["", "bert", "model", "roberta"];

/// Generate all alias names for an embedding-level weight.
///
/// `suffix`: e.g. `"word_embeddings.weight"`, `"position_embeddings.weight"`
/// `gguf_name`: GGUF-specific name, e.g. `"token_embd.weight"`
pub fn embedding_aliases(suffix: &str, gguf_name: Option<&str>) -> Vec<String> {
    let mut out = Vec::with_capacity(ENCODER_PREFIXES.len() + 1);
    for &prefix in ENCODER_PREFIXES {
        if prefix.is_empty() {
            out.push(format!("embeddings.{suffix}"));
        } else {
            out.push(format!("{prefix}.embeddings.{suffix}"));
        }
    }
    if let Some(gg) = gguf_name {
        out.push(gg.to_string());
    }
    out
}

/// Generate all alias names for an encoder layer weight.
///
/// `layer`: layer index (0-based)
/// `suffix`: e.g. `"attention.self.query.weight"`
/// `gguf_suffix`: e.g. `"attn_q.weight"`
pub fn layer_aliases(layer: usize, suffix: &str, gguf_suffix: Option<&str>) -> Vec<String> {
    let mut out = Vec::with_capacity(ENCODER_PREFIXES.len() + 1);
    for &prefix in ENCODER_PREFIXES {
        if prefix.is_empty() {
            out.push(format!("encoder.layer.{layer}.{suffix}"));
        } else {
            out.push(format!("{prefix}.encoder.layer.{layer}.{suffix}"));
        }
    }
    if let Some(gg) = gguf_suffix {
        out.push(format!("blk.{layer}.{gg}"));
    }
    out
}

/// Generate all architecture prefix strings for encoder layer N.
///
/// Returns: `["encoder.layer.{N}", "bert.encoder.layer.{N}", "model.encoder.layer.{N}", "roberta.encoder.layer.{N}"]`
pub fn layer_prefixes(layer: usize) -> Vec<String> {
    ENCODER_PREFIXES
        .iter()
        .map(|&p| {
            if p.is_empty() {
                format!("encoder.layer.{layer}")
            } else {
                format!("{p}.encoder.layer.{layer}")
            }
        })
        .collect()
}

/// GGUF layer prefix: `"blk.{layer}"`
pub fn gguf_layer_prefix(layer: usize) -> String {
    format!("blk.{layer}")
}

/// Check if weights contain any known embedding weight (for transpose detection).
pub fn has_any_embedding_weight(checker: impl Fn(&str) -> bool) -> bool {
    embedding_aliases("word_embeddings.weight", Some("token_embd.weight"))
        .iter()
        .any(|n| checker(n))
}

/// Check if any intermediate dense weight exists (for transpose detection).
pub fn has_any_intermediate_weight(checker: impl Fn(&str) -> bool) -> bool {
    layer_aliases(0, "intermediate.dense.weight", Some("ffn_up.weight"))
        .iter()
        .any(|n| checker(n))
}

/// Classifier head weight aliases for reranker models.
/// Returns aliases for the classifier dense layer and output projection.
pub fn classifier_aliases(suffix: &str) -> Vec<String> {
    vec![
        format!("classifier.{suffix}"),
    ]
}

// ---------------------------------------------------------------------------
// Decoder (LLaMA/Qwen-style) weight name aliases
// ---------------------------------------------------------------------------

use crate::manifest::ArchFamily;

/// Known architecture prefixes for decoder-only models.
const DECODER_PREFIXES: &[&str] = &["", "model"];

/// Unified entry point: get architecture prefixes by family.
pub fn arch_prefixes(family: ArchFamily) -> &'static [&'static str] {
    match family {
        ArchFamily::Encoder => ENCODER_PREFIXES,
        ArchFamily::Decoder => DECODER_PREFIXES,
    }
}

/// Generate all alias names for a decoder layer weight.
///
/// `layer`: layer index (0-based)
/// `suffix`: e.g. `"self_attn.q_proj.weight"`
/// `gguf_suffix`: e.g. `"attn_q.weight"`
pub fn decoder_layer_aliases(layer: usize, suffix: &str, gguf_suffix: Option<&str>) -> Vec<String> {
    let mut out = Vec::with_capacity(DECODER_PREFIXES.len() + 1);
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push(format!("layers.{layer}.{suffix}"));
        } else {
            out.push(format!("{prefix}.layers.{layer}.{suffix}"));
        }
    }
    if let Some(gg) = gguf_suffix {
        out.push(format!("blk.{layer}.{gg}"));
    }
    out
}

/// Generate alias names for the token embedding in decoder models.
pub fn decoder_embed_aliases() -> Vec<String> {
    let mut out = vec![];
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push("embed_tokens.weight".to_string());
        } else {
            out.push(format!("{prefix}.embed_tokens.weight"));
        }
    }
    out.push("token_embd.weight".to_string());
    out
}

/// Generate alias names for the final RMS norm in decoder models.
pub fn decoder_final_norm_aliases() -> Vec<String> {
    let mut out = vec![];
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push("norm.weight".to_string());
        } else {
            out.push(format!("{prefix}.norm.weight"));
        }
    }
    out.push("output_norm.weight".to_string());
    out
}

/// Generate alias names for the lm_head weight in decoder models.
pub fn lm_head_aliases() -> Vec<String> {
    vec![
        "lm_head.weight".to_string(),
        "output.weight".to_string(),
    ]
}

/// Score head weight aliases for decoder-based reranker models (e.g. Qwen3ForSequenceClassification).
///
/// NOTE: `output.weight` is intentionally NOT included here because it is ambiguous:
/// - For `Qwen3ForSequenceClassification`: llama.cpp maps `score.weight` → `output.weight` (shape [num_labels, hidden])
/// - For standard `Qwen3` (generative): `lm_head.weight` → `output.weight` (shape [vocab_size, hidden])
///
/// The disambiguation is handled in `decoder_rerank_forward()` via size-based heuristic.
pub fn decoder_score_aliases() -> Vec<String> {
    vec![
        "score.weight".to_string(),
        "classifier.weight".to_string(),
        "cls.weight".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// MoE (Mixture of Experts) weight name aliases
// ---------------------------------------------------------------------------

/// Generate alias names for the MoE router/gate weight in a decoder layer.
pub fn moe_gate_aliases(layer: usize) -> Vec<String> {
    let mut out = Vec::new();
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push(format!("layers.{layer}.mlp.gate.weight"));
        } else {
            out.push(format!("{prefix}.layers.{layer}.mlp.gate.weight"));
        }
    }
    out.push(format!("blk.{layer}.ffn_gate_inp.weight"));
    out
}

/// Generate alias names for a MoE shared expert weight.
///
/// `suffix`: e.g. `"gate_proj.weight"`, `"up_proj.weight"`, `"down_proj.weight"`
pub fn moe_shared_expert_aliases(layer: usize, suffix: &str) -> Vec<String> {
    let mut out = Vec::new();
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push(format!("layers.{layer}.mlp.shared_experts.{suffix}"));
        } else {
            out.push(format!("{prefix}.layers.{layer}.mlp.shared_experts.{suffix}"));
        }
    }
    out.push(format!("blk.{layer}.ffn_shared_expert.{suffix}"));
    out
}

/// Generate alias names for a MoE routed expert weight.
///
/// `expert`: expert index (0-based)
/// `suffix`: e.g. `"gate_proj.weight"`, `"up_proj.weight"`, `"down_proj.weight"`
pub fn moe_expert_aliases(layer: usize, expert: usize, suffix: &str) -> Vec<String> {
    let mut out = Vec::new();
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push(format!("layers.{layer}.mlp.experts.{expert}.{suffix}"));
        } else {
            out.push(format!("{prefix}.layers.{layer}.mlp.experts.{expert}.{suffix}"));
        }
    }
    out.push(format!("blk.{layer}.ffn_gate_exps.{expert}.{suffix}"));
    out
}
