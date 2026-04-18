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

/// Classifier head weight aliases for encoder-based classification/reranker models.
/// Returns aliases for the classifier dense layer and output projection.
///
/// Common HuggingFace patterns:
/// - `classifier.dense.weight` / `classifier.dense.bias` (BERT-style pooler→dense)
/// - `classifier.out_proj.weight` / `classifier.out_proj.bias` (final projection)
/// - `classifier.weight` / `classifier.bias` (single-layer classifier)
/// - `pre_classifier.weight` / `pre_classifier.bias` (DistilBERT-style)
pub fn classifier_aliases(suffix: &str) -> Vec<String> {
    vec![
        format!("classifier.{suffix}"),
        format!("classifier.dense.{suffix}"),
        format!("classifier.out_proj.{suffix}"),
        format!("pre_classifier.{suffix}"),
    ]
}

/// Pooler (CLS → dense) weight aliases for encoder-based classifier models.
/// The pooler transforms CLS token hidden state before the classifier head.
pub fn pooler_aliases(suffix: &str) -> Vec<String> {
    vec![
        format!("pooler.dense.{suffix}"),
        format!("pooler.{suffix}"),
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
    let mut out = Vec::with_capacity(DECODER_PREFIXES.len() + 2);
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
    out.push(format!("h.{layer}.{suffix}"));
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

/// Generate alias names for the final norm bias (decoder LayerNorm).
pub fn decoder_final_norm_bias_aliases() -> Vec<String> {
    vec![]
}

/// Generate alias names for the lm_head weight in decoder models.
///
/// 权重绑定规则:
/// 1. `lm_head.weight` / `output.weight` 是独立 lm_head 权重名。
/// 2. 许多模型 (SmolLM2 / TinyLlama / 原生 Llama config.tie_word_embeddings=true /
///    llama.cpp GGUF 默认) 会把 lm_head 和 token_embedding 张量捆绑为同一块权重,
///    此时 safetensors/gguf 里不存在独立的 `lm_head.weight`,必须回退到 embed
///    张量的各种别名: `token_embd.weight` (GGUF), `model.embed_tokens.weight`
///    (HF SafeTensors), `embed_tokens.weight` (裸前缀), `roberta./bert.` 等。
///
/// 生成顺序: 先原生 lm_head 名,再跨类别回退到 embed 别名,保证"有就用自己的,
/// 没有就用 tied embed"的语义。
pub fn lm_head_aliases() -> Vec<String> {
    let mut out = vec![
        "lm_head.weight".to_string(),
        "output.weight".to_string(),
    ];
    // Tied-weights fallback: embed 张量的所有候选名称。
    out.extend(decoder_embed_aliases());
    out
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

// ---------------------------------------------------------------------------
// HF → GGUF decoder layer name translation
// ---------------------------------------------------------------------------

/// Mapping from HuggingFace-style decoder layer weight suffix to GGUF llama.cpp suffix.
///
/// Key: HF suffix (e.g. `"self_attn.q_proj.weight"`)
/// Value: GGUF suffix (e.g. `"attn_q.weight"`)
static HF_TO_GGUF_LAYER_SUFFIX: &[(&str, &str)] = &[
    ("self_attn.q_proj.weight",          "attn_q.weight"),
    ("self_attn.k_proj.weight",          "attn_k.weight"),
    ("self_attn.v_proj.weight",          "attn_v.weight"),
    ("self_attn.o_proj.weight",          "attn_output.weight"),
    ("self_attn.q_proj.bias",            "attn_q.bias"),
    ("self_attn.k_proj.bias",            "attn_k.bias"),
    ("self_attn.v_proj.bias",            "attn_v.bias"),
    ("self_attn.o_proj.bias",            "attn_output.bias"),
    ("mlp.gate_proj.weight",             "ffn_gate.weight"),
    ("mlp.up_proj.weight",               "ffn_up.weight"),
    ("mlp.down_proj.weight",             "ffn_down.weight"),
    ("mlp.gate_proj.bias",               "ffn_gate.bias"),
    ("mlp.up_proj.bias",                 "ffn_up.bias"),
    ("mlp.down_proj.bias",               "ffn_down.bias"),
    ("input_layernorm.weight",           "attn_norm.weight"),
    ("input_layernorm.bias",             "attn_norm.bias"),
    ("post_attention_layernorm.weight",  "ffn_norm.weight"),
    ("post_attention_layernorm.bias",    "ffn_norm.bias"),
    // Qwen3/Llama norm variants
    ("post_feedforward_layernorm.weight","post_ffw_norm.weight"),
    ("pre_feedforward_layernorm.weight", "pre_ffw_norm.weight"),
    // Attention norm (Qwen2.5 etc.)
    ("self_attn.q_norm.weight",          "attn_q_norm.weight"),
    ("self_attn.k_norm.weight",          "attn_k_norm.weight"),
];

/// Given a canonical weight name (HuggingFace or GGUF style), generate all possible
/// aliases: every HF prefix variant plus the GGUF `blk.{N}.xxx` name.
///
/// Returns an empty Vec if the name does not match a recognized decoder layer pattern.
pub fn all_decoder_weight_aliases(name: &str) -> Vec<String> {
    // Parse `{optional_model_prefix}.layers.{N}.{suffix}`
    // Accepted leading segments: "" | "model" | "layers" directly
    let layers_part = {
        let mut found = None;
        for &prefix in DECODER_PREFIXES {
            let expected = if prefix.is_empty() {
                format!("layers.")
            } else {
                format!("{prefix}.layers.")
            };
            if let Some(rest) = name.strip_prefix(&expected) {
                found = Some(rest);
                break;
            }
        }
        found
    };
    let Some(layers_rest) = layers_part else { return vec![] };

    // Parse `{N}.{suffix}`
    let Some(dot_pos) = layers_rest.find('.') else { return vec![] };
    let layer_str = &layers_rest[..dot_pos];
    let suffix = &layers_rest[dot_pos + 1..];
    let Ok(layer) = layer_str.parse::<usize>() else { return vec![] };

    // Generate all HF variants via decoder_layer_aliases
    // Then look up GGUF suffix
    let gguf_suffix = HF_TO_GGUF_LAYER_SUFFIX
        .iter()
        .find(|&&(hf, _)| hf == suffix)
        .map(|&(_, gg)| gg);

    decoder_layer_aliases(layer, suffix, gguf_suffix)
}

// ---------------------------------------------------------------------------
// PLE (Per-Layer Embedding) weight name aliases — Gemma 4 E2B/E4B
// ---------------------------------------------------------------------------

/// Per-Layer Embedding token weight: `model.per_layer_embedding.embed_tokens.weight`
/// Shape: [vocab_size, num_layers × dim_per_layer]
pub const PLE_EMBED_TOKENS: &str = "model.per_layer_embedding.embed_tokens.weight";

/// Per-Layer Embedding context-aware projection: `model.per_layer_embedding.per_layer_projection.weight`
pub const PLE_PROJECTION: &str = "model.per_layer_embedding.per_layer_projection.weight";

/// Per-Layer Embedding post-MLP projection prefix: `model.layers.{i}.post_mlp_projection.weight`
pub const PLE_POST_MLP_PROJ_PREFIX: &str = "post_mlp_projection.weight";

/// Generate alias names for PLE embed_tokens weight.
pub fn ple_embed_tokens_aliases() -> Vec<String> {
    let mut out = Vec::new();
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push("per_layer_embedding.embed_tokens.weight".to_string());
        } else {
            out.push(format!("{prefix}.per_layer_embedding.embed_tokens.weight"));
        }
    }
    out
}

/// Generate alias names for PLE per_layer_projection weight.
pub fn ple_projection_aliases() -> Vec<String> {
    let mut out = Vec::new();
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push("per_layer_embedding.per_layer_projection.weight".to_string());
        } else {
            out.push(format!("{prefix}.per_layer_embedding.per_layer_projection.weight"));
        }
    }
    out
}

/// Generate alias names for the per-layer post_mlp_projection weight.
///
/// `layer`: layer index (0-based)
pub fn ple_post_mlp_proj_aliases(layer: usize) -> Vec<String> {
    decoder_layer_aliases(layer, PLE_POST_MLP_PROJ_PREFIX, None)
}

/// 给定 **任意** canonical PLE 权重名 (global 两种 / per-layer post_mlp_projection),
/// 返回其所有等价别名 (含 HF `model.` 前缀 / 裸前缀 / decoder layer 不同前缀)。
///
/// 语义对齐 `all_decoder_weight_aliases`: 由 `bind_weight_shapes_fuzzy` 在 exact
/// match 失败后调用, 用于把 YAML template 侧的 canonical 名字桥接到 provider 实际
/// 存储的名字。不匹配任何 PLE 模式时返回空 Vec, 保持调用方的回退链行为。
pub fn all_ple_weight_aliases(name: &str) -> Vec<String> {
    // Global PLE embed table: `model.per_layer_embedding.embed_tokens.weight` / 裸前缀。
    let ple_embed = ple_embed_tokens_aliases();
    if ple_embed.iter().any(|a| a == name) {
        return ple_embed;
    }

    // Global PLE projection: `model.per_layer_embedding.per_layer_projection.weight` / 裸前缀。
    let ple_proj = ple_projection_aliases();
    if ple_proj.iter().any(|a| a == name) {
        return ple_proj;
    }

    // Per-layer `post_mlp_projection.weight`: 形如
    //   `model.layers.{N}.post_mlp_projection.weight` 或 `layers.{N}.post_mlp_projection.weight`
    // 复用 `decoder_layer_aliases` 统一生成所有前缀变体 (含 GGUF `blk.{N}.`)。
    for &prefix in DECODER_PREFIXES {
        let base = if prefix.is_empty() {
            "layers.".to_string()
        } else {
            format!("{prefix}.layers.")
        };
        if let Some(rest) = name.strip_prefix(&base) {
            // `{N}.post_mlp_projection.weight`
            let Some(dot_pos) = rest.find('.') else { continue; };
            let layer_str = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];
            if suffix != PLE_POST_MLP_PROJ_PREFIX {
                continue;
            }
            let Ok(layer) = layer_str.parse::<usize>() else { continue; };
            return ple_post_mlp_proj_aliases(layer);
        }
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ple_embed_tokens_aliases_contains_canonical_and_bare() {
        let aliases = ple_embed_tokens_aliases();
        assert!(aliases.iter().any(|a| a == "per_layer_embedding.embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "model.per_layer_embedding.embed_tokens.weight"));
        assert_eq!(aliases.len(), DECODER_PREFIXES.len());
    }

    #[test]
    fn ple_projection_aliases_contains_canonical_and_bare() {
        let aliases = ple_projection_aliases();
        assert!(aliases.iter().any(|a| a == "per_layer_embedding.per_layer_projection.weight"));
        assert!(aliases.iter().any(|a| a == "model.per_layer_embedding.per_layer_projection.weight"));
        assert_eq!(aliases.len(), DECODER_PREFIXES.len());
    }

    #[test]
    fn ple_post_mlp_proj_aliases_covers_all_prefixes() {
        let aliases = ple_post_mlp_proj_aliases(5);
        assert!(aliases.iter().any(|a| a == "layers.5.post_mlp_projection.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.5.post_mlp_projection.weight"));
        assert!(aliases.iter().any(|a| a == "h.5.post_mlp_projection.weight"));
    }

    #[test]
    fn all_ple_weight_aliases_global_embed() {
        let from_canonical =
            all_ple_weight_aliases("model.per_layer_embedding.embed_tokens.weight");
        assert!(from_canonical.iter().any(|a| a == "per_layer_embedding.embed_tokens.weight"));
        assert!(from_canonical
            .iter()
            .any(|a| a == "model.per_layer_embedding.embed_tokens.weight"));

        let from_bare = all_ple_weight_aliases("per_layer_embedding.embed_tokens.weight");
        assert!(from_bare
            .iter()
            .any(|a| a == "model.per_layer_embedding.embed_tokens.weight"));
    }

    #[test]
    fn all_ple_weight_aliases_global_projection() {
        let aliases =
            all_ple_weight_aliases("model.per_layer_embedding.per_layer_projection.weight");
        assert!(aliases.iter().any(|a| a == "per_layer_embedding.per_layer_projection.weight"));
        assert!(aliases
            .iter()
            .any(|a| a == "model.per_layer_embedding.per_layer_projection.weight"));
    }

    #[test]
    fn all_ple_weight_aliases_per_layer() {
        let aliases = all_ple_weight_aliases("model.layers.3.post_mlp_projection.weight");
        assert!(aliases.iter().any(|a| a == "layers.3.post_mlp_projection.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.3.post_mlp_projection.weight"));
        assert!(aliases.iter().any(|a| a == "h.3.post_mlp_projection.weight"));

        let bare = all_ple_weight_aliases("layers.7.post_mlp_projection.weight");
        assert!(bare.iter().any(|a| a == "model.layers.7.post_mlp_projection.weight"));
    }

    #[test]
    fn all_ple_weight_aliases_non_ple_returns_empty() {
        // Regular decoder weights must not be intercepted by the PLE matcher.
        assert!(all_ple_weight_aliases("model.layers.0.self_attn.q_proj.weight").is_empty());
        assert!(all_ple_weight_aliases("lm_head.weight").is_empty());
        assert!(all_ple_weight_aliases("model.embed_tokens.weight").is_empty());
    }
}
