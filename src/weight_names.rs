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

/// Given a canonical encoder weight name, generate all possible aliases
/// across all known encoder prefixes.
///
/// Handles two patterns:
/// 1. Embedding-level: `{prefix}.embeddings.{suffix}`
/// 2. Layer-level: `{prefix}.encoder.layer.{N}.{suffix}`
///
/// Returns all variants including the original name and GGUF equivalents.
pub fn all_encoder_weight_aliases(canonical: &str) -> Vec<String> {
    let mut out = vec![canonical.to_string()];

    // Pattern 1: embedding-level — `{prefix}.embeddings.{suffix}`
    for &prefix in ENCODER_PREFIXES {
        let pfx = if prefix.is_empty() {
            "embeddings.".to_string()
        } else {
            format!("{prefix}.embeddings.")
        };
        if let Some(suffix) = canonical.strip_prefix(&pfx) {
            for &p in ENCODER_PREFIXES {
                if p.is_empty() {
                    out.push(format!("embeddings.{suffix}"));
                } else {
                    out.push(format!("{p}.embeddings.{suffix}"));
                }
            }
            return out;
        }
    }

    // Pattern 2: layer-level — `{prefix}.encoder.layer.{N}.{suffix}`
    for &prefix in ENCODER_PREFIXES {
        let pfx = if prefix.is_empty() {
            "encoder.layer.".to_string()
        } else {
            format!("{prefix}.encoder.layer.")
        };
        if let Some(rest) = canonical.strip_prefix(&pfx) {
            let Some(dot_pos) = rest.find('.') else { break };
            let layer_str = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];
            let Ok(layer) = layer_str.parse::<usize>() else { break };

            for &p in ENCODER_PREFIXES {
                if p.is_empty() {
                    out.push(format!("encoder.layer.{layer}.{suffix}"));
                } else {
                    out.push(format!("{p}.encoder.layer.{layer}.{suffix}"));
                }
            }
            out.push(format!("blk.{layer}.{suffix}"));
            return out;
        }
    }

    out
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
///
/// Includes multi-modal nesting variants (`model.language_model`, `language_model`)
/// for models like Gemma 4 that wrap the text decoder under a `language_model`
/// sub-module alongside `vision_tower` / `audio_tower` / `embed_vision`. The
/// canonical YAML template names use the bare `model.` prefix; these extra
/// entries let `bind_weight_shapes_fuzzy` and the executor weight-binding loop
/// resolve the actual storage names.
const DECODER_PREFIXES: &[&str] = &["", "model", "language_model", "model.language_model"];

/// Unified entry point: get architecture prefixes by family.
///
/// Embedding/Reranker share Encoder weight topology (same prefix conventions)
/// but are distinct ArchFamily variants for BUILD-stage strategy selection.
pub fn arch_prefixes(family: ArchFamily) -> &'static [&'static str] {
    match family {
        ArchFamily::Encoder | ArchFamily::Embedding | ArchFamily::Reranker => ENCODER_PREFIXES,
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
// GPT-OSS specific weight name aliases
// ---------------------------------------------------------------------------

/// Attention sinks aliases for GPT-OSS (learnable softmax denominator tokens).
///
/// Canonical: `model.layers.{layer}.self_attn.sinks`
/// GGUF: `blk.{layer}.attn_sinks.weight`
pub fn attn_sinks_aliases(layer: usize) -> Vec<String> {
    decoder_layer_aliases(layer, "self_attn.sinks", Some("attn_sinks.weight"))
}

/// MoE router weight aliases for GPT-OSS (uses `mlp.router` naming).
///
/// GPT-OSS uses `mlp.router.weight/bias` instead of the standard `mlp.gate.weight`.
pub fn moe_router_weight_aliases(layer: usize) -> Vec<String> {
    decoder_layer_aliases(layer, "mlp.router.weight", Some("ffn_gate_inp.weight"))
}

/// MoE router bias aliases for GPT-OSS.
pub fn moe_router_bias_aliases(layer: usize) -> Vec<String> {
    decoder_layer_aliases(layer, "mlp.router.bias", Some("ffn_gate_inp.bias"))
}

/// Packed MoE expert tensor aliases for GPT-OSS (mxfp4 quantized).
///
/// `suffix` is one of: `gate_up_proj_blocks`, `gate_up_proj_scales`, `gate_up_proj_bias`,
/// `down_proj_blocks`, `down_proj_scales`, `down_proj_bias`.
pub fn moe_packed_expert_aliases(layer: usize, suffix: &str) -> Vec<String> {
    let gguf_suffix = HF_TO_GGUF_LAYER_SUFFIX
        .iter()
        .find(|&&(hf, _)| hf == format!("mlp.experts.{suffix}").as_str())
        .map(|&(_, gg)| gg);
    decoder_layer_aliases(layer, &format!("mlp.experts.{suffix}"), gguf_suffix)
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
    // GPT-OSS attention sinks (learnable softmax denominator tokens)
    ("self_attn.sinks",                  "attn_sinks.weight"),
    // GPT-OSS MoE router (uses "router" naming, not "gate")
    ("mlp.router.weight",                "ffn_gate_inp.weight"),
    ("mlp.router.bias",                  "ffn_gate_inp.bias"),
    // GPT-OSS packed MoE expert weights (mxfp4 quantized)
    ("mlp.experts.gate_up_proj_blocks",  "ffn_experts_gate_up_blocks.weight"),
    ("mlp.experts.gate_up_proj_scales",  "ffn_experts_gate_up_scales.weight"),
    ("mlp.experts.gate_up_proj_bias",    "ffn_experts_gate_up_bias.weight"),
    ("mlp.experts.down_proj_blocks",     "ffn_experts_down_blocks.weight"),
    ("mlp.experts.down_proj_scales",     "ffn_experts_down_scales.weight"),
    ("mlp.experts.down_proj_bias",       "ffn_experts_down_bias.weight"),
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
                "layers.".to_string()
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
///
/// Canonical (gllm template): `model.per_layer_embedding.embed_tokens.weight`
/// HuggingFace storage (Gemma 4): `model.language_model.embed_tokens_per_layer.weight`
/// (also handles bare `embed_tokens_per_layer.weight` and `language_model.` nesting).
pub fn ple_embed_tokens_aliases() -> Vec<String> {
    let mut out = Vec::new();
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push("per_layer_embedding.embed_tokens.weight".to_string());
            out.push("embed_tokens_per_layer.weight".to_string());
        } else {
            out.push(format!("{prefix}.per_layer_embedding.embed_tokens.weight"));
            out.push(format!("{prefix}.embed_tokens_per_layer.weight"));
        }
    }
    out
}

/// Generate alias names for PLE per_layer_projection weight.
///
/// Canonical (gllm template): `model.per_layer_embedding.per_layer_projection.weight`
/// HuggingFace storage (Gemma 4): `model.language_model.per_layer_model_projection.weight`
pub fn ple_projection_aliases() -> Vec<String> {
    let mut out = Vec::new();
    for &prefix in DECODER_PREFIXES {
        if prefix.is_empty() {
            out.push("per_layer_embedding.per_layer_projection.weight".to_string());
            out.push("per_layer_model_projection.weight".to_string());
        } else {
            out.push(format!("{prefix}.per_layer_embedding.per_layer_projection.weight"));
            out.push(format!("{prefix}.per_layer_model_projection.weight"));
        }
    }
    out
}

/// Generate alias names for the per-layer post_mlp_projection weight.
///
/// Canonical (gllm template): `model.layers.{i}.post_mlp_projection.weight`
/// HuggingFace storage (Gemma 4): `model.language_model.layers.{i}.per_layer_projection.weight`
///
/// `layer`: layer index (0-based)
pub fn ple_post_mlp_proj_aliases(layer: usize) -> Vec<String> {
    let mut out = decoder_layer_aliases(layer, PLE_POST_MLP_PROJ_PREFIX, None);
    // Gemma 4 storage uses `per_layer_projection.weight` (no `post_mlp_` prefix) under the
    // language_model decoder layer.
    out.extend(decoder_layer_aliases(layer, "per_layer_projection.weight", None));
    out
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
        // 每个 DECODER_PREFIX 产出 canonical (per_layer_embedding.embed_tokens.weight) +
        // Gemma 4 storage (embed_tokens_per_layer.weight) 两个变体
        assert!(aliases.iter().any(|a| a == "per_layer_embedding.embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "model.per_layer_embedding.embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "embed_tokens_per_layer.weight"));
        assert!(aliases.iter().any(|a| a == "model.language_model.embed_tokens_per_layer.weight"));
        assert_eq!(aliases.len(), DECODER_PREFIXES.len() * 2);
    }

    #[test]
    fn ple_projection_aliases_contains_canonical_and_bare() {
        let aliases = ple_projection_aliases();
        // 每个 DECODER_PREFIX 产出 canonical (per_layer_embedding.per_layer_projection.weight) +
        // Gemma 4 storage (per_layer_model_projection.weight) 两个变体
        assert!(aliases.iter().any(|a| a == "per_layer_embedding.per_layer_projection.weight"));
        assert!(aliases.iter().any(|a| a == "model.per_layer_embedding.per_layer_projection.weight"));
        assert!(aliases.iter().any(|a| a == "per_layer_model_projection.weight"));
        assert!(aliases.iter().any(|a| a == "model.language_model.per_layer_model_projection.weight"));
        assert_eq!(aliases.len(), DECODER_PREFIXES.len() * 2);
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

    // ── GPT-OSS weight alias tests ──

    #[test]
    fn gptoss_attn_sinks_aliases() {
        let aliases = attn_sinks_aliases(3);
        assert!(aliases.iter().any(|a| a == "model.layers.3.self_attn.sinks"));
        assert!(aliases.iter().any(|a| a == "layers.3.self_attn.sinks"));
        assert!(aliases.iter().any(|a| a == "blk.3.attn_sinks.weight"));
        assert!(aliases.iter().any(|a| a == "h.3.self_attn.sinks"));
    }

    #[test]
    fn gptoss_moe_router_aliases() {
        let weight = moe_router_weight_aliases(2);
        assert!(weight.iter().any(|a| a == "model.layers.2.mlp.router.weight"));
        assert!(weight.iter().any(|a| a == "blk.2.ffn_gate_inp.weight"));
        assert!(weight.iter().any(|a| a == "h.2.mlp.router.weight"));

        let bias = moe_router_bias_aliases(2);
        assert!(bias.iter().any(|a| a == "model.layers.2.mlp.router.bias"));
        assert!(bias.iter().any(|a| a == "blk.2.ffn_gate_inp.bias"));
    }

    #[test]
    fn gptoss_moe_packed_expert_aliases() {
        let blocks = moe_packed_expert_aliases(0, "gate_up_proj_blocks");
        assert!(blocks.iter().any(|a| a == "model.layers.0.mlp.experts.gate_up_proj_blocks"));
        assert!(blocks.iter().any(|a| a == "blk.0.ffn_experts_gate_up_blocks.weight"));

        let scales = moe_packed_expert_aliases(0, "gate_up_proj_scales");
        assert!(scales.iter().any(|a| a == "model.layers.0.mlp.experts.gate_up_proj_scales"));
        assert!(scales.iter().any(|a| a == "blk.0.ffn_experts_gate_up_scales.weight"));

        let down_blocks = moe_packed_expert_aliases(5, "down_proj_blocks");
        assert!(down_blocks.iter().any(|a| a == "model.layers.5.mlp.experts.down_proj_blocks"));
        assert!(down_blocks.iter().any(|a| a == "blk.5.ffn_experts_down_blocks.weight"));
    }

    #[test]
    fn gptoss_all_decoder_weight_aliases_attn_sinks() {
        let aliases = all_decoder_weight_aliases("model.layers.3.self_attn.sinks");
        assert!(!aliases.is_empty(), "attn sinks must be resolvable");
        assert!(aliases.iter().any(|a| a == "blk.3.attn_sinks.weight"));
    }

    #[test]
    fn gptoss_all_decoder_weight_aliases_router() {
        let aliases = all_decoder_weight_aliases("model.layers.1.mlp.router.weight");
        assert!(!aliases.is_empty(), "MoE router must be resolvable");
        assert!(aliases.iter().any(|a| a == "blk.1.ffn_gate_inp.weight"));
    }

    #[test]
    fn gptoss_all_decoder_weight_aliases_packed_experts() {
        let aliases = all_decoder_weight_aliases("model.layers.0.mlp.experts.gate_up_proj_blocks");
        assert!(!aliases.is_empty(), "packed expert blocks must be resolvable");
        assert!(aliases.iter().any(|a| a == "blk.0.ffn_experts_gate_up_blocks.weight"));
    }

    // ── Encoder alias tests ──

    #[test]
    fn embedding_aliases_without_gguf() {
        let aliases = embedding_aliases("word_embeddings.weight", None);
        assert_eq!(aliases.len(), ENCODER_PREFIXES.len());
        assert!(aliases.iter().any(|a| a == "embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "bert.embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "model.embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "roberta.embeddings.word_embeddings.weight"));
    }

    #[test]
    fn embedding_aliases_with_gguf() {
        let aliases = embedding_aliases("word_embeddings.weight", Some("token_embd.weight"));
        assert_eq!(aliases.len(), ENCODER_PREFIXES.len() + 1);
        assert_eq!(aliases.last().unwrap(), "token_embd.weight");
        assert!(aliases.iter().any(|a| a == "embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "bert.embeddings.word_embeddings.weight"));
    }

    #[test]
    fn embedding_aliases_position_embeddings() {
        let aliases = embedding_aliases("position_embeddings.weight", Some("position_embd.weight"));
        assert!(aliases.iter().any(|a| a == "embeddings.position_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "bert.embeddings.position_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "roberta.embeddings.position_embeddings.weight"));
        assert_eq!(aliases.last().unwrap(), "position_embd.weight");
    }

    #[test]
    fn layer_aliases_generates_all_prefixes_with_gguf() {
        let aliases = layer_aliases(2, "attention.self.query.weight", Some("attn_q.weight"));
        // 4 HF prefixes + 1 GGUF = 5
        assert_eq!(aliases.len(), ENCODER_PREFIXES.len() + 1);
        assert!(aliases.iter().any(|a| a == "encoder.layer.2.attention.self.query.weight"));
        assert!(aliases.iter().any(|a| a == "bert.encoder.layer.2.attention.self.query.weight"));
        assert!(aliases.iter().any(|a| a == "model.encoder.layer.2.attention.self.query.weight"));
        assert!(aliases.iter().any(|a| a == "roberta.encoder.layer.2.attention.self.query.weight"));
        assert!(aliases.iter().any(|a| a == "blk.2.attn_q.weight"));
    }

    #[test]
    fn layer_aliases_without_gguf() {
        let aliases = layer_aliases(0, "output.dense.weight", None);
        assert_eq!(aliases.len(), ENCODER_PREFIXES.len());
        assert!(aliases.iter().any(|a| a == "encoder.layer.0.output.dense.weight"));
        assert!(aliases.iter().any(|a| a == "bert.encoder.layer.0.output.dense.weight"));
        assert!(aliases.iter().any(|a| a == "model.encoder.layer.0.output.dense.weight"));
        assert!(aliases.iter().any(|a| a == "roberta.encoder.layer.0.output.dense.weight"));
        // No GGUF entry
        assert!(!aliases.iter().any(|a| a.starts_with("blk.")));
    }

    #[test]
    fn layer_prefixes_contains_all_prefixes() {
        let prefixes = layer_prefixes(3);
        assert_eq!(prefixes.len(), ENCODER_PREFIXES.len());
        assert!(prefixes.iter().any(|p| p == "encoder.layer.3"));
        assert!(prefixes.iter().any(|p| p == "bert.encoder.layer.3"));
        assert!(prefixes.iter().any(|p| p == "model.encoder.layer.3"));
        assert!(prefixes.iter().any(|p| p == "roberta.encoder.layer.3"));
    }

    #[test]
    fn layer_prefixes_layer_zero() {
        let prefixes = layer_prefixes(0);
        assert!(prefixes.iter().any(|p| p == "encoder.layer.0"));
        assert!(prefixes.iter().any(|p| p == "bert.encoder.layer.0"));
    }

    #[test]
    fn gguf_layer_prefix_format() {
        assert_eq!(gguf_layer_prefix(0), "blk.0");
        assert_eq!(gguf_layer_prefix(5), "blk.5");
        assert_eq!(gguf_layer_prefix(23), "blk.23");
    }

    #[test]
    fn all_encoder_weight_aliases_embedding() {
        let aliases = all_encoder_weight_aliases("embeddings.word_embeddings.weight");
        // Should contain all 4 prefix variants for embedding-level
        assert!(aliases.iter().any(|a| a == "embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "bert.embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "model.embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "roberta.embeddings.word_embeddings.weight"));
        // Original is always first
        assert_eq!(aliases[0], "embeddings.word_embeddings.weight");
    }

    #[test]
    fn all_encoder_weight_aliases_prefixed_embedding() {
        // Input with "bert." prefix should still resolve all variants
        let aliases = all_encoder_weight_aliases("bert.embeddings.word_embeddings.weight");
        assert!(aliases.iter().any(|a| a == "embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "bert.embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "model.embeddings.word_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "roberta.embeddings.word_embeddings.weight"));
    }

    #[test]
    fn all_encoder_weight_aliases_layer() {
        let aliases = all_encoder_weight_aliases("encoder.layer.3.attention.self.query.weight");
        // Should contain all 4 prefix variants + GGUF
        assert!(aliases.iter().any(|a| a == "encoder.layer.3.attention.self.query.weight"));
        assert!(aliases.iter().any(|a| a == "bert.encoder.layer.3.attention.self.query.weight"));
        assert!(aliases.iter().any(|a| a == "model.encoder.layer.3.attention.self.query.weight"));
        assert!(aliases.iter().any(|a| a == "roberta.encoder.layer.3.attention.self.query.weight"));
        assert!(aliases.iter().any(|a| a == "blk.3.attention.self.query.weight"));
        // Original is always first
        assert_eq!(aliases[0], "encoder.layer.3.attention.self.query.weight");
    }

    #[test]
    fn all_encoder_weight_aliases_prefixed_layer() {
        let aliases = all_encoder_weight_aliases("roberta.encoder.layer.1.intermediate.dense.weight");
        assert!(aliases.iter().any(|a| a == "encoder.layer.1.intermediate.dense.weight"));
        assert!(aliases.iter().any(|a| a == "bert.encoder.layer.1.intermediate.dense.weight"));
        assert!(aliases.iter().any(|a| a == "blk.1.intermediate.dense.weight"));
        assert_eq!(aliases[0], "roberta.encoder.layer.1.intermediate.dense.weight");
    }

    #[test]
    fn all_encoder_weight_aliases_unknown_returns_original() {
        let aliases = all_encoder_weight_aliases("some.random.weight.name");
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0], "some.random.weight.name");
    }

    #[test]
    fn has_any_embedding_weight_found() {
        assert!(has_any_embedding_weight(|name| name == "embeddings.word_embeddings.weight"));
        assert!(has_any_embedding_weight(|name| name == "bert.embeddings.word_embeddings.weight"));
        assert!(has_any_embedding_weight(|name| name == "model.embeddings.word_embeddings.weight"));
        assert!(has_any_embedding_weight(|name| name == "roberta.embeddings.word_embeddings.weight"));
        assert!(has_any_embedding_weight(|name| name == "token_embd.weight"));
    }

    #[test]
    fn has_any_embedding_weight_not_found() {
        assert!(!has_any_embedding_weight(|name| name == "encoder.layer.0.attention.self.query.weight"));
        assert!(!has_any_embedding_weight(|_name| false));
        assert!(!has_any_embedding_weight(|name| name == "lm_head.weight"));
    }

    #[test]
    fn has_any_intermediate_weight_found() {
        assert!(has_any_intermediate_weight(|name| name == "encoder.layer.0.intermediate.dense.weight"));
        assert!(has_any_intermediate_weight(|name| name == "bert.encoder.layer.0.intermediate.dense.weight"));
        assert!(has_any_intermediate_weight(|name| name == "model.encoder.layer.0.intermediate.dense.weight"));
        assert!(has_any_intermediate_weight(|name| name == "roberta.encoder.layer.0.intermediate.dense.weight"));
        assert!(has_any_intermediate_weight(|name| name == "blk.0.ffn_up.weight"));
    }

    #[test]
    fn has_any_intermediate_weight_not_found() {
        assert!(!has_any_intermediate_weight(|name| name == "embeddings.word_embeddings.weight"));
        assert!(!has_any_intermediate_weight(|_name| false));
        assert!(!has_any_intermediate_weight(|name| name == "encoder.layer.0.attention.self.query.weight"));
    }

    #[test]
    fn classifier_aliases_four_variants() {
        let aliases = classifier_aliases("weight");
        assert_eq!(aliases.len(), 4);
        assert!(aliases.iter().any(|a| a == "classifier.weight"));
        assert!(aliases.iter().any(|a| a == "classifier.dense.weight"));
        assert!(aliases.iter().any(|a| a == "classifier.out_proj.weight"));
        assert!(aliases.iter().any(|a| a == "pre_classifier.weight"));
    }

    #[test]
    fn classifier_aliases_bias() {
        let aliases = classifier_aliases("bias");
        assert!(aliases.iter().any(|a| a == "classifier.bias"));
        assert!(aliases.iter().any(|a| a == "classifier.dense.bias"));
        assert!(aliases.iter().any(|a| a == "classifier.out_proj.bias"));
        assert!(aliases.iter().any(|a| a == "pre_classifier.bias"));
    }

    #[test]
    fn pooler_aliases_two_variants() {
        let aliases = pooler_aliases("weight");
        assert_eq!(aliases.len(), 2);
        assert!(aliases.iter().any(|a| a == "pooler.dense.weight"));
        assert!(aliases.iter().any(|a| a == "pooler.weight"));
    }

    #[test]
    fn pooler_aliases_bias() {
        let aliases = pooler_aliases("bias");
        assert!(aliases.iter().any(|a| a == "pooler.dense.bias"));
        assert!(aliases.iter().any(|a| a == "pooler.bias"));
    }

    // ── Decoder alias tests ──

    #[test]
    fn arch_prefixes_encoder() {
        let prefixes = arch_prefixes(ArchFamily::Encoder);
        assert_eq!(prefixes, ENCODER_PREFIXES);
    }

    #[test]
    fn arch_prefixes_decoder() {
        let prefixes = arch_prefixes(ArchFamily::Decoder);
        assert_eq!(prefixes, DECODER_PREFIXES);
    }

    #[test]
    fn decoder_layer_aliases_all_prefixes_with_gguf() {
        let aliases = decoder_layer_aliases(3, "self_attn.q_proj.weight", Some("attn_q.weight"));
        // 4 HF prefixes + 1 GGUF + 1 h.{n} = 6
        assert_eq!(aliases.len(), DECODER_PREFIXES.len() + 2);
        assert!(aliases.iter().any(|a| a == "layers.3.self_attn.q_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.3.self_attn.q_proj.weight"));
        assert!(aliases.iter().any(|a| a == "language_model.layers.3.self_attn.q_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.language_model.layers.3.self_attn.q_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.3.attn_q.weight"));
        assert!(aliases.iter().any(|a| a == "h.3.self_attn.q_proj.weight"));
    }

    #[test]
    fn decoder_layer_aliases_without_gguf() {
        let aliases = decoder_layer_aliases(1, "input_layernorm.weight", None);
        // 4 HF prefixes + 1 h.{n} = 5, no GGUF
        assert_eq!(aliases.len(), DECODER_PREFIXES.len() + 1);
        assert!(aliases.iter().any(|a| a == "layers.1.input_layernorm.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.1.input_layernorm.weight"));
        assert!(aliases.iter().any(|a| a == "h.1.input_layernorm.weight"));
        assert!(!aliases.iter().any(|a| a.starts_with("blk.")));
    }

    #[test]
    fn decoder_embed_aliases_includes_gguf() {
        let aliases = decoder_embed_aliases();
        // 4 decoder prefixes + 1 GGUF token_embd = 5
        assert_eq!(aliases.len(), DECODER_PREFIXES.len() + 1);
        assert!(aliases.iter().any(|a| a == "embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "model.embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "language_model.embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "model.language_model.embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "token_embd.weight"));
    }

    #[test]
    fn decoder_final_norm_aliases_includes_all() {
        let aliases = decoder_final_norm_aliases();
        // 4 decoder prefixes + 1 GGUF output_norm = 5
        assert_eq!(aliases.len(), DECODER_PREFIXES.len() + 1);
        assert!(aliases.iter().any(|a| a == "norm.weight"));
        assert!(aliases.iter().any(|a| a == "model.norm.weight"));
        assert!(aliases.iter().any(|a| a == "language_model.norm.weight"));
        assert!(aliases.iter().any(|a| a == "model.language_model.norm.weight"));
        assert!(aliases.iter().any(|a| a == "output_norm.weight"));
    }

    #[test]
    fn decoder_final_norm_bias_aliases_empty() {
        let aliases = decoder_final_norm_bias_aliases();
        assert!(aliases.is_empty());
    }

    #[test]
    fn lm_head_aliases_includes_embed_fallback() {
        let aliases = lm_head_aliases();
        // Starts with lm_head-specific names
        assert!(aliases.iter().any(|a| a == "lm_head.weight"));
        assert!(aliases.iter().any(|a| a == "output.weight"));
        // Then includes all embed aliases for tied-weight fallback
        assert!(aliases.iter().any(|a| a == "embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "model.embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "token_embd.weight"));
        // lm_head names come before embed fallback names
        let lm_pos = aliases.iter().position(|a| a == "lm_head.weight").unwrap();
        let embed_pos = aliases.iter().position(|a| a == "embed_tokens.weight").unwrap();
        assert!(lm_pos < embed_pos, "lm_head.weight should come before embed fallback");
    }

    #[test]
    fn decoder_score_aliases_three_variants() {
        let aliases = decoder_score_aliases();
        assert_eq!(aliases.len(), 3);
        assert!(aliases.iter().any(|a| a == "score.weight"));
        assert!(aliases.iter().any(|a| a == "classifier.weight"));
        assert!(aliases.iter().any(|a| a == "cls.weight"));
    }

    #[test]
    fn moe_gate_aliases_format() {
        let aliases = moe_gate_aliases(2);
        // 4 decoder prefixes + 1 GGUF = 5
        assert_eq!(aliases.len(), DECODER_PREFIXES.len() + 1);
        assert!(aliases.iter().any(|a| a == "layers.2.mlp.gate.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.2.mlp.gate.weight"));
        assert!(aliases.iter().any(|a| a == "language_model.layers.2.mlp.gate.weight"));
        assert!(aliases.iter().any(|a| a == "model.language_model.layers.2.mlp.gate.weight"));
        assert!(aliases.iter().any(|a| a == "blk.2.ffn_gate_inp.weight"));
    }

    #[test]
    fn moe_shared_expert_aliases_format() {
        let aliases = moe_shared_expert_aliases(3, "gate_proj.weight");
        // 4 decoder prefixes + 1 GGUF = 5
        assert_eq!(aliases.len(), DECODER_PREFIXES.len() + 1);
        assert!(aliases.iter().any(|a| a == "layers.3.mlp.shared_experts.gate_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.3.mlp.shared_experts.gate_proj.weight"));
        assert!(aliases.iter().any(|a| a == "language_model.layers.3.mlp.shared_experts.gate_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.3.ffn_shared_expert.gate_proj.weight"));
    }

    #[test]
    fn moe_expert_aliases_format() {
        let aliases = moe_expert_aliases(1, 5, "up_proj.weight");
        // 4 decoder prefixes + 1 GGUF = 5
        assert_eq!(aliases.len(), DECODER_PREFIXES.len() + 1);
        assert!(aliases.iter().any(|a| a == "layers.1.mlp.experts.5.up_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.1.mlp.experts.5.up_proj.weight"));
        assert!(aliases.iter().any(|a| a == "language_model.layers.1.mlp.experts.5.up_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.language_model.layers.1.mlp.experts.5.up_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.1.ffn_gate_exps.5.up_proj.weight"));
    }

    #[test]
    fn moe_expert_aliases_down_proj() {
        let aliases = moe_expert_aliases(0, 0, "down_proj.weight");
        assert!(aliases.iter().any(|a| a == "layers.0.mlp.experts.0.down_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.0.ffn_gate_exps.0.down_proj.weight"));
    }

    #[test]
    fn all_decoder_weight_aliases_standard_q_proj() {
        let aliases = all_decoder_weight_aliases("model.layers.5.self_attn.q_proj.weight");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "layers.5.self_attn.q_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.5.self_attn.q_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.5.attn_q.weight"));
        assert!(aliases.iter().any(|a| a == "h.5.self_attn.q_proj.weight"));
    }

    #[test]
    fn all_decoder_weight_aliases_bare_layers() {
        let aliases = all_decoder_weight_aliases("layers.0.self_attn.q_proj.weight");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "model.layers.0.self_attn.q_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.0.attn_q.weight"));
    }

    #[test]
    fn all_decoder_weight_aliases_language_model_prefix() {
        let aliases = all_decoder_weight_aliases("language_model.layers.2.mlp.gate_proj.weight");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "model.layers.2.mlp.gate_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.2.ffn_gate.weight"));
    }

    #[test]
    fn all_decoder_weight_aliases_model_language_model_prefix() {
        let aliases = all_decoder_weight_aliases("model.language_model.layers.1.mlp.up_proj.weight");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "model.layers.1.mlp.up_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.1.ffn_up.weight"));
    }

    #[test]
    fn all_decoder_weight_aliases_unknown_returns_empty() {
        assert!(all_decoder_weight_aliases("lm_head.weight").is_empty());
        assert!(all_decoder_weight_aliases("model.embed_tokens.weight").is_empty());
        assert!(all_decoder_weight_aliases("some.random.name").is_empty());
        assert!(all_decoder_weight_aliases("model.layers.notanumber.self_attn.q_proj.weight").is_empty());
    }

    #[test]
    fn all_decoder_weight_aliases_norm_weights() {
        let aliases = all_decoder_weight_aliases("model.layers.0.input_layernorm.weight");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "blk.0.attn_norm.weight"));

        let aliases = all_decoder_weight_aliases("model.layers.0.post_attention_layernorm.weight");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "blk.0.ffn_norm.weight"));
    }

    #[test]
    fn all_decoder_weight_aliases_qwen3_norm_variants() {
        let post_ffw = all_decoder_weight_aliases("model.layers.2.post_feedforward_layernorm.weight");
        assert!(!post_ffw.is_empty());
        assert!(post_ffw.iter().any(|a| a == "blk.2.post_ffw_norm.weight"));

        let pre_ffw = all_decoder_weight_aliases("model.layers.2.pre_feedforward_layernorm.weight");
        assert!(!pre_ffw.is_empty());
        assert!(pre_ffw.iter().any(|a| a == "blk.2.pre_ffw_norm.weight"));
    }

    #[test]
    fn all_decoder_weight_aliases_attention_norm() {
        let q_norm = all_decoder_weight_aliases("model.layers.0.self_attn.q_norm.weight");
        assert!(!q_norm.is_empty());
        assert!(q_norm.iter().any(|a| a == "blk.0.attn_q_norm.weight"));

        let k_norm = all_decoder_weight_aliases("model.layers.0.self_attn.k_norm.weight");
        assert!(!k_norm.is_empty());
        assert!(k_norm.iter().any(|a| a == "blk.0.attn_k_norm.weight"));
    }

    #[test]
    fn decoder_layer_aliases_layer_zero() {
        let aliases = decoder_layer_aliases(0, "self_attn.k_proj.weight", Some("attn_k.weight"));
        assert!(aliases.iter().any(|a| a == "layers.0.self_attn.k_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.0.attn_k.weight"));
        assert!(aliases.iter().any(|a| a == "h.0.self_attn.k_proj.weight"));
    }

    // ── Additional coverage tests ──

    // -- ArchFamily trait tests --

    #[test]
    fn arch_family_equality() {
        assert_eq!(ArchFamily::Encoder, ArchFamily::Encoder);
        assert_eq!(ArchFamily::Decoder, ArchFamily::Decoder);
        assert_ne!(ArchFamily::Encoder, ArchFamily::Decoder);
    }

    #[test]
    fn arch_family_copy() {
        let a = ArchFamily::Encoder;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn arch_family_clone() {
        let a = ArchFamily::Decoder;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn arch_family_debug_format() {
        assert_eq!(format!("{:?}", ArchFamily::Encoder), "Encoder");
        assert_eq!(format!("{:?}", ArchFamily::Decoder), "Decoder");
    }

    #[test]
    fn arch_family_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ArchFamily::Encoder);
        set.insert(ArchFamily::Decoder);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&ArchFamily::Encoder));
        assert!(set.contains(&ArchFamily::Decoder));
    }

    // -- PLE constant tests --

    #[test]
    fn ple_embed_tokens_constant_value() {
        assert_eq!(PLE_EMBED_TOKENS, "model.per_layer_embedding.embed_tokens.weight");
    }

    #[test]
    fn ple_projection_constant_value() {
        assert_eq!(PLE_PROJECTION, "model.per_layer_embedding.per_layer_projection.weight");
    }

    #[test]
    fn ple_post_mlp_proj_prefix_constant_value() {
        assert_eq!(PLE_POST_MLP_PROJ_PREFIX, "post_mlp_projection.weight");
    }

    // -- embedding_aliases edge cases --

    #[test]
    fn embedding_aliases_empty_suffix() {
        let aliases = embedding_aliases("", None);
        assert_eq!(aliases.len(), ENCODER_PREFIXES.len());
        assert!(aliases.iter().any(|a| a == "embeddings."));
        assert!(aliases.iter().any(|a| a == "bert.embeddings."));
    }

    #[test]
    fn embedding_aliases_empty_suffix_with_gguf() {
        let aliases = embedding_aliases("", Some("custom.weight"));
        assert_eq!(aliases.last().unwrap(), "custom.weight");
        assert_eq!(aliases.len(), ENCODER_PREFIXES.len() + 1);
    }

    // -- layer_aliases edge cases --

    #[test]
    fn layer_aliases_layer_zero() {
        let aliases = layer_aliases(0, "attention.self.query.weight", None);
        assert!(aliases.iter().any(|a| a == "encoder.layer.0.attention.self.query.weight"));
        assert!(aliases.iter().any(|a| a == "bert.encoder.layer.0.attention.self.query.weight"));
    }

    #[test]
    fn layer_aliases_large_layer_index() {
        let aliases = layer_aliases(999, "output.dense.weight", Some("ffn_out.weight"));
        assert!(aliases.iter().any(|a| a == "encoder.layer.999.output.dense.weight"));
        assert!(aliases.iter().any(|a| a == "blk.999.ffn_out.weight"));
    }

    // -- layer_prefixes edge cases --

    #[test]
    fn layer_prefixes_large_index() {
        let prefixes = layer_prefixes(100);
        assert!(prefixes.iter().any(|p| p == "encoder.layer.100"));
        assert!(prefixes.iter().any(|p| p == "bert.encoder.layer.100"));
    }

    // -- gguf_layer_prefix edge cases --

    #[test]
    fn gguf_layer_prefix_large() {
        assert_eq!(gguf_layer_prefix(999), "blk.999");
    }

    // -- all_encoder_weight_aliases edge cases --

    #[test]
    fn all_encoder_weight_aliases_layer_no_dot_after_number() {
        // "encoder.layer.3" has no dot after layer number — falls through to return original
        let aliases = all_encoder_weight_aliases("encoder.layer.3");
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0], "encoder.layer.3");
    }

    #[test]
    fn all_encoder_weight_aliases_layer_non_numeric() {
        // "encoder.layer.abc.weight" — layer number parse fails, returns original
        let aliases = all_encoder_weight_aliases("encoder.layer.abc.weight");
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0], "encoder.layer.abc.weight");
    }

    #[test]
    fn all_encoder_weight_aliases_empty_string() {
        let aliases = all_encoder_weight_aliases("");
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0], "");
    }

    // -- classifier_aliases edge cases --

    #[test]
    fn classifier_aliases_empty_suffix() {
        let aliases = classifier_aliases("");
        assert_eq!(aliases.len(), 4);
        assert!(aliases.iter().any(|a| a == "classifier."));
        assert!(aliases.iter().any(|a| a == "classifier.dense."));
        assert!(aliases.iter().any(|a| a == "classifier.out_proj."));
        assert!(aliases.iter().any(|a| a == "pre_classifier."));
    }

    #[test]
    fn classifier_aliases_custom_suffix() {
        let aliases = classifier_aliases("inner.weight");
        assert!(aliases.iter().any(|a| a == "classifier.inner.weight"));
        assert!(aliases.iter().any(|a| a == "classifier.dense.inner.weight"));
    }

    // -- pooler_aliases edge cases --

    #[test]
    fn pooler_aliases_empty_suffix() {
        let aliases = pooler_aliases("");
        assert_eq!(aliases.len(), 2);
        assert!(aliases.iter().any(|a| a == "pooler.dense."));
        assert!(aliases.iter().any(|a| a == "pooler."));
    }

    // -- decoder_layer_aliases edge cases --

    #[test]
    fn decoder_layer_aliases_large_layer() {
        let aliases = decoder_layer_aliases(999, "self_attn.v_proj.weight", Some("attn_v.weight"));
        assert!(aliases.iter().any(|a| a == "layers.999.self_attn.v_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.999.self_attn.v_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.999.attn_v.weight"));
        assert!(aliases.iter().any(|a| a == "h.999.self_attn.v_proj.weight"));
    }

    // -- decoder_embed_aliases structure --

    #[test]
    fn decoder_embed_aliases_no_duplicate_gguf() {
        let aliases = decoder_embed_aliases();
        // Exactly one token_embd.weight entry
        assert_eq!(aliases.iter().filter(|a| **a == "token_embd.weight").count(), 1);
    }

    // -- decoder_final_norm_aliases structure --

    #[test]
    fn decoder_final_norm_aliases_no_duplicate_output_norm() {
        let aliases = decoder_final_norm_aliases();
        assert_eq!(aliases.iter().filter(|a| **a == "output_norm.weight").count(), 1);
    }

    // -- lm_head_aliases structure --

    #[test]
    fn lm_head_aliases_ordering() {
        let aliases = lm_head_aliases();
        // output.weight should come before embed fallback
        let output_pos = aliases.iter().position(|a| a == "output.weight").unwrap();
        let embed_pos = aliases.iter().position(|a| a == "embed_tokens.weight").unwrap();
        assert!(output_pos < embed_pos);
    }

    #[test]
    fn lm_head_aliases_contains_all_decoder_embed_entries() {
        let embed = decoder_embed_aliases();
        let lm = lm_head_aliases();
        for e in &embed {
            assert!(lm.contains(e), "lm_head_aliases missing embed fallback: {e}");
        }
    }

    // -- moe_gate_aliases layer 0 --

    #[test]
    fn moe_gate_aliases_layer_zero() {
        let aliases = moe_gate_aliases(0);
        assert!(aliases.iter().any(|a| a == "layers.0.mlp.gate.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.0.mlp.gate.weight"));
        assert!(aliases.iter().any(|a| a == "blk.0.ffn_gate_inp.weight"));
    }

    // -- moe_shared_expert_aliases down_proj --

    #[test]
    fn moe_shared_expert_aliases_down_proj() {
        let aliases = moe_shared_expert_aliases(0, "down_proj.weight");
        assert!(aliases.iter().any(|a| a == "layers.0.mlp.shared_experts.down_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.0.ffn_shared_expert.down_proj.weight"));
    }

    // -- moe_expert_aliases layer 0 expert 0 --

    #[test]
    fn moe_expert_aliases_zero_indices() {
        let aliases = moe_expert_aliases(0, 0, "gate_proj.weight");
        assert!(aliases.iter().any(|a| a == "layers.0.mlp.experts.0.gate_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.0.ffn_gate_exps.0.gate_proj.weight"));
    }

    // -- moe_packed_expert_aliases unknown suffix (no GGUF mapping) --

    #[test]
    fn moe_packed_expert_aliases_unknown_suffix_no_gguf() {
        let aliases = moe_packed_expert_aliases(0, "nonexistent.weight");
        // Should still produce HF prefix variants but no GGUF entry
        assert!(aliases.iter().any(|a| a.contains("mlp.experts.nonexistent.weight")));
        assert!(!aliases.iter().any(|a| a.starts_with("blk.")));
    }

    // -- all_decoder_weight_aliases suffix without GGUF mapping --

    #[test]
    fn all_decoder_weight_aliases_suffix_without_gguf_mapping() {
        // "self_attn.q_proj.bias" has a GGUF mapping: "attn_q.bias"
        // Use a suffix that is NOT in HF_TO_GGUF_LAYER_SUFFIX
        let aliases = all_decoder_weight_aliases("model.layers.0.custom_op.weight");
        assert!(!aliases.is_empty(), "should still produce HF prefix variants");
        assert!(aliases.iter().any(|a| a == "layers.0.custom_op.weight"));
        // No GGUF entry because "custom_op.weight" is not in the mapping table
        assert!(!aliases.iter().any(|a| a.starts_with("blk.")));
    }

    // -- all_decoder_weight_aliases truncated patterns --

    #[test]
    fn all_decoder_weight_aliases_no_dot_after_layer_number() {
        let aliases = all_decoder_weight_aliases("model.layers.3");
        assert!(aliases.is_empty(), "no dot after layer number → no suffix → empty");
    }

    #[test]
    fn all_decoder_weight_aliases_no_layer_number() {
        let aliases = all_decoder_weight_aliases("model.layers..weight");
        assert!(aliases.is_empty(), "empty layer number → parse fails");
    }

    // -- all_ple_weight_aliases with language_model prefix --

    #[test]
    fn all_ple_weight_aliases_per_layer_language_model_prefix() {
        let aliases = all_ple_weight_aliases("language_model.layers.2.post_mlp_projection.weight");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "model.layers.2.post_mlp_projection.weight"));
    }

    #[test]
    fn all_ple_weight_aliases_per_layer_model_language_model_prefix() {
        let aliases = all_ple_weight_aliases("model.language_model.layers.4.post_mlp_projection.weight");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "model.layers.4.post_mlp_projection.weight"));
    }

    // -- ple_embed_tokens_aliases structure --

    #[test]
    fn ple_embed_tokens_aliases_no_duplicates() {
        let aliases = ple_embed_tokens_aliases();
        let mut seen = std::collections::HashSet::new();
        for a in &aliases {
            assert!(seen.insert(a.clone()), "duplicate alias found: {a}");
        }
    }

    // -- ple_projection_aliases structure --

    #[test]
    fn ple_projection_aliases_no_duplicates() {
        let aliases = ple_projection_aliases();
        let mut seen = std::collections::HashSet::new();
        for a in &aliases {
            assert!(seen.insert(a.clone()), "duplicate alias found: {a}");
        }
    }

    // -- ple_post_mlp_proj_aliases includes per_layer_projection fallback --

    #[test]
    fn ple_post_mlp_proj_aliases_includes_per_layer_projection_fallback() {
        let aliases = ple_post_mlp_proj_aliases(0);
        // Should contain both post_mlp_projection.weight and per_layer_projection.weight variants
        assert!(aliases.iter().any(|a| a == "layers.0.post_mlp_projection.weight"));
        assert!(aliases.iter().any(|a| a == "layers.0.per_layer_projection.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.0.per_layer_projection.weight"));
    }

    // -- all_ple_weight_aliases per-layer wrong suffix returns empty --

    #[test]
    fn all_ple_weight_aliases_per_layer_wrong_suffix() {
        let aliases = all_ple_weight_aliases("model.layers.3.self_attn.q_proj.weight");
        assert!(aliases.is_empty(), "non-PLE suffix in a PLE prefix should return empty");
    }

    // -- all_ple_weight_aliases empty string --

    #[test]
    fn all_ple_weight_aliases_empty() {
        assert!(all_ple_weight_aliases("").is_empty());
    }

    // -- attn_sinks_aliases layer 0 --

    #[test]
    fn attn_sinks_aliases_layer_zero() {
        let aliases = attn_sinks_aliases(0);
        assert!(aliases.iter().any(|a| a == "layers.0.self_attn.sinks"));
        assert!(aliases.iter().any(|a| a == "blk.0.attn_sinks.weight"));
        assert!(aliases.iter().any(|a| a == "h.0.self_attn.sinks"));
    }

    // -- moe_router_weight_aliases layer 0 --

    #[test]
    fn moe_router_weight_aliases_layer_zero() {
        let aliases = moe_router_weight_aliases(0);
        assert!(aliases.iter().any(|a| a == "layers.0.mlp.router.weight"));
        assert!(aliases.iter().any(|a| a == "blk.0.ffn_gate_inp.weight"));
    }

    // -- moe_router_bias_aliases layer 0 --

    #[test]
    fn moe_router_bias_aliases_layer_zero() {
        let aliases = moe_router_bias_aliases(0);
        assert!(aliases.iter().any(|a| a == "layers.0.mlp.router.bias"));
        assert!(aliases.iter().any(|a| a == "blk.0.ffn_gate_inp.bias"));
    }

    // -- HF_TO_GGUF_LAYER_SUFFIX coverage: all standard attn entries --

    #[test]
    fn all_decoder_weight_aliases_all_attn_projections() {
        for (hf_suffix, gguf_suffix) in &[
            ("self_attn.k_proj.weight", "attn_k.weight"),
            ("self_attn.v_proj.weight", "attn_v.weight"),
            ("self_attn.o_proj.weight", "attn_output.weight"),
        ] {
            let name = format!("model.layers.7.{hf_suffix}");
            let aliases = all_decoder_weight_aliases(&name);
            let expected_gguf = format!("blk.7.{gguf_suffix}");
            assert!(
                aliases.iter().any(|a| *a == expected_gguf),
                "GGUF alias {expected_gguf} not found for {name}"
            );
        }
    }

    // -- HF_TO_GGUF_LAYER_SUFFIX coverage: all bias entries --

    #[test]
    fn all_decoder_weight_aliases_all_attn_biases() {
        for (hf_suffix, gguf_suffix) in &[
            ("self_attn.q_proj.bias", "attn_q.bias"),
            ("self_attn.k_proj.bias", "attn_k.bias"),
            ("self_attn.v_proj.bias", "attn_v.bias"),
            ("self_attn.o_proj.bias", "attn_output.bias"),
        ] {
            let name = format!("model.layers.1.{hf_suffix}");
            let aliases = all_decoder_weight_aliases(&name);
            let expected_gguf = format!("blk.1.{gguf_suffix}");
            assert!(
                aliases.iter().any(|a| *a == expected_gguf),
                "GGUF alias {expected_gguf} not found for {name}"
            );
        }
    }

    // -- HF_TO_GGUF_LAYER_SUFFIX coverage: ffn entries --

    #[test]
    fn all_decoder_weight_aliases_ffn_weights_and_biases() {
        for (hf_suffix, gguf_suffix) in &[
            ("mlp.gate_proj.weight", "ffn_gate.weight"),
            ("mlp.up_proj.weight", "ffn_up.weight"),
            ("mlp.down_proj.weight", "ffn_down.weight"),
            ("mlp.gate_proj.bias", "ffn_gate.bias"),
            ("mlp.up_proj.bias", "ffn_up.bias"),
            ("mlp.down_proj.bias", "ffn_down.bias"),
        ] {
            let name = format!("model.layers.0.{hf_suffix}");
            let aliases = all_decoder_weight_aliases(&name);
            let expected_gguf = format!("blk.0.{gguf_suffix}");
            assert!(
                aliases.iter().any(|a| *a == expected_gguf),
                "GGUF alias {expected_gguf} not found for {name}"
            );
        }
    }

    // -- all_decoder_weight_aliases h.{n} format is not a recognized input --

    #[test]
    fn all_decoder_weight_aliases_h_prefix_not_recognized() {
        // "h.3.self_attn.q_proj.weight" uses GPT-2 naming, not a recognized input prefix
        let aliases = all_decoder_weight_aliases("h.3.self_attn.q_proj.weight");
        assert!(aliases.is_empty());
    }

    // -- all_encoder_weight_aliases with all four prefixes for layer --

    #[test]
    fn all_encoder_weight_aliases_model_prefix_layer() {
        let aliases = all_encoder_weight_aliases("model.encoder.layer.2.output.dense.weight");
        assert_eq!(aliases[0], "model.encoder.layer.2.output.dense.weight");
        assert!(aliases.iter().any(|a| a == "encoder.layer.2.output.dense.weight"));
        assert!(aliases.iter().any(|a| a == "bert.encoder.layer.2.output.dense.weight"));
        assert!(aliases.iter().any(|a| a == "roberta.encoder.layer.2.output.dense.weight"));
        assert!(aliases.iter().any(|a| a == "blk.2.output.dense.weight"));
    }

    // -- decoder_score_aliases no output.weight --

    #[test]
    fn decoder_score_aliases_excludes_output_weight() {
        let aliases = decoder_score_aliases();
        assert!(!aliases.iter().any(|a| a == "output.weight"), "output.weight must not be in score aliases");
    }

    // ── 13 additional coverage tests ──

    // 1. ENCODER_PREFIXES constant: verify exact count and content
    #[test]
    fn encoder_prefixes_exact_content() {
        assert_eq!(ENCODER_PREFIXES.len(), 4);
        assert_eq!(ENCODER_PREFIXES[0], "");
        assert_eq!(ENCODER_PREFIXES[1], "bert");
        assert_eq!(ENCODER_PREFIXES[2], "model");
        assert_eq!(ENCODER_PREFIXES[3], "roberta");
    }

    // 2. DECODER_PREFIXES constant: verify exact count and content
    #[test]
    fn decoder_prefixes_exact_content() {
        assert_eq!(DECODER_PREFIXES.len(), 4);
        assert_eq!(DECODER_PREFIXES[0], "");
        assert_eq!(DECODER_PREFIXES[1], "model");
        assert_eq!(DECODER_PREFIXES[2], "language_model");
        assert_eq!(DECODER_PREFIXES[3], "model.language_model");
    }

    // 3. embedding_aliases with empty GGUF string (Some("") is distinct from None)
    #[test]
    fn embedding_aliases_with_empty_gguf_string() {
        let aliases = embedding_aliases("word_embeddings.weight", Some(""));
        assert_eq!(aliases.len(), ENCODER_PREFIXES.len() + 1);
        // Last entry is the empty GGUF string
        assert_eq!(aliases.last().unwrap(), "");
    }

    // 4. all_encoder_weight_aliases with "model." prefixed embedding
    #[test]
    fn all_encoder_weight_aliases_model_prefix_embedding() {
        let aliases = all_encoder_weight_aliases("model.embeddings.position_embeddings.weight");
        assert!(aliases.iter().any(|a| a == "embeddings.position_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "bert.embeddings.position_embeddings.weight"));
        assert!(aliases.iter().any(|a| a == "roberta.embeddings.position_embeddings.weight"));
        assert_eq!(aliases[0], "model.embeddings.position_embeddings.weight");
    }

    // 5. all_decoder_weight_aliases with bias suffix (has GGUF mapping)
    #[test]
    fn all_decoder_weight_aliases_mlp_bias_with_gguf() {
        let aliases = all_decoder_weight_aliases("model.layers.3.mlp.gate_proj.bias");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "layers.3.mlp.gate_proj.bias"));
        assert!(aliases.iter().any(|a| a == "blk.3.ffn_gate.bias"));
        assert!(aliases.iter().any(|a| a == "h.3.mlp.gate_proj.bias"));
    }

    // 6. ple_post_mlp_proj_aliases double invocation produces consistent count
    #[test]
    fn ple_post_mlp_proj_aliases_exact_count() {
        let aliases = ple_post_mlp_proj_aliases(1);
        // decoder_layer_aliases produces DECODER_PREFIXES.len() + 1 (h.{n}) per suffix
        // Two suffixes: post_mlp_projection.weight + per_layer_projection.weight
        let expected = (DECODER_PREFIXES.len() + 1) * 2;
        assert_eq!(aliases.len(), expected, "expected {expected} aliases, got {}", aliases.len());
    }

    // 7. lm_head_aliases total count verification
    #[test]
    fn lm_head_aliases_exact_count() {
        let aliases = lm_head_aliases();
        // 2 lm_head-specific + decoder_embed_aliases count (4 decoder prefixes + 1 GGUF)
        let expected = 2 + DECODER_PREFIXES.len() + 1;
        assert_eq!(aliases.len(), expected, "expected {expected} aliases, got {}", aliases.len());
    }

    // 8. moe_shared_expert_aliases with up_proj suffix
    #[test]
    fn moe_shared_expert_aliases_up_proj() {
        let aliases = moe_shared_expert_aliases(2, "up_proj.weight");
        assert!(aliases.iter().any(|a| a == "layers.2.mlp.shared_experts.up_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.layers.2.mlp.shared_experts.up_proj.weight"));
        assert!(aliases.iter().any(|a| a == "language_model.layers.2.mlp.shared_experts.up_proj.weight"));
        assert!(aliases.iter().any(|a| a == "model.language_model.layers.2.mlp.shared_experts.up_proj.weight"));
        assert!(aliases.iter().any(|a| a == "blk.2.ffn_shared_expert.up_proj.weight"));
    }

    // 9. moe_packed_expert_aliases with down_proj_blocks (known GGUF mapping)
    #[test]
    fn moe_packed_expert_aliases_down_proj_blocks() {
        let aliases = moe_packed_expert_aliases(3, "down_proj_blocks");
        assert!(aliases.iter().any(|a| a == "model.layers.3.mlp.experts.down_proj_blocks"));
        assert!(aliases.iter().any(|a| a == "blk.3.ffn_experts_down_blocks.weight"));
        assert!(aliases.iter().any(|a| a == "h.3.mlp.experts.down_proj_blocks"));
    }

    // 10. all_decoder_weight_aliases with "layers." bare prefix and bias suffix
    #[test]
    fn all_decoder_weight_aliases_bare_layers_bias() {
        let aliases = all_decoder_weight_aliases("layers.2.mlp.down_proj.bias");
        assert!(!aliases.is_empty());
        assert!(aliases.iter().any(|a| a == "model.layers.2.mlp.down_proj.bias"));
        assert!(aliases.iter().any(|a| a == "blk.2.ffn_down.bias"));
    }

    // 11. all_ple_weight_aliases from bare embed_tokens_per_layer storage name
    #[test]
    fn all_ple_weight_aliases_from_bare_embed_tokens_per_layer() {
        let aliases = all_ple_weight_aliases("embed_tokens_per_layer.weight");
        assert!(!aliases.is_empty(), "bare embed_tokens_per_layer should resolve PLE aliases");
        assert!(aliases.iter().any(|a| a == "model.per_layer_embedding.embed_tokens.weight"));
        assert!(aliases.iter().any(|a| a == "model.embed_tokens_per_layer.weight"));
    }

    // 12. all_encoder_weight_aliases with roberta prefix on layer pattern
    #[test]
    fn all_encoder_weight_aliases_roberta_prefix_layer() {
        let aliases = all_encoder_weight_aliases("roberta.encoder.layer.5.attention.output.LayerNorm.weight");
        assert!(aliases.iter().any(|a| a == "encoder.layer.5.attention.output.LayerNorm.weight"));
        assert!(aliases.iter().any(|a| a == "bert.encoder.layer.5.attention.output.LayerNorm.weight"));
        assert!(aliases.iter().any(|a| a == "model.encoder.layer.5.attention.output.LayerNorm.weight"));
        assert!(aliases.iter().any(|a| a == "blk.5.attention.output.LayerNorm.weight"));
        assert_eq!(aliases[0], "roberta.encoder.layer.5.attention.output.LayerNorm.weight");
    }

    // 13. arch_prefixes returns correct slices with matching length and elements
    #[test]
    fn arch_prefixes_slice_contents_match_constants() {
        let enc = arch_prefixes(ArchFamily::Encoder);
        let dec = arch_prefixes(ArchFamily::Decoder);
        // Verify slice contents match the module-level constants
        assert_eq!(enc, ENCODER_PREFIXES);
        assert_eq!(dec, DECODER_PREFIXES);
        // Repeated calls return equal slices
        assert_eq!(arch_prefixes(ArchFamily::Encoder), arch_prefixes(ArchFamily::Encoder));
        assert_eq!(arch_prefixes(ArchFamily::Decoder), arch_prefixes(ArchFamily::Decoder));
    }
}
