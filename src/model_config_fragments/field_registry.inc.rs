// === FieldDef Registry — Declarative config.json field mapping ===
// BCE-040: Replace hardcoded find_*/require_* calls with registry-driven parsing.
//
// Architecture:
//   1. normalize_text_config() flattens text_config.* to root (root wins on collision)
//   2. FIELD_DEFS static declares every field with its JSON aliases, required flag, and default
//   3. apply_field_registry() walks FIELD_DEFS, resolves each field, produces CanonicalConfig
//   4. PostProcess fields (head_dim, kv_cache_block_size, multimodal_token_ids) are computed
//      from already-resolved fields in a second pass
//
// Key invariant: all json_keys in FIELD_DEFS list ONLY the non-text_config variant.
// normalize_text_config() handles text_config.* flattening before lookup, so every field
// automatically gains text_config.* coverage without duplicating key strings.

// ── MetaValue: neutral value after parsing, before ModelConfig assembly ──

/// Neutral value representation after parsing, before ModelConfig assembly.
/// Each variant maps to exactly one field type in ModelConfig.
#[derive(Debug, Clone)]
pub enum MetaValue {
    Usize(usize),
    F32(f32),
    Bool(bool),
    Str(String),
    U32(u32),
    UsizeArray(Vec<usize>),
    F32Array(Vec<f32>),
    // Complex types that need custom parsing
    RopeScaling(Option<RopeScalingConfig>),
    AttentionPattern(Vec<u8>),
    MlaConfig(MlaConfig),
    VisionConfig(crate::compat::vision_forward::VisionConfig),
    AudioConfig(crate::compat::audio_forward::AudioConfig),
    MultimodalTokenIds(crate::compat::multimodal::MultimodalTokenIds),
}

// ── FieldKind: how a field is resolved ──

/// How a field is resolved from JSON / GGUF.
#[derive(Clone)]
pub enum FieldKind {
    /// Simple alias lookup: try each key in order, return first match.
    /// json_keys list ONLY non-text_config variants; normalize_text_config handles the rest.
    Alias {
        json_keys: &'static [&'static str],
        /// GGUF metadata keys (without arch prefix). Empty for JSON-only fields.
        /// Runtime resolves as `format!("{arch}.{key}")`.
        gguf_keys: &'static [&'static str],
        /// GGUF reader native method (for fields like embedding_length that don't use arch prefix).
        /// Returns None = use gguf_keys with arch prefix instead.
        /// The closure receives a &GgufLoader (= GgufReader type alias) and returns Option<MetaValue>.
        gguf_reader: Option<fn(&GgufLoader) -> Option<MetaValue>>,
    },
    /// Custom parsing logic (cross-field dependencies, complex structures).
    /// Receives the *entire* config JSON root (after text_config normalization).
    /// Returns Ok(None) when field is absent, Err on invalid data.
    Derived {
        parse_json: fn(&Value) -> ModelConfigResult<Option<MetaValue>>,
        /// GGUF-specific parsing. None = not yet implemented for GGUF.
        parse_gguf: Option<fn(&GgufLoader, &str) -> ModelConfigResult<Option<MetaValue>>>,
    },
    /// Post-processing: depends on other already-resolved fields.
    /// Handled in apply_post_process(), not in the main loop.
    PostProcess,
}

// ── FieldDef: a single field definition in the registry ──

/// A single field definition in the registry.
pub struct FieldDef {
    /// Canonical field name (matches CanonicalConfig field name).
    pub canonical: &'static str,
    /// How to resolve this field.
    pub kind: FieldKind,
    /// Whether this field is required (missing = error).
    pub required: bool,
    /// Default value provider (None = no default for optional fields).
    pub default: Option<fn() -> MetaValue>,
}

// ── CanonicalConfig: intermediate config with all fields as Option ──

/// Intermediate config with all fields as Option — filled by apply_field_registry.
/// Each field is set independently; cross-field defaults (head_dim from hidden/num_heads,
/// kv_cache_block_size from head_dim) are applied in apply_post_process().
#[derive(Debug, Default)]
pub struct CanonicalConfig {
    // Core geometry (required)
    pub hidden_size: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub vocab_size: Option<usize>,

    // Core geometry (optional)
    pub intermediate_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub head_dim: Option<usize>,

    // RoPE
    pub rope_theta: Option<f32>,
    pub rope_scale: Option<f32>,
    pub rope_interleaved: Option<bool>,
    pub global_rope_theta: Option<f32>,
    pub rope_partial_ratio: Option<f32>,
    pub rope_partial_ratio_global: Option<f32>,
    pub rope_scaling: Option<RopeScalingConfig>,

    // Normalization
    pub layer_norm_epsilon: Option<f32>,
    pub attention_dropout: Option<f32>,

    // Dtype
    pub torch_dtype: Option<String>,

    // MoE
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub expert_intermediate_size: Option<usize>,

    // Gemma 4 specific
    pub sliding_window: Option<usize>,
    pub num_kv_shared_layers: Option<usize>,
    pub global_head_dim: Option<usize>,
    pub hidden_size_per_layer_input: Option<usize>,
    pub final_logit_softcapping: Option<f32>,
    pub feed_forward_lengths: Option<Vec<usize>>,
    pub attention_pattern: Option<Vec<u8>>,

    // BUILD-stage hints
    pub qk_norm: Option<bool>,
    pub value_norm: Option<bool>,
    pub embedding_scale_factor: Option<f32>,

    // MTP
    pub mtp_depth: Option<usize>,

    // MLA
    pub mla_config: Option<MlaConfig>,
    pub mla_use_unabsorbed: Option<bool>,

    // Multimodal
    pub vision_config: Option<crate::compat::vision_forward::VisionConfig>,
    pub audio_config: Option<crate::compat::audio_forward::AudioConfig>,
    pub multimodal_token_ids: Option<crate::compat::multimodal::MultimodalTokenIds>,

    // Multimodal token ID inputs (used only in post_process to compute multimodal_token_ids)
    pub image_token_id: Option<u32>,
    pub audio_token_id: Option<u32>,
    pub eoi_token_id: Option<u32>,
    pub eoa_token_id: Option<u32>,

    // Misc
    pub use_cache: Option<bool>,
    pub tie_word_embeddings: Option<bool>,
    pub hidden_act: Option<String>,
    pub use_double_wide_mlp: Option<bool>,
    pub add_special_tokens: Option<bool>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,

    // PostProcess-only fields
    pub kv_cache_block_size: Option<usize>,

    // Unabsorbed threshold (part of MLA but parsed separately as alias)
    pub unabsorbed_threshold: Option<usize>,
}

// ── normalize_text_config: flatten text_config.* fields to root ──

/// Flatten `text_config.*` fields to root level.
///
/// Root-level fields take priority over text_config fields (no overwrite).
/// This eliminates the need to duplicate text_config.* variants in every FieldDef.
///
/// Example: `{ "hidden_size": 768, "text_config": { "hidden_size": 1024, "num_heads": 16 } }`
/// → `{ "hidden_size": 768, "num_heads": 16 }`  (root hidden_size wins)
pub fn normalize_text_config(value: &Value) -> Value {
    let text_config = match value.get("text_config") {
        Some(tc) if tc.is_object() => tc,
        _ => return value.clone(), // No text_config, return as-is
    };

    let mut merged = value.clone();
    if let Some(obj) = merged.as_object_mut() {
        // Remove text_config from merged (we've extracted its fields)
        obj.remove("text_config");

        // Merge text_config fields into root (root takes priority — or_insert, not insert)
        if let Some(tc_obj) = text_config.as_object() {
            for (key, val) in tc_obj {
                obj.entry(key.clone()).or_insert(val.clone());
            }
        }
    }
    merged
}

// ── Derived parse functions ──

/// Parse rope_theta with complex fallback logic:
/// 1. Direct rope_theta / rope_base / rope_base_value
/// 2. rope_parameters.rope_theta (HF LlamaRoPE standard)
/// 3. rope_parameters.sliding_attention.rope_theta (Gemma 4 sliding)
/// 4. Default: 0.0 for encoder (position_embedding_type=absolute), 10000.0 for decoder
fn parse_rope_theta(value: &Value) -> ModelConfigResult<Option<MetaValue>> {
    let theta = find_f32(value, &[
        "rope_theta", "rope_base", "rope_base_value",
    ])
    .or_else(|| find_f32(value, &["rope_parameters.rope_theta"]))
    .or_else(|| find_f32(value, &["rope_parameters.sliding_attention.rope_theta"]))
    .unwrap_or_else(|| {
        // BCE-20260627-035: rope_theta missing → distinguish encoder vs decoder.
        let pos_embed = find_string(value, &["position_embedding_type"]);
        let model_type = find_string(value, &["model_type"]);
        let is_absolute_encoder = pos_embed.as_deref() == Some("absolute")
            || model_type.as_deref().map(|t| {
                matches!(
                    t.to_lowercase().as_str(),
                    "bert" | "xlm-roberta" | "xlm_roberta" | "roberta"
                )
            }).unwrap_or(false);
        if is_absolute_encoder {
            0.0 // Encoder models (BERT/XLM-R) have no RoPE
        } else {
            10000.0 // Llama/Mixtral/Qwen decoder standard base frequency
        }
    });
    Ok(Some(MetaValue::F32(theta)))
}

/// Parse rope_scaling from the rope_scaling sub-object.
fn parse_rope_scaling_json(value: &Value) -> ModelConfigResult<Option<MetaValue>> {
    // rope_scaling_from_metadata_json can return Ok(None) when no scaling info exists,
    // or Err when scaling config is present but invalid (negative factor, etc.).
    // Invalid scaling must still produce a RopeScaling value so validation in from_value
    // can reject it — silently dropping errors would let invalid configs pass through.
    match rope_scaling_from_metadata_json(value) {
        Ok(Some(config)) => Ok(Some(MetaValue::RopeScaling(Some(config)))),
        Ok(None) => Ok(None),
        Err(e) => Err(e)
    }
}

/// Parse attention_pattern from JSON.
/// Sources: "attention_pattern" integer array, or "layer_types" string array.
fn parse_attention_pattern(value: &Value) -> ModelConfigResult<Option<MetaValue>> {
    // Primary: attention_pattern as integer array
    if let Some(arr) = value.get("attention_pattern").and_then(|v| v.as_array()) {
        let pattern: Vec<u8> = arr.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect();
        if !pattern.is_empty() {
            return Ok(Some(MetaValue::AttentionPattern(pattern)));
        }
    }
    // Fallback: layer_types string array → 0=sliding, 1=global
    let lt = match value.get("layer_types").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Ok(None),
    };
    let pattern: Vec<u8> = lt.iter().filter_map(|v| {
        let s = v.as_str()?;
        Some(match s {
            "full_attention" => 1u8,
            _ => 0u8,
        })
    }).collect();
    if !pattern.is_empty() {
        Ok(Some(MetaValue::AttentionPattern(pattern)))
    } else {
        Ok(None)
    }
}

/// Parse MLA config from kv_lora_rank + d_rope.
/// NO-SILENT-FALLBACK: d_rope must be explicit.
fn parse_mla_config_json(value: &Value) -> ModelConfigResult<Option<MetaValue>> {
    let d_c = match find_usize(value, &["kv_lora_rank"]) {
        Some(v) => v,
        None => return Ok(None),
    };
    let d_rope = find_usize(value, &[
        "rope_dimension_count", "qk_rope_head_dim",
    ]).ok_or_else(|| ModelConfigError::InvalidConfig(
        "MLA config missing: d_rope (rope_dimension_count / qk_rope_head_dim) \
         is required when kv_lora_rank is present — NO-SILENT-FALLBACK".to_string()
    ))?;
    let unabsorbed_threshold = find_usize(value, &[
        "mla_use_unabsorbed_threshold",
        "attention.mla.unabsorbed_threshold",
    ]);
    Ok(Some(MetaValue::MlaConfig(MlaConfig {
        d_c,
        d_rope,
        unabsorbed_threshold,
    })))
}

/// Parse vision_config sub-object.
fn parse_vision_config_json(value: &Value) -> ModelConfigResult<Option<MetaValue>> {
    let vc = match value.get("vision_config") {
        Some(v) => v,
        None => return Ok(None),
    };
    let image_size = match find_usize(vc, &["image_size"]) {
        Some(v) => v,
        None => return Ok(None),
    };
    let patch_size = match find_usize(vc, &["patch_size"]) {
        Some(v) => v,
        None => return Ok(None),
    };
    let vis_hidden = match find_usize(vc, &["hidden_size"]) {
        Some(v) => v,
        None => return Ok(None),
    };
    let num_layers = match find_usize(vc, &["num_hidden_layers", "num_layers"]) {
        Some(v) => v,
        None => return Ok(None),
    };
    let num_heads = match find_usize(vc, &["num_attention_heads", "num_heads"]) {
        Some(v) => v,
        None => return Ok(None),
    };
    let vis_intermediate = match find_usize(vc, &["intermediate_size"]) {
        Some(v) => v,
        None => return Ok(None),
    };
    Ok(Some(MetaValue::VisionConfig(crate::compat::vision_forward::VisionConfig {
        image_size,
        patch_size,
        hidden_size: vis_hidden,
        num_layers,
        num_heads,
        intermediate_size: vis_intermediate,
    })))
}

/// Parse audio_config sub-object.
fn parse_audio_config_json(value: &Value) -> ModelConfigResult<Option<MetaValue>> {
    let ac = match value.get("audio_config") {
        Some(v) => v,
        None => return Ok(None),
    };
    let default = crate::compat::audio_forward::AudioConfig::default();
    let hidden_size = find_usize(ac, &["hidden_size"]).unwrap_or(default.hidden_size);
    let num_layers = find_usize(ac, &["num_hidden_layers", "num_layers"]).unwrap_or(default.num_layers);
    let num_heads = find_usize(ac, &["num_attention_heads", "num_heads"]).unwrap_or(default.num_heads);
    let intermediate_size = find_usize(ac, &["intermediate_size"]).unwrap_or(default.intermediate_size);
    let conv_kernel_size = find_usize(ac, &["conv_kernel_size", "depthwise_kernel_size"])
        .unwrap_or(default.conv_kernel_size);
    let sample_rate = find_usize(ac, &["sample_rate", "sampling_rate"]).unwrap_or(default.sample_rate);
    let num_mel_bins = find_usize(ac, &["num_mel_bins", "n_mels"]).unwrap_or(default.num_mel_bins);
    let fft_size = find_usize(ac, &["fft_size", "n_fft"]).unwrap_or(default.fft_size);
    let hop_length = find_usize(ac, &["hop_length"]).unwrap_or(default.hop_length);
    let win_length = find_usize(ac, &["win_length"]).unwrap_or(default.win_length);
    let cfg = crate::compat::audio_forward::AudioConfig {
        sample_rate,
        hidden_size,
        num_layers,
        num_heads,
        conv_kernel_size,
        intermediate_size,
        num_mel_bins,
        fft_size,
        hop_length,
        win_length,
        layer_norm_eps: default.layer_norm_eps,
        stride: default.stride,
    };
    Ok(cfg.validate().ok().map(|_| MetaValue::AudioConfig(cfg)))
}

/// Parse feed_forward_lengths from JSON array.
fn parse_feed_forward_lengths(value: &Value) -> ModelConfigResult<Option<MetaValue>> {
    let arr = match value.get("feed_forward_lengths").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Ok(None),
    };
    let lengths: Vec<usize> = arr.iter()
        .filter_map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()))
        .collect();
    if lengths.is_empty() {
        Ok(None)
    } else {
        Ok(Some(MetaValue::UsizeArray(lengths)))
    }
}

// ── parse_gguf functions: GGUF-specific Derived field parsing ──
//
// These mirror the parse_json functions but read from a GgufReader (GGUF metadata)
// instead of a JSON Value. Each receives the reader and arch prefix, returns
// Ok(None) when the field is absent, Err on invalid data.
//
// Used by apply_gguf_field_registry() to resolve Derived fields in the GGUF path.

/// Parse rope_theta for GGUF: prefer rope_scaling.base, fall back to rope.freq_base.
fn parse_rope_theta_gguf(reader: &GgufLoader, arch: &str) -> ModelConfigResult<Option<MetaValue>> {
    // rope_scaling may already be resolved separately, but we re-derive its base here
    // to stay self-contained (the registry resolves fields independently).
    let scaling_base = gguf_arch_f32(reader, arch, "rope.scaling.base")
        .or_else(|| gguf_arch_f32(reader, arch, "rope.freq_base"));
    let freq_base = reader.rope_freq_base();
    let value = scaling_base.or(freq_base);
    match value {
        Some(v) if v.is_finite() && v > 0.0 => Ok(Some(MetaValue::F32(v))),
        Some(_) => Err(ModelConfigError::InvalidConfig(
            "GGUF metadata invalid: rope.freq_base (rope_theta) must be > 0".to_string(),
        )),
        None => Ok(None),
    }
}

/// Parse rope_scaling for GGUF via the existing rope_scaling_from_gguf helper.
fn parse_rope_scaling_gguf(reader: &GgufLoader, arch: &str) -> ModelConfigResult<Option<MetaValue>> {
    let cfg = rope_scaling_from_gguf(reader, arch)?;
    Ok(cfg.map(|c| MetaValue::RopeScaling(Some(c))))
}

/// Parse attention_pattern for GGUF: array_u8 from multiple sources + bool→u8 fallback.
fn parse_attention_pattern_gguf(reader: &GgufLoader, arch: &str) -> ModelConfigResult<Option<MetaValue>> {
    let pattern = gguf_arch_array_u8(reader, arch, "attention.pattern")
        .or_else(|| gguf_arch_array_u8(reader, arch, "attention_pattern"))
        .or_else(|| {
            // Gemma 4 GGUF (unsloth): bool array → convert to u8
            // (true=sliding→0, false=global→1 per the from_gguf_loader convention)
            gguf_arch_array_bool(reader, arch, "attention.sliding_window_pattern")
                .map(|bools| bools.into_iter().map(|b| if b { 0 } else { 1 }).collect())
        });
    Ok(pattern.map(MetaValue::AttentionPattern))
}

/// Parse feed_forward_lengths for GGUF: per-layer FFN intermediate sizes array.
/// Only returns when the array is heterogeneous (len > 1 and not all-equal),
/// matching the from_gguf_loader contract for Gemma 4 E2B.
fn parse_feed_forward_lengths_gguf(reader: &GgufLoader, arch: &str) -> ModelConfigResult<Option<MetaValue>> {
    let lengths = gguf_arch_array_usize(reader, arch, "feed_forward_length")
        .filter(|arr| arr.len() > 1 && arr.iter().any(|&v| v != arr[0]));
    Ok(lengths.map(MetaValue::UsizeArray))
}

/// Parse mla_config for GGUF: kv_lora_rank + qk_rope_head_dim joint construction.
/// Returns Ok(None) when kv_lora_rank is absent (non-MLA models).
/// Returns Err when kv_lora_rank present but qk_rope_head_dim missing (NO-SILENT-FALLBACK).
fn parse_mla_config_gguf(reader: &GgufLoader, _arch: &str) -> ModelConfigResult<Option<MetaValue>> {
    match reader.kv_lora_rank() {
        Some(d_c) => {
            // NO-SILENT-FALLBACK: MLA d_rope (qk_rope_head_dim) must be explicit.
            let d_rope = reader
                .qk_rope_head_dim()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    ModelConfigError::InvalidConfig(
                        "GGUF metadata missing: qk_rope_head_dim (MLA d_rope)".to_string(),
                    )
                })?;
            Ok(Some(MetaValue::MlaConfig(MlaConfig {
                d_c: usize::try_from(d_c).unwrap_or(0),
                d_rope,
                unabsorbed_threshold: None,
            })))
        }
        None => Ok(None),
    }
}

// ── FIELD_DEFS: the static registry ──

/// Declarative field registry — all config.json field mappings in one place.
///
/// json_keys list ONLY the non-text_config variant. normalize_text_config() handles
/// the text_config.* flattening before lookup, so every field automatically gains
/// text_config.* coverage without duplicating key strings.
///
/// This eliminates ~36 duplicated "text_config.*" key strings from from_value().
static FIELD_DEFS: &[FieldDef] = &[
    // ── Core geometry (required) ──
    FieldDef {
        canonical: "hidden_size",
        kind: FieldKind::Alias { json_keys: &["hidden_size", "n_embd", "d_model"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.embedding_length().and_then(|v| usize::try_from(v).ok()).map(MetaValue::Usize)) },
        required: true,
        default: None,
    },
    FieldDef {
        canonical: "num_attention_heads",
        kind: FieldKind::Alias { json_keys: &["num_attention_heads", "n_head", "num_heads"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.head_count().and_then(|v| usize::try_from(v).ok()).map(MetaValue::Usize)) },
        required: true,
        default: None,
    },
    FieldDef {
        canonical: "num_hidden_layers",
        kind: FieldKind::Alias { json_keys: &["num_hidden_layers", "n_layer", "num_layers"], gguf_keys: &[], gguf_reader: None },
        required: true,
        default: None,
    },
    FieldDef {
        canonical: "vocab_size",
        kind: FieldKind::Alias { json_keys: &["vocab_size"], gguf_keys: &[], gguf_reader: None },
        required: true,
        default: None,
    },

    // ── Core geometry (optional with defaults) ──
    FieldDef {
        canonical: "num_key_value_heads",
        kind: FieldKind::Alias { json_keys: &["num_key_value_heads", "num_kv_heads", "n_kv_head", "attention.num_key_value_heads"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.head_count_kv().and_then(|v| usize::try_from(v).ok()).map(MetaValue::Usize)) },
        required: false,
        default: None, // default = num_attention_heads (set in post_process)
    },
    FieldDef {
        canonical: "intermediate_size",
        kind: FieldKind::Alias { json_keys: &["intermediate_size", "n_inner", "ffn_inter_dim", "d_ff"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.feed_forward_length().and_then(|v| usize::try_from(v).ok()).map(MetaValue::Usize)) },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "max_position_embeddings",
        kind: FieldKind::Alias { json_keys: &["max_position_embeddings", "max_seq_len", "max_sequence_length", "seq_length", "n_positions", "rope_scaling.original_max_position_embeddings"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.context_length().and_then(|v| usize::try_from(v).ok()).map(MetaValue::Usize)) },
        required: false,
        default: None, // no default at registry level; from_value validates after manifest override
    },
    FieldDef {
        canonical: "head_dim",
        kind: FieldKind::Alias { json_keys: &["head_dim", "attention.head_dim", "kv_channels"], gguf_keys: &["attention.head_dim", "attention.key_length_swa"], gguf_reader: None },
        required: false,
        default: None, // default = hidden_size/num_attention_heads (post_process)
    },

    // ── MoE ──
    FieldDef {
        canonical: "num_experts",
        kind: FieldKind::Alias { json_keys: &["num_experts", "num_local_experts", "n_routed_experts"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.num_experts().and_then(|v| usize::try_from(v).ok()).map(MetaValue::Usize)) },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "num_experts_per_tok",
        kind: FieldKind::Alias { json_keys: &["num_experts_per_tok", "num_selected_experts", "num_experts_per_token"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.num_experts_per_tok().and_then(|v| usize::try_from(v).ok()).map(MetaValue::Usize)) },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "expert_intermediate_size",
        kind: FieldKind::Alias { json_keys: &["expert_intermediate_size", "moe_intermediate_size"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.expert_intermediate_size().and_then(|v| usize::try_from(v).ok()).map(MetaValue::Usize)) },
        required: false,
        default: None,
    },

    // ── RoPE ──
    FieldDef {
        canonical: "rope_theta",
        kind: FieldKind::Derived { parse_json: parse_rope_theta, parse_gguf: Some(parse_rope_theta_gguf) },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "rope_scale",
        kind: FieldKind::Alias { json_keys: &["rope_scale", "rope_factor"], gguf_keys: &["rope.scale"], gguf_reader: None },
        required: false,
        default: Some(|| MetaValue::F32(1.0)),
    },
    FieldDef {
        canonical: "rope_interleaved",
        kind: FieldKind::Alias { json_keys: &["rope_interleaved", "rotary_interleaved"], gguf_keys: &["rope.interleaved"], gguf_reader: None },
        required: false,
        default: Some(|| MetaValue::Bool(false)),
    },
    FieldDef {
        canonical: "global_rope_theta",
        kind: FieldKind::Alias { json_keys: &["global_rope_theta", "rope_parameters.full_attention.rope_theta"], gguf_keys: &["rope.global.freq_base", "global_rope_theta"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "rope_partial_ratio",
        kind: FieldKind::Alias { json_keys: &["rope_partial_ratio", "partial_rotary_factor", "rope_parameters.sliding_attention.partial_rotary_factor"], gguf_keys: &["rope.partial_ratio", "rope.global.partial_ratio"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "rope_partial_ratio_global",
        kind: FieldKind::Alias { json_keys: &["rope_parameters.full_attention.partial_rotary_factor"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },
    // Gemma 4 intermediate rope_parameters fields (used by post_process to derive
    // (rope_parameters.* paths are handled within parse_rope_theta and the
    // global_rope_theta/rope_partial_ratio json_keys, so no separate fields needed)
    FieldDef {
        canonical: "rope_scaling",
        kind: FieldKind::Derived { parse_json: parse_rope_scaling_json, parse_gguf: Some(parse_rope_scaling_gguf) },
        required: false,
        default: None,
    },

    // ── Normalization ──
    FieldDef {
        canonical: "layer_norm_epsilon",
        kind: FieldKind::Alias { json_keys: &["layer_norm_epsilon", "layer_norm_eps", "rms_norm_eps", "norm_epsilon"], gguf_keys: &["layer_norm_epsilon", "layer_norm_rms_epsilon", "attention.layer_norm_rms_epsilon"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "attention_dropout",
        kind: FieldKind::Alias { json_keys: &["attention_dropout", "attention.dropout"], gguf_keys: &["attention.dropout"], gguf_reader: None },
        required: false,
        default: None,
    },

    // ── Dtype ──
    FieldDef {
        canonical: "torch_dtype",
        kind: FieldKind::Alias { json_keys: &["torch_dtype", "dtype"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },

    // ── Gemma 4 / special ──
    FieldDef {
        canonical: "sliding_window",
        kind: FieldKind::Alias { json_keys: &["sliding_window", "sliding_window_size"], gguf_keys: &["attention.sliding_window", "sliding_window"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "num_kv_shared_layers",
        kind: FieldKind::Alias { json_keys: &["num_kv_shared_layers"], gguf_keys: &["attention.shared_kv_layers"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "global_head_dim",
        kind: FieldKind::Alias { json_keys: &["global_head_dim"], gguf_keys: &["attention.global_head_dim", "global_head_dim", "attention.key_length"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "hidden_size_per_layer_input",
        kind: FieldKind::Alias { json_keys: &["hidden_size_per_layer_input"], gguf_keys: &["embedding.per_layer_input", "hidden_size_per_layer_input", "embedding_length_per_layer_input"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "final_logit_softcapping",
        kind: FieldKind::Alias { json_keys: &["final_logit_softcapping"], gguf_keys: &["final_logit_softcapping"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "feed_forward_lengths",
        kind: FieldKind::Derived { parse_json: parse_feed_forward_lengths, parse_gguf: Some(parse_feed_forward_lengths_gguf) },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "attention_pattern",
        kind: FieldKind::Derived { parse_json: parse_attention_pattern, parse_gguf: Some(parse_attention_pattern_gguf) },
        required: false,
        default: None,
    },

    // ── BUILD-stage hints ──
    FieldDef {
        canonical: "qk_norm",
        kind: FieldKind::Alias { json_keys: &["qk_norm", "attention.qk_norm"], gguf_keys: &["attention.qk_norm"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "value_norm",
        kind: FieldKind::Alias { json_keys: &["value_norm", "attention.value_norm"], gguf_keys: &["attention.value_norm"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "embedding_scale_factor",
        kind: FieldKind::Alias { json_keys: &["embedding_scale_factor"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },

    // ── MTP ──
    FieldDef {
        canonical: "mtp_depth",
        kind: FieldKind::Alias { json_keys: &["num_nextn_predict_layers", "mtp_depth"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.mtp_depth().and_then(|v| usize::try_from(v).ok()).map(MetaValue::Usize)) },
        required: false,
        default: None,
    },

    // ── MLA ──
    FieldDef {
        canonical: "mla_config",
        kind: FieldKind::Derived { parse_json: parse_mla_config_json, parse_gguf: Some(parse_mla_config_gguf) },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "mla_use_unabsorbed",
        kind: FieldKind::Alias { json_keys: &["mla_use_unabsorbed", "attention.mla.use_unabsorbed"], gguf_keys: &["attention.mla.use_unabsorbed"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "unabsorbed_threshold",
        kind: FieldKind::Alias { json_keys: &["mla_use_unabsorbed_threshold", "attention.mla.unabsorbed_threshold"], gguf_keys: &["attention.mla.unabsorbed_threshold"], gguf_reader: None },
        required: false,
        default: None,
    },

    // ── Multimodal ──
    FieldDef {
        canonical: "vision_config",
        kind: FieldKind::Derived { parse_json: parse_vision_config_json, parse_gguf: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "audio_config",
        kind: FieldKind::Derived { parse_json: parse_audio_config_json, parse_gguf: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "multimodal_token_ids",
        kind: FieldKind::PostProcess,
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "image_token_id",
        kind: FieldKind::Alias { json_keys: &["image_token_id", "boi_token_id"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "audio_token_id",
        kind: FieldKind::Alias { json_keys: &["audio_token_id", "boa_token_id"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "eoi_token_id",
        kind: FieldKind::Alias { json_keys: &["eoi_token_id", "image_end_token_id"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "eoa_token_id",
        kind: FieldKind::Alias { json_keys: &["eoa_token_id", "audio_end_token_id"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },

    // ── Misc ──
    FieldDef {
        canonical: "use_cache",
        kind: FieldKind::Alias { json_keys: &["use_cache"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "tie_word_embeddings",
        kind: FieldKind::Alias { json_keys: &["tie_word_embeddings"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "hidden_act",
        kind: FieldKind::Alias { json_keys: &["hidden_act", "hidden_activation"], gguf_keys: &["feed_forward.activation", "hidden_act"], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "use_double_wide_mlp",
        kind: FieldKind::Alias { json_keys: &["use_double_wide_mlp"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "add_special_tokens",
        kind: FieldKind::Alias { json_keys: &["add_bos_token", "add_special_tokens"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "bos_token_id",
        kind: FieldKind::Alias { json_keys: &["bos_token_id"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.bos_token_id().map(MetaValue::U32)) },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "eos_token_id",
        kind: FieldKind::Alias { json_keys: &["eos_token_id"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.eos_token_id().map(MetaValue::U32)) },
        required: false,
        default: None,
    },
    FieldDef {
        canonical: "pad_token_id",
        kind: FieldKind::Alias { json_keys: &["pad_token_id"], gguf_keys: &[], gguf_reader: Some(|r: &GgufLoader| r.get_metadata_u64("tokenizer.ggml.padding_token_id").and_then(|v| u32::try_from(v).ok()).map(MetaValue::U32)) },
        required: false,
        default: None,
    },

    // ── PostProcess-only fields ──
    FieldDef {
        canonical: "kv_cache_block_size",
        kind: FieldKind::Alias { json_keys: &["kv_cache_block_size", "kv_block_size", "page_size"], gguf_keys: &[], gguf_reader: None },
        required: false,
        default: None, // default = head_dim.max(num_key_value_heads) in post_process
    },
];

// ── set_canonical_field: type-specific extraction from JSON Value ──

/// Set a canonical field from a raw JSON Value.
/// Dispatches on canonical name to determine the target type.
fn set_canonical_field(
    canonical: &mut CanonicalConfig,
    name: &str,
    value: &Value,
) -> ModelConfigResult<()> {
    match name {
        // usize fields
        "hidden_size" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.hidden_size = Some(v);
            }
        }
        "num_attention_heads" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.num_attention_heads = Some(v);
            }
        }
        "num_key_value_heads" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.num_key_value_heads = Some(v);
            }
        }
        "num_hidden_layers" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.num_hidden_layers = Some(v);
            }
        }
        "vocab_size" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.vocab_size = Some(v);
            }
        }
        "intermediate_size" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.intermediate_size = Some(v);
            }
        }
        "max_position_embeddings" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.max_position_embeddings = Some(v);
            }
        }
        "head_dim" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.head_dim = Some(v);
            }
        }
        "num_experts" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.num_experts = Some(v);
            }
        }
        "num_experts_per_tok" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.num_experts_per_tok = Some(v);
            }
        }
        "expert_intermediate_size" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.expert_intermediate_size = Some(v);
            }
        }
        "sliding_window" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.sliding_window = Some(v);
            }
        }
        "num_kv_shared_layers" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.num_kv_shared_layers = Some(v);
            }
        }
        "global_head_dim" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.global_head_dim = Some(v);
            }
        }
        "hidden_size_per_layer_input" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.hidden_size_per_layer_input = Some(v);
            }
        }
        "mtp_depth" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.mtp_depth = Some(v);
            }
        }
        "unabsorbed_threshold" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.unabsorbed_threshold = Some(v);
            }
        }
        "kv_cache_block_size" => {
            if let Some(v) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
                canonical.kv_cache_block_size = Some(v);
            }
        }

        // f32 fields
        "rope_theta" => {
            if let Some(v) = value.as_f64().map(|n| n as f32) {
                canonical.rope_theta = Some(v);
            }
        }
        "rope_scale" => {
            if let Some(v) = value.as_f64().map(|n| n as f32) {
                canonical.rope_scale = Some(v);
            }
        }
        "global_rope_theta" => {
            if let Some(v) = value.as_f64().map(|n| n as f32) {
                canonical.global_rope_theta = Some(v);
            }
        }
        "rope_partial_ratio" => {
            if let Some(v) = value.as_f64().map(|n| n as f32) {
                canonical.rope_partial_ratio = Some(v);
            }
        }
        "rope_partial_ratio_global" => {
            if let Some(v) = value.as_f64().map(|n| n as f32) {
                canonical.rope_partial_ratio_global = Some(v);
            }
        }
        "layer_norm_epsilon" => {
            if let Some(v) = value.as_f64().map(|n| n as f32) {
                canonical.layer_norm_epsilon = Some(v);
            }
        }
        "attention_dropout" => {
            if let Some(v) = value.as_f64().map(|n| n as f32) {
                canonical.attention_dropout = Some(v);
            }
        }
        "final_logit_softcapping" => {
            if let Some(v) = value.as_f64().map(|n| n as f32) {
                canonical.final_logit_softcapping = Some(v);
            }
        }
        "embedding_scale_factor" => {
            if let Some(v) = value.as_f64().map(|n| n as f32) {
                canonical.embedding_scale_factor = Some(v);
            }
        }

        // bool fields
        "rope_interleaved" => {
            if let Some(v) = value.as_bool().or_else(|| {
                value.as_u64().and_then(|num| match num {
                    0 => Some(false),
                    1 => Some(true),
                    _ => None,
                })
            }) {
                canonical.rope_interleaved = Some(v);
            }
        }
        "use_cache" => {
            if let Some(v) = value.as_bool().or_else(|| {
                value.as_u64().map(|n| n != 0)
            }) {
                canonical.use_cache = Some(v);
            }
        }
        "tie_word_embeddings" => {
            if let Some(v) = value.as_bool().or_else(|| {
                value.as_u64().map(|n| n != 0)
            }) {
                canonical.tie_word_embeddings = Some(v);
            }
        }
        "use_double_wide_mlp" => {
            if let Some(v) = value.as_bool().or_else(|| {
                value.as_u64().map(|n| n != 0)
            }) {
                canonical.use_double_wide_mlp = Some(v);
            }
        }
        "add_special_tokens" => {
            if let Some(v) = value.as_bool().or_else(|| {
                value.as_u64().map(|n| n != 0)
            }) {
                canonical.add_special_tokens = Some(v);
            }
        }
        "qk_norm" => {
            if let Some(v) = value.as_bool().or_else(|| {
                value.as_u64().map(|n| n != 0)
            }) {
                canonical.qk_norm = Some(v);
            }
        }
        "value_norm" => {
            if let Some(v) = value.as_bool().or_else(|| {
                value.as_u64().map(|n| n != 0)
            }) {
                canonical.value_norm = Some(v);
            }
        }
        "mla_use_unabsorbed" => {
            if let Some(v) = value.as_bool().or_else(|| {
                value.as_u64().map(|n| n != 0)
            }) {
                canonical.mla_use_unabsorbed = Some(v);
            }
        }

        // string fields
        "torch_dtype" => {
            if let Some(v) = value.as_str().map(ToOwned::to_owned) {
                canonical.torch_dtype = Some(v);
            }
        }
        "hidden_act" => {
            if let Some(v) = value.as_str().map(ToOwned::to_owned) {
                canonical.hidden_act = Some(v);
            }
        }

        // u32 fields
        "bos_token_id" => {
            if let Some(v) = value.as_u64().and_then(|n| u32::try_from(n).ok()) {
                canonical.bos_token_id = Some(v);
            }
        }
        "eos_token_id" => {
            if let Some(v) = value.as_u64().and_then(|n| u32::try_from(n).ok()) {
                canonical.eos_token_id = Some(v);
            }
        }
        "pad_token_id" => {
            if let Some(v) = value.as_u64().and_then(|n| u32::try_from(n).ok()) {
                canonical.pad_token_id = Some(v);
            }
        }
        "image_token_id" => {
            if let Some(v) = value.as_u64().and_then(|n| u32::try_from(n).ok()) {
                canonical.image_token_id = Some(v);
            }
        }
        "audio_token_id" => {
            if let Some(v) = value.as_u64().and_then(|n| u32::try_from(n).ok()) {
                canonical.audio_token_id = Some(v);
            }
        }
        "eoi_token_id" => {
            if let Some(v) = value.as_u64().and_then(|n| u32::try_from(n).ok()) {
                canonical.eoi_token_id = Some(v);
            }
        }
        "eoa_token_id" => {
            if let Some(v) = value.as_u64().and_then(|n| u32::try_from(n).ok()) {
                canonical.eoa_token_id = Some(v);
            }
        }

        // Complex fields handled by Derived — these should not reach set_canonical_field
        // because Derived parse functions return MetaValue directly.
        "rope_scaling" | "attention_pattern" | "mla_config" | "vision_config"
        | "audio_config" | "multimodal_token_ids" | "feed_forward_lengths" => {
            // These are handled by set_canonical_from_meta via Derived parse functions.
            // If we reach here, it's a bug — the caller should use set_canonical_from_meta.
        }

        _ => {
            // Unknown canonical name — silently ignore for forward compatibility
        }
    }
    Ok(())
}

/// Set a canonical field from a MetaValue (produced by Derived parse functions or defaults).
fn set_canonical_from_meta(
    canonical: &mut CanonicalConfig,
    name: &str,
    meta: MetaValue,
) -> ModelConfigResult<()> {
    match name {
        "rope_theta" => { if let MetaValue::F32(v) = meta { canonical.rope_theta = Some(v); } }
        "rope_scale" => { if let MetaValue::F32(v) = meta { canonical.rope_scale = Some(v); } }
        "rope_interleaved" => { if let MetaValue::Bool(v) = meta { canonical.rope_interleaved = Some(v); } }
        "global_rope_theta" => { if let MetaValue::F32(v) = meta { canonical.global_rope_theta = Some(v); } }
        "rope_partial_ratio" => { if let MetaValue::F32(v) = meta { canonical.rope_partial_ratio = Some(v); } }
        "rope_partial_ratio_global" => { if let MetaValue::F32(v) = meta { canonical.rope_partial_ratio_global = Some(v); } }
        "rope_scaling" => { if let MetaValue::RopeScaling(v) = meta { canonical.rope_scaling = v; } }
        "attention_pattern" => { if let MetaValue::AttentionPattern(v) = meta { canonical.attention_pattern = Some(v); } }
        "mla_config" => { if let MetaValue::MlaConfig(v) = meta { canonical.mla_config = Some(v); } }
        "vision_config" => { if let MetaValue::VisionConfig(v) = meta { canonical.vision_config = Some(v); } }
        "audio_config" => { if let MetaValue::AudioConfig(v) = meta { canonical.audio_config = Some(v); } }
        "multimodal_token_ids" => { if let MetaValue::MultimodalTokenIds(v) = meta { canonical.multimodal_token_ids = Some(v); } }
        "feed_forward_lengths" => { if let MetaValue::UsizeArray(v) = meta { canonical.feed_forward_lengths = Some(v); } }
        "embedding_scale_factor" => { if let MetaValue::F32(v) = meta { canonical.embedding_scale_factor = Some(v); } }
        "mla_use_unabsorbed" => { if let MetaValue::Bool(v) = meta { canonical.mla_use_unabsorbed = Some(v); } }
        "add_special_tokens" => { if let MetaValue::Bool(v) = meta { canonical.add_special_tokens = Some(v); } }
        "use_cache" => { if let MetaValue::Bool(v) = meta { canonical.use_cache = Some(v); } }
        "tie_word_embeddings" => { if let MetaValue::Bool(v) = meta { canonical.tie_word_embeddings = Some(v); } }
        "use_double_wide_mlp" => { if let MetaValue::Bool(v) = meta { canonical.use_double_wide_mlp = Some(v); } }
        "qk_norm" => { if let MetaValue::Bool(v) = meta { canonical.qk_norm = Some(v); } }
        "value_norm" => { if let MetaValue::Bool(v) = meta { canonical.value_norm = Some(v); } }
        "hidden_act" => { if let MetaValue::Str(v) = meta { canonical.hidden_act = Some(v); } }
        "torch_dtype" => { if let MetaValue::Str(v) = meta { canonical.torch_dtype = Some(v); } }
        "layer_norm_epsilon" => { if let MetaValue::F32(v) = meta { canonical.layer_norm_epsilon = Some(v); } }
        "attention_dropout" => { if let MetaValue::F32(v) = meta { canonical.attention_dropout = Some(v); } }
        "final_logit_softcapping" => { if let MetaValue::F32(v) = meta { canonical.final_logit_softcapping = Some(v); } }
        "num_key_value_heads" => { if let MetaValue::Usize(v) = meta { canonical.num_key_value_heads = Some(v); } }
        "intermediate_size" => { if let MetaValue::Usize(v) = meta { canonical.intermediate_size = Some(v); } }
        "max_position_embeddings" => { if let MetaValue::Usize(v) = meta { canonical.max_position_embeddings = Some(v); } }
        "head_dim" => { if let MetaValue::Usize(v) = meta { canonical.head_dim = Some(v); } }
        "num_experts" => { if let MetaValue::Usize(v) = meta { canonical.num_experts = Some(v); } }
        "num_experts_per_tok" => { if let MetaValue::Usize(v) = meta { canonical.num_experts_per_tok = Some(v); } }
        "expert_intermediate_size" => { if let MetaValue::Usize(v) = meta { canonical.expert_intermediate_size = Some(v); } }
        "sliding_window" => { if let MetaValue::Usize(v) = meta { canonical.sliding_window = Some(v); } }
        "num_kv_shared_layers" => { if let MetaValue::Usize(v) = meta { canonical.num_kv_shared_layers = Some(v); } }
        "global_head_dim" => { if let MetaValue::Usize(v) = meta { canonical.global_head_dim = Some(v); } }
        "hidden_size_per_layer_input" => { if let MetaValue::Usize(v) = meta { canonical.hidden_size_per_layer_input = Some(v); } }
        "mtp_depth" => { if let MetaValue::Usize(v) = meta { canonical.mtp_depth = Some(v); } }
        "kv_cache_block_size" => { if let MetaValue::Usize(v) = meta { canonical.kv_cache_block_size = Some(v); } }
        "unabsorbed_threshold" => { if let MetaValue::Usize(v) = meta { canonical.unabsorbed_threshold = Some(v); } }
        "bos_token_id" => { if let MetaValue::U32(v) = meta { canonical.bos_token_id = Some(v); } }
        "eos_token_id" => { if let MetaValue::U32(v) = meta { canonical.eos_token_id = Some(v); } }
        "pad_token_id" => { if let MetaValue::U32(v) = meta { canonical.pad_token_id = Some(v); } }
        "image_token_id" => { if let MetaValue::U32(v) = meta { canonical.image_token_id = Some(v); } }
        "audio_token_id" => { if let MetaValue::U32(v) = meta { canonical.audio_token_id = Some(v); } }
        "eoi_token_id" => { if let MetaValue::U32(v) = meta { canonical.eoi_token_id = Some(v); } }
        "eoa_token_id" => { if let MetaValue::U32(v) = meta { canonical.eoa_token_id = Some(v); } }
        _ => {} // Unknown — forward compat
    }
    Ok(())
}

// ── apply_field_registry: main resolution loop ──

/// Walk FIELD_DEFS, resolve each field from the (normalized) JSON, produce CanonicalConfig.
///
/// Resolution order for Alias fields:
///   1. Try each json_key in order via find_value()
///   2. If found, extract typed value via set_canonical_field()
///   3. If not found and required, return error
///   4. If not found and optional, apply default (if any)
///
/// Derived fields call their parse_json function with the full JSON root.
/// PostProcess fields are skipped (handled in apply_post_process).
pub fn apply_field_registry(value: &Value, defs: &[FieldDef]) -> ModelConfigResult<CanonicalConfig> {
    let mut canonical = CanonicalConfig::default();

    for def in defs {
        match &def.kind {
            FieldKind::Alias { json_keys, gguf_keys: _, gguf_reader: _ } => {
                let found = find_value(value, json_keys);
                match found {
                    Some(v) => {
                        set_canonical_field(&mut canonical, def.canonical, v)?;
                    }
                    None if def.required => {
                        return Err(ModelConfigError::InvalidConfig(
                            format!("missing required config field: {}", def.canonical)
                        ));
                    }
                    None => {
                        // Optional field missing — apply default if any
                        if let Some(default_fn) = def.default {
                            set_canonical_from_meta(&mut canonical, def.canonical, default_fn())?;
                        }
                    }
                }
            }
            FieldKind::Derived { parse_json, .. } => {
                match parse_json(value)? {
                    Some(meta) => {
                        set_canonical_from_meta(&mut canonical, def.canonical, meta)?;
                    }
                    None if def.required => {
                        return Err(ModelConfigError::InvalidConfig(
                            format!("missing required derived config field: {}", def.canonical)
                        ));
                    }
                    None => {
                        // Optional derived field missing — apply default if any
                        if let Some(default_fn) = def.default {
                            set_canonical_from_meta(&mut canonical, def.canonical, default_fn())?;
                        }
                    }
                }
            }
            FieldKind::PostProcess => {
                // Handled in apply_post_process
            }
        }
    }

    Ok(canonical)
}

// ── apply_gguf_field_registry: GGUF mirror of apply_field_registry ──

/// Resolve a single Alias field's GGUF value via the arch-prefixed keys.
///
/// Tries each `gguf_key` in order (resolved as `format!("{arch}.{key}")`).
/// The GGUF metadata type read is determined by the canonical field name:
/// usize fields read u64, f32 fields read f32, bool fields read bool,
/// str fields read str. Arrays are not handled here (they use Derived).
///
/// Returns Ok(None) when no key matches.
fn gguf_alias_lookup(
    reader: &GgufLoader,
    arch: &str,
    canonical: &str,
    gguf_keys: &[&str],
) -> Option<MetaValue> {
    for key in gguf_keys {
        let full_key = format!("{arch}.{key}");
        match canonical {
            // usize-typed fields
            "head_dim" | "sliding_window" | "num_kv_shared_layers" | "global_head_dim"
            | "hidden_size_per_layer_input" | "unabsorbed_threshold" => {
                if let Some(v) = reader.get_metadata_u64(&full_key) {
                    if let Ok(u) = usize::try_from(v) {
                        return Some(MetaValue::Usize(u));
                    }
                }
            }
            // f32-typed fields
            "rope_scale" | "global_rope_theta" | "rope_partial_ratio"
            | "layer_norm_epsilon" | "attention_dropout" | "final_logit_softcapping"
            | "embedding_scale_factor" => {
                if let Some(v) = reader.get_metadata_f32(&full_key) {
                    return Some(MetaValue::F32(v));
                }
            }
            // bool-typed fields
            "rope_interleaved" | "qk_norm" | "value_norm" | "mla_use_unabsorbed" => {
                if let Some(value) = reader.get(&full_key) {
                    if let Some(b) = value.as_bool().or_else(|| value.as_u64().map(|v| v != 0)) {
                        return Some(MetaValue::Bool(b));
                    }
                }
            }
            // str-typed fields
            "hidden_act" | "torch_dtype" => {
                if let Some(s) = reader.get_metadata_str(&full_key) {
                    return Some(MetaValue::Str(s.to_string()));
                }
            }
            _ => {
                // No typed GGUF lookup defined for this canonical — skip.
                // (Fields like mtp_depth/bos_token_id use gguf_reader closures instead.)
            }
        }
    }
    None
}

/// Walk FIELD_DEFS resolving each field from GGUF metadata, producing a CanonicalConfig.
///
/// This is the GGUF mirror of `apply_field_registry()`. It unifies the GGUF parsing
/// path with the JSON path so both converge on the same `CanonicalConfig` →
/// `build_model_config()` assembly.
///
/// Resolution order for Alias fields:
///   1. If `gguf_reader` is Some, call it with the reader (native GgufReader methods
///      like embedding_length() that don't use the arch prefix).
///   2. Else try each `gguf_key` via `gguf_alias_lookup()` (arch-prefixed metadata).
///   3. If still unresolved and required, return error.
///   4. If optional, apply default (if any).
///
/// Derived fields call their `parse_gguf` function (when Some).
/// PostProcess fields are skipped (handled in apply_post_process).
pub fn apply_gguf_field_registry(
    reader: &GgufLoader,
    arch: &str,
    defs: &[FieldDef],
) -> ModelConfigResult<CanonicalConfig> {
    let mut canonical = CanonicalConfig::default();

    for def in defs {
        match &def.kind {
            FieldKind::Alias { json_keys: _, gguf_keys, gguf_reader } => {
                let resolved = if let Some(reader_fn) = gguf_reader {
                    reader_fn(reader)
                } else {
                    None
                };
                let resolved = resolved.or_else(|| {
                    gguf_alias_lookup(reader, arch, def.canonical, gguf_keys)
                });
                match resolved {
                    Some(meta) => {
                        set_canonical_from_meta(&mut canonical, def.canonical, meta)?;
                    }
                    None if def.required => {
                        return Err(ModelConfigError::InvalidConfig(
                            format!("missing required GGUF config field: {}", def.canonical)
                        ));
                    }
                    None => {
                        if let Some(default_fn) = def.default {
                            set_canonical_from_meta(&mut canonical, def.canonical, default_fn())?;
                        }
                    }
                }
            }
            FieldKind::Derived { parse_json: _, parse_gguf } => {
                let resolved = match parse_gguf {
                    Some(parse_fn) => parse_fn(reader, arch)?,
                    None => None,
                };
                match resolved {
                    Some(meta) => {
                        set_canonical_from_meta(&mut canonical, def.canonical, meta)?;
                    }
                    None if def.required => {
                        return Err(ModelConfigError::InvalidConfig(
                            format!("missing required derived GGUF config field: {}", def.canonical)
                        ));
                    }
                    None => {
                        if let Some(default_fn) = def.default {
                            set_canonical_from_meta(&mut canonical, def.canonical, default_fn())?;
                        }
                    }
                }
            }
            FieldKind::PostProcess => {
                // Handled in apply_post_process
            }
        }
    }

    Ok(canonical)
}

// ── apply_post_process: compute fields that depend on already-resolved values ──

/// Apply post-processing defaults that depend on other fields being resolved first.
///
/// - num_key_value_heads defaults to num_attention_heads
/// - head_dim defaults to hidden_size / num_attention_heads
/// - kv_cache_block_size defaults to head_dim.max(num_key_value_heads)
pub fn apply_post_process(canonical: &mut CanonicalConfig) -> ModelConfigResult<()> {
    // num_key_value_heads defaults to num_attention_heads (non-GQA models)
    if canonical.num_key_value_heads.is_none() {
        canonical.num_key_value_heads = canonical.num_attention_heads;
    }

    // head_dim defaults to hidden_size / num_attention_heads
    if canonical.head_dim.is_none() {
        if let (Some(hs), Some(nh)) = (canonical.hidden_size, canonical.num_attention_heads) {
            if nh > 0 {
                canonical.head_dim = Some(hs / nh);
            }
        }
    }

    // kv_cache_block_size defaults to head_dim.max(num_key_value_heads)
    if canonical.kv_cache_block_size.is_none() {
        if let (Some(hd), Some(nkv)) = (canonical.head_dim, canonical.num_key_value_heads) {
            canonical.kv_cache_block_size = Some(hd.max(nkv));
        }
    }

    // rope_scale: if not explicitly set, try rope_scaling.runtime_factor()
    if canonical.rope_scale.is_none() {
        if let Some(ref scaling) = canonical.rope_scaling {
            canonical.rope_scale = scaling.runtime_factor();
        }
    }

    // multimodal_token_ids: compute from individual token ID fields + vision/audio presence
    if canonical.multimodal_token_ids.is_none() {
        let img = canonical.image_token_id;
        let aud = canonical.audio_token_id;
        let eoi = canonical.eoi_token_id;
        let eoa = canonical.eoa_token_id;
        canonical.multimodal_token_ids = match (img, aud) {
            (Some(image_token_id), Some(audio_token_id)) => {
                Some(crate::compat::multimodal::MultimodalTokenIds {
                    image_token_id,
                    audio_token_id,
                    eoi_token_id: eoi.unwrap_or(image_token_id + 2),
                    eoa_token_id: eoa.unwrap_or(audio_token_id + 2),
                })
            }
            _ => {
                if canonical.vision_config.is_some() || canonical.audio_config.is_some() {
                    Some(crate::compat::multimodal::MultimodalTokenIds::fallback_multimodal_token_ids())
                } else {
                    None
                }
            }
        };
    }

    Ok(())
}

// ── Helper functions for tests ──

/// Return all canonical field names from FIELD_DEFS (for test assertions).
pub fn all_canonical_names() -> Vec<&'static str> {
    FIELD_DEFS.iter().map(|d| d.canonical).collect()
}

/// Return the number of entries in FIELD_DEFS.
pub fn field_defs_count() -> usize {
    FIELD_DEFS.len()
}

/// Return the number of required fields in FIELD_DEFS.
pub fn required_field_count() -> usize {
    FIELD_DEFS.iter().filter(|d| d.required).count()
}
