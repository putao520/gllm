/// A quantized tensor stored as raw block bytes with its QuantType metadata.
/// These are not uploaded via `Backend::upload_weights()` — they stay as raw bytes
/// and are dispatched to quantized matmul kernels at inference time.
///
/// For AWQ4/GPTQ4 from HuggingFace safetensors, the loader repacks the separate
/// qweight/scales/qzeros tensors into the unified 72-byte interleaved block layout
/// (see `repack_awq_gptq_blocks`) so the JIT QuantGemm microkernel consumes a single
/// contiguous buffer per weight, matching all other quant formats.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub quant_type: QuantType,
    pub shape: Vec<usize>,
    pub ggml_dtype: GgmlDType,
}

/// Block layout for AWQ4/GPTQ4 in JIT QuantGemm:
/// [scale(f16, 2B) | pad(2B) | zero(f16, 2B) | pad(2B) | qweight(64B)] = 72 bytes per 128-element group.
const AWQ_BLOCK_BYTES: usize = 72;
const AWQ_GROUP_SIZE: usize = 128;

/// Repack AWQ/GPTQ raw qweight + scales + zeros into interleaved 72-byte block format
/// expected by JIT QuantGemm tiled codegen.
///
/// Input shapes (HuggingFace AWQ/GPTQ safetensors):
/// - qweight: int32[rows][cols/8] (row-major, each int32 packs 8 4-bit values)
/// - scales:  f16[rows][groups]   (one f16 scale per group per row)
/// - qzeros:  int32[groups][rows_packed] (each int32 packs 8 4-bit zero-points)
/// - g_idx:   Option<i32[cols]>   (GPTQ per-column group index)
///
/// Output: 72-byte blocks per (row, group), total rows × groups × 72 bytes.
pub(crate) fn repack_awq_gptq_blocks(
    qweight: &[u8],
    scales: &[u8],
    zeros: &[u8],
    g_idx: Option<&[i32]>,
    _quant_type: QuantType,
    n: usize, // rows (output features)
    k: usize, // cols (input features)
) -> Vec<u8> {
    let num_groups = k / AWQ_GROUP_SIZE;
    let qweight_u32 = unsafe {
        std::slice::from_raw_parts(qweight.as_ptr() as *const u32, qweight.len() / 4)
    };
    let scales_f16 = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const half::f16, scales.len() / 2)
    };
    let zeros_u32 = unsafe {
        std::slice::from_raw_parts(zeros.as_ptr() as *const u32, zeros.len() / 4)
    };

    let total_blocks = n * num_groups;
    let mut packed = vec![0u8; total_blocks * AWQ_BLOCK_BYTES];
    let qw_cols_packed = k / 8; // int32 count per row of qweight
    let z_rows_packed = n.div_ceil(8); // packed int32 count per group row of qzeros

    for row in 0..n {
        for g in 0..num_groups {
            let block_offset = (row * num_groups + g) * AWQ_BLOCK_BYTES;

            // Scale: f16 at offset 0
            let scale_val = scales_f16[row * num_groups + g];
            packed[block_offset..block_offset + 2].copy_from_slice(&scale_val.to_le_bytes());

            // Zero: decode packed int4 → f16 at offset 4
            let z_word = zeros_u32[g * z_rows_packed + row / 8];
            let z_packed = ((z_word >> (4 * (row % 8))) & 0xF) as u8;
            let z_val = (z_packed + 1) as f32;
            let z_f16 = half::f16::from_f32(z_val);
            packed[block_offset + 4..block_offset + 6].copy_from_slice(&z_f16.to_le_bytes());

            // Qweight: 64 bytes at offset 8 (128 elements × 4 bits = 64 bytes = 16 u32s)
            let qw_src_start = row * qw_cols_packed + g * 16;
            if g_idx.is_none() {
                // AWQ: row-major, copy 16 u32s directly
                for i in 0..16 {
                    let src_val = qweight_u32[qw_src_start + i];
                    let dst_off = block_offset + 8 + i * 4;
                    packed[dst_off..dst_off + 4].copy_from_slice(&src_val.to_le_bytes());
                }
            } else {
                // GPTQ: apply g_idx column reordering
                let g_idx_slice = g_idx.unwrap();
                let mut qw_block = [0u32; 16];
                for col_pack in 0..16 {
                    let src_word = qweight_u32[qw_src_start + col_pack];
                    // De-interleave: extract 8 4-bit values and reassign by g_idx
                    for bit_pos in 0..8 {
                        let orig_col = g * AWQ_GROUP_SIZE + col_pack * 8 + bit_pos;
                        let nibble = ((src_word >> (4 * bit_pos)) & 0xF) as u8;
                        let target_col = g_idx_slice[orig_col] as usize * AWQ_GROUP_SIZE
                            + (orig_col % AWQ_GROUP_SIZE);
                        let target_pack = (target_col % AWQ_GROUP_SIZE) / 8;
                        let target_bit = target_col % 8;
                        qw_block[target_pack] |= (nibble as u32) << (4 * target_bit);
                    }
                }
                for i in 0..16 {
                    let dst_off = block_offset + 8 + i * 4;
                    packed[dst_off..dst_off + 4].copy_from_slice(&qw_block[i].to_le_bytes());
                }
            }
        }
    }
    packed
}

/// A native float tensor stored as raw bytes in its original dtype (BF16/F16).
/// Not converted to F32 — preserves original precision, saves 2× memory vs F32 expansion.
/// Bypasses Backend upload path; the raw bytes are consumed directly by weight packing.
#[derive(Debug, Clone)]
pub struct RawFloatTensor {
    pub data: Vec<u8>,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    HuggingFace,
    ModelScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChecksumPolicy {
    #[default]
    Ignore,
    Verify,
    Default,
}

#[derive(Debug, Clone)]
pub struct LoaderConfig {
    pub cache_dir: PathBuf,
    pub source: ModelSource,
    pub hf_token_path: Option<PathBuf>,
    pub enable_fallback: bool,
    pub checksum_policy: ChecksumPolicy,
    /// GGUF file name substring filter. When set, only GGUF files whose name
    /// contains this substring are considered for download.
    pub gguf_file_filter: Option<String>,
    /// #6: Configurable tensor skip strategy.
    pub tensor_skip_config: TensorSkipConfig,
    /// #5: Runtime-extensible suffix patterns for `match_tensor_role()`.
    pub extra_suffix_patterns: Vec<(Vec<String>, TensorRole, bool)>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        let cache_dir = dirs::home_dir()
            .map(|h| h.join(".gllm").join("models"))
            .unwrap_or_else(|| PathBuf::from(".gllm/models")); // LEGAL: 无 HOME 环境变量时使用当前目录
        Self {
            cache_dir,
            source: ModelSource::HuggingFace,
            hf_token_path: None,
            enable_fallback: true,
            checksum_policy: ChecksumPolicy::Ignore,
            gguf_file_filter: None,
            tensor_skip_config: TensorSkipConfig::default(),
            extra_suffix_patterns: Vec::new(),
        }
    }
}

impl LoaderConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(dir) = std::env::var("GLLM_CACHE_DIR") {
            if !dir.is_empty() {
                config.cache_dir = PathBuf::from(dir);
            }
        }
        config
    }
}

#[derive(Debug)]
pub struct CacheLayout {
    root: PathBuf,
}

impl CacheLayout {
    pub fn new(root: PathBuf) -> Result<Self> {
        Ok(Self { root })
    }

    pub fn ensure(&self) -> Result<()> {
        if !self.root.exists() {
            std::fs::create_dir_all(&self.root)?;
        }
        Ok(())
    }

    pub fn hf_cache_dir(&self) -> PathBuf {
        self.root.join("huggingface")
    }

    pub fn modelscope_cache_dir(&self) -> PathBuf {
        self.root.join("modelscope")
    }
}

pub fn is_recoverable_error(err: &LoaderError) -> bool {
    matches!(err, LoaderError::Network(_) | LoaderError::Io(_) | LoaderError::HfHub(_))
}

pub fn fallback_source(source: ModelSource) -> ModelSource {
    match source {
        ModelSource::HuggingFace => ModelSource::ModelScope,
        ModelSource::ModelScope => ModelSource::HuggingFace,
    }
}

/// #16: GGUF metadata key → config.json key mapping.
/// Used to normalize GGUF-specific keys to the same namespace as config.json fields,
/// so downstream code (model_config derivation) can use a single set of field names.
pub const GGUF_CONFIG_KEY_MAP: &[(&str, &str)] = &[
    ("n_embd", "hidden_size"),
    ("n_head", "num_attention_heads"),
    ("n_kv_head", "num_key_value_heads"),
    ("n_layer", "num_hidden_layers"),
    ("n_ff", "intermediate_size"),
    ("n_embd_head_k", "head_dim"),
    ("context_length", "max_position_embeddings"),
    ("n_ctx", "max_position_embeddings"),
    ("rope_freq_base", "rope_theta"),
    ("layer_norm_eps", "rms_norm_eps"),
    ("layer_norm_epsilon", "layer_norm_eps"),
    ("attention_layer_norm", "layer_norm_eps"),
    ("n_experts", "num_local_experts"),
    ("n_experts_per_tok", "num_experts_per_topk"),
    ("intermediate_size", "intermediate_size"), // 1:1 mapping for clarity
    ("vocab_size", "vocab_size"),
];

/// #16: Normalize a GGUF metadata key to its config.json equivalent.
/// Returns the config.json key if a mapping exists, otherwise returns the original key.
pub fn normalize_gguf_key(key: &str) -> &str {
    for (gguf_key, config_key) in GGUF_CONFIG_KEY_MAP {
        if *gguf_key == key {
            return config_key;
        }
    }
    key
}

// --- Tensor Role & Provider Logic ---

/// Suffix pattern table for 100% precise tensor role matching.
///
/// Each entry: `(suffix_segments, role, is_global)`
/// - `suffix_segments`: exact path segments after stripping layer prefix and terminal
/// - `is_global`: true if this tensor has no layer index
///
/// Order matters: longer suffixes MUST come before shorter ones to ensure
/// longest-match-first priority (e.g., "attn_q_norm" before "attn_q").
///
/// Derived from weight_names.rs mapping tables — single source of truth.
const SUFFIX_PATTERNS: &[(&[&str], TensorRole, bool)] = &[
    // ── Global tensors (is_global = true) ──
    (&["embed_tokens"],                     TensorRole::Embedding,         true),
    (&["word_embeddings"],                  TensorRole::Embedding,         true),
    (&["token_embd"],                       TensorRole::Embedding,         true),
    (&["lm_head"],                          TensorRole::OutputHead,        true),
    (&["output"],                           TensorRole::OutputHead,        true),
    (&["output_layer"],                     TensorRole::OutputHead,        true),
    (&["output_norm"],                      TensorRole::FinalNorm,         true),
    (&["norm"],                             TensorRole::FinalNorm,         true),
    (&["final_layernorm"],                  TensorRole::FinalNorm,         true),
    (&["post_layernorm"],                   TensorRole::FinalNorm,         true),
    (&["classifier", "dense"],              TensorRole::ClassifierDense,   true),
    (&["classifier", "out_proj"],           TensorRole::ClassifierOutProj, true),
    (&["classifier"],                       TensorRole::ClassifierOutProj, true),
    (&["score"],                            TensorRole::ClassifierOutProj, true),
    (&["vision_tower", "patch_embed", "proj"], TensorRole::PatchEmbed,    true),
    (&["patch_embed", "proj"],              TensorRole::PatchEmbed,        true),
    (&["position_embedding"],               TensorRole::PositionEmbedding, true),
    (&["position_embeddings"],              TensorRole::PositionEmbedding, true),
    (&["embeddings", "position_embedding"], TensorRole::PositionEmbedding, true),
    (&["embeddings", "position_embeddings"],TensorRole::PositionEmbedding, true),
    (&["token_type_embeddings"],            TensorRole::TokenTypeEmbedding, true),
    (&["embeddings", "token_type_embeddings"], TensorRole::TokenTypeEmbedding, true),
    (&["embeddings", "layernorm"],          TensorRole::EmbedNorm,         true),
    (&["embeddings", "layer_norm"],         TensorRole::EmbedNorm,         true),
    (&["rope"],                             TensorRole::Rope,              true),

    // PLE / AltUp global tensors (Gemma 4 E2B/E4B GGUF names)
    (&["per_layer_token_embd"],             TensorRole::PerLayerEmbedding, true),
    (&["per_layer_model_proj"],             TensorRole::PerLayerModelProj, true),
    (&["per_layer_proj_norm"],              TensorRole::PerLayerProjNorm,  true),
    (&["output_scale"],                     TensorRole::OutputScale,       true),

    // ── Per-layer tensors (is_global = false) ──
    // Sorted longest-first for unambiguous matching.

    // Attention norms (must come before q_proj/k_proj to win longest match)
    (&["self_attn", "q_norm"],              TensorRole::AttentionQNorm,    false),
    (&["self_attn", "k_norm"],              TensorRole::AttentionKNorm,    false),
    (&["attn_q_norm"],                      TensorRole::AttentionQNorm,    false),
    (&["attn_k_norm"],                      TensorRole::AttentionKNorm,    false),
    (&["self_attn", "sinks"],               TensorRole::AttentionSinks,    false),
    (&["attn_sinks"],                       TensorRole::AttentionSinks,    false),

    // Attention projections
    (&["self_attn", "q_proj"],              TensorRole::AttentionQuery,    false),
    (&["self_attn", "k_proj"],              TensorRole::AttentionKey,      false),
    (&["self_attn", "v_proj"],              TensorRole::AttentionValue,    false),
    (&["self_attn", "o_proj"],              TensorRole::AttentionOutput,   false),
    (&["self_attn", "out_proj"],            TensorRole::AttentionOutput,   false),
    (&["attn_q"],                           TensorRole::AttentionQuery,    false),
    (&["attn_k"],                           TensorRole::AttentionKey,      false),
    (&["attn_v"],                           TensorRole::AttentionValue,    false),
    (&["attn_output"],                      TensorRole::AttentionOutput,   false),
    (&["wq"],                               TensorRole::AttentionQuery,    false),
    (&["wk"],                               TensorRole::AttentionKey,      false),
    (&["wv"],                               TensorRole::AttentionValue,    false),
    (&["wo"],                               TensorRole::AttentionOutput,   false),

    // BERT attention (3-segment paths)
    (&["attention", "self", "query"],       TensorRole::AttentionQuery,    false),
    (&["attention", "self", "key"],         TensorRole::AttentionKey,      false),
    (&["attention", "self", "value"],       TensorRole::AttentionValue,    false),
    (&["attention", "output", "dense"],     TensorRole::AttentionOutput,   false),
    (&["self_attention", "query_key_value"], TensorRole::AttentionQuery,   false),
    (&["self_attn", "qkv_proj"],             TensorRole::AttentionFusedQkv, false),

    // MLA (Multi-head Latent Attention) — DeepSeek V3/R1, Kimi-K2
    (&["self_attn", "q_a_proj"],           TensorRole::MlaQCompress,      false),
    (&["self_attn", "q_b_proj"],           TensorRole::MlaQExpand,        false),
    (&["self_attn", "kv_b_proj"],          TensorRole::MlaKvCompress,     false),
    (&["self_attn", "k_b_proj"],           TensorRole::MlaKeyAbsorb,      false),
    (&["self_attn", "v_b_proj"],           TensorRole::MlaValueAbsorb,    false),
    (&["self_attn", "k_pe_proj"],          TensorRole::MlaRopeKey,        false),

    // Layer norms (longest first)
    (&["attention", "output", "layernorm"], TensorRole::InputNorm,         false),
    (&["output", "layernorm"],              TensorRole::PostAttnNorm,      false),
    (&["input_layernorm"],                  TensorRole::InputNorm,         false),
    (&["post_attention_layernorm"],         TensorRole::PostAttnNorm,      false),
    (&["pre_feedforward_layernorm"],        TensorRole::InputNorm,         false),
    (&["post_feedforward_layernorm"],       TensorRole::PostAttnNorm,      false),
    (&["attn_norm"],                        TensorRole::InputNorm,         false),
    (&["ffn_norm"],                         TensorRole::PostAttnNorm,      false),
    // Gemma 4 sandwich norms (GGUF names)
    (&["post_attention_norm"],              TensorRole::PostAttentionSandwichNorm, false),
    (&["post_ffw_norm"],                    TensorRole::PostFfwSandwichNorm,       false),
    (&["post_norm"],                        TensorRole::PostLayerNorm,             false),
    (&["layer_norm1"],                      TensorRole::InputNorm,         false),
    (&["layer_norm2"],                      TensorRole::PostAttnNorm,      false),
    (&["ln_1"],                             TensorRole::InputNorm,         false),
    (&["ln_2"],                             TensorRole::PostAttnNorm,      false),

    // FFN
    (&["mlp", "gate_up_proj"],              TensorRole::FfnGate,           false),
    (&["mlp", "gate_proj"],                 TensorRole::FfnGate,           false),
    (&["mlp", "up_proj"],                   TensorRole::FfnUp,             false),
    (&["mlp", "down_proj"],                 TensorRole::FfnDown,           false),
    (&["mlp", "fc1"],                       TensorRole::FfnUp,             false),
    (&["mlp", "fc2"],                       TensorRole::FfnDown,           false),
    (&["ffn_gate"],                         TensorRole::FfnGate,           false),
    (&["ffn_up"],                           TensorRole::FfnUp,             false),
    (&["ffn_down"],                         TensorRole::FfnDown,           false),
    (&["intermediate", "dense"],            TensorRole::FfnUp,             false),
    (&["output", "dense"],                  TensorRole::FfnDown,           false),
    (&["w1"],                               TensorRole::FfnGate,           false),
    (&["w2"],                               TensorRole::FfnDown,           false),
    (&["w3"],                               TensorRole::FfnUp,             false),

    // MoE (BCE-036: 对齐 transformers/vLLM/llama.cpp 全量命名变体)
    (&["mlp", "gate"],                      TensorRole::MoEGate,           false),
    (&["mlp", "router"],                    TensorRole::MoEGate,           false),
    (&["block_sparse_moe", "gate"],         TensorRole::MoEGate,           false), // BCE-036: Mixtral HF 原生
    (&["block_sparse_moe", "router"],       TensorRole::MoEGate,           false), // BCE-036: Mixtral fork 变体
    (&["ffn_gate_inp"],                     TensorRole::MoEGate,           false),
    // MoE shared expert (BCE-036: 多种命名变体)
    (&["mlp", "shared_experts", "gate_proj"], TensorRole::MoESharedExpert, false),
    (&["mlp", "shared_experts", "up_proj"], TensorRole::MoESharedExpert,   false),
    (&["mlp", "shared_experts", "down_proj"], TensorRole::MoESharedExpert, false),
    // BCE-036: Qwen/DeepSeek GGUF 单数形式 + w1/w2/w3
    (&["mlp", "shared_expert", "w1"],       TensorRole::MoESharedExpert,   false),
    (&["mlp", "shared_expert", "w2"],       TensorRole::MoESharedExpert,   false),
    (&["mlp", "shared_expert", "w3"],       TensorRole::MoESharedExpert,   false),

    // Audio/Vision special
    (&["conv_module", "depthwise_conv"],    TensorRole::DepthwiseConv,     false),

    // PLE / AltUp (Gemma 4 E2B/E4B GGUF names)
    (&["inp_gate"],                         TensorRole::PerLayerGate,      false),
    (&["proj"],                             TensorRole::PerLayerProj,      false),
    (&["layer_output_scale"],               TensorRole::LayerOutputScale,  false),
];

/// #5: Global registry for runtime-extensible suffix patterns.
/// Set once at model load time via `set_extra_suffix_patterns()`.
/// `match_tensor_role()` automatically includes these after the static `SUFFIX_PATTERNS`.
static EXTRA_SUFFIX_PATTERNS: std::sync::OnceLock<Vec<(Vec<String>, TensorRole, bool)>> =
    std::sync::OnceLock::new();

/// Set the global extra suffix patterns. Can only be called once.
/// Returns `Err(patterns)` if already set.
pub fn set_extra_suffix_patterns(
    patterns: Vec<(Vec<String>, TensorRole, bool)>,
) -> std::result::Result<(), Vec<(Vec<String>, TensorRole, bool)>> {
    EXTRA_SUFFIX_PATTERNS.set(patterns).map_err(|e| e.clone())
}

fn get_extra_suffix_patterns() -> &'static [(Vec<String>, TensorRole, bool)] {
    EXTRA_SUFFIX_PATTERNS
        .get()
        .map(|v| v.as_slice())
        .unwrap_or(&[])
}

/// Matches a tensor name to a role and optional layer index.
///
/// 100% precise: uses segment-sequence exact matching (not `contains()` heuristics).
/// Longest suffix matches first to disambiguate (e.g. `attn_q_norm` before `attn_q`).
/// Unrecognized names return `None` — no guessing.
///
/// Delegates to `match_tensor_role_ext` with the globally registered extra patterns.
pub fn match_tensor_role(name: &str) -> Option<(TensorRole, Option<usize>)> {
    match_tensor_role_ext(name, get_extra_suffix_patterns())
}

/// Extended version of `match_tensor_role` that also checks extra patterns.
/// Extra patterns are checked after the static `SUFFIX_PATTERNS` table.
pub fn match_tensor_role_ext(
    name: &str,
    extra_patterns: &[(Vec<String>, TensorRole, bool)],
) -> Option<(TensorRole, Option<usize>)> {
    // First try the static table
    if let Some(result) = match_tensor_role_static(name) {
        return Some(result);
    }

    // Then try extra patterns
    let lower = name.to_ascii_lowercase();
    let segments: Vec<&str> = lower.split('.').collect();

    let mut layer_idx = None;
    let mut layer_end = 0;
    for (i, seg) in segments.iter().enumerate() {
        if let Ok(idx) = seg.parse::<usize>() {
            if i > 0 {
                let prev = segments[i - 1];
                if matches!(prev, "layers" | "blk" | "blocks" | "h" | "layer" | "block" | "encoder")
                {
                    layer_idx = Some(idx);
                    layer_end = i + 1;
                    break;
                }
            }
        }
    }

    let content_segs = if layer_end > 0 {
        &segments[layer_end..]
    } else {
        &segments[..]
    };

    let content_segs = if content_segs
        .last()
        .is_some_and(|s| matches!(*s, "weight" | "bias" | "scales" | "blocks"))
    {
        &content_segs[..content_segs.len() - 1]
    } else {
        content_segs
    };

    // BCE-036: MoE 上下文守卫（同 match_tensor_role_static）
    let is_moe_context = content_segs.iter().any(|s| {
        matches!(*s, "experts" | "shared_expert" | "shared_experts" | "block_sparse_moe")
    });

    for (suffix_segs, role, is_global) in extra_patterns {
        if *is_global != layer_idx.is_none() && !*is_global {
            continue;
        }
        if suffix_segs.len() > content_segs.len() {
            continue;
        }
        // BCE-036: MoE 上下文下跳过 dense FFN 单 segment w1/w2/w3 模式
        if is_moe_context
            && suffix_segs.len() == 1
            && matches!(suffix_segs[0].as_str(), "w1" | "w2" | "w3")
        {
            continue;
        }
        let start = content_segs.len() - suffix_segs.len();
        let suffix_strs: Vec<&str> = suffix_segs.iter().map(|s| s.as_str()).collect();
        if content_segs[start..] == suffix_strs[..] {
            return Some((*role, layer_idx));
        }
    }

    None
}

/// Static-table-only matching (core logic, used by `match_tensor_role_ext`).
fn match_tensor_role_static(name: &str) -> Option<(TensorRole, Option<usize>)> {
    let lower = name.to_ascii_lowercase();

    // Skip bias tensors
    if lower.ends_with(".bias") || lower.ends_with("_bias") {
        return None;
    }

    let segments: Vec<&str> = lower.split('.').collect();

    // Extract layer index: scan for numeric segment preceded by a layer keyword.
    let mut layer_idx = None;
    let mut layer_end = 0;
    for (i, seg) in segments.iter().enumerate() {
        if let Ok(idx) = seg.parse::<usize>() {
            if i > 0 {
                let prev = segments[i - 1];
                if matches!(prev,
                    "layers" | "blk" | "blocks" | "h" | "layer" | "block" | "encoder"
                ) {
                    layer_idx = Some(idx);
                    layer_end = i + 1;
                    break;
                }
            }
        }
    }

    // Content segments: after layer prefix, before terminal
    let content_segs = if layer_end > 0 {
        &segments[layer_end..]
    } else {
        &segments[..]
    };

    // Strip terminal segment ("weight" / "bias" / "scales" / "blocks")
    let content_segs = if content_segs.last().is_some_and(|s|
        matches!(*s, "weight" | "bias" | "scales" | "blocks")
    ) {
        &content_segs[..content_segs.len() - 1]
    } else {
        content_segs
    };

    // BCE-036: MoE 上下文守卫。
    // 当 content_segs 含 "experts" 或 "shared_expert" segment 时，单 segment `w1/w2/w3`
    // 是 MoE expert/shared_expert 权重（Mixtral HF 原生命名），不是 dense FFN。
    // 必须跳过 dense FFN 的单 segment w1/w2/w3 模式，让多 segment MoE 模式（如
    // `mlp.shared_expert.w1`）或 name_map.rs Pass 1.5（experts.E.proj）处理。
    let is_moe_context = content_segs.iter().any(|s| {
        matches!(*s, "experts" | "shared_expert" | "shared_experts" | "block_sparse_moe")
    });

    // Match against suffix patterns (longest first, already sorted in table)
    for &(suffix_segs, role, is_global) in SUFFIX_PATTERNS {
        if is_global != layer_idx.is_none() && !is_global {
            continue;
        }

        if suffix_segs.len() > content_segs.len() {
            continue;
        }

        // BCE-036: MoE 上下文下跳过 dense FFN 单 segment w1/w2/w3 模式
        if is_moe_context
            && suffix_segs.len() == 1
            && matches!(suffix_segs[0], "w1" | "w2" | "w3")
        {
            continue;
        }

        let start = content_segs.len() - suffix_segs.len();
        if content_segs[start..] == *suffix_segs {
            return Some((role, layer_idx));
        }
    }

    // MTP (Multi-Token Prediction) special detection.
    // MTP projection weights have a variable depth index that doesn't fit
    // the static suffix pattern table. Patterns handled:
    //   model.mtp_head.{k}.weight   (DeepSeek V3 global)
    //   model.mtp.{k}.weight        (Qwen3 global)
    //   model.layers.{N}.mtp_proj.{k}.weight  (per-layer variant)
    // The depth index {k} is a numeric segment after "mtp_head", "mtp", or "mtp_proj".
    for seg in content_segs.iter() {
        if *seg == "mtp_head" || *seg == "mtp" || *seg == "mtp_proj" {
            return Some((TensorRole::MtpProjection, layer_idx));
        }
    }

    None
}

/// Build a reverse index from (TensorRole, Option<layer_idx>) to tensor name.
/// Uses the globally registered extra suffix patterns.
/// Also indexes bias tensors: for each weight tensor "foo.weight", checks if "foo.bias" exists.
#[allow(clippy::type_complexity)]
pub fn build_tensor_role_index<'a>(
    tensor_names: impl Iterator<Item = &'a str>,
) -> (
    HashMap<(TensorRole, Option<usize>), String>,
    HashMap<String, String>,
) {
    build_tensor_role_index_ext(tensor_names, get_extra_suffix_patterns())
}

/// Extended version of `build_tensor_role_index` with explicit extra patterns.
#[allow(clippy::type_complexity)]
pub fn build_tensor_role_index_ext<'a>(
    tensor_names: impl Iterator<Item = &'a str>,
    extra_patterns: &[(Vec<String>, TensorRole, bool)],
) -> (
    HashMap<(TensorRole, Option<usize>), String>,
    HashMap<String, String>,
) {
    let names: Vec<&str> = tensor_names.collect();
    let name_set: std::collections::HashSet<&str> = names.iter().copied().collect();

    let mut role_index: HashMap<(TensorRole, Option<usize>), String> = HashMap::new();
    let mut bias_index: HashMap<String, String> = HashMap::new();

    for &name in &names {
        if let Some((role, layer_idx)) = match_tensor_role_ext(name, extra_patterns) {
            role_index.insert((role, layer_idx), name.to_string());
        }

        // Index bias tensors: if name ends with .weight, check for .bias
        if name.ends_with(".weight") {
            let bias_name = format!("{}bias", &name[..name.len() - 6]);
            if name_set.contains(bias_name.as_str()) {
                bias_index.insert(name.to_string(), bias_name);
            }
        }
    }

    // Also check for standalone bias tensors (e.g. BERT's "embeddings.LayerNorm.bias")
    for &name in &names {
        let lower = name.to_ascii_lowercase();
        if (lower.ends_with(".bias") || lower.ends_with("_bias")) && !bias_index.values().any(|v| v == name) {
            // Try to find the corresponding weight
            let weight_name = if name.ends_with(".bias") {
                format!("{}weight", &name[..name.len() - 4])
            } else if let Some(stripped) = name.strip_suffix("_bias") {
                format!("{stripped}_weight")
            } else {
                continue;
            };
            if name_set.contains(weight_name.as_str()) {
                bias_index.insert(weight_name, name.to_string());
            }
        }
    }

    (role_index, bias_index)
}

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Cache error: {0}")]
    Cache(String),
    #[error("Missing weights file")]
    MissingWeights,
    #[error("Duplicate tensor: {0}")]
    DuplicateTensor(String),
    #[error("Missing tensor: {0}")]
    MissingTensor(String),
    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] ::safetensors::SafeTensorError),
    #[error("ONNX error: {0}")]
    Onnx(String),
    #[error("GGUF error: {0}")]
    Gguf(String),
    #[error("GLLM error: {0}")]
    Gllm(String),
    #[error("HfHub error: {0}")]
    HfHub(String),
    #[error("Invalid quantization metadata: {0}")]
    InvalidQuantization(String),
    #[error("Architecture detection failed: {0}")]
    ArchDetection(String),
    #[error("Authentication error: {hint}")]
    AuthenticationError { hint: String },
    #[error("Backend error: {0}")]
    Backend(String),
    #[error("PyTorch error: {0}")]
    Pytorch(String),
    #[error("Unsupported weight extension: {0}")]
    UnsupportedWeightExtension(String),
    #[error("Format not found: {0:?}")]
    FormatNotFound(WeightFormat),
    #[error("Multiple weight formats found")]
    MultipleWeightFormats(Vec<WeightFormat>),
}

impl From<gguf::GgufError> for LoaderError {
    fn from(err: gguf::GgufError) -> Self {
        LoaderError::Gguf(err.to_string())
    }
}

impl From<gllm::GllmError> for LoaderError {
    fn from(err: gllm::GllmError) -> Self {
        LoaderError::Gllm(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, LoaderError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    SafeTensors,
    Gguf,
    Onnx,
    PyTorch,
    Gllm,
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

pub type TensorInfo = TensorMeta;

/// Abstract provider of tensor metadata and data.
/// Allows unified config derivation across GGUF, SafeTensors, and ONNX.
pub trait TensorProvider {
    fn tensor_info(&self, name: &str) -> Option<TensorMeta>;
    fn iter_tensors(&self) -> impl Iterator<Item = TensorMeta>;

    /// Loads tensor data.
    /// Returns Cow to support both zero-copy (SafeTensors mmap) and allocated (GGUF conversion) data.
    fn load_tensor_data(&self, name: &str) -> Result<Cow<'_, [u8]>>;

    /// Returns the original GGML dtype for a tensor (GGUF only).
    fn ggml_dtype(&self, _name: &str) -> Option<GgmlDType> {
        None
    }

    /// ARCH-WEIGHT-CANONICAL-LAYOUT: Returns an explicit per-tensor hint about
    /// whether the stored 2D shape is HF [out, in] (needs transpose to canonical
    /// [K, N]) or already canonical.
    ///
    /// `Some(true)` — tensor shape is HF [out, in], downstream must transpose.
    /// `Some(false)` — tensor shape is canonical [K, N].
    /// `None` — no per-tensor hint; caller falls back to format-level default.
    ///
    /// Typical implementations:
    ///   - SafeTensors / PyTorch: return None (caller uses format default = true).
    ///   - ONNX: return Some based on Gemm `transB` attribute or MatMul semantics.
    ///   - GGUF: return None (format default = false).
    fn weight_layout_hint(&self, _name: &str) -> Option<bool> {
        None
    }

    /// Load AWQ/GPTQ auxiliary data (scales, zeros, g_idx) for a tensor name.
    ///
    /// Returns `Some((scales_bytes, zeros_bytes, g_idx_opt, group_size))` when the
    /// tensor is part of an AWQ/GPTQ triplet detected by the safetensors scanner.
    /// Returns `None` for non-AWQ/GPTQ tensors or providers without AWQ/GPTQ support.
    fn awq_gptq_aux_data(&self, _name: &str) -> Option<(Cow<'_, [u8]>, Cow<'_, [u8]>, Option<Vec<i32>>, usize)> {
        None
    }
}

/// #6: Configurable tensor skip strategy.
#[derive(Debug, Clone)]
pub struct TensorSkipConfig {
    /// Skip multimodal tower weights (vision_tower, audio_tower, embed_vision, embed_audio).
    pub skip_multimodal_towers: bool,
    /// Skip PLE/AltUp weights (embed_tokens_per_layer, per_layer_embedding, etc.).
    pub skip_ple_altup: bool,
}

impl Default for TensorSkipConfig {
    fn default() -> Self {
        Self {
            skip_multimodal_towers: true,
            skip_ple_altup: true,
        }
    }
}

/// ARCH-TENSOR-FILTER: check if a tensor should be skipped during upload.
/// Uses `TensorSkipConfig` to control which categories are skipped.
fn should_skip_tensor(name: &str, config: &TensorSkipConfig) -> bool {
    let mut skip = false;
    if config.skip_multimodal_towers {
        skip = skip
            || name.contains("vision_tower")
            || name.contains("audio_tower")
            || name.contains("embed_vision")
            || name.contains("embed_audio");
    }
    if config.skip_ple_altup {
        skip = skip
            || name.contains("embed_tokens_per_layer")
            || name.contains("per_layer_embedding")
            || name.contains("per_layer_projection")
            || name.contains("post_mlp_projection");
    }
    skip
}

/// Tensor loading priority for back-to-front ordering.
///
/// Higher value = loaded first = priority access to fastest tier (DeviceLocal).
/// Global weights (embedding, lm_head) get highest priority.
/// Layer weights: last layer (N-1) first, layer 0 last.
fn tensor_load_priority(name: &str) -> u32 {
    // Global weights: highest priority
    if name.contains("embed_tokens") || name.contains("token_embd")
        || name.contains("word_embeddings")
    {
        return 1000;
    }
    if name.contains("lm_head") || name.contains("output.weight") {
        return 999;
    }
    if name.contains("model.norm") || name.contains("norm.weight") {
        return 998;
    }

    // Layer weights: back-to-front (last layer gets higher priority)
    if let Some(layer_idx) = extract_layer_index(name) {
        return 900 - (layer_idx as u32);
    }

    500
}

/// Extract layer index from tensor name patterns.
fn extract_layer_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if let Ok(idx) = part.parse::<usize>() {
            if i > 0
                && matches!(
                    parts[i - 1],
                    "layers" | "layer" | "blk" | "h" | "blocks" | "block"
                )
            {
                return Some(idx);
            }
        }
    }
    None
}
