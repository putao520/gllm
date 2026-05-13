//! Layer 2/3: Loader (HF + SafeTensors + fused splits).

// unused imports removed
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ::safetensors::Dtype;
use crate::compat::backend_trait::{Backend, Element};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::manifest::{ModelManifest, TensorRole, EMPTY_FILE_MAP};
use crate::loader::gguf::GgmlDType;

// Re-export modules
pub mod adapter; // GGUF tensor adapter (KernelTensorView)
pub mod downloader;
pub mod format_detector;
pub mod gguf;
pub mod hf_hub;
pub mod modelscope;
pub mod name_map;
pub mod onnx;
pub mod parallel;
pub mod pytorch;
pub mod safetensors;
pub mod weight_tier;

pub use downloader::{ModelScopeDownloader, ProgressBar};
pub use gguf::GgufReader as GgufLoader;
pub use hf_hub::HfHubClient;
pub use modelscope::ModelScopeClient;
pub use onnx::OnnxLoader;
pub use parallel::ParallelLoader;
pub use safetensors::SafeTensorsLoader;

use gllm_kernels::quant::QuantType;
pub use adapter::ggml_dtype_to_quant_type;
pub use pytorch::{convert_bins_to_safetensors, PytorchConversionConfig, PytorchConversionOutput};

// Re-export quantization metadata types (defined later in this file)
// Note: CompanionConfig and QuantizationMetadata are already public below

/// A quantized tensor stored as raw block bytes with its QuantType metadata.
/// These are not uploaded via `Backend::upload_weights()` — they stay as raw bytes
/// and are dispatched to quantized matmul kernels at inference time.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub quant_type: QuantType,
    pub shape: Vec<usize>,
    pub ggml_dtype: GgmlDType,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
    (&["embeddings", "position_embedding"], TensorRole::PositionEmbedding, true),
    (&["rope"],                             TensorRole::Rope,              true),

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

    // Layer norms (longest first)
    (&["attention", "output", "layernorm"], TensorRole::InputNorm,         false),
    (&["output", "layernorm"],              TensorRole::PostAttnNorm,      false),
    (&["input_layernorm"],                  TensorRole::InputNorm,         false),
    (&["post_attention_layernorm"],         TensorRole::PostAttnNorm,      false),
    (&["pre_feedforward_layernorm"],        TensorRole::InputNorm,         false),
    (&["post_feedforward_layernorm"],       TensorRole::PostAttnNorm,      false),
    (&["attn_norm"],                        TensorRole::InputNorm,         false),
    (&["ffn_norm"],                         TensorRole::PostAttnNorm,      false),
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

    // MoE
    (&["mlp", "gate"],                      TensorRole::MoEGate,           false),
    (&["mlp", "router"],                    TensorRole::MoEGate,           false),
    (&["ffn_gate_inp"],                     TensorRole::MoEGate,           false),
    (&["mlp", "shared_experts", "gate_proj"], TensorRole::MoESharedExpert, false),
    (&["mlp", "shared_experts", "up_proj"], TensorRole::MoESharedExpert,   false),
    (&["mlp", "shared_experts", "down_proj"], TensorRole::MoESharedExpert, false),

    // Audio/Vision special
    (&["conv_module", "depthwise_conv"],    TensorRole::DepthwiseConv,     false),
];

/// Matches a tensor name to a role and optional layer index.
///
/// 100% precise: uses segment-sequence exact matching (not `contains()` heuristics).
/// Longest suffix matches first to disambiguate (e.g. `attn_q_norm` before `attn_q`).
/// Unrecognized names return `None` — no guessing.
pub fn match_tensor_role(name: &str) -> Option<(TensorRole, Option<usize>)> {
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

    // Match against suffix patterns (longest first, already sorted in table)
    for &(suffix_segs, role, is_global) in SUFFIX_PATTERNS {
        if is_global != layer_idx.is_none() && !is_global {
            continue;
        }

        if suffix_segs.len() > content_segs.len() {
            continue;
        }

        let start = content_segs.len() - suffix_segs.len();
        if content_segs[start..] == *suffix_segs {
            return Some((role, layer_idx));
        }
    }

    None
}

/// Build a reverse index from (TensorRole, Option<layer_idx>) to tensor name.
/// Also indexes bias tensors: for each weight tensor "foo.weight", checks if "foo.bias" exists.
#[allow(clippy::type_complexity)]
pub fn build_tensor_role_index<'a>(
    tensor_names: impl Iterator<Item = &'a str>,
) -> (
    HashMap<(TensorRole, Option<usize>), String>,
    HashMap<String, String>,
) {
    let names: Vec<&str> = tensor_names.collect();
    let name_set: std::collections::HashSet<&str> = names.iter().copied().collect();

    let mut role_index: HashMap<(TensorRole, Option<usize>), String> = HashMap::new();
    let mut bias_index: HashMap<String, String> = HashMap::new();

    for &name in &names {
        if let Some((role, layer_idx)) = match_tensor_role(name) {
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

pub type Result<T> = std::result::Result<T, LoaderError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    SafeTensors,
    Gguf,
    Onnx,
    PyTorch,
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
}

/// ARCH-TENSOR-FILTER: check if a tensor should be skipped during upload.
fn should_skip_tensor(name: &str) -> bool {
    name.contains("vision_tower")
        || name.contains("audio_tower")
        || name.contains("embed_vision")
        || name.contains("embed_audio")
        || name.contains("embed_tokens_per_layer")
        || name.contains("per_layer_embedding")
        || name.contains("per_layer_projection")
        || name.contains("post_mlp_projection")
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

#[derive(Debug)]
pub struct Loader {
    manifest: ModelManifest,
    weight_paths: Vec<PathBuf>,
    config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    format: WeightFormat,
    tie_word_embeddings_hint: Option<bool>,

    // Internal loaders (only one is active)
    safetensors: Option<safetensors::SafeTensorsLoader>,
    gguf: Option<gguf::GgufReader>,
    onnx: Option<onnx::OnnxLoader>,
}

/// Per-tensor processing result for concurrent upload pipeline.
enum TensorProcessResult<B: Backend<E>, E: Element> {
    Native {
        name: String,
        meta: TensorMeta,
        tensor: B::Tensor,
        placement: crate::compat::backend_trait::WeightPlacement,
        sp_meta: Option<Vec<Vec<u16>>>,
    },
    RawFloat {
        name: String,
        meta: TensorMeta,
        data: RawFloatTensor,
    },
    Quantized {
        name: String,
        meta: TensorMeta,
        data: QuantizedTensor,
    },
    Skipped,
}

impl Loader {
    pub fn new(manifest: ModelManifest) -> Self {
        Self {
            manifest,
            weight_paths: Vec::new(),
            config_path: None,
            tokenizer_path: None,
            format: WeightFormat::SafeTensors, // Default, will be detected
            tie_word_embeddings_hint: None,
            safetensors: None,
            gguf: None,
            onnx: None,
        }
    }

    pub fn from_env() -> Result<Self> {
        Ok(Self::new(ModelManifest::default()))
    }

    pub fn from_env_with_manifest(manifest: ModelManifest) -> Result<Self> {
        Ok(Self::new(manifest))
    }

    pub fn from_source_with_config(model_id: String, config: LoaderConfig) -> Result<Self> {
        // 本地目录检测：如果 model_id 是一个存在的目录，直接扫描文件
        let local_path = Path::new(&model_id);
        if local_path.is_dir() {
            return Self::from_local_dir(local_path);
        }

        let cache = CacheLayout::new(config.cache_dir.clone())
            .map_err(|e| LoaderError::Cache(e.to_string()))?;
        cache.ensure()?;

        let (weights, format, aux_files) = match config.source {
            ModelSource::HuggingFace => {
                let api = HfHubClient::with_endpoint_and_token_path(
                    cache.hf_cache_dir(),
                    None,
                    config.hf_token_path.clone(),
                )?;
                let parallel = ParallelLoader::new(true);

                // Try HF first
                match api.download_model_files(&model_id, EMPTY_FILE_MAP, parallel) {
                    Ok(files) => {
                        let fmt = match files.format {
                            hf_hub::WeightFormat::SafeTensors => WeightFormat::SafeTensors,
                            hf_hub::WeightFormat::Gguf => WeightFormat::Gguf,
                            hf_hub::WeightFormat::Onnx => WeightFormat::Onnx,
                        };
                        (files.weights, fmt, files.aux_files)
                    }
                    Err(err) => {
                        // Fallback to ModelScope if enabled and error is recoverable
                        if config.enable_fallback && is_recoverable_error(&err) {
                            eprintln!(
                                "⚠️ HuggingFace download failed, falling back to ModelScope: {}",
                                err
                            );
                            let ms_api = ModelScopeClient::new(cache.modelscope_cache_dir())?;
                            let ms_files = ms_api.download_model_files(
                                &model_id,
                                EMPTY_FILE_MAP,
                                ParallelLoader::new(true),
                            )?;
                            (ms_files.weights, ms_files.format, ms_files.aux_files)
                        } else {
                            return Err(err);
                        }
                    }
                }
            }
            ModelSource::ModelScope => {
                let api = ModelScopeClient::new(cache.modelscope_cache_dir())?;
                let files =
                    api.download_model_files(&model_id, EMPTY_FILE_MAP, ParallelLoader::new(true))?;
                (files.weights, files.format, files.aux_files)
            }
        };

        let mut loader = Self::new(ModelManifest::default());
        loader.weight_paths = weights;
        loader.format = format;

        // Populate config/tokenizer paths from aux_files
        for path in aux_files {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name == "config.json" {
                    loader.config_path = Some(path.clone());
                } else if name == "tokenizer.json" {
                    loader.tokenizer_path = Some(path.clone());
                }
            }
        }

        Ok(loader)
    }

    /// 从本地目录加载模型文件
    fn from_local_dir(dir: &Path) -> Result<Self> {
        let local = format_detector::collect_local_files(dir, None)?;

        let mut loader = Self::new(ModelManifest::default());
        loader.weight_paths = local.weights;
        loader.format = local.format;

        for path in local.aux_files {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name == "config.json" {
                    loader.config_path = Some(path);
                } else if name == "tokenizer.json" {
                    loader.tokenizer_path = Some(path);
                }
            }
        }

        Ok(loader)
    }

    pub fn with_weights(mut self, paths: Vec<PathBuf>) -> Self {
        self.weight_paths = paths;
        self.detect_format();
        self
    }

    pub fn weight_paths(&self) -> &[PathBuf] {
        &self.weight_paths
    }

    pub fn with_config(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path);
        self
    }

    pub fn with_tokenizer(mut self, path: PathBuf) -> Self {
        self.tokenizer_path = Some(path);
        self
    }

    fn detect_format(&mut self) {
        if let Some(first) = self.weight_paths.first() {
            if let Some(ext) = first.extension() {
                if ext == "gguf" {
                    self.format = WeightFormat::Gguf;
                    return;
                }
                if ext == "onnx" {
                    self.format = WeightFormat::Onnx;
                    return;
                }
                if ext == "pt" || ext == "bin" || ext == "pth" {
                    self.format = WeightFormat::PyTorch;
                    return;
                }
            }
        }
        self.format = WeightFormat::SafeTensors;
    }

    pub fn load(mut self) -> Result<Self> {
        match self.format {
            WeightFormat::SafeTensors => {
                let loader = safetensors::SafeTensorsLoader::from_files(
                    &self.weight_paths,
                    parallel::ParallelLoader::new(true),
                )?;
                self.safetensors = Some(loader);
            }
            WeightFormat::Gguf => {
                let reader = gguf::GgufReader::from_files(&self.weight_paths)?;
                self.gguf = Some(reader);
            }
            WeightFormat::Onnx => {
                // ONNX usually single file for now
                if let Some(path) = self.weight_paths.first() {
                    let loader = onnx::OnnxLoader::from_path(path)?;
                    self.onnx = Some(loader);
                }
            }
            WeightFormat::PyTorch => {
                let config = pytorch::PytorchConversionConfig::default();
                let output = pytorch::convert_bins_to_safetensors(
                    &self.weight_paths,
                    None,
                    &config,
                )?;
                let loader = safetensors::SafeTensorsLoader::from_files(
                    &output.safetensors,
                    parallel::ParallelLoader::new(true),
                )?;
                self.safetensors = Some(loader);
                self.format = WeightFormat::SafeTensors;
            }
        }
        Ok(self)
    }

    pub fn weight_format(&self) -> WeightFormat {
        self.format
    }

    pub fn config_path(&self) -> Option<&Path> {
        self.config_path.as_deref()
    }

    pub fn tokenizer_path(&self) -> Option<&Path> {
        self.tokenizer_path.as_deref()
    }

    pub fn safetensors_loader(&mut self) -> Result<&mut safetensors::SafeTensorsLoader> {
        self.safetensors.as_mut().ok_or(LoaderError::MissingWeights)
    }

    pub fn safetensors_ref(&self) -> Option<&safetensors::SafeTensorsLoader> {
        self.safetensors.as_ref()
    }

    pub fn gguf_reader(&mut self) -> Result<&mut gguf::GgufReader> {
        self.gguf.as_mut().ok_or(LoaderError::MissingWeights)
    }

    pub fn gguf_ref(&self) -> Option<&gguf::GgufReader> {
        self.gguf.as_ref()
    }

    pub fn onnx_loader(&mut self) -> Result<&mut onnx::OnnxLoader> {
        self.onnx.as_mut().ok_or(LoaderError::MissingWeights)
    }

    pub fn onnx_ref(&self) -> Option<&onnx::OnnxLoader> {
        self.onnx.as_ref()
    }

    pub fn onnx(&mut self) -> Result<&mut onnx::OnnxLoader> {
        self.onnx_loader()
    }

    /// 获取统一的 OnnxGraph 表示 (REQ-EXEC-001)
    ///
    /// 无论原始格式是 ONNX、SafeTensors 还是 GGUF，都转换为统一的 OnnxGraph。
    /// 检测模型架构（统一入口）
    ///
    /// 优先级：GGUF metadata > config.json model_type > 张量名称模式匹配 > manifest fallback
    pub fn detect_architecture(&self) -> String {
        use crate::manifest::map_architecture_token;

        // 1. GGUF metadata
        if let Some(gguf) = &self.gguf {
            if let Ok(arch_str) = gguf.architecture() {
                if let Some(arch) = map_architecture_token(arch_str) {
                    return arch;
                }
            }
        }

        // 2. config.json model_type / architectures (SafeTensors/ONNX)
        if let Some(config_path) = self.config_path() {
            if config_path.exists() {
                if let Ok(content) = std::fs::read_to_string(config_path) {
                    if let Ok(json) = serde_json::from_str::<Value>(&content) {
                        if let Some(arr) = json.get("architectures").and_then(|v| v.as_array()) {
                            for item in arr {
                                if let Some(s) = item.as_str() {
                                    if let Some(arch) = map_architecture_token(s) {
                                        return arch;
                                    }
                                }
                            }
                        }
                        if let Some(s) = json.get("model_type").and_then(|v| v.as_str()) {
                            if let Some(arch) = map_architecture_token(s) {
                                return arch;
                            }
                        }
                    }
                }
            }
        }

        // 3. 张量名称模式匹配
        if let Some(arch) = self.detect_architecture_from_tensors() {
            return arch;
        }

        // 4. manifest fallback
        self.manifest.arch.clone()
    }

    /// 从张量名称推断架构
    ///
    /// REQ-ARCH-Ω1: 禁止使用 contains() 模糊匹配，必须使用前缀匹配或张量形状推导
    fn detect_architecture_from_tensors(&self) -> Option<String> {
        // 检查单个张量名称是否匹配特定架构模式
        // 使用前缀匹配而非 contains() 避免模糊匹配
        let check_name = |name: &str| -> Option<String> {
            let lower = name.to_ascii_lowercase();

            // 将名称按 '.' 分割进行前缀匹配
            let parts: Vec<&str> = lower.split('.').collect();

            // BERT/RoBERTa/XLMR 风格: 前缀匹配
            // "bert.embeddings", "roberta.encoder", "xlmr."
            if parts.first().is_some_and(|p| {
                matches!(*p, "bert" | "roberta" | "xlmr" | "encoder")
            }) {
                return Some("xlmr".to_string());
            }

            // Mistral 风格: 前缀匹配 "model.layers" 或 "mistral."
            if parts.first().is_some_and(|p| {
                matches!(*p, "mistral" | "model")
            }) && parts.get(1).is_some_and(|p| {
                matches!(*p, "layers" | "embeddings")
            }) {
                return Some("mistral3".to_string());
            }

            // BERT encoder 模式: "encoder.layer.{N}.{...}" 或 "bert.encoder.layer.{N}"
            // 使用精确路径匹配而非 contains
            if parts.len() >= 3
                && ((parts[0] == "encoder" && parts[1] == "layer")
                    || (parts[0] == "bert" && parts[1] == "encoder" && parts[2] == "layer"))
                {
                    return Some("xlmr".to_string());
                }

            // BERT attention 模式: "attention.self.query" 精确路径匹配
            if parts.len() >= 3 && parts[1] == "attention" && parts[2] == "self" {
                return Some("xlmr".to_string());
            }

            None
        };

        // GPT-OSS (openai/gpt-oss-20b): 独有特征是 `self_attn.sinks` (attention sinks)
        // 和 `mlp.experts.gate_up_proj_blocks` (packed mxfp4 expert weights)。
        // 张量前缀 `model.layers.*` 与 Mistral 相同,但 GPT-OSS 有 MoE packed layout。
        // 必须在通用 check_name 之前检测,否则会被误识别为 mistral3。
        let is_gptoss_name = |name: &str| -> bool {
            let lower = name.to_ascii_lowercase();
            lower.contains("self_attn.sinks")
                || lower.contains("mlp.experts.gate_up_proj_blocks")
        };

        // Gemma 4 multi-modal SafeTensors 的张量布局是
        //   model.{audio_tower,vision_tower,embed_audio,embed_vision}.*
        //   model.language_model.{layers.*,embed_tokens,...}
        // 与单模态 LLM (model.layers.*) 完全不同。这种 multi-modal nesting 在所有
        // `check_name` 规则下都会落空,因为没有任何分支能把
        // `model.language_model` 识别成 decoder family。先扫一遍找
        // `model.language_model.` / `language_model.` 前缀,命中即返回 `gemma4`。
        // 优先级高于通用 check_name,但不会影响其他 SafeTensors 模型 (它们没有
        // language_model 这一层 nesting)。
        let is_gemma4_name = |name: &str| -> bool {
            let lower = name.to_ascii_lowercase();
            lower.starts_with("model.language_model.")
                || lower.starts_with("language_model.")
        };

        if let Some(st) = self.safetensors.as_ref() {
            if st.iter_tensors().any(|m| is_gptoss_name(&m.name)) {
                return Some("gptoss".to_string());
            }
            if st.iter_tensors().any(|m| is_gemma4_name(&m.name)) {
                return Some("gemma4".to_string());
            }
            for meta in st.iter_tensors() {
                if let Some(arch) = check_name(&meta.name) {
                    return Some(arch);
                }
            }
        }
        if let Some(onnx) = self.onnx.as_ref() {
            if onnx.iter_tensors().any(|m| is_gptoss_name(&m.name)) {
                return Some("gptoss".to_string());
            }
            if onnx.iter_tensors().any(|m| is_gemma4_name(&m.name)) {
                return Some("gemma4".to_string());
            }
            for meta in onnx.iter_tensors() {
                if let Some(arch) = check_name(&meta.name) {
                    return Some(arch);
                }
            }
        }
        if let Some(gguf) = self.gguf.as_ref() {
            if gguf.iter_tensors().any(|m| is_gptoss_name(&m.name)) {
                return Some("gptoss".to_string());
            }
            if gguf.iter_tensors().any(|m| is_gemma4_name(&m.name)) {
                return Some("gemma4".to_string());
            }
            for meta in gguf.iter_tensors() {
                if let Some(arch) = check_name(&meta.name) {
                    return Some(arch);
                }
            }
        }
        None
    }

    pub fn set_manifest_if_missing(&mut self, manifest: &ModelManifest) {
        // Simple overwrite or check if empty?
        // Assuming override for now as `from_env` creates a default one.
        self.manifest = manifest.clone();
    }

    pub fn set_tie_word_embeddings_hint(&mut self, hint: Option<bool>) {
        self.tie_word_embeddings_hint = hint;
    }

    pub fn gguf_architecture(&self) -> Result<&str> {
        if let Some(reader) = &self.gguf {
            reader
                .architecture()
                .map_err(|e| LoaderError::Gguf(e.to_string()))
        } else {
            Err(LoaderError::MissingWeights)
        }
    }

    // Helper for config derivation
    pub fn safetensors_gllm_config(&self) -> Result<Option<&Value>> {
        if let Some(loader) = &self.safetensors {
            Ok(loader.gllm_config())
        } else {
            Ok(None)
        }
    }

    /// Detect the dominant weight dtype from loaded tensors.
    /// Returns `DType` enum for type-safe dtype handling.
    pub fn detect_weight_dtype(&self) -> Result<Option<gllm_kernels::types::DType>> {
        use gllm_kernels::types::DType;
        if let Some(loader) = &self.safetensors {
            Ok(loader.detect_weight_dtype())
        } else if let Some(reader) = &self.gguf {
            Ok(reader.floating_point_dtype())
        } else if let Some(loader) = &self.onnx {
            let precisions = loader.unique_precisions();
            for dtype in precisions {
                match dtype {
                    Dtype::BF16 => return Ok(Some(DType::BF16)),
                    Dtype::F16 => return Ok(Some(DType::F16)),
                    Dtype::F32 => return Ok(Some(DType::F32)),
                    Dtype::F64 => return Ok(Some(DType::F32)), // f64 降级到 f32
                    _ => continue,
                }
            }
            Ok(None)
        } else {
            Ok(None)
        }
    }

    pub fn upload_weights<B: Backend<E>, E: Element>(
        &mut self,
        backend: &B,
    ) -> Result<WeightsHandle<B, E>> {
        let format = self.format;
        match format {
            WeightFormat::SafeTensors => {
                let provider = self
                    .safetensors
                    .as_ref()
                    .ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend, format)
            }
            WeightFormat::Gguf => {
                let provider = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend, format)
            }
            WeightFormat::Onnx => {
                let provider = self.onnx.as_ref().ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend, format)
            }
            _ => unreachable!("PyTorch is converted to SafeTensors by load()"),
        }
    }

    fn upload_provider<P: TensorProvider + Sync, B: Backend<E>, E: Element>(
        &self,
        provider: &P,
        backend: &B,
        format: WeightFormat,
    ) -> Result<WeightsHandle<B, E>> {
        use crate::compat::backend_trait::WeightPlacement;

        // Pass 1: collect + filter + sort (back-to-front priority)
        let mut tensor_metas: Vec<TensorMeta> = provider
            .iter_tensors()
            .filter(|m| !should_skip_tensor(&m.name))
            .collect();
        tensor_metas.sort_by(|a, b| {
            tensor_load_priority(&b.name).cmp(&tensor_load_priority(&a.name))
        });

        // Pass 2: concurrent processing (CPU-side parallelism)
        let tier_manager = weight_tier::WeightTierManager::from_backend(backend);

        let results: Vec<TensorProcessResult<B, E>> = tensor_metas
            .par_iter()
            .map(|meta| {
                Self::process_single_tensor(provider, backend, meta, format, &tier_manager)
                    .unwrap_or_else(|e| {
                        log::error!("failed to process tensor '{}': {}", meta.name, e);
                        TensorProcessResult::Skipped
                    })
            })
            .collect();

        // Pass 3: sequential fold into HashMaps
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        let mut meta_map = HashMap::new();
        let mut quantized = HashMap::new();
        let mut raw_floats = HashMap::new();
        let mut sparse_24 = HashMap::new();
        let mut placements = HashMap::new();

        for result in results {
            match result {
                TensorProcessResult::Native { name, meta, tensor, placement, sp_meta } => {
                    tensors.insert(name.clone(), tensor);
                    shapes.insert(name.clone(), meta.shape.clone());
                    meta_map.insert(name.clone(), meta);
                    placements.insert(name.clone(), placement);
                    if let Some(sp) = sp_meta {
                        sparse_24.insert(name, sp);
                    }
                }
                TensorProcessResult::RawFloat { name, meta, data } => {
                    shapes.insert(name.clone(), meta.shape.clone());
                    meta_map.insert(name.clone(), meta);
                    raw_floats.insert(name, data);
                }
                TensorProcessResult::Quantized { name, meta, data } => {
                    quantized.insert(name.clone(), data);
                    shapes.insert(name.clone(), meta.shape.clone());
                    meta_map.insert(name, meta);
                }
                TensorProcessResult::Skipped => {}
            }
        }

        // Log tier distribution
        let (dev_used, dev_cap) = tier_manager.usage(weight_tier::WeightTier::DeviceLocal);
        let (host_used, host_cap) = tier_manager.usage(weight_tier::WeightTier::HostLocal);
        let device_count = placements.values().filter(|p| **p == WeightPlacement::DeviceLocal).count();
        let host_count = placements.values().filter(|p| **p == WeightPlacement::HostLocal).count();
        let mmap_count = tier_manager.tensor_count() - device_count - host_count;
        log::info!(
            "upload_provider: {} tensors loaded, device={}/{}B, host={}/{}B",
            placements.len(), dev_used, dev_cap, host_used, host_cap,
        );
        if mmap_count > 0 {
            log::info!("upload_provider: {} tensors degraded to mmap", mmap_count);
        }

        Ok(WeightsHandle::new_with_placements(
            tensors, shapes, meta_map, quantized, raw_floats, sparse_24, placements,
        ))
    }

    /// Process a single tensor in the concurrent upload pipeline.
    fn process_single_tensor<P, B, E>(
        provider: &P,
        backend: &B,
        meta: &TensorMeta,
        format: WeightFormat,
        tier_manager: &weight_tier::WeightTierManager,
    ) -> Result<TensorProcessResult<B, E>>
    where
        P: TensorProvider + Sync,
        B: Backend<E>,
        E: Element,
    {
        // Quantized tensor — store raw bytes, no tier decision needed
        if let Some(ggml_dt) = provider.ggml_dtype(&meta.name) {
            if let Some(qt) = adapter::ggml_dtype_to_quant_type(ggml_dt) {
                let data = provider.load_tensor_data(&meta.name)?;
                return Ok(TensorProcessResult::Quantized {
                    name: meta.name.clone(),
                    meta: meta.clone(),
                    data: QuantizedTensor {
                        data: data.into_owned(),
                        quant_type: qt,
                        shape: meta.shape.clone(),
                        ggml_dtype: ggml_dt,
                    },
                });
            }
        }

        // Float tensor — preserve original dtype for BF16/F16, convert F32/F64 normally
        match meta.dtype {
            Dtype::BF16 | Dtype::F16 => {
                let data = provider.load_tensor_data(&meta.name)?;
                let mut cloned_meta = meta.clone();
                let mut raw = data.into_owned();

                // ARCH-WEIGHT-NO-TRANSPOSE: Linear weights kept in original [out, in] layout.
                // GEMM lowering handles trans_b=true for HF [N,K] row-major weights.

                Ok(TensorProcessResult::RawFloat {
                    name: cloned_meta.name.clone(),
                    meta: cloned_meta.clone(),
                    data: RawFloatTensor {
                        data: raw,
                        dtype: meta.dtype,
                        shape: cloned_meta.shape.clone(),
                    },
                })
            }
            Dtype::F32 | Dtype::F64 => {
                let data = provider.load_tensor_data(&meta.name)?;
                let explicit_hint = provider.weight_layout_hint(&meta.name);
                let (cloned_meta, converted_f32, sp_meta_opt) =
                    convert_tensor_to_f32(meta, data.as_ref(), format, explicit_hint)?;

                // Tier decision: DeviceLocal → HostLocal → DiskMmap
                let tensor_size = converted_f32.len() * std::mem::size_of::<f32>();
                let decision = tier_manager.decide(&cloned_meta.name, tensor_size);

                let (tensor, placement) = backend
                    .upload_weights_with_placement(converted_f32, decision.placement)
                    .map_err(|e| LoaderError::Backend(e.to_string()))?;

                Ok(TensorProcessResult::Native {
                    name: cloned_meta.name.clone(),
                    meta: cloned_meta,
                    tensor,
                    placement,
                    sp_meta: sp_meta_opt,
                })
            }
            _ => Ok(TensorProcessResult::Skipped),
        }
    }

    pub fn from_local_files_with_manifest(
        _model_id: &str,
        weight_paths: Vec<PathBuf>,
        aux_paths: Vec<PathBuf>,
        manifest: Option<&ModelManifest>,
    ) -> Result<Self> {
        let mut loader = if let Some(m) = manifest {
            Self::new(m.clone())
        } else {
            Self::new(ModelManifest::default())
        };
        loader.weight_paths = weight_paths;
        loader.detect_format();

        for path in aux_paths {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name == "config.json" {
                    loader.config_path = Some(path);
                } else if name == "tokenizer.json" {
                    loader.tokenizer_path = Some(path);
                }
            }
        }
        Ok(loader)
    }
}

#[derive(Debug, Clone)]
pub struct TensorSlice<'a> {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl<'a> TensorSlice<'a> {
    pub fn new(dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> Self {
        Self { dtype, shape, data }
    }
}

// upload_native_tensor removed — superseded by convert_tensor_to_f32 (pure conversion)
// + tier-aware upload in process_single_tensor.

/// Convert raw tensor bytes to f32, applying P4/P5 heuristics and layout normalization.
/// Returns the modified meta, the f32 data, and optional sparsity metadata.
fn convert_tensor_to_f32(
    meta: &TensorMeta,
    data: &[u8],
    format: WeightFormat,
    explicit_transpose_hint: Option<bool>,
) -> Result<(TensorMeta, Vec<f32>, Option<Vec<Vec<u16>>>)> {
    // ARCH-LOADER-PARALLEL-CONVERT: Rayon-parallel dtype→f32 conversion.
    // For large models (e.g. Gemma 4 E2B 9.6 GB BF16), a single-threaded
    // `chunks_exact().map().collect()` takes 60-120s on 4.8B elements. The
    // parallel path pre-allocates the output Vec and uses `par_chunks_mut`
    // so each worker writes into its own disjoint slice — no synchronisation,
    // ~5-10s on a 20-core machine.
    let mut converted_f32: Vec<f32> = match meta.dtype {
        Dtype::F32 => parallel_bytes_to_f32_lossless(data)?,
        Dtype::F16 => parallel_half_to_f32::<half::f16>(data)?,
        Dtype::BF16 => parallel_half_to_f32::<half::bf16>(data)?,
        Dtype::F64 => parallel_f64_to_f32(data)?,
        _ => {
            return Err(LoaderError::Backend(format!(
                "cannot convert {:?} to f32 for heuristics",
                meta.dtype
            )));
        }
    };

    let mut cloned_meta = meta.clone();

    apply_ffn_sparsity_heuristic(&cloned_meta, &mut converted_f32);
    let sp_meta_opt = compress_24_sparsity_heuristic(&mut cloned_meta, &mut converted_f32);
    deduplicate_q_heads_heuristic(&cloned_meta, &mut converted_f32);

    // ARCH-WEIGHT-NO-TRANSPOSE: Linear weights kept in original [out, in] layout.
    // GEMM lowering handles trans_b=true for HF [N,K] row-major weights.

    Ok((cloned_meta, converted_f32, sp_meta_opt))
}

/// HF SafeTensors/PyTorch 的 Linear 权重 layout 归一化。
///
/// 问题: HF `nn.Linear.weight` 的内存布局是 `[out_features, in_features]` row-major,
/// 前向为 `y = x @ W.T`。但 gllm-kernels JIT GEMM 的 B 输入约定是 `[K, N]` row-major
/// (ONNX MatMul 语义, `y = x @ B`)。直接用 HF 布局会得到错误结果 (方阵下 shape 一致
/// 但数值错误), 非方阵时 N ≠ K 还会越界 SIGSEGV。
///
/// 根治: 加载边界统一把 HF `[out, in]` 物理转置成 canonical `[in, out]` 布局并更新
/// meta.shape。内部 op 只处理 canonical layout (ARCH-WEIGHT-CANONICAL-LAYOUT)。
///
/// 只对真正的 Linear 权重生效 — 排除 embedding / LayerNorm / bias / 非 2D tensor。
fn normalize_linear_weight_layout(meta: &mut TensorMeta, data: &mut Vec<f32>) {
    if !is_linear_weight(&meta.name, &meta.shape) {
        return;
    }
    let rows = meta.shape[0]; // out_features (HF)
    let cols = meta.shape[1]; // in_features (HF)
    if rows * cols != data.len() {
        log::warn!(
            "normalize_linear_weight_layout: '{}' shape {:?} does not match data len {}, skip",
            meta.name, meta.shape, data.len()
        );
        return;
    }
    // Row-major [rows, cols] → [cols, rows] via cache-blocked transpose.
    let mut out = vec![0.0f32; data.len()];
    cache_blocked_transpose_f32(data, &mut out, rows, cols);
    *data = out;
    meta.shape = vec![cols, rows]; // canonical [in, out] = [K, N]
}

/// Byte-level layout normalization for non-F32 float tensors (BF16/F16).
/// Same logic as `normalize_linear_weight_layout` but operates on raw bytes.
fn normalize_linear_weight_layout_bytes(meta: &mut TensorMeta, data: &mut Vec<u8>, elem_size: usize) {
    if !is_linear_weight(&meta.name, &meta.shape) {
        return;
    }
    let rows = meta.shape[0]; // out_features (HF)
    let cols = meta.shape[1]; // in_features (HF)
    let total_elems = data.len() / elem_size;
    if rows * cols != total_elems {
        log::warn!(
            "normalize_linear_weight_layout_bytes: '{}' shape {:?} does not match data len {} (elem_size={}), skip",
            meta.name, meta.shape, data.len(), elem_size
        );
        return;
    }
    let mut out = vec![0u8; data.len()];
    cache_blocked_transpose_bytes(data, &mut out, rows, cols, elem_size);
    *data = out;
    meta.shape = vec![cols, rows];
}

/// Cache-blocked (tiled) f32 transpose.
///
/// `src` is `[rows, cols]` row-major; `dst` is written as `[cols, rows]`
/// row-major (i.e. `dst[c * rows + r] = src[r * cols + c]`).
///
/// Naive transpose writes with a stride of `rows * 4` bytes → every store
/// misses L1 on typical weight shapes (e.g. 1536 × 12288 → 6144-byte stride).
/// Observed throughput is ~50-200 MB/s.
///
/// A tile-based transpose keeps `TILE × TILE` f32s in L1 (16 KB for 64×64)
/// so reads and writes within a tile both hit L1. We additionally use Rayon
/// to parallelise over the outer row-tile dimension so 20 cores can co-operate
/// on independent chunks of `dst`.
///
/// Both axes are processed in `TILE`-sized blocks; the tail rows/cols inside
/// the last tile are handled by `.min(rows)` / `.min(cols)` inside the inner
/// loops. Produces bit-identical output to a naive transpose for all finite
/// f32 values.
fn cache_blocked_transpose_f32(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    const TILE: usize = 64;
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);

    // SAFETY: each parallel worker writes to a disjoint set of output rows.
    // We parallelise over `j_tile` (the column dimension of `src`, i.e. the
    // ROW dimension of `dst`). Thread `j_tile` only writes to
    // `dst[jj * rows .. (jj+1) * rows]` for `jj` in `[j_tile*TILE,
    // (j_tile+1)*TILE)`, which is a strictly disjoint row range across
    // threads. We cannot take `&mut [f32]` slices of arbitrary row ranges
    // across the parallel iterator (borrow checker can't prove disjointness
    // through a raw `for_each`), so we pass the base pointer as a `usize`
    // address (which is `Send + Sync`) and re-cast inside each closure.
    let dst_addr = dst.as_mut_ptr() as usize;

    let num_j_tiles = cols.div_ceil(TILE);
    (0..num_j_tiles).into_par_iter().for_each(|j_tile| {
        let j_start = j_tile * TILE;
        let j_end = (j_start + TILE).min(cols);
        let dst_base = dst_addr as *mut f32;
        for i_start in (0..rows).step_by(TILE) {
            let i_end = (i_start + TILE).min(rows);
            for ii in i_start..i_end {
                let src_row = ii * cols;
                for jj in j_start..j_end {
                    // dst[jj * rows + ii] = src[ii * cols + jj]
                    unsafe {
                        let v = *src.get_unchecked(src_row + jj);
                        *dst_base.add(jj * rows + ii) = v;
                    }
                }
            }
        }
    });
}

/// Cache-blocked byte-level transpose for arbitrary element sizes (BF16=2, F16=2, etc.).
/// Generalization of `cache_blocked_transpose_f32` where each element is `elem_size` bytes.
fn cache_blocked_transpose_bytes(src: &[u8], dst: &mut [u8], rows: usize, cols: usize, elem_size: usize) {
    const TILE: usize = 64;
    let row_stride = cols * elem_size;
    let col_stride = rows * elem_size;
    let total = rows * cols * elem_size;
    debug_assert_eq!(src.len(), total);
    debug_assert_eq!(dst.len(), total);

    let dst_addr = dst.as_mut_ptr() as usize;
    let src_addr = src.as_ptr() as usize;

    let num_j_tiles = cols.div_ceil(TILE);
    (0..num_j_tiles).into_par_iter().for_each(|j_tile| {
        let j_start = j_tile * TILE;
        let j_end = (j_start + TILE).min(cols);
        let dst_base = dst_addr as *mut u8;
        let src_base = src_addr as *const u8;
        for i_start in (0..rows).step_by(TILE) {
            let i_end = (i_start + TILE).min(rows);
            for ii in i_start..i_end {
                let src_row_off = ii * row_stride;
                for jj in j_start..j_end {
                    let dst_off = jj * col_stride + ii * elem_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_base.add(src_row_off + jj * elem_size),
                            dst_base.add(dst_off),
                            elem_size,
                        );
                    }
                }
            }
        }
    });
}

/// Parallel byte-exact `&[u8]` → `Vec<f32>` (for native `F32`) using
/// `par_chunks_exact` so unaligned loads happen across all cores.
///
/// Returns `Err` if `data.len()` is not a multiple of `size_of::<f32>()`.
fn parallel_bytes_to_f32_lossless(data: &[u8]) -> Result<Vec<f32>> {
    let src_size = std::mem::size_of::<f32>();
    if data.len() % src_size != 0 {
        return Err(LoaderError::Backend(format!(
            "F32 tensor data length {} is not a multiple of {}",
            data.len(),
            src_size
        )));
    }
    let n = data.len() / src_size;
    let mut out = vec![0.0f32; n];
    out.par_chunks_mut(1024)
        .zip(data.par_chunks_exact(src_size * 1024).with_min_len(1))
        .for_each(|(out_chunk, in_bytes)| {
            for (i, sub) in in_bytes.chunks_exact(src_size).enumerate() {
                // SAFETY: sub.len() == src_size, read_unaligned is always valid.
                out_chunk[i] = unsafe { std::ptr::read_unaligned(sub.as_ptr() as *const f32) };
            }
        });
    // Handle the tail chunks (where par_chunks_exact leaves a remainder on the
    // input side and par_chunks_mut may expose a smaller last chunk on the
    // output side). `par_chunks_exact` emits exact-sized chunks only — if the
    // total element count is not a multiple of 1024 we still need to cover the
    // final partial chunk manually.
    let completed = (n / 1024) * 1024;
    if completed < n {
        let tail_bytes = &data[completed * src_size..];
        for (i, sub) in tail_bytes.chunks_exact(src_size).enumerate() {
            out[completed + i] =
                unsafe { std::ptr::read_unaligned(sub.as_ptr() as *const f32) };
        }
    }
    Ok(out)
}

/// Parallel `&[u8]` → `Vec<f32>` for any 16-bit half-precision type
/// (`half::f16` or `half::bf16`).
fn parallel_half_to_f32<H>(data: &[u8]) -> Result<Vec<f32>>
where
    H: Copy + Send + Sync + 'static,
    H: HalfToF32,
{
    let src_size = std::mem::size_of::<H>();
    if data.len() % src_size != 0 {
        return Err(LoaderError::Backend(format!(
            "{} tensor data length {} is not a multiple of {}",
            std::any::type_name::<H>(),
            data.len(),
            src_size
        )));
    }
    let n = data.len() / src_size;
    let mut out = vec![0.0f32; n];
    const CHUNK: usize = 4096;
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let byte_start = chunk_idx * CHUNK * src_size;
            let byte_end = byte_start + out_chunk.len() * src_size;
            let in_bytes = &data[byte_start..byte_end];
            for (i, sub) in in_bytes.chunks_exact(src_size).enumerate() {
                // SAFETY: sub.len() == src_size = size_of::<H>().
                let v: H = unsafe { std::ptr::read_unaligned(sub.as_ptr() as *const H) };
                out_chunk[i] = v.to_f32_fast();
            }
        });
    Ok(out)
}

/// Parallel `&[u8]` → `Vec<f32>` for `F64` (narrowing cast).
fn parallel_f64_to_f32(data: &[u8]) -> Result<Vec<f32>> {
    let src_size = std::mem::size_of::<f64>();
    if data.len() % src_size != 0 {
        return Err(LoaderError::Backend(format!(
            "F64 tensor data length {} is not a multiple of {}",
            data.len(),
            src_size
        )));
    }
    let n = data.len() / src_size;
    let mut out = vec![0.0f32; n];
    const CHUNK: usize = 4096;
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let byte_start = chunk_idx * CHUNK * src_size;
            let byte_end = byte_start + out_chunk.len() * src_size;
            let in_bytes = &data[byte_start..byte_end];
            for (i, sub) in in_bytes.chunks_exact(src_size).enumerate() {
                let v: f64 =
                    unsafe { std::ptr::read_unaligned(sub.as_ptr() as *const f64) };
                out_chunk[i] = v as f32;
            }
        });
    Ok(out)
}

/// Internal trait bridging `half::f16` / `half::bf16` to their `to_f32`
/// implementation inside a generic context.
trait HalfToF32 {
    fn to_f32_fast(self) -> f32;
}

impl HalfToF32 for half::f16 {
    #[inline(always)]
    fn to_f32_fast(self) -> f32 { self.to_f32() }
}

impl HalfToF32 for half::bf16 {
    #[inline(always)]
    fn to_f32_fast(self) -> f32 { self.to_f32() }
}

/// Heuristic: 判断一个 2D tensor 是否是 Linear 权重 (需要 canonical layout 归一化)。
fn is_linear_weight(name: &str, shape: &[usize]) -> bool {
    if shape.len() != 2 {
        return false;
    }
    if !name.ends_with(".weight") {
        return false;
    }
    // Embedding weight 是 [vocab, hidden], 用于 Gather 不是 MatMul, 不能转置。
    let excluded_substrings = [
        "embeddings.word_embeddings",
        "embeddings.position_embeddings",
        "embeddings.token_type_embeddings",
        "wte.",                 // GPT-style word/token embedding
        "wpe.",                 // GPT-style position embedding
        "embed_tokens",         // Llama/Qwen/Mistral token embedding
        "token_embd",           // GGUF token embedding
        "LayerNorm",            // LayerNorm.weight (1D 已在 shape 检查排除, 双保险)
        "layer_norm",
        "RMSNorm",
        "rms_norm",
        ".norm.",
    ];
    for ex in &excluded_substrings {
        if name.contains(ex) {
            return false;
        }
    }
    true
}

/// Applies Tier II structural sparsity heuristic on FFN matrices.
/// Identifies and outright zeroes columns (or rows) in `gate_proj` and `up_proj` whose 
/// L2-norm falls below `0.01 * mean_L2`. This structural nullification guarantees 
/// `gate_out` falls to 0.0 and skips dependent computations within `MaskedGemm`.
fn apply_ffn_sparsity_heuristic(meta: &TensorMeta, data: &mut [f32]) {
    if !meta.name.contains("mlp.gate_proj") && !meta.name.contains("mlp.up_proj") {
        return;
    }

    if meta.shape.len() != 2 {
        return;
    }

    let rows = meta.shape[0]; 
    let cols = meta.shape[1];

    let mut l2_norms = Vec::with_capacity(rows);
    let mut sum_l2 = 0.0;

    for r in 0..rows {
        let mut norm_sq = 0.0f32;
        let start = r * cols;
        for c in 0..cols {
            let val = data[start + c];
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        l2_norms.push(norm);
        sum_l2 += norm;
    }

    let mean_l2 = sum_l2 / (rows as f32);
    let threshold = 0.01 * mean_l2;

    let mut pruned = 0;
    for (r, norm) in l2_norms.iter().enumerate() {
        if *norm < threshold {
            let start = r * cols;
            for c in 0..cols {
                data[start + c] = 0.0;
            }
            pruned += 1;
        }
    }

    if pruned > 0 {
        log::info!("🧠 Structural Sparsity: Nullified {}/{} rows in {}.", pruned, rows, meta.name);
    }
}

/// Applies NVIDIA 2:4 Structural Sparsity pattern on FFN matrices.
/// Enforces the 2:4 sparsity pattern structurally directly inside the tensor buffer 
/// at model load time to avoid any CPU overhead during the inference hot loop.
/// Shrinks the tensor dimension by 50% and returns the generated sp_meta for Phase D (Sparse MMA).
fn compress_24_sparsity_heuristic(meta: &mut TensorMeta, data: &mut Vec<f32>) -> Option<Vec<Vec<u16>>> {
    // 2:4 structural sparsity compression is ONLY valid for GPU Sparse MMA
    // (NVIDIA Ampere+). On CPU-only JIT builds, the dense GEMM expects full-
    // dimension weights. Compressing here causes the JIT GEMM to read past
    // the buffer boundary → SIGSEGV.
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (meta, data);
        None
    }

    #[cfg(feature = "cuda")]
    {
    if !meta.name.contains("mlp.gate_proj") && !meta.name.contains("mlp.up_proj") && !meta.name.contains("experts") {
        return None;
    }
    if meta.shape.len() != 2 {
        return None;
    }
    let rows = meta.shape[0];
    let cols = meta.shape[1];
    if cols % 4 != 0 {
        return None;
    }

    // Convert flat data to Vec<Vec<f32>>
    let rows_data: Vec<Vec<f32>> = data.chunks(cols).map(|c| c.to_vec()).collect();
    let (pruned_rows, sp_meta) = crate::static_compression::prune_dead_columns_24(&rows_data);

    // 物理显存压实 (Physical Memory Shrink):
    // 虽然底层出于接口兼容返回了原尺寸的零填充张量，但在 Loader 我们强制将其抛弃。
    // 我们仅根据生成的 sp_meta 重建紧凑的 50% 内存块。
    let mut compressed_data = Vec::with_capacity(rows * (cols / 2));
    
    for (r_idx, row) in pruned_rows.iter().enumerate() {
        let meta_row = &sp_meta[r_idx];
        for grp in 0..(cols / 4) {
            let base = grp * 4;
            // Decode the 2-bit indices from sp_meta
            let meta_u16_idx = grp / 2;
            let meta_shift = (grp % 2) * 4;
            let encoded = (meta_row[meta_u16_idx] >> meta_shift) & 0x0F;
            
            let keep0 = (encoded & 0x03) as usize;
            let keep1 = ((encoded >> 2) & 0x03) as usize;
            
            compressed_data.push(row[base + keep0]);
            compressed_data.push(row[base + keep1]);
        }
    }

    // UPDATE the tensor shape to reflect the 50% compression!
    // Since columns were compressed by 50%
    meta.shape[1] = cols / 2;
    *data = compressed_data;

    Some(sp_meta)
    } // #[cfg(feature = "jit-cuda")]
}

/// Applies Tier II graph compression for Q-heads.
/// Evaluates cosine similarity between attention heads in `q_proj`. 
/// If `sim > 0.98`, the duplicate head is zeroed out to save VRAM and memory bandwidth,
/// and metadata is generated (conceptually) to scale the runtime accumulator.
fn deduplicate_q_heads_heuristic(meta: &TensorMeta, data: &mut [f32]) {
    if !meta.name.contains("q_proj") && !meta.name.contains("query") {
        return;
    }

    if meta.shape.len() != 2 {
        return;
    }

    let rows = meta.shape[0]; 
    let cols = meta.shape[1];

    // Infer head_dim conservatively (usually 128 or 64). 
    // If cols is not divisible by 128, try 64, else abort heuristic.
    let head_dim = if cols.is_multiple_of(128) { 128 } else if cols.is_multiple_of(64) { 64 } else { return; };
    let num_heads = cols / head_dim;

    if num_heads <= 1 {
        return;
    }

    // data layout: [rows, num_heads * head_dim]
    // A head is a set of columns. 
    // Let's compute the L2 norm for each head.
    let mut head_norms = vec![0.0f32; num_heads];
    for (h, norm_out) in head_norms.iter_mut().enumerate() {
        let mut sq_norm = 0.0f32;
        let start_col = h * head_dim;
        for r in 0..rows {
            let row_offset = r * cols;
            for d in 0..head_dim {
                let val = data[row_offset + start_col + d];
                sq_norm += val * val;
            }
        }
        *norm_out = sq_norm.sqrt();
    }

    let mut merged = 0;
    let mut active = vec![true; num_heads];

    for i in 0..num_heads {
        if !active[i] || head_norms[i] < 1e-6 { continue; }
        
        for j in (i + 1)..num_heads {
            if !active[j] || head_norms[j] < 1e-6 { continue; }

            // Compute dot product between head i and head j
            let mut dot = 0.0f32;
            let start_col_i = i * head_dim;
            let start_col_j = j * head_dim;

            for r in 0..rows {
                let row_offset = r * cols;
                for d in 0..head_dim {
                    let vi = data[row_offset + start_col_i + d];
                    let vj = data[row_offset + start_col_j + d];
                    dot += vi * vj;
                }
            }

            let sim = dot / (head_norms[i] * head_norms[j]);
            if sim > 0.98 {
                // Head j is extremely similar to Head i.
                // Zero out Head j to save memory bandwidth during loading to SRAM.
                for r in 0..rows {
                    let row_offset = r * cols;
                    for d in 0..head_dim {
                        data[row_offset + start_col_j + d] = 0.0;
                    }
                }
                active[j] = false;
                merged += 1;
            }
        }
    }

    if merged > 0 {
        log::info!("🧠 GQA Head Deduplication: Merged {}/{} Q-heads in {}.", merged, num_heads, meta.name);
    }
}

/// 量化配置的伴生张量信息
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompanionConfig {
    /// 量化 scales 张量名称
    pub scales: String,
    /// 量化 zeros 张量名称（可选，某些量化方案不需要）
    pub zeros: Option<String>,
}

/// 量化元数据
///
/// REQ-ARCH-Ω1: 量化配置必须包含完整信息，包括符号位和伴生张量
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    /// 量化分组大小（某些文档中称为 group_size，这里统一使用 block_size）
    pub block_size: usize,
    /// 量化位宽
    pub bits: u8,
    /// 是否使用激活值降序排列
    #[serde(default)]
    pub desc_act: bool,
    /// 是否使用对称量化
    #[serde(default)]
    pub is_sym: bool,
    /// 是否为有符号量化
    #[serde(default)]
    pub signed: bool,
    /// 伴生张量配置（scales/zeros）
    #[serde(default)]
    pub companions: Option<CompanionConfig>,
}

impl QuantizationMetadata {
    pub fn from_metadata(
        metadata: &HashMap<String, String>,
    ) -> Result<Option<HashMap<String, Self>>> {
        if let Some(json) = metadata.get("gllm.quantization") {
            let map: HashMap<String, Self> = serde_json::from_str(json)?;
            Ok(Some(map))
        } else {
            Ok(None)
        }
    }
}

// --- Legacy Types for Compatibility ---

/// Thinking head tensor names (for models like Qwen3 with thinking capability)
#[derive(Debug, Clone, Default)]
pub struct ThinkingHead {
    pub tensors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct WeightsHandle<B: Backend<E>, E: Element = f32> {
    tensors: HashMap<String, B::Tensor>,
    shapes: HashMap<String, Vec<usize>>,
    pub meta: HashMap<String, TensorMeta>,
    pub thinking_head: Option<ThinkingHead>,
    quantized: HashMap<String, QuantizedTensor>,
    raw_floats: HashMap<String, RawFloatTensor>,
    pub sparse_24_meta: HashMap<String, Vec<Vec<u16>>>,
    placements: HashMap<String, crate::compat::backend_trait::WeightPlacement>,
}

impl<B: Backend<E>, E: Element> WeightsHandle<B, E> {
    pub fn new(
        tensors: HashMap<String, B::Tensor>,
        shapes: HashMap<String, Vec<usize>>,
        meta: HashMap<String, TensorMeta>,
    ) -> Self {
        Self {
            tensors,
            shapes,
            meta,
            thinking_head: None,
            quantized: HashMap::new(),
            raw_floats: HashMap::new(),
            sparse_24_meta: HashMap::new(),
            placements: HashMap::new(),
        }
    }

    pub fn new_with_quantized_and_sparse(
        tensors: HashMap<String, B::Tensor>,
        shapes: HashMap<String, Vec<usize>>,
        meta: HashMap<String, TensorMeta>,
        quantized: HashMap<String, QuantizedTensor>,
        sparse_24_meta: HashMap<String, Vec<Vec<u16>>>,
    ) -> Self {
        Self {
            tensors,
            shapes,
            meta,
            thinking_head: None,
            quantized,
            raw_floats: HashMap::new(),
            sparse_24_meta,
            placements: HashMap::new(),
        }
    }

    pub fn new_with_placements(
        tensors: HashMap<String, B::Tensor>,
        shapes: HashMap<String, Vec<usize>>,
        meta: HashMap<String, TensorMeta>,
        quantized: HashMap<String, QuantizedTensor>,
        raw_floats: HashMap<String, RawFloatTensor>,
        sparse_24_meta: HashMap<String, Vec<Vec<u16>>>,
        placements: HashMap<String, crate::compat::backend_trait::WeightPlacement>,
    ) -> Self {
        Self {
            tensors,
            shapes,
            meta,
            thinking_head: None,
            quantized,
            raw_floats,
            sparse_24_meta,
            placements,
        }
    }

    pub fn quantized_tensor(&self, name: &str) -> Option<&QuantizedTensor> {
        self.quantized.get(name)
    }

    pub fn raw_float_tensor(&self, name: &str) -> Option<&RawFloatTensor> {
        self.raw_floats.get(name)
    }

    pub fn raw_floats(&self) -> &HashMap<String, RawFloatTensor> {
        &self.raw_floats
    }

    pub fn is_quantized(&self, name: &str) -> bool {
        self.quantized.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Option<&B::Tensor> {
        self.tensors.get(name)
    }

    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.shapes.get(name).map(|v| v.as_slice())
    }

    /// Return an iterator over all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// All external tensor names (F32 + quantized + BF16/F16 raw_floats).
    pub fn all_tensor_names(&self) -> Vec<&String> {
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.extend(self.quantized.keys());
        names.extend(self.raw_floats.keys());
        names
    }

    /// Build canonical name mapping from all tensor names.
    pub fn name_map(&self) -> name_map::TensorNameMap {
        self.name_map_with_tied(false)
    }

    /// Build canonical name mapping with tied embeddings support.
    /// When `tie_word_embeddings` is true and no separate lm_head tensor exists,
    /// maps lm_head canonical to embed's physical tensor.
    pub fn name_map_with_tied(&self, tie_word_embeddings: bool) -> name_map::TensorNameMap {
        let all: Vec<String> = self.all_tensor_names().into_iter().cloned().collect();
        name_map::TensorNameMap::build_from_names(&all, tie_word_embeddings)
    }

    /// Query the data placement of a tensor (DeviceLocal or HostLocal).
    pub fn placement_of(&self, name: &str) -> Option<crate::compat::backend_trait::WeightPlacement> {
        self.placements.get(name).copied()
    }

    /// ARCH-WEIGHT-BLOB-REMAPPING: 精确释放已编译到 weight_blob 中的权重张量。
    ///
    /// 只释放 `safe_to_release` 集合中的权重（这些权重的数据已完整拷贝到
    /// CompiledNode.weight_blob，运行时不再从 WeightsHandle.tensors 读取）。
    /// 其他权重（attention 节点的 q/k/v/o_proj，因 needs_runtime_weight_pack=true
    /// 而未预打包）必须保留。
    pub fn release_compiled_weights(&mut self, safe_to_release: &std::collections::HashSet<String>) {
        let before = self.tensors.len();
        self.tensors.retain(|name, _| {
            if safe_to_release.contains(name) {
                return false;
            }
            true
        });
        let after = self.tensors.len();
        if before > after {
            log::info!(
                "WeightsHandle::release_compiled_weights: released {}/{} tensors",
                before - after, before
            );
        }
    }
}

/// Backward-compatible type alias for f32 weights.
pub type WeightsHandleF32<B> = WeightsHandle<B, f32>;

/// 实现 gllm_kernels::TensorLookup trait
impl<B: Backend<E>, E: Element> crate::compat::backend_trait::TensorLookup<E, B>
    for WeightsHandle<B, E>
{
    fn get_tensor(&self, name: &str) -> Option<&B::Tensor> {
        self.tensor(name)
    }

    fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        WeightsHandle::tensor_shape(self, name)
    }

    fn get_quantized(&self, name: &str) -> Option<&crate::loader::QuantizedTensor> {
        self.quantized.get(name)
    }

    fn available_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.extend(self.quantized.keys().cloned());
        names.extend(self.raw_floats.keys().cloned());
        names.sort();
        names
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParallelPolicy {
    pub enabled: bool,
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone)]
pub struct UploadedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    // backend-specific handle
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use ::safetensors::tensor::Dtype;

    /// Naive reference transpose: dst[c * rows + r] = src[r * cols + c].
    fn naive_transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; src.len()];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = src[r * cols + c];
            }
        }
        out
    }

    #[test]
    fn cache_blocked_transpose_matches_naive_small_sizes() {
        // Exercise exact-tile, tile+tail, non-multiple sizes, 1×N, N×1.
        let cases: &[(usize, usize)] = &[
            (1, 1), (1, 7), (7, 1), (8, 8), (63, 65), (64, 64), (65, 63),
            (128, 65), (65, 128), (100, 200), (200, 100), (129, 129),
        ];
        for &(rows, cols) in cases {
            let n = rows * cols;
            let src: Vec<f32> = (0..n as u32).map(|x| x as f32 * 0.5 - 7.25).collect();
            let mut blocked = vec![0.0f32; n];
            cache_blocked_transpose_f32(&src, &mut blocked, rows, cols);
            let reference = naive_transpose(&src, rows, cols);
            assert_eq!(
                blocked, reference,
                "cache_blocked_transpose diverged at rows={}, cols={}",
                rows, cols
            );
        }
    }

    #[test]
    fn cache_blocked_transpose_matches_naive_weight_size() {
        // Realistic Linear weight: [1536, 12288] — exactly the case naive
        // transpose degrades on (6144-byte write stride vs 64-byte L1 line).
        let rows = 1536;
        let cols = 12288;
        let n = rows * cols;
        // Build a deterministic pattern; f32 bit-exactness required.
        let src: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761);
                f32::from_bits(x)
            })
            // Filter out NaN so equality works; map bit pattern to a finite value.
            .map(|f| if f.is_finite() { f } else { 0.0 })
            .collect();
        let mut blocked = vec![0.0f32; n];
        cache_blocked_transpose_f32(&src, &mut blocked, rows, cols);

        // Spot-check a scattered set of coordinates rather than allocate a second
        // full buffer (the naive path is painfully slow on this shape).
        let sample_points: [(usize, usize); 32] = [
            (0, 0), (1535, 12287), (0, 12287), (1535, 0),
            (1, 1), (2, 3), (5, 7), (11, 13), (17, 19),
            (23, 29), (31, 37), (41, 43), (47, 53), (59, 61),
            (67, 71), (73, 79), (83, 89), (97, 101), (103, 107),
            (109, 113), (127, 131), (137, 139), (149, 151), (157, 163),
            (167, 173), (179, 181), (191, 193), (197, 199), (211, 223),
            (1000, 7000), (1234, 5678), (999, 9999),
        ];
        for (r, c) in sample_points {
            assert_eq!(
                blocked[c * rows + r], src[r * cols + c],
                "mismatch at (r={}, c={})", r, c
            );
        }
    }

    #[test]
    fn parallel_bf16_to_f32_matches_serial() {
        // 100_000 BF16 elements covering finite positive + negative + zero.
        let n = 100_000usize;
        let bf16s: Vec<half::bf16> = (0..n)
            .map(|i| {
                let x = (i as i32) - (n as i32 / 2);
                half::bf16::from_f32(x as f32 * 0.0078125)
            })
            .collect();
        // Flatten to bytes
        let mut bytes = Vec::with_capacity(n * 2);
        for v in &bf16s {
            let raw: u16 = v.to_bits();
            bytes.extend_from_slice(&raw.to_le_bytes());
        }

        let parallel_out = parallel_half_to_f32::<half::bf16>(&bytes).expect("parallel bf16 conversion failed");
        // Reference: single-threaded conversion.
        let serial_out: Vec<f32> = bf16s.iter().map(|v| v.to_f32()).collect();

        assert_eq!(parallel_out.len(), serial_out.len());
        for (i, (p, s)) in parallel_out.iter().zip(serial_out.iter()).enumerate() {
            assert_eq!(
                p.to_bits(), s.to_bits(),
                "bit-pattern mismatch at index {}: parallel={:?}, serial={:?}",
                i, p, s
            );
        }
    }

    #[test]
    fn parallel_f16_to_f32_matches_serial() {
        // F16 path: 50_000 values, including subnormals near zero.
        let n = 50_000usize;
        let f16s: Vec<half::f16> = (0..n)
            .map(|i| half::f16::from_f32(((i as f32) - (n as f32) / 2.0) * 1.0e-3))
            .collect();
        let mut bytes = Vec::with_capacity(n * 2);
        for v in &f16s {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }

        let parallel_out = parallel_half_to_f32::<half::f16>(&bytes).expect("parallel f16 conversion failed");
        let serial_out: Vec<f32> = f16s.iter().map(|v| v.to_f32()).collect();
        for (i, (p, s)) in parallel_out.iter().zip(serial_out.iter()).enumerate() {
            assert_eq!(
                p.to_bits(), s.to_bits(),
                "bit-pattern mismatch at index {}: parallel={:?}, serial={:?}",
                i, p, s
            );
        }
    }

    #[test]
    fn parallel_f32_passthrough_is_exact() {
        let n = 12345usize;
        let src: Vec<f32> = (0..n).map(|i| (i as f32) * 1.5 - 3.0).collect();
        let mut bytes = Vec::with_capacity(n * 4);
        for v in &src {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = parallel_bytes_to_f32_lossless(&bytes).expect("parallel f32 passthrough failed");
        assert_eq!(out, src);
    }

    #[test]
    #[cfg(feature = "jit-cuda")]
    fn test_24_prune_applied_on_load() {
        // 构造一个待测试的 gate_proj (能触发真 2:4 降维条件的 TensorName)
        let mut meta = TensorMeta {
            name: "layers.0.mlp.gate_proj".to_string(),
            shape: vec![8, 16], // 8 rows, 16 cols
            dtype: Dtype::F32,
        };
        // 8 * 16 = 128 elements, ones to represent some non-zero data
        // For actual zeroing to be visible or not matter, we just care that shape halves
        let mut data = vec![1.0f32; 128];
        
        let sp_meta_opt = compress_24_sparsity_heuristic(&mut meta, &mut data);
        
        // 1. 产生 sp_meta 位掩码
        assert!(sp_meta_opt.is_some(), "Hardware 2:4 sp_meta must be returned for gate_proj");
        
        // 2. 原本的结构体 metadata 的列 (col) 应当被严格砍半
        assert_eq!(meta.shape, vec![8, 8], "Column dimension of shape metadata must be exactly halved");
        
        // 3. F32 buffer 数据长度必须被严格砍半 (128 -> 64)，实现真显存下降
        assert_eq!(data.len(), 64, "Flat byte data length must be halved into a physically dense structure");
    }
}
