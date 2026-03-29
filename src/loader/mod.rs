//! Layer 2/3: Loader (HF + SafeTensors + fused splits).

// unused imports removed
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ::safetensors::Dtype;
use crate::compat::backend_trait::{Backend, Element};
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
pub mod onnx;
pub mod parallel;
pub mod pytorch;
pub mod safetensors;

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
            .unwrap_or_else(|| PathBuf::from(".gllm/models"));
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

/// Matches a tensor name to a role and optional layer index
pub fn match_tensor_role(name: &str) -> Option<(TensorRole, Option<usize>)> {
    let lower = name.to_ascii_lowercase();

    // Skip bias tensors — they are 1D and should not be matched as projection weights
    if lower.ends_with(".bias") || lower.ends_with("_bias") {
        return None;
    }

    // 1. Extract layer index if present
    // Common patterns: layers.N, blk.N, blocks.N, .h.N
    let mut layer_idx = None;

    // Simple heuristic: look for numeric segments preceded by layer keywords
    let parts: Vec<&str> = lower.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if let Ok(idx) = part.parse::<usize>() {
            if i > 0 {
                let prefix = parts[i - 1];
                if matches!(
                    prefix,
                    "layers" | "blk" | "blocks" | "h" | "layer" | "block"
                ) {
                    layer_idx = Some(idx);
                    break;
                }
            }
        }
    }

    // 2. Match Role
    // Layer-specific roles — check FIRST to avoid false matches
    // (e.g. "embeddings.LayerNorm.bias" contains "embed" but is a norm tensor)
    if lower.contains("norm") || lower.contains("ln_") {
        return Some((TensorRole::LayerNorm, layer_idx));
    }

    // Embedding
    if (lower.contains("embed") || lower.contains("embd") || lower.contains("wte")) && layer_idx.is_none() {
        // Exclude position embeddings and type embeddings
        if !lower.contains("pos") && !lower.contains("type") {
            return Some((TensorRole::Embedding, None));
        }
    }
    // BERT style embedding
    if lower.contains("word_embeddings") {
        return Some((TensorRole::Embedding, None));
    }

    // Output Head
    if (lower.contains("lm_head") || lower.contains("output"))
        && !lower.contains("layer")
        && !lower.contains("attention")
        && !lower.contains("attn")
    {
        return Some((TensorRole::OutputHead, None));
    }

    // Attention
    if lower.contains("q_proj") || lower.contains("query") || lower.contains("wq") || lower.contains("attn_q") {
        return Some((TensorRole::AttentionQuery, layer_idx));
    }
    if lower.contains("k_proj") || lower.contains("key") || lower.contains("wk") || lower.contains("attn_k") {
        return Some((TensorRole::AttentionKey, layer_idx));
    }
    if lower.contains("v_proj") || lower.contains("value") || lower.contains("wv") || lower.contains("attn_v.") {
        return Some((TensorRole::AttentionValue, layer_idx));
    }
    if lower.contains("o_proj") || lower.contains("wo") || lower.contains("attn_output") {
        return Some((TensorRole::AttentionOutput, layer_idx));
    }
    // BERT Attention Output: "attention.output.dense"
    if lower.contains("attention") && lower.contains("output") {
        return Some((TensorRole::AttentionOutput, layer_idx));
    }

    // MLP
    if lower.contains("gate_proj") || lower.contains("w1") || lower.contains("ffn_gate") {
        return Some((TensorRole::FfnGate, layer_idx));
    }
    if lower.contains("up_proj") || lower.contains("w3") || lower.contains("ffn_up") {
        return Some((TensorRole::FfnUp, layer_idx));
    }
    if lower.contains("down_proj") || lower.contains("w2") || lower.contains("ffn_down") {
        return Some((TensorRole::FfnDown, layer_idx));
    }
    // BERT Intermediate (FFN Up)
    if lower.contains("intermediate") {
        return Some((TensorRole::FfnUp, layer_idx));
    }
    // BERT Output (FFN Down) - must check this AFTER attention output
    if lower.contains("output") && lower.contains("dense") {
        return Some((TensorRole::FfnDown, layer_idx));
    }

    // RoPE (if explicit)
    if lower.contains("rope") {
        return Some((TensorRole::Rope, layer_idx));
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
    /// 对于 ONNX 文件，直接返回解析的图。
    /// 对于其他格式，使用架构模板生成图。
    pub fn to_unified_graph(&mut self) -> Result<onnx::OnnxGraph> {
        match self.format {
            WeightFormat::Onnx => {
                let onnx = self.onnx()?;
                Ok(onnx.graph().clone())
            }
            WeightFormat::SafeTensors | WeightFormat::Gguf => {
                // 使用架构模板系统生成图
                use crate::arch::{
                    get_template_by_arch, register_builtin_templates, resolve_config,
                };

                register_builtin_templates();

                // 1. 检测架构
                let arch = self.detect_architecture();

                // 2. 获取模板
                let template = get_template_by_arch(arch).ok_or_else(|| {
                    LoaderError::Onnx(format!("No template for arch: {:?}", arch))
                })?;

                // 3. 解析配置 - 需要获取 TensorProvider
                let config = match self.format {
                    WeightFormat::SafeTensors => {
                        let st = self
                            .safetensors
                            .as_ref()
                            .ok_or(LoaderError::MissingWeights)?;
                        resolve_config(template, st, self.gguf.as_ref()).map_err(|e| {
                            LoaderError::Onnx(format!("Config resolve failed: {}", e))
                        })?
                    }
                    WeightFormat::Gguf => {
                        let gguf = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;
                        resolve_config(template, gguf, Some(gguf)).map_err(|e| {
                            LoaderError::Onnx(format!("Config resolve failed: {}", e))
                        })?
                    }
                    _ => unreachable!(),
                };

                // 4. 生成图
                template
                    .to_onnx_graph(&config)
                    .map_err(|e| LoaderError::Onnx(format!("Template to graph failed: {}", e)))
            }
            _ => unreachable!("PyTorch is converted to SafeTensors by load()"),
        }
    }

    /// 检测模型架构（统一入口）
    ///
    /// 优先级：GGUF metadata > 张量名称模式匹配 > manifest fallback
    pub fn detect_architecture(&self) -> crate::manifest::ModelArchitecture {
        use crate::manifest::map_architecture_token;

        // 1. GGUF metadata
        if let Some(gguf) = &self.gguf {
            if let Ok(arch_str) = gguf.architecture() {
                if let Some(arch) = map_architecture_token(arch_str) {
                    return arch;
                }
            }
        }

        // 2. 张量名称模式匹配
        if let Some(arch) = self.detect_architecture_from_tensors() {
            return arch;
        }

        // 3. manifest fallback
        self.manifest.arch
    }

    /// 从张量名称推断架构
    fn detect_architecture_from_tensors(&self) -> Option<crate::manifest::ModelArchitecture> {
        use crate::manifest::ModelArchitecture;

        let check_name = |name: &str| -> Option<ModelArchitecture> {
            let lower = name.to_lowercase();
            if lower.contains("bert") || lower.contains("roberta") {
                return Some(ModelArchitecture::XlmR);
            }
            if lower.contains("mistral") {
                return Some(ModelArchitecture::Mistral3);
            }
            if lower.contains("encoder.layer.") || lower.contains("attention.self.query") {
                return Some(ModelArchitecture::XlmR);
            }
            None
        };

        if let Some(st) = self.safetensors.as_ref() {
            for meta in st.iter_tensors() {
                if let Some(arch) = check_name(&meta.name) {
                    return Some(arch);
                }
            }
        }
        if let Some(onnx) = self.onnx.as_ref() {
            for meta in onnx.iter_tensors() {
                if let Some(arch) = check_name(&meta.name) {
                    return Some(arch);
                }
            }
        }
        if let Some(gguf) = self.gguf.as_ref() {
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

    pub fn detect_weight_dtype_size(&self) -> Result<Option<usize>> {
        if let Some(loader) = &self.safetensors {
            Ok(loader.detect_weight_dtype_size())
        } else if let Some(reader) = &self.gguf {
            Ok(reader.floating_point_dtype_size())
        } else if let Some(loader) = &self.onnx {
            // ONNX might have mixed precision, but we can try to find the dominant floating point type
            // For now, let's look at the first few tensors
            let precisions = loader.unique_precisions();
            for dtype in precisions {
                match dtype {
                    Dtype::F32 => return Ok(Some(4)),
                    Dtype::F16 | Dtype::BF16 => return Ok(Some(2)),
                    Dtype::F64 => return Ok(Some(8)),
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
        match self.format {
            WeightFormat::SafeTensors => {
                let provider = self
                    .safetensors
                    .as_ref()
                    .ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend)
            }
            WeightFormat::Gguf => {
                let provider = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend)
            }
            WeightFormat::Onnx => {
                let provider = self.onnx.as_ref().ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend)
            }
            _ => unreachable!("PyTorch is converted to SafeTensors by load()"),
        }
    }

    fn upload_provider<P: TensorProvider, B: Backend<E>, E: Element>(
        &self,
        provider: &P,
        backend: &B,
    ) -> Result<WeightsHandle<B, E>> {
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        let mut meta_map = HashMap::new();
        let mut quantized = HashMap::new();
        let mut sparse_24 = HashMap::new();

        for meta in provider.iter_tensors() {
            // Check if this tensor has a quantized GGML dtype
            if let Some(ggml_dt) = provider.ggml_dtype(&meta.name) {
                if let Some(qt) = adapter::ggml_dtype_to_quant_type(ggml_dt) {
                    // Quantized tensor — store raw bytes with QuantType metadata
                    let data = provider.load_tensor_data(&meta.name)?;
                    quantized.insert(
                        meta.name.clone(),
                        QuantizedTensor {
                            data: data.into_owned(),
                            quant_type: qt,
                            shape: meta.shape.clone(),
                            ggml_dtype: ggml_dt,
                        },
                    );
                    shapes.insert(meta.name.clone(), meta.shape.clone());
                    meta_map.insert(meta.name.clone(), meta);
                    continue;
                }
            }

            // Native float tensor — upload with dtype conversion if needed
            match meta.dtype {
                Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::F64 => {
                    let data = provider.load_tensor_data(&meta.name)?;
                    let (cloned_meta, tensor, sp_meta_opt) =
                        upload_native_tensor_with_convert::<B, E>(backend, &meta, data.as_ref())?;

                    tensors.insert(meta.name.clone(), tensor);
                    shapes.insert(cloned_meta.name.clone(), cloned_meta.shape.clone());
                    meta_map.insert(cloned_meta.name.clone(), cloned_meta);
                    
                    if let Some(sp_meta) = sp_meta_opt {
                        sparse_24.insert(meta.name.clone(), sp_meta);
                    }
                }
                _ => {
                    // Skip unsupported types (e.g. integer indices, boolean masks)
                }
            }
        }

        Ok(WeightsHandle::new_with_quantized_and_sparse(
            tensors, shapes, meta_map, quantized, sparse_24,
        ))
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

fn upload_native_tensor<B: Backend<E>, E: Element>(
    backend: &B,
    meta: &TensorMeta,
    data: &[u8],
) -> Result<B::Tensor> {
    let elem_size = std::mem::size_of::<E>();
    if elem_size == 0 || !data.len().is_multiple_of(elem_size) {
        return Err(LoaderError::InvalidQuantization(format!(
            "tensor {} has invalid byte length {} for element size {}",
            meta.name,
            data.len(),
            elem_size
        )));
    }

    let dtype_matches = match meta.dtype {
        Dtype::F32 => std::any::TypeId::of::<E>() == std::any::TypeId::of::<f32>(),
        Dtype::F16 => std::any::TypeId::of::<E>() == std::any::TypeId::of::<half::f16>(),
        Dtype::BF16 => std::any::TypeId::of::<E>() == std::any::TypeId::of::<half::bf16>(),
        Dtype::F64 => std::any::TypeId::of::<E>() == std::any::TypeId::of::<f64>(),
        _ => false,
    };
    if !dtype_matches {
        return Err(LoaderError::Backend(format!(
            "native-dtype upload required: tensor '{}' is {:?}, backend expects {}",
            meta.name,
            meta.dtype,
            std::any::type_name::<E>()
        )));
    }

    let (prefix, body, suffix) = unsafe { data.align_to::<E>() };
    if prefix.is_empty() && suffix.is_empty() {
        return backend
            .upload_weights(body)
            .map_err(|e| LoaderError::Backend(e.to_string()));
    }

    let count = data.len() / elem_size;
    let mut converted = Vec::with_capacity(count);
    for chunk in data.chunks_exact(elem_size) {
        let value: E = unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const E) };
        converted.push(value);
    }
    backend
        .upload_weights(&converted)
        .map_err(|e| LoaderError::Backend(e.to_string()))
}

fn upload_native_tensor_with_convert<B: Backend<E>, E: Element>(
    backend: &B,
    meta: &TensorMeta,
    data: &[u8],
) -> Result<(TensorMeta, B::Tensor, Option<Vec<Vec<u16>>>)> {
    let is_f32_backend = std::any::TypeId::of::<E>() == std::any::TypeId::of::<f32>();

    // P4/P5 heuristics (sparsity, deduplication) operate on f32.
    // We must extract a mutable f32 buffer regardless of whether conversion is needed.
    let mut converted_f32: Vec<f32> = match meta.dtype {
        Dtype::F32 => {
            let src_size = std::mem::size_of::<f32>();
            data.chunks_exact(src_size)
                .map(|chunk| unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const f32) })
                .collect()
        }
        Dtype::F16 => {
            let src_size = std::mem::size_of::<half::f16>();
            data.chunks_exact(src_size)
                .map(|chunk| {
                    let val: half::f16 =
                        unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const half::f16) };
                    val.to_f32()
                })
                .collect()
        }
        Dtype::BF16 => {
            let src_size = std::mem::size_of::<half::bf16>();
            data.chunks_exact(src_size)
                .map(|chunk| {
                    let val: half::bf16 = unsafe {
                        std::ptr::read_unaligned(chunk.as_ptr() as *const half::bf16)
                    };
                    val.to_f32()
                })
                .collect()
        }
        Dtype::F64 => {
            let src_size = std::mem::size_of::<f64>();
            data.chunks_exact(src_size)
                .map(|chunk| {
                    let val: f64 =
                        unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const f64) };
                    val as f32
                })
                .collect()
        }
        _ => {
            return Err(LoaderError::Backend(format!(
                "cannot convert {:?} to f32 for heuristics",
                meta.dtype
            )));
        }
    };

    let mut cloned_meta = meta.clone();

    // Apply P4/P5 Tier II Structural Sparsity logic on the CPU prior to GPU upload
    apply_ffn_sparsity_heuristic(&cloned_meta, &mut converted_f32);
    let sp_meta_opt = compress_24_sparsity_heuristic(&mut cloned_meta, &mut converted_f32);
    deduplicate_q_heads_heuristic(&mut cloned_meta, &mut converted_f32);

    if is_f32_backend {
        // Safety: we know E is f32 here
        let as_e: &[E] =
            unsafe { std::slice::from_raw_parts(converted_f32.as_ptr() as *const E, converted_f32.len()) };
        let tensor = backend
            .upload_weights(as_e)
            .map_err(|e| LoaderError::Backend(e.to_string()))?;
        return Ok((cloned_meta, tensor, sp_meta_opt));
    }

    Err(LoaderError::Backend(format!(
        "backend dtype {} not fully supported for zero-copy upload after heuristics",
        std::any::type_name::<E>()
    )))
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
    for r in 0..rows {
        if l2_norms[r] < threshold {
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
    #[cfg(not(feature = "jit-cuda"))]
    {
        let _ = (meta, data);
        return None;
    }

    #[cfg(feature = "jit-cuda")]
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
    let head_dim = if cols % 128 == 0 { 128 } else if cols % 64 == 0 { 64 } else { return; };
    let num_heads = cols / head_dim;

    if num_heads <= 1 {
        return;
    }

    // data layout: [rows, num_heads * head_dim]
    // A head is a set of columns. 
    // Let's compute the L2 norm for each head.
    let mut head_norms = vec![0.0f32; num_heads];
    for h in 0..num_heads {
        let mut sq_norm = 0.0f32;
        let start_col = h * head_dim;
        for r in 0..rows {
            let row_offset = r * cols;
            for d in 0..head_dim {
                let val = data[row_offset + start_col + d];
                sq_norm += val * val;
            }
        }
        head_norms[h] = sq_norm.sqrt();
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    pub group_size: usize,
    pub bits: u8,
    #[serde(default)]
    pub desc_act: bool,
    #[serde(default)]
    pub is_sym: bool,
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
    pub sparse_24_meta: HashMap<String, Vec<Vec<u16>>>,
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
            sparse_24_meta: HashMap::new(),
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
            sparse_24_meta,
        }
    }

    pub fn quantized_tensor(&self, name: &str) -> Option<&QuantizedTensor> {
        self.quantized.get(name)
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
    use ::safetensors::tensor::Dtype;

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
