//! Layer 2/3: Loader (HF + SafeTensors + fused splits).

// unused imports removed
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ::safetensors::Dtype;
use gllm_kernels::backend_trait::{Backend, Element};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::manifest::{ModelManifest, TensorRole, EMPTY_FILE_MAP};

// Re-export modules
pub mod adapter;
pub mod config;
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
        Self {
            cache_dir: PathBuf::from("~/.gllm/models"),
            source: ModelSource::HuggingFace,
            hf_token_path: None,
            enable_fallback: true,
            checksum_policy: ChecksumPolicy::Ignore,
        }
    }
}

impl LoaderConfig {
    pub fn from_env() -> Self {
        Self::default()
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
    match err {
        LoaderError::Network(_) | LoaderError::Io(_) | LoaderError::HfHub(_) => true,
        _ => false,
    }
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
    // Embedding
    if (lower.contains("embed") || lower.contains("wte")) && layer_idx.is_none() {
        // Exclude position embeddings if necessary, but usually main embedding is largest
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
    {
        return Some((TensorRole::OutputHead, None));
    }

    // Layer-specific roles
    if lower.contains("norm") || lower.contains("ln_") {
        return Some((TensorRole::LayerNorm, layer_idx));
    }

    // Attention
    if lower.contains("q_proj") || lower.contains("query") || lower.contains("wq") {
        return Some((TensorRole::AttentionQuery, layer_idx));
    }
    if lower.contains("k_proj") || lower.contains("key") || lower.contains("wk") {
        return Some((TensorRole::AttentionKey, layer_idx));
    }
    if lower.contains("v_proj") || lower.contains("value") || lower.contains("wv") {
        return Some((TensorRole::AttentionValue, layer_idx));
    }
    if lower.contains("o_proj") || lower.contains("wo") {
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
                // Not implemented yet
            }
        }
        Ok(self)
    }

    pub fn weight_format(&self) -> WeightFormat {
        self.format.clone()
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

    pub fn onnx_loader(&mut self) -> Result<&mut onnx::OnnxLoader> {
        self.onnx.as_mut().ok_or(LoaderError::MissingWeights)
    }

    pub fn onnx_ref(&self) -> Option<&onnx::OnnxLoader> {
        self.onnx.as_ref()
    }

    pub fn onnx(&mut self) -> Result<&mut onnx::OnnxLoader> {
        self.onnx_loader()
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
            WeightFormat::PyTorch => Err(LoaderError::UnsupportedWeightExtension(
                "PyTorch not supported yet".into(),
            )),
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

        for meta in provider.iter_tensors() {
            match meta.dtype {
                Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::F64 => {
                    let data = provider.load_tensor_data(&meta.name)?;
                    let converted = convert_to_element::<E>(meta.dtype, &data)?;
                    let tensor = backend
                        .upload_weights(&converted)
                        .map_err(|e| LoaderError::Backend(e.to_string()))?;

                    tensors.insert(meta.name.clone(), tensor);
                    shapes.insert(meta.name.clone(), meta.shape.clone());
                    meta_map.insert(meta.name.clone(), meta);
                }
                _ => {
                    // Skip unsupported types (e.g. integer indices, boolean masks)
                    // These are not model weights usually.
                }
            }
        }

        Ok(WeightsHandle::new(tensors, shapes, meta_map))
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

    pub fn as_f32(&self) -> Result<Cow<'a, [f32]>> {
        convert_to_f32(self.dtype, self.data)
    }
}

fn convert_to_f32(dtype: Dtype, data: &[u8]) -> Result<Cow<'_, [f32]>> {
    match dtype {
        Dtype::F32 => cast_or_copy_f32(data),
        Dtype::F16 => {
            if data.len() % 2 != 0 {
                return Err(LoaderError::InvalidQuantization(
                    "F16 data length not multiple of 2".into(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let value = f16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(value.to_f32());
            }
            Ok(Cow::Owned(out))
        }
        Dtype::BF16 => {
            if data.len() % 2 != 0 {
                return Err(LoaderError::InvalidQuantization(
                    "BF16 data length not multiple of 2".into(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let value = bf16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(value.to_f32());
            }
            Ok(Cow::Owned(out))
        }
        Dtype::F64 => {
            if data.len() % 8 != 0 {
                return Err(LoaderError::InvalidQuantization(
                    "F64 data length not multiple of 8".into(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 8);
            for chunk in data.chunks_exact(8) {
                let value = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(value as f32);
            }
            Ok(Cow::Owned(out))
        }
        _ => Err(LoaderError::UnsupportedDtype(dtype)),
    }
}

/// Convert raw tensor data to a vector of generic element type E.
///
/// This function handles dtype conversion from various floating point formats
/// (F32, F16, BF16, F64) to the target element type E.
///
/// Optimized paths:
/// - Same type (e.g., F16 → f16): zero-copy transmute when aligned, otherwise byte copy
/// - Different types: direct conversion without f32 intermediate when possible
fn convert_to_element<E: Element>(dtype: Dtype, data: &[u8]) -> Result<Vec<E>> {
    use std::any::TypeId;

    let target_type = TypeId::of::<E>();

    // Fast path: source dtype matches target E type (zero-copy or direct copy)
    match dtype {
        Dtype::F32 if target_type == TypeId::of::<f32>() => {
            // F32 → f32: direct transmute
            return Ok(cast_or_copy_typed::<f32, E>(data));
        }
        Dtype::F16 if target_type == TypeId::of::<half::f16>() => {
            // F16 → f16: direct transmute
            return Ok(cast_or_copy_typed::<half::f16, E>(data));
        }
        Dtype::BF16 if target_type == TypeId::of::<half::bf16>() => {
            // BF16 → bf16: direct transmute
            return Ok(cast_or_copy_typed::<half::bf16, E>(data));
        }
        _ => {}
    }

    // Cross-type conversion: use the most direct path
    // For float types, we can convert directly using Float trait
    match dtype {
        Dtype::F32 => {
            let f32_slice = cast_or_copy_f32(data)?;
            Ok(f32_slice.iter().map(|&v| E::from_f32(v)).collect())
        }
        Dtype::F16 => {
            let count = data.len() / 2;
            let mut result = Vec::with_capacity(count);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f16_val = half::f16::from_bits(bits);
                result.push(E::from_f32(f16_val.to_f32()));
            }
            Ok(result)
        }
        Dtype::BF16 => {
            let count = data.len() / 2;
            let mut result = Vec::with_capacity(count);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let bf16_val = half::bf16::from_bits(bits);
                result.push(E::from_f32(bf16_val.to_f32()));
            }
            Ok(result)
        }
        Dtype::F64 => {
            let count = data.len() / 8;
            let mut result = Vec::with_capacity(count);
            for chunk in data.chunks_exact(8) {
                let bits = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                let f64_val = f64::from_bits(bits);
                result.push(E::from_f32(f64_val as f32));
            }
            Ok(result)
        }
        _ => Err(LoaderError::UnsupportedDtype(dtype)),
    }
}

/// Zero-copy or byte-copy transmute for same-type conversions.
///
/// Safety: This is safe because we only call this when TypeId confirms
/// the source and target types are the same.
#[inline]
fn cast_or_copy_typed<S: Copy + 'static, E: Element>(data: &[u8]) -> Vec<E> {
    let elem_size = std::mem::size_of::<S>();
    let count = data.len() / elem_size;

    // Try zero-copy if aligned
    let (prefix, body, suffix) = unsafe { data.align_to::<S>() };
    if prefix.is_empty() && suffix.is_empty() {
        // Safety: S and E have the same TypeId, so they're the same type
        // We transmute the reference and copy
        let e_slice: &[E] =
            unsafe { std::slice::from_raw_parts(body.as_ptr() as *const E, body.len()) };
        return e_slice.to_vec();
    }

    // Fallback: byte-by-byte copy
    let mut result = Vec::with_capacity(count);
    for chunk in data.chunks_exact(elem_size) {
        // Copy bytes to properly aligned storage
        let mut bytes = [0u8; 8]; // Max size we handle
        bytes[..elem_size].copy_from_slice(chunk);
        let val: S = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const S) };
        // Safety: S and E are the same type (verified by TypeId)
        let e_val: E = unsafe { std::mem::transmute_copy(&val) };
        result.push(e_val);
    }
    result
}

fn cast_or_copy_f32(data: &[u8]) -> Result<Cow<'_, [f32]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<f32>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(value);
    }
    Ok(Cow::Owned(out))
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

#[derive(Debug, Clone)]
pub struct WeightsHandle<B: Backend<E>, E: Element = f32> {
    tensors: HashMap<String, B::Tensor>,
    shapes: HashMap<String, Vec<usize>>,
    pub meta: HashMap<String, TensorMeta>,
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
        }
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
