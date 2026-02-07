//! Layer 2/3: Loader (HF + SafeTensors + fused splits).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ::safetensors::Dtype;
use gllm_kernels::backend_trait::{Backend, TensorLookup};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::manifest::{ModelKind, ModelManifest, TensorNamingRule, EMPTY_FILE_MAP};
use crate::quantization::dequantize_int8_with_zero;

/// Ω1: 量化元数据 - 完全由模型文件提供，不基于推测
///
/// 存储在 safetensors 元数据的 `gllm.quantization` 字段中：
/// ```json
/// {
///   "qweight": {
///     "bits": 4,
///     "signed": false,
///     "block_size": 128,
///     "group_size": 64,
///     "companions": {
///       "scales": "scales",
///       "zeros": "qzeros"
///     }
///   }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct QuantizationMetadata {
    /// 量化位宽 (4, 8)
    pub bits: u8,
    /// 是否有符号
    #[serde(default)]
    pub signed: bool,
    /// 量化块大小（用于反量化时的索引计算）
    /// Ω1: 必须由模型提供，不允许默认值
    pub block_size: usize,
    /// 关联张量映射
    #[serde(default)]
    pub companions: CompanionTensors,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Default)]
pub struct CompanionTensors {
    /// scales 张量名称
    pub scales: Option<String>,
    /// zeros 张量名称
    pub zeros: Option<String>,
    /// 其他关联张量
    #[serde(flatten)]
    pub others: HashMap<String, String>,
}

impl QuantizationMetadata {
    /// 从 safetensors 元数据解析量化信息
    pub fn from_metadata(
        meta: &std::collections::HashMap<String, String>,
    ) -> Result<Option<HashMap<String, QuantizationMetadata>>> {
        let Some(encoded) = meta.get("gllm.quantization") else {
            return Ok(None);
        };

        serde_json::from_str(encoded)
            .map(Some)
            .map_err(|e| LoaderError::InvalidQuantization(format!("无效的量化元数据: {e}")))
    }

    /// 验证量化配置的完整性
    pub fn validate(&self) -> Result<()> {
        if ![4, 8].contains(&self.bits) {
            return Err(LoaderError::InvalidQuantization(format!(
                "不支持的量化位宽: {} (仅支持 4 或 8)",
                self.bits
            )));
        }
        if self.block_size == 0 {
            return Err(LoaderError::InvalidQuantization(
                "block_size 不能为 0".to_string(),
            ));
        }
        Ok(())
    }
}

pub mod adapter;
pub mod config;
pub mod downloader;
pub mod format_detector;
pub mod gguf;
pub mod hf_hub;
pub mod modelscope;
pub mod onnx;
pub mod parallel;
#[cfg(feature = "candle")]
pub mod pytorch;
pub mod safetensors;

pub use downloader::{
    Downloader, HfHubDownloader, ModelScopeDownloader, NoProgress, ProgressBar, ProgressCallback,
};
pub use gguf::GgufLoader;
pub use hf_hub::{HfHubClient, HfModelFiles, WeightFormat};
pub use modelscope::{ModelScopeClient, MsModelFiles};
pub use onnx::OnnxLoader;
pub use parallel::ParallelLoader;
#[cfg(feature = "candle")]
pub use pytorch::convert_bins_to_safetensors;
#[cfg(feature = "candle")]
pub use pytorch::{PytorchConversionConfig, PytorchConversionOutput};
pub use safetensors::{SafeTensorsLoader, TensorSlice};

pub type Result<T> = std::result::Result<T, LoaderError>;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("missing weights")]
    MissingWeights,
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("duplicate tensor name: {0}")]
    DuplicateTensor(String),
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("invalid fused tensor shape for {0}")]
    InvalidFusedShape(String),
    #[error("checksum mismatch for {0}")]
    ChecksumMismatch(String),
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] ::safetensors::SafeTensorError),
    #[error("gguf error: {0}")]
    Gguf(String),
    #[error("invalid quantization: {0}")]
    InvalidQuantization(String),
    #[error("onnx error: {0}")]
    Onnx(String),
    #[cfg(feature = "candle")]
    #[error("pytorch bin error: {0}")]
    Pytorch(String),
    #[error("hf hub error: {0}")]
    HfHub(String),
    #[error("authentication required: {hint}")]
    #[non_exhaustive]
    AuthenticationError { hint: String },
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("home directory not available")]
    HomeDirUnavailable,
    #[error("weights format not supported: {0:?}")]
    UnsupportedWeights(WeightFormat),
    #[error("weights format not found: {0:?}")]
    FormatNotFound(WeightFormat),
    #[error("multiple weight formats detected: {0:?}")]
    MultipleWeightFormats(Vec<WeightFormat>),
    #[error("unsupported weight extension: {0}")]
    UnsupportedWeightExtension(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumPolicy {
    VerifyOnLoad,
    StoreIfMissing,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelPolicy {
    Auto,
    Force,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelSource {
    #[default]
    HuggingFace,
    ModelScope,
}

impl ModelSource {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "hf" | "huggingface" | "h" => Some(ModelSource::HuggingFace),
            "modelscope" | "ms" | "m" => Some(ModelSource::ModelScope),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ModelSource::HuggingFace => "HuggingFace",
            ModelSource::ModelScope => "ModelScope",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoaderConfig {
    pub cache_dir: Option<PathBuf>,
    /// 覆盖 HuggingFace token 文件路径（默认使用 ~/.huggingface/token）
    pub hf_token_path: Option<PathBuf>,
    pub checksum: ChecksumPolicy,
    pub parallel: ParallelPolicy,
    pub source: ModelSource,
    /// 当 HF 下载失败时是否自动回退到 ModelScope
    pub enable_fallback: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            hf_token_path: None,
            checksum: ChecksumPolicy::VerifyOnLoad,
            parallel: ParallelPolicy::Auto,
            source: ModelSource::HuggingFace,
            enable_fallback: true, // 默认启用 HF → ModelScope 回退
        }
    }
}

impl LoaderConfig {
    pub fn from_env() -> Self {
        Self::default()
    }
}

/// 模型缓存目录: `~/.gllm/models/`
///
/// 内部子目录结构由下载库（hf-hub/ModelScope）管理
#[derive(Debug, Clone)]
pub struct CacheLayout {
    root: PathBuf,
    checksum_db: PathBuf,
}

impl CacheLayout {
    pub fn new(root: Option<PathBuf>) -> Result<Self> {
        // 支持环境变量 GLLM_CACHE_DIR 自定义路径
        let base = if let Ok(custom) = std::env::var("GLLM_CACHE_DIR") {
            PathBuf::from(custom)
        } else if let Some(path) = root {
            path
        } else {
            let mut base = dirs::home_dir().ok_or(LoaderError::HomeDirUnavailable)?;
            base.push(".gllm");
            base.push("models");
            base
        };

        let checksum_db = base.join("checksums.json");

        Ok(Self {
            root: base,
            checksum_db,
        })
    }

    pub fn ensure(&self) -> Result<()> {
        // 创建根目录和源头隔离子目录（ARCH-MODEL-CACHE-001）
        std::fs::create_dir_all(self.root.join("hf"))?;
        std::fs::create_dir_all(self.root.join("ms"))?;
        Ok(())
    }

    /// 获取 HuggingFace 下载缓存目录
    ///
    /// 返回 `root/hf/`，实现源头隔离（ARCH-MODEL-CACHE-001）
    pub fn hf_cache_dir(&self) -> PathBuf {
        self.root.join("hf")
    }

    /// 获取 ModelScope 下载缓存目录
    ///
    /// 返回 `root/ms/`，实现源头隔离（ARCH-MODEL-CACHE-001）
    pub fn modelscope_cache_dir(&self) -> PathBuf {
        self.root.join("ms")
    }

    pub fn checksum_db(&self) -> &Path {
        &self.checksum_db
    }

    /// 获取模型根目录（用于材料化模型文件）
    pub fn models_dir(&self) -> &Path {
        &self.root
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChecksumRecord {
    sha256: String,
    size: u64,
}

#[derive(Debug, Default, Clone)]
struct ChecksumStore {
    records: HashMap<String, ChecksumRecord>,
    path: PathBuf,
}

impl ChecksumStore {
    fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let bytes = std::fs::read(path)?;
            let records: HashMap<String, ChecksumRecord> = serde_json::from_slice(&bytes)?;
            Ok(Self {
                records,
                path: path.to_path_buf(),
            })
        } else {
            Ok(Self {
                records: HashMap::new(),
                path: path.to_path_buf(),
            })
        }
    }

    fn save(&self) -> Result<()> {
        let data = serde_json::to_vec_pretty(&self.records)?;
        std::fs::write(&self.path, data)?;
        Ok(())
    }

    fn verify_or_store(&mut self, path: &Path, policy: ChecksumPolicy) -> Result<()> {
        if matches!(policy, ChecksumPolicy::Disabled) {
            return Ok(());
        }
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let key = canonical.to_string_lossy().to_string();
        let size = std::fs::metadata(&canonical)?.len();

        if let Some(record) = self.records.get(&key) {
            if matches!(policy, ChecksumPolicy::StoreIfMissing) {
                return Ok(());
            }
            let sha256 = hash_file(&canonical)?;
            if sha256 != record.sha256 || size != record.size {
                return Err(LoaderError::ChecksumMismatch(key));
            }
            return Ok(());
        }

        let sha256 = hash_file(&canonical)?;
        self.records.insert(key, ChecksumRecord { sha256, size });
        Ok(())
    }
}

#[derive(Debug)]
pub struct Loader {
    manifest: Option<Arc<ModelManifest>>,
    repo: String,
    source: ModelSource,
    config: LoaderConfig,
    #[allow(dead_code)]
    cache: CacheLayout,
    #[allow(dead_code)]
    hf: HfHubClient,
    files: HfModelFiles,
    safetensors: Option<SafeTensorsLoader>,
    gguf: Option<GgufLoader>,
    onnx: Option<OnnxLoader>,
    tie_word_embeddings_hint: Option<bool>,
}

impl Loader {
    /// 从远程仓库加载模型（主入口）
    ///
    /// # 参数
    ///
    /// * `repo_model` - 仓库/模型标识符，格式为 `"org/model"`
    ///   - 例如: `"HuggingFaceTB/SmolLM2-135M-Instruct"`
    ///   - 例如: `"intfloat/e5-small"`
    ///   - 例如: `"BAAI/bge-reranker-v2-m3"`
    ///
    /// # 行为
    ///
    /// 1. **自动源选择**: 优先 HuggingFace，失败自动回退 ModelScope
    /// 2. **自动格式探测**: 自动识别 SafeTensors/GGUF/ONNX
    /// 3. **缓存复用**: 已下载的模型直接使用缓存
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use gllm::loader::Loader;
    ///
    /// // 直接加载，自动处理一切
    /// let loader = Loader::from("HuggingFaceTB/SmolLM2-135M-Instruct")?;
    ///
    /// // GGUF 格式也支持
    /// let loader = Loader::from("Mungert/SmolLM2-135M-Instruct-GGUF")?;
    ///
    /// // ONNX 格式也支持
    /// let loader = Loader::from("onnx-community/SmolLM2-135M-ONNX")?;
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # 源回退策略
    ///
    /// - 首选: HuggingFace (全球可用)
    /// - 回退: ModelScope (中国优化，某些模型更完整)
    ///
    /// 当 HuggingFace 下载失败时（404、网络问题等），自动尝试 ModelScope。
    pub fn from(repo_model: &str) -> Result<Self> {
        let mut config = LoaderConfig::from_env();
        // 默认启用 HF → ModelScope 回退
        config.enable_fallback = true;
        config.source = ModelSource::HuggingFace;
        Self::from_source_with_config(repo_model, config)
    }

    /// 从 HuggingFace 加载模型（不自动回退）
    ///
    /// # 已弃用
    ///
    /// 请使用 `Loader::from()` 代替，它支持自动回退。
    ///
    /// 如需强制使用 HuggingFace（不回退），使用：
    /// ```ignore
    /// let loader = Loader::from_with_config(
    ///     "org/model",
    ///     LoaderConfig { source: ModelSource::HuggingFace, enable_fallback: false, ..Default::default() }
    /// )?;
    /// ```
    #[deprecated(since = "0.11.0", note = "请使用 Loader::from() 代替")]
    pub fn from_hf(repo_or_alias: &str) -> Result<Self> {
        let config = LoaderConfig {
            source: ModelSource::HuggingFace,
            enable_fallback: false,
            ..Default::default()
        };
        Self::from_source_with_config(repo_or_alias, config)
    }

    /// 从 ModelScope（魔搭社区）加载模型
    ///
    /// # 已弃用
    ///
    /// 请使用 `Loader::from()` 代替，它会自动回退到 ModelScope。
    ///
    /// 如需强制使用 ModelScope，使用：
    /// ```ignore
    /// let loader = Loader::from_with_config(
    ///     "org/model",
    ///     LoaderConfig { source: ModelSource::ModelScope, enable_fallback: false, ..Default::default() }
    /// )?;
    /// ```
    #[deprecated(since = "0.11.0", note = "请使用 Loader::from() 代替")]
    pub fn from_ms(repo_or_alias: &str) -> Result<Self> {
        let config = LoaderConfig {
            source: ModelSource::ModelScope,
            enable_fallback: false,
            ..Default::default()
        };
        Self::from_source_with_config(repo_or_alias, config)
    }

    pub fn from_env(repo_or_alias: &str) -> Result<Self> {
        let config = LoaderConfig::from_env();
        Self::from_source_with_config(repo_or_alias, config)
    }

    pub fn from_env_with_manifest(
        repo_or_alias: &str,
        manifest: Option<&ModelManifest>,
    ) -> Result<Self> {
        let config = LoaderConfig::from_env();
        Self::from_source_with_config_and_manifest(repo_or_alias, config, manifest)
    }

    pub fn from_source(repo_or_alias: &str, source: ModelSource) -> Result<Self> {
        let config = LoaderConfig {
            source,
            ..Default::default()
        };
        Self::from_source_with_config(repo_or_alias, config)
    }

    /// 从 HuggingFace 加载，失败时自动回退到 ModelScope
    ///
    /// # 已弃用
    ///
    /// 请使用 `Loader::from()` 代替，默认已启用自动回退。
    #[deprecated(
        since = "0.11.0",
        note = "请使用 Loader::from() 代替，默认已启用自动回退"
    )]
    pub fn from_hf_with_fallback(repo_or_alias: &str) -> Result<Self> {
        Self::from(repo_or_alias)
    }

    /// 从 HuggingFace 加载（带自定义配置）
    ///
    /// # 已弃用
    ///
    /// 请使用 `Loader::from_with_config()` 代替。
    #[deprecated(since = "0.11.0", note = "请使用 Loader::from_with_config() 代替")]
    pub fn from_hf_with_config(repo_or_alias: &str, mut config: LoaderConfig) -> Result<Self> {
        config.source = ModelSource::HuggingFace;
        config.enable_fallback = false;
        Self::from_source_with_config(repo_or_alias, config)
    }

    /// 加载模型（带自定义配置）
    ///
    /// # 参数
    ///
    /// * `repo_model` - 仓库/模型标识符，格式为 `"org/model"`
    /// * `config` - 自定义配置
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use gllm::loader::Loader;
    /// use gllm::loader::LoaderConfig;
    ///
    /// // 自定义缓存目录
    /// let loader = Loader::from_with_config(
    ///     "HuggingFaceTB/SmolLM2-135M-Instruct",
    ///     LoaderConfig { cache_dir: Some("/custom/cache".into()), ..Default::default() }
    /// )?;
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_with_config(repo_model: &str, mut config: LoaderConfig) -> Result<Self> {
        config.enable_fallback = true;
        config.source = ModelSource::HuggingFace;
        Self::from_source_with_config(repo_model, config)
    }

    /// Create a loader from local files without downloading.
    pub fn from_local_files(
        repo_or_alias: &str,
        weights: Vec<PathBuf>,
        aux_files: Vec<PathBuf>,
    ) -> Result<Self> {
        Self::from_local_files_with_manifest(repo_or_alias, weights, aux_files, None)
    }

    /// Create a loader from local files with an explicit manifest.
    /// This is useful when the manifest is already known (e.g., from config parsing).
    pub fn from_local_files_with_manifest(
        repo_or_alias: &str,
        weights: Vec<PathBuf>,
        aux_files: Vec<PathBuf>,
        manifest: Option<&ModelManifest>,
    ) -> Result<Self> {
        let resolved_manifest = manifest.map(|manifest| Arc::new(manifest.clone()));
        let repo = repo_or_alias.to_string();
        let config = LoaderConfig::default();
        let cache = CacheLayout::new(config.cache_dir.clone())?;
        cache.ensure()?;
        let hf = HfHubClient::with_endpoint_and_token_path(
            cache.hf_cache_dir(),
            None,
            config.hf_token_path.clone(),
        )?;
        let format = detect_weight_format(&weights)?;
        Ok(Self {
            manifest: resolved_manifest,
            repo,
            source: ModelSource::HuggingFace,
            config,
            cache,
            hf,
            files: HfModelFiles {
                repo: repo_or_alias.to_string(),
                weights,
                format,
                aux_files,
            },
            safetensors: None,
            gguf: None,
            onnx: None,
            tie_word_embeddings_hint: None,
        })
    }

    pub fn from_source_with_config(repo_or_alias: &str, config: LoaderConfig) -> Result<Self> {
        Self::from_source_with_config_and_manifest(repo_or_alias, config, None)
    }

    pub fn from_source_with_config_and_manifest(
        repo_or_alias: &str,
        config: LoaderConfig,
        manifest: Option<&ModelManifest>,
    ) -> Result<Self> {
        Self::from_source_with_config_and_manifest_and_format(repo_or_alias, config, manifest, None)
    }

    /// 自动加载模型（`from()` 的别名）
    ///
    /// # 已弃用
    ///
    /// 请使用 `Loader::from()` 代替，API 更简洁清晰。
    #[deprecated(since = "0.11.0", note = "请使用 Loader::from() 代替")]
    pub fn auto(repo_or_alias: &str) -> Result<Self> {
        Self::from(repo_or_alias)
    }

    pub fn auto_with_format(repo_or_alias: &str, format: WeightFormat) -> Result<Self> {
        let config = LoaderConfig::from_env();
        Self::auto_with_config(repo_or_alias, config, Some(format))
    }

    pub fn auto_with_source(repo_or_alias: &str, source: ModelSource) -> Result<Self> {
        let mut config = LoaderConfig::from_env();
        config.source = source;
        Self::auto_with_config(repo_or_alias, config, None)
    }

    fn auto_with_config(
        repo_or_alias: &str,
        config: LoaderConfig,
        format_hint: Option<WeightFormat>,
    ) -> Result<Self> {
        let path = Path::new(repo_or_alias);
        if path.exists() {
            let local = format_detector::collect_local_files(path, format_hint)?;
            return Self::from_local_files_with_manifest(
                repo_or_alias,
                local.weights,
                local.aux_files,
                None,
            );
        }
        Self::from_source_with_config_and_manifest_and_format(
            repo_or_alias,
            config,
            None,
            format_hint,
        )
    }

    fn from_source_with_config_and_manifest_and_format(
        repo_or_alias: &str,
        config: LoaderConfig,
        manifest: Option<&ModelManifest>,
        format_hint: Option<WeightFormat>,
    ) -> Result<Self> {
        let should_try_fallback = config.enable_fallback;
        let result =
            Self::load_from_source_with_format(repo_or_alias, &config, manifest, format_hint);

        if should_try_fallback {
            result.or_else(|err| {
                if is_recoverable_error(&err) {
                    let fallback = fallback_source(config.source);
                    eprintln!(
                        "⚠️  {} 下载失败，尝试 {}...",
                        config.source.label(),
                        fallback.label()
                    );
                    let mut fallback_config = config.clone();
                    fallback_config.source = fallback;
                    fallback_config.enable_fallback = false;
                    Self::load_from_source_with_format(
                        repo_or_alias,
                        &fallback_config,
                        manifest,
                        format_hint,
                    )
                } else {
                    Err(err)
                }
            })
        } else {
            result
        }
    }

    fn load_from_source_with_format(
        repo_or_alias: &str,
        config: &LoaderConfig,
        manifest: Option<&ModelManifest>,
        format_hint: Option<WeightFormat>,
    ) -> Result<Self> {
        let manifest = manifest.map(|manifest| Arc::new(manifest.clone()));
        let file_map = manifest
            .as_ref()
            .map(|m| m.file_map)
            .unwrap_or(EMPTY_FILE_MAP);
        let repo = resolve_repo(repo_or_alias);
        let cache = CacheLayout::new(config.cache_dir.clone())?;
        cache.ensure()?;

        let parallel_download = ParallelLoader::new(match config.parallel {
            ParallelPolicy::Force => true,
            ParallelPolicy::Disabled => false,
            ParallelPolicy::Auto => manifest.as_ref().map(|m| m.is_moe()).unwrap_or(false),
        });

        // 根据源类型选择下载方式
        let files = if config.source == ModelSource::ModelScope {
            // 使用 ModelScope 客户端，使用专门的 ModelScope 缓存目录
            let ms_cache = cache.modelscope_cache_dir();
            let ms_client = ModelScopeClient::new(ms_cache)?;
            let ms_files = ms_client
                .download_model_files_with_format(&repo, file_map, parallel_download, format_hint)
                .map_err(|e| LoaderError::HfHub(format!("ModelScope download failed: {}", e)))?;

            // 转换 MsModelFiles 为 HfModelFiles
            HfModelFiles {
                repo: ms_files.repo,
                weights: ms_files.weights,
                format: ms_files.format,
                aux_files: ms_files.aux_files,
            }
        } else {
            // 使用 HuggingFace 客户端
            let hf = HfHubClient::with_endpoint_and_token_path(
                cache.hf_cache_dir(),
                None,
                config.hf_token_path.clone(),
            )?;
            hf.download_model_files_with_format(&repo, file_map, parallel_download, format_hint)?
        };

        let mut checksum_store = ChecksumStore::load(cache.checksum_db())?;
        for path in &files.weights {
            checksum_store.verify_or_store(path, config.checksum)?;
        }
        checksum_store.save()?;

        let mut all_files = files.weights.clone();
        all_files.extend(files.aux_files.iter().cloned());
        if !all_files.is_empty() {
            materialize_model_dir(cache.models_dir(), &repo, &all_files)?;
        }

        // 创建 HfHubClient（用于后续操作）
        let hf = HfHubClient::with_endpoint_and_token_path(
            cache.hf_cache_dir(),
            None,
            config.hf_token_path.clone(),
        )?;

        Ok(Self {
            manifest,
            repo,
            source: config.source,
            config: config.clone(),
            cache,
            hf,
            files,
            safetensors: None,
            gguf: None,
            onnx: None,
            tie_word_embeddings_hint: None,
        })
    }

    pub fn repo(&self) -> &str {
        &self.repo
    }

    pub fn source(&self) -> ModelSource {
        self.source
    }

    pub fn manifest(&self) -> Option<&ModelManifest> {
        self.manifest.as_deref()
    }

    pub fn set_manifest_if_missing(&mut self, manifest: &ModelManifest) {
        if self.manifest.is_none() {
            self.manifest = Some(Arc::new(manifest.clone()));
        }
    }

    pub fn set_tie_word_embeddings_hint(&mut self, tie_word_embeddings: Option<bool>) {
        self.tie_word_embeddings_hint = tie_word_embeddings;
    }

    pub fn weight_format(&self) -> WeightFormat {
        self.files.format
    }

    pub fn aux_files(&self) -> &[PathBuf] {
        &self.files.aux_files
    }

    pub fn find_aux_file(&self, filename: &str) -> Option<&Path> {
        self.files.aux_files.iter().find_map(|path| {
            if path.file_name().is_some_and(|name| name == filename) {
                Some(path.as_path())
            } else {
                None
            }
        })
    }

    pub fn find_aux_file_any(&self, filenames: &[&str]) -> Option<&Path> {
        for &name in filenames {
            if let Some(path) = self.find_aux_file(name) {
                return Some(path);
            }
        }
        None
    }

    pub fn config_path(&self) -> Option<&Path> {
        self.find_aux_file_any(&["config.json", "configuration.json"])
    }

    pub fn tokenizer_path(&self) -> Option<&Path> {
        self.find_aux_file("tokenizer.json")
    }

    pub fn ensure_safetensors(&mut self) -> Result<()> {
        if self.safetensors.is_some() {
            return Ok(());
        }
        if self.files.format != WeightFormat::SafeTensors {
            return Err(LoaderError::UnsupportedWeights(self.files.format));
        }

        let is_moe = self.manifest.as_ref().map(|m| m.is_moe()).unwrap_or(false);
        let parallel_enabled = match self.config.parallel {
            ParallelPolicy::Force => true,
            ParallelPolicy::Disabled => false,
            ParallelPolicy::Auto => is_moe || self.files.weights.len() > 1,
        };
        let parallel_enabled = parallel::enforce_parallel(is_moe, parallel_enabled);
        let parallel_loader = ParallelLoader::new(parallel_enabled);

        let safetensors = SafeTensorsLoader::from_files(&self.files.weights, parallel_loader)?;
        self.safetensors = Some(safetensors);
        Ok(())
    }

    pub fn ensure_gguf(&mut self) -> Result<()> {
        if self.gguf.is_some() {
            return Ok(());
        }
        if self.files.format != WeightFormat::Gguf {
            return Err(LoaderError::UnsupportedWeights(self.files.format));
        }
        let gguf = GgufLoader::from_files(&self.files.weights)
            .map_err(|err| LoaderError::Gguf(err.to_string()))?;
        self.gguf = Some(gguf);
        Ok(())
    }

    pub fn ensure_onnx(&mut self) -> Result<()> {
        if self.onnx.is_some() {
            return Ok(());
        }
        if self.files.format != WeightFormat::Onnx {
            return Err(LoaderError::UnsupportedWeights(self.files.format));
        }
        if self.files.weights.len() != 1 {
            return Err(LoaderError::Onnx(
                "onnx loader expects a single weight file".into(),
            ));
        }
        let path = self
            .files
            .weights
            .first()
            .ok_or(LoaderError::MissingWeights)?;
        let onnx = OnnxLoader::from_path(path)?;
        self.onnx = Some(onnx);
        Ok(())
    }

    pub fn onnx(&mut self) -> Result<&OnnxLoader> {
        self.ensure_onnx()?;
        self.onnx.as_ref().ok_or(LoaderError::MissingWeights)
    }

    pub fn gguf_architecture_name(&mut self) -> Result<Option<String>> {
        self.ensure_gguf()?;
        let loader = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;
        Ok(loader.architecture_name().map(|name| name.to_string()))
    }

    pub fn gguf_architecture(&mut self) -> Result<crate::manifest::ModelArchitecture> {
        self.ensure_gguf()?;
        let loader = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;
        let architecture = loader
            .architecture()
            .map_err(|err| LoaderError::Gguf(err.to_string()))?;
        map_gguf_architecture(architecture).ok_or_else(|| {
            LoaderError::Gguf(format!(
                "unsupported GGUF architecture metadata: {architecture}"
            ))
        })
    }

    pub fn gguf_quantization_version(&mut self) -> Result<Option<u64>> {
        self.ensure_gguf()?;
        let loader = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;
        Ok(loader.quantization_version())
    }

    pub fn gguf_quantization_types(&mut self) -> Result<Vec<String>> {
        self.ensure_gguf()?;
        let loader = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;
        Ok(loader.quantization_types().to_vec())
    }

    pub fn gguf_reader(&mut self) -> Result<&GgufLoader> {
        self.ensure_gguf()?;
        self.gguf.as_ref().ok_or(LoaderError::MissingWeights)
    }

    pub fn safetensors_gllm_config(&mut self) -> Result<Option<&Value>> {
        if self.files.format != WeightFormat::SafeTensors {
            return Ok(None);
        }
        self.ensure_safetensors()?;
        let loader = self
            .safetensors
            .as_ref()
            .ok_or(LoaderError::MissingWeights)?;
        Ok(loader.gllm_config())
    }

    pub fn safetensors_gllm_tokenizer_config(&mut self) -> Result<Option<&Value>> {
        if self.files.format != WeightFormat::SafeTensors {
            return Ok(None);
        }
        self.ensure_safetensors()?;
        let loader = self
            .safetensors
            .as_ref()
            .ok_or(LoaderError::MissingWeights)?;
        Ok(loader.gllm_tokenizer_config())
    }

    pub fn onnx_tensor_dtype(&mut self, name: &str) -> Result<Dtype> {
        self.ensure_onnx()?;
        let loader = self.onnx.as_ref().ok_or(LoaderError::MissingWeights)?;
        loader.tensor_dtype(name)
    }

    pub fn onnx_precisions(&mut self) -> Result<Vec<Dtype>> {
        self.ensure_onnx()?;
        let loader = self.onnx.as_ref().ok_or(LoaderError::MissingWeights)?;
        Ok(loader.unique_precisions())
    }

    /// Ω1: 从实际权重张量中检测 dtype 大小
    ///
    /// 对于 SafeTensors 格式，直接读取张量的实际 dtype
    pub fn detect_weight_dtype_size(&mut self) -> Result<Option<usize>> {
        if self.files.format == WeightFormat::SafeTensors {
            self.ensure_safetensors()?;
            let loader = self
                .safetensors
                .as_ref()
                .ok_or(LoaderError::MissingWeights)?;
            Ok(loader.detect_weight_dtype_size())
        } else if self.files.format == WeightFormat::Gguf {
            self.ensure_gguf()?;
            let loader = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;
            Ok(loader.floating_point_dtype_size())
        } else if self.files.format == WeightFormat::Onnx {
            self.ensure_onnx()?;
            let loader = self.onnx.as_ref().ok_or(LoaderError::MissingWeights)?;
            Ok(loader
                .unique_precisions()
                .into_iter()
                .filter_map(onnx_dtype_size)
                .min())
        } else {
            Ok(None)
        }
    }

    pub fn upload_weights<B: Backend>(&mut self, backend: &B) -> Result<WeightsHandle<B>> {
        match self.files.format {
            WeightFormat::SafeTensors => self.upload_safetensors(backend),
            WeightFormat::Gguf => self.upload_gguf(backend),
            WeightFormat::Onnx => Err(LoaderError::UnsupportedWeights(self.files.format)),
        }
    }

    fn upload_safetensors<B: Backend>(&mut self, backend: &B) -> Result<WeightsHandle<B>> {
        self.ensure_safetensors()?;
        let loader = self
            .safetensors
            .as_ref()
            .ok_or(LoaderError::MissingWeights)?;
        let quantized = QuantizedIndex::from_loader(loader)?;
        let mut quantized_seen = HashSet::new();

        let is_moe = self.manifest.as_ref().map(|m| m.is_moe()).unwrap_or(false);
        let parallel_enabled = match self.config.parallel {
            ParallelPolicy::Force => true,
            ParallelPolicy::Disabled => false,
            ParallelPolicy::Auto => is_moe || self.files.weights.len() > 1,
        };
        let parallel_enabled = parallel::enforce_parallel(is_moe, parallel_enabled);
        loader.prefetch_parallel(ParallelLoader::new(parallel_enabled))?;

        let mut handle = WeightsHandle::default();
        let mut visited = HashSet::new();
        let tensor_names = loader.names();
        for name in tensor_names {
            if !visited.insert(name.clone()) {
                continue;
            }
            if let Some(group) = quantized.group_for(&name) {
                if quantized_seen.insert(group.base_name.clone()) {
                    let owned = group.dequantize(loader)?;
                    for output in maybe_split_fused_owned(self.rules(), owned) {
                        upload_owned_tensor(backend, &mut handle, output, None)?;
                    }
                }
                continue;
            }
            let tensor = loader.tensor(&name)?;
            if let Some(outputs) = maybe_split_fused(self.rules(), &name, &tensor) {
                for output in outputs {
                    upload_owned_tensor(backend, &mut handle, output, None)?;
                }
                continue;
            }

            upload_tensor_slice(backend, &mut handle, &name, tensor)?;
        }

        if self.should_materialize_tied_lm_head() {
            materialize_tied_lm_head_from_safetensors(backend, &mut handle, loader)?;
        }

        Ok(handle)
    }

    fn upload_gguf<B: Backend>(&mut self, backend: &B) -> Result<WeightsHandle<B>> {
        use gllm_kernels::{dequantize_q4_0, dequantize_q8_0, Q4_0Matrix, Q8_0Matrix};

        self.ensure_gguf()?;
        let loader = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;

        let mut handle = WeightsHandle::default();
        let mut visited = HashSet::new();
        let tensor_names = loader.names();

        for name in tensor_names {
            if !visited.insert(name.clone()) {
                continue;
            }
            let tensor = loader
                .tensor(&name)
                .map_err(|err| LoaderError::Gguf(err.to_string()))?;
            let shape = gguf_shape_to_usize(tensor.shape())?;
            let qtype = match tensor.dtype() {
                gguf::GgmlDType::Q4_0 => Some(gllm_kernels::QuantizedType::Q4_0),
                gguf::GgmlDType::Q8_0 => Some(gllm_kernels::QuantizedType::Q8_0),
                gguf::GgmlDType::Q5_K => Some(gllm_kernels::QuantizedType::Q5_K),
                _ => None,
            };

            if let Some(qtype) = qtype {
                let owned = match qtype {
                    gllm_kernels::QuantizedType::Q4_0 => {
                        let (rows, cols) = gguf_matrix_dims(&shape)?;
                        let blocks = parse_q4_0_blocks(tensor.as_bytes(), rows, cols)?;
                        let matrix = Q4_0Matrix { blocks, rows, cols };
                        let total = rows.checked_mul(cols).ok_or_else(|| {
                            LoaderError::InvalidQuantization("gguf output overflow".into())
                        })?;
                        let mut out = vec![0.0f32; total];
                        dequantize_q4_0(&matrix, &mut out)
                            .map_err(|err| LoaderError::Backend(format!("{err:?}")))?;
                        OwnedTensor::F32 {
                            name: name.clone(),
                            shape: shape.clone(),
                            data: out,
                        }
                    }
                    gllm_kernels::QuantizedType::Q8_0 => {
                        let (rows, cols) = gguf_matrix_dims(&shape)?;
                        let blocks = parse_q8_0_blocks(tensor.as_bytes(), rows, cols)?;
                        let matrix = Q8_0Matrix { blocks, rows, cols };
                        let total = rows.checked_mul(cols).ok_or_else(|| {
                            LoaderError::InvalidQuantization("gguf output overflow".into())
                        })?;
                        let mut out = vec![0.0f32; total];
                        dequantize_q8_0(&matrix, &mut out)
                            .map_err(|err| LoaderError::Backend(format!("{err:?}")))?;
                        OwnedTensor::F32 {
                            name: name.clone(),
                            shape: shape.clone(),
                            data: out,
                        }
                    }
                    gllm_kernels::QuantizedType::Q5_K => {
                        return Err(LoaderError::InvalidQuantization(
                            "q5_k gguf tensors are not supported yet".into(),
                        ));
                    }
                };

                for output in maybe_split_fused_owned(self.rules(), owned) {
                    upload_owned_tensor(backend, &mut handle, output, Some(qtype))?;
                }
                continue;
            }

            let data = gguf_tensor_to_f32(&tensor)?;
            let owned = OwnedTensor::F32 {
                name: name.clone(),
                shape,
                data,
            };
            for output in maybe_split_fused_owned(self.rules(), owned) {
                upload_owned_tensor(backend, &mut handle, output, None)?;
            }
        }

        if self.should_materialize_tied_lm_head() {
            materialize_tied_lm_head_from_gguf(backend, &mut handle, loader)?;
        }

        Ok(handle)
    }

    fn rules(&self) -> Option<TensorNamingRule> {
        self.manifest.as_ref().map(|m| m.tensor_rules)
    }

    fn should_materialize_tied_lm_head(&self) -> bool {
        self.tie_word_embeddings_hint == Some(true)
            && self
                .manifest
                .as_ref()
                .map(|m| m.kind == ModelKind::Chat)
                .unwrap_or(true)
    }
}

fn resolve_repo(repo_or_alias: &str) -> String {
    repo_or_alias.to_string()
}

fn normalize_architecture_token(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', '.'], "_")
}

fn map_gguf_architecture(value: &str) -> Option<crate::manifest::ModelArchitecture> {
    match normalize_architecture_token(value).as_str() {
        "llama" => Some(crate::manifest::ModelArchitecture::Llama4),
        "qwen2" | "qwen2_5" => Some(crate::manifest::ModelArchitecture::Qwen2_5),
        "qwen3" => Some(crate::manifest::ModelArchitecture::Qwen3),
        "deepseek" => Some(crate::manifest::ModelArchitecture::Qwen3MoE),
        "mistral" => Some(crate::manifest::ModelArchitecture::Mistral3),
        "ministral" => Some(crate::manifest::ModelArchitecture::Ministral),
        "gemma" | "gemma2" => Some(crate::manifest::ModelArchitecture::Gemma2),
        _ => None,
    }
}

fn fallback_source(source: ModelSource) -> ModelSource {
    match source {
        ModelSource::HuggingFace => ModelSource::ModelScope,
        ModelSource::ModelScope => ModelSource::HuggingFace,
    }
}

fn onnx_dtype_size(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::F32 => Some(4),
        Dtype::F16 | Dtype::BF16 => Some(2),
        Dtype::F64 => Some(8),
        Dtype::I8 | Dtype::U8 => Some(1),
        Dtype::I16 | Dtype::U16 => Some(2),
        Dtype::I32 | Dtype::U32 => Some(4),
        Dtype::I64 | Dtype::U64 => Some(8),
        Dtype::BOOL => Some(1),
        _ => None,
    }
}

fn detect_weight_format(weights: &[PathBuf]) -> Result<WeightFormat> {
    format_detector::detect_format_from_paths(weights)
}

/// 从远程仓库加载模型（便捷函数）
///
/// 这是 `Loader::from()` 的便捷包装，提供更简洁的调用方式。
///
/// # 参数
///
/// * `repo_model` - 仓库/模型标识符，格式为 `"org/model"`
///
/// # 示例
///
/// ```no_run
/// use gllm::loader;
///
/// // 简洁调用
/// let loader = loader::from("HuggingFaceTB/SmolLM2-135M-Instruct")?;
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn from(repo_model: &str) -> Result<Loader> {
    Loader::from(repo_model)
}

/// 从 HuggingFace 加载模型
///
/// # 已弃用
///
/// 请使用 `from()` 代替。
#[deprecated(since = "0.11.0", note = "请使用 from() 代替")]
pub fn from_hf(repo_or_alias: &str) -> Result<Loader> {
    Loader::from(repo_or_alias)
}

/// Convenience: create a loader from ModelScope (魔搭社区).
///
/// # 已弃用
///
/// 请使用 `gllm::loader::Loader::from()` 代替，自动回退已包含。
#[deprecated(since = "0.11.0", note = "请使用 gllm::loader::Loader::from() 代替")]
pub fn from_ms(repo_or_alias: &str) -> Result<Loader> {
    Loader::from(repo_or_alias)
}

/// Convenience: create a loader with automatic HF->MS fallback.
///
/// # 已弃用
///
/// 请使用 `gllm::loader::Loader::from()` 代替，默认已启用自动回退。
#[deprecated(since = "0.11.0", note = "请使用 gllm::loader::Loader::from() 代替")]
pub fn from_hf_with_fallback(repo_or_alias: &str) -> Result<Loader> {
    Loader::from(repo_or_alias)
}

/// Convenience: create a loader using environment-driven source selection.
pub fn from_env(repo_or_alias: &str) -> Result<Loader> {
    Loader::from_env(repo_or_alias)
}

/// Convenience: create a loader from the selected source.
pub fn from_source(repo_or_alias: &str, source: ModelSource) -> Result<Loader> {
    Loader::from_source(repo_or_alias, source)
}

/// 判断错误是否可以回退到 ModelScope
///
/// 以下情况可以回退：
/// - 认证错误 (401/403)
/// - 权重缺失
/// - 网络超时
fn is_recoverable_error(err: &LoaderError) -> bool {
    match err {
        LoaderError::AuthenticationError { .. } => true,
        LoaderError::MissingWeights => true,
        LoaderError::HfHub(msg) => {
            let msg_lower = msg.to_lowercase();
            msg_lower.contains("401")
                || msg_lower.contains("403")
                || msg_lower.contains("timeout")
                || msg_lower.contains("unauthorized")
                || msg_lower.contains("forbidden")
        }
        _ => false,
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub quantized: Option<gllm_kernels::QuantizedType>,
}

pub enum UploadedTensor<B: Backend> {
    F32(B::Tensor<f32>),
}

pub struct WeightsHandle<B: Backend> {
    pub tensors: HashMap<String, UploadedTensor<B>>,
    pub meta: HashMap<String, TensorInfo>,
}

impl<B: Backend> Default for WeightsHandle<B> {
    fn default() -> Self {
        Self {
            tensors: HashMap::new(),
            meta: HashMap::new(),
        }
    }
}

impl<B: Backend> WeightsHandle<B> {
    pub fn get(&self, name: &str) -> Option<&UploadedTensor<B>> {
        self.tensors.get(name)
    }
}

impl<B: Backend> TensorLookup<B> for WeightsHandle<B> {
    fn tensor_f32(&self, name: &str) -> Option<&B::Tensor<f32>> {
        match self.tensors.get(name)? {
            UploadedTensor::F32(tensor) => Some(tensor),
        }
    }

    fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.meta.get(name).map(|info| info.shape.as_slice())
    }
}

enum OwnedTensor {
    F16 {
        name: String,
        shape: Vec<usize>,
        data: Vec<f16>,
    },
    BF16 {
        name: String,
        shape: Vec<usize>,
        data: Vec<bf16>,
    },
    F32 {
        name: String,
        shape: Vec<usize>,
        data: Vec<f32>,
    },
}

const EMBEDDING_WEIGHT_NAMES: &[&str] = &[
    "model.embed_tokens.weight",
    "tok_embeddings.weight",
    "transformer.wte.weight",
    "model.tok_embeddings.weight",
    "embeddings.word_embeddings.weight",
    "model.embeddings.word_embeddings.weight",
    "roberta.embeddings.word_embeddings.weight",
    "token_embd.weight",
];

const LM_HEAD_WEIGHT_NAMES: &[&str] = &[
    "lm_head.weight",
    "model.lm_head.weight",
    "embed_out.weight",
    "model.embed_out.weight",
    "transformer.lm_head.weight",
];

const TIED_LM_HEAD_ALIAS: &str = "lm_head.weight";

fn has_any_tensor<B: Backend>(handle: &WeightsHandle<B>, names: &[&str]) -> bool {
    names.iter().any(|name| handle.tensors.contains_key(*name))
}

fn materialize_tied_lm_head_from_safetensors<B: Backend>(
    backend: &B,
    handle: &mut WeightsHandle<B>,
    loader: &SafeTensorsLoader,
) -> Result<()> {
    if has_any_tensor(handle, LM_HEAD_WEIGHT_NAMES) {
        return Ok(());
    }

    for name in EMBEDDING_WEIGHT_NAMES {
        let Ok(tensor) = loader.tensor(name) else {
            continue;
        };
        upload_tensor_slice(backend, handle, TIED_LM_HEAD_ALIAS, tensor)?;
        return Ok(());
    }

    Err(LoaderError::MissingTensor(
        "tie_word_embeddings=true but no embedding tensor was found to materialize lm_head.weight"
            .to_string(),
    ))
}

fn materialize_tied_lm_head_from_gguf<B: Backend>(
    backend: &B,
    handle: &mut WeightsHandle<B>,
    loader: &GgufLoader,
) -> Result<()> {
    if has_any_tensor(handle, LM_HEAD_WEIGHT_NAMES) {
        return Ok(());
    }

    for name in EMBEDDING_WEIGHT_NAMES {
        let tensor = match loader.tensor(name) {
            Ok(tensor) => tensor,
            Err(_) => continue,
        };
        let shape = gguf_shape_to_usize(tensor.shape())?;
        let quantized = match tensor.dtype() {
            gguf::GgmlDType::Q4_0 => Some(gllm_kernels::QuantizedType::Q4_0),
            gguf::GgmlDType::Q8_0 => Some(gllm_kernels::QuantizedType::Q8_0),
            _ => None,
        };
        let data = match quantized {
            Some(gllm_kernels::QuantizedType::Q4_0) => {
                let (rows, cols) = gguf_matrix_dims(&shape)?;
                let blocks = parse_q4_0_blocks(tensor.as_bytes(), rows, cols)?;
                let matrix = gllm_kernels::Q4_0Matrix { blocks, rows, cols };
                let total = rows
                    .checked_mul(cols)
                    .ok_or_else(|| LoaderError::InvalidQuantization("gguf output overflow".into()))?;
                let mut out = vec![0.0f32; total];
                gllm_kernels::dequantize_q4_0(&matrix, &mut out)
                    .map_err(|err| LoaderError::Backend(format!("{err:?}")))?;
                out
            }
            Some(gllm_kernels::QuantizedType::Q8_0) => {
                let (rows, cols) = gguf_matrix_dims(&shape)?;
                let blocks = parse_q8_0_blocks(tensor.as_bytes(), rows, cols)?;
                let matrix = gllm_kernels::Q8_0Matrix { blocks, rows, cols };
                let total = rows
                    .checked_mul(cols)
                    .ok_or_else(|| LoaderError::InvalidQuantization("gguf output overflow".into()))?;
                let mut out = vec![0.0f32; total];
                gllm_kernels::dequantize_q8_0(&matrix, &mut out)
                    .map_err(|err| LoaderError::Backend(format!("{err:?}")))?;
                out
            }
            _ => gguf_tensor_to_f32(&tensor)?,
        };
        let owned = OwnedTensor::F32 {
            name: TIED_LM_HEAD_ALIAS.to_string(),
            shape,
            data,
        };
        upload_owned_tensor(backend, handle, owned, quantized)?;
        return Ok(());
    }

    Err(LoaderError::MissingTensor(
        "tie_word_embeddings=true but no embedding tensor was found to materialize lm_head.weight"
            .to_string(),
    ))
}

fn upload_f32_data<B: Backend>(
    backend: &B,
    handle: &mut WeightsHandle<B>,
    name: &str,
    shape: Vec<usize>,
    data: &[f32],
    quantized: Option<gllm_kernels::QuantizedType>,
) -> Result<()> {
    let uploaded = backend
        .upload_weights(data)
        .map_err(|err| LoaderError::Backend(format!("{err:?}")))?;
    insert_tensor(
        handle,
        name.to_string(),
        shape,
        Dtype::F32,
        quantized,
        UploadedTensor::F32(uploaded),
    )
}

fn upload_tensor_slice<B: Backend>(
    backend: &B,
    handle: &mut WeightsHandle<B>,
    name: &str,
    tensor: TensorSlice<'_>,
) -> Result<()> {
    match tensor.dtype {
        Dtype::F16 => {
            let data = tensor.as_f16()?;
            let converted: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::BF16 => {
            let data = tensor.as_bf16()?;
            let converted: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::F32 => {
            let data = tensor.as_f32()?;
            upload_f32_data(backend, handle, name, tensor.shape, data.as_ref(), None)?;
        }
        Dtype::F64 => {
            let data = tensor.as_f64()?;
            let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::I8 => {
            let data = tensor.as_i8()?;
            let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::U8 => {
            let data = tensor.as_u8()?;
            let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::I16 => {
            let data = tensor.as_i16()?;
            let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::U16 => {
            let data = tensor.as_u16()?;
            let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::I32 => {
            let data = tensor.as_i32()?;
            let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::U32 => {
            let data = tensor.as_u32()?;
            let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::I64 => {
            let data = tensor.as_i64()?;
            let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        Dtype::U64 => {
            let data = tensor.as_u64()?;
            let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            upload_f32_data(backend, handle, name, tensor.shape, &converted, None)?;
        }
        other => return Err(LoaderError::UnsupportedDtype(other)),
    }
    Ok(())
}

fn upload_owned_tensor<B: Backend>(
    backend: &B,
    handle: &mut WeightsHandle<B>,
    tensor: OwnedTensor,
    quantized: Option<gllm_kernels::QuantizedType>,
) -> Result<()> {
    match tensor {
        OwnedTensor::F16 { name, shape, data } => {
            let converted: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
            upload_f32_data(backend, handle, &name, shape, &converted, quantized)
        }
        OwnedTensor::BF16 { name, shape, data } => {
            let converted: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
            upload_f32_data(backend, handle, &name, shape, &converted, quantized)
        }
        OwnedTensor::F32 { name, shape, data } => {
            upload_f32_data(backend, handle, &name, shape, &data, quantized)
        }
    }
}

fn insert_tensor<B: Backend>(
    handle: &mut WeightsHandle<B>,
    name: String,
    shape: Vec<usize>,
    dtype: Dtype,
    quantized: Option<gllm_kernels::QuantizedType>,
    tensor: UploadedTensor<B>,
) -> Result<()> {
    if handle.tensors.contains_key(&name) {
        return Err(LoaderError::DuplicateTensor(name));
    }
    handle.meta.insert(
        name.clone(),
        TensorInfo {
            shape,
            dtype,
            quantized,
        },
    );
    handle.tensors.insert(name, tensor);
    Ok(())
}

fn maybe_split_fused(
    rules: Option<TensorNamingRule>,
    name: &str,
    tensor: &TensorSlice<'_>,
) -> Option<Vec<OwnedTensor>> {
    let rules = rules?;

    // Phi4 GQA 特殊处理: qkv_proj 权重不是等分的
    // Phi-4-mini: Q=3072, K=1024, V=1024, total=5120
    if matches!(rules, TensorNamingRule::Phi4) && name.contains("qkv_proj") {
        return split_phi4_qkv(name, tensor);
    }

    let fused = fused_spec(rules, name, &tensor.shape)?;
    let axis = split_axis(&tensor.shape, fused.split)?;
    let mut out_shape = tensor.shape.clone();
    out_shape[axis] /= fused.split;

    match tensor.dtype {
        Dtype::F16 => {
            let data = tensor.as_f16().ok()?;
            let parts = split_tensor(data.as_ref(), &tensor.shape, axis, fused.split);
            Some(
                fused
                    .targets
                    .into_iter()
                    .zip(parts)
                    .map(|(name, data)| OwnedTensor::F16 {
                        name,
                        shape: out_shape.clone(),
                        data,
                    })
                    .collect(),
            )
        }
        Dtype::BF16 => {
            let data = tensor.as_bf16().ok()?;
            let parts = split_tensor(data.as_ref(), &tensor.shape, axis, fused.split);
            Some(
                fused
                    .targets
                    .into_iter()
                    .zip(parts)
                    .map(|(name, data)| OwnedTensor::BF16 {
                        name,
                        shape: out_shape.clone(),
                        data,
                    })
                    .collect(),
            )
        }
        Dtype::F32 => {
            let data = tensor.as_f32().ok()?;
            let parts = split_tensor(data.as_ref(), &tensor.shape, axis, fused.split);
            Some(
                fused
                    .targets
                    .into_iter()
                    .zip(parts)
                    .map(|(name, data)| OwnedTensor::F32 {
                        name,
                        shape: out_shape.clone(),
                        data,
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

fn maybe_split_fused_owned(
    rules: Option<TensorNamingRule>,
    tensor: OwnedTensor,
) -> Vec<OwnedTensor> {
    let (name, shape, data) = match tensor {
        OwnedTensor::F32 { name, shape, data } => (name, shape, data),
        other => return vec![other],
    };
    let Some(rules) = rules else {
        return vec![OwnedTensor::F32 { name, shape, data }];
    };
    let Some(fused) = fused_spec(rules, &name, &shape) else {
        return vec![OwnedTensor::F32 { name, shape, data }];
    };
    let Some(axis) = split_axis(&shape, fused.split) else {
        return vec![OwnedTensor::F32 { name, shape, data }];
    };
    let mut out_shape = shape.clone();
    out_shape[axis] /= fused.split;
    let parts = split_tensor(&data, &shape, axis, fused.split);
    fused
        .targets
        .into_iter()
        .zip(parts)
        .map(|(name, data)| OwnedTensor::F32 {
            name,
            shape: out_shape.clone(),
            data,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct QuantizedGroup {
    base_name: String,
    qweight: String,
    scales: String,
    zeros: Option<String>,
    bits: u8,
    signed: bool,
    /// Ω1: 从元数据读取的块大小（必填，不再使用 Option）
    block_size: usize,
}

impl QuantizedGroup {
    /// Ω1: 从元数据创建量化组
    fn from_metadata(
        base_name: String,
        qweight: String,
        metadata: &QuantizationMetadata,
        name_to_tensor: &impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        metadata.validate()?;

        let scales = metadata
            .companions
            .scales
            .as_ref()
            .and_then(|pattern| name_to_tensor(pattern))
            .or_else(|| {
                // 向后兼容：尝试默认名称
                name_to_tensor("scales").or_else(|| name_to_tensor("scale"))
            })
            .ok_or_else(|| {
                LoaderError::InvalidQuantization(format!("未找到 scales 张量: {base_name}"))
            })?;

        let zeros = metadata
            .companions
            .zeros
            .as_ref()
            .and_then(|pattern| name_to_tensor(pattern))
            .or_else(|| {
                // 向后兼容：尝试默认名称
                name_to_tensor("qzeros").or_else(|| name_to_tensor("zeros"))
            });

        Ok(Self {
            base_name,
            qweight,
            scales,
            zeros,
            bits: metadata.bits,
            signed: metadata.signed,
            block_size: metadata.block_size,
        })
    }

    fn dequantize(&self, loader: &SafeTensorsLoader) -> Result<OwnedTensor> {
        let qweight = loader.tensor(&self.qweight)?;
        let scales = loader.tensor(&self.scales)?;
        let zeros = if let Some(name) = &self.zeros {
            Some(loader.tensor(name)?)
        } else {
            None
        };

        let (values, out_shape) = decode_quantized_values(&qweight, self.bits, self.signed)?;
        let scales = tensor_to_f32(&scales)?;
        if scales.is_empty() {
            return Err(LoaderError::InvalidQuantization(format!(
                "missing scales for {}",
                self.base_name
            )));
        }
        let zeros = if let Some(tensor) = zeros {
            let zeros = tensor_to_f32(&tensor)?;
            if zeros.is_empty() {
                vec![0.0]
            } else {
                zeros
            }
        } else {
            vec![0.0]
        };

        let total = values.len();
        // Ω1: 使用元数据中的 block_size（必填）
        let block_size = self.block_size;
        let mut out = Vec::with_capacity(total);
        for (idx, value) in values.into_iter().enumerate() {
            let block = idx / block_size;
            let scale = scales
                .get(block)
                .copied()
                .unwrap_or_else(|| *scales.last().unwrap_or(&1.0));
            let zero = zeros
                .get(block)
                .copied()
                .unwrap_or_else(|| *zeros.last().unwrap_or(&0.0));
            out.push((value as f32 - zero) * scale);
        }

        Ok(OwnedTensor::F32 {
            name: self.base_name.clone(),
            shape: out_shape,
            data: out,
        })
    }
}

#[derive(Default)]
struct QuantizedIndex {
    groups: HashMap<String, QuantizedGroup>,
    member_to_base: HashMap<String, String>,
}

impl QuantizedIndex {
    /// Ω1: 完全从模型元数据构建量化索引，禁止任何推测
    ///
    /// 对于非量化模型（没有 gllm.quantization 元数据），返回空索引
    fn from_loader(loader: &SafeTensorsLoader) -> Result<Self> {
        let mut index = QuantizedIndex::default();

        // 读取量化元数据（非量化模型没有此字段，返回空）
        let quantization_metadata = match loader.quantization_metadata()? {
            Some(meta) => meta,
            None => return Ok(index), // 非量化模型，返回空索引
        };

        // 验证所有量化组
        for (qweight_name, metadata) in &quantization_metadata {
            metadata.validate()?;

            // 验证 qweight 张量存在
            if loader.tensor(qweight_name).is_err() {
                return Err(LoaderError::MissingTensor(format!(
                    "量化元数据指定的 qweight 张量不存在: {qweight_name}"
                )));
            }

            // 构建名称查找函数
            let name_set: HashSet<_> = loader.names().into_iter().collect();
            let find_tensor = |name: &str| -> Option<String> {
                if name_set.contains(name) {
                    Some(name.to_string())
                } else {
                    None
                }
            };

            // 从元数据创建量化组
            let group = QuantizedGroup::from_metadata(
                qweight_name.clone(),
                qweight_name.clone(),
                metadata,
                &find_tensor,
            )?;

            // 验证关联张量存在
            if loader.tensor(&group.scales).is_err() {
                return Err(LoaderError::MissingTensor(format!(
                    "量化元数据指定的 scales 张量不存在: {}",
                    group.scales
                )));
            }
            if let Some(ref zeros) = group.zeros {
                if loader.tensor(zeros).is_err() {
                    return Err(LoaderError::MissingTensor(format!(
                        "量化元数据指定的 zeros 张量不存在: {zeros}"
                    )));
                }
            }

            // 注册到索引
            for member in [&group.qweight, &group.scales] {
                index
                    .member_to_base
                    .insert(member.to_string(), group.base_name.clone());
            }
            if let Some(ref zeros) = group.zeros {
                index
                    .member_to_base
                    .insert(zeros.to_string(), group.base_name.clone());
            }
            index.groups.insert(group.base_name.clone(), group);
        }

        Ok(index)
    }

    fn group_for(&self, name: &str) -> Option<&QuantizedGroup> {
        let base = self.member_to_base.get(name)?;
        self.groups.get(base)
    }
}

fn decode_quantized_values(
    tensor: &TensorSlice<'_>,
    bits: u8,
    signed: bool,
) -> Result<(Vec<i32>, Vec<usize>)> {
    if bits == 8 {
        match tensor.dtype {
            Dtype::I8 => {
                let data = tensor.as_i8()?;
                let values = data.iter().map(|v| *v as i32).collect::<Vec<_>>();
                return Ok((values, tensor.shape.clone()));
            }
            Dtype::U8 => {
                let data = tensor.as_u8()?;
                let values = data.iter().map(|v| *v as i32).collect::<Vec<_>>();
                return Ok((values, tensor.shape.clone()));
            }
            _ => {
                return Err(LoaderError::InvalidQuantization(format!(
                    "unsupported int8 dtype {:?}",
                    tensor.dtype
                )))
            }
        }
    }

    if bits == 4 {
        let data = tensor.as_u8()?;
        let unpacked_shape = unpacked_shape(&tensor.shape, bits);
        let mut values = Vec::with_capacity(data.len() * 2);
        for &byte in data.iter() {
            let lo = (byte & 0x0f) as i32;
            let hi = ((byte >> 4) & 0x0f) as i32;
            values.push(if signed && lo >= 8 { lo - 16 } else { lo });
            values.push(if signed && hi >= 8 { hi - 16 } else { hi });
        }
        let needed = unpacked_shape.iter().product::<usize>();
        if values.len() < needed {
            return Err(LoaderError::InvalidQuantization(format!(
                "packed int4 data too short for {:?}",
                tensor.shape
            )));
        }
        values.truncate(needed);
        return Ok((values, unpacked_shape));
    }

    Err(LoaderError::InvalidQuantization(format!(
        "unsupported quantization bits {bits}"
    )))
}

fn unpacked_shape(shape: &[usize], bits: u8) -> Vec<usize> {
    if bits >= 8 {
        return shape.to_vec();
    }
    let pack = (8 / bits) as usize;
    let mut out = shape.to_vec();
    if let Some(last) = out.last_mut() {
        *last = last.saturating_mul(pack);
    }
    out
}

fn tensor_to_f32(tensor: &TensorSlice<'_>) -> Result<Vec<f32>> {
    match tensor.dtype {
        Dtype::F16 => Ok(tensor.as_f16()?.iter().map(|v| v.to_f32()).collect()),
        Dtype::BF16 => Ok(tensor.as_bf16()?.iter().map(|v| v.to_f32()).collect()),
        Dtype::F32 => Ok(tensor.as_f32()?.iter().copied().collect()),
        Dtype::I8 => {
            let data = tensor.as_i8()?;
            Ok(dequantize_int8_with_zero(data.as_ref(), 1.0, 0.0))
        }
        Dtype::U8 => Ok(tensor.as_u8()?.iter().map(|v| *v as f32).collect()),
        other => Err(LoaderError::InvalidQuantization(format!(
            "unsupported quantization tensor dtype {other:?}"
        ))),
    }
}

fn gguf_shape_to_usize(shape: &[u64]) -> Result<Vec<usize>> {
    let mut out = Vec::with_capacity(shape.len());
    for &dim in shape {
        let dim = usize::try_from(dim)
            .map_err(|_| LoaderError::Gguf("gguf tensor shape overflow".to_string()))?;
        out.push(dim);
    }
    Ok(out)
}

fn gguf_tensor_to_f32(tensor: &gguf::TensorSlice<'_>) -> Result<Vec<f32>> {
    let data = tensor.as_bytes();
    match tensor.dtype() {
        gguf::GgmlDType::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(LoaderError::Gguf(
                    "gguf f16 tensor has invalid byte length".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        gguf::GgmlDType::BF16 => {
            if !data.len().is_multiple_of(2) {
                return Err(LoaderError::Gguf(
                    "gguf bf16 tensor has invalid byte length".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        gguf::GgmlDType::F32 => {
            if !data.len().is_multiple_of(4) {
                return Err(LoaderError::Gguf(
                    "gguf f32 tensor has invalid byte length".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        gguf::GgmlDType::F64 => {
            if !data.len().is_multiple_of(8) {
                return Err(LoaderError::Gguf(
                    "gguf f64 tensor has invalid byte length".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 8);
            for chunk in data.chunks_exact(8) {
                let value = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(value as f32);
            }
            Ok(out)
        }
        gguf::GgmlDType::I8 => Ok(data.iter().map(|v| (*v as i8) as f32).collect()),
        gguf::GgmlDType::I16 => {
            if !data.len().is_multiple_of(2) {
                return Err(LoaderError::Gguf(
                    "gguf i16 tensor has invalid byte length".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                out.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32);
            }
            Ok(out)
        }
        gguf::GgmlDType::I32 => {
            if !data.len().is_multiple_of(4) {
                return Err(LoaderError::Gguf(
                    "gguf i32 tensor has invalid byte length".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
            }
            Ok(out)
        }
        gguf::GgmlDType::I64 => {
            if !data.len().is_multiple_of(8) {
                return Err(LoaderError::Gguf(
                    "gguf i64 tensor has invalid byte length".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 8);
            for chunk in data.chunks_exact(8) {
                out.push(i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32);
            }
            Ok(out)
        }
        other => Err(LoaderError::InvalidQuantization(format!(
            "unsupported gguf tensor dtype for f32 conversion: {other:?}"
        ))),
    }
}

fn gguf_matrix_dims(shape: &[usize]) -> Result<(usize, usize)> {
    if shape.len() > 2 && shape[2..].iter().any(|&dim| dim != 1) {
        return Err(LoaderError::InvalidQuantization(
            "gguf quantized tensor has more than 2 dimensions".into(),
        ));
    }
    let cols = *shape.first().unwrap_or(&1);
    let rows = *shape.get(1).unwrap_or(&1);
    Ok((rows.max(1), cols.max(1)))
}

fn parse_q4_0_blocks(
    data: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<gllm_kernels::Q4_0Block>> {
    use gllm_kernels::{Q4_0Block, Q4_0_BLOCK_BYTES, QK4_0};

    let blocks_per_row = cols.div_ceil(QK4_0);
    let total_blocks = rows
        .checked_mul(blocks_per_row)
        .ok_or_else(|| LoaderError::InvalidQuantization("gguf block count overflow".into()))?;
    let block_bytes = 2 + Q4_0_BLOCK_BYTES;
    let expected = total_blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| LoaderError::InvalidQuantization("gguf block size overflow".into()))?;
    if data.len() != expected {
        return Err(LoaderError::InvalidQuantization(format!(
            "q4_0 tensor bytes {} do not match expected {}",
            data.len(),
            expected
        )));
    }

    let mut blocks = Vec::with_capacity(total_blocks);
    let mut offset = 0;
    for _ in 0..total_blocks {
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16::from_bits(scale_bits);
        offset += 2;
        let mut payload = [0u8; Q4_0_BLOCK_BYTES];
        payload.copy_from_slice(&data[offset..offset + Q4_0_BLOCK_BYTES]);
        offset += Q4_0_BLOCK_BYTES;
        blocks.push(Q4_0Block {
            scale,
            data: payload,
        });
    }
    Ok(blocks)
}

fn parse_q8_0_blocks(
    data: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<gllm_kernels::Q8_0Block>> {
    use gllm_kernels::{Q8_0Block, Q8_0_BLOCK_BYTES, QK8_0};

    let blocks_per_row = cols.div_ceil(QK8_0);
    let total_blocks = rows
        .checked_mul(blocks_per_row)
        .ok_or_else(|| LoaderError::InvalidQuantization("gguf block count overflow".into()))?;
    let block_bytes = 2 + Q8_0_BLOCK_BYTES;
    let expected = total_blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| LoaderError::InvalidQuantization("gguf block size overflow".into()))?;
    if data.len() != expected {
        return Err(LoaderError::InvalidQuantization(format!(
            "q8_0 tensor bytes {} do not match expected {}",
            data.len(),
            expected
        )));
    }

    let mut blocks = Vec::with_capacity(total_blocks);
    let mut offset = 0;
    for _ in 0..total_blocks {
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16::from_bits(scale_bits);
        offset += 2;
        let mut payload = [0u8; Q8_0_BLOCK_BYTES];
        payload.copy_from_slice(&data[offset..offset + Q8_0_BLOCK_BYTES]);
        offset += Q8_0_BLOCK_BYTES;
        blocks.push(Q8_0Block {
            scale,
            data: payload,
        });
    }
    Ok(blocks)
}

struct FusedSpec {
    split: usize,
    targets: Vec<String>,
}

fn fused_spec(rules: TensorNamingRule, name: &str, shape: &[usize]) -> Option<FusedSpec> {
    match rules {
        TensorNamingRule::GPT2Next => {
            if name.contains("c_attn") {
                return Some(FusedSpec {
                    split: 3,
                    targets: qkv_targets(name, "c_attn")?,
                });
            }
            if name.contains("c_fc") && name.contains("mlp") {
                if !shape_ratio_ok(shape, 2) {
                    return None;
                }
                return Some(FusedSpec {
                    split: 2,
                    targets: gate_up_targets(name, "c_fc")?,
                });
            }
        }
        TensorNamingRule::Qwen3 => {
            if name.contains("W_pack") || name.contains("w_pack") {
                return Some(FusedSpec {
                    split: 3,
                    targets: qkv_targets(name, "W_pack").or_else(|| qkv_targets(name, "w_pack"))?,
                });
            }
            if name.contains("gate_up_proj") {
                return Some(FusedSpec {
                    split: 2,
                    targets: gate_up_targets(name, "gate_up_proj")?,
                });
            }
        }
        _ => {
            if name.contains("qkv_proj") {
                return Some(FusedSpec {
                    split: 3,
                    targets: qkv_targets(name, "qkv_proj")?,
                });
            }
            if name.contains("gate_up_proj") {
                return Some(FusedSpec {
                    split: 2,
                    targets: gate_up_targets(name, "gate_up_proj")?,
                });
            }
        }
    }
    None
}

fn qkv_targets(name: &str, token: &str) -> Option<Vec<String>> {
    Some(vec![
        replace_last(name, token, "q_proj")?,
        replace_last(name, token, "k_proj")?,
        replace_last(name, token, "v_proj")?,
    ])
}

fn gate_up_targets(name: &str, token: &str) -> Option<Vec<String>> {
    Some(vec![
        replace_last(name, token, "gate_proj")?,
        replace_last(name, token, "up_proj")?,
    ])
}

fn replace_last(haystack: &str, needle: &str, with: &str) -> Option<String> {
    let pos = haystack.rfind(needle)?;
    let mut out = String::with_capacity(haystack.len() - needle.len() + with.len());
    out.push_str(&haystack[..pos]);
    out.push_str(with);
    out.push_str(&haystack[pos + needle.len()..]);
    Some(out)
}

/// Phi-4 GQA QKV 权重手动分割
///
/// Phi-4 使用 GQA (Grouped Query Attention)，QKV 融合权重的形状是 [5120, 3072]：
/// - Q: [3072, 3072] - query 投影
/// - K: [1024, 3072] - key 投影 (num_kv_heads * head_dim)
/// - V: [1024, 3072] - value 投影
///
/// 由于不是等分，需要手动按行分割。
fn split_phi4_qkv(name: &str, tensor: &TensorSlice<'_>) -> Option<Vec<OwnedTensor>> {
    // 从原始形状 [5120, 3072] 推断各部分大小
    let [out_dim, in_dim] = [tensor.shape[0], tensor.shape[1]];

    // Phi-4-mini: Q=3072, K=1024, V=1024
    // 通过 in_dim (hidden_size) 推断各部分
    // hidden_size = 3072, num_kv_heads * head_dim = 1024
    let q_dim = in_dim; // Q 投影输出 = hidden_size
    let kv_dim = (out_dim - q_dim) / 2; // K 和 V 各占剩余的一半

    if q_dim + kv_dim * 2 != out_dim {
        return None;
    }

    let targets = qkv_targets(name, "qkv_proj")?;
    let q_name = &targets[0];
    let k_name = &targets[1];
    let v_name = &targets[2];

    match tensor.dtype {
        Dtype::BF16 => {
            let data = tensor.as_bf16().ok()?;
            let _row_size = in_dim * 2; // 每行字节数 (bf16 = 2 bytes)
            let mut q_data = Vec::with_capacity(q_dim * in_dim);
            let mut k_data = Vec::with_capacity(kv_dim * in_dim);
            let mut v_data = Vec::with_capacity(kv_dim * in_dim);

            // 按行分割
            for row in 0..out_dim {
                let start = row * in_dim;
                let end = start + in_dim;
                let row_data = &data[start..end];

                if row < q_dim {
                    q_data.extend_from_slice(row_data);
                } else if row < q_dim + kv_dim {
                    k_data.extend_from_slice(row_data);
                } else {
                    v_data.extend_from_slice(row_data);
                }
            }

            Some(vec![
                OwnedTensor::BF16 {
                    name: q_name.clone(),
                    shape: vec![q_dim, in_dim],
                    data: q_data,
                },
                OwnedTensor::BF16 {
                    name: k_name.clone(),
                    shape: vec![kv_dim, in_dim],
                    data: k_data,
                },
                OwnedTensor::BF16 {
                    name: v_name.clone(),
                    shape: vec![kv_dim, in_dim],
                    data: v_data,
                },
            ])
        }
        Dtype::F16 => {
            let data = tensor.as_f16().ok()?;
            let mut q_data = Vec::with_capacity(q_dim * in_dim);
            let mut k_data = Vec::with_capacity(kv_dim * in_dim);
            let mut v_data = Vec::with_capacity(kv_dim * in_dim);

            for row in 0..out_dim {
                let start = row * in_dim;
                let end = start + in_dim;
                let row_data = &data[start..end];

                if row < q_dim {
                    q_data.extend_from_slice(row_data);
                } else if row < q_dim + kv_dim {
                    k_data.extend_from_slice(row_data);
                } else {
                    v_data.extend_from_slice(row_data);
                }
            }

            Some(vec![
                OwnedTensor::F16 {
                    name: q_name.clone(),
                    shape: vec![q_dim, in_dim],
                    data: q_data,
                },
                OwnedTensor::F16 {
                    name: k_name.clone(),
                    shape: vec![kv_dim, in_dim],
                    data: k_data,
                },
                OwnedTensor::F16 {
                    name: v_name.clone(),
                    shape: vec![kv_dim, in_dim],
                    data: v_data,
                },
            ])
        }
        Dtype::F32 => {
            let data = tensor.as_f32().ok()?;
            let mut q_data = Vec::with_capacity(q_dim * in_dim);
            let mut k_data = Vec::with_capacity(kv_dim * in_dim);
            let mut v_data = Vec::with_capacity(kv_dim * in_dim);

            for row in 0..out_dim {
                let start = row * in_dim;
                let end = start + in_dim;
                let row_data = &data[start..end];

                if row < q_dim {
                    q_data.extend_from_slice(row_data);
                } else if row < q_dim + kv_dim {
                    k_data.extend_from_slice(row_data);
                } else {
                    v_data.extend_from_slice(row_data);
                }
            }

            Some(vec![
                OwnedTensor::F32 {
                    name: q_name.clone(),
                    shape: vec![q_dim, in_dim],
                    data: q_data,
                },
                OwnedTensor::F32 {
                    name: k_name.clone(),
                    shape: vec![kv_dim, in_dim],
                    data: k_data,
                },
                OwnedTensor::F32 {
                    name: v_name.clone(),
                    shape: vec![kv_dim, in_dim],
                    data: v_data,
                },
            ])
        }
        _ => None,
    }
}

fn split_axis(shape: &[usize], split: usize) -> Option<usize> {
    shape.iter().position(|dim| *dim % split == 0)
}

fn shape_ratio_ok(shape: &[usize], ratio: usize) -> bool {
    shape.iter().any(|dim| *dim % ratio == 0)
}

fn split_tensor<T: Copy>(data: &[T], shape: &[usize], axis: usize, split: usize) -> Vec<Vec<T>> {
    let axis_dim = shape[axis];
    let chunk = axis_dim / split;
    let inner = shape[axis + 1..].iter().product::<usize>();
    let outer = shape[..axis].iter().product::<usize>();
    let part_len = outer * chunk * inner;
    let mut parts: Vec<Vec<T>> = (0..split).map(|_| Vec::with_capacity(part_len)).collect();

    for outer_idx in 0..outer {
        for (split_idx, part) in parts.iter_mut().enumerate().take(split) {
            let offset = outer_idx * axis_dim * inner + split_idx * chunk * inner;
            let slice = &data[offset..offset + chunk * inner];
            part.extend_from_slice(slice);
        }
    }
    parts
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(to_hex(&hasher.finalize()))
}

fn to_hex(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

fn materialize_model_dir(base: &Path, repo: &str, files: &[PathBuf]) -> Result<PathBuf> {
    let dir_name = repo.replace('/', "--");
    let model_dir = base.join(dir_name);
    std::fs::create_dir_all(&model_dir)?;
    for src in files {
        let file_name = match src.file_name() {
            Some(name) => name,
            None => continue,
        };
        let dest = model_dir.join(file_name);
        if dest.exists() {
            continue;
        }
        link_or_copy(src, &dest)?;
    }
    Ok(model_dir)
}

fn link_or_copy(src: &Path, dest: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        if let Err(err) = symlink(src, dest) {
            std::fs::copy(src, dest).map_err(|_| err)?;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        std::fs::copy(src, dest)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_tensor_axis0() {
        let data = vec![1u32, 2, 3, 4, 5, 6];
        let shape = vec![3, 2];
        let parts = split_tensor(&data, &shape, 0, 3);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], vec![1, 2]);
        assert_eq!(parts[1], vec![3, 4]);
        assert_eq!(parts[2], vec![5, 6]);
    }

    #[test]
    fn split_tensor_axis1() {
        let data = vec![1u32, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let parts = split_tensor(&data, &shape, 1, 3);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], vec![1, 4]);
        assert_eq!(parts[1], vec![2, 5]);
        assert_eq!(parts[2], vec![3, 6]);
    }

    #[test]
    fn replace_last_token() {
        let name = "model.layers.0.self_attn.c_attn.weight";
        let replaced = replace_last(name, "c_attn", "q_proj").unwrap();
        assert_eq!(replaced, "model.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn gpt2next_split_c_attn() {
        let name = "transformer.h.0.attn.c_attn.weight";
        let shape = vec![6, 2];
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in &values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let tensor = TensorSlice {
            dtype: Dtype::F32,
            shape: shape.clone(),
            data: &data,
        };

        let outputs = maybe_split_fused(Some(TensorNamingRule::GPT2Next), name, &tensor)
            .expect("expected split outputs");
        assert_eq!(outputs.len(), 3);

        match &outputs[0] {
            OwnedTensor::F32 { name, shape, data } => {
                assert_eq!(name, "transformer.h.0.attn.q_proj.weight");
                assert_eq!(shape.as_slice(), &[2, 2]);
                assert_eq!(data.as_slice(), &[0.0, 1.0, 2.0, 3.0]);
            }
            _ => panic!("unexpected tensor type"),
        }
        match &outputs[1] {
            OwnedTensor::F32 { name, shape, data } => {
                assert_eq!(name, "transformer.h.0.attn.k_proj.weight");
                assert_eq!(shape.as_slice(), &[2, 2]);
                assert_eq!(data.as_slice(), &[4.0, 5.0, 6.0, 7.0]);
            }
            _ => panic!("unexpected tensor type"),
        }
        match &outputs[2] {
            OwnedTensor::F32 { name, shape, data } => {
                assert_eq!(name, "transformer.h.0.attn.v_proj.weight");
                assert_eq!(shape.as_slice(), &[2, 2]);
                assert_eq!(data.as_slice(), &[8.0, 9.0, 10.0, 11.0]);
            }
            _ => panic!("unexpected tensor type"),
        }
    }

    #[test]
    fn gpt2next_split_c_fc() {
        let name = "transformer.h.0.mlp.c_fc.weight";
        let shape = vec![4, 2];
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in &values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let tensor = TensorSlice {
            dtype: Dtype::F32,
            shape: shape.clone(),
            data: &data,
        };

        let outputs = maybe_split_fused(Some(TensorNamingRule::GPT2Next), name, &tensor)
            .expect("expected split outputs");
        assert_eq!(outputs.len(), 2);

        match &outputs[0] {
            OwnedTensor::F32 { name, shape, data } => {
                assert_eq!(name, "transformer.h.0.mlp.gate_proj.weight");
                assert_eq!(shape.as_slice(), &[2, 2]);
                assert_eq!(data.as_slice(), &[0.0, 1.0, 2.0, 3.0]);
            }
            _ => panic!("unexpected tensor type"),
        }
        match &outputs[1] {
            OwnedTensor::F32 { name, shape, data } => {
                assert_eq!(name, "transformer.h.0.mlp.up_proj.weight");
                assert_eq!(shape.as_slice(), &[2, 2]);
                assert_eq!(data.as_slice(), &[4.0, 5.0, 6.0, 7.0]);
            }
            _ => panic!("unexpected tensor type"),
        }
    }

    #[test]
    fn gpt2next_skip_c_proj_split() {
        let name = "transformer.h.0.mlp.c_proj.weight";
        let shape = vec![4, 2];
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in &values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let tensor = TensorSlice {
            dtype: Dtype::F32,
            shape,
            data: &data,
        };

        let outputs = maybe_split_fused(Some(TensorNamingRule::GPT2Next), name, &tensor);
        assert!(outputs.is_none());
    }
}
