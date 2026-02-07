//! HuggingFace integration.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;
use serde::Deserialize;

use crate::manifest::{FileMap, EMPTY_FILE_MAP};

#[cfg(feature = "candle")]
use super::pytorch::{convert_bins_to_safetensors, PytorchConversionConfig};
use super::{parallel::ParallelLoader, LoaderError, Result};

/// Token 缓存文件位置 (与 huggingface-cli 一致)
const DEFAULT_HF_TOKEN_PATH: &str = ".huggingface/token";

/// 从多个来源读取 HuggingFace Token
///
/// 优先级:
/// 1. 环境变量 HF_TOKEN
/// 2. ~/.huggingface/token 文件
fn resolve_token_path(token_path_override: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = token_path_override {
        if path.is_absolute() {
            return Some(path.to_path_buf());
        }
        let home = std::env::var("HOME").ok()?;
        return Some(PathBuf::from(home).join(path));
    }

    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(DEFAULT_HF_TOKEN_PATH))
}

fn read_hf_token(token_path_override: Option<&Path>) -> Option<String> {
    // 1. 从环境变量读取
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            return Some(token);
        }
    }

    // 2. 从 ~/.huggingface/token 文件读取
    let token_path = resolve_token_path(token_path_override)?;
    if let Ok(token) = fs::read_to_string(&token_path) {
        let token = token.trim();
        if !token.is_empty() && token.starts_with("hf_") {
            return Some(token.to_string());
        }
    }

    None
}

/// 检查错误消息是否为认证错误
fn is_auth_error(err: &str) -> bool {
    let err_lower = err.to_lowercase();
    err_lower.contains("401")
        || err_lower.contains("403")
        || err_lower.contains("unauthorized")
        || err_lower.contains("forbidden")
        || err_lower.contains("invalid username or password")
        || err_lower.contains("authentication")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    SafeTensors,
    Gguf,
    Onnx,
}

#[derive(Debug)]
pub struct HfModelFiles {
    pub repo: String,
    pub weights: Vec<PathBuf>,
    pub format: WeightFormat,
    pub aux_files: Vec<PathBuf>,
}

#[derive(Debug)]
pub struct HfHubClient {
    api: Api,
}

impl HfHubClient {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        Self::with_endpoint_and_token_path(cache_dir, None, None)
    }

    pub fn with_endpoint(cache_dir: PathBuf, endpoint: Option<String>) -> Result<Self> {
        Self::with_endpoint_and_token_path(cache_dir, endpoint, None)
    }

    pub fn with_endpoint_and_token_path(
        cache_dir: PathBuf,
        endpoint: Option<String>,
        token_path: Option<PathBuf>,
    ) -> Result<Self> {
        // 从多个来源读取 token
        let token = read_hf_token(token_path.as_deref());

        let mut builder = hf_hub::api::sync::ApiBuilder::new().with_cache_dir(cache_dir);

        // 设置 token 用于 gated 模型访问
        if let Some(token) = token {
            builder = builder.with_token(Some(token));
        }

        if let Some(endpoint) = endpoint {
            builder = builder.with_endpoint(endpoint);
        }

        let api = builder.build().map_err(|err| {
            // 检查是否为认证错误，提供更好的错误提示
            let err_msg = err.to_string();
            if is_auth_error(&err_msg) {
                LoaderError::AuthenticationError {
                    hint: "This model requires authentication. Please:\n\
                              1. Visit https://huggingface.co/settings/tokens to create a token\n\
                              2. Accept the model's license on the model page\n\
                              3. Set the HF_TOKEN environment variable:\n\
                                 export HF_TOKEN=hf_xxx...\n\
                              4. Or add the token to ~/.huggingface/token"
                        .to_string(),
                }
            } else {
                LoaderError::HfHub(err_msg)
            }
        })?;
        Ok(Self { api })
    }

    /// 使用指定 token 创建客户端（用于测试或显式传入 token）
    pub fn with_token(cache_dir: PathBuf, token: String) -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .with_token(Some(token))
            .build()
            .map_err(|err| LoaderError::HfHub(err.to_string()))?;
        Ok(Self { api })
    }

    pub fn download_model_files(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
    ) -> Result<HfModelFiles> {
        self.download_model_files_with_format(repo, file_map, parallel, None)
    }

    pub fn download_model_files_with_format(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
        format_hint: Option<WeightFormat>,
    ) -> Result<HfModelFiles> {
        let repo = repo.to_string();
        let aux_files = self.collect_aux_files(&repo, file_map);

        if let Some(format) = format_hint {
            let result = self.download_by_format(&repo, file_map, parallel, &aux_files, format)?;
            return result.ok_or(LoaderError::MissingWeights);
        }

        if let Some(files) = self.try_download_safetensors(&repo, file_map, parallel, &aux_files)? {
            return Ok(files);
        }
        if let Some(files) = self.try_download_gguf(&repo, &aux_files)? {
            return Ok(files);
        }
        if let Some(files) = self.try_download_onnx(&repo, &aux_files)? {
            return Ok(files);
        }

        Err(LoaderError::MissingWeights)
    }

    pub fn download_config_file(&self, repo: &str, file_map: FileMap) -> Result<PathBuf> {
        self.get_file_any_with_base_fallback(repo, file_map, "config.json")
    }

    pub fn download_tokenizer_file(&self, repo: &str, file_map: FileMap) -> Result<PathBuf> {
        self.get_file_any_with_base_fallback(repo, file_map, "tokenizer.json")
    }

    fn collect_aux_files(&self, repo: &str, file_map: FileMap) -> Vec<PathBuf> {
        const AUX_FILES: [&str; 6] = [
            "config.json",
            "configuration.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
        ];

        fn has_any_named(paths: &[PathBuf], names: &[&str]) -> bool {
            paths.iter().any(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| names.contains(&name))
            })
        }

        fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
            if !paths.iter().any(|existing| existing == &path) {
                paths.push(path);
            }
        }

        let mut aux_files = Vec::new();
        for name in AUX_FILES {
            if let Ok(path) = self.get_file_any(repo, file_map, name) {
                push_unique(&mut aux_files, path);
            }
        }

        let has_config = has_any_named(&aux_files, &["config.json", "configuration.json"]);
        let has_tokenizer = has_any_named(&aux_files, &["tokenizer.json"]);
        if has_config && has_tokenizer {
            return aux_files;
        }

        let Some(base_repo) = self.resolve_base_model_repo(repo) else {
            return aux_files;
        };

        for name in AUX_FILES {
            let required = match name {
                "config.json" | "configuration.json" => !has_config,
                "tokenizer.json" => !has_tokenizer,
                _ => true,
            };
            if !required {
                continue;
            }
            if let Ok(path) = self.get_file_any(&base_repo, EMPTY_FILE_MAP, name) {
                push_unique(&mut aux_files, path);
            }
        }

        aux_files
    }

    fn get_file_any_with_base_fallback(
        &self,
        repo: &str,
        file_map: FileMap,
        logical: &str,
    ) -> Result<PathBuf> {
        match self.get_file_any(repo, file_map, logical) {
            Ok(path) => Ok(path),
            Err(LoaderError::MissingWeights) => {
                let base_repo = self
                    .resolve_base_model_repo(repo)
                    .ok_or(LoaderError::MissingWeights)?;
                self.get_file_any(&base_repo, EMPTY_FILE_MAP, logical)
            }
            Err(err) => Err(err),
        }
    }

    fn download_by_format(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
        aux_files: &[PathBuf],
        format: WeightFormat,
    ) -> Result<Option<HfModelFiles>> {
        match format {
            WeightFormat::SafeTensors => {
                self.try_download_safetensors(repo, file_map, parallel, aux_files)
            }
            WeightFormat::Gguf => self.try_download_gguf(repo, aux_files),
            WeightFormat::Onnx => self.try_download_onnx(repo, aux_files),
        }
    }

    fn try_download_safetensors(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
        aux_files: &[PathBuf],
    ) -> Result<Option<HfModelFiles>> {
        if let Ok(index_path) = self.get_file_any(repo, file_map, "model.safetensors.index.json") {
            let shard_index = ShardIndex::from_path(&index_path)?;
            let shard_files = shard_index.shard_files();
            let weights = self.download_shards(repo, &shard_files, parallel)?;
            let mut aux = aux_files.to_vec();
            aux.push(index_path);
            return Ok(Some(HfModelFiles {
                repo: repo.to_string(),
                weights,
                format: WeightFormat::SafeTensors,
                aux_files: aux,
            }));
        }

        if let Ok(path) = self.get_file_any(repo, file_map, "model.safetensors") {
            return Ok(Some(HfModelFiles {
                repo: repo.to_string(),
                weights: vec![path],
                format: WeightFormat::SafeTensors,
                aux_files: aux_files.to_vec(),
            }));
        }

        #[cfg(feature = "candle")]
        if let Some((weights, index_path)) =
            self.try_download_pytorch_bins(repo, file_map, parallel)?
        {
            let mut aux = aux_files.to_vec();
            if let Some(index_path) = index_path {
                aux.push(index_path);
            }
            return Ok(Some(HfModelFiles {
                repo: repo.to_string(),
                weights,
                format: WeightFormat::SafeTensors,
                aux_files: aux,
            }));
        }

        Ok(None)
    }

    fn try_download_gguf(&self, repo: &str, aux_files: &[PathBuf]) -> Result<Option<HfModelFiles>> {
        for candidate in self.ranked_gguf_candidates(repo) {
            if let Ok(path) = self.get_file(repo, &candidate) {
                return Ok(Some(HfModelFiles {
                    repo: repo.to_string(),
                    weights: vec![path],
                    format: WeightFormat::Gguf,
                    aux_files: aux_files.to_vec(),
                }));
            }
        }
        Ok(None)
    }

    fn try_download_onnx(&self, repo: &str, aux_files: &[PathBuf]) -> Result<Option<HfModelFiles>> {
        for candidate in self.ranked_onnx_candidates(repo) {
            if let Ok(path) = self.get_file(repo, &candidate) {
                return Ok(Some(HfModelFiles {
                    repo: repo.to_string(),
                    weights: vec![path],
                    format: WeightFormat::Onnx,
                    aux_files: aux_files.to_vec(),
                }));
            }
        }
        Ok(None)
    }

    /// Ω1: 候选文件名列表（按优先级排序）
    /// 注意：这不基于元数据，仅作为文件存在性检查的顺序
    /// 用户如需特定文件，应通过 file_map 显式指定
    fn gguf_candidate_names(&self) -> Vec<String> {
        vec![
            "model.gguf".to_string(),
            "ggml-model-q4_0.gguf".to_string(),
            "ggml-model-q8_0.gguf".to_string(),
            "ggml-model-f16.gguf".to_string(),
        ]
    }

    /// Ω1: 候选文件名列表（按优先级排序）
    fn onnx_candidate_names(&self) -> Vec<String> {
        vec!["onnx/model.onnx".to_string(), "model.onnx".to_string()]
    }

    fn ranked_gguf_candidates(&self, repo: &str) -> Vec<String> {
        fn preferred_rank(name: &str) -> usize {
            let lower = name.to_ascii_lowercase();
            if lower.ends_with("q4_0.gguf") {
                0
            } else if lower.ends_with("q8_0.gguf") {
                1
            } else if lower.ends_with("f16.gguf") || lower.ends_with("fp16.gguf") {
                2
            } else if lower.ends_with("f32.gguf") || lower.ends_with("fp32.gguf") {
                3
            } else {
                4
            }
        }

        if let Ok(files) = self.list_repo_files(repo) {
            // 优先选择常见可用量化类型，其次按文件名稳定排序。
            let mut gguf_files: Vec<_> = files
                .into_iter()
                .filter(|name| name.ends_with(".gguf"))
                .collect();
            if !gguf_files.is_empty() {
                gguf_files.sort_by(|a, b| {
                    let a_model = a == "model.gguf";
                    let b_model = b == "model.gguf";
                    a_model
                        .cmp(&b_model)
                        .reverse()
                        .then_with(|| preferred_rank(a).cmp(&preferred_rank(b)))
                        .then_with(|| a.cmp(b))
                });
                return gguf_files;
            }
        }

        // 回退到预设的候选列表
        self.gguf_candidate_names()
    }

    fn ranked_onnx_candidates(&self, repo: &str) -> Vec<String> {
        if let Ok(files) = self.list_repo_files(repo) {
            // Ω1: 优先选择 onnx/ 目录下的文件
            let onnx_dir_files: Vec<_> = files
                .iter()
                .filter(|name| name.starts_with("onnx/") && name.ends_with(".onnx"))
                .cloned()
                .collect();
            if !onnx_dir_files.is_empty() {
                let mut result = onnx_dir_files;
                result.sort();
                return result;
            }

            // 其次选择根目录的 onnx 文件
            let root_onnx: Vec<_> = files
                .into_iter()
                .filter(|name| name.ends_with(".onnx"))
                .collect();
            if !root_onnx.is_empty() {
                let mut result = root_onnx;
                result.sort();
                return result;
            }
        }

        // 回退到预设的候选列表
        self.onnx_candidate_names()
    }

    fn list_repo_files(&self, repo: &str) -> Result<Vec<String>> {
        let info = self
            .api
            .model(repo.to_string())
            .info()
            .map_err(|err| LoaderError::HfHub(err.to_string()))?;
        Ok(info.siblings.into_iter().map(|s| s.rfilename).collect())
    }

    fn resolve_base_model_repo(&self, repo: &str) -> Option<String> {
        let endpoint = std::env::var("HF_ENDPOINT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "https://huggingface.co".to_string());
        let url = format!("{}/api/models/{}", endpoint.trim_end_matches('/'), repo);
        let response = ureq::get(&url).call().ok()?;
        if response.status() != 200 {
            return None;
        }
        let body = response.into_string().ok()?;
        let metadata: HfModelMetadata = serde_json::from_str(&body).ok()?;
        metadata
            .base_model_repo()
            .filter(|base_repo| base_repo != repo)
    }

    #[cfg(feature = "candle")]
    fn try_download_pytorch_bins(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
    ) -> Result<Option<(Vec<PathBuf>, Option<PathBuf>)>> {
        if let Ok(index_path) = self.get_file_any(repo, file_map, "pytorch_model.bin.index.json") {
            let shard_index = ShardIndex::from_path(&index_path)?;
            let shard_files = shard_index.shard_files();
            let bin_paths = self.download_shards(repo, &shard_files, parallel)?;
            let config = PytorchConversionConfig::default();
            let output = convert_bins_to_safetensors(&bin_paths, Some(&index_path), &config)?;
            return Ok(Some((output.safetensors, output.index)));
        }

        if let Ok(bin_path) = self.get_file_any(repo, file_map, "pytorch_model.bin") {
            let config = PytorchConversionConfig::default();
            let output =
                convert_bins_to_safetensors(std::slice::from_ref(&bin_path), None, &config)?;
            return Ok(Some((output.safetensors, output.index)));
        }

        Ok(None)
    }

    fn get_file(&self, repo: &str, filename: &str) -> Result<PathBuf> {
        let repo_api = self.api.model(repo.to_string());
        // get() 会自动检查缓存，不存在则下载（无进度显示）
        repo_api
            .get(filename)
            .map_err(|err| LoaderError::HfHub(err.to_string()))
    }

    fn get_file_any(&self, repo: &str, file_map: FileMap, logical: &str) -> Result<PathBuf> {
        for candidate in candidate_names(file_map, logical) {
            if let Ok(path) = self.get_file(repo, &candidate) {
                return Ok(path);
            }
        }
        Err(LoaderError::MissingWeights)
    }

    fn download_shards(
        &self,
        repo: &str,
        shards: &[String],
        parallel: ParallelLoader,
    ) -> Result<Vec<PathBuf>> {
        let api = self.api.clone();
        let repo_id = repo.to_string();
        let shard_paths_list: Vec<PathBuf> = shards.iter().map(PathBuf::from).collect();

        if parallel.enabled() {
            // 并行下载：使用默认进度条
            eprintln!("📥 并行下载 {} 个分片...", shards.len());
            let shard_paths = parallel.map_paths(&shard_paths_list, |path| {
                let filename = path.to_string_lossy().to_string();
                api.model(repo_id.clone())
                    .get(&filename)
                    .map_err(|err| LoaderError::HfHub(err.to_string()))
            })?;
            eprintln!("   ✅ 并行下载完成");
            Ok(shard_paths)
        } else {
            // 串行下载：get() 会自动检查缓存
            let mut result = Vec::new();
            for shard_path in shard_paths_list {
                let filename = shard_path.to_string_lossy().to_string();
                let repo_api = api.model(repo_id.clone());

                let path = repo_api
                    .get(&filename)
                    .map_err(|err| LoaderError::HfHub(err.to_string()))?;
                result.push(path);
            }
            Ok(result)
        }
    }
}

fn map_name(file_map: FileMap, logical: &str) -> &str {
    for (source, target) in file_map {
        if *source == logical {
            return target;
        }
    }
    logical
}

fn candidate_names(file_map: FileMap, logical: &str) -> Vec<String> {
    let mut base_names = Vec::new();
    base_names.push(map_name(file_map, logical).to_string());
    if logical == "config.json" {
        base_names.push(map_name(file_map, "configuration.json").to_string());
    }

    let mut out = Vec::new();
    for base in base_names {
        if !out.contains(&base) {
            out.push(base.clone());
        }
        for prefix in ["model/", "weights/"] {
            let candidate = format!("{prefix}{base}");
            if !out.contains(&candidate) {
                out.push(candidate);
            }
        }
    }
    out
}

#[derive(Debug, Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

impl ShardIndex {
    fn from_path(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(LoaderError::Json)
    }

    fn shard_files(&self) -> Vec<String> {
        let mut shards = BTreeSet::new();
        for shard in self.weight_map.values() {
            shards.insert(shard.clone());
        }
        shards.into_iter().collect()
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BaseModelField {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize, Default)]
struct HfCardData {
    #[serde(default, alias = "baseModel")]
    base_model: Option<BaseModelField>,
}

#[derive(Debug, Deserialize, Default)]
struct HfModelMetadata {
    #[serde(default, rename = "cardData")]
    card_data: HfCardData,
    #[serde(default)]
    tags: Vec<String>,
}

impl HfModelMetadata {
    fn base_model_repo(&self) -> Option<String> {
        if let Some(base_model) = &self.card_data.base_model {
            match base_model {
                BaseModelField::Single(repo) if !repo.trim().is_empty() => {
                    return Some(repo.to_string());
                }
                BaseModelField::Multiple(repos) => {
                    if let Some(repo) = repos.iter().find(|repo| !repo.trim().is_empty()) {
                        return Some(repo.to_string());
                    }
                }
                _ => {}
            }
        }

        self.tags
            .iter()
            .filter_map(|tag| tag.strip_prefix("base_model:"))
            .find_map(|value| {
                if value.starts_with("quantized:") || value.trim().is_empty() {
                    None
                } else {
                    Some(value.to_string())
                }
            })
            .or_else(|| {
                self.tags
                    .iter()
                    .filter_map(|tag| tag.strip_prefix("base_model:quantized:"))
                    .find(|value| !value.trim().is_empty())
                    .map(|value| value.to_string())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let previous = std::env::var(key).ok();
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn candidate_names_include_modelscope_layouts() {
        let empty: FileMap = &[];
        let candidates = candidate_names(empty, "config.json");
        assert!(candidates.iter().any(|c| c == "config.json"));
        assert!(candidates.iter().any(|c| c == "configuration.json"));
        assert!(candidates.iter().any(|c| c == "model/config.json"));
        assert!(candidates.iter().any(|c| c == "model/configuration.json"));
    }

    #[test]
    fn test_is_auth_error() {
        assert!(is_auth_error("Error: 401 Unauthorized"));
        assert!(is_auth_error("Error: 403 Forbidden"));
        assert!(is_auth_error("Invalid username or password"));
        assert!(is_auth_error("Authentication failed"));
        assert!(is_auth_error("Access forbidden"));

        // 非认证错误
        assert!(!is_auth_error("File not found"));
        assert!(!is_auth_error("Connection timeout"));
        assert!(!is_auth_error("Network error"));
    }

    #[test]
    fn test_read_hf_token_priority() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", Some("hf_test_from_hf_token"));

        // 优先级 1: HF_TOKEN
        assert_eq!(
            read_hf_token(None),
            Some("hf_test_from_hf_token".to_string())
        );
    }

    #[test]
    fn test_read_hf_token_no_token() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", None);

        // 注意：如果 ~/.huggingface/token 文件存在，此测试会跳过
        // 这是有意为之 - 在有实际 token 的环境中跳过此测试
        // 检查 token 文件是否存在
        if let Ok(home) = std::env::var("HOME") {
            let token_path = PathBuf::from(home).join(DEFAULT_HF_TOKEN_PATH);
            if token_path.exists() {
                // token 文件存在，跳过测试
                return;
            }
        }

        // 只有在没有 token 文件时才断言
        assert!(read_hf_token(None).is_none());
    }

    #[test]
    fn test_read_hf_token_from_config_path() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", None);

        let temp_dir = std::env::temp_dir().join(format!("gllm-hf-token-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).expect("create temp token dir");
        let token_path = temp_dir.join("token");
        std::fs::write(&token_path, "hf_config_path_token\n").expect("write temp token file");

        let token = read_hf_token(Some(&token_path));
        assert_eq!(token, Some("hf_config_path_token".to_string()));

        let _ = std::fs::remove_file(&token_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn base_model_repo_prefers_card_data() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{
                "cardData": { "base_model": "intfloat/e5-small-v2" },
                "tags": ["base_model:quantized:intfloat/e5-small-v2"]
            }"#,
        )
        .expect("metadata");

        assert_eq!(
            metadata.base_model_repo(),
            Some("intfloat/e5-small-v2".to_string())
        );
    }

    #[test]
    fn base_model_repo_falls_back_to_tags() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{
                "tags": [
                    "base_model:quantized:intfloat/e5-small-v2",
                    "gguf"
                ]
            }"#,
        )
        .expect("metadata");

        assert_eq!(
            metadata.base_model_repo(),
            Some("intfloat/e5-small-v2".to_string())
        );
    }
}
