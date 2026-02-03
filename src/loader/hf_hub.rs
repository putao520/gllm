//! HuggingFace integration.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;
use hf_hub::api::Progress;
use serde::Deserialize;

use crate::manifest::FileMap;

use super::{
    downloader::{ProgressBar, ProgressCallback},
    parallel::ParallelLoader,
    LoaderError, Result,
};

/// Token 缓存文件位置 (与 huggingface-cli 一致)
const HF_TOKEN_PATH: &str = ".huggingface/token";

/// 适配器：将我们的 ProgressCallback 转换为 hf_hub::api::Progress
struct HfProgressAdapter<'a> {
    inner: &'a mut ProgressBar,
}

impl<'a> HfProgressAdapter<'a> {
    fn new(inner: &'a mut ProgressBar) -> Self {
        Self { inner }
    }
}

impl<'a> Progress for HfProgressAdapter<'a> {
    fn init(&mut self, total: usize, filename: &str) {
        self.inner.init(total, filename);
    }

    fn update(&mut self, current: usize) {
        self.inner.update(current);
    }

    fn finish(&mut self) {
        self.inner.finish();
    }
}

/// 从多个来源读取 HuggingFace Token
///
/// 优先级:
/// 1. 环境变量 HF_TOKEN
/// 2. ~/.huggingface/token 文件
fn read_hf_token() -> Option<String> {
    // 1. 从环境变量读取
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            return Some(token);
        }
    }

    // 2. 从 ~/.huggingface/token 文件读取
    if let Some(home) = std::env::var("HOME").ok() {
        let token_path = PathBuf::from(home).join(HF_TOKEN_PATH);
        if let Ok(token) = fs::read_to_string(&token_path) {
            let token = token.trim();
            if !token.is_empty() && token.starts_with("hf_") {
                return Some(token.to_string());
            }
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
    Bin,
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
        Self::with_endpoint(cache_dir, None)
    }

    pub fn with_endpoint(cache_dir: PathBuf, endpoint: Option<String>) -> Result<Self> {
        // 从多个来源读取 token
        let token = read_hf_token();

        let mut builder = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir);

        // 设置 token 用于 gated 模型访问
        if let Some(token) = token {
            builder = builder.with_token(Some(token));
        }

        if let Some(endpoint) = endpoint {
            builder = builder.with_endpoint(endpoint);
        }

        let api = builder
            .build()
            .map_err(|err| {
                // 检查是否为认证错误，提供更好的错误提示
                let err_msg = err.to_string();
                if is_auth_error(&err_msg) {
                    LoaderError::AuthenticationError {
                        hint: "This model requires authentication. Please:\n\
                              1. Visit https://huggingface.co/settings/tokens to create a token\n\
                              2. Accept the model's license on the model page\n\
                              3. Set the HF_TOKEN environment variable:\n\
                                 export HF_TOKEN=hf_xxx...\n\
                              4. Or add the token to ~/.huggingface/token".to_string(),
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
        let repo = repo.to_string();
        let mut aux_files = Vec::new();

        for name in [
            "config.json",
            "configuration.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
        ] {
            if let Ok(path) = self.get_file_any(&repo, file_map, name) {
                aux_files.push(path);
            }
        }

        if let Ok(index_path) = self.get_file_any(&repo, file_map, "model.safetensors.index.json") {
            let shard_index = ShardIndex::from_path(&index_path)?;
            let shard_files = shard_index.shard_files();
            let weights = self.download_shards(&repo, &shard_files, parallel)?;
            aux_files.push(index_path);
            return Ok(HfModelFiles {
                repo,
                weights,
                format: WeightFormat::SafeTensors,
                aux_files,
            });
        }

        if let Ok(path) = self.get_file_any(&repo, file_map, "model.safetensors") {
            return Ok(HfModelFiles {
                repo,
                weights: vec![path],
                format: WeightFormat::SafeTensors,
                aux_files,
            });
        }

        if let Ok(index_path) = self.get_file_any(&repo, file_map, "pytorch_model.bin.index.json") {
            let shard_index = ShardIndex::from_path(&index_path)?;
            let shard_files = shard_index.shard_files();
            let weights = self.download_shards(&repo, &shard_files, parallel)?;
            aux_files.push(index_path);
            return Ok(HfModelFiles {
                repo,
                weights,
                format: WeightFormat::Bin,
                aux_files,
            });
        }

        if let Ok(path) = self.get_file_any(&repo, file_map, "pytorch_model.bin") {
            return Ok(HfModelFiles {
                repo,
                weights: vec![path],
                format: WeightFormat::Bin,
                aux_files,
            });
        }

        Err(LoaderError::MissingWeights)
    }

    fn get_file(&self, repo: &str, filename: &str) -> Result<PathBuf> {
        let repo_api = self.api.model(repo.to_string());

        // 先检查文件是否已缓存（避免显示不必要的进度）
        if let Ok(path) = repo_api.get(filename) {
            return Ok(path);
        }

        // 文件不存在，使用进度报告器下载
        let mut progress = ProgressBar::new(filename.to_string());
        let adapter = HfProgressAdapter::new(&mut progress);
        repo_api.download_with_progress(filename, adapter)
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
            // 串行下载：检查文件是否已缓存
            let mut result = Vec::new();
            for (idx, shard_path) in shard_paths_list.iter().enumerate() {
                let filename = shard_path.to_string_lossy().to_string();
                let repo_api = api.model(repo_id.clone());

                // 先检查文件是否已缓存
                if let Ok(path) = repo_api.get(&filename) {
                    result.push(path);
                    continue;
                }

                // 文件不存在，使用进度报告器下载
                eprintln!("📥 [{}/{}] 下载分片: {}", idx + 1, shards.len(), filename);
                let mut progress = ProgressBar::new(filename.clone());
                let adapter = HfProgressAdapter::new(&mut progress);
                let path = repo_api
                    .download_with_progress(&filename, adapter)
                    .map_err(|err| LoaderError::HfHub(err.to_string()))?;
                result.push(path);
            }
            Ok(result)
        }
    }
}

fn map_name<'a>(file_map: FileMap, logical: &'a str) -> &'a str {
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
        serde_json::from_slice(&bytes).map_err(|err| LoaderError::Json(err))
    }

    fn shard_files(&self) -> Vec<String> {
        let mut shards = BTreeSet::new();
        for shard in self.weight_map.values() {
            shards.insert(shard.clone());
        }
        shards.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // 清理环境
        std::env::remove_var("HF_TOKEN");

        // 优先级 1: HF_TOKEN
        std::env::set_var("HF_TOKEN", "hf_test_from_hf_token");
        assert_eq!(read_hf_token(), Some("hf_test_from_hf_token".to_string()));

        // 清理
        std::env::remove_var("HF_TOKEN");
    }

    #[test]
    fn test_read_hf_token_no_token() {
        // 注意：如果 ~/.huggingface/token 文件存在，此测试会跳过
        // 这是有意为之 - 在有实际 token 的环境中跳过此测试
        std::env::remove_var("HF_TOKEN");

        // 检查 token 文件是否存在
        if let Some(home) = std::env::var("HOME").ok() {
            let token_path = PathBuf::from(home).join(HF_TOKEN_PATH);
            if token_path.exists() {
                // token 文件存在，跳过测试
                return;
            }
        }

        // 只有在没有 token 文件时才断言
        assert!(read_hf_token().is_none());
    }
}
