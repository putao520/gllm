//! HuggingFace integration.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hf_hub::api::sync::Api;
use hf_hub::api::Progress;
use serde::Deserialize;

use crate::manifest::FileMap;

use super::{parallel::ParallelLoader, LoaderError, Result};

/// Token 缓存文件位置 (与 huggingface-cli 一致)
const HF_TOKEN_PATH: &str = ".huggingface/token";

/// 简单的进度报告器 - 确保输出立即可见
struct ProgressReporter {
    filename: String,
    total: usize,
    start: Instant,
    last_print: Instant,
    last_bytes: usize,
}

impl ProgressReporter {
    fn new(filename: String, total: usize) -> Self {
        eprintln!("📥 下载: {} ({:.2} GB)", filename, total as f64 / 1e9);
        Self {
            filename,
            total,
            start: Instant::now(),
            last_print: Instant::now(),
            last_bytes: 0,
        }
    }

    fn print_progress(&mut self, current: usize) {
        let now = Instant::now();
        let elapsed_since_last_print = now.saturating_duration_since(self.last_print).as_secs_f64();

        // 每秒至少打印一次进度
        if elapsed_since_last_print >= 1.0 || current >= self.total {
            let percent = (current as f64 / self.total as f64 * 100.0).min(100.0);
            let total_elapsed = self.start.elapsed().as_secs_f64();
            let speed = if total_elapsed > 0.0 {
                (current as f64 / total_elapsed) / 1e6 // MB/s
            } else {
                0.0
            };

            let eta_secs = if speed > 0.0 {
                ((self.total - current) as f64 / (speed * 1e6)) as u64
            } else {
                0
            };

            eprint!(
                "\r   进度: {:.1}% ({:.1} MB / {:.1} MB) - {:.2} MB/s",
                percent,
                current as f64 / 1e6,
                self.total as f64 / 1e6,
                speed
            );

            if eta_secs > 0 {
                let eta_mins = eta_secs / 60;
                let eta_secs_rem = eta_secs % 60;
                eprint!(" - ETA: {}m{}s", eta_mins, eta_secs_rem);
            }

            eprintln!();

            self.last_print = now;
        }
        self.last_bytes = current;
    }
}

impl Progress for ProgressReporter {
    fn init(&mut self, total: usize, filename: &str) {
        self.total = total;
        self.filename = filename.to_string();
        eprintln!("📥 下载: {} ({:.2} MB)", filename, total as f64 / 1e6);
    }

    fn update(&mut self, current: usize) {
        self.print_progress(current);
    }

    fn finish(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let speed = if elapsed > 0.0 {
            (self.total as f64 / elapsed) / 1e6 // MB/s
        } else {
            0.0
        };
        eprintln!(
            "   ✅ 完成下载: {} ({:.2} MB, {:.2} MB/s, {:.1}s)",
            self.filename,
            self.total as f64 / 1e6,
            speed,
            elapsed
        );
    }
}

/// 从多个来源读取 HuggingFace Token
///
/// 优先级:
/// 1. 环境变量 HF_TOKEN
/// 2. 环境变量 HUGGING_FACE_HUB_TOKEN
/// 3. ~/.huggingface/token 文件
fn read_hf_token() -> Option<String> {
    // 1. 从环境变量读取
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            return Some(token);
        }
    }

    if let Ok(token) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
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
        let repo = self.api.model(repo.to_string());

        // 使用自定义进度下载（内部会检查缓存）
        // 注意：这会显示进度，即使文件已存在
        let progress = ProgressReporter::new(filename.to_string(), 0);
        repo.download_with_progress(filename, progress)
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
            // 串行下载：使用自定义进度报告
            let mut result = Vec::new();
            for (idx, shard_path) in shard_paths_list.iter().enumerate() {
                let filename = shard_path.to_string_lossy().to_string();
                eprintln!("📥 [{}/{}] 下载分片: {}", idx + 1, shards.len(), filename);

                let progress = ProgressReporter::new(filename.clone(), 0);
                let path = api.model(repo_id.clone())
                    .download_with_progress(&filename, progress)
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
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");

        // 优先级 1: HF_TOKEN
        std::env::set_var("HF_TOKEN", "hf_test_from_hf_token");
        assert_eq!(read_hf_token(), Some("hf_test_from_hf_token".to_string()));

        // 优先级 2: HUGGING_FACE_HUB_TOKEN (当 HF_TOKEN 不存在时)
        std::env::remove_var("HF_TOKEN");
        std::env::set_var("HUGGING_FACE_HUB_TOKEN", "hf_test_from_legacy");
        assert_eq!(read_hf_token(), Some("hf_test_from_legacy".to_string()));

        // HF_TOKEN 优先于 HUGGING_FACE_HUB_TOKEN
        std::env::set_var("HF_TOKEN", "hf_priority_test");
        assert_eq!(read_hf_token(), Some("hf_priority_test".to_string()));

        // 清理
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
    }

    #[test]
    fn test_read_hf_token_no_token() {
        // 注意：如果 ~/.huggingface/token 文件存在，此测试会跳过
        // 这是有意为之 - 在有实际 token 的环境中跳过此测试
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");

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
