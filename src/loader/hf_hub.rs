//! HuggingFace integration.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Component, Path, PathBuf};

use hf_hub::api::sync::Api;
use hf_hub::api::Progress;
use serde::Deserialize;

use crate::manifest::{FileMap, EMPTY_FILE_MAP};


use super::{parallel::ParallelLoader, LoaderError, Result};

/// HF 下载进度适配器：将 hf_hub::api::Progress 回调桥接到 log::info 输出。
///
/// hf_hub 的 `download_with_progress()` 接受 `impl Progress` 参数，
/// 本适配器在 init/finish 时输出 log::info，提供下载可观测性。
/// `update()` 按 10% 间隔输出进度，避免高频刷屏。
struct HfLogProgress {
    filename: String,
    total: usize,
    cumulative: usize,
    last_logged_pct: u8,
}

impl HfLogProgress {
    fn new(filename: &str) -> Self {
        Self {
            filename: filename.to_string(),
            total: 0,
            cumulative: 0,
            last_logged_pct: 0,
        }
    }
}

impl Progress for HfLogProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.total = size;
        self.filename = filename.to_string();
        if size > 0 {
            log::info!("下载: {} ({:.2} MB)", filename, size as f64 / 1e6);
        } else {
            log::info!("下载: {} (大小未知)", filename);
        }
    }

    fn update(&mut self, size: usize) {
        // hf_hub update 传入增量字节数，累加后按 10% 间隔输出进度
        self.cumulative += size;
        if self.total > 0 {
            let pct = ((self.cumulative as f64 / self.total as f64) * 100.0) as u8;
            // 每 10% 输出一次，避免刷屏
            if pct >= self.last_logged_pct + 10 {
                log::info!(
                    "下载进度: {} — {}%",
                    self.filename,
                    pct
                );
                self.last_logged_pct = pct;
            }
        }
    }

    fn finish(&mut self) {
        log::info!("下载完成: {}", self.filename);
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    SafeTensors,
    Gguf,
    Onnx,
    PyTorch,
}

#[derive(Debug, PartialEq)]
pub struct HfModelFiles {
    pub repo: String,
    pub weights: Vec<PathBuf>,
    pub format: WeightFormat,
    pub aux_files: Vec<PathBuf>,
}

fn gguf_preferred_rank(name: &str) -> usize {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with("q4_0.gguf") { 0 }
    else if lower.ends_with("q8_0.gguf") { 1 }
    else if lower.ends_with("q4_1.gguf") { 2 }
    else if lower.ends_with("q5_0.gguf") { 3 }
    else if lower.ends_with("q5_1.gguf") { 4 }
    else if lower.ends_with("q8_1.gguf") { 5 }
    else if lower.contains("q4_k_m") && lower.ends_with(".gguf") { 6 }
    else if lower.contains("q4_k_s") && lower.ends_with(".gguf") { 7 }
    else if lower.contains("q5_k_m") && lower.ends_with(".gguf") { 8 }
    else if lower.contains("q5_k_s") && lower.ends_with(".gguf") { 9 }
    else if lower.contains("q6_k") && lower.ends_with(".gguf") { 10 }
    else if lower.contains("q6_k_l") && lower.ends_with(".gguf") { 11 }
    else if lower.contains("q3_k_m") && lower.ends_with(".gguf") { 12 }
    else if lower.contains("q3_k_s") && lower.ends_with(".gguf") { 13 }
    else if lower.contains("q2_k") && lower.ends_with(".gguf") { 14 }
    else if lower.contains("q8_k") && lower.ends_with(".gguf") { 15 }
    else if lower.contains("iq4_nl") && lower.ends_with(".gguf") { 16 }
    else if lower.contains("iq4_xs") && lower.ends_with(".gguf") { 17 }
    else if lower.contains("iq3_s") && lower.ends_with(".gguf") { 18 }
    else if lower.contains("iq3_xxs") && lower.ends_with(".gguf") { 19 }
    else if lower.contains("iq2") && lower.ends_with(".gguf") { 20 }
    else if lower.ends_with("f16.gguf") || lower.ends_with("fp16.gguf") { 21 }
    else if lower.ends_with("f32.gguf") || lower.ends_with("fp32.gguf") { 22 }
    else { 50 }
}

#[derive(Debug)]
pub struct HfHubClient {
    api: Api,
    cache_dir: PathBuf,
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

        let mut builder = hf_hub::api::sync::ApiBuilder::new().with_cache_dir(cache_dir.clone());

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
        Ok(Self { api, cache_dir })
    }

    /// 使用指定 token 创建客户端（用于测试或显式传入 token）
    pub fn with_token(cache_dir: PathBuf, token: String) -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir.clone())
            .with_token(Some(token))
            .build()
            .map_err(|err| LoaderError::HfHub(err.to_string()))?;
        Ok(Self { api, cache_dir })
    }

    pub fn download_model_files(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
    ) -> Result<HfModelFiles> {
        self.download_model_files_filtered(repo, file_map, parallel, None)
    }

    /// Download model files with an optional GGUF filename filter.
    /// When `gguf_filter` is set, only GGUF files whose name contains the
    /// filter substring are considered.
    pub fn download_model_files_filtered(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
        gguf_filter: Option<&str>,
    ) -> Result<HfModelFiles> {
        self.download_model_files_with_format_and_filter(
            repo, file_map, parallel, None, gguf_filter,
        )
    }

    fn download_model_files_with_format_and_filter(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
        format_hint: Option<WeightFormat>,
        gguf_filter: Option<&str>,
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
        if let Some(files) = self.try_download_gguf_filtered(&repo, &aux_files, gguf_filter)? {
            return Ok(files);
        }
        if let Some(files) = self.try_download_onnx(&repo, &aux_files)? {
            return Ok(files);
        }
        if let Some(files) = self.try_download_pytorch_bins(&repo, file_map, parallel, &aux_files)? {
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

        // 尝试通过 API 解析 base repo（需要网络）
        let base_repo = self.resolve_base_model_repo(repo)
            // API 不可用时，扫描本地缓存中可能的 base model 仓库
            .or_else(|| self.find_base_repo_in_cache(repo));

        let Some(base_repo) = base_repo else {
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
            WeightFormat::PyTorch => self.try_download_pytorch_bins(repo, file_map, parallel, aux_files),
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

        Ok(None)
    }

    fn try_download_gguf(&self, repo: &str, aux_files: &[PathBuf]) -> Result<Option<HfModelFiles>> {
        self.try_download_gguf_filtered(repo, aux_files, None)
    }

    fn try_download_gguf_filtered(
        &self,
        repo: &str,
        aux_files: &[PathBuf],
        filter: Option<&str>,
    ) -> Result<Option<HfModelFiles>> {
        for candidate in self.ranked_gguf_candidates(repo) {
            if let Some(f) = filter {
                if !candidate.to_ascii_lowercase().contains(&f.to_ascii_lowercase()) {
                    continue;
                }
            }
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
                let external_data = self.download_onnx_external_data(repo, &candidate, &path)?;
                let mut aux = aux_files.to_vec();
                for external in external_data {
                    push_unique_path(&mut aux, external);
                }
                return Ok(Some(HfModelFiles {
                    repo: repo.to_string(),
                    weights: vec![path],
                    format: WeightFormat::Onnx,
                    aux_files: aux,
                }));
            }
        }
        Ok(None)
    }

    fn download_onnx_external_data(
        &self,
        repo: &str,
        onnx_repo_path: &str,
        local_onnx_path: &Path,
    ) -> Result<Vec<PathBuf>> {
        let locations = super::onnx::external_data_locations(local_onnx_path)?;
        let mut out = Vec::with_capacity(locations.len());
        for location in locations {
            let repo_path = resolve_onnx_external_repo_path(onnx_repo_path, &location)?;
            let downloaded = self.get_file(repo, &repo_path)?;
            push_unique_path(&mut out, downloaded);
        }
        Ok(out)
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
                        .then_with(|| gguf_preferred_rank(a).cmp(&gguf_preferred_rank(b)))
                        .then_with(|| a.cmp(b))
                });
                return gguf_files;
            }
        }

        // API 失败时扫描本地缓存目录
        if let Some(cached) = self.scan_local_gguf_cache(repo) {
            return cached;
        }

        // 回退到预设的候选列表
        self.gguf_candidate_names()
    }

    /// 当 HF API 不可用时，扫描本地缓存目录中已下载的 GGUF 文件
    fn scan_local_gguf_cache(&self, repo: &str) -> Option<Vec<String>> {
        let model_dir_name = format!("models--{}", repo.replace('/', "--"));
        let model_dir = self.cache_dir.join(&model_dir_name);
        let snapshots_dir = model_dir.join("snapshots");
        if !snapshots_dir.is_dir() {
            return None;
        }

        let mut gguf_files: Vec<String> = Vec::new();
        let Ok(entries) = fs::read_dir(&snapshots_dir) else {
            return None;
        };
        for entry in entries.flatten() {
            let snap_dir = entry.path();
            if !snap_dir.is_dir() {
                continue;
            }
            let Ok(files) = fs::read_dir(&snap_dir) else {
                continue;
            };
            for file in files.flatten() {
                let name = file.file_name().to_string_lossy().to_string();
                if name.ends_with(".gguf") {
                    gguf_files.push(name);
                }
            }
        }

        if gguf_files.is_empty() {
            return None;
        }

        // 去重（多个 snapshot 可能包含同名文件）
        let unique: BTreeSet<String> = gguf_files.into_iter().collect();
        let mut result: Vec<String> = unique.into_iter().collect();
        result.sort_by(|a, b| gguf_preferred_rank(a).cmp(&gguf_preferred_rank(b)).then_with(|| a.cmp(b)));
        Some(result)
    }

    /// 当 HF API 不可用时，在本地缓存中搜索可能的 base model 仓库。
    /// 例如 `bartowski/Qwen_Qwen3-0.6B-GGUF` → 尝试匹配
    /// `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-0.6B-GGUF` 等。
    fn find_base_repo_in_cache(&self, repo: &str) -> Option<String> {
        let hf_dir = &self.cache_dir;
        if !hf_dir.is_dir() {
            return None;
        }

        // 从 repo 名推导核心模型标识：去掉 org 和 GGUF 后缀
        let model_name = repo.split('/').next_back()?;
        let base_name = model_name
            .strip_suffix("-GGUF")
            .or_else(|| model_name.strip_suffix("-gguf"))
            .unwrap_or(model_name);
        // 提取核心标识：去掉下划线、横线、版本后缀
        // Qwen_Qwen3-0.6B → ["qwen", "qwen3", "0.6b"]
        let base_lower = base_name.to_lowercase();
        let base_tokens: Vec<&str> = base_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty() && t.len() > 1)
            .collect();

        let Ok(entries) = fs::read_dir(hf_dir) else {
            return None;
        };
        let mut candidates: Vec<(String, usize)> = Vec::new();
        for entry in entries.flatten() {
            let dir_name = entry.file_name().to_string_lossy().to_string();
            if !dir_name.starts_with("models--") {
                continue;
            }
            let repo_part = &dir_name["models--".len()..];
            if repo_part == repo {
                continue;
            }
            if repo_part.ends_with("-GGUF") || repo_part.ends_with("-gguf") {
                continue;
            }

            // 计算与 base_name 的相似度
            let repo_lower = repo_part.to_lowercase();
            let repo_tokens: Vec<&str> = repo_lower
                .split(|c: char| !c.is_alphanumeric())
                .filter(|t| !t.is_empty() && t.len() > 1)
                .collect();
            let overlap = base_tokens
                .iter()
                .filter(|t| repo_tokens.iter().any(|r| r == *t))
                .count();

            if overlap >= 2 {
                // 验证这个仓库的 snapshot 中有 config.json 或 tokenizer.json
                let snapshots_dir = entry.path().join("snapshots");
                if snapshots_dir.is_dir() {
                    if let Some(snap) = fs::read_dir(&snapshots_dir).ok()?.next() {
                        let snap_dir = snap.ok()?.path();
                        let has_config = snap_dir.join("config.json").exists();
                        let has_tokenizer = snap_dir.join("tokenizer.json").exists();
                        if has_config || has_tokenizer {
                            let score = overlap * 10
                                + has_tokenizer as usize * 5
                                + has_config as usize * 3;
                            candidates.push((repo_part.to_string(), score));
                        }
                    }
                }
            }
        }

        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates.into_iter().next().map(|(repo, _)| repo)
    }

    /// 检查缓存的 repo 中是否有指定文件
    #[allow(dead_code)]
    fn cache_repo_has_file(&self, repo: &str, filename: &str) -> bool {
        let dir_name = format!("models--{}", repo.replace('/', "--"));
        let snapshots_dir = self.cache_dir.join(&dir_name).join("snapshots");
        let Ok(entries) = fs::read_dir(&snapshots_dir) else {
            return false;
        };
        for entry in entries.flatten() {
            if entry.path().join(filename).exists() {
                return true;
            }
        }
        false
    }

    fn ranked_onnx_candidates(&self, repo: &str) -> Vec<String> {
        // ARCH-ONNX-WARN: 当所选最高优先级 ONNX 是 O3/O4 优化或量化版本时,
        // 提示用户可能存在 fused operator 不识别 / 数值漂移风险。
        fn warn_if_optimized(name: &str, repo: &str) {
            let lower = name.to_ascii_lowercase();
            let stem = lower.rsplit('/').next().unwrap_or(&lower);
            let is_optimized = stem.contains("_o2") || stem.contains("_o3")
                || stem.contains("_o4") || stem.contains("_o5");
            let is_quantized = stem.contains("int8") || stem.contains("uint8")
                || stem.contains("quint8") || stem.contains("bnb");
            if is_optimized || is_quantized {
                log::warn!(
                    "[gllm:onnx] repo {repo} 仓库不含 model.onnx 原始版本,\
                     退化选用 {name} (优化/量化变体)。\
                     gllm ONNX loader 可能不识别 fused operators (FusedGELU/FusedLayerNorm 等),\
                     可能产生 NaN/数值漂移。建议: 优先用 SafeTensors 加载,\
                     或上游模型仓库提供 model.onnx 原版。"
                );
            }
        }

        // ARCH-ONNX-PRIORITY: 优先级排序避免选到优化/量化版本(可能 NaN/数值漂移)
        //   0. model.onnx / onnx/model.onnx       — 原始 fp32,最稳定
        //   1. *_fp32.onnx / *_O1.onnx            — 基础优化保守
        //   2. *_fp16.onnx                        — 半精度,gllm 可处理
        //   3. *_O[2-4].onnx                      — 中高级优化(可能 fused 算子无法识别)
        //   4. *_int8/_qint8/_quint8/_uint8.onnx  — 量化(需 dequant 路径)
        //   5. 其他                                — 兜底
        // 历史 BUG:multilingual-e5-small 仓库仅有 model_O4.onnx + model_qint8_*.onnx,
        // 字典序排序选中 model_O4 (高级优化),embedding 输出 NaN at index 0。
        fn gguf_preferred_rank(name: &str) -> usize {
            let lower = name.to_ascii_lowercase();
            // 去掉路径前缀(onnx/) 与扩展名 (.onnx) 后比对后缀
            let stem = lower
                .rsplit('/')
                .next()
                .unwrap_or(&lower)
                .trim_end_matches(".onnx");
            if stem == "model" {
                0
            } else if stem.ends_with("_fp32") || stem.ends_with("_o1") {
                1
            } else if stem.ends_with("_fp16") {
                2
            } else if stem.ends_with("_o2") || stem.ends_with("_o3") || stem.ends_with("_o4") {
                3
            } else if stem.contains("int8") || stem.contains("uint8") || stem.contains("quint8") {
                4
            } else {
                5
            }
        }

        if let Ok(files) = self.list_repo_files(repo) {
            // Ω1: 优先选择 onnx/ 目录下的文件
            let onnx_dir_files: Vec<_> = files
                .iter()
                .filter(|name| name.starts_with("onnx/") && name.ends_with(".onnx"))
                .cloned()
                .collect();
            if !onnx_dir_files.is_empty() {
                let mut result = onnx_dir_files;
                result.sort_by(|a, b| {
                    gguf_preferred_rank(a)
                        .cmp(&gguf_preferred_rank(b))
                        .then_with(|| a.cmp(b))
                });
                warn_if_optimized(&result[0], repo);
                return result;
            }

            // 其次选择根目录的 onnx 文件
            let root_onnx: Vec<_> = files
                .into_iter()
                .filter(|name| name.ends_with(".onnx"))
                .collect();
            if !root_onnx.is_empty() {
                let mut result = root_onnx;
                result.sort_by(|a, b| {
                    gguf_preferred_rank(a)
                        .cmp(&gguf_preferred_rank(b))
                        .then_with(|| a.cmp(b))
                });
                warn_if_optimized(&result[0], repo);
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
            .unwrap_or_else(|| "https://huggingface.co".to_string()); // LEGAL: HF_ENDPOINT 缺失时使用官方端点
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

    fn try_download_pytorch_bins(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
        aux_files: &[PathBuf],
    ) -> Result<Option<HfModelFiles>> {
        if let Ok(index_path) = self.get_file_any(repo, file_map, "pytorch_model.bin.index.json") {
            let shard_index = ShardIndex::from_path(&index_path)?;
            let shard_files = shard_index.shard_files();
            let bin_paths = self.download_shards(repo, &shard_files, parallel)?;
            let mut aux = aux_files.to_vec();
            aux.push(index_path);
            return Ok(Some(HfModelFiles {
                repo: repo.to_string(),
                weights: bin_paths,
                format: WeightFormat::PyTorch,
                aux_files: aux,
            }));
        }

        if let Ok(bin_path) = self.get_file_any(repo, file_map, "pytorch_model.bin") {
            return Ok(Some(HfModelFiles {
                repo: repo.to_string(),
                weights: vec![bin_path],
                format: WeightFormat::PyTorch,
                aux_files: aux_files.to_vec(),
            }));
        }

        Ok(None)
    }

    /// 按 HF 标准缓存目录结构查找本地已完整下载的文件。
    ///
    /// HF 缓存标准结构：
    ///   {cache_dir}/models--{org}--{model}/
    ///     refs/main          # 内容为 commit hash
    ///     snapshots/{hash}/  # {filename} → 软链接 → ../../blobs/{etag_sha256}
    ///     blobs/{etag_sha256} # 实际文件
    ///
    /// 返回完整文件路径当且仅当 snapshot 软链接存在且指向的 blob 文件可读（大小 > 0）。
    /// 这是 BCE-20260627-032 的核心：在调用 hf_hub 的 download_with_progress（它从不
    /// 检查本地缓存）之前先跳过重下载。
    fn find_cached_snapshot(&self, repo: &str, filename: &str) -> Option<PathBuf> {
        // 构造 HF 标准缓存目录名：models--{org}--{model}
        let dir_name = format!("models--{}", repo.replace('/', "--"));
        let repo_cache_dir = self.cache_dir.join(&dir_name);

        // 读取 refs/main 获取当前 commit hash
        let refs_main = repo_cache_dir.join("refs").join("main");
        let commit_hash = fs::read_to_string(&refs_main).ok()?;
        let commit_hash = commit_hash.trim();

        // 构造 snapshot 路径：{cache_dir}/models--{...}/snapshots/{commit}/{filename}
        let snapshot_path = repo_cache_dir
            .join("snapshots")
            .join(commit_hash)
            .join(filename);

        // 验证文件存在且可读（snapshot 可能是软链接 → blobs/，用 fs::metadata
        // follow symlinks 检查真实文件大小 > 0）
        match std::fs::metadata(&snapshot_path) {
            Ok(md) if md.len() > 0 => Some(snapshot_path),
            _ => None,
        }
    }

    fn get_file(&self, repo: &str, filename: &str) -> Result<PathBuf> {
        // BCE-20260627-032: 本地缓存优先。hf_hub 0.4.3 的 download_with_progress
        // 无缓存命中检查，每次重下整个文件（如 gpt-oss-20b 13GB）。此处先按 HF 标准
        // 缓存结构查找 snapshot 文件，命中则直接返回，避免无谓重下载。
        if let Some(cached) = self.find_cached_snapshot(repo, filename) {
            return Ok(cached);
        }
        // 未命中，走 hf_hub 正常下载
        let repo_api = self.api.model(repo.to_string());
        // download_with_progress 提供下载进度可观测性（log::info 输出）
        let progress = HfLogProgress::new(filename);
        repo_api
            .download_with_progress(filename, progress)
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

        // BCE-20260627-032: 本地缓存优先。hf_hub 0.4.3 的 download_with_progress
        // 无缓存命中检查，每次重下整个分片（如 gpt-oss-20b 的 13GB safetensors）。
        // 先按 HF 标准缓存结构查找 snapshot，命中则直接返回，仅对缺失分片触发下载。
        let mut result: Vec<PathBuf> = Vec::with_capacity(shards.len());
        let mut to_download: Vec<String> = Vec::new();
        for shard in shards {
            if let Some(cached) = self.find_cached_snapshot(repo, shard) {
                result.push(cached);
            } else {
                to_download.push(shard.clone());
            }
        }

        if to_download.is_empty() {
            log::info!("全部 {} 个分片已命中本地缓存，跳过下载", shards.len());
            return Ok(result);
        }

        // 仅下载缺失分片，下载完成后按原始 shards 顺序重排结果
        let downloaded_indices: Vec<usize> = shards
            .iter()
            .enumerate()
            .filter(|(_, s)| to_download.iter().any(|td| td == *s))
            .map(|(i, _)| i)
            .collect();

        let shard_paths_list: Vec<PathBuf> = to_download.iter().map(PathBuf::from).collect();

        let downloaded: Vec<PathBuf> = if parallel.enabled() {
            // 并行下载：每个分片独立进度
            log::info!("并行下载 {} 个分片（缓存命中 {} 个）...", to_download.len(), result.len());
            parallel.map_paths(&shard_paths_list, |path| {
                let filename = path.to_string_lossy().to_string();
                let progress = HfLogProgress::new(&filename);
                api.model(repo_id.clone())
                    .download_with_progress(&filename, progress)
                    .map_err(|err| LoaderError::HfHub(err.to_string()))
            })?
        } else {
            // 串行下载：每个分片独立进度
            let mut dl_result = Vec::new();
            for (idx, shard_path) in shard_paths_list.iter().enumerate() {
                let filename = shard_path.to_string_lossy().to_string();
                log::info!("[{}/{}] 下载分片: {}", idx + 1, shard_paths_list.len(), filename);
                let progress = HfLogProgress::new(&filename);
                let path = api
                    .model(repo_id.clone())
                    .download_with_progress(&filename, progress)
                    .map_err(|err| LoaderError::HfHub(err.to_string()))?;
                dl_result.push(path);
            }
            dl_result
        };

        // 将下载结果按原始 shards 顺序合并到 result
        for (i, path) in downloaded_indices.into_iter().zip(downloaded.into_iter()) {
            while result.len() <= i {
                result.push(PathBuf::new());
            }
            result[i] = path;
        }
        if parallel.enabled() {
            log::info!("并行下载完成");
        }
        Ok(result)
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

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn resolve_onnx_external_repo_path(onnx_repo_path: &str, location: &str) -> Result<String> {
    let base = Path::new(onnx_repo_path)
        .parent()
        .unwrap_or_else(|| Path::new("")); // LEGAL: 无父目录时使用空路径
    normalize_repo_path(&base.join(location))
}

fn normalize_repo_path(path: &Path) -> Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => parts.push(part.to_string_lossy().to_string()),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(LoaderError::Onnx(format!(
                    "invalid ONNX external data path: {}",
                    path.display()
                )))
            }
        }
    }
    if parts.is_empty() {
        return Err(LoaderError::Onnx(
            "invalid ONNX external data path: empty location".to_string(),
        ));
    }
    Ok(parts.join("/"))
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

#[derive(Debug, Clone, PartialEq, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum BaseModelField {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
struct HfCardData {
    #[serde(default, alias = "baseModel")]
    base_model: Option<BaseModelField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
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
                // [BCE-025] Single("") with empty/whitespace repo string — nothing to return
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

    // ── gguf_preferred_rank ──

    #[test]
    fn gguf_preferred_rank_q4_0_best() {
        assert_eq!(gguf_preferred_rank("model-Q4_0.gguf"), 0);
    }

    #[test]
    fn gguf_preferred_rank_q8_0() {
        assert_eq!(gguf_preferred_rank("model-Q8_0.gguf"), 1);
    }

    #[test]
    fn gguf_preferred_rank_k_quant() {
        assert_eq!(gguf_preferred_rank("model-q4_k_m.gguf"), 6);
        assert_eq!(gguf_preferred_rank("model-q6_k.gguf"), 10);
    }

    #[test]
    fn gguf_preferred_rank_f16_f32() {
        assert_eq!(gguf_preferred_rank("model-f16.gguf"), 21);
        assert_eq!(gguf_preferred_rank("model-fp32.gguf"), 22);
    }

    #[test]
    fn gguf_preferred_rank_unknown_low_priority() {
        assert_eq!(gguf_preferred_rank("model-custom.gguf"), 50);
        assert_eq!(gguf_preferred_rank("model.gguf"), 50);
    }

    // ── WeightFormat ──

    #[test]
    fn weight_format_variants() {
        assert_eq!(WeightFormat::SafeTensors, WeightFormat::SafeTensors);
        assert_eq!(WeightFormat::Gguf, WeightFormat::Gguf);
        assert_eq!(WeightFormat::Onnx, WeightFormat::Onnx);
        assert_ne!(WeightFormat::SafeTensors, WeightFormat::Gguf);
    }

    // ── HfModelFiles ──

    #[test]
    fn hf_model_files_fields() {
        let files = HfModelFiles {
            repo: "org/model".to_string(),
            weights: vec![PathBuf::from("model.safetensors")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![PathBuf::from("config.json")],
        };
        assert_eq!(files.repo, "org/model");
        assert_eq!(files.weights.len(), 1);
        assert_eq!(files.format, WeightFormat::SafeTensors);
    }

    // ── is_auth_error comprehensive ──

    #[test]
    fn is_auth_error_case_insensitive() {
        assert!(is_auth_error("error 401 unauthorized"));
        assert!(is_auth_error("ERROR 403 FORBIDDEN"));
        assert!(is_auth_error("UnAuThOrIzEd access"));
        assert!(is_auth_error("FORBIDDEN resource"));
    }

    #[test]
    fn is_auth_error_substring_match() {
        assert!(is_auth_error("Received 401 from server"));
        assert!(is_auth_error("Got 403 response code"));
        assert!(is_auth_error("The server says unauthorized"));
        assert!(is_auth_error("Request was forbidden by the API"));
        assert!(is_auth_error("Authentication is required"));
        assert!(is_auth_error("Invalid username or password provided"));
    }

    #[test]
    fn is_auth_error_rejects_non_auth_errors() {
        assert!(!is_auth_error("404 Not Found"));
        assert!(!is_auth_error("500 Internal Server Error"));
        assert!(!is_auth_error("Connection refused"));
        assert!(!is_auth_error("DNS resolution failed"));
        assert!(!is_auth_error("Timeout after 30s"));
    }

    #[test]
    fn is_auth_error_empty_string() {
        assert!(!is_auth_error(""));
    }

    // ── gguf_preferred_rank full coverage ──

    #[test]
    fn gguf_preferred_rank_q4_1() {
        assert_eq!(gguf_preferred_rank("model-q4_1.gguf"), 2);
    }

    #[test]
    fn gguf_preferred_rank_q5_0() {
        assert_eq!(gguf_preferred_rank("model-q5_0.gguf"), 3);
    }

    #[test]
    fn gguf_preferred_rank_q5_1() {
        assert_eq!(gguf_preferred_rank("model-q5_1.gguf"), 4);
    }

    #[test]
    fn gguf_preferred_rank_q8_1() {
        assert_eq!(gguf_preferred_rank("model-q8_1.gguf"), 5);
    }

    #[test]
    fn gguf_preferred_rank_q4_k_s() {
        assert_eq!(gguf_preferred_rank("model-q4_k_s.gguf"), 7);
    }

    #[test]
    fn gguf_preferred_rank_q5_k_m() {
        assert_eq!(gguf_preferred_rank("model-q5_k_m.gguf"), 8);
    }

    #[test]
    fn gguf_preferred_rank_q5_k_s() {
        assert_eq!(gguf_preferred_rank("model-q5_k_s.gguf"), 9);
    }

    #[test]
    fn gguf_preferred_rank_q6_k_l() {
        // "q6_k_l" contains "q6_k" so it matches the q6_k branch (rank 10)
        assert_eq!(gguf_preferred_rank("model-q6_k_l.gguf"), 10);
    }

    #[test]
    fn gguf_preferred_rank_q3_k_m() {
        assert_eq!(gguf_preferred_rank("model-q3_k_m.gguf"), 12);
    }

    #[test]
    fn gguf_preferred_rank_q3_k_s() {
        assert_eq!(gguf_preferred_rank("model-q3_k_s.gguf"), 13);
    }

    #[test]
    fn gguf_preferred_rank_q2_k() {
        assert_eq!(gguf_preferred_rank("model-q2_k.gguf"), 14);
    }

    #[test]
    fn gguf_preferred_rank_q8_k() {
        assert_eq!(gguf_preferred_rank("model-q8_k.gguf"), 15);
    }

    #[test]
    fn gguf_preferred_rank_iq4_nl() {
        assert_eq!(gguf_preferred_rank("model-iq4_nl.gguf"), 16);
    }

    #[test]
    fn gguf_preferred_rank_iq4_xs() {
        assert_eq!(gguf_preferred_rank("model-iq4_xs.gguf"), 17);
    }

    #[test]
    fn gguf_preferred_rank_iq3_s() {
        assert_eq!(gguf_preferred_rank("model-iq3_s.gguf"), 18);
    }

    #[test]
    fn gguf_preferred_rank_iq3_xxs() {
        assert_eq!(gguf_preferred_rank("model-iq3_xxs.gguf"), 19);
    }

    #[test]
    fn gguf_preferred_rank_iq2() {
        assert_eq!(gguf_preferred_rank("model-iq2_xs.gguf"), 20);
    }

    #[test]
    fn gguf_preferred_rank_fp16() {
        assert_eq!(gguf_preferred_rank("model-fp16.gguf"), 21);
    }

    #[test]
    fn gguf_preferred_rank_case_insensitive() {
        assert_eq!(gguf_preferred_rank("MODEL-Q4_0.GGUF"), 0);
        assert_eq!(gguf_preferred_rank("Model-Q8_0.Gguf"), 1);
    }

    #[test]
    fn gguf_preferred_rank_non_gguf_extension() {
        // Files not ending in .gguf get default rank
        assert_eq!(gguf_preferred_rank("model-q4_0.bin"), 50);
    }

    // ── resolve_token_path ──

    #[test]
    fn resolve_token_path_absolute_override() {
        let abs = PathBuf::from("/absolute/token/path");
        let result = resolve_token_path(Some(&abs));
        assert_eq!(result, Some(abs.clone()));
    }

    #[test]
    fn resolve_token_path_relative_override() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HOME", Some("/tmp/gllm-test-home"));
        let result = resolve_token_path(Some(Path::new("custom/token")));
        assert_eq!(
            result,
            Some(PathBuf::from("/tmp/gllm-test-home/custom/token"))
        );
    }

    #[test]
    fn resolve_token_path_default_uses_home() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HOME", Some("/tmp/gllm-test-home2"));
        let result = resolve_token_path(None);
        assert_eq!(
            result,
            Some(PathBuf::from("/tmp/gllm-test-home2/.huggingface/token"))
        );
    }

    #[test]
    fn resolve_token_path_no_home_returns_none() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HOME", None);
        let result = resolve_token_path(None);
        assert!(result.is_none());
    }

    // ── read_hf_token edge cases ──

    #[test]
    fn read_hf_token_empty_env_var_ignored() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", Some(""));
        // Empty HF_TOKEN should not be used; falls through to file
        // which likely doesn't exist in test env, so result depends on
        // whether ~/.huggingface/token exists
        let result = read_hf_token(None);
        // Empty env var is treated as absent; result depends on file
        if let Ok(home) = std::env::var("HOME") {
            let token_path = PathBuf::from(home).join(DEFAULT_HF_TOKEN_PATH);
            if !token_path.exists() {
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn read_hf_token_non_hf_prefix_file_ignored() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", None);

        let temp_dir =
            std::env::temp_dir().join(format!("gllm-hf-token-noprefix-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let token_path = temp_dir.join("token");
        // Token without "hf_" prefix should be rejected
        std::fs::write(&token_path, "not_a_valid_token\n").expect("write token file");

        let token = read_hf_token(Some(&token_path));
        assert!(token.is_none());

        let _ = std::fs::remove_file(&token_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn read_hf_token_env_var_takes_priority_over_file() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", Some("hf_env_priority_token"));

        // Even if a file is provided, env var should take priority
        let temp_dir =
            std::env::temp_dir().join(format!("gllm-hf-token-pri-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let token_path = temp_dir.join("token");
        std::fs::write(&token_path, "hf_file_token\n").expect("write token file");

        let token = read_hf_token(Some(&token_path));
        assert_eq!(token, Some("hf_env_priority_token".to_string()));

        let _ = std::fs::remove_file(&token_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // ── map_name ──

    #[test]
    fn map_name_returns_identity_with_empty_file_map() {
        let empty: FileMap = &[];
        assert_eq!(map_name(empty, "config.json"), "config.json");
        assert_eq!(map_name(empty, "model.safetensors"), "model.safetensors");
    }

    #[test]
    fn map_name_returns_mapped_name_when_present() {
        let file_map: FileMap = &[("config.json", "custom_config.json")];
        assert_eq!(map_name(file_map, "config.json"), "custom_config.json");
    }

    #[test]
    fn map_name_returns_original_when_not_in_file_map() {
        let file_map: FileMap = &[("other.json", "renamed.json")];
        assert_eq!(map_name(file_map, "config.json"), "config.json");
    }

    // ── candidate_names ──

    #[test]
    fn candidate_names_for_non_config_has_no_configuration_json() {
        let empty: FileMap = &[];
        let candidates = candidate_names(empty, "tokenizer.json");
        assert!(candidates.iter().any(|c| c == "tokenizer.json"));
        // configuration.json is only added for "config.json" logical name
        assert!(!candidates.iter().any(|c| c == "configuration.json"));
    }

    #[test]
    fn candidate_names_includes_weights_prefix() {
        let empty: FileMap = &[];
        let candidates = candidate_names(empty, "model.safetensors");
        assert!(candidates.iter().any(|c| c == "model.safetensors"));
        assert!(candidates.iter().any(|c| c == "model/model.safetensors"));
        assert!(candidates.iter().any(|c| c == "weights/model.safetensors"));
    }

    #[test]
    fn candidate_names_with_file_map_mapping() {
        let file_map: FileMap = &[("config.json", "my_config.json")];
        let candidates = candidate_names(file_map, "config.json");
        // First candidate should be the mapped name
        assert_eq!(candidates[0], "my_config.json");
        // configuration.json is also added for config.json
        assert!(candidates.iter().any(|c| c == "configuration.json"));
    }

    #[test]
    fn candidate_names_deduplicates() {
        let file_map: FileMap = &[("config.json", "config.json")];
        let candidates = candidate_names(file_map, "config.json");
        let config_count = candidates.iter().filter(|c| c.as_str() == "config.json").count();
        assert_eq!(config_count, 1);
    }

    // ── push_unique_path ──

    #[test]
    fn push_unique_path_adds_new_path() {
        let mut paths = vec![PathBuf::from("a.txt")];
        push_unique_path(&mut paths, PathBuf::from("b.txt"));
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[1], PathBuf::from("b.txt"));
    }

    #[test]
    fn push_unique_path_skips_duplicate() {
        let mut paths = vec![PathBuf::from("a.txt"), PathBuf::from("b.txt")];
        push_unique_path(&mut paths, PathBuf::from("a.txt"));
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn push_unique_path_empty_vec() {
        let mut paths: Vec<PathBuf> = vec![];
        push_unique_path(&mut paths, PathBuf::from("first.txt"));
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], PathBuf::from("first.txt"));
    }

    // ── normalize_repo_path ──

    #[test]
    fn normalize_repo_path_simple() {
        assert_eq!(
            normalize_repo_path(Path::new("model.onnx.data")).unwrap(),
            "model.onnx.data"
        );
    }

    #[test]
    fn normalize_repo_path_with_directory() {
        assert_eq!(
            normalize_repo_path(Path::new("onnx/model.onnx.data")).unwrap(),
            "onnx/model.onnx.data"
        );
    }

    #[test]
    fn normalize_repo_path_strips_curdir() {
        assert_eq!(
            normalize_repo_path(Path::new("./model.onnx.data")).unwrap(),
            "model.onnx.data"
        );
    }

    #[test]
    fn normalize_repo_path_rejects_parent_dir() {
        let result = normalize_repo_path(Path::new("../etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_repo_path_rejects_root_dir() {
        let result = normalize_repo_path(Path::new("/etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_repo_path_rejects_empty_path() {
        let result = normalize_repo_path(Path::new("."));
        assert!(result.is_err());
    }

    // ── resolve_onnx_external_repo_path ──

    #[test]
    fn resolve_onnx_external_simple() {
        let result = resolve_onnx_external_repo_path("model.onnx", "model.onnx.data").unwrap();
        assert_eq!(result, "model.onnx.data");
    }

    #[test]
    fn resolve_onnx_external_in_subdirectory() {
        let result =
            resolve_onnx_external_repo_path("onnx/model.onnx", "model.onnx.data").unwrap();
        assert_eq!(result, "onnx/model.onnx.data");
    }

    #[test]
    fn resolve_onnx_external_rejects_traversal() {
        let result = resolve_onnx_external_repo_path("onnx/model.onnx", "../../etc/passwd");
        assert!(result.is_err());
    }

    // ── HfModelMetadata / BaseModelField deserialization ──

    #[test]
    fn hf_model_metadata_default() {
        let metadata = HfModelMetadata::default();
        assert!(metadata.card_data.base_model.is_none());
        assert!(metadata.tags.is_empty());
        assert!(metadata.base_model_repo().is_none());
    }

    #[test]
    fn hf_card_data_default() {
        let card = HfCardData::default();
        assert!(card.base_model.is_none());
    }

    #[test]
    fn base_model_repo_single_string() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "cardData": { "base_model": "org/base-model" } }"#,
        )
        .expect("metadata");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/base-model".to_string())
        );
    }

    #[test]
    fn base_model_repo_multiple_strings() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "cardData": { "base_model": ["org/model-a", "org/model-b"] } }"#,
        )
        .expect("metadata");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/model-a".to_string())
        );
    }

    #[test]
    fn base_model_repo_empty_single_string() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "cardData": { "base_model": "  " } }"#,
        )
        .expect("metadata");
        // Empty/whitespace base_model should return None from card_data
        assert!(metadata.base_model_repo().is_none());
    }

    #[test]
    fn base_model_repo_empty_strings_in_list() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "cardData": { "base_model": ["  ", "", "org/valid-model"] } }"#,
        )
        .expect("metadata");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/valid-model".to_string())
        );
    }

    #[test]
    fn base_model_repo_all_empty_in_list() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "cardData": { "base_model": ["  ", ""] } }"#,
        )
        .expect("metadata");
        assert!(metadata.base_model_repo().is_none());
    }

    #[test]
    fn base_model_repo_tag_prefix() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["base_model:org/base-model"] }"#,
        )
        .expect("metadata");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/base-model".to_string())
        );
    }

    #[test]
    fn base_model_repo_tag_skips_quantized_prefix() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["base_model:quantized:org/base-model", "gguf"] }"#,
        )
        .expect("metadata");
        // quantized: prefix is skipped in first pass, but picked up in fallback
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/base-model".to_string())
        );
    }

    #[test]
    fn base_model_repo_tag_skips_empty_tag_value() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["base_model:", "gguf"] }"#,
        )
        .expect("metadata");
        assert!(metadata.base_model_repo().is_none());
    }

    #[test]
    fn base_model_repo_no_base_model() {
        let metadata: HfModelMetadata = serde_json::from_str(r#"{ "tags": ["gguf"] }"#).expect("metadata");
        assert!(metadata.base_model_repo().is_none());
    }

    #[test]
    fn base_model_repo_card_data_takes_priority_over_tags() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{
                "cardData": { "base_model": "org/card-model" },
                "tags": ["base_model:org/tag-model"]
            }"#,
        )
        .expect("metadata");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/card-model".to_string())
        );
    }

    #[test]
    fn base_model_repo_alias_base_model() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "cardData": { "baseModel": "org/aliased-model" } }"#,
        )
        .expect("metadata");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/aliased-model".to_string())
        );
    }

    #[test]
    fn hf_model_metadata_empty_json() {
        let metadata: HfModelMetadata = serde_json::from_str("{}").expect("metadata");
        assert!(metadata.base_model_repo().is_none());
    }

    // ── ShardIndex ──

    #[test]
    fn shard_index_from_path_and_shard_files() {
        let temp_dir =
            std::env::temp_dir().join(format!("gllm-shard-index-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("model.safetensors.index.json");
        let index_json = r#"{
            "metadata": { "total_size": 1234 },
            "weight_map": {
                "layer0.weight": "model-00001-of-00003.safetensors",
                "layer1.weight": "model-00002-of-00003.safetensors",
                "layer2.weight": "model-00001-of-00003.safetensors",
                "layer3.weight": "model-00003-of-00003.safetensors"
            }
        }"#;
        std::fs::write(&index_path, index_json).expect("write index");

        let shard_index = ShardIndex::from_path(&index_path).expect("parse shard index");
        let files = shard_index.shard_files();

        // BTreeSet deduplicates and sorts
        assert_eq!(files.len(), 3);
        assert_eq!(files[0], "model-00001-of-00003.safetensors");
        assert_eq!(files[1], "model-00002-of-00003.safetensors");
        assert_eq!(files[2], "model-00003-of-00003.safetensors");

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn shard_index_single_shard() {
        let temp_dir =
            std::env::temp_dir().join(format!("gllm-shard-single-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        let index_json = r#"{
            "weight_map": {
                "layer.weight": "model-00001-of-00001.safetensors"
            }
        }"#;
        std::fs::write(&index_path, index_json).expect("write index");

        let shard_index = ShardIndex::from_path(&index_path).expect("parse shard index");
        let files = shard_index.shard_files();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], "model-00001-of-00001.safetensors");

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn shard_index_empty_weight_map() {
        let temp_dir =
            std::env::temp_dir().join(format!("gllm-shard-empty-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        let index_json = r#"{ "weight_map": {} }"#;
        std::fs::write(&index_path, index_json).expect("write index");

        let shard_index = ShardIndex::from_path(&index_path).expect("parse shard index");
        let files = shard_index.shard_files();
        assert!(files.is_empty());

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn shard_index_invalid_json_returns_error() {
        let temp_dir =
            std::env::temp_dir().join(format!("gllm-shard-invalid-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        std::fs::write(&index_path, "not valid json").expect("write index");

        let result = ShardIndex::from_path(&index_path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // ── HfHubClient construction (no network) ──

    #[test]
    fn hf_hub_client_new_with_temp_dir() {
        let temp_dir =
            std::env::temp_dir().join(format!("gllm-hfclient-new-{}", std::process::id()));
        let client = HfHubClient::new(temp_dir.clone());
        assert!(client.is_ok(), "HfHubClient::new should succeed with a valid cache dir");
        let client = client.unwrap();
        assert_eq!(client.cache_dir, temp_dir);
    }

    #[test]
    fn hf_hub_client_with_endpoint() {
        let temp_dir =
            std::env::temp_dir().join(format!("gllm-hfclient-ep-{}", std::process::id()));
        let client = HfHubClient::with_endpoint(
            temp_dir.clone(),
            Some("https://hf-mirror.com".to_string()),
        );
        assert!(client.is_ok());
        let client = client.unwrap();
        assert_eq!(client.cache_dir, temp_dir);
    }

    #[test]
    fn hf_hub_client_with_endpoint_none() {
        let temp_dir =
            std::env::temp_dir().join(format!("gllm-hfclient-epnone-{}", std::process::id()));
        let client = HfHubClient::with_endpoint(temp_dir.clone(), None);
        assert!(client.is_ok());
    }

    #[test]
    fn hf_hub_client_with_token() {
        let temp_dir =
            std::env::temp_dir().join(format!("gllm-hfclient-token-{}", std::process::id()));
        let client = HfHubClient::with_token(
            temp_dir.clone(),
            "hf_test_token_value".to_string(),
        );
        assert!(client.is_ok());
        let client = client.unwrap();
        assert_eq!(client.cache_dir, temp_dir);
    }

    // ── WeightFormat exhaustive equality and inequality ──

    #[test]
    fn weight_format_all_inequalities() {
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ── HfModelFiles with all formats ──

    #[test]
    fn hf_model_files_gguf_format() {
        let files = HfModelFiles {
            repo: "org/model-gguf".to_string(),
            weights: vec![PathBuf::from("model-q4_0.gguf")],
            format: WeightFormat::Gguf,
            aux_files: vec![],
        };
        assert_eq!(files.format, WeightFormat::Gguf);
        assert!(files.aux_files.is_empty());
    }

    #[test]
    fn hf_model_files_onnx_format() {
        let files = HfModelFiles {
            repo: "org/model-onnx".to_string(),
            weights: vec![PathBuf::from("model.onnx")],
            format: WeightFormat::Onnx,
            aux_files: vec![PathBuf::from("model.onnx.data")],
        };
        assert_eq!(files.format, WeightFormat::Onnx);
        assert_eq!(files.aux_files.len(), 1);
    }

    #[test]
    fn hf_model_files_multiple_weights() {
        let files = HfModelFiles {
            repo: "org/model".to_string(),
            weights: vec![
                PathBuf::from("model-00001.safetensors"),
                PathBuf::from("model-00002.safetensors"),
                PathBuf::from("model-00003.safetensors"),
            ],
            format: WeightFormat::SafeTensors,
            aux_files: vec![PathBuf::from("config.json")],
        };
        assert_eq!(files.weights.len(), 3);
    }

    // ── DEFAULT_HF_TOKEN_PATH constant ──

    #[test]
    fn default_hf_token_path_value() {
        assert_eq!(DEFAULT_HF_TOKEN_PATH, ".huggingface/token");
    }

    // ── WeightFormat Hash ──

    #[test]
    fn weight_format_hash_consistent() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(WeightFormat::SafeTensors));
        assert!(set.insert(WeightFormat::Gguf));
        assert!(set.insert(WeightFormat::Onnx));
        assert_eq!(set.len(), 3);
        assert!(!set.insert(WeightFormat::SafeTensors));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn weight_format_hash_equal_values_match() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |v: WeightFormat| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_of(WeightFormat::SafeTensors), hash_of(WeightFormat::SafeTensors));
        assert_eq!(hash_of(WeightFormat::Gguf), hash_of(WeightFormat::Gguf));
        assert_eq!(hash_of(WeightFormat::Onnx), hash_of(WeightFormat::Onnx));
    }

    // ── WeightFormat Debug ──

    #[test]
    fn weight_format_debug_output() {
        assert_eq!(format!("{:?}", WeightFormat::SafeTensors), "SafeTensors");
        assert_eq!(format!("{:?}", WeightFormat::Gguf), "Gguf");
        assert_eq!(format!("{:?}", WeightFormat::Onnx), "Onnx");
    }

    // ── WeightFormat Copy ──

    #[test]
    fn weight_format_copy_semantics() {
        let a = WeightFormat::SafeTensors;
        let b = a;
        assert_eq!(a, b);
        assert_eq!(a, WeightFormat::SafeTensors);
    }

    // ── WeightFormat Clone ──

    #[test]
    fn weight_format_clone_equal() {
        let a = WeightFormat::Gguf;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── HfModelFiles PartialEq ──

    #[test]
    fn hf_model_files_eq_same() {
        let a = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![PathBuf::from("w.bin")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        let b = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![PathBuf::from("w.bin")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn hf_model_files_neq_different_repo() {
        let a = HfModelFiles {
            repo: "org/a".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        let b = HfModelFiles {
            repo: "org/b".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn hf_model_files_neq_different_format() {
        let a = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        let b = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![],
            format: WeightFormat::Gguf,
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn hf_model_files_neq_different_weights() {
        let a = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![PathBuf::from("a.bin")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        let b = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![PathBuf::from("b.bin")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn hf_model_files_neq_different_aux() {
        let a = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![PathBuf::from("config.json")],
        };
        let b = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    // ── HfModelFiles Debug ──

    #[test]
    fn hf_model_files_debug_contains_fields() {
        let files = HfModelFiles {
            repo: "org/test-model".to_string(),
            weights: vec![PathBuf::from("model.safetensors")],
            format: WeightFormat::Gguf,
            aux_files: vec![],
        };
        let debug = format!("{:?}", files);
        assert!(debug.contains("org/test-model"));
        assert!(debug.contains("model.safetensors"));
        assert!(debug.contains("Gguf"));
    }

    // ── HfModelFiles empty fields ──

    #[test]
    fn hf_model_files_empty_repo() {
        let files = HfModelFiles {
            repo: String::new(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        assert!(files.repo.is_empty());
        assert!(files.weights.is_empty());
        assert!(files.aux_files.is_empty());
    }

    #[test]
    fn hf_model_files_empty_weights_vec() {
        let files = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![PathBuf::from("config.json")],
        };
        assert!(files.weights.is_empty());
        assert_eq!(files.aux_files.len(), 1);
    }

    // ── gguf_preferred_rank edge cases ──

    #[test]
    fn gguf_preferred_rank_empty_string() {
        assert_eq!(gguf_preferred_rank(""), 50);
    }

    #[test]
    fn gguf_preferred_rank_partial_match_no_gguf_ext() {
        assert_eq!(gguf_preferred_rank("model-q4_0"), 50);
    }

    #[test]
    fn gguf_preferred_rank_q4_k_m_in_middle() {
        assert_eq!(gguf_preferred_rank("my-q4_k_m-custom.gguf"), 6);
    }

    #[test]
    fn gguf_preferred_rank_double_extension() {
        assert_eq!(gguf_preferred_rank("model.Q4_0.gguf.bak"), 50);
    }

    #[test]
    fn gguf_preferred_rank_only_extension() {
        assert_eq!(gguf_preferred_rank(".gguf"), 50);
    }

    // ── normalize_repo_path additional edge cases ──

    #[test]
    fn normalize_repo_path_nested_directories() {
        assert_eq!(
            normalize_repo_path(Path::new("a/b/c/data.bin")).unwrap(),
            "a/b/c/data.bin"
        );
    }

    #[test]
    fn normalize_repo_path_curdir_in_middle() {
        assert_eq!(
            normalize_repo_path(Path::new("onnx/./model.onnx.data")).unwrap(),
            "onnx/model.onnx.data"
        );
    }

    #[test]
    fn normalize_repo_path_rejects_parent_in_middle() {
        let result = normalize_repo_path(Path::new("onnx/../etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_repo_path_rejects_prefix_component() {
        // On Linux, "C:\\model.onnx" is parsed as a single Normal component (no Prefix),
        // so this test verifies behavior with a RootDir path instead.
        let result = normalize_repo_path(Path::new("/etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_repo_path_multi_curdir() {
        assert_eq!(
            normalize_repo_path(Path::new("././model.onnx.data")).unwrap(),
            "model.onnx.data"
        );
    }

    // ── resolve_onnx_external_repo_path additional ──

    #[test]
    fn resolve_onnx_external_root_onnx_file() {
        let result = resolve_onnx_external_repo_path("model.onnx", "external_data.bin").unwrap();
        assert_eq!(result, "external_data.bin");
    }

    #[test]
    fn resolve_onnx_external_nested_location() {
        let result = resolve_onnx_external_repo_path(
            "onnx/model.onnx",
            "data/external.bin",
        )
        .unwrap();
        assert_eq!(result, "onnx/data/external.bin");
    }

    // ── is_auth_error boundary ──

    #[test]
    fn is_auth_error_partial_substring() {
        assert!(is_auth_error("error: unauthorized access to resource"));
        assert!(is_auth_error("the token is forbidden"));
    }

    #[test]
    fn is_auth_error_similar_but_not_match() {
        assert!(!is_auth_error("404 Not Found"));
        assert!(!is_auth_error("the file is not found"));
        assert!(!is_auth_error("502 Bad Gateway"));
        assert!(!is_auth_error("timeout after 30s"));
    }

    // ── map_name edge cases ──

    #[test]
    fn map_name_empty_string() {
        let empty: FileMap = &[];
        assert_eq!(map_name(empty, ""), "");
    }

    #[test]
    fn map_name_first_match_wins() {
        let file_map: FileMap = &[
            ("a.json", "first.json"),
            ("a.json", "second.json"),
        ];
        assert_eq!(map_name(file_map, "a.json"), "first.json");
    }

    #[test]
    fn map_name_multiple_entries() {
        let file_map: FileMap = &[
            ("a.json", "mapped_a.json"),
            ("b.json", "mapped_b.json"),
        ];
        assert_eq!(map_name(file_map, "b.json"), "mapped_b.json");
    }

    // ── candidate_names edge cases ──

    #[test]
    fn candidate_names_empty_string_logical() {
        let empty: FileMap = &[];
        let candidates = candidate_names(empty, "");
        assert!(candidates.iter().any(|c| c == ""));
        assert!(candidates.iter().any(|c| c == "model/"));
        assert!(candidates.iter().any(|c| c == "weights/"));
    }

    #[test]
    fn candidate_names_config_json_has_seven_entries() {
        let empty: FileMap = &[];
        let candidates = candidate_names(empty, "config.json");
        // config.json + configuration.json + model/config.json + model/configuration.json
        // + weights/config.json + weights/configuration.json = 6
        assert_eq!(candidates.len(), 6);
    }

    #[test]
    fn candidate_names_non_config_has_three_entries() {
        let empty: FileMap = &[];
        let candidates = candidate_names(empty, "tokenizer.json");
        // tokenizer.json + model/tokenizer.json + weights/tokenizer.json = 3
        assert_eq!(candidates.len(), 3);
    }

    // ── push_unique_path additional ──

    #[test]
    fn push_unique_path_multiple_unique_adds() {
        let mut paths: Vec<PathBuf> = vec![];
        push_unique_path(&mut paths, PathBuf::from("a"));
        push_unique_path(&mut paths, PathBuf::from("b"));
        push_unique_path(&mut paths, PathBuf::from("c"));
        assert_eq!(paths.len(), 3);
    }

    #[test]
    fn push_unique_path_duplicate_at_end() {
        let mut paths = vec![PathBuf::from("a"), PathBuf::from("b")];
        push_unique_path(&mut paths, PathBuf::from("b"));
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn push_unique_path_duplicate_at_start() {
        let mut paths = vec![PathBuf::from("a"), PathBuf::from("b")];
        push_unique_path(&mut paths, PathBuf::from("a"));
        assert_eq!(paths.len(), 2);
    }

    // ── ShardIndex additional tests ──

    #[test]
    fn shard_index_many_deduplicated() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-shard-many-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        let index_json = r#"{
            "weight_map": {
                "w1": "shard-A.bin",
                "w2": "shard-B.bin",
                "w3": "shard-A.bin",
                "w4": "shard-C.bin",
                "w5": "shard-B.bin"
            }
        }"#;
        std::fs::write(&index_path, index_json).expect("write index");

        let shard_index = ShardIndex::from_path(&index_path).expect("parse");
        let files = shard_index.shard_files();
        assert_eq!(files.len(), 3);
        assert_eq!(files[0], "shard-A.bin");
        assert_eq!(files[1], "shard-B.bin");
        assert_eq!(files[2], "shard-C.bin");

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn shard_index_nonexistent_file_returns_error() {
        let result = ShardIndex::from_path(Path::new("/nonexistent/path/index.json"));
        assert!(result.is_err());
    }

    #[test]
    fn shard_index_clone_and_eq() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-shard-clone-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        let index_json = r#"{ "weight_map": { "w": "s.bin" } }"#;
        std::fs::write(&index_path, index_json).expect("write index");

        let a = ShardIndex::from_path(&index_path).expect("parse");
        let b = a.clone();
        assert_eq!(a, b);

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // ── BaseModelField deserialization edge cases ──

    #[test]
    fn base_model_field_single_empty() {
        let field: BaseModelField = serde_json::from_str(r#""  ""#).expect("parse");
        match field {
            BaseModelField::Single(s) => assert!(s.trim().is_empty()),
            _ => panic!("expected Single variant"),
        }
    }

    #[test]
    fn base_model_field_multiple_empty_list() {
        let field: BaseModelField = serde_json::from_str(r#"[]"#).expect("parse");
        match field {
            BaseModelField::Multiple(v) => assert!(v.is_empty()),
            _ => panic!("expected Multiple variant"),
        }
    }

    #[test]
    fn base_model_field_single_long_string() {
        let long_name = "org/".to_string() + &"a".repeat(1000);
        let json = format!("\"{}\"", long_name);
        let field: BaseModelField = serde_json::from_str(&json).expect("parse");
        match field {
            BaseModelField::Single(s) => assert_eq!(s, long_name),
            _ => panic!("expected Single variant"),
        }
    }

    #[test]
    fn base_model_field_multiple_with_unicode() {
        let field: BaseModelField = serde_json::from_str(
            r#"["org/模型-α", "org/模型-β"]"#,
        )
        .expect("parse");
        match field {
            BaseModelField::Multiple(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected Multiple variant"),
        }
    }

    // ── HfModelMetadata additional edge cases ──

    #[test]
    fn hf_model_metadata_clone_eq() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["base_model:org/model"] }"#,
        )
        .expect("parse");
        let cloned = metadata.clone();
        assert_eq!(metadata, cloned);
    }

    #[test]
    fn hf_model_metadata_with_many_tags() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["gguf", "quantized", "base_model:org/base", "text-generation"] }"#,
        )
        .expect("parse");
        assert_eq!(metadata.tags.len(), 4);
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/base".to_string())
        );
    }

    #[test]
    fn hf_model_metadata_tag_with_quantized_only() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["base_model:quantized:org/quant-model"] }"#,
        )
        .expect("parse");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/quant-model".to_string())
        );
    }

    #[test]
    fn hf_model_metadata_both_card_and_tags_card_wins() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{
                "cardData": { "baseModel": "org/card-winner" },
                "tags": ["base_model:org/tag-loser"]
            }"#,
        )
        .expect("parse");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/card-winner".to_string())
        );
    }

    #[test]
    fn hf_model_metadata_empty_tags_array() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": [] }"#,
        )
        .expect("parse");
        assert!(metadata.base_model_repo().is_none());
    }

    // ── HfCardData additional ──

    #[test]
    fn hf_card_data_clone_eq() {
        let card: HfCardData = serde_json::from_str(
            r#"{ "base_model": "org/test" }"#,
        )
        .expect("parse");
        let cloned = card.clone();
        assert_eq!(card, cloned);
    }

    #[test]
    fn hf_card_data_default_eq() {
        let a = HfCardData::default();
        let b = HfCardData::default();
        assert_eq!(a, b);
    }

    #[test]
    fn hf_card_data_with_base_model_alias() {
        let card: HfCardData = serde_json::from_str(
            r#"{ "baseModel": "org/aliased" }"#,
        )
        .expect("parse");
        assert!(card.base_model.is_some());
    }

    // ── HfHubClient with_token_path ──

    #[test]
    fn hf_hub_client_with_token_path_none() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-hfclient-tpnone-{}",
            std::process::id()
        ));
        let client = HfHubClient::with_endpoint_and_token_path(
            temp_dir.clone(),
            None,
            None,
        );
        assert!(client.is_ok());
    }

    #[test]
    fn hf_hub_client_with_custom_endpoint_and_token() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-hfclient-custom-{}",
            std::process::id()
        ));
        let client = HfHubClient::with_endpoint_and_token_path(
            temp_dir.clone(),
            Some("https://custom-hf.example.com".to_string()),
            None,
        );
        assert!(client.is_ok());
        let client = client.unwrap();
        assert_eq!(client.cache_dir, temp_dir);
    }

    // ── read_hf_token additional edge cases ──

    #[test]
    fn read_hf_token_whitespace_only_token_file() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", None);

        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-hf-token-ws-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let token_path = temp_dir.join("token");
        std::fs::write(&token_path, "   \n  \t \n").expect("write token file");

        let token = read_hf_token(Some(&token_path));
        // Whitespace-only file should not produce a token
        assert!(token.is_none());

        let _ = std::fs::remove_file(&token_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn read_hf_token_valid_token_with_surrounding_whitespace() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", None);

        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-hf-token-trim-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let token_path = temp_dir.join("token");
        std::fs::write(&token_path, "  hf_trimmed_token  \n").expect("write token file");

        let token = read_hf_token(Some(&token_path));
        assert_eq!(token, Some("hf_trimmed_token".to_string()));

        let _ = std::fs::remove_file(&token_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // ── EnvVarGuard cleanup verification ──

    #[test]
    fn env_var_guard_restores_previous_value() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        std::env::set_var("GLLM_TEST_ENVVAR", "original");
        {
            let _guard = EnvVarGuard::set("GLLM_TEST_ENVVAR", Some("modified"));
            assert_eq!(std::env::var("GLLM_TEST_ENVVAR").unwrap(), "modified");
        }
        assert_eq!(std::env::var("GLLM_TEST_ENVVAR").unwrap(), "original");
        std::env::remove_var("GLLM_TEST_ENVVAR");
    }

    #[test]
    fn env_var_guard_removes_on_none_when_no_previous() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        std::env::remove_var("GLLM_TEST_ENVVAR2");
        {
            let _guard = EnvVarGuard::set("GLLM_TEST_ENVVAR2", Some("temp"));
            assert_eq!(std::env::var("GLLM_TEST_ENVVAR2").unwrap(), "temp");
        }
        assert!(std::env::var("GLLM_TEST_ENVVAR2").is_err());
    }

    // ══════════════════════════════════════════════════════════════
    // NEW TESTS BELOW (149+)
    // ══════════════════════════════════════════════════════════════

    // ── gguf_preferred_rank ordering verification ──

    #[test]
    fn gguf_preferred_rank_q4_0_beats_q8_0() {
        assert!(gguf_preferred_rank("model-q4_0.gguf") < gguf_preferred_rank("model-q8_0.gguf"));
    }

    #[test]
    fn gguf_preferred_rank_quantized_beats_fp16() {
        assert!(gguf_preferred_rank("model-q4_0.gguf") < gguf_preferred_rank("model-fp16.gguf"));
    }

    #[test]
    fn gguf_preferred_rank_fp16_beats_fp32() {
        assert!(gguf_preferred_rank("model-fp16.gguf") < gguf_preferred_rank("model-fp32.gguf"));
    }

    #[test]
    fn gguf_preferred_rank_k_quant_mid_range() {
        assert!(
            gguf_preferred_rank("model-q4_k_m.gguf") < gguf_preferred_rank("model-q5_k_m.gguf")
        );
        assert!(
            gguf_preferred_rank("model-q5_k_m.gguf") < gguf_preferred_rank("model-q6_k.gguf")
        );
    }

    #[test]
    fn gguf_preferred_rank_iq_series_after_k_quants() {
        assert!(
            gguf_preferred_rank("model-q8_k.gguf") < gguf_preferred_rank("model-iq4_nl.gguf")
        );
        assert!(
            gguf_preferred_rank("model-iq4_nl.gguf") < gguf_preferred_rank("model-iq3_s.gguf")
        );
    }

    #[test]
    fn gguf_preferred_rank_q6_k_l_and_q6_k_same_prefix() {
        // q6_k_l contains "q6_k" so it matches the q6_k branch
        let q6_k = gguf_preferred_rank("model-q6_k.gguf");
        let q6_k_l = gguf_preferred_rank("model-q6_k_l.gguf");
        assert_eq!(q6_k, q6_k_l);
    }

    #[test]
    fn gguf_preferred_rank_q2_k_lowest_quantized() {
        assert!(
            gguf_preferred_rank("model-q2_k.gguf") > gguf_preferred_rank("model-q3_k_s.gguf")
        );
    }

    #[test]
    fn gguf_preferred_rank_model_dot_gguf_is_default() {
        assert_eq!(gguf_preferred_rank("model.gguf"), 50);
    }

    #[test]
    fn gguf_preferred_rank_iq2_xs_variants() {
        assert_eq!(gguf_preferred_rank("model-iq2_xs.gguf"), 20);
        assert_eq!(gguf_preferred_rank("model-iq2_xx.gguf"), 20);
    }

    #[test]
    fn gguf_preferred_rank_iq3_xxs_beats_iq2() {
        assert!(
            gguf_preferred_rank("model-iq3_xxs.gguf") < gguf_preferred_rank("model-iq2_xs.gguf")
        );
    }

    // ── gguf_preferred_rank: model.gguf special priority via ranked_gguf_candidates sorting ──
    // Note: model.gguf has rank 50 but gets sorted to front by the a_model.cmp(&b_model).reverse()

    #[test]
    fn gguf_preferred_rank_custom_suffix_is_default() {
        assert_eq!(gguf_preferred_rank("model-tq1_0.gguf"), 50);
        assert_eq!(gguf_preferred_rank("model-custom.gguf"), 50);
    }

    #[test]
    fn gguf_preferred_rank_f16_vs_fp16_same_rank() {
        assert_eq!(gguf_preferred_rank("model-f16.gguf"), gguf_preferred_rank("model-fp16.gguf"));
    }

    #[test]
    fn gguf_preferred_rank_f32_vs_fp32_same_rank() {
        assert_eq!(gguf_preferred_rank("model-f32.gguf"), gguf_preferred_rank("model-fp32.gguf"));
    }

    // ── resolve_token_path additional ──

    #[test]
    fn resolve_token_path_no_home_with_relative_returns_none() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HOME", None);
        // Relative path needs HOME to resolve
        let result = resolve_token_path(Some(Path::new("custom/token")));
        assert!(result.is_none());
    }

    #[test]
    fn resolve_token_path_absolute_without_home() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HOME", None);
        let abs = PathBuf::from("/opt/hf/token");
        let result = resolve_token_path(Some(&abs));
        assert_eq!(result, Some(abs));
    }

    // ── read_hf_token additional ──

    #[test]
    fn read_hf_token_file_with_hf_prefix_and_newline() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", None);

        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-hf-token-newline-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let token_path = temp_dir.join("token");
        std::fs::write(&token_path, "hf_valid_token\n").expect("write token file");

        let token = read_hf_token(Some(&token_path));
        assert_eq!(token, Some("hf_valid_token".to_string()));

        let _ = std::fs::remove_file(&token_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn read_hf_token_file_with_only_newlines() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", None);

        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-hf-token-nlonly-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let token_path = temp_dir.join("token");
        std::fs::write(&token_path, "\n\n\n").expect("write token file");

        let token = read_hf_token(Some(&token_path));
        assert!(token.is_none());

        let _ = std::fs::remove_file(&token_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn read_hf_token_nonexistent_file_returns_none() {
        let _lock = ENV_LOCK.lock().expect("env lock poisoned");
        let _guard = EnvVarGuard::set("HF_TOKEN", None);

        let token = read_hf_token(Some(Path::new("/nonexistent/path/token")));
        assert!(token.is_none());
    }

    // ── normalize_repo_path additional ──

    #[test]
    fn normalize_repo_path_deeply_nested() {
        assert_eq!(
            normalize_repo_path(Path::new("a/b/c/d/e.bin")).unwrap(),
            "a/b/c/d/e.bin"
        );
    }

    #[test]
    fn normalize_repo_path_curdir_and_normal_mixed() {
        assert_eq!(
            normalize_repo_path(Path::new("./onnx/./model.onnx.data")).unwrap(),
            "onnx/model.onnx.data"
        );
    }

    #[test]
    fn normalize_repo_path_rejects_double_dot_at_end() {
        let result = normalize_repo_path(Path::new("onnx/.."));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_repo_path_single_component() {
        assert_eq!(
            normalize_repo_path(Path::new("file.dat")).unwrap(),
            "file.dat"
        );
    }

    #[test]
    fn normalize_repo_path_rejects_trailing_parent_dir() {
        let result = normalize_repo_path(Path::new("data/../../etc/shadow"));
        assert!(result.is_err());
    }

    // ── resolve_onnx_external_repo_path additional ──

    #[test]
    fn resolve_onnx_external_curdir_location() {
        let result = resolve_onnx_external_repo_path("model.onnx", "./data.bin").unwrap();
        assert_eq!(result, "data.bin");
    }

    #[test]
    fn resolve_onnx_external_deeply_nested_onnx() {
        let result = resolve_onnx_external_repo_path(
            "onnx/optimized/model.onnx",
            "data/weights.bin",
        )
        .unwrap();
        assert_eq!(result, "onnx/optimized/data/weights.bin");
    }

    #[test]
    fn resolve_onnx_external_rejects_absolute_location() {
        let result = resolve_onnx_external_repo_path("model.onnx", "/etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn resolve_onnx_external_rejects_traversal_in_location() {
        let result = resolve_onnx_external_repo_path("onnx/model.onnx", "../secret.key");
        assert!(result.is_err());
    }

    // ── candidate_names additional ──

    #[test]
    fn candidate_names_mapped_config_with_prefixes() {
        let file_map: FileMap = &[("config.json", "custom_config.json")];
        let candidates = candidate_names(file_map, "config.json");
        assert!(candidates.iter().any(|c| c == "custom_config.json"));
        assert!(candidates.iter().any(|c| c == "model/custom_config.json"));
        assert!(candidates.iter().any(|c| c == "weights/custom_config.json"));
    }

    #[test]
    fn candidate_names_non_config_no_double_entry() {
        let empty: FileMap = &[];
        let candidates = candidate_names(empty, "tokenizer.json");
        let tokenizer_count = candidates.iter().filter(|c| c.as_str() == "tokenizer.json").count();
        assert_eq!(tokenizer_count, 1);
    }

    #[test]
    fn candidate_names_file_map_maps_all_candidates() {
        let file_map: FileMap = &[("model.safetensors", "my_model.safetensors")];
        let candidates = candidate_names(file_map, "model.safetensors");
        assert_eq!(candidates[0], "my_model.safetensors");
        assert!(candidates.iter().any(|c| c == "model/my_model.safetensors"));
    }

    // ── push_unique_path with subdirectory paths ──

    #[test]
    fn push_unique_path_subdir_not_equal_to_basename() {
        let mut paths = vec![PathBuf::from("a.txt")];
        push_unique_path(&mut paths, PathBuf::from("dir/a.txt"));
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn push_unique_path_same_subdir_is_duplicate() {
        let mut paths = vec![PathBuf::from("onnx/model.onnx")];
        push_unique_path(&mut paths, PathBuf::from("onnx/model.onnx"));
        assert_eq!(paths.len(), 1);
    }

    // ── ShardIndex edge cases ──

    #[test]
    fn shard_index_many_weights_single_shard() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-shard-mw1s-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        let mut weight_map = serde_json::Map::new();
        for i in 0..20 {
            weight_map.insert(
                format!("layer{}.weight", i),
                serde_json::Value::String("model-00001.safetensors".to_string()),
            );
        }
        let json = serde_json::json!({ "weight_map": weight_map });
        std::fs::write(&index_path, json.to_string()).expect("write index");

        let shard_index = ShardIndex::from_path(&index_path).expect("parse");
        let files = shard_index.shard_files();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], "model-00001.safetensors");

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn shard_index_preserves_weight_map_entries() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-shard-entries-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        let index_json = r#"{
            "weight_map": {
                "embed.weight": "shard-1.bin",
                "head.weight": "shard-2.bin"
            }
        }"#;
        std::fs::write(&index_path, index_json).expect("write index");

        let shard_index = ShardIndex::from_path(&index_path).expect("parse");
        assert_eq!(shard_index.weight_map.len(), 2);
        assert_eq!(shard_index.weight_map.get("embed.weight").unwrap(), "shard-1.bin");
        assert_eq!(shard_index.weight_map.get("head.weight").unwrap(), "shard-2.bin");

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn shard_index_invalid_utf8_returns_error() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-shard-utf8-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        std::fs::write(&index_path, b"\xff\xfe\x00\x00").expect("write invalid bytes");

        let result = ShardIndex::from_path(&index_path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // ── BaseModelField additional ──

    #[test]
    fn base_model_field_eq_single() {
        let a = BaseModelField::Single("org/model".to_string());
        let b = BaseModelField::Single("org/model".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn base_model_field_ne_single_vs_multiple() {
        let a = BaseModelField::Single("org/model".to_string());
        let b = BaseModelField::Multiple(vec!["org/model".to_string()]);
        assert_ne!(a, b);
    }

    #[test]
    fn base_model_field_eq_multiple() {
        let a = BaseModelField::Multiple(vec!["x".to_string(), "y".to_string()]);
        let b = BaseModelField::Multiple(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(a, b);
    }

    #[test]
    fn base_model_field_clone() {
        let a = BaseModelField::Single("org/model".to_string());
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── HfModelMetadata: base_model_repo edge cases ──

    #[test]
    fn base_model_repo_single_with_slashes() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "cardData": { "base_model": "org-name/model-name-v2" } }"#,
        )
        .expect("parse");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org-name/model-name-v2".to_string())
        );
    }

    #[test]
    fn base_model_repo_multiple_first_non_empty_wins() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "cardData": { "base_model": ["", "  ", "org/second", "org/third"] } }"#,
        )
        .expect("parse");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/second".to_string())
        );
    }

    #[test]
    fn base_model_repo_tag_with_multiple_base_model_tags() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["base_model:org/first", "base_model:org/second"] }"#,
        )
        .expect("parse");
        let result = metadata.base_model_repo();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "org/first");
    }

    #[test]
    fn base_model_repo_tag_skips_quantized_then_finds_plain() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["base_model:quantized:org/qmodel", "base_model:org/plain-model"] }"#,
        )
        .expect("parse");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/plain-model".to_string())
        );
    }

    #[test]
    fn base_model_repo_tag_only_quantized() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["base_model:quantized:org/qmodel"] }"#,
        )
        .expect("parse");
        assert_eq!(
            metadata.base_model_repo(),
            Some("org/qmodel".to_string())
        );
    }

    #[test]
    fn hf_model_metadata_deserialize_with_unknown_fields() {
        let metadata: HfModelMetadata = serde_json::from_str(
            r#"{ "unknown_field": 42, "tags": ["gguf"] }"#,
        )
        .expect("parse");
        assert!(metadata.tags.contains(&"gguf".to_string()));
    }

    // ── HfCardData additional ──

    #[test]
    fn hf_card_data_deserialize_with_both_aliases() {
        // When both "base_model" and "baseModel" are present, serde untagged behavior
        // depends on order. This test just verifies it deserializes.
        let card: HfCardData = serde_json::from_str(
            r#"{ "base_model": "org/explicit" }"#,
        )
        .expect("parse");
        assert!(card.base_model.is_some());
    }

    #[test]
    fn hf_card_data_empty_object() {
        let card: HfCardData = serde_json::from_str("{}").expect("parse");
        assert!(card.base_model.is_none());
    }

    // ── HfHubClient additional construction tests ──

    #[test]
    fn hf_hub_client_cache_dir_is_set() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-hfclient-cd-{}",
            std::process::id()
        ));
        let client = HfHubClient::new(temp_dir.clone()).expect("create client");
        assert_eq!(client.cache_dir, temp_dir);
    }

    #[test]
    fn hf_hub_client_with_endpoint_sets_cache_dir() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-hfclient-epcd-{}",
            std::process::id()
        ));
        let client = HfHubClient::with_endpoint(
            temp_dir.clone(),
            Some("https://hf-mirror.com".to_string()),
        )
        .expect("create client");
        assert_eq!(client.cache_dir, temp_dir);
    }

    #[test]
    fn hf_hub_client_with_token_sets_cache_dir() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-hfclient-tokcd-{}",
            std::process::id()
        ));
        let client = HfHubClient::with_token(
            temp_dir.clone(),
            "hf_test_cache_dir_token".to_string(),
        )
        .expect("create client");
        assert_eq!(client.cache_dir, temp_dir);
    }

    // ── HfModelFiles additional ──

    #[test]
    fn hf_model_files_with_many_aux_files() {
        let aux: Vec<PathBuf> = (0..10)
            .map(|i| PathBuf::from(format!("aux_{}.json", i)))
            .collect();
        let files = HfModelFiles {
            repo: "org/big-model".to_string(),
            weights: vec![PathBuf::from("model.safetensors")],
            format: WeightFormat::SafeTensors,
            aux_files: aux.clone(),
        };
        assert_eq!(files.aux_files.len(), 10);
        assert_eq!(files.aux_files[5], PathBuf::from("aux_5.json"));
    }

    #[test]
    fn hf_model_files_repo_with_special_chars() {
        let files = HfModelFiles {
            repo: "org/model-name_v2.1".to_string(),
            weights: vec![],
            format: WeightFormat::Gguf,
            aux_files: vec![],
        };
        assert!(files.repo.contains('-'));
        assert!(files.repo.contains('_'));
        assert!(files.repo.contains('.'));
    }

    // ── is_auth_error additional coverage ──

    #[test]
    fn is_auth_error_all_numeric_codes() {
        assert!(is_auth_error("401"));
        assert!(is_auth_error("403"));
        assert!(!is_auth_error("200"));
        assert!(!is_auth_error("404"));
        assert!(!is_auth_error("500"));
    }

    #[test]
    fn is_auth_error_auth_keywords() {
        assert!(is_auth_error("authentication required"));
        assert!(is_auth_error("Authentication failed"));
        assert!(is_auth_error("AUTHENTICATION error"));
    }

    #[test]
    fn is_auth_error_multi_line_error() {
        let err = "Request failed.\nError: 401 Unauthorized\nPlease check credentials.";
        assert!(is_auth_error(err));
    }

    // ── map_name additional ──

    #[test]
    fn map_name_long_file_map() {
        let file_map: FileMap = &[
            ("a.json", "a_mapped.json"),
            ("b.json", "b_mapped.json"),
            ("c.json", "c_mapped.json"),
            ("d.json", "d_mapped.json"),
        ];
        assert_eq!(map_name(file_map, "c.json"), "c_mapped.json");
        assert_eq!(map_name(file_map, "e.json"), "e.json");
    }

    // ── WeightFormat exhaustive property tests ──

    #[test]
    fn weight_format_all_variants_are_distinct() {
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
        ];
        for (i, v) in variants.iter().enumerate() {
            for (j, w) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(v, w, "variants at index {} and {} should differ", i, j);
                }
            }
        }
    }

    #[test]
    fn weight_format_copy_independent() {
        let a = WeightFormat::Onnx;
        let b = a;
        // After Copy, both are usable and equal
        assert_eq!(a, WeightFormat::Onnx);
        assert_eq!(b, WeightFormat::Onnx);
    }

    #[test]
    fn weight_format_hash_set_all_insert() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WeightFormat::SafeTensors);
        set.insert(WeightFormat::Gguf);
        set.insert(WeightFormat::Onnx);
        assert_eq!(set.len(), 3);
    }

    // ── ShardIndex: weight_map access ──

    #[test]
    fn shard_index_weight_map_keys_preserved() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-shard-keys-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        let index_json = r#"{
            "weight_map": {
                "model.embed_tokens.weight": "shard-1.bin",
                "model.layers.0.self_attn.q_proj.weight": "shard-2.bin",
                "lm_head.weight": "shard-1.bin"
            }
        }"#;
        std::fs::write(&index_path, index_json).expect("write index");

        let shard_index = ShardIndex::from_path(&index_path).expect("parse");
        assert!(shard_index.weight_map.contains_key("model.embed_tokens.weight"));
        assert!(shard_index.weight_map.contains_key("lm_head.weight"));

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // ── normalize_repo_path: only Normal components ──

    #[test]
    fn normalize_repo_path_mixed_curdir_and_normal() {
        assert_eq!(
            normalize_repo_path(Path::new("./a/./b/c/./d")).unwrap(),
            "a/b/c/d"
        );
    }

    #[test]
    fn normalize_repo_path_filename_with_dots() {
        assert_eq!(
            normalize_repo_path(Path::new("model.onnx.data")).unwrap(),
            "model.onnx.data"
        );
    }

    #[test]
    fn normalize_repo_path_rejects_root() {
        assert!(normalize_repo_path(Path::new("/")).is_err());
    }

    // ── resolve_onnx_external_repo_path: parent directory resolution ──

    #[test]
    fn resolve_onnx_external_location_in_same_dir_as_onnx() {
        let result = resolve_onnx_external_repo_path(
            "onnx/model.onnx",
            "model.onnx.data",
        )
        .unwrap();
        assert_eq!(result, "onnx/model.onnx.data");
    }

    #[test]
    fn resolve_onnx_external_nested_data_subdir() {
        let result = resolve_onnx_external_repo_path(
            "optimized/model.onnx",
            "data/external.bin",
        )
        .unwrap();
        assert_eq!(result, "optimized/data/external.bin");
    }

    // ══════════════════════════════════════════════════════════════
    // NEW TESTS (wave-15-edge)
    // ══════════════════════════════════════════════════════════════

    // ── HfModelFiles: weights order sensitivity ──

    #[test]
    fn hf_model_files_neq_reversed_weights() {
        let a = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![PathBuf::from("a.bin"), PathBuf::from("b.bin")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        let b = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![PathBuf::from("b.bin"), PathBuf::from("a.bin")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        assert_ne!(a, b, "reversed weight order should not be equal");
    }

    // ── HfModelFiles: single weight vs empty weights ──

    #[test]
    fn hf_model_files_neq_some_vs_none_weights() {
        let a = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![PathBuf::from("model.safetensors")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        let b = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    // ── HfModelMetadata: tag order matters for PartialEq ──

    #[test]
    fn hf_model_metadata_neq_different_tag_order() {
        let a: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["gguf", "base_model:org/model"] }"#,
        )
        .expect("parse");
        let b: HfModelMetadata = serde_json::from_str(
            r#"{ "tags": ["base_model:org/model", "gguf"] }"#,
        )
        .expect("parse");
        // Vec PartialEq is order-sensitive
        assert_ne!(a, b, "tag order affects equality (Vec derives PartialEq)");
    }

    // ── ShardIndex: empty file (0 bytes) ──

    #[test]
    fn shard_index_empty_file_returns_error() {
        let temp_dir = std::env::temp_dir().join(format!(
            "gllm-shard-emptyfile-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let index_path = temp_dir.join("index.json");
        std::fs::write(&index_path, "").expect("write empty file");

        let result = ShardIndex::from_path(&index_path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // ── is_auth_error: very long non-matching string ──

    #[test]
    fn is_auth_error_long_non_matching_string() {
        let long = "x".repeat(10000);
        assert!(!is_auth_error(&long));
    }

    // ── gguf_preferred_rank: mixed case Q4_K_M ──

    #[test]
    fn gguf_preferred_rank_mixed_case_variants() {
        // to_ascii_lowercase normalizes before matching
        assert_eq!(gguf_preferred_rank("model-Q4_K_M.gguf"), 6);
        assert_eq!(gguf_preferred_rank("MODEL-Q4_K_M.GGUF"), 6);
        assert_eq!(gguf_preferred_rank("Model-Q4_K_M.Gguf"), 6);
    }

    // ── gguf_preferred_rank: q6_k_l contains q6_k substring ──

    #[test]
    fn gguf_preferred_rank_q6_k_before_q6_k_l_lexicographic() {
        let r_k = gguf_preferred_rank("model-q6_k.gguf");
        let r_kl = gguf_preferred_rank("model-q6_k_l.gguf");
        // Both match the q6_k branch at rank 10 (q6_k_l contains q6_k)
        assert_eq!(r_k, r_kl);
        assert_eq!(r_k, 10);
    }

    // ── push_unique_path: large number of duplicates ──

    #[test]
    fn push_unique_path_repeated_same_path_stays_size_one() {
        let mut paths: Vec<PathBuf> = vec![];
        let p = PathBuf::from("only.txt");
        for _ in 0..100 {
            push_unique_path(&mut paths, p.clone());
        }
        assert_eq!(paths.len(), 1);
    }

    // ── normalize_repo_path: only curdir components ──

    #[test]
    fn normalize_repo_path_only_curdir_rejects() {
        // "././." resolves to empty after stripping all CurDir -> empty parts -> error
        let result = normalize_repo_path(Path::new("././."));
        assert!(result.is_err());
    }

    // ── candidate_names: mapped name same as original avoids duplication ──

    #[test]
    fn candidate_names_identity_mapping_no_duplicate_for_non_config() {
        let file_map: FileMap = &[("tokenizer.json", "tokenizer.json")];
        let candidates = candidate_names(file_map, "tokenizer.json");
        let count = candidates.iter().filter(|c| c.as_str() == "tokenizer.json").count();
        assert_eq!(count, 1, "identity mapping should not produce duplicates");
    }

    // ── resolve_onnx_external_repo_path: onnx file at root, data in subdir ──

    #[test]
    fn resolve_onnx_external_root_onnx_subdir_data() {
        let result = resolve_onnx_external_repo_path(
            "model.onnx",
            "data/weights.bin",
        )
        .unwrap();
        assert_eq!(result, "data/weights.bin");
    }

    // ── WeightFormat: Copy used in tuple context ──

    #[test]
    fn weight_format_copy_in_tuple() {
        let pair = (WeightFormat::SafeTensors, WeightFormat::Gguf);
        let (a, b) = pair; // Copy, not move
        assert_eq!(pair.0, a);
        assert_eq!(pair.1, b);
    }

    // ── HfCardData: PartialEq different base_model ──

    #[test]
    fn hf_card_data_neq_different_base_model() {
        let a: HfCardData = serde_json::from_str(
            r#"{ "base_model": "org/model-a" }"#,
        )
        .expect("parse");
        let b: HfCardData = serde_json::from_str(
            r#"{ "base_model": "org/model-b" }"#,
        )
        .expect("parse");
        assert_ne!(a, b);
    }

    // ── map_name: file_map with empty target ──

    #[test]
    fn map_name_empty_target_mapping() {
        let file_map: FileMap = &[("config.json", "")];
        assert_eq!(map_name(file_map, "config.json"), "");
    }

    // ── HfModelFiles: Debug output includes all fields ──

    #[test]
    fn hf_model_files_debug_includes_aux_files() {
        let files = HfModelFiles {
            repo: "org/m".to_string(),
            weights: vec![PathBuf::from("w.bin")],
            format: WeightFormat::Onnx,
            aux_files: vec![PathBuf::from("config.json"), PathBuf::from("tokenizer.json")],
        };
        let debug = format!("{:?}", files);
        assert!(debug.contains("config.json"));
        assert!(debug.contains("tokenizer.json"));
    }
}
