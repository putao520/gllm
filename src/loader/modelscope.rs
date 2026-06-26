//! ModelScope (魔搭社区) 集成
//!
//! ModelScope 是中国的模型托管平台，许多中国模型在那里公开可用

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

use super::downloader::Downloader;
use super::{LoaderError, ModelScopeDownloader, ParallelLoader, ProgressBar, Result, WeightFormat};
use crate::manifest::FileMap;

#[derive(Debug, Clone, PartialEq)]
pub struct MsModelFiles {
    pub repo: String,
    pub weights: Vec<PathBuf>,
    pub format: WeightFormat,
    pub aux_files: Vec<PathBuf>,
}

pub struct ModelScopeClient {
    cache_dir: PathBuf,
}

impl ModelScopeClient {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        Ok(Self { cache_dir })
    }

    /// 下载模型文件（支持从 ModelScope API 直接下载）
    pub fn download_model_files(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
        gguf_file_filter: Option<&str>,
    ) -> Result<MsModelFiles> {
        self.download_model_files_with_format(repo, file_map, parallel, None, gguf_file_filter)
    }

    pub fn download_model_files_with_format(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
        format_hint: Option<WeightFormat>,
        gguf_file_filter: Option<&str>,
    ) -> Result<MsModelFiles> {
        let _ = parallel;
        let repo = repo.to_string();
        let downloader = ModelScopeDownloader::new(
            self.cache_dir.clone(),
            Some("https://www.modelscope.cn".to_string()),
        )?;
        let aux_files = self.collect_aux_files(&repo, file_map, &downloader);

        if let Some(format) = format_hint {
            let result =
                self.download_by_format(&repo, file_map, &downloader, &aux_files, format, gguf_file_filter)?;
            return result.ok_or(LoaderError::MissingWeights);
        }

        if let Some(files) =
            self.try_download_safetensors(&repo, file_map, &downloader, &aux_files)?
        {
            return Ok(files);
        }
        if let Some(files) = self.try_download_gguf(&repo, file_map, &downloader, &aux_files, gguf_file_filter)? {
            return Ok(files);
        }
        if let Some(files) = self.try_download_onnx(&repo, file_map, &downloader, &aux_files)? {
            return Ok(files);
        }

        Err(LoaderError::MissingWeights)
    }

    pub fn download_config_file(&self, repo: &str, file_map: FileMap) -> Result<PathBuf> {
        let downloader = ModelScopeDownloader::new(
            self.cache_dir.clone(),
            Some("https://www.modelscope.cn".to_string()),
        )?;
        self.get_file_any(repo, file_map, "config.json", &downloader)
    }

    pub fn download_tokenizer_file(&self, repo: &str, file_map: FileMap) -> Result<PathBuf> {
        let downloader = ModelScopeDownloader::new(
            self.cache_dir.clone(),
            Some("https://www.modelscope.cn".to_string()),
        )?;
        self.get_file_any(repo, file_map, "tokenizer.json", &downloader)
    }

    fn collect_aux_files(
        &self,
        repo: &str,
        file_map: FileMap,
        downloader: &ModelScopeDownloader,
    ) -> Vec<PathBuf> {
        let mut aux_files = Vec::new();
        for name in [
            "config.json",
            "configuration.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
        ] {
            if let Ok(path) = self.get_file_any(repo, file_map, name, downloader) {
                aux_files.push(path);
            }
        }
        aux_files
    }

    fn download_by_format(
        &self,
        repo: &str,
        file_map: FileMap,
        downloader: &ModelScopeDownloader,
        aux_files: &[PathBuf],
        format: WeightFormat,
        gguf_file_filter: Option<&str>,
    ) -> Result<Option<MsModelFiles>> {
        match format {
            WeightFormat::SafeTensors => {
                self.try_download_safetensors(repo, file_map, downloader, aux_files)
            }
            WeightFormat::Gguf => self.try_download_gguf(repo, file_map, downloader, aux_files, gguf_file_filter),
            WeightFormat::Onnx => self.try_download_onnx(repo, file_map, downloader, aux_files),
            WeightFormat::PyTorch => Ok(None),
            WeightFormat::Gllm => Ok(None), // .gllm is created offline, not downloaded
        }
    }

    fn try_download_safetensors(
        &self,
        repo: &str,
        file_map: FileMap,
        downloader: &ModelScopeDownloader,
        aux_files: &[PathBuf],
    ) -> Result<Option<MsModelFiles>> {
        if let Ok(index_path) =
            self.get_file_any(repo, file_map, "model.safetensors.index.json", downloader)
        {
            let shard_index = self.parse_safetensors_index(&index_path)?;
            let shard_files = shard_index.shard_files();
            let weights = self.download_shards(repo, &shard_files, downloader)?;
            let mut aux = aux_files.to_vec();
            aux.push(index_path);
            return Ok(Some(MsModelFiles {
                repo: repo.to_string(),
                weights,
                format: WeightFormat::SafeTensors,
                aux_files: aux,
            }));
        }

        if let Ok(path) = self.get_file_any(repo, file_map, "model.safetensors", downloader) {
            return Ok(Some(MsModelFiles {
                repo: repo.to_string(),
                weights: vec![path],
                format: WeightFormat::SafeTensors,
                aux_files: aux_files.to_vec(),
            }));
        }

        Ok(None)
    }

    fn try_download_gguf(
        &self,
        repo: &str,
        file_map: FileMap,
        downloader: &ModelScopeDownloader,
        aux_files: &[PathBuf],
        gguf_file_filter: Option<&str>,
    ) -> Result<Option<MsModelFiles>> {
        for candidate in self.gguf_candidate_names(repo) {
            if let Some(filter) = gguf_file_filter {
                if !candidate.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) {
                    continue;
                }
            }
            if let Some(path) = self.try_get_file_any(repo, file_map, &candidate, downloader) {
                return Ok(Some(MsModelFiles {
                    repo: repo.to_string(),
                    weights: vec![path],
                    format: WeightFormat::Gguf,
                    aux_files: aux_files.to_vec(),
                }));
            }
        }
        Ok(None)
    }

    fn try_download_onnx(
        &self,
        repo: &str,
        file_map: FileMap,
        downloader: &ModelScopeDownloader,
        aux_files: &[PathBuf],
    ) -> Result<Option<MsModelFiles>> {
        for candidate in self.onnx_candidate_names() {
            if let Some(path) = self.try_get_file_any(repo, file_map, &candidate, downloader) {
                let external_data =
                    self.download_onnx_external_data(repo, &candidate, &path, downloader)?;
                let mut aux = aux_files.to_vec();
                for external in external_data {
                    push_unique_path(&mut aux, external);
                }
                return Ok(Some(MsModelFiles {
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
        downloader: &ModelScopeDownloader,
    ) -> Result<Vec<PathBuf>> {
        let locations = super::onnx::external_data_locations(local_onnx_path)?;
        let mut out = Vec::with_capacity(locations.len());
        for location in locations {
            let repo_path = resolve_onnx_external_repo_path(onnx_repo_path, &location)?;
            let downloaded = self.get_file(repo, &repo_path, downloader)?;
            push_unique_path(&mut out, downloaded);
        }
        Ok(out)
    }

    /// Ω1: 候选文件名列表（按优先级排序）
    /// 不基于文件名推测，仅作为文件存在性检查的顺序
    fn gguf_candidate_names(&self, repo: &str) -> Vec<String> {
        let mut names = vec![
            "model.gguf".to_string(),
            "ggml-model-q4_0.gguf".to_string(),
            "ggml-model-q8_0.gguf".to_string(),
            "ggml-model-f16.gguf".to_string(),
        ];

        if let Some(base) = repo.split('/').next_back() {
            for quant in ["Q4_0", "Q8_0", "f16"] {
                names.push(format!("{base}-{quant}.gguf"));
                names.push(format!("{base}.{quant}.gguf"));
            }
        }

        // 简单字母排序，不基于文件名推测
        names.sort();
        names
    }

    /// Ω1: 候选文件名列表（按优先级排序）
    fn onnx_candidate_names(&self) -> Vec<String> {
        let names = vec!["onnx/model.onnx".to_string(), "model.onnx".to_string()];

        // 简单字母排序
        let mut result = names;
        result.sort();
        result
    }

    fn try_get_file_any(
        &self,
        repo: &str,
        file_map: FileMap,
        logical: &str,
        downloader: &ModelScopeDownloader,
    ) -> Option<PathBuf> {
        let candidates = if logical.contains('/') {
            vec![logical.to_string()]
        } else {
            self.candidate_names(file_map, logical)
        };

        for candidate in candidates {
            if let Ok(path) = self.get_file(repo, &candidate, downloader) {
                return Some(path);
            }
        }
        None
    }

    fn get_file_any(
        &self,
        repo: &str,
        file_map: FileMap,
        logical: &str,
        downloader: &ModelScopeDownloader,
    ) -> Result<PathBuf> {
        for candidate in self.candidate_names(file_map, logical) {
            if let Ok(path) = self.get_file(repo, &candidate, downloader) {
                return Ok(path);
            }
        }
        Err(LoaderError::MissingWeights)
    }

    fn get_file(
        &self,
        repo: &str,
        filename: &str,
        downloader: &ModelScopeDownloader,
    ) -> Result<PathBuf> {
        let mut progress = ProgressBar::new(filename.to_string());
        downloader.download_file_with_progress(repo, filename, &self.cache_dir, &mut progress)
    }

    fn download_shards(
        &self,
        repo: &str,
        shards: &[String],
        downloader: &ModelScopeDownloader,
    ) -> Result<Vec<PathBuf>> {
        let mut result = Vec::new();
        for (idx, shard_path) in shards.iter().enumerate() {
            let filename = shard_path;
            log::info!("[{}/{}] 下载分片: {}", idx + 1, shards.len(), filename);

            let mut progress = ProgressBar::new(filename.clone());
            let path = downloader.download_file_with_progress(
                repo,
                filename,
                &self.cache_dir,
                &mut progress,
            )?;
            result.push(path);
        }
        Ok(result)
    }

    fn candidate_names(&self, file_map: FileMap, logical: &str) -> Vec<String> {
        let mut base_names = Vec::new();
        base_names.push(self.map_name(file_map, logical).to_string());
        if logical == "config.json" {
            base_names.push(self.map_name(file_map, "configuration.json").to_string());
        }

        let mut out = Vec::new();
        for base in base_names {
            if !out.contains(&base) {
                out.push(base.clone());
            }
            for prefix in ["model/", "weights/"] {
                let candidate = format!("{}{}", prefix, base);
                if !out.contains(&candidate) {
                    out.push(candidate);
                }
            }
        }
        out
    }

    fn map_name<'a>(&'a self, file_map: FileMap, logical: &'a str) -> &'a str {
        for (source, target) in file_map {
            if *source == logical {
                return target;
            }
        }
        logical
    }

    /// 从缓存目录加载已下载的 ModelScope 模型
    pub fn load_from_cache(&self, repo: &str, _file_map: FileMap) -> Result<MsModelFiles> {
        // 标准化仓库名: org/name → models--org--name
        let normalized_repo = repo.replace('/', "--");
        let model_dir = self.cache_dir.join("models--").join(&normalized_repo);

        if !model_dir.exists() {
            return Err(LoaderError::HfHub(format!(
                "ModelScope cache not found: {}. Please download using ModelScope CLI first.",
                repo
            )));
        }

        // 查找最新的 snapshot
        let snapshots_dir = model_dir.join("snapshots");
        let snapshot = self.find_latest_snapshot(&snapshots_dir)?;

        let mut weights = Vec::new();
        let mut format = WeightFormat::SafeTensors;
        let mut aux_files = Vec::new();

        // 查找权重文件
        let safetensors = snapshot.join("model.safetensors");
        if safetensors.exists() {
            weights.push(safetensors);
        }

        // 查找分片权重
        let index_json = snapshot.join("model.safetensors.index.json");
        if index_json.exists() {
            if let Ok(index) = self.parse_safetensors_index(&index_json) {
                for shard_file in index.shard_files() {
                    let shard_path = snapshot.join(&shard_file);
                    if shard_path.exists() {
                        weights.push(shard_path);
                    }
                }
            }
        }

        // 查找辅助文件
        for name in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ] {
            let file = snapshot.join(name);
            if file.exists() {
                aux_files.push(file);
            }
        }

        if weights.is_empty() {
            // Ω1: 不基于文件名推测，选择第一个找到的文件
            if let Some(best) = select_first_cached(&snapshot, "gguf") {
                weights.push(best);
                format = WeightFormat::Gguf;
            }
        }

        if weights.is_empty() {
            // Ω1: 不基于文件名推测，选择第一个找到的文件
            if let Some(best) = select_first_cached(&snapshot, "onnx") {
                weights.push(best);
                format = WeightFormat::Onnx;
            }
        }

        if weights.is_empty() {
            return Err(LoaderError::MissingWeights);
        }

        Ok(MsModelFiles {
            repo: repo.to_string(),
            weights,
            format,
            aux_files,
        })
    }

    fn find_latest_snapshot(&self, snapshots_dir: &Path) -> Result<PathBuf> {
        let mut latest = None;
        let mut latest_mtime: std::time::SystemTime = std::time::SystemTime::UNIX_EPOCH;

        for entry in fs::read_dir(snapshots_dir).map_err(LoaderError::Io)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            let mtime = metadata.modified()?;

            if mtime > latest_mtime {
                latest_mtime = mtime;
                latest = Some(entry.path());
            }
        }

        latest.ok_or_else(|| LoaderError::HfHub("No snapshot found".to_string()))
    }

    fn parse_safetensors_index(&self, path: &Path) -> Result<SafetensorsIndex> {
        let content = fs::read(path)?;
        serde_json::from_slice(&content).map_err(LoaderError::Json)
    }

    /// 列出缓存中可用的模型
    pub fn list_cached_models(&self) -> Result<Vec<String>> {
        let models_dir = self.cache_dir.join("models--");
        let mut models = Vec::new();

        if !models_dir.exists() {
            return Ok(models);
        }

        for entry in fs::read_dir(&models_dir).map_err(LoaderError::Io)? {
            let name = entry?.file_name();
            // 转换回 org/name 格式
            let normalized = name.to_string_lossy().replace("--", "/");
            models.push(normalized);
        }

        models.sort();
        Ok(models)
    }
}

/// Ω1: 选择第一个找到的文件，不基于文件名推测
fn select_first_cached(snapshot: &Path, ext: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if ext.eq_ignore_ascii_case("onnx") {
        let onnx_dir = snapshot.join("onnx");
        if onnx_dir.exists() {
            candidates.extend(find_files_with_extension(&onnx_dir, ext));
            if !candidates.is_empty() {
                candidates.sort();
                return candidates.into_iter().next();
            }
        }
    }

    candidates.extend(find_files_with_extension(snapshot, ext));
    if !candidates.is_empty() {
        candidates.sort();
        candidates.into_iter().next()
    } else {
        None
    }
}

fn find_files_with_extension(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            log::warn!("cannot read directory {}: {e}", dir.display());
            return files;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        let matches = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case(ext))
            .unwrap_or(false); // LEGAL: 无扩展名时视为不匹配
        if matches {
            files.push(path);
        }
    }
    files
}

#[derive(Debug, Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
struct SafetensorsIndex {
    #[serde(rename = "weight_map")]
    weight_map: HashMap<String, String>,
}

impl SafetensorsIndex {
    fn shard_files(&self) -> Vec<String> {
        let mut shards = HashSet::new();
        for shard in self.weight_map.values() {
            shards.insert(shard.clone());
        }
        let mut list: Vec<String> = shards.into_iter().collect();
        list.sort();
        list
    }
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

/// 从 ModelScope 缓存加载模型
pub fn from_cache(cache_dir: PathBuf, repo: &str) -> Result<MsModelFiles> {
    let client = ModelScopeClient::new(cache_dir)?;
    client.load_from_cache(repo, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_safetensors_index() {
        // safetensors index.json 格式：weight_map 是 参数名 -> 分片文件名
        let json = r#"{"weight_map":{"model.tok_embeddings.weight":"model-00001-of-00002.safetensors","model.norm.weight":"model-00002-of-00002.safetensors"}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.shard_files().len(), 2);
        assert!(index
            .shard_files()
            .contains(&"model-00001-of-00002.safetensors".to_string()));
        assert!(index
            .shard_files()
            .contains(&"model-00002-of-00002.safetensors".to_string()));
    }

    #[test]
    fn test_shard_files_dedup() {
        // 测试重复分片被去重
        let json = r#"{"weight_map":{"layer1.weight":"shard.safetensors","layer2.weight":"shard.safetensors"}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.shard_files().len(), 1);
        assert_eq!(index.shard_files()[0], "shard.safetensors");
    }

    #[test]
    fn test_shard_files_sorted() {
        let json = r#"{"weight_map":{"a":"shard-2.st","b":"shard-1.st","c":"shard-3.st"}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        let files = index.shard_files();
        assert_eq!(files, vec!["shard-1.st", "shard-2.st", "shard-3.st"]);
    }

    #[test]
    fn test_shard_files_single() {
        let json = r#"{"weight_map":{"model.weight":"model.safetensors"}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.shard_files(), vec!["model.safetensors"]);
    }

    // ── push_unique_path ──

    #[test]
    fn push_unique_path_adds_new() {
        let mut paths = vec![PathBuf::from("a.st")];
        push_unique_path(&mut paths, PathBuf::from("b.st"));
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn push_unique_path_skips_duplicate() {
        let mut paths = vec![PathBuf::from("a.st")];
        push_unique_path(&mut paths, PathBuf::from("a.st"));
        assert_eq!(paths.len(), 1);
    }

    // ── normalize_repo_path ──

    #[test]
    fn normalize_repo_path_simple() {
        let result = normalize_repo_path(Path::new("onnx/external_data.bin")).unwrap();
        assert_eq!(result, "onnx/external_data.bin");
    }

    #[test]
    fn normalize_repo_path_dot_components() {
        let result = normalize_repo_path(Path::new("./onnx/./data.bin")).unwrap();
        assert_eq!(result, "onnx/data.bin");
    }

    #[test]
    fn normalize_repo_path_rejects_parent() {
        let result = normalize_repo_path(Path::new("../etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_repo_path_rejects_root() {
        let result = normalize_repo_path(Path::new("/etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_repo_path_rejects_empty() {
        let result = normalize_repo_path(Path::new("."));
        assert!(result.is_err());
    }

    // ── resolve_onnx_external_repo_path ──

    #[test]
    fn resolve_onnx_external_simple() {
        let result = resolve_onnx_external_repo_path("onnx/model.onnx", "external.bin").unwrap();
        assert_eq!(result, "onnx/external.bin");
    }

    #[test]
    fn resolve_onnx_external_nested() {
        let result =
            resolve_onnx_external_repo_path("model.onnx", "data/weights.bin").unwrap();
        assert_eq!(result, "data/weights.bin");
    }

    // ── gguf_candidate_names ──

    #[test]
    fn gguf_candidate_names_includes_defaults() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("org/model-q4");
        assert!(names.contains(&"model.gguf".to_string()));
        assert!(names.contains(&"ggml-model-q4_0.gguf".to_string()));
    }

    #[test]
    fn gguf_candidate_names_repo_specific() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("org/Qwen3-8B");
        let has_qwen_specific = names.iter().any(|n| n.contains("Qwen3-8B"));
        assert!(has_qwen_specific, "should include repo-specific candidates");
    }

    // ── onnx_candidate_names ──

    #[test]
    fn onnx_candidate_names_contents() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.onnx_candidate_names();
        assert!(names.contains(&"model.onnx".to_string()));
        assert!(names.contains(&"onnx/model.onnx".to_string()));
    }

    // ── candidate_names ──

    #[test]
    fn candidate_names_basic() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.candidate_names(&[], "model.safetensors");
        assert!(names.contains(&"model.safetensors".to_string()));
    }

    #[test]
    fn candidate_names_config_json_alternate() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.candidate_names(&[], "config.json");
        assert!(names.iter().any(|n| n.contains("configuration.json")));
    }

    #[test]
    fn candidate_names_with_prefix() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.candidate_names(&[], "model.safetensors");
        assert!(names.iter().any(|n| n.starts_with("model/")));
        assert!(names.iter().any(|n| n.starts_with("weights/")));
    }

    #[test]
    fn candidate_names_with_file_map() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let file_map: FileMap = &[("config.json", "my_config.json")];
        let names = client.candidate_names(file_map, "config.json");
        assert!(names[0].contains("my_config.json"));
    }

    // ── map_name ──

    #[test]
    fn map_name_no_mapping() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        assert_eq!(client.map_name(&[], "config.json"), "config.json");
    }

    #[test]
    fn map_name_with_mapping() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let file_map: FileMap = &[("config.json", "my_config.json")];
        assert_eq!(client.map_name(file_map, "config.json"), "my_config.json");
    }

    // ── MsModelFiles struct ──

    #[test]
    fn ms_model_files_fields() {
        let files = MsModelFiles {
            repo: "org/model".to_string(),
            weights: vec![PathBuf::from("model.safetensors")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![PathBuf::from("config.json")],
        };
        assert_eq!(files.repo, "org/model");
        assert_eq!(files.weights.len(), 1);
        assert_eq!(files.aux_files.len(), 1);
    }

    // ── find_files_with_extension (filesystem) ──

    #[test]
    fn find_files_with_extension_on_temp_dir() {
        let dir = std::env::temp_dir().join("gllm_test_find_ext");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("model.gguf"), b"gguf");
        let _ = fs::write(dir.join("config.json"), b"{}");

        let found = find_files_with_extension(&dir, "gguf");
        assert_eq!(found.len(), 1);
        assert!(found[0].to_string_lossy().ends_with("model.gguf"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_files_with_extension_no_match() {
        let dir = std::env::temp_dir().join("gllm_test_find_nomatch");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("config.json"), b"{}");

        let found = find_files_with_extension(&dir, "onnx");
        assert!(found.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_files_with_extension_nonexistent_dir() {
        let found = find_files_with_extension(Path::new("/nonexistent/path"), "gguf");
        assert!(found.is_empty());
    }

    // ── Additional tests ──

    // -- SafetensorsIndex edge cases --

    #[test]
    fn safetensors_index_empty_weight_map() {
        let json = r#"{"weight_map":{}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        assert!(index.shard_files().is_empty());
    }

    #[test]
    fn safetensors_index_many_params_one_shard() {
        let json = r#"{"weight_map":{"a":"shard.bin","b":"shard.bin","c":"shard.bin"}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.shard_files(), vec!["shard.bin"]);
    }

    #[test]
    fn safetensors_index_deserialize_rejects_invalid_json() {
        let result = serde_json::from_str::<SafetensorsIndex>("not json");
        assert!(result.is_err());
    }

    // -- normalize_repo_path additional edge cases --

    #[test]
    fn normalize_repo_path_single_component() {
        let result = normalize_repo_path(Path::new("data.bin")).unwrap();
        assert_eq!(result, "data.bin");
    }

    #[test]
    fn normalize_repo_path_deeply_nested() {
        let result = normalize_repo_path(Path::new("a/b/c/d.bin")).unwrap();
        assert_eq!(result, "a/b/c/d.bin");
    }

    #[test]
    fn normalize_repo_path_only_dots() {
        let result = normalize_repo_path(Path::new("././."));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_repo_path_dot_dot_component() {
        let result = normalize_repo_path(Path::new("onnx/../secret"));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_repo_path_mixed_valid_and_dot() {
        let result = normalize_repo_path(Path::new("./onnx/./model/./data.bin")).unwrap();
        assert_eq!(result, "onnx/model/data.bin");
    }

    // -- resolve_onnx_external_repo_path additional cases --

    #[test]
    fn resolve_onnx_external_in_onnx_subdir() {
        let result =
            resolve_onnx_external_repo_path("onnx/model.onnx", "model_external_data.bin").unwrap();
        assert_eq!(result, "onnx/model_external_data.bin");
    }

    #[test]
    fn resolve_onnx_external_rejects_traversal() {
        let result = resolve_onnx_external_repo_path("onnx/model.onnx", "../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn resolve_onnx_external_deeply_nested_data() {
        let result = resolve_onnx_external_repo_path(
            "subdir/model.onnx",
            "data/external/weights.bin",
        )
        .unwrap();
        assert_eq!(result, "subdir/data/external/weights.bin");
    }

    // -- gguf_candidate_names additional cases --

    #[test]
    fn gguf_candidate_names_is_sorted() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("org/model-name");
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "gguf_candidate_names should be sorted");
    }

    #[test]
    fn gguf_candidate_names_plain_repo_no_slash() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("single-name");
        // No slash, so split('/').next_back() gives "single-name"
        let has_specific = names.iter().any(|n| n.contains("single-name"));
        assert!(has_specific);
    }

    #[test]
    fn gguf_candidate_names_repo_with_deep_path() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("org/subdir/model-v2");
        // Last component is "model-v2"
        let has_specific = names
            .iter()
            .any(|n| n.contains("model-v2") && n.ends_with(".gguf"));
        assert!(has_specific);
    }

    // -- candidate_names additional cases --

    #[test]
    fn candidate_names_no_duplicates() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.candidate_names(&[], "config.json");
        let mut seen = HashSet::new();
        for name in &names {
            assert!(seen.insert(name.clone()), "duplicate candidate: {name}");
        }
    }

    #[test]
    fn candidate_names_config_json_gets_configuration_alias() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.candidate_names(&[], "config.json");
        let has_config = names.iter().any(|n| n == "config.json");
        let has_configuration = names.iter().any(|n| n == "configuration.json");
        assert!(has_config);
        assert!(has_configuration);
    }

    #[test]
    fn candidate_names_non_config_has_no_alias() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.candidate_names(&[], "tokenizer.json");
        let has_configuration = names.iter().any(|n| n == "configuration.json");
        assert!(
            !has_configuration,
            "non-config files should not get configuration.json alias"
        );
    }

    #[test]
    fn candidate_names_file_map_overrides_base() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let file_map: FileMap = &[("tokenizer.json", "custom_tok.json")];
        let names = client.candidate_names(file_map, "tokenizer.json");
        assert_eq!(names[0], "custom_tok.json");
    }

    #[test]
    fn candidate_names_file_map_no_match_passes_through() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let file_map: FileMap = &[("other.json", "mapped.json")];
        let names = client.candidate_names(file_map, "config.json");
        assert_eq!(names[0], "config.json");
    }

    // -- map_name additional cases --

    #[test]
    fn map_name_first_match_wins() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let file_map: FileMap = &[("a", "first"), ("a", "second")];
        assert_eq!(client.map_name(file_map, "a"), "first");
    }

    #[test]
    fn map_name_unmapped_key_returns_original() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let file_map: FileMap = &[("x", "y")];
        assert_eq!(client.map_name(file_map, "z"), "z");
    }

    // -- push_unique_path additional cases --

    #[test]
    fn push_unique_path_empty_vec_adds() {
        let mut paths: Vec<PathBuf> = vec![];
        push_unique_path(&mut paths, PathBuf::from("new.st"));
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], PathBuf::from("new.st"));
    }

    #[test]
    fn push_unique_path_different_dirs_not_duplicate() {
        let mut paths = vec![PathBuf::from("/a/model.st")];
        push_unique_path(&mut paths, PathBuf::from("/b/model.st"));
        assert_eq!(paths.len(), 2);
    }

    // -- MsModelFiles additional cases --

    #[test]
    fn ms_model_files_empty_weights() {
        let files = MsModelFiles {
            repo: "org/model".to_string(),
            weights: vec![],
            format: WeightFormat::Gguf,
            aux_files: vec![],
        };
        assert!(files.weights.is_empty());
        assert!(files.aux_files.is_empty());
        assert_eq!(files.repo, "org/model");
    }

    #[test]
    fn ms_model_files_all_formats() {
        for fmt in [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ] {
            let files = MsModelFiles {
                repo: "test".to_string(),
                weights: vec![],
                format: fmt,
                aux_files: vec![],
            };
            assert_eq!(files.format, fmt);
        }
    }

    // -- WeightFormat trait coverage --

    #[test]
    fn weight_format_equality() {
        assert_eq!(WeightFormat::SafeTensors, WeightFormat::SafeTensors);
        assert_ne!(WeightFormat::SafeTensors, WeightFormat::Gguf);
    }

    #[test]
    fn weight_format_copy() {
        let a = WeightFormat::Onnx;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn weight_format_clone() {
        let a = WeightFormat::Gllm;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // -- onnx_candidate_names additional --

    #[test]
    fn onnx_candidate_names_is_sorted() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.onnx_candidate_names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn onnx_candidate_names_has_two_entries() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.onnx_candidate_names();
        assert_eq!(names.len(), 2);
    }

    // -- select_first_cached --

    #[test]
    fn select_first_cached_empty_dir() {
        let dir = std::env::temp_dir().join("gllm_test_select_empty");
        let _ = fs::create_dir_all(&dir);
        let result = select_first_cached(&dir, "gguf");
        assert!(result.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn select_first_cached_picks_sorted_first_gguf() {
        let dir = std::env::temp_dir().join("gllm_test_select_sorted");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("b-model.gguf"), b"b");
        let _ = fs::write(dir.join("a-model.gguf"), b"a");

        let result = select_first_cached(&dir, "gguf");
        assert!(result.is_some());
        let name = result.unwrap().file_name().unwrap().to_string_lossy().to_string();
        assert_eq!(name, "a-model.gguf", "should pick lexicographically first");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn select_first_cached_onnx_checks_subdir_first() {
        let dir = std::env::temp_dir().join("gllm_test_select_onnx");
        let onnx_dir = dir.join("onnx");
        let _ = fs::create_dir_all(&onnx_dir);
        // Put file in onnx/ subdir
        let _ = fs::write(onnx_dir.join("model.onnx"), b"onnx");
        // No .onnx file at root level

        let result = select_first_cached(&dir, "onnx");
        assert!(result.is_some());
        let path = result.unwrap();
        assert!(path.to_string_lossy().contains("onnx"), "should find in onnx/ subdir");

        let _ = fs::remove_dir_all(&dir);
    }

    // -- find_files_with_extension case insensitivity --

    #[test]
    fn find_files_with_extension_case_insensitive() {
        let dir = std::env::temp_dir().join("gllm_test_case_ext");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("model.GGUF"), b"gguf");

        let found = find_files_with_extension(&dir, "gguf");
        assert_eq!(found.len(), 1, "should match .GGUF case-insensitively");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_files_with_extension_skips_directories() {
        let dir = std::env::temp_dir().join("gllm_test_skip_dir");
        let sub = dir.join("subdir.gguf");
        let _ = fs::create_dir_all(&sub);
        let _ = fs::write(dir.join("real.gguf"), b"gguf");

        let found = find_files_with_extension(&dir, "gguf");
        assert_eq!(found.len(), 1, "should skip directories even with matching extension");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_files_with_extension_no_extension_file() {
        let dir = std::env::temp_dir().join("gllm_test_no_ext");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("README"), b"readme");

        let found = find_files_with_extension(&dir, "gguf");
        assert!(found.is_empty(), "file with no extension should not match");

        let _ = fs::remove_dir_all(&dir);
    }

    // ════════════════════════════════════════════════════════════════
    //  New tests below
    // ════════════════════════════════════════════════════════════════

    // ── WeightFormat trait coverage: Hash ──

    #[test]
    fn weight_format_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(WeightFormat::SafeTensors));
        assert!(set.insert(WeightFormat::Gguf));
        assert!(!set.insert(WeightFormat::SafeTensors), "duplicate insert should return false");
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn weight_format_all_variants_distinct_hash() {
        use std::collections::HashSet;
        let all = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        let set: HashSet<WeightFormat> = all.iter().copied().collect();
        assert_eq!(set.len(), all.len(), "all variants should have distinct hashes");
    }

    #[test]
    fn weight_format_debug_output() {
        assert_eq!(format!("{:?}", WeightFormat::SafeTensors), "SafeTensors");
        assert_eq!(format!("{:?}", WeightFormat::Gguf), "Gguf");
        assert_eq!(format!("{:?}", WeightFormat::Onnx), "Onnx");
        assert_eq!(format!("{:?}", WeightFormat::PyTorch), "PyTorch");
        assert_eq!(format!("{:?}", WeightFormat::Gllm), "Gllm");
    }

    // ── MsModelFiles trait coverage: Clone, PartialEq ──

    #[test]
    fn ms_model_files_clone() {
        let files = MsModelFiles {
            repo: "org/model".to_string(),
            weights: vec![PathBuf::from("model.safetensors")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![PathBuf::from("config.json")],
        };
        let cloned = files.clone();
        assert_eq!(files, cloned);
    }

    #[test]
    fn ms_model_files_partial_eq_same() {
        let a = MsModelFiles {
            repo: "r".to_string(),
            weights: vec![PathBuf::from("w.st")],
            format: WeightFormat::Gguf,
            aux_files: vec![],
        };
        let b = MsModelFiles {
            repo: "r".to_string(),
            weights: vec![PathBuf::from("w.st")],
            format: WeightFormat::Gguf,
            aux_files: vec![],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn ms_model_files_partial_eq_diff_repo() {
        let a = MsModelFiles {
            repo: "a".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        let b = MsModelFiles {
            repo: "b".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn ms_model_files_partial_eq_diff_weights() {
        let a = MsModelFiles {
            repo: "r".to_string(),
            weights: vec![PathBuf::from("a.st")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        let b = MsModelFiles {
            repo: "r".to_string(),
            weights: vec![PathBuf::from("b.st")],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn ms_model_files_partial_eq_diff_format() {
        let a = MsModelFiles {
            repo: "r".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![],
        };
        let b = MsModelFiles {
            repo: "r".to_string(),
            weights: vec![],
            format: WeightFormat::Onnx,
            aux_files: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn ms_model_files_partial_eq_diff_aux_files() {
        let a = MsModelFiles {
            repo: "r".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![PathBuf::from("config.json")],
        };
        let b = MsModelFiles {
            repo: "r".to_string(),
            weights: vec![],
            format: WeightFormat::SafeTensors,
            aux_files: vec![PathBuf::from("tokenizer.json")],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn ms_model_files_debug_output() {
        let files = MsModelFiles {
            repo: "test/repo".to_string(),
            weights: vec![],
            format: WeightFormat::Gguf,
            aux_files: vec![],
        };
        let debug_str = format!("{files:?}");
        assert!(debug_str.contains("MsModelFiles"));
        assert!(debug_str.contains("test/repo"));
        assert!(debug_str.contains("Gguf"));
    }

    // ── ModelScopeClient construction ──

    #[test]
    fn client_new_basic() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp/test_cache")).unwrap();
        assert_eq!(client.cache_dir, PathBuf::from("/tmp/test_cache"));
    }

    #[test]
    fn client_new_empty_path() {
        let client = ModelScopeClient::new(PathBuf::new()).unwrap();
        assert_eq!(client.cache_dir, PathBuf::new());
    }

    #[test]
    fn client_new_long_path() {
        let long = PathBuf::from("/tmp").join("a".repeat(300));
        let client = ModelScopeClient::new(long.clone()).unwrap();
        assert_eq!(client.cache_dir, long);
    }

    // ── list_cached_models ──

    #[test]
    fn list_cached_models_nonexistent_dir() {
        let dir = std::env::temp_dir().join("gllm_test_list_none_").join("no_such_dir");
        let client = ModelScopeClient::new(dir).unwrap();
        let models = client.list_cached_models().unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn list_cached_models_empty_dir() {
        let dir = std::env::temp_dir().join("gllm_test_list_empty_models--");
        let models_dir = dir.join("models--");
        let _ = fs::create_dir_all(&models_dir);
        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let models = client.list_cached_models().unwrap();
        assert!(models.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_cached_models_finds_entries() {
        let dir = std::env::temp_dir().join("gllm_test_list_has_models--");
        let models_dir = dir.join("models--");
        let _ = fs::create_dir_all(models_dir.join("org--model-a"));
        let _ = fs::create_dir_all(models_dir.join("org--model-b"));
        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let models = client.list_cached_models().unwrap();
        assert_eq!(models.len(), 2);
        // Should be sorted and use / separator
        assert!(models.iter().any(|m| m == "org/model-a"));
        assert!(models.iter().any(|m| m == "org/model-b"));
        let _ = fs::remove_dir_all(&dir);
    }

    // ── load_from_cache error paths ──

    #[test]
    fn load_from_cache_missing_dir() {
        let dir = std::env::temp_dir().join("gllm_test_load_missing_").join("no_such");
        let client = ModelScopeClient::new(dir).unwrap();
        let result = client.load_from_cache("org/model", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn load_from_cache_no_weights() {
        let dir = std::env::temp_dir().join("gllm_test_load_no_weights");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let snapshot = snapshots.join("abc123");
        let _ = fs::create_dir_all(&snapshot);
        // Only write aux files, no weight files
        let _ = fs::write(snapshot.join("config.json"), b"{}");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.load_from_cache("org/model", &[]);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_from_cache_with_safetensors() {
        let dir = std::env::temp_dir().join("gllm_test_load_safetensors");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let snapshot = snapshots.join("rev1");
        let _ = fs::create_dir_all(&snapshot);
        let _ = fs::write(snapshot.join("model.safetensors"), b"fake");
        let _ = fs::write(snapshot.join("config.json"), b"{}");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.load_from_cache("org/model", &[]);
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.repo, "org/model");
        assert_eq!(files.format, WeightFormat::SafeTensors);
        assert_eq!(files.weights.len(), 1);
        assert!(files.aux_files.iter().any(|p| p.file_name().unwrap() == "config.json"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_from_cache_with_gguf_fallback() {
        let dir = std::env::temp_dir().join("gllm_test_load_gguf");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let snapshot = snapshots.join("rev1");
        let _ = fs::create_dir_all(&snapshot);
        let _ = fs::write(snapshot.join("model.gguf"), b"fake");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.load_from_cache("org/model", &[]);
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.format, WeightFormat::Gguf);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_from_cache_with_onnx_fallback() {
        let dir = std::env::temp_dir().join("gllm_test_load_onnx");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let snapshot = snapshots.join("rev1");
        let _ = fs::create_dir_all(&snapshot);
        let _ = fs::write(snapshot.join("model.onnx"), b"fake");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.load_from_cache("org/model", &[]);
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.format, WeightFormat::Onnx);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_from_cache_with_sharded_safetensors() {
        let dir = std::env::temp_dir().join("gllm_test_load_sharded");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let snapshot = snapshots.join("rev1");
        let _ = fs::create_dir_all(&snapshot);
        // Write shard index
        let index_json = r#"{"weight_map":{"a":"shard-1.st","b":"shard-2.st"}}"#;
        let _ = fs::write(snapshot.join("model.safetensors.index.json"), index_json);
        let _ = fs::write(snapshot.join("shard-1.st"), b"s1");
        let _ = fs::write(snapshot.join("shard-2.st"), b"s2");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.load_from_cache("org/model", &[]);
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.format, WeightFormat::SafeTensors);
        assert_eq!(files.weights.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_from_cache_picks_latest_snapshot() {
        let dir = std::env::temp_dir().join("gllm_test_latest_snap");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let old_snap = snapshots.join("old_rev");
        let new_snap = snapshots.join("new_rev");
        let _ = fs::create_dir_all(&old_snap);
        let _ = fs::create_dir_all(&new_snap);
        let _ = fs::write(old_snap.join("model.gguf"), b"old");
        let _ = fs::write(new_snap.join("model.gguf"), b"new");
        // Ensure new_snap is modified after old_snap
        let _ = fs::write(new_snap.join("model.gguf"), b"newer");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.load_from_cache("org/model", &[]);
        assert!(result.is_ok());
        let files = result.unwrap();
        // Should pick the snapshot that was modified later
        assert!(files.weights[0].to_string_lossy().contains("new_rev"));
        let _ = fs::remove_dir_all(&dir);
    }

    // ── from_cache convenience function ──

    #[test]
    fn from_cache_missing_dir() {
        let dir = std::env::temp_dir().join("gllm_test_from_cache_missing_").join("no");
        let result = from_cache(dir, "org/model");
        assert!(result.is_err());
    }

    // ── find_latest_snapshot ──

    #[test]
    fn find_latest_snapshot_no_entries() {
        let dir = std::env::temp_dir().join("gllm_test_snapshots_empty");
        let snapshots = dir.join("snapshots");
        let _ = fs::create_dir_all(&snapshots);
        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.find_latest_snapshot(&snapshots);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_latest_snapshot_single_entry() {
        let dir = std::env::temp_dir().join("gllm_test_snapshots_single");
        let snapshots = dir.join("snapshots");
        let snap = snapshots.join("only_one");
        let _ = fs::create_dir_all(&snap);
        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.find_latest_snapshot(&snapshots);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().file_name().unwrap(), "only_one");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_latest_snapshot_nonexistent_dir() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let result = client.find_latest_snapshot(Path::new("/nonexistent/path/snapshots"));
        assert!(result.is_err());
    }

    // ── parse_safetensors_index error paths ──

    #[test]
    fn parse_safetensors_index_invalid_json() {
        let dir = std::env::temp_dir().join("gllm_test_parse_bad_json");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("index.json"), b"not json");
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let result = client.parse_safetensors_index(&dir.join("index.json"));
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_safetensors_index_missing_file() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let result = client.parse_safetensors_index(Path::new("/nonexistent/index.json"));
        assert!(result.is_err());
    }

    // ── download_by_format ──

    #[test]
    fn download_by_format_pytorch_returns_none() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let result = client.download_by_format(
            "org/model",
            &[],
            &ModelScopeDownloader::new(PathBuf::from("/tmp"), Some("https://example.com".to_string())).unwrap(),
            &[],
            WeightFormat::PyTorch,
            None,
        ).unwrap();
        assert!(result.is_none(), "PyTorch format should return None (not downloadable)");
    }

    #[test]
    fn download_by_format_gllm_returns_none() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let downloader = ModelScopeDownloader::new(PathBuf::from("/tmp"), Some("https://example.com".to_string())).unwrap();
        let result = client.download_by_format(
            "org/model",
            &[],
            &downloader,
            &[],
            WeightFormat::Gllm,
            None,
        ).unwrap();
        assert!(result.is_none(), ".gllm format should return None (created offline)");
    }

    // ── resolve_onnx_external_repo_path additional edge cases ──

    #[test]
    fn resolve_onnx_external_empty_location() {
        let result = resolve_onnx_external_repo_path("model.onnx", ".");
        assert!(result.is_err(), "dot-only location should fail");
    }

    #[test]
    fn resolve_onnx_external_root_path() {
        let result = resolve_onnx_external_repo_path("model.onnx", "/etc/passwd");
        assert!(result.is_err(), "absolute path should be rejected");
    }

    // ── select_first_cached additional edge cases ──

    #[test]
    fn select_first_cached_no_match_in_nonempty_dir() {
        let dir = std::env::temp_dir().join("gllm_test_select_no_match");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("model.safetensors"), b"st");
        let result = select_first_cached(&dir, "gguf");
        assert!(result.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn select_first_cached_multiple_gguf_picks_sorted() {
        let dir = std::env::temp_dir().join("gllm_test_select_multi");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("z-model.gguf"), b"z");
        let _ = fs::write(dir.join("a-model.gguf"), b"a");
        let _ = fs::write(dir.join("m-model.gguf"), b"m");

        let result = select_first_cached(&dir, "gguf");
        assert!(result.is_some());
        let name = result.unwrap().file_name().unwrap().to_string_lossy().to_string();
        assert_eq!(name, "a-model.gguf");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn select_first_cached_onnx_prefers_subdir_over_root() {
        let dir = std::env::temp_dir().join("gllm_test_onnx_pref");
        let onnx_dir = dir.join("onnx");
        let _ = fs::create_dir_all(&onnx_dir);
        let _ = fs::write(onnx_dir.join("model.onnx"), b"sub");
        let _ = fs::write(dir.join("root.onnx"), b"root");

        let result = select_first_cached(&dir, "onnx");
        assert!(result.is_some());
        let path_str = result.unwrap().to_string_lossy().to_string();
        assert!(path_str.contains("onnx/model.onnx"), "onnx/ subdir should be checked first: {path_str}");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── find_files_with_extension: multiple files, all returned ──

    #[test]
    fn find_files_with_extension_multiple_matches() {
        let dir = std::env::temp_dir().join("gllm_test_multi_ext");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("a.onnx"), b"a");
        let _ = fs::write(dir.join("b.onnx"), b"b");
        let _ = fs::write(dir.join("c.safetensors"), b"c");

        let found = find_files_with_extension(&dir, "onnx");
        assert_eq!(found.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    // ── candidate_names: empty file_map ──

    #[test]
    fn candidate_names_empty_file_map() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.candidate_names(&[], "model.safetensors");
        assert!(names.contains(&"model.safetensors".to_string()));
        assert!(names.iter().any(|n| n == "model/model.safetensors"));
        assert!(names.iter().any(|n| n == "weights/model.safetensors"));
    }

    // ── gguf_candidate_names: all quant variants present ──

    #[test]
    fn gguf_candidate_names_repo_specific_quant_variants() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("org/my-model");
        let has_q4 = names.iter().any(|n| n.contains("my-model-Q4_0.gguf"));
        let has_q8 = names.iter().any(|n| n.contains("my-model-Q8_0.gguf"));
        let has_f16 = names.iter().any(|n| n.contains("my-model-f16.gguf"));
        assert!(has_q4, "should include Q4_0 variant");
        assert!(has_q8, "should include Q8_0 variant");
        assert!(has_f16, "should include f16 variant");
    }

    #[test]
    fn gguf_candidate_names_dot_variant() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("org/my-model");
        let has_dot_q4 = names.iter().any(|n| n.contains("my-model.Q4_0.gguf"));
        assert!(has_dot_q4, "should include dot-separated variant");
    }

    // ── normalize_repo_path: rooted absolute path rejected ──

    #[test]
    fn normalize_repo_path_rejects_absolute_root() {
        // On Linux, absolute paths start with RootDir component
        let result = normalize_repo_path(Path::new("/etc/passwd"));
        assert!(result.is_err(), "absolute root path should be rejected");
    }

    // ── SafetensorsIndex: serde missing field ──

    #[test]
    fn safetensors_index_missing_weight_map_field() {
        let json = r#"{"other_field": 42}"#;
        let result = serde_json::from_str::<SafetensorsIndex>(json);
        assert!(result.is_err(), "missing weight_map should fail deserialization");
    }

    // ════════════════════════════════════════════════════════════════
    //  New tests — 45+ additional unit tests
    // ════════════════════════════════════════════════════════════════

    // ── LoaderError Display strings ──

    #[test]
    fn loader_error_display_missing_weights() {
        let err = LoaderError::MissingWeights;
        let msg = format!("{err}");
        assert!(
            msg.to_ascii_lowercase().contains("missing"),
            "MissingWeights display should contain 'missing': {msg}"
        );
    }

    #[test]
    fn loader_error_display_io() {
        let err = LoaderError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file gone"));
        let msg = format!("{err}");
        assert!(msg.contains("file gone"), "IO error display should contain message: {msg}");
    }

    #[test]
    fn loader_error_display_json() {
        let err = LoaderError::Json(serde_json::from_str::<i32>("bad").unwrap_err());
        let msg = format!("{err}");
        assert!(!msg.is_empty(), "JSON error display should not be empty");
    }

    #[test]
    fn loader_error_display_network() {
        let err = LoaderError::Network("connection refused".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("connection refused"), "Network error display should contain message: {msg}");
    }

    #[test]
    fn loader_error_display_cache() {
        let err = LoaderError::Cache("corrupted".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("corrupted"), "Cache error display should contain message: {msg}");
    }

    #[test]
    fn loader_error_display_hf_hub() {
        let err = LoaderError::HfHub("hub error detail".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("hub error detail"), "HfHub error display should contain message: {msg}");
    }

    #[test]
    fn loader_error_display_onnx() {
        let err = LoaderError::Onnx("parse failed".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("parse failed"), "Onnx error display should contain message: {msg}");
    }

    #[test]
    fn loader_error_display_gguf() {
        let err = LoaderError::Gguf("bad header".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("bad header"), "Gguf error display should contain message: {msg}");
    }

    #[test]
    fn loader_error_display_backend() {
        let err = LoaderError::Backend("cuda failed".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("cuda failed"), "Backend error display should contain message: {msg}");
    }

    #[test]
    fn loader_error_display_pytorch() {
        let err = LoaderError::Pytorch("invalid pickle".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("invalid pickle"), "Pytorch error display should contain message: {msg}");
    }

    #[test]
    fn loader_error_display_arch_detection() {
        let err = LoaderError::ArchDetection("unknown layout".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("unknown layout"), "ArchDetection display should contain message: {msg}");
    }

    #[test]
    fn loader_error_display_duplicate_tensor() {
        let err = LoaderError::DuplicateTensor("layer.weight".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("layer.weight"), "DuplicateTensor display should contain tensor name: {msg}");
    }

    #[test]
    fn loader_error_display_missing_tensor() {
        let err = LoaderError::MissingTensor("embed.weight".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("embed.weight"), "MissingTensor display should contain tensor name: {msg}");
    }

    // ── LoaderError From conversions ──

    #[test]
    fn loader_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let loader_err: LoaderError = io_err.into();
        match loader_err {
            LoaderError::Io(_) => {}
            other => panic!("expected Io variant, got: {other:?}"),
        }
    }

    #[test]
    fn loader_error_from_json_error() {
        let json_err: serde_json::Error = serde_json::from_str::<i32>("not number").unwrap_err();
        let loader_err: LoaderError = json_err.into();
        match loader_err {
            LoaderError::Json(_) => {}
            other => panic!("expected Json variant, got: {other:?}"),
        }
    }

    // ── WeightFormat all variants Debug format ──

    #[test]
    fn weight_format_debug_all_variants() {
        assert_eq!(format!("{:?}", WeightFormat::SafeTensors), "SafeTensors");
        assert_eq!(format!("{:?}", WeightFormat::Gguf), "Gguf");
        assert_eq!(format!("{:?}", WeightFormat::Onnx), "Onnx");
        assert_eq!(format!("{:?}", WeightFormat::PyTorch), "PyTorch");
        assert_eq!(format!("{:?}", WeightFormat::Gllm), "Gllm");
    }

    #[test]
    fn weight_format_all_ne_pairwise() {
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "{a:?} should not equal {b:?}");
                }
            }
        }
    }

    #[test]
    fn weight_format_eq_self_all() {
        assert_eq!(WeightFormat::SafeTensors, WeightFormat::SafeTensors);
        assert_eq!(WeightFormat::Gguf, WeightFormat::Gguf);
        assert_eq!(WeightFormat::Onnx, WeightFormat::Onnx);
        assert_eq!(WeightFormat::PyTorch, WeightFormat::PyTorch);
        assert_eq!(WeightFormat::Gllm, WeightFormat::Gllm);
    }

    // ── SafetensorsIndex: large weight_map ──

    #[test]
    fn safetensors_index_large_weight_map() {
        let mut entries = Vec::new();
        for i in 0..100 {
            let shard = format!("shard-{:03}.st", i % 5);
            entries.push(format!(r#""layer_{i}.weight":"{shard}""#));
        }
        let json = format!("{{\"weight_map\":{{{}}}}}", entries.join(","));
        let index: SafetensorsIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(index.weight_map.len(), 100);
        let shard_files = index.shard_files();
        assert_eq!(shard_files.len(), 5, "should deduplicate to 5 shards");
        assert!(shard_files.windows(2).all(|w| w[0] <= w[1]), "should be sorted");
    }

    #[test]
    fn safetensors_index_shard_files_deterministic_order() {
        let json = r#"{"weight_map":{"z":"c.st","y":"a.st","x":"b.st"}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        let files1 = index.shard_files();
        let files2 = index.shard_files();
        assert_eq!(files1, files2, "shard_files should return deterministic order");
    }

    #[test]
    fn safetensors_index_single_weight_single_shard() {
        let json = r#"{"weight_map":{"w":"s.bin"}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        let files = index.shard_files();
        assert_eq!(files, vec!["s.bin"]);
    }

    #[test]
    fn safetensors_index_unicode_key() {
        let json = r#"{"weight_map":{"权重":"shard.st"}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        assert!(index.weight_map.contains_key("权重"));
        assert_eq!(index.shard_files(), vec!["shard.st"]);
    }

    // ── normalize_repo_path: additional edge cases ──

    #[test]
    fn normalize_repo_path_two_components() {
        let result = normalize_repo_path(Path::new("a/b")).unwrap();
        assert_eq!(result, "a/b");
    }

    #[test]
    fn normalize_repo_path_many_dot_interleaved() {
        let result = normalize_repo_path(Path::new("./a/./b/./c")).unwrap();
        assert_eq!(result, "a/b/c");
    }

    #[test]
    fn normalize_repo_path_rejects_prefix_component() {
        // On Windows, C:\ prefix would be a Prefix component; on Linux this path has RootDir
        // Test with a path that has RootDir
        let result = normalize_repo_path(Path::new("/absolute/path"));
        assert!(result.is_err(), "root-absolute path should be rejected");
    }

    // ── resolve_onnx_external_repo_path: additional edge cases ──

    #[test]
    fn resolve_onnx_external_repo_path_simple_filename() {
        let result = resolve_onnx_external_repo_path("model.onnx", "data.bin").unwrap();
        assert_eq!(result, "data.bin");
    }

    #[test]
    fn resolve_onnx_external_repo_path_nested_model() {
        let result =
            resolve_onnx_external_repo_path("v2/onnx/model.onnx", "weights.bin").unwrap();
        assert_eq!(result, "v2/onnx/weights.bin");
    }

    #[test]
    fn resolve_onnx_external_repo_path_dot_dot_rejected() {
        let result = resolve_onnx_external_repo_path("onnx/model.onnx", "../secret.bin");
        assert!(result.is_err(), "path traversal should be rejected");
    }

    // ── push_unique_path: additional edge cases ──

    #[test]
    fn push_unique_path_preserves_order() {
        let mut paths = vec![
            PathBuf::from("a.st"),
            PathBuf::from("b.st"),
            PathBuf::from("c.st"),
        ];
        push_unique_path(&mut paths, PathBuf::from("d.st"));
        assert_eq!(paths.len(), 4);
        assert_eq!(paths[3], PathBuf::from("d.st"));
    }

    #[test]
    fn push_unique_path_multiple_duplicates() {
        let mut paths = vec![PathBuf::from("a.st")];
        push_unique_path(&mut paths, PathBuf::from("a.st"));
        push_unique_path(&mut paths, PathBuf::from("a.st"));
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn push_unique_path_same_name_different_parent() {
        let mut paths = vec![PathBuf::from("/dir1/model.st")];
        push_unique_path(&mut paths, PathBuf::from("/dir2/model.st"));
        assert_eq!(paths.len(), 2, "same filename different parent should be distinct");
    }

    // ── candidate_names: additional edge cases ──

    #[test]
    fn candidate_names_no_duplicate_after_file_map() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        // If file_map maps config.json -> config.json (identity), should not duplicate
        let file_map: FileMap = &[("config.json", "config.json")];
        let names = client.candidate_names(file_map, "config.json");
        let config_count = names.iter().filter(|n| **n == "config.json").count();
        assert_eq!(config_count, 1, "identity mapping should not duplicate entries");
    }

    #[test]
    fn candidate_names_file_map_does_not_affect_unmapped() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let file_map: FileMap = &[("config.json", "alt.json")];
        let names = client.candidate_names(file_map, "tokenizer.json");
        assert!(
            names.iter().any(|n| n == "tokenizer.json"),
            "unmapped name should pass through unchanged"
        );
    }

    // ── gguf_candidate_names: additional edge cases ──

    #[test]
    fn gguf_candidate_names_default_count() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("org/model");
        // 4 defaults + 3 quant × 2 variants = 10
        assert_eq!(names.len(), 10, "expected 4 defaults + 6 repo-specific = 10 total");
    }

    #[test]
    fn gguf_candidate_names_all_end_with_gguf() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("org/model");
        for name in &names {
            assert!(
                name.to_ascii_lowercase().ends_with(".gguf"),
                "all candidates should end with .gguf: {name}"
            );
        }
    }

    // ── map_name: additional edge cases ──

    #[test]
    fn map_name_empty_file_map() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let result = client.map_name(&[], "anything.json");
        assert_eq!(result, "anything.json");
    }

    #[test]
    fn map_name_multiple_entries_finds_first_match() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let file_map: FileMap = &[("x", "first"), ("y", "second"), ("x", "third")];
        assert_eq!(client.map_name(file_map, "x"), "first");
        assert_eq!(client.map_name(file_map, "y"), "second");
    }

    // ── MsModelFiles: additional struct tests ──

    #[test]
    fn ms_model_files_multiple_weights() {
        let files = MsModelFiles {
            repo: "org/model".to_string(),
            weights: vec![
                PathBuf::from("shard-1.st"),
                PathBuf::from("shard-2.st"),
                PathBuf::from("shard-3.st"),
            ],
            format: WeightFormat::SafeTensors,
            aux_files: vec![PathBuf::from("config.json")],
        };
        assert_eq!(files.weights.len(), 3);
    }

    #[test]
    fn ms_model_files_format_variants() {
        for (fmt, name) in [
            (WeightFormat::SafeTensors, "SafeTensors"),
            (WeightFormat::Gguf, "Gguf"),
            (WeightFormat::Onnx, "Onnx"),
            (WeightFormat::PyTorch, "PyTorch"),
            (WeightFormat::Gllm, "Gllm"),
        ] {
            let files = MsModelFiles {
                repo: String::new(),
                weights: vec![],
                format: fmt,
                aux_files: vec![],
            };
            assert_eq!(files.format, fmt, "format should be {name}");
        }
    }

    // ── select_first_cached: additional edge cases ──

    #[test]
    fn select_first_cached_case_insensitive_extension() {
        let dir = std::env::temp_dir().join("gllm_test_select_case");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("model.GGUF"), b"gguf");

        let result = select_first_cached(&dir, "gguf");
        assert!(result.is_some(), "should match case-insensitively");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn select_first_cached_no_onnx_in_empty_onnx_dir() {
        let dir = std::env::temp_dir().join("gllm_test_empty_onnx_subdir");
        let onnx_dir = dir.join("onnx");
        let _ = fs::create_dir_all(&onnx_dir);

        let result = select_first_cached(&dir, "onnx");
        assert!(result.is_none(), "empty onnx/ subdir should yield None");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── find_files_with_extension: additional edge cases ──

    #[test]
    fn find_files_with_extension_empty_directory() {
        let dir = std::env::temp_dir().join("gllm_test_find_empty_dir");
        let _ = fs::create_dir_all(&dir);

        let found = find_files_with_extension(&dir, "gguf");
        assert!(found.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_files_with_extension_mixed_extensions() {
        let dir = std::env::temp_dir().join("gllm_test_find_mixed");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("model.gguf"), b"g");
        let _ = fs::write(dir.join("model.onnx"), b"o");
        let _ = fs::write(dir.join("model.safetensors"), b"s");
        let _ = fs::write(dir.join("config.json"), b"j");

        let gguf = find_files_with_extension(&dir, "gguf");
        let onnx = find_files_with_extension(&dir, "onnx");
        let st = find_files_with_extension(&dir, "safetensors");

        assert_eq!(gguf.len(), 1);
        assert_eq!(onnx.len(), 1);
        assert_eq!(st.len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_files_with_extension_uppercase_query() {
        let dir = std::env::temp_dir().join("gllm_test_find_upper_query");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("model.gguf"), b"g");

        // Query "GGUF" should match "gguf" case-insensitively
        let found = find_files_with_extension(&dir, "GGUF");
        assert_eq!(found.len(), 1, "uppercase query should match lowercase extension");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── LoaderError additional variant Display ──

    #[test]
    fn loader_error_display_unsupported_weight_extension() {
        let err = LoaderError::UnsupportedWeightExtension(".xyz".to_string());
        let msg = format!("{err}");
        assert!(msg.contains(".xyz"), "display should contain extension: {msg}");
    }

    #[test]
    fn loader_error_display_format_not_found() {
        let err = LoaderError::FormatNotFound(WeightFormat::Gguf);
        let msg = format!("{err}");
        assert!(
            msg.to_ascii_lowercase().contains("not found"),
            "display should contain 'not found': {msg}"
        );
    }

    #[test]
    fn loader_error_display_multiple_weight_formats() {
        let err = LoaderError::MultipleWeightFormats(vec![
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
        ]);
        let msg = format!("{err}");
        assert!(
            msg.to_ascii_lowercase().contains("multiple"),
            "display should contain 'multiple': {msg}"
        );
    }

    #[test]
    fn loader_error_display_authentication() {
        let err = LoaderError::AuthenticationError {
            hint: "set HF_TOKEN".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("set HF_TOKEN"), "display should contain hint: {msg}");
    }

    #[test]
    fn loader_error_display_invalid_quantization() {
        let err = LoaderError::InvalidQuantization("bad block size".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("bad block size"), "display should contain message: {msg}");
    }

    // ── ModelScopeClient: cache_dir storage ──

    #[test]
    fn client_new_preserves_cache_dir() {
        let path = PathBuf::from("/some/deep/nested/cache/path");
        let client = ModelScopeClient::new(path.clone()).unwrap();
        assert_eq!(client.cache_dir, path);
    }

    #[test]
    fn client_new_relative_path() {
        let path = PathBuf::from("relative/cache");
        let client = ModelScopeClient::new(path.clone()).unwrap();
        assert_eq!(client.cache_dir, path);
    }

    // ── normalize_repo_path: error message content ──

    #[test]
    fn normalize_repo_path_error_message_contains_path() {
        let bad_path = "../etc/passwd";
        let result = normalize_repo_path(Path::new(bad_path));
        match result {
            Err(LoaderError::Onnx(msg)) => {
                assert!(msg.contains(bad_path), "error message should contain path: {msg}");
            }
            Err(other) => panic!("expected Onnx error, got: {other:?}"),
            Ok(_) => panic!("expected error for parent dir traversal"),
        }
    }

    #[test]
    fn normalize_repo_path_empty_location_error_message() {
        let result = normalize_repo_path(Path::new("."));
        match result {
            Err(LoaderError::Onnx(msg)) => {
                assert!(
                    msg.to_ascii_lowercase().contains("empty"),
                    "error for empty path should mention 'empty': {msg}"
                );
            }
            Err(other) => panic!("expected Onnx error, got: {other:?}"),
            Ok(_) => panic!("expected error for dot-only path"),
        }
    }

    // ── from_cache convenience function ──

    #[test]
    fn from_cache_error_on_missing() {
        let dir = std::env::temp_dir()
            .join("gllm_test_from_cache_noexist")
            .join(uuid_path());
        let result = from_cache(dir, "org/nonexistent");
        assert!(result.is_err());
    }

    // ── Helper: unique temp path to avoid test interference ──

    fn uuid_path() -> String {
        format!("test-{}", std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos())
    }

    // ── WeightFormat: Hash consistency across calls ──

    #[test]
    fn weight_format_hash_stable() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(WeightFormat::SafeTensors, "st");
        map.insert(WeightFormat::Gguf, "gguf");
        // Insert again and check values are stable
        assert_eq!(map.get(&WeightFormat::SafeTensors), Some(&"st"));
        assert_eq!(map.get(&WeightFormat::Gguf), Some(&"gguf"));
        assert_eq!(map.get(&WeightFormat::Onnx), None);
    }

    // ── SafetensorsIndex: serde extra field ignored ──

    #[test]
    fn safetensors_index_ignores_extra_fields() {
        let json = r#"{"weight_map":{"a":"s.st"},"extra":123}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.shard_files(), vec!["s.st"]);
    }

    // ── SafetensorsIndex: serde rename weight_map ──

    #[test]
    fn safetensors_index_weight_map_rename() {
        let json = r#"{"weight_map":{"k":"v"}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.weight_map["k"], "v");
    }

    // ── LoaderError: Debug output includes variant name ──

    #[test]
    fn loader_error_debug_format() {
        let err = LoaderError::MissingWeights;
        let debug = format!("{err:?}");
        assert!(debug.contains("MissingWeights"), "Debug should contain variant name: {debug}");
    }

    #[test]
    fn loader_error_debug_network() {
        let err = LoaderError::Network("timeout".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("Network"), "Debug should contain variant name: {debug}");
    }

    // ── onnx_candidate_names: deterministic ──

    #[test]
    fn onnx_candidate_names_deterministic() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let first = client.onnx_candidate_names();
        let second = client.onnx_candidate_names();
        assert_eq!(first, second, "should produce identical results across calls");
    }

    // ════════════════════════════════════════════════════════════════
    //  13 additional tests (164 → 177)
    // ════════════════════════════════════════════════════════════════

    // ── load_from_cache: safetensors priority over gguf when both present ──

    #[test]
    fn load_from_cache_safetensors_priority_over_gguf() {
        let dir = std::env::temp_dir().join("gllm_test_st_over_gguf");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let snapshot = snapshots.join("rev1");
        let _ = fs::create_dir_all(&snapshot);
        let _ = fs::write(snapshot.join("model.safetensors"), b"st");
        let _ = fs::write(snapshot.join("model.gguf"), b"gguf");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.load_from_cache("org/model", &[]);
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.format, WeightFormat::SafeTensors,
            "safetensors should be preferred over gguf when both exist");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── load_from_cache: aux files include merges and vocab ──

    #[test]
    fn load_from_cache_collects_all_aux_files() {
        let dir = std::env::temp_dir().join("gllm_test_all_aux");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let snapshot = snapshots.join("rev1");
        let _ = fs::create_dir_all(&snapshot);
        let _ = fs::write(snapshot.join("model.safetensors"), b"st");
        let _ = fs::write(snapshot.join("config.json"), b"{}");
        let _ = fs::write(snapshot.join("tokenizer.json"), b"{}");
        let _ = fs::write(snapshot.join("merges.txt"), b"merges");
        let _ = fs::write(snapshot.join("vocab.json"), b"{}");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let files = client.load_from_cache("org/model", &[]).unwrap();
        let aux_names: Vec<_> = files.aux_files.iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
            .collect();
        assert!(aux_names.contains(&"config.json"), "should include config.json");
        assert!(aux_names.contains(&"tokenizer.json"), "should include tokenizer.json");
        assert!(aux_names.contains(&"merges.txt"), "should include merges.txt");
        assert!(aux_names.contains(&"vocab.json"), "should include vocab.json");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── load_from_cache: onnx in subdir discovered via fallback ──

    #[test]
    fn load_from_cache_onnx_subdir_fallback() {
        let dir = std::env::temp_dir().join("gllm_test_onnx_subdir_fb");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let snapshot = snapshots.join("rev1");
        let onnx_sub = snapshot.join("onnx");
        let _ = fs::create_dir_all(&onnx_sub);
        let _ = fs::write(onnx_sub.join("model.onnx"), b"onnx");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.load_from_cache("org/model", &[]);
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.format, WeightFormat::Onnx);
        let _ = fs::remove_dir_all(&dir);
    }

    // ── load_from_cache: gguf priority over onnx (neither safetensors) ──

    #[test]
    fn load_from_cache_gguf_priority_over_onnx() {
        let dir = std::env::temp_dir().join("gllm_test_gguf_over_onnx");
        let model_dir = dir.join("models--").join("org--model");
        let snapshots = model_dir.join("snapshots");
        let snapshot = snapshots.join("rev1");
        let _ = fs::create_dir_all(&snapshot);
        let _ = fs::write(snapshot.join("model.gguf"), b"gguf");
        let _ = fs::write(snapshot.join("model.onnx"), b"onnx");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let files = client.load_from_cache("org/model", &[]).unwrap();
        assert_eq!(files.format, WeightFormat::Gguf,
            "gguf should be preferred over onnx when neither safetensors exists");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── gguf_candidate_names: empty string repo (edge case) ──

    #[test]
    fn gguf_candidate_names_empty_repo_string() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.gguf_candidate_names("");
        // Empty string has no '/' so split('/').next_back() gives ""
        // format!("{base}-{quant}.gguf") produces "-Q4_0.gguf" etc.
        // The 4 default candidates must still be present
        assert!(names.contains(&"model.gguf".to_string()),
            "should still contain default candidates for empty repo");
        assert!(names.contains(&"ggml-model-q4_0.gguf".to_string()),
            "should still contain ggml default candidates");
        // Empty base may or may not produce variants like "-Q4_0.gguf"
        // depending on whether the function filters empty repo names.
        // The important invariant is that default candidates are always present.
        let has_defaults = names.contains(&"model.gguf".to_string())
            || names.contains(&"ggml-model-q4_0.gguf".to_string());
        assert!(has_defaults, "default candidates must always be present");
    }

    // ── candidate_names: logical with slash uses exact path ──

    #[test]
    fn candidate_names_config_json_produces_correct_count() {
        let client = ModelScopeClient::new(PathBuf::from("/tmp")).unwrap();
        let names = client.candidate_names(&[], "config.json");
        // config.json + configuration.json + model/ and weights/ prefixes for each
        // = config.json, configuration.json, model/config.json, model/configuration.json,
        //   weights/config.json, weights/configuration.json
        assert!(names.len() >= 4,
            "config.json should produce multiple candidates including alias and prefixes");
    }

    // ── list_cached_models: sorts output ──

    #[test]
    fn list_cached_models_returns_sorted() {
        let dir = std::env::temp_dir().join("gllm_test_list_sorted");
        let models_dir = dir.join("models--");
        let _ = fs::create_dir_all(models_dir.join("zeta--model"));
        let _ = fs::create_dir_all(models_dir.join("alpha--model"));
        let _ = fs::create_dir_all(models_dir.join("mid--model"));

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let models = client.list_cached_models().unwrap();
        assert_eq!(models.len(), 3);
        let mut sorted = models.clone();
        sorted.sort();
        assert_eq!(models, sorted, "list_cached_models should return sorted results");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── find_latest_snapshot: picks newer of two ──

    #[test]
    fn find_latest_snapshot_picks_newer() {
        let dir = std::env::temp_dir().join("gllm_test_snap_newer");
        let snapshots = dir.join("snapshots");
        let old_snap = snapshots.join("old_rev");
        let new_snap = snapshots.join("new_rev");
        let _ = fs::create_dir_all(&old_snap);
        let _ = fs::create_dir_all(&new_snap);
        // Write old first, then new (ensuring different mtime)
        let _ = fs::write(old_snap.join("data.bin"), b"old");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = fs::write(new_snap.join("data.bin"), b"new");

        let client = ModelScopeClient::new(dir.clone()).unwrap();
        let result = client.find_latest_snapshot(&snapshots).unwrap();
        assert_eq!(result.file_name().unwrap(), "new_rev",
            "should pick the newer snapshot");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── select_first_cached: onnx at root level when no subdir ──

    #[test]
    fn select_first_cached_onnx_at_root_no_subdir() {
        let dir = std::env::temp_dir().join("gllm_test_onnx_root_only");
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("model.onnx"), b"onnx_at_root");
        // No onnx/ subdir exists

        let result = select_first_cached(&dir, "onnx");
        assert!(result.is_some(), "should find .onnx at root when no subdir exists");
        let name = result.unwrap().file_name().unwrap().to_string_lossy().to_string();
        assert_eq!(name, "model.onnx");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── push_unique_path: nested path uniqueness ──

    #[test]
    fn push_unique_path_nested_identical_paths() {
        let mut paths = vec![PathBuf::from("/a/b/c/model.st")];
        push_unique_path(&mut paths, PathBuf::from("/a/b/c/model.st"));
        assert_eq!(paths.len(), 1, "identical nested paths should be deduplicated");
    }

    // ── SafetensorsIndex: empty string shard name ──

    #[test]
    fn safetensors_index_empty_string_shard_name() {
        let json = r#"{"weight_map":{"layer.weight":""}}"#;
        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        let files = index.shard_files();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], "", "empty string should be a valid shard name");
    }

    // ── normalize_repo_path: unicode components ──

    #[test]
    fn normalize_repo_path_unicode_components() {
        let result = normalize_repo_path(Path::new("数据/模型/权重.bin")).unwrap();
        assert_eq!(result, "数据/模型/权重.bin");
    }

    // ── resolve_onnx_external_repo_path: multi-level dot-dot ──

    #[test]
    fn resolve_onnx_external_multi_level_traversal() {
        let result = resolve_onnx_external_repo_path(
            "onnx/model.onnx",
            "../../../../../etc/shadow",
        );
        assert!(result.is_err(), "multi-level path traversal should be rejected");
    }
}
