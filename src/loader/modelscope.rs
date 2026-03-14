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

#[derive(Debug)]
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
    ) -> Result<MsModelFiles> {
        self.download_model_files_with_format(repo, file_map, parallel, None)
    }

    pub fn download_model_files_with_format(
        &self,
        repo: &str,
        file_map: FileMap,
        parallel: ParallelLoader,
        format_hint: Option<WeightFormat>,
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
                self.download_by_format(&repo, file_map, &downloader, &aux_files, format)?;
            return result.ok_or(LoaderError::MissingWeights);
        }

        if let Some(files) =
            self.try_download_safetensors(&repo, file_map, &downloader, &aux_files)?
        {
            return Ok(files);
        }
        if let Some(files) = self.try_download_gguf(&repo, file_map, &downloader, &aux_files)? {
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
    ) -> Result<Option<MsModelFiles>> {
        match format {
            WeightFormat::SafeTensors => {
                self.try_download_safetensors(repo, file_map, downloader, aux_files)
            }
            WeightFormat::Gguf => self.try_download_gguf(repo, file_map, downloader, aux_files),
            WeightFormat::Onnx => self.try_download_onnx(repo, file_map, downloader, aux_files),
            WeightFormat::PyTorch => Ok(None),
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
    ) -> Result<Option<MsModelFiles>> {
        for candidate in self.gguf_candidate_names(repo) {
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
            eprintln!("📥 [{}/{}] 下载分片: {}", idx + 1, shards.len(), filename);

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
            log::debug!("cannot read directory {}: {e}", dir.display());
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
            .unwrap_or(false);
        if matches {
            files.push(path);
        }
    }
    files
}

#[derive(Debug, Deserialize)]
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
        .unwrap_or_else(|| Path::new(""));
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
}
