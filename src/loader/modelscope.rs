//! ModelScope (魔搭社区) 集成
//!
//! ModelScope 是中国的模型托管平台，许多中国模型在那里公开可用

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::manifest::FileMap;
use super::{LoaderError, Result, ParallelLoader, ProgressBar, ModelScopeDownloader};
use super::downloader::Downloader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    SafeTensors,
    Bin,
}

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
        _parallel: ParallelLoader,
    ) -> Result<MsModelFiles> {
        use super::ModelScopeDownloader;

        let repo = repo.to_string();
        let mut aux_files = Vec::new();

        // 创建 ModelScope 下载器
        let downloader = ModelScopeDownloader::new(self.cache_dir.clone(), Some("https://www.modelscope.cn".to_string()))?;

        // 下载辅助文件
        for name in [
            "config.json",
            "configuration.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
        ] {
            if let Ok(path) = self.get_file_any(&repo, file_map, name, &downloader) {
                aux_files.push(path);
            }
        }

        // 尝试下载 safetensors 分片
        if let Ok(index_path) = self.get_file_any(&repo, file_map, "model.safetensors.index.json", &downloader) {
            let shard_index = self.parse_safetensors_index(&index_path)?;
            let shard_files = shard_index.shard_files();
            let weights = self.download_shards(&repo, &shard_files, &downloader)?;
            aux_files.push(index_path);
            return Ok(MsModelFiles {
                repo,
                weights,
                format: WeightFormat::SafeTensors,
                aux_files,
            });
        }

        // 尝试下载单个 safetensors 文件
        if let Ok(path) = self.get_file_any(&repo, file_map, "model.safetensors", &downloader) {
            return Ok(MsModelFiles {
                repo,
                weights: vec![path],
                format: WeightFormat::SafeTensors,
                aux_files,
            });
        }

        // 尝试下载 pytorch 分片
        if let Ok(index_path) = self.get_file_any(&repo, file_map, "pytorch_model.bin.index.json", &downloader) {
            let shard_index = self.parse_safetensors_index(&index_path)?;
            let shard_files = shard_index.shard_files();
            let weights = self.download_shards(&repo, &shard_files, &downloader)?;
            aux_files.push(index_path);
            return Ok(MsModelFiles {
                repo,
                weights,
                format: WeightFormat::Bin,
                aux_files,
            });
        }

        // 尝试下载单个 pytorch 文件
        if let Ok(path) = self.get_file_any(&repo, file_map, "pytorch_model.bin", &downloader) {
            return Ok(MsModelFiles {
                repo,
                weights: vec![path],
                format: WeightFormat::Bin,
                aux_files,
            });
        }

        Err(LoaderError::MissingWeights)
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
            let path = downloader.download_file_with_progress(repo, filename, &self.cache_dir, &mut progress)?;
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
    pub fn load_from_cache(
        &self,
        repo: &str,
        _file_map: FileMap,
    ) -> Result<MsModelFiles> {
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

        // 查找 .bin 权重
        let bin_file = snapshot.join("pytorch_model.bin");
        if bin_file.exists() {
            weights.push(bin_file);
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
            return Err(LoaderError::MissingWeights);
        }

        let format = if weights.iter().any(|w| w.to_string_lossy().ends_with(".safetensors")) {
            WeightFormat::SafeTensors
        } else {
            WeightFormat::Bin
        };

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

        for entry in fs::read_dir(snapshots_dir)
            .map_err(|e| LoaderError::Io(e))?
        {
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
        serde_json::from_slice(&content).map_err(|e| LoaderError::Json(e))
    }

    /// 列出缓存中可用的模型
    pub fn list_cached_models(&self) -> Result<Vec<String>> {
        let models_dir = self.cache_dir.join("models--");
        let mut models = Vec::new();

        if !models_dir.exists() {
            return Ok(models);
        }

        for entry in fs::read_dir(&models_dir)
            .map_err(|e| LoaderError::Io(e))?
        {
            let name = entry?.file_name();
            // 转换回 org/name 格式
            let normalized = name.to_string_lossy().replace("--", "/");
            models.push(normalized);
        }

        models.sort();
        Ok(models)
    }
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

/// 从 ModelScope 缓存加载模型
pub fn from_cache(
    cache_dir: PathBuf,
    repo: &str,
) -> Result<MsModelFiles> {
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
        assert!(index.shard_files().contains(&"model-00001-of-00002.safetensors".to_string()));
        assert!(index.shard_files().contains(&"model-00002-of-00002.safetensors".to_string()));
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
