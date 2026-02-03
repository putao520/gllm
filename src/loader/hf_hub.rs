//! HuggingFace integration.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;
use serde::Deserialize;

use crate::manifest::FileMap;

use super::{parallel::ParallelLoader, LoaderError, Result};

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
        let mut builder = hf_hub::api::sync::ApiBuilder::new().with_cache_dir(cache_dir);
        if let Some(endpoint) = endpoint {
            builder = builder.with_endpoint(endpoint);
        }
        let api = builder
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
        repo.get(filename)
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
        let shard_paths = parallel.map_paths(&shard_paths_list, |path| {
            let filename = path.to_string_lossy().to_string();
            api.model(repo_id.clone())
                .get(&filename)
                .map_err(|err| LoaderError::HfHub(err.to_string()))
        })?;

        Ok(shard_paths)
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
}
