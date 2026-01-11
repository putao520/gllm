use crate::engine::TokenizerAdapter;
use crate::registry::{ModelInfo, ModelRegistry};
use crate::types::{ClientConfig, Error, Result};
use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

impl ShardIndex {
    fn from_slice(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|err| Error::DownloadError(format!("Failed to parse shard index: {err}")))
    }

    fn from_path(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|err| {
            Error::DownloadError(format!(
                "Failed to read shard index {}: {err}",
                path.display()
            ))
        })?;
        Self::from_slice(&bytes).map_err(|err| {
            Error::DownloadError(format!(
                "Failed to parse shard index {}: {err}",
                path.display()
            ))
        })
    }

    fn shard_filenames(&self) -> Vec<String> {
        let mut shards = BTreeSet::new();
        for filename in self.weight_map.values() {
            shards.insert(filename.clone());
        }
        shards.into_iter().collect()
    }
}

fn cleanup_partial_downloads(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

/// Model artifacts prepared for inference.
pub(crate) struct ModelArtifacts {
    pub info: ModelInfo,
    pub model_dir: PathBuf,
    pub tokenizer: TokenizerAdapter,
}

/// Manage model resolution, download, and loading.
pub(crate) struct ModelManager {
    registry: ModelRegistry,
    config: ClientConfig,
}

impl ModelManager {
    /// Create a new model manager with the provided configuration.
    pub fn new(config: ClientConfig) -> Self {
        Self {
            registry: ModelRegistry::new(),
            config,
        }
    }

    /// Resolve and prepare a model for use.
    pub fn prepare(&self, model: &str) -> Result<ModelArtifacts> {
        let info = self.registry.resolve(model)?;
        let model_dir = self.repo_dir(&info.repo_id);
        if !model_dir.exists() {
            fs::create_dir_all(&model_dir)?;
            self.download_model(&info, &model_dir)?;
        }

        let tokenizer = TokenizerAdapter::from_dir(&model_dir)?;
        Ok(ModelArtifacts {
            info,
            model_dir,
            tokenizer,
        })
    }

    /// Download a model from HuggingFace (or ModelScope fallback) into the target directory.
    pub fn download_model(&self, info: &ModelInfo, target: &Path) -> Result<()> {
        fs::create_dir_all(target)?;

        let download_fn = |endpoint: Option<&str>| -> Result<()> {
            let api = if let Some(ep) = endpoint {
                use hf_hub::api::sync::ApiBuilder;
                ApiBuilder::new()
                    .with_endpoint(ep.to_string())
                    .build()
                    .map_err(|e| Error::DownloadError(e.to_string()))?
            } else {
                Api::new().map_err(|e| Error::DownloadError(e.to_string()))?
            };

            let repo = api.model(info.repo_id.clone());
            let mut downloaded_files = Vec::new();

            let shard_index_name = "model.safetensors.index.json";
            let shard_index_path = target.join(shard_index_name);
            let mut has_shards = false;

            if shard_index_path.exists() {
                has_shards = true;
            } else if let Ok(src) = repo.get(shard_index_name) {
                if let Err(err) = fs::copy(&src, &shard_index_path) {
                    let _ = fs::remove_file(&shard_index_path);
                    return Err(Error::DownloadError(err.to_string()));
                }
                downloaded_files.push(shard_index_path.clone());
                has_shards = true;
            }

            if has_shards {
                let shard_index = match ShardIndex::from_path(&shard_index_path) {
                    Ok(index) => index,
                    Err(err) => {
                        cleanup_partial_downloads(&downloaded_files);
                        return Err(err);
                    }
                };
                let shard_files = shard_index.shard_filenames();
                if shard_files.is_empty() {
                    cleanup_partial_downloads(&downloaded_files);
                    return Err(Error::DownloadError(
                        "Shard index contains no shard entries".into(),
                    ));
                }

                let total = shard_files.len();
                for (idx, shard) in shard_files.iter().enumerate() {
                    let dest = target.join(shard);
                    if dest.exists() {
                        continue;
                    }

                    eprintln!("Downloading shard {}/{}: {}", idx + 1, total, shard);
                    match repo.get(shard) {
                        Ok(src) => {
                            if let Err(err) = fs::copy(&src, &dest) {
                                let _ = fs::remove_file(&dest);
                                cleanup_partial_downloads(&downloaded_files);
                                return Err(Error::DownloadError(err.to_string()));
                            }
                            downloaded_files.push(dest);
                        }
                        Err(err) => {
                            cleanup_partial_downloads(&downloaded_files);
                            return Err(Error::DownloadError(err.to_string()));
                        }
                    }
                }
            } else {
                let dest = target.join("model.safetensors");
                if !dest.exists() {
                    match repo.get("model.safetensors") {
                        Ok(src) => {
                            if let Err(err) = fs::copy(&src, &dest) {
                                let _ = fs::remove_file(&dest);
                                return Err(Error::DownloadError(err.to_string()));
                            }
                        }
                        Err(err) => {
                            return Err(Error::DownloadError(err.to_string()));
                        }
                    }
                }
            }

            let files = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
            ];

            for file in files {
                let dest = target.join(file);
                if dest.exists() {
                    continue;
                }

                match repo.get(file) {
                    Ok(src) => {
                        if let Err(err) = fs::copy(&src, &dest) {
                            return Err(Error::DownloadError(err.to_string()));
                        }
                    }
                    Err(_) => {
                        // Config or tokenizer files might be absent for some repos.
                    }
                }
            }
            Ok(())
        };

        // 1. Try Hugging Face (default)
        if let Ok(()) = download_fn(None) {
            return Ok(());
        }

        // 2. Fallback: ModelScope (China)
        // Note: ModelScope uses the same repo IDs for most major upstream models.
        eprintln!("HuggingFace download failed, attempting ModelScope fallback...");
        download_fn(Some("https://modelscope.cn/api/v1"))
    }

    /// Validate that model files exist and are readable.
    pub fn validate_model_files(&self, model_dir: &Path) -> Result<()> {
        // Check for SafeTensors weights
        let weights = model_dir.join("model.safetensors");
        if weights.exists() {
            fs::metadata(&weights).map_err(Error::from)?;
        }

        // Check for config.json
        let config = model_dir.join("config.json");
        if config.exists() {
            fs::metadata(&config).map_err(Error::from)?;
        }

        // Check for tokenizer files
        let tokenizer = model_dir.join("tokenizer.json");
        if tokenizer.exists() {
            fs::metadata(&tokenizer).map_err(Error::from)?;
        }

        Ok(())
    }

    fn repo_dir(&self, repo_id: &str) -> PathBuf {
        let safe = repo_id.replace('/', "--");
        self.config.models_dir.join(safe)
    }
}

#[cfg(test)]
mod tests {
    use super::ShardIndex;

    #[test]
    fn shard_index_parses_weights() {
        let payload = r#"{
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00003.safetensors"
            }
        }"#;

        let index: ShardIndex = serde_json::from_str(payload).expect("parse shard index");
        assert_eq!(index.weight_map.len(), 2);
    }

    #[test]
    fn shard_index_extracts_unique_filenames() {
        let payload = r#"{
            "weight_map": {
                "a": "model-00001-of-00003.safetensors",
                "b": "model-00001-of-00003.safetensors",
                "c": "model-00002-of-00003.safetensors"
            }
        }"#;

        let index: ShardIndex = serde_json::from_str(payload).expect("parse shard index");
        let shards = index.shard_filenames();
        assert_eq!(
            shards,
            vec![
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors"
            ]
        );
    }
}
