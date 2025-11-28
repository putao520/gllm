use crate::engine::TokenizerAdapter;
use crate::registry::{ModelInfo, ModelRegistry};
use crate::types::{ClientConfig, Error, Result};
use hf_hub::api::sync::Api;
use std::fs;
use std::path::{Path, PathBuf};

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

    /// Download a model from HuggingFace into the target directory.
    pub fn download_model(&self, info: &ModelInfo, target: &Path) -> Result<()> {
        let skip = std::env::var("GLLM_SKIP_DOWNLOAD").is_ok();
        if skip {
            return Ok(());
        }

        fs::create_dir_all(target)?;
        let api = Api::new().map_err(|e| Error::DownloadError(e.to_string()))?;
        let repo = api.model(info.repo_id.clone());
        let files = [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
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
                Err(err) => {
                    // Config or tokenizer files might be absent for some repos; only error on weights.
                    if file == "model.safetensors" {
                        return Err(Error::DownloadError(err.to_string()));
                    }
                }
            }
        }

        Ok(())
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
