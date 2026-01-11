use crate::engine::TokenizerAdapter;
use crate::generation::{GenerationConfig, GenerationOutput};
use crate::generator_model::GeneratorModel;
use crate::model_config::ModelConfig;
use crate::registry::ModelInfo;
use crate::types::{Error, Result};
use burn::tensor::backend::Backend;

#[derive(Clone)]
pub(crate) struct GeneratorEngine<B: Backend> {
    model: GeneratorModel<B>,
    max_position_embeddings: usize,
}

impl<B: Backend> GeneratorEngine<B> {
    pub fn new(device: B::Device, model_dir: &std::path::Path, info: &ModelInfo) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let repo_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .replace("--", "/");
        let config_file = config_path.exists().then_some(config_path.as_path());
        let (config, _) = ModelConfig::load(&repo_name, config_file)?;

        if info.model_type != crate::registry::ModelType::Generator {
            return Err(Error::InvalidConfig(
                "GeneratorEngine requires a generator model".into(),
            ));
        }

        let mut model = GeneratorModel::new(&device, config.clone())?;

        let safetensors_path = model_dir.join("model.safetensors");
        if safetensors_path.exists() {
            model.load_safetensors(&safetensors_path)?;
        }

        Ok(Self {
            model,
            max_position_embeddings: config.max_position_embeddings,
        })
    }

    pub fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
    ) -> Result<GenerationOutput> {
        self.model.generate(prompt_ids, config, tokenizer)
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
}
