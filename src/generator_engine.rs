use crate::engine::{find_model_file, TokenizerAdapter};
use crate::generation::{GenerationConfig, GenerationOptions, GenerationOutput};
use crate::generator_model::GeneratorModel;
use crate::moe_generator_model::MoEGeneratorModel;
use crate::model_config::ModelConfig;
use crate::registry::{Architecture, ModelInfo, Quantization};
use crate::types::{Error, Result};
use gllm_kernels::backend::auto_select_backend;
use std::path::Path;

pub(crate) trait GeneratorModelTrait {
    fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
        options: &GenerationOptions,
    ) -> Result<GenerationOutput>;
    fn load_safetensors(&mut self, path: &Path) -> Result<()>;
    fn load_awq(&mut self, path: &Path) -> Result<()>;
    fn load_gguf(&mut self, path: &Path) -> Result<()>;
    fn max_position_embeddings(&self) -> usize;
}

impl GeneratorModelTrait for GeneratorModel {
    fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
        options: &GenerationOptions,
    ) -> Result<GenerationOutput> {
        GeneratorModel::generate(self, prompt_ids, config, tokenizer, options)
    }

    fn load_safetensors(&mut self, path: &Path) -> Result<()> {
        GeneratorModel::load_safetensors(self, path)
    }

    fn load_awq(&mut self, path: &Path) -> Result<()> {
        GeneratorModel::load_awq(self, path)
    }

    fn load_gguf(&mut self, path: &Path) -> Result<()> {
        GeneratorModel::load_gguf(self, path)
    }

    fn max_position_embeddings(&self) -> usize {
        GeneratorModel::max_position_embeddings(self)
    }
}

impl GeneratorModelTrait for MoEGeneratorModel {
    fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
        options: &GenerationOptions,
    ) -> Result<GenerationOutput> {
        MoEGeneratorModel::generate(self, prompt_ids, config, tokenizer, options)
    }

    fn load_safetensors(&mut self, path: &Path) -> Result<()> {
        MoEGeneratorModel::load_safetensors(self, path)
    }

    fn load_awq(&mut self, path: &Path) -> Result<()> {
        MoEGeneratorModel::load_awq(self, path)
    }

    fn load_gguf(&mut self, path: &Path) -> Result<()> {
        MoEGeneratorModel::load_gguf(self, path)
    }

    fn max_position_embeddings(&self) -> usize {
        MoEGeneratorModel::max_position_embeddings(self)
    }
}

pub(crate) struct GeneratorEngine {
    model: Box<dyn GeneratorModelTrait>,
}

impl GeneratorEngine {
    pub fn new(model_dir: &std::path::Path, info: &ModelInfo) -> Result<Self> {
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

        // Create backend ONCE at engine level
        let backend = auto_select_backend();

        let mut model: Box<dyn GeneratorModelTrait> = match info.architecture {
            Architecture::GLM4MoE
            | Architecture::Qwen3MoE
            | Architecture::Mixtral
            | Architecture::DeepSeekV3
            | Architecture::GptOss => Box::new(MoEGeneratorModel::new(config.clone(), backend)?),
            _ => Box::new(GeneratorModel::new(config.clone(), backend)?),
        };

        if let Some(model_path) = find_model_file(model_dir, &info.quantization) {
            match info.quantization {
                Quantization::GGUF => model.load_gguf(&model_path)?,
                Quantization::AWQ => model.load_awq(&model_path)?,
                _ => model.load_safetensors(&model_path)?,
            }
        }

        Ok(Self {
            model,
        })
    }

    pub fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
        options: &GenerationOptions,
    ) -> Result<GenerationOutput> {
        self.model.generate(prompt_ids, config, tokenizer, options)
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.model.max_position_embeddings()
    }
}
