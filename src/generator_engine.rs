use crate::engine::{find_model_file, TokenizerAdapter};
use crate::generation::{GenerationConfig, GenerationOutput};
use crate::generator_model::GeneratorModel;
use crate::moe_generator_model::MoEGeneratorModel;
use crate::model_config::ModelConfig;
use crate::registry::{Architecture, ModelInfo, Quantization};
use crate::types::{Error, Result};
use burn::tensor::backend::Backend;
use std::path::Path;

#[derive(Clone)]
enum GeneratorVariant<B: Backend> {
    Dense(GeneratorModel<B>),
    Moe(MoEGeneratorModel<B>),
}

impl<B: Backend> GeneratorVariant<B> {
    fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
    ) -> Result<GenerationOutput> {
        match self {
            Self::Dense(model) => model.generate(prompt_ids, config, tokenizer),
            Self::Moe(model) => model.generate(prompt_ids, config, tokenizer),
        }
    }

    fn load_safetensors(&mut self, path: &Path) -> Result<()> {
        match self {
            Self::Dense(model) => model.load_safetensors(path),
            Self::Moe(model) => model.load_safetensors(path),
        }
    }

    fn load_awq(&mut self, path: &Path) -> Result<()> {
        match self {
            Self::Dense(model) => model.load_awq(path),
            Self::Moe(_) => Err(Error::InvalidConfig(
                "AWQ is not supported for MoE generator models".into(),
            )),
        }
    }

    fn load_gguf(&mut self, path: &Path) -> Result<()> {
        match self {
            Self::Dense(model) => model.load_gguf(path),
            Self::Moe(_) => Err(Error::InvalidConfig(
                "GGUF is not supported for MoE generator models".into(),
            )),
        }
    }
}

#[derive(Clone)]
pub(crate) struct GeneratorEngine<B: Backend> {
    model: GeneratorVariant<B>,
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

        let mut model = match info.architecture {
            Architecture::GLM4MoE
            | Architecture::Qwen3MoE
            | Architecture::Mixtral
            | Architecture::DeepSeekV3 => {
                GeneratorVariant::Moe(MoEGeneratorModel::new(&device, config.clone())?)
            }
            _ => GeneratorVariant::Dense(GeneratorModel::new(&device, config.clone())?),
        };

        if let Some(model_path) = find_model_file(model_dir, &info.quantization) {
            match info.quantization {
                Quantization::GGUF => model.load_gguf(&model_path)?,
                Quantization::AWQ => model.load_awq(&model_path)?,
                Quantization::GPTQ => model.load_safetensors(&model_path)?, // TODO: implement load_gptq
                _ => model.load_safetensors(&model_path)?,
            }
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
