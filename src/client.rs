//! Client API skeleton.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use gllm_kernels::BackendError;
use thiserror::Error;

use crate::backend::{
    detect_backend, BackendContext, BackendContextError, FallbackEmbedder, FallbackGenerator,
    FallbackReranker,
};
use crate::embeddings::{Embedding, EmbeddingsBuilder, EmbeddingsResponse};
use crate::engine::executor::ExecutorError;
use crate::generation::{GenerationBuilder, GenerationResponse};
use crate::loader::{config as loader_config, Loader, LoaderConfig, LoaderError, TensorProvider, WeightFormat};
use crate::manifest::{ModelArchitecture, ModelKind, ModelManifest, EMPTY_FILE_MAP};
use crate::rerank::{RerankBuilder, RerankResponse, RerankResult};

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("unknown model alias: {0}")]
    UnknownModel(String),
    #[error("unsupported architecture: {0:?}")]
    UnsupportedArchitecture(ModelArchitecture),
    #[error("state lock poisoned")]
    ExecutorPoisoned,
    #[error("no model loaded")]
    NoModelLoaded,
    #[error("not implemented: {kind} (queued request {request_id})")]
    NotImplementedQueued { kind: &'static str, request_id: u64 },
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Config(#[from] loader_config::ConfigError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    #[error(transparent)]
    ModelConfig(#[from] crate::model_config::ModelConfigError),
}

pub struct ClientState {
    pub model_id: String,
    pub manifest: Arc<ModelManifest>,
    pub backend: BackendContext,
}

#[derive(Clone)]
pub struct Client {
    #[allow(clippy::arc_with_non_send_sync)]
    state: Arc<RwLock<Option<ClientState>>>,
}

pub struct AsyncClient {
    inner: Client,
}

impl From<BackendContextError> for ClientError {
    fn from(err: BackendContextError) -> Self {
        match err {
            BackendContextError::UnsupportedArchitecture(arch) => {
                ClientError::UnsupportedArchitecture(arch)
            }
            BackendContextError::Loader(err) => ClientError::Loader(err),
            BackendContextError::Executor(err) => ClientError::Executor(err),
        }
    }
}

impl Client {
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(model_id: &str, kind: ModelKind) -> Result<Self, ClientError> {
        let client = Self {
            state: Arc::new(RwLock::new(None)),
        };
        client.load_model(model_id, kind)?;
        Ok(client)
    }

    pub fn new_chat(model_id: &str) -> Result<Self, ClientError> {
        Self::new(model_id, ModelKind::Chat)
    }
    pub fn new_embedding(model_id: &str) -> Result<Self, ClientError> {
        Self::new(model_id, ModelKind::Embedding)
    }

    pub fn manifest(&self) -> Result<Arc<ModelManifest>, ClientError> {
        let state = self.read_state()?;
        state
            .as_ref()
            .map(|loaded| loaded.manifest.clone())
            .ok_or(ClientError::NoModelLoaded)
    }

    pub fn load_model(&self, model_id: &str, kind: ModelKind) -> Result<(), ClientError> {
        let model_id = Self::normalize_model_id(model_id)?;
        let mut guard = self.write_state()?;
        let state = Self::build_state(&model_id, kind)?;
        *guard = Some(state);
        Ok(())
    }

    pub fn unload_model(&self) -> Result<(), ClientError> {
        let mut guard = self.write_state()?;
        *guard = None;
        Ok(())
    }

    pub fn swap_model(&self, model_id: &str) -> Result<(), ClientError> {
        let model_id = Self::normalize_model_id(model_id)?;
        let mut guard = self.write_state()?;
        let kind = guard
            .as_ref()
            .map(|loaded| loaded.manifest.kind)
            .ok_or(ClientError::NoModelLoaded)?;

        *guard = None;
        let state = Self::build_state(&model_id, kind)?;
        *guard = Some(state);
        Ok(())
    }

    pub fn generate(&self, prompt: impl Into<String>) -> GenerationBuilder<'_> {
        GenerationBuilder::from_prompt(self, prompt)
    }

    pub fn embeddings<I, S>(&self, inputs: I) -> EmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let inputs = inputs.into_iter().map(Into::into).collect();
        EmbeddingsBuilder::new(self, inputs)
    }

    pub fn rerank<I, S>(&self, query: impl Into<String>, documents: I) -> RerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let documents = documents.into_iter().map(Into::into).collect();
        RerankBuilder::new(self, query, documents)
    }

    pub fn thinking_head_available(&self) -> Result<bool, ClientError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(ClientError::NoModelLoaded)?;
        let available = {
            let executor = loaded.backend.executor();
            executor.thinking_head_available()
        };
        Ok(available)
    }

    pub(crate) fn execute_generation(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<GenerationResponse, ClientError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(ClientError::NoModelLoaded)?;
        let mut generator = FallbackGenerator::new(&loaded.backend);
        let text = generator.generate(&prompt, max_tokens, temperature, top_k, top_p)?;
        Ok(GenerationResponse {
            text,
            request_id: None,
        })
    }

    pub(crate) fn execute_embeddings(
        &self,
        inputs: Vec<String>,
    ) -> Result<EmbeddingsResponse, ClientError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(ClientError::NoModelLoaded)?;
        let mut embedder = FallbackEmbedder::new(&loaded.backend);
        let embeddings = embedder.embed_batch(&inputs)?;
        let embeddings = embeddings
            .into_iter()
            .map(|embedding| Embedding { embedding })
            .collect();
        Ok(EmbeddingsResponse {
            embeddings,
            request_id: None,
        })
    }

    pub(crate) fn execute_rerank(
        &self,
        query: String,
        documents: Vec<String>,
        top_n: usize,
    ) -> Result<RerankResponse, ClientError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(ClientError::NoModelLoaded)?;
        let mut reranker = FallbackReranker::new(&loaded.backend);
        let scores = reranker.rerank_batch(&query, &documents)?;
        let mut results = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankResult { index, score })
            .collect::<Vec<_>>();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if top_n < results.len() {
            results.truncate(top_n);
        }

        Ok(RerankResponse {
            results,
            request_id: None,
        })
    }

    fn normalize_model_id(model_id: &str) -> Result<String, ClientError> {
        let trimmed = model_id.trim();
        if trimmed.is_empty() {
            return Err(ClientError::UnknownModel(model_id.to_string()));
        }
        Ok(trimmed.to_string())
    }

    fn build_state(model_id: &str, kind: ModelKind) -> Result<ClientState, ClientError> {
        let config = LoaderConfig::from_env();

        let (manifest, weight_paths, config_path, tokenizer_path) = match loader_config::download_config_files(model_id, &config, EMPTY_FILE_MAP)
        {
            Ok(config_files) => {
                let config_value = loader_config::load_config_value(&config_files.config_path)?;
                let manifest = loader_config::manifest_from_config(
                    model_id,
                    &config_value,
                    kind,
                )?;

                // Ensure weights are downloaded
                let loader = Loader::from_source_with_config(model_id.to_string(), config.clone())?;
                let config_path = loader.config_path().map(|p| p.to_path_buf());
                let tokenizer_path = loader.tokenizer_path().map(|p| p.to_path_buf());
                (Arc::new(manifest), loader.weight_paths().to_vec(), config_path, tokenizer_path)
            }
            Err(loader_config::ConfigError::MissingConfig { .. }) => {
                let mut loader = Loader::from_source_with_config(model_id.to_string(), config.clone())?;

                let manifest = match loader.weight_format() {
                    WeightFormat::Gguf => {
                        let arch_str = loader.gguf_architecture()?;
                        if let Some(arch) = loader_config::map_architecture_token(arch_str) {
                             ModelManifest {
                                model_id: Cow::Owned(model_id.to_string()),
                                file_map: EMPTY_FILE_MAP,
                                arch,
                                tensor_rules: loader_config::tensor_rules_for_arch(arch),
                                kind,
                                rope_base_override: None,
                                max_context_override: None,
                                moe_config: None,
                                tensor_map: HashMap::new(),
                            }
                        } else {
                             return Err(ClientError::UnknownModel(format!("Unsupported GGUF architecture: {}", arch_str)));
                        }
                    }
                    WeightFormat::SafeTensors | WeightFormat::Onnx => {
                        // Ω1: Tensor-driven derivation (REQ-LOADER-022, REQ-LOADER-023)
                        loader = loader.load()?;

                        // 1. Validate Topology via ModelConfig
                        let dummy_manifest = ModelManifest {
                            model_id: Cow::Owned(model_id.to_string()),
                            file_map: EMPTY_FILE_MAP,
                            arch: ModelArchitecture::Llama4,
                            tensor_rules: crate::manifest::TensorNamingRule::Llama4,
                            kind,
                            rope_base_override: None,
                            max_context_override: None,
                            moe_config: None,
                            tensor_map: HashMap::new(),
                        };

                        let _derived_config = crate::model_config::ModelConfig::from_loader(&dummy_manifest, &mut loader)?;

                        // 2. Guess Architecture from Tensor Names
                        let arch = detect_architecture(&loader).unwrap_or(ModelArchitecture::Llama4);

                        ModelManifest {
                            model_id: Cow::Owned(model_id.to_string()),
                            file_map: EMPTY_FILE_MAP,
                            arch,
                            tensor_rules: loader_config::tensor_rules_for_arch(arch),
                            kind,
                            rope_base_override: None,
                            max_context_override: None,
                            moe_config: None,
                            tensor_map: HashMap::new(),
                        }
                    }
                    _ => {
                        return Err(loader_config::ConfigError::MissingConfig {
                            model_id: model_id.to_string(),
                        }
                        .into());
                    }
                };
                let config_path = loader.config_path().map(|p| p.to_path_buf());
                let tokenizer_path = loader.tokenizer_path().map(|p| p.to_path_buf());
                (Arc::new(manifest), loader.weight_paths().to_vec(), config_path, tokenizer_path)
            }
            Err(err) => return Err(err.into()),
        };

        let detected_backend = detect_backend()?;
        let backend =
            BackendContext::new(model_id.to_string(), manifest.clone(), detected_backend, weight_paths, config_path, tokenizer_path)?;
        Ok(ClientState {
            model_id: model_id.to_string(),
            manifest,
            backend,
        })
    }

    fn read_state(&self) -> Result<RwLockReadGuard<'_, Option<ClientState>>, ClientError> {
        self.state.read().map_err(|_| ClientError::ExecutorPoisoned)
    }

    fn write_state(&self) -> Result<RwLockWriteGuard<'_, Option<ClientState>>, ClientError> {
        self.state
            .write()
            .map_err(|_| ClientError::ExecutorPoisoned)
    }
}

fn detect_architecture(loader: &Loader) -> Option<ModelArchitecture> {
    let check_name = |name: &str| -> Option<ModelArchitecture> {
        let lower = name.to_lowercase();
        if lower.contains("bert") || lower.contains("roberta") {
            return Some(ModelArchitecture::XlmR);
        }
        if lower.contains("gpt2") {
            return Some(ModelArchitecture::GPT2Next);
        }
        if lower.contains("mistral") {
            return Some(ModelArchitecture::Mistral3);
        }
        None
    };

    match loader.weight_format() {
        WeightFormat::SafeTensors => {
            if let Some(st) = loader.safetensors_ref() {
                for meta in st.iter_tensors() {
                    if let Some(arch) = check_name(&meta.name) {
                        return Some(arch);
                    }
                }
            }
        }
        WeightFormat::Onnx => {
             if let Some(onnx) = loader.onnx_ref() {
                 for meta in onnx.iter_tensors() {
                    if let Some(arch) = check_name(&meta.name) {
                        return Some(arch);
                    }
                 }
             }
        }
        _ => {}
    }
    // Default to Llama4 for most modern LLMs (Qwen, Llama, etc share similar structure)
    Some(ModelArchitecture::Llama4)
}

impl AsyncClient {
    pub fn new(model_id: &str, kind: ModelKind) -> Result<Self, ClientError> {
        Ok(Self {
            inner: Client::new(model_id, kind)?,
        })
    }

    pub fn inner(&self) -> &Client {
        &self.inner
    }
    pub fn generate(&self, prompt: impl Into<String>) -> GenerationBuilder<'_> {
        self.inner.generate(prompt)
    }

    pub fn embeddings<I, S>(&self, inputs: I) -> EmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.embeddings(inputs)
    }

    pub fn rerank<I, S>(&self, query: impl Into<String>, documents: I) -> RerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.rerank(query, documents)
    }
}
