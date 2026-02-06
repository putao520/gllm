//! Client API skeleton.

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use gllm_kernels::BackendError;
use thiserror::Error;

use crate::adapter::Message;
use crate::backend::{
    detect_backend, BackendContext, BackendContextError, FallbackEmbedder, FallbackGenerator,
};
use crate::embeddings::{Embedding, EmbeddingsBuilder, EmbeddingsResponse};
use crate::engine::executor::ExecutorError;
use crate::generation::{GenerationBuilder, GenerationResponse};
use crate::loader::{config as loader_config, LoaderConfig, LoaderError};
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
}

pub struct ClientState {
    pub model_id: String,
    pub manifest: Arc<ModelManifest>,
    pub backend: BackendContext,
}

#[derive(Clone)]
pub struct Client {
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

    pub fn manifest(&self) -> Arc<ModelManifest> {
        let state = self.state.read().unwrap_or_else(|err| err.into_inner());
        state
            .as_ref()
            .map(|loaded| loaded.manifest.clone())
            .expect("model loaded")
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
    pub fn generate_chat(&self, messages: Vec<Message>) -> GenerationBuilder<'_> {
        GenerationBuilder::from_messages(self, messages)
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

    pub(crate) fn render_chat_prompt(&self, messages: &[Message]) -> Result<String, ClientError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(ClientError::NoModelLoaded)?;
        let prompt = {
            let executor = loaded.backend.executor();
            executor.apply_chat_template(messages)
        };
        Ok(prompt)
    }

    pub(crate) fn execute_generation(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GenerationResponse, ClientError> {
        let state = self.read_state()?;
        let loaded = state.as_ref().ok_or(ClientError::NoModelLoaded)?;
        let mut generator = FallbackGenerator::new(&loaded.backend);
        let text = generator.generate(&prompt, max_tokens, temperature)?;
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
        let mut embedder = FallbackEmbedder::new(&loaded.backend);
        let scores = embedder.rerank_batch(&query, &documents)?;
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
        let config_files = loader_config::download_config_files(model_id, &config, EMPTY_FILE_MAP)?;
        let config_value = loader_config::load_config_value(&config_files.config_path)?;
        let manifest = Arc::new(loader_config::manifest_from_config(
            model_id,
            &config_value,
            kind,
        )?);
        let detected_backend = detect_backend()?;
        let backend =
            BackendContext::new(model_id.to_string(), manifest.clone(), detected_backend)?;
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
    pub fn generate_chat(&self, messages: Vec<Message>) -> GenerationBuilder<'_> {
        self.inner.generate_chat(messages)
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
