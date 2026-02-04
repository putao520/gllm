//! Client API skeleton.

use std::sync::{Arc, Mutex};

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
use crate::manifest::{ModelArchitecture, ModelManifest, EMPTY_FILE_MAP};
use crate::registry;
use crate::rerank::{RerankBuilder, RerankResponse, RerankResult};

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("unknown model alias: {0}")]
    UnknownModel(String),
    #[error("unsupported architecture: {0:?}")]
    UnsupportedArchitecture(ModelArchitecture),
    #[error("executor lock poisoned")]
    ExecutorPoisoned,
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

pub struct Client {
    model_id: String,
    manifest: Arc<ModelManifest>,
    backend: Mutex<Option<BackendContext>>,
}

pub struct AsyncClient {
    inner: Client,
}

struct BackendGuard<'a> {
    guard: std::sync::MutexGuard<'a, Option<BackendContext>>,
}

impl<'a> std::ops::Deref for BackendGuard<'a> {
    type Target = BackendContext;

    fn deref(&self) -> &Self::Target {
        self.guard
            .as_ref()
            .expect("backend context initialized")
    }
}

impl<'a> std::ops::DerefMut for BackendGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard
            .as_mut()
            .expect("backend context initialized")
    }
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
    pub fn new(model_or_alias: &str) -> Result<Self, ClientError> {
        let model_id = model_or_alias.trim();
        if model_id.is_empty() {
            return Err(ClientError::UnknownModel(model_or_alias.to_string()));
        }

        let overrides = registry::lookup(model_id);
        let file_map = overrides.map(|m| m.file_map).unwrap_or(EMPTY_FILE_MAP);
        let loader_config = LoaderConfig::from_env();
        let config_files =
            loader_config::download_config_files(model_id, &loader_config, file_map)?;
        let config_value = loader_config::load_config_value(&config_files.config_path)?;
        let manifest = loader_config::manifest_from_config(model_id, &config_value, overrides)?;

        Ok(Self {
            model_id: model_id.to_string(),
            manifest: Arc::new(manifest),
            backend: Mutex::new(None),
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        self.manifest.as_ref()
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
        let backend = self.lock_backend()?;
        Ok(backend.executor().thinking_head_available())
    }

    pub(crate) fn render_chat_prompt(&self, messages: &[Message]) -> Result<String, ClientError> {
        let backend = self.lock_backend()?;
        Ok(backend.executor().apply_chat_template(messages))
    }

    pub(crate) fn execute_generation(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GenerationResponse, ClientError> {
        let mut backend = self.lock_backend()?;
        let mut generator = FallbackGenerator::new(&mut backend);
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
        let mut backend = self.lock_backend()?;
        let mut embedder = FallbackEmbedder::new(&mut backend);
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
        let mut backend = self.lock_backend()?;
        let mut embedder = FallbackEmbedder::new(&mut backend);
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

    fn lock_backend(&self) -> Result<BackendGuard<'_>, ClientError> {
        let mut guard = self
            .backend
            .lock()
            .map_err(|_| ClientError::ExecutorPoisoned)?;
        if guard.is_none() {
            let backend = detect_backend()?;
            let context =
                BackendContext::new(self.model_id.clone(), self.manifest.clone(), backend)?;
            *guard = Some(context);
        }
        Ok(BackendGuard { guard })
    }
}

impl AsyncClient {
    pub fn new(model_or_alias: &str) -> Result<Self, ClientError> {
        Ok(Self {
            inner: Client::new(model_or_alias)?,
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
