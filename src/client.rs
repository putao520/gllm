#[cfg(feature = "async")]
use crate::embeddings::AsyncEmbeddingsBuilder;
use crate::embeddings::EmbeddingsBuilder;
use crate::engine::{EngineBackend, TokenizerAdapter, build_backend};
use crate::model::{ModelArtifacts, ModelManager};
#[cfg(feature = "async")]
use crate::rerank::AsyncRerankBuilder;
use crate::rerank::RerankBuilder;
use crate::types::{ClientConfig, Result};
use log::warn;

/// Synchronous client.
pub struct Client {
    engine: EngineBackend,
    tokenizer: TokenizerAdapter,
    #[allow(dead_code)]
    artifacts: ModelArtifacts,
}

impl Client {
    /// Create a new client with default configuration.
    pub fn new(model: &str) -> Result<Self> {
        Self::with_config(model, ClientConfig::default())
    }

    /// Create a new client with custom configuration.
    pub fn with_config(model: &str, config: ClientConfig) -> Result<Self> {
        let manager = ModelManager::new(config.clone());
        let artifacts = manager.prepare(model)?;
        let engine = build_backend(&artifacts.info, &artifacts.model_dir, &config.device)?;

        // Validate model files if present; continue with initialized parameters if validation fails.
        if let Err(err) = manager.validate_model_files(&artifacts.model_dir) {
            warn!("Skipping model file validation: {err}");
        }

        Ok(Self {
            engine,
            tokenizer: artifacts.tokenizer.clone(),
            artifacts,
        })
    }

    /// Build an embeddings request.
    pub fn embeddings<I, S>(&self, input: I) -> EmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let inputs = input.into_iter().map(|s| s.as_ref().to_string()).collect();
        EmbeddingsBuilder {
            engine: &self.engine,
            tokenizer: &self.tokenizer,
            inputs,
        }
    }

    /// Build a rerank request.
    pub fn rerank<I, S>(&self, query: &str, documents: I) -> RerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let docs = documents
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        RerankBuilder {
            engine: &self.engine,
            tokenizer: &self.tokenizer,
            query: query.to_string(),
            documents: docs,
            top_n: None,
            return_documents: false,
        }
    }
}

/// Asynchronous client (feature = "async").
#[cfg(feature = "async")]
pub struct AsyncClient {
    inner: Client,
}

#[cfg(feature = "async")]
impl AsyncClient {
    /// Create a new client asynchronously.
    pub async fn new(model: &str) -> Result<Self> {
        Client::new(model).map(|inner| Self { inner })
    }

    /// Create a new client with custom configuration asynchronously.
    pub async fn with_config(model: &str, config: ClientConfig) -> Result<Self> {
        Client::with_config(model, config).map(|inner| Self { inner })
    }

    /// Build an async embeddings request.
    pub fn embeddings<I, S>(&self, input: I) -> AsyncEmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        AsyncEmbeddingsBuilder {
            inner: self.inner.embeddings(input),
        }
    }

    /// Build an async rerank request.
    pub fn rerank<I, S>(&self, query: &str, documents: I) -> AsyncRerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        AsyncRerankBuilder {
            inner: self.inner.rerank(query, documents),
        }
    }
}
