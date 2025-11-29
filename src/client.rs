use crate::embeddings::EmbeddingsBuilder;
use crate::engine::{EngineBackend, TokenizerAdapter, build_backend};
use crate::model::{ModelArtifacts, ModelManager};
use crate::rerank::RerankBuilder;
use crate::types::{ClientConfig, Result};
use log::warn;

/// Client for embeddings and reranking.
pub struct Client {
    engine: EngineBackend,
    tokenizer: TokenizerAdapter,
    #[allow(dead_code)]
    artifacts: ModelArtifacts,
}

impl Client {
    fn create(model: &str, config: ClientConfig) -> Result<Self> {
        let manager = ModelManager::new(config.clone());
        let artifacts = manager.prepare(model)?;
        let engine = build_backend(&artifacts.info, &artifacts.model_dir, &config.device)?;

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

// 同步接口
#[cfg(not(feature = "tokio"))]
impl Client {
    /// Create a new client with default configuration.
    pub fn new(model: &str) -> Result<Self> {
        Self::with_config(model, ClientConfig::default())
    }

    /// Create a new client with custom configuration.
    pub fn with_config(model: &str, config: ClientConfig) -> Result<Self> {
        Self::create(model, config)
    }
}

// 异步接口
#[cfg(feature = "tokio")]
impl Client {
    /// Create a new client with default configuration.
    pub async fn new(model: &str) -> Result<Self> {
        Self::with_config(model, ClientConfig::default()).await
    }

    /// Create a new client with custom configuration.
    pub async fn with_config(model: &str, config: ClientConfig) -> Result<Self> {
        tokio::task::block_in_place(|| Self::create(model, config))
    }
}
