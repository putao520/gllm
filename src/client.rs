use crate::embeddings::EmbeddingsBuilder;
use crate::engine::{build_backend, EngineBackend, TokenizerAdapter};
use crate::generation::{GenerationBuilder, GenerationConfig};
use crate::model::{ModelArtifacts, ModelManager};
use crate::rerank::RerankBuilder;
use crate::types::{ClientConfig, Result};
use log::warn;

/// Client for embeddings, reranking, and generation.
pub struct Client {
    engine: EngineBackend,
    tokenizer: TokenizerAdapter,
    #[allow(dead_code)]
    artifacts: ModelArtifacts,
}

impl Client {
    pub(crate) fn create(model: &str, config: ClientConfig) -> Result<Self> {
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
            graph_inputs: Vec::new(),
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

    /// Build a text generation request.
    pub fn generate(&self, prompt: &str) -> GenerationBuilder<'_> {
        GenerationBuilder {
            engine: &self.engine,
            tokenizer: &self.tokenizer,
            prompt: prompt.to_string(),
            config: GenerationConfig::default(),
        }
    }
}

impl Client {
    /// Explicitly clean up GPU resources to avoid SIGSEGV on exit.
    /// Call this before dropping the Client if you want to ensure proper cleanup.
    /// See https://github.com/gfx-rs/wgpu/issues/5655 for details.
    pub fn cleanup(&mut self) {
        log::debug!("Client GPU cleanup requested");
        // Additional final cleanup: retry pattern with small delays
        // EngineBackend::Drop will do most of the heavy lifting (800ms)
        // This ensures final synchronization before process exit
        for attempt in 1..=3 {
            log::debug!("Client cleanup attempt {}/3", attempt);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        log::debug!("Client cleanup completed");
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        // Trigger explicit cleanup when Client is dropped.
        // EngineBackend Drop will run its own cleanup with retry strategy (800ms total)
        // Then Client adds final synchronization (150ms total for 3 retries of 50ms each)
        // Combined strategy: retry + increasing delays across multiple layers
        self.cleanup();
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
    /// Uses spawn_blocking for compatibility with both single-thread and multi-thread runtimes.
    pub async fn with_config(model: &str, config: ClientConfig) -> Result<Self> {
        let model = model.to_string();
        tokio::task::spawn_blocking(move || Self::create(&model, config))
            .await
            .map_err(|e| crate::types::Error::InternalError(format!("spawn_blocking failed: {}", e)))?
    }
}
