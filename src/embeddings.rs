//! Embeddings API — async-first design (per SPEC 04-API-DESIGN §3.2).

use crate::client::{Client, GllmError};

/// Builder for generating embeddings (per SPEC 04-API-DESIGN §3.2).
///
/// # Example
///
/// ```no_run
/// use gllm::Client;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_empty();
/// let embeddings = client.embed(vec![
///     "Hello world",
///     "Machine learning is fascinating"
/// ]).await?;
/// # Ok(())
/// # }
/// ```
pub struct EmbeddingsBuilder<'a> {
    client: &'a Client,
    inputs: Vec<String>,
}

impl<'a> EmbeddingsBuilder<'a> {
    pub(crate) fn new(client: &'a Client, inputs: Vec<String>) -> Self {
        Self { client, inputs }
    }

    /// Execute the embedding generation (async).
    pub async fn generate(self) -> Result<EmbeddingsResponse, GllmError> {
        self.client.execute_embeddings(self.inputs).await
    }
}

/// Response from embedding generation (per SPEC 04-API-DESIGN §3.2).
#[derive(Debug, Clone)]
pub struct EmbeddingsResponse {
    /// Generated embeddings for each input text.
    pub embeddings: Vec<Embedding>,
    /// Request ID (for tracking).
    pub request_id: Option<u64>,
}

/// A single embedding vector (per SPEC 04-API-DESIGN §3.2).
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The embedding vector (dimension depends on model).
    pub embedding: Vec<f32>,
}
