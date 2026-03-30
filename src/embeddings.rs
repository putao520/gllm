//! Embeddings API — async-first design (per SPEC 04-API-DESIGN §3.2).

use std::ops::Index;

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
///
/// Implements `len()` and `Index<usize>` so that SPEC example code compiles:
/// ```ignore
/// assert_eq!(embeddings.len(), 2);
/// assert_eq!(embeddings[0].len(), 1024);
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingsResponse {
    /// Generated embeddings for each input text.
    pub embeddings: Vec<Embedding>,
    /// Request ID (for tracking).
    pub request_id: Option<u64>,
}

impl EmbeddingsResponse {
    /// Returns the number of embeddings (per SPEC 04-API-DESIGN §3.2).
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Returns true if there are no embeddings.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

impl Index<usize> for EmbeddingsResponse {
    type Output = Embedding;

    fn index(&self, index: usize) -> &Self::Output {
        &self.embeddings[index]
    }
}

/// A single embedding vector (per SPEC 04-API-DESIGN §3.2).
///
/// Implements `len()` so that SPEC example code `embeddings[0].len()` compiles.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The embedding vector (dimension depends on model).
    pub embedding: Vec<f32>,
}

impl Embedding {
    /// Returns the dimension of the embedding vector (per SPEC 04-API-DESIGN §3.2).
    pub fn len(&self) -> usize {
        self.embedding.len()
    }

    /// Returns true if the embedding vector is empty.
    pub fn is_empty(&self) -> bool {
        self.embedding.is_empty()
    }
}
