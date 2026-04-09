//! Embeddings API — sync-first design (per SPEC 04-API-DESIGN §3.2).
//!
//! Supports optional pipeline fusion: embed → rerank → generate (RAG).

use std::ops::Index;

use crate::client::{Client, GllmError};

/// Builder for generating embeddings (per SPEC 04-API-DESIGN §3.2).
///
/// Supports optional pipeline extensions via `.rerank_query()` and `.top_n()`:
///
/// - **embed only**: `client.embed_builder(inputs).generate()`
/// - **embed + rerank**: `client.embed_builder(inputs).rerank_query("query").generate()`
/// - **embed + rerank + generate (RAG)**: `client.embed_builder(inputs).rerank_query("query").generate_answer("system prompt")`
///
/// # Example
///
/// ```no_run
/// use gllm::Client;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_empty();
/// let embeddings = client.embed(vec![
///     "Hello world",
///     "Machine learning is fascinating"
/// ])?;
/// # Ok(())
/// # }
/// ```
pub struct EmbeddingsBuilder<'a> {
    client: &'a Client,
    inputs: Vec<String>,
    rerank_query: Option<String>,
    top_n: Option<usize>,
}

impl<'a> EmbeddingsBuilder<'a> {
    pub(crate) fn new(client: &'a Client, inputs: Vec<String>) -> Self {
        Self {
            client,
            inputs,
            rerank_query: None,
            top_n: None,
        }
    }

    /// Set a rerank query to enable the embed+rerank pipeline.
    ///
    /// When set, `.generate()` will first embed all inputs, then rerank them
    /// against this query using the pipeline reranker model. The resulting
    /// `EmbeddingsResponse` will include `rerank_scores` and embeddings sorted
    /// by descending relevance.
    ///
    /// Requires that the `Client` was built with `.reranker()`.
    pub fn rerank_query(mut self, query: impl Into<String>) -> Self {
        self.rerank_query = Some(query.into());
        self
    }

    /// Set the maximum number of results to return after reranking.
    ///
    /// Only effective when `.rerank_query()` is also set.
    pub fn top_n(mut self, n: usize) -> Self {
        self.top_n = Some(n);
        self
    }

    /// Execute embedding generation (sync).
    ///
    /// If `.rerank_query()` was set, executes the embed+rerank pipeline
    /// and returns results sorted by relevance with `rerank_scores` populated.
    pub fn generate(self) -> Result<EmbeddingsResponse, GllmError> {
        if let Some(query) = self.rerank_query {
            self.client
                .execute_embed_rerank_pipeline(self.inputs, query, self.top_n)
        } else {
            self.client.execute_embeddings(self.inputs)
        }
    }

    /// Execute the full RAG pipeline: embed → rerank → generate answer (sync).
    ///
    /// Requires both `.rerank_query()` and that the `Client` was built with
    /// `.reranker()` and `.generator()`.
    pub fn generate_answer(
        self,
        system_prompt: impl Into<String>,
    ) -> Result<RagResponse, GllmError> {
        let query = self.rerank_query.ok_or(GllmError::RuntimeError(
            "generate_answer requires rerank_query to be set via .rerank_query()".into(),
        ))?;
        self.client.execute_rag_pipeline(
            self.inputs,
            query,
            self.top_n.unwrap_or(3),
            system_prompt.into(),
        )
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
    /// Rerank scores (present when embed+rerank pipeline was used).
    /// Sorted in the same order as `embeddings` (descending relevance).
    pub rerank_scores: Option<Vec<f32>>,
    /// Request ID (for tracking).
    pub request_id: Option<u64>,
}

/// Response from the full RAG pipeline (embed → rerank → generate).
#[derive(Debug, Clone)]
pub struct RagResponse {
    /// Generated answer text from the LLM.
    pub text: String,
    /// Indices of the selected source documents (in the original input order).
    pub sources: Vec<usize>,
    /// Rerank scores for the selected documents.
    pub rerank_scores: Vec<f32>,
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
