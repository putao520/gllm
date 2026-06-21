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
    /// Whether to L2-normalize the embedding vectors (per REQ-API-3).
    normalize: bool,
    /// Truncate embedding vectors to this dimension (per REQ-API-3).
    /// `None` means use the model's full hidden size.
    dimension: Option<usize>,
}

impl<'a> EmbeddingsBuilder<'a> {
    pub(crate) fn new(client: &'a Client, inputs: Vec<String>) -> Self {
        Self {
            client,
            inputs,
            rerank_query: None,
            top_n: None,
            normalize: false,
            dimension: None,
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

    /// Enable L2 normalization of embedding vectors (per REQ-API-3).
    ///
    /// When `true`, each embedding vector is divided by its L2 norm,
    /// producing a unit-length vector. The original L2 norm is preserved
    /// in `Embedding::norm`.
    ///
    /// Normalization is essential for cosine similarity search: after
    /// normalization, dot product equals cosine similarity.
    // @trace REQ-API-3 [entity:EmbeddingsBuilder] [api:POST /client/embed]
    pub fn normalize(mut self, yes: bool) -> Self {
        self.normalize = yes;
        self
    }

    /// Truncate embedding vectors to `dim` dimensions (per REQ-API-3).
    ///
    /// Matryoshka Representation Learning (MRL) models support truncation
    /// without re-encoding: the first `dim` dimensions remain semantically
    /// meaningful. Non-MRL models will lose information; the caller must
    /// ensure the model supports dimension truncation.
    ///
    /// `dim` must be > 0 and <= model hidden size. Zero or `None` means
    /// use the full model hidden size.
    // @trace REQ-API-3 [entity:EmbeddingsBuilder] [api:POST /client/embed]
    pub fn dimension(mut self, dim: usize) -> Self {
        self.dimension = if dim > 0 { Some(dim) } else { None };
        self
    }

    /// Execute embedding generation (sync).
    ///
    /// If `.rerank_query()` was set, executes the embed+rerank pipeline
    /// and returns results sorted by relevance with `rerank_scores` populated.
    pub fn generate(self) -> Result<EmbeddingsResponse, GllmError> {
        if let Some(query) = self.rerank_query {
            self.client
                .execute_embed_rerank_pipeline(
                    self.inputs,
                    query,
                    self.top_n,
                    self.normalize,
                    self.dimension,
                )
        } else {
            self.client
                .execute_embeddings(self.inputs, self.normalize, self.dimension)
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
            self.normalize,
            self.dimension,
        )
    }
}

/// Configuration for embedding generation (per REQ-API-3).
///
/// Passed to `Client::embed_with()` to control normalization and dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedConfig {
    /// Whether to L2-normalize the embedding vectors.
    pub normalize: bool,
    /// Truncate embedding vectors to this dimension.
    /// `None` means use the model's full hidden size.
    pub dimension: Option<usize>,
}

impl EmbedConfig {
    /// Create a default `EmbedConfig` (no normalization, full dimension).
    pub fn new() -> Self {
        Self {
            normalize: false,
            dimension: None,
        }
    }

    /// Enable L2 normalization of embedding vectors.
    // @trace REQ-API-3 [entity:EmbedConfig] [api:POST /client/embed]
    pub fn normalize(mut self, yes: bool) -> Self {
        self.normalize = yes;
        self
    }

    /// Truncate embedding vectors to `dim` dimensions.
    // @trace REQ-API-3 [entity:EmbedConfig] [api:POST /client/embed]
    pub fn dimension(mut self, dim: usize) -> Self {
        self.dimension = if dim > 0 { Some(dim) } else { None };
        self
    }
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Response from embedding generation (per SPEC 04-API-DESIGN §3.2).
///
/// Implements `len()` and `Index<usize>` so that SPEC example code compiles:
/// ```ignore
/// assert_eq!(embeddings.len(), 2);
/// assert_eq!(embeddings[0].len(), 1024);
/// ```
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    /// The embedding vector (dimension depends on model, or truncated to `dimension` if configured).
    pub embedding: Vec<f32>,
    /// L2 norm of the embedding vector before optional normalization.
    /// Present when `normalize(true)` was requested via `EmbeddingsBuilder`.
    pub norm: Option<f32>,
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

    /// Returns the L2 norm of the original (pre-normalization) embedding vector.
    // @trace REQ-API-3 [entity:Embedding] [api:POST /client/embed]
    pub fn norm(&self) -> Option<f32> {
        self.norm
    }

    /// Returns true if this embedding was L2-normalized.
    // @trace REQ-API-3 [entity:Embedding] [api:POST /client/embed]
    pub fn is_normalized(&self) -> bool {
        self.norm.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_response_len() {
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![1.0; 128] , norm: None },
                Embedding { embedding: vec![0.5; 128] , norm: None },
            ],
            rerank_scores: None,
            request_id: Some(42),
        };
        assert_eq!(resp.len(), 2);
        assert!(!resp.is_empty());
    }

    #[test]
    fn embeddings_response_empty() {
        let resp = EmbeddingsResponse {
            embeddings: vec![],
            rerank_scores: None,
            request_id: None,
        };
        assert_eq!(resp.len(), 0);
        assert!(resp.is_empty());
    }

    #[test]
    fn embeddings_response_index() {
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![1.0, 2.0] , norm: None },
                Embedding { embedding: vec![3.0, 4.0] , norm: None },
            ],
            rerank_scores: Some(vec![0.9, 0.7]),
            request_id: Some(1),
        };
        assert_eq!(resp[0].len(), 2);
        assert!((resp[1].embedding[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn embeddings_response_rerank_scores() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![0.0; 64] , norm: None }],
            rerank_scores: Some(vec![0.95]),
            request_id: Some(99),
        };
        assert!(resp.rerank_scores.is_some());
        assert_eq!(resp.rerank_scores.unwrap().len(), 1);
        assert_eq!(resp.request_id, Some(99));
    }

    #[test]
    fn embedding_len_and_empty() {
        let e = Embedding { embedding: vec![1.0; 1024] , norm: None };
        assert_eq!(e.len(), 1024);
        assert!(!e.is_empty());

        let empty = Embedding { embedding: vec![] , norm: None };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn rag_response_fields() {
        let resp = RagResponse {
            text: "Paris is the capital of France.".into(),
            sources: vec![0, 3],
            rerank_scores: vec![0.95, 0.82],
            request_id: Some(7),
        };
        assert_eq!(resp.sources, vec![0, 3]);
        assert_eq!(resp.rerank_scores.len(), 2);
        assert!((resp.rerank_scores[0] - 0.95).abs() < 1e-6);
    }

    // ── Clone / Debug ──

    #[test]
    fn embedding_clone() {
        let e = Embedding { embedding: vec![1.0, 2.0, 3.0] , norm: None };
        let cloned = e.clone();
        assert_eq!(e.embedding, cloned.embedding);
    }

    #[test]
    fn embedding_debug() {
        let e = Embedding { embedding: vec![0.5] , norm: None };
        let debug = format!("{e:?}");
        assert!(debug.contains("embedding"));
    }

    #[test]
    fn embeddings_response_clone() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: Some(vec![0.9]),
            request_id: Some(1),
        };
        let cloned = resp.clone();
        assert_eq!(cloned.len(), 1);
        assert!(cloned.rerank_scores.is_some());
    }

    #[test]
    fn embeddings_response_debug() {
        let resp = EmbeddingsResponse {
            embeddings: vec![],
            rerank_scores: None,
            request_id: None,
        };
        let debug = format!("{resp:?}");
        assert!(debug.contains("embeddings"));
        assert!(debug.contains("rerank_scores"));
        assert!(debug.contains("request_id"));
    }

    #[test]
    fn rag_response_clone() {
        let resp = RagResponse {
            text: "answer".into(),
            sources: vec![0],
            rerank_scores: vec![0.5],
            request_id: None,
        };
        let cloned = resp.clone();
        assert_eq!(cloned.text, "answer");
        assert_eq!(cloned.sources, vec![0]);
    }

    #[test]
    fn rag_response_debug() {
        let resp = RagResponse {
            text: "test".into(),
            sources: vec![],
            rerank_scores: vec![],
            request_id: Some(42),
        };
        let debug = format!("{resp:?}");
        assert!(debug.contains("text"));
        assert!(debug.contains("sources"));
        assert!(debug.contains("request_id"));
    }

    // ── EmbeddingsResponse with rerank_scores ──

    #[test]
    fn embeddings_response_no_rerank_scores() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0; 64] , norm: None }],
            rerank_scores: None,
            request_id: Some(10),
        };
        assert!(resp.rerank_scores.is_none());
        assert_eq!(resp.request_id, Some(10));
    }

    #[test]
    fn embeddings_response_multiple_with_scores() {
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![0.1; 128] , norm: None },
                Embedding { embedding: vec![0.2; 128] , norm: None },
                Embedding { embedding: vec![0.3; 128] , norm: None },
            ],
            rerank_scores: Some(vec![0.95, 0.85, 0.75]),
            request_id: Some(5),
        };
        assert_eq!(resp.len(), 3);
        assert_eq!(resp[0].len(), 128);
        let scores = resp.rerank_scores.as_ref().unwrap();
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 0.95).abs() < 1e-6);
    }

    // ── Embedding access patterns ──

    #[test]
    fn embedding_value_access() {
        let e = Embedding { embedding: vec![1.0, 2.0, 3.0, 4.0] , norm: None };
        assert!((e.embedding[0] - 1.0).abs() < 1e-6);
        assert!((e.embedding[3] - 4.0).abs() < 1e-6);
    }

    // ── RagResponse edge cases ──

    #[test]
    fn rag_response_no_sources() {
        let resp = RagResponse {
            text: "No relevant documents found.".into(),
            sources: vec![],
            rerank_scores: vec![],
            request_id: None,
        };
        assert!(resp.sources.is_empty());
        assert!(resp.rerank_scores.is_empty());
        assert!(resp.request_id.is_none());
    }

    // ── Edge cases: request_id boundary values ──

    #[test]
    fn embeddings_response_request_id_max() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![0.0; 4] , norm: None }],
            rerank_scores: None,
            request_id: Some(u64::MAX),
        };
        assert_eq!(resp.request_id, Some(u64::MAX));
    }

    #[test]
    fn embeddings_response_request_id_zero() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![0.0; 4] , norm: None }],
            rerank_scores: None,
            request_id: Some(0),
        };
        assert_eq!(resp.request_id, Some(0));
    }

    #[test]
    fn rag_response_request_id_max() {
        let resp = RagResponse {
            text: "answer".into(),
            sources: vec![1],
            rerank_scores: vec![0.5],
            request_id: Some(u64::MAX),
        };
        assert_eq!(resp.request_id, Some(u64::MAX));
    }

    // ── Edge cases: special f32 values ──

    #[test]
    fn embedding_with_nan_values() {
        let e = Embedding {
            embedding: vec![f32::NAN, 1.0, f32::NAN],
            norm: None,
        };
        assert_eq!(e.len(), 3);
        assert!(!e.is_empty());
        assert!(e.embedding[0].is_nan());
    }

    #[test]
    fn embedding_with_infinity_values() {
        let e = Embedding {
            embedding: vec![f32::INFINITY, f32::NEG_INFINITY, 0.0],
            norm: None,
        };
        assert!(e.embedding[0].is_infinite() && e.embedding[0].is_sign_positive());
        assert!(e.embedding[1].is_infinite() && e.embedding[1].is_sign_negative());
        assert_eq!(e.embedding[2], 0.0);
    }

    #[test]
    fn embeddings_response_rerank_scores_with_special_floats() {
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![0.0; 4] , norm: None },
                Embedding { embedding: vec![0.0; 4] , norm: None },
            ],
            rerank_scores: Some(vec![f32::NAN, f32::INFINITY]),
            request_id: None,
        };
        let scores = resp.rerank_scores.as_ref().unwrap();
        assert!(scores[0].is_nan());
        assert!(scores[1].is_infinite());
    }

    #[test]
    fn rag_response_rerank_scores_negative() {
        let resp = RagResponse {
            text: "result".into(),
            sources: vec![0, 1],
            rerank_scores: vec![-0.5, -100.0],
            request_id: None,
        };
        assert!((resp.rerank_scores[0] - (-0.5)).abs() < 1e-6);
        assert!((resp.rerank_scores[1] - (-100.0)).abs() < 1e-6);
    }

    // ── Embedding single-element vector ──

    #[test]
    fn embedding_single_element() {
        let e = Embedding { embedding: vec![42.0] , norm: None };
        assert_eq!(e.len(), 1);
        assert!(!e.is_empty());
        assert!((e.embedding[0] - 42.0).abs() < 1e-6);
    }

    // ── EmbeddingsResponse with rerank_scores but no request_id ──

    #[test]
    fn embeddings_response_scores_without_request_id() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0; 32] , norm: None }],
            rerank_scores: Some(vec![0.99]),
            request_id: None,
        };
        assert!(resp.rerank_scores.is_some());
        assert!(resp.request_id.is_none());
        assert_eq!(resp.len(), 1);
    }

    // ── EmbeddingsResponse is_empty consistent with len ──

    #[test]
    fn embeddings_response_is_empty_consistent_with_len() {
        let empty = EmbeddingsResponse {
            embeddings: vec![],
            rerank_scores: None,
            request_id: Some(1),
        };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let non_empty = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![0.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        assert_eq!(non_empty.len(), 1);
        assert!(!non_empty.is_empty());
    }

    // ── RagResponse text field access ──

    #[test]
    fn rag_response_text_empty_string() {
        let resp = RagResponse {
            text: String::new(),
            sources: vec![0],
            rerank_scores: vec![0.0],
            request_id: None,
        };
        assert!(resp.text.is_empty());
    }

    #[test]
    fn rag_response_text_multibyte() {
        let resp = RagResponse {
            text: "人工智能推理引擎".into(),
            sources: vec![2],
            rerank_scores: vec![0.88],
            request_id: Some(100),
        };
        assert_eq!(resp.text, "人工智能推理引擎");
        assert_eq!(resp.sources, vec![2]);
    }

    // ── RagResponse sources with large index ──

    #[test]
    fn rag_response_sources_with_max_usize() {
        let resp = RagResponse {
            text: "answer".into(),
            sources: vec![usize::MAX],
            rerank_scores: vec![0.1],
            request_id: None,
        };
        assert_eq!(resp.sources[0], usize::MAX);
    }

    // ── RagResponse sources with duplicates (legal) ──

    #[test]
    fn rag_response_sources_with_duplicates() {
        let resp = RagResponse {
            text: "result".into(),
            sources: vec![0, 0, 2, 2, 2],
            rerank_scores: vec![0.9, 0.9, 0.5, 0.5, 0.5],
            request_id: None,
        };
        assert_eq!(resp.sources.len(), 5);
        assert_eq!(resp.sources[0], resp.sources[1]);
        assert_eq!(resp.rerank_scores.len(), 5);
    }

    // ── EmbeddingsResponse Index boundary ──

    #[test]
    fn embeddings_response_index_last_element() {
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![1.0] , norm: None },
                Embedding { embedding: vec![2.0] , norm: None },
                Embedding { embedding: vec![3.0] , norm: None },
            ],
            rerank_scores: None,
            request_id: None,
        };
        assert!((resp[2].embedding[0] - 3.0).abs() < 1e-6);
        assert_eq!(resp.len(), 3);
    }

    // ══════════════════════════════════════════════════════════════════════
    // NEW: Embedding — PartialEq, Clone independence, subnormal f32
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn embedding_partial_eq_equal() {
        let a = Embedding { embedding: vec![1.0, 2.0, 3.0] , norm: None };
        let b = Embedding { embedding: vec![1.0, 2.0, 3.0] , norm: None };
        assert_eq!(a, b);
    }

    #[test]
    fn embedding_partial_eq_not_equal_values() {
        let a = Embedding { embedding: vec![1.0, 2.0] , norm: None };
        let b = Embedding { embedding: vec![1.0, 3.0] , norm: None };
        assert_ne!(a, b);
    }

    #[test]
    fn embedding_partial_eq_not_equal_lengths() {
        let a = Embedding { embedding: vec![1.0] , norm: None };
        let b = Embedding { embedding: vec![1.0, 2.0] , norm: None };
        assert_ne!(a, b);
    }

    #[test]
    fn embedding_clone_is_independent() {
        let mut original = Embedding { embedding: vec![1.0, 2.0] , norm: None };
        let cloned = original.clone();
        original.embedding[0] = 99.0;
        assert!((cloned.embedding[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_with_subnormal_values() {
        let subnormal = f32::from_bits(1);
        assert!(subnormal.is_subnormal());
        let e = Embedding { embedding: vec![subnormal, 0.0] , norm: None };
        assert_eq!(e.len(), 2);
        assert!((e.embedding[0] - subnormal).abs() < 1e-38);
    }

    // ══════════════════════════════════════════════════════════════════════
    // NEW: EmbeddingsResponse — PartialEq, Clone independence, Index panic
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn embeddings_response_partial_eq_equal() {
        let a = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0, 2.0] , norm: None }],
            rerank_scores: Some(vec![0.9]),
            request_id: Some(1),
        };
        let b = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0, 2.0] , norm: None }],
            rerank_scores: Some(vec![0.9]),
            request_id: Some(1),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn embeddings_response_partial_eq_different_embeddings() {
        let a = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        let b = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![2.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn embeddings_response_partial_eq_different_scores() {
        let a = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: Some(vec![0.9]),
            request_id: None,
        };
        let b = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: Some(vec![0.8]),
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn embeddings_response_partial_eq_scores_some_vs_none() {
        let a = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: Some(vec![0.9]),
            request_id: None,
        };
        let b = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn embeddings_response_partial_eq_different_request_id() {
        let a = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: Some(1),
        };
        let b = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: Some(2),
        };
        assert_ne!(a, b);
    }

    #[test]
    fn embeddings_response_clone_produces_equal() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0, 2.0] , norm: None }],
            rerank_scores: Some(vec![0.9]),
            request_id: Some(1),
        };
        assert_eq!(resp, resp.clone());
    }

    #[test]
    fn embeddings_response_clone_is_independent() {
        let mut resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: Some(vec![0.5]),
            request_id: Some(1),
        };
        let cloned = resp.clone();
        resp.embeddings[0].embedding[0] = 99.0;
        assert!((cloned.embeddings[0].embedding[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic]
    fn embeddings_response_index_out_of_bounds_panics() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        let _ = &resp[1];
    }

    #[test]
    #[should_panic]
    fn embeddings_response_index_empty_panics() {
        let resp = EmbeddingsResponse {
            embeddings: vec![],
            rerank_scores: None,
            request_id: None,
        };
        let _ = &resp[0];
    }

    #[test]
    fn embeddings_response_mixed_embedding_dimensions() {
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![1.0; 128] , norm: None },
                Embedding { embedding: vec![2.0; 256] , norm: None },
                Embedding { embedding: vec![3.0; 512] , norm: None },
            ],
            rerank_scores: Some(vec![0.9, 0.8, 0.7]),
            request_id: None,
        };
        assert_eq!(resp[0].len(), 128);
        assert_eq!(resp[1].len(), 256);
        assert_eq!(resp[2].len(), 512);
    }

    #[test]
    fn embeddings_response_many_embeddings() {
        let embeddings: Vec<Embedding> = (0..1000)
            .map(|i| Embedding { embedding: vec![i as f32; 8] , norm: None })
            .collect();
        let resp = EmbeddingsResponse {
            embeddings,
            rerank_scores: None,
            request_id: Some(999),
        };
        assert_eq!(resp.len(), 1000);
        assert!((resp[0].embedding[0] - 0.0).abs() < 1e-6);
        assert!((resp[999].embedding[0] - 999.0).abs() < 1e-6);
    }

    // ══════════════════════════════════════════════════════════════════════
    // NEW: RagResponse — PartialEq, Clone independence, request_id zero
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn rag_response_partial_eq_equal() {
        let a = RagResponse {
            text: "same".into(),
            sources: vec![0, 1],
            rerank_scores: vec![0.9, 0.8],
            request_id: Some(5),
        };
        let b = RagResponse {
            text: "same".into(),
            sources: vec![0, 1],
            rerank_scores: vec![0.9, 0.8],
            request_id: Some(5),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn rag_response_partial_eq_different_text() {
        let a = RagResponse {
            text: "alpha".into(),
            sources: vec![],
            rerank_scores: vec![],
            request_id: None,
        };
        let b = RagResponse {
            text: "beta".into(),
            sources: vec![],
            rerank_scores: vec![],
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rag_response_partial_eq_different_sources() {
        let a = RagResponse {
            text: "same".into(),
            sources: vec![0],
            rerank_scores: vec![0.5],
            request_id: None,
        };
        let b = RagResponse {
            text: "same".into(),
            sources: vec![1],
            rerank_scores: vec![0.5],
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rag_response_partial_eq_different_request_id() {
        let a = RagResponse {
            text: "same".into(),
            sources: vec![],
            rerank_scores: vec![],
            request_id: Some(1),
        };
        let b = RagResponse {
            text: "same".into(),
            sources: vec![],
            rerank_scores: vec![],
            request_id: Some(2),
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rag_response_clone_produces_equal() {
        let resp = RagResponse {
            text: "answer".into(),
            sources: vec![0, 1],
            rerank_scores: vec![0.9, 0.5],
            request_id: Some(42),
        };
        assert_eq!(resp, resp.clone());
    }

    #[test]
    fn rag_response_clone_is_independent() {
        let mut resp = RagResponse {
            text: "original".into(),
            sources: vec![0],
            rerank_scores: vec![0.5],
            request_id: Some(1),
        };
        let cloned = resp.clone();
        resp.text = "modified".into();
        assert_eq!(cloned.text, "original");
    }

    #[test]
    fn rag_response_request_id_zero() {
        let resp = RagResponse {
            text: "answer".into(),
            sources: vec![0],
            rerank_scores: vec![0.5],
            request_id: Some(0),
        };
        assert_eq!(resp.request_id, Some(0));
    }

    // ══════════════════════════════════════════════════════════════════════
    // NEW: EmbeddingsBuilder — error path (generate_answer without rerank_query)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn embeddings_builder_generate_answer_without_rerank_query_returns_error() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["hello".into()]);
        let result = builder.generate_answer("system prompt");
        assert!(result.is_err());
        let err_msg = match result.unwrap_err() {
            GllmError::RuntimeError(msg) => msg,
            other => format!("{:?}", other),
        };
        assert!(err_msg.contains("generate_answer requires rerank_query"));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Additional edge case tests
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn embedding_with_all_zeros() {
        let e = Embedding { embedding: vec![0.0; 512] , norm: None };
        assert_eq!(e.len(), 512);
        assert!(e.embedding.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn embedding_with_f32_extremes() {
        let e = Embedding {
            embedding: vec![f32::MIN_POSITIVE, f32::MAX],
            norm: None,
        };
        assert!((e.embedding[0] - f32::MIN_POSITIVE).abs() < 1e-38);
        assert!((e.embedding[1] - f32::MAX).abs() < 1e-6);
    }

    #[test]
    fn embedding_nan_is_not_equal_to_itself() {
        let e = Embedding { embedding: vec![f32::NAN] , norm: None };
        assert_ne!(e, e);
    }

    #[test]
    fn embeddings_response_empty_with_scores_present() {
        let resp = EmbeddingsResponse {
            embeddings: vec![],
            rerank_scores: Some(vec![]),
            request_id: None,
        };
        assert!(resp.is_empty());
        assert_eq!(resp.len(), 0);
        assert!(resp.rerank_scores.as_ref().unwrap().is_empty());
    }

    #[test]
    fn embeddings_response_partial_eq_both_scores_none() {
        let a = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        let b = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn embeddings_response_rerank_score_f32_min() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![0.0; 4] , norm: None }],
            rerank_scores: Some(vec![f32::MIN]),
            request_id: None,
        };
        assert!((resp.rerank_scores.unwrap()[0] - f32::MIN).abs() < 1e-6);
    }

    #[test]
    fn embeddings_response_many_rerank_scores() {
        let count = 2000;
        let embeddings: Vec<Embedding> = (0..count)
            .map(|_| Embedding { embedding: vec![0.0; 2] , norm: None })
            .collect();
        let scores: Vec<f32> = (0..count).map(|i| i as f32 / count as f32).collect();
        let resp = EmbeddingsResponse {
            embeddings,
            rerank_scores: Some(scores),
            request_id: None,
        };
        assert_eq!(resp.len(), count);
        assert_eq!(resp.rerank_scores.as_ref().unwrap().len(), count);
    }

    #[test]
    fn embeddings_response_large_single_embedding_dimension() {
        let dim = 8192;
        let e = Embedding { embedding: vec![0.5; dim] , norm: None };
        let resp = EmbeddingsResponse {
            embeddings: vec![e],
            rerank_scores: None,
            request_id: None,
        };
        assert_eq!(resp[0].len(), dim);
    }

    #[test]
    fn embeddings_response_debug_contains_field_values() {
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: Some(vec![0.75]),
            request_id: Some(42),
        };
        let debug = format!("{resp:?}");
        assert!(debug.contains("0.75") || debug.contains("rerank_scores"));
        assert!(debug.contains("42") || debug.contains("request_id"));
    }

    #[test]
    fn rag_response_partial_eq_different_rerank_scores() {
        let a = RagResponse {
            text: "same".into(),
            sources: vec![0],
            rerank_scores: vec![0.9],
            request_id: None,
        };
        let b = RagResponse {
            text: "same".into(),
            sources: vec![0],
            rerank_scores: vec![0.1],
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rag_response_mismatched_sources_and_scores_lengths() {
        let resp = RagResponse {
            text: "result".into(),
            sources: vec![0, 1, 2],
            rerank_scores: vec![0.9],
            request_id: None,
        };
        assert_eq!(resp.sources.len(), 3);
        assert_eq!(resp.rerank_scores.len(), 1);
    }

    #[test]
    fn rag_response_text_with_special_characters() {
        let resp = RagResponse {
            text: "line1\nline2\ttab\rcr\"quote\\slash".into(),
            sources: vec![0],
            rerank_scores: vec![1.0],
            request_id: None,
        };
        assert!(resp.text.contains('\n'));
        assert!(resp.text.contains('\t'));
        assert!(resp.text.contains('\\'));
    }

    #[test]
    fn rag_response_many_sources() {
        let sources: Vec<usize> = (0..500).collect();
        let scores: Vec<f32> = (0..500).map(|i| 1.0 - i as f32 / 500.0).collect();
        let resp = RagResponse {
            text: "many".into(),
            sources,
            rerank_scores: scores,
            request_id: Some(1),
        };
        assert_eq!(resp.sources.len(), 500);
        assert_eq!(resp.sources[499], 499);
        assert!((resp.rerank_scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rag_response_debug_all_fields_present() {
        let resp = RagResponse {
            text: "hello".into(),
            sources: vec![0],
            rerank_scores: vec![0.9],
            request_id: Some(7),
        };
        let debug = format!("{resp:?}");
        assert!(debug.contains("text"));
        assert!(debug.contains("sources"));
        assert!(debug.contains("rerank_scores"));
        assert!(debug.contains("request_id"));
    }

    #[test]
    fn rag_response_clone_scores_independence() {
        let mut resp = RagResponse {
            text: "original".into(),
            sources: vec![0],
            rerank_scores: vec![0.5],
            request_id: None,
        };
        let cloned = resp.clone();
        resp.rerank_scores[0] = 99.0;
        assert!((cloned.rerank_scores[0] - 0.5).abs() < 1e-6);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Additional 15 tests: builder state, structural edge cases, clone
    // independence for remaining fields, large dimensions, Display patterns
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn embeddings_builder_new_has_no_rerank_query_or_top_n() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["input".into()]);
        let resp = builder.generate();
        assert!(resp.is_err());
    }

    #[test]
    fn embeddings_builder_with_rerank_query_set_triggers_pipeline_error() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["doc".into()])
            .rerank_query("query");
        let resp = builder.generate();
        assert!(resp.is_err());
    }

    #[test]
    fn embeddings_builder_with_top_n_only_triggers_plain_embed_error() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["text".into()])
            .top_n(5);
        let resp = builder.generate();
        assert!(resp.is_err());
    }

    #[test]
    fn embeddings_builder_with_empty_inputs_triggers_error() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec![]);
        let resp = builder.generate();
        assert!(resp.is_err());
    }

    #[test]
    fn embeddings_builder_generate_answer_with_rerank_query_triggers_error() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["doc".into()])
            .rerank_query("query")
            .top_n(2);
        let result = builder.generate_answer("system");
        assert!(result.is_err());
    }

    #[test]
    fn embedding_partial_eq_both_empty() {
        let a = Embedding { embedding: vec![] , norm: None };
        let b = Embedding { embedding: vec![] , norm: None };
        assert_eq!(a, b);
        assert!(a.is_empty() && b.is_empty());
    }

    #[test]
    fn embeddings_response_partial_eq_request_id_some_vs_none() {
        let a = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: Some(1),
        };
        let b = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rag_response_partial_eq_request_id_some_vs_none() {
        let a = RagResponse {
            text: "same".into(),
            sources: vec![0],
            rerank_scores: vec![0.5],
            request_id: Some(1),
        };
        let b = RagResponse {
            text: "same".into(),
            sources: vec![0],
            rerank_scores: vec![0.5],
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn embeddings_response_rerank_scores_all_zeros() {
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![0.0; 4] , norm: None },
                Embedding { embedding: vec![0.0; 4] , norm: None },
            ],
            rerank_scores: Some(vec![0.0, 0.0]),
            request_id: None,
        };
        let scores = resp.rerank_scores.as_ref().unwrap();
        assert!(scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn embedding_alternating_positive_negative() {
        let e = Embedding {
            embedding: vec![-1.0, 1.0, -2.0, 2.0, -3.0, 3.0],
            norm: None,
        };
        assert_eq!(e.len(), 6);
        assert!((e.embedding[0] - (-1.0)).abs() < 1e-6);
        assert!((e.embedding[5] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn rag_response_sources_sequential_indices() {
        let sources: Vec<usize> = (0..10).collect();
        let resp = RagResponse {
            text: "ordered".into(),
            sources,
            rerank_scores: vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            request_id: None,
        };
        for i in 0..10 {
            assert_eq!(resp.sources[i], i);
        }
    }

    #[test]
    fn embeddings_response_clone_rerank_scores_independence() {
        let mut resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: Some(vec![0.75]),
            request_id: None,
        };
        let cloned = resp.clone();
        if let Some(ref mut scores) = resp.rerank_scores {
            scores[0] = 0.0;
        }
        assert!((cloned.rerank_scores.as_ref().unwrap()[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn embeddings_response_clone_request_id_independence() {
        let mut resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![1.0] , norm: None }],
            rerank_scores: None,
            request_id: Some(42),
        };
        let cloned = resp.clone();
        resp.request_id = Some(99);
        assert_eq!(cloned.request_id, Some(42));
    }

    #[test]
    fn rag_response_clone_sources_independence() {
        let mut resp = RagResponse {
            text: "text".into(),
            sources: vec![0, 1, 2],
            rerank_scores: vec![0.5, 0.5, 0.5],
            request_id: None,
        };
        let cloned = resp.clone();
        resp.sources[0] = 99;
        assert_eq!(cloned.sources, vec![0, 1, 2]);
    }

    #[test]
    fn embedding_large_dimension_vector() {
        let dim = 16384;
        let e = Embedding { embedding: vec![0.25; dim] , norm: None };
        assert_eq!(e.len(), dim);
        assert!(!e.is_empty());
        assert!(e.embedding.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Wave 13 additional tests: remaining edge cases and boundary conditions
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn embedding_with_negative_infinity() {
        // Only positive infinity was tested; verify negative infinity round-trips.
        let e = Embedding {
            embedding: vec![f32::NEG_INFINITY, 0.0, 1.0],
            norm: None,
        };
        assert_eq!(e.len(), 3);
        assert!(e.embedding[0].is_infinite());
        assert!(e.embedding[0].is_sign_negative());
        assert!(!e.embedding[1].is_infinite());
    }

    #[test]
    fn embedding_with_f32_negative_max() {
        // f32::MIN (negative max) is distinct from NEG_INFINITY.
        let e = Embedding {
            embedding: vec![f32::MIN],
            norm: None,
        };
        assert!(e.embedding[0].is_normal());
        assert!(e.embedding[0].is_sign_negative());
        assert!((e.embedding[0] - f32::MIN).abs() < 1e-6);
    }

    #[test]
    fn rag_response_with_very_long_text() {
        // Verify RagResponse handles a large text string without issue.
        let long_text = "x".repeat(1_000_000);
        let resp = RagResponse {
            text: long_text.clone(),
            sources: vec![0],
            rerank_scores: vec![1.0],
            request_id: None,
        };
        assert_eq!(resp.text.len(), 1_000_000);
        assert_eq!(resp.text, long_text);
    }

    #[test]
    fn rag_response_empty_scores_with_nonempty_sources() {
        // Structural edge: sources present but rerank_scores empty.
        let resp = RagResponse {
            text: "partial".into(),
            sources: vec![0, 1, 2],
            rerank_scores: vec![],
            request_id: Some(5),
        };
        assert_eq!(resp.sources.len(), 3);
        assert!(resp.rerank_scores.is_empty());
        assert_eq!(resp.request_id, Some(5));
    }

    #[test]
    fn embeddings_response_index_first_of_many() {
        // Boundary: verify Index<usize> returns the very first element correctly.
        let resp = EmbeddingsResponse {
            embeddings: (0..50)
                .map(|i| Embedding { embedding: vec![i as f32] , norm: None })
                .collect(),
            rerank_scores: None,
            request_id: None,
        };
        assert!((resp[0].embedding[0] - 0.0).abs() < 1e-6);
        assert!((resp[49].embedding[0] - 49.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_with_negative_subnormal() {
        // Negative subnormal: smallest negative denormalized float.
        let neg_subnormal = -f32::from_bits(1);
        assert!(neg_subnormal.is_subnormal());
        assert!(neg_subnormal.is_sign_negative());
        let e = Embedding {
            embedding: vec![neg_subnormal, 0.0, 1.0],
            norm: None,
        };
        assert_eq!(e.len(), 3);
        assert!((e.embedding[0] - neg_subnormal).abs() < 1e-38);
    }

    #[test]
    fn rag_response_single_source_single_score() {
        // Minimal valid RagResponse: exactly one source and one score.
        let resp = RagResponse {
            text: "only one".into(),
            sources: vec![7],
            rerank_scores: vec![0.42],
            request_id: Some(1),
        };
        assert_eq!(resp.sources.len(), 1);
        assert_eq!(resp.sources[0], 7);
        assert!((resp.rerank_scores[0] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn embeddings_response_identical_embeddings() {
        // Multiple embeddings with identical values should all be equal.
        let e = Embedding { embedding: vec![0.5; 64] , norm: None };
        let resp = EmbeddingsResponse {
            embeddings: vec![e.clone(), e.clone(), e.clone()],
            rerank_scores: Some(vec![1.0, 1.0, 1.0]),
            request_id: None,
        };
        assert_eq!(resp[0], resp[1]);
        assert_eq!(resp[1], resp[2]);
        assert_eq!(resp.len(), 3);
    }

    #[test]
    fn rag_response_text_with_null_byte() {
        // String with embedded NUL byte (legal Rust String, unusual content).
        let text = "before\0after".to_string();
        let resp = RagResponse {
            text: text.clone(),
            sources: vec![0],
            rerank_scores: vec![1.0],
            request_id: None,
        };
        assert!(resp.text.contains('\0'));
        assert_eq!(resp.text, "before\0after");
    }

    #[test]
    fn embeddings_response_rerank_scores_with_zero_count() {
        // Some(Vec::new()) for rerank_scores is distinct from None.
        let resp = EmbeddingsResponse {
            embeddings: vec![],
            rerank_scores: Some(vec![]),
            request_id: Some(0),
        };
        assert!(resp.rerank_scores.is_some());
        assert!(resp.rerank_scores.as_ref().unwrap().is_empty());
        assert!(resp.is_empty());
    }

    #[test]
    fn rag_response_sources_zero_index_only() {
        // Edge: only source index 0, which is the minimum valid usize value.
        let resp = RagResponse {
            text: "first doc".into(),
            sources: vec![0],
            rerank_scores: vec![0.99],
            request_id: Some(100),
        };
        assert_eq!(resp.sources[0], 0);
        assert_eq!(resp.sources.len(), 1);
    }

    #[test]
    fn embedding_debug_shows_struct_name() {
        // Verify the Debug output contains the type name "Embedding".
        let e = Embedding { embedding: vec![1.0] , norm: None };
        let debug = format!("{e:?}");
        assert!(
            debug.contains("Embedding"),
            "Debug output should contain struct name 'Embedding', got: {debug}"
        );
    }

    #[test]
    fn rag_response_text_with_unicode_surrogate_range_characters() {
        // Unicode characters at boundaries: CJK, emoji, combining marks.
        let text = "Hello\u{00E9}world\u{1F600}\u{0301}test".to_string();
        let resp = RagResponse {
            text: text.clone(),
            sources: vec![0],
            rerank_scores: vec![1.0],
            request_id: None,
        };
        assert!(resp.text.contains('\u{1F600}'));
        assert!(resp.text.contains('\u{0301}'));
        assert_eq!(resp.text, text);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Wave 14: 13 additional unit tests — uncovered edge cases
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn embeddings_builder_rerank_query_accepts_str() {
        // Verify rerank_query accepts &str via Into<String>, not only String.
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["doc".into()])
            .rerank_query("a plain &str query");
        let result = builder.generate();
        assert!(result.is_err());
    }

    #[test]
    fn embeddings_builder_top_n_zero_still_triggers_pipeline() {
        // top_n(0) is valid; builder should still attempt the pipeline.
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["doc".into()])
            .rerank_query("query")
            .top_n(0);
        let result = builder.generate();
        assert!(result.is_err());
    }

    #[test]
    fn embeddings_builder_generate_answer_default_top_n_is_three() {
        // Without top_n, generate_answer uses default of 3.
        // We can only verify the error path since Client is empty,
        // but this confirms the method does not panic on missing top_n.
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["doc".into()])
            .rerank_query("query");
        let result = builder.generate_answer("system");
        assert!(result.is_err());
    }

    #[test]
    fn embeddings_response_partial_eq_both_request_id_none() {
        // Both request_id = None should compare equal when other fields match.
        let a = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![2.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        let b = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![2.0] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn embeddings_response_rerank_scores_with_nan_not_equal() {
        // Two EmbeddingsResponse with NaN rerank_scores should not be equal
        // (f32 NaN != NaN).
        let a = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![0.0] , norm: None }],
            rerank_scores: Some(vec![f32::NAN]),
            request_id: None,
        };
        let b = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![0.0] , norm: None }],
            rerank_scores: Some(vec![f32::NAN]),
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rag_response_text_whitespace_only() {
        // RagResponse with whitespace-only text is structurally valid.
        let resp = RagResponse {
            text: "   \t\n  ".into(),
            sources: vec![0],
            rerank_scores: vec![0.5],
            request_id: None,
        };
        assert!(!resp.text.is_empty());
        assert!(resp.text.trim().is_empty());
        assert_eq!(resp.text.len(), 7);
    }

    #[test]
    fn rag_response_sources_non_sequential_indices() {
        // Sources can be arbitrary indices, not necessarily sequential.
        let resp = RagResponse {
            text: "scattered".into(),
            sources: vec![42, 7, 999, 0, 3],
            rerank_scores: vec![0.9, 0.8, 0.7, 0.6, 0.5],
            request_id: None,
        };
        assert_eq!(resp.sources[0], 42);
        assert_eq!(resp.sources[3], 0);
        assert_eq!(resp.sources.len(), 5);
    }

    #[test]
    fn embeddings_response_rerank_scores_descending_order() {
        // Typical usage: rerank_scores sorted descending by relevance.
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![0.1; 4] , norm: None },
                Embedding { embedding: vec![0.2; 4] , norm: None },
                Embedding { embedding: vec![0.3; 4] , norm: None },
            ],
            rerank_scores: Some(vec![0.95, 0.80, 0.65]),
            request_id: Some(55),
        };
        let scores = resp.rerank_scores.as_ref().unwrap();
        for i in 1..scores.len() {
            assert!(
                scores[i - 1] >= scores[i],
                "Scores should be in descending order: {} >= {}",
                scores[i - 1],
                scores[i],
            );
        }
    }

    #[test]
    fn embedding_with_mixed_normal_and_special_floats() {
        // A single embedding containing normal, zero, subnormal, and infinity values.
        let subnormal = f32::from_bits(1);
        let e = Embedding {
            embedding: vec![1.0, 0.0, subnormal, f32::INFINITY, -42.5],
            norm: None,
        };
        assert_eq!(e.len(), 5);
        assert!(e.embedding[0].is_normal());
        assert_eq!(e.embedding[1], 0.0);
        assert!(e.embedding[2].is_subnormal());
        assert!(e.embedding[3].is_infinite());
        assert!((e.embedding[4] - (-42.5)).abs() < 1e-6);
    }

    #[test]
    fn embeddings_response_single_embedding_no_optional_fields() {
        // Minimal EmbeddingsResponse: one embedding, no rerank_scores, no request_id.
        let resp = EmbeddingsResponse {
            embeddings: vec![Embedding { embedding: vec![0.5; 256] , norm: None }],
            rerank_scores: None,
            request_id: None,
        };
        assert_eq!(resp.len(), 1);
        assert!(!resp.is_empty());
        assert!(resp.rerank_scores.is_none());
        assert!(resp.request_id.is_none());
        assert_eq!(resp[0].len(), 256);
    }

    #[test]
    fn rag_response_partial_eq_both_request_id_none() {
        // Both request_id = None should compare equal when other fields match.
        let a = RagResponse {
            text: "same".into(),
            sources: vec![1, 2],
            rerank_scores: vec![0.8, 0.6],
            request_id: None,
        };
        let b = RagResponse {
            text: "same".into(),
            sources: vec![1, 2],
            rerank_scores: vec![0.8, 0.6],
            request_id: None,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn embeddings_response_debug_empty_embeddings() {
        // Debug output for an empty response should still show field names.
        let resp = EmbeddingsResponse {
            embeddings: vec![],
            rerank_scores: None,
            request_id: None,
        };
        let debug = format!("{resp:#?}");
        assert!(debug.contains("embeddings"));
        assert!(debug.contains("rerank_scores"));
        assert!(debug.contains("request_id"));
    }

    #[test]
    fn rag_response_debug_shows_text_content() {
        // Debug output for RagResponse should contain the actual text string.
        let resp = RagResponse {
            text: "visible_text".into(),
            sources: vec![0],
            rerank_scores: vec![1.0],
            request_id: Some(99),
        };
        let debug = format!("{resp:?}");
        assert!(
            debug.contains("visible_text"),
            "Debug should contain the text value, got: {debug}"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // REQ-API-3: normalize, dimension, EmbedConfig
    // ══════════════════════════════════════════════════════════════════════

    // ── Embedding.norm field ──

    #[test]
    // @trace REQ-API-3 [entity:Embedding] embedding with norm
    fn embedding_with_norm_some() {
        let e = Embedding {
            embedding: vec![3.0, 4.0],
            norm: Some(5.0),
        };
        assert_eq!(e.norm(), Some(5.0));
        assert!(e.is_normalized());
    }

    #[test]
    // @trace REQ-API-3 [entity:Embedding] embedding without norm
    fn embedding_with_norm_none() {
        let e = Embedding {
            embedding: vec![1.0, 2.0],
            norm: None,
        };
        assert_eq!(e.norm(), None);
        assert!(!e.is_normalized());
    }

    #[test]
    // @trace REQ-API-3 [entity:Embedding] normalized embedding has unit length
    fn embedding_normalized_unit_length() {
        // [3,4] normalized -> [0.6, 0.8], norm was 5.0
        let e = Embedding {
            embedding: vec![0.6, 0.8],
            norm: Some(5.0),
        };
        let l2 = (e.embedding[0].powi(2) + e.embedding[1].powi(2)).sqrt();
        assert!((l2 - 1.0).abs() < 1e-6, "normalized embedding should have unit L2 norm");
    }

    #[test]
    // @trace REQ-API-3 [entity:Embedding] norm partial_eq with norm
    fn embedding_partial_eq_with_norm() {
        let a = Embedding { embedding: vec![1.0], norm: Some(1.0) };
        let b = Embedding { embedding: vec![1.0], norm: Some(1.0) };
        assert_eq!(a, b);
    }

    #[test]
    // @trace REQ-API-3 [entity:Embedding] norm partial_eq different norm
    fn embedding_partial_eq_different_norm() {
        let a = Embedding { embedding: vec![1.0], norm: Some(1.0) };
        let b = Embedding { embedding: vec![1.0], norm: None };
        assert_ne!(a, b);
    }

    #[test]
    // @trace REQ-API-3 [entity:Embedding] clone independence for norm
    fn embedding_clone_norm_independence() {
        let mut e = Embedding { embedding: vec![1.0], norm: Some(2.0) };
        let cloned = e.clone();
        e.norm = Some(99.0);
        assert_eq!(cloned.norm, Some(2.0));
    }

    // ── EmbedConfig ──

    #[test]
    // @trace REQ-API-3 [entity:EmbedConfig] default config
    fn embed_config_default() {
        let config = EmbedConfig::default();
        assert!(!config.normalize);
        assert_eq!(config.dimension, None);
    }

    #[test]
    // @trace REQ-API-3 [entity:EmbedConfig] builder pattern
    fn embed_config_builder() {
        let config = EmbedConfig::new()
            .normalize(true)
            .dimension(512);
        assert!(config.normalize);
        assert_eq!(config.dimension, Some(512));
    }

    #[test]
    // @trace REQ-API-3 [entity:EmbedConfig] dimension zero means None
    fn embed_config_dimension_zero_means_none() {
        let config = EmbedConfig::new().dimension(0);
        assert_eq!(config.dimension, None);
    }

    #[test]
    // @trace REQ-API-3 [entity:EmbedConfig] partial_eq
    fn embed_config_partial_eq() {
        let a = EmbedConfig::new().normalize(true).dimension(256);
        let b = EmbedConfig::new().normalize(true).dimension(256);
        assert_eq!(a, b);

        let c = EmbedConfig::new().normalize(false);
        assert_ne!(a, c);
    }

    #[test]
    // @trace REQ-API-3 [entity:EmbedConfig] clone
    fn embed_config_clone() {
        let config = EmbedConfig::new().normalize(true).dimension(128);
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    // ── EmbeddingsBuilder.normalize / dimension ──

    #[test]
    // @trace REQ-API-3 [entity:EmbeddingsBuilder] normalize builder
    fn embeddings_builder_normalize_default_false() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["test".into()]);
        // Builder with default normalize=false still calls execute_embeddings(normalize=false, dimension=None)
        let result = builder.generate();
        assert!(result.is_err()); // empty client, but the path is correct
    }

    #[test]
    // @trace REQ-API-3 [entity:EmbeddingsBuilder] normalize true
    fn embeddings_builder_normalize_true() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["test".into()])
            .normalize(true);
        let result = builder.generate();
        assert!(result.is_err()); // empty client
    }

    #[test]
    // @trace REQ-API-3 [entity:EmbeddingsBuilder] dimension config
    fn embeddings_builder_dimension_set() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["test".into()])
            .dimension(256);
        let result = builder.generate();
        assert!(result.is_err()); // empty client
    }

    #[test]
    // @trace REQ-API-3 [entity:EmbeddingsBuilder] normalize + dimension combined
    fn embeddings_builder_normalize_and_dimension() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["test".into()])
            .normalize(true)
            .dimension(512);
        let result = builder.generate();
        assert!(result.is_err()); // empty client
    }

    #[test]
    // @trace REQ-API-3 [entity:EmbeddingsBuilder] dimension zero means full
    fn embeddings_builder_dimension_zero_means_full() {
        let client = Client::new_empty();
        let builder = EmbeddingsBuilder::new(&client, vec!["test".into()])
            .dimension(0);
        let result = builder.generate();
        assert!(result.is_err()); // empty client, dimension(0) => None => full hidden size
    }

    // ── Normalization correctness (without executor) ──

    #[test]
    // @trace REQ-API-3 [entity:Embedding] manual normalization verification
    fn embedding_manual_normalization_correctness() {
        // Simulate what execute_embeddings does: normalize a raw vector
        let mut raw = vec![3.0_f32, 4.0_f32];
        let n = crate::jit::epilogue::compute_l2_norm(&raw);
        assert!((n - 5.0).abs() < 1e-6);
        if n > 0.0 {
            let inv = 1.0 / n;
            for v in raw.iter_mut() {
                *v *= inv;
            }
        }
        assert!((raw[0] - 0.6).abs() < 1e-6);
        assert!((raw[1] - 0.8).abs() < 1e-6);

        let e = Embedding { embedding: raw, norm: Some(n) };
        let unit_norm = (e.embedding[0].powi(2) + e.embedding[1].powi(2)).sqrt();
        assert!((unit_norm - 1.0).abs() < 1e-6);
        assert_eq!(e.norm(), Some(5.0));
    }

    #[test]
    // @trace REQ-API-3 [entity:Embedding] zero vector normalization is no-op
    fn embedding_zero_vector_normalization_noop() {
        let mut raw = vec![0.0_f32; 4];
        let n = crate::jit::epilogue::compute_l2_norm(&raw);
        assert_eq!(n, 0.0);
        // When norm is 0, normalization is skipped (no division by zero)
        if n > 0.0 {
            let inv = 1.0 / n;
            for v in raw.iter_mut() {
                *v *= inv;
            }
        }
        assert!(raw.iter().all(|&v| v == 0.0));
    }

    // ── Dimension truncation correctness ──

    #[test]
    // @trace REQ-API-3 [entity:Embedding] dimension truncation
    fn embedding_dimension_truncation() {
        let mut raw = vec![1.0_f32; 1024];
        let dim = 256;
        if dim < raw.len() {
            raw.truncate(dim);
        }
        let e = Embedding { embedding: raw, norm: None };
        assert_eq!(e.len(), 256);
    }

    #[test]
    // @trace REQ-API-3 [entity:Embedding] dimension larger than embedding is no-op
    fn embedding_dimension_larger_than_embedding_noop() {
        let mut raw = vec![1.0_f32; 128];
        let dim = 512;
        if dim < raw.len() {
            raw.truncate(dim);
        }
        let e = Embedding { embedding: raw, norm: None };
        assert_eq!(e.len(), 128); // unchanged
    }

    // ── EmbeddingsResponse with norm ──

    #[test]
    // @trace REQ-API-3 [entity:EmbeddingsResponse] response with normalized embeddings
    fn embeddings_response_with_normalized_embeddings() {
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![0.6, 0.8], norm: Some(5.0) },
                Embedding { embedding: vec![1.0, 0.0], norm: Some(1.0) },
            ],
            rerank_scores: None,
            request_id: None,
        };
        assert_eq!(resp.len(), 2);
        assert!(resp[0].is_normalized());
        assert!(resp[1].is_normalized());
        assert_eq!(resp[0].norm(), Some(5.0));
        assert_eq!(resp[1].norm(), Some(1.0));
    }

    #[test]
    // @trace REQ-API-3 [entity:EmbeddingsResponse] mixed normalized and unnormalized
    fn embeddings_response_mixed_normalized_unnormalized() {
        let resp = EmbeddingsResponse {
            embeddings: vec![
                Embedding { embedding: vec![0.6, 0.8], norm: Some(5.0) },
                Embedding { embedding: vec![3.0, 4.0], norm: None },
            ],
            rerank_scores: None,
            request_id: None,
        };
        assert!(resp[0].is_normalized());
        assert!(!resp[1].is_normalized());
    }

    // ── Embedding Debug with norm ──

    #[test]
    // @trace REQ-API-3 [entity:Embedding] debug output includes norm
    fn embedding_debug_with_norm() {
        let e = Embedding { embedding: vec![1.0], norm: Some(2.0) };
        let debug = format!("{e:?}");
        assert!(debug.contains("norm"));
    }
}
