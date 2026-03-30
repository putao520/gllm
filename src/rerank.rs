//! Rerank API — async-first design (per SPEC 04-API-DESIGN §3.3).

use crate::client::{Client, GllmError};

/// Builder for document reranking (per SPEC 04-API-DESIGN §3.3).
///
/// # Example
///
/// ```no_run
/// use gllm::Client;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_empty();
/// let scores = client.rerank(
///     "What is the capital of France?",
///     vec![
///         "Paris is the capital of France",
///         "London is in UK",
///         "Berlin is in Germany"
///     ]
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub struct RerankBuilder<'a> {
    client: &'a Client,
    query: String,
    documents: Vec<String>,
    top_n: usize,
}

impl<'a> RerankBuilder<'a> {
    pub(crate) fn new(
        client: &'a Client,
        query: impl Into<String>,
        documents: Vec<String>,
    ) -> Self {
        Self {
            client,
            query: query.into(),
            documents,
            top_n: 5,
        }
    }

    /// Set maximum number of results to return.
    pub fn top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }

    /// Execute the reranking (async).
    pub async fn generate(self) -> Result<RerankResponse, GllmError> {
        self.client
            .execute_rerank(self.query, self.documents, self.top_n)
            .await
    }
}

/// Response from reranking (per SPEC 04-API-DESIGN §3.3).
#[derive(Debug, Clone)]
pub struct RerankResponse {
    /// Sorted results by relevance score.
    pub results: Vec<RerankResult>,
    /// Request ID (for tracking).
    pub request_id: Option<u64>,
}

/// A single reranking result (per SPEC 04-API-DESIGN §3.3).
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Original document index.
    pub index: usize,
    /// Relevance score (higher = more relevant).
    pub score: f32,
}
