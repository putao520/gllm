use crate::engine::{EngineBackend, MAX_SEQ_LEN, TokenizerAdapter};
use crate::types::{Error, RerankResponse, RerankResult, Result};

/// Rerank request builder (synchronous).
pub struct RerankBuilder<'a> {
    pub(crate) engine: &'a EngineBackend,
    pub(crate) tokenizer: &'a TokenizerAdapter,
    pub(crate) query: String,
    pub(crate) documents: Vec<String>,
    pub(crate) top_n: Option<usize>,
    pub(crate) return_documents: bool,
}

impl<'a> RerankBuilder<'a> {
    /// Limit the number of results returned.
    pub fn top_n(mut self, n: usize) -> Self {
        self.top_n = Some(n);
        self
    }

    /// Enable or disable returning raw documents in results.
    pub fn return_documents(mut self, return_docs: bool) -> Self {
        self.return_documents = return_docs;
        self
    }

    /// Generate rerank results synchronously.
    pub fn generate(self) -> Result<RerankResponse> {
        if self.documents.is_empty() {
            return Err(Error::InvalidConfig(
                "At least one document is required for rerank".into(),
            ));
        }

        let mut token_pairs = Vec::with_capacity(self.documents.len());
        for doc in &self.documents {
            let (tokens, _) =
                self.tokenizer
                    .encode_pair(&self.query, doc, MAX_SEQ_LEN.saturating_sub(2));
            token_pairs.push(tokens);
        }

        let scores = self.engine.run_rerank(&token_pairs)?;
        let mut results: Vec<RerankResult> = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankResult {
                index,
                score,
                document: if self.return_documents {
                    self.documents.get(index).cloned()
                } else {
                    None
                },
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(limit) = self.top_n {
            results.truncate(limit.min(results.len()));
        }

        Ok(RerankResponse { results })
    }
}

/// Rerank request builder (asynchronous).
#[cfg(feature = "async")]
pub struct AsyncRerankBuilder<'a> {
    pub(crate) inner: RerankBuilder<'a>,
}

#[cfg(feature = "async")]
impl<'a> AsyncRerankBuilder<'a> {
    /// Limit the number of results returned.
    pub fn top_n(mut self, n: usize) -> Self {
        self.inner.top_n = Some(n);
        self
    }

    /// Enable or disable returning raw documents in results.
    pub fn return_documents(mut self, return_docs: bool) -> Self {
        self.inner.return_documents = return_docs;
        self
    }

    /// Generate rerank results asynchronously.
    pub async fn generate(self) -> Result<RerankResponse> {
        self.inner.generate()
    }
}
