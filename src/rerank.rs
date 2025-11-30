use crate::engine::{EngineBackend, TokenizerAdapter, MAX_SEQ_LEN};
use crate::types::{Error, RerankResponse, RerankResult, Result};

/// Rerank request builder.
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

    fn run(self) -> Result<RerankResponse> {
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

// 同步接口
#[cfg(not(feature = "tokio"))]
impl<'a> RerankBuilder<'a> {
    /// Generate rerank results.
    pub fn generate(self) -> Result<RerankResponse> {
        self.run()
    }
}

// 异步接口
#[cfg(feature = "tokio")]
impl<'a> RerankBuilder<'a> {
    /// Generate rerank results.
    pub async fn generate(self) -> Result<RerankResponse> {
        tokio::task::block_in_place(|| self.run())
    }
}
