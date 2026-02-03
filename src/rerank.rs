//! Rerank API skeleton.

use crate::client::{Client, ClientError};

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

    pub fn top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }

    pub fn generate(self) -> Result<RerankResponse, ClientError> {
        self.client
            .execute_rerank(self.query, self.documents, self.top_n)
    }
}

#[derive(Debug, Clone)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
    pub request_id: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}
