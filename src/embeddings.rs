//! Embeddings API skeleton.

use crate::client::{Client, ClientError};

pub struct EmbeddingsBuilder<'a> {
    client: &'a Client,
    inputs: Vec<String>,
}

impl<'a> EmbeddingsBuilder<'a> {
    pub(crate) fn new(client: &'a Client, inputs: Vec<String>) -> Self {
        Self { client, inputs }
    }

    pub fn generate(self) -> Result<EmbeddingsResponse, ClientError> {
        self.client.execute_embeddings(self.inputs)
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingsResponse {
    pub embeddings: Vec<Embedding>,
    pub request_id: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub embedding: Vec<f32>,
}
