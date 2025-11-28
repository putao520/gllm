use crate::engine::{EngineBackend, MAX_SEQ_LEN, TokenizerAdapter};
use crate::types::{Embedding, EmbeddingResponse, Error, Result, Usage};

/// Embeddings request builder (synchronous).
pub struct EmbeddingsBuilder<'a> {
    pub(crate) engine: &'a EngineBackend,
    pub(crate) tokenizer: &'a TokenizerAdapter,
    pub(crate) inputs: Vec<String>,
}

impl<'a> EmbeddingsBuilder<'a> {
    /// Generate embeddings synchronously.
    pub fn generate(self) -> Result<EmbeddingResponse> {
        if self.inputs.is_empty() {
            return Err(Error::InvalidConfig(
                "At least one input is required for embeddings".into(),
            ));
        }

        let mut token_batches = Vec::with_capacity(self.inputs.len());
        let mut token_count = 0usize;
        for text in &self.inputs {
            let (tokens, used) = self.tokenizer.encode(text, MAX_SEQ_LEN);
            token_batches.push(tokens);
            token_count += used;
        }

        let vectors = self.engine.run_embeddings(&token_batches)?;
        let embeddings = vectors
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| Embedding { index, embedding })
            .collect();

        let usage = Usage {
            prompt_tokens: token_count,
            total_tokens: token_count,
        };

        Ok(EmbeddingResponse { embeddings, usage })
    }
}

/// Embeddings request builder (asynchronous).
#[cfg(feature = "async")]
pub struct AsyncEmbeddingsBuilder<'a> {
    pub(crate) inner: EmbeddingsBuilder<'a>,
}

#[cfg(feature = "async")]
impl<'a> AsyncEmbeddingsBuilder<'a> {
    /// Generate embeddings asynchronously.
    pub async fn generate(self) -> Result<EmbeddingResponse> {
        self.inner.generate()
    }
}
