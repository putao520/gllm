use crate::engine::{EngineBackend, TokenizerAdapter, MAX_SEQ_LEN};
use crate::types::{Embedding, EmbeddingResponse, Error, Result, Usage};

/// Embeddings request builder.
pub struct EmbeddingsBuilder<'a> {
    pub(crate) engine: &'a EngineBackend,
    pub(crate) tokenizer: &'a TokenizerAdapter,
    pub(crate) inputs: Vec<String>,
}

impl<'a> EmbeddingsBuilder<'a> {
    fn run(self) -> Result<EmbeddingResponse> {
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

// 同步接口
#[cfg(not(feature = "tokio"))]
impl<'a> EmbeddingsBuilder<'a> {
    /// Generate embeddings.
    pub fn generate(self) -> Result<EmbeddingResponse> {
        self.run()
    }
}

// 异步接口
#[cfg(feature = "tokio")]
impl<'a> EmbeddingsBuilder<'a> {
    /// Generate embeddings.
    pub async fn generate(self) -> Result<EmbeddingResponse> {
        tokio::task::block_in_place(|| self.run())
    }
}
