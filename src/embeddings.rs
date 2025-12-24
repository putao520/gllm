use crate::engine::{EngineBackend, TokenizerAdapter, MAX_SEQ_LEN};
use crate::types::{Embedding, EmbeddingResponse, Error, Result, Usage};

/// Embeddings request builder.
pub struct EmbeddingsBuilder<'a> {
    pub(crate) engine: &'a EngineBackend,
    pub(crate) tokenizer: &'a TokenizerAdapter,
    pub(crate) inputs: Vec<String>,
    pub(crate) graph_inputs: Vec<crate::types::GraphCodeInput>,
}

impl<'a> EmbeddingsBuilder<'a> {
    fn run(self) -> Result<EmbeddingResponse> {
        let has_text = !self.inputs.is_empty();
        let has_graph = !self.graph_inputs.is_empty();

        if !has_text && !has_graph {
            return Err(Error::InvalidConfig(
                "At least one input (text or graph code) is required for embeddings".into(),
            ));
        }

        let total_items = self.inputs.len() + self.graph_inputs.len();
        let mut token_batches = Vec::with_capacity(total_items);
        let mut token_count = 0usize;

        // Process pure text inputs
        for text in &self.inputs {
            let (tokens, used) = self.tokenizer.encode(text, MAX_SEQ_LEN);
            token_batches.push(tokens);
            token_count += used;
        }

        // Process graph inputs (currently treats code as text, ignoring DFG masks until engine update)
        for graph_input in &self.graph_inputs {
             let (tokens, used) = self.tokenizer.encode(&graph_input.code, MAX_SEQ_LEN);
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
