use crate::types::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model type supported by the library.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    /// Embedding encoder model.
    Embedding,
    /// Cross-encoder reranker model.
    Rerank,
}

/// Architecture of a model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Architecture {
    /// BERT-like encoder.
    Bert,
    /// Cross-encoder architecture.
    CrossEncoder,
}

/// Metadata describing a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ModelInfo {
    /// Alias for quick reference.
    pub alias: String,
    /// HuggingFace repository ID.
    pub repo_id: String,
    /// Model type.
    pub model_type: ModelType,
    /// Architecture descriptor.
    pub architecture: Architecture,
}

/// Registry of built-in model aliases.
pub(crate) struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Build a registry with built-in aliases.
    pub fn new() -> Self {
        let mut models = HashMap::new();
        let entries = [
            // BGE Embedding Models
            (
                "bge-m3",
                "BAAI/bge-m3",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "bge-large-zh",
                "BAAI/bge-large-zh-v1.5",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "bge-small-en",
                "BAAI/bge-small-en-v1.5",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "bge-base-en",
                "BAAI/bge-base-en-v1.5",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "bge-large-en",
                "BAAI/bge-large-en-v1.5",
                ModelType::Embedding,
                Architecture::Bert,
            ),

            // Sentence Transformers Models
            (
                "all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "all-mpnet-base-v2",
                "sentence-transformers/all-mpnet-base-v2",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "paraphrase-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-MiniLM-L6-v2",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "multi-qa-mpnet-base-dot-v1",
                "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                ModelType::Embedding,
                Architecture::Bert,
            ),

            // E5 Models
            (
                "e5-large",
                "intfloat/e5-large",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "e5-base",
                "intfloat/e5-base",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "e5-small",
                "intfloat/e5-small",
                ModelType::Embedding,
                Architecture::Bert,
            ),

            // JINA Embeddings
            (
                "jina-embeddings-v2-base-en",
                "jinaai/jina-embeddings-v2-base-en",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "jina-embeddings-v2-small-en",
                "jinaai/jina-embeddings-v2-small-en",
                ModelType::Embedding,
                Architecture::Bert,
            ),

            // Multilingual Models
            (
                "multilingual-MiniLM-L12-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "distiluse-base-multilingual-cased-v1",
                "sentence-transformers/distiluse-base-multilingual-cased-v1",
                ModelType::Embedding,
                Architecture::Bert,
            ),

            // Light Models for Edge Devices
            (
                "all-MiniLM-L12-v2",
                "sentence-transformers/all-MiniLM-L12-v2",
                ModelType::Embedding,
                Architecture::Bert,
            ),
            (
                "all-distilroberta-v1",
                "sentence-transformers/all-distilroberta-v1",
                ModelType::Embedding,
                Architecture::Bert,
            ),

            // BGE Rerankers
            (
                "bge-reranker-v2",
                "BAAI/bge-reranker-v2-m3",
                ModelType::Rerank,
                Architecture::CrossEncoder,
            ),
            (
                "bge-reranker-large",
                "BAAI/bge-reranker-large",
                ModelType::Rerank,
                Architecture::CrossEncoder,
            ),
            (
                "bge-reranker-base",
                "BAAI/bge-reranker-base",
                ModelType::Rerank,
                Architecture::CrossEncoder,
            ),

            // MS MARCO Rerankers
            (
                "ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ModelType::Rerank,
                Architecture::CrossEncoder,
            ),
            (
                "ms-marco-MiniLM-L-12-v2",
                "cross-encoder/ms-marco-MiniLM-L-12-v2",
                ModelType::Rerank,
                Architecture::CrossEncoder,
            ),
            (
                "ms-marco-TinyBERT-L-2-v2",
                "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                ModelType::Rerank,
                Architecture::CrossEncoder,
            ),
            (
                "ms-marco-electra-base",
                "cross-encoder/ms-marco-electra-base",
                ModelType::Rerank,
                Architecture::CrossEncoder,
            ),

            // Specialized Rerankers
            (
                "quora-distilroberta-base",
                "cross-encoder/quora-distilroberta-base",
                ModelType::Rerank,
                Architecture::CrossEncoder,
            ),
        ];

        for (alias, repo_id, model_type, architecture) in entries {
            let alias_key = alias.to_ascii_lowercase();
            models.insert(
                alias_key.clone(),
                ModelInfo {
                    alias: alias_key,
                    repo_id: repo_id.to_string(),
                    model_type,
                    architecture,
                },
            );
        }

        Self { models }
    }

    /// Resolve an alias or repo ID into a model info record.
    pub fn resolve(&self, name: &str) -> Result<ModelInfo> {
        let key = name.trim().to_ascii_lowercase();
        if let Some(info) = self.models.get(&key) {
            return Ok(info.clone());
        }

        // Allow direct HF repo IDs without registering.
        if name.contains('/') {
            return Ok(self.infer_from_repo(name));
        }

        // Support colon-separated shorthand like qwen2.5:7b.
        if name.contains(':') {
            let repo_id = name.replace(':', "-");
            return Ok(self.infer_from_repo(&repo_id));
        }

        Err(Error::ModelNotFound(name.to_string()))
    }

    fn infer_from_repo(&self, repo_id: &str) -> ModelInfo {
        let model_type = if repo_id.to_ascii_lowercase().contains("reranker") {
            ModelType::Rerank
        } else {
            ModelType::Embedding
        };

        let architecture = match model_type {
            ModelType::Embedding => Architecture::Bert,
            ModelType::Rerank => Architecture::CrossEncoder,
        };

        let alias = repo_id
            .rsplit('/')
            .next()
            .unwrap_or(repo_id)
            .to_ascii_lowercase();

        ModelInfo {
            alias,
            repo_id: repo_id.to_string(),
            model_type,
            architecture,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_builtin_model() {
        let registry = ModelRegistry::new();
        let info = registry.resolve("bge-m3").unwrap();
        assert_eq!(info.repo_id, "BAAI/bge-m3");
        assert!(matches!(info.architecture, Architecture::Bert));
    }

    #[test]
    fn resolves_repo_id_directly() {
        let registry = ModelRegistry::new();
        let repo = "BAAI/custom-model";
        let info = registry.resolve(repo).unwrap();
        assert_eq!(info.repo_id, repo);
        assert_eq!(info.alias, "custom-model");
    }
}
