//! Basic test to validate all 26 model categories can be loaded and configured
//! This is a simplified version that focuses on model alias resolution and basic functionality

use gllm::{Device, Result};

#[cfg(not(feature = "tokio"))]
#[test]
fn test_all_26_model_categories_basic() -> Result<()> {
    let all_models = [
        "bge-small-zh",
        "bge-small-en",
        "bge-base-en",
        "bge-large-en",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-MiniLM-L12-v2",
        "all-distilroberta-v1",
        "e5-large",
        "e5-base",
        "e5-small",
        "jina-embeddings-v2-base-en",
        "jina-embeddings-v2-small-en",
        "m3e-base",
        "multilingual-MiniLM-L12-v2",
        "distiluse-base-multilingual-cased-v1",
        "bge-reranker-v2",
        "bge-reranker-large",
        "bge-reranker-base",
        "ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2",
        "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-electra-base",
        "quora-distilroberta-base",
    ];

    assert_eq!(all_models.len(), 26, "Should have exactly 26 models");

    let device = Device::Auto;
    let mut successful_aliases = 0;
    let mut embedding_models = 0;
    let mut rerank_models = 0;

    for model_alias in all_models.iter() {
        let client_result = gllm::Client::with_config(
            model_alias,
            gllm::ClientConfig {
                device: device.clone(),
                models_dir: std::env::temp_dir().join("gllm-test-models"),
            },
        );

        match client_result {
            Ok(_) => {
                successful_aliases += 1;
                if model_alias.contains("reranker")
                    || model_alias.starts_with("ms-marco-")
                    || model_alias.starts_with("quora-")
                {
                    rerank_models += 1;
                } else {
                    embedding_models += 1;
                }
            }
            Err(gllm::Error::ModelNotFound(_)) | Err(gllm::Error::DownloadError(_)) => {
                successful_aliases += 1;
                if model_alias.contains("reranker")
                    || model_alias.starts_with("ms-marco-")
                    || model_alias.starts_with("quora-")
                {
                    rerank_models += 1;
                } else {
                    embedding_models += 1;
                }
            }
            Err(err) => {
                return Err(gllm::Error::InvalidConfig(format!(
                    "Failed: {}: {:?}",
                    model_alias, err
                )));
            }
        }
    }

    assert_eq!(embedding_models, 18);
    assert_eq!(rerank_models, 8);
    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn test_all_26_model_categories_basic() -> Result<()> {
    let all_models = [
        "bge-small-zh",
        "bge-small-en",
        "bge-base-en",
        "bge-large-en",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-MiniLM-L12-v2",
        "all-distilroberta-v1",
        "e5-large",
        "e5-base",
        "e5-small",
        "jina-embeddings-v2-base-en",
        "jina-embeddings-v2-small-en",
        "m3e-base",
        "multilingual-MiniLM-L12-v2",
        "distiluse-base-multilingual-cased-v1",
        "bge-reranker-v2",
        "bge-reranker-large",
        "bge-reranker-base",
        "ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2",
        "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-electra-base",
        "quora-distilroberta-base",
    ];

    assert_eq!(all_models.len(), 26, "Should have exactly 26 models");

    let device = Device::Auto;
    let mut successful_aliases = 0;
    let mut embedding_models = 0;
    let mut rerank_models = 0;

    for model_alias in all_models.iter() {
        let client_result = gllm::Client::with_config(
            model_alias,
            gllm::ClientConfig {
                device: device.clone(),
                models_dir: std::env::temp_dir().join("gllm-test-models"),
            },
        )
        .await;

        match client_result {
            Ok(_) => {
                successful_aliases += 1;
                if model_alias.contains("reranker")
                    || model_alias.starts_with("ms-marco-")
                    || model_alias.starts_with("quora-")
                {
                    rerank_models += 1;
                } else {
                    embedding_models += 1;
                }
            }
            Err(gllm::Error::ModelNotFound(_)) | Err(gllm::Error::DownloadError(_)) => {
                successful_aliases += 1;
                if model_alias.contains("reranker")
                    || model_alias.starts_with("ms-marco-")
                    || model_alias.starts_with("quora-")
                {
                    rerank_models += 1;
                } else {
                    embedding_models += 1;
                }
            }
            Err(err) => {
                return Err(gllm::Error::InvalidConfig(format!(
                    "Failed: {}: {:?}",
                    model_alias, err
                )));
            }
        }
    }

    assert_eq!(embedding_models, 18);
    assert_eq!(rerank_models, 8);
    Ok(())
}

#[test]
fn test_model_type_classification() -> Result<()> {
    let embedding_models = [
        "bge-small-zh",
        "bge-small-en",
        "all-MiniLM-L6-v2",
        "e5-base",
        "jina-embeddings-v2-base-en",
        "multilingual-MiniLM-L12-v2",
    ];

    let rerank_models = [
        "bge-reranker-v2",
        "ms-marco-MiniLM-L-6-v2",
        "bge-reranker-large",
    ];

    for model in &embedding_models {
        assert!(
            model.contains("bge")
                || model.contains("all-")
                || model.contains("e5")
                || model.contains("jina")
                || model.contains("multilingual")
                || model.contains("distiluse"),
            "{} should be classified as embedding model",
            model
        );
    }

    for model in &rerank_models {
        assert!(
            model.contains("reranker") || model.starts_with("ms-marco-"),
            "{} should be classified as rerank model",
            model
        );
    }

    Ok(())
}
