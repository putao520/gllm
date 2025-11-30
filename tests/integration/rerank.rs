//! Rerank integration tests using real models.

use gllm::{Client, ClientConfig, Device, Result};
use std::collections::HashSet;

fn get_config() -> ClientConfig {
    ClientConfig {
        device: Device::Auto,
        ..Default::default()
    }
}

#[cfg(not(feature = "tokio"))]
#[test]
fn rerank_end_to_end() -> Result<()> {
    let client = Client::with_config("bge-reranker-v2", get_config())?;

    let documents = vec![
        "Machine learning is a subset of AI.",
        "The weather is sunny today.",
        "Rust focuses on safety and speed.",
    ];

    let response = client
        .rerank("What is machine learning?", &documents)
        .top_n(2)
        .return_documents(true)
        .generate()?;

    assert_eq!(response.results.len(), 2);

    let mut seen = HashSet::new();
    for window in response.results.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
    for result in &response.results {
        assert!(result.score.is_finite());
        assert!(result.index < documents.len());
        assert!(seen.insert(result.index));
        assert!(result.document.is_some());
    }

    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn rerank_end_to_end() -> Result<()> {
    let client = Client::with_config("bge-reranker-v2", get_config()).await?;

    let documents = vec![
        "Machine learning is a subset of AI.",
        "The weather is sunny today.",
        "Rust focuses on safety and speed.",
    ];

    let response = client
        .rerank("What is machine learning?", &documents)
        .top_n(2)
        .return_documents(true)
        .generate()
        .await?;

    assert_eq!(response.results.len(), 2);

    let mut seen = HashSet::new();
    for window in response.results.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
    for result in &response.results {
        assert!(result.score.is_finite());
        assert!(result.index < documents.len());
        assert!(seen.insert(result.index));
        assert!(result.document.is_some());
    }

    Ok(())
}
