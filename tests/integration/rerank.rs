use super::common::{
    env_lock, init_test_env, is_backend_unavailable, preferred_device, prepare_context_with_weights,
};
use gllm::Result;
use std::collections::HashSet;

#[cfg(not(feature = "tokio"))]
#[test]
fn rerank_end_to_end() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (client, _ctx) = match prepare_context_with_weights("bge-reranker-v2", device) {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

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
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(result.index < documents.len());
        assert!(seen.insert(result.index));
        assert!(result.document.is_some());
    }

    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn rerank_end_to_end() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (client, _ctx) = match prepare_context_with_weights("bge-reranker-v2", device).await {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

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
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(result.index < documents.len());
        assert!(seen.insert(result.index));
        assert!(result.document.is_some());
    }

    Ok(())
}
