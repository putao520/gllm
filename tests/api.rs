//! API tests using real models.
//! Requires models to be downloaded: ~/.gllm/models/

#![cfg(not(feature = "tokio"))]

use gllm::{Client, ClientConfig, Device};

fn build_client(model: &str) -> Client {
    let config = ClientConfig {
        device: Device::Cpu,
        ..Default::default()
    };
    Client::with_config(model, config).expect("client")
}

#[test]
fn embeddings_flow_cpu() {
    let client = build_client("bge-small-en");
    let response = client
        .embeddings(["hello world", "rust embeddings"])
        .generate()
        .expect("embeddings");

    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.embeddings[0].embedding.len(), 384);
    assert!(response.usage.prompt_tokens > 0);
}

#[test]
fn rerank_flow_cpu() {
    let client = build_client("bge-reranker-base");
    let response = client
        .rerank(
            "search query",
            ["first document content", "second document content"],
        )
        .top_n(1)
        .return_documents(true)
        .generate()
        .expect("rerank");

    assert_eq!(response.results.len(), 1);
    assert!(response.results[0].score >= 0.0);
    assert!(response.results[0].document.is_some());
}
