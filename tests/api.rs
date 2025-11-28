#![cfg(feature = "cpu")]

use gllm::{Client, ClientConfig, Device};
use safetensors::Dtype;
use safetensors::tensor::{TensorView, serialize};
use std::fs;

fn write_dummy_weights(path: &std::path::Path) {
    let weights: Vec<u8> = vec![0u8; 64]; // 16 f32 values = 64 bytes
    let shape = vec![4usize, 4usize];
    let tensor = TensorView::new(Dtype::F32, shape, &weights).expect("tensor view");
    let data = serialize([("dense.weight", tensor)].into_iter(), &None).expect("serialize");
    fs::write(path, data).expect("write weights");
}

fn build_client() -> Client {
    unsafe {
        std::env::set_var("GLLM_SKIP_DOWNLOAD", "1");
        std::env::set_var("HF_HUB_OFFLINE", "1");
        std::env::set_var("GLLM_TEST_MODE", "1");
    }
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let model_dir = temp_dir.path().join("BAAI--bge-m3");
    fs::create_dir_all(&model_dir).expect("create model dir");
    write_dummy_weights(&model_dir.join("model.safetensors"));

    let mut config = ClientConfig::default();
    config.models_dir = temp_dir.keep();
    config.device = Device::Cpu;

    Client::with_config("bge-m3", config).expect("client")
}

#[test]
fn embeddings_flow_cpu() {
    let client = build_client();
    let response = client
        .embeddings(["hello world", "rust embeddings"])
        .generate()
        .expect("embeddings");

    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.embeddings[0].embedding.len(), 128);
    assert!(response.usage.prompt_tokens > 0);
}

#[test]
fn rerank_flow_cpu() {
    let client = build_client();
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
