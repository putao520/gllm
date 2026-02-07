//! E2E 测试: Embedding (嵌入模型)
//!
//! 测试 3 种格式: SafeTensors, GGUF, ONNX
//! 验证同一功能的模型在不同格式下都能正常工作

use gllm::Client;

/// SafeTensors 格式的 Embedding 测试
///
/// 模型: intfloat/e5-small-v2
/// 格式: SafeTensors (.safetensors)
/// 源: [HuggingFace](https://huggingface.co/intfloat/e5-small-v2)
#[test]
fn e2e_embedding_safetensors() {
    const MODEL: &str = "intfloat/e5-small-v2";

    let client = Client::new_embedding(MODEL).expect("Failed to load SafeTensors model");
    let response = client
        .embeddings(["Hello, world!", "Test sentence"])
        .generate()
        .expect("Embedding failed");

    assert_eq!(response.embeddings.len(), 2, "Should have 2 embeddings");

    // 验证第一个 embedding
    let emb1 = &response.embeddings[0].embedding;
    assert!(!emb1.is_empty(), "Embedding should not be empty");
    assert_eq!(
        emb1.len(),
        384,
        "e5-small embedding dimension should be 384"
    );

    // 验证第二个 embedding
    let emb2 = &response.embeddings[1].embedding;
    assert!(!emb2.is_empty(), "Embedding should not be empty");
    assert_eq!(
        emb2.len(),
        384,
        "e5-small embedding dimension should be 384"
    );

    // 验证两个 embedding 不同
    let mut sum_diff = 0.0;
    for (a, b) in emb1.iter().zip(emb2.iter()) {
        sum_diff += (a - b).abs();
    }
    assert!(
        sum_diff > 0.1,
        "Different texts should have different embeddings"
    );
}

/// GGUF 格式的 Embedding 测试
///
/// 模型: Qwen/Qwen3-Embedding-0.6B-GGUF
/// 格式: GGUF (.gguf)
/// 源: [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)
///
#[test]
fn e2e_embedding_gguf() {
    const MODEL: &str = "Qwen/Qwen3-Embedding-0.6B-GGUF";

    let client = Client::new_embedding(MODEL).expect("Failed to load GGUF model");
    let manifest = client.manifest();
    assert_eq!(manifest.kind, gllm::ModelKind::Embedding);
}

/// ONNX 格式的 Embedding 测试
///
/// 模型: intfloat/multilingual-e5-small
/// 格式: ONNX (onnx/model.onnx)
/// 源: [HuggingFace](https://huggingface.co/intfloat/multilingual-e5-small)
#[test]
fn e2e_embedding_onnx() {
    const MODEL: &str = "intfloat/multilingual-e5-small";

    let client = Client::new_embedding(MODEL).expect("Failed to load ONNX model");
    let response = client
        .embeddings(["test query", "test document"])
        .generate()
        .expect("Embedding failed");

    assert_eq!(response.embeddings.len(), 2, "Should have 2 embeddings");

    // 验证 embedding 维度
    let emb1 = &response.embeddings[0].embedding;
    assert!(!emb1.is_empty(), "Embedding should not be empty");
    assert_eq!(
        emb1.len(),
        384,
        "e5-small embedding dimension should be 384"
    );

    // 验证两个 embedding 不同
    let emb2 = &response.embeddings[1].embedding;
    assert!(!emb2.is_empty(), "Embedding should not be empty");

    let mut sum_diff = 0.0;
    for (a, b) in emb1.iter().zip(emb2.iter()) {
        sum_diff += (a - b).abs();
    }
    assert!(
        sum_diff > 0.01,
        "Different texts should have different embeddings"
    );
}
