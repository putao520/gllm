//! E2E 测试: Reranker (重排序模型)
//!
//! 测试 3 种格式: SafeTensors, GGUF, ONNX
//! 验证同一功能的模型在不同格式下都能正常工作

use gllm::Client;

/// SafeTensors 格式的 Reranker 测试
///
/// 模型: BAAI/bge-reranker-v2-m3
/// 格式: SafeTensors (.safetensors)
/// 源: [HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3)
#[test]
fn e2e_reranker_safetensors() {
    const MODEL: &str = "BAAI/bge-reranker-v2-m3";

    let client = Client::new(MODEL, gllm::ModelKind::Reranker)
        .expect("Failed to load SafeTensors model");

    let response = client
        .rerank(
            "What is the capital of France?",
            &["Paris is the capital of France.", "London is in England.", "Berlin is in Germany."],
        )
        .generate()
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // 验证正确答案排名第一
    let top_result = &response.results[0];
    assert!(
        top_result.text.contains("Paris"),
        "Top result should mention Paris"
    );

    // 验证得分排序
    assert!(
        response.results[0].index < response.results[1].index,
        "First result should have better score"
    );
}

/// GGUF 格式的 Reranker 测试
///
/// 模型: gpustack/bge-reranker-v2-m3-GGUF
/// 格式: GGUF (.gguf)
/// 源: [HuggingFace](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)
#[test]
fn e2e_reranker_gguf() {
    const MODEL: &str = "gpustack/bge-reranker-v2-m3-GGUF";

    let client = Client::new(MODEL, gllm::ModelKind::Reranker)
        .expect("Failed to load GGUF model");

    let response = client
        .rerank(
            "What is the capital of France?",
            &["Paris is the capital.", "London is the capital of UK."],
        )
        .generate()
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 2, "Should have 2 results");

    // 验证正确答案排名第一
    let top_result = &response.results[0];
    assert!(
        top_result.text.contains("Paris"),
        "Top result should mention Paris"
    );
}

/// ONNX 格式的 Reranker 测试
///
/// 模型: onnx-community/bge-reranker-v2-m3-ONNX
/// 格式: ONNX (.onnx)
/// 源: [HuggingFace](https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX)
#[test]
fn e2e_reranker_onnx() {
    const MODEL: &str = "onnx-community/bge-reranker-v2-m3-ONNX";

    let client = Client::new(MODEL, gllm::ModelKind::Reranker)
        .expect("Failed to load ONNX model");

    let response = client
        .rerank(
            "What is the capital of China?",
            &["Beijing is the capital.", "Shanghai is a city.", "Tokyo is in Japan."],
        )
        .generate()
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // 验证正确答案排名第一
    let top_result = &response.results[0];
    assert!(
        top_result.text.contains("Beijing"),
        "Top result should mention Beijing"
    );
}
