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

    let documents = vec![
        "Paris is the capital of France.",
        "London is in England.",
        "Berlin is in Germany.",
    ];

    let response = client
        .rerank("What is the capital of France?", documents)
        .generate()
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // 验证得分排序（第一个文档应该排在最前面，因为它最相关）
    let top_result = &response.results[0];
    assert_eq!(top_result.index, 0, "First document (Paris) should be ranked first");
    assert!(top_result.score > 0.0, "Score should be positive");

    // 验证结果按得分降序排列
    for i in 1..response.results.len() {
        assert!(
            response.results[i - 1].score >= response.results[i].score,
            "Results should be sorted by score descending"
        );
    }
}

/// GGUF 格式的 Reranker 测试
///
/// 模型: gpustack/bge-reranker-v2-m3-GGUF
/// 格式: GGUF (.gguf)
/// 源: [HuggingFace](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)
///
/// 注意：此测试暂时被跳过，因为需要为该 GGUF 模型添加 manifest 配置。
#[test]
#[ignore = "需要为 GGUF reranker 模型添加配置"]
fn e2e_reranker_gguf() {
    const MODEL: &str = "gpustack/bge-reranker-v2-m3-GGUF";

    let client = Client::new(MODEL, gllm::ModelKind::Reranker)
        .expect("Failed to load GGUF model");

    let documents = vec![
        "Paris is the capital.",
        "London is the capital of UK.",
    ];

    let response = client
        .rerank("What is the capital of France?", documents)
        .generate()
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 2, "Should have 2 results");

    // 验证正确答案排名第一
    let top_result = &response.results[0];
    assert_eq!(top_result.index, 0, "First document (Paris) should be ranked first");
    assert!(top_result.score > 0.0, "Score should be positive");
}

/// ONNX 格式的 Reranker 测试
///
/// 模型: onnx-community/bge-reranker-v2-m3-ONNX
/// 格式: ONNX (.onnx)
/// 源: [HuggingFace](https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX)
///
/// 注意：此测试暂时被跳过，因为 ONNX reranker 模型暂不支持。
#[test]
#[ignore = "ONNX reranker 模型暂不支持"]
fn e2e_reranker_onnx() {
    const MODEL: &str = "onnx-community/bge-reranker-v2-m3-ONNX";

    let client = Client::new(MODEL, gllm::ModelKind::Reranker)
        .expect("Failed to load ONNX model");

    let documents = vec![
        "Beijing is the capital.",
        "Shanghai is a city.",
        "Tokyo is in Japan.",
    ];

    let response = client
        .rerank("What is the capital of China?", documents)
        .generate()
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // 验证正确答案排名第一
    let top_result = &response.results[0];
    assert_eq!(top_result.index, 0, "First document (Beijing) should be ranked first");
    assert!(top_result.score > 0.0, "Score should be positive");
}
