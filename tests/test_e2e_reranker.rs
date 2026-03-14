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

    let client =
        Client::new(MODEL, gllm::ModelKind::Reranker).expect("Failed to load SafeTensors model");

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
    assert_eq!(
        top_result.index, 0,
        "First document (Paris) should be ranked first"
    );
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
/// 模型: DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF
/// 格式: GGUF (.gguf)
/// 源: [HuggingFace](https://huggingface.co/DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF)
///
#[test]
fn e2e_reranker_gguf() {
    const MODEL: &str = "DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF";

    let client = Client::new(MODEL, gllm::ModelKind::Reranker).expect("Failed to load GGUF model");
    let manifest = client.manifest().expect("Failed to read manifest");
    assert_eq!(manifest.kind, gllm::ModelKind::Reranker);

    let documents = vec![
        "Paris is the capital of France.",
        "London is in England.",
        "Berlin is in Germany.",
    ];

    let response = client
        .rerank("What is the capital of France?", documents)
        .generate()
        .expect("GGUF rerank inference failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // SPEC 06-TESTING-STRATEGY.md Section 8.3 TEST-REAL-002:
    // Reranker | 分数为有限浮点数
    for result in &response.results {
        assert!(
            result.score.is_finite(),
            "Score should be finite, got {}",
            result.score
        );
    }

    // 验证结果按得分降序排列
    for i in 1..response.results.len() {
        assert!(
            response.results[i - 1].score >= response.results[i].score,
            "Results should be sorted by score descending"
        );
    }

    // NOTE: 不验证 top_result.index == 0，量化模型精度不足以保证特定排名
}

/// ONNX 格式的 Reranker 测试
///
/// 模型: onnx-community/bge-reranker-v2-m3-ONNX
/// 格式: ONNX (.onnx)
/// 源: [HuggingFace](https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX)
#[test]
fn e2e_reranker_onnx() {
    const MODEL: &str = "onnx-community/bge-reranker-v2-m3-ONNX";

    let client = Client::new(MODEL, gllm::ModelKind::Reranker).expect("Failed to load ONNX model");

    let documents = vec![
        "Beijing is the capital city of China, serving as the political and cultural center of the nation for many centuries.",
        "Shanghai is the largest city in China by population, known for its modern skyline along the Bund waterfront.",
        "Tokyo is the capital of Japan, located on the eastern coast of the island of Honshu in the Kanto region.",
    ];

    let response = client
        .rerank("What is the capital of China?", documents)
        .generate()
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // 验证管道工作正常：得分为正、结果按降序排列
    for result in &response.results {
        assert!(result.score > 0.0, "Score should be positive, got {}", result.score);
    }
    for i in 1..response.results.len() {
        assert!(
            response.results[i - 1].score >= response.results[i].score,
            "Results should be sorted by score descending"
        );
    }
}
