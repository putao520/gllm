//! Reranker 本地管线测试：3 种格式 (SafeTensors / ONNX / PyTorch)
//!
//! 模型: cross-encoder/ms-marco-MiniLM-L-6-v2 (BertForSequenceClassification, 22M params)
//! 运行: cargo test --test test_local_reranker -- --ignored --test-threads=1

use gllm::{Client, ModelKind};
use gllm::rerank::RerankResult;

const QUERY: &str = "What is the capital of France?";

fn documents() -> Vec<&'static str> {
    vec![
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is located in Paris.",
    ]
}

/// 验证 rerank 结果的合理性
fn assert_rerank_sane(results: &[RerankResult], label: &str) {
    assert_eq!(results.len(), 3, "{label}: expected 3 results");

    // 所有分数必须是有限数
    for r in results {
        assert!(r.score.is_finite(), "{label}: NaN/Inf score at index {}", r.index);
    }

    // 结果按分数降序排列
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "{label}: results not sorted descending at position {i}"
        );
    }

    // Paris 文档 (index 0) 应该排在最前面
    assert_eq!(
        results[0].index, 0,
        "{label}: expected Paris doc (index 0) ranked first, got index {}",
        results[0].index
    );

    // 最高分应该高于最低分（模型能区分相关性）
    let spread = results[0].score - results[results.len() - 1].score;
    assert!(
        spread > 0.001,
        "{label}: score spread {spread} too small — model not discriminating"
    );
}

fn run_reranker(path: &str, label: &str) {
    let client = Client::new(path, ModelKind::Reranker)
        .unwrap_or_else(|e| panic!("{label}: load failed: {e}"));

    let response = client
        .rerank(QUERY, documents())
        .top_n(3)
        .generate()
        .unwrap_or_else(|e| panic!("{label}: rerank failed: {e}"));

    assert_rerank_sane(&response.results, label);
}

/// TEST-RERANK-LOCAL-001: SafeTensors 格式 reranker
#[test]
#[ignore]
fn test_reranker_safetensors() {
    run_reranker("test_models/reranker/safetensors", "reranker-safetensors");
}

/// TEST-RERANK-LOCAL-002: ONNX 格式 reranker
#[test]
#[ignore]
fn test_reranker_onnx() {
    run_reranker("test_models/reranker/onnx", "reranker-onnx");
}

/// TEST-RERANK-LOCAL-003: PyTorch 格式 reranker
#[test]
#[ignore]
fn test_reranker_pytorch() {
    run_reranker("test_models/reranker/pytorch", "reranker-pytorch");
}

/// TEST-RERANK-LOCAL-004: 跨格式一致性 (SafeTensors vs ONNX)
///
/// 同一模型不同格式的 rerank 排序结果应该一致
#[test]
#[ignore]
fn test_reranker_cross_format_consistency() {
    let st_client = Client::new("test_models/reranker/safetensors", ModelKind::Reranker)
        .expect("load safetensors");
    let onnx_client = Client::new("test_models/reranker/onnx", ModelKind::Reranker)
        .expect("load onnx");

    let st_results = st_client
        .rerank(QUERY, documents())
        .top_n(3)
        .generate()
        .expect("st rerank");
    let onnx_results = onnx_client
        .rerank(QUERY, documents())
        .top_n(3)
        .generate()
        .expect("onnx rerank");

    // 排序顺序必须一致
    for i in 0..3 {
        assert_eq!(
            st_results.results[i].index, onnx_results.results[i].index,
            "rank {i}: SafeTensors index {} != ONNX index {}",
            st_results.results[i].index, onnx_results.results[i].index
        );
    }

    // 分数差异应该很小（同一模型不同格式）
    for i in 0..3 {
        let diff = (st_results.results[i].score - onnx_results.results[i].score).abs();
        assert!(
            diff < 0.5,
            "rank {i}: score diff {diff} too large (st={}, onnx={})",
            st_results.results[i].score, onnx_results.results[i].score
        );
    }
}

/// TEST-RERANK-LOCAL-005: 确定性输出
///
/// 同一输入多次 rerank 结果应该完全一致
#[test]
#[ignore]
fn test_reranker_deterministic() {
    let client = Client::new("test_models/reranker/safetensors", ModelKind::Reranker)
        .expect("load");

    let r1 = client
        .rerank(QUERY, documents())
        .top_n(3)
        .generate()
        .expect("rerank 1");
    let r2 = client
        .rerank(QUERY, documents())
        .top_n(3)
        .generate()
        .expect("rerank 2");

    for i in 0..3 {
        assert_eq!(r1.results[i].index, r2.results[i].index, "rank {i} index mismatch");
        assert_eq!(r1.results[i].score, r2.results[i].score, "rank {i} score mismatch");
    }
}
