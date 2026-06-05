//! E2E 测试: Classifier (文本分类模型)
//!
//! 验证 classifier head 输出合理性
//!
//! 反作弊检查: NaN/Inf、全零、全同值、logits 方差退化

use gllm::Client;

// ============================================================================
// Anti-cheating helpers
// ============================================================================

/// 检测退化输出: NaN/Inf、全零、全同值
fn assert_logits_sane(logits: &[f32], label: &str) {
    assert!(!logits.is_empty(), "{label}: logits is empty");

    // 1. 禁止 NaN / Inf
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "{label}: NaN/Inf at index {i}: {v}");
    }

    // 2. 禁止全零
    let all_zero = logits.iter().all(|&v| v == 0.0);
    assert!(!all_zero, "{label}: all zeros — model output is degenerate");

    // 3. 禁止全同值 (logits should have some variance)
    let first = logits[0];
    let all_same = logits.iter().all(|&v| (v - first).abs() < 1e-8);
    assert!(
        !all_same,
        "{label}: all same value ({first}) — logits are degenerate"
    );
}

// ============================================================================
// SafeTensors — encoder-based classifier
// ============================================================================

/// TEST-E2E-CLS-001: SafeTensors classifier model E2E test
///
/// Uses a small BERT-based sentiment classifier model.
/// The model has a classifier head with num_labels=2 (positive/negative).
///
/// **测试类型**: 正向
/// **期望结果**: 成功加载模型并输出分类结果
#[test]
fn e2e_classifier_safetensors() {
    // nlptown/bert-base-multilingual-uncased-sentiment has 5 labels (1-5 stars)
    // but is large. Use a smaller model if available.
    // For now, test with an XLM-R encoder model loaded as classifier.
    // If no dedicated classifier model is small enough, we test with a
    // reranker model loaded as classifier (it has a classifier.weight).
    const MODEL: &str = "BAAI/bge-reranker-v2-m3";

    let client = Client::new_classifier(MODEL).expect("Failed to load classifier model");
    let manifest = client.manifest().expect("Failed to read manifest");
    assert_eq!(manifest.kind, gllm::ModelKind::Classifier);

    let result = client
        .classify(["This is a great product!", "Terrible experience, would not recommend."])
        .expect("Classification failed");

    assert_eq!(result.predictions.len(), 2, "Should have 2 predictions");

    for (i, pred) in result.predictions.iter().enumerate() {
        assert_eq!(pred.index, i, "Prediction index should match");
        assert!(!pred.logits.is_empty(), "Logits should not be empty");
        assert_logits_sane(&pred.logits, &format!("cls[{i}]"));

        // Score should be a valid probability
        assert!(
            pred.score >= 0.0 && pred.score <= 1.0,
            "Score {} should be a probability in [0, 1]",
            pred.score
        );

        eprintln!(
            "[CLS-{i}] label_id={} score={:.4} logits={:?}",
            pred.label_id, pred.score, pred.logits
        );
    }
}

// ============================================================================
// API surface test — ModelKind::Classifier round-trip
// ============================================================================

#[test]
fn e2e_classifier_model_kind_parse() {
    use gllm::ModelKind;

    assert_eq!(ModelKind::parse("classifier"), Some(ModelKind::Classifier));
    assert_eq!(ModelKind::parse("classification"), Some(ModelKind::Classifier));
    assert_eq!(ModelKind::parse("classify"), Some(ModelKind::Classifier));
    assert_eq!(ModelKind::parse("sequence-classification"), Some(ModelKind::Classifier));
    assert_eq!(ModelKind::parse("text-classification"), Some(ModelKind::Classifier));
}
