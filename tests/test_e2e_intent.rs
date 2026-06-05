//! E2E 测试: Intent Recall SDK (REQ-INTENT-001..003)
//!
//! **SSOT**: `SPEC/INTENT.md`, `SPEC/01-REQUIREMENTS.md §15`
//!
//! 运行:
//! ```bash
//! cargo test --test test_e2e_intent -- --test-threads=1
//! ```

use gllm::{Client, LayerAnchor, PoolMode};

const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

/// TEST-INTENT-001 (REQ-INTENT-001): `encode_intent` 返回 hidden_size 维向量, 所有元素 finite.
#[test]
fn test_intent_001_basic_shape_and_finiteness() {
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct");
    let (hidden_size, num_layers) = client.model_dims().expect("model_dims");

    let enc = client
        .encode_intent("Hello world", LayerAnchor::Relative(0.5), PoolMode::MeanPool)
        .expect("encode_intent failed");

    assert_eq!(enc.dim(), hidden_size, "embedding dim must equal hidden_size");
    assert!(
        enc.embedding.iter().all(|v| v.is_finite()),
        "encode_intent returned non-finite values"
    );
    assert!(enc.l2_norm() > 0.0, "L2 norm must be > 0");
    assert_eq!(
        enc.pool, PoolMode::MeanPool,
        "pool mode must be preserved in IntentEncoding"
    );
    assert!(enc.actual_layer < num_layers);
}

/// TEST-INTENT-002 (REQ-INTENT-002): 不同 `PoolMode` 产生不同向量 (MeanPool vs LastToken vs ClsToken).
#[test]
fn test_intent_002_pool_mode_matters() {
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct");
    let text = "The quick brown fox jumps over the lazy dog";

    let mean = client
        .encode_intent(text, LayerAnchor::Relative(0.5), PoolMode::MeanPool)
        .expect("mean pool failed");
    let last = client
        .encode_intent(text, LayerAnchor::Relative(0.5), PoolMode::LastToken)
        .expect("last pool failed");
    let cls = client
        .encode_intent(text, LayerAnchor::Relative(0.5), PoolMode::ClsToken)
        .expect("cls pool failed");

    assert_eq!(mean.dim(), last.dim());
    assert_eq!(mean.dim(), cls.dim());

    // MeanPool != LastToken (unless text has seq_len=1, but our text is many tokens)
    let delta_mean_last: f32 = mean
        .embedding
        .iter()
        .zip(last.embedding.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        delta_mean_last > 1e-3,
        "MeanPool and LastToken vectors are identical — pooling had no effect"
    );

    // ClsToken != LastToken (first vs last position)
    let delta_cls_last: f32 = cls
        .embedding
        .iter()
        .zip(last.embedding.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        delta_cls_last > 1e-3,
        "ClsToken and LastToken vectors are identical — pooling had no effect"
    );
}

/// TEST-INTENT-003 (REQ-INTENT-003): `encode_intent` 和 `encode_to_layer`
/// 产生相同 embedding (delegation contract — intent wraps HR with zero code duplication).
#[test]
fn test_intent_003_delegates_to_encode_to_layer() {
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M-Instruct");
    let text = "hello";

    let emb_hr = client
        .encode_to_layer(text, LayerAnchor::Relative(0.75), PoolMode::MeanPool)
        .expect("encode_to_layer failed");
    let enc_intent = client
        .encode_intent(text, LayerAnchor::Relative(0.75), PoolMode::MeanPool)
        .expect("encode_intent failed");

    assert_eq!(emb_hr.len(), enc_intent.embedding.len());
    for (i, (a, b)) in emb_hr.iter().zip(enc_intent.embedding.iter()).enumerate() {
        // Deterministic forward — bit-exact.
        assert!(
            (a - b).abs() < 1e-5,
            "dim {i}: encode_to_layer={a}, encode_intent={b}"
        );
    }
}
