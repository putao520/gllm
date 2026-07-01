//! E2E: Qwen3Guard-Stream per-token moderation head (B2, 方向B)
//!
//! Loads the guard head from the real Qwen3Guard-Stream-0.6B safetensors
//! (8 head tensors, BF16) and verifies: head load + per-token moderate
//! (4 logit groups: risk 3 + category 8 + query_risk 3 + query_category 9)
//! + sequence moderation + finite logits.
//!
//! The backbone Qwen3 forward is covered by existing qwen3 E2E; this test
//! isolates the guard head numerics on synthetic hidden states.
//! Skips if the checkpoint is not present.

use gllm::Qwen3GuardHead;

const ST_PATH: &str = "/home/putao/.cache/huggingface/hub/models--Qwen--Qwen3Guard-Stream-0.6B/snapshots/419364a715de9840d47b1457982f64ff37f90ed4/model.safetensors";

fn st_available() -> bool {
    std::path::Path::new(ST_PATH).exists()
}

#[test]
fn qwen3_guard_loads_real_head() {
    if !st_available() {
        eprintln!("skipped: {ST_PATH} not present");
        return;
    }
    let head = Qwen3GuardHead::from_safetensors(ST_PATH).expect("load head");
    assert_eq!(head.config().hidden_size, 1024);
    assert_eq!(head.config().guard_inner_size, 512);
    assert_eq!(head.config().num_risk_level, 3);
    assert_eq!(head.config().num_category, 8);
    assert_eq!(head.config().num_query_risk_level, 3);
    assert_eq!(head.config().num_query_category, 9);
}

#[test]
fn qwen3_guard_moderate_token_real_head() {
    if !st_available() {
        eprintln!("skipped: {ST_PATH} not present");
        return;
    }
    let head = Qwen3GuardHead::from_safetensors(ST_PATH).expect("load head");
    // Synthetic 1024-dim hidden state (a real run would feed the backbone's
    // last-layer hidden; here we verify head numerics + finite output).
    let hidden: Vec<f32> = (0..1024).map(|i| ((i as f32) % 7.0 - 3.0) * 0.05).collect();
    let r = head.moderate_token(&hidden).expect("moderate ok");
    assert_eq!(r.risk_level_logits.len(), 3);
    assert_eq!(r.category_logits.len(), 8);
    assert_eq!(r.query_risk_level_logits.len(), 3);
    assert_eq!(r.query_category_logits.len(), 9);
    assert_eq!(r.total_logits(), 23);
    for v in r.risk_level_logits.iter()
        .chain(&r.category_logits)
        .chain(&r.query_risk_level_logits)
        .chain(&r.query_category_logits)
    {
        assert!(v.is_finite(), "non-finite guard logit: {v}");
    }
}

#[test]
fn qwen3_guard_moderate_sequence_real_head() {
    if !st_available() {
        eprintln!("skipped: {ST_PATH} not present");
        return;
    }
    let head = Qwen3GuardHead::from_safetensors(ST_PATH).expect("load head");
    // 4 tokens × 1024 hidden
    let seq: Vec<f32> = (0..(4 * 1024)).map(|i| ((i as f32) % 11.0 - 5.0) * 0.02).collect();
    let results = head.moderate_sequence(&seq).expect("seq ok");
    assert_eq!(results.len(), 4);
    assert_eq!(results[0].total_logits(), 23);
    // Different token hidden states → different risk logits.
    assert_ne!(results[0].risk_level_logits, results[1].risk_level_logits);
}
