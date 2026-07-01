//! E2E: c1 v2 DialogueGate intent tracker (B1, 方向B)
//!
//! Loads the real `v2_granite_best.pt` checkpoint (47 keys, 7-intent model)
//! and verifies: weight load + single-turn step + multi-turn state propagation
//! + output dimensions (intent[7] + difficulty[3]).
//!
//! Skips if the checkpoint is not present (CI without model cache).

use gllm::C1V2Tracker;

const PT_PATH: &str = "/tmp/c1_v2_extract/v2_granite_best.pt";

fn pt_available() -> bool {
    std::path::Path::new(PT_PATH).exists()
}

#[test]
fn c1_v2_loads_real_checkpoint() {
    if !pt_available() {
        eprintln!("skipped: {PT_PATH} not present");
        return;
    }
    let tracker = C1V2Tracker::from_pt(PT_PATH).expect("load .pt");
    assert_eq!(tracker.config().hidden_dim, 768);
    assert_eq!(tracker.config().num_cell_layers, 3);
    assert_eq!(tracker.config().num_intents, 7);
    assert_eq!(tracker.config().num_difficulty, 3);
}

#[test]
fn c1_v2_step_real_checkpoint_output_dims() {
    if !pt_available() {
        eprintln!("skipped: {PT_PATH} not present");
        return;
    }
    let tracker = C1V2Tracker::from_pt(PT_PATH).expect("load .pt");
    let embed = vec![0.1_f32; 768];
    let h_prev = tracker.initial_state();
    let r = tracker.step(&embed, &h_prev).expect("step ok");
    assert_eq!(r.intent_logits.len(), 7);
    assert_eq!(r.diff_logits.len(), 3);
    assert_eq!(r.h_next.len(), 3);
    assert_eq!(r.h_next[0].len(), 768);
    // Logits must be finite (no NaN/Inf from bad dtype inference).
    for v in &r.intent_logits {
        assert!(v.is_finite(), "non-finite intent logit: {v}");
    }
    for v in &r.diff_logits {
        assert!(v.is_finite(), "non-finite diff logit: {v}");
    }
    // State must move off the zero initial state.
    assert!(r.h_next[0].iter().any(|v| *v != 0.0), "state did not update");
}

#[test]
fn c1_v2_multi_turn_state_propagates_real_checkpoint() {
    if !pt_available() {
        eprintln!("skipped: {PT_PATH} not present");
        return;
    }
    let tracker = C1V2Tracker::from_pt(PT_PATH).expect("load .pt");
    let e1 = vec![0.2_f32; 768];
    let e2 = vec![0.5_f32; 768];
    let h0 = tracker.initial_state();
    let r1 = tracker.step(&e1, &h0).expect("turn1");
    let r2 = tracker.step(&e2, &r1.h_next).expect("turn2");
    // Different turn inputs → different intent logits (state is path-dependent).
    assert_ne!(r1.intent_logits, r2.intent_logits, "state not path-dependent");
    // Turn-2 state differs from turn-1 state.
    assert_ne!(r1.h_next[0], r2.h_next[0], "state did not evolve across turns");
}
