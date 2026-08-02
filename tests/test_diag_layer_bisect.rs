//! 逐层二分定位 W512 vs W256 发散起点。
//!
//! This diagnostic intentionally uses the public `Client::encode_to_layer` path
//! so each run exercises the same real-loop early-exit path on the target CPU.
//! Run once on the AVX2 host and once on the AVX-512 host, then compare the
//! `[LAYER-BISECT]` fingerprints.

#![cfg(target_os = "linux")]

use gllm::head_routing::{LayerAnchor, PoolMode};
use gllm::{BackendType, Client, ModelKind};

const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
const PROMPT: &str = "The meaning of life is";
const LAYERS: &[usize] = &[0, 1, 2, 3, 5, 10, 15, 20, 29];

fn rms(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

#[test]
fn diag_layer_bisect() {
    let client = Client::builder()
        .model(MODEL)
        .kind(ModelKind::Chat)
        .backend(BackendType::Cpu)
        .build()
        .expect("CPU client build");

    for &layer in LAYERS {
        // `encode_at_layer_for_prompt` is an Executor method. The public Client
        // wrapper resolves the same absolute anchor and delegates to it without
        // introducing a fallback or a second execution path.
        // M=1 decode/prefill execution writes row 0 only; LastToken would select
        // row 4 for this five-token prompt and therefore returns the intentional
        // zero row. ClsToken selects row 0 without changing the forward path.
        let hidden = client
            .encode_to_layer(PROMPT, LayerAnchor::Absolute(layer), PoolMode::ClsToken)
            .unwrap_or_else(|error| panic!("encode_to_layer({layer}) failed: {error}"));
        let first8: Vec<f32> = hidden.iter().take(8).copied().collect();
        eprintln!(
            "[LAYER-BISECT] layer={} len={} rms={:.6} first8={:?}",
            layer,
            hidden.len(),
            rms(&hidden),
            first8
        );
    }
}
