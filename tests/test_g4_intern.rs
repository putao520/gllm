#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn g4_intern_vals() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let client = Client::builder().model("unsloth/gemma-4-E2B-it-GGUF").kind(ModelKind::Chat).build().unwrap();
    let toks = client.encode("The").unwrap();
    if let Some(sp) = client.diagnostic_prefill_scratchpad(&toks) {
        eprintln!("logits_offset={} len={}", sp.logits_offset, sp.data.len());
        for (name, off, dt) in &sp.named_offsets {
            let v = sp.read_dtype_aware(*off, 8);
            let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let bad = v.iter().filter(|x| x.is_infinite() || x.is_nan()).count();
            eprintln!("  [{}] off={} dt={:?} max={:.4} bad={}", name, off, dt, mx, bad);
        }
    }
    eprintln!("DONE");
}
