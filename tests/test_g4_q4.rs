#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn g4_q4_overflow() {
    let client = Client::builder()
        .model("unsloth/gemma-4-E2B-it-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("Q4_0")
        .build()
        .unwrap_or_else(|e| panic!("load: {:?}", e));
    let toks = client.encode("The").unwrap();
    eprintln!("tokens={:?}", toks);
    let logits = client.diagnostic_prefill_logits(&toks);
    if let Some(ref l) = logits {
        let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let cnt30 = l.iter().filter(|v| (**v - 30.0).abs() < 1e-4).count();
        let nan = l.iter().filter(|v| v.is_nan()).count();
        eprintln!("logits max={:.2} cnt30={} nan={}", mx, cnt30, nan);
    } else { eprintln!("logits None"); }
    if let Some(sp) = client.diagnostic_prefill_scratchpad(&toks) {
        for (name, off, _) in &sp.named_offsets {
            if name == "embed" || name.contains("resid") || name == "L0.input_norm" {
                let v = sp.read_dtype_aware(*off, 8);
                let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let bad = v.iter().filter(|x| x.is_infinite() || x.is_nan()).count();
                eprintln!("[{}] off={} max={:.4} bad={}", name, off, mx, bad);
            }
        }
    }
    eprintln!("DONE");
}
