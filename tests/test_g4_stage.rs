#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn g4_stage_crash() {
    eprintln!("STAGE1: start build");
    let client = Client::builder()
        .model("unsloth/gemma-4-E2B-it-GGUF")
        .kind(ModelKind::Chat)
        .build();
    eprintln!("STAGE2: build ok={}", client.is_ok());
    let client = client.unwrap_or_else(|e| panic!("build: {:?}", e));
    eprintln!("STAGE3: encode start");
    let toks = client.encode("The");
    eprintln!("STAGE4: encode ok={}", toks.is_ok());
    let toks = toks.unwrap_or_else(|e| panic!("encode: {:?}", e));
    eprintln!("STAGE5: prefill start tokens={:?}", toks);
    let logits = client.diagnostic_prefill_logits(&toks);
    eprintln!("STAGE6: logits present={}", logits.is_some());
    if let Some(l) = logits {
        let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let cnt30 = l.iter().filter(|v| (**v - 30.0).abs() < 1e-4).count();
        let nan = l.iter().filter(|v| v.is_nan()).count();
        eprintln!("STAGE7: logits len={} max={:.2} cnt30={} nan={}", l.len(), mx, cnt30, nan);
    }
    eprintln!("DONE");
}
