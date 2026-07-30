//! Gemma4 逐层 overflow 二分：GLLM_TRUNCATE_LAYERS=N 跑各 N，读 final hidden（logits_scratch_offset 前）
//! GDB 下跑避免 SIGSEGV
#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn g4_bisect_overflow() {
    let n: usize = std::env::var("G4_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    if n > 0 { std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string()); }
    let client = Client::builder()
        .model("unsloth/gemma-4-E2B-it-GGUF")
        .kind(ModelKind::Chat)
        .build()
        .unwrap_or_else(|e| panic!("build: {:?}", e));
    let toks = client.encode("The").unwrap();
    eprintln!("layers={} tokens={:?}", n, toks);

    // 读 logits + hidden
    let logits = client.diagnostic_prefill_logits(&toks);
    if let Some(ref l) = logits {
        let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let cnt30 = l.iter().filter(|v| (**v - 30.0).abs() < 1e-4).count();
        eprintln!("logits max={:.2} cnt30={}", mx, cnt30);
    } else { eprintln!("logits: None"); }

    if let Some(sp) = client.diagnostic_prefill_scratchpad(&toks) {
        let lso = sp.logits_offset;
        // hidden 在 logits 之前。读 lso 前 256 个 f32（可能 final hidden）
        if lso >= 1024 {
            let hidden = sp.read_dtype_aware(lso - 1024, 16);
            let mx = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let bad = hidden.iter().filter(|v| v.is_infinite() || v.is_nan()).count();
            eprintln!("hidden[-1024] max={:.2} inf_nan={} first4={:?}", mx, bad, &hidden[..4]);
        }
        // 也读 layer output 区域（activation_alias ping/pong）
        for (name, off, _) in &sp.named_offsets {
            if name.contains("resid") || name == "hidden" || name.contains("layer_out") {
                let v = sp.read_dtype_aware(*off, 8);
                let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let bad = v.iter().filter(|x| x.is_infinite() || x.is_nan()).count();
                eprintln!("  [{}] off={} max={:.2} inf_nan={}", name, off, mx, bad);
            }
        }
    }
    eprintln!("DONE_N{}", n);
}
