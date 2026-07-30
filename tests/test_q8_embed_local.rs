//! 本地（W256 AVX2）Q8_0 embedding 对照 — Gemma4 用 Q8_0
#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn q8_embed_local() {
    // Qwen3-0.6B Q8_0 — Q8_0 embedding (与 Gemma4 同 dtype, 非 hetero, W256)
    let c = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q8_0")
        .build()
        .unwrap_or_else(|e| panic!("load: {:?}", e));
    let t = c.encode("Hello").unwrap();
    eprintln!("tokens: {:?}", t);
    if let Some(sp) = c.diagnostic_prefill_scratchpad(&t) {
        let nan = sp.find_nan_tensors();
        eprintln!("nan count={}", nan.len());
        for h in &nan { eprintln!("  {}", h.name); }
    }
    let lg = c.diagnostic_prefill_logits(&t).expect("prefill");
    let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mn = lg.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!("qwen3 q8_0 logits min={} max={}", mn, mx);
    let am = lg.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap();
    eprintln!("argmax={}", am);
    eprintln!("DONE");
}
