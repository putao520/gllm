//! 本地（AVX2 W256）Q4_K embedding dequant 测试 — 对照 5070Ti AVX-512 W512
#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn q4k_embed_local() {
    // Qwen3-0.6B Q4_K_M — Q4_K embedding, non-hetero, W256 on local AVX2
    let c = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q4_k_m")
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
    eprintln!("qwen3 q4_k_m logits min={} max={}", mn, mx);
    // argmax
    let am = lg.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap();
    eprintln!("argmax={}", am);
    eprintln!("DONE");
}
