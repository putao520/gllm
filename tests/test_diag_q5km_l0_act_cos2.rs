//! 对比 Q5_K_M vs BF16 layer0 中间激活 (N=1) — 复用 client, 减少加载.
#![cfg(test)]
#![allow(dead_code)]
use gllm::{Client, ModelKind};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(a,b)| (*a as f64)*(*b as f64)).sum();
    let na: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 { dot/(na*nb) } else { 0.0 }
}

#[test]
#[ignore]
fn diag_l0_act_cos2() {
    eprintln!("\n=== Q5_K_M vs BF16 layer0 中间激活 cosine (N=1, 复用 client) ===");
    use std::io::Write; let _ = std::io::stderr().flush();
    let hidden = 1024;
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let prompt = "The capital of France is";

    let q5 = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("q5_k_m").build().expect("q5");
    let qt = q5.encode(prompt).expect("e");
    let qsp = q5.diagnostic_prefill_scratchpad(&qt).expect("sp");
    let bf = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("bf16").build().expect("bf");
    let bt = bf.encode(prompt).expect("e");
    let bsp = bf.diagnostic_prefill_scratchpad(&bt).expect("sp");

    for name in &["embed", "layer.q", "layer.k", "layer.v", "layer.attn", "layer.ffn_resid", "final_norm"] {
        let qoff = qsp.named_offsets.iter().find(|(n,_,_)| n == name).map(|(_,o,_)| *o);
        let boff = bsp.named_offsets.iter().find(|(n,_,_)| n == name).map(|(_,o,_)| *o);
        if let (Some(qo), Some(bo)) = (qoff, boff) {
            let q5v = qsp.read_dtype_aware(qo, hidden);
            let bfv = bsp.read_dtype_aware(bo, hidden);
            let cos = cosine(&q5v, &bfv);
            let qm = q5v.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
            let bm = bfv.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
            let mad = q5v.iter().zip(bfv.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
            eprintln!("[{:>16}] cos={:.6} |q5|={:.4} |bf|={:.4} mad={:.4} {}",
                name, cos, qm, bm, mad, if cos > 0.999 { "✓" } else if cos > 0.9 { "⚠️ 偏" } else { "✗ 崩" });
        } else {
            eprintln!("[{:>16}] offset 未找到 (q5={:?} bf={:?})", name, qoff, boff);
        }
    }
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}
