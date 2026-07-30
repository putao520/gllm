#![cfg(test)]
use gllm::{Client, ModelKind};
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(a,b)| (*a as f64)*(*b as f64)).sum();
    let na: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 { dot/(na*nb) } else { 0.0 }
}
#[test]
#[ignore]
fn diag_q6k_perlayer() {
    let n = std::env::var("GLLM_TRUNCATE_LAYERS").unwrap_or_else(|_| "3".into());
    std::env::set_var("GLLM_TRUNCATE_LAYERS", &n);
    let prompt = "The capital of France is";
    let hidden = 1024;
    let q = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("q6_k").build().expect("q");
    let qt = q.encode(prompt).expect("e");
    let qsp = q.diagnostic_prefill_scratchpad(&qt).expect("sp");
    let qcap = qsp.named_offsets.iter().find(|(n,_,_)| n == "layer_capture").map(|(_,o,_)| *o).expect("cap");
    let qstride = q.diagnostic_layer_capture_stride();
    drop(q);
    let bf = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("bf16").build().expect("bf");
    let bt = bf.encode(prompt).expect("e");
    let bsp = bf.diagnostic_prefill_scratchpad(&bt).expect("sp");
    let bcap = bsp.named_offsets.iter().find(|(n,_,_)| n == "layer_capture").map(|(_,o,_)| *o).expect("capbf");
    let bstride = bf.diagnostic_layer_capture_stride();
    drop(bf);
    let n_layers: usize = n.parse().unwrap_or(3);
    let seq = qt.len();
    for li in 0..n_layers {
        let qv = qsp.read_dtype_aware(qcap + li*qstride, seq*hidden);
        let bv = bsp.read_dtype_aware(bcap + li*bstride, seq*hidden);
        let qm = qv.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
        eprintln!("  Q6_K layer[{}] cos={:.6} |q6|={:.4}", li, cosine(&qv,&bv), qm);
    }
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}
