#![cfg(test)]
use gllm::{Client, ModelKind};
fn dump(n: usize, model: &str, label: &str, f: &mut std::fs::File) {
    use std::io::Write;
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter(model).build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    let _ = writeln!(f, "=== {} N={} ===", label, n);
    for name in &["layer.normed","layer.q","layer.k","layer.v","layer.o","layer.post_attn_norm","layer.gate","layer.up","layer.down","layer.ffn_resid"] {
        if let Some(&(_, off, _)) = sp.named_offsets.iter().find(|(nm,_,_)| nm == name) {
            let vals = sp.read_dtype_aware(off, 8);
            let max = vals.iter().fold(0.0f32, |m,&v| if v.is_finite() {m.max(v.abs())} else {m});
            let nan = vals.iter().filter(|v| v.is_nan()).count();
            let _ = writeln!(f, "  {:18} |max|={:.4} NaN={}/8", name, max, nan);
        }
    }
}
#[test]
#[ignore]
fn diag_n2_flow() {
    let mut f = std::fs::File::create("/tmp/n2_flow.txt").expect("f");
    dump(2, "q5_k_m", "Q5_K_M", &mut f);
    dump(2, "q6_k", "Q6_K", &mut f);
}
