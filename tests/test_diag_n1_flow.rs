#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn diag_n1_flow() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/n1_flow.txt").expect("f");
    for name in &["layer.normed","layer.v","layer.ffn_resid"] {
        if let Some(&(_, off, _)) = sp.named_offsets.iter().find(|(nm,_,_)| nm == name) {
            let vals = sp.read_dtype_aware(off, 8);
            let max = vals.iter().fold(0.0f32, |m,&v| if v.is_finite() {m.max(v.abs())} else {m});
            let nan = vals.iter().filter(|v| v.is_nan()).count();
            let _ = writeln!(f, "N=1 {:18} |max|={:.4} NaN={}/8", name, max, nan);
        }
    }
}
