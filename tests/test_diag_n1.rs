#![cfg(test)]
use gllm::{Client, ModelKind};
fn dump_v(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(outfile).expect("f");
    for name in &["layer.normed", "layer.v"] {
        if let Some(&(_, off, _)) = sp.named_offsets.iter().find(|(nm,_,_)| nm == name) {
            let vals = sp.read_dtype_aware(off, 8);
            let max = vals.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
            let nan = vals.iter().filter(|v| v.is_nan()).count();
            let _ = writeln!(f, "[N={}] {:16} |max|={:.4} NaN={}/8 first8={:?}", n, name, max, nan, vals);
        }
    }
    let _ = f.flush();
}
#[test]
#[ignore]
fn diag_n1() { let _ = std::fs::write("/tmp/n1.txt",""); dump_v(1, "/tmp/n1.txt"); }
