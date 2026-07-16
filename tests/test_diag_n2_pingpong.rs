#![cfg(test)]
use gllm::{Client, ModelKind};
fn dump(n: usize) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/n2pp.txt").expect("f");
    // ping at offset 0, pong at offset 167772160
    for (name, off) in [("ping(0)", 0usize), ("pong(167M)", 167772160usize)] {
        let vals = sp.read_dtype_aware(off, 8);
        let max = vals.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
        let nan = vals.iter().filter(|v| v.is_nan()).count();
        let _ = writeln!(f, "[N={}] {:12} off={} |max|={:.4} NaN={}/8 first8={:?}", n, name, off, max, nan, vals);
    }
    let _ = f.flush();
}
#[test]
#[ignore]
fn diag_n2_pp() { let _ = std::fs::write("/tmp/n2pp.txt",""); dump(1); dump(2); dump(3); }
