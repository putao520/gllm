//! 第一性原理: 纯 Q6K 模型 N=4 是否腐败? (对照 Q5_K_M 混合)
//! Q5_K_M: v_proj=Q6K, 其他=Q5K. 若纯 Q6K N=4 正常 -> 腐败是混合布局问题
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_pure_q6k(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    // 纯 Q6K: bartowski Qwen3-0.6B 有 Q6_K 量化
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q6_k").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(outfile).expect("f");
    let _ = writeln!(f, "=== 纯Q6K N={} (len={}) ===", n, sp.data.len());
    for name in &["layer.normed", "layer.q", "layer.k", "layer.v", "layer.o"] {
        if let Some(&(_, off, _)) = sp.named_offsets.iter().find(|(nm,_,_)| nm == name) {
            let vals = sp.read_dtype_aware(off, 8);
            let max = vals.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
            let nan = vals.iter().filter(|v| v.is_nan()).count();
            let _ = writeln!(f, "  {:16} off={:>12} |max|={:.4} NaN={}/8", name, off, max, nan);
        }
    }
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_q6k_pure_n3_n4() {
    let _ = std::fs::write("/tmp/q6k_pure.txt", "");
    dump_pure_q6k(3, "/tmp/q6k_pure.txt");
    dump_pure_q6k(4, "/tmp/q6k_pure.txt");
}
