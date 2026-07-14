//! 第一性原理: N=3 vs N=4, dump v_proj 的输入(normed) + 输出 + q/k 对比
//! v_proj 输出腐败 -> 查输入 normed 是否腐败, 权重是否对
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_flow(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(outfile).expect("f");
    let _ = writeln!(f, "=== N={} (scratchpad len={}) ===", n, sp.data.len());
    // dump 关键中间张量前 8 个值 + |max| + NaN count
    for name in &["layer.normed", "layer.q", "layer.k", "layer.v", "layer.o", "layer.gate", "layer.up", "layer.down"] {
        if let Some(&(_, off, _)) = sp.named_offsets.iter().find(|(nm,_,_)| nm == name) {
            let vals = sp.read_dtype_aware(off, 8);
            let max = vals.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
            let nan = vals.iter().filter(|v| v.is_nan()).count();
            let _ = writeln!(f, "  {:16} off={:>12} first8={:?} |max|={:.4} NaN={}/8", name, off, vals, max, nan);
        }
    }
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_v_flow_n3_n4() {
    let _ = std::fs::write("/tmp/v_flow.txt", "");
    dump_flow(3, "/tmp/v_flow.txt");
    dump_flow(4, "/tmp/v_flow.txt");
    dump_flow(5, "/tmp/v_flow.txt");
}
