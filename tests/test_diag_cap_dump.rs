//! 第一性原理: dump 每层 capture 输出, 定位腐败起始层
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_cap(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    let stride = c.diagnostic_layer_capture_stride();
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(outfile).expect("f");
    let cap_off = sp.named_offsets.iter().find(|(nm,_,_)| nm == "layer_capture").map(|(_,o,_)| *o).unwrap_or(0);
    let _ = writeln!(f, "=== N={} cap_off={} stride={} ===", n, cap_off, stride);
    if stride == 0 || cap_off == 0 {
        let _ = f.flush();
        return;
    }
    // dump 每层前 16 值 + |max| + NaN count
    for layer in 0..n {
        let off = cap_off + layer * stride;
        let vals = sp.read_dtype_aware(off, 16);
        let max = vals.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
        let nan = vals.iter().filter(|v| v.is_nan()).count();
        let _ = writeln!(f, "  layer{} off={} first8={:?} |max|={:.4} NaN={}/16", layer, off, &vals[..8], max, nan);
    }
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_cap_dump_n3_n4_n5() {
    let _ = std::fs::write("/tmp/cap_dump.txt", "");
    dump_cap(3, "/tmp/cap_dump.txt");
    dump_cap(4, "/tmp/cap_dump.txt");
    dump_cap(5, "/tmp/cap_dump.txt");
}
