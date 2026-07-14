//! 第一性原理: 对比 N=3 vs N=4 的 named_offsets (tensor offset)
//! 假设: N=4 的 buffer 布局不同 (capture region 更大), tensor offset 偏移 → 越界/覆盖
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_offsets(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(outfile).expect("f");
    let _ = writeln!(f, "=== N={} named_offsets ({} tensors) data.len={} ===", n, sp.named_offsets.len(), sp.data.len());
    for (name, off, dtype) in &sp.named_offsets {
        let _ = writeln!(f, "  {:30} off={:>12} {:?}", name, off, dtype);
    }
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_offsets_n3_n4() {
    let _ = std::fs::write("/tmp/offsets_cmp.txt", "");
    dump_offsets(3, "/tmp/offsets_cmp.txt");
    dump_offsets(4, "/tmp/offsets_cmp.txt");
}
