//! 第一性原理: 验证 N=4 时 weight_blob 大小 vs layer3 v_proj weight offset
//! 假设: blob 太小, layer3 v_proj weight 越界
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_blob_size() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/blob_size.txt").expect("f");
    // weight_layout offsets 的最大值 + 对应 tensor size = blob 大小估计
    let l0v = sp.named_offsets.iter().find(|(n,_,_)| n == "L0.v_proj").map(|(_,o,_)| *o).unwrap_or(0);
    let _ = writeln!(f, "L0.v_proj weight off = {}", l0v);
    let _ = writeln!(f, "weight_stride = 11379712 (已知)");
    let _ = writeln!(f, "layer3 v_proj weight = {} + 3*11379712 = {}", l0v, l0v + 3*11379712);
    // lm_head 是 global (在 layer 区之后)
    let lm = sp.named_offsets.iter().find(|(n,_,_)| n == "lm_head").map(|(_,o,_)| *o).unwrap_or(0);
    let _ = writeln!(f, "lm_head off = {} (global, 在 layer 区之后?)", lm);
    // 找所有 L0. 的最大 offset + 估算 size
    let mut max_end = 0usize;
    for (name, off, dtype) in &sp.named_offsets {
        if name.starts_with("L0.") {
            let _ = writeln!(f, "  {} off={} {:?}", name, off, dtype);
        }
    }
    let _ = f.flush();
    eprintln!("done /tmp/blob_size.txt");
}
