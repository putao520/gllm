//! 第一性原理: 对比 Q5_K_M vs 纯Q6K 的 weight_stride + 各权重 dtype + block size
//! 假设: Q5_K_M 混合布局 weight_stride 算错, 导致 layer3 v_proj(Q6K) 权重越界
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_layout(quant: &str, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter(quant).build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(outfile).expect("f");
    let _ = writeln!(f, "=== {} (混合? {}) ===", quant, quant.contains("q5"));
    // L0.* 是模板层的 per-tensor named offset (weight blob 内)
    for (name, off, dtype) in &sp.named_offsets {
        if name.starts_with("L0.") || name == "lm_head" {
            let _ = writeln!(f, "  {:20} off={:>12} {:?}", name, off, dtype);
        }
    }
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_stride_cmp() {
    let _ = std::fs::write("/tmp/stride_cmp.txt", "");
    dump_layout("q5_k_m", "/tmp/stride_cmp.txt");
    dump_layout("q6_k", "/tmp/stride_cmp.txt");
}
