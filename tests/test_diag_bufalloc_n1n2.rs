//! Dump N=1 vs N=2 的 buffer_alloc, 对比 scratchpad layout.
//! 假设: num_layers 影响 KV cache size → scratchpad layout 变 → ping/pong offset 变 → layer0 读错
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_buf(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    std::env::set_var("GLLM_DEBUG_BUFFER_ALLOC", "1");
    let _ = std::fs::remove_file("/tmp/gllm_bufalloc.log");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode("x").expect("encode");
    let _sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    std::env::remove_var("GLLM_DEBUG_BUFFER_ALLOC");
    let _ = std::fs::rename("/tmp/gllm_bufalloc.log", outfile);
}

#[test]
#[ignore]
fn diag_bufalloc_n1_n2() {
    eprintln!("=== Dump Q5_K_M buffer_alloc N=1 vs N=2 ===");
    dump_buf(1, "/tmp/buf_n1.log");
    dump_buf(2, "/tmp/buf_n2.log");
    eprintln!("N=1: /tmp/buf_n1.log, N=2: /tmp/buf_n2.log");
}
