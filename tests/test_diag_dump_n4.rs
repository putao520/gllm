//! Dump N=4 VmProgram + buffer_alloc, 对比 N=3 找 layer3 NaN 根因
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_dump_n4() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    std::env::set_var("GLLM_DUMP_MEGA", "/tmp/n4_vm");
    std::env::set_var("GLLM_DEBUG_BUFFER_ALLOC", "1");
    let _ = std::fs::remove_file("/tmp/n4_buf.log");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode("x").expect("encode");
    let _sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    std::env::remove_var("GLLM_DUMP_MEGA");
    std::env::remove_var("GLLM_DEBUG_BUFFER_ALLOC");
    eprintln!("N=4 VmProgram dumped to /tmp/n4_vm/mega_kernel_vm.txt");
}
