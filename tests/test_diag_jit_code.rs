//! Dump N=1 vs N=2 的最终 JIT 机器码, objdump 对比.
//! 用户要求"确认JIT出来的代码" — 算法正确(IR对称), JIT 机器码是否对称?
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_jit(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    std::env::set_var("GLLM_DUMP_JIT_CODE", outfile);
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode("The capital of France is").expect("encode");
    let _sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    std::env::remove_var("GLLM_DUMP_JIT_CODE");
}

#[test]
#[ignore]
fn diag_jit_code_n1_n2() {
    eprintln!("=== Dump Q5_K_M JIT machine code N=1 vs N=2 ===");
    dump_jit(1, "/tmp/jit_n1.bin");
    dump_jit(2, "/tmp/jit_n2.bin");
    eprintln!("N=1: /tmp/jit_n1.bin, N=2: /tmp/jit_n2.bin");
    eprintln!("objdump -D -b binary -m i386:x86-64 /tmp/jit_n1.bin > /tmp/jit_n1.asm");
}
