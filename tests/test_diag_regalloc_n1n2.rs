//! Dump N=1 vs N=2 regalloc 分配, 对比跨 native call (Q5KDecodeStep/Q6KDecodeStep) 的 VReg.
//! 根因假设: num_layers=2 时 regalloc 把跨 call 的 live VReg 分配到 caller-saved
//! → 被 native call clobber → layer0 输出丢失
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_ra(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    std::env::set_var("GLLM_REGALLOC_DEBUG", "1");
    let _ = std::fs::remove_file("/tmp/gllm_regalloc.log");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode("The capital of France is").expect("encode");
    let _sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    std::env::remove_var("GLLM_REGALLOC_DEBUG");
    let _ = std::fs::rename("/tmp/gllm_regalloc.log", outfile);
}

#[test]
#[ignore]
fn diag_regalloc_n1_n2() {
    let _ = std::fs::remove_file("/tmp/ra_n1.log");
    let _ = std::fs::remove_file("/tmp/ra_n2.log");
    eprintln!("=== Dump Q5_K_M regalloc N=2 vs N=4 (N=4 NaN 调查) ===");
    dump_ra(2, "/tmp/ra_n2.log");
    dump_ra(4, "/tmp/ra_n4.log");
    eprintln!("N=2: /tmp/ra_n2.log, N=4: /tmp/ra_n4.log");
    eprintln!("对比: spills 数量, 跨 call VReg 分配 (caller-saved vs callee-saved)");
}
