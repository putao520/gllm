//! Dump N=1 vs N=2 的 VmProgram, 对比 Q5_K_M.
//! 根因假设: num_layers 编译时常量 → N=1/N=2 不同 JIT 代码 → regalloc 差异
//! 若 VmProgram 仅 num_layers 不同 → 问题在 regalloc/lowering; 若结构不同 → plan_lower
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_vm(label: &str, filter: &str, n: usize, outdir: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    std::env::set_var("GLLM_DUMP_MEGA", outdir);
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter(filter).build().expect("client");
    let t = c.encode("The capital of France is").expect("encode");
    let _sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    std::env::remove_var("GLLM_DUMP_MEGA");
    // 重命名 dump 文件
    let src = format!("{}/mega_kernel_vm.txt", outdir);
    let dst = format!("{}/vm_{}_N{}.txt", outdir, label, n);
    let _ = std::fs::rename(&src, &dst);
    eprintln!("[{}] N={} dumped to {}", label, n, dst);
}

#[test]
#[ignore]
fn diag_dump_vm_n1_n2() {
    let outdir = "/tmp/q5km_vm_dump";
    let _ = std::fs::remove_dir_all(outdir);
    let _ = std::fs::create_dir_all(outdir);
    eprintln!("=== Dump VmProgram N=1 vs N=2 vs N=3 ===");
    dump_vm("q5km", "q5_k_m", 1, outdir);
    dump_vm("q5km", "q5_k_m", 2, outdir);
    dump_vm("q5km", "q5_k_m", 3, outdir);
    eprintln!("=== 对比 layer loop body 大小 N=1 vs N=2 vs N=3 ===");
}
