//! Q5_K_M vs Q6_K 编译时 buffer 分配对比 — architect agent 决定性诊断.
//! 用 GLLM_DEBUG_BUFFER_ALLOC=1 dump slot 分配, 比较:
//!   1. ping/pong sentinel slot 的 offset/size 是否相同
//!   2. 是否有 intermediate tensor 落在 ping/pong 范围 (overlap = 根因)
//! 纯编译时结构对比, 不测运行时数值.
//! dump 行直接输出 stderr (不捕获, 避免 pipe 死锁), 人工/grep 分析.
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_dump_bufalloc_q5km() {
    eprintln!("\n========== [Q5_K_M N=1] buf-alloc dump START ==========");
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let c = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m")
        .build().expect("Q5_K_M client");
    let _ = c.encode("The capital of France is").expect("encode");
    drop(c);
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    eprintln!("========== [Q5_K_M N=1] buf-alloc dump END ==========\n");
}

#[test]
#[ignore]
fn diag_dump_bufalloc_q6k() {
    eprintln!("\n========== [Q6_K N=1] buf-alloc dump START ==========");
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let c = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q6_k")
        .build().expect("Q6_K client");
    let _ = c.encode("The capital of France is").expect("encode");
    drop(c);
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    eprintln!("========== [Q6_K N=1] buf-alloc dump END ==========\n");
}
