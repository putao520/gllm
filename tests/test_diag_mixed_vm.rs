#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn diag_mixed_vm() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "3");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t);
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/mixed_vm.txt").expect("f");
    match sp {
        Some(sp) => { let _ = writeln!(f, "OK len={}", sp.data.len()); }
        None => { let _ = writeln!(f, "None (SIGSEGV before return)"); }
    }
    let _ = f.flush();
}
