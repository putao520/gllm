#![cfg(test)]
use gllm::{Client, ModelKind};
fn dump_v(n: usize) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q6_k").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let _sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}
#[test]
#[ignore]
fn diag_n2_q6k() { dump_v(2); }
