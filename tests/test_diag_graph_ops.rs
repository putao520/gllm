#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn diag_graph_ops() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let _sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}
