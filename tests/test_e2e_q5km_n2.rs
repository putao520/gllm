//! N=2 截断 E2E: 验证 prefill 路径修复后 2层推理是否合理
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn e2e_q5km_n2_capital() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "2");
    let client = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat).gguf_file_filter("q5_k_m").build().expect("client");
    let response = client.generate("The capital of France is")
        .max_tokens(10).temperature(0.0).generate().response().expect("gen");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    eprintln!("[Q5_K_M N=2 E2E] output={:?}", response.text);
}
