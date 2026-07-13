//! 完整 E2E 验证: Q5_K_M 28层全模型 推理 "The capital of France is"
//! 修复前: 乱码 ('遇浚lar菊花...'). 修复后 (方案C): 应输出 Paris 相关.
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn e2e_q5km_capital_france() {
    let client = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat).gguf_file_filter("q5_k_m").build().expect("client");
    let response = client.generate("The capital of France is")
        .max_tokens(10).temperature(0.0).generate().response().expect("gen");
    let text = response.text.trim();
    let lower = text.to_lowercase();
    eprintln!("[Q5_K_M E2E] output={:?}", text);
    eprintln!("[Q5_K_M E2E] contains paris/capital: {}", lower.contains("paris") || lower.contains("capital"));
}
