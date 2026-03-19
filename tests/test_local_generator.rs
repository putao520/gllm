//! Decoder (生成模型) 本地管线测试
//!
//! 模型:
//! - SmolLM2-135M (LlamaForCausalLM, 135M params, BF16)
//! - GPT-2 (GPT2LMHeadModel, 124M params, F32)
//!
//! 运行: cargo test --test test_local_generator -- --ignored --test-threads=1

use gllm::Client;

// ─── SmolLM2-135M ───────────────────────────────────────────────────────────

/// TEST-GEN-LOCAL-001: SmolLM2-135M 加载+生成
#[test]
#[ignore]
fn test_smollm2_generate() {
    let client = Client::new_chat("test_models/smollm2-135m/safetensors")
        .expect("load smollm2");
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .expect("generate");

    let text = response.text.trim();
    assert!(!text.is_empty(), "output should not be empty");
    assert!(text.len() > 2, "output too short: {text:?}");
}

/// TEST-GEN-LOCAL-002: SmolLM2-135M 确定性输出
#[test]
#[ignore]
fn test_smollm2_deterministic() {
    let client = Client::new_chat("test_models/smollm2-135m/safetensors")
        .expect("load smollm2");
    let r1 = client
        .generate("Hello world")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .expect("gen1");
    let r2 = client
        .generate("Hello world")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .expect("gen2");
    assert_eq!(r1.text, r2.text, "deterministic output mismatch");
}

// ─── GPT-2 ──────────────────────────────────────────────────────────────────

/// TEST-GEN-LOCAL-003: GPT-2 加载+生成
#[test]
#[ignore]
fn test_gpt2_generate() {
    let client = Client::new_chat("test_models/gpt2/safetensors")
        .expect("load gpt2");
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .expect("generate");

    let text = response.text.trim();
    assert!(!text.is_empty(), "output should not be empty");
    assert!(text.len() > 2, "output too short: {text:?}");
}

/// TEST-GEN-LOCAL-004: GPT-2 确定性输出
#[test]
#[ignore]
fn test_gpt2_deterministic() {
    let client = Client::new_chat("test_models/gpt2/safetensors")
        .expect("load gpt2");
    let r1 = client
        .generate("Hello world")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .expect("gen1");
    let r2 = client
        .generate("Hello world")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .expect("gen2");
    assert_eq!(r1.text, r2.text, "deterministic output mismatch");
}
