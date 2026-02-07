//! E2E 测试: Generator (生成模型)
//!
//! 测试 3 种格式: SafeTensors, GGUF, ONNX
//! 验证同一功能的模型在不同格式下都能正常工作

use gllm::Client;

/// SafeTensors 格式的 Generator 测试
///
/// 模型: HuggingFaceTB/SmolLM2-135M-Instruct
/// 格式: SafeTensors (.safetensors)
/// 源: [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
#[test]
fn e2e_generator_safetensors() {
    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

    let client = Client::new_chat(MODEL).expect("Failed to load SafeTensors model");
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .expect("Generation failed");

    let text = response.text.trim();
    assert!(!text.is_empty(), "Output should not be empty");
    assert!(text.len() > 3, "Output should be at least 4 characters");

    // 验证输出包含合理内容
    let lower = text.to_lowercase();
    let is_reasonable =
        lower.contains("paris") || lower.contains("parís") || lower.contains("capital");
    assert!(is_reasonable, "Output should contain reasonable answer");
}

/// GGUF 格式的 Generator 测试
///
/// 模型: Qwen/Qwen3-0.6B-GGUF
/// 格式: GGUF (.gguf)
/// 源: [HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)
///
#[test]
fn e2e_generator_gguf() {
    const MODEL: &str = "Qwen/Qwen3-0.6B-GGUF";

    let client = Client::new_chat(MODEL).expect("Failed to load GGUF model");
    let manifest = client.manifest().expect("Failed to read manifest");
    assert_eq!(manifest.kind, gllm::ModelKind::Chat);

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .expect("Generation failed");

    let text = response.text.trim();
    assert!(!text.is_empty(), "Output should not be empty");
    assert!(text.len() > 3, "Output should be at least 4 characters");

    // 验证输出包含合理内容（兼容中英文输出）
    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("parís")
        || lower.contains("capital")
        || lower.contains("france")
        || text.contains("巴黎");
    assert!(is_reasonable, "Output should contain reasonable answer");
}

/// ONNX 格式的 Generator 测试
///
/// 模型: onnx-community/SmolLM2-135M-ONNX
/// 格式: ONNX (.onnx)
/// 源: [HuggingFace](https://huggingface.co/onnx-community/SmolLM2-135M-ONNX)
#[test]
fn e2e_generator_onnx() {
    const MODEL: &str = "onnx-community/SmolLM2-135M-ONNX";

    let client = Client::new_chat(MODEL).expect("Failed to load ONNX model");
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.7)
        .top_k(40)
        .top_p(0.95)
        .generate()
        .expect("Generation failed");

    let text = response.text.trim();
    assert!(!text.is_empty(), "Output should not be empty");
    assert!(text.len() > 3, "Output should be at least 4 characters");
}
