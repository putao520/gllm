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
    let is_reasonable = lower.contains("paris")
        || lower.contains("parís")
        || lower.contains("capital");
    assert!(is_reasonable, "Output should contain reasonable answer");
}

/// GGUF 格式的 Generator 测试
///
/// 模型: Mungert/SmolLM2-135M-Instruct-GGUF
/// 格式: GGUF (.gguf)
/// 源: [HuggingFace](https://huggingface.co/Mungert/SmolLM2-135M-Instruct-GGUF)
///
/// 注意：此测试暂时被跳过，因为需要为该 GGUF 模型添加 manifest 配置。
#[test]
#[ignore = "需要为 GGUF generator 模型添加配置"]
fn e2e_generator_gguf() {
    const MODEL: &str = "Mungert/SmolLM2-135M-Instruct-GGUF";

    let client = Client::new_chat(MODEL).expect("Failed to load GGUF model");
    let response = client
        .generate("1 + 1 equals")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .expect("Generation failed");

    let text = response.text.trim();
    assert!(!text.is_empty(), "Output should not be empty");

    // GGUF 量化模型输出应包含数字
    let has_digit = text.chars().any(|c| c.is_ascii_digit());
    assert!(has_digit, "Output should contain a number");
}

/// ONNX 格式的 Generator 测试
///
/// 模型: onnx-community/SmolLM2-135M-ONNX
/// 格式: ONNX (.onnx)
/// 源: [HuggingFace](https://huggingface.co/onnx-community/SmolLM2-135M-ONNX)
///
/// 注意：此测试暂时被跳过，因为 ONNX generator 模型暂不支持（需要 ONNX 推理引擎）。
#[test]
#[ignore = "ONNX generator 模型暂不支持，需要 ONNX 推理引擎"]
fn e2e_generator_onnx() {
    const MODEL: &str = "onnx-community/SmolLM2-135M-ONNX";

    let client = Client::new_chat(MODEL).expect("Failed to load ONNX model");
    let response = client
        .generate("Hello, world!")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .expect("Generation failed");

    let text = response.text.trim();
    assert!(!text.is_empty(), "Output should not be empty");
    assert!(text.len() > 2, "Output should be at least 3 characters");
}
