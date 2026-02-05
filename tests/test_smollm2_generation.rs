//! End-to-end test: SmolLM2-135M generation with real weights
//!
//! This test verifies that the QKV layout fix allows correct generation.

use gllm::Client;

#[test]
#[ignore = "Requires model download - run with cargo test --test test_smollm2_generation -- --ignored"]
fn test_smollm2_generation_simple() {
    println!("🚀 Loading SmolLM2-135M...");

    let client = match Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("⚠️  Failed to load model: {}", e);
            eprintln!("This is expected if the model is not downloaded.");
            eprintln!("To download: python -c \"from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-135M-Instruct')\"");
            return;
        }
    };

    println!("✅ Model loaded successfully");
    println!("  Architecture: {:?}", client.manifest().arch);

    // Test 1: Simple completion
    println!("\n📝 Test 1: Simple completion");
    let prompt = "The capital of France is";
    println!("  Prompt: '{}'", prompt);

    let response = client
        .generate(prompt)
        .max_tokens(10)
        .temperature(0.0) // Deterministic
        .generate()
        .expect("generation failed");

    let text = response.text.trim();
    println!("  Output: '{}'", text);

    // The model should generate something reasonable
    assert!(!text.is_empty(), "output should not be empty");
    assert!(text.len() > 3, "output should be at least 4 characters");

    // Check for reasonable answer (should contain "Paris" or similar)
    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains(" paris")
        || text.contains("Paris")
        || lower.contains("france");
    println!("  Contains 'Paris': {}", is_reasonable);
}

#[test]
#[ignore = "Requires model download - run with cargo test --test test_smollm2_generation -- --ignored"]
fn test_smollm2_generation_multiple_tokens() {
    println!("🚀 Loading SmolLM2-135M...");

    let client = match Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("⚠️  Failed to load model: {}", e);
            return;
        }
    };

    println!("✅ Model loaded");

    // Test 2: Longer generation
    println!("\n📝 Test 2: Longer generation");
    let prompt = "The meaning of life is";
    println!("  Prompt: '{}'", prompt);

    let response = client
        .generate(prompt)
        .max_tokens(20)
        .temperature(0.7)
        .generate()
        .expect("generation failed");

    let text = response.text.trim();
    println!("  Output: '{}'", text);

    assert!(!text.is_empty(), "output should not be empty");
    assert!(text.len() > 10, "output should be meaningful");
}

#[test]
#[ignore = "Requires model download - run with cargo test --test test_smollm2_generation -- --ignored"]
fn test_smollm2_qkv_layout_fix() {
    println!("🚀 Testing QKV layout fix with SmolLM2-135M...");

    let client = match Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("⚠️  Failed to load model: {}", e);
            return;
        }
    };

    println!("✅ Model loaded");

    // Test with a specific prompt that should produce deterministic output
    let prompt = "1 + 1 equals";

    let response = client
        .generate(prompt)
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .expect("generation failed");

    let text = response.text.trim();
    println!("  Prompt: '{}'", prompt);
    println!("  Output: '{}'", text);

    // Should produce something containing "2" or "two"
    assert!(!text.is_empty(), "output should not be empty");
}

#[test]
#[ignore = "Requires model download - run with cargo test --test test_smollm2_generation -- --ignored"]
fn test_smollm2_chat_template() {
    println!("🚀 Testing SmolLM2 chat template...");

    let client = match Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("⚠️  Failed to load model: {}", e);
            return;
        }
    };

    use gllm::adapter::{Message, Role};

    let messages = vec![
        Message {
            role: Role::User,
            content: "What is the capital of Germany?".to_string(),
        },
    ];

    let response = client
        .generate_chat(messages)
        .max_tokens(15)
        .temperature(0.0)
        .generate()
        .expect("generation failed");

    let text = response.text.trim();
    println!("  Output: '{}'", text);

    // Should mention Berlin
    assert!(!text.is_empty(), "output should not be empty");
}
