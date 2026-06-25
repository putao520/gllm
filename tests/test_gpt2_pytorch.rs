//! E2E test: GPT-2 via native PytorchLoader (no conversion to SafeTensors)

use gllm::Client;

/// GPT-2 PyTorch .bin model — native TensorProvider path (Epoch 1)
#[test]
#[ignore] // requires test_models/gpt2-pytorch/
fn test_gpt2_pytorch_chat() {
    let client = Client::new_chat("test_models/gpt2-pytorch")
        .expect("Failed to load GPT-2 PyTorch model");
    
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Failed to generate");

    let text = response.text.trim();
    assert!(!text.is_empty(), "Generated text should not be empty");
    println!("GPT-2 PyTorch output: {}", text);
}
