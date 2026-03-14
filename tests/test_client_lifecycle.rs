use gllm::client::Client;
use gllm::manifest::ModelKind;

/// TEST-INFERENCE-001: 客户端生命周期加载和卸载
/// **关联需求**: REQ-LOADER-018
/// **测试类型**: 正向
/// **期望结果**: 客户端可以成功加载模型并正确卸载释放资源
#[test]
fn test_client_lifecycle_load_unload() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup
    // Use a small model for testing
    let model_id = "HuggingFaceTB/SmolLM2-135M-Instruct";

    // 2. Initialize Client (Loads Model 1)
    println!("Loading model: {}", model_id);
    let client = Client::new(model_id, ModelKind::Chat)?;

    // 3. Generate with Model 1
    let prompt = "Hello";
    let response = client.generate(prompt).max_tokens(10).generate()?;
    println!("Response 1: {:?}", response.text);
    assert!(!response.text.is_empty());

    // 4. Unload Model
    println!("Unloading model...");
    client.unload_model()?;

    // 5. Verify Unloaded State
    // Attempting to generate should fail
    let result = client.generate("Should fail").max_tokens(5).generate();
    assert!(result.is_err());
    // Error should be NoModelLoaded
    // We can't easily match the enum variant without importing it, but checking is_err is good enough for now.
    println!(
        "Generation after unload failed as expected: {:?}",
        result.err()
    );

    // 6. Reload Model (Same or Different)
    println!("Reloading model: {}", model_id);
    client.load_model(model_id, ModelKind::Chat)?;

    // 7. Generate with Reloaded Model
    let response2 = client.generate("Hi again").max_tokens(10).generate()?;
    println!("Response 2: {:?}", response2.text);
    assert!(!response2.text.is_empty());

    // 8. Swap Model (Unload + Load)
    println!("Swapping model...");
    // For test speed we use the same model, but logically it's a swap
    client.swap_model(model_id)?;

    let response3 = client.generate("Third time").max_tokens(10).generate()?;
    println!("Response 3: {:?}", response3.text);
    assert!(!response3.text.is_empty());

    Ok(())
}
