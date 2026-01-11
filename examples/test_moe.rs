use gllm::ModelRegistry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing MoE model registry...\n");

    let registry = ModelRegistry::new();

    let moe_models = [
        "glm-4.7",
        "qwen3-30b-a3b",
        "qwen3-235b-a22b",
        "mixtral-8x7b-instruct",
        "mixtral-8x22b-instruct",
        "deepseek-v3",
    ];

    for model in moe_models {
        match registry.resolve(model) {
            Ok(info) => {
                println!("✅ {} -> {}", model, info.repo_id);
                println!("   Architecture: {:?}", info.architecture);
                println!("   Type: {:?}", info.model_type);
                println!();
            }
            Err(e) => {
                println!("❌ {} -> Error: {}", model, e);
                println!();
            }
        }
    }

    Ok(())
}
