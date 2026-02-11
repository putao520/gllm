// Debug Qwen3 GGUF tensor names and shapes
use gllm::loader::{Loader, LoaderConfig};

fn main() {
    let config = LoaderConfig::from_env();
    let loader = Loader::from_source_with_config("Qwen/Qwen3-0.6B-GGUF".to_string(), config)
        .expect("Failed to load model");

    // Get architecture name
    if let Ok(arch) = loader.gguf_architecture() {
        println!("Architecture: {}", arch);
    }

    // Note: We can't directly access gguf since it's private
    // But we can try to trigger an error that shows the names
    println!(
        "\nNote: This is a placeholder - actual tensor names need to be inspected differently"
    );
}
