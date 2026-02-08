// Debug Qwen3 GGUF tensor names and shapes
use gllm::loader::Loader;

fn main() {
    let mut loader = Loader::from("Qwen/Qwen3-0.6B-GGUF").expect("Failed to load model");

    // Get architecture name
    if let Ok(Some(arch)) = loader.gguf_architecture_name() {
        println!("Architecture: {}", arch);
    }

    // Note: We can't directly access gguf since it's private
    // But we can try to trigger an error that shows the names
    println!(
        "\nNote: This is a placeholder - actual tensor names need to be inspected differently"
    );
}
