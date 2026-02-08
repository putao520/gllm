// List Qwen3 GGUF MLP tensor names
use gllm::loader::Loader;

fn main() {
    let loader = Loader::from("Qwen/Qwen3-0.6B-GGUF")
        .expect("Failed to load model");

    let gguf = loader.gguf.as_ref().expect("No GGUF");
    let names = gguf.names();

    // Print MLP-related tensors for layers 0-2
    println!("=== Qwen3 GGUF MLP 张量名称 ===\n");
    for name in names.iter() {
        if (name.contains("blk.") || name.contains("layers.")) && name.contains(".mlp.") {
            println!("{}", name);
        }
    }
}
