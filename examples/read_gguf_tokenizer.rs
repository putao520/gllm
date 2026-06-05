use gllm::loader::gguf::GgufReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/putao/.gllm/models/Mungert--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-bf16.gguf";

    let model = GgufReader::open(path)?;

    println!("=== GGUF Tokenizer Metadata ===\n");
    println!("Version: {}", model.version());
    println!(
        "Architecture: {}",
        model.architecture().unwrap_or("unknown")
    );
    println!("\nTotal KV pairs: {}", model.metadata().len());

    // 查找所有 tokenizer 相关的 KV
    println!("\n=== Tokenizer Metadata ===");
    for (key, value) in model.metadata() {
        if key.contains("tokenizer") {
            println!("{}: {:?}", key, value);

            // 对于 tokens，显示前 20 个
            if key == "tokenizer.ggml.tokens" {
                if let Some(arr) = value.as_array() {
                    println!("  First 20 tokens:");
                    for (i, v) in arr.items.iter().take(20).enumerate() {
                        if let Some(s) = v.as_str() {
                            println!("    [{}]: '{}'", i, s);
                        }
                    }
                    println!("  ... ({} total tokens)", arr.items.len());
                }
            }
        }
    }

    Ok(())
}
