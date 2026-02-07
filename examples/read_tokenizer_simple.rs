use gguf_rs::get_gguf_container_array_size;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/putao/.gllm/models/Mungert--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-bf16.gguf";

    let mut container = get_gguf_container_array_size(path, 100000)?;
    let model = container.decode()?;

    println!("=== GGUF Tokenizer Metadata ===\n");

    // 按顺序显示所有 tokenizer 相关的 KV
    let tokenizer_keys = vec![
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.sep_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.ggml.add_space_prefix",
    ];

    for key in &tokenizer_keys {
        if let Some(value) = model.metadata().get(key) {
            match value {
                gguf_rs::Value::String(s) => {
                    println!("{}: \"{}\"", key, s);
                }
                gguf_rs::Value::Number(n) => {
                    println!("{}: {}", key, n);
                }
                gguf_rs::Value::Bool(b) => {
                    println!("{}: {}", key, b);
                }
                gguf_rs::Value::Array(arr) => {
                    println!("{}: Array of {} elements", key, arr.len());
                    if *key == "tokenizer.ggml.tokens" {
                        println!("  First 30 tokens:");
                        for (i, v) in arr.iter().take(30).enumerate() {
                            if let Some(s) = v.as_str() {
                                println!("    [{}]: '{}'", i, s);
                            }
                        }
                    } else if *key == "tokenizer.ggml.merges" {
                        println!("  First 30 merges:");
                        for (i, v) in arr.iter().take(30).enumerate() {
                            if let Some(s) = v.as_str() {
                                println!("    [{}]: '{}'", i, s);
                            }
                        }
                    }
                }
                _ => {
                    println!("{}: {:?}", key, value);
                }
            }
        }
    }

    Ok(())
}
