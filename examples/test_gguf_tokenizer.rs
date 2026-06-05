use std::collections::BTreeMap;

fn main() {
    let json = r#"{"tokenizer.ggml.model": [4], "tokenizer.ggml.tokens": [6, 8, 10], "tokenizer.ggml.merges": [6, 8]}"#;
    let metadata: BTreeMap<String, serde_json::Value> = serde_json::from_str(json).unwrap();

    println!("=== GGUF Tokenizer Metadata Format ===");

    // 检查 tokenizer.ggml.model
    if let Some(model) = metadata.get("tokenizer.ggml.model") {
        println!("\ntokenizer.ggml.model: {model}");
        if let Some(arr) = model.as_array() {
            println!("  Is array: true, len={}", arr.len());
            if let Some(num) = arr.first() {
                println!("  First element: {num}, is_i64={}", num.is_i64());
                if let Some(n) = num.as_u64() {
                    println!("  As u64: {n}");
                } else if let Some(n) = num.as_i64() {
                    println!("  As i64: {n}");
                }
            }
        }
    }

    // 检查 tokenizer.ggml.tokens
    if let Some(tokens) = metadata.get("tokenizer.ggml.tokens") {
        println!("\ntokenizer.ggml.tokens (first 3):");
        if let Some(arr) = tokens.as_array() {
            for (i, v) in arr.iter().take(3).enumerate() {
                let type_str = if v.is_i64() {
                    "i64"
                } else if v.is_u64() {
                    "u64"
                } else if v.is_string() {
                    "string"
                } else {
                    "other"
                };
                println!("  [{i}]: {v} ({type_str})");
            }
        }
    }
}
