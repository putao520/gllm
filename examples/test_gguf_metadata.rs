use std::collections::BTreeMap;

fn main() {
    // 模拟 gguf-rs 返回的数据
    let json = r#"{
        "tokenizer.ggml.model": [4],
        "tokenizer.ggml.tokens": [6, 8, 10],
        "tokenizer.ggml.merges": [6, 8, 10]
    }"#;

    let metadata: BTreeMap<String, serde_json::Value> = serde_json::from_str(json).unwrap();

    // 检查 tokenizer.ggml.model
    if let Some(model) = metadata.get("tokenizer.ggml.model") {
        println!("tokenizer.ggml.model: {model}");
        if let Some(arr) = model.as_array() {
            if let Some(first) = arr.first() {
                println!(
                    "  First element type: {}",
                    if first.is_number() { "number" } else { "other" }
                );
                // GGUF tokenizer 类型常量 (来自 llama.cpp)
                let type_name = match first.as_u64().unwrap_or(0) {
                    0 => "normal (Raw)",
                    1 => "bpe (Byte Pair Encoding)",
                    2 => "spx (SentencePiece with unigram)",
                    3 => "unigram",
                    4 => "llama (SentencePiece)",
                    _ => "unknown",
                };
                println!("  Tokenizer type: {type_name}");
            }
        }
    }

    // 检查 tokens 数组
    if let Some(tokens) = metadata.get("tokenizer.ggml.tokens") {
        if let Some(arr) = tokens.as_array() {
            println!("\ntokenizer.ggml.tokens: {} entries", arr.len());
            println!("  First 5 values: {:?}", &arr[..5.min(arr.len())]);
        }
    }

    println!("\n=== 问题分析 ===");
    println!("tokenizer.ggml.tokens 存储的是整数（字符串表偏移量）");
    println!("需要额外的字符串表来解析实际的 token 字符串");
}
