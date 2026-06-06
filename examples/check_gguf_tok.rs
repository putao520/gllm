use std::env;
use gllm::loader::gguf::GgufReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).expect("need GGUF path");
    let reader = GgufReader::open(&path)?;
    
    println!("tokenizer.ggml.model = {:?}", reader.tokenizer_model());
    println!("bos_token_id = {:?}", reader.bos_token_id());
    println!("eos_token_id = {:?}", reader.eos_token_id());
    println!("add_bos_token = {:?}", reader.add_bos_token());
    println!("add_eos_token = {:?}", reader.add_eos_token());
    println!("unknown_token_id = {:?}", reader.tokenizer_unknown_token_id());
    
    if let Ok(tokens) = reader.tokenizer_tokens() {
        println!("tokens count = {}", tokens.len());
        println!("first 10 tokens:");
        for (i, t) in tokens.iter().take(10).enumerate() {
            println!("  [{}]: '{}'", i, t);
        }
    }
    
    if let Ok(merges) = reader.tokenizer_merges() {
        println!("merges count = {}", merges.len());
        for (i, m) in merges.iter().take(5).enumerate() {
            println!("  merge[{}]: '{}'", i, m);
        }
    } else {
        println!("merges: None or error");
    }
    
    Ok(())
}
