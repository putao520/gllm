use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing smallest MoE model: Qwen3-30B-A3B");
    println!("Total: 30B params, Active: 3B params per token\n");

    // Qwen3-30B-A3B 是最小的 MoE 模型
    let client = Client::new("qwen3-30b-a3b")?;

    println!("Client created successfully!");

    let response = client
        .generate("What is 2+2? Answer in one word:")
        .max_new_tokens(8)
        .generate()?;

    println!("Response: {}", response.text);
    println!("Tokens generated: {}", response.tokens.len());

    Ok(())
}
