use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Qwen3-0.6B generation (smallest dense model)");
    println!("Model: Qwen/Qwen3-0.6B (0.6B params, ~1.5GB GPU memory)\n");

    let client = Client::new("qwen3-0.6b")?;

    println!("Client created successfully!");

    let response = client
        .generate("What is 2+2? Answer in one word:")
        .max_new_tokens(8)
        .generate()?;

    println!("Response: {}", response.text);
    println!("Tokens generated: {}", response.tokens.len());

    Ok(())
}
