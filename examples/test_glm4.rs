use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing GLM-4-9B-Chat generation");
    println!("Model: THUDM/glm-4-9b-chat-hf (9B params)\n");

    // GLM-4-9B is the smallest GLM model in gllm registry
    // For quantized version, use: model-scope/glm-4-9b-chat-GPTQ-Int4
    let client = Client::new("glm-4-9b-chat")?;

    println!("Client created successfully!");

    let response = client
        .generate("What is 2+2? Answer in one word:")
        .max_new_tokens(8)
        .generate()?;

    println!("Response: {}", response.text);
    println!("Tokens generated: {}", response.tokens.len());

    Ok(())
}
