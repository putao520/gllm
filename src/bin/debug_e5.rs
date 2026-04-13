use gllm::Client;

fn main() {
    println!("Creating client for e5-small-v2...");
    let client = match Client::new_embedding("intfloat/e5-small-v2") {
        Ok(c) => {
            println!("✓ Client created successfully");
            c
        }
        Err(e) => {
            println!("✗ Failed to create client: {}", e);
            return;
        }
    };
    
    println!("✓ Model loaded successfully");
    
    println!("Running embedding...");
    match client.embed(["test"]) {
        Ok(response) => {
            println!("✓ Embedding generated");
            println!("Embedding shape: {:?}", response.embeddings[0].embedding.len());
        }
        Err(e) => {
            println!("✗ Embedding failed: {}", e);
        }
    }
}
