//! Test download for qwen3-embedding-0.6b
//!
//! Run with: cargo run --example test_download

use gllm::ModelRegistry;

fn main() {
    println!("=== Testing qwen3-embedding-0.6b download ===\n");

    // 1. Test registry resolution
    let registry = ModelRegistry::new();
    let info = registry.resolve("qwen3-embedding-0.6b").expect("Failed to resolve model");

    println!("✅ Registry resolution:");
    println!("   Alias: {}", info.alias);
    println!("   Repo ID: {}", info.repo_id);
    println!("   Model Type: {:?}", info.model_type);
    println!("   Architecture: {:?}", info.architecture);
    println!();

    // 2. Test FallbackEmbedder creation (will trigger download)
    println!("📥 Attempting to load model (will download if not cached)...");
    println!("   This may take a few minutes for first download.\n");

    match gllm::FallbackEmbedder::new("qwen3-embedding-0.6b") {
        Ok(embedder) => {
            println!("✅ Model loaded successfully!");

            // 3. Test embedding generation
            println!("\n📊 Testing embedding generation...");
            match embedder.embed("Hello, this is a test sentence.") {
                Ok(embedding) => {
                    println!("✅ Embedding generated!");
                    println!("   Dimension: {}", embedding.len());
                    println!("   First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
                }
                Err(e) => {
                    println!("❌ Embedding failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Model loading failed: {}", e);
        }
    }

    println!("\n=== Test complete ===");
}
