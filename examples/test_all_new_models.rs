//! Test all new models added for June 2025+
//!
//! Run with: cargo run --example test_all_new_models

use gllm::ModelRegistry;

fn main() {
    println!("=== Testing All New Models (June 2025+) ===\n");

    let registry = ModelRegistry::new();

    // All new models added
    let new_models = [
        // Embedding models
        ("qwen3-embedding-0.6b", "Embedding", true),   // Small, test inference
        ("qwen3-embedding-4b", "Embedding", false),    // Large, registry only
        ("qwen3-embedding-8b", "Embedding", false),    // Large, registry only
        ("jina-embeddings-v4", "Embedding", true),     // Test inference
        ("llama-embed-nemotron-8b", "Embedding", false), // Large, registry only
        // Reranker models
        ("qwen3-reranker-0.6b", "Reranker", true),     // Small, test inference
        ("qwen3-reranker-4b", "Reranker", false),      // Large, registry only
        ("qwen3-reranker-8b", "Reranker", false),      // Large, registry only
        ("jina-reranker-v3", "Reranker", true),        // Test inference
    ];

    println!("=== Part 1: Registry Resolution ===\n");

    let mut registry_passed = 0;

    for (alias, expected_type, _) in &new_models {
        match registry.resolve(alias) {
            Ok(info) => {
                let type_str = match info.model_type {
                    gllm::ModelType::Embedding => "Embedding",
                    gllm::ModelType::Rerank => "Reranker",
                    gllm::ModelType::Generator => "Generator",
                };
                let status = if type_str == *expected_type { "✅" } else { "⚠️" };
                println!("{} {} -> {} ({:?})", status, alias, info.repo_id, info.architecture);
                registry_passed += 1;
            }
            Err(e) => {
                println!("❌ {} -> Error: {}", alias, e);
            }
        }
    }

    println!("\nRegistry: {}/{} passed\n", registry_passed, new_models.len());

    println!("=== Part 2: Download & Inference (Small Models Only) ===\n");

    let small_models: Vec<_> = new_models.iter()
        .filter(|(_, _, test_inference)| *test_inference)
        .collect();

    for (alias, model_type, _) in &small_models {
        println!("--- Testing {} ({}) ---", alias, model_type);

        if *model_type == "Embedding" {
            test_embedding(alias);
        } else {
            test_reranker(alias);
        }
        println!();
    }

    println!("=== Test Complete ===");
}

fn test_embedding(alias: &str) {
    println!("📥 Loading {}...", alias);

    match gllm::FallbackEmbedder::new(alias) {
        Ok(embedder) => {
            println!("✅ Model loaded!");

            match embedder.embed("This is a test sentence for embedding.") {
                Ok(embedding) => {
                    println!("✅ Embedding generated: dim={}, first_val={:.6}",
                        embedding.len(), embedding[0]);
                }
                Err(e) => {
                    println!("❌ Embedding failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Load failed: {}", e);
        }
    }
}

fn test_reranker(alias: &str) {
    println!("📥 Loading {}...", alias);

    // Client::new requires model name
    match gllm::Client::new(alias) {
        Ok(client) => {
            println!("✅ Model loaded!");

            let query = "What is machine learning?";
            let documents = vec![
                "Machine learning is a subset of artificial intelligence.",
                "The weather is nice today.",
                "Deep learning uses neural networks.",
            ];

            match client.rerank(query, documents).generate() {
                Ok(results) => {
                    println!("✅ Rerank completed!");
                    for (i, result) in results.results.iter().enumerate() {
                        println!("   #{}: doc[{}] score={:.4}",
                            i + 1, result.index, result.score);
                    }
                }
                Err(e) => {
                    println!("❌ Rerank failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Load failed: {}", e);
        }
    }
}
