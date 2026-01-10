//! Test quantization support in registry
//!
//! Run with: cargo run --example test_quantization

use gllm::{ModelRegistry, Quantization};

fn main() {
    println!("=== gllm Quantization Support Demo ===\n");

    let registry = ModelRegistry::new();

    // 1. Basic resolution (no quantization)
    println!("1. Basic Resolution:");
    let info = registry.resolve("qwen3-embedding-0.6b").unwrap();
    println!("   qwen3-embedding-0.6b -> {}", info.repo_id);
    println!("   Quantization: {:?}\n", info.quantization);

    // 2. Quantized variants using :suffix syntax
    println!("2. Quantized Variants (alias:quant syntax):");
    let variants = [
        "qwen3-embedding-0.6b:int4",
        "qwen3-embedding-0.6b:int8",
        "qwen3-embedding-8b:awq",
        "qwen3-reranker-4b:gptq",
    ];
    for variant in variants {
        let info = registry.resolve(variant).unwrap();
        println!("   {} -> {} ({:?})", variant, info.repo_id, info.quantization);
    }
    println!();

    // 3. Direct repo ID with quantization inference
    println!("3. Direct Repo ID (auto-detect quantization):");
    let repos = [
        "Qwen/Qwen3-Embedding-8B-Int4",
        "some-org/model-AWQ",
        "nvidia/llama-embed-nemotron-8b-GPTQ",
    ];
    for repo in repos {
        let info = registry.resolve(repo).unwrap();
        println!("   {} -> {:?}", repo, info.quantization);
    }
    println!();

    // 4. Models that don't support quantization
    println!("4. Non-quantizable Models (suffix ignored):");
    let info = registry.resolve("bge-small-en:int4").unwrap();
    println!("   bge-small-en:int4 -> {} (quant: {:?})", info.repo_id, info.quantization);
    println!();

    // 5. Check quantization support
    println!("5. Quantization Support Check:");
    let models = ["qwen3-embedding-8b", "bge-small-en", "jina-embeddings-v4"];
    for model in models {
        let supports = registry.supports_quantization(model);
        let quants = registry.available_quantizations(model);
        println!("   {}: supports_quant={}, variants={:?}", model, supports, quants);
    }
    println!();

    // 6. All supported quantization types
    println!("6. Supported Quantization Types:");
    let quant_suffixes = ["int4", "int8", "awq", "gptq", "gguf", "fp8", "bnb4", "bnb8"];
    for suffix in quant_suffixes {
        if let Some(q) = Quantization::from_suffix(suffix) {
            println!("   :{} -> {:?} (repo_suffix: \"{}\")", suffix, q, q.repo_suffix());
        }
    }

    println!("\n=== Demo Complete ===");
}
