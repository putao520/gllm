//! Debug: Qwen3-Embedding-0.6B GGUF 逐步执行
//!
//! cargo run --example test_qwen3_embed

use gllm::Client;
use std::time::Instant;

fn main() {
    let model = "Qwen/Qwen3-Embedding-0.6B-GGUF";

    println!("[1/4] Loading model: {model} ...");
    let t0 = Instant::now();
    let client = match Client::new_embedding(model) {
        Ok(c) => {
            println!("  OK ({:.2}s)", t0.elapsed().as_secs_f64());
            c
        }
        Err(e) => {
            eprintln!("  FAIL: {e}");
            return;
        }
    };

    let manifest = client.manifest().expect("Failed to read manifest");
    println!("  kind: {:?}", manifest.kind);
    println!("  arch: {:?}", manifest.arch);

    // Test with single short text first
    let text = "Hi";
    println!("[2/4] Running embed on single short text: '{text}' ...");
    let t1 = Instant::now();
    match client.embed([text]) {
        Ok(response) => {
            println!("  OK ({:.2}s)", t1.elapsed().as_secs_f64());
            println!("  embeddings: {}", response.embeddings.len());
            if let Some(emb) = response.embeddings.first() {
                let norm: f32 = emb.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                println!("  dims={}, L2 norm={:.6}", emb.embedding.len(), norm);
            }
        }
        Err(e) => {
            eprintln!("  FAIL: {e}");
        }
    }

    println!("[3/4] Running embed on 2 texts ...");
    let t2 = Instant::now();
    let texts = ["Hello, world!", "Rust programming language"];
    match client.embed(texts) {
        Ok(response) => {
            println!("  OK ({:.2}s)", t2.elapsed().as_secs_f64());
            for (i, emb) in response.embeddings.iter().enumerate() {
                let norm: f32 = emb.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                println!("  emb[{}]: dims={}, L2 norm={:.6}", i, emb.embedding.len(), norm);
            }
        }
        Err(e) => {
            eprintln!("  FAIL: {e}");
        }
    }

    println!("[4/4] Done.");
}
