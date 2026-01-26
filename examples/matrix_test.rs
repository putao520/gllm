use gllm::{Client, ModelType};
use std::env;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let force_cpu = env::var("GLLM_FORCE_CPU").unwrap_or("0".to_string());
    let skip_large = env::var("GLLM_SKIP_LARGE").map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false);
    
    println!("====================================================");
    println!("        GLLM PERFORMANCE & STABILITY MATRIX         ");
    println!("====================================================");
    println!("GLLM_FORCE_CPU: {}", force_cpu);
    println!("SKIP_LARGE:     {}", skip_large);
    println!("OS:             {}", env::consts::OS);
    println!("----------------------------------------------------");

    // Model matrix: (alias, model_type, size_label)
    let models = [
        ("qwen3-embedding-0.6b", ModelType::Embedding, "Small"),
        ("qwen3-embedding-8b", ModelType::Embedding, "Large"),
        ("qwen3-reranker-0.6b", ModelType::Rerank, "Small"),
        ("jina-reranker-v3", ModelType::Rerank, "Large"),
        ("qwen3-next-0.6b", ModelType::Generator, "Small"),
        ("qwen3-8b:gguf", ModelType::Generator, "Large"),
    ];

    for (alias, mtype, size) in &models {
        if *size == "Large" && skip_large {
            println!("\n[SKIP] {} ({}) - large models skipped", alias, size);
            continue;
        }

        println!("\n[TEST] {} ({}, {:?})", alias, size, mtype);
        
        let start_load = Instant::now();
        match Client::new(alias).await {
            Ok(client) => {
                println!("  - Load time: {:?}", start_load.elapsed());
                let start_inf = Instant::now();
                
                match mtype {
                    ModelType::Embedding => {
                        match client.embeddings(["Stability and performance test for GLLM parallel kernels."]).generate().await {
                            Ok(resp) => {
                                let vec = &resp.embeddings[0].embedding;
                                println!("  - Inference time: {:?}", start_inf.elapsed());
                                println!("  - Result: dim={}, first={:.6}", vec.len(), vec[0]);
                                println!("  ✅ PASSED");
                            }
                            Err(e) => println!("  ❌ INFERENCE FAILED: {}", e),
                        }
                    }
                    ModelType::Rerank => {
                        let docs = vec![
                            "GLLM is a high-performance LLM inference library.".to_string(),
                            "The weather is nice today.".to_string(),
                        ];
                        match client.rerank("What is GLLM?", docs).generate().await {
                            Ok(resp) => {
                                println!("  - Inference time: {:?}", start_inf.elapsed());
                                for res in resp.results {
                                    println!("    - Doc {}: score={:.4}", res.index, res.score);
                                }
                                println!("  ✅ PASSED");
                            }
                            Err(e) => println!("  ❌ INFERENCE FAILED: {}", e),
                        }
                    }
                    ModelType::Generator => {
                        match client.generate("Summarize the benefits of parallel computing in one sentence.")
                            .max_new_tokens(32)
                            .generate().await {
                            Ok(output) => {
                                println!("  - Generation time: {:?}", start_inf.elapsed());
                                println!("  - Generated: \"{}\"", output.text.trim().replace('\n', " "));
                                println!("  ✅ PASSED");
                            }
                            Err(e) => println!("  ❌ GENERATION FAILED: {}", e),
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ❌ LOAD FAILED: {}", e);
            }
        }
    }

    println!("\n====================================================");
    println!("            MATRIX TEST COMPLETED                   ");
    println!("====================================================");

    Ok(())
}
