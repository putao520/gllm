//! Real E2E inference tests for all 26 supported models.
//! Tests actual model inference with real data.
//!
//! Run with: cargo test --test integration -- --ignored inference_e2e

use gllm::{Client, ClientConfig, Device, ModelRegistry};
use std::path::PathBuf;

fn get_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".gllm")
        .join("models")
}

/// Test embedding models for inference - 同步版本
#[cfg(not(feature = "tokio"))]
#[test]
fn test_all_embedding_models_inference() {
    let embedding_models = [
        "bge-small-zh",
        "bge-small-en",
        "bge-base-en",
        "bge-large-en",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-MiniLM-L12-v2",
        "all-distilroberta-v1",
        "e5-large",
        "e5-base",
        "e5-small",
        "jina-embeddings-v2-base-en",
        "jina-embeddings-v2-small-en",
        "m3e-base",
        "multilingual-MiniLM-L12-v2",
        "distiluse-base-multilingual-cased-v1",
    ];

    let test_texts = vec![
        ("English", "The quick brown fox jumps over the lazy dog"),
        ("Chinese", "快速的棕色狐狸跳过懒惰的狗"),
        ("Mixed", "Hello世界 world你好"),
    ];

    println!("\n🧪 Testing {} embedding models", embedding_models.len());
    println!("{}", "=".repeat(80));

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    let mut successful = 0;
    let mut failed = 0;

    for (i, model_alias) in embedding_models.iter().enumerate() {
        print!("[{:2}/18] Testing {} ... ", i + 1, model_alias);

        match Client::with_config(model_alias, config.clone()) {
            Ok(client) => {
                match client
                    .embeddings(&test_texts.iter().map(|(_, t)| *t).collect::<Vec<_>>())
                    .generate()
                {
                    Ok(response) => {
                        let embeddings = &response.embeddings;
                        println!(
                            "✅ {} dims, {} vectors",
                            embeddings.first().map(|e| e.embedding.len()).unwrap_or(0),
                            embeddings.len()
                        );
                        successful += 1;
                    }
                    Err(e) => {
                        println!("❌ Inference failed: {}", e);
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                println!("❌ Client creation failed: {}", e);
                failed += 1;
            }
        }
    }

    println!("{}", "=".repeat(80));
    println!("✅ Successful: {}/18", successful);
    println!("❌ Failed: {}/18", failed);

    if failed > 0 {
        panic!("{} embedding models failed inference test", failed);
    }
}

/// Test embedding models for inference - 异步版本
#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn test_all_embedding_models_inference() {
    let embedding_models = [
        "bge-small-zh",
        "bge-small-en",
        "bge-base-en",
        "bge-large-en",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-MiniLM-L12-v2",
        "all-distilroberta-v1",
        "e5-large",
        "e5-base",
        "e5-small",
        "jina-embeddings-v2-base-en",
        "jina-embeddings-v2-small-en",
        "m3e-base",
        "multilingual-MiniLM-L12-v2",
        "distiluse-base-multilingual-cased-v1",
    ];

    let test_texts = vec![
        ("English", "The quick brown fox jumps over the lazy dog"),
        ("Chinese", "快速的棕色狐狸跳过懒惰的狗"),
        ("Mixed", "Hello世界 world你好"),
    ];

    println!("\n🧪 Testing {} embedding models", embedding_models.len());
    println!("{}", "=".repeat(80));

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    let mut successful = 0;
    let mut failed = 0;

    for (i, model_alias) in embedding_models.iter().enumerate() {
        print!("[{:2}/18] Testing {} ... ", i + 1, model_alias);

        match Client::with_config(model_alias, config.clone()).await {
            Ok(client) => {
                match client
                    .embeddings(&test_texts.iter().map(|(_, t)| *t).collect::<Vec<_>>())
                    .generate()
                    .await
                {
                    Ok(response) => {
                        let embeddings = &response.embeddings;
                        println!(
                            "✅ {} dims, {} vectors",
                            embeddings.first().map(|e| e.embedding.len()).unwrap_or(0),
                            embeddings.len()
                        );
                        successful += 1;
                    }
                    Err(e) => {
                        println!("❌ Inference failed: {}", e);
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                println!("❌ Client creation failed: {}", e);
                failed += 1;
            }
        }
    }

    println!("{}", "=".repeat(80));
    println!("✅ Successful: {}/18", successful);
    println!("❌ Failed: {}/18", failed);

    if failed > 0 {
        panic!("{} embedding models failed inference test", failed);
    }
}

/// Test reranking models for inference - 同步版本
#[cfg(not(feature = "tokio"))]
#[test]
fn test_all_rerank_models_inference() {
    let rerank_models = [
        "bge-reranker-v2",
        "bge-reranker-large",
        "bge-reranker-base",
        "ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2",
        "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-electra-base",
        "quora-distilroberta-base",
    ];

    let query = "What is machine learning?";
    let documents = vec![
        "Machine learning is a subset of artificial intelligence",
        "Python is a programming language",
        "Deep learning uses neural networks",
        "Data science involves statistical analysis",
    ];

    println!("\n🧪 Testing {} reranking models", rerank_models.len());
    println!("{}", "=".repeat(80));

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    let mut successful = 0;
    let mut failed = 0;

    for (i, model_alias) in rerank_models.iter().enumerate() {
        print!("[{:2}/8] Testing {} ... ", i + 1, model_alias);

        match Client::with_config(model_alias, config.clone()) {
            Ok(client) => match client.rerank(query, &documents).generate() {
                Ok(response) => {
                    let results = &response.results;
                    if results.is_empty() {
                        println!("❌ No results returned");
                        failed += 1;
                    } else {
                        let top_score = results.first().map(|r| r.score).unwrap_or(0.0);
                        println!("✅ {} results, top score: {:.4}", results.len(), top_score);
                        successful += 1;
                    }
                }
                Err(e) => {
                    println!("❌ Inference failed: {}", e);
                    failed += 1;
                }
            },
            Err(e) => {
                println!("❌ Client creation failed: {}", e);
                failed += 1;
            }
        }
    }

    println!("{}", "=".repeat(80));
    println!("✅ Successful: {}/8", successful);
    println!("❌ Failed: {}/8", failed);

    if failed > 0 {
        panic!("{} reranking models failed inference test", failed);
    }
}

/// Test reranking models for inference - 异步版本
#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn test_all_rerank_models_inference() {
    let rerank_models = [
        "bge-reranker-v2",
        "bge-reranker-large",
        "bge-reranker-base",
        "ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2",
        "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-electra-base",
        "quora-distilroberta-base",
    ];

    let query = "What is machine learning?";
    let documents = vec![
        "Machine learning is a subset of artificial intelligence",
        "Python is a programming language",
        "Deep learning uses neural networks",
        "Data science involves statistical analysis",
    ];

    println!("\n🧪 Testing {} reranking models", rerank_models.len());
    println!("{}", "=".repeat(80));

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    let mut successful = 0;
    let mut failed = 0;

    for (i, model_alias) in rerank_models.iter().enumerate() {
        print!("[{:2}/8] Testing {} ... ", i + 1, model_alias);

        match Client::with_config(model_alias, config.clone()).await {
            Ok(client) => match client.rerank(query, &documents).generate().await {
                Ok(response) => {
                    let results = &response.results;
                    if results.is_empty() {
                        println!("❌ No results returned");
                        failed += 1;
                    } else {
                        let top_score = results.first().map(|r| r.score).unwrap_or(0.0);
                        println!("✅ {} results, top score: {:.4}", results.len(), top_score);
                        successful += 1;
                    }
                }
                Err(e) => {
                    println!("❌ Inference failed: {}", e);
                    failed += 1;
                }
            },
            Err(e) => {
                println!("❌ Client creation failed: {}", e);
                failed += 1;
            }
        }
    }

    println!("{}", "=".repeat(80));
    println!("✅ Successful: {}/8", successful);
    println!("❌ Failed: {}/8", failed);

    if failed > 0 {
        panic!("{} reranking models failed inference test", failed);
    }
}

/// Test a single embedding model with detailed output - 同步版本
#[cfg(not(feature = "tokio"))]
#[test]
fn test_single_embedding_model_detailed() {
    let model = "bge-small-en";
    let texts = vec!["Hello world", "How are you?", "Machine learning"];

    println!("\n🔬 Detailed test for {}", model);
    println!("{}", "=".repeat(80));

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    match Client::with_config(model, config) {
        Ok(client) => {
            match client.embeddings(&texts).generate() {
                Ok(response) => {
                    println!("✅ Model loaded successfully");
                    println!("\nInput texts: {} items", texts.len());
                    println!("Embeddings returned: {} items", response.embeddings.len());

                    for (i, embedding) in response.embeddings.iter().enumerate() {
                        let vec = &embedding.embedding;
                        let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean = vec.iter().sum::<f32>() / vec.len() as f32;

                        println!("\nText {}: '{}'", i, texts.get(i).unwrap_or(&"unknown"));
                        println!("  Dimensions: {}", vec.len());
                        println!("  Min: {:.6}, Max: {:.6}, Mean: {:.6}", min, max, mean);
                        println!("  First 5 values: {:?}", &vec[..5.min(vec.len())]);
                    }

                    // Compute similarity between first two embeddings
                    if response.embeddings.len() >= 2 {
                        let emb1 = &response.embeddings[0].embedding;
                        let emb2 = &response.embeddings[1].embedding;

                        let dot_product: f32 =
                            emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
                        let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let similarity = dot_product / (norm1 * norm2);

                        println!("\nSimilarity between text 0 and text 1: {:.6}", similarity);
                    }
                }
                Err(e) => {
                    panic!("Inference failed: {}", e);
                }
            }
        }
        Err(e) => {
            panic!("Failed to create client: {}", e);
        }
    }

    println!("{}", "=".repeat(80));
    println!("✅ Test completed successfully!");
}

/// Test a single embedding model with detailed output - 异步版本
#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn test_single_embedding_model_detailed() {
    let model = "bge-small-en";
    let texts = vec!["Hello world", "How are you?", "Machine learning"];

    println!("\n🔬 Detailed test for {}", model);
    println!("{}", "=".repeat(80));

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    match Client::with_config(model, config).await {
        Ok(client) => {
            match client.embeddings(&texts).generate().await {
                Ok(response) => {
                    println!("✅ Model loaded successfully");
                    println!("\nInput texts: {} items", texts.len());
                    println!("Embeddings returned: {} items", response.embeddings.len());

                    for (i, embedding) in response.embeddings.iter().enumerate() {
                        let vec = &embedding.embedding;
                        let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean = vec.iter().sum::<f32>() / vec.len() as f32;

                        println!("\nText {}: '{}'", i, texts.get(i).unwrap_or(&"unknown"));
                        println!("  Dimensions: {}", vec.len());
                        println!("  Min: {:.6}, Max: {:.6}, Mean: {:.6}", min, max, mean);
                        println!("  First 5 values: {:?}", &vec[..5.min(vec.len())]);
                    }

                    // Compute similarity between first two embeddings
                    if response.embeddings.len() >= 2 {
                        let emb1 = &response.embeddings[0].embedding;
                        let emb2 = &response.embeddings[1].embedding;

                        let dot_product: f32 =
                            emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
                        let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let similarity = dot_product / (norm1 * norm2);

                        println!("\nSimilarity between text 0 and text 1: {:.6}", similarity);
                    }
                }
                Err(e) => {
                    panic!("Inference failed: {}", e);
                }
            }
        }
        Err(e) => {
            panic!("Failed to create client: {}", e);
        }
    }

    println!("{}", "=".repeat(80));
    println!("✅ Test completed successfully!");
}

/// Test a single reranking model with detailed output - 同步版本
#[cfg(not(feature = "tokio"))]
#[test]
fn test_single_rerank_model_detailed() {
    let model = "bge-reranker-base";
    let query = "What is the capital of France?";
    let documents = vec![
        "Paris is the capital and most populous city of France",
        "The Eiffel Tower is located in Paris",
        "France is in Western Europe",
        "London is the capital of the United Kingdom",
        "Spain is south of France",
    ];

    println!("\n🔬 Detailed test for {}", model);
    println!("{}", "=".repeat(80));

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    match Client::with_config(model, config) {
        Ok(client) => match client.rerank(query, &documents).generate() {
            Ok(response) => {
                println!("✅ Model loaded successfully");
                println!("\nQuery: '{}'", query);
                println!("Documents: {} items", documents.len());
                println!("\nReranked results:");
                println!("{}", "-".repeat(80));

                for (rank, result) in response.results.iter().enumerate() {
                    println!(
                        "Rank {}: Score {:.4} | Doc {}: {}",
                        rank + 1,
                        result.score,
                        result.index,
                        documents
                            .get(result.index)
                            .unwrap_or(&"unknown")
                            .chars()
                            .take(60)
                            .collect::<String>()
                    );
                }
            }
            Err(e) => {
                panic!("Inference failed: {}", e);
            }
        },
        Err(e) => {
            panic!("Failed to create client: {}", e);
        }
    }

    println!("{}", "=".repeat(80));
    println!("✅ Test completed successfully!");
}

/// Test a single reranking model with detailed output - 异步版本
#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn test_single_rerank_model_detailed() {
    let model = "bge-reranker-base";
    let query = "What is the capital of France?";
    let documents = vec![
        "Paris is the capital and most populous city of France",
        "The Eiffel Tower is located in Paris",
        "France is in Western Europe",
        "London is the capital of the United Kingdom",
        "Spain is south of France",
    ];

    println!("\n🔬 Detailed test for {}", model);
    println!("{}", "=".repeat(80));

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    match Client::with_config(model, config).await {
        Ok(client) => match client.rerank(query, &documents).generate().await {
            Ok(response) => {
                println!("✅ Model loaded successfully");
                println!("\nQuery: '{}'", query);
                println!("Documents: {} items", documents.len());
                println!("\nReranked results:");
                println!("{}", "-".repeat(80));

                for (rank, result) in response.results.iter().enumerate() {
                    println!(
                        "Rank {}: Score {:.4} | Doc {}: {}",
                        rank + 1,
                        result.score,
                        result.index,
                        documents
                            .get(result.index)
                            .unwrap_or(&"unknown")
                            .chars()
                            .take(60)
                            .collect::<String>()
                    );
                }
            }
            Err(e) => {
                panic!("Inference failed: {}", e);
            }
        },
        Err(e) => {
            panic!("Failed to create client: {}", e);
        }
    }

    println!("{}", "=".repeat(80));
    println!("✅ Test completed successfully!");
}

/// Test model registry and type detection
#[test]
fn test_model_registry_and_types() {
    let registry = ModelRegistry::new();

    println!("\n📋 Model Registry and Type Detection");
    println!("{}", "=".repeat(80));

    let test_models = vec![
        ("bge-small-zh", gllm::ModelType::Embedding),
        ("bge-small-en", gllm::ModelType::Embedding),
        ("e5-large", gllm::ModelType::Embedding),
        ("m3e-base", gllm::ModelType::Embedding),
        ("bge-reranker-v2", gllm::ModelType::Rerank),
        ("ms-marco-MiniLM-L-6-v2", gllm::ModelType::Rerank),
    ];

    for (alias, expected_type) in test_models {
        match registry.resolve(alias) {
            Ok(info) => {
                let type_match = info.model_type == expected_type;
                let status = if type_match { "✅" } else { "❌" };
                println!("{} {} -> {} ({})", status, alias, info.repo_id, {
                    match info.model_type {
                        gllm::ModelType::Embedding => "Embedding",
                        gllm::ModelType::Rerank => "Rerank",
                    }
                });
            }
            Err(e) => {
                println!("❌ {} -> Error: {}", alias, e);
            }
        }
    }

    println!("{}", "=".repeat(80));
}
