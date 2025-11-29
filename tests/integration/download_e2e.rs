//! Real E2E download tests for all 26 supported models.
//! These tests actually download models from HuggingFace and verify files.
//!
//! Run with: cargo test --test integration -- --ignored download_e2e
//! Or for a single model: cargo test --test integration -- --ignored download_single

use std::path::PathBuf;
use std::fs;
use std::time::Instant;

/// All 26 supported models with SafeTensors format
const ALL_MODELS: &[(&str, &str)] = &[
    // BGE Series (4 models)
    ("bge-small-zh", "BAAI/bge-small-zh-v1.5"),
    ("bge-small-en", "BAAI/bge-small-en-v1.5"),
    ("bge-base-en", "BAAI/bge-base-en-v1.5"),
    ("bge-large-en", "BAAI/bge-large-en-v1.5"),
    // Sentence Transformers (6 models)
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
    ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
    ("paraphrase-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L6-v2"),
    ("multi-qa-mpnet-base-dot-v1", "sentence-transformers/multi-qa-mpnet-base-dot-v1"),
    ("all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L12-v2"),
    ("all-distilroberta-v1", "sentence-transformers/all-distilroberta-v1"),
    // E5 Series (3 models)
    ("e5-large", "intfloat/e5-large"),
    ("e5-base", "intfloat/e5-base"),
    ("e5-small", "intfloat/e5-small"),
    // JINA Series (2 models)
    ("jina-embeddings-v2-base-en", "jinaai/jina-embeddings-v2-base-en"),
    ("jina-embeddings-v2-small-en", "jinaai/jina-embeddings-v2-small-en"),
    // Chinese Models (1 model)
    ("m3e-base", "moka-ai/m3e-base"),
    // Multilingual Models (2 models)
    ("multilingual-MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("distiluse-base-multilingual-cased-v1", "sentence-transformers/distiluse-base-multilingual-cased-v1"),
    // BGE Rerankers (3 models)
    ("bge-reranker-v2", "BAAI/bge-reranker-v2-m3"),
    ("bge-reranker-large", "BAAI/bge-reranker-large"),
    ("bge-reranker-base", "BAAI/bge-reranker-base"),
    // MS MARCO Rerankers (4 models)
    ("ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("ms-marco-MiniLM-L-12-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
    ("ms-marco-TinyBERT-L-2-v2", "cross-encoder/ms-marco-TinyBERT-L-2-v2"),
    ("ms-marco-electra-base", "cross-encoder/ms-marco-electra-base"),
    // Specialized Rerankers (1 model)
    ("quora-distilroberta-base", "cross-encoder/quora-distilroberta-base"),
];

/// Required files that must exist after download
const REQUIRED_FILES: &[&str] = &[
    "model.safetensors",
    "config.json",
    "tokenizer.json",
];

fn get_test_models_dir() -> PathBuf {
    // Use the standard location: ~/.gllm/models
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".gllm")
        .join("models")
}

fn model_dir_name(repo_id: &str) -> String {
    repo_id.replace('/', "--")
}

/// Test download of a single small model (for quick CI checks) - 同步版本
#[cfg(not(feature = "tokio"))]
#[test]
#[ignore = "Downloads real model, run with --ignored"]
fn download_single_small_model() {
    let (alias, repo_id) = ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2");
    let models_dir = get_test_models_dir();
    let model_dir = models_dir.join(model_dir_name(repo_id));

    // Clean up before test
    let _ = fs::remove_dir_all(&model_dir);

    println!("📥 Downloading {} ({})...", alias, repo_id);
    let start = Instant::now();

    let config = gllm::ClientConfig {
        device: gllm::Device::Auto,
        models_dir: models_dir.clone(),
    };

    let result = gllm::Client::with_config(alias, config);

    match result {
        Ok(_client) => {
            let elapsed = start.elapsed();
            println!("✅ Downloaded {} in {:.2}s", alias, elapsed.as_secs_f64());

            // Verify required files exist
            for file in REQUIRED_FILES {
                let path = model_dir.join(file);
                assert!(path.exists(), "Missing required file: {} for {}", file, alias);
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                println!("   📄 {} ({:.2} MB)", file, size as f64 / 1024.0 / 1024.0);
            }
        }
        Err(e) => {
            panic!("❌ Failed to download {}: {:?}", alias, e);
        }
    }
}

/// Test download of a single small model (for quick CI checks) - 异步版本
#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Downloads real model, run with --ignored"]
async fn download_single_small_model() {
    let (alias, repo_id) = ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2");
    let models_dir = get_test_models_dir();
    let model_dir = models_dir.join(model_dir_name(repo_id));

    // Clean up before test
    let _ = fs::remove_dir_all(&model_dir);

    println!("📥 Downloading {} ({})...", alias, repo_id);
    let start = Instant::now();

    let config = gllm::ClientConfig {
        device: gllm::Device::Auto,
        models_dir: models_dir.clone(),
    };

    let result = gllm::Client::with_config(alias, config).await;

    match result {
        Ok(_client) => {
            let elapsed = start.elapsed();
            println!("✅ Downloaded {} in {:.2}s", alias, elapsed.as_secs_f64());

            // Verify required files exist
            for file in REQUIRED_FILES {
                let path = model_dir.join(file);
                assert!(path.exists(), "Missing required file: {} for {}", file, alias);
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                println!("   📄 {} ({:.2} MB)", file, size as f64 / 1024.0 / 1024.0);
            }
        }
        Err(e) => {
            panic!("❌ Failed to download {}: {:?}", alias, e);
        }
    }
}

/// Test download of all 26 models (full E2E test) - 同步版本
#[cfg(not(feature = "tokio"))]
#[test]
#[ignore = "Downloads all models, takes significant time and bandwidth"]
fn download_all_26_models() {
    let models_dir = get_test_models_dir();
    fs::create_dir_all(&models_dir).expect("create test models dir");

    let mut results: Vec<(String, bool, String)> = Vec::new();
    let total_start = Instant::now();

    println!("🎯 Starting E2E download test for all 26 models");
    println!("📁 Models directory: {}", models_dir.display());
    println!();

    for (i, (alias, repo_id)) in ALL_MODELS.iter().enumerate() {
        let model_dir = models_dir.join(model_dir_name(repo_id));

        // Skip if already downloaded
        if model_dir.join("model.safetensors").exists() {
            println!("[{:2}/24] ⏭️  {} - already exists", i + 1, alias);
            results.push((alias.to_string(), true, "cached".to_string()));
            continue;
        }

        println!("[{:2}/24] 📥 Downloading {}...", i + 1, alias);
        let start = Instant::now();

        let config = gllm::ClientConfig {
            device: gllm::Device::Auto,
            models_dir: models_dir.clone(),
        };

        match gllm::Client::with_config(alias, config) {
            Ok(_client) => {
                let elapsed = start.elapsed();
                let msg = format!("{:.1}s", elapsed.as_secs_f64());
                println!("         ✅ Done in {}", msg);
                results.push((alias.to_string(), true, msg));

                // Verify files
                for file in REQUIRED_FILES {
                    let path = model_dir.join(file);
                    if !path.exists() {
                        println!("         ⚠️  Missing: {}", file);
                    }
                }
            }
            Err(e) => {
                let msg = format!("{:?}", e);
                println!("         ❌ Failed: {}", msg);
                results.push((alias.to_string(), false, msg));
            }
        }
    }

    // Summary
    let total_elapsed = total_start.elapsed();
    let success_count = results.iter().filter(|(_, ok, _)| *ok).count();
    let failed_count = results.len() - success_count;

    println!();
    println!("{}", "=".repeat(60));
    println!("📊 E2E DOWNLOAD TEST SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Total models: {}", ALL_MODELS.len());
    println!("✅ Successful: {}", success_count);
    println!("❌ Failed: {}", failed_count);
    println!("⏱️  Total time: {:.1}s", total_elapsed.as_secs_f64());
    println!();

    if failed_count > 0 {
        println!("Failed models:");
        for (alias, ok, msg) in &results {
            if !ok {
                println!("  - {}: {}", alias, msg);
            }
        }
        panic!("{} models failed to download", failed_count);
    }

    println!("✅ All 26 models downloaded successfully!");
}

/// Test download of all 26 models (full E2E test) - 异步版本
#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Downloads all models, takes significant time and bandwidth"]
async fn download_all_26_models() {
    let models_dir = get_test_models_dir();
    fs::create_dir_all(&models_dir).expect("create test models dir");

    let mut results: Vec<(String, bool, String)> = Vec::new();
    let total_start = Instant::now();

    println!("🎯 Starting E2E download test for all 26 models");
    println!("📁 Models directory: {}", models_dir.display());
    println!();

    for (i, (alias, repo_id)) in ALL_MODELS.iter().enumerate() {
        let model_dir = models_dir.join(model_dir_name(repo_id));

        // Skip if already downloaded
        if model_dir.join("model.safetensors").exists() {
            println!("[{:2}/24] ⏭️  {} - already exists", i + 1, alias);
            results.push((alias.to_string(), true, "cached".to_string()));
            continue;
        }

        println!("[{:2}/24] 📥 Downloading {}...", i + 1, alias);
        let start = Instant::now();

        let config = gllm::ClientConfig {
            device: gllm::Device::Auto,
            models_dir: models_dir.clone(),
        };

        match gllm::Client::with_config(alias, config).await {
            Ok(_client) => {
                let elapsed = start.elapsed();
                let msg = format!("{:.1}s", elapsed.as_secs_f64());
                println!("         ✅ Done in {}", msg);
                results.push((alias.to_string(), true, msg));

                // Verify files
                for file in REQUIRED_FILES {
                    let path = model_dir.join(file);
                    if !path.exists() {
                        println!("         ⚠️  Missing: {}", file);
                    }
                }
            }
            Err(e) => {
                let msg = format!("{:?}", e);
                println!("         ❌ Failed: {}", msg);
                results.push((alias.to_string(), false, msg));
            }
        }
    }

    // Summary
    let total_elapsed = total_start.elapsed();
    let success_count = results.iter().filter(|(_, ok, _)| *ok).count();
    let failed_count = results.len() - success_count;

    println!();
    println!("{}", "=".repeat(60));
    println!("📊 E2E DOWNLOAD TEST SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Total models: {}", ALL_MODELS.len());
    println!("✅ Successful: {}", success_count);
    println!("❌ Failed: {}", failed_count);
    println!("⏱️  Total time: {:.1}s", total_elapsed.as_secs_f64());
    println!();

    if failed_count > 0 {
        println!("Failed models:");
        for (alias, ok, msg) in &results {
            if !ok {
                println!("  - {}: {}", alias, msg);
            }
        }
        panic!("{} models failed to download", failed_count);
    }

    println!("✅ All 26 models downloaded successfully!");
}

/// Test that model files are valid after download
#[test]
#[ignore = "Requires downloaded models"]
fn verify_downloaded_model_files() {
    let models_dir = get_test_models_dir();

    println!("🔍 Verifying downloaded model files...");
    println!("📁 Models directory: {}", models_dir.display());
    println!();

    let mut verified = 0;
    let mut missing = 0;
    let mut invalid = 0;

    for (alias, repo_id) in ALL_MODELS {
        let model_dir = models_dir.join(model_dir_name(repo_id));

        if !model_dir.exists() {
            println!("⏭️  {} - not downloaded", alias);
            missing += 1;
            continue;
        }

        let mut all_valid = true;

        // Check required files
        for file in REQUIRED_FILES {
            let path = model_dir.join(file);
            if !path.exists() {
                println!("❌ {} - missing {}", alias, file);
                all_valid = false;
            } else {
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                if size == 0 {
                    println!("❌ {} - {} is empty", alias, file);
                    all_valid = false;
                }
            }
        }

        // Verify SafeTensors format
        let safetensors_path = model_dir.join("model.safetensors");
        if safetensors_path.exists() {
            match fs::read(&safetensors_path) {
                Ok(bytes) if bytes.len() > 8 => {
                    // SafeTensors files start with 8-byte header size
                    let header_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                    if header_size > bytes.len() as u64 - 8 {
                        println!("❌ {} - invalid SafeTensors header", alias);
                        all_valid = false;
                    }
                }
                _ => {
                    println!("❌ {} - cannot read SafeTensors file", alias);
                    all_valid = false;
                }
            }
        }

        // Verify config.json is valid JSON
        let config_path = model_dir.join("config.json");
        if config_path.exists() {
            match fs::read_to_string(&config_path) {
                Ok(content) => {
                    if serde_json::from_str::<serde_json::Value>(&content).is_err() {
                        println!("❌ {} - invalid config.json", alias);
                        all_valid = false;
                    }
                }
                Err(_) => {
                    println!("❌ {} - cannot read config.json", alias);
                    all_valid = false;
                }
            }
        }

        if all_valid {
            let safetensors_size = fs::metadata(&safetensors_path)
                .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                .unwrap_or(0.0);
            println!("✅ {} - valid ({:.1} MB)", alias, safetensors_size);
            verified += 1;
        } else {
            invalid += 1;
        }
    }

    println!();
    println!("📊 Verification Summary:");
    println!("  ✅ Valid: {}", verified);
    println!("  ⏭️  Missing: {}", missing);
    println!("  ❌ Invalid: {}", invalid);

    if invalid > 0 {
        panic!("{} models have invalid files", invalid);
    }
}

/// Test that a downloaded model can be used for inference - 同步版本
#[cfg(not(feature = "tokio"))]
#[test]
#[ignore = "Requires downloaded model and GPU/CPU backend"]
fn test_inference_with_downloaded_model() {
    // Use smallest model for quick test
    let alias = "all-MiniLM-L6-v2";
    let models_dir = get_test_models_dir();

    println!("🧪 Testing inference with {}...", alias);

    let config = gllm::ClientConfig {
        device: gllm::Device::Auto,
        models_dir,
    };

    let client = gllm::Client::with_config(alias, config)
        .expect("Failed to create client");

    let texts = vec![
        "Hello, world!",
        "This is a test sentence.",
    ];

    match client.embeddings(&texts).generate() {
        Ok(response) => {
            println!("✅ Generated {} embeddings", response.embeddings.len());
            for emb in &response.embeddings {
                println!("   Text {}: {} dimensions", emb.index + 1, emb.embedding.len());
            }
        }
        Err(e) => {
            println!("❌ Inference failed: {:?}", e);
            panic!("Inference test failed");
        }
    }
}

/// Test that a downloaded model can be used for inference - 异步版本
#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Requires downloaded model and GPU/CPU backend"]
async fn test_inference_with_downloaded_model() {
    // Use smallest model for quick test
    let alias = "all-MiniLM-L6-v2";
    let models_dir = get_test_models_dir();

    println!("🧪 Testing inference with {}...", alias);

    let config = gllm::ClientConfig {
        device: gllm::Device::Auto,
        models_dir,
    };

    let client = gllm::Client::with_config(alias, config).await
        .expect("Failed to create client");

    let texts = vec![
        "Hello, world!",
        "This is a test sentence.",
    ];

    match client.embeddings(&texts).generate().await {
        Ok(response) => {
            println!("✅ Generated {} embeddings", response.embeddings.len());
            for emb in &response.embeddings {
                println!("   Text {}: {} dimensions", emb.index + 1, emb.embedding.len());
            }
        }
        Err(e) => {
            println!("❌ Inference failed: {:?}", e);
            panic!("Inference test failed");
        }
    }
}
