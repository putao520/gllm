//! Complete Model Test Plan for gllm
//!
//! This module provides comprehensive model testing covering:
//! - 3 model types (Embedding, Rerank, Generator)
//! - Multiple architecture branches
//! - GGUF quantization support
//! - CPU vs GPU performance comparison
//!
//! Run with: cargo test --test integration model_test_plan -- --nocapture

use gllm::{Client, ClientConfig, Device, ModelRegistry, ModelType};
use gllm_kernels::{detect_backend, BackendType};
use std::path::PathBuf;
use std::time::Instant;

fn get_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".gllm")
        .join("models")
}

/// Test Plan Overview:
///
/// ## 1. Embedding Models (Representative Selection)
/// - bge-small-en (BERT, 384d) - smallest, fastest
/// - all-MiniLM-L6-v2 (BERT, 384d) - popular
/// - e5-small (BERT, 384d) - E5 family
///
/// ## 2. Rerank Models (Representative Selection)
/// - bge-reranker-base (BERT cross-encoder)
/// - ms-marco-MiniLM-L-6-v2 (smallest cross-encoder)
///
/// ## 3. Generator Models (All Architecture Branches)
///
/// ### Test Matrix (Optimized for 6GB VRAM):
/// | Architecture         | Model                  | Params | FP16 | GGUF | Note |
/// |---------------------|------------------------|--------|------|------|------|
/// | Qwen3Generator      | qwen3-0.6b             | 0.6B   | ✓    | ✓    | Smallest Qwen3 |
/// | Qwen3Generator      | qwen3-next-0.6b        | 0.6B   | ✓    | ✓    | Latest Qwen3-next (2025) |
/// | MistralGenerator    | ministral-3b-instruct  | 3B     | ✓    | ✓    | Small Mistral (2024) |
/// | Phi4Generator       | phi-4-mini-instruct    | ~3B    | ✗    | ✓    | Smallest Phi-4 |
/// | SmolLM3Generator    | smollm3-3b             | 3B     | ✗    | ✓    | Only size available |
/// | GLM4                | glm-4-9b-chat          | 9B     | ✗    | ✓    | Only size available |
/// | InternLM3Generator  | (excluded)             | -      | -    | ✗    | GGML dtype 23 unsupported |
/// | Qwen3MoE            | (excluded)             | -      | -    | -    | MoE GGUF not available |
///
/// ### Architecture Coverage:
/// - FP16: 3/8 architectures (small models)
/// - GGUF: 6/8 architectures (excludes InternLM3, Qwen3MoE)
/// - Total: 6/8 unique architectures tested
///
/// ## 4. Performance Test Matrix
/// - CPU inference baseline
/// - GPU inference (if available)
/// - FP16 vs GGUF Q4 comparison
/// - Metrics: tokens/s, first token latency

// =============================================================================
// Test 1: Representative Embedding Models
// =============================================================================

#[test]
fn test_embedding_representative() {
    let models = [
        ("bge-small-en", "BERT 384d"),
        ("all-MiniLM-L6-v2", "BERT 384d"),
        ("e5-small", "BERT 384d"),
    ];

    let texts = vec!["Hello world", "Machine learning is great"];

    println!("\n=== Embedding Models (Representative) ===\n");

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    for (alias, desc) in models {
        print!("Testing {} ({})... ", alias, desc);

        let start = Instant::now();
        match Client::with_config(alias, config.clone()) {
            Ok(client) => {
                match client.embeddings(&texts).generate() {
                    Ok(response) => {
                        let elapsed = start.elapsed();
                        let dims = response.embeddings.first()
                            .map(|e| e.embedding.len())
                            .unwrap_or(0);
                        println!("✅ {}d, {:.2}s", dims, elapsed.as_secs_f32());
                    }
                    Err(e) => println!("❌ {}", e),
                }
            }
            Err(e) => println!("❌ {}", e),
        }
    }
}

// =============================================================================
// Test 2: Representative Rerank Models
// =============================================================================

#[test]
fn test_rerank_representative() {
    let models = [
        ("bge-reranker-base", "BERT cross-encoder"),
        ("ms-marco-MiniLM-L-6-v2", "Smallest cross-encoder"),
    ];

    let query = "What is machine learning?";
    let docs = vec![
        "Machine learning is a subset of AI",
        "Python is a programming language",
        "Deep learning uses neural networks",
    ];

    println!("\n=== Rerank Models (Representative) ===\n");

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    for (alias, desc) in models {
        print!("Testing {} ({})... ", alias, desc);

        let start = Instant::now();
        match Client::with_config(alias, config.clone()) {
            Ok(client) => {
                match client.rerank(query, &docs).generate() {
                    Ok(response) => {
                        let elapsed = start.elapsed();
                        let top_score = response.results.first()
                            .map(|r| r.score)
                            .unwrap_or(0.0);
                        println!("✅ top={:.4}, {:.2}s", top_score, elapsed.as_secs_f32());
                    }
                    Err(e) => println!("❌ {}", e),
                }
            }
            Err(e) => println!("❌ {}", e),
        }
    }
}

// =============================================================================
// Test 3: Generator Architecture Coverage (Dense Models)
// =============================================================================

/// Test all dense Generator architectures with smallest available models
#[test]
fn test_generator_dense_architectures() {
    // Models that fit in 6GB VRAM (FP16)
    let dense_models = [
        ("qwen3-0.6b", "Qwen3Generator", "0.6B"),
        ("qwen3-next-0.6b", "Qwen3Generator", "0.6B"),
        ("ministral-3b-instruct", "MistralGenerator", "3B"),
        // These need quantization for 6GB:
        // ("phi-4-mini-instruct", "Phi4Generator", "~3B"),
        // ("smollm3-3b", "SmolLM3Generator", "3B"),
    ];

    let prompt = "Hello, I am";

    println!("\n=== Generator Dense Architectures ===\n");

    let config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    for (alias, arch, params) in dense_models {
        println!("Testing {} ({}, {})...", alias, arch, params);

        let start = Instant::now();
        match Client::with_config(alias, config.clone()) {
            Ok(client) => {
                // Simple generation test
                match client.generate(prompt).max_new_tokens(20).generate() {
                    Ok(response) => {
                        let elapsed = start.elapsed();
                        let output = &response.text;
                        let tokens = output.split_whitespace().count();
                        println!("  ✅ Generated {} tokens in {:.2}s", tokens, elapsed.as_secs_f32());
                        println!("  Output: {}...", &output.chars().take(50).collect::<String>());
                    }
                    Err(e) => println!("  ❌ Generation failed: {}", e),
                }
            }
            Err(e) => println!("  ❌ Client creation failed: {}", e),
        }
        println!();
    }
}

// =============================================================================
// Test 4: GGUF Quantization Support
// =============================================================================

/// Test GGUF quantized model loading and inference
#[test]
fn test_gguf_quantization() {
    println!("\n=== GGUF Quantization Support ===\n");

    let registry = ModelRegistry::new();

    // Test quantization suffix parsing
    let quant_tests = [
        ("qwen3-next-0.6b:gguf", "GGUF format"),
        ("qwen3-0.6b:int4", "INT4 quantization"),
        ("qwen3-0.6b:int8", "INT8 quantization"),
    ];

    println!("1. Quantization suffix parsing:");
    for (alias, desc) in quant_tests {
        match registry.resolve(alias) {
            Ok(info) => {
                println!("  ✅ {} -> {} ({:?})", alias, info.repo_id, info.quantization);
            }
            Err(e) => {
                println!("  ❌ {} -> {}", alias, e);
            }
        }
    }

    // Test actual GGUF model inference (if available)
    println!("\n2. GGUF model inference:");

    let config = ClientConfig {
        device: Device::Cpu, // GGUF typically runs on CPU
        models_dir: get_models_dir(),
    };

    // Try to load a GGUF model
    let gguf_model = "qwen3-next-0.6b:gguf";
    print!("  Testing {}... ", gguf_model);

    match Client::with_config(gguf_model, config) {
        Ok(client) => {
            match client.generate("Hello").max_new_tokens(10).generate() {
                Ok(response) => {
                    println!("✅ Generated: {}...",
                        response.text.chars().take(30).collect::<String>());
                }
                Err(e) => println!("⚠️ Model loaded but inference failed: {}", e),
            }
        }
        Err(e) => println!("⚠️ Not available: {}", e),
    }
}

// =============================================================================
// Test 5: CPU vs GPU Performance Comparison
// =============================================================================

/// Compare CPU and GPU inference performance
#[test]
fn test_cpu_gpu_performance() {
    println!("\n=== CPU vs GPU Performance ===\n");

    let model = "bge-small-en"; // Small embedding model for quick test
    let texts: Vec<&str> = (0..10).map(|_| "The quick brown fox jumps over the lazy dog").collect();

    // CPU Test
    println!("1. CPU Inference:");
    let cpu_config = ClientConfig {
        device: Device::Cpu,
        models_dir: get_models_dir(),
    };

    let cpu_time = match Client::with_config(model, cpu_config) {
        Ok(client) => {
            let start = Instant::now();
            match client.embeddings(&texts).generate() {
                Ok(_) => {
                    let elapsed = start.elapsed();
                    println!("  ✅ {} texts in {:.3}s ({:.1} texts/s)",
                        texts.len(),
                        elapsed.as_secs_f32(),
                        texts.len() as f32 / elapsed.as_secs_f32());
                    Some(elapsed)
                }
                Err(e) => {
                    println!("  ❌ {}", e);
                    None
                }
            }
        }
        Err(e) => {
            println!("  ❌ {}", e);
            None
        }
    };

    // GPU Test (Auto will prefer GPU if available)
    println!("\n2. GPU Inference (Auto):");
    let gpu_config = ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    };

    let gpu_time = match Client::with_config(model, gpu_config) {
        Ok(client) => {
            let start = Instant::now();
            match client.embeddings(&texts).generate() {
                Ok(_) => {
                    let elapsed = start.elapsed();
                    println!("  ✅ {} texts in {:.3}s ({:.1} texts/s)",
                        texts.len(),
                        elapsed.as_secs_f32(),
                        texts.len() as f32 / elapsed.as_secs_f32());
                    Some(elapsed)
                }
                Err(e) => {
                    println!("  ❌ {}", e);
                    None
                }
            }
        }
        Err(e) => {
            println!("  ❌ {}", e);
            None
        }
    };

    // Comparison
    if let (Some(cpu), Some(gpu)) = (cpu_time, gpu_time) {
        let speedup = cpu.as_secs_f32() / gpu.as_secs_f32();
        println!("\n3. Comparison:");
        println!("  CPU: {:.3}s", cpu.as_secs_f32());
        println!("  GPU: {:.3}s", gpu.as_secs_f32());
        println!("  Speedup: {:.2}x", speedup);
    }
}

// =============================================================================
// Test 6: Model Registry Coverage Report
// =============================================================================

/// Generate a coverage report of all registered models by type
#[test]
fn test_model_registry_coverage() {
    println!("\n=== Model Registry Coverage Report ===\n");

    let registry = ModelRegistry::new();

    // Sample models to check
    let test_cases = [
        // Embedding models
        ("bge-small-en", ModelType::Embedding),
        ("bge-small-zh", ModelType::Embedding),
        ("e5-small", ModelType::Embedding),
        ("e5-base", ModelType::Embedding),
        ("e5-large", ModelType::Embedding),
        ("m3e-base", ModelType::Embedding),
        ("all-MiniLM-L6-v2", ModelType::Embedding),
        ("jina-embeddings-v2-small-en", ModelType::Embedding),

        // Rerank models
        ("bge-reranker-base", ModelType::Rerank),
        ("bge-reranker-large", ModelType::Rerank),
        ("bge-reranker-v2", ModelType::Rerank),
        ("ms-marco-MiniLM-L-6-v2", ModelType::Rerank),

        // Generator models (Dense) - Qwen3/Qwen3-next (2025)
        ("qwen3-0.6b", ModelType::Generator),
        ("qwen3-1.7b", ModelType::Generator),
        ("qwen3-next-0.6b", ModelType::Generator),
        ("qwen3-next-2b", ModelType::Generator),
        ("ministral-3b-instruct", ModelType::Generator),
        ("ministral-8b-instruct", ModelType::Generator),

        // Generator models (MoE)
        ("qwen3-30b-a3b", ModelType::Generator),
    ];

    let mut embedding_count = 0;
    let mut rerank_count = 0;
    let mut generator_count = 0;
    let mut failed_count = 0;

    for (alias, expected_type) in test_cases {
        match registry.resolve(alias) {
            Ok(info) => {
                let type_match = info.model_type == expected_type;
                let status = if type_match { "✅" } else { "❌" };

                match info.model_type {
                    ModelType::Embedding => embedding_count += 1,
                    ModelType::Rerank => rerank_count += 1,
                    ModelType::Generator => generator_count += 1,
                }

                println!("{} {} -> {} ({:?})",
                    status, alias, info.repo_id, info.model_type);
            }
            Err(_) => {
                println!("❌ {} -> Not found", alias);
                failed_count += 1;
            }
        }
    }

    println!("\n--- Summary ---");
    println!("Embedding models: {}", embedding_count);
    println!("Rerank models: {}", rerank_count);
    println!("Generator models: {}", generator_count);
    println!("Failed: {}", failed_count);
}

// =============================================================================
// Test 7: All Generator Architecture Branches (FP16 + GGUF)
// =============================================================================

/// Comprehensive test for all Generator architecture branches
/// Tests both FP16 (where possible) and GGUF quantized versions
#[test]
fn test_all_generator_architectures() {
    println!("\n{}", "=".repeat(80));
    println!("=== All Generator Architecture Branches Test ===");
    println!("{}\n", "=".repeat(80));

    // Print detected backend
    let backend = detect_backend();
    println!("🔧 Detected Backend: {} ({})", backend.name(), backend.as_str());
    println!();

    // Architecture test matrix (optimized for 6GB VRAM)
    // (alias, architecture, params, fp16_fits_6gb, has_gguf_repo)
    // GGUF repos: Qwen official, others via bartowski (third-party)
    //
    // EXCLUDED from test matrix:
    // - InternLM3Generator: GGML dtype 23 (TQ1_0/TQ2_0 ternary) not supported
    // - Qwen3MoE: No GGUF repo available for MoE models yet
    let architectures: &[(&str, &str, &str, bool, bool)] = &[
        // Dense Architectures - Qwen3 series (official GGUF, 2025)
        ("qwen3-0.6b", "Qwen3Generator", "0.6B", true, true),
        ("qwen3-next-0.6b", "Qwen3Generator", "0.6B", true, true),
        // Dense Architectures - Ministral (2024, small, perfect for FP16)
        ("ministral-3b-instruct", "MistralGenerator", "3B", true, true),
        // Dense Architectures - Other vendors (bartowski GGUF)
        ("mistral-7b-instruct", "MistralGenerator", "7B", false, true),
        ("phi-4-mini-instruct", "Phi4Generator", "~3B", false, true),
        ("smollm3-3b", "SmolLM3Generator", "3B", false, true),
        // GLM4 (bartowski GGUF)
        ("glm-4-9b-chat", "GLM4", "9B", false, true),
        // NOTE: InternLM3 excluded - GGML dtype 23 unsupported
        // NOTE: Qwen3MoE excluded - no GGUF repo available
    ];

    let prompt = "Hello, I am";
    let models_dir = get_models_dir();

    let mut fp16_passed = 0;
    let mut fp16_failed = 0;
    let mut fp16_skipped = 0;
    let mut gguf_passed = 0;
    let mut gguf_failed = 0;
    let mut gguf_skipped = 0;

    for (alias, arch, params, fp16_fits, has_gguf) in architectures {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📦 {} | {} | {}", arch, alias, params);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Test FP16 (if fits in 6GB VRAM)
        if *fp16_fits {
            print!("  [FP16] Testing {}... ", alias);
            let config = ClientConfig {
                device: Device::Auto,
                models_dir: models_dir.clone(),
            };

            let start = Instant::now();
            match Client::with_config(*alias, config) {
                Ok(client) => {
                    match client.generate(prompt).max_new_tokens(10).generate() {
                        Ok(response) => {
                            let elapsed = start.elapsed();
                            let output_preview: String = response.text.chars().take(40).collect();
                            println!("✅ {:.2}s | Output: {}...", elapsed.as_secs_f32(), output_preview);
                            fp16_passed += 1;
                        }
                        Err(e) => {
                            println!("❌ Generation failed: {}", e);
                            fp16_failed += 1;
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Load failed: {}", e);
                    fp16_failed += 1;
                }
            }
        } else {
            println!("  [FP16] Skipped ({} too large for 6GB VRAM)", params);
            fp16_skipped += 1;
        }

        // Test GGUF (quantized, fits in smaller VRAM)
        if *has_gguf {
            let gguf_alias = format!("{}:gguf", alias);
            print!("  [GGUF] Testing {}... ", gguf_alias);
            let config = ClientConfig {
                device: Device::Auto,
                models_dir: models_dir.clone(),
            };

            let start = Instant::now();
            match Client::with_config(&gguf_alias, config) {
                Ok(client) => {
                    match client.generate(prompt).max_new_tokens(10).generate() {
                        Ok(response) => {
                            let elapsed = start.elapsed();
                            let output_preview: String = response.text.chars().take(40).collect();
                            println!("✅ {:.2}s | Output: {}...", elapsed.as_secs_f32(), output_preview);
                            gguf_passed += 1;
                        }
                        Err(e) => {
                            println!("❌ Generation failed: {}", e);
                            gguf_failed += 1;
                        }
                    }
                }
                Err(e) => {
                    // Check if it's a download error (GGUF repo might not exist)
                    let err_str = format!("{}", e);
                    if err_str.contains("No GGUF file found") || err_str.contains("404") {
                        println!("⚠️ GGUF repo not available");
                        gguf_skipped += 1;
                    } else {
                        println!("❌ Load failed: {}", e);
                        gguf_failed += 1;
                    }
                }
            }
        } else {
            println!("  [GGUF] Skipped (no GGUF repo available)");
            gguf_skipped += 1;
        }
        println!();
    }

    // Summary
    println!("{}", "=".repeat(80));
    println!("=== Summary ===");
    println!("{}", "=".repeat(80));
    println!("🔧 Backend: {}", backend.name());
    println!("📊 FP16:  {} passed, {} failed, {} skipped", fp16_passed, fp16_failed, fp16_skipped);
    println!("📊 GGUF:  {} passed, {} failed, {} skipped", gguf_passed, gguf_failed, gguf_skipped);
    println!("📊 Total: {} passed, {} failed", fp16_passed + gguf_passed, fp16_failed + gguf_failed);

    // Backend verification
    println!("\n🔍 Backend Auto-Selection Verification:");
    match backend {
        BackendType::Cuda => println!("  ✅ CUDA detected - using NVIDIA GPU acceleration"),
        BackendType::Rocm => println!("  ✅ ROCm detected - using AMD GPU acceleration"),
        BackendType::Metal => println!("  ✅ Metal detected - using Apple GPU acceleration"),
        BackendType::Wgpu => println!("  ✅ WGPU detected - using cross-platform GPU"),
        BackendType::Cpu => println!("  ⚠️ CPU fallback - no GPU detected or available"),
    }
}
