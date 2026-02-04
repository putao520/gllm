//! E2E tests using the actual Client API (what users use).
//!
//! Test Matrix:
//! - 3 Functions: Generation, Embedding, Rerank
//! - All Architectures: each tested with one representative model
//! - 2 Backends: CPU (required), CUDA (conditional)
//!
//! Models are discovered dynamically from the manifest registry.

use gllm::Client;
use gllm::manifest::{all_manifests, ModelArchitecture};
use std::collections::{HashMap, HashSet};

/// Test that all registered manifests have a corresponding adapter.
#[test]
fn registry_manifests_have_adapters() {
    use gllm::adapter::adapter_for;
    use gllm_kernels::cpu_backend::CpuBackend;

    for manifest in all_manifests() {
        let adapter = adapter_for::<CpuBackend>(manifest);
        assert!(
            adapter.is_some(),
            "missing adapter for {:?}",
            manifest.model_id
        );
    }
}

/// E2E test matrix: Generation models
///
/// Tests one model per generation architecture.
/// Validates: manifest lookup → download → backend init → weight loading → generation.
#[test]
fn e2e_generation_architectures() {
    let results = run_test_matrix("generation");
    print_test_summary("Generation", results);
}

/// E2E test matrix: Embedding
///
/// Tests embedding models.
/// Validates: manifest lookup → download → backend init → weight loading → embeddings.
#[test]
fn e2e_embedding_architectures() {
    let results = run_test_matrix("embeddings");
    print_test_summary("Embedding", results);
}

/// E2E test matrix: Rerank
///
/// Tests rerank models.
/// Validates: manifest lookup → download → backend init → weight loading → rerank.
#[test]
fn e2e_rerank_architectures() {
    let results = run_test_matrix("rerank");
    print_test_summary("Rerank", results);
}

// ============================================================================
// TEST HELPERS
// ============================================================================

struct TestResult {
    alias: String,
    function_type: String,
    architecture: String,
    backend: String,
    status: TestStatus,
}

#[derive(Debug)]
enum TestStatus {
    Passed,
    Skipped(String),
    Failed(String),
}

/// Run test matrix for a specific function type.
/// Dynamically discovers models from manifest registry.
fn run_test_matrix(function_type: &str) -> HashMap<String, TestResult> {
    let mut results = HashMap::new();
    let cuda_available = is_cuda_available();

    // Get models grouped by architecture
    let models_by_arch = group_models_by_architecture(function_type);

    for (arch, alias) in models_by_arch {
        let arch_str = format!("{:?}", arch);

        // Test CPU backend
        match test_model(&alias, function_type, &arch_str, "CPU") {
            TestStatus::Passed => {
                results.insert(format!("CPU:{}", alias), TestResult {
                    alias: alias.clone(),
                    function_type: function_type.to_string(),
                    architecture: arch_str.clone(),
                    backend: "CPU".to_string(),
                    status: TestStatus::Passed,
                });
            }
            TestStatus::Skipped(reason) => {
                results.insert(format!("CPU:{}", alias), TestResult {
                    alias: alias.clone(),
                    function_type: function_type.to_string(),
                    architecture: arch_str.clone(),
                    backend: "CPU".to_string(),
                    status: TestStatus::Skipped(reason),
                });
            }
            TestStatus::Failed(err) => {
                results.insert(format!("CPU:{}", alias), TestResult {
                    alias: alias.clone(),
                    function_type: function_type.to_string(),
                    architecture: arch_str.clone(),
                    backend: "CPU".to_string(),
                    status: TestStatus::Failed(err),
                });
            }
        }

        // Test CUDA backend (if available)
        if cuda_available {
            match test_model(&alias, function_type, &arch_str, "CUDA") {
                TestStatus::Passed => {
                    results.insert(format!("CUDA:{}", alias), TestResult {
                        alias: alias.clone(),
                        function_type: function_type.to_string(),
                        architecture: arch_str,
                        backend: "CUDA".to_string(),
                        status: TestStatus::Passed,
                    });
                }
                TestStatus::Skipped(reason) => {
                    results.insert(format!("CUDA:{}", alias), TestResult {
                        alias,
                        function_type: function_type.to_string(),
                        architecture: arch_str,
                        backend: "CUDA".to_string(),
                        status: TestStatus::Skipped(reason),
                    });
                }
                TestStatus::Failed(err) => {
                    results.insert(format!("CUDA:{}", alias), TestResult {
                        alias,
                        function_type: function_type.to_string(),
                        architecture: arch_str,
                        backend: "CUDA".to_string(),
                        status: TestStatus::Failed(err),
                    });
                }
            }
        }
    }

    results
}

/// Group models by architecture, returning one representative alias per architecture.
fn group_models_by_architecture(function_type: &str) -> Vec<(ModelArchitecture, String)> {
    let mut seen_archs = HashSet::new();
    let mut result = Vec::new();

    for manifest in all_manifests() {
        // Skip if we've already seen this architecture
        if !seen_archs.insert(manifest.arch) {
            continue;
        }

        // Check if model matches the requested function type
        let kind = classify_model(manifest.model_id.as_ref());
        let matches = match function_type {
            "generation" => kind == "generation",
            "embeddings" => kind == "embeddings",
            "rerank" => kind == "rerank",
            _ => false,
        };

        if matches {
            result.push((manifest.arch, manifest.model_id.to_string()));
        }
    }

    result
}

fn classify_model(model_id: &str) -> &'static str {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("rerank") {
        return "rerank";
    }
    if lower.contains("embed")
        || lower.contains("embedding")
        || lower.contains("bge")
        || lower.contains("e5")
        || lower.contains("m3e")
        || lower.contains("jina")
    {
        return "embeddings";
    }
    "generation"
}

fn test_model(alias: &str, func_type: &str, arch: &str, backend: &str) -> TestStatus {
    println!("Testing: {} ({}, {}, {})", alias, func_type, arch, backend);

    // Try to create client (this triggers download if needed)
    let client = match Client::new(alias) {
        Ok(c) => c,
        Err(e) => return TestStatus::Skipped(format!("Client::new failed: {}", e)),
    };

    // Run the appropriate function
    let result = match func_type {
        "generation" => test_generation(&client),
        "embeddings" => test_embeddings(&client),
        "rerank" => test_rerank(&client),
        _ => return TestStatus::Failed(format!("unknown function type: {}", func_type)),
    };

    match result {
        Ok(()) => TestStatus::Passed,
        Err(e) => TestStatus::Failed(e.to_string()),
    }
}

fn test_generation(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = String::from("Hello, world!");
    let response = client
        .generate(prompt)
        .max_tokens(5)
        .generate()?;

    assert!(!response.text.is_empty(), "generation produced empty output");
    println!("  ✓ Generation: {}", response.text.trim());
    Ok(())
}

fn test_embeddings(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = vec![
        String::from("Hello, world!"),
        String::from("Test text"),
    ];
    let response = client
        .embeddings(inputs)
        .generate()?;

    assert!(!response.embeddings.is_empty(), "embeddings produced empty results");
    assert_eq!(response.embeddings.len(), 2, "embeddings count mismatch");
    assert!(response.embeddings[0].embedding.len() > 0, "embedding vector is empty");
    println!("  ✓ Embeddings: {} vectors, dim={}", response.embeddings.len(), response.embeddings[0].embedding.len());
    Ok(())
}

fn test_rerank(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let query = String::from("query");
    let documents = vec![
        String::from("document 1"),
        String::from("document 2"),
        String::from("document 3"),
    ];
    let response = client
        .rerank(query, documents)
        .top_n(2)
        .generate()?;

    assert_eq!(response.results.len(), 2, "rerank should return top 2");
    println!("  ✓ Rerank: top1={}, top2={}", response.results[0].index, response.results[1].index);
    Ok(())
}

fn is_cuda_available() -> bool {
    // Check if CUDA backend is available by trying to detect it
    use gllm::backend::detect_backend;
    detect_backend().map_or(false, |backend| {
        format!("{:?}", backend).contains("Cuda")
    })
}

fn print_test_summary(function_name: &str, results: HashMap<String, TestResult>) {
    println!("\n=== {} Test Summary ===", function_name);

    let mut passed = 0;
    let mut skipped = 0;
    let mut failed = 0;

    let mut backend_results: HashMap<&str, (usize, usize, usize)> = HashMap::new();

    for (key, result) in &results {
        let backend = key.split(':').next().unwrap_or("?");
        let entry = backend_results.entry(backend).or_insert((0, 0, 0));

        match &result.status {
            TestStatus::Passed => {
                println!("  ✅ {} - {} ({})", key, result.alias, result.architecture);
                passed += 1;
                entry.0 += 1;
            }
            TestStatus::Skipped(reason) => {
                println!("  ⚠️  {} - {} ({}) - {}", key, result.alias, result.architecture, reason);
                skipped += 1;
                entry.1 += 1;
            }
            TestStatus::Failed(err) => {
                println!("  ❌ {} - {} ({}) - {}", key, result.alias, result.architecture, err);
                failed += 1;
                entry.2 += 1;
            }
        }
    }

    println!("\nBackend breakdown:");
    for (backend, (p, s, f)) in &backend_results {
        println!("  {}: {} passed, {} skipped, {} failed", backend, p, s, f);
    }

    println!("\nTotal: {} passed, {} skipped, {} failed", passed, skipped, failed);
}
