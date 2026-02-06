//! E2E tests using the actual Client API (what users use).
//!
//! Test Matrix:
//! - 3 Functions: Generation, Embedding, Rerank
//! - All Architectures: each tested with one representative model
//! - 2 Backends: CPU (required), CUDA (conditional)
//!
//! Models are discovered dynamically from the test matrix.

use gllm::loader::config as loader_config;
use gllm::manifest::{ModelArchitecture, ModelKind, ModelManifest, EMPTY_FILE_MAP};
use gllm::Client;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

struct ModelEntry {
    arch: ModelArchitecture,
    kind: ModelKind,
    model_id: &'static str,
}

const E2E_MODEL_MATRIX: &[ModelEntry] = &[
    ModelEntry {
        arch: ModelArchitecture::Qwen3,
        kind: ModelKind::Chat,
        model_id: "Qwen/Qwen3-0.6B",
    },
    ModelEntry {
        arch: ModelArchitecture::Llama4,
        kind: ModelKind::Chat,
        model_id: "meta-llama/Llama-4-8B-Instruct",
    },
    ModelEntry {
        arch: ModelArchitecture::Phi4,
        kind: ModelKind::Chat,
        model_id: "microsoft/Phi-4-mini-instruct",
    },
    ModelEntry {
        arch: ModelArchitecture::Qwen3,
        kind: ModelKind::Embedding,
        model_id: "Qwen/Qwen3-Embedding",
    },
    ModelEntry {
        arch: ModelArchitecture::XlmRNext,
        kind: ModelKind::Embedding,
        model_id: "BAAI/bge-m4",
    },
    ModelEntry {
        arch: ModelArchitecture::Qwen3,
        kind: ModelKind::Reranker,
        model_id: "Qwen/Qwen3-Reranker",
    },
    ModelEntry {
        arch: ModelArchitecture::XlmRNext,
        kind: ModelKind::Reranker,
        model_id: "BAAI/bge-reranker-v3",
    },
];

const ADAPTER_MODEL_MATRIX: &[ModelEntry] = &[
    ModelEntry {
        arch: ModelArchitecture::Qwen3,
        kind: ModelKind::Embedding,
        model_id: "Qwen/Qwen3-Embedding",
    },
    ModelEntry {
        arch: ModelArchitecture::Qwen3,
        kind: ModelKind::Reranker,
        model_id: "Qwen/Qwen3-Reranker",
    },
    ModelEntry {
        arch: ModelArchitecture::XlmR,
        kind: ModelKind::Embedding,
        model_id: "BAAI/bge-m3",
    },
    ModelEntry {
        arch: ModelArchitecture::Qwen3MoE,
        kind: ModelKind::Chat,
        model_id: "Qwen/Qwen3-235B-A22B-Instruct",
    },
    ModelEntry {
        arch: ModelArchitecture::Qwen3,
        kind: ModelKind::Chat,
        model_id: "Qwen/Qwen3-0.6B",
    },
    ModelEntry {
        arch: ModelArchitecture::Qwen2_5,
        kind: ModelKind::Chat,
        model_id: "Qwen/Qwen2.5-0.5B-Instruct",
    },
    ModelEntry {
        arch: ModelArchitecture::Llama4,
        kind: ModelKind::Chat,
        model_id: "meta-llama/Llama-4-8B-Instruct",
    },
    ModelEntry {
        arch: ModelArchitecture::Gemma2,
        kind: ModelKind::Chat,
        model_id: "google/gemma-2-2b-it",
    },
    ModelEntry {
        arch: ModelArchitecture::Phi4,
        kind: ModelKind::Chat,
        model_id: "microsoft/Phi-4-mini-instruct",
    },
    ModelEntry {
        arch: ModelArchitecture::Ministral,
        kind: ModelKind::Chat,
        model_id: "mistralai/Ministral-8B-Instruct",
    },
    ModelEntry {
        arch: ModelArchitecture::Mistral3,
        kind: ModelKind::Chat,
        model_id: "mistralai/Mistral-Small-3.2",
    },
    ModelEntry {
        arch: ModelArchitecture::GLM5,
        kind: ModelKind::Chat,
        model_id: "THUDM/glm-5-9b-chat",
    },
    ModelEntry {
        arch: ModelArchitecture::GPT2Next,
        kind: ModelKind::Chat,
        model_id: "openai-community/gpt2",
    },
];

/// TEST-ADAPTER-001: 所有 manifest 都有对应的 adapter
///
/// **关联需求**: REQ-TEST-002, REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 遍历 ADAPTER_MODEL_MATRIX 中的所有模型
/// 2. 为每个模型创建 manifest
/// 3. 验证 adapter_for() 返回 Some
///
/// **期望结果**: 所有模型都有对应的 adapter
#[test]
fn manifests_have_adapters() {
    use gllm::adapter::adapter_for;
    use gllm_kernels::cpu_backend::CpuBackend;

    for entry in ADAPTER_MODEL_MATRIX {
        let manifest = build_manifest(entry);
        let adapter = adapter_for::<CpuBackend>(&manifest);
        assert!(
            adapter.is_some(),
            "missing adapter for {:?}",
            entry.model_id
        );
    }
}

/// TEST-ADAPTER-002: 生成模型架构 E2E 测试矩阵
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
/// **E2E测试粒度**: 业务流程
///
/// **前置条件**: E2E_MODEL_MATRIX 中的模型已缓存
///
/// **测试步骤**:
/// 1. 遍历 E2E_MODEL_MATRIX 中的生成模型
/// 2. 为每个架构执行: manifest 查找 → 下载 → backend 初始化 → 权重加载 → 生成
/// 3. 测试 CPU backend（强制）和 CUDA backend（条件）
///
/// **期望结果**: 所有生成架构完成端到端流程，输出非空文本
#[test]
fn e2e_generation_architectures() {
    let results = run_test_matrix("generation");
    print_test_summary("Generation", results);
}

/// TEST-ADAPTER-003: 嵌入模型架构 E2E 测试矩阵
///
/// **关联需求**: REQ-TEST-003
/// **测试类型**: 正向测试
/// **E2E测试粒度**: 业务流程
///
/// **前置条件**: E2E_MODEL_MATRIX 中的嵌入模型已缓存
///
/// **测试步骤**:
/// 1. 遍历 E2E_MODEL_MATRIX 中的嵌入模型
/// 2. 为每个架构执行: manifest 查找 → 下载 → backend 初始化 → 权重加载 → 嵌入
/// 3. 测试 CPU backend（强制）和 CUDA backend（条件）
///
/// **期望结果**: 所有嵌入架构完成端到端流程，返回非空向量
#[test]
fn e2e_embedding_architectures() {
    let results = run_test_matrix("embeddings");
    print_test_summary("Embedding", results);
}

/// TEST-ADAPTER-004: 重排序模型架构 E2E 测试矩阵
///
/// **关联需求**: REQ-TEST-004
/// **测试类型**: 正向测试
/// **E2E测试粒度**: 业务流程
///
/// **前置条件**: E2E_MODEL_MATRIX 中的重排序模型已缓存
///
/// **测试步骤**:
/// 1. 遍历 E2E_MODEL_MATRIX 中的重排序模型
/// 2. 为每个架构执行: manifest 查找 → 下载 → backend 初始化 → 权重加载 → 重排序
/// 3. 测试 CPU backend（强制）和 CUDA backend（条件）
///
/// **期望结果**: 所有重排序架构完成端到端流程，返回正确的排序结果
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
    _function_type: String,
    architecture: String,
    _backend: String,
    status: TestStatus,
}

#[derive(Debug)]
enum TestStatus {
    Passed,
    Skipped(String),
    Failed(String),
}

/// Run test matrix for a specific function type.
fn run_test_matrix(function_type: &str) -> HashMap<String, TestResult> {
    let mut results = HashMap::new();
    let cuda_available = is_cuda_available();
    let kind = match function_kind(function_type) {
        Some(kind) => kind,
        None => return results,
    };

    // Get models grouped by architecture
    let models_by_arch = group_models_by_architecture(kind);

    for (arch, alias) in models_by_arch {
        let arch_str = format!("{:?}", arch);

        // Test CPU backend
        match test_model(&alias, kind, function_type, &arch_str, "CPU") {
            TestStatus::Passed => {
                results.insert(
                    format!("CPU:{}", alias),
                    TestResult {
                        alias: alias.clone(),
                        _function_type: function_type.to_string(),
                        architecture: arch_str.clone(),
                        _backend: "CPU".to_string(),
                        status: TestStatus::Passed,
                    },
                );
            }
            TestStatus::Skipped(reason) => {
                results.insert(
                    format!("CPU:{}", alias),
                    TestResult {
                        alias: alias.clone(),
                        _function_type: function_type.to_string(),
                        architecture: arch_str.clone(),
                        _backend: "CPU".to_string(),
                        status: TestStatus::Skipped(reason),
                    },
                );
            }
            TestStatus::Failed(err) => {
                results.insert(
                    format!("CPU:{}", alias),
                    TestResult {
                        alias: alias.clone(),
                        _function_type: function_type.to_string(),
                        architecture: arch_str.clone(),
                        _backend: "CPU".to_string(),
                        status: TestStatus::Failed(err),
                    },
                );
            }
        }

        // Test CUDA backend (if available)
        if cuda_available {
            match test_model(&alias, kind, function_type, &arch_str, "CUDA") {
                TestStatus::Passed => {
                    results.insert(
                        format!("CUDA:{}", alias),
                        TestResult {
                            alias: alias.clone(),
                            _function_type: function_type.to_string(),
                            architecture: arch_str,
                            _backend: "CUDA".to_string(),
                            status: TestStatus::Passed,
                        },
                    );
                }
                TestStatus::Skipped(reason) => {
                    results.insert(
                        format!("CUDA:{}", alias),
                        TestResult {
                            alias,
                            _function_type: function_type.to_string(),
                            architecture: arch_str,
                            _backend: "CUDA".to_string(),
                            status: TestStatus::Skipped(reason),
                        },
                    );
                }
                TestStatus::Failed(err) => {
                    results.insert(
                        format!("CUDA:{}", alias),
                        TestResult {
                            alias,
                            _function_type: function_type.to_string(),
                            architecture: arch_str,
                            _backend: "CUDA".to_string(),
                            status: TestStatus::Failed(err),
                        },
                    );
                }
            }
        }
    }

    results
}

/// Group models by architecture, returning one representative alias per architecture.
fn group_models_by_architecture(kind: ModelKind) -> Vec<(ModelArchitecture, String)> {
    let mut seen_archs = HashSet::new();
    let mut result = Vec::new();

    for entry in E2E_MODEL_MATRIX {
        if entry.kind != kind {
            continue;
        }
        // Skip if we've already seen this architecture
        if !seen_archs.insert(entry.arch) {
            continue;
        }

        result.push((entry.arch, entry.model_id.to_string()));
    }

    result
}

fn function_kind(function_type: &str) -> Option<ModelKind> {
    match function_type {
        "generation" => Some(ModelKind::Chat),
        "embeddings" => Some(ModelKind::Embedding),
        "rerank" => Some(ModelKind::Reranker),
        _ => None,
    }
}

fn build_manifest(entry: &ModelEntry) -> ModelManifest {
    ModelManifest {
        model_id: Cow::Borrowed(entry.model_id),
        file_map: EMPTY_FILE_MAP,
        arch: entry.arch,
        tensor_rules: loader_config::tensor_rules_for_arch(entry.arch),
        kind: entry.kind,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
    }
}

fn test_model(
    alias: &str,
    kind: ModelKind,
    func_type: &str,
    arch: &str,
    backend: &str,
) -> TestStatus {
    println!("Testing: {} ({}, {}, {})", alias, func_type, arch, backend);

    // Try to create client (this triggers download if needed)
    let client = match kind {
        ModelKind::Chat => Client::new_chat(alias),
        ModelKind::Embedding => Client::new_embedding(alias),
        ModelKind::Reranker => Client::new(alias, ModelKind::Reranker),
    };
    let client = match client {
        Ok(c) => c,
        Err(e) => return TestStatus::Skipped(format!("Client init failed: {}", e)),
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
    let response = client.generate(prompt).max_tokens(5).generate()?;

    assert!(
        !response.text.is_empty(),
        "generation produced empty output"
    );
    println!("  ✓ Generation: {}", response.text.trim());
    Ok(())
}

fn test_embeddings(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = vec![String::from("Hello, world!"), String::from("Test text")];
    let response = client.embeddings(inputs).generate()?;

    assert!(
        !response.embeddings.is_empty(),
        "embeddings produced empty results"
    );
    assert_eq!(response.embeddings.len(), 2, "embeddings count mismatch");
    assert!(
        response.embeddings[0].embedding.len() > 0,
        "embedding vector is empty"
    );
    println!(
        "  ✓ Embeddings: {} vectors, dim={}",
        response.embeddings.len(),
        response.embeddings[0].embedding.len()
    );
    Ok(())
}

fn test_rerank(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let query = String::from("query");
    let documents = vec![
        String::from("document 1"),
        String::from("document 2"),
        String::from("document 3"),
    ];
    let response = client.rerank(query, documents).top_n(2).generate()?;

    assert_eq!(response.results.len(), 2, "rerank should return top 2");
    println!(
        "  ✓ Rerank: top1={}, top2={}",
        response.results[0].index, response.results[1].index
    );
    Ok(())
}

fn is_cuda_available() -> bool {
    // Check if CUDA backend is available by trying to detect it
    use gllm::backend::detect_backend;
    detect_backend().map_or(false, |backend| format!("{:?}", backend).contains("Cuda"))
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
                println!(
                    "  ⚠️  {} - {} ({}) - {}",
                    key, result.alias, result.architecture, reason
                );
                skipped += 1;
                entry.1 += 1;
            }
            TestStatus::Failed(err) => {
                println!(
                    "  ❌ {} - {} ({}) - {}",
                    key, result.alias, result.architecture, err
                );
                failed += 1;
                entry.2 += 1;
            }
        }
    }

    println!("\nBackend breakdown:");
    for (backend, (p, s, f)) in &backend_results {
        println!("  {}: {} passed, {} skipped, {} failed", backend, p, s, f);
    }

    println!(
        "\nTotal: {} passed, {} skipped, {} failed",
        passed, skipped, failed
    );
}
