//! Regression Test Suite for gllm
//!
//! Tests all supported models across different architectures:
//! - Generator models: Qwen3, GPT-OSS, Phi-4, Gemma-2
//! - Embedding models: Qwen3-Embed, BGE-M3, BGE-M4
//! - Rerank models: Qwen3-Rerank, BGE-Rerank-V3
//!
//! Usage:
//!   cargo run --release --example regression -- --cuda    # Use CUDA backend
//!   cargo run --release --example regression              # Use CPU backend (auto)
//!   cargo run --release --example regression -- --loader  # Include PyTorch bin loader check

use gllm::kv_cache::{KvCacheDoubleBuffer, KvCacheState};
#[cfg(feature = "candle")]
use gllm::loader::convert_bins_to_safetensors;
#[cfg(feature = "candle")]
use gllm::loader::PytorchConversionConfig;
use gllm::scheduler::{GroupState, HGALConfig, PagedScheduler, SequenceGroup};
use gllm::{Client, ModelKind};
use gllm::compat::backend_trait::Backend;
use gllm::compat::CpuBackend;
use gllm::engine::KvCacheConfig;
#[cfg(feature = "candle")]
use hf_hub::api::sync::ApiBuilder;
#[cfg(feature = "candle")]
use safetensors::SafeTensors;
use std::time::Instant;

/// Test configuration for a single model
struct ModelTest {
    alias: &'static str,
    model_id: &'static str,
    model_type: ModelType,
    test_prompt: &'static str,
    min_tokens: usize,
}

#[derive(Debug)]
enum ModelType {
    Generator,
    Embedding,
    Rerank,
}

/// All models to test
const MODEL_TESTS: &[ModelTest] = &[
    // === Generator Models (with working adapters) ===
    ModelTest {
        alias: "Qwen/Qwen3-7B-Instruct",
        model_id: "qwen3-7b",
        model_type: ModelType::Generator,
        test_prompt: "What is the capital of France?",
        min_tokens: 10,
    },
    ModelTest {
        alias: "openai/gpt-oss-1.5b",
        model_id: "gpt-oss-1.5b",
        model_type: ModelType::Generator,
        test_prompt: "Complete: The sky is",
        min_tokens: 5,
    },
    // === Generator Models (NEW - need adapters) ===
    ModelTest {
        alias: "microsoft/Phi-4-mini-instruct",
        model_id: "phi-4-mini",
        model_type: ModelType::Generator,
        test_prompt: "What is 2+2?",
        min_tokens: 5,
    },
    ModelTest {
        alias: "google/gemma-2-2b-it",
        model_id: "gemma-2-2b-it",
        model_type: ModelType::Generator,
        test_prompt: "Hello, how are you?",
        min_tokens: 10,
    },
    // === Embedding Models ===
    ModelTest {
        alias: "Qwen/Qwen3-Embedding",
        model_id: "qwen3-embed",
        model_type: ModelType::Embedding,
        test_prompt: "This is a test sentence for embedding.",
        min_tokens: 1,
    },
    // === Embedding Models (NEW - need adapters) ===
    ModelTest {
        alias: "BAAI/bge-m3",
        model_id: "bge-m3",
        model_type: ModelType::Embedding,
        test_prompt: "测试文本嵌入向量生成。", // Chinese text test
        min_tokens: 1,
    },
    ModelTest {
        alias: "BAAI/bge-m4",
        model_id: "bge-m4",
        model_type: ModelType::Embedding,
        test_prompt: "Multilingual embedding test.",
        min_tokens: 1,
    },
    // === Rerank Models ===
    ModelTest {
        alias: "Qwen/Qwen3-Reranker",
        model_id: "qwen3-rerank",
        model_type: ModelType::Rerank,
        test_prompt: "What is machine learning?",
        min_tokens: 1,
    },
    ModelTest {
        alias: "BAAI/bge-reranker-v3",
        model_id: "bge-rerank-v3",
        model_type: ModelType::Rerank,
        test_prompt: "machine learning",
        min_tokens: 1,
    },
];

struct TestResult {
    alias: String,
    model_id: String,
    passed: bool,
    duration_ms: u64,
    error: Option<String>,
}

struct InternalCheck {
    name: &'static str,
    passed: bool,
    error: Option<String>,
}

impl std::fmt::Display for InternalCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "✅ PASS" } else { "❌ FAIL" };
        write!(f, "{} | {}", status, self.name)?;
        if let Some(err) = &self.error {
            write!(f, " | {}", err)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for TestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "✅ PASS" } else { "❌ FAIL" };
        write!(
            f,
            "{} | {:<20} | {:8} ms",
            status, self.alias, self.duration_ms
        )?;
        if let Some(err) = &self.error {
            write!(f, " | {}", err)?;
        }
        Ok(())
    }
}

fn test_generator_model(test: &ModelTest, _use_cuda: bool) -> TestResult {
    let start = Instant::now();
    let alias = test.alias.to_string();
    let model_id = test.model_id.to_string();
    let alias_clone = alias.clone();
    let model_id_clone = model_id.clone();

    let result = std::panic::catch_unwind(|| {
        // Try to load and run the model
        let client = match Client::new_chat(test.alias) {
            Ok(c) => c,
            Err(e) => {
                return TestResult {
                    alias: alias_clone.clone(),
                    model_id: model_id_clone.clone(),
                    passed: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Load failed: {}", e)),
                };
            }
        };

        // Check backend info
        let manifest = match client.manifest() {
            Ok(m) => m,
            Err(e) => {
                return TestResult {
                    alias: alias_clone.clone(),
                    model_id: model_id_clone.clone(),
                    passed: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Read manifest failed: {}", e)),
                };
            }
        };
        println!("    Architecture: {:?}", manifest.arch);

        // Run generation
        let response = match client
            .generate(test.test_prompt)
            .max_tokens(50)
            .temperature(0.7)
            .generate()
        {
            Ok(r) => r,
            Err(e) => {
                return TestResult {
                    alias: alias_clone.clone(),
                    model_id: model_id_clone.clone(),
                    passed: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Generate failed: {}", e)),
                };
            }
        };

        let generated = response.text.trim();
        println!("    Generated: {}", generated);

        TestResult {
            alias: alias_clone.clone(),
            model_id: model_id_clone.clone(),
            passed: !generated.is_empty(),
            duration_ms: start.elapsed().as_millis() as u64,
            error: None,
        }
    });

    result.unwrap_or_else(|_| TestResult {
        alias,
        model_id,
        passed: false,
        duration_ms: start.elapsed().as_millis() as u64,
        error: Some("Panic during test".to_string()),
    })
}

fn test_embedding_model(test: &ModelTest, _use_cuda: bool) -> TestResult {
    let start = Instant::now();
    let alias = test.alias.to_string();
    let model_id = test.model_id.to_string();
    let alias_clone = alias.clone();
    let model_id_clone = model_id.clone();

    let result = std::panic::catch_unwind(|| {
        let client = match Client::new_embedding(test.alias) {
            Ok(c) => c,
            Err(e) => {
                return TestResult {
                    alias: alias_clone.clone(),
                    model_id: model_id_clone.clone(),
                    passed: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Load failed: {}", e)),
                };
            }
        };

        let embeddings = match client
            .embeddings(vec![test.test_prompt.to_string()])
            .generate()
        {
            Ok(e) => e,
            Err(e) => {
                return TestResult {
                    alias: alias_clone.clone(),
                    model_id: model_id_clone.clone(),
                    passed: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Embed failed: {}", e)),
                };
            }
        };

        let has_embedding = !embeddings.embeddings.is_empty()
            && embeddings
                .embeddings
                .first()
                .map(|e| !e.embedding.is_empty())
                .unwrap_or(false);

        if let Some(emb) = embeddings.embeddings.first() {
            println!("    Embedding dim: {}", emb.embedding.len());
        }

        TestResult {
            alias: alias_clone.clone(),
            model_id: model_id_clone.clone(),
            passed: has_embedding,
            duration_ms: start.elapsed().as_millis() as u64,
            error: None,
        }
    });

    result.unwrap_or_else(|_| TestResult {
        alias,
        model_id,
        passed: false,
        duration_ms: start.elapsed().as_millis() as u64,
        error: Some("Panic during test".to_string()),
    })
}

fn test_rerank_model(test: &ModelTest, _use_cuda: bool) -> TestResult {
    let start = Instant::now();
    let alias = test.alias.to_string();
    let model_id = test.model_id.to_string();
    let alias_clone = alias.clone();
    let model_id_clone = model_id.clone();

    let result = std::panic::catch_unwind(|| {
        let client = match Client::new(test.alias, ModelKind::Reranker) {
            Ok(c) => c,
            Err(e) => {
                return TestResult {
                    alias: alias_clone.clone(),
                    model_id: model_id_clone.clone(),
                    passed: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Load failed: {}", e)),
                };
            }
        };

        let documents = vec![
            "Machine learning is a subset of artificial intelligence.".to_string(),
            "Paris is the capital of France.".to_string(),
            "The sky is blue during the day.".to_string(),
        ];

        let results = match client.rerank(test.test_prompt, documents).generate() {
            Ok(r) => r,
            Err(e) => {
                return TestResult {
                    alias: alias_clone.clone(),
                    model_id: model_id_clone.clone(),
                    passed: false,
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("Rerank failed: {}", e)),
                };
            }
        };

        let has_results = !results.results.is_empty();

        if has_results {
            println!(
                "    Top result: index={}, score={:.4}",
                results.results[0].index, results.results[0].score
            );
        }

        TestResult {
            alias: alias_clone.clone(),
            model_id: model_id_clone.clone(),
            passed: has_results,
            duration_ms: start.elapsed().as_millis() as u64,
            error: None,
        }
    });

    result.unwrap_or_else(|_| TestResult {
        alias,
        model_id,
        passed: false,
        duration_ms: start.elapsed().as_millis() as u64,
        error: Some("Panic during test".to_string()),
    })
}

fn run_regression_test(use_cuda: bool, filter: Option<&str>) -> Vec<TestResult> {
    let mut results = Vec::new();

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  gllm Regression Test Suite");
    println!(
        "  Backend: {}",
        if use_cuda { "CUDA" } else { "CPU (Auto)" }
    );
    if let Some(f) = filter {
        println!("  Filter: {}", f);
    }
    println!("════════════════════════════════════════════════════════════════\n");

    for test in MODEL_TESTS {
        // Apply filter if provided
        if let Some(f) = filter {
            if !test.alias.contains(f) && !test.model_id.contains(f) {
                continue;
            }
        }

        println!("Testing: {} ({})", test.alias, test.model_id);
        println!("  Type: {:?}", test.model_type);
        println!("  Prompt: {}", test.test_prompt);
        println!("  Min tokens: {}", test.min_tokens);

        let result = match test.model_type {
            ModelType::Generator => test_generator_model(test, use_cuda),
            ModelType::Embedding => test_embedding_model(test, use_cuda),
            ModelType::Rerank => test_rerank_model(test, use_cuda),
        };

        println!("  {}\n", result);
        results.push(result);
    }

    results
}

fn print_summary(results: &[TestResult]) {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;

    let total_time: u64 = results.iter().map(|r| r.duration_ms).sum();

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("════════════════════════════════════════════════════════════════");
    println!("  Total:  {} tests", total);
    println!("  Passed: {} ✅", passed);
    println!("  Failed: {} ❌", failed);
    println!("  Time:   {} ms", total_time);
    println!("════════════════════════════════════════════════════════════════");

    if failed > 0 {
        println!("\nFailed tests:");
        for r in results.iter().filter(|r| !r.passed) {
            println!("  - {} ({}) : {:?}", r.alias, r.model_id, r.error);
        }
    }

    let exit_code = if failed == 0 { 0 } else { 1 };
    std::process::exit(exit_code);
}

fn scheduler_check() -> InternalCheck {
    let name = "PagedAttention scheduler";
    let mut scheduler = PagedScheduler::new(4, 4, HGALConfig::default());
    let now = Instant::now();
    let group_a = SequenceGroup {
        id: 1,
        pages: Vec::new(),
        context_len: 5,
        state: GroupState::Running,
        access_count: 0,
        last_access: now,
        is_pinned: false,
    };
    let group_b = SequenceGroup {
        id: 2,
        pages: Vec::new(),
        context_len: 3,
        state: GroupState::Running,
        access_count: 0,
        last_access: now,
        is_pinned: false,
    };
    if let Err(err) = scheduler.add_sequence(group_a) {
        return InternalCheck {
            name,
            passed: false,
            error: Some(format!("add sequence A failed: {err}")),
        };
    }
    if let Err(err) = scheduler.add_sequence(group_b) {
        return InternalCheck {
            name,
            passed: false,
            error: Some(format!("add sequence B failed: {err}")),
        };
    }
    if scheduler.num_free_blocks() != 1 {
        return InternalCheck {
            name,
            passed: false,
            error: Some("unexpected free block count after add_sequence".to_string()),
        };
    }

    match scheduler.allocate_next_token(2) {
        Ok(None) => {}
        Ok(Some(_)) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some("sequence B should not allocate on first growth".to_string()),
            };
        }
        Err(err) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some(format!("first growth failed: {err}")),
            };
        }
    }
    match scheduler.allocate_next_token(2) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some("sequence B should allocate a new block".to_string()),
            };
        }
        Err(err) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some(format!("second growth failed: {err}")),
            };
        }
    }

    let victims = scheduler.select_victims(1);
    if victims.is_empty() {
        return InternalCheck {
            name,
            passed: false,
            error: Some("victim selection returned empty".to_string()),
        };
    }
    let victim_ids: Vec<_> = victims.iter().map(|(id, _)| *id).collect();
    if let Err(err) = scheduler.free_victims(&victim_ids) {
        return InternalCheck {
            name,
            passed: false,
            error: Some(format!("free_victims failed: {err}")),
        };
    }
    if scheduler.num_free_blocks() == 0 {
        return InternalCheck {
            name,
            passed: false,
            error: Some("free_victims did not release blocks".to_string()),
        };
    }

    InternalCheck {
        name,
        passed: true,
        error: None,
    }
}

fn kv_cache_check() -> InternalCheck {
    let name = "KV cache double buffer";
    let backend = CpuBackend::<f32>::new();
    let config = KvCacheConfig {
        num_layers: 1,
        num_heads: 1,
        head_dim: 1,
        max_seq_len: 4,
        dtype_size: std::mem::size_of::<f32>(),
        page_size: 0,
        swap_config: None,
    };
    let handle_front = match backend.alloc_kv_cache(&config) {
        Ok(handle) => handle,
        Err(err) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some(format!("alloc kv cache failed: {err}")),
            };
        }
    };
    let handle_back = match backend.alloc_kv_cache(&config) {
        Ok(handle) => handle,
        Err(err) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some(format!("alloc kv cache failed: {err}")),
            };
        }
    };
    let mut front = KvCacheState::new(handle_front, config.clone());
    let mut back = KvCacheState::new(handle_back, config);
    if front.advance(2).is_err() || back.advance(1).is_err() {
        return InternalCheck {
            name,
            passed: false,
            error: Some("failed to advance kv cache".to_string()),
        };
    }

    let mut buffer = KvCacheDoubleBuffer::new(front, back);
    if buffer.front().used() != 2 || buffer.back().used() != 1 {
        return InternalCheck {
            name,
            passed: false,
            error: Some("unexpected kv cache usage".to_string()),
        };
    }
    buffer.swap();
    if buffer.front().used() != 1 || buffer.back().used() != 2 {
        return InternalCheck {
            name,
            passed: false,
            error: Some("kv cache swap failed".to_string()),
        };
    }
    buffer.reset_all();
    if buffer.front().used() != 0 || buffer.back().used() != 0 {
        return InternalCheck {
            name,
            passed: false,
            error: Some("kv cache reset failed".to_string()),
        };
    }

    InternalCheck {
        name,
        passed: true,
        error: None,
    }
}

#[cfg(feature = "candle")]
fn pytorch_loader_check() -> InternalCheck {
    let name = "PyTorch bin loader";
    let cache_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("regression-cache");
    if let Err(err) = std::fs::create_dir_all(&cache_dir) {
        return InternalCheck {
            name,
            passed: false,
            error: Some(format!("create cache dir failed: {err}")),
        };
    }
    let api = match ApiBuilder::new().with_cache_dir(cache_dir).build() {
        Ok(api) => api,
        Err(err) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some(format!("hf api init failed: {err}")),
            };
        }
    };
    let model = api.model("hf-internal-testing/tiny-random-bert".to_string());
    let bin_path = match model.get("pytorch_model.bin") {
        Ok(path) => path,
        Err(err) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some(format!("download bin failed: {err}")),
            };
        }
    };
    let config = PytorchConversionConfig::default();
    let output = match convert_bins_to_safetensors(&[bin_path], None, &config) {
        Ok(output) => output,
        Err(err) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some(format!("convert bins failed: {err}")),
            };
        }
    };
    let safe_path = match output.safetensors.first() {
        Some(path) => path,
        None => {
            return InternalCheck {
                name,
                passed: false,
                error: Some("missing safetensors output".to_string()),
            };
        }
    };
    let bytes = match std::fs::read(safe_path) {
        Ok(bytes) => bytes,
        Err(err) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some(format!("read safetensors failed: {err}")),
            };
        }
    };
    let tensors = match SafeTensors::deserialize(&bytes) {
        Ok(tensors) => tensors,
        Err(err) => {
            return InternalCheck {
                name,
                passed: false,
                error: Some(format!("safetensors decode failed: {err}")),
            };
        }
    };
    if tensors.names().is_empty() {
        return InternalCheck {
            name,
            passed: false,
            error: Some("no tensors found".to_string()),
        };
    }

    InternalCheck {
        name,
        passed: true,
        error: None,
    }
}

#[cfg(not(feature = "candle"))]
fn pytorch_loader_check() -> InternalCheck {
    InternalCheck {
        name: "PyTorch bin loader",
        passed: false,
        error: Some("candle feature disabled; rebuild with --features candle".to_string()),
    }
}

fn run_internal_checks(run_loader: bool) -> Vec<InternalCheck> {
    let mut checks = vec![scheduler_check(), kv_cache_check()];
    if run_loader {
        checks.push(pytorch_loader_check());
    }
    checks
}

fn print_internal_summary(checks: &[InternalCheck]) -> bool {
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  INTERNAL CHECKS");
    println!("════════════════════════════════════════════════════════════════");
    for check in checks {
        println!("  {}", check);
    }
    let failed = checks.iter().filter(|c| !c.passed).count();
    println!("  Failed: {}", failed);
    failed == 0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut use_cuda = false;
    let mut filter = None;
    let mut run_loader_check = false;

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "--cuda" => use_cuda = true,
            "--cpu" => use_cuda = false,
            "--loader" => run_loader_check = true,
            f if f.starts_with("--filter=") => {
                filter = Some(f.trim_start_matches("--filter="));
            }
            f if f.starts_with("-f=") => {
                filter = Some(f.trim_start_matches("-f="));
            }
            _ => {
                // Treat as filter
                filter = Some(arg.as_str());
            }
        }
    }

    // Set CUDA backend if requested
    if use_cuda {
        std::env::set_var("GLLM_BACKEND", "cuda");
    }

    let internal_checks = run_internal_checks(run_loader_check);
    if !print_internal_summary(&internal_checks) {
        std::process::exit(1);
    }

    let results = run_regression_test(use_cuda, filter);
    print_summary(&results);
}
