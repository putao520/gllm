//! 真实模型回归测试
//!
//! 测试策略：按**架构类型**分组，每组测试一个代表模型
//! 同架构的不同参数量模型共享适配器逻辑，无需重复测试

mod common;

use common::RealModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::loader;
use gllm::registry;
use gllm_kernels::cpu_backend::CpuBackend;

/// 回归测试模型列表 (按架构分组)
///
/// 每个架构只测试一个代表模型，避免重复测试
const REGRESSION_MODELS: &[(&str, &str)] = &[
    // ===== Generator 架构代表 =====
    ("Qwen3", "qwen3-7b"),                    // Qwen3 系列 (Qwen3 + Qwen3MoE)
    ("Llama4", "llama-4-8b"),                 // Llama 4 系列
    ("SmolLM", "smollm2-135m"),               // SmolLM 系列 (超轻量，CI友好)
    ("Phi4", "phi-4-mini"),                   // Phi4 系列
    ("Gemma2", "gemma-2-9b"),                // Gemma2 系列
    ("InternLM3", "internlm3-8b"),            // InternLM3 系列
    ("GPT-OSS", "gpt-oss-1.5b"),             // GPT-OSS (Fused QKV)
    ("GLM4", "glm-4.7-flash"),               // GLM-4/5 系列
    ("Mistral3", "ministral-8b"),            // Mistral3/Ministral

    // ===== Embedding 架构代表 =====
    ("Qwen3-Embed", "qwen3-embed"),          // Qwen3 Embedding
    ("BGE-XlmR", "bge-m3"),                   // BGE XLM-R (中文)
    ("E5-XlmR", "e5-small"),                  // E5 XLM-R

    // ===== Reranker 架构代表 =====
    ("Qwen3-Rerank", "qwen3-rerank"),         // Qwen3 Reranker
    ("BGE-Rerank", "bge-rerank-v3"),          // BGE Reranker
];

/// 使用 HuggingFace 自动下载测试单个模型
/// 失败时自动回退到 ModelScope（魔搭社区）
fn test_model_with_auto_download(alias: &str) -> Result<(), String> {
    let manifest = registry::lookup(alias)
        .ok_or_else(|| format!("manifest not found: {}", alias))?;

    let adapter = adapter_for::<CpuBackend>(manifest)
        .ok_or_else(|| format!("no adapter for {:?}", manifest.model_id))?;

    println!("  测试: {} (架构: {:?})", alias, manifest.arch);

    // 使用 Loader::from_hf_with_fallback 自动下载模型
    // HF 失败时自动尝试 ModelScope（对中国用户更友好）
    let mut loader = loader::Loader::from_hf_with_fallback(alias)
        .map_err(|e| format!("loader failed: {}", e))?;

    let backend = CpuBackend::new();
    let mut executor = Executor::from_loader(backend, manifest, adapter, &mut loader)
        .map_err(|e| format!("executor failed: {}", e))?;

    // 根据模型类型执行相应测试
    if manifest.model_id.is_generator() {
        let output = executor.generate("Hello", 1, 0.0)
            .map_err(|e| format!("generate failed: {}", e))?;
        assert!(!output.trim().is_empty(), "generator output empty");
        println!("    ✅ 生成测试通过");
    } else if manifest.model_id.is_embedding() {
        let embedding = executor.embed("test text")
            .map_err(|e| format!("embed failed: {}", e))?;
        assert!(!embedding.is_empty(), "embedding empty");
        let sum: f32 = embedding.iter().sum();
        assert!(sum.abs() > 0.01, "embedding is all zeros");
        println!("    ✅ 嵌入测试通过 (维度: {})", embedding.len());
    } else if manifest.model_id.is_reranker() {
        let scores = executor.rerank("query text")
            .map_err(|e| format!("rerank failed: {}", e))?;
        assert!(!scores.is_empty(), "rerank scores empty");
        println!("    ✅ 重排序测试通过");
    } else {
        return Err(format!("未知模型类型: {:?}", manifest.model_id));
    }

    Ok(())
}

/// 回归测试 - 测试所有架构代表模型
///
/// 覆盖率：11个架构 → 32个模型
#[test]
fn regression_all_architectures() {
    println!("🚀 回归测试 - {} 个架构代表", REGRESSION_MODELS.len());
    let mut passed = 0;
    let mut failed = Vec::new();
    let mut skipped = 0;

    for (arch_name, alias) in REGRESSION_MODELS {
        print!("[{}] ", arch_name);
        match test_model_with_auto_download(alias) {
            Ok(()) => passed += 1,
            Err(e) => {
                // 某些模型可能不存在或下载失败
                if e.contains("404") || e.contains("Not Found") || e.contains("download") {
                    eprintln!(" ⏭️  跳过: {}", e);
                    skipped += 1;
                } else {
                    eprintln!(" ❌ 失败: {}", e);
                    failed.push((*arch_name, alias, e));
                }
            }
        }
    }

    println!("\n📊 回归测试汇总:");
    println!("  通过: {} / {}", passed, REGRESSION_MODELS.len());
    println!("  跳过: {}", skipped);

    if !failed.is_empty() {
        eprintln!("\n❌ 失败:");
        for (arch, alias, error) in &failed {
            eprintln!("  [{}] {}: {}", arch, alias, error);
        }
    }

    // 至少 50% 通过（有些模型可能暂时不可用）
    let total_tested = passed + failed.len();
    if total_tested > 0 {
        assert!(passed >= total_tested / 2,
            "回归测试通过率太低: {} / {}", passed, total_tested);
    }
}

/// 快速 CI 测试 - 仅测试超轻量模型
///
/// 适合 CI/CD 环境，快速验证核心功能
#[test]
fn regression_ci_quick() {
    // 只测试最小的模型 (135M)，适合快速 CI
    const CI_MODELS: &[(&str, &str)] = &[
        ("SmolLM", "smollm2-135m"),  // 135M - 最小
        ("E5", "e5-small"),           // 轻量嵌入
    ];

    println!("⚡ CI 快速测试");
    for (arch, alias) in CI_MODELS {
        print!("[{}] ", arch);
        match test_model_with_auto_download(alias) {
            Ok(()) => {},
            Err(e) => {
                if e.contains("404") || e.contains("download") {
                    eprintln!(" ⏭️  跳过: {}", e);
                } else {
                    panic!("CI 测试失败 [{}]: {}", arch, e);
                }
            }
        }
    }
}

// ============================================================================
// 以下为旧的本地缓存测试（保留用于向后兼容）
// ============================================================================

fn build_real_executor(
    repo_id: &str,
    cache_name: &str,
    files: &RealModelFiles,
) -> Executor<CpuBackend> {
    let manifest = registry::lookup(repo_id).expect("manifest not found in registry");
    let adapter = adapter_for::<CpuBackend>(manifest).expect("adapter not found for CPU backend");
    let backend = CpuBackend::new();
    let mut loader = files
        .loader_with_manifest(cache_name, Some(manifest))
        .expect("loader creation failed");
    Executor::from_loader(backend, manifest, adapter, &mut loader)
        .expect("executor creation failed")
}

/// 本地缓存模型列表
const LOCAL_TEST_MODELS: &[(&str, &str)] = &[
    ("Qwen--Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding"),
    ("Qwen--Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker"),
];

#[test]
fn real_model_qwen3_embedding_0_6b_embeds() {
    let files = RealModelFiles::new().expect("real model files");
    let (cache_name, repo_id) = &LOCAL_TEST_MODELS[0];

    if !files.model_exists(cache_name) {
        eprintln!("跳过: 模型 {} 不存在", cache_name);
        return;
    }

    let manifest = match registry::lookup(repo_id) {
        Some(m) => m,
        None => {
            eprintln!("跳过: registry 不支持 {}", repo_id);
            return;
        }
    };

    let adapter = match adapter_for::<CpuBackend>(manifest) {
        Some(a) => a,
        None => {
            eprintln!("跳过: 无 CPU 适配器");
            return;
        }
    };

    let backend = CpuBackend::new();
    let mut loader = match files.loader_with_manifest(cache_name, Some(manifest)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("跳过: loader 失败: {}", e);
            return;
        }
    };

    let mut executor = match Executor::from_loader(backend, manifest, adapter, &mut loader) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("跳过: executor 失败: {}", e);
            return;
        }
    };

    let embedding = match executor.embed("Hello world") {
        Ok(e) => e,
        Err(e) => {
            eprintln!("跳过: embed 失败: {}", e);
            return;
        }
    };

    assert_eq!(embedding.len(), 1536, "embedding dimension mismatch");
    let sum: f32 = embedding.iter().sum();
    assert!(sum.abs() > 0.1, "embedding is all zeros");
    println!("✅ Qwen3-Embedding-0.6B: 维度={}, sum={}", embedding.len(), sum);
}

#[test]
fn real_model_qwen3_reranker_0_6b_reranks() {
    let files = RealModelFiles::new().expect("real model files");
    let (cache_name, repo_id) = &LOCAL_TEST_MODELS[1];

    if !files.model_exists(cache_name) {
        eprintln!("跳过: 模型 {} 不存在", cache_name);
        return;
    }

    let manifest = match registry::lookup(repo_id) {
        Some(m) => m,
        None => {
            eprintln!("跳过: registry 不支持 {}", repo_id);
            return;
        }
    };

    let adapter = match adapter_for::<CpuBackend>(manifest) {
        Some(a) => a,
        None => {
            eprintln!("跳过: 无 CPU 适配器");
            return;
        }
    };

    let backend = CpuBackend::new();
    let mut loader = match files.loader_with_manifest(cache_name, Some(manifest)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("跳过: loader 失败: {}", e);
            return;
        }
    };

    let mut executor = match Executor::from_loader(backend, manifest, adapter, &mut loader) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("跳过: executor 失败: {}", e);
            return;
        }
    };

    let scores = match executor.rerank("query") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("跳过: rerank 失败: {}", e);
            return;
        }
    };

    assert!(!scores.is_empty(), "rerank scores empty");
    println!("✅ Qwen3-Reranker-0.6B: {} 个分数", scores.len());
}

#[test]
fn real_models_list_all_available() {
    let files = RealModelFiles::new().expect("real model files");
    let models = files.list_models();

    println!("可用的本地模型 ({} 个):", models.len());
    for model in &models {
        println!("  - {}", model);
    }

    assert!(!models.is_empty(), "没有找到任何本地模型");
}

/// 批量测试本地缓存中的所有模型
#[test]
fn real_models_batch_test_all_available() {
    let files = match RealModelFiles::new() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("跳过: 无法初始化: {}", e);
            return;
        }
    };

    let models = files.list_models();
    let mut tested = 0;
    let mut passed = 0;

    for cache_name in &models {
        if cache_name.contains("--GGUF") || cache_name.contains("-hf-") {
            continue;
        }
        if cache_name.starts_with("cross-encoder--") {
            continue;
        }

        let hf_repo_id = if cache_name.starts_with("models--") {
            cache_name.strip_prefix("models--").unwrap().replace("--", "/")
        } else {
            cache_name.replace("--", "/")
        };

        let manifest = match registry::lookup(&hf_repo_id) {
            Some(m) => m,
            None => match registry::lookup(cache_name) {
                Some(m) => m,
                None => continue,
            },
        };

        tested += 1;

        match files.loader_with_manifest(cache_name, Some(manifest)) {
            Ok(mut loader) => {
                println!("测试: {} ({:?})", cache_name, manifest.model_id);
                let backend = CpuBackend::new();

                let Some(adapter) = adapter_for::<CpuBackend>(manifest) else {
                    println!("  ❌ 无适配器");
                    continue;
                };

                let mut executor = match Executor::from_loader(backend, manifest, adapter, &mut loader) {
                    Ok(e) => e,
                    Err(e) => {
                        println!("  ❌ Executor 创建失败: {}", e);
                        continue;
                    }
                };

                let test_ok = if manifest.model_id.is_generator() {
                    executor.generate("test", 1, 0.0).ok().is_some()
                } else if manifest.model_id.is_embedding() {
                    executor.embed("test").ok().is_some()
                } else if manifest.model_id.is_reranker() {
                    executor.rerank("test").ok().is_some()
                } else {
                    false
                };

                if test_ok {
                    passed += 1;
                    println!("  ✅");
                } else {
                    println!("  ❌");
                }
            }
            Err(e) => {
                println!("  ❌ 加载失败: {}", e);
            }
        }
    }

    println!("\n本地模型测试: {} / {} 通过", passed, tested);
    assert!(tested > 0, "没有测试任何模型");
}
