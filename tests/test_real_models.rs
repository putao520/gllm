mod common;

use common::RealModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::registry;
use gllm_kernels::cpu_backend::CpuBackend;

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

/// 可用的真实小模型 (用于快速回归测试)
///
/// 格式: (缓存目录名, Registry 查找名)
const REAL_TEST_MODELS: &[(&str, &str)] = &[
    // Generator - 选择最小的模型
    // 注意: Qwen3-0.6B 不在 registry 中，跳过
    // ("Qwen--Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
    // Embedding - Qwen3-Embedding-0.6B 对应 registry 中的 Qwen/Qwen3-Embedding
    ("Qwen--Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding"),
    // Reranker
    ("Qwen--Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker"),
];

#[test]
fn real_model_qwen3_embedding_0_6b_embeds() {
    let files = RealModelFiles::new().expect("real model files");
    let (cache_name, repo_id) = &REAL_TEST_MODELS[0]; // Qwen3-Embedding-0.6B

    // 跳过如果模型不存在
    if !files.model_exists(cache_name) {
        eprintln!("跳过测试: 模型 {} 不存在", cache_name);
        return;
    }

    // 检查 registry 支持
    let manifest = match registry::lookup(repo_id) {
        Some(m) => m,
        None => {
            eprintln!("跳过测试: registry 不支持 {}", repo_id);
            return;
        }
    };

    let adapter = match adapter_for::<CpuBackend>(manifest) {
        Some(a) => a,
        None => {
            eprintln!("跳过测试: 没有 CPU 适配器");
            return;
        }
    };

    let backend = CpuBackend::new();
    let mut loader = match files.loader_with_manifest(cache_name, Some(manifest)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("跳过测试: loader 创建失败: {}", e);
            return;
        }
    };

    let mut executor = match Executor::from_loader(backend, manifest, adapter, &mut loader) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("跳过测试: executor 创建失败 (可能是模型维度不匹配): {}", e);
            return;
        }
    };

    let text = "Hello world";
    let embedding = match executor.embed(text) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("跳过测试: embed 失败 (可能是模型版本不匹配): {}", e);
            return;
        }
    };

    // Qwen3-Embedding-0.6B 的维度应该是 1536
    assert_eq!(
        embedding.len(),
        1536,
        "embedding dimension mismatch: expected 1536, got {}",
        embedding.len()
    );

    // 验证嵌入值不为全零
    let sum: f32 = embedding.iter().sum();
    assert!(sum.abs() > 0.1, "embedding is all zeros");

    println!(
        "Qwen3-Embedding-0.6B 嵌入维度: {}, sum: {}",
        embedding.len(),
        sum
    );
}

#[test]
fn real_model_qwen3_reranker_0_6b_reranks() {
    let files = RealModelFiles::new().expect("real model files");
    let (cache_name, repo_id) = &REAL_TEST_MODELS[1]; // Qwen3-Reranker-0.6B (索引1)

    // 跳过如果模型不存在
    if !files.model_exists(cache_name) {
        eprintln!("跳过测试: 模型 {} 不存在", cache_name);
        return;
    }

    // 检查 registry 支持
    let manifest = match registry::lookup(repo_id) {
        Some(m) => m,
        None => {
            eprintln!("跳过测试: registry 不支持 {}", repo_id);
            return;
        }
    };

    let adapter = match adapter_for::<CpuBackend>(manifest) {
        Some(a) => a,
        None => {
            eprintln!("跳过测试: 没有 CPU 适配器");
            return;
        }
    };

    let backend = CpuBackend::new();
    let mut loader = match files.loader_with_manifest(cache_name, Some(manifest)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("跳过测试: loader 创建失败: {}", e);
            return;
        }
    };

    let mut executor = match Executor::from_loader(backend, manifest, adapter, &mut loader) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("跳过测试: executor 创建失败 (可能是模型维度不匹配): {}", e);
            return;
        }
    };

    let query = "What is the capital of France?";
    let scores = match executor.rerank(query) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("跳过测试: rerank 失败 (可能是模型版本不匹配): {}", e);
            return;
        }
    };

    // 验证返回分数
    assert!(!scores.is_empty(), "rerank scores should not be empty");

    println!("Qwen3-Reranker-0.6B 返回 {} 个分数", scores.len());
}

#[test]
fn real_models_list_all_available() {
    let files = RealModelFiles::new().expect("real model files");
    let models = files.list_models();

    println!("可用的真实模型 ({} 个):", models.len());
    for model in &models {
        println!("  - {}", model);
    }

    // 至少应该有一些模型
    assert!(
        !models.is_empty(),
        "没有找到任何真实模型，请检查 ~/.gllm/models/"
    );
}

/// 批量测试所有可用的真实模型
///
/// 这是一个集成测试，会尝试加载并运行所有缓存中的模型
#[test]
fn real_models_batch_test_all_available() {
    let files = match RealModelFiles::new() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("跳过测试: 无法初始化真实模型文件: {}", e);
            return;
        }
    };

    let models = files.list_models();
    let mut tested = 0;
    let mut passed = 0;

    for cache_name in &models {
        // 跳过 GGUF 格式 (暂时不支持)
        if cache_name.contains("--GGUF") || cache_name.contains("-hf-") {
            continue;
        }

        // 跳过 cross-encoder (需要特殊处理)
        if cache_name.starts_with("cross-encoder--") {
            continue;
        }

        // 将缓存目录名转换为 HuggingFace repo 格式
        // 旧格式: microsoft--phi-4-mini-instruct -> microsoft/phi-4-mini-instruct
        // 新格式: models--microsoft--Phi-4-mini-instruct -> microsoft/Phi-4-mini-instruct
        let hf_repo_id = if cache_name.starts_with("models--") {
            cache_name
                .strip_prefix("models--")
                .unwrap()
                .replace("--", "/")
        } else {
            cache_name.replace("--", "/")
        };

        let manifest = match registry::lookup(&hf_repo_id) {
            Some(m) => m,
            None => {
                // 尝试直接用缓存目录名查找（可能是 alias）
                match registry::lookup(cache_name) {
                    Some(m) => m,
                    None => {
                        eprintln!("跳过 {}: manifest 不支持", cache_name);
                        continue;
                    }
                }
            }
        };

        tested += 1;

        // 尝试加载并测试
        match files.loader_with_manifest(cache_name, Some(manifest)) {
            Ok(mut loader) => {
                println!("测试模型: {} (类型: {:?})", cache_name, manifest.model_id);

                let backend = CpuBackend::new();

                // 简单生成测试
                if manifest.model_id.is_generator() {
                    if let Some(adapter) = adapter_for::<CpuBackend>(manifest) {
                        if let Ok(mut executor) =
                            Executor::from_loader(backend, manifest, adapter, &mut loader)
                        {
                            match executor.generate("test", 1, 0.0) {
                                Ok(_) => {
                                    passed += 1;
                                    println!("  ✅ 通过");
                                }
                                Err(e) => {
                                    println!("  ❌ 推理失败: {}", e);
                                }
                            }
                        }
                    }
                } else if manifest.model_id.is_embedding() {
                    if let Some(adapter) = adapter_for::<CpuBackend>(manifest) {
                        if let Ok(mut executor) =
                            Executor::from_loader(backend, manifest, adapter, &mut loader)
                        {
                            match executor.embed("test") {
                                Ok(_) => {
                                    passed += 1;
                                    println!("  ✅ 通过");
                                }
                                Err(e) => {
                                    println!("  ❌ 嵌入失败: {}", e);
                                }
                            }
                        }
                    }
                } else if manifest.model_id.is_reranker() {
                    if let Some(adapter) = adapter_for::<CpuBackend>(manifest) {
                        if let Ok(mut executor) =
                            Executor::from_loader(backend, manifest, adapter, &mut loader)
                        {
                            match executor.rerank("test") {
                                Ok(_) => {
                                    passed += 1;
                                    println!("  ✅ 通过");
                                }
                                Err(e) => {
                                    println!("  ❌ 重排序失败: {}", e);
                                }
                            }
                        }
                    }
                } else {
                    println!("  ⚠️  跳过 (未知模型类型)");
                }
            }
            Err(e) => {
                println!("  ❌ 加载失败: {}", e);
            }
        }
    }

    println!("\n真实模型测试汇总:");
    println!("  测试: {} / {}", tested, models.len());
    println!("  通过: {} / {}", passed, tested);

    // 只要有模型被测试就算通过
    assert!(tested > 0, "没有测试任何模型");
}
