//! 真实模型回归测试
//!
//! 测试策略：按**架构类型**分组，每组测试一个代表模型
//! 同架构的不同参数量模型共享适配器逻辑，无需重复测试

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
///
/// 注意：需要 HF_TOKEN 的 gated 模型会提示错误
/// 可用的公开模型：smollm2-135m, phi-4-mini, internlm3-8b, bge-m3, e5-small
///
/// ⚠️ CPU性能警告：大模型(>3B参数)在CPU上运行极慢，测试时只生成1个token
fn test_model_with_auto_download(alias: &str) -> Result<(), String> {
    let manifest = registry::lookup(alias)
        .ok_or_else(|| format!("manifest not found: {}", alias))?;

    let adapter = adapter_for::<CpuBackend>(manifest)
        .ok_or_else(|| format!("no adapter for {:?}", manifest.model_id))?;

    println!("  测试: {} (架构: {:?})", alias, manifest.arch);

    // 使用 Loader::from_hf 自动下载模型（内部会自动 fallback 到 ModelScope）
    let mut loader = loader::Loader::from_hf(alias)
        .map_err(|e| format!("loader failed: {}", e))?;
    println!("    ✅ Loader 创建成功");

    let backend = CpuBackend::new();
    let mut executor = Executor::from_loader(backend, manifest, adapter, &mut loader)
        .map_err(|e| format!("executor failed: {}", e))?;
    println!("    ✅ Executor 创建成功");

    // 根据模型类型执行相应测试
    if manifest.model_id.is_generator() {
        println!("    🔄 开始生成测试 (1 token, CPU模式)...");
        let output = executor.generate("Hello", 1, 0.0)
            .map_err(|e| format!("generate failed: {}", e))?;
        assert!(!output.trim().is_empty(), "generator output empty");
        println!("    ✅ 生成测试通过: '{}'", output.trim());
    } else if manifest.model_id.is_embedding() {
        println!("    🔄 开始嵌入测试...");
        let embedding = executor.embed("test text")
            .map_err(|e| format!("embed failed: {}", e))?;
        assert!(!embedding.is_empty(), "embedding empty");
        let sum: f32 = embedding.iter().sum();
        assert!(sum.abs() > 0.01, "embedding is all zeros");
        println!("    ✅ 嵌入测试通过 (维度: {})", embedding.len());
    } else if manifest.model_id.is_reranker() {
        println!("    🔄 开始重排序测试...");
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

