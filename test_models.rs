use gllm::{ModelRegistry, ModelType};

fn main() {
    let registry = ModelRegistry::new();

    println!("🔍 检查 gllm 模型支持...\n");

    // 测试所有模型别名（26个模型）
    let model_aliases = vec![
        "bge-small-zh",
        "bge-small-en",
        "bge-base-en",
        "bge-large-en",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "multi-qa-mpnet-base-dot-v1",
        "e5-large",
        "e5-base",
        "e5-small",
        "jina-embeddings-v2-base-en",
        "jina-embeddings-v2-small-en",
        "m3e-base",
        "multilingual-MiniLM-L12-v2",
        "distiluse-base-multilingual-cased-v1",
        "all-MiniLM-L12-v2",
        "all-distilroberta-v1",
        "bge-reranker-v2",
        "bge-reranker-large",
        "bge-reranker-base",
        "ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2",
        "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-electra-base",
        "quora-distilroberta-base",
    ];

    let mut success_count = 0;
    let total_count = model_aliases.len();

    println!("📋 测试 {} 个内置模型别名:", total_count);
    println!();

    for alias in &model_aliases {
        match registry.resolve(alias) {
            Ok(info) => {
                println!(
                    "✅ {} -> {} ({})",
                    alias,
                    info.repo_id,
                    if info.model_type == ModelType::Embedding {
                        "Embedding"
                    } else {
                        "Rerank"
                    }
                );
                success_count += 1;
            }
            Err(e) => {
                println!("❌ {} -> 错误: {}", alias, e);
            }
        }
    }

    println!();
    println!("📊 结果统计:");
    println!(
        "✅ 成功解析: {}/{} ({:.1}%)",
        success_count,
        total_count,
        (success_count as f64 / total_count as f64) * 100.0
    );

    // 测试推理引擎限制
    println!();
    println!("⚠️  重要说明:");
    println!("虽然我们能解析这些模型别名，但实际推理能力受限于:");
    println!("1. 推理引擎架构 - 目前使用简化的 BERT 架构");
    println!("2. 模型兼容性 - 只有 SafeTensors 格式的模型能正常工作");
    println!("3. 内部实现 - embedding 输出固定为 {} 维", 128); // EMBEDDING_OUTPUT
    println!("4. 测试模式 - 使用假权重进行验证");

    println!();
    println!("🎯 总结:");
    if success_count == total_count {
        println!("✅ 所有模型别名都能正常解析！这是模型注册表的第一步。");
        println!("🔧 实际推理需要模型的 SafeTensors 文件和正确的架构适配。");
    } else {
        println!(
            "❌ 有 {} 个模型解析失败，需要检查注册表实现。",
            total_count - success_count
        );
    }
}
