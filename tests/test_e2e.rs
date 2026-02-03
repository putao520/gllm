//! E2E 测试 - 最小测试矩阵
//!
//! 测试策略：
//! - 2 大类功能 × 1 个最小模型
//! - 验证端到端流程可用
//!
//! **E2E 测试原则**：像真实用户一样使用公开 Client API

use gllm::Client;

/// E2E 测试矩阵：功能 × 最小模型
const E2E_MATRIX: &[(&str, &str)] = &[
    // Generator - 最小模型
    ("Generator", "smollm2-135m"),
    // Embedding - 最小模型
    ("Embedding", "e5-small"),
];

/// E2E 测试 - 验证单个功能端到端流程
fn test_e2e_feature(feature: &str, alias: &str) -> Result<(), String> {
    // E2E 入口：Client::new()
    let client = Client::new(alias)
        .map_err(|e| format!("Client::new failed: {}", e))?;
    println!("  [{}] 测试: {} (架构: {:?})", feature, alias, client.manifest().arch);

    match feature {
        "Generator" => {
            let response = client.generate("Hello")
                .max_tokens(1)
                .generate()
                .map_err(|e| format!("generate failed: {}", e))?;
            assert!(!response.text.trim().is_empty(), "generator output empty");
            println!("    ✅ 生成: '{}'", response.text.trim());
        }
        "Embedding" => {
            let response = client.embeddings(["test text"])
                .generate()
                .map_err(|e| format!("embed failed: {}", e))?;
            assert!(!response.embeddings.is_empty(), "embeddings empty");
            let embedding = &response.embeddings[0].embedding;
            assert!(!embedding.is_empty(), "embedding empty");
            let sum: f32 = embedding.iter().sum();
            assert!(sum.abs() > 0.01, "embedding is all zeros");
            println!("    ✅ 嵌入维度: {}", embedding.len());
        }
        _ => return Err(format!("未知功能类型: {}", feature)),
    }

    Ok(())
}

/// E2E 测试 - 验证 2 大类功能端到端流程
#[test]
fn e2e_features() {
    println!("🚀 E2E 测试");
    let mut passed = 0;
    let mut failed = Vec::new();

    for (feature, alias) in E2E_MATRIX {
        print!("[{}] ", feature);
        match test_e2e_feature(feature, alias) {
            Ok(()) => {
                passed += 1;
                println!();
            }
            Err(e) => {
                eprintln!(" ❌ 失败: {}", e);
                failed.push((*feature, *alias, e));
            }
        }
    }

    println!("\n📊 E2E 测试汇总:");
    println!("  通过: {} / {}", passed, E2E_MATRIX.len());

    if !failed.is_empty() {
        eprintln!("\n❌ 失败:");
        for (feature, alias, error) in &failed {
            eprintln!("  [{}] {}: {}", feature, alias, error);
        }
    }

    // E2E 测试要求 100% 通过
    assert_eq!(failed.len(), 0, "E2E 测试失败: {} / {}", passed, E2E_MATRIX.len());
}
