//! E2E 测试 - 最小测试矩阵
//!
//! 测试策略：
//! - 3 大类功能 × 1 个最小模型 (ModelScope 可用)
//! - Reranker: 需要 HF_TOKEN (HuggingFace)
//!
//! **E2E 测试原则**：像真实用户一样使用公开 Client API

use gllm::{Client, ModelKind};

/// E2E 测试矩阵：功能 × 最小模型
const E2E_MATRIX: &[(&str, &str)] = &[
    // Generator - 最小模型
    ("Generator", "Qwen/Qwen2.5-0.5B-Instruct"),
    // Embedding - 最小模型
    ("Embedding", "intfloat/e5-small"),
];

/// Reranker 测试 (HuggingFace 模型)
const RERANKER_MODEL: (&str, &str) = ("Reranker", "BAAI/bge-reranker-v2-m3");

/// E2E 测试 - 验证单个功能端到端流程
fn test_e2e_feature(feature: &str, alias: &str) -> Result<(), String> {
    // E2E 入口：显式指定 ModelKind
    let client = match feature {
        "Generator" => Client::new_chat(alias),
        "Embedding" => Client::new_embedding(alias),
        "Reranker" => Client::new(alias, ModelKind::Reranker),
        _ => return Err(format!("未知功能类型: {}", feature)),
    }
    .map_err(|e| format!("Client init failed: {}", e))?;
    println!(
        "  [{}] 测试: {} (架构: {:?})",
        feature,
        alias,
        client.manifest().arch
    );

    match feature {
        "Generator" => {
            let response = client
                .generate("The capital of")
                .max_tokens(10)
                .generate()
                .map_err(|e| format!("generate failed: {}", e))?;
            assert!(!response.text.trim().is_empty(), "generator output empty");
            assert!(
                response.text.len() >= 5,
                "generator output too short (< 5 chars)"
            );
            println!("    ✅ 生成: '{}'", response.text.trim());
        }
        "Embedding" => {
            let response = client
                .embeddings(["test text"])
                .generate()
                .map_err(|e| format!("embed failed: {}", e))?;
            assert!(!response.embeddings.is_empty(), "embeddings empty");
            let embedding = &response.embeddings[0].embedding;
            assert!(!embedding.is_empty(), "embedding empty");
            let sum: f32 = embedding.iter().sum();
            assert!(sum.abs() > 0.01, "embedding is all zeros");
            println!("    ✅ 嵌入维度: {}", embedding.len());
        }
        "Reranker" => {
            let response = client
                .rerank("query", ["doc1", "doc2"])
                .generate()
                .map_err(|e| format!("rerank failed: {}", e))?;
            assert!(!response.results.is_empty(), "rerank results empty");
            println!("    ✅ 重排序结果数: {}", response.results.len());
        }
        _ => return Err(format!("未知功能类型: {}", feature)),
    }

    Ok(())
}

/// TEST-E2E-001: E2E 测试 - 验证功能端到端流程
///
/// **关联需求**: REQ-TEST-001
/// **测试类型**: 正向测试
/// **E2E测试粒度**: 业务流程
///
/// **前置条件**: ModelScope 可用，模型已缓存
///
/// **测试步骤**:
/// 1. 创建 Client (显式指定 ModelKind)
/// 2. 执行 generate/embeddings/rerank
/// 3. 验证输出结果
///
/// **期望结果**: ModelScope 模型 100% 通过
#[test]
fn e2e_features() {
    println!("🚀 E2E 测试");
    let mut passed = 0;
    let mut failed = Vec::new();

    // 测试 ModelScope 可用模型
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

    // 测试 Reranker (HuggingFace，自动读取 ~/.huggingface/token)
    print!("[Reranker] ");
    match test_e2e_feature(RERANKER_MODEL.0, RERANKER_MODEL.1) {
        Ok(()) => {
            passed += 1;
            println!();
        }
        Err(e) => {
            eprintln!(" ❌ 失败: {}", e);
            failed.push((RERANKER_MODEL.0, RERANKER_MODEL.1, e));
        }
    }

    println!("\n📊 E2E 测试汇总:");
    println!("  通过: {} / 3", passed);

    if !failed.is_empty() {
        eprintln!("\n❌ 失败:");
        for (feature, alias, error) in &failed {
            eprintln!("  [{}] {}: {}", feature, alias, error);
        }
    }

    // E2E 测试要求：ModelScope 可用模型必须 100% 通过
    // (Reranker 需要 HuggingFace token + gated 权限，失败不影响基本 E2E 功能)
    let modelscope_failed: Vec<_> = failed
        .iter()
        .filter(|(f, _, _)| *f != "Reranker")
        .cloned()
        .collect();
    assert_eq!(modelscope_failed.len(), 0, "E2E 测试失败 (ModelScope 模型)");
}
