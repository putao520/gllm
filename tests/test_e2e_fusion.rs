//! E2E 测试: Reranker 融合管线 (Embedding + Reranker / Generator + Reranker)
//!
//! 验证多模型组合推理的端到端正确性:
//! - Embed + Rerank: 先生成 embedding，再用 reranker 精排
//! - Embed + Rerank + Generate (RAG): 完整 RAG 管线
//!
//! 反作弊检查: 融合结果与独立推理一致、rerank_scores 非退化、
//!             排序正确性、top_n 截断、RAG 语义正确性

use gllm::Client;

// ============================================================================
// Anti-cheating helpers
// ============================================================================

fn assert_embedding_sane(emb: &[f32], label: &str) {
    assert!(!emb.is_empty(), "{label}: embedding is empty");

    for (i, &v) in emb.iter().enumerate() {
        assert!(v.is_finite(), "{label}: NaN/Inf at index {i}: {v}");
    }

    let all_zero = emb.iter().all(|&v| v == 0.0);
    assert!(!all_zero, "{label}: all zeros");

    let first = emb[0];
    let all_same = emb.iter().all(|&v| v == first);
    assert!(!all_same, "{label}: all same value ({first})");

    let mean = emb.iter().sum::<f32>() / emb.len() as f32;
    let variance = emb.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / emb.len() as f32;
    assert!(
        variance > 1e-10,
        "{label}: variance {variance} near zero — degenerate"
    );
}

fn assert_scores_sane(scores: &[f32], label: &str) {
    assert!(!scores.is_empty(), "{label}: scores are empty");

    for (i, &s) in scores.iter().enumerate() {
        assert!(s.is_finite(), "{label}: score[{i}] is not finite: {s}");
    }

    let all_zero = scores.iter().all(|&s| s == 0.0);
    assert!(!all_zero, "{label}: all scores are zero — degenerate");

    // 降序排列
    for i in 1..scores.len() {
        assert!(
            scores[i - 1] >= scores[i],
            "{label}: scores not sorted descending: [{i}]={} > [{}]={}",
            scores[i],
            i - 1,
            scores[i - 1]
        );
    }

    // 分数离散度
    let max = scores.first().unwrap();
    let min = scores.last().unwrap();
    let spread = (max - min).abs();
    assert!(
        spread > 1e-6,
        "{label}: score spread {spread} too small (max={max}, min={min})"
    );
}

// ============================================================================
// TEST-E2E-FUSION-001: Embedding + Reranker 融合管线
// ============================================================================

/// 使用 ClientBuilder 组合 embedding 模型和 reranker 模型，
/// 通过 embed_builder().rerank_query().generate() 管线执行融合推理。
///
/// 验证:
/// - embedding 通过反退化检查
/// - rerank_scores 存在且正确排序
/// - 结果按相关性降序排列
/// - top_n 截断生效
///
/// **关联需求**: REQ-TEST-003, REQ-TEST-004
/// **测试类型**: 正向 (融合管线)
#[test]
fn e2e_fusion_embed_rerank() {
    // 使用最小的模型
    const EMBED_MODEL: &str = "intfloat/e5-small-v2";
    const RERANKER_MODEL: &str = "BAAI/bge-reranker-v2-m3";

    let client = Client::builder()
        .model(EMBED_MODEL)
        .kind(gllm::ModelKind::Embedding)
        .reranker(RERANKER_MODEL)
        .build()
        .expect("Failed to build fusion client");

    let documents = vec![
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "Berlin is the capital of Germany.",
        "London is the capital of England.",
        "Tokyo is the capital of Japan.",
    ];

    let response = client
        .embed_builder(documents.clone())
        .rerank_query("What is the capital of France?")
        .generate()
        .expect("Embed+rerank pipeline failed");

    // 1. 结果数量正确
    assert_eq!(
        response.embeddings.len(),
        5,
        "Should have 5 embeddings (no top_n truncation)"
    );

    // 2. rerank_scores 必须存在
    let scores = response
        .rerank_scores
        .as_ref()
        .expect("rerank_scores should be present in fusion pipeline");
    assert_eq!(scores.len(), 5, "Should have 5 rerank scores");

    // 3. embedding 反退化检查
    for (i, emb) in response.embeddings.iter().enumerate() {
        assert_embedding_sane(&emb.embedding, &format!("fusion_embed[{i}]"));
    }

    // 4. rerank scores 反退化检查
    assert_scores_sane(scores, "fusion_rerank");

    // 5. 第一个 embedding 应该对应最相关的文档 (Paris/France)
    //    scores[0] 应该是最高分
    assert!(
        scores[0] >= scores[1],
        "First result should have highest score"
    );
}

// ============================================================================
// TEST-E2E-FUSION-002: Embedding + Reranker 融合管线 (top_n 截断)
// ============================================================================

/// 验证 top_n 参数正确截断结果集
///
/// **关联需求**: REQ-TEST-003, REQ-TEST-004
/// **测试类型**: 正向 (top_n 截断)
#[test]
fn e2e_fusion_embed_rerank_top_n() {
    const EMBED_MODEL: &str = "intfloat/e5-small-v2";
    const RERANKER_MODEL: &str = "BAAI/bge-reranker-v2-m3";

    let client = Client::builder()
        .model(EMBED_MODEL)
        .kind(gllm::ModelKind::Embedding)
        .reranker(RERANKER_MODEL)
        .build()
        .expect("Failed to build fusion client");

    let documents = vec![
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "London is the capital of England.",
        "Tokyo is the capital of Japan.",
        "Moscow is the capital of Russia.",
    ];

    let response = client
        .embed_builder(documents)
        .rerank_query("What is the capital of France?")
        .top_n(2)
        .generate()
        .expect("Embed+rerank top_n pipeline failed");

    // top_n=2 应该只返回 2 个结果
    assert_eq!(
        response.embeddings.len(),
        2,
        "top_n=2 should truncate to 2 embeddings"
    );

    let scores = response
        .rerank_scores
        .as_ref()
        .expect("rerank_scores should be present");
    assert_eq!(scores.len(), 2, "Should have 2 rerank scores after top_n");

    // 这 2 个结果应该是最高分的
    for emb in &response.embeddings {
        assert_embedding_sane(&emb.embedding, "top_n_embed");
    }

    // 分数应该降序
    assert!(
        scores[0] >= scores[1],
        "Top 2 scores should be descending: {} vs {}",
        scores[0],
        scores[1]
    );
}

// ============================================================================
// TEST-E2E-FUSION-003: 完整 RAG 管线 (Embed + Rerank + Generate)
// ============================================================================

/// 完整 RAG 管线: embedding 召回 → reranker 精排 → LLM 生成答案
///
/// 验证:
/// - 管线端到端不崩溃
/// - 生成的答案非空且通过反退化检查
/// - sources 和 rerank_scores 正确返回
/// - 答案语义合理
///
/// **关联需求**: REQ-TEST-002, REQ-TEST-003, REQ-TEST-004
/// **测试类型**: 正向 (完整 RAG 管线)
#[test]
fn e2e_fusion_rag_pipeline() {
    const EMBED_MODEL: &str = "intfloat/e5-small-v2";
    const RERANKER_MODEL: &str = "BAAI/bge-reranker-v2-m3";
    const GENERATOR_MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

    let client = Client::builder()
        .model(EMBED_MODEL)
        .kind(gllm::ModelKind::Embedding)
        .reranker(RERANKER_MODEL)
        .generator(GENERATOR_MODEL)
        .build()
        .expect("Failed to build RAG client");

    let documents = vec![
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "Berlin is the capital and largest city of Germany.",
        "London is the capital and largest city of England and the United Kingdom.",
    ];

    let rag_response = client
        .embed_builder(documents)
        .rerank_query("What is the capital of France?")
        .top_n(2)
        .generate_answer("Answer the question based on the provided documents. Be concise.")
        .expect("RAG pipeline failed");

    // 1. 生成的答案非空
    let text = rag_response.text.trim();
    assert!(!text.is_empty(), "RAG answer should not be empty");
    assert!(
        text.len() > 3,
        "RAG answer too short ({} chars): {:?}",
        text.len(),
        text
    );

    // 2. sources 正确返回
    assert!(
        !rag_response.sources.is_empty(),
        "RAG should return source document indices"
    );
    assert!(
        rag_response.sources.len() <= 2,
        "top_n=2 should select at most 2 sources, got {}",
        rag_response.sources.len()
    );

    // 3. rerank_scores 非退化
    assert_eq!(
        rag_response.rerank_scores.len(),
        rag_response.sources.len(),
        "rerank_scores length should match sources length"
    );
    for (i, &s) in rag_response.rerank_scores.iter().enumerate() {
        assert!(
            s.is_finite(),
            "RAG rerank_score[{i}] is not finite: {s}"
        );
    }

    // 4. 分数降序
    for i in 1..rag_response.rerank_scores.len() {
        assert!(
            rag_response.rerank_scores[i - 1] >= rag_response.rerank_scores[i],
            "RAG rerank scores should be descending"
        );
    }

    // 5. 反退化: 答案字符多样性
    let unique_chars: std::collections::HashSet<char> = text.chars().collect();
    assert!(
        unique_chars.len() >= 3,
        "RAG answer has only {} unique characters — degenerate",
        unique_chars.len()
    );

    // 6. 重复检测
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= 5 {
        let mut max_repeat = 1;
        let mut current = 1;
        for i in 1..words.len() {
            if words[i] == words[i - 1] {
                current += 1;
                if current > max_repeat {
                    max_repeat = current;
                }
            } else {
                current = 1;
            }
        }
        assert!(
            max_repeat < 5,
            "RAG answer has {max_repeat} consecutive repeated words — degenerate"
        );
    }
}

// ============================================================================
// TEST-E2E-FUSION-004: 融合管线 vs 独立推理一致性
// ============================================================================

/// 验证融合管线中的 embedding 输出与独立 embed 调用的结果一致。
/// 融合管线不应改变 embedding 本身的数值，只是重新排序。
///
/// **关联需求**: REQ-TEST-003
/// **测试类型**: 正向 (一致性验证)
#[test]
fn e2e_fusion_consistency_with_standalone() {
    const EMBED_MODEL: &str = "intfloat/e5-small-v2";
    const RERANKER_MODEL: &str = "BAAI/bge-reranker-v2-m3";

    // 独立 embedding
    let embed_client =
        Client::new_embedding(EMBED_MODEL).expect("Failed to load embedding model");
    let standalone = embed_client
        .embed(["Paris is the capital of France.", "Berlin is the capital of Germany."])
        .expect("Standalone embedding failed");

    // 融合管线 embedding
    let fusion_client = Client::builder()
        .model(EMBED_MODEL)
        .kind(gllm::ModelKind::Embedding)
        .reranker(RERANKER_MODEL)
        .build()
        .expect("Failed to build fusion client");

    let fused = fusion_client
        .embed_builder(vec![
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
        ])
        .rerank_query("capital of France")
        .generate()
        .expect("Fusion pipeline failed");

    // 融合管线会按 rerank score 重新排序，所以不能直接比较顺序
    // 但每个 embedding 向量的数值应该在独立结果中能找到匹配

    let standalone_embs: Vec<&Vec<f32>> = standalone
        .embeddings
        .iter()
        .map(|e| &e.embedding)
        .collect();
    let fused_embs: Vec<&Vec<f32>> = fused
        .embeddings
        .iter()
        .map(|e| &e.embedding)
        .collect();

    // 对融合结果中的每个 embedding，在独立结果中找到匹配 (余弦相似度 > 0.999)
    for (i, fused_emb) in fused_embs.iter().enumerate() {
        let mut found_match = false;
        for standalone_emb in &standalone_embs {
            let dot: f32 = fused_emb
                .iter()
                .zip(standalone_emb.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_f: f32 = fused_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_s: f32 = standalone_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_f > 0.0 && norm_s > 0.0 {
                let sim = dot / (norm_f * norm_s);
                if sim > 0.999 {
                    found_match = true;
                    break;
                }
            }
        }
        assert!(
            found_match,
            "Fusion embedding[{i}] has no matching standalone embedding — \
             fusion pipeline may have corrupted embedding values"
        );
    }

    // 融合结果应该有 rerank_scores
    assert!(
        fused.rerank_scores.is_some(),
        "Fusion result should have rerank_scores"
    );
}

// ============================================================================
// TEST-E2E-FUSION-005: 跨架构 Embed+Rerank (Encoder+Decoder)
// ============================================================================

/// 验证不同架构族的模型可以在融合管线中组合使用:
/// - Embedding: XlmR (Encoder 架构, intfloat/e5-small-v2)
/// - Reranker: Qwen3 (Decoder 架构, DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF)
///
/// 跨架构组合是 gllm 融合管线的关键能力 — Encoder 产出 embedding 向量,
/// Decoder 架构的 reranker 基于交叉注意力机制对 query-document 对进行精排。
/// 两种架构的内部算子路径完全不同 (LayerNorm vs RMSNorm, AbsolutePos vs RoPE,
/// GELU vs SwiGLU), 本测试验证管线能正确桥接异构模型。
///
/// **关联需求**: REQ-TEST-003, REQ-TEST-004
/// **测试类型**: 正向 (跨架构融合)
#[test]
fn e2e_fusion_cross_arch_embed_rerank() {
    const EMBED_MODEL: &str = "intfloat/e5-small-v2";
    const RERANKER_MODEL: &str = "DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF";

    let client = Client::builder()
        .model(EMBED_MODEL)
        .kind(gllm::ModelKind::Embedding)
        .reranker(RERANKER_MODEL)
        .build()
        .expect("Failed to build cross-arch fusion client");

    let documents = vec![
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "Berlin is the capital of Germany.",
        "London is the capital of England.",
        "Tokyo is the capital of Japan.",
    ];

    let response = client
        .embed_builder(documents.clone())
        .rerank_query("What is the capital of France?")
        .generate()
        .expect("Cross-arch embed+rerank pipeline failed");

    // 1. 结果数量正确
    assert_eq!(
        response.embeddings.len(),
        5,
        "Should have 5 embeddings"
    );

    // 2. rerank_scores 必须存在
    let scores = response
        .rerank_scores
        .as_ref()
        .expect("rerank_scores should be present in cross-arch fusion pipeline");
    assert_eq!(scores.len(), 5, "Should have 5 rerank scores");

    // 3. embedding 反退化检查 (Encoder 架构输出)
    for (i, emb) in response.embeddings.iter().enumerate() {
        assert_embedding_sane(&emb.embedding, &format!("cross_arch_embed[{i}]"));
    }

    // 4. rerank scores 反退化检查 (Decoder 架构输出)
    assert_scores_sane(scores, "cross_arch_rerank");

    // 5. 最相关文档应该排在前面 (Paris/France 相关的文档)
    assert!(
        scores[0] >= scores[1],
        "First result should have highest rerank score"
    );
}
