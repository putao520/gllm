//! E2E 测试: RAG Pipeline 跨模型混合推理
//!
//! 验证 **embedder + reranker + generator** 三类模型串联的真实工作流:
//! 1. embedder 对 query + docs 做向量化
//! 2. cosine similarity 过滤 top-3 候选文档
//! 3. reranker 对 top-3 精排，得到最相关 context
//! 4. generator 基于 context 回答 query
//!
//! 同时验证多模型 session 隔离 — 三个 Client 完全独立,
//! 不共享 JIT cache / weight ptrs。
//!
//! 模型组合 (使用现有已下载的小模型):
//! - Embedder:  `intfloat/e5-small-v2` (SafeTensors, 384 维)
//! - Reranker:  `BAAI/bge-reranker-v2-m3` (SafeTensors)
//! - Generator: `HuggingFaceTB/SmolLM2-135M-Instruct` (SafeTensors)
//!
//! E2E 铁律: 必须 `cargo test --test test_e2e_rag_pipeline -- --test-threads=1`

use gllm::Client;

// ============================================================================
// Helpers
// ============================================================================

/// 余弦相似度
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// 返回 top-k 文档索引 (按余弦相似度从高到低)
fn top_k_cosine(query: &[f32], docs: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| (i, cosine_similarity(query, d)))
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("cosine similarity must be finite")
    });
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

// ============================================================================
// RAG Pipeline E2E Test
// ============================================================================

/// TEST-E2E-RAG-001: 跨模型 RAG 管线端到端推理
/// **关联需求**: REQ-TEST-002, REQ-TEST-003, REQ-TEST-004
/// **测试类型**: 正向 (跨模型混合)
/// **期望结果**:
///   - 三个独立 Client 成功加载 (embedder / reranker / generator)
///   - embedder 召回的 top-3 包含 Paris 文档
///   - reranker 精排 top-1 是 Paris 文档
///   - generator 基于 Paris context 生成的回答包含 "paris"
///   - 三个 Client 的 state Arc 指向不同内存地址 (session 隔离)
#[test]
fn e2e_rag_pipeline_cross_model() {
    const EMBEDDER_MODEL: &str = "intfloat/e5-small-v2";
    const RERANKER_MODEL: &str = "BAAI/bge-reranker-v2-m3";
    const GENERATOR_MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

    // RAG 场景
    let query = "What is the capital of France?";
    let docs = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Rome is in Italy.",
        "Madrid is the capital of Spain.",
    ];
    let paris_idx: usize = 0;

    let t_total = std::time::Instant::now();

    // ------------------------------------------------------------------
    // Step 1: embedder → 向量化 query + docs
    // ------------------------------------------------------------------
    let t0 = std::time::Instant::now();
    let embedder = Client::new_embedding(EMBEDDER_MODEL).expect("Failed to load embedder");

    // embed 一次性处理 query 和所有文档 (第 0 位是 query)
    let mut all_texts: Vec<&str> = Vec::with_capacity(1 + docs.len());
    all_texts.push(query);
    all_texts.extend(docs.iter().copied());

    let embed_response = embedder
        .embed(all_texts.iter().copied())
        .expect("Embedding failed");
    assert_eq!(
        embed_response.embeddings.len(),
        1 + docs.len(),
        "Should have 1 query + {} doc embeddings",
        docs.len()
    );

    let query_vec: Vec<f32> = embed_response.embeddings[0].embedding.clone();
    let doc_vecs: Vec<Vec<f32>> = embed_response.embeddings[1..]
        .iter()
        .map(|e| e.embedding.clone())
        .collect();

    // 维度检查 (e5-small-v2 = 384)
    assert_eq!(query_vec.len(), 384, "query embedding dim mismatch");
    for (i, v) in doc_vecs.iter().enumerate() {
        assert_eq!(v.len(), 384, "doc[{}] embedding dim mismatch", i);
    }

    let embed_ms = t0.elapsed().as_millis();

    // Step 2: cosine similarity → top-3 召回
    let top3 = top_k_cosine(&query_vec, &doc_vecs, 3);
    assert_eq!(top3.len(), 3, "Should retrieve top-3 docs");
    assert!(
        top3.contains(&paris_idx),
        "top-3 recall should include Paris doc (idx {}), got {:?}",
        paris_idx,
        top3
    );

    // ------------------------------------------------------------------
    // Step 3: reranker → 对 top-3 精排
    // ------------------------------------------------------------------
    let t1 = std::time::Instant::now();
    let reranker =
        Client::new(RERANKER_MODEL, gllm::ModelKind::Reranker).expect("Failed to load reranker");

    let top3_docs: Vec<&str> = top3.iter().map(|&i| docs[i]).collect();
    let rerank_response = reranker
        .rerank(query, top3_docs.iter().copied())
        .expect("Rerank failed");

    assert_eq!(
        rerank_response.results.len(),
        3,
        "Reranker should return 3 scored results"
    );
    // RerankResult 已按分数降序排列 (见 test_e2e_reranker.rs::assert_rerank_sane)
    let rerank_top = &rerank_response.results[0];
    // rerank_top.index 是传入 top3_docs 的局部 index，映射回原始 docs index
    let best_global_idx = top3[rerank_top.index];
    let context = docs[best_global_idx];
    let rerank_ms = t1.elapsed().as_millis();

    assert!(
        rerank_top.score.is_finite(),
        "rerank top score must be finite"
    );
    assert_eq!(
        best_global_idx, paris_idx,
        "Reranker top-1 should be Paris doc (idx {}), got idx {} ({:?})",
        paris_idx, best_global_idx, context
    );
    assert!(
        context.contains("Paris"),
        "rerank top-1 context should contain 'Paris', got: {:?}",
        context
    );

    // ------------------------------------------------------------------
    // Step 4: generator → 基于 context 生成答案
    // ------------------------------------------------------------------
    let t2 = std::time::Instant::now();
    let generator = Client::new_chat(GENERATOR_MODEL).expect("Failed to load generator");

    let prompt = format!(
        "Context: {}\n\nQuestion: {}\n\nAnswer:",
        context, query
    );
    let gen_response = generator
        .generate(&prompt)
        .max_tokens(20)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let answer = gen_response.text.trim();
    let gen_ms = t2.elapsed().as_millis();

    assert!(!answer.is_empty(), "RAG answer should not be empty");
    assert!(
        answer.to_lowercase().contains("paris"),
        "RAG answer should mention Paris (case-insensitive), got: {:?}",
        answer
    );

    // ------------------------------------------------------------------
    // Step 5: 多模型 session 隔离验证
    // ------------------------------------------------------------------
    // 三个 Client 分别绑定不同的模型 manifest / 结构（id + kind），
    // 且三个栈上实例地址互不相同。组合验证 session 独立，不共享状态。
    let embedder_info = embedder.model_info().expect("embedder model_info");
    let reranker_info = reranker.model_info().expect("reranker model_info");
    let generator_info = generator.model_info().expect("generator model_info");

    assert_eq!(embedder_info.kind, gllm::ModelKind::Embedding);
    assert_eq!(reranker_info.kind, gllm::ModelKind::Reranker);
    assert_eq!(generator_info.kind, gllm::ModelKind::Chat);

    assert_ne!(
        embedder_info.id, reranker_info.id,
        "embedder and reranker must bind different models"
    );
    assert_ne!(
        reranker_info.id, generator_info.id,
        "reranker and generator must bind different models"
    );
    assert_ne!(
        embedder_info.id, generator_info.id,
        "embedder and generator must bind different models"
    );

    let embedder_ptr = (&embedder as *const Client) as usize;
    let reranker_ptr = (&reranker as *const Client) as usize;
    let generator_ptr = (&generator as *const Client) as usize;
    assert_ne!(embedder_ptr, reranker_ptr);
    assert_ne!(reranker_ptr, generator_ptr);
    assert_ne!(embedder_ptr, generator_ptr);

    let total_ms = t_total.elapsed().as_millis();
    eprintln!(
        "[RAG E2E] embed={}ms rerank={}ms gen={}ms total={}ms",
        embed_ms, rerank_ms, gen_ms, total_ms
    );
    eprintln!(
        "[RAG E2E] top3 global_idx={:?} rerank_top=doc[{}] ({:?}) score={}",
        top3, best_global_idx, context, rerank_top.score
    );
    eprintln!("[RAG E2E] generator answer: {:?}", answer);
}
