//! Pipeline 测试：直接用本地模型跑 Client API，哪里断了就是真实 bug。

use gllm::Client;

/// TEST-INFERENCE-004: 本地 GGUF 管线端到端
/// **关联需求**: REQ-CORE-001
/// **测试类型**: 正向
/// **期望结果**: 本地 GGUF 模型文件成功加载并完成推理
#[test]
#[ignore]
fn test_local_gguf_pipeline() {
    let client = Client::new_embedding("test_models/gguf").expect("load gguf model");
    let response = client
        .embeddings(["hello world"])
        .generate()
        .expect("gguf embedding");
    assert!(!response.embeddings.is_empty());
}

/// TEST-INFERENCE-005: 本地 SafeTensors 管线端到端
/// **关联需求**: REQ-CORE-001
/// **测试类型**: 正向
/// **期望结果**: 本地 SafeTensors 模型文件成功加载并完成推理
#[test]
#[ignore]
fn test_local_safetensors_pipeline() {
    let client = Client::new_embedding("test_models/safetensors").expect("load safetensors model");
    let response = client
        .embeddings(["hello world"])
        .generate()
        .expect("safetensors embedding");
    assert!(!response.embeddings.is_empty());
}

/// TEST-INFERENCE-006: 本地 ONNX 管线端到端
/// **关联需求**: REQ-CORE-001
/// **测试类型**: 正向
/// **期望结果**: 本地 ONNX 模型文件成功加载并完成推理
#[test]
#[ignore]
fn test_local_onnx_pipeline() {
    let client = Client::new_embedding("test_models/onnx").expect("load onnx model");
    let response = client
        .embeddings(["hello world"])
        .generate()
        .expect("onnx embedding");
    assert!(!response.embeddings.is_empty());
}

/// 输出正确性验证：检测退化输出（全零、全相同、NaN/Inf）
fn assert_embedding_sane(emb: &[f32], label: &str) {
    assert!(!emb.is_empty(), "{label}: embedding is empty");

    // 不能有 NaN 或 Inf
    for (i, &v) in emb.iter().enumerate() {
        assert!(v.is_finite(), "{label}: NaN/Inf at index {i}: {v}");
    }

    // 不能全零
    let all_zero = emb.iter().all(|&v| v == 0.0);
    assert!(!all_zero, "{label}: all zeros");

    // 不能全相同值
    let first = emb[0];
    let all_same = emb.iter().all(|&v| v == first);
    assert!(!all_same, "{label}: all same value ({first})");

    // 值域应该合理（embedding 通常在 [-10, 10] 范围内）
    let max_abs = emb.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 100.0,
        "{label}: max absolute value {max_abs} is suspiciously large"
    );

    // 方差不能太小（不能是近似常数）
    let mean = emb.iter().sum::<f32>() / emb.len() as f32;
    let variance = emb.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / emb.len() as f32;
    assert!(
        variance > 1e-10,
        "{label}: variance {variance} is near zero — output is degenerate"
    );
}

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

/// TEST-INFERENCE-007: SafeTensors 输出合理性检查
/// **关联需求**: REQ-CORE-001
/// **测试类型**: 正向
/// **期望结果**: SafeTensors 模型输出的 embedding 向量满足合理性约束
#[test]
#[ignore]
fn test_output_sanity_safetensors() {
    let client = Client::new_embedding("test_models/safetensors").expect("load");
    let r = client
        .embeddings(["hello world", "goodbye moon"])
        .generate()
        .expect("embed");

    let emb1 = &r.embeddings[0].embedding;
    let emb2 = &r.embeddings[1].embedding;

    assert_embedding_sane(emb1, "safetensors[0]");
    assert_embedding_sane(emb2, "safetensors[1]");
    assert_eq!(emb1.len(), 384);

    // 不同文本的 embedding 不能太相似
    let sim = cosine_similarity(emb1, emb2);
    assert!(
        sim < 0.999,
        "safetensors: different texts have cosine similarity {sim} — too similar"
    );
}

/// TEST-INFERENCE-008: ONNX 输出合理性检查
/// **关联需求**: REQ-CORE-001
/// **测试类型**: 正向
/// **期望结果**: ONNX 模型输出的 embedding 向量满足合理性约束
#[test]
#[ignore]
fn test_output_sanity_onnx() {
    let client = Client::new_embedding("test_models/onnx").expect("load");
    let r = client
        .embeddings(["hello world", "goodbye moon"])
        .generate()
        .expect("embed");

    let emb1 = &r.embeddings[0].embedding;
    let emb2 = &r.embeddings[1].embedding;

    assert_embedding_sane(emb1, "onnx[0]");
    assert_embedding_sane(emb2, "onnx[1]");
    assert_eq!(emb1.len(), 384);

    let sim = cosine_similarity(emb1, emb2);
    assert!(
        sim < 0.999,
        "onnx: different texts have cosine similarity {sim} — too similar"
    );
}

/// TEST-INFERENCE-009: GGUF 输出合理性检查
/// **关联需求**: REQ-CORE-001
/// **测试类型**: 正向
/// **期望结果**: GGUF 模型输出的 embedding 向量满足合理性约束
#[test]
#[ignore]
fn test_output_sanity_gguf() {
    let client = Client::new_embedding("test_models/gguf").expect("load");
    let r = client
        .embeddings(["hello world", "goodbye moon"])
        .generate()
        .expect("embed");

    let emb1 = &r.embeddings[0].embedding;
    let emb2 = &r.embeddings[1].embedding;

    assert_embedding_sane(emb1, "gguf[0]");
    assert_embedding_sane(emb2, "gguf[1]");

    let sim = cosine_similarity(emb1, emb2);
    assert!(
        sim < 0.999,
        "gguf: different texts have cosine similarity {sim} — too similar"
    );
}

/// TEST-INFERENCE-010: 跨格式一致性 ONNX vs SafeTensors
/// **关联需求**: REQ-CORE-001
/// **测试类型**: 正向
/// **期望结果**: ONNX 和 SafeTensors 格式的推理结果余弦相似度 > 0.99
#[test]
#[ignore]
fn test_cross_format_consistency_onnx_vs_safetensors() {
    // ONNX 和 SafeTensors 是同一个模型（MiniLM-L6-H384），输出应该高度一致
    let st_client = Client::new_embedding("test_models/safetensors").expect("load st");
    let onnx_client = Client::new_embedding("test_models/onnx").expect("load onnx");

    let st_r = st_client
        .embeddings(["hello world"])
        .generate()
        .expect("st embed");
    let onnx_r = onnx_client
        .embeddings(["hello world"])
        .generate()
        .expect("onnx embed");

    let st_emb = &st_r.embeddings[0].embedding;
    let onnx_emb = &onnx_r.embeddings[0].embedding;

    assert_eq!(st_emb.len(), onnx_emb.len());

    // 同模型同输入，余弦相似度应该 > 0.99
    let sim = cosine_similarity(st_emb, onnx_emb);
    eprintln!("[cross-format] cosine similarity: {sim}");
    assert!(
        sim > 0.99,
        "ONNX vs SafeTensors cosine similarity {sim} — same model should produce near-identical output"
    );

    // 逐元素最大误差
    let max_diff = st_emb
        .iter()
        .zip(onnx_emb)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[cross-format] max element-wise diff: {max_diff}");
}

/// TEST-INFERENCE-011: 确定性输出验证
/// **关联需求**: REQ-CORE-001
/// **测试类型**: 正向
/// **期望结果**: 相同输入多次推理产生完全相同的输出
#[test]
#[ignore]
fn test_deterministic_output() {
    // 同一输入跑两次，结果必须完全一致
    let client = Client::new_embedding("test_models/safetensors").expect("load");

    let r1 = client
        .embeddings(["determinism test"])
        .generate()
        .expect("run 1");
    let r2 = client
        .embeddings(["determinism test"])
        .generate()
        .expect("run 2");

    let emb1 = &r1.embeddings[0].embedding;
    let emb2 = &r2.embeddings[0].embedding;

    assert_eq!(emb1, emb2, "same input should produce identical output");
}

use gllm::loader::{GgufLoader, TensorProvider};

/// TEST-INFERENCE-012: GGUF 张量调试信息
/// **关联需求**: REQ-CORE-001
/// **测试类型**: 正向
/// **期望结果**: 成功加载并打印 GGUF 模型的张量调试信息
#[test]
#[ignore]
fn debug_gguf_tensors() {
    let reader = GgufLoader::from_files(&[std::path::PathBuf::from("test_models/gguf/all-MiniLM-L6-v2-Q4_K_M.gguf")]).expect("open gguf");
    for meta in reader.iter_tensors() {
        println!("  {} {:?} {:?}", meta.name, meta.shape, meta.dtype);
    }
}

#[test]
#[ignore]
fn debug_onnx_tensors() {
    let loader = gllm::loader::onnx::OnnxLoader::from_path(
        std::path::Path::new("test_models/onnx/model.onnx"),
    ).expect("open onnx");
    for meta in <gllm::loader::onnx::OnnxLoader as gllm::loader::TensorProvider>::iter_tensors(&loader) {
        println!("  {} {:?} {:?}", meta.name, meta.shape, meta.dtype);
    }
}

#[test]
#[ignore]
fn debug_onnx_graph() {
    let loader = gllm::loader::onnx::OnnxLoader::from_path(
        std::path::Path::new("test_models/onnx/model.onnx"),
    ).expect("open onnx");
    let graph = loader.graph();
    for node in &graph.nodes {
        if node.op_type == "MatMul" || node.op_type == "Add" || node.op_type == "Gemm" {
            println!("  {} op={} inputs={:?} outputs={:?}", node.name, node.op_type, node.inputs, node.outputs);
        }
    }
}

#[test]
#[ignore]
fn debug_safetensors_tensors() {
    let loader = gllm::loader::SafeTensorsLoader::from_files(
        &[std::path::PathBuf::from("test_models/safetensors/model.safetensors")],
        gllm::loader::ParallelLoader::new(false),
    ).expect("open safetensors");
    for meta in <gllm::loader::SafeTensorsLoader as gllm::loader::TensorProvider>::iter_tensors(&loader) {
        println!("  {} {:?} {:?}", meta.name, meta.shape, meta.dtype);
    }
}
