//! E2E 测试: Embedding (嵌入模型)
//!
//! 测试 3 种格式: SafeTensors, GGUF, ONNX
//! 验证同一功能的模型在不同格式下都能正常工作
//!
//! 反作弊检查: NaN/Inf、全零、全同值、方差退化、值域异常、余弦区分度

use gllm::Client;

fn install_segv_handler() {
    #[repr(C)]
    struct SigAction {
        sa_handler: usize,
        sa_flags: i32,
        sa_restorer: Option<unsafe extern "C" fn()>,
        sa_mask: [u8; 128],
    }
    extern "C" {
        fn sigaction(sig: i32, act: *const SigAction, oact: *mut SigAction) -> i32;
    }
    unsafe {
        let mut sa: SigAction = std::mem::zeroed();
        sa.sa_handler = segv_handler as *const () as usize;
        sa.sa_flags = 4;
        sigaction(11, &sa, std::ptr::null_mut());
    }
}

extern "C" fn segv_handler(
    _sig: i32,
    info: *mut std::ffi::c_void,
    uctx: *mut std::ffi::c_void,
) {
    let fault_addr = unsafe { *(info as *const u8).add(16) as *const std::ffi::c_void };
    eprintln!("[SIGSEGV] fault_addr={:p}", fault_addr);
    unsafe {
        let mc = (uctx as *const u8).add(40) as *const u64;
        let r = |i: usize| *mc.add(i);
        eprintln!("[SIGSEGV] RIP=0x{:x} RAX=0x{:x} RCX=0x{:x} RDX=0x{:x}",
            r(16), r(13), r(14), r(12));
        eprintln!("[SIGSEGV] RSI=0x{:x} RDI=0x{:x} R8=0x{:x} R9=0x{:x}",
            r(9), r(8), r(0), r(1));
        eprintln!("[SIGSEGV] R10=0x{:x} R11=0x{:x} R12=0x{:x} R13=0x{:x}",
            r(2), r(3), r(4), r(5));
        eprintln!("[SIGSEGV] R14=0x{:x} R15=0x{:x} RBP=0x{:x} RSP=0x{:x}",
            r(6), r(7), r(10), r(15));
    }
    std::process::exit(139);
}

// ============================================================================
// Anti-cheating helpers
// ============================================================================

/// 检测退化输出: NaN/Inf、全零、全同值、方差过小、值域异常
fn assert_embedding_sane(emb: &[f32], label: &str) {
    assert!(!emb.is_empty(), "{label}: embedding is empty");

    // 1. 禁止 NaN / Inf
    for (i, &v) in emb.iter().enumerate() {
        assert!(v.is_finite(), "{label}: NaN/Inf at index {i}: {v}");
    }

    // 2. 禁止全零
    let all_zero = emb.iter().all(|&v| v == 0.0);
    assert!(!all_zero, "{label}: all zeros — model output is degenerate");

    // 3. 禁止全同值
    let first = emb[0];
    let all_same = emb.iter().all(|&v| v == first);
    assert!(
        !all_same,
        "{label}: all same value ({first}) — model output is degenerate"
    );

    // 4. 值域合理性 (embedding 正常范围 [-100, 100])
    let max_abs = emb.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 100.0,
        "{label}: max absolute value {max_abs} is suspiciously large"
    );

    // 5. 方差不能趋近于零 (近似常数 = 退化)
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

// ============================================================================
// SafeTensors
// ============================================================================

/// TEST-E2E-EMB-001: SafeTensors 格式 embedding 端到端推理
/// **关联需求**: REQ-TEST-003
/// **测试类型**: 正向
/// **期望结果**: 成功加载 SafeTensors 模型并生成有效 embedding 向量
#[test]
fn e2e_embedding_safetensors() {
    install_segv_handler();
    const MODEL: &str = "intfloat/e5-small-v2";

    let client = Client::new_embedding(MODEL).expect("Failed to load SafeTensors model");
    let response = client
        .embed(["Hello, world!", "Test sentence"])
        .expect("Embedding failed");

    assert_eq!(response.embeddings.len(), 2, "Should have 2 embeddings");

    let emb1 = &response.embeddings[0].embedding;
    let emb2 = &response.embeddings[1].embedding;

    // 维度验证
    assert_eq!(emb1.len(), 384, "e5-small embedding dimension should be 384");
    assert_eq!(emb2.len(), 384, "e5-small embedding dimension should be 384");

    // 反退化检查
    assert_embedding_sane(emb1, "safetensors[0]");
    assert_embedding_sane(emb2, "safetensors[1]");

    // 不同文本必须产生不同嵌入 (余弦相似度 < 0.999)
    let sim = cosine_similarity(emb1, emb2);
    assert!(
        sim < 0.999,
        "Different texts have cosine similarity {sim} — too similar, model may not be discriminating"
    );

    // L1 距离兜底
    let l1_diff: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(l1_diff > 0.1, "L1 diff {l1_diff} too small — embeddings nearly identical");
}

// ============================================================================
// GGUF
// ============================================================================

/// TEST-E2E-EMB-002: GGUF 格式 embedding 端到端推理
/// **关联需求**: REQ-TEST-003
/// **测试类型**: 正向
/// **期望结果**: 成功加载 GGUF 模型并生成有效 embedding 向量
#[test]
fn e2e_embedding_gguf() {
    const MODEL: &str = "Qwen/Qwen3-Embedding-0.6B-GGUF";

    let client = Client::new_embedding(MODEL).expect("Failed to load GGUF model");
    let manifest = client.manifest().expect("Failed to read manifest");
    assert_eq!(manifest.kind, gllm::ModelKind::Embedding);

    let response = client
        .embed(["Hello, world!", "Rust programming language"])
        .expect("GGUF embedding inference failed");

    assert_eq!(response.embeddings.len(), 2, "Should have 2 embeddings");

    let emb1 = &response.embeddings[0].embedding;
    let emb2 = &response.embeddings[1].embedding;

    // 维度一致性
    assert!(!emb1.is_empty(), "Embedding should not be empty");
    assert_eq!(emb1.len(), emb2.len(), "Both embeddings should have the same dimension");

    // 反退化检查
    assert_embedding_sane(emb1, "gguf[0]");
    assert_embedding_sane(emb2, "gguf[1]");

    // L2 norm 正值 (非零输出)
    let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm1 > 0.0, "L2 norm should be positive, got {}", norm1);
    assert!(norm2 > 0.0, "L2 norm should be positive, got {}", norm2);

    // 不同文本必须产生不同嵌入
    let sim = cosine_similarity(emb1, emb2);
    assert!(
        sim < 0.999,
        "Different texts have cosine similarity {sim} — too similar"
    );

    let l1_diff: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(l1_diff > 0.1, "L1 diff {l1_diff} too small");
}

// ============================================================================
// ONNX
// ============================================================================

/// TEST-E2E-EMB-003: ONNX 格式 embedding 端到端推理
/// **关联需求**: REQ-TEST-003
/// **测试类型**: 正向
/// **期望结果**: 成功加载 ONNX 模型并生成有效 embedding 向量
#[test]
fn e2e_embedding_onnx() {
    const MODEL: &str = "intfloat/multilingual-e5-small";

    let client = Client::new_embedding(MODEL).expect("Failed to load ONNX model");
    let response = client
        .embed(["test query", "test document"])
        .expect("Embedding failed");

    assert_eq!(response.embeddings.len(), 2, "Should have 2 embeddings");

    let emb1 = &response.embeddings[0].embedding;
    let emb2 = &response.embeddings[1].embedding;

    // 维度验证
    assert_eq!(emb1.len(), 384, "e5-small embedding dimension should be 384");
    assert_eq!(emb2.len(), 384, "e5-small embedding dimension should be 384");

    // 反退化检查
    assert_embedding_sane(emb1, "onnx[0]");
    assert_embedding_sane(emb2, "onnx[1]");

    // 不同文本必须产生不同嵌入
    let sim = cosine_similarity(emb1, emb2);
    assert!(
        sim < 0.999,
        "Different texts have cosine similarity {sim} — too similar"
    );

    let l1_diff: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(l1_diff > 0.01, "L1 diff {l1_diff} too small");
}
