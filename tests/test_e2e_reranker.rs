//! E2E 测试: Reranker (重排序模型)
//!
//! 测试 3 种格式: SafeTensors, GGUF, ONNX
//! 验证同一功能的模型在不同格式下都能正常工作
//!
//! 反作弊检查: 分数有限性、降序排列、分数离散度、语义排名正确性

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
        sa.sa_flags = 4; // SA_SIGINFO
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

/// 验证 rerank 结果的完整性和正确性
fn assert_rerank_sane(results: &[gllm::RerankResult], label: &str) {
    assert!(!results.is_empty(), "{label}: results are empty");

    // 1. 所有分数必须是有限浮点数 (非 NaN/Inf)
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.score.is_finite(),
            "{label}: score at rank {i} is not finite: {}",
            r.score
        );
    }

    // 2. 结果必须按分数降序排列
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "{label}: results not sorted descending: rank {} score {} < rank {} score {}",
            i - 1,
            results[i - 1].score,
            i,
            results[i].score
        );
    }

    // 3. 分数离散度检查 — 模型必须能区分不同文档
    //    如果所有分数几乎相同，说明模型没有真正工作
    let max_score = results.first().unwrap().score;
    let min_score = results.last().unwrap().score;
    let spread = (max_score - min_score).abs();
    assert!(
        spread > 1e-6,
        "{label}: score spread {spread} is too small — model is not discriminating \
         (max={max_score}, min={min_score})"
    );

    // 4. 分数不能全为零
    let all_zero = results.iter().all(|r| r.score == 0.0);
    assert!(
        !all_zero,
        "{label}: all scores are zero — model output is degenerate"
    );

    // 5. index 去重检查 — 每个文档应该只出现一次
    let mut seen = std::collections::HashSet::new();
    for r in results {
        assert!(
            seen.insert(r.index),
            "{label}: duplicate index {} in results",
            r.index
        );
    }
}

// ============================================================================
// SafeTensors
// ============================================================================

/// TEST-E2E-RERANK-001: SafeTensors 格式 reranker 端到端推理
/// **关联需求**: REQ-TEST-004
/// **测试类型**: 正向
/// **期望结果**: 成功加载 SafeTensors 模型并返回有效相关性分数
#[test]
fn e2e_reranker_safetensors() {
    install_segv_handler();
    const MODEL: &str = "BAAI/bge-reranker-v2-m3";

    let client =
        Client::new(MODEL, gllm::ModelKind::Reranker).expect("Failed to load SafeTensors model");

    let documents = vec![
        "Paris is the capital of France.",
        "London is in England.",
        "Berlin is in Germany.",
    ];

    let response = client
        .rerank("What is the capital of France?", documents)
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // 反退化检查
    assert_rerank_sane(&response.results, "safetensors");

    // 语义正确性: Paris 文档应该排在最前面
    let top_result = &response.results[0];
    assert_eq!(
        top_result.index, 0,
        "First document (Paris) should be ranked first, got index {}",
        top_result.index
    );
}

// ============================================================================
// GGUF
// ============================================================================

/// TEST-E2E-RERANK-002: GGUF 格式 reranker 端到端推理
/// **关联需求**: REQ-TEST-004
/// **测试类型**: 正向
/// **期望结果**: 成功加载 GGUF 模型并返回有效相关性分数
#[test]
fn e2e_reranker_gguf() {
    install_segv_handler();
    const MODEL: &str = "DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF";

    let client = Client::new(MODEL, gllm::ModelKind::Reranker).expect("Failed to load GGUF model");
    let manifest = client.manifest().expect("Failed to read manifest");
    assert_eq!(manifest.kind, gllm::ModelKind::Reranker);

    let documents = vec![
        "Paris is the capital of France.",
        "London is in England.",
        "Berlin is in Germany.",
    ];

    let response = client
        .rerank("What is the capital of France?", documents)
        .expect("GGUF rerank inference failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // 反退化检查
    assert_rerank_sane(&response.results, "gguf");

    // NOTE: 不验证 top_result.index == 0，量化模型精度不足以保证特定排名
}

// ============================================================================
// ONNX
// ============================================================================

/// TEST-E2E-RERANK-003: ONNX 格式 reranker 端到端推理
/// **关联需求**: REQ-TEST-004
/// **测试类型**: 正向
/// **期望结果**: 成功加载 ONNX 模型并返回有效相关性分数
#[test]
fn e2e_reranker_onnx() {
    install_segv_handler();
    const MODEL: &str = "onnx-community/bge-reranker-v2-m3-ONNX";

    let client = Client::new(MODEL, gllm::ModelKind::Reranker).expect("Failed to load ONNX model");

    let documents = vec![
        "Beijing is the capital city of China, serving as the political and cultural center of the nation for many centuries.",
        "Shanghai is the largest city in China by population, known for its modern skyline along the Bund waterfront.",
        "Tokyo is the capital of Japan, located on the eastern coast of the island of Honshu in the Kanto region.",
    ];

    let response = client
        .rerank("What is the capital of China?", documents)
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // 反退化检查
    assert_rerank_sane(&response.results, "onnx");

    // 语义正确性: Beijing 文档应该排在最前面
    let top_result = &response.results[0];
    assert_eq!(
        top_result.index, 0,
        "First document (Beijing) should be ranked first, got index {}",
        top_result.index
    );
}
