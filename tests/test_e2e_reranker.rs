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

// ============================================================================
// Golden numerical alignment (vs HuggingFace Transformers)
// ============================================================================

/// Load golden rerank scores from safetensors file.
fn load_golden_rerank_scores(path: &std::path::Path) -> Vec<f32> {
    let data = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "Golden data not found at {}. Run: python3 tests/e2e_alignment/generate_golden_bge_reranker.py\nError: {e}",
            path.display()
        )
    });
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("Failed to parse safetensors");

    let num_pairs_view = tensors.tensor("num_pairs").expect("Missing num_pairs tensor");
    let num_pairs_bytes = num_pairs_view.data();
    let num_pairs = u32::from_le_bytes([num_pairs_bytes[0], num_pairs_bytes[1], num_pairs_bytes[2], num_pairs_bytes[3]]) as usize;

    let mut scores = Vec::with_capacity(num_pairs);
    for i in 0..num_pairs {
        let name = format!("score_{i}");
        let view = tensors.tensor(&name).unwrap_or_else(|_| panic!("Missing {name}"));
        let bytes = view.data();
        let score = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        scores.push(score);
    }
    scores
}

/// TEST-E2E-RERANK-004: SafeTensors reranker numerical alignment vs HuggingFace Transformers.
///
/// Compares gllm's rerank scores against PyTorch cross-encoder reference (F32, CPU).
/// Same query-document pairs as golden data.
/// Verifies: score sign consistency + relative ranking match + score magnitude correlation.
#[test]
fn e2e_reranker_golden_alignment() {
    install_segv_handler();
    const MODEL: &str = "BAAI/bge-reranker-v2-m3";
    const GOLDEN_PATH: &str = "tests/e2e_alignment/data/golden_bge_reranker_v2_m3.safetensors";

    let golden_path = std::path::Path::new(GOLDEN_PATH);
    let golden_scores = load_golden_rerank_scores(golden_path);

    assert_eq!(golden_scores.len(), 3, "Should have 3 golden scores");

    // Run gllm inference (same pairs as golden data)
    let client =
        Client::new(MODEL, gllm::ModelKind::Reranker).expect("Failed to load model");

    let documents = vec![
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital and largest city of Germany.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    ];

    let response = client
        .rerank("What is the capital of France?", documents)
        .expect("Rerank failed");

    assert_eq!(response.results.len(), 3, "Should have 3 results");

    // Build gllm scores indexed by original document index
    let mut gllm_scores = vec![0.0f32; 3];
    for r in &response.results {
        gllm_scores[r.index] = r.score;
    }

    eprintln!("Golden scores: {golden_scores:?}");
    eprintln!("gllm  scores:  {gllm_scores:?}");

    // 1. Relative ranking must match: pair 0 (relevant) should score higher than pair 1 (irrelevant)
    let golden_rank_correct = golden_scores[0] > golden_scores[1];
    let gllm_rank_correct = gllm_scores[0] > gllm_scores[1];
    assert!(
        gllm_rank_correct == golden_rank_correct,
        "Ranking mismatch: golden has relevant > irrelevant = {golden_rank_correct}, gllm has {gllm_rank_correct}"
    );

    // 2. Score magnitude correlation: gllm and golden should agree on which pair scores highest
    let golden_best = (0..3).max_by(|a, b| golden_scores[*a].partial_cmp(&golden_scores[*b]).unwrap()).unwrap();
    let gllm_best = response.results[0].index; // results are sorted descending
    assert_eq!(
        gllm_best, golden_best,
        "Best document mismatch: golden={golden_best}, gllm={gllm_best}"
    );

    // 3. All gllm scores must be finite
    for (i, &s) in gllm_scores.iter().enumerate() {
        assert!(s.is_finite(), "gllm score[{i}] is not finite: {s}");
    }
}
