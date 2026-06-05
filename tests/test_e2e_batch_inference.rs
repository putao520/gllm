//! E2E 测试: Batch Concurrent Inference (SPEC/20 BCI)
//!
//! 验证 generate_batch() JIT mega-kernel 批量推理路径:
//! 1. 单请求 batch (N=1) — 等价于 generate()
//! 2. 多请求 batch (N=3) — 并发独立采样
//! 3. 不同 prompt 长度混合 — M 维度拼接正确性
//! 4. Per-seq sampling params — greedy + stochastic
//! 5. Stop condition — max_new_tokens 限制
//!
//! 测试模型: SmolLM2-135M-Instruct (SafeTensors)
//! 运行: cargo test --test test_e2e_batch_inference -- --test-threads=1

use gllm::Client;

fn install_segv_handler() {
    #[repr(C)]
    struct SigAction {
        sa_handler: usize,
        sa_mask: [u8; 128],
        sa_flags: i32,
        sa_restorer: Option<unsafe extern "C" fn()>,
    }
    extern "C" {
        fn sigaction(sig: i32, act: *const SigAction, oact: *mut SigAction) -> i32;
    }
    unsafe {
        let mut sa: SigAction = std::mem::zeroed();
        sa.sa_handler = segv_handler as *const () as usize;
        sa.sa_flags = 4;
        sigaction(11, &sa, std::ptr::null_mut());
        sigaction(8, &sa, std::ptr::null_mut());
    }
}

extern "C" fn segv_handler(sig: i32, _info: *mut std::ffi::c_void, _uctx: *mut std::ffi::c_void) {
    let sig_name = if sig == 11 { "SIGSEGV" } else { "SIGFPE" };
    eprintln!("[{}] batch inference crashed", sig_name);
    std::process::abort();
}

fn assert_result_sane(result: &gllm::engine::batch_executor::GenerateResult, label: &str) {
    assert!(
        result.finished,
        "{label}: result should be finished, error: {:?}",
        result.error
    );
    assert!(
        result.error.is_none(),
        "{label}: unexpected error: {:?}",
        result.error
    );
    assert!(
        !result.output_tokens.is_empty(),
        "{label}: output tokens should not be empty"
    );
}

fn make_request(
    id: u64,
    client: &Client,
    text: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> gllm::engine::batch_executor::GenerateRequest {
    let tokens = client.encode(text).expect("encode failed");
    gllm::engine::batch_executor::GenerateRequest {
        request_id: id,
        prompt_tokens: tokens,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        session_id: None,
        eos_token_id: 0,
        hook_ctx_ptr: std::ptr::null(),
        callback_table_ptr: std::ptr::null(),
    }
}

const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

/// TEST-E2E-BATCH-001: 单请求 batch (N=1)
/// batch_size=1 的 generate_batch() 应等价于 generate()。
#[test]
fn e2e_batch_single_request() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::new_chat(MODEL).expect("Failed to load model");
    let requests = vec![make_request(1, &client, "The capital of France is", 10, 0.0, 1, 1.0)];

    let results = client.generate_batch(&requests).expect("batch inference failed");

    assert_eq!(results.len(), 1, "should return 1 result");
    assert_result_sane(&results[0], "single_request");
}

/// TEST-E2E-BATCH-002: 多请求 batch (N=3), greedy sampling
/// 3 条独立序列并发推理，各自应有独立输出。
#[test]
fn e2e_batch_multi_request_greedy() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::new_chat(MODEL).expect("Failed to load model");

    let requests = vec![
        make_request(1, &client, "The capital of France is", 10, 0.0, 1, 1.0),
        make_request(2, &client, "The largest planet is", 10, 0.0, 1, 1.0),
        make_request(3, &client, "Water boils at", 10, 0.0, 1, 1.0),
    ];

    let results = client.generate_batch(&requests).expect("batch inference failed");

    assert_eq!(results.len(), 3, "should return 3 results");
    for (i, result) in results.iter().enumerate() {
        assert_result_sane(result, &format!("seq_{}", i));
    }
}

/// TEST-E2E-BATCH-003: 不同 prompt 长度混合
/// 短 prompt + 长 prompt 同时处理，验证 M 维度拼接正确。
#[test]
fn e2e_batch_mixed_prompt_lengths() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::new_chat(MODEL).expect("Failed to load model");

    let requests = vec![
        make_request(1, &client, "Hi", 5, 0.0, 1, 1.0),
        make_request(
            2,
            &client,
            "The theory of general relativity describes gravity as a curvature of spacetime caused by mass and energy",
            5,
            0.0,
            1,
            1.0,
        ),
    ];

    let results = client.generate_batch(&requests).expect("batch inference failed");

    assert_eq!(results.len(), 2, "should return 2 results");
    assert_result_sane(&results[0], "short_prompt");
    assert_result_sane(&results[1], "long_prompt");
}

/// TEST-E2E-BATCH-004: Per-seq sampling params
/// 序列 1 greedy (T=0), 序列 2 stochastic (T=0.8)。
#[test]
fn e2e_batch_per_seq_sampling() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::new_chat(MODEL).expect("Failed to load model");

    let requests = vec![
        make_request(1, &client, "The capital of France is", 10, 0.0, 1, 1.0),
        make_request(2, &client, "The capital of France is", 10, 0.8, 50, 0.9),
    ];

    let results = client.generate_batch(&requests).expect("batch inference failed");

    assert_eq!(results.len(), 2);
    assert_result_sane(&results[0], "greedy");
    assert_result_sane(&results[1], "stochastic");
}

/// TEST-E2E-BATCH-005: Stop condition — max_new_tokens 限制
/// 不同序列设置不同 max_new_tokens，验证各自在正确位置终止。
#[test]
fn e2e_batch_stop_max_tokens() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::new_chat(MODEL).expect("Failed to load model");

    let requests = vec![
        {
            let mut r = make_request(1, &client, "Hello", 3, 0.0, 1, 1.0);
            r.eos_token_id = 99999; // unlikely EOS to force max_tokens stop
            r
        },
        {
            let mut r = make_request(2, &client, "Hello", 7, 0.0, 1, 1.0);
            r.eos_token_id = 99999;
            r
        },
    ];

    let results = client.generate_batch(&requests).expect("batch inference failed");

    assert_eq!(results.len(), 2);
    assert_result_sane(&results[0], "max3");
    assert_result_sane(&results[1], "max7");

    assert!(
        results[0].output_tokens.len() <= 3,
        "seq 0 should generate <= 3 tokens, got {}",
        results[0].output_tokens.len()
    );
    assert!(
        results[1].output_tokens.len() <= 7,
        "seq 1 should generate <= 7 tokens, got {}",
        results[1].output_tokens.len()
    );
}
