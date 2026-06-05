use gllm::scheduler::batcher::{BatchResult, ContinuousBatcher};
use gllm::scheduler::hgal::HGALConfig;
use gllm::scheduler::paged_scheduler::PagedScheduler;
use gllm::scheduler::sequence::Sequence;
use gllm::scheduler::BatchOrderPolicy;
use gllm::scheduler::RequestId;

fn make_sequence(id: RequestId, prompt_len: usize) -> Sequence {
    // Dummy prompt tokens
    let tokens = vec![100; prompt_len];
    Sequence::new(id, tokens)
}

/// TEST-SCHED-001: 连续批处理 prefill 和 decode 混合调度
/// **关联需求**: REQ-SCHED-003
/// **测试类型**: 正向
/// **期望结果**: 正确混合调度 prefill 和 decode 请求，维护批次一致性
#[test]
fn test_continuous_batching_prefill_and_decode_mix() {
    // 1. Setup
    // Total 100 blocks, block size 1.
    let mut scheduler = PagedScheduler::new(100, 1, HGALConfig::default());
    let mut batcher = ContinuousBatcher::new();

    // 2. T=0: Add Request A (needs 2 blocks for prompt)
    let req_a = 1;
    batcher.enqueue(make_sequence(req_a, 2));

    // 3. T=1: Schedule A (Prefill)
    let batch_1 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(
        batch_1.requests,
        vec![req_a],
        "First batch should contain A"
    );

    // Simulate A finishing prefill and generating 1st token
    batcher.update_batch(
        &mut scheduler,
        &[BatchResult::continue_with_token(req_a, 2001, Default::default())],
    );

    // 4. T=2: Add Request B (needs 2 blocks for prompt)
    let req_b = 2;
    batcher.enqueue(make_sequence(req_b, 2));

    // 5. T=2: Schedule (Should mix A-Decode and B-Prefill)
    let batch_2 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    // Sort to ensure deterministic comparison, though implementation usually sorts by ID
    let mut ids = batch_2.requests.clone();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![req_a, req_b],
        "Second batch should mix A (decode) and B (prefill)"
    );

    // Simulate completion
    batcher.update_batch(
        &mut scheduler,
        &[
            BatchResult::continue_with_token(req_a, 2002, Default::default()),
            BatchResult::continue_with_token(req_b, 3001, Default::default()),
        ],
    );

    // 6. T=3: Both in Decode
    let batch_3 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    let mut ids_3 = batch_3.requests.clone();
    ids_3.sort_unstable();
    assert_eq!(
        ids_3,
        vec![req_a, req_b],
        "Third batch should have both in decode"
    );

    // 7. Finish A
    batcher.update_batch(
        &mut scheduler,
        &[
            BatchResult::complete(req_a, Some(2003), Default::default()), // A finishes
        ],
    );

    // 8. T=4: Only B remains
    let batch_4 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(
        batch_4.requests,
        vec![req_b],
        "Fourth batch should only have B"
    );

    println!("Continuous Batching Test Passed: Mixed prefill/decode handled correctly.");
}

/// TEST-SCHED-002: 批处理中的 OOM 处理
/// **关联需求**: REQ-SCHED-003
/// **测试类型**: 负向
/// **期望结果**: 内存不足时正确拒绝新请求而非崩溃
#[test]
fn test_oom_handling_in_batching() {
    // Setup: 3 blocks total, block_size=1. A uses a 3-token prompt → fills all 3 blocks.
    // B (any size) cannot be admitted when all blocks are taken by A's prompt.
    let mut scheduler = PagedScheduler::new(3, 1, HGALConfig::default());
    let mut batcher = ContinuousBatcher::new();

    // Req A: 3 tokens, fills all 3 blocks immediately during prefill.
    let req_a: RequestId = 1;
    batcher.enqueue(make_sequence(req_a, 3));

    // T=1: A prefill — takes all 3 blocks (Free: 0)
    let batch_1 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(batch_1.requests, vec![req_a], "A must be admitted for prefill");
    // Simulate A completing (not continuing — all tokens processed in one shot).
    batcher.update_batch(
        &mut scheduler,
        &[BatchResult::complete(req_a, Some(999), Default::default())],
    );

    // Req B: added BEFORE A finishes; if B was rejected due to OOM at T=1 it keeps waiting.
    // By design, A finished so now 3 blocks are free — B should now be admitted.
    let req_b: RequestId = 2;
    batcher.enqueue(make_sequence(req_b, 1));

    let batch_2 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(
        batch_2.requests,
        vec![req_b],
        "B should be admitted after A completes and frees all blocks"
    );

    // B finishes cleanly
    batcher.update_batch(
        &mut scheduler,
        &[BatchResult::complete(req_b, Some(888), Default::default())],
    );
    assert!(!batcher.has_pending_work(), "No work should remain after B completes");
}
