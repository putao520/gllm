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
        &[BatchResult::continue_with_token(req_a, 2001)],
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
            BatchResult::continue_with_token(req_a, 2002),
            BatchResult::continue_with_token(req_b, 3001),
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
            BatchResult::complete(req_a, Some(2003)), // A finishes
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

#[test]
fn test_oom_handling_in_batching() {
    // 1. Setup small memory: 4 blocks total.
    let mut scheduler = PagedScheduler::new(4, 1, HGALConfig::default());
    let mut batcher = ContinuousBatcher::new();

    // 2. Add Req A (2 blocks)
    let req_a = 1;
    batcher.enqueue(make_sequence(req_a, 2));

    // T=1: A Prefill (Uses 2 blocks. Free: 2)
    let batch_1 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(batch_1.requests, vec![req_a]);
    batcher.update_batch(
        &mut scheduler,
        &[BatchResult::continue_with_token(req_a, 100)],
    );

    // T=2: A Decode (Uses 3 blocks. Free: 1)
    // We must trigger a decode step for A to actually consume the 3rd block in the scheduler
    let batch_1b = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(batch_1b.requests, vec![req_a]);
    batcher.update_batch(
        &mut scheduler,
        &[BatchResult::continue_with_token(req_a, 101)],
    );

    // 3. Add Req B (2 blocks) - Should fail to admit due to OOM (only 1 free block left)
    // Free blocks: 4 - 3 = 1. Req B needs 2.
    let req_b = 2;
    batcher.enqueue(make_sequence(req_b, 2));

    // T=3: Try to admit B (fail) + A Decode (Uses 4 blocks. Free: 0)
    let batch_2 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );

    // B should stay in waiting because admit_waiting checks capacity
    // A should continue and take the last block
    assert_eq!(
        batch_2.requests,
        vec![req_a],
        "B should wait, A should continue"
    );

    batcher.update_batch(
        &mut scheduler,
        &[BatchResult::continue_with_token(req_a, 102)],
    );

    // 4. A uses 4 blocks (Full). Free = 0.

    // T=4: A tries to grow again -> OOM
    // B still waiting (needs 2, free 0).
    let batch_3 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert!(
        batch_3.requests.is_empty(),
        "Should return empty batch if A is OOM and B cannot fit"
    );

    // 5. Finish A to free space
    // A releases 4 blocks. Free = 4.
    batcher.update_batch(&mut scheduler, &[BatchResult::complete(req_a, None)]);

    // 6. Now B should be admitted
    let batch_4 = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(
        batch_4.requests,
        vec![req_b],
        "B should be admitted after A finishes"
    );
}
