use gllm::scheduler::hgal::HGALConfig;
use gllm::scheduler::paged_scheduler::{PagedScheduler, SchedulerError};
use gllm::scheduler::types::{GroupState, SequenceGroup};
use gllm_kernels::kernel_types::RequestId;
use std::time::Instant;

fn make_group(id: RequestId, context_len: usize) -> SequenceGroup {
    SequenceGroup {
        id,
        pages: Vec::new(),
        state: GroupState::Running,
        access_count: 0,
        last_access: Instant::now(),
        is_pinned: false,
        context_len,
    }
}

#[test]
fn test_swap_out_and_restore_flow() {
    // Total 4 blocks, block size 1 token
    // This allows exact control over memory pressure
    let mut scheduler = PagedScheduler::new(4, 1, HGALConfig::default());

    // 1. Add Sequence A (needs 2 blocks)
    let id_a = 1;
    scheduler
        .add_sequence(make_group(id_a, 2))
        .expect("Failed to add seq A");
    assert_eq!(scheduler.num_free_blocks(), 2);

    // 2. Add Sequence B (needs 2 blocks)
    let id_b = 2;
    scheduler
        .add_sequence(make_group(id_b, 2))
        .expect("Failed to add seq B");
    assert_eq!(scheduler.num_free_blocks(), 0);

    // 3. Try to allocate for B -> Should fail (OOM)
    // allocate_next_token implies growing by 1 token/block (since block size is 1)
    let result = scheduler.allocate_next_token(id_b);
    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(SchedulerError::OutOfMemory { .. })
    ));

    // 4. Trigger Swap Logic: Select victim
    // We need 1 block for B.
    let victims = scheduler.select_victims(1);
    assert!(!victims.is_empty());

    let victim_id = victims[0].0;
    println!("Victim selected: {}", victim_id);

    let victim_ids: Vec<RequestId> = victims.iter().map(|(id, _)| *id).collect();
    scheduler
        .free_victims(&victim_ids)
        .expect("free_victims failed");

    // 5. Verify Victim is Swapped
    // If a sequence with 2 blocks was swapped, we should have 2 free blocks.
    assert_eq!(scheduler.num_free_blocks(), 2);

    // 6. Now allocate for the survivor
    let survivor_id = if victim_id == id_a { id_b } else { id_a };
    scheduler
        .allocate_next_token(survivor_id)
        .expect("Survivor should grow");
    // Survivor grew by 1 block (total 3 used for survivor). Remaining free: 1.
    assert_eq!(scheduler.num_free_blocks(), 1);

    // 7. Now try to resume the victim (Swap In)
    // Victim needs 2 blocks to restore. We only have 1 free.
    // Should fail with OOM.
    let restore_result = scheduler.allocate_next_token(victim_id);
    assert!(restore_result.is_err());
    assert!(matches!(
        restore_result.err(),
        Some(SchedulerError::OutOfMemory { .. })
    ));

    // 8. Make room for victim
    // Swap out the survivor (who now has 3 blocks)
    scheduler
        .free_victims(&[survivor_id])
        .expect("Swap out survivor");
    // Free blocks: 1 (existing) + 3 (survivor) = 4.
    assert_eq!(scheduler.num_free_blocks(), 4);

    // 9. Retry restore victim
    scheduler
        .allocate_next_token(victim_id)
        .expect("Restore victim should succeed now");

    // 10. Verify Pending Swap-ins
    // The scheduler should have recorded the mapping for the backend to reload data
    let swap_ins = scheduler
        .take_pending_swap_in(victim_id)
        .expect("Should have pending swap-ins");
    assert!(!swap_ins.is_empty());
    assert_eq!(swap_ins.len(), 2); // Victim had 2 pages initially.

    // Check key integrity (basic)
    // Keys should be (RequestId << 32) | LogicalIndex
    for (_, key) in swap_ins {
        let req = key >> 32;
        assert_eq!(req, victim_id as u64);
    }

    println!("Test passed: Swap cycle completed successfully.");
}
