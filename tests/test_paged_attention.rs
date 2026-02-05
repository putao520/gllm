use gllm::engine::scheduler::{PagedScheduler, RequestKind, SchedulerConfig};

#[test]
fn paged_attention_allocates_pages() {
    let config = SchedulerConfig {
        page_size: 4,
        total_pages: 6,
        max_batch: 4,
        max_tokens: 64,
        ..SchedulerConfig::default()
    };
    let mut scheduler = PagedScheduler::with_config(config);
    scheduler.enqueue_with_tokens(RequestKind::Generate, "a", 5);
    scheduler.enqueue_with_tokens(RequestKind::Generate, "b", 8);

    let batch = scheduler.next_batch().expect("batch");
    assert_eq!(batch.requests.len(), 2);
    assert_eq!(batch.allocations.len(), 2);
    assert_eq!(batch.allocations[0].pages.len(), 2);
    assert_eq!(batch.allocations[1].pages.len(), 2);

    scheduler.complete_batch(batch);
    assert_eq!(scheduler.free_pages(), 6);
}

#[test]
fn paged_attention_dynamic_batching_respects_limits() {
    let config = SchedulerConfig {
        page_size: 2,
        total_pages: 10,
        max_batch: 2,
        max_tokens: 6,
        ..SchedulerConfig::default()
    };
    let mut scheduler = PagedScheduler::with_config(config);
    scheduler.enqueue_with_tokens(RequestKind::Generate, "a", 3);
    scheduler.enqueue_with_tokens(RequestKind::Generate, "b", 3);
    scheduler.enqueue_with_tokens(RequestKind::Generate, "c", 3);

    let batch = scheduler.next_batch().expect("batch");
    assert_eq!(batch.requests.len(), 2);
    assert_eq!(batch.total_tokens, 6);
    scheduler.complete_batch(batch);

    let next = scheduler.next_batch().expect("next batch");
    assert_eq!(next.requests.len(), 1);
    assert_eq!(next.total_tokens, 3);
    scheduler.complete_batch(next);
}

#[test]
fn paged_attention_prefetches_with_double_buffer() {
    let config = SchedulerConfig {
        page_size: 2,
        total_pages: 4,
        max_batch: 1,
        max_tokens: 8,
        ..SchedulerConfig::default()
    };
    let mut scheduler = PagedScheduler::with_config(config);
    scheduler.enqueue_with_tokens(RequestKind::Generate, "a", 2);
    scheduler.enqueue_with_tokens(RequestKind::Generate, "b", 2);

    let prefetched = scheduler.prefetch_next().expect("prefetch");
    let prefetched_id = prefetched.id;
    let prefetched_slot = prefetched.kv_cache_slot;

    let first = scheduler.next_batch().expect("first batch");
    assert_eq!(first.id, prefetched_id);
    assert_eq!(first.kv_cache_slot, prefetched_slot);

    let second = scheduler.next_batch().expect("second batch");
    assert_ne!(first.kv_cache_slot, second.kv_cache_slot);

    scheduler.complete_batch(first);
    scheduler.complete_batch(second);
}

#[test]
fn paged_attention_rejects_oversized_request() {
    let config = SchedulerConfig {
        page_size: 4,
        total_pages: 1,
        max_batch: 1,
        max_tokens: 32,
        ..SchedulerConfig::default()
    };
    let mut scheduler = PagedScheduler::with_config(config);
    scheduler.enqueue_with_tokens(RequestKind::Generate, "a", 9);

    assert!(scheduler.next_batch().is_none());
    assert_eq!(scheduler.free_pages(), 1);
    assert!(scheduler.prefetch_next().is_none());
}

fn static_batch_cost(lengths: &[usize], max_batch: usize) -> usize {
    lengths
        .chunks(max_batch)
        .map(|chunk| chunk.len() * chunk.iter().copied().max().unwrap_or(0))
        .sum()
}

#[test]
fn continuous_batching_improves_utilization_over_static() {
    let lengths = vec![8, 2, 6, 1, 7, 3];
    let max_batch = 3;
    let max_tokens = 16;

    let static_cost = static_batch_cost(&lengths, max_batch);

    let config = SchedulerConfig {
        page_size: 2,
        total_pages: 64,
        max_batch,
        max_tokens,
        ..SchedulerConfig::default()
    };
    let mut scheduler = PagedScheduler::with_config(config);
    for (idx, tokens) in lengths.iter().enumerate() {
        scheduler.enqueue_with_tokens(RequestKind::Generate, format!("req{idx}"), *tokens);
    }

    let mut dynamic_tokens = 0usize;
    while let Some(batch) = scheduler.next_batch() {
        dynamic_tokens += batch.total_tokens;
        scheduler.complete_batch(batch);
    }

    assert!(dynamic_tokens > 0);
    assert!(dynamic_tokens <= static_cost);
}
