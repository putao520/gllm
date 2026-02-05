use gllm::engine::scheduler::{PagedScheduler, RequestKind, SchedulerConfig};

/// TEST-PAGED-001: PagedAttention 分配页面
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建调度器
/// 2. 入队两个请求
/// 3. 构建批次
/// 4. 验证页面分配
///
/// **期望结果**: 页面正确分配
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

/// TEST-PAGED-002: PagedAttention 动态批处理尊重限制
///
/// **关联需求**: REQ-TEST-005, REQ-SCHED-003
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建调度器 (max_batch=2, max_tokens=6)
/// 2. 入队 3 个请求 (各 3 tokens)
/// 3. 验证批次大小和 token 数
///
/// **期望结果**: 第一批次 2 个请求，第二批次 1 个请求
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

/// TEST-PAGED-003: PagedAttention 双缓冲预取
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建调度器
/// 2. 调用 prefetch_next()
/// 3. 调用 next_batch() 两次
/// 4. 验证 KV cache slot 不同
///
/// **期望结果**: 预取和实际批次使用不同 slot
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

/// TEST-PAGED-004: PagedAttention 拒绝超大请求
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 边界测试
///
/// **测试步骤**:
/// 1. 创建调度器 (1 页面)
/// 2. 入队超大请求 (9 tokens)
/// 3. 尝试构建批次
///
/// **期望结果**: next_batch() 返回 None
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
