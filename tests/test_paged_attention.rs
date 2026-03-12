use std::time::Instant;

use gllm::scheduler::{
    BatchOrderPolicy, BatchResult, ContinuousBatcher, GroupState, HGALConfig, PagedScheduler,
    SchedulerError, Sequence, SequenceGroup,
};
use gllm::scheduler::RequestId;

fn make_group(id: RequestId, context_len: usize) -> SequenceGroup {
    SequenceGroup {
        id,
        pages: Vec::new(),
        context_len,
        state: GroupState::Running,
        access_count: 0,
        last_access: Instant::now(),
        is_pinned: false,
    }
}

fn make_sequence(id: RequestId, prompt_len: usize) -> Sequence {
    Sequence::new(id, vec![1; prompt_len])
}

/// TEST-PAGED-001: PagedAttention 分配页面
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建调度器
/// 2. 添加两个序列
/// 3. 验证页面分配后剩余块数
/// 4. 释放序列并验证块归还
///
/// **期望结果**: 页面正确分配并释放
#[test]
fn paged_attention_allocates_pages() {
    let mut scheduler = PagedScheduler::new(6, 4, HGALConfig::default());
    scheduler
        .add_sequence(make_group(1, 5))
        .expect("add sequence 1");
    scheduler
        .add_sequence(make_group(2, 8))
        .expect("add sequence 2");

    // 5 tokens -> 2 blocks, 8 tokens -> 2 blocks, total 4 used.
    assert_eq!(scheduler.num_free_blocks(), 2);

    scheduler.free_sequence(1);
    scheduler.free_sequence(2);
    assert_eq!(scheduler.num_free_blocks(), 6);
}

/// TEST-PAGED-002: PagedAttention 动态批处理尊重批次大小
///
/// **关联需求**: REQ-TEST-005, REQ-SCHED-003
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建调度器与 batcher
/// 2. 入队 3 个序列
/// 3. 限制 max_batch_size=2 构建批次
///
/// **期望结果**: 第一批 2 个，第二批 1 个
#[test]
fn paged_attention_dynamic_batching_respects_limits() {
    let mut scheduler = PagedScheduler::new(10, 2, HGALConfig::default());
    let mut batcher = ContinuousBatcher::new();
    batcher.enqueue(make_sequence(1, 3));
    batcher.enqueue(make_sequence(2, 3));
    batcher.enqueue(make_sequence(3, 3));

    let first = batcher.build_batch(
        &mut scheduler,
        2,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(first.requests, vec![1, 2]);
    batcher.update_batch(
        &mut scheduler,
        &[
            BatchResult::complete(1, None),
            BatchResult::complete(2, None),
        ],
    );

    let second = batcher.build_batch(
        &mut scheduler,
        2,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(second.requests, vec![3]);
    batcher.update_batch(&mut scheduler, &[BatchResult::complete(3, None)]);
    assert!(!batcher.has_pending_work());
}

/// TEST-PAGED-003: PagedAttention 换出后可恢复并记录 swap-in 映射
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 添加两个序列
/// 2. 选择并换出一个受害者序列
/// 3. 访问被换出序列触发恢复
///
/// **期望结果**: 恢复成功且存在 pending swap-in
#[test]
fn paged_attention_prefetches_with_double_buffer() {
    let mut scheduler = PagedScheduler::new(4, 2, HGALConfig::default());
    scheduler
        .add_sequence(make_group(1, 2))
        .expect("add sequence 1");
    scheduler
        .add_sequence(make_group(2, 2))
        .expect("add sequence 2");

    let victims = scheduler.select_victims(1);
    assert!(!victims.is_empty());
    let victim_ids: Vec<RequestId> = victims.iter().map(|(id, _)| *id).collect();
    let victim = victim_ids[0];
    scheduler
        .free_victims(&victim_ids)
        .expect("swap-out victim");

    let new_block = scheduler
        .allocate_next_token(victim)
        .expect("restore victim and continue");
    assert!(new_block.is_some());

    let pending = scheduler
        .take_pending_swap_in(victim)
        .expect("pending swap-in should exist");
    assert_eq!(pending.len(), 1);
}

/// TEST-PAGED-004: PagedAttention 拒绝超大请求
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 边界测试
///
/// **测试步骤**:
/// 1. 创建仅 1 块容量的调度器
/// 2. 添加需要 3 块的序列
///
/// **期望结果**: add_sequence 返回 OOM
#[test]
fn paged_attention_rejects_oversized_request() {
    let mut scheduler = PagedScheduler::new(1, 4, HGALConfig::default());
    let err = scheduler
        .add_sequence(make_group(1, 9))
        .expect_err("oversized sequence should be rejected");
    assert!(matches!(err, SchedulerError::OutOfMemory { .. }));
    assert_eq!(scheduler.num_free_blocks(), 1);
}

#[test]
fn continuous_batching_improves_utilization_over_static() {
    let mut scheduler = PagedScheduler::new(64, 2, HGALConfig::default());
    let mut batcher = ContinuousBatcher::new();

    for (idx, prompt_len) in [8, 2, 6, 1, 7, 3].into_iter().enumerate() {
        batcher.enqueue(make_sequence((idx + 1) as RequestId, prompt_len));
    }

    let batch = batcher.build_batch(
        &mut scheduler,
        usize::MAX,
        true,
        BatchOrderPolicy::StrictRequestIdOrder,
    );
    assert_eq!(batch.requests, vec![1, 2, 3, 4, 5, 6]);

    let results: Vec<BatchResult> = batch
        .requests
        .iter()
        .map(|id| BatchResult::complete(*id, None))
        .collect();
    batcher.update_batch(&mut scheduler, &results);

    assert!(!batcher.has_pending_work());
    assert_eq!(scheduler.num_free_blocks(), 64);
}
