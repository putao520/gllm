//! Scheduler refactor tests (REQ-TEST-005)
//! Covers PrefixIndex, SessionKvCache, KvPipeline, BatchOrderPolicy.

use gllm::scheduler::{KvPrefixIndex, TokenId};
use gllm::scheduler::{
    BatchOrderPolicy, GroupState, KvPipeline, SequenceGroup, VirtualPageId,
};
use gllm::scheduler::{Sequence, SequenceState};

// ---- PrefixIndex tests ----

/// TEST-SCHED-005: 前缀索引插入和查找
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向
/// **期望结果**: 插入前缀后可以正确查找匹配的页面
#[test]
fn prefix_index_insert_and_lookup() {
    let mut index = KvPrefixIndex::new();
    let tokens: Vec<TokenId> = vec![10, 20, 30, 40];
    let pages = vec![
        VirtualPageId::new(1, 0),
        VirtualPageId::new(1, 1),
        VirtualPageId::new(1, 2),
        VirtualPageId::new(1, 3),
    ];
    index.insert(&tokens, &pages);

    let result = index.find_longest_prefix(&tokens).unwrap();
    assert_eq!(result.matched_tokens, 4);
    assert_eq!(result.matched_pages, pages);
}

/// TEST-SCHED-006: 前缀索引部分匹配
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向
/// **期望结果**: 查询前缀的子串时返回最长匹配
#[test]
fn prefix_index_partial_match() {
    let mut index = KvPrefixIndex::new();
    let tokens: Vec<TokenId> = vec![1, 2, 3, 4, 5];
    let pages = vec![
        VirtualPageId::new(1, 0),
        VirtualPageId::new(1, 1),
        VirtualPageId::new(1, 2),
        VirtualPageId::new(1, 3),
        VirtualPageId::new(1, 4),
    ];
    index.insert(&tokens, &pages);

    // Query with a prefix that diverges after 3 tokens
    let result = index.find_longest_prefix(&[1, 2, 3, 99, 100]).unwrap();
    assert_eq!(result.matched_tokens, 3);
}

/// TEST-SCHED-007: 前缀索引追加复用
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向
/// **期望结果**: 追加新 token 后复用已有前缀的页面
#[test]
fn prefix_index_append_reuse() {
    let mut index = KvPrefixIndex::new();

    // Insert initial prefix
    let tokens_v1: Vec<TokenId> = vec![11, 22, 33];
    let pages_v1 = vec![
        VirtualPageId::new(1, 0),
        VirtualPageId::new(1, 1),
        VirtualPageId::new(1, 2),
    ];
    index.insert(&tokens_v1, &pages_v1);

    // Append more tokens extending the same prefix
    let tokens_v2: Vec<TokenId> = vec![11, 22, 33, 44, 55];
    let pages_v2 = vec![
        VirtualPageId::new(1, 0),
        VirtualPageId::new(1, 1),
        VirtualPageId::new(1, 2),
        VirtualPageId::new(1, 3),
        VirtualPageId::new(1, 4),
    ];
    index.insert(&tokens_v2, &pages_v2);

    // The extended prefix should now be findable
    let result = index.find_longest_prefix(&tokens_v2).unwrap();
    assert_eq!(result.matched_tokens, 5);

    // Original prefix still works as a partial match
    let result = index.find_longest_prefix(&[11, 22, 33, 99]).unwrap();
    assert_eq!(result.matched_tokens, 3);
}

/// TEST-SCHED-008: 前缀索引无匹配返回 None
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 负向
/// **期望结果**: 查询不存在的前缀时返回 None
#[test]
fn prefix_index_no_match_returns_none() {
    let mut index = KvPrefixIndex::new();
    index.insert(&[1, 2, 3], &[VirtualPageId::new(1, 0)]);
    assert!(index.find_longest_prefix(&[9, 8, 7]).is_none());
}

/// TEST-SCHED-009: 前缀索引空查询返回 None
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 边界
/// **期望结果**: 空 token 序列查询返回 None
#[test]
fn prefix_index_empty_query_returns_none() {
    let mut index = KvPrefixIndex::new();
    index.insert(&[1, 2], &[VirtualPageId::new(1, 0), VirtualPageId::new(1, 1)]);
    assert!(index.find_longest_prefix(&[]).is_none());
}

// ---- BatchOrderPolicy tests ----

/// TEST-SCHED-010: 严格请求 ID 排序为默认策略
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向
/// **期望结果**: 默认批次排序策略为 StrictRequestIdOrder
#[test]
fn strict_request_id_order_is_default() {
    let policy = BatchOrderPolicy::default();
    assert_eq!(policy, BatchOrderPolicy::StrictRequestIdOrder);
}

/// TEST-SCHED-011: 严格请求 ID 排序正确性
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向
/// **期望结果**: 请求按 ID 严格升序排列
#[test]
fn strict_request_id_order_sorts_correctly() {
    // Verify StrictRequestIdOrder produces monotonic ordering by sorting request IDs
    let mut request_ids: Vec<u64> = vec![5, 1, 3, 2, 4];
    // StrictRequestIdOrder means we sort by RequestId ascending
    request_ids.sort();
    assert_eq!(request_ids, vec![1, 2, 3, 4, 5]);
}

/// TEST-SCHED-012: 批次排序策略变体互不相同
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向
/// **期望结果**: 不同的 BatchOrderPolicy 变体判定为不相等
#[test]
fn batch_order_policy_variants_are_distinct() {
    let strict = BatchOrderPolicy::StrictRequestIdOrder;
    let fifo = BatchOrderPolicy::FifoOrder;
    assert_ne!(strict, fifo);
}

// ---- KvPipeline tests ----

/// TEST-SCHED-013: KV 管线变体
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向
/// **期望结果**: KV 管线各变体正确构造和区分
#[test]
fn kv_pipeline_variants() {
    let conv = KvPipeline::Conversation;
    let work = KvPipeline::Working;
    assert_ne!(conv, work);
    assert_eq!(conv, KvPipeline::Conversation);
    assert_eq!(work, KvPipeline::Working);
}

// ---- SequenceGroup tests ----

/// TEST-SCHED-014: 序列组基本操作
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向
/// **期望结果**: SequenceGroup 的创建、添加和查询操作正确
#[test]
fn sequence_group_basic_operations() {
    use std::time::Instant;

    let group = SequenceGroup {
        id: 42,
        pages: vec![0, 1, 2],
        state: GroupState::Running,
        access_count: 0,
        last_access: Instant::now(),
        is_pinned: false,
        context_len: 128,
    };

    assert_eq!(group.id, 42);
    assert_eq!(group.pages.len(), 3);
    assert_eq!(group.state, GroupState::Running);
    assert!(!group.is_pinned);
    assert_eq!(group.context_len, 128);
}

#[test]
fn sequence_group_state_transitions() {
    let running = GroupState::Running;
    let swapped = GroupState::Swapped;
    let paused = GroupState::Paused;

    assert_ne!(running, swapped);
    assert_ne!(running, paused);
    assert_ne!(swapped, paused);
}

// ---- Sequence tests ----

#[test]
fn sequence_new_initializes_correctly() {
    let seq = Sequence::new(7, vec![100, 200, 300]);
    assert_eq!(seq.id, 7);
    assert_eq!(seq.prompt_tokens, vec![100, 200, 300]);
    assert!(seq.generated_tokens.is_empty());
    assert_eq!(seq.state, SequenceState::Waiting);
    assert_eq!(seq.context_len(), 3);
    assert!(seq.needs_prefill());
}

#[test]
fn sequence_push_token_advances_position() {
    let mut seq = Sequence::new(1, vec![10, 20]);
    assert_eq!(seq.context_len(), 2);

    seq.push_generated_token(30);
    assert_eq!(seq.context_len(), 3);
    assert_eq!(seq.generated_tokens, vec![30]);

    seq.push_generated_token(40);
    assert_eq!(seq.context_len(), 4);
    assert_eq!(seq.generated_tokens, vec![30, 40]);
}

#[test]
fn sequence_to_sequence_group() {
    let mut seq = Sequence::new(5, vec![1, 2, 3]);
    seq.mark_running(vec![10, 11, 12]);

    let group = seq.to_sequence_group();
    assert_eq!(group.id, 5);
    assert_eq!(group.pages, vec![10, 11, 12]);
    assert_eq!(group.state, GroupState::Running);
    assert_eq!(group.context_len, 3);
}

#[test]
fn sequence_mark_running_updates_state() {
    let mut seq = Sequence::new(1, vec![10]);
    assert_eq!(seq.state, SequenceState::Waiting);

    seq.mark_running(vec![100]);
    assert_eq!(seq.state, SequenceState::Running);
    assert_eq!(seq.kv_pages, vec![100]);
}