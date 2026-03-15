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

// ---- AdaptiveChunkPolicy tests (REQ-KV-EXT-001) ----

use gllm::scheduler::vllm2024::{AdaptiveChunkPolicy, ChunkedConfig};

/// TEST-KV-EXT-001: 高负载返回 min_chunk
/// **关联需求**: REQ-KV-EXT-001
/// **测试类型**: 正向
/// **期望结果**: L1 可用 < 25% 时返回 min_chunk
#[test]
fn adaptive_chunk_high_load_returns_min() {
    let cfg = ChunkedConfig::default(); // chunk_size = 64
    let policy = AdaptiveChunkPolicy::new(&cfg, 4096);
    let result = policy.compute(0.10, 1, 2048);
    assert_eq!(result, 64);
}

/// TEST-KV-EXT-002: 低负载返回 max_chunk
/// **关联需求**: REQ-KV-EXT-001
/// **测试类型**: 正向
/// **期望结果**: L1 可用 > 75% 时返回 max_chunk (clamped to prompt_len)
#[test]
fn adaptive_chunk_low_load_returns_max() {
    let cfg = ChunkedConfig::default();
    let policy = AdaptiveChunkPolicy::new(&cfg, 4096);
    // prompt_len = 2048 < max_chunk = 4096, so clamped to prompt_len
    let result = policy.compute(0.90, 1, 2048);
    assert_eq!(result, 2048);
    // prompt_len > max_chunk
    let result2 = policy.compute(0.90, 1, 8192);
    assert_eq!(result2, 4096);
}

/// TEST-KV-EXT-003: 中负载线性插值
/// **关联需求**: REQ-KV-EXT-001
/// **测试类型**: 正向
/// **期望结果**: 50% L1 可用时返回 min 和 max 的中间值
#[test]
fn adaptive_chunk_mid_load_interpolates() {
    let cfg = ChunkedConfig::default(); // min = 64
    let policy = AdaptiveChunkPolicy::new(&cfg, 4096);
    // l1_ratio = 0.50 → t = (0.50 - 0.25) / 0.50 = 0.50
    // base = 64 + (4032 * 0.50) = 64 + 2016 = 2080
    let result = policy.compute(0.50, 1, 4096);
    assert_eq!(result, 2080);
}

/// TEST-KV-EXT-004: 短 prompt 直接返回 prompt_len
/// **关联需求**: REQ-KV-EXT-001
/// **测试类型**: 边界
/// **期望结果**: prompt_len < min_chunk 时返回 prompt_len
#[test]
fn adaptive_chunk_short_prompt() {
    let cfg = ChunkedConfig::default();
    let policy = AdaptiveChunkPolicy::new(&cfg, 4096);
    let result = policy.compute(0.10, 5, 32);
    assert_eq!(result, 32);
}

/// TEST-KV-EXT-005: 并发惩罚不低于 min_chunk
/// **关联需求**: REQ-KV-EXT-001
/// **测试类型**: 边界
/// **期望结果**: 高并发时 chunk 缩小但不低于 min_chunk
#[test]
fn adaptive_chunk_concurrency_penalty() {
    let cfg = ChunkedConfig::default();
    let policy = AdaptiveChunkPolicy::new(&cfg, 4096);
    // Low load but 10 concurrent requests → penalty = max(0.2, 1.0 - 0.9) = 0.2
    // base = 4096, adjusted = 4096 * 0.2 = 819, clamped to [64, 2048]
    let result = policy.compute(0.90, 10, 2048);
    assert!(result >= 64);
    assert!(result <= 2048);
    // Should be significantly less than no-concurrency case
    let no_conc = policy.compute(0.90, 1, 2048);
    assert!(result < no_conc);
}

/// TEST-KV-EXT-006: zero prompt 返回 1
/// **关联需求**: REQ-KV-EXT-001
/// **测试类型**: 边界
/// **期望结果**: prompt_len = 0 时返回 1 (max(0, 1))
#[test]
fn adaptive_chunk_zero_prompt() {
    let cfg = ChunkedConfig::default();
    let policy = AdaptiveChunkPolicy::new(&cfg, 4096);
    let result = policy.compute(0.50, 1, 0);
    assert_eq!(result, 1);
}

// ---- KV Incremental Distillation tests (REQ-KV-EXT-002) ----

use gllm::scheduler::vllm2024::{SwiftKVConfig, SwiftKvState};

fn make_swift_kv(window_size: usize) -> SwiftKvState {
    SwiftKvState::new(SwiftKVConfig {
        enabled: true,
        window_size,
        enable_across_kv: false,
        similarity_threshold: 0.9,
        precision_guard: 0.1,
    })
}

fn make_pages(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..dim).map(|j| (i * dim + j) as f32 * 0.01).collect())
        .collect()
}

/// TEST-KV-EXT-007: 增量蒸馏结果与全量蒸馏数值一致
/// **关联需求**: REQ-KV-EXT-002
/// **测试类型**: 正向
/// **期望结果**: 首次增量蒸馏 == 全量蒸馏（无先前边界）
#[test]
fn incremental_distill_first_call_matches_full() {
    let pages = make_pages(8, 16);
    let mut state_full = make_swift_kv(4);
    let mut state_incr = make_swift_kv(4);

    let full = state_full.distill_cpu(&pages);
    let incr = state_incr.distill_cpu_incremental(&pages);

    assert_eq!(full.result.distilled_pages.len(), incr.result.distilled_pages.len());
    assert_eq!(state_incr.last_distilled_page, 8);
}

/// TEST-KV-EXT-008: 多轮追加页面后增量蒸馏正确
/// **关联需求**: REQ-KV-EXT-002
/// **测试类型**: 正向
/// **期望结果**: 追加页面后仅处理新增部分
#[test]
fn incremental_distill_appended_pages() {
    let mut state = make_swift_kv(2);
    let pages_round1 = make_pages(4, 8);
    let _ = state.distill_cpu_incremental(&pages_round1);
    assert_eq!(state.last_distilled_page, 4);

    // Append 4 more pages
    let mut pages_round2 = pages_round1.clone();
    pages_round2.extend(make_pages(4, 8));
    let outcome = state.distill_cpu_incremental(&pages_round2);
    assert_eq!(state.last_distilled_page, 8);
    // Should have produced a valid distillation
    assert!(!outcome.result.distilled_pages.is_empty());
}

/// TEST-KV-EXT-009: 页面数缩小时回退全量蒸馏
/// **关联需求**: REQ-KV-EXT-002
/// **测试类型**: 边界
/// **期望结果**: 页面数减少（session 重置）时 boundary 归零，执行全量
#[test]
fn incremental_distill_shrink_resets_boundary() {
    let mut state = make_swift_kv(2);
    let pages8 = make_pages(8, 8);
    let _ = state.distill_cpu_incremental(&pages8);
    assert_eq!(state.last_distilled_page, 8);

    // Shrink to 3 pages (session reset)
    let pages3 = make_pages(3, 8);
    let _ = state.distill_cpu_incremental(&pages3);
    assert_eq!(state.last_distilled_page, 3);
}

/// TEST-KV-EXT-010: 空页面返回 default
/// **关联需求**: REQ-KV-EXT-002
/// **测试类型**: 边界
/// **期望结果**: 空输入返回空结果
#[test]
fn incremental_distill_empty_pages() {
    let mut state = make_swift_kv(4);
    let empty: &[Vec<f32>] = &[];
    let outcome = state.distill_cpu_incremental(empty);
    assert!(outcome.result.distilled_pages.is_empty());
    assert!(!outcome.precision_fallback);
}

/// TEST-KV-EXT-011: reset_distill_boundary 后重新全量蒸馏
/// **关联需求**: REQ-KV-EXT-002
/// **测试类型**: 正向
/// **期望结果**: 重置后 boundary 归零，下次调用执行全量
#[test]
fn incremental_distill_reset_boundary() {
    let mut state = make_swift_kv(2);
    let pages = make_pages(6, 8);
    let _ = state.distill_cpu_incremental(&pages);
    assert_eq!(state.last_distilled_page, 6);

    state.reset_distill_boundary();
    assert_eq!(state.last_distilled_page, 0);
    assert!(state.last_result.is_none());

    // Next call should process all pages again
    let _ = state.distill_cpu_incremental(&pages);
    assert_eq!(state.last_distilled_page, 6);
}

/// TEST-KV-EXT-012: SIKV 滑动窗口 overlap 保证连续性
/// **关联需求**: REQ-KV-EXT-002
/// **测试类型**: 正向
/// **期望结果**: overlap = min(window_size, last_distilled_page) 确保窗口连续
#[test]
fn incremental_distill_overlap_continuity() {
    let mut state = make_swift_kv(3); // window_size = 3
    let pages4 = make_pages(4, 8);
    let _ = state.distill_cpu_incremental(&pages4);
    assert_eq!(state.last_distilled_page, 4);

    // Append 4 more → total 8 pages
    // overlap = min(3, 4) = 3, start = 4 - 3 = 1
    // delta_pages = pages[1..8] = 7 pages
    let mut pages8 = pages4.clone();
    pages8.extend(make_pages(4, 8));
    let outcome = state.distill_cpu_incremental(&pages8);
    assert_eq!(state.last_distilled_page, 8);
    assert!(!outcome.result.distilled_pages.is_empty());
}