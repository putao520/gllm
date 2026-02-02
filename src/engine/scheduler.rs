//! Scheduler (PagedAttention / Continuous Batching / Double Buffering).

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::kv_cache::KvCacheSlot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    Generate,
    Embeddings,
    Rerank,
}

#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    pub id: RequestId,
    pub kind: RequestKind,
    pub prompt: String,
    pub tokens: usize,
}

impl ScheduledRequest {
    fn token_len(&self) -> usize {
        self.tokens.max(1)
    }
}

#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub id: BatchId,
    pub requests: Vec<ScheduledRequest>,
    pub allocations: Vec<PageAllocation>,
    pub total_tokens: usize,
    pub kv_cache_slot: KvCacheSlot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub page_size: usize,
    pub total_pages: usize,
    pub max_batch: usize,
    pub max_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            page_size: 128,
            total_pages: 2048,
            max_batch: 8,
            max_tokens: 4096,
        }
    }
}

/// 页面状态 (与 gllm-kernels 的 PageState 对应)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    /// 在 GPU 内存中且正在使用
    Active,
    /// 在 GPU 内存中但未使用 (可换出)
    Standby,
    /// 已换出到 CPU 内存
    Swapped,
}

/// 页面条目，包含状态和访问时间
#[derive(Debug, Clone)]
pub struct PageEntry {
    pub page_id: PageId,
    pub state: PageState,
    pub last_access: Instant,
    pub owner_request: Option<RequestId>,
}

/// LRU 链表节点
#[derive(Debug, Clone)]
struct LruNode {
    page_id: PageId,
    #[allow(dead_code)]
    last_access: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub usize);

#[derive(Debug, Clone)]
pub struct PageAllocation {
    pub pages: Vec<PageId>,
    pub tokens: usize,
}

#[derive(Debug)]
pub struct PagePool {
    page_size: usize,
    free: VecDeque<PageId>,
    total_pages: usize,
    /// 页面状态跟踪
    page_states: HashMap<PageId, PageEntry>,
    /// LRU 链表 (按最后访问时间排序)
    lru_list: VecDeque<LruNode>,
}

impl PagePool {
    pub fn new(page_size: usize, total_pages: usize) -> Self {
        let mut free = VecDeque::with_capacity(total_pages);
        let mut page_states = HashMap::new();
        for idx in 0..total_pages {
            let page_id = PageId(idx);
            free.push_back(page_id);
            // 初始化所有页面为 Standby 状态
            page_states.insert(page_id, PageEntry {
                page_id,
                state: PageState::Standby,
                last_access: Instant::now(),
                owner_request: None,
            });
        }
        Self {
            page_size: page_size.max(1),
            free,
            total_pages,
            page_states,
            lru_list: VecDeque::new(),
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    /// 标记页面为已访问（更新 LRU）
    pub fn mark_accessed(&mut self, page_ids: &[PageId], owner: RequestId) {
        let now = Instant::now();
        for &page_id in page_ids {
            // 从旧位置移除
            self.lru_list.retain(|node| node.page_id != page_id);

            // 添加到末尾（最近访问）
            self.lru_list.push_back(LruNode {
                page_id,
                last_access: now,
            });

            // 更新页面状态
            if let Some(entry) = self.page_states.get_mut(&page_id) {
                entry.state = PageState::Active;
                entry.last_access = now;
                entry.owner_request = Some(owner);
            }
        }
    }

    /// 获取页面状态
    pub fn get_page_state(&self, page_id: PageId) -> Option<PageState> {
        self.page_states.get(&page_id).map(|entry| entry.state)
    }

    /// 设置页面状态
    pub fn set_page_state(&mut self, page_id: PageId, state: PageState) {
        if let Some(entry) = self.page_states.get_mut(&page_id) {
            entry.state = state;
        }
    }

    /// 选择要换出的页面 (基于 LRU)
    /// 返回处于 Active 或 Standby 状态的最久未使用页面
    pub fn select_victim_pages(&self, count: usize) -> Vec<PageId> {
        self.lru_list.iter()
            .filter(|node| {
                if let Some(entry) = self.page_states.get(&node.page_id) {
                    matches!(entry.state, PageState::Active | PageState::Standby)
                } else {
                    false
                }
            })
            .take(count)
            .map(|node| node.page_id)
            .collect()
    }

    /// 获取所有页面状态（用于调试）
    pub fn page_states_snapshot(&self) -> Vec<(PageId, PageState, Option<RequestId>)> {
        self.page_states.values()
            .map(|entry| (entry.page_id, entry.state, entry.owner_request))
            .collect()
    }

    pub fn allocate(&mut self, tokens: usize) -> Option<PageAllocation> {
        if tokens == 0 {
            return Some(PageAllocation {
                pages: Vec::new(),
                tokens: 0,
            });
        }
        let needed = tokens.div_ceil(self.page_size);
        if self.free.len() < needed {
            return None;
        }
        let mut pages = Vec::with_capacity(needed);
        for _ in 0..needed {
            if let Some(page) = self.free.pop_front() {
                pages.push(page);
            }
        }
        // 标记页面为 Active
        for &page_id in &pages {
            if let Some(entry) = self.page_states.get_mut(&page_id) {
                entry.state = PageState::Active;
                entry.last_access = Instant::now();
            }
        }
        Some(PageAllocation { pages, tokens })
    }

    pub fn release(&mut self, allocation: PageAllocation) {
        for page in allocation.pages {
            // 标记页面为 Standby
            if let Some(entry) = self.page_states.get_mut(&page) {
                entry.state = PageState::Standby;
                entry.owner_request = None;
            }
            self.free.push_back(page);
        }
    }
}

#[derive(Debug)]
pub struct DoubleBuffer<T> {
    front: T,
    back: T,
}

impl<T> DoubleBuffer<T> {
    pub fn new(front: T, back: T) -> Self {
        Self { front, back }
    }

    pub fn front(&self) -> &T {
        &self.front
    }

    pub fn back(&self) -> &T {
        &self.back
    }

    pub fn front_mut(&mut self) -> &mut T {
        &mut self.front
    }

    pub fn back_mut(&mut self) -> &mut T {
        &mut self.back
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.front, &mut self.back);
    }
}

#[derive(Debug)]
pub struct DynamicBatcher {
    max_batch: usize,
    max_tokens: usize,
}

impl DynamicBatcher {
    pub fn new(max_batch: usize, max_tokens: usize) -> Self {
        Self {
            max_batch: max_batch.max(1),
            max_tokens: max_tokens.max(1),
        }
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn drain_batch<T, F>(&self, queue: &mut VecDeque<T>, token_len: F) -> Vec<T>
    where
        F: Fn(&T) -> usize,
    {
        let mut out = Vec::new();
        let mut tokens = 0usize;
        while let Some(item) = queue.front() {
            let item_tokens = token_len(item);
            if out.len() >= self.max_batch || tokens + item_tokens > self.max_tokens {
                break;
            }
            let item = queue.pop_front().expect("queue should contain item");
            tokens += item_tokens;
            out.push(item);
        }
        out
    }
}

#[derive(Debug, Default)]
struct BatchSlot {
    batch: Option<ScheduledBatch>,
}

/// 序列状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    /// 等待调度
    Waiting,
    /// 正在生成
    Running,
    /// 已暂停 (等待 swap-in)
    Paused,
    /// 已完成
    Completed,
    /// 失败
    Failed,
}

/// 序列跟踪信息
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    pub id: RequestId,
    pub state: SequenceState,
    pub generated_tokens: usize,
    pub position: usize,
    pub allocated_pages: Vec<PageId>,
}

#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    next_id: u64,
    next_batch_id: u64,
    queue: VecDeque<ScheduledRequest>,
    page_pool: PagePool,
    batcher: DynamicBatcher,
    buffer: DoubleBuffer<BatchSlot>,
    next_slot: KvCacheSlot,
    /// 序列跟踪（用于 continuous batching）
    sequences: HashMap<RequestId, SequenceInfo>,
    /// 正在运行的序列 ID 列表
    running_sequences: Vec<RequestId>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    pub fn with_config(config: SchedulerConfig) -> Self {
        let page_pool = PagePool::new(config.page_size, config.total_pages);
        let batcher = DynamicBatcher::new(config.max_batch, config.max_tokens);
        Self {
            config,
            next_id: 0,
            next_batch_id: 0,
            queue: VecDeque::new(),
            page_pool,
            batcher,
            buffer: DoubleBuffer::new(BatchSlot::default(), BatchSlot::default()),
            next_slot: KvCacheSlot::Front,
            sequences: HashMap::new(),
            running_sequences: Vec::new(),
        }
    }

    pub fn config(&self) -> SchedulerConfig {
        self.config
    }

    pub fn enqueue(&mut self, kind: RequestKind, prompt: impl Into<String>) -> RequestId {
        self.enqueue_with_tokens(kind, prompt, 0)
    }

    pub fn enqueue_with_tokens(
        &mut self,
        kind: RequestKind,
        prompt: impl Into<String>,
        tokens: usize,
    ) -> RequestId {
        let id = RequestId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.queue.push_back(ScheduledRequest {
            id,
            kind,
            prompt: prompt.into(),
            tokens,
        });
        id
    }

    pub fn next_batch(&mut self) -> Option<ScheduledBatch> {
        if self.buffer.front().batch.is_none() {
            if self.buffer.back().batch.is_some() {
                self.buffer.swap();
            } else {
                self.buffer.front_mut().batch = self.build_batch();
            }
        }
        let batch = self.buffer.front_mut().batch.take();
        if self.buffer.back().batch.is_none() {
            self.buffer.back_mut().batch = self.build_batch();
        }
        batch
    }

    pub fn prefetch_next(&mut self) -> Option<&ScheduledBatch> {
        if self.buffer.back().batch.is_none() {
            self.buffer.back_mut().batch = self.build_batch();
        }
        self.buffer.back().batch.as_ref()
    }

    pub fn complete_batch(&mut self, batch: ScheduledBatch) {
        for allocation in batch.allocations {
            self.page_pool.release(allocation);
        }
        // 标记批次中的所有请求为完成（如果它们被跟踪）
        for request in &batch.requests {
            if self.sequences.contains_key(&request.id) {
                self.complete_sequence(request.id);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
            && self.buffer.front().batch.is_none()
            && self.buffer.back().batch.is_none()
    }

    pub fn free_pages(&self) -> usize {
        self.page_pool.free_pages()
    }

    // ========== 新增：PagedAttention 集成方法 ==========

    /// 标记请求的页面为已访问（更新 LRU）
    pub fn mark_pages_accessed(&mut self, request_id: RequestId, page_ids: Vec<PageId>) {
        self.page_pool.mark_accessed(&page_ids, request_id);
        // 更新序列信息
        if let Some(seq) = self.sequences.get_mut(&request_id) {
            seq.allocated_pages = page_ids;
        }
    }

    /// 获取页面池的状态快照
    pub fn page_states_snapshot(&self) -> Vec<(PageId, PageState, Option<RequestId>)> {
        self.page_pool.page_states_snapshot()
    }

    /// 选择要换出的页面
    pub fn select_victim_pages(&self, count: usize) -> Vec<PageId> {
        self.page_pool.select_victim_pages(count)
    }

    /// 设置页面状态
    pub fn set_page_state(&mut self, page_id: PageId, state: PageState) {
        self.page_pool.set_page_state(page_id, state);
    }

    // ========== 新增：Continuous Batching 集成方法 ==========

    /// 开始跟踪一个序列
    pub fn start_sequence(&mut self, request_id: RequestId, initial_pages: Vec<PageId>) {
        self.sequences.insert(request_id, SequenceInfo {
            id: request_id,
            state: SequenceState::Running,
            generated_tokens: 0,
            position: 0,
            allocated_pages: initial_pages,
        });
        self.running_sequences.push(request_id);
    }

    /// 更新序列状态（用于 continuous batching）
    pub fn update_sequence(&mut self, request_id: RequestId, generated_tokens: usize) {
        if let Some(seq) = self.sequences.get_mut(&request_id) {
            seq.generated_tokens = generated_tokens;
            seq.position += generated_tokens;
        }
    }

    /// 标记序列为完成
    pub fn complete_sequence(&mut self, request_id: RequestId) {
        if let Some(mut seq) = self.sequences.remove(&request_id) {
            seq.state = SequenceState::Completed;
            // 释放页面
            self.page_pool.release(PageAllocation {
                pages: seq.allocated_pages,
                tokens: 0,
            });
        }
        self.running_sequences.retain(|&id| id != request_id);
    }

    /// 标记序列为暂停
    pub fn pause_sequence(&mut self, request_id: RequestId) {
        if let Some(seq) = self.sequences.get_mut(&request_id) {
            seq.state = SequenceState::Paused;
        }
        self.running_sequences.retain(|&id| id != request_id);
    }

    /// 恢复暂停的序列
    pub fn resume_sequence(&mut self, request_id: RequestId) {
        if let Some(seq) = self.sequences.get_mut(&request_id) {
            seq.state = SequenceState::Running;
            self.running_sequences.push(request_id);
        }
    }

    /// 获取序列信息
    pub fn get_sequence(&self, request_id: RequestId) -> Option<&SequenceInfo> {
        self.sequences.get(&request_id)
    }

    /// 获取正在运行的序列列表
    pub fn running_sequences(&self) -> &[RequestId] {
        &self.running_sequences
    }

    /// 检查序列是否正在运行
    pub fn is_sequence_running(&self, request_id: RequestId) -> bool {
        self.running_sequences.contains(&request_id)
    }

    fn build_batch(&mut self) -> Option<ScheduledBatch> {
        let candidates = self
            .batcher
            .drain_batch(&mut self.queue, |req| req.token_len());
        if candidates.is_empty() {
            return None;
        }

        let mut requests = Vec::new();
        let mut allocations = Vec::new();
        let mut total_tokens = 0usize;
        let mut remainder = VecDeque::new();

        let mut iter = candidates.into_iter();
        while let Some(req) = iter.next() {
            let tokens = req.token_len();
            if let Some(allocation) = self.page_pool.allocate(tokens) {
                total_tokens = total_tokens.saturating_add(tokens);
                allocations.push(allocation);
                requests.push(req);
            } else {
                remainder.push_back(req);
                remainder.extend(iter);
                break;
            }
        }

        if !remainder.is_empty() {
            while let Some(req) = remainder.pop_back() {
                self.queue.push_front(req);
            }
        }

        if requests.is_empty() {
            return None;
        }

        let batch = ScheduledBatch {
            id: BatchId(self.next_batch_id),
            requests,
            allocations,
            total_tokens,
            kv_cache_slot: self.next_slot,
        };
        self.next_batch_id = self.next_batch_id.saturating_add(1);
        self.next_slot = self.next_slot.flip();
        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_pool_allocation_roundtrip() {
        let mut pool = PagePool::new(4, 4);
        assert_eq!(pool.free_pages(), 4);
        let allocation = pool.allocate(9).expect("allocate pages");
        assert_eq!(allocation.pages.len(), 3);
        assert_eq!(pool.free_pages(), 1);
        pool.release(allocation);
        assert_eq!(pool.free_pages(), 4);
    }

    #[test]
    fn double_buffer_swaps() {
        let mut buffer = DoubleBuffer::new(1u32, 2u32);
        assert_eq!(*buffer.front(), 1);
        assert_eq!(*buffer.back(), 2);
        buffer.swap();
        assert_eq!(*buffer.front(), 2);
        assert_eq!(*buffer.back(), 1);
    }

    #[test]
    fn dynamic_batcher_limits() {
        let mut queue: VecDeque<usize> = (0..5).collect();
        let batcher = DynamicBatcher::new(2, 3);
        let batch = batcher.drain_batch(&mut queue, |item| item + 1);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn scheduler_allocates_pages() {
        let config = SchedulerConfig {
            page_size: 4,
            total_pages: 4,
            max_batch: 4,
            max_tokens: 32,
        };
        let mut scheduler = Scheduler::with_config(config);
        scheduler.enqueue_with_tokens(RequestKind::Generate, "a", 5);
        scheduler.enqueue_with_tokens(RequestKind::Generate, "b", 3);
        let batch = scheduler.next_batch().expect("batch");
        assert_eq!(batch.allocations.len(), 2);
        scheduler.complete_batch(batch);
        assert_eq!(scheduler.free_pages(), 4);
    }

    #[test]
    fn scheduler_prefetches() {
        let config = SchedulerConfig {
            page_size: 2,
            total_pages: 4,
            max_batch: 1,
            max_tokens: 8,
        };
        let mut scheduler = Scheduler::with_config(config);
        scheduler.enqueue_with_tokens(RequestKind::Generate, "a", 2);
        scheduler.enqueue_with_tokens(RequestKind::Generate, "b", 2);
        scheduler.prefetch_next();
        let batch = scheduler.next_batch().expect("batch");
        assert_eq!(batch.requests.len(), 1);
    }

    // ========== 新增：PagedAttention 和 Continuous Batching 集成测试 ==========

    #[test]
    fn page_pool_tracks_states() {
        let mut pool = PagePool::new(4, 4);
        let allocation = pool.allocate(8).expect("allocate");
        assert_eq!(allocation.pages.len(), 2);

        // 分配的页面应该是 Active 状态
        for &page_id in &allocation.pages {
            let state = pool.get_page_state(page_id);
            assert_eq!(state, Some(PageState::Active));
        }
    }

    #[test]
    fn page_pool_lru_updates_on_access() {
        let mut pool = PagePool::new(4, 4);
        let alloc1 = pool.allocate(4).expect("alloc1");
        let alloc2 = pool.allocate(4).expect("alloc2");

        let req1 = RequestId(1);
        let req2 = RequestId(2);

        pool.mark_accessed(&alloc1.pages, req1);
        pool.mark_accessed(&alloc2.pages, req2);

        // alloc2 应该是最近访问的（在 LRU 链表末尾）
        let victims = pool.select_victim_pages(1);
        assert!(!victims.is_empty());
        // 第一个受害者应该是 alloc1 的第一个页面（更早访问）
        assert_eq!(victims[0], alloc1.pages[0]);
    }

    #[test]
    fn scheduler_tracks_sequences() {
        let mut scheduler = Scheduler::new();
        let req_id = scheduler.enqueue(RequestKind::Generate, "test");

        // 开始跟踪序列
        scheduler.start_sequence(req_id, vec![PageId(0), PageId(1)]);

        // 检查序列正在运行
        assert!(scheduler.is_sequence_running(req_id));
        assert_eq!(scheduler.running_sequences().len(), 1);

        // 完成序列
        scheduler.complete_sequence(req_id);

        // 序列应该不再运行
        assert!(!scheduler.is_sequence_running(req_id));
        assert_eq!(scheduler.running_sequences().len(), 0);
    }

    #[test]
    fn scheduler_page_state_integration() {
        let mut scheduler = Scheduler::new();
        let req_id = scheduler.enqueue(RequestKind::Generate, "test");

        // 模拟分配页面
        let pages = vec![PageId(0), PageId(1)];
        scheduler.mark_pages_accessed(req_id, pages.clone());

        // 检查页面状态
        let snapshot = scheduler.page_states_snapshot();
        let active_pages: Vec<_> = snapshot.iter()
            .filter(|(_, state, _)| *state == PageState::Active)
            .collect();

        assert!(!active_pages.is_empty());
    }

    #[test]
    fn scheduler_pause_resume_sequence() {
        let mut scheduler = Scheduler::new();
        let req_id = scheduler.enqueue(RequestKind::Generate, "test");

        scheduler.start_sequence(req_id, vec![PageId(0)]);
        assert!(scheduler.is_sequence_running(req_id));

        // 暂停序列
        scheduler.pause_sequence(req_id);
        assert!(!scheduler.is_sequence_running(req_id));

        // 恢复序列
        scheduler.resume_sequence(req_id);
        assert!(scheduler.is_sequence_running(req_id));
    }
}
