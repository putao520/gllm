//! Scheduler (PagedAttention / Continuous Batching / Double Buffering).

use std::collections::VecDeque;

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
}

impl PagePool {
    pub fn new(page_size: usize, total_pages: usize) -> Self {
        let mut free = VecDeque::with_capacity(total_pages);
        for idx in 0..total_pages {
            free.push_back(PageId(idx));
        }
        Self {
            page_size: page_size.max(1),
            free,
            total_pages,
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
        Some(PageAllocation { pages, tokens })
    }

    pub fn release(&mut self, allocation: PageAllocation) {
        for page in allocation.pages {
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
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
            && self.buffer.front().batch.is_none()
            && self.buffer.back().batch.is_none()
    }

    pub fn free_pages(&self) -> usize {
        self.page_pool.free_pages()
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
}
