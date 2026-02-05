//! PagedScheduler (PagedAttention / Continuous Batching / Double Buffering).

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::engine::vllm2024::{ChunkedConfig, Scheduler2024Config, Scheduler2024State};
use crate::kv_cache::KvCacheSlot;
use crate::scheduler::{GroupState, HGALConfig, HGALScheduler, SequenceGroup};
pub use gllm_kernels::kernel_types::{PageId, PageState, RequestId};

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
    pub chunk_info: Option<ChunkInfo>,
}

impl ScheduledRequest {
    fn token_len(&self) -> usize {
        self.tokens.max(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkInfo {
    pub chunk_idx: usize,
    pub total_chunks: usize,
}

#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub id: BatchId,
    pub requests: Vec<ScheduledRequest>,
    pub allocations: Vec<PageAllocation>,
    pub total_tokens: usize,
    pub kv_cache_slot: KvCacheSlot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerConfig {
    pub page_size: usize,
    pub total_pages: usize,
    pub max_batch: usize,
    pub max_tokens: usize,
    pub warmup_duration: Duration,
    pub working_set_window: Duration,
    pub hot_threshold: usize,
    pub lir_ratio: f32,
    pub min_warm_access: usize,
    pub enable_clock_pro: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            page_size: 128,
            total_pages: 2048,
            max_batch: 8,
            max_tokens: 4096,
            warmup_duration: Duration::from_millis(100),
            working_set_window: Duration::from_secs(1),
            hot_threshold: 3,
            lir_ratio: 0.3,
            min_warm_access: 2,
            enable_clock_pro: true,
        }
    }
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
}

impl PagePool {
    pub fn new(page_size: usize, total_pages: usize) -> Self {
        let mut free = VecDeque::with_capacity(total_pages);
        let mut page_states = HashMap::new();
        for idx in 0..total_pages {
            let page_id = idx;
            free.push_back(page_id);
            // 初始化所有页面为 Standby 状态
            page_states.insert(
                page_id,
                PageEntry {
                    page_id,
                    state: PageState::Standby,
                    last_access: Instant::now(),
                    owner_request: None,
                },
            );
        }
        Self {
            page_size: page_size.max(1),
            free,
            total_pages,
            page_states,
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

    /// 标记页面为已访问
    pub fn mark_accessed(&mut self, page_ids: &[PageId], owner: RequestId) {
        let now = Instant::now();
        for &page_id in page_ids {
            // 更新页面状态
            if let Some(entry) = self.page_states.get_mut(&page_id) {
                if entry.state != PageState::Swapped {
                    if entry.state == PageState::Standby {
                        entry.state = PageState::Active;
                    }
                    entry.last_access = now;
                    entry.owner_request = Some(owner);
                }
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

    pub fn owner(&self, page_id: PageId) -> Option<RequestId> {
        self.page_states
            .get(&page_id)
            .and_then(|entry| entry.owner_request)
    }

    /// 获取所有页面状态（用于调试）
    pub fn page_states_snapshot(&self) -> Vec<(PageId, PageState, Option<RequestId>)> {
        self.page_states
            .values()
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

    pub fn swap_out_pages(&mut self, pages: &[PageId]) {
        let now = Instant::now();
        for &page in pages {
            if let Some(entry) = self.page_states.get_mut(&page) {
                entry.state = PageState::Swapped;
                entry.last_access = now;
            }
            self.free.retain(|&pid| pid != page);
        }
    }

    pub fn mark_swap_in(&mut self, page_ids: &[PageId], owner: RequestId) {
        let now = Instant::now();
        for &page in page_ids {
            if let Some(entry) = self.page_states.get_mut(&page) {
                entry.state = PageState::Warm;
                entry.owner_request = Some(owner);
                entry.last_access = now;
            }
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
pub struct PagedScheduler {
    config: SchedulerConfig,
    next_id: RequestId,
    next_batch_id: u64,
    queue: VecDeque<ScheduledRequest>,
    page_pool: PagePool,
    hgal: HGALScheduler,
    batcher: DynamicBatcher,
    buffer: DoubleBuffer<BatchSlot>,
    next_slot: KvCacheSlot,
    /// 序列跟踪（用于 continuous batching）
    sequences: HashMap<RequestId, SequenceInfo>,
    /// 正在运行的序列 ID 列表
    running_sequences: Vec<RequestId>,
    /// 2024 vLLM 优化状态（可选）
    vllm24: Option<Scheduler2024State>,
}

impl PagedScheduler {
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    pub fn with_config(config: SchedulerConfig) -> Self {
        let page_pool = PagePool::new(config.page_size, config.total_pages);
        let batcher = DynamicBatcher::new(config.max_batch, config.max_tokens);
        let hgal = HGALScheduler::new(HGALConfig {
            warmup_duration: config.warmup_duration,
            working_set_window: config.working_set_window,
            hot_threshold: config.hot_threshold,
            lir_ratio: config.lir_ratio,
            min_warm_access: config.min_warm_access,
            enable_clock_pro: config.enable_clock_pro,
        });
        Self {
            config,
            next_id: 0,
            next_batch_id: 0,
            queue: VecDeque::new(),
            page_pool,
            hgal,
            batcher,
            buffer: DoubleBuffer::new(BatchSlot::default(), BatchSlot::default()),
            next_slot: KvCacheSlot::Front,
            sequences: HashMap::new(),
            running_sequences: Vec::new(),
            vllm24: None,
        }
    }

    pub fn config(&self) -> SchedulerConfig {
        self.config
    }

    /// Enable 2024 vLLM optimizations (Chunked Prefill, SwiftKV, LMCache-aware scheduling hooks).
    pub fn enable_vllm_2024(&mut self, config: Scheduler2024Config) {
        self.vllm24 = Some(Scheduler2024State::new(config));
    }

    /// LMCache lookup; returns hit information when enabled.
    pub fn lmcache_lookup(
        &mut self,
        model_id: &str,
        prompt: &str,
    ) -> Option<crate::engine::vllm2024::CacheHit> {
        if let Some(v) = self.vllm24.as_mut() {
            if v.config.enable_2024_optimizations {
                let key = crate::engine::vllm2024::LmcacheState::cache_key(
                    model_id,
                    prompt,
                    v.config.lmcache.cache_prefix_len,
                );
                return v.lmcache.get(&key);
            }
        }
        None
    }

    /// Store prefix KV into LMCache (host side metadata only; zero-copy preserved).
    pub fn lmcache_put(
        &mut self,
        model_id: &str,
        prompt: &str,
        kv_handle: Option<gllm_kernels::backend_trait::KvCacheHandle>,
        logits_handle: Option<gllm_kernels::backend_trait::LogitsHandle>,
    ) {
        if let Some(v) = self.vllm24.as_mut() {
            if v.config.enable_2024_optimizations {
                let key = crate::engine::vllm2024::LmcacheState::cache_key(
                    model_id,
                    prompt,
                    v.config.lmcache.cache_prefix_len,
                );
                v.lmcache.put(
                    key,
                    v.config.lmcache.cache_prefix_len,
                    kv_handle,
                    logits_handle,
                );
            }
        }
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
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let prompt = prompt.into();

        if let Some(v) = self.vllm24.as_mut() {
            if v.config.enable_2024_optimizations && matches!(kind, RequestKind::Generate) {
                let chunk_size = v.config.chunked.chunk_size.max(1);
                if tokens > chunk_size {
                    v.chunked.enqueue(id, prompt.clone(), tokens);
                    self.drain_chunk_queue();
                    return id;
                }
            }
        }

        self.queue.push_back(ScheduledRequest {
            id,
            kind,
            prompt,
            tokens,
            chunk_info: None,
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
            // Chunked Prefill progress
            if self
                .vllm24
                .as_ref()
                .map(|v| v.config.enable_2024_optimizations && request.chunk_info.is_some())
                .unwrap_or(false)
            {
                if let Some(v) = self.vllm24.as_mut() {
                    v.chunked.on_chunk_finished(request.id);
                }
                self.drain_chunk_queue();
                if let Some(v) = self.vllm24.as_mut() {
                    if v.chunked.is_request_complete(&request.id) {
                        v.chunked.remove_tracker(&request.id);
                    }
                }
            }

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

    /// 标记请求的页面为已访问，触发 HGAL 元数据更新。
    pub fn mark_pages_accessed(&mut self, request_id: RequestId, page_ids: Vec<PageId>) {
        self.page_pool.mark_accessed(&page_ids, request_id);
        for &pid in &page_ids {
            self.hgal
                .update_page_state(pid, Some(request_id), PageState::Active);
            self.hgal.mark_accessed(pid);
        }
        // 更新序列信息
        if let Some(seq) = self.sequences.get_mut(&request_id) {
            seq.allocated_pages = page_ids.clone();
        }
        // 更新 gang 元数据
        let state = self
            .sequences
            .get(&request_id)
            .map(|seq| match seq.state {
                SequenceState::Paused => GroupState::Paused,
                _ => GroupState::Running,
            })
            .unwrap_or(GroupState::Running);
        self.update_group_tracking(request_id, page_ids, state, false);
        self.hgal.detect_working_set();
    }

    /// 获取页面池的状态快照
    pub fn page_states_snapshot(&self) -> Vec<(PageId, PageState, Option<RequestId>)> {
        self.page_pool.page_states_snapshot()
    }

    /// Gang-aware: choose victim groups instead of single pages.
    pub fn select_victim_groups(&mut self, page_count: usize) -> Vec<RequestId> {
        self.hgal.select_victim_groups(page_count)
    }

    /// 设置页面状态
    pub fn set_page_state(&mut self, page_id: PageId, state: PageState) {
        self.page_pool.set_page_state(page_id, state);
        let owner = self.page_pool.owner(page_id);
        self.hgal.update_page_state(page_id, owner, state);
    }

    /// swap-in 完成后标记页面进入 Warm 保护期
    pub fn on_swap_in(&mut self, request_id: RequestId, page_ids: &[PageId]) {
        self.page_pool.mark_swap_in(page_ids, request_id);
        for &pid in page_ids {
            self.hgal
                .update_page_state(pid, Some(request_id), PageState::Warm);
            self.hgal.on_swap_in(pid);
        }
    }

    /// 从后端同步页面状态（get_page_states 集成）
    pub fn sync_page_states(&mut self, states: &[(PageId, PageState)]) {
        for (pid, state) in states {
            self.set_page_state(*pid, *state);
            if *state == PageState::Warm {
                self.hgal.on_swap_in(*pid);
            }
        }
    }

    // ========== 新增：Continuous Batching 集成方法 ==========

    /// 开始跟踪一个序列
    pub fn start_sequence(&mut self, request_id: RequestId, initial_pages: Vec<PageId>) {
        let pages = initial_pages.clone();
        self.sequences.insert(
            request_id,
            SequenceInfo {
                id: request_id,
                state: SequenceState::Running,
                generated_tokens: 0,
                position: 0,
                allocated_pages: initial_pages,
            },
        );
        self.running_sequences.push(request_id);
        self.mark_pages_accessed(request_id, pages);
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
            let pages = seq.allocated_pages.clone();
            self.page_pool.release(PageAllocation {
                pages: pages.clone(),
                tokens: 0,
            });
            for pid in pages {
                self.hgal.update_page_state(pid, None, PageState::Standby);
            }
        }
        self.hgal.remove_group(request_id);
        self.running_sequences.retain(|&id| id != request_id);
    }

    /// 标记序列为暂停
    pub fn pause_sequence(&mut self, request_id: RequestId) {
        if let Some(seq) = self.sequences.get_mut(&request_id) {
            seq.state = SequenceState::Paused;
            let pages = seq.allocated_pages.clone();
            self.update_group_tracking(request_id, pages, GroupState::Paused, false);
        }
        self.running_sequences.retain(|&id| id != request_id);
    }

    /// 恢复暂停的序列
    pub fn resume_sequence(&mut self, request_id: RequestId) {
        if let Some(seq) = self.sequences.get_mut(&request_id) {
            seq.state = SequenceState::Running;
            self.running_sequences.push(request_id);
            let pages = seq.allocated_pages.clone();
            self.update_group_tracking(request_id, pages, GroupState::Running, false);
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
        if let Some(v) = &self.vllm24 {
            if v.config.enable_2024_optimizations {
                return self.build_chunked_batch();
            }
        }

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

    /// Chunked Prefill + Decode interleaving (REQ-SCHED-007).
    fn build_chunked_batch(&mut self) -> Option<ScheduledBatch> {
        let chunk_cfg = self
            .vllm24
            .as_ref()
            .map(|v| v.config.chunked)
            .unwrap_or(ChunkedConfig::default());

        let mut requests = Vec::new();
        let mut allocations = Vec::new();
        let mut total_tokens = 0usize;
        let mut decode_taken = 0usize;
        let mut remainder = VecDeque::new();

        while let Some(req) = self.queue.pop_front() {
            let is_decode = req.token_len() <= 1 && decode_taken < chunk_cfg.decode_slots;
            let fits_batch = requests.len() < self.config.max_batch;
            let fits_tokens = total_tokens + req.token_len() <= self.config.max_tokens;

            if (is_decode || fits_batch) && fits_tokens {
                if is_decode {
                    decode_taken += 1;
                }
                if let Some(allocation) = self.page_pool.allocate(req.token_len()) {
                    total_tokens = total_tokens.saturating_add(req.token_len());
                    allocations.push(allocation);
                    requests.push(req);
                } else {
                    remainder.push_back(req);
                    break;
                }
            } else {
                remainder.push_back(req);
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

    // ========== 核心调度逻辑 (REQ-SCHED-001, REQ-SCHED-003) ==========

    /// 处理页面错误 (页面不在 GPU 内存)
    /// 当访问的页面已被换出到 CPU 时触发
    pub fn handle_page_fault(
        &mut self,
        page_id: PageId,
        requesting_sequence: RequestId,
    ) -> PageLocation {
        match self.page_pool.get_page_state(page_id) {
            Some(PageState::Swapped) => {
                // 页面已换出到 CPU，需要 swap-in
                // 标记序列为暂停状态
                self.pause_sequence(requesting_sequence);
                // 添加到 swap-in 等待队列（通过 sequences 记录）
                if let Some(seq) = self.sequences.get_mut(&requesting_sequence) {
                    seq.state = SequenceState::Paused;
                }
                self.hgal
                    .update_page_state(page_id, Some(requesting_sequence), PageState::Swapped);
                self.update_group_tracking(
                    requesting_sequence,
                    self.sequences
                        .get(&requesting_sequence)
                        .map(|s| s.allocated_pages.clone())
                        .unwrap_or_default(),
                    GroupState::Paused,
                    false,
                );
                PageLocation::Cpu
            }
            Some(PageState::Active | PageState::Protected | PageState::Warm) => {
                self.hgal.mark_accessed(page_id);
                PageLocation::Gpu
            }
            Some(PageState::Standby) => {
                // 页面在 GPU 中但未使用，标记为 Active
                self.set_page_state(page_id, PageState::Active);
                self.page_pool
                    .mark_accessed(&[page_id], requesting_sequence);
                self.hgal.mark_accessed(page_id);
                PageLocation::Gpu
            }
            None => {
                // 页面不存在
                PageLocation::Invalid
            }
        }
    }

    /// 获取批次的页表
    /// 用于批处理前向传播，构建每个序列的页表映射
    pub fn get_batch_page_table(&self, request_ids: &[RequestId]) -> BatchPageTable {
        let mut page_table = BatchPageTable {
            sequence_pages: HashMap::new(),
            total_pages: 0,
        };

        for &request_id in request_ids {
            if let Some(seq_info) = self.sequences.get(&request_id) {
                let pages: Vec<_> = seq_info
                    .allocated_pages
                    .iter()
                    .filter_map(|&pid| self.page_pool.get_page_state(pid).map(|state| (pid, state)))
                    .collect();
                let page_count = pages.len();
                page_table.sequence_pages.insert(request_id, pages);
                page_table.total_pages += page_count;
            }
        }

        page_table
    }

    /// 更新批次状态 (Continuous Batching 核心逻辑)
    /// 根据生成结果决定序列继续/完成/暂停
    pub fn update_batch(&mut self, batch: &ScheduledBatch, results: &BatchResult) -> BatchAction {
        let mut continue_ids = Vec::new();
        let mut complete_ids = Vec::new();
        let mut pause_ids = Vec::new();

        for (request, result) in batch.requests.iter().zip(results.results.iter()) {
            match result {
                SequenceResult::Continue => {
                    // 序列继续生成
                    if let Some(seq) = self.sequences.get_mut(&request.id) {
                        seq.state = SequenceState::Running;
                        seq.generated_tokens += 1;
                    }
                    continue_ids.push(request.id);
                }
                SequenceResult::Complete => {
                    // 序列完成
                    self.complete_sequence(request.id);
                    complete_ids.push(request.id);
                }
                SequenceResult::PageFault { page_id } => {
                    // 页面错误，需要暂停并等待 swap-in
                    let location = self.handle_page_fault(*page_id, request.id);
                    if location == PageLocation::Cpu {
                        pause_ids.push(request.id);
                    } else {
                        // 页面应该可用，继续
                        continue_ids.push(request.id);
                    }
                }
                SequenceResult::Error { .. } => {
                    // 序列失败
                    if let Some(seq) = self.sequences.get_mut(&request.id) {
                        seq.state = SequenceState::Failed;
                    }
                    complete_ids.push(request.id);
                }
            }
        }

        BatchAction {
            continue_ids,
            complete_ids,
            pause_ids,
        }
    }

    /// 为序列分配页面（带自动 swap 支持）
    /// 如果 GPU 内存不足，会先换出 LRU 页面
    pub fn allocate_pages_for_sequence(
        &mut self,
        request_id: RequestId,
        token_count: usize,
    ) -> Result<Vec<PageId>, SchedulerError> {
        let pages_needed = (token_count + self.config.page_size - 1) / self.config.page_size;

        // 尝试直接分配
        let mut allocated = Vec::new();
        for _ in 0..pages_needed {
            if let Some(allocation) = self.page_pool.allocate(self.config.page_size) {
                for page in allocation.pages {
                    allocated.push(page);
                }
            } else {
                // 内存不足，需要换出一些页面
                break;
            }
        }

        // 如果分配不足，尝试 swap-out
        if allocated.len() < pages_needed {
            let additional = pages_needed - allocated.len();
            let victims = self.select_victim_groups(additional);

            for victim_id in victims {
                if let Some(seq) = self.sequences.get_mut(&victim_id) {
                    let pages = seq.allocated_pages.clone();
                    self.page_pool.swap_out_pages(&pages);
                    if let Some(v) = self.vllm24.as_mut() {
                        if v.config.enable_2024_optimizations {
                            v.swift_kv.distill_pages(pages.len());
                        }
                    }
                    for pid in &pages {
                        self.hgal
                            .update_page_state(*pid, Some(victim_id), PageState::Swapped);
                    }
                    seq.state = SequenceState::Paused;
                    self.update_group_tracking(victim_id, pages, GroupState::Swapped, false);
                    self.running_sequences.retain(|&id| id != victim_id);
                } else if let Some((pages, pinned)) = self
                    .hgal
                    .sequence_groups
                    .get(&victim_id)
                    .map(|group| (group.pages.clone(), group.is_pinned))
                {
                    self.page_pool.swap_out_pages(&pages);
                    if let Some(v) = self.vllm24.as_mut() {
                        if v.config.enable_2024_optimizations {
                            v.swift_kv.distill_pages(pages.len());
                        }
                    }
                    for pid in &pages {
                        self.hgal
                            .update_page_state(*pid, Some(victim_id), PageState::Swapped);
                    }
                    self.update_group_tracking(victim_id, pages, GroupState::Swapped, pinned);
                    self.running_sequences.retain(|&id| id != victim_id);
                }
            }

            // 重新尝试分配
            while allocated.len() < pages_needed {
                if let Some(allocation) = self.page_pool.allocate(self.config.page_size) {
                    for page in allocation.pages {
                        allocated.push(page);
                    }
                } else {
                    return Err(SchedulerError::OutOfMemory {
                        requested: pages_needed,
                        available: allocated.len(),
                    });
                }
            }
        }

        // 记录分配给序列的页面
        for &page_id in &allocated {
            self.page_pool.mark_accessed(&[page_id], request_id);
            self.hgal
                .update_page_state(page_id, Some(request_id), PageState::Active);
            self.hgal.mark_accessed(page_id);
        }

        Ok(allocated)
    }

    fn update_group_tracking(
        &mut self,
        request_id: RequestId,
        pages: Vec<PageId>,
        state: GroupState,
        pinned: bool,
    ) {
        let now = Instant::now();
        let (access_count, last_access) = self
            .hgal
            .sequence_groups
            .get(&request_id)
            .map(|g| (g.access_count, g.last_access))
            .unwrap_or((0, now));
        let last_access = if matches!(state, GroupState::Running) {
            now
        } else {
            last_access
        };
        let group = SequenceGroup {
            id: request_id,
            pages,
            state,
            access_count,
            last_access,
            is_pinned: pinned,
        };
        self.hgal.upsert_group(group);
    }

    /// Drain prepared prefill chunks (Chunked Prefill) into the scheduling queue.
    fn drain_chunk_queue(&mut self) {
        if let Some(v) = self.vllm24.as_mut() {
            while let Some(chunk) = v.chunked.pop_chunk() {
                self.queue.push_back(ScheduledRequest {
                    id: chunk.request_id,
                    kind: RequestKind::Generate,
                    prompt: chunk.prompt.clone(),
                    tokens: chunk.tokens,
                    chunk_info: Some(ChunkInfo {
                        chunk_idx: chunk.chunk_idx,
                        total_chunks: chunk.total_chunks,
                    }),
                });
            }
        }
    }
}

/// 页面位置（用于页面错误处理）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLocation {
    /// 页面在 GPU 内存中
    Gpu,
    /// 页面在 CPU 内存中（已换出）
    Cpu,
    /// 页面无效
    Invalid,
}

/// 批次页表
/// 映射每个序列到其页面列表和状态
#[derive(Debug, Clone)]
pub struct BatchPageTable {
    /// 序列 ID -> (PageId, PageState) 列表
    pub sequence_pages: HashMap<RequestId, Vec<(PageId, PageState)>>,
    /// 总页面数
    pub total_pages: usize,
}

/// 序列生成结果
#[derive(Debug, Clone)]
pub enum SequenceResult {
    /// 继续生成
    Continue,
    /// 生成完成
    Complete,
    /// 页面错误，需要 swap-in
    PageFault { page_id: PageId },
    /// 生成错误
    Error { message: String },
}

/// 批次处理结果
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// 每个序列的生成结果
    pub results: Vec<SequenceResult>,
}

impl BatchResult {
    /// 创建成功批次结果（所有序列继续）
    pub fn continued(count: usize) -> Self {
        Self {
            results: (0..count).map(|_| SequenceResult::Continue).collect(),
        }
    }

    /// 创建完成批次结果
    pub fn completed(indices: Vec<usize>) -> Self {
        let max_idx = indices.iter().copied().max().unwrap_or(0);
        let mut results = vec![SequenceResult::Continue; max_idx + 1];
        for idx in indices {
            if idx < results.len() {
                results[idx] = SequenceResult::Complete;
            }
        }
        Self { results }
    }
}

/// 批次动作（Continuous Batching 决策）
#[derive(Debug, Clone)]
pub struct BatchAction {
    /// 继续生成的序列 ID
    pub continue_ids: Vec<RequestId>,
    /// 完成的序列 ID
    pub complete_ids: Vec<RequestId>,
    /// 需要暂停的序列 ID（等待 swap-in）
    pub pause_ids: Vec<RequestId>,
}

impl BatchAction {
    /// 检查是否有任何动作
    pub fn has_actions(&self) -> bool {
        !self.continue_ids.is_empty() || !self.complete_ids.is_empty() || !self.pause_ids.is_empty()
    }

    /// 获取所有受影响的序列 ID
    pub fn all_affected_ids(&self) -> Vec<RequestId> {
        let mut ids = Vec::new();
        ids.extend(&self.continue_ids);
        ids.extend(&self.complete_ids);
        ids.extend(&self.pause_ids);
        ids
    }
}

/// 调度器错误类型
#[derive(Debug, Clone)]
pub enum SchedulerError {
    /// 内存不足
    OutOfMemory { requested: usize, available: usize },
    /// 页面不存在
    PageNotFound { page_id: PageId },
    /// 序列不存在
    SequenceNotFound { request_id: RequestId },
    /// 配置错误
    InvalidConfig(String),
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Out of memory: requested {}, available {}",
                    requested, available
                )
            }
            SchedulerError::PageNotFound { page_id } => {
                write!(f, "Page not found: {:?}", page_id)
            }
            SchedulerError::SequenceNotFound { request_id } => {
                write!(f, "Sequence not found: {:?}", request_id)
            }
            SchedulerError::InvalidConfig(msg) => {
                write!(f, "Invalid config: {}", msg)
            }
        }
    }
}

impl std::error::Error for SchedulerError {}

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
            ..SchedulerConfig::default()
        };
        let mut scheduler = PagedScheduler::with_config(config);
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
            ..SchedulerConfig::default()
        };
        let mut scheduler = PagedScheduler::with_config(config);
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

        let req1: RequestId = 1;
        let req2: RequestId = 2;

        pool.mark_accessed(&alloc1.pages, req1);
        pool.mark_accessed(&alloc2.pages, req2);

        for &page in &alloc1.pages {
            assert_eq!(pool.get_page_state(page), Some(PageState::Active));
            assert_eq!(pool.owner(page), Some(req1));
        }
        for &page in &alloc2.pages {
            assert_eq!(pool.get_page_state(page), Some(PageState::Active));
            assert_eq!(pool.owner(page), Some(req2));
        }
    }

    #[test]
    fn scheduler_tracks_sequences() {
        let mut scheduler = PagedScheduler::new();
        let req_id = scheduler.enqueue(RequestKind::Generate, "test");

        // 开始跟踪序列
        scheduler.start_sequence(req_id, vec![0, 1]);

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
        let mut scheduler = PagedScheduler::new();
        let req_id = scheduler.enqueue(RequestKind::Generate, "test");

        // 模拟分配页面
        let pages = vec![0, 1];
        scheduler.mark_pages_accessed(req_id, pages.clone());

        // 检查页面状态
        let snapshot = scheduler.page_states_snapshot();
        let active_pages: Vec<_> = snapshot
            .iter()
            .filter(|(_, state, _)| *state == PageState::Active)
            .collect();

        assert!(!active_pages.is_empty());
    }

    #[test]
    fn scheduler_pause_resume_sequence() {
        let mut scheduler = PagedScheduler::new();
        let req_id = scheduler.enqueue(RequestKind::Generate, "test");

        scheduler.start_sequence(req_id, vec![0]);
        assert!(scheduler.is_sequence_running(req_id));

        // 暂停序列
        scheduler.pause_sequence(req_id);
        assert!(!scheduler.is_sequence_running(req_id));

        // 恢复序列
        scheduler.resume_sequence(req_id);
        assert!(scheduler.is_sequence_running(req_id));
    }

    // ========== 新增：核心调度逻辑测试 ==========

    #[test]
    fn handle_page_fault_with_swapped_page() {
        let mut scheduler = PagedScheduler::new();
        let req_id = scheduler.enqueue(RequestKind::Generate, "test");
        let page_id = 0;

        scheduler.start_sequence(req_id, vec![page_id]);
        // 模拟页面已换出
        scheduler.set_page_state(page_id, PageState::Swapped);

        // 处理页面错误
        let location = scheduler.handle_page_fault(page_id, req_id);

        // 应该返回 CPU 位置（页面已换出）
        assert_eq!(location, PageLocation::Cpu);
        // 序列应该被暂停
        assert!(!scheduler.is_sequence_running(req_id));
    }

    #[test]
    fn handle_page_fault_with_active_page() {
        let mut scheduler = PagedScheduler::new();
        let req_id = scheduler.enqueue(RequestKind::Generate, "test");
        let page_id = 0;

        scheduler.start_sequence(req_id, vec![page_id]);
        // 页面应该是 Active 状态

        let location = scheduler.handle_page_fault(page_id, req_id);

        // 应该返回 GPU 位置
        assert_eq!(location, PageLocation::Gpu);
    }

    #[test]
    fn get_batch_page_table_returns_correct_pages() {
        let mut scheduler = PagedScheduler::new();
        let req1 = scheduler.enqueue(RequestKind::Generate, "test1");
        let req2 = scheduler.enqueue(RequestKind::Generate, "test2");

        let pages1 = vec![0, 1];
        let pages2 = vec![2, 3];

        scheduler.start_sequence(req1, pages1.clone());
        scheduler.start_sequence(req2, pages2.clone());

        // 获取批次页表
        let page_table = scheduler.get_batch_page_table(&[req1, req2]);

        assert_eq!(page_table.sequence_pages.len(), 2);
        assert!(page_table.sequence_pages.contains_key(&req1));
        assert!(page_table.sequence_pages.contains_key(&req2));
        assert_eq!(page_table.total_pages, 4);
    }

    #[test]
    fn update_batch_continues_running_sequences() {
        let mut scheduler = PagedScheduler::new();
        let req1 = scheduler.enqueue(RequestKind::Generate, "test1");
        let req2 = scheduler.enqueue(RequestKind::Generate, "test2");

        scheduler.start_sequence(req1, vec![0]);
        scheduler.start_sequence(req2, vec![1]);

        let batch = ScheduledBatch {
            id: BatchId(1),
            requests: vec![
                ScheduledRequest {
                    id: req1,
                    kind: RequestKind::Generate,
                    prompt: "a".into(),
                    tokens: 1,
                    chunk_info: None,
                },
                ScheduledRequest {
                    id: req2,
                    kind: RequestKind::Generate,
                    prompt: "b".into(),
                    tokens: 1,
                    chunk_info: None,
                },
            ],
            allocations: vec![],
            total_tokens: 2,
            kv_cache_slot: KvCacheSlot::Front,
        };

        let results = BatchResult::continued(2);
        let action = scheduler.update_batch(&batch, &results);

        // 两个序列都应该继续
        assert_eq!(action.continue_ids.len(), 2);
        assert!(action.continue_ids.contains(&req1));
        assert!(action.continue_ids.contains(&req2));
        assert!(action.complete_ids.is_empty());
        assert!(action.pause_ids.is_empty());
    }

    #[test]
    fn update_batch_complies_finished_sequences() {
        let mut scheduler = PagedScheduler::new();
        let req1 = scheduler.enqueue(RequestKind::Generate, "test1");
        let req2 = scheduler.enqueue(RequestKind::Generate, "test2");

        scheduler.start_sequence(req1, vec![0]);
        scheduler.start_sequence(req2, vec![1]);

        let batch = ScheduledBatch {
            id: BatchId(1),
            requests: vec![
                ScheduledRequest {
                    id: req1,
                    kind: RequestKind::Generate,
                    prompt: "a".into(),
                    tokens: 1,
                    chunk_info: None,
                },
                ScheduledRequest {
                    id: req2,
                    kind: RequestKind::Generate,
                    prompt: "b".into(),
                    tokens: 1,
                    chunk_info: None,
                },
            ],
            allocations: vec![],
            total_tokens: 2,
            kv_cache_slot: KvCacheSlot::Front,
        };

        // 创建包含两个结果的批次：第一个完成，第二个继续
        let results = BatchResult {
            results: vec![SequenceResult::Complete, SequenceResult::Continue],
        };
        let action = scheduler.update_batch(&batch, &results);

        // req1 应该完成，req2 应该继续
        assert!(action.complete_ids.contains(&req1));
        assert!(action.continue_ids.contains(&req2));
    }

    #[test]
    fn allocate_pages_for_sequence_with_sufficient_memory() {
        let config = SchedulerConfig {
            page_size: 4,
            total_pages: 10,
            max_batch: 4,
            max_tokens: 32,
            ..SchedulerConfig::default()
        };
        let mut scheduler = PagedScheduler::with_config(config);
        let req_id = scheduler.enqueue(RequestKind::Generate, "test");

        // 分配 2 个页面（8 个 token）
        let pages = scheduler
            .allocate_pages_for_sequence(req_id, 8)
            .expect("allocation");

        assert_eq!(pages.len(), 2);
    }

    #[test]
    fn chunked_prefill_progresses() {
        let mut scheduler = PagedScheduler::with_config(SchedulerConfig {
            page_size: 4,
            total_pages: 16,
            max_batch: 4,
            max_tokens: 64,
            ..SchedulerConfig::default()
        });
        scheduler.enable_vllm_2024(Scheduler2024Config {
            enable_2024_optimizations: true,
            chunked: ChunkedConfig {
                chunk_size: 4,
                decode_slots: 1,
                enable_splitfuse: true,
            },
            ..Scheduler2024Config::default()
        });

        let req_id = scheduler.enqueue_with_tokens(RequestKind::Generate, "long", 8);
        let batch1 = scheduler.next_batch().expect("first batch");
        assert_eq!(batch1.requests.len(), 1);
        assert_eq!(batch1.requests[0].chunk_info.unwrap().chunk_idx, 0);
        scheduler.complete_batch(batch1);

        let batch2 = scheduler.next_batch().expect("second batch");
        assert_eq!(batch2.requests.len(), 1);
        assert_eq!(batch2.requests[0].chunk_info.unwrap().chunk_idx, 1);
        let batch2_copy = batch2.clone();
        scheduler.complete_batch(batch2);

        assert!(scheduler.next_batch().is_none());
        assert_eq!(batch2_copy.requests[0].id, req_id);
    }

    #[test]
    fn batch_action_has_actions_detection() {
        let action = BatchAction {
            continue_ids: vec![1],
            complete_ids: vec![2],
            pause_ids: vec![3],
        };

        assert!(action.has_actions());
        assert_eq!(action.all_affected_ids().len(), 3);
    }

    #[test]
    fn batch_action_empty() {
        let action = BatchAction {
            continue_ids: vec![],
            complete_ids: vec![],
            pause_ids: vec![],
        };

        assert!(!action.has_actions());
        assert!(action.all_affected_ids().is_empty());
    }
}
