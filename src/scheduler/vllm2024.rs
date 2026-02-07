//! 2024 vLLM optimizations (Chunked Prefill / SwiftKV / LMCache).
//!
//! This module keeps the orchestration logic fully on the host side so it
//! remains compatible with the AOT CUBIN strategy (no runtime kernel
//! compilation) and the zero-copy generation loop.

use std::collections::{HashMap, VecDeque};

use sha2::{Digest, Sha256};

use gllm_kernels::backend_trait::{KvCacheHandle, LogitsHandle};
use gllm_kernels::kernel_types::RequestId;

/// Chunked Prefill configuration (ARCH-SCHED-CHUNKED).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkedConfig {
    pub chunk_size: usize,
    pub decode_slots: usize,
    pub enable_splitfuse: bool,
}

impl Default for ChunkedConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64,
            decode_slots: 2,
            enable_splitfuse: true,
        }
    }
}

/// SwiftKV configuration (ARCH-SCHED-SWIFTKV).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwiftKVConfig {
    pub enabled: bool,
    pub window_size: usize,
    pub enable_across_kv: bool,
    pub similarity_threshold: f32,
    pub precision_guard: f32,
}

impl Default for SwiftKVConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window_size: 4,
            enable_across_kv: false,
            similarity_threshold: 0.9,
            precision_guard: 0.1,
        }
    }
}

/// LMCache L3 backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L3Backend {
    Redis,
    LocalDisk,
    Disabled,
}

/// LMCache configuration (ARCH-SCHED-LMCACHE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LMCacheConfig {
    pub l1_capacity_mb: usize,
    pub l2_capacity_mb: usize,
    pub l3_backend: L3Backend,
    pub cache_prefix_len: usize,
}

impl Default for LMCacheConfig {
    fn default() -> Self {
        Self {
            l1_capacity_mb: 128,
            l2_capacity_mb: 8_192,
            l3_backend: L3Backend::LocalDisk,
            cache_prefix_len: 512,
        }
    }
}

/// Unified configuration gate for the three optimizations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Scheduler2024Config {
    pub enable_2024_optimizations: bool,
    pub chunked: ChunkedConfig,
    pub swift_kv: SwiftKVConfig,
    pub lmcache: LMCacheConfig,
}

// ----------------------------- Chunked Prefill -----------------------------

#[derive(Debug, Clone)]
pub struct PrefillChunk {
    pub request_id: RequestId,
    pub chunk_idx: usize,
    pub total_chunks: usize,
    pub tokens: usize,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct ChunkTracker {
    pub prompt: String,
    pub total_tokens: usize,
    pub chunk_size: usize,
    pub completed_chunks: usize,
    pub pending_tokens: usize,
}

impl ChunkTracker {
    fn next_chunk(&mut self) -> Option<PrefillChunk> {
        if self.pending_tokens == 0 {
            return None;
        }
        let tokens = self.pending_tokens.min(self.chunk_size);
        let chunk_idx = self.completed_chunks;
        self.pending_tokens = self.pending_tokens.saturating_sub(tokens);
        let total_chunks = self.total_tokens.div_ceil(self.chunk_size);
        Some(PrefillChunk {
            request_id: 0, // patched by caller
            chunk_idx,
            total_chunks,
            tokens,
            prompt: self.prompt.clone(),
        })
    }
}

#[derive(Debug)]
pub struct ChunkedState {
    pub config: ChunkedConfig,
    pub chunk_queue: VecDeque<PrefillChunk>,
    trackers: HashMap<RequestId, ChunkTracker>,
}

impl ChunkedState {
    pub fn new(config: ChunkedConfig) -> Self {
        Self {
            config,
            chunk_queue: VecDeque::new(),
            trackers: HashMap::new(),
        }
    }

    /// Register a prefill request and enqueue its first chunk.
    pub fn enqueue(&mut self, request_id: RequestId, prompt: String, tokens: usize) {
        if tokens == 0 {
            return;
        }
        let tracker = ChunkTracker {
            prompt: prompt.clone(),
            total_tokens: tokens,
            chunk_size: self.config.chunk_size,
            completed_chunks: 0,
            pending_tokens: tokens,
        };
        self.trackers.insert(request_id, tracker);
        self.maybe_push_next_chunk(request_id);
    }

    /// Mark a chunk as finished and enqueue the next one if pending.
    pub fn on_chunk_finished(&mut self, request_id: RequestId) {
        if let Some(tracker) = self.trackers.get_mut(&request_id) {
            tracker.completed_chunks = tracker.completed_chunks.saturating_add(1);
        }
        self.maybe_push_next_chunk(request_id);
    }

    pub fn is_request_complete(&self, request_id: &RequestId) -> bool {
        self.trackers
            .get(request_id)
            .map(|t| t.pending_tokens == 0)
            .unwrap_or(true)
    }

    pub fn pop_chunk(&mut self) -> Option<PrefillChunk> {
        self.chunk_queue.pop_front()
    }

    fn maybe_push_next_chunk(&mut self, request_id: RequestId) {
        if let Some(tracker) = self.trackers.get_mut(&request_id) {
            if let Some(mut chunk) = tracker.next_chunk() {
                chunk.request_id = request_id;
                self.chunk_queue.push_back(chunk);
            }
        }
    }

    pub fn remove_tracker(&mut self, request_id: &RequestId) {
        self.trackers.remove(request_id);
    }
}

// ------------------------------- SwiftKV ----------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistillResult {
    pub original_pages: usize,
    pub distilled_pages: usize,
}

/// Detailed CPU distillation outcome with optional precision signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistillOutcome {
    pub result: DistillResult,
    /// Approximated perplexity delta between original and distilled KV.
    pub ppl_diff: f32,
    /// Whether precision guard forced a fallback to original KV.
    pub precision_fallback: bool,
}

#[derive(Debug)]
pub struct SwiftKvState {
    pub config: SwiftKVConfig,
    /// Last distillation result (for metrics/tests).
    pub last_result: Option<DistillResult>,
}

impl SwiftKvState {
    pub fn new(config: SwiftKVConfig) -> Self {
        Self {
            config,
            last_result: None,
        }
    }

    /// Distill a page set (CPU-side) to model SwiftKV SIKV + AKV effects.
    pub fn distill_pages(&mut self, page_count: usize) -> DistillResult {
        if !self.config.enabled || page_count == 0 {
            let result = DistillResult {
                original_pages: page_count,
                distilled_pages: page_count,
            };
            self.last_result = Some(result);
            return result;
        }

        let mut distilled = page_count.div_ceil(self.config.window_size);
        if self.config.enable_across_kv {
            // AKV: further share across layers; model it as a 50% reduction when enabled.
            distilled = distilled.div_ceil(2);
        }

        let result = DistillResult {
            original_pages: page_count,
            distilled_pages: distilled.max(1),
        };
        self.last_result = Some(result);
        result
    }

    /// CPU-side SIKV + AKV distillation over concrete KV page vectors.
    /// This keeps zero-copy on GPU: only CPU buffers are manipulated.
    pub fn distill_cpu(&mut self, pages: &[Vec<f32>]) -> DistillOutcome {
        if !self.config.enabled || pages.is_empty() {
            let res = DistillResult {
                original_pages: pages.len(),
                distilled_pages: pages.len(),
            };
            return DistillOutcome {
                result: res,
                ppl_diff: 0.0,
                precision_fallback: false,
            };
        }

        // 1) SIKV: sliding-window merge.
        let mut merged: Vec<Vec<f32>> = Vec::new();
        let w = self.config.window_size.max(1);
        for chunk in pages.chunks(w) {
            let mut acc = vec![0f32; chunk[0].len()];
            let mut weight_sum = 0f32;
            for page in chunk {
                let w = 1.0; // simple average; attention weights unknown on CPU-only path.
                weight_sum += w;
                for (dst, src) in acc.iter_mut().zip(page.iter()) {
                    *dst += w * *src;
                }
            }
            if weight_sum > 0.0 {
                for v in acc.iter_mut() {
                    *v /= weight_sum;
                }
            }
            merged.push(acc);
        }

        // 2) AKV: cosine similarity across adjacent layers.
        let mut akv_shared: Vec<Vec<f32>> = Vec::new();
        let mut precision_fallback = false;
        for i in 0..merged.len() {
            if i > 0 && self.config.enable_across_kv {
                let sim = cosine_similarity(&merged[i - 1], &merged[i]);
                if sim >= self.config.similarity_threshold {
                    // Share previous layer; no extra storage needed.
                    continue;
                }
            }
            akv_shared.push(merged[i].clone());
        }

        // 3) Precision guard: compare reconstruction error as proxy for PPL diff.
        let ppl_diff = approx_ppl_delta(pages, &akv_shared);
        if ppl_diff > self.config.precision_guard {
            // Fallback to original pages for correctness.
            precision_fallback = true;
            let res = DistillResult {
                original_pages: pages.len(),
                distilled_pages: pages.len(),
            };
            self.last_result = Some(res);
            return DistillOutcome {
                result: res,
                ppl_diff,
                precision_fallback,
            };
        }

        let res = DistillResult {
            original_pages: pages.len(),
            distilled_pages: akv_shared.len().max(1),
        };
        self.last_result = Some(res);
        DistillOutcome {
            result: res,
            ppl_diff,
            precision_fallback,
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn approx_ppl_delta(original: &[Vec<f32>], distilled: &[Vec<f32>]) -> f32 {
    // Use normalized L2 reconstruction error as a proxy; small -> low PPL shift.
    let mut total = 0f32;
    let mut count = 0usize;
    for (i, page) in original.iter().enumerate() {
        let distilled_page = distilled.get(i.min(distilled.len().saturating_sub(1)));
        if let Some(dp) = distilled_page {
            let mut l2 = 0f32;
            for (x, y) in page.iter().zip(dp.iter()) {
                let d = x - y;
                l2 += d * d;
            }
            let norm = page.iter().map(|v| v * v).sum::<f32>().max(1e-6);
            total += (l2 / norm).sqrt();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

// -------------------------------- LMCache ---------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
}

#[derive(Debug, Clone)]
pub struct CacheHit {
    pub level: CacheLevel,
    pub prefix_tokens: usize,
    pub kv_handle: Option<KvCacheHandle>,
    pub logits_handle: Option<LogitsHandle>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    key: String,
    prefix_tokens: usize,
    bytes: usize,
    kv_handle: Option<KvCacheHandle>,
    logits_handle: Option<LogitsHandle>,
}

#[derive(Debug)]
pub struct LmcacheState {
    config: LMCacheConfig,
    l1: HashMap<String, CacheEntry>,
    l1_order: VecDeque<String>,
    l1_bytes: usize,
    l2: HashMap<String, CacheEntry>,
    l2_order: VecDeque<String>,
    l2_bytes: usize,
    l3: HashMap<String, CacheEntry>,
}

impl LmcacheState {
    pub fn new(config: LMCacheConfig) -> Self {
        Self {
            config,
            l1: HashMap::new(),
            l1_order: VecDeque::new(),
            l1_bytes: 0,
            l2: HashMap::new(),
            l2_order: VecDeque::new(),
            l2_bytes: 0,
            l3: HashMap::new(),
        }
    }

    pub fn cache_key(model_id: &str, prompt: &str, prefix_len: usize) -> String {
        let truncated: String = prompt.chars().take(prefix_len).collect();
        let mut hasher = Sha256::new();
        hasher.update(model_id.as_bytes());
        hasher.update(truncated.as_bytes());
        let digest = hasher.finalize();
        digest.iter().map(|b| format!("{:02x}", b)).collect()
    }

    pub fn get(&mut self, key: &str) -> Option<CacheHit> {
        if let Some(entry) = self.l1.get(key).cloned() {
            self.touch_l1(key);
            return Some(CacheHit {
                level: CacheLevel::L1,
                prefix_tokens: entry.prefix_tokens,
                kv_handle: entry.kv_handle,
                logits_handle: entry.logits_handle,
            });
        }
        if let Some(entry) = self.l2.get(key).cloned() {
            self.promote_l2_to_l1(key.to_owned());
            return Some(CacheHit {
                level: CacheLevel::L2,
                prefix_tokens: entry.prefix_tokens,
                kv_handle: entry.kv_handle,
                logits_handle: entry.logits_handle,
            });
        }
        if self.config.l3_backend != L3Backend::Disabled {
            if let Some(entry) = self.l3.get(key).cloned() {
                // L3 → L2 (asynchronous in real system; synchronous here for simplicity)
                self.insert_l2(entry.clone());
                return Some(CacheHit {
                    level: CacheLevel::L3,
                    prefix_tokens: entry.prefix_tokens,
                    kv_handle: entry.kv_handle,
                    logits_handle: entry.logits_handle,
                });
            }
        }
        None
    }

    pub fn put(
        &mut self,
        key: String,
        prefix_tokens: usize,
        kv_handle: Option<KvCacheHandle>,
        logits_handle: Option<LogitsHandle>,
    ) {
        let bytes = self.approx_bytes(prefix_tokens);
        let entry = CacheEntry {
            key: key.clone(),
            prefix_tokens,
            bytes,
            kv_handle,
            logits_handle,
        };
        self.insert_l1(entry.clone());
        self.insert_l2(entry.clone());
        if self.config.l3_backend != L3Backend::Disabled {
            self.l3.insert(key, entry);
        }
    }

    fn approx_bytes(&self, tokens: usize) -> usize {
        // Rough estimate: assume fp16 KV of 2 bytes * (K+V). Use 4 bytes per token per head slice.
        tokens * 4
    }

    fn touch_l1(&mut self, key: &str) {
        if let Some(pos) = self.l1_order.iter().position(|k| k == key) {
            self.l1_order.remove(pos);
        }
        self.l1_order.push_back(key.to_owned());
    }

    fn insert_l1(&mut self, entry: CacheEntry) {
        if let Some(existing) = self.l1.remove(&entry.key) {
            self.l1_bytes = self.l1_bytes.saturating_sub(existing.bytes);
            if let Some(pos) = self.l1_order.iter().position(|k| k == &entry.key) {
                self.l1_order.remove(pos);
            }
        }
        self.evict_until_fit(entry.bytes, true);
        self.l1_bytes += entry.bytes;
        self.l1_order.push_back(entry.key.clone());
        self.l1.insert(entry.key.clone(), entry);
    }

    fn insert_l2(&mut self, entry: CacheEntry) {
        if let Some(existing) = self.l2.remove(&entry.key) {
            self.l2_bytes = self.l2_bytes.saturating_sub(existing.bytes);
            if let Some(pos) = self.l2_order.iter().position(|k| k == &entry.key) {
                self.l2_order.remove(pos);
            }
        }
        self.evict_until_fit(entry.bytes, false);
        self.l2_bytes += entry.bytes;
        self.l2_order.push_back(entry.key.clone());
        self.l2.insert(entry.key.clone(), entry);
    }

    fn promote_l2_to_l1(&mut self, key: String) {
        if let Some(entry) = self.l2.get(&key).cloned() {
            self.touch_l2(&key);
            self.insert_l1(entry);
        }
    }

    fn touch_l2(&mut self, key: &str) {
        if let Some(pos) = self.l2_order.iter().position(|k| k == key) {
            self.l2_order.remove(pos);
        }
        self.l2_order.push_back(key.to_owned());
    }

    fn evict_until_fit(&mut self, incoming: usize, l1: bool) {
        let (cap_bytes, order, map, used) = if l1 {
            (
                self.config.l1_capacity_mb * 1024 * 1024,
                &mut self.l1_order,
                &mut self.l1,
                &mut self.l1_bytes,
            )
        } else {
            (
                self.config.l2_capacity_mb * 1024 * 1024,
                &mut self.l2_order,
                &mut self.l2,
                &mut self.l2_bytes,
            )
        };

        while *used + incoming > cap_bytes {
            if let Some(oldest_key) = order.pop_front() {
                if let Some(entry) = map.remove(&oldest_key) {
                    *used = used.saturating_sub(entry.bytes);
                }
            } else {
                break;
            }
        }
    }
}

// --------------------------- State Aggregation ----------------------------

#[derive(Debug)]
pub struct Scheduler2024State {
    pub config: Scheduler2024Config,
    pub chunked: ChunkedState,
    pub swift_kv: SwiftKvState,
    pub lmcache: LmcacheState,
}

impl Scheduler2024State {
    pub fn new(config: Scheduler2024Config) -> Self {
        Self {
            chunked: ChunkedState::new(config.chunked),
            swift_kv: SwiftKvState::new(config.swift_kv),
            lmcache: LmcacheState::new(config.lmcache),
            config,
        }
    }
}
