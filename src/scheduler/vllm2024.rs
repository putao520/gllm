//! 2024 vLLM optimizations (Chunked Prefill / SwiftKV).
//!
//! This module keeps the orchestration logic fully on the host side so it
//! remains compatible with the AOT CUBIN strategy (no runtime kernel
//! compilation) and the zero-copy generation loop.

use std::collections::HashMap;

use crate::compat::backend_trait::Element;
use crate::compat::cpu_kernels::Float;
use super::types::RequestId;

/// Chunked Prefill configuration (ARCH-SCHED-CHUNKED).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkedConfig {
    pub min_chunk: usize,
    pub max_chunk: usize,
    pub decode_slots: usize,
    /// ⛔ PERMANENTLY DISABLED (REQ-SCHED-007): SplitFuse 混批路径已废弃。
    /// 此字段硬编码为 false，禁止运行时修改。
    pub enable_splitfuse: bool, // always false
}

impl Default for ChunkedConfig {
    fn default() -> Self {
        Self {
            min_chunk: 64,
            max_chunk: 2048,
            decode_slots: 2,
            enable_splitfuse: false, // ⛔ REQ-SCHED-007: SplitFuse permanently disabled
        }
    }
}

/// Adaptive chunk size policy (REQ-KV-EXT-001).
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveChunkPolicy {
    pub min_chunk: usize,
    pub max_chunk: usize,
}

impl AdaptiveChunkPolicy {
    pub fn new(config: &ChunkedConfig) -> Self {
        Self {
            min_chunk: config.min_chunk.max(1),
            max_chunk: config.max_chunk.max(config.min_chunk),
        }
    }

    pub fn compute(&self, l1_available_ratio: f32, concurrent_reqs: usize, remaining_budget: usize) -> usize {
        let base = if l1_available_ratio < 0.25 {
            self.min_chunk
        } else if l1_available_ratio > 0.75 {
            self.max_chunk
        } else {
            let t = (l1_available_ratio - 0.25) / 0.50;
            let range = self.max_chunk - self.min_chunk;
            self.min_chunk + (range as f32 * t) as usize
        };

        let penalty = (1.0 - 0.1 * concurrent_reqs.saturating_sub(1) as f32).max(0.2);
        let adjusted = (base as f32 * penalty) as usize;

        adjusted.clamp(self.min_chunk, self.max_chunk).min(remaining_budget.max(1))
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

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Scheduler2024Config {
    pub enable_2024_optimizations: bool,
    pub chunked: ChunkedConfig,
    pub swift_kv: SwiftKVConfig,
}

// ----------------------------- Chunked Prefill -----------------------------

#[derive(Debug, Clone)]
pub struct PrefillChunk {
    pub request_id: RequestId,
    pub chunk_idx: usize,
    pub tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ChunkTracker {
    pub total_tokens: usize,
    pub completed_chunks: usize,
    pub pending_tokens: usize,
}

#[derive(Debug)]
pub struct ChunkedState {
    pub config: ChunkedConfig,
    trackers: HashMap<RequestId, ChunkTracker>,
}

impl ChunkedState {
    pub fn new(config: ChunkedConfig) -> Self {
        Self {
            config,
            trackers: HashMap::new(),
        }
    }

    pub fn enqueue(&mut self, request_id: RequestId, tokens: usize) {
        if tokens == 0 || self.trackers.contains_key(&request_id) { return; }
        self.trackers.insert(request_id, ChunkTracker {
            total_tokens: tokens,
            completed_chunks: 0,
            pending_tokens: tokens,
        });
    }

    pub fn on_chunk_finished(&mut self, request_id: RequestId) {
        if let Some(tracker) = self.trackers.get_mut(&request_id) {
            tracker.completed_chunks = tracker.completed_chunks.saturating_add(1);
        }
    }

    pub fn is_request_complete(&self, request_id: &RequestId) -> bool {
        self.trackers.get(request_id).map_or(true, |t| t.pending_tokens == 0)
    }

    pub fn pop_adaptive_chunk(&mut self, request_id: RequestId, l1_available_ratio: f32, concurrent_reqs: usize, remaining_budget: usize) -> Option<PrefillChunk> {
        let tracker = self.trackers.get_mut(&request_id)?;
        if tracker.pending_tokens == 0 { return None; }

        let policy = AdaptiveChunkPolicy::new(&self.config);
        let dynamic_chunk_size = policy.compute(l1_available_ratio, concurrent_reqs, remaining_budget);
        
        // Exact tokens for this chunk
        let tokens_to_run = tracker.pending_tokens.min(dynamic_chunk_size);
        let chunk_idx = tracker.completed_chunks;
        
        tracker.pending_tokens -= tokens_to_run;

        Some(PrefillChunk {
            request_id,
            chunk_idx,
            tokens: tokens_to_run,
        })
    }

    pub fn remove_tracker(&mut self, request_id: &RequestId) {
        self.trackers.remove(request_id);
    }
}

// ------------------------------- SwiftKV ----------------------------------
// Keeping existing working implementation unchanged as it conforms precisely to SIKV/AKV
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistillPagesSummary {
    pub original_pages: usize,
    pub distilled_pages: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DistillResult<E: Element> {
    pub original_pages: Vec<Vec<E>>,
    pub distilled_pages: Vec<Vec<E>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DistillOutcome<E: Element> {
    pub result: DistillResult<E>,
    pub ppl_diff: E,
    pub precision_fallback: bool,
}

#[derive(Debug)]
pub struct SwiftKvState {
    pub config: SwiftKVConfig,
    pub last_result: Option<DistillPagesSummary>,
    pub last_distilled_page: usize,
}

impl SwiftKvState {
    pub fn new(config: SwiftKVConfig) -> Self {
        Self { config, last_result: None, last_distilled_page: 0 }
    }

    pub fn distill_pages(&mut self, page_count: usize) -> DistillPagesSummary {
        if !self.config.enabled || page_count == 0 {
            let res = DistillPagesSummary { original_pages: page_count, distilled_pages: page_count };
            self.last_result = Some(res);
            return res;
        }
        let mut distilled = page_count.div_ceil(self.config.window_size);
        if self.config.enable_across_kv { distilled = distilled.div_ceil(2); }
        let res = DistillPagesSummary { original_pages: page_count, distilled_pages: distilled.max(1) };
        self.last_result = Some(res);
        res
    }

    pub fn distill_cpu<E: Element + Float>(&mut self, pages: &[Vec<E>]) -> DistillOutcome<E> {
        if !self.config.enabled || pages.is_empty() {
            let res = DistillResult { original_pages: pages.to_vec(), distilled_pages: pages.to_vec() };
            return DistillOutcome { result: res, ppl_diff: E::ZERO, precision_fallback: false };
        }
        let mut merged = Vec::new();
        let w = self.config.window_size.max(1);
        for chunk in pages.chunks(w) {
            let mut acc = vec![E::ZERO; chunk[0].len()];
            let mut weight_sum = E::ZERO;
            for page in chunk {
                weight_sum += E::ONE;
                for (dst, src) in acc.iter_mut().zip(page.iter()) { *dst += E::ONE * *src; }
            }
            if weight_sum > E::ZERO {
                for v in acc.iter_mut() { *v = *v / weight_sum; }
            }
            merged.push(acc);
        }
        let mut akv_shared = Vec::new();
        let mut pf = false;
        for i in 0..merged.len() {
            if i > 0 && self.config.enable_across_kv {
                if cosine_similarity(&merged[i - 1], &merged[i]) >= E::from_f32(self.config.similarity_threshold) { continue; }
            }
            akv_shared.push(merged[i].clone());
        }
        let ppl_diff = approx_ppl_delta(pages, &akv_shared);
        if ppl_diff > E::from_f32(self.config.precision_guard) {
            pf = true;
            let res = DistillResult { original_pages: pages.to_vec(), distilled_pages: pages.to_vec() };
            self.last_result = Some(DistillPagesSummary { original_pages: pages.len(), distilled_pages: pages.len() });
            return DistillOutcome { result: res, ppl_diff, precision_fallback: pf };
        }
        let distilled = if akv_shared.is_empty() { pages.to_vec() } else { akv_shared };
        self.last_result = Some(DistillPagesSummary { original_pages: pages.len(), distilled_pages: distilled.len() });
        DistillOutcome { result: DistillResult { original_pages: pages.to_vec(), distilled_pages: distilled }, ppl_diff, precision_fallback: pf }
    }

    pub fn distill_cpu_incremental<E: Element + Float>(&mut self, pages: &[Vec<E>]) -> DistillOutcome<E> {
        if pages.is_empty() {
            return DistillOutcome { result: DistillResult { original_pages: vec![], distilled_pages: vec![] }, ppl_diff: E::ZERO, precision_fallback: false };
        }
        if pages.len() < self.last_distilled_page { self.last_distilled_page = 0; }
        if self.last_distilled_page >= pages.len() { return self.distill_cpu(pages); }
        let overlap = self.config.window_size.min(self.last_distilled_page);
        let start = self.last_distilled_page.saturating_sub(overlap);
        let outcome = self.distill_cpu(&pages[start..]);
        self.last_distilled_page = pages.len();
        outcome
    }

    pub fn reset_distill_boundary(&mut self) { self.last_distilled_page = 0; self.last_result = None; }
}

pub fn cosine_similarity<E: Element>(a: &[E], b: &[E]) -> E {
    let mut dot = E::ZERO; let mut na = E::ZERO; let mut nb = E::ZERO;
    for (x, y) in a.iter().zip(b.iter()) { dot += (*x) * (*y); na += (*x) * (*x); nb += (*y) * (*y); }
    if na == E::ZERO || nb == E::ZERO { return E::ZERO; }
    dot / (na.sqrt() * nb.sqrt())
}

pub fn approx_ppl_delta<E: Element>(original: &[Vec<E>], distilled: &[Vec<E>]) -> E {
    let mut total = E::ZERO; let mut count = 0usize;
    if distilled.is_empty() { return E::ZERO; }
    for (i, page) in original.iter().enumerate() {
        if let Some(dp) = distilled.get(i.min(distilled.len().saturating_sub(1))) {
            let mut l2 = E::ZERO;
            for (x, y) in page.iter().zip(dp.iter()) { l2 += (*x - *y) * (*x - *y); }
            let norm = page.iter().fold(E::ZERO, |acc, v| acc + (*v) * (*v)).max(E::from_f32(1e-6));
            total += (l2 / norm).sqrt(); count += 1;
        }
    }
    if count == 0 { E::ZERO } else { total / E::from_f32(count as f32) }
}

#[derive(Debug)]
pub struct Scheduler2024State {
    pub config: Scheduler2024Config,
    pub chunked: ChunkedState,
    pub swift_kv: SwiftKvState,
}

impl Scheduler2024State {
    pub fn new(config: Scheduler2024Config) -> Self {
        Self { chunked: ChunkedState::new(config.chunked), swift_kv: SwiftKvState::new(config.swift_kv), config }
    }
}
