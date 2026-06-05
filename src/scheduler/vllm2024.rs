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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillChunk {
    pub request_id: RequestId,
    pub chunk_idx: usize,
    pub tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
        self.trackers.get(request_id).is_none_or(|t| t.pending_tokens == 0)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
            if i > 0 && self.config.enable_across_kv
                && cosine_similarity(&merged[i - 1], &merged[i]) >= E::from_f32(self.config.similarity_threshold) { continue; }
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── AdaptiveChunkPolicy ──

    #[test]
    fn adaptive_policy_low_l1_gives_min_chunk() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.1, 1, 10000);
        assert_eq!(chunk, policy.min_chunk);
    }

    #[test]
    fn adaptive_policy_high_l1_gives_max_chunk() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.9, 1, 10000);
        assert_eq!(chunk, policy.max_chunk);
    }

    #[test]
    fn adaptive_policy_mid_l1_between_min_max() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.5, 1, 10000);
        assert!(chunk > policy.min_chunk);
        assert!(chunk < policy.max_chunk);
    }

    #[test]
    fn adaptive_policy_respects_remaining_budget() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.9, 1, 50);
        assert_eq!(chunk, 50);
    }

    #[test]
    fn adaptive_policy_concurrent_penalty() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let solo = policy.compute(0.5, 1, 10000);
        let crowded = policy.compute(0.5, 10, 10000);
        assert!(crowded < solo);
    }

    // ── ChunkedState ──

    #[test]
    fn chunked_enqueue_and_pop() {
        let mut state = ChunkedState::new(ChunkedConfig::default());
        state.enqueue(1, 500);
        let chunk = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(chunk.request_id, 1);
        assert_eq!(chunk.chunk_idx, 0);
        assert!(chunk.tokens > 0);
    }

    #[test]
    fn chunked_enqueue_zero_tokens_ignored() {
        let mut state = ChunkedState::new(ChunkedConfig::default());
        state.enqueue(1, 0);
        assert!(state.pop_adaptive_chunk(1, 0.9, 1, 10000).is_none());
    }

    #[test]
    fn chunked_duplicate_enqueue_ignored() {
        let mut state = ChunkedState::new(ChunkedConfig::default());
        state.enqueue(1, 500);
        state.enqueue(1, 1000); // duplicate ignored
        let chunk = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(chunk.tokens, 500); // original amount
    }

    #[test]
    fn chunked_complete_after_all_tokens() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 500,
            max_chunk: 500,
            decode_slots: 2,
            enable_splitfuse: false,
        });
        state.enqueue(1, 500);
        let chunk = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(chunk.tokens, 500);
        assert!(state.is_request_complete(&1));
    }

    #[test]
    fn chunked_not_complete_with_remaining() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 100,
            max_chunk: 100,
            decode_slots: 2,
            enable_splitfuse: false,
        });
        state.enqueue(1, 500);
        state.pop_adaptive_chunk(1, 0.9, 1, 10000);
        assert!(!state.is_request_complete(&1));
    }

    #[test]
    fn chunked_unknown_request_is_complete() {
        let state = ChunkedState::new(ChunkedConfig::default());
        assert!(state.is_request_complete(&999));
    }

    #[test]
    fn chunked_on_chunk_finished_increments() {
        let mut state = ChunkedState::new(ChunkedConfig::default());
        state.enqueue(1, 1000);
        state.on_chunk_finished(1);
        let chunk = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(chunk.chunk_idx, 1);
    }

    // ── SwiftKvState ──

    #[test]
    fn swiftkv_distill_pages_disabled_passthrough() {
        let mut skv = SwiftKvState::new(SwiftKVConfig::default()); // enabled = false
        let res = skv.distill_pages(10);
        assert_eq!(res.original_pages, 10);
        assert_eq!(res.distilled_pages, 10);
    }

    #[test]
    fn swiftkv_distill_pages_enabled_reduces() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 4,
            ..Default::default()
        });
        let res = skv.distill_pages(16);
        assert_eq!(res.original_pages, 16);
        assert_eq!(res.distilled_pages, 4); // ceil(16/4)
    }

    #[test]
    fn swiftkv_distill_pages_zero_is_passthrough() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            ..Default::default()
        });
        let res = skv.distill_pages(0);
        assert_eq!(res.distilled_pages, 0);
    }

    // ── cosine_similarity ──

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a: Vec<f32> = vec![1.0, 0.0];
        let b: Vec<f32> = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        let a: Vec<f32> = vec![0.0, 0.0];
        let b: Vec<f32> = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ── ChunkedConfig defaults & derives ──

    #[test]
    fn chunked_config_default_values() {
        let cfg = ChunkedConfig::default();
        assert_eq!(cfg.min_chunk, 64);
        assert_eq!(cfg.max_chunk, 2048);
        assert_eq!(cfg.decode_slots, 2);
        assert!(!cfg.enable_splitfuse);
    }

    #[test]
    fn chunked_config_copy_clone_eq() {
        let a = ChunkedConfig::default();
        let b = a; // Copy
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
        let d = ChunkedConfig { min_chunk: 1, ..Default::default() };
        assert_ne!(a, d);
    }

    #[test]
    fn chunked_config_debug_contains_fields() {
        let cfg = ChunkedConfig::default();
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("min_chunk"));
        assert!(dbg.contains("max_chunk"));
        assert!(dbg.contains("enable_splitfuse"));
    }

    // ── AdaptiveChunkPolicy edge cases ──

    #[test]
    fn adaptive_policy_new_clamps_min_to_one() {
        let cfg = ChunkedConfig { min_chunk: 0, max_chunk: 100, ..Default::default() };
        let policy = AdaptiveChunkPolicy::new(&cfg);
        assert_eq!(policy.min_chunk, 1);
    }

    #[test]
    fn adaptive_policy_new_ensures_max_ge_min() {
        let cfg = ChunkedConfig { min_chunk: 200, max_chunk: 50, ..Default::default() };
        let policy = AdaptiveChunkPolicy::new(&cfg);
        assert!(policy.max_chunk >= policy.min_chunk);
        assert_eq!(policy.max_chunk, 200);
    }

    #[test]
    fn adaptive_policy_zero_remaining_budget_returns_min() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.9, 1, 0);
        assert_eq!(chunk, 1); // remaining_budget.max(1) = 1, clamped to min_chunk
    }

    #[test]
    fn adaptive_policy_penalty_clamped_at_0_2() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        // With 20 concurrent reqs: penalty = max(0.2, 1 - 0.1*19) = max(0.2, -0.9) = 0.2
        let chunk = policy.compute(0.9, 20, 10000);
        assert!(chunk >= policy.min_chunk);
        assert!(chunk <= policy.max_chunk);
    }

    // ── SwiftKVConfig defaults & derives ──

    #[test]
    fn swift_kv_config_default_values() {
        let cfg = SwiftKVConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.window_size, 4);
        assert!(!cfg.enable_across_kv);
        assert!((cfg.similarity_threshold - 0.9).abs() < 1e-6);
        assert!((cfg.precision_guard - 0.1).abs() < 1e-6);
    }

    #[test]
    fn swift_kv_config_clone_eq() {
        let a = SwiftKVConfig::default();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn scheduler_2024_config_default_nests_correctly() {
        let cfg = Scheduler2024Config::default();
        assert!(!cfg.enable_2024_optimizations);
        assert_eq!(cfg.chunked, ChunkedConfig::default());
        assert_eq!(cfg.swift_kv, SwiftKVConfig::default());
    }

    // ── DistillPagesSummary derives ──

    #[test]
    fn distill_pages_summary_clone_eq() {
        let a = DistillPagesSummary { original_pages: 10, distilled_pages: 5 };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── approx_ppl_delta ──

    #[test]
    fn approx_ppl_delta_empty_distilled_returns_zero() {
        let original: Vec<Vec<f32>> = vec![vec![1.0, 2.0]];
        let distilled: Vec<Vec<f32>> = vec![];
        let delta = approx_ppl_delta(&original, &distilled);
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn approx_ppl_delta_identical_pages_returns_zero() {
        let pages: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let delta = approx_ppl_delta(&pages, &pages);
        assert!(delta.abs() < 1e-5);
    }

    // ── SwiftKvState::distill_cpu ──

    #[test]
    fn distill_cpu_disabled_returns_passthrough() {
        let mut skv = SwiftKvState::new(SwiftKVConfig::default()); // enabled = false
        let pages: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let outcome = skv.distill_cpu(&pages);
        assert_eq!(outcome.result.original_pages, outcome.result.distilled_pages);
        assert!(!outcome.precision_fallback);
        assert_eq!(outcome.ppl_diff, 0.0);
    }

    #[test]
    fn distill_cpu_empty_pages_returns_empty() {
        let mut skv = SwiftKvState::new(SwiftKVConfig { enabled: true, ..Default::default() });
        let pages: Vec<Vec<f32>> = vec![];
        let outcome = skv.distill_cpu(&pages);
        assert!(outcome.result.original_pages.is_empty());
        assert!(outcome.result.distilled_pages.is_empty());
        assert!(!outcome.precision_fallback);
    }

    #[test]
    fn distill_cpu_enabled_merges_by_window() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0, // high guard so no fallback
        });
        // 4 pages with window=2 -> 2 merged pages
        let pages: Vec<Vec<f32>> = vec![
            vec![2.0, 4.0],
            vec![4.0, 8.0],
            vec![6.0, 12.0],
            vec![8.0, 16.0],
        ];
        let outcome = skv.distill_cpu(&pages);
        assert_eq!(outcome.result.distilled_pages.len(), 2);
        // First merged page: average of [2,4] and [4,8] = [3,6]
        let first = &outcome.result.distilled_pages[0];
        assert!((first[0] - 3.0).abs() < 1e-4);
        assert!((first[1] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn distill_cpu_precision_fallback_triggers() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 0.0001, // very low guard -> fallback triggers
        });
        let pages: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0], // very different from first
        ];
        let outcome = skv.distill_cpu(&pages);
        assert!(outcome.precision_fallback);
        // fallback returns original pages unchanged
        assert_eq!(outcome.result.distilled_pages.len(), pages.len());
    }

    // ── SwiftKvState::distill_cpu_incremental ──

    #[test]
    fn distill_cpu_incremental_empty_pages() {
        let mut skv = SwiftKvState::new(SwiftKVConfig { enabled: true, ..Default::default() });
        let outcome = skv.distill_cpu_incremental::<f32>(&[]);
        assert!(outcome.result.original_pages.is_empty());
        assert!(outcome.result.distilled_pages.is_empty());
    }

    #[test]
    fn distill_cpu_incremental_advances_boundary() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        let pages: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0]];
        skv.distill_cpu_incremental(&pages);
        assert_eq!(skv.last_distilled_page, 2);

        // Second call with more pages uses incremental overlap
        let more_pages: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let outcome = skv.distill_cpu_incremental(&more_pages);
        assert_eq!(skv.last_distilled_page, 4);
        assert!(!outcome.result.distilled_pages.is_empty());
    }

    // ── SwiftKvState::reset_distill_boundary ──

    #[test]
    fn reset_distill_boundary_clears_state() {
        let mut skv = SwiftKvState::new(SwiftKVConfig { enabled: true, ..Default::default() });
        let pages: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0]];
        skv.distill_cpu_incremental(&pages);
        assert!(skv.last_result.is_some());
        assert_eq!(skv.last_distilled_page, 2);

        skv.reset_distill_boundary();
        assert!(skv.last_result.is_none());
        assert_eq!(skv.last_distilled_page, 0);
    }

    // ── ChunkedState::remove_tracker ──

    #[test]
    fn chunked_remove_tracker_clears_entry() {
        let mut state = ChunkedState::new(ChunkedConfig::default());
        state.enqueue(42, 500);
        assert!(!state.is_request_complete(&42)); // has pending tokens

        state.remove_tracker(&42);
        // After removal, is_request_complete returns true (tracker absent)
        assert!(state.is_request_complete(&42));
    }

    // ── Scheduler2024State integration ──

    #[test]
    fn scheduler_2024_state_new_initializes_substates() {
        let cfg = Scheduler2024Config::default();
        let state = Scheduler2024State::new(cfg);
        assert_eq!(state.chunked.config, ChunkedConfig::default());
        assert_eq!(state.swift_kv.config, SwiftKVConfig::default());
        assert!(state.swift_kv.last_result.is_none());
    }

    // ── cosine_similarity different lengths ──

    #[test]
    fn cosine_similarity_unequal_lengths_truncates() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![1.0, 2.0]; // shorter
        let sim = cosine_similarity(&a, &b);
        // zip truncates to min length, so compares [1,2] vs [1,2] => cos = 1.0
        assert!((sim - 1.0).abs() < 1e-5);
    }

    // ── SwiftKV distill_pages with across_kv ──

    #[test]
    fn swiftkv_distill_pages_across_kv_halves() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 4,
            enable_across_kv: true,
            ..Default::default()
        });
        let res = skv.distill_pages(16);
        // window: ceil(16/4) = 4, then across_kv: ceil(4/2) = 2
        assert_eq!(res.distilled_pages, 2);
        assert!(skv.last_result.is_some());
    }

    // ── ChunkedConfig Hash ──

    #[test]
    fn chunked_config_hash_equal_inputs_equal_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = ChunkedConfig::default();
        let b = ChunkedConfig::default();
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn chunked_config_hash_different_inputs_different_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = ChunkedConfig::default();
        let b = ChunkedConfig { min_chunk: 1, ..Default::default() };
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish());
    }

    // ── AdaptiveChunkPolicy derives ──

    #[test]
    fn adaptive_policy_copy_clone_eq() {
        let a = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let b = a; // Copy
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn adaptive_policy_debug_contains_fields() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let dbg = format!("{:?}", policy);
        assert!(dbg.contains("min_chunk"));
        assert!(dbg.contains("max_chunk"));
    }

    #[test]
    fn adaptive_policy_equal_min_max_boundary() {
        let cfg = ChunkedConfig { min_chunk: 100, max_chunk: 100, ..Default::default() };
        let policy = AdaptiveChunkPolicy::new(&cfg);
        let chunk_low = policy.compute(0.0, 1, 10000);
        let chunk_high = policy.compute(1.0, 1, 10000);
        assert_eq!(chunk_low, 100);
        assert_eq!(chunk_high, 100);
    }

    #[test]
    fn adaptive_policy_compute_concurrent_reqs_zero() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.5, 0, 10000);
        assert!(chunk >= policy.min_chunk);
        assert!(chunk <= policy.max_chunk);
    }

    // ── PrefillChunk derives ──

    #[test]
    fn prefill_chunk_clone_eq() {
        let a = PrefillChunk { request_id: 7, chunk_idx: 2, tokens: 128 };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn prefill_chunk_ne_different_fields() {
        let a = PrefillChunk { request_id: 1, chunk_idx: 0, tokens: 64 };
        let b = PrefillChunk { request_id: 1, chunk_idx: 1, tokens: 64 };
        assert_ne!(a, b);
    }

    #[test]
    fn prefill_chunk_debug_format() {
        let chunk = PrefillChunk { request_id: 42, chunk_idx: 0, tokens: 256 };
        let dbg = format!("{:?}", chunk);
        assert!(dbg.contains("42"));
        assert!(dbg.contains("256"));
    }

    // ── ChunkTracker derives ──

    #[test]
    fn chunk_tracker_clone_eq() {
        let a = ChunkTracker { total_tokens: 1000, completed_chunks: 3, pending_tokens: 200 };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn chunk_tracker_ne_different_pending() {
        let a = ChunkTracker { total_tokens: 500, completed_chunks: 0, pending_tokens: 500 };
        let b = ChunkTracker { total_tokens: 500, completed_chunks: 0, pending_tokens: 300 };
        assert_ne!(a, b);
    }

    #[test]
    fn chunk_tracker_debug_format() {
        let tracker = ChunkTracker { total_tokens: 100, completed_chunks: 1, pending_tokens: 50 };
        let dbg = format!("{:?}", tracker);
        assert!(dbg.contains("total_tokens"));
        assert!(dbg.contains("pending_tokens"));
    }

    // ── ChunkedState multi-chunk sequencing ──

    #[test]
    fn chunked_multi_chunk_sequential_indices() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 100,
            max_chunk: 100,
            decode_slots: 2,
            enable_splitfuse: false,
        });
        state.enqueue(1, 350);

        let c0 = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(c0.chunk_idx, 0);
        assert_eq!(c0.tokens, 100);

        state.on_chunk_finished(1);
        let c1 = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(c1.chunk_idx, 1);
        assert_eq!(c1.tokens, 100);

        state.on_chunk_finished(1);
        let c2 = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(c2.chunk_idx, 2);
        assert_eq!(c2.tokens, 100);

        state.on_chunk_finished(1);
        let c3 = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(c3.chunk_idx, 3);
        assert_eq!(c3.tokens, 50);

        assert!(state.pop_adaptive_chunk(1, 0.9, 1, 10000).is_none());
        assert!(state.is_request_complete(&1));
    }

    #[test]
    fn chunked_multiple_requests_independent() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 200,
            max_chunk: 200,
            decode_slots: 4,
            enable_splitfuse: false,
        });
        state.enqueue(10, 300);
        state.enqueue(20, 500);

        let c_a = state.pop_adaptive_chunk(10, 0.9, 2, 10000).unwrap();
        assert_eq!(c_a.request_id, 10);
        assert_eq!(c_a.tokens, 200);

        let c_b = state.pop_adaptive_chunk(20, 0.9, 2, 10000).unwrap();
        assert_eq!(c_b.request_id, 20);
        assert_eq!(c_b.tokens, 200);

        assert!(!state.is_request_complete(&10));
        assert!(!state.is_request_complete(&20));
    }

    #[test]
    fn chunked_on_chunk_finished_unknown_request_is_noop() {
        let mut state = ChunkedState::new(ChunkedConfig::default());
        state.on_chunk_finished(999); // unknown — should not panic
        assert!(state.is_request_complete(&999));
    }

    #[test]
    fn chunked_pop_unknown_request_returns_none() {
        let mut state = ChunkedState::new(ChunkedConfig::default());
        assert!(state.pop_adaptive_chunk(999, 0.9, 1, 10000).is_none());
    }

    // ── DistillPagesSummary derives ──

    #[test]
    fn distill_pages_summary_copy() {
        let a = DistillPagesSummary { original_pages: 10, distilled_pages: 5 };
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn distill_pages_summary_hash_equal() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = DistillPagesSummary { original_pages: 8, distilled_pages: 4 };
        let b = DistillPagesSummary { original_pages: 8, distilled_pages: 4 };
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn distill_pages_summary_debug_format() {
        let s = DistillPagesSummary { original_pages: 16, distilled_pages: 2 };
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("original_pages"));
        assert!(dbg.contains("distilled_pages"));
    }

    // ── SwiftKVConfig derives ──

    #[test]
    fn swift_kv_config_copy() {
        let a = SwiftKVConfig::default();
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn swift_kv_config_debug_format() {
        let cfg = SwiftKVConfig::default();
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("enabled"));
        assert!(dbg.contains("window_size"));
        assert!(dbg.contains("similarity_threshold"));
    }

    // ── Scheduler2024Config derives ──

    #[test]
    fn scheduler_2024_config_clone_eq() {
        let a = Scheduler2024Config::default();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn scheduler_2024_config_copy() {
        let a = Scheduler2024Config::default();
        let b = a; // Copy (all fields are Copy)
        assert_eq!(a, b);
    }

    #[test]
    fn scheduler_2024_config_debug_format() {
        let cfg = Scheduler2024Config::default();
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("enable_2024_optimizations"));
        assert!(dbg.contains("chunked"));
        assert!(dbg.contains("swift_kv"));
    }

    #[test]
    fn scheduler_2024_config_custom_values() {
        let cfg = Scheduler2024Config {
            enable_2024_optimizations: true,
            chunked: ChunkedConfig { min_chunk: 32, max_chunk: 1024, decode_slots: 4, enable_splitfuse: false },
            swift_kv: SwiftKVConfig { enabled: true, window_size: 8, ..Default::default() },
        };
        assert!(cfg.enable_2024_optimizations);
        assert_eq!(cfg.chunked.min_chunk, 32);
        assert_eq!(cfg.swift_kv.window_size, 8);
    }

    // ── cosine_similarity additional cases ──

    #[test]
    fn cosine_similarity_negative_vectors() {
        let a: Vec<f32> = vec![-1.0, -2.0];
        let b: Vec<f32> = vec![-1.0, -2.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_opposite_directions() {
        let a: Vec<f32> = vec![1.0, 0.0];
        let b: Vec<f32> = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_both_zero_returns_zero() {
        let a: Vec<f32> = vec![0.0, 0.0];
        let b: Vec<f32> = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ── approx_ppl_delta edge cases ──

    #[test]
    fn approx_ppl_delta_single_element_pages() {
        let original: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0]];
        let distilled: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0]];
        let delta = approx_ppl_delta(&original, &distilled);
        assert!(delta.abs() < 1e-5);
    }

    #[test]
    fn approx_ppl_delta_empty_original_returns_zero() {
        let original: Vec<Vec<f32>> = vec![];
        let distilled: Vec<Vec<f32>> = vec![vec![1.0]];
        let delta = approx_ppl_delta(&original, &distilled);
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn approx_ppl_delta_differing_pages_positive_delta() {
        let original: Vec<Vec<f32>> = vec![vec![1.0, 0.0]];
        let distilled: Vec<Vec<f32>> = vec![vec![0.0, 1.0]];
        let delta = approx_ppl_delta(&original, &distilled);
        assert!(delta > 0.0);
    }

    // ── distill_pages edge cases ──

    #[test]
    fn swiftkv_distill_pages_window_size_1() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 1,
            ..Default::default()
        });
        let res = skv.distill_pages(8);
        assert_eq!(res.distilled_pages, 8); // ceil(8/1) = 8, no reduction
    }

    #[test]
    fn swiftkv_distill_pages_non_divisible_count() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 4,
            ..Default::default()
        });
        let res = skv.distill_pages(7);
        assert_eq!(res.distilled_pages, 2); // ceil(7/4) = 2
    }

    #[test]
    fn swiftkv_distill_pages_minimum_one_distilled() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 100,
            ..Default::default()
        });
        let res = skv.distill_pages(1);
        assert_eq!(res.distilled_pages, 1); // max(1, ceil(1/100)) = 1
    }

    #[test]
    fn swiftkv_distill_pages_across_kv_non_divisible() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 4,
            enable_across_kv: true,
            ..Default::default()
        });
        let res = skv.distill_pages(9);
        // ceil(9/4) = 3, then ceil(3/2) = 2
        assert_eq!(res.distilled_pages, 2);
    }

    // ── SwiftKvState::distill_cpu advanced cases ──

    #[test]
    fn distill_cpu_single_page_returns_merged() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 4,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        let pages: Vec<Vec<f32>> = vec![vec![3.0, 6.0]];
        let outcome = skv.distill_cpu(&pages);
        // 1 page, window=4 -> 1 merged page (the single page itself averaged)
        assert_eq!(outcome.result.distilled_pages.len(), 1);
        assert!(!outcome.precision_fallback);
    }

    #[test]
    fn distill_cpu_across_kv_skips_similar() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: true,
            similarity_threshold: 0.5, // low threshold -> most pages skipped
            precision_guard: 1.0,
        });
        // 4 pages -> window=2 -> 2 merged pages: [1.5, 3.5] and [5.5, 7.5]
        // cosine_sim of [1.5, 3.5] and [5.5, 7.5] will be very high (both proportional)
        let pages: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0],
            vec![2.0, 5.0],
            vec![5.0, 7.0],
            vec![6.0, 8.0],
        ];
        let outcome = skv.distill_cpu(&pages);
        assert!(!outcome.precision_fallback);
        // across_kv should skip the second page if similar enough to the first
        assert!(outcome.result.distilled_pages.len() <= 2);
    }

    // ── SwiftKvState::distill_cpu_incremental edge cases ──

    #[test]
    fn distill_cpu_incremental_resets_when_boundary_exceeds() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        // First: process 4 pages, boundary = 4
        let pages_a: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        skv.distill_cpu_incremental(&pages_a);
        assert_eq!(skv.last_distilled_page, 4);

        // Now supply fewer pages (2) — last_distilled_page > pages.len(), so boundary resets
        let pages_b: Vec<Vec<f32>> = vec![vec![10.0], vec![20.0]];
        let outcome = skv.distill_cpu_incremental(&pages_b);
        assert_eq!(skv.last_distilled_page, 2);
        assert_eq!(outcome.result.original_pages.len(), 2);
    }

    // ── DistillResult and DistillOutcome derives ──

    #[test]
    fn distill_result_clone_eq() {
        let a = DistillResult {
            original_pages: vec![vec![1.0_f32, 2.0]],
            distilled_pages: vec![vec![1.0_f32, 2.0]],
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn distill_outcome_clone_eq() {
        let a = DistillOutcome {
            result: DistillResult {
                original_pages: vec![vec![1.0_f32]],
                distilled_pages: vec![vec![1.0_f32]],
            },
            ppl_diff: 0.0_f32,
            precision_fallback: false,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── ChunkedConfig custom construction ──

    #[test]
    fn chunked_config_custom_values() {
        let cfg = ChunkedConfig {
            min_chunk: 1,
            max_chunk: 8192,
            decode_slots: 8,
            enable_splitfuse: false,
        };
        assert_eq!(cfg.min_chunk, 1);
        assert_eq!(cfg.max_chunk, 8192);
        assert_eq!(cfg.decode_slots, 8);
    }

    // ── SwiftKvState last_result tracking ──

    #[test]
    fn swiftkv_distill_pages_stores_last_result() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 4,
            ..Default::default()
        });
        assert!(skv.last_result.is_none());
        skv.distill_pages(8);
        let summary = skv.last_result.unwrap();
        assert_eq!(summary.original_pages, 8);
        assert_eq!(summary.distilled_pages, 2);
    }

    // ── Scheduler2024State integration: config propagation ──

    #[test]
    fn scheduler_2024_state_custom_config_propagates() {
        let cfg = Scheduler2024Config {
            enable_2024_optimizations: true,
            chunked: ChunkedConfig { min_chunk: 32, max_chunk: 512, decode_slots: 1, enable_splitfuse: false },
            swift_kv: SwiftKVConfig { enabled: true, window_size: 2, ..Default::default() },
        };
        let state = Scheduler2024State::new(cfg);
        assert_eq!(state.chunked.config.min_chunk, 32);
        assert_eq!(state.chunked.config.max_chunk, 512);
        assert!(state.swift_kv.config.enabled);
    }

    // ── AdaptiveChunkPolicy boundary: l1 at exact thresholds ──

    #[test]
    fn adaptive_policy_exact_low_threshold_boundary() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.25, 1, 10000);
        // 0.25 is NOT < 0.25, so it falls to the else branch (mid-range)
        assert!(chunk >= policy.min_chunk);
        assert!(chunk <= policy.max_chunk);
    }

    #[test]
    fn adaptive_policy_exact_high_threshold_boundary() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.75, 1, 10000);
        // 0.75 is NOT > 0.75, so it falls to the else branch (mid-range)
        assert!(chunk >= policy.min_chunk);
        assert!(chunk <= policy.max_chunk);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // NEW TESTS (~40)
    // ─────────────────────────────────────────────────────────────────────────

    // ── AdaptiveChunkPolicy: monotonicity with increasing l1 ratio ──

    #[test]
    fn adaptive_policy_monotonic_increasing_l1() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk_low = policy.compute(0.1, 1, 10000);
        let chunk_mid = policy.compute(0.5, 1, 10000);
        let chunk_high = policy.compute(0.9, 1, 10000);
        assert!(chunk_low <= chunk_mid);
        assert!(chunk_mid <= chunk_high);
    }

    // ── AdaptiveChunkPolicy: l1 exactly 0.0 and 1.0 ──

    #[test]
    fn adaptive_policy_l1_zero_gives_min() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.0, 1, 10000);
        assert_eq!(chunk, policy.min_chunk);
    }

    #[test]
    fn adaptive_policy_l1_one_gives_max() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(1.0, 1, 10000);
        assert_eq!(chunk, policy.max_chunk);
    }

    // ── AdaptiveChunkPolicy: remaining_budget smaller than min_chunk ──

    #[test]
    fn adaptive_policy_budget_below_min_clamped_to_budget() {
        let cfg = ChunkedConfig { min_chunk: 500, max_chunk: 2000, ..Default::default() };
        let policy = AdaptiveChunkPolicy::new(&cfg);
        let chunk = policy.compute(0.9, 1, 10);
        // remaining_budget.max(1) = 10, then .min(10) = 10, but clamp(min=500, max=2000) = 500
        // Wait: clamp(min, max).min(remaining_budget.max(1))
        // adjusted.clamp(500, 2000) could be 500+.min(10) = 10, but 10 < min_chunk
        // Actually: clamp first, then .min(remaining.max(1))
        // adjusted is likely >10 for high l1, clamp makes it min 500, .min(10) = 10
        assert!(chunk <= 10);
        assert!(chunk >= 1);
    }

    // ── AdaptiveChunkPolicy: concurrent_reqs = 1 no penalty ──

    #[test]
    fn adaptive_policy_single_req_no_penalty() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk_solo = policy.compute(0.5, 1, 10000);
        // penalty = 1.0 - 0.1 * saturating_sub(1) = 1.0 - 0 = 1.0 (no penalty)
        let chunk_zero = policy.compute(0.5, 0, 10000);
        // penalty = 1.0 - 0.1 * saturating_sub(0-1)=1.0, also 1.0
        assert_eq!(chunk_solo, chunk_zero);
    }

    // ── AdaptiveChunkPolicy: result always within bounds ──

    #[test]
    fn adaptive_policy_result_always_within_bounds() {
        let cfg = ChunkedConfig { min_chunk: 50, max_chunk: 300, ..Default::default() };
        let policy = AdaptiveChunkPolicy::new(&cfg);
        for l1 in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            for reqs in [0, 1, 5, 20] {
                let chunk = policy.compute(l1, reqs, 10000);
                assert!(chunk >= policy.min_chunk, "l1={}, reqs={}", l1, reqs);
                assert!(chunk <= policy.max_chunk, "l1={}, reqs={}", l1, reqs);
            }
        }
    }

    // ── ChunkedState: pop returns None when pending_tokens exhausted but tracker exists ──

    #[test]
    fn chunked_pop_returns_none_after_all_tokens_consumed() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 200,
            max_chunk: 200,
            decode_slots: 2,
            enable_splitfuse: false,
        });
        state.enqueue(1, 200);
        let _c = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        // pending_tokens is now 0
        assert!(state.pop_adaptive_chunk(1, 0.9, 1, 10000).is_none());
    }

    // ── ChunkedState: total tokens split across chunks sum correctly ──

    #[test]
    fn chunked_tokens_sum_equals_total() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 7,
            max_chunk: 7,
            decode_slots: 2,
            enable_splitfuse: false,
        });
        state.enqueue(1, 23);
        let mut total = 0usize;
        while let Some(c) = state.pop_adaptive_chunk(1, 0.9, 1, 10000) {
            total += c.tokens;
            state.on_chunk_finished(1);
        }
        assert_eq!(total, 23);
    }

    // ── ChunkedState: remove_tracker on unknown request (no panic) ──

    #[test]
    fn chunked_remove_tracker_unknown_is_noop() {
        let mut state = ChunkedState::new(ChunkedConfig::default());
        state.remove_tracker(&999); // should not panic
        assert!(state.is_request_complete(&999));
    }

    // ── ChunkedState: re-enqueue after removal works ──

    #[test]
    fn chunked_reenqueue_after_removal() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 500,
            max_chunk: 500,
            decode_slots: 2,
            enable_splitfuse: false,
        });
        state.enqueue(1, 500);
        state.remove_tracker(&1);
        // Re-enqueue should work since tracker was removed
        state.enqueue(1, 300);
        let chunk = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(chunk.tokens, 300);
    }

    // ── ChunkedState: many requests enqueued independently ──

    #[test]
    fn chunked_many_requests_independent_tracking() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 100,
            max_chunk: 100,
            decode_slots: 4,
            enable_splitfuse: false,
        });
        for id in 0..20u64 {
            state.enqueue(id, 250);
        }
        for id in 0..20u64 {
            let c = state.pop_adaptive_chunk(id, 0.9, 20, 10000).unwrap();
            assert_eq!(c.request_id, id);
            assert!(c.tokens > 0);
        }
    }

    // ── ChunkedState: on_chunk_finished saturating_add ──

    #[test]
    fn chunked_on_chunk_finished_multiple_increments() {
        let mut state = ChunkedState::new(ChunkedConfig::default());
        state.enqueue(1, 500);
        state.on_chunk_finished(1);
        state.on_chunk_finished(1);
        state.on_chunk_finished(1);
        let c = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(c.chunk_idx, 3);
    }

    // ── cosine_similarity: single element vectors ──

    #[test]
    fn cosine_similarity_single_element() {
        let a: Vec<f32> = vec![3.0];
        let b: Vec<f32> = vec![4.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    // ── cosine_similarity: b longer than a truncates ──

    #[test]
    fn cosine_similarity_b_longer_truncates() {
        let a: Vec<f32> = vec![1.0, 2.0];
        let b: Vec<f32> = vec![1.0, 2.0, 999.0]; // extra element ignored
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    // ── cosine_similarity: empty slices return zero ──

    #[test]
    fn cosine_similarity_empty_slices() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ── cosine_similarity: large random-like vectors bounded ──

    #[test]
    fn cosine_similarity_output_bounded_minus1_to_1() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim >= -1.0 - 1e-5);
        assert!(sim <= 1.0 + 1e-5);
    }

    // ── approx_ppl_delta: distilled shorter than original ──

    #[test]
    fn approx_ppl_delta_distilled_shorter_reuses_last() {
        let original: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let distilled: Vec<Vec<f32>> = vec![vec![1.0]]; // shorter, last page reused
        let delta = approx_ppl_delta(&original, &distilled);
        assert!(delta >= 0.0);
    }

    // ── approx_ppl_delta: distilled longer than original ──

    #[test]
    fn approx_ppl_delta_distilled_longer_clamps_index() {
        let original: Vec<Vec<f32>> = vec![vec![1.0]];
        let distilled: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let delta = approx_ppl_delta(&original, &distilled);
        // Only index 0 matches; extra distilled pages are ignored by enumerate
        assert!(delta.abs() < 1e-5);
    }

    // ── approx_ppl_delta: pages with different element counts ──

    #[test]
    fn approx_ppl_delta_mismatched_element_counts() {
        let original: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]];
        let distilled: Vec<Vec<f32>> = vec![vec![1.0, 2.0]]; // zip truncates
        let delta = approx_ppl_delta(&original, &distilled);
        assert!(delta.abs() < 1e-5);
    }

    // ── distill_cpu: across_kv with all identical pages ──

    #[test]
    fn distill_cpu_across_kv_identical_pages_skips_all_but_first() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 1,
            enable_across_kv: true,
            similarity_threshold: 0.5, // low threshold -> high similarity match
            precision_guard: 1.0,
        });
        // All pages identical, window=1 -> 4 merged, then across_kv skips similar ones
        let pages: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];
        let outcome = skv.distill_cpu(&pages);
        assert!(!outcome.precision_fallback);
        // With cosine_sim == 1.0 >= 0.5, all but first merged page should be skipped
        assert!(outcome.result.distilled_pages.len() == 1);
    }

    // ── distill_cpu: across_kv disabled no skipping ──

    #[test]
    fn distill_cpu_across_kv_disabled_no_similarity_check() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 1,
            enable_across_kv: false,
            similarity_threshold: 0.5,
            precision_guard: 1.0,
        });
        let pages: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];
        let outcome = skv.distill_cpu(&pages);
        assert_eq!(outcome.result.distilled_pages.len(), 4);
    }

    // ── distill_cpu: window_size larger than page count ──

    #[test]
    fn distill_cpu_window_larger_than_pages() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 100,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        let pages: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let outcome = skv.distill_cpu(&pages);
        // All pages in one chunk -> 1 merged page
        assert_eq!(outcome.result.distilled_pages.len(), 1);
    }

    // ── distill_cpu: precision_guard zero triggers fallback ──

    #[test]
    fn distill_cpu_zero_precision_guard_triggers_fallback() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 0.0, // zero guard -> any non-zero ppl_diff triggers fallback
        });
        let pages: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let outcome = skv.distill_cpu(&pages);
        assert!(outcome.precision_fallback);
        assert_eq!(outcome.result.distilled_pages.len(), 2); // original returned
    }

    // ── distill_cpu: last_result updated after distill_cpu ──

    #[test]
    fn distill_cpu_updates_last_result() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        assert!(skv.last_result.is_none());
        let pages: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0]];
        skv.distill_cpu(&pages);
        assert!(skv.last_result.is_some());
        let summary = skv.last_result.unwrap();
        assert_eq!(summary.original_pages, 2);
    }

    // ── distill_cpu: single-element pages ──

    #[test]
    fn distill_cpu_single_element_pages() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 3,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        let pages: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let outcome = skv.distill_cpu(&pages);
        // window=3 -> 1 merged page: average of [1,2,3] = 2.0
        assert_eq!(outcome.result.distilled_pages.len(), 1);
        let merged = &outcome.result.distilled_pages[0];
        assert!((merged[0] - 2.0).abs() < 1e-4);
    }

    // ── distill_cpu_incremental: fresh state processes all pages ──

    #[test]
    fn distill_cpu_incremental_fresh_processes_all() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        assert_eq!(skv.last_distilled_page, 0);
        let pages: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let outcome = skv.distill_cpu_incremental(&pages);
        assert_eq!(skv.last_distilled_page, 4);
        assert_eq!(outcome.result.original_pages.len(), 4);
    }

    // ── distill_cpu_incremental: overlap is window_size when boundary is large ──

    #[test]
    fn distill_cpu_incremental_overlap_equals_window_size() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 3,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        // First batch: 6 pages
        let batch1: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0], vec![6.0]];
        skv.distill_cpu_incremental(&batch1);
        assert_eq!(skv.last_distilled_page, 6);

        // Second batch: 8 pages. overlap = min(3, 6) = 3, start = 6 - 3 = 3
        let batch2: Vec<Vec<f32>> = vec![
            vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0], vec![6.0], vec![7.0], vec![8.0],
        ];
        let outcome = skv.distill_cpu_incremental(&batch2);
        assert_eq!(skv.last_distilled_page, 8);
        // Processed from index 3..8 (5 pages), merged by window=3 -> 2 merged pages
        assert!(!outcome.result.distilled_pages.is_empty());
    }

    // ── distill_cpu_incremental: multiple sequential calls ──

    #[test]
    fn distill_cpu_incremental_sequential_boundary_growth() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        skv.distill_cpu_incremental(&[vec![1.0], vec![2.0]]);
        assert_eq!(skv.last_distilled_page, 2);

        skv.distill_cpu_incremental(&[vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(skv.last_distilled_page, 3);

        skv.distill_cpu_incremental(&[vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]);
        assert_eq!(skv.last_distilled_page, 5);
    }

    // ── reset_distill_boundary: multiple resets ──

    #[test]
    fn reset_distill_boundary_multiple_resets() {
        let mut skv = SwiftKvState::new(SwiftKVConfig { enabled: true, ..Default::default() });
        skv.distill_cpu_incremental(&[vec![1.0], vec![2.0]]);
        skv.reset_distill_boundary();
        assert_eq!(skv.last_distilled_page, 0);
        assert!(skv.last_result.is_none());

        skv.distill_cpu_incremental(&[vec![3.0], vec![4.0], vec![5.0]]);
        assert_eq!(skv.last_distilled_page, 3);
        skv.reset_distill_boundary();
        assert_eq!(skv.last_distilled_page, 0);
        assert!(skv.last_result.is_none());
    }

    // ── DistillPagesSummary: ne ──

    #[test]
    fn distill_pages_summary_ne() {
        let a = DistillPagesSummary { original_pages: 10, distilled_pages: 5 };
        let b = DistillPagesSummary { original_pages: 10, distilled_pages: 10 };
        assert_ne!(a, b);
    }

    // ── DistillPagesSummary: hash different values produce different hashes ──

    #[test]
    fn distill_pages_summary_hash_different() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = DistillPagesSummary { original_pages: 10, distilled_pages: 5 };
        let b = DistillPagesSummary { original_pages: 10, distilled_pages: 10 };
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish());
    }

    // ── PrefillChunk: PartialEq symmetry ──

    #[test]
    fn prefill_chunk_eq_symmetry() {
        let a = PrefillChunk { request_id: 1, chunk_idx: 0, tokens: 100 };
        let b = PrefillChunk { request_id: 1, chunk_idx: 0, tokens: 100 };
        assert!(a == b);
        assert!(b == a);
    }

    // ── PrefillChunk: all fields equal ──

    #[test]
    fn prefill_chunk_all_fields_equal() {
        let a = PrefillChunk { request_id: 5, chunk_idx: 3, tokens: 200 };
        let b = PrefillChunk { request_id: 5, chunk_idx: 3, tokens: 200 };
        assert_eq!(a, b);
    }

    // ── ChunkTracker: all fields zero ──

    #[test]
    fn chunk_tracker_zero_fields() {
        let tracker = ChunkTracker { total_tokens: 0, completed_chunks: 0, pending_tokens: 0 };
        assert_eq!(tracker.total_tokens, 0);
        assert_eq!(tracker.completed_chunks, 0);
        assert_eq!(tracker.pending_tokens, 0);
    }

    // ── ChunkTracker: ne across all fields ──

    #[test]
    fn chunk_tracker_ne_total_tokens() {
        let a = ChunkTracker { total_tokens: 100, completed_chunks: 0, pending_tokens: 100 };
        let b = ChunkTracker { total_tokens: 200, completed_chunks: 0, pending_tokens: 100 };
        assert_ne!(a, b);
    }

    // ── Scheduler2024Config: ne between different configs ──

    #[test]
    fn scheduler_2024_config_ne() {
        let a = Scheduler2024Config::default();
        let b = Scheduler2024Config {
            enable_2024_optimizations: true,
            ..Default::default()
        };
        assert_ne!(a, b);
    }

    // ── SwiftKVConfig: ne between different configs ──

    #[test]
    fn swift_kv_config_ne() {
        let a = SwiftKVConfig::default();
        let b = SwiftKVConfig { enabled: true, ..Default::default() };
        assert_ne!(a, b);
    }

    // ── SwiftKVConfig: ne window_size ──

    #[test]
    fn swift_kv_config_ne_window_size() {
        let a = SwiftKVConfig::default();
        let b = SwiftKVConfig { window_size: 8, ..Default::default() };
        assert_ne!(a, b);
    }

    // ── SwiftKvState: new initializes last_result None ──

    #[test]
    fn swift_kv_state_new_initial_state() {
        let skv = SwiftKvState::new(SwiftKVConfig::default());
        assert!(skv.last_result.is_none());
        assert_eq!(skv.last_distilled_page, 0);
    }

    // ── ChunkedState: new has no trackers ──

    #[test]
    fn chunked_state_new_empty() {
        let state = ChunkedState::new(ChunkedConfig::default());
        // Unknown request is "complete" because tracker doesn't exist
        assert!(state.is_request_complete(&1));
    }

    // ── distill_pages: window_size > page_count ──

    #[test]
    fn swiftkv_distill_pages_window_exceeds_count() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 100,
            ..Default::default()
        });
        let res = skv.distill_pages(3);
        assert_eq!(res.distilled_pages, 1); // ceil(3/100) = 1, max(1, 1) = 1
    }

    // ── distill_pages: large page count ──

    #[test]
    fn swiftkv_distill_pages_large_count() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 4,
            ..Default::default()
        });
        let res = skv.distill_pages(1000);
        assert_eq!(res.distilled_pages, 250); // ceil(1000/4) = 250
    }

    // ── distill_pages: across_kv with window_size 1 ──

    #[test]
    fn swiftkv_distill_pages_across_kv_window_1() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 1,
            enable_across_kv: true,
            ..Default::default()
        });
        let res = skv.distill_pages(8);
        // ceil(8/1) = 8, then ceil(8/2) = 4
        assert_eq!(res.distilled_pages, 4);
    }

    // ── distill_pages: last_result overwritten on subsequent call ──

    #[test]
    fn swiftkv_distill_pages_overwrites_last_result() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 4,
            ..Default::default()
        });
        skv.distill_pages(16);
        assert_eq!(skv.last_result.unwrap().distilled_pages, 4);
        skv.distill_pages(8);
        assert_eq!(skv.last_result.unwrap().distilled_pages, 2);
    }

    // ── DistillResult: ne between different pages ──

    #[test]
    fn distill_result_ne() {
        let a = DistillResult {
            original_pages: vec![vec![1.0_f32]],
            distilled_pages: vec![vec![1.0_f32]],
        };
        let b = DistillResult {
            original_pages: vec![vec![2.0_f32]],
            distilled_pages: vec![vec![1.0_f32]],
        };
        assert_ne!(a, b);
    }

    // ── DistillOutcome: ne between different outcomes ──

    #[test]
    fn distill_outcome_ne_precision_fallback() {
        let a = DistillOutcome {
            result: DistillResult {
                original_pages: vec![vec![1.0_f32]],
                distilled_pages: vec![vec![1.0_f32]],
            },
            ppl_diff: 0.0_f32,
            precision_fallback: false,
        };
        let b = DistillOutcome {
            result: DistillResult {
                original_pages: vec![vec![1.0_f32]],
                distilled_pages: vec![vec![1.0_f32]],
            },
            ppl_diff: 0.0_f32,
            precision_fallback: true,
        };
        assert_ne!(a, b);
    }

    // ── DistillOutcome: ne via ppl_diff ──

    #[test]
    fn distill_outcome_ne_ppl_diff() {
        let a = DistillOutcome {
            result: DistillResult {
                original_pages: vec![vec![1.0_f32]],
                distilled_pages: vec![vec![1.0_f32]],
            },
            ppl_diff: 0.0_f32,
            precision_fallback: false,
        };
        let b = DistillOutcome {
            result: DistillResult {
                original_pages: vec![vec![1.0_f32]],
                distilled_pages: vec![vec![1.0_f32]],
            },
            ppl_diff: 1.0_f32,
            precision_fallback: false,
        };
        assert_ne!(a, b);
    }

    // ── ChunkedConfig: enable_splitfuse field is accessible ──

    #[test]
    fn chunked_config_enable_splitfuse_field_readable() {
        let cfg = ChunkedConfig { enable_splitfuse: true, ..Default::default() };
        assert!(cfg.enable_splitfuse);
    }

    // ── Scheduler2024State: config field matches ──

    #[test]
    fn scheduler_2024_state_config_field_accessible() {
        let cfg = Scheduler2024Config {
            enable_2024_optimizations: true,
            ..Default::default()
        };
        let state = Scheduler2024State::new(cfg);
        assert!(state.config.enable_2024_optimizations);
    }

    // ── AdaptiveChunkPolicy: very large concurrent_reqs clamped ──

    #[test]
    fn adaptive_policy_very_large_concurrent_clamped() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk = policy.compute(0.9, 1000, 10000);
        // penalty = max(0.2, 1 - 0.1 * 999) = max(0.2, -98.9) = 0.2
        assert!(chunk >= policy.min_chunk);
        assert!(chunk <= policy.max_chunk);
    }

    // ── AdaptiveChunkPolicy: mid-range interpolation smoothness ──

    #[test]
    fn adaptive_policy_mid_range_proportional() {
        let policy = AdaptiveChunkPolicy::new(&ChunkedConfig::default());
        let chunk_50 = policy.compute(0.50, 1, 10000);
        let chunk_25 = policy.compute(0.25, 1, 10000);
        let chunk_75 = policy.compute(0.75, 1, 10000);
        // 0.25 is NOT < 0.25 so mid-range; 0.75 is NOT > 0.75 so mid-range
        // 0.50 is solid mid-range
        // Verify ordering: result at 0.5 should be between 0.25 and 0.75
        assert!(chunk_50 >= chunk_25);
        assert!(chunk_50 <= chunk_75);
    }

    // ── AdaptiveChunkPolicy: new with both min and max zero ──

    #[test]
    fn adaptive_policy_new_both_zero_min_clamped_max_stays_zero() {
        let cfg = ChunkedConfig { min_chunk: 0, max_chunk: 0, ..Default::default() };
        let policy = AdaptiveChunkPolicy::new(&cfg);
        // min_chunk clamped to 1, max_chunk = max(0, 0) = 0 (uses raw config.min_chunk)
        assert_eq!(policy.min_chunk, 1);
        assert_eq!(policy.max_chunk, 0);
    }

    // ── AdaptiveChunkPolicy: compute mid-range at exact 0.5 yields expected interpolation ──

    #[test]
    fn adaptive_policy_mid_exact_0_5_interpolation() {
        let cfg = ChunkedConfig { min_chunk: 0, max_chunk: 1000, ..Default::default() };
        let policy = AdaptiveChunkPolicy::new(&cfg);
        // min clamped to 1, max = 1000; t = (0.5 - 0.25) / 0.50 = 0.5
        // base = 1 + (999 * 0.5) = 500 (approx), penalty = 1.0
        let chunk = policy.compute(0.5, 1, 10000);
        let expected = 1 + ((1000 - 1) as f32 * 0.5) as usize;
        assert_eq!(chunk, expected.clamp(policy.min_chunk, policy.max_chunk));
    }

    // ── ChunkedState: pop with remaining_budget=1 ──

    #[test]
    fn chunked_pop_with_remaining_budget_one() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 100,
            max_chunk: 2048,
            decode_slots: 2,
            enable_splitfuse: false,
        });
        state.enqueue(1, 500);
        let chunk = state.pop_adaptive_chunk(1, 0.9, 1, 1).unwrap();
        // remaining_budget=1, so dynamic_chunk_size clamped to min(adjusted, 1)
        assert_eq!(chunk.tokens, 1);
    }

    // ── ChunkedState: enqueue with very large token count ──

    #[test]
    fn chunked_enqueue_large_token_count() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 1024,
            max_chunk: 1024,
            decode_slots: 2,
            enable_splitfuse: false,
        });
        state.enqueue(1, usize::MAX / 2);
        let chunk = state.pop_adaptive_chunk(1, 0.9, 1, usize::MAX).unwrap();
        assert_eq!(chunk.tokens, 1024);
        assert!(!state.is_request_complete(&1));
    }

    // ── SwiftKvState::distill_cpu: pages with zero-length inner vector ──

    #[test]
    fn distill_cpu_pages_with_empty_inner_vector() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        let pages: Vec<Vec<f32>> = vec![vec![], vec![]];
        let outcome = skv.distill_cpu(&pages);
        // Empty inner vectors: chunk[0].len() == 0, acc = vec![0.0; 0] = empty
        // weight_sum = 2.0, merged has one empty vec
        assert_eq!(outcome.result.distilled_pages.len(), 1);
        assert!(outcome.result.distilled_pages[0].is_empty());
    }

    // ── cosine_similarity: mixed positive and negative values ──

    #[test]
    fn cosine_similarity_mixed_signs() {
        let a: Vec<f32> = vec![1.0, -2.0, 3.0];
        let b: Vec<f32> = vec![-1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        // dot = -1 + -4 + 9 = 4, na = 1+4+9=14, nb = 1+4+9=14
        // cos = 4 / (sqrt(14) * sqrt(14)) = 4/14
        assert!((sim - (4.0_f32 / 14.0)).abs() < 1e-4);
    }

    // ── approx_ppl_delta: both original and distilled empty ──

    #[test]
    fn approx_ppl_delta_both_empty_returns_zero() {
        let original: Vec<Vec<f32>> = vec![];
        let distilled: Vec<Vec<f32>> = vec![];
        let delta = approx_ppl_delta(&original, &distilled);
        assert_eq!(delta, 0.0);
    }

    // ── DistillResult: debug format contains pages ──

    #[test]
    fn distill_result_debug_format() {
        let r = DistillResult {
            original_pages: vec![vec![1.0_f32, 2.0]],
            distilled_pages: vec![vec![3.0_f32]],
        };
        let dbg = format!("{:?}", r);
        assert!(dbg.contains("original_pages"));
        assert!(dbg.contains("distilled_pages"));
    }

    // ── DistillOutcome: debug format contains fields ──

    #[test]
    fn distill_outcome_debug_format() {
        let o = DistillOutcome {
            result: DistillResult {
                original_pages: vec![vec![1.0_f32]],
                distilled_pages: vec![vec![2.0_f32]],
            },
            ppl_diff: 0.5_f32,
            precision_fallback: true,
        };
        let dbg = format!("{:?}", o);
        assert!(dbg.contains("ppl_diff"));
        assert!(dbg.contains("precision_fallback"));
    }

    // ── SwiftKVConfig: all custom field values ──

    #[test]
    fn swift_kv_config_all_custom_fields() {
        let cfg = SwiftKVConfig {
            enabled: true,
            window_size: 16,
            enable_across_kv: true,
            similarity_threshold: 0.5,
            precision_guard: 0.05,
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.window_size, 16);
        assert!(cfg.enable_across_kv);
        assert!((cfg.similarity_threshold - 0.5).abs() < 1e-6);
        assert!((cfg.precision_guard - 0.05).abs() < 1e-6);
    }

    // ── distill_cpu_incremental: boundary exactly equals pages_len ──

    #[test]
    fn distill_cpu_incremental_boundary_equals_pages_len() {
        let mut skv = SwiftKvState::new(SwiftKVConfig {
            enabled: true,
            window_size: 2,
            enable_across_kv: false,
            similarity_threshold: 0.99,
            precision_guard: 1.0,
        });
        let pages: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0]];
        skv.distill_cpu_incremental(&pages);
        assert_eq!(skv.last_distilled_page, 2);

        // Same length again: last_distilled_page == pages.len(), falls through to distill_cpu
        let same_pages: Vec<Vec<f32>> = vec![vec![10.0], vec![20.0]];
        let outcome = skv.distill_cpu_incremental(&same_pages);
        assert_eq!(skv.last_distilled_page, 2);
        assert_eq!(outcome.result.original_pages.len(), 2);
    }

    // ── Scheduler2024State: disabled optimizations still creates valid state ──

    #[test]
    fn scheduler_2024_state_disabled_optimizations_functional() {
        let cfg = Scheduler2024Config {
            enable_2024_optimizations: false,
            ..Default::default()
        };
        let mut state = Scheduler2024State::new(cfg);
        assert!(!state.config.enable_2024_optimizations);

        // Chunked operations still work even when optimizations disabled
        state.chunked.enqueue(1, 200);
        let chunk = state.chunked.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(chunk.request_id, 1);
        assert!(chunk.tokens > 0);

        // SwiftKV disabled passthrough
        let res = state.swift_kv.distill_pages(10);
        assert_eq!(res.original_pages, 10);
        assert_eq!(res.distilled_pages, 10);
    }

    // ── ChunkedState: enqueue then remove then enqueue different tokens ──

    #[test]
    fn chunked_enqueue_remove_enqueue_different_tokens() {
        let mut state = ChunkedState::new(ChunkedConfig {
            min_chunk: 1000,
            max_chunk: 1000,
            decode_slots: 2,
            enable_splitfuse: false,
        });
        state.enqueue(1, 1000);
        state.remove_tracker(&1);
        state.enqueue(1, 500);
        let chunk = state.pop_adaptive_chunk(1, 0.9, 1, 10000).unwrap();
        assert_eq!(chunk.tokens, 500);
        assert_eq!(chunk.chunk_idx, 0);
        assert!(state.is_request_complete(&1));
    }
}
