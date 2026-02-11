//! 2024 vLLM optimizations (Chunked Prefill / SwiftKV).
//!
//! This module keeps the orchestration logic fully on the host side so it
//! remains compatible with the AOT CUBIN strategy (no runtime kernel
//! compilation) and the zero-copy generation loop.

use std::collections::{HashMap, VecDeque};

use gllm_kernels::{backend_trait::Element, cpu_kernels::Float, kernel_types::RequestId};

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

/// Unified configuration gate for the three optimizations.
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
pub struct DistillPagesSummary {
    pub original_pages: usize,
    pub distilled_pages: usize,
}

/// 蒸馏结果（泛型）
#[derive(Debug, Clone, PartialEq)]
pub struct DistillResult<E: Element> {
    pub original_pages: Vec<Vec<E>>,
    pub distilled_pages: Vec<Vec<E>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DistillOutcome<E: Element> {
    pub result: DistillResult<E>,
    pub ppl_diff: E,
    /// Whether precision guard forced a fallback to original KV.
    pub precision_fallback: bool,
}

#[derive(Debug)]
pub struct SwiftKvState {
    pub config: SwiftKVConfig,
    /// Last distillation result (for metrics/tests).
    pub last_result: Option<DistillPagesSummary>,
}

impl SwiftKvState {
    pub fn new(config: SwiftKVConfig) -> Self {
        Self {
            config,
            last_result: None,
        }
    }

    /// Distill a page set (CPU-side) to model SwiftKV SIKV + AKV effects.
    pub fn distill_pages(&mut self, page_count: usize) -> DistillPagesSummary {
        if !self.config.enabled || page_count == 0 {
            let result = DistillPagesSummary {
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

        let result = DistillPagesSummary {
            original_pages: page_count,
            distilled_pages: distilled.max(1),
        };
        self.last_result = Some(result);
        result
    }

    /// CPU-side SIKV + AKV distillation over concrete KV page vectors.
    /// This keeps zero-copy on GPU: only CPU buffers are manipulated.
    pub fn distill_cpu<E: Element + Float>(&mut self, pages: &[Vec<E>]) -> DistillOutcome<E> {
        if !self.config.enabled || pages.is_empty() {
            let cloned = pages.to_vec();
            let res = DistillResult {
                original_pages: cloned.clone(),
                distilled_pages: cloned,
            };
            return DistillOutcome {
                result: res,
                ppl_diff: E::ZERO,
                precision_fallback: false,
            };
        }

        // 1) SIKV: sliding-window merge.
        let mut merged: Vec<Vec<E>> = Vec::new();
        let w = self.config.window_size.max(1);
        for chunk in pages.chunks(w) {
            let mut acc = vec![E::ZERO; chunk[0].len()];
            let mut weight_sum = E::ZERO;
            for page in chunk {
                let w = E::ONE; // simple average; attention weights unknown on CPU-only path.
                weight_sum += w;
                for (dst, src) in acc.iter_mut().zip(page.iter()) {
                    *dst += w * *src;
                }
            }
            if weight_sum > E::ZERO {
                for v in acc.iter_mut() {
                    *v = *v / weight_sum;
                }
            }
            merged.push(acc);
        }

        // 2) AKV: cosine similarity across adjacent layers.
        let mut akv_shared: Vec<Vec<E>> = Vec::new();
        let mut precision_fallback = false;
        for i in 0..merged.len() {
            if i > 0 && self.config.enable_across_kv {
                let sim = cosine_similarity(&merged[i - 1], &merged[i]);
                if sim >= E::from_f32(self.config.similarity_threshold) {
                    // Share previous layer; no extra storage needed.
                    continue;
                }
            }
            akv_shared.push(merged[i].clone());
        }

        // 3) Precision guard: compare reconstruction error as proxy for PPL diff.
        let ppl_diff = approx_ppl_delta(pages, &akv_shared);
        if ppl_diff > E::from_f32(self.config.precision_guard) {
            // Fallback to original pages for correctness.
            precision_fallback = true;
            let cloned = pages.to_vec();
            let res = DistillResult {
                original_pages: cloned.clone(),
                distilled_pages: cloned,
            };
            self.last_result = Some(DistillPagesSummary {
                original_pages: pages.len(),
                distilled_pages: pages.len(),
            });
            return DistillOutcome {
                result: res,
                ppl_diff,
                precision_fallback,
            };
        }

        let res = DistillResult {
            original_pages: pages.to_vec(),
            distilled_pages: if akv_shared.is_empty() {
                pages.to_vec()
            } else {
                akv_shared.clone()
            },
        };
        self.last_result = Some(DistillPagesSummary {
            original_pages: pages.len(),
            distilled_pages: res.distilled_pages.len(),
        });
        DistillOutcome {
            result: res,
            ppl_diff,
            precision_fallback,
        }
    }
}

pub fn cosine_similarity<E: Element>(a: &[E], b: &[E]) -> E {
    let mut dot = E::ZERO;
    let mut na = E::ZERO;
    let mut nb = E::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        dot = dot + (*x) * (*y);
        na = na + (*x) * (*x);
        nb = nb + (*y) * (*y);
    }
    if na == E::ZERO || nb == E::ZERO {
        return E::ZERO;
    }
    dot / (na.sqrt() * nb.sqrt())
}

pub fn approx_ppl_delta<E: Element>(original: &[Vec<E>], distilled: &[Vec<E>]) -> E {
    // Use normalized L2 reconstruction error as a proxy; small -> low PPL shift.
    let mut total = E::ZERO;
    let mut count = 0usize;
    if distilled.is_empty() {
        return E::ZERO;
    }
    for (i, page) in original.iter().enumerate() {
        let distilled_page = distilled.get(i.min(distilled.len().saturating_sub(1)));
        if let Some(dp) = distilled_page {
            let mut l2 = E::ZERO;
            for (x, y) in page.iter().zip(dp.iter()) {
                let d = *x - *y;
                l2 = l2 + d * d;
            }
            let norm = page
                .iter()
                .fold(E::ZERO, |acc, v| acc + (*v) * (*v))
                .max(E::from_f32(1e-6));
            total = total + (l2 / norm).sqrt();
            count += 1;
        }
    }
    if count == 0 {
        E::ZERO
    } else {
        total / E::from_f32(count as f32)
    }
}

// --------------------------- State Aggregation ----------------------------

#[derive(Debug)]
pub struct Scheduler2024State {
    pub config: Scheduler2024Config,
    pub chunked: ChunkedState,
    pub swift_kv: SwiftKvState,
}

impl Scheduler2024State {
    pub fn new(config: Scheduler2024Config) -> Self {
        Self {
            chunked: ChunkedState::new(config.chunked),
            swift_kv: SwiftKvState::new(config.swift_kv),
            config,
        }
    }
}
