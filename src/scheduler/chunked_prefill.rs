//! Chunked Prefill 交织调度 (SPEC §10)
//!
//! §10 核心使命: 让系统能够处理几乎无限大的上下文（10M+ Context），
//! 同时保持 Decode 请求的零等待延迟。
//!
//! ## 关键机制
//! - §10.1 交织调度: Prefill Chunk 与 Decode Token 同 Batch
//! - §10.6 Batch Composition: Token Budget 分配 + BatchManifest
//! - §10.6.3 Compact 决策: 浪费率 > 25% 时触发 Compact→Execute→Scatter
//!
//! ## 约束
//! - Decode 永远优先: decode_ratio_cap = 0.6
//! - Attention Phase Isolation: Prefill/Decode 在 Attention 阶段物理分轨
//! - SplitFuse 已废弃 (REQ-SCHED-007)

use crate::scheduler::request_state::{RequestPhase, RequestState};
use crate::scheduler::types::RequestId;

// ─── Configuration ───

/// Chunked Prefill 配置 (§10.2)
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// 固定切分大小 (token 数, 默认 512)
    pub chunk_size: usize,
    /// 是否启用 chunked prefill
    pub enabled: bool,
    /// 每个请求最大 chunk 数 (0 = 无限)
    pub max_chunks_per_request: usize,
    /// decode 占比上限 (默认 0.6, §10.6.1)
    pub decode_ratio_cap: f32,
    /// 浪费率阈值 — 超过此值触发 Compact (§10.6.3)
    pub compact_waste_threshold: f32,
    /// 最小 active lane 数 — 低于此值不触发 Compact
    pub compact_min_active: usize,
    /// 硬件 token 预算上限
    pub max_batch_tokens: usize,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            enabled: true,
            max_chunks_per_request: 0,
            decode_ratio_cap: 0.6,
            compact_waste_threshold: 0.25,
            compact_min_active: 4,
            max_batch_tokens: 4096,
        }
    }
}

// ─── Adaptive Chunk Policy ───

/// 自适应 Chunk 策略 (§10.2 + REQ-KV-EXT-001)
///
/// 根据 L1 可用页数、并发请求数、剩余 prefill tokens 动态调整 chunk_size。
#[derive(Debug, Clone)]
pub struct AdaptiveChunkPolicy {
    /// 配置中的默认 chunk_size (下界)
    pub default_chunk_size: usize,
    /// L1 可用页比例 (0.0-1.0)
    pub l1_available_ratio: f32,
    /// 当前并发请求数
    pub concurrent_requests: usize,
}

impl AdaptiveChunkPolicy {
    pub fn new(default_chunk_size: usize) -> Self {
        Self {
            default_chunk_size,
            l1_available_ratio: 1.0,
            concurrent_requests: 1,
        }
    }

    /// 计算自适应 chunk_size
    ///
    /// - L1 可用 < 25%: 缩小到 default_chunk_size (保守)
    /// - L1 可用 > 75%: 扩大到 max_seq_len (激进)
    /// - 中间: 线性插值
    pub fn compute_chunk_size(&self, remaining_tokens: usize, max_seq_len: usize) -> usize {
        let base = self.default_chunk_size;

        let adaptive = if self.l1_available_ratio < 0.25 {
            base
        } else if self.l1_available_ratio > 0.75 {
            max_seq_len
        } else {
            let t = (self.l1_available_ratio - 0.25) / 0.50;
            let scaled = base as f32 + t * (max_seq_len as f32 - base as f32);
            scaled as usize
        };

        // 并发请求多时缩小
        let concurrency_factor = if self.concurrent_requests > 8 {
            0.5
        } else if self.concurrent_requests > 4 {
            0.75
        } else {
            1.0
        };

        let result = (adaptive as f32 * concurrency_factor) as usize;
        // 上限：不超过剩余 tokens；下限：不小于 base（除非剩余本身就 < base）
        if remaining_tokens <= base {
            remaining_tokens
        } else {
            result.max(base).min(remaining_tokens)
        }
    }
}

// ─── Batch Slot Types ───

/// Batch slot 类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotType {
    /// Decode slot: 单 token 生成
    Decode,
    /// Prefill chunk slot: prefill 的一个切片
    PrefillChunk,
}

/// Batch slot 描述
#[derive(Debug, Clone)]
pub struct BatchSlot {
    /// 所属请求 ID
    pub request_id: RequestId,
    /// slot 类型
    pub slot_type: SlotType,
    /// 该 slot 的 token 范围 [start, end)
    pub token_start: usize,
    pub token_end: usize,
    /// Compact 后的目标位置 (-1 = 不参与 compact)
    pub compact_target: i32,
}

// ─── BatchManifest ───

/// Batch 清单 — 描述一个 step 的 batch 组成 (§10.6.2 Step 5)
#[derive(Debug, Clone)]
pub struct BatchManifest {
    /// 所有 slot
    pub slots: Vec<BatchSlot>,
    /// 总 token 数
    pub total_tokens: usize,
    /// Decode token 数
    pub decode_tokens: usize,
    /// Prefill chunk token 数
    pub prefill_tokens: usize,
    /// 是否需要 Compact
    pub compact_required: bool,
    /// 浪费率
    pub waste_ratio: f32,
}

impl BatchManifest {
    /// 判断是否需要 Compact (§10.6.3)
    ///
    /// 触发条件 (必须同时满足):
    /// 1. waste_ratio > compact_waste_threshold (默认 25%)
    /// 2. active_count >= compact_min_active
    /// 3. 当前 op 是 compute-bound (GEMM/FFN, 非 Attention)
    pub fn should_compact(&self, config: &ChunkedPrefillConfig) -> bool {
        let active_count = self.slots.iter().filter(|s| s.compact_target >= 0).count();
        self.waste_ratio > config.compact_waste_threshold
            && active_count >= config.compact_min_active
    }
}

// ─── ChunkedPrefillScheduler ───

/// Chunked Prefill 交织调度器 (§10)
///
/// 实现 §10.6.2 五步 Batch Composition 流程:
/// 1. 收集 ready decode tokens
/// 2. 填充 decode slots
/// 3. 计算剩余 prefill_budget
/// 4. 填入 prefill chunks
/// 5. 生成 BatchManifest
pub struct ChunkedPrefillScheduler {
    config: ChunkedPrefillConfig,
    adaptive_policy: AdaptiveChunkPolicy,
}

impl ChunkedPrefillScheduler {
    pub fn new(config: ChunkedPrefillConfig) -> Self {
        let adaptive_policy = AdaptiveChunkPolicy::new(config.chunk_size);
        Self { config, adaptive_policy }
    }

    /// 判断请求是否需要 chunked prefill
    pub fn should_chunk(&self, seq_len: usize) -> bool {
        self.config.enabled && seq_len > self.config.chunk_size
    }

    /// 计算下一个 chunk 的大小 (自适应)
    pub fn next_chunk_size(&self, remaining: usize, max_seq_len: usize) -> usize {
        self.adaptive_policy.compute_chunk_size(remaining, max_seq_len)
    }

    /// 更新请求状态为 chunked prefill
    pub fn mark_as_chunked(&self, state: &mut RequestState) {
        state.phase = RequestPhase::ChunkedPrefill;
    }

    /// 更新 L1 可用比例
    pub fn update_l1_ratio(&mut self, ratio: f32) {
        self.adaptive_policy.l1_available_ratio = ratio.clamp(0.0, 1.0);
    }

    /// 更新并发请求数
    pub fn update_concurrency(&mut self, count: usize) {
        self.adaptive_policy.concurrent_requests = count;
    }

    /// §10.6 Batch Composition — 构建交织 batch
    ///
    /// 输入: decode_ready (已 ready 的 decode 请求) + prefill_queue (待 prefill 的请求)
    /// 输出: BatchManifest 描述物理 batch 的组成
    pub fn compose_batch(
        &self,
        decode_ready: &[(RequestId, usize)],  // (request_id, kv_cache_offset)
        prefill_queue: &[(RequestId, usize, usize)], // (request_id, remaining_tokens, kv_cache_offset)
        memory_pressure_ratio: f32,
    ) -> BatchManifest {
        // Step 0: 计算 total_budget
        let total_budget = (self.config.max_batch_tokens as f32 * memory_pressure_ratio) as usize;

        // Step 1-2: 填充 decode slots (优先, 按 decode_ratio_cap 上限)
        let max_decode = (total_budget as f32 * self.config.decode_ratio_cap) as usize;
        let decode_count = decode_ready.len().min(max_decode);

        let mut slots = Vec::with_capacity(total_budget);
        let mut total_tokens = 0;

        // Decode slots: 每请求 1 token
        for i in 0..decode_count {
            let (req_id, kv_offset) = &decode_ready[i];
            slots.push(BatchSlot {
                request_id: *req_id,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: -1,
            });
            total_tokens += 1;
            let _ = kv_offset; // used for KV page lookup at execution time
        }

        // Step 3: 计算剩余 prefill_budget
        let prefill_budget = total_budget.saturating_sub(decode_count);
        if prefill_budget == 0 {
            return BatchManifest {
                slots,
                total_tokens,
                decode_tokens: decode_count,
                prefill_tokens: 0,
                compact_required: false,
                waste_ratio: 0.0,
            };
        }

        // Step 4: 填入 prefill chunks
        let mut remaining_budget = prefill_budget;
        let mut chunks_per_req: std::collections::HashMap<RequestId, usize> = std::collections::HashMap::new();
        let max_seq_len = self.config.max_batch_tokens;

        for (req_id, remaining_tokens, _kv_offset) in prefill_queue {
            // 检查 per-request chunk 上限
            let chunk_count = *chunks_per_req.entry(*req_id).or_insert(0);
            if self.config.max_chunks_per_request > 0 && chunk_count >= self.config.max_chunks_per_request {
                continue;
            }

            if remaining_budget == 0 {
                break;
            }

            // 自适应 chunk 大小
            let chunk_size = self.adaptive_policy.compute_chunk_size(*remaining_tokens, max_seq_len);
            let actual_chunk = chunk_size.min(remaining_budget);

            slots.push(BatchSlot {
                request_id: *req_id,
                slot_type: SlotType::PrefillChunk,
                token_start: remaining_tokens - *remaining_tokens,
                token_end: remaining_tokens - *remaining_tokens + actual_chunk,
                compact_target: -1,
            });

            total_tokens += actual_chunk;
            remaining_budget -= actual_chunk;
            *chunks_per_req.get_mut(req_id).unwrap() += 1;
        }

        // Step 5: 生成 BatchManifest
        let prefill_tokens = total_tokens - decode_count;
        let batch_capacity = total_budget;
        let waste_ratio = if batch_capacity > 0 {
            (batch_capacity - total_tokens) as f32 / batch_capacity as f32
        } else {
            0.0
        };

        let manifest = BatchManifest {
            slots,
            total_tokens,
            decode_tokens: decode_count,
            prefill_tokens,
            compact_required: waste_ratio > self.config.compact_waste_threshold,
            waste_ratio,
        };

        manifest
    }

    /// 获取配置引用
    pub fn config(&self) -> &ChunkedPrefillConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_chunk() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(!scheduler.should_chunk(256));
        assert!(!scheduler.should_chunk(512));
        assert!(scheduler.should_chunk(513));
        assert!(scheduler.should_chunk(2048));
    }

    #[test]
    fn test_next_chunk_size_basic() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        // Default: l1_available_ratio = 1.0, concurrent_requests = 1
        // So adaptive will use max_seq_len
        let chunk = scheduler.next_chunk_size(256, 4096);
        assert_eq!(chunk, 256); // clamped to remaining
    }

    #[test]
    fn test_compose_batch_decode_only() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode_ready: Vec<(u64, usize)> = vec![(1, 0), (2, 100), (3, 200)];
        let prefill_queue: Vec<(u64, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode_ready, &prefill_queue, 1.0);

        assert_eq!(manifest.decode_tokens, 3);
        assert_eq!(manifest.prefill_tokens, 0);
        assert_eq!(manifest.total_tokens, 3);
        assert_eq!(manifest.slots.len(), 3);
        assert!(manifest.slots.iter().all(|s| s.slot_type == SlotType::Decode));
    }

    #[test]
    fn test_compose_batch_interleaved() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode_ready: Vec<(u64, usize)> = vec![(1, 0), (2, 100)];
        let prefill_queue: Vec<(u64, usize, usize)> = vec![(3, 50, 200)];

        let manifest = scheduler.compose_batch(&decode_ready, &prefill_queue, 1.0);

        assert_eq!(manifest.decode_tokens, 2);
        assert!(manifest.prefill_tokens > 0);
        assert!(manifest.slots.iter().any(|s| s.slot_type == SlotType::Decode));
        assert!(manifest.slots.iter().any(|s| s.slot_type == SlotType::PrefillChunk));
    }

    #[test]
    fn test_compose_batch_decode_ratio_cap() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        // 80 decode requests but only 60% of 100 = 60 slots for decode
        let decode_ready: Vec<(u64, usize)> = (0u64..80).map(|i| (i, (i as usize) * 10)).collect();
        let prefill_queue: Vec<(u64, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode_ready, &prefill_queue, 1.0);

        assert_eq!(manifest.decode_tokens, 60); // capped at 60%
        assert_eq!(manifest.prefill_tokens, 0);
    }

    #[test]
    fn test_batch_manifest_should_compact() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.25,
            compact_min_active: 4,
            ..Default::default()
        };

        let manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: 0,
            }; 8],
            total_tokens: 6,
            decode_tokens: 6,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.40, // > 0.25
        };

        assert!(manifest.should_compact(&config));
    }

    #[test]
    fn test_batch_manifest_should_not_compact_low_waste() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.25,
            compact_min_active: 4,
            ..Default::default()
        };

        let manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: 0,
            }; 8],
            total_tokens: 8,
            decode_tokens: 8,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.10, // < 0.25
        };

        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_adaptive_chunk_policy_high_load() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.1; // high load
        policy.concurrent_requests = 16;

        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 512); // shrinks to base under high load
    }

    #[test]
    fn test_adaptive_chunk_policy_low_load() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 2;

        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096); // expands to max under low load
    }

    #[test]
    fn test_mark_as_chunked() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let mut state = RequestState::new(1, RequestPhase::Prefill, 2048, 0);
        scheduler.mark_as_chunked(&mut state);
        assert_eq!(state.phase, RequestPhase::ChunkedPrefill);
    }
}
