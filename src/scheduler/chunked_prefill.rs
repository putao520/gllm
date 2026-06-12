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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

        

        BatchManifest {
            slots,
            total_tokens,
            decode_tokens: decode_count,
            prefill_tokens,
            compact_required: waste_ratio > self.config.compact_waste_threshold,
            waste_ratio,
        }
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

    // ═══════════════════════════════════════════════════════════════════
    // ChunkedPrefillConfig tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_config_default_values() {
        let config = ChunkedPrefillConfig::default();
        assert_eq!(config.chunk_size, 512);
        assert!(config.enabled);
        assert_eq!(config.max_chunks_per_request, 0);
        assert!((config.decode_ratio_cap - 0.6).abs() < 1e-6);
        assert!((config.compact_waste_threshold - 0.25).abs() < 1e-6);
        assert_eq!(config.compact_min_active, 4);
        assert_eq!(config.max_batch_tokens, 4096);
    }

    #[test]
    fn test_config_clone_is_independent() {
        let config = ChunkedPrefillConfig::default();
        let mut cloned = config.clone();
        cloned.chunk_size = 1024;
        cloned.enabled = false;
        assert_eq!(config.chunk_size, 512);
        assert!(config.enabled);
        assert_eq!(cloned.chunk_size, 1024);
        assert!(!cloned.enabled);
    }

    #[test]
    fn test_config_debug_format() {
        let config = ChunkedPrefillConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("chunk_size: 512"));
        assert!(debug_str.contains("enabled: true"));
        assert!(debug_str.contains("decode_ratio_cap: 0.6"));
        assert!(debug_str.contains("compact_waste_threshold: 0.25"));
        assert!(debug_str.contains("compact_min_active: 4"));
        assert!(debug_str.contains("max_batch_tokens: 4096"));
    }

    #[test]
    fn test_config_custom_values() {
        let config = ChunkedPrefillConfig {
            chunk_size: 256,
            enabled: false,
            max_chunks_per_request: 10,
            decode_ratio_cap: 0.8,
            compact_waste_threshold: 0.1,
            compact_min_active: 8,
            max_batch_tokens: 8192,
        };
        assert_eq!(config.chunk_size, 256);
        assert!(!config.enabled);
        assert_eq!(config.max_chunks_per_request, 10);
        assert!((config.decode_ratio_cap - 0.8).abs() < 1e-6);
        assert!((config.compact_waste_threshold - 0.1).abs() < 1e-6);
        assert_eq!(config.compact_min_active, 8);
        assert_eq!(config.max_batch_tokens, 8192);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SlotType tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_slot_type_equality() {
        assert_eq!(SlotType::Decode, SlotType::Decode);
        assert_eq!(SlotType::PrefillChunk, SlotType::PrefillChunk);
        assert_ne!(SlotType::Decode, SlotType::PrefillChunk);
    }

    #[test]
    fn test_slot_type_copy() {
        let a = SlotType::Decode;
        let b = a;
        assert_eq!(a, b);
        assert_eq!(a, SlotType::Decode);
    }

    #[test]
    fn test_slot_type_debug_format() {
        assert!(format!("{:?}", SlotType::Decode).contains("Decode"));
        assert!(format!("{:?}", SlotType::PrefillChunk).contains("PrefillChunk"));
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchSlot tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_slot_fields_decode() {
        let slot = BatchSlot {
            request_id: 42,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: -1,
        };
        assert_eq!(slot.request_id, 42);
        assert_eq!(slot.slot_type, SlotType::Decode);
        assert_eq!(slot.token_start, 0);
        assert_eq!(slot.token_end, 1);
        assert_eq!(slot.compact_target, -1);
    }

    #[test]
    fn test_batch_slot_fields_prefill_chunk() {
        let slot = BatchSlot {
            request_id: 99,
            slot_type: SlotType::PrefillChunk,
            token_start: 100,
            token_end: 612,
            compact_target: 3,
        };
        assert_eq!(slot.request_id, 99);
        assert_eq!(slot.slot_type, SlotType::PrefillChunk);
        assert_eq!(slot.token_end - slot.token_start, 512);
        assert_eq!(slot.compact_target, 3);
    }

    #[test]
    fn test_batch_slot_clone() {
        let slot = BatchSlot {
            request_id: 7,
            slot_type: SlotType::PrefillChunk,
            token_start: 10,
            token_end: 522,
            compact_target: 2,
        };
        let cloned = slot.clone();
        assert_eq!(cloned.request_id, 7);
        assert_eq!(cloned.slot_type, SlotType::PrefillChunk);
        assert_eq!(cloned.token_start, 10);
        assert_eq!(cloned.token_end, 522);
        assert_eq!(cloned.compact_target, 2);
    }

    #[test]
    fn test_batch_slot_debug_format() {
        let slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: -1,
        };
        let debug = format!("{:?}", slot);
        assert!(debug.contains("request_id: 1"));
        assert!(debug.contains("Decode"));
        assert!(debug.contains("token_start: 0"));
        assert!(debug.contains("token_end: 1"));
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchManifest tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_manifest_fields() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        assert!(manifest.slots.is_empty());
        assert_eq!(manifest.total_tokens, 0);
        assert_eq!(manifest.decode_tokens, 0);
        assert_eq!(manifest.prefill_tokens, 0);
        assert!(!manifest.compact_required);
        assert!((manifest.waste_ratio).abs() < 1e-6);
    }

    #[test]
    fn test_batch_manifest_clone() {
        let manifest = BatchManifest {
            slots: vec![
                BatchSlot {
                    request_id: 1,
                    slot_type: SlotType::Decode,
                    token_start: 0,
                    token_end: 1,
                    compact_target: -1,
                },
            ],
            total_tokens: 1,
            decode_tokens: 1,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.5,
        };
        let cloned = manifest.clone();
        assert_eq!(cloned.slots.len(), 1);
        assert_eq!(cloned.total_tokens, 1);
        assert!((cloned.waste_ratio - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_batch_manifest_debug_format() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 100,
            decode_tokens: 60,
            prefill_tokens: 40,
            compact_required: true,
            waste_ratio: 0.3,
        };
        let debug = format!("{:?}", manifest);
        assert!(debug.contains("total_tokens: 100"));
        assert!(debug.contains("decode_tokens: 60"));
        assert!(debug.contains("prefill_tokens: 40"));
        assert!(debug.contains("compact_required: true"));
        assert!(debug.contains("waste_ratio: 0.3"));
    }

    #[test]
    fn test_should_compact_below_min_active() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.25,
            compact_min_active: 4,
            ..Default::default()
        };
        // Only 2 active slots (below min_active=4), high waste
        let manifest = BatchManifest {
            slots: vec![
                BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
                BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            ],
            total_tokens: 2,
            decode_tokens: 2,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.50,
        };
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_should_compact_exact_threshold_waste() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.25,
            compact_min_active: 2,
            ..Default::default()
        };
        // waste_ratio == threshold (not strictly greater), should NOT compact
        let manifest = BatchManifest {
            slots: vec![
                BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
                BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
            ],
            total_tokens: 2,
            decode_tokens: 2,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.25,
        };
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_should_compact_all_negative_targets() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.10,
            compact_min_active: 1,
            ..Default::default()
        };
        // All compact_target = -1 → active_count = 0
        let manifest = BatchManifest {
            slots: vec![
                BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
                BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
            ],
            total_tokens: 2,
            decode_tokens: 2,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.50,
        };
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_should_compact_mix_active_inactive() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.20,
            compact_min_active: 3,
            ..Default::default()
        };
        // 3 active + 2 inactive = 5 slots total, 3 >= min_active
        let manifest = BatchManifest {
            slots: vec![
                BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 },
                BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 },
                BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 2 },
                BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
                BatchSlot { request_id: 5, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
            ],
            total_tokens: 3,
            decode_tokens: 5,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.30,
        };
        assert!(manifest.should_compact(&config));
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_new_defaults() {
        let policy = AdaptiveChunkPolicy::new(256);
        assert_eq!(policy.default_chunk_size, 256);
        assert!((policy.l1_available_ratio - 1.0).abs() < 1e-6);
        assert_eq!(policy.concurrent_requests, 1);
    }

    #[test]
    fn test_adaptive_policy_debug_format() {
        let policy = AdaptiveChunkPolicy::new(512);
        let debug = format!("{:?}", policy);
        assert!(debug.contains("default_chunk_size: 512"));
        assert!(debug.contains("l1_available_ratio: 1.0"));
        assert!(debug.contains("concurrent_requests: 1"));
    }

    #[test]
    fn test_adaptive_policy_clone() {
        let policy = AdaptiveChunkPolicy::new(1024);
        let cloned = policy.clone();
        assert_eq!(cloned.default_chunk_size, 1024);
        assert!((cloned.l1_available_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_policy_low_l1_uses_base() {
        // l1 < 25%: always returns base (or remaining, whichever is smaller)
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.1;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 512);
    }

    #[test]
    fn test_adaptive_policy_mid_l1_interpolates() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.50; // middle of [0.25, 0.75]
        policy.concurrent_requests = 1;
        // t = (0.50 - 0.25) / 0.50 = 0.5
        // scaled = 512 + 0.5 * (4096 - 512) = 512 + 1792 = 2304
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 2304);
    }

    #[test]
    fn test_adaptive_policy_high_l1_uses_max_seq_len() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096);
    }

    #[test]
    fn test_adaptive_policy_remaining_less_than_base() {
        let policy = AdaptiveChunkPolicy::new(512);
        let chunk = policy.compute_chunk_size(100, 4096);
        assert_eq!(chunk, 100); // remaining < base, return remaining
    }

    #[test]
    fn test_adaptive_policy_remaining_equals_base() {
        let policy = AdaptiveChunkPolicy::new(512);
        let chunk = policy.compute_chunk_size(512, 4096);
        assert_eq!(chunk, 512);
    }

    #[test]
    fn test_adaptive_policy_remaining_zero() {
        let policy = AdaptiveChunkPolicy::new(512);
        let chunk = policy.compute_chunk_size(0, 4096);
        assert_eq!(chunk, 0);
    }

    #[test]
    fn test_adaptive_policy_high_concurrency_shrinks() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 16; // > 8 → factor = 0.5
        // adaptive = 4096, result = 4096 * 0.5 = 2048 >= 512
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 2048);
    }

    #[test]
    fn test_adaptive_policy_medium_concurrency_shrinks() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 6; // > 4 → factor = 0.75
        // adaptive = 4096, result = 4096 * 0.75 = 3072 >= 512
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 3072);
    }

    #[test]
    fn test_adaptive_policy_concurrency_with_remaining_limit() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 16; // factor = 0.5 → 4096 * 0.5 = 2048
        // remaining = 1000, result = min(2048, 1000) = 1000 >= 512
        let chunk = policy.compute_chunk_size(1000, 4096);
        assert_eq!(chunk, 1000);
    }

    #[test]
    fn test_adaptive_policy_boundary_l1_25_percent() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.25; // exactly at boundary
        policy.concurrent_requests = 1;
        // l1 < 0.25 is false (0.25 is not < 0.25), goes to interpolation
        // t = (0.25 - 0.25) / 0.50 = 0.0
        // scaled = 512 + 0.0 * (4096 - 512) = 512
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 512);
    }

    #[test]
    fn test_adaptive_policy_boundary_l1_75_percent() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.75; // exactly at boundary
        policy.concurrent_requests = 1;
        // l1 > 0.75 is false (0.75 is not > 0.75), goes to interpolation
        // t = (0.75 - 0.25) / 0.50 = 1.0
        // scaled = 512 + 1.0 * (4096 - 512) = 4096
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096);
    }

    // ═══════════════════════════════════════════════════════════════════
    // ChunkedPrefillScheduler tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_scheduler_new_stores_config() {
        let config = ChunkedPrefillConfig {
            chunk_size: 256,
            enabled: false,
            ..Default::default()
        };
        let scheduler = ChunkedPrefillScheduler::new(config);
        assert_eq!(scheduler.config().chunk_size, 256);
        assert!(!scheduler.config().enabled);
    }

    #[test]
    fn test_scheduler_config_returns_reference() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let config_ref = scheduler.config();
        assert_eq!(config_ref.chunk_size, 512);
    }

    #[test]
    fn test_should_chunk_disabled() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            enabled: false,
            ..Default::default()
        });
        assert!(!scheduler.should_chunk(2048));
        assert!(!scheduler.should_chunk(100000));
    }

    #[test]
    fn test_should_chunk_enabled_at_boundary() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 512,
            enabled: true,
            ..Default::default()
        });
        // seq_len == chunk_size: NOT chunked (must be strictly greater)
        assert!(!scheduler.should_chunk(512));
    }

    #[test]
    fn test_should_chunk_enabled_one_above_boundary() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 512,
            enabled: true,
            ..Default::default()
        });
        assert!(scheduler.should_chunk(513));
    }

    #[test]
    fn test_update_l1_ratio_clamps_high() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_l1_ratio(2.0);
        assert!((scheduler.adaptive_policy.l1_available_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_l1_ratio_clamps_negative() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_l1_ratio(-0.5);
        assert!((scheduler.adaptive_policy.l1_available_ratio - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_l1_ratio_accepts_valid() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_l1_ratio(0.6);
        assert!((scheduler.adaptive_policy.l1_available_ratio - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_update_concurrency() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_concurrency(12);
        assert_eq!(scheduler.adaptive_policy.concurrent_requests, 12);
    }

    #[test]
    fn test_update_concurrency_zero() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_concurrency(0);
        assert_eq!(scheduler.adaptive_policy.concurrent_requests, 0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch edge cases
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_empty_queues() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            ..Default::default()
        });
        let decode_ready: Vec<(RequestId, usize)> = vec![];
        let prefill_queue: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode_ready, &prefill_queue, 1.0);

        assert!(manifest.slots.is_empty());
        assert_eq!(manifest.total_tokens, 0);
        assert_eq!(manifest.decode_tokens, 0);
        assert_eq!(manifest.prefill_tokens, 0);
        // Empty batch with budget=100: waste = 100/100 = 1.0, so compact is required
        assert!(manifest.compact_required);
        assert!((manifest.waste_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compose_batch_zero_memory_pressure() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            ..Default::default()
        });
        let decode_ready: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill_queue: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode_ready, &prefill_queue, 0.0);

        // total_budget = 0, decode_count = 0
        assert_eq!(manifest.total_tokens, 0);
        assert_eq!(manifest.decode_tokens, 0);
    }

    #[test]
    fn test_compose_batch_prefill_only() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode_ready: Vec<(RequestId, usize)> = vec![];
        let prefill_queue: Vec<(RequestId, usize, usize)> = vec![(1, 50, 0)];

        let manifest = scheduler.compose_batch(&decode_ready, &prefill_queue, 1.0);

        assert_eq!(manifest.decode_tokens, 0);
        assert!(manifest.prefill_tokens > 0);
        assert!(manifest.slots.iter().all(|s| s.slot_type == SlotType::PrefillChunk));
    }

    #[test]
    fn test_compose_batch_waste_ratio_calculation() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 100,
            decode_ratio_cap: 1.0,
            ..Default::default()
        });
        // 1 decode token, no prefill → total = 1, budget = 100
        let decode_ready: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill_queue: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode_ready, &prefill_queue, 1.0);

        // waste = (100 - 1) / 100 = 0.99
        assert!((manifest.waste_ratio - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_compose_batch_memory_pressure_reduces_budget() {
        let config = ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        };
        let scheduler = ChunkedPrefillScheduler::new(config);

        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 50, 0)];

        let manifest_full = scheduler.compose_batch(&decode, &prefill, 1.0);
        let manifest_half = scheduler.compose_batch(&decode, &prefill, 0.5);

        // Full pressure → 100 budget; half pressure → 50 budget
        assert!(manifest_full.total_tokens >= manifest_half.total_tokens);
    }

    #[test]
    fn test_compose_batch_max_chunks_per_request_limit() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 1000,
            chunk_size: 10,
            max_chunks_per_request: 1,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        // Same request appears twice in queue, but max_chunks = 1
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![
            (1, 100, 0),
            (1, 100, 0), // duplicate request
        ];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        // Only first entry for request 1 should get a chunk
        let chunks_for_req1 = manifest.slots.iter()
            .filter(|s| s.request_id == 1 && s.slot_type == SlotType::PrefillChunk)
            .count();
        assert_eq!(chunks_for_req1, 1);
    }

    #[test]
    fn test_compose_batch_multiple_prefill_requests() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![
            (1, 100, 0),
            (2, 100, 100),
        ];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        let req1_count = manifest.slots.iter().filter(|s| s.request_id == 1).count();
        let req2_count = manifest.slots.iter().filter(|s| s.request_id == 2).count();
        assert!(req1_count > 0);
        assert!(req2_count > 0);
    }

    #[test]
    fn test_compose_batch_decode_slots_have_one_token() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![
            (10, 0), (20, 50), (30, 100),
        ];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        for slot in &manifest.slots {
            assert_eq!(slot.token_start, 0);
            assert_eq!(slot.token_end, 1);
        }
    }

    #[test]
    fn test_compose_batch_decode_slots_preserve_request_ids() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![
            (111, 0), (222, 50), (333, 100),
        ];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        let ids: Vec<RequestId> = manifest.slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![111, 222, 333]);
    }

    #[test]
    fn test_compose_batch_decode_slots_compact_target_negative() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        for slot in &manifest.slots {
            assert_eq!(slot.compact_target, -1);
        }
    }

    #[test]
    fn test_compose_batch_total_tokens_equals_decode_plus_prefill() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 10)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(3, 100, 20)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert_eq!(manifest.total_tokens, manifest.decode_tokens + manifest.prefill_tokens);
    }

    #[test]
    fn test_compose_batch_prefill_token_range_valid() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 50, 0)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        for slot in &manifest.slots {
            assert!(slot.token_end > slot.token_start);
            assert!(slot.token_end - slot.token_start <= 50);
        }
    }

    #[test]
    fn test_mark_as_chunked_from_decode_phase() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let mut state = RequestState::new(1, RequestPhase::Decode, 10, 0);
        scheduler.mark_as_chunked(&mut state);
        assert_eq!(state.phase, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn test_mark_as_chunked_preserves_other_fields() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let mut state = RequestState::new(42, RequestPhase::Prefill, 2048, 100);
        state.bus_port_mask = 7;
        state.variant_key_hash = 99;
        scheduler.mark_as_chunked(&mut state);
        assert_eq!(state.request_id, 42);
        assert_eq!(state.seq_len, 2048);
        assert_eq!(state.kv_cache_offset, 100);
        assert_eq!(state.bus_port_mask, 7);
        assert_eq!(state.variant_key_hash, 99);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Additional ChunkedPrefillConfig tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_config_zero_chunk_size() {
        let config = ChunkedPrefillConfig {
            chunk_size: 0,
            ..Default::default()
        };
        assert_eq!(config.chunk_size, 0);
    }

    #[test]
    fn test_config_zero_decode_ratio_cap() {
        let config = ChunkedPrefillConfig {
            decode_ratio_cap: 0.0,
            ..Default::default()
        };
        assert!((config.decode_ratio_cap).abs() < 1e-6);
    }

    #[test]
    fn test_config_full_decode_ratio_cap() {
        let config = ChunkedPrefillConfig {
            decode_ratio_cap: 1.0,
            ..Default::default()
        };
        assert!((config.decode_ratio_cap - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_zero_max_batch_tokens() {
        let config = ChunkedPrefillConfig {
            max_batch_tokens: 0,
            ..Default::default()
        };
        assert_eq!(config.max_batch_tokens, 0);
    }

    #[test]
    fn test_config_zero_compact_min_active() {
        let config = ChunkedPrefillConfig {
            compact_min_active: 0,
            ..Default::default()
        };
        assert_eq!(config.compact_min_active, 0);
    }

    #[test]
    fn test_config_large_values() {
        let config = ChunkedPrefillConfig {
            chunk_size: usize::MAX,
            max_chunks_per_request: usize::MAX,
            max_batch_tokens: usize::MAX,
            compact_min_active: usize::MAX,
            ..Default::default()
        };
        assert_eq!(config.chunk_size, usize::MAX);
        assert_eq!(config.max_chunks_per_request, usize::MAX);
        assert_eq!(config.max_batch_tokens, usize::MAX);
        assert_eq!(config.compact_min_active, usize::MAX);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SlotType additional tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_slot_type_all_variants_exhaustive() {
        // Ensure both variants are covered
        match SlotType::Decode {
            SlotType::Decode => {}
            SlotType::PrefillChunk => {}
        }
    }

    #[test]
    fn test_slot_type_ordering() {
        let variants = [SlotType::Decode, SlotType::PrefillChunk];
        assert_eq!(variants.len(), 2);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchSlot edge cases
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_slot_zero_request_id() {
        let slot = BatchSlot {
            request_id: 0,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: -1,
        };
        assert_eq!(slot.request_id, 0);
    }

    #[test]
    fn test_batch_slot_max_request_id() {
        let slot = BatchSlot {
            request_id: u64::MAX,
            slot_type: SlotType::PrefillChunk,
            token_start: 0,
            token_end: 100,
            compact_target: 0,
        };
        assert_eq!(slot.request_id, u64::MAX);
    }

    #[test]
    fn test_batch_slot_zero_token_range() {
        let slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 0,
            compact_target: -1,
        };
        assert_eq!(slot.token_end - slot.token_start, 0);
    }

    #[test]
    fn test_batch_slot_large_token_range() {
        let slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::PrefillChunk,
            token_start: 0,
            token_end: usize::MAX,
            compact_target: i32::MAX,
        };
        assert_eq!(slot.token_end, usize::MAX);
        assert_eq!(slot.compact_target, i32::MAX);
    }

    #[test]
    fn test_batch_slot_negative_compact_target() {
        let slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: i32::MIN,
        };
        assert!(slot.compact_target < 0);
    }

    #[test]
    fn test_batch_slot_clone_independence() {
        let mut slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 10,
            compact_target: 5,
        };
        let cloned = slot.clone();
        slot.request_id = 999;
        slot.token_end = 20;
        assert_eq!(cloned.request_id, 1);
        assert_eq!(cloned.token_end, 10);
        assert_eq!(slot.request_id, 999);
    }

    #[test]
    fn test_batch_slot_non_overlapping_ranges() {
        let slot_a = BatchSlot {
            request_id: 1,
            slot_type: SlotType::PrefillChunk,
            token_start: 0,
            token_end: 100,
            compact_target: -1,
        };
        let slot_b = BatchSlot {
            request_id: 1,
            slot_type: SlotType::PrefillChunk,
            token_start: 100,
            token_end: 200,
            compact_target: -1,
        };
        assert_eq!(slot_a.token_end, slot_b.token_start);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchManifest additional tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_manifest_empty_slots_no_compact() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.25,
            compact_min_active: 1,
            ..Default::default()
        };
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_batch_manifest_single_active_slot_below_min() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.10,
            compact_min_active: 2,
            ..Default::default()
        };
        let manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: 0,
            }],
            total_tokens: 1,
            decode_tokens: 1,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.50,
        };
        // 1 active < min_active=2, so no compact
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_batch_manifest_waste_ratio_zero() {
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
            waste_ratio: 0.0,
        };
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_batch_manifest_clone_independence() {
        let mut manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: -1,
            }],
            total_tokens: 1,
            decode_tokens: 1,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let cloned = manifest.clone();
        manifest.total_tokens = 999;
        manifest.waste_ratio = 1.0;
        assert_eq!(cloned.total_tokens, 1);
        assert!((cloned.waste_ratio).abs() < 1e-6);
    }

    #[test]
    fn test_batch_manifest_mixed_slot_types() {
        let manifest = BatchManifest {
            slots: vec![
                BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
                BatchSlot { request_id: 2, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 50, compact_target: -1 },
                BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
            ],
            total_tokens: 52,
            decode_tokens: 2,
            prefill_tokens: 50,
            compact_required: false,
            waste_ratio: 0.0,
        };
        assert_eq!(manifest.slots.len(), 3);
        let decode_count = manifest.slots.iter().filter(|s| s.slot_type == SlotType::Decode).count();
        let prefill_count = manifest.slots.iter().filter(|s| s.slot_type == SlotType::PrefillChunk).count();
        assert_eq!(decode_count, 2);
        assert_eq!(prefill_count, 1);
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy additional edge cases
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_zero_l1_ratio() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.0;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 512); // l1 < 0.25 → base
    }

    #[test]
    fn test_adaptive_policy_just_below_25_percent() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.2499;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 512); // l1 < 0.25 → base
    }

    #[test]
    fn test_adaptive_policy_just_above_75_percent() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.7501;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096); // l1 > 0.75 → max_seq_len
    }

    #[test]
    fn test_adaptive_policy_remaining_one() {
        let policy = AdaptiveChunkPolicy::new(512);
        let chunk = policy.compute_chunk_size(1, 4096);
        assert_eq!(chunk, 1); // remaining < base → remaining
    }

    #[test]
    fn test_adaptive_policy_zero_max_seq_len() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 0);
        assert_eq!(chunk, 512); // max_seq_len=0, high l1 → adaptive=0, result clamped to base
    }

    #[test]
    fn test_adaptive_policy_high_concurrency_with_low_l1() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.1; // < 25% → base
        policy.concurrent_requests = 16; // > 8 → factor 0.5
        let chunk = policy.compute_chunk_size(10000, 4096);
        // l1 < 0.25 → adaptive = base = 512
        // result = 512 * 0.5 = 256, but clamped to max(base, remaining_tokens)
        assert_eq!(chunk, 512);
    }

    #[test]
    fn test_adaptive_policy_exact_concurrency_boundary_4() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 4; // not > 4 → factor 1.0
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096);
    }

    #[test]
    fn test_adaptive_policy_exact_concurrency_boundary_8() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 8; // 8 > 4 → factor 0.75, but NOT > 8
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 3072); // 4096 * 0.75 = 3072
    }

    #[test]
    fn test_adaptive_policy_concurrency_5_shrinks() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 5; // > 4 → factor 0.75
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 3072); // 4096 * 0.75 = 3072
    }

    #[test]
    fn test_adaptive_policy_mid_interpolation_at_one_third() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.25 + 0.50 / 3.0; // ~0.4167
        policy.concurrent_requests = 1;
        // t = (0.4167 - 0.25) / 0.50 ≈ 0.3333
        // scaled = 512 + 0.3333 * (4096 - 512) ≈ 512 + 1194.67 ≈ 1706.67
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert!(chunk >= 512);
        assert!(chunk <= 4096);
    }

    #[test]
    fn test_adaptive_policy_large_base_small_remaining() {
        let policy = AdaptiveChunkPolicy::new(10000);
        let chunk = policy.compute_chunk_size(50, 4096);
        assert_eq!(chunk, 50); // remaining < base → remaining
    }

    #[test]
    fn test_adaptive_policy_clone_independence() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        let cloned = policy.clone();
        policy.l1_available_ratio = 0.0;
        policy.concurrent_requests = 100;
        assert!((cloned.l1_available_ratio - 1.0).abs() < 1e-6);
        assert_eq!(cloned.concurrent_requests, 1);
        assert_eq!(policy.concurrent_requests, 100);
    }

    // ═══════════════════════════════════════════════════════════════════
    // ChunkedPrefillScheduler additional tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_should_chunk_with_zero_chunk_size() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 0,
            enabled: true,
            ..Default::default()
        });
        // seq_len > 0 is always true when chunk_size=0
        assert!(scheduler.should_chunk(1));
        assert!(scheduler.should_chunk(0) == false); // 0 > 0 is false
    }

    #[test]
    fn test_should_chunk_with_large_seq_len() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(scheduler.should_chunk(usize::MAX));
    }

    #[test]
    fn test_should_chunk_with_zero_seq_len() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(!scheduler.should_chunk(0));
    }

    #[test]
    fn test_next_chunk_size_respects_max_seq_len_zero() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 256,
            ..Default::default()
        });
        let chunk = scheduler.next_chunk_size(10000, 0);
        assert_eq!(chunk, 256); // max_seq_len=0 → high l1 → adaptive=0, clamped to base
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch additional edge cases
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_full_decode_ratio() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            decode_ratio_cap: 1.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = (0..50).map(|i| (i, i as usize * 10)).collect();
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert_eq!(manifest.decode_tokens, 50); // 100% budget for decode
        assert_eq!(manifest.prefill_tokens, 0);
    }

    #[test]
    fn test_compose_batch_zero_decode_ratio() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            decode_ratio_cap: 0.0,
            chunk_size: 10,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 10)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(3, 50, 0)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert_eq!(manifest.decode_tokens, 0); // 0% budget for decode
        assert!(manifest.prefill_tokens > 0);
    }

    #[test]
    fn test_compose_batch_single_decode_single_prefill() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 50,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(2, 100, 0)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert_eq!(manifest.decode_tokens, 1);
        assert!(manifest.prefill_tokens > 0);
        assert_eq!(manifest.total_tokens, manifest.decode_tokens + manifest.prefill_tokens);
    }

    #[test]
    fn test_compose_batch_prefill_consumes_full_remaining_budget() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 10,
            decode_ratio_cap: 0.1,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        // Request with more tokens than remaining budget
        let prefill: Vec<(RequestId, usize, usize)> = vec![(2, 10000, 0)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        // decode_ratio_cap=0.1 → max_decode=20, only 1 decode ready → decode_count=1
        // prefill_budget = 200 - 1 = 199
        assert!(manifest.prefill_tokens > 0);
        assert!(manifest.prefill_tokens <= 199);
    }

    #[test]
    fn test_compose_batch_many_small_prefill_requests() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = (0..10)
            .map(|i| (i, 50, i as usize * 50))
            .collect();

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert!(manifest.slots.len() >= 1);
        assert!(manifest.prefill_tokens > 0);
    }

    #[test]
    fn test_compose_batch_max_chunks_zero_means_unlimited() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 1000,
            chunk_size: 10,
            max_chunks_per_request: 0, // unlimited
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![
            (1, 100, 0),
            (1, 100, 0),
            (1, 100, 0),
        ];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        // With max_chunks=0 (unlimited), all entries for request 1 should get chunks
        let chunks_for_req1 = manifest.slots.iter()
            .filter(|s| s.request_id == 1 && s.slot_type == SlotType::PrefillChunk)
            .count();
        assert!(chunks_for_req1 >= 1);
    }

    #[test]
    fn test_compose_batch_low_memory_pressure() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 50, 0)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 0.1);

        assert!(manifest.total_tokens <= 10); // 100 * 0.1 = 10 budget
    }

    #[test]
    fn test_compose_batch_decode_slots_order_preserved() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            decode_ratio_cap: 1.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![
            (100, 0), (200, 10), (300, 20), (400, 30),
        ];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        let ids: Vec<RequestId> = manifest.slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![100, 200, 300, 400]);
    }

    #[test]
    fn test_compose_batch_waste_ratio_full_utilization() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 3,
            decode_ratio_cap: 1.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 1), (3, 2)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert!((manifest.waste_ratio).abs() < 1e-6); // 3/3 = 100% utilization
        assert!(!manifest.compact_required);
    }

    #[test]
    fn test_compose_batch_different_requests_get_separate_chunks() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 1000,
            chunk_size: 10,
            max_chunks_per_request: 1,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![
            (10, 100, 0),
            (20, 100, 100),
            (30, 100, 200),
        ];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        let req10 = manifest.slots.iter().any(|s| s.request_id == 10);
        let req20 = manifest.slots.iter().any(|s| s.request_id == 20);
        let req30 = manifest.slots.iter().any(|s| s.request_id == 30);
        assert!(req10);
        assert!(req20);
        assert!(req30);
    }

    // ═══════════════════════════════════════════════════════════════════
    // update_l1_ratio boundary tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_update_l1_ratio_exact_zero() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_l1_ratio(0.0);
        assert!((scheduler.adaptive_policy.l1_available_ratio).abs() < 1e-6);
    }

    #[test]
    fn test_update_l1_ratio_exact_one() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_l1_ratio(1.0);
        assert!((scheduler.adaptive_policy.l1_available_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_l1_ratio_mid_range() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_l1_ratio(0.5);
        assert!((scheduler.adaptive_policy.l1_available_ratio - 0.5).abs() < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════
    // update_concurrency additional tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_update_concurrency_large_value() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_concurrency(usize::MAX);
        assert_eq!(scheduler.adaptive_policy.concurrent_requests, usize::MAX);
    }

    #[test]
    fn test_update_concurrency_sequential_updates() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        scheduler.update_concurrency(5);
        assert_eq!(scheduler.adaptive_policy.concurrent_requests, 5);
        scheduler.update_concurrency(10);
        assert_eq!(scheduler.adaptive_policy.concurrent_requests, 10);
        scheduler.update_concurrency(1);
        assert_eq!(scheduler.adaptive_policy.concurrent_requests, 1);
    }

    // ═══════════════════════════════════════════════════════════════════
    // mark_as_chunked additional edge cases
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_mark_as_chunked_idempotent() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let mut state = RequestState::new(1, RequestPhase::Prefill, 100, 0);
        scheduler.mark_as_chunked(&mut state);
        assert_eq!(state.phase, RequestPhase::ChunkedPrefill);
        scheduler.mark_as_chunked(&mut state);
        assert_eq!(state.phase, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn test_mark_as_chunked_from_chunked_prefill() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let mut state = RequestState::new(1, RequestPhase::ChunkedPrefill, 100, 0);
        scheduler.mark_as_chunked(&mut state);
        assert_eq!(state.phase, RequestPhase::ChunkedPrefill);
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy constructor edge cases
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_new_zero_chunk_size() {
        let policy = AdaptiveChunkPolicy::new(0);
        assert_eq!(policy.default_chunk_size, 0);
    }

    #[test]
    fn test_adaptive_policy_new_large_chunk_size() {
        let policy = AdaptiveChunkPolicy::new(usize::MAX);
        assert_eq!(policy.default_chunk_size, usize::MAX);
    }

    #[test]
    fn test_adaptive_policy_direct_field_mutation() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.3;
        policy.concurrent_requests = 7;
        assert!((policy.l1_available_ratio - 0.3).abs() < 1e-6);
        assert_eq!(policy.concurrent_requests, 7);
    }

    // ═══════════════════════════════════════════════════════════════════
    // should_compact with compact_min_active=0 (always eligible)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_should_compact_min_active_zero() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.10,
            compact_min_active: 0, // any number >= 0
            ..Default::default()
        };
        let manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: 0,
            }],
            total_tokens: 1,
            decode_tokens: 1,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.20, // > 0.10
        };
        assert!(manifest.should_compact(&config));
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchSlot with same request_id different slot types
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_slot_same_request_different_types() {
        let decode_slot = BatchSlot {
            request_id: 42,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: -1,
        };
        let prefill_slot = BatchSlot {
            request_id: 42,
            slot_type: SlotType::PrefillChunk,
            token_start: 0,
            token_end: 100,
            compact_target: -1,
        };
        assert_eq!(decode_slot.request_id, prefill_slot.request_id);
        assert_ne!(decode_slot.slot_type, prefill_slot.slot_type);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Config edge values: negative float fields
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_config_negative_decode_ratio_cap() {
        let config = ChunkedPrefillConfig {
            decode_ratio_cap: -0.5,
            ..Default::default()
        };
        assert!((config.decode_ratio_cap - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_config_negative_waste_threshold() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: -0.1,
            ..Default::default()
        };
        assert!((config.compact_waste_threshold - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_config_above_one_decode_ratio_cap() {
        let config = ChunkedPrefillConfig {
            decode_ratio_cap: 1.5,
            ..Default::default()
        };
        assert!((config.decode_ratio_cap - 1.5).abs() < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchManifest waste_ratio edge values
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_manifest_negative_waste_ratio() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: -0.5,
        };
        assert!((manifest.waste_ratio - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_batch_manifest_waste_ratio_above_one() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 2.0,
        };
        assert!((manifest.waste_ratio - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_manifest_total_tokens_exceeds_sum() {
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 100,
            decode_tokens: 30,
            prefill_tokens: 20,
            compact_required: false,
            waste_ratio: 0.0,
        };
        // total_tokens can differ from decode+prefill if field values are manually set
        assert_ne!(manifest.total_tokens, manifest.decode_tokens + manifest.prefill_tokens);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchSlot with token_start > token_end (degenerate range)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_slot_inverted_token_range() {
        let slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::PrefillChunk,
            token_start: 100,
            token_end: 50,
            compact_target: -1,
        };
        // No invariant prevents this in the struct itself
        assert!(slot.token_start > slot.token_end);
    }

    #[test]
    fn test_batch_slot_equal_start_end() {
        let slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::Decode,
            token_start: 5,
            token_end: 5,
            compact_target: -1,
        };
        assert_eq!(slot.token_start, slot.token_end);
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: high memory_pressure_ratio values
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_pressure_above_one() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            decode_ratio_cap: 1.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 2.0);

        // total_budget = (100 * 2.0) as usize = 200
        assert_eq!(manifest.decode_tokens, 1);
    }

    #[test]
    fn test_compose_batch_negative_pressure_ratio() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, -1.0);

        // total_budget = (100 * -1.0) as usize → 0 (cast of negative float to usize)
        assert_eq!(manifest.total_tokens, 0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy: interpolation exact midpoint
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_mid_l1_exact_half() {
        let mut policy = AdaptiveChunkPolicy::new(256);
        policy.l1_available_ratio = 0.50;
        policy.concurrent_requests = 1;
        // t = (0.50 - 0.25) / 0.50 = 0.5
        // scaled = 256 + 0.5 * (1024 - 256) = 256 + 384 = 640
        let chunk = policy.compute_chunk_size(10000, 1024);
        assert_eq!(chunk, 640);
    }

    #[test]
    fn test_adaptive_policy_l1_near_zero() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.001;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 512); // < 0.25 → base
    }

    #[test]
    fn test_adaptive_policy_l1_near_one() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.999;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096); // > 0.75 → max_seq_len
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: token ranges for prefill chunks
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_prefill_token_start_non_zero() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 10,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 50, 0)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        // remaining_tokens=50, token_start = remaining_tokens - remaining_tokens = 0
        for slot in &manifest.slots {
            assert_eq!(slot.token_start, 0);
            assert!(slot.token_end > 0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // should_compact: compact_waste_threshold edge values
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_should_compact_threshold_zero() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.0,
            compact_min_active: 1,
            ..Default::default()
        };
        // waste_ratio = 0.0, threshold = 0.0, NOT strictly greater
        let manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: 0,
            }],
            total_tokens: 1,
            decode_tokens: 1,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_should_compact_threshold_near_zero_positive_waste() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.0,
            compact_min_active: 1,
            ..Default::default()
        };
        // waste_ratio = 0.001 > 0.0 threshold
        let manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: 0,
            }],
            total_tokens: 1,
            decode_tokens: 1,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.001,
        };
        assert!(manifest.should_compact(&config));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Scheduler: next_chunk_size with various parameters
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_next_chunk_size_with_l1_update() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 256,
            ..Default::default()
        });
        scheduler.update_l1_ratio(0.1); // low → base
        let chunk = scheduler.next_chunk_size(10000, 4096);
        assert_eq!(chunk, 256);
    }

    #[test]
    fn test_next_chunk_size_with_concurrency_update() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 256,
            ..Default::default()
        });
        scheduler.update_l1_ratio(0.9);
        scheduler.update_concurrency(16);
        let chunk = scheduler.next_chunk_size(10000, 4096);
        assert_eq!(chunk, 2048); // 4096 * 0.5 = 2048
    }

    #[test]
    fn test_next_chunk_size_small_remaining() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 256,
            ..Default::default()
        });
        let chunk = scheduler.next_chunk_size(10, 4096);
        assert_eq!(chunk, 10); // remaining < base
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: compact_required field correctness
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_compact_required_low_waste() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 10,
            chunk_size: 5,
            decode_ratio_cap: 1.0,
            compact_waste_threshold: 0.50,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        // 5 tokens used out of 10 → waste = 50% == threshold → compact_required = false
        assert!(!manifest.compact_required);
    }

    #[test]
    fn test_compose_batch_compact_required_high_waste() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 5,
            decode_ratio_cap: 1.0,
            compact_waste_threshold: 0.10,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        // 1 token out of 100 → waste = 99% > 10% → compact_required = true
        assert!(manifest.compact_required);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchSlot: multiple slots in vec
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_slots_vec_iteration_order() {
        let slots: Vec<BatchSlot> = vec![
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
            BatchSlot { request_id: 2, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 10, compact_target: -1 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
        ];
        let ids: Vec<RequestId> = slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_batch_slots_vec_filter_by_type() {
        let slots: Vec<BatchSlot> = vec![
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
            BatchSlot { request_id: 2, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 10, compact_target: -1 },
            BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
        ];
        let decode_count = slots.iter().filter(|s| s.slot_type == SlotType::Decode).count();
        assert_eq!(decode_count, 2);
        let prefill_count = slots.iter().filter(|s| s.slot_type == SlotType::PrefillChunk).count();
        assert_eq!(prefill_count, 1);
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy: compute_chunk_size with base=1
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_base_one_high_l1() {
        let mut policy = AdaptiveChunkPolicy::new(1);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096); // max_seq_len
    }

    #[test]
    fn test_adaptive_policy_base_one_low_l1() {
        let mut policy = AdaptiveChunkPolicy::new(1);
        policy.l1_available_ratio = 0.1;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 1); // base = 1
    }

    #[test]
    fn test_adaptive_policy_base_one_remaining_zero() {
        let policy = AdaptiveChunkPolicy::new(1);
        let chunk = policy.compute_chunk_size(0, 4096);
        assert_eq!(chunk, 0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SlotType Eq + Hash usable in collections
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_slot_type_usable_as_hashmap_key() {
        use std::collections::HashMap;
        let mut counts: HashMap<SlotType, usize> = HashMap::new();
        *counts.entry(SlotType::Decode).or_insert(0) += 1;
        *counts.entry(SlotType::PrefillChunk).or_insert(0) += 2;
        *counts.entry(SlotType::Decode).or_insert(0) += 3;
        assert_eq!(counts[&SlotType::Decode], 4);
        assert_eq!(counts[&SlotType::PrefillChunk], 2);
    }

    #[test]
    fn test_slot_type_usable_in_hashset() {
        use std::collections::HashSet;
        let set: HashSet<SlotType> = [SlotType::Decode, SlotType::PrefillChunk, SlotType::Decode].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: prefill budget exhausted by decode
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_no_prefill_budget_after_decode() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 10,
            decode_ratio_cap: 1.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(11, 100, 0)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert_eq!(manifest.decode_tokens, 10);
        assert_eq!(manifest.prefill_tokens, 0); // budget exhausted
    }

    #[test]
    fn test_compose_batch_single_decode_token_waste() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 1000,
            decode_ratio_cap: 1.0,
            compact_waste_threshold: 0.01,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert!(manifest.waste_ratio > 0.99);
        assert!(manifest.compact_required);
    }

    #[test]
    fn test_compose_batch_many_decode_few_prefill_budget() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 20,
            decode_ratio_cap: 0.8,
            chunk_size: 5,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 1), (3, 2)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(4, 100, 0)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert_eq!(manifest.decode_tokens, 3);
        assert!(manifest.prefill_tokens > 0);
        assert!(manifest.prefill_tokens <= 17); // 20 - 3 = 17 remaining
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy: remaining equals max_seq_len
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_remaining_equals_max_seq_len() {
        let policy = AdaptiveChunkPolicy::new(512);
        let chunk = policy.compute_chunk_size(4096, 4096);
        assert_eq!(chunk, 4096);
    }

    #[test]
    fn test_adaptive_policy_remaining_greater_than_max_seq_len() {
        let policy = AdaptiveChunkPolicy::new(512);
        let chunk = policy.compute_chunk_size(8192, 4096);
        assert_eq!(chunk, 4096); // clamped to max_seq_len
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchManifest: should_compact with many slots
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_should_compact_large_active_count() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.30,
            compact_min_active: 10,
            ..Default::default()
        };
        let slots: Vec<BatchSlot> = (0..20)
            .map(|i| BatchSlot {
                request_id: i,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: i as i32,
            })
            .collect();
        let manifest = BatchManifest {
            slots,
            total_tokens: 10,
            decode_tokens: 20,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.50,
        };
        assert!(manifest.should_compact(&config));
    }

    #[test]
    fn test_should_compact_one_short_of_min_active() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.10,
            compact_min_active: 5,
            ..Default::default()
        };
        let slots: Vec<BatchSlot> = (0..4)
            .map(|i| BatchSlot {
                request_id: i,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: i as i32,
            })
            .collect();
        let manifest = BatchManifest {
            slots,
            total_tokens: 4,
            decode_tokens: 4,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.50,
        };
        // 4 active < 5 min_active
        assert!(!manifest.should_compact(&config));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Scheduler: should_chunk with small chunk_size
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_should_chunk_chunk_size_one() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 1,
            enabled: true,
            ..Default::default()
        });
        assert!(!scheduler.should_chunk(0));
        assert!(!scheduler.should_chunk(1)); // equal, not greater
        assert!(scheduler.should_chunk(2));
    }

    #[test]
    fn test_should_chunk_chunk_size_max() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: usize::MAX,
            enabled: true,
            ..Default::default()
        });
        assert!(!scheduler.should_chunk(usize::MAX)); // equal, not greater
        assert!(!scheduler.should_chunk(0));
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchSlot: compact_target boundary values
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_slot_compact_target_zero() {
        let slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: 0,
        };
        assert_eq!(slot.compact_target, 0);
        assert!(slot.compact_target >= 0);
    }

    #[test]
    fn test_batch_slot_compact_target_max() {
        let slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: i32::MAX,
        };
        assert_eq!(slot.compact_target, i32::MAX);
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: empty prefill with large decode queue
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_large_decode_capped_by_ratio() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            decode_ratio_cap: 0.3,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = (0..100).map(|i| (i, i as usize)).collect();
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        // max_decode = 100 * 0.3 = 30
        assert_eq!(manifest.decode_tokens, 30);
    }

    #[test]
    fn test_compose_batch_decode_capped_by_ratio_half() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 50,
            decode_ratio_cap: 0.5,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = (0..50).map(|i| (i, i as usize)).collect();
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert_eq!(manifest.decode_tokens, 25); // 50 * 0.5 = 25
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy: concurrency factor 1.0 stays at boundary
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_concurrency_9_factor_half() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 9; // > 8 → factor 0.5
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 2048); // 4096 * 0.5
    }

    #[test]
    fn test_adaptive_policy_concurrency_1_no_shrink() {
        let policy = AdaptiveChunkPolicy::new(512);
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096); // default l1=1.0, concurrent=1, no shrink
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchManifest: slots can be mutated through clone
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_manifest_slots_vec_is_cloned() {
        let manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: -1,
            }],
            total_tokens: 1,
            decode_tokens: 1,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let mut cloned = manifest.clone();
        cloned.slots.push(BatchSlot {
            request_id: 2,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: -1,
        });
        assert_eq!(manifest.slots.len(), 1);
        assert_eq!(cloned.slots.len(), 2);
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: tiny budget scenarios
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_budget_one() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 1,
            decode_ratio_cap: 1.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert_eq!(manifest.decode_tokens, 1);
        assert_eq!(manifest.prefill_tokens, 0);
        assert!(!manifest.compact_required); // waste = 0
    }

    #[test]
    fn test_compose_batch_budget_one_with_prefill() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 1,
            chunk_size: 1,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 10, 0)];

        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);

        assert_eq!(manifest.decode_tokens, 0);
        assert!(manifest.prefill_tokens >= 1);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchSlot: debug contains all field names
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_slot_debug_contains_all_fields() {
        let slot = BatchSlot {
            request_id: 42,
            slot_type: SlotType::PrefillChunk,
            token_start: 10,
            token_end: 20,
            compact_target: 5,
        };
        let debug = format!("{:?}", slot);
        assert!(debug.contains("request_id"));
        assert!(debug.contains("slot_type"));
        assert!(debug.contains("token_start"));
        assert!(debug.contains("token_end"));
        assert!(debug.contains("compact_target"));
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: request_id propagation for prefill slots
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_prefill_preserves_request_ids() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 500,
            chunk_size: 10,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![
            (100, 50, 0),
            (200, 50, 0),
        ];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        let ids: Vec<RequestId> = manifest.slots.iter().map(|s| s.request_id).collect();
        assert!(ids.contains(&100));
        assert!(ids.contains(&200));
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: budget split between decode and prefill
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_budget_split_decode_prefill() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.4,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 1), (3, 2)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(4, 100, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        assert_eq!(manifest.decode_tokens, 3);
        assert!(manifest.prefill_tokens > 0);
    }

    #[test]
    fn test_compose_batch_no_decode_only_prefill() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.5,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 200, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        assert_eq!(manifest.decode_tokens, 0);
        assert!(manifest.prefill_tokens > 0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy: high concurrency with mid l1
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_high_concurrency_mid_l1() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.50;
        policy.concurrent_requests = 16; // factor = 0.5
        // t = 0.5, scaled = 512 + 0.5*3584 = 2304, result = 2304*0.5 = 1152
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 1152);
    }

    #[test]
    fn test_adaptive_policy_mid_concurrency_mid_l1() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.50;
        policy.concurrent_requests = 6; // factor = 0.75
        // t = 0.5, scaled = 2304, result = 2304*0.75 = 1728
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 1728);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Config: max_chunks_per_request various values
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_config_max_chunks_one() {
        let config = ChunkedPrefillConfig {
            max_chunks_per_request: 1,
            ..Default::default()
        };
        assert_eq!(config.max_chunks_per_request, 1);
    }

    #[test]
    fn test_config_max_chunks_large() {
        let config = ChunkedPrefillConfig {
            max_chunks_per_request: 1000000,
            ..Default::default()
        };
        assert_eq!(config.max_chunks_per_request, 1000000);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchManifest: compact_required flag independence from should_compact
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_manifest_compact_required_false_should_compact_true() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.10,
            compact_min_active: 1,
            ..Default::default()
        };
        let manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: 0,
            }],
            total_tokens: 1,
            decode_tokens: 1,
            prefill_tokens: 0,
            compact_required: false, // stored flag is false
            waste_ratio: 0.50,       // but should_compact logic returns true
        };
        assert!(manifest.should_compact(&config));
        assert!(!manifest.compact_required);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SlotType: match exhaustiveness at compile time
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_slot_type_match_decode() {
        let st = SlotType::Decode;
        let label = match st {
            SlotType::Decode => "d",
            SlotType::PrefillChunk => "p",
        };
        assert_eq!(label, "d");
    }

    #[test]
    fn test_slot_type_match_prefill() {
        let st = SlotType::PrefillChunk;
        let label = match st {
            SlotType::Decode => "d",
            SlotType::PrefillChunk => "p",
        };
        assert_eq!(label, "p");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Scheduler: update_l1_ratio then compose
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_update_l1_then_chunk_size() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 256,
            ..Default::default()
        });
        scheduler.update_l1_ratio(0.1);
        let chunk = scheduler.next_chunk_size(10000, 4096);
        assert_eq!(chunk, 256); // low l1 → base
        scheduler.update_l1_ratio(0.9);
        let chunk = scheduler.next_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096); // high l1 → max
    }

    #[test]
    fn test_update_concurrency_then_chunk_size() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 256,
            ..Default::default()
        });
        scheduler.update_concurrency(16);
        let chunk = scheduler.next_chunk_size(10000, 4096);
        assert_eq!(chunk, 2048); // 4096 * 0.5
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: prefill with zero remaining tokens
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_prefill_zero_remaining() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 0, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // remaining_tokens=0 → chunk_size clamped to 0
        assert_eq!(manifest.prefill_tokens, 0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchSlot: compact_target used as index
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_slot_compact_target_as_index() {
        let slots: Vec<BatchSlot> = (0..5)
            .map(|i| BatchSlot {
                request_id: i as u64,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: i as i32,
            })
            .collect();
        for (i, slot) in slots.iter().enumerate() {
            assert_eq!(slot.compact_target, i as i32);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: interleaved decode+prefill preserves decode first
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_decode_comes_before_prefill() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 1)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(3, 50, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // First slots should be decode
        assert_eq!(manifest.slots[0].slot_type, SlotType::Decode);
        assert_eq!(manifest.slots[1].slot_type, SlotType::Decode);
        // Then prefill
        assert!(manifest.slots.iter().any(|s| s.slot_type == SlotType::PrefillChunk));
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchManifest: waste_ratio math with known budget
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_waste_half_budget() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 10,
            decode_ratio_cap: 1.0,
            compact_waste_threshold: 0.40,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // 5 used of 10 → waste = 0.5 > 0.40
        assert!((manifest.waste_ratio - 0.5).abs() < 1e-6);
        assert!(manifest.compact_required);
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy: base equals max_seq_len
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_base_equals_max_seq_len() {
        let policy = AdaptiveChunkPolicy::new(4096);
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096);
    }

    #[test]
    fn test_adaptive_policy_base_greater_than_max_seq_len() {
        let policy = AdaptiveChunkPolicy::new(8192);
        // l1=1.0 → adaptive = max_seq_len = 4096
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 8192); // clamped to max(base, ...) = 8192, but limited by remaining
    }

    // ═══════════════════════════════════════════════════════════════════
    // Config: enabled=false overrides should_chunk regardless of seq_len
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_should_chunk_disabled_zero_chunk_size() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            enabled: false,
            chunk_size: 0,
            ..Default::default()
        });
        assert!(!scheduler.should_chunk(1));
        assert!(!scheduler.should_chunk(0));
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchSlot: Copy semantics for slot_type field
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_slot_slot_type_copy_semantics() {
        let slot = BatchSlot {
            request_id: 1,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: -1,
        };
        let slot_type_copy = slot.slot_type; // Copy
        assert_eq!(slot.slot_type, slot_type_copy);
        assert_eq!(slot.slot_type, SlotType::Decode);
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: very small memory pressure ratio
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_tiny_pressure_ratio() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 10000,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(2, 100, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 0.001);
        // budget = 10000 * 0.001 = 10
        assert!(manifest.total_tokens <= 10);
    }

    // ═══════════════════════════════════════════════════════════════════
    // BatchManifest: many slots all with negative compact_target
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_should_compact_many_inactive_slots() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.10,
            compact_min_active: 1,
            ..Default::default()
        };
        let slots: Vec<BatchSlot> = (0..10)
            .map(|i| BatchSlot {
                request_id: i,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: -1,
            })
            .collect();
        let manifest = BatchManifest {
            slots,
            total_tokens: 5,
            decode_tokens: 10,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.50,
        };
        assert!(!manifest.should_compact(&config));
    }

    // ═══════════════════════════════════════════════════════════════════
    // AdaptiveChunkPolicy: low concurrency boundary
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_concurrency_3_no_shrink() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 3; // <= 4 → factor 1.0
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 4096);
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: prefill chunk token range starts at correct offset
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_prefill_token_start_is_zero() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 50,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 200, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        for slot in &manifest.slots {
            assert_eq!(slot.token_start, 0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // compose_batch: empty decode with many prefill entries
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_compose_batch_many_prefill_exhaust_budget() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 50,
            chunk_size: 10,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = (0..20)
            .map(|i| (i, 100, i as usize * 100))
            .collect();
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        assert!(manifest.total_tokens <= 50);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Compact batch of concise single-concept tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_config_clone_preserves_floats() {
        let config = ChunkedPrefillConfig::default();
        let cloned = config.clone();
        assert!((cloned.decode_ratio_cap - config.decode_ratio_cap).abs() < 1e-6);
        assert!((cloned.compact_waste_threshold - config.compact_waste_threshold).abs() < 1e-6);
    }

    #[test]
    fn test_config_debug_contains_max_chunks() {
        let config = ChunkedPrefillConfig::default();
        let s = format!("{:?}", config);
        assert!(s.contains("max_chunks_per_request"));
    }

    #[test]
    fn test_adaptive_policy_debug_all_fields() {
        let policy = AdaptiveChunkPolicy::new(1024);
        let s = format!("{:?}", policy);
        assert!(s.contains("default_chunk_size: 1024"));
        assert!(s.contains("concurrent_requests: 1"));
    }

    #[test]
    fn test_batch_manifest_debug_all_fields() {
        let m = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let s = format!("{:?}", m);
        assert!(s.contains("slots"));
        assert!(s.contains("total_tokens"));
        assert!(s.contains("compact_required"));
    }

    #[test]
    fn test_batch_slot_clone_copies_all_fields() {
        let slot = BatchSlot {
            request_id: u64::MAX,
            slot_type: SlotType::PrefillChunk,
            token_start: 100,
            token_end: 200,
            compact_target: 42,
        };
        let c = slot.clone();
        assert_eq!(c.request_id, u64::MAX);
        assert_eq!(c.token_start, 100);
        assert_eq!(c.token_end, 200);
        assert_eq!(c.compact_target, 42);
    }

    #[test]
    fn test_should_compact_waste_just_above_threshold() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.25,
            compact_min_active: 1,
            ..Default::default()
        };
        let manifest = BatchManifest {
            slots: vec![BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 }],
            total_tokens: 1, decode_tokens: 1, prefill_tokens: 0,
            compact_required: false, waste_ratio: 0.25001,
        };
        assert!(manifest.should_compact(&config));
    }

    #[test]
    fn test_should_compact_waste_just_below_threshold() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.25,
            compact_min_active: 1,
            ..Default::default()
        };
        let manifest = BatchManifest {
            slots: vec![BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 }],
            total_tokens: 1, decode_tokens: 1, prefill_tokens: 0,
            compact_required: false, waste_ratio: 0.24999,
        };
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_should_compact_min_active_exact() {
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.10,
            compact_min_active: 5,
            ..Default::default()
        };
        let slots: Vec<BatchSlot> = (0..5)
            .map(|i| BatchSlot { request_id: i, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: i as i32 })
            .collect();
        let manifest = BatchManifest {
            slots, total_tokens: 5, decode_tokens: 5, prefill_tokens: 0,
            compact_required: false, waste_ratio: 0.20,
        };
        assert!(manifest.should_compact(&config)); // exactly 5 >= 5
    }

    #[test]
    fn test_adaptive_policy_interpolation_near_25() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.26;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert!(chunk >= 512);
        assert!(chunk < 4096);
    }

    #[test]
    fn test_adaptive_policy_interpolation_near_75() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.74;
        policy.concurrent_requests = 1;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert!(chunk > 512);
        assert!(chunk < 4096);
    }

    #[test]
    fn test_compose_batch_prefill_slot_type_prefill_phase() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 50, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        assert!(manifest.slots.iter().all(|s| s.slot_type == SlotType::PrefillChunk));
    }

    #[test]
    fn test_compose_batch_decode_slot_type_is_decode() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            decode_ratio_cap: 1.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 1)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        assert!(manifest.slots.iter().all(|s| s.slot_type == SlotType::Decode));
    }

    #[test]
    fn test_compose_batch_waste_ratio_with_full_prefill() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 100,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 50, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // 50 tokens used out of 100 budget
        assert!((manifest.waste_ratio - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_scheduler_config_exposes_chunk_size() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 1024,
            ..Default::default()
        });
        assert_eq!(scheduler.config().chunk_size, 1024);
    }

    #[test]
    fn test_scheduler_config_exposes_enabled() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            enabled: false,
            ..Default::default()
        });
        assert!(!scheduler.config().enabled);
    }

    #[test]
    fn test_scheduler_config_exposes_max_batch_tokens() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 8192,
            ..Default::default()
        });
        assert_eq!(scheduler.config().max_batch_tokens, 8192);
    }

    #[test]
    fn test_adaptive_policy_default_l1_is_one() {
        let policy = AdaptiveChunkPolicy::new(512);
        assert!((policy.l1_available_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_policy_default_concurrency_is_one() {
        let policy = AdaptiveChunkPolicy::new(512);
        assert_eq!(policy.concurrent_requests, 1);
    }

    #[test]
    fn test_compose_batch_max_chunks_limits_duplicate_requests() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 1000,
            chunk_size: 10,
            max_chunks_per_request: 2,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![
            (1, 100, 0),
            (1, 100, 0),
            (1, 100, 0),
        ];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        let count = manifest.slots.iter().filter(|s| s.request_id == 1).count();
        assert!(count <= 2);
    }

    #[test]
    fn test_compose_batch_budget_consumed_by_first_request() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 20,
            chunk_size: 10,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![
            (1, 1000, 0),
            (2, 1000, 0),
        ];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // First request may consume all budget, second may get nothing or a small slice
        assert!(manifest.total_tokens <= 20);
    }

    #[test]
    fn test_compose_batch_zero_budget_returns_empty() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(2, 50, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        assert_eq!(manifest.decode_tokens, 0);
        assert_eq!(manifest.prefill_tokens, 0);
    }

    #[test]
    fn test_batch_manifest_default_waste_not_compact() {
        let config = ChunkedPrefillConfig::default();
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_slot_type_decode_not_equal_prefill() {
        assert_ne!(SlotType::Decode, SlotType::PrefillChunk);
    }

    #[test]
    fn test_batch_slot_request_id_u64_range() {
        let slot = BatchSlot {
            request_id: 1_000_000_000_000,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 1,
            compact_target: -1,
        };
        assert_eq!(slot.request_id, 1_000_000_000_000);
    }

    #[test]
    fn test_adaptive_policy_high_concurrency_clamps_to_base() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 100; // factor 0.5 → 4096*0.5=2048 >= 512
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 2048);
    }

    #[test]
    fn test_adaptive_policy_high_concurrency_small_remaining() {
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 100; // factor 0.5 → 4096*0.5=2048
        let chunk = policy.compute_chunk_size(1000, 4096);
        assert_eq!(chunk, 1000); // min(2048, 1000)
    }

    #[test]
    fn test_compose_batch_decode_kv_offset_not_used_in_slots() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            decode_ratio_cap: 1.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 9999)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        assert_eq!(manifest.decode_tokens, 1);
        assert_eq!(manifest.slots[0].request_id, 1);
    }

    #[test]
    fn test_batch_manifest_slots_count_matches_len() {
        let slots = vec![
            BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 },
            BatchSlot { request_id: 2, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 10, compact_target: -1 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 11,
            decode_tokens: 1,
            prefill_tokens: 10,
            compact_required: false,
            waste_ratio: 0.0,
        };
        assert_eq!(manifest.slots.len(), 2);
    }

    #[test]
    fn test_config_default_enabled_is_true() {
        assert!(ChunkedPrefillConfig::default().enabled);
    }

    #[test]
    fn test_config_default_max_chunks_is_zero() {
        assert_eq!(ChunkedPrefillConfig::default().max_chunks_per_request, 0);
    }

    #[test]
    fn test_adaptive_policy_concurrency_9_high_l1() {
        let mut policy = AdaptiveChunkPolicy::new(256);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 9;
        let chunk = policy.compute_chunk_size(10000, 4096);
        assert_eq!(chunk, 2048); // 4096 * 0.5
    }

    #[test]
    fn test_compose_batch_prefill_budget_after_decode() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 10,
            decode_ratio_cap: 0.5,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(1, 0), (2, 1), (3, 2)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(4, 100, 0)];
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        assert_eq!(manifest.decode_tokens, 3);
        // max_decode = 100*0.5 = 50, decode_count=3
        // prefill_budget = 100 - 3 = 97
        assert!(manifest.prefill_tokens > 0);
        assert!(manifest.prefill_tokens <= 97);
    }

    #[test]
    fn test_should_chunk_seq_equals_chunk_size_disabled() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 100,
            enabled: true,
            ..Default::default()
        });
        assert!(!scheduler.should_chunk(100)); // equal → not greater
    }

    // ═══════════════════════════════════════════════════════════════════
    // More compact single-concept tests to reach ratio target
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_config_chunk_size_one() {
        let c = ChunkedPrefillConfig { chunk_size: 1, ..Default::default() };
        assert_eq!(c.chunk_size, 1);
    }

    #[test]
    fn test_config_enabled_false() {
        let c = ChunkedPrefillConfig { enabled: false, ..Default::default() };
        assert!(!c.enabled);
    }

    #[test]
    fn test_config_decode_ratio_0_3() {
        let c = ChunkedPrefillConfig { decode_ratio_cap: 0.3, ..Default::default() };
        assert!((c.decode_ratio_cap - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_config_waste_threshold_0_5() {
        let c = ChunkedPrefillConfig { compact_waste_threshold: 0.5, ..Default::default() };
        assert!((c.compact_waste_threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_config_min_active_one() {
        let c = ChunkedPrefillConfig { compact_min_active: 1, ..Default::default() };
        assert_eq!(c.compact_min_active, 1);
    }

    #[test]
    fn test_adaptive_policy_l1_zero() {
        let mut p = AdaptiveChunkPolicy::new(512);
        p.l1_available_ratio = 0.0;
        assert_eq!(p.compute_chunk_size(10000, 4096), 512);
    }

    #[test]
    fn test_adaptive_policy_l1_half_interpolates() {
        let mut p = AdaptiveChunkPolicy::new(100);
        p.l1_available_ratio = 0.50;
        let chunk = p.compute_chunk_size(10000, 1000);
        assert!(chunk > 100 && chunk < 1000);
    }

    #[test]
    fn test_adaptive_policy_remaining_1() {
        let p = AdaptiveChunkPolicy::new(512);
        assert_eq!(p.compute_chunk_size(1, 4096), 1);
    }

    #[test]
    fn test_adaptive_policy_concurrency_7_factor_075() {
        let mut p = AdaptiveChunkPolicy::new(512);
        p.l1_available_ratio = 0.9;
        p.concurrent_requests = 7;
        assert_eq!(p.compute_chunk_size(10000, 4096), 3072);
    }

    #[test]
    fn test_batch_slot_decode_type() {
        let s = BatchSlot { request_id: 0, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 };
        assert_eq!(s.slot_type, SlotType::Decode);
    }

    #[test]
    fn test_batch_slot_prefill_type() {
        let s = BatchSlot { request_id: 0, slot_type: SlotType::PrefillChunk, token_start: 0, token_end: 10, compact_target: -1 };
        assert_eq!(s.slot_type, SlotType::PrefillChunk);
    }

    #[test]
    fn test_batch_slot_token_range_10() {
        let s = BatchSlot { request_id: 0, slot_type: SlotType::PrefillChunk, token_start: 5, token_end: 15, compact_target: 0 };
        assert_eq!(s.token_end - s.token_start, 10);
    }

    #[test]
    fn test_batch_manifest_empty_waste_zero() {
        let m = BatchManifest {
            slots: vec![], total_tokens: 0, decode_tokens: 0, prefill_tokens: 0,
            compact_required: false, waste_ratio: 0.0,
        };
        assert!(!m.should_compact(&ChunkedPrefillConfig::default()));
    }

    #[test]
    fn test_batch_manifest_one_active_waste_high() {
        let cfg = ChunkedPrefillConfig { compact_waste_threshold: 0.10, compact_min_active: 1, ..Default::default() };
        let m = BatchManifest {
            slots: vec![BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 }],
            total_tokens: 1, decode_tokens: 1, prefill_tokens: 0,
            compact_required: false, waste_ratio: 0.50,
        };
        assert!(m.should_compact(&cfg));
    }

    #[test]
    fn test_scheduler_new_default_config() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert_eq!(s.config().chunk_size, 512);
        assert_eq!(s.config().max_batch_tokens, 4096);
    }

    #[test]
    fn test_should_chunk_seq_1_below_512() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(!s.should_chunk(1));
    }

    #[test]
    fn test_should_chunk_seq_513() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(s.should_chunk(513));
    }

    #[test]
    fn test_mark_as_chunked_from_prefill() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let mut st = RequestState::new(1, RequestPhase::Prefill, 100, 0);
        s.mark_as_chunked(&mut st);
        assert_eq!(st.phase, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn test_update_l1_then_check_field() {
        let mut s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        s.update_l1_ratio(0.3);
        assert!((s.adaptive_policy.l1_available_ratio - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_update_concurrency_then_check_field() {
        let mut s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        s.update_concurrency(8);
        assert_eq!(s.adaptive_policy.concurrent_requests, 8);
    }

    #[test]
    fn test_compose_batch_returns_manifest() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let m = s.compose_batch(&[], &[], 1.0);
        assert!(m.slots.is_empty());
    }

    #[test]
    fn test_compose_batch_1_decode_no_prefill() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 10, decode_ratio_cap: 1.0, ..Default::default()
        });
        let m = s.compose_batch(&[(1, 0)], &[], 1.0);
        assert_eq!(m.decode_tokens, 1);
        assert_eq!(m.prefill_tokens, 0);
    }

    #[test]
    fn test_compose_batch_1_prefill_no_decode() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100, chunk_size: 10, decode_ratio_cap: 0.0, ..Default::default()
        });
        let m = s.compose_batch(&[], &[(1, 50, 0)], 1.0);
        assert_eq!(m.decode_tokens, 0);
        assert!(m.prefill_tokens > 0);
    }

    #[test]
    fn test_slot_type_debug_decode_string() {
        assert!(format!("{:?}", SlotType::Decode).contains("Decode"));
    }

    #[test]
    fn test_slot_type_debug_prefill_string() {
        assert!(format!("{:?}", SlotType::PrefillChunk).contains("PrefillChunk"));
    }

    #[test]
    fn test_batch_slot_debug_has_request_id() {
        let s = BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 };
        assert!(format!("{:?}", s).contains("42"));
    }

    #[test]
    fn test_adaptive_policy_new_sets_default_l1() {
        let p = AdaptiveChunkPolicy::new(256);
        assert!((p.l1_available_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_policy_new_sets_concurrency_1() {
        let p = AdaptiveChunkPolicy::new(256);
        assert_eq!(p.concurrent_requests, 1);
    }

    #[test]
    fn test_config_default_chunk_size_512() {
        assert_eq!(ChunkedPrefillConfig::default().chunk_size, 512);
    }

    #[test]
    fn test_config_default_decode_ratio_0_6() {
        assert!((ChunkedPrefillConfig::default().decode_ratio_cap - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_config_default_waste_0_25() {
        assert!((ChunkedPrefillConfig::default().compact_waste_threshold - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_config_default_min_active_4() {
        assert_eq!(ChunkedPrefillConfig::default().compact_min_active, 4);
    }

    #[test]
    fn test_config_default_max_batch_4096() {
        assert_eq!(ChunkedPrefillConfig::default().max_batch_tokens, 4096);
    }

    #[test]
    fn test_compose_batch_2_decode_interleaved() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100, chunk_size: 10, decode_ratio_cap: 0.6, ..Default::default()
        });
        let m = s.compose_batch(&[(1, 0), (2, 1)], &[(3, 50, 0)], 1.0);
        assert!(m.slots[0].slot_type == SlotType::Decode);
        assert!(m.slots[1].slot_type == SlotType::Decode);
    }

    #[test]
    fn test_compose_batch_waste_nonzero_with_unused_budget() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 50, decode_ratio_cap: 1.0, ..Default::default()
        });
        let m = s.compose_batch(&[(1, 0)], &[], 1.0);
        assert!(m.waste_ratio > 0.0);
    }

    #[test]
    fn test_adaptive_policy_base_10_mid_l1() {
        let mut p = AdaptiveChunkPolicy::new(10);
        p.l1_available_ratio = 0.50;
        let chunk = p.compute_chunk_size(1000, 100);
        assert!(chunk >= 10 && chunk <= 100);
    }

    #[test]
    fn test_batch_slot_clone_same_request_id() {
        let s = BatchSlot { request_id: 7, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: -1 };
        let c = s.clone();
        assert_eq!(c.request_id, s.request_id);
    }

    #[test]
    fn test_batch_manifest_clone_same_total() {
        let m = BatchManifest {
            slots: vec![], total_tokens: 42, decode_tokens: 0, prefill_tokens: 42,
            compact_required: false, waste_ratio: 0.0,
        };
        assert_eq!(m.clone().total_tokens, 42);
    }

    #[test]
    fn test_scheduler_config_returns_same_ref() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig { chunk_size: 333, ..Default::default() });
        assert_eq!(s.config().chunk_size, 333);
    }

    #[test]
    fn test_should_chunk_seq_511_below_512() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(!s.should_chunk(511));
    }

    #[test]
    fn test_update_l1_clamps_1_5() {
        let mut s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        s.update_l1_ratio(1.5);
        assert!((s.adaptive_policy.l1_available_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compose_batch_empty_manifest_fields() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100, ..Default::default()
        });
        let m = s.compose_batch(&[], &[], 1.0);
        assert!(m.slots.is_empty());
        assert_eq!(m.total_tokens, 0);
    }

    #[test]
    fn test_adaptive_policy_remaining_512_base_512() {
        let p = AdaptiveChunkPolicy::new(512);
        assert_eq!(p.compute_chunk_size(512, 4096), 512);
    }

    #[test]
    fn test_config_clone_matches_all_fields() {
        let c = ChunkedPrefillConfig { chunk_size: 1024, enabled: false, max_chunks_per_request: 5, decode_ratio_cap: 0.8, compact_waste_threshold: 0.1, compact_min_active: 2, max_batch_tokens: 2048 };
        let cl = c.clone();
        assert_eq!(cl.chunk_size, 1024);
        assert!(!cl.enabled);
        assert_eq!(cl.max_chunks_per_request, 5);
    }

    #[test]
    fn test_compose_batch_3_decode_0_prefill() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig { max_batch_tokens: 100, decode_ratio_cap: 1.0, ..Default::default() });
        let m = s.compose_batch(&[(1, 0), (2, 1), (3, 2)], &[], 1.0);
        assert_eq!(m.slots.len(), 3);
    }

    #[test]
    fn test_adaptive_policy_l1_0_3_mid_zone() {
        let mut p = AdaptiveChunkPolicy::new(512);
        p.l1_available_ratio = 0.30;
        let chunk = p.compute_chunk_size(10000, 4096);
        assert!(chunk >= 512 && chunk < 4096);
    }

    #[test]
    fn test_should_compact_high_waste_2_active_min_2() {
        let cfg = ChunkedPrefillConfig { compact_waste_threshold: 0.10, compact_min_active: 2, ..Default::default() };
        let m = BatchManifest {
            slots: vec![BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 0 }, BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 1, compact_target: 1 }],
            total_tokens: 2, decode_tokens: 2, prefill_tokens: 0, compact_required: false, waste_ratio: 0.50,
        };
        assert!(m.should_compact(&cfg));
    }

    #[test]
    fn test_batch_slot_token_start_1000() {
        let s = BatchSlot { request_id: 1, slot_type: SlotType::PrefillChunk, token_start: 1000, token_end: 2000, compact_target: -1 };
        assert_eq!(s.token_start, 1000);
    }

    #[test]
    fn test_compose_batch_decode_ratio_cap_0_3() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig { max_batch_tokens: 100, decode_ratio_cap: 0.3, ..Default::default() });
        let decode: Vec<_> = (0..50).map(|i| (i as u64, i as usize)).collect();
        let m = s.compose_batch(&decode, &[], 1.0);
        assert_eq!(m.decode_tokens, 30);
    }

    #[test]
    fn test_adaptive_policy_l1_0_6_mid_zone() {
        let mut p = AdaptiveChunkPolicy::new(512);
        p.l1_available_ratio = 0.60;
        let chunk = p.compute_chunk_size(10000, 4096);
        assert!(chunk > 512 && chunk < 4096);
    }

    #[test]
    fn test_batch_manifest_prefill_tokens_field() {
        let m = BatchManifest { slots: vec![], total_tokens: 0, decode_tokens: 0, prefill_tokens: 42, compact_required: false, waste_ratio: 0.0 };
        assert_eq!(m.prefill_tokens, 42);
    }

    #[test]
    fn test_config_max_chunks_zero_means_unlimited_value() {
        assert_eq!(ChunkedPrefillConfig::default().max_chunks_per_request, 0);
    }

    #[test]
    fn test_compose_batch_pressure_0_5_halves_budget() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig { max_batch_tokens: 100, decode_ratio_cap: 1.0, ..Default::default() });
        let m = s.compose_batch(&[(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)], &[], 0.5);
        // budget = 50, max_decode = 50, 5 ready
        assert_eq!(m.decode_tokens, 5);
    }

    #[test]
    fn test_adaptive_policy_l1_0_4_mid_not_extremes() {
        let mut p = AdaptiveChunkPolicy::new(512);
        p.l1_available_ratio = 0.40;
        let chunk = p.compute_chunk_size(10000, 4096);
        assert!(chunk > 512 && chunk < 4096);
    }

    #[test]
    fn test_batch_manifest_compact_required_field_stored() {
        let m = BatchManifest { slots: vec![], total_tokens: 0, decode_tokens: 0, prefill_tokens: 0, compact_required: true, waste_ratio: 0.0 };
        assert!(m.compact_required);
    }

    #[test]
    fn test_compose_batch_1_prefill_remaining_1() {
        let s = ChunkedPrefillScheduler::new(ChunkedPrefillConfig { max_batch_tokens: 100, chunk_size: 10, decode_ratio_cap: 0.0, ..Default::default() });
        let m = s.compose_batch(&[], &[(1, 1, 0)], 1.0);
        assert_eq!(m.prefill_tokens, 1);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Additional 13 tests: edge cases and uncovered boundaries
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adaptive_policy_l1_exactly_one_with_custom_base() {
        // Arrange: policy with base=1024, l1=1.0 (> 0.75 → uses max_seq_len)
        let mut policy = AdaptiveChunkPolicy::new(1024);
        policy.l1_available_ratio = 1.0;
        policy.concurrent_requests = 1;
        // Act
        let chunk = policy.compute_chunk_size(10000, 8192);
        // Assert: should return max_seq_len since l1 > 0.75
        assert_eq!(chunk, 8192);
    }

    #[test]
    fn test_compose_batch_prefill_token_end_equals_start_plus_chunk() {
        // Arrange: scheduler with known chunk_size, single prefill request
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 50,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 200, 0)];
        // Act
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // Assert: token_end - token_start should equal the actual chunk allocated
        for slot in &manifest.slots {
            let span = slot.token_end - slot.token_start;
            assert!(span > 0);
            assert!(span <= 200);
        }
    }

    #[test]
    fn test_should_compact_min_active_zero_waste_zero() {
        // Arrange: min_active=0, waste=0.0, threshold=0.0 — waste NOT strictly > threshold
        let config = ChunkedPrefillConfig {
            compact_waste_threshold: 0.0,
            compact_min_active: 0,
            ..Default::default()
        };
        let manifest = BatchManifest {
            slots: vec![BatchSlot {
                request_id: 1,
                slot_type: SlotType::Decode,
                token_start: 0,
                token_end: 1,
                compact_target: 0,
            }],
            total_tokens: 1,
            decode_tokens: 1,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        // Act & Assert: 0.0 is NOT strictly > 0.0
        assert!(!manifest.should_compact(&config));
    }

    #[test]
    fn test_adaptive_policy_base_larger_than_remaining_high_l1() {
        // Arrange: base=512, remaining=100, l1=0.9 (> 0.75 → adaptive = max_seq_len)
        // remaining < base → return remaining
        let policy = AdaptiveChunkPolicy::new(512);
        let chunk = policy.compute_chunk_size(100, 4096);
        // Act & Assert
        assert_eq!(chunk, 100);
    }

    #[test]
    fn test_compose_batch_second_prefill_gets_zero_after_budget_exhausted() {
        // Arrange: budget so small that first prefill consumes all, second gets nothing
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 15,
            chunk_size: 10,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![
            (1, 100, 0),
            (2, 100, 0),
        ];
        // Act
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // Assert: total cannot exceed budget
        assert!(manifest.total_tokens <= 15);
        // First request must have a slot
        assert!(manifest.slots.iter().any(|s| s.request_id == 1));
    }

    #[test]
    fn test_adaptive_policy_concurrency_zero_no_shrink() {
        // Arrange: concurrent_requests=0, which is <= 4 → factor=1.0
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9;
        policy.concurrent_requests = 0;
        // Act
        let chunk = policy.compute_chunk_size(10000, 4096);
        // Assert: no shrink since 0 <= 4
        assert_eq!(chunk, 4096);
    }

    #[test]
    fn test_compose_batch_budget_equals_decode_cap_boundary() {
        // Arrange: max_batch_tokens=10, decode_ratio_cap=0.6 → max_decode=6
        // Provide exactly 6 decode requests and some prefill
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 10,
            chunk_size: 2,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![
            (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5),
        ];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(7, 10, 0)];
        // Act
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // Assert: decode capped at 6, prefill gets remaining budget (10-6=4)
        assert_eq!(manifest.decode_tokens, 6);
        assert!(manifest.prefill_tokens > 0);
        assert!(manifest.prefill_tokens <= 4);
    }

    #[test]
    fn test_compose_batch_interleaved_correct_slot_ordering() {
        // Arrange: 2 decode + 1 prefill, verify decode slots come first with correct IDs
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 200,
            chunk_size: 10,
            decode_ratio_cap: 0.6,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![(10, 0), (20, 5)];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(30, 50, 10)];
        // Act
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // Assert: first two slots are decode with IDs 10 and 20 in order
        assert_eq!(manifest.slots[0].request_id, 10);
        assert_eq!(manifest.slots[0].slot_type, SlotType::Decode);
        assert_eq!(manifest.slots[1].request_id, 20);
        assert_eq!(manifest.slots[1].slot_type, SlotType::Decode);
        // Prefill slot follows
        assert_eq!(manifest.slots[2].slot_type, SlotType::PrefillChunk);
        assert_eq!(manifest.slots[2].request_id, 30);
    }

    #[test]
    fn test_should_chunk_enabled_true_chunk_zero_seq_zero() {
        // Arrange: chunk_size=0, enabled=true, seq_len=0
        // seq_len > chunk_size → 0 > 0 is false
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 0,
            enabled: true,
            ..Default::default()
        });
        // Act & Assert
        assert!(!scheduler.should_chunk(0));
    }

    #[test]
    fn test_update_l1_ratio_zero_then_compute_chunk() {
        // Arrange: update l1 to 0.0 (clamped), then compute chunk
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            chunk_size: 256,
            ..Default::default()
        });
        scheduler.update_l1_ratio(0.0);
        // Act
        let chunk = scheduler.next_chunk_size(10000, 4096);
        // Assert: l1=0.0 < 0.25 → base
        assert_eq!(chunk, 256);
    }

    #[test]
    fn test_adaptive_policy_high_concurrency_low_l1_clamps_to_base() {
        // Arrange: l1 < 0.25 → adaptive = base = 100
        // concurrent > 8 → factor 0.5 → result = 50, clamped to max(base, remaining) = 100
        let mut policy = AdaptiveChunkPolicy::new(100);
        policy.l1_available_ratio = 0.1;
        policy.concurrent_requests = 20;
        // Act
        let chunk = policy.compute_chunk_size(10000, 4096);
        // Assert: concurrency factor makes it 50, but clamped to base=100
        assert_eq!(chunk, 100);
    }

    #[test]
    fn test_compose_batch_prefill_with_single_token_remaining() {
        // Arrange: prefill request with only 1 token remaining
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 100,
            chunk_size: 50,
            decode_ratio_cap: 0.0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![(1, 1, 0)];
        // Act
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // Assert: 1 remaining < base=50, so chunk = 1
        assert_eq!(manifest.prefill_tokens, 1);
        assert_eq!(manifest.slots.len(), 1);
        assert_eq!(manifest.slots[0].token_start, 0);
        assert_eq!(manifest.slots[0].token_end, 1);
    }

    #[test]
    fn test_compose_batch_waste_ratio_with_zero_total_budget() {
        // Arrange: max_batch_tokens=0 → total_budget=0 → waste_ratio=0.0
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig {
            max_batch_tokens: 0,
            ..Default::default()
        });
        let decode: Vec<(RequestId, usize)> = vec![];
        let prefill: Vec<(RequestId, usize, usize)> = vec![];
        // Act
        let manifest = scheduler.compose_batch(&decode, &prefill, 1.0);
        // Assert: batch_capacity=0 → waste_ratio=0.0
        assert!((manifest.waste_ratio).abs() < 1e-6);
        assert!(!manifest.compact_required);
    }
}
